#!/usr/bin/env python3
"""The compiled decision bundle on the Core AI runtime (Mac), one compute unit per run, against the publisher's model.

    python3 conversion/d1_omni/runtime_check.py <work>/compiled/fp32-L256-h16c --compute cpu_only
    python3 conversion/d1_omni/runtime_check.py <work>/compiled/wfp16-L256-h16c --compute gpu
    python3 conversion/d1_omni/runtime_check.py <work>/compiled/fp16-L256-ane --compute neural_engine

Loads the AOT .aimodelc of <compiled dir> (aot_decide.py; files checked against its manifest), never a JIT .aimodel,
with explicit specialization options: cpu_only(), or from_preferred_compute_unit_kind(gpu / neural_engine). Never
default(): it falls back to the CPU and hides an accelerator's error. coreai.runtime reports no placement, so what
ran where is read from the numbers (and from the ANE regions aot_decide.py counted).

Rows = the source bundle's reference.json: every reference row of the bucket, an audio row with the reference's own
prefix (ref/npz, sha256 checked). One row: host.graph_inputs -> main -> scores -> the scores at P + markers ->
host.probabilities_from_logits -> the publisher's fp32 model (_metrics). Every row runs --repeats times (default 3);
drift = max |d| of the real positions' scores between each repeat and the first run.

Status: the run's bar is _metrics.FP32_BAR for the fp32 bundle on any compute unit (the reference: it settles the
lowering and the unit's fp32 numerics) and SHIP_BAR for wfp16 / fp16 (both verdicts are recorded either way); the
wrong-pairing control must FAIL; cpu_only must be deterministic (drift <= 1e-6). Recorded: per row the marker logits (unrounded), p, max |dp|, max |dlogit|, argmax, drift; aggregates by
source; load seconds; ru_maxrss and the phys footprint before load, after load and at the end; the compile-cache
entries this run created; the other GPU jobs seen while it ran (recorded only: a correctness run may overlap other
lanes' correctness runs). No milliseconds per call are measured.

Output: <work>/results/runtime_<precision>_L<L>_<compute>.json (or --out) and <compiled dir>/provenance/runtime-gate.json
(keyed by compute unit); an existing file or key is never replaced (a .runN name / key is used instead; an existing
--out stops the run).
"""
from __future__ import annotations

import argparse
import asyncio
import ctypes
import datetime
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import gpu_lock, work_path  # noqa: E402

import coreai.runtime as rt  # noqa: E402

import host  # noqa: E402
from _metrics import FP32_BAR, SHIP_BAR, row_record, summarize, wrong_pairing  # noqa: E402

WORK = work_path("_d1_omni")
INPUT_NAMES = ["input_ids", "prefix_embeds", "pad_mask", "prefix_mask", "keep_right", "qtype_onehot"]
CACHE_ROOT = Path.home() / "Library" / "Caches" / "coreai-cache"
DRIFT_BAR_CPU = 1e-6
GPU_JOB_PATTERN = (r"readout_gate|coreai_gate|gate_.*\.py|export_.*\.py|runtime_check\.py|llm-runner|llm-benchmark|"
                   r"dashboard_job\.py|decide-cli|coreai-build")


# --------------------------------------------------------------------------- small helpers (no torch in this process)
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n")


def _command(argv: list[str]) -> str:
    try:
        return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable ({type(error).__name__})"


def environment() -> dict:
    from importlib import metadata

    versions = {}
    for name in ("coreai-core", "numpy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"date": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds"),
            "machine": _command(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "macos_version": _command(["sw_vers", "-productVersion"]),
            "macos_build": _command(["sw_vers", "-buildVersion"]),
            "python": platform.python_version(), "executable": sys.executable, "packages": versions,
            "argv": sys.argv, "pid": os.getpid()}


def free_name(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(2, 100):
        candidate = path.with_name(f"{path.stem}.run{i}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise SystemExit(f"no free name next to {path}")


# --------------------------------------------------------------------------- memory
_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")


def footprint() -> dict:
    """ru_maxrss (bytes on macOS) and proc_pid_rusage(RUSAGE_INFO_V4): resident, phys_footprint, lifetime max."""
    out = {"ru_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    buf = ctypes.create_string_buffer(512)
    if _libproc.proc_pid_rusage(os.getpid(), 4, buf) == 0:
        u64 = lambda offset: int.from_bytes(buf.raw[offset:offset + 8], "little")  # noqa: E731
        out.update(resident_bytes=u64(64), phys_footprint_bytes=u64(72), lifetime_max_phys_footprint_bytes=u64(240))
    return out


# --------------------------------------------------------------------------- other GPU jobs (recorded, never acted on)
def _ps(pid: int, field: str) -> str:
    return subprocess.run(["ps", "-o", f"{field}=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()


def other_gpu_jobs() -> list[str]:
    """PIDs whose argv matches GPU_JOB_PATTERN (pgrep -f, read only), minus this process's ancestors, Claude
    processes and shells (a shell only carries the text in its argv; the job is its child)."""
    ancestors, pid = set(), os.getpid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        parent = _ps(pid, "ppid")
        pid = int(parent) if parent.isdigit() else 1
    found = subprocess.run(["pgrep", "-f", GPU_JOB_PATTERN], capture_output=True, text=True).stdout.split()
    jobs = []
    for pid in (int(p) for p in found if p.isdigit()):
        command = _ps(pid, "comm")
        if (pid in ancestors or not command or "claude" in command.lower()
                or Path(command).name.lstrip("-") in ("zsh", "bash", "sh", "fish", "dash")):
            continue
        jobs.append(f"{pid} {command} {' '.join(_ps(pid, 'args').split())[:200]}")
    return jobs


def lock_state() -> dict:
    """The measurement-window lock as other lanes see it: its text and the processes holding it open (read only)."""
    path = gpu_lock()
    try:
        content = path.read_text().strip()
    except OSError:
        content = None
    holders = subprocess.run(["lsof", "-t", str(path)], capture_output=True, text=True).stdout.split()
    return {"path": str(path), "content": content, "lsof_pids": holders}


class JobWatch:
    """Samples other_gpu_jobs() every few seconds while the run goes on (each new job also goes to the trace)."""

    def __init__(self, every_s: float = 5.0, note=None):
        self.seen, self.samples, self.every_s, self._stop = [], 0, every_s, threading.Event()
        self._note = note or (lambda **event: None)
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self):
        self.samples += 1
        for job in other_gpu_jobs():
            if job not in self.seen:
                self.seen.append(job)
                self._note(event="job", job=job)

    def _loop(self):
        while not self._stop.wait(self.every_s):
            self._sample()

    def __enter__(self):
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._sample()


# --------------------------------------------------------------------------- the run
def options_for(compute: str):
    if compute == "cpu_only":
        return rt.SpecializationOptions.cpu_only()
    kind = rt.ComputeUnitKind.gpu() if compute == "gpu" else rt.ComputeUnitKind.neural_engine()
    return rt.SpecializationOptions.from_preferred_compute_unit_kind(kind)


def cache_entries(build: str) -> dict[str, list[str]]:
    root = CACHE_ROOT / build
    if not root.exists():
        return {}
    return {d.name: sorted(e.name for e in d.iterdir()) for d in sorted(root.iterdir()) if d.is_dir()}


def new_cache_entries(before: dict, after: dict, build: str) -> list[dict]:
    out = []
    for process, names in after.items():
        for name in names:
            if name not in before.get(process, []):
                path = CACHE_ROOT / build / process / name
                out.append({"process_dir": process, "entry": name, "path": str(path),
                            "files": sorted(str(p.relative_to(path)) for p in path.rglob("*") if p.is_file())[:40]})
    return out


async def call(fn, inputs: dict) -> np.ndarray:
    out = await fn({k: rt.NDArray(inputs[k]) for k in INPUT_NAMES})
    return np.array(out["scores"].numpy(), copy=True).reshape(-1)


def load_reference(compiled_dir: Path) -> tuple[dict, dict, Path, dict, list, list]:
    aot = json.loads((compiled_dir / "provenance" / "aot-manifest.json").read_text())
    if aot.get("status") != "COMPILED" or not aot.get("aimodelc_kept"):
        raise SystemExit(f"{compiled_dir}: no compiled bundle ({aot.get('status')}, kept {aot.get('aimodelc_kept')})")
    aimodelc = compiled_dir / aot["aimodelc"]
    for item in aot["files"]:
        file = aimodelc / item["path"]
        if file.stat().st_size != item["bytes"] or sha256_file(file) != item["sha256"]:
            raise SystemExit(f"{file} changed since its compile")
    source_dir = Path(aot["source"]["folder"])
    manifest_path = source_dir / "provenance" / "export-manifest.json"
    if sha256_file(manifest_path) != aot["source"]["export_manifest_sha256"]:
        raise SystemExit(f"{manifest_path} changed since the compile")
    manifest = json.loads(manifest_path.read_text())
    reference_path = source_dir / "reference.json"
    if sha256_file(reference_path) != manifest["reference_json_sha256"]:
        raise SystemExit(f"{reference_path} changed since the export")
    reference = json.loads(reference_path.read_text())
    rows = reference["rows"]
    prefixes, cache = [], {}
    for row in rows:
        npz = row["prefix_npz"]
        if npz is None:
            prefixes.append(None)
            continue
        if npz["file"] not in cache:
            path = WORK / npz["file"]
            if sha256_file(path) != npz["sha256"]:
                raise SystemExit(f"{path} differs from the reference's npz")
            with np.load(path) as z:
                cache[npz["file"]] = np.asarray(z["prefix"], dtype=np.float32).copy()
        prefixes.append(cache[npz["file"]])
    hrows = [host.Row(question=host.as_question(r["question"]), ids=r["ids"], markers=r["markers"],
                      calibrate=r["calibrate"], prefix_len=r["prefix_len"], mode=r["mode"], max_len=0, qid=r["qid"])
             for r in rows]
    # the bucket set the export read each row's native bucket from (round 12's L64 / L128: host.ALL_BUCKETS)
    buckets = tuple(reference.get("reference", {}).get("native_buckets", host.BUCKETS))
    for r, h in zip(rows, hrows):  # a pad / prefix row (round 4) runs padded past its own bucket
        assert h.positions == r["positions"] and h.positions <= reference["seq_len"], r["id"]
        assert host.bucket_for(h.positions, buckets) == r.get("native_bucket", reference["seq_len"]), r["id"]
    return aot, manifest, aimodelc, reference, hrows, prefixes


async def run(compiled_dir: Path, compute: str, repeats: int, note=None) -> dict:
    note = note or (lambda **event: None)
    started = time.perf_counter()
    aot, manifest, aimodelc, reference, hrows, prefixes = load_reference(compiled_dir)
    rows, L, precision = reference["rows"], reference["seq_len"], manifest["precision"]
    build = _command(["sw_vers", "-buildVersion"])
    options = options_for(compute)
    entries_before = cache_entries(build)
    memory = {"before_load": footprint()}
    t0 = time.perf_counter()
    model = await rt.AIModel.load(aimodelc, options)
    load_s = time.perf_counter() - t0
    if list(model.function_names) != ["main"]:
        raise SystemExit(f"functions {model.function_names}")
    fn = model.load_function("main")
    memory["after_load"] = footprint()
    first_inputs, first_markers = host.graph_inputs(hrows[0], L, prefixes[0])
    t0 = time.perf_counter()
    first = await call(fn, first_inputs)
    first_call_s = time.perf_counter() - t0
    memory["after_first_call"] = footprint()
    print(f"loaded {aimodelc.name} ({compute}) in {load_s:.1f} s, first call {first_call_s:.1f} s", flush=True)
    note(event="loaded", load_s=load_s, first_call_s=first_call_s, memory=memory["after_first_call"])

    records = []
    for i, (row, hrow, prefix) in enumerate(zip(rows, hrows, prefixes)):
        inputs, markers = host.graph_inputs(hrow, L, prefix)
        outs = [await call(fn, inputs) for _ in range(repeats)]
        n = row["positions"]
        base = outs[0]
        drift = max((float(np.max(np.abs(o[:n].astype(np.float64) - base[:n]))) for o in outs[1:]), default=0.0)
        drift_markers = max((float(np.max(np.abs(o[markers].astype(np.float64) - base[markers]))) for o in outs[1:]),
                            default=0.0)
        record = row_record(row, base[markers])
        record.update(set=row.get("set", "native"), native_bucket=row.get("native_bucket", L), repeats=repeats,
                      repeat_drift_scores=drift, repeat_drift_markers=drift_markers,
                      scores_finite_real_positions=bool(np.isfinite(base[:n]).all()))
        if i == 0:
            record["first_call_vs_first_repeat_max_abs"] = float(np.max(np.abs(first[:n].astype(np.float64) - base[:n])))
        records.append(record)
        note(event="row", i=i, row=f"{row['id']}/{row['qid']}", max_abs_dp=record["max_abs_dp"],
             max_abs_dlogit=record["max_abs_dlogit"], argmax_equal=record["argmax_equal"], drift=drift)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)} rows", flush=True)
    memory["end"] = footprint()
    note(event="rows_done", rows=len(records), memory=memory["end"])
    assert model is not None  # the model stays alive until every output is copied

    summary = summarize(records)
    summary["by_set"] = {name: {k: s[k] for k in ("rows", "argmax_equal", "non_near_tie", "near_tie", "max_abs_dp",
                                                   "mean_row_max_abs_dp", "max_abs_dlogit")}
                         for name in ("native", "pad", "prefix")
                         if (s := summarize([r for r in records if r["set"] == name])
                             if any(r["set"] == name for r in records) else None)}
    control = wrong_pairing(rows, records)
    drift_max = max(r["repeat_drift_scores"] for r in records)
    bar_key = "fp32_bar" if precision == "fp32" else "ship_bar"
    failures = []
    if summary[bar_key]["status"] != "PASS":
        failures.append(f"{bar_key}: {summary[bar_key]['verdict']}")
    if not control["caught"]:
        failures.append("wrong-pairing control did not FAIL")
    if compute == "cpu_only" and drift_max > DRIFT_BAR_CPU:
        failures.append(f"cpu_only not deterministic (drift {drift_max:.3g})")
    entries_after = cache_entries(build)
    return {
        "status": "FAIL" if failures else "PASS", "failures": failures, "bar_applied": bar_key,
        "stage": "runtime (Mac, Core AI Python runtime, AOT .aimodelc)", "compute": compute,
        "specialization_options": " ".join(str(options).split()), "precision": precision, "seq_len": L,
        "compiled": {"folder": str(compiled_dir), "aimodelc": aimodelc.name, "preferred_compute": aot["preferred_compute"],
                     "architecture": aot["architecture"], "bytes": aot["bytes"], "resources_bin_bytes": aot["resources_bin_bytes"],
                     "ane_regions": aot["ane"]["regions"], "hashes": aot["hashes"]},
        "source_bundle": {"folder": aot["source"]["folder"], "bundle": manifest["bundle"], "bytes": manifest["bytes"]},
        "summary": summary, "wrong_pairing_control": control,
        "repeat_drift": {"repeats_per_row": repeats, "max_abs_scores": drift_max,
                         "max_abs_markers": max(r["repeat_drift_markers"] for r in records),
                         "rows_nonzero": sum(r["repeat_drift_scores"] > 0 for r in records),
                         "cpu_only_bar": DRIFT_BAR_CPU if compute == "cpu_only" else None},
        "load_s": load_s,
        "first_call_s": first_call_s,
        "first_call_note": "one-time cost after load (specialization if the runtime does it there); not a speed figure",
        "memory": memory,
        "cache_entries_created": new_cache_entries(entries_before, entries_after, build),
        "cache_entries_note": "every entry that appeared under the cache root while this run went on (other lanes' "
                              "runs included); this bundle's entry is the one named by its main.hash",
        "own_cache_entry": {"name": aot["hashes"]["main.hash"],
                            "present": aot["hashes"]["main.hash"] in entries_after.get("python", [])},
        "cache_root": str(CACHE_ROOT / build),
        "run_seconds": time.perf_counter() - started,
        "code_sha256": {f: sha256_file(HERE / f) for f in ("runtime_check.py", "_metrics.py", "host.py")},
        "environment": environment(),
        "rows": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("compiled_dir", type=Path, help="<work>/compiled/<precision>-L<L>-h16c|ane")
    parser.add_argument("--compute", choices=["cpu_only", "gpu", "neural_engine"], required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trace", type=Path, help="append one JSON line per event (start, each other GPU job seen, "
                        "load, each row, end), flushed as it happens: what survives a crash")
    parser.add_argument("--out", type=Path, help="the results file (default results/runtime_<precision>_L<L>_<compute>"
                        ".json, or a .runN name); an existing file is never replaced")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("at least 2 repeats (the drift needs a second run)")
    if args.out is not None and args.out.exists():
        parser.error(f"{args.out} exists: never replaced")
    compiled_dir = args.compiled_dir.resolve()
    print(f"pid {os.getpid()}", flush=True)
    trace = open(args.trace, "a", buffering=1) if args.trace else None

    def note(**event):
        if trace is not None:
            stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
            trace.write(json.dumps({"t": stamp, **event}, ensure_ascii=False) + "\n")

    lock_at_start = lock_state()
    note(event="start", pid=os.getpid(), compiled=str(compiled_dir), compute=args.compute, lock=lock_at_start)
    with JobWatch(note=note) as watch:
        result = asyncio.run(run(compiled_dir, args.compute, args.repeats, note))
    result["other_gpu_jobs"] = {"seen": watch.seen, "samples": watch.samples, "every_s": watch.every_s,
                                "note": "recorded only; correctness runs may overlap (no ms measured)"}
    result["measurement_lock"] = {"at_start": lock_at_start, "at_end": lock_state()}
    note(event="end", status=result["status"], lock=result["measurement_lock"]["at_end"])
    out = args.out.resolve() if args.out is not None else free_name(
        WORK / "results" / f"runtime_{result['precision']}_L{result['seq_len']}_{args.compute}.json")
    write_json(out, result)
    gate = compiled_dir / "provenance" / "runtime-gate.json"
    merged = json.loads(gate.read_text()) if gate.exists() else {}
    key = args.compute
    for i in range(2, 100):
        if key not in merged:
            break
        key = f"{args.compute}.run{i}"
    merged[key] = {k: v for k, v in result.items() if k != "rows"} | {"rows_record": str(out)}
    write_json(gate, merged)
    s = result["summary"]
    print(result["status"], args.compute, result["precision"],
          f"argmax {s['argmax_equal']}/{s['rows']} (non-near-tie {s['non_near_tie']['argmax_equal']}/{s['non_near_tie']['rows']},"
          f" near-tie {s['near_tie']['argmax_equal']}/{s['near_tie']['rows']})",
          f"max|dp| {s['max_abs_dp']:.3e} mean {s['mean_row_max_abs_dp']:.3e} max|dlogit| {s['max_abs_dlogit']:.3e}",
          f"drift {result['repeat_drift']['max_abs_scores']:.3e}",
          f"control {result['wrong_pairing_control']['status']}",
          f"load {result['load_s']:.1f} s", f"maxrss {result['memory']['end']['ru_maxrss_bytes'] / 2**30:.2f} GiB",
          f"other GPU jobs {len(watch.seen)}", "failures", result["failures"], "->", out, flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
