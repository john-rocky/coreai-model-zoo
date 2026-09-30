#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["coreai-core==1.0.0b2", "numpy>=2.2", "tokenizers>=0.22"]
# ///
"""Stage 4 (Mac): the exported bundle on the Core AI runtime, per compute unit, vs the oracle's rows.

    python3 gate_julia_runtime.py <exports>/julia-1/macos/fp32-s1024 --compute cpu_only
    python3 gate_julia_runtime.py <dir> --compute gpu              # waits for the flock GPU lock, runs only when quiet

The specialization options are always explicit (cpu_only(), or a preferred GPU kind), never None.
Refuses an iOS-targeted folder: an iPhone bundle is never loaded on a Mac.

Per row: `main` -> token logits at the marker positions, the deployed pipeline, once for every row of the
folder's reference.json (the typed-decisions questions that fit the window, the publisher's parity
requests, the window-filling rows) and 1 + --repeats times for every --drift-stride-th row. Recorded
unrounded per row: the raw marker logits and the milliseconds. Gate: argmax on every row and max |dp|
<= 1e-3 at T = 1 vs the publisher's engine; repeat drift <= 1e-6 on cpu_only (reported on the GPU); a
wrong-pairing control (each row judged against another same-shape row's outputs) must FAIL. The tensor
bar vs the publisher's padded batch-1 run (marker |d| <= 1e-3) is reported; a miss is recorded as a
finding, not hidden and not re-tuned. The typed rows re-score the benchmark from the bundle's argmax.
Timing: load ms and >= 30 warm samples of one whole question (NumPy in, graph, gather out). A GPU number
measured while another GPU job runs is not recorded: gpu runs hold the machine-wide flock on _GPU_LOCK for
their whole duration, start only when no other GPU job is visible, and discard the result if one appears.

Results merge into <variant>/provenance/runtime-gate.json under the compute unit's key.
"""
import argparse
import asyncio
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import coreai.runtime as rt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_SHA, environment, sha256_of, write_json  # noqa: E402
from _gate_metrics import POLICY, evaluate_row, summarize, wrong_pairing  # noqa: E402
from _julia_host import gather_markers, graph_inputs  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import gpu_lock  # noqa: E402

DRIFT_BAR = 1e-6
QTYPES = ("choice", "score", "noul")


def options_for(compute: str):
    if compute == "cpu_only":
        return rt.SpecializationOptions.cpu_only()
    return rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())


GPU_JOB_PATTERN = r"readout_gate|coreai_gate|gate_.*\.py|export_.*\.py|llm-runner|llm-benchmark|dashboard_job\.py|decide-cli"


def _ps(pid: int, field: str) -> str:
    return subprocess.run(["ps", "-o", f"{field}=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()


def other_gpu_jobs() -> list[str]:
    """Other processes that may use the GPU. PIDs from `pgrep -f`, never its -l text: a Claude session's
    argv holds its whole prompt (script names included). This process's own ancestors and Claude
    processes are skipped; a shell only carries the matching text, its child is the job."""
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
        jobs.append(f"{pid} {command} {' '.join(_ps(pid, 'args').split())[:160]}")
    return jobs


def lock_openers(path: Path) -> list[str]:
    pids = subprocess.run(["lsof", "-t", str(path)], capture_output=True, text=True).stdout.split()
    return [f"{pid} {_ps(int(pid), 'comm')}" for pid in pids if pid.isdigit() and int(pid) != os.getpid()]


def acquire_gpu_lock(max_wait_s: float):
    """The machine-wide GPU lock is an fcntl.flock on _GPU_LOCK (coreai-kit scripts/with-gpu-lock.py); the
    0-byte file stays after release, so its existence means nothing. Try LOCK_EX without blocking every
    30 s; once held, run only if no other GPU job is visible."""
    path = gpu_lock()
    handle = open(path, "a")
    started, attempts = time.monotonic(), 0
    while True:
        attempts += 1
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if time.monotonic() - started > max_wait_s:
                handle.close()
                raise SystemExit(f"GPU lock still held after {max_wait_s:.0f} s: {lock_openers(path)}")
            if attempts == 1 or attempts % 10 == 0:
                print(f"GPU lock held ({lock_openers(path)}); retrying every 30 s", flush=True)
            time.sleep(30)
            continue
        jobs = other_gpu_jobs()
        if not jobs:
            return handle, {"path": str(path), "waited_s": time.monotonic() - started, "attempts": attempts,
                            "other_openers_at_acquire": lock_openers(path)}
        fcntl.flock(handle, fcntl.LOCK_UN)
        if time.monotonic() - started > max_wait_s:
            handle.close()
            raise SystemExit(f"other GPU jobs keep running without the lock: {jobs}")
        print(f"lock free but other GPU jobs run ({jobs}); retrying in 30 s", flush=True)
        time.sleep(30)


class GpuWatch:
    """Samples other_gpu_jobs() every few seconds while a GPU gate runs."""

    def __init__(self, every_s: float = 5.0):
        import threading
        self.seen, self.every_s, self._stop = [], every_s, threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.wait(self.every_s):
            self.seen.extend(job for job in other_gpu_jobs() if job not in self.seen)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.seen.extend(job for job in other_gpu_jobs() if job not in self.seen)


async def question(main_fn, arrays: dict, markers) -> tuple[list[float], float]:
    """One whole question: NumPy in, main, the host's gather. Returns the K logits and wall ms."""
    t0 = time.perf_counter()
    out = await main_fn({k: rt.NDArray(v) for k, v in arrays.items()})
    marker = gather_markers(np.array(out["token_logits"].numpy(), copy=True), markers)
    return marker, (time.perf_counter() - t0) * 1000


async def run(variant_dir: Path, compute: str, repeats: int, drift_stride: int, warm_samples: int) -> dict:
    manifest = json.loads((variant_dir / "provenance" / "export-manifest.json").read_text())
    assert manifest["intended_runtime"] == "macos" and not manifest["aot"], "Mac gate only: never load an iOS bundle here"
    bundle = variant_dir / manifest["bundle"]
    assert bundle.suffix == ".aimodel", bundle
    for item in manifest["files"]:
        file = bundle / item["path"]
        assert file.stat().st_size == item["bytes"] and sha256_of(file) == item["sha256"], f"artifact changed: {item['path']}"
    reference = json.loads((variant_dir / "reference.json").read_text())
    assert reference["model_sha"] == manifest["model_sha"] == MODEL_SHA and reference["window"] == manifest["window"]
    window, rows = reference["window"], reference["rows"]
    for row in rows:
        row["window"] = window
        row["qtype"] = QTYPES.index(row["type"])
    options = options_for(compute)

    start = time.perf_counter()
    model = await rt.AIModel.load(bundle, options)
    load_ms = (time.perf_counter() - start) * 1000
    assert sorted(model.function_names) == ["main"], model.function_names
    main_fn = model.load_function("main")
    first_call_ms = None
    for _ in range(5):  # specialization and first-call work happen here, outside every recorded row
        _, ms = await question(main_fn, graph_inputs(rows[0], window), rows[0]["markers"])
        first_call_ms = ms if first_call_ms is None else first_call_ms

    records, candidates = [], {}
    for index, row in enumerate(rows):
        arrays = graph_inputs(row, window)
        runs = 1 + (repeats if index % drift_stride == 0 else 0)
        results = [await question(main_fn, arrays, row["markers"]) for _ in range(runs)]
        marker, _ = results[0]
        drift = max((float(np.max(np.abs(np.asarray(m) - np.asarray(marker)))) for m, _ in results[1:]), default=None)
        record = evaluate_row(row, marker)
        record.update(ms=float(np.median([ms for _, ms in results])), repeat_max_abs=drift)
        records.append(record)
        candidates[row["row_id"]] = marker
    samples = []
    for i in range(warm_samples):
        row = rows[(i * 37) % len(rows)]
        samples.append((await question(main_fn, graph_inputs(row, window), row["markers"]))[1])
    assert model is not None  # keep the model alive until every output is copied

    summary = summarize(records)
    control = wrong_pairing(rows, candidates)
    drifts = [r["repeat_max_abs"] for r in records if r["repeat_max_abs"] is not None]
    max_drift = max(drifts)
    failures, findings = [], []
    if summary["answer_status"] != "PASS":
        failures.append("answer gate")
    if compute == "cpu_only" and max_drift > DRIFT_BAR:
        failures.append(f"non-deterministic on cpu_only (max repeat drift {max_drift:.3g})")
    if control["status"] != "FAIL":
        failures.append("wrong-pairing control passed")
    if summary["tensor_status"] != "PASS":
        findings.append(f"tensor bar missed: marker {summary['max_marker_abs_error']:.3g} (bar 1e-3) on "
                        f"{len(summary['tensor_failures'])} rows")
    return {
        "status": "FAIL" if failures else "PASS", "failures": failures, "findings": findings,
        "stage": "runtime (Mac)", "compute": compute, "specialization_options": str(options).strip(),
        "bundle": manifest["bundle"], "precision": manifest["precision"], "window": window,
        "model_sha": MODEL_SHA, "policy": POLICY, "drift_bar_cpu_only": DRIFT_BAR, "summary": summary,
        "repeat_max_abs": max_drift, "repeat_rows": len(drifts), "repeats_per_drift_row": repeats,
        "wrong_pairing_control": control, "load_ms": load_ms, "first_question_ms": first_call_ms,
        "warm": {"samples": len(samples), "median_ms": float(np.median(samples)), "p10_ms": float(np.percentile(samples, 10)),
                 "p90_ms": float(np.percentile(samples, 90)), "min_ms": min(samples), "max_ms": max(samples),
                 "what": "one question: NumPy inputs, main, the host's marker gather",
                 "note": "cpu_only is the parity option; its time is a reference value, not a speed claim" if compute == "cpu_only" else None,
                 "samples_ms": samples},
        "row_ms_median": float(np.median([r["ms"] for r in records])),
        "environment": environment(("coreai-core", "numpy")), "process_pid": os.getpid(),
        "rows": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("variant_dir", type=Path, help="an exported macos/<dtype>-s<S> directory")
    parser.add_argument("--compute", choices=["cpu_only", "gpu"], required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--drift-stride", type=int, default=10, help="repeat every n-th row (drift)")
    parser.add_argument("--warm-samples", type=int, default=60)
    parser.add_argument("--lock-wait-minutes", type=float, default=30, help="gpu: how long to wait for the GPU lock")
    parser.add_argument("--output", type=Path, help="write the result here instead of merging into <variant>/provenance/runtime-gate.json")
    args = parser.parse_args()
    if args.warm_samples < 30:
        parser.error("at least 30 warm samples")
    handle, lock_record, watch = None, None, None
    if args.compute == "gpu":
        handle, lock_record = acquire_gpu_lock(args.lock_wait_minutes * 60)
    try:
        if args.compute == "gpu":
            with GpuWatch() as watch:
                result = asyncio.run(run(args.variant_dir, args.compute, args.repeats, args.drift_stride, args.warm_samples))
        else:
            result = asyncio.run(run(args.variant_dir, args.compute, args.repeats, args.drift_stride, args.warm_samples))
    finally:
        if handle is not None:
            handle.close()  # releases the flock; the file stays
    if watch is not None:
        result["gpu_lock"] = lock_record | {"other_gpu_jobs_seen_during_run": watch.seen}
        if watch.seen:
            raise SystemExit(f"another GPU job ran during the gate ({watch.seen}); the numbers are discarded, not recorded")
    if args.output:
        write_json(args.output, {args.compute: result})
    else:
        out = args.variant_dir / "provenance" / "runtime-gate.json"
        merged = json.loads(out.read_text()) if out.exists() else {}
        merged[args.compute] = result
        write_json(out, merged)
    s = result["summary"]
    print(result["status"], args.compute, f"rows {s['rows']}", f"argmax {s['argmax_identical']}/{s['rows']}",
          f"max dp {s['max_probability_error']:.2e}", f"marker {s['max_marker_abs_error']:.2e}",
          f"typed {json.dumps(s.get('typed_accuracy'))}",
          f"drift {result['repeat_max_abs']:.1e}", f"control {result['wrong_pairing_control']['status']}",
          f"load {result['load_ms']:.0f} ms", f"warm median {result['warm']['median_ms']:.2f} ms",
          "failures", result["failures"], "findings", result["findings"])
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
