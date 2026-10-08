#!/usr/bin/env python3
"""Compute-unit and AOT-flag probes of the d1 bundles: the facts the speed ladder's `ane` row and `aot` lever rest on.

    ane     The decoder (fp16, S=16) and the tower (fp16w32) compiled with `--preferred-compute neural-engine` (the
            decoder with --expect-frequent-reshapes as it ships, the tower without, as it ships), each into its own
            directory: the compile's full stdout / stderr (a log file), seconds, bytes, the compiled asset's Neural
            Engine regions (the unique `*_ANE_region_<n>` names in its tree), its delegate kinds and stats.json, and
            whether its main-h16c.mlirb equals the GPU asset's (then the two share one runtime cache entry name).
            Then each asset is loaded with `SpecializationOptions.from_preferred_compute_unit_kind(
            ComputeUnitKind.neural_engine())` and run on a fixed subset, compared bit for bit with the GPU AOT's
            record and with the oracle:
              decoder  every 10th row of the GPU gate's order (rows 0, 10, .., 390: 40 rows), the hidden rows'
                       sha256 against the gate transcript's, the readout's p against the fp32 oracle
              tower    every 7th crop of the oracle's order (crops 0, 7, .., 63: 10 crops), the output's sha256
                       against the GPU gate worker's, the rows against transformers' (cos, lowest row cos, max |d|)
            A control per graph: the shipped GPU AOT asset loaded with the same neural-engine preference.
    noefr   The decoder (fp16, S=16) compiled without --expect-frequent-reshapes (gpu preferred, h16c) into its own
            directory (bytes, seconds, main.hash), loaded with `SpecializationOptions.default()` and run on one row of
            49..64 ids (the first in the GPU gate's order: 4 calls, position lengths 16, 32, 48, 64) twice in one
            process (lap 1, lap 2), in two processes one after the other: the wall of every call (a first-time
            specialization per new position length shows as seconds), the hidden rows against the efr asset's
            record and the probabilities against the oracle, the runtime cache entry and the MPSGraph scratch
            ($TMPDIR/com.apple.MetalPerformanceShadersGraph) before and after each process.
            `--options neural_engine --asset <that asset>`: the same calls with the Neural Engine preferred.

Each worker's output is read for the runtime's own Neural Engine attempts (ANECCompile failures and their error text,
the MPSGraph `failed:` warnings, the ANE validation messages and the ops / source lines they name), and its MPSGraph
scratch directory (`mpsgraph-<pid>-*`: a region's IR and compiler options when the runtime tried the Neural Engine) is
listed; each compiled asset's `*.mpsgraph` files are read for their region names and ANE validation messages.
`ane --graphs tower --tower <another tower bundle> --tower-gate <its gate> --no-control --tag <t>` probes one graph.

Never the JIT: only compiled `.aimodelc` assets are loaded. Every asset this script compiles shares the source
`.aimodel` with a shipped GPU asset, and the Python runtime names its cache entry by the asset's main.hash (round 5a's
trap: an entry made by another asset of the same name runs that asset's graph, silently). So before a probe asset is
loaded, an entry under its main.hash that holds another asset is renamed aside (`<hash>__aside_r6a_<label>`); after
the run the entry's manifest.plist sha256 set is checked against the probe asset's, the probe's entry is renamed
`<hash>__probe_r6a_<label>` (removed at the round's close, by name) and the aside entry gets its name back. A compile
that runs past its limit (`--compile-limit`, counted only while no measurement window is open) is killed and recorded.
The transcript opens with the rules (the subsets, what is compared, the words a result maps to) before any compile.
The GPU is shared with other sessions: every ms here is a contended reference value, never the ladder.

    cd conversion/d1
    export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer PY=<coreai-models venv>/bin/python
    Q="$HOME/code/standup/tools/quiet/quiet_wait.py --max-wait 3600 --"
    $Q $PY compute_unit_probe.py ane --transcript $ZOO_WORK_ROOT/_d1_3b/results/r6a_probe_ane.json
    $Q $PY compute_unit_probe.py noefr --transcript $ZOO_WORK_ROOT/_d1_3b/results/r6a_probe_noefr.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _paths import gpu_lock, work_path  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")

LANE = work_path("_d1_3b")
DECODER = LANE / "exports" / "bundles" / "d1_3b_decode_fp16_pf16"
TOWER = LANE / "exports" / "vision" / "d1_3b_vision_fp16w32"
DECODER_GATE = LANE / "results" / "r4_readout_fp16_pf16.json"     # the GPU AOT's record of every row
TOWER_GATE = LANE / "results" / "r5b_tower_fp16w32.json"           # the GPU AOT's record of every crop
ROW_STEP, CROP_STEP = 10, 7
NOEFR_ROW_TOKENS = (49, 64)
COMPILE_LIMIT_S = 30 * 60
RUN_LIMIT_S = 30 * 60
STOP_GAP_S = 5.0                        # a poll gap longer than this = the job tree was stopped by a window's guard
MIN_FREE_BYTES = 40 * 2**30             # a compile or a run is killed below this much free space
ANE_REGION = re.compile(r"([A-Za-z0-9_-]*?ANE_region_\d+)")
MANIFEST = "manifest.plist"
DP_BAR = {"max_abs_dp": 0.02, "near_tie_top2_margin": 0.02}
TOWER_BAR = {"cos": 0.99999, "min_row": 0.9999}        # gate_tower.py's fp16w32 bar
RULES = {
    "ane": {
        "compile": "coreai-build compile <x>.aimodel --output <probe dir> --platform macOS --preferred-compute "
                   "neural-engine --architecture h16c (+ --expect-frequent-reshapes for the decoder, as it ships)",
        "regions": "the unique `*_ANE_region_<n>` names in the compiled asset's tree (a region's .bc directory and the "
                   ".mlir.bc inside it are one region)",
        "load": "AIModel.load(<asset>, SpecializationOptions.from_preferred_compute_unit_kind("
                "ComputeUnitKind.neural_engine()))",
        "decoder_subset": f"every {ROW_STEP}th row of the GPU gate's run order (0, {ROW_STEP}, ..): 40 rows",
        "tower_subset": f"every {CROP_STEP}th crop of the oracle's crop order (0, {CROP_STEP}, ..): 10 crops",
        "compared": "decoder: each row's hidden sha256 against the GPU gate transcript's (bit-equal = the same "
                    "arithmetic as the GPU asset), the readout's p against the fp32 oracle (argmax on non-near-tie "
                    "rows, max |dp| <= 0.02), finite; tower: each crop's output sha256 against the GPU gate worker's, "
                    "the rows against transformers' (cos >= 0.99999, lowest row cos >= 0.9999, gate_tower's fp16w32 bar)",
        "words": "measured = the asset holds at least one Neural Engine region and loads and runs (its parity is "
                 "recorded PASS or FAIL); no-go = the compile makes no Neural Engine region (whatever runs is not the "
                 "Neural Engine), or the compile does not end within the limit, or the load or a call fails",
        "control": "the shipped GPU AOT asset loaded with the same neural-engine preference, the same subset",
    },
    "noefr": {
        "compile": "coreai-build compile <decoder>.aimodel --output <probe dir> --platform macOS --preferred-compute gpu "
                   "--architecture h16c (no --expect-frequent-reshapes)",
        "load": "AIModel.load(<asset>, SpecializationOptions.default())",
        "row": f"the first row of the GPU gate's run order with {NOEFR_ROW_TOKENS[0]}..{NOEFR_ROW_TOKENS[1]} ids: "
               "4 calls of S=16, position lengths 16, 32, 48, 64",
        "processes": "two, one after the other; in each, lap 1 then lap 2 on the same row from fresh zero states",
        "compared": "each call's wall (a new position length's first call = a specialization), each lap's hidden "
                    "sha256 against the efr asset's record of the row, the readout's p against the oracle; the "
                    "runtime cache entry's size and the MPSGraph scratch's before and after each process",
    },
    "static": {
        "asset": "the static form's AOT (export_decoder.py --static --aot: h16c, --preferred-compute gpu, no "
                 "--expect-frequent-reshapes), loaded with SpecializationOptions.default(); nothing is compiled here",
        "rows": "tv4x_emotion_00's question (57 ids, 4 calls at S=16) and the first 32 calls of long_34k's first "
                "question (the row cut at 32 x S ids), from fresh zero states, position_ids = each call's own S positions",
        "processes": "two, one after the other; in each, lap 1 then lap 2 over both rows",
        "compared": "each call's wall: a position's first call in the process (lap 1) against its second (lap 2) and "
                    "the process's first call; each row's hidden sha256 lap 1 vs lap 2 and process 1 vs 2; the "
                    "readout's p against the oracle for the uncut row; the runtime cache entry named by the asset's "
                    "main.hash and every entry made, before and after each process; the MPSGraph scratch",
        "words": "no re-specialization = no call of lap 1 after the process's first call takes seconds (a new "
                 "position's first call within the same order of magnitude as its lap 2 call) in both processes, "
                 "and one cache entry",
    },
}
STATIC_ROWS = (("tv4x_emotion_00", 0, None), ("long_34k", 0, 32))   # (record, question, the calls kept; None = all)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def du_kib(path: Path) -> int | None:
    if not path.exists():
        return None
    out = subprocess.run(["du", "-sk", str(path)], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def free_bytes() -> int:
    return shutil.disk_usage("/System/Volumes/Data").free


def window() -> str | None:
    """quiet_wait.window(): the lock's content while a live `timing` window is open, else None."""
    p = gpu_lock()
    try:
        st = p.stat()
        content = p.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not content or "timing" not in content.lower():
        return None
    m = re.search(r"\bpid\s+(\d+)", content)
    if m:
        try:
            os.kill(int(m.group(1)), 0)
        except ProcessLookupError:
            return None
        except PermissionError:
            pass
    elif time.time() - st.st_mtime > 3 * 3600:
        return None
    return content


def run_limited(cmd: list[str], limit_s: float, stdout_path: Path, stderr_path: Path, env: dict | None = None) -> dict:
    """Run cmd with stdout / stderr to files; kill it when it has run `limit_s` seconds outside measurement windows,
    or when the data volume's free space falls under MIN_FREE_BYTES. A window's guard stops this whole job tree
    (this driver too), so a poll gap longer than STOP_GAP_S is stopped time and does not count, nor does time seen
    with a window open."""
    t0 = time.monotonic()
    active, last, window_s, stopped_s = 0.0, t0, 0.0, 0.0
    min_free = free_bytes()
    with open(stdout_path, "w") as so, open(stderr_path, "w") as se:
        proc = subprocess.Popen(cmd, stdout=so, stderr=se, env=env, start_new_session=True)
        killed = None
        while proc.poll() is None:
            time.sleep(1.0)
            t = time.monotonic()
            dt, last = t - last, t
            if dt > STOP_GAP_S:
                stopped_s += dt
            elif window():
                window_s += dt
            else:
                active += dt
            fb = free_bytes()
            min_free = min(min_free, fb)
            if active > limit_s or fb < MIN_FREE_BYTES:
                killed = "limit" if active > limit_s else f"free space {fb} B < {MIN_FREE_BYTES} B"
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                break
    return {"returncode": proc.returncode, "killed": killed, "killed_after_limit": killed == "limit", "limit_s": limit_s,
            "wall_seconds": time.monotonic() - t0, "active_seconds": active, "window_seconds_seen": window_s,
            "stopped_seconds_seen": stopped_s, "min_free_bytes_seen": min_free}


# --------------------------------------------------------------------------- compiled assets
def coreai_build() -> str:
    if not os.environ.get("DEVELOPER_DIR"):
        sys.exit("set DEVELOPER_DIR to the Xcode 27 RC (its Metal toolchain carries coreai-build)")
    cb = subprocess.run(["xcrun", "-f", "coreai-build"], capture_output=True, text=True)
    if cb.returncode != 0 or not cb.stdout.strip():
        sys.exit("xcrun -f coreai-build failed:\n" + cb.stderr)
    return cb.stdout.strip()


def flag_values(cb: str) -> dict:
    """What coreai-build accepts for --preferred-compute: its help line, and the parse of `ane` (on a path that does
    not exist, so nothing compiles whichever way the parse goes)."""
    h = subprocess.run([cb, "compile", "--help"], capture_output=True, text=True)
    line = " ".join(ln.strip() for ln in (h.stdout + h.stderr).splitlines()
                    if "preferred-compute" in ln or "gpu, neural-engine" in ln or "values:" in ln and "gpu" in ln)
    p = subprocess.run([cb, "compile", "/nonexistent/x.aimodel", "--preferred-compute", "ane"], capture_output=True,
                       text=True, timeout=60)
    return {"help": line, "ane_value": {"returncode": p.returncode, "stderr": p.stderr.strip()[:600],
                                        "stdout": p.stdout.strip()[:600]}}


def graph_strings(asset: Path) -> dict:
    """What the compiled MPSGraph files (`*.mpsgraph`, small; never resources.bin) say about the Neural Engine: their
    region names (`*_ANE_region_*` / `*_GPU_region_*`) and the ANE validation messages the compiler left in them."""
    out = {}
    for p in sorted(asset.rglob("*.mpsgraph")):
        text = re.findall(rb"[\x20-\x7e]{8,}", p.read_bytes())
        s = [t.decode() for t in text]
        out[str(p.relative_to(asset))] = {
            "regions": sorted({m.group(0) for x in s for m in [re.search(r"\w*_(?:ANE|GPU)_region_\w*", x)] if m}),
            "ane_messages": sorted({x for x in s if "ANE" in x and " " in x})}
    return out


def asset_tree(asset: Path) -> dict:
    files = sorted(p for p in asset.rglob("*") if p.is_file())
    regions = sorted({m.group(1) for p in asset.rglob("*") for m in [ANE_REGION.search(p.name)] if m})
    delegates = sorted({str(p.relative_to(asset)) for d in asset.glob("*-delegates") for p in d.iterdir()})
    return {"bytes": sum(p.stat().st_size for p in files),
            "files": {str(p.relative_to(asset)): p.stat().st_size for p in files},
            "dirs": sorted(str(p.relative_to(asset)) for p in asset.rglob("*") if p.is_dir()),
            "ane_regions": regions, "ane_region_count": len(regions), "delegate_kinds": delegates,
            "graph_strings": graph_strings(asset),
            "main_hash": (asset / "main.hash").read_bytes().hex() if (asset / "main.hash").exists() else None,
            "main_mlirb_sha256": sha256_file(asset / "main-h16c.mlirb") if (asset / "main-h16c.mlirb").exists() else None,
            "stats": json.loads((asset / "stats.json").read_text()) if (asset / "stats.json").exists() else None,
            "manifests": manifests(asset)}


def manifests(path: Path) -> list[str]:
    return sorted(sha256_file(p) for p in path.rglob(MANIFEST)) if path.exists() else []


def compile_probe(cb: str, aimodel: Path, out_dir: Path, preferred: str, efr: bool, label: str, log_dir: Path,
                  limit_s: float, gpu_asset: Path | None) -> dict:
    target = out_dir / f"{aimodel.stem}.h16c.aimodelc"
    if target.exists():
        raise SystemExit(f"{target} exists: a probe never overwrites an asset (remove it first, on purpose)")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [cb, "compile", str(aimodel), "--output", str(out_dir), "--platform", "macOS", "--preferred-compute",
           preferred, "--architecture", "h16c"] + (["--expect-frequent-reshapes"] if efr else [])
    so, se = log_dir / f"r6a_probe_compile_{label}.stdout.log", log_dir / f"r6a_probe_compile_{label}.stderr.log"
    print(f"[{label}] {' '.join(cmd)}", flush=True)
    rec = {"label": label, "command": cmd, "aimodel": str(aimodel),
           "aimodel_main_hash": (aimodel / "main.hash").read_bytes().hex(), "free_bytes_before": free_bytes(),
           "started": now()}
    rec.update(run_limited(cmd, limit_s, so, se))
    rec["finished"] = now()
    rec["free_bytes_after"] = free_bytes()
    rec["stdout_log"], rec["stderr_log"] = str(so), str(se)
    rec["stdout_lines"] = so.read_text(errors="replace").splitlines()
    rec["stderr_lines"] = se.read_text(errors="replace").splitlines()
    rec["ok"] = rec["returncode"] == 0 and not rec["killed"] and (target / "main.hash").exists()
    rec["asset"] = str(target)
    if rec["ok"]:
        rec["tree"] = asset_tree(target)
        if gpu_asset is not None:
            g = asset_tree(gpu_asset)
            rec["vs_gpu_asset"] = {"gpu_asset": str(gpu_asset), "gpu_main_hash": g["main_hash"],
                                   "same_main_hash": g["main_hash"] == rec["tree"]["main_hash"],
                                   "same_main_mlirb": g["main_mlirb_sha256"] == rec["tree"]["main_mlirb_sha256"],
                                   "gpu_bytes": g["bytes"], "gpu_files": g["files"], "gpu_ane_region_count": g["ane_region_count"],
                                   "same_manifests": g["manifests"] == rec["tree"]["manifests"],
                                   "same_stats": g["stats"] == rec["tree"]["stats"]}
    print(f"[{label}] rc {rec['returncode']} killed {rec['killed']} wall {rec['wall_seconds']:.1f} s "
          f"active {rec['active_seconds']:.1f} s; regions {rec.get('tree', {}).get('ane_region_count')}", flush=True)
    return rec


# --------------------------------------------------------------------------- the runtime cache entry
def cache_dir() -> Path:
    build = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    return Path.home() / "Library/Caches/coreai-cache" / build / "python"


def entries() -> list[str]:
    c = cache_dir()
    return sorted(p.name for p in c.iterdir()) if c.exists() else []


class EntryGuard:
    """The probe asset's runtime cache entry: another asset's entry under the same name renamed aside first; after
    the run the probe's entry checked against the asset and renamed __probe_r6a_<label>; the aside entry restored."""

    def __init__(self, asset: Path, label: str, own: bool = False):
        self.asset, self.label, self.own = asset, label, own
        self.hash = (asset / "main.hash").read_bytes().hex()
        self.entry = cache_dir() / self.hash
        self.rec: dict = {"asset": str(asset), "main_hash": self.hash, "entry": str(self.entry),
                          "asset_manifests": manifests(asset)}

    def __enter__(self):
        self.rec["entries_before"] = entries()
        self.aside = None
        if self.entry.exists():
            held = manifests(self.entry)
            self.rec["before"] = {"exists": True, "manifests": held,
                                  "holds_this_asset": bool(held) and held == self.rec["asset_manifests"]}
            if not self.own:
                self.aside = self.entry.with_name(f"{self.hash}__aside_r6a_{self.label}")
                if self.aside.exists():
                    raise SystemExit(f"{self.aside} exists")
                self.entry.rename(self.aside)
                self.rec["renamed_aside"] = str(self.aside)
        else:
            self.rec["before"] = {"exists": False}
        self.rec["entry_kib_before"] = du_kib(self.entry)
        return self

    def __exit__(self, *exc):
        after = entries()
        self.rec["new_entries"] = sorted(set(after) - set(self.rec["entries_before"]))
        held = manifests(self.entry)
        self.rec["after"] = {"exists": self.entry.exists(), "manifests": held, "kib": du_kib(self.entry),
                             "holds_this_asset": bool(held) and held == self.rec["asset_manifests"]}
        if not self.own and self.entry.exists():
            probe = self.entry.with_name(f"{self.hash}__probe_r6a_{self.label}")
            if probe.exists():
                raise SystemExit(f"{probe} exists")
            self.entry.rename(probe)
            self.rec["probe_entry"] = str(probe)
        if self.aside is not None:
            self.aside.rename(self.entry)
            self.rec["restored"] = str(self.entry)
            self.rec["restored_manifests"] = manifests(self.entry)
        return False


# --------------------------------------------------------------------------- worker: one process
def worker(spec_path: Path) -> int:
    import coreai.runtime as rt
    from readout_gate import check_contract, fn_desc, maybe

    spec = json.loads(spec_path.read_text())
    prefix = Path(spec["out"])
    out: dict = {"pid": os.getpid(), "started": time.time(), "graph": spec["graph"], "options_kind": spec["options"],
                 "calls": [], "rows": []}
    store: dict[str, np.ndarray] = {}

    def save() -> None:
        out["finished"] = time.time()
        np.savez(prefix.with_suffix(".npz"), **store)
        prefix.with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")

    def fail(stage: str, e: BaseException, **where) -> None:
        out["error"] = {"stage": stage, **where, "type": type(e).__name__, "text": str(e)[:4000],
                        "traceback": traceback.format_exc()[-6000:]}

    if spec["options"] == "neural_engine":
        opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.neural_engine())
    elif spec["options"] == "default":
        opts = rt.SpecializationOptions.default()
    else:
        raise SystemExit(f"options {spec['options']}")
    out["options"] = {"str": str(opts), "allowed": [str(k) for k in opts.allowed_compute_unit_kinds],
                      "preferred": str(opts.preferred_compute_unit_kind),
                      "available_kinds": [str(k) for k in rt.ComputeUnitKind.available_kinds()]}

    async def go() -> None:
        t0 = time.perf_counter()
        try:
            model = await maybe(rt.AIModel.load(spec["aimodelc"], opts))
            out["load_model_seconds"] = time.perf_counter() - t0
            fn = await maybe(model.load_function("main"))
        except Exception as e:  # noqa: BLE001
            fail("load", e)
            return
        out["load_seconds"] = time.perf_counter() - t0
        try:
            di = model._debug_infos
            raw = di if isinstance(di, (bytes, bytearray)) else str(di).encode()
            prefix.with_suffix(".debug_infos.json").write_bytes(raw)
            out["debug_infos"] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                                  "path": str(prefix.with_suffix(".debug_infos.json"))}
        except Exception as e:  # noqa: BLE001
            out["debug_infos"] = {"error": repr(e)}
        desc = fn_desc(fn)
        out["descriptor"] = desc
        if spec["graph"] == "decoder":
            await decoder_rows(rt, fn, desc, spec, out, store, fail, check_contract)
        else:
            await tower_crops(rt, fn, spec, out, store, fail)

    try:
        asyncio.run(go())
    except Exception as e:  # noqa: BLE001
        fail("run", e)
    save()
    return 0


async def decoder_rows(rt, fn, desc, spec, out, store, fail, check_contract) -> None:
    from readout_gate import maybe

    bad = check_contract(desc, spec["contract"])
    out["contract_mismatch"] = bad
    if bad:
        out["error"] = {"stage": "contract", "text": str(bad)}
        return
    S, max_ctx, H, pad = int(spec["S"]), int(spec["max_ctx"]), int(spec["hidden"]), int(spec["pad_id"])

    def nd(a):
        return rt.NDArray(np.ascontiguousarray(a))

    img_shape, img_dt = spec["contract"]["inputs"]["image_embeds"]
    img = nd(np.zeros(img_shape, np.dtype(img_dt)))
    static = bool(spec["contract"].get("static"))   # lfm2_d1_static.py: position_ids = the call's own S positions
    seen_lengths: set[int] = set()
    for lap in spec["laps"]:
        for j, run in enumerate(lap["rows"]):
            ids = np.asarray(run["ids"], np.int32)
            T = len(ids)
            n = -(-T // S)
            x = np.full(n * S, pad, np.int32)
            x[:T] = ids
            state = {name: nd(np.zeros([max_ctx if s < 0 else s for s in shape], np.dtype(dt)))
                     for name, (shape, dt) in desc["states"].items()}
            hid = np.zeros((n * S, H), np.float16)
            t_run = time.perf_counter()
            for c in range(n):
                L = (c + 1) * S      # the positions seen after this call (the dynamic form's position_ids length)
                t1 = time.perf_counter()
                try:
                    res = await maybe(fn(inputs={"input_ids": nd(x[c * S:(c + 1) * S].reshape(1, S)),
                                                 "position_ids": nd(np.arange(c * S if static else 0, L,
                                                                              dtype=np.int32)[None]),
                                                 "image_embeds": img}, state=state))
                    h = np.asarray(res["hidden"].numpy())
                except Exception as e:  # noqa: BLE001
                    fail("call", e, lap=lap["label"], row=f"{run['id']}:{run['k']}", call=c, position_length=L)
                    return
                ms = (time.perf_counter() - t1) * 1e3
                hid[c * S:(c + 1) * S] = h[0]
                out["calls"].append({"lap": lap["label"], "row": f"{run['id']}:{run['k']}", "call": c,
                                     "position_length": L, "first_position": c * S,
                                     "new_length_in_process": L not in seen_lengths, "ms": ms})
                seen_lengths.add(L)
            hid = hid[:T].copy()
            key = f"{lap['label']}__{j:02d}"
            store[f"{key}__slot"] = hid[run["slot"]].copy()
            out["rows"].append({"lap": lap["label"], "index": j, "key": key, "id": run["id"], "k": run["k"], "tokens": T,
                                "calls": n, "wall_seconds": time.perf_counter() - t_run,
                                "hidden_sha256": hashlib.sha256(hid.tobytes()).hexdigest(),
                                "finite": bool(np.isfinite(hid.astype(np.float32)).all()), "all_zero": bool(not np.any(hid))})
            print(f"  [{os.getpid()}] {lap['label']} {run['id']}:{run['k']}: {T} tok, {n} calls, "
                  f"{out['rows'][-1]['wall_seconds']:.3f} s", flush=True)


async def tower_crops(rt, fn, spec, out, store, fail) -> None:
    from gate_tower import INPUTS
    from readout_gate import maybe

    dt = {"float16": np.float16, "float32": np.float32, "int32": np.int32}
    in_dtype = {k: dt[v[1]] for k, v in spec["inputs"].items()}
    for i, crop in enumerate(spec["crops"]):
        with np.load(crop["npz"]) as z:
            feeds = {k: rt.NDArray(np.ascontiguousarray(z[k].astype(in_dtype[k]))) for k in INPUTS}
        t1 = time.perf_counter()
        try:
            res = await maybe(fn(inputs=feeds))
            o = np.asarray(res["image_embeds"].numpy()).copy()
        except Exception as e:  # noqa: BLE001
            fail("call", e, crop=crop["key"])
            return
        store[f"{i:03d}"] = o
        out["calls"].append({"index": i, "key": crop["key"], "ms": (time.perf_counter() - t1) * 1e3,
                             "shape": list(o.shape), "dtype": str(o.dtype),
                             "sha256": hashlib.sha256(o.tobytes()).hexdigest(),
                             "finite": bool(np.isfinite(o.astype(np.float64)).all())})


def ane_log_messages(*logs: Path) -> dict:
    """The runtime's Neural Engine attempts in a worker's output: ANECCompile failures and their error text, the
    MPSGraph warnings (`failed: ..`), the ANE validation messages and the ops / source lines they name."""
    text = "\n".join(p.read_text(errors="replace") for p in logs if p.exists())

    def count(pattern: str, group: int = 0) -> dict:
        c: dict[str, int] = {}
        for m in re.finditer(pattern, text):
            k = m.group(group).strip()
            c[k] = c.get(k, 0) + 1
        return dict(sorted(c.items(), key=lambda kv: -kv[1]))

    return {"anec_compile_failed": text.count("appleneuralengine.compiler Code=1 "),     # one per failed ANE compile
            "anec_errors": count(r'Code=22 "ANECCompile\([^)]*\) FAILED: err=\(\s*"([^"]*)"', 1),
            "warnings_failed": count(r"failed: ([A-Za-z][^\"\n]*)", 1),
            "ane_validation_messages": count(r'"ane_validation_message"\("([^"]*)"', 1),
            "ops": count(r'identifiers" = \[#aicode\.serialization\.array<\[#aicode\.serialization\.string<"(\w+)">', 1),
            "source_lines": count(r'filename = "([^"]+)", directory = "", sha256Sum = "">, startLine = (\d+)', 0)}


def mpsgraph_scratch(pid: int | None, since: float | None = None) -> dict | None:
    """The MPSGraph scratch a worker left ($TMPDIR/com.apple.MetalPerformanceShadersGraph/mpsgraph-<pid>-*, made after
    the worker started: a pid is reused): the files (Neural Engine region IR and compiler options when the runtime
    tried the Neural Engine) and their bytes."""
    if pid is None:
        return None
    dirs = sorted(d for d in scratch_dir().glob(f"mpsgraph-{pid}-*") if since is None or d.stat().st_mtime >= since - 5)
    return {"dirs": [str(d) for d in dirs],
            "files": {f"{d.name}/{p.name}": p.stat().st_size for d in dirs for p in sorted(d.iterdir()) if p.is_file()},
            "kib": sum(du_kib(d) or 0 for d in dirs)}


def run_worker(spec: dict, limit_s: float) -> dict:
    prefix = Path(spec["out"])
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".json", ".npz", ".debug_infos.json"):
        if prefix.with_suffix(suffix).exists():
            raise SystemExit(f"{prefix.with_suffix(suffix)} exists: a probe never overwrites a record")
    spec_path = prefix.with_suffix(".spec.json")
    spec_path.write_text(json.dumps(spec) + "\n")
    print(f"[worker] {prefix.name}: {spec['graph']} {spec['options']} {spec['aimodelc']}", flush=True)
    lim = run_limited([sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--spec", str(spec_path)], limit_s,
                      prefix.with_suffix(".stdout.log"), prefix.with_suffix(".stderr.log"))
    js = prefix.with_suffix(".json")
    rec = json.loads(js.read_text()) if js.exists() else {
        "error": {"stage": "process", "text": f"no worker record (exit {lim['returncode']}, killed {lim['killed']})"}}
    rec["process"] = lim
    rec["stderr_tail"] = prefix.with_suffix(".stderr.log").read_text(errors="replace").splitlines()[-40:]
    rec["ane_messages"] = ane_log_messages(prefix.with_suffix(".stdout.log"), prefix.with_suffix(".stderr.log"))
    rec["mpsgraph_scratch"] = mpsgraph_scratch(rec.get("pid"), rec.get("started"))
    rec["npz"] = str(prefix.with_suffix(".npz"))
    return rec


# --------------------------------------------------------------------------- comparisons
def decoder_subset(gate: dict, recs: dict) -> list[dict]:
    rows = []
    for i, r in enumerate(gate["runs"]):
        if i % ROW_STEP:
            continue
        q = recs[r["id"]]["questions"][r["k"]]
        rows.append({"id": r["id"], "k": r["k"], "ids": q["row_ids"], "slot": q["slot"], "gate_index": i})
    return rows


def noefr_row(gate: dict, recs: dict) -> dict:
    lo, hi = NOEFR_ROW_TOKENS
    for i, r in enumerate(gate["runs"]):
        if lo <= r["tokens"] <= hi:
            q = recs[r["id"]]["questions"][r["k"]]
            return {"id": r["id"], "k": r["k"], "ids": q["row_ids"], "slot": q["slot"], "gate_index": i}
    raise SystemExit(f"no row of {lo}..{hi} ids in the gate")


def gate_slots(gate: dict) -> dict:
    out, zs = {}, {}
    for r in gate["runs"]:
        npz = next((p["npz"] for p in gate["processes"] if p.get("shard") == r["shard"] and p.get("npz")), None)
        if npz and Path(npz).exists():
            z = zs.setdefault(npz, np.load(npz))
            out[(r["id"], r["k"])] = z[f"{r['npz_key']}__slot"]
    return out


def score_decoder(rec: dict, gate: dict, recs: dict, table: dict) -> dict:
    from parity_decoder_torch import score_probs
    from readout_gate import readout_probs

    g = {(r["id"], r["k"]): r for r in gate["runs"]}
    gslot = gate_slots(gate)
    z = np.load(rec["npz"]) if Path(rec["npz"]).exists() else None
    rows = []
    for r in rec.get("rows", []):
        q = recs[r["id"]]["questions"][r["k"]]
        slot = z[f"{r['key']}__slot"]
        s = score_probs(readout_probs(slot, q["groups"], table), q)
        gr = g[(r["id"], r["k"])]
        rows.append({**{k: r[k] for k in ("lap", "id", "k", "tokens", "calls", "finite", "all_zero", "hidden_sha256")},
                     "bit_equal_gpu_aot": r["hidden_sha256"] == gr["hidden_sha256"],
                     "slot_max_abs_diff_vs_gpu_aot": float(np.max(np.abs(slot.astype(np.float32)
                                                                         - gslot[(r["id"], r["k"])].astype(np.float32)))),
                     "argmax_equal": s["argmax_equal"], "near_tie": s["near_tie"], "max_abs_dp": s["max_abs_dp"],
                     "gpu_aot_max_abs_dp": gr["max_abs_dp"]})
    far = [x for x in rows if not x["near_tie"]]
    summ = {"rows": len(rows), "bit_equal_gpu_aot": sum(x["bit_equal_gpu_aot"] for x in rows),
            "argmax_equal_non_near_tie": f"{sum(x['argmax_equal'] for x in far)}/{len(far)}",
            "max_abs_dp": max((x["max_abs_dp"] for x in rows), default=None),
            "gpu_aot_max_abs_dp_same_rows": max((x["gpu_aot_max_abs_dp"] for x in rows), default=None),
            "slot_max_abs_diff_vs_gpu_aot": max((x["slot_max_abs_diff_vs_gpu_aot"] for x in rows), default=None),
            "finite_all": all(x["finite"] and not x["all_zero"] for x in rows)}
    summ["parity_pass"] = (bool(rows) and summ["finite_all"] and all(x["argmax_equal"] for x in far)
                           and summ["max_abs_dp"] <= DP_BAR["max_abs_dp"])
    return {"summary": summ, "rows": rows}


def score_tower(rec: dict, gpu_calls: dict, crops: list[dict]) -> dict:
    from gate_tower import compare

    z = np.load(rec["npz"]) if Path(rec["npz"]).exists() else None
    rows = []
    for c in rec.get("calls", []):
        crop = crops[c["index"]]
        with np.load(crop["npz"]) as f:
            want = f["image_embeds"].astype(np.float64)
        s = compare(z[f"{c['index']:03d}"][:crop["n_tokens"]], want)
        rows.append({"key": c["key"], "kind": crop["kind"], "n_tokens": crop["n_tokens"], "sha256": c["sha256"],
                     "bit_equal_gpu_aot": c["sha256"] == gpu_calls.get(c["key"]), **s})
    summ = {"crops": len(rows), "bit_equal_gpu_aot": sum(x["bit_equal_gpu_aot"] for x in rows),
            "min_cos": min((x["cos"] for x in rows), default=None),
            "min_row": min((x["min_row"] for x in rows), default=None),
            "max_abs": max((x["max_abs"] for x in rows), default=None),
            "finite_all": all(x["finite"] for x in rows)}
    summ["parity_pass"] = (bool(rows) and summ["finite_all"] and summ["min_cos"] >= TOWER_BAR["cos"]
                           and summ["min_row"] >= TOWER_BAR["min_row"])
    return {"summary": summ, "rows": rows}


def ms_summary(rec: dict) -> dict:
    ms = [c["ms"] for c in rec.get("calls", [])]
    return {"contended": True, "calls": len(ms), "median_ms": float(np.median(ms)) if ms else None,
            "first_call_ms": ms[0] if ms else None, "load_seconds": rec.get("load_seconds")}


# --------------------------------------------------------------------------- drivers
def env_record() -> dict:
    from readout_gate import env_record as gate_env

    e = gate_env()
    e["runtime"] = "coreai python runtime, compiled .aimodelc assets only (no JIT); the options per run are recorded"
    return e


def open_transcript(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")


def lock_state() -> dict:
    p = gpu_lock()
    return {"path": str(p), "content": p.read_text()[:200] if p.exists() else None}


def word_ane(compile_rec: dict, run: dict | None, parity: dict | None) -> dict:
    """The ladder's word for the Neural Engine row of one graph, with its cause, from the measured facts."""
    if not compile_rec.get("ok"):
        if compile_rec.get("killed_after_limit"):
            return {"word": "no-go", "cause": f"the compile did not end within {compile_rec['limit_s'] / 60:.0f} min "
                                              "(killed)"}
        if compile_rec.get("killed"):
            return {"word": "unresolved", "cause": f"the compile was killed: {compile_rec['killed']}"}
        return {"word": "no-go", "cause": f"the compile failed (exit {compile_rec.get('returncode')}): "
                                          + " | ".join(compile_rec.get("stderr_lines", [])[-3:])[:300]}
    n = compile_rec["tree"]["ane_region_count"]
    if run is None:
        return {"word": "no-go" if n == 0 else "unresolved", "cause": "not loaded"}
    if run.get("error"):
        return {"word": "no-go", "cause": f"{run['error']['stage']} failed: {run['error'].get('type')} "
                                          f"{run['error'].get('text', '')[:200]}"}
    if n == 0:
        same = parity and parity["summary"]["bit_equal_gpu_aot"] == parity["summary"].get("rows", parity["summary"].get("crops"))
        return {"word": "no-go",
                "cause": "coreai-build --preferred-compute neural-engine made 0 Neural Engine regions (the asset is "
                         "the GPU's MPSGraph package)" + ("; the neural-engine-preferred load ran bit-equal to the GPU "
                                                          "AOT on the subset = the GPU ran it" if same else "")}
    return {"word": "measured", "cause": f"{n} Neural Engine region(s); parity "
                                          f"{'PASS' if parity and parity['summary']['parity_pass'] else 'FAIL'}"}


def ane_cmd(args) -> int:
    from parity_decoder_torch import load_oracle
    from readout_gate import Bundle

    out = Path(args.transcript)
    if out.exists():
        raise SystemExit(f"{out} exists: transcripts are never overwritten")
    cb = coreai_build()
    dec, tow = Bundle(args.decoder), Path(args.tower).resolve()
    tmeta = json.loads((tow / "metadata.json").read_text())
    tname = tmeta["name"]
    tow_gpu = tow.parent.parent / f"{tow.parent.name}_aotc" / f"{tname}.h16c.aimodelc"
    gate = json.loads(Path(args.decoder_gate).read_text())
    oracle_path = Path(gate["oracle"]["path"])
    if sha256_file(oracle_path) != gate["oracle"]["sha256"]:
        raise SystemExit("the decoder gate transcript was read against another oracle")
    if gate["bundle"]["aimodelc"]["path"] != str(dec.aimodelc):
        raise SystemExit(f"the decoder gate transcript is of {gate['bundle']['aimodelc']['path']}, not {dec.aimodelc}")
    _, recs = load_oracle(oracle_path)
    tgate = json.loads(Path(args.tower_gate).read_text())
    tor = json.loads(Path(tgate["oracle"]["path"]).read_text())
    crops_all = [{"key": f"{p['id']}/{c['crop']}", **c} for p in tor["pictures"] for c in p["crops"]]
    crops = [c for i, c in enumerate(crops_all) if i % CROP_STEP == 0]
    gpu_calls = {}
    for p in tgate["processes"]:
        if p["phase"] == "main":
            for c in json.loads(Path(p["npz"]).with_suffix(".json").read_text())["calls"]:
                if c["variant"] == "base":
                    gpu_calls[c["key"]] = c["sha256"]
    rows = decoder_subset(gate, recs)
    work = Path(args.work)
    graphs = [g for g in args.graphs.split(",") if g]
    tag = args.tag
    doc = {"schema": "d1-compute-unit-probe/1", "probe": "ane", "status": "running", "started": now(),
           "rules": RULES["ane"], "graphs": graphs, "tag": tag, "note": args.note, "control": not args.no_control,
           "decoder": {"bundle": str(dec.dir), "gpu_asset": str(dec.aimodelc),
                       "gpu_gate": str(Path(args.decoder_gate).resolve()),
                       "subset": [f"{r['id']}:{r['k']}" for r in rows]},
           "tower": {"bundle": str(tow), "gpu_asset": str(tow_gpu), "gpu_gate": str(Path(args.tower_gate).resolve()),
                     "subset": [c["key"] for c in crops]},
           "out_dir": str(Path(args.out_dir).resolve()), "work": str(work)}
    for g in ("decoder", "tower"):
        if g not in graphs:
            doc[g] = {"skipped": True}
    open_transcript(out, doc)
    doc["environment"] = env_record()
    doc["coreai_build"] = {"path": cb, "version": subprocess.run([cb, "--version"], capture_output=True, text=True).stdout.strip(),
                           **flag_values(cb)}
    doc["lock_start"] = lock_state()
    t0 = time.monotonic()
    log_dir = LANE / "logs"
    od = Path(args.out_dir).resolve()
    table = dec.table()

    # the decoder: compile, then the probe asset and the GPU asset (control) with the neural-engine preference
    if "decoder" in graphs:
        dcomp = compile_probe(cb, dec.dir / f"{dec.name}.aimodel", od, "neural-engine", True, f"ane_decoder{tag}",
                              log_dir, args.compile_limit, dec.aimodelc)
        doc["decoder"]["compile"] = dcomp
        open_transcript(out, doc)
        base = {"graph": "decoder", "S": dec.S, "max_ctx": dec.max_ctx, "hidden": dec.hidden, "pad_id": dec.pad_id,
                "contract": dec.contract, "laps": [{"label": "subset", "rows": rows}]}
        drun = dpar = None
        if dcomp["ok"]:
            asset = Path(dcomp["asset"])
            with EntryGuard(asset, f"ane_decoder{tag}") as eg:
                drun = run_worker({**base, "aimodelc": str(asset), "options": "neural_engine",
                                   "out": str(work / f"ane_decoder{tag}")}, args.run_limit)
            doc["decoder"]["entry"] = eg.rec
            doc["decoder"]["run"] = {k: v for k, v in drun.items() if k not in ("calls", "rows")}
            if not drun.get("error"):
                dpar = score_decoder(drun, gate, recs, table)
                doc["decoder"]["parity"] = dpar
                doc["decoder"]["ms"] = ms_summary(drun)
        open_transcript(out, doc)
        if not args.no_control:
            with EntryGuard(dec.aimodelc, f"gpu_decoder_control{tag}", own=True) as eg:
                crun = run_worker({**base, "aimodelc": str(dec.aimodelc), "options": "neural_engine",
                                   "out": str(work / f"control_gpu_decoder{tag}")}, args.run_limit)
            doc["decoder"]["control"] = {"what": "the shipped GPU AOT asset loaded with the neural-engine preference",
                                         "entry": eg.rec,
                                         "run": {k: v for k, v in crun.items() if k not in ("calls", "rows")}}
            if not crun.get("error"):
                doc["decoder"]["control"]["parity"] = score_decoder(crun, gate, recs, table)
                doc["decoder"]["control"]["ms"] = ms_summary(crun)
        doc["decoder"]["verdict"] = word_ane(dcomp, drun, dpar)
        open_transcript(out, doc)

    # the tower
    if "tower" in graphs:
        tcomp = compile_probe(cb, tow / f"{tname}.aimodel", od, "neural-engine", False, f"ane_tower{tag}", log_dir,
                              args.compile_limit, tow_gpu)
        doc["tower"]["compile"] = tcomp
        open_transcript(out, doc)
        tbase = {"graph": "tower", "inputs": tmeta["graph"]["inputs"],
                 "crops": [{"key": c["key"], "npz": c["npz"]} for c in crops]}
        trun = tpar = None
        if tcomp["ok"]:
            asset = Path(tcomp["asset"])
            with EntryGuard(asset, f"ane_tower{tag}") as eg:
                trun = run_worker({**tbase, "aimodelc": str(asset), "options": "neural_engine",
                                   "out": str(work / f"ane_tower{tag}")}, args.run_limit)
            doc["tower"]["entry"] = eg.rec
            doc["tower"]["run"] = {k: v for k, v in trun.items() if k != "calls"}
            if not trun.get("error"):
                tpar = score_tower(trun, gpu_calls, crops)
                doc["tower"]["parity"] = tpar
                doc["tower"]["ms"] = ms_summary(trun)
        open_transcript(out, doc)
        if not args.no_control:
            with EntryGuard(tow_gpu, f"gpu_tower_control{tag}", own=True) as eg:
                ctrun = run_worker({**tbase, "aimodelc": str(tow_gpu), "options": "neural_engine",
                                    "out": str(work / f"control_gpu_tower{tag}")}, args.run_limit)
            doc["tower"]["control"] = {"what": "the shipped GPU AOT asset loaded with the neural-engine preference",
                                       "entry": eg.rec, "run": {k: v for k, v in ctrun.items() if k != "calls"}}
            if not ctrun.get("error"):
                doc["tower"]["control"]["parity"] = score_tower(ctrun, gpu_calls, crops)
                doc["tower"]["control"]["ms"] = ms_summary(ctrun)
        doc["tower"]["verdict"] = word_ane(tcomp, trun, tpar)
    doc["lock_end"] = lock_state()
    doc["seconds"] = time.monotonic() - t0
    doc["status"] = "done"
    doc["script"] = {"path": "conversion/d1/compute_unit_probe.py", "sha256": sha256_file(Path(__file__).resolve())}
    doc["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    open_transcript(out, doc)
    for g in graphs:
        v = doc[g]["verdict"]
        print(f"{g}: {v['word']}: {v['cause']}")
    return 0


def scratch_dir() -> Path:
    return Path(tempfile.gettempdir()) / "com.apple.MetalPerformanceShadersGraph"


def noefr_cmd(args) -> int:
    from parity_decoder_torch import load_oracle
    from readout_gate import Bundle

    out = Path(args.transcript)
    if out.exists():
        raise SystemExit(f"{out} exists: transcripts are never overwritten")
    cb = coreai_build()
    dec = Bundle(args.decoder)
    gate = json.loads(Path(args.decoder_gate).read_text())
    oracle_path = Path(gate["oracle"]["path"])
    if sha256_file(oracle_path) != gate["oracle"]["sha256"]:
        raise SystemExit("the decoder gate transcript was read against another oracle")
    if gate["bundle"]["aimodelc"]["path"] != str(dec.aimodelc):
        raise SystemExit(f"the decoder gate transcript is of {gate['bundle']['aimodelc']['path']}, not {dec.aimodelc}")
    _, recs = load_oracle(oracle_path)
    row = noefr_row(gate, recs)
    g = next(r for r in gate["runs"] if (r["id"], r["k"]) == (row["id"], row["k"]))
    work = Path(args.work)
    doc = {"schema": "d1-compute-unit-probe/1", "probe": "noefr", "status": "running", "started": now(),
           "rules": RULES["noefr"], "decoder": {"bundle": str(dec.dir), "efr_asset": str(dec.aimodelc),
                                                "gpu_gate": str(Path(args.decoder_gate).resolve())},
           "row": {"id": row["id"], "k": row["k"], "tokens": len(row["ids"]), "gate_index": row["gate_index"],
                   "position_lengths": [(c + 1) * dec.S for c in range(-(-len(row["ids"]) // dec.S))],
                   "efr_asset_record": {"hidden_sha256": g["hidden_sha256"], "max_abs_dp": g["max_abs_dp"],
                                        "call_ms_contended": g["call_ms"], "first_in_process": g["first_in_process"]}},
           "out_dir": str(Path(args.out_dir).resolve()), "work": str(work), "options": args.options,
           "tag": args.tag, "note": args.note}
    open_transcript(out, doc)
    doc["environment"] = env_record()
    doc["lock_start"] = lock_state()
    t0 = time.monotonic()
    if args.asset:     # an efr-less asset this probe compiled before: reused as it is, its tree recorded
        a = Path(args.asset).resolve()
        comp = {"reused": True, "asset": str(a), "ok": (a / "main.hash").exists(), "tree": asset_tree(a),
                "wall_seconds": None, "active_seconds": None}
    else:
        comp = compile_probe(cb, dec.dir / f"{dec.name}.aimodel", Path(args.out_dir).resolve(), "gpu", False,
                             f"noefr_decoder{args.tag}", LANE / "logs", args.compile_limit, dec.aimodelc)
    doc["compile"] = comp
    open_transcript(out, doc)
    if comp["ok"]:
        asset = Path(comp["asset"])
        table = dec.table()
        base = {"graph": "decoder", "S": dec.S, "max_ctx": dec.max_ctx, "hidden": dec.hidden, "pad_id": dec.pad_id,
                "contract": dec.contract, "aimodelc": str(asset), "options": args.options,
                "laps": [{"label": "lap1", "rows": [row]}, {"label": "lap2", "rows": [row]}]}
        procs = []
        with EntryGuard(asset, f"noefr_decoder{args.tag}") as eg:
            for n in range(1, args.processes + 1):
                before = {"entry_kib": du_kib(eg.entry), "scratch_kib": du_kib(scratch_dir()), "free_bytes": free_bytes(),
                          "entries": entries()}
                rec = run_worker({**base, "out": str(work / f"noefr{args.tag}_process{n}")}, args.run_limit)
                after = {"entry_kib": du_kib(eg.entry), "scratch_kib": du_kib(scratch_dir()), "free_bytes": free_bytes(),
                         "entries": entries()}
                p = {"process": n, "before": {k: v for k, v in before.items() if k != "entries"},
                     "after": {k: v for k, v in after.items() if k != "entries"},
                     "new_entries": sorted(set(after["entries"]) - set(before["entries"])),
                     "run": {k: v for k, v in rec.items() if k not in ("rows",)}}
                if not rec.get("error"):
                    p["parity"] = score_decoder(rec, gate, recs, table)
                    p["laps"] = {lap: [round(c["ms"], 3) for c in rec["calls"] if c["lap"] == lap] for lap in ("lap1", "lap2")}
                    p["lap_hidden_bit_equal"] = len({r["hidden_sha256"] for r in rec["rows"]}) == 1
                procs.append(p)
                doc["processes"] = procs
                open_transcript(out, doc)
                if rec.get("error"):
                    break
        doc["entry"] = eg.rec
        doc["scratch_dir"] = str(scratch_dir())
        lap1 = [p["laps"]["lap1"] for p in procs if "laps" in p]
        doc["summary"] = {
            "runs": all(not p["run"].get("error") for p in procs) and len(procs) == args.processes,
            "options": args.options,
            "bytes": comp["tree"]["bytes"], "efr_asset_bytes": comp.get("vs_gpu_asset", {}).get("gpu_bytes"),
            "compile_seconds": comp["wall_seconds"], "compile_active_seconds": comp["active_seconds"],
            "load_seconds": [p["run"].get("load_seconds") for p in procs],
            "lap_ms": {f"process{p['process']}": p.get("laps") for p in procs},
            "hidden_bit_equal_efr_asset": [all(r["bit_equal_gpu_aot"] for r in p["parity"]["rows"]) if "parity" in p
                                           else None for p in procs],
            "max_abs_dp": [p["parity"]["summary"]["max_abs_dp"] if "parity" in p else None for p in procs],
            "first_lap_ms_per_new_length": lap1[0] if lap1 else None,
            "anec_compile_failed": [p["run"].get("ane_messages", {}).get("anec_compile_failed") for p in procs],
            "anec_errors": [p["run"].get("ane_messages", {}).get("anec_errors") for p in procs],
            "ane_region_ir_files": [sorted(k for k in ((p["run"].get("mpsgraph_scratch") or {}).get("files") or {})
                                           if "_ANE_region_" in k and k.endswith(".bc.mlir")) for p in procs],
            "scratch_kib": [(p["run"].get("mpsgraph_scratch") or {}).get("kib") for p in procs]}
    doc["lock_end"] = lock_state()
    doc["seconds"] = time.monotonic() - t0
    doc["status"] = "done"
    doc["script"] = {"path": "conversion/d1/compute_unit_probe.py", "sha256": sha256_file(Path(__file__).resolve())}
    doc["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    open_transcript(out, doc)
    print(json.dumps(doc.get("summary"), indent=1))
    return 0


def static_cmd(args) -> int:
    """The static form's AOT (no efr): does a new position's first call still pay a specialization? (RULES["static"])"""
    from parity_decoder_torch import load_oracle, score_probs
    from readout_gate import ORACLE, Bundle, readout_probs

    out = Path(args.transcript)
    if out.exists():
        raise SystemExit(f"{out} exists: transcripts are never overwritten")
    dec = Bundle(args.decoder)
    if not dec.contract.get("static"):
        raise SystemExit(f"{dec.name} is not a static-form bundle (metadata language.contract.static)")
    oracle_path = Path(args.oracle or ORACLE)
    _, recs = load_oracle(oracle_path)
    rows = []
    for rid, k, keep in STATIC_ROWS:
        q = recs[rid]["questions"][k]
        ids = q["row_ids"] if keep is None else q["row_ids"][:keep * dec.S]
        rows.append({"id": rid, "k": k, "ids": ids, "slot": q["slot"] if keep is None else len(ids) - 1,
                     "cut_to_calls": keep, "tokens_of_the_row": len(q["row_ids"])})
    asset = dec.aimodelc
    work = Path(args.work)
    doc = {"schema": "d1-compute-unit-probe/1", "probe": "static", "status": "running", "started": now(),
           "rules": RULES["static"], "decoder": {"bundle": str(dec.dir), "asset": str(asset), "contract": dec.contract},
           "rows": [{k: v for k, v in r.items() if k != "ids"} | {"tokens": len(r["ids"]),
                                                                  "calls": -(-len(r["ids"]) // dec.S)} for r in rows],
           "oracle": {"path": str(oracle_path), "sha256": sha256_file(oracle_path)}, "work": str(work),
           "tag": args.tag, "note": args.note, "gpu": "shared with other sessions: every ms is a contended reference "
                                                    "value, never the ladder's"}
    open_transcript(out, doc)
    doc["environment"] = env_record()
    doc["asset_tree"] = {k: v for k, v in asset_tree(asset).items() if k != "graph_strings"}
    doc["lock_start"] = lock_state()
    t0 = time.monotonic()
    table = dec.table()
    base = {"graph": "decoder", "S": dec.S, "max_ctx": dec.max_ctx, "hidden": dec.hidden, "pad_id": dec.pad_id,
            "contract": dec.contract, "aimodelc": str(asset), "options": "default",
            "laps": [{"label": "lap1", "rows": rows}, {"label": "lap2", "rows": rows}]}
    procs = []
    with EntryGuard(asset, f"static{args.tag}", own=True) as eg:
        for n in range(1, args.processes + 1):
            before = {"entry_kib": du_kib(eg.entry), "scratch_kib": du_kib(scratch_dir()), "free_bytes": free_bytes(),
                      "entries": entries()}
            rec = run_worker({**base, "out": str(work / f"static{args.tag}_process{n}")}, args.run_limit)
            after = {"entry_kib": du_kib(eg.entry), "scratch_kib": du_kib(scratch_dir()), "free_bytes": free_bytes(),
                     "entries": entries()}
            p = {"process": n, "before": {k: v for k, v in before.items() if k != "entries"},
                 "after": {k: v for k, v in after.items() if k != "entries"},
                 "new_entries": sorted(set(after["entries"]) - set(before["entries"])),
                 "run": {k: v for k, v in rec.items() if k not in ("rows", "calls")}}
            calls = rec.get("calls", [])
            if calls and not rec.get("error"):
                lap = {(c["lap"], c["row"], c["call"]): c for c in calls}
                pairs = [{"row": c["row"], "call": c["call"], "first_position": c["first_position"],
                          "lap1_ms": round(c["ms"], 3), "lap2_ms": round(lap[("lap2", c["row"], c["call"])]["ms"], 3)}
                         for c in calls if c["lap"] == "lap1" and ("lap2", c["row"], c["call"]) in lap]
                l1 = np.asarray([x["lap1_ms"] for x in pairs][1:])
                l2 = np.asarray([x["lap2_ms"] for x in pairs])
                p["calls"] = pairs
                p["first_call_of_process_ms"] = round(calls[0]["ms"], 3)
                p["lap1_after_first"] = {"median_ms": float(np.median(l1)), "max_ms": float(l1.max())} if l1.size else None
                p["lap2"] = {"median_ms": float(np.median(l2)), "max_ms": float(l2.max())} if l2.size else None
                p["lap1_over_lap2_max"] = float(max(x["lap1_ms"] / x["lap2_ms"] for x in pairs[1:])) if len(pairs) > 1 else None
                shas = {}
                for r in rec.get("rows", []):
                    shas.setdefault(f"{r['id']}:{r['k']}", {})[r["lap"]] = r["hidden_sha256"]
                p["hidden_sha256"] = shas
                p["lap_hidden_bit_equal"] = all(len(set(v.values())) == 1 for v in shas.values())
                z = np.load(rec["npz"])
                scored = []
                for r in rec.get("rows", []):
                    src = next(x for x in rows if (x["id"], x["k"]) == (r["id"], r["k"]))
                    if src["cut_to_calls"] is not None:
                        continue
                    q = recs[r["id"]]["questions"][r["k"]]
                    s = score_probs(readout_probs(z[f"{r['key']}__slot"], q["groups"], table), q)
                    scored.append({"lap": r["lap"], "row": f"{r['id']}:{r['k']}",
                                   **{k: s[k] for k in ("argmax_equal", "max_abs_dp", "near_tie")}})
                p["parity"] = scored
            procs.append(p)
            doc["processes"] = procs
            open_transcript(out, doc)
            if rec.get("error"):
                break
    doc["entry"] = eg.rec
    ok = [p for p in procs if "calls" in p]
    doc["summary"] = {
        "processes_run": len(ok), "processes_asked": args.processes,
        "first_call_of_process_ms": [p["first_call_of_process_ms"] for p in ok],
        "lap1_after_first_max_ms": [p["lap1_after_first"]["max_ms"] if p["lap1_after_first"] else None for p in ok],
        "lap2_median_ms": [p["lap2"]["median_ms"] if p["lap2"] else None for p in ok],
        "lap1_over_lap2_max": [p["lap1_over_lap2_max"] for p in ok],
        "lap_hidden_bit_equal": [p["lap_hidden_bit_equal"] for p in ok],
        "hidden_equal_across_processes": (len(ok) > 1 and all(p["hidden_sha256"] == ok[0]["hidden_sha256"]
                                                                for p in ok[1:])),
        "new_entries": [p["new_entries"] for p in procs],
        "entry_kib_after": [p["after"]["entry_kib"] for p in procs],
        "load_seconds": [p["run"].get("load_seconds") for p in procs],
        "errors": [p["run"].get("error") for p in procs if p["run"].get("error")],
        "parity": [x for p in ok for x in p.get("parity", [])],
        "anec_compile_failed": [p["run"].get("ane_messages", {}).get("anec_compile_failed") for p in procs],
        "scratch_kib": [(p["run"].get("mpsgraph_scratch") or {}).get("kib") for p in procs]}
    doc["lock_end"] = lock_state()
    doc["seconds"] = time.monotonic() - t0
    doc["status"] = "done"
    doc["script"] = {"path": "conversion/d1/compute_unit_probe.py", "sha256": sha256_file(Path(__file__).resolve())}
    doc["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    open_transcript(out, doc)
    print(json.dumps(doc["summary"], indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("static", help="the static form's AOT (no efr): the call walls of new positions, two processes")
    st.add_argument("--decoder", required=True, help="the static-form bundle (its AOT under <bundles>_aotc/)")
    st.add_argument("--oracle", help="records_oracle.json (default readout_gate.ORACLE)")
    st.add_argument("--processes", type=int, default=2)
    st.add_argument("--tag", default="")
    st.add_argument("--note")
    st.add_argument("--transcript", required=True)
    st.add_argument("--work", default=str(LANE / "readout" / "r9b_probe_static"))
    st.add_argument("--run-limit", type=float, default=RUN_LIMIT_S)
    a = sub.add_parser("ane", help="the decoder and the tower compiled with the Neural Engine preferred, loaded with it")
    a.add_argument("--tower", default=str(TOWER))
    a.add_argument("--tower-gate", default=str(TOWER_GATE))
    a.add_argument("--out-dir", default=str(LANE / "exports" / "probe_ane"))
    a.add_argument("--graphs", default="decoder,tower", help="which graphs to probe (comma list of decoder, tower)")
    a.add_argument("--no-control", action="store_true", help="skip the shipped GPU asset loaded with the preference")
    n = sub.add_parser("noefr", help="the decoder compiled without --expect-frequent-reshapes, loaded with default()")
    n.add_argument("--out-dir", default=str(LANE / "exports" / "probe_noefr"))
    n.add_argument("--options", default="default", choices=["default", "neural_engine"],
                   help="the runtime's specialization options for the load (neural_engine: the Neural Engine preferred)")
    n.add_argument("--asset", help="reuse an efr-less asset compiled before (no compile)")
    n.add_argument("--processes", type=int, default=2)
    for sp, wd in ((a, "probe_ane"), (n, "probe_noefr")):
        sp.add_argument("--tag", default="", help="a suffix for this run's labels, logs, worker files and entry names")
        sp.add_argument("--note", help="free text kept in the transcript")
        sp.add_argument("--transcript", required=True)
        sp.add_argument("--decoder", default=str(DECODER))
        sp.add_argument("--decoder-gate", default=str(DECODER_GATE))
        sp.add_argument("--work", default=str(LANE / "readout" / f"r6a_{wd}"), help="the workers' records and arrays")
        sp.add_argument("--compile-limit", type=float, default=COMPILE_LIMIT_S,
                        help="seconds a compile may run outside measurement windows before it is killed")
        sp.add_argument("--run-limit", type=float, default=RUN_LIMIT_S,
                        help="seconds a worker process may run outside measurement windows before it is killed")
    w = sub.add_parser("worker")
    w.add_argument("--spec", required=True)
    args = ap.parse_args()
    if args.cmd == "worker":
        return worker(Path(args.spec))
    if args.cmd == "static":
        return static_cmd(args)
    return ane_cmd(args) if args.cmd == "ane" else noefr_cmd(args)


if __name__ == "__main__":
    raise SystemExit(main())
