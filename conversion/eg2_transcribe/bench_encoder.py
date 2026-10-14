#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.9.0",
#     "coreai-core==1.0.0b2",
#     "safetensors>=0.7.0",
#     "numpy>=2.2",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
# Community port — NOT an Apple or Google model.
"""Time exported audio-encoder bundles on the Mac GPU / Neural Engine through the Core AI Python runtime.

    prepare  cut the long fixture's log-mel frames into the timing inputs, before any measurement window:
             20 windows of 3000 frames (rows 0..60000 = the opening 10 minutes) and 20 windows of 1000 frames
             (rows 0..20000), each with its half mask. Writes <out>/bench_inputs_T<Tb>.npz.
    run      one process, one bundle: load (wall clock), 3 warm-up calls, then 20 timed calls cycling through
             the 20 windows; prints one JSON line (ms per call median / p90, process CPU ms per call, x real time,
             the process's peak physical footprint). --cold deletes this bundle's own Core AI cache entry before the load.
             --backend gpu / ane = the h16c AOT bundle; gpu-jit / ane-jit = the source .aimodel specialized at
             load with that preferred compute unit. Form ``everpaper`` times the EverPaper v1.2 audio asset
             (EverPaper/embeddinggemma-2-coreai, Apache-2.0; its h16c AOT compiles in --everpaper-dir, function
             audio_<Tb>, float16 features + frame / half / token masks) for a comparison row.
             --front F --body B (backend gpu+ane) times the two-bundle form of split_encoder.py instead: the
             float32 front on the GPU (h16c AOT, default()), the host cast of its hidden to float16, the float16
             body on the Neural Engine (h16c AOT); per call it records front / cast / body / total ms. --pipeline
             adds a second pass in which the front of window i+1 is issued together with the body of window i
             (asyncio.gather), reported as ms per window.
    ab       the comparison driver: runs `run` for every --config in turn, --reps rounds (A/B/A/B ...), so
             the forms compared share one measurement window; writes <out>/<form>/timing_<backend>_T<Tb>.json
             per config and <out>/timing_windows/<label>.json (with --json-dir: <dir>/timing_<form>_<backend>_T<Tb>.json,
             <dir>/timing_split<s>_T<Tb>.json and <dir>/timing_windows/<label>.json). Config split<s>:<Tb>:gpu+ane
             times the pair eg2_front_<s>_fp32_T<Tb>_gpu_h16c / eg2_body_<s>_fp16_T<Tb>_ane_h16c in --split-dir.

x real time = real audio per call (min(Tb, 1000 or 3000 frames) * 10 ms) / wall seconds per call. Run `run` / `ab`
inside a measurement window; the calls are timed alone (inputs are built before the clock starts).
    <venv>/bin/python conversion/eg2_transcribe/bench_encoder.py prepare
    <venv>/bin/python conversion/eg2_transcribe/bench_encoder.py ab --label eg2stt-r3-fp32-gpu \\
        --config fp32:3000:gpu --config fp16w32:3000:gpu --reps 8
    <venv>/bin/python conversion/eg2_transcribe/bench_encoder.py ab --label eg2stt-r5-split30 --json-dir <split> \\
        --config splitlayer8:3000:gpu+ane --config fp16:3000:gpu --config fp16:3000:ane --reps 8
"""
from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import BUCKETS, exports_dir, fixtures_dir, fp16_io, write_json  # noqa: E402

N_WINDOWS, WARMUP, CALLS = 20, 3, 20
AOT_NAME = {"gpu": "gpu_h16c", "ane": "ane_h16c"}


def peak_footprint_bytes() -> dict:
    """phys_footprint and lifetime max phys_footprint of this process (proc_pid_rusage, RUSAGE_INFO_V4)."""
    buf = (ctypes.c_uint64 * 64)()
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    import os
    if libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)) != 0:
        return {}
    base = 2                                   # 16-byte uuid = 2 uint64
    return {"phys_footprint": int(buf[base + 7]), "lifetime_max_phys_footprint": int(buf[base + 28])}


def bundle_path(out: Path, form: str, bucket: int, backend: str, everpaper_dir: Path | None = None) -> Path:
    """This port's AOT bundle, or for form ``everpaper`` the h16c AOT compile of the EverPaper v1.2 audio asset
    (one multi-function bundle; the function is audio_<bucket>)."""
    if backend.endswith("-jit"):        # the source bundle, specialized by the runtime at load
        return out / form / f"eg2_audio_{form}_T{bucket}.aimodel"
    if form == "everpaper":
        return (everpaper_dir or out.parent / "everpaper_aot") / f"ep_audio_{AOT_NAME[backend]}.aimodelc"
    return out / form / f"eg2_audio_{form}_T{bucket}_{AOT_NAME[backend]}.aimodelc"


def cache_entry(bundle: Path) -> Path | None:
    """The Core AI cache entry this Python process creates for a bundle: <cache>/<OS build>/python/<main.hash>."""
    h = (bundle / "main.hash").read_bytes().hex()
    build = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    p = Path.home() / "Library" / "Caches" / "coreai-cache" / build / Path(sys.executable).name / h
    return p


def prepare(args) -> None:
    feats = np.load(fixtures_dir(args.fixtures) / "long-speckled-band" / "features.npy", mmap_mode="r")
    for bucket in BUCKETS:
        x = np.stack([np.asarray(feats[i * bucket:(i + 1) * bucket], dtype=np.float32) for i in range(N_WINDOWS)])
        m = np.ones((N_WINDOWS, (bucket + 1) // 2), dtype=np.float32)
        out = exports_dir(args.out_dir) / f"bench_inputs_T{bucket}.npz"
        np.savez(out, features=x, half_mask=m)
        print(f"wrote {out} {x.shape}")


async def run_one(args) -> dict:
    import coreai.runtime as rt

    out = exports_dir(args.out_dir)
    ep_dir = Path(args.everpaper_dir).expanduser() if args.everpaper_dir else None
    bundle = bundle_path(out, args.form, args.bucket, args.backend, ep_dir)
    entry = cache_entry(bundle)
    cold = False
    if args.cold and entry is not None and entry.exists():
        shutil.rmtree(entry)
        cold = True
    everpaper = args.form == "everpaper"
    data = np.load(out / f"bench_inputs_T{3000 if args.bucket == 3000 else 1000}.npz")
    dtype = np.float16 if fp16_io(args.form) else np.float32

    def window(i: int) -> dict:
        x = np.zeros((1, args.bucket, data["features"].shape[-1]), dtype=dtype)
        n = min(args.bucket, data["features"].shape[1])
        x[0, :n] = data["features"][i][:n]                 # EverPaper's 1024 bucket: 1000 frames + 24 zero rows
        d = {"features": rt.NDArray(np.ascontiguousarray(x))}
        hm = np.zeros((1, (args.bucket + 1) // 2), dtype=dtype)
        hm[0, : (n + 1) // 2] = 1
        d["half_mask"] = rt.NDArray(np.ascontiguousarray(hm))
        if everpaper:
            fm = np.zeros((1, args.bucket), dtype=dtype)
            fm[0, :n] = 1
            tm = np.zeros((1, args.bucket // 4), dtype=dtype)
            tm[0, : ((n + 1) // 2 + 1) // 2] = 1
            d["frame_mask"] = rt.NDArray(np.ascontiguousarray(fm))
            d["token_mask"] = rt.NDArray(np.ascontiguousarray(tm))
        return d

    inputs = [window(i) for i in range(N_WINDOWS)]
    opts = {"gpu": rt.SpecializationOptions.default(),
            "gpu-jit": rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())}.get(
        args.backend, rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.neural_engine()))
    fp0 = peak_footprint_bytes()
    cpu0 = time.process_time()
    t0 = time.perf_counter()
    model = await rt.AIModel.load(bundle, opts)
    fn = model.load_function(f"audio_{args.bucket}" if everpaper else "main")
    load_s = time.perf_counter() - t0
    cpu_load_s = time.process_time() - cpu0
    fp_load = peak_footprint_bytes()
    warm = []
    for i in range(WARMUP):
        t1 = time.perf_counter()
        await fn(inputs[i % N_WINDOWS])
        warm.append(time.perf_counter() - t1)
    secs = []
    cpu1 = time.process_time()
    for i in range(CALLS):
        t1 = time.perf_counter()
        res = await fn(inputs[i % N_WINDOWS])
        secs.append(time.perf_counter() - t1)
    cpu_per_call_ms = (time.process_time() - cpu1) / CALLS * 1000
    _ = res["soft_tokens"].numpy()
    pipe, pipe_error = None, None
    if args.pipeline:
        # two windows in flight (asyncio.gather), the same depth as the two-bundle pipelined pass
        try:
            p0 = time.perf_counter()
            for i in range(0, CALLS, 2):
                await asyncio.gather(fn(inputs[i % N_WINDOWS]), fn(inputs[(i + 1) % N_WINDOWS]))
            pipe = (time.perf_counter() - p0) / CALLS * 1000
        except Exception as exc:  # noqa: BLE001 - recorded; the sequential timing above stands
            pipe_error = repr(exc)[:300]
    fp_end = peak_footprint_bytes()
    ms = sorted(s * 1000 for s in secs)
    med = statistics.median(ms)
    audio_s = min(args.bucket, data["features"].shape[1]) * 0.01   # real audio per call (10 ms per frame)
    return {"form": args.form, "bucket_frames": args.bucket, "backend": args.backend, "bundle": str(bundle),
            "pipelined_ms_per_window": round(pipe, 3) if pipe is not None else None, "pipelined_error": pipe_error,
            "function": f"audio_{args.bucket}" if everpaper else "main",
            "cold_requested": args.cold, "cache_entry_deleted": cold, "load_s": round(load_s, 3),
            "load_cpu_s": round(cpu_load_s, 3),
            "warmup_ms": [round(w * 1000, 2) for w in warm], "ms": [round(s * 1000, 3) for s in secs],
            "ms_median": round(med, 3), "ms_p90": round(float(np.percentile(ms, 90)), 3),
            "cpu_ms_per_call": round(cpu_per_call_ms, 3),
            "x_realtime": round(audio_s / (med / 1000), 1), "audio_s_per_call": audio_s,
            "footprint_before_load": fp0, "footprint_after_load": fp_load, "footprint_end": fp_end,
            "t_start": time.strftime("%H:%M:%S")}


def split_pair(split_dir: Path, form: str, bucket: int | str) -> tuple[Path, Path]:
    """split_encoder.py's AOT pair for form split<s>: (float32 GPU front, float16 Neural Engine body)."""
    s = form.removeprefix("split")
    return (split_dir / f"eg2_front_{s}_fp32_T{bucket}_gpu_h16c.aimodelc",
            split_dir / f"eg2_body_{s}_fp16_T{bucket}_ane_h16c.aimodelc")


async def run_split_one(args) -> dict:
    """The two-bundle form: float32 front (GPU AOT, default()) -> host cast to float16 -> float16 body (ANE AOT)."""
    import coreai.runtime as rt

    out = exports_dir(args.out_dir)
    front_b, body_b = Path(args.front).expanduser().resolve(), Path(args.body).expanduser().resolve()
    data = np.load(out / f"bench_inputs_T{3000 if args.bucket == 3000 else 1000}.npz")
    n = min(args.bucket, data["features"].shape[1])

    def window(i: int) -> dict:
        x = np.zeros((1, args.bucket, data["features"].shape[-1]), dtype=np.float32)
        x[0, :n] = data["features"][i][:n]
        hm = np.zeros((1, (args.bucket + 1) // 2), dtype=np.float32)
        hm[0, : (n + 1) // 2] = 1
        return {"features": rt.NDArray(np.ascontiguousarray(x)), "half_mask": rt.NDArray(hm)}

    inputs = [window(i) for i in range(N_WINDOWS)]
    fp0 = peak_footprint_bytes()
    cpu0 = time.process_time()
    t0 = time.perf_counter()
    front = (await rt.AIModel.load(front_b, rt.SpecializationOptions.default())).load_function("main")
    t1 = time.perf_counter()
    body = (await rt.AIModel.load(body_b, rt.SpecializationOptions.from_preferred_compute_unit_kind(
        rt.ComputeUnitKind.neural_engine()))).load_function("main")
    t2 = time.perf_counter()
    cpu_load_s = time.process_time() - cpu0
    fp_load = peak_footprint_bytes()

    def to_body(res) -> dict:
        return {"hidden": rt.NDArray(np.ascontiguousarray(res["hidden"].numpy().astype(np.float16)))}

    async def call(inp: dict) -> tuple:
        a = time.perf_counter()
        r = await front(inp)
        b = time.perf_counter()
        h = to_body(r)
        c = time.perf_counter()
        res = await body(h)
        d = time.perf_counter()
        return res, (b - a, c - b, d - c, d - a)

    warm = []
    for i in range(WARMUP):
        warm.append((await call(inputs[i % N_WINDOWS]))[1][3])
    parts = []
    cpu1 = time.process_time()
    for i in range(CALLS):
        res, p = await call(inputs[i % N_WINDOWS])
        parts.append(p)
    cpu_per_call_ms = (time.process_time() - cpu1) / CALLS * 1000
    _ = res["soft_tokens"].numpy()
    pipe, pipe_error = None, None
    if args.pipeline:
        # front(i + 1) issued together with body(i): the GPU and the Neural Engine work on different windows
        try:
            p0 = time.perf_counter()
            h = to_body(await front(inputs[0]))
            for i in range(1, CALLS):
                r, _ = await asyncio.gather(front(inputs[i % N_WINDOWS]), body(h))
                h = to_body(r)
            await body(h)
            pipe = (time.perf_counter() - p0) / CALLS * 1000
        except Exception as exc:  # noqa: BLE001 - recorded; the sequential timing above stands
            pipe_error = repr(exc)[:300]
    fp_end = peak_footprint_bytes()
    total = sorted(p[3] * 1000 for p in parts)
    med = statistics.median(total)
    audio_s = n * 0.01
    rec = {"form": args.form, "bucket_frames": args.bucket, "backend": "gpu+ane",
           "front_bundle": str(front_b), "body_bundle": str(body_b), "function": "main",
           "load_s": round(t2 - t0, 3), "load_s_front": round(t1 - t0, 3), "load_s_body": round(t2 - t1, 3),
           "load_cpu_s": round(cpu_load_s, 3), "warmup_ms": [round(w * 1000, 2) for w in warm],
           "ms": [round(p[3] * 1000, 3) for p in parts],
           "front_ms": [round(p[0] * 1000, 3) for p in parts], "cast_ms": [round(p[1] * 1000, 3) for p in parts],
           "body_ms": [round(p[2] * 1000, 3) for p in parts],
           "ms_median": round(med, 3), "ms_p90": round(float(np.percentile(total, 90)), 3),
           "front_ms_median": round(statistics.median(p[0] * 1000 for p in parts), 3),
           "cast_ms_median": round(statistics.median(p[1] * 1000 for p in parts), 3),
           "body_ms_median": round(statistics.median(p[2] * 1000 for p in parts), 3),
           "pipelined_ms_per_window": round(pipe, 3) if pipe is not None else None, "pipelined_error": pipe_error,
           "cpu_ms_per_call": round(cpu_per_call_ms, 3),
           "x_realtime": round(audio_s / (med / 1000), 1), "audio_s_per_call": audio_s,
           "footprint_before_load": fp0, "footprint_after_load": fp_load, "footprint_end": fp_end,
           "t_start": time.strftime("%H:%M:%S")}
    return rec


def ab(args) -> None:
    out = exports_dir(args.out_dir)
    split_dir = Path(args.split_dir).expanduser() if args.split_dir else out / "split"
    configs = [tuple(c.split(":")) for c in args.config]
    runs: dict[str, list[dict]] = {c: [] for c in args.config}
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for rep in range(args.reps):
        for spec, (form, bucket, backend) in zip(args.config, configs):
            pair = ["--pipeline"] if args.pipeline else []
            if form.startswith("split"):
                f, b = split_pair(split_dir, form, bucket)
                pair += ["--front", str(f), "--body", str(b)]
            cmd = [sys.executable, str(Path(__file__).resolve()), "run", "--form", form, "--bucket", bucket,
                   "--backend", backend] + (["--out-dir", args.out_dir] if args.out_dir else []) + \
                  (["--everpaper-dir", args.everpaper_dir] if args.everpaper_dir else []) + pair + \
                  (["--cold"] if (rep == 0 and args.cold_first) else [])
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            line = [s for s in r.stdout.splitlines() if s.startswith("{")]
            if r.returncode != 0 or not line:
                rec = {"error": (r.stderr or r.stdout).strip().splitlines()[-5:], "rep": rep}
            else:
                rec = json.loads(line[-1])
                rec["rep"] = rep
            runs[spec].append(rec)
            print(f"rep {rep} {spec}: " + (f"load {rec['load_s']} s, median {rec['ms_median']} ms, p90 "
                                           f"{rec['ms_p90']} ms, x{rec['x_realtime']}" if "ms_median" in rec
                                           else f"ERROR {rec['error']}"), flush=True)
    finished = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    window = {"label": args.label, "started": started, "finished": finished, "reps": args.reps,
              "order": args.config, "warmup": WARMUP, "calls_per_rep": CALLS, "configs": {}}
    for spec, (form, bucket, backend) in zip(args.config, configs):
        ok = [r for r in runs[spec] if "ms_median" in r]
        meds = [r["ms_median"] for r in ok]
        allms = sorted(m for r in ok for m in r["ms"])
        loads = [r["load_s"] for r in ok]
        audio_s = ok[0]["audio_s_per_call"] if ok else None
        summ = {
            "reps_ok": len(ok), "reps_failed": len(runs[spec]) - len(ok),
            "audio_s_per_call": audio_s,
            "ms_median_of_rep_medians": round(statistics.median(meds), 3) if meds else None,
            "ms_rep_median_range": [min(meds), max(meds)] if meds else None,
            "ms_p90_all_calls": round(float(np.percentile(allms, 90)), 3) if allms else None,
            "x_realtime": round(audio_s / (statistics.median(meds) / 1000), 1) if meds else None,
            "cpu_ms_per_call_median": round(statistics.median([r["cpu_ms_per_call"] for r in ok]), 3) if ok else None,
            "load_s_first_process": loads[0] if loads else None,
            "load_s_first_process_cold": bool(ok and ok[0].get("cache_entry_deleted")),
            "load_s_warm_median": round(statistics.median(loads[1:]), 3) if len(loads) > 1 else None,
            "peak_footprint_gb_max": round(max(r["footprint_end"].get("lifetime_max_phys_footprint", 0)
                                               for r in ok) / 1e9, 3) if ok else None,
        }
        pipes = [r["pipelined_ms_per_window"] for r in ok if r.get("pipelined_ms_per_window") is not None]
        if pipes:
            summ["pipelined_ms_per_window_median"] = round(statistics.median(pipes), 3)
            summ["pipelined_ms_per_window_range"] = [min(pipes), max(pipes)]
            summ["pipelined_x_realtime"] = round(audio_s / (summ["pipelined_ms_per_window_median"] / 1000), 1)
        if form.startswith("split") and ok:
            for part in ("front", "cast", "body"):
                summ[f"{part}_ms_median_of_rep_medians"] = round(statistics.median(
                    r[f"{part}_ms_median"] for r in ok), 3)
            summ["load_s_body_median"] = round(statistics.median(r["load_s_body"] for r in ok), 3)
        window["configs"][spec] = summ
        rec = {"form": form, "bucket_frames": int(bucket), "backend": backend, "window_label": args.label,
               "window_started": started, "window_finished": finished, "compared_in_window_with": args.config,
               "summary": summ, "reps": runs[spec]}
        if args.json_dir:
            jd = Path(args.json_dir).expanduser()
            dest = jd / (f"timing_{form}_T{bucket}.json" if form.startswith("split")
                         else f"timing_{form}_{backend}_T{bucket}.json")
            if dest.exists() and not args.overwrite:
                dest = dest.with_name(f"{dest.stem}_{args.label}.json")
        else:
            dest = out / form / f"timing_{backend}_T{bucket}.json"
            if dest.exists() and not args.overwrite:
                dest = out / form / f"timing_{backend}_T{bucket}_{args.label}.json"
        write_json(dest, rec)
        extra = (f" [front {summ['front_ms_median_of_rep_medians']} + cast {summ['cast_ms_median_of_rep_medians']} + "
                 f"body {summ['body_ms_median_of_rep_medians']} ms]" if "front_ms_median_of_rep_medians" in summ else "")
        if "pipelined_ms_per_window_median" in summ:
            extra += f" [2 in flight: {summ['pipelined_ms_per_window_median']} ms/window, x{summ['pipelined_x_realtime']}]"
        print(f"{spec}: median {summ['ms_median_of_rep_medians']} ms (reps {summ['ms_rep_median_range']}), "
              f"p90 {summ['ms_p90_all_calls']} ms, x{summ['x_realtime']}, load at process start {summ['load_s_first_process']} s"
              f" (cold {summ['load_s_first_process_cold']}) warm {summ['load_s_warm_median']} s, "
              f"peak footprint {summ['peak_footprint_gb_max']} GB{extra} -> {dest}", flush=True)
    wdir = Path(args.json_dir).expanduser() if args.json_dir else out
    write_json(wdir / "timing_windows" / f"{args.label}.json", window)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--fixtures")
    p.add_argument("--out-dir")
    r = sub.add_parser("run")
    r.add_argument("--form", required=True)
    r.add_argument("--bucket", type=int, required=True)
    r.add_argument("--backend", choices=("gpu", "ane", "gpu-jit", "ane-jit", "gpu+ane"), required=True,
                   help="gpu / ane = the h16c AOT bundle; *-jit = the source .aimodel specialized at load; "
                        "gpu+ane = the two-bundle form (--front / --body)")
    r.add_argument("--cold", action="store_true")
    r.add_argument("--out-dir")
    r.add_argument("--everpaper-dir", help="dir of the EverPaper h16c AOT bundles ep_audio_{gpu,ane}_h16c.aimodelc")
    r.add_argument("--front", help="two-bundle form: the float32 front, GPU AOT (split_encoder.py)")
    r.add_argument("--body", help="two-bundle form: the float16 body, Neural Engine AOT (split_encoder.py)")
    r.add_argument("--pipeline", action="store_true",
                   help="also time two windows in flight: two-bundle front(i+1) || body(i); one bundle, two calls "
                        "issued together")
    a = sub.add_parser("ab")
    a.add_argument("--config", action="append", required=True,
                   help="form:bucket:backend, e.g. fp32:3000:gpu; form `everpaper` times the EverPaper asset; "
                        "split<s>:<Tb>:gpu+ane the two-bundle form cut after stage s")
    a.add_argument("--reps", type=int, default=8)
    a.add_argument("--label", required=True)
    a.add_argument("--cold-first", action="store_true", help="delete each bundle's cache entry before rep 0")
    a.add_argument("--overwrite", action="store_true")
    a.add_argument("--out-dir")
    a.add_argument("--everpaper-dir")
    a.add_argument("--split-dir", help="split_encoder.py bundles (default: <out>/split)")
    a.add_argument("--json-dir", help="write every timing json here instead of <out>/<form>/")
    a.add_argument("--pipeline", action="store_true", help="every config: also time two windows in flight")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "run":
        if (args.backend == "gpu+ane") != bool(args.front and args.body):
            raise SystemExit("--backend gpu+ane goes with --front and --body")
        print(json.dumps(asyncio.run(run_split_one(args) if args.front else run_one(args))))
    else:
        ab(args)


if __name__ == "__main__":
    main()
