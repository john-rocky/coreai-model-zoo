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
"""Gate an exported audio-encoder bundle on the Core AI Python runtime, on all 20 fixture clips.

Backends (``--backend``), each with the bundle it is meant for:
    cpu  the JIT ``.aimodel`` with ``SpecializationOptions.cpu_only()`` — the correctness reference
    gpu  the h16c GPU AOT ``.aimodelc`` with ``SpecializationOptions.default()`` (the Python GPU JIT path is not
         used: on macOS 26A428 it returned wrong values for other graphs)
    ane  the h16c Neural Engine AOT ``.aimodelc`` with the Neural Engine as preferred compute unit; the ANE
         regions of the bundle are counted by unique name (0 = the graph silently runs on the GPU)

Each clip's log-mel frames are right-zero-padded to the bucket (or cut to its leading Tb frames, whose leading
Tb/4 tokens are the clip's own; see ``_common.window_input``), ``half_mask`` marks the leading ceil(n/2) rows,
and the leading valid rows of ``soft_tokens`` are compared per token with the fixture reference
(transformers 5.19.0 fp32 sdpa). PASS = on every clip, per-token cosine mean >= 0.999 and min >= 0.99, and no
non-finite value. Every clip runs twice; the drift between the runs is reported (expected <= 1e-6).
A red arm proves the gate can fail: ``half_mask`` all ones on the three clips with n_frames % 4 == 2 must
break their last token.

``--contract everpaper`` gates the EverPaper v1.2 audio asset (EverPaper/embeddinggemma-2-coreai, Apache-2.0)
the same way: function ``audio_<Tb>``, float16 features + frame / half / token masks (all ones in the red arm).

``--front F --body B`` gates the two-bundle form of split_encoder.py instead of one bundle: the float32 front
(h16c GPU AOT, ``default()``) maps features + half_mask to the hidden after the cut, the host casts it to float16,
and the float16 body (h16c AOT on ``--backend``, normally ane) maps it to the soft tokens. Same clips, bar, drift
and red arm; the json adds the body's (and front's) ANE regions and per-part call times. With ``--ref-hidden``
(the npz of ``split_encoder.py reference`` for this bucket, all 20 clips) it also reports the front alone (its
hidden vs torch float32) and the body alone (torch float32 hidden, cast to float16, through the body). With
``--pipeline`` it also runs the clips in the two-in-flight order bench_encoder.py --pipeline times (front of clip
i+1 issued together with the body of clip i) and compares those soft tokens with the sequential pass.

Writes ``<out>/<form>/gate_<backend>_T<Tb>.json`` (two-bundle: ``<body dir>/gate_split<s>_T<Tb>.json``). Run it
inside a measurement window (it uses the GPU / ANE):
    <venv>/bin/python conversion/eg2_transcribe/gate_encoder.py --bundle <x>.aimodelc --backend gpu
    <venv>/bin/python conversion/eg2_transcribe/gate_encoder.py --front <split>/eg2_front_layer8_fp32_T3000_gpu_h16c.aimodelc \\
        --body <split>/eg2_body_layer8_fp16_T3000_ane_h16c.aimodelc --backend ane
"""
from __future__ import annotations

import argparse
import asyncio
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (BAR_COS_MEAN, BAR_COS_MIN, compared_tokens, fixtures_dir, fp16_io, load_clips,  # noqa: E402
                     token_stats, window_input, write_json)

import coreai.runtime as rt  # noqa: E402

SPLIT_RE = re.compile(r"eg2_(front|body)_([a-z0-9]+)_(fp16|fp32)_T(\d+)")
SPECIALIZATION = {"cpu": "cpu_only()", "gpu": "default()", "ane": "from_preferred_compute_unit_kind(neural_engine)"}


def half_mask(n_valid: int, bucket: int, dtype) -> np.ndarray:
    m = np.zeros((1, (bucket + 1) // 2), dtype=dtype)
    m[0, : (n_valid + 1) // 2] = 1
    return m


def make_inputs(contract: str, x: np.ndarray, n_valid: int, bucket: int, dtype, ones_mask: bool = False) -> dict:
    """This port: features + half_mask. ``everpaper``: features + frame_mask + half_mask + token_mask (float16)."""
    n = bucket if ones_mask else n_valid
    inputs = {"features": rt.NDArray(np.ascontiguousarray(x[None].astype(dtype))),
              "half_mask": rt.NDArray(np.ascontiguousarray(half_mask(n, bucket, dtype)))}
    if contract == "everpaper":
        fm = np.zeros((1, bucket), dtype=dtype)
        fm[0, :n] = 1
        tm = np.zeros((1, bucket // 4), dtype=dtype)
        tm[0, : ((n + 1) // 2 + 1) // 2] = 1
        inputs["frame_mask"] = rt.NDArray(np.ascontiguousarray(fm))
        inputs["token_mask"] = rt.NDArray(np.ascontiguousarray(tm))
    return inputs


def ane_regions(bundle: Path) -> list[str]:
    names = set()
    for p in bundle.rglob("*"):
        m = re.search(r"([A-Za-z0-9_.-]*_ANE_region_\d+)", p.name)
        if m:
            names.add(m.group(1))
    return sorted(names)


def options(backend: str):
    if backend == "cpu":
        return rt.SpecializationOptions.cpu_only()
    if backend == "gpu":
        return rt.SpecializationOptions.default()
    return rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.neural_engine())


def parse_name(bundle: Path) -> tuple[str, int]:
    m = re.match(r"eg2_audio_([a-z0-9]+)_T(\d+)", bundle.name)
    if not m:
        raise SystemExit(f"cannot read form and bucket from {bundle.name}; pass --form and --bucket")
    return m.group(1), int(m.group(2))


async def run_all(fn, clips, bucket: int, dtype, contract: str,
                  ones_mask: bool = False) -> tuple[list[np.ndarray], list[float]]:
    outs, secs = [], []
    for c in clips:
        x, nv = window_input(c["features"], bucket)
        inputs = make_inputs(contract, x, nv, bucket, dtype, ones_mask)
        t0 = time.perf_counter()
        res = await asyncio.wait_for(fn(inputs), timeout=900)
        secs.append(time.perf_counter() - t0)
        outs.append(res["soft_tokens"].numpy().astype(np.float32)[0])
    return outs, secs


async def run_split(front, body, clips, bucket: int, ones_mask: bool = False):
    """front (float32) -> host cast to float16 -> body. Returns soft tokens, hiddens, (front, cast, body) seconds."""
    outs, hiddens, secs = [], [], []
    for c in clips:
        x, nv = window_input(c["features"], bucket)
        inputs = make_inputs("eg2stt", x, nv, bucket, np.float32, ones_mask)
        t0 = time.perf_counter()
        h = (await asyncio.wait_for(front(inputs), timeout=900))["hidden"].numpy()
        t1 = time.perf_counter()
        hb = rt.NDArray(np.ascontiguousarray(h.astype(np.float16)))
        t2 = time.perf_counter()
        res = await asyncio.wait_for(body({"hidden": hb}), timeout=900)
        outs.append(res["soft_tokens"].numpy().astype(np.float32)[0])
        secs.append((t1 - t0, t2 - t1, time.perf_counter() - t2))
        hiddens.append(h[0].astype(np.float32))
    return outs, hiddens, secs


async def run_split_pipelined(front, body, clips, bucket: int) -> list[np.ndarray]:
    """The same pair with two windows in flight: front(clip i + 1) issued together with body(clip i)."""
    def feed(c):
        x, nv = window_input(c["features"], bucket)
        return make_inputs("eg2stt", x, nv, bucket, np.float32)

    def to_body(res):
        return {"hidden": rt.NDArray(np.ascontiguousarray(res["hidden"].numpy().astype(np.float16)))}

    outs = []
    h = to_body(await front(feed(clips[0])))
    for c in clips[1:]:
        r, y = await asyncio.gather(front(feed(c)), body(h))
        outs.append(y["soft_tokens"].numpy().astype(np.float32)[0])
        h = to_body(r)
    outs.append((await body(h))["soft_tokens"].numpy().astype(np.float32)[0])
    return outs


def clip_rows(clips, outs1, outs2, secs1, bucket: int) -> list[dict]:
    rows = []
    for c, o1, o2, s in zip(clips, outs1, outs2, secs1):
        k = compared_tokens(c, bucket)
        st = token_stats(o1[:k], c["ref"][:k])
        st["drift_max_abs"] = float(np.abs(o1[:k].astype(np.float64) - o2[:k]).max())
        st["pass"] = st["cos_mean"] >= BAR_COS_MEAN and st["cos_min"] >= BAR_COS_MIN and st["nonfinite"] == 0
        st.update({"id": c["id"], "n_frames": c["n_frames"], "cut": c["n_frames"] > bucket, "call_s": round(s, 4)})
        rows.append(st)
        print(f"{c['id']}: k={k} cos mean {st['cos_mean']:.7f} min {st['cos_min']:.7f} max|d| {st['max_abs']:.3e} "
              f"drift {st['drift_max_abs']:.1e} {'PASS' if st['pass'] else 'FAIL'}", flush=True)
    return rows


def red_rows(red_clips, red_outs, bucket: int) -> list[dict]:
    red = []
    for c, o in zip(red_clips, red_outs):
        k = compared_tokens(c, bucket)
        st = token_stats(o[:k], c["ref"][:k])
        red.append({"id": c["id"], "n_frames": c["n_frames"], "cos_min": st["cos_min"],
                    "worst_pos": st["worst5"][0]["pos"], "last_pos": k - 1,
                    "fails_bar": st["cos_min"] < BAR_COS_MIN})
    return red


def summarize(rows: list[dict], red: list[dict]) -> dict:
    all_pos = sorted(((w["cos"], r["id"], w["pos"]) for r in rows for w in r["worst5"]))[:5]
    return {
        "pass": all(r["pass"] for r in rows),
        "clips": len(rows), "tokens": int(sum(r["n"] for r in rows)),
        "cos_mean_of_means": float(np.mean([r["cos_mean"] for r in rows])),
        "cos_mean_worst_clip": min(r["cos_mean"] for r in rows),
        "cos_min": min(r["cos_min"] for r in rows),
        "cos_min_clip": min(rows, key=lambda r: r["cos_min"])["id"],
        "worst5_positions": [{"cos": a, "id": b, "pos": p} for a, b, p in all_pos],
        "clips_failing": [r["id"] for r in rows if not r["pass"]],
        "max_abs": max(r["max_abs"] for r in rows),
        "nonfinite": int(sum(r["nonfinite"] for r in rows)),
        "drift_max_abs": max(r["drift_max_abs"] for r in rows),
        "drift_ok": max(r["drift_max_abs"] for r in rows) <= 1e-6,
        "red_arm_all_fail": bool(red) and all(r["fails_bar"] for r in red),
    }


def runtime_info() -> dict:
    try:
        from importlib.metadata import version
        core = version("coreai-core")
    except Exception:  # noqa: BLE001
        core = None
    build = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    return {"coreai-core": core, "macos_build": build, "python": platform.python_version()}


def print_summary(name: str, s: dict, regions: int, load_s: float, out: Path) -> None:
    print(f"{name}: {'PASS' if s['pass'] else 'FAIL'} cos mean {s['cos_mean_of_means']:.7f} "
          f"(worst clip {s['cos_mean_worst_clip']:.7f}) min {s['cos_min']:.7f} max|d| {s['max_abs']:.3e} "
          f"drift {s['drift_max_abs']:.1e} red-arm-fails {s['red_arm_all_fail']} ANE regions {regions} "
          f"load {load_s:.1f} s -> {out}")


async def main_split(args) -> None:
    front_b, body_b = (Path(p).expanduser().resolve() for p in (args.front, args.body))
    mf, mb = SPLIT_RE.match(front_b.name), SPLIT_RE.match(body_b.name)
    if not (mf and mb and mf.group(1) == "front" and mb.group(1) == "body"):
        raise SystemExit("--front / --body take split_encoder.py bundles eg2_front_<s>_fp32_T<Tb>_gpu_<arch>.aimodelc "
                         "and eg2_body_<s>_fp16_T<Tb>_ane_<arch>.aimodelc")
    stage, bucket = mb.group(2), int(mb.group(4))
    if (mf.group(2), int(mf.group(4))) != (stage, bucket):
        raise SystemExit(f"{front_b.name} and {body_b.name} are not cut at the same stage and bucket")
    if front_b.suffix != ".aimodelc" or body_b.suffix != ".aimodelc":
        raise SystemExit("the two-bundle gate takes the AOT .aimodelc bundles")
    clips = load_clips(fixtures_dir(args.fixtures))
    front_regions, body_regions = ane_regions(front_b), ane_regions(body_b)

    t0 = time.perf_counter()
    front = (await rt.AIModel.load(front_b, options("gpu"))).load_function("main")
    t1 = time.perf_counter()
    body = (await rt.AIModel.load(body_b, options(args.backend))).load_function("main")
    t2 = time.perf_counter()
    outs1, hid1, secs1 = await run_split(front, body, clips, bucket)
    outs2, _, secs2 = await run_split(front, body, clips, bucket)
    rows = clip_rows(clips, outs1, outs2, [sum(s) for s in secs1], bucket)
    red_clips = [c for c in clips if c["n_frames"] % 4 == 2 and c["n_frames"] <= bucket]
    red_outs, _, _ = await run_split(front, body, red_clips, bucket, ones_mask=True)
    red = red_rows(red_clips, red_outs, bucket)
    summary = summarize(rows, red)
    pipelined = None
    if args.pipeline:
        # the order bench_encoder.py --pipeline times: must give the sequential pass's soft tokens
        pouts = await run_split_pipelined(front, body, clips, bucket)
        prow = []
        for c, o, p in zip(clips, outs1, pouts):
            k = compared_tokens(c, bucket)
            st = token_stats(p[:k], c["ref"][:k])
            prow.append({"id": c["id"], "cos_mean": st["cos_mean"], "cos_min": st["cos_min"],
                         "max_abs_vs_sequential": float(np.abs(p[:k].astype(np.float64) - o[:k]).max())})
        pipelined = {"what": "two windows in flight (front of clip i+1 together with body of clip i) vs the "
                             "sequential pass 1", "cos_min": min(r["cos_min"] for r in prow),
                     "cos_mean_worst_clip": min(r["cos_mean"] for r in prow),
                     "max_abs_vs_sequential": max(r["max_abs_vs_sequential"] for r in prow),
                     "pass": all(r["cos_mean"] >= BAR_COS_MEAN and r["cos_min"] >= BAR_COS_MIN for r in prow),
                     "clips": prow}
        print(f"pipelined: cos min {pipelined['cos_min']:.7f}, max|d| vs sequential "
              f"{pipelined['max_abs_vs_sequential']:.1e}", flush=True)

    diag = None
    if args.ref_hidden:
        # front alone: its hidden vs torch float32; body alone: torch float32 hidden (valid rows, zero rows after —
        # every op after the cut is causal or per token) cast to float16 through the body, vs the fixture
        ref = np.load(args.ref_hidden)
        t4 = ((bucket + 1) // 2 + 1) // 2
        f_rows, b_rows = [], []
        for c, h in zip(clips, hid1):
            k = compared_tokens(c, bucket)
            r = ref[f"{c['id']}/{stage}/fp32"]
            assert r.shape[0] == k, (c["id"], r.shape, k)
            st = token_stats(h[:k], r)
            f_rows.append({"id": c["id"], "cos_mean": st["cos_mean"], "cos_min": st["cos_min"],
                           "max_abs": st["max_abs"], "ref_max_abs": st["ref_max_abs"]})
            hb = np.zeros((1, t4, r.shape[1]), dtype=np.float16)
            hb[0, :k] = r.astype(np.float16)
            y = (await body({"hidden": rt.NDArray(hb)}))["soft_tokens"].numpy().astype(np.float32)[0]
            st = token_stats(y[:k], c["ref"][:k])
            b_rows.append({"id": c["id"], "cos_mean": st["cos_mean"], "cos_min": st["cos_min"],
                           "worst": st["worst5"][0], "max_abs": st["max_abs"]})
        diag = {"ref_hidden": str(args.ref_hidden),
                "front_alone": {"what": "front hidden (GPU, float32) vs torch float32 hidden after " + stage,
                                "cos_min": min(r["cos_min"] for r in f_rows),
                                "max_abs": max(r["max_abs"] for r in f_rows), "clips": f_rows},
                "body_alone": {"what": "torch float32 hidden after " + stage + ", cast to float16, through the body; "
                                       "soft tokens vs the fixture",
                               "cos_min": min(r["cos_min"] for r in b_rows),
                               "cos_mean_worst_clip": min(r["cos_mean"] for r in b_rows),
                               "pass": all(r["cos_mean"] >= BAR_COS_MEAN and r["cos_min"] >= BAR_COS_MIN
                                           for r in b_rows), "clips": b_rows}}
        print(f"front alone cos min {diag['front_alone']['cos_min']:.9f} max|d| {diag['front_alone']['max_abs']:.3e}; "
              f"body alone cos min {diag['body_alone']['cos_min']:.7f}", flush=True)

    med = {name: round(float(np.median([s[i] for s in secs2])) * 1000, 3)
           for i, name in enumerate(("front", "cast", "body"))}
    med["total"] = round(float(np.median([sum(s) for s in secs2])) * 1000, 3)
    result = {
        "form": f"split{stage}", "contract": "eg2stt two-bundle (features + half_mask -> front -> fp16 cast -> body)",
        "stage": stage, "bucket_frames": bucket,
        "front": {"bundle": str(front_b), "backend": "gpu", "specialization": SPECIALIZATION["gpu"],
                  "dtype": "float32", "ane_regions": len(front_regions), "load_s": round(t1 - t0, 2)},
        "body": {"bundle": str(body_b), "backend": args.backend, "specialization": SPECIALIZATION[args.backend],
                 "dtype": "float16", "ane_regions": len(body_regions), "ane_region_names": body_regions[:32],
                 "load_s": round(t2 - t1, 2)},
        "host": "the front's hidden (float32) cast to float16 with numpy before the body call",
        "bar": {"cos_mean_per_clip": BAR_COS_MEAN, "cos_min_per_clip": BAR_COS_MIN},
        "summary": summary, "red_arm_masks_all_ones": red, "diagnostics": diag, "pipelined": pipelined,
        "call_ms_median_pass2": med,
        "timing_note": "one pass of 20 clips; the timing gate is bench_encoder.py",
        "runtime": runtime_info(), "label": args.label, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "clips_table": rows,
    }
    out = Path(args.out) if args.out else body_b.parent / f"gate_split{stage}_T{bucket}.json"
    write_json(out, result)
    print_summary(f"split{stage} T{bucket} gpu+{args.backend}", summary, len(body_regions), t2 - t0, out)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--bundle", help=".aimodel (cpu) or .aimodelc (gpu / ane); not with --front / --body")
    ap.add_argument("--backend", choices=("cpu", "gpu", "ane"), required=True,
                    help="with --front / --body: the body's backend (the front always runs on the GPU)")
    ap.add_argument("--front", help="two-bundle form: the float32 front AOT bundle (split_encoder.py)")
    ap.add_argument("--body", help="two-bundle form: the float16 body AOT bundle (split_encoder.py)")
    ap.add_argument("--ref-hidden", help="two-bundle form: npz of split_encoder.py reference (all clips, this "
                                         "bucket) for the front-alone and body-alone diagnostics")
    ap.add_argument("--pipeline", action="store_true",
                    help="also gate the two-in-flight order against pass 1 (two-bundle: front i+1 || body i; one "
                         "bundle: two calls issued together)")
    ap.add_argument("--form", help="default: read from the bundle name eg2_audio_<form>_T<Tb>")
    ap.add_argument("--bucket", type=int, help="default: read from the bundle name")
    ap.add_argument("--contract", choices=("eg2stt", "everpaper"), default="eg2stt")
    ap.add_argument("--function", help="default: main (eg2stt) or audio_<Tb> (everpaper)")
    ap.add_argument("--fixtures")
    ap.add_argument("--out", help="default: <bundle dir>/gate_<backend>_T<Tb>.json")
    ap.add_argument("--label", default="", help="free text recorded in the json (e.g. the measurement window)")
    ap.add_argument("--jit", action="store_true",
                    help="gpu / ane on the JIT .aimodel (preferred compute unit); ANE regions are then counted in "
                         "this process's Core AI cache entry")
    args = ap.parse_args()
    if args.front or args.body:
        if not (args.front and args.body) or args.bundle or args.backend == "cpu":
            raise SystemExit("the two-bundle gate needs --front and --body, no --bundle, and --backend gpu or ane")
        await main_split(args)
        return
    if not args.bundle:
        raise SystemExit("--bundle is required (or --front and --body)")
    bundle = Path(args.bundle).expanduser().resolve()
    if args.contract == "everpaper":
        if not (args.form and args.bucket):
            raise SystemExit("--contract everpaper needs --form (a row label) and --bucket")
        form, bucket = args.form, args.bucket
    else:
        form, bucket = parse_name(bundle)
        form, bucket = args.form or form, args.bucket or bucket
    fname = args.function or ("main" if args.contract == "eg2stt" else f"audio_{bucket}")
    if args.backend == "cpu" and bundle.suffix != ".aimodel":
        raise SystemExit("cpu gates the JIT .aimodel (an h16c AOT bundle cannot be specialized for the CPU)")
    if args.backend != "cpu" and bundle.suffix != (".aimodel" if args.jit else ".aimodelc"):
        raise SystemExit("gpu / ane gate the AOT .aimodelc (or the JIT .aimodel with --jit)")
    dtype = np.float16 if (fp16_io(form) or args.contract == "everpaper") else np.float32
    clips = load_clips(fixtures_dir(args.fixtures))
    regions = ane_regions(bundle) if bundle.suffix == ".aimodelc" else []

    t0 = time.perf_counter()
    opts = (rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
            if (args.jit and args.backend == "gpu") else options(args.backend))
    model = await rt.AIModel.load(bundle, opts)
    fn = model.load_function(fname)
    load_s = time.perf_counter() - t0
    cache_regions = None
    if args.jit:      # the runtime specialized the source bundle here: its ANE regions live in the cache entry
        build_id = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
        entry = (Path.home() / "Library" / "Caches" / "coreai-cache" / build_id / Path(sys.executable).name
                 / (bundle / "main.hash").read_bytes().hex())
        regions = ane_regions(entry) if entry.exists() else []
        cache_regions = str(entry)
    desc = {"inputs": list(fn.desc.input_names), "outputs": list(fn.desc.output_names)}

    outs1, secs1 = await run_all(fn, clips, bucket, dtype, args.contract)
    outs2, secs2 = await run_all(fn, clips, bucket, dtype, args.contract)
    rows = clip_rows(clips, outs1, outs2, secs1, bucket)
    red_clips = [c for c in clips if c["n_frames"] % 4 == 2 and c["n_frames"] <= bucket]
    red_outs, _ = await run_all(fn, red_clips, bucket, dtype, args.contract, ones_mask=True)
    red = red_rows(red_clips, red_outs, bucket)
    summary = summarize(rows, red)
    pipelined = None
    if args.pipeline:
        # two calls in flight (the order bench_encoder.py --pipeline times) must give the sequential outputs
        feeds = [make_inputs(args.contract, *window_input(c["features"], bucket), bucket, dtype) for c in clips]
        pouts = []
        for i in range(0, len(feeds), 2):
            res = await asyncio.gather(*(fn(f) for f in feeds[i:i + 2]))
            pouts += [r["soft_tokens"].numpy().astype(np.float32)[0] for r in res]
        dev = [float(np.abs(p[:compared_tokens(c, bucket)].astype(np.float64) - o[:compared_tokens(c, bucket)]).max())
               for c, o, p in zip(clips, outs1, pouts)]
        pipelined = {"what": "two calls in flight vs the sequential pass 1", "max_abs_vs_sequential": max(dev)}
        print(f"pipelined: max|d| vs sequential {max(dev):.1e}", flush=True)
    result = {
        "bundle": str(bundle), "function": fname, "contract": args.contract, "form": form, "bucket_frames": bucket,
        "backend": args.backend,
        "specialization": ("JIT from_preferred_compute_unit_kind(gpu)" if (args.jit and args.backend == "gpu") else
                           ("JIT " if args.jit else "") + SPECIALIZATION[args.backend]),
        "ane_regions": len(regions), "ane_region_names": regions[:32], "ane_regions_counted_in": cache_regions,
        "input_dtype": np.dtype(dtype).name, "function_descriptor": desc,
        "bar": {"cos_mean_per_clip": BAR_COS_MEAN, "cos_min_per_clip": BAR_COS_MIN},
        "summary": summary, "red_arm_masks_all_ones": red, "pipelined": pipelined,
        "timing_note": "one pass of 20 clips; the timing gate is bench_encoder.py",
        "load_s": round(load_s, 2), "first_call_s": round(secs1[0], 4),
        "median_call_s_pass2": round(float(np.median(secs2)), 4),
        "runtime": runtime_info(), "label": args.label, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "clips_table": rows,
    }
    if args.out:
        out = Path(args.out)
    elif args.contract == "everpaper":
        out = bundle.parent / f"gate_everpaper_{fname}_{args.backend}.json"
    else:
        out = bundle.parent / f"gate_{args.backend}{'_jit' if args.jit else ''}_T{bucket}.json"
    write_json(out, result)
    print_summary(f"{form} T{bucket} {args.backend}", summary, len(regions), load_s, out)


if __name__ == "__main__":
    asyncio.run(main())
