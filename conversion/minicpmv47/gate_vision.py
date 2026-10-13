#!/usr/bin/env python3
"""G3: one MiniCPM-V-4.7 vision tower form on the Mac GPU (AOT h16c, python runtime) vs the fp32 oracle.

One process, one form (`<exports>/minicpmv47_aotc/minicpmv47_vision_<mode>.h16c.aimodelc`,
`SpecializationOptions.default()`). Per image prompt P1..P4: `host.preprocess(<fixture>.448.png)` (Pillow
BICUBIC 448, (x/255 - 0.5)/0.5, CHW) cast to the function's input dtype -> `image_features` [64, 1024] against
the oracle's tower rows `oracle.npz <P>__image_features` (fp32): per-token cosine (mean / min over the 64
rows), max |d|, the oracle's absmax, finiteness.

Timing (run the whole process inside a quiet_hold window): the first call after load (cold, P1), then the
four prompts, then 5 more calls on P1 (warm; median and all values). The process ends with one more P1 call
whose output must be bit-equal to the first P1 output.

Bars (the supervisor's, fixed): fp16w32 cos mean >= 0.9999 and min >= 0.999; int8lin mean >= 0.9995 and
min >= 0.99; fp16 recorded only (said to pass when it meets the fp16w32 bar).

    QW QH mv47-r3-g3-<mode> -- coreai-models/.venv/bin/python conversion/minicpmv47/gate_vision.py \
        <exports>/minicpmv47_vision_<mode> --out <lane>/r3/gate_vision_<mode>.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_common as G  # noqa: E402
import host as H  # noqa: E402

BARS = {"fp16w32": {"cos_mean": 0.9999, "cos_min": 0.999}, "int8lin": {"cos_mean": 0.9995, "cos_min": 0.99},
        "fp16": {"cos_mean": 0.9999, "cos_min": 0.999}}
N_WARM = 5


def compare(got: np.ndarray, want: np.ndarray) -> dict:
    g, w = got.astype(np.float64), want.astype(np.float64)
    rows = (g * w).sum(-1) / (np.linalg.norm(g, axis=-1) * np.linalg.norm(w, axis=-1))
    return {"cos_mean": float(rows.mean()), "cos_min": float(rows.min()), "cos_min_row": int(rows.argmin()),
            "rows_below_0_999": int((rows < 0.999).sum()), "max_abs_d": float(np.abs(g - w).max()),
            "ref_absmax": float(np.abs(w).max()), "got_absmax": float(np.abs(g).max()),
            "finite": bool(np.isfinite(g).all())}


async def main_async(args) -> int:
    import coreai.runtime as rt

    bundle = Path(args.bundle).expanduser().resolve()
    meta = json.loads((bundle / "metadata.json").read_text())
    mode = meta["vision"]["dtype"]
    aimodelc = G.aimodelc_for(bundle)
    z = np.load(G.ORACLE / "oracle.npz")
    fixtures = G.ORACLE / "fixtures"
    window = G.window_record(f"g3-{Path(args.bundle).name}")
    gs = G.GpuSampler().__enter__()
    t_all = time.perf_counter()
    t0 = time.perf_counter()
    model = await G.maybe(rt.AIModel.load(str(aimodelc), rt.SpecializationOptions.default()))
    t1 = time.perf_counter()
    names = list(getattr(model, "function_names", []) or [])
    fn = await G.maybe(model.load_function(names[0] if names else "main"))
    load = {"model": t1 - t0, "function": time.perf_counter() - t1}
    desc = G.fn_desc(fn)
    in_shape, in_dtype = desc["inputs"]["pixel_values"]
    out_shape, out_dtype = desc["outputs"]["image_features"]
    if in_shape != [1, 3, 448, 448] or out_shape != [64, 1024]:
        raise SystemExit(f"descriptor {desc}")
    np_in = np.dtype(in_dtype)

    async def encode(px: np.ndarray) -> tuple[np.ndarray, float]:
        t = time.perf_counter()
        res = await G.maybe(fn(inputs={"pixel_values": G.nd(rt, px.astype(np_in))}))
        o = np.asarray(res["image_features"].numpy()).copy()
        return o, (time.perf_counter() - t) * 1e3

    px = {P: H.preprocess(fixtures / f"{stem}.448.png") for P, stem in G.PROMPT_IMAGES.items()}
    first, cold_ms = await encode(px["P1"])
    prompts = {}
    for P in ("P1", "P2", "P3", "P4"):
        o, ms = await encode(px[P])
        prompts[P] = {"image": f"{G.PROMPT_IMAGES[P]}.448.png", "encode_ms": ms,
                      **compare(o, z[f"{P}__image_features"])}
        if P == "P1":
            prompts[P]["equal_to_cold_call"] = bool(np.array_equal(o, first))
        print(f"[G3 {mode} {P}] cos mean {prompts[P]['cos_mean']:.6f} min {prompts[P]['cos_min']:.6f} "
              f"max|d| {prompts[P]['max_abs_d']:.4f} (ref absmax {prompts[P]['ref_absmax']:.3f}) {ms:.1f} ms", flush=True)
    warm = []
    for _ in range(N_WARM):
        _, ms = await encode(px["P1"])
        warm.append(ms)
    last, _ = await encode(px["P1"])
    bar = BARS[mode]
    mean_min = min(p["cos_mean"] for p in prompts.values())
    min_min = min(p["cos_min"] for p in prompts.values())
    ok = mean_min >= bar["cos_mean"] and min_min >= bar["cos_min"] and all(p["finite"] for p in prompts.values())
    out = {
        "schema": "minicpmv47-gate-vision/1", "mode": mode, "bundle": str(bundle),
        "aimodelc": G.tree_digest(aimodelc), "metadata_sha256": G.sha256_file(bundle / "metadata.json"),
        "oracle_npz_sha256": G.sha256_file(G.ORACLE / "oracle.npz"),
        "environment": G.env_record(), "window": window, "descriptor": desc, "input_fed_as": str(np_in),
        "host": "host.preprocess: Pillow BICUBIC 448, (x/255 - 0.5)/0.5, CHW [1,3,448,448] float32 -> input dtype",
        "load_seconds": load, "prompts": prompts,
        "summary": {"cos_mean_min_over_prompts": mean_min, "cos_min_over_prompts": min_min,
                    "max_abs_d": max(p["max_abs_d"] for p in prompts.values()),
                    "rows_below_0_999": sum(p["rows_below_0_999"] for p in prompts.values())},
        "bar": bar if mode != "fp16" else {**bar, "note": "fp16 is recorded only; 'passes' = meets the fp16w32 bar"},
        "timing": {"cold_first_call_ms": cold_ms, "warm_ms": warm, "warm_median_ms": float(np.median(warm)),
                   "warm_n": N_WARM, "per_prompt_ms": {P: p["encode_ms"] for P, p in prompts.items()}},
        "repro_last_call_bit_equal_first": bool(np.array_equal(first, last)),
    }
    out["result"] = ("PASS" if ok else "FAIL") if mode != "fp16" else ("meets the fp16w32 bar" if ok else "below the fp16w32 bar")
    out["wall_seconds"] = time.perf_counter() - t_all
    gs.__exit__()
    out["gpu_utilization"] = gs.summary()
    out["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["script"] = {"path": "conversion/minicpmv47/gate_vision.py", "sha256": G.sha256_file(Path(__file__))}
    G.write_json(Path(args.out), out)
    print(f"[G3 {mode}] {out['result']}: cos mean >= {mean_min:.6f}, min {min_min:.6f}; cold {cold_ms:.1f} ms, warm median "
          f"{out['timing']['warm_median_ms']:.2f} ms {[round(w, 2) for w in warm]}; repro bit-equal "
          f"{out['repro_last_call_bit_equal_first']} -> {args.out}", flush=True)
    return 0 if (ok or mode == "fp16") else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle")
    ap.add_argument("--out", required=True)
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
