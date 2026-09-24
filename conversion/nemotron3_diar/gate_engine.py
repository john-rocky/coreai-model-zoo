"""Gate the exported bundles against transformers' per-step chunk logits (low-latency streaming
steps captured by make_reference.py), padding rows [L, 541) filled with seeded N(0, 1) noise.

    (a) fp32 bundle, SpecializationOptions.cpu_only():            max|Δlogit| <= 1e-3
    (b) fp16 bundle, from_preferred_compute_unit_kind(gpu()):      reported: max|Δlogit|, max|Δp|
        (after sigmoid), speaker-activity agreement @0.5 over every real L*8 x 8 element
        (guide: agreement >= 99.9 %, max|Δp| <= 0.02)

Default = every captured step of both fixtures (L=541 steps and the compressed steps included).

    ~/code/coreai/coreai-models/.venv/bin/python gate_engine.py                       # (a) + (b)
    ~/code/coreai/coreai-models/.venv/bin/python gate_engine.py --runs float16:gpu    # one run
    ~/code/coreai/coreai-models/.venv/bin/python gate_engine.py --runs float16:aot-h16c  # AOT fallback
    ~/code/coreai/coreai-models/.venv/bin/python gate_engine.py --tag _safeln --runs float32:cpu,float16:ane
    ~/code/coreai/coreai-models/.venv/bin/python gate_engine.py --profile offline --runs float32:cpu,float16:gpu

`aot-h16c` loads _work/artifacts/aot_h16c/n3d_streaming_<dtype>.aimodelc (built with
`xcrun coreai-build compile <aimodel> --output <dir> --platform macOS --architecture h16c
--preferred-compute gpu`) with SpecializationOptions.default().

Per-step numbers go to _work/gate_engine_<dtype>_<unit>.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gate_reauthor import corr, pack  # noqa: E402
from n3d_model import SUB, T_OFFLINE, T_STREAM  # noqa: E402

WORK = HERE / "_work"
ART = WORK / "artifacts"
FP32_MAX_ABS = 1e-3


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def bundle_path(dtype: str, unit: str, tag: str = "", profile: str = "streaming") -> Path:
    if unit.startswith("aot-"):
        return ART / f"aot_{unit[4:]}" / f"n3d_{profile}{tag}_{dtype}.aimodelc"
    return ART / f"n3d_{profile}{tag}_{dtype}.aimodel"


def options(rt, unit: str):
    if unit == "cpu":
        return rt.SpecializationOptions.cpu_only()
    if unit == "gpu":
        return rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    if unit == "ane":
        return rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.neural_engine())
    if unit.startswith("aot-"):
        return rt.SpecializationOptions.default()
    raise ValueError(unit)


async def run(dtype: str, unit: str, fixtures: list[str], tag: str = "", profile: str = "streaming") -> dict:
    import coreai.runtime as rt

    T = T_STREAM if profile == "streaming" else T_OFFLINE
    path = bundle_path(dtype, unit, tag, profile)
    t0 = time.time()
    model = await rt.AIModel.load(path, options(rt, unit))       # keep the reference alive
    fn = model.load_function("main")
    load_s = time.time() - t0
    print(f"[{dtype}:{unit}] loaded {path.name} in {load_s:.1f} s", flush=True)

    out = {"dtype": dtype, "unit": unit, "profile": profile, "bundle": str(path), "load_s": round(load_s, 2),
           "fixtures": {}}
    agree_n = agree_ok = 0
    worst_abs = worst_p = 0.0
    min_corr = 1.0
    zero_outputs = 0
    first_call_s = None
    for name in fixtures:
        d = np.load(WORK / f"chunk_io_{name}_{'ll' if profile == 'streaming' else 'offline'}.npz")
        rows = []
        for i in range(int(d["n_steps"])):
            emb, ref = d[f"s{i:03d}_inputs_embeds"], d[f"s{i:03d}_chunk_logits"]
            L = emb.shape[0]
            packed, valid = pack(emb, T, "noise", i)
            n_cmp = L
            mask = d[f"s{i:03d}_step_mask"] if f"s{i:03d}_step_mask" in d.files else None
            if mask is not None:
                # offline: transformers masks the extractor's extra centered frame as a key but does not zero
                # it before the conv; the graph's single `valid` does both, so the last real row's conv
                # neighbour differs: leave the last 2 encoder rows out of this step's comparison
                valid[0, :L] = torch.from_numpy(mask.astype(np.float32))
                n_cmp = L - 2
            t1 = time.time()
            res = await fn({"packed": rt.NDArray(packed.numpy()), "valid": rt.NDArray(valid.numpy())})
            if first_call_s is None:
                first_call_s = time.time() - t1
            full = res["logits"].numpy().astype(np.float32)
            assert full.shape == (1, T * SUB, 8), full.shape
            got = full[0, : n_cmp * SUB]
            ref = ref[: n_cmp * SUB]
            if not np.any(full):
                zero_outputs += 1
            dabs = float(np.abs(got - ref).max())
            pg, pr = sigmoid(got), sigmoid(ref)
            dp = float(np.abs(pg - pr).max())
            agree = int(((pg > 0.5) == (pr > 0.5)).sum())
            c = corr(got, ref)
            rows.append({"step": i, "L": L, "compared_rows": n_cmp, "masked_step": mask is not None,
                         "compressed_after": bool(d["compressed_after"][i]),
                         "max_abs": dabs, "corr": c, "max_dp": dp, "agree": agree, "n": int(got.size)})
            agree_n += got.size
            agree_ok += agree
            worst_abs, worst_p, min_corr = max(worst_abs, dabs), max(worst_p, dp), min(min_corr, c)
        out["fixtures"][name] = rows
        n = len(rows)
        print(f"[{dtype}:{unit}] {name}: {n} steps, worst max|Δlogit| {max(r['max_abs'] for r in rows):.3e}, "
              f"min corr {min(r['corr'] for r in rows):.7f}, worst max|Δp| {max(r['max_dp'] for r in rows):.3e}, "
              f"agreement {sum(r['agree'] for r in rows) / sum(r['n'] for r in rows) * 100:.4f} %", flush=True)
    out.update({"worst_max_abs": worst_abs, "min_corr": min_corr, "worst_max_dp": worst_p,
                "agreement": agree_ok / agree_n, "elements": agree_n, "disagree": agree_n - agree_ok,
                "zero_output_steps": zero_outputs, "first_call_s": round(first_call_s or 0.0, 2)})
    (WORK / f"gate_engine{'_offline' if profile == 'offline' else ''}{tag}_{dtype}_{unit}.json").write_text(
        json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="float32:cpu,float16:gpu")
    ap.add_argument("--fixtures", default="diarization_example,test_multispk")
    ap.add_argument("--tag", default="", help="bundle name tag, e.g. _safeln -> n3d_streaming_safeln_<dtype>.aimodel")
    ap.add_argument("--profile", choices=["streaming", "offline"], default="streaming",
                    help="offline: the T=684 bundle on chunk_io_<fixture>_offline.npz")
    args = ap.parse_args()
    fixtures = args.fixtures.split(",")

    ok_all = True
    for spec in args.runs.split(","):
        dtype, unit = spec.split(":")
        r = asyncio.run(run(dtype, unit, fixtures, args.tag, args.profile))
        line = (f"[{dtype}:{unit}] ALL: worst max|Δlogit| {r['worst_max_abs']:.3e}, min corr {r['min_corr']:.7f}, "
                f"worst max|Δp| {r['worst_max_dp']:.3e}, agreement {r['agreement'] * 100:.4f} % "
                f"({r['disagree']} of {r['elements']} differ), zero-output steps {r['zero_output_steps']}")
        if dtype == "float32" and unit == "cpu":
            ok = r["worst_max_abs"] <= FP32_MAX_ABS and r["zero_output_steps"] == 0
            ok_all &= ok
            line += f" -> {'PASS' if ok else 'FAIL'} (bar max|Δlogit| <= {FP32_MAX_ABS})"
        elif r["zero_output_steps"]:
            ok_all = False
            line += " -> ZERO OUTPUT (see the lane trap: switch to the AOT h16c path)"
        print(line, flush=True)
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
