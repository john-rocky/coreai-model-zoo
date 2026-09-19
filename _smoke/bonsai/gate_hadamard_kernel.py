"""Gate the standalone Bonsai Hadamard kernel (bonsai_hadamard_metal.py) on the Core AI GPU path.

For each real Bonsai 2 27B width (5120 / 6144 / 17408, sign vectors from the shipped GGUF
header in hadamard_signs.json) and S in {1, 64}: export a one-op graph with the kernel, load it
through coreai.runtime on the GPU, run random fp16 activations, and compare against the exact
fp64 transform rounded to fp16 and against the kernel's own torch reference. Then time it.

Pass criteria: every element within 1 fp16 ulp (at the rounded value, or a 2^-19 fp32 noise
floor near zero) of the exact result, and
the exact inverse (transform, then sign) applied to the kernel's fp16 output reproduces the
input to within one fp16 ulp at the output's scale (the rounding noise the inverse spreads).

Run from the zoo root with the overlay venv:
    .venv/bin/python _smoke/bonsai/gate_hadamard_kernel.py [--widths 5120,6144,17408] [--chunks 1,64]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SIGNS = HERE / "hadamard_signs.json"
# fp32 butterfly over 1024 terms of magnitude ~0.05: worst-case ~1e-4, observed ~1e-7 absolute
FP32_NOISE_FLOOR = np.float32(2.0 ** -19)   # 1.9e-6


def load_signs() -> dict[int, torch.Tensor]:
    d = json.loads(SIGNS.read_text())
    assert d["block_size"] == 1024 and d["sign_mode"] == "explicit"
    return {int(w): torch.tensor(v, dtype=torch.float32) for w, v in d["signs"].items()}


def fp16_ulp(y: np.ndarray) -> np.ndarray:
    """One fp16 ulp at each value: the gap to the next representable fp16 above |round16(y)|."""
    a = np.abs(y.astype(np.float16))
    return (np.nextafter(a, np.float16(np.inf)) - a).astype(np.float32)


async def run_one(aimodel: Path, x: np.ndarray, warm: int, reps: int):
    import coreai.runtime as rt
    m = await rt.AIModel.load(
        str(aimodel), rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    fn = m.load_function("main")
    feed = {"x": rt.NDArray(np.ascontiguousarray(x))}
    out = (await fn(inputs=feed))["y"].numpy()
    for _ in range(warm):
        await fn(inputs=feed)
    t0 = time.perf_counter()
    for _ in range(reps):
        await fn(inputs=feed)
    dt = (time.perf_counter() - t0) / reps
    return out, dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="5120,6144,17408")
    ap.add_argument("--chunks", default="1,64")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--keep", action="store_true", help="keep the exported .aimodel bundles")
    args = ap.parse_args()

    from coreai_models.models.macos.bonsai_hadamard_metal import (
        BonsaiHadamard, build_fwht_kernel, signed_hadamard_reference,
    )
    from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels

    signs = load_signs()
    kernel = build_fwht_kernel()
    torch.manual_seed(0)
    work = Path(tempfile.mkdtemp(prefix="bonsai_fwht_gate_"))
    print(f"[gate] scratch {work}", flush=True)

    all_ok = True
    rows = []
    for k in (int(w) for w in args.widths.split(",")):
        sg = signs[k]
        for s in (int(c) for c in args.chunks.split(",")):
            model = BonsaiHadamard(sg, kernel).eval()
            # activations at a realistic post-norm scale, fp16 like the decode graph
            x = (torch.randn(s, k) * 1.5).to(torch.float16)
            with torch.no_grad():
                ref_defn = model(x)                                   # torch_defn path (fp32 matmul)
            exact = signed_hadamard_reference(x, sg)                  # fp64 truth
            exact16 = exact.to(torch.float16)

            name = f"fwht_k{k}_s{s}"
            prog = export_to_coreai_with_kernels(
                model, {"x": x}, custom_kernels=[kernel], input_names=("x",), output_names=("y",))
            prog.optimize()
            aimodel = work / f"{name}.aimodel"
            import coreai.runtime as rt
            prog.save_asset(aimodel, rt.AIModelAssetMetadata())

            y, dt = asyncio.run(run_one(aimodel, x.numpy(), warm=5, reps=args.reps))
            y = y.astype(np.float32)
            ex = exact.numpy().astype(np.float32)
            err = np.abs(y - ex)
            # the kernel rounds an fp32 result to fp16 once: the exact value may sit on the far
            # side of a rounding boundary, so the bound is one fp16 ulp at the rounded value
            # ...and near zero the fp16 ulp (down to 6e-8) is far below the fp32 accumulation
            # noise of a 1024-term butterfly (~1e-7 absolute after cancellation), so the bound
            # there is an absolute fp32 noise floor instead
            ulps = err / np.maximum(fp16_ulp(ex), FP32_NOISE_FLOOR)
            n_off = int((np.abs(y - exact16.numpy().astype(np.float32)) > 0).sum())
            n_defn = int((np.abs(y - ref_defn.numpy().astype(np.float32)) > 0).sum())
            ok = bool(ulps.max() <= 1.0 + 1e-6)

            # round trip: inverse = transform first, then sign (S·H·y/32) must give x back up to
            # the fp16 rounding noise of y, which the orthogonal inverse spreads at output scale
            back = signed_hadamard_reference(torch.from_numpy(y), sg, inverse=True)
            rt_err = float((back.numpy() - x.numpy().astype(np.float64)).__abs__().max())
            rt_tol = float(fp16_ulp(np.array([np.abs(y).max()]))[0])
            rt_ulps = rt_err / rt_tol
            ok = ok and rt_ulps <= 1.0 + 1e-6

            all_ok &= ok
            rows.append((k, s, err.max(), ulps.max(), n_off, y.size, n_defn, rt_err, dt))
            print(f"[gate] K={k:5d} S={s:3d}  max|err| {err.max():.3e}  max ulp {ulps.max():.2f}  "
                  f"fp16-mismatch vs exact {n_off}/{y.size}  vs torch_defn {n_defn}  "
                  f"round-trip max|err| {rt_err:.3e} ({rt_ulps:.2f} of an output-scale ulp)  "
                  f"{dt * 1e3:.3f} ms/call  {'OK' if ok else 'FAIL'}", flush=True)
            if not args.keep:
                shutil.rmtree(aimodel, ignore_errors=True)

    print()
    print("| K | S | rows of 1024 | max ulp (floor 2^-19) | fp16 mismatches vs exact | vs torch_defn | ms/call |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for k, s, e, u, n_off, n, n_defn, rt_err, dt in rows:
        print(f"| {k} | {s} | {s * k // 1024} | {u:.2f} | {n_off}/{n} | {n_defn} | {dt * 1e3:.3f} |")
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n[gate] {'ALL OK' if all_ok else 'FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
