"""Gate the Bonsai ternary matvec, GEMM and packed-embedding kernels on REAL layer-0 weights.

One-op graphs per kernel, exported with the custom-kernel hook, run on the GPU through
coreai.runtime, compared against the exact dequantised reference. Needs the PQ2_0 GGUF in the
HF cache (export_bonsai2_27b_decode_pipelined.py downloads it).

    .venv/bin/python _smoke/bonsai/gate_kernels_real.py [--chunk 64]
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


async def run(aimodel: Path, feed: dict, reps: int = 20):
    import coreai.runtime as rt
    m = await rt.AIModel.load(
        str(aimodel), rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    fn = m.load_function("main")
    nd = {k: rt.NDArray(np.ascontiguousarray(v)) for k, v in feed.items()}
    out = (await fn(inputs=nd))["y"].numpy()
    t0 = time.perf_counter()
    for _ in range(reps):
        await fn(inputs=nd)
    return out, (time.perf_counter() - t0) / reps


def export_run(model, feed: dict, kernels: list, work: Path, name: str):
    from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels
    import coreai.runtime as rt
    prog = export_to_coreai_with_kernels(model, feed, custom_kernels=kernels,
                                         input_names=tuple(feed), output_names=("y",))
    prog.optimize()
    aimodel = work / f"{name}.aimodel"
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    return asyncio.run(run(aimodel, {k: v.numpy() for k, v in feed.items()}))


class Wrap(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)


def report(tag, y, ref, dt):
    y = y.astype(np.float32)
    ref = ref.astype(np.float32)
    err = np.abs(y - ref)
    scale = np.abs(ref).max()
    rel = err.max() / max(scale, 1e-9)
    ok = np.isfinite(y).all() and rel < 2e-3
    print(f"[gate] {tag:34s} max|err| {err.max():.3e}  max|ref| {scale:.3e}  rel {rel:.2e}  "
          f"mean|err| {err.mean():.2e}  {dt * 1e3:.3f} ms/call  {'OK' if ok else 'FAIL'}", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=64)
    args = ap.parse_args()

    import gguf
    from huggingface_hub import hf_hub_download
    from coreai_models.models.macos.bonsai2 import _Src, hadamard_signs_from_gguf
    from coreai_models.models.macos.bonsai_hadamard_metal import (
        PackedEmbedding, build_embed_kernel, _embed_torch_defn,
    )
    from coreai_models.models.macos.bonsai_ternary_metal import (
        TernaryLinear128, build_gemm_kernel, build_mv_kernel, dequant_reference,
    )

    path = hf_hub_download("prism-ml/Ternary-Bonsai-2-27B-gguf", "Ternary-Bonsai-2-27B-PQ2_0.gguf")
    reader = gguf.GGUFReader(path)
    src = _Src(reader)
    _, signs, _ = hadamard_signs_from_gguf(reader)
    work = Path(tempfile.mkdtemp(prefix="bonsai_kgate_"))
    torch.manual_seed(0)
    ok = True

    # 1) embedding: gather + dequant + inverse transform on the real table (first 4096 rows)
    qp, d = src.packed("token_embd.weight")
    qp, d = qp[:4096].contiguous(), d[:4096].contiguous()
    emb = PackedEmbedding(qp, d, signs[5120], build_embed_kernel()).eval()
    ids = torch.tensor([[17, 4095, 0, 2048, 1000, 7, 7, 3]], dtype=torch.int32)
    y, dt = export_run(Wrap(emb), {"x": ids}, [emb.kernel], work, "embed")
    with torch.no_grad():
        ref = _embed_torch_defn(ids.reshape(-1), qp, d, signs[5120]).numpy()
    ok &= report("embed (8 ids, K=5120)", y.reshape(-1, 5120), ref, dt)

    # 2) matvec on attn_qkv of layer 0 (K=5120, N=10240) and ffn_down (K=17408, N=5120)
    mv = build_mv_kernel()
    for name in ("blk.0.attn_qkv.weight", "blk.0.ffn_down.weight"):
        qp, d = src.packed(name)
        lin = TernaryLinear128(qp, d, mv, None, site=None).eval()
        x = (torch.randn(1, 1, lin.K) * 0.5).to(torch.float16)
        y, dt = export_run(Wrap(lin), {"x": x}, [mv], work, "mv_" + name.replace(".", "_"))
        with torch.no_grad():
            ref = torch.nn.functional.linear(x.float().reshape(1, -1), dequant_reference(qp, d)).numpy()
        ok &= report(f"matvec {name} S=1", y.reshape(1, -1), ref, dt)

    # 3) GEMM at S=chunk on attn_qkv
    if args.chunk > 1:
        gemm = build_gemm_kernel(args.chunk)
        qp, d = src.packed("blk.0.attn_qkv.weight")
        lin = TernaryLinear128(qp, d, mv, gemm, site=None).eval()
        x = (torch.randn(1, args.chunk, lin.K) * 0.5).to(torch.float16)
        y, dt = export_run(Wrap(lin), {"x": x}, [mv, gemm], work, "gemm_qkv")
        with torch.no_grad():
            ref = torch.nn.functional.linear(x.float().reshape(args.chunk, -1), dequant_reference(qp, d)).numpy()
        ok &= report(f"gemm blk.0.attn_qkv S={args.chunk}", y.reshape(args.chunk, -1), ref, dt)

    shutil.rmtree(work, ignore_errors=True)
    print(f"[gate] {'ALL OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
