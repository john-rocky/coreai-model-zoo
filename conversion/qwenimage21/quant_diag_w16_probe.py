"""Diagnostic: what the int8 ``--struct w16`` probe bundle actually computes (round 4).

The 2-layer random probe of ``quant_export_encoder.py --struct w16`` (bf16 scales, dequant -> bf16 ->
cast fp32 -> fp32 ``F.linear``) matched its own torch module only to rel ~1.2e-3, where the
``--struct lin32`` probe (fp32 scales, dequant -> fp32) matched to rel ~2e-6. The engine output is
scored here against three torch readings of the same quantized weights, one per hypothesis:

  exact       dequant in bf16 (the op's eager kernel), upcast, fp32 matmul   (what the graph says)
  deq32       dequant in fp32 (no bf16 rounding of q*s), fp32 matmul
  act_bf16    dequant in bf16, the matmul INPUT cast to bf16 (bf16 x bf16, fp32 out)

The closest reading is what the Core AI GPU path computes.

Run (base venv, from conversion/qwenimage21/, after the probe exists):
  python quant_diag_w16_probe.py
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import quant_export_encoder as qe  # noqa: E402


def deq32_forward(self, x):
    p = self.parametrizations.weight[0]
    w = torch.ops.coreai.constexpr_blockwise_shift_scale(p.quantized_data, p.scale.to(torch.float32))
    return F.linear(x.to(torch.float32), w)


def act_bf16_forward(self, x):
    return F.linear(x.to(torch.bfloat16), self.weight).to(torch.float32)


def rel(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


async def engine(path, Ls, seed, vocab):
    import coreai.runtime as rt
    aim = await rt.AIModel.load(path, rt.SpecializationOptions.default())
    fn = aim.load_function("main")
    outs = {}
    for L in Ls:
        g = torch.Generator().manual_seed(seed + L)
        ids = torch.randint(0, vocab, (1, L), generator=g, dtype=torch.int32)
        r = await fn({"input_ids": rt.NDArray(ids.contiguous())})
        outs[L] = (ids, np.array(r["hidden"].numpy(), dtype=np.float32))
    del fn, aim
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wbits", type=int, default=8)
    ap.add_argument("--block", type=int, default=32)
    args = ap.parse_args()
    a = argparse.Namespace(struct="w16", random_init=True, layers=2, seed=0, L_max=512)
    model = qe.build(a)
    from coreai_models.export.compression import quantize_pytorch_model
    g = torch.Generator().manual_seed(1)
    ref_ids = torch.randint(0, model.embed_tokens.num_embeddings, (1, 64), generator=g, dtype=torch.int32)
    model = quantize_pytorch_model(model, (ref_ids,), None, qe.quant_config(args.wbits, args.block, "w16"))
    name = qe.bundle_name(2, qe.variant_tag(args.wbits, args.block), "w16", True)
    _, aimodelc = qe.paths(name)
    outs = asyncio.run(engine(aimodelc, [32, 128], 2, model.embed_tokens.num_embeddings))
    lin = [m for m in model.modules() if isinstance(m, qe.ToLinear)]
    for L, (ids, out) in outs.items():
        row = {}
        for tag, fwd in (("exact", None), ("deq32", deq32_forward), ("act_bf16", act_bf16_forward)):
            for m in lin:               # per-instance override: the parametrized class keeps its weight property
                if fwd is None:
                    m.__dict__.pop("forward", None)
                else:
                    m.forward = types.MethodType(fwd, m)
            with torch.no_grad():
                row[tag] = rel(out, model(ids).float().numpy())
        for m in lin:
            m.__dict__.pop("forward", None)
        print(f"[diag] {name} L={L}: rel(engine, torch reading) " +
              "  ".join(f"{k} {v:.3e}" for k, v in row.items()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
