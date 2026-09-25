"""Diagnostic (CPU torch): which parts of the text encoder must run in fp32 for every token to hold?

References: ``_work/ref_hf_fp32_text/<prompt>/`` (HF fp32; the fp32 re-author matches them bit for bit).
Weights are the checkpoint's bf16 values throughout; "fp32" below means fp32 COMPUTE on those
values (exact upcast), so an all-fp32 run must reproduce the references.

1. Scan (all-fp32 run): per prompt, the tokens whose residual exceeds |h| 1000 in some layer, and
   the layers where each one jumps up (x5) and comes back down.
2. Variants, each on every prompt, per-token corr vs the reference:
     plain    bf16 residual, bf16 compute (the spec'd graph)
     r32      fp32 residual, bf16 compute (the exported candidate)
     r32+out  r32, plus o_proj / down_proj produce fp32 (bf16 inputs, no rounding of their outputs)
     r32+mlp  r32, plus the whole MLP in fp32 in every layer
     r32+attn r32, plus the whole attention in fp32 in every layer
     r32+L{..} r32, plus whole layers in fp32 (the scan's jump layers, ``--layers-fp32`` to override)
     fp32     everything fp32 (sanity: must equal the reference)

Run (base venv; ~16 GB RAM + a few GB of temporaries):
  python diag_text_islands.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import hf_snapshot  # noqa: E402
from qi21_text import apply_rope, load_qi21_text  # noqa: E402
from sweep_encoder_L import load_refs, tok_corr  # noqa: E402

MODEL = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
BF, F32 = torch.bfloat16, torch.float32


def lin(x, mod, fp32: bool):
    return F.linear(x.float(), mod.weight.float()) if fp32 else mod(x.to(BF))


def norm(x, n, dt):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + n.eps)
    return n.weight.to(dt) * xf.to(dt)


def attention(a, h, cos, sin, fp32: bool, out32: bool):
    B, L, _ = h.shape
    hd, dt = a.head_dim, (F32 if fp32 else BF)
    q = norm(lin(h, a.q_proj, fp32).view(B, L, a.heads, hd), a.q_norm, dt).transpose(1, 2)
    k = norm(lin(h, a.k_proj, fp32).view(B, L, a.kv_heads, hd), a.k_norm, dt).transpose(1, 2)
    v = lin(h, a.v_proj, fp32).view(B, L, a.kv_heads, hd).transpose(1, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    rep = a.heads // a.kv_heads
    k = k[:, :, None].expand(B, a.kv_heads, rep, L, hd).reshape(B, a.heads, L, hd)
    v = v[:, :, None].expand(B, a.kv_heads, rep, L, hd).reshape(B, a.heads, L, hd)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, L, a.heads * hd)
    return lin(o, a.o_proj, fp32 or out32)


def mlp(m, h, fp32: bool, out32: bool):
    x = F.silu(lin(h, m.gate_proj, fp32)) * lin(h, m.up_proj, fp32)
    return lin(x, m.down_proj, fp32 or out32)


def run(model, ids, residual32=True, attn32=False, mlp32=False, out32=False, layers32=(), capture=False):
    L = ids.shape[1]
    cos, sin = model.rope_cos[:L], model.rope_sin[:L]
    x = model.embed_tokens(ids)
    x = x.float() if residual32 else x
    peaks = []
    for i, layer in enumerate(model.layers):
        whole = i in layers32
        a32, m32 = attn32 or whole, mlp32 or whole
        h = norm(x, layer.input_layernorm, F32 if a32 else BF)
        x = x + attention(layer.self_attn, h, cos, sin, a32, out32).to(x.dtype)
        h = norm(x, layer.post_attention_layernorm, F32 if m32 else BF)
        x = x + mlp(layer.mlp, h, m32, out32).to(x.dtype)
        if capture:
            peaks.append(x[0].float().abs().amax(-1))
    return x[0].float(), (torch.stack(peaks) if capture else None)       # peaks [layers, L]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers-fp32", default=None, help="comma list; default = the scan's jump layers")
    args = ap.parse_args()
    refs = load_refs()
    tdir = Path(hf_snapshot(MODEL, revision=REVISION)) / "text_encoder"
    model = load_qi21_text(tdir, dtype=BF)
    res = dict(scan={}, variants={})
    jump_layers = set()
    with torch.no_grad():
        for key, (ids, ref) in refs.items():
            t = torch.from_numpy(ids)
            out, peaks = run(model, t, residual32=True, attn32=True, mlp32=True, capture=True)
            exact = float((out.numpy() - ref).__abs__().max())
            big = [int(j) for j in torch.nonzero(peaks.max(0).values > 1000).flatten()]
            info = {}
            for j in big:
                p = peaks[:, j]
                prev = torch.cat([p.new_tensor([1.0]), p[:-1]])
                up = [int(i) for i in torch.nonzero(p > 5 * prev).flatten()]
                down = [int(i) for i in torch.nonzero(p < prev / 5).flatten()]
                info[j] = dict(id=int(ids[0, j]), peak=float(p.max()), final=float(p[-1]), up=up, down=down)
                jump_layers.update(up + down)
            res["scan"][key] = dict(fp32_run_max_abs_vs_ref=exact, tokens=info)
            print(f"[islands] scan {key:<8} (all-fp32 run vs ref max|d| {exact:.3e}): " + "; ".join(
                f"tok {j} id {v['id']} peak {v['peak']:.0f} final {v['final']:.0f} up@{v['up']} down@{v['down']}"
                for j, v in info.items()), flush=True)
        layers32 = (sorted(int(v) for v in args.layers_fp32.split(",")) if args.layers_fp32 else sorted(jump_layers))
        print(f"[islands] fp32 layer set for 'r32+L': {layers32}", flush=True)
        variants = {
            "plain": dict(residual32=False),
            "r32": dict(),
            "r32+out": dict(out32=True),
            "r32+mlp": dict(mlp32=True),
            "r32+attn": dict(attn32=True),
            f"r32+L{','.join(map(str, layers32))}": dict(layers32=set(layers32)),
            "fp32": dict(attn32=True, mlp32=True),
        }
        for name, kw in variants.items():
            t0 = time.time()
            row = {}
            for key, (ids, ref) in refs.items():
                out, _ = run(model, torch.from_numpy(ids), **kw)
                c = tok_corr(out.numpy(), ref)
                row[key] = dict(min=float(c.min()), argmin=int(c.argmin()), n_below=int((c < 0.999).sum()),
                                tok14=float(c[14]), worst3=[(int(i), float(c[i])) for i in np.argsort(c)[:3]])
            res["variants"][name] = row
            print(f"[islands] {name:<22} ({time.time() - t0:4.1f}s) " + "  ".join(
                f"{k}: min {v['min']:.6f} @{v['argmin']} tok14 {v['tok14']:.6f} n<0.999 {v['n_below']}"
                for k, v in row.items()), flush=True)
    json.dump(res, open(HERE / "_work" / "diag_text_islands.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
