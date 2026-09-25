"""Diagnostic: where does the bf16 text encoder lose precision on the oracle prompt? (CPU torch only)

Per layer, the bf16 re-author's residual stream vs the fp32 re-author's, for one token
(default 14 = the user turn's ``<|im_start|>``, the first token the DiT reads) and the rest.
Variants, each changing ONE thing against plain bf16 (bf16 weights, bf16 compute):

  bf16        the exported graph's numerics (RoPE rotated in fp32, norms with fp32 statistics)
  res32       residual stream kept in fp32: norms read the fp32 stream, sublayer outputs are
              added in fp32; every matmul still takes bf16 inputs and bf16 weights
  attn32      bf16, except q/k/v enter SDPA in fp32 (softmax and PV in fp32)
  mlp32       bf16, except silu(gate) * up and down_proj's input in fp32 (down weights bf16->fp32)

``--dit`` then feeds each variant's ``hidden[:, drop:]`` (and the HF bf16 hidden saved by
``check_mrope_hf.py``) to the fp32 DiT (real weights) on oracle steps 0/20/39 and scores the
velocity against the oracle's.

Run (base venv; ~46 GB RAM for the two encoders, then ~28 GB for the DiT):
  python diag_text_bf16.py --dit
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

MODEL = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"


def normed(x32, norm, wdt):
    """RMSNorm of an fp32 stream, output in the weight dtype (the fp32-residual variant)."""
    xn = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + norm.eps)
    return norm.weight * xn.to(wdt)


def attn_fp32_sdpa(a, x, cos, sin):
    B, L, _ = x.shape
    hd = a.head_dim
    q = a.q_norm(a.q_proj(x).view(B, L, a.heads, hd)).transpose(1, 2)
    k = a.k_norm(a.k_proj(x).view(B, L, a.kv_heads, hd)).transpose(1, 2)
    v = a.v_proj(x).view(B, L, a.kv_heads, hd).transpose(1, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    rep = a.heads // a.kv_heads
    k = k[:, :, None].expand(B, a.kv_heads, rep, L, hd).reshape(B, a.heads, L, hd)
    v = v[:, :, None].expand(B, a.kv_heads, rep, L, hd).reshape(B, a.heads, L, hd)
    o = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True).to(x.dtype)
    return a.o_proj(o.transpose(1, 2).reshape(B, L, a.heads * hd))


def mlp_fp32_mid(m, x):
    h = F.silu(m.gate_proj(x).float()) * m.up_proj(x).float()
    return F.linear(h, m.down_proj.weight.float()).to(x.dtype)


def run_layers(model, ids, variant: str):
    """-> list of per-layer residual streams (fp32 copies), final entry = the output."""
    L = ids.shape[1]
    cos, sin = model.rope_cos[:L], model.rope_sin[:L]
    wdt = model.embed_tokens.weight.dtype
    x = model.embed_tokens(ids)
    if variant == "res32":
        x = x.float()
    outs = []
    for layer in model.layers:
        if variant == "res32":
            x = x + layer.self_attn(normed(x, layer.input_layernorm, wdt), cos, sin).float()
            x = x + layer.mlp(normed(x, layer.post_attention_layernorm, wdt)).float()
        else:
            h = layer.input_layernorm(x)
            a = attn_fp32_sdpa(layer.self_attn, h, cos, sin) if variant == "attn32" else layer.self_attn(h, cos, sin)
            x = x + a
            h = layer.post_attention_layernorm(x)
            x = x + (mlp_fp32_mid(layer.mlp, h) if variant == "mlp32" else layer.mlp(h))
        outs.append(x.float().clone())
    return outs


def tok_corr(a, b):
    a = a.double().numpy()
    b = b.double().numpy()
    return np.array([np.corrcoef(a[i], b[i])[0, 1] for i in range(a.shape[0])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default=str(HERE / "oracle" / "256"))
    ap.add_argument("--token", type=int, default=14)
    ap.add_argument("--variants", default="bf16,res32,attn32,mlp32")
    ap.add_argument("--dit", action="store_true")
    ap.add_argument("--dit-steps", default="0,20,39")
    args = ap.parse_args()
    od = Path(args.oracle)
    meta = json.load(open(od / "meta.json"))
    Lfull, drop = int(meta["Lfull"]), int(meta["drop_idx"])
    ids = torch.from_numpy(np.fromfile(od / "enc_input_ids.i32", "<i4").reshape(1, Lfull).copy())
    ref = torch.from_numpy(np.fromfile(od / "enc_hidden_full.f32", "<f4").reshape(Lfull, -1))
    tdir = Path(hf_snapshot(MODEL, revision=REVISION)) / "text_encoder"
    T = args.token

    with torch.no_grad():
        m32 = load_qi21_text(tdir, dtype=torch.float32)
        ref_layers = run_layers(m32, ids, "fp32")
        assert torch.equal(ref_layers[-1][0], ref), "fp32 layer loop != oracle"
        del m32
        print(f"[diag] fp32 layer loop == oracle bit-exact; token {T} id {int(ids[0, T])}", flush=True)
        print("[diag] fp32 residual |h| max per layer, token", T, ":",
              " ".join(f"{float(r[0, T].abs().max()):.0f}" for r in ref_layers), flush=True)
        print("[diag] fp32 residual |h| max per layer, token 0 :",
              " ".join(f"{float(r[0, 0].abs().max()):.0f}" for r in ref_layers), flush=True)

        mb = load_qi21_text(tdir, dtype=torch.bfloat16)
        finals = {}
        for v in args.variants.split(","):
            t0 = time.time()
            outs = run_layers(mb, ids, v)
            finals[v] = outs[-1][0]
            per_layer_T = [float(tok_corr(o[0, T:T + 1], r[0, T:T + 1])[0]) for o, r in zip(outs, ref_layers)]
            per_layer_rel = [float((o[0, T] - r[0, T]).norm() / r[0, T].norm()) for o, r in zip(outs, ref_layers)]
            tc = tok_corr(outs[-1][0], ref)
            others = np.delete(tc, T)
            print(f"[diag] {v:<6} ({time.time() - t0:.1f}s): final token {T} corr {tc[T]:.6f}  min other token corr "
                  f"{others.min():.6f}  global corr {np.corrcoef(outs[-1][0].double().numpy().ravel(), ref.double().numpy().ravel())[0, 1]:.6f}  "
                  f"n tokens < 0.999: {int((tc < 0.999).sum())}", flush=True)
            print(f"         token {T} corr by layer: " + " ".join(f"{c:.4f}" for c in per_layer_T), flush=True)
            print(f"         token {T} |d|/|ref| by layer: " + " ".join(f"{r:.1e}" for r in per_layer_rel), flush=True)
        del mb

    if args.dit:
        from qi21_dit import load_qi21_dit
        from qi21_host import build_inputs
        from qi21_oracle import Oracle, parse_steps
        o = Oracle(od)
        hf = HERE / "_work" / "hf_bf16_text" / od.name / "hidden.f32"
        if hf.exists():
            finals["hf_bf16"] = torch.from_numpy(np.fromfile(hf, "<f4").reshape(Lfull, -1))
        tdir_dit = Path(hf_snapshot(MODEL, revision=REVISION)) / "transformer"
        axes = tuple(json.load(open(tdir_dit / "config.json")).get("axes_dims_rope", (16, 56, 56)))
        t0 = time.time()
        dit = load_qi21_dit(tdir_dit, dtype=torch.float32)
        print(f"[diag] DiT fp32 loaded in {time.time() - t0:.0f}s", flush=True)
        with torch.no_grad():
            for s in parse_steps(args.dit_steps, o.steps):
                ref_v = o.vel(s).double().numpy().ravel()
                line = []
                for name in ["oracle"] + list(finals):
                    pe = o.prompt_embeds if name == "oracle" else finals[name][drop:][None]
                    out = dit(**build_inputs(o.latent(s), pe, o.t(s), o.H, o.W, axes_dim=axes)).double().numpy().ravel()
                    c = float(np.corrcoef(out, ref_v)[0, 1])
                    line.append(f"{name} {c:.6f}")
                print(f"[diag] DiT step {s:2d} vel corr vs oracle, by encoder hidden: " + "  ".join(line), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
