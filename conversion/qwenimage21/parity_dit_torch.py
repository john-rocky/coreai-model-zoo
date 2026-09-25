"""Gate: the plain-torch re-author (qi21_dit.py) + host RoPE (qi21_host.py) vs diffusers, fp32.

No checkpoint needed. Five checks, each prints PASS/RED with its numbers:

1. **Re-author vs diffusers** on a small random config (2 layers, 4 heads x 32, context 64,
   axes 4/14/14), both built from one random state_dict, ``L=13, H=W=4`` (N=16). diffusers is
   called the way ``QwenImage21Pipeline`` calls it (``img_shapes=[[(1,H,W)]]``, ``img_mask`` =
   L x False + N/4 x True, no text mask) and its output sliced to the last N rows.
   Bar: max|d| <= 1e-4 and corr >= 0.999999.
2. **Negative controls** — the same comparison with one deliberate bug each; every one must fail
   the bar (a gate that cannot go red is not a gate): (a) text/image RoPE tables swapped,
   (b) text tokens modulated from the sampled t instead of t=0, (c) bidirectional text attention,
   plus (d, extra) the image grid not centred on zero.
3. **diffusers prefix KV cache on vs off** (``extract`` then ``cached``) — the reason the graph
   may recompute the text prefix every step. Also the re-author vs the ``cached`` output.
4. **RoPE tables** vs ``QwenImage21Rope`` (the model's own ``pos_embed``): real/imag parts,
   max|d| <= 1e-6, at (L,H,W) = (13,4,4) and (40,16,16) (required) plus (18,16,24) and
   (512,64,64) (extra).
5. **bf16 cast keeps the timestep sinusoid fp32** — the export casts the module with
   ``.to(torch.bfloat16)``; the 256-d sinusoid of that module must equal the fp32 one exactly
   (the bf16-frequency version, which the old buffer produced, is the red control: at t = 1 it
   is corr 0.958).

Run (qi21 venv — the only one with diffusers main):
  ~/code/coreai/coreai-models/.venv-qi21/bin/python parity_dit_torch.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from qi21_dit import QI21DiT, rope_interleaved  # noqa: E402
from qi21_host import build_inputs, rope_tables  # noqa: E402

CFG = dict(patch_size=1, in_channels=64, out_channels=64, num_layers=2, attention_head_dim=32,
           num_attention_heads=4, context_in_dim=64, mlp_ratio=3, axes_dims_rope=(4, 14, 14),
           eps=1e-6, causal_condition=True)
BAR_MAXD, BAR_CORR = 1e-4, 0.999999
BAR_ROPE = 1e-6


def stats(a, b):
    a = a.detach().double().flatten()
    b = b.detach().double().flatten()
    c = float(np.corrcoef(a.numpy(), b.numpy())[0, 1])
    return float((a - b).abs().max()), c


def passes(maxd, corr):
    return maxd <= BAR_MAXD and corr >= BAR_CORR


def randomize(model, seed):
    """Unit-variance weights so every path (gates, modulation, attention) carries O(1) signal."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            if p.ndim == 2:
                p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(p.shape[1]))
            elif name.endswith("text_norm.weight"):          # zero-centred: effective scale w + 1
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:                                            # RMSNorm scale
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))


def ref_call(ref, img, txt, t, H, W, **kw):
    L, N = txt.shape[1], img.shape[1]
    mask = torch.cat([torch.zeros(L, dtype=torch.bool), torch.ones(N // 4, dtype=torch.bool)])[None]
    out = ref(hidden_states=img, encoder_hidden_states=txt, timestep=t, img_shapes=[[(1, H, W)]],
              img_mask=mask, return_dict=False, **kw)[0]
    return out[:, -N:]


# ------------------------------------------------------------------ negative controls
class TextFromT(QI21DiT):
    """(b) every token — text included — modulated from the sampled t row."""

    @staticmethod
    def _per_token(rows, L, N):
        D = rows.shape[-1]
        return rows[0:1].reshape(1, 1, D).expand(1, L + N, D)


class BidirText(QI21DiT):
    """(c) text queries see every text key (no causal triangle)."""

    def _attention(self, blk, h, L, cos, sin):
        a = blk.attn
        B, S, _ = h.shape
        q = a.norm_q(a.to_q(h).view(B, S, self.heads, self.dim_head))
        k = a.norm_k(a.to_k(h).view(B, S, self.heads, self.dim_head))
        v = a.to_v(h).view(B, S, self.heads, self.dim_head)
        q = rope_interleaved(q, cos, sin, self.rope_fp32).to(v.dtype)
        k = rope_interleaved(k, cos, sin, self.rope_fp32).to(v.dtype)
        qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))
        out_txt = F.scaled_dot_product_attention(qt[:, :, :L], kt[:, :, :L], vt[:, :, :L])
        out_img = F.scaled_dot_product_attention(qt[:, :, L:], kt, vt)
        o = torch.cat([out_txt, out_img], dim=2).transpose(1, 2).reshape(B, S, self.heads * self.dim_head)
        return a.to_out[0](o)


def swapped_rope(ins):
    """(a) the joint table built image-first, then split at L: tables land on the wrong tokens."""
    L = ins["txt_cos"].shape[1]
    out = dict(ins)
    for kind in ("cos", "sin"):
        joint = torch.cat([ins[f"img_{kind}"], ins[f"txt_{kind}"]], dim=1)
        out[f"txt_{kind}"], out[f"img_{kind}"] = joint[:, :L], joint[:, L:]
    return out


def uncentred_rope(ins, L, H, W, axes):
    """(d) image grid at h in range(0, H), w in range(0, W) instead of centred on zero."""
    from qi21_host import _axis_table
    txt = torch.arange(L)
    pos = (torch.cat([txt, torch.full((H * W,), L)]),
           torch.cat([txt, torch.arange(H).repeat_interleave(W)]),
           torch.cat([txt, torch.arange(W).repeat(H)]))
    cs = [_axis_table(p, d) for p, d in zip(pos, axes)]
    cos = torch.cat([c for c, _ in cs], -1)[None]
    sin = torch.cat([s for _, s in cs], -1)[None]
    out = dict(ins)
    out["txt_cos"], out["img_cos"] = cos[:, :L], cos[:, L:]
    out["txt_sin"], out["img_sin"] = sin[:, :L], sin[:, L:]
    return out


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=str(HERE / "_work" / "parity_dit_torch.json"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    import diffusers
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21KVCache, QwenImage21Rope, QwenImage21Transformer2DModel)
    print(f"[parity] diffusers {diffusers.__version__}  torch {torch.__version__}", flush=True)
    res = {"diffusers": diffusers.__version__, "torch": torch.__version__, "cfg": CFG}
    ok_all = True

    ref = QwenImage21Transformer2DModel(**CFG).eval()
    randomize(ref, args.seed)
    mine = QI21DiT.from_config(CFG).eval()
    mine.load_state_dict(ref.state_dict(), strict=True)
    axes = CFG["axes_dims_rope"]

    L, H, W = 13, 4, 4
    N = H * W
    g = torch.Generator().manual_seed(args.seed + 1)
    img = torch.randn(1, N, CFG["in_channels"], generator=g)
    txt = torch.randn(1, L, CFG["context_in_dim"], generator=g)
    t = torch.tensor([0.6180339], dtype=torch.float32)
    ins = build_inputs(img, txt, t, H, W, axes_dim=axes)

    with torch.no_grad():
        # ---- 1. re-author vs diffusers
        r = ref_call(ref, img, txt, t, H, W)
        o = mine(**ins)
        maxd, c = stats(o, r)
        ok = passes(maxd, c)
        ok_all &= ok
        res["reauthor_vs_diffusers"] = dict(L=L, H=H, W=W, maxd=maxd, corr=c, max_ref=float(r.abs().max()),
                                            ok=ok)
        print(f"1  re-author vs diffusers  L={L} N={N}: max|d| {maxd:.3e}  corr {c:.9f}  "
              f"(|ref| max {float(r.abs().max()):.3f})  {'PASS' if ok else 'FAIL'}", flush=True)

        # ---- 2. negative controls (must be RED)
        controls = {}
        sd = ref.state_dict()
        tf = TextFromT.from_config(CFG).eval()
        tf.load_state_dict(sd, strict=True)
        bd = BidirText.from_config(CFG).eval()
        bd.load_state_dict(sd, strict=True)
        cases = [("a_rope_swapped", lambda: mine(**swapped_rope(ins))),
                 ("b_text_mod_from_t", lambda: tf(**ins)),
                 ("c_text_bidirectional", lambda: bd(**ins)),
                 ("d_extra_uncentred_grid", lambda: mine(**uncentred_rope(ins, L, H, W, axes)))]
        for name, fn in cases:
            md, cc = stats(fn(), r)
            red = not passes(md, cc)
            ok_all &= red
            controls[name] = dict(maxd=md, corr=cc, red=red)
            print(f"2{name[0]} control {name[2:]:<22}: max|d| {md:.3e}  corr {cc:.9f}  "
                  f"{'RED (good)' if red else 'NOT RED (gate blind)'}", flush=True)
        res["negative_controls"] = controls

        # ---- 3. diffusers KV cache on vs off (+ re-author vs cached)
        img2 = torch.randn(1, N, CFG["in_channels"], generator=g)
        t2 = torch.tensor([0.25], dtype=torch.float32)
        kv = QwenImage21KVCache(CFG["num_layers"])
        oA = ref_call(ref, img, txt, t, H, W, kv_cache=kv, kv_cache_mode="extract")
        oB = ref_call(ref, img2, txt, t2, H, W, kv_cache=kv, kv_cache_mode="cached")
        nB = ref_call(ref, img2, txt, t2, H, W)
        mB = mine(**build_inputs(img2, txt, t2, H, W, axes_dim=axes))
        kvA, kvB, mcB = stats(oA, r), stats(oB, nB), stats(mB, oB)
        ok = passes(*kvA) and passes(*kvB) and passes(*mcB)
        ok_all &= ok
        res["kv_cache"] = dict(extract_vs_nocache=kvA, cached_vs_nocache=kvB, reauthor_vs_cached=mcB, ok=ok)
        print(f"3  diffusers kv extract vs off: max|d| {kvA[0]:.3e} corr {kvA[1]:.9f} | cached vs off: "
              f"max|d| {kvB[0]:.3e} corr {kvB[1]:.9f} | re-author vs cached: max|d| {mcB[0]:.3e} "
              f"corr {mcB[1]:.9f}  {'PASS' if ok else 'FAIL'}", flush=True)

        # ---- 4. RoPE tables vs QwenImage21Rope
        rope_res = []
        full_rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
        for (Lr, Hr, Wr, req) in [(13, 4, 4, True), (40, 16, 16, True), (18, 16, 24, False),
                                  (512, 64, 64, False)]:
            pad = torch.cat([torch.zeros(Lr, dtype=torch.bool), torch.ones(Hr * Wr, dtype=torch.bool)])
            f = full_rope([(1, Hr, Wr)], pad, torch.device("cpu"))            # complex [S, 64]
            tc, ts, ic, is_ = rope_tables(Lr, Hr, Wr)
            dc = float((torch.cat([tc, ic], 1)[0] - f.real).abs().max())
            ds = float((torch.cat([ts, is_], 1)[0] - f.imag).abs().max())
            ok = max(dc, ds) <= BAR_ROPE
            if req:
                ok_all &= ok
            rope_res.append(dict(L=Lr, H=Hr, W=Wr, required=req, maxd_cos=dc, maxd_sin=ds, ok=ok))
            print(f"4  rope L={Lr} H={Hr} W={Wr}{'' if req else ' (extra)'}: max|d| cos {dc:.3e} "
                  f"sin {ds:.3e}  {'PASS' if ok else 'FAIL'}", flush=True)
        # the small config's tables (used by check 1) against its own pos_embed
        pad = torch.cat([torch.zeros(L, dtype=torch.bool), torch.ones(N, dtype=torch.bool)])
        f = ref.pos_embed([(1, H, W)], pad, torch.device("cpu"))
        dsm = max(float((torch.cat([ins["txt_cos"], ins["img_cos"]], 1)[0] - f.real).abs().max()),
                  float((torch.cat([ins["txt_sin"], ins["img_sin"]], 1)[0] - f.imag).abs().max()))
        rope_res.append(dict(L=L, H=H, W=W, axes=list(axes), required=False, maxd=dsm, ok=dsm <= BAR_ROPE))
        print(f"4  rope small-config axes {axes} L={L} H={H} W={W}: max|d| {dsm:.3e}", flush=True)
        res["rope"] = rope_res

        # ---- 5. bf16 cast keeps the sinusoid fp32 (+ the bf16-frequency red control)
        m16 = QI21DiT.from_config(CFG).to(torch.bfloat16)
        tt = torch.tensor([1.0, 0.5823754, 0.02])
        args32 = (1000.0 * tt)[:, None] * mine.time_text_embed.freqs[None]
        args16 = (1000.0 * tt)[:, None] * m16.time_text_embed.freqs[None]
        argsbf = (1000.0 * tt)[:, None] * mine.time_text_embed.freqs.to(torch.bfloat16).float()[None]
        emb = lambda a: torch.cat([a.cos(), a.sin()], -1)          # noqa: E731
        d16 = float((emb(args16) - emb(args32)).abs().max())
        cbf = min(stats(emb(argsbf)[i], emb(args32)[i])[1] for i in range(3))
        ok = m16.time_text_embed.freqs.dtype == torch.float32 and d16 == 0.0
        red = not passes(float((emb(argsbf) - emb(args32)).abs().max()), cbf)
        ok_all &= ok and red
        res["sinusoid_bf16_cast"] = dict(freqs_dtype=str(m16.time_text_embed.freqs.dtype), maxd=d16, ok=ok,
                                         control_bf16_freqs_min_corr=cbf, control_red=red)
        print(f"5  bf16-cast sinusoid: freqs {m16.time_text_embed.freqs.dtype}, max|d| vs fp32 {d16:.1e}  "
              f"{'PASS' if ok else 'FAIL'} | control bf16 freqs: min corr {cbf:.6f} "
              f"{'RED (good)' if red else 'NOT RED'}", flush=True)

    res["ok"] = bool(ok_all)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.json, "w"), indent=2)
    print(f"\n[parity] {'ALL PASS' if ok_all else 'FAIL'}  -> {args.json}", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
