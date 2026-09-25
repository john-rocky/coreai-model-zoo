"""Qwen-Image-2.1 DiT, re-authored in plain torch for the Core AI export.

Mirrors ``diffusers.models.transformers.transformer_qwenimage21`` (main @ 4295ee3) module
by module, with the SAME parameter names, so ``transformer/diffusion_pytorch_model*.safetensors``
loads with ``strict=True``. It differs only in what the export needs:

* **Text-to-image, batch 1, no condition images.** The joint sequence is
  ``[text L | image N]`` in that order (the VLM text tokens, then the target image tokens
  in raster order — one token per 16×16 px tile, no 2×2 packing at this boundary).
* **Block-causal attention as two SDPA calls** per block — text queries attend causally to
  text keys, image queries attend to every key — which is exactly the reference's
  ``QwenImage21AttnProcessor`` segment decomposition (one text segment + the target), so no
  attention mask enters the graph. The 2.1 prefix KV cache is dropped: recomputing the
  text prefix every step is mathematically identical because text tokens modulate from
  ``t = 0`` and never attend to the image.
* **RoPE as real interleaved pairs** fed precomputed cos/sin (the reference's
  ``view_as_complex`` form is export-hostile). Host builds the tables (``qi21_host.py``).
* **The shared modulation** (one Linear for all 32 blocks) is evaluated once per forward
  for the two rows the reference uses — the sampled ``t`` (image tokens) and ``t = 0``
  (text tokens) — and broadcast per token by concatenation, never by ``torch.where``.
* Output is ``proj_out(norm_out(x))`` over the **image tokens only** (the reference computes
  text rows too and the pipeline slices them off).

Graph contract (``forward``):
    img_tokens [1,N,64]  txt_feats [1,L,4096]  timestep [1] (t/1000)
    txt_cos/txt_sin [1,L,64]  img_cos/img_sin [1,N,64]     (hd/2 = 64 pairs; 8+28+28 axes)
    -> vel [1,N,64]
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- norms
class RMSNorm(nn.Module):
    """diffusers RMSNorm (weight, no bias): fp32 statistics, then weight in its own dtype."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xn.to(self.weight.dtype) * self.weight


class ZeroCenterRMSNorm(nn.Module):
    """RMSNorm whose stored weight is ``scale - 1`` (effective scale ``weight + 1``), fp32."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        xf = x.float()
        rrms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * rrms * (self.weight.float() + 1)).to(dt)


# ------------------------------------------------------------------------- sub-modules
class TextProjection(nn.Module):
    def __init__(self, context_in_dim: int, hidden: int, eps: float):
        super().__init__()
        self.text_norm = ZeroCenterRMSNorm(context_in_dim, eps)
        self.in_layer = nn.Linear(context_in_dim, hidden, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.out_layer = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.out_layer(self.act(self.in_layer(self.text_norm(x))))


class TimestepEmbedder(nn.Module):
    """diffusers ``TimestepEmbedding(256, dim, sample_proj_bias=False)``."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(256, dim, bias=False)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear_2(self.act(self.linear_1(x)))


class TimeTextEmbed(nn.Module):
    """``QwenImage21TimestepProjEmbeddings``: sinusoid(256, cos|sin) -> MLP.

    The reference casts ``timestep`` to the model dtype BEFORE the sinusoid (a bf16 model
    rounds t to 8 mantissa bits). We keep the sinusoid in fp32 and cast the 256-vector — the
    fp32 oracle is the target, and this is strictly closer to it.

    ``freqs`` is a plain fp32 attribute, NOT a buffer, so ``model.to(torch.bfloat16)`` cannot
    round it: bf16 frequencies put up to 1.39 rad of phase error on ``1000 * t * freq`` at t = 1
    (sinusoid corr 0.958 vs fp32; measured 2026-09-25, round 1).
    """

    def __init__(self, dim: int):
        super().__init__()
        half = 128
        self.freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device="cpu") / half)
        self.timestep_embedder = TimestepEmbedder(dim)

    def forward(self, timestep_f32, dtype):
        args = (1000.0 * timestep_f32.float())[:, None] * self.freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)   # [B,256]
        return self.timestep_embedder(emb.to(dtype))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.proj = nn.Linear(dim, hidden, bias=False)
        self.out = nn.Linear(hidden, dim, bias=False)
        self.gate_layer = nn.Linear(dim, hidden, bias=False)
        self.activation_fn = nn.SiLU()

    def forward(self, x):
        return self.out(self.activation_fn(self.gate_layer(x)) * self.proj(x))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, eps: float):
        super().__init__()
        self.heads, self.dim_head = heads, dim_head
        self.to_q = nn.Linear(dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(dim, heads * dim_head, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(heads * dim_head, dim, bias=False)])
        self.norm_q = RMSNorm(dim_head, eps)
        self.norm_k = RMSNorm(dim_head, eps)


class AdaLayerNormOut(nn.Module):
    """``QwenImage21AdaLayerNormContinuous``: scale only, ``norm(x) * (1 + scale)``."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine=False, bias=False)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_ratio: int, eps: float):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(dim, heads, dim_head, eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = SwiGLU(dim, dim * mlp_ratio)


def rope_interleaved(x, cos, sin, fp32: bool):
    """x [B,S,H,hd]; cos/sin [B,S,1,hd/2]. out[2i] = x[2i]c - x[2i+1]s, out[2i+1] = x[2i]s + x[2i+1]c.

    Bit-equivalent to the reference ``view_as_complex(x.float().reshape(..,-1,2)) * polar``.
    """
    dt = x.dtype
    xi = x.float() if fp32 else x
    c = cos if fp32 else cos.to(dt)
    s = sin if fp32 else sin.to(dt)
    xp = xi.reshape(*xi.shape[:-1], xi.shape[-1] // 2, 2)
    x0, x1 = xp[..., 0], xp[..., 1]
    o = torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1).reshape(xi.shape)
    return o.to(dt)


# --------------------------------------------------------------------------- the model
class QI21DiT(nn.Module):
    def __init__(self, num_layers=32, dim=4096, heads=32, dim_head=128, in_channels=64,
                 out_channels=64, context_in_dim=4096, mlp_ratio=3, eps=1e-6,
                 io_fp32=False, rope_fp32=True):
        super().__init__()
        self.dim, self.heads, self.dim_head = dim, heads, dim_head
        self.io_fp32, self.rope_fp32 = io_fp32, rope_fp32
        self.img_in = nn.Linear(in_channels, dim, bias=False)
        self.txt_in = TextProjection(context_in_dim, dim, eps)
        self.time_text_embed = TimeTextEmbed(dim)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim, bias=False))
        self.transformer_blocks = nn.ModuleList(
            [Block(dim, heads, dim_head, mlp_ratio, eps) for _ in range(num_layers)])
        self.norm_out = AdaLayerNormOut(dim, eps)
        self.proj_out = nn.Linear(dim, out_channels, bias=False)

    @classmethod
    def from_config(cls, cfg: dict, **kw):
        return cls(num_layers=cfg["num_layers"], dim=cfg["num_attention_heads"] * cfg["attention_head_dim"],
                   heads=cfg["num_attention_heads"], dim_head=cfg["attention_head_dim"],
                   in_channels=cfg["in_channels"], out_channels=cfg["out_channels"],
                   context_in_dim=cfg["context_in_dim"], mlp_ratio=cfg["mlp_ratio"], eps=cfg["eps"], **kw)

    # ---- pieces -------------------------------------------------------------------
    def _mod_rows(self, timestep, dtype):
        """Rows [t, 0] -> (scale1, gate1, scale2, gate2, scale_out), each [2, D] (row 0 = t)."""
        t2 = torch.cat([timestep.reshape(1).float(), timestep.new_zeros(1, dtype=torch.float32)])
        temb = self.time_text_embed(t2, dtype)                    # [2, D]
        mod = self.modulation(temb)                                # [2, 4D]
        s1, g1, s2, g2 = mod.chunk(4, dim=-1)
        so = self.norm_out.linear(self.norm_out.silu(temb))        # [2, D]
        return s1, g1.tanh(), s2, g2.tanh(), so

    @staticmethod
    def _per_token(rows, L, N):
        """rows [2, D] -> [1, L+N, D]: text tokens take row 1 (t=0), image tokens row 0 (t)."""
        D = rows.shape[-1]
        return torch.cat([rows[1:2].reshape(1, 1, D).expand(1, L, D),
                          rows[0:1].reshape(1, 1, D).expand(1, N, D)], dim=1)

    def _attention(self, blk, h, L, cos, sin):
        a = blk.attn
        B, S, _ = h.shape
        q = a.norm_q(a.to_q(h).view(B, S, self.heads, self.dim_head))
        k = a.norm_k(a.to_k(h).view(B, S, self.heads, self.dim_head))
        v = a.to_v(h).view(B, S, self.heads, self.dim_head)
        q = rope_interleaved(q, cos, sin, self.rope_fp32).to(v.dtype)
        k = rope_interleaved(k, cos, sin, self.rope_fp32).to(v.dtype)
        qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))                 # [B,H,S,hd]
        out_txt = F.scaled_dot_product_attention(qt[:, :, :L], kt[:, :, :L], vt[:, :, :L], is_causal=True)
        out_img = F.scaled_dot_product_attention(qt[:, :, L:], kt, vt)
        o = torch.cat([out_txt, out_img], dim=2).transpose(1, 2).reshape(B, S, self.heads * self.dim_head)
        return a.to_out[0](o)

    # ---- forward -------------------------------------------------------------------
    def forward(self, img_tokens, txt_feats, timestep, txt_cos, txt_sin, img_cos, img_sin):
        wdt = self.img_in.weight.dtype
        if self.io_fp32:
            img_tokens, txt_feats = img_tokens.to(wdt), txt_feats.to(wdt)
        L, N = txt_feats.shape[1], img_tokens.shape[1]
        cos = torch.cat([txt_cos, img_cos], dim=1).unsqueeze(2)        # [1,S,1,64] fp32
        sin = torch.cat([txt_sin, img_sin], dim=1).unsqueeze(2)
        if not self.rope_fp32:
            cos, sin = cos.to(wdt), sin.to(wdt)

        s1, g1, s2, g2, so = self._mod_rows(timestep, wdt)
        s1, g1, s2, g2 = (self._per_token(r, L, N) for r in (s1, g1, s2, g2))

        x = torch.cat([self.txt_in(txt_feats), self.img_in(img_tokens)], dim=1)   # [1,S,D]
        for blk in self.transformer_blocks:
            h = blk.img_norm1(x) * (1 + s1)
            x = x + g1 * self._attention(blk, h, L, cos, sin)
            h = blk.img_norm2(x) * (1 + s2)
            x = x + g2 * blk.img_mlp(h)

        xi = x[:, L:]
        out = self.proj_out(self.norm_out.norm(xi) * (1 + so[0:1].reshape(1, 1, -1)))
        return out.float() if self.io_fp32 else out


# --------------------------------------------------------------------------- weights
def load_qi21_dit(transformer_dir: str | Path, dtype=torch.bfloat16, n_layers: int | None = None,
                  **kw) -> QI21DiT:
    """Build from ``transformer/config.json`` and load the sharded safetensors (strict).

    The module is built on the meta device and the checkpoint tensors are assigned, so peak
    memory is one copy of the weights in ``dtype`` (no fp32 random init of 7B parameters first).
    """
    from safetensors import safe_open
    d = Path(transformer_dir)
    cfg = json.load(open(d / "config.json"))
    if n_layers is not None:
        cfg = dict(cfg, num_layers=n_layers)
    with torch.device("meta"):
        m = QI21DiT.from_config(cfg, **kw)
    index = d / "diffusion_pytorch_model.safetensors.index.json"
    want = set(m.state_dict().keys())
    if index.exists():
        idx = json.load(open(index))["weight_map"]
    else:                                   # single-file checkpoint (tiny test pipelines)
        with safe_open(d / "diffusion_pytorch_model.safetensors", framework="pt") as fh:
            idx = {k: "diffusion_pytorch_model.safetensors" for k in fh.keys()}
    sd = {}
    by_file: dict[str, list[str]] = {}
    for k, f in idx.items():
        if k in want:
            by_file.setdefault(f, []).append(k)
    for f, keys in by_file.items():
        with safe_open(d / f, framework="pt") as fh:
            for k in keys:
                sd[k] = fh.get_tensor(k).to(dtype)
    missing, unexpected = m.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected, unexpected
    assert not missing, missing
    assert not any(p.is_meta for p in m.parameters())
    assert m.time_text_embed.freqs.dtype == torch.float32 and not m.time_text_embed.freqs.is_meta
    return m.to(dtype).eval()
