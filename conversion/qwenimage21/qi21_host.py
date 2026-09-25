"""Host-side helpers for the Qwen-Image-2.1 DiT graph (numpy/torch only — no diffusers).

Everything the app computes around one ``main`` call of the exported DiT:

* ``rope_tables(L, H, W)`` — the 3-axis RoPE as real cos/sin tables for ``[text L | image N]``.
  Text token ``i`` sits at ``(i, i, i)``; image token ``(h, w)`` at ``(L, h', w')`` with
  ``h' in range(-(H - H//2), H//2)`` (the grid centred on zero, raster order) and the same for
  ``w'``. Frequencies are ``QwenImage21Rope.rope_params`` (theta 10000, axes 16/56/56 ->
  8 + 28 + 28 = 64 pairs), concatenated in axis order; cos = real part, sin = imaginary part.
  The reference looks negative positions up in a separate table built from
  ``arange(1024).flip(0) * -1 - 1``; ``outer(pos, inv_freq)`` of the negative position itself is
  the same number, and ``torch.polar`` evaluates cos/sin per element, so the tables are
  bit-identical to the reference (gated in ``parity_dit_torch.py``).
* ``pack`` / ``unpack`` — the pipeline's plain spatial flatten (one token per 16×16 px tile).
* ``build_inputs`` — the graph's input dict, fp32.
"""
from __future__ import annotations

import numpy as np
import torch

THETA = 10000
AXES_DIM = (16, 56, 56)


def _inv_freq(dim: int, theta: int = THETA) -> torch.Tensor:
    # Same expression as QwenImage21Rope.rope_params (fp32 throughout).
    return 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim))


def _axis_table(pos: torch.Tensor, dim: int, theta: int = THETA):
    f = torch.polar(torch.ones(pos.shape[0], dim // 2), torch.outer(pos, _inv_freq(dim, theta)))
    return f.real, f.imag


def rope_positions(L: int, H: int, W: int):
    """(frame, height, width) position of every token of ``[text L | image H*W]``, int64."""
    txt = torch.arange(L, dtype=torch.long)
    hs = torch.arange(-(H - H // 2), H // 2, dtype=torch.long)
    ws = torch.arange(-(W - W // 2), W // 2, dtype=torch.long)
    img_f = torch.full((H * W,), L, dtype=torch.long)
    img_h = hs.repeat_interleave(W)
    img_w = ws.repeat(H)
    return (torch.cat([txt, img_f]), torch.cat([txt, img_h]), torch.cat([txt, img_w]))


def rope_tables(L: int, H: int, W: int, axes_dim=AXES_DIM, theta: int = THETA):
    """-> (txt_cos, txt_sin, img_cos, img_sin), fp32, ``[1, L, P]`` / ``[1, H*W, P]``, P = sum(axes)/2."""
    cos, sin = [], []
    for pos, dim in zip(rope_positions(L, H, W), axes_dim):
        c, s = _axis_table(pos, dim, theta)
        cos.append(c)
        sin.append(s)
    cos = torch.cat(cos, dim=-1)[None].contiguous()
    sin = torch.cat(sin, dim=-1)[None].contiguous()
    return cos[:, :L], sin[:, :L], cos[:, L:], sin[:, L:]


def pack(latent):
    """VAE latent ``[1, C, 1, H, W]`` -> DiT tokens ``[1, H*W, C]`` (raster order)."""
    b, c, _, h, w = latent.shape
    return latent.reshape(b, c, h * w).transpose(1, 2)


def unpack(vel, H: int, W: int):
    """DiT output ``[1, H*W, C]`` -> ``[1, C, 1, H, W]``."""
    b, _, c = vel.shape
    return vel.transpose(1, 2).reshape(b, c, 1, H, W)


def _f32(x) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(x)) if not isinstance(x, torch.Tensor) else x
    return t.detach().to(torch.float32).contiguous()


def build_inputs(latent_packed, txt_feats, t, H: int, W: int, axes_dim=AXES_DIM) -> dict:
    """Graph inputs for one step. ``t`` = the scheduler timestep / 1000 (what the transformer
    sees). Keys are the forward-argument names of ``QI21DiT``."""
    img = _f32(latent_packed)
    txt = _f32(txt_feats)
    assert img.shape[1] == H * W, (img.shape, H, W)
    L = txt.shape[1]
    tc, ts, ic, is_ = rope_tables(L, H, W, axes_dim)
    return {
        "img_tokens": img,
        "txt_feats": txt,
        "timestep": _f32(t).reshape(1),
        "txt_cos": tc, "txt_sin": ts,
        "img_cos": ic, "img_sin": is_,
    }
