"""The Neural Engine form of the Timestral graph (`--form ane` in export_timestral.py).

TimestralCore (timestral_core.py) with four rewrites, each exact in exact arithmetic (in torch fp32 on the fixture the
rewritten graphs stay within 1.0e-4 to 1.5e-4 of the released module, as the plain re-author does; the L1 bar is 0.129)
and each kept because a measurement on the Neural Engine asked for it (knowledge/timestral-port.md):

  p7    RMSNorm on x / 2^7 with eps / 4^7 (timestral_core's "prescaled" mode): the fp16 sum of squares of the
        768-wide residual reaches 1.6e8 in fp32, far above fp16's 65,504
  qkv4  the fused wQKV as three Linears (Q, K, V, the rows of the reordered fused weight) and the heads as rank-4
        tensors (R, heads, S, head_dim)
  tanh  every sigmoid gate and SiLU as 0.5 * tanh(x / 2) + 0.5: the Neural Engine's sigmoid / SiLU is what kept the
        graph 0.09 % above the line on one fixture group (the patch embedding alone: 3.4 % off with it, 0.19 % through
        tanh)
  conv  every Linear as a 1x1 Conv2d over (R, C, 1, S): with Linear projections the Neural Engine compiler aborts on
        an R = 2 graph (`SetDimensionOrderHint`, ZinIrTensor.cpp:986); the numbers are bit-identical to the Linears

The whole graph computes in fp16, the attention scores included. That is right on the Neural Engine only: the same
graph on the GPU overflows (scores reach |q.k/8| = 2.4e5) and its output saturates at +/-65,504. Compile it with
`--preferred-compute neural-engine` and check the placement by the output, not by a region's name.
"""
import torch
import torch.nn as nn

from timestral_core import (Linear, MASK_FILL, ResidualBlock, SwiGLU, VariateProjection, const, load_core,
                            rotate_half)


def sigmoid_tanh(x, half):
    """sigmoid(x) = 0.5 * tanh(x / 2) + 0.5, the constant a buffer in the module's dtype."""
    return half * torch.tanh(half * x) + half


class TimeAttentionQKV(nn.Module):
    """timestral_core.TimeAttention with Q / K / V as three Linears and rank-4 heads (same weights, same function)."""

    def __init__(self, src, tanh):
        super().__init__()
        d = src.wQKV.in_features
        self.heads, self.hd, self.tanh = src.heads, src.hd, tanh
        w = src.wQKV.weight.data  # rows already in (qkv, heads, head_dim) order
        self.wq, self.wk, self.wv = (Linear(d, d, bias=False) for _ in range(3))
        self.wq.weight.data = w[:d].clone()
        self.wk.weight.data = w[d:2 * d].clone()
        self.wv.weight.data = w[2 * d:].clone()
        self.wO, self._gate = src.wO, src._gate
        for name in ("qscale", "thresh", "fill"):
            self.register_buffer(name, getattr(src, name).clone(), persistent=False)
        self.register_buffer("c_half", const(0.5), persistent=False)

    def forward(self, x, cos, sin, attn_mask):
        r, s, d = x.shape
        q = self.wq(x).reshape(r, s, self.heads, self.hd).transpose(1, 2)  # (R, heads, S, hd)
        k = self.wk(x).reshape(r, s, self.heads, self.hd).transpose(1, 2)
        v = self.wv(x).reshape(r, s, self.heads, self.hd).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        allowed = attn_mask > self.thresh
        scores = torch.matmul(q * self.qscale, k.transpose(-1, -2))
        att = torch.softmax(torch.where(allowed, scores, self.fill), dim=-1)
        out = torch.matmul(att, v).transpose(1, 2).reshape(r, s, d)
        gate = self._gate(x)
        return self.wO(out * (sigmoid_tanh(gate, self.c_half) if self.tanh else torch.sigmoid(gate)))


class VariateProjectionT(nn.Module):
    def forward(self, x):
        return self.wO(self.wV(x) * sigmoid_tanh(self._gate(x), self.c_half))


class SwiGLUT(nn.Module):
    def forward(self, x):
        gu = self.w12(x)
        up = gu[..., self.hidden:]
        return self.w3(gu[..., : self.hidden] * (up * sigmoid_tanh(up, self.c_half)))


class ResidualBlockT(nn.Module):
    def forward(self, x):
        h = self.hidden_layer(x)
        return self.output_layer(h * sigmoid_tanh(h, self.c_half)) + self.residual_layer(x)


_TANH = {VariateProjection: VariateProjectionT, SwiGLU: SwiGLUT, ResidualBlock: ResidualBlockT}


def use_tanh(core):
    """Every sigmoid gate and SiLU outside the attention through tanh (in place; the attention takes `tanh` itself)."""
    for m in core.modules():
        if type(m) in _TANH:
            m.register_buffer("c_half", const(0.5), persistent=False)
            m.__class__ = type(f"{type(m).__name__}T", (_TANH[type(m)], type(m)), {})
    return core


class ConvLinear(nn.Module):
    """A Linear as a 1x1 Conv2d over (R, C, 1, S); (R, S, C) in and out, the same weights."""

    def __init__(self, lin):
        super().__init__()
        o, i = lin.weight.shape
        self.conv = nn.Conv2d(i, o, 1, bias=lin.bias is not None)
        self.conv.weight.data = lin.weight.data.view(o, i, 1, 1).clone()
        if lin.bias is not None:
            self.conv.bias.data = lin.bias.data.clone()

    def forward(self, x):
        return self.conv(x.transpose(-1, -2).unsqueeze(-2)).squeeze(-2).transpose(-1, -2).contiguous()


def use_conv(module):
    """Replace every timestral_core.Linear under `module` by ConvLinear (in place)."""
    for name, child in list(module.named_children()):
        if isinstance(child, Linear):
            setattr(module, name, ConvLinear(child))
        else:
            use_conv(child)
    return module


def load_ane_core(st_path, horizon_patches, tanh=True):
    """The `ane` form as an fp16 module: load_core in the p7 RMSNorm mode, then qkv4, tanh (unless tanh=False, the
    sigmoid control that misses L3 on the Neural Engine), conv, and the cast to fp16."""
    core = load_core(st_path, horizon_patches, norm_mode="prescaled", prescale_log2=7,
                     out_clamp=float(torch.finfo(torch.float16).max), mask_fill=MASK_FILL[torch.float16])
    for blk in core.blocks:
        blk.time_attn = TimeAttentionQKV(blk.time_attn, tanh)
    if tanh:
        use_tanh(core)
    use_conv(core)
    return core.half().eval()
