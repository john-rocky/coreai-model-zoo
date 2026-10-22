"""Re-authored, torch.export-clean Timestral graph core (univariate, one decode block).

Graph boundary = the transformer over [context patches ; forecast queries] plus the quantile head:

    patches (R,L,64) , cos (S,64) , sin (S,64) , attn_mask (1,1,S,S)        S = L + H
      -> patch embedding -> [Enc ; H forecast-query tokens] -> 24 blocks -> out_norm
      -> head on the H decoder tokens -> monotone quantiles (R, H*32, 21), normalized space

The host (host_forecast.py) does everything else: NaN -> mask, left padding, the causal float64
statistics and the asinh, positions -> cos/sin, the attention mask, denormalization, the advancing
refine frame, the cap, symmetric averaging and the non-negative clip.

The attention mask is 1 (attend) / 0 and the graph selects `where(mask, scores, fill)`: this
checkpoint's attention scores reach |q.k/8| ~ 2.4e5 on the fixture, so an additive finite mask
(TimesFM's -1e4) does not mask; the fill is -1e30 in fp32 and the dtype's most negative finite value
otherwise.

Univariate only. With one variate per series the variate attention has one key, its softmax is 1,
and the sublayer is wO(sigmoid(gate(n)) * Wv(n)). Symmetric averaging stacks [s, -s] as two series,
whose variate mask is the diagonal, so the same holds at R = 2.

Weights load from the released model.safetensors (strict: every tensor is used or named as unused).
The fused wQKV stores its outputs as (qkv, head_dim, n_heads); the loader reorders the rows to
(qkv, n_heads, head_dim) so the graph can split heads with a plain view.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CFG = dict(patch=32, d=768, heads=12, head_dim=64, mlp=2048, eps=1e-6, width_factor=3.0,
           quantiles=(0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                      0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99),
           schedule=("time", "time", "time", "time_variate") * 6, rope_theta=10000.0, rope_dim=32)
MASK_FILL = {torch.float32: -1.0e30, torch.float16: -65504.0}  # the score of a masked key


def const(value):
    """A one-element buffer, so a constant follows the module's dtype (`.half()`) instead of entering the
    graph as an fp32 literal (an fp32 constant keeps a graph off the Neural Engine)."""
    return torch.tensor([float(value)])


class RMSNorm(nn.Module):
    """`x * rsqrt(mean(x^2) + eps) * scale`.

    mode "fp32": compute in fp32 and cast back, as the released code does (`x.float()`).
    mode "native": compute in the input dtype.
    mode "prescaled": compute in the input dtype on x / 2**k, with eps / 4**k. Equal in exact
    arithmetic; keeps the fp16 sum of squares below 65504 when |x| is large.
    mode "layernorm": LayerNorm over [x, -x] (mean 0, variance mean(x^2)), first half. Equal in exact
    arithmetic; the Neural Engine runs LayerNorm with an fp32-accumulating kernel.
    """

    def __init__(self, dim, eps, mode="fp32", prescale_log2=0):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.dim, self.eps, self.mode, self.k = dim, eps, mode, prescale_log2
        s = 2.0 ** -prescale_log2
        self.register_buffer("eps_t", const(eps), persistent=False)
        self.register_buffer("pre_s", const(s), persistent=False)
        self.register_buffer("pre_eps", const(eps * s * s), persistent=False)

    def forward(self, x):
        if self.mode == "fp32":
            xf = x.float()
            return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.scale
        if self.mode == "layernorm":
            y = F.layer_norm(torch.cat([x, -x], dim=-1), (2 * self.dim,), eps=self.eps)
            return y[..., : self.dim] * self.scale
        if self.mode == "prescaled" and self.k:
            xs = x * self.pre_s
            return xs * torch.rsqrt((xs * xs).mean(-1, keepdim=True) + self.pre_eps) * self.scale
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps_t) * self.scale


class Linear(nn.Linear):
    """nn.Linear that casts its weight to the input dtype where it is used: nothing in an all-fp32 or
    all-fp16 graph; with fp16 weights and fp32 inputs it is fp16 storage with fp32 compute (fp16w32)."""

    def forward(self, x):
        w = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
        b = self.bias
        if b is not None and b.dtype != x.dtype:
            b = b.to(x.dtype)
        return F.linear(x, w, b)


def store_fp16_weights(core):
    """fp16 storage for every Linear (weights and biases); norms, the forecast query and constants stay fp32."""
    for m in core.modules():
        if isinstance(m, Linear):
            m.weight.data = m.weight.data.half()
            if m.bias is not None:
                m.bias.data = m.bias.data.half()
    return core


class ResidualBlock(nn.Module):
    def __init__(self, n_in, n_hid, n_out, bias):
        super().__init__()
        self.hidden_layer = Linear(n_in, n_hid, bias=bias)
        self.output_layer = Linear(n_hid, n_out, bias=bias)
        self.residual_layer = Linear(n_in, n_out, bias=bias)

    def forward(self, x):
        return self.output_layer(F.silu(self.hidden_layer(x))) + self.residual_layer(x)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class TimeAttention(nn.Module):
    """Self-attention over the time axis with partial RoPE (cos/sin from the host) and a sigmoid gate."""

    stats = None  # gate_core.Absmax puts a dict here to record the score absmax (never set for export)

    def __init__(self, d, heads, mask_fill, attn_fp32=False):
        super().__init__()
        self.heads, self.hd, self.attn_fp32 = heads, d // heads, attn_fp32
        self.wQKV = Linear(d, 3 * d, bias=False)  # rows already in (qkv, heads, head_dim) order
        self.wO = Linear(d, d, bias=False)
        self._gate = Linear(d, d, bias=False)
        self.register_buffer("qscale", const(1.0 / math.sqrt(self.hd)), persistent=False)  # 1/8, exact
        self.register_buffer("thresh", const(0.5), persistent=False)
        self.register_buffer("fill", const(mask_fill), persistent=False)

    def forward(self, x, cos, sin, attn_mask):
        r, s, d = x.shape
        qkv = self.wQKV(x).view(r, s, 3, self.heads, self.hd)
        q = qkv[:, :, 0].transpose(1, 2)  # (R, heads, S, hd)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        allowed = attn_mask > self.thresh
        if self.attn_fp32:  # fp16 q, k, v; the scores (|q.k/8| up to 2.4e5) and the softmax in fp32
            scores = torch.matmul((q * self.qscale).float(), k.float().transpose(-1, -2))
            att = torch.softmax(torch.where(allowed, scores, torch.full_like(scores, -1.0e30)), dim=-1)
            out = torch.matmul(att.to(v.dtype), v).transpose(1, 2).reshape(r, s, d)
            return self.wO(out * torch.sigmoid(self._gate(x)))
        scores = torch.matmul(q * self.qscale, k.transpose(-1, -2))
        if self.stats is not None:
            z = torch.zeros_like(scores)
            for key, part in (("scores", scores), ("scores_allowed", torch.where(allowed, scores, z)),
                              ("scores_masked", torch.where(allowed, z, scores))):
                name = f"{self.stats_name}.{key}"
                self.stats[name] = max(self.stats.get(name, 0.0), float(part.abs().max()))
        att = torch.softmax(torch.where(allowed, scores, self.fill), dim=-1)
        out = torch.matmul(att, v).transpose(1, 2).reshape(r, s, d)
        return self.wO(out * torch.sigmoid(self._gate(x)))


class VariateProjection(nn.Module):
    """The variate attention of a one-variate series: softmax over one key is 1."""

    def __init__(self, d):
        super().__init__()
        self.wV = Linear(d, d, bias=False)  # the V rows of the fused wQKV, reordered
        self.wO = Linear(d, d, bias=False)
        self._gate = Linear(d, d, bias=False)

    def forward(self, x):
        return self.wO(self.wV(x) * torch.sigmoid(self._gate(x)))


class SwiGLU(nn.Module):
    """`w3(gate * silu(up))` with `gate, up = chunk(w12(x), 2)` (the released order)."""

    def __init__(self, d, hidden):
        super().__init__()
        self.hidden = hidden
        self.w12 = Linear(d, 2 * hidden, bias=False)
        self.w3 = Linear(hidden, d, bias=False)

    def forward(self, x):
        gu = self.w12(x)
        return self.w3(gu[..., : self.hidden] * F.silu(gu[..., self.hidden:]))


class Block(nn.Module):
    def __init__(self, kind, cfg, norm_mode, prescale_log2, mask_fill, attn_fp32=False):
        super().__init__()
        d = cfg["d"]
        self.kind = kind
        self.time_norm = RMSNorm(d, cfg["eps"], norm_mode, prescale_log2)
        self.time_attn = TimeAttention(d, cfg["heads"], mask_fill, attn_fp32)
        if kind == "time_variate":
            self.var_norm = RMSNorm(d, cfg["eps"], norm_mode, prescale_log2)
            self.var_attn = VariateProjection(d)
        self.mlp_norm = RMSNorm(d, cfg["eps"], norm_mode, prescale_log2)
        self.mlp = SwiGLU(d, cfg["mlp"])

    def forward(self, x, cos, sin, attn_mask):
        x = x + self.time_attn(self.time_norm(x), cos, sin, attn_mask)
        if self.kind == "time_variate":
            x = x + self.var_attn(self.var_norm(x))
        return x + self.mlp(self.mlp_norm(x))


def monotone_matrix(levels, width_factor):
    """(Q, Q) matrix M with `quantiles = raw[..., m] + softplus(raw) @ M` (the released cumsum form)."""
    q = len(levels)
    m = list(levels).index(0.5)
    widths = [0.0] * q
    for i in range(m + 1, q):
        widths[i] = levels[i] - levels[i - 1]
    for i in range(m):
        widths[i] = levels[i + 1] - levels[i]
    inc = width_factor * torch.tensor(widths, dtype=torch.float32)  # the released float32 increment scale
    mat = torch.zeros(q, q, dtype=torch.float32)
    for i in range(q):  # output column i
        for j in range(q):
            if m < j <= i:
                mat[j, i] = inc[j]
            elif i <= j < m:
                mat[j, i] = -inc[j]
    return mat, m


class TimestralCore(nn.Module):
    """Exportable graph. Inputs: patches (R,L,64), cos/sin (S,64), attn_mask (1,1,S,S) 1 = attend / 0.
    Output: quantiles (R, H*32, 21), monotone, in the normalized (asinh) space of the last context patch."""

    def __init__(self, horizon_patches, cfg=CFG, norm_mode="fp32", prescale_log2=0, out_clamp=None,
                 mask_fill=MASK_FILL[torch.float32], attn_fp32=False):
        super().__init__()
        self.cfg, self.h = cfg, horizon_patches
        d, p, nq = cfg["d"], cfg["patch"], len(cfg["quantiles"])
        self.patch_proj = ResidualBlock(2 * p, 4 * d, d, bias=True)
        self.forecast_query = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([Block(k, cfg, norm_mode, prescale_log2, mask_fill, attn_fp32)
                                     for k in cfg["schedule"]])
        self.out_norm = RMSNorm(d, cfg["eps"], norm_mode, prescale_log2)
        self.head = ResidualBlock(d, 4 * d, p * nq, bias=False)
        mat, self.m = monotone_matrix(cfg["quantiles"], cfg["width_factor"])
        self.register_buffer("mono", mat, persistent=False)
        self.register_buffer("one", const(1.0), persistent=False)
        self.out_clamp = out_clamp  # None, or the finite bound of the export dtype
        self.register_buffer("bound", const(out_clamp or 0.0), persistent=False)

    def forward(self, patches, cos, sin, attn_mask):
        r = patches.shape[0]
        d, p = self.cfg["d"], self.cfg["patch"]
        x_enc = self.patch_proj(patches)
        x_dec = self.forecast_query.view(1, 1, d).expand(r, self.h, d)
        x = torch.cat([x_enc, x_dec], dim=1)
        for blk in self.blocks:
            x = blk(x, cos, sin, attn_mask)
        x = self.out_norm(x)
        raw = self.head(x[:, -self.h:, :]).reshape(r, self.h * p, len(self.cfg["quantiles"]))
        sp = F.relu(raw) + torch.log(self.one + torch.exp(-torch.abs(raw)))  # softplus, overflow-free
        out = raw[..., self.m:self.m + 1] + torch.matmul(sp, self.mono)
        if self.out_clamp is not None:
            out = torch.minimum(torch.maximum(out, -self.bound), self.bound)
        return out


def reorder_qkv_rows(w, heads, head_dim):
    """(3*d, d) rows (qkv, head_dim, heads) -> (qkv, heads, head_dim)."""
    d_in = w.shape[1]
    return w.view(3, head_dim, heads, d_in).permute(0, 2, 1, 3).reshape(3 * heads * head_dim, d_in)


def load_core(st_path, horizon_patches, qkv_layout="released", **kw):
    """Build TimestralCore from model.safetensors.

    qkv_layout "released" reorders the fused rows (correct); "naive" reads them as if they were
    (qkv, heads, head_dim) (the deliberately broken arm for the gate's red test).
    """
    from safetensors.torch import load_file
    sd = load_file(str(st_path))
    core = TimestralCore(horizon_patches, **kw)
    cfg = core.cfg
    heads, hd, d = cfg["heads"], cfg["head_dim"], cfg["d"]
    new, used = {}, set()

    def take(src, dst, tensor=None):
        new[dst] = sd[src] if tensor is None else tensor
        used.add(src)

    for n in ("hidden_layer", "output_layer", "residual_layer"):
        for t in ("weight", "bias"):
            take(f"representation.patch_proj.{n}.{t}", f"patch_proj.{n}.{t}")
        take(f"head.projection.{n}.weight", f"head.{n}.weight")
    take("representation.forecast_query", "forecast_query")
    take("transformer.out_norm.scale", "out_norm.scale")
    unused = []
    for i, kind in enumerate(cfg["schedule"]):
        src, dst = f"transformer.layers.{i}", f"blocks.{i}"
        take(f"{src}.attn_norms.0.scale", f"{dst}.time_norm.scale")
        qkv = sd[f"{src}.attns.0.wQKV.weight"]
        take(f"{src}.attns.0.wQKV.weight", f"{dst}.time_attn.wQKV.weight",
             reorder_qkv_rows(qkv, heads, hd) if qkv_layout == "released" else qkv.clone())
        take(f"{src}.attns.0.wO.weight", f"{dst}.time_attn.wO.weight")
        take(f"{src}.attns.0._gate.weight", f"{dst}.time_attn._gate.weight")
        if kind == "time_variate":
            take(f"{src}.attn_norms.1.scale", f"{dst}.var_norm.scale")
            vqkv = sd[f"{src}.attns.1.wQKV.weight"]
            v_rows = (reorder_qkv_rows(vqkv, heads, hd) if qkv_layout == "released" else vqkv)[2 * d:3 * d]
            take(f"{src}.attns.1.wQKV.weight", f"{dst}.var_attn.wV.weight", v_rows.clone())
            unused.append(f"{src}.attns.1.wQKV.weight[0:{2 * d}] (Q, K: one key per query)")
            take(f"{src}.attns.1.wO.weight", f"{dst}.var_attn.wO.weight")
            take(f"{src}.attns.1._gate.weight", f"{dst}.var_attn._gate.weight")
        take(f"{src}.mlp_norm.scale", f"{dst}.mlp_norm.scale")
        take(f"{src}.mlp.w12.weight", f"{dst}.mlp.w12.weight")
        take(f"{src}.mlp.w3.weight", f"{dst}.mlp.w3.weight")
    left = sorted(set(sd) - used)
    assert not left, f"unused checkpoint tensors: {left}"
    missing, unexpected = core.load_state_dict(new, strict=False)
    assert not unexpected, unexpected
    assert set(missing) <= {"mono"}, missing
    core.unused_rows = unused
    return core.eval()
