#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.9.0",
#     "safetensors>=0.7.0",
#     "numpy>=2.2",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
# Community port — NOT an Apple or Google model.
# Design reference (numbers and ideas only, no code copied): EverPaper, EverPaper/embeddinggemma-2-coreai v1.2, Apache-2.0.
"""EmbeddingGemma 2 audio encoder (Gemma 4 audio tower + ``embed_audio``) re-authored in plain PyTorch.

Weights: google/embeddinggemma-2 (Apache-2.0; deployments follow the Gemma Prohibited Use Policy), read straight
from ``model.safetensors``. Graph: Hugging Face transformers ``Gemma4AudioModel`` followed by
``EmbeddingGemma2Model.embed_audio`` (Apache-2.0), i.e. ``get_audio_features(...).pooler_output``.

Contract of the exported graph (one static graph per frame bucket ``Tb``, e.g. 1000 = 10 s, 3000 = 30 s):

    features    [1, Tb, 128]    log-mel frames from the HF ``Gemma4AudioFeatureExtractor``; rows >= n_frames are 0
    half_mask   [1, Tb/2]       1 for the leading ceil(n_frames / 2) rows, 0 after
    soft_tokens [1, Tb/4, 512]  the leading ceil(ceil(n_frames / 2) / 2) rows are valid, the rest are padding

``half_mask`` is the only mask the static graph needs. Every op after the stride-2 subsampling is causal or per
token, so right padding never reaches a valid token, except through the 3x3 conv look-ahead: when n_frames % 4 == 2
the second conv reads half-frame ceil(n_frames / 2), which sees the last real frame and which upstream zeroes with
its mask. Without ``half_mask`` that clip's last token comes out at cosine ~0.85 (parity_encoder.py measures it).
``forward(features)`` without the mask is exact for an unpadded clip (the conv padding plays the mask's role).

What differs from the upstream graph (each change is exact in real arithmetic):
  - Attention is written in its banded form. Every query t sees the 12 keys t - d, d = 0..11 (self + 11 past);
    the logit is q_t . (k_{t-d} + relk_d) with relk_d = relative_k_proj(sinusoid(d)), a per-layer constant.
    Keys left of the clip start get an additive -1e4 after the softcap (a constant, not an input).
  - The query scale (head_dim^-0.5 / ln 2 * softplus(per_dim_scale)) and the key scale (ln(1 + e) / ln 2) are
    folded into q_proj / k_proj and into their output clip bounds: clamp(Wx, lo, hi) * s = clamp(sWx, s lo, s hi).
  - The +-1e10 "gradient clipping" clamps are dropped (no-ops at inference).
  - The subsampling LayerNorm is mean / variance over the channel axis of NCHW (no permutes); GLU is
    slice * sigmoid; the causal depthwise conv is a grouped Conv1d after a 4-step left pad.
  - ``norm="maxabs"`` (the fp16 graph): every RMSNorm divides the row by its max |x| (floored) before squaring, because
    the residual stream reaches a few hundred and its square overflows float16. The two subsampling LayerNorms
    keep the plain form (inputs <= 51, see absmax.json; the max-abs variant of that channel-axis norm makes the
    h16c Neural Engine compile fail).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

N_MELS = 128
CTX = 12                # keys per query: self + 11 past (attention_chunk_size 12, attention_context_left 13)
MASK_BIAS = -1.0e4      # finite in float16; exp() of it is 0 in both precisions
NORM_FLOOR = 1.0e-3     # max-abs floor of the fp16 RMSNorm (rows of zeros stay zero)
MAX_TOKENS = 1024       # rows of the precomputed left-edge bias table (Tb <= 4096 frames)

_PROBE: dict | None = None   # set by fp16 headroom measurements; never active during export


def _probe(name: str, x: torch.Tensor) -> None:
    if _PROBE is not None:
        v = float(x.detach().abs().amax())
        if v > _PROBE.get(name, 0.0):
            _PROBE[name] = v


class AudioConfig:
    """The fields of config.json the graph depends on, with the assumptions the re-authoring makes."""

    def __init__(self, model_dir: Path):
        full = json.loads((Path(model_dir) / "config.json").read_text())
        a, t = full["audio_config"], full["text_config"]
        self.hidden = a["hidden_size"]
        self.layers = a["num_hidden_layers"]
        self.heads = a["num_attention_heads"]
        self.head_dim = self.hidden // self.heads
        self.eps = a["rms_norm_eps"]
        self.sub_channels = tuple(a["subsampling_conv_channels"])
        self.conv_kernel = a["conv_kernel_size"]
        self.softcap = float(a["attention_logit_cap"])
        self.residual_weight = float(a["residual_weight"])
        self.out_dims = a["output_proj_dims"]
        self.text_hidden = t["hidden_size"]
        assert a["attention_context_right"] == 0, "the banded form assumes a causal encoder"
        assert a["attention_context_left"] - 1 == CTX and a["attention_chunk_size"] == CTX
        assert a["hidden_act"] == "silu" and a["use_clipped_linears"]
        assert self.sub_channels[0] // 4 * self.sub_channels[1] == self.hidden


class RMSNorm(nn.Module):
    """Gemma RMSNorm (the weight multiplies as stored). ``maxabs`` rescales each row by 1 / max|x| before squaring."""

    def __init__(self, dim: int, eps: float, with_scale: bool = True, maxabs: bool = False, tag: str = ""):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=False) if with_scale else None
        self.eps = eps
        self.maxabs = maxabs
        self.tag = tag

    def meansq(self, u: torch.Tensor) -> torch.Tensor:
        """Mean of squares over the last axis; a matmul with a constant 1/D column after use_matmul_meansq()."""
        avg = getattr(self, "avg", None)
        return (u * u) @ avg if avg is not None else (u * u).mean(-1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _PROBE is not None:
            _probe(f"{self.tag}.norm_in", x)
            _probe(f"{self.tag}.norm_meansq", x.double().pow(2).mean(-1))
        if self.maxabs:
            m = torch.clamp(x.abs().amax(-1, keepdim=True), min=NORM_FLOOR)
            r = torch.rsqrt(m)
            inv_m = r * r                                   # 1 / m without a divide op
            u = x * inv_m
            e = inv_m * math.sqrt(self.eps)                 # sqrt(eps) / m; its square is eps / m^2
            y = u * torch.rsqrt(self.meansq(u) + e * e)
        else:
            y = x * torch.rsqrt(self.meansq(x) + self.eps)
        return y * self.weight if self.weight is not None else y


class ClippedLinear(nn.Module):
    """Linear without bias whose input and output are clamped to bounds stored in the checkpoint."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=False)
        self.register_buffer("in_lo", torch.zeros(()))
        self.register_buffer("in_hi", torch.zeros(()))
        self.register_buffer("out_lo", torch.zeros(d_out))
        self.register_buffer("out_hi", torch.zeros(d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.minimum(torch.maximum(x, self.in_lo), self.in_hi)
        return torch.minimum(torch.maximum(self.linear(x), self.out_lo), self.out_hi)


class FeedForward(nn.Module):
    def __init__(self, cfg: AudioConfig, maxabs: bool, tag: str):
        super().__init__()
        self.pre_norm = RMSNorm(cfg.hidden, cfg.eps, maxabs=maxabs, tag=f"{tag}.pre_norm")
        self.up = ClippedLinear(cfg.hidden, 4 * cfg.hidden)
        self.down = ClippedLinear(4 * cfg.hidden, cfg.hidden)
        self.post_norm = RMSNorm(cfg.hidden, cfg.eps, maxabs=maxabs, tag=f"{tag}.post_norm")
        self.residual_weight = cfg.residual_weight
        self.tag = tag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _probe(f"{self.tag}.residual_in", x)
        h = F.silu(self.up(self.pre_norm(x)))
        _probe(f"{self.tag}.ffn_mid", h)
        h = self.down(h)
        _probe(f"{self.tag}.ffn_out", h)
        return x + self.post_norm(h) * self.residual_weight


class BandedAttention(nn.Module):
    """Local causal attention with relative positions, one query against its 12 keys t - d."""

    def __init__(self, cfg: AudioConfig, tag: str):
        super().__init__()
        self.heads, self.head_dim = cfg.heads, cfg.head_dim
        self.q_proj = ClippedLinear(cfg.hidden, cfg.hidden)   # carries the folded query scale
        self.k_proj = ClippedLinear(cfg.hidden, cfg.hidden)   # carries the folded key scale
        self.v_proj = ClippedLinear(cfg.hidden, cfg.hidden)
        self.post = ClippedLinear(cfg.hidden, cfg.hidden)
        self.register_buffer("relk", torch.zeros(CTX, cfg.heads, cfg.head_dim))
        self.softcap = cfg.softcap
        self.tag = tag

    def forward(self, x: torch.Tensor, edge_bias: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.heads, self.head_dim)
        k = F.pad(self.k_proj(x).view(b, t, self.heads, self.head_dim), (0, 0, 0, 0, CTX - 1, 0))
        v = F.pad(self.v_proj(x).view(b, t, self.heads, self.head_dim), (0, 0, 0, 0, CTX - 1, 0))
        # after the left pad, the key at offset d for query t sits at row t + (CTX - 1 - d)
        logits = torch.stack(
            [(q * (k[:, CTX - 1 - d: CTX - 1 - d + t] + self.relk[d])).sum(-1) for d in range(CTX)], dim=-1)
        _probe(f"{self.tag}.logit", logits)
        logits = torch.tanh(logits * (1.0 / self.softcap)) * self.softcap + edge_bias
        w = torch.softmax(logits, dim=-1)                     # [b, t, heads, CTX]
        out = w[..., 0:1] * v[:, CTX - 1: CTX - 1 + t]
        for d in range(1, CTX):
            out = out + w[..., d:d + 1] * v[:, CTX - 1 - d: CTX - 1 - d + t]
        return self.post(out.reshape(b, t, self.heads * self.head_dim))


class LightConv(nn.Module):
    def __init__(self, cfg: AudioConfig, maxabs: bool, tag: str):
        super().__init__()
        self.pre_norm = RMSNorm(cfg.hidden, cfg.eps, maxabs=maxabs, tag=f"{tag}.pre_norm")
        self.start = ClippedLinear(cfg.hidden, 2 * cfg.hidden)
        self.depthwise = nn.Conv1d(cfg.hidden, cfg.hidden, cfg.conv_kernel, groups=cfg.hidden, bias=False)
        self.conv_norm = RMSNorm(cfg.hidden, cfg.eps, maxabs=maxabs, tag=f"{tag}.conv_norm")
        self.end = ClippedLinear(cfg.hidden, cfg.hidden)
        self.hidden = cfg.hidden
        self.left_pad = cfg.conv_kernel - 1
        self.tag = tag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.start(self.pre_norm(x))
        h = h[..., : self.hidden] * torch.sigmoid(h[..., self.hidden:])
        h = self.depthwise(F.pad(h.transpose(1, 2), (self.left_pad, 0))).transpose(1, 2)
        _probe(f"{self.tag}.conv_out", h)
        return x + self.end(F.silu(self.conv_norm(h)))


class ConformerLayer(nn.Module):
    def __init__(self, cfg: AudioConfig, maxabs: bool, i: int):
        super().__init__()
        tag = f"layer{i:02d}"
        self.ffn1 = FeedForward(cfg, maxabs, f"{tag}.ffn1")
        self.norm_pre_attn = RMSNorm(cfg.hidden, cfg.eps, maxabs=maxabs, tag=f"{tag}.norm_pre_attn")
        self.attn = BandedAttention(cfg, tag)
        self.norm_post_attn = RMSNorm(cfg.hidden, cfg.eps, maxabs=maxabs, tag=f"{tag}.norm_post_attn")
        self.lconv = LightConv(cfg, maxabs, f"{tag}.lconv")
        self.ffn2 = FeedForward(cfg, maxabs, f"{tag}.ffn2")
        self.norm_out = RMSNorm(cfg.hidden, cfg.eps, maxabs=maxabs, tag=f"{tag}.norm_out")
        self.tag = tag

    def forward(self, x: torch.Tensor, edge_bias: torch.Tensor) -> torch.Tensor:
        x = self.ffn1(x)
        _probe(f"{self.tag}.residual_attn", x)
        a = self.attn(self.norm_pre_attn(x), edge_bias)
        _probe(f"{self.tag}.attn_out", a)
        x = x + self.norm_post_attn(a)
        x = self.lconv(x)
        x = self.ffn2(x)
        _probe(f"{self.tag}.residual_out", x)
        return self.norm_out(x)


class SubsampleLayer(nn.Module):
    """Conv2d 3x3 stride 2 (no bias) + LayerNorm over channels (weight only) + ReLU, NCHW = [B, C, T, F]."""

    def __init__(self, c_in: int, c_out: int, eps: float, maxabs: bool, tag: str):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
        self.norm_weight = nn.Parameter(torch.ones(c_out), requires_grad=False)
        self.eps = eps
        self.maxabs = maxabs
        self.tag = tag
        self.mean_folded = False        # fold_subsample_mean(): the conv already returns y - mean_c(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        _probe(f"{self.tag}.conv_out", y)
        c = y if self.mean_folded else y - y.mean(1, keepdim=True)
        if self.maxabs:
            m = torch.clamp(c.abs().amax(1, keepdim=True), min=NORM_FLOOR)
            r = torch.rsqrt(m)
            inv_m = r * r
            u = c * inv_m
            e = inv_m * math.sqrt(self.eps)
            y = u * torch.rsqrt((u * u).mean(1, keepdim=True) + e * e)
        else:
            y = c * torch.rsqrt((c * c).mean(1, keepdim=True) + self.eps)
        return F.relu(y * self.norm_weight[None, :, None, None])


class Eg2AudioEncoder(nn.Module):
    """features [B, T, 128] (+ optional half_mask [B, ceil(T/2)]) -> soft_tokens [B, ceil(ceil(T/2)/2), 512]."""

    def __init__(self, cfg: AudioConfig, norm: str = "exact", sub_maxabs: bool = False):
        super().__init__()
        assert norm in ("exact", "maxabs")
        maxabs = norm == "maxabs"
        c0, c1 = cfg.sub_channels
        # The subsampling LayerNorms stay plain even in the fp16 graph: their inputs peak at 22.6 / 51.1 (absmax.json),
        # far from overflow, and the max-abs form of this channel-axis norm is what the Neural Engine compiler rejects.
        self.sub0 = SubsampleLayer(1, c0, cfg.eps, sub_maxabs, "sub0")
        self.sub1 = SubsampleLayer(c0, c1, cfg.eps, sub_maxabs, "sub1")
        self.input_proj = nn.Linear(cfg.hidden, cfg.hidden, bias=False)
        self.layers = nn.ModuleList([ConformerLayer(cfg, maxabs, i) for i in range(cfg.layers)])
        self.output_proj = nn.Linear(cfg.hidden, cfg.out_dims, bias=True)
        self.pre_projection_norm = RMSNorm(cfg.out_dims, cfg.eps, with_scale=False, maxabs=maxabs,
                                           tag="embed_audio.pre_projection_norm")
        self.embedding_projection = nn.Linear(cfg.out_dims, cfg.text_hidden, bias=False)
        t = torch.arange(MAX_TOKENS)[:, None]
        d = torch.arange(CTX)[None, :]
        bias = torch.where(t - d < 0, torch.tensor(MASK_BIAS), torch.tensor(0.0))
        self.register_buffer("edge_bias", bias[None, :, None, :].clone())   # [1, MAX_TOKENS, 1, CTX]

    def forward(self, features: torch.Tensor, half_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.sub0(features.unsqueeze(1))                  # [B, 128, T/2, 64]
        if half_mask is not None:
            x = x * half_mask[:, None, :, None]
        x = self.sub1(x)                                      # [B, 32, T/4, 32]
        b, c, t4, f = x.shape
        assert t4 <= MAX_TOKENS
        x = self.input_proj(x.permute(0, 2, 3, 1).reshape(b, t4, f * c))   # feature index = f * 32 + c
        edge_bias = self.edge_bias[:, :t4]
        for layer in self.layers:
            x = layer(x, edge_bias)
        x = self.output_proj(x)
        _probe("output_proj", x)
        return self.embedding_projection(self.pre_projection_norm(x))


def n_tokens(n_frames: int) -> int:
    """Valid soft tokens for n_frames valid log-mel frames: ceil(ceil(n / 2) / 2)."""
    return ((n_frames + 1) // 2 + 1) // 2


def half_mask_for(n_frames: int, bucket: int, dtype=torch.float32) -> torch.Tensor:
    m = torch.zeros(1, (bucket + 1) // 2, dtype=dtype)
    m[0, : (n_frames + 1) // 2] = 1
    return m


def relative_keys(cfg: AudioConfig, w: torch.Tensor) -> torch.Tensor:
    """relk[d] = relative_k_proj(sinusoid(d)) for d = 0..11, as [CTX, heads, head_dim], in float64 then float32.

    The upstream table holds positions 12, 11, ..., 0 and its relative shift pairs the key at offset d with
    position d; offset 12 is outside the 12-key window, so it is not needed.
    """
    n = cfg.hidden // 2
    inv = torch.exp(torch.arange(n, dtype=torch.float64) * -(math.log(10000.0) / max(n - 1, 1)))
    pos = torch.arange(CTX, dtype=torch.float64)[:, None] * inv[None, :]
    table = torch.cat([torch.sin(pos), torch.cos(pos)], dim=-1)           # [CTX, hidden]
    return (table @ w.double().T).float().view(CTX, cfg.heads, cfg.head_dim)


def load_encoder(model_dir: Path, norm: str = "exact", sub_maxabs: bool = False) -> Eg2AudioEncoder:
    """Build the encoder in float32 from ``model.safetensors`` (bf16 tensors are widened exactly)."""
    from safetensors import safe_open

    model_dir = Path(model_dir)
    cfg = AudioConfig(model_dir)
    enc = Eg2AudioEncoder(cfg, norm=norm, sub_maxabs=sub_maxabs)
    with safe_open(str(model_dir / "model.safetensors"), "pt") as fh:
        raw = {k: fh.get_tensor(k).float() for k in fh.keys()
               if k.startswith("audio_tower.") or k.startswith("embed_audio.")}
    used: set[str] = set()

    def take(key: str) -> torch.Tensor:
        used.add(key)
        return raw[key]

    sd: dict[str, torch.Tensor] = {}
    q_scale = cfg.head_dim ** -0.5 / math.log(2.0)
    k_scale = math.log(1.0 + math.e) / math.log(2.0)

    def clipped(dst: str, src: str, scale: torch.Tensor | float | None = None) -> None:
        w = take(f"{src}.linear.weight")
        lo = take(f"{src}.output_min").expand(w.shape[0]).clone()
        hi = take(f"{src}.output_max").expand(w.shape[0]).clone()
        if scale is not None:
            s = torch.as_tensor(scale, dtype=torch.float32)
            assert bool((s > 0).all()), "folding a clamp through a scale needs s > 0"
            w = w * (s[:, None] if s.ndim else s)
            lo, hi = lo * s, hi * s
        sd[f"{dst}.linear.weight"] = w
        sd[f"{dst}.in_lo"] = take(f"{src}.input_min")
        sd[f"{dst}.in_hi"] = take(f"{src}.input_max")
        sd[f"{dst}.out_lo"] = lo
        sd[f"{dst}.out_hi"] = hi

    p = "audio_tower.subsample_conv_projection"
    for i in (0, 1):
        sd[f"sub{i}.conv.weight"] = take(f"{p}.layer{i}.conv.weight")
        sd[f"sub{i}.norm_weight"] = take(f"{p}.layer{i}.norm.weight")
    sd["input_proj.weight"] = take(f"{p}.input_proj_linear.weight")
    for i in range(cfg.layers):
        src, dst = f"audio_tower.layers.{i}", f"layers.{i}"
        for name in ("1", "2"):
            clipped(f"{dst}.ffn{name}.up", f"{src}.feed_forward{name}.ffw_layer_1")
            clipped(f"{dst}.ffn{name}.down", f"{src}.feed_forward{name}.ffw_layer_2")
            sd[f"{dst}.ffn{name}.pre_norm.weight"] = take(f"{src}.feed_forward{name}.pre_layer_norm.weight")
            sd[f"{dst}.ffn{name}.post_norm.weight"] = take(f"{src}.feed_forward{name}.post_layer_norm.weight")
        for name in ("norm_pre_attn", "norm_post_attn", "norm_out"):
            sd[f"{dst}.{name}.weight"] = take(f"{src}.{name}.weight")
        a = f"{src}.self_attn"
        per_dim = F.softplus(take(f"{a}.per_dim_scale")) * q_scale          # [head_dim], shared by the heads
        clipped(f"{dst}.attn.q_proj", f"{a}.q_proj", per_dim.repeat(cfg.heads))
        clipped(f"{dst}.attn.k_proj", f"{a}.k_proj", k_scale)
        clipped(f"{dst}.attn.v_proj", f"{a}.v_proj")
        clipped(f"{dst}.attn.post", f"{a}.post")
        sd[f"{dst}.attn.relk"] = relative_keys(cfg, take(f"{a}.relative_k_proj.weight"))
        c = f"{src}.lconv1d"
        sd[f"{dst}.lconv.pre_norm.weight"] = take(f"{c}.pre_layer_norm.weight")
        clipped(f"{dst}.lconv.start", f"{c}.linear_start")
        sd[f"{dst}.lconv.depthwise.weight"] = take(f"{c}.depthwise_conv1d.weight")
        sd[f"{dst}.lconv.conv_norm.weight"] = take(f"{c}.conv_norm.weight")
        clipped(f"{dst}.lconv.end", f"{c}.linear_end")
    sd["output_proj.weight"] = take("audio_tower.output_proj.weight")
    sd["output_proj.bias"] = take("audio_tower.output_proj.bias")
    sd["embedding_projection.weight"] = take("embed_audio.embedding_projection.weight")

    unused = sorted(set(raw) - used)
    assert not unused, f"checkpoint tensors the graph does not read: {unused[:8]}"
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert missing == ["edge_bias"], missing                     # the only constant not from the checkpoint
    return enc.eval()


def fold_subsample_mean(enc: Eg2AudioEncoder) -> Eg2AudioEncoder:
    """Variant fp16sf: center each subsampling conv's weights over the output channels, W - mean_k W, and drop the
    LayerNorm's mean subtraction. Exact in real arithmetic (the conv is linear without bias, so the channel mean
    of its output is the conv with the mean kernel at every position, padding included)."""
    for layer in (enc.sub0, enc.sub1):
        w = layer.conv.weight.data.double()
        layer.conv.weight.data = (w - w.mean(0, keepdim=True)).to(layer.conv.weight.dtype)
        layer.mean_folded = True
    return enc


def use_matmul_meansq(enc: nn.Module) -> nn.Module:
    """Variant fp16mm: every RMSNorm takes its mean of squares as a matmul with a constant 1/D column instead of a
    reduce_mean (exact in real arithmetic; a different accumulation path on the device)."""
    for m in enc.modules():
        if isinstance(m, RMSNorm):
            d = m.weight.shape[0] if m.weight is not None else enc.output_proj.out_features
            m.register_buffer("avg", torch.full((d, 1), 1.0 / d, dtype=torch.float32))
    return enc


class _Upcast(nn.Module):
    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w.float()


def store_parameters_fp16(module: nn.Module) -> nn.Module:
    """fp16w32: keep every parameter in float16 and hand the forward a float32 copy (casts stay in the graph)."""
    import torch.nn.utils.parametrize as P

    for m in list(module.modules()):
        for name, param in list(m.named_parameters(recurse=False)):
            setattr(m, name, nn.Parameter(param.detach().half(), requires_grad=False))
            P.register_parametrization(m, name, _Upcast(), unsafe=True)
    return module


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _common import model_dir as _model_dir

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model-dir", help="google/embeddinggemma-2 snapshot (default: see _common.py)")
    ap.add_argument("--bucket", type=int, default=3000, help="frames of the smoke run (default 3000 = 30 s)")
    a = ap.parse_args()
    enc = load_encoder(_model_dir(a.model_dir))
    with torch.no_grad():
        y = enc(torch.zeros(1, a.bucket, N_MELS), half_mask_for(a.bucket, a.bucket))
    print(f"{sum(p.numel() for p in enc.parameters()):,} parameters; features [1, {a.bucket}, {N_MELS}] -> "
          f"soft_tokens {list(y.shape)}")
