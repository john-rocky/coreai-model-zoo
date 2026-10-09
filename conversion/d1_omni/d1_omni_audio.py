#!/usr/bin/env python3
"""Static, plain-PyTorch authoring graph for d1-omni-600M's audio path: one clip's normalised log-mel through the 8x
subsampling, the 17-layer FastConformer, the adapter and the residual, out as the clip's prefix embeddings. One graph
per clip-length bucket (5 / 10 / 20 / 30 s).

Re-authored from the publisher's `audio.py` (LiquidAI/d1-omni-600M @ 414f8d64, sha256 in d1_omni_model.SOURCE_SHA256)
and the tensor names of model.safetensors; it imports neither the publisher's code nor NeMo.

  main  mel [1,128,F] float32        the normalised log-mel (mel_host), columns t >= frames 0.0, zero-padded to F
        mask_f [1,F] float32         1.0 on t < l0 = frames                                (F  = 1 + 100 * sec)
        mask_f2 [1,F2] float32       1.0 on t < l1 = (l0 - 1) // 2 + 1                      (F2 = (F - 1) // 2 + 1)
        mask_f4 [1,F4] float32       1.0 on t < l2 = (l1 - 1) // 2 + 1                      (F4 = (F2 - 1) // 2 + 1)
        mask_t [1,T] float32         1.0 on t < P = l3 = (l2 - 1) // 2 + 1                  (T  = (F4 - 1) // 2 + 1)
        -> prefix [1,T,1024] float32 rows [0, P) are the clip's prefix; the rest are not read
   sec 5 / 10 / 20 / 30 -> F 501 / 1001 / 2001 / 3001, T 63 / 126 / 251 / 376 (P <= 63 / 125 / 250 / 375)

The host (host.audio_inputs) makes the mel (mel_host: the publisher's waveform() and MelFrontend), picks the smallest
bucket whose F holds the clip's STFT columns (1 + n // 160 <= F), pads the mel with zero columns, and builds the four
masks from `frames` with the stride-2 length rule ConvSubsampling applies (the masks are its `_time_mask` before each
layer). The prefix is the graph's first P rows.

What the checkpoint runs (audio.py, read in full):
  - ConvSubsampling: x = mel^T [1, 1, F, 128]; before every one of its 8 layers x *= the time mask of the current
    lengths (Conv2d(1, 256, 3, s2, p1), ReLU, depthwise Conv2d(256, 3, s2, p1), pointwise Conv2d(256, 256, 1), ReLU,
    depthwise, pointwise, ReLU; lengths -> (l + 2 - 3) // 2 + 1 after each stride-2 conv) and once more at the end;
    out = Linear(256 * 16, 512) over [t, (c, f)]
  - Conformer: valid = t < lengths; pos_emb = the sinusoids of relative positions t-1 .. -(t-1) (fp32, CPU);
    17 x ConformerLayer: x += 0.5 FF1(LN x); x += RelPositionAttention(LN x); x += ConvModule(LN x); x += 0.5 FF2(LN x);
    x = LN_out(x). FF = linear2(silu(linear1)), hidden 2048. Attention: 8 heads x 64, ac = (q + u) k^T, bd = (q + v) p^T
    with p = linear_pos(pos_emb) (no bias), Transformer-XL rel_shift (pad, view, slice), (ac + bd[..., :t]) / 8,
    masked_fill(~(valid_q & valid_k), -10000), softmax, masked_fill(.., 0), @ v, linear_out. ConvModule: pointwise_conv1
    (512 -> 1024) -> GLU -> the pad positions set to 0 -> depthwise Conv1d(512, 9, pad 4) -> BatchNorm1d (eval) -> SiLU
    -> pointwise_conv2. LayerNorm eps 1e-5 everywhere.
  - Audio.forward: x[:, :lengths] -> Adapter (LN(512) -> Linear(512, 1024) -> GELU (erf) -> Linear(1024, 1024)) ->
    Residual (x + up(GELU(down(LN(x)))), LN(1024), 1024 -> 512 -> 1024)

How this graph departs from that code, and why each departure is exact:
  - the clip is padded to its bucket's length and the conformer runs over T >= the publisher's t steps: every valid
    step reads only valid steps (attention masks the pad keys, the ConvModule zeroes the pad steps before its depthwise
    conv, the subsampling's masks zero the pad frames before every conv), and the relative-position table holds the
    same row for the same relative position at any length, so the valid rows do not depend on the padding.
  - the subsampling's redundant mask products are left out: a mask before a ReLU that follows a mask of the same length
    (relu(m x) m = relu(m x) for m in {0, 1}), and the last one (after the last ReLU, same length). Every product that
    zeroes a nonzero value (after each conv) stays.
  - attention is matmul -> softmax (fp32) -> matmul with an additive mask: MASK = -1e4 on the pad keys (exp(-1e4 + ...)
    = 0 in fp32, as with the publisher's masked_fill to -10000), then the probabilities times the valid-query mask
    (the publisher's masked_fill(mask, 0): a pad query row becomes 0, a valid row's pad keys are 0 already). The scale
    1/8 is a power of two. At most 376 keys: the plain chain (the Mac GPU's 4,032-key limit is not reached).
  - BatchNorm1d (eval) is the affine it computes: scale = weight / sqrt(running_var + eps), shift = bias -
    running_mean * scale, folded in float64 at load and kept as two constants per layer (num_batches_tracked unread).
  - the rel_shift is the publisher's pad / view / slice on the static [1, 8, T, 2T-1] scores.

Precision (`set_precision`): fp32 is the reference. wfp16 = fp16 storage of every Linear / Conv weight and bias and of
pos_bias_u / v, fp32 compute (each cast at its use); fp16 = fp16 compute, with every LayerNorm, the attention softmax,
the BatchNorm affine, both GELUs (adapter and residual) and the output in fp32. The mel and the masks are fp32 inputs in
every precision; the relative-position table is an fp32 constant (cast to fp16 at its use in fp16 compute).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

import d1_omni_model as dm

CLIP_SECONDS = (5, 10, 20, 30)          # the clip-length buckets
HOP_SECONDS = 0.01                      # one mel column per 10 ms: F = 1 + 100 * sec
MASK_VALUE = -1.0e4
BN_EPS = 1e-5                           # nn.BatchNorm1d default (the publisher's ConvModule)
PRECISIONS = ("fp32", "wfp16", "fp16")
INPUT_NAMES = ("mel", "mask_f", "mask_f2", "mask_f4", "mask_t")
# checkpoint prefix -> this module's prefix (the BatchNorm tensors are folded, not loaded as modules)
KEY_MAP = (("audio.encoder.", ""), ("audio.adapter.", "adapter."), ("audio.residual.", "residual."))
BN_FIELDS = ("weight", "bias", "running_mean", "running_var")
CHECKPOINT_KEYS = 704                   # 24 outside the layers + 17 x 40 (num_batches_tracked included)
EXPECTED_AUDIO_CONFIG = {"feat_in": 128, "n_layers": 17, "d_model": 512, "subsampling_conv_channels": 256,
                         "ff_expansion_factor": 4, "n_heads": 8, "conv_kernel_size": 9, "residual_width": 512}


def subsample(length: int) -> int:
    """ConvSubsampling's length after one stride-2 conv (kernel 3, padding 1)."""
    return (length + 2 - 3) // 2 + 1


def bucket_shapes(sec: int) -> dict:
    """The static lengths of one clip bucket: F (mel columns) and the three subsampled lengths F2, F4, T."""
    if sec not in CLIP_SECONDS:
        raise ValueError(f"clip bucket must be one of {CLIP_SECONDS} s")
    f = 1 + round(sec / HOP_SECONDS)
    f2 = subsample(f)
    f4 = subsample(f2)
    return {"sec": sec, "F": f, "F2": f2, "F4": f4, "T": subsample(f4)}


def validate_config(config: dict) -> None:
    """The whole config.json, key for key (d1_omni_model.validate_config, audio_config included), and the audio sizes
    this graph is written for."""
    dm.validate_config(config)
    a = config["audio_config"]
    problems = [f"{k}: {a.get(k)} != {v}" for k, v in EXPECTED_AUDIO_CONFIG.items() if a.get(k) != v]
    if set(a) != set(EXPECTED_AUDIO_CONFIG):
        problems.append(f"audio_config keys {sorted(a)}")
    if config["text_config"]["hidden_size"] != 1024:
        problems.append("text hidden size != 1024")
    if problems:
        raise ValueError("Unsupported d1-omni audio configuration:\n  " + "\n  ".join(problems))


def relative_positions(t: int, d: int) -> torch.Tensor:
    """Conformer.pos_emb: sinusoids for relative positions t-1 .. -(t-1), built in fp32 on the CPU as the publisher
    builds them -> [1, 2t-1, d]."""
    positions = torch.arange(t - 1, -t, -1, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * -(math.log(10000.0) / d))
    pe = torch.zeros(len(positions), d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(positions * div), torch.cos(positions * div)
    return pe[None]


# --------------------------------------------------------------------------- modules (names = checkpoint names)
class _FeedForward(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.linear1, self.linear2 = nn.Linear(d, d_ff), nn.Linear(d_ff, d)


class _Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.linear_q, self.linear_k, self.linear_v, self.linear_out = (nn.Linear(d, d) for _ in range(4))
        self.linear_pos = nn.Linear(d, d, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(heads, d // heads))
        self.pos_bias_v = nn.Parameter(torch.zeros(heads, d // heads))


class _ConvModule(nn.Module):
    def __init__(self, d: int, kernel: int):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d, 2 * d, 1)
        self.depthwise_conv = nn.Conv1d(d, d, kernel, groups=d)
        self.pointwise_conv2 = nn.Conv1d(d, d, 1)
        self.pad = (kernel - 1) // 2
        # BatchNorm1d (eval) folded: y = x * bn_scale + bn_shift (load_weights)
        self.register_buffer("bn_scale", torch.ones(d))
        self.register_buffer("bn_shift", torch.zeros(d))


class _Layer(nn.Module):
    def __init__(self, d: int, d_ff: int, heads: int, kernel: int):
        super().__init__()
        self.norm_feed_forward1, self.feed_forward1 = nn.LayerNorm(d), _FeedForward(d, d_ff)
        self.norm_self_att, self.self_attn = nn.LayerNorm(d), _Attention(d, heads)
        self.norm_conv, self.conv = nn.LayerNorm(d), _ConvModule(d, kernel)
        self.norm_feed_forward2, self.feed_forward2 = nn.LayerNorm(d), _FeedForward(d, d_ff)
        self.norm_out = nn.LayerNorm(d)


class _Subsampling(nn.Module):
    def __init__(self, feat_in: int, channels: int, d: int):
        super().__init__()
        self.conv = nn.Sequential(  # the publisher's indices: 0, 2, 3, 5, 6 hold weights
            nn.Conv2d(1, channels, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 2, 1, groups=channels), nn.Conv2d(channels, channels, 1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 2, 1, groups=channels), nn.Conv2d(channels, channels, 1), nn.ReLU())
        freq = feat_in
        for _ in range(3):
            freq = subsample(freq)
        self.out = nn.Linear(channels * freq, d)


class _Adapter(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.norm, self.linear_1, self.linear_2 = nn.LayerNorm(d_in), nn.Linear(d_in, d_out), nn.Linear(d_out, d_out)


class _Residual(nn.Module):
    def __init__(self, d: int, width: int):
        super().__init__()
        self.ln, self.down, self.up = nn.LayerNorm(d), nn.Linear(d, width), nn.Linear(width, d)


class D1Audio(nn.Module):
    """Subsampling + conformer + adapter + residual for one clip bucket (the module docstring's `main`). Parameter
    names are the checkpoint's under KEY_MAP; each ConvModule's BatchNorm is two folded constants."""

    def __init__(self, config: dict, sec: int):
        super().__init__()
        a = config["audio_config"]
        d = a["d_model"]
        self.shapes = bucket_shapes(sec)
        self.sec, self.d, self.heads = sec, d, a["n_heads"]
        self.head_dim = d // self.heads
        self.feat_in = a["feat_in"]
        self.out_dim = config["text_config"]["hidden_size"]
        self.pre_encode = _Subsampling(a["feat_in"], a["subsampling_conv_channels"], d)
        self.layers = nn.ModuleList(_Layer(d, d * a["ff_expansion_factor"], a["n_heads"], a["conv_kernel_size"])
                                    for _ in range(a["n_layers"]))
        self.adapter = _Adapter(d, self.out_dim)
        self.residual = _Residual(self.out_dim, a["residual_width"])
        self.register_buffer("pos_emb", relative_positions(self.shapes["T"], d), persistent=False)
        self.precision, self.compute_dtype, self.float_dtype = "fp32", torch.float32, torch.float32
        self._bn_raw: dict[int, dict[str, torch.Tensor]] = {}  # layer -> the checkpoint's BatchNorm tensors (fp32)
        self._config = config

    def rebucket(self, sec: int) -> "D1Audio":
        """The same weights (the tensors shared, not copied) for another clip bucket: only the shapes and the
        relative-position table differ."""
        with torch.device("meta"):
            other = D1Audio(self._config, sec)
        other.load_state_dict(self.state_dict(), assign=True, strict=True)
        other.pos_emb = relative_positions(other.shapes["T"], self.d).to(self.pos_emb.dtype)
        other.precision, other.compute_dtype, other.float_dtype = self.precision, self.compute_dtype, self.float_dtype
        other._bn_raw = self._bn_raw
        return other.eval().requires_grad_(False)

    # ---- precision
    def set_precision(self, precision: str) -> "D1Audio":
        """fp32 / wfp16 / fp16 (module docstring). LayerNorm weights and the folded BatchNorm stay fp32."""
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        storage = torch.float32 if precision == "fp32" else torch.float16
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                continue
            for param in module.parameters(recurse=False):
                param.data = param.data.to(storage)
        self.precision = precision
        self.compute_dtype = torch.float16 if precision == "fp16" else torch.float32
        return self

    def as_float64(self) -> "D1Audio":
        """A diagnostic, never exported: every weight and every op in float64 (the BatchNorm folded again in float64
        from the checkpoint's tensors; the relative-position table built in fp32 as the publisher builds it)."""
        self.double()
        for i, layer in enumerate(self.layers):
            scale, shift = fold_batch_norm(self._bn_raw[i], torch.float64)
            layer.conv.bn_scale.data, layer.conv.bn_shift.data = scale, shift
        self.precision, self.compute_dtype, self.float_dtype = "float64", torch.float64, torch.float64
        return self

    # ---- pieces (each weight is cast to the activation dtype at its use: a no-op in fp32)
    @staticmethod
    def _lin(x: torch.Tensor, module: nn.Linear) -> torch.Tensor:
        bias = None if module.bias is None else module.bias.to(x.dtype)
        return F.linear(x, module.weight.to(x.dtype), bias)

    @staticmethod
    def _conv(x: torch.Tensor, module: nn.Conv1d | nn.Conv2d, padding=None) -> torch.Tensor:
        weight, bias = module.weight.to(x.dtype), module.bias.to(x.dtype)
        pad = module.padding if padding is None else padding
        conv = F.conv2d if isinstance(module, nn.Conv2d) else F.conv1d
        return conv(x, weight, bias, module.stride, pad, module.dilation, module.groups)

    def _ln(self, norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        f = self.float_dtype
        y = F.layer_norm(x.to(f), norm.normalized_shape, norm.weight.to(f), norm.bias.to(f), norm.eps)
        return y.to(x.dtype)

    def subsampling(self, mel, mask_f, mask_f2, mask_f4, mask_t) -> torch.Tensor:
        """[1, 128, F] -> [1, T, 512] (ConvSubsampling with its time masks)."""
        dtype, conv = self.compute_dtype, self.pre_encode.conv
        m = [w.to(dtype).reshape(1, 1, -1, 1) for w in (mask_f, mask_f2, mask_f4, mask_t)]
        x = mel.to(dtype).transpose(1, 2).unsqueeze(1) * m[0]          # [1, 1, F, 128]
        x = torch.relu(self._conv(x, conv[0]) * m[1])                   # [1, 256, F2, 64]
        x = self._conv(x, conv[2]) * m[2]                               # depthwise, [1, 256, F4, 32]
        x = torch.relu(self._conv(x, conv[3]) * m[2])                   # pointwise
        x = self._conv(x, conv[5]) * m[3]                               # depthwise, [1, 256, T, 16]
        x = torch.relu(self._conv(x, conv[6]) * m[3])                   # pointwise
        b, c, t, f = x.shape
        return self._lin(x.transpose(1, 2).reshape(b, t, c * f), self.pre_encode.out)

    def _attention(self, attn: _Attention, x: torch.Tensor, pos: torch.Tensor, key_mask: torch.Tensor,
                   query_mask: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h, dk = self.heads, self.head_dim
        q = self._lin(x, attn.linear_q).reshape(b, t, h, dk)
        k = self._lin(x, attn.linear_k).reshape(b, t, h, dk).transpose(1, 2)
        v = self._lin(x, attn.linear_v).reshape(b, t, h, dk).transpose(1, 2)
        p = self._lin(pos, attn.linear_pos).reshape(1, -1, h, dk).transpose(1, 2)       # [1, h, 2t-1, dk]
        ac = torch.matmul((q + attn.pos_bias_u.to(x.dtype)).transpose(1, 2), k.transpose(-2, -1))
        bd = torch.matmul((q + attn.pos_bias_v.to(x.dtype)).transpose(1, 2), p.transpose(-2, -1))
        bh, hh, qlen, pos_len = bd.shape
        bd = F.pad(bd, (1, 0)).reshape(bh, hh, pos_len + 1, qlen)[:, :, 1:].reshape(bh, hh, qlen, pos_len)
        scores = (ac + bd[:, :, :, :t]).to(self.float_dtype) * (1.0 / math.sqrt(dk)) + key_mask
        probs = torch.softmax(scores, dim=-1) * query_mask
        y = torch.matmul(probs.to(v.dtype), v)
        return self._lin(y.transpose(1, 2).reshape(b, t, h * dk), attn.linear_out)

    def _conv_module(self, conv: _ConvModule, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        y = F.glu(self._conv(x.transpose(1, 2), conv.pointwise_conv1), dim=1) * valid    # [1, 512, T]
        y = self._conv(y, conv.depthwise_conv, padding=conv.pad)
        f = self.float_dtype
        y = (y.to(f) * conv.bn_scale.to(f)[None, :, None] + conv.bn_shift.to(f)[None, :, None]).to(y.dtype)
        return self._conv(F.silu(y), conv.pointwise_conv2).transpose(1, 2)

    def encoder(self, x: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
        """[1, T, 512] -> the 17 conformer layers -> [1, T, 512] (compute dtype)."""
        t = x.shape[1]
        f = self.float_dtype
        valid = mask_t.to(f)
        key_mask = ((1.0 - valid) * MASK_VALUE).reshape(1, 1, 1, t)
        query_mask = valid.reshape(1, 1, t, 1)
        conv_valid = mask_t.to(x.dtype).reshape(1, 1, t)
        pos = self.pos_emb.to(x.dtype)
        for layer in self.layers:
            x = x + self._ff(layer.feed_forward1, self._ln(layer.norm_feed_forward1, x)) * 0.5
            x = x + self._attention(layer.self_attn, self._ln(layer.norm_self_att, x), pos, key_mask, query_mask)
            x = x + self._conv_module(layer.conv, self._ln(layer.norm_conv, x), conv_valid)
            x = x + self._ff(layer.feed_forward2, self._ln(layer.norm_feed_forward2, x)) * 0.5
            x = self._ln(layer.norm_out, x)
        return x

    def _ff(self, ff: _FeedForward, x: torch.Tensor) -> torch.Tensor:
        return self._lin(F.silu(self._lin(x, ff.linear1)), ff.linear2)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Adapter then Residual -> [1, T, 1024] float32 (both GELUs exact erf, in the float dtype)."""
        f = self.float_dtype
        a = self._lin(self._ln(self.adapter.norm, x), self.adapter.linear_1)
        a = self._lin(F.gelu(a.to(f)).to(a.dtype), self.adapter.linear_2)
        r = self._lin(self._ln(self.residual.ln, a), self.residual.down)
        r = self._lin(F.gelu(r.to(f)).to(r.dtype), self.residual.up)
        return (a + r).to(f)

    def forward(self, mel, mask_f, mask_f2, mask_f4, mask_t):
        s = self.shapes
        for name, t, shape in (("mel", mel, (1, self.feat_in, s["F"])), ("mask_f", mask_f, (1, s["F"])),
                               ("mask_f2", mask_f2, (1, s["F2"])), ("mask_f4", mask_f4, (1, s["F4"])),
                               ("mask_t", mask_t, (1, s["T"]))):
            if tuple(t.shape) != shape:
                raise ValueError(f"{name}: expected {shape}, got {tuple(t.shape)}")
        x = self.subsampling(mel, mask_f, mask_f2, mask_f4, mask_t)
        return self.project(self.encoder(x, mask_t))


# --------------------------------------------------------------------------- weights
def fold_batch_norm(raw: dict[str, torch.Tensor], dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """BatchNorm1d (eval) as y = x * scale + shift, computed in float64 and stored in dtype."""
    w, b, mean, var = (raw[k].to(torch.float64) for k in BN_FIELDS)
    scale = w / torch.sqrt(var + BN_EPS)
    return scale.to(dtype), (b - mean * scale).to(dtype)


def map_key(checkpoint_key: str) -> tuple[str, Any]:
    """An audio checkpoint key -> ("param", module key) | ("bn", (layer, field)) | ("skip", None)."""
    if checkpoint_key.endswith("num_batches_tracked"):
        return "skip", None
    if ".conv.batch_norm." in checkpoint_key:
        head, field = checkpoint_key.rsplit(".", 1)
        layer = int(head.split(".layers.")[1].split(".")[0])
        return "bn", (layer, field)
    for source, target in KEY_MAP:
        if checkpoint_key.startswith(source):
            return "param", target + checkpoint_key[len(source):]
    raise KeyError(f"unexpected audio key {checkpoint_key!r}")


def load_weights(model: D1Audio, safetensors_path: str | Path, verify_sha256: bool = False) -> dict:
    """Strict load of the audio tensors (fp32) from model.safetensors (trunk, head and vision skipped and counted); the
    17 BatchNorms folded into each ConvModule's bn_scale / bn_shift."""
    from safetensors import safe_open

    path = Path(safetensors_path)
    if path.stat().st_size != dm.WEIGHTS["bytes"]:
        raise ValueError(f"{path}: {path.stat().st_size} bytes, expected {dm.WEIGHTS['bytes']}")
    if verify_sha256:
        import hashlib
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 24), b""):
                digest.update(block)
        if digest.hexdigest() != dm.WEIGHTS["sha256"]:
            raise ValueError(f"{path}: sha256 {digest.hexdigest()} != {dm.WEIGHTS['sha256']}")
    state, bn, skipped_other, skipped_counters, parameters, keys = {}, {}, 0, 0, 0, 0
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if not key.startswith("audio."):
                skipped_other += 1
                continue
            keys += 1
            kind, target = map_key(key)
            if kind == "skip":
                skipped_counters += 1
                continue
            tensor = handle.get_tensor(key)
            if tensor.dtype != torch.float32:
                raise ValueError(f"{key}: {tensor.dtype}, the checkpoint is float32")
            parameters += tensor.numel()
            if kind == "bn":
                bn.setdefault(target[0], {})[target[1]] = tensor
            else:
                state[target] = tensor
    if keys != CHECKPOINT_KEYS or parameters != dm.PARAMETERS["audio"]:
        raise ValueError(f"audio keys {keys} / parameters {parameters}: expected {CHECKPOINT_KEYS} / "
                         f"{dm.PARAMETERS['audio']}")
    if sorted(bn) != list(range(len(model.layers))) or any(set(v) != set(BN_FIELDS) for v in bn.values()):
        raise ValueError("BatchNorm tensors: expected weight / bias / running_mean / running_var for every layer")
    for i, layer in enumerate(model.layers):
        state[f"layers.{i}.conv.bn_scale"], state[f"layers.{i}.conv.bn_shift"] = fold_batch_norm(bn[i])
    model.load_state_dict(state, strict=True)
    model._bn_raw = bn
    return {"tensors": len(state) - 2 * len(model.layers), "batch_norm_folded": len(bn),
            "num_batches_tracked_skipped": skipped_counters, "skipped_other": skipped_other,
            "parameters": parameters, "graph_parameters": sum(p.numel() for p in model.parameters()),
            "graph_constants": sum(b.numel() for b in model.buffers()), "sha256_verified": verify_sha256}


def build(config: dict, sec: int) -> D1Audio:
    validate_config(config)
    return D1Audio(config, sec)


def load_d1_audio(snapshot_dir: str | Path, sec: int, precision: str = "fp32",
                  verify_sha256: bool = False) -> tuple[D1Audio, dict]:
    """The pinned snapshot's config.json + model.safetensors -> the eval-mode audio graph module of one bucket."""
    snapshot = Path(snapshot_dir)
    model = build(json.loads((snapshot / "config.json").read_text()), sec)
    record = load_weights(model, snapshot / dm.WEIGHTS["file"], verify_sha256)
    model.eval().requires_grad_(False)
    return model.set_precision(precision), record


def _self_check() -> None:  # python3 d1_omni_audio.py: config + module structure against the pinned files
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _paths import hf_snapshot  # noqa: E402

    snapshot = Path(hf_snapshot(dm.MODEL_ID, revision=dm.MODEL_SHA))
    config: dict[str, Any] = json.loads((snapshot / "config.json").read_text())
    validate_config(config)
    for sec in CLIP_SECONDS:
        with torch.device("meta"):
            model = build(config, sec)
        print(f"{sec} s: {bucket_shapes(sec)}; graph parameters {sum(p.numel() for p in model.parameters()):,} + "
              f"{sum(b.numel() for b in model.buffers()):,} constants")


if __name__ == "__main__":
    _self_check()
