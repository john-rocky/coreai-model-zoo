#!/usr/bin/env python3
"""Static, plain-PyTorch authoring graph for d1-omni-600M's decision path: the bidirectional LFM2 trunk and the
decision head, one question row per call.

Re-authored from the publisher's `encoder.py` (LiquidAI/d1-omni-600M @ 414f8d64, sha256 in SOURCE_SHA256) and the
tensor names of its model.safetensors; it imports neither transformers nor the publisher's code. The vision and
audio towers are separate graphs; their output reaches this one as `prefix_embeds`.

  main  input_ids [1,L] int32, prefix_embeds [1,L,1024] float32, pad_mask [1,L] float32,
        prefix_mask [1,L] float32, keep_right [1,L] float32, qtype_onehot [1,3] float32 (choice / score / noul)
        -> scores [1,L] float32 (the scorer at every position)

The positions are the publisher's `cat([prefix, embed(ids)])`, right-padded to the bucket length L: [0, P) the
media prefix (prefix_mask 1, its embeddings in prefix_embeds), [P, P+n) the text ids, [P+n, L) pad (pad_mask 0).
`keep_right` is the publisher's `t != prefix - 1`: 0 at position P-1 when P > 0, 1 everywhere else. The host
(host.py) builds these vectors, gathers the scores at P + markers, and applies the temperature, the softmax and
the noul flip.

What the checkpoint runs (encoder.py, read in full):
  - trunk: token embedding; 16 LFM2 blocks (layer_types: 10 short convolutions, 6 GQA attentions), each
    `h += op(operator_norm(h)); h += mlp(ffn_norm(h))`; RMSNorm computed in fp32 (x * rsqrt(mean(x^2) + eps),
    then weight * x); the final norm is `embedding_norm`
  - short conv: `b, c, u = in_proj(x * pad)`; a centred 3-tap depthwise filter over b*u whose right tap is
    multiplied by keep_right (a media position never reads the text); `out_proj(c * y)`
  - attention: per-head RMSNorm on q and k, RoPE (theta 1e6, rotate_half over all 64 dims, inv_freq built in
    fp32 on the CPU), 16 query heads over 8 key/value heads, scale 64**-0.5; bidirectional, masked only on pad
    keys and on media query x text key (a media position sees the media, a text position sees everything)
  - MLP: w2(silu(w1 x) * w3 x), hidden 4608
  - head: `h + type_emb[qtype]` at every position, two `nn.TransformerEncoderLayer(1024, 16, 4096, ReLU,
    norm_first=True)` over the text positions with pad keys masked, then the scorer LayerNorm -> Linear(1024,
    1024) -> GELU (erf) -> Linear(1024, 1). The publisher gathers the markers before the scorer; the scorer is
    position-wise, so scoring every position and gathering on the host reads the same numbers.

How this graph departs from the publisher's code, and why each departure is exact:
  - masks are additive float arithmetic on the vector inputs (no boolean or integer ops), MASK = -1e4: a masked
    key gets weight exp(-1e4 + ...) = 0 in fp32 and fp16, as with the publisher's -1e9 / -inf, and a key masked
    twice (-2e4) stays finite in fp16. Every query row keeps at least one open key, so no row is fully masked.
  - attention is matmul -> softmax -> matmul (no scaled_dot_product_attention composite); the scale 0.125 is a
    power of two, so where it is applied does not round.
  - the head runs over all L positions with every non-text key masked, instead of over the text positions cut
    out of the sequence; the head has no positional input, so a text position sees the same keys and values.
  - type_emb is a one-hot matmul (1.0 * row + 0.0 * other rows is the row, bit for bit).
  Pad positions hold the embedding of whatever id the host puts there; nothing real reads them (the conv input is
  multiplied by pad_mask before in_proj, and every attention masks pad keys), so the outputs at real positions
  do not depend on pad content (toy_module_check.py checks this bit for bit).

Key blocks: at a static length above `key_block` (KEY_BLOCK = 2048) every attention, trunk and head, splits its keys
into blocks of at most key_block that share one max: s_b = q k_b^T * scale + mask_b, m = max over the blocks of
max_k s_b, e_b = exp(s_b - m), and each block's share of the softmax e_b / (sum_b sum_k e_b), in fp32, times v_b,
summed over the blocks. That is the softmax of the plain form, divided before the product with v rather than after
(in fp16 compute, e_b v_b summed over 4,096 keys could pass 65,504 before the division; the share of a block stays
<= 1, as the plain form's probabilities do). Why: compiled for the Mac GPU (macOS 27 26A428, AOT h16c), the plain
chain softmax(q k^T) v returns wrong values that change from call to call once the key length reaches 4,032, while
blocks of 2,048 keys with a shared max are exact (zoo knowledge/clef-flash-port.md "Attention over 4,032 keys").
At L <= key_block the graph is the plain form, unchanged. In fp32 the two forms differ by rounding only: in float64
they agree to 6.7e-14 in the marker logits, and each fp32 form sits about 3e-5 from its own float64 run
(block_softmax_check.json, block_softmax_diag.json, round 4).

Precision (`set_precision`): fp32 is the reference. wfp16 = fp16 weight storage with fp32 compute (each weight
cast to fp32 at its use); fp16 = fp16 compute, with RMSNorm (fp32 in the publisher's code too), LayerNorm,
softmax and the scorer in fp32. Quant mode (`set_quant_mode`, the int8 export) runs the fp16 frame with every
trunk and head linear called as its module, where a module-level quantizer replaces the weight.

Mutations change one behaviour each and are negative controls (toy_module_check.py), never export variants.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

MODEL_ID = "LiquidAI/d1-omni-600M"
MODEL_SHA = "414f8d6438174f5b2133a9c21a478fc42625e308"
SOURCE_SHA256 = {  # the publisher's files this graph and host.py were written from (snapshot @ MODEL_SHA)
    "encoder.py": "5f2e20319ee42a7367febf06c7653d7b3f9b28ef99a64cb9ec7739563dcfcffe",
    "modeling_d1.py": "2b71a5c909ec5d3aa6307378a41a0fd24af1e17d5c1c1f8d510582147051a686",
    "prompt.py": "a6b29a55ec8345f1fc1fdbfcc4b64d80d473dc4316f095a62c51cb6cce1194cf",
    "vision.py": "43ad71029f82e20fd26b97574627773d47639d7a3742c8b04fe5bf55f41c4c28",
    "audio.py": "c5ff09d52d6a079e79d0505c89dbd0582ba2bc9abf2b26595d8d5b1d0c347bdf",
    "config.json": "dfccae241822f81752426629c88c8b8d7efdd03dd19418b6645c604fad43d619",
}
WEIGHTS = {"file": "model.safetensors", "bytes": 2_348_774_500, "tensors": 1084,
           "sha256": "0713bb05270c2685ad106522f4092bceeeb3a93cf79b401f399a712296c911e1"}
# Parameter counts per top-level prefix, from the safetensors header (F32; the 17 I64 tensors are the audio
# BatchNorm `num_batches_tracked` counters and are not parameters).
PARAMETERS = {"encoder": 354_483_968, "head": 26_248_193, "vision": 94_234_880, "audio": 112_194_048}
CHECKPOINT_KEYS = {"encoder": 148, "head": 31}
KEY_MAP = (("encoder.", "trunk."), ("head.", "head."))  # checkpoint prefix -> this module's prefix
IGNORED_PREFIXES = ("vision.", "audio.")                 # separate graphs
MASK_VALUE = -1.0e4
KEY_BLOCK = 2048  # keys per attention block above this static length (module docstring, "Key blocks")
MUTATIONS = ("none", "keep_right_ones", "no_media_text_mask", "no_head_key_mask")
PRECISIONS = ("fp32", "wfp16", "fp16")
GQA_FORMS = ("repeat", "broadcast")

# config.json at MODEL_SHA, every key (validate_config rejects anything else: this graph is written for this
# checkpoint, not for the d1 family).
EXPECTED_CONFIG: dict[str, Any] = {
    "architectures": ["D1OmniModel"],
    "auto_map": {"AutoConfig": "modeling_d1.D1OmniConfig", "AutoModel": "modeling_d1.D1OmniModel"},
    "model_type": "d1_omni",
    "dtype": "float32",
    "max_length": 16384,
    "image_text_length": 896,
    "audio_text_length": 15360,
    "head_layers": 2,
    "projector_hidden_size": 2048,
    "temperatures": {
        "choice:11+": 1.372515082359314, "choice:2": 1.7465145587921143, "choice:3-5": 1.3998981714248657,
        "choice:6-10": 1.1751071214675903, "noul:2": 1.6663223505020142, "score:3-5": 1.7301132678985596,
        "score:6-10": 1.0, "choice": 1.0, "score": 1.0, "noul": 1.0,
    },
    "bos_token_id": 1,
    "pad_token_id": 0,
    "text_config": {
        "vocab_size": 65536, "hidden_size": 1024, "intermediate_size": 6656, "num_hidden_layers": 16,
        "num_attention_heads": 16, "num_key_value_heads": 8,
        "layer_types": ["conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "conv",
                        "full_attention", "conv", "full_attention", "conv", "full_attention", "conv",
                        "full_attention", "conv"],
        "norm_eps": 1e-05, "conv_L_cache": 3, "block_ffn_dim_multiplier": 1.0, "block_multiple_of": 256,
        "max_position_embeddings": 128000, "rope_theta": 1000000.0,
    },
    "vision_config": {
        "attention_dropout": 0.0, "hidden_act": "gelu_pytorch_tanh", "hidden_size": 768, "intermediate_size": 3072,
        "layer_norm_eps": 1e-06, "model_type": "siglip2_vision_model", "num_attention_heads": 12, "num_channels": 3,
        "num_hidden_layers": 12, "num_patches": 256, "patch_size": 16, "vision_use_head": False,
    },
    "audio_config": {
        "feat_in": 128, "n_layers": 17, "d_model": 512, "subsampling_conv_channels": 256, "ff_expansion_factor": 4,
        "n_heads": 8, "conv_kernel_size": 9, "residual_width": 512,
    },
}
# Derived from text_config the way encoder.py derives it (checked by validate_config).
DERIVED = {"head_dim": 64, "mlp_hidden": 4608, "attention_layers": [2, 5, 8, 10, 12, 14], "head_heads": 16,
           "head_ffn": 4096}


def _diff(actual: Any, expected: Any, path: str, out: list[str]) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(actual) | set(expected)):
            if key not in actual:
                out.append(f"{path}{key}: missing")
            elif key not in expected:
                out.append(f"{path}{key}: unexpected key ({actual[key]!r})")
            else:
                _diff(actual[key], expected[key], f"{path}{key}.", out)
    elif type(actual) is not type(expected) or actual != expected:
        out.append(f"{path.rstrip('.')}: {actual!r} != {expected!r}")


def mlp_hidden(text_config: dict) -> int:
    """encoder.MLP's hidden width: multiple_of(int(multiplier * int(2 * intermediate / 3)))."""
    hidden = int(2 * text_config["intermediate_size"] / 3)
    hidden = int(text_config["block_ffn_dim_multiplier"] * hidden)
    multiple = text_config["block_multiple_of"]
    return multiple * ((hidden + multiple - 1) // multiple)


def validate_config(config: dict) -> None:
    """Reject any config.json that is not this checkpoint's, key for key and value for value."""
    problems: list[str] = []
    _diff(config, EXPECTED_CONFIG, "", problems)
    text = config.get("text_config", {})
    if not problems:
        derived = {
            "head_dim": text["hidden_size"] // text["num_attention_heads"],
            "mlp_hidden": mlp_hidden(text),
            "attention_layers": [i for i, kind in enumerate(text["layer_types"]) if kind == "full_attention"],
            "head_heads": text["hidden_size"] // 64,
            "head_ffn": 4 * text["hidden_size"],
        }
        _diff(derived, DERIVED, "derived.", problems)
        if len(text["layer_types"]) != text["num_hidden_layers"]:
            problems.append("layer_types length != num_hidden_layers")
    if problems:
        raise ValueError("Unsupported d1-omni configuration:\n  " + "\n  ".join(problems))


def check_text_config(text_config: dict) -> None:
    """The structural assumptions this module makes, for any size (the toy configs included)."""
    d, heads, kv = text_config["hidden_size"], text_config["num_attention_heads"], text_config["num_key_value_heads"]
    if d % heads or heads % kv or (d // heads) % 2:
        raise ValueError(f"hidden {d} / heads {heads} / kv heads {kv}: need d % heads == 0, heads % kv == 0, even head_dim")
    if d % 64:
        raise ValueError("the head uses d // 64 heads of 64: hidden_size must be a multiple of 64")
    if text_config["conv_L_cache"] != 3:
        raise ValueError("the centred short convolution is written for 3 taps")
    unknown = set(text_config["layer_types"]) - {"conv", "full_attention"}
    if unknown:
        raise ValueError(f"unknown layer types {sorted(unknown)}")


def rope_tables(text_config: dict, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """encoder.Trunk.rope, run once on the CPU in fp32 for positions 0..seq_len-1: cos, sin [1, 1, L, head_dim]."""
    head_dim = text_config["hidden_size"] // text_config["num_attention_heads"]
    exponent = torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim
    inv_freq = 1.0 / (text_config["rope_theta"] ** exponent)
    positions = torch.arange(seq_len).float()
    freqs = (inv_freq[None, :, None] @ positions[None, None, :]).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[:, None].contiguous(), emb.sin()[:, None].contiguous()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# --------------------------------------------------------------------------- modules (names = checkpoint names)
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps


class Attention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["hidden_size"]
        self.heads, self.kv_heads = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.head_dim = d // self.heads
        self.q_proj = nn.Linear(d, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.heads * self.head_dim, d, bias=False)
        self.q_layernorm = RMSNorm(self.head_dim, cfg["norm_eps"])
        self.k_layernorm = RMSNorm(self.head_dim, cfg["norm_eps"])


class _DepthwiseFilter(nn.Module):
    """Holds the depthwise filter under the checkpoint's name `conv.conv.weight` [d, 1, 3]."""

    def __init__(self, d: int, taps: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d, 1, taps))


class ShortConv(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["hidden_size"]
        self.conv = _DepthwiseFilter(d, cfg["conv_L_cache"])
        self.in_proj = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)


class MLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d, hidden = cfg["hidden_size"], mlp_hidden(cfg)
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)


class Layer(nn.Module):
    def __init__(self, cfg: dict, kind: str):
        super().__init__()
        self.is_attention_layer = kind == "full_attention"
        if self.is_attention_layer:
            self.self_attn = Attention(cfg)
        else:
            self.conv = ShortConv(cfg)
        self.feed_forward = MLP(cfg)
        self.operator_norm = RMSNorm(cfg["hidden_size"], cfg["norm_eps"])
        self.ffn_norm = RMSNorm(cfg["hidden_size"], cfg["norm_eps"])


class Trunk(nn.Module):
    def __init__(self, cfg: dict, seq_len: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList(Layer(cfg, kind) for kind in cfg["layer_types"])
        self.embedding_norm = RMSNorm(cfg["hidden_size"], cfg["norm_eps"])
        cos, sin = rope_tables(cfg, seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)


class _HeadAttention(nn.Module):
    """nn.MultiheadAttention's parameters under its names (in_proj_weight / in_proj_bias / out_proj)."""

    def __init__(self, d: int):
        super().__init__()
        self.in_proj_weight = nn.Parameter(torch.zeros(3 * d, d))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * d))
        self.out_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The in-projection as this module's own call (quant mode: where a module-level quantizer sees it)."""
        return F.linear(x, self.in_proj_weight, self.in_proj_bias)


class HeadLayer(nn.Module):
    """nn.TransformerEncoderLayer(d, d // 64, 4 * d, dropout 0, batch_first, norm_first=True), eval, explicit."""

    def __init__(self, d: int):
        super().__init__()
        self.self_attn = _HeadAttention(d)
        self.linear1 = nn.Linear(d, 4 * d)
        self.linear2 = nn.Linear(4 * d, d)
        self.norm1 = nn.LayerNorm(d, eps=1e-5)
        self.norm2 = nn.LayerNorm(d, eps=1e-5)


class _HeadStack(nn.Module):
    def __init__(self, d: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList(HeadLayer(d) for _ in range(layers))


class DecisionHead(nn.Module):
    def __init__(self, d: int, layers: int):
        super().__init__()
        self.type_emb = nn.Embedding(3, d)
        self.head = _HeadStack(d, layers)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))


# --------------------------------------------------------------------------- the graph
class D1Decide(nn.Module):
    """Trunk + decision head for one row of static length `seq_len`. Parameter names are the checkpoint's with
    `encoder.` -> `trunk.` (KEY_MAP); the RoPE tables are non-persistent buffers."""

    def __init__(self, text_config: dict, head_layers: int = 2, seq_len: int = 256, mutation: str = "none",
                 gqa: str = "repeat", key_block: int = KEY_BLOCK):
        super().__init__()
        check_text_config(text_config)
        if mutation not in MUTATIONS:
            raise ValueError(f"mutation must be one of {MUTATIONS}")
        if gqa not in GQA_FORMS:
            raise ValueError(f"gqa must be one of {GQA_FORMS}")
        if key_block < 1:
            raise ValueError("key_block must be positive")
        self.text_config = dict(text_config)
        self.seq_len, self.mutation, self.gqa, self.key_block = int(seq_len), mutation, gqa, int(key_block)
        d = text_config["hidden_size"]
        self.hidden = d
        self.head_heads = d // 64
        self.trunk = Trunk(text_config, self.seq_len)
        self.head = DecisionHead(d, head_layers)
        self.precision, self.compute_dtype = "fp32", torch.float32
        self.softmax_dtype = torch.float32  # every precision; a float64 diagnostic may raise it
        self.quant = False

    # ---- precision
    def set_precision(self, precision: str) -> "D1Decide":
        """fp32 / wfp16 / fp16 (module docstring). Norm weights stay fp32 tensors in every mode."""
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        storage = torch.float32 if precision == "fp32" else torch.float16
        for module in self.modules():
            if isinstance(module, (RMSNorm, nn.LayerNorm)):
                continue
            for param in module.parameters(recurse=False):
                param.data = param.data.to(storage)
        self.precision = precision
        self.compute_dtype = torch.float16 if precision == "fp16" else torch.float32
        return self

    def set_quant_mode(self) -> "D1Decide":
        """Quant mode (the int8 export): every trunk and head linear runs as its module's own call -- nn.Linear's
        forward, and _HeadAttention's for the head's in-projection -- so a module-level quantizer (coreai-opt's
        eager quantizer, through coreai_models.export.compression.quantize_pytorch_model) sees each weight where it
        is used and can replace it. The embedding, the norms, the depthwise filter, type_emb and the scorer keep their
        paths. Only on the fp16 frame: there every one of those linears takes an fp16 input with fp16 weights, so the
        module is called with the activation as it is (the module's dtype), and the numbers equal the fp16 frame's
        until a quantizer changes a weight."""
        if self.precision != "fp16":
            raise ValueError("quant mode runs on the fp16 frame: set_precision('fp16') first")
        self.quant = True
        return self

    # ---- pieces (each weight is cast to the activation dtype at its use: a no-op in fp32)
    @staticmethod
    def _lin(x: torch.Tensor, module: nn.Module | None = None, weight=None, bias=None) -> torch.Tensor:
        if module is not None:
            weight, bias = module.weight, module.bias
        return F.linear(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))

    def _linear(self, x: torch.Tensor, module: nn.Linear) -> torch.Tensor:
        """A trunk / head linear: the module's own call in quant mode, else _lin."""
        return module(x) if self.quant else self._lin(x, module)

    @staticmethod
    def _rms(norm: RMSNorm, x: torch.Tensor) -> torch.Tensor:
        """encoder.RMSNorm: computed in fp32, then weight * x in the input dtype."""
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + norm.eps)
        return norm.weight.to(dtype) * y.to(dtype)

    def _ln(self, norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        y = F.layer_norm(x.float(), norm.normalized_shape, norm.weight.float(), norm.bias.float(), norm.eps)
        return y.to(x.dtype)

    def _attend(self, q, k, v, mask, scale: float) -> torch.Tensor:
        """Raw attention: (q k^T) * scale + mask, softmax over keys in fp32, then * v; above key_block keys, in key
        blocks sharing one max (module docstring, "Key blocks")."""
        keys, dtype = k.shape[-2], self.softmax_dtype
        if keys <= self.key_block:
            scores = torch.matmul(q, k.transpose(-1, -2)).to(dtype) * scale + mask
            return torch.matmul(torch.softmax(scores, dim=-1).to(v.dtype), v)
        bounds = [(a, min(a + self.key_block, keys)) for a in range(0, keys, self.key_block)]
        s = [torch.matmul(q, k[..., a:z, :].transpose(-1, -2)).to(dtype) * scale + mask[..., a:z] for a, z in bounds]
        m = s[0].amax(dim=-1, keepdim=True)
        for x in s[1:]:
            m = torch.maximum(m, x.amax(dim=-1, keepdim=True))
        e = [torch.exp(x - m) for x in s]
        den = e[0].sum(dim=-1, keepdim=True)
        for x in e[1:]:
            den = den + x.sum(dim=-1, keepdim=True)
        y = torch.matmul((e[0] / den).to(v.dtype), v[..., bounds[0][0]:bounds[0][1], :])
        for x, (a, z) in zip(e[1:], bounds[1:]):
            y = y + torch.matmul((x / den).to(v.dtype), v[..., a:z, :])
        return y

    def _attention(self, attn: Attention, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        L, heads, kv, hd = self.seq_len, attn.heads, attn.kv_heads, attn.head_dim
        q = self._rms(attn.q_layernorm, self._linear(x, attn.q_proj).reshape(1, L, heads, hd)).transpose(1, 2)
        k = self._rms(attn.k_layernorm, self._linear(x, attn.k_proj).reshape(1, L, kv, hd)).transpose(1, 2)
        v = self._linear(x, attn.v_proj).reshape(1, L, kv, hd).transpose(1, 2)
        cos, sin = self.trunk.rope_cos.to(q.dtype), self.trunk.rope_sin.to(q.dtype)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        groups, scale = heads // kv, hd ** -0.5
        if self.gqa == "repeat":
            y = self._attend(q, k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1), mask, scale)
        else:  # query heads grouped per key/value head, keys and values broadcast over the group axis
            y = self._attend(q.reshape(1, kv, groups, L, hd), k[:, :, None], v[:, :, None], mask[:, :, None], scale)
            y = y.reshape(1, heads, L, hd)
        return self._linear(y.transpose(1, 2).reshape(1, L, heads * hd), attn.out_proj)

    def _short_conv(self, conv: ShortConv, x: torch.Tensor, pad3: torch.Tensor, keep3: torch.Tensor) -> torch.Tensor:
        """encoder.ShortConv in [1, L, D] layout, the same products and sums in the same order."""
        L = self.seq_len
        b, c, u = self._linear(x * pad3, conv.in_proj).chunk(3, dim=-1)
        bx = b * u
        w = conv.conv.weight[:, 0, :].to(bx.dtype)
        xp = F.pad(bx, (0, 0, 1, 1))
        right = xp[:, 2:2 + L] * keep3  # media never reads the text
        y = xp[:, 0:L] * w[:, 0]
        y = y + xp[:, 1:1 + L] * w[:, 1]
        y = y + right * w[:, 2]
        return self._linear(c * y, conv.out_proj)

    def _mlp(self, mlp: MLP, x: torch.Tensor) -> torch.Tensor:
        return self._linear(F.silu(self._linear(x, mlp.w1)) * self._linear(x, mlp.w3), mlp.w2)

    def _head_layer(self, layer: HeadLayer, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        L, d, heads = self.seq_len, self.hidden, self.head_heads
        hd = d // heads
        x = self._ln(layer.norm1, h)
        if self.quant:
            qkv = layer.self_attn(x)
        else:
            qkv = self._lin(x, None, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
        q, k, v = (t.reshape(1, L, heads, hd).transpose(1, 2) for t in qkv.chunk(3, dim=-1))
        y = self._attend(q, k, v, mask, hd ** -0.5)
        h = h + self._linear(y.transpose(1, 2).reshape(1, L, d), layer.self_attn.out_proj)
        x = F.relu(self._linear(self._ln(layer.norm2, h), layer.linear1))
        return h + self._linear(x, layer.linear2)

    def _scorer(self, h: torch.Tensor) -> torch.Tensor:
        norm, first, _, last = self.head.scorer
        x = self._ln(norm, h.float())
        x = F.gelu(self._lin(x, first))
        return self._lin(x, last).reshape(1, self.seq_len).float()

    def masks(self, pad_mask, prefix_mask, keep_right):
        """The three float masks the graph builds from the vector inputs (exposed for the checks)."""
        L = self.seq_len
        pad = pad_mask.float().reshape(1, L)
        prefix = prefix_mask.float().reshape(1, L)
        keep = torch.ones_like(pad) if self.mutation == "keep_right_ones" else keep_right.float().reshape(1, L)
        trunk_mask = ((1.0 - pad) * MASK_VALUE).reshape(1, 1, 1, L)
        if self.mutation != "no_media_text_mask":  # media query x text key (pad keys count as text, as upstream)
            trunk_mask = trunk_mask + prefix.reshape(1, 1, L, 1) * (1.0 - prefix.reshape(1, 1, 1, L)) * MASK_VALUE
        text = pad * (1.0 - prefix)
        head_mask = ((1.0 - text) * MASK_VALUE).reshape(1, 1, 1, L)
        if self.mutation == "no_head_key_mask":
            head_mask = torch.zeros_like(head_mask)
        return trunk_mask, head_mask, pad, prefix, keep

    def forward(self, input_ids, prefix_embeds, pad_mask, prefix_mask, keep_right, qtype_onehot):
        L, d, dtype = self.seq_len, self.hidden, self.compute_dtype
        if tuple(input_ids.shape) != (1, L) or tuple(prefix_embeds.shape) != (1, L, d):
            raise ValueError(f"expected input_ids [1,{L}] and prefix_embeds [1,{L},{d}], got "
                             f"{tuple(input_ids.shape)} and {tuple(prefix_embeds.shape)}")
        trunk_mask, head_mask, pad, prefix, keep = self.masks(pad_mask, prefix_mask, keep_right)
        emb = F.embedding(input_ids, self.trunk.embed_tokens.weight).to(dtype)
        prefix3 = prefix.reshape(1, L, 1).to(dtype)
        h = prefix3 * prefix_embeds.to(dtype) + (1.0 - prefix3) * emb
        pad3, keep3 = pad.reshape(1, L, 1).to(dtype), keep.reshape(1, L, 1).to(dtype)
        for layer in self.trunk.layers:
            x = self._rms(layer.operator_norm, h)
            if layer.is_attention_layer:
                x = self._attention(layer.self_attn, x, trunk_mask)
            else:
                x = self._short_conv(layer.conv, x, pad3, keep3)
            h = h + x
            h = h + self._mlp(layer.feed_forward, self._rms(layer.ffn_norm, h))
        h = self._rms(self.trunk.embedding_norm, h)
        table = self.head.type_emb.weight
        h = h + torch.matmul(qtype_onehot.to(table.dtype), table).reshape(1, 1, d).to(dtype)
        for layer in self.head.head.layers:
            h = self._head_layer(layer, h, head_mask)
        return self._scorer(h)


# --------------------------------------------------------------------------- weights
def map_key(checkpoint_key: str) -> str | None:
    """A model.safetensors key -> this module's key; None for the vision / audio tensors (other graphs)."""
    for source, target in KEY_MAP:
        if checkpoint_key.startswith(source):
            return target + checkpoint_key[len(source):]
    if checkpoint_key.startswith(IGNORED_PREFIXES):
        return None
    raise KeyError(f"unexpected checkpoint key {checkpoint_key!r}")


def key_table(header: dict) -> dict[str, str]:
    """checkpoint key -> module key for every trunk / head tensor of a safetensors header (or key list)."""
    return {k: m for k in header if k != "__metadata__" and (m := map_key(k)) is not None}


def load_weights(model: D1Decide, safetensors_path: str | Path, verify_sha256: bool = False) -> dict:
    """Strict load of the trunk and head tensors (fp32) from model.safetensors; vision / audio are skipped and
    counted. Returns a record of what was read."""
    from safetensors import safe_open

    path = Path(safetensors_path)
    if path.stat().st_size != WEIGHTS["bytes"]:
        raise ValueError(f"{path}: {path.stat().st_size} bytes, expected {WEIGHTS['bytes']}")
    if verify_sha256:
        import hashlib
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 24), b""):
                digest.update(block)
        if digest.hexdigest() != WEIGHTS["sha256"]:
            raise ValueError(f"{path}: sha256 {digest.hexdigest()} != {WEIGHTS['sha256']}")
    state, skipped, counts = {}, 0, {"encoder": 0, "head": 0}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            target = map_key(key)
            if target is None:
                skipped += 1
                continue
            tensor = handle.get_tensor(key)
            if tensor.dtype != torch.float32:
                raise ValueError(f"{key}: {tensor.dtype}, the checkpoint is float32")
            counts[key.split(".")[0]] += tensor.numel()
            state[target] = tensor
    expected = {k: PARAMETERS[k] for k in counts}
    if counts != expected:
        raise ValueError(f"parameter counts {counts} != {expected}")
    model.load_state_dict(state, strict=True)
    return {"tensors": len(state), "skipped_vision_audio": skipped, "parameters": counts,
            "sha256_verified": verify_sha256}


def build(config: dict, seq_len: int, mutation: str = "none", gqa: str = "repeat",
          key_block: int = KEY_BLOCK) -> D1Decide:
    """validate_config, then the module for one bucket length (weights not loaded)."""
    validate_config(config)
    return D1Decide(config["text_config"], config["head_layers"], seq_len, mutation=mutation, gqa=gqa,
                    key_block=key_block)


def load_d1_decide(snapshot_dir: str | Path, seq_len: int, precision: str = "fp32", mutation: str = "none",
                   gqa: str = "repeat", verify_sha256: bool = False,
                   key_block: int = KEY_BLOCK) -> tuple[D1Decide, dict]:
    """The pinned snapshot's config.json + model.safetensors -> the eval-mode graph module."""
    import json

    snapshot = Path(snapshot_dir)
    model = build(json.loads((snapshot / "config.json").read_text()), seq_len, mutation, gqa, key_block)
    record = load_weights(model, snapshot / WEIGHTS["file"], verify_sha256)
    model.eval().requires_grad_(False)
    return model.set_precision(precision), record


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def expected_shapes(header: dict) -> dict[str, list[int]]:
    """module key -> shape for the trunk / head tensors of a safetensors header."""
    return {m: header[k]["shape"] for k, m in key_table(header).items()}


def _self_check() -> None:  # python3 d1_omni_model.py: config + module structure against the pinned files
    import json
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _paths import hf_snapshot  # noqa: E402

    snapshot = Path(hf_snapshot(MODEL_ID, revision=MODEL_SHA))
    config = json.loads((snapshot / "config.json").read_text())
    validate_config(config)
    with torch.device("meta"):
        model = build(config, 256)
    shapes = {k: list(v.shape) for k, v in model.state_dict().items()}
    print(f"config ok; module parameters {parameter_count(model):,} "
          f"(trunk {parameter_count(model.trunk):,}, head {parameter_count(model.head):,}); {len(shapes)} tensors")


if __name__ == "__main__":
    _self_check()
