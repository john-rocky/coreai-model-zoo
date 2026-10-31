# LFM3 text decoder (conv + full-attention + sliding-attention hybrid) for the
# Core AI authoring path.
#
# Community port — NOT an Apple model.  Authored to be exportable via
# coreai_models.export.macos.export_to_coreai.  Decoupled from transformers:
# weights load directly from the HF safetensors, config is parsed from the raw
# config.json (or a sub-dict of it), so the export env needs no LFM3 support.
#
# Built on lfm2.py: the short-conv mixer (Lfm2ShortConv), the MLP / RMSNorm /
# SDPA / KVCache primitives, the state names and the fused conv write-back are
# imported from there unchanged. What LFM3 changes (HF Lfm3, modeling_lfm3.py,
# read against the checkpoint's own tensors):
#   - a THIRD layer type, "sliding_attention" (window 256), beside "conv" and
#     "full_attention" (1.7B: 13 conv + 6 full + 5 sliding)
#   - attention: per-head q/k RMSNorm (as LFM2) + a per-head SIGMOID OUTPUT
#     GATE: gate_proj maps the attention input to one logit per head, and
#     sigmoid(gate) scales that head's output before o_proj
#   - two RoPEs: full layers rotate with theta 1e6 and partial_rotary_factor
#     0.5 in the ZERO-TAIL layout — the first 32 frequencies keep the
#     progression of the full head_dim (theta^(-2i/128)) and the last 32 are
#     zero, so dims 32-63 and 96-127 stay unrotated (cos 1 / sin 0). It is NOT
#     "rotate the first 64 dims". Sliding layers rotate all dims, theta 1e4
#   - SANDWICH norm: post_operator_norm / post_ffn_norm normalize each mixer /
#     MLP output before it enters the residual stream
#   - the short conv is width 7 (conv_L_cache 7) -> a 6-column decode state
#   - the MLP width is intermediate_size as stated (no LFM2 ff_dim adjust)
#
# Sliding window: full-length KV (ONE KV pair for all attention layers, full
# and sliding, slot = attention-layer counter) and the SDPA composite's own
# window (`window_size`, lower-right causal): a query at absolute position q
# sees keys k with q - k < window, which is HF's `kv_idx > q_idx - window`.
# Same form as the stock gemma3_text.py. A graph with a static S > 1 query
# (chunk prefill) needs the masks built outside the composite instead —
# `Lfm3Model.set_explicit_mask()`. A ring-buffer KV for the sliding layers is
# a state lever for a speed round, not this module.
#
# IMPORTANT (macOS-27-beta GPU delegate, inherited from lfm2.py #1): conv
# layers RETURN their new state columns and the stack issues ONE fused
# full-state write per step; chained per-slot writes on one state handle are
# dropped by the delegate.
#
# IMPORTANT #2 (inherited from lfm2.py #2, widened): the attention
# projections run in fp32 on the fp16 export (fp32_attn_proj) — the delegate's
# fused attention prologue accumulates fp16 under a dynamic-shape graph and
# the q/k-norm gains amplify it. LFM3 adds gate_proj and the q/k norm weights
# to the fp32 set and keeps q/k fp32 through the norm and the RoPE (HF applies
# RoPE in fp32); they drop to the cache dtype only at the KV write. The gate
# (sigmoid and its product with the attention output) runs in the gate_proj
# dtype, then o_proj. Conv-mixer and MLP matmuls stay in the export dtype.
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.models.macos.lfm2 import (
    _MLP_KEY_MAP,
    DECODE_STATE_NAMES,
    Lfm2ShortConv,
    _rotate_half,
)
from coreai_models.primitives._ops import mutable_slice_update
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.sdpa import SDPA

CONV = "conv"
FULL = "full_attention"
SLIDING = "sliding_attention"
ATTENTION_TYPES = (FULL, SLIDING)


@dataclass
class Lfm3Config:
    """Subset of the LFM3 config needed for authoring (from config.json, or the
    ``text_config`` / ``lfm`` sub-dict of a VL / Omni / Audio checkpoint)."""

    hidden_size: int = 2048
    num_hidden_layers: int = 24
    vocab_size: int = 128000
    intermediate_size: int = 7680
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 128
    norm_eps: float = 1e-5
    conv_L_cache: int = 7
    conv_bias: bool = False
    sliding_window: int = 256
    use_sandwich_norm: bool = True
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 262144
    bos_token_id: int | None = 124894
    eos_token_id: int | list[int] | None = 124900
    pad_token_id: int | None = 124893
    layer_types: list[str] = field(default_factory=list)
    # {"full_attention": {"rope_theta", "partial_rotary_factor"?, "rope_type"},
    #  "sliding_attention": {...}} — keyed by layer type, conv has no entry
    rope_parameters: dict = field(default_factory=dict)

    def is_attention(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] in ATTENTION_TYPES

    @property
    def attention_layer_ids(self) -> list[int]:
        """Layer index of each KV slot (slot i holds layer attention_layer_ids[i])."""
        return [i for i in range(self.num_hidden_layers) if self.is_attention(i)]

    @property
    def conv_layer_ids(self) -> list[int]:
        """Layer index of each conv-state slot."""
        return [i for i in range(self.num_hidden_layers) if not self.is_attention(i)]

    @property
    def num_attention_layers(self) -> int:
        return len(self.attention_layer_ids)

    @property
    def num_conv_layers(self) -> int:
        return len(self.conv_layer_ids)

    @property
    def conv_state_width(self) -> int:
        # Minimal decode conv state = kernel-1 columns of gated input (Bx).
        return self.conv_L_cache - 1


def lfm3_config_from_dict(raw: dict) -> Lfm3Config:
    """Build the authoring config from a raw LFM3 config dict, with the HF
    Lfm3Config defaults for the keys a dict may omit (configuration_lfm3.py
    __post_init__): head_dim, rope_parameters, layer_types."""
    n_layers = raw.get("num_hidden_layers", 24)
    n_heads = raw.get("num_attention_heads", 16)
    hidden = raw.get("hidden_size", 2048)
    layer_types = raw.get("layer_types") or [
        FULL if i % 4 == 2 else SLIDING if i % 4 == 0 and i > 0 else CONV
        for i in range(n_layers)
    ]
    rope = raw.get("rope_parameters") or {
        FULL: {"rope_type": "default", "rope_theta": 1_000_000.0, "partial_rotary_factor": 0.5},
        SLIDING: {"rope_type": "default", "rope_theta": 10_000.0},
    }
    for t in set(layer_types) & set(ATTENTION_TYPES):
        if t not in rope:
            raise ValueError(f"rope_parameters has no entry for the {t} layers")
        if rope[t].get("rope_type", "default") != "default":
            # HF evaluates scaled types over the full head_dim and truncates;
            # none of the LFM3 checkpoints uses one, so do not guess.
            raise NotImplementedError(f"rope_type {rope[t]['rope_type']!r} on {t} layers")
    # HF spells the tie flag `tie_word_embeddings`; Lfm3Config also accepts
    # `tie_embedding` (popped into the same field).
    tie = raw.get("tie_word_embeddings")
    if tie is None:
        tie = raw.get("tie_embedding")
    return Lfm3Config(
        hidden_size=hidden,
        num_hidden_layers=n_layers,
        vocab_size=raw.get("vocab_size", 128000),
        intermediate_size=raw.get("intermediate_size", 7680),
        num_attention_heads=n_heads,
        num_key_value_heads=raw.get("num_key_value_heads", 4),
        head_dim=raw.get("head_dim") or hidden // n_heads,
        norm_eps=raw.get("norm_eps", 1e-5),
        conv_L_cache=raw.get("conv_L_cache", 7),
        conv_bias=raw.get("conv_bias", False),
        sliding_window=raw.get("sliding_window", 256),
        use_sandwich_norm=raw.get("use_sandwich_norm", True),
        tie_word_embeddings=True if tie is None else bool(tie),
        max_position_embeddings=raw.get("max_position_embeddings", 128000),
        bos_token_id=raw.get("bos_token_id"),
        eos_token_id=raw.get("eos_token_id"),
        pad_token_id=raw.get("pad_token_id"),
        layer_types=list(layer_types),
        rope_parameters={k: dict(v) for k, v in rope.items() if isinstance(v, dict)},
    )


# --------------------------------------------------------------------------- #
# RoPE — zero-tail partial rotary, cos/sin and the rotation in fp32
# --------------------------------------------------------------------------- #
def rope_inv_freq(config: Lfm3Config, layer_type: str) -> torch.Tensor:
    """HF Lfm3RotaryEmbedding.compute_default_rope_parameters: the first
    ``int(partial * head_dim) // 2`` inverse frequencies follow the progression
    of the FULL head_dim, the rest are zero (those dims stay unrotated).

    Always fp32 on CPU (explicit device, so it is real even when the model is
    built under ``torch.device("meta")``)."""
    p = config.rope_parameters[layer_type]
    base = p["rope_theta"]
    dim = config.head_dim
    angles = int(p.get("partial_rotary_factor", 1.0) * dim) // 2
    inv = 1.0 / (base ** (torch.arange(0, 2 * angles, 2, dtype=torch.float32, device="cpu") / dim))
    return torch.cat((inv, torch.zeros(dim // 2 - angles, dtype=torch.float32, device="cpu")))


def apply_rope_fp32(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q,k: [b, n_heads, s, head_dim]; cos,sin: [b, s, head_dim] fp32.

    The rotation runs in fp32 like HF's apply_rotary_pos_emb (no cast of the
    tables to the activation dtype, unlike lfm2.apply_rope); results come
    back in fp32 — the caller casts to the cache dtype."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = q.float()
    k = k.float()
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


# --------------------------------------------------------------------------- #
# Attention (GQA, per-head q/k RMSNorm, sigmoid output gate, full or sliding)
# --------------------------------------------------------------------------- #
class Lfm3Attention(nn.Module):
    def __init__(self, config: Lfm3Config, layer_type: str) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        d = config.hidden_size
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.gate_proj = nn.Linear(d, self.n_heads, bias=False)
        window = config.sliding_window if layer_type == SLIDING else 0
        self.sdpa = SDPA(scale=self.head_dim**-0.5, is_causal=True, window_size=window)

    def gate(self, xp: torch.Tensor) -> torch.Tensor:
        """Per-head output gate [b, s, n_heads] in the gate_proj dtype."""
        return torch.sigmoid(self.gate_proj(xp))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        offset: int = 0,
        seq_len: int | None = None,
        attn_idx: int = 0,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prefill when ``kv_cache`` is None (causal / windowed SDPA over the
        query block). With a cache, write the query's k/v at ``offset`` into
        slot ``attn_idx`` and attend over [0:seq_len] (prefill AND decode).
        ``attn_mask`` (bool, True = attend) replaces the composite's own
        causal / window mask — see ``Lfm3Model.set_explicit_mask``."""
        b, s, _ = x.shape
        H, HKV, D = self.n_heads, self.n_kv_heads, self.head_dim

        # q/k/v/gate projections and the q/k norms run in the weights' dtype
        # (fp32 on the fp16 export — IMPORTANT #2); no-op in the fp32 tier.
        xp = x.to(self.q_proj.weight.dtype)
        q = self.q_norm(self.q_proj(xp).view(b, s, H, D)).transpose(1, 2)
        k = self.k_norm(self.k_proj(xp).view(b, s, HKV, D)).transpose(1, 2)
        v = self.v_proj(xp).to(x.dtype).view(b, s, HKV, D).transpose(1, 2)

        q, k = apply_rope_fp32(q, k, cos, sin)
        q, k = q.to(x.dtype), k.to(x.dtype)
        if kv_cache is not None:
            k, v = kv_cache.update_and_fetch(
                attn_idx, offset, k, v, seq_len=seq_len, query_len=s
            )
        out = self.sdpa(q, k, v, attn_mask)  # GQA + causal (+ window); [b,H,s,D]

        g = self.gate(xp)  # [b, s, H]
        out = out.transpose(1, 2).to(g.dtype) * g.unsqueeze(-1)
        out = out.reshape(b, s, H * D)
        return self.o_proj(out.to(self.o_proj.weight.dtype)).to(x.dtype)


# --------------------------------------------------------------------------- #
# Decoder layer — conv / full / sliding mixer, sandwich norm
# --------------------------------------------------------------------------- #
class Lfm3DecoderLayer(nn.Module):
    def __init__(self, config: Lfm3Config, layer_idx: int) -> None:
        super().__init__()
        d = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        self.is_attention = self.layer_type in ATTENTION_TYPES
        self.use_sandwich_norm = config.use_sandwich_norm
        self.operator_norm = RMSNorm(d, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(d, eps=config.norm_eps)
        if self.is_attention:
            self.self_attn = Lfm3Attention(config, self.layer_type)
        else:
            self.conv = Lfm2ShortConv(config)  # width conv_L_cache (7)
        self.feed_forward = MLP(d, config.intermediate_size)
        if self.use_sandwich_norm:
            self.post_operator_norm = RMSNorm(d, eps=config.norm_eps)
            self.post_ffn_norm = RMSNorm(d, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        kv_cache: KVCache | None = None,
        conv_in: torch.Tensor | None = None,
        offset: int = 0,
        seq_len: int | None = None,
        attn_idx: int = 0,
        attn_mask: torch.Tensor | None = None,
    ):
        """Conv layers in the stateful path (``conv_in`` given) return
        ``(hidden, new_conv)`` — the caller issues the fused state write."""
        normed = self.operator_norm(x)
        new_conv = None
        if self.is_attention:
            r = self.self_attn(normed, cos, sin, kv_cache, offset, seq_len, attn_idx, attn_mask)
        elif conv_in is None:
            r = self.conv(normed)
        else:
            r, new_conv = self.conv(normed, conv_in)
        if self.use_sandwich_norm:
            r = self.post_operator_norm(r)
        h = x + r
        f = self.feed_forward(self.ffn_norm(h))
        if self.use_sandwich_norm:
            f = self.post_ffn_norm(f)
        h = h + f
        if conv_in is not None:
            return h, new_conv
        return h


# --------------------------------------------------------------------------- #
# Decoder stack (in-graph RoPE from position_ids)
# --------------------------------------------------------------------------- #
class Lfm3Model(nn.Module):
    """The LFM3 decoder without a head. Public for wrappers (VL / Omni /
    Audio): ``embed_tokens`` is the token table, and every stateful entry point
    takes the same ``(position_ids, KVCache, conv_state)`` contract whether the
    query arrives as token ids or as embeddings."""

    def __init__(self, config: Lfm3Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Lfm3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        # final norm before the head; HF names it embedding_norm
        self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        # Plain fp32 attributes, NOT buffers: `.to(fp16)` must not round the
        # small frequencies (conversion-guide.md, "fp16 conversion").
        self.inv_freq = {
            t: rope_inv_freq(config, t)
            for t in ATTENTION_TYPES
            if t in config.layer_types
        }
        self.explicit_mask = False

    def set_explicit_mask(self, on: bool = True) -> None:
        """Build the causal / window masks from positions outside the SDPA
        composite and pass them in (bool, True = attend), with every composite
        reconfigured to is_causal=False, window_size=0.

        Needed to export a graph whose query length is a static S > 1 over a
        dynamic KV (a chunk-prefill graph): the composite's own lower-right
        mask asserts `k_len - q_len >= 0`, and its externalized submodule is
        exported without the KV dim's lower bound, so torch.export raises
        `Constraints violated` (S = 1 never trips it). Same masks as the
        composite's (index-based, q - k < window). Off = the S=1 ship graph."""
        self.explicit_mask = on
        for layer in self.layers:
            if layer.is_attention:
                a = layer.self_attn
                a.sdpa.is_causal = not on
                a.sdpa.window_size = 0 if on or a.layer_type != SLIDING else self.config.sliding_window

    def attention_masks(self, query_len: int, seq_len: int, offset: int) -> dict:
        """{layer_type: bool [query_len, seq_len]} for ``set_explicit_mask``,
        else {} (the composites mask by themselves)."""
        if not self.explicit_mask:
            return {}
        diff = (torch.arange(query_len) + offset)[:, None] - torch.arange(seq_len)[None, :]
        causal = diff >= 0
        return {FULL: causal, SLIDING: causal & (diff < self.config.sliding_window)}

    def rope_cos_sin(self, position_ids: torch.Tensor) -> dict:
        """position_ids [b,s] -> {layer_type: (cos, sin)}, each [b,s,head_dim] fp32."""
        pos = position_ids[..., None].float()
        tables = {}
        for t, inv_freq in self.inv_freq.items():
            freqs = pos * inv_freq  # [b,s,head_dim/2]
            emb = torch.cat([freqs, freqs], dim=-1)  # [b,s,head_dim]
            tables[t] = (emb.cos(), emb.sin())
        return tables

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return self.forward_embeds(self.embed_tokens(input_ids), position_ids)

    def forward_embeds(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Stateless prefill over the whole query block -> final-norm hidden."""
        rope = self.rope_cos_sin(position_ids)
        s = inputs_embeds.shape[1]
        masks = self.attention_masks(s, s, 0)
        h = inputs_embeds
        for layer in self.layers:
            cos, sin = rope.get(layer.layer_type, (None, None))
            h = layer(h, cos, sin, attn_mask=masks.get(layer.layer_type))
        return self.embedding_norm(h)

    def forward_stateful(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: KVCache,
        conv_state: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_stateful_embeds(
            self.embed_tokens(input_ids), position_ids, kv_cache, conv_state
        )

    def forward_stateful_embeds(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: KVCache,
        conv_state: torch.Tensor,
    ) -> torch.Tensor:
        """Stateful prefill/decode -> final-norm hidden [b, query_len, hidden].

        ``inputs_embeds`` carries the query ([b, query_len, hidden], any
        query_len >= 1); ``position_ids`` carries the FULL positions
        ([b, seq_len]) so ``offset = seq_len - query_len``. Attention layers
        index ``kv_cache`` by attention-layer counter (full and sliding share
        the pair); conv layers read slot conv_idx of ``conv_state`` and the
        updated columns are written back in ONE fused slice_update at the end
        (IMPORTANT #1).
        """
        query_len = inputs_embeds.shape[1]
        seq_len = position_ids.shape[1]
        offset = seq_len - query_len
        q_pos = position_ids.narrow(1, offset, query_len)
        rope = self.rope_cos_sin(q_pos)
        masks = self.attention_masks(query_len, seq_len, offset)
        h = inputs_embeds
        attn_idx = 0
        conv_idx = 0
        new_convs: list[torch.Tensor] = []
        for layer in self.layers:
            if layer.is_attention:
                cos, sin = rope[layer.layer_type]
                h = layer(h, cos, sin, kv_cache=kv_cache, offset=offset,
                          seq_len=seq_len, attn_idx=attn_idx,
                          attn_mask=masks.get(layer.layer_type))
                attn_idx += 1
            else:
                conv_in = conv_state.narrow(0, conv_idx, 1).squeeze(0)
                h, new_conv = layer(h, None, None, conv_in=conv_in)
                new_convs.append(new_conv)
                conv_idx += 1
        new_full = torch.cat([n.unsqueeze(0) for n in new_convs], dim=0)
        zero = torch.tensor((0,), dtype=torch.int32)
        mutable_slice_update(
            x=conv_state,
            update=new_full,
            begin=torch.cat([zero, zero, zero, zero]),
            end=torch.tensor(tuple(conv_state.shape), dtype=torch.int32),
        )
        return self.embedding_norm(h)


class _Lfm3HeadMixin(nn.Module):
    """Tied LM head + the optional second output (final-norm ``hidden``)."""

    def _init_head(self, config: Lfm3Config, emit_hidden: bool) -> None:
        self.config = config
        self.model = Lfm3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        # emit_hidden: ONE graph returns (logits, hidden), hidden = the
        # embedding_norm output that feeds the head.
        self.emit_hidden = emit_hidden
        # When True the graph returns only the last token's rows -> a static
        # [b, 1, ...] output even under a dynamic query length.
        self.last_token_only = False

    def _head(self, h: torch.Tensor):
        if self.last_token_only:
            h = h[:, -1:, :]
        logits = self.lm_head(h)
        if self.emit_hidden:
            return logits, h
        return logits

    @property
    def output_names(self) -> tuple[str, ...]:
        return ("logits", "hidden") if self.emit_hidden else ("logits",)


class Lfm3ForCausalLM(_Lfm3HeadMixin):
    """Prefill text decoder + tied LM head (no cache; causal SDPA over S)."""

    def __init__(self, config: Lfm3Config, emit_hidden: bool = False) -> None:
        super().__init__()
        self._init_head(config, emit_hidden)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor):
        return self._head(self.model(input_ids, position_ids))


def build_decode_state(
    config: Lfm3Config,
    max_seq_len: int,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Allocate the hybrid decode state tensors (all zero) for a fresh sequence.

    Hybrid layout — only the layers that need a given state get a slot:
      * k_cache / v_cache: [num_attention_layers, 1, n_kv_heads, max_seq_len, head_dim]
        (full and sliding layers alike; slot i = config.attention_layer_ids[i])
      * conv_state:        [num_conv_layers, 1, hidden_size, conv_L_cache-1]
        (slot i = config.conv_layer_ids[i])
    """
    na, nc = config.num_attention_layers, config.num_conv_layers
    return {
        "k_cache": torch.zeros(na, 1, config.num_key_value_heads, max_seq_len,
                               config.head_dim, dtype=dtype),
        "v_cache": torch.zeros(na, 1, config.num_key_value_heads, max_seq_len,
                               config.head_dim, dtype=dtype),
        "conv_state": torch.zeros(nc, 1, config.hidden_size,
                                  config.conv_state_width, dtype=dtype),
    }


def build_pipelined_export_spec(
    config: Lfm3Config,
    *,
    target_dtype: torch.dtype,
    max_context_length: int,
    trace_kv_len: int = TRACE_KV_CACHE_SEQ_LEN,
    trace_query: int = 1,
    trace_past: int = 64,
    embeds: bool = False,
    emit_hidden: bool = False,
    extra_inputs: dict[str, torch.Tensor] | None = None,
    static_query: bool | None = None,
) -> dict:
    """Pipelined-engine export spec for an LFM3 decoder graph.

    Inputs, in the order the model's forward must take them:
    ``input_ids`` int32 [1, q] (or ``inputs_embeds`` [1, q, hidden] with
    ``embeds``), ``position_ids`` int32 [1, past + q], then ``extra_inputs``
    (fixed-shape riders such as a VL wrapper's ``image_embeds``, in dict
    order), then the states ``k_cache`` / ``v_cache`` / ``conv_state``
    (surfaced as ``keyCache`` / ``valueCache`` / ``convState``).

    The query is static when ``trace_query == 1`` (the S=1 ship contract;
    pass ``static_query=True`` with a larger ``trace_query`` for a static
    chunk-prefill graph, and call ``model.model.set_explicit_mask()`` on the
    model before exporting it); positions and the KV seq dim stay dynamic.
    """
    b = 1
    if embeds:
        query = torch.randn(b, trace_query, config.hidden_size, dtype=target_dtype)
        query_name = "inputs_embeds"
    else:
        query = torch.randint(1, config.vocab_size, (b, trace_query), dtype=torch.int32)
        query_name = "input_ids"
    position_ids = torch.arange(trace_past + trace_query, dtype=torch.int32).unsqueeze(0)
    state = build_decode_state(config, max_seq_len=trace_kv_len, dtype=target_dtype)
    extra_inputs = dict(extra_inputs or {})
    reference_inputs = {query_name: query, "position_ids": position_ids, **extra_inputs,
                        "k_cache": state["k_cache"], "v_cache": state["v_cache"],
                        "conv_state": state["conv_state"]}
    if static_query is None:
        static_query = trace_query == 1
    pos_min = max(2, trace_query) if static_query else 2
    seq_pos = torch.export.Dim("seq_pos", min=pos_min, max=max_context_length - 1)
    k_seq = torch.export.Dim("k_seq", min=trace_kv_len, max=max_context_length)
    v_seq = torch.export.Dim("v_seq", min=trace_kv_len, max=max_context_length)
    query_shape = (
        None if static_query
        else {1: torch.export.Dim("seq_ids", min=1, max=max_context_length - 2)}
    )
    dynamic_shapes = {
        query_name: query_shape,
        "position_ids": {1: seq_pos},
        **{name: None for name in extra_inputs},
        "k_cache": {KVCache.seq_len_dim(): k_seq},
        "v_cache": {KVCache.seq_len_dim(): v_seq},
        "conv_state": None,  # fixed-shape extra state
    }
    return {
        "reference_inputs": reference_inputs,
        "dynamic_shapes": dynamic_shapes,
        "input_names": (query_name, "position_ids", *extra_inputs),
        "output_names": ("logits", "hidden") if emit_hidden else ("logits",),
        "state_names": DECODE_STATE_NAMES,
    }


class Lfm3ForCausalLMStateful(_Lfm3HeadMixin):
    """Stateful text decoder: one graph for prefill (S > 1) and decode (S = 1).

    forward inputs: input_ids [b, query_len], position_ids [b, seq_len], plus
    the three state tensors from ``build_decode_state``. The state tensors are
    mutated in place (surfaced as Core AI states via ``DECODE_STATE_NAMES``);
    logits for the query tokens are returned (plus ``hidden`` with
    ``emit_hidden``).
    """

    def __init__(self, config: Lfm3Config, emit_hidden: bool = False) -> None:
        super().__init__()
        self._init_head(config, emit_hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        conv_state: torch.Tensor,
    ):
        kv = KVCache(k_cache, v_cache)
        return self._head(self.model.forward_stateful(input_ids, position_ids, kv, conv_state))

    def build_macos_export_spec(
        self,
        target_dtype: torch.dtype,
        max_context_length: int,
        query_len: int,
        offset: int,
        trace_kv_len: int,
    ) -> dict:
        """Reference inputs + dynamic shapes for the hybrid export (the
        lfm2.py hook signature); see ``build_pipelined_export_spec``."""
        return build_pipelined_export_spec(
            self.config, target_dtype=target_dtype, max_context_length=max_context_length,
            trace_kv_len=trace_kv_len, trace_query=query_len, trace_past=offset,
            emit_hidden=self.emit_hidden, static_query=False,
        )


class Lfm3EmbedsForCausalLMStateful(_Lfm3HeadMixin):
    """Embeds-input stateful decoder: the query arrives already embedded
    (``inputs_embeds`` [b, query_len, hidden]) — for callers that assemble
    mixed text / image / audio embeddings on the host (LFM3-VL, -Omni,
    -Audio). Same states, positions and fused conv write-back as the
    token-input graph; ``model.embed_tokens`` holds the text table for the
    host-side lookup."""

    def __init__(self, config: Lfm3Config, emit_hidden: bool = False) -> None:
        super().__init__()
        self._init_head(config, emit_hidden)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        conv_state: torch.Tensor,
    ):
        kv = KVCache(k_cache, v_cache)
        return self._head(
            self.model.forward_stateful_embeds(inputs_embeds, position_ids, kv, conv_state)
        )


# --------------------------------------------------------------------------- #
# Weight loading (safetensors-direct; no transformers LFM3 support needed)
# --------------------------------------------------------------------------- #
# On the fp16 export these stay fp32 (IMPORTANT #2).
_ATTN_FP32_RE = re.compile(
    r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj|gate_proj|q_norm|k_norm)\.weight$"
)


def resolve_lfm3_source(src: str, revision: str | None = None) -> str:
    """A local checkpoint dir as is; an HF id through snapshot_download."""
    if os.path.isdir(src):
        return src
    from huggingface_hub import snapshot_download

    return snapshot_download(
        src,
        revision=revision,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
    )


def resolve_lfm3_config(
    config: Lfm3Config | dict | str | None,
    src_dir: str,
    config_key: str | None = None,
) -> Lfm3Config:
    """``config``: an Lfm3Config, a raw dict, a path to a JSON file, or None
    (``<src_dir>/config.json``). ``config_key`` picks a sub-dict first
    (``text_config`` for VL / Omni, ``lfm`` for Audio)."""
    if isinstance(config, Lfm3Config):
        return config
    if config is None:
        config = os.path.join(src_dir, "config.json")
    if isinstance(config, str):
        with open(config) as f:
            config = json.load(f)
    if config_key:
        config = config[config_key]
    return lfm3_config_from_dict(config)


def lfm3_state_dict(
    src_dir: str,
    key_prefix: str = "model.",
    target_dtype: torch.dtype = torch.float16,
    fp32_attn_proj: bool = True,
) -> dict[str, torch.Tensor]:
    """The decoder tensors under ``key_prefix`` as ``model.<...>`` keys of
    this module (MLP w1/w3/w2 -> gate/up/down; every other name matches HF).
    Prefixes: 1.7B ``model.``; VL / Omni ``model.language_model.``; Audio
    ``lfm.model.``. Other towers in the same file are not read."""
    from safetensors import safe_open

    files = sorted(glob.glob(os.path.join(src_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors files in {src_dir}")
    sd: dict[str, torch.Tensor] = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                if not key.startswith(key_prefix):
                    continue
                local = "model." + key[len(key_prefix):]
                for old, new in _MLP_KEY_MAP.items():
                    if old in local:
                        local = local.replace(old, new)
                        break
                tensor = f.get_tensor(key)
                dtype = target_dtype
                if fp32_attn_proj and _ATTN_FP32_RE.search(local):
                    dtype = torch.float32
                if tensor.dtype != dtype:
                    tensor = tensor.to(dtype)
                sd[local] = tensor
    if not sd:
        raise KeyError(f"no tensor under {key_prefix!r} in {src_dir}")
    return sd


def lfm3_from_hf(
    src: str,
    *,
    key_prefix: str = "model.",
    config: Lfm3Config | dict | str | None = None,
    config_key: str | None = None,
    target_dtype: torch.dtype = torch.float16,
    stateful: bool = True,
    fp32_attn_proj: bool = True,
    emit_hidden: bool = False,
    embeds: bool = False,
    model_cls: type[nn.Module] | None = None,
    revision: str | None = None,
):
    """Load an LFM3 decoder from an HF checkpoint into the authored module.

    ``src``: local checkpoint dir or HF id (``revision`` pins the snapshot).
    ``key_prefix`` / ``config`` (+ ``config_key``) select the decoder inside a
    multimodal checkpoint. Class: ``Lfm3EmbedsForCausalLMStateful`` with
    ``embeds``, else ``Lfm3ForCausalLMStateful`` / ``Lfm3ForCausalLM`` by
    ``stateful``, or ``model_cls`` (a wrapper constructed as
    ``model_cls(config)`` with ``.model`` = Lfm3Model and ``.lm_head``).
    With ``fp32_attn_proj`` (default) the attention q/k/v/o/gate projections
    and q/k norms stay fp32 on an fp16 load (IMPORTANT #2).
    """
    src_dir = resolve_lfm3_source(src, revision)
    cfg = resolve_lfm3_config(config, src_dir, config_key)

    if model_cls is None:
        if embeds:
            model_cls = Lfm3EmbedsForCausalLMStateful
        else:
            model_cls = Lfm3ForCausalLMStateful if stateful else Lfm3ForCausalLM
    with torch.device("meta"):
        model = model_cls(cfg)
    if hasattr(model, "emit_hidden"):
        model.emit_hidden = emit_hidden
    model.to(dtype=target_dtype)

    sd = lfm3_state_dict(src_dir, key_prefix, target_dtype, fp32_attn_proj)
    missing, unexpected = model.load_state_dict(sd, assign=True, strict=False)
    missing = [k for k in missing if not (k == "lm_head.weight" and cfg.tie_word_embeddings)]
    if missing or unexpected:
        raise RuntimeError(f"load mismatch: missing={missing} unexpected={unexpected}")
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight

    meta_params = [n for n, p in model.named_parameters() if p.is_meta]
    if meta_params:
        raise RuntimeError(f"Parameters not loaded: {meta_params}")
    model.eval()
    return model
