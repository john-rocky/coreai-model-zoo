# LFM3 text decoder (short-conv + full-attention + sliding-attention hybrid) on Apple's
# static iOS contract: load_embeddings / gather_embeddings / extend / prompt_opt graphs,
# static (cache_len, query_len) shapes, and key_cache / value_cache as the ONLY two
# runner-bound states (Apple's StaticShapeEngine binds nothing else).
#
# Community port — NOT an Apple model. The layer math is the macOS authoring module's
# (models/macos/lfm3.py, line 1 PASS vs HF fp32: config parsing, the zero-tail RoPE
# frequencies and the weight loading are imported from it); the layout and the
# primitives are Apple's iOS ones (models/ios/qwen3.py: Conv2d 1x1 projections,
# [b, q, 1, d] residual stream, cached RoPE tables gathered by position_ids, the per-head
# SDPA with the runner's additive causal_mask, the int8 per-tensor embedding table that
# is also the tied head). The conv mixer follows the zoo's LFM2.5 static builder
# (conversion/overlay/.../models/ios/lfm2.py, 31/31 ANE regions on h18p), restated here
# on this checkout's primitives: that file imports TraceSpec / coreai_models._constants /
# mutable_cache_update_and_fetch, which exist only on the zoo-0.4 line.
#
# How the conv mixer fits the two-state contract
# ----------------------------------------------
# A conv layer's only cross-step state is its previous conv_L_cache - 1 (= 6) gated-input
# columns Bx (hidden 2048 each). Every conv layer writes the Bx column of each batch
# position into ITS OWN row of key_cache / value_cache at position in_step.. (exactly
# where an attention layer writes k / v): channels [0:1024) into the key row, [1024:2048)
# into the value row. Rows are therefore CACHE_WIDTH = hidden/2 = 1024 channels wide for
# every layer; attention layers (4 kv heads x 128 = 512) use channels [0:512).
# Memory: 24 rows x 1024 x 2 x ctx x 2 B = 402 MB at ctx 4096.
#
# The history is read back by position, never "last written": the engine re-runs the
# aligned batch (batchStart = position // q * q) on every decode step and switches
# query lengths (prompt_opt q=64 -> extend q=64 -> decode q=8). Apple's ladder is
# q in {8, 16, 64}, so in_step (= batchStart) is always a multiple of 8 and the six
# history columns in_step-6 .. in_step-1 lie inside the 8-position block that ends at
# in_step. The read (history="block", default) splits the row's position axis into
# blocks of 8 and picks that block with a one-hot over blocks — one multiply + reduce
# per row. history="column" reads each of the six columns with its own one-hot over
# positions (the LFM2.5 builder's form: six multiply + reduce passes over the row).
# Both selectors are made from column 0 of the runner's causal_mask (0 where
# p <= in_step, the sentinel -40000 elsewhere) with fp16 add / clamp / slice / sub
# only, so in_step never becomes tensor data on the Neural Engine; neither read is a
# [W, L] @ [L, n] matmul (the h18p ANE compiler rejects that, LFM2.5 S6 bisect).
#
# Sliding window (5 layers, 256): the key p is too old for the query at position t when
# p <= t - 256, i.e. when the causal_mask column admits p + 256. The sliding layers get
# causal_mask + sentinel * shift_256(clamp(causal_mask + 1, 0, 1)), made once per call
# with the same fp16 ops (pad, then slice: no empty tensor at ctx 256).
#
# The output gate (per head, sigmoid(gate_proj(x)) scaling the head before o_proj) and
# the MLP's SiLU are written as 0.5 * tanh(x / 2) + 0.5 (act="tanh", default; exact):
# the Neural Engine's sigmoid / SiLU kernels lost 3.4 % on Timestral where tanh lost
# 0.19 % (lane memory reference_ane_sigmoid_silu_tanh). act="stock" = sigmoid / SiLU.
#
# Norms: RMSNorm computed on K * x with eps' = K^2 * eps (RMSNormK from the LFM2.5
# builder; exact for a power-of-2 K): LFM3's eps 1e-5 is a subnormal fp16 constant, and
# the h18p ANE flushed LFM2.5's. It matters here: the mixer output entering
# post_operator_norm has a mean square down to 1.35e-7 (r5 tiny probe on the real
# first 5 layers), where eps sets the scale. K is per norm kind (NORM_K); K * absmax of
# each kind's fp32 input stays well under 256 so the squares stay finite in fp16.
#
# The depthwise 7-tap conv is seven shifted multiply-adds, not F.conv1d (the grouped
# conv1d failed the MPSGraph -> ANEC conversion in the Qwen3.5 iOS port).
#
# emit_hidden (default off): the extend graph also returns the embedding_norm output (the
# head's input) as a second output `hidden` [1, q, 1, hidden] (the residual stream's layout,
# = transformer_input's), fp16 in the fp16 graph — the tensor the macOS module's emit_hidden
# graph returns as [1, S, hidden], for the speech paths of LFM3-Omni / -Audio (their depth
# decoders read it). prompt_opt (prefill mode) is unchanged, and with it off the extend
# graph is the one of the version before (r5).

from __future__ import annotations

import glob
import os

import torch
import torch.nn as nn

from coreai_models.models.base import BaseForCausalLMForiOS
from coreai_models.models.macos.lfm3 import (
    ATTENTION_TYPES,
    FULL,
    SLIDING,
    Lfm3Config,
    resolve_lfm3_config,
    resolve_lfm3_source,
    rope_inv_freq,
)
from coreai_models.primitives._ops import mutable_slice_update
from coreai_models.primitives.ios.quantization import (
    dequantize_per_tensor,
    quantize_per_tensor,
)
from coreai_models.primitives.ios.rope import apply_rope, rope_gather_cached_cos_sin
from coreai_models.primitives.ios.sdpa import SDPA

# The runner's causal_mask sentinel (CoreAILanguageModels InputHandler.swift
# `causalMaskSentinel = Float16(-40000)`).
MASK_SENTINEL = -40000.0
# The extend graph's second output with emit_hidden (the macOS module's name for the same tensor).
HIDDEN_OUTPUT_NAME = "hidden"
# in_step granularity: Apple's static ladder is q in {8, 16, 64}.
HISTORY_BLOCK = 8
# Diagnostic knobs for compile bisects (comma list in LFM3_IOS_DIAG; production = unset):
#   nowindow  the sliding layers get the plain causal mask (no window)
#   onerope   the sliding layers rotate with the full layers' cos / sin table
#   hidden3d  emit_hidden's output reshaped to [b, q, d]: reproduces the ANE compiler crash that made it 4-D (below)
_DIAG = {d for d in os.environ.get("LFM3_IOS_DIAG", "").split(",") if d}
# per norm kind; a float applies one K to every norm. 4 everywhere: K^2 * 1e-5 = 1.6e-4 is a normal fp16
# number (the subnormal eps constant is what the ANE flushed for LFM2.5; its mul keeps subnormal products,
# memory reference_ane_div), and 4 x the largest input (30.2, the MLP output, r1's absmax) squares to 1.5e4.
NORM_K = {"operator_norm": 4.0, "ffn_norm": 4.0, "post_operator_norm": 4.0, "post_ffn_norm": 4.0,
          "embedding_norm": 4.0, "q_norm": 4.0, "k_norm": 4.0}
ACTS = ("tanh", "stock")
HISTORIES = ("block", "column", "none")


def _norm_k(norm_k, kind: str) -> float:
    return float(norm_k[kind] if isinstance(norm_k, dict) else norm_k)


def _cpu_buffer(value: float) -> nn.Buffer:
    with torch.device("cpu"):
        return nn.Buffer(torch.tensor(value), persistent=False)


def _sigmoid(x: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
    """sigmoid(x) = 0.5 * tanh(x / 2) + 0.5 (exact); ``half`` is a 0.5 buffer in the
    module's dtype, so no fp32 constant enters an fp16 graph."""
    return half * torch.tanh(half * x) + half


# --------------------------------------------------------------------------- #
# Norm, MLP
# --------------------------------------------------------------------------- #
class RMSNormK(nn.Module):
    """x / sqrt(mean(x^2) + eps) * w, computed on y = K * x with eps' = K^2 * eps
    (the LFM2.5 builder's norm; mathematically identical, bit-identical in IEEE fp16
    for a power-of-2 K). Normalizes over the last dim, like Apple's iOS RMSNorm."""

    def __init__(self, dim: int, eps: float, k: float = 4.0) -> None:
        super().__init__()
        with torch.device("cpu"):
            self.weight = nn.Parameter(torch.zeros(dim))
        self._eps = _cpu_buffer(eps * k * k)
        self._k = _cpu_buffer(float(k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * self._k
        mean_square = (y * y).mean(-1, keepdim=True)
        return y * torch.rsqrt(mean_square + self._eps) * self.weight


class MLP(nn.Module):
    """Apple's iOS MLP (Conv2d 1x1 gate / up / down over the tokens folded into the
    batch dim) with the SiLU written through tanh (act="tanh")."""

    def __init__(self, dim: int, hidden_dim: int, act: str = "tanh") -> None:
        super().__init__()
        self.gate_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.up_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.down_proj = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)
        self.act = act
        self._half = _cpu_buffer(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, query_len, _, dim = x.shape
        x = x.reshape(batch_size * query_len, dim, 1, 1)
        gate = self.gate_proj(x)
        if self.act == "tanh":
            gate = gate * _sigmoid(gate, self._half)
        else:
            gate = nn.functional.silu(gate)
        out = self.down_proj(self.up_proj(x) * gate)
        return out.reshape(batch_size, query_len, 1, dim)


# --------------------------------------------------------------------------- #
# RoPE: the attention types' cos / sin tables side by side, one gather
# --------------------------------------------------------------------------- #
class Lfm3RoPETables(nn.Module):
    """Cached cos / sin for every attention type in ONE table [max_pos, n_types * head_dim] (type order =
    ``types``), gathered once by position_ids with Apple's `coreai::rope_gather_cached_cos_sin` and split per type.

    One table, not one RoPECache per type: two gathers by the same position_ids from two cached tables crash
    coreai-build (h16c, gpu and neural-engine alike: SIGSEGV in MPSGraph's
    CanonicalizeCopyWithConstraints, r5 tiny bisect) — WeVisOne's in-graph M-RoPE (three gathers) hit the same.
    The values are RoPECache's: angle = int32 position x fp32 inv_freq (HF's product), cos / sin in fp32, then
    the module dtype; recomputed on every dtype change. full = theta 1e6 with the zero tail (inv_freq 0 for the
    last 32 of 64 frequencies -> cos 1 / sin 0, exact); sliding = theta 1e4 over all 64."""

    def __init__(self, config: Lfm3Config, max_positions: int, types: list[str]) -> None:
        super().__init__()
        self.types = list(types)
        self.head_dim = config.head_dim
        self.max_positions = max_positions
        self._inv_freq = {t: rope_inv_freq(config, t) for t in self.types}  # fp32, CPU (plain attributes)
        self._compute(torch.float32)

    def _compute(self, dtype: torch.dtype) -> None:
        with torch.device("cpu"):
            pos = torch.arange(self.max_positions, dtype=torch.int32)
            cos, sin = [], []
            for t in self.types:
                freqs = pos[:, None] * self._inv_freq[t]
                emb = torch.cat((freqs, freqs), dim=-1)
                cos.append(torch.cos(emb))
                sin.append(torch.sin(emb))
            self.cos_cached = nn.Buffer(torch.cat(cos, dim=-1).to(dtype), persistent=False)
            self.sin_cached = nn.Buffer(torch.cat(sin, dim=-1).to(dtype), persistent=False)

    def _apply(self, fn):
        super()._apply(fn)
        transformed = fn(torch.tensor(0.0))
        self._compute(transformed.dtype)
        self.cos_cached = self.cos_cached.to(device=transformed.device)
        self.sin_cached = self.sin_cached.to(device=transformed.device)
        return self

    def gather(self, position_ids: torch.Tensor) -> dict:
        """position_ids [1, q] -> {type: (cos, sin)}, each [1, q, head_dim]."""
        cos, sin = rope_gather_cached_cos_sin(position_ids, self.cos_cached, self.sin_cached)
        d = self.head_dim
        return {t: (cos[..., i * d:(i + 1) * d], sin[..., i * d:(i + 1) * d]) for i, t in enumerate(self.types)}


# --------------------------------------------------------------------------- #
# The two runner-bound states as one row per layer
# --------------------------------------------------------------------------- #
class _KVRows:
    """key_cache / value_cache [n_layers, 1, width, 1, ctx]: row i belongs to layer i.

    ``write`` puts k into the key row and v into the value row at positions
    [offset, offset + n), each over its own channel count from channel 0, and returns
    both full rows [1, width, 1, ctx] as they are after the write (Apple's
    KVCacheHandler.update_and_fetch with a per-call channel count)."""

    def __init__(self, n_layers: int, widths: tuple[int, ...]) -> None:
        self._k_cache: torch.Tensor | None = None
        self._v_cache: torch.Tensor | None = None
        with torch.device("cpu"):
            self._zero = torch.zeros(1, dtype=torch.int32)
            self._one = torch.ones(1, dtype=torch.int32)
            self._rows = torch.arange(n_layers, dtype=torch.int32).unsqueeze(1)
            self._rows_end = torch.arange(1, n_layers + 1, dtype=torch.int32).unsqueeze(1)
            self._widths = {w: torch.tensor([w], dtype=torch.int32) for w in sorted(set(widths))}

    def register(self, key_cache: torch.Tensor, value_cache: torch.Tensor) -> None:
        assert key_cache.shape == value_cache.shape, (key_cache.shape, value_cache.shape)
        self._k_cache = key_cache
        self._v_cache = value_cache

    def _slice_args(self, row: int, offset: torch.Tensor, width: int, n: int):
        begin = torch.cat([self._rows[row], self._zero, self._zero, self._zero, offset])
        end = torch.cat([self._rows_end[row], self._one, self._widths[width], self._one, offset + n])
        return begin, end

    def write(self, row: int, offset: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n: int):
        assert self._k_cache is not None and self._v_cache is not None
        torch._check_is_size(row)
        torch._check(row < self._k_cache.size(0))
        bk, ek = self._slice_args(row, offset, k.shape[1], n)
        mutable_slice_update(x=self._k_cache, update=k.unsqueeze(0), begin=bk, end=ek)
        bv, ev = self._slice_args(row, offset, v.shape[1], n)
        mutable_slice_update(x=self._v_cache, update=v.unsqueeze(0), begin=bv, end=ev)
        return self._k_cache[row], self._v_cache[row]

    @property
    def k_cache(self) -> torch.Tensor:
        return self._k_cache

    @property
    def v_cache(self) -> torch.Tensor:
        return self._v_cache


# --------------------------------------------------------------------------- #
# Selectors made from the runner's causal_mask (fp16 add / clamp / slice / sub)
# --------------------------------------------------------------------------- #
def _admitted(causal_mask: torch.Tensor) -> torch.Tensor:
    """1 where the mask admits the key, 0 where it holds the sentinel (exact in fp16)."""
    return torch.clamp(causal_mask + 1.0, 0.0, 1.0)


def sliding_window_mask(causal_mask: torch.Tensor, window: int, sentinel: torch.Tensor) -> torch.Tensor:
    """causal_mask [1, L, 1, q] -> the same mask that also holds back keys p <= t - window.

    a[p, i] = 1 iff p <= t_i (query i at position t_i); a[p + window, i] = 1 iff
    p <= t_i - window, which is where the window ends. Padding queries (all-sentinel
    columns) stay all-sentinel. L >= window for every graph (Apple's ladder starts at 256)."""
    a = _admitted(causal_mask)
    cache_len = a.shape[1]
    torch._check(cache_len >= window)
    pad = torch.zeros_like(a[:, :window])
    shifted = torch.cat([a, pad], dim=1)[:, window : window + cache_len]
    return causal_mask + shifted * sentinel


def history_block_selector(causal_mask: torch.Tensor) -> torch.Tensor:
    """One-hot over the 8-position blocks of the cache: block in_step/8 - 1 (the block
    that ends at in_step; none when in_step = 0). Returns [1, 1, L/8, 1].

    Column 0 of the mask admits p <= in_step (every call has >= 1 real token), so
    A[b] = a[8b] is 1 for b <= in_step/8, and A[b+1] - A[b+2] is 1 at b = in_step/8 - 1."""
    a = _admitted(causal_mask[:, :, :, 0:1])  # [1, L, 1, 1]
    cache_len = a.shape[1]
    n_blocks = cache_len // HISTORY_BLOCK
    starts = a.reshape(1, n_blocks, HISTORY_BLOCK)[:, :, 0]  # [1, L/8]
    starts = torch.cat([starts, torch.zeros_like(starts[:, 0:2])], dim=1)
    sel = starts[:, 1 : 1 + n_blocks] - starts[:, 2 : 2 + n_blocks]
    return sel.reshape(1, 1, n_blocks, 1)


def history_column_selector(causal_mask: torch.Tensor, n_prev: int) -> torch.Tensor:
    """One-hot selectors for positions in_step - n_prev .. in_step - 1 (oldest first),
    the LFM2.5 builder's form: onehot(in_step - j)[p] = a[p + j] - a[p + j + 1].
    Returns [1, 1, L, n_prev]."""
    a = _admitted(causal_mask[:, :, :, 0:1])  # [1, L, 1, 1]
    cache_len = a.shape[1]
    a = torch.cat([a, torch.zeros_like(a[:, 0 : n_prev + 1])], dim=1)
    cols = [a[:, j : j + cache_len] - a[:, j + 1 : j + 1 + cache_len] for j in range(n_prev, 0, -1)]
    return torch.cat(cols, dim=-1).reshape(1, 1, cache_len, n_prev)


# --------------------------------------------------------------------------- #
# Attention (GQA, per-head q/k RMSNorm, per-head output gate, full or sliding)
# --------------------------------------------------------------------------- #
class Lfm3AttentionIOS(nn.Module):
    def __init__(self, config: Lfm3Config, layer_idx: int, act: str, norm_k) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        d = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.kv_width = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Conv2d(d, self.n_heads * self.head_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(d, self.kv_width, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(d, self.kv_width, kernel_size=1, bias=False)
        self.o_proj = nn.Conv2d(self.n_heads * self.head_dim, d, kernel_size=1, bias=False)
        self.gate_proj = nn.Conv2d(d, self.n_heads, kernel_size=1, bias=False)
        self.q_norm = RMSNormK(self.head_dim, config.norm_eps, _norm_k(norm_k, "q_norm"))
        self.k_norm = RMSNormK(self.head_dim, config.norm_eps, _norm_k(norm_k, "k_norm"))
        self.sdpa = SDPA(head_dim=self.head_dim)
        self.act = act
        self._half = _cpu_buffer(0.5)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.Tensor,
        mask: torch.Tensor,
        kv: _KVRows,
    ) -> torch.Tensor:
        batch_size, query_len, _, _ = x.shape
        H, HKV, D = self.n_heads, self.n_kv_heads, self.head_dim

        x = x.transpose(-3, -1)  # [b, d, 1, q]
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        gate = self.gate_proj(x)  # [b, H, 1, q]

        query = query.transpose(-3, -1).reshape(batch_size, query_len, H, D).transpose(-2, -3)
        key = key.transpose(-3, -1).reshape(batch_size, query_len, HKV, D).transpose(-2, -3)
        query = self.q_norm(query)
        key = self.k_norm(key)

        torch._check_is_size(query_len)
        torch._check_is_size(rope_cos.shape[1])
        query = apply_rope(query, rope_cos, rope_sin)
        key = apply_rope(key, rope_cos, rope_sin)

        query = query.transpose(-2, -3).reshape(batch_size, query_len, 1, H * D).transpose(-3, -1)
        key = key.transpose(-3, -2).reshape(batch_size, query_len, 1, HKV * D).transpose(-3, -1)

        k_row, v_row = kv.write(self.layer_idx, in_step, key, value, query_len)
        key = k_row[:, : self.kv_width]
        value = v_row[:, : self.kv_width]

        out = self.sdpa(query, key, value, mask)  # [b, H*D, 1, q]
        if self.act == "tanh":
            gate = _sigmoid(gate, self._half)
        else:
            gate = torch.sigmoid(gate)
        out = (out.reshape(batch_size, H, D, query_len) * gate.reshape(batch_size, H, 1, query_len)).reshape(
            batch_size, H * D, 1, query_len
        )
        return self.o_proj(out).transpose(-3, -1)


# --------------------------------------------------------------------------- #
# Short-conv mixer with its history in the layer's own KV rows
# --------------------------------------------------------------------------- #
class _ConvTaps(nn.Module):
    """The depthwise taps as a bare parameter under the HF name (`conv.conv.weight`,
    [d, 1, k]): the palettizer leaves bare parameters alone and the graph never emits
    a grouped conv."""

    def __init__(self, dim: int, kernel: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, 1, kernel))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None


class Lfm3ShortConvIOS(nn.Module):
    def __init__(self, config: Lfm3Config, layer_idx: int, cache_width: int, history: str) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        d = config.hidden_size
        self.dim = d
        self.kernel = int(config.conv_L_cache)
        self.n_prev = self.kernel - 1
        assert self.n_prev <= HISTORY_BLOCK, (self.n_prev, HISTORY_BLOCK)
        self.split = cache_width  # Bx channels [0:split) -> key row, [split:d) -> value row
        assert d == 2 * cache_width, (d, cache_width)
        self.history = history
        bias = bool(config.conv_bias)
        self.in_proj = nn.Conv2d(d, 3 * d, kernel_size=1, bias=bias)
        self.out_proj = nn.Conv2d(d, d, kernel_size=1, bias=bias)
        self.conv = _ConvTaps(d, self.kernel, bias)

    def _read_history(self, k_row: torch.Tensor, v_row: torch.Tensor, selector: torch.Tensor) -> torch.Tensor:
        """The n_prev Bx columns before in_step, [1, d, 1, n_prev] (oldest first)."""
        width = k_row.shape[1]
        cache_len = k_row.shape[-1]
        if self.history == "block":
            n_blocks = cache_len // HISTORY_BLOCK
            hk = (k_row.reshape(1, width, n_blocks, HISTORY_BLOCK) * selector).sum(2)  # [1, w, 8]
            hv = (v_row.reshape(1, width, n_blocks, HISTORY_BLOCK) * selector).sum(2)
            prev = torch.cat([hk, hv], dim=1)[:, :, HISTORY_BLOCK - self.n_prev :]
            return prev.reshape(1, self.dim, 1, self.n_prev)
        cols = []
        for j in range(self.n_prev):
            col = selector[:, :, :, j].reshape(1, 1, 1, cache_len)
            cols.append(torch.cat([(k_row * col).sum(-1, keepdim=True), (v_row * col).sum(-1, keepdim=True)], dim=1))
        return torch.cat(cols, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        in_step: torch.Tensor,
        selector: torch.Tensor | None,
        kv: _KVRows,
    ) -> torch.Tensor:
        batch_size, query_len, _, _ = x.shape
        torch._check_is_size(query_len)
        d = self.dim

        x = x.transpose(-3, -1)  # [b, d, 1, q]
        gate_b, gate_c, xv = torch.split(self.in_proj(x), d, dim=1)
        bx = gate_b * xv  # [b, d, 1, q]

        k_row, v_row = kv.write(self.layer_idx, in_step, bx[:, : self.split], bx[:, self.split :], query_len)
        if self.history == "none":  # negative control: the history is written, never read
            prev = torch.zeros_like(bx[..., 0:1]).expand(batch_size, d, 1, self.n_prev)
        else:
            prev = self._read_history(k_row, v_row, selector)

        window = torch.cat([prev, bx], dim=-1)  # [b, d, 1, n_prev + q]
        taps = self.conv.weight  # out[t] = sum_j w[j] * Bx[t - (k-1) + j]
        conv_out = taps[:, 0, 0].reshape(1, d, 1, 1) * window[..., 0:query_len]
        for j in range(1, self.kernel):
            conv_out = conv_out + taps[:, 0, j].reshape(1, d, 1, 1) * window[..., j : j + query_len]
        if self.conv.bias is not None:
            conv_out = conv_out + self.conv.bias.reshape(1, d, 1, 1)
        return self.out_proj(gate_c * conv_out).transpose(-3, -1)


# --------------------------------------------------------------------------- #
# Decoder layer (sandwich norm), stack, extend graph
# --------------------------------------------------------------------------- #
class Lfm3DecoderLayerIOS(nn.Module):
    def __init__(self, config: Lfm3Config, layer_idx: int, cache_width: int, act: str, history: str,
                 norm_k) -> None:
        super().__init__()
        d = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        self.is_attention = self.layer_type in ATTENTION_TYPES
        self.use_sandwich_norm = config.use_sandwich_norm
        if self.is_attention:
            self.self_attn = Lfm3AttentionIOS(config, layer_idx, act, norm_k)
        else:
            self.conv = Lfm3ShortConvIOS(config, layer_idx, cache_width, history)
        self.feed_forward = MLP(d, config.intermediate_size, act)
        self.operator_norm = RMSNormK(d, config.norm_eps, _norm_k(norm_k, "operator_norm"))
        self.ffn_norm = RMSNormK(d, config.norm_eps, _norm_k(norm_k, "ffn_norm"))
        if self.use_sandwich_norm:
            self.post_operator_norm = RMSNormK(d, config.norm_eps, _norm_k(norm_k, "post_operator_norm"))
            self.post_ffn_norm = RMSNormK(d, config.norm_eps, _norm_k(norm_k, "post_ffn_norm"))

    def forward(self, x, rope, in_step, masks, selector, kv):
        normed = self.operator_norm(x)
        if self.is_attention:
            cos, sin = rope[self.layer_type]
            r = self.self_attn(normed, cos, sin, in_step, masks[self.layer_type], kv)
        else:
            r = self.conv(normed, in_step, selector, kv)
        if self.use_sandwich_norm:
            r = self.post_operator_norm(r)
        h = x + r
        f = self.feed_forward(self.ffn_norm(h))
        if self.use_sandwich_norm:
            f = self.post_ffn_norm(f)
        return h + f


class Lfm3ModelIOS(nn.Module):
    def __init__(self, config: Lfm3Config, cache_width: int, act: str, history: str, norm_k) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [Lfm3DecoderLayerIOS(config, i, cache_width, act, history, norm_k)
             for i in range(config.num_hidden_layers)]
        )
        # HF names the final norm embedding_norm
        self.embedding_norm = RMSNormK(config.hidden_size, config.norm_eps, _norm_k(norm_k, "embedding_norm"))


class Lfm3Extend(nn.Module):
    def __init__(self, config: Lfm3Config, max_context_length: int, act: str = "tanh",
                 history: str = "block", norm_k=NORM_K, emit_hidden: bool = False) -> None:
        super().__init__()
        if act not in ACTS or history not in HISTORIES:
            raise ValueError(f"act {act!r} / history {history!r}")
        self.config = config
        self.cache_width = config.hidden_size // 2
        self.model = Lfm3ModelIOS(config, self.cache_width, act, history, norm_k)
        self.history = history
        self.emb_zero_point = nn.Parameter(torch.zeros([], dtype=torch.int8), requires_grad=False)
        self.emb_scale = nn.Parameter(torch.ones([], dtype=torch.float16), requires_grad=False)
        self.prefill_mode = False
        # a plain attribute like prefill_mode: read when the graph is traced, settable after loading
        self.emit_hidden = emit_hidden
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = None
        kv_width = config.num_key_value_heads * config.head_dim
        self.kv_cache = _KVRows(config.num_hidden_layers, (kv_width, self.cache_width))
        self.layer_types = sorted(set(config.layer_types) & set(ATTENTION_TYPES))
        # Sized to the export's context, not the checkpoint's 262,144 positions.
        self.rope = Lfm3RoPETables(config, max_context_length, self.layer_types)
        self._sentinel = _cpu_buffer(MASK_SENTINEL)

    def forward(
        self,
        transformer_input: torch.Tensor,
        position_ids: torch.Tensor,
        in_step: torch.Tensor,
        causal_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        embedding_table: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """logits [b, 1, q, V]; with emit_hidden (logits, hidden [b, q, 1, hidden_size]); in prefill mode the
        cache-keeping scalar of Apple's prompt_opt."""
        self.kv_cache.register(key_cache, value_cache)
        rope = self.rope.gather(position_ids)
        if "onerope" in _DIAG and FULL in rope:
            rope[SLIDING] = rope[FULL]
        masks = {FULL: causal_mask}
        if SLIDING in self.layer_types:
            masks[SLIDING] = causal_mask if "nowindow" in _DIAG else sliding_window_mask(
                causal_mask, self.config.sliding_window, self._sentinel.to(causal_mask.dtype))
        if self.history == "block":
            selector = history_block_selector(causal_mask)
        elif self.history == "column":
            selector = history_column_selector(causal_mask, self.config.conv_L_cache - 1)
        else:
            selector = None

        batch_size, seq_len, _, hidden_dim = transformer_input.shape
        h = transformer_input
        for layer in self.model.layers:
            h = layer(h, rope, in_step, masks, selector, self.kv_cache)
        out = self.model.embedding_norm(h)
        if self.prefill_mode:
            return self.kv_cache.k_cache[0, 0, 0, 0, 0] + self.kv_cache.v_cache[0, 0, 0, 0, 0]
        # the [b, q, 1, d] stream as is (transformer_input's layout), taken before the tied head reshapes `out`;
        # off = no op added. Not [b, q, d]: the head matmul's input reshaped into an output crashes the ANE compiler
        # (h16c and h18p, SIGSEGV in ZinIrScheduleGraph::CompressMatmulKernelRewind; r7 tiny bisect, knob hidden3d).
        hidden = out if self.emit_hidden else None
        if self.emit_hidden and "hidden3d" in _DIAG:
            hidden = out.reshape(batch_size, seq_len, hidden_dim)

        if self.lm_head is not None:
            logits = self.lm_head(out.transpose(-2, -3))
        else:
            if embedding_table.dtype == torch.int8:
                embedding_table = dequantize_per_tensor(embedding_table, self.emb_scale, self.emb_zero_point,
                                                        out.dtype)
            embedding_table = embedding_table.reshape(
                embedding_table.shape[1], embedding_table.shape[0], embedding_table.shape[2]
            )
            out = out.transpose(-3, -1).reshape(batch_size, 1, hidden_dim, seq_len)
            logits = (embedding_table @ out).transpose(-2, -1)
        return (logits, hidden) if self.emit_hidden else logits


# --------------------------------------------------------------------------- #
# iOS model (Apple's 4-entrypoint contract)
# --------------------------------------------------------------------------- #
class Lfm3ForCausalLMForiOS(BaseForCausalLMForiOS):
    _HF_MODEL_CLASS = None  # loaded from the safetensors directly (lfm3_ios_from_hf)

    def __init__(self, config: Lfm3Config, model_device: str = "meta", disable_embedding_quantization: bool = False,
                 *, max_context_length: int = 4096, act: str = "tanh", history: str = "block",
                 norm_k=NORM_K, emit_hidden: bool = False) -> None:
        # read by _init_model, which the base class calls inside its __init__
        object.__setattr__(self, "_ios_opts", dict(max_context_length=max_context_length, act=act,
                                                   history=history, norm_k=norm_k, emit_hidden=emit_hidden))
        super().__init__(config, model_device, disable_embedding_quantization)

    def _init_model(self, config: Lfm3Config) -> None:
        self.extend = Lfm3Extend(config, **self._ios_opts)

    @property
    def cache_width(self) -> int:
        return self.extend.cache_width

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        in_step: torch.Tensor,
        causal_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Eager composition (palettization's trace; never exported as is); returns what extend returns."""
        token_embeddings = self.gather_embeddings(input_ids, self.load_embeddings.embedding_table)
        return self.extend(token_embeddings, position_ids, in_step, causal_mask, key_cache, value_cache,
                           self.load_embeddings.embedding_table)

    def _mutate_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """HF names -> this module's: projections to Conv2d [out, in, 1, 1], MLP w1 / w3 / w2
        -> gate / up / down, the embedding into the iOS table (int8 per tensor unless
        disabled), everything under `extend.`."""
        n_layers = 1 + max(int(k.split(".")[2]) for k in state_dict if k.startswith("model.layers."))
        mlp_map = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
        for i in range(n_layers):
            prefix = f"model.layers.{i}."
            for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                         "self_attn.gate_proj", "conv.in_proj", "conv.out_proj"):
                key = f"{prefix}{name}.weight"
                if key in state_dict:
                    state_dict[key] = state_dict[key][:, :, None, None]
            for src, dst in mlp_map.items():
                key = f"{prefix}feed_forward.{src}.weight"
                if key in state_dict:
                    state_dict[f"{prefix}feed_forward.{dst}.weight"] = state_dict.pop(key)[:, :, None, None]

        embedding_table = state_dict.pop("model.embed_tokens.weight").unsqueeze(1)  # [V, 1, d]
        if not self.disable_embedding_quantization:
            # in fp16 like Apple's path (the state dict is in the export dtype there), so
            # the int8 codes are made with the fp16 scale the graphs multiply by
            embedding_table, scale, zero_point = quantize_per_tensor(
                embedding_table.to(torch.float16), nbits=8, symmetric=True)
        else:
            scale = torch.tensor(1.0, dtype=torch.float16)
            zero_point = torch.tensor(0, dtype=torch.int8)
        state_dict["load_embeddings.embedding_table"] = embedding_table
        state_dict["gather_embeddings.scale"] = scale
        state_dict["gather_embeddings.zero_point"] = zero_point
        state_dict["extend.emb_scale"] = scale
        state_dict["extend.emb_zero_point"] = zero_point

        for k in [k for k in state_dict if k.startswith("model.")]:
            state_dict["extend." + k] = state_dict.pop(k)
        if not self.config.tie_word_embeddings and "lm_head.weight" in state_dict:
            state_dict["extend.lm_head.weight"] = state_dict.pop("lm_head.weight")
        state_dict.pop("lm_head.weight", None)


def truncate_config(config: Lfm3Config, num_layers: int) -> Lfm3Config:
    """The first ``num_layers`` layers (probe / bisect exports). LFM3's first five layers
    hold all three types: conv, conv, full, conv, sliding."""
    import dataclasses

    return dataclasses.replace(config, num_hidden_layers=num_layers, layer_types=list(config.layer_types[:num_layers]))


def lfm3_ios_from_hf(
    src: str,
    *,
    key_prefix: str = "model.",
    config: Lfm3Config | dict | str | None = None,
    config_key: str | None = None,
    dtype: torch.dtype = torch.float16,
    disable_embedding_quantization: bool = False,
    max_context_length: int = 4096,
    act: str = "tanh",
    history: str = "block",
    norm_k=NORM_K,
    num_layers: int | None = None,
    revision: str | None = None,
    emit_hidden: bool = False,
) -> Lfm3ForCausalLMForiOS:
    """Load an LFM3 decoder from an HF checkpoint into the static iOS builder.

    ``src``: local checkpoint dir or HF id; ``key_prefix`` / ``config`` / ``config_key``
    as in macos/lfm3.lfm3_from_hf. The embedding is quantized from the checkpoint's
    values (int8 per tensor, Apple's iOS table) unless ``disable_embedding_quantization``
    (the fp32 reference keeps it float). ``emit_hidden``: extend also returns the
    final-norm hidden (``model.extend.emit_hidden`` can be flipped after loading)."""
    from safetensors import safe_open

    src_dir = resolve_lfm3_source(src, revision)
    cfg = resolve_lfm3_config(config, src_dir, config_key)
    if num_layers is not None:
        cfg = truncate_config(cfg, num_layers)
    model = Lfm3ForCausalLMForiOS(cfg, "meta", disable_embedding_quantization,
                                  max_context_length=max_context_length, act=act, history=history, norm_k=norm_k,
                                  emit_hidden=emit_hidden)

    sd: dict[str, torch.Tensor] = {}
    for path in sorted(glob.glob(os.path.join(src_dir, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                if not key.startswith(key_prefix):
                    continue
                local = "model." + key[len(key_prefix):]
                parts = local.split(".")
                if parts[1] == "layers" and int(parts[2]) >= cfg.num_hidden_layers:
                    continue
                sd[local] = f.get_tensor(key).to(torch.float32)
    if not sd:
        raise KeyError(f"no tensor under {key_prefix!r} in {src_dir}")
    model._mutate_state_dict(sd)
    for k, v in sd.items():
        if v.is_floating_point() and not k.endswith(("emb_scale", "gather_embeddings.scale")):
            sd[k] = v.to(dtype)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"load mismatch: missing={missing} unexpected={unexpected}")
    model.to(dtype)
    meta = [n for n, p in model.named_parameters() if p.is_meta]
    if meta:
        raise RuntimeError(f"Parameters not loaded: {meta}")
    return model.eval()


# --------------------------------------------------------------------------- #
# Trace inputs (the exporter passes them to Apple's export/ios.py functions)
# --------------------------------------------------------------------------- #
def build_ios_reference_inputs(model: Lfm3ForCausalLMForiOS, max_context_length: int, query_len: int = 8) -> dict:
    """Apple's `export/ios.py::_build_ios_reference_inputs` with this builder's cache rows
    (hidden/2 wide) and the cache length declared as 8 x blocks (the history read splits
    it into 8-position blocks)."""
    from coreai_models.export import ios as ios_export

    cfg = model.config
    input_ids = torch.randint(1, cfg.vocab_size, (1, query_len), dtype=torch.int32)
    position_ids = torch.arange(query_len).to(torch.uint16).unsqueeze(0)
    in_step = torch.zeros((1,), dtype=torch.int32)
    causal_mask = torch.zeros(1, max_context_length, 1, query_len, dtype=torch.float16)
    key_cache = torch.zeros(cfg.num_hidden_layers, 1, model.cache_width, 1, max_context_length, dtype=torch.float16)
    value_cache = key_cache.clone()
    embedding_table = model.load_embeddings.embedding_table
    transformer_input = model.gather_embeddings(input_ids, embedding_table)

    forward_inputs = {
        ios_export.TRANSFORMER_INPUT_NAME: transformer_input,
        ios_export.POSITION_IDS_INPUT_NAME: position_ids,
        ios_export.IN_STEP_INPUT_NAME: in_step,
        ios_export.CAUSAL_MASK_INPUT_NAME: causal_mask,
        ios_export.KEY_CACHE_INPUT_NAME: key_cache,
        ios_export.VALUE_CACHE_INPUT_NAME: value_cache,
        ios_export.EMBEDDING_TABLE_INPUT_NAME: embedding_table,
    }
    seq_len_dim = torch.export.Dim("seq_len", max=max_context_length)
    cache_blocks = torch.export.Dim("cache_blocks", min=256 // HISTORY_BLOCK, max=max_context_length // HISTORY_BLOCK)
    cache_len_dim = HISTORY_BLOCK * cache_blocks
    forward_dynamic_shapes = {
        ios_export.TRANSFORMER_INPUT_NAME: {1: seq_len_dim},
        ios_export.POSITION_IDS_INPUT_NAME: {1: seq_len_dim},
        ios_export.IN_STEP_INPUT_NAME: None,
        ios_export.CAUSAL_MASK_INPUT_NAME: {1: cache_len_dim, 3: seq_len_dim},
        ios_export.KEY_CACHE_INPUT_NAME: {4: cache_len_dim},
        ios_export.VALUE_CACHE_INPUT_NAME: {4: cache_len_dim},
        ios_export.EMBEDDING_TABLE_INPUT_NAME: None,
    }
    return {
        "forward_inputs": forward_inputs,
        "embed_tokens_inputs": (input_ids, embedding_table),
        "forward_dynamic_shapes": forward_dynamic_shapes,
        "embed_tokens_dynamic_shapes": {"input_ids": {1: seq_len_dim}, ios_export.EMBEDDING_TABLE_INPUT_NAME: None},
    }


def palettization_example_inputs(model: Lfm3ForCausalLMForiOS, cache_len: int = 256, query_len: int = 8) -> tuple:
    """Example inputs for coreai-opt's palettizer trace of ``model.forward``."""
    cfg = model.config
    input_ids = torch.randint(1, cfg.vocab_size, (1, query_len), dtype=torch.int32)
    position_ids = torch.arange(query_len).to(torch.uint16).unsqueeze(0)
    in_step = torch.zeros((1,), dtype=torch.int32)
    causal_mask = torch.zeros(1, cache_len, 1, query_len, dtype=torch.float16)
    key_cache = torch.zeros(cfg.num_hidden_layers, 1, model.cache_width, 1, cache_len, dtype=torch.float16)
    return (input_ids, position_ids, in_step, causal_mask, key_cache, key_cache.clone())


def register() -> None:
    """Register this builder as the iOS class for model_type `lfm3` (Apple's registry.py
    stays unmodified; the macOS class, if any, is kept). The stock CLI still cannot export
    LFM3 on this checkout — its iOS path sizes the cache rows from n_kv x head_dim and
    transformers 4.57 has no lfm3 config — so the zoo exporter
    (conversion/export_lfm3_ane_static.py) calls Apple's export/ios.py functions with
    this builder's inputs."""
    from coreai_models.models.registry import ModelEntry, _get_registry

    registry = _get_registry()
    old = registry.get("lfm3")
    registry["lfm3"] = ModelEntry(
        macos_class=old.macos_class if old is not None else None,
        ios_class=Lfm3ForCausalLMForiOS,
    )
