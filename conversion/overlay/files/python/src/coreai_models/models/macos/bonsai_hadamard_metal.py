"""Blockwise signed Walsh-Hadamard transform (block 1024) for Core AI — Bonsai 2's activation rotation.

Bonsai 2 27B (prism-ml/Ternary-Bonsai-2-27B) stores every ternary matrix in a rotated basis
``W' = W · Rᵀ`` with ``R = (1/√1024) · H₁₀₂₄ · S`` applied blockwise along K, so before each folded
matmul the activation must be transformed: multiply by a fixed ±1 sign vector over the full K
width, then a 1024-point Sylvester (natural-order) Walsh-Hadamard transform on each consecutive
1024-block, scaled by 1/32. The same kernel with the sign applied AFTER the transform is the
inverse (used once, after the embedding lookup). See knowledge/bonsai-ternary-hadamard.md §4.

Kernel design (mirrors PrismML's `kernel_fwht_tg<1024, 256>` in their llama.cpp fork,
ggml/src/ggml-metal/kernels/misc.metal): one row of 1024 per 256-thread threadgroup, four values
per thread. Butterflies at strides 1..16 are `simd_shuffle_xor` within a simdgroup, strides
32..128 go through 4 KB of threadgroup memory, strides 256 and 512 stay in registers. Sign and
1/32 scale are folded into the load; accumulation is fp32; the output is rounded once to the
activation dtype — the same rounding chain as the MLX reference runtime (fp16 in, fp32
transform, fp16 out).

Layout (MSL sees torch shapes reversed):
  X torch [S, K] -> X[k, s]      K % 1024 == 0
  SG torch [K]   -> SG[k]        ±1 as float32 (or the activation dtype)
  Y torch [S, K] -> Y[k, s]
Grid: threads_per_thread_group (32, 8, 1); threads_per_grid (32, 8 * S * K/1024, 1), so
tgid.y enumerates the S·K/1024 rows.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coreai_torch import MetalParameter, TorchMetalKernel

BLOCK = 1024
_NT = 256                 # threads per row
_NE = BLOCK // _NT        # 4 values per thread
_TG = (32, 8, 1)

_FWHT_SRC = """
    const uint N = 1024u, NT = 256u, NE = 4u, NW = 32u;
    const uint K = X.get_extent(0);              // X torch [S, K] -> X[k, s]
    const uint nblk = K / N;
    const uint r = tgid.y;                       // row of 1024 = (s, block)
    const uint s = r / nblk;
    const uint base = (r - s * nblk) * N;
    const uint lane = tid.x;                     // 0..31
    const uint t = tid.y * NW + lane;            // 0..255

    threadgroup float shmem[1024];

    // load: sign flip and 1/sqrt(1024) folded in, exact (factors are +-1 and a power of two)
    float reg[4];
    for (uint i = 0; i < NE; ++i) {
        const uint e = base + i * NT + t;
        reg[i] = float(X[e, s]) * float(SG[e]) * 0.03125f;
    }
    // strides 1..16: element bits 0..4 live in the lane index
    for (uint i = 1; i < NW; i *= 2) {
        for (uint j = 0; j < NE; ++j) {
            const float val  = reg[j];
            const float val2 = simd_shuffle_xor(val, i);
            reg[j] = (lane & i) == 0 ? val2 + val : val2 - val;
        }
    }
    // strides 32..128: bits 5..7 live in the simdgroup index -> threadgroup memory
    for (uint i = NW; i < NT; i *= 2) {
        for (uint j = 0; j < NE; ++j) shmem[j * NT + t] = reg[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < NE; ++j) {
            const float val  = reg[j];
            const float val2 = shmem[j * NT + (t ^ i)];
            reg[j] = (t & i) == 0 ? val2 + val : val2 - val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // strides 256, 512: bits 8..9 are the register index
    for (uint i = NT; i < N; i *= 2) {
        const uint step = i / NT;
        for (uint j = 0; j < NE; j += 2u * step) {
            for (uint k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
    for (uint i = 0; i < NE; ++i) {
        Y[base + i * NT + t, s] = TYPE(reg[i]);
    }
"""

_PARAMS = [MetalParameter("tid", "uint2", "thread_position_in_threadgroup"),
           MetalParameter("tgid", "uint2", "threadgroup_position_in_grid")]


def hadamard_matrix(n: int = BLOCK, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Sylvester (natural-order) Walsh-Hadamard matrix, unnormalised: H[i, j] = (-1)^popcount(i & j).

    This is the order PrismML's loader materialises (llama-model.cpp, `parity = row & col`) and
    the order `mx.hadamard_transform` / the in-register butterfly produce.
    """
    idx = torch.arange(n, dtype=torch.int64)
    p = idx[:, None] & idx[None, :]
    p = p ^ (p >> 8)
    p = p ^ (p >> 4)
    p = p ^ (p >> 2)
    p = p ^ (p >> 1)
    return (1 - 2 * (p & 1)).to(dtype)


def _fwht_torch_defn(x: torch.Tensor, sg: torch.Tensor) -> torch.Tensor:
    """Reference: y = ((x * sg) per-1024-block @ H) / 32, fp32 compute, output in x's dtype."""
    s, k = x.shape
    h = hadamard_matrix(BLOCK, torch.float32)
    y = (x.float() * sg.float()).reshape(s, k // BLOCK, BLOCK) @ h
    return (y * (1.0 / 32.0)).reshape(s, k).to(x.dtype)


def signed_hadamard_reference(x: torch.Tensor, signs: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Exact (fp64) forward or inverse transform for gating: forward = H·S·x/32, inverse = S·H·x/32."""
    s, k = x.shape
    h = hadamard_matrix(BLOCK, torch.float64)
    xx = x.to(torch.float64)
    if not inverse:
        xx = xx * signs.to(torch.float64)
    y = (xx.reshape(s, k // BLOCK, BLOCK) @ h).reshape(s, k) / 32.0
    if inverse:
        y = y * signs.to(torch.float64)
    return y


def build_fwht_kernel(name: str = "bonsai_fwht1024") -> TorchMetalKernel:
    """Signed blockwise FWHT-1024 kernel: Y = FWHT(X * SG) / 32 per 1024-block, fp32 inside."""
    return TorchMetalKernel(name, input_names=["X", "SG"], result_names=["Y"], src=_FWHT_SRC,
                            torch_defn=_fwht_torch_defn, metal_params=_PARAMS,
                            template_dtypes={"X": "TYPE"})


def fwht_call(kernel: TorchMetalKernel, x2d: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Dispatch on x2d [S, K] (K % 1024 == 0) with signs [K] -> [S, K]."""
    s, k = int(x2d.shape[0]), int(x2d.shape[1])
    rows = s * (k // BLOCK)
    return kernel(x2d, signs, threads_per_grid=(32, 8 * rows, 1),
                  threads_per_thread_group=_TG, result_shapes=[[s, k]])


class HadamardSite(nn.Module):
    """One activation site: x -> FWHT₁₀₂₄(x ⊙ signs) / 32, computed once per input tensor.

    Several folded linears read the same activation (q/k/v; qkv/z; gate/up). Each holds a
    reference to the layer's site and calls `transform(x)`; the site memoises on tensor
    identity, so within one forward (and one export trace) the transform is emitted once and
    the linears share its output — the same sharing PrismML's graph does with `hadamard_memo`.
    """

    def __init__(self, signs: torch.Tensor, kernel: TorchMetalKernel) -> None:
        super().__init__()
        k = int(signs.numel())
        if k % BLOCK:
            raise ValueError(f"K={k} is not a multiple of the Hadamard block {BLOCK}")
        if not torch.all((signs == 1) | (signs == -1)):
            raise ValueError("sign vector must be ±1")
        self.kernel = kernel
        self.K = k
        self.register_buffer("signs", signs.to(torch.float32).contiguous())
        object.__setattr__(self, "_memo", None)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        memo = self._memo
        if memo is not None and memo[0] is x:
            return memo[1]
        lead = x.shape[:-1]
        y = fwht_call(self.kernel, x.reshape(-1, self.K), self.signs).reshape(*lead, self.K)
        object.__setattr__(self, "_memo", (x, y))
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


# ----------------------------------------------------------------------------- embedding
# The embedding table stores rotated rows (R·e); the lookup result needs the INVERSE:
# e = S · (H z) / 32 — transform first, then sign. One kernel does gather + dequant + inverse
# transform, so the 248320 x 5120 table stays packed (0.34 GB) instead of fp16 (2.5 GB).
#   IDS torch [S] int32; QP torch [N, K/16] u32 -> QP[w, n]; D torch [N, K/128] f16 -> D[g, n];
#   SG torch [K] f32; Y torch [S, K] f16 -> Y[k, s]
_EMBED_SRC = """
    const uint N = 1024u, NT = 256u, NE = 4u, NW = 32u;
    const uint K = SG.get_extent(0);
    const uint nblk = K / N;
    const uint r = tgid.y;
    const uint s = r / nblk;
    const uint base = (r - s * nblk) * N;
    const uint lane = tid.x;
    const uint t = tid.y * NW + lane;
    const uint row = uint(IDS[s]);

    threadgroup float shmem[1024];

    float reg[4];
    for (uint i = 0; i < NE; ++i) {
        const uint e = base + i * NT + t;
        const uint packed = uint(QP[e >> 4, row]);
        const int q = int((packed >> ((e & 15u) * 2u)) & 0x3u);
        reg[i] = float(q - 1) * float(D[e >> 7, row]) * 0.03125f;
    }
    for (uint i = 1; i < NW; i *= 2) {
        for (uint j = 0; j < NE; ++j) {
            const float val  = reg[j];
            const float val2 = simd_shuffle_xor(val, i);
            reg[j] = (lane & i) == 0 ? val2 + val : val2 - val;
        }
    }
    for (uint i = NW; i < NT; i *= 2) {
        for (uint j = 0; j < NE; ++j) shmem[j * NT + t] = reg[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < NE; ++j) {
            const float val  = reg[j];
            const float val2 = shmem[j * NT + (t ^ i)];
            reg[j] = (t & i) == 0 ? val2 + val : val2 - val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = NT; i < N; i *= 2) {
        const uint step = i / NT;
        for (uint j = 0; j < NE; j += 2u * step) {
            for (uint k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
    for (uint i = 0; i < NE; ++i) {
        const uint e = base + i * NT + t;
        Y[e, s] = TYPE(reg[i] * float(SG[e]));
    }
"""


def _embed_torch_defn(ids: torch.Tensor, qp: torch.Tensor, d: torch.Tensor,
                      sg: torch.Tensor) -> torch.Tensor:
    """Reference: gather packed rows, dequant (code-1)*d, inverse transform S·H·z/32, fp16 out."""
    s = ids.shape[0]
    k = int(sg.shape[0])
    rows = qp[ids.to(torch.int64)].to(torch.int64)                      # [S, K/16]
    shifts = (2 * torch.arange(16, dtype=torch.int64)).view(1, 1, 16)
    codes = ((rows.unsqueeze(-1) >> shifts) & 0x3).reshape(s, k)         # [S, K]
    scale = d[ids.to(torch.int64)].to(torch.float32).repeat_interleave(BLOCK // 8, dim=-1)
    z = (codes - 1).to(torch.float32) * scale
    h = hadamard_matrix(BLOCK, torch.float32)
    y = (z.reshape(s, k // BLOCK, BLOCK) @ h).reshape(s, k) * (1.0 / 32.0)
    return (y * sg.to(torch.float32)).to(d.dtype)


def build_embed_kernel(name: str = "bonsai_embed_ifwht1024") -> TorchMetalKernel:
    """Gather + dequant + inverse signed FWHT-1024 for a packed (PQ2_0) embedding table."""
    return TorchMetalKernel(name, input_names=["IDS", "QP", "D", "SG"], result_names=["Y"],
                            src=_EMBED_SRC, torch_defn=_embed_torch_defn, metal_params=_PARAMS,
                            template_dtypes={"D": "TYPE"})


class PackedEmbedding(nn.Module):
    """nn.Embedding replacement over a rotated, PQ2_0-packed table (qp [V, K/16], d [V, K/128])."""

    def __init__(self, qp: torch.Tensor, d: torch.Tensor, signs: torch.Tensor,
                 kernel: TorchMetalKernel) -> None:
        super().__init__()
        self.V, k16 = int(qp.shape[0]), int(qp.shape[1])
        self.K = k16 * 16
        if self.K % BLOCK or int(signs.numel()) != self.K:
            raise ValueError("embedding width must be a multiple of 1024 and match the sign vector")
        self.kernel = kernel
        self.register_buffer("qp", qp.contiguous())
        self.register_buffer("d", d.to(torch.float16).contiguous())
        self.register_buffer("signs", signs.to(torch.float32).contiguous())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        lead = input_ids.shape
        ids = input_ids.reshape(-1).to(torch.int32)
        s = int(ids.shape[0])
        rows = s * (self.K // BLOCK)
        y = self.kernel(ids, self.qp, self.d, self.signs,
                        threads_per_grid=(32, 8 * rows, 1), threads_per_thread_group=_TG,
                        result_shapes=[[s, self.K]])
        return y.reshape(*lead, self.K)


class BonsaiHadamard(nn.Module):
    """The activation-side rotation for one K width: x -> FWHT₁₀₂₄(x ⊙ signs) / 32.

    One instance per K width (5120 / 6144 / 17408 for Bonsai 2 27B); the sign vector is the
    model's `prism.hadamard.sign_values` slice for that width. Apply once per activation site
    and feed the result to every folded linear reading that activation.
    """

    def __init__(self, signs: torch.Tensor, kernel: TorchMetalKernel) -> None:
        super().__init__()
        k = int(signs.numel())
        if k % BLOCK:
            raise ValueError(f"K={k} is not a multiple of the Hadamard block {BLOCK}")
        if not torch.all((signs == 1) | (signs == -1)):
            raise ValueError("sign vector must be ±1")
        self.kernel = kernel
        self.K = k
        self.register_buffer("signs", signs.to(torch.float32).contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        k = int(x.shape[-1])
        if k != self.K:
            raise ValueError(f"x has K={k}, transform built for K={self.K}")
        y = fwht_call(self.kernel, x.reshape(-1, k), self.signs)
        return y.reshape(*lead, k)
