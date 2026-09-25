"""Fused activation-site kernels for Bonsai 2 decode (S=1): the ops around each Hadamard site
folded into the transform's own dispatch.

Every ternary linear reads a Hadamard-rotated activation (`bonsai_hadamard_metal`). At S=1 the
ops that produce that activation are a handful of tiny dispatches, each behind ~10 µs of gap, and
every custom-kernel boundary adds a copy-in and a copy-out (the converter wraps each kernel input
in `copy_with_constraints` and each result in `copy_discarding_constraints`; measured: a chain of
two kernels keeps all of them). So the sites are fused with their producers:

  * `bonsai_add_norm_fwht`  h = x + r;  n = RMSNormPlusOne(h);  y = FWHT(n ⊙ sign) / 32
        -> the residual add, the pre-norm and the transform of the attention / MLP input in one
           dispatch; returns h (the new residual stream), n (the un-rotated normalized vector,
           which the GDN's a/b projections read) and y.
  * `bonsai_norm_fwht`      same without the add (the first layer's input, the embedding).
  * `bonsai_swiglu_fwht`    y = FWHT((up ⊙ silu(gate)) ⊙ sign) / 32  (the down_proj input).

Numerics follow the graph they replace: h and every activation the graph materializes in fp16
are rounded to fp16 at the same points; the mean of squares, the transform and SiLU are fp32.
RMSNormPlusOne: n = half(h · rsqrt(mean(h²) + eps) · (w + 1)) with h read as fp16 and the gain
promoted to fp32 (`RMSNormImpl` with an fp32 scale).

Geometry = the FWHT kernel's: one 1024-block per 256-thread threadgroup, 4 values per thread;
the norm kernels first reduce the whole K-vector's sum of squares redundantly per threadgroup
(K/256 loads per thread; K = 5120 here, 5 threadgroups), which is far cheaper than a second
dispatch. Layout (MSL sees torch shapes reversed): X, R, U, G torch [1, K] f16 -> X[k, 0];
W torch [K] f16; SG torch [K] f32; outputs torch [1, K] f16.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from coreai_torch import MetalParameter, TorchMetalKernel
from coreai_models.models.macos.bonsai_hadamard_metal import BLOCK, hadamard_matrix

_PARAMS = [MetalParameter("tid", "uint2", "thread_position_in_threadgroup"),
           MetalParameter("tgid", "uint2", "threadgroup_position_in_grid")]
_TG = (32, 8, 1)

# the in-register / threadgroup-memory butterfly of bonsai_hadamard_metal (reg[4], shmem[1024], t, lane)
_BUTTERFLY = """
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
"""

_NORM_SRC = """
    const uint N = 1024u, NT = 256u, NE = 4u, NW = 32u;
    const uint K = X.get_extent(0);              // X torch [1, K] -> X[k, 0]
    const uint base = tgid.y * N;
    const uint lane = tid.x;
    const uint t = tid.y * NW + lane;

    threadgroup float shmem[1024];
    threadgroup float red[8];

    // sum of squares of h over the whole vector (fp16 h, fp32 squares and sum), once per threadgroup
    float ss = 0.0f;
    for (uint e = t; e < K; e += NT) {
        const float h = float(__H16__);
        ss += h * h;
    }
    ss = simd_sum(ss);
    if (lane == 0) red[tid.y] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < 8u; ++i) tot += red[i];
    const float inv = rsqrt(tot / float(K) + __EPS__);

    float reg[4];
    for (uint i = 0; i < NE; ++i) {
        const uint e = base + i * NT + t;
        const half h = __H16__;
        __STORE_H__
        const half n16 = half(float(h) * inv * (float(W[e]) + 1.0f));   // RMSNormPlusOne, fp32 gain
        NRM[e, 0] = n16;
        reg[i] = float(n16) * float(SG[e]) * 0.03125f;
    }
""" + _BUTTERFLY + """
    for (uint i = 0; i < NE; ++i) Y[base + i * NT + t, 0] = TYPE(reg[i]);
"""

_SWIGLU_SRC = """
    const uint N = 1024u, NT = 256u, NE = 4u, NW = 32u;
    const uint base = tgid.y * N;
    const uint lane = tid.x;
    const uint t = tid.y * NW + lane;

    threadgroup float shmem[1024];

    float reg[4];
    for (uint i = 0; i < NE; ++i) {
        const uint e = base + i * NT + t;
        const float g = float(G[e, 0]);
        const float s = float(half(g / (1.0f + exp(-g))));           // silu(gate) is an fp16 tensor
        const half p = half(float(U[e, 0]) * s);                      // up * silu(gate), fp16
        reg[i] = float(p) * float(SG[e]) * 0.03125f;
    }
""" + _BUTTERFLY + """
    for (uint i = 0; i < NE; ++i) Y[base + i * NT + t, 0] = TYPE(reg[i]);
"""


def _round16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).to(torch.float32)


def _fwht_blocks(v: torch.Tensor, sg: torch.Tensor) -> torch.Tensor:
    """[1, K] fp32 -> FWHT((v ⊙ sg) per 1024-block) / 32 in fp32."""
    k = v.shape[-1]
    h = hadamard_matrix(BLOCK, torch.float32)
    return ((v * sg.to(torch.float32)).reshape(1, k // BLOCK, BLOCK) @ h).reshape(1, k) * (1.0 / 32.0)


def add_norm_fwht_reference(X, R, W, SG, eps: float):
    h = X.to(torch.float32) + (R.to(torch.float32) if R is not None else 0.0)
    h = h.to(torch.float16)
    hf = h.to(torch.float32)
    inv = torch.rsqrt((hf * hf).mean(-1, keepdim=True) + eps)
    n = (hf * inv * (W.to(torch.float32) + 1.0)).to(torch.float16)
    y = _fwht_blocks(n.to(torch.float32), SG).to(torch.float16)
    return h, n, y


def swiglu_fwht_reference(U, G, SG):
    s = _round16(F.silu(G.to(torch.float32)))
    p = _round16(U.to(torch.float32) * s)
    return _fwht_blocks(p, SG).to(U.dtype)


def build_add_norm_fwht_kernel(eps: float, name: str = "bonsai_add_norm_fwht") -> TorchMetalKernel:
    def _torch_defn(X: torch.Tensor, R: torch.Tensor, W: torch.Tensor,
                    SG: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return add_norm_fwht_reference(X, R, W, SG, eps)

    src = (_NORM_SRC.replace("__H16__", "half(float(X[e, 0]) + float(R[e, 0]))")
           .replace("__STORE_H__", "H[e, 0] = h;").replace("__EPS__", f"{eps!r}f"))
    return TorchMetalKernel(name, input_names=["X", "R", "W", "SG"], result_names=["H", "NRM", "Y"],
                            src=src, torch_defn=_torch_defn, metal_params=_PARAMS,
                            template_dtypes={"X": "TYPE"})


def build_norm_fwht_kernel(eps: float, name: str = "bonsai_norm_fwht") -> TorchMetalKernel:
    def _torch_defn(X: torch.Tensor, W: torch.Tensor, SG: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, n, y = add_norm_fwht_reference(X, None, W, SG, eps)
        return n, y

    src = (_NORM_SRC.replace("__H16__", "X[e, 0]").replace("__STORE_H__", "")
           .replace("__EPS__", f"{eps!r}f"))
    return TorchMetalKernel(name, input_names=["X", "W", "SG"], result_names=["NRM", "Y"],
                            src=src, torch_defn=_torch_defn, metal_params=_PARAMS,
                            template_dtypes={"X": "TYPE"})


def build_swiglu_fwht_kernel(name: str = "bonsai_swiglu_fwht") -> TorchMetalKernel:
    def _torch_defn(U: torch.Tensor, G: torch.Tensor, SG: torch.Tensor) -> torch.Tensor:
        return swiglu_fwht_reference(U, G, SG)

    return TorchMetalKernel(name, input_names=["U", "G", "SG"], result_names=["Y"],
                            src=_SWIGLU_SRC, torch_defn=_torch_defn, metal_params=_PARAMS,
                            template_dtypes={"U": "TYPE"})


def _grid(k: int):
    return dict(threads_per_grid=(32, 8 * (k // BLOCK), 1), threads_per_thread_group=_TG)


def add_norm_fwht(kernel, x: torch.Tensor, r: torch.Tensor, w: torch.Tensor, signs: torch.Tensor):
    """x, r [1, 1, K] -> (h, nrm, y) each [1, 1, K]."""
    k = x.shape[-1]
    h, n, y = kernel(x.reshape(1, k), r.reshape(1, k), w, signs, result_shapes=[[1, k]] * 3, **_grid(k))
    return h.reshape(1, 1, k), n.reshape(1, 1, k), y.reshape(1, 1, k)


def norm_fwht(kernel, x: torch.Tensor, w: torch.Tensor, signs: torch.Tensor):
    k = x.shape[-1]
    n, y = kernel(x.reshape(1, k), w, signs, result_shapes=[[1, k]] * 2, **_grid(k))
    return n.reshape(1, 1, k), y.reshape(1, 1, k)


def swiglu_fwht(kernel, up: torch.Tensor, gate: torch.Tensor, signs: torch.Tensor):
    k = up.shape[-1]
    y = kernel(up.reshape(1, k), gate.reshape(1, k), signs, result_shapes=[[1, k]], **_grid(k))
    return y.reshape(1, 1, k)


def seed_site(site, y: torch.Tensor) -> torch.Tensor:
    """Tell a HadamardSite that `y` is already its transform of itself: the linears reading the
    site then get `y` back from the memo instead of transforming it again."""
    object.__setattr__(site, "_memo", (y, y))
    return y
