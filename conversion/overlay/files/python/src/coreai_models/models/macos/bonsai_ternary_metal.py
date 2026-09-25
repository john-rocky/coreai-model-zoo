"""Ternary g128 (PQ2_0) matvec + tiled GEMM for Core AI — the BitCPM kernels at scale group 128.

Bonsai 2's PQ2_0 blocks are one fp16 scale + 32 bytes of 2-bit codes per **128** weights
(`lc:ggml/src/ggml-common.h:202-206`); read as little-endian words the codes land exactly in the
zoo's BitCPM `[N, K/16] uint32` layout (code j of a 16-group at bits 2j, value = code - 1), so the
kernels below are `bitcpm_ternary_metal.py` / `bitcpm_ternary_gemm.py` with the scale index
`k0 >> 8` -> `k0 >> 7` and `D` shaped `[N, K/128]`. Both invariants those kernels rely on still
hold at 128: a matvec lane's 16 codes sit inside one scale group (16 | 128) and a GEMM K-step of
64 never straddles one (64 | 128). See knowledge/bonsai-ternary-hadamard.md §3.1 and §8.

The Hadamard rotation is NOT in here: these kernels take the already-transformed activation
(`bonsai_hadamard_metal.HadamardSite`), because one transform feeds several folded linears.

Layout (MSL sees torch shapes reversed):
  A  torch [M, K]         -> A[k, m]
  QP torch [N, K/16] u32  -> QP[w, n]
  D  torch [N, K/128] f16 -> D[g, n]
  C  torch [M, N]         -> C[n, m]
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from coreai_torch import MetalParameter, TorchMetalKernel

SCALE_BLK = 128          # PQ2_0 group size along K
_R, _SGY = 4, 8          # matvec: rows per simdgroup, simdgroups per threadgroup (32 rows/TG)
_KB = 512                # matvec: K per step (32 lanes x 16 codes)
_BN, _BK, _TN = 64, 64, 4
_PARAMS = [MetalParameter("tid", "uint2", "thread_position_in_threadgroup"),
           MetalParameter("tgid", "uint2", "threadgroup_position_in_grid")]

# ----------------------------------------------------------------------------- M=1 matvec
#
# The code -> float step is the whole cost of this kernel. Profiled on the M4 Pro (2026-09-18,
# Metal System Trace + a standalone microbench of this exact loop): with one int->float
# conversion per code the matvec runs every shape at ~132 GB/s, half the chip's ~245 GB/s
# streaming read, and neither fewer rows per simdgroup, vector loads, prefetch, nor fewer
# integer ops around it move that number — a loads-only twin of the same access pattern
# streams at 248. So the conversion instruction is the wall (a reduced-rate op on this GPU),
# not bandwidth, ALU count or the access pattern. The unorm unpack unit converts four codes
# in one instruction: split the word into four bytes-of-codes with a shift and a mask and
# read them back as c/255; the -1 of the ternary map is one subtraction of the lane's
# activation sum, since sum(x*(c-1)) = 255*sum(x*c/255) - sum(x). Same fp32 accumulation,
# same launch geometry (R=4, SGY=8, 16 codes per lane); 231-246 GB/s on every Bonsai shape,
# rel err vs the fp32 reference unchanged (4e-4, the fp16 output rounding).
_MV_SRC = """
    const uint R = __R__, SGY = __SGY__;
    const uint K = A.get_extent(0);        // A torch [1, K]
    const uint lane = tid.x;               // 0..31
    const uint sg = tid.y;                 // 0..SGY-1
    const uint base_row = (tgid.y * SGY + sg) * R;

    float acc[__R__];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 512) {      // 32 lanes * 16 codes = 512
        uint k0 = kb + lane * 16;
        float xr[16];
        float sx = 0.0f;
        for (uint j = 0; j < 16; ++j) { xr[j] = float(A[k0 + j, 0]); sx += xr[j]; }
        uint w0 = (kb >> 4) + lane;             // one uint32 (16 codes) per lane
        uint g  = k0 >> 7;                       // 128-scale group (this lane's 16 codes are inside it)
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint packed = uint(QP[w0, n]);
            float4 p4 = 0.0f;
            for (uint j = 0; j < 4; ++j) {
                // bytes hold codes j, j+4, j+8, j+12; the unpack unit turns them into c/255
                float4 c = unpack_unorm4x8_to_float((packed >> (2 * j)) & 0x03030303u);
                p4 = fma(c, float4(xr[j], xr[j + 4], xr[j + 8], xr[j + 12]), p4);
            }
            float p = 255.0f * (p4.x + p4.y + p4.z + p4.w) - sx;   // sum(x * (c - 1))
            acc[r] += p * float(D[g, n]);        // per-lane-group scale, then accumulate
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) C[base_row + r, 0] = TYPE(tot);
    }
"""

# ----------------------------------------------------------------------------- M=1 multi matvec
#
# Round 3: the runtime's CPU-side encode of a step (one MPSGraph region + copies per custom
# kernel call) is nearly as long as the GPU work, so kernel CALLS are the currency. Linears that
# read the same activation (gate/up, qkv/z, q/k/v) become one dispatch: the grid spans the rows
# of all weight sets, each simdgroup picks its set from its row index (a uniform branch), and
# the per-row code is exactly the single-matvec loop, so results are bit-identical to it.
_MV_MULTI_HEAD = """
    const uint R = __R__, SGY = __SGY__;
    const uint K = A.get_extent(0);
    const uint lane = tid.x;
    const uint sg = tid.y;
    uint row = (tgid.y * SGY + sg) * R;         // row in the concatenated [N1 + N2 (+ N3)] space
    float xr[16];
    float acc[__R__];
"""
_MV_MULTI_SET = """
    if (row < __N__) {
        for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
        for (uint kb = 0; kb < K; kb += 512) {
            uint k0 = kb + lane * 16;
            float sx = 0.0f;
            for (uint j = 0; j < 16; ++j) { xr[j] = float(A[k0 + j, 0]); sx += xr[j]; }
            uint w0 = (kb >> 4) + lane;
            uint g  = k0 >> 7;
            for (uint r = 0; r < R; ++r) {
                uint n = row + r;
                uint packed = uint(__QP__[w0, n]);
                float4 p4 = 0.0f;
                for (uint j = 0; j < 4; ++j) {
                    float4 c = unpack_unorm4x8_to_float((packed >> (2 * j)) & 0x03030303u);
                    p4 = fma(c, float4(xr[j], xr[j + 4], xr[j + 8], xr[j + 12]), p4);
                }
                float p = 255.0f * (p4.x + p4.y + p4.z + p4.w) - sx;
                acc[r] += p * float(__D__[g, n]);
            }
        }
        for (uint r = 0; r < R; ++r) {
            float tot = simd_sum(acc[r]);
            if (lane == 0) __C__[row + r, 0] = TYPE(tot);
        }
        return;
    }
    row -= __N__;
"""


def _mv_multi_src(nsets: int) -> str:
    src = _MV_MULTI_HEAD
    for i in range(1, nsets + 1):
        src += (_MV_MULTI_SET.replace("__N__", f"QP{i}.get_extent(1)").replace("__QP__", f"QP{i}")
                .replace("__D__", f"D{i}").replace("__C__", f"C{i}"))
    return src.replace("__R__", str(_R)).replace("__SGY__", str(_SGY))


# ----------------------------------------------------------------------------- tiled GEMM
_GEMM_SRC = """
    const uint BM = __BM__, BN = __BN__, BK = __BK__, TM = __TM__, TN = __TN__;
    const uint K = A.get_extent(0);
    const uint t = tid.y * 32u + tid.x;          // 0..255
    const uint n_base = tgid.y * BN;

    threadgroup half xs[__BK__][__BM__];
    threadgroup half ws[__BK__][__BN__];

    const uint tm = t / 16u;                     // 0..15 -> rows [tm*TM, +TM)
    const uint tn = t % 16u;                     // 0..15 -> cols [tn*TN, +TN)

    float acc[__TM__][__TN__];
    for (uint i = 0; i < TM; ++i)
        for (uint j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    for (uint k0 = 0; k0 < K; k0 += BK) {
        for (uint idx = t; idx < BK * BM; idx += 256u) {
            uint kk = idx / BM, mm = idx - kk * BM;
            xs[kk][mm] = half(A[k0 + kk, mm]);
        }
        {
            uint n = t % BN;                     // one (n, 16-code word) pair per thread
            uint wpart = t / BN;                 // 0..3  (BK/16)
            uint packed = uint(QP[(k0 >> 4) + wpart, n_base + n]);
            float d = float(D[k0 >> 7, n_base + n]);   // 64 | 128: one scale per K-step
            for (uint j = 0; j < 16; ++j) {
                int q = int((packed >> (j * 2u)) & 0x3u);
                ws[wpart * 16u + j][n] = half(float(q - 1) * d);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < BK; ++kk) {
            float xr[__TM__], wr[__TN__];
            for (uint i = 0; i < TM; ++i) xr[i] = float(xs[kk][tm * TM + i]);
            for (uint j = 0; j < TN; ++j) wr[j] = float(ws[kk][tn * TN + j]);
            for (uint i = 0; i < TM; ++i)
                for (uint j = 0; j < TN; ++j) acc[i][j] += xr[i] * wr[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < TM; ++i)
        for (uint j = 0; j < TN; ++j)
            C[n_base + tn * TN + j, tm * TM + i] = TYPE(acc[i][j]);
"""


# ----------------------------------------------------------------------------- packing
def pq2_words_and_scales(raw: np.ndarray, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """PQ2_0 row bytes -> (qp [N, K/16] uint32, d [N, K/128] fp16). A reinterpret, no arithmetic.

    `raw` is the tensor's byte buffer (any shape; N*K/128*34 bytes). Block = 2 B fp16 scale then
    32 B of codes; the 32 code bytes are 8 little-endian uint32 words with code j at bits 2j.
    """
    blocks = n * k // SCALE_BLK
    data = np.ascontiguousarray(raw).reshape(-1).view(np.uint8)
    if data.size != blocks * 34:
        raise ValueError(f"PQ2_0 byte count {data.size} != {blocks} blocks x 34 B for [{n}, {k}]")
    data = data.reshape(blocks, 34)
    scales = data[:, :2].copy().view("<f2").reshape(n, k // SCALE_BLK)
    words = data[:, 2:].copy().view("<u4").reshape(n, k // 16)
    return torch.from_numpy(words).contiguous(), torch.from_numpy(scales).contiguous()


def _unpack_words(qp: torch.Tensor) -> torch.Tensor:
    """[N, K/16] (u32 as int32) -> [N, K] codes {0,1,2}."""
    n, k16 = qp.shape
    p = qp.to(torch.int64)
    return torch.stack([(p >> (2 * j)) & 0x3 for j in range(16)], dim=-1).reshape(n, k16 * 16)


def dequant_reference(qp: torch.Tensor, d: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """[N, K] = (code - 1) * d, exact."""
    codes = _unpack_words(qp)
    return ((codes - 1).to(torch.float32) * d.to(torch.float32).repeat_interleave(SCALE_BLK, dim=1)).to(dtype)


def _tern_torch_defn(x: torch.Tensor, qp: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Reference: C[M, N] = x @ W.T with W dequantised per 128-group."""
    w = dequant_reference(qp, d, x.dtype)
    return torch.nn.functional.linear(x, w)


# ----------------------------------------------------------------------------- kernels
def build_mv_kernel(name: str = "bonsai_tern_mv128") -> TorchMetalKernel:
    return TorchMetalKernel(
        name, input_names=["A", "QP", "D"], result_names=["C"],
        src=_MV_SRC.replace("__R__", str(_R)).replace("__SGY__", str(_SGY)),
        torch_defn=_tern_torch_defn, metal_params=_PARAMS, template_dtypes={"A": "TYPE"})


def _mv_multi_torch_defn(nsets: int):
    if nsets == 2:
        def defn(A: torch.Tensor, QP1: torch.Tensor, D1: torch.Tensor, QP2: torch.Tensor,
                 D2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return _tern_torch_defn(A, QP1, D1), _tern_torch_defn(A, QP2, D2)
    elif nsets == 3:
        def defn(A: torch.Tensor, QP1: torch.Tensor, D1: torch.Tensor, QP2: torch.Tensor, D2: torch.Tensor,
                 QP3: torch.Tensor, D3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return _tern_torch_defn(A, QP1, D1), _tern_torch_defn(A, QP2, D2), _tern_torch_defn(A, QP3, D3)
    else:
        raise ValueError(nsets)
    return defn


def build_mv_multi_kernel(nsets: int, name: str | None = None) -> TorchMetalKernel:
    """The M=1 matvec over `nsets` (2 or 3) weight sets that share one activation: one dispatch,
    results C1..Cn, bit-identical to `nsets` calls of `bonsai_tern_mv128`."""
    names = ["A"] + [f"{p}{i}" for i in range(1, nsets + 1) for p in ("QP", "D")]
    return TorchMetalKernel(name or f"bonsai_tern_mv128_x{nsets}", input_names=names,
                            result_names=[f"C{i}" for i in range(1, nsets + 1)], src=_mv_multi_src(nsets),
                            torch_defn=_mv_multi_torch_defn(nsets), metal_params=_PARAMS,
                            template_dtypes={"A": "TYPE"})


def ternary_multi(kernel: TorchMetalKernel, x: torch.Tensor, lins: list) -> list[torch.Tensor]:
    """x [..., K] (already Hadamard-transformed for the linears' shared site) through the multi
    matvec over `lins` (TernaryLinear128s of the same K); returns one [..., N_i] per linear."""
    lead = x.shape[:-1]
    k = lins[0].K
    if any(l.K != k for l in lins):
        raise ValueError("ternary_multi: linears must share K")
    x2 = x.reshape(1, k)
    args = [x2]
    for l in lins:
        args += [l.qp, l.d]
    total = sum(l.N for l in lins)
    outs = kernel(*args, threads_per_grid=(32, total // _R, 1), threads_per_thread_group=(32, _SGY, 1),
                  result_shapes=[[1, l.N] for l in lins])
    return [o.reshape(*lead, l.N) for o, l in zip(outs, lins)]


def build_gemm_kernel(bm: int, name: str | None = None) -> TorchMetalKernel:
    if bm % 16 or not 16 <= bm <= 128:
        raise ValueError(f"BM={bm} must be a multiple of 16 in [16,128]")
    src = (_GEMM_SRC.replace("__BM__", str(bm)).replace("__BN__", str(_BN))
           .replace("__BK__", str(_BK)).replace("__TM__", str(bm // 16))
           .replace("__TN__", str(_TN)))
    return TorchMetalKernel(name or f"bonsai_tern_gemm128_m{bm}",
                            input_names=["A", "QP", "D"], result_names=["C"],
                            src=src, torch_defn=_tern_torch_defn,
                            metal_params=_PARAMS, template_dtypes={"A": "TYPE"})


class TernaryLinear128(nn.Module):
    """y = x' @ W'.T on packed PQ2_0 words; x' is the Hadamard-transformed activation.

    Two kernels, one weight: the M=1 matvec when the traced query length is 1, the tiled GEMM
    otherwise (`s` is a Python int under torch.export, so the branch resolves at trace time and
    each entrypoint holds exactly one kernel). `site` is the shared HadamardSite for this
    activation (memoised, so q/k/v pay for one transform); pass None for an un-rotated input.
    """

    def __init__(self, qp: torch.Tensor, d: torch.Tensor, mv_kernel: TorchMetalKernel,
                 gemm_kernel: TorchMetalKernel | dict | None, site=None) -> None:
        super().__init__()
        self.N, k16 = int(qp.shape[0]), int(qp.shape[1])
        self.K = k16 * 16
        if self.N % (_R * _SGY) or self.K % _KB:
            raise ValueError(f"N={self.N} must be %{_R * _SGY}, K={self.K} must be %{_KB}")
        if self.N % _BN or self.K % _BK:
            raise ValueError(f"N={self.N} must be %{_BN}, K={self.K} must be %{_BK}")
        if tuple(d.shape) != (self.N, self.K // SCALE_BLK):
            raise ValueError(f"scale shape {tuple(d.shape)} != {(self.N, self.K // SCALE_BLK)}")
        self.mv_kernel = mv_kernel
        # one tiled GEMM per static query length the bundle exports (BM is baked into the MSL);
        # a single kernel is accepted for the one-chunk case and keyed by its BM
        if gemm_kernel is None or isinstance(gemm_kernel, dict):
            self.gemm_kernels = gemm_kernel or {}
        else:
            self.gemm_kernels = {int(gemm_kernel.name.rsplit("_m", 1)[1]): gemm_kernel}
        self.register_buffer("qp", qp.contiguous())
        self.register_buffer("d", d.to(torch.float16).contiguous())
        object.__setattr__(self, "_site", site)      # shared, owned by the layer — not a child

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._site is not None:
            x = self._site.transform(x)
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.K)
        s = x2.shape[0]
        if isinstance(s, int) and s != 1:
            if s not in self.gemm_kernels:
                raise RuntimeError(f"no GEMM kernel bound for S={s} (have {sorted(self.gemm_kernels)})")
            y = self.gemm_kernels[s](x2, self.qp, self.d,
                                 threads_per_grid=(32, 8 * (self.N // _BN), 1),
                                 threads_per_thread_group=(32, 8, 1),
                                 result_shapes=[[s, self.N]])
        else:
            y = self.mv_kernel(x2, self.qp, self.d,
                               threads_per_grid=(32, self.N // _R, 1),
                               threads_per_thread_group=(32, _SGY, 1),
                               result_shapes=[[s, self.N]])
        return y.reshape(*lead, self.N)
