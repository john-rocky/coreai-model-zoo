"""Fused S=1 gated-delta-net step for Core AI — the whole Bonsai GDN block between the projections
in ONE dispatch.

At S=1 the zoo's GDN block is ~30 small graph ops per layer around the chunk-scan kernel: the
causal conv step (cat, conv, slice for the new state), SiLU, the q/k/v split with the GVA head
repeat, two l2-norms, the q scale, sigmoid/softplus for beta and the decay, the scan kernel with
its contiguous copies, the output slice/transpose, and the gated RMSNorm. Each is a few
microseconds of GPU work behind ~10 µs of dispatch gap; over 42 linear layers that is most of the
non-matvec time in a decode token (knowledge/bonsai-ternary-hadamard.md §8.7). This kernel does
all of it (plus the two dense a/b projections) for one value head per threadgroup, one thread per
value column, and returns the three
things the layer needs: the normalized, gated output (ready for the out_proj Hadamard site) and
the two new states.

Numerics mirror the graph path it replaces: fp32 arithmetic with an fp16 rounding wherever the
graph materializes an fp16 tensor (conv output, SiLU output, normalized q/k, the scaled q, beta,
the scan output, the normalized output, the gated output) and the fp32 recurrence of the chunk
kernel. The torch reference below is the same chain in torch, so it doubles as the CPU
implementation under torch.export and as the gate reference (`_smoke/bonsai/gate_gdn_step.py`).

Layout (MSL sees torch shapes reversed):
  MIXED torch [1, conv_dim] f16   -> MIXED[ch, 0]   in_proj_qkv output: [q (nk*dk) | k (nk*dk) | v (nv*dv)]
  Z     torch [1, nv*dv]    f16   -> Z[i, 0]        in_proj_z output (HF grouped head order)
  NRM   torch [1, d_model]  f16   -> NRM[i, 0]      the layer's normalized input BEFORE the Hadamard
                                                    rotation (the in_proj_a / in_proj_b operand)
  WA, WB torch [nv, d_model] f16  -> WA[i, h]       in_proj_a / in_proj_b weights (the two dense
                                                    [d_model -> nv] projections are done in here)
  CW    torch [conv_dim, kw] f16  -> CW[j, ch]      conv1d weight, kw = 4 (taps over [s0, s1, s2, x])
  ALOG, DTB torch [nv] f32        -> ALOG[h]      (fp32, as the pack stores them)
  NW    torch [dv] f16            -> NW[c]          gated RMSNorm weight
  CS    torch [conv_dim, kw-1] f16 -> CS[j, ch]     conv state (the last kw-1 inputs)
  RS    torch [nv, dk, dv] f16    -> RS[c, d, h]    recurrent state
  Y     torch [1, nv*dv] f16      -> Y[i, 0]
  CSN   torch [conv_dim, kw-1] f16
  RSN   torch [nv, dk, dv] f16
Grid: threads_per_grid (dv, nv, 1), threads_per_thread_group (dv, 1, 1): gid.x = value column,
gid.y = value head. Requires dv == dk == 128 (this family) and nv % nk == 0.

(A variant with 8 heads per 1024-thread threadgroup, the state re-read in two passes instead of
held in 128 registers, and the out_proj site's signed FWHT as its epilogue was measured at
44-51 µs in situ against 33 for this one: one kernel call fewer per layer was not worth it.)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_torch import MetalParameter, TorchMetalKernel

_GDN_STEP_SRC = """
    const uint DK = 128u, DV = 128u;
    const uint NV = RS.get_extent(2);            // torch [nv, dk, dv] -> extents (dv, dk, nv)
    const uint CONV = CW.get_extent(1);          // torch [conv_dim, kw] -> extents (kw, conv_dim)
    const uint KW = CW.get_extent(0);
    const uint KEYD = (CONV - NV * DV) / 2u;      // nk * dk
    const uint NK = KEYD / DK;
    const uint REP = NV / NK;                     // value heads per key head (GVA)
    const uint c = gid.x;                         // value column / state column
    const uint hh = gid.y;                        // value head
    const uint kh = hh / REP;                     // its key head
    const uint cq = kh * DK + c;
    const uint ck = KEYD + kh * DK + c;
    const uint cv = 2u * KEYD + hh * DV + c;

    threadgroup float qsh[128];
    threadgroup float ksh[128];
    threadgroup float osh[128];

    // causal conv step + SiLU for this thread's q, k and v channel (q/k recomputed by the REP
    // value heads that share the key head: 256 MACs, cheaper than a barrier across threadgroups)
    uint chs[3] = {cq, ck, cv};
    float act[3];
    for (uint i = 0; i < 3u; ++i) {
        const uint ch = chs[i];
        float acc = 0.0f;
        for (uint j = 0; j + 1u < KW; ++j) acc += float(CW[j, ch]) * float(CS[j, ch]);
        acc += float(CW[KW - 1u, ch]) * float(MIXED[ch, 0]);
        const float cf = float(half(acc));                       // conv output is an fp16 tensor
        act[i] = float(half(cf / (1.0f + exp(-cf))));             // so is silu(conv)
    }
    // new conv state = the last kw-1 inputs; q/k rows written once per key head
    {
        const bool own_qk = (hh % REP) == 0u;
        for (uint j = 0; j + 2u < KW; ++j) {
            CSN[j, cv] = CS[j + 1u, cv];
            if (own_qk) { CSN[j, cq] = CS[j + 1u, cq]; CSN[j, ck] = CS[j + 1u, ck]; }
        }
        CSN[KW - 2u, cv] = MIXED[cv, 0];
        if (own_qk) { CSN[KW - 2u, cq] = MIXED[cq, 0]; CSN[KW - 2u, ck] = MIXED[ck, 0]; }
    }

    // a = in_proj_a(nrm), b = in_proj_b(nrm) for this head: 128 threads over d_model, fp32
    // accumulation, rounded to fp16 like the graph's dense projection outputs
    const uint DM = NRM.get_extent(0);
    float pa = 0.0f, pb = 0.0f;
    for (uint i = c; i < DM; i += DV) {
        const float xi = float(NRM[i, 0]);
        pa += xi * float(WA[i, hh]);
        pb += xi * float(WB[i, hh]);
    }
    pa = simd_sum(pa);
    pb = simd_sum(pb);
    if ((c & 31u) == 0u) { qsh[c >> 5] = pa; ksh[c >> 5] = pb; }   // 4 simdgroups; qsh/ksh are free here
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float a16 = float(half(qsh[0] + qsh[1] + qsh[2] + qsh[3]));
    const float b16 = float(half(ksh[0] + ksh[1] + ksh[2] + ksh[3]));
    threadgroup_barrier(mem_flags::mem_threadgroup);              // before qsh/ksh are reused

    // l2-norm of q and k over the head (fp16 tensors in the graph), q scaled by dk^-1/2
    qsh[c] = act[0];
    ksh[c] = act[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sq = 0.0f, sk = 0.0f;
    for (uint d = 0; d < DK; ++d) { sq += qsh[d] * qsh[d]; sk += ksh[d] * ksh[d]; }
    float qn = float(half(act[0] * rsqrt(sq + 1e-6f)));
    qn = float(half(qn * 0.08838834764831845f));                 // 128^-0.5
    const float kn = float(half(act[1] * rsqrt(sk + 1e-6f)));
    threadgroup_barrier(mem_flags::mem_threadgroup);              // everyone is done reading the raw q/k
    qsh[c] = qn;
    ksh[c] = kn;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // per-head gate and decay (fp32, as the graph computes them from the fp16 projections)
    const float ap = a16 + float(DTB[hh]);
    const float sp = ap > 20.0f ? ap : log(1.0f + exp(ap));      // F.softplus, threshold 20 (no log1p in MSL)
    const float ge = exp(-exp(float(ALOG[hh])) * sp);            // exp(g), g = -exp(A_log) * softplus
    const float bt = float(half(1.0f / (1.0f + exp(-b16))));      // sigmoid -> fp16 tensor

    // one gated-delta step on this thread's state column (fp32, as the chunk kernel)
    float st[128];
    for (uint d = 0; d < DK; ++d) st[d] = float(RS[c, d, hh]);
    float kv = 0.0f;
    for (uint d = 0; d < DK; ++d) { st[d] *= ge; kv += st[d] * ksh[d]; }
    const float delta = (act[2] - kv) * bt;
    float oc = 0.0f;
    for (uint d = 0; d < DK; ++d) { st[d] += ksh[d] * delta; oc += st[d] * qsh[d]; }
    for (uint d = 0; d < DK; ++d) RSN[c, d, hh] = half(st[d]);
    const float och = float(half(oc));                            // the scan output is fp16

    // gated RMSNorm over the head: half(half(x * rsqrt(mean(x^2) + eps)) * w) * silu(z) -> half
    osh[c] = och;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ms = 0.0f;
    for (uint d = 0; d < DV; ++d) ms += osh[d] * osh[d];
    ms *= (1.0f / 128.0f);
    float n = float(half(och * rsqrt(ms + __EPS__)));
    n = float(half(n * float(NW[c])));
    const float zz = float(Z[hh * DV + c, 0]);
    Y[hh * DV + c, 0] = half(n * (zz / (1.0f + exp(-zz))));
"""


def _round16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).to(torch.float32)


def gdn_step_reference(MIXED: torch.Tensor, Z: torch.Tensor, NRM: torch.Tensor, WA: torch.Tensor,
                       WB: torch.Tensor, CW: torch.Tensor, ALOG: torch.Tensor, DTB: torch.Tensor,
                       NW: torch.Tensor, CS: torch.Tensor, RS: torch.Tensor,
                       eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The kernel's math in torch: fp32 with the graph's fp16 roundings. Returns (Y, CSN, RSN)."""
    conv_dim, kw = CW.shape
    nv, dk, dv = RS.shape
    xf = NRM.reshape(-1).to(torch.float32)
    A = _round16(WA.to(torch.float32) @ xf)                                     # [nv] in_proj_a(nrm)
    B = _round16(WB.to(torch.float32) @ xf)
    key_dim = (conv_dim - nv * dv) // 2
    nk = key_dim // dk
    x = MIXED.reshape(conv_dim).to(torch.float32)
    w = torch.cat([CS.to(torch.float32), x[:, None]], dim=1)                    # [conv_dim, kw]
    conv = _round16((w * CW.to(torch.float32)).sum(1))
    act = _round16(F.silu(conv))
    csn = w[:, 1:].to(CS.dtype)
    q = act[:key_dim].reshape(nk, dk)
    k = act[key_dim:2 * key_dim].reshape(nk, dk)
    v = act[2 * key_dim:].reshape(nv, dv)
    qn = _round16(q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6))
    qn = _round16(qn * (dk ** -0.5))
    kn = _round16(k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6))
    rep = nv // nk
    qn = qn.repeat_interleave(rep, 0)
    kn = kn.repeat_interleave(rep, 0)
    g = -ALOG.to(torch.float32).exp() * F.softplus(A.reshape(nv).to(torch.float32) + DTB.to(torch.float32))
    ge = g.exp()
    beta = _round16(torch.sigmoid(B.reshape(nv).to(torch.float32)))
    st = RS.to(torch.float32) * ge[:, None, None]
    kv = (st * kn[:, :, None]).sum(1)                                            # [nv, dv]
    delta = (v - kv) * beta[:, None]
    st = st + kn[:, :, None] * delta[:, None, :]
    oc = _round16((st * qn[:, :, None]).sum(1))                                  # [nv, dv]
    ms = (oc * oc).mean(-1, keepdim=True)
    n = _round16(oc * torch.rsqrt(ms + eps))
    n = _round16(n * NW.to(torch.float32))
    zz = Z.reshape(nv, dv).to(torch.float32)
    y = (n * F.silu(zz)).to(MIXED.dtype)
    return y.reshape(1, nv * dv), csn, st.to(RS.dtype)


def build_gdn_step_kernel(eps: float, name: str = "bonsai_gdn_step") -> TorchMetalKernel:
    def _torch_defn(MIXED: torch.Tensor, Z: torch.Tensor, NRM: torch.Tensor, WA: torch.Tensor,
                    WB: torch.Tensor, CW: torch.Tensor, ALOG: torch.Tensor, DTB: torch.Tensor,
                    NW: torch.Tensor, CS: torch.Tensor,
                    RS: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return gdn_step_reference(MIXED, Z, NRM, WA, WB, CW, ALOG, DTB, NW, CS, RS, eps)

    return TorchMetalKernel(
        name, input_names=["MIXED", "Z", "NRM", "WA", "WB", "CW", "ALOG", "DTB", "NW", "CS", "RS"],
        result_names=["Y", "CSN", "RSN"], src=_GDN_STEP_SRC.replace("__EPS__", f"{eps!r}f"),
        torch_defn=_torch_defn,
        metal_params=[MetalParameter("gid", "uint2", "thread_position_in_grid")],
        template_dtypes={"MIXED": "TYPE"})


def fused_gdn_step(layer: nn.Module, kernel: TorchMetalKernel, x: torch.Tensor,
                   conv_in: torch.Tensor, rec_in: torch.Tensor, nrm: torch.Tensor | None = None,
                   mv2: TorchMetalKernel | None = None):
    """The S=1 stateful GDN forward of a `Qwen3_5GatedDeltaNet` with the block between the
    projections replaced by the fused kernel. Same signature/returns as the layer's own forward.
    `x` feeds the ternary qkv/z projections (through their Hadamard site); `nrm` is the
    un-rotated normalized input for the dense a/b projections (defaults to `x`, correct when the
    layer has no Hadamard sites, as in the kernel gate)."""
    b, s, _ = x.shape
    assert b == 1 and s == 1, "fused GDN step is the S=1 decode path"
    nv, dk, dv, conv_dim, kw = layer.num_v, layer.dk, layer.dv, layer.conv_dim, layer.kernel
    if mv2 is not None:                       # the two ternary projections in one dispatch
        from coreai_models.models.macos.bonsai_ternary_metal import ternary_multi
        mixed, z = ternary_multi(mv2, x, [layer.in_proj_qkv, layer.in_proj_z])
    else:
        mixed, z = layer.in_proj_qkv(x), layer.in_proj_z(x)
    mixed = mixed.reshape(1, conv_dim)
    z = z.reshape(1, layer.value_dim)
    nrm2 = (x if nrm is None else nrm).reshape(1, -1)
    cw = layer.conv1d.weight.reshape(conv_dim, kw)
    y, csn, rsn = kernel(
        mixed, z, nrm2, layer.in_proj_a.weight, layer.in_proj_b.weight, cw,
        layer.A_log, layer.dt_bias, layer.norm.weight,
        conv_in.reshape(conv_dim, kw - 1), rec_in.reshape(nv, dk, dv),
        threads_per_grid=(dv, nv, 1), threads_per_thread_group=(dv, 1, 1),
        result_shapes=[[1, nv * dv], [conv_dim, kw - 1], [nv, dk, dv]])
    out = layer.out_proj(y.reshape(1, 1, layer.value_dim))      # the out_proj site transforms y
    return out, csn.reshape(1, conv_dim, kw - 1), rsn.reshape(1, nv, dk, dv)
