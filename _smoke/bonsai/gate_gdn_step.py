"""Gate the fused S=1 GDN step kernel (`bonsai_gdn_step_metal`) two ways.

1. torch vs torch: the kernel's reference math (`gdn_step_reference`, also its CPU implementation
   under torch.export) against the zoo's own GDN layer forward at S=1 (`use_loopfree_step`) on the
   same random weights and states. Isolates the fused block's rounding chain from the projections.
2. MSL vs torch: a one-op graph around the kernel, exported and run on the GPU through
   coreai.runtime, against the reference on the same inputs.

    .venv/bin/python _smoke/bonsai/gate_gdn_step.py
"""
from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def report(tag, y, ref, tol):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    err = np.abs(y - ref)
    scale = float(np.abs(ref).max())
    rel = float(err.max()) / max(scale, 1e-9)
    ok = bool(np.isfinite(y).all()) and rel < tol
    print(f"[gate] {tag:40s} max|err| {err.max():.3e}  max|ref| {scale:.3e}  rel {rel:.2e}  "
          f"mean|err| {err.mean():.2e}  {'OK' if ok else 'FAIL'}", flush=True)
    return ok


class KernelOnly(nn.Module):
    """Graph = the kernel call with the weights as constants and the activations/states as inputs."""

    def __init__(self, kernel, wa, wb, cw, alog, dtb, nw, nv, dk, dv, conv_dim, kw):
        super().__init__()
        self.kernel = kernel
        self.register_buffer("wa", wa)
        self.register_buffer("wb", wb)
        self.register_buffer("cw", cw)
        self.register_buffer("alog", alog)
        self.register_buffer("dtb", dtb)
        self.register_buffer("nw", nw)
        self.dims = (nv, dk, dv, conv_dim, kw)

    def forward(self, mixed, z, nrm, cs, rs):
        nv, dk, dv, conv_dim, kw = self.dims
        return self.kernel(mixed, z, nrm, self.wa, self.wb, self.cw, self.alog, self.dtb, self.nw, cs, rs,
                           threads_per_grid=(dv, nv, 1), threads_per_thread_group=(dv, 1, 1),
                           result_shapes=[[1, nv * dv], [conv_dim, kw - 1], [nv, dk, dv]])


async def run_device(aimodel: Path, feed: dict, reps: int = 50):
    import coreai.runtime as rt
    m = await rt.AIModel.load(
        str(aimodel), rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    fn = m.load_function("main")
    nd = {k: rt.NDArray(np.ascontiguousarray(v)) for k, v in feed.items()}
    res = await fn(inputs=nd)
    out = {k: res[k].numpy() for k in ("y", "csn", "rsn")}
    t0 = time.perf_counter()
    for _ in range(reps):
        await fn(inputs=nd)
    return out, (time.perf_counter() - t0) / reps


def main() -> int:
    import gguf
    from huggingface_hub import hf_hub_download
    from coreai_models.models.macos.bonsai2 import BonsaiGDN, config_from_gguf
    from coreai_models.models.macos.bonsai_gdn_step_metal import build_gdn_step_kernel, gdn_step_reference
    from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels
    from coreai_models.models.macos.qwen3_5 import Qwen3_5GatedDeltaNet
    import coreai.runtime as rt

    path = hf_hub_download("prism-ml/Ternary-Bonsai-2-27B-gguf", "Ternary-Bonsai-2-27B-PQ2_0.gguf")
    cfg = config_from_gguf(gguf.GGUFReader(path), 4)
    torch.manual_seed(0)
    ok = True

    # random but realistically scaled layer
    layer = Qwen3_5GatedDeltaNet(cfg)
    with torch.no_grad():
        for lin in (layer.in_proj_qkv, layer.in_proj_z, layer.in_proj_a, layer.in_proj_b, layer.out_proj):
            lin.weight.normal_(0, 0.02)
        layer.conv1d.weight.normal_(0, 0.5)
        layer.A_log.normal_(0, 1)
        layer.dt_bias.normal_(0, 1)
        layer.norm.weight.normal_(1, 0.1)
    layer = layer.half().eval()
    with torch.no_grad():                       # the loader keeps these two fp32 (bonsai2.py)
        layer.A_log = torch.nn.Parameter(layer.A_log.float(), requires_grad=False)
        layer.dt_bias = torch.nn.Parameter(layer.dt_bias.float(), requires_grad=False)
    nv, nk, dk, dv = layer.num_v, layer.num_k, layer.dk, layer.dv
    conv_dim, kw = layer.conv_dim, layer.kernel
    x = (torch.randn(1, 1, cfg.hidden_size) * 1.0).half()
    conv_in = torch.randn(1, conv_dim, kw - 1).half()
    rec_in = (torch.randn(1, nv, dk, dv) * 0.3).half()

    # 1) zoo layer (loop-free step, in-graph ops) vs the fused reference through the same layer
    with torch.no_grad():
        layer.use_loopfree_step, layer.use_metal_chunk = True, False
        out_ref, cs_ref, rs_ref = layer(x, conv_in, rec_in)
        kernel = build_gdn_step_kernel(cfg.rms_norm_eps)
        layer.__class__ = BonsaiGDN
        layer.step_kernel = kernel
        layer.use_fused_step = True
        out_new, cs_new, rs_new = layer(x, conv_in, rec_in)
    ok &= report("torch: out_proj output (fused vs zoo)", out_new, out_ref, 2e-2)
    ok &= report("torch: new conv state", cs_new, cs_ref, 1e-6)
    ok &= report("torch: new rec state", rs_new, rs_ref, 5e-3)

    # 2) the kernel alone on the GPU vs its reference
    with torch.no_grad():
        mixed = layer.in_proj_qkv(x).reshape(1, conv_dim)
        z = layer.in_proj_z(x).reshape(1, nv * dv)
        nrm = x.reshape(1, -1).contiguous()
        wa, wb = layer.in_proj_a.weight.data.contiguous(), layer.in_proj_b.weight.data.contiguous()
        cw = layer.conv1d.weight.reshape(conv_dim, kw).contiguous()
        cs = conv_in.reshape(conv_dim, kw - 1).contiguous()
        rs = rec_in.reshape(nv, dk, dv).contiguous()
        y_ref, csn_ref, rsn_ref = gdn_step_reference(mixed, z, nrm, wa, wb, cw, layer.A_log, layer.dt_bias,
                                                     layer.norm.weight, cs, rs, cfg.rms_norm_eps)
    feed = {"mixed": mixed, "z": z, "nrm": nrm, "cs": cs, "rs": rs}
    graph = KernelOnly(kernel, wa, wb, cw, layer.A_log.data, layer.dt_bias.data, layer.norm.weight.data,
                       nv, dk, dv, conv_dim, kw).eval()
    work = Path(tempfile.mkdtemp(prefix="bonsai_gdnstep_"))
    prog = export_to_coreai_with_kernels(graph, feed, custom_kernels=[kernel],
                                         input_names=tuple(feed), output_names=("y", "csn", "rsn"))
    prog.optimize()
    aimodel = work / "gdn_step.aimodel"
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    out, dt = asyncio.run(run_device(aimodel, {k: v.numpy() for k, v in feed.items()}))
    shutil.rmtree(work, ignore_errors=True)
    ok &= report("device: Y (gated, normalized output)", out["y"], y_ref.numpy(), 2e-3)
    ok &= report("device: new conv state", out["csn"], csn_ref.numpy(), 1e-6)
    ok &= report("device: new rec state", out["rsn"], rsn_ref.numpy(), 2e-3)
    print(f"[gate] kernel call {dt * 1e3:.3f} ms (one-op graph, includes runtime overhead)")
    print(f"[gate] {'ALL OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
