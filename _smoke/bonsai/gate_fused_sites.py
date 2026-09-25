"""Gate the fused site kernels (`bonsai_fused_sites_metal`) on the GPU against the graph math.

For each of add+norm+transform, norm+transform and swiglu+transform: a one-op graph, exported
with the custom-kernel hook and run through coreai.runtime, compared with (a) the kernel's own
torch reference and (b) the un-fused zoo chain (RMSNormPlusOne + HadamardSite reference / the MLP
formula + the transform reference) it replaces.

    .venv/bin/python _smoke/bonsai/gate_fused_sites.py
"""
from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from coreai_models.models.macos.bonsai_fused_sites_metal import add_norm_fwht, norm_fwht, swiglu_fwht


def report(tag, y, ref, tol):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    err = np.abs(y - ref)
    scale = float(np.abs(ref).max())
    rel = float(err.max()) / max(scale, 1e-9)
    ok = bool(np.isfinite(y).all()) and rel < tol
    print(f"[gate] {tag:44s} max|err| {err.max():.3e}  max|ref| {scale:.3e}  rel {rel:.2e}  "
          f"mean|err| {err.mean():.2e}  {'OK' if ok else 'FAIL'}", flush=True)
    return ok


class Consts(nn.Module):
    def __init__(self, **consts):
        super().__init__()
        for k, v in consts.items():
            self.register_buffer(k, v)


class AddNormGraph(Consts):          # torch.export receives the feed as keyword arguments
    def forward(self, x, r):
        return add_norm_fwht(self.kernel, x, r, self.w, self.sg)


class NormGraph(Consts):
    def forward(self, x):
        return norm_fwht(self.kernel, x, self.w, self.sg)


class SingleGraph(nn.Module):
    def __init__(self, lins):
        super().__init__()
        self.lins = nn.ModuleList(lins)

    def forward(self, x):
        return tuple(l(x) for l in self.lins)


class MultiGraph(nn.Module):
    def __init__(self, lins):
        super().__init__()
        self.lins = nn.ModuleList(lins)

    def forward(self, x):
        from coreai_models.models.macos.bonsai_ternary_metal import ternary_multi
        return tuple(ternary_multi(self.kernel, x, list(self.lins)))


class SwigluGraph(Consts):
    def forward(self, up, gate):
        return swiglu_fwht(self.kernel, up, gate, self.sg)


async def run_device(aimodel: Path, feed: dict, names):
    import coreai.runtime as rt
    m = await rt.AIModel.load(
        str(aimodel), rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    fn = m.load_function("main")
    nd = {k: rt.NDArray(np.ascontiguousarray(v)) for k, v in feed.items()}
    res = await fn(inputs=nd)
    return {k: res[k].numpy() for k in names}


def export_run(graph, feed, kernels, names):
    from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels
    import coreai.runtime as rt
    work = Path(tempfile.mkdtemp(prefix="bonsai_sites_"))
    prog = export_to_coreai_with_kernels(graph.eval(), feed, custom_kernels=kernels,
                                         input_names=tuple(feed), output_names=tuple(names))
    prog.optimize()
    a = work / "g.aimodel"
    prog.save_asset(a, rt.AIModelAssetMetadata())
    out = asyncio.run(run_device(a, {k: v.numpy() for k, v in feed.items()}, names))
    shutil.rmtree(work, ignore_errors=True)
    return out


def main() -> int:
    from coreai_models.models.macos.bonsai_fused_sites_metal import (
        add_norm_fwht_reference, build_add_norm_fwht_kernel, build_norm_fwht_kernel,
        build_swiglu_fwht_kernel, swiglu_fwht_reference,
    )
    from coreai_models.models.macos.bonsai_hadamard_metal import signed_hadamard_reference
    from coreai_models.primitives.macos.rms_norm import RMSNormPlusOne

    torch.manual_seed(0)
    eps = 1e-6
    ok = True
    K = 5120
    x = (torch.randn(1, 1, K) * 2.0).half()
    r = (torch.randn(1, 1, K) * 0.5).half()
    w = (torch.randn(K) * 0.2).half()                 # stored gain (RMSNormPlusOne adds 1)
    sg = (torch.randint(0, 2, (K,)) * 2 - 1).float()
    norm = RMSNormPlusOne(K, eps=eps)
    with torch.no_grad():
        norm.weight.copy_(w)
    norm = norm.half().eval()

    # un-fused zoo chain: h = x + r (fp16), n = RMSNormPlusOne(h), y = signed FWHT reference (fp64)
    with torch.no_grad():
        h_ref = (x + r)
        n_ref = norm(h_ref)
        y_ref = signed_hadamard_reference(n_ref.reshape(1, K).double(), sg).half()
        n0_ref = norm(x)
        y0_ref = signed_hadamard_reference(n0_ref.reshape(1, K).double(), sg).half()

    ka = build_add_norm_fwht_kernel(eps)
    kn = build_norm_fwht_kernel(eps)
    ga = AddNormGraph(w=w, sg=sg)
    ga.kernel = ka
    out = export_run(ga, {"x": x, "r": r}, [ka], ("h", "nrm", "y"))
    with torch.no_grad():
        h_t, n_t, y_t = add_norm_fwht_reference(x.reshape(1, K), r.reshape(1, K), w, sg, eps)
    ok &= report("add_norm_fwht: h vs zoo add", out["h"], h_ref.numpy(), 1e-6)
    ok &= report("add_norm_fwht: nrm vs zoo RMSNormPlusOne", out["nrm"], n_ref.numpy(), 2e-3)
    ok &= report("add_norm_fwht: y vs zoo norm+transform", out["y"], y_ref.numpy(), 2e-3)
    ok &= report("add_norm_fwht: y vs torch reference", out["y"], y_t.numpy(), 1e-3)

    gn = NormGraph(w=w, sg=sg)
    gn.kernel = kn
    out = export_run(gn, {"x": x}, [kn], ("nrm", "y"))
    ok &= report("norm_fwht: nrm vs zoo RMSNormPlusOne", out["nrm"], n0_ref.numpy(), 2e-3)
    ok &= report("norm_fwht: y vs zoo norm+transform", out["y"], y0_ref.numpy(), 2e-3)

    KI = 17408
    up = (torch.randn(1, 1, KI) * 1.0).half()
    gate = (torch.randn(1, 1, KI) * 1.5).half()
    sgi = (torch.randint(0, 2, (KI,)) * 2 - 1).float()
    ks = build_swiglu_fwht_kernel()
    gs = SwigluGraph(sg=sgi)
    gs.kernel = ks
    out = export_run(gs, {"up": up, "gate": gate}, [ks], ("y",))
    with torch.no_grad():
        p_ref = up * torch.nn.functional.silu(gate)              # the zoo MLP formula in fp16
        ys_ref = signed_hadamard_reference(p_ref.reshape(1, KI).double(), sgi).half()
        ys_t = swiglu_fwht_reference(up.reshape(1, KI), gate.reshape(1, KI), sgi)
    ok &= report("swiglu_fwht: y vs zoo swiglu+transform", out["y"], ys_ref.numpy(), 2e-3)
    ok &= report("swiglu_fwht: y vs torch reference", out["y"], ys_t.numpy(), 1e-3)
    # multi matvec: bit-identical to the single matvec on each weight set
    from coreai_models.models.macos.bonsai_ternary_metal import (
        TernaryLinear128, build_mv_kernel, build_mv_multi_kernel, ternary_multi,
    )
    mv = build_mv_kernel()
    lins, refs = [], []
    xk = (torch.randn(1, 1, 1024) * 0.5).half()
    for n in (256, 512, 128):
        qp = torch.randint(0, 2**31 - 1, (n, 1024 // 16), dtype=torch.int32)
        d = (torch.rand(n, 1024 // 128) * 0.02 + 0.001).half()
        lins.append(TernaryLinear128(qp, d, mv, None, site=None))
    single = SingleGraph(lins)
    single.kernel = mv
    ref = export_run(single, {"x": xk}, [mv], ("y1", "y2", "y3"))
    for nsets in (2, 3):
        km = build_mv_multi_kernel(nsets)
        multi = MultiGraph(lins[:nsets])
        multi.kernel = km
        out = export_run(multi, {"x": xk}, [km], tuple(f"y{i + 1}" for i in range(nsets)))
        for i in range(nsets):
            ok &= report(f"mv_x{nsets}: C{i + 1} vs single matvec", out[f"y{i + 1}"], ref[f"y{i + 1}"], 1e-6)
    print(f"[gate] {'ALL OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
