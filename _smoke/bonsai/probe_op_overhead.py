"""Per-op cost of the Core AI runtime at decode granularity, measured as a slope.

Graphs of N chained tiny ops on a [1, 1024] fp16 vector, timed on the GPU through
coreai.runtime for several N; the slope of time vs N is the cost of one op boundary as the
runtime really executes it (dispatch gap + the converter's copy-in/copy-out around custom
kernels + the op's own few microseconds). The intercept is the per-call fixed cost.

  kernel   : N x bonsai_fwht1024 (a custom Metal kernel; 2 inputs, 1 result)
  matvec   : N x bonsai_tern_mv128 on [1,512] x [512,512] (square, so the calls chain) (3 inputs, 2 of them constants)
  graph    : N x (x * (sum(x) * 1e-9 + 1)): a reduce + a broadcast multiply per step, which the
             runtime cannot fuse into its neighbours (the reduce is a barrier)
  eltwise  : N x (x * c + d) with distinct constants: a chain the runtime is free to fuse

    .venv/bin/python _smoke/bonsai/probe_op_overhead.py [--ns 1,8,32] [--reps 100]
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class KernelChain(nn.Module):
    def __init__(self, n, kernel, signs):
        super().__init__()
        self.n, self.kernel = n, kernel
        self.register_buffer("sg", signs)

    def forward(self, x):
        from coreai_models.models.macos.bonsai_hadamard_metal import fwht_call
        for _ in range(self.n):
            x = fwht_call(self.kernel, x, self.sg)
        return x


class MatvecChain(nn.Module):
    def __init__(self, n, mv, qp, d):
        super().__init__()
        from coreai_models.models.macos.bonsai_ternary_metal import TernaryLinear128
        self.lins = nn.ModuleList([TernaryLinear128(qp, d, mv, None, site=None) for _ in range(n)])

    def forward(self, x):
        for lin in self.lins:
            x = lin(x)
        return x


class GraphChain(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        for _ in range(self.n):
            x = x * (x.sum(-1, keepdim=True) * 1e-9 + 1.0)
        return x


class EltwiseChain(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("c", (1.0 + torch.arange(n).float() * 1e-4).half())
        self.register_buffer("d", (torch.arange(n).float() * 1e-3).half())
        self.n = n

    def forward(self, x):
        for i in range(self.n):
            x = x * self.c[i] + self.d[i]
        return x


async def time_graph(aimodel: Path, feed: dict, reps: int):
    import coreai.runtime as rt
    m = await rt.AIModel.load(
        str(aimodel), rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    fn = m.load_function("main")
    nd = {k: rt.NDArray(np.ascontiguousarray(v)) for k, v in feed.items()}
    for _ in range(10):
        await fn(inputs=nd)
    t0 = time.perf_counter()
    for _ in range(reps):
        await fn(inputs=nd)
    return (time.perf_counter() - t0) / reps


def measure(model, feed, kernels, reps):
    from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels
    import coreai.runtime as rt
    work = Path(tempfile.mkdtemp(prefix="bonsai_probe_"))
    prog = export_to_coreai_with_kernels(model.eval(), feed, custom_kernels=kernels,
                                         input_names=tuple(feed), output_names=("y",))
    prog.optimize()
    a = work / "g.aimodel"
    prog.save_asset(a, rt.AIModelAssetMetadata())
    dt = asyncio.run(time_graph(a, {k: v.numpy() for k, v in feed.items()}, reps))
    shutil.rmtree(work, ignore_errors=True)
    return dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="1,8,32")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--which", default="kernel,matvec,graph,eltwise")
    args = ap.parse_args()
    ns = [int(v) for v in args.ns.split(",")]
    from coreai_models.models.macos.bonsai_hadamard_metal import build_fwht_kernel
    from coreai_models.models.macos.bonsai_ternary_metal import build_mv_kernel
    torch.manual_seed(0)
    fwht, mv = build_fwht_kernel(), build_mv_kernel()
    signs = (torch.randint(0, 2, (1024,)) * 2 - 1).float()
    qp = torch.randint(0, 2**31 - 1, (512, 512 // 16), dtype=torch.int32)
    d = (torch.rand(512, 512 // 128) * 0.01).half()
    x1024 = (torch.randn(1, 1024) * 0.5).half()
    x512 = (torch.randn(1, 1, 512) * 0.5).half()
    builders = {
        "kernel": lambda n: (KernelChain(n, fwht, signs), {"x": x1024}, [fwht]),
        "matvec": lambda n: (MatvecChain(n, mv, qp, d), {"x": x512}, [mv]),
        "graph": lambda n: (GraphChain(n), {"x": x1024}, []),
        "eltwise": lambda n: (EltwiseChain(n), {"x": x1024}, []),
    }
    for which in args.which.split(","):
        pts = []
        for n in ns:
            model, feed, kernels = builders[which](n)
            dt = measure(model, feed, kernels, args.reps)
            pts.append((n, dt))
            print(f"[probe] {which:8s} N={n:3d}: {dt * 1e3:.3f} ms/call", flush=True)
        (n0, t0), (n1, t1) = pts[0], pts[-1]
        slope = (t1 - t0) / (n1 - n0)
        print(f"[probe] {which:8s} slope {slope * 1e6:.1f} us/op, intercept {(t0 - slope * n0) * 1e3:.3f} ms", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
