"""Diagnostic: what an fp32-compute text encoder would cost on Core AI, measured on a random-weight probe.

Same probe for every variant (first ``--layers`` decoder layers + the full 151936 x 4096 embedding,
random weights, dynamic L 16..512, ids in / fp32 hidden out), exported -> AOT (h16c, gpu,
--expect-frequent-reshapes) -> ``SpecializationOptions.default()``:

  bf16r32  bf16 weights, bf16 matmuls, fp32 residual   (the exported candidate)
  fp32     fp32 weights, fp32 compute
  w16a32   bf16 weights stored, every Linear / norm computed in fp32 (upcast in the graph) — does
           the converter keep the bf16 constants and cast at run time, or fold them to fp32?

Per variant: .aimodel / .aimodelc size, corr + max|d| vs the same module in fp32 torch (L=32 and
128), s/call (first call, then median of ``--warm``). Random weights have no massive activations,
so this measures cost and compute precision, not the per-token problem itself.

Run (base venv; GPU lock):
  python3 ~/code/coreai-kit/scripts/with-gpu-lock.py -- ~/code/coreai/coreai-models/.venv/bin/python probe_fp32_cost.py
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import gc
import json
import shutil
import statistics
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from export_encoder import DEFAULT_TEXT_CFG, aot_compile, free_gib, paths, randomize  # noqa: E402
from qi21_text import QI21TextEncoder, RMSNorm  # noqa: E402


def _lin32(self, x):
    return F.linear(x.float(), self.weight.float())


def _norm32(self, x):
    xf = x.float()
    return self.weight.float() * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps))


def make(variant: str, layers: int, seed: int):
    cfg = dict(DEFAULT_TEXT_CFG, num_hidden_layers=layers)
    m = QI21TextEncoder.from_config(cfg, io_fp32=True, residual_fp32=True)
    randomize(m, seed)
    m = m.to(torch.float32 if variant == "fp32" else torch.bfloat16).eval()
    if variant == "w16a32":
        for mod in m.modules():
            if isinstance(mod, nn.Linear):
                mod.forward = types.MethodType(_lin32, mod)
            elif isinstance(mod, RMSNorm):
                mod.forward = types.MethodType(_norm32, mod)
    return m


def size_gib(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2**30


async def engine(aimodelc: Path, ref_model, Ls, warm: int):
    import coreai.runtime as rt
    t0 = time.perf_counter()
    model = await rt.AIModel.load(aimodelc, rt.SpecializationOptions.default())
    fn = model.load_function("main")
    t_load = time.perf_counter() - t0
    rows = []
    for L in Ls:
        g = torch.Generator().manual_seed(100 + L)
        ids = torch.randint(0, 151643, (1, L), generator=g, dtype=torch.int32)
        payload = {"input_ids": rt.NDArray(ids.contiguous())}

        async def call():
            t = time.perf_counter()
            r = await fn(payload)
            out = np.array(r["hidden"].numpy(), dtype=np.float32)
            return time.perf_counter() - t, out

        first, out = await call()
        times = [(await call())[0] for _ in range(warm)]
        with torch.no_grad():
            ref = ref_model(ids).numpy()
        a, b = out.astype(np.float64).ravel(), ref.astype(np.float64).ravel()
        rows.append(dict(L=L, first_s=first, median_s=statistics.median(times), min_s=min(times),
                         corr=float(np.corrcoef(a, b)[0, 1]), maxd=float(np.abs(a - b).max()),
                         rel=float(np.linalg.norm(a - b) / np.linalg.norm(b)), nan=int(np.isnan(out).sum())))
    del fn, model
    return t_load, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--variants", default="bf16r32,fp32,w16a32")
    ap.add_argument("--L", default="32,128")
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from torch.export import Dim
    from coreai_models.export.macos import export_to_coreai
    import coreai.runtime as rt

    Ls = [int(v) for v in args.L.split(",")]
    res = {}
    for variant in args.variants.split(","):
        if free_gib() < 60:
            print(f"[cost] STOP: disk free {free_gib():.1f} GiB", flush=True)
            return 2
        name = f"qi21_encoder_L{args.layers}_dynL_{variant}_costprobe_ids_iofp32"
        bundle, aimodelc = paths(name)
        m = make(variant, args.layers, args.seed)
        ref32 = copy.deepcopy(m)
        if variant != "fp32":
            ref32 = make("fp32", args.layers, args.seed)          # same random weights, fp32 torch
            ref32.load_state_dict({k: v.float() for k, v in m.state_dict().items()})
        ids = torch.randint(0, 151643, (1, 64), generator=torch.Generator().manual_seed(1), dtype=torch.int32)
        t0 = time.time()
        prog = export_to_coreai(m, {"input_ids": ids}, dynamic_shapes={"input_ids": {1: Dim("L", min=16, max=512)}},
                                input_names=("input_ids",), output_names=("hidden",))
        prog.optimize()
        shutil.rmtree(bundle.parent, ignore_errors=True)
        bundle.parent.mkdir(parents=True)
        meta = rt.AIModelAssetMetadata()
        meta.license = "qwen-research"
        meta.model_description = f"cost probe ({variant}, {args.layers} random layers) — diagnostic, not for use"
        prog.save_asset(bundle, meta)
        t_export = time.time() - t0
        del prog
        gc.collect()
        t_aot = aot_compile(bundle, aimodelc)
        t_load, rows = asyncio.run(engine(aimodelc, ref32, Ls, args.warm))
        res[variant] = dict(aimodel_gib=size_gib(bundle), aimodelc_gib=size_gib(aimodelc), export_s=t_export,
                            aot_s=t_aot, load_s=t_load, rows=rows)
        print(f"[cost] {variant:<8} .aimodel {res[variant]['aimodel_gib']:.2f} GiB  .aimodelc "
              f"{res[variant]['aimodelc_gib']:.2f} GiB  load {t_load:.1f}s  " + "  ".join(
                  f"L={r['L']}: first {r['first_s']:.3f}s median {r['median_s']:.4f}s corr {r['corr']:.7f} "
                  f"|d|/|ref| {r['rel']:.2e} NaN {r['nan']}" for r in rows), flush=True)
        del m, ref32
        gc.collect()
    json.dump(res, open(HERE / "_work" / f"probe_fp32_cost_L{args.layers}.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
