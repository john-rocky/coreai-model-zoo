"""s/forward of the exported Qwen-Image-2.1 DiT on the Mac GPU (``SpecializationOptions.default()``).

Per shape (``LxHxW``; default 40x16x16 = 256² and 40x32x32 = 512²): the first call — which pays
for specializing that shape — timed on its own, then ``--warm`` calls -> median (min / max shown).
Inputs are random latents / text features with the real RoPE tables (speed does not depend on
the values). The clock covers the call and reading the output back.

Take the machine-wide GPU lock (other sessions share this GPU):
  python3 ~/code/coreai-kit/scripts/with-gpu-lock.py -- \\
      ~/code/coreai/coreai-models/.venv/bin/python bench_dit.py <bundle.aimodel>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from qi21_host import build_inputs  # noqa: E402

ORDER = ("img_tokens", "txt_feats", "timestep", "txt_cos", "txt_sin", "img_cos", "img_sin")


async def run(args):
    import coreai.runtime as rt
    t0 = time.perf_counter()
    model = await rt.AIModel.load(Path(args.bundle), rt.SpecializationOptions.default())
    fn = model.load_function("main")
    t_load = time.perf_counter() - t0
    print(f"[bench] {Path(args.bundle).name}: load {t_load:.1f}s  loadavg {os.getloadavg()[0]:.1f}", flush=True)
    rows = []
    for spec in args.shapes.split(","):
        L, H, W = (int(v) for v in spec.split("x"))
        g = torch.Generator().manual_seed(0)
        ins = build_inputs(torch.randn(1, H * W, 64, generator=g), torch.randn(1, L, 4096, generator=g),
                           torch.tensor([0.5]), H, W)
        payload = {k: rt.NDArray(ins[k].contiguous()) for k in ORDER}

        async def call():
            t = time.perf_counter()
            r = await fn(payload)
            out = r["vel"].numpy()
            return time.perf_counter() - t, out

        first, out = await call()
        nan = int(np.isnan(np.asarray(out, dtype=np.float32)).sum())
        warm = [(await call())[0] for _ in range(args.warm)]
        row = dict(L=L, H=H, W=W, N=H * W, px=f"{H * 16}x{W * 16}", first_s=first,
                   median_s=statistics.median(warm), min_s=min(warm), max_s=max(warm), warm=warm, nan=nan,
                   loadavg=os.getloadavg()[0])
        rows.append(row)
        print(f"[bench] {row['px']} (N={row['N']}, L={L}): first {first:.2f}s  warm x{args.warm} median "
              f"{row['median_s']:.3f}s (min {row['min_s']:.3f} / max {row['max_s']:.3f})  NaN {nan}  "
              f"loadavg {row['loadavg']:.1f}", flush=True)
    del model
    return t_load, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--shapes", default="40x16x16,40x32x32")
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--json", default=str(HERE / "_work" / "bench_dit.json"))
    args = ap.parse_args()
    t_load, rows = asyncio.run(run(args))
    json.dump(dict(bundle=str(args.bundle), unit="default", load_s=t_load, rows=rows,
                   note="provisional: other sessions share this Mac"), open(args.json, "w"), indent=2)
    print(f"[bench] -> {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
