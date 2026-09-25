"""s/call of the exported Qwen-Image-2.1 text encoder on the Mac GPU (AOT bundle, ``SpecializationOptions.default()``).

Per L (default 32 and 128): the first call — which pays for that shape's specialization — timed on
its own, then ``--warm`` calls -> median (min / max shown). Inputs are random token ids (speed does
not depend on the values). The clock covers the call and reading the ``hidden`` output back.

Take the machine-wide GPU lock (other sessions share this GPU):
  python3 ~/code/coreai-kit/scripts/with-gpu-lock.py -- \\
      ~/code/coreai/coreai-models/.venv/bin/python bench_encoder.py <name>.h16c.aimodelc
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


async def run(args):
    import coreai.runtime as rt
    t0 = time.perf_counter()
    model = await rt.AIModel.load(Path(args.bundle), rt.SpecializationOptions.default())
    fn = model.load_function("main")
    t_load = time.perf_counter() - t0
    print(f"[bench] {Path(args.bundle).name}: load {t_load:.1f}s  loadavg {os.getloadavg()[0]:.1f}", flush=True)
    rows = []
    for L in (int(v) for v in args.L.split(",")):
        g = torch.Generator().manual_seed(L)
        ids = torch.randint(0, 151643, (1, L), generator=g, dtype=torch.int32)
        payload = {"input_ids": rt.NDArray(ids.contiguous())}

        async def call():
            t = time.perf_counter()
            r = await fn(payload)
            out = r["hidden"].numpy()
            return time.perf_counter() - t, out

        first, out = await call()
        nan = int(np.isnan(np.asarray(out, dtype=np.float32)).sum())
        warm = [(await call())[0] for _ in range(args.warm)]
        row = dict(L=L, first_s=first, median_s=statistics.median(warm), min_s=min(warm), max_s=max(warm),
                   warm=warm, nan=nan, shape=list(np.asarray(out).shape), loadavg=os.getloadavg()[0])
        rows.append(row)
        print(f"[bench] L={L}: first {first:.3f}s  warm x{args.warm} median {row['median_s']:.4f}s "
              f"(min {row['min_s']:.4f} / max {row['max_s']:.4f})  NaN {nan}  loadavg {row['loadavg']:.1f}",
              flush=True)
    del fn, model
    return t_load, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--L", default="32,128")
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    t_load, rows = asyncio.run(run(args))
    js = Path(args.json or HERE / "_work" / f"bench_encoder_{Path(args.bundle).name.split('.')[0]}.json")
    json.dump(dict(bundle=str(args.bundle), unit="default(aot)", load_s=t_load, rows=rows,
                   note="provisional: other sessions share this Mac"), open(js, "w"), indent=2)
    print(f"[bench] -> {js}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
