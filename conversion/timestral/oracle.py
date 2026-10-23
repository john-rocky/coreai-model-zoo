"""Run the official Timestral forecaster on every fixture window (the oracle).

`TimestralForecaster.from_pretrained(<weights dir>).forecast(context, prediction_length=H)` with the
default arguments, one call per saved window, fp32 on CPU. Writes `<group>.oracle_q.npy` float32
[B, 1, H, 21] next to the contexts, plus `oracle_run.json` (versions, threads, timings, and a
second pass over the first window of every group for the run-to-run difference).

Run with the oracle venv (uv sync --frozen of github.com/mistralai/timestral @ a534bdb).
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import timestral
from timestral import TimestralForecaster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--fixtures", required=True, type=Path)
    ap.add_argument("--repo", required=True, type=Path, help="timestral git checkout (for the commit id)")
    a = ap.parse_args()

    man = json.loads((a.fixtures / "manifest.json").read_text())
    t0 = time.time()
    fc = TimestralForecaster.from_pretrained(a.weights)
    load_s = time.time() - t0
    commit = subprocess.run(["git", "-C", str(a.repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    info = dict(torch=torch.__version__, numpy=np.__version__, timestral=timestral.__version__, commit=commit,
                python=platform.python_version(), machine=platform.machine(), threads=torch.get_num_threads(),
                weights=str(a.weights), load_s=round(load_s, 3), forecast_kwargs="defaults (symmetric_averaging=True, "
                "clip_nonnegative=True; forecaster defaults max_context=32768, per_shot_horizon_patches=32, "
                "quantile_real_cap_k=1e4)", device=str(fc.device), dtype=str(next(fc.model.parameters()).dtype),
                quantile_levels=list(fc.quantile_levels), groups={})
    print(json.dumps({k: v for k, v in info.items() if k != "groups"}))
    for g in man["groups"]:
        name, h = g["name"], g["horizon"]
        ctx = np.load(a.fixtures / f"{name}.context.npy")
        out = np.full((ctx.shape[0], 1, h, len(fc.quantile_levels)), np.nan, dtype=np.float32)
        times, errors = [], []
        for b in range(ctx.shape[0]):
            t1 = time.time()
            try:
                q = fc.forecast(ctx[b], prediction_length=h).quantiles
                out[b] = q[0].numpy()
            except ValueError as e:
                errors.append(dict(window=b, error=str(e)))
            times.append(time.time() - t1)
        np.save(a.fixtures / f"{name}.oracle_q.npy", out)
        # Run-to-run: the first window again in the same process.
        rerun = fc.forecast(ctx[0], prediction_length=h).quantiles[0].numpy()
        d = float(np.nanmax(np.abs(rerun - out[0]))) if np.isfinite(out[0]).any() else None
        mono = bool((np.diff(out, axis=-1)[np.isfinite(out[..., 1:])] >= 0).all())
        info["groups"][name] = dict(n=int(ctx.shape[0]), horizon=h, context_length=int(ctx.shape[1]),
                                    s_per_window_median=round(float(np.median(times)), 4),
                                    s_total=round(float(np.sum(times)), 3), errors=errors,
                                    nan=int(np.isnan(out).sum()), monotone=mono, rerun_max_abs_diff=d)
        print(f"{name:28s} B={ctx.shape[0]:3d} H={h:4d} {np.median(times):.3f}s/window nan={int(np.isnan(out).sum())} "
              f"mono={mono} rerun|d|={d} errors={len(errors)}", flush=True)
    (a.fixtures / "oracle_run.json").write_text(json.dumps(info, indent=1) + "\n")
    print("ORACLE_DONE")


if __name__ == "__main__":
    main()
