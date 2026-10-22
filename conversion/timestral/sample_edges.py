"""What the released forecaster does at the edges of its input: which inputs raise (and with what), and what comes
back for the degenerate ones. The kit sample's error paths are set against this table.

Run with the oracle venv (uv sync --frozen of github.com/mistralai/timestral @ a534bdb):
    python sample_edges.py --weights <Timestral-209M dir> --out edges.json
"""
import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch

from timestral import TimestralForecaster

NAN, INF = float("nan"), float("inf")


def cases():
    t = np.arange(400, dtype=np.float32)
    seasonal = (10 + 3 * np.sin(2 * np.pi * t / 24)).astype(np.float32)
    yield "empty context", np.zeros(0, np.float32), 24
    yield "every value NaN", np.full(100, NAN, np.float32), 24
    yield "every value +inf", np.full(50, INF, np.float32), 24
    yield "one value", np.array([5.0], np.float32), 24
    yield "one observed value among NaN", np.array([NAN] * 40 + [7.0] + [NAN] * 10, np.float32), 24
    yield "constant 3.0 x 200", np.full(200, 3.0, np.float32), 24
    yield "20 values (shorter than one patch)", seasonal[:20], 24
    yield "horizon 0", seasonal, 0
    yield "horizon 1", seasonal, 1
    yield "horizon 1024", seasonal, 1024
    yield "horizon 1025 (two decode blocks)", seasonal, 1025
    yield "values around 1e30", (seasonal * np.float32(1e29)).astype(np.float32), 24
    yield "values around 3e38 (float32 max)", np.full(100, 3e38, np.float32), 24
    yield "context of 40,000 steps (over max_context 32,768)", np.tile(seasonal, 100), 24
    yield "negative and positive", (seasonal - 10).astype(np.float32), 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    fc = TimestralForecaster.from_pretrained(a.weights)
    rows = []
    for name, ctx, h in cases():
        row = dict(case=name, context_length=int(ctx.size), horizon=h)
        try:
            q = fc.forecast(ctx, prediction_length=h).quantiles[0, 0].numpy()
            row.update(outcome="forecast", shape=list(q.shape), nan=int(np.isnan(q).sum()),
                       inf=int(np.isinf(q).sum()),
                       min=float(np.nanmin(q)) if np.isfinite(q).any() else None,
                       max=float(np.nanmax(q)) if np.isfinite(q).any() else None,
                       median_first=[float(v) for v in q[:3, 10]])
        except Exception as e:  # the type and message are the point
            row.update(outcome="raises", error_type=type(e).__name__, message=str(e)[:400],
                       where=traceback.extract_tb(e.__traceback__)[-1].name)
        rows.append(row)
        print(json.dumps(row), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(torch=torch.__version__, numpy=np.__version__, cases=rows), indent=1) + "\n")
    print("SAMPLE_EDGES_DONE", flush=True)


if __name__ == "__main__":
    main()
