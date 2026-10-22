"""L1 on the engine: a Core AI bundle vs the released module's fp32 output on the saved graph inputs.

Reads the npz that gate_core.py --save-ref wrote for the bucket (per window: patches for [s, -s],
cos, sin, attn_mask, the released fp32 output and the true-float64 output, and the bar = 2 x the
released fp32-vs-.double() floor), runs the bundle on the same inputs, and reports max |d| vs fp32
(the L1 number), vs float64, the median channel's max |d| and correlation, NaN / inf, and whether
every row is monotone over the 21 levels.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import engines
import host_forecast as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--unit", choices=["cpu", "gpu", "neural_engine"], required=True)
    ap.add_argument("--dtype", choices=["float32", "float16"], required=True)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    z = np.load(a.ref)
    t0 = time.perf_counter()
    eng = engines.CoreAIGraph(a.bundle, a.unit, a.dtype, a.rows)
    load_s = time.perf_counter() - t0
    rows, worst = [], dict(max_abs=0.0, max_abs_vs_true64=0.0, median_max=0.0)
    corr_min, nan_inf, nonmono = 1.0, 0, 0
    for n in range(z["patches"].shape[0]):
        out = eng(z["patches"][n], z["cos"][n], z["sin"][n], z["attn_mask"][n]).astype(np.float64)
        ref = z["ref32"][n].astype(np.float64)
        bad = int((~np.isfinite(out)).sum())
        d = np.abs(out - ref)
        med_o, med_r = out[..., H.MEDIAN].ravel(), ref[..., H.MEDIAN].ravel()
        c = float(np.corrcoef(med_o, med_r)[0, 1]) if bad == 0 and med_r.std() > 0 else None
        mono = bool((np.diff(out, axis=-1) >= 0).all()) if bad == 0 else False
        r = dict(group=str(z["group"][n]), window=int(z["window"][n]), nan_inf=bad,
                 max_abs=float(np.nanmax(d)) if bad < d.size else None,
                 max_abs_vs_true64=float(np.nanmax(np.abs(out - z["ref64t"][n]))) if bad < d.size else None,
                 median_max=float(np.nanmax(np.abs(med_o - med_r))) if bad < d.size else None,
                 median_corr=c, monotone=mono)
        rows.append(r)
        nan_inf += bad
        nonmono += 0 if mono else 1
        for k in worst:
            if r[k] is not None:
                worst[k] = max(worst[k], r[k])
        if c is not None:
            corr_min = min(corr_min, c)
    bar = float(z["bar"])
    res = dict(bundle=str(a.bundle), unit=a.unit, spec_used=eng.spec_used, load_error=eng.load_error, dtype=a.dtype,
               rows_graph=a.rows, ref=str(a.ref),
               bucket=z["bucket"].tolist(), windows=len(rows), load_s=round(load_s, 2), bar=bar, **worst,
               median_corr_min=corr_min, nan_inf=nan_inf, non_monotone=nonmono,
               l1_pass=bool(nan_inf == 0 and nonmono == 0 and worst["max_abs"] <= bar), rows=rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    print("ENGINE_L1_DONE", "PASS" if res["l1_pass"] else "FAIL")


if __name__ == "__main__":
    main()
