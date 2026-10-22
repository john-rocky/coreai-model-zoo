"""Ladders 2 and 3: host_forecast + a graph vs the oracle on every fixture window.

The oracle is the released forecaster's output on the same saved contexts (<fixtures>/<group>.oracle_q.npy,
`TimestralForecaster.forecast()` with its defaults, fp32 CPU). Per group, for ours and for the oracle:

  MASE  mean |y - median| / seasonal error, the seasonal error = mean |c_t - c_(t-m)| over the saved
        context (NaN pairs left out), m = the group's seasonality; a window without a defined seasonal
        error is left out of MASE and counted; the group's MASE = the mean over its windows
  CRPS  weighted quantile loss over the nine deciles 0.1 ... 0.9: for each level, the sum over the group's
        windows and steps of 2 |(y - q)(1{y <= q} - level)| divided by the summed |y|, then the mean over
        the levels

ratio = ours / oracle for each, per group and for the aggregate (the mean over the groups).
L2 (fp32 graph): |ratio - 1| <= 0.001 everywhere.  L3 (ship candidates): ratio <= 1.010 everywhere, no
NaN / inf, every forecast monotone over the 21 levels. Also recorded: the quantiles' max |difference| per
window over the window's max |oracle quantile|, and the windows that do not fit any bucket (out of scope).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import host_forecast as H

DECILES = [H.QUANTILES.index(q) for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]


def seasonal_error(ctx, m):
    c = ctx.astype(np.float64)
    if c.size <= m:
        return math.nan
    d = np.abs(c[m:] - c[:-m])
    return float(np.nanmean(d)) if np.isfinite(d).any() else math.nan


def group_metrics(ctxs, ys, qs, m):
    mase, left_out = [], 0
    for c, y, q in zip(ctxs, ys, qs):
        se = seasonal_error(c, m)
        e = np.abs(y.astype(np.float64) - q[:, H.MEDIAN])
        if not (np.isfinite(se) and se > 0) or not np.isfinite(e).any():
            left_out += 1
            continue
        mase.append(float(np.nanmean(e)) / se)
    y = np.stack(ys).astype(np.float64)
    q = np.stack(qs).astype(np.float64)
    fin = np.isfinite(y)
    den = float(np.abs(y[fin]).sum())
    crps = math.nan
    if den > 0:
        per = []
        for k in DECILES:
            lev = H.QUANTILES[k]
            ql = 2.0 * np.abs((y - q[..., k]) * ((y <= q[..., k]).astype(np.float64) - lev))
            per.append(float(ql[fin].sum()) / den)
        crps = float(np.mean(per))
    return dict(mase=float(np.mean(mase)) if mase else math.nan, crps=crps, mase_left_out=left_out)


def ratio(a, b):
    return a / b if (np.isfinite(a) and np.isfinite(b) and b != 0) else None


def make_core(args, n_hor, n_ctx):
    import engines
    if args.engine == "torch":
        import torch
        from timestral_core import load_core
        core = load_core(args.st, n_hor, qkv_layout="naive" if args.red == "qkv" else "released")
        return engines.TorchCore(core, torch.float32)
    bundle = args.bundle_main if n_ctx == args.main_bucket[0] and n_hor == args.main_bucket[1] else args.bundle_long
    if bundle is None:
        return None
    return engines.CoreAIGraph(bundle, args.unit, args.dtype, args.rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True, type=Path)
    ap.add_argument("--engine", choices=["torch", "coreai"], default="torch")
    ap.add_argument("--st", type=Path, help="model.safetensors (torch engine)")
    ap.add_argument("--bundle-main", type=Path, default=None)
    ap.add_argument("--bundle-long", type=Path, default=None)
    ap.add_argument("--unit", choices=["cpu", "gpu", "neural_engine"], default="gpu")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--rows", type=int, default=2, help="the Core AI graph's R")
    ap.add_argument("--main-bucket", default="64,32")
    ap.add_argument("--long-bucket", default="256,32")
    ap.add_argument("--mode", choices=["batched", "split"], default="batched")
    ap.add_argument("--red", choices=["none", "qkv", "positions"], default="none",
                    help="deliberately broken arm: wQKV read as (qkv, heads, head_dim), or absolute positions "
                         "reversed (the RoPE tables of arange(S)[::-1])")
    ap.add_argument("--positions", choices=["first-observed", "const"], default="first-observed",
                    help="const = RoPE tables of arange(L+h) for every window (relative positions are unchanged; "
                         "tests whether the tables can be baked into the graph)")
    ap.add_argument("--level", choices=["l2", "l3"], default="l2")
    ap.add_argument("--groups", default=None, help="comma-separated group names (default: all)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.main_bucket = tuple(int(v) for v in args.main_bucket.split(","))
    args.long_bucket = tuple(int(v) for v in args.long_bucket.split(","))
    man = json.loads((args.fixtures / "manifest.json").read_text())
    want = set(args.groups.split(",")) if args.groups else None

    if args.red == "positions":
        orig = H.rope_tables
        H.rope_tables = lambda pos: orig(pos[::-1].copy())

    cores, groups, out_of_scope, failures = {}, [], [], []
    worst_q, nan_inf, nonmono, n_fc = 0.0, 0, 0, 0
    t0 = time.time()
    for g in man["groups"]:
        if want and g["name"] not in want:
            continue
        ctxs = np.load(args.fixtures / f"{g['name']}.context.npy")
        ys = np.load(args.fixtures / f"{g['name']}.target.npy")
        orc = np.load(args.fixtures / f"{g['name']}.oracle_q.npy")[:, 0]
        h_len = g["horizon"]
        n_pred = math.ceil(h_len / H.PATCH)
        bucket = None
        for bk in (args.main_bucket, args.long_bucket):
            if g["context_length"] <= bk[0] * H.PATCH and n_pred <= bk[1]:
                bucket = H.Bucket(*bk)
                break
        if bucket is None:
            out_of_scope.append(dict(group=g["name"], windows=int(ctxs.shape[0]), context_length=g["context_length"],
                                     horizon=h_len, reason="no bucket holds this context and horizon"))
            continue
        key = (bucket.n_ctx, bucket.n_hor)
        if key not in cores:
            cores[key] = make_core(args, bucket.n_hor, bucket.n_ctx)
        core = cores[key]
        if core is None:
            out_of_scope.append(dict(group=g["name"], windows=int(ctxs.shape[0]), context_length=g["context_length"],
                                     horizon=h_len, reason=f"no bundle given for bucket {key}"))
            continue
        ours, orcs, keep_c, keep_y, per_window = [], [], [], [], []
        for b in range(ctxs.shape[0]):
            if not np.isfinite(orc[b]).all():
                failures.append(dict(group=g["name"], window=b, error="the oracle has no finite forecast here"))
                continue
            try:
                q = H.forecast(core, ctxs[b], h_len, bucket, mode=args.mode,
                               positions_from_first_observed=args.positions == "first-observed")
            except ValueError as e:
                failures.append(dict(group=g["name"], window=b, error=str(e)))
                continue
            n_fc += 1
            bad = int((~np.isfinite(q)).sum())
            nan_inf += bad
            mono = bool((np.diff(q, axis=-1) >= 0).all())
            nonmono += 0 if mono else 1
            d = np.abs(q.astype(np.float64) - orc[b])
            rel = float(d.max() / max(np.abs(orc[b]).max(), 1e-30))
            worst_q = max(worst_q, rel)
            per_window.append(dict(window=b, max_abs_diff=float(d.max()), max_rel_to_window=rel, nan_inf=bad,
                                   monotone=mono))
            ours.append(q)
            orcs.append(orc[b])
            keep_c.append(ctxs[b])
            keep_y.append(ys[b])
        if not ours:
            continue
        m_ours = group_metrics(keep_c, keep_y, ours, g["seasonality"])
        m_orc = group_metrics(keep_c, keep_y, orcs, g["seasonality"])
        groups.append(dict(group=g["name"], bucket=list(key), windows=len(ours), horizon=h_len,
                           ours=m_ours, oracle=m_orc,
                           mase_ratio=ratio(m_ours["mase"], m_orc["mase"]),
                           crps_ratio=ratio(m_ours["crps"], m_orc["crps"]),
                           max_rel_to_window=max(w["max_rel_to_window"] for w in per_window),
                           per_window=per_window))
        print(f"{g['name']:28s} bucket={key} B={len(ours):3d} MASE {m_ours['mase']:.6f}/{m_orc['mase']:.6f} "
              f"CRPS {m_ours['crps']:.6f}/{m_orc['crps']:.6f} max|dq|/max|q| {groups[-1]['max_rel_to_window']:.2e}",
              flush=True)

    def agg(key):
        vals = [(gr["ours"][key], gr["oracle"][key]) for gr in groups
                if np.isfinite(gr["ours"][key]) and np.isfinite(gr["oracle"][key])]
        o, r = float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals]))
        return dict(ours=o, oracle=r, ratio=ratio(o, r), groups=len(vals))

    aggregate = dict(mase=agg("mase"), crps=agg("crps"))
    ratios = [gr[k] for gr in groups for k in ("mase_ratio", "crps_ratio") if gr[k] is not None]
    ratios += [r for r in (aggregate["mase"]["ratio"], aggregate["crps"]["ratio"]) if r is not None]
    if args.level == "l2":
        passed = all(abs(r - 1.0) <= 1e-3 for r in ratios) and not failures
        rule = "|ours/oracle - 1| <= 0.001 for MASE and CRPS in every group and on the aggregate"
    else:
        passed = all(r <= 1.010 for r in ratios) and nan_inf == 0 and nonmono == 0 and not failures
        rule = "ours/oracle <= 1.010 for MASE and CRPS in every group and on the aggregate, no NaN/inf, monotone"
    res = dict(engine=args.engine, unit=args.unit if args.engine == "coreai" else "torch-cpu", dtype=args.dtype,
               rows=args.rows, mode=args.mode, red=args.red, positions=args.positions, level=args.level, rule=rule,
               passed=passed,
               bundles=dict(main=str(args.bundle_main), long=str(args.bundle_long)),
               specs_used={str(k): getattr(c, "spec_used", "torch") for k, c in cores.items() if c is not None},
               forecasts=n_fc, nan_inf=nan_inf, non_monotone=nonmono, failures=failures,
               worst_max_rel_to_window=worst_q, aggregate=aggregate,
               max_abs_log_ratio=max(abs(math.log(r)) for r in ratios) if ratios else None,
               out_of_scope=out_of_scope, seconds=round(time.time() - t0, 1), groups=groups)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("engine", "unit", "dtype", "mode", "red", "level", "passed", "forecasts",
                                           "nan_inf", "non_monotone", "worst_max_rel_to_window", "aggregate",
                                           "max_abs_log_ratio", "seconds")}, indent=1))
    print("out of scope:", out_of_scope)
    print("E2E_DONE", "PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
