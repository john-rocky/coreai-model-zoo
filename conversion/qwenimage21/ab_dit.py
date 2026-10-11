"""Paired A/B of two DiT bundles in ONE process: s/forward, alternated A, B, A, B, ...

Both bundles are loaded with ``SpecializationOptions.default()`` and fed the same inputs: random
latents / text features with the real RoPE tables (``qi21_host.build_inputs``, seed 0, t = 0.5; speed
does not depend on the values). Order: A first call, B first call (each timed on its own: it pays for
specializing the shape), ``--warm`` interleaved warm pairs (not scored), then ``--reps`` pairs A, B.
The clock covers the call and reading the output back, as in ``bench_dit.py``.

Reported per form: median / min / max / IQR (q1..q3) of the scored calls; per pair: B - A, its median
and IQR; NaN count of every output; corr / max|d| of B's output vs A's on the same input.
Decision rule (fixed before timing): B is faster when median(B) < 0.97 x median(A) (the speed ladder's
3 % resolution); the difference counts as resolved when |median(B) - median(A)| > max(IQR(A), IQR(B)).
``--known-a``: a known-good median of A on this shape (e.g. ``bench_dit.py`` in an earlier window);
the ratio is printed so a contended window shows up.

Run inside the measurement window (base venv, from conversion/qwenimage21/):
  python3 ~/code/standup/tools/quiet/quiet_hold.py "<label>" -- \\
      ~/code/coreai/coreai-models/.venv/bin/python ab_dit.py --a <A.aimodelc> --b <B.aimodelc> \\
      --L 18 --hw 64 --reps 8 --warm 2 --json <out.json>
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
FASTER = 0.97


def spread(xs: list[float]) -> dict:
    q1, q3 = np.percentile(xs, [25, 75])
    return dict(median=statistics.median(xs), min=min(xs), max=max(xs), q1=float(q1), q3=float(q3),
                iqr=float(q3 - q1), n=len(xs))


async def run(args):
    import coreai.runtime as rt
    g = torch.Generator().manual_seed(0)
    H = W = args.hw
    ins = build_inputs(torch.randn(1, H * W, 64, generator=g), torch.randn(1, args.L, 4096, generator=g),
                       torch.tensor([0.5]), H, W)
    forms = {}
    for key, path in (("A", args.a), ("B", args.b)):
        t0 = time.perf_counter()
        model = await rt.AIModel.load(Path(path), rt.SpecializationOptions.default())
        fn = model.load_function("main")
        forms[key] = dict(path=str(path), model=model, fn=fn, load_s=time.perf_counter() - t0,
                          payload={k: rt.NDArray(ins[k].contiguous()) for k in ORDER}, nan=0, calls=0)
        print(f"[ab] {key} = {Path(path).name}: load {forms[key]['load_s']:.1f}s", flush=True)

    async def call(key):
        f = forms[key]
        t = time.perf_counter()
        r = await f["fn"](f["payload"])
        out = np.asarray(r["vel"].numpy(), dtype=np.float32)
        sec = time.perf_counter() - t
        f["nan"] += int(np.isnan(out).sum())
        f["calls"] += 1
        return sec, out

    la0 = os.getloadavg()[0]
    first = {}
    outs = {}
    for key in ("A", "B"):
        first[key], outs[key] = await call(key)
        print(f"[ab] {key} first call {first[key]:.2f}s", flush=True)
    warm = {"A": [], "B": []}
    for _ in range(args.warm):
        for key in ("A", "B"):
            warm[key].append((await call(key))[0])
    times = {"A": [], "B": []}
    order = []
    for i in range(args.reps):
        for key in ("A", "B"):
            sec, _ = await call(key)
            times[key].append(sec)
            order.append(key)
        print(f"[ab] pair {i + 1}/{args.reps}: A {times['A'][-1]:.3f}s  B {times['B'][-1]:.3f}s  "
              f"B-A {times['B'][-1] - times['A'][-1]:+.3f}s", flush=True)
    la1 = os.getloadavg()[0]

    a, b = outs["A"].astype(np.float64).ravel(), outs["B"].astype(np.float64).ravel()
    same_input = dict(corr=float(np.corrcoef(a, b)[0, 1]), maxd=float(np.abs(a - b).max()),
                      max_abs_a=float(np.abs(a).max()), rel_l2=float(np.linalg.norm(a - b) / np.linalg.norm(a)))
    sa, sb = spread(times["A"]), spread(times["B"])
    diffs = [tb - ta for ta, tb in zip(times["A"], times["B"])]
    sd = spread(diffs)
    d_med = sb["median"] - sa["median"]
    res = dict(
        shape=dict(L=args.L, H=H, W=W, N=H * W, px=f"{H * 16}x{W * 16}"),
        A=dict(path=forms["A"]["path"], load_s=forms["A"]["load_s"], first_s=first["A"], warm_s=warm["A"],
               times_s=times["A"], nan=forms["A"]["nan"], calls=forms["A"]["calls"], **{"stats": sa}),
        B=dict(path=forms["B"]["path"], load_s=forms["B"]["load_s"], first_s=first["B"], warm_s=warm["B"],
               times_s=times["B"], nan=forms["B"]["nan"], calls=forms["B"]["calls"], **{"stats": sb}),
        order=order, pair_diff_s=dict(values=diffs, **sd),
        median_diff_s=d_med, median_ratio=sb["median"] / sa["median"],
        b_faster=bool(sb["median"] < FASTER * sa["median"]),
        resolved=bool(abs(d_med) > max(sa["iqr"], sb["iqr"])),
        rule=f"B faster iff median(B) < {FASTER} x median(A); resolved iff |median(B)-median(A)| > max(IQR(A), IQR(B))",
        b_vs_a_output=same_input, loadavg=dict(start=la0, end=la1), warm=args.warm, reps=args.reps,
        unit="SpecializationOptions.default()")
    if args.known_a:
        res["known_a_s"] = args.known_a
        res["a_vs_known"] = sa["median"] / args.known_a
    for key, s in (("A", sa), ("B", sb)):
        print(f"[ab] {key}: median {s['median']:.4f}s  min {s['min']:.4f}  max {s['max']:.4f}  "
              f"IQR {s['iqr']:.4f} ({s['q1']:.4f}..{s['q3']:.4f})  NaN {forms[key]['nan']}", flush=True)
    print(f"[ab] B-A: median of medians {d_med:+.4f}s ({100 * (res['median_ratio'] - 1):+.2f} %), paired median "
          f"{sd['median']:+.4f}s (IQR {sd['iqr']:.4f}, min {sd['min']:+.4f} / max {sd['max']:+.4f})", flush=True)
    print(f"[ab] B faster (< {FASTER} x A): {res['b_faster']}; resolved (|diff| > max IQR): {res['resolved']}; "
          f"B vs A output corr {same_input['corr']:.6f} max|d| {same_input['maxd']:.3e}; loadavg "
          f"{la0:.1f} -> {la1:.1f}" + (f"; A / known {res['a_vs_known']:.4f}" if args.known_a else ""), flush=True)
    del forms
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="bundle A (the reference form)")
    ap.add_argument("--b", required=True, help="bundle B (the candidate)")
    ap.add_argument("--L", type=int, default=18, help="text tokens (apple prompt = 18)")
    ap.add_argument("--hw", type=int, default=64, help="latent side (N = hw^2; 64 = 1024x1024)")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--known-a", type=float, default=None, help="known-good median of A on this shape (s)")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    res = asyncio.run(run(args))
    json.dump(res, open(args.json, "w"), indent=2)
    print(f"[ab] -> {args.json}", flush=True)
    return 0 if res["A"]["nan"] == 0 and res["B"]["nan"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
