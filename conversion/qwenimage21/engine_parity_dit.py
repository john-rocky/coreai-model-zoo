"""Gate: teacher-forced parity of the EXPORTED Core AI DiT bundle vs the fp32 oracle.

Per step s: ``qi21_host.build_inputs(latent_s, prompt_embeds, t_s)`` (t_s = the exact fp32
timestep/1000 the pipeline fed) -> the bundle's ``main`` -> ``vel`` vs the oracle's ``vel_s``.
Bar: every step corr >= 0.999 and no NaN. Prints per-step corr / max|d| / |d|/|ref| and the
worst of each. If ``_work/torch_fp32/<tag>/vel_<s>.f32`` exists (``parity_dit_oracle.py
--save-outputs``) the engine is also scored against the fp32 re-author on the same inputs.

An all-zero or NaN output is first checked against the known Python-runtime GPU JIT failure
(``MTL4CommandQueueErrorDomain error 1`` in the log): re-run the same steps with ``--cpu-only``
to tell a runtime failure from a graph error.

Run (coreai-models base venv, from conversion/qwenimage21/):
  python engine_parity_dit.py <bundle.aimodel> --oracle oracle/256 [--steps all] [--cpu-only]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from qi21_host import build_inputs  # noqa: E402
from qi21_oracle import Oracle, parse_steps  # noqa: E402

ORDER = ("img_tokens", "txt_feats", "timestep", "txt_cos", "txt_sin", "img_cos", "img_sin")
BAR_CORR = 0.999


def score(out: np.ndarray, ref: np.ndarray) -> dict:
    a = out.astype(np.float64).ravel()
    b = ref.astype(np.float64).ravel()
    nan = int(np.isnan(a).sum())
    if nan:
        return dict(corr=float("nan"), maxd=float("nan"), rel_l2=float("nan"), nan=nan,
                    zero=bool(np.nanmax(np.abs(a)) == 0) if nan < a.size else False)
    return dict(corr=float(np.corrcoef(a, b)[0, 1]), maxd=float(np.abs(a - b).max()),
                rel_l2=float(np.linalg.norm(a - b) / np.linalg.norm(b)), nan=0, zero=bool(np.abs(a).max() == 0))


async def run(args):
    import coreai.runtime as rt
    o = Oracle(args.oracle)
    print(f"[engine-parity] {o.describe()}", flush=True)
    # `default()` auto-partitions and put part of the full 32-layer graph on the ANE, which then
    # failed (`ANERegion.mm:414 failed assertion ... Code=-19`, 2026-09-25). `--gpu` asks for the GPU
    # as the preferred unit (the API has no setter for the allowed set).
    unit = "cpu_only" if args.cpu_only else ("gpu" if args.gpu else "default")
    if args.cpu_only:
        opts = rt.SpecializationOptions.cpu_only()
    elif args.gpu:
        opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    else:
        opts = rt.SpecializationOptions.default()
    t0 = time.time()
    model = await rt.AIModel.load(Path(args.bundle), opts)       # keep the AIModel alive
    fn = model.load_function("main")
    print(f"[engine-parity] {Path(args.bundle).name} loaded ({unit}) in {time.time() - t0:.1f}s", flush=True)
    tf_dir = HERE / "_work" / "torch_fp32" / o.dir.name

    rows = []
    for s in parse_steps(args.steps, o.steps):
        ins = build_inputs(o.latent(s), o.prompt_embeds, o.t(s), o.H, o.W,
                           axes_dim=tuple(int(v) for v in args.axes.split(",")))
        t1 = time.time()
        r = await fn({k: rt.NDArray(ins[k].contiguous()) for k in ORDER})
        out = np.array(r["vel"].numpy(), dtype=np.float32).reshape(1, o.N, o.C)
        sec = time.time() - t1
        row = dict(step=s, t=float(o.t(s)), sec=sec, **score(out, o.vel(s).numpy()))
        tf = tf_dir / f"vel_{s}.f32"
        if tf.exists():
            ref32 = np.fromfile(tf, "<f4").reshape(1, o.N, o.C)
            row["vs_torch_fp32"] = score(out, ref32)
        row["ok"] = row["nan"] == 0 and not row["zero"] and row["corr"] >= BAR_CORR
        rows.append(row)
        extra = (f"  | vs torch fp32 corr {row['vs_torch_fp32']['corr']:.6f}" if "vs_torch_fp32" in row else "")
        print(f"  step {s:2d} t={row['t']:.6f}: corr {row['corr']:.6f}  max|d| {row['maxd']:.3e}  "
              f"|d|/|ref| {row['rel_l2']:.3e}  NaN {row['nan']}  ({sec:.2f}s){extra}  "
              f"{'PASS' if row['ok'] else 'FAIL'}", flush=True)
        if row["nan"] or row["zero"]:
            print("  !! NaN/zero output: check this log for 'MTL4CommandQueueErrorDomain' and re-run the "
                  "same steps with --cpu-only", flush=True)
    del model
    return o, unit, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--oracle", default=str(HERE / "oracle" / "256"))
    ap.add_argument("--steps", default="all")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="preferred compute unit = GPU (avoids the ANE region failure)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--axes", default="16,56,56", help="axes_dims_rope of the exported model")
    args = ap.parse_args()
    o, unit, rows = asyncio.run(run(args))

    valid = [r for r in rows if r["nan"] == 0]
    ok = bool(rows) and all(r["ok"] for r in rows)
    print(f"\n[engine-parity] {unit}, {len(rows)} steps, NaN total {sum(r['nan'] for r in rows)}", flush=True)
    if valid:
        wc = min(valid, key=lambda r: r["corr"])
        wd = max(valid, key=lambda r: r["maxd"])
        wr = max(valid, key=lambda r: r["rel_l2"])
        print(f"  min corr {wc['corr']:.6f} (step {wc['step']})  max max|d| {wd['maxd']:.3e} (step {wd['step']})  "
              f"max |d|/|ref| {wr['rel_l2']:.3e} (step {wr['step']})", flush=True)
        if all("vs_torch_fp32" in r for r in valid):
            wt = min(valid, key=lambda r: r["vs_torch_fp32"]["corr"])
            print(f"  vs torch fp32 re-author: min corr {wt['vs_torch_fp32']['corr']:.6f} (step {wt['step']})",
                  flush=True)
    print(f"[engine-parity] {'PASS' if ok else 'FAIL'} (bar: every step corr >= {BAR_CORR}, NaN 0)", flush=True)
    js = Path(args.json or HERE / "_work" / f"engine_parity_dit_{o.dir.name}_{unit}.json")
    js.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(bundle=str(args.bundle), oracle=o.describe(), unit=unit, bar_corr=BAR_CORR, steps=rows, ok=ok),
              open(js, "w"), indent=2)
    print(f"[engine-parity] -> {js}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
