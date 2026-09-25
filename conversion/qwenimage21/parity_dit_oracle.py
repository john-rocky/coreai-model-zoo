"""Gate: the re-authored DiT with the REAL weights, fp32 on CPU, teacher-forced against the oracle.

Per step s: ``latent_s`` + ``prompt_embeds`` + ``t_s`` (the exact fp32 timestep/1000 the pipeline fed)
-> ``QI21DiT`` (``load_qi21_dit(..., dtype=float32)``) -> compared with ``vel_s``.
Bar: corr >= 0.999999 and max|d| / max|ref| <= 1e-3 on every step run.

``--controls`` also runs the three negative-control variants of ``parity_dit_torch.py`` (RoPE
tables swapped / text modulated from t / bidirectional text) on a few steps against the same
oracle — calibration, not part of the verdict (structure is gated by ``parity_dit_torch.py``):
it tells whether this bar, and the engine gate's corr >= 0.999, could see each bug class with the
real weights.

``--save-outputs`` writes the fp32 re-author outputs to ``_work/torch_fp32/<tag>/vel_<s>.f32``.

Run (either venv — no diffusers import; 28 GB RAM):
  python parity_dit_oracle.py --oracle oracle/256 --steps all --controls 0,20,39
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import hf_snapshot  # noqa: E402
from qi21_dit import load_qi21_dit  # noqa: E402
from qi21_host import build_inputs  # noqa: E402
from qi21_oracle import Oracle, parse_steps  # noqa: E402

MODEL = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
BAR_CORR, BAR_REL_MAX = 0.999999, 1e-3


def score(out: torch.Tensor, ref: torch.Tensor) -> dict:
    a = out.detach().double().flatten().numpy()
    b = ref.detach().double().flatten().numpy()
    maxd = float(np.abs(a - b).max())
    return dict(corr=float(np.corrcoef(a, b)[0, 1]), maxd=maxd, max_ref=float(np.abs(b).max()),
                rel_max=maxd / float(np.abs(b).max()), rel_l2=float(np.linalg.norm(a - b) / np.linalg.norm(b)),
                nan=int(np.isnan(a).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default=str(HERE / "oracle" / "256"))
    ap.add_argument("--steps", default="all", help="'all' or comma list (negative = from the end)")
    ap.add_argument("--controls", default="", help="comma list of steps for the negative controls")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save-outputs", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--transformer-dir", default=None, help="default: the pinned HF snapshot's transformer/")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    o = Oracle(args.oracle)
    print(f"[parity-oracle] {o.describe()}", flush=True)
    tdir = Path(args.transformer_dir or Path(hf_snapshot(MODEL, revision=REVISION)) / "transformer")
    axes = tuple(json.load(open(tdir / "config.json")).get("axes_dims_rope", (16, 56, 56)))
    t0 = time.time()
    model = load_qi21_dit(tdir, dtype=torch.float32)
    print(f"[parity-oracle] loaded {tdir} fp32 in {time.time() - t0:.0f}s "
          f"({sum(p.numel() for p in model.parameters()) / 1e9:.3f} B params), torch threads "
          f"{torch.get_num_threads()}", flush=True)
    assert o.C == model.img_in.in_features and o.ctx == model.txt_in.in_layer.in_features

    steps = parse_steps(args.steps, o.steps)
    out_dir = HERE / "_work" / "torch_fp32" / o.dir.name
    if args.save_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for s in steps:
            ins = build_inputs(o.latent(s), o.prompt_embeds, o.t(s), o.H, o.W, axes_dim=axes)
            t1 = time.time()
            out = model(**ins)
            sec = time.time() - t1
            r = dict(step=s, t=float(o.t(s)), sec=sec, **score(out, o.vel(s)))
            r["ok"] = r["nan"] == 0 and r["corr"] >= BAR_CORR and r["rel_max"] <= BAR_REL_MAX
            rows.append(r)
            if args.save_outputs:
                np.ascontiguousarray(out.numpy(), "<f4").tofile(out_dir / f"vel_{s}.f32")
            print(f"  step {s:2d} t={r['t']:.6f}: corr {r['corr']:.9f}  max|d| {r['maxd']:.3e}  "
                  f"max|d|/max|ref| {r['rel_max']:.2e}  |d|/|ref| {r['rel_l2']:.2e}  ({sec:.1f}s)  "
                  f"{'PASS' if r['ok'] else 'FAIL'}", flush=True)

        controls = []
        if args.controls:
            from parity_dit_torch import BidirText, TextFromT, swapped_rope
            variants = {"a_rope_swapped": None, "b_text_mod_from_t": TextFromT,
                        "c_text_bidirectional": BidirText}
            for s in parse_steps(args.controls, o.steps):
                ins = build_inputs(o.latent(s), o.prompt_embeds, o.t(s), o.H, o.W, axes_dim=axes)
                for key, cls in variants.items():
                    if cls is None:
                        out = model(**swapped_rope(ins))
                    else:
                        v = copy.copy(model)       # shallow: same parameters, different forward
                        v.__class__ = cls
                        out = v(**ins)
                    r = dict(step=s, control=key, **score(out, o.vel(s)))
                    r["red_vs_this_bar"] = not (r["corr"] >= BAR_CORR and r["rel_max"] <= BAR_REL_MAX)
                    r["red_vs_engine_bar_0.999"] = r["corr"] < 0.999
                    controls.append(r)
                    print(f"  control {key:<22} step {s:2d}: corr {r['corr']:.6f}  max|d|/max|ref| "
                          f"{r['rel_max']:.2e}  |d|/|ref| {r['rel_l2']:.2e}  "
                          f"{'RED' if r['red_vs_this_bar'] else 'NOT RED'} here, "
                          f"{'red' if r['red_vs_engine_bar_0.999'] else 'NOT red'} at corr 0.999", flush=True)

    ok = all(r["ok"] for r in rows)
    if controls:
        # calibration, not part of this gate's verdict: structure is gated by parity_dit_torch.py
        # (max|d| <= 1e-4 on a random config, where all three are red by 2-3 orders of magnitude)
        n = len(controls)
        print(f"[parity-oracle] controls red at this bar: {sum(c['red_vs_this_bar'] for c in controls)}/{n}; "
              f"red at the engine bar corr 0.999: {sum(c['red_vs_engine_bar_0.999'] for c in controls)}/{n}",
              flush=True)
    worst = min(rows, key=lambda r: r["corr"])
    worst_rel = max(rows, key=lambda r: r["rel_max"])
    print(f"\n[parity-oracle] {len(rows)} steps: min corr {worst['corr']:.9f} (step {worst['step']}), "
          f"max max|d|/max|ref| {worst_rel['rel_max']:.2e} (step {worst_rel['step']})  "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    js = Path(args.json or HERE / "_work" / f"parity_dit_oracle_{o.dir.name}.json")
    js.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(oracle=o.describe(), bar=dict(corr=BAR_CORR, rel_max=BAR_REL_MAX), steps=rows,
                   controls=controls, ok=ok), open(js, "w"), indent=2)
    print(f"[parity-oracle] -> {js}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
