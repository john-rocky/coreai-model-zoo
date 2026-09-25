"""Diagnostic: what the encoder bundles' per-token errors do to the DiT, for prompts other than the oracle's.

For each bundle x prompt (``_work/ref_hf_fp32_text/``) x L (exact Lfull, and right-padded to
``--pad-L``): the engine's ``hidden[:, drop_idx:Lfull]`` and the HF fp32 reference's go through
the same fp32 DiT (real weights) on the apple oracle's latent/t at ``--steps`` (teacher-forced:
the latent is fixed, only the text condition differs), and the two velocities are compared.
A sensitivity measurement, not the oracle gate (only the apple prompt has an oracle trajectory).

Run (base venv; GPU lock for the engine part, then ~28 GB RAM for the DiT):
  python3 ~/code/coreai-kit/scripts/with-gpu-lock.py -- ~/code/coreai/coreai-models/.venv/bin/python \\
      downstream_prompts.py <a>.h16c.aimodelc [<b>.h16c.aimodelc ...]
"""
from __future__ import annotations

import argparse
import asyncio
import gc
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
from sweep_encoder_L import PAD, load_refs, tok_corr  # noqa: E402

MODEL = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
DROP = 14


async def engine_outputs(bundle: Path, refs, pad_L: int):
    import coreai.runtime as rt
    model = await rt.AIModel.load(bundle, rt.SpecializationOptions.default())
    fn = model.load_function("main")
    outs = {}
    for key, (ids, _) in refs.items():
        Lf = ids.shape[1]
        for L in (Lf, pad_L):
            x = np.full((1, L), PAD, dtype=np.int32)
            x[0, :Lf] = ids[0]
            r = await fn({"input_ids": rt.NDArray(torch.from_numpy(x).contiguous())})
            outs[(key, L)] = np.array(r["hidden"].numpy(), dtype=np.float32)[0, :Lf]
    del fn, model
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundles", nargs="+")
    ap.add_argument("--pad-L", type=int, default=64)
    ap.add_argument("--steps", default="0,20,39")
    ap.add_argument("--json", default=str(HERE / "_work" / "downstream_prompts.json"))
    args = ap.parse_args()
    refs = load_refs()
    engine = {}
    for b in args.bundles:
        engine[Path(b).name.split(".")[0]] = asyncio.run(engine_outputs(Path(b), refs, args.pad_L))
    gc.collect()

    from qi21_dit import load_qi21_dit
    from qi21_host import build_inputs
    from qi21_oracle import Oracle, parse_steps
    o = Oracle(HERE / "oracle" / "256")
    tdir = Path(hf_snapshot(MODEL, revision=REVISION)) / "transformer"
    axes = tuple(json.load(open(tdir / "config.json")).get("axes_dims_rope", (16, 56, 56)))
    t0 = time.time()
    dit = load_qi21_dit(tdir, dtype=torch.float32)
    print(f"[downstream] DiT fp32 loaded in {time.time() - t0:.0f}s; latents/t from oracle/256 steps {args.steps}",
          flush=True)
    steps = parse_steps(args.steps, o.steps)
    rows = []
    with torch.no_grad():
        vref = {}
        for key, (ids, ref) in refs.items():
            pe = torch.from_numpy(ref[DROP:][None].copy())
            vref[key] = {s: dit(**build_inputs(o.latent(s), pe, o.t(s), o.H, o.W, axes_dim=axes)).double().numpy().ravel()
                         for s in steps}
        for name, outs in engine.items():
            for (key, L), h in outs.items():
                c_tok = tok_corr(h[DROP:], refs[key][1][DROP:])
                pe = torch.from_numpy(h[DROP:][None].copy())
                cs = []
                for s in steps:
                    v = dit(**build_inputs(o.latent(s), pe, o.t(s), o.H, o.W, axes_dim=axes)).double().numpy().ravel()
                    cs.append(float(np.corrcoef(v, vref[key][s])[0, 1]))
                rows.append(dict(bundle=name, prompt=key, L=L, min_tok_corr_embeds=float(c_tok.min()),
                                 vel_corr=dict(zip(map(str, steps), cs))))
                print(f"  {name:<42} {key:<8} L {L:3d}: prompt_embeds min tok corr {c_tok.min():.6f} (tok "
                      f"{int(c_tok.argmin()) + DROP})  -> DiT vel corr vs fp32-ref condition: "
                      + "  ".join(f"step {s} {c:.6f}" for s, c in zip(steps, cs)), flush=True)
    json.dump(rows, open(args.json, "w"), indent=2)
    print(f"[downstream] -> {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
