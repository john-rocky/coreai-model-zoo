"""Characterise the engine's per-token precision over prompts and sequence lengths (diagnostic).

References: ``_work/ref_hf_fp32_text/<prompt>/`` (``ref_text_hf.py``: HF fp32, the oracle's path).
``--reauthor`` first checks the fp32 re-author against them (bit-exact expected) on CPU.
Then per bundle (AOT, ``SpecializationOptions.default()``), per prompt, per L (the exact Lfull,
then right-padded with 151643 to each ``--Ls`` value above Lfull): the first Lfull outputs vs the
reference, per token. Prints token 14's corr (the user turn's ``<|im_start|>``), the minimum over
the other tokens and how many tokens fall under 0.999.

Run (base venv; GPU lock):
  python3 ~/code/coreai-kit/scripts/with-gpu-lock.py -- ~/code/coreai/coreai-models/.venv/bin/python \\
      sweep_encoder_L.py <a>.h16c.aimodelc [<b>.h16c.aimodelc ...] [--reauthor]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import hf_snapshot  # noqa: E402

MODEL = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
PAD = 151643
TOK = 14


def load_refs():
    refs = {}
    for d in sorted((HERE / "_work" / "ref_hf_fp32_text").iterdir()):
        if d.is_dir():
            ids = np.fromfile(d / "ids.i32", "<i4")
            refs[d.name] = (ids[None].astype(np.int32), np.fromfile(d / "hidden.f32", "<f4").reshape(len(ids), -1))
    return refs


def tok_corr(out, ref):
    a = out.astype(np.float64)
    b = ref.astype(np.float64)
    return np.array([np.corrcoef(a[i], b[i])[0, 1] for i in range(b.shape[0])])


async def sweep(bundle: Path, refs, Ls):
    import coreai.runtime as rt
    model = await rt.AIModel.load(bundle, rt.SpecializationOptions.default())
    fn = model.load_function("main")
    rows = []
    for key, (ids, ref) in refs.items():
        Lf = ids.shape[1]
        for L in [Lf] + [v for v in Ls if v > Lf]:
            x = np.full((1, L), PAD, dtype=np.int32)
            x[0, :Lf] = ids[0]
            r = await fn({"input_ids": rt.NDArray(torch.from_numpy(x).contiguous())})
            out = np.array(r["hidden"].numpy(), dtype=np.float32)[0, :Lf]
            nan = int(np.isnan(out).sum())
            c = tok_corr(out, ref) if nan == 0 else np.full(Lf, np.nan)
            others = np.delete(c, TOK)
            rows.append(dict(prompt=key, Lfull=Lf, L=L, nan=nan, tok14=float(c[TOK]), min_other=float(others.min()),
                             min_other_at=int(np.argmin(others) + (np.argmin(others) >= TOK)),
                             n_below=int((c < 0.999).sum())))
            print(f"  {bundle.name.split('.')[0]:<42} {key:<8} Lfull {Lf:3d} L {L:3d}: tok14 corr {c[TOK]:.6f}  "
                  f"min other {others.min():.6f} (tok {rows[-1]['min_other_at']})  n<0.999 {rows[-1]['n_below']}  "
                  f"NaN {nan}", flush=True)
    del fn, model
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundles", nargs="*")
    ap.add_argument("--Ls", default="48,64,96,128,256")
    ap.add_argument("--reauthor", action="store_true",
                    help="CPU: the fp32 re-author and the w16a32 torch module vs the HF fp32 references first")
    ap.add_argument("--json", default=str(HERE / "_work" / "sweep_encoder_L.json"))
    args = ap.parse_args()
    refs = load_refs()
    print(f"[sweep] references: " + ", ".join(f"{k} (Lfull {v[0].shape[1]})" for k, v in refs.items()), flush=True)
    res = dict(reauthor={}, engine={})
    if args.reauthor:
        from qi21_text import load_qi21_text
        tdir = Path(hf_snapshot(MODEL, revision=REVISION)) / "text_encoder"
        for tag, kw in (("fp32", dict(dtype=torch.float32)),
                        ("w16a32", dict(dtype=torch.bfloat16, compute_fp32=True, io_fp32=True))):
            m = load_qi21_text(tdir, **kw)
            with torch.no_grad():
                for key, (ids, ref) in refs.items():
                    out = m(torch.from_numpy(ids)).numpy()[0]
                    same = bool(np.array_equal(out, ref))
                    res["reauthor"][f"{tag}/{key}"] = dict(bit_exact=same, max_abs=float(np.abs(out - ref).max()))
                    print(f"[sweep] {tag} torch vs HF fp32, {key:<8}: bit-exact {same}  max|d| "
                          f"{float(np.abs(out - ref).max()):.3e}", flush=True)
            del m
    Ls = [int(v) for v in args.Ls.split(",")]
    for b in args.bundles:
        res["engine"][Path(b).name] = asyncio.run(sweep(Path(b), refs, Ls))
    json.dump(res, open(args.json, "w"), indent=2)
    print(f"[sweep] -> {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
