"""Export the Qwen-Image-2.1 DiT (``qi21_dit.QI21DiT``) to a Core AI ``.aimodel``.

bf16 weights + bf16 compute, fp32 graph boundary (``io_fp32`` — a Swift host cannot fill a
bfloat16 NDArray), RoPE rotated in fp32 (``rope_fp32``), both sequence axes dynamic —
text ``Dim("ntxt", 8..512)``, image ``Dim("nimg", 64..4096)``. One function, ``main``:

    img_tokens [1,N,64]  txt_feats [1,L,4096]  timestep [1] (t/1000)
    txt_cos/txt_sin [1,L,64]  img_cos/img_sin [1,N,64]        ->  vel [1,N,64]   (all fp32)

The host side (RoPE tables, pack/unpack) is ``qi21_host.py``. Bundle:
``_paths.exports_dir()/qwenimage21/<name>/<name>.aimodel``,
name = ``qi21_dit_{full|L<n>}_bf16_dyn_iofp32``.

``--check`` (default for ``--random-init``) loads the saved bundle with
``SpecializationOptions.default()`` (or ``--cpu-only``), runs one forward at N=256 / L=40 and
compares it with the same module in fp32 torch: NaN count, all-zero, corr, max|d|.

Run (coreai-models base venv — coreai-torch + coreai-opt; diffusers is not imported):
  python export_dit.py --layers 2 --random-init       # probe: no checkpoint needed
  python export_dit.py                                # full 32 layers from the HF snapshot
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import gc
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import exports_dir, hf_snapshot  # noqa: E402
from qi21_dit import QI21DiT, load_qi21_dit  # noqa: E402
from qi21_host import build_inputs  # noqa: E402

MODEL = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
ORDER = ("img_tokens", "txt_feats", "timestep", "txt_cos", "txt_sin", "img_cos", "img_sin")
DEFAULT_CFG = dict(num_layers=32, num_attention_heads=32, attention_head_dim=128, in_channels=64,
                   out_channels=64, context_in_dim=4096, mlp_ratio=3, eps=1e-6)
MIN_FREE_GIB = 30


def free_gib() -> float:
    return shutil.disk_usage("/System/Volumes/Data").free / 2**30


def randomize(model, seed):
    """Unit-variance random weights (same scheme as parity_dit_torch.py)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            if p.ndim == 2:
                p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(p.shape[1]))
            elif name.endswith("text_norm.weight"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))


AXES = (16, 56, 56)            # overwritten from the transformer config in main()


def example_inputs(L: int, H: int, W: int, seed: int, cin: int = 64, ctx: int = 4096) -> dict:
    g = torch.Generator().manual_seed(seed)
    img = torch.randn(1, H * W, cin, generator=g)
    txt = torch.randn(1, L, ctx, generator=g)
    return build_inputs(img, txt, torch.tensor([0.6180339]), H, W, axes_dim=AXES)


def stats(a: np.ndarray, b: np.ndarray):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(np.abs(a - b).max()), float(np.corrcoef(a, b)[0, 1])


def rel(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def control_outputs(m32, ins) -> dict:
    """fp32 outputs of the negative-control variants (same weights, one deliberate bug each)."""
    from parity_dit_torch import BidirText, TextFromT, swapped_rope
    out = {}
    with torch.no_grad():
        out["a_rope_swapped"] = m32(**swapped_rope(ins)).numpy()
        for key, cls in (("b_text_mod_from_t", TextFromT), ("c_text_bidirectional", BidirText)):
            v = copy.copy(m32)          # shallow: shares the parameters, only the class differs
            v.__class__ = cls
            out[key] = v(**ins).numpy()
    return out


async def engine_check(bundle: Path, model, shapes, seed: int, cpu_only: bool, controls: bool = False):
    """One bundle load, then per (L, H, W): engine forward vs the same module in fp32 torch."""
    import coreai.runtime as rt
    opts = rt.SpecializationOptions.cpu_only() if cpu_only else rt.SpecializationOptions.default()
    unit = "cpu_only" if cpu_only else "default"
    t0 = time.time()
    aim = await rt.AIModel.load(bundle, opts)
    fn = aim.load_function("main")
    print(f"[check] {unit}: load {time.time() - t0:.1f}s", flush=True)
    m32 = copy.deepcopy(model).to(torch.float32)
    res = []
    for (L, H, W) in shapes:
        ins = example_inputs(L, H, W, seed, model.img_in.in_features, model.txt_in.in_layer.in_features)
        t1 = time.time()
        r = await fn({k: rt.NDArray(ins[k].contiguous()) for k in ORDER})
        t_fwd = time.time() - t1
        out = np.array(r["vel"].numpy(), dtype=np.float32)
        with torch.no_grad():
            ref = m32(**ins).numpy()
        nan = int(np.isnan(out).sum())
        maxd, c = stats(out, ref) if nan == 0 else (float("nan"), float("nan"))
        zero = bool(np.abs(out).max() == 0)
        ok = nan == 0 and not zero and c >= 0.999 and out.shape == ref.shape
        if controls and nan == 0:
            r_ok = rel(out, ref)
            worst = float("inf")
            for key, v in control_outputs(m32, ins).items():
                r_bug, sep = rel(out, v), rel(ref, v)
                worst = min(worst, r_bug / r_ok)
                print(f"[check]   control {key:<22}: rel(engine, variant) {r_bug:.3e}  "
                      f"rel(correct, variant) {sep:.3e}  corr(engine, variant) {stats(out, v)[1]:.6f}", flush=True)
            # diagnostic only: with random weights the bf16 engine noise can exceed the variants'
            # separation, so "not distinguished" says the probe is blind, not that the graph is wrong
            print(f"[check]   rel(engine, correct) {r_ok:.3e}; nearest variant is {worst:.1f}x farther "
                  f"{'(structure distinguished)' if worst >= 3.0 else '(structure NOT distinguished by this probe)'}",
                  flush=True)
        res.append(dict(L=L, H=H, W=W, nan=nan, zero=zero, corr=c, maxd=maxd, ok=ok))
        print(f"[check] {unit} L={L} N={H * W} ({H}x{W}): fwd {t_fwd:.2f}s  shape {list(out.shape)}  NaN {nan}  "
              f"zero {zero}  corr {c:.6f}  max|d| {maxd:.3e}  |ref| max {float(np.abs(ref).max()):.3f}  "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    del aim
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=None, help="first n blocks only (probe)")
    ap.add_argument("--random-init", action="store_true", help="random weights, no checkpoint (probe)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--txt-min", type=int, default=8)
    ap.add_argument("--txt-max", type=int, default=512)
    ap.add_argument("--img-min", type=int, default=64)
    ap.add_argument("--img-max", type=int, default=4096)
    ap.add_argument("--trace-L", type=int, default=40, help="text length of the tracing inputs")
    ap.add_argument("--trace-hw", type=int, default=16, help="latent side of the tracing inputs (N = hw^2)")
    ap.add_argument("--no-optimize", action="store_true")
    ap.add_argument("--check", dest="check", action="store_true", default=None)
    ap.add_argument("--no-check", dest="check", action="store_false")
    ap.add_argument("--check-only", action="store_true", help="skip the export, re-run --check on the saved bundle")
    ap.add_argument("--cpu-only", action="store_true", help="--check with SpecializationOptions.cpu_only()")
    ap.add_argument("--check-shapes", default="40x16x16",
                    help="comma list of LxHxW for --check (one bundle load, one forward each)")
    ap.add_argument("--controls", action="store_true",
                    help="--check also scores the engine output against the fp32 outputs of the three "
                         "negative-control variants (parity_dit_torch.py) — diagnostic: says whether this "
                         "input/weight set can tell the correct structure from each bug at bf16 precision")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                    help="fp32 = diagnostic probe only (converter-structure check at fp32 precision)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--transformer-dir", default=None, help="default: the pinned HF snapshot's transformer/")
    args = ap.parse_args()
    check = args.random_init if args.check is None else args.check

    tag = "full" if args.layers is None else f"L{args.layers}"
    wdt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    name = args.name or (f"qi21_dit_{tag}_bf16_dyn_iofp32" if args.dtype == "bf16" else f"qi21_dit_{tag}_fp32_dyn")
    out_dir = exports_dir() / "qwenimage21" / name
    bundle = out_dir / f"{name}.aimodel"

    t_start = time.time()
    if args.random_init:
        cfg = dict(DEFAULT_CFG)
        if args.layers is not None:
            cfg["num_layers"] = args.layers
        model = QI21DiT.from_config(cfg, io_fp32=True, rope_fp32=True)
        randomize(model, args.seed)
        model = model.to(wdt).eval()
    else:
        tdir = Path(args.transformer_dir or Path(hf_snapshot(MODEL, revision=REVISION)) / "transformer")
        global AXES
        AXES = tuple(json.load(open(tdir / "config.json")).get("axes_dims_rope", AXES))
        print(f"[export] loading {tdir} ({args.dtype}, axes {AXES}) ...", flush=True)
        model = load_qi21_dit(tdir, dtype=wdt, n_layers=args.layers, io_fp32=True, rope_fp32=True)
    assert model.time_text_embed.freqs.dtype == torch.float32
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export] {name}: {n_params / 1e9:.3f} B params, {args.dtype}, built in {time.time() - t_start:.0f}s",
          flush=True)

    if not args.check_only:
        fg = free_gib()
        print(f"[export] disk free {fg:.1f} GiB", flush=True)
        if fg < MIN_FREE_GIB:
            print(f"[export] STOP: free {fg:.1f} GiB < {MIN_FREE_GIB} GiB", flush=True)
            return 2
        from torch.export import Dim
        from coreai_models.export.macos import export_to_coreai
        import coreai.runtime as rt

        ref = example_inputs(args.trace_L, args.trace_hw, args.trace_hw, args.seed + 1,
                             model.img_in.in_features, model.txt_in.in_layer.in_features)
        ntxt = Dim("ntxt", min=args.txt_min, max=args.txt_max)
        nimg = Dim("nimg", min=args.img_min, max=args.img_max)
        dyn = {"img_tokens": {1: nimg}, "txt_feats": {1: ntxt}, "timestep": None,
               "txt_cos": {1: ntxt}, "txt_sin": {1: ntxt}, "img_cos": {1: nimg}, "img_sin": {1: nimg}}
        t0 = time.time()
        prog = export_to_coreai(model, ref, dynamic_shapes=dyn, input_names=ORDER, output_names=("vel",))
        print(f"[export] converted in {time.time() - t0:.0f}s", flush=True)
        if not check:                       # the program holds its own copy of the weights now
            del model, ref
            gc.collect()
        if not args.no_optimize:
            t0 = time.time()
            prog.optimize()
            print(f"[export] optimized in {time.time() - t0:.0f}s", flush=True)
        shutil.rmtree(out_dir, ignore_errors=True)                 # save_asset does not overwrite
        out_dir.mkdir(parents=True)
        meta = rt.AIModelAssetMetadata()
        meta.license = "qwen-research"
        meta.model_description = (f"Qwen-Image-2.1 DiT ({tag}{', random weights' if args.random_init else ''}), "
                                  f"{args.dtype} weights+compute, fp32 I/O, dynamic text/image axes. "
                                  f"Source: {MODEL}@{REVISION[:7]}.")
        t0 = time.time()
        prog.save_asset(bundle, meta)
        size = sum(f.stat().st_size for f in bundle.rglob("*") if f.is_file())
        print(f"[export] saved {bundle} ({size / 2**30:.2f} GiB) in {time.time() - t0:.0f}s; "
              f"disk free {free_gib():.1f} GiB", flush=True)
        del prog

    if check:
        shapes = [tuple(int(v) for v in sh.split("x")) for sh in args.check_shapes.split(",")]
        res = asyncio.run(engine_check(bundle, model, shapes, args.seed + 2, args.cpu_only, args.controls))
        return 0 if all(r["ok"] for r in res) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
