"""Export the Qwen-Image-2.1 text encoder (``qi21_text.QI21TextEncoder``) to a Core AI ``.aimodel``.

One function, ``main``:   input_ids [1,L] int32  ->  hidden [1,L,4096] fp32
L dynamic (``Dim("L", 16..512)``), no mask input (causal = SDPA ``is_causal``), ``embed_tokens``
inside the graph, bf16 weights + bf16 compute, fp32 output (``io_fp32`` — a Swift host cannot
read a bfloat16 NDArray). ``--r32`` keeps the residual stream in fp32 (norm statistics and the
residual adds; every matmul still bf16 x bf16); ``--w16a32`` stores bf16 weights and computes
everything in fp32 (the shipped variant) — see ``qi21_text.py`` for why.

Bundle: ``_paths.exports_dir()/qwenimage21/<name>/<name>.aimodel``,
name = ``qi21_encoder_dynL_{bf16|bf16_r32|w16a32}_ids_iofp32`` (probes: ``qi21_encoder_L<n>_dynL_...``).

``--aot`` compiles it for this Mac's GPU — the Python runtime's JIT put part of the full DiT graph
on the ANE and asserted (``ANERegion.mm:414 ... Code=-19``), plain AOT too; what runs is
``xcrun coreai-build compile <bundle> --platform macOS --architecture h16c --preferred-compute gpu
--expect-frequent-reshapes`` -> ``<name>_aot_efr/<name>.h16c.aimodelc`` (supervisor, 2026-09-25).

``--check`` (default for ``--random-init``) loads the AOT bundle with
``SpecializationOptions.default()`` (``--cpu-only``: the ``.aimodel`` on ``cpu_only()``, for
isolation), runs one forward per ``--check-L`` on random ids and compares it with the same
module in fp32 torch: NaN count, all-zero, corr, max|d|.

Run (coreai-models base venv — coreai-torch + coreai-opt; transformers is not imported):
  python export_encoder.py --layers 2 --random-init --aot            # probe, no checkpoint
  python export_encoder.py --aot --w16a32                            # full 36 layers, shipped variant
  python export_encoder.py --aot [--r32]                             # the bf16 / r32 controls
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import gc
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import exports_dir, hf_snapshot  # noqa: E402
from qi21_text import QI21TextEncoder, load_qi21_text, text_config  # noqa: E402

MODEL = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
DEFAULT_TEXT_CFG = dict(vocab_size=151936, hidden_size=4096, num_hidden_layers=36, num_attention_heads=32,
                        num_key_value_heads=8, head_dim=128, intermediate_size=12288, rms_norm_eps=1e-6,
                        rope_theta=5000000, hidden_act="silu", attention_bias=False)
MIN_FREE_GIB = 30


def free_gib() -> float:
    return shutil.disk_usage("/System/Volumes/Data").free / 2**30


def bundle_name(layers: int | None, variant: str) -> str:
    tag = "" if layers is None else f"L{layers}_"
    return f"qi21_encoder_{tag}dynL_{variant}_ids_iofp32"


def paths(name: str) -> tuple[Path, Path]:
    root = exports_dir() / "qwenimage21"
    return root / name / f"{name}.aimodel", root / f"{name}_aot_efr" / f"{name}.h16c.aimodelc"


def randomize(model, seed):
    """Unit-variance random weights (the r1 DiT probe's scheme)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            if name == "embed_tokens.weight":
                p.copy_(torch.randn(p.shape, generator=g))
            elif p.ndim == 2:
                p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(p.shape[1]))
            else:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))


def aot_compile(bundle: Path, aimodelc: Path) -> float:
    """xcrun coreai-build compile for this Mac's GPU (h16c, --expect-frequent-reshapes)."""
    out_dir = aimodelc.parent
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    cmd = ["xcrun", "coreai-build", "compile", str(bundle), "--output", str(out_dir), "--platform", "macOS",
           "--architecture", "h16c", "--preferred-compute", "gpu", "--expect-frequent-reshapes"]
    print(f"[aot] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    sec = time.time() - t0
    assert aimodelc.exists(), f"no {aimodelc} after compile: {sorted(p.name for p in out_dir.iterdir())}"
    size = sum(f.stat().st_size for f in aimodelc.rglob("*") if f.is_file())
    print(f"[aot] {aimodelc} ({size / 2**30:.2f} GiB) in {sec:.0f}s; disk free {free_gib():.1f} GiB", flush=True)
    return sec


def stats(a: np.ndarray, b: np.ndarray):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(np.abs(a - b).max()), float(np.corrcoef(a, b)[0, 1])


async def engine_check(path: Path, model, Ls, seed: int, cpu_only: bool):
    """One bundle load, then per L: engine forward vs the same module in fp32 torch."""
    import coreai.runtime as rt
    opts = rt.SpecializationOptions.cpu_only() if cpu_only else rt.SpecializationOptions.default()
    unit = "cpu_only" if cpu_only else "default(aot)"
    t0 = time.time()
    aim = await rt.AIModel.load(path, opts)
    fn = aim.load_function("main")
    print(f"[check] {unit}: load {time.time() - t0:.1f}s ({path.name})", flush=True)
    m32 = copy.deepcopy(model).to(torch.float32)
    vocab = model.embed_tokens.num_embeddings
    res = []
    for L in Ls:
        g = torch.Generator().manual_seed(seed + L)
        ids = torch.randint(0, vocab, (1, L), generator=g, dtype=torch.int32)
        t1 = time.time()
        r = await fn({"input_ids": rt.NDArray(ids.contiguous())})
        t_fwd = time.time() - t1
        out = np.array(r["hidden"].numpy(), dtype=np.float32)
        with torch.no_grad():
            ref = m32(ids).numpy()
        nan = int(np.isnan(out).sum())
        maxd, c = stats(out, ref) if nan == 0 else (float("nan"), float("nan"))
        zero = bool(np.abs(out).max() == 0) if nan < out.size else False
        ok = nan == 0 and not zero and c >= 0.999 and out.shape == ref.shape
        res.append(dict(L=L, nan=nan, zero=zero, corr=c, maxd=maxd, ok=ok, sec=t_fwd))
        print(f"[check] {unit} L={L}: fwd {t_fwd:.2f}s  shape {list(out.shape)}  NaN {nan}  zero {zero}  "
              f"corr {c:.6f}  max|d| {maxd:.3e}  |ref| max {float(np.abs(ref).max()):.3f}  "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    del aim
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=None, help="first n decoder layers only (probe)")
    ap.add_argument("--random-init", action="store_true", help="random weights, no checkpoint (probe)")
    vg = ap.add_mutually_exclusive_group()
    vg.add_argument("--r32", action="store_true", help="fp32 residual stream (bf16 weights + bf16 matmuls)")
    vg.add_argument("--w16a32", action="store_true", help="bf16 weights stored, fp32 compute (shipped)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L-min", type=int, default=16)
    ap.add_argument("--L-max", type=int, default=512)
    ap.add_argument("--trace-L", type=int, default=64, help="sequence length of the tracing input")
    ap.add_argument("--no-optimize", action="store_true")
    ap.add_argument("--aot", action="store_true", help="xcrun coreai-build compile (h16c, gpu, efr) after saving")
    ap.add_argument("--check", dest="check", action="store_true", default=None)
    ap.add_argument("--no-check", dest="check", action="store_false")
    ap.add_argument("--check-only", action="store_true", help="skip export (and AOT unless --aot), re-run --check")
    ap.add_argument("--cpu-only", action="store_true", help="--check the .aimodel with SpecializationOptions.cpu_only()")
    ap.add_argument("--check-L", default="32,128", help="comma list of L for --check")
    ap.add_argument("--text-encoder-dir", default=None, help="default: the pinned HF snapshot's text_encoder/")
    args = ap.parse_args()
    check = args.random_init if args.check is None else args.check

    variant = "w16a32" if args.w16a32 else ("bf16_r32" if args.r32 else "bf16")
    name = bundle_name(args.layers, variant)
    bundle, aimodelc = paths(name)
    t_start = time.time()
    if args.random_init:
        cfg = dict(DEFAULT_TEXT_CFG)
        if args.layers is not None:
            cfg["num_hidden_layers"] = args.layers
        model = QI21TextEncoder.from_config(cfg, max_len=args.L_max, io_fp32=True, residual_fp32=args.r32,
                                            compute_fp32=args.w16a32)
        randomize(model, args.seed)
        model = model.to(torch.bfloat16).eval()
    else:
        tdir = Path(args.text_encoder_dir or Path(hf_snapshot(MODEL, revision=REVISION)) / "text_encoder")
        print(f"[export] loading {tdir} ({variant}) ...", flush=True)
        model = load_qi21_text(tdir, dtype=torch.bfloat16, n_layers=args.layers, max_len=args.L_max,
                               io_fp32=True, residual_fp32=args.r32, compute_fp32=args.w16a32)
        assert text_config(tdir)["vocab_size"] == model.embed_tokens.num_embeddings
    assert model.rope_cos.dtype == torch.float32 and model.rope_cos.shape[0] == args.L_max
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export] {name}: {n_params / 1e9:.3f} B params, {len(model.layers)} layers, bf16, "
          f"residual {'fp32' if model.residual_fp32 else 'bf16'}, compute {'fp32' if model.compute_fp32 else 'bf16'}, "
          f"built in {time.time() - t_start:.0f}s", flush=True)

    if not args.check_only:
        fg = free_gib()
        print(f"[export] disk free {fg:.1f} GiB", flush=True)
        if fg < MIN_FREE_GIB:
            print(f"[export] STOP: free {fg:.1f} GiB < {MIN_FREE_GIB} GiB", flush=True)
            return 2
        from torch.export import Dim
        from coreai_models.export.macos import export_to_coreai
        import coreai.runtime as rt

        g = torch.Generator().manual_seed(args.seed + 1)
        ref = {"input_ids": torch.randint(0, model.embed_tokens.num_embeddings, (1, args.trace_L), generator=g,
                                          dtype=torch.int32)}
        dyn = {"input_ids": {1: Dim("L", min=args.L_min, max=args.L_max)}}
        t0 = time.time()
        prog = export_to_coreai(model, ref, dynamic_shapes=dyn, input_names=("input_ids",), output_names=("hidden",))
        print(f"[export] converted in {time.time() - t0:.0f}s", flush=True)
        if not check:                       # the program holds its own copy of the weights now
            del model, ref
            gc.collect()
        if not args.no_optimize:
            t0 = time.time()
            prog.optimize()
            print(f"[export] optimized in {time.time() - t0:.0f}s", flush=True)
        shutil.rmtree(bundle.parent, ignore_errors=True)            # save_asset does not overwrite
        bundle.parent.mkdir(parents=True)
        meta = rt.AIModelAssetMetadata()
        meta.license = "qwen-research"
        meta.model_description = (
            f"Qwen-Image-2.1 text encoder (Qwen3-VL-8B text stack, "
            f"{'first %d layers' % args.layers if args.layers else 'all 36 layers'}"
            f"{', random weights' if args.random_init else ''}), "
            + {"bf16": "bf16 weights and compute", "bf16_r32": "bf16 weights and matmuls, fp32 residual stream",
               "w16a32": "bf16 weights, fp32 compute"}[variant]
            + f", int32 input_ids -> fp32 hidden (last layer, "
            f"before the final norm), dynamic L {args.L_min}..{args.L_max}. Source: {MODEL}@{REVISION[:7]}.")
        t0 = time.time()
        prog.save_asset(bundle, meta)
        size = sum(f.stat().st_size for f in bundle.rglob("*") if f.is_file())
        print(f"[export] saved {bundle} ({size / 2**30:.2f} GiB) in {time.time() - t0:.0f}s; "
              f"disk free {free_gib():.1f} GiB", flush=True)
        del prog
        gc.collect()

    if args.aot and not args.cpu_only:
        fg = free_gib()
        if fg < MIN_FREE_GIB:
            print(f"[aot] STOP: free {fg:.1f} GiB < {MIN_FREE_GIB} GiB", flush=True)
            return 2
        aot_compile(bundle, aimodelc)

    if check:
        path = bundle if args.cpu_only else aimodelc
        Ls = [int(v) for v in args.check_L.split(",")]
        res = asyncio.run(engine_check(path, model, Ls, args.seed + 2, args.cpu_only))
        print(f"[export] total {time.time() - t_start:.0f}s", flush=True)
        return 0 if all(r["ok"] for r in res) else 1
    print(f"[export] total {time.time() - t_start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
