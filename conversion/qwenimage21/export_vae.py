"""Export the Qwen-Image-2.1 VAE decoder (``qi21_vae.QI21VAEDecode``) to a Core AI ``.aimodel``, one size per graph.

One function, ``main``:   latents_packed [1,N,64] fp32  ->  image [1,4,H,W] fp32 (RGBA, [-1,1])
N = (size/16)^2; the unpack, the ``* std + mean`` un-normalisation, the one-frame decode, the clamp
and the ``[:, :, 0]`` are all in the graph. fp32 weights and compute (the VAE runs once per image).
Fixed shape per size — ``qi21_vae_{256|512|1024}_fp32`` (a dynamic latent side hit ``Constraints
violated`` in the Z-Image VAE). ``--patch-upsample`` swaps the ``nearest-exact`` 2x upsample for
``repeat_interleave`` (bit-exact in torch, ``parity_vae_torch.py`` (c)).

Built with the plain skeleton (``coreai_models.export.macos`` needs ``coreai_opt``, which cannot
share this venv with diffusers main): ``torch.export`` -> ``run_decompositions(get_decomp_table())``
-> ``TorchConverter().add_exported_program`` -> ``to_coreai()`` -> ``optimize()`` -> ``save_asset``.

``--aot`` compiles it for this Mac's GPU: ``xcrun coreai-build compile <bundle> --platform macOS
--architecture h16c --preferred-compute gpu --expect-frequent-reshapes``
-> ``<name>_aot_efr/<name>.h16c.aimodelc`` (the recipe the DiT and encoder bundles run on).

Bundle: ``_paths.exports_dir()/qwenimage21/<name>/<name>.aimodel``.

Run (``.venv-qi21``, from conversion/qwenimage21/):
  python export_vae.py --size 256 --aot [--patch-upsample]
"""
from __future__ import annotations

import argparse
import gc
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import exports_dir  # noqa: E402
from qi21_vae import MODEL, REVISION, QI21VAEDecode, load_vae, patch_nearest_upsample  # noqa: E402

MIN_FREE_GIB = 30


def free_gib() -> float:
    return shutil.disk_usage("/System/Volumes/Data").free / 2**30


def paths(name: str) -> tuple[Path, Path]:
    root = exports_dir() / "qwenimage21"
    return root / name / f"{name}.aimodel", root / f"{name}_aot_efr" / f"{name}.h16c.aimodelc"


def dir_gib(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2**30


def aot_compile(bundle: Path, aimodelc: Path) -> float:
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
    print(f"[aot] {aimodelc} ({dir_gib(aimodelc):.2f} GiB) in {sec:.0f}s; disk free {free_gib():.1f} GiB", flush=True)
    return sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True, help="image side in px (multiple of 32)")
    ap.add_argument("--patch-upsample", action="store_true", help="nearest-exact x2 -> repeat_interleave")
    ap.add_argument("--no-optimize", action="store_true")
    ap.add_argument("--aot", action="store_true", help="xcrun coreai-build compile (h16c, gpu, efr) after saving")
    ap.add_argument("--aot-only", action="store_true", help="skip the export, compile the saved bundle")
    args = ap.parse_args()
    assert args.size % 32 == 0, args.size
    lat = args.size // 16
    name = f"qi21_vae_{args.size}_fp32"
    bundle, aimodelc = paths(name)
    t_start = time.time()

    if not args.aot_only:
        fg = free_gib()
        print(f"[export] {name}: latents_packed [1,{lat * lat},64] -> image [1,4,{args.size},{args.size}]; "
              f"disk free {fg:.1f} GiB", flush=True)
        if fg < MIN_FREE_GIB:
            print(f"[export] STOP: free {fg:.1f} GiB < {MIN_FREE_GIB} GiB", flush=True)
            return 2
        import coreai.runtime as rt
        from coreai_torch import TorchConverter, get_decomp_table

        vae = load_vae(dtype=torch.float32)
        if args.patch_upsample:
            print(f"[export] patched {patch_nearest_upsample(vae.decoder)} nearest-exact upsamples", flush=True)
        wrap = QI21VAEDecode(vae, lat, lat).eval()
        n_params = sum(p.numel() for p in list(vae.decoder.parameters()) + list(vae.post_quant_conv.parameters()))
        print(f"[export] decoder + post_quant_conv {n_params / 1e6:.1f} M params fp32", flush=True)
        ref = {"latents_packed": torch.randn(1, lat * lat, 64, generator=torch.Generator().manual_seed(0))}
        t0 = time.time()
        with torch.no_grad():
            ep = torch.export.export(wrap, args=(), kwargs=ref)
        print(f"[export] torch.export in {time.time() - t0:.0f}s", flush=True)
        t0 = time.time()
        ep = ep.run_decompositions(get_decomp_table())
        prog = (TorchConverter()
                .add_exported_program(exported_program=ep, input_names=["latents_packed"], output_names=["image"])
                .to_coreai())
        print(f"[export] converted in {time.time() - t0:.0f}s", flush=True)
        del ep
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
            f"Qwen-Image-2.1 VAE decoder, {args.size}x{args.size}, fp32: latents_packed [1,{lat * lat},64] "
            f"(sampler output, normalised) -> image [1,4,{args.size},{args.size}] RGBA in [-1,1]; unpack and "
            f"latents*std+mean inside the graph. Source: {MODEL}@{REVISION[:7]}.")
        t0 = time.time()
        prog.save_asset(bundle, meta)
        print(f"[export] saved {bundle} ({dir_gib(bundle.parent):.2f} GiB) in {time.time() - t0:.0f}s; "
              f"disk free {free_gib():.1f} GiB", flush=True)
        del prog
        gc.collect()

    if args.aot or args.aot_only:
        fg = free_gib()
        if fg < MIN_FREE_GIB:
            print(f"[aot] STOP: free {fg:.1f} GiB < {MIN_FREE_GIB} GiB", flush=True)
            return 2
        aot_compile(bundle, aimodelc)
    print(f"[export] total {time.time() - t_start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
