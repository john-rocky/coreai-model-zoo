"""Gate: the EXPORTED Core AI VAE decoder bundle vs the fp32 oracle.

The oracle's ``final_latents [1,N,64]`` (the sampler output, normalised) -> the bundle's ``main``
-> ``image [1,4,H,W]`` vs the oracle's ``vae_out[:, :, 0]`` (fp32, clamped, [-1,1]). Scored over the
whole tensor and per channel (R, G, B, A). Bar, every channel: corr >= 0.9999, max|d| <= 1e-2,
NaN 0. Also reported: the uint8 PNG the pipeline would write (``(x*0.5+0.5).clamp(0,1)*255`` rounded,
RGBA) vs ``image_ref.png`` — PSNR over RGBA and the max byte difference.

Engine output is saved to ``_work/engine_vae/<bundle stem>_<unit>/image_<oracle>.f32``.

Run (coreai-models base venv, from conversion/qwenimage21/):
  python engine_parity_vae.py <name>.aimodel --cpu-only --oracle oracle/256      # isolation
  python engine_parity_vae.py <name>.h16c.aimodelc --oracle oracle/256           # AOT efr, default()
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
BAR_CORR, BAR_MAXD = 0.9999, 1e-2
CH = ("R", "G", "B", "A")


def score(out: np.ndarray, ref: np.ndarray) -> dict:
    a = out.astype(np.float64).ravel()
    b = ref.astype(np.float64).ravel()
    nan = int(np.isnan(a).sum())
    if nan:
        return dict(nan=nan, corr=float("nan"), maxd=float("nan"), rel_l2=float("nan"))
    return dict(nan=0, corr=float(np.corrcoef(a, b)[0, 1]), maxd=float(np.abs(a - b).max()),
                mean_abs=float(np.abs(a - b).mean()), rel_l2=float(np.linalg.norm(a - b) / np.linalg.norm(b)))


def to_u8(img: np.ndarray) -> np.ndarray:
    """[1,4,H,W] in [-1,1] -> [H,W,4] uint8, as ``VaeImageProcessor.postprocess`` + ``numpy_to_pil``."""
    x = np.clip(img[0].astype(np.float32) * 0.5 + 0.5, 0.0, 1.0).transpose(1, 2, 0)
    return (x * 255).round().astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return 99.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


async def run_engine(bundle: Path, cpu_only: bool, x: np.ndarray, warm: int):
    import coreai.runtime as rt
    import torch
    opts = rt.SpecializationOptions.cpu_only() if cpu_only else rt.SpecializationOptions.default()
    t0 = time.time()
    model = await rt.AIModel.load(bundle, opts)            # keep the AIModel alive while calling
    fn = model.load_function("main")
    t_load = time.time() - t0
    feed = {"latents_packed": rt.NDArray(torch.from_numpy(x).contiguous())}
    secs, out = [], None
    for _ in range(1 + warm):
        t1 = time.time()
        r = await fn(feed)
        out = np.array(r["image"].numpy(), dtype=np.float32)
        secs.append(time.time() - t1)
    del fn, model
    return out, t_load, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", help="<name>.h16c.aimodelc (default options) or <name>.aimodel with --cpu-only")
    ap.add_argument("--oracle", default=str(HERE / "oracle" / "256"))
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--warm", type=int, default=2, help="extra calls after the first (timing)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    bundle = Path(args.bundle)
    unit = "cpu_only" if args.cpu_only else ("default(aot)" if bundle.suffix == ".aimodelc" else "default(jit)")
    stem = bundle.name.split(".")[0]
    od = Path(args.oracle)
    meta = json.load(open(od / "meta.json"))
    _, h, w = (int(v) for v in meta["img_shapes"][-1])
    x = np.fromfile(od / "final_latents.f32", "<f4").reshape(1, h * w, 64)
    ref = np.fromfile(od / "vae_out.f32", "<f4").reshape(meta["vae_out_shape"])[:, :, 0]
    print(f"[vae-parity] {bundle.name} ({unit}); oracle {od.name}: final_latents {list(x.shape)} -> "
          f"vae_out[:, :, 0] {list(ref.shape)}", flush=True)

    out, t_load, secs = asyncio.run(run_engine(bundle, args.cpu_only, x, args.warm))
    print(f"[vae-parity] load {t_load:.1f}s; calls " + " / ".join(f"{s:.2f}s" for s in secs), flush=True)
    sd = HERE / "_work" / "engine_vae" / f"{stem}_{unit.split('(')[0]}"
    sd.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(out, "<f4").tofile(sd / f"image_{od.name}.f32")
    assert out.shape == ref.shape, (out.shape, ref.shape)

    res = dict(bundle=str(bundle), unit=unit, oracle=str(od), load_s=t_load, call_s=secs,
               all=score(out, ref), channels={})
    ok = True
    for c, name in enumerate(CH):
        s = score(out[:, c], ref[:, c])
        s["ok"] = s["nan"] == 0 and s["corr"] >= BAR_CORR and s["maxd"] <= BAR_MAXD
        res["channels"][name] = s
        ok &= s["ok"]
        rng = (float(ref[:, c].min()), float(ref[:, c].max()))
        print(f"  {name}: corr {s['corr']:.7f}  max|d| {s['maxd']:.3e}  mean|d| {s.get('mean_abs', float('nan')):.3e}  "
              f"|d|/|ref| {s['rel_l2']:.3e}  NaN {s['nan']}  (ref range {rng[0]:+.3f}..{rng[1]:+.3f})  "
              f"{'PASS' if s['ok'] else 'FAIL'}", flush=True)
    a = res["all"]
    print(f"  all: corr {a['corr']:.7f}  max|d| {a['maxd']:.3e}  |d|/|ref| {a['rel_l2']:.3e}  NaN {a['nan']}", flush=True)

    ref_png = od / "image_ref.png"
    if ref_png.exists() and a["nan"] == 0:
        from PIL import Image
        r8 = np.asarray(Image.open(ref_png))
        o8 = to_u8(out)
        assert r8.shape == o8.shape, (r8.shape, o8.shape)
        d8 = np.abs(o8.astype(np.int16) - r8.astype(np.int16))
        res["png"] = dict(psnr_rgba=psnr(o8, r8), psnr_rgb=psnr(o8[..., :3], r8[..., :3]), max_byte_diff=int(d8.max()),
                          frac_bytes_diff=float((d8 > 0).mean()), alpha_min=int(o8[..., 3].min()),
                          ref_alpha_min=int(r8[..., 3].min()))
        p = res["png"]
        print(f"  uint8 vs image_ref.png: PSNR RGBA {p['psnr_rgba']:.2f} dB, RGB {p['psnr_rgb']:.2f} dB, "
              f"max byte diff {p['max_byte_diff']}, {100 * p['frac_bytes_diff']:.3f}% of bytes differ; "
              f"alpha min {p['alpha_min']} (ref {p['ref_alpha_min']})", flush=True)
    res["ok"] = bool(ok)
    print(f"\n[vae-parity] {'PASS' if ok else 'FAIL'} (bar per channel R/G/B/A: corr >= {BAR_CORR}, "
          f"max|d| <= {BAR_MAXD}, NaN 0)", flush=True)
    js = Path(args.json or HERE / "_work" / f"engine_parity_vae_{stem}_{od.name}_{unit.split('(')[0]}.json")
    json.dump(res, open(js, "w"), indent=2)
    print(f"[vae-parity] -> {js}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
