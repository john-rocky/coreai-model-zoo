#!/usr/bin/env python3
"""Host side of the mask path in NumPy: Pillow's BICUBIC resize written out, and the decoder's input tensors.

The package resizes twice with `PIL.Image.resize(..., BICUBIC)`: the original photo to 1024 x 1024 for the mask
decoder's CNN canvas (`engine.py` `decode_masks`), and — in this port's fixed-grid form — to 448 / 512 for the
vision tower. `pil_bicubic` is Pillow's `ImagingResample` for 8-bit images (libImaging/Resample.c): the a = -0.5
cubic kernel widened by the downscale factor, coefficients normalized in double and rounded to 22-bit fixed point,
a horizontal pass to uint8, then a vertical pass to uint8 (`clip8`). Integer sums are exact, so the copy is meant
to be bit-equal to Pillow; `check` measures that on every fixture image and size.

    python3 -I mask_host.py check --fixtures <fixtures.json> [--fixtures <fixtures_r2.json>] --out <json>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

PRECISION_BITS = 32 - 8 - 2


def _bicubic(x: float) -> float:
    a = -0.5
    x = abs(x)
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1
    if x < 2.0:
        return (((x - 5) * x + 8) * x - 4) * a
    return 0.0


def _coeffs(in_size: int, out_size: int):
    """precompute_coeffs + normalize_coeffs_8bpc for the box (0, in_size)."""
    scale = float(np.float32(in_size) - np.float32(0.0)) / out_size
    filterscale = max(scale, 1.0)
    support = 2.0 * filterscale
    ksize = int(math.ceil(support)) * 2 + 1
    bounds = np.zeros((out_size, 2), np.int64)
    kk = np.zeros((out_size, ksize), np.int64)
    ss = 1.0 / filterscale
    for xx in range(out_size):
        center = 0.0 + (xx + 0.5) * scale
        xmin = int(center - support + 0.5)
        xmin = max(xmin, 0)
        xmax = int(center + support + 0.5)
        xmax = min(xmax, in_size) - xmin
        w = [_bicubic((x + xmin - center + 0.5) * ss) for x in range(xmax)]
        ww = 0.0
        for v in w:
            ww += v
        for x, v in enumerate(w):
            k = v / ww if ww != 0.0 else v
            kk[xx, x] = int(-0.5 + k * (1 << PRECISION_BITS)) if k < 0 else int(0.5 + k * (1 << PRECISION_BITS))
        bounds[xx] = (xmin, xmax)
    return bounds, kk


def _clip8(acc: np.ndarray) -> np.ndarray:
    out = np.right_shift(acc, PRECISION_BITS)
    out = np.where(acc >= (1 << PRECISION_BITS << 8), 255, out)
    out = np.where(acc <= 0, 0, out)
    return out.astype(np.uint8)


def _pass(src: np.ndarray, bounds: np.ndarray, kk: np.ndarray, axis: int) -> np.ndarray:
    """One 8-bit resample pass along `axis` (1 = horizontal over [H, W, C], 0 = vertical)."""
    a = np.moveaxis(src, axis, 0).astype(np.int64)            # [in, other, C]
    n_in = a.shape[0]
    acc = np.full((kk.shape[0],) + a.shape[1:], 1 << (PRECISION_BITS - 1), np.int64)
    for t in range(kk.shape[1]):
        idx = np.minimum(bounds[:, 0] + t, n_in - 1)          # past xmax the coefficient is 0
        acc += a[idx] * kk[:, t].reshape(-1, *([1] * (a.ndim - 1)))
    return np.moveaxis(_clip8(acc), 0, axis)


def pil_bicubic(rgb: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """uint8 [H, W, 3] -> uint8 [out_h, out_w, 3], as `Image.fromarray(rgb).resize((out_w, out_h), BICUBIC)`."""
    h, w = rgb.shape[:2]
    need_h, need_v = out_w != w, out_h != h
    x = rgb
    if need_h:
        bx, kx = _coeffs(w, out_w)
        x = _pass(x, bx, kx, axis=1)
    if need_v:
        by, ky = _coeffs(h, out_h)
        x = _pass(x, by, ky, axis=0)
    return np.ascontiguousarray(x)


def canvas(rgb: np.ndarray, side: int = 1024) -> np.ndarray:
    """The mask decoder's CNN canvas [3, side, side] float32 in [0, 1] (`_pil_to_float_rgb` of the resized photo)."""
    return (pil_bicubic(rgb, side, side).astype(np.float32) / 255.0).transpose(2, 0, 1).copy()


def patches(rgb_side: np.ndarray) -> np.ndarray:
    """The processor's pixel_values for an image already at a multiple of 32 (r1's `host_patches`): (x/255 - 0.5)/0.5,
    frame duplicated (temporal 2), block-major [H/32, W/32, 2, 2] patch order, each patch [C, T, 16, 16]."""
    x = rgb_side.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    h, w, _ = x.shape
    p, m = 16, 2
    gh, gw = h // p, w // p
    x = x.transpose(2, 0, 1)
    x = np.stack([x, x], axis=0)
    x = x.reshape(2, 3, gh // m, m, p, gw // m, m, p)
    x = x.transpose(2, 5, 3, 6, 1, 0, 4, 7)
    return x.reshape(gh * gw, 3 * 2 * p * p)


def check(args) -> int:
    from PIL import Image

    seen, rows = set(), []
    for fx in args.fixtures:
        base = Path(fx).resolve().parent
        for r in json.loads(Path(fx).read_text())["rows"]:
            if r["image"] in seen:
                continue
            seen.add(r["image"])
            im = Image.open(base / r["image"]).convert("RGB")
            arr = np.asarray(im)
            for side in (448, 512, 1024):
                ref = np.asarray(im.resize((side, side), resample=Image.Resampling.BICUBIC))
                got = pil_bicubic(arr, side, side)
                d = np.abs(ref.astype(np.int16) - got.astype(np.int16))
                rows.append({"image": r["image"], "size_wh": list(im.size), "side": side,
                             "bit_equal": bool(np.array_equal(ref, got)), "max_abs_d": int(d.max()),
                             "pixels_differ": int((d.max(axis=-1) > 0).sum())})
                print(f"[host] {r['image']} {im.size} -> {side}: bit-equal {rows[-1]['bit_equal']} max|d| "
                      f"{rows[-1]['max_abs_d']} ({rows[-1]['pixels_differ']} px)", flush=True)
    import PIL

    out = {"schema": "wevisone-r2-host-resize/1", "pillow": PIL.__version__, "rows": rows,
           "all_bit_equal": all(r["bit_equal"] for r in rows), "images": len(seen)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"[host] {len(rows)} resizes over {len(seen)} images, all bit-equal {out['all_bit_equal']}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["check"])
    ap.add_argument("--fixtures", action="append", required=True)
    ap.add_argument("--out", required=True)
    return check(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
