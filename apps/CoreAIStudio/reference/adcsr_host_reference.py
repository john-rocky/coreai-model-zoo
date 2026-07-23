#!/usr/bin/env python3
"""AdcSR host pipeline — NumPy reference for `Sources/UpscaleEngine.swift`.

Per PORTING.md §5 (Gate B): host processing gets implemented in NumPy FIRST, gated, THEN
translated to Swift — host-side mismatches are the #1 source of "the graph is perfect but the
output is garbage," and are unfindable once the only implementation is inside an app. This file
is that NumPy-first step for CoreAIStudio's native Upscale mode.

**This is an AUTHORED reference, not a reverse-engineered clone of CoreAIKitVision's
`SuperResolver`.** That type is closed-source (external `coreai-kit` package, not checked out
in this repo) — `knowledge/adcsr-super-resolution.md` documents its BEHAVIOR (tile 128→512,
`maxInputSide=512`, feather-blend overlapping tiles, then a GLOBAL per-image color-match applied
once after stitching — never per-tile, which divides by a near-zero std on uniform tiles and
produces pure-white squares) but not its exact tile stride / feather-curve shape / color-match
formula. The choices below (tile stride, linear feather ramp, mean/std channel match) are this
port's own, reasonable implementation of the documented CONTRACT — gated for internal
correctness (coverage, finiteness, and that the color-match actually does what it claims), not
against the original's exact pixel values, which aren't available to compare against.

Graph contract (`conversion/export_adcsr.py`, ✓ VERIFIED by reading the actual export code, not
just its docstring): `lr [1,3,128,128]` in `[-1,1]` → `sr [1,3,512,512]`, no in-graph
normalization — the host does `rgb01 * 2 - 1` in and `(sr + 1) / 2` (clamped) out.

Usage:
  python adcsr_host_reference.py --bundle adcsr_x4_float32.aimodel --image photo.jpg --out sr.png
  python adcsr_host_reference.py --bundle adcsr_x4_float32.aimodel --self-test
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np

TILE = 128
SCALE = 4
SR_TILE = TILE * SCALE
MAX_INPUT_SIDE = 512
OVERLAP = 16  # LR-space px; authored choice, see module docstring
STRIDE = TILE - OVERLAP


def cap_max_side(rgb01: np.ndarray, max_side: int = MAX_INPUT_SIDE) -> np.ndarray:
    """Downscale (area-average) so the long side <= max_side; upscale (bilinear-ish, via PIL)
    to at least TILE if smaller (tiling needs at least one full tile). No-op otherwise.
    """
    from PIL import Image

    h, w = rgb01.shape[:2]
    long_side = max(h, w)
    scale = 1.0
    if long_side > max_side:
        scale = max_side / long_side
    elif min(h, w) < TILE:
        scale = TILE / min(h, w)
    if scale == 1.0:
        return rgb01
    new_h, new_w = max(TILE, round(h * scale)), max(TILE, round(w * scale))
    img = Image.fromarray((np.clip(rgb01, 0, 1) * 255 + 0.5).astype(np.uint8))
    resample = Image.BILINEAR if scale > 1 else Image.BOX
    img = img.resize((new_w, new_h), resample)
    return np.asarray(img).astype(np.float32) / 255.0


def tile_origins(size: int, tile: int, stride: int) -> list[int]:
    """Tile top-left offsets covering [0, size), every tile fully inside bounds, last tile
    clamped flush with the far edge (standard overlap-tile coverage — no gaps, no padding).
    """
    if size <= tile:
        return [0]
    origins = list(range(0, size - tile + 1, stride))
    if origins[-1] != size - tile:
        origins.append(size - tile)
    return origins


def _feather_axis(tile_px: int, ramp: int, has_prev: bool, has_next: bool) -> np.ndarray:
    """1D weight for one axis: linear 0->1 ramp only on edges that have a NEIGHBORING tile to
    blend into. An edge on the true image boundary (no neighbor) keeps full weight — ramping it
    down anyway is the bug the gate caught: every tile's outer border would drop to 0, leaving
    the whole image's outer rim at weight_sum == 0 (a real hole, not a rounding artifact).
    """
    w = np.ones(tile_px, dtype=np.float32)
    if ramp > 0 and has_prev:
        w[:ramp] = np.linspace(0.0, 1.0, ramp, endpoint=False, dtype=np.float32)
    if ramp > 0 and has_next:
        w[-ramp:] = np.linspace(0.0, 1.0, ramp, endpoint=False, dtype=np.float32)[::-1]
    return w


def feather_weight(
    tile_px: int, overlap_px: int, has_left: bool, has_right: bool, has_top: bool, has_bottom: bool
) -> np.ndarray:
    """2D separable linear feather for ONE tile at a specific grid position — edge-aware, per
    the note in `_feather_axis`. Ramp width in SR space = overlap_px * SCALE (the overlap is
    defined in LR space; the blend happens on the SR canvas).
    """
    ramp = overlap_px * SCALE
    wx = _feather_axis(tile_px, ramp, has_left, has_right)
    wy = _feather_axis(tile_px, ramp, has_top, has_bottom)
    return wy[:, None] * wx[None, :]


async def run_tile(fn, lr_tile01: np.ndarray) -> np.ndarray:
    """lr_tile01: [TILE,TILE,3] float32 in [0,1] -> sr_tile01: [SR_TILE,SR_TILE,3] in [0,1]."""
    import coreai.runtime as rt

    x = lr_tile01 * 2.0 - 1.0  # [0,1] -> [-1,1]
    x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)  # HWC -> [1,3,H,W]
    out = await fn({"lr": rt.NDArray(x)})
    sr = out["sr"].numpy().astype(np.float32)[0]  # [3,SR,SR]
    sr = np.transpose(sr, (1, 2, 0))  # -> HWC
    return np.clip((sr + 1.0) / 2.0, 0.0, 1.0)


async def upscale_x4(fn, rgb01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """rgb01: [H,W,3] float32 in [0,1] -> (sr [H*4,W*4,3] in [0,1], weight_sum [H*4,W*4]).

    weight_sum is returned for the gate (coverage sanity) — a real UpscaleEngine caller only
    needs the first element.
    """
    capped = cap_max_side(rgb01)
    h, w = capped.shape[:2]
    xs, ys = tile_origins(w, TILE, STRIDE), tile_origins(h, TILE, STRIDE)

    canvas = np.zeros((h * SCALE, w * SCALE, 3), dtype=np.float32)
    weight_sum = np.zeros((h * SCALE, w * SCALE), dtype=np.float32)

    for iy, oy in enumerate(ys):
        for ix, ox in enumerate(xs):
            tile = capped[oy:oy + TILE, ox:ox + TILE]
            sr_tile = await run_tile(fn, tile)
            feather = feather_weight(
                SR_TILE, OVERLAP,
                has_left=ix > 0, has_right=ix < len(xs) - 1,
                has_top=iy > 0, has_bottom=iy < len(ys) - 1,
            )
            sy, sx = oy * SCALE, ox * SCALE
            canvas[sy:sy + SR_TILE, sx:sx + SR_TILE] += sr_tile * feather[..., None]
            weight_sum[sy:sy + SR_TILE, sx:sx + SR_TILE] += feather

    blended = canvas / np.clip(weight_sum, 1e-6, None)[..., None]

    # GLOBAL per-channel color-match, once, after stitching (not per-tile — see module
    # docstring for why per-tile color-match produces pure-white squares on uniform tiles).
    matched = np.empty_like(blended)
    for c in range(3):
        lr_mean, lr_std = capped[..., c].mean(), capped[..., c].std() + 1e-6
        sr_mean, sr_std = blended[..., c].mean(), blended[..., c].std() + 1e-6
        matched[..., c] = (blended[..., c] - sr_mean) / sr_std * lr_std + lr_mean
    matched = np.clip(matched, 0.0, 1.0)
    return matched, weight_sum


# ---------------------------------------------------------------------------
# Gate: internal correctness (coverage, finiteness, color-match does what it claims).
# NOT a comparison against the original closed-source SuperResolver — see module docstring.
# ---------------------------------------------------------------------------
async def self_test(fn) -> bool:
    rng = np.random.default_rng(0)
    # Deliberately non-square, not a multiple of TILE-STRIDE, and includes both a smooth
    # (uniform) region and noise — a uniform region is exactly the case the GLOBAL (not
    # per-tile) color-match rule exists to protect against div-by-near-zero std blowups.
    img = np.zeros((150, 230, 3), dtype=np.float32)
    img[:, :] = 0.6
    img[40:110, 60:180] = rng.random((70, 120, 3)).astype(np.float32)

    sr, weight_sum = await upscale_x4(fn, img)
    ok = True

    h, w = min(150, MAX_INPUT_SIDE), min(230, MAX_INPUT_SIDE)
    expect_shape = (h * SCALE, w * SCALE, 3)
    shape_ok = sr.shape == expect_shape
    ok &= shape_ok
    print(f"[gate] shape: got {sr.shape}, expect {expect_shape} -> {'PASS' if shape_ok else 'FAIL'}")

    finite_ok = bool(np.isfinite(sr).all()) and float(sr.min()) >= 0.0 and float(sr.max()) <= 1.0
    ok &= finite_ok
    print(f"[gate] finite + in-range [0,1]: min={sr.min():.4f} max={sr.max():.4f} -> "
          f"{'PASS' if finite_ok else 'FAIL'}")

    coverage_ok = bool((weight_sum > 0).all())
    ok &= coverage_ok
    print(f"[gate] full tile coverage (no holes): min weight={weight_sum.min():.4f} -> "
          f"{'PASS' if coverage_ok else 'FAIL'}")

    # Color-match self-consistency: per-channel mean/std of the SR output should equal the
    # (capped) LR's, within a small tolerance — this is the actual claim the algorithm makes,
    # and it's directly checkable without an external reference.
    capped = cap_max_side(img)
    match_ok = True
    for c in range(3):
        lr_mean, lr_std = capped[..., c].mean(), capped[..., c].std()
        sr_mean, sr_std = sr[..., c].mean(), sr[..., c].std()
        d_mean, d_std = abs(sr_mean - lr_mean), abs(sr_std - lr_std)
        c_ok = d_mean < 0.05 and d_std < 0.05  # clipping at [0,1] prevents an exact match
        match_ok &= c_ok
        print(f"[gate] channel {c} color-match: |Δmean|={d_mean:.4f} |Δstd|={d_std:.4f} -> "
              f"{'PASS' if c_ok else 'FAIL'}")
    ok &= match_ok

    print(f"\n=== {'ALL PASS' if ok else 'FAIL'} (self-consistency gate — see module docstring)")
    return ok


async def main_async():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--out", default="sr_out.png")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--unit", default="gpu", help="gpu | neural_engine | cpu")
    args = ap.parse_args()

    import coreai.runtime as rt

    opts = (
        rt.SpecializationOptions.cpu_only()
        if args.unit == "cpu"
        else rt.SpecializationOptions.from_preferred_compute_unit_kind(getattr(rt.ComputeUnitKind, args.unit)())
    )
    model = await rt.AIModel.load(args.bundle, opts)
    fn = model.load_function("main")

    if args.self_test:
        ok = await self_test(fn)
        raise SystemExit(0 if ok else 1)

    if not args.image:
        raise SystemExit("pass --image or --self-test")

    from PIL import Image

    img = Image.open(args.image).convert("RGB")
    rgb01 = np.asarray(img).astype(np.float32) / 255.0
    sr, _ = await upscale_x4(fn, rgb01)
    out_img = Image.fromarray((sr * 255 + 0.5).astype(np.uint8))
    out_img.save(args.out)
    print(f"[out] {args.image} {img.size} -> {args.out} {out_img.size}")


if __name__ == "__main__":
    asyncio.run(main_async())
