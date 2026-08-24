#!/usr/bin/env python3
"""Qwen3.5-family host image preprocessing in NumPy — the spec any app-side code
reproduces, gated against the HF processor BEFORE any Swift exists.

Shared by every Qwen3.5-family vision bundle (Qwen3.8-27B, OvisOCR2): the tower
is a fixed grid, so only the tile size differs. The gate below runs the 27B's
512x512 tile; pass ``tile``/``tile_w`` for the others.

The vision bundle bakes ONE grid (the 27B: 32x32 patches = a 512x512 tile ->
256 merged tokens), so the host's whole job is:

    RGB uint8 [H,W,3] -> resize 512x512 (PIL BICUBIC, antialiased)
                      -> /255 -> (x-0.5)/0.5
                      -> patchify -> patches [1024, 1536] float32

Two Qwen-specific layout facts, both silent when wrong (fluent text about the
wrong image):

  * PATCH ORDER is merge-BLOCK-major: patches iterate (block_row, block_col,
    y-in-block, x-in-block) over 2x2 blocks — NOT plain row-major. This is the
    order the tower's baked positional constants assume.
  * The 1536-vector layout is (C, T=2, 16, 16) with the SAME 16x16 pixel patch
    repeated at both temporal slots (temporal_patch_size 2, still image), and
    the CHANNEL outermost — matching Conv3D weight [D,C,T,P,P].reshape(D,-1).

The resize filter is Pillow's antialiased BICUBIC (`resample: 3` — checked from
the checkpoint, never guessed); implementation shared with
`lfm25vl_preprocess.resize_antialias`. Aspect ratio is NOT preserved: the fixed
grid stretches non-square images, and the oracle suite is captured through the
same 512x512 tile, so gates measure the shipped path.

Run the gate (numpy + Pillow; suite fixture from qwen38vl_suite_ref.py):
    python3 _smoke/qwen38vl_preprocess.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lfm25vl_preprocess import BICUBIC, resize_antialias

TILE = 512
PATCH = 16
MERGE = 2
TEMPORAL = 2
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5
DEFAULT_SUITE = Path(__file__).parent / "qwen38vl_suite_512.npz"


def qwen_patchify(image: np.ndarray, patch: int = PATCH, merge: int = MERGE,
                  temporal: int = TEMPORAL) -> np.ndarray:
    """Normalized [H,W,C] -> [grid_h*grid_w, C*T*P*P] in Qwen's block-major order.

    Mirrors Qwen2VLImageProcessor._preprocess: reshape to
    (C, gh/m, m, P, gw/m, m, P), permute to (gh/m, gw/m, m, m, C, P, P), then
    repeat the still frame at both temporal slots.
    """
    h, w, c = image.shape
    if h % (patch * merge) or w % (patch * merge):
        raise ValueError(f"{h}x{w} not divisible by patch*merge {patch * merge}")
    gh, gw = h // patch, w // patch
    x = image.transpose(2, 0, 1)  # [C, H, W]
    x = x.reshape(c, gh // merge, merge, patch, gw // merge, merge, patch)
    x = x.transpose(1, 4, 2, 5, 0, 3, 6)  # [gh/m, gw/m, m, m, C, P, P]
    x = x.reshape(gh * gw, 1, c, patch, patch)
    x = np.broadcast_to(x[:, :, :, None], (gh * gw, 1, c, temporal, patch, patch))
    return x.reshape(gh * gw, c * temporal * patch * patch)


def preprocess(image: np.ndarray, tile: int = TILE,
               tile_w: int | None = None) -> np.ndarray:
    """RGB uint8 [H,W,3] -> patches [n_patch, 1536] float32 (block-major).

    ``tile`` is the target height and ``tile_w`` the width (default: square).
    A non-square tile is how a portrait document page keeps its aspect instead
    of being squashed into the square VLM tile — OvisOCR2 bakes 1280x896.
    """
    x = resize_antialias(image, tile, tile_w if tile_w is not None else tile, BICUBIC)
    x = (x / 255.0 - IMAGE_MEAN) / IMAGE_STD
    return qwen_patchify(x).astype(np.float32)


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=str(DEFAULT_SUITE))
    args = ap.parse_args()

    from PIL import Image

    suite = np.load(args.suite)
    n_cases = int(suite["_meta_cases"])
    failures: list[str] = []

    def check(name: str, got: np.ndarray, want: np.ndarray, tol_cos: float,
              tol_max: float) -> None:
        g = np.asarray(got, np.float64).reshape(-1)
        w = np.asarray(want, np.float64).reshape(-1)
        c = float(g @ w / (np.linalg.norm(g) * np.linalg.norm(w)))
        mx = float(np.abs(g - w).max())
        ok = c >= tol_cos and mx <= tol_max
        print(f"  {'PASS' if ok else 'FAIL'} {name:24s} cos {c:.7f}  max|d| {mx:.3e}")
        if not ok:
            failures.append(name)

    # A. the resize filter vs Pillow itself, on a real downscale (the suite's
    # tiles are already 512, so fabricate a 640x480 source). Pillow quantizes
    # its filter weights to 8-bit fixed point for uint8 images, and bicubic's
    # negative lobes amplify that to ~1 of 255 at edge pixels — so the bar is
    # max|d| <= 1.5 levels, not bit-exactness (the lfm25vl gate documents the
    # same effect at bilinear; the normalized-patch impact, 1/127.5, is far
    # inside the tower gate's cos >= 0.999).
    print("A. antialiased BICUBIC resize vs Pillow (640x480 -> 512x512)")
    img_dir = Path(__file__).parent / "qwen38vl_images"
    raws = sorted(img_dir.glob("*_512.png"))
    if raws:
        src = np.asarray(Image.open(raws[0]).convert("RGB").resize((640, 480), 2))
        pil = np.asarray(Image.fromarray(src).resize((TILE, TILE), BICUBIC), np.float64)
        mine = resize_antialias(src, TILE, TILE, BICUBIC)
        check("resize vs PIL", mine, pil, 0.999995, 1.5)

    # B. the whole host path vs the processor's own pixel_values, per case
    print("\nB. host patches vs processor pixel_values (the tensor the tower gates on)")
    for case in range(n_cases):
        img = suite[f"image{int(suite[f'case{case}_image_idx'])}_u8"]
        got = preprocess(img)
        want = suite[f"case{case}_patches"]
        if got.shape != want.shape:
            print(f"  FAIL case{case} shape {got.shape} vs {want.shape}")
            failures.append(f"case{case} shape")
            continue
        check(f"case{case} pixel_values", got, want, 0.999999, 1e-4)

    print("\nALL PASS" if not failures else f"\nFAILED: {', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
