#!/usr/bin/env python3
"""Gate: the authored Qwen3.5 tower at OvisOCR2's NON-SQUARE grid vs HF-fp32.

The shipped Qwen3.8-27B tower was only ever gated on a SQUARE grid (32x32
patches). OvisOCR2 bakes 80x56 — the first rectangular grid through this
authoring, so the row/col paths in `_init_positional_constants` (bilinear
pos-embed interpolation and the 2D rope coordinate build) are exercised here
for the first time. A row/col transposition is silent everywhere else.

The HF fp32 reference is captured into a fixture first, because the two halves
need DIFFERENT environments: the overlay venv's transformers has no `qwen3_5`
classes (the authored tower does not need them), so the oracle only runs under
a system python new enough to know the architecture. As in the 27B gate, the
target is HF **fp32** — never a bf16 full-model dump.

    python3 _smoke/test_ovisocr2_tower_gate.py <image> --capture-ref   # system python

  torch   — re-authored fp32 tower, eager CPU.       cos >= 0.999
  aimodel — exported fp16 .aimodel, GPU delegate.    cos >= 0.999
            (GPU run: hold _GPU_LOCK, python-GPU SOLO.)

A negative control runs in both stages: the same pixels in raster patch order
instead of merge-block-major. If that does NOT fail, the gate cannot go red and
its passes mean nothing.

The page: `_smoke/ovisocr2_jp_page.html` is the fixture's source (a Japanese A4
technical page — headings, justified body, a bordered table, a display formula,
a numbered list). Render it, then capture, then gate:

    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless \
      --disable-gpu --hide-scrollbars --window-size=794,1123 \
      --force-device-scale-factor=2 --screenshot=/tmp/jp_page.png \
      file://$PWD/_smoke/ovisocr2_jp_page.html
    python3 _smoke/test_ovisocr2_tower_gate.py /tmp/jp_page.png --capture-ref

Run (coreai-models/.venv):
    ../coreai-models/.venv/bin/python _smoke/test_ovisocr2_tower_gate.py <image>
    ../coreai-models/.venv/bin/python _smoke/test_ovisocr2_tower_gate.py <image> --stage aimodel
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

HF_ID = "ATH-MaaS/OvisOCR2"
TILE_H, TILE_W = 1280, 896      # -> 80x56 patches -> 40x28 = 1120 merged tokens
GRID_H, GRID_W = 40, 28
COS_BAR = 0.999
REF_NPZ = Path(__file__).parent / "ovisocr2_tower_fp32_ref.npz"


async def maybe_await(x):
    return await x if inspect.isawaitable(x) else x


def cos_stats(got: np.ndarray, want: np.ndarray) -> tuple[float, float]:
    g = got.astype(np.float64)
    w = want.astype(np.float64)
    c = float(g.ravel() @ w.ravel() / (np.linalg.norm(g) * np.linalg.norm(w)))
    row = (g * w).sum(-1) / (np.linalg.norm(g, axis=-1) * np.linalg.norm(w, axis=-1))
    return c, float(row.min())


def raster_patches(u8: np.ndarray) -> np.ndarray:
    """Negative control: same pixels, plain row-major patch order."""
    from lfm25vl_preprocess import BICUBIC, resize_antialias
    from qwen38vl_preprocess import IMAGE_MEAN, IMAGE_STD, PATCH, TEMPORAL
    x = resize_antialias(u8, TILE_H, TILE_W, BICUBIC)
    x = (x / 255.0 - IMAGE_MEAN) / IMAGE_STD
    h, w, c = x.shape
    gh, gw = h // PATCH, w // PATCH
    t = x.transpose(2, 0, 1).reshape(c, gh, PATCH, gw, PATCH)
    t = t.transpose(1, 3, 0, 2, 4)                       # [gh, gw, C, P, P] raster
    t = np.broadcast_to(t[:, :, None], (gh, gw, TEMPORAL, c, PATCH, PATCH))
    t = t.transpose(0, 1, 3, 2, 4, 5)
    return np.ascontiguousarray(
        t.reshape(gh * gw, c * TEMPORAL * PATCH * PATCH)).astype(np.float32)


def hf_reference(patches: np.ndarray) -> np.ndarray:
    import torch
    from transformers import AutoModelForImageTextToText
    m = AutoModelForImageTextToText.from_pretrained(HF_ID, dtype=torch.float32).eval()
    visual = m.model.visual
    thw = torch.tensor([[1, GRID_H * 2, GRID_W * 2]], dtype=torch.long)
    with torch.no_grad():
        out = visual(torch.from_numpy(patches).float(), grid_thw=thw)
    # NOTE: the merged rows the decoder consumes are `pooler_output`.
    # `last_hidden_state` is the PRE-merger [n_patch, 768] stack — gating on it
    # would skip the merger entirely and still look plausible.
    if hasattr(out, "pooler_output"):
        out = out.pooler_output
    elif isinstance(out, (tuple, list)):
        out = out[0]
    return out.squeeze().float().numpy()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--stage", default="torch", choices=["torch", "aimodel"])
    ap.add_argument("--capture-ref", action="store_true",
                    help="write the HF fp32 tower fixture (needs a transformers "
                         "that knows qwen3_5 — the system python, not the overlay venv)")
    ap.add_argument("--asset", default=None,
                    help="vision asset; default exports_dir()/ovisocr2_vision_fp16/...")
    ap.add_argument("--bench", type=int, default=10,
                    help="aimodel stage: timed encodes after the gate")
    args = ap.parse_args()

    from PIL import Image
    from qwen38vl_preprocess import preprocess

    u8 = np.asarray(Image.open(args.image).convert("RGB"))
    patches = preprocess(u8, TILE_H, TILE_W)
    assert patches.shape == (GRID_H * 2 * GRID_W * 2, 1536), patches.shape

    print(f"grid {GRID_H*2}x{GRID_W*2} patches -> {GRID_H}x{GRID_W} = "
          f"{GRID_H*GRID_W} merged tokens")

    if args.capture_ref:
        print("computing HF fp32 tower reference ...")
        want = hf_reference(patches)
        np.savez_compressed(REF_NPZ, embeds=want, patches=patches,
                            image=np.asarray(Image.open(args.image).convert("RGB")))
        print(f"  wrote {REF_NPZ} {want.shape}")
        return 0

    if not REF_NPZ.exists():
        print(f"missing {REF_NPZ} — run --capture-ref under the system python first")
        return 2
    ref = np.load(str(REF_NPZ))
    want = ref["embeds"]
    if not np.array_equal(ref["patches"], patches):
        print("FAIL: this image does not reproduce the fixture's patches")
        return 2
    print(f"  reference {want.shape} (fixture)")

    bad = 0
    if args.stage == "torch":
        import torch
        from coreai_models.models.macos.qwen3_5_vision import Qwen3_5VisionEncoder
        vis = Qwen3_5VisionEncoder.from_hf(
            HF_ID, target_dtype=torch.float32, grid_h=GRID_H, grid_w=GRID_W)
        with torch.no_grad():
            got = vis(torch.from_numpy(patches).float()).numpy()
            ctrl = vis(torch.from_numpy(raster_patches(u8)).float()).numpy()
    else:
        import time

        import coreai.runtime as rt
        opts = rt.SpecializationOptions.default()
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "conversion"))
        from _paths import exports_dir
        asset = Path(args.asset) if args.asset else (
            exports_dir() / "ovisocr2_vision_fp16" / "ovisocr2_vision_fp16.aimodel")
        m = await maybe_await(rt.AIModel.load(str(asset), opts))
        fn = await maybe_await(m.load_function(m.function_names[0]))
        print(f"  fn {fn.desc.name} in {fn.desc.input_names} out {fn.desc.output_names}")

        async def run(pt):
            out = await maybe_await(fn(inputs={"patches": rt.NDArray(
                np.ascontiguousarray(pt.astype(np.float16)))}))
            return np.asarray(out["image_embeds"].numpy())

        got = await run(patches)
        ctrl = await run(raster_patches(u8))

        ms = []
        for _ in range(args.bench):
            t0 = time.perf_counter()
            await run(patches)
            ms.append((time.perf_counter() - t0) * 1e3)
        if ms:
            ms.sort()
            print(f"  encode {args.bench}x: median {ms[len(ms) // 2]:.1f} ms "
                  f"(min {ms[0]:.1f}, max {ms[-1]:.1f})")

    c, rmin = cos_stats(got, want)
    ok = c >= COS_BAR and rmin >= COS_BAR
    print(f"  {'PASS' if ok else 'FAIL'} {args.stage:8s} cos {c:.6f}  min-row {rmin:.6f}")
    if not ok:
        # Show the shape of the miss, so a fp16 rounding tail is not read as a
        # structural break. A wiring bug moves EVERY row (see the neg-control);
        # fp16 noise leaves a handful in the 0.997-0.999 band.
        g64, w64 = got.astype(np.float64), want.astype(np.float64)
        rows = (g64 * w64).sum(-1) / (np.linalg.norm(g64, axis=-1)
                                      * np.linalg.norm(w64, axis=-1))
        below = rows < COS_BAR
        print(f"       rows below {COS_BAR}: {below.sum()}/{len(rows)}"
              f"   median row-cos {np.median(rows):.6f}")
    bad += 0 if ok else 1

    cc, crmin = cos_stats(ctrl, want)
    ctrl_ok = cc < COS_BAR          # the control MUST miss
    print(f"  {'PASS' if ctrl_ok else 'FAIL'} neg-control (raster patch order) "
          f"cos {cc:.6f}  min-row {crmin:.6f}  -- must be < {COS_BAR}")
    bad += 0 if ctrl_ok else 1

    print("\nALL PASS" if not bad else f"\nFAILED ({bad})")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
