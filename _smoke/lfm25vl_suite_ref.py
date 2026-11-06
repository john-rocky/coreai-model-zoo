#!/usr/bin/env python3
"""Multi-case fp32 oracle for LFM2.5-VL — the suite the COMPRESSED bundle is judged on.

`lfm25vl_ref.py` captures one image and one prompt with every intermediate, which
is what proves the port is wired correctly. It cannot answer the other question:
whether int8 weights change what the model SAYS. One prompt agreeing 48/48 is
one prompt; a cosine on one logits vector is not a quality measure at all.

So this writes a small suite — a few images x a few coarse questions, at the
FIXED 512x512 grid the bundle bakes — keeping only what a token-exactness gate
needs: the processor's patches, the prompt ids, and the fp32 greedy ids.

Coarse questions on purpose: at 450M this family is documented (and measured, on
the LiteRT side) to miss fine-grained shape/geometry questions while answering
scene-level ones, so a suite of fine-grained probes would measure the checkpoint,
not the port.

Run:
    ~/code/litertlm-convert/.venv-vl093/bin/python _smoke/lfm25vl_suite_ref.py \
        [--hf-id LiquidAI/LFM2.5-VL-450M] [--tile 512] [--max-new-tokens 48]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

FP32 = torch.float32
DEFAULT_ID = "LiquidAI/LFM2.5-VL-450M"

# Fixed COCO val2017 images: two cats on a couch, two kitchens.
IMAGES = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000397133.jpg",
    "http://images.cocodataset.org/val2017/000000037777.jpg",
]
PROMPTS = [
    "What is in this image?",
    "Describe the main colors in this image.",
    "Where is this scene?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default=DEFAULT_ID)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import transformers

    if int(transformers.__version__.split(".")[0]) < 5:
        raise SystemExit(
            f"transformers {transformers.__version__} applies the projector LayerNorm "
            "unconditionally; see lfm25vl_ref.py. Use a transformers>=5 interpreter."
        )

    import requests
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    slug = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    out = Path(args.out or Path(__file__).parent / f"{slug}_suite_{args.tile}.npz")

    processor = AutoProcessor.from_pretrained(args.hf_id)
    print(f"loading {args.hf_id} fp32 ...")
    model = AutoModelForImageTextToText.from_pretrained(args.hf_id, dtype=FP32)
    model.eval()

    saved: dict[str, np.ndarray] = {}
    texts: list[str] = []
    case = 0
    # The checkpoint's own resampler (450M: BILINEAR, 3B: BICUBIC) — the host will use
    # the same one, so the fixture has to.
    resample = int(processor.image_processor.resample)
    print(f"resample {resample} (from processor_config)")
    for url in IMAGES:
        raw = Image.open(requests.get(url, stream=True, timeout=60).raw).convert("RGB")
        image = raw.resize((args.tile, args.tile), resample)
        for prompt in PROMPTS:
            conversation = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]
            inputs = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            )
            n_real = int(inputs["pixel_attention_mask"][0].sum())
            grid = tuple(int(v) for v in inputs["spatial_shapes"][0])
            expect = (args.tile // 16, args.tile // 16)
            if grid != expect or n_real != expect[0] * expect[1]:
                raise SystemExit(
                    f"case {case}: processor emitted grid {grid} / {n_real} real patches, "
                    f"expected {expect} fully packed -- the bundle bakes that grid"
                )
            with torch.no_grad():
                gen = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
                )
            new_ids = gen[0, inputs["input_ids"].shape[1]:]
            text = processor.decode(new_ids, skip_special_tokens=True)
            texts.append(text)
            saved[f"case{case}_patches"] = (
                inputs["pixel_values"][0].to(torch.float16).numpy()
            )
            saved[f"case{case}_ids"] = inputs["input_ids"][0].numpy().astype(np.int32)
            saved[f"case{case}_gen"] = new_ids.numpy().astype(np.int32)
            print(f"case {case}: {url.rsplit('/', 1)[-1]} | {prompt!r}\n  -> {text!r}")
            case += 1

    saved["_meta_hf_id"] = np.array(args.hf_id)
    saved["_meta_transformers"] = np.array(transformers.__version__)
    saved["_meta_tile"] = np.array(args.tile)
    saved["_meta_cases"] = np.array(case)
    saved["_meta_resample"] = np.array(resample)
    saved["_meta_texts"] = np.array(texts)
    np.savez(out, **saved)
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB, {case} cases)")


if __name__ == "__main__":
    main()
