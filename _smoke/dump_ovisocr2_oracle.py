#!/usr/bin/env python3
"""Capture the OvisOCR2 greedy oracle AT THE SHIPPED GRID, for the suite gate.

Runs under the SYSTEM python (the overlay venv's transformers has no `qwen3_5`).
The image is pre-resized to the bundle's 1280x896 tile so the processor cannot
pick its own grid — the oracle must see the tensor the bundle sees, or it is
measuring a different model.

    python3 _smoke/dump_ovisocr2_oracle.py <image> [--max-new 128]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent))
from qwen38vl_preprocess import preprocess  # noqa: E402

HF_ID = "ATH-MaaS/OvisOCR2"
TILE_H, TILE_W = 1280, 896
OUT = Path(__file__).parent / "ovisocr2_oracle.npz"
EOS = 248046  # <|im_end|> — NOT config.json's eos_token_id 248044
PROMPT = ('\nExtract all readable content from the image in natural human reading order and '
          'output the result as a single Markdown document. For charts or images, represent them '
          'using an HTML image tag: <img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, '
          'where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). '
          'Format formulas as LaTeX. Format tables as HTML: <table>...</table>. Transcribe all '
          'other text as standard Markdown. Preserve the original text without translation or '
          'paraphrasing.')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    args = ap.parse_args()

    dt = getattr(torch, args.dtype)
    dev = "mps" if (dt is torch.float16 and torch.backends.mps.is_available()) else "cpu"
    proc = AutoProcessor.from_pretrained(HF_ID)
    model = AutoModelForImageTextToText.from_pretrained(HF_ID, dtype=dt).to(dev).eval()

    img = Image.open(args.image).convert("RGB").resize((TILE_W, TILE_H), Image.BICUBIC)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    inputs = proc(text=[text], images=[img], return_tensors="pt")
    thw = inputs["image_grid_thw"][0].tolist()
    assert thw[1:] == [TILE_H // 16, TILE_W // 16], f"processor chose {thw}, not the ship grid"

    ids = inputs["input_ids"][0].numpy().astype(np.int64)
    print(f"prompt {len(ids)} tokens, image grid {thw}, dtype {dt} on {dev}")
    with torch.no_grad():
        out = model.generate(**{k: (v.to(dev) if hasattr(v, "to") else v)
                                for k, v in inputs.items()},
                             max_new_tokens=args.max_new, do_sample=False,
                             eos_token_id=[EOS])
    gen = out[0][len(ids):].cpu().numpy().astype(np.int64)

    np.savez_compressed(
        OUT,
        input_ids=ids, gen_ids=gen,
        patches=preprocess(np.asarray(img), TILE_H, TILE_W),
        grid_thw=np.asarray(thw, np.int64),
        dtype=np.asarray(args.dtype), device=np.asarray(dev))
    print(f"wrote {OUT}: {len(gen)} generated tokens")
    print(proc.tokenizer.decode(gen, skip_special_tokens=True)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
