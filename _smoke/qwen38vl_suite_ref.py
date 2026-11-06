"""Multi-case bf16 oracle for the Qwen3.8-27B VISION path — the suite every stage gates on.

Fixture per case (3 COCO images x 2 coarse prompts, one case text-before-image), at the
FIXED 512x512 tile the vision bundle bakes (grid_thw (1,32,32) -> 1024 patches -> 256
merged tokens):

  * the resized uint8 image (self-contained; no network on re-run)
  * the processor's fp32 patches [1024, 1536] (NumPy-preprocessor gate target)
  * the HF vision-tower output [256, 5120] fp32 (tower gate target)
  * prompt ids, the ACTUAL 3-plane rope positions the text stack used (captured via a
    hook on the text rotary module — prefill [3,S] and every decode step), rope_deltas
  * bf16 greedy ids + per-step top-2 margins; full fp16 logits rows for two cases
    (mixed-sequence eager gate: per-position argmax with the margin>=0.1 rule)

bf16 like the text oracle (fp32 27.8B is ~111 GB; margin rule absorbs bf16 noise —
gen_qwen38_27b_ref.py). Slow (PIL) image processor pinned with use_fast=False: the host
preprocessor is NumPy+PIL, so the fixture must be too.

Usage: ~/.venvs/qwen38-oracle/bin/python _smoke/qwen38vl_suite_ref.py
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch

HF_ID = "Qwen/Qwen3.8-27B"
TILE = 512
N_GEN = 24
LOGITS_CASES = (0, 3)  # save fp16 logits rows for these (image-first + text-first)
N_LOGITS_STEPS = 16
OUT = Path(__file__).resolve().parent / "qwen38vl_suite_512.npz"
IMG_DIR = Path(__file__).resolve().parent / "qwen38vl_images"

IMAGES = [  # fixed COCO val2017: two cats on a couch, two kitchens
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000397133.jpg",
    "http://images.cocodataset.org/val2017/000000037777.jpg",
]
# (image_idx, prompt, image_first)
CASES = [
    (0, "What is in this image?", True),
    (0, "How many cats are in this image?", True),
    (1, "Where is this scene?", True),
    (1, "Describe the main colors in this image.", False),  # text BEFORE image
    (2, "What is in this image?", True),
    (2, "Describe the main colors in this image.", True),
]


def load_images() -> list["Image.Image"]:
    """512x512 PIL BICUBIC tiles, cached as PNG so re-runs are offline."""
    from PIL import Image

    IMG_DIR.mkdir(exist_ok=True)
    tiles = []
    for url in IMAGES:
        name = url.rsplit("/", 1)[-1].replace(".jpg", f"_{TILE}.png")
        p = IMG_DIR / name
        if not p.exists():
            import requests

            raw = Image.open(requests.get(url, stream=True, timeout=60).raw).convert("RGB")
            raw.resize((TILE, TILE), Image.BICUBIC).save(p)
            print(f"fetched {url} -> {p.name}")
        tiles.append(Image.open(p).convert("RGB"))
    return tiles


def main() -> None:
    import transformers
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(HF_ID, use_fast=False)
    print("processor:", type(processor.image_processor).__name__,
          "| resample", processor.image_processor.resample)
    tiles = load_images()

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        HF_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager"
    )
    model.eval()
    print("27B VLM loaded bf16 | visual depth:",
          model.config.vision_config.depth, "| merge",
          model.config.vision_config.spatial_merge_size)

    # Capture the ACTUAL rope position ids the text stack uses (3-plane mrope).
    captured: list[torch.Tensor] = []
    rot = model.model.language_model.rotary_emb
    orig_fwd = rot.forward

    def hooked(x, position_ids):
        captured.append(position_ids.detach().to(torch.int64).clone())
        return orig_fwd(x, position_ids)

    rot.forward = hooked

    saved: dict[str, np.ndarray] = {}
    texts: list[str] = []
    for case, (img_idx, prompt, image_first) in enumerate(CASES):
        content = [{"type": "image", "image": tiles[img_idx]},
                   {"type": "text", "text": prompt}]
        if not image_first:
            content = content[::-1]
        conversation = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        ids = inputs["input_ids"]
        grid = inputs["image_grid_thw"]
        pv = inputs["pixel_values"]
        assert tuple(grid[0].tolist()) == (1, TILE // 16, TILE // 16), grid
        assert pv.shape == (1024, 1536), pv.shape

        with torch.no_grad():
            vis = model.model.get_image_features(
                pixel_values=pv.clone(), image_grid_thw=grid)
        embeds = vis.pooler_output[0].float()
        assert embeds.shape == (256, 5120), embeds.shape

        captured.clear()
        gen, margins, rows = [], [], []
        with torch.no_grad():
            out = model(input_ids=ids, pixel_values=pv, image_grid_thw=grid,
                        mm_token_type_ids=inputs["mm_token_type_ids"],
                        use_cache=True)
            past = out.past_key_values
            row = out.logits[0, -1].float()
            for step in range(N_GEN):
                nxt = int(row.argmax())
                p = torch.softmax(row, dim=-1)
                top2 = torch.topk(p, 2).values
                margins.append(float(top2[0] - top2[1]))
                if case in LOGITS_CASES and step < N_LOGITS_STEPS:
                    rows.append(row.to(torch.float16).clone())
                gen.append(nxt)
                if step == N_GEN - 1:
                    break
                out = model(input_ids=torch.tensor([[nxt]], dtype=torch.long),
                            past_key_values=past, use_cache=True)
                past = out.past_key_values
                row = out.logits[0, -1].float()

        text = processor.decode(gen, skip_special_tokens=True)
        texts.append(text)
        # hook captures: [0] = prefill [3,1,S]; then one [3,1,1] per decode step
        prefill_pos = captured[0][:, 0, :].numpy().astype(np.int32)   # [3, S]
        step_pos = (
            torch.cat(captured[1:], dim=-1)[:, 0, :].numpy().astype(np.int32)
            if len(captured) > 1 else np.zeros((3, 0), np.int32)
        )
        saved[f"case{case}_image_idx"] = np.array(img_idx)
        saved[f"case{case}_patches"] = pv.numpy().astype(np.float32)
        saved[f"case{case}_grid_thw"] = grid.numpy().astype(np.int32)
        saved[f"case{case}_image_embeds"] = embeds.numpy().astype(np.float32)
        saved[f"case{case}_ids"] = ids[0].numpy().astype(np.int32)
        saved[f"case{case}_pos_prefill"] = prefill_pos
        saved[f"case{case}_pos_steps"] = step_pos
        saved[f"case{case}_rope_delta"] = np.array(
            int(model.model.rope_deltas[0, 0]), dtype=np.int64)
        saved[f"case{case}_gen"] = np.array(gen, dtype=np.int32)
        saved[f"case{case}_margins"] = np.array(margins, dtype=np.float32)
        if rows:
            saved[f"case{case}_logits_rows"] = torch.stack(rows).numpy()
        print(f"case {case}: img{img_idx} {'img-first' if image_first else 'TEXT-FIRST'}"
              f" | {prompt!r}\n  ids {ids.shape[1]} | delta"
              f" {int(model.model.rope_deltas[0,0])} | -> {text!r}")

    for i, tile in enumerate(tiles):
        saved[f"image{i}_u8"] = np.asarray(tile, dtype=np.uint8)
    saved["_meta_hf_id"] = np.array(HF_ID)
    saved["_meta_transformers"] = np.array(transformers.__version__)
    saved["_meta_tile"] = np.array(TILE)
    saved["_meta_cases"] = np.array(len(CASES))
    saved["_meta_texts"] = np.array(texts)
    saved["_meta_oracle_dtype"] = np.array("bf16")
    np.savez(OUT, **saved)
    print(f"\nwrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(CASES)} cases)")


if __name__ == "__main__":
    main()
