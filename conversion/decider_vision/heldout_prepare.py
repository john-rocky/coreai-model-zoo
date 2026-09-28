#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.9.0",
#     "transformers==5.17.0",
#     "torchvision",
#     "pillow",
#     "huggingface_hub>=1.5.0,<2",
#     "numpy",
#     "pyarrow",
# ]
# [tool.uv]
# index-url = "https://pypi.org/simple"
# ///
"""Held-out inputs for the decider-2b-vision generalization gate: the 250 grid-price photos, re-made.

Round 6 chose which decoder layers stay fp16 on the self-made fixture. This script prepares the
inputs of the check that the choice was not fitted to that fixture: the 250 photos of the grid-price
table (`grid_price.py`: visual7w 150 + vsr 100 from The Cauldron, the two subsets the author holds
out of training), which no round has used to choose anything.

The rows are rebuilt with `grid_price.load_items` itself (same shards, file order, the author's
`parse()` and context, `thumbnail((768, 768))`, RGB) and matched one by one to the round-1 record
`price/items.jsonl`, whose `arms.g256` / `arms.g448` probabilities (the checkpoint's own code, MPS
fp32) are the reference; `size_thumb` (and row index, input size, option count, gold) must agree.
Then, per grid, the thumbnailed image is resized with PIL BICUBIC to the grid's square, as
grid_price.py did, and the author's `VisionDecisionModel.prepare()` builds the processor inputs.
`prepare()` reads only the processor, the tokenizer and three token ids, so it runs here on those
alone (the same lines as the class's `__init__`, without the weights): no model is loaded.

Written to `<work>/_decider2bv/heldout/` (outside the repo; the photos are never copied anywhere
else, and nothing but numbers leaves this directory):

* `images/<subset>_<row>.png`: the thumbnailed RGB image (lossless; re-read and compared);
* `npz/<subset>_<row>.npz`: per grid, the processor's `input_ids`, `pixel_values`,
  `image_grid_thw`, `slot_idx`, `nopts`;
* `items.jsonl`: per item the Example (context, question, options, gold), the file names and
  sha256s, and the reference probabilities of both grids with their top-2 margin;
* `prepare.json`: provenance (dataset revision, shard sha256s, checkpoint files, versions) and the
  checks.

    HF_HOME=~/code/coreai/_decider2bv/hf HF_HUB_OFFLINE=1 ~/code/coreai/_decider2bv/venv-oracle/bin/python \\
        conversion/decider_vision/heldout_prepare.py
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import grid_price  # noqa: E402  (load_items, parse, the dataset pins)
from _paths import hf_snapshot, work_path  # noqa: E402

SUBSETS = (("visual7w", 150), ("vsr", 100))       # grid_price.py's order and counts
VISION_START, VISION_END = 248053, 248054
SLOT_IDS = (318, 25, 15666)                       # " (", ":", "Answer"
NEAR_TIE = 0.02


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def top2_margin(p) -> float:
    o = np.sort(np.asarray(p, np.float64))[::-1]
    return float(o[0] - o[1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hf-id", default=grid_price.DEFAULT_HF_ID)
    ap.add_argument("--revision", default=grid_price.DEFAULT_REVISION)
    ap.add_argument("--parquet-dir", default=str(work_path("_decider2bv", "price", "data")),
                    help="local copies of the shards (<subset>-train-00000.parquet)")
    ap.add_argument("--price-items", default=str(work_path("_decider2bv", "price", "items.jsonl")))
    ap.add_argument("--out-dir", default=str(work_path("_decider2bv", "heldout")))
    args = ap.parse_args()

    import PIL
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor

    t_start = time.monotonic()
    snapshot = Path(hf_snapshot(args.hf_id, revision=args.revision))
    sys.path.insert(0, str(snapshot))
    from decider.vision import VisionDecisionModel               # noqa: E402  (author's code, unchanged)
    from decider.infer import Example, Q                         # noqa: E402

    # VisionDecisionModel.__init__ without the weights: prepare() reads only these five attributes.
    proc = AutoProcessor.from_pretrained(str(snapshot))
    tok = proc.tokenizer
    shim = SimpleNamespace(proc=proc, tok=tok,
                           slot_tok=tok.encode(" (", add_special_tokens=False)[-1],
                           colon=tok.encode(":", add_special_tokens=False)[-1],
                           answer_tok=tok.encode("Answer", add_special_tokens=False)[0])
    assert (shim.slot_tok, shim.colon, shim.answer_tok) == SLOT_IDS, (shim.slot_tok, shim.colon, shim.answer_tok)

    out = Path(args.out_dir).expanduser()
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "npz").mkdir(parents=True, exist_ok=True)
    price_path = Path(args.price_items).expanduser()
    price = [json.loads(line) for line in price_path.read_text().splitlines() if line.strip()]
    assert len(price) == sum(n for _, n in SUBSETS), len(price)

    shards = {}
    for sub, _ in SUBSETS:
        p = Path(args.parquet_dir).expanduser() / f"{sub}-train-00000.parquet"
        shards[sub] = {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p),
                       "hub_file": grid_price.SHARDS[sub]}

    records, pos = [], 0
    for sub, n in SUBSETS:
        items, rejected = grid_price.load_items(Path(shards[sub]["path"]), sub, n)
        assert len(items) == n, (sub, len(items))
        shards[sub]["rejected_before_n"] = rejected
        for it in items:
            ref = price[pos]
            pos += 1
            got = {"dataset": sub, "row_index": it["index"], "size_in": it["size_in"],
                   "size_thumb": it["size_thumb"], "nopts": len(it["options"]), "gold": it["gold"]}
            want = {k: ref[k] for k in got}
            assert got == want, (pos - 1, got, want)          # size_thumb included

            key = f"{sub}_{it['index']:05d}"
            im = it["image"]
            assert im.mode == "RGB" and list(im.size) == it["size_thumb"], (key, im.mode, im.size)
            png = out / "images" / f"{key}.png"
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            png.write_bytes(buf.getvalue())
            back = np.asarray(Image.open(png).convert("RGB"))
            px = np.asarray(im)
            assert back.shape == px.shape and np.array_equal(back, px), (key, "PNG round trip")

            ex = Example(it["context"], [Q(it["question"], it["options"], it["gold"])])
            arrays, arms = {}, {}
            for arm, side in grid_price.GRIDS.items():
                img = im.resize((side, side), Image.Resampling.BICUBIC)     # grid_price.py's arm image
                inp = VisionDecisionModel.prepare(shim, [(img, ex)])
                ids = inp["input_ids"][0].numpy().astype(np.int64)
                grid = inp["image_grid_thw"][0].tolist()
                g = side // 16
                n_img = int((ids == grid_price.IMAGE_PAD).sum())
                assert grid == [1, g, g] == ref["arms"][arm]["grid_thw"], (key, arm, grid, ref["arms"][arm]["grid_thw"])
                assert n_img == g * g // 4 == ref["arms"][arm]["n_image_tokens"], (key, arm, n_img)
                assert ids[0] == VISION_START and (ids[1:n_img + 1] == grid_price.IMAGE_PAD).all() \
                    and ids[n_img + 1] == VISION_END, (key, arm, "image block is not first")
                slot_idx = inp["slot_idx"].numpy().astype(np.int64)
                nopts = inp["nopts"].numpy().astype(np.int64)
                assert slot_idx.tolist() == [len(ids) - 1] and nopts.tolist() == [len(it["options"])], (key, arm, slot_idx)
                pv = inp["pixel_values"].numpy().astype(np.float32)
                assert pv.shape == (g * g, 1536), (key, arm, pv.shape)
                arrays.update({f"{arm}_input_ids": ids, f"{arm}_pixel_values": pv,
                               f"{arm}_image_grid_thw": np.asarray(grid, np.int64),
                               f"{arm}_slot_idx": slot_idx, f"{arm}_nopts": nopts})
                rp = ref["arms"][arm]["probs"]
                arms[arm] = {"tokens": int(len(ids)), "n_image_tokens": n_img, "grid_thw": grid,
                             "slot_idx": slot_idx.tolist(), "ids_sha256": sha256_bytes(ids.tobytes()),
                             "pixel_values_sha256": sha256_bytes(pv.tobytes()),
                             "ref_probs": rp, "ref_argmax": ref["arms"][arm]["argmax"],
                             "ref_top2_margin": top2_margin(rp), "ref_near_tie": top2_margin(rp) < NEAR_TIE,
                             "ref_correct": ref["arms"][arm]["correct"]}
                assert int(np.argmax(rp)) == ref["arms"][arm]["argmax"], (key, arm)
            npz = out / "npz" / f"{key}.npz"
            tmp = npz.with_name(npz.name + ".tmp")
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **arrays)
            os.replace(tmp, npz)
            rec = dict(got, key=key, context=it["context"], question=it["question"], options=it["options"],
                       png=f"images/{key}.png", png_sha256=sha256_file(png),
                       rgb_sha256=sha256_bytes(px.tobytes()), npz=f"npz/{key}.npz", npz_sha256=sha256_file(npz),
                       arms=arms)
            records.append(rec)
            if len(records) % 25 == 1:
                print(f"{len(records):3d}/250 {key} thumb {it['size_thumb']} tokens g256 {arms['g256']['tokens']} "
                      f"g448 {arms['g448']['tokens']} ref g256 {[round(v, 3) for v in arms['g256']['ref_probs']]}",
                      flush=True)
    assert pos == len(price)

    items_path = out / "items.jsonl"
    items_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    margins = {arm: [r["arms"][arm]["ref_top2_margin"] for r in records] for arm in grid_price.GRIDS}
    summary = {
        "items": len(records),
        "per_subset": {sub: sum(r["dataset"] == sub for r in records) for sub, _ in SUBSETS},
        "runs": len(records) * len(grid_price.GRIDS),
        "size_thumb_equal_price_items": len(records),
        "thumbnailed": sum(r["size_in"] != r["size_thumb"] for r in records),
        "ref_near_tie_runs": {arm: sum(m < NEAR_TIE for m in v) for arm, v in margins.items()},
        "ref_min_top2_margin": {arm: min(v) for arm, v in margins.items()},
        "tokens": {arm: {"min": min(r["arms"][arm]["tokens"] for r in records),
                         "max": max(r["arms"][arm]["tokens"] for r in records)} for arm in grid_price.GRIDS},
    }
    snap_files = {n: sha256_file(snapshot / n) for n in
                  ("decider/vision.py", "decider/prompt.py", "decider/infer.py", "processor_config.json",
                   "tokenizer.json", "tokenizer_config.json")}
    ip = proc.image_processor
    res = {
        "schema": "coreai-decider-vision-heldout-inputs/1",
        "purpose": "inputs of the round-7 generalization gate: photos no round used to choose anything",
        "source": {"hf_id": args.hf_id, "revision": args.revision, "sha256": snap_files},
        "reference": {"path": str(price_path), "sha256": sha256_file(price_path),
                      "what": "arms.g256 / arms.g448 probs of grid_price.py: the checkpoint's own "
                              "VisionDecisionModel.prepare() -> slot_logits(), MPS fp32 (4-row CPU fp32 check "
                              "max |dp| 8.5e-7, price/grid_price.json mps_check)"},
        "dataset": {"repo": grid_price.DATASET, "revision": grid_price.DATASET_REVISION, "shards": shards,
                    "license": ("dataset card has no licence field; 'Licensing Information': each sub-dataset is "
                                "governed by its own licence, the prompts (to the extent of HuggingFaceM4's rights) "
                                "are CC-BY-4.0"),
                    "held_out": "visual7w and vsr are held_out=True in decider/vision/data.py SUBSETS (never in train)",
                    "selection": "grid_price.load_items: first rows in file order that pass the author's filter",
                    "preprocessing": "thumbnail((768, 768)) when max side > 768, then RGB; context as in data.py",
                    "redistribution": "the images stay in this directory; they are not copied to the zoo, shared/ "
                                      "or any upload, only numbers are published"},
        "arm_image": "the thumbnailed RGB image resized with PIL BICUBIC to 256x256 / 448x448 (grid_price.py)",
        "processor_inputs": "decider/vision.py VisionDecisionModel.prepare() with the processor and tokenizer "
                            "only (AutoProcessor.from_pretrained(snapshot); the three slot ids as in __init__)",
        "processor": {"processor_class": type(proc).__name__, "image_processor_class": type(ip).__name__,
                      "tokenizer_class": type(tok).__name__, "resample": int(ip.resample)},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "pillow": PIL.__version__, "numpy": np.__version__},
        "summary": summary,
        "items_jsonl": {"path": str(items_path), "sha256": sha256_file(items_path)},
        "wall_seconds": time.monotonic() - t_start,
    }
    (out / "prepare.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"wrote {out}/items.jsonl, prepare.json, images/ and npz/ ({len(records)} items, "
          f"{time.monotonic() - t_start:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
