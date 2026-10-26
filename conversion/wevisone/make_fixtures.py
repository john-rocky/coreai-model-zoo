#!/usr/bin/env python3
"""Build the Core AI port's fixture set for WeVisOne-VL-2B from the shared fixture set.

The shared set (`--base`, a `fixtures.json` with an `img/` directory next to it) holds two seeded
synthetic images, four NASA public-domain photos, and eleven images from the package's repository:
single-panel crops of its case-study figures and its demo photo (parity only; not cleared for redistribution).
This script copies every image the base rows use, checks each file's sha256 and decoded-pixel
sha256 against the base row, keeps every base row unchanged, and appends the rows this port needs
to reach its per-task floor (detection >= 6, point >= 4, count_box >= 4, ocr >= 3,
segmentation + matting >= 5). Added rows reuse the base images; their prompt is the package's own
`build_prompt` (prompts.py loaded by path, no torch import).

    python -I make_fixtures.py --base <shared>/fixtures.json --package <WeVisOne-VL clone> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

PACKAGE_COMMIT = "40f735b8daed437ece982db9718e9c9856f52a79"
FLOOR = {"detection": 6, "point": 4, "count_box": 4, "ocr": 3, "segmentation+matting": 5}

# (id, base row whose image is reused, task, query, expect builder)
ADDED = [
    ("syn_b_det", "syn_b_point", "detection", ["red rectangle", "blue circle", "yellow rectangle"], "synth_all"),
    ("syn_a_point", "syn_a_det", "point", ["blue circle"], "synth:blue circle"),
    ("syn_b_cntbox", "syn_b_point", "count_box", "red rectangles", "synth_count:red rectangle"),
    ("nasa_guppy_det", "nasa_guppy_ref", "detection", ["airplane", "tree"],
     {"form": "boxes", "approx": "1 airplane in the sky; a tree line along the horizon"}),
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pixel_sha256(path: Path) -> str:
    with Image.open(path) as im:
        return hashlib.sha256(np.asarray(im.convert("RGB"), dtype=np.uint8).tobytes()).hexdigest()


def load_build_prompt(package: Path):
    spec = importlib.util.spec_from_file_location("wevisone_prompts", package / "wevisone_vl" / "prompts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_prompt


def synth_expect(kind: str, image_key: str, gt: dict) -> dict:
    objs = gt[image_key]["objects"]
    if kind == "synth_all":
        boxes: dict[str, list] = {}
        for o in objs:
            boxes.setdefault(o["label"], []).append(o["box_xyxy"])
        return {"form": "boxes", "gt": boxes}
    name, label = kind.split(":", 1)
    hits = [o["box_xyxy"] for o in objs if o["label"] == label]
    if name == "synth":
        return {"form": "points", "gt": {label: hits}}
    if name == "synth_count":
        return {"form": "count", "gt_count": len(hits), "gt": {label: hits}}
    raise ValueError(kind)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, help="shared fixtures.json (img/ next to it)")
    ap.add_argument("--package", required=True, help="Tencent/WeVisOne-VL clone at %s" % PACKAGE_COMMIT)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base_path = Path(args.base).resolve()
    base_dir = base_path.parent
    base = json.loads(base_path.read_text())
    if base.get("package_commit") != PACKAGE_COMMIT:
        raise SystemExit(f"base package_commit {base.get('package_commit')} != {PACKAGE_COMMIT}")
    out = Path(args.out).resolve()
    (out / "img").mkdir(parents=True, exist_ok=True)
    build_prompt = load_build_prompt(Path(args.package).resolve())

    rows = []
    copied = {}
    for row in base["rows"]:
        src = base_dir / row["image"]
        dst = out / row["image"]
        if row["image"] not in copied:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied[row["image"]] = (sha256_file(dst), pixel_sha256(dst))
        file_sha, pix_sha = copied[row["image"]]
        if file_sha != row["sha256"] or pix_sha != row["pixel_sha256"]:
            raise SystemExit(f"{row['id']}: hash mismatch file {file_sha} / pixels {pix_sha}")
        if build_prompt(row["task"], row["query"]) != row["prompt"]:
            raise SystemExit(f"{row['id']}: prompt differs from build_prompt()")
        rows.append({**row, "origin": "shared"})
    gt_src = base_dir / "img" / "synth_gt.json"
    shutil.copy2(gt_src, out / "img" / "synth_gt.json")
    gt = json.loads(gt_src.read_text())

    by_id = {r["id"]: r for r in rows}
    for rid, like, task, query, expect in ADDED:
        if rid in by_id:
            raise SystemExit(f"added id {rid} already in the base set")
        ref = by_id[like]
        if isinstance(expect, str):
            key = Path(ref["image"]).stem
            expect = synth_expect(expect, key, gt)
        rows.append({"id": rid, "image": ref["image"], "sha256": ref["sha256"], "pixel_sha256": ref["pixel_sha256"],
                     "source": ref["source"], "public_ok": ref["public_ok"], "size": ref["size"], "task": task,
                     "query": query, "prompt": build_prompt(task, query), "expect": expect,
                     "origin": "coreai-r1 (added to reach the per-task floor)"})

    counts = Counter(r["task"] for r in rows)
    have = {"detection": counts["detection"], "point": counts["point"], "count_box": counts["count_box"],
            "ocr": counts["ocr"], "segmentation+matting": counts["segmentation"] + counts["matting"]}
    short = {k: (have[k], v) for k, v in FLOOR.items() if have[k] < v}
    if short:
        raise SystemExit(f"per-task floor not met: {short}")
    doc = {"format": 1, "package_commit": PACKAGE_COMMIT,
           "base": {"path": str(base_path), "sha256": sha256_file(base_path), "rows": len(base["rows"])},
           "note": base.get("note"), "floor": FLOOR, "tasks": dict(sorted(counts.items())), "rows": rows}
    target = out / "fixtures.json"
    target.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "images": len(copied), "tasks": doc["tasks"], "floor_have": have,
                      "fixtures_sha256": sha256_file(target)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
