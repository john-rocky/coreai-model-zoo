#!/usr/bin/env python3
"""More fixture rows for the mask decoder: segmentation / matting queries with several objects in one image.

The base set's dense rows (`fixtures.json`) all answer with one `<OBJ>`, so the decoder self-attention across the
`<OBJ>` rows of one response was never exercised. These rows reuse the base set's images (`<fixtures dir>/img/`,
same files, same sha256) and the package's own `build_prompt`; `fixtures.json` is not touched. Each call writes
the whole file (`fixtures_r2.json` next to `fixtures.json`) from the ROWS table below; a row that is replaced
keeps its old query in `history` so the change is on record.

    python3 -I make_fixtures_r2.py --fixtures-dir <fixtures dir> --package <WeVisOne-VL clone>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

PACKAGE_COMMIT = "40f735b8daed437ece982db9718e9c9856f52a79"

# id, fixtures.json row whose image is reused, task, query, expected objects (from synth_gt.json where synthetic)
ROWS = [
    ("syn_b_seg_circles", "syn_b_point", "segmentation", "blue circles", "synth:blue circle"),
    ("syn_a_seg_circles", "syn_a_seg", "segmentation", "blue circles", "synth:blue circle"),
    ("nasa_jungle_mat_people", "nasa_jungle_mat", "matting", "people", {"form": "boxes+OBJ",
                                                                       "approx": "the astronaut group, several people"}),
    ("cs_balls_seg", "cs_counting_balls", "segmentation", "green balls", {"form": "boxes+OBJ",
                                                                          "approx": "several green balls"}),
]
HISTORY: dict[str, list] = {}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixtures-dir", required=True)
    ap.add_argument("--package", required=True, help="the Tencent/WeVisOne-VL clone (prompts.py is loaded by path)")
    args = ap.parse_args()
    fx = Path(args.fixtures_dir).expanduser().resolve()
    base = json.loads((fx / "fixtures.json").read_text())
    by_id = {r["id"]: r for r in base["rows"]}
    gt = json.loads((fx / "img" / "synth_gt.json").read_text())
    spec = importlib.util.spec_from_file_location("wv_prompts", Path(args.package) / "wevisone_vl" / "prompts.py")
    prompts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompts)
    rows = []
    for rid, src, task, query, expect in ROWS:
        b = by_id[src]
        img = fx / b["image"]
        if sha256_file(img) != b["sha256"]:
            raise SystemExit(f"{img}: sha256 differs from round 1's row {src}")
        pix = np.asarray(Image.open(img).convert("RGB"))
        if hashlib.sha256(pix.tobytes()).hexdigest() != b["pixel_sha256"]:
            raise SystemExit(f"{img}: pixel sha256 differs from round 1's row {src}")
        if isinstance(expect, str) and expect.startswith("synth:"):
            label = expect.split(":", 1)[1]
            key = Path(b["image"]).stem
            boxes = [o["box_xyxy"] for o in gt[key]["objects"] if o["label"] == label]
            expect = {"form": "boxes+OBJ", "gt": {label: boxes}, "n": len(boxes)}
        rows.append({"id": rid, "image": b["image"], "sha256": b["sha256"], "pixel_sha256": b["pixel_sha256"],
                     "source": b["source"], "public_ok": b["public_ok"], "size": b["size"], "task": task,
                     "query": query, "prompt": prompts.build_prompt(task, query), "expect": expect,
                     "origin": f"coreai-r2 multi-instance (image of {src})", "history": HISTORY.get(rid, [])})
    out = {"format": 1, "package_commit": PACKAGE_COMMIT, "round": "wv-c r2",
           "note": "segmentation / matting rows with several objects in one image (round 1's dense rows all had one "
                   "<OBJ>); images are round 1's files; prompt = build_prompt(task, query).",
           "base": {"path": str(fx / "fixtures.json"), "sha256": sha256_file(fx / "fixtures.json")}, "rows": rows}
    path = fx / "fixtures_r2.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"{path}: {len(rows)} rows, sha256 {sha256_file(path)}")
    for r in rows:
        print(f"  {r['id']}: {r['task']} {r['query']!r} on {r['image']} -> {r['prompt']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
