#!/usr/bin/env python3
"""Fixture images for the MiniCPM-V-4.7-1B oracle: one drawn sign, two NASA photographs.

* `synthetic.png` (800x600) — drawn from Pillow primitives with a fixed seed: three coloured
  rectangles and a green sign whose word "EXIT" is built from polygons, so no font file is
  involved. Same every run, byte for byte.
* `saturn_v.orig.jpg`, `earth.orig.jpg` — NASA photographs (public domain, NASA media guidelines),
  fetched from the images-api.nasa.gov search the way the round-1 launch names it: the item whose
  `nasa_id` matches, `links[0].href`. Nothing here shows a person.

Every image is then resized to 448x448 with Pillow BICUBIC (`*.448.png`): the thumbnail-only
input the oracle hands the processor (one 32x32-patch crop, 8x8 = 64 LLM tokens). The aspect
ratio is not kept, as in the shipped host path. `fixtures.json` records each source URL, the
NASA metadata and the sha256 of every file; `README.md` states where each image comes from.

The fixtures live outside the repository (`$ZOO_WORK_ROOT/_minicpmv47/oracle/fixtures`); the
photographs are not committed anywhere.

    python3 -I conversion/minicpmv47/make_fixtures.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import work_path  # noqa: E402

SIDE = 448
SEED = 47
NASA_SEARCH = "https://images-api.nasa.gov/search?q={q}&media_type=image"
PHOTOS = (
    # (file stem, search words, nasa_id) — picked by hand on 2026-10-13 from the search results:
    # NASA-made (JSC / KSC), no people, no third-party credit in the description.
    ("saturn_v", "saturn v rollout", "KSC-69P-372"),
    ("earth", "as17-148-22727", "as17-148-22727"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bar(d, x0, y0, x1, y1, fill):
    d.rectangle([x0, y0, x1, y1], fill=fill)


def draw_exit(d, x0, y0, h, fill):
    """Block letters E X I T from rectangles and two parallelograms, cap height h."""
    t = h // 5                      # stroke
    w = (h * 3) // 5                # letter width
    gap = h // 5
    # E
    _bar(d, x0, y0, x0 + t, y0 + h, fill)
    _bar(d, x0, y0, x0 + w, y0 + t, fill)
    _bar(d, x0, y0 + (h - t) // 2, x0 + (w * 4) // 5, y0 + (h + t) // 2, fill)
    _bar(d, x0, y0 + h - t, x0 + w, y0 + h, fill)
    x = x0 + w + gap
    # X: two diagonal strokes, t wide horizontally
    d.polygon([(x, y0), (x + t, y0), (x + w, y0 + h), (x + w - t, y0 + h)], fill=fill)
    d.polygon([(x + w - t, y0), (x + w, y0), (x + t, y0 + h), (x, y0 + h)], fill=fill)
    x += w + gap
    # I
    _bar(d, x, y0, x + t, y0 + h, fill)
    x += t + gap
    # T
    _bar(d, x, y0, x + w, y0 + t, fill)
    _bar(d, x + (w - t) // 2, y0, x + (w + t) // 2, y0 + h, fill)
    return x + w


def synthetic():
    from PIL import Image, ImageDraw

    rng = random.Random(SEED)
    im = Image.new("RGB", (800, 600), (232, 232, 232))
    d = ImageDraw.Draw(im)
    # three coloured rectangles in the lower half, seeded positions and sizes
    for color in ((214, 39, 40), (31, 87, 214), (240, 196, 25)):
        w, h = rng.randint(110, 190), rng.randint(90, 150)
        x, y = rng.randint(20, 800 - w - 20), rng.randint(330, 600 - h - 20)
        d.rectangle([x, y, x + w, y + h], fill=color)
    # green sign with white EXIT, centred in the upper half
    cap = 120
    sign_w = 3 * (cap * 3) // 5 + cap // 5 + 3 * (cap // 5) + 2 * 48   # letters + gaps + margins
    sx0 = (800 - sign_w) // 2
    d.rectangle([sx0, 70, sx0 + sign_w, 70 + cap + 2 * 40], fill=(0, 122, 51))
    draw_exit(d, sx0 + 48, 110, cap, (255, 255, 255))
    return im


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode())


def fetch_photo(out: Path, stem: str, words: str, nasa_id: str) -> dict:
    url = NASA_SEARCH.format(q=urllib.parse.quote(words))
    items = fetch_json(url)["collection"]["items"]
    hit = next((it for it in items if it["data"][0].get("nasa_id") == nasa_id), None)
    if hit is None:
        raise SystemExit(f"{nasa_id} not in the search results for {words!r} ({url})")
    meta = hit["data"][0]
    href = hit["links"][0]["href"]
    dst = out / f"{stem}.orig{Path(urllib.parse.urlparse(href).path).suffix}"
    with urllib.request.urlopen(href, timeout=120) as r:
        dst.write_bytes(r.read())
    return {"source": "NASA / public domain", "search_url": url, "href": href, "nasa_id": nasa_id,
            "title": meta.get("title"), "center": meta.get("center"), "date_created": meta.get("date_created"),
            "description": meta.get("description"), "file": dst.name, "sha256": sha256(dst)}


def to_448(src: Path, dst: Path) -> dict:
    from PIL import Image

    with Image.open(src) as im:
        size = im.size
        im.convert("RGB").resize((SIDE, SIDE), Image.Resampling.BICUBIC).save(dst)
    return {"source_size": list(size), "file": dst.name, "sha256": sha256(dst)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(work_path("_minicpmv47", "oracle", "fixtures")))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import PIL

    record = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "pillow": PIL.__version__, "side": SIDE, "resample": "PIL BICUBIC", "images": {}}
    syn = out / "synthetic.png"
    synthetic().save(syn)
    record["images"]["synthetic"] = {"source": "drawn here (Pillow primitives, seed 47; no font file)",
                                     "file": syn.name, "sha256": sha256(syn),
                                     "448": to_448(syn, out / "synthetic.448.png")}
    for stem, words, nasa_id in PHOTOS:
        rec = fetch_photo(out, stem, words, nasa_id)
        rec["448"] = to_448(out / rec["file"], out / f"{stem}.448.png")
        record["images"][stem] = rec
    (out / "fixtures.json").write_text(json.dumps(record, indent=1) + "\n")

    lines = ["# MiniCPM-V-4.7-1B oracle fixtures", "",
             f"Made by `conversion/minicpmv47/make_fixtures.py` on {record['generated_at']} (Pillow {PIL.__version__}).",
             "Each `*.448.png` is its source resized to 448x448 with Pillow BICUBIC (aspect not kept).", "",
             "| file | source | sha256 |", "|---|---|---|",
             f"| `synthetic.png` | drawn from Pillow primitives, seed {SEED}, no font file | `{record['images']['synthetic']['sha256']}` |"]
    for stem, _, _ in PHOTOS:
        r = record["images"][stem]
        lines.append(f"| `{r['file']}` | NASA / public domain — {r['nasa_id']} \"{r['title']}\" ({r['center']}, "
                     f"{(r['date_created'] or '')[:10]}), {r['href']} | `{r['sha256']}` |")
    lines += ["", "No image shows a person. The photographs are NASA works (public domain under the NASA media",
              "guidelines); they stay in this directory and are not committed to any repository.", ""]
    (out / "README.md").write_text("\n".join(lines))
    print(json.dumps({k: {kk: v[kk] for kk in ("file", "sha256")} for k, v in record["images"].items()}, indent=1))


if __name__ == "__main__":
    main()
