#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pillow>=11",
#     "tokenizers>=0.22",
# ]
# [tool.uv]
# index-url = "https://pypi.org/simple"
# ///
"""Fixture records for the clef-flash oracle: SystemOne-shaped requests, drawn and written from nothing.

Every record is a request body the checkpoint's own `joint_schema_model.systemone()` accepts as is
(`model`, `state`, `questions`; images are attached at run time from `image_files`). Four sources:

* `own_t*` - text states written here (support tickets, an incident report, an email thread,
  meeting notes, a review, a chat log, scheduling, a recipe, a request, a questionnaire summary,
  a restaurant order, a notice) and three long ones of about 2,000 tokens (a service log, a
  contract excerpt, the minutes of a members' meeting);
* `own_j*` - JSON states (an invoice, an order, sensor readings, a calendar event, a user profile,
  an API error payload, an inventory, a config diff, a tracking history, a survey, room
  availability as a top-level array, a basket with non-ASCII strings);
* `img_*` - images drawn with Pillow's primitives (shapes, a bar chart, seven-segment digits, a
  receipt and a floor plan in a 5x7 dot-matrix lettering defined below, a signal, a gauge,
  arrows, circles of three sizes, a clock); no photo, font file or downloaded asset is involved,
  so the images are CC0-1.0;
* `photo_*` - four CC0 photographs from Wikimedia Commons (a fruit bowl, a cat, a lake, a coffee cup;
  no people, text or brands), the 960-pixel thumbnails, fetched with `--fetch-photos` into
  `photos_src/` (sha256 pinned) and stored as PNG; the licence is read from the file's
  `extmetadata.LicenseShortName` (== "CC0") at fetch time;
* `semif_<id>` - SemIf's authored144 (MIT, `benchmarks/data/authored144.jsonl` at ca3ba65f), each
  row mapped to one `choice` question: state -> state, question -> instructions, options ->
  criteria {id: description}, gold = options[label].id.

The own records mix the three question types (noul / choice / score), 1 to 8 questions per record,
2 to 10 options per choice, 2 to 7 levels per score, questions without `instructions`, option
descriptions that are JSON objects, and noul `criteria` overrides. People, organisations, products
and places are invented; no URL or e-mail address appears. Organisation and place names use stems
that resolved to no A record on .com / .net / .io / .co.uk / .app when they were picked (a DNS
screen only; see `names_screen` in records.json). Four requests (own_t02, own_t15, own_t16, own_j01)
are not in this file: a later web search found one of their names in use (WITHHELD_REQUEST_SHA256).

    python3 conversion/clef_flash/make_fixtures.py --fetch-photos   # once (network), then offline
    python3 conversion/clef_flash/make_fixtures.py                  # -> $ZOO_WORK_ROOT/_clefflash/fixtures

Writes `records.json`, `images/<name>.png` and `images_meta.json` (size, PNG and raw-RGB sha256,
generator and arguments, licence). The drawing is deterministic: a re-run reproduces every
raw-RGB sha256 (the PNG bytes depend on the zlib build, the pixels do not).

`--heldout` writes a separate held-out set instead (`HELDOUT_*`, `heldout_*_records()`): 30 records
written on 2026-10-03, after round 4, to re-check a choice that was tuned on the set above. Ten text
states, eight JSON states, two long states, eight drawn images and two more CC0 photographs; none of
their text, names, images or photos comes from the set above, and `novelty_check` asserts that
against the round-1 files and stores the result in the held-out records.json (`novelty`). Image
records are meant for the g256 and g448 arms only. Nothing may be tuned on this set.

    python3 conversion/clef_flash/make_fixtures.py --heldout --fetch-photos   # once (network), then offline
    python3 conversion/clef_flash/make_fixtures.py --heldout                  # -> $ZOO_WORK_ROOT/_clefflash/fixtures/heldout

It writes the same files plus `photos_src/` and `contact_sheet.png` (every held-out image with its
name, for review).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import code_path, hf_snapshot, record_path, work_path  # noqa: E402

MODEL = "clef-flash"
HF_ID = "Cloudflare/clef-flash"
REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"
IMAGE_LICENSE = "CC0-1.0"
IMAGE_LICENSE_NOTE = "self-made: drawn by conversion/clef_flash/make_fixtures.py with Pillow primitives"
SEMIF = {
    "path": str(code_path("codex-conversions", "2026-09-21", "semif-ondevice", "fixtures", "authored144.jsonl")),
    "sha256": "8162d1c73f925af64453f1ec05ef36d583b3815bf698e60f0d454bd11537e079",
    "upstream_repo": "github.com/TheoLeeCJ/SemIf",
    "upstream_path": "benchmarks/data/authored144.jsonl",
    "upstream_commit": "ca3ba65f142967030ecb453346e94d6f476a69df",
    "license": "MIT",
    "copyright_notice": "Copyright (c) 2026 TheoLeeCJ",
    "rows": 144,
}
SEMIF_QUESTION_ID = "decision"
LONG_STATE_TOKENS = (1800, 2200)
# As written into records.json on 2026-10-03; the screens after it are in fixtures-clef-flash.json names_screen.
NAMES_SCREEN = {
    "method": "dig +short <stem>.<tld> A for tld in com, net, io, co.uk, app; a stem is used only when none resolves",
    "date": "2026-10-03",
    "stems_used": ["ashmerrow", "brindlecourt", "cradlewick", "fennistone", "harrowfen", "ostlebury",
                   "pennimere", "thistlecombe", "wendlecote"],
    "stems_rejected_resolving": ["quillmoor", "pellwick", "orrinvale", "saltreach", "tarnwick", "wrenmoor",
                                 "ferrylane", "marrowgate", "wimbleford", "larkhollow", "sorrelby", "dunmarra",
                                 "velloway", "kettlemoor", "oakhatch", "tamberlyn", "grisdale"],
    "not_checked": "exact-phrase web search, App Store search, company registries (to do before any publication)",
}

# --------------------------------------------------------------------------- drawing helpers
SEGMENTS = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
            "5": "afgcd", "6": "afgecd", "7": "abc", "8": "abcdefg", "9": "abcdfg"}

# 5x7 dot-matrix glyphs drawn for this file ('#' = dot on).
GLYPHS = {
    "A": [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "B": ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."],
    "C": [".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."],
    "D": ["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."],
    "E": ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
    "F": ["#####", "#....", "#....", "####.", "#....", "#....", "#...."],
    "G": [".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".####"],
    "H": ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "I": [".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "J": ["..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."],
    "K": ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"],
    "L": ["#....", "#....", "#....", "#....", "#....", "#....", "#####"],
    "M": ["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"],
    "N": ["#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"],
    "O": [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "P": ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."],
    "Q": [".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"],
    "R": ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
    "S": [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
    "T": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    "U": ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "V": ["#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."],
    "W": ["#...#", "#...#", "#...#", "#.#.#", "#.#.#", "#.#.#", ".#.#."],
    "X": ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"],
    "Y": ["#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."],
    "Z": ["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"],
    "0": [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
    "1": ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "2": [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    "3": ["####.", "....#", "....#", ".###.", "....#", "....#", "####."],
    "4": ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    "5": ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
    "6": [".###.", "#....", "#....", "####.", "#...#", "#...#", ".###."],
    "7": ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    "8": [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
    "9": [".###.", "#...#", "#...#", ".####", "....#", "....#", ".###."],
    ".": [".....", ".....", ".....", ".....", ".....", ".##..", ".##.."],
    ":": [".....", ".##..", ".##..", ".....", ".##..", ".##..", "....."],
    "-": [".....", ".....", ".....", "#####", ".....", ".....", "....."],
    "/": ["....#", "...#.", "...#.", "..#..", ".#...", ".#...", "#...."],
    "&": [".##..", "#..#.", "#.#..", ".#...", "#.#.#", "#..#.", ".##.#"],
    "#": [".#.#.", ".#.#.", "#####", ".#.#.", "#####", ".#.#.", ".#.#."],
    " ": [".....", ".....", ".....", ".....", ".....", ".....", "....."],
    "_": [".....", ".....", ".....", ".....", ".....", ".....", "#####"],   # contact-sheet labels only
}
for _g in GLYPHS.values():
    assert len(_g) == 7 and all(len(r) == 5 for r in _g)


def dot_text(d, text, x0, y0, dot, fill, gap=None):
    """Draw `text` in the 5x7 dot matrix: each dot a filled square of side `dot`, one blank column between glyphs."""
    gap = dot if gap is None else gap
    x = x0
    for ch in text:
        g = GLYPHS[ch]
        for r, row in enumerate(g):
            for c, v in enumerate(row):
                if v == "#":
                    d.rectangle([x + c * dot, y0 + r * dot, x + c * dot + dot - 2, y0 + r * dot + dot - 2], fill=fill)
        x += 5 * dot + gap
    return x


def dot_text_width(text, dot, gap=None):
    gap = dot if gap is None else gap
    return len(text) * (5 * dot + gap) - gap


def _hseg(d, x1, x2, y, t, fill):
    h = t / 2
    d.polygon([(x1, y), (x1 + h, y - h), (x2 - h, y - h), (x2, y), (x2 - h, y + h), (x1 + h, y + h)], fill=fill)


def _vseg(d, x, y1, y2, t, fill):
    h = t / 2
    d.polygon([(x, y1), (x + h, y1 + h), (x + h, y2 - h), (x, y2), (x - h, y2 - h), (x - h, y1 + h)], fill=fill)


def seven_segment(d, digit, x0, y0, w, h, t, fill, gap=2):
    on = SEGMENTS[digit]
    xl, xr = x0 + t / 2, x0 + w - t / 2
    yt, ym, yb = y0 + t / 2, y0 + h / 2, y0 + h - t / 2
    if "a" in on: _hseg(d, xl + gap, xr - gap, yt, t, fill)
    if "g" in on: _hseg(d, xl + gap, xr - gap, ym, t, fill)
    if "d" in on: _hseg(d, xl + gap, xr - gap, yb, t, fill)
    if "f" in on: _vseg(d, xl, yt + gap, ym - gap, t, fill)
    if "b" in on: _vseg(d, xr, yt + gap, ym - gap, t, fill)
    if "e" in on: _vseg(d, xl, ym + gap, yb - gap, t, fill)
    if "c" in on: _vseg(d, xr, ym + gap, yb - gap, t, fill)


def canvas(w, h, bg):
    im = Image.new("RGB", (w, h), bg)
    return im, ImageDraw.Draw(im)


# --------------------------------------------------------------------------- the images
def shapes_224():
    """Three red circles, two blue squares, one green triangle on white, 224x224."""
    im, d = canvas(224, 224, "white")
    for cx, cy in ((48, 52), (150, 40), (96, 150)):
        d.ellipse([cx - 22, cy - 22, cx + 22, cy + 22], fill=(215, 35, 35))
    for x, y in ((150, 110), (30, 180)):
        d.rectangle([x, y, x + 36, y + 36], fill=(35, 70, 210))
    d.polygon([(178, 170), (152, 214), (204, 214)], fill=(30, 150, 60))
    return im


def bars_640x480():
    """Five vertical bars, left to right red / orange / yellow / green / blue, heights 120/260/180/330/90."""
    im, d = canvas(640, 480, "white")
    base = 420
    colors = [(215, 35, 35), (240, 140, 25), (235, 205, 30), (30, 150, 60), (35, 70, 210)]
    for k, (col, h) in enumerate(zip(colors, (120, 260, 180, 330, 90))):
        x = 80 + k * 105
        d.rectangle([x, base - h, x + 70, base], fill=col)
    d.line([(60, base), (600, base)], fill=(0, 0, 0), width=3)
    d.line([(60, 40), (60, base)], fill=(0, 0, 0), width=3)
    for k in range(1, 8):
        y = base - 50 * k
        d.line([(54, y), (60, y)], fill=(0, 0, 0), width=2)
    return im


def sevenseg_47_300x200():
    """A two-digit red seven-segment display reading 47 on a dark panel."""
    im, d = canvas(300, 200, (25, 25, 30))
    d.rectangle([40, 30, 260, 170], fill=(10, 10, 12))
    seven_segment(d, "4", 70, 45, 70, 110, 14, (240, 40, 40))
    seven_segment(d, "7", 160, 45, 70, 110, 14, (240, 40, 40))
    return im


RECEIPT_LINES = [("PENNIMERE PANTRY", None), ("", None), ("BREAD", "3.20"), ("MILK", "1.10"),
                 ("APPLES", "2.45"), ("", None), ("TOTAL", "6.75"), ("CASH", "10.00"), ("CHANGE", "3.25")]


def receipt_1024x768():
    """A till receipt (dot-matrix lettering) on a grey desk, 1024x768; the store name is invented."""
    im, d = canvas(1024, 768, (150, 150, 155))
    d.rectangle([352, 40, 672, 728], fill=(250, 250, 245))
    ink, dot = (35, 35, 40), 4
    y = 80
    for left, right in RECEIPT_LINES:
        if left and right is None:
            w = dot_text_width(left, dot)
            dot_text(d, left, 512 - w // 2, y, dot, ink)
        elif left:
            dot_text(d, left, 380, y, dot, ink)
            dot_text(d, right, 644 - dot_text_width(right, dot), y, dot, ink)
        y += 52
    d.line([(380, y), (644, y)], fill=ink, width=2)
    dot_text(d, "THANK YOU", 512 - dot_text_width("THANK YOU", dot) // 2, y + 30, dot, ink)
    return im


def signal_green_256x240():
    """A three-lamp traffic signal with the bottom (green) lamp lit, 256x240."""
    im, d = canvas(256, 240, (185, 215, 240))
    d.rectangle([123, 200, 133, 239], fill=(70, 70, 70))
    d.rectangle([93, 15, 163, 205], fill=(35, 35, 35))
    for k, cy in enumerate((50, 110, 170)):
        col = (30, 220, 70) if k == 2 else (70, 70, 70)
        d.ellipse([128 - 24, cy - 24, 128 + 24, cy + 24], fill=col)
    return im


def gauge_300x200():
    """A semicircular pressure gauge, red zone from 75 % to 100 %, needle at about 85 %."""
    im, d = canvas(300, 200, "white")
    cx, cy, r = 150, 170, 120
    d.pieslice([cx - r, cy - r, cx + r, cy + r], 180, 360, fill=(235, 235, 235), outline=(0, 0, 0), width=3)
    d.pieslice([cx - r, cy - r, cx + r, cy + r], 180 + 0.75 * 180, 360, fill=(230, 60, 50))
    d.pieslice([cx - r + 25, cy - r + 25, cx + r - 25, cy + r - 25], 180, 360, fill="white")
    for k in range(11):
        a = math.pi + k * math.pi / 10
        d.line([(cx + (r - 18) * math.cos(a), cy + (r - 18) * math.sin(a)), (cx + r * math.cos(a), cy + r * math.sin(a))],
               fill=(0, 0, 0), width=3)
    a = math.pi + 0.85 * math.pi
    d.line([(cx, cy), (cx + (r - 30) * math.cos(a), cy + (r - 30) * math.sin(a))], fill=(20, 20, 20), width=6)
    d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=(20, 20, 20))
    return im


def _arrow(d, cx, cy, direction, size, fill):
    s = size
    if direction == "left":
        d.rectangle([cx - s // 4, cy - s // 8, cx + s // 2, cy + s // 8], fill=fill)
        d.polygon([(cx - s // 2, cy), (cx - s // 6, cy - s // 3), (cx - s // 6, cy + s // 3)], fill=fill)
    elif direction == "up":
        d.rectangle([cx - s // 8, cy - s // 4, cx + s // 8, cy + s // 2], fill=fill)
        d.polygon([(cx, cy - s // 2), (cx - s // 3, cy - s // 6), (cx + s // 3, cy - s // 6)], fill=fill)
    else:
        raise ValueError(direction)


def arrows_640x480():
    """Four black arrows in a row; the second points up, the others left."""
    im, d = canvas(640, 480, "white")
    for k, direction in enumerate(("left", "up", "left", "left")):
        _arrow(d, 95 + k * 150, 240, direction, 120, (0, 0, 0))
    return im


def floorplan_1024x768():
    """A four-room floor plan (walls, door gaps, room names in dot-matrix lettering), 1024x768."""
    im, d = canvas(1024, 768, "white")
    wall, w = (30, 30, 30), 8
    # outer walls: x 112..912, y 84..684; rooms: living (left, big), kitchen (top right),
    # bedroom (bottom right), bath (between kitchen and bedroom on the right edge).
    d.rectangle([112, 84, 912, 684], outline=wall, width=w)
    d.line([(560, 84), (560, 684)], fill=wall, width=w)            # living | right side
    d.line([(560, 324), (912, 324)], fill=wall, width=w)           # kitchen | rest
    d.line([(560, 474), (912, 474)], fill=wall, width=w)           # bath | bedroom
    d.line([(760, 324), (760, 474)], fill=wall, width=w)           # hall | bath
    for x0, y0, x1, y1 in ((556, 200, 566, 270), (556, 560, 566, 630), (640, 320, 710, 330), (756, 380, 766, 440)):
        d.rectangle([x0, y0, x1, y1], fill="white")                # door gaps
    d.rectangle([108, 360, 118, 440], fill="white")                # front door
    ink = (60, 60, 160)
    for text, (cx, cy) in (("LIVING", (336, 384)), ("KITCHEN", (736, 204)), ("BED", (736, 579)),
                           ("BATH", (836, 399)), ("HALL", (660, 399))):
        dot = 5 if text in ("LIVING", "KITCHEN", "BED") else 3
        dot_text(d, text, cx - dot_text_width(text, dot) // 2, cy - 7 * dot // 2, dot, ink)
    return im


def circles_640x480():
    """Three circles of different sizes: small blue (left), large red (middle), medium green (right)."""
    im, d = canvas(640, 480, (245, 245, 240))
    for cx, r, col in ((110, 40, (35, 70, 210)), (320, 130, (215, 35, 35)), (535, 75, (30, 150, 60))):
        d.ellipse([cx - r, 240 - r, cx + r, 240 + r], fill=col)
    return im


def clock_3oclock_256x240():
    """An analog wall clock at three o'clock: hour hand to 3, minute hand to 12."""
    im, d = canvas(256, 240, (225, 220, 210))
    cx, cy, r = 128, 120, 100
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill="white", outline=(40, 40, 40), width=6)
    for k in range(12):
        a = -math.pi / 2 + k * math.pi / 6
        r0 = r - (22 if k % 3 == 0 else 14)
        d.line([(cx + r0 * math.cos(a), cy + r0 * math.sin(a)), (cx + (r - 6) * math.cos(a), cy + (r - 6) * math.sin(a))],
               fill=(40, 40, 40), width=5 if k % 3 == 0 else 3)
    d.line([(cx, cy), (cx + 52, cy)], fill=(20, 20, 20), width=9)       # hour hand -> 3
    d.line([(cx, cy), (cx, cy - 78)], fill=(20, 20, 20), width=5)       # minute hand -> 12
    d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=(20, 20, 20))
    return im


IMAGES = {
    "shapes_224": (shapes_224, {}),
    "bars_640x480": (bars_640x480, {}),
    "sevenseg_47_300x200": (sevenseg_47_300x200, {}),
    "receipt_1024x768": (receipt_1024x768, {}),
    "signal_green_256x240": (signal_green_256x240, {}),
    "gauge_300x200": (gauge_300x200, {}),
    "arrows_640x480": (arrows_640x480, {}),
    "floorplan_1024x768": (floorplan_1024x768, {}),
    "circles_640x480": (circles_640x480, {}),
    "clock_3oclock_256x240": (clock_3oclock_256x240, {}),
}


COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "clef-flash-fixture-fetch/0.1 (offline model test fixture; python-urllib)"
PHOTOS = {  # name -> Commons file, the 960 px thumbnail actually used, its sha256
    "photo_fruit": {"title": "File:Fruits in bowl oranges lime apples (1).jpg", "size": [960, 1280],
                    "sha256": "e781f1deefe098097204409fa0b97fe06da8dac96f403a1f6186a2fdddb92331"},
    "photo_cat": {"title": "File:Tabby cat-3337027.jpg", "size": [960, 640],
                  "sha256": "242708bd1a489a885acebeec4a0f83d15d2551b9b210ddfd08daee11f82923bf"},
    "photo_lake": {"title": "File:Lake Mountain Landscape.jpg", "size": [960, 640],
                   "sha256": "cf5e281fab8eb63ba4335067d8a69fe81d9217fac4a2365e9655146d7cfa35d4"},
    "photo_coffee": {"title": "File:Coffee cup and coffee bean.jpg", "size": [960, 635],
                     "sha256": "672b86815fae409f073df0ce6f0411f9a908e5251b01fa0c058c7b9c6eadb32c"},
}


def fetch_photos(dst: Path, photos: dict = PHOTOS) -> dict:
    """Download the pinned thumbnails (Commons API imageinfo, iiurlwidth=960) after checking the licence field."""
    import urllib.parse
    import urllib.request
    dst.mkdir(parents=True, exist_ok=True)
    q = urllib.parse.urlencode({"action": "query", "format": "json", "prop": "imageinfo", "iiprop": "url|size|extmetadata",
                                "iiurlwidth": "960", "iiextmetadatafilter": "LicenseShortName|UsageTerms",
                                "titles": "|".join(p["title"] for p in photos.values())})
    req = urllib.request.Request(f"{COMMONS_API}?{q}", headers={"User-Agent": USER_AGENT})
    info = json.loads(urllib.request.urlopen(req, timeout=60).read())
    by_title = {p["title"]: p["imageinfo"][0] for p in info["query"]["pages"].values()}
    out = {}
    for name, ph in photos.items():
        ii = by_title[ph["title"]]
        lic = ii["extmetadata"]["LicenseShortName"]["value"]
        assert lic == "CC0", (name, lic)
        url = ii["thumburl"].split("?")[0]
        path = dst / f"{name}.jpg"
        if not path.exists():
            data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=120).read()
            path.write_bytes(data)
        out[name] = {"license_short_name": lic, "thumb_url": url, "file_page": ii["descriptionurl"]}
    (dst / "fetch_record.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


# --------------------------------------------------------------------------- question builders
def noul(instructions=None, criteria=None):
    q = {"type": "noul"}
    if instructions is not None:
        q["instructions"] = instructions
    if criteria is not None:
        q["criteria"] = criteria
    return q


def choice(instructions, criteria):
    q = {"type": "choice"}
    if instructions is not None:
        q["instructions"] = instructions
    q["criteria"] = criteria
    return q


def score(instructions, levels):
    q = {"type": "score"}
    if instructions is not None:
        q["instructions"] = instructions
    q["criteria"] = list(levels)
    return q


def record(rid, source, state, questions, gold=None, note="", image_files=None):
    out = {"id": rid, "source": source, "request": {"model": MODEL, "state": state, "questions": questions}}
    if image_files:
        out["image_files"] = list(image_files)
    out["gold"] = gold or {}
    out["note"] = note
    return out


DEPARTMENTS = {
    "billing": "Payments, invoices, charges and refunds",
    "technical": "A product or service does not work: bugs, faults, outages",
    "shipping": "Delivery, tracking, lost or damaged parcels",
    "account": "Sign-in, profile and password problems",
}
COUNT_WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


# --------------------------------------------------------------------------- long states
def service_log() -> str:
    """A deterministic service log (about 2,000 tokens): a payments-gateway outage from 03:12:07 to 03:19:02."""
    lines = []
    seq = 10231
    lat = 61

    def step():
        nonlocal seq, lat
        seq += 7
        lat = (lat * 37 + 11) % 97 + 40
        return seq, lat

    def t(m, s, ms):
        return f"2026-09-18T03:{m:02d}:{s:02d}.{ms:03d}Z"

    for k in range(8):                       # 03:00 - 03:08 normal traffic
        m, s = divmod(4 + k * 62, 60)
        r, l = step()
        if k % 4 == 3:
            lines.append(f"{t(m, s, (r * 13) % 1000)} INFO  inventory-svc   request_id=r-{r} GET /stock/BOLT-M6 status=200 latency_ms={l // 2} cache=hit")
        elif k % 4 == 1:
            lines.append(f"{t(m, s, (r * 13) % 1000)} INFO  payments-gateway request_id=r-{r} POST /authorize status=200 latency_ms={l + 90} pool_in_use={12 + k}")
        else:
            lines.append(f"{t(m, s, (r * 13) % 1000)} INFO  orders-api      request_id=r-{r} POST /orders status=201 latency_ms={l}")
    lines.append(f"{t(9, 2, 410)} WARN  payments-gateway p95_latency_ms=1210 threshold_ms=800 pool_in_use=176 pool_size=200")
    for k in range(2):
        r, l = step()
        lines.append(f"{t(9, 30 + k * 14, (r * 13) % 1000)} INFO  orders-api      request_id=r-{r} POST /orders status=201 latency_ms={l + 900}")
    lines.append(f"{t(11, 52, 18)} WARN  payments-gateway connection pool saturated pool_in_use=200 pool_size=200 waiting=37")
    lines.append(f"{t(12, 7, 233)} ERROR payments-gateway upstream connect error: connection refused (card-processor pool exhausted) attempt=1/3")
    for k in range(10):                      # 03:12 - 03:17 failures
        m, s = divmod(12 * 60 + 9 + k * 30, 60)
        r, l = step()
        if k % 3 == 0:
            lines.append(f"{t(m, s, (r * 13) % 1000)} ERROR orders-api      request_id=r-{r} POST /orders status=502 error=payment_authorization_failed upstream=payments-gateway")
        elif k % 3 == 1:
            lines.append(f"{t(m, s, (r * 13) % 1000)} ERROR payments-gateway request_id=r-{r} POST /authorize status=503 error=upstream_unavailable attempt={1 + k % 3}/3")
        else:
            lines.append(f"{t(m, s, (r * 13) % 1000)} WARN  orders-api      request_id=r-{r} retry scheduled backoff_ms={250 * (1 + k % 4)} reason=payment_authorization_failed")
        if k == 4:
            lines.append(f"{t(m, s + 1, 5)} INFO  notify-worker   queued 41 customer e-mails template=payment_delayed")
        if k == 6:
            lines.append(f"{t(m, s + 1, 77)} INFO  alerting        page sent to on-call rotation=payments severity=high")
    lines.append(f"{t(17, 30, 440)} INFO  payments-gateway restart requested by on-call config_change=pool_size:200->400")
    lines.append(f"{t(18, 41, 902)} INFO  payments-gateway process started version=5.12.3 pool_size=400")
    lines.append(f"{t(19, 2, 116)} INFO  payments-gateway health check passed upstream=card-processor latency_ms=88")
    for k in range(6):                       # recovery
        m, s = divmod(19 * 60 + 10 + k * 40, 60)
        r, l = step()
        if k % 5 == 2:
            lines.append(f"{t(m, s, (r * 13) % 1000)} INFO  payments-gateway request_id=r-{r} POST /authorize status=200 latency_ms={l + 40} pool_in_use={30 + k}")
        elif k % 5 == 4:
            lines.append(f"{t(m, s, (r * 13) % 1000)} INFO  orders-api      request_id=r-{r} retry succeeded original_status=502 attempts=2")
        else:
            lines.append(f"{t(m, s, (r * 13) % 1000)} INFO  orders-api      request_id=r-{r} POST /orders status=201 latency_ms={l}")
    lines.append(f"{t(24, 48, 300)} INFO  orders-api      backlog drained: 312 orders re-authorized, 0 failed permanently")
    lines.append(f"{t(25, 3, 12)} INFO  notify-worker   queued 41 customer e-mails template=payment_completed")
    return "\n".join(lines)


# Four round-1 requests are not in this file. On 2026-10-04 an exact-phrase web search found an invented name in each
# in use (fixtures-clef-flash.json names_screen.fixture.stems_in_use_found_2026-10-04). The published fixture keeps
# their ids, gold answers and numbers; the requests come from a private copy (--withheld), checked against these
# sha256 of json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":")) as UTF-8.
WITHHELD_REQUEST_SHA256 = {
    "own_t02": "727341c27bc2ff3570eee6790dc40e572091414c4d587031377940b8c6e23111",
    "own_t15": "03ea2157d2721f94b568632995fe50ab5b1bc702c75e5074abc4845c46687cd9",
    "own_t16": "f86f4edabf30e7ffdd91b8480b886052baef417cfd804012e3d4c34c351eed0b",
    "own_j01": "d963c4013aa890a4f2411b2d66a122068a2a73eafb2828a0b2c87bb28fb595e0",
}
WITHHELD_FILE = work_path("_clefflash", "results", "withheld_records_2026-10-04.json")


def load_withheld(path: Path) -> dict:
    """{record id: request} for the withheld records; stops when the copy is missing or a request changed."""
    if not path.exists():
        raise SystemExit(f"{path}: not found. The requests of {', '.join(WITHHELD_REQUEST_SHA256)} are not published "
                         "(an invented-name collision found 2026-10-04), so the round-1 records.json cannot be rebuilt "
                         "without the private copy (--withheld).")
    out = {}
    for r in json.loads(path.read_text())["records"]:
        if r["id"] in WITHHELD_REQUEST_SHA256:
            text = json.dumps(r["request"], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            assert hashlib.sha256(text.encode()).hexdigest() == WITHHELD_REQUEST_SHA256[r["id"]], (r["id"], "request changed")
            out[r["id"]] = r["request"]
    assert set(out) == set(WITHHELD_REQUEST_SHA256), ("withheld copy lacks", sorted(set(WITHHELD_REQUEST_SHA256) - set(out)))
    return out


def withheld_record(withheld, rid, source, gold, note):
    req = withheld[rid]
    return record(rid, source, req["state"], req["questions"], gold=gold, note=note)


# --------------------------------------------------------------------------- the own records
def own_text_records(withheld):
    R = []
    R.append(record("own_t01", "own_text",
        "Ticket 4471 from Mara Ellison: Since this morning the DL-40 desk lamp I bought from Wendlecote Home will not "
        "turn on. I tried two different outlets and a different cable. It is still within the 90-day warranty. I would "
        "like it repaired or replaced, not refunded.",
        {"department": choice("Which team should handle this ticket?", DEPARTMENTS),
         "wants_refund": noul("The customer asks for a refund."),
         "urgency": score("How urgent is the ticket?", ["Can wait", "This week", "Today", "Immediately"])},
        gold={"department": "technical", "wants_refund": "false", "urgency": None},
        note="support ticket"))
    R.append(withheld_record(withheld, "own_t02", "own_text",
        gold={"severity": None, "resolved": "true", "root_cause": "config_change", "data_loss": "false"},
        note="incident report; noul criteria override (both keys)"))
    R.append(record("own_t03", "own_text",
        "From: Ines Valcourt\nTo: Theo Brandt\nSubject: Re: Thursday site visit\n\n"
        "Theo, the inspector from Fennistone Testing can no longer come on Thursday. She offered Monday at 9:00 or "
        "Tuesday at 14:00 instead. Our loading bay is closed all day on Monday for resurfacing, so she could not get "
        "the equipment in. Can you confirm which slot works for you?\n\n"
        "> From: Theo Brandt\n> Ines, please confirm the Thursday 10:00 inspection is still on.",
        {"new_slot": choice("Which slot should be booked?", {"mon_0900": "Monday at 9:00", "tue_1400": "Tuesday at 14:00",
                                                            "thu_1000": "Thursday at 10:00, as originally planned"}),
         "thursday_still_on": noul("The Thursday inspection still takes place."),
         "reply_needed": noul()},
        gold={"new_slot": "tue_1400", "thursday_still_on": "false", "reply_needed": "true"},
        note="email thread; one question without instructions"))
    R.append(record("own_t04", "own_text",
        "Weekly sync, design team. Attendees: Noor, Felix, Abebe. Decisions: ship the dark-mode toggle in v2.3; "
        "postpone the onboarding redesign to the first quarter. Action items: Felix drafts the release note by Friday; "
        "Abebe fixes the contrast bug on the settings page. Open question: whether to keep the beta label on the toggle.",
        {"dark_mode": choice("What was decided about the dark-mode toggle?", {"ship_v23": "Ship it in v2.3",
                                                                             "postpone": "Postpone it", "cancel": "Cancel it"}),
         "release_note_owner": choice("Who drafts the release note?", {"noor": "Noor", "felix": "Felix", "abebe": "Abebe"}),
         "open_items": score("How many questions are still open?", ["None", "One", "Several"]),
         "beta_label_decided": noul("The team decided whether to keep the beta label.")},
        gold={"dark_mode": "ship_v23", "release_note_owner": "felix", "open_items": "1", "beta_label_decided": "false"},
        note="meeting notes"))
    R.append(record("own_t05", "own_text",
        "Rated 2 out of 5. The KT-3 kettle looks great and boils fast, but after three weeks the lid hinge cracked, and "
        "once the automatic shut-off did not trigger, which is a real safety problem. Customer support replied within a "
        "day and offered a replacement.",
        {"sentiment": score("What is the overall sentiment of the review?", ["very negative", "negative", "mixed", "positive", "very positive"]),
         "safety_issue": noul("The review reports a safety problem."),
         "main_complaint": choice("Which aspect is the main complaint?", {
             "durability": {"covers": "parts breaking or wearing out", "example": "a hinge snapped after a month"},
             "speed": {"covers": "how fast the product works", "example": "boils in two minutes"},
             "design": {"covers": "looks and form", "example": "a matte finish that shows fingerprints"},
             "support": {"covers": "the customer service experience", "example": "an agent who never replied"}}),
         "recommend": noul("Would the reviewer recommend the kettle?", {"true": "The reviewer would recommend the product to others."})},
        gold={"sentiment": "1", "safety_issue": "true", "main_complaint": "durability", "recommend": "false"},
        note="product review; option descriptions are JSON objects; noul override of 'true' only"))
    R.append(record("own_t06", "own_text",
        "[09:02] Pia: is the staging database down for anyone else?\n[09:03] Omar: yes, timeouts since about 08:50\n"
        "[09:05] Pia: production looks fine\n[09:07] Omar: restarted the staging replica, it's back up\n"
        "[09:08] Pia: thanks! running the migration test again",
        {"environment": choice("Which environment was affected?", {"production": "Production only", "staging": "Staging only",
                                                                  "both": "Production and staging", "neither": "No environment"}),
         "fixed": noul("The problem was fixed during the conversation."),
         "urgency": score(None, ["low", "medium", "high"])},
        gold={"environment": "staging", "fixed": "true", "urgency": None},
        note="chat log; score question without instructions"))
    R.append(record("own_t07", "own_text",
        "Can we find 30 minutes next week for the budget review? I'm out on Monday and on Wednesday afternoon. Jonas can "
        "only do mornings. Priya is travelling on Thursday and Friday. - Lena",
        {"best_slot": choice("Which slot works for everyone?", {
             "mon_am": "Monday morning", "tue_am": "Tuesday morning", "tue_pm": "Tuesday afternoon",
             "wed_pm": "Wednesday afternoon", "thu_am": "Thursday morning", "fri_am": "Friday morning"}),
         "length": choice("How long should the meeting be?", {"15": "15 minutes", "30": "30 minutes", "60": "one hour", "90": "90 minutes"}),
         "needs_travel": noul("Someone has to travel to attend the meeting.")},
        gold={"best_slot": "tue_am", "length": "30", "needs_travel": "false"},
        note="scheduling"))
    R.append(record("own_t08", "own_text",
        "Overnight oats for two: 1 cup rolled oats, 1 cup milk, half a cup of yogurt, 2 tablespoons chia seeds, "
        "1 tablespoon honey. Stir, cover and refrigerate for at least 6 hours. Top with berries before serving. Contains "
        "dairy. Can be made vegan with oat milk, soy yogurt and maple syrup.",
        {"diet": choice("As written, which diet does the recipe fit?", {"vegan": "Vegan: no animal products",
                                                                       "vegetarian": "Vegetarian: no meat or fish, may contain dairy or honey",
                                                                       "contains_meat": "Contains meat or fish"}),
         "needs_cooking": noul("The dish needs heat to prepare."),
         "effort": score("How much preparation effort does it take?", ["minimal", "some", "a lot"]),
         "servings": choice("How many servings does the recipe make?", {str(k + 1): COUNT_WORDS[k] for k in range(10)})},
        gold={"diet": "vegetarian", "needs_cooking": "false", "effort": "0", "servings": "2"},
        note="recipe; a 10-option choice (ids '1'..'10' sort as strings)"))
    R.append(record("own_t09", "own_text",
        "Hi, I was charged twice for my March subscription: two charges of 14.99 on March 3. Please refund the duplicate "
        "charge. This is the second time this has happened and I'm starting to lose patience. Thanks, Ari Feldmann",
        {"department": choice("Which team should handle this message?", DEPARTMENTS),
         "duplicate_charge": noul("The customer was charged more than once for the same thing."),
         "refund_amount": choice("How much should be refunded?", {"0": "nothing", "14.99": "one charge of 14.99",
                                                                  "29.98": "both charges"}),
         "frustration": score("How frustrated is the customer?", ["calm", "slightly annoyed", "annoyed", "frustrated",
                                                                   "very frustrated", "angry", "furious"])},
        gold={"department": "billing", "duplicate_charge": "true", "refund_amount": "14.99", "frustration": None},
        note="billing ticket; a 7-level score"))
    R.append(record("own_t10", "own_text",
        "Request from Malik Osei (warehouse, night shift): I need to swap my Saturday 22:00-06:00 shift on the 14th with "
        "Dara Lindqvist, who has agreed. We are both certified forklift operators. I am asking two weeks in advance, as "
        "the policy requires.",
        {"meets_conditions": noul("The shift swap meets the stated conditions."),
         "policy_issue": choice("Is there a policy problem with the request?", {
             "none": "No policy problem", "notice": "Not enough notice", "certification": "A missing certification",
             "consent": "The other worker has not agreed"}),
         "review": score("Does the request need a manager's review?", ["can be approved", "needs review"])},
        gold={"meets_conditions": "true", "policy_issue": "none", "review": "0"},
        note="shift request; a 2-level score"))
    R.append(record("own_t11", "own_text",
        "The vendor's security questionnaire says customer files are encrypted at rest with AES-256 and in transit with "
        "TLS 1.3, backups are kept for 35 days, and staff access requires hardware keys. Penetration tests are run "
        "yearly; the most recent report is 14 months old.",
        {"encrypted_at_rest": noul("Customer files are encrypted at rest."),
         "pentest_current": noul("Is the penetration test current?", {
             "true": "A penetration test was done within the last 12 months.",
             "false": "The last penetration test is older than 12 months, or its date is unknown."}),
         "risk": score("How risky is the vendor overall?", ["low", "moderate", "elevated", "high"]),
         "next_step": choice("What should happen next?", {"approve": "Approve the vendor as is",
                                                          "request_pentest": "Ask for a new penetration test before approving",
                                                          "reject": "Reject the vendor"})},
        gold={"encrypted_at_rest": "true", "pentest_current": "false", "risk": None, "next_step": "request_pentest"},
        note="security questionnaire summary; noul override (both keys)"))
    R.append(record("own_t12", "own_text",
        "Order note from table 12: two adults and one child (age 6). Adults: one vegetarian lasagne, and one grilled trout "
        "without butter (dairy allergy). Child: plain pasta, no sauce. Drinks: one sparkling water, one lemonade, one "
        "orange juice. They asked for the bill to be split in two and want the dessert menu after the mains. Please put "
        "a birthday candle on the child's dessert.",
        {"dairy_allergy": noul("Someone at the table has a dairy allergy."),
         "guests": choice("How many people are at the table?", {str(k): COUNT_WORDS[k - 1] for k in range(1, 7)}),
         "split_bill": noul("The bill should be split."),
         "split_ways": choice("Into how many parts should the bill be split?", {"2": "two", "3": "three", "4": "four"}),
         "occasion": choice("What is the occasion?", {"none": "No special occasion", "birthday": "A birthday",
                                                      "anniversary": "An anniversary", "business": "A business meal",
                                                      "graduation": "A graduation"}),
         "kitchen_effort": score("How demanding is the order for the kitchen?", ["trivial", "easy", "moderate", "involved", "complex"]),
         "dessert_planned": noul("The guests plan to order dessert."),
         "child_meal": choice("What does the child eat?", {"plain_pasta": "Plain pasta", "lasagne": "Vegetarian lasagne",
                                                           "trout": "Grilled trout", "burger": "A burger"})},
        gold={"dairy_allergy": "true", "guests": "3", "split_bill": "true", "split_ways": "2", "occasion": "birthday",
              "kitchen_effort": None, "dessert_planned": "true", "child_meal": "plain_pasta"},
        note="restaurant order; the 8-question record"))
    R.append(record("own_t13", "own_text",
        "Notice: the staff car park at Brindlecourt is closed for repairs from June 2 to June 6. Staff may park in the "
        "east lot; visitors should use the street. The badge readers at the side entrance will be offline on June 3.",
        {"car_park_open_june_4": noul(),
         "street_parking": choice("Who should park on the street?", {"staff": "Staff", "visitors": "Visitors", "everyone": "Everyone"}),
         "disruption": score("How disruptive is the closure?", ["minor", "moderate", "major"])},
        gold={"car_park_open_june_4": "false", "street_parking": "visitors", "disruption": None},
        note="notice; noul without instructions"))
    R.append(record("own_t14", "own_text_long", service_log(),
        {"first_failing_component": choice("Which component failed first?", {
             "auth": "The authentication service", "database": "The primary database", "payments": "The payments gateway",
             "inventory": "The inventory service", "notifications": "The notification worker"}),
         "recovered": noul("By the end of the log the failing component has recovered."),
         "severity": score("How severe was the event for customers?", ["informational", "minor", "major", "critical"]),
         "error_window": choice("How long did the errors last?", {"under_1m": "Under one minute", "1_to_5m": "One to five minutes",
                                                                  "5_to_15m": "Five to fifteen minutes", "over_15m": "More than fifteen minutes"})},
        gold={"first_failing_component": "payments", "recovered": "true", "severity": None, "error_window": "5_to_15m"},
        note="long state: service log (generated deterministically)"))
    R.append(withheld_record(withheld, "own_t15", "own_text_long",
        gold={"non_renewal_notice": "90", "auto_renews": "true", "liability_cap": "fees_12m",
              "assign_without_consent": "false", "client_risk": None, "uptime_target": "99.5"},
        note="long state: contract excerpt"))
    R.append(withheld_record(withheld, "own_t16", "own_text_long",
        gold={"water_decision": "replace", "fees_increase": "false", "new_chair": "somerfield", "water_vote_consensus": "2",
              "waiting_list_order": "application_order", "bonfire_ban": "true"},
        note="long state: minutes of a members' meeting"))
    return R


def own_json_records(withheld):
    R = []
    R.append(withheld_record(withheld, "own_j01", "own_json",
        gold={"status": "overdue", "large": "false", "priority": None},
        note="invoice"))
    R.append(record("own_j02", "own_json",
        {"order_id": "B-55102", "items": [{"sku": "MUG-11", "qty": 2}, {"sku": "TEA-04", "qty": 1}],
         "shipping": {"method": "express", "address_verified": False}, "payment": {"status": "authorized"},
         "customer_note": "Gift - no prices in the box please"},
        {"can_ship": noul("The order can ship now."),
         "blocker": choice("What is blocking the order, if anything?", {"none": "Nothing", "payment": "Payment",
                                                                        "address": "The shipping address", "stock": "Stock"}),
         "gift": noul("The order is a gift."),
         "priority": score("How should the order be prioritised?", ["normal", "high", "urgent"])},
        gold={"can_ship": "false", "blocker": "address", "gift": "true", "priority": None},
        note="order"))
    R.append(record("own_j03", "own_json",
        {"device": "greenhouse-3", "unit": "celsius",
         "readings": [{"t": "06:00", "temp": 14.2, "humidity": 81}, {"t": "09:00", "temp": 19.8, "humidity": 70},
                      {"t": "12:00", "temp": 31.5, "humidity": 52}, {"t": "15:00", "temp": 33.1, "humidity": 47}],
         "limits": {"temp_max": 30, "humidity_min": 50}},
        {"over_limit": noul("A reading exceeded a limit."),
         "first_violation": choice("When was a limit first exceeded?", {"06:00": "06:00", "09:00": "09:00", "12:00": "12:00",
                                                                        "15:00": "15:00", "never": "Never"}),
         "trend": choice("How is the temperature changing?", {"rising": "Rising", "falling": "Falling", "flat": "Flat"}),
         "action": score("What response is appropriate?", ["no action", "keep monitoring", "open the vents", "emergency cooling"])},
        gold={"over_limit": "true", "first_violation": "12:00", "trend": "rising", "action": None},
        note="sensor readings"))
    R.append(record("own_j04", "own_json",
        {"event": {"title": "Quarterly planning", "start": "2026-10-12T15:00", "end": "2026-10-12T16:30",
                   "timezone": "UTC+2", "organizer": "Hana Ruiz",
                   "attendees": [{"name": "Hana Ruiz", "response": "accepted"}, {"name": "Elio Marsh", "response": "declined"},
                                 {"name": "Sven Kaur", "response": "tentative"}, {"name": "Ada Nwosu", "response": "no_response"}],
                   "location": "Room 4B", "recurrence": None}},
        {"duration": choice("How long is the event?", {"30": "30 minutes", "60": "60 minutes", "90": "90 minutes", "120": "120 minutes"}),
         "recurring": noul("The event repeats."),
         "accepted": choice("How many attendees have accepted?", {str(k): COUNT_WORDS[k - 1] if k else "zero" for k in range(5)}),
         "attendance_risk": score("How likely is it that too few people attend?", ["unlikely", "possible", "likely"])},
        gold={"duration": "90", "recurring": "false", "accepted": "1", "attendance_risk": None},
        note="calendar event"))
    R.append(record("own_j05", "own_json",
        {"user": {"id": 88213, "display_name": "quietfern", "created": "2019-04-11", "plan": "free", "verified_email": True,
                  "two_factor": False, "last_login": "2026-09-30", "country": "PT", "flags": ["password_reused"]}},
        {"plan": choice("Which plan is the user on?", {"free": "Free", "pro": "Pro", "team": "Team", "enterprise": "Enterprise"}),
         "two_factor": noul("Two-factor authentication is enabled."),
         "account_risk": score("How at risk is the account?", ["very low", "low", "medium", "high", "very high"]),
         "action": choice("What should the security team do?", {"none": "Nothing", "prompt_2fa": "Prompt the user to enable two-factor authentication",
                                                                "force_reset": "Force a password reset", "suspend": "Suspend the account"})},
        gold={"plan": "free", "two_factor": "false", "account_risk": None, "action": None},
        note="user profile"))
    R.append(record("own_j06", "own_json",
        {"error": {"status": 429, "code": "rate_limited", "message": "Too many requests", "retry_after_seconds": 30,
                   "request_id": "req_7f3a"},
         "client": {"requests_last_minute": 312, "plan_limit_per_minute": 300}},
        {"error_class": choice("What kind of error is this?", {"auth": "Authentication or permission", "rate_limit": "Rate limiting",
                                                               "not_found": "Resource not found", "server": "Server fault",
                                                               "validation": "Invalid request", "timeout": "Timeout"}),
         "retryable": noul("The request can be retried later."),
         "wait_seconds": choice("How long should the client wait before retrying?", {"0": "No wait", "30": "30 seconds",
                                                                                    "300": "5 minutes", "3600": "1 hour"}),
         "client_at_fault": noul("The client caused the error."),
         "severity": score("How severe is the error?", ["low", "medium", "high"])},
        gold={"error_class": "rate_limit", "retryable": "true", "wait_seconds": "30", "client_at_fault": "true", "severity": None},
        note="API error payload"))
    R.append(record("own_j07", "own_json",
        {"warehouse": "north", "items": [{"sku": "BOLT-M6", "on_hand": 1200, "reorder_point": 500},
                                         {"sku": "NUT-M6", "on_hand": 340, "reorder_point": 500},
                                         {"sku": "WASHER-6", "on_hand": 0, "reorder_point": 300},
                                         {"sku": "BRACKET-L", "on_hand": 75, "reorder_point": 50}]},
        {"out_of_stock": choice("Which item is out of stock?", {"BOLT-M6": "BOLT-M6", "NUT-M6": "NUT-M6", "WASHER-6": "WASHER-6",
                                                                "BRACKET-L": "BRACKET-L", "none": "No item"}),
         "reorder_count": choice("How many items are at or below their reorder point?", {str(k): str(k) for k in range(5)}),
         "urgency": score("How urgent is restocking?", ["not urgent", "soon", "urgent", "critical"]),
         "any_below": noul("At least one item is below its reorder point.")},
        gold={"out_of_stock": "WASHER-6", "reorder_count": "2", "urgency": None, "any_below": "true"},
        note="inventory; choice ids are SKUs (mixed case sort)"))
    R.append(record("own_j08", "own_json",
        {"service": "billing-api", "environment": "production", "change_window": False,
         "diff": [{"key": "replicas", "old": 6, "new": 2}, {"key": "log_level", "old": "info", "new": "debug"},
                  {"key": "db_pool_size", "old": 50, "new": 50}, {"key": "feature_flags.new_invoice_pdf", "old": False, "new": True}]},
        {"risky": noul("The change is risky to deploy now."),
         "riskiest_change": choice("Which change carries the most risk?", {
             "replicas": {"key": "replicas", "effect": "serving capacity"},
             "log_level": {"key": "log_level", "effect": "log volume and cost"},
             "db_pool_size": {"key": "db_pool_size", "effect": "database connections"},
             "feature_flag": {"key": "feature_flags.new_invoice_pdf", "effect": "a new code path for invoices"}}),
         "review": score("What review does the change need?", ["auto-approve", "peer review", "senior review",
                                                               "block until a change window"])},
        gold={"risky": "true", "riskiest_change": "replicas", "review": None},
        note="config diff; option descriptions are JSON objects"))
    R.append(record("own_j09", "own_json",
        {"tracking": "TRK-4410-22", "promised_delivery": "2026-09-30",
         "events": [{"time": "2026-09-28 08:10", "status": "picked_up", "location": "Depot A"},
                    {"time": "2026-09-29 19:40", "status": "in_transit", "location": "Hub 3"},
                    {"time": "2026-09-30 07:15", "status": "exception", "detail": "address label unreadable", "location": "Hub 3"}]},
        {"status": choice("What is the current status of the parcel?", {"picked_up": "Picked up", "in_transit": "In transit",
                                                                        "exception": "Held because of a problem",
                                                                        "out_for_delivery": "Out for delivery", "delivered": "Delivered"}),
         "late": noul("The parcel will miss the promised delivery date."),
         "customer_action": noul("The customer needs to do something."),
         "escalation": score("How far should this be escalated?", ["no escalation", "team lead", "carrier liaison"])},
        gold={"status": "exception", "late": "true", "customer_action": None, "escalation": None},
        note="tracking history"))
    R.append(record("own_j10", "own_json",
        {"survey": "post-workshop", "responses": [{"q": "content", "score": 5}, {"q": "pace", "score": 2},
                                                  {"q": "materials", "score": 4}, {"q": "venue", "score": 3}],
         "comment": "Great examples, but we rushed through the last two sections."},
        {"weakest": choice("Which aspect scored worst?", {"content": "Content", "pace": "Pace", "materials": "Materials", "venue": "Venue"}),
         "overall": score("How satisfied is the respondent overall?", ["very dissatisfied", "dissatisfied", "neutral",
                                                                      "satisfied", "very satisfied"]),
         "satisfied": noul("The respondent is broadly satisfied.")},
        gold={"weakest": "pace", "overall": None, "satisfied": "true"},
        note="survey response"))
    R.append(record("own_j11", "own_json",
        [{"room": "A", "capacity": 4, "projector": True, "free": ["10:00", "14:00"]},
         {"room": "B", "capacity": 12, "projector": True, "free": ["14:00"]},
         {"room": "C", "capacity": 8, "projector": False, "free": ["10:00", "11:00", "14:00"]}],
        {"room": choice("Which room fits eight people with a projector at 14:00?", {"A": "Room A", "B": "Room B", "C": "Room C",
                                                                                   "none": "No room fits"}),
         "slot_10": choice("Which rooms are free at 10:00?", {"A_C": "Rooms A and C", "B": "Room B only", "all": "All rooms",
                                                             "none": "No room"}),
         "feasible": noul("A room with a projector for eight people is free at 10:00.")},
        gold={"room": "B", "slot_10": "A_C", "feasible": "false"},
        note="state is a top-level JSON array"))
    R.append(record("own_j12", "own_json",
        {"customer": {"name": "Jürgen Faß", "city": "São Paulo"},
         "basket": [{"item": "café crème", "price_eur": 3.8}, {"item": "crème brûlée", "price_eur": 6.5},
                    {"item": "Spätzle", "price_eur": 9.2}],
         "coupon": {"code": "AUTUMN10", "percent": 10, "valid_until": "2026-09-30"}, "today": "2026-10-03"},
        {"coupon_valid": noul("The coupon can still be used today."),
         "total_band": choice("What is the basket total before any discount?", {"under_10": "Under 10 EUR", "10_to_20": "10 to 20 EUR",
                                                                               "20_to_30": "20 to 30 EUR", "over_30": "Over 30 EUR"}),
         "dessert_items": choice("How many items are desserts?", {"0": "none", "1": "one", "2": "two", "3": "three"}),
         "basket_size": score("How large is the basket?", ["small", "medium", "large"])},
        gold={"coupon_valid": "false", "total_band": "10_to_20", "dessert_items": "1", "basket_size": None},
        note="non-ASCII strings in the state (ensure_ascii=False rendering)"))
    return R


def image_records():
    R = []
    R.append(record("img_01", "own_image", "Count the shapes in the attached picture.",
        {"circles": choice("How many circles are there?", {str(k): COUNT_WORDS[k - 1] for k in range(1, 7)}),
         "more_circles": noul("There are more circles than squares."),
         "clutter": score("How crowded is the picture?", ["sparse", "moderate", "crowded"])},
        gold={"circles": "3", "more_circles": "true", "clutter": None}, image_files=["shapes_224.png"],
        note="224x224"))
    R.append(record("img_02", "own_image", {"task": "Read the attached bar chart.", "bars": "left to right: red, orange, yellow, green, blue"},
        {"tallest": choice("Which bar is the tallest?", {"red": "The red bar", "orange": "The orange bar", "yellow": "The yellow bar",
                                                         "green": "The green bar", "blue": "The blue bar"}),
         "green_over_orange": noul("The green bar is taller than the orange bar."),
         "spread": score("How uneven are the bar heights?", ["about equal", "somewhat uneven", "uneven", "very uneven"])},
        gold={"tallest": "green", "green_over_orange": "true", "spread": None}, image_files=["bars_640x480.png"],
        note="640x480, JSON state"))
    R.append(record("img_03", "own_image", "The attached image shows a two-digit display.",
        {"tens_digit": choice("What is the left digit?", {str(k): str(k) for k in range(10)}),
         "even": noul("The displayed number is even."),
         "readability": score("How easy is the display to read?", ["hard", "moderate", "easy"])},
        gold={"tens_digit": "4", "even": "false", "readability": None}, image_files=["sevenseg_47_300x200.png"],
        note="300x200; a 10-option choice"))
    R.append(record("img_04", "own_image", {"task": "Read the attached receipt."},
        {"total": choice("What is the receipt total?", {"5.75": "5.75", "6.57": "6.57", "6.75": "6.75", "7.65": "7.65"}),
         "paid_cash": noul("The purchase was paid in cash."),
         "items": choice("How many different items were bought?", {str(k): COUNT_WORDS[k - 1] for k in range(1, 7)}),
         "legibility": score("How legible is the receipt?", ["illegible", "partly legible", "fully legible"])},
        gold={"total": "6.75", "paid_cash": "true", "items": "3", "legibility": None}, image_files=["receipt_1024x768.png"],
        note="1024x768, dot-matrix lettering, invented store name"))
    R.append(record("img_05", "own_image", "You are driving and approach this signal.",
        {"action": choice("What should the driver do?", {"stop": "Stop", "go": "Proceed", "slow": "Slow down and prepare to stop"}),
         "lit_color": choice("Which lamp is lit?", {"red": "Red", "yellow": "Yellow", "green": "Green"}),
         "may_proceed": noul("The driver may proceed."),
         "caution": score("How much caution is needed?", ["little", "some", "a lot"])},
        gold={"action": "go", "lit_color": "green", "may_proceed": "true", "caution": None}, image_files=["signal_green_256x240.png"],
        note="256x240"))
    R.append(record("img_06", "own_image", {"instrument": "boiler pressure gauge", "red_zone_starts_at_percent": 75},
        {"in_red_zone": noul("The needle is in the red zone."),
         "reading": choice("Where does the needle point?", {"0_25": "0 to 25 percent", "25_50": "25 to 50 percent",
                                                            "50_75": "50 to 75 percent", "75_100": "75 to 100 percent"}),
         "alarm": score("What alarm level fits the reading?", ["none", "advisory", "warning", "shutdown"])},
        gold={"in_red_zone": "true", "reading": "75_100", "alarm": None}, image_files=["gauge_300x200.png"],
        note="300x200, JSON state"))
    R.append(record("img_07", "own_image", "Find the odd one out among the arrows.",
        {"odd_direction": choice("Which way does the odd arrow point?", {"up": "Up", "down": "Down", "left": "Left", "right": "Right"}),
         "odd_position": choice("Which arrow is the odd one, counting from the left?", {"1": "the first", "2": "the second",
                                                                                      "3": "the third", "4": "the fourth"}),
         "all_same": noul("All arrows point the same way.")},
        gold={"odd_direction": "up", "odd_position": "2", "all_same": "false"}, image_files=["arrows_640x480.png"],
        note="640x480"))
    R.append(record("img_08", "own_image", {"task": "Answer questions about the attached floor plan."},
        {"largest_room": choice("Which room is the largest?", {"living": "Living room", "kitchen": "Kitchen", "bed": "Bedroom",
                                                               "bath": "Bathroom"}),
         "rooms": choice("How many labelled spaces are there?", {str(k): COUNT_WORDS[k - 1] for k in range(2, 7)}),
         "bath_next_to_bed": noul("The bathroom is next to the bedroom."),
         "complexity": score("How complex is the layout?", ["simple", "moderate", "complex"])},
        gold={"largest_room": "living", "rooms": "5", "bath_next_to_bed": "true", "complexity": None},
        image_files=["floorplan_1024x768.png"], note="1024x768, JSON state"))
    R.append(record("img_09", "own_image", "Compare the circles.",
        {"largest": choice("What color is the largest circle?", {"blue": "Blue", "red": "Red", "green": "Green"}),
         "smallest": choice("What color is the smallest circle?", {"blue": "Blue", "red": "Red", "green": "Green"}),
         "same_size": noul("All circles are the same size."),
         "spread": score("How different are the sizes?", ["similar", "somewhat different", "very different"])},
        gold={"largest": "red", "smallest": "blue", "same_size": "false", "spread": None}, image_files=["circles_640x480.png"],
        note="640x480"))
    R.append(record("img_10", "own_image", "The attached image shows a wall clock.",
        {"hour": choice("What hour does the clock show?", {"12": "twelve", "3": "three", "6": "six", "9": "nine"}),
         "minute_hand_up": noul("The minute hand points straight up."),
         "confidence": score("How clearly can the time be read?", ["unclear", "fairly clear", "very clear"])},
        gold={"hour": "3", "minute_hand_up": "true", "confidence": None}, image_files=["clock_3oclock_256x240.png"],
        note="256x240"))
    return R


def photo_records(available):
    R = []
    if "photo_fruit" in available:
        R.append(record("photo_01", "cc0_photo", {"task": "Check the fruit bowl in the attached photo."},
            {"limes": choice("How many limes are in the bowl?", {str(k): COUNT_WORDS[k - 1] for k in range(1, 7)}),
             "has_banana": noul("There is a banana in the bowl."),
             "kinds": choice("How many different kinds of fruit are there?", {str(k): COUNT_WORDS[k - 1] for k in range(2, 7)}),
             "freshness": score("How fresh does the fruit look?", ["spoiled", "acceptable", "fresh"])},
            gold={"limes": "3", "has_banana": "false", "kinds": "4", "freshness": None}, image_files=["photo_fruit.png"],
            note="CC0 photo, 960x1280"))
    if "photo_cat" in available:
        R.append(record("photo_02", "cc0_photo", "Identify the animal in the attached photo.",
            {"animal": choice("Which animal is shown?", {"cat": "A cat", "dog": "A dog", "rabbit": "A rabbit", "fox": "A fox"}),
             "eye_color": choice("What color are its eyes?", {"blue": "Blue", "green": "Green", "yellow": "Yellow or amber",
                                                                "brown": "Brown"}),
             "background": score("How busy is the background?", ["plain", "somewhat busy", "busy"]),
             "striped": noul("The animal's fur has stripes.")},
            gold={"animal": "cat", "eye_color": "yellow", "background": None, "striped": "true"}, image_files=["photo_cat.png"],
            note="CC0 photo, 960x640"))
    if "photo_lake" in available:
        R.append(record("photo_03", "cc0_photo", {"task": "Answer questions about the attached landscape photo."},
            {"snow_visible": noul("Snow is visible in the photo."),
             "reflection": noul("The water reflects the landscape."),
             "sky": choice("What is the sky like?", {"clear": "Clear and sunny", "overcast": "Overcast", "rain": "Raining",
                                                     "night": "Night"}),
             "wildness": score("How wild is the place?", ["urban", "suburban", "rural", "wilderness"])},
            gold={"snow_visible": "true", "reflection": "true", "sky": "overcast", "wildness": None}, image_files=["photo_lake.png"],
            note="CC0 photo, 960x640"))
    if "photo_coffee" in available:
        R.append(record("photo_04", "cc0_photo", "What is shown in the attached photo?",
            {"drink": choice("Which drink is in the cup?", {"tea": "Tea", "coffee": "Coffee", "milk": "Milk", "juice": "Juice"}),
             "milk_added": noul("Milk has been added to the drink."),
             "cup_color": choice("What color is the cup?", {"white": "White", "black": "Black", "red": "Red", "blue": "Blue"}),
             "beans": score("How many coffee beans are visible?", ["none", "a few", "many"])},
            gold={"drink": "coffee", "milk_added": "false", "cup_color": "white", "beans": "2"}, image_files=["photo_coffee.png"],
            note="CC0 photo, 960x635"))
    return R


def semif_records(path: Path):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SEMIF["sha256"], "authored144.jsonl changed"
    out = []
    for line in raw.decode().splitlines():
        r = json.loads(line)
        assert len(r["options"]) == 3 and isinstance(r["state"], str)
        criteria = {o["id"]: o["description"] for o in r["options"]}
        assert len(criteria) == 3
        gold = r["options"][r["label"]]["id"]
        out.append(record(f"semif_{r['id']}", "semif_authored144", r["state"],
                          {SEMIF_QUESTION_ID: choice(r["question"], criteria)},
                          gold={SEMIF_QUESTION_ID: gold},
                          note=f"family={r['family']} variant={r['provenance']['variant']} group={r['group_id']}"))
        out[-1]["semif"] = {"family": r["family"], "variant": r["provenance"]["variant"], "group_id": r["group_id"],
                            "label": r["label"], "option_order": [o["id"] for o in r["options"]]}
    assert len(out) == SEMIF["rows"]
    return out


# --------------------------------------------------------------------------- checks
def option_ids(q):
    if q["type"] == "noul":
        return ["true", "false"]
    if q["type"] == "choice":
        return sorted(str(k) for k in q["criteria"])
    return [str(i) for i in range(len(q["criteria"]))]


def check_records(records, image_names):
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate record id"
    own = [r for r in records if r["source"] != "semif_authored144"]
    stats = {"noul": 0, "choice": 0, "score": 0, "choice_10_options": 0, "no_instructions": 0,
             "json_object_descriptions": 0, "noul_overrides": 0, "max_questions": 0}
    for r in records:
        req = r["request"]
        assert req["model"] == MODEL and "state" in req and isinstance(req["questions"], dict) and req["questions"]
        assert 1 <= len(req["questions"]) <= 8, r["id"]
        for f in r.get("image_files", []):
            assert Path(f).stem in image_names, (r["id"], f)
        for qid, q in req["questions"].items():
            t = q["type"]
            assert t in ("noul", "choice", "score")
            if t == "choice":
                assert 2 <= len(q["criteria"]) <= 10, (r["id"], qid)
            if t == "score":
                assert 2 <= len(q["criteria"]) <= 7, (r["id"], qid)
            if t == "noul" and q.get("criteria"):
                assert set(q["criteria"]) <= {"true", "false"}, (r["id"], qid)
            g = r["gold"].get(qid)
            assert g is None or g in option_ids(q), (r["id"], qid, g)
            assert set(r["gold"]) <= set(req["questions"]), r["id"]
        text = json.dumps(req, ensure_ascii=False)
        for bad in ("http", "www.", "@", ".com"):
            assert bad not in text, (r["id"], bad)
    for r in own:
        qs = r["request"]["questions"]
        stats["max_questions"] = max(stats["max_questions"], len(qs))
        for q in qs.values():
            stats[q["type"]] += 1
            stats["choice_10_options"] += q["type"] == "choice" and len(q["criteria"]) == 10
            stats["no_instructions"] += "instructions" not in q
            stats["json_object_descriptions"] += q["type"] == "choice" and any(isinstance(v, dict) for v in q["criteria"].values())
            stats["noul_overrides"] += q["type"] == "noul" and bool(q.get("criteria"))
    assert stats["noul"] >= 20 and stats["choice"] >= 20 and stats["score"] >= 20, stats
    assert stats["choice_10_options"] >= 2 and stats["no_instructions"] >= 3, stats
    assert stats["json_object_descriptions"] >= 2 and stats["noul_overrides"] >= 2 and stats["max_questions"] == 8, stats
    return stats


def write_if_changed(path: Path, data: bytes) -> bool:
    """Replace `path` atomically and only when its bytes change, so a reader never sees a partial file."""
    if path.exists() and path.read_bytes() == data:
        return False
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return True


# =========================================================================== held-out set (--heldout)
# Written on 2026-10-03, after round 4, to re-check a choice tuned on the records above. Nothing below reuses a record,
# sentence, person, organisation or place name, image or photo from them; novelty_check asserts it against the round-1
# files. Image records are meant for the g256 / g448 arms only (there is no native arm for this set).
HELDOUT_SOURCES = {
    "heldout_text": "written for the held-out set (invented people, organisations, products and places)",
    "heldout_json": "written for the held-out set (invented)",
    "heldout_text_long": "written for the held-out set; state 1,800-2,200 tokens",
    "heldout_image": f"images drawn by make_fixtures.py ({IMAGE_LICENSE}); text written for the held-out set",
    "heldout_photo": ("Wikimedia Commons photographs whose extmetadata.LicenseShortName is CC0 (checked at fetch time); "
                      "960 px thumbnails, sha256 pinned in HELDOUT_PHOTOS; text written for the held-out set"),
}
HELDOUT_COUNTS = {"heldout_text": 10, "heldout_json": 8, "heldout_text_long": 2, "heldout_image": 8, "heldout_photo": 2}
HELDOUT_SHORT_STATE_MAX = 1000          # tokens; every state except the two long ones stays below this
HELDOUT_NAMES_SCREEN = {
    "method": "dig +short <stem>.<tld> A for tld in com, net, io, co.uk, app; a stem is used only when none resolves",
    "date": "2026-10-03",
    "stems_used": ["bramblequist", "brackwenna", "calderwisp", "cobbleswick", "corvassen", "drossingham", "ellerwyke",
                   "emberdowne", "fernacombe", "glenvarrow", "grallowmere", "harlowick", "hollindrake", "kinnerwold",
                   "lindenquay", "ostrevale", "ottervane", "pendrithy", "pottersmire", "quorrendale", "sallowmead",
                   "skelmardine", "tolverack", "tumbrelby", "vintlecombe", "yarrowdeep"],
    "stems_rejected_resolving": ["quibbleton"],
    "not_checked": "exact-phrase web search, App Store search, company registries (to do before any publication)",
}
# Every person named in a held-out request (invented); novelty_check asserts no part of any of them occurs in round 1.
HELDOUT_PEOPLE = ["Tamsin Okonkwo", "Bertrand", "Wiebke Antonelli", "Ilse Barrowclough", "Jovan Petrakis",
                  "Rosalind Achterberg", "Ezinne Valdés-Holt", "Augustin Lemaire-Kovac", "Florentyna Abernathy",
                  "Ottilie Varga", "Casimir Oyelaran", "Bartholomew", "Ingrid", "Cosima", "Barnaby", "Yevgenia", "Laszlo",
                  "Henrike", "Ludovica Ferreira", "Ruairi Esposito", "Saoirse Lindahl", "Fintan Gallagher",
                  "Solveig Arnesen", "Imogen Thackeray", "Dmitri Volkonsky", "Philippa Caradine", "Anselm Brightwater"]
NOVELTY_SHINGLE_WORDS = 8


# --------------------------------------------------------------------------- held-out images
def ho_pie_720x480():
    """A pie chart, clockwise from 12 o'clock: orange 40 %, blue 30 %, grey 20 %, yellow 10 %; legend HEAT / WATER /
    OTHER / LIGHT in dot-matrix lettering, 720x480."""
    im, d = canvas(720, 480, (250, 248, 240))
    cx, cy, r = 230, 240, 180
    slices = [("HEAT", 0.40, (240, 130, 30)), ("WATER", 0.30, (40, 90, 200)), ("OTHER", 0.20, (140, 140, 140)),
              ("LIGHT", 0.10, (235, 200, 40))]
    a = -90.0
    for _, frac, col in slices:
        d.pieslice([cx - r, cy - r, cx + r, cy + r], a, a + frac * 360, fill=col, outline="white", width=3)
        a += frac * 360
    for k, (label, _, col) in enumerate(slices):
        y = 120 + k * 70
        d.rectangle([470, y, 510, y + 40], fill=col)
        dot_text(d, label, 530, y + 3, 5, (30, 30, 30))
    return im


HO_DEPARTURES = [("TIME", "TO", "PLAT"), ("08:05", "GLENVARROW", "3"), ("08:20", "HARLOWICK", "1"),
                 ("08:40", "TUMBRELBY", "3"), ("09:10", "EMBERDOWNE", "2")]


def ho_table_1000x600():
    """A departure table with grid lines (TIME / TO / PLAT, four rows; the places are invented), dot-matrix, 1000x600."""
    im, d = canvas(1000, 600, (245, 245, 250))
    ink, line = (25, 25, 60), (60, 60, 90)
    cols = [30, 280, 760, 970]
    rows = [40, 140, 245, 350, 455, 560]
    d.rectangle([cols[0], rows[0], cols[-1], rows[1]], fill=(210, 220, 240))     # header row
    for x in cols:
        d.line([(x, rows[0]), (x, rows[-1])], fill=line, width=4)
    for y in rows:
        d.line([(cols[0], y), (cols[-1], y)], fill=line, width=4)
    dot = 7
    for r, row in enumerate(HO_DEPARTURES):
        y0 = (rows[r] + rows[r + 1]) // 2 - 7 * dot // 2
        for c, cell in enumerate(row):
            dot_text(d, cell, (cols[c] + cols[c + 1]) // 2 - dot_text_width(cell, dot) // 2, y0, dot, ink)
    return im


def ho_sign_200x150():
    """A round speed-limit sign reading 60 (red ring, dot-matrix digits) on a post against a pale sky, 200x150."""
    im, d = canvas(200, 150, (200, 225, 245))
    d.rectangle([0, 128, 199, 149], fill=(120, 150, 90))           # verge
    d.rectangle([96, 100, 104, 149], fill=(110, 110, 115))          # post
    cx, cy, r = 100, 62, 56
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(215, 30, 35))
    d.ellipse([cx - r + 11, cy - r + 11, cx + r - 11, cy + r - 11], fill="white")
    dot = 6
    dot_text(d, "60", cx - dot_text_width("60", dot) // 2, cy - 7 * dot // 2, dot, (20, 20, 20))
    return im


def ho_thermometer_160x400():
    """A liquid thermometer, scale -20 to 50 (labels every 10, ticks every 5, dot-matrix), red column up to 35, 160x400."""
    im, d = canvas(160, 400, (235, 240, 236))
    x0, x1, y_top, y_bottom = 100, 120, 30, 330

    def y_of(t):
        return y_bottom - (t + 20) * (y_bottom - y_top) / 70

    d.rounded_rectangle([x0, y_top - 15, x1, y_bottom + 10], radius=10, fill="white", outline=(90, 90, 90), width=3)
    d.ellipse([x0 - 14, y_bottom, x1 + 14, y_bottom + 48], fill=(210, 30, 30), outline=(90, 90, 90), width=3)  # bulb
    d.rectangle([x0 + 5, y_of(35), x1 - 5, y_bottom + 10], fill=(210, 30, 30))                               # column
    for t in range(-20, 51, 5):
        y = y_of(t)
        d.line([(x1 + 2, y), (x1 + (16 if t % 10 == 0 else 9), y)], fill=(40, 40, 40), width=2)
        if t % 10 == 0:
            dot_text(d, str(t), x0 - 14 - dot_text_width(str(t), 4), int(y) - 14, 4, (40, 40, 40))
    dot_text(d, "C", 138, 4, 4, (40, 40, 40))
    return im


def ho_calendar_760x700():
    """A wall-calendar page for FEBRUARY 2027 (the 1st is a Monday; 28 days, four full weeks), the 14th circled in red,
    the 22nd to the 26th (Monday to Friday) shaded blue, 760x700."""
    import datetime
    assert datetime.date(2027, 2, 1).weekday() == 0
    im, d = canvas(760, 700, "white")
    ink, grid = (30, 30, 40), (150, 150, 160)
    title = "FEBRUARY 2027"
    dot_text(d, title, 380 - dot_text_width(title, 6) // 2, 28, 6, (160, 30, 40))
    left, top, cw, ch = 30, 170, 100, 120
    for k, head in enumerate(("MO", "TU", "WE", "TH", "FR", "SA", "SU")):
        dot_text(d, head, left + k * cw + (cw - dot_text_width(head, 4)) // 2, 118, 4, ink)
    for day in range(1, 29):
        r, c = divmod(day - 1, 7)
        x, y = left + c * cw, top + r * ch
        if 22 <= day <= 26:
            d.rectangle([x, y, x + cw, y + ch], fill=(190, 215, 245))
        dot_text(d, str(day), x + (cw - dot_text_width(str(day), 5)) // 2, y + (ch - 35) // 2, 5, ink)
        if day == 14:
            d.ellipse([x + 12, y + 14, x + cw - 12, y + ch - 14], outline=(220, 30, 30), width=5)
    for k in range(8):
        d.line([(left + k * cw, top), (left + k * cw, top + 4 * ch)], fill=grid, width=2)
    for k in range(5):
        d.line([(left, top + k * ch), (left + 7 * cw, top + k * ch)], fill=grid, width=2)
    return im


def ho_digits_960x240():
    """Six green seven-segment digits 0 3 9 2 1 5 on a black strip; the rightmost digit sits on a red panel, 960x240."""
    im, d = canvas(960, 240, (30, 32, 36))
    d.rectangle([30, 30, 930, 210], fill=(8, 10, 8))
    for k, ch in enumerate("039215"):
        x = 60 + k * 145
        if k == 5:
            d.rectangle([x - 15, 38, x + 115, 202], fill=(150, 20, 25))
        seven_segment(d, ch, x, 50, 100, 140, 18, (60, 235, 90))
    return im


HO_SLIP_HEADER = ("COBBLESWICK", "DELI")
HO_SLIP_ITEMS = [("SOUP", "4.50"), ("ROLL", "1.20"), ("TEA", "2.10"), ("CAKE", "3.40")]
HO_SLIP_TOTALS = [("TOTAL", "11.20"), ("CARD", "11.20")]


def ho_receipt_480x840():
    """A cafe till slip (dot-matrix lettering) on a brown desk, 480x840; the shop name is invented."""
    im, d = canvas(480, 840, (120, 85, 60))
    d.rectangle([40, 20, 440, 820], fill=(252, 250, 240))
    ink, dot = (40, 40, 45), 5
    y = 50
    for line in HO_SLIP_HEADER:
        dot_text(d, line, 240 - dot_text_width(line, dot) // 2, y, dot, ink)
        y += 55
    y += 30
    for left, right in HO_SLIP_ITEMS:
        dot_text(d, left, 65, y, dot, ink)
        dot_text(d, right, 415 - dot_text_width(right, dot), y, dot, ink)
        y += 60
    d.line([(65, y), (415, y)], fill=ink, width=3)
    y += 25
    for left, right in HO_SLIP_TOTALS:
        dot_text(d, left, 65, y, dot, ink)
        dot_text(d, right, 415 - dot_text_width(right, dot), y, dot, ink)
        y += 60
    dot_text(d, "SEE YOU SOON", 240 - dot_text_width("SEE YOU SOON", 4) // 2, y + 70, 4, ink)
    return im


def ho_floorplan_1200x900():
    """An office floor plan, 1200x900: MEETING (top left), OFFICE (top right, the largest), LOBBY (bottom left, with the
    outside door and a mat), WC and STORE (bottom right, side by side, sharing a wall); dot-matrix labels."""
    bg = (252, 252, 248)
    im, d = canvas(1200, 900, bg)
    wall, w = (35, 35, 35), 10
    d.rectangle([80, 80, 1120, 820], outline=wall, width=w)
    d.line([(80, 470), (1120, 470)], fill=wall, width=w)           # top row | bottom row
    d.line([(460, 80), (460, 470)], fill=wall, width=w)            # meeting | office
    d.line([(620, 470), (620, 820)], fill=wall, width=w)           # lobby | wc
    d.line([(820, 470), (820, 820)], fill=wall, width=w)           # wc | store (no door)
    for box in ((180, 462, 260, 478),      # lobby -> meeting
                (500, 462, 580, 478),      # lobby -> office
                (612, 600, 628, 680),      # lobby -> wc
                (950, 462, 1030, 478),     # office -> store
                (260, 805, 380, 830)):     # outside door, bottom wall of the lobby
        d.rectangle(box, fill=bg)
    d.rectangle([255, 834, 385, 870], fill=(150, 150, 150))        # door mat outside
    ink = (40, 40, 140)
    for text, (cx, cy), dot in (("MEETING", (270, 275), 8), ("OFFICE", (790, 275), 8), ("LOBBY", (350, 645), 8),
                                ("WC", (720, 645), 6), ("STORE", (970, 645), 6)):
        dot_text(d, text, cx - dot_text_width(text, dot) // 2, cy - 7 * dot // 2, dot, ink)
    return im


# Invented place and shop names drawn into the held-out images (screened like the names in the text).
HELDOUT_DRAWN_NAMES = [row[1] for row in HO_DEPARTURES[1:]] + [HO_SLIP_HEADER[0]]
HELDOUT_IMAGES = {
    "ho_pie_720x480": (ho_pie_720x480, {}),
    "ho_table_1000x600": (ho_table_1000x600, {}),
    "ho_sign_200x150": (ho_sign_200x150, {}),
    "ho_thermometer_160x400": (ho_thermometer_160x400, {}),
    "ho_calendar_760x700": (ho_calendar_760x700, {}),
    "ho_digits_960x240": (ho_digits_960x240, {}),
    "ho_receipt_480x840": (ho_receipt_480x840, {}),
    "ho_floorplan_1200x900": (ho_floorplan_1200x900, {}),
}
HELDOUT_PHOTOS = {  # name -> Commons file, the 960 px thumbnail actually used, its sha256 (subjects new to this set)
    "ho_photo_lighthouse": {"title": "File:Cap de Barbaria lighthouse.jpg", "size": [960, 960],
                            "sha256": "6daf0cb176e6150a97ba138b0374c282df8ad0d15e80a47431a8f1287e8b0253"},
    "ho_photo_sunflowers": {"title": "File:Helianthus × laetiflorus flower (05).jpeg", "size": [960, 1280],
                            "sha256": "83ad983d60ace81885d370ee84fc820bb771d1124d5340dc0563443e0bfd88dd"},
}


def contact_sheet(items, cell=400, cols=5):
    """Every (name, image) in a grid, aspect kept (BICUBIC), with its name in dot-matrix lettering below; for review."""
    rows = (len(items) + cols - 1) // cols
    pitch_x, pitch_y = cell + 20, cell + 60
    im, d = canvas(20 + cols * pitch_x, 20 + rows * pitch_y, (235, 235, 235))
    for k, (name, src) in enumerate(items):
        r, c = divmod(k, cols)
        x0, y0 = 20 + c * pitch_x, 20 + r * pitch_y
        s = min(cell / src.width, cell / src.height)
        t = src.resize((max(1, round(src.width * s)), max(1, round(src.height * s))), Image.BICUBIC)
        im.paste(t, (x0 + (cell - t.width) // 2, y0 + (cell - t.height) // 2))
        d.rectangle([x0 - 1, y0 - 1, x0 + cell, y0 + cell], outline=(160, 160, 160))
        dot_text(d, name.upper(), x0, y0 + cell + 12, 3, (20, 20, 20))
    return im


# --------------------------------------------------------------------------- held-out long states
HELDOUT_TIMELINE = """SKELMARDINE COLD STORE (operated by Ostrevale Logistics)
Incident timeline: temperature excursion in freezer chamber 3, night of Friday 11 to Saturday 12 September 2026
Compiled by Imogen Thackeray, quality manager, from the building management system export, the shift log, the data loggers and interviews with the staff on duty. All times are local.

Background
Chamber 3 stores frozen goods at a set point of -24 °C. Its high-temperature alarm is set at -18 °C, and a pre-alarm text message goes out at -21 °C. The chamber is cooled by two compressors: compressor 1 carries the load, and compressor 2 is the standby unit, which starts by itself if compressor 1 stops. At midnight on 11 September the chamber held 412 pallets: 168 of ice cream for two retail customers, 196 of frozen vegetables (including the 22 pallets of peas delivered at 22:15) and 48 of frozen bread dough for a bakery chain. Chamber 5, next door, runs at -25 °C and had 190 empty pallet spaces that night.
At 15:00 on 11 September compressor 2 was isolated for a planned bearing replacement. The maintenance contractor expected to finish the work by 10:00 the next morning. No temporary cooling was hired for the period, because the job had been booked as low risk.

Timeline
21:40 The night shift starts. Shift lead Fintan Gallagher signs the handover sheet, which records "compressor 2 out of service until about 10:00 tomorrow". Chamber 3 reads -24.3 °C.
22:15 The last lorry of the day unloads 22 pallets of frozen peas into chamber 3. The loading door stands open for 11 minutes; the chamber warms to -22.9 °C and is back at its set point within 20 minutes, which is normal after a delivery.
23:58 Night operator Dmitri Volkonsky completes the routine walk-round. All doors are shut, chamber 3 reads -24.1 °C and there is no ice build-up on the evaporator coils.
00:36 Compressor 1 trips. The building management system logs "C1 motor contactor fault" and the compressor stops. Because compressor 2 is isolated, no standby unit takes over.
00:37 A "compressor tripped" warning appears on the screen in the control room. The control room is not staffed at night and the warning is not sent anywhere else, so nobody sees it.
01:20 Chamber 3 reaches -21.0 °C. The system sends the pre-alarm text by itself to the on-call refrigeration number stored in its contact list.
01:21 The text is delivered to a mobile phone that was handed back in July by an engineer who has since left the company. The contact list in the system had never been updated, so nobody reads the message.
02:47 Chamber 3 passes the alarm limit of -18 °C. The siren sounds in the dispatch area and a second text goes to the same old number.
02:52 Dmitri Volkonsky hears the siren while picking an order in chamber 1. He checks the alarm panel and phones Fintan Gallagher, who is in the office upstairs.
02:58 Fintan Gallagher confirms that compressor 1 has stopped and tries a reset from the panel. The compressor runs for about 40 seconds and trips again with the same fault.
03:05 Fintan Gallagher calls the refrigeration contractor's 24-hour line. The call handler says the on-call engineer, Solveig Arnesen, can be on site in about an hour.
03:10 All picking from chamber 3 is stopped and its doors are kept shut to hold in the cold. Chamber 3 reads -16.8 °C.
03:25 After a phone call with Imogen Thackeray, Fintan Gallagher decides to move the ice cream first, because it is the product most sensitive to warming, into the free spaces in chamber 5. The vegetables and the dough stay in chamber 3 for now.
03:31 Two reach-truck drivers start moving the ice cream pallets. Each door opening is kept as short as possible, and a strip curtain is hung across the doorway of chamber 3.
04:09 Solveig Arnesen arrives on site and starts fault-finding on compressor 1.
04:30 140 of the 168 ice cream pallets have been moved into chamber 5. Chamber 3 reads -14.9 °C.
04:44 Solveig Arnesen finds that the main contactor of compressor 1 has burnt out. The coil had overheated, and the contacts show heavy wear that must have built up over several years.
04:52 The remaining 28 ice cream pallets are now in chamber 5. Pallet moves stop, so the doors of chamber 3 stay shut during the repair.
05:15 Chamber 3 reaches its highest temperature of the night, -12.4 °C, recorded by both the wall probe and the independent data logger.
05:20 Solveig Arnesen fits a spare contactor from the stock in her van.
05:23 Compressor 1 restarts and runs normally. Chamber 3 starts to cool.
05:40 Fintan Gallagher asks the maintenance contractor to send a technician early to put compressor 2 back into service. A technician is promised for 08:00.
06:30 Chamber 3 reads -16.0 °C and is falling by roughly 1.5 °C every half hour.
07:12 Chamber 3 falls below the alarm limit of -18 °C again, and the alarm clears by itself.
08:05 The maintenance technician arrives and finishes fitting the new bearing in compressor 2.
09:40 Compressor 2 is back in service as the standby unit, earlier than the original plan.
10:15 Chamber 3 is back at its set point of -24 °C.

Product assessment (12 September, 09:00 to 13:00)
Imogen Thackeray took core temperatures from sample cartons on pallets in every product group and compared them with each customer's specification.
Ice cream: the 140 pallets moved before 04:30 never rose above -16 °C at the core. Both retail customers accept short rises to -15 °C, so these pallets were released. The last 28 pallets moved had core temperatures of -13 °C. Ice cream that has warmed that far loses its texture when it refreezes, so these 28 pallets were rejected and sent for destruction.
Frozen vegetables: the core temperatures stayed between -15 °C and -14 °C. The supplier's specification allows short rises to -12 °C for vegetables that will be cooked before eating, so all 196 pallets were released.
Bread dough: core temperatures reached -13.5 °C. The bakery chain asked for the 48 pallets to be held until its laboratory has tested whether the yeast is still active. The pallets were labelled as on hold and moved to the quarantine bay of chamber 5.

Corrective actions agreed on 14 September
1. The on-call contact list in the alarm system is to be checked every Monday by the shift lead, and again whenever an engineer joins or leaves.
2. A "compressor tripped" warning will also be sent by text to the shift lead on duty, not only shown on the control-room screen.
3. A standby compressor may be taken out of service for planned work only if a hired mobile cooling unit has been connected to the chamber first.
4. The contactors on all compressors in the building will be replaced now, and after that every five years.
5. Night operators will be trained to recognise a compressor fault on the panel and to call the contractor directly.

Cost of the incident
28 pallets of ice cream destroyed (about 14,300 in product value), the contractor's night call-out fee, 6.5 hours of overtime for the moving crew, and the cost of the bakery's laboratory tests if the dough is rejected.

Summary
The contactor failure itself was ordinary wear. The damage came from three gaps that lined up on one night: the standby compressor was out of service with no cover, the first warning was only shown on an empty screen, and the alarm texts went to a phone that no one carried. More than two hours passed between the trip at 00:36 and the moment someone on site knew about it at 02:52."""


HELDOUT_LEASE = """LEASE OF COMMERCIAL PREMISES (extract: particulars and clauses 1 to 15)

Particulars
Landlord: Drossingham Estates Limited
Tenant: Bramblequist Ceramics Limited
Premises: Unit 4, Lindenquay Trade Park, a single-storey workshop of about 640 square metres with a yard and six parking spaces, shown edged red on the attached plan
Term: ten (10) years starting on 1 November 2026 (the "Term Start Date")
Initial rent: 48,000 a year, exclusive of service charge, insurance rent and any tax
Rent-free period: the first three (3) months of the Term
Rent deposit: a sum equal to six (6) months of the initial rent
Permitted use: the design, manufacture, storage and sale of ceramic goods, with ancillary offices

1. Interpretation
1.1 In this lease, the "Quarter Days" are 1 January, 1 April, 1 July and 1 October in each year.
1.2 A reference to the Premises includes the Landlord's fixtures and fittings in them, but not the Tenant's own trade fixtures.
1.3 Headings are for convenience only and do not change how the lease is read.

2. Grant
2.1 The Landlord lets the Premises to the Tenant for the Term, together with the right to use the estate roads and the shared refuse area alongside the other occupiers of the trade park.
2.2 The Landlord keeps the right to enter the Premises, after giving at least forty-eight (48) hours' notice except in an emergency, to inspect them, to carry out repairs for which it is responsible, and to show them to possible buyers or, during the last six months of the Term, to possible new tenants.

3. Rent
3.1 The Tenant shall pay the rent by four equal instalments in advance on the Quarter Days. The first payment, apportioned for the part of the quarter it covers, falls due on the day after the rent-free period ends.
3.2 If any rent or other sum is more than fourteen (14) days late, the Tenant shall pay interest on it at four percent (4%) a year above the base rate of the Landlord's bank, from the due date until the day it is paid.

4. Rent Review
4.1 The rent shall be reviewed on the fifth anniversary of the Term Start Date (the "Review Date").
4.2 The reviewed rent shall be the higher of (a) the rent payable immediately before the Review Date and (b) the open market rent of the Premises at the Review Date, as agreed between the parties or, if they have not agreed within three months, as decided by an independent surveyor.
4.3 Until the reviewed rent has been decided, the Tenant shall go on paying the old rent, and shall pay any shortfall within twenty-one (21) days after the decision.

5. Service Charge and Insurance
5.1 The Tenant shall pay a fair share of the Landlord's costs of looking after the estate roads, landscaping, lighting and the shared refuse area. For the first three years of the Term this service charge shall not be more than 4,500 in any year.
5.2 The Landlord shall insure the building that contains the Premises against fire, storm, flood, subsidence and the other usual risks, for the full cost of rebuilding it, and the Tenant shall repay to the Landlord the part of the premium that relates to the Premises (the "insurance rent") within twenty-one days of a written demand.
5.3 If an insured risk makes the Premises unfit for use, no rent is payable until they are fit for use again, for at most three (3) years.

6. Repairs
6.1 The Landlord is responsible for keeping the roof, the foundations, the outside walls and the structural frame of the Premises in good repair.
6.2 The Tenant shall keep the inside of the Premises, including floor surfaces, doors, windows, glazing, electrical installations and the drains that serve only the Premises, in good repair and decorative order, and shall redecorate the inside in the last year of the Term.
6.3 The Tenant does not have to put the Premises into a better state of repair than the one shown in the photographic schedule of condition that both parties signed on the Term Start Date.

7. Use
7.1 The Tenant shall use the Premises only for the Permitted Use.
7.2 The Tenant may install and run electric or gas-fired kilns, as long as it (a) obtains any approval that the fire authority or the local council requires, (b) keeps an up-to-date fire risk assessment that covers the kilns, and (c) has every kiln inspected by a qualified engineer at least once a year.
7.3 The Tenant shall not keep on the Premises more flammable material than it reasonably needs for the Permitted Use, and shall not disturb the other occupiers with fumes, noise or vibration.

8. Alterations and Signs
8.1 The Tenant shall not make any structural alteration to the Premises and shall not cut into the roof or the outside walls.
8.2 The Tenant may make internal alterations that are not structural, such as putting up or taking down partitions, if the Landlord agrees in writing; the Landlord shall not refuse or delay its agreement without good reason.
8.3 The Tenant may show its name and logo on the fascia board of the Premises, in a size and style that the Landlord has approved beforehand.

9. Assignment and Underletting
9.1 The Tenant shall not assign, underlet, share or part with possession of only a part of the Premises.
9.2 The Tenant shall not underlet the whole of the Premises either.
9.3 The Tenant may assign the whole of the Premises if the Landlord gives its written consent beforehand, which the Landlord shall not refuse without good reason, and if the Tenant signs an authorised guarantee agreement for the obligations of the new tenant.

10. Break Clause
10.1 The Tenant may end this lease on the fifth anniversary of the Term Start Date by giving the Landlord at least six (6) months' notice in writing.
10.2 A notice under clause 10.1 only takes effect if, on the break date, (a) the Tenant has paid all the rent that has fallen due under this lease, and (b) the Tenant hands back the Premises with nobody occupying them and none of its goods left inside.
10.3 This clause gives the Landlord no right to end the lease early.

11. Rent Deposit
11.1 When this lease is signed the Tenant shall pay the rent deposit to the Landlord, who shall hold it in a separate account that earns interest.
11.2 The Landlord may take from the deposit any rent or other sum that has been unpaid for more than fourteen (14) days, and the Tenant shall then top the deposit back up within ten (10) working days.
11.3 The deposit, together with the interest earned on it, shall be returned within one month after the end of the Term, less any amounts the Landlord has properly taken from it.

12. Forfeiture
12.1 The Landlord may re-enter the Premises and bring this lease to an end if (a) any rent is still unpaid twenty-one (21) days after it falls due, whether or not it has been formally demanded, (b) the Tenant seriously breaks any other obligation and does not put it right within a reasonable time after being told about it, or (c) the Tenant goes into liquidation or administration.

13. Quiet Enjoyment
13.1 As long as the Tenant pays the rent and keeps to this lease, the Landlord shall not disturb the Tenant's use of the Premises.

14. End of the Term
14.1 At the end of the Term the Tenant shall hand back the Premises in the repair and condition that this lease requires, take away its trade fixtures, kilns and signs, and repair any damage caused by taking them away.
14.2 If the Landlord asks in writing at least six months before the end of the Term, the Tenant shall also take out any internal alterations it has made and put back the original layout.

15. Notices and Law
15.1 A notice under this lease is valid only if it is written down and either handed over in person or sent by recorded post to the registered office of the party receiving it.
15.2 This lease is governed by the law of the country in which the Premises are situated, and its courts alone may decide any dispute about it.

Signed for the Landlord by Philippa Caradine, director
Signed for the Tenant by Anselm Brightwater, director"""


# --------------------------------------------------------------------------- held-out records
def heldout_text_records():
    R = []
    R.append(record("ho_t01", "heldout_text",
        "Hi Bertrand, I would like to book annual leave from Monday 9 November to Friday 13 November, five working days, "
        "to attend my sister's wedding abroad. I still have 12 days of leave for this year, so 7 would remain afterwards. "
        "Wiebke Antonelli has agreed to take over the weekly supplier calls while I am away, and I will hand over the open "
        "purchase orders to her on Friday 6 November. I can be reached by phone in a real emergency, but I would rather "
        "not check messages. Could you let me know by the end of next week, so that I can book my flights? Many thanks, "
        "Tamsin Okonkwo",
        {"leave_days": choice("How many working days of leave are requested?", {"3": "three", "4": "four", "5": "five", "6": "six"}),
         "cover_arranged": noul("Someone has agreed to look after the requester's duties during the leave."),
         "days_left_after": choice("How many leave days will the requester have left afterwards?",
                                   {"5": "5 days", "7": "7 days", "12": "12 days", "17": "17 days"}),
         "reason": choice("Why is the leave requested?", {"family_event": "To attend a family event",
                                                          "medical": "For medical treatment or illness",
                                                          "study": "To sit an exam or attend a course", "moving": "To move house"}),
         "ease_of_approval": score("How easy would it be for a manager to approve this request?",
                                   ["hard", "needs a conversation", "straightforward"])},
        gold={"leave_days": "5", "cover_arranged": "true", "days_left_after": "7", "reason": "family_event",
              "ease_of_approval": None},
        note="leave request"))
    R.append(record("ho_t02", "heldout_text",
        "Warranty claim W-58213, received 28 September 2026. Customer: Ilse Barrowclough. Product: Tolverack TV-210 cordless "
        "vacuum cleaner, bought new on 2 February 2025 with a two-year manufacturer's warranty. Reported fault: the battery no "
        "longer holds its charge; after a whole night on the dock the cleaner runs for less than four minutes, against the "
        "forty minutes stated in the manual. She has kept the original receipt and quotes serial number TV210-77K-0419. She "
        "mentions that the cleaner fell down a flight of stairs last summer, but says it kept working normally for months "
        "after that. She asks for a replacement battery to be posted to her home address, because she no longer has a box to "
        "send the whole cleaner back in.",
        {"in_warranty": noul("On the date the claim was received, the product is still inside its warranty period."),
         "faulty_part": choice("Which part does the customer say is faulty?", {
             "battery": "The battery", "motor": "The motor", "brush": "The brush roll", "dock": "The charging dock",
             "filter": "The dust filter"}),
         "possible_damage": noul("Could the product have been damaged by the customer?", {
             "true": "The customer mentions an accident such as a fall, a spill or a knock.",
             "false": "No accident or rough handling is mentioned."}),
         "remedy": choice("What does the customer ask for?", {"refund": "Her money back",
                                                              "part_by_post": "A replacement part posted to her",
                                                              "repair": "A repair after sending the product in",
                                                              "new_unit": "A complete new product"}),
         "claim_strength": score("How strong is the claim?", ["weak", "uncertain", "strong"])},
        gold={"in_warranty": "true", "faulty_part": "battery", "possible_damage": "true", "remedy": "part_by_post",
              "claim_strength": None},
        note="warranty claim; noul criteria override (both keys)"))
    R.append(record("ho_t03", "heldout_text",
        "Dear Corvassen Lettings, I saw your advert for the two-bedroom flat on the third floor of Ellerwyke House "
        "(reference CL-2207) and would like to arrange a viewing. I work night shifts as a nurse, so weekday mornings after "
        "10:00 suit me best. I would be moving in with my partner and our elderly greyhound. The advert says small pets only, "
        "so please tell me whether a calm 30 kg dog would be accepted. We do not own a car, so the lack of a parking space "
        "is no problem for us. Could you also confirm whether the rent of 1,450 a month includes heating, and how large the "
        "deposit is? We hope to move in on 1 December. Kind regards, Jovan Petrakis",
        {"asks_about_pet": noul("The writer asks whether an animal would be allowed in the flat."),
         "bedrooms": choice(None, {"1": "one bedroom", "2": "two bedrooms", "3": "three bedrooms", "studio": "a studio"}),
         "needs_parking": noul("The writer needs a parking space."),
         "viewing_time": choice("When would the writer like to view the flat?", {
             "weekday_morning": "On a weekday morning", "weekday_evening": "On a weekday evening",
             "weekend": "At the weekend", "any": "No preference is given"}),
         "other_question": choice("Apart from pets, what else does the writer ask about?", {
             "rent_deposit": "What the rent covers and the size of the deposit", "garden": "Use of a garden",
             "lease_length": "How long the tenancy lasts", "internet": "The internet connection"}),
         "match": score("How well does the flat seem to suit the writer?", ["poorly", "partly", "well", "very well"])},
        gold={"asks_about_pet": "true", "bedrooms": "2", "needs_parking": "false", "viewing_time": "weekday_morning",
              "other_question": "rent_deposit", "match": None},
        note="rental enquiry; a choice without instructions"))
    R.append(record("ho_t04", "heldout_text",
        "Feedback on the Grallowmere Food Fair, Saturday. The tasting tents were the best part of the day: the cheese stall "
        "and the smoked fish stall both sold out by early afternoon. Parking was a mess. The overflow field opened an hour "
        "late and the queue of cars stretched back to the main road, so we spent forty minutes just getting in. Entry was 8 "
        "per adult, which felt fair, but charging children over five the full adult price did not. The cookery "
        "demonstrations started on time, although the speakers in the demo tent kept cutting out and we missed half of the "
        "talk on fermenting vegetables. The toilets stayed clean all day. We would come back next year if the parking is "
        "sorted out. Rosalind Achterberg",
        {"biggest_problem": choice("What was the biggest problem for this visitor?", {
             "parking": "Parking and getting in", "food": "The quality of the food", "price": "The entry price",
             "toilets": "The toilets", "weather": "The weather"}),
         "would_return": noul("Would the visitor come again?", {
             "true": "The visitor would attend again, possibly on a condition.",
             "false": "The visitor would not attend again."}),
         "food_left": noul("Every stall still had food at the end of the day."),
         "tone": score("How favourable is the feedback overall?", ["hostile", "very critical", "critical", "mixed",
                                                                  "favourable", "very favourable", "glowing"])},
        gold={"biggest_problem": "parking", "would_return": "true", "food_left": "false", "tone": None},
        note="event feedback; noul criteria override (both keys); a 7-level score"))
    R.append(record("ho_t05", "heldout_text",
        "Bug report, Calderwisp Notes 3.8.2 on a tablet with the latest system update. Steps: open a notebook that has more "
        "than 200 pages, switch to the outline view, then turn the tablet to landscape. Expected: the outline stays open on "
        "the same page. Actual: the app freezes for about ten seconds and then closes, and the last two minutes of typing "
        "are gone. It happens every time with large notebooks and never with small ones. Version 3.7.9 did not behave like "
        "this. Workaround: lock the screen rotation before opening the outline. Reported by Ezinne Valdés-Holt.",
        {"defect_kind": choice("What kind of defect is this?", {
             "crash": {"symptom": "the app stops responding or closes", "usual_cause": "an unhandled error"},
             "display": {"symptom": "something is drawn in the wrong place", "usual_cause": "a layout mistake"},
             "sync": {"symptom": "changes do not reach other devices", "usual_cause": "a server or network fault"},
             "slowness": {"symptom": "the app is slow but keeps working", "usual_cause": "inefficient processing"}}),
         "regression": noul("The problem did not exist in an earlier version."),
         "loses_work": noul("Some of the user's work is lost when the problem happens."),
         "small_notebooks": noul("The problem also occurs with small notebooks."),
         "priority": score("What priority should the fix get?", ["P4 backlog", "P3 normal", "P2 high", "P1 urgent"])},
        gold={"defect_kind": "crash", "regression": "true", "loses_work": "true", "small_notebooks": "false",
              "priority": None},
        note="bug report; option descriptions are JSON objects"))
    R.append(record("ho_t06", "heldout_text",
        "Order 66-1904, sent with Pendrithy Parcels for next-day delivery. The tracking page says 'delivered, left in a safe "
        "place' at 14:52 yesterday, but there was nothing at my door, in the porch or with the neighbours on either side. I "
        "was working from home all afternoon and nobody rang the bell. The parcel holds a replacement phone screen worth 89, "
        "and I need it for a repair I have booked on Thursday. The caretaker of our building has not received anything "
        "either. Please investigate, and either find the parcel or send a new one before Thursday. I do not want my money "
        "back. Augustin Lemaire-Kovac",
        {"problem": choice("What went wrong?", {
             "late": "The parcel is late", "damaged": "The parcel arrived damaged",
             "missing_after_scan": "Tracking says delivered but nothing arrived", "wrong_item": "The wrong item arrived",
             "redirected": "The parcel was sent to another address on request"}),
         "was_home": noul("The customer was at home when the delivery was recorded."),
         "outcome_wanted": choice("What outcome does the customer want?", {
             "refund": "A refund", "resend": "The parcel found or sent again", "compensation": "Extra compensation",
             "cancel": "The order cancelled"}),
         "time_sensitive": noul(),
         "mood": score("How upset does the customer sound?", ["calm", "concerned", "upset", "furious"])},
        gold={"problem": "missing_after_scan", "was_home": "true", "outcome_wanted": "resend", "time_sensitive": "true",
              "mood": None},
        note="delivery problem; a noul without instructions"))
    R.append(record("ho_t07", "heldout_text",
        "I am scaling up my grandmother's flapjack recipe for a school bake sale. The original makes one tray of 12 squares: "
        "250 g oats, 125 g butter, 100 g brown sugar, 4 tablespoons of golden syrup and a pinch of salt, baked at 180 °C for "
        "25 minutes. I need 36 squares, which means three trays. My kitchen scales only go up to 500 g and I have lost my "
        "measuring spoons, but I know that one tablespoon of syrup weighs about 20 g. Should the baking time change if I bake "
        "the trays one after another in the same oven? And can I replace half of the butter with sunflower oil without the "
        "bars falling apart?",
        {"factor": choice("By what factor must the quantities be multiplied?", {"1.5": "1.5", "2": "2", "3": "3", "4": "4", "6": "6"}),
         "oats_total": choice("How much oats is needed in total?", {"500": "500 g", "750": "750 g", "1000": "1,000 g",
                                                                    "1250": "1,250 g"}),
         "syrup_total": choice("How many grams of golden syrup are needed in total?", {"60": "60 g", "80": "80 g",
                                                                                      "240": "240 g", "320": "320 g"}),
         "oats_one_weighing": noul("All of the oats can be weighed in one go on the cook's scales."),
         "asks_swap": noul("The cook asks about replacing one ingredient with another."),
         "oven_temperature": choice("At what temperature are the trays baked?", {"160": "160 °C", "170": "170 °C",
                                                                                "180": "180 °C", "200": "200 °C"}),
         "questions_asked": choice("How many questions does the message ask?", {"1": "one", "2": "two", "3": "three", "4": "four"}),
         "difficulty": score("How tricky is this scaling job?", ["trivial", "easy", "moderate", "tricky"])},
        gold={"factor": "3", "oats_total": "750", "syrup_total": "240", "oats_one_weighing": "false", "asks_swap": "true",
              "oven_temperature": "180", "questions_asked": "2", "difficulty": None},
        note="recipe scaling; the 8-question record"))
    R.append(record("ho_t08", "heldout_text",
        "Policy SM-447120, Sallowmead Travel Cover, single trip from 3 to 17 August. My connecting flight home was cancelled "
        "because of a strike, and I had to stay two extra nights in a hotel before the airline could rebook me. I paid 236 "
        "for the hotel and 41 for meals and I have every receipt. The airline refused to pay, saying a strike is outside its "
        "control. My policy wording covers 'travel delay of more than 12 hours' up to 300, with an excess of 50 per claim, "
        "and excludes 'costs that another company has agreed to refund'. My suitcase also arrived a day late, but nothing in "
        "it was lost. What can I claim, and do I need a written statement from the airline? Florentyna Abernathy",
        {"policy_section": choice("Which section of the policy fits the main claim?", {
             "medical": "Emergency medical costs", "delay": "Travel delay", "baggage": "Lost or stolen baggage",
             "cancellation": "Cancelling the trip before leaving", "liability": "Damage to other people or their property"}),
         "costs_total": choice("What do the hotel and meal costs add up to?", {"236": "236", "277": "277", "286": "286", "300": "300"}),
         "payout": choice("If the claim is accepted in full, how much will the insurer pay?",
                          {"227": "227", "250": "250", "277": "277", "300": "300"}),
         "airline_paid": noul("The airline has agreed to refund the costs."),
         "baggage_lost": noul("Some of the traveller's baggage was lost for good."),
         "clarity": score(None, ["unclear", "mostly clear", "very clear"])},
        gold={"policy_section": "delay", "costs_total": "277", "payout": "227", "airline_paid": "false",
              "baggage_lost": "false", "clarity": None},
        note="travel insurance question; a score without instructions"))
    R.append(record("ho_t09", "heldout_text",
        "October timetable, Kinnerwold Learning Centre. Ottilie Varga teaches maths on Mondays and Wednesdays from 16:00 to "
        "17:30 in room 2. Casimir Oyelaran teaches chemistry on Tuesdays from 17:00 to 18:00 in room 5, and gives the same "
        "lesson online on Thursdays at 18:00. Each pupil may book at most three sessions a week. Bartholomew (Year 10) has "
        "football training every Tuesday from 16:30 to 18:30 and wants two maths sessions and one chemistry session each "
        "week. Ingrid (Year 12) is free every day after 15:30 but has no laptop at home. The centre is closed on Wednesday "
        "21 October for staff training; the lessons of that day move to Friday 23 October at the same times.",
        {"bartholomew_chemistry": choice("Which chemistry session can Bartholomew attend?", {
             "tuesday": "Tuesday in room 5", "thursday": "Thursday online", "neither": "Neither"}),
         "ingrid_laptop": noul("Ingrid has a laptop at home."),
         "within_limit": noul("Bartholomew's wished-for sessions stay within the weekly booking limit."),
         "moved_lesson": choice("What happens to the maths lesson of Wednesday 21 October?", {
             "cancelled": "It is cancelled", "fri_23": "It moves to Friday 23 October, same time",
             "thu_22": "It moves to Thursday 22 October", "online": "It is held online that day"}),
         "maths_length": choice("How long is one maths session?", {"60": "60 minutes", "75": "75 minutes", "90": "90 minutes",
                                                                   "120": "120 minutes"}),
         "planning_effort": score("How hard is this timetable to plan?", ["simple", "manageable", "complicated"])},
        gold={"bartholomew_chemistry": "thursday", "ingrid_laptop": "false", "within_limit": "true",
              "moved_lesson": "fri_23", "maths_length": "90", "planning_effort": None},
        note="tutoring schedule"))
    R.append(record("ho_t10", "heldout_text",
        "Replies to the September issue of the Yarrowdeep Bulletin.\n"
        "Cosima: Loved the article about the old mill. Please print more local history.\n"
        "Barnaby: Far too long this month. I gave up after page six.\n"
        "Yevgenia: Thanks for including the new bin collection days. Two of the crossword answers were wrong, though.\n"
        "Laszlo: More local history, please! The photographs were lovely too.\n"
        "Henrike: Could you stop posting it to me on paper? I would rather read it on my screen.\n"
        "Cosima: I agree with Henrike, the paper copies get soaked in my letterbox.",
        {"most_wanted": choice("Which topic do readers ask to see more of?", {
             "local_history": "Local history", "recipes": "Recipes", "sport": "Sport", "puzzles": "Puzzles",
             "council": "Council news"}),
         "mistake_reported": noul("A reader points out an error in the issue."),
         "all_positive": noul("Every reply is positive."),
         "digital_readers": choice("How many different readers want a digital copy instead of paper?",
                                   {"0": "none", "1": "one", "2": "two", "3": "three"}),
         "reply_count": score("How many replies are listed?", ["one or two", "three or four", "five or six", "more than six"]),
         "reception": score("How well was the issue received overall?", ["badly", "with mixed feelings", "well", "very well"])},
        gold={"most_wanted": "local_history", "mistake_reported": "true", "all_positive": "false", "digital_readers": "2",
              "reply_count": "2", "reception": None},
        note="newsletter replies"))
    return R


def heldout_json_records():
    R = []
    R.append(record("ho_j01", "heldout_json",
        {"timesheet": {"employee": "Ludovica Ferreira", "week_starting": "2026-09-28", "contract_hours": 37.5,
                       "days": [{"date": "2026-09-28", "in": "08:58", "out": "17:30", "break_min": 30},
                                {"date": "2026-09-29", "in": "09:05", "out": "17:35", "break_min": 30},
                                {"date": "2026-09-30", "status": "sick"},
                                {"date": "2026-10-01", "in": "08:45", "out": "19:15", "break_min": 45},
                                {"date": "2026-10-02", "in": "09:00", "out": "13:00", "break_min": 0, "note": "half-day leave"}],
                       "approved_by": None}},
        {"sick_days": choice("How many sick days are recorded?", {"0": "none", "1": "one", "2": "two", "3": "three"}),
         "longest_day": choice("On which date was the longest shift worked?", {
             "2026-09-28": "Monday", "2026-09-29": "Tuesday", "2026-10-01": "Thursday", "2026-10-02": "Friday"}),
         "approved": noul("Someone has approved the timesheet."),
         "late_start": noul("Did the employee start late on any day?", {
             "true": "On at least one day the clock-in time is later than 09:00.",
             "false": "Every clock-in time is 09:00 or earlier."}),
         "hours_concern": score("How concerning are the hours worked this week?", ["no concern", "minor concern", "serious concern"])},
        gold={"sick_days": "1", "longest_day": "2026-10-01", "approved": "false", "late_start": "true", "hours_concern": None},
        note="timesheet; noul criteria override (both keys)"))
    R.append(record("ho_j02", "heldout_json",
        {"stock_count": {"store": "Hollindrake Hardware", "counted_on": "2026-09-27", "tolerance_units": 3,
                         "lines": [{"sku": "HH-HAMMER-16", "expected": 42, "counted": 42},
                                   {"sku": "HH-TAPE-5M", "expected": 120, "counted": 113},
                                   {"sku": "HH-GLOVES-L", "expected": 60, "counted": 66},
                                   {"sku": "HH-DRILLBITS", "expected": 18, "counted": 9},
                                   {"sku": "HH-SEALANT-W", "expected": 35, "counted": 35}]}},
        {"beyond_tolerance": choice("How many lines differ from the expected count by more than the tolerance?",
                                    {"0": "none", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five"})},
        gold={"beyond_tolerance": "3"},
        note="inventory difference (expected vs counted); the 1-question record"))
    R.append(record("ho_j03", "heldout_json",
        {"log": "vaccine fridge B", "unit": "°C", "allowed_range": [2, 8],
         "entries": [{"time": "07:30", "reading": 4.1, "by": "TO"}, {"time": "11:30", "reading": 5.0, "by": "TO"},
                     {"time": "15:30", "reading": 8.9, "by": "RA", "comment": "door found ajar after delivery"},
                     {"time": "16:00", "reading": 7.2, "by": "RA"}, {"time": "19:30", "reading": 3.6, "by": "RA"}],
         "since_last_reset": {"max": 9.4, "min": 3.1}},
        {"excursion": noul("A reading outside the allowed range was logged."),
         "first_excursion": choice("At which logged time was the range first exceeded?", {
             "07:30": "07:30", "11:30": "11:30", "15:30": "15:30", "16:00": "16:00", "19:30": "19:30", "none": "never"}),
         "cause": choice("What does the log suggest as the cause?", {
             "door": "The door was not shut properly", "power": "A power cut", "defrost": "A defrost cycle",
             "sensor": "A faulty sensor"}),
         "recovered": noul("By the last entry the temperature is back inside the allowed range."),
         "risk": score("How serious is this for the stored vaccines?", ["negligible", "low", "moderate", "high", "severe"])},
        gold={"excursion": "true", "first_excursion": "15:30", "cause": "door", "recovered": "true", "risk": None},
        note="temperature log"))
    R.append(record("ho_j04", "heldout_json",
        {"reservation": {"ref": "OV-3318", "venue": "Ottervane Bistro", "date": "2026-10-17", "time": "19:30", "party_size": 6,
                         "requirements": [{"guest": 2, "need": "nut allergy"}, {"guest": 5, "need": "step-free access"}],
                         "deposit": {"required": True, "amount": 60, "paid": True}, "status": "confirmed",
                         "cancellation": "free until 48 hours before the booking; the deposit is kept after that",
                         "notes": "anniversary dinner, a quiet corner if possible"},
         "now": "2026-10-16T12:00"},
        {"free_cancellation": noul("Cancelling now would cost nothing."),
         "party": choice("How many people is the table for?", {"2": "two", "4": "four", "6": "six", "8": "eight"}),
         "kitchen_need": choice("Which requirement must the kitchen deal with?", {
             "allergy": {"handled_by": "kitchen", "detail": "an allergy"},
             "access": {"handled_by": "front of house", "detail": "step-free seating"},
             "high_chair": {"handled_by": "front of house", "detail": "a seat for a small child"},
             "vegan": {"handled_by": "kitchen", "detail": "a plant-based menu"}}),
         "special_occasion": noul()},
        gold={"free_cancellation": "false", "party": "6", "kitchen_need": "allergy", "special_occasion": "true"},
        note="reservation; option descriptions are JSON objects; a noul without instructions"))
    R.append(record("ho_j05", "heldout_json",
        {"member": 20417, "currency": "EUR", "plan": {"name": "Brackwenna Gym monthly", "fee": 39.0, "due_day": 1},
         "payments": [{"month": "2026-06", "paid_on": "2026-06-01", "amount": 39.0},
                      {"month": "2026-07", "paid_on": "2026-07-09", "amount": 39.0},
                      {"month": "2026-08", "paid_on": None, "amount": 0.0},
                      {"month": "2026-09", "paid_on": "2026-09-02", "amount": 78.0}],
         "late_fee_rule": "5.00 for each payment made more than 5 days after the due day"},
        {"missed_month": choice("Which month has no payment recorded?", {
             "2026-06": "June", "2026-07": "July", "2026-08": "August", "2026-09": "September", "none": "none"}),
         "july_late": noul("The July payment was made more than five days after the due day."),
         "double_payment": noul("The September payment equals two monthly fees."),
         "owed": choice("Ignoring late fees, how much is still owed for June to September?",
                        {"0": "nothing", "39": "39.00", "78": "78.00", "117": "117.00"}),
         "reliability": score("How reliable a payer is this member?", ["unreliable", "somewhat reliable", "reliable",
                                                                      "very reliable"])},
        gold={"missed_month": "2026-08", "july_late": "true", "double_payment": "true", "owed": "0", "reliability": None},
        note="payment history"))
    R.append(record("ho_j06", "heldout_json",
        {"app": "Fernacombe Run", "version": "5.2.0",
         "settings": {"units": "imperial", "auto_pause": True,
                      "voice_cues": {"enabled": True, "every_km": 1, "language": "it"},
                      "privacy": {"share_routes": "followers", "hide_start_end": False, "public_profile": True},
                      "notifications": {"weekly_summary": True, "challenges": False, "marketing": False},
                      "theme": "system", "heart_rate_zones": [120, 140, 160, 175]}},
        {"hides_endpoints": noul("The start and end points of routes are hidden."),
         "route_audience": choice("Who can see the user's routes?", {"everyone": "Anyone", "followers": "Followers only",
                                                                    "nobody": "Only the user"}),
         "marketing": noul("Marketing notifications are switched on.", {"false": "Marketing notifications are off or not set."}),
         "zone_boundaries": choice("How many heart-rate zone boundaries are set?", {"3": "3", "4": "4", "5": "5", "6": "6"}),
         "unit_mismatch": noul("The spoken cues use a different unit system from the one shown on screen."),
         "privacy_level": score("How privacy-conscious are these settings?", ["very open", "open", "moderate", "cautious",
                                                                             "very cautious"])},
        gold={"hides_endpoints": "false", "route_audience": "followers", "marketing": "false", "zone_boundaries": "4",
              "unit_mismatch": "true", "privacy_level": None},
        note="app settings; noul criteria override ('false' only)"))
    R.append(record("ho_j07", "heldout_json",
        [{"id": "VR-1101", "product": "Vintlecombe router X2", "opened": "2026-09-29", "priority": "high", "status": "open",
          "subject": "No internet after firmware update", "assignee": None},
         {"id": "VR-1102", "product": "Vintlecombe mesh point", "opened": "2026-09-30", "priority": "low",
          "status": "waiting_on_customer", "subject": "Mesh light blinks orange", "assignee": "Ruairi Esposito"},
         {"id": "VR-1103", "product": "Vintlecombe router X2", "opened": "2026-10-01", "priority": "high", "status": "open",
          "subject": "Firmware update stuck at 80%", "assignee": "Ruairi Esposito"},
         {"id": "VR-1104", "product": "Vintlecombe router X1", "opened": "2026-10-02", "priority": "medium", "status": "closed",
          "subject": "Parental controls schedule ignored", "assignee": "Saoirse Lindahl"}],
        {"unassigned": choice("Which ticket has nobody assigned?", {
             "VR-1101": "VR-1101", "VR-1102": "VR-1102", "VR-1103": "VR-1103", "VR-1104": "VR-1104",
             "none": "every ticket is assigned"}),
         "open_high": choice("How many tickets are open with high priority?", {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4"}),
         "shared_theme": choice("What do the high-priority tickets have in common?", {
             "firmware": "A firmware update", "parental": "Parental controls", "mesh": "Mesh points", "billing": "Billing"}),
         "all_closed": noul("All of the tickets are closed."),
         "wider_incident": score("Could a firmware release be behind a wider incident?", ["unlikely", "likely"])},
        gold={"unassigned": "VR-1101", "open_high": "2", "shared_theme": "firmware", "all_closed": "false",
              "wider_incident": None},
        note="support tickets; state is a top-level JSON array; a 2-level score"))
    R.append(record("ho_j08", "heldout_json",
        {"part": "bracket QB-7", "lab": "Quorrendale Labs", "batch": "B-2611", "unit": "mm",
         "spec": {"length": {"nominal": 120.0, "tol": 0.5}, "width": {"nominal": 35.0, "tol": 0.2},
                  "hole_diameter": {"nominal": 8.0, "tol": 0.05}, "thickness": {"nominal": 4.0, "tol": 0.1}},
         "samples": [{"n": 1, "length": 120.2, "width": 35.1, "hole_diameter": 8.03, "thickness": 4.02},
                     {"n": 2, "length": 119.7, "width": 34.9, "hole_diameter": 8.07, "thickness": 3.97},
                     {"n": 3, "length": 120.4, "width": 35.0, "hole_diameter": 8.01, "thickness": 4.05}]},
        {"all_pass": noul("Every sample is within tolerance on every dimension."),
         "failing_dimension": choice("Which dimension is out of tolerance?", {
             "length": "length", "width": "width", "hole_diameter": "hole diameter", "thickness": "thickness", "none": "none"}),
         "failing_sample": choice("Which sample fails?", {"1": "sample 1", "2": "sample 2", "3": "sample 3", "none": "no sample"}),
         "disposition": choice("What should happen to the batch?", {
             "release": {"action": "release the batch", "when": "every sample passes"},
             "hold": {"action": "hold the batch and measure more parts", "when": "one sample fails by a small margin"},
             "scrap": {"action": "scrap the batch", "when": "most samples fail by a wide margin"}}),
         "capability": score("How capable does the process look?", ["poor", "marginal", "adequate", "good"])},
        gold={"all_pass": "false", "failing_dimension": "hole_diameter", "failing_sample": "2", "disposition": "hold",
              "capability": None},
        note="QC measurements with tolerances; option descriptions are JSON objects"))
    return R


def heldout_long_records():
    R = []
    R.append(record("ho_l01", "heldout_text_long", HELDOUT_TIMELINE,
        {"root_cause": choice("What started the incident?", {
             "contactor": "An electrical contactor on a compressor failed", "door": "A loading door was left open",
             "power_cut": "The mains power failed", "refrigerant_leak": "Refrigerant leaked out",
             "sensor": "A temperature sensor gave false readings"}),
         "peak_temperature": choice("What was the highest air temperature recorded in chamber 3?", {
             "-18.0": "-18.0 °C", "-14.9": "-14.9 °C", "-12.4": "-12.4 °C", "-9.6": "-9.6 °C"}),
         "first_text_reached": noul("The first automatic alarm text reached someone who could act on it."),
         "dough_outcome": choice("What happened to the bread dough pallets?", {
             "released": "Released to the customer", "destroyed": "Sent for destruction",
             "held": "Held until laboratory tests are done", "left_in_chamber_3": "Left in chamber 3 without any check"}),
         "standby_ready": noul("A standby compressor was ready to take over when compressor 1 stopped."),
         "above_limit": choice("For how long was chamber 3 warmer than its alarm limit?", {
             "under_1h": "Less than an hour", "1_to_3h": "One to three hours", "3_to_6h": "Three to six hours",
             "over_6h": "More than six hours"}),
         "seriousness": score("How serious was the incident for the business?", ["minor", "moderate", "serious", "critical"])},
        gold={"root_cause": "contactor", "peak_temperature": "-12.4", "first_text_reached": "false", "dough_outcome": "held",
              "standby_ready": "false", "above_limit": "3_to_6h", "seriousness": None},
        note="long state: cold-store incident timeline"))
    R.append(record("ho_l02", "heldout_text_long", HELDOUT_LEASE,
        {"break_notice": choice("How much notice must the tenant give to use the break option?", {
             "3m": "3 months", "6m": "6 months", "9m": "9 months", "12m": "12 months", "none": "There is no break option"}),
         "review_direction": choice("How can the rent change at the review?", {
             "up_only": "It can stay the same or go up, but not down", "up_or_down": "It can go up or down",
             "index": "It follows a price index", "fixed": "It cannot change"}),
         "sublet_part": noul("The tenant may sublet part of the premises."),
         "roof": choice("Who must keep the roof in repair?", {"landlord": "The landlord", "tenant": "The tenant",
                                                              "shared": "Both, in equal shares", "insurer": "The insurer only"}),
         "deposit": choice("How large is the rent deposit?", {"12000": "12,000", "24000": "24,000", "48000": "48,000",
                                                              "none": "There is no deposit"}),
         "kilns": noul("The tenant may run kilns in the premises if it meets certain conditions."),
         "balance": score("Which side does the lease favour?", ["the landlord", "neither side", "the tenant"])},
        gold={"break_notice": "6m", "review_direction": "up_only", "sublet_part": "false", "roof": "landlord",
              "deposit": "24000", "kilns": "true", "balance": None},
        note="long state: commercial lease extract"))
    return R


def heldout_image_records():
    R = []
    R.append(record("ho_i01", "heldout_image", "A household's yearly energy use, split by purpose, is drawn as a pie chart.",
        {"biggest_slice": choice("Which colour is the biggest slice?", {"orange": "Orange", "blue": "Blue", "grey": "Grey",
                                                                       "yellow": "Yellow"}),
         "over_half": noul("A single slice takes up more than half of the circle."),
         "smallest_label": choice("Which legend label belongs to the smallest slice?", {
             "heat": "HEAT", "water": "WATER", "other": "OTHER", "light": "LIGHT"}),
         "blue_share": score("Roughly what share of the pie is blue?", ["under a quarter", "between a quarter and a half",
                                                                       "over a half"])},
        gold={"biggest_slice": "orange", "over_half": "false", "smallest_label": "light", "blue_share": "1"},
        image_files=["ho_pie_720x480.png"], note="720x480"))
    R.append(record("ho_i02", "heldout_image", {"task": "Look up trains on the departure table.", "station": "Pottersmire"},
        {"harlowick_platform": choice("Which platform does the Harlowick train leave from?", {
             "1": "Platform 1", "2": "Platform 2", "3": "Platform 3", "4": "Platform 4"}),
         "earliest": choice("Where does the earliest departure go?", {
             "glenvarrow": "Glenvarrow", "harlowick": "Harlowick", "tumbrelby": "Tumbrelby", "emberdowne": "Emberdowne"}),
         "platform_3": choice("How many departures use platform 3?", {"0": "none", "1": "one", "2": "two", "3": "three"}),
         "after_nine": noul("The table lists a departure later than 09:00.")},
        gold={"harlowick_platform": "1", "earliest": "glenvarrow", "platform_3": "2", "after_nine": "true"},
        image_files=["ho_table_1000x600.png"], note="1000x600 (5:3), JSON state"))
    R.append(record("ho_i03", "heldout_image", "A driver passes this road sign.",
        {"limit": choice("What limit does the sign show?", {str(v): str(v) for v in (20, 30, 40, 50, 60, 70, 80, 90, 100, 120)}),
         "outline": choice("What is the outline of the sign?", {"round": "Round", "triangle": "Triangular", "square": "Square",
                                                               "octagon": "Eight-sided"}),
         "seventy_ok": noul("Driving at 70 stays within the limit shown.")},
        gold={"limit": "60", "outline": "round", "seventy_ok": "false"},
        image_files=["ho_sign_200x150.png"], note="200x150; a 10-option choice"))
    R.append(record("ho_i04", "heldout_image", {"instrument": "outdoor thermometer", "scale": "degrees Celsius"},
        {"reading": choice("What temperature does the thermometer show?", {"15": "about 15", "25": "about 25", "35": "about 35",
                                                                          "45": "about 45"}),
         "below_zero": noul("The reading is below zero."),
         "warmth": score("How warm is it, going by the reading?", ["freezing", "cool", "mild", "hot"])},
        gold={"reading": "35", "below_zero": "false", "warmth": "3"},
        image_files=["ho_thermometer_160x400.png"], note="160x400 (portrait), JSON state"))
    R.append(record("ho_i05", "heldout_image", "One page of a wall calendar is included.",
        {"first_weekday": choice("On which weekday does the month begin?", {
             "mon": "Monday", "tue": "Tuesday", "wed": "Wednesday", "thu": "Thursday", "fri": "Friday", "sat": "Saturday",
             "sun": "Sunday"}),
         "circled": choice("Which date is circled?", {"7": "the 7th", "14": "the 14th", "21": "the 21st", "28": "the 28th"}),
         "month_length": choice("How many days does the month shown have?", {"28": "28", "29": "29", "30": "30", "31": "31"}),
         "shaded_weekend": noul("The shaded block of days includes a Saturday or a Sunday."),
         "shaded_count": score("How many days are shaded?", ["none", "one to three", "four to six", "seven or more"])},
        gold={"first_weekday": "mon", "circled": "14", "month_length": "28", "shaded_weekend": "false", "shaded_count": "2"},
        image_files=["ho_calendar_760x700.png"], note="760x700"))
    R.append(record("ho_i06", "heldout_image", {"device": "water meter", "task": "Report what the meter display shows."},
        {"third_digit": choice("What is the third digit from the left?", {str(k): str(k) for k in range(10)}),
         "digit_count": choice("How many digits does the display show?", {"4": "four", "5": "five", "6": "six", "7": "seven",
                                                                         "8": "eight"}),
         "has_seven": noul("The digit 7 appears on the display."),
         "red_last": noul("The rightmost digit sits on a red background.")},
        gold={"third_digit": "9", "digit_count": "6", "has_seven": "false", "red_last": "true"},
        image_files=["ho_digits_960x240.png"], note="960x240 (4:1), JSON state; a 10-option choice"))
    R.append(record("ho_i07", "heldout_image", "A customer kept this till slip.",
        {"total": choice("What total does the slip show?", {"10.20": "10.20", "11.20": "11.20", "11.40": "11.40", "12.10": "12.10"}),
         "paid_by": choice("How was the bill paid?", {"cash": "In cash", "card": "By card", "voucher": "With a voucher"}),
         "priciest": choice("Which item cost the most?", {"soup": "Soup", "roll": "Roll", "tea": "Tea", "cake": "Cake"}),
         "change_shown": noul("The slip shows change handed back to the customer.")},
        gold={"total": "11.20", "paid_by": "card", "priciest": "soup", "change_shown": "false"},
        image_files=["ho_receipt_480x840.png"], note="480x840 (portrait), dot-matrix lettering, invented shop name"))
    R.append(record("ho_i08", "heldout_image", {"task": "Study the office layout in the drawing."},
        {"biggest_space": choice("Which named space has the largest floor area?", {
             "meeting": "MEETING", "office": "OFFICE", "lobby": "LOBBY", "wc": "WC", "store": "STORE"}),
         "entrance": choice("Into which space does the outside door open?", {
             "meeting": "MEETING", "office": "OFFICE", "lobby": "LOBBY", "wc": "WC", "store": "STORE"}),
         "wc_store_wall": noul("The WC and the store share a wall."),
         "space_count": choice("How many named spaces does the plan show?", {"3": "three", "4": "four", "5": "five", "6": "six",
                                                                            "7": "seven"})},
        gold={"biggest_space": "office", "entrance": "lobby", "wc_store_wall": "true", "space_count": "5"},
        image_files=["ho_floorplan_1200x900.png"], note="1200x900, JSON state"))
    return R


def heldout_photo_records(available):
    R = []
    if "ho_photo_lighthouse" in available:
        R.append(record("ho_p01", "heldout_photo", "Questions follow about the building shown in the picture.",
            {"lamp_on": noul("The lamp at the top of the tower is shining."),
             "time_of_day": choice("When was the photo most likely taken?", {
                 "midday": "Around midday in bright sun", "twilight": "At dusk or dawn, with the sky still partly lit",
                 "night": "In the middle of a dark night", "grey_day": "On a grey, overcast day"}),
             "sea_behind": noul("The sea can be seen behind the tower."),
             "tower_colour": choice("What colour is the tower painted?", {"white": "White", "red_white": "Red and white bands",
                                                                        "black": "Black", "yellow": "Yellow"}),
             "remoteness": score("How remote does the place look?", ["in a busy town", "at the edge of a village", "isolated"])},
            gold={"lamp_on": "true", "time_of_day": "twilight", "sea_behind": "true", "tower_colour": "white",
                  "remoteness": None},
            image_files=["ho_photo_lighthouse.png"], note="CC0 photo, 960x960"))
    if "ho_photo_sunflowers" in available:
        R.append(record("ho_p02", "heldout_photo", {"task": "Describe the flowering plants pictured."},
            {"petal_colour": choice("What colour are the petals of the flowers in front?", {
                 "yellow": "Yellow", "white": "White", "purple": "Purple", "red": "Red"}),
             "insect": noul("An insect is sitting on one of the flowers."),
             "centre_colour": choice("What colour is the middle of the largest flower?", {
                 "brown": "Brown or dark orange", "green": "Green", "blue": "Blue", "white": "White"}),
             "setting": choice("Where are the flowers?", {"vase": "In a vase indoors",
                                                          "outdoors": "Outdoors among leaves and other plants",
                                                          "sand": "In bare sand", "snow": "In snow"}),
             "density": score("How dense is the vegetation?", ["sparse", "moderate", "dense"])},
            gold={"petal_colour": "yellow", "insect": "true", "centre_colour": "brown", "setting": "outdoors",
                  "density": None},
            image_files=["ho_photo_sunflowers.png"], note="CC0 photo, 960x1280"))
    return R


# --------------------------------------------------------------------------- held-out checks
def render_value(value):
    """The author's `render`: a string as is, anything else compact JSON (sorted keys, ensure_ascii off)."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def check_heldout_records(records, image_names):
    """check_records' per-record asserts (plus the author's request validation) and the held-out set's own counts."""
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate record id"
    stats = {"noul": 0, "choice": 0, "score": 0, "choice_10_options": 0, "no_instructions": 0,
             "json_object_descriptions": 0, "noul_overrides": 0, "min_questions": 8, "max_questions": 0}
    types_by_source = {}
    for r in records:
        req = r["request"]
        assert isinstance(req.get("model"), str) and req["model"] == MODEL and "state" in req, r["id"]
        assert isinstance(req["questions"], dict) and req["questions"], r["id"]
        n = len(req["questions"])
        assert 1 <= n <= 8, r["id"]
        if r["source"] in ("heldout_image", "heldout_photo"):
            assert 2 <= n <= 5 and len(r.get("image_files", [])) == 1, r["id"]
        else:
            assert "image_files" not in r, r["id"]
        if r["source"] == "heldout_text_long":
            assert 5 <= n <= 8, r["id"]
        for f in r.get("image_files", []):
            assert Path(f).stem in image_names, (r["id"], f)
        assert set(r["gold"]) == set(req["questions"]), r["id"]          # every question has a gold entry (or None)
        stats["min_questions"] = min(stats["min_questions"], n)
        stats["max_questions"] = max(stats["max_questions"], n)
        for qid, q in req["questions"].items():
            t = q["type"]
            assert t in ("noul", "choice", "score"), (r["id"], qid)
            if t != "noul":
                assert q.get("criteria"), (r["id"], qid)
            if t == "choice":
                assert 2 <= len(q["criteria"]) <= 10, (r["id"], qid)
            if t == "score":
                assert 2 <= len(q["criteria"]) <= 7, (r["id"], qid)
            if t == "noul" and q.get("criteria"):
                assert set(q["criteria"]) <= {"true", "false"}, (r["id"], qid)
            g = r["gold"][qid]
            assert g is None or g in option_ids(q), (r["id"], qid, g)
            assert g is not None or t == "score", (r["id"], qid)             # only opinion-like scores lack a gold
            stats[t] += 1
            stats["choice_10_options"] += t == "choice" and len(q["criteria"]) == 10
            stats["no_instructions"] += "instructions" not in q
            stats["json_object_descriptions"] += t == "choice" and any(isinstance(v, dict) for v in q["criteria"].values())
            stats["noul_overrides"] += t == "noul" and bool(q.get("criteria"))
            types_by_source.setdefault(r["source"], set()).add(t)
        text = json.dumps(req, ensure_ascii=False).lower()
        for bad in ("http", "www.", "@", ".com"):
            assert bad not in text, (r["id"], bad)
    by_source = {}
    for r in records:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    assert len(records) == sum(HELDOUT_COUNTS.values()) and by_source == HELDOUT_COUNTS, by_source
    assert all(v == {"noul", "choice", "score"} for v in types_by_source.values()), types_by_source
    assert stats["noul"] >= 15 and stats["choice"] >= 15 and stats["score"] >= 15, stats
    assert stats["choice_10_options"] >= 1 and stats["no_instructions"] >= 2, stats
    assert stats["json_object_descriptions"] >= 2 and stats["noul_overrides"] >= 2, stats
    assert stats["min_questions"] == 1 and stats["max_questions"] == 8, stats
    return stats


def _text_pieces(request):
    """What the author renders as text for one request: the state, each instructions, each option description."""
    pieces = [render_value(request["state"])]
    for q in request["questions"].values():
        if q.get("instructions"):
            pieces.append(render_value(q["instructions"]))
        crit = q.get("criteria") or {}
        pieces += [render_value(v) for v in (crit.values() if isinstance(crit, dict) else crit)]
    return pieces


def _words(text):
    """Lowercase runs of letters and digits; everything else (punctuation, underscores, spaces) separates words."""
    return re.findall(r"[^\W_]+", text.lower())


def _shingles(text, n=NOVELTY_SHINGLE_WORDS):
    w = _words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def novelty_check(heldout_records, round1_records_json: Path, round1_images_meta: Path, heldout_meta) -> dict:
    """Assert that the held-out set shares nothing with round 1: no record id, image name, PNG or raw-RGB sha256, person
    name, invented name stem, identical state or instruction, or 8-word shingle. Returns the record for records.json."""
    raw1 = round1_records_json.read_bytes()
    meta_raw1 = round1_images_meta.read_bytes()
    r1 = json.loads(raw1)["records"]
    m1 = json.loads(meta_raw1)["images"]
    r1_text = raw1.decode("utf-8").lower()
    ho_text = "\n".join(json.dumps(r["request"], ensure_ascii=False) for r in heldout_records).lower()
    ho_text += "\n" + " ".join(HELDOUT_DRAWN_NAMES).lower()        # invented names that are only drawn

    shared_ids = {r["id"] for r in heldout_records} & {r["id"] for r in r1}
    shared_names = {m["name"] for m in heldout_meta} & {m["name"] for m in m1}
    shared_files = ({f for r in heldout_records for f in r.get("image_files", [])}
                    & {f for r in r1 for f in r.get("image_files", [])})
    shared_rgb = {m["rgb_sha256"] for m in heldout_meta} & {m["rgb_sha256"] for m in m1}
    shared_png = {m["sha256"] for m in heldout_meta} & {m["sha256"] for m in m1}
    assert not (shared_ids or shared_names or shared_files or shared_rgb or shared_png), \
        (shared_ids, shared_names, shared_files, shared_rgb, shared_png)

    shared_people = []
    for person in HELDOUT_PEOPLE:
        assert person.lower() in ho_text, ("HELDOUT_PEOPLE lists a name the records do not use", person)
        for part in re.split(r"[\s\-]+", person):
            if re.search(r"\b" + re.escape(part.lower()) + r"\b", r1_text):
                shared_people.append(part)
    round1_stems = set(NAMES_SCREEN["stems_used"]) | set(NAMES_SCREEN["stems_rejected_resolving"])
    ho_stems = HELDOUT_NAMES_SCREEN["stems_used"]
    for stem in ho_stems:
        assert stem in ho_text, ("HELDOUT_NAMES_SCREEN lists a stem the records do not use", stem)
    shared_stems = sorted({s for s in ho_stems if s in round1_stems or s in r1_text}
                          | {s for s in HELDOUT_NAMES_SCREEN["stems_rejected_resolving"] if s in round1_stems}
                          | {s for s in round1_stems if s in ho_text})
    assert not shared_people and not shared_stems, (shared_people, shared_stems)

    # Identical states or instructions (any length), then 8-word shingles over every text piece.
    def statements(recs):
        out = set()
        for r in recs:
            out.add(" ".join(_words(render_value(r["request"]["state"]))))
            out |= {" ".join(_words(render_value(q["instructions"]))) for q in r["request"]["questions"].values()
                    if q.get("instructions")}
        return out - {""}
    shared_statements = statements(heldout_records) & statements(r1)
    assert not shared_statements, sorted(shared_statements)[:10]
    sh1 = set().union(*(_shingles(p) for r in r1 for p in _text_pieces(r["request"])))
    sh_ho = {}
    for r in heldout_records:
        for p in _text_pieces(r["request"]):
            for s in _shingles(p):
                sh_ho.setdefault(s, r["id"])
    shared_shingles = sorted((sh_ho[s], " ".join(s)) for s in set(sh_ho) & sh1)
    assert not shared_shingles, shared_shingles[:20]
    return {
        "compared_with": {"records_json": "fixtures/records.json", "records_json_sha256": hashlib.sha256(raw1).hexdigest(),
                          "images_meta_json": "fixtures/images_meta.json",
                          "images_meta_json_sha256": hashlib.sha256(meta_raw1).hexdigest(),
                          "records": len(r1), "images": len(m1)},
        "method": {"text": ("per record: the state (a string as is, anything else compact JSON with sorted keys), each "
                            "instructions and each option description or score level, as separate pieces"),
                   "words": "lowercased runs of letters and digits; punctuation, underscores and spaces separate words",
                   "shingle_words": NOVELTY_SHINGLE_WORDS,
                   "people": "every part of every HELDOUT_PEOPLE name, whole word, case-insensitive, in the round-1 records.json",
                   "stems": ("HELDOUT_NAMES_SCREEN stems as substrings of the round-1 records.json and of its names screen; "
                             "round-1 stems (used and rejected) as substrings of the held-out requests"),
                   "statements": "identical states or instructions after the word rule, any length"},
        "heldout": {"records": len(heldout_records), "images": len(heldout_meta), "shingles": len(sh_ho),
                    "people": len(HELDOUT_PEOPLE), "stems": len(ho_stems)},
        "round1_shingles": len(sh1),
        "overlaps": {"record_ids": 0, "image_names": 0, "image_png_sha256": 0, "image_rgb_sha256": 0, "person_names": 0,
                     "stems": 0, "identical_states_or_instructions": 0, "shingles_8_words": 0},
    }


def _png(im):
    """PNG bytes of an RGB image, asserting the lossless round trip."""
    assert im.mode == "RGB"
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    png = buf.getvalue()
    assert Image.open(io.BytesIO(png)).convert("RGB").tobytes() == im.tobytes()
    return png


def main_heldout(args) -> int:
    out = Path(args.out_dir or work_path("_clefflash", "fixtures", "heldout")).expanduser()
    round1 = Path(args.round1_dir).expanduser()
    assert out.resolve() != round1.resolve(), "the held-out set must not be written over the round-1 fixtures"
    (out / "images").mkdir(parents=True, exist_ok=True)

    meta, changed, sheet = [], [], []
    for name, (fn, kw) in HELDOUT_IMAGES.items():
        im = fn(**kw)
        png = _png(im)
        if write_if_changed(out / "images" / f"{name}.png", png):
            changed.append(f"{name}.png")
        meta.append({"name": name, "path": f"images/{name}.png", "size": list(im.size),
                     "sha256": hashlib.sha256(png).hexdigest(), "rgb_sha256": hashlib.sha256(im.tobytes()).hexdigest(),
                     "generator": {"script": "conversion/clef_flash/make_fixtures.py", "function": fn.__name__, "args": kw},
                     "license": IMAGE_LICENSE, "license_note": IMAGE_LICENSE_NOTE})
        sheet.append((name, im))

    src = out / "photos_src"
    fetched = fetch_photos(src, HELDOUT_PHOTOS) if args.fetch_photos else (
        json.loads((src / "fetch_record.json").read_text()) if (src / "fetch_record.json").exists() else {})
    available = []
    for name, ph in HELDOUT_PHOTOS.items():
        jpg = src / f"{name}.jpg"
        if not jpg.exists() or name not in fetched:
            raise SystemExit(f"photo {name}: not fetched; run once with --heldout --fetch-photos")
        data = jpg.read_bytes()
        assert hashlib.sha256(data).hexdigest() == ph["sha256"], (name, "thumbnail changed")
        im = Image.open(io.BytesIO(data)).convert("RGB")
        assert list(im.size) == ph["size"], (name, im.size)
        png = _png(im)
        if write_if_changed(out / "images" / f"{name}.png", png):
            changed.append(f"{name}.png")
        meta.append({"name": name, "path": f"images/{name}.png", "size": list(im.size),
                     "sha256": hashlib.sha256(png).hexdigest(), "rgb_sha256": hashlib.sha256(im.tobytes()).hexdigest(),
                     "source": {"commons_title": ph["title"], "file_page": fetched[name]["file_page"],
                                "thumb_url": fetched[name]["thumb_url"], "thumb_jpeg_sha256": ph["sha256"],
                                "decode": "Pillow JPEG decode -> RGB -> PNG (the raw-RGB sha256 is the Pillow build's decode)"},
                     "license": "CC0-1.0",
                     "license_note": f"Wikimedia Commons extmetadata.LicenseShortName = {fetched[name]['license_short_name']}"})
        available.append(name)
        sheet.append((name, im))

    records = (heldout_text_records() + heldout_json_records() + heldout_long_records() + heldout_image_records()
               + heldout_photo_records(available))
    stats = check_heldout_records(records, set(HELDOUT_IMAGES) | set(available))

    # State token counts with the checkpoint's own tokenizer (pinned snapshot), as encode_record tokenizes them.
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(hf_snapshot(HF_ID, "tokenizer.json", revision=REVISION))
    for r in records:
        r["state_tokens"] = len(tok.encode(render_value(r["request"]["state"]), add_special_tokens=False).ids)
        if r["source"] == "heldout_text_long":
            lo, hi = LONG_STATE_TOKENS
            assert lo <= r["state_tokens"] <= hi, (r["id"], r["state_tokens"])
        else:
            assert r["state_tokens"] < HELDOUT_SHORT_STATE_MAX, (r["id"], r["state_tokens"])

    novelty = novelty_check(records, round1 / "records.json", round1 / "images_meta.json", meta)
    by_source = {}
    for r in records:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    doc = {
        "schema": "clef-flash-fixtures/1",
        "heldout": True,
        "model": MODEL,
        "checkpoint": {"hf_id": HF_ID, "revision": REVISION},
        "description": ("Held-out confirmation set, written on 2026-10-03 after round 4 by a separate step, to re-check a "
                        "quantization choice that was tuned on the round-1 fixtures (fixtures/records.json). No earlier step "
                        "has seen it: every text, JSON state, person and organisation name, drawn image and photograph is "
                        "new, and novelty_check asserts that no record id, image name or pixel hash, person name, invented "
                        "name stem, identical state or instruction, or 8-word shingle is shared with the round-1 files "
                        "(see `novelty`). Use it only to confirm a decision already made: anything tuned or selected on it "
                        "spends it, and a new held-out set is then needed."),
        "request_form": ("request is the /v1/systemone body joint_schema_model.systemone() takes; for image records the "
                         "runner adds request['images'] = [PIL RGB image per image_files entry] for each arm"),
        "image_arms": {"g256": "PIL BICUBIC resize to 256x256 before the processor",
                       "g448": "PIL BICUBIC resize to 448x448 before the processor"},
        "text_arm": "text",
        "sources": HELDOUT_SOURCES,
        "names_screen": HELDOUT_NAMES_SCREEN,
        "counts": {"records": len(records), "by_source": by_source, "question_stats": stats},
        "novelty": novelty,
        "records": records,
    }
    for name, data in (("records.json", (json.dumps(doc, indent=1, ensure_ascii=False) + "\n").encode()),
                       ("images_meta.json", (json.dumps({"license": IMAGE_LICENSE, "license_note": IMAGE_LICENSE_NOTE,
                                                         "images": meta}, indent=1, ensure_ascii=False) + "\n").encode()),
                       ("contact_sheet.png", _png(contact_sheet(sheet)))):
        if write_if_changed(out / name, data):
            changed.append(name)
    longs = {r["id"]: r["state_tokens"] for r in records if r["source"] == "heldout_text_long"}
    short_max = max(r["state_tokens"] for r in records if r["source"] != "heldout_text_long")
    print(f"{len(records)} held-out records {by_source}; question stats {stats}; long states {longs}; "
          f"largest other state {short_max} tokens; novelty: {novelty['heldout']['shingles']} shingles vs "
          f"{novelty['round1_shingles']} round-1, overlaps {novelty['overlaps']} -> {out}")
    print(f"files rewritten: {changed if changed else 'none (all byte-identical)'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=None,
                    help="default $ZOO_WORK_ROOT/_clefflash/fixtures; with --heldout, $ZOO_WORK_ROOT/_clefflash/fixtures/heldout")
    ap.add_argument("--semif", default=SEMIF["path"])
    ap.add_argument("--fetch-photos", action="store_true", help="download the pinned CC0 thumbnails into <out-dir>/photos_src")
    ap.add_argument("--heldout", action="store_true", help="write the held-out set (HELDOUT_*) instead of the round-1 fixtures")
    ap.add_argument("--round1-dir", default=str(work_path("_clefflash", "fixtures")),
                    help="with --heldout: the round-1 fixtures novelty_check compares against (read only)")
    ap.add_argument("--withheld", default=str(WITHHELD_FILE),
                    help="private copy of the withheld round-1 requests (WITHHELD_REQUEST_SHA256)")
    args = ap.parse_args()
    if args.heldout:
        return main_heldout(args)
    withheld = load_withheld(Path(args.withheld).expanduser())
    out = Path(args.out_dir or work_path("_clefflash", "fixtures")).expanduser()
    (out / "images").mkdir(parents=True, exist_ok=True)

    meta, changed = [], []
    for name, (fn, kw) in IMAGES.items():
        im = fn(**kw)
        assert im.mode == "RGB"
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        png = buf.getvalue()
        assert Image.open(io.BytesIO(png)).convert("RGB").tobytes() == im.tobytes(), name  # lossless round trip
        if write_if_changed(out / "images" / f"{name}.png", png):
            changed.append(f"{name}.png")
        meta.append({"name": name, "path": f"images/{name}.png", "size": list(im.size),
                     "sha256": hashlib.sha256(png).hexdigest(), "rgb_sha256": hashlib.sha256(im.tobytes()).hexdigest(),
                     "generator": {"script": "conversion/clef_flash/make_fixtures.py", "function": fn.__name__, "args": kw},
                     "license": IMAGE_LICENSE, "license_note": IMAGE_LICENSE_NOTE})

    src = out / "photos_src"
    fetched = fetch_photos(src) if args.fetch_photos else (json.loads((src / "fetch_record.json").read_text())
                                                           if (src / "fetch_record.json").exists() else {})
    available = []
    for name, ph in PHOTOS.items():
        jpg = src / f"{name}.jpg"
        if not jpg.exists() or name not in fetched:
            print(f"photo {name}: not fetched (run with --fetch-photos); its record is skipped")
            continue
        data = jpg.read_bytes()
        assert hashlib.sha256(data).hexdigest() == ph["sha256"], (name, "thumbnail changed")
        im = Image.open(io.BytesIO(data)).convert("RGB")
        assert list(im.size) == ph["size"], (name, im.size)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        png = buf.getvalue()
        assert Image.open(io.BytesIO(png)).convert("RGB").tobytes() == im.tobytes(), name
        if write_if_changed(out / "images" / f"{name}.png", png):
            changed.append(f"{name}.png")
        meta.append({"name": name, "path": f"images/{name}.png", "size": list(im.size),
                     "sha256": hashlib.sha256(png).hexdigest(), "rgb_sha256": hashlib.sha256(im.tobytes()).hexdigest(),
                     "source": {"commons_title": ph["title"], "file_page": fetched[name]["file_page"],
                                "thumb_url": fetched[name]["thumb_url"], "thumb_jpeg_sha256": ph["sha256"],
                                "decode": "Pillow JPEG decode -> RGB -> PNG (the raw-RGB sha256 is the Pillow build's decode)"},
                     "license": "CC0-1.0",
                     "license_note": f"Wikimedia Commons extmetadata.LicenseShortName = {fetched[name]['license_short_name']}"})
        available.append(name)

    records = (own_text_records(withheld) + own_json_records(withheld) + image_records() + photo_records(available)
               + semif_records(Path(args.semif).expanduser()))
    stats = check_records(records, set(IMAGES) | set(available))

    # State token counts with the checkpoint's own tokenizer (pinned snapshot), as encode_record tokenizes them.
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(hf_snapshot(HF_ID, "tokenizer.json", revision=REVISION))
    for r in records:
        s = r["request"]["state"]
        text = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        r["state_tokens"] = len(tok.encode(text, add_special_tokens=False).ids)
    for r in records:
        if r["source"] == "own_text_long":
            lo, hi = LONG_STATE_TOKENS
            assert lo <= r["state_tokens"] <= hi, (r["id"], r["state_tokens"])

    by_source = {}
    for r in records:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    doc = {
        "schema": "clef-flash-fixtures/1",
        "model": MODEL,
        "checkpoint": {"hf_id": HF_ID, "revision": REVISION},
        "request_form": ("request is the /v1/systemone body joint_schema_model.systemone() takes; for image records the "
                         "runner adds request['images'] = [PIL RGB image per image_files entry] for each arm"),
        "image_arms": {"native": "the image as drawn; the processor picks the grid",
                       "g256": "PIL BICUBIC resize to 256x256 before the processor",
                       "g448": "PIL BICUBIC resize to 448x448 before the processor"},
        "text_arm": "text",
        "sources": {
            "own_text": "written for this fixture (invented people, organisations, products and places)",
            "own_text_long": "written / generated for this fixture; state about 1,800-2,200 tokens",
            "own_json": "written for this fixture (invented)",
            "own_image": f"images drawn by make_fixtures.py ({IMAGE_LICENSE}); text written for this fixture",
            "cc0_photo": ("Wikimedia Commons photographs whose extmetadata.LicenseShortName is CC0 (checked at fetch time); "
                          "960 px thumbnails, sha256 pinned in PHOTOS; text written for this fixture"),
            "semif_authored144": dict(SEMIF, path=record_path(SEMIF["path"]), mapping=("state -> state; question -> questions['decision'].instructions; "
                                                      "options -> criteria {id: description}; type choice; "
                                                      "gold = options[label].id")),
        },
        "names_screen": NAMES_SCREEN,
        "counts": {"records": len(records), "by_source": by_source, "own_question_stats": stats,
                   "semif_questions": sum(r["source"] == "semif_authored144" for r in records)},
        "records": records,
    }
    for name, obj in (("records.json", doc), ("images_meta.json", {"license": IMAGE_LICENSE, "license_note": IMAGE_LICENSE_NOTE,
                                                                   "images": meta})):
        if write_if_changed(out / name, (json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode()):
            changed.append(name)
    longs = {r["id"]: r["state_tokens"] for r in records if r["source"] == "own_text_long"}
    print(f"{len(records)} records {by_source}; own question stats {stats}; long states {longs} -> {out}")
    print(f"files rewritten: {changed if changed else 'none (all byte-identical)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
