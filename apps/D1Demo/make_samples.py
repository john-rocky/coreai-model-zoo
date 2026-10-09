#!/usr/bin/env python3
"""The D1Demo samples: a support inbox of 12 messages (8 texts, 4 with a picture of the delivery), the order log of the
port's fixture, and the warm-up requests. Everything is written here, nothing is downloaded: the texts are new and name
no person, company or product; the four pictures are drawn by this script (PIL, CC0-1.0); the log is the fixture's
`long_34k` state (written for the d1-3B port, no names).

    python make_samples.py --fixtures <lane>/fixtures --out <dir>

writes <dir>/samples.json (what the app shows and the request it sends for every item, with the answer the author
expected when the item was written), <dir>/pictures/*.png (384 x 384: one crop, 144 image tokens), <dir>/requests/
<id>.json (each item's request with its picture path, the shape conversion/d1/decide.py run reads) and
<dir>/media_sources.json (licence and sha256 of every picture). The pictures come out the same bytes on every run
(no randomness but a seeded generator, PNG without metadata).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE = 384          # the picture's side: one crop of 24 x 24 patches, 144 image tokens
SS = 4              # drawn at 4x, then downsampled
W = SIZE * SS

# ---------------------------------------------------------------------------------------------------------- questions
TEXT_QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Charges, refunds, invoices", "shipping": "Deliveries, lost or late parcels",
                          "technical": "App or site faults", "fraud": "Suspected unauthorised use"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["Can wait", "Today", "Blocking the customer now"]},
}
PICTURE_QUESTIONS = {
    "damage": {"type": "score", "instructions": "How damaged is the package in the picture?",
               "criteria": ["Not damaged", "Dented or scuffed", "Crushed or torn"]},
    "boxes": {"type": "choice", "instructions": "How many boxes are in the picture?",
              "criteria": {"one": "One", "two": "Two", "three": "Three or more"}},
    "opened": {"type": "noul", "instructions": "Is the box open or its tape torn?"},
}
# what the screen calls each question and answer (the request's own words where they are short)
DISPLAY = {
    "refund": {"label": "Refund", "yes": "Asked", "no": "Not asked", "chip_yes": "Refund", "chip_no": "No refund"},
    "team": {"label": "Team", "options": {"billing": "Billing", "shipping": "Shipping", "technical": "Technical",
                                          "fraud": "Fraud"}},
    "urgency": {"label": "Urgency", "levels": ["Can wait", "Today", "Now"],
                "chips": ["Can wait", "Today", "Urgent now"]},
    "damage": {"label": "Damage", "levels": ["None", "Dented", "Crushed or torn"],
               "chips": ["No damage", "Dented", "Crushed"]},
    "boxes": {"label": "Boxes", "options": {"one": "One", "two": "Two", "three": "Three or more"},
              "chips": {"one": "1 box", "two": "2 boxes", "three": "3+ boxes"}},
    "opened": {"label": "Opened or torn tape", "yes": "Yes", "no": "No", "chip_yes": "Opened", "chip_no": "Sealed"},
    "refunded_twice": {"label": "Which order was refunded twice?",
                       "options": {"o00402": "Order 00402", "o00406": "Order 00406", "o00409": "Order 00409",
                                   "o00411": "Order 00411"}},
    "big_refunds_approved": {"label": "Did the supervisor approve every refund over 500.00?", "yes": "Yes",
                             "no": "No"},
    "cancellations": {"label": "How many orders were cancelled?", "levels": ["None", "One", "Two", "Three or more"]},
}

# ------------------------------------------------------------------------------------------------------------- inbox
# kind, text, picture (drawn below), the answer the author expected when writing it
INBOX = [
    ("m01", "text", "I was charged twice this month, please refund one of them.", None,
     {"refund": "yes", "team": "billing", "urgency": 1}),
    ("m02", "text", "The tracking page says my parcel was delivered yesterday, but nothing came.", None,
     {"refund": "no", "team": "shipping", "urgency": 1}),
    ("m03", "picture", "This is how it arrived.", "collapsed",
     {"damage": 2, "boxes": "one", "opened": "no"}),
    ("m04", "text", "There are two orders on my account that I didn't place. Please lock it now.", None,
     {"refund": "no", "team": "fraud", "urgency": 2}),
    ("m05", "text", "The app crashes every time I open my cart, so I can't check out.", None,
     {"refund": "no", "team": "technical", "urgency": 2}),
    ("m06", "picture", "Photo of the box I got today.", "intact",
     {"damage": 0, "boxes": "one", "opened": "no"}),
    ("m07", "text", "I sent the shoes back three weeks ago and still haven't got my money.", None,
     {"refund": "yes", "team": "billing", "urgency": 1}),
    ("m08", "text", "Could you deliver my next order to my office instead?", None,
     {"refund": "no", "team": "shipping", "urgency": 0}),
    ("m09", "picture", "It was left at my door like this.", "opened",
     {"damage": 2, "boxes": "one", "opened": "yes"}),
    ("m10", "text", "The password reset email never arrives. I've tried four times.", None,
     {"refund": "no", "team": "technical", "urgency": 1}),
    ("m11", "text", "My card was charged for an order in another country while I was at home.", None,
     {"refund": "no", "team": "fraud", "urgency": 2}),
    ("m12", "picture", "Here is my delivery.", "two",
     {"damage": 0, "boxes": "two", "opened": "no"}),
]
LOG_QUESTIONS = ("refunded_twice", "big_refunds_approved", "cancellations")   # the record's first three, in its order
WARMUP_TEXT = "Please send my invoices to the new billing address from now on."


# ------------------------------------------------------------------------------------------------------------ drawing
def rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


def noise(seed: int, scale: float, blur: float = 0.0) -> np.ndarray:
    """A W x W field of grain in [-1, 1] (seeded), optionally softened."""
    g = np.random.default_rng(seed).standard_normal((W, W)).astype(np.float32)
    if blur:
        im = Image.fromarray(np.clip(g * 40 + 128, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur))
        g = (np.asarray(im, np.float32) - 128) / 40
    return np.clip(g, -3, 3) * scale


class Canvas:
    def __init__(self, seed: int):
        self.seed = seed
        self.px = np.zeros((W, W, 3), np.float32)
        self.k = 0

    def grain(self, scale: float, blur: float = 0.0) -> np.ndarray:
        self.k += 1
        return noise(self.seed * 1000 + self.k, scale, blur)

    def mask(self, polys, blur: float = 0.0) -> np.ndarray:
        m = Image.new("L", (W, W), 0)
        d = ImageDraw.Draw(m)
        for p in polys:
            d.polygon([(x * SS, y * SS) for x, y in p], fill=255)
        if blur:
            m = m.filter(ImageFilter.GaussianBlur(blur * SS))
        return np.asarray(m, np.float32)[..., None] / 255

    def fill(self, polys, color: str, light=(1.0, 1.0), axis: str = "y", grain: float = 6.0, fiber: float = 0.0,
             alpha: float = 1.0, blur: float = 0.0):
        """Paint the polygons (coordinates in 384 units) with a colour shaded along an axis, grain and fibres."""
        m = self.mask(polys, blur) * alpha
        ys, xs = np.mgrid[0:W, 0:W].astype(np.float32) / W
        pts = np.array([pt for p in polys for pt in p], np.float32) / SIZE
        lo, hi = (pts[:, 1].min(), pts[:, 1].max()) if axis == "y" else (pts[:, 0].min(), pts[:, 0].max())
        t = np.clip(((ys if axis == "y" else xs) - lo) / max(hi - lo, 1e-6), 0, 1)
        f = light[0] + (light[1] - light[0]) * t
        col = rgb(color)[None, None, :] * f[..., None]
        if grain:
            col = col + self.grain(grain)[..., None]
        if fiber:
            fib = noise(self.seed * 1000 + 500 + self.k, 1.0, 0)
            fib = np.asarray(Image.fromarray(np.clip(fib * 40 + 128, 0, 255).astype(np.uint8))
                             .filter(ImageFilter.BoxBlur(1)).resize((W // 8, W), Image.BILINEAR)
                             .resize((W, W), Image.BILINEAR), np.float32)
            col = col + ((fib - 128) / 40 * fiber)[..., None]
        self.px = self.px * (1 - m) + col * m

    def line(self, pts, color: str, width: float, alpha: float = 1.0, blur: float = 0.6):
        m = Image.new("L", (W, W), 0)
        ImageDraw.Draw(m).line([(x * SS, y * SS) for x, y in pts], fill=255, width=max(1, round(width * SS)),
                               joint="curve")
        if blur:
            m = m.filter(ImageFilter.GaussianBlur(blur * SS))
        a = np.asarray(m, np.float32)[..., None] / 255 * alpha
        self.px = self.px * (1 - a) + rgb(color)[None, None, :] * a

    def shadow(self, polys, strength: float, blur: float):
        m = self.mask(polys, blur) * strength
        self.px = self.px * (1 - m)

    def image(self) -> Image.Image:
        im = Image.fromarray(np.clip(self.px, 0, 255).round().astype(np.uint8), "RGB")
        return im.resize((SIZE, SIZE), Image.LANCZOS)


def scene(c: Canvas):
    """A porch: a pale wall, a skirting board, a concrete floor and a dark doormat."""
    c.fill([[(0, 0), (SIZE, 0), (SIZE, 128), (0, 128)]], "#ebe6de", light=(1.02, 0.95), grain=3)
    c.fill([[(0, 120), (SIZE, 120), (SIZE, 132), (0, 132)]], "#d4ccc0", light=(1.0, 0.92), grain=2)
    c.fill([[(0, 132), (SIZE, 132), (SIZE, SIZE), (0, SIZE)]], "#c3bcb1", light=(1.04, 0.86), grain=9)
    mat = [(52, 214), (332, 214), (368, 350), (16, 350)]
    c.shadow([[(p[0] + 2, p[1] + 3) for p in mat]], 0.25, 4)
    c.fill([mat], "#4a4d50", light=(0.95, 1.08), grain=10)
    inner = [(64, 222), (320, 222), (352, 340), (32, 340)]
    c.fill([inner], "#55585b", light=(0.95, 1.06), grain=12)


class Box:
    """A carton in an oblique view: the front face x0..x1 by yb - h..yb, the depth offset (dx, -dy)."""

    def __init__(self, x0: float, yb: float, w: float, h: float, d: float):
        self.x0, self.yb, self.w, self.h = x0, yb, w, h
        self.dx, self.dy = 0.55 * d, 0.42 * d
        x1, yt = x0 + w, yb - h
        self.FTL, self.FTR, self.FBL, self.FBR = (x0, yt), (x1, yt), (x0, yb), (x1, yb)
        self.BTL, self.BTR, self.BBR = (x0 + self.dx, yt - self.dy), (x1 + self.dx, yt - self.dy), (x1 + self.dx, yb - self.dy)

    def floor_shadow(self, c: Canvas):
        x0, yb, w, dx, dy = self.x0, self.yb, self.w, self.dx, self.dy
        c.shadow([[(x0 - 8, yb + 4), (x0 + w + 6, yb + 6), (x0 + w + dx + 26, yb - dy + 10), (x0 + dx + 4, yb - dy - 2)]],
                 0.42, 7)

    def faces(self, c: Canvas, ftr=None, crushed: bool = False):
        FTR = ftr or self.FTR
        c.fill([[self.FTL, FTR, self.FBR, self.FBL]], "#c39760", light=(1.03, 0.9), grain=7, fiber=5)
        c.fill([[FTR, self.BTR, self.BBR, self.FBR]], "#a47a46", light=(0.98, 0.84), grain=7, fiber=5)

    def top(self, c: Canvas, ftr=None):
        FTR = ftr or self.FTR
        c.fill([[self.FTL, FTR, self.BTR, self.BTL]], "#dbb683", light=(1.0, 1.04), axis="x", grain=6, fiber=4)

    def edges(self, c: Canvas, ftr=None, top: bool = True):
        FTR = ftr or self.FTR
        e = "#7d5d36"
        c.line([self.FBL, self.FTL, FTR, self.FBR, self.FBL], e, 1.1, 0.8)
        c.line([FTR, self.BTR, self.BBR, self.FBR], e, 1.1, 0.8)
        if top:
            c.line([self.FTL, self.BTL, self.BTR], e, 1.1, 0.7)

    def seam(self, c: Canvas):
        """The flaps' seam across the top, front to back."""
        m0 = ((self.FTL[0] + self.FTR[0]) / 2, self.FTL[1])
        m1 = ((self.BTL[0] + self.BTR[0]) / 2, self.BTL[1])
        c.line([m0, m1], "#9c7848", 0.9, 0.7)

    def tape(self, c: Canvas, top: bool = True, front: float = 0.32):
        """Packing tape along the seam and down the front face."""
        tw = self.w * 0.16
        mx = self.x0 + self.w / 2
        yt = self.yb - self.h
        if top:
            c.fill([[(mx - tw / 2, yt), (mx + tw / 2, yt), (mx + tw / 2 + self.dx, yt - self.dy),
                     (mx - tw / 2 + self.dx, yt - self.dy)]], "#e9d7ae", light=(1.05, 0.97), axis="x", grain=3, alpha=0.9)
        c.fill([[(mx - tw / 2, yt), (mx + tw / 2, yt), (mx + tw / 2, yt + self.h * front),
                 (mx - tw / 2, yt + self.h * front)]], "#dcc79c", light=(1.04, 0.96), grain=3, alpha=0.9)
        c.line([(mx - tw / 2 + 1.5, yt + 2), (mx - tw / 2 + 1.5, yt + self.h * front - 2)], "#fff7e6", 0.8, 0.35)

    def label(self, c: Canvas):
        """A blank shipping label on the front face: white, a few grey bars, no text."""
        x, y = self.x0 + self.w * 0.12, self.yb - self.h * 0.52
        lw, lh = self.w * 0.34, self.h * 0.30
        c.shadow([[(x + 1.5, y + 2), (x + lw + 1.5, y + 2), (x + lw + 1.5, y + lh + 2), (x + 1.5, y + lh + 2)]], 0.18, 1.5)
        c.fill([[(x, y), (x + lw, y), (x + lw, y + lh), (x, y + lh)]], "#f4f2ec", light=(1.0, 0.96), grain=2)
        for k, frac in enumerate((0.72, 0.5, 0.62)):
            yy = y + lh * (0.2 + 0.17 * k)
            c.line([(x + lw * 0.1, yy), (x + lw * (0.1 + frac * 0.8), yy)], "#a9a59c", 1.6, 0.8, blur=0.3)
        for k in range(14):   # bars of a code
            xx = x + lw * (0.1 + 0.058 * k)
            c.line([(xx, y + lh * 0.72), (xx, y + lh * 0.9)], "#5d5a55", 0.6 + 0.9 * (k % 3 == 0), 0.85, blur=0.2)


def draw_intact(c: Canvas, box: Box):
    box.floor_shadow(c)
    box.faces(c)
    box.top(c)
    box.seam(c)
    box.tape(c)
    box.label(c)
    box.edges(c)


def draw_crushed(c: Canvas, box: Box):
    """The front top right corner pushed in and down: the top sags, creases run over both faces, a flap tears."""
    box.floor_shadow(c)
    FTR = (box.FTR[0] - box.w * 0.22, box.FTR[1] + box.h * 0.30)
    box.faces(c, ftr=FTR)
    box.top(c, ftr=FTR)
    # the hollow the crush left: darker, with the paper buckled into ridges
    hollow = [FTR, (box.FTR[0] + box.dx * 0.25, box.FTR[1] - box.dy * 0.15), (box.FTR[0] + box.dx * 0.1, box.FTR[1] + box.h * 0.24),
              (box.FTR[0] - box.w * 0.05, box.FTR[1] + box.h * 0.45)]
    c.fill([hollow], "#7a5732", light=(0.9, 1.1), grain=8)
    ridges = [
        [FTR, (FTR[0] - 18, FTR[1] + 40)], [FTR, (FTR[0] - 34, FTR[1] + 18)], [FTR, (FTR[0] + 10, FTR[1] + 46)],
        [FTR, (FTR[0] - 30, FTR[1] - 8)], [(FTR[0] + 6, FTR[1] - 4), (box.BTR[0] - 6, box.BTR[1] + 22)],
        [FTR, (FTR[0] + 24, FTR[1] + 30)], [(FTR[0] - 12, FTR[1] + 22), (FTR[0] - 40, FTR[1] + 52)],
    ]
    for r in ridges:
        c.line(r, "#5e4325", 1.6, 0.85)
    for r in ([(FTR[0] - 6, FTR[1] + 6), (FTR[0] - 26, FTR[1] + 44)], [(FTR[0] + 4, FTR[1] + 8), (FTR[0] + 18, FTR[1] + 36)],
              [(FTR[0] - 8, FTR[1] - 2), (FTR[0] - 36, FTR[1] + 6)]):
        c.line(r, "#ecd0a2", 1.2, 0.7)
    # a torn flap corner sticking up
    c.fill([[(FTR[0] - 8, FTR[1] - 2), (FTR[0] + 14, FTR[1] - 26), (FTR[0] + 22, FTR[1] - 6)]], "#cfa772", grain=6)
    c.line([(FTR[0] - 8, FTR[1] - 2), (FTR[0] + 14, FTR[1] - 26), (FTR[0] + 22, FTR[1] - 6)], "#6e5030", 1.0, 0.8)
    box.seam(c)
    box.tape(c)
    box.label(c)
    box.edges(c, ftr=FTR)


def draw_collapsed(c: Canvas):
    """m03: a carton crushed down on its right half, the front and top buckled along a fold, the board split open along
    that fold so the inside (dark, with packing paper) shows. The first m03 drawing (draw_crushed: one corner pushed in)
    was read as not damaged; this one follows the round's rule for its one redraw: the shape collapses and the inside
    shows."""
    x0, yb, w, d = 96.0, 318.0, 168.0, 130.0
    dx, dy = 0.55 * d, 0.42 * d
    hl, hr = 124.0, 70.0                       # the left edge's height, the crushed right edge's
    x1 = x0 + w
    FTL, FBL, FBR = (x0, yb - hl), (x0, yb), (x1, yb)
    FTR = (x1, yb - hr)
    fold = [(x0 + 52, yb - hl + 8), (x0 + 96, yb - hl + 38), (x0 + 128, yb - hr - 14)]   # the buckled top edge
    BTL, BTR, BBR = (x0 + dx, yb - hl - dy), (x1 + dx, yb - hr - dy + 6), (x1 + dx, yb - dy)
    BF = [(p[0] + dx, p[1] - dy + 10) for p in fold]                                       # the fold across the top
    c.shadow([[(x0 - 8, yb + 4), (x1 + 6, yb + 6), (x1 + dx + 26, yb - dy + 10), (x0 + dx + 4, yb - dy - 2)]], 0.42, 7)
    # the front face under the buckled edge, darker in the fold's valley
    c.fill([[FTL] + fold + [FTR, FBR, FBL]], "#c39760", light=(1.02, 0.9), grain=7, fiber=5)
    c.fill([[fold[0], fold[1], (fold[1][0] + 6, yb - 30), (fold[0][0] + 10, yb - 52)]], "#a9814f", light=(0.95, 1.05),
           grain=7)
    c.fill([[fold[1], fold[2], (fold[2][0] - 4, yb - 40), (fold[1][0] + 6, yb - 30)]], "#b88c57", light=(0.95, 1.05),
           grain=7)
    # the side face, squashed
    c.fill([[FTR, BTR, BBR, FBR]], "#a47a46", light=(0.98, 0.84), grain=7, fiber=5)
    c.line([(FTR[0] + 14, FTR[1] + 6), (FTR[0] + 30, FTR[1] + 40), (FTR[0] + 24, FTR[1] + 74)], "#6e5030", 1.4, 0.8)
    # the top in two slabs either side of the fold
    c.fill([[FTL, fold[0], fold[1], BF[1], BF[0], BTL]], "#dbb683", light=(1.0, 1.04), axis="x", grain=6, fiber=4)
    c.fill([[fold[1], fold[2], FTR, BTR, BF[2], BF[1]]], "#c9a26f", light=(1.0, 0.92), axis="x", grain=6, fiber=4)
    # the split along the fold: the inside, dark, with packing paper in it
    split = [fold[0], (fold[1][0] - 2, fold[1][1] - 6), BF[1], (BF[2][0] - 10, BF[2][1] + 2), (BF[2][0] - 6, BF[2][1] + 16),
             (BF[1][0] + 4, BF[1][1] + 16), (fold[1][0] + 6, fold[1][1] + 10), (fold[0][0] + 8, fold[0][1] + 14)]
    c.fill([split], "#2f2113", light=(0.85, 1.1), grain=5)
    c.fill([[(BF[1][0] - 22, BF[1][1] + 4), (BF[1][0] - 4, BF[1][1] - 2), (BF[1][0] + 6, BF[1][1] + 10),
             (BF[1][0] - 14, BF[1][1] + 14)]], "#e4e1da", light=(1.02, 0.9), grain=6)
    for e in ([fold[0], (fold[1][0] - 2, fold[1][1] - 6), BF[1]], [(fold[0][0] + 8, fold[0][1] + 14),
                                                                    (fold[1][0] + 6, fold[1][1] + 10), (BF[1][0] + 4, BF[1][1] + 16)]):
        c.line(e, "#5e4325", 1.4, 0.9)
    # the tape, broken where the board split
    tw = w * 0.16
    for seg in ([(x0 + 18, FTL[1] + 2), (x0 + 18 + tw, FTL[1] + 6), (x0 + 18 + tw + dx * 0.55, FTL[1] - dy * 0.55 + 6),
                 (x0 + 18 + dx * 0.55, FTL[1] - dy * 0.55 + 2)],
                [(x0 + 18 + dx * 0.78, FTL[1] - dy * 0.78 + 4), (x0 + 18 + tw + dx * 0.78, FTL[1] - dy * 0.78 + 8),
                 (x0 + 18 + tw + dx, FTL[1] - dy + 8), (x0 + 18 + dx, FTL[1] - dy + 4)]):
        c.fill([seg], "#e9d7ae", light=(1.05, 0.97), axis="x", grain=3, alpha=0.9)
    # creases over the front
    for r in ([(fold[0][0] + 4, fold[0][1] + 6), (fold[0][0] + 14, yb - 48)], [(fold[1][0], fold[1][1] + 4), (fold[1][0] + 8, yb - 26)],
              [(fold[2][0] - 6, fold[2][1] + 6), (fold[2][0] - 2, yb - 36)], [(x0 + 30, yb - hl + 20), (x0 + 44, yb - 70)]):
        c.line(r, "#6e5030", 1.3, 0.8)
    for r in ([(fold[0][0] + 10, fold[0][1] + 10), (fold[0][0] + 18, yb - 58)], [(fold[1][0] + 6, fold[1][1] + 10), (fold[1][0] + 12, yb - 40)]):
        c.line(r, "#ecd0a2", 1.0, 0.6)
    # a blank label, skewed with the face
    lx, ly = x0 + 16, yb - 50
    lab = [(lx, ly), (lx + 52, ly + 4), (lx + 50, ly + 34), (lx - 2, ly + 30)]
    c.fill([lab], "#f4f2ec", light=(1.0, 0.96), grain=2)
    for k in range(3):
        c.line([(lx + 6, ly + 8 + 7 * k), (lx + 30 + 8 * (k % 2), ly + 9 + 7 * k)], "#a9a59c", 1.4, 0.8, blur=0.3)
    c.line([FBL, FTL] + fold + [FTR, FBR, FBL], "#7d5d36", 1.1, 0.8)
    c.line([FTR, BTR, BBR, FBR], "#7d5d36", 1.1, 0.8)
    c.line([FTL, BTL, BF[0]], "#7d5d36", 1.0, 0.7)
    c.line([BF[2], BTR], "#7d5d36", 1.0, 0.7)


def draw_opened(c: Canvas, box: Box):
    """The flaps up and out, the tape torn through, crumpled packing paper inside."""
    box.floor_shadow(c)
    # the back and the left / right flaps, behind the opening
    up = box.h * 0.42
    back = [box.BTL, box.BTR, (box.BTR[0] + 6, box.BTR[1] - up), (box.BTL[0] + 6, box.BTL[1] - up)]
    left = [box.FTL, box.BTL, (box.BTL[0] - 44, box.BTL[1] - up * 0.55), (box.FTL[0] - 44, box.FTL[1] - up * 0.55)]
    # the right flap lies out to the right, a little lifted: in this view it falls in front of the side face
    right = [box.FTR, box.BTR, (box.BTR[0] + 46, box.BTR[1] - 8), (box.FTR[0] + 46, box.FTR[1] - 8)]
    c.fill([back], "#cfae80", light=(1.06, 0.92), grain=6, fiber=4)
    c.fill([left], "#d8b888", light=(1.04, 0.92), axis="x", grain=6, fiber=4)
    for p in (back, left):
        c.line(p + [p[0]], "#7d5d36", 1.0, 0.75)
    box.faces(c)
    # the opening, dark, and the paper inside
    c.fill([[box.FTL, box.FTR, box.BTR, box.BTL]], "#3e2c19", light=(0.8, 1.15), grain=5)
    blobs = [[(box.FTL[0] + 30, box.FTL[1] - 6), (box.FTL[0] + 62, box.FTL[1] - 26), (box.FTL[0] + 92, box.FTL[1] - 14),
              (box.FTL[0] + 80, box.FTL[1] - 2)],
             [(box.FTL[0] + 96, box.FTL[1] - 20), (box.FTL[0] + 128, box.FTL[1] - 34), (box.FTL[0] + 150, box.FTL[1] - 18),
              (box.FTL[0] + 120, box.FTL[1] - 6)]]
    for b in blobs:
        c.fill([b], "#e4e1da", light=(1.02, 0.9), grain=6)
        c.line(b + [b[0]], "#a9a39a", 0.9, 0.7)
    c.shadow([[(x + 3, y + 6) for x, y in right]], 0.25, 3)
    c.fill([right], "#d5b585", light=(1.04, 0.94), axis="x", grain=6, fiber=4)
    c.line(right + [right[0]], "#7d5d36", 1.0, 0.75)
    # the front flap, folded down over the front face, with a torn strip of tape
    ff = [box.FTL, box.FTR, (box.FTR[0] - 6, box.FTR[1] + box.h * 0.36), (box.FTL[0] + 4, box.FTL[1] + box.h * 0.38)]
    c.shadow([[(x, y + 5) for x, y in ff]], 0.3, 4)
    c.fill([ff], "#d1a96f", light=(1.06, 0.95), grain=7, fiber=5)
    c.line(ff + [ff[0]], "#7d5d36", 1.1, 0.8)
    tw = box.w * 0.16
    mx = box.x0 + box.w / 2
    yt = box.FTL[1]
    rag = [(mx - tw / 2, yt + 2), (mx + tw / 2, yt + 2), (mx + tw / 2, yt + 16), (mx + tw / 4, yt + 11), (mx + 2, yt + 19),
           (mx - tw / 4, yt + 12), (mx - tw / 2, yt + 18)]
    c.fill([rag], "#e9d7ae", grain=3, alpha=0.92)
    rag2 = [(left[3][0] + 10, left[3][1] + 12), (left[3][0] + 34, left[3][1] + 16), (left[3][0] + 30, left[3][1] + 30),
            (left[3][0] + 22, left[3][1] + 24), (left[3][0] + 14, left[3][1] + 32)]
    c.fill([rag2], "#e9d7ae", grain=3, alpha=0.9)
    box.label(c)
    box.edges(c, top=False)


def draw_two(c: Canvas):
    big = Box(70, 316, 150, 112, 120)
    small = Box(214, 330, 104, 74, 84)
    draw_intact(c, big)
    draw_intact(c, small)


PICTURES = {
    "intact": lambda c: draw_intact(c, Box(96, 318, 168, 124, 130)),
    "crushed": lambda c: draw_crushed(c, Box(96, 318, 168, 124, 130)),   # the first m03, not shown
    "collapsed": draw_collapsed,
    "opened": lambda c: draw_opened(c, Box(96, 324, 168, 118, 130)),
    "two": draw_two,
}


def picture(name: str, seed: int) -> Image.Image:
    c = Canvas(seed)
    scene(c)
    PICTURES[name](c)
    return c.image()


# ------------------------------------------------------------------------------------------------------------- writing
def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixtures", required=True, help="the port's fixtures directory (records.json, images/)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fx, out = Path(a.fixtures), Path(a.out)
    (out / "pictures").mkdir(parents=True, exist_ok=True)
    (out / "requests").mkdir(parents=True, exist_ok=True)

    media = []
    seeds = {"intact": 11, "collapsed": 15, "opened": 13, "two": 14}   # crushed (seed 12): the first m03, not shown
    for name, seed in seeds.items():
        p = out / "pictures" / f"{name}.png"
        picture(name, seed).save(p, optimize=False)
        media.append({"file": f"pictures/{p.name}", "size": [SIZE, SIZE], "sha256": sha256(p),
                      "source": "drawn by apps/D1Demo/make_samples.py (PIL), seed " + str(seed),
                      "license": "CC0-1.0 (self-made)", "people": False, "text": False})

    items, requests = [], {}
    for mid, kind, text, pic, gold in INBOX:
        if kind == "text":
            req = {"state": text, "questions": TEXT_QUESTIONS}
        else:
            req = {"state": text, "questions": PICTURE_QUESTIONS, "images": [f"../pictures/{pic}.png"]}
        items.append({"id": mid, "kind": kind, "text": text, "picture": f"pictures/{pic}.png" if pic else None,
                      "questions": list(req["questions"]), "expected": gold})
        requests[mid] = req

    recs = {r["id"]: r for r in json.load(open(fx / "records.json"))["records"]}
    log = recs["long_34k"]
    lq = {k: log["request"]["questions"][k] for k in LOG_QUESTIONS}
    requests["log"] = {"state": log["request"]["state"], "questions": lq}
    log_item = {"id": "log", "title": "Order log", "entries": len(log["request"]["state"]),
                "questions": list(LOG_QUESTIONS), "expected": {k: log["gold"][k] for k in LOG_QUESTIONS},
                "source": "fixtures/records.json long_34k (" + log["note"] + ")",
                "left_out": {"shipped_unpaid": "the record's fourth question: the provider's fp32 p(yes) is 0.519, a "
                                               "near-tie whose argmax can differ between two GPUs"}}

    # warm-up: one text, one picture (the fixture's img01, CC0), the log with its fourth question; never shown
    requests["warm_text"] = {"state": WARMUP_TEXT, "questions": TEXT_QUESTIONS}
    img01 = fx / "images" / "img01_shapes_384x384.png"
    (out / "pictures" / "warm_img01.png").write_bytes(img01.read_bytes())
    requests["warm_picture"] = {"state": "Photo attached.", "questions": PICTURE_QUESTIONS,
                                "images": ["../pictures/warm_img01.png"]}
    requests["warm_log"] = {"state": log["request"]["state"],
                            "questions": {"shipped_unpaid": log["request"]["questions"]["shipped_unpaid"]}}
    media.append({"file": "pictures/warm_img01.png", "size": [384, 384], "sha256": sha256(out / "pictures/warm_img01.png"),
                  "source": "the d1-3B port's fixture img01_shapes_384x384.png (conversion/d1/make_fixture_images.py)",
                  "license": "CC0-1.0 (self-made)", "people": False, "text": False, "use": "warm-up only, never shown"})

    for rid, req in requests.items():
        (out / "requests" / f"{rid}.json").write_text(json.dumps(req, indent=2, ensure_ascii=False) + "\n")
    samples = {"schema": "d1demo-samples/1", "inbox": items, "log": log_item, "display": DISPLAY,
               "warmup": ["warm_text", "warm_picture", "warm_log"],
               "requests": {rid: f"requests/{rid}.json" for rid in requests}}
    (out / "samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False) + "\n")
    (out / "media_sources.json").write_text(json.dumps({"schema": "d1demo-media/1", "media": media}, indent=1) + "\n")
    print(f"{len(items)} inbox items ({sum(i['kind'] == 'picture' for i in items)} with a picture), the log "
          f"({log_item['entries']} entries, {len(lq)} questions), {len(requests)} requests -> {out}")
    for m in media:
        print(f"  {m['file']} {m['sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
