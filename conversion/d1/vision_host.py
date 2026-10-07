#!/usr/bin/env python3
"""d1 vision host reference: a picture -> crops -> patches, mask, position table, unshuffle index -> image token ids.

NumPy and Pillow only; torch, torchvision, transformers and the provider's code are not imported. This file is the
specification a Swift host copies: every rule below is the provider's image path (LiquidAI/d1-3B @ da1fe36a,
`runner.py` `cap_pixels` / `_image_markup` / `_image_inputs` / `_request`, `lfm2_vl.py` `answer`) and what it runs in
transformers 5.19.0 (`processing_lfm2_vl.py`, `image_processing_lfm2_vl.py`, `modeling_lfm2_vl.py`,
`modeling_siglip2.py`), torchvision 0.24.1 and torch 2.9.1 on CPU, written out. `test_vision_host.py` gates it
against those (ids, pixel_values bit for bit, spatial_shapes, mask, position table, unshuffle) before any Swift.

1. The provider's path for one request with pictures (`runner._request`)

     pics   = [cap_pixels(im) for im in images]
     prefix = BOS + "<|im_start|>user\\n" + "<image>" * len(pics) + state part      (`prompt.prefix_text`; the
              chat template writes one "<image>" per picture and nothing between them, `_image_markup`)
     one question:  processor(text=[prefix + suffix], images=[pics]) -> one plain pass over the whole row
     2+ questions:  processor(text=[prefix], images=[pics]) -> the trunk; branches = encode(suffix) as for text
     cap_pixels(im) = im.convert("RGB"); when w * h > 1024 * 1024: Pillow `resize((max(1, int(w * s)),
       max(1, int(h * s))), BICUBIC)`, s = sqrt(1024 * 1024 / (w * h)); otherwise the picture as it is.
     The picture arrives decoded (the card loads it with `transformers.image_utils.load_image`, which applies the
     EXIF orientation and converts to RGB): a host decodes with the orientation applied.
     After the processor the provider cuts `pixel_values` / `pixel_attention_mask` to n = the largest real patch
     count over the request's crops (numerically the same: the cut rows are padding).

2. Crop plan (`Lfm2VlImageProcessor.resize_and_split`; P = 16, f = 2, T = 512, F = P * f = 32)

     round_f(v) = round(v / F) * F (Python's round: half to even)
     too_large  = max(P, round_f(h)) * max(P, round_f(w)) > 256 * P^2 * f^2 * 2.0 = 524,288   (`_is_image_too_large`)
     smart size (`smart_resize`, min 64 / max 256 image tokens):
       h_bar = max(F, round_f(h)), w_bar = max(F, round_f(w))
       h_bar * w_bar > 262,144: b = sqrt(h * w / 262,144); h_bar = max(F, floor(h / b / F) * F), same for w
       h_bar * w_bar <  65,536: b = sqrt(65,536 / (h * w)); h_bar = ceil(h * b / F) * F,          same for w
     not too_large -> one crop: the picture resized to (h_bar, w_bar).
     too_large     -> tiles: (cols, rows) = the closest aspect ratio cols/rows to w/h among the (c, r) with
       2 <= c * r <= 10, scanned in `_target_ratios` order (TARGET_RATIOS below); on an equal |difference| the later
       ratio wins when w * h > 0.5 * 512^2 * c * r (`find_closest_aspect_ratio`). The picture is resized to
       (512 * rows, 512 * cols) and cut into rows x cols tiles of 512 x 512, row-major; then a thumbnail = the
       ORIGINAL picture resized to (h_bar, w_bar). Crops in order: tile (1, 1), (1, 2), .., (rows, cols), thumbnail.
     A crop of H x W pixels has spatial shape (H / 16, W / 16) patches and H * W / 1024 image tokens.
     Every crop is <= 1024 patches up to an aspect of 256:1: past 64:1 the smart size's short side clamps at 32 px
     and its long side is about 512 * sqrt(aspect), over 8192 px (1024 patches at 2 rows) past 256:1. Up to 4:1 a
     picture has at most 11 crops (10 tiles + thumbnail) and 2,810 image tokens (`vision_grid_table.json`); what
     the processor does with a crop over 1024 patches is not covered here (`crop_pixels` refuses it).

3. Resize (`TorchvisionBackend.resize` -> `tvF.resize(uint8 CHW, BICUBIC, antialias=True)`)

     The processor's image class IS the torchvision backend (transformers 5 has no Pillow class for LFM2-VL), and
     torchvision resizes a uint8 image natively on CPU for BICUBIC (`_do_native_uint8_resize_on_cpu`), i.e. torch's
     `_upsample_bicubic2d_aa` uint8 kernel (`UpSampleKernel.cpp`, the separable generic path on aarch64 for a
     channels-first image; `torch_bicubic` is bit-exact to torch 2.9.1 on every crop the test runs). Per axis
     (in -> out, scale = in / out in double):
       support = 2 * scale if scale >= 1 else 2;  kmax = ceil(support) * 2 + 1;  inv = 1 / scale if scale >= 1 else 1
       for out index i: center = scale * (i + 0.5); xmin = max(int(center - support + 0.5), 0);
         n = clamp(min(int(center + support + 0.5), in) - xmin, 0, kmax);
         w_j = cubic((j + xmin - center + 0.5) * inv), j < n, then w_j /= sum(w)   (int() truncates toward zero)
         cubic(x) = ((A + 2)|x| - (A + 3))|x|^2 + 1 for |x| < 1, ((A|x| - 5A)|x| + 8A)|x| - 4A for |x| < 2, A = -0.5
       precision p = the largest p <= 22 with int(0.5 + max_w * 2^p) < 2^15 (max_w over the whole axis), so p is
         per axis and per (in, out) -- NOT Pillow's fixed 22 bits; w16 = round half away from zero of w * 2^p
       out = clamp((2^(p - 1) + sum_j src[xmin + j] * w16_j) >> p, 0, 255)   (integer sums, arithmetic shift)
     Width first (the result stored as uint8), then height; an axis whose size does not change is skipped, and
     a picture already at the target size is not touched. `cap_pixels` (Pillow) is the same filter with Pillow's
     weights: a = -0.5 written as (((|x| - 5)|x| + 8)|x| - 4) * a, the 22-bit fixed point of `normalize_coeffs_8bpc`
     and `clip8` (`pillow_bicubic`). The two differ by 1 level on some pixels; the processor's is the one that
     reaches the tower.

4. Pixels (`rescale_and_normalize` fused, `convert_image_to_patches`, `pad_along_first_dim`)

     value = (x - 127.5) / 127.5 in float32 (mean 0.5 * 255, std 0.5 * 255; one of 256 values per byte)
     patches [H/16 * W/16, 768]: patch (py, px) row-major, inside it [y][x][c] (channel fastest)
     pixel_values [n_crop, 1024, 768] float32 (rows past the crop's patches are 0.0),
     pixel_attention_mask [n_crop, 1024] int32 (1 for a real patch), spatial_shapes [n_crop, 2] int64 (h, w)

5. Image token ids (`Lfm2VlProcessor._build_image_tokens`, `use_image_special_tokens` true)

     one crop: <|image_start|> + <image> * tokens + <|image_end|>
     tiles:    <|image_start|> + for r, c: <|img_row_r_col_c|> + <image> * 256, then <|img_thumbnail|>
               + <image> * thumbnail tokens + <|image_end|>                                (no spaces or newlines)
     ids: <image> 124907, <|img_row_r_col_c|> 124908 + 10 (r - 1) + (c - 1) (r, c in 1..10),
          <|img_thumbnail|> 125008, <|image_start|> 125009, <|image_end|> 125010
     The prompt's "<image>" markup is replaced by that string and the whole text is tokenized with
     add_special_tokens=False; the special tokens split the text, so the ids are encode(text before) + image ids +
     encode(text after) (`prompt_ids`).

6. Decoder slots (`get_placeholder_mask`, `masked_scatter`; round 1 `lfm2_d1_decoder.py`)

     Only <image> (124907) is a slot; start / end / row-col / thumbnail tokens are ordinary embedding rows.
     The k-th <image> of the row (k counted over every picture and crop in text order) is replaced by the
     extension id V + k (V = 128,000) and reads row k of image_embeds = the crops' tower outputs concatenated in
     crop order (each crop's tokens row-major over its merged grid). n_image_tokens = the count of <image>.

7. Position table (`Siglip2VisionEmbeddings.resize_positional_embeddings`)

     The [256, 1152] table as [16, 16, 1152] is resized to the crop's (h, w) patches by
     F.interpolate(mode="bilinear", align_corners=False, antialias=True) in float32 (CPU upcast), then laid out
     row-major [h * w, 1152]. torch's float kernel: per axis scale = float32(16) / out, support = scale if
     scale >= 1 else 1, center = float32(scale * (i + 0.5)), the triangle 1 - |x| at
     float32((float32(j + xmin) - center + 0.5) * inv), weights normalized by their float32 sum; width first, then
     height, accumulated in tap order as acc = t0 * w0, acc = fma(t_j, w_j, acc) (torch's aarch64 build fuses
     `output += t * wts`; a separate multiply and add lands within 4.8e-7 of it). HF fills the rows past h * w
     with row 0; this host fills them with 0 (the tower masks those patches out as keys and drops their outputs,
     so either is exact for the real rows).

8. Unshuffle and the tower's inputs (`Lfm2VlMultiModalProjector.pixel_unshuffle`)

     merged token k = i * (w / 2) + j (row-major over the (h / 2, w / 2) merged grid) concatenates the hidden rows
     of patches [2i * w + 2j, 2i * w + 2j + 1, (2i + 1) * w + 2j, (2i + 1) * w + 2j + 1] in that order: channel
     m * 1152 + c of the 4608-vector comes from offset (dy, dx) = (m // 2, m % 2).
     tower_inputs(crop) = patches [1024, 768] + pos_table [1024, 1152] (rows past the crop's patches 0) +
       key_bias [1024] float32 (0 for a real patch, -inf for padding, added to every query's attention scores) +
       unshuffle_idx [256, 4] int32 (rows k >= tokens point at patch 0 and their outputs are dropped).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# processor_config.json "image_processor" (LiquidAI/d1-3B @ da1fe36a)
PATCH = 16                       # encoder_patch_size
FACTOR = 2                       # downsample_factor
TILE = 512                       # tile_size
MIN_TILES, MAX_TILES = 2, 10
MIN_IMAGE_TOKENS, MAX_IMAGE_TOKENS = 64, 256
MAX_PIXELS_TOLERANCE = 2.0
USE_THUMBNAIL = True
MAX_NUM_PATCHES = max(MAX_IMAGE_TOKENS * FACTOR ** 2, (TILE // PATCH) ** 2)      # 1024
MEAN_255 = np.float32(0.5 * 255.0)                                                  # 127.5
STD_255 = np.float32(0.5 * 255.0)
PATCH_DIM = PATCH * PATCH * 3                                                       # 768

# config.json "vision_config" / the decoder contract
POS_SIDE = 16                    # sqrt(num_patches 256)
HIDDEN = 1152
TOKENS_PER_TILE = ((TILE // PATCH) // FACTOR) ** 2                                  # 256

VISION_MAX_PIXELS = 1024 * 1024  # runner.VISION_MAX_PIXELS

IMAGE_TOKEN = "<image>"
IMAGE_ID = 124907
ROW_COL_BASE = 124908            # <|img_row_1_col_1|>
THUMBNAIL_ID = 125008
IMAGE_START_ID, IMAGE_END_ID = 125009, 125010
V = 128000                       # extension id base (config text vocab_size; round 1 lfm2_d1_decoder.py)


def row_col_id(r: int, c: int) -> int:
    """<|img_row_r_col_c|>, 1-based."""
    if not (1 <= r <= 10 and 1 <= c <= 10):
        raise ValueError(f"row {r} col {c} outside the tokenizer's 10 x 10")
    return ROW_COL_BASE + 10 * (r - 1) + (c - 1)


# --------------------------------------------------------------------------- #
# 3. Resamplers (integer, bit-exact)
# --------------------------------------------------------------------------- #
def _apply_taps(src: np.ndarray, axis: int, starts: np.ndarray, w: np.ndarray, precision: int) -> np.ndarray:
    """out[i] = clamp((2^(p-1) + sum_j src[starts[i] + j] * w[i, j]) >> p, 0, 255) along `axis` of a uint8 [H, W, C]."""
    n_out, k = w.shape
    in_size = src.shape[axis]
    idx = np.minimum(starts[:, None] + np.arange(k)[None, :], in_size - 1)   # taps past the window carry weight 0
    s = src.astype(np.int64)
    acc = np.full((n_out,) + tuple(np.delete(np.array(src.shape), axis)), 1 << (precision - 1), dtype=np.int64)
    for j in range(k):
        taps = np.take(s, idx[:, j], axis=axis)
        wj = w[:, j].astype(np.int64)
        if axis == 1:
            acc += np.moveaxis(taps, 1, 0) * wj[:, None, None]
        else:
            acc += taps * wj[:, None, None]
    out = np.clip(acc >> precision, 0, 255).astype(np.uint8)
    return np.moveaxis(out, 0, 1) if axis == 1 else out


def _torch_cubic(x: float) -> float:
    """HelperInterpCubic::aa_filter<double, true> (A = -0.5), with cubic_convolution1 / 2 of UpSample.h."""
    a = -0.5
    x = abs(x)
    if x < 1.0:
        return ((a + 2) * x - (a + 3)) * x * x + 1
    if x < 2.0:
        return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
    return 0.0


def torch_u8_weights(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    """torch's uint8 antialiased bicubic weights for one axis: (starts [out], int16 weights [out, kmax], precision).

    `HelperInterpBase::_compute_index_ranges_weights<double>` + `_compute_indices_min_size_weights_aa` +
    `_compute_index_ranges_int16_weights` (align_corners False, no scale factor)."""
    scale = in_size / out_size
    support = (4 * 0.5) * scale if scale >= 1.0 else 4 * 0.5
    kmax = int(math.ceil(support)) * 2 + 1
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    starts = np.zeros(out_size, dtype=np.int64)
    wf = np.zeros((out_size, kmax), dtype=np.float64)
    wt_max = 0.0
    for i in range(out_size):
        center = scale * (i + 0.5)
        xmin = max(int(center - support + 0.5), 0)
        xsize = min(int(center + support + 0.5), in_size) - xmin
        xsize = min(max(xsize, 0), kmax)
        ws = [_torch_cubic((j + xmin - center + 0.5) * invscale) for j in range(xsize)]
        total = 0.0
        for v in ws:
            total += v
        if total != 0.0:
            ws = [v / total for v in ws]
            for v in ws:
                wt_max = max(wt_max, v)
        starts[i] = xmin
        wf[i, :xsize] = ws
    precision = 0
    while precision < 22:
        if int(0.5 + wt_max * (1 << (precision + 1))) >= (1 << 15):
            break
        precision += 1
    v = wf * (1 << precision)
    w16 = np.where(v < 0, np.trunc(-0.5 + v), np.trunc(0.5 + v)).astype(np.int64)
    return starts, w16, precision


def torch_bicubic(u8: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """`tvF.resize(uint8, [out_h, out_w], BICUBIC, antialias=True)` of an [H, W, 3] uint8 image (section 3)."""
    x = np.ascontiguousarray(u8)
    h, w = x.shape[:2]
    if (h, w) == (out_h, out_w):
        return x
    if w != out_w:
        x = _apply_taps(x, 1, *torch_u8_weights(w, out_w))
    if h != out_h:
        x = _apply_taps(x, 0, *torch_u8_weights(h, out_h))
    return x


PIL_PRECISION_BITS = 32 - 8 - 2


def _pil_cubic(x: float) -> float:
    """Resample.c `bicubic_filter` (a = -0.5), in its operation order."""
    a = -0.5
    if x < 0.0:
        x = -x
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1
    if x < 2.0:
        return (((x - 5) * x + 8) * x - 4) * a
    return 0.0


def pil_weights(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Resample.c `precompute_coeffs` (in0 = 0, in1 = in_size) + `normalize_coeffs_8bpc`: (starts, int weights, 22)."""
    scale = float(in_size) / out_size
    filterscale = max(scale, 1.0)
    support = 2.0 * filterscale
    ksize = int(math.ceil(support)) * 2 + 1
    starts = np.zeros(out_size, dtype=np.int64)
    kk = np.zeros((out_size, ksize), dtype=np.int64)
    one = float(1 << PIL_PRECISION_BITS)
    for xx in range(out_size):
        center = 0.0 + (xx + 0.5) * scale
        ww = 0.0
        ss = 1.0 / filterscale
        xmin = int(center - support + 0.5)
        if xmin < 0:
            xmin = 0
        xmax = int(center + support + 0.5)
        if xmax > in_size:
            xmax = in_size
        xmax -= xmin
        if xmax > ksize:
            raise AssertionError(f"window {xmax} over ksize {ksize}")
        k = [0.0] * ksize
        for x in range(xmax):
            w = _pil_cubic((x + xmin - center + 0.5) * ss)
            k[x] = w
            ww += w
        for x in range(xmax):
            if ww != 0.0:
                k[x] /= ww
        kk[xx] = [int(-0.5 + v * one) if v < 0 else int(0.5 + v * one) for v in k]
        starts[xx] = xmin
    return starts, kk, PIL_PRECISION_BITS


def pillow_bicubic(u8: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Pillow 12's `Image.resize((out_w, out_h), BICUBIC)` of an RGB [H, W, 3] uint8 image: horizontal pass first
    (when the width changes), then vertical, each on uint8 (`ImagingResampleHorizontal_8bpc` / `Vertical`)."""
    x = np.ascontiguousarray(u8)
    h, w = x.shape[:2]
    if w != out_w:
        x = _apply_taps(x, 1, *pil_weights(w, out_w))
    if h != out_h:
        x = _apply_taps(x, 0, *pil_weights(h, out_h))
    return x


# --------------------------------------------------------------------------- #
# 1. The provider's cap
# --------------------------------------------------------------------------- #
def cap_size(w: int, h: int, max_pixels: int = VISION_MAX_PIXELS) -> tuple[int, int]:
    """runner.cap_pixels' target (w, h)."""
    if w * h <= max_pixels:
        return w, h
    scale = math.sqrt(max_pixels / (w * h))
    return max(1, int(w * scale)), max(1, int(h * scale))


def to_rgb(image) -> np.ndarray:
    """A path, a PIL image or an [H, W, 3] uint8 array -> [H, W, 3] uint8 (PIL `convert("RGB")`)."""
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected uint8 [H, W, 3], got {image.dtype} {image.shape}")
        return image
    from PIL import Image

    im = Image.open(image) if isinstance(image, (str, Path)) else image
    return np.asarray(im.convert("RGB"), dtype=np.uint8)


def cap_pixels(rgb: np.ndarray, max_pixels: int = VISION_MAX_PIXELS) -> np.ndarray:
    """runner.cap_pixels on an RGB array: at most `max_pixels`, Pillow BICUBIC (section 1)."""
    h, w = rgb.shape[:2]
    if w * h <= max_pixels:
        return rgb
    nw, nh = cap_size(w, h, max_pixels)
    return pillow_bicubic(rgb, nh, nw)


# --------------------------------------------------------------------------- #
# 2. Crop plan
# --------------------------------------------------------------------------- #
def round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def is_too_large(h: int, w: int) -> bool:
    """Lfm2VlImageProcessor._is_image_too_large."""
    total = PATCH * FACTOR
    h_bar = max(PATCH, round_by_factor(h, total))
    w_bar = max(PATCH, round_by_factor(w, total))
    return h_bar * w_bar > MAX_IMAGE_TOKENS * PATCH ** 2 * FACTOR ** 2 * MAX_PIXELS_TOLERANCE


def smart_size(h: int, w: int) -> tuple[int, int]:
    """Lfm2VlImageProcessor.smart_resize -> (h_bar, w_bar) (the processor returns (w_bar, h_bar))."""
    total = PATCH * FACTOR
    min_px = MIN_IMAGE_TOKENS * PATCH ** 2 * FACTOR ** 2
    max_px = MAX_IMAGE_TOKENS * PATCH ** 2 * FACTOR ** 2
    h_bar = max(total, round_by_factor(h, total))
    w_bar = max(total, round_by_factor(w, total))
    if h_bar * w_bar > max_px:
        beta = math.sqrt((h * w) / max_px)
        h_bar = max(total, math.floor(h / beta / total) * total)
        w_bar = max(total, math.floor(w / beta / total) * total)
    elif h_bar * w_bar < min_px:
        beta = math.sqrt(min_px / (h * w))
        h_bar = math.ceil(h * beta / total) * total
        w_bar = math.ceil(w * beta / total) * total
    return h_bar, w_bar


def _target_ratios(min_tiles: int = MIN_TILES, max_tiles: int = MAX_TILES) -> list[tuple[int, int]]:
    """Lfm2VlImageProcessor._target_ratios: (cols, rows), sorted by cols * rows from a Python set (CPython's set
    order of small int tuples is fixed; TARGET_RATIOS is that order, which a Swift host copies as a table)."""
    ratios = [(w, h) for n in range(min_tiles, max_tiles + 1) for w in range(1, n + 1) for h in range(1, n + 1)
              if min_tiles <= w * h <= max_tiles]
    return sorted(set(ratios), key=lambda x: x[0] * x[1])


TARGET_RATIOS = _target_ratios()


def grid_layout(h: int, w: int) -> tuple[int, int]:
    """find_closest_aspect_ratio over TARGET_RATIOS -> (rows, cols)."""
    aspect = w / h
    best_diff, best = float("inf"), (1, 1)
    area = w * h
    for ratio in TARGET_RATIOS:
        diff = abs(aspect - ratio[0] / ratio[1])
        if diff < best_diff:
            best_diff, best = diff, ratio
        elif diff == best_diff:
            if area > 0.5 * (TILE * TILE * ratio[0] * ratio[1]):
                best = ratio
    cols, rows = best
    return rows, cols


@dataclass
class Crop:
    kind: str                     # "single" | "tile" | "thumbnail"
    resize_to: tuple[int, int]    # (H, W) of the resized picture the crop is cut from
    box: tuple[int, int, int, int]  # (y0, x0, y1, x1) in that resized picture
    row: int = 0                  # 1-based tile row / col (0 for single / thumbnail)
    col: int = 0

    @property
    def size(self) -> tuple[int, int]:
        return self.box[2] - self.box[0], self.box[3] - self.box[1]

    @property
    def grid(self) -> tuple[int, int]:
        h, w = self.size
        return h // PATCH, w // PATCH

    @property
    def n_patches(self) -> int:
        gh, gw = self.grid
        return gh * gw

    @property
    def n_tokens(self) -> int:
        gh, gw = self.grid
        return math.ceil(gh / FACTOR) * math.ceil(gw / FACTOR)


@dataclass
class Plan:
    size: tuple[int, int]         # (h, w) of the picture the processor received (after cap_pixels)
    rows: int
    cols: int
    thumb: tuple[int, int]        # smart size (h_bar, w_bar): the single crop's or the thumbnail's
    crops: list[Crop] = field(default_factory=list)

    @property
    def tiled(self) -> bool:
        return self.rows > 1 or self.cols > 1


def plan(h: int, w: int) -> Plan:
    """The processor's crops for an (h, w) picture (section 2)."""
    h_bar, w_bar = smart_size(h, w)
    if not is_too_large(h, w):
        return Plan((h, w), 1, 1, (h_bar, w_bar), [Crop("single", (h_bar, w_bar), (0, 0, h_bar, w_bar))])
    rows, cols = grid_layout(h, w)
    full = (TILE * rows, TILE * cols)
    crops = [Crop("tile", full, (r * TILE, c * TILE, (r + 1) * TILE, (c + 1) * TILE), r + 1, c + 1)
             for r in range(rows) for c in range(cols)]
    if USE_THUMBNAIL and rows * cols != 1:
        crops.append(Crop("thumbnail", (h_bar, w_bar), (0, 0, h_bar, w_bar)))
    return Plan((h, w), rows, cols, (h_bar, w_bar), crops)


# --------------------------------------------------------------------------- #
# 4. Pixels
# --------------------------------------------------------------------------- #
def normalize(u8: np.ndarray) -> np.ndarray:
    """(x - 127.5) / 127.5 in float32."""
    return (u8.astype(np.float32) - MEAN_255) / STD_255


def patchify(x: np.ndarray) -> np.ndarray:
    """[H, W, C] -> [H/16 * W/16, 16 * 16 * C]: patches row-major, [y][x][c] inside (channel fastest)."""
    h, w, c = x.shape
    if h % PATCH or w % PATCH:
        raise ValueError(f"{h}x{w} not divisible by {PATCH}")
    gh, gw = h // PATCH, w // PATCH
    return x.reshape(gh, PATCH, gw, PATCH, c).transpose(0, 2, 1, 3, 4).reshape(gh * gw, PATCH * PATCH * c)


def crop_images(rgb: np.ndarray, p: Plan) -> list[np.ndarray]:
    """Every crop's uint8 [H, W, 3] pixels, in crop order (section 3: one resize per distinct target)."""
    resized: dict[tuple[int, int], np.ndarray] = {}
    out = []
    for c in p.crops:
        if c.resize_to not in resized:
            resized[c.resize_to] = torch_bicubic(rgb, *c.resize_to)
        y0, x0, y1, x1 = c.box
        out.append(resized[c.resize_to][y0:y1, x0:x1])
    return out


def crop_pixels(crop_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One crop -> (pixel_values [1024, 768] float32 zero-padded, pixel_attention_mask [1024] int32)."""
    patches = patchify(normalize(crop_u8))
    n = patches.shape[0]
    if n > MAX_NUM_PATCHES:
        raise ValueError(f"crop of {n} patches over {MAX_NUM_PATCHES} (the processor would not pad it)")
    pv = np.zeros((MAX_NUM_PATCHES, PATCH_DIM), dtype=np.float32)
    pv[:n] = patches
    mask = np.zeros(MAX_NUM_PATCHES, dtype=np.int32)
    mask[:n] = 1
    return pv, mask


def image_inputs(image) -> dict:
    """One picture through cap_pixels and the processor: plan, crops, pixel_values, mask, spatial_shapes."""
    rgb = cap_pixels(to_rgb(image))
    p = plan(*rgb.shape[:2])
    crops = crop_images(rgb, p)
    pv, masks = zip(*(crop_pixels(c) for c in crops))
    return {"rgb": rgb, "plan": p, "crops": crops, "pixel_values": np.stack(pv), "pixel_attention_mask": np.stack(masks),
            "spatial_shapes": np.array([c.grid for c in p.crops], dtype=np.int64)}


# --------------------------------------------------------------------------- #
# 5. / 6. Token ids and slots
# --------------------------------------------------------------------------- #
def image_tokens(p: Plan) -> list[int]:
    """The id run one picture expands to (section 5)."""
    ids = [IMAGE_START_ID]
    if p.tiled:
        for c in p.crops:
            if c.kind == "tile":
                ids += [row_col_id(c.row, c.col)] + [IMAGE_ID] * c.n_tokens
            else:
                ids += [THUMBNAIL_ID] + [IMAGE_ID] * c.n_tokens
    else:
        ids += [IMAGE_ID] * p.crops[0].n_tokens
    return ids + [IMAGE_END_ID]


def image_token_text(p: Plan) -> str:
    """The same run as the string the processor puts in place of "<image>"."""
    parts = ["<|image_start|>"]
    if p.tiled:
        for c in p.crops:
            if c.kind == "tile":
                parts.append(f"<|img_row_{c.row}_col_{c.col}|>" + IMAGE_TOKEN * c.n_tokens)
            else:
                parts.append("<|img_thumbnail|>" + IMAGE_TOKEN * c.n_tokens)
    else:
        parts.append(IMAGE_TOKEN * p.crops[0].n_tokens)
    return "".join(parts) + "<|image_end|>"


def n_image_tokens(plans: list[Plan]) -> int:
    return sum(c.n_tokens for p in plans for c in p.crops)


def image_prefix_text(state: Any, n_images: int) -> str:
    """prompt.prefix_text(tok, state, BOS, "json_only", "none", images="<image>" * n_images) (section 1)."""
    from host import BOS, IM_START, state_block

    body = "" if state is None else f"{state_block(state)}\nQUESTION:\n"
    return f"{BOS}{IM_START}user\n{IMAGE_TOKEN * n_images}{body}"


def prompt_ids(tok, text: str, plans: list[Plan]) -> list[int]:
    """A prompt holding len(plans) "<image>" markers -> ids: the text pieces encoded apart, each marker -> its run."""
    from host import token_ids

    pieces = text.split(IMAGE_TOKEN)
    if len(pieces) - 1 != len(plans):
        raise ValueError(f"{len(pieces) - 1} <image> markers for {len(plans)} pictures")
    ids: list[int] = []
    for k, piece in enumerate(pieces):
        if piece:
            ids += token_ids(tok, piece)
        if k < len(plans):
            ids += image_tokens(plans[k])
    return ids


def expanded_text(text: str, plans: list[Plan]) -> str:
    """The processor's text after the marker replacement (for a whole-string encode)."""
    pieces = text.split(IMAGE_TOKEN)
    if len(pieces) - 1 != len(plans):
        raise ValueError(f"{len(pieces) - 1} <image> markers for {len(plans)} pictures")
    return "".join(piece + (image_token_text(plans[k]) if k < len(plans) else "") for k, piece in enumerate(pieces))


def extension_ids(ids: list[int], start: int = 0) -> list[int]:
    """<image> -> V + k (k = start, start + 1, ..) in order; every other id unchanged (section 6)."""
    out, k = [], start
    for t in ids:
        if t == IMAGE_ID:
            out.append(V + k)
            k += 1
        else:
            out.append(t)
    return out


def slot_map(plans: list[Plan]) -> list[tuple[int, int, int]]:
    """For each slot k: (picture index, crop index, merged token index inside the crop)."""
    out = []
    for pi, p in enumerate(plans):
        for ci, c in enumerate(p.crops):
            out += [(pi, ci, t) for t in range(c.n_tokens)]
    return out


# --------------------------------------------------------------------------- #
# 7. Position table
# --------------------------------------------------------------------------- #
def _f32(v) -> np.float32:
    return np.float32(v)


def torch_linear_aa_weights_f32(in_size: int, out_size: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """torch's float32 antialiased bilinear weights for one axis (`HelperInterpLinear`, scalar_t = float):
    (starts [out], [weights per out index] float32)."""
    scale = _f32(in_size) / _f32(out_size)
    support = _f32((2 * 0.5) * float(scale)) if scale >= 1.0 else _f32(2 * 0.5)
    kmax = int(math.ceil(float(support))) * 2 + 1
    invscale = _f32(1.0 / float(scale)) if scale >= 1.0 else _f32(1.0)
    starts = np.zeros(out_size, dtype=np.int64)
    weights = []
    for i in range(out_size):
        center = _f32(float(scale) * (i + 0.5))
        xmin = max(int(float(_f32(center - support)) + 0.5), 0)
        xsize = min(int(float(_f32(center + support)) + 0.5), in_size) - xmin
        xsize = min(max(xsize, 0), kmax)
        ws, total = [], _f32(0.0)
        for j in range(xsize):
            arg = _f32((float(_f32(_f32(j + xmin) - center)) + 0.5) * float(invscale))
            x = abs(arg)
            w = _f32(1.0 - float(x)) if x < 1.0 else _f32(0.0)
            ws.append(w)
            total = _f32(total + w)
        if total != 0.0:
            ws = [_f32(w / total) for w in ws]
        starts[i] = xmin
        weights.append(np.array(ws, dtype=np.float32))
    return starts, weights


def _linear_pass(x: np.ndarray, axis: int, out_size: int, fma: bool) -> np.ndarray:
    """One float32 pass along `axis` of [H, W, C]: out = t0 * w0, then out = fma(t_j, w_j, out) in tap order
    (`fma=False`: a rounded product, then a rounded add)."""
    starts, weights = torch_linear_aa_weights_f32(x.shape[axis], out_size)
    shape = list(x.shape)
    shape[axis] = out_size
    out = np.empty(shape, dtype=np.float32)
    for i in range(out_size):
        ws = weights[i]
        acc = None
        for j, w in enumerate(ws):
            t = np.take(x, starts[i] + j, axis=axis)
            if acc is None:
                acc = (t * w).astype(np.float32)
            elif fma:
                acc = (acc.astype(np.float64) + t.astype(np.float64) * np.float64(w)).astype(np.float32)
            else:
                acc = (acc + (t * w).astype(np.float32)).astype(np.float32)
        if axis == 0:
            out[i] = acc
        else:
            out[:, i] = acc
    return out


def pos_table(table: np.ndarray, h: int, w: int, fma: bool = True) -> np.ndarray:
    """[256, 1152] (or [16, 16, 1152]) -> [h * w, 1152] float32 (section 7): width pass, then height pass."""
    t = np.asarray(table, dtype=np.float32).reshape(POS_SIDE, POS_SIDE, -1)
    if w != POS_SIDE:
        t = _linear_pass(t, 1, w, fma)
    if h != POS_SIDE:
        t = _linear_pass(t, 0, h, fma)
    return t.reshape(h * w, -1)


# --------------------------------------------------------------------------- #
# 8. Unshuffle and the tower's inputs
# --------------------------------------------------------------------------- #
def unshuffle_index(h: int, w: int) -> np.ndarray:
    """[h/2 * w/2, 4] int32: the 4 patch rows merged token k concatenates, in channel order (section 8)."""
    if h % FACTOR or w % FACTOR:
        raise ValueError(f"grid {h}x{w} not divisible by {FACTOR}")
    i, j = np.meshgrid(np.arange(h // FACTOR), np.arange(w // FACTOR), indexing="ij")
    i, j = i.reshape(-1), j.reshape(-1)
    return np.stack([(2 * i) * w + 2 * j, (2 * i) * w + 2 * j + 1,
                     (2 * i + 1) * w + 2 * j, (2 * i + 1) * w + 2 * j + 1], axis=1).astype(np.int32)


def unshuffle(x: np.ndarray, h: int, w: int) -> np.ndarray:
    """[h * w, d] hidden rows -> [h/2 * w/2, 4d] by the index (HF's pixel_unshuffle on [1, h, w, d])."""
    idx = unshuffle_index(h, w)
    return np.concatenate([x[idx[:, m]] for m in range(4)], axis=1)


def tower_inputs(crop_u8: np.ndarray, table: np.ndarray | None = None) -> dict:
    """One crop -> the fixed-shape inputs of a grid-independent tower graph (section 8)."""
    pv, mask = crop_pixels(crop_u8)
    gh, gw = crop_u8.shape[0] // PATCH, crop_u8.shape[1] // PATCH
    n, n_tok = gh * gw, (gh // FACTOR) * (gw // FACTOR)
    out = {"patches": pv, "mask": mask, "grid": (gh, gw), "n_patches": n, "n_tokens": n_tok}
    out["key_bias"] = np.where(mask == 1, np.float32(0.0), np.float32(-np.inf)).astype(np.float32)
    idx = np.zeros((MAX_IMAGE_TOKENS, 4), dtype=np.int32)
    idx[:n_tok] = unshuffle_index(gh, gw)
    out["unshuffle_idx"] = idx
    if table is not None:
        pt = np.zeros((MAX_NUM_PATCHES, table.shape[-1]), dtype=np.float32)
        pt[:n] = pos_table(table, gh, gw)
        out["pos_table"] = pt
    return out
