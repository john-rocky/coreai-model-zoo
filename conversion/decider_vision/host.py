#!/usr/bin/env python3
"""decider-2b-vision host reference: image -> tower patches, (context, questions) -> decoder ids.

NumPy + Pillow only; the tokenizer is passed in (a `tokenizers.Tokenizer` or a transformers
tokenizer built from the checkpoint's `tokenizer.json`). This is the spec a Swift host
reproduces, gated before any Swift exists:

* `preprocess` against the processor's `pixel_values` for every fixture image at both grids
  (`gate_tower.py` G1a, max|d| == 0);
* `build_ids` against the round-1 oracle ids, slots and captured M-RoPE planes of all 111 runs
  (`test_host.py`).

Image path. The tower bakes ONE merged grid (g256: 8x8 = 64 tokens, g448: 14x14 = 196), so the
host's whole job is

    RGB -> PIL BICUBIC resize to (32*grid)^2 -> /255 -> (x-0.5)/0.5 -> merge-block-major patchify
        -> patches [4*grid^2, 1536] float32, vector layout (C, T=2, 16, 16), the same frame twice.

The aspect ratio is NOT preserved (a 256x240 game frame is stretched to 256x256): the oracle's
g256 / g448 arms are captured through the same resize, so the gates measure the shipped path.
The resize must be Pillow's antialiased BICUBIC (`resample: 3` in processor_config.json); the
all-NumPy filter (`_smoke/lfm25vl_preprocess.resize_antialias`) is within ~1 level of it and its
effect on the embeddings is measured separately (gate_tower.py G1b / G4).

Token path. The checkpoint's own `decider/prompt.py build()` (no shuffling), decoded and
re-encoded the way `decider/vision.py VisionDecisionModel.prepare()` hands the text to the
processor, with the image block in front:

    [<|vision_start|>, V+0 .. V+N-1, <|vision_end|>]   (image rows only; N = H*W merged tokens)
    + encode(decode(build(context, questions)))

* An id >= V (the text vocab, 248320) is row (id - V) of the tower output: the round-3
  ids-input decoder gathers it from `image_embeds` instead of the embedding table. The
  processor's form of the same row is N copies of <|image_pad|> (248056).
* Slots follow the author's rule: token " (" (318) preceded by ":" (25) with "Answer" (15666)
  among the 5 tokens before it; letters A..J are ids 32..41, read at every slot of one pass.
* M-RoPE from (ids, start, amount) alone: an image id V+k at index i has s0 = i - k and
  (t, h, w) = (s0, s0 + k // W, s0 + k % W) = (1, 1 + k // W, 1 + k % W); any other index i
  gets i - amount * (i >= start) on all three planes. start = 1 + N (the <|vision_end|> index),
  amount = N - max(H, W). Text-only rows: start = 1 << 30, amount = 0 (positions = indices).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PATCH = 16
MERGE = 2
TEMPORAL = 2
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5

VOCAB = 248320                      # text_config.vocab_size; image rows are ids VOCAB + k
VISION_START, IMAGE_PAD, VISION_END = 248053, 248056, 248054
SLOT_TOKEN, COLON, ANSWER = 318, 25, 15666
LETTERS = "ABCDEFGHIJ"
LETTER_IDS = tuple(range(32, 42))   # bare "A".."J"
MAX_OPTIONS = len(LETTERS)
MAX_CTX_TOKENS = 1536
NO_SHIFT = 1 << 30                  # rope_shift_start of a text-only row


def tile_side(grid: int) -> int:
    """Pixels per side of the square tile a merged grid of `grid` x `grid` covers."""
    return PATCH * MERGE * grid


def patchify(x: np.ndarray) -> np.ndarray:
    """Normalized [H,W,C] -> [gh*gw, C*T*P*P] in Qwen's merge-block-major order.

    Same layout as `_smoke/qwen38vl_preprocess.qwen_patchify` (the Qwen2VLImageProcessor
    reshape/permute): patches iterate (block_row, block_col, y-in-block, x-in-block), channel
    outermost inside the vector, the still frame repeated at both temporal slots.
    """
    h, w, c = x.shape
    if h % (PATCH * MERGE) or w % (PATCH * MERGE):
        raise ValueError(f"{h}x{w} not divisible by patch*merge {PATCH * MERGE}")
    gh, gw = h // PATCH, w // PATCH
    t = x.transpose(2, 0, 1).reshape(c, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    t = t.transpose(1, 4, 2, 5, 0, 3, 6).reshape(gh * gw, 1, c, PATCH, PATCH)
    t = np.broadcast_to(t[:, :, :, None], (gh * gw, 1, c, TEMPORAL, PATCH, PATCH))
    return t.reshape(gh * gw, c * TEMPORAL * PATCH * PATCH)


def resize_bicubic(u8: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Pillow's uint8 BICUBIC resize in NumPy — the form a non-Pillow host has to reproduce.

    Same per-axis filter as `_smoke/lfm25vl_preprocess.resize_antialias`, but in Pillow's order:
    the HORIZONTAL pass first, and the intermediate stored as uint8 (rounded, clipped) before the
    vertical pass. Bicubic rings on hard edges, so both details move pixels: the float,
    vertical-first filter is up to 17 levels off Pillow on the fixture's shapes; this form is
    bit-exact at 256 and within 2 levels at 448 (gate_tower.py G1c).
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_smoke"))
    from lfm25vl_preprocess import BICUBIC, _resample_axis

    x = np.asarray(u8, dtype=np.float64)
    if x.shape[1] != out_w:
        x = np.clip(np.floor(_resample_axis(x, out_w, axis=1, resample=BICUBIC) + 0.5), 0, 255)
    if x.shape[0] != out_h:
        x = np.clip(np.floor(_resample_axis(x, out_h, axis=0, resample=BICUBIC) + 0.5), 0, 255)
    return x.astype(np.uint8)


def preprocess(image, grid: int, resize: str = "pil") -> np.ndarray:
    """Image (path, PIL image or uint8 [H,W,3]) -> patches [4*grid^2, 1536] float32.

    `resize="pil"` is the gated reference (bit-equal to the processor); `"numpy"` is
    `resize_bicubic`, the portable form of the same filter.
    """
    from PIL import Image

    if isinstance(image, (str, Path)):
        im = Image.open(image)
    elif isinstance(image, Image.Image):
        im = image
    else:
        im = Image.fromarray(np.asarray(image, dtype=np.uint8))
    side = tile_side(grid)
    if resize == "pil":
        px = np.asarray(im.convert("RGB").resize((side, side), Image.Resampling.BICUBIC))
    elif resize == "numpy":
        px = resize_bicubic(np.asarray(im.convert("RGB")), side, side)
    else:
        raise ValueError(f"resize {resize!r} (pil | numpy)")
    x = (px.astype(np.float64) / 255.0 - IMAGE_MEAN) / IMAGE_STD
    return patchify(x).astype(np.float32)


# --------------------------------------------------------------------------- #
# Token ids
# --------------------------------------------------------------------------- #
def _encode(tok, text: str) -> list[int]:
    if hasattr(tok, "encode_batch"):                 # tokenizers.Tokenizer
        return list(tok.encode(text, add_special_tokens=False).ids)
    return list(tok.encode(text, add_special_tokens=False))


def _decode(tok, ids: list[int]) -> str:
    if hasattr(tok, "encode_batch"):
        return tok.decode(ids, skip_special_tokens=False)
    return tok.decode(ids)


def _question(q):
    """(text, options) from {"text", "options"} or the author's Q(text, options, gold)."""
    if isinstance(q, dict):
        return q["text"], list(q["options"])
    return q.text, list(q.options)


def build_text_ids(context: str, questions, tok, max_ctx_tokens: int = MAX_CTX_TOKENS) -> list[int]:
    """The author's `build()` with no shuffling: context, then one lettered block per question."""
    ids = _encode(tok, "Context:\n" + context)[:max_ctx_tokens]
    multi = len(questions) > 1
    for k, q in enumerate(questions):
        text, options = _question(q)
        if len(options) > MAX_OPTIONS:
            # The author samples down to 10 (keeping the gold and abstain options) at training
            # time; a host has no gold to keep, so it refuses instead of guessing that rule.
            raise ValueError(f"question {k}: {len(options)} options (at most {MAX_OPTIONS})")
        num = f" {k + 1}" if multi else ""
        lines = [f"\n\nQuestion{num}: {text}\nOptions:"]
        lines += [f"\n({LETTERS[j]}) {o}" for j, o in enumerate(options)]
        lines.append(f"\nAnswer{num}: (")
        ids.extend(_encode(tok, "".join(lines)))
    return ids


def find_slots(ids) -> list[int]:
    """The author's slot rule (decider/vision.py prepare()), on the final id row."""
    return [i for i in range(2, len(ids))
            if ids[i] == SLOT_TOKEN and ids[i - 1] == COLON and ANSWER in ids[max(0, i - 5):i]]


def build_ids(has_image: bool, context: str, questions, tok, merged_hw=None,
              max_ctx_tokens: int = MAX_CTX_TOKENS, vocab: int = VOCAB):
    """-> (ids, slots, rope_shift_start, rope_shift_amount) for one row.

    `ids` carries the image block as V+k (k = 0..H*W-1, row-major over the merged grid);
    `merged_hw` = (H, W) of the tower that produced the embeddings (g256: (8, 8)).
    """
    text = build_text_ids(context, questions, tok, max_ctx_tokens)
    text = _encode(tok, _decode(tok, text))          # prepare() hands decoded text to the processor
    if has_image:
        h, w = (int(v) for v in merged_hw)
        n = h * w
        ids = [VISION_START] + [vocab + k for k in range(n)] + [VISION_END] + text
        start, amount = 1 + n, n - max(h, w)
    else:
        ids, start, amount = text, NO_SHIFT, 0
    slots = find_slots(ids)
    if len(slots) != len(questions):                 # the author asserts the same
        raise ValueError(f"{len(slots)} slots for {len(questions)} questions")
    return ids, slots, start, amount


def rope_positions(ids, start: int, amount: int, merged_w: int, vocab: int = VOCAB) -> np.ndarray:
    """The three M-RoPE planes [3, T] from (ids, start, amount): what the decoder derives in-graph."""
    ids = np.asarray(ids, dtype=np.int64)
    i = np.arange(len(ids), dtype=np.int64)
    pos = np.broadcast_to(i - amount * (i >= start), (3, len(ids))).copy()
    img = ids >= vocab
    k = ids[img] - vocab
    s0 = i[img] - k
    pos[0, img] = s0
    pos[1, img] = s0 + k // merged_w
    pos[2, img] = s0 + k % merged_w
    return pos


def to_processor_ids(ids, vocab: int = VOCAB) -> list[int]:
    """V+k back to <|image_pad|>: the processor's form of the same row."""
    return [IMAGE_PAD if t >= vocab else int(t) for t in ids]
