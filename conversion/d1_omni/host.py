#!/usr/bin/env python3
"""d1-omni-600M host reference: a request -> one token row per question -> the graph's inputs -> the answers.

This file is the specification a Swift host copies. Every rule is the publisher's code (LiquidAI/d1-omni-600M @
414f8d64), written out so that nothing imports the publisher's package or transformers:

  - the row builder and the answer dictionaries: `prompt.py`, copied verbatim below (QTYPES, DELIM, MARKER,
    Question, as_question, escape, serialize, _criterion, render_options, encode, answer, temperature_key);
  - the per-mode dispatch, the temperature, the softmax and the noul flip: `modeling_d1.py`
    (`D1OmniModel.probabilities_batch` and `_forward`), written out as `request_rows` and
    `probabilities_from_logits`;
  - the prefix lengths of the media: `vision.py` `layout()` (copied verbatim) and `preprocess()` / `prefix_length()`
    (the crop shapes only), and `audio.py` `waveform()` / `MelFrontend` / `ConvSubsampling` (the length
    arithmetic; the mel itself is mel_host.py, round 6).

encode_rows.py gates this file against the publisher's `prompt.encode` and `probabilities_batch` on every fixture
row, and media_lengths.py gates the prefix lengths against the publisher's `preprocess` and audio front end.

1. Rows (`probabilities_batch`)

     mode   max_len (text positions)      noul default        option text after   state None   temperature
     text   16384                          none (criteria)     -                   ""           yes
     image  min(896, 16384 - P)            {false: no, true: yes}                  ""           no
     audio  min(15360, 16384 - P)          {false: no, true: yes}  option_000: ... {}  (JSON)   no
   P = the media prefix length (0 for text); the publisher refuses a request whose max_len falls below 64.
   One row per question: `encode(tok, state, q, max_len, noul_default, audio)` -> (ids, markers).
     ids = [1] + [17] + enc(state)[:room] + [18] + enc(instructions)... + per option [19, 16] + enc(" " + text)[:per]
           + [20] + [21]; markers = the positions of the 16s (text-relative)
     enc(s) = the tokenizer's ids of escape(s) without special tokens (`<|name|>` -> `<¦name¦>` first)
   usage.input_tokens = sum over the request's rows of (P + len(ids)).

2. Graph inputs (`graph_inputs`, d1_omni_model.py's `main` at bucket length L >= P + len(ids))

     input_ids [1,L] int32       0 at the prefix positions, the row's ids at [P, P+n), 0 (<|pad|>) after
     prefix_embeds [1,L,1024]    the media prefix at [0, P), 0.0 elsewhere (must be finite: it is multiplied by 0)
     pad_mask [1,L]              1.0 on [0, P+n)
     prefix_mask [1,L]           1.0 on [0, P)
     keep_right [1,L]            0.0 at P-1 when P > 0, 1.0 elsewhere
     qtype_onehot [1,3]          choice / score / noul
   scores [1,L] -> logits = scores[0, P + markers].

3. Probabilities (`_forward`)

     z = logits[:K] (K options); text only: z = z / T, T = temperatures[temperature_key(q)] if present, else
     temperatures[q.type] (1.0 in this config); p = softmax(z) in fp32; a noul is read as [false, true] and
     reported as [yes, no] (p reversed).

4. Answers (`answer`, verbatim): noul {"type", "noul": p_yes}; choice {"type", "choice", "confidence",
   "probabilities"}; score {"type", "score": sum i * p_i, "confidence", "probabilities", "legend"}; the response is
   {"answers": {name: answer}, "usage": {"input_tokens": n, "output_tokens": 0}}.

5. Image prefix (`vision_prefix`, the vision graph d1_omni_vision.py once per crop; round 5)

     crops         vision.preprocess(): a large image's 512 px tiles (row-major) then the thumbnail, each resized by
                   torchvision (bilinear, antialias) from the RGB image (`image_crop_pixels`); `crop_pixels_numpy` =
                   the same in NumPy (`resize_uint8_antialias_numpy`: torchvision's float32 path on a CPU without
                   AVX2, rounded half to even), the form the Swift host copies (round 9)
     pixel_values  (x - 127.5) / 127.5 in float32, 16 px patches row-major, [py][px][c] inside one (`patchify`),
                   0.0 past the crop's ph * pw patches (1,024 rows)
     pos_embed     the checkpoint's 16x16 table resized to (ph, pw), bilinear + antialias, float32 (`position_embeddings`
                   = the publisher's F.interpolate call; `position_embeddings_numpy` = the same filter in NumPy)
     patch_mask    1.0 on the crop's patches; unshuffle_index [256, 4]: token (i, j)'s four patches (`unshuffle_index`)
   prefix = each crop's first (ph/2)(pw/2) output rows, crops in order, images in order; P = image_prefix_length.

6. Audio prefix (`audio_prefix`, the audio graph d1_omni_audio.py once per clip, one bundle per clip bucket; round 6)

     mel           mel_host: waveform() (cut to 30 s, padded to 0.5 s) and MelFrontend (preemphasis, |STFT|^2 with a
                   centred Hann(400) in 512, Slaney mel 128, log(x + 2^-24), per-row normalisation over the valid
                   frames, 0.0 after); mel_torch = the publisher's code, mel_numpy = the same steps in NumPy (float64)
     bucket        the smallest of 5 / 10 / 20 / 30 s whose F = 1 + 100 * sec columns hold the clip's 1 + n // 160
                   (`audio_bucket_for`); the mel is zero-padded to F
     masks         mask_f / mask_f2 / mask_f4 / mask_t: 1.0 on t < l0, l1, l2, l3 with l0 = the valid frames (n // 160)
                   and l(k+1) = (lk - 1) // 2 + 1 (ConvSubsampling's lengths); P = l3 = audio_prefix_length
   prefix = the graph's first P rows; this is the decision graph's prefix_embeds [0, P) (`audio_request_rows`).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "LiquidAI/d1-omni-600M"
MODEL_SHA = "414f8d6438174f5b2133a9c21a478fc42625e308"
COPIED_FROM = {  # sha256 of the publisher's files at MODEL_SHA that the code below copies
    "prompt.py": "a6b29a55ec8345f1fc1fdbfcc4b64d80d473dc4316f095a62c51cb6cce1194cf",
    "modeling_d1.py": "2b71a5c909ec5d3aa6307378a41a0fd24af1e17d5c1c1f8d510582147051a686",
    "vision.py": "43ad71029f82e20fd26b97574627773d47639d7a3742c8b04fe5bf55f41c4c28",
    "audio.py": "c5ff09d52d6a079e79d0505c89dbd0582ba2bc9abf2b26595d8d5b1d0c347bdf",
}
TOKENIZER_SHA256 = "1efc3a6609abf6b63b1f47188d139f3b59973a6a434dffe970a7261a51ed2711"  # tokenizer.json @ MODEL_SHA
# The ids the row builder uses, read from tokenizer.json (check_token_ids asserts them on any tokenizer object).
TOKEN_IDS = {"<|pad|>": 0, "<|startoftext|>": 1, "<|im_end|>": 7, "<|mask|>": 16, "<|reserved_7|>": 17,
             "<|reserved_8|>": 18, "<|reserved_9|>": 19, "<|reserved_10|>": 20, "<|reserved_11|>": 21}
BOS_ID, PAD_ID, MASK_ID = 1, 0, 16
# config.json values the host reads (d1_omni_model.EXPECTED_CONFIG holds the whole file).
CONFIG = {
    "max_length": 16384, "image_text_length": 896, "audio_text_length": 15360,
    "temperatures": {
        "choice:11+": 1.372515082359314, "choice:2": 1.7465145587921143, "choice:3-5": 1.3998981714248657,
        "choice:6-10": 1.1751071214675903, "noul:2": 1.6663223505020142, "score:3-5": 1.7301132678985596,
        "score:6-10": 1.0, "choice": 1.0, "score": 1.0, "noul": 1.0,
    },
}
MODES = ("text", "image", "audio")
BUCKETS = (256, 512, 1024, 2048, 4096)  # the decision graph's lengths (supervisor's design, round 1)
# Round 12 (the chunk lever): the decision graph also exported at L64 / L128. BUCKETS stays the shipped set, the default
# of bucket_for: every reference, gate and host that recorded a bucket (records_ref*.json, reference.json, gate_swift,
# the Swift host's folder scan) was made with it. A host that ships the small buckets calls
# bucket_for(positions, ALL_BUCKETS).
SMALL_BUCKETS = (64, 128)
ALL_BUCKETS = (*SMALL_BUCKETS, *BUCKETS)
MIN_TEXT_POSITIONS = 64                 # modeling_d1: "the media take p of the max_length positions"


# =========================================================================== prompt.py @ 414f8d64, verbatim
QTYPES = {"choice": 0, "score": 1, "noul": 2}
DELIM = {"state": "<|reserved_7|>", "q": "<|reserved_8|>", "opt": "<|reserved_9|>", "opt_end": "<|reserved_10|>",
         "decide": "<|reserved_11|>"}
MARKER = "<|mask|>"
_SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


@dataclass
class Question:
    type: str
    instructions: str
    criteria: Any = None

    def __post_init__(self):
        if self.type not in QTYPES:
            raise ValueError(f"question type must be one of {sorted(QTYPES)}, got {self.type!r}")
        if self.type == "choice" and (not isinstance(self.criteria, dict) or len(self.criteria) < 2):
            raise ValueError("a choice needs criteria {name: description} with at least two options")
        if self.type == "score" and (not isinstance(self.criteria, (list, tuple)) or not 2 <= len(self.criteria) <= 10):
            raise ValueError("a score needs criteria: a list of 2 to 10 level descriptions, lowest first")
        if self.type == "noul" and self.criteria is not None and not isinstance(self.criteria, dict):
            raise ValueError('noul criteria are optional: {"true": "...", "false": "..."} (or "yes", "no")')

    @property
    def options(self) -> int:
        return 2 if self.type == "noul" else len(self.criteria)


def as_question(q: Any) -> Question:
    if isinstance(q, Question):
        return q
    if not isinstance(q, dict) or "type" not in q or "instructions" not in q:
        raise ValueError("a question is a dict with `type`, `instructions` and, for choice and score, `criteria`")
    return Question(q["type"], str(q["instructions"]), q.get("criteria"))


def escape(text: str) -> str:
    """`<|name|>` -> `<¦name¦>`, so caller text cannot emit a delimiter or marker token."""
    return _SPECIAL.sub(r"<¦\1¦>", text)


def serialize(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def _criterion(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(", ", ": "),
                                                           default=str)


def render_options(q: Question, noul_default: dict | None = None, audio: bool = False) -> list[str]:
    """Option texts in the model's order. A noul is read as [false, true]. After an audio prefix, options are
    written as the audio questions were trained: `option_000: text`, and a noul as `false: no`, `true: yes`."""
    if q.type == "choice":
        if audio:
            return [f"option_{i:03d}: {_criterion(k if v is None or v == '' else v)}"
                    for i, (k, v) in enumerate(q.criteria.items())]
        return [k if v is None or v == "" else f"{k}: {_criterion(v)}" for k, v in q.criteria.items()]
    if q.type == "score":
        return [f"level {i}: {_criterion(c)}" for i, c in enumerate(q.criteria)]
    if audio:
        return ["false: no", "true: yes"]
    crit = q.criteria or noul_default or {}
    false, true = crit.get("false", crit.get("no")), crit.get("true", crit.get("yes"))
    return ["false: " + (_criterion(false) if false not in (None, "") else "no, the statement does not hold"),
            "true: " + (_criterion(true) if true not in (None, "") else "yes, the statement holds")]


def encode(tok, state: Any, q: Question, max_len: int, noul_default: dict | None = None, audio: bool = False,
           per_option: int = 24) -> tuple[list[int], list[int]]:
    """Token ids of one question over one state, and the position of each option's marker.

    The option block gets max(96, min(24k + 32, max_len / 2)) tokens, shared out evenly; the state is
    truncated on the right to the room that is left.
    """
    ids_of = tok.convert_tokens_to_ids
    enc = lambda s: tok(escape(s), add_special_tokens=False)["input_ids"]  # noqa: E731
    opts = render_options(q, noul_default, audio)
    budget = max(96, min(len(opts) * per_option + 32, max_len // 2))
    per = max(2, (budget - 3 * len(opts)) // len(opts))
    question = ([ids_of(DELIM["q"])] + enc(q.instructions))[: max(16, budget)]
    markers = []
    for text in opts:
        markers.append(len(question) + 1)
        question += [ids_of(DELIM["opt"]), ids_of(MARKER)] + enc(" " + text)[:per] + [ids_of(DELIM["opt_end"])]
    question.append(ids_of(DELIM["decide"]))
    room = max(0, max_len - len(question) - 2)
    state_ids = [ids_of(DELIM["state"])] + enc(serialize(state))[:room]
    ids = ([tok.bos_token_id] + state_ids + question)[:max_len]
    markers = [m + 1 + len(state_ids) for m in markers]
    if markers[-1] >= max_len:
        raise ValueError("the options do not fit in the context")
    return ids, markers


def answer(q: Question, probs: list[float]) -> dict:
    """A noul's P(yes) (its probabilities are [yes, no]); a choice's pick and its probabilities; a score's
    expected level."""
    if q.type == "noul":
        return {"type": "noul", "noul": probs[0]}
    best = max(range(len(probs)), key=probs.__getitem__)
    if q.type == "choice":
        names = list(q.criteria)
        return {"type": "choice", "choice": names[best], "confidence": probs[best],
                "probabilities": dict(zip(names, probs))}
    return {"type": "score", "score": sum(i * p for i, p in enumerate(probs)), "confidence": probs[best],
            "probabilities": {str(i): p for i, p in enumerate(probs)},
            "legend": {str(i): _criterion(text) for i, text in enumerate(q.criteria)}}


def temperature_key(q: Question) -> str:
    k = q.options
    return f"{q.type}:" + ("2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+")
# =========================================================================== end of the prompt.py copy


# --------------------------------------------------------------------------- tokenizer
class RawTokenizer:
    """tokenizer.json through the `tokenizers` library, with the three things `encode` calls on a transformers
    tokenizer: `convert_tokens_to_ids`, `bos_token_id`, and `tok(text, add_special_tokens=False)["input_ids"]`."""

    def __init__(self, tokenizer_json: str | Path):
        from tokenizers import Tokenizer

        self.path = Path(tokenizer_json)
        self._tok = Tokenizer.from_file(str(self.path))
        self.bos_token_id = self._tok.token_to_id("<|startoftext|>")
        self.pad_token_id = self._tok.token_to_id("<|pad|>")
        self.mask_token_id = self._tok.token_to_id(MARKER)

    def convert_tokens_to_ids(self, token: str) -> int:
        token_id = self._tok.token_to_id(token)
        if token_id is None:
            raise KeyError(token)
        return token_id

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        return {"input_ids": self._tok.encode(text, add_special_tokens=add_special_tokens).ids}


def check_token_ids(tok) -> dict:
    """The row builder's ids on this tokenizer object = TOKEN_IDS (raises otherwise)."""
    got = {name: tok.convert_tokens_to_ids(name) for name in TOKEN_IDS}
    got_special = {"bos_token_id": tok.bos_token_id, "pad_token_id": getattr(tok, "pad_token_id", None),
                   "mask_token_id": getattr(tok, "mask_token_id", None)}
    expected_special = {"bos_token_id": BOS_ID, "pad_token_id": PAD_ID, "mask_token_id": MASK_ID}
    if got != TOKEN_IDS or got_special != expected_special:
        raise ValueError(f"token ids {got} {got_special} != {TOKEN_IDS} {expected_special}")
    return {**got, **got_special}


# --------------------------------------------------------------------------- rows (modeling_d1.probabilities_batch)
YES_NO = {"false": "no", "true": "yes"}  # how image and audio questions were trained


@dataclass
class Row:
    question: Question
    ids: list[int]
    markers: list[int]          # text-relative, as encode returns them
    calibrate: bool             # text rows take the temperature
    prefix_len: int             # P
    mode: str
    max_len: int                # the text positions this row was built against
    qid: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def positions(self) -> int:
        return self.prefix_len + len(self.ids)


def mode_settings(mode: str, prefix_len: int = 0, config: dict = CONFIG) -> dict:
    """max_len / noul default / audio flag / temperature for one request, as probabilities_batch picks them."""
    if mode == "image":
        max_len, noul, calibrate, spoken = config["image_text_length"], YES_NO, False, False
    elif mode == "audio":
        max_len, noul, calibrate, spoken = config["audio_text_length"], YES_NO, False, True
    elif mode == "text":
        if prefix_len:
            raise ValueError("a text request has no media prefix")
        max_len, noul, calibrate, spoken = config["max_length"], None, True, False
    else:
        raise ValueError(f"mode must be one of {MODES}")
    max_len = min(max_len, config["max_length"] - prefix_len)
    if max_len < MIN_TEXT_POSITIONS:
        raise ValueError(f"the media take {prefix_len} of the {config['max_length']} positions; send fewer images")
    return {"max_len": max_len, "noul_default": noul, "calibrate": calibrate, "audio": spoken}


def request_rows(tok, state: Any, questions, mode: str = "text", prefix_len: int = 0,
                 config: dict = CONFIG) -> list[Row]:
    """One request (a state, its questions as a {name: question} dict or a list, and the media's prefix length)
    -> one Row per question, in order."""
    named = list(questions.items()) if isinstance(questions, dict) else [(None, q) for q in questions]
    qs = [(name, as_question(q)) for name, q in named]
    settings = mode_settings(mode, prefix_len, config)
    if mode == "audio":
        state = {} if state is None else state  # how the audio questions were trained
    state = "" if state is None else state
    rows = []
    for name, q in qs:
        ids, markers = encode(tok, state, q, settings["max_len"], settings["noul_default"], settings["audio"])
        rows.append(Row(q, ids, markers, settings["calibrate"], prefix_len, mode, settings["max_len"], name))
    return rows


def usage_tokens(rows: list[Row]) -> int:
    return sum(row.positions for row in rows)


# --------------------------------------------------------------------------- graph inputs
def bucket_for(positions: int, buckets=BUCKETS) -> int | None:
    """The smallest bucket that holds the row's positions (prefix + text), None if none does."""
    return next((b for b in buckets if positions <= b), None)


def graph_inputs(row: Row, seq_len: int, prefix: np.ndarray | None = None, hidden: int = 1024) -> tuple[dict, list[int]]:
    """The decision graph's six inputs for one row at bucket length seq_len, and the marker positions in it."""
    p, n = row.prefix_len, len(row.ids)
    if p + n > seq_len:
        raise ValueError(f"row of {p} + {n} positions does not fit length {seq_len}")
    input_ids = np.full((1, seq_len), PAD_ID, dtype=np.int32)
    input_ids[0, p:p + n] = row.ids
    prefix_embeds = np.zeros((1, seq_len, hidden), dtype=np.float32)
    if p:
        if prefix is None or tuple(prefix.shape[-2:]) != (p, hidden):
            raise ValueError(f"a row with a {p}-position prefix needs prefix [.., {p}, {hidden}]")
        prefix_embeds[0, :p] = np.asarray(prefix, dtype=np.float32).reshape(p, hidden)
    pad_mask = np.zeros((1, seq_len), dtype=np.float32)
    pad_mask[0, :p + n] = 1.0
    prefix_mask = np.zeros((1, seq_len), dtype=np.float32)
    prefix_mask[0, :p] = 1.0
    keep_right = np.ones((1, seq_len), dtype=np.float32)
    if p:
        keep_right[0, p - 1] = 0.0
    qtype_onehot = np.zeros((1, 3), dtype=np.float32)
    qtype_onehot[0, QTYPES[row.question.type]] = 1.0
    inputs = {"input_ids": input_ids, "prefix_embeds": prefix_embeds, "pad_mask": pad_mask,
              "prefix_mask": prefix_mask, "keep_right": keep_right, "qtype_onehot": qtype_onehot}
    return inputs, [p + m for m in row.markers]


# --------------------------------------------------------------------------- probabilities (modeling_d1._forward)
def temperature(q: Question, temperatures: dict = CONFIG["temperatures"]) -> float:
    return temperatures.get(temperature_key(q), temperatures.get(q.type, 1.0))


def softmax32(z) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    e = np.exp(z - z.max())
    return e / e.sum()


def probabilities_from_logits(logits, q: Question, calibrate: bool,
                              temperatures: dict = CONFIG["temperatures"]) -> list[float]:
    """Marker logits (at least K of them, in option order) -> the publisher's option distribution: the first K,
    divided by T for text rows, softmax in fp32, a noul reported as [yes, no]."""
    z = np.asarray(logits, dtype=np.float32).reshape(-1)[: q.options]
    if calibrate:
        z = z / np.float32(temperature(q, temperatures))
    p = softmax32(z).tolist()
    return p[::-1] if q.type == "noul" else p


def response(rows: list[Row], probabilities: list[list[float]]) -> dict:
    """system_one's response body for one request's rows (named questions)."""
    return {"answers": {row.qid: answer(row.question, p) for row, p in zip(rows, probabilities)},
            "usage": {"input_tokens": usage_tokens(rows), "output_tokens": 0}}


# =========================================================================== vision.py @ 414f8d64: layout, verbatim
TILE, PATCH, MAX_PATCHES = 512, 16, 1024


def layout(width: int, height: int) -> dict:
    """LFM2-VL's smart resize, tile grid and thumbnail (projected-patch budget 64 to 256 per crop)."""
    if min(width, height) < 1:
        raise ValueError("empty image")
    factor, maximum, minimum = 32, 256 * 1024, 64 * 1024
    h, w = max(factor, round(height / factor) * factor), max(factor, round(width / factor) * factor)
    if h * w > maximum:
        beta = math.sqrt(height * width / maximum)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < minimum:
        beta = math.sqrt(minimum / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    large = max(16, round(height / factor) * factor) * max(16, round(width / factor) * factor) > maximum * 2
    grid = (1, 1)
    if large:
        ratios = sorted({(x, y) for n in range(2, 11) for x in range(1, n + 1) for y in range(1, n + 1)
                         if 2 <= x * y <= 10}, key=lambda r: r[0] * r[1])
        best = float("inf")
        for ratio in ratios:
            diff = abs(width / height - ratio[0] / ratio[1])
            if diff < best or (diff == best and width * height > 0.5 * TILE * TILE * ratio[0] * ratio[1]):
                grid, best = ratio, diff
    return {"grid": grid, "thumbnail": (h, w), "tiled": large}
# =========================================================================== end of the layout copy


def image_crops(width: int, height: int) -> list[tuple[int, int]]:
    """preprocess()'s crops in order, as (height, width) in pixels: the grid's 512 px tiles (row-major) when the
    image is tiled, then the thumbnail."""
    plan = layout(width, height)
    crops = []
    if plan["tiled"]:
        gw, gh = plan["grid"]
        crops = [(TILE, TILE)] * (gw * gh)
    return crops + [tuple(plan["thumbnail"])]


def image_prefix_length(sizes) -> int:
    """prefix_length(): sum over every image's crops of (h / 16) * (w / 16) / 4 (one embedding per 2x2 patches).
    `sizes` = [(width, height), ...] in request order."""
    return sum((h // PATCH) * (w // PATCH) // 4 for width, height in sizes for h, w in image_crops(width, height))


# --------------------------------------------------------------------------- vision graph inputs (round 5)
# The vision graph (d1_omni_vision.py, function main) runs once per crop: pixel_values [1,1024,768], pos_embed
# [1,1024,768], patch_mask [1,1024] (float32) and unshuffle_index [256,4] (int32) -> prefix [1,256,1024]; the crop's
# first (ph/2)(pw/2) rows are its prefix. The crops and their order are vision.preprocess()'s; the image's prefix is
# its crops' prefixes in that order (Vision.forward), and a request's is its images' in request order.
VISION_PATCHES, VISION_TOKENS, PATCH_DIM, VISION_HIDDEN = MAX_PATCHES, MAX_PATCHES // 4, 3 * PATCH * PATCH, 768
POSITION_KEY = "vision.tower.vision_model.embeddings.position_embedding.weight"


def load_position_table(safetensors_path) -> np.ndarray:
    """The checkpoint's 16x16 NaFlex position table, [16, 16, 768] float32 (the host's constant: the graph does not
    hold it, because its resize depends on the crop)."""
    from safetensors import safe_open

    with safe_open(str(safetensors_path), framework="np") as f:
        table = f.get_tensor(POSITION_KEY)
    side = int(round(table.shape[0] ** 0.5))
    return np.ascontiguousarray(table.reshape(side, side, -1), dtype=np.float32)


def image_crop_pixels(image) -> list[np.ndarray]:
    """preprocess()'s crops of one PIL image as uint8 [3, h, w] arrays, in order (the tiles row-major, then the
    thumbnail): torchvision's resize (bilinear, antialias) on the RGB image, as the publisher calls it."""
    import torch  # noqa: F401  (torchvision needs it)
    from torchvision.transforms.v2 import functional as tvf

    image = image.convert("RGB")
    plan = layout(*image.size)
    x = tvf.pil_to_tensor(image)
    crops = []
    if plan["tiled"]:
        gw, gh = plan["grid"]
        big = tvf.resize(x, [gh * TILE, gw * TILE], interpolation=tvf.InterpolationMode.BILINEAR, antialias=True)
        crops = [big[:, r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] for r in range(gh) for c in range(gw)]
    crops.append(tvf.resize(x, list(plan["thumbnail"]), interpolation=tvf.InterpolationMode.BILINEAR, antialias=True))
    return [c.numpy() for c in crops]


def patchify(crop: np.ndarray) -> np.ndarray:
    """uint8 [3, h, w] -> the crop's patches [ph * pw, 768] float32: (x - 127.5) / 127.5 in float32, then
    reshape(3, ph, 16, pw, 16).permute(1, 3, 2, 4, 0): row-major patches, each [py][px][c] (channel fastest)."""
    c, h, w = crop.shape
    ph, pw = h // PATCH, w // PATCH
    x = (crop.astype(np.float32) - np.float32(127.5)) / np.float32(127.5)
    return x.reshape(c, ph, PATCH, pw, PATCH).transpose(1, 3, 2, 4, 0).reshape(ph * pw, c * PATCH * PATCH)


def position_embeddings(table: np.ndarray, ph: int, pw: int) -> np.ndarray:
    """Siglip2VisionEmbeddings.resize_positional_embeddings for one crop: the [16, 16, 768] table resized to (ph, pw)
    with F.interpolate(bilinear, align_corners=False, antialias=True) in float32 on the CPU, the same call on the same
    memory layout -> [ph * pw, 768] float32, row-major."""
    import torch
    from torch.nn import functional as F

    flat = torch.from_numpy(np.ascontiguousarray(table, dtype=np.float32).reshape(-1, table.shape[-1]))
    side = table.shape[0]
    grid = flat.reshape(side, side, -1).permute(2, 0, 1).unsqueeze(0)  # (h, w, d) -> (1, d, h, w), as the publisher
    out = F.interpolate(grid, size=(ph, pw), mode="bilinear", align_corners=False, antialias=True)
    return out.reshape(table.shape[-1], ph * pw).transpose(0, 1).contiguous().numpy()


def _aa_weights_f32(n_in: int, n_out: int) -> list[tuple[int, np.ndarray]]:
    """The antialiased bilinear (triangle) weights of one axis as torch computes them for a float32 input (aten
    _compute_weights_aa, scalar_t = float): scale = n_in / n_out and the support max(1, scale) in float32; the centre
    scale * (i + 0.5) and each tap's filter argument pass through double, as the C++ expressions do; the tap count is
    clamped to ceil(support) * 2 + 1 (torch's max_interp_size); the weights are summed and divided in float32.
    -> [(first tap, weights)] per output index."""
    f32 = np.float32
    scale = f32(n_in) / f32(n_out)
    support = scale if scale >= 1.0 else f32(1.0)
    invscale = f32(1.0) / scale if scale >= 1.0 else f32(1.0)
    max_taps = math.ceil(support) * 2 + 1
    rows = []
    for i in range(n_out):
        centre = f32(float(scale) * (i + 0.5))
        lo = max(int(float(centre - support) + 0.5), 0)
        size = min(max(min(int(float(centre + support) + 0.5), n_in) - lo, 0), max_taps)
        weights, total = [], f32(0.0)
        for j in range(size):
            x = abs(f32((float(f32(j + lo) - centre) + 0.5) * float(invscale)))
            w = f32(1.0) - x if x < 1.0 else f32(0.0)
            weights.append(w)
            total = f32(total + w)
        if total != 0.0:
            weights = [f32(w / total) for w in weights]
        rows.append((lo, np.asarray(weights, dtype=np.float32)))
    return rows


def _aa_pass_f32(x: np.ndarray, rows: list[tuple[int, np.ndarray]], axis: int) -> np.ndarray:
    """One separable pass in float32: out = t0 * w0, then out = fma(tj, wj, out) tap by tap (torch's loop, built with
    fused multiply-add on arm64). The product of two float32 values is exact in float64, so the fused step is the
    float64 sum rounded once to float32 (a Swift host: Float.addingProduct)."""
    src = np.moveaxis(x, axis, 0)
    out = np.empty((len(rows),) + src.shape[1:], dtype=np.float32)
    for i, (lo, weights) in enumerate(rows):
        acc = src[lo] * weights[0]
        for j in range(1, len(weights)):
            acc = (acc.astype(np.float64) + src[lo + j].astype(np.float64) * np.float64(weights[j])).astype(np.float32)
        out[i] = acc
    return np.moveaxis(out, 0, axis)


def position_embeddings_numpy(table: np.ndarray, ph: int, pw: int) -> np.ndarray:
    """position_embeddings in NumPy, the form a Swift host copies: torch's float32 antialias kernel written out --
    the width pass first (the contiguous dimension), then the height pass, each tap a fused multiply-add in float32
    with the weights of _aa_weights_f32. Equal to the publisher's F.interpolate bit for bit on every crop shape
    vision_check.py runs (a float64 filter differs by up to 2e-5: torch rounds its weights to float32)."""
    x = np.asarray(table, dtype=np.float32)
    x = _aa_pass_f32(x, _aa_weights_f32(x.shape[1], pw), axis=1)
    x = _aa_pass_f32(x, _aa_weights_f32(x.shape[0], ph), axis=0)
    return np.ascontiguousarray(x.reshape(ph * pw, -1))


def resize_uint8_antialias_numpy(crop: np.ndarray, height: int, width: int) -> np.ndarray:
    """torchvision.transforms.v2.functional.resize(uint8 [3, h, w], [height, width], BILINEAR, antialias=True) in
    NumPy, the form a Swift host copies -- as torchvision 0.24 runs it on a CPU without AVX2 (Apple silicon: no native
    uint8 kernel; `_do_native_uint8_resize_on_cpu` is False): the same size returns the image unchanged; otherwise the
    image is cast to float32, resized by torch's separable float32 antialias kernel (the width pass, then the height
    pass, a pass skipped when its side does not change; the weights of _aa_weights_f32, each tap a fused
    multiply-add: _aa_pass_f32), rounded half to even (torch.round) and cast back to uint8 (no clamp: the triangle
    weights are non-negative and sum to 1, so every value stays in [0, 255]). -> uint8 [3, height, width]."""
    _, h, w = crop.shape
    if (height, width) == (h, w):
        return crop
    x = crop.astype(np.float32)
    if width != w:
        x = _aa_pass_f32(x, _aa_weights_f32(w, width), axis=2)
    if height != h:
        x = _aa_pass_f32(x, _aa_weights_f32(h, height), axis=1)
    return np.rint(x).astype(np.uint8)


def rgb_uint8(image) -> np.ndarray:
    """A PIL image as the publisher reads it: `convert("RGB")` (no EXIF rotation, no colour management) -> uint8
    [3, h, w]. The bytes a Swift host's decoder must reproduce."""
    return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1))


def crop_pixels_numpy(rgb: np.ndarray) -> list[np.ndarray]:
    """image_crop_pixels on a uint8 [3, h, w] RGB array with resize_uint8_antialias_numpy instead of torchvision:
    preprocess()'s crops in order (the tiles of the grid's resize, row-major, then the thumbnail)."""
    _, h, w = rgb.shape
    plan = layout(w, h)
    crops = []
    if plan["tiled"]:
        gw, gh = plan["grid"]
        big = resize_uint8_antialias_numpy(rgb, gh * TILE, gw * TILE)
        crops = [big[:, r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] for r in range(gh) for c in range(gw)]
    crops.append(resize_uint8_antialias_numpy(rgb, *plan["thumbnail"]))
    return [np.ascontiguousarray(c) for c in crops]


def unshuffle_index(ph: int, pw: int) -> np.ndarray:
    """vision.Projector's 2x2 pixel unshuffle as a gather: token t = i * (pw / 2) + j takes the patches
    (2i, 2j), (2i, 2j+1), (2i+1, 2j), (2i+1, 2j+1) of the (ph, pw) grid, in that order -> [256, 4] int32 (0 past the
    crop's tokens)."""
    if ph % 2 or pw % 2 or ph * pw > VISION_PATCHES:
        raise ValueError(f"grid {ph}x{pw}: needs even sides and at most {VISION_PATCHES} patches")
    i, j = np.divmod(np.arange((ph // 2) * (pw // 2)), pw // 2)
    index = np.zeros((VISION_TOKENS, 4), dtype=np.int32)
    index[:i.size] = np.stack([2 * i * pw + 2 * j, 2 * i * pw + 2 * j + 1, (2 * i + 1) * pw + 2 * j,
                               (2 * i + 1) * pw + 2 * j + 1], axis=1)
    return index


def crop_inputs(crop: np.ndarray, table: np.ndarray, numpy_positions: bool = False) -> dict:
    """One uint8 [3, h, w] crop -> the vision graph's four inputs (batch 1, padded to 1,024 patches / 256 tokens),
    plus `grid` (ph, pw) and `tokens` (the prefix rows the crop gives)."""
    _, h, w = crop.shape
    ph, pw = h // PATCH, w // PATCH
    n = ph * pw
    pixel_values = np.zeros((1, VISION_PATCHES, PATCH_DIM), dtype=np.float32)
    pixel_values[0, :n] = patchify(crop)
    pos_embed = np.zeros((1, VISION_PATCHES, VISION_HIDDEN), dtype=np.float32)
    pos_embed[0, :n] = (position_embeddings_numpy if numpy_positions else position_embeddings)(table, ph, pw)
    patch_mask = np.zeros((1, VISION_PATCHES), dtype=np.float32)
    patch_mask[0, :n] = 1.0
    return {"pixel_values": pixel_values, "pos_embed": pos_embed, "patch_mask": patch_mask,
            "unshuffle_index": unshuffle_index(ph, pw), "grid": (ph, pw), "tokens": n // 4}


def image_crops_inputs(image, table: np.ndarray, numpy_positions: bool = False, numpy_resize: bool = False) -> list[dict]:
    """A PIL image -> crop_inputs for each of preprocess()'s crops, in order. numpy_resize / numpy_positions: the
    NumPy forms (crop_pixels_numpy, position_embeddings_numpy) a Swift host copies, instead of torchvision / torch."""
    crops = crop_pixels_numpy(rgb_uint8(image)) if numpy_resize else image_crop_pixels(image)
    return [crop_inputs(c, table, numpy_positions) for c in crops]


VISION_INPUT_NAMES = ("pixel_values", "pos_embed", "patch_mask", "unshuffle_index")


def vision_prefix(run_fn, images, table: np.ndarray, numpy_positions: bool = False) -> np.ndarray:
    """The images' prefix [P, 1024]: run_fn(inputs) -> the graph's prefix [1, 256, 1024] for each crop (inputs = the
    four arrays by name), its first `tokens` rows kept, crops in order, images in order (Vision.forward)."""
    parts = []
    for image in (images if isinstance(images, (list, tuple)) else [images]):
        for crop in image_crops_inputs(image, table, numpy_positions):
            out = np.asarray(run_fn({k: crop[k] for k in VISION_INPUT_NAMES}), dtype=np.float32)
            parts.append(out.reshape(VISION_TOKENS, -1)[:crop["tokens"]])
    return np.concatenate(parts, axis=0)


def image_request_rows(tok, state: Any, questions, images, config: dict = CONFIG) -> list[Row]:
    """request_rows for an image request: P = image_prefix_length of the images' sizes (the prefix the vision graph
    returns has exactly P rows), max_len = min(896, 16384 - P)."""
    sizes = [image.size for image in (images if isinstance(images, (list, tuple)) else [images])]
    return request_rows(tok, state, questions, "image", image_prefix_length(sizes), config)


# --------------------------------------------------------------------------- audio.py @ 414f8d64: lengths only
SAMPLE_RATE, MIN_SAMPLES, MAX_SECONDS, HOP = 16000, 8000, 30, 160


def audio_samples(n: int) -> int:
    """waveform(): cut to 30 s, zero-padded to 0.5 s."""
    return max(min(n, MAX_SECONDS * SAMPLE_RATE), MIN_SAMPLES)


def audio_frames(n: int) -> dict:
    """MelFrontend: STFT columns (center=True) and the valid frames the normalisation reads, for n raw samples."""
    s = audio_samples(n)
    return {"samples": s, "stft_frames": 1 + s // HOP, "valid_frames": (s + 512 // 2 * 2 - 512) // HOP}


def _subsample(length: int) -> int:
    return (length + 2 - 3) // 2 + 1  # ConvSubsampling: floor((l + 2*pad - kernel) / stride) + 1, three times


def audio_prefix_length(n: int) -> int:
    """The prefix positions of one clip of n samples: the valid frames through the 8x subsampling."""
    length = audio_frames(n)["valid_frames"]
    for _ in range(3):
        length = _subsample(length)
    return length


def audio_encoder_steps(n: int) -> int:
    """The conformer's time axis (all STFT columns through the subsampling; the last steps are masked)."""
    length = audio_frames(n)["stft_frames"]
    for _ in range(3):
        length = _subsample(length)
    return length


# --------------------------------------------------------------------------- audio graph inputs (round 6)
# The audio graph (d1_omni_audio.py, function main, one bundle per clip bucket) runs once per clip: mel [1,128,F] and
# four time masks mask_f [1,F], mask_f2 [1,F2], mask_f4 [1,F4], mask_t [1,T] (float32) -> prefix [1,T,1024]; the clip's
# prefix is its first P rows (P = audio_prefix_length). The mel is mel_host's (the publisher's waveform() and
# MelFrontend); the masks are ConvSubsampling's _time_mask lengths: l0 = the valid frames, l(k+1) = _subsample(lk).
AUDIO_CLIP_SECONDS = (5, 10, 20, 30)
AUDIO_FEATURES, AUDIO_PREFIX_HIDDEN = 128, 1024
AUDIO_INPUT_NAMES = ("mel", "mask_f", "mask_f2", "mask_f4", "mask_t")


def audio_bucket_shapes(sec: int) -> dict:
    """One clip bucket's static lengths: F = 1 + 100 * sec mel columns, then F2, F4, T through the subsampling."""
    if sec not in AUDIO_CLIP_SECONDS:
        raise ValueError(f"clip bucket must be one of {AUDIO_CLIP_SECONDS} s")
    f = 1 + sec * SAMPLE_RATE // HOP
    f2 = _subsample(f)
    f4 = _subsample(f2)
    return {"sec": sec, "F": f, "F2": f2, "F4": f4, "T": _subsample(f4)}


def audio_bucket_for(n: int) -> int:
    """The smallest clip bucket whose F holds the clip's STFT columns (1 + n // 160 after waveform()'s cut and pad).
    Equal to the smallest sec >= the clip's seconds, except that a clip up to 159 samples past a bucket's seconds
    still fits that bucket (its columns are the same)."""
    columns = audio_frames(n)["stft_frames"]
    return next(sec for sec in AUDIO_CLIP_SECONDS if columns <= audio_bucket_shapes(sec)["F"])


def audio_inputs(audio, sec: int | None = None, numpy_mel: bool = False) -> dict:
    """One clip (1-D 16 kHz int16 or float samples) -> the audio graph's five inputs at bucket `sec` (default: the
    smallest that holds it), plus `sec`, `frames`, `prefix_rows` (P) and `steps` (T). numpy_mel: mel_host.mel_numpy (the
    Swift host's form) instead of mel_host.mel_torch (the publisher's code)."""
    import mel_host

    n = len(audio)
    sec = audio_bucket_for(n) if sec is None else sec
    shapes = audio_bucket_shapes(sec)
    mel, frames = (mel_host.mel_numpy if numpy_mel else mel_host.mel_torch)(audio)
    if mel.shape[1] > shapes["F"]:
        raise ValueError(f"a clip of {mel.shape[1]} mel columns does not fit the {sec} s bucket ({shapes['F']})")
    if frames != audio_frames(n)["valid_frames"]:
        raise AssertionError(f"frames {frames} != host.audio_frames {audio_frames(n)}")
    inputs = {"mel": np.zeros((1, AUDIO_FEATURES, shapes["F"]), dtype=np.float32)}
    inputs["mel"][0, :, :mel.shape[1]] = mel
    length = frames
    for name, key in (("mask_f", "F"), ("mask_f2", "F2"), ("mask_f4", "F4"), ("mask_t", "T")):
        mask = np.zeros((1, shapes[key]), dtype=np.float32)
        mask[0, :length] = 1.0
        inputs[name] = mask
        length = _subsample(length)
    prefix_rows = int(inputs["mask_t"].sum())
    if prefix_rows != audio_prefix_length(n):
        raise AssertionError(f"P {prefix_rows} != host.audio_prefix_length {audio_prefix_length(n)}")
    return inputs | {"sec": sec, "frames": frames, "prefix_rows": prefix_rows, "steps": shapes["T"]}


def audio_prefix(run_fn, audio, sec: int | None = None, numpy_mel: bool = False) -> np.ndarray:
    """The clip's prefix [P, 1024]: run_fn(inputs) -> the graph's prefix [1, T, 1024] (inputs = the five arrays by name;
    run_fn picks the bundle of inputs' bucket), its first P rows (Audio.forward's x[:, :lengths])."""
    inputs = audio_inputs(audio, sec, numpy_mel)
    out = np.asarray(run_fn({k: inputs[k] for k in AUDIO_INPUT_NAMES}), dtype=np.float32)
    return out.reshape(inputs["steps"], -1)[:inputs["prefix_rows"]]


def audio_request_rows(tok, state: Any, questions, audio, config: dict = CONFIG) -> list[Row]:
    """request_rows for an audio request: P = audio_prefix_length of the clip's samples (the prefix the audio graph
    returns has exactly P rows), max_len = min(15360, 16384 - P), state None -> {}."""
    return request_rows(tok, state, questions, "audio", audio_prefix_length(len(audio)), config)
