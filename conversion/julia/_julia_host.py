"""The host half of the graph contract, in NumPy — the algorithm the Swift host ports.

Step for step the publisher's own code at the pinned revision (Apache-2.0):

  build_row      julia/data.py `sequence()` (with `validate_row`): the row
                 [CLS] tok("{type} question: {question}") [SEP] ([MASK] tok(" " + option)[:48])... [SEP] tok(state) [SEP]
                 CLS = <bos> 2, SEP = <eos> 1, MASK = <mask> 4, PAD 0; every piece tokenized alone without
                 special tokens; a dict / list state is json.dumps(state, ensure_ascii=False); strict
                 encoding refuses any cut, and any text holding the literal "<mask>", instead of cutting
                 (non-strict, the publisher's builder turns "<mask>" into a space and cuts)
  render_named   julia/typed.py `predict_typed`: a named question's options are its criteria text as given —
                 choice = the mapping's descriptions in order (the answer is the caller's id), score = the
                 rubric in order, noul = [criteria["false"], criteria["true"]] or the literal ["false", "true"]
  answer         julia/typed.py: softmax of the raw marker logits at T = 1 (no calibration exists:
                 inference-policy.json `calibration: null`, the temperature buffer is [1, 1, 1] and unused);
                 choice = the argmax id, score = the expected rubric index sum(i * p_i) (not rounded),
                 noul = p[true]; choice and score also give max_probability

This is the contract where Julia differs from laya: laya renders options as `label: description`,
`level i: …` and `false: …` / `true: …`; Julia feeds the descriptions alone. A laya host gives Julia
different rows, and different answers.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

QTYPES = ("choice", "score", "noul")
CLS_ID, SEP_ID, PAD_ID, MASK_ID = 2, 1, 0, 4
MASK_TEXT = "<mask>"
OPTION_TOKENS = 48
MIN_OPTIONS, MAX_OPTIONS = 2, 20


class Tokenizer:
    """The checkpoint's tokenizer.json through the `tokenizers` library, one piece at a time."""

    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer as _Tokenizer

        self._tokenizer = _Tokenizer.from_file(str(path))
        if self._tokenizer.token_to_id(MASK_TEXT) != MASK_ID:
            raise ValueError("tokenizer.json does not map <mask> to 4")

    def __call__(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids if text else []


def validate_request(request: dict) -> None:
    """julia/data.py validate_row, for one request."""
    if not isinstance(request.get("state"), (str, dict, list)) or not isinstance(request.get("question"), str):
        raise ValueError("state must be text/JSON and question must be text")
    options = request.get("options")
    if (not isinstance(options, list) or not MIN_OPTIONS <= len(options) <= MAX_OPTIONS
            or not all(isinstance(x, str) and x for x in options)):
        raise ValueError("options must contain 2–20 nonempty rendered descriptions")
    if request.get("type", "choice") not in QTYPES:
        raise ValueError("type must be choice, score, or noul")
    if request.get("type") == "noul" and len(options) != 2:
        raise ValueError("noul options must be ordered [false, true]")


def build_row(tokenizer: Tokenizer, request: dict, max_length: int, head_length: int, strict: bool = True) -> dict:
    """julia/data.py sequence(): ids, marker positions, question type index, whether the state was cut."""
    validate_request(request)
    if head_length + 4 >= max_length:
        raise ValueError("max_length must leave room beyond the question head")
    state = request["state"] if isinstance(request["state"], str) else json.dumps(request["state"], ensure_ascii=False)
    if strict and any(MASK_TEXT in text for text in [state, request["question"], *request["options"]]):
        raise ValueError("Reserved model marker in request")

    def clean(text: str) -> str:
        return text.replace(MASK_TEXT, " ")

    kind = request.get("type", "choice")
    head = tokenizer(f"{kind} question: {clean(request['question'])}")
    option_ids = [tokenizer(" " + clean(x)) for x in request["options"]]
    if strict and any(len(x) > OPTION_TOKENS for x in option_ids):
        raise ValueError("Option exceeds 48-token model contract")
    options = [[MASK_ID] + x[:OPTION_TOKENS] for x in option_ids]
    budget = head_length - sum(map(len, options))
    if budget < 16:
        per_option = max(4, (head_length - 16) // len(options))
        options = [x[:per_option] for x in options]
        budget = head_length - sum(map(len, options))
    if strict and (len(head) > budget or any(len(x) != len(y) + 1 for x, y in zip(options, option_ids))):
        raise ValueError("Question/options exceed lossless head budget")
    ids = [CLS_ID] + head[:max(8, budget)] + [SEP_ID]
    markers = []
    for option in options:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(SEP_ID)
    state_ids = tokenizer(clean(state))
    room = max_length - len(ids) - 1
    if room < 1:
        raise ValueError("Question/options exceed sequence budget; shorten descriptions")
    if strict and len(state_ids) > room:
        raise ValueError("Game state exceeds lossless context budget")
    return {"ids": ids + state_ids[:room] + [SEP_ID], "markers": markers, "qtype": QTYPES.index(kind),
            "truncated": len(state_ids) > room}


def graph_inputs(row: dict, window: int) -> dict[str, np.ndarray]:
    """input_ids / attention_mask int32 [1,S] right-padded with PAD 0, qtype_onehot float32 [1,3]."""
    n = len(row["ids"])
    if n > window:
        raise ValueError(f"{n} tokens do not fit the {window}-token window")
    input_ids = np.full((1, window), PAD_ID, dtype=np.int32)
    input_ids[0, :n] = row["ids"]
    attention_mask = np.zeros((1, window), dtype=np.int32)
    attention_mask[0, :n] = 1
    qtype_onehot = np.zeros((1, 3), dtype=np.float32)
    qtype_onehot[0, row["qtype"]] = 1.0
    return {"input_ids": input_ids, "attention_mask": attention_mask, "qtype_onehot": qtype_onehot}


def gather_markers(token_logits, markers) -> list[float]:
    """The K option logits, in option order, from the graph's [1,S] output."""
    flat = np.asarray(token_logits, dtype=np.float32).reshape(-1)
    return [float(flat[m]) for m in markers]


def render_named(question: dict) -> tuple[str, list[str], list[str]]:
    """julia/typed.py: (type, answer keys, option texts) for one named question."""
    kind, criteria = question.get("type"), question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict) or any(not isinstance(k, str) or not k for k in criteria):
            raise ValueError("Choice criteria must map nonempty IDs to descriptions")
        keys, labels = list(criteria), list(criteria.values())
    elif kind == "score":
        if not isinstance(criteria, list):
            raise ValueError("Score requires an ordered rubric")
        labels, keys = list(criteria), [str(i) for i in range(len(criteria))]
    elif kind == "noul":
        keys = ["false", "true"]
        if criteria is None:
            labels = list(keys)
        else:
            if not isinstance(criteria, dict) or set(criteria) != set(keys):
                raise ValueError("Noul criteria must map false and true to descriptions")
            labels = [criteria[key] for key in keys]
    else:
        raise ValueError("Unsupported question type")
    return kind, keys, labels


def softmax(logits) -> list[float]:
    """julia/typed.py's softmax in Python floats (T = 1)."""
    z = [float(x) for x in logits]
    if not all(math.isfinite(x) for x in z):
        raise ValueError("Invalid model scores")
    top = max(z)
    p = [math.exp(x - top) for x in z]
    total = sum(p)
    return [x / total for x in p]


def answer(kind: str, keys: list[str], logits) -> dict:
    """julia/typed.py's answer for one question: full softmax probabilities, no display rounding."""
    if len(logits) != len(keys):
        raise ValueError("Model returned an incorrect answer count")
    p = softmax(logits)
    result = {"type": kind, "probabilities": dict(zip(keys, p))}
    if kind == "choice":
        result["choice"] = keys[max(range(len(p)), key=p.__getitem__)]
    elif kind == "score":
        result["score"] = sum(i * x for i, x in enumerate(p))
    else:
        result["noul"] = p[1]
    if kind != "noul":
        result["max_probability"] = max(p)
    return result
