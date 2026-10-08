#!/usr/bin/env python3
"""d1 host reference: a System One request -> one decoder row per question -> option logits -> the response.

`tokenizers`, NumPy and json only; the provider's code is not imported. This file is the specification a Swift host
copies: every rule below is the provider's code (LiquidAI/d1-3B @ da1fe36a, `prompt.py`, `api.py`, `runner.py`)
written out, and `test_host.py` gates it against that code and transformers' tokenizer on every fixture question
(ids, answer slot, readout groups, option keys, the trunk / branch split, the readout arithmetic) before any Swift.

1. Request (the arguments of `model.system_one(state, questions)`; `api.as_question` on each question)

     {"state": <JSON> | null, "questions": {name: question, ...}}           names in request order
     question = {"type": "noul",   "instructions": str, "criteria"?: {"true"?: str, "false"?: str} | null}
              | {"type": "choice", "instructions": str, "criteria": {label: str | null, ...}}   1+ options
              | {"type": "score",  "instructions": str, "criteria": [str, ...]}               1..10 levels

   `type` defaults to "choice" when absent (`q.get("type", "choice")`); any other value but noul / score is a
   choice. `instructions` is required: the provider indexes `q["instructions"]` and raises KeyError without it.
   The provider checks nothing else; this host accepts exactly the documented shapes above and raises ValueError
   for anything else (a non-string instructions, a choice without options, a score with no levels or more than 10,
   a value of another JSON type), so every request it accepts renders as the provider renders it. More than 10
   levels: the card's range is 2 to 10 and the prompt asks for "a single digit"; the provider's code would read up
   to 1,000 (this tokenizer has every number 0..999 as one token), the host refuses them. A state is any JSON value
   or null (no state block).

2. Text (`prompt.render` with the defaults the API uses: style "json_only", system "none", lead "",
   option_style "desc", no images)

     prefix = BOS + "<|im_start|>user\\n" + state_block + "\\nQUESTION:\\n"   (state null: BOS + "<|im_start|>user\\n")
       BOS = the tokenizer's bos_token string "<|startoftext|>" (`SystemOne.bos`)
       state_block = state + "\\n\\n" for a string; json.dumps(state, ensure_ascii=False, indent=2) + "\\n\\n" otherwise
         (Python's json: ", " is "," + newline + indent, ": " after keys, floats as repr, non-ASCII as is,
         NaN / Infinity spelled so)
     suffix = question_block + "<|im_end|>\\n<|im_start|>assistant\\n"
       choice: instructions + "\\n\\nOptions:\\n" + "\\n".join(code + " " + (desc or label.replace("_", " ")))
               + "\\n\\nReply with the option code only."        (an empty or null desc falls back to the label)
       noul:   instructions + ("\\nYes: " + str(criteria.get("true")) + "\\nNo: " + str(criteria.get("false"))
               when criteria is a non-empty object; a missing side prints "None") + "\\n\\nReply with yes or no only."
       score:  instructions + "\\n\\n" + "\\n".join(str(i) + " " + level) + "\\n\\nReply with a single digit 0-" + str(K-1)
               + " only."
     row text = prefix + suffix

3. Option codes and readout groups (`prompt.option_codes`, `aliases`, `readout_ids`)

     codes = the labels themselves when every label (str.strip()) is one alphabetic character (Python isalpha);
             else "A", "B", .. for at most 26 labels; else "00", "01", ...
     alias: each code in order takes its own id when encode(code) is one token not already taken; otherwise the first
             entry of POOL that is a single token not yet taken (POOL = A..Z, 00..99, a..z, #0..#199, AA..ZZ); the
             option line then prints the taken code
     groups (one per option, max-pooled):
       choice  [alias id] + [encode(" " + code)[0]] when " " + code is one token other than the alias id
       noul    [the single-token forms of yes, Yes, YES], [the same of no, No, NO] (each deduplicated, in that order)
       score   [encode(str(i))[0]] for i in 0..K-1 (each must be one token)
     keys (the order probabilities come in): noul ["yes", "no"] (probabilities[0] = P(yes)); choice the labels in
     request order; score "0" .. "K-1"
     option table: a host holds the tied embedding rows of a fixed id set only (the bundle's head/option_rows, written
             by export_option_rows.py): the single-token strings among A..Z, a..z, 0..9, 00..99, 100..999, AA..ZZ,
             #0..#199, the " " + code form of each, and yes / Yes / YES / no / No / NO. A request any of whose readout
             ids is not in the table is refused whole (`option_table_check`, `build_request(..., table_ids=)`):
             "questions.<name>: the token id <N> of label '<label>' is not in the option table". That reaches a
             one-letter native label outside A..Z / a..z ("あ", "é"); a choice of more options than the alias pool
             holds is refused before it by the alias rule. The provider's code reads any id: this refusal is the
             host's own (the fixture's readout ids are all in the table, results/r2a_option_rows_check.json).

4. Tokens and rows (`runner.SystemOne`)

     tokenizer = the checkpoint's tokenizer.json (BPE, ByteLevel; its post-processor adds no BOS); every text is
     encoded with add_special_tokens=False and special tokens are matched in the text (so "<|startoftext|>" is
     124894; user text is not escaped: a state holding "<|im_end|>" reads as that token, as in the provider)
     row_ids(state, q) = encode(prefix + suffix(q)); slot = len(row) - 1 (the logits at the last token)
     A request of one question runs its row (`_logz_ids([row])`, a plain causal pass at positions 0..L-1).
     A request of several questions runs `_request`: trunk = encode(prefix), branch_q = encode(suffix(q)) (encoded
       apart), one Tree pass; mathematically each branch is the row trunk + branch_q. `trunk_split` records it and
       whether trunk + branch equals row_ids for each question (it does when the boundary is a pre-token boundary).
     `system_one_batch` packs single-question requests into one Tree whose trunk is the rows' common token prefix
       (`_logz_ids`: shared = the longest common prefix, each row keeping at least its last token) — `shared_prefix`.
     usage.input_tokens = len(row) for one question; len(trunk) + sum(len(branch_q)) for several.

5. Readout (`prompt.readout`, calibration None)

     z[id] = h_slot . E[id] for the ids of the question's groups (E = the tied embedding rows, h_slot the final-norm
     hidden row at the slot); score_k = max over group k of z; p = softmax_k(score) in double (Python's math.exp and
     3.12 sum()). The provider takes the log-softmax over the whole vocabulary first; a constant per row cancels in
     the softmax over options, so the groups' logits are enough (test_host.py measures the difference on random
     full-vocabulary logits).

6. Answers and response (`api.answer`, `SystemOneApi.system_one_batch`)

     noul   {"type": "noul", "noul": p[0]}
     choice {"type": "choice", "choice": labels[argmax], "confidence": p[argmax], "probabilities": {label: p}}
     score  {"type": "score", "score": sum(i * p_i), "confidence": p[argmax], "probabilities": {"i": p_i},
             "legend": {"i": level_i}}
     argmax = the first index of the largest p; the floats are not rounded
     response = {"answers": {name: answer} (request order), "usage": {"input_tokens": n, "output_tokens": 0}}

7. The graph's rows (`lfm2_d1_decoder.py`)

     a row of T ids runs as ceil(T / S) calls of S ids from zero states, call c with position_ids 0 .. cS + S - 1;
     the last call is padded with <|pad|> (124893) and its padded rows dropped; the row limit is
     ceil(T / S) * S <= max_ctx - 1 (the position axis' upper bound: T <= 4080 at S = 16, max_ctx 4096).
     The static form (`lfm2_d1_static.py`, metadata `language.contract.static`): call c gets position_ids
     cS .. cS + S - 1 (its own S positions only), the states have no dynamic axis, and the row limit is
     ceil(T / S) * S <= max_ctx (the KV cache's slots).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

BOS = "<|startoftext|>"
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
PAD_TOKEN, PAD_ID = "<|pad|>", 124893
QUESTION_TYPES = ("noul", "choice", "score")
YES_FORMS = ("yes", "Yes", "YES")
NO_FORMS = ("no", "No", "NO")
NOUL_KEYS = ("yes", "no")
MAX_SCORE_LEVELS = 10          # the card's 2..10 (the prompt asks for a single digit); the provider's code reads 0..999
FALLBACK_POOL = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [f"{i:02d}" for i in range(100)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [f"#{i}" for i in range(200)]
    + [chr(a) + chr(b) for a in range(ord("A"), ord("Z") + 1) for b in range(ord("A"), ord("Z") + 1)]
)
KEV_NOUL_GOLD = {"true": "yes", "false": "no"}   # fixture gold keys (Kev's) -> d1 keys


# --------------------------------------------------------------------------- #
# 1. Request
# --------------------------------------------------------------------------- #
def _is_json(v: Any) -> bool:
    if v is None or isinstance(v, (str, bool, int, float)):
        return True
    if isinstance(v, list):
        return all(_is_json(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _is_json(x) for k, x in v.items())
    return False


def validate_question(name: str, q: Any) -> dict:
    """One question -> {"type", "instructions", "criteria"} (criteria normalized to the provider's view: noul
    dict | None, choice dict in request order, score list). ValueError outside the documented shapes."""
    if not isinstance(q, dict):
        raise ValueError(f"questions.{name}: must be an object")
    kind = q.get("type", "choice")
    if not isinstance(kind, str):
        raise ValueError(f"questions.{name}.type: must be a string")
    kind = kind if kind in ("noul", "score") else "choice"     # as_question: anything else is a choice
    if "instructions" not in q:
        raise ValueError(f"questions.{name}.instructions: field required")
    instructions = q["instructions"]
    if not isinstance(instructions, str):
        raise ValueError(f"questions.{name}.instructions: must be a string")
    criteria = q.get("criteria")
    if kind == "noul":
        if criteria is not None:
            if not isinstance(criteria, dict):
                raise ValueError(f"questions.{name}.criteria: a noul question takes an object or null")
            for side in ("true", "false"):
                if side in criteria and not (criteria[side] is None or isinstance(criteria[side], str)):
                    raise ValueError(f"questions.{name}.criteria.{side}: must be a string or null")
            if not all(_is_json(v) for v in criteria.values()):
                raise ValueError(f"questions.{name}.criteria: not JSON")
    elif kind == "choice":
        if "criteria" not in q:
            raise ValueError(f"questions.{name}.criteria: field required")
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError(f"questions.{name}.criteria: a choice question takes an object of 1+ options")
        for label, desc in criteria.items():
            if not (desc is None or isinstance(desc, str)):
                raise ValueError(f"questions.{name}.criteria.{label}: a description is a string or null")
    else:
        if "criteria" not in q:
            raise ValueError(f"questions.{name}.criteria: field required")
        if not isinstance(criteria, list) or not 1 <= len(criteria) <= MAX_SCORE_LEVELS:
            raise ValueError(f"questions.{name}.criteria: a score question takes a list of 1..{MAX_SCORE_LEVELS} levels")
        if not all(isinstance(x, str) for x in criteria):
            raise ValueError(f"questions.{name}.criteria: every level is a string")
    return {"type": kind, "instructions": instructions, "criteria": criteria}


def validate_request(request: Any) -> dict:
    """{"state", "questions"} -> {"state", "questions": [(name, question)]}; ValueError for anything the host
    does not accept (see section 1)."""
    if not isinstance(request, dict):
        raise ValueError("the request must be a JSON object")
    if "state" not in request:
        raise ValueError("state: field required (null for no state)")
    if not _is_json(request["state"]):
        raise ValueError("state: not a JSON value")
    questions = request.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions: must be an object of name -> question with at least one entry")
    return {"state": request["state"], "questions": [(n, validate_question(n, q)) for n, q in questions.items()]}


# --------------------------------------------------------------------------- #
# 2. Text
# --------------------------------------------------------------------------- #
def state_block(state: Any) -> str:
    """prompt.state_block(state, "json_only")."""
    if isinstance(state, str):
        return f"{state}\n\n"
    return json.dumps(state, ensure_ascii=False, indent=2) + "\n\n"


def prefix_text(state: Any, bos: str = BOS) -> str:
    """prompt.prefix_text(tok, state, bos, "json_only", "none", images="")."""
    body = "" if state is None else f"{state_block(state)}\nQUESTION:\n"
    return f"{bos}{IM_START}user\n{body}"


def option_codes(labels: list[str]) -> list[str]:
    """prompt.option_codes."""
    labs = [str(x).strip() for x in labels]
    if labs and all(len(k) == 1 and k.isalpha() for k in labs):
        return labs
    if len(labs) <= 26:
        return [chr(ord("A") + i) for i in range(len(labs))]
    return [f"{i:02d}" for i in range(len(labs))]


def question_block(q: dict, codes: list[str] | None = None) -> str:
    """prompt.question_block(tok, q, "desc"); `codes` = the aliases' codes (choice only)."""
    kind, instr, crit = q["type"], q["instructions"], q["criteria"]
    if kind == "choice":
        labels = list(crit)
        lines = "\n".join(f"{codes[i]} {crit[lab] or lab.replace('_', ' ')}" for i, lab in enumerate(labels))
        return f"{instr}\n\nOptions:\n{lines}\n\nReply with the option code only."
    if kind == "noul":
        extra = ""
        if crit:
            extra = f"\nYes: {crit.get('true')}\nNo: {crit.get('false')}"
        return f"{instr}{extra}\n\nReply with yes or no only."
    legend = "\n".join(f"{i} {name}" for i, name in enumerate(crit))
    return f"{instr}\n\n{legend}\n\nReply with a single digit 0-{len(crit) - 1} only."


def suffix_text(q: dict, codes: list[str] | None = None) -> str:
    """prompt.suffix_text(tok, q, lead="", "desc")."""
    return f"{question_block(q, codes)}{IM_END}\n{IM_START}assistant\n"


# --------------------------------------------------------------------------- #
# 3. Tokens, aliases, groups
# --------------------------------------------------------------------------- #
def load_tokenizer(path: str | Path):
    """The checkpoint's tokenizer.json as a `tokenizers.Tokenizer`."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


def token_ids(tok, text: str) -> list[int]:
    """`text` -> ids with add_special_tokens=False, for a `tokenizers.Tokenizer` or a transformers tokenizer."""
    if hasattr(tok, "encode_batch") and hasattr(tok, "token_to_id"):        # tokenizers.Tokenizer
        return list(tok.encode(text, add_special_tokens=False).ids)
    return list(tok.encode(text, add_special_tokens=False))


def single_token(tok, text: str) -> int | None:
    ids = token_ids(tok, text)
    return ids[0] if len(ids) == 1 else None


def aliases(tok, labels: list[str]) -> list[tuple[str, int]]:
    """prompt.aliases: a distinct single-token code per label, [(code, id)]."""
    used: set[int] = set()
    out: list[tuple[str, int]] = []

    def take(raw: str) -> bool:
        i = single_token(tok, raw)
        if i is None or i in used:
            return False
        out.append((raw, i))
        used.add(i)
        return True

    codes = option_codes(labels)
    for code in codes:
        if take(code):
            continue
        if not any(take(raw) for raw in FALLBACK_POOL):
            raise ValueError(f"no single-token alias left for {len(codes)} options")
    return out


def _forms(tok, texts) -> list[int]:
    """prompt._ids: the single-token forms of `texts`, deduplicated in order."""
    out: list[int] = []
    for t in texts:
        i = single_token(tok, t)
        if i is not None and i not in out:
            out.append(i)
    return out


def readout_groups(tok, q: dict, alias: list[tuple[str, int]] | None = None) -> list[list[int]]:
    """prompt.readout_ids: one id group per option, max-pooled."""
    if q["type"] == "noul":
        yes, no = _forms(tok, YES_FORMS), _forms(tok, NO_FORMS)
        if not yes or not no:
            raise ValueError("tokenizer has no single-token yes/no")
        return [yes, no]
    if q["type"] == "score":
        groups = [_forms(tok, [str(i)]) for i in range(len(q["criteria"]))]
        if any(not g for g in groups):
            raise ValueError(f"score with {len(q['criteria'])} levels needs single-token digits")
        return groups
    groups = []
    for code, tid in (alias if alias is not None else aliases(tok, list(q["criteria"]))):
        groups.append([tid] + [i for i in _forms(tok, [f" {code}"]) if i != tid])
    return groups


def option_keys(q: dict) -> list[str]:
    """The keys the probabilities come in: noul yes / no, choice the labels, score "0".."K-1"."""
    if q["type"] == "noul":
        return list(NOUL_KEYS)
    if q["type"] == "choice":
        return list(q["criteria"])
    return [str(i) for i in range(len(q["criteria"]))]


def gold_key(q: dict, gold: str | None) -> str | None:
    """A fixture gold key -> the d1 key (Kev's noul "true" / "false" -> "yes" / "no"; others unchanged)."""
    if gold is None:
        return None
    return KEV_NOUL_GOLD[gold] if q["type"] == "noul" else gold


def build_question(tok, state: Any, name: str, q: dict, bos: str = BOS) -> dict:
    """One validated question -> its row: text, ids, slot, aliases, groups, keys."""
    alias = aliases(tok, list(q["criteria"])) if q["type"] == "choice" else None
    codes = [c for c, _ in alias] if alias is not None else None
    prefix, suffix = prefix_text(state, bos), suffix_text(q, codes)
    ids = token_ids(tok, prefix + suffix)
    return {"name": name, "type": q["type"], "text": prefix + suffix, "row_ids": ids, "row_len": len(ids),
            "slot": len(ids) - 1, "codes": codes, "alias_ids": [i for _, i in alias] if alias is not None else None,
            "groups": readout_groups(tok, q, alias), "keys": option_keys(q), "suffix": suffix}


def shared_prefix(rows: list[list[int]]) -> int:
    """runner._logz_ids: the trunk length of a Tree over these rows (each row keeps at least its last token)."""
    if len(rows) < 2:
        return 0
    shared = 0
    while shared < min(map(len, rows)) - 1 and len({r[shared] for r in rows}) == 1:
        shared += 1
    return shared


def trunk_split(tok, state: Any, questions: list[dict], bos: str = BOS) -> dict:
    """runner._request for a text request of 2+ questions: trunk = encode(prefix), branch = encode(suffix) apart."""
    trunk = token_ids(tok, prefix_text(state, bos))
    branches = [token_ids(tok, q["suffix"]) for q in questions]
    return {"trunk_ids": trunk, "trunk_len": len(trunk), "branch_lens": [len(b) for b in branches],
            "branch_ids": branches,
            "equals_row": [trunk + b == q["row_ids"] for b, q in zip(branches, questions)],
            "input_tokens": len(trunk) + sum(len(b) for b in branches)}


def option_table_check(questions: list[dict], table_ids) -> None:
    """Section 3's option table: ValueError for the first readout id outside it (the request is refused whole)."""
    have = {int(i) for i in table_ids}
    for q in questions:
        for key, g in zip(q["keys"], q["groups"]):
            for i in g:
                if int(i) not in have:
                    raise ValueError(f"questions.{q['name']}: the token id {int(i)} of label {key!r} is not in the "
                                     "option table")


def build_request(request: Any, tok, bos: str = BOS, table_ids=None) -> dict:
    """request -> {"state", "questions": [build_question ...], "path": "row" | "tree", "trunk": trunk_split | None,
    "shared": _logz_ids' trunk over the rows, "input_tokens"}. With `table_ids` (the bundle's option-row ids) a request
    reading an id outside them is refused (option_table_check)."""
    req = validate_request(request)
    qs = [build_question(tok, req["state"], n, q, bos) for n, q in req["questions"]]
    if table_ids is not None:
        option_table_check(qs, table_ids)
    tree = trunk_split(tok, req["state"], qs, bos) if len(qs) > 1 else None
    return {"state": req["state"], "questions": qs, "validated": req["questions"], "path": "tree" if tree else "row",
            "trunk": tree, "shared": shared_prefix([q["row_ids"] for q in qs]),
            "input_tokens": tree["input_tokens"] if tree else qs[0]["row_len"]}


# --------------------------------------------------------------------------- #
# 5. Readout
# --------------------------------------------------------------------------- #
def py_sum(values) -> float:
    """Python 3.12's sum() of floats: left to right with Neumaier's compensation, added at the end when nonzero and
    finite (3.11's plain sum can differ in the last bit)."""
    it = iter(values)
    try:
        first = next(it)
    except StopIteration:
        return 0
    f = 0 + first
    c = 0.0
    for x in it:
        x = float(x)
        t = f + x
        if abs(f) >= abs(x):
            c += (f - t) + x
        else:
            c += (x - t) + f
        f = t
    if c and math.isfinite(c):
        f += c
    return f


def probs_from_logits(z: dict | np.ndarray, groups: list[list[int]]) -> list[float]:
    """prompt.readout on any per-id scores (logits or log-probabilities): group max, then softmax in double."""
    scores = [max(float(z[i]) for i in g) for g in groups]
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = py_sum(exps)
    return [e / total for e in exps]


def option_logits(h_slot: np.ndarray, rows: np.ndarray, ids: list[int]) -> dict[int, float]:
    """z[id] = h_slot . E[id] in float64 (the fp32 or fp16 values converted exactly); `rows` [n, d] = E[ids]."""
    h = np.asarray(h_slot, np.float64)
    e = np.asarray(rows, np.float64)
    return {int(i): float(v) for i, v in zip(ids, e @ h)}


def readout(h_slot: np.ndarray, rows: np.ndarray, ids: list[int], groups: list[list[int]]) -> list[float]:
    """The answer slot's hidden row and the option rows of the tied table -> the option probabilities."""
    return probs_from_logits(option_logits(h_slot, rows, ids), groups)


def group_ids(groups: list[list[int]]) -> list[int]:
    """Every id the readout reads, in first-seen order (the rows a host gathers)."""
    out: list[int] = []
    for g in groups:
        out += [i for i in g if i not in out]
    return out


# --------------------------------------------------------------------------- #
# 6. Answers and response
# --------------------------------------------------------------------------- #
def answer(q: dict, probs: list[float]) -> dict:
    """api.answer."""
    if q["type"] == "noul":
        return {"type": "noul", "noul": probs[0]}
    best = max(range(len(probs)), key=probs.__getitem__)
    if q["type"] == "choice":
        names = list(q["criteria"])
        return {"type": "choice", "choice": names[best], "confidence": probs[best],
                "probabilities": dict(zip(names, probs))}
    return {"type": "score", "score": py_sum(i * p for i, p in enumerate(probs)), "confidence": probs[best],
            "probabilities": {str(i): p for i, p in enumerate(probs)},
            "legend": {str(i): text for i, text in enumerate(q["criteria"])}}


def response(validated: list[tuple[str, dict]], probs: list[list[float]], input_tokens: int) -> dict:
    """SystemOneApi.system_one's body."""
    return {"answers": {n: answer(q, p) for (n, q), p in zip(validated, probs)},
            "usage": {"input_tokens": int(input_tokens), "output_tokens": 0}}


# --------------------------------------------------------------------------- #
# 7. The graph's rows
# --------------------------------------------------------------------------- #
def chunk_calls(T: int, S: int) -> int:
    return -(-T // S)


def padded_len(T: int, S: int) -> int:
    return chunk_calls(T, S) * S


def graph_context_check(T: int, S: int = 16, max_ctx: int = 4096, static: bool = False) -> None:
    """A row of T ids fits the static-S graph: its padded end <= max_ctx - 1 (the dynamic form's position axis upper
    bound), or <= max_ctx for the static form (`lfm2_d1_static.py`: the KV cache's max_ctx slots)."""
    limit = max_ctx if static else max_ctx - 1
    if padded_len(T, S) > limit:
        raise ValueError(f"a row of {T} tokens runs {padded_len(T, S)} padded positions, over the graph's "
                         f"{limit} (rows of at most {limit // S * S} tokens at S = {S})")


def plan_calls(ids: list[int], S: int, pad_id: int = PAD_ID) -> list[tuple[list[int], int]]:
    """A row as static-S calls: [(the call's S ids, real ids in it)], the last padded with pad_id."""
    out = []
    for c in range(chunk_calls(len(ids), S)):
        piece = ids[c * S:(c + 1) * S]
        out.append((piece + [pad_id] * (S - len(piece)), len(piece)))
    return out
