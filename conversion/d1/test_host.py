#!/usr/bin/env python3
"""Test: host.py rebuilds every fixture question the way the provider's code does, with two tokenizers.

The provider's side is `oracle_d1.dry_run`: every record of `fixtures/records.json` through the provider's own code
(`K/src/d1` = the six .py files of LiquidAI/d1-3B @ da1fe36a, verbatim; transformers' AutoTokenizer on the snapshot)
on a stand-in backbone that records the ids it is handed and returns seeded random fp32 logits. The host's side is
`host.py` with the checkpoint's tokenizer.json through `tokenizers` (what a Swift host reads) and, again, through
transformers' tokenizer. Per question, equal bit for bit or the test fails:

  * text, row ids, slot (= len - 1), option codes, readout groups, option keys;
  * the API path's ids: a one-question request's row as handed to the backbone; a several-question request's Tree
    trunk and branches (`_request` encodes the prefix and each suffix apart) and usage.input_tokens;
  * the answers: host.answer / host.response on the provider's probabilities == api.answer / the provider's body
    (json.dumps equal);
  * the probabilities: host.probs_from_logits on the groups' logits alone (the stand-in's) against the provider's
    readout of its full-vocabulary log-softmax: max |dp| <= 1e-6.

Also: the tokenizer contract (`results/tokenizer_check.json`: bos / eos / pad ids, encode("<|startoftext|>") ==
[124894], the single-token table, the aliases for 2..30 positional labels, the noul and score groups, the image-related
added tokens); the readout's equivalence on random logits (10 vectors over 128,000 ids per distinct group set, half
of them with the options at the top of the vocabulary: the provider's readout of `z - logsumexp(z)` in fp32 against
the host's softmax of the groups' raw logits, max |dp| <= 1e-6); a table of valid and invalid requests (the host
accepts exactly the documented shapes, renders them as the provider does, and refuses the rest; what the provider does
with each is recorded); negative controls (one word of one question's instructions changed: the ids go red; the readout
with each group's first form only: the probabilities go red); the fixture's lengths (`results/fixture_lengths.json`)
and every row (`results/render_ids.json`).

    cd conversion/d1
    $ZOO_WORK_ROOT/_d1_3b/venv-oracle/bin/python test_host.py      # transformers >= 5.14 (the tokenizer's class)

-> $ZOO_WORK_ROOT/_d1_3b/results/{test_host.json, tokenizer_check.json, render_ids.json, fixture_lengths.json}
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import host  # noqa: E402
import oracle_d1  # noqa: E402
from _paths import work_path  # noqa: E402

LANE = work_path("_d1_3b")
EQUIV_BAR = 1e-6
EQUIV_VECTORS = 10
NEG_RECORD, NEG_QUESTION, NEG_WORD = "card_refund", "team", "zqxvortmund"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


# --------------------------------------------------------------------------- 1. the tokenizer contract
def tokenizer_contract(snap: Path, tf, rs) -> dict:
    prompt, _ = oracle_d1.provider()
    tj = json.loads((snap / "tokenizer.json").read_text())
    out: dict = {"snapshot": str(snap), "tokenizer_json_sha256": sha256_file(snap / "tokenizer.json"),
                 "tokenizer_config": json.loads((snap / "tokenizer_config.json").read_text()),
                 "transformers": {"class": type(tf).__name__, "len": len(tf), "bos_token": tf.bos_token,
                                  "bos_token_id": tf.bos_token_id, "eos_token": tf.eos_token, "eos_token_id": tf.eos_token_id,
                                  "pad_token": tf.pad_token, "pad_token_id": tf.pad_token_id,
                                  "padding_side": getattr(tf, "padding_side", None)},
                 "tokenizers": {"vocab_size_with_added": rs.get_vocab_size(with_added_tokens=True),
                                "vocab_size_without_added": rs.get_vocab_size(with_added_tokens=False)},
                 "pipeline": {"normalizer": tj.get("normalizer"), "pre_tokenizer": tj.get("pre_tokenizer"),
                              "post_processor": tj.get("post_processor"), "model": tj["model"]["type"],
                              "bpe_vocab": len(tj["model"]["vocab"]), "merges": len(tj["model"].get("merges", []))},
                 "config_vocab_size": 128000}
    ids = {t: rs.token_to_id(t) for t in ("<|startoftext|>", "<|im_end|>", "<|pad|>", "<|im_start|>", "<image>",
                                           "<|image_start|>", "<|image_end|>", "<|endoftext|>")}
    out["special_ids"] = ids
    assert tf.bos_token_id == 124894 and tf.eos_token_id == 124900 and tf.pad_token_id == 124893, out["transformers"]
    assert ids["<|startoftext|>"] == 124894 and ids["<|im_end|>"] == 124900 and ids["<|pad|>"] == 124893
    out["image_token"] = {"id": 124907, "transformers": tf.convert_ids_to_tokens(124907), "tokenizers": rs.id_to_token(124907)}
    bos_tf = tf.encode("<|startoftext|>", add_special_tokens=False)
    bos_rs = rs.encode("<|startoftext|>", add_special_tokens=False).ids
    out["bos_encode"] = {"transformers": bos_tf, "tokenizers": bos_rs, "equals_[124894]": bos_tf == bos_rs == [124894]}
    assert bos_tf == bos_rs == [124894], "encode('<|startoftext|>') != [124894]: the ids contract changes"
    out["add_special_tokens_true_on_x"] = {"transformers": tf.encode("x", add_special_tokens=True),
                                           "tokenizers": rs.encode("x", add_special_tokens=True).ids,
                                           "plain": rs.encode("x", add_special_tokens=False).ids,
                                           "adds_bos": tf.encode("x", add_special_tokens=True)[:1] == [124894]}
    strings = ([chr(c) for c in range(65, 91)] + [" " + chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)]
               + [" " + chr(c) for c in range(97, 123)] + [str(i) for i in range(10)] + [" " + str(i) for i in range(10)]
               + [f"{i:02d}" for i in range(100)] + [f"#{i}" for i in range(10)] + ["10", "11"]
               + list(host.YES_FORMS) + list(host.NO_FORMS))
    table, disagree = {}, []
    for s in strings:
        a, b = tf.encode(s, add_special_tokens=False), rs.encode(s, add_special_tokens=False).ids
        if a != b:
            disagree.append(s)
        table[s] = {"ids": b, "single": b[0] if len(b) == 1 else None}
    assert not disagree, disagree
    out["single_token_table"] = table

    def all_single(xs):
        return sum(table[x]["single"] is not None for x in xs)
    out["single_token_summary"] = {
        "A..Z": f"{all_single([chr(c) for c in range(65, 91)])}/26", "' A'..' Z'": f"{all_single([' ' + chr(c) for c in range(65, 91)])}/26",
        "a..z": f"{all_single([chr(c) for c in range(97, 123)])}/26", "' a'..' z'": f"{all_single([' ' + chr(c) for c in range(97, 123)])}/26",
        "0..9": f"{all_single([str(i) for i in range(10)])}/10", "' 0'..' 9'": f"{all_single([' ' + str(i) for i in range(10)])}/10",
        "00..99": f"{all_single([f'{i:02d}' for i in range(100)])}/100", "#0..#9": f"{all_single([f'#{i}' for i in range(10)])}/10",
        "yes/Yes/YES/no/No/NO": f"{all_single(list(host.YES_FORMS) + list(host.NO_FORMS))}/6",
        "10": table["10"]["single"] is not None}
    al = {}
    for k in range(2, 31):
        labels = [f"opt{i}" for i in range(k)]
        mine, theirs = host.aliases(rs, labels), prompt.aliases(tf, labels)
        assert mine == theirs, (k, mine, theirs)
        want = host.option_codes(labels)
        al[str(k)] = {"codes": [c for c, _ in mine], "ids": [i for _, i in mine],
                      "fallback": [c for (c, _), w in zip(mine, want) if c != w]}
    out["aliases_positional"] = al
    native = {}
    for labels in (["a", "b", "c", "d"], ["A", "B", "C"], ["a", "B"], [" a", "b "], ["x", "y", "x"]):
        mine, theirs = host.aliases(rs, labels), prompt.aliases(tf, labels)
        assert mine == theirs, (labels, mine, theirs)
        native[json.dumps(labels)] = {"codes": [c for c, _ in mine], "ids": [i for _, i in mine]}
    out["aliases_native"] = native
    noul_q = {"type": "noul", "instructions": "x", "criteria": None}
    out["noul_groups"] = {"host": host.readout_groups(rs, noul_q),
                          "provider": prompt.readout_ids(tf, prompt.as_question(noul_q))}
    assert out["noul_groups"]["host"] == out["noul_groups"]["provider"] and all(out["noul_groups"]["host"])
    sc = {}
    for k in range(2, 12):
        q = {"type": "score", "instructions": "x", "criteria": [f"l{i}" for i in range(k)]}
        try:
            theirs = prompt.readout_ids(tf, prompt.as_question(q))
        except Exception as e:  # noqa: BLE001
            theirs = f"{type(e).__name__}: {e}"
        try:
            mine = host.readout_groups(rs, q)
        except ValueError as e:
            mine = f"ValueError: {e}"
        sc[str(k)] = {"provider": theirs, "host": mine}
        if k <= 10:
            assert mine == theirs and all(theirs), (k, mine, theirs)
    out["score_groups"] = sc
    names = ("image", "img", "thumbnail", "row", "col")
    out["image_added_tokens"] = [{"id": a["id"], "content": a["content"], "special": a["special"]}
                                 for a in tj["added_tokens"] if any(n in a["content"].lower() for n in names)]
    out["image_added_tokens_count"] = len(out["image_added_tokens"])
    out["added_tokens"] = {"count": len(tj["added_tokens"]), "special": sum(a["special"] for a in tj["added_tokens"]),
                           "first": tj["added_tokens"][0]["id"], "last": tj["added_tokens"][-1]["id"]}
    return out


# --------------------------------------------------------------------------- 3. host vs provider, per record
def host_side(r: dict, tok) -> dict:
    """The host's rows for one record: built per question (a refused question records the host's refusal), and the
    request's Tree split when every question is accepted."""
    state, qdict = r["request"]["state"], r["request"]["questions"]
    out = {"questions": {}, "refused": {}, "validated": []}
    for name, qd in qdict.items():
        try:
            q = host.validate_question(name, qd)
        except ValueError as e:
            out["refused"][name] = str(e)
            continue
        out["validated"].append((name, q))
        out["questions"][name] = host.build_question(tok, state, name, q)
    if not out["refused"]:
        b = host.build_request(r["request"], tok)
        out["request"] = b
    return out


def compare_record(e: dict, r: dict, hs: dict) -> tuple[list[str], dict]:
    bad, info = [], {}
    if set(hs["refused"]) != {x["name"] for x in e["refused"]}:
        bad.append(f"refused: host {sorted(hs['refused'])} provider {[x['name'] for x in e['refused']]}")
    for pq in e["questions"]:
        hq = hs["questions"].get(pq["name"])
        if hq is None:
            bad.append(f"{pq['name']}: host has no row")
            continue
        for k_host, k_prov in (("text", "text"), ("row_ids", "row_ids"), ("slot", "slot"), ("codes", "codes"),
                               ("groups", "groups"), ("keys", "keys")):
            if hq[k_host] != pq[k_prov]:
                bad.append(f"{pq['name']}: {k_host} differs")
    if e["refused"]:
        info["api"] = {"provider_error": e["api"].get("error"), "host_error": "; ".join(hs["refused"].values())}
        return bad, info
    b = hs["request"]
    calls = e["calls"]
    if b["path"] == "row":
        if not (len(calls) == 1 and calls[0]["kind"] == "row" and calls[0]["ids"] == b["questions"][0]["row_ids"]):
            bad.append("api row ids differ")
    else:
        if not (len(calls) == 1 and calls[0]["kind"] == "tree" and calls[0]["trunk"] == b["trunk"]["trunk_ids"]
                and calls[0]["branches"] == b["trunk"]["branch_ids"]):
            bad.append("api tree trunk / branches differ")
        info["trunk_plus_branch_equals_row"] = b["trunk"]["equals_row"]
    if e["api"]["input_tokens"] != b["input_tokens"]:
        bad.append(f"input_tokens {b['input_tokens']} != {e['api']['input_tokens']}")
    info["path"] = b["path"]
    return bad, info


def answers_check(e: dict, hs: dict) -> dict:
    """host.answer / host.response on the provider's probabilities against the provider's (json.dumps equal)."""
    val = dict(hs["validated"])
    per_q = all(json.dumps(host.answer(val[q["name"]], q["probs"])) == json.dumps(q["answer"]) for q in e["questions"])
    body = None
    if not e["refused"]:
        mine = host.response(hs["validated"], e["api"]["probs"], hs["request"]["input_tokens"])
        body = json.dumps(mine) == json.dumps(e["api"]["response"])
    return {"answers_equal": per_q, "body_equal": body}


def probs_check(e: dict, hs: dict, seed: int) -> float:
    """The host's readout on the stand-in's logits of each row (the groups' entries only) against the provider's."""
    worst = 0.0
    for q in e["questions"]:
        hq = hs["questions"][q["name"]]
        z = oracle_d1.standin_logits(hq["row_ids"], seed).numpy()
        mine = host.probs_from_logits(z, hq["groups"])
        worst = max(worst, max(abs(a - b) for a, b in zip(mine, q["probs"])))
    return worst


# --------------------------------------------------------------------------- 4. readout equivalence on random logits
def readout_equivalence(entries: list[dict], records: dict, tf, seed: int) -> dict:
    import torch

    prompt, _ = oracle_d1.provider()
    sets = {}
    for e in entries:
        for q in e["questions"]:
            key = json.dumps(q["groups"])
            if key not in sets:
                qd = records[e["id"]]["request"]["questions"][q["name"]]
                sets[key] = (prompt.as_question(qd), q["groups"], f"{e['id']}/{q['name']}")
    g = torch.Generator().manual_seed(seed + 7)
    worst = {"plain": 0.0, "options_on_top": 0.0}
    worst_logz_input = 0.0
    rows = []
    for key, (qobj, groups, where) in sets.items():
        ids = host.group_ids(groups)
        for v in range(EQUIV_VECTORS):
            z = torch.randn(oracle_d1.VOCAB, generator=g) * 3.0
            kind = "plain" if v < EQUIV_VECTORS // 2 else "options_on_top"
            if kind == "options_on_top":
                top = float(z.max())
                z[torch.tensor(ids)] = top + (torch.rand(len(ids), generator=g) * 3.0 - 1.5)
            logz = z - torch.logsumexp(z, dim=-1)          # runner._logz_ids' log-softmax, fp32
            theirs = prompt.readout(tf, qobj, logz)
            mine = host.probs_from_logits(z.numpy(), groups)
            same_input = host.probs_from_logits(logz.numpy(), groups)
            d = max(abs(a - b) for a, b in zip(mine, theirs))
            worst[kind] = max(worst[kind], d)
            worst_logz_input = max(worst_logz_input, max(abs(a - b) for a, b in zip(same_input, theirs)))
            rows.append({"group_set": where, "kind": kind, "max_abs_dp": d})
    total = max(worst.values())
    return {"group_sets": len(sets), "vectors_per_set": EQUIV_VECTORS, "vocab": oracle_d1.VOCAB,
            "logits": "N(0, 3^2) fp32; options_on_top: the groups' ids set to max(z) + U(-1.5, 1.5)",
            "provider": "prompt.readout(tok, q, z - logsumexp(z)) (runner._logz_ids' fp32 log-softmax)",
            "host": "host.probs_from_logits(z[group ids]) (raw logits, double)",
            "max_abs_dp": total, "max_abs_dp_by_kind": worst, "bar": EQUIV_BAR, "pass": total <= EQUIV_BAR,
            "host_on_the_same_logz_max_abs_dp": worst_logz_input,
            "worst": sorted(rows, key=lambda x: -x["max_abs_dp"])[:5]}


# --------------------------------------------------------------------------- 5. negative controls
def negative_ids(records: dict, entries: dict, tok) -> dict:
    r = copy.deepcopy(records[NEG_RECORD])
    q = r["request"]["questions"][NEG_QUESTION]
    words = q["instructions"].split(" ")
    words[1] = NEG_WORD
    q["instructions"] = " ".join(words)
    hs = host_side(r, tok)
    bad, _ = compare_record(entries[NEG_RECORD], r, hs)
    return {"record": NEG_RECORD, "question": NEG_QUESTION, "changed_instructions": q["instructions"], "red": bool(bad),
            "messages": bad[:4]}


def negative_first_form(entries: list[dict], hs_all: dict, seed: int) -> dict:
    """The readout with each group's first form only (no max-pool over 'A' / ' A'): must move some question."""
    worst, moved = 0.0, 0
    for e in entries:
        for q in e["questions"]:
            hq = hs_all[e["id"]]["questions"][q["name"]]
            z = oracle_d1.standin_logits(hq["row_ids"], seed).numpy()
            mine = host.probs_from_logits(z, [g[:1] for g in hq["groups"]])
            d = max(abs(a - b) for a, b in zip(mine, q["probs"]))
            worst = max(worst, d)
            moved += d > EQUIV_BAR
    return {"what": "host readout with each group's first id only, against the provider's max-pooled readout",
            "questions_moved": moved, "max_abs_dp": worst, "red": moved > 0}


# --------------------------------------------------------------------------- 6. the request table
VALID = {
    "noul_plain": {"state": "s", "questions": {"a": {"type": "noul", "instructions": "Is it?"}}},
    "noul_criteria": {"state": "s", "questions": {"a": {"type": "noul", "instructions": "Is it?",
                                                        "criteria": {"true": "it is", "false": "it is not"}}}},
    "noul_criteria_true_only": {"state": "s", "questions": {"a": {"type": "noul", "instructions": "Is it?",
                                                                  "criteria": {"true": "it is"}}}},
    "noul_criteria_empty": {"state": "s", "questions": {"a": {"type": "noul", "instructions": "Is it?", "criteria": {}}}},
    "noul_criteria_other_key": {"state": "s", "questions": {"a": {"type": "noul", "instructions": "Is it?",
                                                                  "criteria": {"maybe": "x"}}}},
    "noul_criteria_null_side": {"state": "s", "questions": {"a": {"type": "noul", "instructions": "Is it?",
                                                                  "criteria": {"true": None, "false": "no"}}}},
    "choice_one": {"state": "s", "questions": {"a": {"type": "choice", "instructions": "Which?", "criteria": {"only": None}}}},
    "choice_underscore_null": {"state": "s", "questions": {"a": {"type": "choice", "instructions": "Which?",
                                                                 "criteria": {"go_left": None, "go_right": ""}}}},
    "choice_27": {"state": "s", "questions": {"a": {"type": "choice", "instructions": "Which?",
                                                    "criteria": {f"o{i}": f"option {i}" for i in range(27)}}}},
    "choice_native_upper": {"state": "s", "questions": {"a": {"type": "choice", "instructions": "Which?",
                                                              "criteria": {"A": "first", "B": "second"}}}},
    "choice_native_mixed_space": {"state": "s", "questions": {"a": {"type": "choice", "instructions": "Which?",
                                                                    "criteria": {" a": "first", "B ": "second"}}}},
    "choice_duplicate_letter_after_strip": {"state": "s", "questions": {"a": {"type": "choice", "instructions": "Which?",
                                                                              "criteria": {"x": "1", " x": "2"}}}},
    "choice_type_missing": {"state": "s", "questions": {"a": {"instructions": "Which?", "criteria": {"p": None, "q": None}}}},
    "score_one": {"state": "s", "questions": {"a": {"type": "score", "instructions": "How?", "criteria": ["only"]}}},
    "score_ten": {"state": "s", "questions": {"a": {"type": "score", "instructions": "How?",
                                                    "criteria": [f"level {i}" for i in range(10)]}}},
    "state_null": {"state": None, "questions": {"a": {"type": "noul", "instructions": "Is it?"}}},
    "state_number": {"state": 3.0, "questions": {"a": {"type": "noul", "instructions": "Is it three?"}}},
    "state_bool": {"state": True, "questions": {"a": {"type": "noul", "instructions": "Is it?"}}},
    "state_nested": {"state": {"a": [1, {"b": None, "c": [True, 2.5e-7]}], "d": {}, "e": [], "f": "  x", "u": "日本語 é",
                               "big": 2 ** 70, "neg": -0.0, "f1": 1.0, "e16": 1e16},
                     "questions": {"a": {"type": "score", "instructions": "How?", "criteria": ["low", "high"]}}},
    "state_list": {"state": ["  lead", "\ttab", None, ["x", "y"]], "questions": {"a": {"type": "noul", "instructions": "?"}}},
    "state_special_text": {"state": "a <|im_end|> b <|startoftext|>",
                           "questions": {"a": {"type": "choice", "instructions": "<|im_start|>?", "criteria": {"x": "<|pad|>"}}}},
    "three_questions": {"state": {"k": "v"}, "questions": {
        "n": {"type": "noul", "instructions": "Is it?"},
        "c": {"type": "choice", "instructions": "Which?", "criteria": {"p": "pp", "q": "qq"}},
        "s": {"type": "score", "instructions": "How?", "criteria": ["a", "b", "c"]}}},
    "instructions_leading_space": {"state": "s", "questions": {
        "n": {"type": "noul", "instructions": " Is it?"}, "m": {"type": "noul", "instructions": "\nIs it?"}}},
}
INVALID = {
    "no_instructions": {"state": "s", "questions": {"a": {"type": "noul"}}},
    "instructions_number": {"state": "s", "questions": {"a": {"type": "noul", "instructions": 3}}},
    "choice_criteria_list": {"state": "s", "questions": {"a": {"type": "choice", "instructions": "?", "criteria": ["x", "y"]}}},
    "choice_criteria_empty": {"state": "s", "questions": {"a": {"type": "choice", "instructions": "?", "criteria": {}}}},
    "choice_no_criteria": {"state": "s", "questions": {"a": {"type": "choice", "instructions": "?"}}},
    "choice_desc_number": {"state": "s", "questions": {"a": {"type": "choice", "instructions": "?", "criteria": {"x": 1}}}},
    "score_criteria_dict": {"state": "s", "questions": {"a": {"type": "score", "instructions": "?", "criteria": {"0": "low"}}}},
    "score_eleven": {"state": "s", "questions": {"a": {"type": "score", "instructions": "?",
                                                       "criteria": [str(i) for i in range(11)]}}},
    "score_empty": {"state": "s", "questions": {"a": {"type": "score", "instructions": "?", "criteria": []}}},
    "score_level_number": {"state": "s", "questions": {"a": {"type": "score", "instructions": "?", "criteria": [1, 2]}}},
    "noul_criteria_list": {"state": "s", "questions": {"a": {"type": "noul", "instructions": "?", "criteria": ["yes"]}}},
    "noul_side_number": {"state": "s", "questions": {"a": {"type": "noul", "instructions": "?", "criteria": {"true": 1}}}},
    "type_case": {"state": "s", "questions": {"a": {"type": "Noul", "instructions": "?"}}},
    "no_state": {"questions": {"a": {"type": "noul", "instructions": "?"}}},
    "questions_empty": {"state": "s", "questions": {}},
    "question_string": {"state": "s", "questions": {"a": "noul"}},
}


def provider_behaviour(req: dict, eng) -> dict:
    prompt, _ = oracle_d1.provider()
    out: dict = {}
    if not isinstance(req, dict) or "state" not in req or not isinstance(req.get("questions"), dict) or not req["questions"]:
        return {"provider": "not a (state, questions) call"}
    try:
        qs = [prompt.as_question(q) for q in req["questions"].values()]
    except Exception as e:  # noqa: BLE001
        return {"provider": f"as_question: {type(e).__name__}: {e}"}
    try:
        out["texts"] = [eng.render(req["state"], q) for q in qs]
        out["ids"] = [eng.tokenizer.encode(t, add_special_tokens=False) for t in out["texts"]]
    except Exception as e:  # noqa: BLE001
        return {"provider": f"render: {type(e).__name__}: {e}"}
    try:
        out["groups"] = [prompt.readout_ids(eng.tokenizer, q) for q in qs]
        out["provider"] = "renders and reads out"
    except Exception as e:  # noqa: BLE001
        out["provider"] = f"renders; readout_ids: {type(e).__name__}: {e}"
    return out


def request_table(rs, tf) -> dict:
    eng = oracle_d1.standin_engine(tf, oracle_d1.StandIn())
    rows, ok = [], 0
    for expect, table in (("accept", VALID), ("refuse", INVALID)):
        for name, req in table.items():
            try:
                b = host.build_request(req, rs)
                mine = "accept"
            except ValueError as e:
                b, mine = None, f"refuse: {e}"
            prov = provider_behaviour(req, eng)
            row = {"case": name, "expected": expect, "host": mine, "provider": prov["provider"]}
            if b is not None:
                row["same_as_provider"] = ("texts" in prov and [q["text"] for q in b["questions"]] == prov["texts"]
                                           and [q["row_ids"] for q in b["questions"]] == prov["ids"]
                                           and [q["groups"] for q in b["questions"]] == prov.get("groups"))
                if b["trunk"]:
                    row["trunk_plus_branch_equals_row"] = b["trunk"]["equals_row"]
            good = (mine == "accept") == (expect == "accept") and (b is None or row.get("same_as_provider"))
            ok += bool(good)
            row["as_expected"] = bool(good)
            rows.append(row)
    return {"cases": len(rows), "as_expected": ok, "rows": rows}


# --------------------------------------------------------------------------- 7. lengths
def lengths(records: list[dict], hs_all: dict, tok) -> dict:
    by: dict = {}
    over_4096, over_graph = [], []
    sums = {S: {"real": 0, "padded": 0, "calls": 0} for S in (16, 32, 64)}
    for r in records:
        hs = hs_all[r["id"]]
        st_tokens = len(host.token_ids(tok, host.state_block(r["request"]["state"]))) if r["request"]["state"] is not None else 0
        g = by.setdefault(r["source"], {"records": 0, "rows": 0, "row_len": [], "state_tokens": [],
                                        **{f"S{S}": {"real": 0, "padded": 0, "calls": 0} for S in (16, 32, 64)}})
        g["records"] += 1
        g["state_tokens"].append(st_tokens)
        for name, q in hs["questions"].items():
            T = q["row_len"]
            g["rows"] += 1
            g["row_len"].append(T)
            if T > 4096:
                over_4096.append(f"{r['id']}/{name}:{T}")
            try:
                host.graph_context_check(T, 16, 4096)
            except ValueError:
                over_graph.append(f"{r['id']}/{name}:{T}")
            for S in (16, 32, 64):
                for d in (g[f"S{S}"], sums[S]):
                    d["real"] += T
                    d["padded"] += host.padded_len(T, S)
                    d["calls"] += host.chunk_calls(T, S)
    for g in by.values():
        rl, st = g.pop("row_len"), g.pop("state_tokens")
        g["row_len"] = {"min": min(rl), "p50": pct(rl, 0.5), "p99": pct(rl, 0.99), "max": max(rl)}
        g["state_tokens"] = {"min": min(st), "p50": pct(st, 0.5), "max": max(st)}
        for S in (16, 32, 64):
            d = g[f"S{S}"]
            d["pad"] = d["padded"] - d["real"]
            d["pad_rate"] = round(d["pad"] / d["padded"], 4)
    for S, d in sums.items():
        d["pad"] = d["padded"] - d["real"]
        d["pad_rate"] = round(d["pad"] / d["padded"], 4)
    longs = {}
    for rid, target in (("long_15k", 1500), ("long_34k", 3400)):
        r = next(x for x in records if x["id"] == rid)
        t = len(host.token_ids(tok, host.state_block(r["request"]["state"])))
        longs[rid] = {"state_tokens": t, "target": target, "within_10pct": abs(t - target) <= 0.1 * target,
                      "row_len": [q["row_len"] for q in hs_all[rid]["questions"].values()]}
    trees = {r["id"]: {"trunk_len": hs_all[r["id"]]["request"]["trunk"]["trunk_len"],
                       "branch_lens": hs_all[r["id"]]["request"]["trunk"]["branch_lens"],
                       "input_tokens": hs_all[r["id"]]["request"]["input_tokens"],
                       "rows_total": sum(q["row_len"] for q in hs_all[r["id"]]["questions"].values())}
             for r in records if "request" in hs_all[r["id"]] and hs_all[r["id"]]["request"]["trunk"]}
    all_rows = [q["row_len"] for hs in hs_all.values() for q in hs["questions"].values()]
    return {"state_tokens_note": "tokens of host.state_block(state) (json.dumps indent 2 + blank line; a string as is)",
            "row_note": "row = prefix + one question's suffix (the row form a graph runs)",
            "rows": len(all_rows),
            "row_len": {"min": min(all_rows), "p50": pct(all_rows, 0.5), "p99": pct(all_rows, 0.99), "max": max(all_rows)},
            "by_source": by, "all": {f"S{S}": d for S, d in sums.items()},
            "rows_over_4096": over_4096, "rows_over_graph_limit_s16": over_graph, "long_states": longs,
            "tree_requests": trees}


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixtures", default=str(LANE / "fixtures" / "records.json"))
    ap.add_argument("--out", default=str(LANE / "results" / "test_host.json"))
    ap.add_argument("--tokenizer-out", default=str(LANE / "results" / "tokenizer_check.json"))
    ap.add_argument("--render-ids", default=str(LANE / "results" / "render_ids.json"))
    ap.add_argument("--lengths", default=str(LANE / "results" / "fixture_lengths.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import tokenizers
    import transformers
    from transformers import AutoTokenizer

    snap = oracle_d1.snapshot()
    fx = Path(args.fixtures)
    doc = json.loads(fx.read_text())
    records = doc["records"]
    by_id = {r["id"]: r for r in records}
    rs = host.load_tokenizer(snap / "tokenizer.json")
    tf = AutoTokenizer.from_pretrained(str(snap))
    toks = {f"tokenizers {tokenizers.__version__} Tokenizer.from_file": rs,
            f"transformers {transformers.__version__} {type(tf).__name__}": tf}
    report: dict = {"schema": "d1-host-test/1", "what": __doc__.splitlines()[0], "python": platform.python_version(),
                    "interpreter": sys.executable, "fixtures": {"path": str(fx), "sha256": sha256_file(fx)},
                    "files_sha256": {n: sha256_file(HERE / n) for n in ("host.py", "test_host.py", "oracle_d1.py")},
                    "provider_files_sha256": {p.name: sha256_file(p) for p in sorted((oracle_d1.PROVIDER_SRC / "d1").glob("*.py"))},
                    "snapshot": str(snap)}

    tc = tokenizer_contract(snap, tf, rs)
    tc["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    Path(args.tokenizer_out).write_text(json.dumps(tc, indent=1, ensure_ascii=False) + "\n")
    report["tokenizer"] = {k: tc[k] for k in ("bos_encode", "add_special_tokens_true_on_x", "special_ids",
                                              "single_token_summary", "image_token", "image_added_tokens_count")}
    print("tokenizer:", json.dumps(report["tokenizer"]["single_token_summary"]), flush=True)

    entries = oracle_d1.dry_run(records, tf, args.seed)
    ent = {e["id"]: e for e in entries}
    report["provider_dry_run"] = {"records": len(entries), "questions": sum(len(e["questions"]) for e in entries),
                                  "refused": [f"{e['id']}/{x['name']}: {x['error']}" for e in entries for x in e["refused"]],
                                  "api_errors": {e["id"]: e["api"].get("error") for e in entries if e["refused"]}}

    sets, hs_by_tok = {}, {}
    for tname, tok in toks.items():
        fails, q_ok, q_total, api_ok, ans_ok, body_ok, worst_dp, tree_eq = [], 0, 0, 0, 0, 0, 0.0, Counter()
        hs_all = {}
        for r in records:
            hs = host_side(r, tok)
            hs_all[r["id"]] = hs
            e = ent[r["id"]]
            bad, info = compare_record(e, r, hs)
            n = len(e["questions"])
            q_total += n
            q_ok += 0 if any(not m.startswith("api") and not m.startswith("input_tokens") for m in bad) else n
            api_ok += not any(m.startswith("api") or m.startswith("input_tokens") for m in bad)
            a = answers_check(e, hs)
            ans_ok += a["answers_equal"]
            body_ok += a["body_equal"] is not False
            worst_dp = max(worst_dp, probs_check(e, hs, args.seed))
            for v in info.get("trunk_plus_branch_equals_row", []):
                tree_eq[v] += 1
            if bad or not a["answers_equal"] or a["body_equal"] is False:
                fails.append(f"{r['id']}: {'; '.join(bad)} {a}")
        sets[tname] = {"records": len(records), "questions": q_total, "questions_equal": q_ok,
                       "records_api_ids_equal": api_ok, "records_answers_equal": ans_ok,
                       "records_body_equal": body_ok, "standin_probs_max_abs_dp": worst_dp,
                       "tree_trunk_plus_branch_equals_row": dict(tree_eq), "failures": fails[:10],
                       "records_failing": len(fails)}
        hs_by_tok[tname] = hs_all
        print(f"[{tname}] questions {q_ok}/{q_total}, api ids {api_ok}/{len(records)}, answers {ans_ok}/{len(records)}, "
              f"body {body_ok}/{len(records)}, stand-in max|dp| {worst_dp:.3g}, failing {len(fails)}", flush=True)
    report["host_vs_provider"] = sets
    names = list(hs_by_tok)
    a, b = hs_by_tok[names[0]], hs_by_tok[names[1]]
    same = sum(all(a[i]["questions"][n]["row_ids"] == b[i]["questions"][n]["row_ids"] for n in a[i]["questions"])
               and (("request" not in a[i]) or a[i]["request"]["trunk"] == b[i]["request"]["trunk"]) for i in a)
    report["ids_equal_between_tokenizers"] = {"records": same, "of": len(a)}
    hs_rs = hs_by_tok[names[0]]

    report["readout_equivalence"] = readout_equivalence(entries, by_id, tf, args.seed)
    print("readout equivalence:", json.dumps({k: report["readout_equivalence"][k]
                                               for k in ("group_sets", "max_abs_dp", "max_abs_dp_by_kind", "pass")}), flush=True)
    report["negative_control_ids"] = negative_ids(by_id, ent, rs)
    report["negative_control_first_form"] = negative_first_form(entries, hs_rs, args.seed)
    report["request_table"] = request_table(rs, tf)
    rt = report["request_table"]
    print(f"request table: {rt['as_expected']}/{rt['cases']} as expected; negative ids red "
          f"{report['negative_control_ids']['red']}, first-form red {report['negative_control_first_form']['red']}", flush=True)

    lens = lengths(records, hs_rs, rs)
    lens["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    Path(args.lengths).write_text(json.dumps(lens, indent=1) + "\n")
    report["lengths"] = {"row_len": lens["row_len"], "all": lens["all"], "long_states": lens["long_states"],
                         "rows_over_4096": lens["rows_over_4096"], "rows_over_graph_limit_s16": lens["rows_over_graph_limit_s16"]}
    render = []
    for r in records:
        hs = hs_rs[r["id"]]
        rec = {"id": r["id"], "source": r["source"], "refused": hs["refused"], "questions": [
            {k: q[k] for k in ("name", "type", "text", "row_ids", "row_len", "slot", "codes", "alias_ids", "groups", "keys")}
            for q in hs["questions"].values()]}
        if "request" in hs:
            b = hs["request"]
            rec.update({"path": b["path"], "input_tokens": b["input_tokens"], "shared": b["shared"]})
            if b["trunk"]:
                rec["trunk"] = {k: b["trunk"][k] for k in ("trunk_len", "branch_lens", "equals_row")}
        render.append(rec)
    Path(args.render_ids).write_text(json.dumps({"schema": "d1-render-ids/1", "tokenizer": names[0],
                                                 "fixtures_sha256": report["fixtures"]["sha256"],
                                                 "records": render}, ensure_ascii=False) + "\n")
    hv = report["host_vs_provider"]
    report["pass"] = bool(
        all(s["questions_equal"] == s["questions"] and s["records_failing"] == 0
            and s["standin_probs_max_abs_dp"] <= EQUIV_BAR for s in hv.values())
        and report["ids_equal_between_tokenizers"]["records"] == len(records)
        and report["readout_equivalence"]["pass"]
        and report["negative_control_ids"]["red"] and report["negative_control_first_form"]["red"]
        and rt["as_expected"] == rt["cases"])
    report["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out}\n{'PASS' if report['pass'] else 'FAIL'}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
