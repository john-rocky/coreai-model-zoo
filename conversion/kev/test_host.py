#!/usr/bin/env python3
"""Test: `host.py` rebuilds every oracle record from its raw request — ids, rows, readout indices, keys, answers, usage.

For both checkpoints (Kev-0.8B: `oracle/`, Kev-4B: `oracle_4b/`) and both fixture files (`fixtures/records.json`,
384 records / 434 questions; `fixtures/heldout.json`, 130 / 130), every record's request goes through
`host.build_rows` with each tokenizer under test (the bundle's `tokenizer.json` through `tokenizers`, and the same
directory through transformers' AutoTokenizer when it imports), and must equal the author's oracle:

  * packed      ids, seg, pos (kev.model.encode) and the state token count;
  * rows        per question: qid, type, keys, row_ids (state + branch), row_len, state_len, decide, opts;
  * answers     `host.to_answers(oracle probs)` serialized == the oracle's `response.answers` serialized (order too);
  * usage       input_tokens and output_tokens == the oracle's; the whole body from `host.response` with the
                request's model == the oracle's body but latency_ms (the author's body echoes the request's model).

Negative controls (each must go red): one word of one question's instructions changed (the ids); the 0.8B oracle's
probabilities scored against the 4B oracle's answers (the answers). Also recorded: the request checks
(`host.validate_request`) on a fixed table of valid and invalid requests, cross-checked against the author's pydantic
models, `to_record` and `encode` when `kev` imports (the oracle venv; the table has delimiter text such as
`<|fim_suffix|>` in the state, instructions and options); the serving limits (`admit`'s messages, against the
author's `kev.model.admit` when it imports) and the graph limit; and how many answers Python 3.11's plain sum would
change (`host.py_sum` is Python 3.12's).

Round 15, the call plan (`host.plan`, `shared_prefix_plan`, `graph_context_check`, `graph_shape`; no tokenizer, no
graph): for every n = 1..4,095 ids and every (L, q) in {128, 256, 512} x {1, 8, 16} (L = the call max; the dynamic-S
graph's smallest call 2): the call lengths sum to ceil(n / q) * q (q = 1: n), every call is a multiple of q in
max(q, 2)..L, only the last call holds a pad (fewer than q ids), the real ids add up to n, and the plan equals
readout_gate.host_pieces (the gate's own implementation) call for call; at q = 1 a 1-id run is refused. graph_shape
reads the language blocks a bundle can carry (prefill_chunk; query_len_range with and without query_len_call_max /
query_len_multiple, and a host's overrides) and refuses the inconsistent ones. For the static-S case cap = q = 16 the plan is the
fixed grid every earlier host ran (ceil(n / 16) calls of 16, the last padded). The shared plan, for every state length
Ls = 1..1,200 and branches of 4..300 ids: the prefix is a multiple of q with no pad, prefix + tail = the row, the
tail's positions start at the prefix, and its padded end equals the direct run's (so the row limit is the same both
ways). The row limits: 4,080 / 4,081 ids at q = 16 (and the static S = 16), 4,095 / 4,096 at q = 1.

    cd conversion/kev
    $ZOO_WORK_ROOT/_kev/venv-oracle/bin/python test_host.py --out $ZOO_WORK_ROOT/_kev/results/host_test_oracle_venv.json
    <coreai-models venv>/bin/python test_host.py             # -> $ZOO_WORK_ROOT/_kev/results/host_test.json
"""
from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import host  # noqa: E402
from _paths import record_path, work_path  # noqa: E402

LANE = work_path("_kev")
BUNDLES = {"kev-0.8b": LANE / "exports" / "bundles" / "kev_0_8b_decode_fp16_pf16",
           "kev-4b": LANE / "exports" / "bundles" / "kev_4b_decode_fp16_pf16"}
SETS = [("kev-0.8b", "fixture", "oracle/records_oracle.json", "fixtures/records.json"),
        ("kev-0.8b", "heldout", "oracle/heldout/records_oracle.json", "fixtures/heldout.json"),
        ("kev-4b", "fixture", "oracle_4b/records_oracle.json", "fixtures/records.json"),
        ("kev-4b", "heldout", "oracle_4b/heldout/records_oracle.json", "fixtures/heldout.json")]
NEGATIVE_RECORD, NEGATIVE_QUESTION = "own_t01", "team"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokenizers_under_test(bundle: Path) -> tuple[dict, dict]:
    import tokenizers
    toks = {f"tokenizers {tokenizers.__version__} Tokenizer.from_file":
            host.load_tokenizer(bundle / "tokenizer" / "tokenizer.json")}
    errors = {}
    try:
        import transformers
        from transformers import AutoTokenizer
        toks[f"transformers {transformers.__version__} AutoTokenizer"] = AutoTokenizer.from_pretrained(str(bundle / "tokenizer"))
    except Exception as e:  # noqa: BLE001
        errors["transformers AutoTokenizer"] = f"{type(e).__name__}: {e}"[:300]
    return toks, errors


def check_record(e: dict, request: dict, tok) -> tuple[list[str], dict]:
    """Mismatch messages for one oracle record (empty = pass) and the host's rows."""
    b = host.build_rows(request, tok)
    rec, meta = host.to_record(host.validate_request(request))
    enc = host.admit(tok, rec)
    bad = []
    for name, mine, want in (("ids", b["ids"], e["ids"]), ("seg", enc["seg"], e["seg"]), ("pos", enc["pos"], e["pos"]),
                             ("state_tokens", enc["state_tokens"], e["state_tokens"])):
        if mine != want:
            bad.append(f"{name} differ")
    if len(b["rows"]) != len(e["questions"]):
        bad.append(f"{len(b['rows'])} rows vs {len(e['questions'])} questions")
    for k, (r, q) in enumerate(zip(b["rows"], e["questions"])):
        mine = (r["qid"], r["type"], r["keys"], r["row_ids"], len(r["row_ids"]), b["state_len"], r["decide"], r["opts"])
        want = (q["qid"], q["type"], q["keys"], q["row_ids"], q["row_len"], q["state_len"], q["decide"], q["opts"])
        if mine != want:
            names = ("qid", "type", "keys", "row_ids", "row_len", "state_len", "decide", "opts")
            bad.append(f"question {k}: " + ", ".join(n for n, a, w in zip(names, mine, want) if a != w))
    if b["input_tokens"] != e["response"]["usage"]["input_tokens"]:
        bad.append("input_tokens")
    return bad, b


def answers_check(e: dict, b: dict, tok) -> dict:
    """to_answers(oracle probs), output_tokens and the whole body against the oracle's response."""
    answers = host.to_answers([q["probs"] for q in e["questions"]], b["meta"])
    out_tok = host.output_tokens(tok, answers)
    body = host.response(b["model"], answers, b["input_tokens"], out_tok, e["response"]["latency_ms"])
    return {"answers_equal": json.dumps(answers) == json.dumps(e["response"]["answers"]),
            "output_tokens_equal": out_tok == e["response"]["usage"]["output_tokens"],
            "body_equal": json.dumps(body) == json.dumps(e["response"]),
            "legend_equal": all(answers[q]["legend"] == a["legend"] for q, a in e["response"]["answers"].items()
                                if a["type"] == "score")}


def naive_sum(values):
    f = 0
    for x in values:
        f = f + x
    return f


def py311_answers_differ(e: dict, meta: list) -> bool:
    a = host.to_answers([q["probs"] for q in e["questions"]], meta)
    keep = host.py_sum
    host.py_sum = naive_sum
    try:
        b = host.to_answers([q["probs"] for q in e["questions"]], meta)
    finally:
        host.py_sum = keep
    return json.dumps(a) != json.dumps(b)


def negative_ids(recs: dict, oracle: dict, tok) -> dict:
    """One word of one question's instructions changed: the record must go red."""
    rec = copy.deepcopy(recs[NEGATIVE_RECORD])
    q = rec["request"]["questions"][NEGATIVE_QUESTION]
    words = q["instructions"].split(" ")
    words[1] = "zqxvortmund"                                  # "Which team ..." -> "Which zqxvortmund ..."
    q["instructions"] = " ".join(words)
    bad, b = check_record(oracle[NEGATIVE_RECORD], rec["request"], tok)
    return {"record": NEGATIVE_RECORD, "question": NEGATIVE_QUESTION, "changed_instructions": q["instructions"],
            "red": bool(bad), "messages": bad[:4],
            "row_len_host_vs_oracle": [len(b["rows"][0]["row_ids"]), oracle[NEGATIVE_RECORD]["questions"][0]["row_len"]]}


def negative_answers(o08: dict, o4: dict, metas: dict) -> dict:
    """The 0.8B oracle's probabilities against the 4B oracle's answers: the answers check must go red."""
    red = sum(json.dumps(host.to_answers([q["probs"] for q in o08[i]["questions"]], metas[i]))
              != json.dumps(o4[i]["response"]["answers"]) for i in o08)
    return {"records": len(o08), "answers_differing": red, "red": red > 0}


# --------------------------------------------------------------------------- the request checks
VALID = {
    "state_null": {"state": None, "questions": {"a": {"type": "noul"}}},
    "state_number": {"state": 3.0, "questions": {"a": {"type": "noul", "instructions": "Is it three?"}}},
    "state_bool": {"state": True, "questions": {"a": {"type": "noul"}}},
    "state_nested": {"state": {"a": [1, {"b": None, "c": [True, 2.5e-7]}], "d": {}, "e": [], "f": "  x"},
                     "questions": {"a": {"type": "score", "criteria": ["low", {"lvl": 2}, 3, None]}}},
    "state_list_ws": {"state": ["  lead", "\ttab", None, ["x", "y"]], "questions": {"a": {"type": "noul"}}},
    "noul_criteria_null": {"state": "s", "questions": {"a": {"type": "noul", "criteria": None}}},
    "noul_criteria_empty": {"state": "s", "questions": {"a": {"type": "noul", "criteria": {}}}},
    "noul_criteria_extra_key": {"state": "s", "questions": {"a": {"type": "noul", "criteria": {"true": "t", "maybe": 1}}}},
    "noul_criteria_values": {"state": "s", "questions": {"a": {"type": "noul", "criteria": {"true": 0, "false": False}}}},
    "choice_1": {"state": "s", "questions": {"a": {"type": "choice", "criteria": {"only": None}}}},
    "choice_255": {"state": "s", "questions": {"a": {"type": "choice", "criteria": {f"o{i}": None for i in range(255)}}}},
    "choice_desc_types": {"state": "s", "questions": {"a": {"type": "choice", "criteria": {"x": "", "y": [1, 2], "z": {"k": "v"}}}}},
    "score_1": {"state": "s", "questions": {"a": {"type": "score", "criteria": ["one"]}}},
    "score_255": {"state": "s", "questions": {"a": {"type": "score", "criteria": [str(i) for i in range(255)]}}},
    "instructions_object": {"state": "s", "questions": {"a": {"type": "noul", "instructions": {"q": "Is it?", "n": 2}}}},
    "model_given": {"model": "jev-latest", "state": "s", "questions": {"a": {"type": "noul"}}},
    "extra_fields": {"state": "s", "extra": 1, "questions": {"a": {"type": "noul", "label": 3}}},
    "special_text": {"state": "a <|fim_suffix|> b <|im_end|>", "questions": {"a": {"type": "choice",
                     "instructions": "<|box_end|>?", "criteria": {"<|x|>": "<|y|>"}}}},
    "state_exotic": {"state": {"big": 2 ** 70, "tiny": 1e-7, "huge": 1e16, "neg_zero": -0.0, "uni": "日本語 ¦ é",
                               "list": ["\n  lead newline", [], {}, True, None, [1, [2, {"k": [3]}]]], "sp ace": {"x": ""}},
                     "questions": {"a": {"type": "score", "instructions": ["step 1", {"then": "step 2"}],
                                         "criteria": [{"lvl": 0, "txt": "none"}, ["a", "b"], 1.5, False]}}},
}
INVALID = {
    "not_object": ["state"],
    "no_state": {"questions": {"a": {"type": "noul"}}},
    "model_number": {"model": 3, "state": "s", "questions": {"a": {"type": "noul"}}},
    "model_null": {"model": None, "state": "s", "questions": {"a": {"type": "noul"}}},
    "questions_missing": {"state": "s"},
    "questions_empty": {"state": "s", "questions": {}},
    "questions_list": {"state": "s", "questions": [{"type": "noul"}]},
    "question_string": {"state": "s", "questions": {"a": "noul"}},
    "type_missing": {"state": "s", "questions": {"a": {"criteria": {"x": None}}}},
    "type_unknown": {"state": "s", "questions": {"a": {"type": "rank", "criteria": ["x"]}}},
    "type_case": {"state": "s", "questions": {"a": {"type": "Noul"}}},
    "noul_criteria_list": {"state": "s", "questions": {"a": {"type": "noul", "criteria": ["yes", "no"]}}},
    "choice_no_criteria": {"state": "s", "questions": {"a": {"type": "choice"}}},
    "choice_criteria_null": {"state": "s", "questions": {"a": {"type": "choice", "criteria": None}}},
    "choice_empty": {"state": "s", "questions": {"a": {"type": "choice", "criteria": {}}}},
    "choice_256": {"state": "s", "questions": {"a": {"type": "choice", "criteria": {f"o{i}": None for i in range(256)}}}},
    "choice_list": {"state": "s", "questions": {"a": {"type": "choice", "criteria": ["x", "y"]}}},
    "score_no_criteria": {"state": "s", "questions": {"a": {"type": "score"}}},
    "score_empty": {"state": "s", "questions": {"a": {"type": "score", "criteria": []}}},
    "score_256": {"state": "s", "questions": {"a": {"type": "score", "criteria": [str(i) for i in range(256)]}}},
    "score_object": {"state": "s", "questions": {"a": {"type": "score", "criteria": {"0": "low"}}}},
}


def request_checks(tok) -> dict:
    """host.validate_request on the table; with `kev` importable, the author's pydantic decision and to_record too."""
    try:
        import kev.model as km
        from kev.api import SystemOneRequest
        from kev.api import to_record as kev_to_record
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(str(BUNDLES["kev-0.8b"] / "tokenizer"))   # the author's encode calls tok(text)
        author = True
    except Exception:  # noqa: BLE001
        author = False
    rows = []
    for expect, table in (("accept", VALID), ("reject", INVALID)):
        for name, req in table.items():
            try:
                host.validate_request(req)
                mine = "accept"
            except ValueError:
                mine = "reject"
            row = {"case": name, "expected": expect, "host": mine}
            if author:
                try:
                    r = SystemOneRequest.model_validate(req)
                    row["author"] = "accept"
                    if mine == "accept":
                        rec_a, meta_a = kev_to_record(r)
                        rec_h, meta_h = host.to_record(host.validate_request(req))
                        row["to_record_equal"] = (rec_h["state"] == rec_a["state"] and meta_h == meta_a and
                                                  [(q["instr"], q["options"]) for q in rec_h["questions"]]
                                                  == [(q["instr"], q["options"]) for q in rec_a["questions"]])
                        enc_a = km.encode(hf_tok, rec_a, max_state=km.SERVE_MAX_STATE, max_branch=km.SERVE_MAX_BRANCH, strict=True)
                        enc_h = host.encode(tok, rec_h)
                        row["encode_equal"] = all(enc_a[k] == enc_h[k] for k in ("ids", "seg", "pos", "decide_idx", "opt_idx",
                                                                                   "state_tokens"))
                except Exception:  # noqa: BLE001
                    row["author"] = "reject"
            if mine == "accept":
                row["rows"] = [len(r["row_ids"]) for r in host.build_rows(req, tok)["rows"]]
            rows.append(row)
    return {"author_pydantic_checked": author, "cases": len(rows),
            "host_as_expected": sum(r["host"] == r["expected"] for r in rows),
            "author_agrees": sum(r.get("author") == r["host"] for r in rows) if author else None,
            "to_record_equal": sum(bool(r.get("to_record_equal")) for r in rows) if author else None,
            "to_record_compared": sum("to_record_equal" in r for r in rows) if author else None,
            "encode_equal": sum(bool(r.get("encode_equal")) for r in rows) if author else None, "rows": rows}


def limit_checks(tok) -> dict:
    """The serving limits (admit's messages; the author's kev.model.admit when it imports) and the graph limit."""
    long_state = "word " * 70000
    ok_state = "word " * 100
    out = {}
    cases = {"state_over_65536": {"state": long_state, "questions": {"a": {"type": "noul"}}},
             "branch_over_73728": {"state": ok_state, "questions": {"a": {"type": "noul", "instructions": "word " * 74000}}}}
    try:
        import kev.model as km
        from kev.api import SystemOneRequest
        from kev.api import to_record as kev_to_record

        from transformers import AutoTokenizer

        class _Stub:            # admit() calls model.encode(tok, rec, ...)
            encode = staticmethod(functools.partial(km.encode, option_isolation=False))
        hf_tok = AutoTokenizer.from_pretrained(str(BUNDLES["kev-0.8b"] / "tokenizer"))   # the author's encode calls tok(text)
        author = True
    except Exception:  # noqa: BLE001
        author = False
    for name, req in cases.items():
        try:
            host.build_rows(req, tok)
            out[name] = {"host": "accepted"}
        except host.ContextOverflow as e:
            out[name] = {"host": str(e), "state_tokens": e.state_tokens, "max_state": e.max_state}
        if author:
            rec, _ = kev_to_record(SystemOneRequest.model_validate(req))
            try:
                km.admit(_Stub(), hf_tok, rec)
                out[name]["author"] = "accepted"
            except km.ContextOverflow as e:
                out[name]["author"] = str(e)
            out[name]["same_message"] = out[name]["author"] == out[name]["host"]
    graph = {}
    for T in (4080, 4081):
        try:
            host.graph_context_check([{"qid": "q", "row_ids": [0] * T}], max_ctx=4096, chunk=16)
            graph[str(T)] = "fits"
        except ValueError as e:
            graph[str(T)] = str(e)
    out["graph_4096_s16"] = graph
    out["author_checked"] = author
    return out


def plan_checks() -> dict:
    """Round 15: host.plan / shared_prefix_plan / graph_context_check against their contract and readout_gate's own
    host_pieces."""
    import readout_gate
    out: dict = {"cases": {}, "failures": []}

    def fail(msg: str) -> None:
        if len(out["failures"]) < 20:
            out["failures"].append(msg)

    qmin = 2
    for cap in (128, 256, 512):
        for q in (1, 8, 16):
            n_ok = refused = 0
            for n in range(1, 4096):
                try:
                    pl = host.plan(n, cap, q, qmin)
                except ValueError:
                    refused += 1
                    if not (q == 1 and n < qmin):
                        fail(f"cap {cap} q {q} n {n}: refused")
                    continue
                if q == 1 and n < qmin:
                    fail(f"cap {cap} q {q} n {n}: not refused")
                calls, real = [c for c, _ in pl], [r for _, r in pl]
                bad = []
                if sum(real) != n:
                    bad.append("real ids != n")
                if sum(calls) != (-(-n // q) * q if q > 1 else n):
                    bad.append("call lengths do not sum to ceil(n / q) * q")
                if any(c % q or c < max(q, qmin) or c > cap for c in calls):
                    bad.append("a call is not a multiple of q in max(q, 2)..cap")
                if any(c != r for c, r in pl[:-1]) or not (0 <= calls[-1] - real[-1] < q):
                    bad.append("a pad outside the last call, or q or more pad ids")
                try:
                    theirs = [tuple(x) for x in readout_gate.host_pieces(n, cap, q, qmin)]
                except SystemExit as e:   # the gate refuses the run
                    theirs = f"refused: {e}"
                if pl != theirs:
                    bad.append("differs from readout_gate.host_pieces")
                if bad:
                    fail(f"cap {cap} q {q} n {n}: {'; '.join(bad)}")
                else:
                    n_ok += 1
            out["cases"][f"cap{cap}_q{q}"] = {"n": 4095, "ok": n_ok, "refused": refused,
                                              "call_lengths": host.call_lengths(cap, q, qmin)}
    # the static-S grid (cap = q = qmin = 16) = ceil(n / 16) calls of 16, the last padded
    grid_ok = sum(host.plan(n, 16, 16, 16) == [(16, 16)] * (-(-n // 16) - 1) + [(16, n - 16 * (-(-n // 16) - 1))]
                  for n in range(1, 4096))
    out["cases"]["static_s16_grid"] = {"n": 4095, "ok": grid_ok}
    if grid_ok != 4095:
        fail(f"static S16 grid: {4095 - grid_ok} plans differ")
    # the shared plan
    sh_ok = sh_n = 0
    for cap, q in ((128, 16), (128, 8), (256, 16), (512, 16), (512, 8), (128, 1), (16, 16)):
        mn = qmin if (cap, q) != (16, 16) else 16
        for Ls in range(1, 1201):
            for br in (4, 5, 17, 100, 129, 300):
                n = Ls + br
                sh_n += 1
                sp = host.shared_prefix_plan(Ls, cap, q, mn)
                k = sp["shared_tokens"]
                bad = []
                if k % q or k > Ls or (k and k < mn) or (k == 0 and Ls // q * q >= mn):
                    bad.append(f"k {k}")
                try:
                    pre = host.plan(k, cap, q, mn) if k else []
                    if any(c != r for c, r in pre):
                        bad.append("a pad in the prefix")
                    tail = host.plan(n - k, cap, q, mn)
                    if sum(r for _, r in pre) + sum(r for _, r in tail) != n:
                        bad.append("prefix + tail != the row")
                    end_shared = k + sum(c for c, _ in tail)
                    if end_shared != host.padded_end(n, cap, q, mn):
                        bad.append(f"padded end {end_shared} != the direct run's {host.padded_end(n, cap, q, mn)}")
                except ValueError as e:
                    bad.append(f"refused: {e}")
                if bad:
                    fail(f"shared cap {cap} q {q} Ls {Ls} branch {br}: {'; '.join(bad)}")
                else:
                    sh_ok += 1
    out["cases"]["shared"] = {"n": sh_n, "ok": sh_ok}
    limits = {}
    for label, kw, (fits, over) in (("dynamic_q16_cap128", {"chunk": 128, "q": 16, "qmin": 2}, (4080, 4081)),
                                    ("dynamic_q1_cap128", {"chunk": 128, "q": 1, "qmin": 2}, (4095, 4096)),
                                    ("static_s16", {"chunk": 16}, (4080, 4081))):
        res = {}
        for T in (fits, over):
            try:
                host.graph_context_check([{"qid": "q", "row_ids": [0] * T}], max_ctx=4096, **kw)
                res[str(T)] = "fits"
            except ValueError as e:
                res[str(T)] = str(e)
        limits[label] = res
        if res[str(fits)] != "fits" or res[str(over)] == "fits":
            fail(f"row limit {label}: {res}")
    out["row_limits"] = limits
    # graph_shape on the language blocks a bundle can carry, and a host's overrides
    shapes = {}
    for label, lang, kw, want in (
            ("static_s16", {"prefill_chunk": 16}, {}, {"dynamic": False, "graph_max": 16, "cap": 16, "q": 16, "qmin": 16}),
            ("d128_round14", {"query_len_range": [2, 128]}, {}, {"dynamic": True, "graph_max": 128, "cap": 128, "q": 1, "qmin": 2}),
            ("d128_q16", {"query_len_range": [2, 128], "query_len_multiple": 16}, {},
             {"dynamic": True, "graph_max": 128, "cap": 128, "q": 16, "qmin": 2}),
            ("d512_L128_q16", {"query_len_range": [2, 512], "query_len_call_max": 128, "query_len_multiple": 16}, {},
             {"dynamic": True, "graph_max": 512, "cap": 128, "q": 16, "qmin": 2}),
            ("d512_override_L256_q8", {"query_len_range": [2, 512], "query_len_call_max": 128, "query_len_multiple": 16},
             {"call_max": 256, "multiple": 8}, {"dynamic": True, "graph_max": 512, "cap": 256, "q": 8, "qmin": 2}),
            ("refuse_L_over_graph", {"query_len_range": [2, 128], "query_len_call_max": 256}, {}, "ValueError"),
            ("refuse_L_not_multiple", {"query_len_range": [2, 512], "query_len_call_max": 120, "query_len_multiple": 16}, {},
             "ValueError"),
            ("refuse_static_override", {"prefill_chunk": 16}, {"call_max": 16}, "ValueError")):
        try:
            got = host.graph_shape(lang, **kw)
        except ValueError:
            got = "ValueError"
        shapes[label] = got == want
        if got != want:
            fail(f"graph_shape {label}: {got} != {want}")
    out["graph_shape"] = shapes
    out["pass"] = not out["failures"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(LANE / "results" / "host_test.json"))
    args = ap.parse_args()
    report = {"schema": "kev-host-test/1",
              "what": "host.build_rows / to_answers / output_tokens / response vs the author's fp32 oracle, every record",
              "python": platform.python_version(), "interpreter": record_path(sys.executable),
              "host_py_sha256": sha256_file(HERE / "host.py"), "test_host_py_sha256": sha256_file(Path(__file__).resolve()),
              "tokenizer_sha256": {m: {f.name: sha256_file(f) for f in sorted((b / "tokenizer").iterdir())}
                                   for m, b in BUNDLES.items()},
              "sets": {}, "tokenizer_errors": {}}
    oracles, fixtures, metas = {}, {}, {}
    for model, which, opath, fpath in SETS:
        doc = json.loads((LANE / opath).read_text())
        recs = {r["id"]: r for r in json.loads((LANE / fpath).read_text())["records"]}
        oracles[(model, which)] = {e["id"]: e for e in doc["records"]}
        fixtures[which] = recs
        toks, errs = tokenizers_under_test(BUNDLES[model])
        report["tokenizer_errors"].update(errs)
        entry = {"oracle": {"path": str(LANE / opath), "sha256": sha256_file(LANE / opath)},
                 "fixtures": {"path": str(LANE / fpath), "sha256": sha256_file(LANE / fpath)},
                 "records": len(doc["records"]), "rows": sum(len(e["questions"]) for e in doc["records"]),
                 "tokenizers": {}}
        for tname, tok in toks.items():
            fails, rows_ok, ans_ok, out_ok, body_ok, legend_ok, py311 = [], 0, 0, 0, 0, 0, 0
            ids_by_record = {}
            for e in doc["records"]:
                bad, b = check_record(e, recs[e["id"]]["request"], tok)
                ids_by_record[e["id"]] = b["ids"]
                rows_ok += 0 if bad else len(e["questions"])
                a = answers_check(e, b, tok)
                ans_ok += a["answers_equal"]
                out_ok += a["output_tokens_equal"]
                body_ok += a["body_equal"]
                legend_ok += a["legend_equal"]
                py311 += py311_answers_differ(e, b["meta"])
                metas.setdefault((which, e["id"]), b["meta"])
                if bad or not all(a.values()):
                    fails.append(f"{e['id']}: {'; '.join(bad)} {a}")
            entry["tokenizers"][tname] = {"rows_equal": rows_ok, "records_answers_equal": ans_ok,
                                          "records_output_tokens_equal": out_ok, "records_body_equal_but_latency": body_ok,
                                          "records_legend_equal": legend_ok, "records_failing": len(fails),
                                          "failures": fails[:10], "py311_plain_sum_changes_answers": py311}
            print(f"{model} {which} [{tname}]: rows {rows_ok}/{entry['rows']}, answers {ans_ok}/{entry['records']}, "
                  f"output_tokens {out_ok}/{entry['records']}, body {body_ok}/{entry['records']}, "
                  f"py3.11 sum would change {py311}", flush=True)
            entry.setdefault("_ids", ids_by_record)
        report["sets"][f"{model}:{which}"] = entry
    # the two tokenizers (and the two bundles' tokenizer files) give the same ids
    same = {}
    for which in ("fixture", "heldout"):
        a, b = report["sets"][f"kev-0.8b:{which}"].pop("_ids"), report["sets"][f"kev-4b:{which}"].pop("_ids")
        same[which] = sum(a[i] == b[i] for i in a)
    report["ids_equal_between_bundles"] = same
    tok0 = host.load_tokenizer(BUNDLES["kev-0.8b"] / "tokenizer" / "tokenizer.json")
    report["negative_control_ids"] = negative_ids(fixtures["fixture"], oracles[("kev-0.8b", "fixture")], tok0)
    report["negative_control_answers"] = negative_answers(oracles[("kev-0.8b", "fixture")], oracles[("kev-4b", "fixture")],
                                                          {i: metas[("fixture", i)] for i in oracles[("kev-0.8b", "fixture")]})
    print("negative controls:", json.dumps({"ids": report["negative_control_ids"]["red"],
                                            "answers": report["negative_control_answers"]}))
    report["request_checks"] = request_checks(tok0)
    rc = report["request_checks"]
    print(f"request checks: host as expected {rc['host_as_expected']}/{rc['cases']}, author agrees {rc['author_agrees']}, "
          f"to_record equal {rc['to_record_equal']}/{rc['to_record_compared']}, encode equal {rc['encode_equal']}")
    report["limit_checks"] = limit_checks(tok0)
    report["plan_checks"] = plan_checks()
    pc = report["plan_checks"]
    print("plan checks:", json.dumps({k: v.get("ok") for k, v in pc["cases"].items()}), "failures", pc["failures"][:3],
          "PASS" if pc["pass"] else "FAIL")
    print("limits:", json.dumps({k: (v if isinstance(v, (bool, str)) else {kk: str(vv)[:60] for kk, vv in v.items()})
                                 for k, v in report["limit_checks"].items()}))
    totals = {}
    for key, entry in report["sets"].items():
        for tname, t in entry["tokenizers"].items():
            totals.setdefault(key, []).append(t["rows_equal"] == entry["rows"] and t["records_failing"] == 0)
    lc = report["limit_checks"]
    report["pass"] = (all(all(v) for v in totals.values())
                      and all(v == len(fixtures[w]) for w, v in same.items())
                      and report["negative_control_ids"]["red"] and report["negative_control_answers"]["red"]
                      and rc["host_as_expected"] == rc["cases"]
                      and (not rc["author_pydantic_checked"] or (rc["author_agrees"] == rc["cases"]
                                                                  and rc["to_record_equal"] == rc["to_record_compared"]
                                                                  and rc["encode_equal"] == rc["to_record_compared"]))
                      and lc["graph_4096_s16"]["4080"] == "fits" and lc["graph_4096_s16"]["4081"] != "fits"
                      and all(lc[n]["host"] != "accepted" for n in ("state_over_65536", "branch_over_73728"))
                      and (not lc["author_checked"] or all(lc[n]["same_message"] for n in ("state_over_65536", "branch_over_73728")))
                      and pc["pass"])
    report["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out}\n{'PASS' if report['pass'] else 'FAIL'}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
