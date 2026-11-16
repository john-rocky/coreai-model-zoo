#!/usr/bin/env python3
"""Swift host gate: the `kev` CLI (apps/Kev) against the author's fp32 oracle and the Python reference host.

The Swift side answers every fixture record from its raw request by itself — the request JSON parsed with its
number literals kept, the author's `render` / `option_text` / keys, `user_tokens` through swift-transformers'
tokenizer, the row form, the AOT graph in its chunk order (direct, and the shared prefix), the float64 pointer head,
`to_answers` with Python 3.12's float sum, `json.dumps(answers)` for the output tokens — and writes one JSON per pass
(`kev fixture`). This script runs the model-free checks and scores the passes:

  render    `kev render-test` vs `host.py` on every fixture and held-out request (render of the state, the
            instructions and every option; keys; legend; the request checks of test_host.py's table), edge values
            (floats, big ints, -0.0, booleans, null, nesting, empty containers, unicode, delimiter text), raw literals
            json.loads accepts but json.dumps never writes, round(x, 4) and repr on 200,000 doubles, and json.dumps
            (Python's defaults, ensure_ascii) of answer-shaped objects   -> results/swift_render_test.json
  ids       `kev rows` (tokenizer only, no graph) vs the oracle: packed ids, row ids, <decide> / </opt> indices, keys,
            input / output token counts, on every record of both checkpoints   -> folded into the score transcript
  score     one model's fixture pass + held-out pass (direct and shared) -> results/swift_gate_<model>.json:
            G1 ids / row ids / decide / opts / keys = the oracle's on every row;
            G2 the bar of the readout gate (argmax per question = the oracle's outside near-ties, max |dp| <= 0.02
               over every option, mean over records of the record's mean |dp| <= 0.002; finite; the reset re-run);
            G3 Swift vs Python on the same AOT asset: each row's hidden sha256 vs the readout gate's hidden rows (which
               round 5 showed equal to decide.py's, bit for bit), p vs the Python host's p (decide.py check), the
               answers' json.dumps vs the Python host's, byte for byte; shared = direct
  negative  one word of one question's instructions changed -> `kev rows` on it: G1 must go red
  jit       `kev fixture --asset jit` (the .aimodel specialized by Swift) vs the AOT pass on the same records
  longrow   the graph's longest rows (diagnosis, not part of the gate): a fixture of rows of about 3,000 and 4,030 -
            4,080 tokens built from the fixture's own long states, the author's fp32 oracle on it (oracle_kev.py, with
            the hidden rows kept), the Python graph run (decide.Kev) and the Swift pass -> results/longrow_<model>.json
  timing    `_time_mac.sh`'s window -> results/swift_timing_r7.json, beside round 5's Python numbers

Round 15 (a dynamic-S bundle, round 14's `query_len_range` with `query_len_multiple` q): `score --bundle` reads the
bundle's call plan (host.graph_shape / plan) and checks every pass's call lengths against it. Its shared prefix is cut
at other places than the direct run, so its hidden rows are not the direct run's: G3 then compares the Swift shared
rows with the Python reference's shared rows (`--shared-gate-fixture` / `--shared-gate-heldout`: readout_gate.py
--shared of the same asset; the shared p and answers with decide.py check --shared), and G2 adds the shared p's bar.
`--swift-dir` (before the command) moves the Swift work directory (the binary `<dir>/.build/release/kev`, `<dir>/gate`,
`<dir>/longrow`; `--bin` names another binary); `negative --out`, `longrow-python / longrow-score --bundle --tag` and
`pytime --warm` (Kev.warm_up before the items) keep a round's records apart from round 7's.

Run from conversion/kev with the shared venv (`longrow-oracle` with the oracle venv):
    PY=~/code/coreai/coreai-models/.venv/bin/python
    $PY gate_swift.py render
    $PY gate_swift.py score --model kev-0.8b --fixture <pass.json> --heldout <pass.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import work_path  # noqa: E402

LANE = work_path("_kev")
SWIFT = LANE / "swift"
BIN = SWIFT / ".build" / "release" / "kev"
PKG = REPO / "apps" / "Kev"
RESULTS = LANE / "results"
BUNDLES = {"kev-0.8b": LANE / "exports" / "bundles" / "kev_0_8b_decode_fp16_pf16",
           "kev-4b": LANE / "exports" / "bundles" / "kev_4b_decode_fp16_pf16"}
ORACLES = {("kev-0.8b", "fixture"): LANE / "oracle" / "records_oracle.json",
           ("kev-0.8b", "heldout"): LANE / "oracle" / "heldout" / "records_oracle.json",
           ("kev-4b", "fixture"): LANE / "oracle_4b" / "records_oracle.json",
           ("kev-4b", "heldout"): LANE / "oracle_4b" / "heldout" / "records_oracle.json"}
FIXTURES = {"fixture": LANE / "fixtures" / "records.json", "heldout": LANE / "fixtures" / "heldout.json"}
GATES = {("kev-0.8b", "fixture"): RESULTS / "readout_fp16_pf16.json",
         ("kev-0.8b", "heldout"): RESULTS / "readout_heldout_fp16_pf16.json",
         ("kev-4b", "fixture"): RESULTS / "readout_fp16_pf16_4b.json",
         ("kev-4b", "heldout"): RESULTS / "readout_heldout_fp16_pf16_4b.json"}
E2E = {("kev-0.8b", "fixture"): RESULTS / "e2e_kev-0.8b.json", ("kev-0.8b", "heldout"): RESULTS / "e2e_kev-0.8b_heldout.json",
       ("kev-4b", "fixture"): RESULTS / "e2e_kev-4b.json", ("kev-4b", "heldout"): RESULTS / "e2e_kev-4b_heldout.json"}
BAR = {"max_abs_dp": 0.02, "mean_of_run_mean_abs_dp": 0.002, "near_tie_top2_margin": 0.02}   # readout_gate.BAR
CHECKPOINTS = {"kev-0.8b": ("jaredpalmer/kev-0.8b@788ddbdd65715bb03a56788c822f6c632c9a551d",
                            "Qwen/Qwen3.5-0.8B-Base@dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"),
               "kev-4b": ("jaredpalmer/kev-4b@591dcb5bd6d05eb0b5131ea6608f93f10243335c",
                          "Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b")}
LONGROW = LANE / "fixtures" / "longrow.json"
SHARED_GATES: dict = {}   # round 15: (model, set) -> readout_gate.py --shared transcript of a dynamic-S bundle
LONGROW_ORACLE = {"kev-0.8b": LANE / "oracle" / "longrow", "kev-4b": LANE / "oracle_4b" / "longrow"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- longrow: the fixture
LONG_SOURCES = ("own_L01", "own_L02", "own_L03")
LONG_QUESTIONS = ("own_L02", ("cap", "subcontract"))         # one choice and one noul question of own_L02
LONG_TARGETS = {"longrow_4k": (4030, 4080), "longrow_3k": (2980, 3040)}


def cmd_longrow_fixture(args) -> int:
    """Rows near the graph's limit from the fixture's own long states: own_L01 + own_L02 + own_L03 joined with a blank
    line (own_L01 again when that is short), cut at a sentence end — or, when no sentence end does, at a word end — so
    that every row lands in the target range (the 4k window is 9 state tokens wide: 4,030 - 4,080 for a 20-token and a
    62-token branch)."""
    import host
    tok = host.load_tokenizer(BUNDLES["kev-0.8b"] / "tokenizer" / "tokenizer.json")
    recs = {r["id"]: r for r in json.loads(FIXTURES["fixture"].read_text())["records"]}
    parts = [recs[i]["request"]["state"] for i in LONG_SOURCES]
    assert all(isinstance(p, str) for p in parts)
    text = "\n\n".join(parts + [parts[0]])
    src_q = recs[LONG_QUESTIONS[0]]["request"]["questions"]
    questions = {q: src_q[q] for q in LONG_QUESTIONS[1]}
    out = []
    for rid, (lo, hi) in LONG_TARGETS.items():
        # sentence ends ('.' or newline) as cut points, then word ends; the first cut whose rows all fit [lo, hi]
        chosen = None
        for kind, marks in (("sentence", ".\n"), ("word", " ")):
            for cut in (i + 1 for i, c in enumerate(text) if c in marks and i > 1000):
                state = text[:cut].rstrip()
                req = {"model": "kev-0.8b", "state": state, "questions": questions}
                b = host.build_rows(req, tok)
                lens = [len(r["row_ids"]) for r in b["rows"]]
                if min(lens) >= lo and max(lens) <= hi:
                    host.graph_context_check(b["rows"], 4096, 16)
                    chosen = (state, b, lens, kind)
                    break
                if min(lens) > hi:
                    break
            if chosen:
                break
        if chosen is None:
            raise SystemExit(f"{rid}: no sentence cut puts every row in [{lo}, {hi}]")
        state, b, lens, kind = chosen
        out.append({"id": rid, "source": "longrow", "request": {"model": "kev-0.8b", "state": state, "questions": questions},
                    "gold": {}, "note": f"state = own_L01 + own_L02 + own_L03 (+ own_L01) joined by a blank line, cut at a "
                                        f"{kind} end to {b['state_len']} state tokens; questions = own_L02's "
                                        f"{', '.join(LONG_QUESTIONS[1])}; rows {lens} tokens (target {lo}-{hi})",
                    "provenance": {"state_from": list(LONG_SOURCES), "questions_from": LONG_QUESTIONS[0],
                                   "chars": len(state)},
                    "tokens": {"state": b["state_len"], "rows": lens, "input": b["input_tokens"]}})
        print(f"{rid}: state {b['state_len']} tokens, rows {lens}, {len(state)} chars")
    doc = {"schema": "kev-fixture/1", "role": "longrow diagnosis (round 7): the graph's longest rows",
           "records": out, "sources": {"own": str(FIXTURES["fixture"]), "sha256": sha256_file(FIXTURES["fixture"])}}
    LONGROW.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {LONGROW}")
    return 0


# --------------------------------------------------------------------------- longrow: the oracle (oracle venv)
def cmd_longrow_oracle(args) -> int:
    """oracle_kev.py on the longrow fixture, every row's fp32 hidden states kept (oracle_kev.HIDDEN_RECORDS is set to
    the fixture's ids for this process; the script itself is unchanged). Its own --out-dir / --results-dir."""
    os.environ.setdefault("HF_HOME", str(LANE / "hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import oracle_kev
    ids = [r["id"] for r in json.loads(LONGROW.read_text())["records"]]
    oracle_kev.HIDDEN_RECORDS = ids
    out = LONGROW_ORACLE[args.model]
    out.mkdir(parents=True, exist_ok=True)
    ck, base = CHECKPOINTS[args.model]
    sys.argv = ["oracle_kev.py", "--fixtures", str(LONGROW), "--out-dir", str(out), "--results-dir", str(out),
                "--threads", "1", "--checkpoint", ck, "--base", base, "--summary-name", "oracle_summary_longrow.json"]
    return oracle_kev.main()


# --------------------------------------------------------------------------- render (model-free)
EDGE_VALUES = [
    1250, 1250.0, 0.0, -0.0, 1e-07, 1e-05, 0.0001, 1e16, 1e15, 1234567890123456.0, 12345678901234567.0, 1.5e300,
    5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 0.1, 0.3, 1 / 3, 100.0, -12.4, 2 ** 70, -(10 ** 25), 10 ** 30,
    -0, True, False, None, "", "plain", 'quote " and \\ backslash / slash', "tab\tnewline\ncr\rbs\bff\f",
    "ctrl \x00\x01\x1f\x7f end", "unicode é 日本 😀  nbsp 　ideo é", "<|fim_suffix|> <|x|> <||> <|a b|>",
    "  lead", "\n  lead newline", [], {}, [[]], [{}], [[], {}, None, "", "  x"],
    {"b": 1, "a": [1, {"c": None, "d": [True, 2.5e-07]}], "e": {}, "f": [], "g": "  x", "h": {"i": {"j": [[1, [2]]]}}},
    ["  lead", "\ttab", None, ["x", "y"], {"k": "v", "n": [1, 2]}, [], {}],
    {"sp ace": {"x": ""}, "uni ¦": "日本語", "<|k|>": "<|v|>", "list": ["\n  lead newline", [], {}, True, None, [1, [2, {"k": [3]}]]]},
    {"option_id": "x", "description": {"covers": "parts", "example": "a hinge"}},
]
RAW_LITERALS = ['1.50', '1E5', '-0', '-0.0', '1e400', '-1e400', '1e-400', '1.0e-5', '123456789012345678901234567890', '0.1e1',
                '{"a": 1, "a": 2}', '{"b": 0, "a": {"d": 1.10, "c": [1e2, 2E-3]}}', 'NaN', 'Infinity', '-Infinity',
                '"\\u00e9\\ud83d\\ude00\\/"', '[1, 1.0, 1.00, 1e0, 10e-1]', '{"\\u0000": 1, " ": 2}', '[1e16, 1e-7, 100000000000000000000.0]']
TOKEN_TEXTS = ["<|fim_prefix|>", "a <|fim_suffix|> b <|im_end|>", "<|x|>", "<||>", "<|a b|>", "<|日本|>", "<|a|", "<|<|a|>|>",
               "¦", "<¦x¦>", "<|endoftext|><|box_start|>", "日本語 é", "  leading spaces", "trailing   ", "\n\n", "\r\n",
               "tab\tx", "😀👍🏽", "é", "<|a|>́", "12345 3.14159 -0.0 1e-07", "It's we'll they're", "ALL CAPS'S",
               "a" + " " * 20 + "b", " nbsp", "　ideographic", "x y z", "", " ", "<think>no</think>",
               "{\"type\": \"noul\", \"noul\": 0.1234}"]


def render_inputs() -> dict:
    """Every JSON value the record renders (fixture + held-out), every request (and test_host.py's table), the edge
    values, raw literals, doubles, json.dumps inputs and tokenizer texts."""
    import random
    import struct as st

    import numpy as np
    import test_host
    values, where, requests, rwhere = [], [], [], []
    for set_name, path in FIXTURES.items():
        for r in json.loads(path.read_text())["records"]:
            req = r["request"]
            requests.append(json.dumps(req, ensure_ascii=False))
            rwhere.append(f"{set_name}/{r['id']}")
            values.append(json.dumps(req["state"], ensure_ascii=False))
            where.append(f"{set_name}/{r['id']}/state")
            for qid, q in req["questions"].items():
                values.append(json.dumps(q.get("instructions"), ensure_ascii=False))
                where.append(f"{set_name}/{r['id']}/{qid}/instructions")
                c = q.get("criteria")
                for k, v in (c.items() if isinstance(c, dict) else enumerate(c) if isinstance(c, list) else []):
                    values.append(json.dumps(v, ensure_ascii=False))
                    where.append(f"{set_name}/{r['id']}/{qid}/criteria/{k}")
    for name, req in {**test_host.VALID, **test_host.INVALID}.items():
        requests.append(json.dumps(req, ensure_ascii=False))
        rwhere.append(f"table/{name}")
    for i, v in enumerate(EDGE_VALUES):
        values.append(json.dumps(v, ensure_ascii=False))
        where.append(f"edge/{i}")
        requests.append(json.dumps({"state": v, "questions": {"a": {"type": "choice", "instructions": v, "criteria": {"x": v, "y": None}},
                                                              "b": {"type": "score", "criteria": [v, "lvl"]},
                                                              "c": {"type": "noul", "criteria": {"true": v, "false": ""}}}},
                                   ensure_ascii=False))
        rwhere.append(f"edge_request/{i}")
    for t in RAW_LITERALS:
        values.append(t)
        where.append(f"raw/{t}")
    rng = random.Random(7)
    doubles = []
    for oracle in ORACLES.values():
        for e in json.loads(oracle.read_text())["records"]:
            for q in e["questions"]:
                doubles += [float(x) for x in q["probs"]]
    doubles += [float(np.float32(rng.random())) for _ in range(100_000)]
    doubles += [rng.random() * rng.choice([1, 2, 3, 4, 6, 9, 254]) for _ in range(50_000)]
    doubles += [st.unpack("<d", st.pack("<Q", rng.getrandbits(63)))[0] * rng.choice([1, -1]) for _ in range(50_000)]
    doubles += [0.5, 0.25, 0.00005, 0.00015, 0.00025, 0.12345, 0.99995, 2.675, 1.0000500000000001, 0.3, 1e-07, 1e16, -0.0,
                0.0, 1e22, 5e-324, 0.05, 0.15, 0.25, 0.35, 0.45, 1234.56785, float("nan"), float("inf"), float("-inf")]
    dumps = []
    for oracle in ORACLES.values():
        for e in json.loads(oracle.read_text())["records"]:
            dumps.append(json.dumps(e["response"]["answers"], ensure_ascii=False))
    dumps += [json.dumps(v, ensure_ascii=False) for v in EDGE_VALUES] + RAW_LITERALS
    dumps.append(json.dumps({"é": "日本 😀", "ctrl": "\x00\x01\x1f\x7f", "q": 'a"b\\c/d', "nums": [1, 1.0, -0.0, 1e-07, 1e16, 2 ** 70, 0.1],
                             "nested": {"a": [], "b": {}}, "bool": [True, False, None], "sep": "  "}, ensure_ascii=False))
    return {"values": values, "where": where, "requests": requests, "rwhere": rwhere, "doubles": doubles, "dumps": dumps,
            "texts": TOKEN_TEXTS}


def py_request(text: str) -> dict:
    """host.validate_request + host.to_record on one request text, in render-test's output shape."""
    import host
    try:
        req = host.validate_request(json.loads(text))
    except ValueError as e:
        return {"accept": False, "error": str(e)}
    rec, meta = host.to_record(req)
    return {"accept": True, "model": req["model"], "state": rec["state"],
            "questions": [{"instr": q["instr"], "options": q["options"]} for q in rec["questions"]],
            "meta": [{"id": m["id"], "type": m["type"], "keys": m["keys"], "legend": m.get("legend")} for m in meta]}


def diff(a: list, b: list, labels: list | None = None) -> dict:
    bad = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return {"n": len(b), "equal": len(b) - len(bad) if len(a) == len(b) else 0, "count_match": len(a) == len(b),
            "first_differences": [{"i": i, "where": labels[i] if labels else None, "swift": a[i], "python": b[i]} for i in bad[:8]]}


def cmd_render(args) -> int:
    import host
    work = SWIFT / "checks"
    work.mkdir(parents=True, exist_ok=True)
    inp = render_inputs()
    src = work / "render_in.json"
    src.write_text(json.dumps({k: inp[k] for k in ("values", "requests", "doubles", "dumps", "texts")}, ensure_ascii=False))
    out = work / "render_out.json"
    subprocess.run([str(BIN), "render-test", "--in", str(src), "--out", str(out), "--bundle", str(BUNDLES["kev-0.8b"])], check=True)
    got = json.loads(out.read_text())
    tok = host.load_tokenizer(BUNDLES["kev-0.8b"] / "tokenizer" / "tokenizer.json")
    want = {
        "render": [host.render(json.loads(t)) for t in inp["values"]],
        "requests": [py_request(t) for t in inp["requests"]],
        "str": [str(d) for d in inp["doubles"]],
        "repr": [json.dumps(d) for d in inp["doubles"]],
        "round4": [json.dumps(round(d, 4)) for d in inp["doubles"]],
        "round1": [json.dumps(round(d, 1)) for d in inp["doubles"]],
        "dumps": [json.dumps(json.loads(t)) for t in inp["dumps"]],
        "rewrite": [host._SPECIAL_RE.sub("<¦\\1¦>", t) for t in inp["texts"]],
        "user_tokens": [host.user_tokens(tok, t) for t in inp["texts"]],
        "plain_tokens": [host.token_ids(tok, t) for t in inp["texts"]],
    }
    labels = {"render": inp["where"], "requests": inp["rwhere"], "rewrite": inp["texts"], "user_tokens": inp["texts"],
              "plain_tokens": inp["texts"]}
    rep = {"schema": "kev-swift-render/1", "generated_at": now(),
           "what": "apps/Kev (render, option texts, keys, the request checks, Python str / repr / round, json.dumps, "
                   "user_tokens) through `kev render-test` vs host.py / CPython "
                   f"{sys.version.split()[0]} / tokenizers on the same inputs",
           "inputs": {"values": len(inp["values"]), "requests": len(inp["requests"]), "doubles": len(inp["doubles"]),
                      "dumps": len(inp["dumps"]), "texts": len(inp["texts"]),
                      "requests_accepted_by_host": sum(w["accept"] for w in want["requests"])},
           "binary_sha256": sha256_file(BIN)}
    # a rejected request is compared on the decision; its message (KevError prints a "request: " prefix) apart
    msgs = [(g.get("error", "").removeprefix("request: "), w.get("error")) for g, w in zip(got["requests"], want["requests"])
            if not w["accept"]]
    got["requests"] = [g if g["accept"] else {"accept": False} for g in got["requests"]]
    want["requests"] = [w if w["accept"] else {"accept": False} for w in want["requests"]]
    for k in want:
        rep[k] = diff(got[k], want[k], labels.get(k))
    rep["requests"]["rejected"] = len(msgs)
    rep["requests"]["rejected_messages_equal"] = sum(a == b for a, b in msgs)
    rep["requests"]["rejected_message_differences"] = [{"swift": a, "python": b} for a, b in msgs if a != b][:8]
    rep["pass"] = all(rep[k]["equal"] == rep[k]["n"] for k in want)
    path = RESULTS / "swift_render_test.json"
    path.write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    for k in want:
        print(f"{k}: {rep[k]['equal']}/{rep[k]['n']}", rep[k]["first_differences"][:2] if rep[k]["first_differences"] else "")
    print(f"{'PASS' if rep['pass'] else 'FAIL'} -> {path}")
    return 0 if rep["pass"] else 1


# --------------------------------------------------------------------------- score
def g1_record(s: dict, o: dict) -> tuple[list[bool], list[str]]:
    """G1 for one record: per row, the Swift ids / indices / keys == the oracle's (the packed ids, the state length and
    input_tokens of the record included)."""
    if "error" in s:
        return [False] * len(o["questions"]), [f"{o['id']}: {s['error']}"]
    rec_ok = (s["packed_ids"] == o["ids"] and s["state_len"] == o["state_tokens"]
              and s["input_tokens"] == o["response"]["usage"]["input_tokens"] and len(s["rows"]) == len(o["questions"]))
    oks, bad = [], []
    names = ("qid", "type", "keys", "row_ids", "row_len", "decide", "opts")
    for k, q in enumerate(o["questions"]):
        r = s["rows"][k] if k < len(s["rows"]) else {}
        diffs = [n for n in names if r.get(n) != q[n]]
        ok = rec_ok and not diffs
        oks.append(ok)
        if not ok:
            bad.append(f"{o['id']}:q{k}: " + (", ".join(diffs) if diffs else "packed ids / state / input_tokens"))
    return oks, bad


def run_rows(model: str, set_name: str, records: Path | None = None, tag: str | None = None) -> dict:
    out = SWIFT / "gate" / f"rows_{tag or model + '_' + set_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(BIN), "rows", "--bundle", str(BUNDLES[model]), "--records", str(records or FIXTURES[set_name]),
           "--oracle", str(ORACLES[(model, set_name)]), "--out", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    return json.loads(out.read_text())


def score_set(model: str, set_name: str, sw_path: Path, only: list[str] | None = None) -> dict:
    import numpy as np

    import decide
    import host
    sw = json.loads(sw_path.read_text())
    oracle = {r["id"]: r for r in json.loads(ORACLES[(model, set_name)].read_text())["records"]}
    if only:   # round 15: a subset of the records (a short window's pass)
        oracle = {i: oracle[i] for i in only}
    gate = json.loads(GATES[(model, set_name)].read_text())
    gate_rows = {(r["id"], r["q"]): r for r in gate["runs"] if r.get("variant", "base") == "base"}
    npz_of = {p["shard"]: p["npz"] for p in gate["processes"] if "shard" in p}
    e2e_doc = json.loads(E2E[(model, set_name)].read_text())
    e2e = {r["id"]: r for r in e2e_doc["records"]}
    head = host.load_head(BUNDLES[model] / "head")
    recs = {r["id"]: r for r in sw["records"]}
    model_free = {r["id"]: r for r in run_rows(model, set_name)["records"]}
    shape = host.graph_shape(json.loads((BUNDLES[model] / "metadata.json").read_text())["language"])
    dyn = shape["dynamic"]
    sgate_path = SHARED_GATES.get((model, set_name)) if dyn else None
    sgate_rows, sgate_npz = {}, {}
    if sgate_path:   # round 15: readout_gate.py --shared of the same asset (the Python reference's shared rows)
        sgate = json.loads(sgate_path.read_text())
        sgate_rows = {(r["id"], r["q"]): r for r in sgate["runs"] if r.get("variant", "base") == "base"}
        sgate_npz = {p["shard"]: p["npz"] for p in sgate["processes"] if "shard" in p}

    def lengths(T: int) -> list[int]:
        return [c for c, _ in host.plan(T, shape["cap"], shape["q"], shape["qmin"])]
    plan_eq = {"direct": 0, "shared": 0, "records": 0}
    npzs: dict = {}
    g1_ok, g1_bad, items, rows_out = [], [], [], []
    ans_eq = usage_eq = sh_ans_eq = sh_usage_eq = mf_ans = mf_out = mf_g1 = 0
    for rid, o in oracle.items():
        s = recs.get(rid)
        if s is None:
            g1_bad.append(f"{rid}: missing from the Swift pass")
            g1_ok += [False] * len(o["questions"])
            continue
        oks, bad = g1_record(s, o)
        g1_ok += oks
        g1_bad += bad
        mo, _ = g1_record(model_free[rid], o)
        mf_g1 += all(mo)
        mf_ans += model_free[rid].get("answers_json_from_oracle_p") == json.dumps(o["response"]["answers"])
        mf_out += model_free[rid].get("output_tokens_from_oracle_p") == o["response"]["usage"]["output_tokens"]
        e = e2e[rid]
        if "direct" in s and "call_lengths" in s["direct"]:   # round 15: the calls = host.plan's
            plan_eq["records"] += 1
            Ts = [len(q["row_ids"]) for q in o["questions"]]
            plan_eq["direct"] += s["direct"]["call_lengths"] == [c for T in Ts for c in lengths(T)]
            if "shared" in s:
                sp = host.shared_prefix_plan(o["state_tokens"], shape["cap"], shape["q"], shape["qmin"])
                k0 = sp["shared_tokens"]
                want = (lengths(k0) + [c for T in Ts for c in lengths(T - k0)]) if sp["k"] else [c for T in Ts for c in lengths(T)]
                plan_eq["shared"] += s["shared"]["call_lengths"] == want
        for k, (r, q) in enumerate(zip(s["rows"], o["questions"])):
            g = gate_rows[(rid, k)]
            npz = npz_of[g["shard"]]
            if npz not in npzs:
                npzs[npz] = np.load(npz)
            hg = npzs[npz][f"{g['npz_key']}__hidden"]
            hidden_eq = r["hidden_sha256"] == hashlib.sha256(np.ascontiguousarray(hg).tobytes()).hexdigest()
            zn = host.head_logits(hg[r["decide"]], hg[r["opts"]], head)
            er = e["rows"][k]
            p_sw = np.asarray(r["p_bits"], np.uint32).view(np.float32)
            p_py = np.asarray(er["p_host"], np.float32)
            ps_sw = np.asarray(r["shared_p_bits"], np.uint32).view(np.float32) if "shared_p_bits" in r else None
            ps_py = np.asarray(er["p_host_shared"], np.float32) if "p_host_shared" in er else None
            row = {"row": f"{rid}:q{k}", "T": r["row_len"], "type": q["type"], "near_tie": bool(q["near_tie"]),
                   "g1": oks[k], "hidden_sha256_equal_python": hidden_eq,
                   "finite": bool(r["hidden_finite"]) and not r["hidden_all_zero"],
                   "z_max_abs_diff_numpy_float64": float(np.max(np.abs(np.asarray(r["logits"], np.float64) - zn))),
                   "p_bit_equal_python_host": bool(np.array_equal(p_sw.view(np.uint32), p_py.view(np.uint32))),
                   "p_max_abs_diff_python_host": float(np.max(np.abs(p_sw.astype(np.float64) - p_py.astype(np.float64)))),
                   "p_max_abs_diff_gate_torch_head": float(np.max(np.abs(p_sw.astype(np.float64) - np.asarray(g["probs"], np.float64))))}
            if ps_sw is not None:
                row.update({"shared_hidden_bit_equal_direct": bool(r["shared_hidden_bit_equal_direct"]),
                            "shared_p_bit_equal_direct": bool(r["shared_p_bit_equal_direct"]),
                            "shared_p_bit_equal_python_host_shared": bool(ps_py is not None and np.array_equal(ps_sw.view(np.uint32), ps_py.view(np.uint32)))})
                if "prepared_p_bits" in r:   # round 15: the prepared state = the shared run (Swift), = Python's prepared p
                    pp_py = np.asarray(er["p_host_prepared"], np.float32) if "p_host_prepared" in er else None
                    row.update({"prepared_hidden_bit_equal_shared": bool(r["prepared_hidden_bit_equal_shared"]),
                                "prepared_p_bit_equal_shared": bool(r["prepared_p_bit_equal_shared"]),
                                "prepared_single_p_bit_equal_shared": bool(r["prepared_single_p_bit_equal_shared"]),
                                "prepared_p_bit_equal_python_prepared": (bool(np.array_equal(np.asarray(r["prepared_p_bits"], np.uint32),
                                                                                             pp_py.view(np.uint32)))
                                                                         if pp_py is not None else None)})
                if (rid, k) in sgate_rows:   # round 15: the shared rows of the Python reference (same asset, same plan)
                    gs = sgate_rows[(rid, k)]
                    znpz = sgate_npz[gs["shard"]]
                    if znpz not in npzs:
                        npzs[znpz] = np.load(znpz)
                    hs = npzs[znpz][f"{gs['npz_key']}__hidden"]
                    row["shared_hidden_sha256_equal_python_shared"] = (
                        r.get("shared_hidden_sha256") == hashlib.sha256(np.ascontiguousarray(hs).tobytes()).hexdigest())
            rows_out.append(row)
            items.append({"row": row["row"], "probs_oracle": q["probs"], "near_tie": bool(q["near_tie"]),
                          "argmax_oracle": q["keys"].index(q["argmax"]), "p_swift": p_sw.astype(np.float64).tolist(),
                          "p_python_host": p_py.astype(np.float64).tolist(),
                          **({"p_swift_shared": ps_sw.astype(np.float64).tolist()} if ps_sw is not None else {})})
        ans_eq += s["direct"]["answers_json"] == json.dumps(e["direct"]["response"]["answers"])
        usage_eq += s["direct"]["response"]["usage"] == e["direct"]["response"]["usage"]
        if "shared" in s and "shared" in e:
            sh_ans_eq += s["shared"]["answers_json"] == json.dumps(e["shared"]["response"]["answers"])
            sh_usage_eq += s["shared"]["response"]["usage"] == e["shared"]["response"]["usage"]
    n_rows, n_rec = len(rows_out), len(oracle)
    bars = {"swift": decide.bar_summary(items, "p_swift"), "python_host": decide.bar_summary(items, "p_python_host")}
    if all("p_swift_shared" in it for it in items):
        bars["swift_shared"] = decide.bar_summary(items, "p_swift_shared")
    gs = gate["summary"]
    keys = ("questions_non_near_tie", "argmax_equal_non_near_tie", "near_tie_questions", "argmax_equal_near_tie",
            "max_abs_dp", "mean_of_run_mean_abs_dp")
    reset = sw.get("reset_check", {})
    summ = {
        "records": n_rec, "rows": n_rows,
        "G1_rows_equal_oracle": sum(g1_ok), "G1_failures": g1_bad[:20],
        "G1_model_free": {"records_all_rows_equal": mf_g1, "answers_from_oracle_p_byte_equal": mf_ans,
                          "output_tokens_from_oracle_p_equal": mf_out},
        "G2_bar_swift": {k: bars["swift"][k] for k in (*keys, "worst_row", "pass")},
        "G2_bar_python_host_e2e": {k: bars["python_host"][k] for k in keys},
        "G2_bar_readout_gate": {k: gs[k] for k in keys},
        "G2_swift_equals_python_host_bar": all(bars["swift"][k] == bars["python_host"][k] for k in keys),
        "G2_finite_rows": sum(r["finite"] for r in rows_out),
        "G2_reset_bit_equal": bool(reset.get("hidden_bit_equal") and reset.get("p_bit_equal")),
        "G3_hidden_sha256_equal_python": sum(r["hidden_sha256_equal_python"] for r in rows_out),
        "G3_p_bit_equal_python_host": sum(r["p_bit_equal_python_host"] for r in rows_out),
        "G3_p_max_abs_diff_python_host": max(r["p_max_abs_diff_python_host"] for r in rows_out),
        "G3_p_max_abs_diff_gate_torch_head": max(r["p_max_abs_diff_gate_torch_head"] for r in rows_out),
        "G3_z_max_abs_diff_numpy_float64": max(r["z_max_abs_diff_numpy_float64"] for r in rows_out),
        "G3_answers_json_byte_equal_python_host": ans_eq, "G3_usage_equal_python_host": usage_eq,
    }
    if "swift_shared" in bars:
        summ.update({
            "G3_shared_hidden_bit_equal_direct": sum(r.get("shared_hidden_bit_equal_direct", False) for r in rows_out),
            "G3_shared_p_bit_equal_direct": sum(r.get("shared_p_bit_equal_direct", False) for r in rows_out),
            "G3_shared_p_bit_equal_python_host_shared": sum(r.get("shared_p_bit_equal_python_host_shared", False) for r in rows_out),
            "G3_shared_answers_json_equal_direct": sum(bool(x.get("shared_answers_json_equal_direct")) for x in sw["records"]),
            "G3_shared_answers_json_byte_equal_python_host_shared": sh_ans_eq, "G3_shared_usage_equal_python_host": sh_usage_eq,
            "G2_bar_swift_shared": {k: bars["swift_shared"][k] for k in (*keys, "pass")}})
    checks = {"G1": summ["G1_rows_equal_oracle"] == n_rows and mf_g1 == n_rec and mf_ans == n_rec and mf_out == n_rec,
              "G2": bars["swift"]["pass"] and summ["G2_finite_rows"] == n_rows and summ["G2_reset_bit_equal"],
              "G3_hidden": summ["G3_hidden_sha256_equal_python"] == n_rows,
              "G3_p": summ["G3_p_bit_equal_python_host"] == n_rows,
              "G3_answers": ans_eq == n_rec and usage_eq == n_rec}
    pr_rows = [r for r in rows_out if "prepared_hidden_bit_equal_shared" in r]
    if pr_rows:   # round 15: the prepared state
        py_pr = [r for r in pr_rows if r["prepared_p_bit_equal_python_prepared"] is not None]
        pr_recs = [x for x in sw["records"] if "prepared" in x]
        summ["G4_prepared"] = {"rows": len(pr_rows),
                               "hidden_bit_equal_shared": sum(r["prepared_hidden_bit_equal_shared"] for r in pr_rows),
                               "p_bit_equal_shared": sum(r["prepared_p_bit_equal_shared"] for r in pr_rows),
                               "single_question_p_bit_equal_shared": sum(r["prepared_single_p_bit_equal_shared"] for r in pr_rows),
                               "p_bit_equal_python_prepared": f"{sum(r['prepared_p_bit_equal_python_prepared'] for r in py_pr)}/{len(py_pr)}",
                               "records_answers_equal_shared": f"{sum(x['prepared']['answers_json_equal_shared'] for x in pr_recs)}/{len(pr_recs)}",
                               "records_usage_equal_shared": f"{sum(x['prepared']['usage_equal_shared'] for x in pr_recs)}/{len(pr_recs)}"}
    summ["plan"] = {"graph": shape, "records_with_call_lengths": plan_eq["records"],
                    "direct_call_lengths_equal_host_plan": plan_eq["direct"], "shared_call_lengths_equal_host_plan": plan_eq["shared"]}
    if plan_eq["records"]:
        checks["plan"] = plan_eq["direct"] == n_rec and ("swift_shared" not in bars or plan_eq["shared"] == n_rec)
    if pr_rows:
        checks["G4_prepared"] = (all(r["prepared_hidden_bit_equal_shared"] and r["prepared_p_bit_equal_shared"]
                                     and r["prepared_single_p_bit_equal_shared"] for r in pr_rows)
                                 and all(r["prepared_p_bit_equal_python_prepared"] for r in py_pr)
                                 and all(x["prepared"]["answers_json_equal_shared"] and x["prepared"]["usage_equal_shared"] for x in pr_recs))
    if "swift_shared" in bars and not dyn:
        checks["G3_shared"] = (summ["G3_shared_hidden_bit_equal_direct"] == n_rows and summ["G3_shared_p_bit_equal_direct"] == n_rows
                               and summ["G3_shared_answers_json_equal_direct"] == n_rec)
    elif "swift_shared" in bars:   # round 15: shared = the Python reference's shared run, and on the oracle's bar
        sh_rows = [r for r in rows_out if "shared_hidden_sha256_equal_python_shared" in r]
        summ.update({"G3_shared_hidden_sha256_equal_python_shared": f"{sum(r['shared_hidden_sha256_equal_python_shared'] for r in sh_rows)}/{len(sh_rows)}",
                     "G2_shared_bar_pass": bool(bars["swift_shared"]["pass"])})
        checks["G3_shared"] = (len(sh_rows) == n_rows and all(r["shared_hidden_sha256_equal_python_shared"] for r in sh_rows)
                               and summ["G3_shared_p_bit_equal_python_host_shared"] == n_rows
                               and sh_ans_eq == n_rec and sh_usage_eq == n_rec)
        checks["G2_shared"] = bool(bars["swift_shared"]["pass"])
    return {"set": set_name, "pass_json": str(sw_path), "pass_sha256": sha256_file(sw_path), "label": sw.get("label"),
            "assets": sw.get("assets"), "loaded": sw.get("loaded"), "load": sw.get("load"), "load_wall_s": sw.get("load_wall_s"),
            "reset_check": reset, "runs_wall_s": sw.get("runs_wall_s"), "warm_up": sw.get("warm_up"),
            "shared_gate": str(sgate_path) if sgate_path else None,
            "footprint": {k: sw.get(k) for k in ("footprint_start_bytes", "footprint_after_load_bytes", "footprint_bytes_min_max",
                                                 "footprint_end_bytes")},
            "environment": sw.get("environment"),
            "references": {"oracle": str(ORACLES[(model, set_name)]), "gate": str(GATES[(model, set_name)]),
                           "gate_sha256": sha256_file(GATES[(model, set_name)]), "e2e": str(E2E[(model, set_name)]),
                           "e2e_sha256": sha256_file(E2E[(model, set_name)])},
            "summary": summ, "checks": checks, "result": "PASS" if all(checks.values()) else "FAIL", "rows": rows_out}


def package_record() -> dict:
    files = sorted(p for p in PKG.rglob("*") if p.is_file() and ".build" not in p.parts and ".swiftpm" not in p.parts)
    rec = {"path": str(PKG), "files_sha256": {str(p.relative_to(PKG)): sha256_file(p) for p in files}}
    resolved = PKG / "Package.resolved"
    if resolved.exists():
        rec["resolved"] = {p["identity"]: p["state"].get("version") or p["state"].get("revision")
                           for p in json.loads(resolved.read_text()).get("pins", [])}
    if BIN.exists():
        rec["binary"] = {"path": str(BIN), "sha256": sha256_file(BIN), "bytes": BIN.stat().st_size}
    return rec


def cmd_score(args) -> int:
    if args.bundle:   # round 11: another form of the same checkpoint, scored against its own gate / e2e transcripts
        BUNDLES[args.model] = Path(args.bundle).expanduser().resolve()
        for set_name, g in (("fixture", args.shared_gate_fixture), ("heldout", args.shared_gate_heldout)):
            if g:   # round 15
                SHARED_GATES[(args.model, set_name)] = Path(g).expanduser().resolve()
        for set_name, g, e in (("fixture", args.gate_fixture, args.e2e_fixture), ("heldout", args.gate_heldout, args.e2e_heldout)):
            if g:
                GATES[(args.model, set_name)] = Path(g).expanduser().resolve()
            if e:
                E2E[(args.model, set_name)] = Path(e).expanduser().resolve()
    rec = {"schema": "kev-swift-gate/1", "model": args.model, "generated_at": now(),
           "gate": "the Swift host (apps/Kev, `kev fixture`) vs the author's fp32 oracle (G1 ids / indices / keys, G2 the "
                   "readout gate's bar) and vs the Python reference on the same AOT asset (G3: the readout gate's hidden "
                   "rows, decide.py check's host p and answers)",
           "bar": {**BAR, "source": "readout_gate.BAR / decide.bar_summary (unchanged)"}, "package": package_record(), "sets": {}}
    for set_name, path in (("fixture", args.fixture), ("heldout", args.heldout)):
        if not path:
            continue
        p = score_set(args.model, set_name, Path(path), args.records.split(",") if args.records else None)
        rec["sets"][set_name] = p
        s = p["summary"]
        print(f"{args.model} {set_name} {p['result']}: G1 {s['G1_rows_equal_oracle']}/{s['rows']} (model-free {s['G1_model_free']}), "
              f"G2 {s['G2_bar_swift']}, finite {s['G2_finite_rows']}, reset {s['G2_reset_bit_equal']}")
        print(f"   G3 hidden {s['G3_hidden_sha256_equal_python']}/{s['rows']}, p bit {s['G3_p_bit_equal_python_host']}/{s['rows']} "
              f"(max {s['G3_p_max_abs_diff_python_host']:.3g}; z vs numpy {s['G3_z_max_abs_diff_numpy_float64']:.3g}), answers "
              f"{s['G3_answers_json_byte_equal_python_host']}/{s['records']}, usage {s['G3_usage_equal_python_host']}/{s['records']}"
              + (f", shared = direct hidden {s['G3_shared_hidden_bit_equal_direct']} p {s['G3_shared_p_bit_equal_direct']} answers "
                 f"{s['G3_shared_answers_json_equal_direct']}, shared p = python shared {s['G3_shared_p_bit_equal_python_host_shared']}"
                 if "G3_shared_hidden_bit_equal_direct" in s else ""))
        for k, ok in p["checks"].items():
            if not ok:
                print(f"   check {k}: FAIL")
        for f in s["G1_failures"][:5]:
            print("   ", f)
    rec["result"] = "PASS" if all(p["result"] == "PASS" for p in rec["sets"].values()) else "FAIL"
    if args.bundle:
        rec["bundle_override"] = {"bundle": str(BUNDLES[args.model]),
                                  "gates": {k[1]: str(v) for k, v in GATES.items() if k[0] == args.model},
                                  "e2e": {k[1]: str(v) for k, v in E2E.items() if k[0] == args.model}}
    out = Path(args.transcript or RESULTS / f"swift_gate_{args.model}.json")
    if args.bundle and out.exists():
        raise SystemExit(f"{out} exists: records are never overwritten")
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    print(f"{rec['result']} -> {out}")
    return 0 if rec["result"] == "PASS" else 1


# --------------------------------------------------------------------------- negative control
NEGATIVE_RECORD, NEGATIVE_QUESTION = "own_t01", "team"


def cmd_negative(args) -> int:
    import copy
    recs = {r["id"]: r for r in json.loads(FIXTURES["fixture"].read_text())["records"]}
    changed = copy.deepcopy(recs[NEGATIVE_RECORD])
    q = changed["request"]["questions"][NEGATIVE_QUESTION]
    words = q["instructions"].split(" ")
    words[1] = "zqxvortmund"                                  # "Which team ..." -> "Which zqxvortmund ..."
    q["instructions"] = " ".join(words)
    path = SWIFT / "gate" / "negative_records.json"
    path.write_text(json.dumps({"records": [changed, recs[NEGATIVE_RECORD]]}, ensure_ascii=False) + "\n")
    rows = {r["id"]: r for r in []}
    if getattr(args, "bundle", None):   # round 15
        BUNDLES["kev-0.8b"] = Path(args.bundle).expanduser().resolve()
    out = run_rows("kev-0.8b", "fixture", records=path, tag="negative")
    oracle = {r["id"]: r for r in json.loads(ORACLES[("kev-0.8b", "fixture")].read_text())["records"]}[NEGATIVE_RECORD]
    res = []
    for r in out["records"]:
        oks, bad = g1_record(r, oracle)
        res.append({"which": "changed" if r is out["records"][0] else "unchanged", "rows_equal_oracle": sum(oks),
                    "rows": len(oks), "messages": bad[:4],
                    "answers_from_oracle_p_equal": r.get("answers_json_from_oracle_p") == json.dumps(oracle["response"]["answers"])})
    del rows
    rec = {"schema": "kev-swift-negative/1", "generated_at": now(), "record": NEGATIVE_RECORD, "question": NEGATIVE_QUESTION,
           "changed_instructions": q["instructions"], "records_json": str(path), "results": res,
           "red": res[0]["rows_equal_oracle"] < res[0]["rows"] and res[1]["rows_equal_oracle"] == res[1]["rows"]}
    dest = Path(args.out) if getattr(args, "out", None) else RESULTS / "swift_negative_control.json"
    if getattr(args, "out", None) and dest.exists():
        raise SystemExit(f"{dest} exists: records are never overwritten")
    rec.update({"bundle": str(BUNDLES["kev-0.8b"]), "binary": str(BIN)})
    dest.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: rec[k] for k in ("changed_instructions", "results", "red")}, ensure_ascii=False))
    return 0 if rec["red"] else 1


# --------------------------------------------------------------------------- JIT vs AOT
JIT_RECORDS = ["tv4_000", "tv4_001", "tv4x_emotion_00", "tv4x_qnli_00", "tv4s_00", "semif_a3f18f3a63d45345942b", "own_t01",
               "own_j03", "own_m01", "own_L02"]


def cmd_jit(args) -> int:
    import numpy as np

    import decide
    aot, jit = json.loads(Path(args.aot).read_text()), json.loads(Path(args.jit).read_text())
    oracle = {r["id"]: r for r in json.loads(ORACLES[("kev-0.8b", "fixture")].read_text())["records"]}
    a = {r["id"]: r for r in aot["records"]}
    rows, items = [], []
    for r in jit["records"]:
        ra = a[r["id"]]
        for k, (x, y) in enumerate(zip(r["rows"], ra["rows"])):
            row = {"row": f"{r['id']}:q{k}", "T": x["row_len"], "hidden_sha256_equal": x["hidden_sha256"] == y["hidden_sha256"],
                   "p_bit_equal": x["p_bits"] == y["p_bits"], "finite": bool(x["hidden_finite"]) and not x["hidden_all_zero"],
                   "p_max_abs_diff": float(np.max(np.abs(np.asarray(x["p"]) - np.asarray(y["p"]))))}
            if not row["hidden_sha256_equal"] and x.get("hidden_dump") and y.get("hidden_dump"):
                hx = np.fromfile(x["hidden_dump"], np.float16).astype(np.float32)
                hy = np.fromfile(y["hidden_dump"], np.float16).astype(np.float32)
                row["hidden_max_abs_diff"] = float(np.max(np.abs(hx - hy)))
            rows.append(row)
            q = oracle[r["id"]]["questions"][k]
            items.append({"row": row["row"], "probs_oracle": q["probs"], "near_tie": bool(q["near_tie"]),
                          "argmax_oracle": q["keys"].index(q["argmax"]), "p_jit": x["p"], "p_aot": y["p"]})
    rec = {"schema": "kev-swift-jit/1", "generated_at": now(), "model": "kev-0.8b",
           "what": "the bundle's .aimodel specialized by Swift (GPU preferred + expectFrequentReshapes, the exporter's AOT "
                   "flags) vs the AOT .aimodelc (SpecializationOptions.default), same records, same binary",
           "jit_pass": args.jit, "aot_pass": args.aot,
           "records": len(jit["records"]), "rows": len(rows),
           "hidden_sha256_equal": sum(r["hidden_sha256_equal"] for r in rows),
           "p_bit_equal": sum(r["p_bit_equal"] for r in rows),
           "p_max_abs_diff": max(r["p_max_abs_diff"] for r in rows),
           "hidden_max_abs_diff_where_unequal": max((r.get("hidden_max_abs_diff", 0.0) for r in rows), default=0.0),
           "finite_rows": sum(r["finite"] for r in rows),
           "bar_vs_oracle": {"jit": decide.bar_summary(items, "p_jit"), "aot": decide.bar_summary(items, "p_aot")},
           "load": {"jit": {k: jit.get(k) for k in ("load", "load_wall_s")}, "aot": {k: aot.get(k) for k in ("load", "load_wall_s")}},
           "runs_wall_s": {"jit": jit.get("runs_wall_s"), "aot": aot.get("runs_wall_s")},
           "mpsgraph_scratch": {"jit": {k: jit.get(k) for k in ("mpsgraph_scratch_before_load", "mpsgraph_scratch_after_load", "mpsgraph_scratch_end")},
                                "aot": {k: aot.get(k) for k in ("mpsgraph_scratch_before_load", "mpsgraph_scratch_after_load", "mpsgraph_scratch_end")}},
           "coreai_cache": {"jit": {k: jit.get(k) for k in ("coreai_cache_before_load", "coreai_cache_after_load", "coreai_cache_end")},
                            "aot": {k: aot.get(k) for k in ("coreai_cache_before_load", "coreai_cache_after_load", "coreai_cache_end")}},
           "footprint": {"jit": {k: jit.get(k) for k in ("footprint_after_load_bytes", "footprint_bytes_min_max", "footprint_end_bytes")},
                         "aot": {k: aot.get(k) for k in ("footprint_after_load_bytes", "footprint_bytes_min_max", "footprint_end_bytes")}},
           "reset_check": {"jit": jit.get("reset_check"), "aot": aot.get("reset_check")},
           "per_record_seconds": {"jit": {r["id"]: r["direct"]["seconds"].get("wall") for r in jit["records"]},
                                  "aot": {r["id"]: r["direct"]["seconds"].get("wall") for r in aot["records"]}},
           "rows_detail": rows}
    out = Path(args.out) if args.out else RESULTS / "swift_jit_kev-0.8b.json"
    if args.out and out.exists():
        raise SystemExit(f"{out} exists: records are never overwritten")
    rec["label"] = args.label
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("rows", "hidden_sha256_equal", "p_bit_equal", "p_max_abs_diff",
                                          "hidden_max_abs_diff_where_unequal", "finite_rows", "load", "runs_wall_s")}, indent=None)[:1500])
    print({k: {kk: v[kk] for kk in ("max_abs_dp", "mean_of_run_mean_abs_dp", "pass")} for k, v in rec["bar_vs_oracle"].items()})
    print(f"-> {out}")
    return 0


# --------------------------------------------------------------------------- longrow: the Python graph and the scores
LONG_WORK = SWIFT / "longrow"


def cmd_longrow_python(args) -> int:
    """decide.Kev (the AOT graph, direct) on the longrow records: p and every row's hidden rows (GPU; one process)."""
    import asyncio

    import numpy as np

    import decide
    LONG_WORK.mkdir(parents=True, exist_ok=True)
    e = decide.Kev(Path(args.bundle) if args.bundle else BUNDLES[args.model])
    name = args.tag or args.model
    recs = json.loads(LONGROW.read_text())["records"]
    doc = {"model": args.model, "pid": os.getpid(), "aimodelc": str(e.aimodelc), "records": []}
    arrays = {}

    async def go():
        doc["load"] = await e.load()
        for r in recs:
            tr: dict = {}
            body = await e.decide(r["request"], shared=False, trace=tr)
            rows = []
            for k, (row, h, p) in enumerate(zip(tr["_rows"]["rows"], tr["_hidden"], tr["_probs"])):
                arrays[f"{r['id']}__q{k}"] = h
                rows.append({"qid": row["qid"], "T": len(row["row_ids"]), "decide": row["decide"], "opts": row["opts"],
                             "p": [float(x) for x in p], "finite": bool(np.isfinite(h.astype(np.float32)).all()),
                             "hidden_sha256": hashlib.sha256(np.ascontiguousarray(h).tobytes()).hexdigest()})
            doc["records"].append({"id": r["id"], "rows": rows, "response": body, "calls": tr["calls"], "graph_ms": tr["graph_ms"]})
            print(f"{r['id']}: rows {[x['T'] for x in rows]} calls {tr['calls']} graph {tr['graph_ms']:.0f} ms", flush=True)

    asyncio.run(go())
    np.savez(LONG_WORK / f"py_{name}.npz", **arrays)
    (LONG_WORK / f"py_{name}.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"-> {LONG_WORK / f'py_{name}.json'}")
    return 0


def cmd_longrow_score(args) -> int:
    import numpy as np
    model = args.model
    name = args.tag or model
    od = LONGROW_ORACLE[model]
    oracle = {r["id"]: r for r in json.loads((od / "records_oracle.json").read_text())["records"]}
    py = {r["id"]: r for r in json.loads((LONG_WORK / f"py_{name}.json").read_text())["records"]}
    pyz = np.load(LONG_WORK / f"py_{name}.npz")
    sw = {r["id"]: r for r in json.loads((LONG_WORK / f"sw_{name}.json").read_text())["records"]}
    rows = []
    for rid, o in oracle.items():
        hz = np.load(od / "hidden" / f"{rid}.npz")
        for k, q in enumerate(o["questions"]):
            ho = hz[f"q{k}_hidden"].astype(np.float64)
            hp = pyz[f"{rid}__q{k}"].astype(np.float64)
            s = sw[rid]["rows"][k]
            hs = np.fromfile(s["hidden_dump"], np.float16).reshape(hp.shape).astype(np.float64)
            cos = (hp * ho).sum(1) / (np.linalg.norm(hp, axis=1) * np.linalg.norm(ho, axis=1))
            po = np.asarray(q["probs"], np.float64)
            pp = np.asarray(py[rid]["rows"][k]["p"], np.float64)
            ps = np.asarray(s["p"], np.float64)
            T = len(q["row_ids"])
            far = np.arange(T) >= 4032
            rows.append({"row": f"{rid}:q{k}", "qid": q["qid"], "type": q["type"], "T": T, "near_tie": bool(q["near_tie"]),
                         "oracle_top2_margin": q["top2_margin"], "probs_oracle": q["probs"],
                         "python": {"p": pp.tolist(), "max_abs_dp": float(np.abs(pp - po).max()),
                                    "argmax_equal": int(pp.argmax()) == int(po.argmax()), "finite": py[rid]["rows"][k]["finite"]},
                         "swift": {"p": ps.tolist(), "max_abs_dp": float(np.abs(ps - po).max()),
                                   "argmax_equal": int(ps.argmax()) == int(po.argmax()), "finite": bool(s["hidden_finite"])},
                         "swift_hidden_bit_equal_python": s["hidden_sha256"] == py[rid]["rows"][k]["hidden_sha256"],
                         "swift_p_bit_equal_python": bool(np.array_equal(np.asarray(s["p"], np.float32), np.asarray(pp, np.float32))),
                         "pos_cos_graph_vs_oracle": {"min": float(cos.min()), "argmin": int(cos.argmin()), "mean": float(cos.mean()),
                                                     "positions_below_0.99": int((cos < 0.99).sum()),
                                                     "positions_below_0.999": int((cos < 0.999).sum()),
                                                     "min_at_or_after_4032": float(cos[far].min()) if far.any() else None,
                                                     "min_before_4032": float(cos[~far].min()),
                                                     "readout_positions_cos": [float(cos[i]) for i in [q["decide"], *q["opts"]]]},
                         "hidden_max_abs_diff_graph_vs_oracle": float(np.abs(hp - ho).max())})
    rec = {"schema": "kev-longrow/1", "generated_at": now(), "model": model,
           "what": "diagnosis (not part of the gate): rows of about 3,000 and 4,030 - 4,080 tokens (the graph's limit is "
                   "4,080) through the AOT graph (Python decide.Kev and the Swift host, direct) against the author's fp32 "
                   "oracle with its hidden rows (oracle_kev.py, threads 1)",
           "bar": BAR, "fixture": str(LONGROW), "oracle": str(od / "records_oracle.json"),
           "rows": rows,
           "summary": {"rows": len(rows), "max_abs_dp_python": max(r["python"]["max_abs_dp"] for r in rows),
                       "max_abs_dp_swift": max(r["swift"]["max_abs_dp"] for r in rows),
                       "argmax_equal_python": sum(r["python"]["argmax_equal"] for r in rows),
                       "argmax_equal_swift": sum(r["swift"]["argmax_equal"] for r in rows),
                       "swift_hidden_bit_equal_python": sum(r["swift_hidden_bit_equal_python"] for r in rows),
                       "min_pos_cos": min(r["pos_cos_graph_vs_oracle"]["min"] for r in rows),
                       "positions_below_0.99": sum(r["pos_cos_graph_vs_oracle"]["positions_below_0.99"] for r in rows),
                       "within_bar_0.02": all(r["swift"]["max_abs_dp"] <= BAR["max_abs_dp"] and r["python"]["max_abs_dp"] <= BAR["max_abs_dp"]
                                             for r in rows)}}
    out = RESULTS / (f"r15_longrow_{name}.json" if args.tag else f"longrow_{model}.json")
    if args.tag and out.exists():
        raise SystemExit(f"{out} exists: records are never overwritten")
    rec.update({"bundle": args.bundle, "tag": args.tag})
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec["summary"]))
    for r in rows:
        c = r["pos_cos_graph_vs_oracle"]
        print(f"  {r['row']} T {r['T']}: |dp| py {r['python']['max_abs_dp']:.4f} sw {r['swift']['max_abs_dp']:.4f} argmax "
              f"{r['python']['argmax_equal']}/{r['swift']['argmax_equal']} cos min {c['min']:.6f} (>=4032 {c['min_at_or_after_4032']}) "
              f"<0.99 {c['positions_below_0.99']} margin {r['oracle_top2_margin']:.4f}")
    print(f"-> {out}")
    return 0


# --------------------------------------------------------------------------- timing
def cmd_pytime(args) -> int:
    """The Python reference's two same-window items (timing.py's protocol): tv4_000:q0 and own_m01's first 5 questions
    shared, one warm-up each, then --reps decisions."""
    import asyncio
    import time

    import numpy as np

    import decide
    import host
    import timing
    recs = {r["id"]: r for r in json.loads(FIXTURES["fixture"].read_text())["records"]}
    e = decide.Kev(Path(args.bundle) if args.bundle else BUNDLES[args.model])
    doc = {"model": args.label or args.model, "checkpoint": args.model, "bundle": e.name, "pid": os.getpid(), "runtime": "python",
           "aimodelc": str(e.aimodelc), "started": time.time()}

    async def one(req, shared):
        tr: dict = {}
        t0 = time.perf_counter()
        await e.decide(req, shared=shared, trace=tr)
        return {"latency_ms": tr["decide_ms"], "graph_ms": tr["graph_ms"], "e2e_ms": (time.perf_counter() - t0) * 1e3,
                "calls": tr["calls"], "padded_tokens": tr["padded_tokens"]}

    async def go():
        doc["load_cold"] = await e.load()
        doc["load_warm"] = await e.load()
        if args.warm:   # round 15: every call length of the plan once before the items
            doc["warm_up"] = await e.warm_up()
        items = []
        for name, req, shared in (("tv4_000:q0", timing.sub_request(recs["tv4_000"]["request"], [0]), False),
                                  ("b_m01_first5 shared", timing.sub_request(recs["own_m01"]["request"], list(range(5))), True)):
            await one(req, shared)
            reps = [await one(req, shared) for _ in range(args.reps)]
            lat = [r["latency_ms"] for r in reps]
            T = host.build_rows(req, e.tok)["input_tokens"]
            items.append({"item": name, "input_tokens": T, "calls": reps[0]["calls"], "latency_ms": timing.stats(lat),
                          "e2e_ms": timing.stats([r["e2e_ms"] for r in reps]), "graph_ms": timing.stats([r["graph_ms"] for r in reps]),
                          "tokens_per_s_at_median": T / (float(np.median(lat)) / 1e3), "reps": reps})
            print(f"[{args.model}] python {name}: median {np.median(lat):.1f} ms", flush=True)
        doc["items"] = items

    asyncio.run(go())
    doc["finished"] = time.time()
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    return 0


def cmd_timing(args) -> int:
    import numpy as np

    import timing
    run_dir = Path(args.run_dir)
    window = json.loads((run_dir / "window.json").read_text())
    procs = [json.loads((run_dir / f).read_text()) for f in window["swift_processes"]]
    py_same = [json.loads((run_dir / f).read_text()) for f in window.get("python_processes", []) if (run_dir / f).exists()]
    summary = timing.summarize(procs)
    r5 = json.loads((RESULTS / "timing_r5.json").read_text())
    table = []
    for model, m in summary.items():
        r5m = r5["summary"][model]
        for it in m["single"]:
            ref = next(x for x in r5m["single"] if x["item"] == it["item"])
            table.append({"model": model, "item": it["item"], "mode": "one question", "tokens": it["row_tokens"], "calls": it["calls"],
                          "swift_ms": [it["latency_ms"][k] for k in ("median", "p10", "p90")],
                          "swift_tokens_per_s": it["tokens_per_s_at_median"],
                          "python_r5_ms": [ref["latency_ms"][k] for k in ("median", "p10", "p90")],
                          "python_r5_tokens_per_s": ref["tokens_per_s_at_median"]})
        for it in m["multi"]:
            ref = next(x for x in r5m["multi"] if x["item"] == it["item"])
            for mode in ("direct", "shared"):
                table.append({"model": model, "item": it["item"], "mode": mode, "tokens": it["input_tokens"], "calls": it[mode]["calls"],
                              "swift_ms": [it[mode]["latency_ms"][k] for k in ("median", "p10", "p90")],
                              "swift_tokens_per_s": it[mode]["input_tokens_per_s_at_median"],
                              "python_r5_ms": [ref[mode]["latency_ms"][k] for k in ("median", "p10", "p90")],
                              "python_r5_tokens_per_s": ref[mode]["input_tokens_per_s_at_median"]})
        table.append({"model": model, "item": "load (decoder)", "swift_cold_s": m["load_seconds"]["cold"],
                      "swift_warm_s": m["load_seconds"]["warm"], "python_r5_cold_s": r5m["load_seconds"]["cold"],
                      "python_r5_warm_s": r5m["load_seconds"]["warm"]})
    same = []
    for d in py_same:
        for it in d["items"]:
            sw_item = None
            m = summary[d["model"]]
            if it["item"] == "tv4_000:q0":
                sw_item = next(x for x in m["single"] if x["item"] == "tv4_000:q0")["latency_ms"]
            else:
                sw_item = next(x for x in m["multi"] if x["item"] == "b_m01_first5")["shared"]["latency_ms"]
            same.append({"model": d["model"], "item": it["item"], "python_same_window_ms": [it["latency_ms"][k] for k in ("median", "p10", "p90")],
                         "swift_ms": [sw_item[k] for k in ("median", "p10", "p90")], "python_pid": d["pid"],
                         "python_load_s": [d["load_cold"]["seconds"], d["load_warm"]["seconds"]]})
    doc = {"schema": "kev-swift-timing/1", "generated_at": now(),
           "what": "the Swift host's decision latency (`kev time`, Release, AOT .aimodelc, SpecializationOptions.default): "
                   "latency_ms = state resets + graph calls + the float64 head (decide.py's latency_ms), A B A B processes in "
                   "one _GPU_LOCK window; beside round 5's Python numbers (a different window: reference only) and, when "
                   "present, the Python reference's two items run in the same window",
           "window": window, "raw": {"dir": str(run_dir), "files": sorted(p.name for p in run_dir.iterdir())},
           "binary_sha256": sha256_file(BIN), "summary": summary, "table": table, "python_same_window": same,
           "python_r5": {"file": str(RESULTS / "timing_r5.json"), "window": r5["window"].get("lock"), "contended": r5.get("contended"),
                         "note": "round 5's window (21:45-22:17, CPU saturated by other lanes): a different window, reference only"}}
    out = Path(args.out)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    for t in table:
        if "swift_ms" in t:
            print(f"{t['model']:9s} {t['item']:14s} {t['mode']:12s} T {t['tokens']:5d} calls {t['calls']:4d} swift "
                  f"{t['swift_ms'][0]:8.1f} ({t['swift_ms'][1]:.0f}-{t['swift_ms'][2]:.0f}) {t['swift_tokens_per_s']:6.0f} tok/s | "
                  f"py r5 {t['python_r5_ms'][0]:8.1f} ({t['python_r5_ms'][1]:.0f}-{t['python_r5_ms'][2]:.0f}) {t['python_r5_tokens_per_s']:6.0f}")
        else:
            print(t)
    for s in same:
        print("same window:", s)
    print(f"-> {out}")
    return 0


# --------------------------------------------------------------------------- round 11: forms in one timing window
R11_ITEMS = (("single", "tv4_000:q0", None), ("single", "own_j03:q0", None), ("single", "own_L02:q0", None),
             ("single", "own_L01:q2", None), ("multi", "b_m01_first5", "direct"), ("multi", "b_m01_first5", "shared"),
             ("multi", "c_m01_all8", "direct"), ("multi", "c_m01_all8", "shared"), ("multi", "d_L02_all4", "direct"),
             ("multi", "d_L02_all4", "shared"))
R11_RULES = {"candidates": "the forms that passed round 11's full gate (fixture 434 + held-out 130 + red arms + reset), and U16",
             "rank": "the median of tv4_000:q0 (94 tokens, one decision) over every decision of the form, smallest first",
             "excluded": "a form 5 % or more slower than U16 on any of the 10 items (median vs median)",
             "tie": "within 3 % = a tie, broken by b_m01_first5 shared's median (smaller first), then within 3 % again by "
                    "the smaller S"}


def summarize_forms(procs: list[dict]) -> dict:
    """timing.summarize per form label (the `--model-label` of `kev time`), in the window's order, with each process's
    median graph ms per call over its one-question decisions."""
    import numpy as np

    import timing
    out = {}
    for lab in dict.fromkeys(p["model"] for p in procs):
        ps = [p for p in procs if p["model"] == lab]
        per_call = [float(np.median([r["graph_ms"] / r["calls"] for it in p["single"] for r in it["reps"]])) for p in ps]
        m = {"bundle": ps[0]["bundle"], "asset": ps[0]["aimodelc"], "processes": [p["slot"] for p in ps],
             "pids": [p["pid"] for p in ps],
             "load_seconds": {"cold": [p["load_cold"]["seconds"] for p in ps], "warm": [p["load_warm"]["seconds"] for p in ps]},
             "ms_per_call_median_per_process": per_call, "ms_per_call_median": float(np.median(per_call)),
             "footprint_bytes": [p.get("footprint_bytes") for p in ps], "single": [], "multi": []}
        for k, item in enumerate(ps[0]["single"]):
            reps = [r for p in ps for r in p["single"][k]["reps"]]
            lat = [r["latency_ms"] for r in reps]
            m["single"].append({"item": item["item"], "row_tokens": item["row_tokens"], "calls": item["calls"],
                                "padded_tokens": item["padded_tokens"], "latency_ms": timing.stats(lat),
                                "latency_ms_median_per_process": [p["single"][k]["latency_ms"]["median"] for p in ps],
                                "graph_ms": timing.stats([r["graph_ms"] for r in reps]),
                                "head_ms": timing.stats([r["head_ms"] for r in reps]),
                                "tokens_per_s_at_median": item["row_tokens"] / (float(np.median(lat)) / 1e3)})
        for k, item in enumerate(ps[0]["multi"]):
            row = {"item": item["item"], "questions": item["questions"], "input_tokens": item["input_tokens"],
                   "shared_plan": item["shared_plan"],
                   "p_bit_equal_every_rep": all(p["multi"][k]["direct_and_shared_p_bit_equal_every_rep"] for p in ps)}
            for mode in ("direct", "shared"):
                reps = [r for p in ps for r in p["multi"][k][mode]["reps"]]
                lat = [r["latency_ms"] for r in reps]
                row[mode] = {"calls": item[mode]["calls"], "padded_tokens": item[mode]["padded_tokens"],
                             "latency_ms": timing.stats(lat),
                             "latency_ms_median_per_process": [p["multi"][k][mode]["latency_ms"]["median"] for p in ps],
                             "graph_ms": timing.stats([r["graph_ms"] for r in reps])}
            m["multi"].append(row)
        out[lab] = m
    return out


def r11_item(m: dict, item: tuple) -> dict:
    kind, name, mode = item
    if kind == "single":
        return next(x for x in m["single"] if x["item"] == name)["latency_ms"]
    return next(x for x in m["multi"] if x["item"] == name)[mode]["latency_ms"]


def r11_rank(summary: dict, chunks: dict, candidates: list[str], base: str = "U16") -> dict:
    """The fixed ranking rules (R11_RULES) on one window's summary."""
    import functools
    b = summary[base]
    ratios, excluded = {}, {}
    for lab, m in summary.items():
        r = {f"{n}{':' + md if md else ''}": r11_item(m, (k, n, md))["median"] / r11_item(b, (k, n, md))["median"]
             for k, n, md in R11_ITEMS}
        ratios[lab] = r
        slow = {k: v for k, v in r.items() if v >= 1.05}
        if slow and lab != base:
            excluded[lab] = slow
    pool = [lab for lab in summary if (lab == base or lab in candidates) and lab not in excluded]

    def key(lab):
        return (r11_item(summary[lab], R11_ITEMS[0])["median"], r11_item(summary[lab], R11_ITEMS[5])["median"], chunks[lab])

    decisions = []

    def cmp(x, y):
        (tx, sx, cx), (ty, sy, cy) = key(x), key(y)
        if abs(tx - ty) / min(tx, ty) > 0.03:
            why, v = "tv4_000:q0 median", tx - ty
        elif abs(sx - sy) / min(sx, sy) > 0.03:
            why, v = "tie on tv4_000:q0 (within 3 %): b_m01_first5 shared median", sx - sy
        else:
            why, v = "tie on both (within 3 %): smaller S", cx - cy
        decisions.append({"a": x, "b": y, "by": why, "first": x if v < 0 else y})
        return -1 if v < 0 else (1 if v > 0 else 0)

    order = sorted(pool, key=functools.cmp_to_key(cmp))
    return {"rules": R11_RULES, "candidates_given": candidates, "pool": pool, "excluded_5pct_slower_than_u16": excluded,
            "order": order, "rank1": order[0] if order else None, "ratios_vs_u16": ratios, "pairwise": decisions,
            "tv4_000_q0_median_ms": {lab: key(lab)[0] for lab in summary},
            "b_m01_first5_shared_median_ms": {lab: key(lab)[1] for lab in summary}, "chunk": chunks}


def cmd_timing_r11(args) -> int:
    import numpy as np
    run_dir = Path(args.run_dir)
    window = json.loads((run_dir / "window.json").read_text()) if (run_dir / "window.json").exists() else {}
    files = window.get("swift_processes") or sorted(p.name for p in run_dir.glob("p*_*.json"))
    clean = json.loads((run_dir / "clean.json").read_text()) if (run_dir / "clean.json").exists() else None
    if clean:   # the clean-process rule: only these processes enter the medians (every process is recorded)
        files = clean["clean_swift"]
    procs = [json.loads((run_dir / f).read_text()) for f in files if (run_dir / f).exists()]
    summary = summarize_forms(procs)
    forms = window.get("forms") or {}
    chunks = {}
    for lab in summary:
        b = Path(forms.get(lab, LANE / "exports" / "bundles" / summary[lab]["bundle"]))
        chunks[lab] = int(json.loads((b / "metadata.json").read_text())["language"]["prefill_chunk"])
    cands = [x for x in args.candidates.split(",") if x]
    rank = r11_rank(summary, chunks, cands) if "U16" in summary else None
    if args.print_rank1:
        print(rank["rank1"] if rank else "")
        if not args.out:
            return 0
    jit_files, py_files = window.get("jit_processes", []), window.get("python_processes", [])
    if clean:
        pick = lambda kind: list({p["label"]: p["file"] for p in clean["processes"] if p["kind"] == kind and p["clean"]}.values())  # noqa: E731
        jit_files, py_files = pick("swift-jit"), pick("python")
    jit = [json.loads((run_dir / f).read_text()) for f in jit_files if (run_dir / f).exists()]
    pyt = [json.loads((run_dir / f).read_text()) for f in py_files if (run_dir / f).exists()]
    u16 = summary.get("U16")
    u16_check = None
    if u16:
        meds = next(x for x in u16["single"] if x["item"] == "tv4_000:q0")["latency_ms_median_per_process"]
        u16_check = {"tv4_000_q0_median_per_process": meds, "spread": max(meds) / min(meds) - 1,
                     "within_5pct": max(meds) / min(meds) - 1 <= 0.05}
    doc = {"schema": "kev-r11-swift-timing/1", "generated_at": now(), "run_dir": str(run_dir), "window": window,
           "what": "round 11: the Swift host's decision latency (`kev time`, Release, AOT .aimodelc, SpecializationOptions.default) "
                   "per GDN-scan form x S, the forms A B C ... A B C ... in one _GPU_LOCK window, 2 processes x 10 decisions "
                   "per item; then the JIT processes (.aimodel specialized by Swift) and the Python reference",
           "binary_sha256": sha256_file(BIN), "summary": summary, "chunk": chunks, "ranking": rank, "u16_processes": u16_check,
           "clean": clean, "files_in_medians": files,
           "jit": [{"label": d["model"], "asset": d["aimodelc"], "load_cold": d["load_cold"], "load_warm": d["load_warm"],
                    "single": [{"item": x["item"], "latency_ms": x["latency_ms"]} for x in d["single"]],
                    "multi": [{"item": x["item"], "shared": x["shared"]["latency_ms"], "direct": x["direct"]["latency_ms"]}
                              for x in d["multi"]]} for d in jit],
           "python_same_window": [{"label": d["model"], "bundle": d.get("bundle"), "asset": d["aimodelc"],
                                   "items": [{"item": x["item"], "latency_ms": x["latency_ms"], "calls": x["calls"]} for x in d["items"]],
                                   "load": [d["load_cold"], d["load_warm"]]} for d in pyt]}
    if args.out:
        out = Path(args.out)
        if out.exists():
            raise SystemExit(f"{out} exists: records are never overwritten")
        out.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"-> {out}")
    for lab, m in summary.items():
        it = {x["item"]: x["latency_ms"] for x in m["single"]}
        sh = {x["item"]: x["shared"]["latency_ms"] for x in m["multi"]}
        print(f"{lab:6s} S={chunks[lab]:<4d} tv4_000 {it['tv4_000:q0']['median']:8.1f} ms ({it['tv4_000:q0']['p10']:.0f}-"
              f"{it['tv4_000:q0']['p90']:.0f}) own_L02 {it['own_L02:q0']['median']:8.1f} b5 shared {sh['b_m01_first5']['median']:8.1f} "
              f"call {m['ms_per_call_median']:.2f} ms (per process {['%.2f' % x for x in m['ms_per_call_median_per_process']]})")
    if rank:
        print(f"rank: {rank['order']} excluded {list(rank['excluded_5pct_slower_than_u16'])}")
    if u16_check:
        print(f"U16 per-process tv4_000 medians {u16_check['tv4_000_q0_median_per_process']} spread {u16_check['spread']:.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--swift-dir", help="round 15: the Swift work directory (default <work>/_kev/swift): its "
                                        ".build/release/kev, gate/ and longrow/")
    ap.add_argument("--bin", help="round 15: the kev binary (default <swift dir>/.build/release/kev)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("render")
    s = sub.add_parser("score")
    s.add_argument("--model", choices=sorted(BUNDLES), required=True)
    s.add_argument("--fixture")
    s.add_argument("--heldout")
    s.add_argument("--transcript")
    s.add_argument("--bundle", help="round 11: another bundle of --model; then the --gate-* / --e2e-* transcripts of it")
    s.add_argument("--gate-fixture")
    s.add_argument("--gate-heldout")
    s.add_argument("--e2e-fixture")
    s.add_argument("--e2e-heldout")
    s.add_argument("--records", help="round 15: score only these records (comma list; a short window's pass)")
    s.add_argument("--shared-gate-fixture", help="round 15: readout_gate.py --shared of the bundle (dynamic S)")
    s.add_argument("--shared-gate-heldout", help="round 15: the same for the held-out set")
    ng = sub.add_parser("negative")
    ng.add_argument("--out", help="round 15: the record (default results/swift_negative_control.json)")
    ng.add_argument("--bundle", help="round 15: the bundle whose tokenizer `kev rows` reads")
    j = sub.add_parser("jit")
    j.add_argument("--aot", required=True)
    j.add_argument("--jit", required=True)
    j.add_argument("--out", help="round 11: the record (default results/swift_jit_kev-0.8b.json)")
    j.add_argument("--label", help="round 11: the form (kept in the record)")
    sub.add_parser("longrow-fixture")
    for name in ("longrow-oracle", "longrow-python", "longrow-score"):
        o = sub.add_parser(name)
        o.add_argument("--model", choices=sorted(BUNDLES), required=True)
        if name != "longrow-oracle":
            o.add_argument("--bundle", help="round 15: another bundle of --model")
            o.add_argument("--tag", help="round 15: py_/sw_<tag> files and results/r15_longrow_<tag>.json")
    t = sub.add_parser("pytime")
    t.add_argument("--model", choices=sorted(BUNDLES), required=True)
    t.add_argument("--bundle", help="round 11: another bundle of --model (default: the lane's fp16 pf16)")
    t.add_argument("--label", help="round 11: the form label written as the file's `model` (default --model)")
    t.add_argument("--reps", type=int, default=10)
    t.add_argument("--warm", action="store_true", help="round 15: Kev.warm_up (every call length) before the items")
    t.add_argument("--out", required=True)
    w = sub.add_parser("timing")
    w.add_argument("--run-dir", required=True)
    w.add_argument("--out", default=str(RESULTS / "swift_timing_r7.json"))
    w11 = sub.add_parser("timing-r11", help="round 11: one window of forms (KEV_FORMS) -> per-form summary and ranking")
    w11.add_argument("--run-dir", required=True)
    w11.add_argument("--candidates", default="", help="comma list of the forms that passed the full gate (U16 always)")
    w11.add_argument("--print-rank1", action="store_true", help="print the rank-1 form's label (the window's @rank1)")
    w11.add_argument("--out")
    args = ap.parse_args()
    global SWIFT, BIN, LONG_WORK
    if args.swift_dir:
        SWIFT = Path(args.swift_dir).expanduser().resolve()
        BIN, LONG_WORK = SWIFT / ".build" / "release" / "kev", SWIFT / "longrow"
    if args.bin:
        BIN = Path(args.bin).expanduser().resolve()
    return {"render": cmd_render, "score": cmd_score, "negative": cmd_negative, "jit": cmd_jit,
            "longrow-fixture": cmd_longrow_fixture, "longrow-oracle": cmd_longrow_oracle, "longrow-python": cmd_longrow_python,
            "longrow-score": cmd_longrow_score, "pytime": cmd_pytime, "timing": cmd_timing,
            "timing-r11": cmd_timing_r11}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
