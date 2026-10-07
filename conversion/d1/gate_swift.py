#!/usr/bin/env python3
"""Swift host gate, text side: the `d1` CLI (apps/D1) against the Python reference host and HF `tokenizers`.

No graph. Every section runs the Swift binary on inputs written here, recomputes the same thing with host.py /
vision_host.py / `tokenizers` 0.23.2 on the snapshot's tokenizer.json, and compares them exactly:

  render    `d1 render-test` vs host.py: the state block of every fixture state and of edge values (floats, big ints,
            -0.0, booleans, null, nesting, empty containers, unicode, special-token text, newlines, raw literals);
            every fixture request, test_host.py's request table and edge requests (accept / refuse with host.py's text,
            prefix, per question the question block / suffix / codes / keys / legend), and render_ids.json's text
            (prefix + suffix) of all 393 questions; repr / str / round(x, 4) of 200,000 doubles; json.dumps with
            indent=2, ensure_ascii=False and with the defaults; str.strip / isalpha / repr and option_codes of labels
  ids       `d1 rows` vs render_ids.json (393 questions: text, row_ids, slot, codes, alias_ids, groups, keys) and
            host.build_request with the option table (361 records: path, trunk / branch ids, equals_row, shared,
            input_tokens, or the refusal text); the option-table / alias / request-check controls; `d1 encode-test` vs
            `tokenizers` on 2,000 stress texts (n-grams of the fixture's text, CJK, emoji, combining marks, full width,
            special-token text, runs of newlines and spaces, digit strings); swift-transformers' own ids beside them;
            the encode time per row
  image     `d1 image-plan` vs vision_grid_table.json (800 sizes, as given and after cap_pixels; + 7 outside 1:4..4:1)
            and vision_host.plan / image_tokens crop by crop; `d1 image-rows` vs vision_host.prompt_ids /
            extension_ids recomputed and vision_host.json's stored lengths (12 fixture pictures + 6 random + the pair)
  readout   `d1 readout-test` vs host.readout: a random fp32 option table written as a bundle holds it
            (export_option_rows.write_table) x 200 random hidden rows (fp32- and fp16-valued) x the fixture's readout
            groups: logits and p bit for bit; `d1 answers-test` vs host.response: json.dumps(answers) and
            json.dumps(response, indent=2, ensure_ascii=False) byte for byte
  negative  one word of one question changed -> its ids differ; the state's keys reordered -> the text differs;
            1250 <-> 1250.0 -> the text differs (each must be red)
  bundle    `d1 bundle-check` on round 2a's toy bundle: the load contract (metadata, tokenizer, head/option_rows), the
            option-table refusal on its toy table and `decide` stopping at the graph; `d1 readout-test` on that bundle's
            head with the slot hidden rows round 2a's gate took from the Mac GPU (393 runs): p = the recorded p
  all       every section -> results/r3a_swift_text.json

    cd conversion/d1
    source $ZOO_WORK_ROOT/_d1_3b/venv-oracle/bin/activate && python gate_swift.py all
    (the binary: swift build -c release --package-path apps/D1 --scratch-path $ZOO_WORK_ROOT/_d1_3b/swift/.build)

-> $ZOO_WORK_ROOT/_d1_3b/results/r3a_swift_text.json; the CLI's inputs and outputs under _d1_3b/swift/r3a/.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402

import host  # noqa: E402
import vision_host as vh  # noqa: E402
from _paths import hf_snapshot, work_path  # noqa: E402

LANE = work_path("_d1_3b")
SWIFT = LANE / "swift"
BIN = SWIFT / ".build" / "release" / "d1"
WORK = SWIFT / "r3a"
PKG = REPO / "apps" / "D1"
RESULTS = LANE / "results"
TRANSCRIPT = RESULTS / "r3a_swift_text.json"
MODEL = {"hf_id": "LiquidAI/d1-3B", "revision": "da1fe36a861f24690f27f622dca1d8688503d113"}
FIXTURES = LANE / "fixtures" / "records.json"
IMAGE_FIXTURES = LANE / "fixtures" / "image_records.json"
RENDER_IDS = RESULTS / "render_ids.json"
OPTION_IDS = SWIFT / "option_ids.json"
GRID_TABLE = RESULTS / "vision_grid_table.json"
VISION_HOST = RESULTS / "vision_host.json"
TOY_BUNDLE = LANE / "exports" / "toy_bundles" / "d1_toy_decode_fp16_pf16_tbl2"
READOUT_CASES = 200
READOUT_BAR = 1e-12            # used only if the logits are not bit-equal (NumPy's BLAS call is the same one)
ENCODE_MS_BAR = 5.0            # per row, a target: over it the numbers are recorded, not a failure
STRESS_N = 2000
DOUBLES_N = 200_000
NEG_RECORD, NEG_QUESTION, NEG_WORD = "card_refund", "team", "zqxvortmund"   # test_host.py's
KEYORDER_RECORD = "tv4_000"

_TOK = None


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot() -> Path:
    return Path(hf_snapshot(MODEL["hf_id"], revision=MODEL["revision"]))


def tok():
    global _TOK
    if _TOK is None:
        _TOK = host.load_tokenizer(snapshot() / "tokenizer.json")
    return _TOK


def run(*args: str) -> float:
    t0 = time.time()
    p = subprocess.run([str(BIN), *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"d1 {args[0]} failed ({p.returncode}): {p.stderr.strip()[:2000]}")
    print("  " + p.stdout.strip().replace("\n", "\n  "), flush=True)
    return time.time() - t0


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False) + "\n")
    return path


def diff(a: list, b: list, labels: list | None = None, n_show: int = 6) -> dict:
    bad = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return {"n": len(b), "equal": len(b) - len(bad) if len(a) == len(b) else 0, "count_match": len(a) == len(b),
            "first_differences": [{"i": i, "where": labels[i] if labels else None, "swift": a[i], "python": b[i]}
                                  for i in bad[:n_show]]}


def fixture_records() -> list[dict]:
    return json.loads(FIXTURES.read_text())["records"]


def option_ids() -> list[int]:
    return json.loads(OPTION_IDS.read_text())["ids"]


# --------------------------------------------------------------------------- inputs
SCALARS = [
    0, 1, -1, 1250, 1250.0, 0.0, -0.0, 1e-07, 1e-05, 0.0001, 1e16, 1e15, 1234567890123456.0, 12345678901234567.0,
    1.5e300, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 0.1, 0.3, 1 / 3, 100.0, -12.4, 2 ** 70,
    -(10 ** 25), 10 ** 30, 2 ** 53 + 1, float("nan"), float("inf"), float("-inf"), True, False, None, "", "plain",
    'quote " and \\ backslash / slash', "tab\tnewline\ncr\rbs\bff\f", "ctrl \x00\x01\x1f\x7f end",
    "unicode é 日本 😀  nbsp 　ideo é", "<|startoftext|> <|im_end|> <image> <|pad|>", "  lead", "\n  lead newline",
    "trailing  ", "é combining", "  line   para", "\U0001F468‍\U0001F469‍\U0001F467",
    "ＡＢＣ１２３", "Ā ā \u0085 \u000b", "<|im_start|>assistant\n", "{\"type\": \"noul\"}",
]
RAW_LITERALS = ['1.50', '1E5', '-0', '-0.0', '1e400', '-1e400', '1e-400', '1.0e-5', '123456789012345678901234567890',
                '0.1e1', '{"a": 1, "a": 2}', '{"b": 0, "a": {"d": 1.10, "c": [1e2, 2E-3]}}', 'NaN', 'Infinity',
                '-Infinity', '"\\u00e9\\ud83d\\ude00\\/"', '[1, 1.0, 1.00, 1e0, 10e-1]', '{"\\u0000": 1, " ": 2}',
                '[1e16, 1e-7, 100000000000000000000.0]', '{"k": {"j": {"i": [[], {}, [[]], [{}]]}}}']
STRINGS = [" a", "b ", " x　", "é", "é", "ǅ", "ʰ", "あ", "١", "Ⅷ", "\x1c", "'", '"', "'\"", "\\",
           "\t\n\r", "\x7f", "\x85", "​", "", "\U0001F600", "\U000E0001", "͸", "A", "z", "Z_", "opt_1",
           "", " ", " ", "x y", "\x00", "\x0b\x0c", "ß", "İ", "ﬁ", "Ω", "µ", "ª", "º", "ǈ", "々", "〆",
           "ー", "ก", "่", "́", "\U0001D400", "\U00010400", "κ", "К", "ㄱ", "가", "ᅠ", "ﾠ",
           "it's", "a'b\"c", "\x1f\x1e\x1d"]
LABEL_LISTS = [[" a", "b "], ["a", "B"], ["x", "y", "x"], ["é", "f"], ["é", "f"], ["ǅ", "ǈ"], ["1", "2"],
               ["あ", "い"], ["a"], [], [f"o{i}" for i in range(27)], [f"o{i}" for i in range(101)], ["Ⅷ", "a"],
               ["ʰ", "ª"], [" a ", "b"], ["\x1ca", "b"], ["ß", "z"]]


def edge_values() -> list:
    """>= 200 JSON values: every scalar alone, in an object, in a list, and in a nested object."""
    out = []
    for v in SCALARS:
        out += [v, {"k": v}, [v], {"a": [v, {"b": v}], "c": {}, "d": []}]
    out += [[], {}, [[]], [{}], [[], {}, None, "", "  x"],
            {"b": 1, "a": [1, {"c": None, "d": [True, 2.5e-07]}], "e": {}, "f": [], "g": "  x", "h": {"i": {"j": [[1, [2]]]}}},
            {"sp ace": {"x": ""}, "uni ¦": "日本語", "<|k|>": "<|v|>", "a\nb": 1, "\u0000": 2, "é": "é"},
            {"deep": [[[[[[[[[[["x"]]]]]]]]]]]}, {"amount": 1250}, {"amount": 1250.0}, [1e16, 1e-7, 100000000000000000000.0]]
    return out


def doubles() -> list[float]:
    rng = random.Random(7)
    ds: list[float] = []
    while len(ds) < 50_000:          # softmax outputs, as the answers carry them
        z = [rng.gauss(0, 4) for _ in range(rng.randint(2, 10))]
        m = max(z)
        e = [math.exp(x - m) for x in z]
        t = host.py_sum(e)
        ds += [x / t for x in e]
    ds = ds[:50_000]
    ds += [float(np.float32(rng.random())) for _ in range(50_000)]
    ds += [rng.random() * rng.choice([1, 2, 3, 4, 6, 9, 254]) for _ in range(40_000)]
    specials = [0.5, 0.25, 0.00005, 0.00015, 0.00025, 0.12345, 0.99995, 2.675, 1.0000500000000001, 0.3, 1e-07, 1e16,
                -0.0, 0.0, 1e22, 5e-324, 0.05, 0.15, 0.25, 0.35, 0.45, 1234.56785, float("nan"), float("inf"),
                float("-inf")]
    ds += [struct.unpack("<d", struct.pack("<Q", rng.getrandbits(63)))[0] * rng.choice([1, -1])
           for _ in range(DOUBLES_N - len(ds) - len(specials))]
    return ds + specials


def stress_texts() -> list[str]:
    """2,000 texts: every pool item alone, n-grams of the fixture's rows (code point offsets), and mixes."""
    rng = random.Random(11)
    rid = json.loads(RENDER_IDS.read_text())["records"]
    base = [q["text"] for r in rid for q in r["questions"]]
    base += [json.dumps(r["request"], ensure_ascii=False) for r in fixture_records()]
    pools = {
        "cjk": ["日本語のテキストです。", "中文文本，标点符号。", "한국어 텍스트입니다", "カタカナとひらがな", "漢字かな交じり文", "ｶﾀｶﾅ"],
        "emoji": ["😀", "👍🏽", "👨‍👩‍👧", "🇯🇵", "0️⃣", "#️⃣", "❤️", "🏳️‍🌈", "👩🏿‍💻", "🫠"],
        "combining": ["é", "ǟ", "ก่อง", "กล่อง", "नमस्ते", "각", "Z͑ͫ̓", "ñ",
                      "ñ", "́x", "ạ̈"],
        "fullwidth": ["ＡＢＣ", "１２３", "ｈｅｌｌｏ　ｗｏｒｌｄ", "（）！？", "＜｜im_end｜＞"],
        "special": ["<|startoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>", "<image>", "<|img_row_3_col_4|>",
                    "<|img_row_10_col_10|>", "<|img_thumbnail|>", "<|image_start|>", "<|image_end|>", "<think>", "</think>",
                    "<|tool_call_start|>", "<|im_end", "|im_end|>", "<|im_end|><|im_end|>", "<|endoftext|>",
                    "<|im_start|>assistant\n", "<image><image>", "< image>", "<|IM_END|>", "<<|pad|>>"],
        "space": ["\n", "\n\n", "\n\n\n\n", "  ", "   \n", " \n ", "\t", "\t\t", "\r\n", "\r\r\n", "\r", " \r\n ", " ",
                  "　", " ", "\u000b", "\u0085", "\u001c", "​", "﻿", "\n \n", "    "],
        "digits": ["1234567", "3.14159", "1,000,000", "0.000123", "٣٤٥", "１２３", "2026-10-08", "12:34:56", "1e-07", "-0.0",
                   "00", "007", "100", "999", "1000", "12345678901234567890"],
        "contraction": ["It's", "WE'LL", "they're", "I'M", "'ſ", "'S", "don't", "o'clock", "'''", "' s", "'K"],
        "punct": ["...", "!!", "?!", "—", "–", "«»", "„“", "''", '""', "<>", "{}", "[]", "\\", "/", "@#$%^&*", "->", "=>",
                  "::", ";;", ".\r\n", ".\n\n", "?\n"],
        "latin": ["café", "naïve", "Straße", "ÀÉÎÕÜ", "Ωμέγα", "Привет мир", "שלום", "مرحبا", "İstanbul", "ǅemal",
                  " tpubli", "\tpubli", " impleme", "nclu"],
    }
    texts = [x for p in pools.values() for x in p]
    while len(texts) < STRESS_N * 2 // 5:
        s = rng.choice(base)
        a = rng.randrange(len(s))
        texts.append(s[a:a + rng.randint(1, 200)])
    while len(texts) < STRESS_N:
        parts = []
        for _ in range(rng.randint(1, 6)):
            parts.append(rng.choice(pools[rng.choice(list(pools))]))
            parts.append(rng.choice(["", " ", "\n", "  ", rng.choice(pools["space"]), rng.choice(base)[:rng.randint(0, 20)]]))
        texts.append("".join(parts))
    return texts[:STRESS_N]


# --------------------------------------------------------------------------- render
def py_request(text: str) -> dict:
    req = json.loads(text)
    out: dict = {}
    try:
        host.validate_request(req)
        out["accept"] = True
    except ValueError as e:
        out.update(accept=False, error=str(e))
    if isinstance(req, dict) and "state" in req:
        out["prefix"] = host.prefix_text(req["state"])
    qs = []
    questions = req.get("questions") if isinstance(req, dict) else None
    for name, qd in (questions.items() if isinstance(questions, dict) else []):
        try:
            q = host.validate_question(name, qd)
            alias = host.aliases(tok(), list(q["criteria"])) if q["type"] == "choice" else None
            codes = [c for c, _ in alias] if alias is not None else None
            qs.append({"name": name, "type": q["type"], "question_block": host.question_block(q, codes),
                       "suffix": host.suffix_text(q, codes), "codes": codes, "keys": host.option_keys(q),
                       "legend": list(q["criteria"]) if q["type"] == "score" else None})
        except ValueError as e:
            qs.append({"name": name, "error": str(e)})
    out["questions"] = qs
    return out


def edge_requests() -> list[dict]:
    out = []
    for v in edge_values():
        out.append({"state": v, "questions": {
            "c": {"type": "choice", "instructions": "Pick one.", "criteria": {"x_y": None, "z": "", "w": "desc"}},
            "n": {"type": "noul", "instructions": "Is it?", "criteria": {"true": "yes it is", "false": None}},
            "s": {"type": "score", "instructions": "How much?", "criteria": ["low", "high"]}}})
    for labels in LABEL_LISTS:
        out.append({"state": "s", "questions": {"a": {"type": "choice", "instructions": "Which?",
                                                      "criteria": {lab: None for lab in labels}}}})
    for s in STRINGS:
        out.append({"state": s, "questions": {"q": {"type": "noul", "instructions": s or "?", "criteria": {"true": s, "x": 1}}}})
    return out


def section_render() -> dict:
    import test_host
    recs = fixture_records()
    states = [json.dumps(r["request"]["state"], ensure_ascii=False) for r in recs if r["request"].get("state") is not None]
    states += [json.dumps(v, ensure_ascii=False) for v in edge_values() if v is not None] + RAW_LITERALS
    req_objs = [r["request"] for r in recs] + list(test_host.VALID.values()) + list(test_host.INVALID.values()) + edge_requests()
    rwhere = ([f"fixture/{r['id']}" for r in recs] + [f"table/{k}" for k in test_host.VALID] + [f"table/{k}" for k in test_host.INVALID]
              + [f"edge/{i}" for i in range(len(req_objs) - len(recs) - len(test_host.VALID) - len(test_host.INVALID))])
    requests = [json.dumps(r, ensure_ascii=False) for r in req_objs]
    ds = doubles()
    dumps_in = []
    for path in sorted(RESULTS.glob("r2a_toy_readout_*_tbl2.json")):   # answer-shaped bodies the gates wrote
        doc = json.loads(path.read_text())
        for run_ in doc.get("runs", [])[:200]:
            if "probs" in run_:
                dumps_in.append(json.dumps({"probs": run_["probs"], "keys": run_.get("keys")}, ensure_ascii=False))
    dumps_in += [json.dumps(v, ensure_ascii=False) for v in edge_values()] + RAW_LITERALS
    dumps_in.append(json.dumps({"answers": {"é": {"type": "choice", "choice": "日本 😀", "confidence": 0.5,
                                                   "probabilities": {"日本 😀": 0.5, "ctrl\x00\x1f\x7f": 0.25, 'q"\\/': 0.25}}},
                                "usage": {"input_tokens": 12, "output_tokens": 0}}, ensure_ascii=False))
    inp = {"states": states, "requests": requests, "doubles": ds, "dumps": dumps_in, "strings": STRINGS,
           "label_lists": LABEL_LISTS}
    src = write_json(WORK / "render_in.json", inp)
    out = WORK / "render_out.json"
    secs = run("render-test", "--in", str(src), "--tokenizer", str(snapshot()), "--out", str(out))
    got = json.loads(out.read_text())
    want = {
        "state_blocks": [host.state_block(json.loads(t)) for t in states],
        "requests": [py_request(t) for t in requests],
        "repr": [json.dumps(d) for d in ds], "str": [str(d) for d in ds], "round4": [json.dumps(round(d, 4)) for d in ds],
        "dumps_indent2": [json.dumps(json.loads(t), indent=2, ensure_ascii=False) for t in dumps_in],
        "dumps_default": [json.dumps(json.loads(t)) for t in dumps_in],
        "repr_str": [repr(s) for s in STRINGS], "strip": [s.strip() for s in STRINGS],
        "isalpha": [s.isalpha() for s in STRINGS], "option_codes": [host.option_codes(x) for x in LABEL_LISTS],
    }
    labels = {"requests": rwhere}
    rep: dict = {"what": "d1 render-test vs host.py / CPython " + sys.version.split()[0], "seconds_swift": round(secs, 2),
                 "inputs": {"states": len(states), "requests": len(requests), "requests_accepted_by_host":
                            sum(w["accept"] for w in want["requests"]), "doubles": len(ds), "dumps": len(dumps_in),
                            "strings": len(STRINGS), "label_lists": len(LABEL_LISTS), "edge_values": len(edge_values())}}
    for k in want:
        rep[k] = diff(got[k], want[k], labels.get(k))
    # render_ids.json's text of every question = Swift's prefix + suffix of the fixture request
    rid = {r["id"]: r for r in json.loads(RENDER_IDS.read_text())["records"]}
    texts_ok, texts_n, bad = 0, 0, []
    for r, g in zip(recs, got["requests"][:len(recs)]):
        sq = {q["name"]: q for q in g["questions"]}
        for q in rid[r["id"]]["questions"]:
            texts_n += 1
            s = sq.get(q["name"], {})
            ok = "suffix" in s and g.get("prefix", "") + s["suffix"] == q["text"]
            texts_ok += ok
            if not ok and len(bad) < 4:
                bad.append(f"{r['id']}/{q['name']}")
    rep["render_ids_text"] = {"n": texts_n, "equal": texts_ok, "first_differences": bad}
    rep["pass"] = all(rep[k]["equal"] == rep[k]["n"] and rep[k].get("count_match", True) for k in want) and texts_ok == texts_n
    return rep


# --------------------------------------------------------------------------- ids
def controls() -> list[dict]:
    """Requests the option table, the alias rule or the checks decide (host.py §1, §3)."""
    return [
        {"id": "ctl_table_kana", "request": {"state": "s", "questions": {"a": {"type": "choice", "instructions": "Which?",
                                                                            "criteria": {"あ": "first", "い": "second"}}}}},
        {"id": "ctl_table_second_question", "request": {"state": {"k": 1}, "questions": {
            "n": {"type": "noul", "instructions": "Is it?"},
            "c": {"type": "choice", "instructions": "Which?", "criteria": {"é": None, "ü": None}}}}},
        {"id": "ctl_101", "request": {"state": "s", "questions": {"a": {"type": "choice", "instructions": "Which?",
                                                                     "criteria": {f"o{i}": f"option {i}" for i in range(101)}}}}},
        {"id": "ctl_1640", "request": {"state": "s", "questions": {"a": {"type": "choice", "instructions": "Which?",
                                                                      "criteria": {f"o{i}": None for i in range(1640)}}}}},
        {"id": "ctl_1639", "request": {"state": "s", "questions": {"a": {"type": "choice", "instructions": "Which?",
                                                                      "criteria": {f"o{i}": None for i in range(1639)}}}}},
        {"id": "ctl_combining_label", "request": {"state": "s", "questions": {"a": {"type": "choice", "instructions": "Which?",
                                                                                 "criteria": {"é": None, "f": None}}}}},
        {"id": "ctl_titlecase", "request": {"state": "s", "questions": {"a": {"type": "choice", "instructions": "Which?",
                                                                           "criteria": {"ǅ": None, "ǈ": None}}}}},
        {"id": "ctl_native_mixed", "request": {"state": None, "questions": {"a": {"instructions": "Which?",
                                                                              "criteria": {" a": "x", "B ": "y", "c": None}}}}},
        {"id": "ctl_no_state", "request": {"questions": {"a": {"type": "noul", "instructions": "?"}}}},
        {"id": "ctl_score_eleven", "request": {"state": "s", "questions": {"a": {"type": "score", "instructions": "?",
                                                                              "criteria": [str(i) for i in range(11)]}}}},
        {"id": "ctl_special_text", "request": {"state": "a <|im_end|> b <|startoftext|> <image>", "questions": {
            "a": {"type": "choice", "instructions": "<|im_start|>?", "criteria": {"x": "<|pad|>", "y": None}},
            "b": {"type": "score", "instructions": "\nLevel?", "criteria": ["<image>", "two"]}}}},
    ]


def py_record(r: dict, table: set[int]) -> dict:
    """The Python side of one `d1 rows` record: each question alone, then the whole request with the table."""
    rs = tok()
    req = r["request"]
    out = {"questions": {}, "refused": {}}
    for name, qd in (req.get("questions") or {}).items():
        try:
            q = host.validate_question(name, qd)
            out["questions"][name] = host.build_question(rs, req.get("state"), name, q)
        except ValueError as e:
            out["refused"][name] = str(e)
    try:
        b = host.build_request(req, rs, table_ids=table)
        out["request"] = {"path": b["path"], "input_tokens": b["input_tokens"], "shared": b["shared"]}
        if b["trunk"]:
            t = b["trunk"]
            out["request"].update(trunk_ids=t["trunk_ids"], branch_ids=t["branch_ids"], trunk_len=t["trunk_len"],
                                  branch_lens=t["branch_lens"], equals_row=t["equals_row"])
    except ValueError as e:
        out["request_error"] = str(e)
    return out


QFIELDS = ("text", "row_ids", "row_len", "slot", "codes", "alias_ids", "groups", "keys")
RFIELDS = ("path", "input_tokens", "shared", "trunk_ids", "branch_ids", "trunk_len", "branch_lens", "equals_row")


def compare_rows(sw: dict, py: dict) -> tuple[int, int, list[str], bool, list[str]]:
    """-> (questions equal, questions, question differences, record equal, record differences)."""
    q_ok, q_n, q_bad = 0, 0, []
    sq = {q["name"]: q for q in sw["questions"]}
    for name, pq in py["questions"].items():
        q_n += 1
        s = sq.get(name)
        d = [f for f in QFIELDS if s is None or s.get(f) != pq[f]]
        q_ok += not d
        if d:
            q_bad.append(f"{name}: {d}")
    r_bad = []
    if sw.get("refused", {}) != py["refused"]:
        r_bad.append("refused")
    if "request_error" in py or "request_error" in sw:
        if sw.get("request_error") != py.get("request_error"):
            r_bad.append(f"request_error swift {sw.get('request_error')!r} python {py.get('request_error')!r}")
    else:
        for f in RFIELDS:
            if sw.get(f) != py["request"].get(f):
                r_bad.append(f)
    return q_ok, q_n, q_bad, not r_bad, r_bad


def section_ids() -> dict:
    rs = tok()
    table = set(option_ids())
    recs = fixture_records()
    out = WORK / "rows_fixture.json"
    secs = run("rows", "--records", str(FIXTURES), "--tokenizer", str(snapshot()), "--option-ids", str(OPTION_IDS),
               "--out", str(out), "--plain")
    sw = json.loads(out.read_text())
    swr = {r["id"]: r for r in sw["records"]}
    rid = {r["id"]: r for r in json.loads(RENDER_IDS.read_text())["records"]}
    q_ok = q_n = rec_ok = plain_ok = plain_n = rid_rec_ok = 0
    q_bad, rec_bad, plain_bad = [], [], []
    for r in recs:
        s = swr[r["id"]]
        ref = rid[r["id"]]
        # the questions against render_ids.json (round 1's reference rows)
        sq = {q["name"]: q for q in s["questions"]}
        for q in ref["questions"]:
            q_n += 1
            d = [f for f in QFIELDS if sq.get(q["name"], {}).get(f) != q[f]]
            q_ok += not d
            if d and len(q_bad) < 8:
                q_bad.append(f"{r['id']}/{q['name']}: {d}")
            if "plain_row_ids" in sq.get(q["name"], {}):
                plain_n += 1
                same = sq[q["name"]]["plain_row_ids"] == q["row_ids"]
                plain_ok += same
                if not same and len(plain_bad) < 8:
                    a, b = sq[q["name"]]["plain_row_ids"], q["row_ids"]
                    k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                    plain_bad.append({"q": f"{r['id']}/{q['name']}", "len_swift_plain": len(a), "len_tokenizers": len(b),
                                      "first_diff_at": k, "plain": a[k:k + 4], "tokenizers": b[k:k + 4],
                                      "text_at": rs.decode(b[max(0, k - 2):k + 4])})
        # the record against host.build_request with the table, and against render_ids.json's stored record fields
        py = py_record(r, table)
        _, _, _, ok, d = compare_rows(s, py)
        if s.get("refused", {}) != ref["refused"]:
            ok, d = False, d + ["refused vs render_ids"]
        rec_ok += ok
        if not ok and len(rec_bad) < 8:
            rec_bad.append(f"{r['id']}: {d}")
        stored = ({"path": ref.get("path"), "input_tokens": ref.get("input_tokens"), "shared": ref.get("shared")}
                  if "path" in ref else None)
        mine = {"path": s.get("path"), "input_tokens": s.get("input_tokens"), "shared": s.get("shared")} if "path" in s else None
        st_ok = stored == mine and (("trunk" not in ref) or (ref["trunk"]["trunk_len"] == s.get("trunk_len")
                                                              and ref["trunk"]["branch_lens"] == s.get("branch_lens")
                                                              and ref["trunk"]["equals_row"] == s.get("equals_row")))
        rid_rec_ok += st_ok
    refused = [{"id": r["id"], "swift": swr[r["id"]].get("request_error"), "python": py_record(r, table).get("request_error"),
                "swift_refused": swr[r["id"]]["refused"], "render_ids_refused": rid[r["id"]]["refused"]}
               for r in recs if rid[r["id"]]["refused"]]
    # controls
    ctl = controls()
    import test_host
    ctl += [{"id": f"table/{k}", "request": v} for k, v in {**test_host.VALID, **test_host.INVALID}.items()]
    cin = write_json(WORK / "controls_in.json", {"records": ctl})
    cout = WORK / "rows_controls.json"
    run("rows", "--records", str(cin), "--tokenizer", str(snapshot()), "--option-ids", str(OPTION_IDS), "--out", str(cout))
    cs = {r["id"]: r for r in json.loads(cout.read_text())["records"]}
    ctl_rows = []
    for c in ctl:
        py = py_record(c, table)
        qok, qn, qbad, ok, d = compare_rows(cs[c["id"]], py)
        ctl_rows.append({"id": c["id"], "equal": ok and qok == qn, "python_request_error": py.get("request_error"),
                         "swift_request_error": cs[c["id"]].get("request_error"), "differences": (qbad + d)[:4]})
    # stress texts
    texts = stress_texts()
    tin = write_json(WORK / "stress_texts.json", {"texts": texts})
    tout = WORK / "stress_out.json"
    run("encode-test", "--texts", str(tin), "--tokenizer", str(snapshot()), "--out", str(tout), "--plain")
    st = json.loads(tout.read_text())
    want = [rs.encode(t, add_special_tokens=False).ids for t in texts]
    s_ok = sum(a == b for a, b in zip(st["ids"], want))
    p_ok = sum(a == b for a, b in zip(st["plain_ids"], want))
    s_bad = [{"i": i, "text": texts[i][:80], "swift": a[:12], "tokenizers": b[:12]}
             for i, (a, b) in enumerate(zip(st["ids"], want)) if a != b][:6]
    p_bad = [{"i": i, "text": texts[i][:60]} for i, (a, b) in enumerate(zip(st["plain_ids"], want)) if a != b][:6]
    ms = sorted(st["ms"])
    specials = ([("<|startoftext|>", 124894), ("<|im_start|>", 124899), ("<|im_end|>", 124900), ("<|pad|>", 124893),
                 ("<image>", vh.IMAGE_ID), ("<|image_start|>", vh.IMAGE_START_ID), ("<|image_end|>", vh.IMAGE_END_ID),
                 ("<|img_thumbnail|>", vh.THUMBNAIL_ID)]
                + [(f"<|img_row_{a}_col_{b}|>", vh.row_col_id(a, b)) for a in range(1, 11) for b in range(1, 11)])
    special_ok = sum(rs.token_to_id(t) == i and rs.encode(t, add_special_tokens=False).ids == [i] for t, i in specials)
    contract = sw["tokenizer"]
    rep = {
        "what": "d1 rows / encode-test vs render_ids.json, host.build_request (option table) and tokenizers "
                f"{__import__('tokenizers').__version__}",
        "seconds_swift_rows": round(secs, 2),
        "tokenizer": contract,
        "questions": {"n": q_n, "equal": q_ok, "first_differences": q_bad},
        "records": {"n": len(recs), "equal_host_build_request": rec_ok, "equal_render_ids_fields": rid_rec_ok,
                    "first_differences": rec_bad},
        "refused_records": refused,
        "plain_swift_transformers_rows": {"n": plain_n, "equal": plain_ok, "differences": plain_bad},
        "controls": {"n": len(ctl_rows), "equal": sum(c["equal"] for c in ctl_rows), "rows": ctl_rows},
        "stress": {"n": len(texts), "equal": s_ok, "plain_swift_transformers_equal": p_ok, "first_differences": s_bad,
                   "plain_first_differences": p_bad, "ms_p50": ms[len(ms) // 2], "ms_max": ms[-1],
                   "texts_sha256": hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()},
        "encode_ms_per_row": {**sw["row_ms"], "bar": ENCODE_MS_BAR, "within_bar": sw["row_ms"]["max"] <= ENCODE_MS_BAR},
        "special_tokens": {"n": len(specials), "tokenizers_equal": special_ok,
                           "swift_checked_at_load": contract["special_tokens_checked"]},
    }
    rep["pass"] = (q_ok == q_n == 393 and rec_ok == len(recs) == 361 and rid_rec_ok == len(recs)
                   and all(c["equal"] for c in ctl_rows) and s_ok == len(texts)
                   and len(refused) == 1 and refused[0]["swift"] == refused[0]["python"]
                   and special_ok == len(specials) == contract["special_tokens_checked"])
    return rep


# --------------------------------------------------------------------------- image
def plan_py(p: vh.Plan, size_wh: list[int]) -> dict:
    run_ = vh.image_tokens(p)
    return {"size": size_wh, "rows": p.rows, "cols": p.cols, "n_crops": len(p.crops),
            "crops": [{"kind": c.kind, "grid": list(c.grid), "tokens": c.n_tokens, "row": c.row, "col": c.col} for c in p.crops],
            "tokens": vh.n_image_tokens([p]), "max_patches": max(c.n_patches for c in p.crops), "run_len": len(run_),
            "run_head": run_[:3], "run_tail": run_[-3:],
            "run_sha256": hashlib.sha256(",".join(map(str, run_)).encode()).hexdigest()}


def section_image() -> dict:
    from test_vision_host import OUTSIDE_SIZES, PAIR, PAIR_QUESTIONS, RANDOM_QUESTION
    gt = json.loads(GRID_TABLE.read_text())
    sizes = [[r["w"], r["h"]] for r in gt["rows"]] + [[w, h] for w, h in OUTSIDE_SIZES]
    sin = write_json(WORK / "image_sizes.json", {"sizes": sizes})
    sout = WORK / "image_plan.json"
    run("image-plan", "--sizes", str(sin), "--out", str(sout))
    sp = json.loads(sout.read_text())
    keys = ("size", "rows", "cols", "n_crops", "tokens", "max_patches")
    table_ok, crop_ok, bad = 0, 0, []
    for k, (w, h) in enumerate(sizes):
        s = sp["rows"][k]
        cw, ch = vh.cap_size(w, h)
        mine = {"direct": plan_py(vh.plan(h, w), [w, h]), "capped": plan_py(vh.plan(ch, cw), [cw, ch])}
        crop_eq = all(s[kind][f] == mine[kind][f] for kind in mine for f in mine[kind])
        crop_ok += crop_eq
        if k < len(gt["rows"]):
            r = gt["rows"][k]
            t_eq = all(s[kind][f] == r[kind][f] for kind in ("direct", "capped") for f in keys) and all(
                sorted({tuple(c["grid"]) for c in s[kind]["crops"]}) == [tuple(g) for g in r[kind]["grids"]]
                for kind in ("direct", "capped"))
            table_ok += t_eq
        else:
            t_eq = True
        if (not crop_eq or not t_eq) and len(bad) < 6:
            bad.append({"w": w, "h": h, "swift": s, "python": mine})
    ratios_eq = sp["target_ratios"] == [list(r) for r in vh.TARGET_RATIOS]
    # the requests with pictures
    vhost = json.loads(VISION_HOST.read_text())
    fx = {r["id"]: r for r in json.loads(IMAGE_FIXTURES.read_text())["records"]}
    recs = []
    for r in vhost["records"]:
        if r["kind"] == "fixture":
            st, qs = fx[r["id"]]["request"]["state"], fx[r["id"]]["request"]["questions"]
        elif r["kind"] == "random":
            st, qs = None, RANDOM_QUESTION
        else:
            st, qs = "Two drawings.", PAIR_QUESTIONS
        recs.append({"id": r["id"], "state": st, "questions": qs, "pictures": r["size_in"]})
    assert sum(r["kind"] == "pair" for r in vhost["records"]) == 1 and PAIR
    rin = write_json(WORK / "image_rows_in.json", {"records": recs})
    rout = WORK / "image_rows.json"
    run("image-rows", "--records", str(rin), "--tokenizer", str(snapshot()), "--out", str(rout))
    sw = {r["id"]: r for r in json.loads(rout.read_text())["records"]}
    rs = tok()
    rows_ok, stored_ok, rbad = 0, 0, []
    for rec, stored in zip(recs, vhost["records"]):
        req = host.validate_request({"state": rec["state"], "questions": rec["questions"]})
        built = [host.build_question(rs, req["state"], n, q) for n, q in req["questions"]]
        plans = []
        for w, h in rec["pictures"]:
            cw, ch = vh.cap_size(w, h)
            plans.append(vh.plan(ch, cw))
        prefix = vh.image_prefix_text(req["state"], len(plans))
        text = prefix + built[0]["suffix"] if len(built) == 1 else prefix
        ids = vh.prompt_ids(rs, text, plans)
        branches = [host.token_ids(rs, b["suffix"]) for b in built] if len(built) > 1 else []
        want = {"text": text, "ids": ids, "extension_ids": vh.extension_ids(ids), "branch_ids": branches,
                "input_tokens": len(ids) + sum(map(len, branches)), "n_image_tokens": vh.n_image_tokens(plans)}
        s = sw[rec["id"]]
        d = [k for k in want if s.get(k) != want[k]]
        crops_sw = [[(c["kind"], c["grid"], c["tokens"], c["row"], c["col"]) for c in p["crops"]] for p in s.get("plans", [])]
        crops_py = [[(c.kind, list(c.grid), c.n_tokens, c.row, c.col) for c in p.crops] for p in plans]
        if crops_sw != crops_py:
            d.append("plans")
        rows_ok += not d
        st_d = []
        if stored["ids_len"] != len(s.get("ids", [])):
            st_d.append("ids_len")
        if stored["n_image_tokens"] != s.get("n_image_tokens"):
            st_d.append("n_image_tokens")
        if stored["input_tokens"]["provider"] != s.get("input_tokens"):
            st_d.append("input_tokens")
        for pic, p in zip(stored["pictures"], s.get("plans", [])):
            if pic["size_capped"] != p["size"] or pic["rows"] != p["rows"] or pic["cols"] != p["cols"] or \
                    [(c["kind"], c["grid"], c["tokens"], c["row"], c["col"]) for c in pic["crops"]] != \
                    [(c["kind"], c["grid"], c["tokens"], c["row"], c["col"]) for c in p["crops"]]:
                st_d.append("pictures")
        stored_ok += not st_d
        if (d or st_d) and len(rbad) < 6:
            rbad.append({"id": rec["id"], "vs_python": d, "vs_vision_host_json": st_d})
    n_main = sum(r["kind"] in ("fixture", "random") for r in vhost["records"])
    rep = {"what": "d1 image-plan / image-rows vs vision_grid_table.json, vision_host.plan / prompt_ids / extension_ids "
                   "and vision_host.json",
           "grid_table": {"n": len(gt["rows"]), "equal": table_ok},
           "crop_by_crop_and_run": {"n": len(sizes), "equal": crop_ok, "outside_sizes": len(OUTSIDE_SIZES)},
           "target_ratios_equal": ratios_eq, "first_differences": bad,
           "requests": {"n": len(recs), "n_single_pictures": n_main, "equal_python": rows_ok,
                        "equal_vision_host_json": stored_ok, "first_differences": rbad},
           "max_image_tokens_capped": max(r["capped"]["tokens"] for r in sp["rows"][:len(gt["rows"])])}
    rep["pass"] = (table_ok == len(gt["rows"]) == 800 and crop_ok == len(sizes) and ratios_eq
                   and rows_ok == stored_ok == len(recs))
    return rep


# --------------------------------------------------------------------------- readout
def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def section_readout() -> dict:
    import export_option_rows as eor
    ids = option_ids()
    rng = np.random.default_rng(31)
    d = 2048
    table = (rng.standard_normal((len(ids), d)) * 0.05).astype(np.float32)
    index = {i: k for k, i in enumerate(ids)}
    head = WORK / "head_random"
    if head.exists():
        shutil.rmtree(head)
    eor.write_table(lambda want: table[[index[int(i)] for i in want]], ids, head,
                    source={"what": "round 3a gate: random N(0, 0.05^2) fp32 rows over the option table's ids", "seed": 31})
    got_ids, got_rows = eor.table_arrays(head)
    assert list(map(int, got_ids)) == ids and np.array_equal(got_rows.view(np.uint32), table.view(np.uint32))
    hidden = (rng.standard_normal((READOUT_CASES, d)) * 2.0).astype(np.float32)
    hidden[READOUT_CASES // 2:] = hidden[READOUT_CASES // 2:].astype(np.float16).astype(np.float32)   # fp16-valued rows
    hidden[0] = 0.0                                                                                      # all logits equal
    hpath = WORK / "readout_hidden.f32"
    hpath.write_bytes(hidden.astype("<f4").tobytes())
    rid = json.loads(RENDER_IDS.read_text())["records"]
    sets = []
    for r in rid:
        for q in r["questions"]:
            if q["groups"] not in sets:
                sets.append(q["groups"])
    rs = tok()
    for extra in ({"type": "score", "instructions": "x", "criteria": ["only"]},
                  {"type": "choice", "instructions": "x", "criteria": {"only": None}},
                  {"type": "choice", "instructions": "x", "criteria": {f"o{i}": None for i in range(27)}},
                  {"type": "score", "instructions": "x", "criteria": [f"l{i}" for i in range(10)]}):
        g = host.readout_groups(rs, host.validate_question("x", extra))
        if g not in sets:
            sets.append(g)
    cases = [{"hidden_index": k, "groups": sets[k % len(sets)]} for k in range(READOUT_CASES)]
    cin = write_json(WORK / "readout_in.json", {"table_n": len(ids), "cases": cases})
    cout = WORK / "readout_out.json"
    run("readout-test", "--in", str(cin), "--head", str(head), "--hidden", str(hpath), "--out", str(cout))
    sw = json.loads(cout.read_text())["cases"]
    z_bit = p_bit = 0
    z_max = p_max = 0.0
    single_id_cases = 0
    for c, s in zip(cases, sw):
        gids = host.group_ids(c["groups"])
        rows = table[[index[i] for i in gids]]
        z = host.option_logits(hidden[c["hidden_index"]], rows, gids)
        p = host.probs_from_logits(z, c["groups"])
        single_id_cases += len(gids) == 1
        zs = [int(b, 16) for b in s["logit_bits"]]
        ps = [int(b, 16) for b in s["p_bits"]]
        z_bit += s["ids"] == gids and zs == [bits(z[i]) for i in gids]
        p_bit += ps == [bits(x) for x in p]
        z_max = max(z_max, max(abs(struct.unpack("<d", struct.pack("<Q", a))[0] - z[i]) for a, i in zip(zs, gids)))
        p_max = max(p_max, max(abs(struct.unpack("<d", struct.pack("<Q", a))[0] - b) for a, b in zip(ps, p)))
    # answers and the response body
    acases, want_a, want_r = [], [], []
    arng = random.Random(5)
    table_set = set(ids)
    for r in fixture_records():
        try:
            b = host.build_request(r["request"], rs, table_ids=table_set)
        except ValueError:
            continue
        probs = []
        for _, q in b["validated"]:
            n = len(host.option_keys(q))
            g = [arng.gammavariate(0.3, 1.0) + 1e-300 for _ in range(n)]
            t = sum(g)
            probs.append([x / t for x in g])
        acases.append({"request": json.dumps(r["request"], ensure_ascii=False), "probs": probs, "input_tokens": b["input_tokens"]})
        resp = host.response(b["validated"], probs, b["input_tokens"])
        want_a.append(json.dumps(resp["answers"]))
        want_r.append(json.dumps(resp, indent=2, ensure_ascii=False))
    edge_req = {"state": "s", "questions": {"n": {"type": "noul", "instructions": "?"},
                                            "c": {"type": "choice", "instructions": "?", "criteria": {"é": None, "x": "", "日本": "d"}},
                                            "s": {"type": "score", "instructions": "?", "criteria": ["a", "b", "c"]}}}
    vq = host.validate_request(edge_req)["questions"]
    for probs in ([[1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.5, 0.5], [1 / 3, 1 / 3, 1 / 3], [0.25, 0.5, 0.25]],
                  [[5e-324, 1.0], [1e-300, 0.5, 0.5], [0.1, 0.2, 0.7]], [[0.0, 1.0], [0.2, 0.4, 0.4], [0.4, 0.4, 0.2]]):
        acases.append({"request": json.dumps(edge_req, ensure_ascii=False), "probs": probs, "input_tokens": 77})
        resp = host.response(vq, probs, 77)
        want_a.append(json.dumps(resp["answers"]))
        want_r.append(json.dumps(resp, indent=2, ensure_ascii=False))
    ain = write_json(WORK / "answers_in.json", {"cases": acases})
    aout = WORK / "answers_out.json"
    run("answers-test", "--in", str(ain), "--out", str(aout))
    sa = json.loads(aout.read_text())["cases"]
    a_ok = sum(s.get("answers_dumps") == w for s, w in zip(sa, want_a))
    r_ok = sum(s.get("response_dumps_indent2") == w for s, w in zip(sa, want_r))
    rep = {"what": "d1 readout-test / answers-test vs host.readout / host.response (CPython " + sys.version.split()[0]
                   + ", NumPy " + np.__version__ + ")",
           "table": {"ids": len(ids), "hidden": d, "dir": str(head), "sha256": sha256_file(head / "option_rows.safetensors")},
           "cases": len(cases), "group_sets": len(sets), "single_id_cases": single_id_cases,
           "hidden_rows": {"fp32": READOUT_CASES // 2, "fp16_valued": READOUT_CASES - READOUT_CASES // 2, "zero_rows": 1},
           "logits_bit_equal": z_bit, "logits_max_abs_diff": z_max, "p_bit_equal": p_bit, "p_max_abs_diff": p_max,
           "bar": {"p_max_abs_diff": 0.0, "fallback_if_not_bit_equal": READOUT_BAR},
           "answers": {"n": len(acases), "answers_dumps_equal": a_ok, "response_indent2_equal": r_ok}}
    rep["pass"] = (p_bit == len(cases) or p_max <= READOUT_BAR) and a_ok == r_ok == len(acases)
    return rep


# --------------------------------------------------------------------------- negative
def section_negative() -> dict:
    recs = {r["id"]: r for r in fixture_records()}
    rid = {r["id"]: r for r in json.loads(RENDER_IDS.read_text())["records"]}
    changed = copy.deepcopy(recs[NEG_RECORD])
    q = changed["request"]["questions"][NEG_QUESTION]
    words = q["instructions"].split(" ")
    words[1] = NEG_WORD
    q["instructions"] = " ".join(words)
    changed["id"] = "neg_word"
    reordered = copy.deepcopy(recs[KEYORDER_RECORD])
    reordered["request"]["state"] = dict(reversed(list(reordered["request"]["state"].items())))
    reordered["id"] = "neg_keyorder"
    qn = {"a": {"type": "noul", "instructions": "Is the amount over 1000?"}}
    n_int = {"id": "neg_1250", "request": {"state": {"amount": 1250}, "questions": qn}}
    n_flt = {"id": "neg_1250.0", "request": {"state": {"amount": 1250.0}, "questions": qn}}
    inp = write_json(WORK / "negative_in.json", {"records": [changed, reordered, n_int, n_flt]})
    out = WORK / "negative_out.json"
    run("rows", "--records", str(inp), "--tokenizer", str(snapshot()), "--out", str(out))
    sw = {r["id"]: r for r in json.loads(out.read_text())["records"]}
    table = set(option_ids())

    def q_of(rec_id, name):
        return next(x for x in sw[rec_id]["questions"] if x["name"] == name)
    ref_word = next(x for x in rid[NEG_RECORD]["questions"] if x["name"] == NEG_QUESTION)
    ref_order = rid[KEYORDER_RECORD]["questions"][0]
    py_word = py_record(changed, table)["questions"][NEG_QUESTION]
    py_order = py_record(reordered, table)["questions"][ref_order["name"]]
    py_i, py_f = py_record(n_int, table)["questions"]["a"], py_record(n_flt, table)["questions"]["a"]
    arms = {
        "one_word": {"changed_instructions": q["instructions"], "red": q_of("neg_word", NEG_QUESTION)["row_ids"] != ref_word["row_ids"],
                     "swift_equals_python_on_the_changed": q_of("neg_word", NEG_QUESTION)["row_ids"] == py_word["row_ids"]},
        "state_key_order": {"red": q_of("neg_keyorder", ref_order["name"])["text"] != ref_order["text"],
                            "swift_equals_python_on_the_changed": q_of("neg_keyorder", ref_order["name"])["text"] == py_order["text"]},
        "int_vs_float": {"red": q_of("neg_1250", "a")["text"] != q_of("neg_1250.0", "a")["text"],
                         "swift_equals_python_on_both": q_of("neg_1250", "a")["text"] == py_i["text"]
                         and q_of("neg_1250.0", "a")["text"] == py_f["text"]},
    }
    return {"arms": arms, "red": sum(a["red"] for a in arms.values()), "of": len(arms),
            "pass": all(a["red"] for a in arms.values())
            and all(v for a in arms.values() for k, v in a.items() if k.startswith("swift_equals"))}


# --------------------------------------------------------------------------- bundle
def section_bundle() -> dict:
    import export_option_rows as eor
    out = WORK / "bundle_check.json"
    run("bundle-check", "--bundle", str(TOY_BUNDLE), "--out", str(out))
    sw = json.loads(out.read_text())
    ids, rows = eor.table_arrays(TOY_BUNDLE)
    ids_sha = hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()
    toy_ids = {int(i) for i in ids}
    try:   # the toy's table folds the ids (id % 256): a noul's yes / no ids are outside it, the host refuses
        host.build_request({"state": "s", "questions": {"a": {"type": "noul", "instructions": "Is it?"}}}, tok(), table_ids=toy_ids)
        py_noul = "accepted"
    except ValueError as e:
        py_noul = str(e)
    rep = {"bundle": str(TOY_BUNDLE),
           "swift": {k: sw[k] for k in ("metadata", "table", "decide_noul", "decide_score", "load_s")},
           "python_table": {"n": int(len(ids)), "hidden": int(rows.shape[1]), "ids_sha256": ids_sha},
           "python_noul_on_the_toy_table": py_noul, "tokenizer_class": sw["tokenizer"]["class"]}
    rep["toy_gpu_slots"] = toy_slot_readout()
    rep["pass"] = (sw["table"]["n"] == len(ids) and sw["table"]["hidden"] == rows.shape[1]
                   and sw["table"]["ids_sha256"] == ids_sha and sw["decide_noul"] == py_noul
                   and "not wired" in sw["decide_score"] and rep["toy_gpu_slots"]["pass"])
    return rep


def toy_slot_readout() -> dict:
    """Round 2a's toy gate on the Mac GPU (AOT h16c, fp16): every base run's slot hidden row (fp16, from the gate's
    npz shards) through `d1 readout-test` on the toy bundle's own head/option_rows with the toy oracle's groups, against
    the probabilities that gate recorded (host.readout) and host.readout recomputed now: bit for bit."""
    import export_option_rows as eor
    gate = json.loads((RESULTS / "r2a_toy_readout_fp16_tbl2.json").read_text())
    oracle = {r["id"]: r for r in json.loads((LANE / "oracle_toy_tbl2" / "records_oracle.json").read_text())["records"]}
    npz_of = {p["shard"]: p["npz"] for p in gate["processes"] if "shard" in p}
    runs = [r for r in gate["runs"] if r.get("variant", "base") == "base"]
    arrays: dict = {}
    slots, cases, recorded = [], [], []
    for r in runs:
        path = npz_of[r["shard"]]
        if path not in arrays:
            arrays[path] = np.load(path)
        slots.append(arrays[path][f"{r['npz_key']}__slot"].astype(np.float32))
        groups = oracle[r["id"]]["questions"][r["k"]]["groups"]
        cases.append({"hidden_index": len(cases), "groups": groups})
        recorded.append(r["probs"])
    table = eor.read_table(TOY_BUNDLE)
    hpath = WORK / "toy_slots.f32"
    hpath.write_bytes(np.stack(slots).astype("<f4").tobytes())
    cin = write_json(WORK / "toy_slots_in.json", {"table_n": len(table), "cases": cases})
    cout = WORK / "toy_slots_out.json"
    run("readout-test", "--in", str(cin), "--head", str(TOY_BUNDLE / "head"), "--hidden", str(hpath), "--out", str(cout))
    sw = json.loads(cout.read_text())["cases"]
    eq_rec = eq_now = 0
    worst = 0.0
    for c, s, rec in zip(cases, sw, recorded):
        ps = [struct.unpack("<d", struct.pack("<Q", int(b, 16)))[0] for b in s["p_bits"]]
        ids = host.group_ids(c["groups"])
        now_p = host.readout(slots[c["hidden_index"]], eor.rows_for(table, ids), ids, c["groups"])
        eq_rec += [bits(x) for x in ps] == [bits(x) for x in rec]
        eq_now += [bits(x) for x in ps] == [bits(x) for x in now_p]
        worst = max(worst, max(abs(a - b) for a, b in zip(ps, rec)))
    return {"gate": str(RESULTS / "r2a_toy_readout_fp16_tbl2.json"), "runs": len(runs),
            "p_bit_equal_recorded": eq_rec, "p_bit_equal_host_now": eq_now, "p_max_abs_diff_recorded": worst,
            "pass": eq_rec == eq_now == len(runs)}


# --------------------------------------------------------------------------- package / main
def package_record() -> dict:
    files = sorted(p for p in PKG.rglob("*") if p.is_file() and ".build" not in p.parts and ".swiftpm" not in p.parts)
    rec = {"path": str(PKG), "files_sha256": {str(p.relative_to(PKG)): sha256_file(p) for p in files}}
    resolved = PKG / "Package.resolved"
    if resolved.exists():
        rec["resolved"] = {p["identity"]: p["state"].get("version") or p["state"].get("revision")
                           for p in json.loads(resolved.read_text()).get("pins", [])}
    if BIN.exists():
        rec["binary"] = {"path": str(BIN), "sha256": sha256_file(BIN), "bytes": BIN.stat().st_size}
    rec["gate_swift_py_sha256"] = sha256_file(Path(__file__))
    return rec


SECTIONS = {"render": section_render, "ids": section_ids, "image": section_image, "readout": section_readout,
            "negative": section_negative, "bundle": section_bundle}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=[*SECTIONS, "all"])
    ap.add_argument("--out", help="the transcript (all; default results/r3a_swift_text.json)")
    args = ap.parse_args()
    import tokenizers
    names = list(SECTIONS) if args.cmd == "all" else [args.cmd]
    rec: dict = {"schema": "d1-swift-text/1", "generated_at": now(),
                 "gate": "apps/D1 (`d1` CLI, no graph) vs conversion/d1/host.py, vision_host.py and HF tokenizers on the "
                         "snapshot's tokenizer.json, exact (text, ids, plans, readout bits, answer bytes)",
                 "env": {"python": sys.version.split()[0], "tokenizers": tokenizers.__version__, "numpy": np.__version__,
                         "snapshot": str(snapshot()), "tokenizer_json_sha256": sha256_file(snapshot() / "tokenizer.json")},
                 "package": package_record(), "sections": {}}
    WORK.mkdir(parents=True, exist_ok=True)
    for name in names:
        t0 = time.time()
        print(f"[{name}]", flush=True)
        r = SECTIONS[name]()
        r["seconds"] = round(time.time() - t0, 1)
        rec["sections"][name] = r
        print(f"[{name}] {'PASS' if r['pass'] else 'FAIL'} ({r['seconds']} s)", flush=True)
        write_json(WORK / f"section_{name}.json", r)
    rec["pass"] = all(r["pass"] for r in rec["sections"].values())
    if args.cmd == "all":
        out = Path(args.out) if args.out else TRANSCRIPT
        out.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
        print(f"{'PASS' if rec['pass'] else 'FAIL'} -> {out}")
    return 0 if rec["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
