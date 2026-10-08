#!/usr/bin/env python3
"""Swift host gate: the `d1omni` CLI (apps/D1Omni) against the Python host (host.py), the publisher's encode and
the Python runtime on the same compiled graphs.

    PY=~/code/coreai/coreai-models/.venv/bin/python
    $PY conversion/d1_omni/gate_swift.py strings     # the tokenizer's edge strings -> results/swift_tokenize_in.json
    $PY conversion/d1_omni/gate_swift.py rows        # d1omni rows (v1 / v2 / v3) + tokenize -> results/swift_rows_gate.json
    $PY conversion/d1_omni/gate_swift.py pyref       # the Python runtime on the fp16 AOTs (GPU) -> results/swift_ref/
    $PY conversion/d1_omni/gate_swift.py parity      # d1omni parity (aot, jit) -> results/swift_parity_gate.json
    $PY conversion/d1_omni/gate_swift.py --ship pyref|parity|media   # round 10: the same on the stripped bundles
    $PY conversion/d1_omni/gate_swift.py ship-summary                # round 10 -> results/ship_swift_parity.json
    ~/code/standup/tools/quiet/quiet_hold.py d1d-r8-swift-run1 -- $PY conversion/d1_omni/gate_swift.py window --run run1 --cold
    ~/code/standup/tools/quiet/quiet_hold.py d1d-r8-swift-run2 -- $PY conversion/d1_omni/gate_swift.py window --run run2
    $PY conversion/d1_omni/gate_swift.py timing      # d1omni time + the Python control, two windows -> ranking_r8.json

  rows    `d1omni rows` writes, for every record of a fixture version in text mode and in its media mode, each row's
          ids, markers, encode details and the sha256 of five graph inputs at the row's bucket. Against
          results/encode_rows*.json (the publisher's prompt.encode, round 1 / 5 / 6, venv-d1): ids (json.dumps sha256),
          markers, prefix, positions, bucket, max_len, the cuts, budget, per, temperature; against host.py run here on
          the same fixtures: input_ids / pad_mask / prefix_mask / keep_right / qtype_onehot bit for bit (sha256 of the
          array bytes). Against the oracle (the publisher's model on the same fixtures): ids and markers; the Swift
          readout of the oracle's raw marker logits = host.probabilities_from_logits bit for bit; the Swift response
          from the oracle's probabilities = json.dumps(the oracle's response) byte for byte (answer(), Python 3.12's
          sum, the float reprs, the usage). Control: the Swift v1 rows of the three image records rewritten in v2,
          judged against v2's rows, must differ (the comparator can go red). `d1omni tokenize` on the edge strings:
          plain and escaped ids against the `tokenizers` library.
  pyref   the reference the Swift parity is judged against: every row of the round-2 oracle (ref/records_ref.json:
          459 text rows + 11 media rows with the oracle's own prefix) through host.py and the Python Core AI runtime on
          the shipping fp16 AOT .aimodelc of the row's bucket (GPU preferred, explicit options), twice; per row the
          ids, the graph inputs' sha256, the marker logits and probabilities (and their bits), the scores' sha256, the
          oracle's numbers; the media prefixes as raw float32 files for the Swift side. (= runtime_check.py with a
          dump, over every bucket at once.)
  parity  `d1omni parity` (the same rows on the Swift runtime: AOT .aimodelc, and the .aimodel specialized by
          Swift = JIT): the FACTS §7 bar against the oracle (_metrics.summarize), the wrong-pairing control (must
          FAIL), the Python runtime's logits / p (bit-equal rows, max |d|), JIT against AOT.
  window  one measurement window (run it under quiet_hold.py; results/timing/ranking_rule_r8.md §5): the lock must
          hold this run's label; snapshots of the lock, load, swap, GPU jobs and the busiest processes around each
          process; up to 10 min of waiting for other lanes' GPU jobs that do not read the lock; `d1omni time` (W1-W3,
          AOT and JIT alternated; --cold: the d1omni cache entries of its assets removed first), then the Python
          control (`timing_run.py main` on the same AOTs) -> run3_swift_<run>.json, r8-<run>-py_main.json,
          run3_swift_<run>_window.json.
  timing  the two measurement windows of round 8 (`d1omni time` + `timing_run.py main` on the same fp16 bundles,
          one quiet_hold window each): per workload and form the median of each window, the score (the mean of the
          two medians), the co-candidates within 3 % (results/timing/ranking_rule_r8.md, written before the windows).
  --ship  (round 10) pyref, parity and media on the stripped bundles (strip_ship.py: bundles/.../macos-ship/, their
          AOT in compiled/ship-h16c/, aot_ship.py), every file and folder they read or write named ship_<name>
          (results/ship_swift_ref/, ship_swift_parity_{aot,jit}.json, ship_swift_ref_media/ ...).
  ship-summary  (round 10) the Swift parity on the stripped bundles in one verdict, results/ship_swift_parity.json:
          text 470 + image 46 + audio 46 rows, AOT and JIT, bit for bit against the Python runtime on the same stripped
          AOT, the FACTS §7 bar and the controls; and the stripped against the unstripped bundles (the Python references
          of round 8 / 9 and the Swift outputs, bit for bit).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import hf_snapshot, work_path  # noqa: E402

import host  # noqa: E402

WORK = work_path("_d1_omni")
RESULTS = WORK / "results"
MACOS = WORK / "bundles" / "d1-omni-600m" / "macos"
COMPILED = WORK / "compiled"
BIN = WORK / "swift" / ".build" / "out" / "Products" / "Release" / "d1omni"
VERSIONS = {  # fixture version -> (fixtures file, encode_rows file, oracle file, Swift rows file)
    "v1": ("fixtures/records.v1.json", "results/encode_rows.json", "ref/records_ref.json", "results/swift_rows.json"),
    "v2": ("fixtures/records.v2.json", "results/encode_rows.v2.json", "ref/records_ref_images.json", "results/swift_rows.v2.json"),
    "v3": ("fixtures/records.json", "results/encode_rows.v3.json", "ref/records_ref_audio.json", "results/swift_rows.v3.json"),
}
ROW_FIELDS = ("ids_sha256", "markers", "prefix", "n_ids", "positions", "bucket", "max_len", "q_delim_pos", "state_tokens",
              "state_room", "state_cut", "instructions_tokens", "instructions_cut", "budget", "per_option", "option_cut",
              "row_cut", "calibrate", "type", "K")
GRAPH_INPUTS = ("input_ids", "pad_mask", "prefix_mask", "keep_right", "qtype_onehot")
# The shipped decision buckets: round 13 adds round 12's L64 / L128 (the supervisor's ship decision, 2026-10-08) =
# host.ALL_BUCKETS. A record of the ship form only: no subcommand routes by it (pyref routes by PYREF_BUCKETS below, the
# Swift host by the folders it is given), so the round 8-12 runs and their results are unchanged.
SHIP_BUCKETS = host.ALL_BUCKETS
SHIP_MODE = False  # --ship (round 10): the stripped bundles (macos-ship/, compiled/ship-h16c/), results named ship_*
# --small (round 12): the stripped small decision buckets (macos-ship-small/fp16-L64, -L128; their AOT in
# compiled/ship-h16c/), the oracle rows whose bucket under host.ALL_BUCKETS is 64 or 128, results named small_*
SMALL_MODE = False
PYREF_BUCKETS = host.BUCKETS  # the bucket set pyref routes a row by (--small: host.ALL_BUCKETS)


def rname(stem: str) -> str:
    """A results file or folder name; --ship prefixes ship_ (round 10: the runs on the stripped bundles), --small
    prefixes small_ (round 12: the small decision buckets)."""
    return f"ship_{stem}" if SHIP_MODE else f"small_{stem}" if SMALL_MODE else stem


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=True) + "\n")
    return path


def f32_bits(values) -> list[int]:
    return [int(v) for v in np.asarray(values, dtype=np.float32).view(np.uint32)]


def tokenizer() -> host.RawTokenizer:
    tok = host.RawTokenizer(Path(hf_snapshot(host.MODEL_ID, revision=host.MODEL_SHA)) / "tokenizer.json")
    host.check_token_ids(tok)
    return tok


def media_of(record: dict) -> tuple[str | None, int]:
    """encode_rows.py's media_of without its torch import: (mode, prefix length) from the provenance's sizes."""
    media = record.get("media")
    if media is None:
        return None, 0
    prov = record["provenance"]
    if "images" in media:
        size = (prov.get("image_info") or {}).get("px") or prov.get("px")
        return "image", host.image_prefix_length([tuple(size)])
    samples = prov["clip"]["frames"] if "clip" in prov else prov["audio_info"]["frames"]
    return "audio", host.audio_prefix_length(samples)


def python_rows(tok, record: dict, mode: str) -> list[host.Row]:
    request = record["request"]
    prefix = 0 if mode == "text" else media_of(record)[1]
    return host.request_rows(tok, request.get("state"), request["questions"], mode, prefix)


# =========================================================================== strings
EDGE_STRINGS = [
    "", " ", "  ", "\t", "\n", "\r\n", "a\u000bb", "a\u000cb", "a\u0085b", "a b", "a b", "a　b", "x​y",
    "x‍y", "x﻿y", "x­y", "trailing   ", "  leading", "\n\n\n", "a  b", "a \n\n b", "a\r\nb\r\n\r\nc",
    "café", "café", "é́", "́abc", "naïve résumé", "Ångström",
    "ﬁne", "Straße", "İstanbul", "ǅ", "ſs 'ſ 'S 'Ll",
    "नमस्ते दुनिया", "สวัสดีครับ",
    "مَرْحَبًا", "שָׁלוֹם",
    "안녕하세요", "각", "こんにちは世界、テスト。",
    "你好，世界！", "\U0001f44d\U0001f3fd \U0001f468‍\U0001f469‍\U0001f467 \U0001f1ef\U0001f1f5 ❤️",
    "1234567 3.14159 1e10 ½ ① ٣", "12345678901234567890", "Ｆｕｌｌｗｉｄｔｈ ＡＢＣ",
    " private use", "python", "Python", "pythonic", " python ", "monty_python", "a python b", "Mathias", "Mathias's",
    "<think>", "</think>", "<think>x</think>", "<image>", "<|pad|>", "<|mask|>", "<|reserved_7|>", "<|startoftext|>",
    "<|not_a_token|>", "<|a b|>", "<||>", "<|é|>", "<|<|mask|>|>", "<¦mask¦>", "x<|im_end|>y",
    "I'm you'RE it'S they'll we'd can't 's", "’s it’s", "{\"a\": 1, \"b\": [true, null]}",
    "Refund the second charge of $12.50 on 2026-10-08, please!!!", "tabs\tand\ttabs", "end.\n",
    "  \n  x", "x\n  ", "  lead nbsp", "mixed é́ marks कि",
]


def cmd_strings(args) -> int:
    out = write_json(RESULTS / "swift_tokenize_in.json", {"strings": EDGE_STRINGS})
    print(f"{len(EDGE_STRINGS)} strings -> {out}")
    return 0


# =========================================================================== rows
def compare_version(name: str, tok) -> dict:
    fixtures_rel, encode_rel, oracle_rel, swift_rel = VERSIONS[name]
    swift = json.loads((WORK / swift_rel).read_text())
    encode = json.loads((WORK / encode_rel).read_text())
    fixtures = json.loads((WORK / fixtures_rel).read_text())
    if swift["fixtures_sha256"] != encode["fixtures_sha256"]:
        raise SystemExit(f"{name}: Swift rows of {swift['fixtures_sha256'][:8]}, encode_rows of {encode['fixtures_sha256'][:8]}")
    key = lambda r: (r["id"], r["mode"], r["qid"])  # noqa: E731
    sw = {key(r): r for r in swift["rows"]}
    py = {key(r): r for r in encode["rows"]}
    field_bad: dict[str, list] = {f: [] for f in ROW_FIELDS + ("T", "temperature_key")}
    for k, p in py.items():
        s = sw.get(k)
        if s is None:
            continue
        for f in ROW_FIELDS:
            if s.get(f) != p.get(f):
                field_bad[f].append("/".join(k))
        if "T" in p and (s.get("T") != p["T"] or s.get("temperature_key") != p.get("temperature_key")):
            field_bad["T"].append("/".join(k))
    # the graph inputs: host.py here on the same fixtures (ids checked against encode_rows' sha first)
    records = {r["id"]: r for r in fixtures["records"]}
    graph_bad = {f: [] for f in GRAPH_INPUTS}
    host_ids_bad, graph_rows = [], 0
    for (rid, mode, qid), s in sw.items():
        rows = {r.qid: r for r in python_rows(tok, records[rid], mode)}
        r = rows[qid]
        if hashlib.sha256(json.dumps(r.ids).encode()).hexdigest() != py[(rid, mode, qid)]["ids_sha256"]:
            host_ids_bad.append(f"{rid}/{mode}/{qid}")
        L = host.bucket_for(r.positions)
        if L is None or "graph" not in s:
            continue
        prefix = np.zeros((r.prefix_len, 1024), dtype=np.float32) if r.prefix_len else None
        inputs, markers = host.graph_inputs(r, L, prefix)
        graph_rows += 1
        if s["graph"]["L"] != L or s["graph"]["markers"] != markers:
            graph_bad["input_ids"].append(f"{rid}/{mode}/{qid} (L / markers)")
        for f in GRAPH_INPUTS:
            if sha256_array(inputs[f]) != s["graph"]["sha256"][f]:
                graph_bad[f].append(f"{rid}/{mode}/{qid}")
    # the oracle: ids, markers, the readout of its raw logits, the response
    oracle = json.loads((WORK / oracle_rel).read_text())
    o_by = {(r["id"], r["mode"]): r for r in oracle["records"]}
    o_rows = o_ids_bad = o_markers_bad = o_p_bits_bad = 0
    o_p_max_vs_oracle = 0.0
    resp_bad, resp_n = [], 0
    for item in swift.get("oracle", []):
        rec = o_by[(item["id"], item["mode"])]
        qs = {q["qid"]: q for q in rec["questions"]}
        resp_n += 1
        if item["response_json"] != json.dumps(rec["response"]):
            resp_bad.append(f"{item['id']}/{item['mode']}")
        for sq in item["questions"]:
            q = qs[sq["qid"]]
            o_rows += 1
            o_ids_bad += not sq["ids_equal"]
            o_markers_bad += not sq["markers_equal"]
            question = host.as_question(records[item["id"]]["request"]["questions"][sq["qid"]])
            p = host.probabilities_from_logits(q["logits_raw"], question, q["calibrate"])
            o_p_bits_bad += sq["probs_bits"] != f32_bits(p)
            o_p_max_vs_oracle = max(o_p_max_vs_oracle, max(abs(a - b) for a, b in zip(sq["probs"], q["probs"])))
    out = {
        "fixtures": fixtures_rel, "fixtures_sha256": swift["fixtures_sha256"], "encode_rows": encode_rel,
        "rows": {"swift": len(sw), "python": len(py), "keys_equal": set(sw) == set(py),
                 "missing_in_swift": sorted("/".join(k) for k in set(py) - set(sw))[:20],
                 "extra_in_swift": sorted("/".join(k) for k in set(sw) - set(py))[:20]},
        "fields_equal": {f: f"{len(py) - len(v)}/{len(py)}" for f, v in field_bad.items() if f != "T"}
        | {"T": f"{sum(1 for p in py.values() if 'T' in p) - len(field_bad['T'])}/{sum(1 for p in py.values() if 'T' in p)}"},
        "fields_first_differences": {f: v[:10] for f, v in field_bad.items() if v},
        "host_here_ids_vs_encode_rows": {"bad": host_ids_bad[:10], "equal": len(sw) - len(host_ids_bad)},
        "graph_inputs_equal": {f: f"{graph_rows - len(v)}/{graph_rows}" for f, v in graph_bad.items()},
        "graph_inputs_first_differences": {f: v[:10] for f, v in graph_bad.items() if v},
        "oracle": {"file": oracle_rel, "rows": o_rows, "ids_equal": o_rows - o_ids_bad, "markers_equal": o_rows - o_markers_bad,
                   "probs_from_raw_logits_bit_equal_host": o_rows - o_p_bits_bad,
                   "probs_from_raw_logits_max_abs_d_vs_oracle_probs": o_p_max_vs_oracle,
                   "responses": resp_n, "responses_equal": resp_n - len(resp_bad), "responses_different": resp_bad[:10]},
        "swift_errors": swift.get("errors", []),
        "swift_timing": {"tokenizer_load_seconds": swift["tokenizer_load_seconds"], "encode_seconds_total": swift["encode_seconds_total"],
                         "encode_ms_per_request": swift["encode_ms_per_request"]},
    }
    ok = (out["rows"]["keys_equal"] and not any(field_bad.values()) and not host_ids_bad and not any(graph_bad.values())
          and o_ids_bad == 0 and o_markers_bad == 0 and o_p_bits_bad == 0 and not resp_bad and not swift.get("errors"))
    out["status"] = "PASS" if ok else "FAIL"
    return out


CONTROL_EDIT = ("card_text", "refund", "Is the customer asking for a refund?", "Is the customer asking for a return?")


def control_mutated(tok) -> dict:
    """The comparator can go red: fixtures v1 with one word of one question changed (card_text/refund, "refund" ->
    "return"), through `d1omni rows`, judged against encode_rows of the unchanged v1. Exactly the changed question's
    rows (text mode) must differ, every other row must not."""
    fixtures = json.loads((WORK / VERSIONS["v1"][0]).read_text())
    rid, qid, old, new = CONTROL_EDIT
    rec = next(r for r in fixtures["records"] if r["id"] == rid)
    q = rec["request"]["questions"][qid]
    if q["instructions"] != old:
        return {"status": "not run", "why": f"{rid}/{qid} instructions are {q['instructions']!r}, not {old!r}"}
    q["instructions"] = new
    path = write_json(RESULTS / "swift_rows_control_fixtures.json", fixtures)
    out = RESULTS / "swift_rows_control.json"
    done = subprocess.run([str(BIN), "rows", "--bundle-dir", str(MACOS), "--fixtures", str(path), "--out", str(out)],
                          capture_output=True, text=True)
    if done.returncode != 0:
        return {"status": "not run", "why": done.stderr[-500:]}
    swift = {(r["id"], r["mode"], r["qid"]): r for r in json.loads(out.read_text())["rows"]}
    py = {(r["id"], r["mode"], r["qid"]): r for r in json.loads((WORK / VERSIONS["v1"][1]).read_text())["rows"]}
    flagged = sorted("/".join(k) for k, p in py.items() if swift[k]["ids_sha256"] != p["ids_sha256"])
    want = [f"{rid}/text/{qid}"]
    return {"what": f"{rid}/{qid}: {old!r} -> {new!r} in a copy of fixtures v1, Swift rows against encode_rows v1",
            "rows": len(py), "rows_flagged": flagged, "must_flag_exactly": want,
            "status": "FAIL (as it must, on that row only)" if flagged == want else "the control did not behave"}


def tokenize_check(tok) -> dict | None:
    path = RESULTS / "swift_tokenize.json"
    if not path.exists():
        return None
    items = json.loads(path.read_text())["items"]
    bad_plain, bad_enc = [], []
    for it in items:
        s = it["text"]
        if tok._tok.encode(s, add_special_tokens=False).ids != it["plain"] if s else it["plain"] != []:
            bad_plain.append(s)
        if (tok(host.escape(s), add_special_tokens=False)["input_ids"] if s else []) != it["enc"] or host.escape(s) != it["escaped"]:
            bad_enc.append(s)
    return {"strings": len(items), "plain_equal": len(items) - len(bad_plain), "enc_equal": len(items) - len(bad_enc),
            "plain_different": [repr(s) for s in bad_plain], "enc_different": [repr(s) for s in bad_enc]}


def cmd_rows(args) -> int:
    tok = tokenizer()
    out = {"schema": "d1-omni-swift-rows-gate/1", "written": now(), "binary": str(BIN),
           "binary_sha256": sha256_file(BIN) if BIN.exists() else None, "versions": {}}
    for name in VERSIONS:
        if (WORK / VERSIONS[name][3]).exists():
            out["versions"][name] = compare_version(name, tok)
    out["control"] = control_mutated(tok)
    out["tokenize_edge_strings"] = tokenize_check(tok)
    out["code_sha256"] = {f: sha256_file(HERE / f) for f in ("gate_swift.py", "host.py")}
    v1 = out["versions"].get("v1", {})
    out["acceptance"] = {"rows": v1.get("rows", {}).get("swift"), "status_v1": v1.get("status"),
                         "status_all": "PASS" if all(v["status"] == "PASS" for v in out["versions"].values()) else "FAIL"}
    path = write_json(RESULTS / "swift_rows_gate.json", out)
    for name, v in out["versions"].items():
        print(name, v["status"], v["rows"]["swift"], "rows;", "fields", v["fields_equal"]["ids_sha256"], "ids,",
              "graph", v["graph_inputs_equal"], "oracle", {k: v["oracle"][k] for k in ("rows", "ids_equal",
              "probs_from_raw_logits_bit_equal_host", "responses_equal")})
    print("control:", out["control"]["status"], "| tokenize:", {k: out["tokenize_edge_strings"][k] for k in ("strings", "plain_equal", "enc_equal")}
          if out["tokenize_edge_strings"] else None, "->", path)
    return 0 if out["acceptance"]["status_all"] == "PASS" else 1


# =========================================================================== pyref (GPU)
def ship_aot(L: int) -> Path:
    """The Mac AOT .aimodelc of the fp16 bucket L (compiled from bundles/.../fp16-L<L>, gpu, h16c)."""
    source = MACOS / f"fp16-L{L}"
    found = []
    for d in sorted(COMPILED.iterdir()):
        m = d / "provenance" / "aot-manifest.json"
        if not m.exists():
            continue
        aot = json.loads(m.read_text())
        if (aot.get("status") == "COMPILED" and aot.get("preferred_compute") == "gpu" and aot.get("architecture") == "h16c"
                and Path(aot["source"]["folder"]) == source and (d / aot["aimodelc"]).exists()):
            found.append(d / aot["aimodelc"])
    if len(found) != 1:
        raise SystemExit(f"{len(found)} AOT assets for {source}: {found}")
    return found[0]


async def pyref_async(out_dir: Path) -> dict:
    import coreai.runtime as rt

    tok = tokenizer()
    oracle_path = WORK / "ref" / "records_ref.json"
    oracle = json.loads(oracle_path.read_text())
    fixtures_path = WORK / VERSIONS["v1"][0]
    assert sha256_file(fixtures_path) == oracle["fixtures"]["sha256"], "records_ref.json is not of fixtures v1"
    records = {r["id"]: r for r in json.loads(fixtures_path.read_text())["records"]}
    opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    fns, models, loads = {}, {}, {}
    rows_out, requests = [], []
    (out_dir / "prefix").mkdir(parents=True, exist_ok=True)
    for rec in oracle["records"]:
        rid, mode = rec["id"], rec["mode"]
        hrows = python_rows(tok, records[rid], mode)
        by_q = {r.qid: r for r in hrows}
        prefix, prefix_file = None, None
        if mode != "text":
            pin = oracle["npz"][rid]
            npz = WORK / "ref" / "npz" / f"{rid}.npz"
            if sha256_file(npz) != pin["sha256"]:
                raise SystemExit(f"{npz} differs from records_ref.json's pin")
            with np.load(npz) as z:
                prefix = np.ascontiguousarray(z["prefix"], dtype=np.float32)
            prefix_file = f"prefix/{rid}.f32"
            (out_dir / prefix_file).write_bytes(prefix.tobytes())
        probs_for_response = []
        for q in rec["questions"]:
            r = by_q[q["qid"]]
            if r.ids != q["ids"] or r.markers != q["markers"] or r.positions != q["positions"]:
                raise SystemExit(f"{rid}/{q['qid']}/{mode}: host row differs from the oracle's")
            L = host.bucket_for(r.positions, PYREF_BUCKETS)
            if SMALL_MODE and L not in host.SMALL_BUCKETS:
                continue  # --small: only the rows the small buckets take
            if L not in fns:
                aimodelc = ship_aot(L)
                t0 = time.perf_counter()
                models[L] = await rt.AIModel.load(aimodelc, opts)
                fns[L] = models[L].load_function("main")
                loads[L] = {"aimodelc": str(aimodelc), "load_s": time.perf_counter() - t0,
                            "main_hash": (aimodelc / "main.hash").read_bytes().hex() if (aimodelc / "main.hash").exists() else None}
                print(f"loaded L{L} {aimodelc.name} in {loads[L]['load_s']:.2f} s", flush=True)
            inputs, markers = host.graph_inputs(r, L, prefix)
            outs = []
            for _ in range(2):
                res = await fns[L]({k: rt.NDArray(inputs[k]) for k in host_input_names()})
                outs.append(np.array(res["scores"].numpy(), copy=True).reshape(-1))
            z = outs[0][markers]
            p = host.probabilities_from_logits(z, r.question, r.calibrate)
            probs_for_response.append(p)
            rows_out.append({
                "id": rid, "qid": q["qid"], "mode": mode, "source": rec["source"], "type": q["type"], "K": q["K"],
                "prefix_len": r.prefix_len, "positions": r.positions, "bucket": L, "markers": r.markers,
                "graph_markers": markers, "ids": r.ids, "ids_sha256": hashlib.sha256(json.dumps(r.ids).encode()).hexdigest(),
                "inputs_sha256": {k: sha256_array(inputs[k]) for k in host_input_names()},
                "prefix_file": prefix_file, "calibrate": r.calibrate, "temperature_key": host.temperature_key(r.question),
                "T": host.temperature(r.question) if r.calibrate else None,
                "question": {"type": r.question.type, "instructions": r.question.instructions, "criteria": r.question.criteria},
                "logits": [float(v) for v in z], "logits_bits": f32_bits(z), "probs": [float(v) for v in p], "probs_bits": f32_bits(p),
                "scores_sha256": sha256_array(outs[0][: r.positions]),
                "drift_scores": float(np.max(np.abs(outs[1][: r.positions].astype(np.float64) - outs[0][: r.positions]))),
                "oracle_logits_raw": q["logits_raw"], "oracle_probs": q["probs"], "argmax_index": q["argmax_index"],
                "near_tie": q["near_tie"], "top2_margin": q["top2_margin"]})
        if len(probs_for_response) == len(rec["questions"]):  # --small: a request with a longer row has no response
            requests.append({"id": rid, "mode": mode, "response_json": json.dumps(host.response(hrows, probs_for_response))})
    assert models
    what = ("the Python Core AI runtime (coreai-core) on the shipping fp16 AOT .aimodelc of each row's bucket, GPU "
            "preferred, every row of ref/records_ref.json, 2 calls")
    if SMALL_MODE:
        what = ("round 12: the Python Core AI runtime (coreai-core) on the stripped fp16 L64 / L128 AOT .aimodelc, GPU "
                "preferred, every row of ref/records_ref.json whose bucket under host.ALL_BUCKETS is 64 or 128, 2 calls")
    return {"schema": "d1-omni-swift-ref/1", "written": now(), "what": what, "buckets": list(PYREF_BUCKETS),
            "specialization_options": " ".join(str(opts).split()), "fixtures": {"path": str(fixtures_path),
            "sha256": sha256_file(fixtures_path)}, "oracle": {"path": str(oracle_path), "sha256": sha256_file(oracle_path)},
            "ship": SHIP_MODE, "macos": str(MACOS), "compiled": str(COMPILED),
            "loads": {str(k): v for k, v in loads.items()}, "rows": rows_out, "requests": requests,
            "code_sha256": {f: sha256_file(HERE / f) for f in ("gate_swift.py", "host.py", "_metrics.py")}}


def host_input_names() -> tuple:
    return ("input_ids", "prefix_embeds", "pad_mask", "prefix_mask", "keep_right", "qtype_onehot")


def cmd_pyref(args) -> int:
    out_dir = RESULTS / rname("swift_ref")
    if (out_dir / "swift_ref.json").exists():
        raise SystemExit(f"{out_dir / 'swift_ref.json'} exists: never replaced")
    t0 = time.perf_counter()
    doc = asyncio.run(pyref_async(out_dir))
    doc["seconds"] = time.perf_counter() - t0
    path = write_json(out_dir / "swift_ref.json", doc)
    drift = max(r["drift_scores"] for r in doc["rows"])
    print(f"{len(doc['rows'])} rows, {len(doc['requests'])} requests, drift {drift}, {doc['seconds']:.1f} s -> {path}")
    return 0


# =========================================================================== parity
def metrics_row(ref: dict) -> dict:
    """A swift_ref row in the shape _metrics.row_record reads."""
    return {"id": ref["id"], "qid": ref["qid"], "mode": ref["mode"], "source": ref["source"], "type": ref["type"],
            "K": ref["K"], "positions": ref["positions"], "near_tie": ref["near_tie"], "top2_margin": ref["top2_margin"],
            "argmax_index": ref["argmax_index"], "calibrate": ref["calibrate"], "question": ref["question"],
            "oracle": {"logits_raw": ref["oracle_logits_raw"], "probs": ref["oracle_probs"]}}


def score_parity(swift: dict, ref: dict) -> dict:
    from _metrics import row_record, summarize, wrong_pairing

    by = {(r["id"], r["mode"], r["qid"]): r for r in swift["rows"]}
    rows, records, py_bad, py_dl, py_dp, bits_equal = [], [], [], 0.0, 0.0, 0
    for r in ref["rows"]:
        s = by[(r["id"], r["mode"], r["qid"])]
        m = metrics_row(r)
        rec = row_record(m, s["logits"])
        rows.append(m)
        records.append(rec)
        same = s["logits_bits"] == r["logits_bits"] and s["probs_bits"] == r["probs_bits"]
        bits_equal += same
        if not same:
            py_bad.append(f"{r['id']}/{r['qid']}/{r['mode']}")
        py_dl = max(py_dl, max(abs(a - b) for a, b in zip(s["logits"], r["logits"])))
        py_dp = max(py_dp, max(abs(a - b) for a, b in zip(s["probs"], r["probs"])))
        if [float(v) for v in host.probabilities_from_logits(s["logits"], host.as_question(r["question"]), r["calibrate"])] != s["probs"]:
            raise SystemExit(f"{r['id']}/{r['qid']}: the Swift p is not host.py's p of the Swift logits")
    summary = summarize(records)
    control = wrong_pairing(rows, records)
    text = [i for i, r in enumerate(rows) if r["mode"] == "text"]
    return {"asset_kind": swift["setup"]["kind"], "rows": len(records), "text_rows": len(text),
            "media_rows_with_oracle_prefix": len(records) - len(text),
            "ship_bar": summary["ship_bar"], "argmax_equal": summary["argmax_equal"], "non_near_tie": summary["non_near_tie"],
            "near_tie": summary["near_tie"], "max_abs_dp": summary["max_abs_dp"], "max_abs_dp_row": summary["max_abs_dp_row"],
            "mean_row_max_abs_dp": summary["mean_row_max_abs_dp"], "max_abs_dlogit_vs_oracle": summary["max_abs_dlogit"],
            "text_only": {k: v for k, v in summarize([records[i] for i in text]).items()
                          if k in ("rows", "argmax_equal", "max_abs_dp", "mean_row_max_abs_dp", "ship_bar")},
            "wrong_pairing_control": control,
            "python_runtime": {"rows_bit_equal_logits_and_p": bits_equal, "max_abs_dlogit": py_dl, "max_abs_dp": py_dp,
                               "first_different": py_bad[:10]},
            "drift_marker_logits_max": max(r["drift_marker_logits"] for r in swift["rows"]),
            "loads": swift["loads"], "status": "PASS" if summary["ship_bar"]["status"] == "PASS" and control["caught"] else "FAIL"}


def cmd_parity(args) -> int:
    ref_path = RESULTS / rname("swift_ref") / "swift_ref.json"
    ref = json.loads(ref_path.read_text())
    out = {"schema": "d1-omni-swift-parity-gate/1", "written": now(), "reference": str(ref_path), "ship": SHIP_MODE}
    docs = {}
    for kind in ("aot", "jit"):
        path = RESULTS / rname(f"swift_parity_{kind}.json")
        if path.exists():
            docs[kind] = json.loads(path.read_text())
            out[kind] = score_parity(docs[kind], ref) | {"file": str(path)}
    if "aot" in docs and "jit" in docs:
        a = {(r["id"], r["mode"], r["qid"]): r for r in docs["aot"]["rows"]}
        equal, dl, dp, first = 0, 0.0, 0.0, []
        for r in docs["jit"]["rows"]:
            x = a[(r["id"], r["mode"], r["qid"])]
            same = r["logits_bits"] == x["logits_bits"] and r["probs_bits"] == x["probs_bits"]
            equal += same
            if not same:
                first.append(f"{r['id']}/{r['qid']}/{r['mode']}")
            dl = max(dl, max(abs(p - q) for p, q in zip(r["logits"], x["logits"])))
            dp = max(dp, max(abs(p - q) for p, q in zip(r["probs"], x["probs"])))
        out["jit_vs_aot"] = {"rows": len(docs["jit"]["rows"]), "bit_equal": equal, "max_abs_dlogit": dl, "max_abs_dp": dp,
                             "first_different": first[:10]}
    if SMALL_MODE:  # round 12: the verdict per small bucket, and the launch's file name
        from _metrics import row_record, summarize

        out["buckets"] = list(PYREF_BUCKETS)
        out["by_bucket"] = {}
        for L in host.SMALL_BUCKETS:
            refs = [r for r in ref["rows"] if r["bucket"] == L]
            entry = {"rows": len(refs), "sources": dict(sorted({s: sum(1 for r in refs if r["source"] == s)
                                                               for s in {r["source"] for r in refs}}.items()))}
            for kind, doc in docs.items():
                by = {(r["id"], r["mode"], r["qid"]): r for r in doc["rows"]}
                recs = [row_record(metrics_row(r), by[(r["id"], r["mode"], r["qid"])]["logits"]) for r in refs]
                s = summarize(recs)
                entry[kind] = {"ship_bar": s["ship_bar"], "argmax_equal": s["argmax_equal"],
                               "non_near_tie": s["non_near_tie"], "near_tie": s["near_tie"],
                               "max_abs_dp": s["max_abs_dp"], "mean_row_max_abs_dp": s["mean_row_max_abs_dp"],
                               "max_abs_dlogit": s["max_abs_dlogit"],
                               "python_bit_equal": sum(by[(r["id"], r["mode"], r["qid"])]["logits_bits"] == r["logits_bits"]
                                                       and by[(r["id"], r["mode"], r["qid"])]["probs_bits"] == r["probs_bits"]
                                                       for r in refs),
                               "swift_bucket_equal": all(by[(r["id"], r["mode"], r["qid"])]["bucket"] == L for r in refs)}
            out["by_bucket"][str(L)] = entry
        bits = all(out[k]["python_runtime"]["rows_bit_equal_logits_and_p"] == out[k]["rows"] for k in ("aot", "jit") if k in out)
        out["status"] = ("PASS" if {"aot", "jit"} <= set(docs) and bits and out.get("jit_vs_aot", {}).get("bit_equal")
                         == out["jit_vs_aot"]["rows"] and all(out[k]["status"] == "PASS" for k in ("aot", "jit"))
                         else "FAIL")
        final = RESULTS / "swift_parity_L64_128.json"
        if final.exists():
            raise SystemExit(f"{final} exists: never replaced")
        write_json(final, out)
        print("small buckets:", out["status"], {L: {k: (v[k]["python_bit_equal"], v["rows"]) for k in ("aot", "jit") if k in v}
                                                for L, v in out["by_bucket"].items()}, "->", final)
    path = write_json(RESULTS / rname("swift_parity_gate.json"), out)
    for kind in ("aot", "jit"):
        if kind in out:
            s = out[kind]
            print(kind, s["status"], f"rows {s['rows']} (text {s['text_rows']})", f"argmax {s['argmax_equal']}/{s['rows']}",
                  f"max|dp| {s['max_abs_dp']:.3e} mean {s['mean_row_max_abs_dp']:.3e}",
                  f"control {s['wrong_pairing_control']['status']}",
                  f"python bit-equal {s['python_runtime']['rows_bit_equal_logits_and_p']}/{s['rows']} max|dlogit| {s['python_runtime']['max_abs_dlogit']:.3e}")
    if "jit_vs_aot" in out:
        print("jit vs aot", out["jit_vs_aot"])
    print("->", path)
    return 0 if all(out[k]["status"] == "PASS" for k in ("aot", "jit") if k in out) else 1


# =========================================================================== resize (round 9, venv-d1)
def media_images() -> list[dict]:
    """The 16 images of the fixtures (img_01..03, imgm_01..12 and the card's cats), in reference order, with their
    reference npz (the publisher's preprocess() output) and the fixture record."""
    fixtures = {r["id"]: r for r in json.loads((WORK / "fixtures" / "records.json").read_text())["records"]}
    out = []
    for ref_name, only in (("records_ref_images.json", None), ("records_ref.json", "card_cats")):
        ref = json.loads((WORK / "ref" / ref_name).read_text())
        for entry in ref["records"]:
            if entry["mode"] != "image" or (only and entry["id"] != only):
                continue
            npz = WORK / "ref" / "npz" / f"{entry['id']}.npz"
            if sha256_file(npz) != ref["npz"][entry["id"]]["sha256"]:
                raise SystemExit(f"{npz} differs from {ref_name}'s pin")
            record = fixtures[entry["id"]]
            out.append({"id": entry["id"], "file": WORK / record["media"]["images"][0], "public": record["public"],
                        "npz": npz, "record": record, "entry": entry, "reference": ref_name})
    return out


def _float64_resize(crop: np.ndarray, height: int, width: int) -> np.ndarray:
    """Control: the same triangle filter with float64 weights and float64 sums (PIL's form, no float32 rounding)."""
    def weights(n_in, n_out):
        scale = n_in / n_out
        support = max(1.0, scale)
        m = np.zeros((n_out, n_in))
        for i in range(n_out):
            centre = (i + 0.5) * scale
            lo, hi = max(int(centre - support + 0.5), 0), min(int(centre + support + 0.5), n_in)
            xs = np.arange(lo, hi)
            w = np.clip(1.0 - np.abs((xs - centre + 0.5) / support), 0.0, None)
            m[i, lo:hi] = w / w.sum()
        return m
    _, h, w = crop.shape
    if (height, width) == (h, w):
        return crop
    x = crop.astype(np.float64)
    if width != w:
        x = np.einsum("chw,ow->cho", x, weights(w, width))
    if height != h:
        x = np.einsum("chw,oh->cow", x, weights(h, height))
    return np.rint(x).astype(np.uint8)


def _crops_with(rgb: np.ndarray, resize) -> list[np.ndarray]:
    """host.crop_pixels_numpy with another resize function (the controls)."""
    _, h, w = rgb.shape
    plan = host.layout(w, h)
    crops = []
    if plan["tiled"]:
        gw, gh = plan["grid"]
        big = resize(rgb, gh * host.TILE, gw * host.TILE)
        crops = [big[:, r * host.TILE:(r + 1) * host.TILE, c * host.TILE:(c + 1) * host.TILE]
                 for r in range(gh) for c in range(gw)]
    crops.append(resize(rgb, *plan["thumbnail"]))
    return crops


def _level_diff(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"shape": [list(a.shape), list(b.shape)], "bit_equal": False}
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {"bit_equal": bool(not d.any()), "diff_elems": int((d > 0).sum()), "max_level_diff": int(d.max()),
            "diff_1_level": int((d == 1).sum()), "diff_over_1_level": int((d > 1).sum()), "elems": int(d.size)}


def cmd_resize(args) -> int:
    """host.resize_uint8_antialias_numpy (crop_pixels_numpy) vs torchvision (host.image_crop_pixels, the publisher's
    call) on every crop of the 16 fixture images; the patches of the NumPy crops vs the publisher's preprocess() output
    (the oracle npz); controls that must differ: a float64 filter, PIL's BILINEAR resize, round half up."""
    import torch
    import torchvision
    from PIL import Image
    from torchvision.transforms.v2.functional import _geometry as tvg

    out_path = RESULTS / "resize_numpy_gate.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists: never replaced")
    t_start = time.perf_counter()
    images, records = media_images(), []
    totals = {"crops": 0, "bit_equal": 0, "resized_crops": 0, "diff_elems": 0, "max_level_diff": 0,
              "pixel_values_equal_oracle": 0}
    controls = {name: {"crops_differing": 0, "diff_elems": 0, "max_level_diff": 0, "resized_crops": 0}
                for name in ("float64_filter", "pil_bilinear", "round_half_up")}
    numpy_s = 0.0
    for img in images:
        with Image.open(img["file"]) as pil:
            pil.load()
            fmt, info_keys = pil.format, sorted(pil.info.keys())
            exif_orientation = pil.getexif().get(0x0112)
            tv = host.image_crop_pixels(pil)
            rgb = host.rgb_uint8(pil)
            t0 = time.perf_counter()
            mine = host.crop_pixels_numpy(rgb)
            numpy_s += time.perf_counter() - t0
            f64 = _crops_with(rgb, _float64_resize)

            def pil_resize(a, hh, ww):
                if (hh, ww) == tuple(a.shape[1:]):
                    return a
                im = Image.fromarray(np.ascontiguousarray(a.transpose(1, 2, 0)))
                return np.ascontiguousarray(np.asarray(im.resize((ww, hh), Image.BILINEAR)).transpose(2, 0, 1))

            pil_crops = _crops_with(rgb, pil_resize)
        plan = host.layout(rgb.shape[2], rgb.shape[1])
        big_resized = plan["tiled"] and (plan["grid"][1] * host.TILE, plan["grid"][0] * host.TILE) != tuple(rgb.shape[1:])
        thumb_resized = tuple(plan["thumbnail"]) != tuple(rgb.shape[1:])
        with np.load(img["npz"]) as z:
            oracle_pixels = z["pixel_values"]
        crops = []
        for k, (a, b) in enumerate(zip(tv, mine)):
            h, w = a.shape[1:]
            resized = bool(big_resized if k < len(tv) - 1 else thumb_resized)
            n = (h // host.PATCH) * (w // host.PATCH)
            pv_equal = bool(np.array_equal(host.patchify(b), oracle_pixels[k][:n]))
            c = {"k": k, "size_hw": [h, w], "resized": resized, **_level_diff(a, b), "pixel_values_equal_oracle": pv_equal}
            for name, other in (("float64_filter", f64[k]), ("pil_bilinear", pil_crops[k])):
                d = _level_diff(other, a)
                c[name] = {kk: d[kk] for kk in ("bit_equal", "diff_elems", "max_level_diff") if kk in d}
                if resized:
                    controls[name]["resized_crops"] += 1
                    controls[name]["crops_differing"] += int(not d["bit_equal"])
                    controls[name]["diff_elems"] += d.get("diff_elems", 0)
                    controls[name]["max_level_diff"] = max(controls[name]["max_level_diff"], d.get("max_level_diff", 0))
            crops.append(c)
            totals["crops"] += 1
            totals["bit_equal"] += c["bit_equal"]
            totals["resized_crops"] += resized
            totals["diff_elems"] += c.get("diff_elems", 0)
            totals["max_level_diff"] = max(totals["max_level_diff"], c.get("max_level_diff", 0))
            totals["pixel_values_equal_oracle"] += pv_equal
        # round half up on the same float32 resize (counts the exact .5 values the rounding rule decides)
        half_up = []
        for size in ([(plan["grid"][1] * host.TILE, plan["grid"][0] * host.TILE)] if plan["tiled"] else []) + [plan["thumbnail"]]:
            hh, ww = size
            if (hh, ww) == tuple(rgb.shape[1:]):
                continue
            x = rgb.astype(np.float32)
            if ww != rgb.shape[2]:
                x = host._aa_pass_f32(x, host._aa_weights_f32(rgb.shape[2], ww), axis=2)
            if hh != rgb.shape[1]:
                x = host._aa_pass_f32(x, host._aa_weights_f32(rgb.shape[1], hh), axis=1)
            diff = int((np.floor(x + np.float32(0.5)) != np.rint(x)).sum())
            half_up.append({"size_hw": [hh, ww], "elems_rounded_differently": diff,
                            "exact_halves": int((x - np.floor(x) == np.float32(0.5)).sum())})
            controls["round_half_up"]["resized_crops"] += 1
            controls["round_half_up"]["crops_differing"] += int(diff > 0)
            controls["round_half_up"]["diff_elems"] += diff
        records.append({"id": img["id"], "public": img["public"], "file": str(img["file"].relative_to(WORK)),
                        "sha256": sha256_file(img["file"]), "format": fmt, "info_keys": info_keys,
                        "exif_orientation": exif_orientation, "px_wh": [int(rgb.shape[2]), int(rgb.shape[1])],
                        "layout": {"grid": list(plan["grid"]), "thumbnail_hw": list(plan["thumbnail"]), "tiled": plan["tiled"]},
                        "crops": crops, "round_half_up": half_up})
        print(f"{img['id']}: {len(crops)} crops, bit-equal {sum(c['bit_equal'] for c in crops)}, "
              f"max level diff {max(c.get('max_level_diff', 0) for c in crops)}, oracle pixel_values "
              f"{sum(c['pixel_values_equal_oracle'] for c in crops)}/{len(crops)}", flush=True)
    status = ("PASS" if totals["bit_equal"] == totals["crops"] else
              "PASS_WITHIN_1_LEVEL" if totals["max_level_diff"] <= 1 else "FAIL")
    controls_red = all(c["crops_differing"] > 0 for name, c in controls.items() if name != "round_half_up")
    doc = {"schema": "d1-omni-resize-numpy-gate/1", "written": now(), "status": status,
           "what": "host.crop_pixels_numpy (resize_uint8_antialias_numpy) vs host.image_crop_pixels (torchvision "
                   "resize, BILINEAR, antialias, on the uint8 tensor = the publisher's call) on every crop of the 16 "
                   "fixture images; the NumPy crops' patches vs the publisher's preprocess() pixel_values (oracle npz)",
           "torchvision_path": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                                "cpu_capability": torch.backends.cpu.get_cpu_capability(),
                                "native_uint8_bilinear": tvg._do_native_uint8_resize_on_cpu(tvg.InterpolationMode.BILINEAR),
                                "rule": "uint8 -> float32 -> F.interpolate(bilinear, align_corners=False, antialias=True) "
                                        "-> round_() (half to even) -> uint8"},
           "summary": totals | {"images": len(records), "numpy_resize_seconds": numpy_s},
           "controls": controls | {"red": controls_red,
                                   "note": "float64_filter and pil_bilinear must differ on the resized crops (the "
                                           "comparator can go red); round_half_up counts the float32 values at exactly "
                                           ".5, where the rounding rule decides the byte"},
           "images": records, "seconds": time.perf_counter() - t_start,
           "code_sha256": {f: sha256_file(HERE / f) for f in ("gate_swift.py", "host.py")}}
    write_json(out_path, doc)
    print(json.dumps({"status": status, **totals, "controls": {k: v for k, v in controls.items()}}, indent=1))
    print("->", out_path)
    return 0 if status != "FAIL" else 1


# =========================================================================== media (round 9)
MEDIA_REF = RESULTS / "swift_ref_media"


def _media_rows(index: dict, swift: dict, mode: str, py_arm: str) -> tuple[list, list, list, list]:
    """The Swift rows of one mode against the Python rows (arm) and the oracle: (metrics rows, Swift records, control
    records, per-row comparisons)."""
    from _metrics import row_record

    py = {(r["id"], r["qid"]): r for r in index["rows"] if r["mode"] == mode and r["arm"] == py_arm}
    rows, recs, ctl, cmp = [], [], [], []
    for s in swift["rows"]:
        if s["mode"] != mode:
            continue
        r = py[(s["id"], s["qid"])]
        m = metrics_row(r)
        rows.append(m)
        rec = row_record(m, s["logits"])
        rec.update(bucket=s["bucket"])
        recs.append(rec)
        c = row_record(m, s["control_logits"])
        c.update(donor=s["control_donor"])
        ctl.append(c)
        if [float(v) for v in host.probabilities_from_logits(s["logits"], host.as_question(r["question"]), r["calibrate"])] != s["probs"]:
            raise SystemExit(f"{s['id']}/{s['qid']}: the Swift p is not host.py's p of the Swift logits")
        cmp.append({"id": s["id"], "qid": s["qid"], "bucket": s["bucket"], "positions": s["positions"], "prefix": s["prefix"],
                    "python_bit_equal": s["logits_bits"] == r["logits_bits"] and s["probs_bits"] == r["probs_bits"],
                    "python_max_abs_dlogit": max(abs(a - b) for a, b in zip(s["logits"], r["logits"])),
                    "python_max_abs_dp": max(abs(a - b) for a, b in zip(s["probs"], r["probs"])),
                    "oracle_max_abs_dp": rec["max_abs_dp"], "drift": s["drift_marker_logits"]})
    return rows, recs, ctl, cmp


def _e2e(rows, recs, ctl) -> dict:
    from _metrics import summarize, wrong_pairing

    s = summarize(recs)
    c = summarize(ctl)
    w = wrong_pairing(rows, recs)
    keep = ("rows", "argmax_equal", "non_near_tie", "near_tie", "max_abs_dp", "max_abs_dp_row", "mean_row_max_abs_dp",
            "max_abs_dlogit", "ship_bar")
    by_bucket = {}
    for b in sorted({r["bucket"] for r in recs}):
        sb = summarize([r for r in recs if r["bucket"] == b])
        by_bucket[str(b)] = {k: sb[k] for k in ("rows", "argmax_equal", "max_abs_dp", "mean_row_max_abs_dp")}
    return {"swift": {k: s[k] for k in keep} | {"by_bucket": by_bucket},
            "control_next_item_prefix": {k: c[k] for k in ("rows", "argmax_equal", "max_abs_dp", "mean_row_max_abs_dp")}
            | {"status": c["ship_bar"]["status"], "must_be": "FAIL"},
            "wrong_pairing_oracle_swap": w}


def cmd_media(args) -> int:
    """Swift's media path (d1omni parity-media, AOT and JIT) against the Python host's dump and the oracle ->
    swift_media_image.json and swift_media_audio.json."""
    index = json.loads((MEDIA_REF / "index.json").read_text())
    runs = {k: json.loads((RESULTS / rname(f"swift_media_parity_{k}.json")).read_text()) for k in ("aot", "jit")
            if (RESULTS / rname(f"swift_media_parity_{k}.json")).exists()}
    if "aot" not in runs:
        raise SystemExit("results/swift_media_parity_aot.json is missing")
    resize = json.loads((RESULTS / "resize_numpy_gate.json").read_text())
    common = {"written": now(), "reference": {"dump": str(MEDIA_REF / "index.json"),
                                              "dump_sha256": sha256_file(MEDIA_REF / "index.json")},
              "swift_runs": {k: {"file": str(RESULTS / rname(f"swift_media_parity_{k}.json")),
                                 "binary_sha256": v["environment"]["binary_sha256"], "setup": v["setup"],
                                 "media_assets": v["media_assets"], "loads": v["loads"]} for k, v in runs.items()},
              "code_sha256": {f: sha256_file(HERE / f) for f in ("gate_swift.py", "host.py", "host_dump_media.py", "_metrics.py")}}
    a = runs["aot"]

    def jit_vs_aot(kind: str, items: str, key) -> dict:
        if "jit" not in runs:
            return {"status": "not run"}
        ja = {x["id"]: x for x in runs["jit"][items]}
        same = [key(x) == key(ja[x["id"]]) for x in a[items]]
        rows_a = {(r["id"], r["qid"]): r for r in a["rows"] if r["mode"] == kind}
        rows_j = [r for r in runs["jit"]["rows"] if r["mode"] == kind]
        req = [r["logits_bits"] == rows_a[(r["id"], r["qid"])]["logits_bits"] and r["probs_bits"] == rows_a[(r["id"], r["qid"])]["probs_bits"]
               for r in rows_j]
        return {"media_outputs_equal": f"{sum(same)}/{len(same)}", "rows_bit_equal": f"{sum(req)}/{len(req)}",
                "status": "PASS" if all(same) and all(req) else "DIFFERS"}

    # ---------------------------------------------------------------- image
    imgs = a["images"]
    crops = [c for i in imgs for c in i["crops"]]
    rgb_equal = [i["id"] for i in imgs if i["rgb"]["bit_equal"]]
    rgb_diff = {i["id"]: i["rgb"] for i in imgs if not i["rgb"]["bit_equal"]}
    crop_levels = max(c["crop_u8"].get("max_diff", 0) for c in crops)
    inputs = {name: {"bit_equal": sum(c[name]["bit_equal"] for c in crops), "crops": len(crops),
                     "max_abs": max(c[name].get("max_abs", c[name].get("max_diff", 0)) for c in crops)}
              for name in ("pixel_values", "pos_embed", "patch_mask", "unshuffle_index")}
    rows, recs, ctl, cmp = _media_rows(index, a, "image", "python")
    e2e = _e2e(rows, recs, ctl)
    png = [r for r, m in zip(recs, rows) if m["id"] in rgb_equal]
    from _metrics import summarize
    png_s = summarize(png)
    image = {
        "schema": "d1-omni-swift-media-image/1", **common,
        "decode": {"images": len(imgs), "rgb_bit_equal": f"{len(rgb_equal)}/{len(imgs)}", "bit_equal_ids": rgb_equal,
                   "differs": rgb_diff,
                   "rule": "ImageIO's decoded bytes (CGImage data provider), no EXIF rotation, no colour matching, alpha dropped"},
        "resize": {"numpy_vs_torchvision": {k: resize["summary"][k] for k in ("crops", "bit_equal", "max_level_diff")}
                   | {"status": resize["status"], "file": str(RESULTS / "resize_numpy_gate.json")},
                   "swift_vs_python_crops": {"crops": len(crops), "bit_equal": sum(c["crop_u8"]["bit_equal"] for c in crops),
                                             "max_level_diff": crop_levels,
                                             "bit_equal_on_bit_equal_rgb": f"{sum(c['crop_u8']['bit_equal'] for i in imgs if i['id'] in rgb_equal for c in i['crops'])}/"
                                                                           f"{sum(len(i['crops']) for i in imgs if i['id'] in rgb_equal)}"}},
        "graph_inputs": inputs,
        "prefix": {"bit_equal_images": f"{sum(i['prefix']['bit_equal'] for i in imgs)}/{len(imgs)}",
                   "max_abs": max(i["prefix"]["max_abs"] for i in imgs),
                   "max_abs_on_bit_equal_rgb": max(i["prefix"]["max_abs"] for i in imgs if i["id"] in rgb_equal),
                   "arm_b_python_inputs_through_swift_graph": f"{sum(c['arm_b_equal_python'] for c in crops)}/{len(crops)} crops bit-equal",
                   "drift_max": max(i["vision_drift"] for i in imgs)},
        "e2e": e2e | {"rows_bit_equal_python": f"{sum(c['python_bit_equal'] for c in cmp)}/{len(cmp)}",
                      "python_max_abs_dp": max(c["python_max_abs_dp"] for c in cmp),
                      "png_rows_only": {k: png_s[k] for k in ("rows", "argmax_equal", "max_abs_dp", "mean_row_max_abs_dp")}
                      | {"status": png_s["ship_bar"]["status"]}},
        "jit_vs_aot": jit_vs_aot("image", "images", lambda x: [c["graph_output_sha256"] for c in x["crops"]]),
        "rows": cmp, "images": [{"id": i["id"], "rgb": i["rgb"], "prefix": i["prefix"], "crops": len(i["crops"]),
                                  "times": i["times"]} for i in imgs]}
    image["status"] = ("PASS" if e2e["swift"]["ship_bar"]["status"] == "PASS" and e2e["control_next_item_prefix"]["status"] == "FAIL"
                       and e2e["wrong_pairing_oracle_swap"]["caught"] else "FAIL")
    # ---------------------------------------------------------------- audio
    clips = a["clips"]
    rows, recs, ctl, cmp = _media_rows(index, a, "audio", "python_numpy_mel")
    e2e_a = _e2e(rows, recs, ctl)
    mel = [c["mel_vs_numpy"] for c in clips]
    audio = {
        "schema": "d1-omni-swift-media-audio/1", **common,
        "samples": {"clips": len(clips), "bit_equal": f"{sum(c['samples']['bit_equal'] for c in clips)}/{len(clips)}",
                    "from": {c["id"]: c["samples_from"] for c in clips}},
        "mel": {"vs_mel_numpy": {"clips_bit_equal": f"{sum(m['bit_equal'] for m in mel)}/{len(mel)}",
                                 "elements_differing": sum(m["diff_elems"] for m in mel), "elements": sum(m["elems"] for m in mel),
                                 "max_abs": max(m["max_abs"] for m in mel), "max_ulp": max(m["max_ulp"] for m in mel)},
                "vs_mel_torch": {"max_abs": max(c["mel_vs_torch"]["max_abs"] for c in clips)},
                "rule": "float64 (mel_numpy) cast to float32; the FFT is vDSP's (NumPy: pocketfft), every other step NumPy's "
                        "arithmetic in NumPy's order (libm log / cos, Accelerate dgemm, pairwise sums)"},
        "masks": {name: f"{sum(c[name]['bit_equal'] for c in clips)}/{len(clips)}" for name in ("mask_f", "mask_f2", "mask_f4", "mask_t")},
        "prefix": {"bit_equal_numpy_mel_prefix": f"{sum(c['prefix_vs_numpy']['bit_equal'] for c in clips)}/{len(clips)}",
                   "max_abs_vs_numpy_mel_prefix": max(c["prefix_vs_numpy"]["max_abs"] for c in clips),
                   "max_abs_vs_torch_mel_prefix": max(c["prefix_vs_torch"]["max_abs"] for c in clips),
                   "arm_b_python_inputs_through_swift_graph": f"{sum(c['arm_b_equal_python'] for c in clips)}/{len(clips)} clips bit-equal",
                   "drift_max": max(c["audio_drift"] for c in clips)},
        "e2e": e2e_a | {"rows_bit_equal_python_numpy_mel": f"{sum(c['python_bit_equal'] for c in cmp)}/{len(cmp)}",
                        "python_max_abs_dp": max(c["python_max_abs_dp"] for c in cmp)},
        "jit_vs_aot": jit_vs_aot("audio", "clips", lambda x: x["graph_output_sha256"]),
        "rows": cmp, "clips": [{k: c[k] for k in ("id", "bucket_s", "frames", "prefix_rows", "samples", "mel_vs_numpy", "mel_vs_torch",
                                                  "prefix_vs_numpy", "prefix_vs_torch", "times")} for c in clips]}
    audio["status"] = ("PASS" if e2e_a["swift"]["ship_bar"]["status"] == "PASS" and e2e_a["control_next_item_prefix"]["status"] == "FAIL"
                       and e2e_a["wrong_pairing_oracle_swap"]["caught"] else "FAIL")
    for name, doc in ((rname("swift_media_image.json"), image), (rname("swift_media_audio.json"), audio)):
        path = RESULTS / name
        if path.exists() and not args.replace:
            raise SystemExit(f"{path} exists: pass --replace to write a new verdict over it")
        write_json(path, doc)
    for tag, doc in (("image", image), ("audio", audio)):
        s = doc["e2e"]["swift"]
        print(tag, doc["status"], f"rows {s['rows']} argmax {s['argmax_equal']}/{s['rows']} max|dp| {s['max_abs_dp']:.3e} "
              f"mean {s['mean_row_max_abs_dp']:.3e}", "control", doc["e2e"]["control_next_item_prefix"]["status"],
              "wrong pairing", doc["e2e"]["wrong_pairing_oracle_swap"]["status"], "jit vs aot", doc["jit_vs_aot"])
    print("image decode", image["decode"]["rgb_bit_equal"], "prefix", image["prefix"]["bit_equal_images"],
          "rows python bit-equal", image["e2e"]["rows_bit_equal_python"])
    print("audio mel", audio["mel"]["vs_mel_numpy"], "prefix", audio["prefix"]["bit_equal_numpy_mel_prefix"],
          "rows python bit-equal", audio["e2e"]["rows_bit_equal_python_numpy_mel"])
    return 0 if image["status"] == "PASS" and audio["status"] == "PASS" else 1


# =========================================================================== timing
TIMED = {"W1": "dec-fp16-L256", "W2": "dec-fp16-L256", "W3": "dec-fp16-L4096"}
RULE_R8 = RESULTS / "timing" / "ranking_rule_r8.md"
PY_CONTROL_FORMS = "dec-fp16-L256-gpu,dec-fp16-L4096-gpu"


def rule_r8() -> dict:
    st = RULE_R8.stat()
    return {"path": str(RULE_R8), "sha256": sha256_file(RULE_R8),
            "mtime": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(st.st_mtime))}


def scratch_of(pid: int) -> list[dict]:
    """The MPSGraph scratch folders a process left (mpsgraph-<pid>-<date>-...), with their bytes (recorded only)."""
    out = []
    for d in sorted(Path("/private/var/folders").glob(f"*/*/T/com.apple.MetalPerformanceShadersGraph/mpsgraph-{pid}-*")):
        out.append({"path": str(d), "bytes": sum(p.stat().st_size for p in d.rglob("*") if p.is_file())})
    return out


def run_logged(argv: list[str], log: Path) -> dict:
    t0 = time.monotonic()
    started = now()
    with open(log, "a") as fh:
        fh.write(f"[{started}] {' '.join(argv)}\n")
        fh.flush()
        p = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT)
        rc = p.wait()
    return {"argv": argv, "rc": rc, "pid": p.pid, "started": started, "finished": now(),
            "wall_s": time.monotonic() - t0, "log": str(log), "mpsgraph_scratch": scratch_of(p.pid)}


def stopped_pids() -> set[int]:
    """Processes in the stopped state (ps STAT T): another lane's job paused by its own window-yield wrapper while a
    window is open (round 8's first try waited on two of them)."""
    out = subprocess.run(["ps", "-Ao", "pid=,stat="], capture_output=True, text=True).stdout
    return {int(p) for p, s in (line.split() for line in out.splitlines() if len(line.split()) == 2) if s.startswith("T")}


def wait_gpu_quiet(tr, max_s: float) -> dict:
    """timing_run.wait_gpu_quiet without the stopped processes: a job its own wrapper paused for this window does
    not run on the GPU while it lasts (recorded apart)."""
    t0 = time.monotonic()
    seen, paused = [], []
    while True:
        stopped = stopped_pids()
        jobs = []
        for j in tr.gpu_jobs():
            (paused if int(j.split()[0]) in stopped else jobs).append(j)
        paused = sorted(set(paused))
        if not jobs:
            return {"waited_s": round(time.monotonic() - t0, 1), "jobs_seen": seen, "quiet": True, "stopped_jobs": paused}
        seen = sorted(set(seen) | set(jobs))
        if time.monotonic() - t0 > max_s:
            return {"waited_s": round(time.monotonic() - t0, 1), "jobs_seen": seen, "quiet": False, "left": jobs,
                    "stopped_jobs": paused}
        print(f"waiting for {len(jobs)} GPU job(s): {jobs[0][:140]}", flush=True)
        time.sleep(15)


def cmd_window(args) -> int:
    import timing_run as tr

    label = f"d1d-r8-swift-{args.run}"
    out_dir = RESULTS / "timing"
    swift_out = out_dir / f"run3_swift_{args.run}.json"
    py_out = out_dir / f"r8-{args.run}-py_main.json"
    win_out = out_dir / f"run3_swift_{args.run}_window.json"
    for p in (swift_out, py_out, win_out):
        if p.exists():
            raise SystemExit(f"{p} exists: never replaced")
    doc = {"schema": "d1-omni-r8-window/1", "run": args.run, "label": label, "cold": args.cold, "started": now(),
           "rule": rule_r8(), "lock_at_start": tr.lock_state(), "snapshots": {}}
    content = doc["lock_at_start"]["content"] or ""
    doc["lock_is_ours"] = content.startswith(f"{label} timing")
    if not doc["lock_is_ours"]:
        print(f"the lock holds {content!r}, not {label}: run this under quiet_hold.py", flush=True)
        return 2
    doc["snapshots"]["before"] = tr.snapshot()
    doc["gpu_quiet_wait"] = wait_gpu_quiet(tr, tr.GPU_JOB_WAIT_S)
    doc["contended"] = not doc["gpu_quiet_wait"]["quiet"]
    doc["snapshots"]["after_wait"] = tr.snapshot()
    argv = [str(BIN), "time", "--bundle-dir", str(MACOS), "--aot-root", str(COMPILED),
            "--fixtures", str(WORK / VERSIONS["v1"][0]), "--reference", str(RESULTS / "swift_ref" / "swift_ref.json"),
            "--out", str(swift_out), "--workloads", "W1,W2,W3", "--rounds", "35", "--warmup", "5", "--label", args.run]
    if args.cold:
        argv.append("--cold")
    doc["swift"] = run_logged(argv, WORK / "logs" / f"r8_{args.run}_swift_time.log")
    print(f"swift: rc {doc['swift']['rc']}, {doc['swift']['wall_s']:.0f} s", flush=True)
    doc["snapshots"]["after_swift"] = tr.snapshot()
    doc["lock_after_swift"] = tr.lock_state()
    if doc["swift"]["rc"] == 0:
        argv = [sys.executable, str(HERE / "timing_run.py"), "main", "--label", f"r8-{args.run}-py", "--out", str(py_out),
                "--trace", str(WORK / "logs" / f"r8_{args.run}_py_main.trace.jsonl"), "--workloads", "W1,W2,W3",
                "--forms", PY_CONTROL_FORMS]
        doc["python"] = run_logged(argv, WORK / "logs" / f"r8_{args.run}_py_main.log")
        print(f"python: rc {doc['python']['rc']}, {doc['python']['wall_s']:.0f} s", flush=True)
        doc["snapshots"]["after_python"] = tr.snapshot()
    doc["lock_at_end"] = tr.lock_state()
    doc["finished"] = now()
    write_json(win_out, doc)
    print("->", win_out, flush=True)
    return doc["swift"]["rc"] or doc.get("python", {}).get("rc", 0)


RULE_R9 = RESULTS / "timing" / "ranking_rule_r9.md"
PY_MEDIA_FORMS = "vis-fp16-gpu+dec-fp16-L256-gpu,aud-fp16-10s-gpu+dec-fp16-L256-gpu"


def rule_r9() -> dict:
    st = RULE_R9.stat()
    return {"path": str(RULE_R9), "sha256": sha256_file(RULE_R9),
            "mtime": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(st.st_mtime))}


def cmd_media_window(args) -> int:
    """The round-9 window (results/timing/ranking_rule_r9.md §5): `d1omni time --workloads W4,W5 --cold`, then the
    Python control on the same AOTs. Run it under quiet_hold.py d1d-r9-swift-media."""
    import timing_run as tr

    label = "d1d-r9-swift-media"
    out_dir = RESULTS / "timing"
    swift_out = out_dir / "run4_swift_media.json"
    py_out = out_dir / "r9-py-media_main.json"
    win_out = out_dir / "run4_swift_media_window.json"
    for p in (swift_out, py_out, win_out):
        if p.exists():
            raise SystemExit(f"{p} exists: never replaced")
    doc = {"schema": "d1-omni-r9-window/1", "label": label, "started": now(), "rule": rule_r9(),
           "lock_at_start": tr.lock_state(), "snapshots": {}}
    content = doc["lock_at_start"]["content"] or ""
    doc["lock_is_ours"] = content.startswith(f"{label} timing")
    if not doc["lock_is_ours"]:
        print(f"the lock holds {content!r}, not {label}: run this under quiet_hold.py", flush=True)
        return 2
    doc["snapshots"]["before"] = tr.snapshot()
    doc["gpu_quiet_wait"] = wait_gpu_quiet(tr, tr.GPU_JOB_WAIT_S)
    doc["contended"] = not doc["gpu_quiet_wait"]["quiet"]
    doc["snapshots"]["after_wait"] = tr.snapshot()
    argv = [str(BIN), "time", "--bundle-dir", str(MACOS), "--aot-root", str(COMPILED),
            "--fixtures", str(WORK / "fixtures" / "records.json"), "--media-reference", str(MEDIA_REF),
            "--media-root", str(WORK), "--out", str(swift_out), "--workloads", "W4,W5", "--rounds", "35", "--warmup", "5",
            "--label", "r9-media", "--cold"]
    doc["swift"] = run_logged(argv, WORK / "logs" / "r9_window_swift_time.log")
    print(f"swift: rc {doc['swift']['rc']}, {doc['swift']['wall_s']:.0f} s", flush=True)
    doc["snapshots"]["after_swift"] = tr.snapshot()
    doc["lock_after_swift"] = tr.lock_state()
    if doc["swift"]["rc"] == 0:
        argv = [sys.executable, str(HERE / "timing_run.py"), "main", "--label", "r9-py-media", "--out", str(py_out),
                "--trace", str(WORK / "logs" / "r9_py_media_main.trace.jsonl"), "--workloads", "W4,W5",
                "--forms", PY_MEDIA_FORMS]
        doc["python"] = run_logged(argv, WORK / "logs" / "r9_window_py_media.log")
        print(f"python: rc {doc['python']['rc']}, {doc['python']['wall_s']:.0f} s", flush=True)
        doc["snapshots"]["after_python"] = tr.snapshot()
    doc["lock_at_end"] = tr.lock_state()
    doc["finished"] = now()
    write_json(win_out, doc)
    print("->", win_out, flush=True)
    return doc["swift"]["rc"] or doc.get("python", {}).get("rc", 0)


def cmd_media_timing(args) -> int:
    """The round-9 window's numbers -> results/timing/ranking_r9.json (ranking_rule_r9.md §7)."""
    sw = json.loads((RESULTS / "timing" / "run4_swift_media.json").read_text())
    py = json.loads((RESULTS / "timing" / "r9-py-media_main.json").read_text())
    win = json.loads((RESULTS / "timing" / "run4_swift_media_window.json").read_text())
    rule = rule_r9()
    if win["rule"]["sha256"] != rule["sha256"] or not win.get("lock_is_ours"):
        raise SystemExit("the window ran under another rule, or the window was not ours")
    out = {"schema": "d1-omni-r9-ranking/1", "written": now(), "rule": rule,
           "window": {"label": win["label"], "contended": win["contended"], "gpu_quiet_wait": win["gpu_quiet_wait"],
                      "swift": sw["started"] + " .. " + sw["finished"], "python": py["started"] + " .. " + py["finished"],
                      "record": str(RESULTS / "timing" / "run4_swift_media_window.json")},
           "loads": {fid: [{"pass": x["pass"], "model_s": x["model_seconds"], "function_s": x["function_seconds"],
                            "wall_s": x["wall_seconds"]} for x in v["loads"]] for fid, v in sw["loads"].items()},
           "python_loads": {bid: b.get("load_s") for bid, b in py.get("bundles", {}).items()},
           "evicted_cache_entries": sw["evicted_cache_entries"], "workloads": {}}
    for w, wl in sw["workloads"].items():
        forms = {}
        for fid, f in wl["forms"].items():
            forms[fid] = {"median": f["stats_ms"]["median"], "p10": f["stats_ms"]["p10"], "p90": f["stats_ms"]["p90"],
                          "parts": f["parts_median_ms"], "check": wl["output_check"]["status"],
                          "jit_aot_logits_bit_equal": wl["output_check"]["jit_and_aot_logits_bit_equal_every_call"],
                          "jit_aot_prefix_bit_equal": wl["output_check"]["jit_and_aot_prefix_bit_equal_every_call"],
                          "max_abs_dp": wl["output_check"]["max_abs_dp_vs_oracle"]}
        pw = py["workloads"][w]
        for fid, f in pw["forms"].items():
            forms[fid + "-python"] = {"median": f["stats_ms"]["median"], "p10": f["stats_ms"]["p10"], "p90": f["stats_ms"]["p90"],
                                      "parts": f["parts_median_ms"], "check": f["output_check"]["status"],
                                      "max_abs_dp": f["output_check"]["max_abs_dp"]}
        best = min(v["median"] for v in forms.values() if v["check"] == "PASS")
        for v in forms.values():
            v["candidate"] = v["check"] == "PASS" and v["median"] <= best * 1.03
            v["vs_fastest"] = v["median"] / best - 1
        out["workloads"][w] = {"preprocess_ms": {k: s["median"] for k, s in wl["preprocess_ms"].items()},
                               "python_preprocess_ms": {k: s["median"] for k, s in pw.get("preprocess_ms", {}).items()},
                               "forms": dict(sorted(forms.items(), key=lambda kv: kv[1]["median"]))}
    path = write_json(RESULTS / "timing" / "ranking_r9.json", out)
    for w, x in out["workloads"].items():
        for fid, v in x["forms"].items():
            print(w, fid, f"{v['median']:.2f} ms (p10 {v['p10']:.2f}, p90 {v['p90']:.2f})", v["check"],
                  "candidate" if v["candidate"] else f"+{100 * v['vs_fastest']:.1f} %")
        print(w, "preprocess", {k: round(v, 3) for k, v in x["preprocess_ms"].items()})
    print("->", path)
    return 0


def cmd_timing(args) -> int:
    out = {"schema": "d1-omni-r8-ranking/1", "written": now(), "rule": rule_r8(), "windows": {}, "workloads": {}, "loads": {}}
    per = {}
    for run in ("run1", "run2"):
        sw = json.loads((RESULTS / "timing" / f"run3_swift_{run}.json").read_text())
        py = json.loads((RESULTS / "timing" / f"r8-{run}-py_main.json").read_text())
        win_path = RESULTS / "timing" / f"run3_swift_{run}_window.json"
        win = json.loads(win_path.read_text()) if win_path.exists() else {}
        if win and (win.get("rule", {}).get("sha256") != out["rule"]["sha256"] or not win.get("lock_is_ours")):
            raise SystemExit(f"{win_path}: another rule, or the window was not ours")
        out["windows"][run] = {"label": win.get("label"), "cold": sw.get("cold"), "contended": win.get("contended"),
                               "gpu_quiet_wait_s": win.get("gpu_quiet_wait", {}).get("waited_s"),
                               "swift": sw["started"] + " .. " + sw["finished"], "python": py["started"] + " .. " + py["finished"],
                               "window_record": str(win_path) if win else None}
        out["loads"][run] = {fid: [{"pass": x["pass"], "model_s": x["model_seconds"], "function_s": x["function_seconds"],
                                    "wall_s": x["wall_seconds"]} for x in v["loads"]] for fid, v in sw["loads"].items()}
        out["loads"][run]["python"] = {bid: b["load_s"] for bid, b in py["bundles"].items()}
        for w, wl in sw["workloads"].items():
            for fid, f in wl["forms"].items():
                per.setdefault(w, {}).setdefault(fid, {})[run] = {"median": f["stats_ms"]["median"], "p10": f["stats_ms"]["p10"],
                                                                 "p90": f["stats_ms"]["p90"], "graph_ms": f["parts_median_ms"]["graph_ms"],
                                                                 "check": wl["output_check"]["status"],
                                                                 "jit_aot_bit_equal": wl["output_check"]["jit_and_aot_logits_bit_equal_every_call"],
                                                                 "max_abs_dp": wl["output_check"]["max_abs_dp_vs_oracle"]}
        for w, wl in py["workloads"].items():
            for fid, f in wl["forms"].items():
                per.setdefault(w, {}).setdefault(fid + "-python", {})[run] = {
                    "median": f["stats_ms"]["median"], "p10": f["stats_ms"]["p10"], "p90": f["stats_ms"]["p90"],
                    "graph_ms": f["parts_median_ms"]["graph_ms"], "check": f["output_check"]["status"],
                    "max_abs_dp": f["output_check"]["max_abs_dp"], "ranked": TIMED[w] in fid}
    for w, forms in per.items():
        scored = {fid: dict(v, score=float(np.mean([v[r]["median"] for r in ("run1", "run2")])),
                            spread=abs(v["run1"]["median"] - v["run2"]["median"]) / min(v["run1"]["median"], v["run2"]["median"]),
                            ranked=v["run1"].get("ranked", True))
                  for fid, v in forms.items() if "run1" in v and "run2" in v}
        best = min(s["score"] for s in scored.values() if s["ranked"] and all(s[r]["check"] == "PASS" for r in ("run1", "run2")))
        for fid, s in scored.items():
            ok = s["ranked"] and all(s[r]["check"] == "PASS" for r in ("run1", "run2"))
            s["candidate"] = ok and s["score"] <= best * 1.03
            s["vs_fastest"] = s["score"] / best - 1
            s["flag_30pct"] = s["spread"] >= 0.30
        out["workloads"][w] = dict(sorted(scored.items(), key=lambda kv: kv[1]["score"]))
    path = write_json(RESULTS / "timing" / "ranking_r8.json", out)
    for w, forms in out["workloads"].items():
        for fid, s in forms.items():
            print(w, fid, f"{s['score']:.2f} ms", f"(run1 {s['run1']['median']:.2f}, run2 {s['run2']['median']:.2f}, spread {100 * s['spread']:.1f} %)",
                  "candidate" if s["candidate"] else ("not ranked" if not s["ranked"] else ""))
    print("->", path)
    return 0


# =========================================================================== ship summary (round 10)
def _same(new: list[dict], old: list[dict], key, fields: tuple) -> dict:
    """Items of two runs paired by key: how many agree on every field (bit-level fields: bits, sha256)."""
    o = {key(x): x for x in old}
    equal, missing, first = 0, 0, []
    for x in new:
        y = o.get(key(x))
        if y is None:
            missing += 1
            continue
        if all(x.get(f) == y.get(f) for f in fields):
            equal += 1
        elif len(first) < 10:
            first.append(str(key(x)))
    return {"items": len(new), "old_items": len(old), "equal": equal, "missing_in_old": missing,
            "fields": list(fields), "first_different": first, "status": "PASS" if equal == len(new) == len(old) else "DIFFERS"}


def _media_vs_python(swift: dict, index: dict, mode: str, arm: str) -> dict:
    py = {(r["id"], r["qid"]): r for r in index["rows"] if r["mode"] == mode and r["arm"] == arm}
    rows = [r for r in swift["rows"] if r["mode"] == mode]
    equal = [r["logits_bits"] == py[(r["id"], r["qid"])]["logits_bits"] and r["probs_bits"] == py[(r["id"], r["qid"])]["probs_bits"]
             for r in rows]
    return {"rows": len(rows), "python_rows": len(py), "bit_equal": sum(equal), "python_arm": arm}


def cmd_ship_summary(args) -> int:
    """Round 10: the Swift host on the stripped bundles in one verdict (results/ship_swift_parity.json): text 470 rows
    (ship_swift_parity_gate.json) + image 46 + audio 46 (ship_swift_media_{image,audio}.json and the raw parity-media
    outputs), AOT and JIT, each against the Python runtime on the same stripped AOT, the oracle's FACTS §7 bar and the
    controls; beside it, the stripped bundles against the unstripped ones (round 8 / 9): the Python references (every
    row's marker logits, p, graph inputs and the sha256 of all its scores; every media prefix and graph output) and the
    Swift outputs, bit for bit."""
    def load(name: str) -> dict:
        return json.loads((RESULTS / name).read_text())

    text = load("ship_swift_parity_gate.json")
    image, audio = load("ship_swift_media_image.json"), load("ship_swift_media_audio.json")
    index = load("ship_swift_ref_media/index.json")
    media_runs = {k: load(f"ship_swift_media_parity_{k}.json") for k in ("aot", "jit")}
    per_kind = {}
    for kind in ("aot", "jit"):
        t = text[kind]
        im = _media_vs_python(media_runs[kind], index, "image", "python")
        au = _media_vs_python(media_runs[kind], index, "audio", "python_numpy_mel")
        per_kind[kind] = {
            "text": {"rows": t["rows"], "text_rows": t["text_rows"], "media_rows_with_oracle_prefix": t["media_rows_with_oracle_prefix"],
                     "python_bit_equal": t["python_runtime"]["rows_bit_equal_logits_and_p"],
                     "python_max_abs_dlogit": t["python_runtime"]["max_abs_dlogit"], "facts_bar": t["ship_bar"]["status"],
                     "argmax_equal": t["argmax_equal"], "max_abs_dp": t["max_abs_dp"], "mean_row_max_abs_dp": t["mean_row_max_abs_dp"],
                     "wrong_pairing_control": t["wrong_pairing_control"]["status"], "status": t["status"]},
            "image": im, "audio": au,
            "rows": t["rows"] + im["rows"] + au["rows"],
            "python_bit_equal": t["python_runtime"]["rows_bit_equal_logits_and_p"] + im["bit_equal"] + au["bit_equal"]}
    media = {name: {"status": doc["status"], "rows": doc["e2e"]["swift"]["rows"],
                    "argmax_equal": doc["e2e"]["swift"]["argmax_equal"], "max_abs_dp": doc["e2e"]["swift"]["max_abs_dp"],
                    "mean_row_max_abs_dp": doc["e2e"]["swift"]["mean_row_max_abs_dp"],
                    "facts_bar": doc["e2e"]["swift"]["ship_bar"]["status"],
                    "control_next_item_prefix": doc["e2e"]["control_next_item_prefix"]["status"],
                    "wrong_pairing_oracle_swap": doc["e2e"]["wrong_pairing_oracle_swap"]["status"],
                    "jit_vs_aot": doc["jit_vs_aot"]}
             for name, doc in (("image", image), ("audio", audio))}
    media["image"]["decode_rgb_bit_equal"] = image["decode"]["rgb_bit_equal"]
    media["image"]["prefix_bit_equal_images"] = image["prefix"]["bit_equal_images"]
    media["audio"]["prefix_bit_equal_numpy_mel"] = audio["prefix"]["bit_equal_numpy_mel_prefix"]
    media["audio"]["mel_vs_numpy"] = audio["mel"]["vs_mel_numpy"]

    # the stripped bundles against the unstripped ones (the same ops: every number should be the same)
    ref_new, ref_old = load("ship_swift_ref/swift_ref.json"), load("swift_ref/swift_ref.json")
    row_key = lambda r: (r["id"], r["mode"], r["qid"])  # noqa: E731
    old_index = load("swift_ref_media/index.json")
    crop_items = lambda ix: [{"key": (i["id"], c["k"]), "out": c["files"]["graph_output_sha256"]}  # noqa: E731
                             for i in ix["images"] for c in i["crops"]]
    clip_items = lambda ix: [{"key": c["id"], **{k: c["files"][k] for k in ("graph_output_numpy_sha256",  # noqa: E731
                                                                              "graph_output_torch_sha256")}}
                             for c in ix["clips"]]
    swift_old = {k: load(f"swift_parity_{k}.json") for k in ("aot", "jit")}
    swift_new = {k: load(f"ship_swift_parity_{k}.json") for k in ("aot", "jit")}
    media_old = {k: load(f"swift_media_parity_{k}.json") for k in ("aot", "jit")}
    invariance = {
        "python_text_rows": _same(ref_new["rows"], ref_old["rows"], row_key,
                                  ("logits_bits", "probs_bits", "scores_sha256", "inputs_sha256", "ids_sha256")),
        "python_media_crops": _same(crop_items(index), crop_items(old_index), lambda x: x["key"], ("out",)),
        "python_media_clips": _same(clip_items(index), clip_items(old_index), lambda x: x["key"],
                                    ("graph_output_numpy_sha256", "graph_output_torch_sha256")),
        "python_media_rows": _same(index["rows"], old_index["rows"], lambda r: (r["id"], r["qid"], r["arm"]),
                                   ("logits_bits", "probs_bits", "inputs_sha256")),
        **{f"swift_text_{k}": _same(swift_new[k]["rows"], swift_old[k]["rows"], row_key, ("logits_bits", "probs_bits"))
           for k in ("aot", "jit")},
        **{f"swift_media_rows_{k}": _same(media_runs[k]["rows"], media_old[k]["rows"],
                                          lambda r: (r["mode"], r["id"], r["qid"]), ("logits_bits", "probs_bits"))
           for k in ("aot", "jit")},
    }
    total = {k: {"rows": v["rows"], "python_bit_equal": v["python_bit_equal"]} for k, v in per_kind.items()}
    ok = (all(v["rows"] == v["python_bit_equal"] == 562 for v in total.values())
          and all(per_kind[k]["text"]["status"] == "PASS" for k in per_kind)
          and all(m["status"] == "PASS" and m["rows"] == 46 for m in media.values())
          and all(m["jit_vs_aot"]["status"] == "PASS" for m in media.values())
          and text["jit_vs_aot"]["bit_equal"] == text["jit_vs_aot"]["rows"])
    doc = {"schema": "d1-omni-ship-swift-parity/1", "written": now(), "status": "PASS" if ok else "FAIL",
           "what": "the Swift host (apps/D1Omni d1omni parity / parity-media) on the stripped bundles (macos-ship/; AOT = "
                   "compiled/ship-h16c/, JIT = the .aimodel specialized by Swift) against the Python Core AI runtime on the "
                   "same stripped AOT (bit for bit), the oracle (FACTS §7) and the controls; 470 rows of ref/records_ref.json "
                   "(459 text + 11 media rows with the oracle's prefix) + 46 image + 46 audio rows (the Swift media path)",
           "rows_total": total, "per_kind": per_kind, "text_jit_vs_aot": text["jit_vs_aot"], "media": media,
           "stripped_vs_unstripped": invariance,
           "stripped_vs_unstripped_status": "PASS" if all(v["status"] == "PASS" for v in invariance.values()) else "DIFFERS",
           "inputs": {name: str(RESULTS / name) for name in (
               "ship_swift_parity_gate.json", "ship_swift_media_image.json", "ship_swift_media_audio.json",
               "ship_swift_parity_aot.json", "ship_swift_parity_jit.json", "ship_swift_media_parity_aot.json",
               "ship_swift_media_parity_jit.json", "ship_swift_ref/swift_ref.json", "ship_swift_ref_media/index.json")},
           "code_sha256": {f: sha256_file(HERE / f) for f in ("gate_swift.py",)}}
    path = RESULTS / "ship_swift_parity.json"
    if path.exists():
        raise SystemExit(f"{path} exists: never replaced")
    write_json(path, doc)
    print(doc["status"], json.dumps(total), "stripped vs unstripped", doc["stripped_vs_unstripped_status"],
          {k: f"{v['equal']}/{v['items']}" for k, v in invariance.items()}, "->", path)
    return 0 if ok and doc["stripped_vs_unstripped_status"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ship", action="store_true", help="pyref / parity / media on the stripped bundles (round 10: "
                        "macos-ship/, compiled/ship-h16c/; results named ship_*)")
    parser.add_argument("--small", action="store_true", help="pyref / parity on round 12's small decision buckets "
                        "(macos-ship-small/, their AOT in compiled/ship-h16c/; the rows of bucket 64 / 128 under "
                        "host.ALL_BUCKETS; results named small_*, the verdict results/swift_parity_L64_128.json)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("strings", "rows", "pyref", "parity", "timing", "resize", "media-window", "media-timing", "ship-summary"):
        sub.add_parser(name)
    m = sub.add_parser("media", help="the Swift media path (parity-media aot / jit) against the dump and the oracle")
    m.add_argument("--replace", action="store_true", help="write over an earlier verdict (the inputs are never replaced)")
    w = sub.add_parser("window", help="one measurement window (under quiet_hold.py d1d-r8-swift-<run>)")
    w.add_argument("--run", required=True, choices=["run1", "run2"])
    w.add_argument("--cold", action="store_true", help="d1omni time --cold (the first loads specialize in the window)")
    args = parser.parse_args()
    if args.ship:
        if args.cmd not in ("pyref", "parity", "media"):
            parser.error(f"--ship: {args.cmd} runs on the unstripped bundles only")
        global MACOS, COMPILED, MEDIA_REF, SHIP_MODE
        SHIP_MODE = True
        MACOS = WORK / "bundles" / "d1-omni-600m" / "macos-ship"
        COMPILED = WORK / "compiled" / "ship-h16c"
        MEDIA_REF = RESULTS / rname("swift_ref_media")
    if args.small:
        if args.ship or args.cmd not in ("pyref", "parity"):
            parser.error("--small: pyref and parity only (not with --ship)")
        global SMALL_MODE, PYREF_BUCKETS
        SMALL_MODE = True
        PYREF_BUCKETS = host.ALL_BUCKETS
        MACOS = WORK / "bundles" / "d1-omni-600m" / "macos-ship-small"
        COMPILED = WORK / "compiled" / "ship-h16c"
    return {"strings": cmd_strings, "rows": cmd_rows, "pyref": cmd_pyref, "parity": cmd_parity, "window": cmd_window,
            "timing": cmd_timing, "resize": cmd_resize, "media": cmd_media, "media-window": cmd_media_window,
            "media-timing": cmd_media_timing, "ship-summary": cmd_ship_summary}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
