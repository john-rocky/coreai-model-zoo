#!/usr/bin/env python3
"""Fixture records for the d1-3B oracle and gates: System One requests copied from the Kev fixture or written here.

Every record is `{id, source, request: {state, questions: {qid: {type, instructions, criteria}}}, gold: {qid: key or
null}, note, provenance}`. `request` is the arguments of d1's `model.system_one(state, questions)` (the Decision Index
schema the provider's `api.as_question` reads; questions keep their request order). `gold` keys follow the Kev fixture:
a criteria name for choice, "true" / "false" for noul, the level index as a string for score; d1 reports a noul's
probabilities as [yes, no], so `host.gold_key` maps "true" -> "yes" and "false" -> "no". Four sources:

* Kev's round-1 fixture (the LiteRT lane's `kev_work/fixtures/requests.json`, sha256 dfe55fb1…, 377 records), read from
  the lane's copy `$ZOO_WORK_ROOT/_d1_3b/fixtures/src/requests.json`: 356 records copied as they are (id, source,
  request, gold, note, provenance; nothing in them changes). Left out: the 20 `tv4x_tweet_offensive_*` records
  (licence unknown on the Hub card, real public figures' names and profanity: Kev fixture README) and `red_arm_000`
  (a gate arm, not a fixture: its request is the first arm of red_arms.json). The 356 are tv4 60 (MMLU), tv4x 120
  (emotion, qnli, paws, sciq, legacy_holdout, composition_holdout, 20 each), tv4s 20, semif 144, own 12. One question
  in them has no `instructions` (`own_email_03` / `next_step`): the provider's `as_question` raises KeyError on it, so
  that request is refused by d1's API as it stands (`summary.questions_refused`); the record stays as copied.
* `card_refund`, `card_ticket_00`, `card_ticket_01`: the three requests of the model card's example (README.md of
  LiquidAI/d1-3B @ da1fe36a, the text as printed there): the refund / team / urgency questions over "I was charged
  twice this month, please refund one of them.", and the team question over each of the two tickets of its
  `system_one_batch` line. The card prints no answers: gold null.
* `long_15k`, `long_34k`: two states written here as JSON (a cold-room monitoring log, an order ledger's audit log)
  at about 1,500 and 3,400 d1 tokens of state block (the card's 3.4k-token speed column), each with 4 questions mixing
  noul / choice / score and the answers they were written to have. No person, organisation, product or place name:
  codes, dates, amounts and role words only ("unit-07", "order 00409", "the supervisor"). The length is measured with
  the checkpoint's tokenizer on `host.state_block(state)` and held within 5 % of the target.

    python make_fixtures.py        # -> $ZOO_WORK_ROOT/_d1_3b/fixtures/{records.json, red_arms.json, LICENSE-SemIf-MIT.txt}

red_arms.json defines the pairs the round-2+ gates run (definitions only): Kev's word arm on tv4_000 ("correctly" ->
"incorrectly"), a grammatical "not" in one qnli and one paws noul, and two state swaps (tv4_000 <-> tv4_001, the
first <-> the third SemIf record, which ask the same claim of opposite evidence).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import host  # noqa: E402
from _paths import hf_snapshot, work_path  # noqa: E402

LANE = work_path("_d1_3b")
MODEL = {"hf_id": "LiquidAI/d1-3B", "revision": "da1fe36a861f24690f27f622dca1d8688503d113",
         "license": "LFM Open License v1.0 (license: other, license_name: lfm1.0)"}
README_SHA256 = "cf063f71ce85"   # prefix; the full value is checked against the snapshot's README.md below
KEV = {
    "local_copy": str(LANE / "fixtures" / "src" / "requests.json"),
    "original": "litertlm-convert/kev_work/fixtures/requests.json (the LiteRT lane's Kev-0.8B round-1 fixture)",
    "sha256": "dfe55fb145df7a3967ed7213ae24d67b5d4ec42da0315fdb409d5e4b702e3a48",
    "records": 377,
    "readme_sha256": "ec8164faa8e55d3d9e2a966d25272ce3e8b8b8da0830d80af6302adf011a76c0",
}
SEMIF_LICENSE = {"local_copy": str(LANE / "fixtures" / "src" / "LICENSE-SemIf-MIT.txt"),
                 "sha256": "f765f2140f8507a8f0d81ec0fd2c4bd72fe6a066841ef27883ff876a76bf61be",
                 "notice": "Copyright (c) 2026 TheoLeeCJ", "upstream": "github.com/TheoLeeCJ/SemIf @ ca3ba65f142967030ecb453346e94d6f476a69df",
                 "file": "benchmarks/data/authored144.jsonl (sha256 8162d1c73f925af64453f1ec05ef36d583b3815bf698e60f0d454bd11537e079)"}
EXCLUDED_PREFIXES = {"tv4x_tweet_offensive_": "licence unknown on the Hub card (cardiffnlp/tweet_eval); social-media posts quoted "
                                              "verbatim with real public figures' names and profanity (Kev fixture README)"}
EXCLUDED_IDS = {"red_arm_000": "a gate arm, not a fixture (tv4_000 with one word changed); its request is red_arms.json's "
                               "first arm"}
KEV_SOURCE_LICENSES = {   # Kev fixture README, "Upstream licences of the transfer-v4 sources" (Hub cards at the suite's pins)
    "mmlu": {"dataset": "cais/mmlu @ c30699e8", "licence": "mit"},
    "emotion": {"dataset": "dair-ai/emotion @ cab853a1", "licence": "other"},
    "tweet_offensive": {"dataset": "cardiffnlp/tweet_eval @ b3a375ba", "licence": "unknown", "d1": "excluded"},
    "qnli": {"dataset": "nyu-mll/glue @ bcdcba79", "licence": "other"},
    "paws": {"dataset": "google-research-datasets/paws @ 161ece95", "licence": "other"},
    "sciq": {"dataset": "allenai/sciq @ 2c94ad3e", "licence": "cc-by-nc-3.0"},
    "legacy_holdout": {"dataset": None, "licence": "generated by the Kev suite code (the kev repository's Apache-2.0)"},
    "composition_holdout": {"dataset": None, "licence": "generated by the Kev suite code (the kev repository's Apache-2.0)"},
}
PUBLICATION = ("measurement use only for sciq (CC BY-NC 3.0) and for emotion / qnli / paws (licence 'other'): do not put their "
               "text in a published fixture (publish ids, hashes and numbers); tweet_offensive is not in this fixture. The "
               "SemIf records carry the MIT notice (LICENSE-SemIf-MIT.txt); the own, card and own_long records carry none "
               "of these restrictions (the card's text is LiquidAI's README, LFM Open License v1.0).")
LONG_TARGETS = {"long_15k": 1500, "long_34k": 3400}
LONG_TOLERANCE = 0.05


# --------------------------------------------------------------------------- helpers
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def record(rid, source, state, questions, gold, note, provenance):
    return {"id": rid, "source": source, "request": {"state": state, "questions": questions}, "gold": gold,
            "note": note, "provenance": provenance}


# --------------------------------------------------------------------------- Kev records
def kev_records() -> tuple[list[dict], dict, list[dict]]:
    raw = Path(KEV["local_copy"]).read_bytes()
    if sha256_bytes(raw) != KEV["sha256"]:
        raise SystemExit(f"{KEV['local_copy']}: sha256 differs from the Kev fixture's {KEV['sha256'][:12]}")
    doc = json.loads(raw)
    recs = doc["records"]
    assert len(recs) == KEV["records"], len(recs)
    keep, dropped = [], []
    for r in recs:
        why = EXCLUDED_IDS.get(r["id"]) or next((v for p, v in EXCLUDED_PREFIXES.items() if r["id"].startswith(p)), None)
        if why:
            dropped.append(r)
            continue
        keep.append(r)
    tweets = [r for r in dropped if r["id"].startswith("tv4x_tweet_offensive_")]
    assert len(tweets) == 20 and all(r["provenance"]["_meta"]["source"] == "tweet_offensive" for r in tweets)
    assert [r["id"] for r in dropped if r["id"] in EXCLUDED_IDS] == ["red_arm_000"]
    assert len(keep) == 356, len(keep)
    assert not any(r["provenance"].get("_meta", {}).get("source") == "tweet_offensive" for r in keep)
    by_id = {r["id"]: r for r in recs}
    return keep, by_id, dropped


# --------------------------------------------------------------------------- card records
CARD_QUESTIONS = {   # README.md "How to use", as printed
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                          "fraud": "Suspected unauthorised use"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["Can wait", "Today", "Blocking the customer now"]},
}
CARD_STATE = "I was charged twice this month, please refund one of them."
CARD_TICKETS = ["Where is my parcel? It was due Monday.", "The app crashes when I open settings."]


def card_records(readme: str) -> list[dict]:
    for text in (CARD_STATE, *CARD_TICKETS, *(q["instructions"] for q in CARD_QUESTIONS.values()),
                 *CARD_QUESTIONS["team"]["criteria"].values(), *CARD_QUESTIONS["urgency"]["criteria"]):
        assert text in readme, f"not in the card: {text!r}"
    prov = {"file": "README.md", "repo": MODEL["hf_id"], "revision": MODEL["revision"], "section": "How to use",
            "readme_sha256": sha256_bytes(readme.encode())}
    out = [record("card_refund", "card", CARD_STATE, copy.deepcopy(CARD_QUESTIONS),
                  {q: None for q in CARD_QUESTIONS}, "model card example: model.system_one(state, questions)", prov)]
    for k, t in enumerate(CARD_TICKETS):
        out.append(record(f"card_ticket_{k:02d}", "card", t, {"team": copy.deepcopy(CARD_QUESTIONS["team"])},
                          {"team": None}, f"model card example: system_one_batch ticket {k}", prov))
    return out


# --------------------------------------------------------------------------- long states (written here)
def cold_room_log(n: int) -> dict:
    """A cold-room monitoring log: 3 units read every 15 minutes from 06:00. unit-07 is the only unit over a limit: its
    temperature reads 9.3 and 8.9 at 06:45 and 07:00 (limit 8.0, back to 6.2 at 07:15) with its door open at 06:45."""
    units = ["unit-03", "unit-05", "unit-07"]
    base = {"unit-03": 3.8, "unit-05": 4.4, "unit-07": 5.1}
    readings = []
    for k in range(n):
        minute = 15 * k
        t = f"2026-03-02 {6 + minute // 60:02d}:{minute % 60:02d}"
        for j, u in enumerate(units):
            temp = round(base[u] + 0.1 * ((k * 3 + j * 5) % 7) - 0.2, 1)
            hum = 58 + (k * 7 + j * 11) % 12
            door = "closed"
            if u == "unit-07" and t.endswith("06:30"):
                temp = 7.6
            if u == "unit-07" and t.endswith("06:45"):
                temp, door = 9.3, "open"
            if u == "unit-07" and t.endswith("07:00"):
                temp = 8.9
            if u == "unit-07" and t.endswith("07:15"):
                temp = 6.2
            readings.append({"time": t, "unit": u, "temp_c": temp, "humidity_pct": hum, "door": door})
    return {"log": "cold-room monitoring, early shift", "site_code": "CR-2",
            "limits": {"temp_c_max": 8.0, "humidity_pct_max": 75}, "units": units,
            "readings": readings,
            "notes": ["readings every 15 minutes per unit", "the supervisor reviews any excursion at the end of the shift"]}


def cold_room_questions() -> tuple[dict, dict]:
    q = {
        "temp_excursion": {"type": "noul", "instructions": "Did any reading go above the temperature limit?"},
        "first_unit": {"type": "choice", "instructions": "Which unit went over a limit first?",
                       "criteria": {"u03": "unit-03", "u05": "unit-05", "u07": "unit-07", "none": "No unit went over a limit"}},
        "severity": {"type": "score", "instructions": "How long did the worst temperature excursion last?",
                     "criteria": ["No excursion", "One reading only, back within limits by the next reading",
                                  "Two or more readings in a row above the limit"]},
        "door_open": {"type": "noul", "instructions": "Was a door open on a unit while its temperature was above the limit?",
                      "criteria": {"true": "A door reads open at a reading above the limit",
                                   "false": "Every reading above the limit has its door closed"}},
    }
    return q, {"temp_excursion": "true", "first_unit": "u07", "severity": "2", "door_open": "true"}


def order_ledger(n: int) -> list:
    """An order ledger's audit log: routine orders 00400.. are created, paid and shipped. Designed exceptions: order 00409
    is refunded twice (30.00 and 45.00, approved by the supervisor), order 00406 has one refund of 640.00 approved by the
    clerk, orders 00403 and 00414 are cancelled, order 00411 is shipped before it is paid."""
    entries, seq = [], 0
    minute = 0

    def add(action, order, amount, actor, **extra):
        nonlocal seq, minute
        seq += 1
        minute += 3
        e = {"seq": seq, "time": f"2026-04-14 {9 + minute // 60:02d}:{minute % 60:02d}", "actor": actor,
             "action": action, "order": f"order {order:05d}", "amount": amount}
        e.update(extra)
        entries.append(e)

    for k in range(n):
        o = 400 + k
        amt = round(40 + (k * 37) % 400 + 0.25 * (k % 4), 2)
        add("create", o, amt, "the clerk")
        if o in (403, 414):
            add("cancel", o, amt, "the clerk", reason="customer request")
            continue
        if o == 411:
            add("ship", o, amt, "the dispatcher")
            add("pay", o, amt, "the cashier")
            continue
        add("pay", o, amt, "the cashier")
        add("ship", o, amt, "the dispatcher")
        if o == 409:
            add("refund", o, 30.0, "the clerk", approved_by="the supervisor")
            add("refund", o, 45.0, "the clerk", approved_by="the supervisor")
        if o == 406:
            add("refund", o, 640.0, "the clerk", approved_by="the clerk")
    return entries


def order_ledger_questions() -> tuple[dict, dict]:
    q = {
        "refunded_twice": {"type": "choice", "instructions": "Which order was refunded twice?",
                           "criteria": {"o00402": "order 00402", "o00406": "order 00406", "o00409": "order 00409",
                                        "o00411": "order 00411"}},
        "big_refunds_approved": {"type": "noul",
                                 "instructions": "Did the supervisor approve every refund of more than 500.00?"},
        "cancellations": {"type": "score", "instructions": "How many orders were cancelled?",
                          "criteria": ["None", "One", "Two", "Three or more"]},
        "shipped_unpaid": {"type": "noul", "instructions": "Was any order shipped before it was paid?"},
    }
    return q, {"refunded_twice": "o00409", "big_refunds_approved": "false", "cancellations": "2", "shipped_unpaid": "true"}


def state_tokens(tok, state) -> int:
    return len(host.token_ids(tok, host.state_block(state)))


def fit_length(make, target: int, tok, lo: int, hi: int) -> tuple[int, object, int]:
    best = None
    for n in range(lo, hi + 1):
        st = make(n)
        t = state_tokens(tok, st)
        if best is None or abs(t - target) < abs(best[2] - target):
            best = (n, st, t)
    n, st, t = best
    if abs(t - target) > LONG_TOLERANCE * target:
        raise SystemExit(f"no length within {LONG_TOLERANCE:.0%} of {target}: best n={n} -> {t} tokens")
    return best


def long_records(tok) -> tuple[list[dict], dict]:
    out, fit = [], {}
    n, st, t = fit_length(cold_room_log, LONG_TARGETS["long_15k"], tok, 4, 40)
    q, gold = cold_room_questions()
    assert n * 15 >= 90, "the log must reach 07:15 (the excursion and its end)"
    out.append(record("long_15k", "own_long", st, q, gold,
                      f"cold-room log, {n} reading rounds of 3 units: unit-07 above 8.0 at 06:45 and 07:00, door open at 06:45",
                      {"written_for": "d1-3B Core AI port (round 1)", "generator": "make_fixtures.cold_room_log", "n": n}))
    fit["long_15k"] = {"n": n, "state_tokens": t, "target": LONG_TARGETS["long_15k"]}
    n, st, t = fit_length(order_ledger, LONG_TARGETS["long_34k"], tok, 15, 60)
    q, gold = order_ledger_questions()
    assert n >= 15, "the ledger must reach order 00414"
    out.append(record("long_34k", "own_long", st, q, gold,
                      f"order ledger audit log, orders 00400-{399 + n:05d}: 00409 refunded twice, 00406 refund 640.00 approved by "
                      "the clerk, 00403 and 00414 cancelled, 00411 shipped before payment",
                      {"written_for": "d1-3B Core AI port (round 1)", "generator": "make_fixtures.order_ledger", "n": n}))
    fit["long_34k"] = {"n": n, "state_tokens": t, "target": LONG_TARGETS["long_34k"]}
    return out, fit


# --------------------------------------------------------------------------- red arms
def red_arms(by_id: dict, records: list[dict]) -> dict:
    arms = []
    r0 = by_id["red_arm_000"]
    base = by_id["tv4_000"]
    a, b = base["request"]["questions"]["answer"]["instructions"], r0["request"]["questions"]["answer"]["instructions"]
    assert a.replace("correctly", "incorrectly", 1) == b and r0["request"]["state"] == base["request"]["state"]
    arms.append({"id": "red_word_tv4_000", "kind": "word", "base": "tv4_000", "question": "answer",
                 "change": {"field": "questions.answer.instructions", "from": a, "to": b},
                 "request": r0["request"], "source": "Kev fixture red_arm_000, request as it is"})
    for rid, qid, old, new in (("tv4x_qnli_00", "answers", "Does the sentence contain", "Does the sentence not contain"),
                               ("tv4x_paws_00", "paraphrase", "Does this sentence mean", "Does this sentence not mean")):
        rec = by_id[rid]
        req = copy.deepcopy(rec["request"])
        instr = req["questions"][qid]["instructions"]
        assert instr.count(old) == 1, (rid, instr)
        req["questions"][qid]["instructions"] = instr.replace(old, new)
        arms.append({"id": f"red_not_{rid.removeprefix('tv4x_')}", "kind": "not", "base": rid, "question": qid,
                     "change": {"field": f"questions.{qid}.instructions", "from": instr,
                                "to": req["questions"][qid]["instructions"]},
                     "request": req, "base_gold": rec["gold"][qid],
                     "note": ("a grammatical 'not' in the question; the criteria (paws: Yes = same meaning) are left as "
                              "they are" if rid.startswith("tv4x_paws") else "a grammatical 'not' in the question")})
    semif = [r["id"] for r in records if r["source"] == "semif"]
    for x, y in (("tv4_000", "tv4_001"), (semif[0], semif[2])):
        rx, ry = by_id[x], by_id[y]
        assert list(rx["request"]["questions"]) == list(ry["request"]["questions"])
        arms.append({"id": f"state_swap_{x}__{y}", "kind": "state_swap", "pair": [x, y],
                     "requests": {f"{x}_with_state_of_{y}": {"state": ry["request"]["state"], "questions": rx["request"]["questions"]},
                                  f"{y}_with_state_of_{x}": {"state": rx["request"]["state"], "questions": ry["request"]["questions"]}},
                     "note": ("SemIf: the same claim over evidence that establishes it / says it is not done"
                              if x.startswith("semif") else "MMLU: each question's options over the other record's question")})
    return {"schema": "d1-red-arms/1",
            "what": "the perturbed requests the round-2+ gates run beside the fixture (definitions only in round 1)",
            "rule": ("an arm is red when its probabilities move by more than 0.02 on some option against its base "
                     "record's (the same question names) — on the oracle and on the graph; a graph that ignored the "
                     "changed input would not move"),
            "arms": arms}


# --------------------------------------------------------------------------- checks and summary
def own_text_checks(records: list[dict]) -> dict:
    """The records written here (card excepted: LiquidAI's text) carry no URL, e-mail, or the name the plan keeps out;
    their capitalised words are listed for review (sentence starts and option codes only)."""
    out = {}
    for r in records:
        if r["source"] != "own_long":
            continue
        text = json.dumps(r["request"], ensure_ascii=False)
        for bad in ("http", "www.", "@", ".com", "Jev"):
            assert bad not in text, (r["id"], bad)
        caps = sorted(set(re.findall(r"\b[A-Z][a-zA-Z]+\b", text)))
        out[r["id"]] = caps
    return out


def summarize(records: list[dict], refused: list[dict]) -> dict:
    by_source: dict = {}
    for r in records:
        s = by_source.setdefault(r["source"], {"records": 0, "questions": 0, "types": Counter(), "options": Counter(),
                                               "gold": 0})
        s["records"] += 1
        for qid, q in r["request"]["questions"].items():
            s["questions"] += 1
            s["types"][q.get("type", "choice")] += 1
            n = 2 if q.get("type") == "noul" else len(q.get("criteria") or [])
            s["options"][n] += 1
            s["gold"] += (r["gold"] or {}).get(qid) is not None
    for s in by_source.values():
        s["types"] = dict(sorted(s["types"].items()))
        s["options"] = {str(k): v for k, v in sorted(s["options"].items())}
    total_q = sum(s["questions"] for s in by_source.values())
    types = Counter(q.get("type", "choice") for r in records for q in r["request"]["questions"].values())
    return {"records": len(records), "questions": total_q, "by_source": by_source, "types": dict(sorted(types.items())),
            "options": dict(sorted(Counter(str(2 if q.get("type") == "noul" else len(q.get("criteria") or []))
                                           for r in records for q in r["request"]["questions"].values()).items(),
                                   key=lambda kv: int(kv[0]))),
            "multi_question_records": sum(len(r["request"]["questions"]) > 1 for r in records),
            "state_types": dict(Counter(type(r["request"]["state"]).__name__ for r in records)),
            "questions_refused": refused, "questions_answerable": total_q - len(refused)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(LANE / "fixtures"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    snap = Path(hf_snapshot(MODEL["hf_id"], revision=MODEL["revision"]))
    readme = (snap / "README.md").read_text()
    assert sha256_bytes(readme.encode()).startswith(README_SHA256)
    tok = host.load_tokenizer(snap / "tokenizer.json")

    kev, by_id, dropped = kev_records()
    card = card_records(readme)
    longs, fit = long_records(tok)
    records = kev + card + longs
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)) == 361, len(ids)
    refused = []
    for r in records:
        for qid, q in r["request"]["questions"].items():
            try:
                host.validate_question(qid, q)
            except ValueError as e:
                refused.append({"record": r["id"], "question": qid, "host": str(e),
                                "provider": "api.as_question raises KeyError('instructions') (prompt.Choice needs it)"
                                if "instructions" not in q else "see test_host.json"})
            g = (r["gold"] or {}).get(qid)
            if g is not None:
                keys = (["true", "false"] if q.get("type") == "noul" else list(q["criteria"]) if q.get("type", "choice") == "choice"
                        else [str(i) for i in range(len(q["criteria"]))])
                assert g in keys, (r["id"], qid, g)
    caps = own_text_checks(records)

    lic = Path(SEMIF_LICENSE["local_copy"]).read_bytes()
    assert sha256_bytes(lic) == SEMIF_LICENSE["sha256"] and SEMIF_LICENSE["notice"].encode() in lic
    write_atomic(out / "LICENSE-SemIf-MIT.txt", lic)
    kev_readme = Path(KEV["local_copy"]).with_name("README.md").read_bytes()
    assert sha256_bytes(kev_readme) == KEV["readme_sha256"]

    doc = {
        "schema": "d1-fixtures/1",
        "model": MODEL,
        "record_format": ("{id, source, request: {state, questions: {qid: {type, instructions, criteria}}}, gold: {qid: key or "
                          "null}, note, provenance}; request = the arguments of d1's model.system_one(state, questions)"),
        "gold_keys": ("the Kev fixture's: choice = criteria name, noul = 'true' / 'false', score = level index as a string; "
                      "d1's noul probabilities are [yes, no]: host.gold_key maps 'true' -> 'yes', 'false' -> 'no'"),
        "records": records,
        "summary": {**summarize(records, refused), "long_states": fit, "own_long_capitalised_words": caps},
        "sources": {
            "kev_fixture": {**{k: v for k, v in KEV.items() if k != "local_copy"},
                            "copied": len(kev), "copied_as_is": "id, source, request, gold, note, provenance unchanged",
                            "excluded": {**{f"{p}*": v for p, v in EXCLUDED_PREFIXES.items()}, **EXCLUDED_IDS},
                            "excluded_ids": [r["id"] for r in dropped],
                            "source_licenses": KEV_SOURCE_LICENSES},
            "semif_authored144": {**{k: v for k, v in SEMIF_LICENSE.items() if k != "local_copy"},
                                  "license": "MIT (LICENSE-SemIf-MIT.txt beside this file, byte-identical to the Kev fixture's)"},
            "card": {"repo": MODEL["hf_id"], "revision": MODEL["revision"], "file": "README.md",
                     "readme_sha256": sha256_bytes(readme.encode()), "license": MODEL["license"],
                     "records": [r["id"] for r in card]},
            "own_long": {"license": "written for this port (same licence as the conversion code)",
                         "names": "no person, organisation, product or place name: codes, dates, amounts, role words",
                         "records": [r["id"] for r in longs]},
            "publication": PUBLICATION,
        },
    }
    data = (json.dumps(doc, ensure_ascii=False, indent=1) + "\n").encode()
    write_atomic(out / "records.json", data)
    arms = red_arms(by_id, records)
    arms_data = (json.dumps(arms, ensure_ascii=False, indent=1) + "\n").encode()
    write_atomic(out / "red_arms.json", arms_data)
    s = doc["summary"]
    print(f"{out / 'records.json'}: {s['records']} records, {s['questions']} questions ({s['questions_answerable']} answerable), "
          f"sha256 {sha256_bytes(data)}")
    for src, v in s["by_source"].items():
        print(f"  {src}: {v['records']} records, {v['questions']} questions, {v['types']}, options {v['options']}")
    print("  refused:", refused)
    print("  long states:", fit)
    print("  own_long capitalised words:", caps)
    print(f"{out / 'red_arms.json'}: {len(arms['arms'])} arms, sha256 {sha256_bytes(arms_data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
