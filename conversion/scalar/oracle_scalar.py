#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.9.0", "transformers==5.17.0", "peft==0.21.0", "safetensors==0.8.0", "huggingface-hub==1.32.0", "numpy==2.3.5", "tokenizers==0.23.2", "accelerate==1.15.0"]
# ///
"""Build coreai-scalar-fixtures/1 for pngwn/system-one-qwen3.5-4b-scorer.

Model licence: cc-by-nc-4.0. Twenty embedded synthetic requests yield 48
questions and 280 option rows. Import the pinned author's encode, pad_batch,
score_options and softmax unchanged. Load the base through author.load_model,
attach PEFT, then merge_and_unload in CPU fp32 memory; do not reload the BF16
storage snapshot. Optional --compare checks the original unmerged PEFT oracle.
The source revisions are pinned below; inputs are local verified snapshots.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
import traceback

LICENSE = "cc-by-nc-4.0"
ADAPTER_REPO = "pngwn/system-one-qwen3.5-4b-scorer"
ADAPTER_REV = "e6464dce15f013c2ef641593a85cc6afcdaea928"
BASE_REPO = "Qwen/Qwen3.5-4B-Base"
BASE_REV = "1001bb4d826a52d1f399e183466143f4da7b741b"
TEMPERATURE = 1.75
MAX_LEN = 384

AUTHOR_SHA256 = 'cd9865bc82e1b49972955986e74856f10f0a66d4b9d057114958d1b4625906ff'

def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_requests():
    requests = []

    def add(request_id, state, specs, *, long_state=False, zoo_only=False):
        questions = []
        for index, (kind, question, options) in enumerate(specs, 1):
            questions.append({
                "id": f"{request_id}-q{index:02d}",
                "type": kind,
                "question": question,
                "options": options,
                "zoo_only": zoo_only,
            })
        requests.append({
            "id": request_id,
            "state": state,
            "long_state_intended": long_state,
            "questions": questions,
        })

    long_cases = [
        ("The ceiling lamp flickers whenever its loose cable is moved.",
         "Which work is needed?", ["electrical repair", "surface cleaning"]),
        ("The parcel has remained at a depot for nine days past its due date.",
         "What is the problem?", ["late delivery", "wrong color"]),
        ("The storage door cannot close because its hinge is bent.",
         "What needs attention?", ["a damaged hinge", "a missing label"]),
        ("The cooling fan has stopped and the room is becoming very warm.",
         "What should be checked?", ["the cooling equipment", "the floor covering"]),
        ("The same paid invoice was charged a second time.",
         "Which issue is described?", ["a duplicate charge", "a delivery delay"]),
        ("Water is escaping from a cracked pipe and covering the floor.",
         "Which response fits?", ["stop the water", "repaint the ceiling"]),
    ]
    filler = (" The record also contains an ordinary note about a box beside a "
              "shelf. The note adds no new detail about the reported problem.")
    for index, (opening, question, options) in enumerate(long_cases, 1):
        state = opening + filler * 24 + " The final note says the record ends here."
        add(f"request-{index:02d}", state, [("choice", question, options)],
            long_state=True)

    cases = [
        ("A seed sprouted roots and two green leaves.",
         "Which topic fits?",
         ["plants", "rocks", "stars", "poems", "roads", "music", "clothing", "numbers", "weather"],
         "Does this describe a plant?",
         "Rate the evidence of plant growth from 1 to 10.",
         [f"level {i}" for i in range(1, 11)]),
        ("A broken door lock prevents entry to the room.",
         "Which category fits?",
         ["lock", "roof", "window", "carpet", "wall", "drain", "lamp", "shelf", "chair", "curtain"],
         "Is entry blocked?",
         "Rate the disruption from 1, very low, to 10, very high.",
         [f"level {i}" for i in range(1, 11)]),
        ("Heavy rain left puddles along the path.",
         "Which condition is described?",
         ["rain", "snow", "fog", "heat", "frost", "wind", "dust", "smoke", "sunshine", "hail", "drought"],
         "Is the path dry?",
         "How wet is the path?", ["dry", "wet"]),
        ("The bowl contains rice and cooked beans.",
         "What is in the bowl?",
         ["rice", "sand", "water", "stones", "paper", "coins", "soil", "paint", "thread", "salt", "flour", "leaves"],
         "Does the bowl contain beans?",
         "Is the stated food quantity empty or nonempty?", ["empty", "nonempty"]),
        ("A folded cotton towel is on the shelf.",
         "Which material is named?",
         ["cotton", "wool", "silk", "linen", "wood", "glass", "stone", "steel", "copper", "rubber", "paper", "clay", "leather", "plastic"],
         "Is the towel unfolded?",
         "How much folding is described?", ["none", "some", "complete"]),
        ("A sparrow landed on a branch and sang.",
         "Which animal is named?",
         ["sparrow", "robin", "raven", "owl", "duck", "goose", "swan", "finch", "gull", "hawk", "eagle", "dove", "heron", "crane", "wren", "lark"],
         "Does the state mention singing?",
         "How active is the bird?", ["still", "slightly active", "active"]),
    ]
    for index, (state, question, options, yesno, score, levels) in enumerate(cases, 7):
        add(f"request-{index:02d}", state, [
            ("choice", question, options),
            ("noul", yesno, ["yes", "no"]),
            ("score", score, levels),
        ])

    small_cases = [
        ("The requested blue bowl arrived cracked. A replacement is wanted.", [
            ("choice", "Which request fits?", ["send an intact replacement", "cancel all future deliveries"]),
            ("choice", "What went wrong?", ["the bowl was damaged", "the address was missing"]),
            ("choice", "Which description fits?", ["a complaint about damage", "a request for directions"]),
            ("noul", "Is a replacement requested?", ["yes", "no"]),
            ("score", "How usable is the bowl?", ["unusable", "limited", "mostly usable", "fully usable"]),
        ]),
        ("The quiet room was clean, comfortable, and ready early.", [
            ("choice", "Which summary fits?", ["a pleasant quiet stay", "a noisy delayed arrival"]),
            ("choice", "How was the room?", ["clean and ready early", "dirty and still locked"]),
            ("choice", "Which feeling fits?", ["pleased with the experience", "angry about the delay"]),
            ("noul", "Does the state describe noise?", ["yes", "no"]),
            ("score", "Rate the experience.", ["poor", "fair", "good", "excellent"]),
        ]),
        ("The lift is stuck and no alternative access is available.", [
            ("choice", "Which action fits?", ["arrange an urgent repair", "update the wall color"]),
            ("choice", "What is unavailable?", ["an alternative access route", "a printed lunch menu"]),
            ("choice", "Which queue fits?", ["building access and maintenance", "gardening plans and supplies"]),
            ("noul", "Is another route available?", ["yes", "no"]),
            ("score", "Rate the urgency.", ["very low", "low", "medium", "high", "critical"]),
        ]),
        ("The copied page has a faint mark, but every word remains clear.", [
            ("choice", "Which summary fits?", ["a minor visible blemish", "a completely unreadable page"]),
            ("choice", "What remains usable?", ["all of the text", "none of the text"]),
            ("choice", "Which next step fits?", ["keep the readable copy", "report a missing parcel"]),
            ("noul", "Can every word be read?", ["yes", "no"]),
            ("score", "Rate the severity.", ["very low", "low", "medium", "high", "critical"]),
        ]),
    ]
    for index, (state, specs) in enumerate(small_cases, 13):
        add(f"request-{index:02d}", state, specs)

    categories = [
        "lighting", "plumbing", "heating", "cooling", "doors", "windows", "floors", "walls",
        "ceilings", "roofs", "stairs", "lifts", "locks", "shelves", "desks", "chairs",
        "curtains", "carpets", "drains", "pipes", "valves", "tanks", "pumps", "fans",
        "sockets", "switches", "cables", "sensors", "meters", "signs", "paths", "gates",
    ]
    zoo_states = [
        "The ceiling light no longer turns on.",
        "Water drips from the pipe below the sink.",
        "The fan rattles whenever it starts.",
        "The outer gate will not close.",
    ]
    for index, (count, state) in enumerate(zip((20, 24, 28, 32), zoo_states), 17):
        add(f"request-{index:02d}", state,
            [("choice", "Which maintenance category fits?", categories[:count])],
            zoo_only=True)
    return requests


def encoded_row(author, tok, base_tok, request, question, option, index):
    state = request["state"]
    head_text = "State:\n" + state
    tail_text = "\n\nQuestion:\n" + question["question"] + "\n\nOption:\n" + option
    head = tok(head_text, add_special_tokens=False, return_offsets_mapping=True)
    tail = tok(tail_text, add_special_tokens=False)["input_ids"]
    ids = author.encode(tok, state, question["question"], option, MAX_LEN)
    base_ids = author.encode(base_tok, state, question["question"], option, MAX_LEN)
    if ids != base_ids:
        raise AssertionError("The adapter and base tokenizer encode this fixture differently")
    retained = 0 if len(tail) >= MAX_LEN else min(len(head["input_ids"]), MAX_LEN - len(tail))
    expected = tail[-MAX_LEN:] if len(tail) >= MAX_LEN else head["input_ids"][:retained] + tail
    assert ids == expected and ids and len(ids) <= MAX_LEN
    assert tok.pad_token_id not in ids, "A fixture contains pad/eos; last-token pooling would differ"
    assert tok.eos_token_id not in ids, "A fixture contains eos"
    prefix_size = len("State:\n")
    state_offsets = [end > prefix_size for _, end in head["offset_mapping"]]
    return {
        "id": f"{question['id']}-option-{index:02d}",
        "request_id": request["id"],
        "question_id": question["id"],
        "type": question["type"],
        "zoo_only": question["zoo_only"],
        "option_index": index,
        "option": option,
        "ids": ids,
        "slot": len(ids) - 1,
        "length": len(ids),
        "state_cut": retained < len(head["input_ids"]),
        "state_tokens_total": sum(state_offsets),
        "state_tokens_survived": sum(state_offsets[:retained]),
        "head_tokens_total": len(head["input_ids"]),
        "head_tokens_survived": retained,
        "tail_tokens_total": len(tail),
        "tail_cut": len(tail) > MAX_LEN,
        "truncation": "tail suffix only" if len(tail) >= MAX_LEN else "state prefix kept; state end removed",
        "last_token_text": tok.decode([ids[-1]]),
        "scalar": None,
    }


def prepare_fixtures(author, tok, base_tok):
    requests, questions, rows = [], [], []
    for request in build_requests():
        raw_count = len(tok(request["state"], add_special_tokens=False)["input_ids"])
        if request["long_state_intended"]:
            assert raw_count > MAX_LEN
        requests.append({
            "id": request["id"],
            "state": request["state"],
            "state_tokens": raw_count,
            "longer_than_max_len": raw_count > MAX_LEN,
            "question_ids": [q["id"] for q in request["questions"]],
        })
        for question in request["questions"]:
            question_rows = [encoded_row(author, tok, base_tok, request, question, option, index)
                             for index, option in enumerate(question["options"])]
            questions.append({**question, "request_id": request["id"],
                              "row_ids": [row["id"] for row in question_rows],
                              "p_oracle": None, "argmax": None, "top2_margin": None})
            rows.extend(question_rows)
    counts = {kind: sum(q["type"] == kind and not q["zoo_only"] for q in questions)
              for kind in ("choice", "noul", "score")}
    wide = sum(q["type"] == "choice" and not q["zoo_only"] and 9 <= len(q["options"]) <= 16
               for q in questions)
    descriptive = sum(q["type"] == "choice" and not q["zoo_only"]
                      and all(len(option.split()) >= 3 for option in q["options"])
                      for q in questions)
    score_ten = sum(q["type"] == "score" and len(q["options"]) == 10 for q in questions)
    zoo = [q for q in questions if q["zoo_only"]]
    assert len(requests) >= 16 and len(questions) - len(zoo) >= 44
    assert counts["choice"] >= 24 and counts["noul"] >= 10 and counts["score"] >= 10
    assert wide >= 6 and descriptive >= 6 and score_ten >= 2
    assert len(zoo) >= 4 and all(20 <= len(q["options"]) <= 32 for q in zoo)
    assert all(2 <= len(q["options"]) <= 16 for q in questions if q["type"] == "choice" and not q["zoo_only"])
    assert all(2 <= len(q["options"]) <= 10 for q in questions if q["type"] == "score")
    assert all(q["options"] == ["yes", "no"] for q in questions if q["type"] == "noul")
    truncated_requests = sorted({r["request_id"] for r in rows if r["state_cut"]})
    assert len(truncated_requests) >= 6
    return {
        "schema": "coreai-scalar-fixtures/1",
        "license": LICENSE,
        "temperature": TEMPERATURE,
        "max_len": MAX_LEN,
        "layout": "State/Question/Option, last token",
        "requests": requests,
        "questions": questions,
        "rows": rows,
        "summary": {
            "requests": len(requests), "questions": len(questions),
            "questions_non_zoo": len(questions) - len(zoo), "rows": len(rows),
            "question_counts_non_zoo": counts, "wide_choice_questions": wide,
            "descriptive_choice_questions": descriptive, "score_ten_level_questions": score_ten,
            "zoo_only_questions": len(zoo), "zoo_only_rows": sum(r["zoo_only"] for r in rows),
            "states_truncated": len(truncated_requests), "truncated_request_ids": truncated_requests,
            "truncated_rows": sum(r["state_cut"] for r in rows),
            "state_longer_than_384_requests": sum(r["longer_than_max_len"] for r in requests),
            "total_tokens": sum(len(r["ids"]) for r in rows),
            "min_row_tokens": min(len(r["ids"]) for r in rows),
            "max_row_tokens": max(len(r["ids"]) for r in rows),
        },
    }



def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-snapshot", type=Path, required=True)
    parser.add_argument("--adapter-snapshot", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("scalar-work"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--comparison-out", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--deadline-epoch", type=float, default=float("inf"))
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    out = (args.out or work / "oracle-fixtures.json").resolve()
    comparison_out = (args.comparison_out or work / "oracle-comparison.json").resolve()
    started = time.monotonic()
    def progress(stage, **fields):
        item = {"stage": stage, "epoch": time.time(), "wall_seconds_contended": time.monotonic()-started, **fields}
        atomic_json(work / "oracle-progress.json", item)
        print(json.dumps(item, allow_nan=False), flush=True)
    def check_deadline():
        if time.time() >= args.deadline_epoch:
            raise TimeoutError("supplied deadline elapsed")
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        base, adapter = args.base_snapshot.resolve(), args.adapter_snapshot.resolve()
        author_path = adapter / "system_one.py"
        assert sha256(author_path) == AUTHOR_SHA256, "author script differs from pinned snapshot"
        spec = importlib.util.spec_from_file_location("system_one_pinned_author", author_path)
        author = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(author)
        tok = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
        base_tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if base_tok.pad_token is None:
            base_tok.pad_token = base_tok.eos_token
        fixture = prepare_fixtures(author, tok, base_tok)
        check_deadline()
        progress("loading_author_fp32_cpu", **fixture["summary"])
        author_tok, model = author.load_model(str(base), lora=False, dtype=torch.float32)
        assert tok.pad_token_id == author_tok.pad_token_id
        model = PeftModel.from_pretrained(model, str(adapter)).eval()
        assert sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()}) == ["torch.float32"]
        parity_rows = [next(r for r in fixture["rows"] if r["request_id"] == request_id)
            for request_id in ("request-07", "request-08", "request-09")]
        parity_ids, parity_mask = author.pad_batch([r["ids"] for r in parity_rows], tok.pad_token_id, MAX_LEN)
        with torch.no_grad():
            adapter_scores = model(input_ids=parity_ids, attention_mask=parity_mask).logits.squeeze(-1).float().tolist()
        progress("merging_fp32_in_memory")
        model = model.merge_and_unload(safe_merge=True).eval()
        with torch.no_grad():
            merged_scores = model(input_ids=parity_ids, attention_mask=parity_mask).logits.squeeze(-1).float().tolist()
        parity = {"tolerance": 1e-3, "max_abs_delta_scalar": max(abs(a-b) for a,b in zip(adapter_scores,merged_scores)),
            "sequences": [{"id": r["id"], "adapter_fp32": a, "merged_fp32": b, "abs_delta_scalar": abs(a-b)}
                for r,a,b in zip(parity_rows,adapter_scores,merged_scores)]}
        assert parity["max_abs_delta_scalar"] <= parity["tolerance"]
        atomic_json(work / "oracle-merge-parity.json", parity)
        assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters())
        fixture["oracle"] = {"model_class": type(model).__name__, "peft": False, "merged_in_memory": True,
            "device": "CPU", "dtype": "float32", "threads": 8, "batch_size": args.batch_size,
            "storage_rounding_applied": False, "merge_parity": parity, "author_script": str(author_path),
            "author_script_sha256": AUTHOR_SHA256, "adapter_revision": ADAPTER_REV,
            "base_revision": BASE_REV, "functions": ["load_model", "encode", "pad_batch", "score_options", "softmax"],
            "tokenizer": "adapter repository; every row agrees with base tokenizer",
            "state_token_count_definition": "tokens from State-prefixed head whose offset ends after State prefix"}
        requests = {r["id"]: r for r in fixture["requests"]}
        rows = {r["id"]: r for r in fixture["rows"]}
        score_start = time.monotonic()
        for index, question in enumerate(fixture["questions"], 1):
            check_deadline()
            question_start = time.monotonic()
            scalars = author.score_options(model, tok, requests[question["request_id"]]["state"],
                question["question"], question["options"], MAX_LEN, args.batch_size, "cpu")
            assert len(scalars) == len(question["row_ids"]) and all(math.isfinite(x) for x in scalars)
            for row_id, scalar in zip(question["row_ids"], scalars):
                rows[row_id]["scalar"] = scalar
            p = author.softmax([x / TEMPERATURE for x in scalars])
            ordered = sorted(p, reverse=True)
            question.update(scalars=scalars, p_oracle=p, argmax=max(range(len(p)), key=p.__getitem__),
                top2_margin=ordered[0]-ordered[1], wall_seconds=time.monotonic()-question_start)
            fixture["summary"].update(questions_completed=index, rows_completed=sum(r["scalar"] is not None for r in rows.values()))
            atomic_json(work / "oracle-fixtures.partial.json", fixture)
            progress("oracle_question_complete", question_id=question["id"], question_index=index,
                questions_total=len(fixture["questions"]), rows_completed=fixture["summary"]["rows_completed"])
        scalars = [r["scalar"] for r in fixture["rows"]]
        fixture["summary"].update(status="PASS", all_finite=True, scalar_min=min(scalars), scalar_max=max(scalars),
            scalar_max_absolute=max(abs(x) for x in scalars), wall=time.monotonic()-score_start,
            wall_seconds=time.monotonic()-score_start, total_wall_seconds_contended=time.monotonic()-started,
            near_tie_questions=[q["id"] for q in fixture["questions"] if q["top2_margin"]<0.02])
        atomic_json(out, fixture)
        if args.compare:
            canonical_sha_before = sha256(args.compare)
            canonical = json.loads(args.compare.read_text())
            identity_fields = ["id", "question_id", "request_id", "option_index", "ids", "slot", "state_cut", "state_tokens_survived", "tail_cut"]
            ids_identical = len(canonical["rows"]) == len(fixture["rows"]) and all(
                all(a[k] == b[k] for k in identity_fields) for a,b in zip(canonical["rows"], fixture["rows"]))
            scalar_deltas = [abs(a["scalar"]-b["scalar"]) for a,b in zip(canonical["rows"], fixture["rows"])]
            questions = [{"id": a["id"], "argmax_equal": a["argmax"] == b["argmax"],
                "max_abs_delta_p": max(abs(x-y) for x,y in zip(a["p_oracle"],b["p_oracle"]))}
                for a,b in zip(canonical["questions"],fixture["questions"])]
            max_dp = max(q["max_abs_delta_p"] for q in questions)
            record = {"schema": "coreai-scalar-oracle-comparison/1", "license": LICENSE,
                "canonical": str(args.compare.resolve()), "canonical_sha256_before": canonical_sha_before,
                "canonical_sha256_after": sha256(args.compare), "candidate": str(out), "candidate_sha256": sha256(out),
                "canonical_model": "unmerged PEFT CPU fp32", "candidate_model": "PEFT merge_and_unload in-memory CPU fp32",
                "rows": len(fixture["rows"]), "questions": questions, "ids_and_truncation_identical": ids_identical,
                "argmax_agreement": sum(q["argmax_equal"] for q in questions), "max_abs_delta_p": max_dp,
                "max_abs_delta_scalar": max(scalar_deltas), "mean_abs_delta_scalar": sum(scalar_deltas)/len(scalar_deltas),
                "tolerance_scalar": 1e-3, "probability_difference_is_acceptance_gate": False,
                "comparison_policy": "IDs and truncation exact, argmax exact, fp32 scalar difference <=1e-3; per-option probability drift informational. GPU aggregate max-error reproduction is checked separately at1e-6.",
                "result": "PASS" if ids_identical and max(scalar_deltas)<=1e-3 and all(q["argmax_equal"] for q in questions) else "FAIL"}
            assert record["canonical_sha256_before"] == record["canonical_sha256_after"]
            atomic_json(comparison_out, record)
            if record["result"] != "PASS":
                raise AssertionError(f"merged oracle reproduction differs: {record}")
        progress("oracle_complete", output=str(out), **fixture["summary"])
    except Exception as error:
        atomic_json(work / "oracle-error.json", {"result": "FAIL", "error": str(error), "traceback": traceback.format_exc()})
        raise

if __name__ == "__main__":
    main()
