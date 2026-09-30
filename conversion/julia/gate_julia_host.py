#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["coreai-core==1.0.0b2", "numpy>=2.2", "tokenizers>=0.22"]
# ///
"""Stage 2: the host (NumPy, `_julia_host.py`) vs the publisher's own host code — then text to answers on a bundle.

    python3 gate_julia_host.py                                            # rows, rendering, readout
    python3 gate_julia_host.py --bundle <exports>/julia-1/macos/fp32-s1024 --compute gpu   # + the bundle, end to end

Without a bundle (no model runs):
  rows       `build_row` (the `tokenizers` library on the checkpoint's tokenizer.json) rebuilds every oracle row
             from its request text — the 2,000 typed questions (strict, 1024 / 512), the 100 parity requests
             (strict, head 256), the window-filling rows (non-strict) — and at the 512 window's budget every row
             that fits it: ids and marker positions must be identical to the publisher's builder, every row
  rendering  `render_named` on each typed question's criteria gives the options and keys the publisher's
             typed-decisions reproduction feeds (2,000 of 2,000)
  readout    `answer` on the raw logits the publisher's engine gave for the rows predict_typed renders
             reproduces the publisher's answer dictionaries (oracle/named.json, 400 cases): same choice id, and
             every probability / score / noul within 1e-12 (the same Python float arithmetic)

With --bundle: `JuliaCoreAI.predict(state, questions)` — text in, the tokenizer, the bundle, the readout —
on the 400 cases (the questions that fit the folder's window) vs the publisher's `engine.predict`: the same
choice / argmax on every question, max |dp| <= 1e-3, and the typed-decisions accuracy re-scored from these
answers. gpu holds the machine-wide GPU lock (gate_julia_runtime.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (HEAD_LENGTH, MODEL_SHA, environment, hashes, oracle_dir, results_dir, source_dir,  # noqa: E402
                     verify_hashes, write_json)
from _julia_host import Tokenizer, answer, build_row, render_named  # noqa: E402

READOUT_TOLERANCE = 1e-12
BUNDLE_PROBABILITY_BAR = 1e-3


def check_rows(tokenizer, rows) -> dict:
    mismatches, checked, window512 = [], 0, {"checked": 0, "identical": 0, "refused": 0}
    for row in rows:
        request = row["request"]
        if row["set"] == "typed":
            args = (1024, HEAD_LENGTH[1024], True)
        elif row["set"] == "parity":
            args = (1024, 256, True)
        else:
            args = (row["fill_window"], HEAD_LENGTH[row["fill_window"]], False)
        built = build_row(tokenizer, request, *args)
        checked += 1
        if built["ids"] != row["ids"] or built["markers"] != row["markers"]:
            mismatches.append(row["row_id"])
        if row["set"] != "fill" and len(row["ids"]) <= 512:
            window512["checked"] += 1
            try:
                again = build_row(tokenizer, request, 512, HEAD_LENGTH[512], True)
            except ValueError:
                window512["refused"] += 1
                continue
            window512["identical"] += int(again["ids"] == row["ids"] and again["markers"] == row["markers"])
    return {"rows": checked, "identical": checked - len(mismatches), "mismatches": mismatches[:20], "window_512_budget": window512}


def check_rendering(rows) -> dict:
    typed = [r for r in rows if r["set"] == "typed"]
    same = 0
    for row in typed:
        question = row["named"]["question"]
        kind, keys, labels = render_named(question)
        same += int(kind == row["type"] and keys == row["keys"] and labels == row["request"]["options"])
    return {"typed_questions": len(typed), "identical": same}


def check_readout(named) -> dict:
    questions, same, worst = 0, 0, 0.0
    for case in named["cases"]:
        for (question_id, question), logits in zip(case["questions"].items(), case["rows_logits"]):
            kind, keys, _ = render_named(question)
            mine, theirs = answer(kind, keys, logits), case["answers"][question_id]
            questions += 1
            error = max(abs(mine["probabilities"][k] - theirs["probabilities"][k]) for k in keys)
            for field in ("score", "noul", "max_probability"):
                if field in theirs:
                    error = max(error, abs(mine[field] - theirs[field]))
            worst = max(worst, error)
            same += int(mine.get("choice") == theirs.get("choice") and error <= READOUT_TOLERANCE
                        and set(mine) == set(theirs))
    return {"questions": questions, "identical": same, "max_abs_difference": worst}


def check_bundle(folder: Path, compute: str, named, rows) -> dict:
    from julia_coreai import JuliaCoreAI

    julia = JuliaCoreAI(folder, compute)
    gold = {r["row_id"]: r["gold"] for r in rows if r["set"] == "typed"}
    records, refused, started = [], [], time.perf_counter()
    for case in named["cases"]:
        for question_id, question in case["questions"].items():
            theirs = case["answers"][question_id]
            try:
                mine = julia.predict(case["state"], {question_id: question})["answers"][question_id]
            except ValueError as error:
                refused.append({"question": f"{case['id']}:{question_id}", "reason": str(error)})
                continue
            keys = list(theirs["probabilities"])
            p = np.asarray([mine["probabilities"][k] for k in keys])
            q = np.asarray([theirs["probabilities"][k] for k in keys])
            winner = keys[int(p.argmax())]
            records.append({"question": f"{case['id']}:{question_id}", "type": theirs["type"],
                            "argmax_identical": int(p.argmax()) == int(q.argmax()),
                            "max_probability_error": float(np.max(np.abs(p - q))),
                            "answer_value_error": abs(float(mine.get("score", mine.get("noul", mine.get("max_probability"))))
                                                      - float(theirs.get("score", theirs.get("noul", theirs.get("max_probability"))))),
                            "winner": winner, "correct": winner == gold.get(f"typed:{case['id']}:{question_id}")})
    by_type = {}
    for kind in ("choice", "score", "noul"):
        of_kind = [r for r in records if r["type"] == kind]
        by_type[kind] = {"count": len(of_kind), "correct": sum(r["correct"] for r in of_kind)}
    worst = max(r["max_probability_error"] for r in records)
    status = "PASS" if all(r["argmax_identical"] for r in records) and worst <= BUNDLE_PROBABILITY_BAR else "FAIL"
    return {"status": status, "folder": str(folder), "compute": compute, "questions": len(records),
            "refused": len(refused), "refused_examples": refused[:5],
            "argmax_identical": sum(r["argmax_identical"] for r in records), "max_probability_error": worst,
            "max_answer_value_error": max(r["answer_value_error"] for r in records),
            "typed_accuracy_from_these_answers": by_type, "seconds": time.perf_counter() - started, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, help="an exported macos/<dtype>-s<S> folder: also run text -> answers on it")
    parser.add_argument("--compute", choices=["gpu", "cpu_only"], default="gpu")
    args = parser.parse_args()
    started = time.perf_counter()
    oracle_record = json.loads((results_dir() / "oracle.json").read_text())
    named_record = json.loads((results_dir() / "oracle_named.json").read_text())
    rows_file, named_file = oracle_dir() / "rows.json", oracle_dir() / "named.json"
    verify_hashes({str(rows_file): oracle_record["output_hashes"][str(rows_file)],
                   str(named_file): named_record["output_hashes"][str(named_file)]})
    rows = json.loads(rows_file.read_text())["rows"]
    named = json.loads(named_file.read_text())
    tokenizer = Tokenizer(source_dir() / "tokenizer" / "tokenizer.json")

    result = {"stage": "host", "model_sha": MODEL_SHA, "rows": check_rows(tokenizer, rows),
              "rendering": check_rendering(rows), "readout": check_readout(named)}
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "mismatches"} for k, v in result.items() if isinstance(v, dict)}), flush=True)
    failures = []
    if result["rows"]["identical"] != result["rows"]["rows"]:
        failures.append("rows")
    w = result["rows"]["window_512_budget"]
    if w["identical"] + w["refused"] != w["checked"] or w["refused"]:
        failures.append("512-window budget")
    if result["rendering"]["identical"] != result["rendering"]["typed_questions"]:
        failures.append("rendering")
    if result["readout"]["identical"] != result["readout"]["questions"]:
        failures.append("readout")
    name = "host"
    if args.bundle:
        handle = None
        if args.compute == "gpu":
            from gate_julia_runtime import acquire_gpu_lock
            handle, lock = acquire_gpu_lock(30 * 60)
        try:
            result["bundle"] = check_bundle(args.bundle, args.compute, named, rows)
        finally:
            if handle is not None:
                handle.close()
        b = result["bundle"]
        print(json.dumps({k: v for k, v in b.items() if k != "records"}), flush=True)
        if b["status"] != "PASS":
            failures.append("bundle end to end")
        name = f"host_{args.bundle.name}_{args.compute}"
    result.update(status="FAIL" if failures else "PASS", failures=failures, environment=environment(("coreai-core", "numpy", "tokenizers")),
                  input_hashes=hashes([rows_file, named_file, Path(__file__), Path(__file__).parent / "_julia_host.py",
                                       Path(__file__).parent / "julia_coreai.py"]),
                  seconds=time.perf_counter() - started)
    out = results_dir() / f"{name}.json"
    write_json(out, result)
    print(result["status"], failures, "->", out, flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
