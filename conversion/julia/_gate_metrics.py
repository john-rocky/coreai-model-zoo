"""One row-level gate shared by every stage (authoring, export, Mac runtime, device).

A row is one question. The candidate's input is its raw marker logits (K floats, the graph output at the
marker positions). Two references, both the publisher's own model in fp32 (oracle_julia.py):

  answer reference  `engine.logits` of the publisher's resident engine, one question per forward — the
                    run that reproduces the publisher's 426 / 542 / 483 on typed-decisions
  tensor reference  the publisher's model on the same ids right-padded to the export window, batch 1

Answer gate: finite; argmax identical on every row (choice, score and noul alike, no near-tie exemption);
max |dp| <= 1e-3 over the options at T = 1. Tensor gate: marker max |d| <= 1e-3. The typed rows also
re-score the benchmark from the candidate's argmax, so every stage states its own 426 / 542 / 483.
Per-row records keep the raw arrays unrounded so every aggregate can be recomputed.
"""
from __future__ import annotations

import numpy as np

from _julia_host import answer, softmax

POLICY = {
    "probability_max_abs": 1e-3,
    "marker_max_abs": 1e-3,
    "argmax": "identical on every row, choice / score / noul (no near-tie exemption)",
    "temperature": "1 (the checkpoint has no calibration; inference-policy.json calibration null)",
    "answer_reference": "the publisher's engine.logits, one question per forward, CPU fp32",
    "tensor_reference": "the publisher's model on the same ids right-padded to the window, batch 1, CPU fp32",
}
QTYPES = ("choice", "score", "noul")


def evaluate_row(row: dict, marker_logits) -> dict:
    marker = np.asarray(marker_logits, dtype=np.float32).reshape(-1)
    k = len(row["markers"])
    finite = bool(np.isfinite(marker).all()) and marker.shape == (k,)
    record = {"row_id": row["row_id"], "set": row["set"], "window": row["window"], "tokens": len(row["ids"]),
              "K": k, "type": row["type"], "raw_marker_logits": [float(v) for v in marker], "finite": finite}
    if not finite:
        record.update(answer_pass=False, tensor_pass=False)
        return record
    reference = np.asarray(row["publisher_logits"], dtype=np.float64)
    tensor = np.asarray(row["marker_logits"], dtype=np.float64)
    p = np.asarray(softmax(marker), dtype=np.float64)
    reference_p = np.asarray(softmax(reference), dtype=np.float64)
    ordered = np.sort(reference_p)
    mine, theirs = answer(row["type"], row["keys"], marker), answer(row["type"], row["keys"], reference)
    value = {"choice": "max_probability", "score": "score", "noul": "noul"}[row["type"]]
    record.update(
        probabilities=[float(v) for v in p],
        max_probability_error=float(np.max(np.abs(p - reference_p))),
        argmax_identical=bool(int(p.argmax()) == int(reference_p.argmax())),
        reference_top_two_gap=float(ordered[-1] - ordered[-2]),
        answer_value_error=abs(float(mine[value]) - float(theirs[value])),
        marker_max_abs_error=float(np.max(np.abs(marker.astype(np.float64) - tensor))),
        prediction=row["keys"][int(p.argmax())])
    record["answer_pass"] = bool(record["max_probability_error"] <= POLICY["probability_max_abs"] and record["argmax_identical"])
    record["tensor_pass"] = bool(record["marker_max_abs_error"] <= POLICY["marker_max_abs"])
    if row.get("gold") is not None:
        record["correct"] = record["prediction"] == row["gold"]
    return record


def summarize(records: list[dict]) -> dict:
    """Aggregates recomputable from the per-row records; `answer_status` and `tensor_status` separately."""
    finite = [r for r in records if r["finite"]]
    typed = [r for r in finite if r["set"] == "typed"]
    summary = {
        "rows": len(records),
        "rows_by_set": {s: sum(r["set"] == s for r in records) for s in ("typed", "parity", "fill")},
        "finite_rows": len(finite),
        "argmax_identical": sum(r["argmax_identical"] for r in finite),
        "argmax_identical_gap_over_1e-3": sum(r["argmax_identical"] for r in finite if r["reference_top_two_gap"] > 1e-3),
        "rows_gap_over_1e-3": sum(r["reference_top_two_gap"] > 1e-3 for r in finite),
        "max_probability_error": max((r["max_probability_error"] for r in finite), default=None),
        "max_answer_value_error": max((r["answer_value_error"] for r in finite), default=None),
        "max_marker_abs_error": max((r["marker_max_abs_error"] for r in finite), default=None),
        "min_reference_top_two_gap": min((r["reference_top_two_gap"] for r in finite), default=None),
        "answer_failures": [r["row_id"] for r in records if not r["answer_pass"]],
        "tensor_failures": [r["row_id"] for r in records if not r["tensor_pass"]],
    }
    if typed:
        summary["typed_accuracy"] = {k: {"count": sum(r["type"] == k for r in typed),
                                         "correct": sum(bool(r.get("correct")) for r in typed if r["type"] == k)}
                                     for k in QTYPES}
    summary["answer_status"] = "PASS" if records and not summary["answer_failures"] else "FAIL"
    summary["tensor_status"] = "PASS" if records and not summary["tensor_failures"] else "FAIL"
    return summary


def wrong_pairing(rows: list[dict], candidates: dict[str, list[float]]) -> dict:
    """Each row judged against the outputs of the NEXT row with the same (type, K): must FAIL.

    Pairing within a shape class keeps the control about values, not about a shape mismatch.
    """
    by_class: dict[tuple, list[str]] = {}
    for row in rows:
        by_class.setdefault((row["type"], len(row["markers"])), []).append(row["row_id"])
    lookup = {row["row_id"]: row for row in rows}
    records, unpaired = [], []
    for members in by_class.values():
        if len(members) < 2:
            unpaired.extend(members)
            continue
        for i, row_id in enumerate(members):
            records.append(evaluate_row(lookup[row_id], candidates[members[(i + 1) % len(members)]]))
    failing = [r["row_id"] for r in records if not r["answer_pass"]]
    return {"status": "FAIL" if failing else "PASS", "paired_rows": len(records), "rows_failing": len(failing),
            "unpaired_rows": unpaired}
