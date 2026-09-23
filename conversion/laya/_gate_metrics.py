"""One row-level gate shared by every stage (oracle, authoring, export, Mac runtime, device).

A row is one question. Its inputs are the candidate's raw marker logits (K floats, the graph output
at the marker positions) and act logits (2 floats). Two references:

  answer reference  the frozen fixture: official probabilities at T=1, official act probability,
                    official answer dictionary (the batched `Agent.predict` of the LiteRT lane)
  tensor reference  the official DecisionModel on the same ids padded to the same window, batch 1
                    (this port's oracle), or any other named tensor reference

Answer gate: finite, argmax identical on every choice/score row, max |dp| <= 1e-3 over the options
at T=1, |d act_probability| <= 1e-3. Tensor gate: marker max |d| <= 1e-3, act max |d| / max |act_ref|
<= 1e-4. Per-row records keep the raw arrays unrounded so every aggregate can be recomputed.
"""
from __future__ import annotations

import numpy as np

from _laya_host import decode, probabilities, softmax

POLICY = {
    "marker_max_abs": 1e-3,
    "act_relative": 1e-4,
    "probability_max_abs": 1e-3,
    "act_probability_max_abs": 1e-3,
    "argmax": "identical on every choice/score row (no near-tie exemption)",
    "temperature": "the source config's [1,1,1], no buckets — the frozen fixture's temperature",
}


def evaluate_row(row: dict, marker_logits, act_logits, config: dict, tensor_reference: dict | None = None) -> dict:
    """`tensor_reference` = {"marker_logits": [...K], "act_logits": [...2]} or None (answer gate only)."""
    marker = np.asarray(marker_logits, dtype=np.float32).reshape(-1)
    act = np.asarray(act_logits, dtype=np.float32).reshape(-1)
    k, qtype = row["K"], row["qtype"]
    finite = bool(np.isfinite(marker).all() and np.isfinite(act).all()) and marker.shape == (k,) and act.shape == (2,)
    record = {"row_id": row["row_id"], "window": row["window"], "sequence_length": row["sequence_length"],
              "K": k, "qtype": qtype, "raw_marker_logits": [float(v) for v in marker],
              "act_logits": [float(v) for v in act], "finite": finite}
    if not finite:
        record.update(answer_pass=False, tensor_pass=False if tensor_reference is not None else None)
        return record
    p = probabilities(marker, qtype, config)
    reference_p = np.asarray(row["probabilities"], dtype=np.float64)
    act_probability = float(softmax(act)[0])
    probability_error = float(np.max(np.abs(p.astype(np.float64) - reference_p)))
    act_probability_error = abs(act_probability - float(row["act_probability"]))
    choice_score = qtype in (0, 1)
    ordered = np.sort(reference_p)
    argmax_identical = bool(int(p.argmax()) == int(reference_p.argmax())) if choice_score else None
    decoded = decode(marker, act, row["question"], config)
    record.update(
        probabilities=[float(v) for v in p], act_probability=act_probability,
        max_probability_error=probability_error, act_probability_error=act_probability_error,
        argmax_identical=argmax_identical, reference_top_two_gap=float(ordered[-1] - ordered[-2]),
        exact_dict=decoded == row["official_answer"],
        answer_pass=bool(probability_error <= POLICY["probability_max_abs"]
                         and act_probability_error <= POLICY["act_probability_max_abs"]
                         and (argmax_identical is not False)))
    if tensor_reference is not None:
        ref_marker = np.asarray(tensor_reference["marker_logits"], dtype=np.float64).reshape(-1)
        ref_act = np.asarray(tensor_reference["act_logits"], dtype=np.float64).reshape(-1)
        marker_error = float(np.max(np.abs(marker.astype(np.float64) - ref_marker)))
        act_error = float(np.max(np.abs(act.astype(np.float64) - ref_act)))
        act_scale = float(np.max(np.abs(ref_act)))
        if act_scale == 0:
            raise ValueError(f"{row['row_id']}: zero act reference scale")
        record.update(marker_max_abs_error=marker_error, act_max_abs_error=act_error, act_reference_scale=act_scale,
                      act_relative_error=act_error / act_scale,
                      tensor_pass=bool(marker_error <= POLICY["marker_max_abs"]
                                       and act_error / act_scale <= POLICY["act_relative"]))
    return record


def summarize(records: list[dict]) -> dict:
    """Aggregates recomputable from the per-row records; `answer_status` and `tensor_status` separately."""
    finite = [r for r in records if r["finite"]]
    choice_score = [r for r in finite if r["qtype"] in (0, 1)]
    summary = {
        "rows": len(records),
        "finite_rows": len(finite),
        "choice_score_rows": sum(r["qtype"] in (0, 1) for r in records),
        "argmax_identical": sum(r["argmax_identical"] is True for r in choice_score),
        "max_probability_error": max((r["max_probability_error"] for r in finite), default=None),
        "max_act_probability_error": max((r["act_probability_error"] for r in finite), default=None),
        "exact_dict_rows": sum(bool(r.get("exact_dict")) for r in finite),
        "min_reference_top_two_gap": min((r["reference_top_two_gap"] for r in choice_score), default=None),
        "answer_failures": [r["row_id"] for r in records if not r["answer_pass"]],
    }
    summary["answer_status"] = "PASS" if records and not summary["answer_failures"] else "FAIL"
    tensor = [r for r in records if r.get("tensor_pass") is not None]
    if tensor:
        summary.update(
            max_marker_abs_error=max(r.get("marker_max_abs_error", float("inf")) for r in tensor),
            max_act_abs_error=max(r.get("act_max_abs_error", float("inf")) for r in tensor),
            max_act_relative_error=max(r.get("act_relative_error", float("inf")) for r in tensor),
            max_act_reference_scale=max(r.get("act_reference_scale", 0.0) for r in tensor),
            tensor_failures=[r["row_id"] for r in tensor if not r["tensor_pass"]])
        summary["tensor_status"] = "PASS" if len(tensor) == len(records) and not summary["tensor_failures"] else "FAIL"
    return summary


def wrong_pairing(rows: list[dict], candidates: dict[str, dict], config: dict) -> dict:
    """Each row judged against the outputs of the NEXT row with the same (qtype, K): must FAIL.

    Pairing within a shape class keeps the control about values, not about a shape mismatch.
    """
    by_class: dict[tuple, list[str]] = {}
    for row in rows:
        by_class.setdefault((row["qtype"], row["K"]), []).append(row["row_id"])
    lookup = {row["row_id"]: row for row in rows}
    records, unpaired = [], []
    for members in by_class.values():
        if len(members) < 2:
            unpaired.extend(members)
            continue
        for i, row_id in enumerate(members):
            donor = candidates[members[(i + 1) % len(members)]]
            records.append(evaluate_row(lookup[row_id], donor["marker_logits"], donor["act_logits"], config))
    failing = [r["row_id"] for r in records if not r["answer_pass"]]
    return {"status": "FAIL" if failing else "PASS", "paired_rows": len(records), "rows_failing": len(failing),
            "unpaired_rows": unpaired}
