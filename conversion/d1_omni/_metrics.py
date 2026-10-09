"""Row-level readout metrics shared by the export gate (export_decide.py) and the runtime gate (runtime_check.py).

A row is one question of a bundle's reference.json: the candidate's raw marker logits (K floats, the graph's
scores at P + markers) against the publisher's fp32 model on the same ids (ref/records_ref.json, round 2). The
logits become the answer distribution exactly as host.probabilities_from_logits makes it (the per-type temperature
for text rows, the fp32 softmax, a noul reported as [yes, no]).

Bars:
  FP32_BAR  round 2's eager bar (eager_check.py): argmax equal on every row, near ties included; max |dp| <= 2e-5;
            marker logits max |d| <= 1e-3
  SHIP_BAR  FACTS §7: argmax equal on every non-near-tie row (near tie = the oracle's top-2 margin <= 0.02, reported
            apart); max |dp| <= 0.02; the mean over rows of each row's max |dp| <= 0.002
The wrong-pairing control judges each row's candidate against the oracle of the next row of the same class
(type, K, mode, so the same temperature) under SHIP_BAR. It must FAIL, or the instrument cannot go red.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

import host

FP32_BAR = {"name": "fp32 (round 2 eager bar)", "argmax": "every row, near ties included", "max_abs_dp": 2e-5,
            "max_abs_dlogit": 1e-3}
SHIP_BAR = {"name": "ship (FACTS §7)", "argmax": "every non-near-tie row; near ties reported apart",
            "max_abs_dp": 0.02, "mean_row_max_abs_dp": 0.002}
NEAR_TIE_MARGIN = 0.02


def question_of(row: dict) -> host.Question:
    return host.as_question(row["question"])


def row_record(row: dict, logits) -> dict:
    """One row's candidate marker logits against its oracle. Arrays are kept unrounded."""
    z = np.asarray(logits, dtype=np.float32).reshape(-1)
    oracle = row["oracle"]
    record = {"id": row["id"], "qid": row["qid"], "mode": row["mode"], "source": row["source"], "type": row["type"],
              "K": row["K"], "positions": row["positions"], "near_tie": row["near_tie"],
              "top2_margin": row["top2_margin"], "logits": [float(v) for v in z],
              "finite": bool(z.shape == (row["K"],) and np.isfinite(z).all())}
    if not record["finite"]:
        record.update(probs=None, max_abs_dp=float("inf"), max_abs_dlogit=float("inf"), argmax_equal=False)
        return record
    p = host.probabilities_from_logits(z, question_of(row), row["calibrate"])
    record.update(
        probs=[float(v) for v in p],
        max_abs_dp=float(max(abs(a - b) for a, b in zip(p, oracle["probs"]))),
        max_abs_dlogit=float(np.max(np.abs(z.astype(np.float64) - np.asarray(oracle["logits_raw"], dtype=np.float64)))),
        argmax_equal=int(np.argmax(p)) == row["argmax_index"])
    return record


def _worst(records: list[dict], key: str) -> dict:
    r = max(records, key=lambda x: x[key])
    return {"value": r[key], "row": f"{r['id']}/{r['qid']}", "mode": r["mode"], "near_tie": r["near_tie"]}


def summarize(records: list[dict]) -> dict:
    """Aggregates (recomputable from the per-row records) and the verdict of both bars."""
    finite = [r for r in records if r["finite"]]
    ties = [r for r in records if r["near_tie"]]
    clear = [r for r in records if not r["near_tie"]]
    n = len(records)
    max_dp = max((r["max_abs_dp"] for r in records), default=float("nan"))
    max_dl = max((r["max_abs_dlogit"] for r in records), default=float("nan"))
    mean_dp = float(np.mean([r["max_abs_dp"] for r in records])) if records and len(finite) == n else float("inf")
    by_source = defaultdict(list)
    for r in records:
        by_source[f"{r['source']}/{r['mode']}"].append(r)
    worst = {s: {"rows": len(rs), "argmax_equal": sum(r["argmax_equal"] for r in rs),
                 "max_abs_dp": _worst(rs, "max_abs_dp"), "max_abs_dlogit": _worst(rs, "max_abs_dlogit"),
                 "mean_row_max_abs_dp": float(np.mean([r["max_abs_dp"] for r in rs]))}
             for s, rs in sorted(by_source.items())}
    fp32 = {"argmax": all(r["argmax_equal"] for r in records), "max_abs_dp": max_dp <= FP32_BAR["max_abs_dp"],
            "max_abs_dlogit": max_dl <= FP32_BAR["max_abs_dlogit"], "finite": len(finite) == n}
    ship = {"argmax_non_near_tie": all(r["argmax_equal"] for r in clear), "max_abs_dp": max_dp <= SHIP_BAR["max_abs_dp"],
            "mean_row_max_abs_dp": mean_dp <= SHIP_BAR["mean_row_max_abs_dp"], "finite": len(finite) == n}
    return {
        "rows": n, "finite_rows": len(finite),
        "argmax_equal": sum(r["argmax_equal"] for r in records),
        "non_near_tie": {"rows": len(clear), "argmax_equal": sum(r["argmax_equal"] for r in clear)},
        "near_tie": {"rows": len(ties), "argmax_equal": sum(r["argmax_equal"] for r in ties),
                     "flipped": [f"{r['id']}/{r['qid']} (margin {r['top2_margin']:.4f})" for r in ties if not r["argmax_equal"]]},
        "argmax_flips": [f"{r['id']}/{r['qid']}" for r in records if not r["argmax_equal"]],
        "max_abs_dp": max_dp, "max_abs_dp_row": _worst(records, "max_abs_dp") if records else None,
        "mean_row_max_abs_dp": mean_dp,
        "p99_row_max_abs_dp": float(np.percentile([r["max_abs_dp"] for r in records], 99)) if finite else None,
        "max_abs_dlogit": max_dl, "max_abs_dlogit_row": _worst(records, "max_abs_dlogit") if records else None,
        "by_source_mode": worst,
        "fp32_bar": {"bar": FP32_BAR, "verdict": fp32, "status": "PASS" if all(fp32.values()) else "FAIL"},
        "ship_bar": {"bar": SHIP_BAR, "verdict": ship, "status": "PASS" if all(ship.values()) else "FAIL"},
    }


def wrong_pairing(rows: list[dict], records: list[dict]) -> dict:
    """Each row's candidate logits judged against the oracle of the next row of its class (type, K, mode). The
    control must FAIL under SHIP_BAR (a pairing within a class is about values, not about a shape mismatch)."""
    by_class = defaultdict(list)
    for i, row in enumerate(rows):
        by_class[(row["type"], row["K"], row["mode"])].append(i)
    judged, unpaired = [], []
    for members in by_class.values():
        if len(members) < 2:
            unpaired.extend(f"{rows[i]['id']}/{rows[i]['qid']}" for i in members)
            continue
        for j, i in enumerate(members):
            donor = rows[members[(j + 1) % len(members)]]
            impostor = dict(rows[i], oracle=donor["oracle"], argmax_index=donor["argmax_index"],
                            near_tie=donor["near_tie"], top2_margin=donor["top2_margin"])
            judged.append(row_record(impostor, records[i]["logits"]))
    summary = summarize(judged)
    return {"status": summary["ship_bar"]["status"], "must_be": "FAIL", "caught": summary["ship_bar"]["status"] == "FAIL",
            "paired_rows": len(judged), "unpaired_rows": unpaired,
            "rows_over_dp_bar": sum(r["max_abs_dp"] > SHIP_BAR["max_abs_dp"] for r in judged),
            "argmax_mismatch_non_near_tie": summary["non_near_tie"]["rows"] - summary["non_near_tie"]["argmax_equal"],
            "max_abs_dp": summary["max_abs_dp"], "mean_row_max_abs_dp": summary["mean_row_max_abs_dp"],
            "verdict": summary["ship_bar"]["verdict"]}
