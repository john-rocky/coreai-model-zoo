#!/usr/bin/env python3
"""Write models/julia-1/gate-julia-1.json: every gate's summary + the sha256 of its full record.

    python3 conversion/julia/transcript_julia.py

Reads what the stages wrote (the work dir's results/, each exported folder's provenance/) and copies no number
by hand: each entry names the record it came from, by path and sha256, so any figure on the card can be traced
to a record and recomputed from its per-row arrays.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import HF_REPO, MODEL_ID, MODEL_SHA, export_root, results_dir, sha256_of, work_dir  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import repo_root  # noqa: E402

ROW_KEYS = ("rows", "rows_by_set", "argmax_identical", "max_probability_error", "max_answer_value_error",
            "max_marker_abs_error", "min_reference_top_two_gap", "typed_accuracy", "answer_status", "tensor_status")


def record(path: Path, root: Path, label: str) -> dict:
    return {"path": f"{label}/{path.relative_to(root)}", "sha256": sha256_of(path)}


def pick(summary: dict, keys) -> dict:
    return {k: summary.get(k) for k in keys}


def main():
    work, exports = work_dir(), export_root()
    oracle = json.loads((results_dir() / "oracle.json").read_text())
    named = json.loads((results_dir() / "oracle_named.json").read_text())
    transcript = {
        "model": MODEL_ID, "model_sha": MODEL_SHA, "hf_repo": HF_REPO,
        "gate": ("the publisher's julia package (FastEngine, transformers 5.0.0, SDPA attention, CPU fp32, strict "
                 "encoding, max_length 1024, head_length 512, marker-only head off), one question per forward"),
        "oracle": {"status": oracle["status"], "record": record(results_dir() / "oracle.json", work, "<work>"),
                   "typed_accuracy": oracle["typed_accuracy"], "vs_publisher_reproduce_typed": oracle["vs_publisher_reproduce_typed"],
                   "parity": oracle["parity"], "budget_check": {k: v for k, v in oracle["budget_check"].items() if k != "refused_at_head_256"},
                   "rows_per_window": oracle["rows_per_window"], "rows_by_set": oracle["rows_by_set"],
                   "boundary": oracle["boundary"], "tokens": oracle["tokens"],
                   "code_path_diagnostics": {w: {k: v for k, v in d.items() if k in ("rows", "max_padded_vs_engine", "sdpa_vs_eager_marker_max_abs")}
                                             for w, d in oracle["code_path_diagnostics"].items()},
                   "facts": oracle["facts"], "environment": oracle["environment"]},
        "oracle_named": {"status": named["status"], "cases": named["cases"], "questions": named["questions"],
                         "record": record(results_dir() / "oracle_named.json", work, "<work>")},
        "authoring": [], "host": [], "export": [], "runtime_rows": [],
    }
    for path in sorted(results_dir().glob("authoring_*_s*.json")):
        a = json.loads(path.read_text())
        layer = a["layer_gate"]
        transcript["authoring"].append({
            "precision": a["precision"], "window": a["window"], "status": a["status"], "failures": a["failures"],
            "rows": pick(a["summary"], ROW_KEYS),
            "layer_gate": {"status": layer["status"], "max_absolute_tier": layer["max_absolute_tier"],
                           "max_relative_tier": layer["max_relative_tier"], "max_token_logits_real": layer["max_token_logits_real"],
                           "per_state": {n: {k: s.get(k) for k in ("tier", "value", "bar", "max_abs_real", "max_relative_real",
                                                                  "relative_real_vs_eager", "official_sdpa_vs_eager_relative_real",
                                                                  "reference_abs_max_real")}
                                         for n, s in layer["per_state"].items()}},
            "pad_isolation": a["pad_isolation"],
            "negative_controls": {m: {k: c[k] for k in ("caught", "caught_by", "first_failing_state", "value_at_first_failing_state",
                                                        "min_relative_tier_value", "max_absolute_tier_value", "rows")}
                                  for m, c in a["negative_controls"].items()},
            "record": record(path, work, "<work>"), "environment": a["environment"]})
    transcript["exactness"] = []
    for path in sorted(results_dir().glob("exactness_s*.json")):
        x = json.loads(path.read_text())
        transcript["exactness"].append({"window": x["window"], "status": x["status"], "torch": x["torch"], "rows": x["rows"],
                                        "bars": x["bars"], "markers": x["markers"],
                                        "max_relative_vs_eager": max(v["vs_eager_relative"] for v in x["per_state"].values()),
                                        "final_norm": x["per_state"]["final_norm"], "record": record(path, work, "<work>")})
    transcript["swift"] = []
    for path in sorted(results_dir().glob("swift_parity_*.json")):
        x = json.loads(path.read_text())
        transcript["swift"].append({"runs": {c: json.loads(v["stdout"]) for c, v in x.items() if c in ("gpu", "cpu_only")},
                                    "what": ("swift/JuliaDecisions.swift on the strict fixture rows, the tokenizer replaced by "
                                             "the publisher's token ids per text piece (swift_pieces_julia.py)"),
                                    "record": record(path, work, "<work>")})
    for path in sorted(results_dir().glob("host*.json")):
        h = json.loads(path.read_text())
        entry = {"status": h["status"], "failures": h["failures"], "rows": {k: v for k, v in h["rows"].items() if k != "mismatches"},
                 "rendering": h["rendering"], "readout": h["readout"], "record": record(path, work, "<work>")}
        if "bundle" in h:
            entry["bundle"] = {k: v for k, v in h["bundle"].items() if k not in ("records", "refused_examples")}
        transcript["host"].append(entry)
    for manifest_path in sorted(exports.glob("*/*/provenance/export-manifest.json")):
        folder = manifest_path.parent.parent
        m = json.loads(manifest_path.read_text())
        entry = {"folder": str(folder.relative_to(exports)), "status": m["status"], "measure_only": m.get("measure_only", False),
                 "bundle": m["bundle"], "format": m["format"], "precision": m["precision"], "window": m["window"],
                 "bytes": m["bytes"], "torch_export_gate": m["torch_export_gate"], "record": record(manifest_path, exports, "<exports>")}
        transcript["export"].append(entry)
        gate_path = folder / "provenance" / "runtime-gate.json"
        if gate_path.exists():
            for compute, r in json.loads(gate_path.read_text()).items():
                transcript["runtime_rows"].append({
                    "folder": entry["folder"], "compute": compute, "status": r["status"], "findings": r["findings"],
                    "precision": r["precision"], "window": r["window"], **pick(r["summary"], ROW_KEYS),
                    "repeat_max_abs": r["repeat_max_abs"], "repeat_rows": r["repeat_rows"],
                    "wrong_pairing_control": r["wrong_pairing_control"], "load_ms": r["load_ms"],
                    "first_question_ms": r["first_question_ms"],
                    "warm": {k: v for k, v in r["warm"].items() if k != "samples_ms"}, "gpu_lock": r.get("gpu_lock"),
                    "device": {k: r["environment"][k] for k in ("machine", "macos_version", "macos_build", "coreai_build")},
                    "date": r["environment"]["date"], "record": record(gate_path, exports, "<exports>")})
    out = repo_root() / "models" / "julia-1" / "gate-julia-1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(transcript, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    print(out, len(transcript["authoring"]), "authoring,", len(transcript["export"]), "folders,",
          len(transcript["runtime_rows"]), "runtime rows,", len(transcript["host"]), "host")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
