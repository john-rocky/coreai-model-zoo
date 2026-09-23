#!/usr/bin/env python3
"""Write models/laya-multilingual/gate-laya-multilingual.json: every gate's summary + the sha256 of its full record.

    python3 conversion/laya/transcript_laya.py

Reads what the stages wrote (the work dir's results/ and authored144/, each exported folder's provenance/)
and copies no number by hand: each row names the record it came from, by path and sha256, so any figure
on the card can be traced to a record and recomputed from its per-row arrays.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_ID, MODEL_SHA, SUBFOLDER, export_root, results_dir, sha256_of, work_dir  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import repo_root  # noqa: E402

HF_REPO = "mlboydaisuke/Laya-Multilingual-CoreAI"


def record(path: Path, root: Path, label: str) -> dict:
    return {"path": f"{label}/{path.relative_to(root)}", "sha256": sha256_of(path)}


def pick(summary: dict, keys) -> dict:
    return {k: summary.get(k) for k in keys}


def main():
    work, exports = work_dir(), export_root()
    oracle = json.loads((results_dir() / "oracle.json").read_text())
    transcript = {
        "model": MODEL_ID, "subfolder": SUBFOLDER, "model_sha": MODEL_SHA, "hf_repo": HF_REPO,
        "gate": ("the publisher's laya 0.3.4 model (transformers 5.17.0, SDPA attention), CPU fp32, each question row "
                 "alone (batch 1) right-padded to its window; answers from the frozen LiteRT-lane fixture: 201 rows per "
                 "window (37 choice, 44 score, 120 noul), official Agent.predict answers at T=1"),
        "oracle": {"status": oracle["status"], "record": record(results_dir() / "oracle.json", work, "<work>"),
                   "sequence_identity": {w: v["identical"] for w, v in oracle["sequence_identity"].items()},
                   "official_batched_rerun_exact_dicts": {w: v["exact_answer_dicts"] for w, v in oracle["official_batched_rerun"].items()},
                   "windows": {w: {"vs_fixture": pick(v["vs_fixture"], ("argmax_identical", "choice_score_rows", "max_probability_error")),
                                   "vs_litert_capture": pick(v["vs_litert_original"], ("max_marker_abs_error", "max_act_relative_error"))}
                               for w, v in oracle["windows"].items()},
                   "code_path_diagnostics": {k: {w: {kk: vv for kk, vv in x.items() if kk != "hidden_states"}
                                                 for w, x in v.items() if w in ("256", "512")}
                                             for k, v in oracle["code_path_diagnostics"].items()},
                   "environment": oracle["environment"]},
        "authoring": [], "export": [], "runtime_rows": [], "aot": [], "authored144": None,
    }
    transcript["fp16_recipe_torch_previews"] = []
    for path in sorted(results_dir().glob("authoring_*_s*.json")):
        a = json.loads(path.read_text())
        if a["precision"] == "fp16":
            # The fp16 recipe (and one variant with more fp32 islands) in torch: rows + per-state relative errors.
            # Evidence for why fp16 compute is not shipped; recorded before the wfp16 refactor of _laya_model.py,
            # and the recipe's row numbers were re-measured identically by its export gate.
            transcript["fp16_recipe_torch_previews"].append({
                "window": a["window"], "fp32_islands": a["fp32_islands"], "status": a["status"],
                "rows": pick(a["summary"], ("argmax_identical", "choice_score_rows", "max_probability_error",
                                            "max_marker_abs_error", "max_act_relative_error")),
                "max_relative_by_state": {n: st.get("max_relative_real") for n, st in a["layer_gate"]["per_state"].items()},
                "record": record(path, work, "<work>")})
            continue
        layer = a["layer_gate"]
        transcript["authoring"].append({
            "precision": a["precision"], "window": a["window"], "status": a["status"], "failures": a["failures"],
            "rows": pick(a["summary"], ("argmax_identical", "choice_score_rows", "max_probability_error",
                                        "max_act_probability_error", "max_marker_abs_error", "max_act_relative_error")),
            "layer_gate": {"status": layer["status"], "policy": layer["policy"], "max_absolute_tier": layer["max_absolute_tier"],
                           "max_relative_tier": layer["max_relative_tier"],
                           "per_state": {n: {k: s.get(k) for k in ("tier", "value", "bar", "max_abs_real", "max_relative_real",
                                                                  "relative_real_vs_eager", "official_sdpa_vs_eager_relative_real")}
                                         for n, s in layer["per_state"].items()}},
            "pad_isolation": a["pad_isolation"],
            "negative_controls": {m: {k: c[k] for k in ("caught", "caught_by", "first_failing_state", "value_at_first_failing_state",
                                                        "min_relative_tier_value", "max_absolute_tier_value")}
                                  for m, c in a["negative_controls"].items()},
            "record": record(path, work, "<work>"), "environment": a["environment"]})
    for manifest_path in sorted(exports.glob("*/*/provenance/export-manifest.json")):
        folder = manifest_path.parent.parent
        m = json.loads(manifest_path.read_text())
        entry = {"folder": str(folder.relative_to(exports)), "status": m["status"], "measure_only": m.get("measure_only", False),
                 "bundle": m["bundle"], "format": m["format"], "precision": m["precision"], "window": m["window"],
                 "bytes": m["bytes"], "torch_export_gate": m["torch_export_gate"],
                 "record": record(manifest_path, exports, "<exports>")}
        if "copied_from" in m:
            entry["copied_from"] = m["copied_from"]
        transcript["export"].append(entry)
        gate_path = folder / "provenance" / "runtime-gate.json"
        if gate_path.exists():
            for compute, r in json.loads(gate_path.read_text()).items():
                s = r["summary"]
                transcript["runtime_rows"].append({
                    "folder": entry["folder"], "compute": compute, "status": r["status"], "findings": r["findings"],
                    "measure_only_bundle": entry["measure_only"], "precision": r["precision"], "window": r["window"],
                    **pick(s, ("argmax_identical", "choice_score_rows", "max_probability_error", "max_act_probability_error",
                               "max_marker_abs_error", "max_act_relative_error")),
                    "repeat_max_abs": r["repeat_max_abs"], "wrong_pairing_control": r["wrong_pairing_control"],
                    "load_ms": r["load_ms"], "warm": {k: v for k, v in r["warm"].items() if k != "samples_ms"},
                    "gpu_lock": r.get("gpu_lock"), "device": {k: r["environment"][k] for k in ("machine", "macos_version", "macos_build", "coreai_build")},
                    "date": r["environment"]["date"], "record": record(gate_path, exports, "<exports>")})
    for manifest_path in sorted(exports.glob("*/*/provenance/aot-manifest.json")):
        m = json.loads(manifest_path.read_text())
        probe = m.get("neural_engine_probe") or {}
        transcript["aot"].append({"folder": str(manifest_path.parent.parent.relative_to(exports)), "bundle": m["bundle"],
                                  "preferred_compute": m["preferred_compute"], "seconds": m["seconds"], "bytes": m["bytes"],
                                  "ane_in_this_bundle": {k: m["ane"][k] for k in ("regions", "regions_listing_weights", "ir_bytes", "verdict")},
                                  "neural_engine_probe": {k: probe.get(k) for k in ("regions", "regions_listing_weights", "ir_bytes",
                                                                                     "verdict", "seconds")} if probe else None,
                                  "source": m["source"]["variant"], "record": record(manifest_path, exports, "<exports>")})
    transcript["repeat_runs"] = []
    for path in sorted(results_dir().glob("runtime_*_repeat.json")):
        for compute, r in json.loads(path.read_text()).items():
            transcript["repeat_runs"].append({
                "what": "a second run of the same bundle and compute unit, to see whether a failure reproduces",
                "bundle": r["bundle"], "compute": compute, "status": r["status"], "window": r["window"],
                **pick(r["summary"], ("argmax_identical", "choice_score_rows", "max_probability_error",
                                      "max_marker_abs_error", "max_act_relative_error")),
                "repeat_max_abs": r["repeat_max_abs"], "load_ms": r["load_ms"], "date": r["environment"]["date"],
                "record": record(path, work, "<work>")})
    authored = work / "authored144" / "authored144.json"
    if authored.exists():
        a = json.loads(authored.read_text())
        transcript["authored144"] = {"what": a["what"], "status": a["status"], "gold_sha256": a["gold_sha256"],
                                     "runs": {k: {kk: v[kk] for kk in ("mean_family_balanced_accuracy", "accuracy", "mean_family_macro_f1",
                                                                        "family_balanced_accuracy", "coverage")}
                                              for k, v in a["runs"].items()},
                                     "record": record(authored, work, "<work>")}
    out = repo_root() / "models" / "laya-multilingual" / "gate-laya-multilingual.json"
    out.write_text(json.dumps(transcript, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    print(out, len(transcript["runtime_rows"]), "runtime rows,", len(transcript["export"]), "folders,", len(transcript["aot"]), "AOT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
