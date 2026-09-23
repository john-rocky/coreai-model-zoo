#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["coreai-core==1.0.0b2", "numpy>=2.2"]
# ///
"""Stage 4 (Mac): the exported bundle on the Core AI runtime, per compute unit, vs the same 201 rows.

    python3 gate_laya_runtime.py <exports>/laya-multilingual/macos/fp32-s256 --compute cpu_only
    python3 gate_laya_runtime.py <dir> --compute gpu              # takes the advisory GPU lock only when quiet
    python3 gate_laya_runtime.py <dir> --compute neural_engine

The specialization options are always explicit (cpu_only(), or a preferred GPU / Neural Engine kind),
never None. Refuses an iOS-targeted folder: an iPhone bundle is never loaded on a Mac.

Per row: `main` -> token logits at the marker positions -> the NumPy host's act features -> `act`,
exactly the deployed pipeline, run 1 + --repeats times. Recorded unrounded per row: the raw marker
logits, the act logits, the repeat drift and this row's warm milliseconds (main + host + act).
Gate: argmax on every choice/score row, max |dp| <= 1e-3 at T=1 and |d act probability| <= 1e-3 vs
the frozen official answers; repeat drift <= 1e-6 on cpu_only (reported on the accelerators); a
wrong-pairing control (each row judged against another same-shape row's outputs) must FAIL. The
tensor bar vs the official batch-1 run (marker |d| <= 1e-3, act relative <= 1e-4) is reported; a miss
is recorded as a finding, not hidden and not re-tuned. Timing: load ms and >= 30 warm samples of one
whole question (main + act). A number measured while another GPU job runs is not recorded.

Results merge into <variant>/provenance/runtime-gate.json under the compute unit's key.
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import coreai.runtime as rt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_SHA, environment, sha256_of, write_json  # noqa: E402
from _gate_metrics import POLICY, evaluate_row, summarize, wrong_pairing  # noqa: E402
from _laya_host import act_features, gather_markers  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import gpu_lock  # noqa: E402

DRIFT_BAR = 1e-6


def options_for(compute: str):
    if compute == "cpu_only":
        return rt.SpecializationOptions.cpu_only()
    kind = rt.ComputeUnitKind.gpu() if compute == "gpu" else rt.ComputeUnitKind.neural_engine()
    return rt.SpecializationOptions.from_preferred_compute_unit_kind(kind)


def other_gpu_jobs() -> list[str]:
    listing = subprocess.run(["pgrep", "-fl", r"readout_gate|coreai_gate|gate_.*\.py|export_.*\.py"],
                             capture_output=True, text=True).stdout.splitlines()
    return [line for line in listing if str(os.getpid()) != line.split()[0] and "claude" not in line]


def row_arrays(row: dict, window: int) -> dict:
    n = row["sequence_length"]
    ids = np.zeros((1, window), dtype=np.int32)
    ids[0, :n] = row["sequence_ids"]
    mask = np.zeros((1, window), dtype=np.int32)
    mask[0, :n] = 1
    onehot = np.zeros((1, 3), dtype=np.float32)
    onehot[0, row["qtype"]] = 1.0
    return {"input_ids": ids, "attention_mask": mask, "qtype_onehot": onehot}


async def question(main_fn, act_fn, arrays: dict, markers) -> tuple[np.ndarray, np.ndarray, float]:
    """One whole question: main, the host's gather + act features, act. Returns copies and wall ms."""
    t0 = time.perf_counter()
    out = await main_fn({k: rt.NDArray(v) for k, v in arrays.items()})
    token_logits = np.array(out["token_logits"].numpy(), copy=True)
    pooled = np.array(out["pooled_cls"].numpy(), copy=True)
    marker = gather_markers(token_logits, markers)
    feats = act_features(marker)
    act = await act_fn({"pooled_cls": rt.NDArray(pooled), "feats": rt.NDArray(feats)})
    act_logits = np.array(act["act_logits"].numpy(), copy=True).reshape(-1)
    return marker, act_logits, (time.perf_counter() - t0) * 1000


async def run(variant_dir: Path, compute: str, repeats: int, warm_samples: int) -> dict:
    manifest = json.loads((variant_dir / "provenance" / "export-manifest.json").read_text())
    assert manifest["intended_runtime"] == "macos" and not manifest["aot"], "Mac gate only: never load an iOS bundle here"
    bundle = variant_dir / manifest["bundle"]
    assert bundle.suffix == ".aimodel", bundle
    for item in manifest["files"]:
        file = bundle / item["path"]
        assert file.stat().st_size == item["bytes"] and sha256_of(file) == item["sha256"], f"artifact changed: {item['path']}"
    reference = json.loads((variant_dir / "reference.json").read_text())
    assert reference["model_sha"] == manifest["model_sha"] == MODEL_SHA and reference["window"] == manifest["window"]
    window, rows = reference["window"], reference["rows"]
    for row in rows:
        row["window"] = window
    config = {"temperature": reference["temperature"], "temperature_by_options": reference["temperature_by_options"]}
    options = options_for(compute)

    start = time.perf_counter()
    model = await rt.AIModel.load(bundle, options)
    load_ms = (time.perf_counter() - start) * 1000
    assert sorted(model.function_names) == ["act", "main"], model.function_names
    main_fn, act_fn = model.load_function("main"), model.load_function("act")
    first_call_ms = None
    for _ in range(5):  # specialization and first-call work happen here, outside every recorded row
        _, _, ms = await question(main_fn, act_fn, row_arrays(rows[0], window), rows[0]["marker_positions"])
        first_call_ms = ms if first_call_ms is None else first_call_ms

    records, candidates = [], {}
    for row in rows:
        arrays = row_arrays(row, window)
        results = [await question(main_fn, act_fn, arrays, row["marker_positions"]) for _ in range(1 + repeats)]
        marker, act, _ = results[0]
        drift = max((max(float(np.max(np.abs(m - marker))), float(np.max(np.abs(a - act)))) for m, a, _ in results[1:]),
                    default=0.0)
        record = evaluate_row(row, marker, act, config, tensor_reference=row["oracle"])
        record.update(ms=float(np.median([ms for _, _, ms in results])), repeat_max_abs=drift)
        records.append(record)
        candidates[row["row_id"]] = {"marker_logits": marker, "act_logits": act}
    samples = []
    for i in range(warm_samples):
        row = rows[(i * 37) % len(rows)]
        samples.append((await question(main_fn, act_fn, row_arrays(row, window), row["marker_positions"]))[2])
    assert model is not None  # keep the model alive until every output is copied

    summary = summarize(records)
    control = wrong_pairing(rows, candidates, config)
    max_drift = max(r["repeat_max_abs"] for r in records)
    failures, findings = [], []
    if summary["answer_status"] != "PASS":
        failures.append("answer gate")
    if compute == "cpu_only" and max_drift > DRIFT_BAR:
        failures.append(f"non-deterministic on cpu_only (max repeat drift {max_drift:.3g})")
    if control["status"] != "FAIL":
        failures.append("wrong-pairing control passed")
    if summary.get("tensor_status") != "PASS":
        findings.append(f"tensor bar missed: marker {summary['max_marker_abs_error']:.3g} (bar 1e-3), "
                        f"act relative {summary['max_act_relative_error']:.3g} (bar 1e-4)")
    return {
        "status": "FAIL" if failures else "PASS", "failures": failures, "findings": findings,
        "stage": "runtime (Mac)", "compute": compute, "specialization_options": str(options).strip(),
        "bundle": manifest["bundle"], "precision": manifest["precision"], "window": window,
        "model_sha": MODEL_SHA, "policy": POLICY, "drift_bar_cpu_only": DRIFT_BAR, "summary": summary,
        "repeat_max_abs": max_drift, "repeats_per_row": repeats, "wrong_pairing_control": control,
        "load_ms": load_ms, "first_question_ms": first_call_ms,
        "warm": {"samples": len(samples), "median_ms": float(np.median(samples)), "p10_ms": float(np.percentile(samples, 10)),
                 "p90_ms": float(np.percentile(samples, 90)), "min_ms": min(samples), "max_ms": max(samples),
                 "what": "one question: main + host gather/act features + act, NumPy I/O included",
                 "note": "cpu_only is the parity option; its time is a reference value, not a speed claim" if compute == "cpu_only" else None,
                 "samples_ms": samples},
        "row_ms_median": float(np.median([r["ms"] for r in records])),
        "environment": environment(("coreai-core", "numpy")), "process_pid": os.getpid(),
        "rows": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("variant_dir", type=Path, help="an exported macos/<dtype>-s<S> directory")
    parser.add_argument("--compute", choices=["cpu_only", "gpu", "neural_engine"], required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warm-samples", type=int, default=40)
    args = parser.parse_args()
    if args.warm_samples < 30:
        parser.error("at least 30 warm samples")
    lock = None
    if args.compute != "cpu_only":
        if gpu_lock().exists():
            raise SystemExit(f"{gpu_lock()} exists: another lane holds the GPU; not taking it and waiting")
        others = other_gpu_jobs()
        if others:
            raise SystemExit(f"other GPU jobs are running, a timing now would not be a number: {others}")
        lock = gpu_lock()
        lock.write_text(f"laya runtime gate pid {os.getpid()} {args.variant_dir} {args.compute}\n")
    try:
        result = asyncio.run(run(args.variant_dir, args.compute, args.repeats, args.warm_samples))
    finally:
        if lock is not None:
            lock.unlink(missing_ok=True)
    out = args.variant_dir / "provenance" / "runtime-gate.json"
    merged = json.loads(out.read_text()) if out.exists() else {}
    merged[args.compute] = result
    write_json(out, merged)
    s = result["summary"]
    print(result["status"], args.compute, f"argmax {s['argmax_identical']}/{s['choice_score_rows']}",
          f"max dp {s['max_probability_error']:.2e}", f"act p {s['max_act_probability_error']:.2e}",
          f"marker {s['max_marker_abs_error']:.2e}", f"act rel {s['max_act_relative_error']:.2e}",
          f"drift {result['repeat_max_abs']:.1e}", f"control {result['wrong_pairing_control']['status']}",
          f"load {result['load_ms']:.0f} ms", f"warm median {result['warm']['median_ms']:.2f} ms",
          "failures", result["failures"], "findings", result["findings"])
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
