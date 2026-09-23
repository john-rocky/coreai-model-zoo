#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.9.0", "safetensors>=0.7.0", "numpy>=2.2"]
# ///
"""Stage 1: the re-authored graph vs the oracle — every row, every saved hidden state, negative controls.

    python3 gate_laya_authoring.py --window 256 --negative-controls
    python3 gate_laya_authoring.py --window 512 --negative-controls
    python3 gate_laya_authoring.py --window 256 --dtype wfp16         # fp16 storage, fp32 compute (rows + layers)
    python3 gate_laya_authoring.py --window 256 --dtype fp16          # the fp16 recipe in torch (rows only)

Row gate (all 201 rows of the window): main graph -> token logits at the marker positions -> the NumPy
host's act features -> act head, exactly the deployed pipeline. Answer gate vs the frozen fixture
(argmax on every choice/score row, max |dp| <= 1e-3 at T=1, |d act probability| <= 1e-3); tensor gate
vs the oracle's official batch-1 run at the same window (marker |d| <= 1e-3, act relative <= 1e-4).

Layer gate (fp32, the SUBSET rows): every saved hidden state (embeddings, 22 residual layers,
final_norm, after type_emb, both head layers) at real positions, max |err| <= 1e-4 against the official
SDPA path. Reported beside it for every row and state: the error against the official model run with
eager attention (the same algorithm this graph authors), and the official model's own SDPA-vs-eager
distance — from layer 11 on, the CLS position carries a ~1.4e4 activation (one fp32 ulp ~1e-3) and
that distance alone exceeds 1e-4, so the bar is reported as measured, not moved. Pad positions are
reported separately: no output reads them (every attention masks pad keys), the official SDPA path
zeroes fully masked sliding rows while this graph's finite mask averages them, and a pad-isolation
check shows real positions are bit-identical when the pad ids are replaced by random tokens.

Negative controls (fp32): all_global, all_local, window63, ignore_padding, no_type_emb must each be
caught; the record says by which gate (layer gate inside the states where the clean graph is within
1e-4, tensor gate, answer gate).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (MODEL_SHA, SUBSET, environment, hashes, load_rows, oracle_dir, results_dir,  # noqa: E402
                     row_inputs, verify_hashes, verify_source, write_json)
from _gate_metrics import POLICY, evaluate_row, summarize  # noqa: E402
from _laya_host import act_features, gather_markers  # noqa: E402
from _laya_model import FP16_RECIPE, FP32_ISLANDS, MUTATIONS, load_laya  # noqa: E402

LAYER_BAR = 1e-4
STATES = ["embeddings", *[f"layer_{i:02d}" for i in range(22)], "final_norm", "head_input", "head_0", "head_1"]


def decode_config(source: Path) -> dict:
    agent = json.loads((source / "rl_agent_config.json").read_text())
    assert agent["temperature"] == [1.0, 1.0, 1.0] and not agent["temperature_by_options"], "fixture temperature is T=1"
    return {"temperature": agent["temperature"], "temperature_by_options": agent["temperature_by_options"]}


def tensors(x: dict[str, np.ndarray]) -> tuple[torch.Tensor, ...]:
    return tuple(torch.from_numpy(x[k]) for k in ("input_ids", "attention_mask", "qtype_onehot"))


def run_rows(model, rows, window, config, oracle) -> list[dict]:
    records = []
    for index, row in enumerate(rows):
        t0 = time.perf_counter()
        with torch.inference_mode():
            token_logits, pooled = model(*tensors(row_inputs(row, window)))
            marker = gather_markers(token_logits[0].numpy(), row["marker_positions"])
            feats = act_features(marker)
            act = model.act(pooled, torch.from_numpy(feats))[0].numpy()
        ms = (time.perf_counter() - t0) * 1000
        k, n = row["K"], row["sequence_length"]
        record = evaluate_row(row, marker, act, config, tensor_reference={
            "marker_logits": oracle["marker_logits"][index, :k], "act_logits": oracle["act_logits"][index]})
        record.update(
            ms=ms,
            pooled_cls_max_abs_vs_oracle=float(np.max(np.abs(pooled[0].numpy().astype(np.float64) - oracle["pooled_cls"][index]))),
            feats_max_abs_vs_oracle=float(np.max(np.abs(feats[0].astype(np.float64) - oracle["feats"][index]))),
            token_logits_real_max_abs_vs_oracle=float(np.max(np.abs(
                token_logits[0].numpy()[:n].astype(np.float64) - oracle["token_logits"][index, :n]))))
        records.append(record)
    return records


def layer_errors(model, rows, window, eager: bool = True) -> dict:
    """Per SUBSET row and saved state: max |err| at real positions vs SDPA (gated) and vs eager, the
    official SDPA-vs-eager distance, the reference scale, and the pad-position error (informational)."""
    out = {}
    for index in SUBSET:
        row = rows[index]
        n = row["sequence_length"]
        with torch.inference_mode():
            mine = model.forward_intermediates(*tensors(row_inputs(row, window)))
        with np.load(oracle_dir() / "hidden" / f"s{window}" / f"{index:03d}.npz") as data:
            sdpa = {k: data[k] for k in STATES}
        eager_states = {}
        if eager:
            with np.load(oracle_dir() / "hidden_eager" / f"s{window}" / f"{index:03d}.npz") as data:
                eager_states = {k: data[k] for k in STATES}
        per_state = {}
        for name in STATES:
            a = mine[name][0].float().numpy().astype(np.float64)
            entry = {"max_abs_real": float(np.max(np.abs(a[:n] - sdpa[name][:n]))),
                     "reference_abs_max_real": float(np.max(np.abs(sdpa[name][:n])))}
            if n < window:
                entry["max_abs_pad"] = float(np.max(np.abs(a[n:] - sdpa[name][n:])))
                entry["finite_pad"] = bool(np.isfinite(a[n:]).all())
            if eager:
                entry["max_abs_real_vs_eager"] = float(np.max(np.abs(a[:n] - eager_states[name][:n])))
                entry["official_sdpa_vs_eager_max_abs_real"] = float(np.max(np.abs(
                    sdpa[name][:n].astype(np.float64) - eager_states[name][:n])))
            per_state[name] = entry
        out[row["row_id"]] = per_state
    return out


def pad_isolation(model, rows, window) -> dict:
    """Replace every pad id with a random token: real positions must not change by a single bit."""
    generator = np.random.default_rng(0)
    checked, identical = 0, 0
    for index in SUBSET:
        row = rows[index]
        n = row["sequence_length"]
        if n == window:
            continue
        x = row_inputs(row, window)
        noisy = {k: v.copy() for k, v in x.items()}
        noisy["input_ids"][0, n:] = generator.integers(5, 256000, size=window - n, dtype=np.int32)
        with torch.inference_mode():
            a = model.forward_intermediates(*tensors(x))
            b = model.forward_intermediates(*tensors(noisy))
        same = all(torch.equal(a[name][0, :n], b[name][0, :n]) for name in STATES)
        same &= torch.equal(a["pooled_cls"], b["pooled_cls"]) and torch.equal(a["token_logits"][0, :n], b["token_logits"][0, :n])
        checked += 1
        identical += bool(same)
    return {"rows_checked": checked, "rows_bit_identical_at_real_positions": identical,
            "status": "PASS" if checked and identical == checked else "FAIL"}


def summarize_layers(errors: dict) -> dict:
    per_state = {name: {"max_abs_real": max(e[name]["max_abs_real"] for e in errors.values()),
                        "max_relative_real": max(e[name]["max_abs_real"] / e[name]["reference_abs_max_real"] for e in errors.values()),
                        "max_abs_pad": max((e[name].get("max_abs_pad", 0.0) for e in errors.values()), default=0.0)}
                 for name in STATES}
    for name in STATES:
        first = next(iter(errors.values()))[name]
        if "max_abs_real_vs_eager" in first:
            per_state[name]["max_abs_real_vs_eager"] = max(e[name]["max_abs_real_vs_eager"] for e in errors.values())
            per_state[name]["official_sdpa_vs_eager_max_abs_real"] = max(
                e[name]["official_sdpa_vs_eager_max_abs_real"] for e in errors.values())
    failures = [(row_id, name) for row_id, e in errors.items() for name in STATES if e[name]["max_abs_real"] > LAYER_BAR]
    clean = [name for name in STATES if per_state[name]["max_abs_real"] <= LAYER_BAR]
    return {"bar_max_abs_real": LAYER_BAR, "status": "PASS" if not failures else "FAIL",
            "states_over_bar": [name for name in STATES if per_state[name]["max_abs_real"] > LAYER_BAR],
            "failing_row_state_pairs": len(failures), "states_within_bar": clean,
            "all_pad_positions_finite": all(e[name].get("finite_pad", True) for e in errors.values() for name in STATES),
            "per_state": per_state}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", type=int, required=True, choices=[256, 512])
    parser.add_argument("--dtype", choices=["fp32", "fp16", "wfp16"], default="fp32",
                        help="wfp16 = fp16 weight storage, fp32 compute")
    parser.add_argument("--fp32-islands", default=",".join(FP16_RECIPE),
                        help=f"fp16 only: comma list from {FP32_ISLANDS} (default: the recipe {FP16_RECIPE})")
    parser.add_argument("--negative-controls", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.perf_counter()
    source = verify_source()
    oracle_record = json.loads((results_dir() / "oracle.json").read_text())
    assert oracle_record["status"] == "PASS", "the oracle must pass first"
    outputs = oracle_dir() / f"outputs_s{args.window}.npz"
    verify_hashes({str(outputs): oracle_record["output_hashes"][str(outputs)]})
    with np.load(outputs) as data:
        oracle = {k: data[k] for k in data.files}
    rows = load_rows(args.window)
    config = decode_config(source)
    islands = tuple(sorted(filter(None, args.fp32_islands.split(",")))) if args.dtype == "fp16" else FP32_ISLANDS
    model = load_laya(source, args.window, precision=args.dtype, fp32_islands=islands)

    records = run_rows(model, rows, args.window, config, oracle)
    summary = summarize(records)
    print(f"rows: answer {summary['answer_status']} (argmax {summary['argmax_identical']}/{summary['choice_score_rows']}, "
          f"max dp {summary['max_probability_error']:.2e}, act p {summary['max_act_probability_error']:.2e}); "
          f"tensor {summary['tensor_status']} (marker {summary['max_marker_abs_error']:.2e}, "
          f"act rel {summary['max_act_relative_error']:.2e})", flush=True)
    layers = summarize_layers(layer_errors(model, rows, args.window))
    print(f"layers: {layers['status']} at the {LAYER_BAR:g} bar; over the bar: {layers['states_over_bar']}", flush=True)
    isolation = pad_isolation(model, rows, args.window) if args.dtype != "fp16" else None
    if isolation:
        print("pad isolation", isolation, flush=True)

    controls = {}
    if args.negative_controls:
        if args.dtype != "fp32":
            parser.error("negative controls run on the fp32 graph")
        for mutation in MUTATIONS[1:]:
            mutated = load_laya(source, args.window, mutation=mutation)
            m_records = run_rows(mutated, rows, args.window, config, oracle)
            m_summary = summarize(m_records)
            m_layers = summarize_layers(layer_errors(mutated, rows, args.window, eager=False))
            caught_by_layers = [name for name in layers["states_within_bar"] if m_layers["per_state"][name]["max_abs_real"] > LAYER_BAR]
            caught = {"layer_gate": bool(caught_by_layers), "tensor_gate": m_summary["tensor_status"] == "FAIL",
                      "answer_gate": m_summary["answer_status"] == "FAIL"}
            controls[mutation] = {
                "caught": any(caught.values()), "caught_by": caught,
                "first_state_over_bar_where_clean_is_within": caught_by_layers[0] if caught_by_layers else None,
                "layer_error_at_that_state": m_layers["per_state"][caught_by_layers[0]]["max_abs_real"] if caught_by_layers else None,
                "rows": {k: m_summary[k] for k in ("answer_status", "tensor_status", "argmax_identical", "choice_score_rows",
                                                   "max_probability_error", "max_marker_abs_error", "max_act_relative_error")},
                "answer_failures": len(m_summary["answer_failures"]), "tensor_failures": len(m_summary["tensor_failures"]),
                "per_state_max_abs_real": {name: m_layers["per_state"][name]["max_abs_real"] for name in STATES}}
            print("NEGATIVE", mutation, json.dumps({k: controls[mutation][k] for k in ("caught_by", "first_state_over_bar_where_clean_is_within")}), flush=True)
            del mutated

    failures = []
    if summary["answer_status"] != "PASS":
        failures.append("answer gate")
    if summary["tensor_status"] != "PASS":
        failures.append("tensor gate")
    if args.dtype != "fp16":
        if layers["status"] != "PASS":
            failures.append(f"layer gate: {len(layers['states_over_bar'])} states over {LAYER_BAR:g} ({layers['states_over_bar'][0]} …)")
        if isolation["status"] != "PASS":
            failures.append("pad isolation")
        if args.negative_controls:
            missed = [m for m, c in controls.items() if not c["caught"]]
            if missed:
                failures.append(f"negative controls not caught: {missed}")
            if not controls["window63"]["caught_by"]["layer_gate"]:
                failures.append("window63 not caught by the layer gate")
    result = {
        "status": "FAIL" if failures else "PASS", "failures": failures,
        "status_excluding_layer_gate": "PASS" if not [f for f in failures if not f.startswith("layer gate")] else "FAIL",
        "stage": "authoring", "model_sha": MODEL_SHA, "window": args.window, "precision": args.dtype,
        "fp32_islands": list(islands), "policy": POLICY, "summary": summary, "layer_gate": layers,
        "pad_isolation": isolation, "negative_controls": controls,
        "negative_controls_run": bool(args.negative_controls),
        "environment": environment(), "seconds": time.perf_counter() - started,
        "input_hashes": hashes([source / "model.safetensors", source / "encoder" / "config.json",
                                source / "rl_agent_config.json", outputs, Path(__file__), Path(__file__).parent / "_laya_model.py",
                                Path(__file__).parent / "_laya_host.py", Path(__file__).parent / "_gate_metrics.py"]),
        "rows": records,
    }
    tag = "" if args.dtype != "fp16" or islands == tuple(sorted(FP16_RECIPE)) else "_" + "-".join(islands)
    out = results_dir() / f"authoring_{args.dtype}{tag}_s{args.window}.json"
    write_json(out, result)
    print(result["status"], failures, f"{result['seconds']:.0f} s ->", out, flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
