#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.9.0", "safetensors>=0.7.0", "numpy>=2.2", "tokenizers>=0.22"]
# ///
"""Stage 1: the re-authored graph vs the oracle — every row, every saved hidden state, negative controls.

    python3 gate_julia_authoring.py --window 1024 --negative-controls
    python3 gate_julia_authoring.py --window 512 --negative-controls
    python3 gate_julia_authoring.py --window 1024 --dtype wfp16      # fp16 weight storage (rounded), fp32 compute

Row gate (every oracle row that fits the window: the 2,000 typed-decisions questions or the 1,965 that fit
512, the publisher's 100 parity requests, the window-filling rows): main graph -> token logits at the
marker positions, the deployed pipeline. Answer gate vs the publisher's engine (argmax on every row,
max |dp| <= 1e-3 at T = 1); tensor gate vs the publisher's model padded to the same window (marker
|d| <= 1e-3). The typed rows re-score the benchmark from this graph's argmax.

Layer gate (the SUBSET rows), two tiers against the publisher's SDPA path at real positions (LAYER_POLICY;
`--rejudge` re-applies the current bars to an existing record without recomputing it),
with the error against the publisher's model run with eager attention (the algorithm this graph authors)
and the publisher's own SDPA-vs-eager distance reported beside it. Pad positions are reported separately:
no output reads them, and a pad-isolation check shows real positions are bit-identical under random pad ids.

Negative controls (fp32): all_global, all_local, window63, ignore_padding, no_type_emb must each be
caught; the record says by which gate (layer, tensor, answer) and where.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (MODEL_SHA, environment, hashes, load_rows, oracle_dir, results_dir, row_inputs,  # noqa: E402
                     subset, verify_hashes, verify_source, write_json)
from _gate_metrics import POLICY, evaluate_row, summarize  # noqa: E402
from _julia_host import gather_markers  # noqa: E402
from _julia_model import MUTATIONS, load_julia  # noqa: E402

STATES = ["embeddings", *[f"layer_{i:02d}" for i in range(22)], "final_norm", "head_input", "head_0", "head_1"]
ABSOLUTE_BAR, RELATIVE_BAR = 1e-4, 5e-4
ABSOLUTE_STATES = STATES[:11]   # embeddings, layer_00 … layer_09
RELATIVE_STATES = STATES[11:]   # layer_10 … layer_21, final_norm, head_input, head_0, head_1
LAYER_POLICY = {
    "absolute": {"states": "embeddings, layer_00 … layer_09", "bar": "max |err| at real positions <= 1e-4"},
    "relative": {"states": "layer_10 … layer_21, final_norm, head_input, head_0, head_1",
                 "bar": "max |err| / max |ref| at the real positions of the same state and row <= 5e-4 (max over rows)"},
    "reference": "the publisher's SDPA path (FastEngine's specialized encoder), real positions; pad positions are reported, never read",
    "reason": ("The laya multilingual port's two-tier rule, re-measured for this checkpoint: an absolute 1e-4 where it "
               "still separates a correct graph from a wrong one (embeddings, layers 0-9), and above it a relative bar of "
               "twice the publisher's own SDPA-vs-eager distance, rounded down. That distance (results/oracle.json) is "
               "2.87e-4 at final_norm at S=1024 and 1.33e-4 at S=512, so 2 x 2.87e-4 = 5.74e-4 -> 5e-4. laya's 2e-4 came "
               "from its own 1.2e-4 and was first applied here unchanged: it failed this checkpoint at final_norm, where "
               "the publisher's eager path itself sits at 2.87e-4 (S=1024). From layer 11 a few dimensions carry ~4.5e3."),
    "residual": ("The residual that remains is the torch build's, not the graph's: under the publisher's torch 2.14.0 this "
                 "graph equals the publisher's eager path to 1.3e-7 relative at every state (results/exactness_s{S}.json); "
                 "under the zoo's torch 2.9.0 the encoder layers stay as close to SDPA as eager is (2.1e-5 at layer 21) and "
                 "the final LayerNorm, normalizing the ~4.5e3 dimensions, adds the rest (3.4e-4 at S=512)."),
    "decided": "J2 session 2026-09-30 09:4x, after the first run with laya's 2e-4 (records kept: rejudged.from_sha256)",
}
NEGATIVE_ROW_STRIDE = 13  # negative controls run on every 13th row plus the SUBSET rows


def tensors(x: dict[str, np.ndarray]) -> tuple[torch.Tensor, ...]:
    return tuple(torch.from_numpy(x[k]) for k in ("input_ids", "attention_mask", "qtype_onehot"))


def run_rows(model, rows, window) -> list[dict]:
    records = []
    for row in rows:
        t0 = time.perf_counter()
        with torch.inference_mode():
            token_logits = model(*tensors(row_inputs(row, window)))
        marker = gather_markers(token_logits[0].numpy(), row["markers"])
        record = evaluate_row(row, marker)
        record["ms"] = (time.perf_counter() - t0) * 1000
        records.append(record)
    return records


def layer_errors(model, rows, window, eager: bool = True) -> dict:
    """Per SUBSET row and saved state: max |err| at real positions vs SDPA (gated) and vs eager, the
    publisher's SDPA-vs-eager distance, the reference scale, and the pad-position error (informational)."""
    lookup = {r["row_id"]: r for r in rows}
    out = {}
    for index, row_id in enumerate(subset(window)):
        row = lookup[row_id]
        n = len(row["ids"])
        with torch.inference_mode():
            mine = model.forward_intermediates(*tensors(row_inputs(row, window)))
        with np.load(oracle_dir() / "hidden" / f"s{window}" / f"{index:03d}.npz") as data:
            sdpa = {k: data[k] for k in [*STATES, "token_logits"]}
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
                entry["relative_real_vs_eager"] = entry["max_abs_real_vs_eager"] / entry["reference_abs_max_real"]
                entry["official_sdpa_vs_eager_relative_real"] = (entry["official_sdpa_vs_eager_max_abs_real"]
                                                                 / entry["reference_abs_max_real"])
            per_state[name] = entry
        per_state["token_logits_real_max_abs"] = float(np.max(np.abs(
            mine["token_logits"][0].numpy()[:n].astype(np.float64) - sdpa["token_logits"][:n])))
        out[row_id] = per_state
    return out


def pad_isolation(model, rows, window) -> dict:
    """Replace every pad id with a random token: real positions must not change by a single bit."""
    generator = np.random.default_rng(0)
    lookup = {r["row_id"]: r for r in rows}
    checked, identical = 0, 0
    for row_id in subset(window):
        row = lookup[row_id]
        n = len(row["ids"])
        if n == window:
            continue
        x = row_inputs(row, window)
        noisy = {k: v.copy() for k, v in x.items()}
        noisy["input_ids"][0, n:] = generator.integers(5, 256000, size=window - n, dtype=np.int32)
        with torch.inference_mode():
            a = model.forward_intermediates(*tensors(x))
            b = model.forward_intermediates(*tensors(noisy))
        same = all(torch.equal(a[name][0, :n], b[name][0, :n]) for name in STATES)
        same &= torch.equal(a["token_logits"][0, :n], b["token_logits"][0, :n])
        checked += 1
        identical += bool(same)
    return {"rows_checked": checked, "rows_bit_identical_at_real_positions": identical,
            "status": "PASS" if checked and identical == checked else "FAIL"}


def summarize_layers(errors: dict) -> dict:
    """The two-tier layer gate (LAYER_POLICY) over the SUBSET rows."""
    per_state = {}
    for name in STATES:
        rows = errors.values()
        entry = {"tier": "absolute" if name in ABSOLUTE_STATES else "relative",
                 "max_abs_real": max(e[name]["max_abs_real"] for e in rows),
                 "max_relative_real": max(e[name]["max_abs_real"] / e[name]["reference_abs_max_real"] for e in rows),
                 "reference_abs_max_real": max(e[name]["reference_abs_max_real"] for e in rows),
                 "max_abs_pad": max((e[name].get("max_abs_pad", 0.0) for e in rows), default=0.0)}
        first = next(iter(errors.values()))[name]
        if "max_abs_real_vs_eager" in first:
            for key in ("max_abs_real_vs_eager", "official_sdpa_vs_eager_max_abs_real", "relative_real_vs_eager",
                        "official_sdpa_vs_eager_relative_real"):
                entry[key] = max(e[name][key] for e in rows)
        entry["value"] = entry["max_abs_real"] if entry["tier"] == "absolute" else entry["max_relative_real"]
        entry["bar"] = ABSOLUTE_BAR if entry["tier"] == "absolute" else RELATIVE_BAR
        entry["pass"] = entry["value"] <= entry["bar"]
        per_state[name] = entry
    failing = [name for name in STATES if not per_state[name]["pass"]]
    return {"policy": LAYER_POLICY, "status": "PASS" if not failing else "FAIL", "states_failing": failing,
            "max_absolute_tier": max(per_state[name]["value"] for name in ABSOLUTE_STATES),
            "max_relative_tier": max(per_state[name]["value"] for name in RELATIVE_STATES),
            "max_token_logits_real": max(e["token_logits_real_max_abs"] for e in errors.values()),
            "all_pad_positions_finite": all(e[name].get("finite_pad", True) for e in errors.values() for name in STATES),
            "per_state": per_state, "rows": errors}


def judge(summary, layers, isolation, controls, negative_controls_run) -> list[str]:
    failures = []
    if summary["answer_status"] != "PASS":
        failures.append("answer gate")
    if summary["tensor_status"] != "PASS":
        failures.append("tensor gate")
    if layers["status"] != "PASS":
        failures.append(f"layer gate: {layers['states_failing']}")
    if isolation["status"] != "PASS":
        failures.append("pad isolation")
    if negative_controls_run:
        missed = [m for m, c in controls.items() if not c["caught"]]
        if missed:
            failures.append(f"negative controls not caught: {missed}")
        if not controls["window63"]["caught_by"]["layer_gate"]:
            failures.append("window63 not caught by the layer gate")
    return failures


def rejudge(args) -> int:
    """The existing record under the current LAYER_POLICY: the per-row and per-state numbers are copied, only
    the bars (and so the verdicts) are applied again; the record names the file and policy it was computed with."""
    from _common import sha256_of
    path = results_dir() / f"authoring_{args.dtype}_s{args.window}.json"
    old = json.loads(path.read_text())
    old_sha, old_policy = sha256_of(path), old["layer_gate"]["policy"]
    layers = old["layer_gate"]
    for name in STATES:
        entry = layers["per_state"][name]
        entry["bar"] = ABSOLUTE_BAR if entry["tier"] == "absolute" else RELATIVE_BAR
        entry["pass"] = entry["value"] <= entry["bar"]
    failing = [name for name in STATES if not layers["per_state"][name]["pass"]]
    layers.update(policy=LAYER_POLICY, status="PASS" if not failing else "FAIL", states_failing=failing)
    for control in old["negative_controls"].values():
        values = control["per_state_value"]
        over = [n for n in STATES if values[n] > (ABSOLUTE_BAR if n in ABSOLUTE_STATES else RELATIVE_BAR)]
        control["caught_by"]["layer_gate"] = bool(over)
        control["caught"] = any(control["caught_by"].values())
        control.update(first_failing_state=over[0] if over else None,
                       value_at_first_failing_state=values[over[0]] if over else None, layer_states_failing=len(over))
    failures = judge(old["summary"], layers, old["pad_isolation"], old["negative_controls"], old["negative_controls_run"])
    source = verify_source()
    old.update(status="FAIL" if failures else "PASS", failures=failures,
               rejudged={"from_sha256": old_sha, "from_policy": old_policy, "computed_with_input_hashes": old["input_hashes"],
                         "environment": environment()},
               input_hashes=hashes([source / "model.safetensors", source / "encoder" / "config.json",
                                    source / "julia_config.json", oracle_dir() / "rows.json", Path(__file__),
                                    Path(__file__).parent / "_julia_model.py", Path(__file__).parent / "_julia_host.py",
                                    Path(__file__).parent / "_gate_metrics.py"]))
    write_json(path, old)
    print(old["status"], failures, f"layer relative max {layers['max_relative_tier']:.2e} (bar {RELATIVE_BAR:g});",
          {m: (c["caught"], c["first_failing_state"]) for m, c in old["negative_controls"].items()}, "->", path)
    return 0 if old["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", type=int, required=True, choices=[512, 1024])
    parser.add_argument("--dtype", choices=["fp32", "wfp16"], default="fp32",
                        help="wfp16 = fp16 weight storage (rounded from the F32 checkpoint), fp32 compute")
    parser.add_argument("--negative-controls", action="store_true")
    parser.add_argument("--rejudge", action="store_true",
                        help="re-apply the current bars to this window/dtype's existing record (rows and states unchanged)")
    args = parser.parse_args()
    if args.rejudge:
        return rejudge(args)
    torch.set_num_threads(8)
    started = time.perf_counter()
    source = verify_source()
    oracle_record = json.loads((results_dir() / "oracle.json").read_text())
    assert oracle_record["status"] == "PASS", "the oracle must pass first"
    rows_file = oracle_dir() / "rows.json"
    verify_hashes({str(rows_file): oracle_record["output_hashes"][str(rows_file)]})
    rows = load_rows(args.window)
    model = load_julia(source, args.window, precision=args.dtype)

    records = run_rows(model, rows, args.window)
    summary = summarize(records)
    print(f"rows {summary['rows']}: answer {summary['answer_status']} (argmax {summary['argmax_identical']}/{summary['rows']}, "
          f"max dp {summary['max_probability_error']:.2e}); tensor {summary['tensor_status']} "
          f"(marker {summary['max_marker_abs_error']:.2e}); typed {json.dumps(summary.get('typed_accuracy'))}", flush=True)
    layers = summarize_layers(layer_errors(model, rows, args.window))
    worst_abs = max(ABSOLUTE_STATES, key=lambda n: layers["per_state"][n]["value"])
    worst_rel = max(RELATIVE_STATES, key=lambda n: layers["per_state"][n]["value"])
    print(f"layers: {layers['status']} (absolute max {layers['per_state'][worst_abs]['value']:.2e} at {worst_abs}, "
          f"relative max {layers['per_state'][worst_rel]['value']:.2e} at {worst_rel}); failing {layers['states_failing']}",
          flush=True)
    isolation = pad_isolation(model, rows, args.window)
    print("pad isolation", isolation, flush=True)

    controls = {}
    if args.negative_controls:
        if args.dtype != "fp32":
            parser.error("negative controls run on the fp32 graph")
        keep = set(subset(args.window))
        control_rows = [r for i, r in enumerate(rows) if i % NEGATIVE_ROW_STRIDE == 0 or r["row_id"] in keep]
        for mutation in MUTATIONS[1:]:
            mutated = load_julia(source, args.window, mutation=mutation)
            m_summary = summarize(run_rows(mutated, control_rows, args.window))
            m_layers = summarize_layers(layer_errors(mutated, rows, args.window, eager=False))
            failing = m_layers["states_failing"]
            caught = {"layer_gate": bool(failing), "tensor_gate": m_summary["tensor_status"] == "FAIL",
                      "answer_gate": m_summary["answer_status"] == "FAIL"}
            controls[mutation] = {
                "caught": any(caught.values()), "caught_by": caught, "rows_checked": len(control_rows),
                "first_failing_state": failing[0] if failing else None,
                "value_at_first_failing_state": m_layers["per_state"][failing[0]]["value"] if failing else None,
                "layer_states_failing": len(failing),
                "min_relative_tier_value": min(m_layers["per_state"][n]["value"] for n in RELATIVE_STATES),
                "max_absolute_tier_value": max(m_layers["per_state"][n]["value"] for n in ABSOLUTE_STATES),
                "rows": {k: m_summary[k] for k in ("answer_status", "tensor_status", "argmax_identical", "rows",
                                                   "max_probability_error", "max_marker_abs_error")},
                "answer_failures": len(m_summary["answer_failures"]), "tensor_failures": len(m_summary["tensor_failures"]),
                "per_state_value": {name: m_layers["per_state"][name]["value"] for name in STATES}}
            print("NEGATIVE", mutation, json.dumps({k: controls[mutation][k] for k in (
                "caught_by", "first_failing_state", "value_at_first_failing_state", "layer_states_failing")}),
                  json.dumps(controls[mutation]["rows"]), flush=True)
            del mutated

    failures = judge(summary, layers, isolation, controls, args.negative_controls)
    result = {
        "status": "FAIL" if failures else "PASS", "failures": failures,
        "stage": "authoring", "model_sha": MODEL_SHA, "window": args.window, "precision": args.dtype,
        "policy": POLICY, "summary": summary, "layer_gate": layers, "pad_isolation": isolation,
        "negative_controls": controls, "negative_controls_run": bool(args.negative_controls),
        "environment": environment(), "seconds": time.perf_counter() - started,
        "input_hashes": hashes([source / "model.safetensors", source / "encoder" / "config.json",
                                source / "julia_config.json", rows_file, Path(__file__),
                                Path(__file__).parent / "_julia_model.py", Path(__file__).parent / "_julia_host.py",
                                Path(__file__).parent / "_gate_metrics.py"]),
        "rows": records,
    }
    out = results_dir() / f"authoring_{args.dtype}_s{args.window}.json"
    write_json(out, result)
    print(result["status"], failures, f"{result['seconds']:.0f} s ->", out, flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
