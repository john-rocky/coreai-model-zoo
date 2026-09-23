#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch==2.12.1",
#     "transformers==5.17.0",
#     "laya==0.3.4",
#     "safetensors==0.8.0",
#     "numpy==2.5.3",
#     "tokenizers==0.23.2",
#     "huggingface-hub==1.32.0",
# ]
# ///
"""The publisher's own answers on SemIf authored144 — the reference a port's decisions are compared with.

    uv run --python 3.12 conversion/laya/oracle_authored144_laya.py

Each authored144 row (id, state, question, three options, integer label) becomes one choice question,
criteria = {option.id: option.description}, answered by `laya.load(<pinned snapshot>, subfolder=
"multilingual").predict(state, {...})` with max_len = 256 and = 512 (head_max_len 256), CPU fp32, at two
temperatures: T = 1 (the checkpoint's config) and the fitted calibration of the LiteRT lane
(laya_ml_calibration.json set as agent.temperature / agent.temperature_by_options).

The package rounds its probabilities to four decimals; the raw marker logits and act logits are captured
from the model call, and the unrounded probabilities are recomputed from them with the package's own
arithmetic (bucket first, then per type; checked against the rounded dictionary on every row). Output per
window and temperature, in <work>/laya-multilingual/authored144/:

  laya-multilingual.s{S}.{t1|cal}.jsonl         {"id", "option_ids", "probabilities", "raw_marker_logits",
                                                 "act_logits", "official_answer", ...} per row — what
                                                 `decide-cli oracle --reference` and SemIf's evaluate.py read
  laya-multilingual.s{S}.{t1|cal}.report.json   SemIf evaluate.py on it (mean_family_balanced_accuracy …)
  authored144.json                              what was run, versions, hashes, the four headline numbers
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (HEAD_MAX_LEN, MODEL_ID, MODEL_SHA, SOURCE_SHA256, SUBFOLDER, WINDOWS, environment,  # noqa: E402
                     hashes, load_calibration, sha256_of, snapshot_root, verify_source, work_dir, write_json)
from _laya_host import probabilities as host_probabilities, softmax, temperature  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import code_path  # noqa: E402

SEMIF = code_path("codex-conversions", "2026-09-21", "semif-ondevice")  # the SemIf run owns these files; hashes are recorded


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold", type=Path, default=SEMIF / "fixtures" / "authored144.jsonl")
    parser.add_argument("--evaluate", type=Path, default=SEMIF / "semif" / "benchmarks" / "evaluate.py")
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.perf_counter()
    source = verify_source()
    import laya
    from laya.common import build_sequence

    agent = laya.load(str(snapshot_root()), subfolder=SUBFOLDER, device="cpu")
    assert sha256_of(source / "tokenizer" / "tokenizer_config.json") == SOURCE_SHA256["tokenizer/tokenizer_config.json"]
    agent.model.float().eval().requires_grad_(False)
    source_temperature = (list(agent.temperature), dict(agent.temperature_by_options))
    assert source_temperature == ([1.0, 1.0, 1.0], {}), source_temperature
    calibration = load_calibration()
    gold = [json.loads(line) for line in args.gold.read_text().splitlines() if line.strip()]
    assert len(gold) == 144 and len({r["id"] for r in gold}) == 144

    captured = []
    hook = agent.model.register_forward_hook(
        lambda module, inputs, output: captured.append((output[0].detach().clone(), output[1].detach().clone())))
    out_dir = work_dir() / "authored144"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs, files = {}, []
    try:
        for window in WINDOWS:
            agent.cfg["max_len"] = window
            for tag, (t_default, t_buckets) in {"t1": source_temperature,
                                                "cal": (calibration["temperature"], calibration["temperature_by_options"])}.items():
                agent.temperature, agent.temperature_by_options = list(t_default), dict(t_buckets)
                config = {"temperature": list(t_default), "temperature_by_options": dict(t_buckets)}
                rows, mismatched_rounding = [], []
                for item in gold:
                    option_ids = [o["id"] for o in item["options"]]
                    question = {"type": "choice", "instructions": item["question"],
                                "criteria": {o["id"]: o["description"] for o in item["options"]}}
                    captured.clear()
                    t0 = time.perf_counter()
                    result = agent.predict(item["state"], {"q": question})
                    ms = (time.perf_counter() - t0) * 1000
                    assert len(captured) == 1
                    logits, act = captured[0]
                    k = len(option_ids)
                    raw = logits[0, :k].numpy().astype(np.float32)
                    act_logits = act[0].numpy().astype(np.float32)
                    p = host_probabilities(raw, "choice", config)
                    answer = result["answers"]["q"]
                    if [round(float(v), 4) for v in p] != [answer["probabilities"][i] for i in option_ids]:
                        mismatched_rounding.append(item["id"])
                    ids, markers = build_sequence(agent.tok, item["state"], agent._to_internal(question),
                                                  max_len=window, head_max_len=HEAD_MAX_LEN)
                    rows.append({"id": item["id"], "option_ids": option_ids, "probabilities": [float(v) for v in p],
                                 "raw_marker_logits": [float(v) for v in raw], "act_logits": [float(v) for v in act_logits],
                                 "act_probability": float(softmax(act_logits)[0]),
                                 "temperature": temperature("choice", k, config), "window": window,
                                 "sequence_length": len(ids), "marker_positions": markers, "official_answer": answer,
                                 "family": item["family"], "label": item["label"], "ms": ms})
                name = f"laya-multilingual.s{window}.{tag}"
                predictions = out_dir / f"{name}.jsonl"
                predictions.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
                report = out_dir / f"{name}.report.json"
                report.unlink(missing_ok=True)  # evaluate.py opens its output with mode "x"
                subprocess.run([sys.executable, str(args.evaluate), "--gold", str(args.gold), "--predictions", str(predictions),
                                "--output", str(report)], check=True, capture_output=True, text=True)
                evaluated = json.loads(report.read_text())
                runs[name] = {"window": window, "temperature": config,
                              "mean_family_balanced_accuracy": evaluated["mean_family_balanced_accuracy"],
                              "mean_family_macro_f1": evaluated["mean_family_macro_f1"],
                              "family_balanced_accuracy": {f: r["balanced_accuracy"] for f, r in evaluated["family_results"].items()},
                              "accuracy": sum(fr["accuracy"] * fr["n"] for fr in evaluated["family_results"].values())
                                          / sum(fr["n"] for fr in evaluated["family_results"].values()),
                              "coverage": evaluated["coverage"], "invalid": evaluated["invalid"], "missing": evaluated["missing"],
                              "rows_whose_rounded_probabilities_differ_from_the_package": mismatched_rounding,
                              "max_sequence_length": max(r["sequence_length"] for r in rows),
                              "median_ms": float(np.median([r["ms"] for r in rows]))}
                files += [predictions, report]
                print(f"{name}: mean family balanced accuracy {evaluated['mean_family_balanced_accuracy']:.4f} "
                      f"(coverage {evaluated['coverage']}, rounding mismatches {len(mismatched_rounding)})", flush=True)
    finally:
        hook.remove()
    record = {"status": "PASS" if all(not r["rows_whose_rounded_probabilities_differ_from_the_package"] and r["coverage"] == 1
                                      for r in runs.values()) else "CHECK",
              "what": "the publisher's laya 0.3.4 model on SemIf authored144, one choice question per row",
              "model": MODEL_ID, "model_sha": MODEL_SHA, "subfolder": SUBFOLDER, "head_max_len": HEAD_MAX_LEN,
              "gold": str(args.gold), "gold_sha256": sha256_of(args.gold), "evaluate": str(args.evaluate),
              "evaluate_sha256": sha256_of(args.evaluate), "runs": runs, "environment": environment(
                  ("torch", "transformers", "laya", "tokenizers", "numpy")), "threads": 4,
              "output_hashes": hashes(files), "seconds": time.perf_counter() - started}
    write_json(out_dir / "authored144.json", record)
    print(record["status"], f"{record['seconds']:.0f} s", flush=True)
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
