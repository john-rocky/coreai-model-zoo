#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.0", "safetensors==0.8.0", "numpy==2.5.3"]
# ///
"""Stage 1a: the re-authored graph under the publisher's own torch — is it the eager algorithm, exactly?

    uv run --python 3.12 conversion/julia/gate_julia_exactness.py --window 1024
    uv run --python 3.12 conversion/julia/gate_julia_exactness.py --window 512

Runs `_julia_model.py` (fp32) in the oracle's environment (torch 2.14.0, the publisher's pin) on the SUBSET
rows and compares every hidden state with the oracle's two dumps of the publisher's model: its eager-attention
path (the matmul-softmax-matmul algorithm this graph writes out) and its SDPA path (what its engine runs).
Bar: against eager, max |err| / max |ref| <= 1e-6 at every state (the head's fused eval path is the only
kernel that differs) and marker logits <= 1e-5 absolute. The SDPA distance is reported beside it.

Why this stage exists: under the zoo's torch 2.9.0 (stage 1) the same graph sits farther from the SDPA
reference than the publisher's own eager path does, and the only change between the two runs is the torch
build. This record shows the graph itself is exact, so stage 1's residual is the kernels', not the graph's.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (MODEL_SHA, environment, hashes, load_rows, oracle_dir, results_dir, row_inputs, subset,  # noqa: E402
                     verify_source, write_json)
from _julia_host import gather_markers  # noqa: E402
from _julia_model import load_julia  # noqa: E402

STATES = ["embeddings", *[f"layer_{i:02d}" for i in range(22)], "final_norm", "head_input", "head_0", "head_1"]
RELATIVE_BAR, MARKER_BAR = 1e-6, 1e-5


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", type=int, required=True, choices=[512, 1024])
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.perf_counter()
    source = verify_source()
    model = load_julia(source, args.window)
    rows = {r["row_id"]: r for r in load_rows(args.window)}
    per_state = {name: {"vs_eager_relative": 0.0, "vs_sdpa_relative": 0.0, "sdpa_vs_eager_relative": 0.0} for name in STATES}
    markers = {"vs_eager": 0.0, "vs_sdpa_padded": 0.0}
    for index, row_id in enumerate(subset(args.window)):
        row = rows[row_id]
        n = len(row["ids"])
        x = row_inputs(row, args.window)
        with torch.inference_mode():
            mine = model.forward_intermediates(*(torch.from_numpy(x[k]) for k in ("input_ids", "attention_mask", "qtype_onehot")))
        with np.load(oracle_dir() / "hidden" / f"s{args.window}" / f"{index:03d}.npz") as data:
            sdpa = {k: data[k] for k in STATES}
        with np.load(oracle_dir() / "hidden_eager" / f"s{args.window}" / f"{index:03d}.npz") as data:
            eager = {k: data[k] for k in [*STATES, "marker_logits"]}
        for name in STATES:
            a = mine[name][0].numpy().astype(np.float64)[:n]
            s, e = sdpa[name][:n].astype(np.float64), eager[name][:n].astype(np.float64)
            scale = float(np.max(np.abs(s)))
            entry = per_state[name]
            entry["vs_eager_relative"] = max(entry["vs_eager_relative"], float(np.max(np.abs(a - e))) / scale)
            entry["vs_sdpa_relative"] = max(entry["vs_sdpa_relative"], float(np.max(np.abs(a - s))) / scale)
            entry["sdpa_vs_eager_relative"] = max(entry["sdpa_vs_eager_relative"], float(np.max(np.abs(s - e))) / scale)
        marker = np.asarray(gather_markers(mine["token_logits"][0].numpy(), row["markers"]), dtype=np.float64)
        markers["vs_eager"] = max(markers["vs_eager"], float(np.max(np.abs(marker - eager["marker_logits"]))))
        markers["vs_sdpa_padded"] = max(markers["vs_sdpa_padded"], float(np.max(np.abs(marker - np.asarray(row["marker_logits"])))))
    failing = [name for name in STATES if per_state[name]["vs_eager_relative"] > RELATIVE_BAR]
    failures = [f"states over {RELATIVE_BAR:g} vs eager: {failing}"] if failing else []
    if markers["vs_eager"] > MARKER_BAR:
        failures.append(f"marker logits {markers['vs_eager']:.3g} from eager")
    result = {"status": "FAIL" if failures else "PASS", "failures": failures, "stage": "authoring exactness (publisher's torch)",
              "model_sha": MODEL_SHA, "window": args.window, "torch": torch.__version__, "rows": len(subset(args.window)),
              "bars": {"relative_vs_eager": RELATIVE_BAR, "marker_vs_eager": MARKER_BAR}, "per_state": per_state, "markers": markers,
              "environment": environment(("torch", "numpy", "safetensors")), "seconds": time.perf_counter() - started,
              "input_hashes": hashes([source / "model.safetensors", oracle_dir() / "rows.json", Path(__file__),
                                      Path(__file__).parent / "_julia_model.py"])}
    out = results_dir() / f"exactness_s{args.window}.json"
    write_json(out, result)
    worst = max(STATES, key=lambda n: per_state[n]["vs_eager_relative"])
    print(result["status"], f"torch {torch.__version__}", f"max vs eager {per_state[worst]['vs_eager_relative']:.2e} at {worst};",
          f"final_norm vs sdpa {per_state['final_norm']['vs_sdpa_relative']:.2e} (publisher's own {per_state['final_norm']['sdpa_vs_eager_relative']:.2e});",
          f"marker vs eager {markers['vs_eager']:.2e}", failures, "->", out)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
