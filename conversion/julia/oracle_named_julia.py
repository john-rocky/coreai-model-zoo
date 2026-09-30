#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.14.0",
#     "transformers==5.0.0",
#     "safetensors==0.8.0",
#     "numpy==2.5.3",
#     "pyarrow==25.0.1",
# ]
# ///
"""Stage 0b: the publisher's named-question answers — what a host must reproduce from text.

    uv run --python 3.12 conversion/julia/oracle_named_julia.py

For each of the 400 typed-decisions test cases, the publisher's README call
`engine.predict(state=<state>, questions=<the case's named questions>)` (julia/typed.py predict_typed over
the same FastEngine as oracle_julia.py: CPU fp32, strict, max_length 1024, head_length 512,
marker_only_head False) — every answer dictionary as returned (choice id / expected score / p[true],
full softmax probabilities) — and the raw logits `engine.logits` gives for the very rows predict_typed
renders, in the same batch. Writes oracle/named.json and results/oracle_named.json in the work dir.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_ID, MODEL_SHA, dataset_path, environment, hashes, oracle_dir, results_dir, verify_source, write_json  # noqa: E402

THREADS = 4


def main():
    started = time.perf_counter()
    torch.set_num_threads(THREADS)
    source = verify_source()
    sys.path.insert(0, str(source))
    import pyarrow.parquet as pq
    from julia.router.engine import FastEngine

    engine = FastEngine(source, device="cpu", transformer_backend="torch", strict_encoding=True,
                        max_length=1024, head_length=512, batch_size=16, marker_only_head=False)
    torch.set_num_threads(THREADS)
    cases = []
    for case in pq.read_table(dataset_path()).to_pylist():
        state, questions = json.loads(case["state"]), json.loads(case["questions"])
        rows = []
        for question in questions.values():
            kind, criteria = question["type"], question.get("criteria")
            if kind == "choice":
                options = list(criteria.values())
            elif kind == "score":
                options = list(criteria)
            else:
                options = ["false", "true"] if criteria is None else [criteria["false"], criteria["true"]]
            rows.append(dict(state=state, question=question.get("instructions"), type=kind, options=options))
        logits = engine.logits(rows)  # the same batch predict_typed runs (encoding cache, same order)
        answers = engine.predict(state=state, questions=questions)["answers"]
        cases.append({"id": case["id"], "state": state, "questions": questions, "rows_logits": logits, "answers": answers})
    write_json(oracle_dir() / "named.json", {"schema": "julia-oracle-named/1", "model": MODEL_ID, "model_sha": MODEL_SHA,
                                             "call": "engine.predict(state=..., questions=...)", "cases": cases})
    report = {"status": "PASS", "stage": "oracle (named questions)", "cases": len(cases),
              "questions": sum(len(c["answers"]) for c in cases), "environment": environment(("torch", "transformers", "numpy")),
              "output_hashes": hashes([oracle_dir() / "named.json"]), "seconds": time.perf_counter() - started}
    write_json(results_dir() / "oracle_named.json", report)
    print(json.dumps({k: report[k] for k in ("status", "cases", "questions", "seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
