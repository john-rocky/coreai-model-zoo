#!/usr/bin/env python3
"""Engine argmax gate: the Swift pipelined engine's first greedy token vs the oracle's letter.

The `coreai-pipelined` engine samples on the GPU and exposes no logits, so the probability
gate (`readout_gate_decider.py`) runs through the Python runtime; this gate proves the ENGINE
path on the same rows: `llm-runner --raw-tokens <row ids> --max-tokens 1 --temperature 0.0`
must emit exactly the oracle's argmax label string ("B", "GD", ...). Same protocol as the
zoo's `coreai_gate.py`: greedy, warmup off, `COREAI_CHUNK_THRESHOLD=1`, generated text taken
between the "Generating..." banner and the summary. Every call is its own process.

    python3 conversion/decider/engine_argmax_decider.py exports/decider_0_8b_decode_int8hu_block32_sym \
        models/decider-0.8b/fixtures-decider-0.8b.json --runner <fork>/.build/release/llm-runner \
        --transcript models/decider-0.8b/gate-decider-0.8b-engine.json

PASS = every row's text equals its oracle label (no near-tie exemption).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def run_engine(runner: str, bundle: Path, ids: list[int], timeout: int) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"tokens": ids}, f)
        raw = f.name
    env = {"COREAI_CHUNK_THRESHOLD": "1", "PATH": "/usr/bin:/bin"}
    t0 = time.monotonic()
    try:
        r = subprocess.run([runner, "--model", str(bundle), "--raw-tokens", raw, "--max-tokens", "1",
                            "--temperature", "0.0", "--inference-engine-variant", "coreai-pipelined",
                            "--warmup", "off"], capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"text": None, "returncode": None, "timed_out": True, "seconds": time.monotonic() - t0}
    finally:
        Path(raw).unlink(missing_ok=True)
    text = None
    if "Generating..." in r.stdout and "⏱" in r.stdout.split("Generating...", 1)[1]:
        body = r.stdout.split("Generating...", 1)[1].split("⏱", 1)[0]
        if body.startswith("\n"):
            body = body[1:]
        if body.endswith("\n\n"):
            body = body[:-2]
        text = body
    return {"text": text, "returncode": r.returncode, "timed_out": False,
            "seconds": time.monotonic() - t0, "stderr_tail": r.stderr[-400:] if r.returncode else ""}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle")
    ap.add_argument("fixtures", help="oracle_decider.py output")
    ap.add_argument("--runner", default=os.environ.get("ZOO_LLM_RUNNER"), help="llm-runner (Release)")
    ap.add_argument("--transcript")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per call (the first call specializes)")
    args = ap.parse_args()
    if not args.runner or not Path(args.runner).is_file():
        sys.exit("pass --runner or set ZOO_LLM_RUNNER to a Release llm-runner")

    bundle = Path(args.bundle).resolve()
    fx = json.loads(Path(args.fixtures).read_text())
    record = {"schema": "coreai-decider-engine-gate/1", "bundle": str(bundle), "runner": args.runner,
              "protocol": "llm-runner --raw-tokens --max-tokens 1 --temperature 0.0 "
                          "--inference-engine-variant coreai-pipelined --warmup off, COREAI_CHUNK_THRESHOLD=1",
              "environment": {"platform": platform.platform()}, "rows": []}
    for row in fx["rows"]:
        got = run_engine(args.runner, bundle, row["ids"], args.timeout)
        rec = {"id": row["id"], "expected": row["argmax_label"], "text": got["text"],
               "agrees": got["text"] == row["argmax_label"], "is_row_label": got["text"] in row["label_strings"],
               "oracle_margin": row["top2_margin"], **{k: got[k] for k in ("returncode", "timed_out", "seconds")}}
        if got.get("stderr_tail"):
            rec["stderr_tail"] = got["stderr_tail"]
        record["rows"].append(rec)
        print(f"{row['id']:>22} expected {row['argmax_label']!r:6} got {got['text']!r:8} "
              f"{'ok' if rec['agrees'] else 'NO'}  {got['seconds']:.1f}s", flush=True)

    rows = record["rows"]
    s = {"rows": len(rows), "agreement": sum(r["agrees"] for r in rows),
         "calls_succeeded": sum(1 for r in rows if r["returncode"] == 0 and not r["timed_out"]),
         "disagreements": [{"id": r["id"], "expected": r["expected"], "text": r["text"],
                            "oracle_margin": r["oracle_margin"]} for r in rows if not r["agrees"]]}
    record["summary"] = s
    record["result"] = "PASS" if s["agreement"] == s["rows"] else "FAIL"
    record["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"{record['result']}: {s['agreement']}/{s['rows']} rows emit the oracle label")
    if args.transcript:
        Path(args.transcript).parent.mkdir(parents=True, exist_ok=True)
        Path(args.transcript).write_text(json.dumps(record, indent=1) + "\n")
        print(f"  transcript: {args.transcript}")
    sys.exit(0 if record["result"] == "PASS" else 1)


if __name__ == "__main__":
    main()
