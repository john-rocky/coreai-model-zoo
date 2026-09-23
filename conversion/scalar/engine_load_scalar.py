#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.3.5", "transformers==4.57.6", "tokenizers==0.22.2"]
# ///
"""Load both scorer bundles through both Swift engines, two rows per bundle.

This is a load/completion check. The one-wide scalar head always samples id 0;
the emitted text does not validate option probabilities. Model: cc-by-nc-4.0.
Adapted from the OpenThai sibling's engine_argmax_decider.py.

  python conversion/scalar/engine_load_scalar.py FIXTURES --bundle-root BUNDLES \
    --runner RELEASE_LLM_RUNNER --work-dir WORK
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import readout_gate_scalar as gate
from readout_gate_scalar import (check_stop, local_path, remaining, round_deadline,
                                utc_now, validate_fixtures, write_json)
RUN = Path("scalar-work").resolve()

ENGINES = ("coreai-pipelined", "coreai-sequential")
MODES = ("fp16", "int8lin")
BUNDLE_PREFIX = "system_one_qwen3_5_4b_scorer_decode_"


def generated_text(stdout: bytes) -> str | None:
    """Use the exact 0.2.4-zoo runner delimiters, preserving generated whitespace."""
    banner = b"Generating...\n"
    end = "\n\n⏱️  Performance Summary:\n".encode()
    if banner not in stdout:
        return None
    body = stdout.split(banner, 1)[1]
    if end not in body:
        return None
    return body.split(end, 1)[0].decode("utf-8", errors="replace")


def parse_preparation(stdout: bytes) -> dict:
    text = stdout.decode("utf-8", errors="replace")
    preparation = re.search(r"done in ([0-9.]+)s( \(cache hit\))?", text)
    load = re.search(r"Model Load:\s*([0-9.]+)ms", text)
    generated = re.search(r"Generation:\s*[0-9.]+ms,\s*(\d+) tokens", text)
    return {"prepare_seconds_contended": float(preparation.group(1)) if preparation else None,
            "cache_hit": bool(preparation.group(2)) if preparation else None,
            "model_load_seconds_contended": float(load.group(1)) / 1000 if load else None,
            "generated_token_count": int(generated.group(1)) if generated else None}


def safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"row id is not a safe filename: {value!r}")
    return value


def select_rows(fixtures: dict, requested: list[str] | None) -> list[dict]:
    rows = fixtures["rows"]
    lookup = {row["id"]: row for row in rows}
    if requested:
        if len(requested) != 2 or len(set(requested)) != 2:
            raise ValueError("exactly two distinct --row-id values are required")
        return [lookup[row_id] for row_id in requested]
    # Includes both fixture row 1 and a truncation-length case when the fixtures provide it.
    others = [row for row in rows if row["id"] != rows[0]["id"]]
    if not others:
        raise ValueError("at least two fixture rows are required")
    return [rows[0], max(others, key=lambda row: len(row["ids"]))]


def run_engine(runner: Path, bundle: Path, row: dict, engine: str, mode: str,
               timeout: float, deadline: float) -> dict:
    check_stop(deadline)
    raw_path = RUN / f"results/engine_inputs/{safe_id(row['id'])}.json"
    write_json(raw_path, {"tokens": row["ids"]})
    stem = f"engine-load-{engine}-{mode}-{safe_id(row['id'])}"
    stdout_path, stderr_path = RUN / f"logs/{stem}.log", RUN / f"logs/{stem}.stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(runner), "--model", str(bundle), "--raw-tokens", str(raw_path),
               "--max-tokens", "1", "--temperature", "0.0",
               "--inference-engine-variant", engine, "--warmup", "off"]
    env = os.environ.copy()
    env["COREAI_CHUNK_THRESHOLD"] = "1"
    allotted = min(timeout, remaining(deadline))
    started_at = utc_now()
    start = time.monotonic()
    timed_out = False
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        try:
            result = subprocess.run(command, cwd=RUN, env=env, stdout=stdout_file, stderr=stderr_file,
                                    timeout=allotted)
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            timed_out, returncode = True, None
    stdout = stdout_path.read_bytes()
    stderr = stderr_path.read_bytes()
    return {"text": generated_text(stdout), "returncode": returncode, "timed_out": timed_out,
            "started_at": started_at, "wall_seconds_contended": time.monotonic() - start,
            "timeout_seconds": allotted, "command": command, "environment_overrides": {"COREAI_CHUNK_THRESHOLD": "1"},
            "stdout_log": str(stdout_path), "stderr_log": str(stderr_path),
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(), "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "stderr_tail": stderr.decode("utf-8", errors="replace")[-2000:] if returncode != 0 else "",
            **parse_preparation(stdout)}


def summarize(record: dict) -> dict:
    modes = {}
    for mode in record["modes"]:
        rows = [row for row in record["calls"] if row["mode"] == mode]
        modes[mode] = {"calls": len(rows), "expected_calls": 4,
                       "calls_completed": sum(row["returncode"] == 0 and not row["timed_out"] for row in rows),
                       "one_token_calls": sum(row["generated_token_count"] == 1 for row in rows),
                       "id_zero_text_matches": sum(row["text_matches_id_zero"] for row in rows),
                       "passed_calls": sum(row["passed"] for row in rows),
                       "engines": {engine: {"calls": sum(row["engine"] == engine for row in rows),
                                            "passed_calls": sum(row["engine"] == engine and row["passed"] for row in rows)}
                                   for engine in ENGINES}}
    return {"modes": modes, "expected_calls": len(record["modes"]) * 4, "calls": len(record["calls"]),
            "passed_calls": sum(row["passed"] for row in record["calls"]),
            "failed_calls": [{key: row[key] for key in ("mode", "engine", "id", "text", "expected_text",
                                                        "returncode", "timed_out", "generated_token_count")}
                             for row in record["calls"] if not row["passed"]],
            "probability_gate_executed": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    global RUN
    parser.add_argument("fixtures")
    parser.add_argument("--work-dir", default="scalar-work")
    parser.add_argument("--environment-json")
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--mode", choices=MODES, help="default: both modes")
    parser.add_argument("--runner", default=os.environ.get("ZOO_LLM_RUNNER"))
    parser.add_argument("--row-id", action="append", help="use exactly two named fixture rows")
    parser.add_argument("--transcript")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--deadline-epoch", type=float)
    args = parser.parse_args()
    gate.configure_work(args.work_dir, args.deadline_epoch, args.environment_json)
    RUN = gate.RUN
    if not args.runner:
        parser.error("--runner (or ZOO_LLM_RUNNER) must name a Release llm-runner")
    if not 0 < args.timeout <= 600:
        parser.error("--timeout must be >0 and <=600 seconds")
    runner = local_path(args.runner)
    if not runner.is_file():
        parser.error("--runner must name this run's Release llm-runner")
    from transformers import AutoTokenizer

    fixtures_path, bundle_root = local_path(args.fixtures), local_path(args.bundle_root)
    transcript = local_path(args.transcript or RUN / (f"results/engine_load_{args.mode}.json" if args.mode else "results/engine_load.json"))
    deadline = min(args.deadline_epoch or round_deadline(), round_deadline())
    fixtures = json.loads(fixtures_path.read_text())
    validate_fixtures(fixtures)
    selected = select_rows(fixtures, args.row_id)
    modes = [args.mode] if args.mode else list(MODES)
    with runner.open("rb") as handle:
        runner_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    record = {"schema": "coreai-scalar-engine-load/1", "modes": modes, "runner": str(runner),
              "runner_sha256": runner_hash, "fixtures": str(fixtures_path), "license": "cc-by-nc-4.0",
              "selected_row_ids": [row["id"] for row in selected],
              "selection": "two explicit --row-id values" if args.row_id else "first row and longest distinct row",
              "protocol": "Two rows per bundle, both engines; --raw-tokens --max-tokens 1 --temperature 0.0 "
                          "--warmup off; COREAI_CHUNK_THRESHOLD=1",
              "scope": "Engine LOAD/completion and fresh-process reset only. A one-wide head always selects id 0; its generated text is meaningless. No Swift probability gate.",
              "token_id_note": "Runner reports emitted text and generated-token count; no numeric generated token ID is logged.",
              "timing": "contended; correctness only, no performance claim",
              "environment": gate.ENVIRONMENT,
              "started_at": utc_now(), "bundles": {}, "calls": [], "result": "RUNNING"}
    write_json(transcript, record)
    start = time.monotonic()
    try:
        for mode in modes:
            check_stop(deadline)
            bundle = local_path(bundle_root / (BUNDLE_PREFIX + mode))
            metadata = json.loads((bundle / "metadata.json").read_text())
            if metadata["language"]["vocab_size"] != 1 or metadata["decision"]["head"] != "scalar":
                raise ValueError(f"{mode}: expected the one-wide scalar-head bundle")
            if metadata["decision"]["license"] != "cc-by-nc-4.0":
                raise ValueError(f"{mode}: expected cc-by-nc-4.0")
            tokenizer = AutoTokenizer.from_pretrained(bundle / "tokenizer", local_files_only=True)
            expected = tokenizer.decode([0], skip_special_tokens=False)
            record["bundles"][mode] = {"path": str(bundle), "logits_width": 1, "expected_id": 0,
                                        "expected_text": expected, "metadata": metadata}
            for engine in ENGINES:
                for row in selected:
                    check_stop(deadline)
                    got = run_engine(runner, bundle, row, engine, mode, args.timeout, deadline)
                    value = {"id": row["id"], "question_id": row["question_id"], "tokens": len(row["ids"]),
                             "engine": engine, "mode": mode, "expected_text": expected,
                             **got, "text_matches_id_zero": got["text"] == expected}
                    value["passed"] = (got["returncode"] == 0 and not got["timed_out"]
                                       and got["generated_token_count"] == 1 and value["text_matches_id_zero"])
                    record["calls"].append(value)
                    record["summary"] = summarize(record)
                    write_json(transcript, record)
                    print(f"{mode} {engine} {row['id']}: expected={expected!r}, got={got['text']!r}, "
                          f"count={got['generated_token_count']}, rc={got['returncode']}, "
                          f"{'PASS' if value['passed'] else 'FAIL'}", flush=True)
                    if got["timed_out"]:
                        raise TimeoutError(f"{mode} {engine} {row['id']}: runner timeout, no retry")
        value = summarize(record)
        passed = value["calls"] == value["expected_calls"] and value["passed_calls"] == value["expected_calls"]
        record["result"] = "PASS" if passed else "FAIL"
    except Exception as exc:
        record["result"] = "PARTIAL"
        record["error"] = {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        print(record["error"]["traceback"], file=sys.stderr, flush=True)
    finally:
        record["summary"] = summarize(record)
        record["wall_seconds_contended"] = time.monotonic() - start
        record["completed_at"] = utc_now()
        write_json(transcript, record)
    value = record["summary"]
    print(f"{record['result']}: {value['passed_calls']}/{value['calls']} engine calls completed with one id-0 text token", flush=True)
    return 0 if record["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
