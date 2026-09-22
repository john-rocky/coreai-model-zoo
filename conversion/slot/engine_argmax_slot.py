#!/usr/bin/env python3
"""Both Swift engines' first greedy text vs the same bundle's raw256 argmax.

Adapted from zoo 082fe55's decider engine gate. Slot indices are not option token
IDs: compare exact tokenizer.decode([raw256_argmax]) from this bundle's Python
readout, without temperature, option masking, or the fp32 oracle argmax. Calls
inherit the isolated environment, run in the foreground with <=600-second limits,
and preserve stdout bytes in logs/engine-<engine>-<mode>-<row>.log. Mismatches
are results and never trigger another invocation.

  python conversion/slot/engine_argmax_slot.py <bundle> <fixtures> \
    --readout <readout.json> --runner <llm-runner> --engine pipelined \
    --engine sequential --transcript <gate.json>

PASS = every selected engine call succeeds and its exact emitted text matches.
Repeated --engine flags aggregate both engines in the same decider gate envelope.
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

from readout_gate_slot import local_path, remaining, round_deadline, utc_now, write_json

WORK = Path.cwd()

ENGINES = ("coreai-pipelined",)


def generated_text(stdout: bytes) -> str | None:
    """Strip exact runner delimiters, preserving a generated CR/LF/space."""
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


def run_engine(runner: Path, bundle: Path, row: dict, engine: str, mode: str,
               timeout: float, deadline: float) -> dict:
    raw_path = WORK / f"{safe_id(row['id'])}.tokens.json"
    write_json(raw_path, {"tokens": row["ids"]})
    stem = f"engine-{engine}-{mode}-{safe_id(row['id'])}"
    stdout_path, stderr_path = WORK / f"{stem}.log", WORK / f"{stem}.stderr.log"
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
            result = subprocess.run(command, cwd=Path.cwd(), env=env, stdout=stdout_file, stderr=stderr_file,
                                    timeout=allotted)
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            timed_out, returncode = True, None
    stdout = stdout_path.read_bytes()
    return {"text": generated_text(stdout), "returncode": returncode, "timed_out": timed_out,
            "started_at": started_at, "wall_seconds_contended": time.monotonic() - start,
            "seconds": time.monotonic() - start,
            "timeout_seconds": allotted, "command": command,
            "stdout_log": str(stdout_path), "stderr_log": str(stderr_path),
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_tail": stderr_path.read_text(errors="replace")[-2000:] if returncode != 0 else "",
            **parse_preparation(stdout)}


def summarize(record: dict, expected: int) -> dict:
    engines = {}
    for engine in ENGINES:
        rows = [row for row in record["rows"] if row["engine"] == engine]
        engines[engine] = {"rows": len(rows), "expected_rows": expected,
                           "agreement": sum(row["agrees"] for row in rows),
                           "calls_succeeded": sum(row["returncode"] == 0 and not row["timed_out"] for row in rows),
                           "not_single_byte_token_rows": [row["id"] for row in rows if not row["is_single_byte_token"]],
                           "disagreements": [{key: row[key] for key in ("id", "raw256_argmax", "expected", "text", "returncode", "timed_out")}
                                             for row in rows if not row["agrees"]],
                           "first_call_preparation": {key: rows[0][key] for key in (
                               "id", "prepare_seconds_contended", "model_load_seconds_contended", "cache_hit",
                               "wall_seconds_contended", "stdout_log")} if rows else None}
    return {"engines": engines, "expected_calls": expected * len(ENGINES), "calls": len(record["rows"]),
            "rows": len(record["rows"]),
            "calls_succeeded": sum(e["calls_succeeded"] for e in engines.values()),
            "disagreements": [dict(engine=engine, **r) for engine,e in engines.items() for r in e["disagreements"]],
            "agreement": sum(row["agrees"] for row in record["rows"])}


def main() -> int:
    global WORK, ENGINES
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle")
    parser.add_argument("fixtures")
    parser.add_argument("--mode", choices=("fp16", "int8lin"))
    parser.add_argument("--engine", choices=("pipelined", "sequential"), action="append",
                        help="engine to check; repeat to combine both in one transcript (default pipelined)")
    parser.add_argument("--readout", required=True)
    parser.add_argument("--runner", default=os.environ.get("ZOO_LLM_RUNNER"))
    parser.add_argument("--transcript")
    parser.add_argument("--work-dir", help="token input and stdout logs directory; default beside transcript")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--deadline-epoch", type=float)
    args = parser.parse_args()
    if not 0 < args.timeout <= 600:
        parser.error("--timeout must be >0 and <=600 seconds")
    if not args.runner or not local_path(args.runner).is_file():
        parser.error("pass --runner or set ZOO_LLM_RUNNER to a Release llm-runner")
    from transformers import AutoTokenizer

    runner, bundle = local_path(args.runner), local_path(args.bundle)
    fixtures_path, readout_path = local_path(args.fixtures), local_path(args.readout)
    args.mode = args.mode or ("int8lin" if bundle.name.endswith("int8lin") else "fp16")
    ENGINES = tuple(dict.fromkeys("coreai-" + engine for engine in (args.engine or ["pipelined"])))
    transcript = local_path(args.transcript or f"gate-{bundle.name}-engine.json")
    WORK = local_path(args.work_dir) if args.work_dir else transcript.parent / ("." + transcript.stem + "-work")
    WORK.mkdir(parents=True, exist_ok=True)
    deadline = args.deadline_epoch or round_deadline()
    fx, readout = json.loads(fixtures_path.read_text()), json.loads(readout_path.read_text())
    assert fx["schema"] == "coreai-slot-fixtures/1"
    assert readout["schema"] == "coreai-decider-readout-gate/1" and readout["head"] == "slot"
    assert local_path(readout["bundle"]) == bundle and readout["mode"] == args.mode
    references = {row["id"]: row for row in readout["rows"]}
    missing = [row["id"] for row in fx["rows"] if row["id"] not in references]
    if missing:
        raise ValueError(f"Python readout missing fixture rows: {missing}")
    tokenizer = AutoTokenizer.from_pretrained(bundle / "tokenizer", local_files_only=True)
    decoded_bytes = {slot: tokenizer.decode([slot]) for slot in range(256)}
    byte_texts = set(decoded_bytes.values())
    record = {"schema": "coreai-decider-engine-gate/1", "head": "slot", "mode": args.mode, "engines": ENGINES, "bundle": str(bundle),
              "runner": str(runner), "readout": str(readout_path), "fixtures": str(fixtures_path),
              "runner_sha256": hashlib.file_digest(runner.open("rb"), "sha256").hexdigest(),
              "protocol": "Selected engines, --raw-tokens --max-tokens 1 --temperature 0.0 --warmup off; "
                          "COREAI_CHUNK_THRESHOLD=1; exact decoded text of same-bundle raw256 argmax",
              "timing": "contended; first-call preparation recorded separately, no performance claim",
              "environment": {"python": sys.version.split()[0], "platform": sys.platform,
                              "DEVELOPER_DIR": os.environ.get("DEVELOPER_DIR")},
              "byte_token_texts": decoded_bytes,
              "text_comparison_note": "Different bytes can decode to the same Unicode replacement character; "
                                      "the prescribed gate compares emitted text, not recovered token IDs.",
              "started_at": utc_now(), "rows": [], "result": "RUNNING"}
    write_json(transcript, record)
    start = time.monotonic()
    try:
        for engine in ENGINES:
            for row in fx["rows"]:
                remaining(deadline)
                reference = references[row["id"]]
                raw_argmax = int(reference["raw256_argmax"])
                assert 0 <= raw_argmax < 256
                expected = decoded_bytes[raw_argmax]
                got = run_engine(runner, bundle, row, engine, args.mode, args.timeout, deadline)
                value = {"id": row["id"], "engine": engine, "raw256_argmax": raw_argmax,
                         "expected": expected, **got,
                         "agrees": got["text"] == expected,
                         "is_single_byte_token": got["text"] in byte_texts,
                         "matching_byte_slot_ids": [slot for slot, text in decoded_bytes.items() if text == got["text"]],
                         "oracle_margin": row["top2_margin"]}
                record["rows"].append(value)
                record["summary"] = summarize(record, len(fx["rows"]))
                write_json(transcript, record)
                print(f"{engine} {row['id']}: raw256={raw_argmax}, expected={expected!r}, "
                      f"got={got['text']!r}, {'ok' if value['agrees'] else 'MISMATCH'}, "
                      f"rc={got['returncode']}, {got['wall_seconds_contended']:.2f}s contended", flush=True)
        value = summarize(record, len(fx["rows"]))
        passed = (value["calls"] == value["expected_calls"] and value["agreement"] == value["expected_calls"]
                  and all(engine["calls_succeeded"] == engine["expected_rows"] for engine in value["engines"].values()))
        record["result"] = "PASS" if passed else "FAIL"
    except Exception as exc:
        record["result"] = "PARTIAL"
        record["error"] = {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        print(record["error"]["traceback"], file=sys.stderr, flush=True)
    finally:
        record["summary"] = summarize(record, len(fx["rows"]))
        record["wall_seconds_contended"] = time.monotonic() - start
        record["completed_at"] = utc_now()
        record["generated_at"] = record["completed_at"]
        write_json(transcript, record)
    print(f"{record['result']}: {record['summary']['agreement']}/{record['summary']['calls']} "
          f"calls match same-bundle Python raw256 text", flush=True)
    return 0 if record["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
