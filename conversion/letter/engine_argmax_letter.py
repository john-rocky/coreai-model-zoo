#!/usr/bin/env python3
# /// script
# requires-python = "==3.11.*"
# dependencies = [
#   "coreai-core==1.0.0b2", "coreai-torch==0.4.1", "coreai-opt==0.2.1",
#   "torch==2.9.0", "transformers==4.57.6", "safetensors==0.7.0",
#   "huggingface_hub==0.36.2", "numpy==2.3.5", "tokenizers==0.22.2",
#   "accelerate==1.15.0",
# ]
# ///
"""Both Swift engines' first greedy text vs the bundle's full-vocabulary argmax.

Adapted from zoo 082fe55 and the sibling's two-engine gate. Every fixture row is
sent to each Release engine in its own foreground process, warmup off, S=1,
temperature zero. Expected text is tokenizer.decode([full_vocab_argmax_id])
from this bundle's Python readout. Agreement with the fp32 oracle's best LABEL
is separately counted. Raw stdout bytes are retained, and generated whitespace
is preserved by removing only the runner's exact framing. No mismatch retry.

  .venv/bin/python -B conversion/letter/engine_argmax_letter.py \
    exports/apus_openjev_v1_4b_decode_int8hu_block32_sym results/fixtures.json \
    --readout results/readout_int8hu.json --runner <RUN Release llm-runner>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import readout_gate_letter as gate
from readout_gate_letter import (
    RUN, RUNTIME_STOP, VOCAB_SIZE, alarm_returncode, deadline_for, foreground,
    local_path, remaining, safe_id, sha256, utc_now, validate_rows, validate_tokenization, write_json,
)

ENGINES = ("coreai-pipelined", "coreai-sequential")


def generated_text(stdout: bytes) -> str | None:
    """Remove exact Swift framing; preserve any generated CR, LF, or space."""
    banner = b"Generating...\n"
    end = "\n\n⏱️  Performance Summary:\n".encode("utf-8")
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


def run_engine(runner: Path, bundle: Path, row: dict, engine: str, mode: str,
               timeout: float, deadline: float) -> dict:
    raw_path = RUN / f"results/engine_inputs/{safe_id(row['id'])}.json"
    write_json(raw_path, {"tokens": row["ids"]})
    stem = f"engine-{engine}-{mode}-{safe_id(row['id'])}"
    stdout_path, stderr_path = RUN / f"logs/{stem}.log", RUN / f"logs/{stem}.stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path = RUN / f"results/readout_pids/{stem}.json"
    command = [str(runner), "--model", str(bundle), "--raw-tokens", str(raw_path),
               "--max-tokens", "1", "--temperature", "0.0",
               "--inference-engine-variant", engine, "--warmup", "off"]
    env = os.environ.copy()
    env["COREAI_CHUNK_THRESHOLD"] = "1"
    allotted = min(timeout, remaining(deadline))
    started_at = utc_now()
    start = time.monotonic()
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        result = foreground(command, min(deadline, time.time() + allotted), pid_path,
                            env=env, stdout=stdout_file, stderr=stderr_file)
    returncode = result.returncode
    stdout = stdout_path.read_bytes()
    return {"text": generated_text(stdout), "returncode": returncode,
            "timed_out": alarm_returncode(returncode), "started_at": started_at,
            "wall_seconds_contended": time.monotonic() - start, "timeout_seconds": allotted,
            "command": command, "pid_receipt": str(pid_path),
            "stdout_log": str(stdout_path), "stderr_log": str(stderr_path),
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(), "stderr_sha256": sha256(stderr_path),
            "stderr_tail": stderr_path.read_text(errors="replace")[-2000:] if returncode != 0 else "",
            **parse_preparation(stdout)}


def summarize(record: dict, expected: int) -> dict:
    engines = {}
    for engine in ENGINES:
        rows = [row for row in record["rows"] if row["engine"] == engine]
        engines[engine] = {"rows": len(rows), "expected_rows": expected,
                           "agreement": sum(row["agrees"] for row in rows),
                           "calls_succeeded": sum(row["returncode"] == 0 and not row["timed_out"] for row in rows),
                           "oracle_label_agreement": sum(row["oracle_label_agrees"] for row in rows),
                           "emits_row_label": sum(row["is_row_label"] for row in rows),
                           "disagreements": [{key: row[key] for key in (
                               "id", "full_vocab_argmax_id", "expected", "text", "oracle_argmax_label",
                               "returncode", "timed_out")} for row in rows if not row["agrees"]],
                           "oracle_label_disagreements": [{key: row[key] for key in (
                               "id", "oracle_argmax_label", "expected", "text", "oracle_margin")}
                               for row in rows if not row["oracle_label_agrees"]],
                           "first_call_preparation": {key: rows[0][key] for key in (
                               "id", "prepare_seconds_contended", "model_load_seconds_contended", "cache_hit",
                               "wall_seconds_contended", "stdout_log")} if rows else None}
        author_rows = [row for row in rows if "author_label_agrees" in row]
        if author_rows:
            engines[engine].update(author_rows=len(author_rows), author_label_agreement=sum(row["author_label_agrees"] for row in author_rows))
    result = {"engines": engines, "expected_calls": expected * len(ENGINES), "calls": len(record["rows"]),
            "agreement": sum(row["agrees"] for row in record["rows"]),
            "oracle_label_agreement": sum(row["oracle_label_agrees"] for row in record["rows"])}
    author_rows = [row for row in record["rows"] if "author_label_agrees" in row]
    if author_rows:
        result.update(author_rows=len(author_rows), author_label_agreement=sum(row["author_label_agrees"] for row in author_rows))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle")
    parser.add_argument("fixtures")
    parser.add_argument("--mode", default="int8hu", choices=("fp16", "int8hu"))
    parser.add_argument("--label-style", choices=("plain", "space-prefixed"), default="plain")
    parser.add_argument("--prompt-format", choices=("chat", "decision-function"), default="chat")
    parser.add_argument("--readout", required=True)
    parser.add_argument("--runner", default=os.environ.get("ZOO_LLM_RUNNER"))
    parser.add_argument("--transcript")
    parser.add_argument("--work-dir", type=Path, help="logs and raw-token files; default beside transcript")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--deadline-epoch", type=float, help="absolute wall-clock deadline as Unix seconds")
    args = parser.parse_args()
    global RUN, RUNTIME_STOP
    output = local_path(args.transcript or f"engine-argmax-{args.mode}.json")
    args.transcript = str(output)
    gate.configure_work(args.work_dir or output.parent / ("." + output.stem + "-work"))
    RUN, RUNTIME_STOP = gate.RUN, gate.RUNTIME_STOP
    if not 0 < args.timeout <= 600:
        parser.error("--timeout must be >0 and <=600 seconds")
    if not args.runner or not local_path(args.runner).is_file():
        parser.error("pass --runner to this run's Release llm-runner")
    if RUNTIME_STOP.exists():
        raise RuntimeError(f"round Python-runtime stop condition already fired: {RUNTIME_STOP}")
    from transformers import AutoTokenizer

    runner, bundle = local_path(args.runner), local_path(args.bundle)
    fixtures_path, readout_path = local_path(args.fixtures), local_path(args.readout)
    transcript = local_path(args.transcript or RUN / f"results/engine_argmax_{args.mode}.json")
    deadline = deadline_for(args.deadline_epoch)
    fx, readout = json.loads(fixtures_path.read_text()), json.loads(readout_path.read_text())
    fixture_rows = validate_rows(fx, 4096, args.label_style, args.prompt_format)
    assert readout["schema"] == "coreai-letter-readout-gate/1"
    assert local_path(readout["bundle"]) == bundle and readout["mode"] == args.mode
    assert readout["fixtures_sha256"] == sha256(fixtures_path)
    assert readout.get("label_style", "plain") == args.label_style
    assert readout.get("prompt_format", "chat") == args.prompt_format
    references = {row["id"]: row for row in readout["rows"]}
    missing = [row["id"] for row in fixture_rows if row["id"] not in references]
    if missing:
        raise ValueError(f"Python readout missing fixture rows: {missing}")
    tokenizer = AutoTokenizer.from_pretrained(bundle / "tokenizer", local_files_only=True)
    validate_tokenization(fixture_rows, tokenizer, args.prompt_format)
    record = {"schema": "coreai-letter-engine-gate/1", "mode": args.mode, "bundle": str(bundle),
              "label_style": args.label_style, "prompt_format": args.prompt_format, "source": fx["source"],
              "runner": str(runner), "readout": str(readout_path), "fixtures": str(fixtures_path),
              "runner_sha256": sha256(runner), "fixtures_sha256": sha256(fixtures_path),
              "readout_sha256": sha256(readout_path), "readout_result": readout["result"],
              "protocol": "Both engines, --raw-tokens --max-tokens 1 --temperature 0.0 --warmup off; "
                          "COREAI_CHUNK_THRESHOLD=1; exact decoded text of same-bundle full-vocabulary argmax",
              "text_comparison_note": "The prescribed comparison uses decoded text; distinct token IDs can decode identically.",
              "oracle_comparison": "Separately count exact text equality to the oracle argmax LABEL letter.",
              "timing": "contended; correctness only, no performance claim",
              "environment": {"python": sys.version, "deadline_epoch": deadline, "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve())},
              "started_at": utc_now(), "rows": [], "result": "RUNNING"}
    write_json(transcript, record)
    start = time.monotonic()
    try:
        for engine in ENGINES:
            for row in fixture_rows:
                remaining(deadline)
                if RUNTIME_STOP.exists():
                    raise RuntimeError(f"round runtime stop condition: {RUNTIME_STOP}")
                reference = references[row["id"]]
                full_id = int(reference["full_vocab_argmax_id"])
                assert 0 <= full_id < VOCAB_SIZE
                expected = tokenizer.decode([full_id])
                assert expected == reference["full_vocab_argmax_text"]
                oracle_label = row["labels"][row["argmax"]]
                got = run_engine(runner, bundle, row, engine, args.mode, args.timeout, deadline)
                value = {"id": row["id"], "engine": engine, "primitive": row["primitive"],
                         "zoo_only": bool(row.get("zoo_only", False)), "tokens": len(row["ids"]),
                         "full_vocab_argmax_id": full_id, "expected": expected, **got,
                         "agrees": got["text"] == expected, "oracle_argmax_label": oracle_label,
                         "oracle_label_agrees": got["text"] == oracle_label,
                         "is_row_label": got["text"] in row["labels"], "oracle_margin": row["top2_margin"]}
                if "p_author" in row:
                    author_label = row["labels"][row["author_argmax"]]
                    value.update(author_argmax_label=author_label, author_label_agrees=got["text"] == author_label)
                for key in ("kind", "request_id", "evidence_only"):
                    if key in row:
                        value[key] = row[key]
                record["rows"].append(value)
                record["summary"] = summarize(record, len(fixture_rows))
                write_json(transcript, record)
                print(f"{engine} {row['id']}: full={full_id}, expected={expected!r}, "
                      f"got={got['text']!r}, oracle_label={oracle_label!r}, "
                      f"{'ok' if value['agrees'] else 'MISMATCH'}, rc={got['returncode']}", flush=True)
        value = summarize(record, len(fixture_rows))
        passed = (value["calls"] == value["expected_calls"] and value["agreement"] == value["expected_calls"]
                  and all(engine["calls_succeeded"] == engine["expected_rows"] for engine in value["engines"].values()))
        record["result"] = "PASS" if passed else "FAIL"
    except Exception as exc:
        record["result"] = "PARTIAL"
        record["error"] = {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        print(record["error"]["traceback"], file=sys.stderr, flush=True)
    finally:
        record["summary"] = summarize(record, len(fixture_rows))
        record["wall_seconds_contended"] = time.monotonic() - start
        record["completed_at"] = utc_now()
        write_json(transcript, record)
    print(f"{record['result']}: {record['summary']['agreement']}/{record['summary']['calls']} "
          "calls match same-bundle Python full-vocabulary argmax text", flush=True)
    return 0 if record["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
