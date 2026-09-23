#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["coreai-core==1.0.0b2", "coreai-torch==0.4.1", "coreai-opt==0.2.1", "torch==2.9.0", "transformers==4.57.6", "safetensors==0.7.0", "numpy==2.3.5", "tokenizers==0.22.2"]
# ///
"""AOT h16c GPU scalar readout, with at most 15 row executions per process.

Adapted from the OpenThai sibling's readout_gate_decider.py. Each independent
option sequence gets four fresh zero states, S=1 calls and full position_ids.
The final logits[0, 0, 0] is its scalar. Questions use the author's softmax at
temperature 1.75. A session executes 14 rows plus its first row again; a final
session also repeats global row 1. All elapsed values describe contended work,
not performance measurements. Model licence: cc-by-nc-4.0.

  python conversion/scalar/readout_gate_scalar.py BUNDLE FIXTURES --mode fp16 \
    --snapshot MERGED --work-dir WORK
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RUN = Path("scalar-work").resolve()
DEADLINE = time.time() + 7200
ENVIRONMENT = {}


def configure_work(work_dir, deadline=None, environment_json=None):
    global RUN, DEADLINE, ENVIRONMENT, RUNTIME_ERROR_LEDGER, RUNTIME_STOP
    RUN = Path(work_dir).resolve()
    (RUN / "results").mkdir(parents=True, exist_ok=True)
    (RUN / "logs").mkdir(parents=True, exist_ok=True)
    DEADLINE = float(deadline) if deadline is not None else time.time() + 7200
    ENVIRONMENT = json.loads(Path(environment_json).read_text()) if environment_json else {}
    RUNTIME_ERROR_LEDGER = RUN / "results/runtime_errors.json"
    RUNTIME_STOP = RUN / "results/runtime_stop.json"


def load_text_config(snapshot, max_context_length=4096):
    from types import SimpleNamespace
    from coreai_models.models.macos.qwen3_5 import qwen3_5_config_from_hf
    raw = json.loads((Path(snapshot) / "config.json").read_text())
    text = raw.get("text_config", raw)
    if text["model_type"] != "qwen3_5_text" or text["hidden_size"] != 2560 or text["vocab_size"] != 248320:
        raise ValueError("expected the pinned 4B text config with its full 248320-row input embedding")
    return qwen3_5_config_from_hf(SimpleNamespace(**text), max_context_length)

AOT_FLAGS = ["--platform", "macOS", "--preferred-compute", "gpu", "--architecture", "h16c",
             "--expect-frequent-reshapes"]
STATE_KEYS = ("k_cache", "v_cache", "conv_state", "rec_state")
RUNTIME_ERROR_LEDGER = RUN / "results/runtime_errors.json"
RUNTIME_STOP = RUN / "results/runtime_stop.json"
TEMPERATURE = 1.75
MAX_LEN = 384
ROWS_PER_WORKER = 14


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def local_path(path: str | Path) -> Path:
    return Path(path).resolve()


def write_json(path: Path, value: dict) -> None:
    path = local_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def round_deadline() -> float:
    return DEADLINE


def remaining(deadline: float) -> float:
    seconds = deadline - time.time()
    if seconds <= 0:
        raise TimeoutError("supervised round wall-clock deadline reached")
    return seconds


def check_stop(deadline: float) -> None:
    remaining(deadline)
    if RUNTIME_STOP.exists():
        raise RuntimeError(f"the round runtime stop condition fired: {RUNTIME_STOP}")


def aot_compile(aimodel: Path, out_dir: Path, mode: str, deadline: float) -> tuple[Path, dict]:
    check_stop(deadline)
    target = out_dir / f"{aimodel.stem}.h16c.aimodelc"
    receipt = RUN / f"results/aot_{mode}.json"
    fingerprint = {"path": str(aimodel), "bytes": aimodel.stat().st_size,
                   "mtime_ns": aimodel.stat().st_mtime_ns}
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if (previous.get("result") == "PASS" and previous.get("source") == fingerprint
                and previous.get("flags") == AOT_FLAGS and target.exists()):
            return target, {**previous, "reused": True}
    if not os.environ.get("DEVELOPER_DIR"):
        raise RuntimeError("DEVELOPER_DIR must point to Xcode-27.0.0-RC.app/Contents/Developer")
    locate = subprocess.run(["xcrun", "-f", "coreai-build"], capture_output=True, text=True,
                            timeout=min(30, remaining(deadline)))
    if locate.returncode or not locate.stdout.strip():
        raise RuntimeError("xcrun -f coreai-build failed: " + locate.stderr)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = RUN / f"logs/aot-{mode}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [locate.stdout.strip(), "compile", str(aimodel), "--output", str(out_dir), *AOT_FLAGS]
    record = {"source": fingerprint, "asset": str(target), "command": command, "flags": AOT_FLAGS,
              "license": "cc-by-nc-4.0", "log": str(log), "started_at": utc_now(),
              "result": "RUNNING", "reused": False}
    write_json(receipt, record)
    start = time.monotonic()
    try:
        with log.open("wb") as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                    timeout=remaining(deadline), cwd=RUN)
        record["returncode"] = result.returncode
        if result.returncode or not target.exists():
            raise RuntimeError(f"AOT compile failed (returncode={result.returncode}); see {log}")
        record["result"] = "PASS"
    except Exception as exc:
        record["result"] = "FAIL"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["wall_seconds_contended"] = time.monotonic() - start
        record["completed_at"] = utc_now()
        write_json(receipt, record)
    return target, record


def validate_fixtures(fx: dict) -> None:
    if fx.get("schema") != "coreai-scalar-fixtures/1":
        raise ValueError("expected coreai-scalar-fixtures/1")
    if fx.get("temperature") != TEMPERATURE or fx.get("max_len") != MAX_LEN:
        raise ValueError("fixtures must specify temperature 1.75 and max_len 384")
    rows = {row["id"]: row for row in fx["rows"]}
    if not rows or len(rows) != len(fx["rows"]):
        raise ValueError("fixture rows must be nonempty and uniquely identified")
    seen = []
    for row in rows.values():
        if not 1 <= len(row["ids"]) <= MAX_LEN or row["slot"] != len(row["ids"]) - 1:
            raise ValueError(f"{row['id']}: invalid sequence length or scalar slot")
        if not all(isinstance(token, int) and token >= 0 for token in row["ids"]):
            raise ValueError(f"{row['id']}: token IDs must be nonnegative integers")
        if not math.isfinite(row["scalar"]):
            raise ValueError(f"{row['id']}: nonfinite oracle scalar")
    qids = [question["id"] for question in fx["questions"]]
    if not qids or len(set(qids)) != len(qids):
        raise ValueError("questions must be nonempty and uniquely identified")
    for question in fx["questions"]:
        ids = question["row_ids"]
        p = question["p_oracle"]
        if len(ids) < 2 or len(ids) != len(p) or len(ids) != len(question["options"]):
            raise ValueError(f"{question['id']}: option, row and probability widths differ")
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in p) or abs(sum(p) - 1) > 1e-6:
            raise ValueError(f"{question['id']}: invalid probability distribution")
        if not 0 <= question["argmax"] < len(ids):
            raise ValueError(f"{question['id']}: invalid argmax")
        if question["argmax"] != int(np.argmax(p)):
            raise ValueError(f"{question['id']}: oracle argmax disagrees with p_oracle")
        for row_id in ids:
            if row_id not in rows or rows[row_id]["question_id"] != question["id"]:
                raise ValueError(f"{question['id']}: row mapping is inconsistent")
        seen.extend(ids)
    if sorted(seen) != sorted(rows):
        raise ValueError("each fixture row must belong to exactly one question")


def author_softmax(scalars: list[float]) -> list[float]:
    """Author's system_one.py softmax: Python math.exp, max shift and sum."""
    values = [value / TEMPERATURE for value in scalars]
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def scalar_record(value: np.ndarray, row: dict, seconds: float) -> dict:
    if value.shape != (1, 1, 1):
        raise ValueError(f"{row['id']}: expected logits [1,1,1], got {value.shape}")
    if not np.isfinite(value).all():
        raise FloatingPointError(f"{row['id']}: nonfinite scalar")
    scalar = float(value[0, 0, 0])
    return {"id": row["id"], "request_id": row["request_id"], "question_id": row["question_id"],
            "type": row["type"], "tokens": len(row["ids"]), "slot": row["slot"],
            "scalar": scalar, "scalar_oracle": row["scalar"],
            "abs_delta_scalar": abs(scalar - float(row["scalar"])),
            "scalar_dtype": value.dtype.str, "scalar_bits": value.tobytes().hex(),
            "logits_shape": list(value.shape), "finite": True,
            "wall_seconds_contended": seconds}


def run_worker(args: argparse.Namespace) -> int:
    config = json.loads(local_path(args.worker).read_text())
    transcript = local_path(config["transcript"])
    fixtures = json.loads(local_path(config["fixtures"]).read_text())
    rows = [fixtures["rows"][index] for index in config["indices"]]
    record = {"schema": "coreai-scalar-readout-session/1", "session": config["session"],
              "mode": config["mode"], "indices": config["indices"], "pid": os.getpid(),
              "asset": config["asset"], "snapshot": config["snapshot"],
              "license": "cc-by-nc-4.0", "started_at": utc_now(), "phase": "starting",
              "rows": [], "runtime_calls": 0, "row_executions": 0,
              "planned_row_executions": len(rows) + 1,
              "planned_runtime_calls": sum(len(row["ids"]) for row in rows) + len(rows[0]["ids"]),
              "result": "RUNNING"}
    write_json(transcript, record)
    start = time.monotonic()
    try:
        if not rows or len(rows) > ROWS_PER_WORKER or record["planned_row_executions"] > 15:
            raise ValueError("session exceeds the 15-row-execution cap (including its reset repeat)")
        if any(len(row["ids"]) > MAX_LEN for row in rows):
            raise ValueError("session contains a sequence longer than 384 tokens")
        import torch
        import coreai.runtime as rt
        from coreai_models.models.macos import qwen3_5 as q

        cfg = load_text_config(config["snapshot"], max_context_length=config["max_ctx"])
        if len(q.DECODE_STATE_NAMES) != len(STATE_KEYS):
            raise ValueError("expected four decode states")

        def nd(array):
            return rt.NDArray(np.ascontiguousarray(array))

        async def run() -> None:
            check_stop(config["deadline"])
            record["phase"] = "load"
            write_json(transcript, record)
            load_start = time.monotonic()
            model = await rt.AIModel.load(Path(config["asset"]), rt.SpecializationOptions.default())
            function = model.load_function("main")
            record["load_wall_seconds_contended"] = time.monotonic() - load_start

            async def row_logits(row: dict) -> np.ndarray:
                if row["slot"] != len(row["ids"]) - 1 or not 1 <= len(row["ids"]) <= MAX_LEN:
                    raise ValueError(f"{row['id']}: invalid scalar slot or sequence length")
                st = q.build_decode_state(cfg, max_seq_len=config["max_ctx"], dtype=torch.float16)
                if not all(torch.count_nonzero(st[key]).item() == 0 for key in STATE_KEYS):
                    raise ValueError("decode states were not freshly zero initialized")
                state = {name: nd(st[key].numpy()) for name, key in zip(q.DECODE_STATE_NAMES, STATE_KEYS)}
                record["phase"], record["active_row"] = "step", row["id"]
                record["row_executions"] += 1
                write_json(transcript, record)
                output = None
                for step, token in enumerate(row["ids"]):
                    check_stop(config["deadline"])
                    record["active_step"] = step
                    output = await function(inputs={"input_ids": nd(np.array([[token]], np.int32)),
                                                    "position_ids": nd(np.arange(step + 1, dtype=np.int32)[None])},
                                            state=state)
                    record["runtime_calls"] += 1
                value = output["logits"].numpy().copy()
                if value.shape != (1, 1, 1):
                    raise ValueError(f"expected logits [1,1,1], got {value.shape}")
                return value

            first = None
            for row in rows:
                row_start = time.monotonic()
                value = await row_logits(row)
                if first is None:
                    first = value.copy()
                record["phase"] = "readout"
                result = scalar_record(value, row, time.monotonic() - row_start)
                record["rows"].append(result)
                write_json(transcript, record)
                print(f"{row['id']}: {len(row['ids'])} tokens, scalar={result['scalar']:.8g}, "
                      f"oracle={result['scalar_oracle']:.8g}, |ds|={result['abs_delta_scalar']:.8g}", flush=True)
            again = await row_logits(rows[0])
            if not np.isfinite(again).all():
                raise FloatingPointError("nonfinite reset scalar")
            identical = first.dtype == again.dtype and first.tobytes() == again.tobytes()
            record["reset_check"] = {"row": rows[0]["id"], "identical": identical,
                                      "comparison": "dtype and exact bytes",
                                      "abs_diff": float(abs(float(first[0, 0, 0]) - float(again[0, 0, 0]))),
                                      "first_scalar": float(first[0, 0, 0]), "final_scalar": float(again[0, 0, 0]),
                                      "scalar_dtype": first.dtype.str, "first_bits": first.tobytes().hex(),
                                      "final_bits": again.tobytes().hex(), "fresh_state_each_row": True,
                                      "same_process": True}
            record["phase"] = "complete"
            record["result"] = "PASS" if identical else "FAIL"

        asyncio.run(run())
    except Exception as exc:
        record["result"] = "ERROR"
        record["error"] = {"stage": record["phase"], "class": type(exc).__name__,
                           "message": str(exc), "traceback": traceback.format_exc(),
                           "row": record.get("active_row"), "step": record.get("active_step")}
        print(record["error"]["traceback"], file=sys.stderr, flush=True)
    finally:
        record["completed_at"] = utc_now()
        record["wall_seconds_contended"] = time.monotonic() - start
        write_json(transcript, record)
    return 0 if record["result"] == "PASS" else 2


def session_plan(rows: list[dict]) -> list[dict]:
    plans = [{"kind": "rows", "indices": list(range(start, min(start + ROWS_PER_WORKER, len(rows))))}
             for start in range(0, len(rows), ROWS_PER_WORKER)]
    return plans + [{"kind": "global_reset", "indices": [0]}]


def record_runtime_error(error: dict, session: dict) -> int:
    lock = RUN / "results/runtime_errors.lock"
    with lock.open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        ledger = json.loads(RUNTIME_ERROR_LEDGER.read_text()) if RUNTIME_ERROR_LEDGER.exists() else {"errors": []}
        message = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", error["message"])
        signature = hashlib.sha256(f"{error['stage']}:{error['class']}:{message}".encode()).hexdigest()
        item = {**error, "signature": signature, "session": session["session"],
                "mode": session["mode"], "at": utc_now()}
        ledger["errors"].append(item)
        count = sum(previous["signature"] == signature for previous in ledger["errors"])
        write_json(RUNTIME_ERROR_LEDGER, ledger)
        if count >= 2:
            write_json(RUNTIME_STOP, {"reason": "Python-runtime load/step failed twice with the same error",
                                      "count": count, "error": item})
        return count


def question_records(record: dict, fixtures: dict) -> list[dict]:
    rows = {row["id"]: row for row in record["rows"]}
    questions = []
    for question in fixtures["questions"]:
        if not all(row_id in rows for row_id in question["row_ids"]):
            continue
        selected = [rows[row_id] for row_id in question["row_ids"]]
        scalars = [row["scalar"] for row in selected]
        p = author_softmax(scalars)
        delta = [abs(a - b) for a, b in zip(p, question["p_oracle"])]
        argmax = int(np.argmax(p))
        questions.append({"id": question["id"], "request_id": selected[0]["request_id"],
                          "type": question.get("type", selected[0]["type"]),
                          "zoo_only": question.get("zoo_only", len(scalars) > 16),
                          "row_ids": question["row_ids"], "options": question["options"],
                          "nopts": len(scalars), "scalars": scalars,
                          "scalars_oracle": [row["scalar_oracle"] for row in selected],
                          "p": p, "p_oracle": question["p_oracle"], "argmax": argmax,
                          "oracle_argmax": question["argmax"], "argmax_agrees": argmax == question["argmax"],
                          "selected_option": question["options"][argmax],
                          "oracle_selected_option": question["options"][question["argmax"]],
                          "oracle_margin": question["top2_margin"], "abs_delta_p": delta,
                          "max_abs_delta_p": max(delta), "mean_abs_delta_p": sum(delta) / len(delta),
                          "finite": all(math.isfinite(value) for value in p)})
    return questions


def summarize(record: dict, fixtures: dict) -> dict:
    questions = question_records(record, fixtures)
    record["questions"] = questions
    rows = record["rows"]
    strict = [q for q in questions if q["oracle_margin"] >= 0.02]
    near = [q for q in questions if q["oracle_margin"] < 0.02]
    successful = [s for s in record["sessions"] if s.get("result") == "PASS" and s.get("returncode") == 0]
    max_dp = max((q["max_abs_delta_p"] for q in questions), default=None)
    expectation = 0.02 if record["mode"] == "fp16" else 0.05
    mean_q = sum(q["mean_abs_delta_p"] for q in questions) / len(questions) if questions else None
    return {"rows": len(rows), "expected_rows": len(fixtures["rows"]),
            "questions": len(questions), "expected_questions": len(fixtures["questions"]),
            "argmax_agreement": sum(q["argmax_agrees"] for q in questions),
            "margin_ge_0_02_questions": len(strict),
            "margin_ge_0_02_argmax_agreement": sum(q["argmax_agrees"] for q in strict),
            "near_ties": [{key: q[key] for key in ("id", "oracle_margin", "oracle_argmax", "argmax",
                                                  "argmax_agrees", "p", "p_oracle")} for q in near],
            "strict_disagreements": [q["id"] for q in strict if not q["argmax_agrees"]],
            "max_abs_delta_p": max_dp, "mean_of_question_mean_abs_delta_p": mean_q,
            "mean_of_row_mean_abs_delta_p": mean_q,
            "mean_of_row_mean_note": "Each probability row is one question; options are weighted equally within it.",
            "mean_option_abs_delta_p": (sum(sum(q["abs_delta_p"]) for q in questions)
                                         / sum(q["nopts"] for q in questions)) if questions else None,
            "max_abs_delta_scalar": max((r["abs_delta_scalar"] for r in rows), default=None),
            "mean_abs_delta_scalar": sum(r["abs_delta_scalar"] for r in rows) / len(rows) if rows else None,
            "scalar_min": min((r["scalar"] for r in rows), default=None),
            "scalar_max": max((r["scalar"] for r in rows), default=None),
            "scalar_absmax": max((abs(r["scalar"]) for r in rows), default=None),
            "oracle_scalar_min": min((r["scalar_oracle"] for r in rows), default=None),
            "oracle_scalar_max": max((r["scalar_oracle"] for r in rows), default=None),
            "oracle_scalar_absmax": max((abs(r["scalar_oracle"]) for r in rows), default=None),
            "finite_all": bool(rows) and all(r["finite"] for r in rows) and all(q["finite"] for q in questions),
            "reset_identical": record.get("reset_check", {}).get("identical", False),
            "session_resets_identical": bool(successful) and all(s.get("reset_check", {}).get("identical", False)
                                                                   for s in successful),
            "max_row_executions_per_process": max((s.get("row_executions", 0) for s in record["sessions"]), default=0),
            "provisional_max_delta_p_expectation": expectation,
            "provisional_expectation_exceeded": max_dp > expectation if max_dp is not None else None,
            "provisional_expectation_is_acceptance_gate": False,
            "worst_five_questions": sorted(questions, key=lambda q: q["max_abs_delta_p"], reverse=True)[:5]}


def controller(args: argparse.Namespace) -> int:
    bundle, fixtures_path = local_path(args.bundle), local_path(args.fixtures)
    transcript = local_path(args.transcript or RUN / f"results/readout_{args.mode}.json")
    deadline = min(args.deadline_epoch or round_deadline(), round_deadline())
    snapshot = local_path(args.snapshot)
    fixtures = json.loads(fixtures_path.read_text())
    validate_fixtures(fixtures)
    rows = fixtures["rows"]
    metadata = json.loads((bundle / "metadata.json").read_text())
    if metadata["language"]["vocab_size"] != 1 or metadata["decision"]["head"] != "scalar":
        raise ValueError("bundle must have the scalar head and logits width 1")
    if metadata["decision"]["temperature"] != TEMPERATURE or metadata["decision"]["license"] != "cc-by-nc-4.0":
        raise ValueError("bundle must declare temperature 1.75 and cc-by-nc-4.0")
    max_ctx = metadata["language"]["max_context_length"]
    if max_ctx < MAX_LEN:
        raise ValueError("bundle context length cannot hold a 384-token scorer row")
    if args.max_ctx is not None and args.max_ctx != max_ctx:
        raise ValueError("--max-ctx must match the exported bundle metadata")
    aimodel = local_path(bundle / metadata["assets"]["main"])
    out_dir = local_path(args.aot_dir or RUN / "aotc")
    record = {"schema": "coreai-scalar-readout-gate/1", "mode": args.mode, "bundle": str(bundle),
              "fixtures": str(fixtures_path), "snapshot": str(snapshot), "max_ctx": max_ctx,
              "runtime": "coreai Python runtime, AOT h16c GPU asset, SpecializationOptions.default()",
              "source": fixtures.get("source"), "temperature": TEMPERATURE, "max_len": MAX_LEN,
              "layout": "State/Question/Option, last token", "logits_width": 1, "license": "cc-by-nc-4.0",
              "environment": ENVIRONMENT, "started_at": utc_now(),
              "timing": "contended; correctness only, no performance claim", "rows_per_worker": ROWS_PER_WORKER,
              "max_row_executions_per_worker": 15, "rows": [], "questions": [], "sessions": [], "result": "RUNNING"}
    write_json(transcript, record)
    start = time.monotonic()
    try:
        check_stop(deadline)
        asset, compile_record = aot_compile(aimodel, out_dir, args.mode, deadline)
        record["asset"], record["aot"] = str(asset), compile_record
        write_json(transcript, record)
        print(f"AOT asset ready: {asset}", flush=True)
        for number, plan in enumerate(session_plan(rows)):
            check_stop(deadline)
            succeeded = False
            for attempt in (1, 2):
                check_stop(deadline)
                session_name = f"{args.mode}-{number:03d}-{plan['kind']}-a{attempt}"
                session_path = RUN / f"results/readout_sessions/{session_name}.json"
                config_path = RUN / f"results/readout_sessions/{session_name}.config.json"
                log_path = RUN / f"logs/readout-{session_name}.log"
                config = {"session": session_name, "mode": args.mode, "asset": str(asset),
                          "snapshot": str(snapshot), "fixtures": str(fixtures_path),
                          "indices": plan["indices"], "max_ctx": max_ctx,
                          "deadline": deadline, "transcript": str(session_path), "work_dir": str(RUN), "environment": ENVIRONMENT}
                write_json(config_path, config)
                call_start = time.monotonic()
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(config_path)]
                timed_out = False
                with log_path.open("wb") as output:
                    try:
                        result = subprocess.run(command, cwd=RUN, stdout=output, stderr=subprocess.STDOUT,
                                                timeout=remaining(deadline))
                        returncode = result.returncode
                    except subprocess.TimeoutExpired:
                        timed_out, returncode = True, None
                session = json.loads(session_path.read_text()) if session_path.exists() else {
                    "session": session_name, "mode": args.mode, "rows": [], "result": "ERROR", "phase": "starting"}
                session.update({"kind": plan["kind"], "transcript": str(session_path), "log": str(log_path),
                                "command": command, "returncode": returncode, "timed_out": timed_out,
                                "process_wall_seconds_contended": time.monotonic() - call_start})
                write_json(session_path, session)
                record["sessions"].append({key: value for key, value in session.items() if key != "rows"})
                canonical = {row["id"]: row for row in record["rows"]}
                if plan["kind"] != "global_reset":
                    canonical.update({row["id"]: row for row in session["rows"]})
                    record["rows"] = [canonical[row["id"]] for row in rows if row["id"] in canonical]
                record["summary"] = summarize(record, fixtures)
                write_json(transcript, record)
                if timed_out:
                    raise TimeoutError("round deadline reached during runtime process")
                if session.get("result") == "PASS" and returncode == 0:
                    if plan["kind"] == "global_reset":
                        original = canonical[rows[0]["id"]]
                        final = session["reset_check"]
                        identical = (original["scalar_dtype"] == final["scalar_dtype"]
                                     and original["scalar_bits"] == final["final_bits"])
                        record["reset_check"] = {"row": rows[0]["id"], "identical": identical,
                                                 "comparison": "dtype and exact bytes",
                                                 "abs_diff": abs(original["scalar"] - final["final_scalar"]),
                                                 "first_scalar": original["scalar"], "final_scalar": final["final_scalar"],
                                                 "first_bits": original["scalar_bits"], "final_bits": final["final_bits"],
                                                 "final_session": session_name, "cross_process": True,
                                                 "local_resets_in_every_session": True}
                    succeeded = True
                    print(f"completed {session_name}: {len(session['rows'])} rows + reset, "
                          f"{session.get('runtime_calls')} calls; total {len(record['rows'])}/{len(rows)}", flush=True)
                    break
                if session.get("result") == "FAIL":
                    raise RuntimeError(f"state reset mismatch in {session_name}; no retry")
                error = session.get("error") or {
                    "stage": session.get("phase", "step"), "class": "RuntimeProcessFailure",
                    "message": "\n".join(log_path.read_text(errors="replace").splitlines()[-12:])}
                if error["stage"] not in ("load", "step"):
                    raise RuntimeError(f"{session_name} failed in {error['stage']}: {error['message']}")
                count = record_runtime_error(error, session)
                if count >= 2:
                    raise RuntimeError(f"STOP: repeated Python-runtime error; {RUNTIME_STOP}")
                print(f"runtime error in {session_name}; one fresh-process retry permitted", flush=True)
            if not succeeded:
                raise RuntimeError(f"runtime session {number} did not complete")
        value = summarize(record, fixtures)
        passed = (value["rows"] == value["expected_rows"] and value["questions"] == value["expected_questions"]
                  and value["finite_all"] and value["margin_ge_0_02_argmax_agreement"] == value["margin_ge_0_02_questions"]
                  and value["reset_identical"] and value["session_resets_identical"]
                  and value["max_row_executions_per_process"] <= 15)
        record["result"] = "PASS" if passed else "FAIL"
    except Exception as exc:
        record["result"] = "PARTIAL"
        record["error"] = {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        print(record["error"]["traceback"], file=sys.stderr, flush=True)
    finally:
        record["summary"] = summarize(record, fixtures)
        record["wall_seconds_contended"] = time.monotonic() - start
        record["completed_at"] = utc_now()
        write_json(transcript, record)
    value = record["summary"]
    print(f"{record['result']}: {value['argmax_agreement']}/{value['questions']} question argmax; "
          f"margin>=0.02 {value['margin_ge_0_02_argmax_agreement']}/{value['margin_ge_0_02_questions']}; "
          f"max|dp|={value['max_abs_delta_p']}; max|dscalar|={value['max_abs_delta_scalar']}", flush=True)
    return 0 if record["result"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle", nargs="?")
    parser.add_argument("fixtures", nargs="?")
    parser.add_argument("--mode", choices=("fp16", "int8lin"))
    parser.add_argument("--snapshot", help="base or merged text config directory (required for readout)")
    parser.add_argument("--work-dir", default="scalar-work")
    parser.add_argument("--environment-json")
    parser.add_argument("--transcript")
    parser.add_argument("--aot-dir")
    parser.add_argument("--max-ctx", type=int, help="optional assertion against bundle metadata")
    parser.add_argument("--deadline-epoch", type=float)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--compile-only", action="store_true", help="AOT compile without Python-runtime sessions")
    args = parser.parse_args()
    if args.worker:
        config = json.loads(Path(args.worker).read_text())
        configure_work(config["work_dir"], config["deadline"])
        globals()["ENVIRONMENT"] = config.get("environment", {})
        return run_worker(args)
    configure_work(args.work_dir, args.deadline_epoch, args.environment_json)
    if not args.bundle or not args.mode or (not args.fixtures and not args.compile_only):
        parser.error("bundle, fixtures and --mode are required (fixtures may be omitted with --compile-only)")
    if args.compile_only:
        bundle = local_path(args.bundle)
        metadata = json.loads((bundle / "metadata.json").read_text())
        asset, receipt = aot_compile(local_path(bundle / metadata["assets"]["main"]),
                                     local_path(args.aot_dir or RUN / "aotc"), args.mode,
                                     min(args.deadline_epoch or round_deadline(), round_deadline()))
        print(json.dumps({"asset": str(asset), "aot": receipt}, indent=1))
        return 0
    if not args.snapshot:
        parser.error("--snapshot is required for readout")
    return controller(args)


if __name__ == "__main__":
    sys.exit(main())
