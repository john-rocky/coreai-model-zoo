#!/usr/bin/env python3
"""AOT GPU slot-probability gate, adapted from zoo 082fe55's decider readout.

Each row uses four fresh states, S=1 input_ids, full position_ids, and the last
256 logits. Divide by the row temperature, mask >=k except abstain=255, softmax,
then renormalise option probabilities. Raw256 argmax is recorded for Swift.
The controller limits each runtime process to 30 rows and 18,000 calls, puts
wide rows in separate processes, and repeats the first row at the very end.
Every session also repeats its own first row to prove state reset in process.

  python conversion/slot/readout_gate_slot.py <bundle> <fixtures> --transcript <gate.json>

PASS = every row with oracle margin >= 0.02 has the same option argmax, all
logits and option probabilities are finite, and every state-reset check is exact.
Probability and abstain errors are recorded, not used as additional thresholds.
The transcript retains the decider gate envelope with head="slot"; slot-specific
fields replace letter semantics. Raw256 argmax is not required to be an option.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

WORK = Path.cwd()
AOT_FLAGS = ["--platform", "macOS", "--preferred-compute", "gpu", "--architecture", "h16c",
             "--expect-frequent-reshapes"]
STATE_KEYS = ("k_cache", "v_cache", "conv_state", "rec_state")
RUNTIME_ERROR_LEDGER = WORK / "runtime_errors.json"
RUNTIME_STOP = WORK / "runtime_stop.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def local_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def write_json(path: Path, value: dict) -> None:
    path = local_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def round_deadline() -> float:
    # Optional --deadline-epoch can impose a supervising run's earlier limit.
    return time.time() + 24 * 3600


def remaining(deadline: float) -> float:
    seconds = deadline - time.time()
    if seconds <= 0:
        raise TimeoutError("gate wall-clock deadline reached")
    return seconds


def scratch_snapshot() -> dict:
    """Read-only du snapshots; never remove runtime scratch."""
    patterns = ["/private/var/folders/*/*/T/com.apple.MetalPerformanceShadersGraph",
                str(Path(os.environ.get("TMPDIR", str(WORK))) /
                    "com.apple.MetalPerformanceShadersGraph")]
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    records = []
    for path in paths:
        result = subprocess.run(["du", "-sh", path], capture_output=True, text=True, timeout=30)
        records.append({"path": path, "command": ["du", "-sh", path],
                        "stdout": result.stdout, "stderr": result.stderr,
                        "returncode": result.returncode})
    return {"at": utc_now(), "glob_patterns": patterns, "paths": records,
            "note": "No matching directory" if not paths else "Read-only; nothing deleted"}


def aot_compile(aimodel: Path, out_dir: Path, mode: str, deadline: float) -> tuple[Path, dict]:
    target = out_dir / f"{aimodel.stem}.h16c.aimodelc"
    receipt = WORK / f"aot_{mode}.json"
    fingerprint = {"path": str(aimodel), "bytes": aimodel.stat().st_size,
                   "mtime_ns": aimodel.stat().st_mtime_ns}
    if target.exists():
        return target, {"source": fingerprint, "asset": str(target), "result": "PASS",
                        "reused": True, "wall_seconds": 0.0,
                        "note": "Existing explicitly selected AOT asset reused; no GPU JIT."}
    if not os.environ.get("DEVELOPER_DIR"):
        raise RuntimeError("DEVELOPER_DIR must point to Xcode-27.0.0-RC.app/Contents/Developer")
    locate = subprocess.run(["xcrun", "-f", "coreai-build"], capture_output=True, text=True,
                            timeout=min(30, remaining(deadline)))
    if locate.returncode or not locate.stdout.strip():
        raise RuntimeError("xcrun -f coreai-build failed: " + locate.stderr)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = WORK / f"aot-{mode}.log"
    command = [locate.stdout.strip(), "compile", str(aimodel), "--output", str(out_dir), *AOT_FLAGS]
    record = {"source": fingerprint, "asset": str(target), "command": command,
              "log": str(log), "started_at": utc_now(), "result": "RUNNING", "reused": False}
    write_json(receipt, record)
    start = time.monotonic()
    try:
        with log.open("wb") as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                    timeout=remaining(deadline), cwd=Path.cwd())
        record["returncode"] = result.returncode
        if result.returncode or not target.exists():
            raise RuntimeError(f"AOT compile failed (returncode={result.returncode}); see {log}")
        record["result"] = "PASS"
    except Exception as exc:
        record["result"] = "FAIL"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["wall_seconds"] = time.monotonic() - start
        record["completed_at"] = utc_now()
        record["generated_at"] = record["completed_at"]
        write_json(receipt, record)
    return target, record


def slot_readout(logits: np.ndarray, row: dict) -> tuple[np.ndarray, np.ndarray]:
    # The author's Torch float32 softmax and the client's renormalisation.
    import torch
    scores = torch.from_numpy(logits).float() / float(row["temperature"])
    k = int(row["nopts"])
    scores[k:255] = -torch.inf
    full = scores.softmax(dim=-1)
    options = full[:k] / full[:k].sum()
    return full.numpy().copy(), options.numpy().copy()


def row_record(logits: np.ndarray, row: dict, seconds: float) -> dict:
    finite = bool(np.isfinite(logits).all())
    if not finite:
        raise FloatingPointError(f"{row['id']}: nonfinite slot logits")
    if logits.shape != (256,):
        raise ValueError(f"{row['id']}: expected 256 slot logits, got {logits.shape}")
    full, p = slot_readout(logits, row)
    if not np.isfinite(p).all():
        raise FloatingPointError(f"{row['id']}: nonfinite renormalised option probabilities")
    delta = np.abs(p.astype(np.float64) - np.asarray(row["p_oracle"], np.float64))
    oracle_raw = np.asarray(row["raw_logits"], np.float64)
    raw_delta = np.abs(logits.astype(np.float64) - oracle_raw)
    return {"id": row["id"], "request_id": row["request_id"], "question_id": row["question_id"],
            "type": row["type"], "qtype_index": row["qtype_index"], "kind": "slot",
            "tokens": len(row["ids"]), "nopts": row["nopts"], "temperature": row["temperature"],
            "raw_logits": logits.tolist(), "raw256_argmax": int(logits.argmax()),
            "option_argmax": int(p.argmax()), "oracle_argmax": row["argmax"],
            "argmax_agrees": int(p.argmax()) == row["argmax"],
            "p_full": full.tolist(), "p": p.tolist(), "p_oracle": row["p_oracle"],
            "abstain": float(full[255]), "abstain_oracle": row["abstain"],
            "abs_delta_abstain": abs(float(full[255]) - float(row["abstain"])),
            "oracle_margin": row["top2_margin"],
            "max_abs_delta_p": float(delta.max()), "mean_abs_delta_p": float(delta.mean()),
            "max_abs_delta_raw_logits": float(raw_delta.max()),
            "mean_abs_delta_raw_logits": float(raw_delta.mean()),
            "logits_min": float(logits.min()), "logits_max": float(logits.max()),
            "raw_logits_min": float(logits.min()), "raw_logits_max": float(logits.max()),
            "raw_logits_absmax": float(np.max(np.abs(logits))),
            "oracle_logits_min": float(oracle_raw.min()), "oracle_logits_max": float(oracle_raw.max()),
            "oracle_raw_logits_absmax": float(np.max(np.abs(oracle_raw))),
            "finite": finite, "nonconstant": bool(logits.max() > logits.min()),
            "wall_seconds_contended": seconds, "seconds": seconds,
            "full_vocab_argmax_id": int(logits.argmax()),
            "full_vocab_argmax_is_label": int(logits.argmax()) in row["label_ids"]}


def run_worker(args: argparse.Namespace) -> int:
    """One process owns one AIModel; persist each completed row immediately."""
    import torch
    import coreai.runtime as rt
    from coreai_models.models.macos import qwen3_5 as q

    config = json.loads(local_path(args.worker).read_text())
    transcript = local_path(config["transcript"])
    fixtures = json.loads(local_path(config["fixtures"]).read_text())
    rows = [fixtures["rows"][index] for index in config["indices"]]
    from types import SimpleNamespace
    source_config = json.loads(Path(config["config_path"]).read_text())
    cfg = q.qwen3_5_config_from_hf(SimpleNamespace(**source_config["text_config"]), config["max_ctx"], None)
    record = {"schema": "coreai-slot-readout-session/1", "session": config["session"],
              "mode": config["mode"], "indices": config["indices"], "pid": os.getpid(),
              "asset": config["asset"], "config_path": config["config_path"],
              "started_at": utc_now(), "phase": "starting", "rows": [], "runtime_calls": 0,
              "planned_runtime_calls": sum(len(row["ids"]) for row in rows) + len(rows[0]["ids"]),
              "result": "RUNNING"}
    if len(rows) > 30 or record["planned_runtime_calls"] > 18000:
        raise ValueError("session exceeds the 30-row / 18,000-call cap")
    assert len(q.DECODE_STATE_NAMES) == len(STATE_KEYS) == 4
    write_json(transcript, record)
    start = time.monotonic()

    def nd(array):
        return rt.NDArray(np.ascontiguousarray(array))

    async def run() -> None:
        remaining(config["deadline"])
        record["phase"] = "load"
        write_json(transcript, record)
        load_start = time.monotonic()
        model = await rt.AIModel.load(Path(config["asset"]), rt.SpecializationOptions.default())
        function = model.load_function("main")
        record["load_wall_seconds_contended"] = time.monotonic() - load_start

        async def row_logits(row: dict) -> np.ndarray:
            assert row["slot"] == len(row["ids_full"]) - 2
            assert row["ids"] == row["ids_full"][:row["slot"] + 1]
            assert row["ids"][-1] == 248082 and len(row["ids"]) <= config["max_ctx"]
            st = q.build_decode_state(cfg, max_seq_len=config["max_ctx"], dtype=torch.float16)
            assert all(torch.count_nonzero(st[key]).item() == 0 for key in STATE_KEYS)
            state = {name: nd(st[key].numpy()) for name, key in zip(q.DECODE_STATE_NAMES, STATE_KEYS)}
            record["phase"] = "step"
            record["active_row"] = row["id"]
            write_json(transcript, record)
            output = None
            for step, token in enumerate(row["ids"]):
                remaining(config["deadline"])
                record["active_step"] = step
                output = await function(inputs={"input_ids": nd(np.array([[token]], np.int32)),
                                                "position_ids": nd(np.arange(step + 1, dtype=np.int32)[None])},
                                        state=state)
                record["runtime_calls"] += 1
            logits = output["logits"].numpy()
            if logits.shape != (1, 1, 256):
                raise ValueError(f"expected logits [1,1,256], got {logits.shape}")
            return logits[0, -1].astype(np.float32).copy()

        first = None
        for row in rows:
            row_start = time.monotonic()
            logits = await row_logits(row)
            if first is None:
                first = logits.copy()
            record["phase"] = "readout"
            result = row_record(logits, row, time.monotonic() - row_start)
            record["rows"].append(result)
            write_json(transcript, record)
            print(f"{row['id']}: {len(row['ids'])} tokens, option argmax "
                  f"{result['option_argmax']}/{result['oracle_argmax']}, "
                  f"max|dp|={result['max_abs_delta_p']:.8f}, "
                  f"|dabstain|={result['abs_delta_abstain']:.8f}", flush=True)
        again = await row_logits(rows[0])
        record["reset_check"] = {"row": rows[0]["id"], "identical": bool(np.array_equal(first, again)),
                                  "max_abs_diff": float(np.max(np.abs(first - again))),
                                  "first_raw_logits": first.tolist(), "final_raw_logits": again.tolist(),
                                  "fresh_state_each_row": True, "same_process": True}
        record["phase"] = "complete"
        record["result"] = "PASS" if record["reset_check"]["identical"] else "FAIL"

    try:
        asyncio.run(run())
    except Exception as exc:
        record["result"] = "ERROR"
        record["error"] = {"stage": record["phase"], "class": type(exc).__name__,
                           "message": str(exc), "traceback": traceback.format_exc(),
                           "row": record.get("active_row"), "step": record.get("active_step")}
        print(record["error"]["traceback"], file=sys.stderr, flush=True)
    finally:
        record["completed_at"] = utc_now()
        record["generated_at"] = record["completed_at"]
        record["wall_seconds_contended"] = time.monotonic() - start
        write_json(transcript, record)
    return 0 if record["result"] == "PASS" else 2


def session_plan(rows: list[dict]) -> list[dict]:
    kit = [i for i, row in enumerate(rows) if int(row["nopts"]) <= 16]
    wide = [i for i, row in enumerate(rows) if int(row["nopts"]) > 16]
    plans, batch, calls = [], [], 0
    for index in kit:
        size = len(rows[index]["ids"])
        repeat = len(rows[batch[0]]["ids"]) if batch else size
        if batch and (len(batch) == 30 or calls + size + repeat > 18000):
            plans.append({"kind": "kit", "indices": batch})
            batch, calls = [], 0
        batch.append(index)
        calls += size
    if batch:
        plans.append({"kind": "kit", "indices": batch})
    plans.extend({"kind": "wide", "indices": [index]} for index in wide)
    plans.append({"kind": "global_reset", "indices": [0]})
    return plans


def record_runtime_error(error: dict, session: dict) -> int:
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


def summary(record: dict, expected: int) -> dict:
    rows = record["rows"]
    strict = [row for row in rows if row["oracle_margin"] >= 0.02]
    near = [row for row in rows if row["oracle_margin"] < 0.02]
    sessions = [session for session in record["sessions"] if session.get("result") == "PASS"]
    return {"rows": len(rows), "expected_rows": expected,
            "full_vocab_argmax_is_label": sum(row["full_vocab_argmax_is_label"] for row in rows),
            "argmax_agreement": sum(row["argmax_agrees"] for row in rows),
            "margin_ge_0_02_rows": len(strict),
            "margin_ge_0_02_argmax_agreement": sum(row["argmax_agrees"] for row in strict),
            "near_ties": [{"id": row["id"], "oracle_margin": row["oracle_margin"],
                            "oracle_argmax": row["oracle_argmax"], "option_argmax": row["option_argmax"],
                            "p": row["p"], "p_oracle": row["p_oracle"]} for row in near],
            "max_abs_delta_p": max((row["max_abs_delta_p"] for row in rows), default=None),
            "mean_of_row_mean_abs_delta_p": float(np.mean([row["mean_abs_delta_p"] for row in rows])) if rows else None,
            "max_abs_delta_abstain": max((row["abs_delta_abstain"] for row in rows), default=None),
            "max_abs_delta_raw_logits": max((row["max_abs_delta_raw_logits"] for row in rows), default=None),
            "finite_all": all(row["finite"] for row in rows),
            "nonconstant_all": all(row["nonconstant"] for row in rows),
            "reset_identical": record.get("reset_check", {}).get("identical", False),
            "session_resets_identical": all(session.get("reset_check", {}).get("identical", False) for session in sessions),
            "provisional_fp16_max_delta_p_expectation": 0.02,
            "provisional_expectation_exceeded": any(row["max_abs_delta_p"] > 0.02 for row in rows)
                                                if record["mode"] == "fp16" else None,
            "worst_five": sorted(rows, key=lambda row: row["max_abs_delta_p"], reverse=True)[:5]}


def controller(args: argparse.Namespace) -> int:
    global WORK, RUNTIME_ERROR_LEDGER, RUNTIME_STOP
    bundle, fixtures_path = local_path(args.bundle), local_path(args.fixtures)
    args.mode = args.mode or ("int8lin" if bundle.name.endswith("int8lin") else "fp16")
    transcript = local_path(args.transcript or f"gate-{bundle.name}-readout.json")
    WORK = local_path(args.work_dir) if args.work_dir else transcript.parent / ("." + transcript.stem + "-work")
    WORK.mkdir(parents=True, exist_ok=True)
    RUNTIME_ERROR_LEDGER = WORK / "runtime_errors.json"
    RUNTIME_STOP = WORK / "runtime_stop.json"
    if RUNTIME_STOP.exists():
        raise RuntimeError(f"runtime stop condition already fired: {RUNTIME_STOP}")
    deadline = args.deadline_epoch or round_deadline()
    fx = json.loads(fixtures_path.read_text())
    assert fx["schema"] == "coreai-slot-fixtures/1"
    assert fx["n_slots"] == 256 and fx["abstain_slot"] == 255 and fx["answer_token_id"] == 248082
    from huggingface_hub import hf_hub_download
    config_path = local_path(args.config or hf_hub_download(
        fx["source"]["hf_id"], "config.json", revision=fx["source"]["revision"]))
    source_config = json.loads(config_path.read_text())
    assert source_config["text_config"]["vocab_size"] == 248339
    assert source_config["text_config"]["tie_word_embeddings"] is False
    rows = fx["rows"]
    if not rows:
        raise ValueError("fixture has no rows")
    meta = json.loads((bundle / "metadata.json").read_text())
    assert meta["language"]["vocab_size"] == 256
    aimodel = local_path(bundle / meta["assets"]["main"])
    out_dir = local_path(args.aot_dir or bundle.parent / "aotc")
    record = {"schema": "coreai-decider-readout-gate/1", "head": "slot", "mode": args.mode, "bundle": str(bundle),
              "fixtures": str(fixtures_path), "config_path": str(config_path), "max_ctx": args.max_ctx,
              "runtime": "coreai Python runtime, AOT h16c GPU asset, SpecializationOptions.default()",
              "source": fx["source"], "temperature_by_type": fx["temperature_by_type"],
              "n_slots": 256, "abstain_slot": 255, "answer_token_id": 248082,
              "environment": {"python": sys.version.split()[0], "platform": sys.platform,
                              "DEVELOPER_DIR": os.environ.get("DEVELOPER_DIR")},
              "started_at": utc_now(), "timing": "contended; correctness only, no performance claim",
              "rows": [], "sessions": [], "result": "RUNNING"}
    write_json(transcript, record)
    start = time.monotonic()
    try:
        asset, compile_record = aot_compile(aimodel, out_dir, args.mode, deadline)
        record["asset"], record["aot"] = str(asset), compile_record
        record["aot_seconds"] = compile_record["wall_seconds"]
        write_json(transcript, record)
        print(f"AOT asset ready: {asset}", flush=True)
        for number, plan in enumerate(session_plan(rows)):
            remaining(deadline)
            if RUNTIME_STOP.exists():
                raise RuntimeError(f"runtime stop condition: {RUNTIME_STOP}")
            succeeded = False
            for attempt in (1, 2):
                session_name = f"{args.mode}-{number:02d}-{plan['kind']}-a{attempt}"
                session_path = WORK / f"{session_name}.json"
                worker_config_path = WORK / f"{session_name}.config.json"
                log_path = WORK / f"readout-{session_name}.log"
                config = {"session": session_name, "mode": args.mode, "asset": str(asset),
                          "config_path": str(config_path), "fixtures": str(fixtures_path),
                          "indices": plan["indices"], "max_ctx": args.max_ctx,
                          "deadline": deadline, "transcript": str(session_path)}
                write_json(worker_config_path, config)
                before = scratch_snapshot()
                call_start = time.monotonic()
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(worker_config_path)]
                timed_out = False
                with log_path.open("wb") as output:
                    try:
                        result = subprocess.run(command, cwd=Path.cwd(), stdout=output, stderr=subprocess.STDOUT,
                                                timeout=remaining(deadline))
                        returncode = result.returncode
                    except subprocess.TimeoutExpired:
                        timed_out, returncode = True, None
                after = scratch_snapshot()
                session = json.loads(session_path.read_text()) if session_path.exists() else {
                    "session": session_name, "mode": args.mode, "rows": [], "result": "ERROR", "phase": "starting"}
                session.update({"kind": plan["kind"], "transcript": str(session_path), "log": str(log_path),
                                "command": command, "returncode": returncode, "timed_out": timed_out,
                                "process_wall_seconds_contended": time.monotonic() - call_start,
                                "mpsgraph_scratch_before": before, "mpsgraph_scratch_after": after})
                write_json(session_path, session)
                record["sessions"].append({key: value for key, value in session.items() if key != "rows"})
                canonical = {row["id"]: row for row in record["rows"]}
                if plan["kind"] != "global_reset":
                    canonical.update({row["id"]: row for row in session["rows"]})
                    record["rows"] = [canonical[row["id"]] for row in rows if row["id"] in canonical]
                record["summary"] = summary(record, len(rows))
                write_json(transcript, record)
                if timed_out:
                    raise TimeoutError("round deadline reached during runtime process")
                if session.get("result") == "PASS" and returncode == 0:
                    if plan["kind"] == "global_reset":
                        original = canonical[rows[0]["id"]]["raw_logits"]
                        final = session["reset_check"]["final_raw_logits"]
                        same = bool(np.array_equal(np.asarray(original, np.float32), np.asarray(final, np.float32)))
                        record["reset_check"] = {"row": rows[0]["id"], "identical": same,
                                                 "max_abs_diff": float(np.max(np.abs(np.asarray(original) - final))),
                                                 "final_session": session_name, "final_raw_logits": final,
                                                 "cross_process": True, "local_resets_in_every_session": True}
                    succeeded = True
                    print(f"completed {session_name}: {len(session['rows'])} rows, "
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
        value = summary(record, len(rows))
        passed = (value["rows"] == value["expected_rows"] and value["finite_all"]
                  and value["margin_ge_0_02_argmax_agreement"] == value["margin_ge_0_02_rows"]
                  and value["reset_identical"] and value["session_resets_identical"])
        record["result"] = "PASS" if passed else "FAIL"
    except Exception as exc:
        record["result"] = "PARTIAL"
        record["error"] = {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        print(record["error"]["traceback"], file=sys.stderr, flush=True)
    finally:
        record["summary"] = summary(record, len(rows))
        record["wall_seconds_contended"] = time.monotonic() - start
        record["completed_at"] = utc_now()
        record["generated_at"] = record["completed_at"]
        write_json(transcript, record)
    value = record["summary"]
    print(f"{record['result']}: {value['argmax_agreement']}/{value['rows']} argmax; "
          f"margin>=0.02 {value['margin_ge_0_02_argmax_agreement']}/{value['margin_ge_0_02_rows']}; "
          f"max|dp|={value['max_abs_delta_p']}; max|dabstain|={value['max_abs_delta_abstain']}", flush=True)
    return 0 if record["result"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle", nargs="?")
    parser.add_argument("fixtures", nargs="?")
    parser.add_argument("--mode", choices=("fp16", "int8lin"))
    parser.add_argument("--config", help="local source config.json; otherwise fetch the fixture-pinned config")
    parser.add_argument("--transcript")
    parser.add_argument("--work-dir", help="session JSON and logs directory; default beside transcript")
    parser.add_argument("--aot-dir")
    parser.add_argument("--max-ctx", type=int, default=4096)
    parser.add_argument("--deadline-epoch", type=float)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return run_worker(args)
    if not args.bundle or not args.fixtures:
        parser.error("bundle and fixtures are required")
    return controller(args)


if __name__ == "__main__":
    sys.exit(main())
