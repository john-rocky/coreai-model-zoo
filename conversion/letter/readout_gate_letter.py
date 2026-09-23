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
"""AOT Mac GPU letter probabilities for chat or plain decision-function fixtures.

Adapted from zoo 082fe55 and the sibling OpenThai process-splitting gate. The
unchanged 248320-vocabulary bundle receives each FULL compiled chat prompt as
S=1 steps, with four fresh zero states for every evaluation. Its last fp16 logits
are gathered at the fixture label IDs and softmaxed in Torch float32. Defaults
preserve APUS T=1; OpenJev uses bare A–Z then a–z and explicit T=0.85.
At most 14 fixture rows plus one reset, and 14000 calls, occupy one process;
each zoo_only row has its own process. The first row is repeated again at the
very end. Full-vocabulary logits are saved as .npy evidence, including repeats.

  .venv/bin/python -B conversion/letter/readout_gate_letter.py \
    exports/apus_decision_v1_4b_decode_int8hu_block32_sym results/fixtures.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
import math
import resource
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

RUN = Path.cwd()
AOT_FLAGS = ["--platform", "macOS", "--preferred-compute", "gpu", "--architecture", "h16c",
             "--expect-frequent-reshapes"]
STATE_KEYS = ("k_cache", "v_cache", "conv_state", "rec_state")
VOCAB_SIZE = 248320
MAX_PROMPT_EVALUATIONS = 15
MAX_CALLS = 14000
RUNTIME_ERROR_LEDGER = RUN / "results/readout_runtime_errors.json"
RUNTIME_STOP = RUN / "results/readout_runtime_stop.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def local_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def configure_work(path: str | Path) -> None:
    global RUN, RUNTIME_ERROR_LEDGER, RUNTIME_STOP
    RUN = local_path(path)
    (RUN / "results").mkdir(parents=True, exist_ok=True)
    (RUN / "logs").mkdir(parents=True, exist_ok=True)
    RUNTIME_ERROR_LEDGER = RUN / "results/readout_runtime_errors.json"
    RUNTIME_STOP = RUN / "results/readout_runtime_stop.json"


def write_json(path: Path, value: dict) -> None:
    path = local_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"row id is not a safe filename: {value!r}")
    return value


def remaining(deadline: float) -> float:
    seconds = deadline - time.time()
    if seconds <= 0:
        raise TimeoutError("gate wall-clock deadline reached")
    return seconds


def deadline_for(override: float | None) -> float:
    return override if override is not None else time.time() + 3 * 3600


def foreground(command: list[str], deadline: float, receipt: Path, **kwargs) -> subprocess.CompletedProcess:
    """Child sets its own alarm and execs; parent never sends a child a signal."""
    remaining(deadline)
    wrapper = [sys.executable, "-B", str(Path(__file__).resolve()), "--exec-child",
               str(deadline), str(local_path(receipt)), *command]
    return subprocess.run(wrapper, cwd=RUN, **kwargs)


def exec_child(arguments: list[str]) -> None:
    deadline, receipt, *command = arguments
    delay = remaining(float(deadline))
    write_json(Path(receipt), {"pid": os.getpid(), "parent_pid": os.getppid(),
               "command": command, "deadline_epoch": float(deadline), "started_at": utc_now(),
               "cwd": os.getcwd(), "termination": "self ITIMER_REAL/SIGALRM; parent sends no signal"})
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, delay)
    os.execvpe(command[0], command, os.environ)


def alarm_returncode(returncode: int) -> bool:
    return returncode in (-signal.SIGALRM, 128 + signal.SIGALRM, 124)


def aot_compile(aimodel: Path, out_dir: Path, mode: str, deadline: float) -> tuple[Path, dict]:
    target = out_dir / f"{aimodel.stem}.h16c.aimodelc"
    receipt = RUN / f"results/aot_{mode}.json"
    model_files = sorted(path for path in aimodel.rglob("*") if path.is_file()) if aimodel.is_dir() else [aimodel]
    fingerprint = {"path": str(aimodel), "files": [
        {"path": str(path.relative_to(aimodel)) if aimodel.is_dir() else path.name,
         "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in model_files]}
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous.get("result") == "PASS" and previous.get("source") == fingerprint and target.exists():
            return target, {**previous, "reused": True}
    if os.environ.get("DEVELOPER_DIR") != "/Applications/Xcode-27.0.0-RC.app/Contents/Developer":
        raise RuntimeError("source scripts/env.sh: DEVELOPER_DIR must point to Xcode 27 RC")
    locate = foreground(["xcrun", "-f", "coreai-build"], min(deadline, time.time() + 30),
                        RUN / "results/readout_pids/locate-coreai-build.json", capture_output=True, text=True)
    if locate.returncode or not locate.stdout.strip():
        raise RuntimeError("xcrun -f coreai-build failed: " + locate.stderr)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = RUN / f"logs/aot-{mode}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [locate.stdout.strip(), "compile", str(aimodel), "--output", str(out_dir), *AOT_FLAGS]
    pid_receipt = RUN / f"results/readout_pids/aot-{mode}.json"
    record = {"source": fingerprint, "asset": str(target), "command": command,
              "log": str(log), "pid_receipt": str(pid_receipt), "started_at": utc_now(),
              "result": "RUNNING", "reused": False}
    write_json(receipt, record)
    start = time.monotonic()
    try:
        with log.open("wb") as output:
            result = foreground(command, deadline, pid_receipt, stdout=output, stderr=subprocess.STDOUT)
        record["returncode"] = result.returncode
        if alarm_returncode(result.returncode):
            raise TimeoutError("round deadline reached during AOT compile")
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


def validate_rows(fx: dict, max_ctx: int, label_style: str = "plain", prompt_format: str = "chat", temperature: float = 1.0) -> list[dict]:
    if fx.get("schema") != "coreai-letter-fixtures/1":
        raise ValueError("expected coreai-letter-fixtures/1")
    if float(fx.get("temperature", 1.0)) != temperature:
        raise ValueError("fixture/CLI temperature mismatch")
    if label_style == "bare" and (fx.get("label_style") != "bare" or fx.get("template") != prompt_format):
        raise ValueError("fixture/CLI bare label style or template mismatch")
    if label_style == "space-prefixed" and fx.get("label_style") != label_style:
        raise ValueError("space-prefixed mode requires matching fixture label_style")
    rows = fx["rows"]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("fixtures must contain rows with unique IDs")
    for row in rows:
        safe_id(row["id"])
        labels, ids = row["labels"], row["ids"]
        n = len(labels)
        alphabet = ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" if label_style == "bare" else
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if label_style == "space-prefixed" else "ABCDEFGHIJKLMNOP")
        expected_labels = [(" " if label_style == "space-prefixed" else "") + x for x in alphabet[:n]]
        if not 2 <= n <= len(alphabet) or labels != expected_labels:
            raise ValueError(f"{row['id']}: invalid {label_style} option labels")
        if not ids or row["slot"] != len(ids) - 1 or len(ids) > max_ctx:
            raise ValueError(f"{row['id']}: invalid compiled sequence/slot/context")
        if len(set(row["label_ids"])) != n or any(not 0 <= tok < VOCAB_SIZE for tok in row["label_ids"]):
            raise ValueError(f"{row['id']}: invalid label IDs")
        if len(row["raw_logits"]) != n or len(row["p_oracle"]) != n:
            raise ValueError(f"{row['id']}: invalid oracle vectors")
        if "p_author" in row:
            for key in ("p_author", "author_raw_logits" if "author_raw_logits" in row else "raw_logits"):
                values = np.asarray(row[key], np.float64)
                if values.shape != (n,) or not np.isfinite(values).all():
                    raise ValueError(f"{row['id']}: invalid optional {key}")
            if row.get("author_argmax", row["argmax"]) != int(np.argmax(row["p_author"])):
                raise ValueError(f"{row['id']}: invalid author argmax")
        if prompt_format == "decision-function":
            if row["tokens"] != len(ids) or len(row["options"]) != n:
                raise ValueError(f"{row['id']}: invalid decision row token/option count")
            if row["kind"] == "bool" and row["options"] != ["yes", "no"]:
                raise ValueError(f"{row['id']}: bool must use ['yes', 'no']")
            if not row.get("prompt", "").endswith("\n\nAnswer:"):
                raise ValueError(f"{row['id']}: invalid plain-text decision answer slot")
        if not 0 <= row["argmax"] < n or float(row.get("temperature", 1.0)) != temperature:
            raise ValueError(f"{row['id']}: invalid argmax or temperature")
        if label_style == "bare":
            for key in ("raw_logprobs", "p_author"):
                value = np.asarray(row[key], np.float64)
                if value.shape != (n,) or not np.isfinite(value).all():
                    raise ValueError(f"{row['id']}: invalid {key}")
            if row["p_oracle"] != row["p_author"]:
                raise ValueError("OpenJev uses exactly one reference, author helper oracle A")
        if row.get("zoo_only"):
            if not 1500 <= len(ids) <= 2000:
                raise ValueError(f"{row['id']}: zoo_only must have 1500-2000 compiled tokens")
        elif len(ids) > 1024:
            raise ValueError(f"{row['id']}: ordinary row exceeds 1024 compiled tokens")
    return rows


def validate_tokenization(rows: list[dict], tokenizer, prompt_format: str, label_style: str = "plain") -> None:
    if label_style == "bare":
        for row in rows:
            encoded = [tokenizer.encode(label, add_special_tokens=False) for label in row["labels"]]
            if encoded != [[i] for i in row["label_ids"]]:
                raise ValueError(f"{row['id']}: bundle tokenizer disagrees with bare label IDs")
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["composed_user_text"]}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False)
            if hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            if ids != row["ids"]:
                raise ValueError(f"{row['id']}: bundle tokenizer disagrees with source chat IDs")
    if prompt_format != "decision-function":
        return
    for row in rows:
        if [tokenizer.encode(label, add_special_tokens=False) for label in row["labels"]] != [[i] for i in row["label_ids"]]:
            raise ValueError(f"{row['id']}: bundle tokenizer disagrees with space-prefixed label IDs")
        if tokenizer.encode(row["prompt"], add_special_tokens=False) != row["ids"]:
            raise ValueError(f"{row['id']}: bundle tokenizer disagrees with raw plain-text prompt IDs")


def row_record(logits: np.ndarray, row: dict, seconds: float, tokenizer, dump_path: Path) -> dict:
    import torch
    if logits.shape != (VOCAB_SIZE,) or logits.dtype != np.float16:
        raise ValueError(f"{row['id']}: expected fp16 [{VOCAB_SIZE}], got {logits.shape}/{logits.dtype}")
    finite = bool(np.isfinite(logits).all())
    if not finite:
        raise FloatingPointError(f"{row['id']}: nonfinite full-vocabulary logits")
    gathered = logits[row["label_ids"]].astype(np.float32)
    temperature = float(row.get("temperature", 1.0))
    p = (torch.from_numpy(gathered) / temperature).softmax(dim=-1).numpy().copy()
    if not np.isfinite(p).all():
        raise FloatingPointError(f"{row['id']}: nonfinite letter probabilities")
    delta = np.abs(p.astype(np.float64) - np.asarray(row["p_oracle"], np.float64))
    oracle_raw = np.asarray(row["raw_logits"], np.float64)
    raw_delta = np.abs(gathered.astype(np.float64) - oracle_raw)
    full_id = int(logits.argmax())
    option = int(p.argmax())
    dump_path = local_path(dump_path)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(dump_path, logits, allow_pickle=False)
    result = {"id": row["id"], "primitive": row["primitive"], "zoo_only": bool(row.get("zoo_only", False)),
            "tokens": len(row["ids"]), "slot": row["slot"], "nopts": len(row["labels"]),
            "labels": row["labels"], "label_ids": row["label_ids"], "temperature": temperature,
            "raw_logits": gathered.tolist(), "full_logits_npy": str(dump_path),
            "full_logits_sha256": sha256(dump_path), "full_logits_dtype": str(logits.dtype),
            "full_logits_shape": list(logits.shape), "full_vocab_argmax_id": full_id,
            "full_vocab_argmax_text": tokenizer.decode([full_id]),
            "full_vocab_argmax_is_label": full_id in row["label_ids"],
            "option_argmax": option, "letter_argmax": option, "option_argmax_label": row["labels"][option],
            "oracle_argmax": row["argmax"], "oracle_argmax_label": row["labels"][row["argmax"]],
            "argmax_agrees": option == row["argmax"], "p": p.tolist(), "p_oracle": row["p_oracle"],
            "oracle_margin": row["top2_margin"],
            "max_abs_delta_p": float(delta.max()), "mean_abs_delta_p": float(delta.mean()),
            "max_abs_delta_raw_logits": float(raw_delta.max()), "mean_abs_delta_raw_logits": float(raw_delta.mean()),
            "full_logits_min": float(logits.min()), "full_logits_max": float(logits.max()),
            "full_logits_absmax": float(np.max(np.abs(logits.astype(np.float32)))),
            "label_logits_min": float(gathered.min()), "label_logits_max": float(gathered.max()),
            "label_logits_absmax": float(np.max(np.abs(gathered))),
            "oracle_logits_min": float(oracle_raw.min()), "oracle_logits_max": float(oracle_raw.max()),
            "oracle_logits_absmax": float(np.max(np.abs(oracle_raw))),
            "finite": finite, "nonconstant": bool(logits.max() > logits.min()),
            "wall_seconds_contended": seconds,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if "p_author" in row:
        author = np.asarray(row["p_author"], np.float64)
        author_delta = np.abs(p.astype(np.float64) - author)
        author_raw = np.asarray(row.get("author_raw_logits", row["raw_logits"]), np.float64)
        author_argmax = row.get("author_argmax", row["argmax"])
        author_raw_delta = np.abs(gathered.astype(np.float64) - author_raw)
        ordered = sorted(row["p_author"], reverse=True)
        result.update(author_argmax=author_argmax, author_argmax_label=row["labels"][author_argmax],
                      author_argmax_agrees=option == author_argmax, p_author=row["p_author"],
                      author_margin=float(ordered[0]-ordered[1]), author_raw_logits=author_raw.tolist(),
                      author_max_abs_delta_p=float(author_delta.max()), author_mean_abs_delta_p=float(author_delta.mean()),
                      author_max_abs_delta_raw_logits=float(author_raw_delta.max()), author_mean_abs_delta_raw_logits=float(author_raw_delta.mean()),
                      author_logits_min=float(author_raw.min()), author_logits_max=float(author_raw.max()),
                      author_logits_absmax=float(np.abs(author_raw).max()), author_probability_sum=float(author.sum()))
    if "raw_logprobs" in row:
        lp = torch.from_numpy(logits.astype(np.float32)).log_softmax(dim=-1).numpy()[row["label_ids"]]
        result.update(license=row.get("license"), oracle="A: author helper / MLX bf16",
                      raw_logprobs=lp.tolist(), author_raw_logprobs=row["raw_logprobs"],
                      max_abs_delta_logprobs=float(np.max(np.abs(lp.astype(np.float64)-row["raw_logprobs"]))))
    if row.get("kind") == "noul":
        p_yes = min(max(float(p[0]), 1e-4), 1-1e-4)
        noul = 1 / (1 + math.exp(-math.log(p_yes / (1-p_yes)) / 1.829074))
        result.update(p_yes=float(p[0]), noul=noul, author_noul=row["noul"], abs_delta_noul=abs(noul-row["noul"]))
    if row.get("kind") == "score" and "expected_index" in row:
        score = sum(i*float(value) for i, value in enumerate(p))
        result.update(expected_index=score, author_expected_index=row["expected_index"],
                      abs_delta_expected_index=abs(score-row["expected_index"]))
    for key in ("kind", "request_id", "evidence_only"):
        if key in row:
            result[key] = row[key]
    return result


def run_worker(args: argparse.Namespace) -> int:
    """One process owns one AIModel; persist every completed row immediately."""
    import torch
    import coreai.runtime as rt
    from transformers import AutoConfig, AutoTokenizer
    from coreai_models.models.macos import qwen3_5 as q
    from coreai_models.models.macos.qwen3_5_config import register_qwen3_5_configs

    global MAX_PROMPT_EVALUATIONS
    config = json.loads(local_path(args.worker).read_text())
    MAX_PROMPT_EVALUATIONS = config.get("max_prompt_evaluations", 15)
    configure_work(config["work_dir"])
    transcript = local_path(config["transcript"])
    fixtures = json.loads(local_path(config["fixtures"]).read_text())
    rows = [fixtures["rows"][index] for index in config["indices"]]
    if config["prompt_format"] == "decision-function":
        # MLX's nested text-only config lacks vision_config; AutoConfig 4.57.6
        # fails serializing it. This accepts unchanged source text attributes.
        raw = json.loads((local_path(config["snapshot"]) / "config.json").read_text())
        hf_text = SimpleNamespace(**raw.get("text_config", raw))
        expected_layers = 24
    else:
        register_qwen3_5_configs()
        raw = AutoConfig.from_pretrained(local_path(config["snapshot"]), local_files_only=True)
        hf_text = raw.text_config
        expected_layers = 32
    expected_layers = config.get("num_layers") or expected_layers
    cfg = q.qwen3_5_config_from_hf(hf_text, config["max_ctx"], None)
    if cfg.vocab_size != VOCAB_SIZE or cfg.num_hidden_layers != expected_layers:
        raise ValueError(f"expected full-depth Qwen3.5 text tower with {expected_layers} layers")
    tokenizer = AutoTokenizer.from_pretrained(local_path(config["tokenizer"]), local_files_only=True)
    validate_tokenization(rows, tokenizer, config["prompt_format"], config["label_style"])
    record = {"schema": "coreai-letter-readout-session/1", "session": config["session"],
              "mode": config["mode"], "indices": config["indices"], "pid": os.getpid(),
              "asset": config["asset"], "snapshot": config["snapshot"],
              "started_at": utc_now(), "phase": "starting", "rows": [], "runtime_calls": 0,
              "planned_prompt_evaluations": len(rows) + 1,
              "planned_runtime_calls": sum(len(row["ids"]) for row in rows) + len(rows[0]["ids"]),
              "fresh_state_every_evaluation": True, "result": "RUNNING"}
    if len(rows) + 1 > MAX_PROMPT_EVALUATIONS or record["planned_runtime_calls"] > MAX_CALLS:
        raise ValueError(f"session exceeds {MAX_PROMPT_EVALUATIONS} evaluations / {MAX_CALLS} calls including reset")
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
        model = await rt.AIModel.load(local_path(config["asset"]), rt.SpecializationOptions.default())
        function = model.load_function("main")
        record["load_wall_seconds_contended"] = time.monotonic() - load_start

        async def row_logits(row: dict) -> np.ndarray:
            assert row["slot"] == len(row["ids"]) - 1 and len(row["ids"]) <= config["max_ctx"]
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
            if logits.shape != (1, 1, VOCAB_SIZE) or logits.dtype != np.float16:
                raise ValueError(f"expected fp16 logits [1,1,{VOCAB_SIZE}], got {logits.shape}/{logits.dtype}")
            return logits[0, -1].copy()

        first = None
        for row in rows:
            row_start = time.monotonic()
            logits = await row_logits(row)
            if first is None:
                first = logits.copy()
            record["phase"] = "readout"
            dump = RUN / f"results/readout_logits/{config['session']}/{safe_id(row['id'])}.npy"
            result = row_record(logits, row, time.monotonic() - row_start, tokenizer, dump)
            record["rows"].append(result)
            write_json(transcript, record)
            print(f"{row['id']}: {len(row['ids'])} tokens, letter {result['option_argmax_label']}/"
                  f"{result['oracle_argmax_label']}, max|dp|={result['max_abs_delta_p']:.8f}, "
                  f"full={result['full_vocab_argmax_id']}:{result['full_vocab_argmax_text']!r}", flush=True)
        again = await row_logits(rows[0])
        repeat_path = RUN / f"results/readout_logits/{config['session']}/reset-{safe_id(rows[0]['id'])}.npy"
        np.save(repeat_path, again, allow_pickle=False)
        same = first.tobytes() == again.tobytes()
        record["reset_check"] = {"row": rows[0]["id"], "identical": same, "bit_identical": same,
                                  "max_abs_diff": float(np.max(np.abs(first.astype(np.float32) - again.astype(np.float32)))),
                                  "first_full_logits_npy": record["rows"][0]["full_logits_npy"],
                                  "final_full_logits_npy": str(repeat_path), "final_full_logits_sha256": sha256(repeat_path),
                                  "fresh_state_each_row": True, "same_process": True}
        record["phase"] = "complete"
        record["result"] = "PASS" if same else "FAIL"

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
        record["wall_seconds_contended"] = time.monotonic() - start
        record["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(transcript, record)
    return 0 if record["result"] == "PASS" else 2


def session_plan(rows: list[dict]) -> list[dict]:
    kit = [i for i, row in enumerate(rows) if not row.get("zoo_only", False)]
    wide = [i for i, row in enumerate(rows) if row.get("zoo_only", False)]
    plans, batch, calls = [], [], 0
    for index in kit:
        size = len(rows[index]["ids"])
        repeat = len(rows[batch[0]]["ids"]) if batch else size
        if batch and (len(batch) + 2 > MAX_PROMPT_EVALUATIONS or calls + size + repeat > MAX_CALLS):
            plans.append({"kind": "kit", "indices": batch})
            batch, calls = [], 0
        batch.append(index)
        calls += size
    if batch:
        plans.append({"kind": "kit", "indices": batch})
    plans.extend({"kind": "wide", "indices": [index]} for index in wide)
    plans.append({"kind": "global_reset", "indices": [0]})
    for plan in plans:
        selected = [rows[index] for index in plan["indices"]]
        plan["prompt_evaluations"] = len(selected) + 1
        plan["runtime_calls"] = sum(len(row["ids"]) for row in selected) + len(selected[0]["ids"])
        if plan["prompt_evaluations"] > MAX_PROMPT_EVALUATIONS or plan["runtime_calls"] > MAX_CALLS:
            raise ValueError("cannot plan session inside resource caps")
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
    in_label = sum(row["full_vocab_argmax_is_label"] for row in rows)
    expectation = 0.02 if record["mode"] == "fp16" else 0.05
    result = {"rows": len(rows), "expected_rows": expected,
            "argmax_agreement": sum(row["argmax_agrees"] for row in rows),
            "margin_ge_0_02_rows": len(strict),
            "margin_ge_0_02_argmax_agreement": sum(row["argmax_agrees"] for row in strict),
            "near_ties": [{"id": row["id"], "oracle_margin": row["oracle_margin"],
                            "oracle_argmax_label": row["oracle_argmax_label"], "option_argmax_label": row["option_argmax_label"],
                            "argmax_agrees": row["argmax_agrees"], "p": row["p"], "p_oracle": row["p_oracle"]} for row in near],
            "high_margin_disagreements": [row for row in strict if not row["argmax_agrees"]],
            "full_vocab_argmax_is_label": in_label,
            "full_vocab_argmax_label_fraction": in_label / len(rows) if rows else None,
            "off_label_rows": [{key: row[key] for key in ("id", "full_vocab_argmax_id", "full_vocab_argmax_text", "labels", "label_ids")}
                               for row in rows if not row["full_vocab_argmax_is_label"]],
            "max_abs_delta_p": max((row["max_abs_delta_p"] for row in rows), default=None),
            "mean_of_row_mean_abs_delta_p": float(np.mean([row["mean_abs_delta_p"] for row in rows])) if rows else None,
            "max_abs_delta_raw_logits": max((row["max_abs_delta_raw_logits"] for row in rows), default=None),
            "finite_all": bool(rows) and all(row["finite"] for row in rows),
            "nonconstant_all": bool(rows) and all(row["nonconstant"] for row in rows),
            "reset_identical": record.get("reset_check", {}).get("identical", False),
            "session_resets_identical": bool(sessions) and all(session.get("reset_check", {}).get("identical", False) for session in sessions),
            "provisional_int8hu_max_delta_p_expectation": 0.05,
            "provisional_max_delta_p_expectation": expectation,
            "provisional_expectation_exceeded": any(row["max_abs_delta_p"] > expectation for row in rows),
            "provisional_expectation_policy": "Exceedance is recorded for supervisor judgment; no investigation or new tolerance.",
            "max_abs_delta_noul": max((row["abs_delta_noul"] for row in rows if "abs_delta_noul" in row), default=None),
            "max_process_peak_rss_bytes": max((session.get("peak_rss_bytes", 0) for session in sessions), default=None),
            "worst_five": sorted(rows, key=lambda row: row["max_abs_delta_p"], reverse=True)[:5]}
    if record.get("prompt_format") == "decision-function":
        result["provisional_expectation_policy"] = "Exceedance is recorded, not a hard failure; no investigation or new tolerance."
    author_rows = [row for row in rows if "p_author" in row]
    if author_rows:
        strict_author = [row for row in author_rows if row["author_margin"] >= 0.02]
        result.update(author_rows=len(author_rows),
                      author_argmax_agreement=sum(row["author_argmax_agrees"] for row in author_rows),
                      author_margin_ge_0_02_rows=len(strict_author),
                      author_margin_ge_0_02_argmax_agreement=sum(row["author_argmax_agrees"] for row in strict_author),
                      author_max_abs_delta_p=max(row["author_max_abs_delta_p"] for row in author_rows),
                      author_mean_of_row_mean_abs_delta_p=float(np.mean([row["author_mean_abs_delta_p"] for row in author_rows])),
                      author_max_abs_delta_raw_logits=max(row["author_max_abs_delta_raw_logits"] for row in author_rows),
                      author_near_ties=[row for row in author_rows if row["author_margin"] < 0.02],
                      author_disagreements=[row for row in author_rows if not row["author_argmax_agrees"]],
                      author_worst_five=sorted(author_rows, key=lambda row: row["author_max_abs_delta_p"], reverse=True)[:5])
    return result


def controller(args: argparse.Namespace) -> int:
    if RUNTIME_STOP.exists():
        raise RuntimeError(f"the round runtime stop condition already fired: {RUNTIME_STOP}")
    bundle, fixtures_path = local_path(args.bundle), local_path(args.fixtures)
    transcript = local_path(args.transcript or RUN / f"results/readout_{args.mode}.json")
    deadline = deadline_for(args.deadline_epoch)
    if not args.snapshot:
        raise ValueError("pass --snapshot to the pinned source config/tokenizer snapshot")
    snapshot = local_path(args.snapshot)
    fx = json.loads(fixtures_path.read_text())
    rows = validate_rows(fx, args.max_ctx, args.label_style, args.prompt_format, args.temperature)
    if snapshot.name != fx["source"]["revision"]:
        raise ValueError("fixture/source snapshot revision mismatch")
    meta = json.loads((bundle / "metadata.json").read_text())
    assert meta["language"]["vocab_size"] == VOCAB_SIZE
    assert meta["language"]["max_context_length"] == args.max_ctx == 4096
    aimodel = local_path(bundle / meta["assets"]["main"])
    out_dir = local_path(args.aot_dir or bundle.parent / "aotc")
    plans = session_plan(rows)
    record = {"schema": "coreai-letter-readout-gate/1", "mode": args.mode, "bundle": str(bundle),
              "label_style": args.label_style, "prompt_format": args.prompt_format,
              "fixtures": str(fixtures_path), "fixtures_sha256": sha256(fixtures_path),
              "snapshot": str(snapshot), "max_ctx": args.max_ctx, "temperature": args.temperature,
              "license": fx.get("license"), "template": args.prompt_format,
              "runtime": "coreai Python runtime, AOT h16c GPU asset, SpecializationOptions.default()",
              "source": fx["source"], "vocab_size": VOCAB_SIZE,
              "environment": {"python": sys.version, "deadline_epoch": deadline, "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve())},
              "started_at": utc_now(), "timing": "contended; correctness only, no performance claim",
              "caps": {"prompt_evaluations_per_process_including_reset": MAX_PROMPT_EVALUATIONS,
                       "calls_per_process_including_reset": MAX_CALLS},
              "session_plan": plans, "rows": [], "sessions": [], "result": "RUNNING"}
    write_json(transcript, record)
    start = time.monotonic()
    try:
        if args.aot_asset:
            asset = local_path(args.aot_asset)
            if not asset.is_dir() or not asset.name.endswith(".h16c.aimodelc"):
                raise ValueError("--aot-asset must be an existing h16c .aimodelc directory")
            compile_record = {"reused": True, "result": "PASS", "asset": str(asset),
                              "source": str(aimodel), "selection": "explicit existing AOT asset", "flags": AOT_FLAGS}
        else:
            asset, compile_record = aot_compile(aimodel, out_dir, args.mode, deadline)
        record["asset"], record["aot"] = str(asset), compile_record
        write_json(transcript, record)
        print(f"AOT asset ready: {asset}", flush=True)
        for number, plan in enumerate(plans):
            remaining(deadline)
            if RUNTIME_STOP.exists():
                raise RuntimeError(f"runtime stop condition: {RUNTIME_STOP}")
            succeeded = False
            for attempt in (1, 2):
                session_name = f"{args.mode}-{number:02d}-{plan['kind']}-a{attempt}"
                session_path = RUN / f"results/readout_sessions/{session_name}.json"
                config_path = RUN / f"results/readout_sessions/{session_name}.config.json"
                log_path = RUN / f"logs/readout-{session_name}.log"
                pid_path = RUN / f"results/readout_pids/{session_name}.json"
                config = {"session": session_name, "mode": args.mode, "asset": str(asset),
                          "label_style": args.label_style, "prompt_format": args.prompt_format,
                          "snapshot": str(snapshot), "tokenizer": str(bundle / "tokenizer"),
                          "fixtures": str(fixtures_path), "indices": plan["indices"], "max_ctx": args.max_ctx,
                          "deadline": deadline, "transcript": str(session_path), "work_dir": str(RUN),
                          "num_layers": args.num_layers, "max_prompt_evaluations": args.max_prompt_evaluations}
                write_json(config_path, config)
                call_start = time.monotonic()
                command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(config_path)]
                with log_path.open("wb") as output:
                    result = foreground(command, deadline, pid_path, stdout=output, stderr=subprocess.STDOUT)
                returncode = result.returncode
                timed_out = alarm_returncode(returncode)
                session = json.loads(session_path.read_text()) if session_path.exists() else {
                    "session": session_name, "mode": args.mode, "rows": [], "result": "ERROR", "phase": "starting"}
                session.update({"kind": plan["kind"], "transcript": str(session_path), "log": str(log_path),
                                "command": command, "pid_receipt": str(pid_path), "returncode": returncode,
                                "timed_out": timed_out, "process_wall_seconds_contended": time.monotonic() - call_start})
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
                        original_path = local_path(canonical[rows[0]["id"]]["full_logits_npy"])
                        final_path = local_path(session["reset_check"]["final_full_logits_npy"])
                        original = np.load(original_path, allow_pickle=False)
                        final = np.load(final_path, allow_pickle=False)
                        same = original.dtype == final.dtype and original.shape == final.shape and original.tobytes() == final.tobytes()
                        record["reset_check"] = {"row": rows[0]["id"], "identical": same, "bit_identical": same,
                                                 "max_abs_diff": float(np.max(np.abs(original.astype(np.float32) - final.astype(np.float32)))),
                                                 "first_full_logits_npy": str(original_path), "final_full_logits_npy": str(final_path),
                                                 "final_session": session_name, "cross_process": True,
                                                 "local_resets_in_every_session": True}
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
        hard_passed = (value["rows"] == value["expected_rows"] and value["finite_all"]
                       and value["margin_ge_0_02_argmax_agreement"] == value["margin_ge_0_02_rows"]
                       and value["reset_identical"] and value["session_resets_identical"])
        if args.prompt_format == "chat":
            hard_passed = hard_passed and value["nonconstant_all"] and value["full_vocab_argmax_label_fraction"] >= args.min_label_fraction
        record["hard_gates_result"] = "PASS" if hard_passed else "FAIL"
        record["result"] = "FAIL" if not hard_passed else (
            "REVIEW_REQUIRED" if value["provisional_expectation_exceeded"] and args.prompt_format == "chat" else "PASS")
    except Exception as exc:
        record["result"] = "PARTIAL"
        record["error"] = {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        print(record["error"]["traceback"], file=sys.stderr, flush=True)
    finally:
        record["summary"] = summary(record, len(rows))
        record["wall_seconds_contended"] = time.monotonic() - start
        record["completed_at"] = utc_now()
        write_json(transcript, record)
    value = record["summary"]
    print(f"{record['result']}: {value['argmax_agreement']}/{value['rows']} argmax; "
          f"margin>=0.02 {value['margin_ge_0_02_argmax_agreement']}/{value['margin_ge_0_02_rows']}; "
          f"max|dp|={value['max_abs_delta_p']}; full-vocab label fraction={value['full_vocab_argmax_label_fraction']}", flush=True)
    return 0 if record["result"] == "PASS" else 1


def main() -> int:
    global MAX_PROMPT_EVALUATIONS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle", nargs="?")
    parser.add_argument("fixtures", nargs="?")
    parser.add_argument("--mode", choices=("fp16", "int8hu"), default="int8hu")
    parser.add_argument("--label-style", choices=("plain", "space-prefixed", "bare"), default="plain",
                        help="plain A-P preserves APUS; decision-function uses space-prefixed A-Z")
    parser.add_argument("--prompt-format", "--template", dest="prompt_format", choices=("chat", "decision-function"), default="chat",
                        help="fixture form and acceptance policy; ids are always used verbatim, no template applied")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num-layers", type=int, help="override expected text depth; APUS=32 and decision-function=24 by default")
    parser.add_argument("--max-prompt-evaluations", type=int, default=15, help="includes per-process reset; OpenJev 27B uses 6")
    parser.add_argument("--min-label-fraction", type=float, default=0.90, help="APUS default; OpenJev records this with threshold 0")
    parser.add_argument("--snapshot", help="pinned local HF snapshot containing source config and tokenizer")
    parser.add_argument("--work-dir", type=Path, help="logs, sessions and logits; default beside transcript")
    parser.add_argument("--aot-asset", type=Path, help="reuse this existing h16c GPU .aimodelc without compiling")
    parser.add_argument("--transcript")
    parser.add_argument("--aot-dir")
    parser.add_argument("--max-ctx", type=int, default=4096)
    parser.add_argument("--deadline-epoch", type=float, help="absolute wall-clock deadline as Unix seconds")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--compile-only", action="store_true", help="AOT compile without Python-runtime sessions")
    parser.add_argument("--plan-only", action="store_true", help="validate fixture and print bounded session plan; no runtime")
    args = parser.parse_args()
    MAX_PROMPT_EVALUATIONS = args.max_prompt_evaluations
    if not args.temperature > 0 or MAX_PROMPT_EVALUATIONS < 2:
        parser.error("positive temperature and at least two evaluations required")
    if args.worker:
        return run_worker(args)
    output = local_path(args.transcript or f"readout-{args.mode}.json")
    args.transcript = str(output)
    configure_work(args.work_dir or output.parent / ("." + output.stem + "-work"))
    if not args.bundle or (not args.fixtures and not args.compile_only):
        parser.error("bundle and fixtures required (fixtures may be omitted with --compile-only)")
    if args.plan_only:
        fx = json.loads(local_path(args.fixtures).read_text())
        print(json.dumps(session_plan(validate_rows(fx, args.max_ctx, args.label_style, args.prompt_format, args.temperature)), indent=1))
        return 0
    if args.compile_only:
        bundle = local_path(args.bundle)
        metadata = json.loads((bundle / "metadata.json").read_text())
        asset, receipt = aot_compile(local_path(bundle / metadata["assets"]["main"]),
                                     local_path(args.aot_dir or bundle.parent / "aotc"), args.mode,
                                     deadline_for(args.deadline_epoch))
        print(json.dumps({"asset": str(asset), "aot": receipt}, indent=1))
        return 0
    return controller(args)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--exec-child"]:
        exec_child(sys.argv[2:])
    else:
        sys.exit(main())
