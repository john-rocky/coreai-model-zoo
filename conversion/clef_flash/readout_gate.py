#!/usr/bin/env python3
"""Readout gate: the clef-flash decoder bundle on the Mac GPU, read through the author's fp32 head, vs the author's fp32 oracle.

Every oracle run is fed to the bundle's one static-S function `main` (AOT h16c `.aimodelc`,
`SpecializationOptions.default()`, no JIT) from fresh zero states, in S-token chunks: call k gets
ids[kS : kS + S] with position_ids 0..kS+S-1, the last call padded with <|endoftext|> (248044) and
the padded positions' hidden rows discarded (`parity_decoder_torch.py`'s chunk order). The ids are
the oracle's (`oracle/records_oracle.json`) with the <|image_pad|> block mapped to V + k by
`qwen3_5_clef_decoder.host_static_inputs` (n_image_max 1024); `image_embeds` = the oracle's fp32
tower output (`oracle/npz/<id>__<arm>.npz`) cast to fp16 and zero-padded to 1024 rows. So this
isolates the decoder graph. The fp16 hidden rows [T, 4096] are cast to fp32 and go through the
checkpoint's own `JointSchemaHead` (`parity_decoder_torch.AuthorHead`: path import of the snapshot's
joint_schema_model.py, sha256 pinned; joint_head.safetensors and the untied lm_head.weight in
fp32, CPU); the per-question softmax is compared with the oracle's.

photo_01/native (N = 40 x 30 = 1,200 image rows) exceeds the graph's 1024-row buffer and is not run:
the shipped path sends a fixed grid (the transcript lists it under `skipped`).

Bar, fixed before any result: every run present; per-question argmax = the oracle's on every
question whose oracle top-2 margin is above 0.02 (the near-ties, margin <= 0.02, are listed apart);
max |dp| <= 0.02 over every option of every question (near-ties included); the mean over runs of
the run's mean |dp| (over all of its question-option entries) <= 0.002; every process re-runs its
first run at the end and reproduces its hidden rows bit for bit (state reset proof); every hidden
value finite.

Red arm (`--red`, its own process): img_07/g448 and img_01/g256, each unperturbed and with
`image_embeds` zeroed (ids, image_rc and the rope shift unchanged): argmax changes and max |dp| of
the zeroed run against the unperturbed one. A graph that ignored its image input would not move.

Process split: the Python runtime leaks one IOSurface per call, so a process takes at most 40 runs
+ the reset re-run. The driver runs the processes one after another; each worker writes
`<work>/shard_NN.json` (per run: T, calls, ms) and `.npz` (the fp16 hidden rows and every call's
wall ms); the driver scores them with the head and writes the transcript. The GPU is shared with
other sessions: the times are contended reference values.

`--subset s64` = the 80-run set of the chunk-width trial (own_text, own_text_long, own_json, the
g448 arms of own_image and cc0_photo, the first 38 SemIf records); `s64rest` = the other 133 runs;
`list` = the [id, arm] pairs of `--runs-file` (a JSON {"runs": [[id, arm], ..], "why": ..}).
`--compare-with <transcript>` adds the per-run p difference, the hidden rows' position cosine
against the oracle and the timing side by side for the runs the two transcripts share.

Every run records the position cosine of its hidden rows against the oracle's: the minimum, and
how many positions fall below 0.99 / 0.9 / 0.5 (image rows counted apart), with the lowest few.

`merge` joins transcripts of the same bundle over disjoint run sets (one gate split across
invocations) into one: every run entry is kept as its invocation scored it, tagged with the
invocation; the summary, the checks and the per-group tables are recomputed over the union.

`--oracle <dir>` reads another oracle directory of the same layout (`records_oracle.json` + `npz/`),
e.g. the held-out record set (`oracle_clef.py --fixtures .../fixtures/heldout --out-dir
.../oracle/heldout`); the default is the fixture oracle. A set without SemIf records keeps no SemIf
hidden rows, and the photo_01/native skip applies only where that run exists.

    cd conversion/clef_flash
    HF_HOME=$ZOO_WORK_ROOT/_clefflash/hf HF_HUB_OFFLINE=1 PY=<coreai-models venv>/bin/python
    $PY readout_gate.py run $ZOO_WORK_ROOT/_clefflash/exports/bundles/clef_flash_decode_fp16_pf16 \\
        --red --transcript $ZOO_WORK_ROOT/_clefflash/results/readout_fp16_pf16.json
    $PY readout_gate.py run .../clef_flash_decode_fp16_pf64 --subset s64 \\
        --compare-with .../readout_fp16_pf16.json --transcript .../readout_fp16_pf64.json
    $PY readout_gate.py run .../clef_flash_decode_fp16_pf64 --subset s64rest --tag fp16_pf64_rest \\
        --transcript .../readout_fp16_pf64_rest.json
    $PY readout_gate.py merge .../readout_fp16_pf64.json .../readout_fp16_pf64_rest.json \\
        --transcript .../readout_fp16_pf64_full.json
    $PY readout_gate.py run .../clef_flash_decode_fp16_pf64 --oracle $ZOO_WORK_ROOT/_clefflash/oracle/heldout \\
        --tag heldout_fp16_pf64 --transcript .../readout_heldout_fp16_pf64.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _paths import record_path, work_path  # noqa: E402

LANE = work_path("_clefflash")
os.environ.setdefault("HF_HOME", str(LANE / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ORACLE = LANE / "oracle"
VOCAB = 248320
HIDDEN = 4096
VISION_START, IMAGE_PAD, PAD_ID = 248053, 248056, 248044
N_IMAGE_MAX = 1024
BAR = {"max_abs_dp": 0.02, "mean_of_run_mean_abs_dp": 0.002, "near_tie_top2_margin": 0.02}
RUNS_PER_PROCESS = 40                               # + the reset re-run
RED_RUNS = (("img_07", "g448"), ("img_01", "g256"))
KEEP_HIDDEN = (("own_t01", "text"), ("own_t14", "text"), ("img_01", "g256"), ("img_04", "g448"),
               ("photo_02", "g448"))                # + the first SemIf record
STATE_SHAPES = {"keyCache": [8, 1, 4, -1, 256], "valueCache": [8, 1, 4, -1, 256],
                "convState": [24, 1, 8192, 3], "recState": [24, 1, 32, 128, 128]}
OTHER_GPU = re.compile(r"yardstick|litert|llm-bench|coreai_verify|coreai-build|readout_gate|gate_|parity_")
COS_THRESHOLDS = (0.99, 0.9, 0.5)                   # position cosine vs the oracle, counted per run
LOWEST_POSITIONS = 6


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(path: Path) -> dict:
    files = sorted(p for p in path.rglob("*") if p.is_file())
    per = {str(p.relative_to(path)): sha256_file(p) for p in files}
    tree = hashlib.sha256("".join(f"{k}\0{v}\n" for k, v in per.items()).encode()).hexdigest()
    return {"path": str(path), "bytes": sum(p.stat().st_size for p in files), "tree_sha256": tree, "files": per}


def set_oracle(path) -> None:
    """Read another oracle directory (records_oracle.json + npz/) for the rest of the process."""
    global ORACLE
    ORACLE = Path(path).expanduser().resolve()


def load_oracle() -> tuple[dict, list[dict]]:
    doc = json.loads((ORACLE / "records_oracle.json").read_text())
    assert doc["complete"], "oracle incomplete"
    return doc, doc["rows"]


def contract(S: int) -> dict:
    return {"inputs": {"input_ids": [[1, S], "int32"], "position_ids": [[1, -1], "int32"],
                       "image_embeds": [[N_IMAGE_MAX, HIDDEN], "float16"], "image_rc": [[N_IMAGE_MAX, 2], "int32"],
                       "rope_shift_start": [[1], "int32"], "rope_shift_amount": [[1], "int32"]},
            "outputs": {"hidden": [[1, S, HIDDEN], "float16"]},
            "states": {n: [s, "float16"] for n, s in STATE_SHAPES.items()}}


# --------------------------------------------------------------------------- #
# Worker: one process, <= 40 runs + the reset re-run
# --------------------------------------------------------------------------- #
async def maybe(x):
    return await x if inspect.isawaitable(x) else x


def dsc(d) -> list:
    return [[int(x) for x in d.shape], str(d.dtype).split(".")[-1]]


def fn_desc(fn) -> dict:
    d = fn.desc
    return {"function": getattr(d, "name", None),
            "inputs": {n: dsc(d.input_descriptor(n)) for n in d.input_names},
            "outputs": {n: dsc(d.output_descriptor(n)) for n in d.output_names},
            "states": {n: dsc(d.state_descriptor(n)) for n in d.state_names}}


def check_contract(desc: dict, S: int) -> list[str]:
    want = contract(S)
    bad = []
    for part in ("inputs", "outputs", "states"):
        if set(desc[part]) != set(want[part]):
            bad.append(f"{part} names {sorted(desc[part])} != {sorted(want[part])}")
            continue
        for n, w in want[part].items():
            if [list(desc[part][n][0]), desc[part][n][1]] != [list(w[0]), w[1]]:
                bad.append(f"{part} {n}: {desc[part][n]} != {w}")
    return bad


def worker(spec_path: Path) -> int:
    import coreai.runtime as rt
    import torch
    from qwen3_5_clef_decoder import host_static_inputs

    spec = json.loads(spec_path.read_text())
    if spec.get("oracle"):
        set_oracle(spec["oracle"])
    _, rows = load_oracle()
    orc = {(r["id"], r["arm"]): r for r in rows}
    S, max_ctx = int(spec["chunk"]), int(spec["max_ctx"])

    def nd(a):
        return rt.NDArray(np.ascontiguousarray(a))

    out: dict = {"spec": spec, "pid": os.getpid(), "runs": [], "started": time.time()}
    store: dict[str, np.ndarray] = {}

    async def go() -> None:
        t0 = time.perf_counter()
        model = await maybe(rt.AIModel.load(spec["aimodelc"], rt.SpecializationOptions.default()))
        t1 = time.perf_counter()
        fn = await maybe(model.load_function("main"))
        t2 = time.perf_counter()
        out["load_seconds"] = t2 - t0
        out["load_split_seconds"] = {"model": t1 - t0, "main": t2 - t1}
        out["function_names"] = list(getattr(model, "function_names", []) or [])
        desc = fn_desc(fn)
        out["descriptor"] = desc
        bad = check_contract(desc, S)
        out["contract_mismatch"] = bad
        if bad:
            raise SystemExit(f"descriptor differs from the contract: {bad}")

        def fresh_state() -> dict:
            return {n: nd(np.zeros([max_ctx if s < 0 else s for s in shape], np.dtype(dt)))
                    for n, (shape, dt) in desc["states"].items()}

        async def one(run: list) -> tuple[dict, dict[str, np.ndarray]]:
            rid, arm, variant = run
            o = orc[(rid, arm)]
            t_run = time.perf_counter()
            mh = o.get("merged_hw")
            hw = tuple(mh) if (mh and arm != "text") else None
            ids, rc, start, amount = host_static_inputs(o["ids"], hw, VOCAB, IMAGE_PAD, VISION_START, N_IMAGE_MAX)
            emb = np.zeros((N_IMAGE_MAX, HIDDEN), np.float16)
            n_img = 0
            if hw is not None:
                n_img = hw[0] * hw[1]
                if int(start[0]) != o["token_offset"] + 1 + n_img or int(amount[0]) != o["rope_shift_amount"]:
                    raise SystemExit(f"{rid}/{arm}: host static inputs differ from the oracle's layout")
                if variant != "embeds_zero":
                    e = np.load(ORACLE / "npz" / f"{rid}__{arm}.npz")["image_embeds"]
                    assert e.shape == (n_img, HIDDEN), (e.shape, hw)
                    emb[:n_img] = e.astype(np.float16)
            ids = ids.numpy().astype(np.int32)
            T = len(ids)
            n_calls = -(-T // S)
            ids_p = np.full(n_calls * S, PAD_ID, np.int32)
            ids_p[:T] = ids
            static = {"image_embeds": nd(emb), "image_rc": nd(rc.numpy().astype(np.int32)),
                      "rope_shift_start": nd(start.numpy().astype(np.int32)),
                      "rope_shift_amount": nd(amount.numpy().astype(np.int32))}
            state = fresh_state()
            hid = np.zeros((n_calls * S, HIDDEN), np.float16)
            call_ms = np.zeros(n_calls, np.float64)
            t_dec = time.perf_counter()
            for c in range(n_calls):
                t1 = time.perf_counter()
                res = await maybe(fn(inputs={"input_ids": nd(ids_p[c * S:(c + 1) * S].reshape(1, S)),
                                             "position_ids": nd(np.arange((c + 1) * S, dtype=np.int32)[None]),
                                             **static}, state=state))
                h = np.asarray(res["hidden"].numpy())
                if h.shape != (1, S, HIDDEN) or h.dtype != np.float16:
                    raise SystemExit(f"{rid}/{arm}: output {h.shape} {h.dtype} != (1, {S}, {HIDDEN}) float16")
                hid[c * S:(c + 1) * S] = h[0]
                call_ms[c] = (time.perf_counter() - t1) * 1e3
            dec = time.perf_counter() - t_dec
            rec = {"id": rid, "arm": arm, "variant": variant, "tokens": T, "calls": n_calls,
                   "padded_tokens": n_calls * S, "grid": list(hw) if hw else None, "n_image_rows": n_img,
                   "start": int(start[0]), "amount": int(amount[0]),
                   "image_tokens": int((ids >= VOCAB).sum()), "decode_seconds": dec,
                   "wall_seconds": time.perf_counter() - t_run,
                   "finite": bool(np.isfinite(hid[:T].astype(np.float32)).all())}
            return rec, {"hidden": hid[:T].copy(), "call_ms": call_ms}

        runs = [list(r) for r in spec["runs"]]
        first = None
        for i, run in enumerate(runs + [runs[0]]):
            rec, arrays = await one(run)
            if i < len(runs):
                key = f"{i:02d}"
                for k, a in arrays.items():
                    store[f"{key}__{k}"] = a
                rec["index"] = i
                out["runs"].append(rec)
                if i == 0:
                    first = arrays
                cm = arrays["call_ms"]
                print(f"  [{os.getpid()}] {run[0]}/{run[1]}/{run[2]}: {rec['tokens']} tok, {rec['calls']} calls, "
                      f"{rec['wall_seconds']:.2f} s (median {np.median(cm):.1f} ms/call, first {cm[0]:.1f})",
                      flush=True)
            else:
                same = bool(np.array_equal(first["hidden"], arrays["hidden"]))
                diff = float(np.max(np.abs(first["hidden"].astype(np.float32) - arrays["hidden"].astype(np.float32))))
                out["reset_check"] = {"run": run, "bit_equal": same, "hidden_max_abs_diff": diff,
                                      "wall_seconds": rec["wall_seconds"]}
                print(f"  [{os.getpid()}] reset re-run {run[0]}/{run[1]}: bit-equal {same} (max|d| {diff})", flush=True)

    asyncio.run(go())
    out["finished"] = time.time()
    prefix = Path(spec["out"])
    np.savez(prefix.with_suffix(".npz"), **store)
    prefix.with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def other_gpu_processes() -> list[str]:
    ps = subprocess.run(["ps", "-axo", "pid=,etime=,command="], capture_output=True, text=True).stdout
    me = os.getpid()
    # Agent CLIs carry their settings JSON on the command line, and shells their scripts; neither is GPU work.
    skip = ("/.local/bin/claude", "shell-snapshots", "until grep", "zsh -c")
    return [ln.strip()[:200] for ln in ps.splitlines()
            if OTHER_GPU.search(ln) and not ln.strip().startswith(f"{me} ") and not any(s in ln for s in skip)]


def env_record() -> dict:
    import importlib.metadata as md

    v = {"python": sys.version.split()[0], "numpy": np.__version__}
    for p in ("coreai-core", "coreai-torch", "coreai-models", "torch", "safetensors"):
        try:
            v[p] = md.version(p)
        except Exception as e:  # noqa: BLE001
            v[p] = repr(e)
    osb = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    return {"versions": v, "platform": platform.platform(), "macos_build": osb,
            "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                   text=True).stdout.strip(),
            "runtime": "coreai python runtime, AOT h16c GPU .aimodelc, SpecializationOptions.default(), no JIT",
            "head": "the author's JointSchemaHead, fp32, CPU (parity_decoder_torch.AuthorHead)",
            "gpu": "shared with other sessions, no _GPU_LOCK held (times are contended reference values)"}


def split(runs: list, per: int = RUNS_PER_PROCESS) -> list[list]:
    n = math.ceil(len(runs) / per)
    size = math.ceil(len(runs) / n)
    return [runs[i:i + size] for i in range(0, len(runs), size)]


def run_shards(tag: str, shards: list[dict]) -> list[dict]:
    """Run the worker processes one after another; return their JSON records."""
    got = []
    for sp in shards:
        spec_path = Path(sp["out"]).with_suffix(".spec.json")
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(sp, indent=1) + "\n")
        print(f"[{tag}] {Path(sp['out']).name}: {len(sp['runs'])} runs + reset re-run", flush=True)
        t0 = time.monotonic()
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--spec", str(spec_path)])
        wall = time.monotonic() - t0
        got.append(read_shard(sp, proc.returncode, wall))
    return got


def read_shard(sp: dict, returncode: int | None = 0, wall: float | None = None) -> dict:
    js = Path(sp["out"]).with_suffix(".json")
    if returncode != 0 or not js.exists():
        print(f"{Path(sp['out']).name}: worker FAILED (exit {returncode})", flush=True)
        return {"spec": sp, "failed": True, "returncode": returncode, "process_wall_seconds": wall}
    rec = json.loads(js.read_text())
    rec["process_wall_seconds"] = wall if wall is not None else rec["finished"] - rec["started"]
    rec["returncode"] = returncode
    return rec


def collect(shard_recs: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Merge the shards: run records (+ arrays), and the process records."""
    runs, procs, arrays = [], [], {}
    for sr in shard_recs:
        prefix = Path(sr["spec"]["out"])
        entry = {"shard": prefix.name, "runs": [f"{r[0]}/{r[1]}/{r[2]}" for r in sr["spec"]["runs"]],
                 "prompts": len(sr["spec"]["runs"]) + 1}
        if sr.get("failed"):
            entry.update({"failed": True, "returncode": sr["returncode"]})
            procs.append(entry)
            continue
        z = np.load(prefix.with_suffix(".npz"))
        entry.update({"pid": sr["pid"], "load_seconds": sr["load_seconds"],
                      "load_split_seconds": sr["load_split_seconds"], "function_names": sr["function_names"],
                      "process_wall_seconds": sr["process_wall_seconds"], "reset_check": sr["reset_check"],
                      "descriptor": sr["descriptor"], "contract_mismatch": sr["contract_mismatch"],
                      "npz": str(prefix.with_suffix(".npz")), "npz_sha256": sha256_file(prefix.with_suffix(".npz"))})
        procs.append(entry)
        for rec in sr["runs"]:
            key = f"{rec['index']:02d}"
            arrays[(rec["id"], rec["arm"], rec["variant"])] = {"hidden": z[f"{key}__hidden"],
                                                               "call_ms": z[f"{key}__call_ms"]}
            runs.append({**rec, "shard": prefix.name, "npz_key": key, "first_in_process": rec["index"] == 0})
    return runs, procs, arrays


def position_cos(hidden16: np.ndarray, o: dict) -> dict:
    """Position cosine of the hidden rows vs the oracle's last_hidden: counts below each threshold
    (image rows apart) and the lowest positions."""
    from parity_decoder_torch import cos_rows

    z = np.load(ORACLE / "npz" / f"{o['id']}__{o['arm']}.npz")
    ids = np.asarray(z["input_ids"])
    c = cos_rows(hidden16.astype(np.float32), z["last_hidden"])
    img = ids == IMAGE_PAD
    return {"positions": int(c.size),
            "positions_below": {str(t): int((c < t).sum()) for t in COS_THRESHOLDS},
            "positions_below_0.99_image_rows": int(((c < 0.99) & img).sum()),
            "lowest_positions": [{"pos": int(i), "cos": float(c[i]), "id": int(ids[i]), "image_row": bool(img[i])}
                                 for i in np.argsort(c)[:LOWEST_POSITIONS]]}


def score(head, o: dict, hidden16: np.ndarray) -> dict:
    """Hidden (fp16 -> fp32) vs the oracle's last_hidden, and the author's head on it vs the oracle's probs."""
    import torch
    from parity_decoder_torch import compare, probs_of

    z = np.load(ORACLE / "npz" / f"{o['id']}__{o['arm']}.npz")
    ids = [int(x) for x in z["input_ids"]]
    assert ids == o["ids"], f"{o['id']}/{o['arm']}: npz ids differ from the record's"
    h32 = hidden16.astype(np.float32)
    hid = {**compare(h32, z["last_hidden"]), **position_cos(hidden16, o)}
    logits = head.logits(torch.from_numpy(h32), ids, o)
    probs = probs_of(logits)
    qs, deltas = [], []
    for q, lg, p in zip(o["questions"], logits, probs):
        po = np.asarray(q["probs"], np.float64)
        dp = np.abs(p - po)
        deltas.append(dp)
        qs.append({"question_id": q["question_id"], "type": q["type"], "n_options": len(po),
                   "argmax": int(p.argmax()), "argmax_oracle": q["argmax_index"],
                   "argmax_equal": int(p.argmax()) == q["argmax_index"],
                   "max_abs_dp": float(dp.max()), "mean_abs_dp": float(dp.mean()),
                   "max_abs_dlogit": float(np.abs(lg.double().numpy() - np.asarray(q["logits"], np.float64)).max()),
                   "oracle_top2_margin": q["top2_margin"], "near_tie": bool(q["near_tie"]),
                   "probs": [float(v) for v in p], "probs_oracle": [float(v) for v in po]})
    flat = np.concatenate(deltas)
    return {"hidden": hid, "questions": qs,
            "argmax_all_equal": all(q["argmax_equal"] for q in qs),
            "argmax_equal_non_near_tie": all(q["argmax_equal"] for q in qs if not q["near_tie"]),
            "max_abs_dp": float(flat.max()), "mean_abs_dp": float(flat.mean()),
            "max_abs_dlogit": max(q["max_abs_dlogit"] for q in qs)}


def summarize(runs: list[dict], expected: list[tuple[str, str]]) -> dict:
    have = {(r["id"], r["arm"]) for r in runs}
    qs = [q for r in runs for q in r["questions"]]
    far = [q for q in qs if not q["near_tie"]]
    near = [q for q in qs if q["near_tie"]]
    worst = max(runs, key=lambda r: r["max_abs_dp"]) if runs else None
    ms = np.concatenate([r["_call_ms"] for r in runs]) if runs else np.zeros(0)
    warm = np.concatenate([r["_call_ms"] for r in runs if not r["first_in_process"]] or [np.zeros(0)])
    return {
        "runs": len(runs), "expected_runs": len(expected),
        "missing_runs": [f"{a}/{b}" for a, b in expected if (a, b) not in have],
        "questions": len(qs), "argmax_equal": sum(q["argmax_equal"] for q in qs),
        "questions_non_near_tie": len(far), "argmax_equal_non_near_tie": sum(q["argmax_equal"] for q in far),
        "near_tie_questions": len(near), "argmax_equal_near_tie": sum(q["argmax_equal"] for q in near),
        "max_abs_dp": max((q["max_abs_dp"] for q in qs), default=None),
        "max_abs_dp_non_near_tie": max((q["max_abs_dp"] for q in far), default=None),
        "mean_of_run_mean_abs_dp": float(np.mean([r["mean_abs_dp"] for r in runs])) if runs else None,
        "mean_question_max_abs_dp": float(np.mean([q["max_abs_dp"] for q in qs])) if qs else None,
        "max_abs_dlogit": max((q["max_abs_dlogit"] for q in qs), default=None),
        "min_pos_cos": min((r["hidden"]["min_pos_cos"] for r in runs), default=None),
        "positions": int(sum(r["hidden"]["positions"] for r in runs)) if all("positions" in r["hidden"] for r in runs) else None,
        "positions_below": ({str(t): int(sum(r["hidden"]["positions_below"][str(t)] for r in runs)) for t in COS_THRESHOLDS}
                            if all("positions_below" in r["hidden"] for r in runs) else None),
        "positions_below_0.99_image_rows": (int(sum(r["hidden"]["positions_below_0.99_image_rows"] for r in runs))
                                            if all("positions_below" in r["hidden"] for r in runs) else None),
        "runs_with_positions_below_0.99": ([f"{r['id']}/{r['arm']}" for r in runs if r["hidden"]["positions_below"]["0.99"]]
                                           if all("positions_below" in r["hidden"] for r in runs) else None),
        "hidden_max_abs_diff": max((r["hidden"]["max_abs_diff"] for r in runs), default=None),
        "hidden_max_rel_diff": max((r["hidden"]["rel_max_abs_diff"] for r in runs), default=None),
        "worst_run": None if worst is None else {"id": worst["id"], "arm": worst["arm"],
                                                 "max_abs_dp": worst["max_abs_dp"]},
        "finite_all": all(r["finite"] for r in runs),
        "tokens": int(sum(r["tokens"] for r in runs)), "calls": int(sum(r["calls"] for r in runs)),
        "ms_per_call_median": float(np.median(ms)) if ms.size else None,
        "ms_per_call_warm_median": float(np.median(warm)) if warm.size else None,
    }


def verdict(s: dict, resets_ok: bool) -> tuple[bool, dict]:
    checks = {
        "all_runs": s["runs"] == s["expected_runs"] and not s["missing_runs"],
        "argmax_non_near_tie": s["argmax_equal_non_near_tie"] == s["questions_non_near_tie"],
        "max_abs_dp": s["max_abs_dp"] is not None and s["max_abs_dp"] <= BAR["max_abs_dp"],
        "mean_of_run_mean_abs_dp": (s["mean_of_run_mean_abs_dp"] is not None
                                    and s["mean_of_run_mean_abs_dp"] <= BAR["mean_of_run_mean_abs_dp"]),
        "finite": s["finite_all"],
        "reset_bit_equal_all_processes": resets_ok,
    }
    return all(checks.values()), checks


def timing(runs: list[dict], procs: list[dict], S: int) -> dict:
    """Contended reference times. `warm` = every run but the first of its process."""
    def q(a) -> dict:
        a = np.asarray(a, np.float64)
        return {"n": int(a.size), "median": float(np.median(a)) if a.size else None,
                "p10": float(np.quantile(a, 0.1)) if a.size else None,
                "p90": float(np.quantile(a, 0.9)) if a.size else None}

    warm = [r for r in runs if not r["first_in_process"]]
    loads = [p["load_seconds"] for p in procs if "load_seconds" in p]
    return {
        "contended": True, "S": S,
        "load_seconds": {"first_process": loads[0] if loads else None,
                         "median": float(np.median(loads)) if loads else None, "all": loads},
        "ms_per_call_warm_runs": q(np.concatenate([r["_call_ms"] for r in warm] or [np.zeros(0)])),
        "ms_per_call_all_runs": q(np.concatenate([r["_call_ms"] for r in runs] or [np.zeros(0)])),
        "first_call_of_process_ms": [float(r["_call_ms"][0]) for r in runs if r["first_in_process"]],
        "ms_per_token_warm_runs": q([r["decode_seconds"] * 1e3 / r["tokens"] for r in warm]),
        "run_decode_seconds_warm_runs": q([r["decode_seconds"] for r in warm]),
        "run_wall_seconds_warm_runs": q([r["wall_seconds"] for r in warm]),
        "by_source_arm_warm": {
            k: {"runs": len(v), "median_tokens": float(np.median([r["tokens"] for r in v])),
                "median_calls": float(np.median([r["calls"] for r in v])),
                "ms_per_call_median": float(np.median(np.concatenate([r["_call_ms"] for r in v]))),
                "run_decode_seconds_median": float(np.median([r["decode_seconds"] for r in v]))}
            for k, v in sorted(group(warm).items())},
    }


def group(runs: list[dict]) -> dict:
    out: dict = {}
    for r in runs:
        out.setdefault(f"{r['source']}/{r['arm']}", []).append(r)
    return out


def subset_runs(rows: list[dict], subset: str, runs_file: str | None = None) -> list[tuple[str, str]]:
    keys = [(r["id"], r["arm"]) for r in rows if not (r["id"] == "photo_01" and r["arm"] == "native")]
    if subset == "all":
        return keys
    if subset == "list":
        want = [tuple(k) for k in json.loads(Path(runs_file).read_text())["runs"]]
        bad = [k for k in want if k not in set(keys)] + [k for k in set(want) if want.count(k) > 1]
        if bad or not want:
            raise SystemExit(f"--runs-file: unknown or repeated runs {bad}")
        return want
    src = {(r["id"], r["arm"]): r["source"] for r in rows}
    semif = [k for k in keys if src[k] == "semif_authored144"][:38]
    pick = [k for k in keys if (src[k] in ("own_text", "own_text_long", "own_json"))
            or (src[k] in ("own_image", "cc0_photo") and k[1] == "g448")]
    if subset == "s64rest":
        s64 = set(pick + semif)
        return [k for k in keys if k not in s64]
    return pick + semif


def compare_with(runs: list[dict], arrays: dict, other_path: Path, S: int) -> dict:
    """The same runs in another transcript (another chunk width): p and hidden side by side, and the times."""
    other = json.loads(other_path.read_text())
    o_runs = {(r["id"], r["arm"]): r for r in other["runs"]}
    o_arrays, zs = {}, {}
    for r in other["runs"]:   # a merged transcript names each run's npz; a plain one names it per shard
        npz = r.get("npz") or next((p["npz"] for p in other["processes"]
                                    if p.get("shard") == r["shard"] and p.get("npz")), None)
        if npz and Path(npz).exists():
            z = zs.setdefault(npz, np.load(npz))
            o_arrays[(r["id"], r["arm"])] = z[f"{r['npz_key']}__hidden"]
    rows, mine_t, other_t = [], [], []
    for r in runs:
        o = o_runs.get((r["id"], r["arm"]))
        if o is None:
            continue
        dps = [float(np.max(np.abs(np.asarray(a["probs"]) - np.asarray(b["probs"]))))
               for a, b in zip(r["questions"], o["questions"])]
        row = {"id": r["id"], "arm": r["arm"], "source": r["source"], "tokens": r["tokens"],
               "max_abs_dp": max(dps), "argmax_equal": all(a["argmax"] == b["argmax"]
                                                             for a, b in zip(r["questions"], o["questions"])),
               "calls": [o["calls"], r["calls"]],
               "decode_seconds": [o["decode_seconds"], r["decode_seconds"]],
               "max_abs_dp_vs_oracle": [o["max_abs_dp"], r["max_abs_dp"]],
               "min_pos_cos": [o["hidden"]["min_pos_cos"], r["hidden"]["min_pos_cos"]]}
        if "positions_below" in o["hidden"] and "positions_below" in r["hidden"]:
            row["positions_below_0.99"] = [o["hidden"]["positions_below"]["0.99"], r["hidden"]["positions_below"]["0.99"]]
        h_other = o_arrays.get((r["id"], r["arm"]))
        if h_other is not None:
            a = arrays[(r["id"], r["arm"], "base")]["hidden"].astype(np.float32)
            row["hidden_max_abs_diff"] = float(np.max(np.abs(a - h_other.astype(np.float32))))
            row["hidden_bit_equal"] = bool(np.array_equal(arrays[(r["id"], r["arm"], "base")]["hidden"], h_other))
        rows.append(row)
        if not r["first_in_process"]:
            mine_t.append(r)
        if not o["first_in_process"]:
            other_t.append(o)
    S_other = other["chunk"]

    def times(rs: list[dict], s: int, call_ms: list) -> dict:
        return {"S": s, "runs": len(rs),
                "ms_per_call_median": float(np.median(np.concatenate(call_ms))) if call_ms else None,
                "ms_per_token_median": float(np.median([x["decode_seconds"] * 1e3 / x["tokens"] for x in rs]))
                if rs else None,
                "run_decode_seconds_median": float(np.median([x["decode_seconds"] for x in rs])) if rs else None,
                "calls_total": int(sum(x["calls"] for x in rs))}

    common = {(x["id"], x["arm"]) for x in mine_t} & {(x["id"], x["arm"]) for x in other_t}
    mine_c = [x for x in mine_t if (x["id"], x["arm"]) in common]
    other_c = [x for x in other_t if (x["id"], x["arm"]) in common]
    other_ms = [np.asarray(x["call_ms"]) for x in other_c]
    below = [x["positions_below_0.99"] for x in rows if "positions_below_0.99" in x]
    return {"other": str(other_path), "other_chunk": S_other, "other_bundle": other["bundle"]["name"], "runs": len(rows),
            "max_abs_dp": max((x["max_abs_dp"] for x in rows), default=None),
            "argmax_equal_runs": sum(x["argmax_equal"] for x in rows),
            "hidden_max_abs_diff": max((x.get("hidden_max_abs_diff", 0.0) for x in rows), default=None),
            "vs_oracle_on_shared_runs": {
                "order": ["other", "this"],
                "max_abs_dp": [max((x["max_abs_dp_vs_oracle"][i] for x in rows), default=None) for i in (0, 1)],
                "min_pos_cos": [min((x["min_pos_cos"][i] for x in rows), default=None) for i in (0, 1)],
                "positions_below_0.99": ([int(sum(b[i] for b in below)) for i in (0, 1)]
                                         if len(below) == len(rows) else None)},
            "timing_warm_common_runs": {"other": times(other_c, S_other, other_ms),
                                        "this": times(mine_c, S, [x["_call_ms"] for x in mine_c])},
            "rows": rows}


def bundle_record(bundle: Path, aimodelc: Path) -> dict:
    meta = json.loads((bundle / "metadata.json").read_text())
    mlirb = bundle / meta["assets"]["main"] / "main.mlirb"
    return {"bundle": str(bundle), "name": meta["name"], "kind": meta["kind"], "compression": meta.get("compression"),
            "prefill_chunk": meta["language"].get("prefill_chunk"),
            "metadata_sha256": sha256_file(bundle / "metadata.json"),
            "main_mlirb": {"bytes": mlirb.stat().st_size, "sha256": sha256_file(mlirb)},
            "tokenizer_sha256": {f.name: sha256_file(f) for f in sorted((bundle / "tokenizer").iterdir())},
            "aimodelc": tree_digest(aimodelc)}


def gate(args) -> int:
    from parity_decoder_torch import AuthorHead

    bundle = Path(args.bundle).expanduser().resolve()
    meta = json.loads((bundle / "metadata.json").read_text())
    S = int(meta["language"]["prefill_chunk"])
    max_ctx = int(meta["language"]["max_context_length"])
    aimodelc = bundle.parent.parent / "bundles_aotc" / f"{meta['name']}.h16c.aimodelc"
    if not aimodelc.exists():
        raise SystemExit(f"no AOT asset {aimodelc} (export_decoder.py --aot)")
    if args.oracle:
        set_oracle(args.oracle)
    doc, rows = load_oracle()
    orc = {(r["id"], r["arm"]): r for r in rows}
    if (args.subset == "list") != bool(args.runs_file):
        raise SystemExit("--runs-file goes with --subset list, and only with it")
    expected = subset_runs(rows, args.subset, args.runs_file)
    semif_first = next((r["id"] for r in rows if r["source"] == "semif_authored144"), None)
    keep = set(KEEP_HIDDEN) | ({(semif_first, "text")} if semif_first else set())
    tag = args.tag or meta["name"].replace("clef_flash_decode_", "")
    work = Path(args.work_dir).expanduser() / tag if args.work_dir else LANE / "readout" / tag
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    others_start = other_gpu_processes()
    shards = [{"aimodelc": str(aimodelc), "chunk": S, "max_ctx": max_ctx, "runs": [[i, a, "base"] for i, a in part],
               "out": str(work / f"shard_{k:02d}"), **({"oracle": str(ORACLE)} if args.oracle else {})}
              for k, part in enumerate(split(expected))]
    red_spec = {"aimodelc": str(aimodelc), "chunk": S, "max_ctx": max_ctx,
                "runs": [[i, a, v] for i, a in RED_RUNS for v in ("base", "embeds_zero")],
                "out": str(work / "red_embeds_zero")}
    if args.rescore:
        recs = [read_shard(sp) for sp in shards]
        red_recs = [read_shard(red_spec)] if args.red and Path(red_spec["out"]).with_suffix(".json").exists() else []
    else:
        recs = run_shards(f"gate {tag}", shards)
        red_recs = run_shards(f"red {tag}", [red_spec]) if args.red else []
    gpu_seconds = time.monotonic() - t0
    others_end = other_gpu_processes()
    runs, procs, arrays = collect(recs)
    print(f"scoring {len(runs)} runs through the author's head ...", flush=True)
    t1 = time.monotonic()
    head = AuthorHead()
    scored = []
    for r in runs:
        o = orc[(r["id"], r["arm"])]
        a = arrays[(r["id"], r["arm"], r["variant"])]
        sc = score(head, o, a["hidden"])
        scored.append({**r, "source": o["source"], **sc, "call_ms": a["call_ms"].tolist(), "_call_ms": a["call_ms"]})
        if (r["id"], r["arm"]) in keep:
            p = work / f"hidden_{r['id']}__{r['arm']}.npz"
            np.savez(p, hidden=a["hidden"], call_ms=a["call_ms"],
                     probs=np.concatenate([np.asarray(q["probs"]) for q in sc["questions"]]))
            scored[-1]["hidden_npz"] = str(p)
    s = summarize(scored, expected)
    resets = all(p.get("reset_check", {}).get("bit_equal", False) for p in procs)
    by_sa = {k: summarize(v, [(x["id"], x["arm"]) for x in v]) for k, v in sorted(group(scored).items())}
    red = None
    if red_recs:
        rruns, rprocs, rarr = collect(red_recs)
        red = {"arm": "image_embeds zeroed (ids, image_rc and the rope shift unchanged)", "process": rprocs, "runs": []}
        for rid, arm in RED_RUNS:
            o = orc[(rid, arm)]
            base = score(head, o, rarr[(rid, arm, "base")]["hidden"])
            zero = score(head, o, rarr[(rid, arm, "embeds_zero")]["hidden"])
            gate_h = arrays.get((rid, arm, "base"))
            item = {"id": rid, "arm": arm,
                    "argmax_changed": sum(a["argmax"] != b["argmax"] for a, b in zip(zero["questions"], base["questions"])),
                    "questions": len(base["questions"]),
                    "max_abs_dp_vs_base": max(float(np.max(np.abs(np.asarray(a["probs"]) - np.asarray(b["probs"]))))
                                              for a, b in zip(zero["questions"], base["questions"])),
                    "base_argmax": [q["argmax"] for q in base["questions"]],
                    "zeroed_argmax": [q["argmax"] for q in zero["questions"]],
                    "oracle_argmax": [q["argmax_oracle"] for q in base["questions"]],
                    "base_probs": [q["probs"] for q in base["questions"]],
                    "zeroed_probs": [q["probs"] for q in zero["questions"]],
                    "base_hidden_bit_equal_gate_run": (bool(np.array_equal(gate_h["hidden"], rarr[(rid, arm, "base")]["hidden"]))
                                                       if gate_h is not None else None)}
            red["runs"].append(item)
        red["moves_argmax_every_run"] = all(x["argmax_changed"] > 0 for x in red["runs"])
        red["result"] = "RED (argmax moved on every run)" if red["moves_argmax_every_run"] else "NOT RED on every run"
        red["reset_bit_equal"] = all(p.get("reset_check", {}).get("bit_equal", False) for p in rprocs)
    ok, checks = verdict(s, resets and (red is None or red["reset_bit_equal"]))
    near = [{"id": r["id"], "arm": r["arm"], "question_id": q["question_id"], "oracle_top2_margin": q["oracle_top2_margin"],
             "argmax_equal": q["argmax_equal"], "max_abs_dp": q["max_abs_dp"], "probs": q["probs"],
             "probs_oracle": q["probs_oracle"]}
            for r in scored for q in r["questions"] if q["near_tie"]]
    descs = [p["descriptor"] for p in procs if "descriptor" in p]
    record = {
        "schema": "clef-flash-decoder-readout-gate/1",
        "gate": "decoder alone on the Mac GPU (oracle ids, oracle fp32 image_embeds -> fp16, zero-padded to 1024 "
                "rows), the fp16 hidden read through the author's fp32 head",
        "bundle": bundle_record(bundle, aimodelc), "chunk": S, "max_ctx": max_ctx,
        "readout": meta["decision"]["readout"],
        "descriptor": descs[0] if descs else None,
        "descriptor_same_in_every_process": all(d == descs[0] for d in descs) if descs else None,
        "contract": contract(S),
        "oracle": {"path": str(ORACLE / "records_oracle.json"), "sha256": sha256_file(ORACLE / "records_oracle.json"),
                   "source": doc["source"]["hf_id"] + "@" + doc["source"]["revision"], "versions": doc["versions"],
                   "npz_sha256": {f"{i}__{a}": sha256_file(ORACLE / "npz" / f"{i}__{a}.npz") for i, a in expected}},
        "subset": args.subset,
        "runs_file": ({"path": record_path(Path(args.runs_file).resolve()), "sha256": sha256_file(Path(args.runs_file)),
                       "why": json.loads(Path(args.runs_file).read_text()).get("why")} if args.runs_file else None),
        "script": {"path": "conversion/clef_flash/readout_gate.py", "sha256": sha256_file(Path(__file__).resolve())},
        "skipped": ([{"run": "photo_01/native", "reason": "N = 40 x 30 = 1,200 image rows > the graph's 1024-row "
                      "buffer (outside the fixed-grid path), not run", "questions": len(orc[("photo_01", "native")]["questions"])}]
                    if ("photo_01", "native") in orc else []),
        "bar": {**BAR, "argmax": "every question with an oracle top-2 margin above 0.02; near-ties listed apart",
                "max_abs_dp_applies_to": "every option of every question, near-ties included",
                "reset": "bit-equal hidden rows in every process",
                "mean_of_run_mean_abs_dp_definition": "mean over runs of the mean |dp| over all (question, option) "
                                                      "entries of the run"},
        "environment": env_record(),
        "other_gpu_processes": {"start": others_start, "end": others_end},
        "processes": procs, "summary": s, "by_source_arm": by_sa, "near_ties": near,
        "checks": checks, "result": "PASS" if ok else "FAIL",
        "red_arm": red, "timing": timing(scored, procs, S),
        "seconds": {"gpu_processes": gpu_seconds, "scoring": time.monotonic() - t1,
                    "total": time.monotonic() - t0},
        "runs": [{k: v for k, v in r.items() if k != "_call_ms"} for r in scored],
    }
    if args.compare_with:
        record["compare_with"] = compare_with(scored, arrays, Path(args.compare_with), S)
    if args.note:
        record["note"] = args.note
    record["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    Path(args.transcript).parent.mkdir(parents=True, exist_ok=True)
    Path(args.transcript).write_text(json.dumps(record, indent=1) + "\n")
    print(f"{record['result']}: {meta['name']} runs {s['runs']}/{s['expected_runs']} argmax {s['argmax_equal']}/"
          f"{s['questions']} (non-near-tie {s['argmax_equal_non_near_tie']}/{s['questions_non_near_tie']}, near-tie "
          f"{s['argmax_equal_near_tie']}/{s['near_tie_questions']}) max|dp| {s['max_abs_dp']:.6f} mean "
          f"{s['mean_of_run_mean_abs_dp']:.6f} min cos {s['min_pos_cos']:.6f} reset {resets} worst {s['worst_run']}")
    print(f"  positions {s['positions']}, below cos {json.dumps(s['positions_below'])} (image rows below 0.99: "
          f"{s['positions_below_0.99_image_rows']})")
    for k, v in checks.items():
        print(f"  {k}: {'ok' if v else 'FAIL'}")
    for k, v in by_sa.items():
        print(f"  {k:28s} runs {v['runs']:3d} q {v['questions']:3d} argmax {v['argmax_equal']}/{v['questions']} "
              f"max|dp| {v['max_abs_dp']:.2e} mean {v['mean_of_run_mean_abs_dp']:.2e} min cos {v['min_pos_cos']:.6f} "
              f"ms/call {v['ms_per_call_median']:.1f}")
    if red:
        for x in red["runs"]:
            print(f"red {x['id']}/{x['arm']}: argmax changed {x['argmax_changed']}/{x['questions']} max|dp| "
                  f"{x['max_abs_dp_vs_base']:.3f} base = gate run {x['base_hidden_bit_equal_gate_run']}")
    if "compare_with" in record:
        c = record["compare_with"]
        print(f"vs {c['other_bundle']} (S={c['other_chunk']}): runs {c['runs']} max|dp| {c['max_abs_dp']:.2e} argmax-equal "
              f"runs {c['argmax_equal_runs']} hidden max|d| {c['hidden_max_abs_diff']} vs oracle [other, this] "
              f"{json.dumps(c['vs_oracle_on_shared_runs'])} timing {json.dumps(c['timing_warm_common_runs'])}")
    print(f"transcript: {args.transcript}")
    return 0 if ok else 1


def red_base_cross_check(red: dict, runs_by_key: dict, arrays: dict) -> list[dict]:
    """A red arm's unperturbed run whose gate run lives in another invocation: compare the bits."""
    out = []
    procs = red.get("process") or []
    for x in red["runs"]:
        if x.get("base_hidden_bit_equal_gate_run") is not None:
            continue
        key = (x["id"], x["arm"])
        if key not in runs_by_key:
            continue
        for p in procs:
            label = f"{x['id']}/{x['arm']}/base"
            if label in p.get("runs", []) and p.get("npz") and Path(p["npz"]).exists():
                h = np.load(p["npz"])[f"{p['runs'].index(label):02d}__hidden"]
                g = arrays[key]
                out.append({"id": x["id"], "arm": x["arm"], "gate_run_invocation": runs_by_key[key]["invocation"],
                            "bit_equal": bool(np.array_equal(h, g)),
                            "max_abs_diff": float(np.max(np.abs(h.astype(np.float32) - g.astype(np.float32))))})
    return out


def merge(args) -> int:
    """One transcript from transcripts of the same bundle over disjoint run sets."""
    parts = [Path(p).expanduser().resolve() for p in args.parts]
    docs = [json.loads(p.read_text()) for p in parts]
    one = {k: {json.dumps(f(d)) for d in docs} for k, f in (
        ("bundle", lambda d: d["bundle"]["name"]), ("main_mlirb", lambda d: d["bundle"]["main_mlirb"]),
        ("chunk", lambda d: d["chunk"]), ("max_ctx", lambda d: d["max_ctx"]),
        ("oracle", lambda d: d["oracle"]["sha256"]), ("descriptor", lambda d: d["descriptor"]))}
    if any(len(v) != 1 for v in one.values()):
        raise SystemExit(f"the parts are not one bundle / graph / oracle: {one}")
    if args.oracle:
        set_oracle(args.oracle)
    doc, rows = load_oracle()
    if sha256_file(ORACLE / "records_oracle.json") != docs[0]["oracle"]["sha256"]:
        raise SystemExit("the oracle changed since the parts were scored")
    orc = {(r["id"], r["arm"]): r for r in rows}
    S = int(docs[0]["chunk"])
    invs, runs, procs, arrays, seen = [], [], [], {}, {}
    for k, (p, d) in enumerate(zip(parts, docs)):
        invs.append({"invocation": k, "transcript": str(p), "transcript_sha256": sha256_file(p), "subset": d["subset"],
                     "runs": len(d["runs"]), "result": d["result"], "generated_at": d["generated_at"],
                     "aimodelc_tree_sha256": d["bundle"]["aimodelc"]["tree_sha256"],
                     "script_sha256": (d.get("script") or {}).get("sha256"),
                     "seconds": d["seconds"], "environment": d["environment"],
                     "other_gpu_processes": d["other_gpu_processes"]})
        npz_of = {pr["shard"]: pr.get("npz") for pr in d["processes"]}
        procs += [{**pr, "invocation": k} for pr in d["processes"]]
        for r in d["runs"]:
            key = (r["id"], r["arm"])
            if key in seen:
                raise SystemExit(f"{key[0]}/{key[1]} is in invocations {seen[key]} and {k}")
            seen[key] = k
            rr = {**r, "invocation": k, "npz": npz_of.get(r["shard"])}
            h = np.load(rr["npz"])[f"{r['npz_key']}__hidden"]
            arrays[key] = h
            if "positions_below" not in r["hidden"]:   # the same hidden rows that invocation scored
                rr["hidden"] = {**r["hidden"], **position_cos(h, orc[key]), "position_cos_added_at_merge": True}
            rr["_call_ms"] = np.asarray(r["call_ms"], np.float64)
            runs.append(rr)
    expected = subset_runs(rows, args.subset)
    s = summarize(runs, expected)
    resets = all(p.get("reset_check", {}).get("bit_equal", False) for p in procs)
    reds = []
    for k, d in enumerate(docs):
        if d.get("red_arm"):
            reds.append({"invocation": k, **d["red_arm"],
                         "base_vs_gate_run_of_another_invocation": red_base_cross_check(
                             d["red_arm"], {kk: {"invocation": seen[kk]} for kk in seen}, arrays)})
    red_ok = all(x["reset_bit_equal"] for x in reds)
    ok, checks = verdict(s, resets and red_ok)
    by_sa = {g: summarize(v, [(x["id"], x["arm"]) for x in v]) for g, v in sorted(group(runs).items())}
    near = [{"id": r["id"], "arm": r["arm"], "invocation": r["invocation"], "question_id": q["question_id"],
             "oracle_top2_margin": q["oracle_top2_margin"], "argmax_equal": q["argmax_equal"],
             "max_abs_dp": q["max_abs_dp"], "probs": q["probs"], "probs_oracle": q["probs_oracle"]}
            for r in runs for q in r["questions"] if q["near_tie"]]
    record = {
        "schema": "clef-flash-decoder-readout-gate/1",
        "merged_from": invs,
        "merge_rule": "every run entry as its invocation scored it (+ invocation, + npz; position-cosine counts added "
                      "from the run's own npz where the invocation predates them); summary, checks, per-group tables "
                      "and timing recomputed over the union",
        "gate": docs[0]["gate"],
        "bundle": {**docs[-1]["bundle"], "per_invocation": [d["bundle"] for d in docs],
                   "aimodelc_same_bytes_every_invocation": len({d["bundle"]["aimodelc"]["tree_sha256"] for d in docs}) == 1},
        "chunk": S, "max_ctx": docs[0]["max_ctx"], "readout": docs[0]["readout"],
        "descriptor": docs[0]["descriptor"], "descriptor_same_in_every_process": all(
            d.get("descriptor_same_in_every_process") for d in docs),
        "contract": docs[0]["contract"],
        "oracle": {**{k: v for k, v in docs[0]["oracle"].items() if k != "npz_sha256"},
                   "npz_sha256": {kk: v for d in docs for kk, v in d["oracle"]["npz_sha256"].items()}},
        "subset": f"{args.subset} (merged)",
        "skipped": docs[0]["skipped"], "bar": docs[0]["bar"],
        "script": {"path": "conversion/clef_flash/readout_gate.py", "sha256": sha256_file(Path(__file__).resolve())},
        "processes": procs, "summary": s, "by_source_arm": by_sa, "near_ties": near,
        "checks": checks, "result": "PASS" if ok else "FAIL",
        "red_arm": reds,
        "timing": {"all": timing(runs, procs, S),
                   "per_invocation": {str(k): timing([r for r in runs if r["invocation"] == k],
                                                     [p for p in procs if p["invocation"] == k], S)
                                      for k in range(len(docs))}},
        "runs": [{k: v for k, v in r.items() if k != "_call_ms"} for r in runs],
    }
    if args.compare_with:
        record["compare_with"] = compare_with(runs, {(k[0], k[1], "base"): {"hidden": v} for k, v in arrays.items()},
                                              Path(args.compare_with), S)
    if args.note:
        record["note"] = args.note
    record["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    Path(args.transcript).parent.mkdir(parents=True, exist_ok=True)
    Path(args.transcript).write_text(json.dumps(record, indent=1) + "\n")
    print(f"{record['result']} (merged {len(docs)}): {docs[0]['bundle']['name']} runs {s['runs']}/{s['expected_runs']} "
          f"argmax {s['argmax_equal_non_near_tie']}/{s['questions_non_near_tie']} + near-tie "
          f"{s['argmax_equal_near_tie']}/{s['near_tie_questions']} max|dp| {s['max_abs_dp']:.6f} mean "
          f"{s['mean_of_run_mean_abs_dp']:.6f} min cos {s['min_pos_cos']:.6f} worst {s['worst_run']}")
    print(f"  positions {s['positions']}, below cos {json.dumps(s['positions_below'])} (image rows below 0.99: "
          f"{s['positions_below_0.99_image_rows']})")
    for k, v in checks.items():
        print(f"  {k}: {'ok' if v else 'FAIL'}")
    for x in reds:
        print(f"  red (invocation {x['invocation']}): cross-invocation base check {x['base_vs_gate_run_of_another_invocation']}")
    if "compare_with" in record:
        c = record["compare_with"]
        print(f"  vs {c['other_bundle']} (S={c['other_chunk']}): runs {c['runs']} max|dp| {c['max_abs_dp']:.2e} "
              f"argmax-equal runs {c['argmax_equal_runs']} vs oracle {json.dumps(c['vs_oracle_on_shared_runs'])}")
    print(f"transcript: {args.transcript}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run", help="the gate: workers on the GPU, then the author's head on their hidden rows")
    a.add_argument("bundle")
    a.add_argument("--transcript", required=True)
    a.add_argument("--subset", default="all", choices=["all", "s64", "s64rest", "list"])
    a.add_argument("--runs-file", help="--subset list: JSON {\"runs\": [[id, arm], ..], \"why\": ..}")
    a.add_argument("--red", action="store_true", help="add the image_embeds-zeroed red arm (own process)")
    a.add_argument("--compare-with", help="another transcript of the same runs (another chunk width)")
    a.add_argument("--tag", help="shard directory name under <work-dir> (default: from the bundle name)")
    a.add_argument("--work-dir", help=f"shard root (default {LANE / 'readout'})")
    a.add_argument("--rescore", action="store_true", help="re-score existing shards, no GPU runs")
    a.add_argument("--note", help="free text kept in the transcript")
    a.add_argument("--oracle", help=f"oracle directory (default {ORACLE}), e.g. the held-out set's")
    am = sub.add_parser("merge", help="join transcripts of one bundle over disjoint run sets")
    am.add_argument("parts", nargs="+")
    am.add_argument("--transcript", required=True)
    am.add_argument("--subset", default="all", choices=["all", "s64"], help="the run set the union must cover")
    am.add_argument("--compare-with", help="another transcript of the same runs (another chunk width)")
    am.add_argument("--note")
    am.add_argument("--oracle", help=f"oracle directory (default {ORACLE})")
    aw = sub.add_parser("worker")
    aw.add_argument("--spec", required=True)
    args = ap.parse_args()
    if args.cmd == "worker":
        return worker(Path(args.spec))
    if args.cmd == "merge":
        return merge(args)
    return gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
