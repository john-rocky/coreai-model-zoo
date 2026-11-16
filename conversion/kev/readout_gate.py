#!/usr/bin/env python3
"""Readout gate: the Kev-0.8B decoder bundle on the Mac GPU, read through the author's fp32 pointer head, vs the author's fp32 oracle.

Every oracle question is one row (`oracle/records_oracle.json` `row_ids` = state + branch, positions 0..L-1),
fed to the bundle's one static-S function `main` (AOT h16c `.aimodelc`, `SpecializationOptions.default()`,
no JIT) from fresh zero states, in S-token chunks: call c gets ids[cS : cS + S] with position_ids 0..cS+S-1,
the last call padded with <|endoftext|> (248044) and the padded positions' hidden rows discarded
(`parity_decoder_torch.py`'s chunk order). The fp16 hidden rows [T, 1024] are cast to fp32 and go through the
pointer head (`parity_decoder_torch.KevHead`: head.safetensors, fp32, CPU; z = (k(h_opts) @ q(h_decide)) *
0.0625 / T); the per-question softmax is compared with the oracle's.

Bar, fixed before any result: every row present; per-question argmax = the oracle's on every question whose
oracle top-2 margin is above 0.02 (the near-ties, margin <= 0.02, are listed apart with their argmax count);
max |dp| <= 0.02 over every option of every question (near-ties included); the mean over runs of the run's
mean |dp| over its options <= 0.002 (a run = one row = one question); every process re-runs its first row at
the end and reproduces its hidden rows bit for bit (state reset proof); every hidden value finite. The 21 rows
of the six records with oracle hidden states also get the position cosine of the fp16 hidden rows against
the oracle's fp32 ones (minimum, and how many positions fall below 0.99 / 0.9 / 0.5).

Red arms (`--red`, their own process; the rows and pairs are fixed in `<work>/red_arms.json` before running):
the first question of tv4_000, the first and third SemIf records, own_t01 and tv4x_qnli_00, each run
unperturbed (`base`, compared bit for bit with the gate's run of the same row) and twice perturbed:
  state_swap    the row's state (indices 1 .. Ls-1) replaced by the state ids of a record of the same kind
                (tv4_001, the second / fourth SemIf record, own_t03, tv4x_paws_00); the readout indices shift
                by the length difference
  instr_negate  the id of " not" (the base tokenizer, recorded) inserted right after the <q> delimiter
                (248061); the readout indices shift by one
Per arm: argmax changes against the unperturbed run / rows, and max |dp| against it. Red (the perturbed
runs fail the gate's own bar against the unperturbed ones) = an argmax moves on a question whose oracle top-2 margin
is above 0.02, or max |dp| > 0.02, or the mean of the runs' mean |dp| > 0.002. (The v1 run of 2026-10-03 was scored
with a stricter rule written before it, argmax AND max |dp|; both are in the transcript.)

`red` (a subcommand) runs a rows file alone against an existing gate transcript of the same asset.
`red_arms_v2.json` (written by its first `red` run): the state_swap pairs with the first SemIf pair replaced
by two records that ask the same claim on different states (v1's first pair share one state: a no-op), and
instr_negate_v2 = a grammatical "not" inside five noul questions (`V2_NEGATIONS`: "Is this post offensive?" ->
"Is this post not offensive?", ...), the instructions re-encoded with the base tokenizer and the row rebuilt
around them (the readout indices shift by the token-count difference). v1's bare " not" right after <q> moves
p by at most 0.057 and no argmax, and the fp32 torch module does the same: the model barely reads it.

Process split: the Python runtime leaks one IOSurface per call, so a process takes at most 40 rows + the reset
re-run. The driver runs the processes one after another; each worker writes `<work>/shard_NN.json` (per row:
T, calls, ms) and `.npz` (the fp16 hidden rows and every call's wall ms); the driver scores them with the head
and writes the transcript. The GPU is shared with other sessions: the times are contended reference values.

Round 14 (a bundle whose metadata names `language.query_len_range` [2, cap] instead of `prefill_chunk`: the
dynamic-S Metal-kernel graph): a row of T ids runs in pieces of the real ids, no padding (`dyn_pieces`: cap-sized pieces,
the remainder last, a remainder of 1 takes one id from the piece before it), call k with position_ids 0..p+s-1; the
hidden output is read as rows [0:s] (dynamic [1, s, H] or the padded [1, cap, H]). `--shared` groups the rows by record:
the state ids (row_ids up to the first <q>) run once in pieces, the four states are copied, and every question's branch
(from <q>) runs on a copy with positions continuing after the state; the reset re-run repeats the process's first
record the same way. A record whose state is 1 id runs direct (no 1-id call); it is listed in the process record.
`--host-q q` (default 1 = the above): with q >= 2 every call's length is a multiple of q
(<= cap): the ids are cut into cap-sized pieces and the last piece is padded with <|endoftext|> (248044) up to the next
multiple of q, the padded rows' hidden discarded (`host_pieces`); --shared runs the first floor(Ls / q) * q state ids once
(cap-sized calls, no padding: cap is a multiple of q) and each question's remaining state ids + branch on a copy.
Round 15: q comes from the bundle's metadata (`language.query_len_multiple`, absent = 1, the round-14 bundles) and so
does the longest call L (`language.query_len_call_max`, absent = the graph's cap): the pieces are L ids, not the cap's,
so the gate reads a bundle as its hosts do (host.plan, which test_host.py checks against `host_pieces` call for call);
`--host-q` and `--call-max` override them.

`--subset s80` = 80 records (`<work>/../s80.json`, fixed before running): the 20 own records, the 20 tv4s
records, the first 3 records of each tv4x source (21), the first 19 SemIf records. `--compare-with
<transcript>` adds the per-row p difference, hidden difference and timing against another transcript (another
chunk width) on the rows both have.

    cd conversion/kev
    HF_HOME=$ZOO_WORK_ROOT/_kev/hf HF_HUB_OFFLINE=1 PY=<coreai-models venv>/bin/python
    $PY readout_gate.py run $ZOO_WORK_ROOT/_kev/exports/bundles/kev_0_8b_decode_fp16_pf16 --red \\
        --transcript $ZOO_WORK_ROOT/_kev/results/readout_fp16_pf16.json
    $PY readout_gate.py run .../kev_0_8b_decode_fp16_pf64 --compare-with .../readout_fp16_pf16.json \\
        --transcript .../readout_fp16_pf64.json
    $PY readout_gate.py run .../kev_0_8b_decode_fp16_pf32 --subset s80 --compare-with .../readout_fp16_pf16.json \\
        --transcript .../readout_fp16_pf32.json
    $PY readout_gate.py red .../kev_0_8b_decode_fp16_pf16 --red-file $ZOO_WORK_ROOT/_kev/readout/red_arms_v2.json \\
        --gate-transcript .../readout_fp16_pf16.json --transcript .../readout_fp16_pf16_red_v2.json

Kev-4B (round 4): `--model kev-4b` on `run` and `red` sets the contract (hidden 2560; KV [8, 1, 4, -1, 256] x 2,
convState [24, 1, 8192, 3], recState [24, 1, 32, 128, 128]), the head (`<work>/oracle_4b/head`), the oracle
(`<work>/oracle_4b`; the held-out run passes `--oracle <work>/oracle_4b/heldout`) and the shard directory
(`<work>/readout_4b/<tag>`). The red rows file is round 2's `readout/red_arms_v2.json`: its ids are the oracle's rows,
which the 4B tokenizer path reproduces id for id (round 4 tokenizer_check_4b.json).
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
from _paths import gpu_lock, hf_snapshot, work_path  # noqa: E402

LANE = work_path("_kev")
os.environ.setdefault("HF_HOME", str(LANE / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ORACLE = LANE / "oracle"
READOUT = LANE / "readout"
HIDDEN = 1024
PAD_ID, STATE_ID, Q_ID = 248044, 248060, 248061
BASE_REPO, BASE_REVISION = "Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
BAR = {"max_abs_dp": 0.02, "mean_of_run_mean_abs_dp": 0.002, "near_tie_top2_margin": 0.02}
RUNS_PER_PROCESS = 40                               # + the reset re-run
STATE_SHAPES = {"keyCache": [6, 1, 2, -1, 256], "valueCache": [6, 1, 2, -1, 256],
                "convState": [18, 1, 6144, 3], "recState": [18, 1, 16, 128, 128]}
HIDDEN_RECORDS = ("tv4_000", "semif_a3f18f3a63d45345942b", "own_t01", "own_j03", "own_L02", "own_m01")
MODELS = {"kev-0.8b": {"hidden": HIDDEN, "states": STATE_SHAPES, "oracle": "oracle", "readout": "readout"},
          "kev-4b": {"hidden": 2560, "states": {"keyCache": [8, 1, 4, -1, 256], "valueCache": [8, 1, 4, -1, 256],
                                                "convState": [24, 1, 8192, 3], "recState": [24, 1, 32, 128, 128]},
                     "oracle": "oracle_4b", "readout": "readout_4b"}}
HEAD_DIR: Path | None = None                        # None = parity_decoder_torch's (the 0.8B oracle's head/)
DYNAMIC: dict | None = None                         # round 14: {"min", "max", "padded"} for a dynamic-S bundle
OTHER_GPU = re.compile(r"yardstick|litert|llm-bench|coreai_verify|coreai-build|readout_gate|gate_|parity_|mlx")
COS_THRESHOLDS = (0.99, 0.9, 0.5)
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


def load_oracle() -> tuple[dict, dict]:
    doc = json.loads((ORACLE / "records_oracle.json").read_text())
    return doc, {r["id"]: r for r in doc["records"]}


def set_oracle(path: Path) -> None:
    """`run --oracle <dir>`: another oracle directory (the held-out one); the workers get it through their spec."""
    global ORACLE
    ORACLE = Path(path).expanduser().resolve()
    if not (ORACLE / "records_oracle.json").exists():
        raise SystemExit(f"no records_oracle.json in {ORACLE}")


def configure(model: str, head_dir: str | None = None, readout_dir: str | None = None) -> None:
    """--model: the contract (hidden size, state shapes), the default oracle / head / shard directories."""
    global HIDDEN, STATE_SHAPES, ORACLE, READOUT, HEAD_DIR
    m = MODELS[model]
    HIDDEN, STATE_SHAPES = m["hidden"], m["states"]
    ORACLE = LANE / m["oracle"]
    READOUT = Path(readout_dir).expanduser().resolve() if readout_dir else LANE / m["readout"]
    HEAD_DIR = Path(head_dir).expanduser().resolve() if head_dir else (None if model == "kev-0.8b" else ORACLE / "head")


def head_of():
    from parity_decoder_torch import KevHead
    return KevHead(HEAD_DIR, HIDDEN)


def contract(S: int) -> dict:
    return {"inputs": {"input_ids": [[1, S], "int32"], "position_ids": [[1, -1], "int32"]},
            "outputs": {"hidden": [[1, S, HIDDEN], "float16"]},
            "states": {n: [s, "float16"] for n, s in STATE_SHAPES.items()}}


def dyn_pieces(n: int, cap: int, qmin: int = 2) -> list[int]:
    """Round 14: piece lengths of an n-id run on the dynamic-S graph: cap-sized pieces, the remainder last; a remainder
    below qmin takes ids from the piece before it (no 1-id call)."""
    if n < qmin:
        raise SystemExit(f"a run of {n} ids is shorter than the graph's smallest call ({qmin})")
    k = -(-n // cap)
    sizes = [cap] * (k - 1) + [n - cap * (k - 1)]
    if sizes[-1] < qmin:
        sizes[-2] -= qmin - sizes[-1]
        sizes[-1] = qmin
    assert sum(sizes) == n and all(qmin <= x <= cap for x in sizes), sizes
    return sizes


def host_pieces(n: int, cap: int, q: int = 1, qmin: int = 2) -> list[tuple[int, int]]:
    """(call length, real ids in it) for an n-id run: q = 1 -> dyn_pieces (no padding); q >= 2 -> cap-sized pieces, the
    last one padded up to the next multiple of q (no call shorter than q)."""
    if q == 1:
        return [(x, x) for x in dyn_pieces(n, cap, qmin)]
    if cap % q:
        raise SystemExit(f"the host policy q={q} needs cap {cap} to be a multiple of q")
    out, left = [], n
    while left > 0:
        r = min(cap, left)
        out.append((-(-r // q) * q, r))
        left -= r
    return out


def dyn_contract(cap: int, padded: bool) -> dict:
    return {"inputs": {"input_ids": [[1, "2..%d" % cap], "int32"], "position_ids": [[1, -1], "int32"]},
            "outputs": {"hidden": [[1, cap if padded else "s", HIDDEN], "float16"]},
            "states": {n: [s, "float16"] for n, s in STATE_SHAPES.items()}}


def run_label(run: dict) -> str:
    return f"{run['id']}:q{run['q']}:{run['variant']}"


# --------------------------------------------------------------------------- #
# Worker: one process, <= 40 rows + the reset re-run
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

    spec = json.loads(spec_path.read_text())
    if spec.get("oracle"):
        set_oracle(Path(spec["oracle"]))
    if spec.get("hidden"):   # the driver's --model contract
        global HIDDEN, STATE_SHAPES
        HIDDEN, STATE_SHAPES = int(spec["hidden"]), spec["state_shapes"]
    _, recs = load_oracle()
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
        dyn = spec.get("dynamic")
        if dyn:   # round 14: names, dtypes and the fixed state shapes; the query / output dims are dynamic
            want = contract(S)
            bad = [f"{part} names {sorted(desc[part])} != {sorted(want[part])}" for part in ("inputs", "outputs", "states")
                   if set(desc[part]) != set(want[part])]
            bad += [f"{part} {n} dtype {desc[part][n][1]} != {w[1]}" for part in ("inputs", "outputs", "states")
                    for n, w in want[part].items() if n in desc[part] and desc[part][n][1] != w[1]]
            bad += [f"states {n}: {desc['states'][n][0]} != {w[0]}" for n, w in want["states"].items()
                    if n in desc["states"] and list(desc["states"][n][0]) != list(w[0])]
        else:
            bad = check_contract(desc, S)
        out["contract_mismatch"] = bad
        if bad:
            raise SystemExit(f"descriptor differs from the contract: {bad}")

        def fresh_state() -> dict:
            return {n: nd(np.zeros([max_ctx if s < 0 else s for s in shape], np.dtype(dt)))
                    for n, (shape, dt) in desc["states"].items()}

        async def call(ids_piece: np.ndarray, p: int, state: dict) -> np.ndarray:
            s = len(ids_piece)
            res = await maybe(fn(inputs={"input_ids": nd(ids_piece.reshape(1, s)),
                                         "position_ids": nd(np.arange(p + s, dtype=np.int32)[None])}, state=state))
            h = np.asarray(res["hidden"].numpy())
            if h.ndim != 3 or h.shape[0] != 1 or h.shape[1] not in (s, S) or h.shape[2] != HIDDEN or h.dtype != np.float16:
                raise SystemExit(f"output {h.shape} {h.dtype} != (1, {s} or {S}, {HIDDEN}) float16")
            return h[0, :s]

        HQ = int(dyn.get("q", 1)) if dyn else 1

        async def pieces(ids: np.ndarray, p0: int, state: dict) -> tuple[np.ndarray, list, list]:
            """Round 14: the ids from position p0 on, in host_pieces (policy HQ); -> hidden rows, per-call ms, per-call
            length (padded calls count their pad)."""
            hid = np.zeros((len(ids), HIDDEN), np.float16)
            ms, ss, q = [], [], 0
            for c, r in host_pieces(len(ids), int(dyn.get("call_max", S)), HQ, dyn["min"]):
                x = np.full(c, PAD_ID, np.int32)
                x[:r] = ids[q:q + r]
                t1 = time.perf_counter()
                hid[q:q + r] = (await call(x, p0 + q, state))[:r]
                ms.append((time.perf_counter() - t1) * 1e3)
                ss.append(c)
                q += r
            return hid, ms, ss

        async def one_dyn(run: dict) -> tuple[dict, dict[str, np.ndarray]]:
            ids = np.asarray(run.get("ids") or recs[run["id"]]["questions"][run["q"]]["row_ids"], np.int32)
            T = len(ids)
            if T > max_ctx - 1:
                raise SystemExit(f"{run_label(run)}: {T} positions > {max_ctx - 1}")
            t_run = time.perf_counter()
            state = fresh_state()
            t_dec = time.perf_counter()
            hid, ms, ss = await pieces(ids, 0, state)
            dec = time.perf_counter() - t_dec
            rec = {"id": run["id"], "q": run["q"], "variant": run["variant"], "tokens": T, "calls": len(ss),
                   "padded_tokens": int(sum(ss)), "call_s": ss, "decode_seconds": dec, "wall_seconds": time.perf_counter() - t_run,
                   "finite": bool(np.isfinite(hid.astype(np.float32)).all()), "all_zero": bool(not np.any(hid))}
            return rec, {"hidden": hid, "call_ms": np.asarray(ms, np.float64)}

        async def group_shared(group: list[dict]) -> tuple[list[tuple[dict, dict]], dict]:
            """Round 14 --shared: one record's questions; the state ids once, then each branch on a copy."""
            rows = [np.asarray(recs[r["id"]]["questions"][r["q"]]["row_ids"], np.int32) for r in group]
            Ls = int(np.flatnonzero(rows[0] == Q_ID)[0])
            if any(len(x) <= Ls or not np.array_equal(x[:Ls], rows[0][:Ls]) or x[Ls] != Q_ID for x in rows):
                raise SystemExit(f"{group[0]['id']}: the rows do not share the state ids [0:{Ls}]")
            info = {"id": group[0]["id"], "state_len": Ls, "questions": len(group), "host_q": HQ}
            Ls = (Ls // HQ) * HQ                         # the shared part: whole multiples of q (all of it at q = 1)
            info["shared_tokens"] = Ls
            if Ls < dyn["min"]:
                info["direct_fallback"] = f"shared part of {Ls} id(s) < the smallest call"
                return [await one_dyn(r) for r in group], info
            t_pre = time.perf_counter()
            state = fresh_state()
            h_pre, ms_pre, ss_pre = await pieces(rows[0][:Ls], 0, state)
            snap = {n: np.array(v.numpy(), copy=True) for n, v in state.items()}
            info.update({"prefix_calls": len(ss_pre), "prefix_call_s": ss_pre, "prefix_call_ms": ms_pre,
                         "prefix_seconds": time.perf_counter() - t_pre})
            res = []
            for r, ids in zip(group, rows):
                t_run = time.perf_counter()
                st = {n: nd(a.copy()) for n, a in snap.items()}   # a fresh copy per branch (decide.py's to_state)
                h_b, ms, ss = await pieces(ids[Ls:], Ls, st)
                hid = np.concatenate([h_pre, h_b])
                dec = time.perf_counter() - t_run
                rec = {"id": r["id"], "q": r["q"], "variant": r["variant"], "tokens": len(ids), "calls": len(ss),
                       "padded_tokens": Ls + int(sum(ss)), "computed_tokens": len(ids) - Ls, "call_s": ss,
                       "decode_seconds": dec,
                       "wall_seconds": time.perf_counter() - t_run, "shared_state_len": Ls,
                       "finite": bool(np.isfinite(hid.astype(np.float32)).all()), "all_zero": bool(not np.any(hid))}
                res.append((rec, {"hidden": hid, "call_ms": np.asarray(ms, np.float64)}))
            return res, info

        async def one(run: dict) -> tuple[dict, dict[str, np.ndarray]]:
            if dyn:
                return await one_dyn(run)
            ids = run.get("ids") or recs[run["id"]]["questions"][run["q"]]["row_ids"]
            ids = np.asarray(ids, np.int32)
            T = len(ids)
            n_calls = -(-T // S)
            if n_calls * S > max_ctx - 1:
                raise SystemExit(f"{run_label(run)}: {n_calls * S} padded positions > {max_ctx - 1}")
            ids_p = np.full(n_calls * S, PAD_ID, np.int32)
            ids_p[:T] = ids
            t_run = time.perf_counter()
            state = fresh_state()
            hid = np.zeros((n_calls * S, HIDDEN), np.float16)
            call_ms = np.zeros(n_calls, np.float64)
            t_dec = time.perf_counter()
            for c in range(n_calls):
                t1 = time.perf_counter()
                res = await maybe(fn(inputs={"input_ids": nd(ids_p[c * S:(c + 1) * S].reshape(1, S)),
                                             "position_ids": nd(np.arange((c + 1) * S, dtype=np.int32)[None])},
                                     state=state))
                h = np.asarray(res["hidden"].numpy())
                if h.shape != (1, S, HIDDEN) or h.dtype != np.float16:
                    raise SystemExit(f"{run_label(run)}: output {h.shape} {h.dtype} != (1, {S}, {HIDDEN}) float16")
                hid[c * S:(c + 1) * S] = h[0]
                call_ms[c] = (time.perf_counter() - t1) * 1e3
            dec = time.perf_counter() - t_dec
            rec = {"id": run["id"], "q": run["q"], "variant": run["variant"], "tokens": T, "calls": n_calls,
                   "padded_tokens": n_calls * S, "decode_seconds": dec, "wall_seconds": time.perf_counter() - t_run,
                   "finite": bool(np.isfinite(hid[:T].astype(np.float32)).all()),
                   "all_zero": bool(not np.any(hid[:T]))}
            return rec, {"hidden": hid[:T].copy(), "call_ms": call_ms}

        runs = spec["runs"]
        first = None
        if dyn and spec.get("shared"):   # round 14: record groups, the reset re-run = the first group again
            groups: list[list[dict]] = []
            for run in runs:
                if groups and groups[-1][0]["id"] == run["id"]:
                    groups[-1].append(run)
                else:
                    groups.append([run])
            out["shared_groups"] = []
            i = 0
            for gi, g in enumerate(groups + [groups[0]]):
                res, info = await group_shared(g)
                if gi < len(groups):
                    out["shared_groups"].append(info)
                    for rec, arrays in res:
                        key = f"{i:02d}"
                        for k, a in arrays.items():
                            store[f"{key}__{k}"] = a
                        rec["index"] = i
                        out["runs"].append(rec)
                        if i == 0:
                            first = [a["hidden"] for _, a in res]
                        i += 1
                    print(f"  [{os.getpid()}] shared {info['id']}: {info['questions']} q, state {info['state_len']} ids "
                          f"({info.get('prefix_calls', 0)} calls), branches {[r['calls'] for r, _ in res]} calls", flush=True)
                else:
                    again = [a["hidden"] for _, a in res]
                    same = len(again) == len(first) and all(np.array_equal(a, b) for a, b in zip(first, again))
                    diff = max(float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))
                               for a, b in zip(first, again))
                    out["reset_check"] = {"run": f"{g[0]['id']} (shared, {len(g)} q)", "bit_equal": bool(same),
                                          "hidden_max_abs_diff": diff}
                    print(f"  [{os.getpid()}] reset re-run {g[0]['id']} shared: bit-equal {same} (max|d| {diff})", flush=True)
            return
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
                print(f"  [{os.getpid()}] {run_label(run)}: {rec['tokens']} tok, {rec['calls']} calls, "
                      f"{rec['wall_seconds']:.3f} s (median {np.median(cm):.2f} ms/call, first {cm[0]:.1f})", flush=True)
            else:
                same = bool(np.array_equal(first["hidden"], arrays["hidden"]))
                diff = float(np.max(np.abs(first["hidden"].astype(np.float32) - arrays["hidden"].astype(np.float32))))
                out["reset_check"] = {"run": run_label(run), "bit_equal": same, "hidden_max_abs_diff": diff,
                                      "wall_seconds": rec["wall_seconds"]}
                print(f"  [{os.getpid()}] reset re-run {run_label(run)}: bit-equal {same} (max|d| {diff})", flush=True)

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
    skip = ("/.local/bin/claude", "shell-snapshots", "until grep", "zsh -c", "/bin/zsh", "/bin/bash")
    return [ln.strip()[:200] for ln in ps.splitlines()
            if OTHER_GPU.search(ln) and not ln.strip().startswith(f"{me} ") and not any(s in ln for s in skip)
            and "readout_gate.py worker" not in ln]


def gpu_lock_state() -> dict:
    p = gpu_lock()
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {"path": str(p), "exists": True, "bytes": st.st_size, "content": p.read_text()[:500],
            "mtime": datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds")}


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
            "head": "the pointer head from head.safetensors, fp32, CPU (parity_decoder_torch.KevHead)",
            "gpu": "shared with other sessions, no _GPU_LOCK held (times are contended reference values)"}


def shape_of(meta: dict) -> tuple[int, dict | None]:
    """(S, None) for a static-S bundle; (cap, {"min", "max", "padded", "q"}) for round 14's dynamic-S bundle (q =
    round 15's `query_len_multiple`, absent = 1)."""
    global DYNAMIC
    lang = meta["language"]
    if "query_len_range" in lang:
        lo, hi = (int(x) for x in lang["query_len_range"])
        DYNAMIC = {"min": lo, "max": hi, "padded": "rows 0..s-1" in str(lang.get("output", "")),
                   "q": int(lang.get("query_len_multiple", 1)), "q_from": "metadata",
                   "call_max": int(lang.get("query_len_call_max", hi))}
        return hi, DYNAMIC
    DYNAMIC = None
    return int(lang["prefill_chunk"]), None


def split_records(rows: list, per: int = RUNS_PER_PROCESS) -> list[list]:
    """Round 14 --shared: whole records per process (a record's questions share one state run), <= per rows each."""
    groups: list[list] = []
    for r in rows:
        if groups and groups[-1][0][0] == r[0]:
            groups[-1].append(r)
        else:
            groups.append([r])
    parts: list[list] = [[]]
    for g in groups:
        if parts[-1] and len(parts[-1]) + len(g) > per:
            parts.append([])
        parts[-1].extend(g)
    return parts


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
        print(f"[{tag}] {Path(sp['out']).name}: {len(sp['runs'])} rows + reset re-run", flush=True)
        t0 = time.monotonic()
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--spec", str(spec_path)])
        wall = time.monotonic() - t0
        got.append(read_shard(sp, proc.returncode, wall))
        if got[-1].get("failed"):
            raise SystemExit(f"{Path(sp['out']).name}: worker failed (exit {proc.returncode}); see its output above")
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
        entry = {"shard": prefix.name, "runs": [run_label(r) for r in sr["spec"]["runs"]],
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
            arrays[(rec["id"], rec["q"], rec["variant"])] = {"hidden": z[f"{key}__hidden"],
                                                            "call_ms": z[f"{key}__call_ms"]}
            runs.append({**rec, "shard": prefix.name, "npz_key": key, "first_in_process": rec["index"] == 0})
    return runs, procs, arrays


def oracle_hidden(rid: str, k: int) -> np.ndarray | None:
    p = ORACLE / "hidden" / f"{rid}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return z[f"q{k}_hidden"] if f"q{k}_hidden" in z.files else None


def position_cos(hidden16: np.ndarray, ref: np.ndarray, ids: list[int]) -> dict:
    from parity_decoder_torch import compare_hidden, cos_rows

    c = cos_rows(hidden16.astype(np.float32), ref)
    return {**compare_hidden(hidden16.astype(np.float32), ref), "positions": int(c.size),
            "positions_below": {str(t): int((c < t).sum()) for t in COS_THRESHOLDS},
            "lowest_positions": [{"pos": int(i), "cos": float(c[i]), "id": int(ids[i])}
                                 for i in np.argsort(c)[:LOWEST_POSITIONS]]}


def score(head, q: dict, hidden16: np.ndarray, decide: int | None = None, opts: list[int] | None = None) -> dict:
    """The pointer head on the fp16 hidden rows (cast to fp32) vs the oracle's probs for that question."""
    import torch
    h32 = torch.from_numpy(hidden16.astype(np.float32))
    lt = head.logits_T(h32, q["decide"] if decide is None else decide, q["opts"] if opts is None else opts)
    p = torch.softmax(lt, -1).double().numpy()
    po = np.asarray(q["probs"], np.float64)
    dp = np.abs(p - po)
    am, am_o = int(p.argmax()), q["keys"].index(q["argmax"])
    return {"argmax": am, "argmax_oracle": am_o, "argmax_equal": am == am_o,
            "max_abs_dp": float(dp.max()), "mean_abs_dp": float(dp.mean()),
            "max_abs_dlogit_T": float(np.abs(lt.double().numpy() - np.asarray(q["logits_T"], np.float64)).max()),
            "oracle_top2_margin": q["top2_margin"], "near_tie": bool(q["near_tie"]), "n_options": len(po),
            "probs": [float(v) for v in p], "probs_oracle": [float(v) for v in po]}


def summarize(runs: list[dict], expected: list[tuple[str, int]]) -> dict:
    have = {(r["id"], r["q"]) for r in runs}
    far = [r for r in runs if not r["near_tie"]]
    near = [r for r in runs if r["near_tie"]]
    worst = max(runs, key=lambda r: r["max_abs_dp"]) if runs else None
    hid = [r for r in runs if r.get("hidden")]
    warm = [r for r in runs if not r["first_in_process"]]
    ms = np.concatenate([r["_call_ms"] for r in runs]) if runs else np.zeros(0)
    wms = np.concatenate([r["_call_ms"] for r in warm] or [np.zeros(0)])
    tokens = int(sum(r["tokens"] for r in runs))
    padded = int(sum(r["padded_tokens"] for r in runs))
    by_rec: dict = {}
    for r in runs:
        by_rec.setdefault(r["id"], []).append(r)
    return {
        "runs": len(runs), "expected_runs": len(expected),
        "missing_runs": [f"{a}:q{b}" for a, b in expected if (a, b) not in have],
        "questions": len(runs), "argmax_equal": sum(r["argmax_equal"] for r in runs),
        "questions_non_near_tie": len(far), "argmax_equal_non_near_tie": sum(r["argmax_equal"] for r in far),
        "near_tie_questions": len(near), "argmax_equal_near_tie": sum(r["argmax_equal"] for r in near),
        "max_abs_dp": max((r["max_abs_dp"] for r in runs), default=None),
        "max_abs_dp_non_near_tie": max((r["max_abs_dp"] for r in far), default=None),
        "mean_of_run_mean_abs_dp": float(np.mean([r["mean_abs_dp"] for r in runs])) if runs else None,
        "mean_of_record_mean_abs_dp": float(np.mean([np.mean(np.concatenate(
            [np.abs(np.asarray(x["probs"]) - np.asarray(x["probs_oracle"])) for x in v])) for v in by_rec.values()]))
        if runs else None,
        "mean_question_max_abs_dp": float(np.mean([r["max_abs_dp"] for r in runs])) if runs else None,
        "max_abs_dlogit_T": max((r["max_abs_dlogit_T"] for r in runs), default=None),
        "worst_run": None if worst is None else {"id": worst["id"], "q": worst["q"], "max_abs_dp": worst["max_abs_dp"]},
        "finite_all": all(r["finite"] for r in runs), "all_zero_runs": sum(r["all_zero"] for r in runs),
        "hidden_rows": len(hid),
        "min_pos_cos": min((r["hidden"]["min_pos_cos"] for r in hid), default=None),
        "hidden_max_abs_diff": max((r["hidden"]["max_abs_diff"] for r in hid), default=None),
        "positions": int(sum(r["hidden"]["positions"] for r in hid)),
        "positions_below": {str(t): int(sum(r["hidden"]["positions_below"][str(t)] for r in hid)) for t in COS_THRESHOLDS},
        "tokens": tokens, "padded_tokens": padded, "calls": int(sum(r["calls"] for r in runs)),
        "pad_waste": (padded - tokens) / padded if padded else None,
        "decode_ms_total": float(sum(r["decode_seconds"] for r in runs) * 1e3),
        "ms_per_call_median": float(np.median(ms)) if ms.size else None,
        "ms_per_call_warm_median": float(np.median(wms)) if wms.size else None,
    }


def verdict(s: dict, resets_ok: bool) -> tuple[bool, dict]:
    checks = {
        "all_runs": s["runs"] == s["expected_runs"] and not s["missing_runs"],
        "argmax_non_near_tie": s["argmax_equal_non_near_tie"] == s["questions_non_near_tie"],
        "max_abs_dp": s["max_abs_dp"] is not None and s["max_abs_dp"] <= BAR["max_abs_dp"],
        "mean_of_run_mean_abs_dp": (s["mean_of_run_mean_abs_dp"] is not None
                                    and s["mean_of_run_mean_abs_dp"] <= BAR["mean_of_run_mean_abs_dp"]),
        "finite": s["finite_all"] and s["all_zero_runs"] == 0,
        "reset_bit_equal_all_processes": resets_ok,
    }
    return all(checks.values()), checks


def timing(runs: list[dict], procs: list[dict], S: int, recs: dict) -> dict:
    """Contended reference times. `warm` = every row but the first of its process. The fixture projection
    = the warm median ms per call x the calls the whole 434-row fixture takes at this S."""
    def q(a) -> dict:
        a = np.asarray(a, np.float64)
        return {"n": int(a.size), "median": float(np.median(a)) if a.size else None,
                "p10": float(np.quantile(a, 0.1)) if a.size else None,
                "p90": float(np.quantile(a, 0.9)) if a.size else None}

    warm = [r for r in runs if not r["first_in_process"]]
    loads = [p["load_seconds"] for p in procs if "load_seconds" in p]
    wms = np.concatenate([r["_call_ms"] for r in warm] or [np.zeros(0)])
    rows_all = [q_["row_len"] for r in recs.values() for q_ in r["questions"]]
    if DYNAMIC:   # round 14: no padding; the projection uses a least-squares a + b*s over the warm calls
        cap = DYNAMIC["max"]
        pieces_all = [c for t in rows_all for c, _ in host_pieces(t, int(DYNAMIC.get("call_max", cap)), int(DYNAMIC.get("q", 1)),
                                                                  DYNAMIC["min"])]
        ws = np.concatenate([np.asarray(r["call_s"], np.float64) for r in warm] or [np.zeros(0)])
        fit = None
        if ws.size >= 2 and np.ptp(ws) > 0:
            b, a = np.polyfit(ws, wms, 1)
            fit = {"a_ms": float(a), "b_ms_per_token": float(b), "calls": int(ws.size),
                   "projected_ms": float(sum(a + b * x for x in pieces_all))}
        return {
            "contended": True, "dynamic": DYNAMIC,
            "load_seconds": {"first_process": loads[0] if loads else None,
                             "median": float(np.median(loads)) if loads else None, "all": loads},
            "ms_per_call_warm_runs": q(wms),
            "ms_per_call_all_runs": q(np.concatenate([r["_call_ms"] for r in runs] or [np.zeros(0)])),
            "first_call_of_process_ms": [float(r["_call_ms"][0]) for r in runs if r["first_in_process"]],
            "ms_per_token_warm_runs": q([r["decode_seconds"] * 1e3 / r["tokens"] for r in warm]),
            "run_decode_seconds_warm_runs": q([r["decode_seconds"] for r in warm]),
            "warm_call_fit": fit,
            "fixture_434_rows": {"calls": len(pieces_all), "tokens": int(sum(rows_all)), "padded_tokens": int(sum(pieces_all)),
                                 "pad_waste": (sum(pieces_all) - sum(rows_all)) / sum(pieces_all),
                                 "projected_ms_at_warm_median_per_call": float(np.median(wms)) * len(pieces_all)
                                 if wms.size else None,
                                 "projected_ms_at_warm_fit": fit["projected_ms"] if fit else None},
        }
    calls_all = int(sum(-(-t // S) for t in rows_all))
    padded_all = calls_all * S
    return {
        "contended": True, "S": S,
        "load_seconds": {"first_process": loads[0] if loads else None,
                         "median": float(np.median(loads)) if loads else None, "all": loads},
        "ms_per_call_warm_runs": q(wms),
        "ms_per_call_all_runs": q(np.concatenate([r["_call_ms"] for r in runs] or [np.zeros(0)])),
        "first_call_of_process_ms": [float(r["_call_ms"][0]) for r in runs if r["first_in_process"]],
        "ms_per_token_warm_runs": q([r["decode_seconds"] * 1e3 / r["tokens"] for r in warm]),
        "run_decode_seconds_warm_runs": q([r["decode_seconds"] for r in warm]),
        "fixture_434_rows": {"calls": calls_all, "tokens": int(sum(rows_all)), "padded_tokens": padded_all,
                             "pad_waste": (padded_all - sum(rows_all)) / padded_all,
                             "projected_ms_at_warm_median_per_call": float(np.median(wms)) * calls_all if wms.size else None},
    }


def subset_rows(recs: dict, subset: str, s80_path: Path) -> list[tuple[str, int]]:
    rows = [(rid, k) for rid, r in recs.items() for k in range(len(r["questions"]))]
    if subset == "all":
        return rows
    want = json.loads(s80_path.read_text())["records"]
    missing = [x for x in want if x not in recs]
    if missing or len(want) != len(set(want)):
        raise SystemExit(f"{s80_path}: unknown or repeated records {missing}")
    keep = set(want)
    return [(rid, k) for rid, k in rows if rid in keep]


def write_s80(recs: dict, path: Path) -> None:
    """The 80-record subset: own 20, tv4s 20, the first 3 of each tv4x source (21), the first 19 SemIf (file order)."""
    ids = list(recs)
    by_src = lambda s: [i for i in ids if recs[i]["source"] == s]  # noqa: E731
    own = [i for i in ids if recs[i]["source"].startswith("own_")]
    tv4s = by_src("transfer_v4_dev_score")
    tv4x: dict = {}
    for i in by_src("transfer_v4_dev_sources"):
        tv4x.setdefault(i.rsplit("_", 1)[0], []).append(i)
    pick = own + tv4s + [i for src in tv4x.values() for i in src[:3]] + by_src("semif_authored144")[:19]
    doc = {"records": pick, "counts": {"own": len(own), "tv4s": len(tv4s), "tv4x": sum(min(3, len(v)) for v in tv4x.values()),
                                       "semif": 19, "total": len(pick)},
           "rows": sum(len(recs[i]["questions"]) for i in pick),
           "why": "the chunk-width trial set: every own record (all three question types, the 1.4-1.9k-token states, "
                  "the 8-question record), every score record, three of each tv4x source, the first 19 SemIf records",
           "written": datetime.now().astimezone().isoformat(timespec="seconds")}
    if doc["counts"]["total"] != 80:
        raise SystemExit(f"s80 has {doc['counts']['total']} records")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")


def not_token_id() -> dict:
    from transformers import AutoTokenizer

    snap = hf_snapshot(BASE_REPO, revision=BASE_REVISION)
    tok = AutoTokenizer.from_pretrained(snap)
    ids = tok(" not", add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise SystemExit(f"' not' is {len(ids)} tokens under the base tokenizer: {ids}")
    return {"text": " not", "id": int(ids[0]), "tokenizer": f"{BASE_REPO}@{BASE_REVISION} (AutoTokenizer, {snap})",
            "decoded": tok.decode(ids)}


def red_arms(recs: dict, path: Path) -> dict:
    """The red-arm rows, fixed before running (written once, then read)."""
    if path.exists():
        return json.loads(path.read_text())
    semif = [i for i in recs if recs[i]["source"] == "semif_authored144"]
    pairs = [("tv4_000", "tv4_001"), (semif[0], semif[1]), (semif[2], semif[3]), ("own_t01", "own_t03"),
             ("tv4x_qnli_00", "tv4x_paws_00")]
    neg = not_token_id()
    runs = []
    for a, b in pairs:
        qa, qb = recs[a]["questions"][0], recs[b]["questions"][0]
        if qa["type"] != qb["type"] or recs[a]["source"] != recs[b]["source"]:
            raise SystemExit(f"red pair {a} / {b} is not the same kind")
        la, lb = qa["state_len"], qb["state_len"]
        ia, ib = qa["row_ids"], qb["row_ids"]
        assert ia[0] == STATE_ID and ib[0] == STATE_ID and ia[la] == Q_ID and ib[lb] == Q_ID
        runs.append({"id": a, "q": 0, "variant": "base"})
        sh = lb - la
        runs.append({"id": a, "q": 0, "variant": "state_swap", "state_from": b,
                     "ids": ia[:1] + ib[1:lb] + ia[la:], "decide": qa["decide"] + sh, "opts": [o + sh for o in qa["opts"]]})
        runs.append({"id": a, "q": 0, "variant": "instr_negate",
                     "ids": ia[:la + 1] + [neg["id"]] + ia[la + 1:], "decide": qa["decide"] + 1,
                     "opts": [o + 1 for o in qa["opts"]]})
    doc = {"pairs": [list(p) for p in pairs], "not_token": neg, "runs": runs,
           "rule": "red = the arm moves an argmax on at least one row AND its max |dp| against the unperturbed run "
                   "exceeds 0.02",
           "written": datetime.now().astimezone().isoformat(timespec="seconds")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


# v2 instruction arm: a grammatical "not" inside the question of five noul rows (old -> new, each exactly once).
V2_NEGATIONS = (
    ("tv4x_tweet_offensive_00", 0, "Is this post offensive?", "Is this post not offensive?"),
    ("tv4x_qnli_00", 0, "Does the sentence contain the answer", "Does the sentence not contain the answer"),
    ("tv4x_paws_00", 0, "Does this sentence mean the same thing", "Does this sentence not mean the same thing"),
    ("tv4x_legacy_holdout_00", 0, "Is the refund authorized?", "Is the refund not authorized?"),
    ("own_t05", 2, "Would the reviewer recommend this jacket?", "Would the reviewer not recommend this jacket?"),
)
RED_RULE = ("red = the perturbed runs against the unperturbed ones fail the gate's own bar: (a) an "
            "argmax moves on at least one question whose oracle top-2 margin is above 0.02, or (b) max |dp| > "
            "0.02, or (c) the mean over runs of the run's mean |dp| > 0.002")


def user_tokens(tok, text: str) -> list[int]:
    """kev.model.user_tokens: `<|name|>` -> `<¦name¦>`, then tokenized without special tokens."""
    return tok(re.sub(r"<\|([A-Za-z0-9_]+)\|>", r"<¦\1¦>", text), add_special_tokens=False).input_ids


def write_red_v2(recs: dict, path: Path) -> dict:
    """red_arms_v2.json, written once before any v2 run: state_swap with SemIf pairs whose states differ (the v1 first
    pair shares one state, a no-op), and instr_negate_v2 = a grammatical "not" inside five noul questions, the
    instructions re-encoded with the base tokenizer and the row rebuilt around them."""
    from transformers import AutoTokenizer

    if path.exists():
        raise SystemExit(f"{path} exists: the v2 rows are fixed once")
    snap = hf_snapshot(BASE_REPO, revision=BASE_REVISION)
    tok = AutoTokenizer.from_pretrained(snap)

    def state(rid: str) -> list[int]:
        q = recs[rid]["questions"][0]
        return q["row_ids"][:q["state_len"]]

    semif = [i for i in recs if recs[i]["source"] == "semif_authored144"]
    # pair 1: the first SemIf record with the first later record that asks the same claim on a different state
    s0 = semif[0]
    s0_b = next(i for i in semif[1:] if recs[i]["questions"][0]["instr"] == recs[s0]["questions"][0]["instr"]
                and state(i) != state(s0))
    pairs = [("tv4_000", "tv4_001"), (s0, s0_b), (semif[2], semif[3]), ("own_t01", "own_t03"),
             ("tv4x_qnli_00", "tv4x_paws_00")]
    runs, bases = [], []

    def base(rid: str, k: int) -> None:
        if (rid, k) not in bases:
            bases.append((rid, k))
            runs.append({"id": rid, "q": k, "variant": "base"})

    for a, b in pairs:
        qa, qb = recs[a]["questions"][0], recs[b]["questions"][0]
        if qa["type"] != qb["type"] or recs[a]["source"] != recs[b]["source"] or state(a) == state(b):
            raise SystemExit(f"red pair {a} / {b} is not the same kind with two different states")
        la, lb = qa["state_len"], qb["state_len"]
        ia, ib = qa["row_ids"], qb["row_ids"]
        assert ia[0] == STATE_ID and ib[0] == STATE_ID and ia[la] == Q_ID and ib[lb] == Q_ID
        base(a, 0)
        sh = lb - la
        runs.append({"id": a, "q": 0, "variant": "state_swap", "state_from": b,
                     "ids": ia[:1] + ib[1:lb] + ia[la:], "decide": qa["decide"] + sh, "opts": [o + sh for o in qa["opts"]]})
    for rid, k, old, new in V2_NEGATIONS:
        q = recs[rid]["questions"][k]
        instr = q["instr"]
        if q["type"] != "noul" or instr.count(old) != 1:
            raise SystemExit(f"{rid} q{k}: not a noul question with exactly one {old!r}")
        instr2 = instr.replace(old, new)
        L_, ids = q["state_len"], q["row_ids"]
        old_ids, new_ids = user_tokens(tok, instr), user_tokens(tok, instr2)
        if ids[L_] != Q_ID or ids[L_ + 1:L_ + 1 + len(old_ids)] != old_ids:
            raise SystemExit(f"{rid} q{k}: the base tokenizer does not reproduce the oracle's instruction ids")
        sh = len(new_ids) - len(old_ids)
        base(rid, k)
        runs.append({"id": rid, "q": k, "variant": "instr_negate_v2", "ids": ids[:L_ + 1] + new_ids + ids[L_ + 1 + len(old_ids):],
                     "decide": q["decide"] + sh, "opts": [o + sh for o in q["opts"]],
                     "instr_old": instr, "instr_new": instr2, "instr_ids_old": old_ids, "instr_ids_new": new_ids})
    n_states = len({tuple(state(i)) for i in semif})
    doc = {"version": 2, "pairs": [list(p) for p in pairs],
           "semif_states": {"records": len(semif), "distinct_states": n_states,
                            "v1_first_pair": [semif[0], semif[1], "same state: the v1 swap was a no-op (dp exactly 0)"]},
           "negations": [{"id": r, "q": k, "old": o, "new": n} for r, k, o, n in V2_NEGATIONS],
           "tokenizer": f"{BASE_REPO}@{BASE_REVISION} (AutoTokenizer, {snap})", "runs": runs, "rule": RED_RULE,
           "written": datetime.now().astimezone().isoformat(timespec="seconds")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def score_red(head, recs: dict, red_doc: dict, rarr: dict, gate_arrays: dict) -> dict:
    """Every arm of a red rows file: the perturbed runs against the unperturbed runs of the same process, under the
    current red rule (and the v1 rule, kept for the v1 transcript); the unperturbed runs against the gate's runs."""
    base_bits = []
    for x in red_doc["runs"]:
        if x["variant"] != "base":
            continue
        g = gate_arrays.get((x["id"], x["q"]))
        b = rarr[(x["id"], x["q"], "base")]["hidden"]
        base_bits.append({"row": f"{x['id']}:q{x['q']}",
                          "bit_equal_gate_run": bool(np.array_equal(g, b)) if g is not None else None})
    arms = {}
    for arm in sorted({x["variant"] for x in red_doc["runs"]} - {"base"}):
        items = []
        for x in red_doc["runs"]:
            if x["variant"] != arm:
                continue
            q = recs[x["id"]]["questions"][x["q"]]
            base = score(head, q, rarr[(x["id"], x["q"], "base")]["hidden"])
            pert = score(head, q, rarr[(x["id"], x["q"], arm)]["hidden"], decide=x["decide"], opts=x["opts"])
            dp = np.abs(np.asarray(pert["probs"]) - np.asarray(base["probs"]))
            items.append({"row": f"{x['id']}:q{x['q']}", "type": q["type"], "state_from": x.get("state_from"),
                          "instr_old": x.get("instr_old"), "instr_new": x.get("instr_new"),
                          "tokens": len(x["ids"]), "near_tie": bool(q["near_tie"]),
                          "argmax_moved": pert["argmax"] != base["argmax"],
                          "max_abs_dp_vs_base": float(dp.max()), "mean_abs_dp_vs_base": float(dp.mean()),
                          "base_probs": base["probs"], "perturbed_probs": pert["probs"],
                          "base_argmax": base["argmax"], "perturbed_argmax": pert["argmax"],
                          "oracle_argmax": base["argmax_oracle"]})
        moved = sum(i["argmax_moved"] for i in items)
        moved_far = sum(i["argmax_moved"] and not i["near_tie"] for i in items)
        mdp = max(i["max_abs_dp_vs_base"] for i in items)
        mean_run = float(np.mean([i["mean_abs_dp_vs_base"] for i in items]))
        arms[arm] = {"rows": len(items), "argmax_moved": moved, "argmax_moved_non_near_tie": moved_far,
                     "max_abs_dp_vs_base": mdp, "mean_of_run_mean_abs_dp_vs_base": mean_run,
                     "red_facts7": {"a_argmax_non_near_tie": moved_far >= 1, "b_max_abs_dp": mdp > BAR["max_abs_dp"],
                                    "c_mean_of_run_means": mean_run > BAR["mean_of_run_mean_abs_dp"]},
                     "red": moved_far >= 1 or mdp > BAR["max_abs_dp"] or mean_run > BAR["mean_of_run_mean_abs_dp"],
                     "red_v1_rule": moved >= 1 and mdp > BAR["max_abs_dp"], "items": items}
    return {"rule": RED_RULE, "v1_rule": "argmax moved on >= 1 row AND max |dp| > 0.02", "arms": arms,
            "base_vs_gate_run": base_bits,
            "base_bit_equal_all": all(b["bit_equal_gate_run"] for b in base_bits if b["bit_equal_gate_run"] is not None),
            "all_arms_red": all(a["red"] for a in arms.values())}


def gate_arrays_of(transcript: dict) -> dict:
    """(id, q) -> the fp16 hidden rows of every base run of a gate transcript (its shard npz)."""
    out, zs = {}, {}
    for r in transcript["runs"]:
        npz = next((p["npz"] for p in transcript["processes"] if p.get("shard") == r["shard"] and p.get("npz")), None)
        if npz and Path(npz).exists():
            z = zs.setdefault(npz, np.load(npz))
            out[(r["id"], r["q"])] = z[f"{r['npz_key']}__hidden"]
    return out


def compare_with(runs: list[dict], arrays: dict, other_path: Path, S: int) -> dict:
    """The same rows in another transcript (another chunk width): p and hidden side by side, and the times."""
    other = json.loads(other_path.read_text())
    o_runs = {(r["id"], r["q"]): r for r in other["runs"]}
    o_arrays, zs = {}, {}
    for r in other["runs"]:
        npz = next((p["npz"] for p in other["processes"] if p.get("shard") == r["shard"] and p.get("npz")), None)
        if npz and Path(npz).exists():
            z = zs.setdefault(npz, np.load(npz))
            o_arrays[(r["id"], r["q"])] = z[f"{r['npz_key']}__hidden"]
    rows, mine_t, other_t = [], [], []
    for r in runs:
        o = o_runs.get((r["id"], r["q"]))
        if o is None:
            continue
        row = {"id": r["id"], "q": r["q"], "tokens": r["tokens"],
               "max_abs_dp": float(np.max(np.abs(np.asarray(r["probs"]) - np.asarray(o["probs"])))),
               "argmax_equal": r["argmax"] == o["argmax"], "calls": [o["calls"], r["calls"]],
               "decode_seconds": [o["decode_seconds"], r["decode_seconds"]],
               "max_abs_dp_vs_oracle": [o["max_abs_dp"], r["max_abs_dp"]]}
        h_other = o_arrays.get((r["id"], r["q"]))
        if h_other is not None:
            a = arrays[(r["id"], r["q"], "base")]["hidden"]
            row["hidden_max_abs_diff"] = float(np.max(np.abs(a.astype(np.float32) - h_other.astype(np.float32))))
            row["hidden_bit_equal"] = bool(np.array_equal(a, h_other))
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
                "decode_ms_total": float(sum(x["decode_seconds"] for x in rs) * 1e3),
                "calls_total": int(sum(x["calls"] for x in rs))}

    common = {(x["id"], x["q"]) for x in mine_t} & {(x["id"], x["q"]) for x in other_t}
    mine_c = [x for x in mine_t if (x["id"], x["q"]) in common]
    other_c = [x for x in other_t if (x["id"], x["q"]) in common]
    return {"other": str(other_path), "other_chunk": S_other, "other_bundle": other["bundle"]["name"], "runs": len(rows),
            "max_abs_dp": max((x["max_abs_dp"] for x in rows), default=None),
            "argmax_equal_runs": sum(x["argmax_equal"] for x in rows),
            "hidden_max_abs_diff": max((x.get("hidden_max_abs_diff", 0.0) for x in rows), default=None),
            "hidden_bit_equal_runs": sum(bool(x.get("hidden_bit_equal")) for x in rows),
            "vs_oracle_on_shared_runs": {"order": ["other", "this"],
                                         "max_abs_dp": [max((x["max_abs_dp_vs_oracle"][i] for x in rows), default=None)
                                                        for i in (0, 1)]},
            "timing_warm_common_runs": {"other": times(other_c, S_other, [np.asarray(x["call_ms"]) for x in other_c]),
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
            "head_sha256": {f.name: sha256_file(f) for f in sorted((bundle / "head").iterdir())},
            "aimodelc": tree_digest(aimodelc)}


def gate(args) -> int:
    configure(args.model, args.head_dir, args.readout_dir)
    bundle = Path(args.bundle).expanduser().resolve()
    meta = json.loads((bundle / "metadata.json").read_text())
    S, dyn = shape_of(meta)
    if args.shared and not dyn:
        raise SystemExit("--shared: round 14's dynamic-S bundles only")
    if dyn and args.host_q is not None:   # round 15: an override of the metadata's query_len_multiple
        dyn.update({"q": int(args.host_q), "q_from": "--host-q"})
    elif args.host_q is not None:
        raise SystemExit("--host-q: round 14's dynamic-S bundles only")
    if dyn and args.call_max is not None:   # round 15: an override of the metadata's query_len_call_max
        dyn.update({"call_max": int(args.call_max), "call_max_from": "--call-max"})
    elif args.call_max is not None:
        raise SystemExit("--call-max: round 14's dynamic-S bundles only")
    if dyn and not (dyn["min"] <= dyn["call_max"] <= dyn["max"]):
        raise SystemExit(f"call max {dyn['call_max']} outside the graph's {dyn['min']}..{dyn['max']}")
    max_ctx = int(meta["language"]["max_context_length"])
    aimodelc = (Path(args.aimodelc).expanduser().resolve() if args.aimodelc
                else bundle.parent.parent / "bundles_aotc" / f"{meta['name']}.h16c.aimodelc")
    if not aimodelc.exists():
        raise SystemExit(f"no AOT asset {aimodelc} (export_decoder.py --aot)")
    if Path(args.transcript).exists() and not args.rescore:
        raise SystemExit(f"{args.transcript} exists: transcripts are never overwritten")
    if args.oracle:
        set_oracle(Path(args.oracle))
    doc, recs = load_oracle()
    s80_path = READOUT / "s80.json"
    if args.subset == "s80" and not s80_path.exists():
        write_s80(recs, s80_path)
    expected = subset_rows(recs, args.subset, s80_path)
    if args.rows:   # round 6: a fixed list of [id, q] rows (the no-efr probes, which re-specialize per position length)
        want = [tuple(x) for x in json.loads(Path(args.rows).read_text())["rows"]]
        unknown = [x for x in want if x not in set(expected)]
        if unknown:
            raise SystemExit(f"{args.rows}: rows not in the oracle / subset: {unknown[:5]}")
        expected = want
    if args.records:   # round 14: keep these records (or every own record) of the subset, in subset order
        keep = ({rid for rid, r in recs.items() if r["source"].startswith("own_")} if args.records == "own"
                else set(args.records.split(",")))
        expected = [x for x in expected if x[0] in keep]
        if not expected:
            raise SystemExit(f"--records {args.records}: no rows left")
    tag = args.tag or meta["name"].replace("kev_0_8b_decode_", "")
    work = READOUT / tag
    work.mkdir(parents=True, exist_ok=True)
    red_doc = red_arms(recs, READOUT / "red_arms.json") if args.red else None
    t0 = time.monotonic()
    others_start, lock_start = other_gpu_processes(), gpu_lock_state()
    cspec = {"hidden": HIDDEN, "state_shapes": STATE_SHAPES, "dynamic": dyn, "shared": bool(args.shared)}
    parts = split_records(expected) if args.shared else split(expected)
    shards = [{"aimodelc": str(aimodelc), "chunk": S, "max_ctx": max_ctx, "oracle": str(ORACLE), **cspec,
               "runs": [{"id": i, "q": k, "variant": "base"} for i, k in part], "out": str(work / f"shard_{n:02d}")}
              for n, part in enumerate(parts)]
    cspec = {**cspec, "shared": False}   # the red arms run direct
    red_spec = ({"aimodelc": str(aimodelc), "chunk": S, "max_ctx": max_ctx, "oracle": str(ORACLE), **cspec,
                 "runs": red_doc["runs"], "out": str(work / "red_arms")} if red_doc else None)
    if args.rescore:
        recs_s = [read_shard(sp) for sp in shards]
        red_recs = [read_shard(red_spec)] if red_spec else []
    else:
        recs_s = run_shards(f"gate {tag}", shards)
        red_recs = run_shards(f"red {tag}", [red_spec]) if red_spec else []
    gpu_seconds = time.monotonic() - t0
    others_end, lock_end = other_gpu_processes(), gpu_lock_state()
    runs, procs, arrays = collect(recs_s)
    print(f"scoring {len(runs)} rows through the pointer head ...", flush=True)
    t1 = time.monotonic()
    head = head_of()
    scored = []
    for r in runs:
        rec_o = recs[r["id"]]
        q = rec_o["questions"][r["q"]]
        a = arrays[(r["id"], r["q"], r["variant"])]
        sc = score(head, q, a["hidden"])
        item = {**r, "source": rec_o["source"], "qid": q["qid"], "type": q["type"], **sc,
                "call_ms": a["call_ms"].tolist(), "_call_ms": a["call_ms"]}
        ref = (oracle_hidden(r["id"], r["q"]) if r["id"] in HIDDEN_RECORDS
               or (DYNAMIC and (ORACLE / "hidden" / f"{r['id']}.npz").exists()) else None)   # r14: the longrow oracle too
        if ref is not None:
            item["hidden"] = position_cos(a["hidden"], ref, q["row_ids"])
        scored.append(item)
    s = summarize(scored, expected)
    resets = all(p.get("reset_check", {}).get("bit_equal", False) for p in procs)
    by_src = {}
    for x in scored:
        by_src.setdefault(x["source"], []).append(x)
    by_src = {k: summarize(v, [(x["id"], x["q"]) for x in v]) for k, v in sorted(by_src.items())}
    red = None
    if red_recs:
        rruns, rprocs, rarr = collect(red_recs)
        red = {"rows_file": str(READOUT / "red_arms.json"), "rows_file_sha256": sha256_file(READOUT / "red_arms.json"),
               "pairs": red_doc["pairs"], "not_token": red_doc["not_token"], "process": rprocs,
               **score_red(head, recs, red_doc, rarr, {(i, q): a["hidden"] for (i, q, v), a in arrays.items()
                                                       if v == "base"})}
        red["reset_bit_equal"] = all(p.get("reset_check", {}).get("bit_equal", False) for p in rprocs)
    ok, checks = verdict(s, resets and (red is None or red["reset_bit_equal"]))
    near = [{"id": r["id"], "q": r["q"], "qid": r["qid"], "oracle_top2_margin": r["oracle_top2_margin"],
             "argmax_equal": r["argmax_equal"], "max_abs_dp": r["max_abs_dp"], "probs": r["probs"],
             "probs_oracle": r["probs_oracle"]} for r in scored if r["near_tie"]]
    descs = [p["descriptor"] for p in procs if "descriptor" in p]
    record = {
        "schema": "kev-decoder-readout-gate/1",
        "gate": "decoder alone on the Mac GPU (oracle row ids), the fp16 hidden read through the fp32 pointer head",
        "bundle": bundle_record(bundle, aimodelc), "chunk": S, "max_ctx": max_ctx,
        "dynamic": dyn, "mode": ("shared" if args.shared else "direct") if dyn else "static",
        "shared_groups": [g for p in recs_s for g in p.get("shared_groups", [])] if args.shared else None,
        "readout": meta["decision"]["readout"],
        "descriptor": descs[0] if descs else None,
        "descriptor_same_in_every_process": all(d == descs[0] for d in descs) if descs else None,
        "contract": dyn_contract(S, dyn["padded"]) if dyn else contract(S),
        "oracle": {"path": str(ORACLE / "records_oracle.json"), "sha256": sha256_file(ORACLE / "records_oracle.json"),
                   "checkpoint": doc["model"]["checkpoint"], "versions": doc["versions"]},
        "model": args.model, "head": head.provenance,
        "subset": args.subset, "rows_file": ({"path": str(Path(args.rows).resolve()), "sha256": sha256_file(Path(args.rows))}
                                             if args.rows else None),
        "subset_file": ({"path": str(s80_path), "sha256": sha256_file(s80_path)} if args.subset == "s80" else None),
        "script": {"path": "conversion/kev/readout_gate.py", "sha256": sha256_file(Path(__file__).resolve())},
        "bar": {**BAR, "argmax": "every question with an oracle top-2 margin above 0.02; near-ties listed apart",
                "max_abs_dp_applies_to": "every option of every question, near-ties included",
                "reset": "bit-equal hidden rows in every process",
                "mean_of_run_mean_abs_dp_definition": "mean over runs (one run = one row = one question) of the "
                                                      "mean |dp| over that question's options"},
        "environment": env_record(),
        "gpu_lock": {"start": lock_start, "end": lock_end},
        "other_gpu_processes": {"start": others_start, "end": others_end},
        "processes": procs, "summary": s, "by_source": by_src, "near_ties": near,
        "checks": checks, "result": "PASS" if ok else "FAIL",
        "red_arm": red, "timing": timing(scored, procs, S, recs),
        "seconds": {"gpu_processes": gpu_seconds, "scoring": time.monotonic() - t1, "total": time.monotonic() - t0},
        "runs": [{k: v for k, v in r.items() if k != "_call_ms"} for r in scored],
    }
    if args.compare_with:
        record["compare_with"] = compare_with(scored, arrays, Path(args.compare_with), S)
    if args.note:
        record["note"] = args.note
    record["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    Path(args.transcript).parent.mkdir(parents=True, exist_ok=True)
    Path(args.transcript).write_text(json.dumps(record, indent=1) + "\n")
    print(f"{record['result']}: {meta['name']} rows {s['runs']}/{s['expected_runs']} argmax {s['argmax_equal']}/"
          f"{s['questions']} (non-near-tie {s['argmax_equal_non_near_tie']}/{s['questions_non_near_tie']}, near-tie "
          f"{s['argmax_equal_near_tie']}/{s['near_tie_questions']}) max|dp| {s['max_abs_dp']:.6f} mean "
          f"{s['mean_of_run_mean_abs_dp']:.6f} min cos {s['min_pos_cos']} reset {resets} worst {s['worst_run']}")
    print(f"  hidden rows {s['hidden_rows']}, positions {s['positions']}, below cos {json.dumps(s['positions_below'])}; "
          f"calls {s['calls']}, pad waste {s['pad_waste']:.4f}, ms/call warm median {s['ms_per_call_warm_median']}")
    for k, v in checks.items():
        print(f"  {k}: {'ok' if v else 'FAIL'}")
    for k, v in by_src.items():
        print(f"  {k:26s} rows {v['runs']:3d} argmax {v['argmax_equal']}/{v['questions']} max|dp| {v['max_abs_dp']:.2e} "
              f"mean {v['mean_of_run_mean_abs_dp']:.2e} ms/call {v['ms_per_call_median']:.2f}")
    if red:
        for arm, a in red["arms"].items():
            print(f"red {arm}: argmax moved {a['argmax_moved']}/{a['rows']} max|dp| vs base {a['max_abs_dp_vs_base']:.3f} "
                  f"-> {'RED' if a['red'] else 'NOT RED'}")
        print(f"red base = gate run (bit): {[b['bit_equal_gate_run'] for b in red['base_vs_gate_run']]}, reset "
              f"{red['reset_bit_equal']}")
    if "compare_with" in record:
        c = record["compare_with"]
        print(f"vs {c['other_bundle']} (S={c['other_chunk']}): rows {c['runs']} max|dp| {c['max_abs_dp']:.2e} argmax-equal "
              f"{c['argmax_equal_runs']} hidden max|d| {c['hidden_max_abs_diff']} timing {json.dumps(c['timing_warm_common_runs'])}")
    print(f"transcript: {args.transcript}")
    return 0 if ok else 1


def red_cmd(args) -> int:
    """Red arms only: the rows of a fixed rows file in their own process, scored against the unperturbed runs of
    that process and compared bit for bit with an existing gate transcript's runs of the same rows."""
    configure(args.model, args.head_dir, args.readout_dir)
    if args.oracle:
        set_oracle(Path(args.oracle))
    bundle = Path(args.bundle).expanduser().resolve()
    meta = json.loads((bundle / "metadata.json").read_text())
    S, dyn = shape_of(meta)
    if dyn and args.host_q is not None:
        dyn.update({"q": int(args.host_q), "q_from": "--host-q"})
    if dyn and args.call_max is not None:
        dyn.update({"call_max": int(args.call_max), "call_max_from": "--call-max"})
    max_ctx = int(meta["language"]["max_context_length"])
    aimodelc = (Path(args.aimodelc).expanduser().resolve() if args.aimodelc
                else bundle.parent.parent / "bundles_aotc" / f"{meta['name']}.h16c.aimodelc")
    if Path(args.transcript).exists():
        raise SystemExit(f"{args.transcript} exists: transcripts are never overwritten")
    red_file = Path(args.red_file).expanduser().resolve()
    doc, recs = load_oracle()
    if not red_file.exists():
        if red_file.name != "red_arms_v2.json":
            raise SystemExit(f"no rows file {red_file}")
        write_red_v2(recs, red_file)
    red_doc = json.loads(red_file.read_text())
    gate_t = json.loads(Path(args.gate_transcript).read_text())
    if gate_t["bundle"]["aimodelc"]["tree_sha256"] != tree_digest(aimodelc)["tree_sha256"]:
        raise SystemExit("the gate transcript was taken on another asset")
    tag = args.tag or meta["name"].replace("kev_0_8b_decode_", "")
    work = READOUT / tag
    spec = {"aimodelc": str(aimodelc), "chunk": S, "max_ctx": max_ctx, "oracle": str(ORACLE), "hidden": HIDDEN,
            "state_shapes": STATE_SHAPES, "dynamic": dyn, "shared": False, "runs": red_doc["runs"],
            "out": str(work / red_file.stem)}
    t0 = time.monotonic()
    others_start, lock_start = other_gpu_processes(), gpu_lock_state()
    rruns, rprocs, rarr = collect(run_shards(f"red {tag}", [spec]))
    others_end, lock_end = other_gpu_processes(), gpu_lock_state()
    head = head_of()
    red = score_red(head, recs, red_doc, rarr, gate_arrays_of(gate_t))
    red["reset_bit_equal"] = all(p.get("reset_check", {}).get("bit_equal", False) for p in rprocs)
    record = {
        "schema": "kev-decoder-readout-red-arms/1",
        "bundle": bundle_record(bundle, aimodelc), "chunk": S, "max_ctx": max_ctx,
        "rows_file": {"path": str(red_file), "sha256": sha256_file(red_file), "version": red_doc.get("version", 1)},
        "model": args.model, "head": head.provenance,
        "oracle": {"path": str(ORACLE / "records_oracle.json"), "sha256": sha256_file(ORACLE / "records_oracle.json"),
                   "checkpoint": doc["model"]["checkpoint"]},
        "gate_transcript": {"path": str(Path(args.gate_transcript).resolve()),
                            "sha256": sha256_file(Path(args.gate_transcript)), "result": gate_t["result"]},
        "script": {"path": "conversion/kev/readout_gate.py", "sha256": sha256_file(Path(__file__).resolve())},
        "environment": env_record(), "gpu_lock": {"start": lock_start, "end": lock_end},
        "other_gpu_processes": {"start": others_start, "end": others_end},
        "process": rprocs, **red,
        "result": "RED (every arm)" if red["all_arms_red"] and red["reset_bit_equal"] and red["base_bit_equal_all"]
        else "NOT RED on every arm",
        "seconds": time.monotonic() - t0,
    }
    if args.note:
        record["note"] = args.note
    record["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    Path(args.transcript).parent.mkdir(parents=True, exist_ok=True)
    Path(args.transcript).write_text(json.dumps(record, indent=1) + "\n")
    for arm, a in red["arms"].items():
        print(f"red {arm}: argmax moved {a['argmax_moved']}/{a['rows']} (non-near-tie {a['argmax_moved_non_near_tie']}) "
              f"max|dp| {a['max_abs_dp_vs_base']:.3f} mean {a['mean_of_run_mean_abs_dp_vs_base']:.4f} -> "
              f"{'RED' if a['red'] else 'NOT RED'} {json.dumps(a['red_facts7'])}")
        for i in a["items"]:
            print(f"   {i['row']}: moved {i['argmax_moved']} max|dp| {i['max_abs_dp_vs_base']:.3f} "
                  f"{[round(x, 3) for x in i['base_probs']]} -> {[round(x, 3) for x in i['perturbed_probs']]}")
    print(f"base = gate run (bit): {[b['bit_equal_gate_run'] for b in red['base_vs_gate_run']]}, reset {red['reset_bit_equal']}")
    print(f"{record['result']}; transcript: {args.transcript}")
    return 0 if record["result"].startswith("RED") else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rd = sub.add_parser("red", help="red arms only, from a fixed rows file, against an existing gate transcript")
    rd.add_argument("bundle")
    rd.add_argument("--red-file", required=True, help="rows file; red_arms_v2.json is written first if missing")
    rd.add_argument("--gate-transcript", required=True, help="the gate transcript of the same bundle (base rows)")
    rd.add_argument("--transcript", required=True)
    rd.add_argument("--tag", help="shard directory name under readout/ (default: from the bundle name)")
    rd.add_argument("--note")
    rd.add_argument("--oracle", help="oracle directory holding records_oracle.json (default: the --model's)")
    rd.add_argument("--host-q", type=int, help="round 14: call lengths are multiples of q (1 = no padding); default: "
                                               "the bundle's language.query_len_multiple (absent = 1)")
    rd.add_argument("--call-max", type=int, help="round 15: the longest call; default: the bundle's "
                                                 "language.query_len_call_max (absent = the graph's cap)")
    a = sub.add_parser("run", help="the gate: workers on the GPU, then the pointer head on their hidden rows")
    a.add_argument("bundle")
    a.add_argument("--transcript", required=True)
    a.add_argument("--subset", default="all", choices=["all", "s80"])
    a.add_argument("--red", action="store_true", help="add the two red arms (own process)")
    a.add_argument("--compare-with", help="another transcript of the same rows (another chunk width)")
    a.add_argument("--tag", help="shard directory name under readout/ (default: from the bundle name)")
    a.add_argument("--rescore", action="store_true", help="re-score existing shards, no GPU runs")
    a.add_argument("--rows", help="a JSON file {\"rows\": [[id, q], ...]}: run only these rows (in this order)")
    a.add_argument("--shared", action="store_true",
                   help="round 14 (dynamic-S bundles): the state ids once per record, every question's branch on a copy")
    a.add_argument("--records", help="round 14: comma list of record ids (or 'own') to keep from the subset")
    a.add_argument("--host-q", type=int, help="round 14: call lengths are multiples of q (1 = no padding); default: "
                                              "the bundle's language.query_len_multiple (absent = 1)")
    a.add_argument("--call-max", type=int, help="round 15: the longest call; default: the bundle's "
                                                "language.query_len_call_max (absent = the graph's cap)")
    a.add_argument("--note", help="free text kept in the transcript")
    a.add_argument("--oracle", help="oracle directory holding records_oracle.json (default <work>/oracle; the held-out "
                                    "gate passes <work>/oracle/heldout); not with --red or --subset s80")
    for sp in (rd, a):
        sp.add_argument("--model", default="kev-0.8b", choices=sorted(MODELS),
                        help="contract (hidden, state shapes) and the default oracle / head / shard directories")
        sp.add_argument("--head-dir", help="head.safetensors + kev_head.json (default: <the --model's oracle>/head)")
        sp.add_argument("--readout-dir", help="where the shard directories go (default <work>/readout[_4b])")
        sp.add_argument("--aimodelc", help="the compiled asset to load (default <bundles_aotc>/<name>.h16c.aimodelc; "
                                           "round 6 gates assets compiled without --expect-frequent-reshapes this way)")
    aw = sub.add_parser("worker")
    aw.add_argument("--spec", required=True)
    args = ap.parse_args()
    if args.cmd == "worker":
        return worker(Path(args.spec))
    if args.cmd == "red":
        return red_cmd(args)
    return gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
