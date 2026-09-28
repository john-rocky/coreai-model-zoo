#!/usr/bin/env python3
"""Readout gate: the decider-2b-vision LanguageBundles on the Mac GPU vs the author's fp32 oracle.

Every fixture run is fed one token per step through the S=1 decode graph (AOT h16c `.aimodelc`,
`SpecializationOptions.default()`), from fresh zero states, with the full-length position ramp.
The fp16 logits are read at EVERY slot step of the same pass, restricted to the question's
letter ids (bare `A`.. = 32..), and turned into probabilities with a softmax at T = 1 (float64
here); the oracle is `fixture_oracle.json` from `oracle_decider_vision.py` (the checkpoint's own
`VisionDecisionModel.prepare()` -> `slot_logits()`, fp32 CPU).

  b1    the decoder alone, all 111 runs (35 image rows x g256 / g448 / native + 6 text rows):
        ids = the oracle's ids with the <|image_pad|> block mapped to V+k, image_embeds = the
        oracle's fp32 tower output (npz) cast to fp16 and zero-padded to 256 rows.
        `--red` adds the red arm: r14 g256 with image_embeds zeroed (the fp32 torch module moved
        its argmax A -> D in round 3), run in its own process next to its own base run.
  b2    end to end, the 70 g256 / g448 runs: the fixture's image file -> `host.preprocess`
        (Pillow BICUBIC to the tile) -> fp16w32 tower `.aimodelc` -> image_embeds (fp32 -> fp16)
        -> decoder, with the ids built by `host.build_ids` from the bundle's own tokenizer and
        checked against the oracle's. `--fp16-tower-bundle` adds the arm with the fp16 tower
        (min-row 0.53..0.85 in round 2) in front of that bundle: values only, no verdict.

Chunk order (`--order chunk`, a bundle with a "prefill" function of static S = C next to "main"):
per row, slots ascending, from fresh zero states: while s - cursor + 1 >= C, "prefill" gets
ids[cursor : cursor + C] with position_ids 0..cursor+C-1 and its (last-position) logits are the
slot's when cursor + C - 1 == s, cursor += C; then "main" one token at a time up to and including
s (the logits at cursor == s are the slot's). The last slot is the row's last token. b1
`--also-s1` then runs the same bundle in S=1 order ("main" only) and adds the per-slot chunk vs
S=1 table (`chunk_vs_s1`) and the S=1 order's own verdict (`s1_order`). b2 `--append` adds the
new arms to an existing e2e transcript and leaves its arms as they are.

Bar, fixed before any result (the same for b1 and b2, per bundle): every run present; slots found
by the host rule = the oracle's, every run; letter argmax = the oracle's on every slot (the
fixture's smallest oracle top-2 gap is 0.056, so no near-tie exemption); full-vocabulary top-1 =
the oracle's argmax letter on every slot; max |dp| <= 0.02; mean over runs of the run's mean |dp|
(over all of its slots and options) <= 0.002; every process re-runs its first run at the end and
reproduces its slot logits bit for bit (state reset proof).

Process split: the Python runtime leaks one IOSurface per call, so a process takes at most 14
runs + the reset re-run (15 prompts). The driver runs the processes one after another; each
worker writes `<work>/<gate>/<bundle>/shard_NN.json` (per run) and `.npz` (the full fp16 slot
logits + every step's wall ms), which the driver merges into the transcript. GPU shared with
other sessions (no _GPU_LOCK): the times are contended reference values.

Run from the worktree root (shared venv, offline):
    HF_HOME=~/code/coreai/_decider2bv/hf HF_HUB_OFFLINE=1 \\
      ../coreai-models/.venv/bin/python conversion/decider_vision/readout_gate_vision.py b1 \\
        ~/code/coreai/_decider2bv/exports/bundles/decider_2b_vision_decode_fp16 \\
        --transcript models/decider-2b-vision/gate-decider-2b-vision-readout-fp16.json
    ... readout_gate_vision.py b1 <exports>/bundles/decider_2b_vision_decode_int8lin_pf16 \\
        --order chunk --also-s1 --red --work-dir ~/code/coreai/_decider2bv/readout_r5 \\
        --transcript models/decider-2b-vision/gate-decider-2b-vision-readout-int8lin_pf16.json
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
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import host  # noqa: E402
from _paths import work_path  # noqa: E402

LANE = work_path("_decider2bv")
ORACLE = LANE / "oracle"
WORK = LANE / "readout_r4"
V = host.VOCAB
LETTER0 = host.LETTER_IDS[0]
GRIDS = {"g256": 8, "g448": 14}
TOWERS = {"fp16w32": "decider_2b_vision_{arm}_vision_fp16w32_aotc", "fp16": "decider_2b_vision_{arm}_vision_aotc"}
BAR = {"max_abs_dp": 0.02, "mean_of_run_mean_abs_dp": 0.002}
RUNS_PER_PROCESS = 14                              # + the reset re-run = 15 prompts
AOT_FLAGS = ["--platform", "macOS", "--preferred-compute", "gpu", "--architecture", "h16c",
             "--expect-frequent-reshapes"]
CONTRACT_INPUTS = {"input_ids": ([1, 1], "int32"), "position_ids": ([1, -1], "int32"),
                   "image_embeds": ([256, 2048], "float16"), "image_rc": ([256, 2], "int32"),
                   "rope_shift_start": ([1], "int32"), "rope_shift_amount": ([1], "int32")}


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


def load_oracle() -> tuple[dict, list[dict]]:
    fx = json.loads((ORACLE / "fixture_oracle.json").read_text())
    return fx, fx["rows"]


def aimodelc_for(bundle: Path) -> Path:
    """`<exports>/bundles_aotc/<name>.h16c.aimodelc` next to `<exports>/bundles/<name>/`; compiled
    here with the exporter's flags if it is missing."""
    meta = json.loads((bundle / "metadata.json").read_text())
    aimodel = bundle / meta["assets"]["main"]
    out_dir = bundle.parent.parent / "bundles_aotc"
    target = out_dir / f"{aimodel.stem}.h16c.aimodelc"
    if target.exists():
        return target
    cb = subprocess.run(["xcrun", "-f", "coreai-build"], capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run([cb, "compile", str(aimodel), "--output", str(out_dir), *AOT_FLAGS], check=True)
    return target


# --------------------------------------------------------------------------- #
# Worker: one process, <= 14 runs + the reset re-run
# --------------------------------------------------------------------------- #
async def maybe(x):
    return await x if inspect.isawaitable(x) else x


def dsc(d) -> tuple[list[int], str]:
    return [int(x) for x in d.shape], str(d.dtype).split(".")[-1]


def fn_desc(fn) -> dict:
    d = fn.desc
    return {"inputs": {n: dsc(d.input_descriptor(n)) for n in d.input_names},
            "outputs": {n: dsc(d.output_descriptor(n)) for n in d.output_names},
            "states": {n: dsc(d.state_descriptor(n)) for n in d.state_names}}


def check_contract(desc: dict, name: str, query_len: int) -> None:
    """The function's inputs = the contract (input_ids [1, query_len]), logits [1, 1, V]."""
    want = dict(CONTRACT_INPUTS, input_ids=([1, query_len], "int32"))
    bad = {n: (desc["inputs"].get(n), w) for n, w in want.items()
           if tuple(desc["inputs"].get(n, ([], ""))[0]) != tuple(w[0]) or desc["inputs"].get(n, ([], ""))[1] != w[1]}
    if bad or set(desc["inputs"]) != set(want):
        raise SystemExit(f"{name}: descriptor differs from the contract: {bad} {sorted(desc['inputs'])}")
    if [tuple(v[0]) for v in desc["outputs"].values()] != [(1, 1, V)]:
        raise SystemExit(f"{name}: outputs {desc['outputs']} != logits [1, 1, {V}]")


def worker(spec_path: Path) -> int:
    import coreai.runtime as rt

    spec = json.loads(spec_path.read_text())
    _, rows = load_oracle()
    orc = {(r["id"], r["arm"]): r for r in rows}
    fx_rows = {r["id"]: r for r in json.loads((LANE / "fixtures" / "rows.json").read_text())["rows"]}
    meta = {im["name"]: im for im in json.loads((LANE / "fixtures" / "meta.json").read_text())["images"]}
    max_ctx = int(spec["max_ctx"])
    tok = None
    if spec["gate"] == "b2":
        import tokenizers
        tok = tokenizers.Tokenizer.from_file(spec["tokenizer"])

    def nd(a):
        return rt.NDArray(np.ascontiguousarray(a))

    out: dict = {"spec": spec, "pid": os.getpid(), "runs": [], "started": time.time()}
    logits_store: dict[str, np.ndarray] = {}

    order = spec.get("order", "s1")
    C = int(spec["chunk"]) if order == "chunk" else None

    async def go() -> None:
        t0 = time.perf_counter()
        model = await maybe(rt.AIModel.load(spec["aimodelc"], rt.SpecializationOptions.default()))
        t1 = time.perf_counter()
        fn = await maybe(model.load_function("main"))
        t2 = time.perf_counter()
        pf = await maybe(model.load_function("prefill")) if order == "chunk" else None
        t3 = time.perf_counter()
        out["load_seconds"] = t3 - t0
        out["load_split_seconds"] = {"model": t1 - t0, "main": t2 - t1, "prefill": (t3 - t2) if pf else None}
        out["function_names"] = list(getattr(model, "function_names", []) or [])
        desc = fn_desc(fn)
        out["descriptor"] = desc
        check_contract(desc, "main", 1)
        if pf is not None:
            pdesc = fn_desc(pf)
            out["prefill_descriptor"] = pdesc
            check_contract(pdesc, "prefill", C)
            if pdesc["states"] != desc["states"]:
                raise SystemExit(f"prefill states {pdesc['states']} != main states {desc['states']}")
        n_max = CONTRACT_INPUTS["image_embeds"][0][0]
        hidden = CONTRACT_INPUTS["image_embeds"][0][1]

        tower_fn = None
        if spec.get("tower"):
            t1 = time.perf_counter()
            tm = await maybe(rt.AIModel.load(spec["tower"]["aimodelc"], rt.SpecializationOptions.default()))
            tower_fn = await maybe(tm.load_function(tm.function_names[0]))
            out["tower_load_seconds"] = time.perf_counter() - t1
            td = tower_fn.desc
            out["tower_descriptor"] = {"inputs": {n: dsc(td.input_descriptor(n)) for n in td.input_names},
                                       "outputs": {n: dsc(td.output_descriptor(n)) for n in td.output_names}}
            tower_in_dtype = np.dtype(out["tower_descriptor"]["inputs"]["patches"][1])

        def fresh_state() -> dict:
            st = {}
            for n, (shape, dt) in desc["states"].items():
                st[n] = nd(np.zeros([max_ctx if s < 0 else s for s in shape], np.dtype(dt)))
            return st

        async def one(run: list) -> tuple[dict, dict[str, np.ndarray]]:
            rid, arm, variant = run
            o = orc[(rid, arm)]
            rec: dict = {"id": rid, "arm": arm, "variant": variant}
            t_run = time.perf_counter()
            emb = np.zeros((n_max, hidden), np.float16)
            arrays: dict[str, np.ndarray] = {}
            if spec["gate"] == "b1":
                if arm == "text":
                    hw = None
                else:
                    _, gh, gw = o["grid_thw"]
                    hw = (gh // host.MERGE, gw // host.MERGE)
                ids = list(o["ids"])
                if hw is not None:
                    first = ids.index(host.IMAGE_PAD)
                    n = hw[0] * hw[1]
                    ids = [V + (i - first) if x == host.IMAGE_PAD else x for i, x in enumerate(ids)]
                    if ids[first:first + n] != [V + k for k in range(n)] or first != 1:
                        raise SystemExit(f"{rid}/{arm}: image block is not one run of {n} ids at index 1")
                    if variant != "embeds_zero":
                        z = np.load(ORACLE / "npz" / f"{rid}__{arm}.npz")
                        emb[:n] = z["image_embeds"].astype(np.float16)
                    start, amount = 1 + n, n - max(hw)
                else:
                    start, amount = host.NO_SHIFT, 0
                slots = host.find_slots(ids)
            else:
                g = GRIDS[arm]
                hw = (g, g)
                fx = fx_rows[rid]
                t1 = time.perf_counter()
                ids, slots, start, amount = host.build_ids(True, fx["context"], fx["questions"], tok, hw)
                rec["tokenize_ms"] = (time.perf_counter() - t1) * 1e3
                rec["ids_equal_oracle"] = host.to_processor_ids(ids) == o["ids"]
                t1 = time.perf_counter()
                patches = host.preprocess(LANE / "fixtures" / meta[fx["image"]]["path"], g)
                rec["preprocess_ms"] = (time.perf_counter() - t1) * 1e3
                t1 = time.perf_counter()
                tout = await maybe(tower_fn(inputs={"patches": nd(patches.astype(tower_in_dtype))}))
                tower_emb = np.asarray(tout["image_embeds"].numpy()).astype(np.float32)
                rec["tower_ms"] = (time.perf_counter() - t1) * 1e3
                want = np.load(ORACLE / "npz" / f"{rid}__{arm}.npz")["image_embeds"].astype(np.float64)
                gt = tower_emb.astype(np.float64)
                rows_cos = (gt * want).sum(-1) / (np.linalg.norm(gt, axis=-1) * np.linalg.norm(want, axis=-1))
                rec["tower_vs_oracle"] = {
                    "cos": float(gt.ravel() @ want.ravel() / (np.linalg.norm(gt) * np.linalg.norm(want))),
                    "min_row": float(rows_cos.min()), "max_abs": float(np.abs(gt - want).max())}
                emb[: g * g] = tower_emb.astype(np.float16)
                arrays["tower_embeds"] = tower_emb.astype(np.float32)
            n_img = 0 if hw is None else hw[0] * hw[1]
            rc = np.zeros((n_max, 2), np.int32)
            if hw is not None:
                k = np.arange(n_img)
                rc[:n_img, 0], rc[:n_img, 1] = k // hw[1], k % hw[1]
            rec.update({"tokens": len(ids), "grid": list(hw) if hw else None, "start": int(start),
                        "amount": int(amount), "slots_host": slots, "slots_oracle": o["slot_idx"],
                        "nopts": o["nopts"]})
            static = {"image_embeds": nd(emb), "image_rc": nd(rc),
                      "rope_shift_start": nd(np.array([start], np.int32)),
                      "rope_shift_amount": nd(np.array([amount], np.int32))}
            state = fresh_state()
            want_slots = set(slots)
            got = []
            t_dec = time.perf_counter()
            if order == "s1":
                step_ms = np.zeros(len(ids), np.float64)
                for t, tk in enumerate(ids):
                    t1 = time.perf_counter()
                    res = await maybe(fn(inputs={"input_ids": nd(np.array([[tk]], np.int32)),
                                                 "position_ids": nd(np.arange(t + 1, dtype=np.int32)[None]),
                                                 **static}, state=state))
                    if t in want_slots:
                        lg = np.asarray(res["logits"].numpy())
                        assert lg.shape == (1, 1, V), lg.shape
                        got.append(lg[0, -1].copy())
                    step_ms[t] = (time.perf_counter() - t1) * 1e3
                arrays["step_ms"] = step_ms
            else:
                # Chunk order: prefill chunks of C while a whole chunk fits before the slot, then
                # S=1 up to and including the slot; the slot's logits come from whichever call
                # processed it as its last token.
                call_ms, call_kind, read_from = [], [], []
                cursor = 0
                for sl in sorted(slots):
                    while sl - cursor + 1 >= C:
                        t1 = time.perf_counter()
                        res = await maybe(pf(inputs={
                            "input_ids": nd(np.array([ids[cursor:cursor + C]], np.int32)),
                            "position_ids": nd(np.arange(cursor + C, dtype=np.int32)[None]), **static},
                            state=state))
                        if cursor + C - 1 == sl:
                            lg = np.asarray(res["logits"].numpy())
                            assert lg.shape == (1, 1, V), lg.shape
                            got.append(lg[0, -1].copy())
                            read_from.append("prefill")
                        call_ms.append((time.perf_counter() - t1) * 1e3)
                        call_kind.append(1)
                        cursor += C
                    while cursor <= sl:
                        t1 = time.perf_counter()
                        res = await maybe(fn(inputs={
                            "input_ids": nd(np.array([[ids[cursor]]], np.int32)),
                            "position_ids": nd(np.arange(cursor + 1, dtype=np.int32)[None]), **static},
                            state=state))
                        if cursor == sl:
                            lg = np.asarray(res["logits"].numpy())
                            assert lg.shape == (1, 1, V), lg.shape
                            got.append(lg[0, -1].copy())
                            read_from.append("main")
                        call_ms.append((time.perf_counter() - t1) * 1e3)
                        call_kind.append(0)
                        cursor += 1
                if cursor != len(ids):
                    raise SystemExit(f"{rid}/{arm}: the row does not end at its last slot ({cursor} of {len(ids)})")
                arrays["call_ms"] = np.asarray(call_ms, np.float64)
                arrays["call_kind"] = np.asarray(call_kind, np.int8)   # 1 = prefill, 0 = main
                rec.update({"prefill_calls": int(sum(call_kind)), "main_calls": len(call_kind) - int(sum(call_kind)),
                            "slot_read_from": read_from})
            rec["decode_seconds"] = time.perf_counter() - t_dec
            rec["wall_seconds"] = time.perf_counter() - t_run
            arrays["slot_logits"] = np.stack(got) if got else np.zeros((0, V), np.float16)
            return rec, arrays

        runs = [list(r) for r in spec["runs"]]
        first_arrays = None
        for i, run in enumerate(runs + [runs[0]]):
            rec, arrays = await one(run)
            if i < len(runs):
                key = f"{i:02d}"
                for k, a in arrays.items():
                    logits_store[f"{key}__{k}"] = a
                rec["index"] = i
                out["runs"].append(rec)
                if i == 0:
                    first_arrays = arrays
                if order == "s1":
                    speed = f"{np.median(arrays['step_ms']):.1f} ms/step"
                else:
                    k = arrays["call_kind"]
                    speed = (f"{rec['prefill_calls']} prefill x {np.median(arrays['call_ms'][k == 1]):.1f} ms + "
                             f"{rec['main_calls']} main x {np.median(arrays['call_ms'][k == 0]):.1f} ms")
                print(f"  [{os.getpid()}] {run[0]}/{run[1]}/{run[2]}: {rec['tokens']} tok, "
                      f"{len(rec['slots_host'])} slots, {rec['wall_seconds']:.2f} s ({speed})", flush=True)
            else:
                same = {k: bool(np.array_equal(first_arrays[k], arrays[k]))
                        for k in first_arrays if k not in ("step_ms", "call_ms")}
                diff = float(np.max(np.abs(first_arrays["slot_logits"].astype(np.float32)
                                           - arrays["slot_logits"].astype(np.float32)))) \
                    if first_arrays["slot_logits"].shape == arrays["slot_logits"].shape else None
                out["reset_check"] = {"run": run, "bit_equal": all(same.values()), "per_array": same,
                                      "slot_logits_max_abs_diff": diff, "wall_seconds": rec["wall_seconds"]}
                print(f"  [{os.getpid()}] reset re-run {run[0]}/{run[1]}: bit-equal {all(same.values())}", flush=True)

    asyncio.run(go())
    out["finished"] = time.time()
    prefix = Path(spec["out"])
    np.savez(prefix.with_suffix(".npz"), **logits_store)
    prefix.with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def softmax64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    p = np.exp(x - x.max())
    return p / p.sum()


def score_run(rec: dict, slot_logits: np.ndarray, o: dict) -> dict:
    """Per-slot and per-run numbers vs the oracle row."""
    slots = []
    deltas = []
    slots_equal = rec["slots_host"] == o["slot_idx"]
    for s, n in enumerate(o["nopts"]):
        if s >= len(slot_logits):
            break
        lg = slot_logits[s]
        letters = lg[LETTER0:LETTER0 + n].astype(np.float64)
        p = softmax64(letters)
        po = np.asarray(o["probs"][s][:n], np.float64)
        d = np.abs(p - po)
        deltas.append(d)
        top5 = np.argsort(-lg.astype(np.float32), kind="stable")[:5]
        full = int(lg.argmax())
        slots.append({"t": rec["slots_host"][s], "nopts": n, "letter_logits": letters.tolist(),
                      "letter_logits_oracle": o["letter_logits"][s][:n], "probs": p.tolist(),
                      "probs_oracle": po.tolist(), "argmax": int(p.argmax()), "argmax_oracle": o["argmax"][s],
                      "argmax_equal": int(p.argmax()) == o["argmax"][s],
                      "full_vocab_top1_id": full, "full_vocab_top1_logit": float(lg[full]),
                      "full_vocab_top1_is_oracle_letter": full == LETTER0 + o["argmax"][s],
                      "full_vocab_top5_ids": top5.tolist(),
                      "oracle_full_vocab_top1_id": o["full_vocab_top1_id"][s] if isinstance(o["full_vocab_top1_id"], list) else o["full_vocab_top1_id"],
                      "max_abs_dp": float(d.max()), "mean_abs_dp": float(d.mean()),
                      "oracle_top2_margin": float(np.sort(po)[-1] - np.sort(po)[-2]) if n > 1 else None,
                      "finite": bool(np.isfinite(lg.astype(np.float32)).all())})
    flat = np.concatenate(deltas) if deltas else np.zeros(0)
    return {"slots_equal": slots_equal, "slots": slots,
            "n_slots_read": len(slots), "n_slots_oracle": len(o["slot_idx"]),
            "argmax_all_equal": all(s["argmax_equal"] for s in slots) and len(slots) == len(o["slot_idx"]),
            "max_abs_dp": float(flat.max()) if flat.size else None,
            "mean_abs_dp": float(flat.mean()) if flat.size else None}


def summarize(runs: list[dict], expected: list[tuple[str, str]]) -> dict:
    have = {(r["id"], r["arm"]) for r in runs}
    slots = [s for r in runs for s in r["slots"]]
    worst = max(runs, key=lambda r: r["max_abs_dp"] or 0.0) if runs else None
    return {
        "runs": len(runs), "expected_runs": len(expected),
        "missing_runs": [f"{a}/{b}" for a, b in expected if (a, b) not in have],
        "slots_expected": sum(r["n_slots_oracle"] for r in runs),
        "slots_equal": sum(r["n_slots_oracle"] for r in runs if r["slots_equal"]),
        "slots_read": len(slots),
        "argmax_equal": sum(s["argmax_equal"] for s in slots),
        "full_vocab_top1_is_oracle_letter": sum(s["full_vocab_top1_is_oracle_letter"] for s in slots),
        "max_abs_dp": max((s["max_abs_dp"] for s in slots), default=None),
        "mean_of_run_mean_abs_dp": float(np.mean([r["mean_abs_dp"] for r in runs])) if runs else None,
        "worst_run": None if worst is None else {"id": worst["id"], "arm": worst["arm"],
                                                 "max_abs_dp": worst["max_abs_dp"]},
        "finite_all": all(s["finite"] for s in slots),
        "min_oracle_top2_margin": min((s["oracle_top2_margin"] for s in slots
                                       if s["oracle_top2_margin"] is not None), default=None),
    }


def verdict(s: dict, resets_ok: bool) -> tuple[bool, dict]:
    checks = {
        "all_runs": s["runs"] == s["expected_runs"] and not s["missing_runs"],
        "slots": s["slots_equal"] == s["slots_expected"] == s["slots_read"],
        "argmax": s["argmax_equal"] == s["slots_expected"],
        "full_vocab_top1": s["full_vocab_top1_is_oracle_letter"] == s["slots_expected"],
        "max_abs_dp": s["max_abs_dp"] is not None and s["max_abs_dp"] <= BAR["max_abs_dp"],
        "mean_of_run_mean_abs_dp": (s["mean_of_run_mean_abs_dp"] is not None
                                    and s["mean_of_run_mean_abs_dp"] <= BAR["mean_of_run_mean_abs_dp"]),
        "finite": s["finite_all"],
        "reset_bit_equal_all_processes": resets_ok,
    }
    return all(checks.values()), checks


def env_record() -> dict:
    import importlib.metadata as md

    v = {"python": sys.version.split()[0], "numpy": np.__version__}
    for p in ("coreai-core", "coreai-torch", "coreai-models", "tokenizers", "pillow", "transformers"):
        try:
            v[p] = md.version(p)
        except Exception as e:  # noqa: BLE001
            v[p] = repr(e)
    osb = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    return {"versions": v, "platform": platform.platform(), "macos_build": osb,
            "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                   text=True).stdout.strip(),
            "runtime": "coreai python runtime, AOT h16c GPU .aimodelc, SpecializationOptions.default(), no JIT",
            "gpu": "shared with other sessions, no _GPU_LOCK (times are contended reference values)"}


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
        js = Path(sp["out"]).with_suffix(".json")
        if proc.returncode != 0 or not js.exists():
            got.append({"spec": sp, "failed": True, "returncode": proc.returncode, "wall_seconds": wall})
            print(f"[{tag}] {Path(sp['out']).name}: worker FAILED (exit {proc.returncode})", flush=True)
            continue
        rec = json.loads(js.read_text())
        rec["process_wall_seconds"] = wall
        rec["returncode"] = proc.returncode
        got.append(rec)
    return got


def split(runs: list, per: int = RUNS_PER_PROCESS) -> list[list]:
    n = math.ceil(len(runs) / per)
    size = math.ceil(len(runs) / n)
    return [runs[i:i + size] for i in range(0, len(runs), size)]


def collect(shard_recs: list[dict], orc: dict, raw: dict | None = None) -> tuple[list[dict], list[dict], dict]:
    """Merge the shards: scored runs, the process split record, and every call's ms.

    `steps[(id, arm, variant)]` = (ms array, first run of its process, call kinds or None): per
    token in S=1 order, per call in chunk order (kinds 1 = prefill, 0 = main). `raw`, when given,
    receives the full slot logits per (id, arm, variant)."""
    runs, procs, steps = [], [], {}
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
                      "load_split_seconds": sr.get("load_split_seconds"),
                      "function_names": sr.get("function_names"),
                      "tower_load_seconds": sr.get("tower_load_seconds"),
                      "process_wall_seconds": sr["process_wall_seconds"], "reset_check": sr["reset_check"],
                      "npz": str(prefix.with_suffix(".npz")), "npz_sha256": sha256_file(prefix.with_suffix(".npz"))})
        procs.append(entry)
        for rec in sr["runs"]:
            key = f"{rec['index']:02d}"
            o = orc[(rec["id"], rec["arm"])]
            sc = score_run(rec, z[f"{key}__slot_logits"], o)
            if raw is not None:
                raw[(rec["id"], rec["arm"], rec["variant"])] = z[f"{key}__slot_logits"]
            if f"{key}__call_ms" in z:
                sm, kinds = z[f"{key}__call_ms"], z[f"{key}__call_kind"]
            else:
                sm, kinds = z[f"{key}__step_ms"], None
            steps[(rec["id"], rec["arm"], rec["variant"])] = (sm, rec["index"] == 0, kinds)
            row = {k: rec[k] for k in ("id", "arm", "variant", "tokens", "grid", "start", "amount",
                                       "slots_host", "slots_oracle", "nopts", "decode_seconds", "wall_seconds")}
            for k in ("ids_equal_oracle", "tokenize_ms", "preprocess_ms", "tower_ms", "tower_vs_oracle",
                      "prefill_calls", "main_calls", "slot_read_from"):
                if k in rec:
                    row[k] = rec[k]
            row.update(sc)
            if kinds is None:
                row["step_ms_median"] = float(np.median(sm))
            else:
                row["prefill_call_ms_median"] = float(np.median(sm[kinds == 1])) if (kinds == 1).any() else None
                row["main_call_ms_median"] = float(np.median(sm[kinds == 0])) if (kinds == 0).any() else None
            row["shard"] = prefix.name
            row["npz_key"] = key
            row["first_in_process"] = rec["index"] == 0
            runs.append(row)
    return runs, procs, steps


def timing(runs: list[dict], procs: list[dict], steps: dict) -> dict:
    """Contended reference times. `warm` = every run but the first of its process (that one carries
    the process's first-call cost, reported separately)."""
    loads = [p["load_seconds"] for p in procs if "load_seconds" in p]

    def q(a: np.ndarray) -> dict:
        return {"n": int(a.size), "median": float(np.median(a)) if a.size else None,
                "p10": float(np.quantile(a, 0.1)) if a.size else None,
                "p90": float(np.quantile(a, 0.9)) if a.size else None}

    def cat(parts: list) -> np.ndarray:
        return np.concatenate(parts) if parts else np.zeros(0)

    t = {"contended": True,
         "load_seconds": {"first_process": loads[0] if loads else None,
                          "median": float(np.median(loads)) if loads else None, "all": loads}}
    splits = [p["load_split_seconds"] for p in procs if p.get("load_split_seconds")]
    if splits:
        t["load_split_seconds_median"] = {
            k: (float(np.median([x[k] for x in splits])) if all(x.get(k) is not None for x in splits) else None)
            for k in ("model", "main", "prefill")}
    if all(k is None for _, _, k in steps.values()):
        t["ms_per_step_warm_runs"] = q(cat([sm for sm, first, _ in steps.values() if not first]))
        t["ms_per_step_all_runs"] = q(cat([sm for sm, _, _ in steps.values()]))
    else:
        for kind, label in ((1, "prefill"), (0, "main")):
            t[f"ms_per_{label}_call_warm_runs"] = q(cat([sm[k == kind] for sm, first, k in steps.values()
                                                        if not first]))
            t[f"ms_per_{label}_call_all_runs"] = q(cat([sm[k == kind] for sm, _, k in steps.values()]))
        t["calls_per_run"] = {"prefill": int(sum(int((k == 1).sum()) for _, _, k in steps.values())),
                              "main": int(sum(int((k == 0).sum()) for _, _, k in steps.values())),
                              "tokens": int(sum(r["tokens"] for r in runs))}
    t["first_run_of_process_wall_seconds"] = [r["wall_seconds"] for r in runs if r.get("first_in_process")]
    t["prompt_wall_seconds_warm"] = {}
    for arm in sorted({r["arm"] for r in runs}):
        rs = [r for r in runs if r["arm"] == arm and not r.get("first_in_process")]
        if rs:
            t["prompt_wall_seconds_warm"][arm] = {
                "n": len(rs), "median": float(np.median([r["wall_seconds"] for r in rs])),
                "decode_median": float(np.median([r["decode_seconds"] for r in rs])),
                "median_tokens": float(np.median([r["tokens"] for r in rs]))}
    if any("tower_ms" in r for r in runs):
        t["tower_ms_warm"] = {}
        for arm in sorted({r["arm"] for r in runs if "tower_ms" in r}):
            v = [r["tower_ms"] for r in runs if r["arm"] == arm and "tower_ms" in r and not r.get("first_in_process")]
            f = [r["tower_ms"] for r in runs if r["arm"] == arm and "tower_ms" in r and r.get("first_in_process")]
            pre = [r["preprocess_ms"] for r in runs if r["arm"] == arm and "preprocess_ms" in r]
            t["tower_ms_warm"][arm] = {"n": len(v), "median": float(np.median(v)) if v else None,
                                       "min": min(v, default=None), "max": max(v, default=None),
                                       "first_call_of_process": f,
                                       "host_preprocess_ms_median": float(np.median(pre)) if pre else None}
        loads_t = [p["tower_load_seconds"] for p in procs if p.get("tower_load_seconds") is not None]
        t["tower_load_seconds"] = {"median": float(np.median(loads_t)), "all": loads_t}
    return t


def bundle_record(bundle: Path, aimodelc: Path) -> dict:
    meta = json.loads((bundle / "metadata.json").read_text())
    mlirb = bundle / meta["assets"]["main"] / "main.mlirb"
    return {"bundle": str(bundle), "name": meta["name"], "compression": meta.get("compression"),
            "metadata_sha256": sha256_file(bundle / "metadata.json"),
            "main_mlirb": {"bytes": mlirb.stat().st_size, "sha256": sha256_file(mlirb)},
            "tokenizer_sha256": {f.name: sha256_file(f) for f in sorted((bundle / "tokenizer").iterdir())},
            "aimodelc": tree_digest(aimodelc)}


def per_arm(runs: list[dict], expected: list[tuple[str, str]]) -> dict:
    out = {}
    for arm in sorted({a for _, a in expected}):
        out[arm] = summarize([r for r in runs if r["arm"] == arm], [e for e in expected if e[1] == arm])
    return out


READOUT = {
    "s1": "S=1 from fresh zero states through the whole row, logits at every slot step, letters "
          "ids 32.. over the first nopts, softmax T=1 (float64)",
    "chunk": "chunk order from fresh zero states (prefill S={C} while a whole chunk fits before the next slot, "
             "then S=1 up to and including it; the slot's logits from the call that processed it last), "
             "letters ids 32.. over the first nopts, softmax T=1 (float64)",
}


def order_of(args, meta: dict) -> tuple[str, int | None]:
    order = args.order
    chunk = meta["language"].get("prefill_chunk")
    if order == "chunk":
        fm = meta["language"].get("function_map", {}).get("main", [])
        if not chunk or "prefill" not in fm:
            raise SystemExit(f"--order chunk needs a bundle with a prefill function (function_map {fm}, "
                             f"prefill_chunk {chunk})")
        return order, int(chunk)
    return order, None


def b1_pass(aimodelc: Path, meta: dict, orc: dict, expected: list, order: str, chunk: int | None,
            work: Path, tag: str) -> dict:
    """One full b1 pass (111 runs) in one order: shards, merge, summary, verdict, timing."""
    runs_all = [[i, a, "base"] for i, a in expected]
    shards = [{"gate": "b1", "aimodelc": str(aimodelc), "max_ctx": meta["language"]["max_context_length"],
               "order": order, "chunk": chunk, "runs": part, "out": str(work / f"shard_{k:02d}")}
              for k, part in enumerate(split(runs_all))]
    t0 = time.monotonic()
    recs = run_shards(tag, shards)
    raw: dict = {}
    runs, procs, steps = collect(recs, orc, raw)
    runs = [r for r in runs if r["variant"] == "base"]
    s = summarize(runs, expected)
    resets = all(p.get("reset_check", {}).get("bit_equal", False) for p in procs)
    ok, checks = verdict(s, resets)
    if order == "chunk":
        s["slots_read_from_prefill"] = sum(r.get("slot_read_from", []).count("prefill") for r in runs)
        s["rows_without_a_prefill_call"] = [f"{r['id']}/{r['arm']}" for r in runs if not r.get("prefill_calls")]
        s["min_tokens"] = min(r["tokens"] for r in runs)
    return {"order": order, "chunk": chunk, "processes": procs, "summary": s, "per_arm": per_arm(runs, expected),
            "checks": checks, "result": "PASS" if ok else "FAIL", "ok": ok,
            "timing": timing(runs, procs, steps), "wall_seconds_total": time.monotonic() - t0,
            "runs": runs, "raw": raw}


def chunk_vs_s1(chunk_pass: dict, s1_pass: dict) -> dict:
    """Same bundle, two orders: per-slot letter probabilities, argmax and full-vocab top-1."""
    s1_runs = {(r["id"], r["arm"]): r for r in s1_pass["runs"]}
    rows, per_run_mean = [], []
    for r in chunk_pass["runs"]:
        o = s1_runs.get((r["id"], r["arm"]))
        if o is None:
            continue
        lc = chunk_pass["raw"][(r["id"], r["arm"], "base")].astype(np.float32)
        ls = s1_pass["raw"][(r["id"], r["arm"], "base")].astype(np.float32)
        deltas = []
        for k, (a, b) in enumerate(zip(r["slots"], o["slots"])):
            pa, pb = np.asarray(a["probs"]), np.asarray(b["probs"])
            d = np.abs(pa - pb)
            deltas.append(d)
            n = a["nopts"]
            rows.append({"id": r["id"], "arm": r["arm"], "t": a["t"], "nopts": n,
                         "read_from": (r.get("slot_read_from") or ["?"] * len(r["slots"]))[k],
                         "max_abs_dp": float(d.max()), "argmax_chunk": a["argmax"], "argmax_s1": b["argmax"],
                         "argmax_equal": a["argmax"] == b["argmax"],
                         "full_vocab_top1_equal": a["full_vocab_top1_id"] == b["full_vocab_top1_id"],
                         "letter_logits_max_abs_diff": float(np.max(np.abs(
                             lc[k, LETTER0:LETTER0 + n] - ls[k, LETTER0:LETTER0 + n]))),
                         "full_logits_max_abs_diff": float(np.max(np.abs(lc[k] - ls[k]))),
                         "full_logits_bit_equal": bool(np.array_equal(lc[k], ls[k])),
                         "p_argmax_chunk": float(pa[a["argmax"]]), "p_argmax_s1": float(pb[b["argmax"]]),
                         "oracle_top2_margin": a["oracle_top2_margin"]})
        per_run_mean.append(float(np.concatenate(deltas).mean()))

    def agg(sel: list[dict]) -> dict:
        if not sel:
            return {"slots": 0}
        worst = max(sel, key=lambda x: x["max_abs_dp"])
        return {"slots": len(sel), "argmax_equal": sum(x["argmax_equal"] for x in sel),
                "full_vocab_top1_equal": sum(x["full_vocab_top1_equal"] for x in sel),
                "full_logits_bit_equal": sum(x["full_logits_bit_equal"] for x in sel),
                "max_abs_dp": worst["max_abs_dp"],
                "median_slot_max_abs_dp": float(np.median([x["max_abs_dp"] for x in sel])),
                "max_letter_logit_diff": max(x["letter_logits_max_abs_diff"] for x in sel),
                "worst": {k: worst[k] for k in ("id", "arm", "t", "read_from", "max_abs_dp", "p_argmax_chunk",
                                                "p_argmax_s1", "oracle_top2_margin")}}

    out = {"definition": "per slot: |p_chunk - p_s1| over the question's letters (same bundle, same fixture "
                         "inputs); mean_of_run_mean_abs_dp as in the bar",
           "all": agg(rows), "mean_of_run_mean_abs_dp": float(np.mean(per_run_mean)) if per_run_mean else None,
           "by_read_from": {k: agg([x for x in rows if x["read_from"] == k]) for k in ("prefill", "main")},
           "by_arm": {a: agg([x for x in rows if x["arm"] == a]) for a in sorted({x["arm"] for x in rows})},
           "rows_without_a_prefill_call": chunk_pass["summary"].get("rows_without_a_prefill_call"),
           "min_tokens": chunk_pass["summary"].get("min_tokens"),
           "stop_condition": "any slot |dp| > 0.02 or argmax differs",
           "slots": rows}
    out["stop_condition_hit"] = bool(out["all"].get("max_abs_dp", 0) > 0.02
                                     or out["all"].get("argmax_equal") != out["all"].get("slots"))
    return out


def public(pass_rec: dict) -> dict:
    return {k: v for k, v in pass_rec.items() if k not in ("raw", "ok")}


def gate_b1(args) -> int:
    bundle = Path(args.bundle).expanduser().resolve()
    meta = json.loads((bundle / "metadata.json").read_text())
    aimodelc = aimodelc_for(bundle)
    order, chunk = order_of(args, meta)
    fx, rows = load_oracle()
    orc = {(r["id"], r["arm"]): r for r in rows}
    expected = [(r["id"], r["arm"]) for r in rows]
    work_root = Path(args.work_dir).expanduser() if args.work_dir else WORK
    sub = meta["name"] + ("" if order == "s1" else f"__{order}{chunk}")
    work = work_root / "b1" / sub
    t0 = time.monotonic()
    main_pass = b1_pass(aimodelc, meta, orc, expected, order, chunk, work, f"b1 {sub}")
    s, checks, ok = main_pass["summary"], main_pass["checks"], main_pass["ok"]
    resets = checks["reset_bit_equal_all_processes"]
    record = {
        "schema": "coreai-decider-vision-readout-gate/1",
        "gate": "b1 decoder alone (oracle ids, oracle fp32 image_embeds -> fp16, zero-padded to 256 rows)",
        "bundle": bundle_record(bundle, aimodelc),
        "oracle": {"path": str(ORACLE / "fixture_oracle.json"),
                   "sha256": sha256_file(ORACLE / "fixture_oracle.json"),
                   "source": fx["source"]["hf_id"] + "@" + fx["source"]["revision"], "oracle": fx["oracle"],
                   "npz_sha256": {f"{i}__{a}": sha256_file(ORACLE / "npz" / f"{i}__{a}.npz")
                                  for i, a in expected if a != "text"}},
        "order": order, "chunk": chunk,
        "readout": READOUT[order].format(C=chunk),
        "bar": {**BAR, "slots": "all equal the oracle", "argmax": "all slots", "full_vocab_top1": "the oracle's "
                "argmax letter on all slots", "reset": "bit-equal slot logits in every process",
                "mean_of_run_mean_abs_dp_definition": "mean over runs of the mean |dp| over all (slot, option) "
                                                      "entries of the run"},
        "environment": env_record(),
        "processes": main_pass["processes"],
        "summary": s, "per_arm": main_pass["per_arm"],
        "checks": checks, "result": main_pass["result"],
        "timing": main_pass["timing"],
        "runs": main_pass["runs"],
    }
    if args.note:
        record["note"] = args.note
    if args.also_s1 and order != "s1":
        s1 = b1_pass(aimodelc, meta, orc, expected, "s1", None, work_root / "b1" / meta["name"],
                     f"b1 {meta['name']} (S=1 order)")
        record["s1_order"] = {**public(s1), "readout": READOUT["s1"]}
        record["chunk_vs_s1"] = chunk_vs_s1(main_pass, s1)
    if args.red:
        red_spec = {"gate": "b1", "aimodelc": str(aimodelc), "max_ctx": meta["language"]["max_context_length"],
                    "order": order, "chunk": chunk,
                    "runs": [["r14", "g256", "base"], ["r14", "g256", "embeds_zero"]],
                    "out": str(work / "red_r14_g256_embeds_zero")}
        rr = run_shards(f"b1 red {sub}", [red_spec])
        rruns, rprocs, _ = collect(rr, orc)
        base = next((r for r in rruns if r["variant"] == "base"), None)
        zero = next((r for r in rruns if r["variant"] == "embeds_zero"), None)
        red = {"run": "r14/g256", "arm": "image_embeds zeroed (image ids and rope unchanged)", "order": order,
               "process": rprocs}
        if base and zero:
            pb, pz = base["slots"][0]["probs"], zero["slots"][0]["probs"]
            red.update({"base_probs": pb, "zeroed_probs": pz, "oracle_probs": base["slots"][0]["probs_oracle"],
                        "base_argmax": base["slots"][0]["argmax"], "zeroed_argmax": zero["slots"][0]["argmax"],
                        "argmax_changed": zero["slots"][0]["argmax"] != base["slots"][0]["argmax"],
                        "zeroed_argmax_equals_oracle": zero["slots"][0]["argmax_equal"],
                        "max_abs_dp_vs_base": float(np.max(np.abs(np.asarray(pz) - np.asarray(pb)))),
                        "base_equals_gate_run": base["slots"][0]["letter_logits"] == next(
                            r for r in main_pass["runs"] if (r["id"], r["arm"]) == ("r14", "g256"))["slots"][0]["letter_logits"],
                        "torch_round3_zeroed_probs": [0.17376171052455902, 0.1938318908214569,
                                                      0.1262502372264862, 0.5061562061309814]})
            red["result"] = "RED (argmax moved)" if red["argmax_changed"] else "NOT RED"
        record["red_arm"] = red
    record["wall_seconds_total"] = time.monotonic() - t0
    record["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    Path(args.transcript).parent.mkdir(parents=True, exist_ok=True)
    Path(args.transcript).write_text(json.dumps(record, indent=1) + "\n")
    print(f"{record['result']}: {meta['name']} ({order}{chunk or ''}) runs {s['runs']}/{s['expected_runs']} slots "
          f"{s['slots_equal']}/{s['slots_expected']} argmax {s['argmax_equal']} full-vocab "
          f"{s['full_vocab_top1_is_oracle_letter']} max|dp| {s['max_abs_dp']:.6f} mean "
          f"{s['mean_of_run_mean_abs_dp']:.6f} reset {resets} worst {s['worst_run']}")
    for k, v in checks.items():
        print(f"  {k}: {'ok' if v else 'FAIL'}")
    if "s1_order" in record:
        s1s = record["s1_order"]["summary"]
        print(f"S=1 order on the same bundle: {record['s1_order']['result']} max|dp| {s1s['max_abs_dp']:.6f} "
              f"mean {s1s['mean_of_run_mean_abs_dp']:.6f} worst {s1s['worst_run']}")
        cv = record["chunk_vs_s1"]
        print(f"chunk vs S=1: {json.dumps(cv['all'])} mean-of-run-means {cv['mean_of_run_mean_abs_dp']:.6f} "
              f"stop condition hit: {cv['stop_condition_hit']}")
    if "red_arm" in record:
        print(f"red arm: {record['red_arm'].get('result')} {record['red_arm'].get('zeroed_probs')}")
    print(f"transcript: {args.transcript}")
    return 0 if ok else 1


def gate_b2(args) -> int:
    fx, rows = load_oracle()
    orc = {(r["id"], r["arm"]): r for r in rows}
    expected = [(r["id"], r["arm"]) for r in rows if r["arm"] in GRIDS]
    exports = LANE / "exports"
    arms_cfg = []
    for b in args.bundles.split(","):
        arms_cfg.append({"bundle": Path(b).expanduser().resolve(), "tower": "fp16w32", "verdict": True})
    if args.fp16_tower_bundle:
        arms_cfg.append({"bundle": Path(args.fp16_tower_bundle).expanduser().resolve(), "tower": "fp16",
                         "verdict": False})
    t0 = time.monotonic()
    work_root = Path(args.work_dir).expanduser() if args.work_dir else WORK
    prior = None
    if args.append and Path(args.transcript).exists():
        prior = json.loads(Path(args.transcript).read_text())
    record: dict = {
        "schema": "coreai-decider-vision-e2e-gate/1",
        "gate": "b2 end to end: fixture image -> host.preprocess (Pillow BICUBIC to the tile) -> tower .aimodelc -> "
                "image_embeds (fp32 -> fp16) -> decoder; ids from host.build_ids with the bundle's tokenizer",
        "oracle": {"path": str(ORACLE / "fixture_oracle.json"), "sha256": sha256_file(ORACLE / "fixture_oracle.json"),
                   "source": fx["source"]["hf_id"] + "@" + fx["source"]["revision"],
                   "rows_json_sha256": sha256_file(LANE / "fixtures" / "rows.json"),
                   "meta_json_sha256": sha256_file(LANE / "fixtures" / "meta.json")},
        "bar": {**BAR, "applies_to": "fp16w32-tower arms, per bundle; the fp16-tower arm is values only"},
        "environment": env_record(), "arms": {}, "towers": {},
    }
    for tv in sorted({c["tower"] for c in arms_cfg}):
        for arm in GRIDS:
            d = exports / TOWERS[tv].format(arm=arm)
            c = sorted(d.glob("*.h16c.aimodelc"))
            assert len(c) == 1, (d, c)
            record["towers"][f"{tv}/{arm}"] = tree_digest(c[0])
            if prior is not None and f"{tv}/{arm}" in prior.get("towers", {}):
                if prior["towers"][f"{tv}/{arm}"]["tree_sha256"] != record["towers"][f"{tv}/{arm}"]["tree_sha256"]:
                    raise SystemExit(f"--append: tower {tv}/{arm} differs from the transcript's")
    ok_all = True
    for cfg in arms_cfg:
        bundle = cfg["bundle"]
        meta = json.loads((bundle / "metadata.json").read_text())
        aimodelc = aimodelc_for(bundle)
        order, chunk = order_of(args, meta)
        label = f"{meta['name']}+tower_{cfg['tower']}" + ("" if order == "s1" else f"+{order}{chunk}")
        if prior is not None and label in prior.get("arms", {}):
            raise SystemExit(f"--append: arm {label} is already in {args.transcript} (arms are never replaced)")
        work = work_root / "b2" / label
        shards = []
        for arm in GRIDS:
            tower = sorted((exports / TOWERS[cfg["tower"]].format(arm=arm)).glob("*.h16c.aimodelc"))[0]
            runs = [[i, a, "e2e"] for i, a in expected if a == arm]
            for k, part in enumerate(split(runs)):
                shards.append({"gate": "b2", "aimodelc": str(aimodelc), "order": order, "chunk": chunk,
                               "max_ctx": meta["language"]["max_context_length"],
                               "tokenizer": str(bundle / "tokenizer" / "tokenizer.json"),
                               "tower": {"aimodelc": str(tower), "variant": cfg["tower"], "grid": GRIDS[arm]},
                               "runs": part, "out": str(work / f"{arm}_shard_{k:02d}")})
        recs = run_shards(f"b2 {label}", shards)
        runs, procs, steps = collect(recs, orc)
        s = summarize(runs, expected)
        s["ids_equal_oracle"] = sum(bool(r.get("ids_equal_oracle")) for r in runs)
        s["tower_vs_oracle_min_row"] = min((r["tower_vs_oracle"]["min_row"] for r in runs), default=None)
        s["tower_vs_oracle_min_cos"] = min((r["tower_vs_oracle"]["cos"] for r in runs), default=None)
        resets = all(p.get("reset_check", {}).get("bit_equal", False) for p in procs)
        if order == "chunk":
            s["slots_read_from_prefill"] = sum(r.get("slot_read_from", []).count("prefill") for r in runs)
            s["rows_without_a_prefill_call"] = [f"{r['id']}/{r['arm']}" for r in runs if not r.get("prefill_calls")]
        entry = {"bundle": bundle_record(bundle, aimodelc), "tower": cfg["tower"], "order": order, "chunk": chunk,
                 "readout": READOUT[order].format(C=chunk), "processes": procs,
                 "summary": s, "per_grid": per_arm(runs, expected), "timing": timing(runs, procs, steps),
                 "runs": runs}
        if args.note:
            entry["note"] = args.note
        if cfg["verdict"]:
            ok, checks = verdict(s, resets)
            ok = ok and s["ids_equal_oracle"] == s["runs"]
            checks["ids_equal_oracle"] = s["ids_equal_oracle"] == s["runs"]
            entry.update({"checks": checks, "result": "PASS" if ok else "FAIL"})
            ok_all &= ok
        else:
            entry["result"] = "values only (no verdict)"
            entry["reset_bit_equal_all_processes"] = resets
        record["arms"][label] = entry
        print(f"{entry['result']}: {label} runs {s['runs']}/{s['expected_runs']} slots {s['slots_equal']}/"
              f"{s['slots_expected']} argmax {s['argmax_equal']} full-vocab {s['full_vocab_top1_is_oracle_letter']} "
              f"max|dp| {s['max_abs_dp']:.6f} mean {s['mean_of_run_mean_abs_dp']:.6f} reset {resets} "
              f"ids {s['ids_equal_oracle']} tower min-row {s['tower_vs_oracle_min_row']:.6f} worst {s['worst_run']}",
              flush=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if prior is not None:
        # Keep every existing key and arm as it is; add the new arms, their towers, and a log line.
        for label, entry in record["arms"].items():
            entry["environment"] = record["environment"]
            entry["generated_at"] = now
            prior["arms"][label] = entry
        for k, v in record["towers"].items():
            prior.setdefault("towers", {}).setdefault(k, v)
        prior.setdefault("appended", []).append({"arms": list(record["arms"]), "generated_at": now,
                                                 "wall_seconds": time.monotonic() - t0})
        record = prior
    else:
        record["wall_seconds_total"] = time.monotonic() - t0
        record["generated_at"] = now
    Path(args.transcript).parent.mkdir(parents=True, exist_ok=True)
    Path(args.transcript).write_text(json.dumps(record, indent=1) + "\n")
    print(f"transcript: {args.transcript}")
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("b1", help="decoder alone, all 111 runs")
    a1.add_argument("bundle")
    a1.add_argument("--transcript", required=True)
    a1.add_argument("--red", action="store_true", help="add the r14 g256 image_embeds-zeroed red arm")
    a1.add_argument("--also-s1", action="store_true",
                    help="with --order chunk: run the same bundle in S=1 order too, add the chunk vs S=1 table")
    a2 = sub.add_parser("b2", help="end to end, the 70 g256 / g448 runs")
    a2.add_argument("--bundles", required=True, help="comma list of bundle dirs (fp16w32 tower, gated)")
    a2.add_argument("--fp16-tower-bundle", help="bundle dir run once more behind the fp16 tower (values only)")
    a2.add_argument("--transcript", required=True)
    a2.add_argument("--append", action="store_true",
                    help="add the arms to an existing transcript (its arms are kept; an existing label is refused)")
    for a in (a1, a2):
        a.add_argument("--order", default="s1", choices=["s1", "chunk"],
                       help="s1 = 'main' through the whole row; chunk = 'prefill' chunks + S=1 remainder "
                            "(the bundle's language.prefill_chunk)")
        a.add_argument("--work-dir", help=f"shard files go to <work-dir>/<gate>/... (default {WORK})")
        a.add_argument("--note", help="free text kept in the transcript (b1: top level, b2: in the new arm), "
                                      "e.g. that the fixture includes rows used to choose the bundle")
    aw = sub.add_parser("worker")
    aw.add_argument("--spec", required=True)
    args = ap.parse_args()
    if args.cmd == "worker":
        return worker(Path(args.spec))
    return gate_b1(args) if args.cmd == "b1" else gate_b2(args)


if __name__ == "__main__":
    raise SystemExit(main())
