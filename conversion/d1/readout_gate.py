#!/usr/bin/env python3
"""Readout gate: the d1 decoder bundle on the Mac GPU, read out through host.py, against the oracle.

Every oracle question is one row (`row_ids`, positions 0..T-1), fed to the bundle's one static-S function `main` (the
AOT h16c `.aimodelc`, `SpecializationOptions.default()`, never the JIT) from fresh zero states in S-token calls: call c
gets ids[cS : cS + S] with position_ids 0..cS+S-1 and image_embeds zero, the last call padded with <|pad|> (124893)
and the padded positions' rows discarded (`parity_decoder_torch.py`'s order). Everything the gate needs comes from the
bundle: S, max_context_length and the input / output / state contract (metadata.json `language`), the option rows
(`head/option_rows.safetensors`), the tokenizer for the red arms (`tokenizer/`). The slot's hidden row (the row's last
token, fp16) goes through the host's readout — z[id] = h_slot . E[id] in float64 over the question's group ids, group
max, softmax over the options — and is compared with the oracle's probabilities.

Bar, fixed before any result (the transcript is opened with it before the first GPU process):
  (a) argmax = the oracle's on every question whose oracle top-2 margin is above 0.02 (near-ties listed apart)
  (b) max |dp| <= 0.02 over every option of every question, near-ties included
  (c) the mean over runs of the run's mean |dp| over its options <= 0.002 (a run = one row = one question)
  (d) every process re-runs its first row at the end and reproduces its hidden rows bit for bit (the state reset)
  (e) every hidden value of every row finite (and no row all zero)
plus every expected row present.

Red arms (`--red`, their own process): the five arms of `--arms` (default `fixtures/red_arms_r4.json`, round 4's set:
a word, two grammatical "not"s, two state swaps; `fixtures/red_arms.json` is round 1's set, kept as its record, and the
default for a toy bundle, whose arms oracle is built from it) rendered by the host from their requests with the
bundle's tokenizer (`parity_decoder_torch.red_rows`, which also checks that the host reproduces the base records' rows),
each perturbed row against its base row (the same question of the unperturbed record) run in the same process. The
transcript names the arms file and its sha256. An arm is red when the perturbed rows fail the gate's own
bar against the base rows: an argmax moves on a non-near-tie question, or max |dp| > 0.02, or the mean of the rows'
mean |dp| > 0.002. The base rows must equal the gate's runs of the same rows bit for bit (sha256 of the hidden rows).
The oracle comes first (`--red-oracle`, the arms' oracle: oracle_d1.py run on the arms' fixture records, or for a toy
bundle the toy module's own forward, built when absent; by default the one `<work>/oracle/*/records_oracle.json` whose
fixture file names the arms file's sha256): the same rule on the oracle's probabilities (perturbed against base)
says which arms move the model at all; an arm that is not red on the oracle is listed under `replace` (swap it for one
that moves) and is not asked of the graph. An arms oracle run on another arms file is refused. The graph must be red
on every arm that is red on the oracle, and on every
row the graph's dp (perturbed - base) must equal the oracle's within the gate's 0.02. Without an arms oracle the
pre-check is reported as not run and every arm is asked of the graph.

Process split: the Python runtime leaks an IOSurface per call, so a process takes at most 40 rows + the reset re-run;
the driver runs the processes one after another. Each worker writes `<work>/shard_NN.json` (per row: T, calls, the
hidden rows' sha256, finite, per-call ms) and `.npz` (the slot's fp16 hidden row and the per-call ms; the full hidden
rows of the records the oracle kept hidden rows for, for the position cosine). The GPU is shared with other sessions:
the ms are contended reference values for the transcript only (timing.py measures in a quiet window).

`--toy-oracle <json>` (a toy bundle from `export_decoder.py --toy`): the oracle is the toy module's own fp32 forward
(`parity_decoder_torch.build_toy_oracle`: each fixture row's ids folded into the toy vocabulary, random readout groups
of the real groups' shape from the toy table, probabilities by the host's arithmetic), built from the bundle's
`metadata.json` `toy.seed` and `head/` when the file does not exist. Rows, pad and red-arm ids are folded the same way.

    cd conversion/d1
    PY=<coreai-models venv>/bin/python Q="$HOME/code/standup/tools/quiet/quiet_wait.py --max-wait 3600 --"
    $Q $PY readout_gate.py run $ZOO_WORK_ROOT/_d1_3b/exports/toy_bundles/d1_toy_decode_fp16_pf16 \\
        --toy-oracle $ZOO_WORK_ROOT/_d1_3b/oracle_toy/records_oracle.json --red --transcript <json>
    $Q $PY readout_gate.py run $ZOO_WORK_ROOT/_d1_3b/exports/bundles/d1_3b_decode_fp16_pf16 --red --transcript <json>
    $Q $PY readout_gate.py run .../d1_3b_decode_int8lin_pf16 --red --compare-with <fp16 transcript> --transcript <json>
    $Q $PY readout_gate.py red <bundle> --gate-transcript <json> --transcript <json>     # the arms alone
    #   --arms $ZOO_WORK_ROOT/_d1_3b/fixtures/red_arms.json: round 1's set (its oracle: oracle/red/)
    $PY readout_gate.py red-records          # round 3: fixtures/red_arms_records.json, the arms for oracle_d1.py:
    #   oracle_d1.py --fixtures $K/fixtures/red_arms_records.json --out-dir $K/oracle/red --results-dir $K/oracle/red
    $Q $PY readout_gate.py merge <t1> <t2> .. --transcript <json>                    # disjoint row sets, one asset
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
import host  # noqa: E402
from _paths import gpu_lock, work_path  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")

LANE = work_path("_d1_3b")
ORACLE = LANE / "oracle" / "records_oracle.json"
RED_ARMS = LANE / "fixtures" / "red_arms_r4.json"         # round 4's set: the default
RED_ARMS_ROUND1 = LANE / "fixtures" / "red_arms.json"     # round 1's set: its record, and a toy bundle's default
BAR = {"max_abs_dp": 0.02, "mean_of_run_mean_abs_dp": 0.002, "near_tie_top2_margin": 0.02,
       "argmax": "every question with an oracle top-2 margin above 0.02; near-ties listed apart",
       "max_abs_dp_applies_to": "every option of every question, near-ties included",
       "mean_of_run_mean_abs_dp_definition": "mean over runs (one run = one row = one question) of the mean |dp| "
                                             "over that question's options",
       "reset": "every process re-runs its first row last: hidden rows bit-equal",
       "finite": "every hidden value of every row finite, no row all zero", "rows": "every expected row present"}
RUNS_PER_PROCESS = 40                               # + the reset re-run
OTHER_GPU = re.compile(r"yardstick|litert|llm-bench|coreai_verify|coreai-build|readout_gate|gate_|parity_|mlx|"
                       r"decide\.py|timing\.py|--accel gpu")
COS_THRESHOLDS = (0.99, 0.9, 0.5)
LOWEST_POSITIONS = 6
QUICK_PER_SOURCE = 1


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


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- the bundle
class Bundle:
    """What the gate reads from a bundle directory: metadata, S, the contract, the option rows, the AOT asset."""

    def __init__(self, path: str, aimodelc: str | None = None):
        self.dir = Path(path).expanduser().resolve()
        self.meta = json.loads((self.dir / "metadata.json").read_text())
        lang = self.meta["language"]
        self.name = self.meta["name"]
        self.S = int(lang["prefill_chunk"])
        self.max_ctx = int(lang["max_context_length"])
        self.contract = lang["contract"]
        self.hidden = int(self.contract["outputs"]["hidden"][0][2])
        self.image_embeds = self.contract["inputs"]["image_embeds"]
        self.toy = self.meta.get("toy")
        self.vocab = int(lang["vocab_size"])
        self.pad_id = int(self.toy["pad_folded"]) if self.toy else host.PAD_ID
        self.aimodelc = (Path(aimodelc).expanduser().resolve() if aimodelc else
                         self.dir.parent.parent / f"{self.dir.parent.name}_aotc" / f"{self.name}.h16c.aimodelc")

    def table(self) -> dict[int, np.ndarray]:
        import export_option_rows as eor
        return eor.read_table(self.dir)

    def record(self) -> dict:
        mlirb = self.dir / self.meta["assets"]["main"] / "main.mlirb"
        aot = tree_digest(self.aimodelc)
        mh = self.aimodelc / "main.hash"
        return {"bundle": str(self.dir), "name": self.name, "kind": self.meta["kind"],
                "compression": self.meta.get("compression"), "prefill_chunk": self.S, "max_ctx": self.max_ctx,
                "toy": self.toy, "metadata_sha256": sha256_file(self.dir / "metadata.json"),
                "main_mlirb": {"bytes": mlirb.stat().st_size, "sha256": sha256_file(mlirb)},
                "tokenizer_sha256": {f.name: sha256_file(f) for f in sorted((self.dir / "tokenizer").iterdir())},
                "head_sha256": {f.name: sha256_file(f) for f in sorted((self.dir / "head").iterdir())},
                "license_sha256": sha256_file(self.dir / "LICENSE") if (self.dir / "LICENSE").exists() else None,
                "aimodelc": {k: aot[k] for k in ("path", "bytes", "tree_sha256")},
                "aimodelc_main_hash": mh.read_bytes().hex() if mh.exists() else None}


def runtime_cache_entry(main_hash: str | None) -> dict | None:
    """The Python runtime's cache entry for an AOT asset (named by the asset's main.hash) and its size."""
    if not main_hash:
        return None
    build = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    p = Path.home() / "Library/Caches/coreai-cache" / build / "python" / main_hash
    if not p.exists():
        return {"path": str(p), "exists": False}
    du = subprocess.run(["du", "-sk", str(p)], capture_output=True, text=True).stdout.split()
    return {"path": str(p), "exists": True, "kib": int(du[0]) if du else None}


# --------------------------------------------------------------------------- worker: one process, <= 40 rows + the reset re-run
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


def check_contract(desc: dict, contract: dict) -> list[str]:
    bad = []
    for part in ("inputs", "outputs", "states"):
        if set(desc[part]) != set(contract[part]):
            bad.append(f"{part} names {sorted(desc[part])} != {sorted(contract[part])}")
            continue
        for n, w in contract[part].items():
            if [list(desc[part][n][0]), desc[part][n][1]] != [list(w[0]), w[1]]:
                bad.append(f"{part} {n}: {desc[part][n]} != {w}")
    return bad


def worker(spec_path: Path) -> int:
    import coreai.runtime as rt

    spec = json.loads(spec_path.read_text())
    S, max_ctx, H, pad = int(spec["S"]), int(spec["max_ctx"]), int(spec["hidden"]), int(spec["pad_id"])
    keep_full = set(spec.get("keep_full", []))

    def nd(a):
        return rt.NDArray(np.ascontiguousarray(a))

    out: dict = {"spec": {k: v for k, v in spec.items() if k != "runs"}, "pid": os.getpid(), "runs": [],
                 "started": time.time()}
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
        bad = check_contract(desc, spec["contract"])
        out["contract_mismatch"] = bad
        if bad:
            raise SystemExit(f"descriptor differs from the bundle's contract: {bad}")
        img_shape, img_dt = spec["contract"]["inputs"]["image_embeds"]
        img = nd(np.zeros(img_shape, np.dtype(img_dt)))

        def fresh_state() -> dict:
            return {n: nd(np.zeros([max_ctx if s < 0 else s for s in shape], np.dtype(dt)))
                    for n, (shape, dt) in desc["states"].items()}

        async def one(run: dict) -> tuple[dict, np.ndarray, np.ndarray]:
            ids = np.asarray(run["ids"], np.int32)
            T = len(ids)
            n = -(-T // S)
            if n * S > max_ctx - 1:
                raise SystemExit(f"{run['id']}:{run['k']}: {n * S} padded positions > {max_ctx - 1}")
            x = np.full(n * S, pad, np.int32)
            x[:T] = ids
            t_run = time.perf_counter()
            state = fresh_state()
            hid = np.zeros((n * S, H), np.float16)
            ms = np.zeros(n, np.float64)
            for c in range(n):
                t1 = time.perf_counter()
                res = await maybe(fn(inputs={"input_ids": nd(x[c * S:(c + 1) * S].reshape(1, S)),
                                             "position_ids": nd(np.arange((c + 1) * S, dtype=np.int32)[None]),
                                             "image_embeds": img}, state=state))
                h = np.asarray(res["hidden"].numpy())
                if h.shape != (1, S, H) or h.dtype != np.float16:
                    raise SystemExit(f"{run['id']}:{run['k']}: output {h.shape} {h.dtype} != (1, {S}, {H}) float16")
                hid[c * S:(c + 1) * S] = h[0]
                ms[c] = (time.perf_counter() - t1) * 1e3
            hid = hid[:T].copy()
            rec = {"id": run["id"], "k": run["k"], "variant": run["variant"], "tokens": T, "calls": n,
                   "padded_tokens": n * S, "wall_seconds": time.perf_counter() - t_run,
                   "hidden_sha256": hashlib.sha256(hid.tobytes()).hexdigest(),
                   "finite": bool(np.isfinite(hid.astype(np.float32)).all()), "all_zero": bool(not np.any(hid))}
            return rec, hid, ms

        runs = spec["runs"]
        first = None
        for i, run in enumerate(runs + [runs[0]]):
            rec, hid, ms = await one(run)
            if i < len(runs):
                key = f"{i:02d}"
                store[f"{key}__slot"] = hid[run["slot"]].copy()
                store[f"{key}__call_ms"] = ms
                if i in keep_full:
                    store[f"{key}__hidden"] = hid
                rec["index"] = i
                out["runs"].append(rec)
                if i == 0:
                    first = hid
                print(f"  [{os.getpid()}] {run['id']}:{run['k']}:{run['variant']}: {rec['tokens']} tok, {rec['calls']} "
                      f"calls, {rec['wall_seconds']:.3f} s (median {np.median(ms):.2f} ms/call, first {ms[0]:.1f})",
                      flush=True)
            else:
                same = bool(np.array_equal(first, hid))
                diff = float(np.max(np.abs(first.astype(np.float32) - hid.astype(np.float32))))
                out["reset_check"] = {"run": f"{run['id']}:{run['k']}:{run['variant']}", "bit_equal": same,
                                      "hidden_max_abs_diff": diff, "wall_seconds": rec["wall_seconds"]}
                print(f"  [{os.getpid()}] reset re-run {run['id']}:{run['k']}: bit-equal {same} (max|d| {diff})", flush=True)

    asyncio.run(go())
    out["finished"] = time.time()
    prefix = Path(spec["out"])
    np.savez(prefix.with_suffix(".npz"), **store)
    prefix.with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


# --------------------------------------------------------------------------- driver
def other_gpu_processes() -> list[str]:
    ps = subprocess.run(["ps", "-axo", "pid=,etime=,command="], capture_output=True, text=True).stdout
    me = os.getpid()
    skip = ("/.local/bin/claude", "claude --", "shell-snapshots", "until grep", "zsh -c", "/bin/zsh", "/bin/bash",
            "quiet_wait.py")
    return [ln.strip()[:200] for ln in ps.splitlines()
            if OTHER_GPU.search(ln) and not ln.strip().startswith(f"{me} ") and not any(s in ln for s in skip)
            and "readout_gate.py worker" not in ln]


def gpu_lock_state() -> dict:
    """The measurement lock, read only (this gate never takes it)."""
    p = gpu_lock()
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {"path": str(p), "exists": True, "bytes": st.st_size, "content": p.read_text()[:300],
            "mtime": datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds")}


def load_line() -> dict:
    la = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout.strip()
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
    return {"at": now(), "loadavg": la, "swapusage": sw}


def env_record() -> dict:
    import importlib.metadata as md

    v = {"python": sys.version.split()[0], "numpy": np.__version__}
    for p in ("coreai-core", "coreai-torch", "coreai-models", "torch", "safetensors", "tokenizers"):
        try:
            v[p] = md.version(p)
        except Exception as e:  # noqa: BLE001
            v[p] = repr(e)
    osb = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    return {"versions": v, "platform": platform.platform(), "macos_build": osb,
            "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
            "runtime": "coreai python runtime, AOT h16c GPU .aimodelc, SpecializationOptions.default(), no JIT",
            "readout": "host.py: z = h_slot . E[id] in float64 over the group ids (head/option_rows, fp32), group max, "
                       "softmax over the options",
            "gpu": "shared with other sessions, _GPU_LOCK read only (the ms are contended reference values)"}


def split(runs: list, per: int = RUNS_PER_PROCESS) -> list[list]:
    if not runs:
        return []
    n = math.ceil(len(runs) / per)
    size = math.ceil(len(runs) / n)
    return [runs[i:i + size] for i in range(0, len(runs), size)]


def run_shards(tag: str, shards: list[dict]) -> list[dict]:
    """Run the worker processes one after another; return their JSON records."""
    got = []
    for sp in shards:
        spec_path = Path(sp["out"]).with_suffix(".spec.json")
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(sp) + "\n")
        print(f"[{tag}] {Path(sp['out']).name}: {len(sp['runs'])} rows + reset re-run", flush=True)
        t0 = time.monotonic()
        proc = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--spec", str(spec_path)])
        got.append(read_shard(sp, proc.returncode, time.monotonic() - t0))
        if got[-1].get("failed"):
            raise SystemExit(f"{Path(sp['out']).name}: worker failed (exit {proc.returncode}); see its output above")
    return got


def read_shard(sp: dict, returncode: int | None = 0, wall: float | None = None) -> dict:
    js = Path(sp["out"]).with_suffix(".json")
    if returncode != 0 or not js.exists():
        print(f"{Path(sp['out']).name}: worker FAILED (exit {returncode})", flush=True)
        return {"spec": sp, "failed": True, "returncode": returncode, "process_wall_seconds": wall}
    rec = json.loads(js.read_text())
    rec["spec_runs"] = sp["runs"]
    rec["process_wall_seconds"] = wall if wall is not None else rec["finished"] - rec["started"]
    rec["returncode"] = returncode
    return rec


def collect(shard_recs: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Merge the shards: run records (+ arrays), and the process records."""
    runs, procs, arrays = [], [], {}
    for sr in shard_recs:
        prefix = Path(sr["spec"]["out"])
        entry = {"shard": prefix.name, "rows": len(sr.get("spec_runs", sr["spec"].get("runs", []))),
                 "prompts": len(sr.get("spec_runs", sr["spec"].get("runs", []))) + 1}
        if sr.get("failed"):
            entry.update({"failed": True, "returncode": sr["returncode"]})
            procs.append(entry)
            continue
        z = np.load(prefix.with_suffix(".npz"))
        entry.update({"pid": sr["pid"], "load_seconds": sr["load_seconds"], "load_split_seconds": sr["load_split_seconds"],
                      "function_names": sr["function_names"], "process_wall_seconds": sr["process_wall_seconds"],
                      "reset_check": sr["reset_check"], "descriptor": sr["descriptor"],
                      "contract_mismatch": sr["contract_mismatch"], "npz": str(prefix.with_suffix(".npz")),
                      "npz_sha256": sha256_file(prefix.with_suffix(".npz"))})
        procs.append(entry)
        for rec in sr["runs"]:
            key = f"{rec['index']:02d}"
            arrays[(rec["id"], rec["k"], rec["variant"])] = {
                "slot": z[f"{key}__slot"], "call_ms": z[f"{key}__call_ms"],
                "hidden": z[f"{key}__hidden"] if f"{key}__hidden" in z.files else None}
            runs.append({**rec, "shard": prefix.name, "npz_key": key, "first_in_process": rec["index"] == 0})
    return runs, procs, arrays


def readout_probs(slot16: np.ndarray, groups: list[list[int]], table: dict) -> list[float]:
    import export_option_rows as eor

    ids = host.group_ids(groups)
    return host.readout(slot16.astype(np.float32), eor.rows_for(table, ids), ids, groups)


def position_cos(hidden16: np.ndarray, ref: np.ndarray, ids: list[int]) -> dict:
    from parity_decoder_torch import compare_hidden, cos_rows

    c = cos_rows(hidden16.astype(np.float32), ref)
    return {**compare_hidden(hidden16.astype(np.float32), ref), "positions": int(c.size),
            "positions_below": {str(t): int((c < t).sum()) for t in COS_THRESHOLDS},
            "lowest_positions": [{"pos": int(i), "cos": float(c[i]), "id": int(ids[i])}
                                 for i in np.argsort(c)[:LOWEST_POSITIONS]]}


def summarize(runs: list[dict], expected: list[tuple[str, int]]) -> dict:
    have = {(r["id"], r["k"]) for r in runs}
    far = [r for r in runs if not r["near_tie"]]
    near = [r for r in runs if r["near_tie"]]
    worst = max(runs, key=lambda r: r["max_abs_dp"]) if runs else None
    hid = [r for r in runs if r.get("hidden")]
    warm = [r for r in runs if not r["first_in_process"]]
    ms = np.concatenate([r["_call_ms"] for r in runs]) if runs else np.zeros(0)
    wms = np.concatenate([r["_call_ms"] for r in warm] or [np.zeros(0)])
    tokens = int(sum(r["tokens"] for r in runs))
    padded = int(sum(r["padded_tokens"] for r in runs))
    return {
        "runs": len(runs), "expected_runs": len(expected),
        "missing_runs": [f"{a}:{b}" for a, b in expected if (a, b) not in have],
        "questions": len(runs), "argmax_equal": sum(r["argmax_equal"] for r in runs),
        "questions_non_near_tie": len(far), "argmax_equal_non_near_tie": sum(r["argmax_equal"] for r in far),
        "near_tie_questions": len(near), "argmax_equal_near_tie": sum(r["argmax_equal"] for r in near),
        "max_abs_dp": max((r["max_abs_dp"] for r in runs), default=None),
        "max_abs_dp_non_near_tie": max((r["max_abs_dp"] for r in far), default=None),
        "mean_of_run_mean_abs_dp": float(np.mean([r["mean_abs_dp"] for r in runs])) if runs else None,
        "mean_question_max_abs_dp": float(np.mean([r["max_abs_dp"] for r in runs])) if runs else None,
        "worst_run": None if worst is None else {"id": worst["id"], "name": worst["name"], "max_abs_dp": worst["max_abs_dp"]},
        "finite_all": all(r["finite"] for r in runs), "all_zero_runs": sum(r["all_zero"] for r in runs),
        "hidden_rows": len(hid), "min_pos_cos": min((r["hidden"]["min_pos_cos"] for r in hid), default=None),
        "hidden_max_abs_diff": max((r["hidden"]["max_abs_diff"] for r in hid), default=None),
        "tokens": tokens, "padded_tokens": padded, "calls": int(sum(r["calls"] for r in runs)),
        "pad_waste": (padded - tokens) / padded if padded else None,
        "ms_per_call_median": float(np.median(ms)) if ms.size else None,
        "ms_per_call_warm_median": float(np.median(wms)) if wms.size else None,
    }


def verdict(s: dict, resets_ok: bool) -> tuple[bool, dict]:
    checks = {
        "all_runs": s["runs"] == s["expected_runs"] and not s["missing_runs"],
        "a_argmax_non_near_tie": s["argmax_equal_non_near_tie"] == s["questions_non_near_tie"],
        "b_max_abs_dp": s["max_abs_dp"] is not None and s["max_abs_dp"] <= BAR["max_abs_dp"],
        "c_mean_of_run_mean_abs_dp": (s["mean_of_run_mean_abs_dp"] is not None
                                      and s["mean_of_run_mean_abs_dp"] <= BAR["mean_of_run_mean_abs_dp"]),
        "d_reset_bit_equal_all_processes": resets_ok,
        "e_finite": s["finite_all"] and s["all_zero_runs"] == 0,
    }
    return all(checks.values()), checks


def timing(runs: list[dict], procs: list[dict], S: int) -> dict:
    """Contended reference times: `warm` = every row but the first of its process."""
    def q(a) -> dict:
        a = np.asarray(a, np.float64)
        return {"n": int(a.size), "median": float(np.median(a)) if a.size else None,
                "p10": float(np.quantile(a, 0.1)) if a.size else None, "p90": float(np.quantile(a, 0.9)) if a.size else None}

    warm = [r for r in runs if not r["first_in_process"]]
    loads = [p["load_seconds"] for p in procs if "load_seconds" in p]
    return {"contended": True, "S": S,
            "load_seconds": {"first_process": loads[0] if loads else None,
                             "median": float(np.median(loads)) if loads else None, "all": loads},
            "ms_per_call_warm_runs": q(np.concatenate([r["_call_ms"] for r in warm] or [np.zeros(0)])),
            "ms_per_call_all_runs": q(np.concatenate([r["_call_ms"] for r in runs] or [np.zeros(0)])),
            "first_call_of_process_ms": [float(r["_call_ms"][0]) for r in runs if r["first_in_process"]],
            "row_wall_seconds_warm_runs": q([r["wall_seconds"] for r in warm])}


def select_rows(recs: dict, args) -> list[tuple[str, int]]:
    rows = [(rid, k) for rid, r in recs.items() for k in range(len(r["questions"]))]
    if args.rows:
        want = []
        for rid, name in json.loads(Path(args.rows).read_text())["rows"]:
            if rid not in recs:
                raise SystemExit(f"{args.rows}: no record {rid}")
            k = next((i for i, q in enumerate(recs[rid]["questions"]) if q["name"] == name), None)
            if k is None:
                raise SystemExit(f"{args.rows}: {rid} has no question {name}")
            want.append((rid, k))
        return want
    if args.subset == "quick":   # the first record of every source, all of its questions
        seen, keep = {}, set()
        for rid, r in recs.items():
            if seen.get(r["source"], 0) < QUICK_PER_SOURCE:
                seen[r["source"]] = seen.get(r["source"], 0) + 1
                keep.add(rid)
        rows = [x for x in rows if x[0] in keep]
    if args.records:
        keep = set(args.records.split(","))
        rows = [x for x in rows if x[0] in keep]
    if not rows:
        raise SystemExit("no rows selected")
    return rows


def resolve_oracle(args, b: Bundle) -> tuple[Path, dict, dict, Path | None]:
    """(oracle path, document, {id: record}, the hidden-row directory) — the toy oracle built when asked for and absent."""
    from parity_decoder_torch import build_toy_oracle, load_oracle

    if args.toy_oracle:
        if not b.toy:
            raise SystemExit(f"{b.name} is not a toy bundle: --toy-oracle applies to toy bundles only")
        path = Path(args.toy_oracle).expanduser().resolve()
        if not path.exists():
            build_toy_oracle(int(b.toy["seed"]), path, table_dir=b.dir / "head")
        doc, recs = load_oracle(path)
        if int(doc["toy"]["seed"]) != int(b.toy["seed"]):
            raise SystemExit(f"{path} is the seed-{doc['toy']['seed']} toy; the bundle is seed {b.toy['seed']}")
        tdir = doc["table"].get("dir")
        if tdir:
            import export_option_rows as eor
            ia, ra = eor.table_arrays(Path(tdir))
            ib, rb = eor.table_arrays(b.dir)
            if not (np.array_equal(ia, ib) and np.array_equal(ra.view(np.uint32), rb.view(np.uint32))):
                raise SystemExit(f"the toy oracle's table ({tdir}) differs from {b.dir}/head")
    else:
        if b.toy:
            raise SystemExit(f"{b.name} is a toy bundle: pass --toy-oracle")
        path = Path(args.oracle).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"no oracle {path} (oracle_d1.py, round 3)")
        doc, recs = load_oracle(path)
    hidden_dir = path.parent / "hidden"
    return path, doc, recs, hidden_dir if hidden_dir.exists() else None


def score_runs(runs: list[dict], arrays: dict, recs: dict, table: dict, hidden_dir: Path | None) -> list[dict]:
    from parity_decoder_torch import oracle_hidden, score_probs

    scored = []
    for r in runs:
        q = recs[r["id"]]["questions"][r["k"]]
        a = arrays[(r["id"], r["k"], r["variant"])]
        p = readout_probs(a["slot"], q["groups"], table)
        item = {**r, "source": recs[r["id"]]["source"], "name": q["name"], "type": q["type"], **score_probs(p, q),
                "call_ms": a["call_ms"].tolist(), "_call_ms": a["call_ms"]}
        if a.get("hidden") is not None:
            ref = oracle_hidden(hidden_dir, r["id"], r["k"])
            if ref is not None:
                item["hidden"] = position_cos(a["hidden"], ref, q["row_ids"])
        scored.append(item)
    return scored


def red_spec_runs(red: dict, recs: dict) -> list[dict]:
    runs = [{"id": rid, "k": k, "variant": "base", "ids": recs[rid]["questions"][k]["row_ids"],
             "slot": recs[rid]["questions"][k]["slot"]} for rid, k in red["base_rows"]]
    for arm in red["arms"]:
        for i, row in enumerate(arm["rows"]):
            runs.append({"id": row["base"][0], "k": row["base"][1], "variant": f"{arm['id']}#{i}", "ids": row["ids"],
                         "slot": row["slot"]})
    return runs


def arms_path(args, b: Bundle) -> Path:
    """--arms, else round 4's set (a toy bundle: round 1's, the set its arms oracle is built from)."""
    if args.arms:
        return Path(args.arms).expanduser().resolve()
    return RED_ARMS_ROUND1 if b.toy else RED_ARMS


def load_red(args, b: Bundle, recs: dict, table: dict) -> dict:
    """The arms' rows (`parity_decoder_torch.red_rows`) with the arms file's own path and sha256 (red_rows names
    fixtures/red_arms.json whatever it is given)."""
    from parity_decoder_torch import red_rows

    path = arms_path(args, b)
    if not path.exists():
        raise SystemExit(f"no arms file {path}")
    tok = host.load_tokenizer(b.dir / "tokenizer" / "tokenizer.json")
    red = red_rows(json.loads(path.read_text()), recs, tok, fold_vocab=b.vocab if b.toy else None,
                   table_ids=None if b.toy else list(table))
    red["file"], red["file_sha256"] = str(path), sha256_file(path)
    return red


def oracle_arms_sha256(path: Path) -> str | None:
    """The sha256 of the arms file an arms oracle was run on: a toy arms oracle names it (`red_arms`); an oracle_d1.py
    run names its fixture file (`fixtures`, checked against its sha256), whose `red_arms` names the arms file."""
    doc = json.loads(path.read_text())
    if doc.get("red_arms"):
        return doc["red_arms"].get("sha256")
    fx = doc.get("fixtures") or {}
    if not fx.get("path") or not Path(fx["path"]).exists() or sha256_file(Path(fx["path"])) != fx.get("sha256"):
        return None
    return (json.loads(Path(fx["path"]).read_text()).get("red_arms") or {}).get("sha256")


def resolve_red_oracle(args, b: Bundle, red: dict) -> tuple[Path, dict] | None:
    """The arms' oracle: --red-oracle, else <toy oracle dir>/red/records_oracle.json for a toy bundle (built from the
    toy module when absent), else the one <work>/oracle/*/records_oracle.json run on the arms file; None = no
    pre-check. An oracle run on another arms file is refused."""
    from parity_decoder_torch import build_toy_red_oracle, load_oracle

    if args.red_oracle:
        path = Path(args.red_oracle).expanduser().resolve()
        if not path.exists() and not b.toy:
            raise SystemExit(f"no arms oracle {path}")
    elif b.toy:
        path = Path(args.toy_oracle).expanduser().resolve().parent / "red" / "records_oracle.json"
    else:
        found = [p for p in sorted((LANE / "oracle").glob("*/records_oracle.json"))
                 if oracle_arms_sha256(p) == red["file_sha256"]]
        if len(found) > 1:
            raise SystemExit(f"several arms oracles were run on {red['file']}: {[str(p) for p in found]}; pass --red-oracle")
        if not found:
            return None
        path = found[0]
    if not path.exists():
        build_toy_red_oracle(int(b.toy["seed"]), Path(args.toy_oracle).expanduser().resolve(), path)
    if oracle_arms_sha256(path) != red["file_sha256"]:
        raise SystemExit(f"{path} was not run on {red['file']} (sha256 {red['file_sha256'][:16]}): pass the arms "
                         "oracle of that file (--red-oracle) or the arms file of that oracle (--arms)")
    doc, recs = load_oracle(path)
    return path, recs


def oracle_precheck(red: dict, recs: dict, red_oracle) -> dict:
    """The red rule on the oracle's own probabilities: each perturbed row's oracle p against its base row's oracle p."""
    from parity_decoder_torch import score_red_arm

    if red_oracle is None:
        return {"available": False, "why": "no arms oracle: readout_gate.py red-records, then oracle_d1.py on it (round 3)"}
    path, rrecs = red_oracle
    base = {(rid, k): recs[rid]["questions"][k]["probs"] for rid, k in red["base_rows"]}
    arms = []
    for arm in red["arms"]:
        probs = []
        for row in arm["rows"]:
            rec = rrecs.get(row["request"])
            q = next((x for x in (rec or {}).get("questions", []) if x["name"] == row["question"]), None)
            if q is None:
                raise SystemExit(f"{path}: no oracle row for {row['request']}/{row['question']}")
            if q.get("keys", row["keys"]) != row["keys"]:
                raise SystemExit(f"{path}: {row['request']}/{row['question']} keys differ from the base question's")
            probs.append(q["probs"])
        arms.append(score_red_arm(arm, probs, base, recs))
    return {"available": True, "path": str(path), "sha256": sha256_file(path), "arms": arms,
            "red_on_oracle": [a["id"] for a in arms if a["red"]], "replace": [a["id"] for a in arms if not a["red"]]}


def score_red(red: dict, rruns: list[dict], rarr: dict, recs: dict, table: dict, gate_sha: dict, red_oracle=None) -> dict:
    from parity_decoder_torch import RED_RULE, score_red_arm

    base_probs = {(rid, k): readout_probs(rarr[(rid, k, "base")]["slot"], recs[rid]["questions"][k]["groups"], table)
                  for rid, k in red["base_rows"]}
    sha = {(r["id"], r["k"], r["variant"]): r["hidden_sha256"] for r in rruns}
    base_bits = [{"row": f"{rid}:{recs[rid]['questions'][k]['name']}",
                  "bit_equal_gate_run": (sha[(rid, k, "base")] == gate_sha[(rid, k)]) if (rid, k) in gate_sha else None}
                 for rid, k in red["base_rows"]]
    arms = []
    for arm in red["arms"]:
        probs = [readout_probs(rarr[(row["base"][0], row["base"][1], f"{arm['id']}#{i}")]["slot"], row["groups"], table)
                 for i, row in enumerate(arm["rows"])]
        arms.append(score_red_arm(arm, probs, base_probs, recs))
    pre = oracle_precheck(red, recs, red_oracle)
    verdict_red: dict = {"all_arms_red": all(a["red"] for a in arms)}
    if pre["available"]:
        for g, o in zip(arms, pre["arms"]):
            dd = [float(np.max(np.abs((np.asarray(gi["perturbed_probs"]) - np.asarray(gi["base_probs"]))
                                      - (np.asarray(oi["perturbed_probs"]) - np.asarray(oi["base_probs"])))))
                  for gi, oi in zip(g["items"], o["items"])]
            g["red_on_oracle"] = o["red"]
            g["oracle_max_abs_dp_vs_base"] = o["max_abs_dp_vs_base"]
            g["oracle_mean_of_run_mean_abs_dp_vs_base"] = o["mean_of_run_mean_abs_dp_vs_base"]
            g["graph_dp_minus_oracle_dp_max_abs"] = max(dd)
        required = [a for a in arms if a["red_on_oracle"]]
        verdict_red = {"required_arms": [a["id"] for a in required], "replace": pre["replace"],
                       "required_arms_red": bool(required) and all(a["red"] for a in required),
                       "graph_dp_equals_oracle_dp": all(a["graph_dp_minus_oracle_dp_max_abs"] <= BAR["max_abs_dp"]
                                                        for a in arms),
                       "graph_dp_minus_oracle_dp_max_abs": max(a["graph_dp_minus_oracle_dp_max_abs"] for a in arms),
                       "all_arms_red": all(a["red"] for a in arms)}
    return {"rule": RED_RULE, "rows_file": red["file"], "rows_file_sha256": red["file_sha256"], "folded": red["folded"],
            "oracle_precheck": {k: v for k, v in pre.items() if k != "arms"} | (
                {"arms": [{k: v for k, v in a.items() if k != "items"} for a in pre["arms"]]} if pre["available"] else {}),
            "arms": arms, "verdict": verdict_red, "base_vs_gate_run": base_bits,
            "base_bit_equal_all": all(x["bit_equal_gate_run"] for x in base_bits if x["bit_equal_gate_run"] is not None),
            "all_arms_red": verdict_red["all_arms_red"]}


def red_checks(red_rec: dict) -> dict:
    """The red arm's entries in the transcript's checks: with the oracle pre-check, the arms red on the oracle must be
    red on the graph and the graph's dp must equal the oracle's; without it, every arm red."""
    v = red_rec["verdict"]
    if "required_arms" in v:
        return {"red_required_arms_red": v["required_arms_red"], "red_graph_dp_equals_oracle_dp": v["graph_dp_equals_oracle_dp"],
                "red_base_bit_equal_gate_runs": red_rec["base_bit_equal_all"]}
    return {"red_all_arms_red": v["all_arms_red"], "red_base_bit_equal_gate_runs": red_rec["base_bit_equal_all"]}


def open_transcript(path: Path, skeleton: dict) -> None:
    """The transcript starts as the bar and what is about to run, written before the first GPU process."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(skeleton, indent=1) + "\n")


def gate(args) -> int:
    b = Bundle(args.bundle, args.aimodelc)
    if not b.aimodelc.exists():
        raise SystemExit(f"no AOT asset {b.aimodelc} (export_decoder.py --aot)")
    out = Path(args.transcript)
    if out.exists() and not args.rescore:
        raise SystemExit(f"{out} exists: transcripts are never overwritten")
    oracle_path, odoc, recs, hidden_dir = resolve_oracle(args, b)
    table = b.table()
    outside = sorted({i for r in recs.values() for q in r["questions"] for g in q["groups"] for i in g} - set(table))
    if outside:
        raise SystemExit(f"the oracle reads ids outside {b.dir}/head: {outside[:10]}")
    expected = select_rows(recs, args)
    tag = args.tag or b.name
    work = (LANE / ("readout_toy" if b.toy else "readout")) / tag
    work.mkdir(parents=True, exist_ok=True)
    red = load_red(args, b, recs, table) if args.red else None
    red_oracle = resolve_red_oracle(args, b, red) if red else None
    arms = {"arms_file": red["file"], "arms_sha256": red["file_sha256"],
            "red_oracle": None if red_oracle is None else {"path": str(red_oracle[0]),
                                                           "sha256": sha256_file(red_oracle[0])}} if red else {}
    bundle_rec = b.record()
    skeleton = {"schema": "d1-decoder-readout-gate/1", "status": "running", "started": now(), "bar": BAR,
                "red_rule": red["rule"] if red else None, "bundle": bundle_rec["name"],
                "aimodelc_tree_sha256": bundle_rec["aimodelc"]["tree_sha256"],
                "oracle": {"path": str(oracle_path), "sha256": sha256_file(oracle_path), "toy": bool(b.toy)},
                "rows": len(expected), "processes": len(split(expected)), "red_arms": len(red["arms"]) if red else 0,
                **arms}
    if not args.rescore:
        open_transcript(out, skeleton)
    hidden_keep = {rid for rid in recs if hidden_dir is not None and (hidden_dir / f"{rid}.npz").exists()}
    base_spec = {"aimodelc": str(b.aimodelc), "S": b.S, "max_ctx": b.max_ctx, "hidden": b.hidden, "pad_id": b.pad_id,
                 "contract": b.contract}
    shards = []
    for n, part in enumerate(split(expected)):
        runs = [{"id": rid, "k": k, "variant": "base", "ids": recs[rid]["questions"][k]["row_ids"],
                 "slot": recs[rid]["questions"][k]["slot"]} for rid, k in part]
        shards.append({**base_spec, "runs": runs, "out": str(work / f"shard_{n:02d}"),
                       "keep_full": [i for i, (rid, _) in enumerate(part) if rid in hidden_keep]})
    red_spec = {**base_spec, "runs": red_spec_runs(red, recs), "out": str(work / "red_arms"), "keep_full": []} if red else None
    t0 = time.monotonic()
    others_start, lock_start, load_start = other_gpu_processes(), gpu_lock_state(), load_line()
    if args.rescore:
        shard_recs = [read_shard(sp) for sp in shards]
        red_recs = [read_shard(red_spec)] if red_spec else []
    else:
        shard_recs = run_shards(f"gate {tag}", shards)
        red_recs = run_shards(f"red {tag}", [red_spec]) if red_spec else []
    gpu_seconds = time.monotonic() - t0
    others_end, lock_end, load_end = other_gpu_processes(), gpu_lock_state(), load_line()
    runs, procs, arrays = collect(shard_recs)
    t1 = time.monotonic()
    scored = score_runs(runs, arrays, recs, table, hidden_dir)
    s = summarize(scored, expected)
    resets = all(p.get("reset_check", {}).get("bit_equal", False) for p in procs)
    by_src: dict = {}
    for x in scored:
        by_src.setdefault(x["source"], []).append(x)
    by_src = {k: summarize(v, [(x["id"], x["k"]) for x in v]) for k, v in sorted(by_src.items())}
    red_rec = None
    if red_recs:
        rruns, rprocs, rarr = collect(red_recs)
        red_rec = {**score_red(red, rruns, rarr, recs, table, {(r["id"], r["k"]): r["hidden_sha256"] for r in runs},
                               red_oracle),
                   "process": rprocs, "runs": rruns,
                   "reset_bit_equal": all(p.get("reset_check", {}).get("bit_equal", False) for p in rprocs)}
    ok, checks = verdict(s, resets and (red_rec is None or red_rec["reset_bit_equal"]))
    if red_rec:
        checks.update(red_checks(red_rec))
        ok = all(checks.values())
    descs = [p["descriptor"] for p in procs if "descriptor" in p]
    record = {
        "schema": "d1-decoder-readout-gate/1",
        "gate": "the decoder alone on the Mac GPU (the oracle's row ids), the fp16 slot hidden read out through host.py",
        "bar": BAR, "red_rule": red_rec["rule"] if red_rec else None, **arms,
        "result": "PASS" if ok else "FAIL", "checks": checks,
        "bundle": bundle_rec, "chunk": b.S, "max_ctx": b.max_ctx, "readout": b.meta["decision"]["readout"],
        "descriptor": descs[0] if descs else None,
        "descriptor_same_in_every_process": all(d == descs[0] for d in descs) if descs else None,
        "contract": b.contract,
        "oracle": {"path": str(oracle_path), "sha256": sha256_file(oracle_path), "schema": odoc.get("schema"),
                   "toy": odoc.get("toy"), "hidden_dir": str(hidden_dir) if hidden_dir else None},
        "subset": args.subset, "rows_file": args.rows, "records": args.records,
        "script": {"path": "conversion/d1/readout_gate.py", "sha256": sha256_file(Path(__file__).resolve())},
        "environment": env_record(),
        "gpu_lock": {"start": lock_start, "end": lock_end}, "load": {"start": load_start, "end": load_end},
        "other_gpu_processes": {"start": others_start, "end": others_end},
        "runtime_cache_entry": runtime_cache_entry(bundle_rec["aimodelc_main_hash"]),
        "processes": procs, "summary": s, "by_source": by_src,
        "near_ties": [{k: r[k] for k in ("id", "name", "oracle_top2_margin", "argmax_equal", "max_abs_dp", "probs",
                                         "probs_oracle")} for r in scored if r["near_tie"]],
        "red_arm": red_rec, "timing": timing(scored, procs, b.S),
        "seconds": {"gpu_processes": gpu_seconds, "scoring": time.monotonic() - t1, "total": time.monotonic() - t0},
        "runs": [{k: v for k, v in r.items() if k != "_call_ms"} for r in scored],
    }
    if args.compare_with:
        record["compare_with"] = compare_with(scored, arrays, Path(args.compare_with))
    if args.note:
        record["note"] = args.note
    record["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out.write_text(json.dumps(record, indent=1) + "\n")
    print_summary(record)
    return 0 if ok else 1


def print_summary(record: dict) -> None:
    s, b = record["summary"], record["bundle"]
    print(f"{record['result']}: {b['name']} rows {s['runs']}/{s['expected_runs']} argmax {s['argmax_equal']}/{s['questions']} "
          f"(non-near-tie {s['argmax_equal_non_near_tie']}/{s['questions_non_near_tie']}, near-tie "
          f"{s['argmax_equal_near_tie']}/{s['near_tie_questions']}) max|dp| {s['max_abs_dp']:.3e} mean "
          f"{s['mean_of_run_mean_abs_dp']:.3e} min cos {s['min_pos_cos']} worst {s['worst_run']}")
    print(f"  calls {s['calls']}, pad waste {s['pad_waste']:.4f}, ms/call warm median {s['ms_per_call_warm_median']} "
          f"(contended); load {record['timing']['load_seconds']['median']} s")
    for k, v in record["checks"].items():
        print(f"  {k}: {'ok' if v else 'FAIL'}")
    red = record.get("red_arm")
    if red:
        for a in red["arms"]:
            print(f"red {a['id']}: argmax moved {a['argmax_moved']}/{a['rows']} max|dp| {a['max_abs_dp_vs_base']:.4f} "
                  f"mean {a['mean_of_run_mean_abs_dp_vs_base']:.5f} -> {'RED' if a['red'] else 'NOT RED'} "
                  f"{json.dumps(a['red_facts7'])}")
        print(f"red base = gate run (bit): {[x['bit_equal_gate_run'] for x in red['base_vs_gate_run']]}, "
              f"reset {red.get('reset_bit_equal')}")
        pre = red["oracle_precheck"]
        if pre["available"]:
            print(f"red oracle pre-check: red on the oracle {pre['red_on_oracle']}, replace {pre['replace']}; graph dp - "
                  f"oracle dp max {red['verdict']['graph_dp_minus_oracle_dp_max_abs']:.2e}")
        else:
            print(f"red oracle pre-check: not run ({pre['why']})")
    if "compare_with" in record:
        c = record["compare_with"]
        print(f"vs {c['other_bundle']}: rows {c['runs']} max|dp| {c['max_abs_dp']} argmax-equal {c['argmax_equal_runs']} "
              f"slot hidden max|d| {c['slot_hidden_max_abs_diff']}")
    print(f"transcript written ({record['generated_at']})")


def shard_arrays(transcript: dict) -> dict:
    """(id, k) -> the slot row of every base run of a transcript (its shard npz)."""
    out, zs = {}, {}
    for r in transcript["runs"]:
        npz = next((p["npz"] for p in transcript["processes"] if p.get("shard") == r["shard"] and p.get("npz")), None)
        if npz and Path(npz).exists():
            z = zs.setdefault(npz, np.load(npz))
            out[(r["id"], r["k"])] = z[f"{r['npz_key']}__slot"]
    return out


def compare_with(runs: list[dict], arrays: dict, other_path: Path) -> dict:
    """The same rows in another transcript (another mode or width): p and the slot hidden row side by side."""
    other = json.loads(other_path.read_text())
    o_runs = {(r["id"], r["k"]): r for r in other["runs"]}
    o_slot = shard_arrays(other)
    rows = []
    for r in runs:
        o = o_runs.get((r["id"], r["k"]))
        if o is None:
            continue
        row = {"id": r["id"], "name": r["name"], "tokens": r["tokens"],
               "max_abs_dp": float(np.max(np.abs(np.asarray(r["probs"]) - np.asarray(o["probs"])))),
               "argmax_equal": r["argmax"] == o["argmax"], "max_abs_dp_vs_oracle": [o["max_abs_dp"], r["max_abs_dp"]]}
        if (r["id"], r["k"]) in o_slot:
            a = arrays[(r["id"], r["k"], "base")]["slot"].astype(np.float32)
            row["slot_hidden_max_abs_diff"] = float(np.max(np.abs(a - o_slot[(r["id"], r["k"])].astype(np.float32))))
        rows.append(row)
    return {"other": str(other_path), "other_bundle": other["bundle"]["name"], "other_chunk": other["chunk"],
            "runs": len(rows), "max_abs_dp": max((x["max_abs_dp"] for x in rows), default=None),
            "argmax_equal_runs": sum(x["argmax_equal"] for x in rows),
            "slot_hidden_max_abs_diff": max((x.get("slot_hidden_max_abs_diff", 0.0) for x in rows), default=None),
            "vs_oracle_max_abs_dp": {"other": max((x["max_abs_dp_vs_oracle"][0] for x in rows), default=None),
                                     "this": max((x["max_abs_dp_vs_oracle"][1] for x in rows), default=None)},
            "rows": rows}


def red_cmd(args) -> int:
    """The red arms alone, in their own process, against an existing gate transcript of the same asset."""
    b = Bundle(args.bundle, args.aimodelc)
    out = Path(args.transcript)
    if out.exists():
        raise SystemExit(f"{out} exists: transcripts are never overwritten")
    gate_t = json.loads(Path(args.gate_transcript).read_text())
    if gate_t["bundle"]["aimodelc"]["tree_sha256"] != tree_digest(b.aimodelc)["tree_sha256"]:
        raise SystemExit("the gate transcript was taken on another asset")
    oracle_path, odoc, recs, _ = resolve_oracle(args, b)
    if sha256_file(oracle_path) != gate_t["oracle"]["sha256"]:
        raise SystemExit("the gate transcript was read against another oracle")
    table = b.table()
    red = load_red(args, b, recs, table)
    red_oracle = resolve_red_oracle(args, b, red)
    arms = {"arms_file": red["file"], "arms_sha256": red["file_sha256"],
            "red_oracle": None if red_oracle is None else {"path": str(red_oracle[0]), "sha256": sha256_file(red_oracle[0])}}
    open_transcript(out, {"schema": "d1-decoder-readout-red-arms/1", "status": "running", "started": now(),
                          "red_rule": red["rule"], "bar": BAR, "bundle": b.name, **arms})
    work = (LANE / ("readout_toy" if b.toy else "readout")) / (args.tag or b.name)
    spec = {"aimodelc": str(b.aimodelc), "S": b.S, "max_ctx": b.max_ctx, "hidden": b.hidden, "pad_id": b.pad_id,
            "contract": b.contract, "runs": red_spec_runs(red, recs), "out": str(work / "red_arms_alone"), "keep_full": []}
    t0 = time.monotonic()
    rruns, rprocs, rarr = collect(run_shards(f"red {b.name}", [spec]))
    gate_sha = {(r["id"], r["k"]): r["hidden_sha256"] for r in gate_t["runs"]}
    red_rec = {**score_red(red, rruns, rarr, recs, table, gate_sha, red_oracle), "process": rprocs, "runs": rruns,
               "reset_bit_equal": all(p.get("reset_check", {}).get("bit_equal", False) for p in rprocs)}
    checks = {**red_checks(red_rec), "red_reset_bit_equal": red_rec["reset_bit_equal"]}
    ok = all(checks.values())
    record = {"schema": "d1-decoder-readout-red-arms/1", "red_rule": red_rec["rule"], "bar": BAR, "checks": checks,
              "result": "PASS" if ok else "FAIL", **arms, "bundle": b.record(),
              "gate_transcript": {"path": str(Path(args.gate_transcript).resolve()),
                                  "sha256": sha256_file(Path(args.gate_transcript)), "result": gate_t["result"]},
              "oracle": {"path": str(oracle_path), "sha256": sha256_file(oracle_path)},
              "script": {"path": "conversion/d1/readout_gate.py", "sha256": sha256_file(Path(__file__).resolve())},
              "environment": env_record(), "gpu_lock": gpu_lock_state(), "red_arm": red_rec,
              "seconds": time.monotonic() - t0, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out.write_text(json.dumps(record, indent=1) + "\n")
    for a in red_rec["arms"]:
        print(f"red {a['id']}: argmax moved {a['argmax_moved']}/{a['rows']} max|dp| {a['max_abs_dp_vs_base']:.4f} -> "
              f"{'RED' if a['red'] else 'NOT RED'}; on the oracle {a.get('oracle_max_abs_dp_vs_base')} "
              f"({'red' if a.get('red_on_oracle') else 'not red' if 'red_on_oracle' in a else 'no pre-check'}), graph dp - "
              f"oracle dp {a.get('graph_dp_minus_oracle_dp_max_abs')}")
    print(f"{record['result']} {json.dumps(checks)}; replace {red_rec['verdict'].get('replace')}")
    return 0 if ok else 1


def red_records_cmd(args) -> int:
    """fixtures/red_arms_records.json: the arms' perturbed requests as fixture records (oracle_d1.py --fixtures)."""
    from parity_decoder_torch import red_records

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists: never overwritten")
    src = LANE / "fixtures" / "red_arms.json"
    recs = red_records(json.loads(src.read_text()))
    doc = {"schema": "d1-fixtures/1", "what": "the red arms' perturbed requests as fixture records, for oracle_d1.py "
                                              "--fixtures (the oracle side of readout_gate.py's red pre-check)",
           "red_arms": {"path": str(src), "sha256": sha256_file(src)}, "records": recs,
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(recs)} records -> {out}")
    return 0


def merge_cmd(args) -> int:
    """Transcripts of the same asset and oracle over disjoint row sets -> one transcript (summary and checks over the
    union; each run kept as its transcript scored it, tagged with the transcript)."""
    out = Path(args.transcript)
    if out.exists():
        raise SystemExit(f"{out} exists: transcripts are never overwritten")
    docs = [(Path(p), json.loads(Path(p).read_text())) for p in args.transcripts]
    keys = {(d["bundle"]["aimodelc"]["tree_sha256"], d["oracle"]["sha256"]) for _, d in docs}
    if len(keys) != 1:
        raise SystemExit(f"the transcripts are not of one asset and one oracle: {keys}")
    runs, procs, seen = [], [], set()
    for p, d in docs:
        for r in d["runs"]:
            if (r["id"], r["k"]) in seen:
                raise SystemExit(f"{r['id']}:{r['k']} is in two transcripts: the row sets must be disjoint")
            seen.add((r["id"], r["k"]))
            runs.append({**r, "from": str(p), "_call_ms": np.asarray(r["call_ms"])})
        procs += [{**x, "from": str(p)} for x in d["processes"]]
    s = summarize(runs, [(r["id"], r["k"]) for r in runs])
    resets = all(x.get("reset_check", {}).get("bit_equal", False) for x in procs)
    reds = [d["red_arm"] for _, d in docs if d.get("red_arm")]
    ok, checks = verdict(s, resets and all(r["reset_bit_equal"] for r in reds))
    for r in reds:
        checks.update(red_checks(r))
    ok = all(checks.values())
    first = docs[0][1]
    record = {"schema": "d1-decoder-readout-gate/1", "merged_from": [str(p) for p, _ in docs], "bar": BAR,
              "result": "PASS" if ok else "FAIL", "checks": checks, "bundle": first["bundle"], "chunk": first["chunk"],
              "oracle": first["oracle"], "processes": procs, "summary": s, "red_arm": reds[0] if reds else None,
              "timing": timing(runs, procs, first["chunk"]),
              "runs": [{k: v for k, v in r.items() if k != "_call_ms"} for r in runs],
              "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"{record['result']}: merged {len(docs)} transcripts, rows {s['runs']}, max|dp| {s['max_abs_dp']}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run", help="the gate: workers on the GPU, then the host's readout on their slot rows")
    a.add_argument("bundle")
    a.add_argument("--transcript", required=True)
    a.add_argument("--subset", default="all", choices=["all", "quick"],
                   help="quick = the first record of every source (all of its questions)")
    a.add_argument("--rows", help='a JSON file {"rows": [[record id, question name], ...]}: only these rows, in order')
    a.add_argument("--records", help="comma list of record ids to keep")
    a.add_argument("--red", action="store_true", help="add the five red arms of --arms (their own process)")
    a.add_argument("--compare-with", help="another transcript of the same rows (another mode or width)")
    a.add_argument("--tag", help="shard directory name under readout[_toy]/ (default: the bundle name)")
    a.add_argument("--rescore", action="store_true", help="re-score existing shards, no GPU runs")
    a.add_argument("--note", help="free text kept in the transcript")
    rd = sub.add_parser("red", help="the red arms alone, against an existing gate transcript of the same asset")
    rd.add_argument("bundle")
    rd.add_argument("--gate-transcript", required=True)
    rd.add_argument("--transcript", required=True)
    rd.add_argument("--tag")
    for sp in (a, rd):
        sp.add_argument("--oracle", default=str(ORACLE), help="records_oracle.json (oracle_d1.py; round 3)")
        sp.add_argument("--toy-oracle", help="a toy bundle's oracle JSON (built from the bundle when it does not exist)")
        sp.add_argument("--aimodelc", help="the compiled asset (default <bundles>_aotc/<name>.h16c.aimodelc)")
        sp.add_argument("--arms", help="the red arms file (default <work>/fixtures/red_arms_r4.json, round 4's set; a toy "
                                       "bundle: <work>/fixtures/red_arms.json, round 1's set)")
        sp.add_argument("--red-oracle", help="the arms' oracle, run on the --arms file (default: <toy oracle dir>/red/ for "
                                             "a toy bundle, built when absent; for the model the one "
                                             "<work>/oracle/*/records_oracle.json whose fixture file names the arms file)")
    rr = sub.add_parser("red-records", help="round 3: the arms' requests as fixture records for oracle_d1.py")
    rr.add_argument("--out", default=str(LANE / "fixtures" / "red_arms_records.json"))
    mg = sub.add_parser("merge", help="transcripts of one asset over disjoint row sets -> one")
    mg.add_argument("transcripts", nargs="+")
    mg.add_argument("--transcript", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--spec", required=True)
    args = ap.parse_args()
    if args.cmd == "worker":
        return worker(Path(args.spec))
    if args.cmd == "red":
        return red_cmd(args)
    if args.cmd == "merge":
        return merge_cmd(args)
    if args.cmd == "red-records":
        return red_records_cmd(args)
    return gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
