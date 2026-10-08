#!/usr/bin/env python3
"""Readout gate, end to end: a picture record's file -> vision_host -> the tower bundle -> the decoder bundle -> host.py's
readout, on the Mac GPU, against the provider's fp32 oracle of the same record (oracle_d1.py --images).

Per fixture record (`fixtures/image_records.json`; a URL record is skipped, as the oracle skipped it):
  pictures  each file -> vision_host.to_rgb -> cap_pixels -> plan -> crop_images -> tower_inputs(crop, the tower bundle's
            host/position_embedding.safetensors) -> the tower's AOT h16c `.aimodelc` (`SpecializationOptions.default()`,
            never the JIT), one call per crop in crop order -> each crop's first h w / 4 rows, pictures in text order ->
            cast to fp16 into the decoder's image_embeds [N, d] (rows after zero; one buffer bound to every call)
  rows      per question: host.validate_question + host.build_question on the decoder bundle's tokenizer (aliases,
            readout groups, keys, suffix); ids = vision_host.prompt_ids(image_prefix_text(state, pictures) + suffix,
            plans), which must equal the oracle's processor ids (and the groups / keys the oracle's), then <image> ->
            V + k (vision_host.extension_ids); the bundle's option table and the position bound are checked as in decide.py
  decoder   the decoder bundle's AOT `main` from fresh zero states, S ids a call (position_ids 0..cS+S-1), the last call
            padded with <|pad|>, the padded rows dropped -> hidden [T, d] fp16; the slot's row -> host.readout over the
            bundle's head/option_rows (float64, group max, softmax)

Compared with the oracle in its two forms: `row` = each question's whole prompt in one plain pass (oracle
`questions[k].probs`, what the graph computes; readout_gate.py's oracle form) and `api` = the provider's API path
(`api.probs`: one question = that plain pass, several = its Tree). The design asks both to hold the bar.

Bar (FACTS §7, readout_gate.py's, fixed before any result: written at the top of the transcript before the first GPU
process), on the `e2e` arm, against each oracle form:
  (a) argmax = the oracle's on every question whose oracle top-2 margin is above 0.02 (near-ties listed apart)
  (b) max |dp| <= 0.02 over every option of every question, near-ties included
  (c) the mean over runs of the run's mean |dp| over its options <= 0.002 (a run = one row = one question)
  (d) every process re-runs its first row at the end and reproduces its hidden rows bit for bit (the state reset)
  (e) every hidden value of every row finite, no row all zero
plus every expected row present, and the host's ids, groups and keys equal to the oracle's on every row.

Arms (each row of each arm a run; a record's arms in one process, the tower run once per record per process):
  e2e       the tower's rows: the gate
  zero      image_embeds all zero, the extension ids kept: a control that must be red (= the arm fails (a), (b) or (c)
            against the row-form oracle), so the gate is seen to depend on the pictures at all
  reversed  the crops' rows concatenated in reverse crop order (records with more than one crop): recorded
  hf_rows   the HF tower oracle's rows (vision_oracle.py, fp32 -> fp16) in place of the tower's: the decoder alone,
            recorded; beside e2e it splits the error between the tower and the decoder
Recorded beside the bar: per crop the tower's rows against the HF tower oracle (cos, lowest row, max |d|) and its four
inputs against the oracle npz's (bit for bit); the oracle's --hidden rows against the graph's (per-position cosine, image
and text positions apart); the tower's outputs of a record run in two processes (bit for bit); contended ms.

Process split: the Python runtime leaks an IOSurface per call, so a process takes at most CALLS_PER_PROCESS decoder
calls (+ its tower calls and the reset re-run); the processes run one after another. The GPU is shared with other
sessions: the ms are contended reference values for the transcript only.

    cd conversion/d1
    PY=<coreai-models venv>/bin/python Q="$HOME/code/standup/tools/quiet/quiet_wait.py --max-wait 3600 --"
    $Q $PY readout_gate_vision.py run --tower $ZOO_WORK_ROOT/_d1_3b/exports/vision/d1_3b_vision_fp16w32 \\
        --decoder $ZOO_WORK_ROOT/_d1_3b/exports/bundles/d1_3b_decode_fp16_pf16 --transcript <json>
        [--records <image_records.json>] [--oracle <records_oracle_images.json>] [--arms e2e,zero,reversed,hf_rows]
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
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
import vision_host as vh  # noqa: E402
from _paths import gpu_lock, work_path  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")

LANE = work_path("_d1_3b")
RECORDS = LANE / "fixtures" / "image_records.json"
ORACLE = LANE / "oracle" / "records_oracle_images.json"
TOWER_ORACLE = LANE / "oracle" / "images" / "oracle.json"
WORK = LANE / "readout_vision"
BAR = {"max_abs_dp": 0.02, "mean_of_run_mean_abs_dp": 0.002, "near_tie_top2_margin": 0.02,
       "argmax": "every question with an oracle top-2 margin above 0.02; near-ties listed apart",
       "max_abs_dp_applies_to": "every option of every question, near-ties included",
       "mean_of_run_mean_abs_dp_definition": "mean over runs (one run = one row = one question) of the mean |dp| "
                                             "over that question's options",
       "reset": "every process re-runs its first row last: hidden rows bit-equal",
       "finite": "every hidden value of every row finite, no row all zero", "rows": "every expected row present",
       "ids": "the host's ids, groups and keys = the oracle's on every row",
       "applies_to": "the e2e arm, against the oracle's row form and its API path"}
RED_RULE = ("the zero arm is red when its rows fail the bar against the row-form oracle: an argmax moves on a "
            "non-near-tie question, or max |dp| > 0.02, or the mean of the rows' mean |dp| > 0.002")
ARMS = ("e2e", "zero", "reversed", "hf_rows")
CALLS_PER_PROCESS = 700
LOWEST_POSITIONS = 6
INPUTS = ("patches", "pos_table", "key_bias", "unshuffle_idx")
OTHER_GPU = re.compile(r"yardstick|litert|llm-bench|coreai_verify|coreai-build|readout_gate|gate_|parity_|mlx|"
                       r"decide\.py|timing\.py|--accel gpu")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def tree_digest(path: Path) -> dict:
    files = sorted(p for p in path.rglob("*") if p.is_file())
    per = {str(p.relative_to(path)): sha256_file(p) for p in files}
    tree = hashlib.sha256("".join(f"{k}\0{v}\n" for k, v in per.items()).encode()).hexdigest()
    return {"path": str(path), "bytes": sum(p.stat().st_size for p in files), "tree_sha256": tree}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def aot_path(bundle: Path, name: str) -> Path:
    """<bundles>/<name> -> <bundles>_aotc/<name>.h16c.aimodelc (readout_gate.Bundle's and export_vision.py's rule)."""
    return bundle.parent.parent / f"{bundle.parent.name}_aotc" / f"{name}.h16c.aimodelc"


# --------------------------------------------------------------------------- the two bundles
class Decoder:
    def __init__(self, path: Path):
        self.dir = Path(path).expanduser().resolve()
        self.meta = json.loads((self.dir / "metadata.json").read_text())
        lang = self.meta["language"]
        self.name = self.meta["name"]
        self.S = int(lang["prefill_chunk"])
        self.max_ctx = int(lang["max_context_length"])
        self.contract = lang["contract"]
        self.hidden = int(self.contract["outputs"]["hidden"][0][2])
        self.N = int(self.contract["inputs"]["image_embeds"][0][0])
        self.vocab = int(lang["vocab_size"])
        if self.meta.get("toy"):
            raise SystemExit(f"{self.name} is a toy bundle: this gate reads the model's oracle")
        if self.vocab != vh.V:
            raise SystemExit(f"vocab_size {self.vocab} != the extension base {vh.V}")
        self.aimodelc = aot_path(self.dir, self.name)

    def record(self) -> dict:
        mh = self.aimodelc / "main.hash"
        return {"bundle": str(self.dir), "name": self.name, "compression": self.meta.get("compression"),
                "prefill_chunk": self.S, "max_ctx": self.max_ctx, "n_image_rows": self.N,
                "metadata_sha256": sha256_file(self.dir / "metadata.json"),
                "head_sha256": {f.name: sha256_file(f) for f in sorted((self.dir / "head").iterdir())},
                "tokenizer_sha256": {f.name: sha256_file(f) for f in sorted((self.dir / "tokenizer").iterdir())},
                "aimodelc": tree_digest(self.aimodelc), "aimodelc_main_hash": mh.read_bytes().hex() if mh.exists() else None}


class Tower:
    def __init__(self, path: Path):
        self.dir = Path(path).expanduser().resolve()
        self.meta = json.loads((self.dir / "metadata.json").read_text())
        if self.meta.get("kind") != "vision-tower":
            raise SystemExit(f"{self.dir}: kind {self.meta.get('kind')!r} is not a vision tower")
        if self.meta.get("toy"):
            raise SystemExit(f"{self.dir} is the toy tower: this gate reads the model's oracle")
        self.name = self.meta["name"]
        self.dtype = self.meta["dtype"]["name"]
        g = self.meta["graph"]
        self.contract = {"inputs": g["inputs"], "outputs": g["outputs"], "states": {}}
        self.d = int(g["outputs"]["image_embeds"][0][1])
        self.table_path = self.dir / "host" / "position_embedding.safetensors"
        self.aimodelc = aot_path(self.dir, self.name)

    def record(self) -> dict:
        mh = self.aimodelc / "main.hash"
        return {"bundle": str(self.dir), "name": self.name, "dtype": self.meta["dtype"],
                "metadata_sha256": sha256_file(self.dir / "metadata.json"),
                "host_table_sha256": sha256_file(self.table_path), "aimodelc": tree_digest(self.aimodelc),
                "aimodelc_main_hash": mh.read_bytes().hex() if mh.exists() else None}


# --------------------------------------------------------------------------- worker: one process
async def maybe(x):
    return await x if inspect.isawaitable(x) else x


def dsc(d) -> list:
    return [[int(x) for x in d.shape], str(d.dtype).split(".")[-1]]


def fn_desc(fn, states: bool = True) -> dict:
    d = fn.desc
    out = {"inputs": {n: dsc(d.input_descriptor(n)) for n in d.input_names},
           "outputs": {n: dsc(d.output_descriptor(n)) for n in d.output_names}}
    out["states"] = {n: dsc(d.state_descriptor(n)) for n in d.state_names} if states else {}
    return out


def check_contract(desc: dict, contract: dict) -> list[str]:
    bad = []
    for part in ("inputs", "outputs", "states"):
        want = contract.get(part, {})
        if set(desc[part]) != set(want):
            bad.append(f"{part} names {sorted(desc[part])} != {sorted(want)}")
            continue
        for n, w in want.items():
            if [list(desc[part][n][0]), desc[part][n][1]] != [list(w[0]), w[1]]:
                bad.append(f"{part} {n}: {desc[part][n]} != {w}")
    return bad


def worker(spec_path: Path) -> int:
    import coreai.runtime as rt
    from safetensors.numpy import load_file

    spec = json.loads(spec_path.read_text())
    S, max_ctx, H, N, pad = (int(spec[k]) for k in ("S", "max_ctx", "hidden", "n_image_rows", "pad_id"))
    keep_full = {tuple(x) for x in spec.get("keep_full", [])}
    table = load_file(spec["tower_table"])["position_embedding"]
    tower_in = {k: np.dtype(v[1]) for k, v in spec["tower_contract"]["inputs"].items()}

    def nd(a):
        return rt.NDArray(np.ascontiguousarray(a))

    out: dict = {"pid": os.getpid(), "runs": [], "records": {}, "started": time.time()}
    store: dict[str, np.ndarray] = {}

    async def go() -> None:
        t0 = time.perf_counter()
        dm = await maybe(rt.AIModel.load(spec["decoder_aimodelc"], rt.SpecializationOptions.default()))
        fn = await maybe(dm.load_function("main"))
        t1 = time.perf_counter()
        tm = await maybe(rt.AIModel.load(spec["tower_aimodelc"], rt.SpecializationOptions.default()))
        tfn = await maybe(tm.load_function("main"))
        t2 = time.perf_counter()
        out["load_seconds"] = {"decoder": t1 - t0, "tower": t2 - t1}
        desc, tdesc = fn_desc(fn), fn_desc(tfn, states=False)
        out["descriptor"], out["tower_descriptor"] = desc, tdesc
        bad = check_contract(desc, spec["decoder_contract"]) + check_contract(tdesc, spec["tower_contract"])
        if bad:
            raise SystemExit(f"a descriptor differs from its bundle's contract: {bad}")
        zero = nd(np.zeros((N, H), np.float16))

        def fresh_state() -> dict:
            return {n: nd(np.zeros([max_ctx if s < 0 else s for s in shape], np.dtype(dt)))
                    for n, (shape, dt) in desc["states"].items()}

        towers: dict[str, dict] = {}

        async def tower_rows(rec: dict) -> dict:
            if rec["id"] in towers:
                return towers[rec["id"]]
            crops, plans = [], []
            for pi, pic in enumerate(rec["pictures"]):
                rgb = vh.cap_pixels(vh.to_rgb(Path(pic)))
                p = vh.plan(*rgb.shape[:2])
                plans.append({"size": list(rgb.shape[:2]), "crops": len(p.crops)})
                for ci, u8 in enumerate(vh.crop_images(rgb, p)):
                    ti = vh.tower_inputs(u8, table)
                    feeds = {k: nd(ti[k].astype(tower_in[k])) for k in INPUTS}
                    tc = time.perf_counter()
                    res = await maybe(tfn(inputs=feeds))
                    o = np.asarray(res["image_embeds"].numpy()).copy()
                    ms = (time.perf_counter() - tc) * 1e3
                    n = int(ti["n_tokens"])
                    key = f"tower__{rec['id']}__{pi}_{ci:02d}"
                    store[key] = o[:n].astype(np.float32)
                    crops.append({"picture": pi, "crop": ci, "grid": [int(x) for x in ti["grid"]], "n_tokens": n,
                                  "ms": ms, "out_dtype": str(o.dtype), "out_sha256": sha256_array(o),
                                  "padding_rows_finite": bool(np.isfinite(o[n:].astype(np.float32)).all()),
                                  "inputs_sha256": {k: sha256_array(ti[k]) for k in INPUTS}, "npz_key": key,
                                  "_rows": o[:n].astype(np.float16)})
            towers[rec["id"]] = {"crops": crops, "plans": plans}
            out["records"][rec["id"]] = {"plans": plans, "crops": [{k: v for k, v in c.items() if k != "_rows"}
                                                                   for c in crops]}
            return towers[rec["id"]]

        buffers: dict[tuple[str, str], object] = {}

        async def image_buffer(rec: dict, arm: str):
            if arm == "zero":
                return zero, 0
            if (rec["id"], arm) in buffers:
                return buffers[(rec["id"], arm)]
            if arm == "hf_rows":
                rows = [np.load(p)["image_embeds"].astype(np.float16) for p in rec["hf_npz"]]
            else:
                crops = (await tower_rows(rec))["crops"]
                rows = [c["_rows"] for c in (crops[::-1] if arm == "reversed" else crops)]
            emb = np.concatenate(rows) if rows else np.zeros((0, H), np.float16)
            if len(emb) > N:
                raise SystemExit(f"{rec['id']}: {len(emb)} image rows over the decoder's {N}")
            a = np.zeros((N, H), np.float16)
            a[:len(emb)] = emb
            buffers[(rec["id"], arm)] = (nd(a), len(emb))
            store[f"embeds__{rec['id']}__{arm}"] = a[:len(emb)].copy()
            return buffers[(rec["id"], arm)]

        async def one(run: dict) -> tuple[dict, np.ndarray, np.ndarray]:
            rec = spec["records"][run["record"]]
            img, n_rows = await image_buffer(rec, run["arm"])
            ids = np.asarray(run["ids"], np.int32)
            T = len(ids)
            n = -(-T // S)
            if n * S > max_ctx - 1:
                raise SystemExit(f"{run['record']}/{run['name']}: {n * S} padded positions > {max_ctx - 1}")
            x = np.full(n * S, pad, np.int32)
            x[:T] = ids
            state = fresh_state()
            hid = np.zeros((n * S, H), np.float16)
            ms = np.zeros(n, np.float64)
            tr = time.perf_counter()
            for c in range(n):
                tc = time.perf_counter()
                res = await maybe(fn(inputs={"input_ids": nd(x[c * S:(c + 1) * S].reshape(1, S)),
                                             "position_ids": nd(np.arange((c + 1) * S, dtype=np.int32)[None]),
                                             "image_embeds": img}, state=state))
                h = np.asarray(res["hidden"].numpy())
                if h.shape != (1, S, H) or h.dtype != np.float16:
                    raise SystemExit(f"output {h.shape} {h.dtype} != (1, {S}, {H}) float16")
                hid[c * S:(c + 1) * S] = h[0]
                ms[c] = (time.perf_counter() - tc) * 1e3
            hid = hid[:T].copy()
            return ({"record": run["record"], "name": run["name"], "arm": run["arm"], "tokens": T, "calls": n,
                     "padded_tokens": n * S, "image_rows": n_rows, "wall_seconds": time.perf_counter() - tr,
                     "hidden_sha256": sha256_array(hid), "finite": bool(np.isfinite(hid.astype(np.float32)).all()),
                     "all_zero": bool(not np.any(hid))}, hid, ms)

        runs = spec["runs"]
        first = None
        for i, run in enumerate(runs + [runs[0]]):
            rec, hid, ms = await one(run)
            if i < len(runs):
                key = f"{i:03d}"
                store[f"{key}__slot"] = hid[run["slot"]].copy()
                store[f"{key}__call_ms"] = ms
                if (run["record"], run["name"], run["arm"]) in keep_full:
                    store[f"{key}__hidden"] = hid
                rec["index"] = i
                out["runs"].append(rec)
                if i == 0:
                    first = hid
                print(f"  [{os.getpid()}] {run['record']}/{run['name']}/{run['arm']}: {rec['tokens']} tok, {rec['calls']} "
                      f"calls, {rec['image_rows']} image rows, {rec['wall_seconds']:.2f} s", flush=True)
            else:
                same = bool(np.array_equal(first, hid))
                out["reset_check"] = {"run": f"{run['record']}/{run['name']}/{run['arm']}", "bit_equal": same,
                                      "hidden_max_abs_diff": float(np.max(np.abs(first.astype(np.float32)
                                                                                 - hid.astype(np.float32))))}
                print(f"  [{os.getpid()}] reset re-run {out['reset_check']['run']}: bit-equal {same}", flush=True)

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
            and "readout_gate_vision.py worker" not in ln]


def gpu_lock_state() -> dict:
    p = gpu_lock()
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {"path": str(p), "exists": True, "bytes": st.st_size, "content": p.read_text()[:300],
            "mtime": datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds")}


def env_record() -> dict:
    import importlib.metadata as md

    v = {"python": sys.version.split()[0], "numpy": np.__version__}
    for p in ("coreai-core", "coreai-torch", "coreai-models", "torch", "safetensors", "tokenizers", "pillow"):
        try:
            v[p] = md.version(p)
        except Exception as e:  # noqa: BLE001
            v[p] = repr(e)
    return {"versions": v, "platform": platform.platform(),
            "macos_build": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
            "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
            "runtime": "coreai python runtime, AOT h16c GPU .aimodelc (tower and decoder), SpecializationOptions.default(), "
                       "no JIT",
            "readout": "host.py: z = h_slot . E[id] in float64 over the group ids (head/option_rows, fp32), group max, "
                       "softmax over the options",
            "gpu": "shared with other sessions, _GPU_LOCK read only (the ms are contended reference values)",
            "loadavg": os.getloadavg()}


def build_rows(rec: dict, orec: dict, tok, table_ids: list[int], N: int, S: int, max_ctx: int,
               base: Path) -> tuple[list[dict], dict]:
    """The host's rows of one picture record (module docstring) and the checks against the oracle's; `base` = the
    records file's directory (the pictures' paths are relative to it)."""
    req = host.validate_request(rec["request"])
    pics = [base / x if not Path(x).is_absolute() else Path(x) for x in rec["images"]]
    plans = [vh.plan(*vh.cap_pixels(vh.to_rgb(p)).shape[:2]) for p in pics]
    n_img = vh.n_image_tokens(plans)
    if n_img > N:
        raise SystemExit(f"{rec['id']}: images: {n_img} image tokens over the graph's {N} image rows")
    prefix = vh.image_prefix_text(req["state"], len(plans))
    oq = {q["name"]: q for q in orec["questions"]}
    rows, checks = [], {"ids_equal": [], "groups_equal": [], "keys_equal": []}
    for k, (name, q) in enumerate(req["questions"]):
        b = host.build_question(tok, req["state"], name, q)
        ids = vh.prompt_ids(tok, prefix + b["suffix"], plans)
        o = oq[name]
        checks["ids_equal"].append(ids == o["row_ids"])
        checks["groups_equal"].append(b["groups"] == o["groups"])
        checks["keys_equal"].append(b["keys"] == o["keys"])
        ext = vh.extension_ids(ids)
        host.graph_context_check(len(ext), S, max_ctx)
        rows.append({"record": rec["id"], "name": name, "k": k, "type": q["type"], "ids_host": ids, "ids": ext,
                     "slot": len(ext) - 1, "groups": b["groups"], "keys": b["keys"],
                     "image_slots": [i for i, t in enumerate(ext) if t >= vh.V]})
    host.option_table_check([{"name": r["name"], "keys": r["keys"], "groups": r["groups"]} for r in rows], table_ids)
    return rows, {"pictures": [str(p) for p in pics], "n_image_tokens": n_img,
                  "crops": [len(p.crops) for p in plans], "plans": [{"rows": p.rows, "cols": p.cols,
                                                                    "grids": [list(c.grid) for c in p.crops]} for p in plans],
                  **checks}


def split_runs(runs: list[dict], per_calls: int, S: int) -> list[list[dict]]:
    """Consecutive runs, at most `per_calls` decoder calls a process (a run is never split)."""
    parts, cur, used = [], [], 0
    for r in runs:
        n = -(-len(r["ids"]) // S)
        if cur and used + n > per_calls:
            parts.append(cur)
            cur, used = [], 0
        cur.append(r)
        used += n
    if cur:
        parts.append(cur)
    return parts


def run_shards(tag: str, shards: list[dict]) -> list[dict]:
    got = []
    for sp in shards:
        spec_path = Path(sp["out"]).with_suffix(".spec.json")
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(sp) + "\n")
        print(f"[{tag}] {Path(sp['out']).name}: {len(sp['runs'])} runs + reset re-run", flush=True)
        t0 = time.monotonic()
        proc = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--spec", str(spec_path)])
        js = Path(sp["out"]).with_suffix(".json")
        if proc.returncode != 0 or not js.exists():
            raise SystemExit(f"{Path(sp['out']).name}: worker failed (exit {proc.returncode}); see its output above")
        rec = json.loads(js.read_text())
        rec["process_wall_seconds"] = time.monotonic() - t0
        rec["spec"] = {k: v for k, v in sp.items() if k not in ("runs", "records")}
        rec["npz"] = str(Path(sp["out"]).with_suffix(".npz"))
        got.append(rec)
    return got


def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def compare_rows(got: np.ndarray, want: np.ndarray) -> dict:
    g, w = np.asarray(got, np.float64), np.asarray(want, np.float64)
    rows = cos_rows(g, w)
    i = int(rows.argmin())
    return {"cos": float(g.ravel() @ w.ravel() / (np.linalg.norm(g) * np.linalg.norm(w))), "min_row": float(rows[i]),
            "min_row_index": i, "max_abs": float(np.abs(g - w).max()), "finite": bool(np.isfinite(g).all())}


def score(p: list[float], probs_oracle: list[float], keys: list[str]) -> dict:
    p, po = np.asarray(p, np.float64), np.asarray(probs_oracle, np.float64)
    dp = np.abs(p - po)
    srt = np.sort(po)[::-1]
    margin = float(srt[0] - srt[1]) if po.size > 1 else 1.0
    return {"argmax": keys[int(p.argmax())], "argmax_oracle": keys[int(po.argmax())],
            "argmax_equal": int(p.argmax()) == int(po.argmax()), "max_abs_dp": float(dp.max()),
            "mean_abs_dp": float(dp.mean()), "oracle_top2_margin": margin, "near_tie": margin <= BAR["near_tie_top2_margin"],
            "probs": [float(x) for x in p], "probs_oracle": [float(x) for x in po]}


def summarize(scored: list[dict], form: str) -> dict:
    """readout_gate.summarize's numbers over one oracle form's scores."""
    sc = [r[form] for r in scored]
    far = [s for s in sc if not s["near_tie"]]
    near = [s for s in sc if s["near_tie"]]
    worst = max(scored, key=lambda r: r[form]["max_abs_dp"]) if scored else None
    return {"runs": len(sc), "argmax_equal": sum(s["argmax_equal"] for s in sc),
            "questions_non_near_tie": len(far), "argmax_equal_non_near_tie": sum(s["argmax_equal"] for s in far),
            "near_tie_questions": len(near), "argmax_equal_near_tie": sum(s["argmax_equal"] for s in near),
            "max_abs_dp": max((s["max_abs_dp"] for s in sc), default=None),
            "mean_of_run_mean_abs_dp": float(np.mean([s["mean_abs_dp"] for s in sc])) if sc else None,
            "worst_run": None if worst is None else {"record": worst["record"], "name": worst["name"],
                                                     "max_abs_dp": worst[form]["max_abs_dp"]}}


def verdict_abc(s: dict) -> dict:
    return {"a_argmax_non_near_tie": s["argmax_equal_non_near_tie"] == s["questions_non_near_tie"],
            "b_max_abs_dp": s["max_abs_dp"] is not None and s["max_abs_dp"] <= BAR["max_abs_dp"],
            "c_mean_of_run_mean_abs_dp": (s["mean_of_run_mean_abs_dp"] is not None
                                          and s["mean_of_run_mean_abs_dp"] <= BAR["mean_of_run_mean_abs_dp"])}


def position_cos(hid16: np.ndarray, ref: np.ndarray, image_slots: list[int], ids: list[int]) -> dict:
    c = cos_rows(hid16.astype(np.float32), ref)
    d = np.abs(hid16.astype(np.float64) - ref.astype(np.float64))
    img = np.zeros(len(c), bool)
    img[image_slots] = True
    return {"positions": int(c.size), "min_pos_cos": float(c.min()), "min_pos_cos_index": int(c.argmin()),
            "mean_pos_cos": float(c.mean()), "image_positions_min_cos": float(c[img].min()) if img.any() else None,
            "text_positions_min_cos": float(c[~img].min()), "slot_cos": float(c[-1]), "max_abs_diff": float(d.max()),
            "ref_absmax": float(np.abs(ref).max()), "positions_below_0.9999": int((c < 0.9999).sum()),
            "lowest_positions": [{"pos": int(i), "cos": float(c[i]), "id": int(ids[i]), "image": bool(img[i])}
                                 for i in np.argsort(c)[:LOWEST_POSITIONS]]}


def gate(args) -> int:
    import export_option_rows as eor

    dec, tow = Decoder(Path(args.decoder)), Tower(Path(args.tower))
    for a in (dec.aimodelc, tow.aimodelc):
        if not a.exists():
            raise SystemExit(f"no AOT asset {a}")
    if tow.d != dec.hidden:
        raise SystemExit(f"the tower's width {tow.d} != the decoder's {dec.hidden}")
    out = Path(args.transcript)
    if out.exists():
        raise SystemExit(f"{out} exists: transcripts are never overwritten")
    arms = [a for a in args.arms.split(",") if a]
    if arms[0] != "e2e" or any(a not in ARMS for a in arms):
        raise SystemExit(f"--arms: e2e first, then any of {ARMS[1:]}")
    oracle_path, records_path = Path(args.oracle).resolve(), Path(args.records).resolve()
    odoc = json.loads(oracle_path.read_text())
    if odoc.get("schema") != "d1-oracle-images/1":
        raise SystemExit(f"{oracle_path}: schema {odoc.get('schema')!r} is not oracle_d1.py --images'")
    if odoc["fixtures"]["sha256"] != sha256_file(records_path):
        raise SystemExit("the oracle was made on another image_records.json")
    orecs = {r["id"]: r for r in odoc["records"]}
    tdoc = json.loads(Path(args.tower_oracle).read_text())
    hf = {p["id"]: p for p in tdoc["pictures"]}
    from safetensors.numpy import load_file
    tower_table = np.ascontiguousarray(load_file(str(tow.table_path))["position_embedding"])
    if sha256_array(tower_table) != tdoc["position_table"]["sha256"]:
        raise SystemExit("the tower bundle's position table differs from the HF tower oracle's")
    table = eor.read_table(dec.dir)
    tok = host.load_tokenizer(dec.dir / "tokenizer" / "tokenizer.json")
    fx = json.loads(records_path.read_text())["records"]
    skipped = [{"id": r["id"], "images": r["images"], "why": "not in the oracle (a URL record: not fetched)"}
               for r in fx if r["id"] not in orecs]
    recs, rows, rec_checks = {}, [], {}
    for r in fx:
        if r["id"] not in orecs:
            continue
        rr, chk = build_rows(r, orecs[r["id"]], tok, list(table), dec.N, dec.S, dec.max_ctx, records_path.parent)
        rec_checks[r["id"]] = chk
        pics = chk["pictures"]
        if len(pics) != 1:
            raise SystemExit(f"{r['id']}: {len(pics)} pictures (the HF tower oracle holds one picture per record)")
        hf_npz = [c["npz"] for c in hf[r["id"]]["crops"]]
        if len(hf_npz) != chk["crops"][0]:
            raise SystemExit(f"{r['id']}: the HF tower oracle has {len(hf_npz)} crops, the host plans {chk['crops'][0]}")
        recs[r["id"]] = {"id": r["id"], "pictures": pics, "hf_npz": hf_npz}
        rows += rr
    multi = {rid for rid, c in rec_checks.items() if sum(c["crops"]) > 1}
    runs = []
    for arm in arms:
        for r in rows:
            if arm == "reversed" and r["record"] not in multi:
                continue
            runs.append({"record": r["record"], "name": r["name"], "arm": arm, "ids": r["ids"], "slot": r["slot"]})
    runs.sort(key=lambda x: (list(recs).index(x["record"]), arms.index(x["arm"])))   # a record's arms together
    hidden_dir = Path(odoc["hidden_records"]["dir"])
    keep_full = [[r["record"], r["name"], "e2e"] for r in rows if (hidden_dir / f"{r['record']}.npz").exists()]
    tag = args.tag or f"{tow.name}+{dec.name}"
    work = WORK / tag
    work.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    skeleton = {"schema": "d1-vision-readout-gate/1", "status": "running", "started": now(), "bar": BAR,
                "red_rule": RED_RULE, "arms": arms, "tower": tow.name, "decoder": dec.name,
                "oracle": {"path": str(oracle_path), "sha256": sha256_file(oracle_path)},
                "rows": len(rows), "runs": len(runs)}
    out.write_text(json.dumps(skeleton, indent=1) + "\n")
    base = {"decoder_aimodelc": str(dec.aimodelc), "tower_aimodelc": str(tow.aimodelc), "tower_table": str(tow.table_path),
            "S": dec.S, "max_ctx": dec.max_ctx, "hidden": dec.hidden, "n_image_rows": dec.N, "pad_id": host.PAD_ID,
            "decoder_contract": dec.contract, "tower_contract": tow.contract, "keep_full": keep_full}
    parts = split_runs(runs, args.calls_per_process, dec.S)
    shards = [{**base, "runs": part, "records": {rid: recs[rid] for rid in {x["record"] for x in part}},
               "out": str(work / f"shard_{n:02d}")} for n, part in enumerate(parts)]
    t0 = time.monotonic()
    others_start, lock_start = other_gpu_processes(), gpu_lock_state()
    shard_recs = run_shards(f"e2e {tag}", shards)
    gpu_seconds = time.monotonic() - t0
    others_end, lock_end = other_gpu_processes(), gpu_lock_state()

    # ---- collect
    by_key = {(r["record"], r["name"]): r for r in rows}
    scored, procs, tower_seen = [], [], {}
    for sr in shard_recs:
        z = np.load(sr["npz"])
        procs.append({"shard": Path(sr["npz"]).stem, "pid": sr["pid"], "runs": len(sr["runs"]),
                      "load_seconds": sr["load_seconds"], "process_wall_seconds": sr["process_wall_seconds"],
                      "reset_check": sr["reset_check"], "descriptor": sr["descriptor"],
                      "tower_descriptor": sr["tower_descriptor"], "npz": sr["npz"], "npz_sha256": sha256_file(Path(sr["npz"]))})
        for rid, trec in sr["records"].items():
            tower_seen.setdefault(rid, []).append({"shard": Path(sr["npz"]).stem, "crops": trec["crops"],
                                                  "rows": [z[c["npz_key"]] for c in trec["crops"]]})
        for run in sr["runs"]:
            key = f"{run['index']:03d}"
            r = by_key[(run["record"], run["name"])]
            o = orecs[run["record"]]
            oq = next(q for q in o["questions"] if q["name"] == run["name"])
            k = [q["name"] for q in o["questions"]].index(run["name"])
            p = host.readout(z[f"{key}__slot"].astype(np.float32),
                             eor.rows_for(table, host.group_ids(r["groups"])), host.group_ids(r["groups"]), r["groups"])
            item = {**run, "shard": Path(sr["npz"]).stem, "npz_key": key, "type": r["type"], "keys": r["keys"],
                    "row": score(p, oq["probs"], r["keys"]), "api": score(p, o["api"]["probs"][k], r["keys"]),
                    "oracle_path": o["api"]["path"], "call_ms": z[f"{key}__call_ms"].tolist(),
                    "first_in_process": run["index"] == 0}
            if f"{key}__hidden" in z.files:
                ref = np.load(hidden_dir / f"{run['record']}.npz")[f"q{k}_hidden"]
                item["hidden"] = position_cos(z[f"{key}__hidden"], ref, r["image_slots"], r["ids"])
            scored.append(item)

    # ---- per arm
    expected = {arm: sum(1 for r in rows if not (arm == "reversed" and r["record"] not in multi)) for arm in arms}
    by_arm = {}
    for arm in arms:
        sc = [x for x in scored if x["arm"] == arm]
        by_arm[arm] = {"runs": len(sc), "expected_runs": expected[arm], "row": summarize(sc, "row"),
                       "api": summarize(sc, "api"), "finite_all": all(x["finite"] for x in sc),
                       "all_zero_runs": sum(x["all_zero"] for x in sc)}
        by_arm[arm]["row_bar"] = verdict_abc(by_arm[arm]["row"])
        by_arm[arm]["api_bar"] = verdict_abc(by_arm[arm]["api"])
        by_arm[arm]["rows_moved_over_bar"] = [f"{x['record']}/{x['name']}" for x in sc
                                              if x["row"]["max_abs_dp"] > BAR["max_abs_dp"] or not x["row"]["argmax_equal"]]
    e = by_arm["e2e"]
    resets = all(p["reset_check"]["bit_equal"] for p in procs)
    ids_ok = all(all(c["ids_equal"]) and all(c["groups_equal"]) and all(c["keys_equal"]) for c in rec_checks.values())
    checks = {"all_runs": all(by_arm[a]["runs"] == by_arm[a]["expected_runs"] for a in arms),
              "ids_groups_keys_equal_oracle": ids_ok,
              **{f"row_{k}": v for k, v in e["row_bar"].items()}, **{f"api_{k}": v for k, v in e["api_bar"].items()},
              "d_reset_bit_equal_all_processes": resets,
              "e_finite": e["finite_all"] and e["all_zero_runs"] == 0}
    if "zero" in by_arm:
        by_arm["zero"]["red"] = not all(by_arm["zero"]["row_bar"].values())
        checks["zero_arm_red"] = by_arm["zero"]["red"]
    ok = all(checks.values())
    # ---- the tower against the HF tower oracle, and across processes
    tower_rows = []
    for rid, seen in tower_seen.items():
        first = seen[0]
        for c, got in zip(first["crops"], first["rows"]):
            hc = hf[rid]["crops"][c["crop"]]
            with np.load(hc["npz"]) as zz:
                want = zz["image_embeds"]
                inputs_equal = all(c["inputs_sha256"][k] == sha256_array(zz[k]) for k in INPUTS)
            tower_rows.append({"record": rid, "crop": hc["crop"], "kind": hc["kind"], "grid": c["grid"],
                               "n_tokens": c["n_tokens"], **compare_rows(got, want), "inputs_equal_oracle_npz": inputs_equal,
                               "padding_rows_finite": c["padding_rows_finite"], "ms": c["ms"], "out_sha256": c["out_sha256"],
                               "same_in_every_process": all(s["crops"][i]["out_sha256"] == c["out_sha256"]
                                                            for s in seen for i in [first["crops"].index(c)])})
    hid = [x for x in scored if "hidden" in x]
    timing = {"contended": True, "decoder_ms_per_call": {
        "median": float(np.median(np.concatenate([x["call_ms"] for x in scored]))),
        "p10": float(np.quantile(np.concatenate([x["call_ms"] for x in scored]), 0.1)),
        "p90": float(np.quantile(np.concatenate([x["call_ms"] for x in scored]), 0.9))},
        "tower_ms_per_crop": {"median": float(np.median([t["ms"] for t in tower_rows])),
                              "max": float(max(t["ms"] for t in tower_rows))},
        "load_seconds": [p["load_seconds"] for p in procs], "gpu_processes_seconds": gpu_seconds}
    record = {
        "schema": "d1-vision-readout-gate/1",
        "gate": "end to end on the Mac GPU: picture file -> vision_host -> tower AOT -> image_embeds fp16 -> decoder AOT "
                "-> host.readout, against the provider's fp32 oracle (oracle_d1.py --images), row form and API path",
        "bar": BAR, "red_rule": RED_RULE, "result": "PASS" if ok else "FAIL", "checks": checks, "arms": arms,
        "decoder": dec.record(), "tower": tow.record(),
        "oracle": {"path": str(oracle_path), "sha256": sha256_file(oracle_path), "schema": odoc["schema"],
                   "fixtures_sha256": odoc["fixtures"]["sha256"], "hidden_dir": str(hidden_dir)},
        "tower_oracle": {"path": str(Path(args.tower_oracle).resolve()), "sha256": sha256_file(Path(args.tower_oracle)),
                         "position_table_sha256": tdoc["position_table"]["sha256"]},
        "records": {"path": str(records_path), "sha256": sha256_file(records_path), "skipped": skipped,
                    "checks": rec_checks},
        "script": {"path": "conversion/d1/readout_gate_vision.py", "sha256": sha256_file(Path(__file__).resolve())},
        "environment": env_record(), "gpu_lock": {"start": lock_start, "end": lock_end},
        "other_gpu_processes": {"start": others_start, "end": others_end},
        "processes": procs, "by_arm": by_arm,
        "summary": {"e2e_row": e["row"], "e2e_api": e["api"],
                    "hidden_rows": len(hid), "min_pos_cos": min((x["hidden"]["min_pos_cos"] for x in hid), default=None),
                    "min_image_pos_cos": min((x["hidden"]["image_positions_min_cos"] for x in hid), default=None),
                    "min_text_pos_cos": min((x["hidden"]["text_positions_min_cos"] for x in hid), default=None),
                    "min_slot_cos": min((x["hidden"]["slot_cos"] for x in hid), default=None),
                    "tower_vs_hf": {"crops": len(tower_rows), "min_cos": min(t["cos"] for t in tower_rows),
                                    "min_row": min(t["min_row"] for t in tower_rows),
                                    "max_abs": max(t["max_abs"] for t in tower_rows),
                                    "inputs_equal_oracle_npz": sum(t["inputs_equal_oracle_npz"] for t in tower_rows),
                                    "same_in_every_process": all(t["same_in_every_process"] for t in tower_rows)}},
        "tower_rows": tower_rows, "timing": timing,
        "runs": [{k: v for k, v in x.items() if k not in ("ids",)} for x in scored],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if args.note:
        record["note"] = args.note
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"{record['result']}: {tow.name} -> {dec.name}, e2e rows {e['runs']}/{e['expected_runs']}")
    for form in ("row", "api"):
        s = e[form]
        print(f"  vs oracle {form}: argmax {s['argmax_equal']}/{s['runs']} (non-near-tie {s['argmax_equal_non_near_tie']}/"
              f"{s['questions_non_near_tie']}) max|dp| {s['max_abs_dp']:.3e} mean {s['mean_of_run_mean_abs_dp']:.3e} "
              f"worst {s['worst_run']}")
    def num(v) -> str:
        return "-" if v is None else f"{v:.3e}"

    for arm in arms[1:]:
        s = by_arm[arm]["row"]
        print(f"  arm {arm}: rows {by_arm[arm]['runs']}, argmax {s['argmax_equal']}/{s['runs']} max|dp| {num(s['max_abs_dp'])} "
              f"mean {num(s['mean_of_run_mean_abs_dp'])}" + (f" -> {'RED' if by_arm[arm]['red'] else 'NOT RED'}"
                                                             if arm == "zero" else ""))
    tv = record["summary"]["tower_vs_hf"]
    print(f"  tower vs HF: {tv['crops']} crops min cos {tv['min_cos']:.9f} min row {tv['min_row']:.9f} max|d| "
          f"{tv['max_abs']:.3e}; hidden rows {len(hid)} min pos cos {record['summary']['min_pos_cos']}")
    for k, v in checks.items():
        print(f"  {k}: {'ok' if v else 'FAIL'}")
    print(f"  decoder ms/call (contended) median {timing['decoder_ms_per_call']['median']:.2f}, tower ms/crop median "
          f"{timing['tower_ms_per_crop']['median']:.1f}; GPU processes {gpu_seconds:.0f} s")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="the gate: workers on the GPU, then the host's readout against the oracle")
    r.add_argument("--tower", required=True, help="the tower bundle (export_vision.py)")
    r.add_argument("--decoder", required=True, help="the decoder bundle (export_decoder.py)")
    r.add_argument("--records", default=str(RECORDS))
    r.add_argument("--oracle", default=str(ORACLE), help="oracle_d1.py --images' records_oracle_images.json")
    r.add_argument("--tower-oracle", default=str(TOWER_ORACLE), help="vision_oracle.py's oracle.json")
    r.add_argument("--arms", default=",".join(ARMS))
    r.add_argument("--calls-per-process", type=int, default=CALLS_PER_PROCESS)
    r.add_argument("--tag", help="shard directory name under readout_vision/ (default <tower>+<decoder>)")
    r.add_argument("--transcript", required=True)
    r.add_argument("--note")
    w = sub.add_parser("worker")
    w.add_argument("--spec", required=True)
    args = ap.parse_args()
    if args.cmd == "worker":
        return worker(Path(args.spec))
    return gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
