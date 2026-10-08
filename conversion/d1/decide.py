#!/usr/bin/env python3
"""d1 Python reference: a System One request -> the System One response, on the Core AI graphs (Mac GPU, AOT only).

The whole read-out, in the order a Swift host repeats it (`host.py` and `vision_host.py` are the specification of every
host step; `apps/D1` is the Swift copy and `gate_swift.py` compares the two):

    request JSON --host.build_request--> one row per question (host.py §1-4): ids, slot, readout groups, keys, usage
      pictures (`images`): vision_host (to_rgb, cap_pixels, plan, crop_images, tower_inputs with the tower bundle's
      position table) -> the tower bundle's AOT asset per crop -> the crop's first h w / 4 rows, crops in order,
      pictures in text order -> image_embeds [N, d] fp16 (rows after zero; one buffer bound to every call of the
      request); each question's row = vision_host.prompt_ids(prefix with one "<image>" per picture + its suffix)
      with <image> -> V + k (vision_host.extension_ids)
    host.graph_context_check: every row's padded end <= max_context_length - 1
    graph (the bundle's AOT `.aimodelc`, function `main`, SpecializationOptions.default(); never the JIT):
        direct   every row from fresh zero states in ceil(T / S) calls of S ids (call c: position_ids 0..p+cS+S-1;
                 a static-form bundle, metadata `language.contract.static`: p+cS..p+cS+S-1), the last padded with
                 <|pad|>; the padded positions' rows dropped -> hidden [T, d] fp16
        shared   (--shared) the state's first k = floor(Ls / S) * S row tokens once from zero states, the three states
                 read back once and every question's remaining tokens run from a fresh copy of them at positions k..:
                 the direct run's calls on a static-S graph, so its hidden rows bit for bit
        Ls       the state's stable tokens: the prefix's ids (the prefix the rows start with: BOS + user turn + pictures
                 + state block + "\\nQUESTION:\\n") minus the ids of its last pre-token, the only piece a question's text
                 can change (":\\n" + "\\n" -> ":\\n\\n"); `stable_prefix`. Every row is checked to start with them.
        prepared `prepare(state)` runs those k tokens once and keeps the states; `decide_prepared(prepared, questions)`
                 answers later questions on a copy of them: the shared path split in two (the same calls, the same bits)
    readout  host.readout on the bundle's head/option_rows (float64, NumPy's BLAS, the group max, Python's sum)
    response host.response (the answers in request order, usage) -> json.dumps(response, indent=2, ensure_ascii=False)

A toy bundle (`export_decoder.py --toy`, metadata `toy`) has a 256-id vocabulary: every real id is folded into it the
way round 2a's gate folded it (`export_decoder.toy_fold`: id % V; an extension id 128,000 + k -> V + k, round 3b's
e2e), the pad is metadata `toy.pad_folded`, and the readout reads the toy table with the groups `--toy-oracle` gives per
(record, question) (round 2a's random toy groups), else the real groups folded the same way.

    cd conversion/d1
    PY=<coreai-models venv>/bin/python Q="$HOME/code/standup/tools/quiet/quiet_wait.py --max-wait 3600 --"
    $Q $PY decide.py run --bundle <bundle> --request req.json [--tower <tower bundle>] [--shared] --out resp.json \\
        [--trace trace.json] [--toy-oracle <records_oracle.json> --record-id <id>]
    $Q $PY decide.py check --bundle $K/exports/toy_bundles/d1_toy_decode_fp16_pf16_tbl2 \\
        --toy-oracle $K/oracle_toy_tbl2/records_oracle.json --gate $K/results/r2a_toy_readout_fp16_tbl2.json \\
        --out $K/results/r3c_py_check_fp16.json [--shared-out $K/results/r3c_py_shared_vs_direct.json]
    $Q $PY decide.py e2e --tower $K/exports/toy_vision/d1_toy_vision_fp32 \\
        --bundle $K/exports/toy_bundles/d1_toy_decode_fp16_n2816_pf16 --out $K/results/r3c_py_e2e.json

`check` answers every fixture record from its raw request (direct; shared and prepared for every accepted request;
a refused request's text, and each of its questions alone as a one-question request), at most 40 records a process and
a re-run of the process's first record at its end (the Python runtime leaks an IOSurface per call), and compares with
the readout gate's transcript of the same asset: every row's hidden rows (sha256 of the fp16 [T, d] bytes) and p (bits).
`e2e` runs round 3b's three picture rows (the first question of img01 / img06 / img12) through the tower and the
decoder, records every array a Swift run is compared with, and recomputes round 3b's comparison with the torch fp32
eager composite (its numbers must come back unchanged).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import re
import resource
import struct
import subprocess
import sys
import time
from datetime import datetime
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
FIXTURES = LANE / "fixtures" / "records.json"
IMAGE_FIXTURES = LANE / "fixtures" / "image_records.json"
WORK = LANE / "decide"
RECORDS_PER_PROCESS = 40
E2E_RECORDS = ("img01_shapes_384x384", "img06_grid_1024x768", "img12_small_300x300")   # round 3b's
E2E_FIELDS = ("answer_slot_cos", "answer_slot_max_abs", "answer_slot_rel_change", "every_position_cos_min",
              "every_position_cos_argmin", "image_positions_cos_min", "text_positions_cos_min", "max_abs")
OTHER_GPU = re.compile(r"yardstick|litert|llm-bench|coreai_verify|coreai-build|readout_gate|gate_|parity_|mlx|"
                       r"timing\.py|--accel gpu|/d1 (decide|fixture|prepare-test)")


async def maybe(x):
    return await x if inspect.isawaitable(x) else x


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def bits(x: float) -> str:
    return format(struct.unpack("<Q", struct.pack("<d", float(x)))[0], "x")


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def aot_path(bundle: Path, name: str) -> Path:
    """<bundles>/<name> -> <bundles>_aotc/<name>.h16c.aimodelc (readout_gate.Bundle's rule)."""
    return bundle.parent.parent / f"{bundle.parent.name}_aotc" / f"{name}.h16c.aimodelc"


def fn_contract(fn) -> dict:
    d = fn.desc

    def dsc(x) -> list:
        return [[int(v) for v in x.shape], str(x.dtype).split(".")[-1]]
    return {"inputs": {n: dsc(d.input_descriptor(n)) for n in d.input_names},
            "outputs": {n: dsc(d.output_descriptor(n)) for n in d.output_names},
            "states": {n: dsc(d.state_descriptor(n)) for n in d.state_names}}


def contract_mismatch(desc: dict, want: dict) -> list[str]:
    bad = []
    for part in ("inputs", "outputs", "states"):
        w = want.get(part, {})
        if set(desc[part]) != set(w):
            bad.append(f"{part} names {sorted(desc[part])} != {sorted(w)}")
            continue
        for n, spec in w.items():
            if [list(desc[part][n][0]), desc[part][n][1]] != [list(spec[0]), spec[1]]:
                bad.append(f"{part} {n}: {desc[part][n]} != {spec}")
    return bad


# --------------------------------------------------------------------------- the tower
class Tower:
    """A tower bundle (export_vision.py): its AOT asset, the position table, the contract in metadata.json."""

    def __init__(self, bundle: Path, aimodelc: Path | None = None):
        from safetensors.numpy import load_file

        self.dir = Path(bundle).expanduser().resolve()
        self.meta = json.loads((self.dir / "metadata.json").read_text())
        self.name = self.meta["name"]
        g = self.meta["graph"]
        self.contract = {"inputs": g["inputs"], "outputs": g["outputs"], "states": {}}
        self.d = int(g["outputs"]["image_embeds"][0][1])
        self.in_dtype = {k: np.dtype(v[1]) for k, v in g["inputs"].items()}
        self.table = load_file(str(self.dir / "host" / "position_embedding.safetensors"))["position_embedding"]
        self.aimodelc = Path(aimodelc).expanduser().resolve() if aimodelc else aot_path(self.dir, self.name)
        if not self.aimodelc.exists():
            raise SystemExit(f"no AOT asset {self.aimodelc}")

    async def load(self) -> dict:
        import coreai.runtime as rt
        self.rt = rt
        t0 = time.perf_counter()
        model = await maybe(rt.AIModel.load(str(self.aimodelc), rt.SpecializationOptions.default()))
        self.fn = await maybe(model.load_function("main"))
        self._model = model
        desc = fn_contract(self.fn)
        bad = contract_mismatch(desc, self.contract)
        if bad:
            raise SystemExit(f"tower {self.name}: the descriptor differs from metadata.json: {bad}")
        return {"seconds": time.perf_counter() - t0, "descriptor": desc}

    async def crop(self, ti: dict) -> np.ndarray:
        """One crop's four inputs (vision_host.tower_inputs) -> image_embeds [256, d] as the graph returns it."""
        feeds = {k: self.rt.NDArray(np.ascontiguousarray(ti[k].astype(self.in_dtype[k]))) for k in self.contract["inputs"]}
        res = await maybe(self.fn(inputs=feeds))
        return np.asarray(res["image_embeds"].numpy()).copy()

    def inputs(self, picture: Path) -> tuple[vh.Plan, list[dict]]:
        """A picture file -> its plan and every crop's four inputs (vision_host, the bundle's position table)."""
        rgb = vh.cap_pixels(vh.to_rgb(picture))
        p = vh.plan(*rgb.shape[:2])
        return p, [vh.tower_inputs(u8, self.table) for u8 in vh.crop_images(rgb, p)]


# --------------------------------------------------------------------------- the decoder
class D1:
    """One decoder bundle's graph, tokenizer and option table, loaded once; `decide()` answers one request."""

    def __init__(self, bundle: Path, aimodelc: Path | None = None):
        import export_option_rows as eor

        self.dir = Path(bundle).expanduser().resolve()
        self.meta = json.loads((self.dir / "metadata.json").read_text())
        lang = self.meta["language"]
        self.name = self.meta["name"]
        self.S = int(lang["prefill_chunk"])
        self.max_ctx = int(lang["max_context_length"])
        self.contract = lang["contract"]
        self.static = bool(self.contract.get("static"))   # lfm2_d1_static.py: position_ids = the call's S positions
        self.d = int(self.contract["outputs"]["hidden"][0][2])
        self.N = int(self.contract["inputs"]["image_embeds"][0][0])
        self.V = int(lang["vocab_size"])
        self.toy = self.meta.get("toy")
        special = {v["token"]: int(v["id"]) for v in self.meta["decision"]["prompt"]["special"].values()}
        if special.get(host.PAD_TOKEN) != host.PAD_ID:
            raise SystemExit(f"metadata pad {special.get(host.PAD_TOKEN)} != host.PAD_ID {host.PAD_ID}")
        if self.toy:
            if self.toy.get("fold") != f"id % {self.V}" or int(self.toy["pad_folded"]) != host.PAD_ID % self.V:
                raise SystemExit(f"toy block {self.toy.get('fold')!r} / pad {self.toy.get('pad_folded')} is not id % {self.V}")
            self.pad = int(self.toy["pad_folded"])
        else:
            if self.V != vh.V:
                raise SystemExit(f"vocab_size {self.V} != the extension base {vh.V}")
            self.pad = host.PAD_ID
        self.tok = host.load_tokenizer(self.dir / "tokenizer" / "tokenizer.json")
        self.added = sorted(self.tok.get_added_tokens_decoder().values(), key=lambda t: t.content)
        self.table = eor.read_table(self.dir)
        self.aimodelc = Path(aimodelc).expanduser().resolve() if aimodelc else aot_path(self.dir, self.name)
        if not self.aimodelc.exists():
            raise SystemExit(f"no AOT asset {self.aimodelc} (the JIT is never used here)")

    async def load(self) -> dict:
        import coreai.runtime as rt
        self.rt = rt
        t0 = time.perf_counter()
        model = await maybe(rt.AIModel.load(str(self.aimodelc), rt.SpecializationOptions.default()))
        t1 = time.perf_counter()
        self.fn = await maybe(model.load_function("main"))
        t2 = time.perf_counter()
        self._model = model
        desc = fn_contract(self.fn)
        bad = contract_mismatch(desc, self.contract)
        if bad:
            raise SystemExit(f"decoder {self.name}: the descriptor differs from metadata.json language.contract: {bad}")
        self.state_desc = desc["states"]
        self.zero_img = self.rt.NDArray(np.zeros((self.N, self.d), np.float16))
        return {"model_seconds": t1 - t0, "main_seconds": t2 - t1, "seconds": t2 - t0, "descriptor": desc}

    # ids in the graph's vocabulary ---------------------------------------------
    def graph_ids(self, ids: list[int]) -> list[int]:
        """The ids the graph reads: the toy folds every real id (id % V; 128,000 + k -> V + k)."""
        if self.toy:
            return [self.V + (t - vh.V) if t >= vh.V else t % self.V for t in ids]
        return list(ids)

    def graph_groups(self, groups: list[list[int]]) -> list[list[int]]:
        return [[i % self.V for i in g] for g in groups] if self.toy else groups

    # states -----------------------------------------------------------------------
    def zero_arrays(self) -> dict[str, np.ndarray]:
        return {n: np.zeros([self.max_ctx if s < 0 else s for s in shape], np.dtype(dt))
                for n, (shape, dt) in self.state_desc.items()}

    def to_state(self, arrays: dict[str, np.ndarray]) -> dict:
        return {n: self.rt.NDArray(np.ascontiguousarray(a)) for n, a in arrays.items()}   # the runtime copies

    @staticmethod
    def snapshot(state: dict) -> dict[str, np.ndarray]:
        return {n: np.array(v.numpy(), copy=True) for n, v in state.items()}

    def image_nd(self, rows: np.ndarray | None):
        """image_embeds [N, d] fp16: the request's image rows first, zero after; the zero buffer for a text request."""
        if rows is None or len(rows) == 0:
            return self.zero_img
        if len(rows) > self.N:
            raise ValueError(f"images: {len(rows)} image tokens over the graph's {self.N} image rows")
        a = np.zeros((self.N, self.d), np.float16)
        a[:len(rows)] = rows
        return self.rt.NDArray(a)

    # graph calls ------------------------------------------------------------------
    def positions(self, p: int) -> np.ndarray:
        """position_ids [1, ..] of a call after p earlier ids: 0..p+S-1 (dynamic form), p..p+S-1 (static form)."""
        return np.arange(p if self.static else 0, p + self.S, dtype=np.int32)[None]

    async def call(self, x: np.ndarray, p: int, state: dict, img) -> np.ndarray:
        """One call: S ids after p earlier ids (position_ids 0..p+S-1; static form p..p+S-1) -> hidden [S, d] fp16."""
        res = await maybe(self.fn(inputs={"input_ids": self.rt.NDArray(np.ascontiguousarray(x.reshape(1, self.S))),
                                          "position_ids": self.rt.NDArray(self.positions(p)),
                                          "image_embeds": img}, state=state))
        h = np.asarray(res["hidden"].numpy())
        if h.shape != (1, self.S, self.d) or h.dtype != np.float16:
            raise SystemExit(f"hidden {h.shape} {h.dtype} != (1, {self.S}, {self.d}) float16")
        return h[0]

    async def run_ids(self, ids: list[int], p0: int, state: dict, img, call_ms: list) -> np.ndarray:
        """Graph ids from position p0 on, S at a time (the last call padded) -> hidden [len(ids), d] fp16."""
        n = -(-len(ids) // self.S)
        x = np.full(n * self.S, self.pad, np.int32)
        x[:len(ids)] = ids
        out = np.zeros((n * self.S, self.d), np.float16)
        for c in range(n):
            t = time.perf_counter()
            out[c * self.S:(c + 1) * self.S] = await self.call(x[c * self.S:(c + 1) * self.S], p0 + c * self.S, state, img)
            call_ms.append((time.perf_counter() - t) * 1e3)
        return out[:len(ids)].copy()

    # the state's stable tokens ---------------------------------------------------------
    def stable_prefix(self, prefix_text: str, prefix_ids: list[int]) -> int:
        """The number of the prefix's ids every row starting with this prefix also starts with: all but the ids of the
        prefix's last pre-token (the text after its last added token, cut by the Split regex). 0 when the prefix does
        not end the way the rule reads it (no sharing: still exact)."""
        cut = max((prefix_text.rfind(t.content) + len(t.content) for t in self.added if t.content in prefix_text), default=0)
        tail = prefix_text[cut:]
        if not tail:
            return len(prefix_ids)
        pieces = self.tok.pre_tokenizer.pre_tokenize_str(tail)
        last = tail[pieces[-1][1][0]:]
        last_ids = host.token_ids(self.tok, last)
        n = len(prefix_ids) - len(last_ids)
        if n < 0 or prefix_ids[n:] != last_ids:
            return 0
        return n

    # the request -> rows ---------------------------------------------------------
    def build(self, request: dict, groups: dict | None = None, plans: list | None = None) -> dict:
        """host.build_request (with the bundle's option table for a model bundle) -> every row in the graph's ids and its
        readout groups. `groups` = {question name: groups} replaces a toy's groups (round 2a's toy oracle); `plans` =
        the pictures' plans (their "<image>" markers go into the prefix)."""
        b = host.build_request(request, self.tok, table_ids=None if self.toy else list(self.table))
        n_img = len(plans or [])
        img_prefix = vh.image_prefix_text(b["state"], n_img) if n_img else host.prefix_text(b["state"])
        rows = []
        for q in b["questions"]:
            ids = (vh.extension_ids(vh.prompt_ids(self.tok, img_prefix + q["suffix"], plans)) if n_img else q["row_ids"])
            g = groups[q["name"]] if groups is not None and q["name"] in groups else self.graph_groups(q["groups"])
            rows.append({**q, "row_ids": ids, "row_len": len(ids), "slot": len(ids) - 1, "graph_ids": self.graph_ids(ids),
                         "read_groups": g})
        host.option_table_check([{**r, "groups": r["read_groups"]} for r in rows], list(self.table))
        for r in rows:
            host.graph_context_check(r["row_len"], self.S, self.max_ctx, self.static)
        trunk = (vh.extension_ids(vh.prompt_ids(self.tok, img_prefix, plans)) if n_img
                 else host.token_ids(self.tok, img_prefix))
        stable = self.stable_prefix(img_prefix, trunk)
        if any(r["row_ids"][:stable] != trunk[:stable] for r in rows):
            stable = 0
        if n_img:
            texts = [img_prefix] if len(rows) > 1 else [img_prefix + rows[0]["suffix"]]
            ids0 = vh.prompt_ids(self.tok, texts[0], plans)
            branches = [host.token_ids(self.tok, r["suffix"]) for r in rows] if len(rows) > 1 else []
            input_tokens = len(ids0) + sum(map(len, branches))
        else:
            input_tokens = b["input_tokens"]
        return {"validated": b["validated"], "rows": rows, "path": b["path"], "shared": b["shared"],
                "input_tokens": input_tokens, "state_tokens": stable, "trunk_len": len(trunk)}

    def readout(self, h_slot: np.ndarray, groups: list[list[int]]) -> list[float]:
        import export_option_rows as eor

        ids = host.group_ids(groups)
        return host.readout(h_slot.astype(np.float32), eor.rows_for(self.table, ids), ids, groups)

    # the decision ---------------------------------------------------------------------
    async def hidden(self, rows: list[dict], img, mode: str, k: int, call_ms: list) -> list[np.ndarray]:
        if mode == "direct" or k == 0:
            return [await self.run_ids(r["graph_ids"], 0, self.to_state(self.zero_arrays()), img, call_ms) for r in rows]
        prefix = rows[0]["graph_ids"][:k]
        if any(r["graph_ids"][:k] != prefix for r in rows):
            raise SystemExit("the rows do not share their first k ids")
        state = self.to_state(self.zero_arrays())
        h_pre = await self.run_ids(prefix, 0, state, img, call_ms)
        snap = self.snapshot(state)
        return [np.concatenate([h_pre, await self.run_ids(r["graph_ids"][k:], k, self.to_state(snap), img, call_ms)])
                for r in rows]

    def answer(self, b: dict, hs: list[np.ndarray], tr: dict) -> dict:
        ps = [self.readout(h[r["slot"]], r["read_groups"]) for r, h in zip(b["rows"], hs)]
        body = host.response(b["validated"], ps, b["input_tokens"])
        tr.update({"_hidden": hs, "_probs": ps, "_rows": b})
        return body

    async def decide(self, request: dict, mode: str = "direct", groups: dict | None = None, images: dict | None = None,
                     trace: dict | None = None) -> dict:
        """request -> the response body. `images` = {"plans", "rows" (image_embeds rows fp16)} from `image_rows`."""
        tr = trace if trace is not None else {}
        t0 = time.perf_counter()
        b = self.build(request, groups, images["plans"] if images else None)
        k = (b["state_tokens"] // self.S) * self.S if mode == "shared" else 0
        img = self.image_nd(images["rows"] if images else None)
        t1 = time.perf_counter()
        call_ms: list = []
        hs = await self.hidden(b["rows"], img, mode, k, call_ms)
        t2 = time.perf_counter()
        body = self.answer(b, hs, tr)
        tr.update({"mode": mode, "S": self.S, "state_tokens": b["state_tokens"], "shared_k": k,
                   "row_tokens": [r["row_len"] for r in b["rows"]], "input_tokens": b["input_tokens"], "calls": len(call_ms),
                   "call_ms": call_ms, "graph_ms": (t2 - t1) * 1e3, "host_ms": (t1 - t0) * 1e3,
                   "readout_ms": (time.perf_counter() - t2) * 1e3})
        return body

    async def prepare(self, state, trace: dict | None = None) -> dict:
        """The state's first k = floor(Ls / S) * S ids (text only) run once from zero states and kept."""
        t0 = time.perf_counter()
        prefix = host.prefix_text(state)
        trunk = host.token_ids(self.tok, prefix)
        stable = self.stable_prefix(prefix, trunk)
        k = (stable // self.S) * self.S
        call_ms: list = []
        st = self.to_state(self.zero_arrays())
        h = await self.run_ids(self.graph_ids(trunk[:k]), 0, st, self.zero_img, call_ms) if k else np.zeros((0, self.d), np.float16)
        out = {"state": state, "ids": trunk[:k], "k": k, "state_tokens": stable, "snapshot": self.snapshot(st), "hidden": h,
               "calls": len(call_ms), "call_ms": call_ms, "ms": (time.perf_counter() - t0) * 1e3}
        if trace is not None:
            trace.update({x: v for x, v in out.items() if x not in ("snapshot", "hidden", "state")})
        return out

    async def decide_prepared(self, prepared: dict, questions: dict, groups: dict | None = None,
                              trace: dict | None = None) -> dict:
        """{state: the prepared state, questions} on the kept states: every row's ids from k on, from a copy of them. A row
        that does not start with the prepared ids runs whole from zero states (counted in the trace)."""
        tr = trace if trace is not None else {}
        b = self.build({"state": prepared["state"], "questions": questions}, groups)
        k = prepared["k"]
        call_ms: list = []
        hs, whole = [], 0
        for r in b["rows"]:
            if r["row_ids"][:k] == prepared["ids"]:
                h = await self.run_ids(r["graph_ids"][k:], k, self.to_state(prepared["snapshot"]), self.zero_img, call_ms)
                hs.append(np.concatenate([prepared["hidden"], h]))
            else:
                whole += 1
                hs.append(await self.run_ids(r["graph_ids"], 0, self.to_state(self.zero_arrays()), self.zero_img, call_ms))
        body = self.answer(b, hs, tr)
        tr.update({"mode": "prepared", "shared_k": k, "calls": len(call_ms), "call_ms": call_ms, "rows_run_whole": whole})
        return body

    async def warm_up(self) -> float:
        t = time.perf_counter()
        await self.call(np.full(self.S, self.pad, np.int32), 0, self.to_state(self.zero_arrays()), self.zero_img)
        return (time.perf_counter() - t) * 1e3


async def image_rows(tower: Tower, pictures: list[Path], record: dict | None = None) -> dict:
    """Pictures -> {"plans", "rows" (every crop's first h w / 4 rows, fp16, in order), per-crop records}."""
    plans, rows, crops = [], [], []
    for pic in pictures:
        p, tis = tower.inputs(pic)
        plans.append(p)
        for c, ti in zip(p.crops, tis):
            out = await tower.crop(ti)
            n = ti["n_tokens"]
            rows.append(out[:n].astype(np.float16))
            crops.append({"picture": pic.name, "kind": c.kind, "grid": list(ti["grid"]), "n_tokens": n,
                          "output_sha256": sha256_bytes(out), "output_dtype": str(out.dtype),
                          "inputs_sha256": {k: sha256_bytes(ti[k]) for k in ("patches", "pos_table", "key_bias", "unshuffle_idx")},
                          "_inputs": ti, "_out": out})
    if record is not None:
        record["crops"] = crops
    return {"plans": plans, "rows": np.concatenate(rows) if rows else np.zeros((0, tower.d), np.float16), "crops": crops}


def toy_groups_of(path: Path | None) -> dict:
    """{record id: {question name: groups}} of a toy oracle (round 2a's random toy groups)."""
    if not path:
        return {}
    doc = json.loads(Path(path).read_text())
    return {r["id"]: {q["name"]: q["groups"] for q in r["questions"]} for r in doc["records"]}


def public(tr: dict) -> dict:
    return {k: v for k, v in tr.items() if not k.startswith("_")}


def row_record(r: dict, h: np.ndarray, p: list[float]) -> dict:
    return {"name": r["name"], "type": r["type"], "T": r["row_len"], "slot": r["slot"],
            "row_ids_sha256": hashlib.sha256(json.dumps(r["row_ids"]).encode()).hexdigest(),
            "graph_ids_sha256": hashlib.sha256(json.dumps(r["graph_ids"]).encode()).hexdigest(),
            "read_groups": r["read_groups"], "hidden_sha256": sha256_bytes(h),
            "slot_hidden_sha256": sha256_bytes(h[r["slot"]]),
            "finite": bool(np.isfinite(h.astype(np.float32)).all()), "all_zero": bool(not np.any(h)),
            "p": [float(x) for x in p], "p_bits": [bits(x) for x in p]}


# --------------------------------------------------------------------------- run
def cmd_run(args) -> int:
    request = json.loads(Path(args.request).read_text())
    pictures = [Path(args.request).parent / x if not Path(x).is_absolute() else Path(x) for x in request.pop("images", [])]
    e = D1(Path(args.bundle), args.aimodelc)
    tower = Tower(Path(args.tower)) if args.tower else None
    if pictures and tower is None:
        raise SystemExit("a request with images needs --tower")
    groups = toy_groups_of(args.toy_oracle).get(args.record_id) if args.toy_oracle else None
    trace: dict = {}

    async def go():
        trace["load"] = {k: v for k, v in (await e.load()).items() if k != "descriptor"}
        images = None
        if pictures:
            trace["tower_load"] = {k: v for k, v in (await tower.load()).items() if k != "descriptor"}
            images = await image_rows(tower, pictures, trace)
        try:
            return await e.decide(request, "shared" if args.shared else "direct", groups, images, trace)
        except ValueError as err:
            return {"error": str(err)}

    resp = asyncio.run(go())
    Path(args.out).write_text(json.dumps(resp, indent=2, ensure_ascii=False))
    print(json.dumps(resp, ensure_ascii=False))
    if args.trace:
        tr = public(trace)
        if "_probs" in trace:
            tr["rows"] = [row_record(r, h, p) for r, h, p in zip(trace["_rows"]["rows"], trace["_hidden"], trace["_probs"])]
        tr["crops"] = [{k: v for k, v in c.items() if not k.startswith("_")} for c in trace.get("crops", [])]
        tr.update({"bundle": str(e.dir), "aimodelc": str(e.aimodelc)})
        Path(args.trace).write_text(json.dumps(tr, indent=1) + "\n")
    return 0


# --------------------------------------------------------------------------- check: worker
def worker(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text())
    recs = {r["id"]: r for r in json.loads(Path(spec["fixtures"]).read_text())["records"]}
    e = D1(Path(spec["bundle"]), Path(spec["aimodelc"]))
    tg = toy_groups_of(spec.get("toy_oracle"))
    out = {"pid": os.getpid(), "started": time.time(), "records": []}

    async def one(rid: str) -> dict:
        request = recs[rid]["request"]
        groups = tg.get(rid)
        item: dict = {"id": rid, "source": recs[rid].get("source")}
        tr: dict = {}
        try:
            body = await e.decide(request, "direct", groups, None, tr)
        except ValueError as err:   # refused: its text, then each question alone (round 2a's rows)
            item.update({"accepted": False, "error": str(err), "sub": []})
            for name, q in (request.get("questions") or {}).items():
                try:
                    host.validate_question(name, q)
                except ValueError:
                    continue
                t1: dict = {}
                await e.decide({"state": request.get("state"), "questions": {name: q}}, "direct", groups, None, t1)
                r = t1["_rows"]["rows"][0]
                item["sub"].append({**row_record(r, t1["_hidden"][0], t1["_probs"][0]), "calls": t1["calls"],
                                    "call_ms": t1["call_ms"]})
            return item
        rows = [row_record(r, h, p) for r, h, p in zip(tr["_rows"]["rows"], tr["_hidden"], tr["_probs"])]
        item.update({"accepted": True, "rows": rows, "input_tokens": tr["input_tokens"], "state_tokens": tr["state_tokens"],
                     "response_indent2": json.dumps(body, indent=2, ensure_ascii=False),
                     "answers_dumps": json.dumps(body["answers"]),
                     "direct": {k: tr[k] for k in ("calls", "graph_ms", "call_ms")}, "_hidden": tr["_hidden"]})
        if spec["shared"]:
            ts: dict = {}
            body_s = await e.decide(request, "shared", groups, None, ts)
            item["shared"] = {"k": ts["shared_k"], "calls": ts["calls"], "graph_ms": ts["graph_ms"],
                              "hidden_bit_equal_direct": [bool(np.array_equal(a, b)) for a, b in zip(ts["_hidden"], tr["_hidden"])],
                              "hidden_sha256": [sha256_bytes(h) for h in ts["_hidden"]],
                              "p_bits": [[bits(x) for x in p] for p in ts["_probs"]],
                              "response_indent2_equal_direct": json.dumps(body_s, indent=2, ensure_ascii=False)
                              == item["response_indent2"]}
            if len(rows) > 1:   # prepared: once for the state, then all questions, then each alone
                pr = await e.prepare(request.get("state"))
                tp: dict = {}
                body_p = await e.decide_prepared(pr, request["questions"], groups, tp)
                singles = []
                for name, q in request["questions"].items():
                    t1 = {}
                    await e.decide_prepared(pr, {name: q}, groups, t1)
                    singles.append(t1)
                item["prepared"] = {"k": pr["k"], "prepare_calls": pr["calls"], "calls": tp["calls"],
                                    "rows_run_whole": tp["rows_run_whole"],
                                    "hidden_bit_equal_shared": [bool(np.array_equal(a, b)) for a, b in zip(tp["_hidden"], ts["_hidden"])],
                                    "p_bit_equal_shared": [[bits(x) for x in a] == [bits(x) for x in b]
                                                           for a, b in zip(tp["_probs"], ts["_probs"])],
                                    "single_hidden_bit_equal_shared": [bool(np.array_equal(t1["_hidden"][0], ts["_hidden"][i]))
                                                                       for i, t1 in enumerate(singles)],
                                    "response_indent2_equal_shared": json.dumps(body_p, indent=2, ensure_ascii=False)
                                    == json.dumps(body_s, indent=2, ensure_ascii=False)}
        return item

    async def go():
        out["load"] = {k: v for k, v in (await e.load()).items() if k != "descriptor"}
        first = None
        for i, rid in enumerate(spec["records"] + [spec["records"][0]]):
            item = await one(rid)
            if i == len(spec["records"]):
                again = item.get("_hidden")
                same = first is not None and again is not None and all(np.array_equal(a, b) for a, b in zip(first, again))
                if first is None:   # the first record was refused: compare its sub rows' hashes
                    same = [x["hidden_sha256"] for x in item["sub"]] == [x["hidden_sha256"] for x in out["records"][0]["sub"]]
                out["reset_check"] = {"record": rid, "bit_equal": bool(same)}
                print(f"  [{os.getpid()}] reset re-run {rid}: bit-equal {same}", flush=True)
                continue
            if i == 0:
                first = item.get("_hidden")
            item.pop("_hidden", None)
            out["records"].append(item)
            sh = item.get("shared", {}).get("hidden_bit_equal_direct")
            print(f"  [{os.getpid()}] {rid}: {'accepted' if item['accepted'] else 'refused'}, "
                  f"{len(item.get('rows', item.get('sub', [])))} rows" + (f", shared = direct {all(sh)}" if sh else ""), flush=True)

    asyncio.run(go())
    out["finished"] = time.time()
    out["max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    Path(spec["out"]).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    return 0


# --------------------------------------------------------------------------- check: driver
def other_gpu_processes() -> list[str]:
    ps = subprocess.run(["ps", "-axo", "pid=,etime=,command="], capture_output=True, text=True).stdout
    me = os.getpid()
    skip = ("/.local/bin/claude", "claude --", "shell-snapshots", "zsh -c", "/bin/zsh", "/bin/bash", "quiet_wait.py")
    return [ln.strip()[:200] for ln in ps.splitlines()
            if OTHER_GPU.search(ln) and not ln.strip().startswith(f"{me} ") and not any(s in ln for s in skip)
            and "decide.py worker" not in ln]


def lock_state() -> dict:
    p = gpu_lock()
    if not p.exists():
        return {"path": str(p), "exists": False}
    return {"path": str(p), "bytes": p.stat().st_size, "content": p.read_text()[:200],
            "mtime": datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(timespec="seconds")}


def env_record() -> dict:
    import importlib.metadata as md

    v = {"python": sys.version.split()[0], "numpy": np.__version__}
    for p in ("coreai-core", "coreai-torch", "tokenizers", "safetensors", "torch"):
        try:
            v[p] = md.version(p)
        except Exception as err:  # noqa: BLE001
            v[p] = repr(err)
    return {"versions": v, "macos_build": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
            "runtime": "coreai python runtime, AOT h16c .aimodelc, SpecializationOptions.default() (GPU), no JIT",
            "gpu": "shared with other sessions; _GPU_LOCK read only; the ms are contended"}


def cmd_check(args) -> int:
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists: never overwritten")
    e = D1(Path(args.bundle), args.aimodelc)
    recs = json.loads(FIXTURES.read_text())["records"]
    ids = [r["id"] for r in recs]
    if args.records:
        keep = set(args.records.split(","))
        ids = [x for x in ids if x in keep]
    work = Path(args.work) / (args.tag or e.name)
    work.mkdir(parents=True, exist_ok=True)
    n = -(-len(ids) // RECORDS_PER_PROCESS)
    size = -(-len(ids) // n)
    parts = [ids[i:i + size] for i in range(0, len(ids), size)]
    lock0, others0, t0 = lock_state(), other_gpu_processes(), time.monotonic()
    shards = []
    for k, part in enumerate(parts):
        spec = {"bundle": str(e.dir), "aimodelc": str(e.aimodelc), "fixtures": str(FIXTURES), "records": part,
                "shared": not args.no_shared, "toy_oracle": args.toy_oracle, "out": str(work / f"shard_{k:02d}.json")}
        sp = work / f"shard_{k:02d}.spec.json"
        sp.write_text(json.dumps(spec) + "\n")
        print(f"[check {e.name}] shard {k}: {len(part)} records + reset re-run", flush=True)
        rc = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--spec", str(sp)]).returncode
        if rc != 0 or not Path(spec["out"]).exists():
            raise SystemExit(f"shard {k} failed (exit {rc})")
        shards.append(json.loads(Path(spec["out"]).read_text()))
    gpu_s = time.monotonic() - t0
    items = [it for s in shards for it in s["records"]]
    # against the readout gate's transcript of the same asset (round 2a): per row, hidden sha256 and p bits
    gate_cmp = None
    if args.gate:
        g = json.loads(Path(args.gate).read_text())
        if g["bundle"]["aimodelc"]["tree_sha256"] != __import__("readout_gate").tree_digest(e.aimodelc)["tree_sha256"]:
            raise SystemExit("the gate transcript was taken on another asset")
        runs = {(r["id"], r["name"]): r for r in g["runs"] if r.get("variant", "base") == "base"}
        hid_eq = p_eq = n_rows = 0
        bad = []
        for it in items:
            for r in it.get("rows", it.get("sub", [])):
                n_rows += 1
                gr = runs.get((it["id"], r["name"]))
                he = gr is not None and gr["hidden_sha256"] == r["hidden_sha256"]
                pe = gr is not None and [bits(x) for x in gr["probs"]] == r["p_bits"]
                hid_eq += he
                p_eq += pe
                if (not he or not pe) and len(bad) < 8:
                    bad.append({"row": f"{it['id']}/{r['name']}", "hidden": he, "p": pe})
        gate_cmp = {"transcript": str(args.gate), "sha256": sha256_file(Path(args.gate)), "gate_rows": len(runs),
                    "rows": n_rows, "hidden_sha256_equal": hid_eq, "p_bit_equal": p_eq, "first_differences": bad}
    multi = [it for it in items if it.get("accepted") and len(it["rows"]) > 1]
    shared_all = [it for it in items if "shared" in it]
    summary = {
        "records": len(items), "accepted": sum(it["accepted"] for it in items),
        "refused": [{"id": it["id"], "error": it["error"]} for it in items if not it["accepted"]],
        "rows": sum(len(it.get("rows", it.get("sub", []))) for it in items),
        "finite_rows": sum(r["finite"] and not r["all_zero"] for it in items for r in it.get("rows", it.get("sub", []))),
        "reset_bit_equal": [s["reset_check"]["bit_equal"] for s in shards],
        "shared_multi_question_records": len(multi),
        "shared_multi_hidden_bit_equal_direct": sum(all(it["shared"]["hidden_bit_equal_direct"]) for it in multi),
        "shared_all_records": len(shared_all),
        "shared_all_hidden_bit_equal_direct": sum(all(it["shared"]["hidden_bit_equal_direct"]) for it in shared_all),
        "shared_all_response_equal_direct": sum(it["shared"]["response_indent2_equal_direct"] for it in shared_all),
        "prepared_records": sum("prepared" in it for it in items),
        "prepared_hidden_bit_equal_shared": sum(all(it["prepared"]["hidden_bit_equal_shared"]) for it in items if "prepared" in it),
        "prepared_single_bit_equal_shared": sum(all(it["prepared"]["single_hidden_bit_equal_shared"])
                                                for it in items if "prepared" in it),
        "prepared_response_equal_shared": sum(it["prepared"]["response_indent2_equal_shared"] for it in items if "prepared" in it),
        "prepared_rows_run_whole": sum(it["prepared"]["rows_run_whole"] for it in items if "prepared" in it),
    }
    calls = [m for it in items for m in it.get("direct", {}).get("call_ms", [])]
    rec = {"schema": "d1-decide-check/1", "generated_at": now(), "bundle": str(e.dir), "aimodelc": str(e.aimodelc),
           "aimodelc_main_hash": (e.aimodelc / "main.hash").read_bytes().hex(), "toy_oracle": args.toy_oracle,
           "script": {"path": "conversion/d1/decide.py", "sha256": sha256_file(Path(__file__))}, "env": env_record(),
           "gpu_lock": {"start": lock0, "end": lock_state()}, "other_gpu_processes": {"start": others0, "end": other_gpu_processes()},
           "summary": summary, "gate": gate_cmp,
           "timing_contended": {"gpu_processes_seconds": gpu_s, "processes": len(shards),
                                "load_seconds": [s["load"]["seconds"] for s in shards],
                                "ms_per_call_median_direct": float(np.median(calls)) if calls else None},
           "processes": [{k: v for k, v in s.items() if k != "records"} for s in shards], "records": items}
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    if args.shared_out:
        so = Path(args.shared_out)
        if so.exists():
            raise SystemExit(f"{so} exists: never overwritten")
        so.write_text(json.dumps({
            "schema": "d1-py-shared-vs-direct/1", "generated_at": now(), "bundle": str(e.dir), "check": str(out),
            "what": "decide.py on the AOT graph: every accepted request direct, shared (the state's first floor(Ls / S) * S "
                    "tokens once, states copied per question) and, for the multi-question ones, prepared (prepare(state) "
                    "then decide_prepared, all questions and each alone): hidden rows bit for bit",
            "multi_question_records": [{"id": it["id"], "rows": len(it["rows"]), "row_tokens": [r["T"] for r in it["rows"]],
                                        "state_tokens": it["state_tokens"], "k": it["shared"]["k"],
                                        "calls_direct": it["direct"]["calls"], "calls_shared": it["shared"]["calls"],
                                        "hidden_bit_equal_direct": it["shared"]["hidden_bit_equal_direct"],
                                        "response_equal_direct": it["shared"]["response_indent2_equal_direct"],
                                        "prepared": it.get("prepared")} for it in multi],
            "refused_multi_question": [{"id": it["id"], "error": it["error"]} for it in items
                                       if not it["accepted"] and len(recs_q(it["id"])) > 1],
            "summary": {k: summary[k] for k in summary if k.startswith(("shared", "prepared"))}}, indent=1) + "\n")
    print(json.dumps(summary, indent=None)[:2000])
    if gate_cmp:
        print(f"vs gate: hidden {gate_cmp['hidden_sha256_equal']}/{gate_cmp['rows']}, p {gate_cmp['p_bit_equal']}/{gate_cmp['rows']}")
    print(f"-> {out}")
    return 0


def recs_q(rid: str) -> dict:
    return next(r for r in json.loads(FIXTURES.read_text())["records"] if r["id"] == rid)["request"].get("questions") or {}


# --------------------------------------------------------------------------- e2e (round 3b's three picture rows)
def e2e_compare(got: np.ndarray, ref: np.ndarray, slots_pos: np.ndarray) -> dict:
    """round 3b's r3b_toy_vlm_e2e.compare (the same arithmetic, so its numbers come back unchanged)."""
    a, b = got.astype(np.float64), ref.astype(np.float64)
    c = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    t = len(c) - 1
    img = c[slots_pos] if len(slots_pos) else np.array([1.0])
    txt = np.delete(c, slots_pos) if len(slots_pos) else c
    return {"answer_slot_cos": float(c[t]), "answer_slot_max_abs": float(d[t].max()),
            "answer_slot_rel_change": float(np.linalg.norm(got[t].astype(np.float64) - ref[t]) / np.linalg.norm(ref[t])),
            "every_position_cos_min": float(c.min()), "every_position_cos_argmin": int(c.argmin()),
            "image_positions_cos_min": float(img.min()), "text_positions_cos_min": float(txt.min()),
            "max_abs": float(d.max()), "finite": bool(np.isfinite(got.astype(np.float64)).all())}


def cmd_e2e(args) -> int:
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists: never overwritten")
    e = D1(Path(args.bundle))
    tower = Tower(Path(args.tower))
    if tower.d != e.d:
        raise SystemExit(f"the tower's width {tower.d} != the decoder's {e.d}")
    fx = {r["id"]: r for r in json.loads(IMAGE_FIXTURES.read_text())["records"]}
    r3b = {r["id"]: r for r in json.loads(Path(args.r3b).read_text())["records"]} if args.r3b else {}
    arrays = Path(args.work) / "e2e"
    arrays.mkdir(parents=True, exist_ok=True)
    results = []

    async def go():
        load = {"decoder": {k: v for k, v in (await e.load()).items() if k != "descriptor"},
                "tower": {k: v for k, v in (await tower.load()).items() if k != "descriptor"}}
        for rid in E2E_RECORDS:
            r = fx[rid]
            name, q = next(iter(r["request"]["questions"].items()))
            request = {"state": r["request"]["state"], "questions": {name: q}}
            pics = [IMAGE_FIXTURES.parent / x for x in r["images"]]
            item: dict = {"id": rid, "question": name}
            images = await image_rows(tower, pics, item)
            tr: dict = {}
            body = await e.decide(request, "direct", None, images, tr)
            row = tr["_rows"]["rows"][0]
            h = tr["_hidden"][0]
            # the oracle's npz inputs (what a Swift run reads with --tower-inputs) against the ones made here
            ocrops = sorted((LANE / "oracle_toy_vision" / rid).glob("crop*.npz"))
            same_inputs = len(ocrops) == len(item["crops"]) and all(
                all(np.array_equal(np.load(o)[k], c["_inputs"][k]) for k in ("patches", "pos_table", "key_bias", "unshuffle_idx"))
                for o, c in zip(ocrops, item["crops"]))
            zero_tr: dict = {}
            await e.decide(request, "direct", None, {"plans": images["plans"], "rows": None}, zero_tr)
            np.savez(arrays / f"{rid}.npz", hidden=h, hidden_zero_images=zero_tr["_hidden"][0],
                     image_rows=images["rows"], graph_ids=np.asarray(row["graph_ids"], np.int32),
                     **{f"crop{i:02d}_out": c["_out"] for i, c in enumerate(item["crops"])})
            ext = row["row_ids"]
            slot_pos = np.array([i for i, t in enumerate(ext) if t >= vh.V], np.int64)
            res = {"id": rid, "question": name, "row_tokens": row["row_len"], "calls": tr["calls"],
                   "ids_sha256": hashlib.sha256(np.asarray(vh.prompt_ids(e.tok, vh.image_prefix_text(request["state"], 1)
                                                                         + row["suffix"], images["plans"]),
                                                           np.int64).tobytes()).hexdigest(),
                   "n_image_tokens": int(len(images["rows"])), "slots": [int(slot_pos[0]), int(slot_pos[-1])],
                   "hidden_sha256": sha256_bytes(h), "slot_hidden_sha256": sha256_bytes(h[row["slot"]]),
                   "image_rows_sha256": sha256_bytes(images["rows"]), "inputs_equal_oracle_npz": bool(same_inputs),
                   "crops": [{k: v for k, v in c.items() if not k.startswith("_")} for c in item["crops"]],
                   "p": tr["_probs"][0], "p_bits": [bits(x) for x in tr["_probs"][0]], "read_groups": row["read_groups"],
                   "response_indent2": json.dumps(body, indent=2, ensure_ascii=False), "input_tokens": tr["input_tokens"],
                   "ms_per_call_median_contended": float(np.median(tr["call_ms"])),
                   "zero_images_hidden_sha256": sha256_bytes(zero_tr["_hidden"][0]), "arrays": str(arrays / f"{rid}.npz")}
            if args.eager:   # round 3b's torch fp32 eager composite, recomputed
                res["eager"] = eager_compare(rid, e, tower, item, row, h, zero_tr["_hidden"][0], slot_pos, arrays)
                if rid in r3b:
                    res["eager"]["equal_r3b"] = {f: res["eager"]["base"][f] == r3b[rid][f] for f in E2E_FIELDS}
                    res["eager"]["equal_r3b_zero_control"] = {
                        f: res["eager"]["zero"][f] == r3b[rid]["controls"]["image_embeds_zero"][f] for f in E2E_FIELDS}
                    res["eager"]["ids_sha256_equal_r3b"] = res["ids_sha256"] == r3b[rid]["ids_sha256"]
            results.append(res)
            print(f"{rid}: {res['row_tokens']} tokens, {res['n_image_tokens']} image rows, inputs = oracle npz {same_inputs}, "
                  f"hidden {res['hidden_sha256'][:12]}" + (f", r3b numbers equal {all(res['eager']['equal_r3b'].values())}"
                                                           if args.eager and rid in r3b else ""), flush=True)
        return load

    t0 = time.monotonic()
    load = asyncio.run(go())
    rec = {"schema": "d1-py-e2e/1", "generated_at": now(), "decoder": {"bundle": str(e.dir), "aimodelc": str(e.aimodelc),
                                                                       "main_hash": (e.aimodelc / "main.hash").read_bytes().hex()},
           "tower": {"bundle": str(tower.dir), "aimodelc": str(tower.aimodelc),
                     "main_hash": (tower.aimodelc / "main.hash").read_bytes().hex()},
           "r3b": args.r3b, "load": load, "records": results, "seconds": time.monotonic() - t0, "env": env_record(),
           "gpu_lock": lock_state()}
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    print(f"-> {out}")
    return 0


def eager_compare(rid: str, e: D1, tower: Tower, item: dict, row: dict, h: np.ndarray, h_zero: np.ndarray,
                  slot_pos: np.ndarray, arrays: Path) -> dict:
    """round 3b's eager side: the toy tower's eager forward per crop -> image_embeds fp32 -> the toy decoder's one-shot
    eager forward over the whole folded row; compared with the graph's hidden rows as r3b compared them."""
    import torch

    import export_decoder as ed
    import lfm2_vl_tower as T
    from coreai_models.models.macos.lfm2 import build_decode_state

    tower_eager = T.Lfm2VlTowerExact.toy(0, LANE / "oracle_toy_vision/toy_snapshot").eval()
    dec_eager = ed.toy_model(0, e.N).eval()
    e_rows = []
    for c in item["crops"]:
        with torch.no_grad():
            e_rows.append(tower_eager(*[torch.from_numpy(c["_inputs"][k]) for k in T.INPUT_NAMES])[:c["n_tokens"]].numpy())
    e_img = torch.zeros(e.N, e.d)
    n_img = sum(c["n_tokens"] for c in item["crops"])
    e_img[:n_img] = torch.from_numpy(np.concatenate(e_rows))
    folded = np.asarray(row["graph_ids"], np.int32)
    with torch.no_grad():
        st = build_decode_state(dec_eager.config, max_seq_len=e.max_ctx, dtype=torch.float32)
        ref = dec_eager(torch.from_numpy(folded)[None], torch.arange(len(folded), dtype=torch.int32)[None], e_img,
                        st["k_cache"], st["v_cache"], st["conv_state"])[0].numpy()
    np.save(arrays / f"{rid}_eager_ref.npy", ref)
    return {"base": e2e_compare(h, ref, slot_pos), "zero": e2e_compare(h_zero, ref, slot_pos),
            "ref": str(arrays / f"{rid}_eager_ref.npy"), "slots_pos": [int(slot_pos[0]), int(slot_pos[-1]), int(len(slot_pos))]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one request -> its response")
    r.add_argument("--bundle", required=True)
    r.add_argument("--aimodelc")
    r.add_argument("--request", required=True, help='{"state", "questions"[, "images": [paths]]}')
    r.add_argument("--tower", help="a tower bundle (export_vision.py) for a request with images")
    r.add_argument("--shared", action="store_true")
    r.add_argument("--toy-oracle", help="a toy oracle's records (round 2a's toy groups), with --record-id")
    r.add_argument("--record-id")
    r.add_argument("--out", required=True)
    r.add_argument("--trace")
    c = sub.add_parser("check", help="every fixture record from its raw request, against the readout gate's transcript")
    c.add_argument("--bundle", required=True)
    c.add_argument("--aimodelc")
    c.add_argument("--toy-oracle")
    c.add_argument("--gate", help="the readout gate's transcript of the same asset (hidden sha256, p)")
    c.add_argument("--records", help="comma list of record ids")
    c.add_argument("--no-shared", action="store_true")
    c.add_argument("--tag")
    c.add_argument("--work", default=str(WORK), help="the shards' directory (default <lane>/decide)")
    c.add_argument("--out", required=True)
    c.add_argument("--shared-out", help="the shared = direct / prepared = shared summary")
    x = sub.add_parser("e2e", help="round 3b's three picture rows through the tower and the decoder")
    x.add_argument("--bundle", required=True)
    x.add_argument("--tower", required=True)
    x.add_argument("--r3b", default=str(LANE / "results" / "r3b_toy_vlm_e2e.json"))
    x.add_argument("--eager", action="store_true", help="recompute round 3b's torch fp32 eager comparison")
    x.add_argument("--work", default=str(WORK), help="the arrays' directory (default <lane>/decide)")
    x.add_argument("--out", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--spec", required=True)
    args = ap.parse_args()
    if args.cmd == "worker":
        return worker(Path(args.spec))
    return {"run": cmd_run, "check": cmd_check, "e2e": cmd_e2e}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
