#!/usr/bin/env python3
"""fp32 torch parity: the d1 decoder module driven the way its graph runs, read out through host.py, against the oracle.

`lfm2_d1_decoder.Lfm2D1Decoder` in fp32 on the CPU (`--threads`), one row per question: the oracle's `row_ids` from
fresh zero states in static S-token chunks (call c: ids[cS : cS + S], position_ids 0..cS+S-1), the last chunk padded
with <|pad|> 124893 and the padded positions' outputs dropped — `readout_gate.py`'s call order. The slot's hidden row
(the row's last token) goes through the host's readout (`host.option_logits` on the module's tied embedding rows of
the question's group ids, fp32 -> float64; group max; softmax over the options) and is compared with the oracle's
probabilities (`oracle/records_oracle.json` from oracle_d1.py: the provider's code, fp32, CPU):

  p1   every question at S = 16: argmax, max |dp| and mean |dp| against the oracle; with `--hidden-npz <dir>` (the
       oracle's `hidden/<id>.npz`, q<k>_hidden [T, 2048]) the position cosine of every row the oracle kept.
       Bar (fixed before running): argmax equal on every question (near-ties included), max |dp| <= 1e-4, minimum
       position cosine >= 0.9999 where the oracle kept hidden rows.
  p2   P2_ROWS: the S-chunk order against one forward over the whole row (the module's forward with every id in one
       call -> forward_stateful_embeds, positions 0..T-1, zero states): max |d| and the lowest cosine over every
       position. Bar: max |d| <= 1e-4 (the toy: <= 1e-5).
  p2s  round 9b: P2_ROWS through the static form (lfm2_d1_static.py, `--kv-write slice,roll`, `--context C`: call c
       with position_ids cS..cS+S-1, the KV axis at C, the mask built from the positions) against the dynamic form,
       both in S = 16 calls from zero states on the same weights: max |d|, the lowest position cosine, bit equality,
       both p against the oracle -> `--out`. Bar: max |d| <= 1e-4.
  p3   `--widths 16,32,64` on P3_ROWS: each width's hidden rows against S = 16's (max |d|) and its p against the
       oracle (recorded, not a bar).
  p4   `--red`: the five arms of `fixtures/red_arms.json` on fp32 torch, every perturbed row against its base row
       (the same question of the unperturbed record): red when (a) an argmax moves on a question whose oracle top-2
       margin is above 0.02, or (b) max |dp| > 0.02, or (c) the mean over the arm's rows of the row's mean |dp| >
       0.002 — the gate's own bar (FACTS §7). The arms show whether the model itself moves; readout_gate.py asks the
       same of the graph.
  merge   the stage files -> results/parity_decoder_torch.json

`--toy`: the toy module (`export_decoder.toy_model(--toy-seed)`: toy_graph_check.py's config, seeded random weights)
and the toy oracle (`--toy-oracle <json>`, built here when it does not exist): the same module's one forward over each
fixture row, the row's real ids folded into the toy vocabulary (id % 256), each question's readout groups drawn at
random (seeded by record and question, the real groups' shape) from the folded candidate ids, the probabilities by the
host's arithmetic on the toy embedding. So `--toy` checks the instruments, not a model: P1 and P2 compare the chunk
order with the one forward it must equal; P4 says whether the arms move a random network's readout.

    cd conversion/d1
    PY=<coreai-models venv>/bin/python
    for k in 0 1 2; do $PY parity_decoder_torch.py p1 --shard $k --shards 3 --threads 1 & done; wait
    $PY parity_decoder_torch.py p2 --threads 1; $PY parity_decoder_torch.py p3 --threads 1
    $PY parity_decoder_torch.py p4 --threads 1; $PY parity_decoder_torch.py merge    # round 3 (needs the weights)
    $PY parity_decoder_torch.py toy --toy-oracle $ZOO_WORK_ROOT/_d1_3b/oracle_toy/records_oracle.json \\
        --out $ZOO_WORK_ROOT/_d1_3b/results/<json>                                    # P1-P4 on the toy, one process

The toy oracle is also what `readout_gate.py --toy-oracle` reads; this file holds what the two share (the toy oracle,
the red-arm rows, the position cosine).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
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
from _paths import hf_snapshot, work_path  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")

LANE = work_path("_d1_3b")
ORACLE = LANE / "oracle"
PARITY = LANE / "parity"
FIXTURES = LANE / "fixtures"
RENDER_IDS = LANE / "results" / "render_ids.json"
MODEL = {"hf_id": "LiquidAI/d1-3B", "revision": "da1fe36a861f24690f27f622dca1d8688503d113"}
PAD_ID = host.PAD_ID
CHUNK = 16
WIDTHS = (16, 32, 64)
BAR = {"argmax_equal": "every question, near-ties included", "max_abs_dp": 1e-4, "min_pos_cos": 0.9999,
       "p2_max_abs_diff": 1e-4, "p2_max_abs_diff_toy": 1e-5}
RED_BAR = {"max_abs_dp": 0.02, "mean_of_run_mean_abs_dp": 0.002, "near_tie_top2_margin": 0.02}
RED_RULE = ("red = the perturbed rows against their base rows fail the gate's own bar (FACTS §7): (a) an argmax moves "
            "on a question whose oracle top-2 margin is above 0.02, or (b) max |dp| > 0.02, or (c) the mean over the "
            "arm's rows of the row's mean |dp| > 0.002")
P2_ROWS = (("tv4_000", "answer"), ("card_refund", "team"), ("long_34k", "refunded_twice"))
P3_ROWS = (("tv4_000", "answer"), ("tv4x_qnli_00", "answers"), ("tv4s_00", None), ("semif_a3f18f3a63d45345942b", "answer"),
           ("own_ticket_01", None), ("card_refund", None), ("long_15k", "temp_excursion"), ("long_34k", "refunded_twice"))
NEAR_TIE = 0.02
TOY_HIDDEN_ROWS = ("tv4_000", "card_refund", "long_34k")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def snapshot() -> Path:
    return Path(hf_snapshot(MODEL["hf_id"], revision=MODEL["revision"]))


# --------------------------------------------------------------------------- comparisons
def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def compare_hidden(mine: np.ndarray, ref: np.ndarray) -> dict:
    d = np.abs(mine.astype(np.float64) - ref.astype(np.float64))
    c = cos_rows(mine, ref)
    i = int(c.argmin())
    ref_max = float(np.abs(ref).max())
    return {"max_abs_diff": float(d.max()), "ref_absmax": ref_max, "rel_max_abs_diff": float(d.max()) / ref_max,
            "min_pos_cos": float(c[i]), "min_pos_cos_index": i, "mean_pos_cos": float(c.mean()),
            "positions_below_0.9999": int((c < 0.9999).sum()), "finite": bool(np.isfinite(mine).all())}


def score_probs(p: list[float], q: dict) -> dict:
    """Probabilities against an oracle question (probs, keys, argmax key, top2_margin, near_tie)."""
    p = np.asarray(p, np.float64)
    po = np.asarray(q["probs"], np.float64)
    dp = np.abs(p - po)
    am, am_o = int(p.argmax()), q["keys"].index(q["argmax"])
    return {"argmax": am, "argmax_oracle": am_o, "argmax_equal": am == am_o, "max_abs_dp": float(dp.max()),
            "mean_abs_dp": float(dp.mean()), "oracle_top2_margin": q["top2_margin"], "near_tie": bool(q["near_tie"]),
            "n_options": int(po.size), "probs": [float(x) for x in p], "probs_oracle": [float(x) for x in po]}


def margin_of(probs: list[float]) -> float:
    s = sorted(probs, reverse=True)
    return s[0] - s[1] if len(s) > 1 else 1.0


# --------------------------------------------------------------------------- the oracle
def load_oracle(path: Path) -> tuple[dict, dict]:
    """(document, {record id: record}) of records_oracle.json (oracle_d1.py) or a toy oracle (same layout)."""
    doc = json.loads(Path(path).read_text())
    return doc, {r["id"]: r for r in doc["records"]}


def question_of(rec: dict, name: str | None) -> tuple[int, dict]:
    """(index, question) of a record by name (None = the first)."""
    for k, q in enumerate(rec["questions"]):
        if name is None or q["name"] == name:
            return k, q
    raise SystemExit(f"{rec['id']}: no question {name!r}")


def all_rows(recs: dict) -> list[tuple[str, int]]:
    return [(rid, k) for rid, r in recs.items() for k in range(len(r["questions"]))]


def oracle_hidden(hidden_dir: Path | None, rid: str, k: int) -> np.ndarray | None:
    if hidden_dir is None:
        return None
    p = Path(hidden_dir) / f"{rid}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return z[f"q{k}_hidden"] if f"q{k}_hidden" in z.files else None


# --------------------------------------------------------------------------- the module
class Runner:
    """A d1 decoder module in fp32 on the CPU, its chunk loop and one-call forward, and the host's readout on its
    tied embedding."""

    def __init__(self, model, threads: int, pad_id: int = PAD_ID):
        import torch

        torch.set_num_threads(threads)
        self.torch = torch
        self.model = model.float().eval()
        self.cfg = model.config
        self.pad_id = pad_id
        self.E = model.model.embed_tokens.weight.detach().float().numpy()
        self.img = torch.zeros(model.n_image_tokens, self.cfg.hidden_size)

    def states(self, length: int) -> dict:
        from coreai_models.models.macos.lfm2 import build_decode_state

        return build_decode_state(self.cfg, max_seq_len=length, dtype=self.torch.float32)

    def chunked(self, ids: list[int], S: int = CHUNK) -> tuple[np.ndarray, dict]:
        """One row from zero states in S-token calls (the graph's order) -> hidden [T, H] fp32."""
        torch = self.torch
        T = len(ids)
        n = -(-T // S)
        x = torch.full((n * S,), self.pad_id, dtype=torch.int32)
        x[:T] = torch.tensor(ids, dtype=torch.int32)
        st = self.states(n * S)
        outs = []
        t0 = time.monotonic()
        with torch.inference_mode():
            for c in range(n):
                h = self.model(x[c * S:(c + 1) * S].reshape(1, S), torch.arange((c + 1) * S, dtype=torch.int32)[None],
                               self.img, st["k_cache"], st["v_cache"], st["conv_state"])
                outs.append(h[0])
        hid = torch.cat(outs)[:T].numpy().astype(np.float32)
        return hid, {"T": T, "S": S, "chunks": n, "padded": n * S, "seconds": time.monotonic() - t0}

    def chunked_static(self, ids: list[int], S: int, context: int, kv_write: str) -> tuple[np.ndarray, dict]:
        """One row through the static form (lfm2_d1_static.py, the same weights: the module's class switched for the
        run and back) from zero states at `context` KV slots, call c with position_ids cS..cS+S-1 -> hidden [T, H]."""
        from lfm2_d1_decoder import Lfm2D1Decoder
        from lfm2_d1_static import make_static

        torch = self.torch
        T = len(ids)
        n = -(-T // S)
        if n * S > context:
            raise ValueError(f"a row of {T} ids runs {n * S} padded positions, over the context {context}")
        x = torch.full((n * S,), self.pad_id, dtype=torch.int32)
        x[:T] = torch.tensor(ids, dtype=torch.int32)
        st = self.states(context)
        outs = []
        t0 = time.monotonic()
        m = make_static(self.model, context, kv_write)
        try:
            with torch.inference_mode():
                for c in range(n):
                    h = m(x[c * S:(c + 1) * S].reshape(1, S), torch.arange(c * S, (c + 1) * S, dtype=torch.int32)[None],
                          self.img, st["k_cache"], st["v_cache"], st["conv_state"])
                    outs.append(h[0])
        finally:
            self.model.__class__ = Lfm2D1Decoder
        hid = torch.cat(outs)[:T].numpy().astype(np.float32)
        return hid, {"T": T, "S": S, "chunks": n, "padded": n * S, "context": context, "kv_write": kv_write,
                     "seconds": time.monotonic() - t0}

    def oneshot(self, ids: list[int]) -> tuple[np.ndarray, dict]:
        """The whole row in one call from zero states (forward -> forward_stateful_embeds) -> hidden [T, H] fp32."""
        torch = self.torch
        T = len(ids)
        st = self.states(T)
        t0 = time.monotonic()
        with torch.inference_mode():
            h = self.model(torch.tensor(ids, dtype=torch.int32)[None], torch.arange(T, dtype=torch.int32)[None],
                           self.img, st["k_cache"], st["v_cache"], st["conv_state"])
        return h[0].numpy().astype(np.float32), {"T": T, "seconds": time.monotonic() - t0}

    def readout(self, h_slot: np.ndarray, groups: list[list[int]]) -> list[float]:
        ids = host.group_ids(groups)
        return host.readout(h_slot, self.E[ids], ids, groups)


# --------------------------------------------------------------------------- the toy oracle
def toy_groups(rid: str, qname: str, real_groups: list[list[int]], table_ids: list[int]) -> list[list[int]]:
    """Random readout groups of the real groups' shape, distinct ids from the toy table, seeded by record and question."""
    seed = int.from_bytes(hashlib.sha256(f"{rid}/{qname}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    n = sum(len(g) for g in real_groups)
    pick = [int(x) for x in rng.choice(np.asarray(table_ids), size=n, replace=False)]
    out, i = [], 0
    for g in real_groups:
        out.append(pick[i:i + len(g)])
        i += len(g)
    return out


def build_toy_oracle(seed: int, out_json: Path, table_dir: Path | None = None, render_ids: Path = RENDER_IDS) -> dict:
    """The toy oracle (see the module docstring) -> out_json (+ hidden/<id>.npz for TOY_HIDDEN_ROWS); never overwritten.
    `table_dir` = a toy bundle's head/ (its rows must be the toy embedding's); None = the rows from the module."""
    import export_decoder as ed
    import export_option_rows as eor

    out_json = Path(out_json)
    if out_json.exists():
        raise SystemExit(f"{out_json} exists: the toy oracle is never overwritten")
    model = ed.toy_model(seed)
    round1 = ed.toy_round1_check(model, seed)
    V = model.config.vocab_size
    runner = Runner(model, threads=1, pad_id=ed.toy_fold(PAD_ID, V))
    if table_dir is not None:
        t_ids, t_rows = eor.table_arrays(Path(table_dir))
        t_ids = [int(i) for i in t_ids]
        if not np.array_equal(t_rows.view(np.uint32), runner.E[t_ids].astype(np.float32).view(np.uint32)):
            raise SystemExit(f"{table_dir}: the table rows are not this toy's embedding rows (seed {seed})")
        table = {"dir": str(table_dir), "sha256": sha256_file(Path(table_dir) / eor.TABLE_FILE), "n": len(t_ids)}
    else:
        t_ids, _ = ed.toy_table_ids(V)
        table = {"dir": None, "n": len(t_ids), "rows": "the module's embedding"}
    rdoc = json.loads(Path(render_ids).read_text())
    hidden_dir = out_json.parent / "hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    records, t0 = [], time.monotonic()
    for r in rdoc["records"]:
        qs, arrays = [], {}
        for k, q in enumerate(r["questions"]):
            ids = [ed.toy_fold(i, V) for i in q["row_ids"]]
            groups = toy_groups(r["id"], q["name"], q["groups"], t_ids)
            h, _ = runner.oneshot(ids)
            probs = runner.readout(h[-1], groups)
            am = max(range(len(probs)), key=probs.__getitem__)
            m = margin_of(probs)
            qs.append({"name": q["name"], "type": q["type"], "row_ids": ids, "row_len": len(ids), "slot": len(ids) - 1,
                       "groups": groups, "keys": q["keys"], "real_groups": q["groups"], "probs": probs,
                       "argmax": q["keys"][am], "top2_margin": m, "near_tie": m <= NEAR_TIE,
                       "hidden_finite": bool(np.isfinite(h).all())})
            if r["id"] in TOY_HIDDEN_ROWS:
                arrays[f"q{k}_hidden"] = h
                arrays[f"q{k}_ids"] = np.asarray(ids, np.int32)
        if arrays:
            np.savez(hidden_dir / f"{r['id']}.npz", **arrays)
        records.append({"id": r["id"], "source": r["source"], "questions": qs, "refused": r["refused"]})
    margins = [q["top2_margin"] for r in records for q in r["questions"]]
    doc = {"schema": "d1-toy-oracle/1",
           "what": ("the toy module's one forward per fixture row (fp32, CPU, zero states, positions 0..T-1), the row's "
                    f"real ids folded id % {V}, random readout groups (seeded by record/question, the real groups' "
                    "shape) from the folded candidate ids, probabilities by host.readout on the toy embedding rows"),
           "toy": {"seed": seed, "vocab_size": V, "fold": f"id % {V}", "pad_folded": runner.pad_id, **round1},
           "table": table, "render_ids": {"path": str(render_ids), "sha256": sha256_file(Path(render_ids))},
           "hidden": {"dir": str(hidden_dir), "records": list(TOY_HIDDEN_ROWS)},
           "summary": {"records": len(records), "questions": len(margins),
                       "near_ties": sum(m <= NEAR_TIE for m in margins),
                       "top2_margin": {"min": min(margins), "p10": float(np.quantile(margins, 0.1)),
                                       "p50": float(np.median(margins)), "max": max(margins)},
                       "hidden_finite": all(q["hidden_finite"] for r in records for q in r["questions"]),
                       "seconds": round(time.monotonic() - t0, 2)},
           "versions": versions(), "records": records,
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc) + "\n")
    print(f"toy oracle: {len(records)} records, {len(margins)} questions, near-ties {doc['summary']['near_ties']}, "
          f"top-2 margin p50 {doc['summary']['top2_margin']['p50']:.3f} -> {out_json}", flush=True)
    return doc


# --------------------------------------------------------------------------- the red arms
def red_records(arms_doc: dict) -> list[dict]:
    """The arms' perturbed requests as fixture records (oracle_d1.py --fixtures reads them unchanged): id = the request
    name (the arm id for a word / not arm, `<x>_with_state_of_<y>` for a state swap)."""
    out = []
    for arm in arms_doc["arms"]:
        reqs = ([(arm["id"], arm["request"])] if arm["kind"] in ("word", "not") else list(arm["requests"].items()))
        for name, req in reqs:
            out.append({"id": name, "source": f"red_{arm['kind']}", "request": req, "gold": {}, "note": f"red arm {arm['id']}",
                        "provenance": {"red_arms": str(FIXTURES / "red_arms.json"), "arm": arm["id"]}})
    return out


def red_rows(arms_doc: dict, oracle_recs: dict, tok, fold_vocab: int | None = None, table_ids=None) -> dict:
    """The rows of the red arms: every arm's perturbed request rendered by the host (the bundle's / snapshot's
    tokenizer), each compared question tied to its base (the same question of the unperturbed record in the oracle).
    For the real model the host must reproduce the base rows of the fixture (render_ids.json) and the base questions'
    groups; for the toy (`fold_vocab`) the ids are folded and the base's toy groups are used.
    -> {"arms": [{id, kind, rows: [{request, question, base: [id, k], ids, slot, groups, keys}]}], "base_rows": [[id, k]]}"""
    rdoc = json.loads(RENDER_IDS.read_text())
    render = {r["id"]: r for r in rdoc["records"]}
    fixtures = {r["id"]: r for r in json.loads((FIXTURES / "records.json").read_text())["records"]}
    arms, bases = [], []
    for arm in arms_doc["arms"]:
        if arm["kind"] in ("word", "not"):
            reqs = [(arm["id"], arm["request"], arm["base"], [arm["question"]])]
        elif arm["kind"] == "state_swap":
            reqs = [(name, req, name.split("_with_state_of_")[0], None) for name, req in arm["requests"].items()]
        else:
            raise SystemExit(f"unknown arm kind {arm['kind']}")
        rows = []
        for name, req, base_id, only in reqs:
            base_req = fixtures[base_id]["request"]
            b_base = host.build_request(base_req, tok)
            want = {q["name"]: q["row_ids"] for q in render[base_id]["questions"]}
            if any(q["row_ids"] != want.get(q["name"]) for q in b_base["questions"]):
                raise SystemExit(f"{base_id}: the host's rows differ from render_ids.json (another tokenizer?)")
            b = host.build_request(req, tok, table_ids=table_ids)
            for q in b["questions"]:
                if only is not None and q["name"] not in only:
                    continue
                k, oq = question_of(oracle_recs[base_id], q["name"])
                if fold_vocab is None:
                    if q["groups"] != oq["groups"] or q["keys"] != oq["keys"]:
                        raise SystemExit(f"{name}/{q['name']}: groups or keys differ from the base question's")
                    ids, groups = q["row_ids"], q["groups"]
                else:
                    ids, groups = [int(i) % fold_vocab for i in q["row_ids"]], oq["groups"]
                rows.append({"request": name, "question": q["name"], "base": [base_id, k], "ids": ids,
                             "slot": len(ids) - 1, "groups": groups, "keys": oq["keys"], "row_len": len(ids)})
                if [base_id, k] not in bases:
                    bases.append([base_id, k])
        arms.append({"id": arm["id"], "kind": arm["kind"], "rows": rows})
    return {"arms": arms, "base_rows": bases, "rule": RED_RULE, "file": str(FIXTURES / "red_arms.json"),
            "file_sha256": sha256_file(FIXTURES / "red_arms.json"), "folded": fold_vocab}


def build_toy_red_oracle(seed: int, toy_oracle: Path, out_json: Path) -> dict:
    """The toy side of the gate's oracle pre-check: the toy module's one forward over every red-arm row (the perturbed
    request rendered by the host with the snapshot's tokenizer, folded), read out with its base question's toy groups
    -> a records_oracle-like JSON keyed by the arm request names (as oracle_d1.py writes for red_records)."""
    import export_decoder as ed

    out_json = Path(out_json)
    if out_json.exists():
        raise SystemExit(f"{out_json} exists: the toy red oracle is never overwritten")
    doc, recs = load_oracle(Path(toy_oracle))
    if int(doc["toy"]["seed"]) != seed:
        raise SystemExit(f"{toy_oracle} is the seed-{doc['toy']['seed']} toy, not {seed}")
    model = ed.toy_model(seed)
    V = model.config.vocab_size
    runner = Runner(model, threads=1, pad_id=ed.toy_fold(PAD_ID, V))
    tok = host.load_tokenizer(snapshot() / "tokenizer.json")
    red = red_rows(json.loads((FIXTURES / "red_arms.json").read_text()), recs, tok, fold_vocab=V)
    records: dict = {}
    for arm in red["arms"]:
        for row in arm["rows"]:
            h, _ = runner.oneshot(row["ids"])
            probs = runner.readout(h[-1], row["groups"])
            am = max(range(len(probs)), key=probs.__getitem__)
            rec = records.setdefault(row["request"], {"id": row["request"], "source": f"red_{arm['kind']}", "arm": arm["id"],
                                                      "questions": [], "refused": {}})
            rec["questions"].append({"name": row["question"], "row_ids": row["ids"], "row_len": len(row["ids"]),
                                     "slot": len(row["ids"]) - 1, "groups": row["groups"], "keys": row["keys"],
                                     "probs": probs, "argmax": row["keys"][am], "top2_margin": margin_of(probs),
                                     "near_tie": margin_of(probs) <= NEAR_TIE, "base": row["base"]})
    out = {"schema": "d1-toy-red-oracle/1",
           "what": "the toy module's one forward (fp32, CPU) over every red-arm row, read out with its base question's "
                   "toy groups (the oracle side of readout_gate's pre-check)",
           "toy": doc["toy"], "toy_oracle": {"path": str(toy_oracle), "sha256": sha256_file(Path(toy_oracle))},
           "red_arms": {"path": red["file"], "sha256": red["file_sha256"]}, "records": list(records.values()),
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out) + "\n")
    print(f"toy red oracle: {len(records)} requests -> {out_json}", flush=True)
    return out


def score_red_arm(arm: dict, probs: dict, base_probs: dict, oracle_recs: dict) -> dict:
    """One arm: every row's perturbed p against its base row's p (both from the same instrument) -> the FACTS §7 rule."""
    items = []
    for i, row in enumerate(arm["rows"]):
        rid, k = row["base"]
        q = oracle_recs[rid]["questions"][k]
        p, pb = np.asarray(probs[i], np.float64), np.asarray(base_probs[(rid, k)], np.float64)
        dp = np.abs(p - pb)
        items.append({"request": row["request"], "question": row["question"], "base": f"{rid}:{q['name']}",
                      "tokens": row["row_len"], "oracle_top2_margin": q["top2_margin"], "near_tie": bool(q["near_tie"]),
                      "argmax_moved": int(p.argmax()) != int(pb.argmax()),
                      "max_abs_dp_vs_base": float(dp.max()), "mean_abs_dp_vs_base": float(dp.mean()),
                      "base_probs": [float(x) for x in pb], "perturbed_probs": [float(x) for x in p]})
    moved_far = sum(x["argmax_moved"] and not x["near_tie"] for x in items)
    mdp = max(x["max_abs_dp_vs_base"] for x in items)
    mean_run = float(np.mean([x["mean_abs_dp_vs_base"] for x in items]))
    facts7 = {"a_argmax_non_near_tie": moved_far >= 1, "b_max_abs_dp": mdp > RED_BAR["max_abs_dp"],
              "c_mean_of_run_means": mean_run > RED_BAR["mean_of_run_mean_abs_dp"]}
    return {"id": arm["id"], "kind": arm["kind"], "rows": len(items),
            "argmax_moved": sum(x["argmax_moved"] for x in items), "argmax_moved_non_near_tie": moved_far,
            "max_abs_dp_vs_base": mdp, "mean_of_run_mean_abs_dp_vs_base": mean_run, "red_facts7": facts7,
            "red": any(facts7.values()), "red_fixture_rule": mdp > 0.02, "items": items}


# --------------------------------------------------------------------------- stages
def stage_p1(runner: Runner, recs: dict, rows: list[tuple[str, int]], S: int, hidden_dir: Path | None, log=None) -> list[dict]:
    out = []
    for rid, k in rows:
        q = recs[rid]["questions"][k]
        h, info = runner.chunked(q["row_ids"], S)
        sc = score_probs(runner.readout(h[q["slot"]], q["groups"]), q)
        rec = {"kind": "p1", "id": rid, "q": k, "name": q["name"], "type": q["type"], "source": recs[rid]["source"],
               **info, **sc, "finite": bool(np.isfinite(h).all())}
        ref = oracle_hidden(hidden_dir, rid, k)
        if ref is not None:
            rec["hidden"] = compare_hidden(h, ref)
        out.append(rec)
        if log:
            log(rec)
    return out


def stage_p2(runner: Runner, recs: dict, rows=P2_ROWS) -> list[dict]:
    out = []
    for rid, name in rows:
        k, q = question_of(recs[rid], name)
        a, ia = runner.chunked(q["row_ids"], CHUNK)
        b, ib = runner.oneshot(q["row_ids"])
        c = compare_hidden(a, b)
        out.append({"id": rid, "q": k, "name": q["name"], "T": ia["T"], "chunks": ia["chunks"],
                    "max_abs_diff": c["max_abs_diff"], "min_pos_cos": c["min_pos_cos"],
                    "bit_equal": bool(np.array_equal(a, b)), "seconds": [ia["seconds"], ib["seconds"]]})
        print(f"[p2] {rid}:{q['name']} T={ia['T']} chunked vs one forward max|d| {c['max_abs_diff']:.3e} "
              f"cos {c['min_pos_cos']:.12f}", flush=True)
    return out


def stage_p3(runner: Runner, recs: dict, widths=WIDTHS, rows=P3_ROWS) -> list[dict]:
    out = []
    for rid, name in rows:
        if rid not in recs:
            raise SystemExit(f"P3 row {rid} is not in the oracle")
        k, q = question_of(recs[rid], name)
        ref = None
        for S in widths:
            h, info = runner.chunked(q["row_ids"], S)
            sc = score_probs(runner.readout(h[q["slot"]], q["groups"]), q)
            if ref is None:
                ref = (S, h, sc)
            out.append({"id": rid, "q": k, "name": q["name"], "S": S, "T": info["T"], "chunks": info["chunks"],
                        "seconds": info["seconds"], "vs_oracle_max_abs_dp": sc["max_abs_dp"],
                        "argmax_equal_oracle": sc["argmax_equal"],
                        f"vs_s{ref[0]}_hidden_max_abs_diff": float(np.abs(h.astype(np.float64) - ref[1]).max()),
                        f"vs_s{ref[0]}_max_abs_dp": float(np.abs(np.asarray(sc["probs"]) - np.asarray(ref[2]["probs"])).max())})
    return out


def stage_p2s(runner: Runner, recs: dict, kv_writes, context: int, S: int = CHUNK, rows=P2_ROWS) -> list[dict]:
    """Round 9b's fp32 control: each row through the static form (each kv_write) against the dynamic form, both in
    S-token calls from zero states: max |d|, the lowest position cosine, bit equality; the readout's p of both against
    the oracle."""
    out = []
    for rid, name in rows:
        k, q = question_of(recs[rid], name)
        a, ia = runner.chunked(q["row_ids"], S)
        pa = score_probs(runner.readout(a[q["slot"]], q["groups"]), q)
        for kv in kv_writes:
            b, ib = runner.chunked_static(q["row_ids"], S, context, kv)
            c = compare_hidden(b, a)
            pb = score_probs(runner.readout(b[q["slot"]], q["groups"]), q)
            out.append({"id": rid, "q": k, "name": q["name"], "T": ia["T"], "S": S, "chunks": ia["chunks"],
                        "context": context, "kv_write": kv, "max_abs_diff": c["max_abs_diff"],
                        "rel_max_abs_diff": c["rel_max_abs_diff"], "min_pos_cos": c["min_pos_cos"],
                        "bit_equal": bool(np.array_equal(a, b)), "finite": c["finite"],
                        "dynamic_vs_oracle_max_abs_dp": pa["max_abs_dp"], "static_vs_oracle_max_abs_dp": pb["max_abs_dp"],
                        "static_vs_dynamic_max_abs_dp": float(np.abs(np.asarray(pb["probs"]) - np.asarray(pa["probs"])).max()),
                        "argmax_equal_oracle": [pa["argmax_equal"], pb["argmax_equal"]],
                        "seconds": [ia["seconds"], ib["seconds"]]})
            print(f"[p2s] {rid}:{q['name']} T={ia['T']} static ({kv}, C={context}) vs dynamic max|d| "
                  f"{c['max_abs_diff']:.3e} cos {c['min_pos_cos']:.12f} bit {out[-1]['bit_equal']}", flush=True)
    return out


def stage_p4(runner: Runner, recs: dict, red: dict, S: int = CHUNK) -> dict:
    base = {}
    for rid, k in red["base_rows"]:
        q = recs[rid]["questions"][k]
        h, _ = runner.chunked(q["row_ids"], S)
        base[(rid, k)] = runner.readout(h[q["slot"]], q["groups"])
    arms = []
    for arm in red["arms"]:
        probs = []
        for row in arm["rows"]:
            h, _ = runner.chunked(row["ids"], S)
            probs.append(runner.readout(h[row["slot"]], row["groups"]))
        a = score_red_arm(arm, probs, base, recs)
        arms.append(a)
        print(f"[p4] {a['id']}: argmax moved {a['argmax_moved']}/{a['rows']} max|dp| {a['max_abs_dp_vs_base']:.4f} mean "
              f"{a['mean_of_run_mean_abs_dp_vs_base']:.5f} -> {'RED' if a['red'] else 'NOT RED'}", flush=True)
    return {"rule": RED_RULE, "arms": arms, "all_red": all(a["red"] for a in arms)}


def summarize_p1(p1: list[dict]) -> dict:
    near = [x for x in p1 if x["near_tie"]]
    hid = [x for x in p1 if "hidden" in x]
    return {"questions": len(p1), "argmax_equal": sum(x["argmax_equal"] for x in p1),
            "near_tie_questions": len(near), "argmax_equal_near_tie": sum(x["argmax_equal"] for x in near),
            "max_abs_dp": max((x["max_abs_dp"] for x in p1), default=None),
            "mean_of_run_mean_abs_dp": float(np.mean([x["mean_abs_dp"] for x in p1])) if p1 else None,
            "worst": (lambda w: {"id": w["id"], "name": w["name"], "max_abs_dp": w["max_abs_dp"]})(
                max(p1, key=lambda x: x["max_abs_dp"])) if p1 else None,
            "finite": all(x["finite"] for x in p1),
            "hidden_rows": len(hid), "min_pos_cos": min((x["hidden"]["min_pos_cos"] for x in hid), default=None),
            "hidden_max_abs_diff": max((x["hidden"]["max_abs_diff"] for x in hid), default=None),
            "tokens": int(sum(x["T"] for x in p1)), "padded_tokens": int(sum(x["padded"] for x in p1)),
            "seconds": float(sum(x["seconds"] for x in p1))}


def p1_pass(s: dict) -> bool:
    return (s["argmax_equal"] == s["questions"] and s["max_abs_dp"] is not None and s["max_abs_dp"] <= BAR["max_abs_dp"]
            and s["finite"] and (s["min_pos_cos"] is None or s["min_pos_cos"] >= BAR["min_pos_cos"]))


def versions() -> dict:
    import importlib.metadata as md

    import torch
    out = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
           "platform": platform.platform()}
    for d in ("coreai-torch", "coreai-models", "tokenizers"):
        try:
            out[d] = md.version(d)
        except md.PackageNotFoundError:
            out[d] = None
    return out


def module_record() -> dict:
    return {"path": "conversion/d1/lfm2_d1_decoder.py", "sha256": sha256_file(HERE / "lfm2_d1_decoder.py"),
            "harness": {"path": "conversion/d1/parity_decoder_torch.py", "sha256": sha256_file(Path(__file__).resolve())}}


# --------------------------------------------------------------------------- the toy: P1-P4 in one process
def cmd_toy(args) -> int:
    import export_decoder as ed

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists: records are never overwritten")
    oracle_path = Path(args.toy_oracle)
    if not oracle_path.exists():
        build_toy_oracle(args.toy_seed, oracle_path, Path(args.table) if args.table else None)
    doc, recs = load_oracle(oracle_path)
    if doc["toy"]["seed"] != args.toy_seed:
        raise SystemExit(f"{oracle_path} is the seed-{doc['toy']['seed']} toy, not {args.toy_seed}")
    model = ed.toy_model(args.toy_seed)
    V = model.config.vocab_size
    runner = Runner(model, args.threads, pad_id=ed.toy_fold(PAD_ID, V))
    t0 = time.monotonic()
    p1 = stage_p1(runner, recs, all_rows(recs), CHUNK, oracle_path.parent / "hidden")
    s1 = summarize_p1(p1)
    print(f"[p1] {s1['questions']} rows: argmax {s1['argmax_equal']}/{s1['questions']} max|dp| {s1['max_abs_dp']:.3e} "
          f"min cos {s1['min_pos_cos']}", flush=True)
    p2 = stage_p2(runner, recs)
    p3 = stage_p3(runner, recs, tuple(args.widths))
    red = None
    if args.red:
        tok = host.load_tokenizer(snapshot() / "tokenizer.json")
        rows = red_rows(json.loads((FIXTURES / "red_arms.json").read_text()), recs, tok, fold_vocab=V)
        red = {**stage_p4(runner, recs, rows), "rows_file": rows["file"], "rows_file_sha256": rows["file_sha256"],
               "folded": V}
    p2_max = max(x["max_abs_diff"] for x in p2)
    checks = {"p1": p1_pass(s1), "p2": p2_max <= BAR["p2_max_abs_diff_toy"], "p3_ran": bool(p3),
              "p4_all_red": None if red is None else red["all_red"]}
    rec = {"schema": "d1-decoder-torch-parity-toy/1",
           "what": "the instruments on the toy: the chunk order (S = 16) against the same module's one forward, read "
                   "out through host.py (P1), the hidden rows (P2), the chunk widths (P3), the red arms on fp32 torch (P4)",
           "bar": {**BAR, "toy_note": "P1's oracle is the module's own one forward: equal up to float rounding"},
           "toy": doc["toy"], "oracle": {"path": str(oracle_path), "sha256": sha256_file(oracle_path)},
           "module": module_record(), "threads": args.threads, "versions": versions(),
           "p1": {"summary": s1, "pass": checks["p1"],
                  "rows": [{k: v for k, v in x.items() if k not in ("probs", "probs_oracle")} for x in p1]},
           "p2": {"rows": p2, "max_abs_diff": p2_max, "pass": checks["p2"]},
           "p3": {"widths": list(args.widths), "rows": p3,
                  "max_hidden_diff_vs_s16": {str(S): max((x[f"vs_s{args.widths[0]}_hidden_max_abs_diff"] for x in p3
                                                          if x["S"] == S), default=None) for S in args.widths},
                  "max_abs_dp_vs_oracle": {str(S): max((x["vs_oracle_max_abs_dp"] for x in p3 if x["S"] == S),
                                                       default=None) for S in args.widths}},
           "p4": red, "checks": checks, "seconds": round(time.monotonic() - t0, 1),
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"P2 max|d| {p2_max:.3e}; checks {json.dumps(checks)} -> {out}")
    return 0 if checks["p1"] and checks["p2"] else 1


# --------------------------------------------------------------------------- the model (round 3)
def real_runner(threads: int) -> Runner:
    import torch
    from lfm2_d1_decoder import Lfm2D1Decoder

    model = Lfm2D1Decoder.from_hf(str(snapshot()), target_dtype=torch.float32, fp32_attn_proj=True)
    rep = model.load_report
    if (rep["unread_checkpoint_keys_under_prefix"] or rep["module_tensors_not_in_checkpoint"] or rep["meta_params"]
            or rep["module_has_lm_head_weight"]):
        raise SystemExit(f"load mismatch: {json.dumps(rep)}")
    r = Runner(model, threads)
    r.load_report = rep
    return r


def shard_rows(rows: list[tuple[str, int]], recs: dict, shard: int, shards: int) -> list[tuple[str, int]]:
    """Balance by padded tokens: longest first, each to the lightest shard (deterministic)."""
    load, own = [0] * shards, [[] for _ in range(shards)]
    for rid, k in sorted(rows, key=lambda x: (-recs[x[0]]["questions"][x[1]]["row_len"], x[0], x[1])):
        i = min(range(shards), key=lambda j: (load[j], j))
        own[i].append((rid, k))
        load[i] += -(-recs[rid]["questions"][k]["row_len"] // CHUNK) * CHUNK
    return own[shard]


def cmd_stage(args) -> int:
    doc, recs = load_oracle(Path(args.oracle))
    PARITY.mkdir(parents=True, exist_ok=True)
    runner = real_runner(args.threads)
    meta = {"pid": os.getpid(), "started": now(), "threads": args.threads, "load_report": runner.load_report,
            "module": module_record(), "oracle": {"path": args.oracle, "sha256": sha256_file(Path(args.oracle))}}
    if args.stage == "p1":
        rows = shard_rows(all_rows(recs), recs, args.shard, args.shards)
        path = PARITY / f"p1_shard{args.shard}of{args.shards}.jsonl"
        done = set()
        if path.exists():
            done = {(x["id"], x["q"]) for x in map(json.loads, path.read_text().splitlines()) if x.get("kind") == "p1"}
        with open(path, "a") as f:
            f.write(json.dumps({"kind": "meta", **meta, "rows": len(rows)}) + "\n")

            def log(rec):
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"[p1 {args.shard}/{args.shards}] {rec['id']}:{rec['name']} T={rec['T']} argmax "
                      f"{'ok' if rec['argmax_equal'] else 'NO'} max|dp| {rec['max_abs_dp']:.2e} {rec['seconds']:.1f}s", flush=True)
            stage_p1(runner, recs, [r for r in rows if r not in done], CHUNK,
                     Path(args.hidden_npz) if args.hidden_npz else None, log)
            f.write(json.dumps({"kind": "end", "finished": now()}) + "\n")
    elif args.stage == "p2":
        (PARITY / "p2.json").write_text(json.dumps({**meta, "rows": stage_p2(runner, recs)}, indent=1) + "\n")
    elif args.stage == "p2s":
        kvs = [x for x in args.kv_write.split(",") if x]
        rows = stage_p2s(runner, recs, kvs, args.context)
        worst = max(x["max_abs_diff"] for x in rows)
        rec = {**meta, "schema": "d1-decoder-torch-parity-static/1",
               "what": "round 9b: the static form (lfm2_d1_static.py) against the dynamic form, fp32 CPU, S-token calls "
                       "from zero states, the same weights",
               "bar": {"max_abs_diff": BAR["p2_max_abs_diff"]}, "static_module_sha256": sha256_file(HERE / "lfm2_d1_static.py"),
               "kv_writes": kvs, "context": args.context, "rows": rows, "max_abs_diff": worst,
               "result": "PASS" if worst <= BAR["p2_max_abs_diff"] and all(x["finite"] for x in rows) else "FAIL",
               "finished": now()}
        Path(args.out).write_text(json.dumps(rec, indent=1) + "\n")
        print(f"P2s {rec['result']}: max|d| {worst:.3e} -> {args.out}")
    elif args.stage == "p3":
        (PARITY / "p3.json").write_text(json.dumps({**meta, "widths": args.widths,
                                                   "rows": stage_p3(runner, recs, tuple(args.widths))}, indent=1) + "\n")
    elif args.stage == "p4":
        tok = host.load_tokenizer(snapshot() / "tokenizer.json")
        rows = red_rows(json.loads((FIXTURES / "red_arms.json").read_text()), recs, tok)
        (PARITY / "p4.json").write_text(json.dumps({**meta, **stage_p4(runner, recs, rows)}, indent=1) + "\n")
    return 0


def cmd_merge(args) -> int:
    doc, recs = load_oracle(Path(args.oracle))
    parts = [json.loads(x) for p in sorted(PARITY.glob("p1_shard*of*.jsonl")) for x in p.read_text().splitlines()]
    p1_by = {(x["id"], x["q"]): x for x in parts if x["kind"] == "p1"}
    expected = all_rows(recs)
    p1 = [p1_by[k] for k in expected if k in p1_by]
    s1 = summarize_p1(p1)
    missing = [f"{r}:{k}" for r, k in expected if (r, k) not in p1_by]
    p2 = json.loads((PARITY / "p2.json").read_text()) if (PARITY / "p2.json").exists() else None
    p3 = json.loads((PARITY / "p3.json").read_text()) if (PARITY / "p3.json").exists() else None
    p4 = json.loads((PARITY / "p4.json").read_text()) if (PARITY / "p4.json").exists() else None
    rec = {"schema": "d1-decoder-torch-parity/1",
           "purpose": "the hidden-output decoder module (fp32 torch, S-token chunks from zero states) read out through "
                      "host.py against the provider's fp32 oracle",
           "bar": BAR, "module": module_record(), "versions": versions(),
           "oracle": {"path": args.oracle, "sha256": sha256_file(Path(args.oracle))},
           "p1": {"result": "PASS" if (not missing and p1_pass(s1)) else "FAIL", "summary": s1, "missing_rows": missing,
                  "near_ties": [{k: x[k] for k in ("id", "name", "oracle_top2_margin", "argmax_equal", "max_abs_dp")}
                                for x in p1 if x["near_tie"]],
                  "rows": [{k: v for k, v in x.items() if k not in ("probs", "probs_oracle")} for x in p1]},
           "p2": p2, "p3": p3, "p4": p4, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"P1 {rec['p1']['result']}: {s1['argmax_equal']}/{s1['questions']} max|dp| {s1['max_abs_dp']} -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["p1", "p2", "p2s", "p3", "p4", "merge", "toy"])
    ap.add_argument("--kv-write", default="slice", help="p2s: the static form's KV writes to run (comma list of slice, roll)")
    ap.add_argument("--context", type=int, default=4096, help="p2s: the static form's KV slots C")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--oracle", default=str(ORACLE / "records_oracle.json"))
    ap.add_argument("--hidden-npz", default=None, help="p1: the oracle's hidden/ directory (position cosine)")
    ap.add_argument("--widths", type=lambda s: [int(x) for x in s.split(",")], default=list(WIDTHS),
                    help="p3 / toy: chunk widths, the first is the reference (default 16,32,64)")
    ap.add_argument("--red", action="store_true", help="toy: add P4 (the red arms on the toy module)")
    ap.add_argument("--toy-oracle", default=str(LANE / "oracle_toy" / "records_oracle.json"))
    ap.add_argument("--toy-seed", type=int, default=0)
    ap.add_argument("--table", default=None, help="toy: a toy bundle's head/ to build the toy oracle from")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.stage == "toy":
        args.out = args.out or str(LANE / "results" / "parity_decoder_torch_toy.json")
        return cmd_toy(args)
    if args.stage == "merge":
        args.out = args.out or str(LANE / "results" / "parity_decoder_torch.json")
        if Path(args.out).exists():
            raise SystemExit(f"{args.out} exists: records are never overwritten")
        return cmd_merge(args)
    if args.stage == "p2s":
        if not args.out:
            raise SystemExit("p2s writes to --out")
        if Path(args.out).exists():
            raise SystemExit(f"{args.out} exists: records are never overwritten")
    return cmd_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
