#!/usr/bin/env python3
"""Held-out gate: decider-2b-vision end to end on 500 photo runs that no round used to choose anything.

Round 6 kept decoder layers 0, 2 and 5 fp16 (the rest int8 per-block-32) because that set passed
the self-made fixture. This gate asks whether that choice generalizes, on the 250 photos of the
grid-price table (visual7w 150 + vsr 100, the author's held-out subsets of The Cauldron) at both
grids = 500 runs per bundle, prepared by `heldout_prepare.py` (thumbnailed PNG, Example, the
author's processor ids and `pixel_values`, and the author's fp32 probabilities from
`price/items.jsonl`).

Per run, the shipped path: the PNG -> `host.preprocess` (Pillow BICUBIC to the tile) -> fp16w32
tower `.aimodelc` -> image_embeds (fp32 -> fp16, zero-padded to 256 rows) -> decoder `.aimodelc`
in chunk order (prefill S=16 while a whole chunk fits before the slot, then S=1 up to and
including it; `readout_gate_vision.py --order chunk`), ids from `host.build_ids` with the bundle's
own tokenizer. Letters ids 32.. over the question's options, softmax at T=1 (float64). AOT h16c
GPU, `SpecializationOptions.default()`, fresh zero states per run; a process takes at most 14
runs + a re-run of its first one (15 prompts), whose tower output and slot logits must be
bit-equal (state reset proof).

Bar, per bundle, fixed before any result (round-7 brief):
  1. ids = the author's processor ids, 500/500;
  2. host patches vs the processor's `pixel_values` within 2 levels (1 level = 2/255), 500/500;
  3. letter argmax = the author's on every run whose author top-2 margin is >= 0.02 (runs below
     are near-ties: counted and listed separately, agree or not, never dropped);
  4. max |dp| <= 0.02 over all 500 runs;
  5. mean over runs of the run's mean |dp| <= 0.002.
Bars 1-2 run first on all 500 runs (host only, no GPU); if either fails, the gate stops there and
lists the runs that differ. Ship rule: int8mix_pf16 ships if it meets 3, 4 and 5; if it does not
and fp16_pf16 fails any of 3-5 as well, both are recorded as held-out FAIL. No layer is re-chosen
on these data.

Red checks: bars 1-2 on the first item with one question character edited and one pixel moved by
40 levels (`host.guards`); bars 3-5 on a known-wrong pairing, each bundle's g448 probabilities vs
the author's g256 reference of the same photo (`red_arm`). Each run records the sha256 of its
tower output, so the int8mix vs fp16 table also counts the runs whose decoders got bit-equal
image_embeds.

The transcript holds numbers and dataset row indices only: no image, question or option text.

Run from the worktree root (shared venv, offline):
    HF_HOME=~/code/coreai/_decider2bv/hf HF_HUB_OFFLINE=1 \\
      ../coreai-models/.venv/bin/python conversion/decider_vision/heldout_eval.py run \\
        --bundles ~/code/coreai/_decider2bv/exports/bundles/decider_2b_vision_decode_int8mix_pf16,\\
~/code/coreai/_decider2bv/exports/bundles/decider_2b_vision_decode_fp16_pf16 \\
        --transcript models/decider-2b-vision/gate-decider-2b-vision-heldout.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import host  # noqa: E402
import readout_gate_vision as rg  # noqa: E402  (contract, digests, environment, split)
from _paths import work_path  # noqa: E402

LANE = work_path("_decider2bv")
HELDOUT = LANE / "heldout"
V = host.VOCAB
LETTER0 = host.LETTER_IDS[0]
GRIDS = rg.GRIDS                                   # {"g256": 8, "g448": 14} merged grid per side
SUBSETS = ("visual7w", "vsr")
LEVEL = 1.0 / (255.0 * host.IMAGE_STD)             # one uint8 level in normalized pixel units
BAR = {"ids_equal": "500/500", "pixel_max_levels": 2, "near_tie_margin": 0.02,
       "argmax": "100% of runs with author top-2 margin >= 0.02", "max_abs_dp": 0.02,
       "mean_of_run_mean_abs_dp": 0.002}
SHIP = "decider_2b_vision_decode_int8mix_pf16"
FALLBACK = "decider_2b_vision_decode_fp16_pf16"
TOWER = "fp16w32"


def load_items() -> list[dict]:
    return [json.loads(x) for x in (HELDOUT / "items.jsonl").read_text().splitlines() if x.strip()]


def question(rec: dict) -> list[dict]:
    return [{"text": rec["question"], "options": rec["options"]}]


# --------------------------------------------------------------------------- #
# Bars 1-2: host only
# --------------------------------------------------------------------------- #
def host_check(items: list[dict], tokenizer_json: Path) -> list[dict]:
    """Per run: host ids vs the processor's, host patches vs its pixel_values (in levels)."""
    import tokenizers

    tok = tokenizers.Tokenizer.from_file(str(tokenizer_json))
    out = []
    for rec in items:
        z = np.load(HELDOUT / rec["npz"])
        for arm, g in GRIDS.items():
            ids, slots, start, amount = host.build_ids(True, rec["context"], question(rec), tok, (g, g))
            want = z[f"{arm}_input_ids"].tolist()
            got = host.to_processor_ids(ids)
            first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
            if first is None and len(got) != len(want):
                first = min(len(got), len(want))
            patches = host.preprocess(HELDOUT / rec["png"], g)
            pv = z[f"{arm}_pixel_values"]
            d = np.abs(patches.astype(np.float64) - pv.astype(np.float64)) if patches.shape == pv.shape else None
            out.append({"key": rec["key"], "grid": arm, "tokens_host": len(ids), "tokens_processor": len(want),
                        "ids_equal": got == want, "first_id_mismatch": first,
                        "slots_equal": slots == z[f"{arm}_slot_idx"].tolist(),
                        "start": int(start), "amount": int(amount),
                        "patches_shape_equal": d is not None,
                        "pixel_max_levels": float(d.max() / LEVEL) if d is not None else None,
                        "pixel_elements_off": int((d > LEVEL / 2).sum()) if d is not None else None})
    return out


def host_bar(rows: list[dict]) -> dict:
    ok_ids = [r for r in rows if r["ids_equal"] and r["slots_equal"]]
    ok_px = [r for r in rows if r["patches_shape_equal"] and r["pixel_max_levels"] <= BAR["pixel_max_levels"] + 1e-6]
    levels = [r["pixel_max_levels"] for r in rows if r["pixel_max_levels"] is not None]
    return {"runs": len(rows), "ids_equal": len(ok_ids), "pixels_within_2_levels": len(ok_px),
            "pixel_max_levels": max(levels) if levels else None,
            "pixel_runs_exact": sum(r["pixel_elements_off"] == 0 for r in rows),
            "bar1_ok": len(ok_ids) == len(rows), "bar2_ok": len(ok_px) == len(rows),
            "differing": [r for r in rows if r not in ok_ids or r not in ok_px]}


def host_guards(items: list[dict], tokenizer_json: Path) -> dict:
    """Bars 1-2 must be able to fail: the first item with one character of its question edited, and
    with one pixel moved by 40 levels, at both grids."""
    import tokenizers
    from PIL import Image

    tok = tokenizers.Tokenizer.from_file(str(tokenizer_json))
    rec = items[0]
    z = np.load(HELDOUT / rec["npz"])
    q = rec["question"]
    edited = [{"text": q[:-1] + ("x" if q[-1] != "x" else "y"), "options": rec["options"]}]
    px = np.asarray(Image.open(HELDOUT / rec["png"]).convert("RGB")).copy()
    px[0, 0, 0] = (int(px[0, 0, 0]) + 40) % 256
    out = {"item": rec["key"], "edit": "last character of the question", "pixel": "(0, 0) red +40 levels (mod 256)"}
    for arm, g in GRIDS.items():
        ids, *_ = host.build_ids(True, rec["context"], edited, tok, (g, g))
        d = np.abs(host.preprocess(px, g).astype(np.float64) - z[f"{arm}_pixel_values"].astype(np.float64))
        out[arm] = {"edited_ids_equal": host.to_processor_ids(ids) == z[f"{arm}_input_ids"].tolist(),
                    "perturbed_pixel_max_levels": float(d.max() / LEVEL)}
    out["red"] = all(not out[a]["edited_ids_equal"] and out[a]["perturbed_pixel_max_levels"] > BAR["pixel_max_levels"]
                     for a in GRIDS)
    return out


# --------------------------------------------------------------------------- #
# Worker: one process, <= 14 runs + the reset re-run
# --------------------------------------------------------------------------- #
def worker(spec_path: Path) -> int:
    import coreai.runtime as rt
    import tokenizers

    spec = json.loads(spec_path.read_text())
    items = {r["key"]: r for r in load_items()}
    tok = tokenizers.Tokenizer.from_file(spec["tokenizer"])
    C = int(spec["chunk"])
    g = int(spec["tower"]["grid"])
    arm = spec["tower"]["arm"]

    def nd(a):
        return rt.NDArray(np.ascontiguousarray(a))

    out: dict = {"spec": spec, "pid": os.getpid(), "runs": []}
    store: dict[str, np.ndarray] = {}

    async def go() -> None:
        model = await rg.maybe(rt.AIModel.load(spec["aimodelc"], rt.SpecializationOptions.default()))
        fn = await rg.maybe(model.load_function("main"))
        pf = await rg.maybe(model.load_function("prefill"))
        desc, pdesc = rg.fn_desc(fn), rg.fn_desc(pf)
        rg.check_contract(desc, "main", 1)
        rg.check_contract(pdesc, "prefill", C)
        if pdesc["states"] != desc["states"]:
            raise SystemExit(f"prefill states {pdesc['states']} != main states {desc['states']}")
        out["function_names"] = list(getattr(model, "function_names", []) or [])
        out["descriptor"], out["prefill_descriptor"] = desc, pdesc
        n_max, hidden = rg.CONTRACT_INPUTS["image_embeds"][0]
        tm = await rg.maybe(rt.AIModel.load(spec["tower"]["aimodelc"], rt.SpecializationOptions.default()))
        tower_fn = await rg.maybe(tm.load_function(tm.function_names[0]))
        td = tower_fn.desc
        out["tower_descriptor"] = {"inputs": {n: rg.dsc(td.input_descriptor(n)) for n in td.input_names},
                                   "outputs": {n: rg.dsc(td.output_descriptor(n)) for n in td.output_names}}
        tower_in_dtype = np.dtype(out["tower_descriptor"]["inputs"]["patches"][1])

        def fresh_state() -> dict:
            return {n: nd(np.zeros([spec["max_ctx"] if s < 0 else s for s in shape], np.dtype(dt)))
                    for n, (shape, dt) in desc["states"].items()}

        async def one(key: str) -> tuple[dict, dict[str, np.ndarray]]:
            rec = items[key]
            z = np.load(HELDOUT / rec["npz"])
            ids, slots, start, amount = host.build_ids(True, rec["context"], question(rec), tok, (g, g))
            patches = host.preprocess(HELDOUT / rec["png"], g)
            pv = z[f"{arm}_pixel_values"]
            tout = await rg.maybe(tower_fn(inputs={"patches": nd(patches.astype(tower_in_dtype))}))
            tower_emb = np.asarray(tout["image_embeds"].numpy()).astype(np.float32)
            if tower_emb.shape != (g * g, hidden):
                raise SystemExit(f"{key}/{arm}: tower output {tower_emb.shape}")
            emb = np.zeros((n_max, hidden), np.float16)
            emb[: g * g] = tower_emb.astype(np.float16)
            rc = np.zeros((n_max, 2), np.int32)
            k = np.arange(g * g)
            rc[: g * g, 0], rc[: g * g, 1] = k // g, k % g
            static = {"image_embeds": nd(emb), "image_rc": nd(rc),
                      "rope_shift_start": nd(np.array([start], np.int32)),
                      "rope_shift_amount": nd(np.array([amount], np.int32))}
            state = fresh_state()
            got, read_from, kinds = [], [], []
            cursor = 0
            for sl in sorted(slots):                  # the readout_gate_vision.py chunk order
                while sl - cursor + 1 >= C:
                    res = await rg.maybe(pf(inputs={
                        "input_ids": nd(np.array([ids[cursor:cursor + C]], np.int32)),
                        "position_ids": nd(np.arange(cursor + C, dtype=np.int32)[None]), **static}, state=state))
                    if cursor + C - 1 == sl:
                        lg = np.asarray(res["logits"].numpy())
                        assert lg.shape == (1, 1, V), lg.shape
                        got.append(lg[0, -1].copy())
                        read_from.append("prefill")
                    kinds.append(1)
                    cursor += C
                while cursor <= sl:
                    res = await rg.maybe(fn(inputs={
                        "input_ids": nd(np.array([[ids[cursor]]], np.int32)),
                        "position_ids": nd(np.arange(cursor + 1, dtype=np.int32)[None]), **static}, state=state))
                    if cursor == sl:
                        lg = np.asarray(res["logits"].numpy())
                        assert lg.shape == (1, 1, V), lg.shape
                        got.append(lg[0, -1].copy())
                        read_from.append("main")
                    kinds.append(0)
                    cursor += 1
            if cursor != len(ids):
                raise SystemExit(f"{key}/{arm}: the row does not end at its last slot ({cursor} of {len(ids)})")
            full = np.stack(got)                                        # [slots, V] fp16
            top5 = np.argsort(-full.astype(np.float32), axis=-1, kind="stable")[:, :5]
            r = {"key": key, "grid": arm, "tokens": len(ids), "slots": slots, "start": int(start), "amount": int(amount),
                 "ids_equal_processor": host.to_processor_ids(ids) == z[f"{arm}_input_ids"].tolist(),
                 "pixel_max_levels": float(np.abs(patches.astype(np.float64) - pv.astype(np.float64)).max() / LEVEL),
                 "prefill_calls": kinds.count(1), "main_calls": kinds.count(0), "slot_read_from": read_from,
                 "tower_finite": bool(np.isfinite(tower_emb).all()),
                 "tower_sha256": hashlib.sha256(np.ascontiguousarray(tower_emb).tobytes()).hexdigest(),
                 "logits_finite": bool(np.isfinite(full.astype(np.float32)).all())}
            arrays = {"full": full, "tower": tower_emb,
                      "letters": full[:, LETTER0:LETTER0 + host.MAX_OPTIONS].copy(),
                      "top5_ids": top5.astype(np.int64),
                      "top5_logits": np.take_along_axis(full, top5, axis=-1)}
            return r, arrays

        keys = [k for k, _ in spec["runs"]]
        first = None
        for i, key in enumerate(keys + [keys[0]]):
            r, arrays = await one(key)
            if i < len(keys):
                r["index"] = i
                out["runs"].append(r)
                for name in ("letters", "top5_ids", "top5_logits"):
                    store[f"{i:02d}__{name}"] = arrays[name]
                if i == 0:
                    first = arrays
                print(f"  [{os.getpid()}] {key}/{arm}: {r['tokens']} tok, {r['prefill_calls']} prefill + "
                      f"{r['main_calls']} main", flush=True)
            else:
                same = {n: bool(np.array_equal(first[n], arrays[n])) for n in ("full", "tower")}
                out["reset_check"] = {"run": key, "bit_equal": all(same.values()), "per_array": same}
                print(f"  [{os.getpid()}] reset re-run {key}/{arm}: bit-equal {same}", flush=True)

    asyncio.run(go())
    prefix = Path(spec["out"])
    np.savez(prefix.with_suffix(".npz"), **store)
    prefix.with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_shards(tag: str, shards: list[dict]) -> list[dict]:
    """Worker processes one after another; a shard whose record exists for the same spec is reused."""
    got = []
    for sp in shards:
        prefix = Path(sp["out"])
        spec_path, js = prefix.with_suffix(".spec.json"), prefix.with_suffix(".json")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        if js.exists() and prefix.with_suffix(".npz").exists() and json.loads(js.read_text())["spec"] == sp:
            print(f"[{tag}] {prefix.name}: reusing the existing record", flush=True)
            got.append(json.loads(js.read_text()))
            continue
        spec_path.write_text(json.dumps(sp, indent=1) + "\n")
        print(f"[{tag}] {prefix.name}: {len(sp['runs'])} runs + reset re-run", flush=True)
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--spec", str(spec_path)])
        if proc.returncode != 0 or not js.exists():
            got.append({"spec": sp, "failed": True, "returncode": proc.returncode})
            print(f"[{tag}] {prefix.name}: worker FAILED (exit {proc.returncode})", flush=True)
            continue
        got.append(json.loads(js.read_text()))
    return got


def softmax64(x) -> np.ndarray:
    return rg.softmax64(np.asarray(x, np.float64))


def score(items: dict, shard_recs: list[dict]) -> tuple[dict, list[dict]]:
    """(key, grid) -> per-run numbers vs the author's probabilities; the process records."""
    runs, procs = {}, []
    for sr in shard_recs:
        prefix = Path(sr["spec"]["out"])
        entry = {"shard": prefix.name, "runs": [f"{k}/{a}" for k, a in sr["spec"]["runs"]],
                 "prompts": len(sr["spec"]["runs"]) + 1}
        if sr.get("failed"):
            procs.append(dict(entry, failed=True, returncode=sr["returncode"]))
            continue
        z = np.load(prefix.with_suffix(".npz"))
        procs.append(dict(entry, pid=sr["pid"], reset_check=sr["reset_check"],
                          npz=str(prefix.with_suffix(".npz")), npz_sha256=rg.sha256_file(prefix.with_suffix(".npz"))))
        for r in sr["runs"]:
            key, arm = r["key"], r["grid"]
            ref = items[key]["arms"][arm]
            n = items[key]["nopts"]
            letters = z[f"{r['index']:02d}__letters"][0]                # the run's one slot, fp16
            p = softmax64(letters[:n].astype(np.float64))
            po = np.asarray(ref["ref_probs"], np.float64)
            d = np.abs(p - po)
            top1 = int(z[f"{r['index']:02d}__top5_ids"][0, 0])
            runs[(key, arm)] = {
                "probs": p.tolist(), "argmax": int(p.argmax()), "argmax_equal": int(p.argmax()) == ref["ref_argmax"],
                "abs_dp": d.tolist(), "max_abs_dp": float(d.max()), "mean_abs_dp": float(d.mean()),
                "letter_logits_fp16": letters[:n].astype(np.float64).tolist(),
                "full_vocab_top1_id": top1, "full_vocab_top1_is_argmax_letter": top1 == LETTER0 + int(p.argmax()),
                "ids_equal_processor": r["ids_equal_processor"], "pixel_max_levels": r["pixel_max_levels"],
                "tokens": r["tokens"], "prefill_calls": r["prefill_calls"], "main_calls": r["main_calls"],
                "slot_read_from": r["slot_read_from"][0], "finite": r["tower_finite"] and r["logits_finite"],
                "tower_sha256": r["tower_sha256"],
                "shard": prefix.name, "npz_key": f"{r['index']:02d}"}
    return runs, procs


def quantiles(a) -> dict:
    a = np.asarray(a, np.float64)
    return {"n": int(a.size), "p50": float(np.quantile(a, 0.5)), "p90": float(np.quantile(a, 0.9)),
            "p99": float(np.quantile(a, 0.99)), "max": float(a.max())}


def summarize(sel: list[tuple[dict, dict]]) -> dict:
    """sel = [(item-run meta, bundle run)]: the table-1 numbers."""
    decisive = [(m, b) for m, b in sel if not m["near_tie"]]
    near = [(m, b) for m, b in sel if m["near_tie"]]
    worst = max(sel, key=lambda x: x[1]["max_abs_dp"])
    return {"runs": len(sel),
            "decisive_runs": len(decisive), "decisive_argmax_equal": sum(b["argmax_equal"] for _, b in decisive),
            "near_tie_runs": len(near), "near_tie_argmax_equal": sum(b["argmax_equal"] for _, b in near),
            "max_abs_dp": worst[1]["max_abs_dp"],
            "mean_of_run_mean_abs_dp": float(np.mean([b["mean_abs_dp"] for _, b in sel])),
            "max_abs_dp_quantiles": quantiles([b["max_abs_dp"] for _, b in sel]),
            "worst_run": {"key": worst[0]["key"], "grid": worst[0]["grid"], "max_abs_dp": worst[1]["max_abs_dp"],
                          "ref_top2_margin": worst[0]["ref_top2_margin"]},
            "full_vocab_top1_is_argmax_letter": sum(b["full_vocab_top1_is_argmax_letter"] for _, b in sel),
            "finite_all": all(b["finite"] for _, b in sel)}


def bundle_verdict(s: dict, hb: dict, resets_ok: bool, complete: bool) -> tuple[dict, dict]:
    bars = {"1_ids_equal": hb["bar1_ok"] and s["ids_equal_processor"] == s["runs"],
            "2_pixels_within_2_levels": hb["bar2_ok"] and s["pixels_within_2_levels"] == s["runs"],
            "3_argmax_decisive": s["decisive_argmax_equal"] == s["decisive_runs"],
            "4_max_abs_dp": s["max_abs_dp"] <= BAR["max_abs_dp"],
            "5_mean_of_run_mean_abs_dp": s["mean_of_run_mean_abs_dp"] <= BAR["mean_of_run_mean_abs_dp"]}
    integrity = {"all_500_runs_read": complete, "reset_bit_equal_all_processes": resets_ok,
                 "finite": s["finite_all"]}
    return bars, integrity


def main_run(args) -> int:
    items_list = load_items()
    items = {r["key"]: r for r in items_list}
    assert len(items) == 250, len(items)
    bundles = [Path(b).expanduser().resolve() for b in args.bundles.split(",")]
    metas = {b: json.loads((b / "metadata.json").read_text()) for b in bundles}
    names = [metas[b]["name"] for b in bundles]
    runs_meta = []
    for rec in items_list:
        for arm in GRIDS:
            a = rec["arms"][arm]
            runs_meta.append({"key": rec["key"], "dataset": rec["dataset"], "row_index": rec["row_index"],
                              "grid": arm, "nopts": rec["nopts"], "gold": rec["gold"],
                              "ref_probs": a["ref_probs"], "ref_argmax": a["ref_argmax"],
                              "ref_top2_margin": a["ref_top2_margin"],
                              "near_tie": a["ref_top2_margin"] < BAR["near_tie_margin"]})
    work = Path(args.work_dir).expanduser() if args.work_dir else HELDOUT / "eval"
    prep = json.loads((HELDOUT / "prepare.json").read_text())
    record: dict = {
        "schema": "coreai-decider-vision-heldout-gate/1",
        "gate": "held-out end to end: thumbnailed photo -> host.preprocess (Pillow BICUBIC to the tile) -> fp16w32 "
                "tower .aimodelc -> image_embeds (fp32 -> fp16) -> decoder .aimodelc, chunk order (prefill S=16 + "
                "S=1 remainder), ids from host.build_ids with the bundle's tokenizer; vs the author's fp32",
        "purpose": "round 6 chose the fp16 layer set [0, 2, 5] on the self-made fixture; these 500 runs were used "
                   "by no selection (layer set, bundle or bar)",
        "bar": dict(BAR, preregistered="fixed in the round-7 brief before any held-out result; not changed",
                    level="1 level = 2/255 in normalized pixel units (one uint8 step)",
                    mean_of_run_mean_abs_dp_definition="mean over runs of the mean |dp| over the run's options",
                    near_tie="author top-2 margin < 0.02: counted and listed separately, never dropped; bars 4 and 5 "
                             "cover every run"),
        "ship_rule": "int8mix_pf16 ships if it meets bars 3, 4 and 5 (1 and 2 are the host's); if it does not and "
                     "fp16_pf16 fails any of 3-5 too, both are recorded as held-out FAIL; no layer re-selection "
                     "on these data",
        "dataset": dict(prep["dataset"], redistribution="the images are not redistributed: they stay in the lane "
                        "work directory; this transcript holds dataset row indices and numbers only (no image, "
                        "question or option text)"),
        "reference": prep["reference"],
        "inputs": {"prepare_json": {"path": str(HELDOUT / "prepare.json"), "sha256": rg.sha256_file(HELDOUT / "prepare.json")},
                   "items_jsonl": {"path": str(HELDOUT / "items.jsonl"), "sha256": rg.sha256_file(HELDOUT / "items.jsonl")},
                   "script": "conversion/decider_vision/heldout_prepare.py",
                   "processor": prep["processor"], "versions": prep["versions"], "source": prep["source"]},
        "environment": dict(rg.env_record(), gpu="shared with other sessions; no times are reported by this gate"),
    }

    # Bars 1-2 on all 500 runs, per bundle tokenizer, before any GPU run.
    host_rows, host_bars = {}, {}
    for b in bundles:
        name = metas[b]["name"]
        rows = host_check(items_list, b / "tokenizer" / "tokenizer.json")
        host_rows[name] = rows
        host_bars[name] = host_bar(rows)
        hb = host_bars[name]
        print(f"host {name}: ids {hb['ids_equal']}/{hb['runs']} pixels<=2 levels {hb['pixels_within_2_levels']}/"
              f"{hb['runs']} (max {hb['pixel_max_levels']:.4f} levels, exact {hb['pixel_runs_exact']})", flush=True)
    record["host"] = {"preprocess": "host.preprocess(png, grid) resize='pil'",
                      "ids": "host.build_ids(True, context, [question], bundle tokenizer, (g, g)) -> to_processor_ids",
                      "per_bundle": {n: {k: v for k, v in hb.items() if k != "differing"} for n, hb in host_bars.items()},
                      "differing_runs": {n: hb["differing"] for n, hb in host_bars.items() if hb["differing"]},
                      "guards": host_guards(items_list, bundles[0] / "tokenizer" / "tokenizer.json")}
    print(f"host guards (bars 1-2 can fail): {json.dumps(record['host']['guards'])}", flush=True)
    if not all(hb["bar1_ok"] and hb["bar2_ok"] for hb in host_bars.values()) or args.host_only:
        stopped = not all(hb["bar1_ok"] and hb["bar2_ok"] for hb in host_bars.values())
        record["result"] = "STOPPED: bar 1 or 2 failed (host differs from the author)" if stopped else "host only"
        record["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        Path(args.transcript).parent.mkdir(parents=True, exist_ok=True)
        Path(args.transcript).write_text(json.dumps(record, indent=1) + "\n")
        print(f"{record['result']}; transcript: {args.transcript}")
        return 2 if stopped else 0

    # Towers: one per grid, the same files for both bundles.
    towers = {}
    for arm in GRIDS:
        c = sorted((LANE / "exports" / rg.TOWERS[TOWER].format(arm=arm)).glob("*.h16c.aimodelc"))
        assert len(c) == 1, c
        towers[arm] = c[0]
    record["towers"] = {f"{TOWER}/{arm}": rg.tree_digest(p) for arm, p in towers.items()}

    per_bundle, bundle_runs = {}, {}
    for b in bundles:
        meta = metas[b]
        name = meta["name"]
        order, chunk = rg.order_of(argparse.Namespace(order="chunk"), meta)
        aimodelc = rg.aimodelc_for(b)
        shards = []
        for arm, g in GRIDS.items():
            runs = [[k, arm] for k in items]
            for i, part in enumerate(rg.split(runs)):
                shards.append({"aimodelc": str(aimodelc), "chunk": chunk,
                               "max_ctx": meta["language"]["max_context_length"],
                               "tokenizer": str(b / "tokenizer" / "tokenizer.json"),
                               "tower": {"aimodelc": str(towers[arm]), "variant": TOWER, "grid": g, "arm": arm},
                               "runs": part, "out": str(work / name / f"{arm}_shard_{i:02d}")})
        recs = run_shards(f"heldout {name}", shards)
        runs, procs = score(items, recs)
        bundle_runs[name] = runs
        sel = [(m, runs[(m["key"], m["grid"])]) for m in runs_meta if (m["key"], m["grid"]) in runs]
        complete = len(sel) == len(runs_meta)
        s = summarize(sel)
        s["ids_equal_processor"] = sum(b_["ids_equal_processor"] for _, b_ in sel)
        s["pixels_within_2_levels"] = sum(b_["pixel_max_levels"] <= BAR["pixel_max_levels"] + 1e-6 for _, b_ in sel)
        s["missing_runs"] = [f"{m['key']}/{m['grid']}" for m in runs_meta if (m["key"], m["grid"]) not in runs]
        s["slots_read_from_prefill"] = sum(b_["slot_read_from"] == "prefill" for _, b_ in sel)
        resets_ok = all(p.get("reset_check", {}).get("bit_equal", False) for p in procs)
        bars, integrity = bundle_verdict(s, host_bars[name], resets_ok, complete)
        near = [{"key": m["key"], "grid": m["grid"], "ref_top2_margin": m["ref_top2_margin"],
                 "ref_argmax": m["ref_argmax"], "argmax": b_["argmax"], "agree": b_["argmax_equal"],
                 "ref_probs": m["ref_probs"], "probs": b_["probs"]} for m, b_ in sel if m["near_tie"]]
        by = {}
        for sub in SUBSETS:
            for arm in GRIDS:
                part = [(m, b_) for m, b_ in sel if m["dataset"] == sub and m["grid"] == arm]
                if part:
                    by[f"{sub}/{arm}"] = summarize(part)
        mismatches = [{"key": m["key"], "grid": m["grid"], "ref_top2_margin": m["ref_top2_margin"],
                       "ref_argmax": m["ref_argmax"], "argmax": b_["argmax"]}
                      for m, b_ in sel if not b_["argmax_equal"]]
        per_bundle[name] = {
            "bundle": rg.bundle_record(b, aimodelc), "order": order, "chunk": chunk,
            "readout": rg.READOUT["chunk"].format(C=chunk), "tower": TOWER, "processes": procs,
            "summary": s, "by_dataset_grid": by, "near_tie_runs": near, "argmax_mismatches": mismatches,
            "bars": bars, "integrity": integrity,
            "result": "PASS" if all(bars.values()) and all(integrity.values()) else "FAIL"}
        print(f"{per_bundle[name]['result']}: {name} runs {s['runs']}/{len(runs_meta)} ids {s['ids_equal_processor']} "
              f"argmax decisive {s['decisive_argmax_equal']}/{s['decisive_runs']} near-tie "
              f"{s['near_tie_argmax_equal']}/{s['near_tie_runs']} max|dp| {s['max_abs_dp']:.6f} mean "
              f"{s['mean_of_run_mean_abs_dp']:.6f} q {json.dumps(s['max_abs_dp_quantiles'])} reset {resets_ok}",
              flush=True)
        for k, v in {**bars, **integrity}.items():
            print(f"  {k}: {'ok' if v else 'FAIL'}", flush=True)
    record["bundles"] = per_bundle

    # int8mix vs fp16, run by run (same tower, same inputs).
    if SHIP in bundle_runs and FALLBACK in bundle_runs:
        a, f = bundle_runs[SHIP], bundle_runs[FALLBACK]
        both = [m for m in runs_meta if (m["key"], m["grid"]) in a and (m["key"], m["grid"]) in f]
        diffs = [(m, float(np.max(np.abs(np.asarray(a[(m["key"], m["grid"])]["probs"])
                                         - np.asarray(f[(m["key"], m["grid"])]["probs"]))))) for m in both]
        ms = [float(np.mean(np.abs(np.asarray(a[(m["key"], m["grid"])]["probs"])
                                   - np.asarray(f[(m["key"], m["grid"])]["probs"])))) for m in both]

        def part(sel):
            return {"runs": len(sel), "max_abs_dp": quantiles([d for _, d in sel]),
                    "argmax_equal": sum(a[(m["key"], m["grid"])]["argmax"] == f[(m["key"], m["grid"])]["argmax"]
                                        for m, _ in sel)}
        worst = max(diffs, key=lambda x: x[1])
        record["int8mix_vs_fp16"] = dict(
            part(diffs), definition="per run max |p_int8mix_pf16 - p_fp16_pf16| over the options",
            tower_output_bit_equal_runs=sum(a[(m["key"], m["grid"])]["tower_sha256"] == f[(m["key"], m["grid"])]["tower_sha256"]
                                            for m in both),
            mean_of_run_mean_abs_dp=float(np.mean(ms)),
            worst={"key": worst[0]["key"], "grid": worst[0]["grid"], "max_abs_dp": worst[1]},
            by_dataset_grid={f"{sub}/{arm}": part([(m, d) for m, d in diffs if m["dataset"] == sub and m["grid"] == arm])
                             for sub in SUBSETS for arm in GRIDS},
            dp_vs_author_max={n: per_bundle[n]["summary"]["max_abs_dp"] for n in (SHIP, FALLBACK)})

    # Red arm (no GPU): each bundle's g448 probabilities scored against the author's g256 reference of the
    # same photo. The bars must go red on that wrong pairing, or they cannot see a real difference either.
    red = {}
    for name, runs in bundle_runs.items():
        sel = []
        for m in runs_meta:
            if m["grid"] != "g448" or (m["key"], "g448") not in runs:
                continue
            ref = items[m["key"]]["arms"]["g256"]
            p = np.asarray(runs[(m["key"], "g448")]["probs"])
            d = np.abs(p - np.asarray(ref["ref_probs"]))
            sel.append({"near_tie": ref["ref_top2_margin"] < BAR["near_tie_margin"],
                        "argmax_equal": int(p.argmax()) == ref["ref_argmax"], "max": float(d.max()),
                        "mean": float(d.mean())})
        if sel:
            dec = [x for x in sel if not x["near_tie"]]
            r = {"runs": len(sel), "decisive_argmax_equal": sum(x["argmax_equal"] for x in dec), "decisive_runs": len(dec),
                 "max_abs_dp": max(x["max"] for x in sel), "mean_of_run_mean_abs_dp": float(np.mean([x["mean"] for x in sel]))}
            r["bars_3_4_5"] = {"3": r["decisive_argmax_equal"] == r["decisive_runs"],
                               "4": r["max_abs_dp"] <= BAR["max_abs_dp"],
                               "5": r["mean_of_run_mean_abs_dp"] <= BAR["mean_of_run_mean_abs_dp"]}
            r["result"] = "RED" if not all(r["bars_3_4_5"].values()) else "NOT RED"
            red[name] = r
    record["red_arm"] = {"what": "bundle g448 probs vs the author's g256 reference of the same photo (a known-different "
                                 "pairing: the grid-price table has g448 vs g256 argmax agreement below 1)",
                         "per_bundle": red}

    # Ship rule.
    def ok(n, keys=None):
        bars = per_bundle.get(n, {}).get("bars", {})
        integ = per_bundle.get(n, {}).get("integrity", {})
        return bool(bars) and all(v for k, v in bars.items() if keys is None or k in keys) and all(integ.values())

    def ok345(n):
        return ok(n, ("3_argmax_decisive", "4_max_abs_dp", "5_mean_of_run_mean_abs_dp"))
    host_ok = all(hb["bar1_ok"] and hb["bar2_ok"] for hb in host_bars.values())
    if not host_ok:
        concl = "host bars failed"
    elif ok(SHIP):
        concl = f"{SHIP} meets bars 1-5 on the held-out runs: ship it (ship rule)"
    elif ok345(FALLBACK):
        concl = f"{SHIP} held-out FAIL; {FALLBACK} meets bars 3-5 (the rule names no ship for this case: the supervisor decides)"
    else:
        concl = f"both held-out FAIL ({SHIP} and {FALLBACK})"
    record["ship_rule_result"] = concl

    # Per-run table: numbers and row indices only.
    table = []
    for m in runs_meta:
        row = {k: m[k] for k in ("key", "dataset", "row_index", "grid", "nopts", "gold", "ref_probs", "ref_argmax",
                                 "ref_top2_margin", "near_tie")}
        hr = next(x for x in host_rows[names[0]] if x["key"] == m["key"] and x["grid"] == m["grid"])
        row.update({"tokens": hr["tokens_host"], "ids_equal": all(
            next(x for x in host_rows[n] if x["key"] == m["key"] and x["grid"] == m["grid"])["ids_equal"] for n in names),
            "pixel_max_levels": hr["pixel_max_levels"], "bundles": {}})
        for n, runs in bundle_runs.items():
            b_ = runs.get((m["key"], m["grid"]))
            if b_ is not None:
                row["bundles"][n] = {k: b_[k] for k in ("probs", "argmax", "argmax_equal", "abs_dp", "max_abs_dp",
                                                        "mean_abs_dp", "letter_logits_fp16", "full_vocab_top1_id",
                                                        "slot_read_from", "shard", "npz_key")}
        if SHIP in row["bundles"] and FALLBACK in row["bundles"]:
            row["int8mix_vs_fp16_max_abs_dp"] = float(np.max(np.abs(np.asarray(row["bundles"][SHIP]["probs"])
                                                                    - np.asarray(row["bundles"][FALLBACK]["probs"]))))
        table.append(row)
    record["runs"] = table
    record["result"] = {n: per_bundle[n]["result"] for n in per_bundle}
    record["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    Path(args.transcript).parent.mkdir(parents=True, exist_ok=True)
    Path(args.transcript).write_text(json.dumps(record, indent=1) + "\n")
    print(f"ship rule: {concl}")
    if "int8mix_vs_fp16" in record:
        print(f"int8mix vs fp16: {json.dumps({k: record['int8mix_vs_fp16'][k] for k in ('runs', 'max_abs_dp', 'argmax_equal', 'mean_of_run_mean_abs_dp', 'worst')})}")
    print(f"red arm: {json.dumps({n: r['result'] for n, r in red.items()})}")
    print(f"transcript: {args.transcript}")
    return 0 if all(p["result"] == "PASS" for p in per_bundle.values()) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ar = sub.add_parser("run", help="bars 1-2 (host), then 500 GPU runs per bundle and bars 3-5")
    ar.add_argument("--bundles", required=True, help="comma list of bundle dirs (with a prefill function)")
    ar.add_argument("--transcript", required=True)
    ar.add_argument("--work-dir", help=f"shard files (default {HELDOUT / 'eval'})")
    ar.add_argument("--host-only", action="store_true", help="bars 1-2 only, no GPU")
    aw = sub.add_parser("worker")
    aw.add_argument("--spec", required=True)
    args = ap.parse_args()
    if args.cmd == "worker":
        return worker(Path(args.spec))
    return main_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
