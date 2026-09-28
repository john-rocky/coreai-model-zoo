#!/usr/bin/env python3
"""Swift read-out gate: the `decider-vision` CLI (apps/DeciderVision) against the author's fp32 oracle, the Python host
reference and the Python read-out of the same bundle.

The Swift side runs everything itself — ImageIO decode, `host.resize_bicubic`'s bicubic, the patches, the tower, the
author's prompt with swift-transformers' tokenizer, the slot rule, the decoder's chunk order, the letter softmax — and
writes one JSON per fixture pass (`decider-vision fixture`) with its dump directory (resized tiles, tower outputs, full
slot logits). This script only reads those files and scores them.

  score   S1-S4 into the transcript, from one fixture pass per asset kind:
          S1 ids (V+k -> <|image_pad|>) and slots = the round-1 oracle's, every g256 / g448 / text run (76);
             rope_shift_amount = the oracle's rope_shift on every image run
          S2 every (image, grid) tile the pass resized: Swift's RGB8 - host.resize_bicubic in levels (bar max <= 1,
             expected 0), the decoded RGB's sha256 = fixtures/meta.json rgb_sha256, the patches' sha256 = host.patchify
             of the reference tile; Pillow's own resize is listed beside it (values only)
          S3 (--aot) / S4 (--jit) the round-4 bar (readout_gate_vision.BAR, unchanged): all runs, slots, letter argmax,
             full-vocabulary top-1 = the oracle's argmax letter, max |dp| <= 0.02, mean of run means <= 0.002, finite,
             the pass's reset re-run bit-equal; plus the Python read-out of the same bundle: the round-6 e2e arm (image
             runs: same tower, same decoder, Pillow's resize) and the round-6 b1 transcript (text runs: identical
             inputs), per slot |dp|, letter logits and full logits bit-equal, tower output bit-equal
  ask     C: `decider-vision ask --json` beside the same question read by Python (host.build_ids, host.preprocess with
          Pillow's resize and with resize_bicubic, the same tower and decoder .aimodelc, the chunk order)
  timing  D: the timed fixture passes (Release, --asset aot) and the GPU lock record -> per arm (g256 / g448 / text) the
          decision wall (tower included, from the file) median and min-max, load seconds cold / warm, tower ms

Run from the worktree root (shared venv, offline):
    HF_HOME=~/code/coreai/_decider2bv/hf HF_HUB_OFFLINE=1 ../coreai-models/.venv/bin/python \\
        conversion/decider_vision/gate_swift.py score --aot ~/code/coreai/_decider2bv/swift/aot/fixture_aot.json \\
        [--jit .../fixture_jit.json] --transcript models/decider-2b-vision/gate-decider-2b-vision-swift.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
from readout_gate_vision import BAR, softmax64, tree_digest  # noqa: E402

LANE = work_path("_decider2bv")
ORACLE = LANE / "oracle"
V = host.VOCAB
LETTER0 = host.LETTER_IDS[0]
ARMS = ("g256", "g448", "text")
E2E = REPO / "models/decider-2b-vision/gate-decider-2b-vision-e2e.json"
PKG = REPO / "apps/DeciderVision"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def load_transcript(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_transcript(path: Path, rec: dict) -> None:
    rec["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rec["result_line"] = result_line(rec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1) + "\n")


def result_line(rec: dict) -> str:
    parts = []
    for k in ("s1", "s2", "s3", "s4", "c", "d", "s3_alt", "d_cpu_shared"):
        if k in rec:
            parts.append(f"{k.upper()} {rec[k].get('result', '?')}")
    return " | ".join(parts)


def oracle_runs() -> dict:
    fx = json.loads((ORACLE / "fixture_oracle.json").read_text())
    return {(r["id"], r["arm"]): r for r in fx["rows"] if r["arm"] in ARMS}


def package_record() -> dict:
    files = sorted(p for p in PKG.rglob("*.swift") if ".build" not in p.parts) + [PKG / "Package.swift"]
    rec = {"path": str(PKG), "sources_sha256": {str(p.relative_to(PKG)): sha256_file(p) for p in files if p.exists()}}
    resolved = PKG / "Package.resolved"
    if resolved.exists():
        pins = json.loads(resolved.read_text()).get("pins", [])
        rec["resolved"] = {p["identity"]: p["state"].get("version") or p["state"].get("revision") for p in pins}
    return rec


# --------------------------------------------------------------------------- #
# S1 / S2
# --------------------------------------------------------------------------- #
def score_s1(sw: dict, orc: dict) -> dict:
    runs = {(r["id"], r["arm"]): r for r in sw["runs"]}
    bad, ids_ok, slots_ok, amount_ok, n_image = [], 0, 0, 0, 0
    for key, o in sorted(orc.items()):
        r = runs.get(key)
        if r is None:
            bad.append(f"{key[0]}/{key[1]}: missing")
            continue
        proc = [host.IMAGE_PAD if t >= V else t for t in r["ids"]]
        if proc == o["ids"]:
            ids_ok += 1
        else:
            i = next((k for k, (a, b) in enumerate(zip(proc, o["ids"])) if a != b), min(len(proc), len(o["ids"])))
            bad.append(f"{key[0]}/{key[1]}: ids differ at {i} (len {len(proc)} vs {len(o['ids'])}): "
                       f"swift {proc[max(0, i - 3):i + 3]} oracle {o['ids'][max(0, i - 3):i + 3]}")
        if r["slots"] == o["slot_idx"]:
            slots_ok += 1
        else:
            bad.append(f"{key[0]}/{key[1]}: slots {r['slots']} vs {o['slot_idx']}")
        if key[1] != "text":
            n_image += 1
            g = r["grid"][0]
            ok = (r["rope_shift_amount"] == o["rope_shift"] and r["rope_shift_start"] == 1 + g * g
                  and r["ids"][1:1 + g * g] == [V + k for k in range(g * g)])
            amount_ok += ok
            if not ok:
                bad.append(f"{key[0]}/{key[1]}: rope start/amount {r['rope_shift_start']}/{r['rope_shift_amount']} "
                           f"vs {1 + g * g}/{o['rope_shift']}")
        elif r["rope_shift_start"] != host.NO_SHIFT or r["rope_shift_amount"] != 0:
            bad.append(f"{key[0]}/{key[1]}: text row rope {r['rope_shift_start']}/{r['rope_shift_amount']}")
    extra = sorted(f"{a}/{b}" for a, b in runs if (a, b) not in orc)
    ok = ids_ok == slots_ok == len(orc) and amount_ok == n_image and not bad and not extra
    return {"runs_expected": len(orc), "ids_equal_oracle": ids_ok, "slots_equal_oracle": slots_ok,
            "image_runs_rope_and_block_equal": amount_ok, "image_runs": n_image, "extra_runs": extra,
            "failures": bad, "result": "PASS" if ok else "FAIL",
            "definition": "Swift ids with V+k mapped to <|image_pad|> = fixture_oracle.json ids token for token; slots = "
                          "slot_idx; image runs: rope_shift_start = 1 + G^2, rope_shift_amount = the oracle's rope_shift, "
                          "ids[1 : 1 + G^2] = V + 0 .. V + G^2 - 1; text runs: start 1 << 30, amount 0"}


def score_s2(sw: dict) -> dict:
    from PIL import Image

    dump = Path(sw["dump_dir"])
    meta = {m["name"]: m for m in json.loads((LANE / "fixtures" / "meta.json").read_text())["images"]}
    tiles, seen = [], set()
    for r in sw["runs"]:
        if r.get("image") is None or "resized_file" not in r:
            continue
        key = (r["image"], r["arm"])
        if key in seen:
            continue
        seen.add(key)
        side = host.tile_side(r["grid"][0])
        im = Image.open(LANE / "fixtures" / meta[r["image"]]["path"])
        u8 = np.asarray(im.convert("RGB"))
        ref = host.resize_bicubic(u8, side, side)
        got = np.fromfile(dump / r["resized_file"], np.uint8).reshape(side, side, 3)
        d = np.abs(got.astype(np.int16) - ref.astype(np.int16))
        pil = np.asarray(im.convert("RGB").resize((side, side), Image.Resampling.BICUBIC))
        dp = np.abs(got.astype(np.int16) - pil.astype(np.int16))
        x = (ref.astype(np.float64) / 255.0 - host.IMAGE_MEAN) / host.IMAGE_STD
        patches_sha = hashlib.sha256(host.patchify(x).astype(np.float32).tobytes()).hexdigest()
        where = [int(v) for v in np.unravel_index(int(d.argmax()), d.shape)] if d.max() else None
        tiles.append({"image": r["image"], "grid": r["arm"], "size_in": list(u8.shape[1::-1]),
                      "decode_path": r.get("decode_path"),
                      "decoded_rgb_sha256_equals_meta": r["decoded_rgb_sha256"] == meta[r["image"]]["rgb_sha256"],
                      "max_abs_level_vs_resize_bicubic": int(d.max()), "where_yxc": where,
                      "n_channel_values_differing": int((d > 0).sum()),
                      "patches_sha256_equals_host": r["patches_sha256"] == patches_sha,
                      "max_abs_level_vs_pillow": int(dp.max()), "n_differing_vs_pillow": int((dp > 0).sum())})
    worst = max(tiles, key=lambda t: t["max_abs_level_vs_resize_bicubic"]) if tiles else None
    worst_pil = max(tiles, key=lambda t: t["max_abs_level_vs_pillow"]) if tiles else None
    ok = (len(tiles) == 58 and worst["max_abs_level_vs_resize_bicubic"] <= 1
          and all(t["decoded_rgb_sha256_equals_meta"] for t in tiles))
    return {"tiles": len(tiles), "tiles_expected": 58,
            "max_abs_level": worst["max_abs_level_vs_resize_bicubic"] if worst else None,
            "worst": {k: worst[k] for k in ("image", "grid", "max_abs_level_vs_resize_bicubic", "where_yxc")} if worst else None,
            "tiles_bit_equal": sum(t["max_abs_level_vs_resize_bicubic"] == 0 for t in tiles),
            "decoded_rgb_equal_meta": sum(t["decoded_rgb_sha256_equals_meta"] for t in tiles),
            "patches_sha256_equal_host": sum(t["patches_sha256_equals_host"] for t in tiles),
            "vs_pillow_values_only": {"max_abs_level": worst_pil["max_abs_level_vs_pillow"] if worst_pil else None,
                                      "worst": {k: worst_pil[k] for k in ("image", "grid", "max_abs_level_vs_pillow")}
                                      if worst_pil else None,
                                      "tiles_bit_equal": sum(t["max_abs_level_vs_pillow"] == 0 for t in tiles)},
            "bar": "max |level| <= 1 vs host.resize_bicubic (expected 0); decoded RGB = meta rgb_sha256",
            "result": "PASS" if ok else "FAIL", "tiles_detail": tiles}


# --------------------------------------------------------------------------- #
# S3 / S4
# --------------------------------------------------------------------------- #
def e2e_arm_label(bundle_name: str) -> str:
    return f"{bundle_name}+tower_fp16w32+chunk16"


def python_refs(bundle_name: str) -> tuple[dict, dict, dict]:
    """(e2e runs by (id, arm) with their npz arrays, b1 runs by (id, arm) with arrays, bundle records) of the round-6
    Python read-out of `bundle_name` (decider_2b_vision_decode_<scheme>)."""
    e2e = json.loads(E2E.read_text())
    arm = e2e["arms"][e2e_arm_label(bundle_name)]
    short = bundle_name.replace("decider_2b_vision_decode_", "")
    b1 = json.loads((REPO / f"models/decider-2b-vision/gate-decider-2b-vision-readout-{short}.json").read_text())

    def with_arrays(entry: dict) -> dict:
        out = {}
        npz = {p["shard"]: np.load(p["npz"]) for p in entry["processes"] if "npz" in p}
        for r in entry["runs"]:
            z = npz[r["shard"]]
            k = r["npz_key"]
            out[(r["id"], r["arm"])] = {"run": r, "slot_logits": z[f"{k}__slot_logits"],
                                        "tower_embeds": z[f"{k}__tower_embeds"] if f"{k}__tower_embeds" in z else None}
        return out

    return with_arrays(arm), with_arrays(b1), {"e2e": arm["bundle"], "b1": b1["bundle"], "e2e_towers": e2e["towers"]}


def score_readout(sw: dict, orc: dict, kind: str) -> dict:
    """The round-4 bar on one Swift pass, plus the Python read-out of the same bundle."""
    dump = Path(sw["dump_dir"])
    runs = {(r["id"], r["arm"]): r for r in sw["runs"]}
    bundle_name = Path(sw["assets"]["bundle"]).name
    e2e, b1, bundles = python_refs(bundle_name)
    slots, per_run_mean, run_rows = [], [], []
    vs_py = []
    for key, o in sorted(orc.items()):
        r = runs.get(key)
        if r is None:
            continue
        lg_all = np.fromfile(dump / r["logits_file"], np.float16).reshape(len(r["answers"]), V)
        deltas = []
        ref = e2e.get(key) if key[1] != "text" else b1.get(key)
        ref_name = "e2e" if key[1] != "text" else "b1"
        emb_equal = emb_max = None
        if ref is not None and ref["tower_embeds"] is not None and r.get("embeds_file"):
            got = np.fromfile(dump / r["embeds_file"], np.float32)
            want = ref["tower_embeds"].astype(np.float32).ravel()
            emb_equal = bool(got.shape == want.shape and np.array_equal(got, want))
            emb_max = float(np.abs(got.astype(np.float64) - want).max()) if got.shape == want.shape else None
        for s, a in enumerate(r["answers"]):
            n = o["nopts"][s]
            lg = lg_all[s]
            letters = lg[LETTER0:LETTER0 + n].astype(np.float64)
            p = softmax64(letters)
            po = np.asarray(o["probs"][s][:n], np.float64)
            d = np.abs(p - po)
            deltas.append(d)
            full = int(lg.argmax())
            rec = {"id": key[0], "arm": key[1], "t": a["t"], "nopts": n, "probs": p.tolist(), "probs_oracle": po.tolist(),
                   "probs_swift_json_equal": bool(np.array_equal(np.asarray(a["probs"]), p)),
                   "argmax": int(p.argmax()), "argmax_oracle": o["argmax"][s], "argmax_equal": int(p.argmax()) == o["argmax"][s],
                   "full_vocab_top1_id": full, "full_vocab_top1_is_oracle_letter": full == LETTER0 + o["argmax"][s],
                   "full_vocab_top1_equals_swift_json": full == a["full_vocab_top1_id"],
                   "max_abs_dp": float(d.max()), "mean_abs_dp": float(d.mean()),
                   "finite": bool(np.isfinite(lg.astype(np.float32)).all()), "read_from": a["read_from"],
                   "oracle_top2_margin": float(np.sort(po)[-1] - np.sort(po)[-2]) if n > 1 else None}
            if ref is not None and s < len(ref["slot_logits"]):
                pl = ref["slot_logits"][s]
                pp = softmax64(pl[LETTER0:LETTER0 + n].astype(np.float64))
                rec["python"] = {"ref": ref_name, "max_abs_dp": float(np.abs(p - pp).max()),
                                 "letter_logits_bit_equal": bool(np.array_equal(lg[LETTER0:LETTER0 + n].view(np.uint16),
                                                                                pl[LETTER0:LETTER0 + n].view(np.uint16))),
                                 "full_logits_bit_equal": bool(np.array_equal(lg.view(np.uint16), pl.view(np.uint16))),
                                 "full_logits_max_abs_diff": float(np.abs(lg.astype(np.float32) - pl.astype(np.float32)).max()),
                                 "argmax_equal": int(pp.argmax()) == int(p.argmax())}
                vs_py.append(rec["python"] | {"id": key[0], "arm": key[1], "t": a["t"],
                                              "tower_embeds_bit_equal": emb_equal, "tower_embeds_max_abs_diff": emb_max})
            slots.append(rec)
        flat = np.concatenate(deltas)
        per_run_mean.append(float(flat.mean()))
        run_rows.append({"id": key[0], "arm": key[1], "tokens": r["tokens"], "max_abs_dp": float(flat.max()),
                         "mean_abs_dp": float(flat.mean()), "calls": r["calls"],
                         "tower_embeds_bit_equal_python": emb_equal, "tower_embeds_max_abs_diff_python": emb_max})
    n_expected = sum(len(o["slot_idx"]) for o in orc.values())
    worst = max(slots, key=lambda x: x["max_abs_dp"]) if slots else None
    summary = {"runs": len(run_rows), "runs_expected": len(orc), "slots_read": len(slots), "slots_expected": n_expected,
               "argmax_equal": sum(x["argmax_equal"] for x in slots),
               "full_vocab_top1_is_oracle_letter": sum(x["full_vocab_top1_is_oracle_letter"] for x in slots),
               "max_abs_dp": worst["max_abs_dp"] if worst else None,
               "worst": {k: worst[k] for k in ("id", "arm", "t", "max_abs_dp", "probs", "probs_oracle")} if worst else None,
               "mean_of_run_mean_abs_dp": float(np.mean(per_run_mean)) if per_run_mean else None,
               "finite_all": all(x["finite"] for x in slots),
               "reset_bit_equal": bool(sw.get("reset_check", {}).get("bit_equal")),
               "swift_json_probs_equal_recomputed": sum(x["probs_swift_json_equal"] for x in slots),
               "slots_read_from_prefill": sum(x["read_from"] == "prefill" for x in slots)}
    checks = {"all_runs": summary["runs"] == summary["runs_expected"],
              "slots": summary["slots_read"] == summary["slots_expected"],
              "argmax": summary["argmax_equal"] == n_expected,
              "full_vocab_top1": summary["full_vocab_top1_is_oracle_letter"] == n_expected,
              "max_abs_dp": summary["max_abs_dp"] is not None and summary["max_abs_dp"] <= BAR["max_abs_dp"],
              "mean_of_run_mean_abs_dp": (summary["mean_of_run_mean_abs_dp"] is not None
                                          and summary["mean_of_run_mean_abs_dp"] <= BAR["mean_of_run_mean_abs_dp"]),
              "finite": summary["finite_all"], "reset_bit_equal": summary["reset_bit_equal"]}
    per_arm = {}
    for arm in ARMS:
        sel = [x for x in slots if x["arm"] == arm]
        rr = [x for x in run_rows if x["arm"] == arm]
        per_arm[arm] = {"runs": len(rr), "slots": len(sel), "argmax_equal": sum(x["argmax_equal"] for x in sel),
                        "full_vocab_top1_is_oracle_letter": sum(x["full_vocab_top1_is_oracle_letter"] for x in sel),
                        "max_abs_dp": max((x["max_abs_dp"] for x in sel), default=None),
                        "mean_of_run_mean_abs_dp": float(np.mean([x["mean_abs_dp"] for x in rr])) if rr else None}

    def agg(sel: list[dict]) -> dict:
        if not sel:
            return {"slots": 0}
        return {"slots": len(sel), "max_abs_dp": max(x["max_abs_dp"] for x in sel),
                "mean_abs_dp_of_slot_max": float(np.mean([x["max_abs_dp"] for x in sel])),
                "letter_logits_bit_equal": sum(x["letter_logits_bit_equal"] for x in sel),
                "full_logits_bit_equal": sum(x["full_logits_bit_equal"] for x in sel),
                "full_logits_max_abs_diff": max(x["full_logits_max_abs_diff"] for x in sel),
                "argmax_equal": sum(x["argmax_equal"] for x in sel),
                "tower_embeds_bit_equal_runs": len({(x["id"], x["arm"]) for x in sel if x["tower_embeds_bit_equal"]}),
                "tower_embeds_max_abs_diff": max((x["tower_embeds_max_abs_diff"] or 0.0 for x in sel), default=None)}

    vs = {"definition": f"per slot, the same bundle read by Python: image runs vs the round-6 e2e arm "
                        f"({e2e_arm_label(bundle_name)}: Pillow resize -> same tower "
                        ".aimodelc -> same decoder .aimodelc, chunk order), text runs vs the round-6 b1 transcript "
                        "(identical inputs). |dp| over the question's letters; bit-equality of the fp16 logits and of "
                        "the tower's float32 output",
          "all": agg(vs_py), "by_arm": {a: agg([x for x in vs_py if x["arm"] == a]) for a in ARMS}}
    if kind == "jit":
        vs["note"] = ("the Python references are the AOT .aimodelc runs (the Python runtime's JIT mis-executes hybrids "
                      "above 0.8B), so this compares Swift's JIT specialization with the AOT compile of the same "
                      ".aimodel: not expected bit-equal")
    ok = all(checks.values())
    return {"asset": kind, "bundle": bundle_name, "fixture_json": sw.get("_path"), "label": sw.get("label"),
            "assets": sw.get("assets"),
            "environment": sw.get("environment"), "load_first": sw.get("load_first"),
            "function_names": sw.get("function_names"), "reset_check": sw.get("reset_check"),
            "bar": {**BAR, "source": "readout_gate_vision.BAR (round 4, unchanged)", "slots": "all", "argmax": "all",
                    "full_vocab_top1": "the oracle's argmax letter on every slot", "reset": "bit-equal re-run"},
            "summary": summary, "checks": checks, "result": "PASS" if ok else "FAIL", "per_arm": per_arm,
            "vs_python_same_bundle": vs, "python_bundle_records": {k: v.get("aimodelc", {}).get("tree_sha256") if isinstance(v, dict) else None
                                                                   for k, v in bundles.items() if k != "e2e_towers"},
            "runs": run_rows, "slots": slots}


def asset_identity(sw: dict, bundles_checked: dict) -> dict:
    """The Swift pass's decoder asset and towers = the ones the round-6 Python transcripts ran (tree sha256)."""
    out = {}
    dec = Path(sw["assets"]["decoder"])
    if dec.suffix == ".aimodelc":
        e2e = json.loads(E2E.read_text())
        arm = e2e["arms"][e2e_arm_label(Path(sw["assets"]["bundle"]).name)]
        want = arm["bundle"]["aimodelc"]["tree_sha256"]
        key = str(dec)
        if key not in bundles_checked:
            bundles_checked[key] = tree_digest(dec)["tree_sha256"]
        out["decoder_aimodelc_tree_sha256"] = bundles_checked[key]
        out["decoder_equals_round6"] = bundles_checked[key] == want
        for g in ("g256", "g448"):
            t = Path(sw["towers"][g])
            if t.suffix == ".aimodelc":
                if str(t) not in bundles_checked:
                    bundles_checked[str(t)] = tree_digest(t)["tree_sha256"]
                out[f"tower_{g}_equals_round6"] = bundles_checked[str(t)] == e2e["towers"][f"fp16w32/{g}"]["tree_sha256"]
    return out


def attribution(readout: dict, s2: dict, sw: dict) -> dict:
    """Which image runs' tower output differs from the Python e2e arm's, against which tiles differ between Pillow's
    resize (what the e2e arm fed) and resize_bicubic (what Swift feeds): the two sets must be the same."""
    tiles = {(d["image"], d["grid"]): d["max_abs_level_vs_pillow"] for d in s2["tiles_detail"]}
    image_of = {(r["id"], r["arm"]): r.get("image") for r in sw["runs"]}
    tower_diff = sorted(f"{r['id']}/{r['arm']}" for r in readout["runs"]
                        if r["arm"] != "text" and r["tower_embeds_bit_equal_python"] is False)
    tile_diff = sorted(f"{r['id']}/{r['arm']}" for r in readout["runs"]
                       if r["arm"] != "text" and tiles.get((image_of[(r["id"], r["arm"])], r["arm"]), 0) > 0)
    slot_diff = sorted({f"{x['id']}/{x['arm']}" for x in readout["slots"]
                        if "python" in x and not x["python"]["full_logits_bit_equal"]})
    return {"runs_tower_output_differs_from_python": tower_diff,
            "runs_whose_tile_differs_pillow_vs_resize_bicubic": tile_diff,
            "runs_with_slot_logits_differing_from_python": slot_diff,
            "same_set": tower_diff == tile_diff and set(slot_diff) <= set(tile_diff),
            "reading": "every Swift/Python difference sits on a run whose tile Pillow and resize_bicubic resize "
                       "differently; the rest is bit-equal (tower output and fp16 logits)"}


def cmd_score(args) -> int:
    orc = oracle_runs()
    tpath = Path(args.transcript)
    rec = load_transcript(tpath)
    rec.update({"schema": "coreai-decider-vision-swift-gate/1",
                "gate": "Swift read-out (apps/DeciderVision, decider-vision CLI) vs the author's fp32 oracle, the Python "
                        "host reference and the Python read-out of the same bundle",
                "oracle": {"path": str(ORACLE / "fixture_oracle.json"), "sha256": sha256_file(ORACLE / "fixture_oracle.json"),
                           "rows_json_sha256": sha256_file(LANE / "fixtures" / "rows.json"),
                           "runs": len(orc), "slots": sum(len(o["slot_idx"]) for o in orc.values())},
                "package": package_record()})
    checked: dict = {}
    if args.aot:
        sw = json.loads(Path(args.aot).read_text())
        sw["_path"] = str(Path(args.aot).resolve())
        rec["s1"] = score_s1(sw, orc)
        rec["s2"] = score_s2(sw)
        rec["s3"] = score_readout(sw, orc, "aot")
        rec["s3"]["asset_identity"] = asset_identity(sw, checked)
        rec["s3"]["vs_python_attribution"] = attribution(rec["s3"], rec["s2"], sw)
        print(f"S1 {rec['s1']['result']}: ids {rec['s1']['ids_equal_oracle']}/{rec['s1']['runs_expected']} slots "
              f"{rec['s1']['slots_equal_oracle']}/{rec['s1']['runs_expected']} rope {rec['s1']['image_runs_rope_and_block_equal']}"
              f"/{rec['s1']['image_runs']}")
        for f in rec["s1"]["failures"][:10]:
            print("   ", f)
        s2 = rec["s2"]
        print(f"S2 {s2['result']}: {s2['tiles']} tiles, max {s2['max_abs_level']} level(s) vs resize_bicubic "
              f"({s2['tiles_bit_equal']} bit-equal), decoded = meta {s2['decoded_rgb_equal_meta']}, patches = host "
              f"{s2['patches_sha256_equal_host']}; vs Pillow max {s2['vs_pillow_values_only']['max_abs_level']} (values only)")
        report_readout("S3", rec["s3"])
    if args.alt:
        sw = json.loads(Path(args.alt).read_text())
        sw["_path"] = str(Path(args.alt).resolve())
        rec["s3_alt"] = score_readout(sw, orc, "aot")
        rec["s3_alt"]["asset_identity"] = asset_identity(sw, checked)
        s1 = score_s1(sw, orc)
        rec["s3_alt"]["s1_same_pass"] = {k: s1[k] for k in ("ids_equal_oracle", "slots_equal_oracle", "result")}
        rec["s3_alt"]["note"] = ("extra arm, not in the acceptance table: the alternative ship candidate (round 7 "
                                 "decides between the two bundles), same Swift code, same bar")
        report_readout("S3-alt", rec["s3_alt"])
    if args.jit:
        sw = json.loads(Path(args.jit).read_text())
        sw["_path"] = str(Path(args.jit).resolve())
        rec["s4"] = score_readout(sw, orc, "jit")
        s1 = score_s1(sw, orc)
        rec["s4"]["s1_same_pass"] = {k: s1[k] for k in ("ids_equal_oracle", "slots_equal_oracle", "result")}
        if args.jit_note:
            rec["s4"]["note"] = args.jit_note
        report_readout("S4", rec["s4"])
    save_transcript(tpath, rec)
    print(f"transcript: {tpath}")
    return 0


def report_readout(tag: str, r: dict) -> None:
    s, v = r["summary"], r["vs_python_same_bundle"]["all"]
    print(f"{tag} {r['result']} ({r['asset']}): runs {s['runs']}/{s['runs_expected']} slots {s['slots_read']}/"
          f"{s['slots_expected']} argmax {s['argmax_equal']} full-vocab {s['full_vocab_top1_is_oracle_letter']} "
          f"max|dp| {s['max_abs_dp']:.6f} mean {s['mean_of_run_mean_abs_dp']:.6f} reset {s['reset_bit_equal']} "
          f"worst {s['worst']['id']}/{s['worst']['arm']}")
    for k, ok in r["checks"].items():
        if not ok:
            print(f"   check {k}: FAIL")
    if v.get("slots"):
        print(f"   vs Python same bundle: slots {v['slots']} max|dp| {v['max_abs_dp']:.6f} letter logits bit-equal "
              f"{v['letter_logits_bit_equal']} full logits bit-equal {v['full_logits_bit_equal']} (max diff "
              f"{v['full_logits_max_abs_diff']:.4g}) tower bit-equal runs {v['tower_embeds_bit_equal_runs']}")
        for a, x in r["vs_python_same_bundle"]["by_arm"].items():
            if x.get("slots"):
                print(f"     {a}: slots {x['slots']} max|dp| {x['max_abs_dp']:.6f} letter bit-equal "
                      f"{x['letter_logits_bit_equal']} full bit-equal {x['full_logits_bit_equal']} tower bit-equal runs "
                      f"{x['tower_embeds_bit_equal_runs']} tower max|d| {x['tower_embeds_max_abs_diff']}")
    if "asset_identity" in r:
        print(f"   assets = round 6: {r['asset_identity']}")


# --------------------------------------------------------------------------- #
# C: one question outside the fixture, Swift beside Python
# --------------------------------------------------------------------------- #
def python_read(bundle: Path, decoder: Path, tower: Path, grid: int, image: Path | None, context: str,
                questions: list[dict], resize: str) -> dict:
    import asyncio
    import inspect

    import coreai.runtime as rt
    import tokenizers

    async def maybe(x):
        return await x if inspect.isawaitable(x) else x

    def nd(a):
        return rt.NDArray(np.ascontiguousarray(a))

    tok = tokenizers.Tokenizer.from_file(str(bundle / "tokenizer" / "tokenizer.json"))
    meta = json.loads((bundle / "metadata.json").read_text())
    C = int(meta["language"]["prefill_chunk"])
    out: dict = {"resize": resize}

    async def go():
        emb = np.zeros((256, 2048), np.float16)
        rc = np.zeros((256, 2), np.int32)
        hw = (grid, grid) if image is not None else None
        ids, slots, start, amount = host.build_ids(image is not None, context, questions, tok, hw)
        if image is not None:
            tm = await maybe(rt.AIModel.load(str(tower), rt.SpecializationOptions.default()))
            tf = await maybe(tm.load_function(tm.function_names[0]))
            patches = host.preprocess(image, grid, resize=resize)
            t = await maybe(tf(inputs={"patches": nd(patches)}))
            te = np.asarray(t["image_embeds"].numpy()).astype(np.float32)
            emb[:grid * grid] = te.astype(np.float16)
            k = np.arange(grid * grid)
            rc[:grid * grid, 0], rc[:grid * grid, 1] = k // grid, k % grid
            out["patches_sha256"] = hashlib.sha256(patches.tobytes()).hexdigest()
            out["tower_embeds_sha256"] = hashlib.sha256(te.tobytes()).hexdigest()
        m = await maybe(rt.AIModel.load(str(decoder), rt.SpecializationOptions.default()))
        fm = await maybe(m.load_function("main"))
        fp = await maybe(m.load_function("prefill"))
        state = {}
        for n, d in ((n, fm.desc.state_descriptor(n)) for n in fm.desc.state_names):
            state[n] = nd(np.zeros([int(meta["language"]["max_context_length"]) if s < 0 else int(s) for s in d.shape],
                                   np.float16))
        static = {"image_embeds": nd(emb), "image_rc": nd(rc), "rope_shift_start": nd(np.array([start], np.int32)),
                  "rope_shift_amount": nd(np.array([amount], np.int32))}
        got, cur = {}, 0
        for s in slots:
            while s - cur + 1 >= C:
                r = await maybe(fp(inputs={"input_ids": nd(np.array([ids[cur:cur + C]], np.int32)),
                                           "position_ids": nd(np.arange(cur + C, dtype=np.int32)[None]), **static},
                                   state=state))
                if cur + C - 1 == s:
                    got[s] = np.asarray(r["logits"].numpy())[0, -1].copy()
                cur += C
            while cur <= s:
                r = await maybe(fm(inputs={"input_ids": nd(np.array([[ids[cur]]], np.int32)),
                                           "position_ids": nd(np.arange(cur + 1, dtype=np.int32)[None]), **static},
                                   state=state))
                if cur == s:
                    got[s] = np.asarray(r["logits"].numpy())[0, -1].copy()
                cur += 1
        out["ids"] = [int(x) for x in ids]
        out["slots"] = slots
        out["answers"] = []
        for q, s in zip(questions, slots):
            n = len(q["options"])
            lg = got[s]
            p = softmax64(lg[LETTER0:LETTER0 + n].astype(np.float64))
            out["answers"].append({"t": s, "letter_logits": lg[LETTER0:LETTER0 + n].astype(np.float32).tolist(),
                                   "probs": p.tolist(), "argmax": int(p.argmax()), "full_vocab_top1_id": int(lg.argmax())})

    asyncio.run(go())
    return out


def cmd_ask(args) -> int:
    sw = json.loads(Path(args.swift_json).read_text())
    bundle = Path(sw["assets"]["bundle"])
    decoder = Path(sw["assets"]["decoder"])
    image = Path(sw["image"]) if sw.get("image") else None
    grid = sw["grid"][0] if sw.get("grid") else 8
    tower = Path(sw["tower"]) if sw.get("tower") else None
    qs = sw["questions"]
    py = {r: python_read(bundle, decoder, tower, grid, image, sw["context"], qs, r) for r in ("pil", "numpy")}
    rows = []
    for k, q in enumerate(qs):
        a = sw["answers"][k]
        for j, o in enumerate(q["options"]):
            rows.append({"question": q["text"], "option": o, "letter": host.LETTERS[j],
                         "swift": a["probs"][j], "python_resize_bicubic": py["numpy"]["answers"][k]["probs"][j],
                         "python_pillow": py["pil"]["answers"][k]["probs"][j]})
    d_same = max(abs(r["swift"] - r["python_resize_bicubic"]) for r in rows)
    d_pil = max(abs(r["swift"] - r["python_pillow"]) for r in rows)
    ids_equal = [host.IMAGE_PAD if t >= V else t for t in sw["ids"]] == [host.IMAGE_PAD if t >= V else t
                                                                         for t in py["numpy"]["ids"]]
    rec = {"image": str(image) if image else None, "image_sha256": sha256_file(image) if image else None,
           "grid": grid, "context": sw["context"], "questions": qs, "swift_json": str(Path(args.swift_json).resolve()),
           "rows": rows, "ids_equal": ids_equal, "slots_swift": sw["slots"], "slots_python": py["numpy"]["slots"],
           "patches_sha256_equal_python_resize_bicubic": sw.get("patches_sha256") == py["numpy"].get("patches_sha256"),
           "tower_embeds_sha256_equal_python_resize_bicubic":
               sw.get("tower_embeds_sha256") == py["numpy"].get("tower_embeds_sha256"),
           "letter_logits": {"swift": [a["letter_logits"] for a in sw["answers"]],
                             "python_resize_bicubic": [a["letter_logits"] for a in py["numpy"]["answers"]],
                             "python_pillow": [a["letter_logits"] for a in py["pil"]["answers"]]},
           "max_abs_dp_swift_vs_python_resize_bicubic": d_same, "max_abs_dp_swift_vs_python_pillow": d_pil,
           "swift_wall_s": sw["seconds"].get("wall"), "decoder": str(decoder), "tower": str(tower) if tower else None,
           "result": "done (values side by side)"}
    tpath = Path(args.transcript)
    t = load_transcript(tpath)
    t["c"] = rec
    save_transcript(tpath, t)
    print(f"C: {rec['image']} grid {grid}: ids equal {ids_equal}, patches equal {rec['patches_sha256_equal_python_resize_bicubic']}, "
          f"tower equal {rec['tower_embeds_sha256_equal_python_resize_bicubic']}")
    print(f"{'option':>12} {'Swift':>10} {'Py (same resize)':>17} {'Py (Pillow)':>12}")
    for r in rows:
        print(f"{r['letter'] + ' ' + r['option']:>12} {r['swift']:10.6f} {r['python_resize_bicubic']:17.6f} {r['python_pillow']:12.6f}")
    print(f"max |dp| Swift vs Python same resize {d_same:.2e}, vs Pillow {d_pil:.2e}")
    print(f"transcript: {tpath}")
    return 0


# --------------------------------------------------------------------------- #
# D: time per decision under the GPU lock
# --------------------------------------------------------------------------- #
def cmd_timing(args) -> int:
    passes = []
    for p in args.passes:
        sw = json.loads(Path(p).read_text())
        sw["_path"] = str(Path(p).resolve())
        passes.append(sw)
    lock = json.loads(Path(args.gpu_lock).read_text()) if args.gpu_lock else None

    def q(xs: list[float]) -> dict:
        return {"n": len(xs), "median": float(np.median(xs)) if xs else None, "min": min(xs, default=None),
                "max": max(xs, default=None)}

    table, loads = {}, []
    for i, sw in enumerate(passes):
        runs = sw["runs"]
        first = runs[0] if runs else None
        ent = {"pass": i + 1, "json": sw["_path"], "label": sw.get("label"),
               "bundle": Path(sw["assets"]["bundle"]).name, "asset": sw["assets"]["asset"],
               "build": sw["environment"].get("build_configuration"),
               "first_decision_of_process": {"id": first["id"], "arm": first["arm"],
                                             "wall_from_file_s": first["wall_from_file_s"]} if first else None}
        for arm in ARMS:
            rs = [r for r in runs[1:] if r["arm"] == arm]
            ent[arm] = {"wall_from_file_s": q([r["wall_from_file_s"] for r in rs]),
                        "decision_wall_s": q([r["seconds"]["wall"] for r in rs]),
                        "decoder_s": q([r["seconds"]["decoder"] for r in rs]),
                        "tokens_median": float(np.median([r["tokens"] for r in rs])) if rs else None,
                        "slots": sum(len(r["slots"]) for r in rs)}
            if arm != "text":
                ent[arm]["tower_ms"] = q([r["seconds"]["tower"] * 1e3 for r in rs])
                ent[arm]["host_preprocess_ms"] = q([(r["seconds"]["decode_rgb"] + r["seconds"]["resize"]
                                                     + r["seconds"]["patches"]) * 1e3 + r["image_file_decode_s"] * 1e3
                                                    for r in rs])
        calls_p = [ms for r in runs[1:] for ms, k in zip(r["call_ms"], r["call_is_prefill"]) if k]
        calls_m = [ms for r in runs[1:] for ms, k in zip(r["call_ms"], r["call_is_prefill"]) if not k]
        ent["prefill_call_ms"] = q(calls_p)
        ent["main_call_ms"] = q(calls_m)
        ent["state_reset_ms"] = q([r["state_reset_ms"] for r in runs[1:]])
        ent["footprint_mb"] = {"after_load": sw.get("footprint_after_load_bytes", 0) / 1e6,
                               "after_runs": sw.get("footprint_after_runs_bytes", 0) / 1e6,
                               "max_during_runs": max((r.get("footprint_bytes", 0) for r in runs), default=0) / 1e6}
        key = f"{ent['bundle'].replace('decider_2b_vision_decode_', '')}_pass{sum(1 for e in table.values() if e['bundle'] == ent['bundle']) + 1}"
        table[key] = ent
        loads.append({"pass": key, "first_load_in_process": sw.get("load_first"), "reload_in_process": sw.get("load_reload")})
    rec = {"passes": table, "loads": loads, "gpu_lock": lock,
           "contended": (lock is None) or (not str(lock.get("state", "")).startswith("taken")),
           "definition": "wall_from_file = ImageIO decode of the PNG + RGB8 + resize + patches + tower + prompt + "
                         "static inputs + decoder calls (chunk order) + read-out, per run (one row = one decision with "
                         "1..3 questions); the first run of each process is reported apart (first-call costs); "
                         "load = AIModel(contentsOf:) + loadFunction, tokenizer apart",
           "result": "done" if lock else "done (no lock record)"}
    tpath = Path(args.transcript)
    if args.note:
        rec["note"] = args.note
    t = load_transcript(tpath)
    t[args.section] = rec
    save_transcript(tpath, t)
    print(f"GPU lock: {lock.get('state') if lock else 'no record'}")
    print(f"{'pass':>20} {'arm':>5} {'n':>3} {'wall median':>12} {'min-max':>16} {'tower ms':>9} {'tokens':>7}")
    for k, ent in table.items():
        for arm in ARMS:
            e = ent[arm]
            w = e["wall_from_file_s"]
            tw = e.get("tower_ms", {}).get("median")
            print(f"{k:>20} {arm:>5} {w['n']:>3} {w['median'] * 1e3:10.1f}ms {w['min'] * 1e3:7.1f}-{w['max'] * 1e3:.1f}ms "
                  f"{(f'{tw:.1f}' if tw else '-'):>9} {e['tokens_median']:>7}")
        print(f"{k:>20} first decision of the process {ent['first_decision_of_process']['wall_from_file_s'] * 1e3:.1f} ms; "
              f"prefill call median {ent['prefill_call_ms']['median']:.1f} ms, main call {ent['main_call_ms']['median']:.1f} ms, "
              f"state reset {ent['state_reset_ms']['median']:.1f} ms")
    for ld in loads:
        f, r = ld["first_load_in_process"], ld["reload_in_process"]

        def one(x: dict) -> str:
            return (f"{x['wall_s']:.2f}s (tokenizer {x['tokenizer_s']:.2f}, decoder main {x['decoder_s']['main']:.2f} "
                    f"prefill {x['decoder_s']['prefill']:.2f}, towers g256 {x['tower_s']['g256']:.2f} g448 "
                    f"{x['tower_s']['g448']:.2f})")
        print(f"{ld['pass']:>20} load first {one(f)}" + (f"; reload {one(r)}" if r else ""))
    print(f"transcript: {tpath}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("score")
    a.add_argument("--aot")
    a.add_argument("--jit")
    a.add_argument("--alt", help="an extra AOT pass on another bundle (values + the same bar, section s3_alt)")
    a.add_argument("--jit-note")
    a.add_argument("--transcript", required=True)
    b = sub.add_parser("ask")
    b.add_argument("--swift-json", required=True)
    b.add_argument("--transcript", required=True)
    c = sub.add_parser("timing")
    c.add_argument("--passes", nargs="+", required=True)
    c.add_argument("--gpu-lock")
    c.add_argument("--section", default="d", help="transcript key (d = the headline timing)")
    c.add_argument("--note")
    c.add_argument("--transcript", required=True)
    args = ap.parse_args()
    return {"score": cmd_score, "ask": cmd_ask, "timing": cmd_timing}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
