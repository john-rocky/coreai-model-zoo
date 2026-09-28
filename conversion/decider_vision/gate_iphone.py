#!/usr/bin/env python3
"""iPhone gate of decider-2b-vision: the device's DeciderVisionGate runs (apps/DeciderVisionGate) scored on the Mac
against the author's fp32 oracle and the Mac's Swift JIT read-out of the same assets (round 8).

The app runs the shipped set as shipped — the int8mix_pf16 decoder `.aimodel` and the fp16w32 g256 / g448 tower
`.aimodel`s, specialized by the phone's own JIT — through the DeciderVision library (VisionDecider.trace) over the
round-1 fixture (g256 / g448 / text: 76 runs, 108 slots), and writes per run the letter logits (fp16 bit patterns), the
probabilities, the full-vocabulary top-1, the ids and slots, the sha256 of the pixels / patches / tower output / slot
logits, and every step's time; its dump/ holds each run's full fp16 slot logits and tower output. This script only
reads those files:

  - the round-4 bar (readout_gate_vision.BAR, unchanged) on the 108 slots of the cold run: every run and slot present,
    ids and slots = the oracle's, letter argmax = the oracle's on every slot, full-vocabulary top-1 = the oracle's
    argmax letter, max |dp| <= 0.02, mean over runs of the run's mean |dp| over all its (slot, option) pairs <= 0.002
    (the round-4 definition; the other reading, a run's mean of its slot means, is shown beside it), finite, and the
    reset re-run of each phase bit-equal. The probabilities are recomputed here from the fp16 letter logits (softmax at
    T = 1 in float64) and must equal the app's.
  - the Mac Swift JIT of the same assets (swift/jit/fixture_jit.json + its dump): argmax agreement, |dp| max and median,
    letter logits / full slot logits / tower output bit-equality (and their max |diff|).
  - the warm run (the same container, cache present): the same bar, and bit-equality of every slot's logits with the
    cold run.
  - loads (cold / warm: tower, decoder, peak footprint, least os_proc_available_memory) and the time of one decision
    (bench rows: 1 warm-up + 5 back to back after a rest and the thermal state nominal), each with thermal state and
    battery (the phone is on USB power throughout).

Run from the worktree root (shared venv, offline):
    ../coreai-models/.venv/bin/python conversion/decider_vision/gate_iphone.py \\
        --cold apps/DeciderVisionGate/_work/device_runs/<cold run> --warm apps/DeciderVisionGate/_work/device_runs/<warm run> \\
        --transcript models/decider-2b-vision/gate-decider-2b-vision-iphone.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import work_path  # noqa: E402
from readout_gate_vision import BAR, softmax64  # noqa: E402

LANE = work_path("_decider2bv")
ORACLE = LANE / "oracle" / "fixture_oracle.json"
MAC_JIT = LANE / "swift" / "jit" / "fixture_jit.json"
V = 248320
IMAGE_PAD = 248056
LETTER0 = 32
ARMS = ("g256", "g448", "text")
E2E_STAGES = {"g256": "e2e_g256", "g448": "e2e_g448", "text": "e2e_text"}
APP = REPO / "apps" / "DeciderVisionGate"
STAGE_DIR = APP / "_work" / "device_stage" / "DeciderVisionAssets"
SOURCES = {
    "decoder/decider_2b_vision_decode_int8mix_pf16.aimodel/main.mlirb":
        LANE / "exports/bundles/decider_2b_vision_decode_int8mix_pf16/decider_2b_vision_decode_int8mix_pf16.aimodel/main.mlirb",
    "towers/decider_2b_vision_g256_vision_fp16w32.aimodel/main.mlirb":
        LANE / "exports/decider_2b_vision_g256_vision_fp16w32/decider_2b_vision_g256_vision_fp16w32.aimodel/main.mlirb",
    "towers/decider_2b_vision_g448_vision_fp16w32.aimodel/main.mlirb":
        LANE / "exports/decider_2b_vision_g448_vision_fp16w32/decider_2b_vision_g448_vision_fp16w32.aimodel/main.mlirb",
    "decoder/tokenizer/tokenizer.json": LANE / "exports/bundles/decider_2b_vision_decode_int8mix_pf16/tokenizer/tokenizer.json",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def f16_from_bits(bits: list[int]) -> np.ndarray:
    return np.asarray(bits, np.uint16).view(np.float16)


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def oracle_runs() -> dict:
    fx = json.loads(ORACLE.read_text())
    return {(r["id"], r["arm"]): r for r in fx["rows"] if r["arm"] in ARMS}


def mac_runs() -> tuple[dict, Path, dict]:
    m = json.loads(MAC_JIT.read_text())
    return {(r["id"], r["arm"]): r for r in m["runs"]}, Path(m["dump_dir"]), m


def device_runs(result: dict) -> dict:
    out = {}
    for arm in ARMS:
        st = result.get("stages", {}).get(E2E_STAGES[arm], {})
        for r in st.get("runs", []):
            out[(r["id"], r["arm"])] = r
    return out


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score_run_set(run_dir: Path, orc: dict, mac: dict, mac_dump: Path, cold_logits: dict | None = None) -> dict:
    """The bar and the Mac comparison over one device run's e2e stages."""
    result = json.loads((run_dir / "result.json").read_text())
    dev = device_runs(result)
    dump = run_dir / "dump"
    slots, run_rows = [], []
    logits_by_key = {}
    for key, o in sorted(orc.items()):
        r = dev.get(key)
        if r is None or "answers" not in r:
            continue
        m = mac.get(key)
        full = None
        lf = dump / "logits" / f"{key[0]}__{key[1]}.logits.f16"
        if lf.exists():
            full = np.fromfile(lf, np.float16).reshape(len(r["answers"]), V)
            logits_by_key[key] = full
        mfull = None
        if m is not None and m.get("logits_file"):
            mf = mac_dump / m["logits_file"]
            if mf.exists():
                mfull = np.fromfile(mf, np.float16).reshape(len(m["answers"]), V)
        emb_equal = emb_max = None
        ef = dump / "embeds" / f"{key[0]}__{key[1]}.embeds.f32"
        if ef.exists() and m is not None and m.get("embeds_file"):
            got = np.fromfile(ef, np.float32)
            want = np.fromfile(mac_dump / m["embeds_file"], np.float32)
            emb_equal = bool(got.shape == want.shape and np.array_equal(got.view(np.uint32), want.view(np.uint32)))
            emb_max = float(np.abs(got.astype(np.float64) - want).max()) if got.shape == want.shape else None
        proc = [IMAGE_PAD if t >= V else t for t in r["ids"]]
        deltas = []
        for s, a in enumerate(r["answers"]):
            n = o["nopts"][s]
            bits = a["letter_logits_bits"]
            lg = f16_from_bits(bits).astype(np.float64)
            p = softmax64(lg)
            po = np.asarray(o["probs"][s][:n], np.float64)
            d = np.abs(p - po)
            deltas.append(d)
            rec = {"id": key[0], "arm": key[1], "t": a["t"], "nopts": n, "probs": p.tolist(), "probs_oracle": po.tolist(),
                   "probs_app_equal_recomputed": bool(np.array_equal(np.asarray(a["probs"]), p)),
                   "argmax": int(p.argmax()), "argmax_oracle": o["argmax"][s],
                   "argmax_equal": int(p.argmax()) == o["argmax"][s],
                   "full_vocab_top1_id": a["full_vocab_top1_id"],
                   "full_vocab_top1_is_oracle_letter": a["full_vocab_top1_id"] == LETTER0 + o["argmax"][s],
                   "max_abs_dp": float(d.max()), "mean_abs_dp": float(d.mean()), "finite": bool(a["finite"]),
                   "read_from": a["read_from"], "oracle_top2_margin": float(np.sort(po)[-1] - np.sort(po)[-2]) if n > 1 else None}
            if full is not None:
                fl = full[s]
                rec["dump_letter_logits_equal_json"] = bool(np.array_equal(fl[LETTER0:LETTER0 + n].view(np.uint16),
                                                                           np.asarray(bits, np.uint16)))
                rec["dump_full_vocab_top1_id"] = int(fl.astype(np.float32).argmax())
                rec["dump_full_vocab_top1_equal_json"] = rec["dump_full_vocab_top1_id"] == a["full_vocab_top1_id"]
                rec["finite_full"] = bool(np.isfinite(fl.astype(np.float32)).all())
            if m is not None and s < len(m["answers"]):
                ma = m["answers"][s]
                ml = np.asarray(ma["letter_logits"], np.float64)
                pm = softmax64(ml)
                dm = np.abs(p - pm)
                mrec = {"probs": pm.tolist(), "argmax": int(pm.argmax()), "argmax_equal": int(pm.argmax()) == int(p.argmax()),
                        "max_abs_dp": float(dm.max()),
                        "letter_logits_bit_equal": bool(np.array_equal(ml.astype(np.float16).view(np.uint16),
                                                                       np.asarray(bits, np.uint16))),
                        "full_vocab_top1_equal": ma["full_vocab_top1_id"] == a["full_vocab_top1_id"],
                        "read_from_equal": ma.get("read_from") == a["read_from"]}
                if full is not None and mfull is not None:
                    mrec["full_logits_bit_equal"] = bool(np.array_equal(full[s].view(np.uint16), mfull[s].view(np.uint16)))
                    mrec["full_logits_max_abs_diff"] = float(np.abs(full[s].astype(np.float32) - mfull[s].astype(np.float32)).max())
                rec["mac"] = mrec
            if cold_logits is not None and key in cold_logits and full is not None:
                rec["cold_full_logits_bit_equal"] = bool(np.array_equal(full[s].view(np.uint16), cold_logits[key][s].view(np.uint16)))
            slots.append(rec)
        flat = np.concatenate(deltas) if deltas else np.zeros(0)
        slot_means = [float(x.mean()) for x in deltas]
        run_rows.append({"id": key[0], "arm": key[1], "tokens": r["tokens"], "max_abs_dp": float(flat.max()) if flat.size else None,
                         "mean_abs_dp": float(flat.mean()) if flat.size else None,
                         "mean_of_slot_means": float(np.mean(slot_means)) if slot_means else None,
                         "ids_equal_oracle": proc == o["ids"], "slots_equal_oracle": r["slots"] == o["slot_idx"],
                         "ids_equal_mac": (m["ids"] == r["ids"]) if m else None,
                         "tower_embeds_sha256_equal_mac": (r.get("tower_embeds_sha256") == m.get("tower_embeds_sha256")) if m and key[1] != "text" else None,
                         "tower_embeds_dump_bit_equal_mac": emb_equal, "tower_embeds_max_abs_diff_mac": emb_max,
                         "pixels_equal_mac": (r.get("resized_rgb_sha256") == m.get("resized_rgb_sha256")
                                              and r.get("patches_sha256") == m.get("patches_sha256")) if m and key[1] != "text" else None,
                         "decoded_rgb_equal_meta": r.get("decoded_rgb_sha256_equals_meta"), "decode_path": r.get("decode_path"),
                         "wall_from_file_s": r["wall_from_file_s"], "t_start_s": r.get("t_start_s"),
                         "tower_s": r["seconds"].get("tower"), "decoder_s": r["seconds"].get("decoder"),
                         "calls": r["calls"], "thermal": r.get("thermal"), "footprint_mb": r.get("footprint_mb")})
    return {"result": result, "slots": slots, "runs": run_rows, "logits": logits_by_key}


def summarize(slots: list[dict], runs: list[dict], n_runs_expected: int, n_slots_expected: int) -> dict:
    mac = [x["mac"] for x in slots if "mac" in x]
    s = {"runs": len(runs), "runs_expected": n_runs_expected, "slots": len(slots), "slots_expected": n_slots_expected,
         "ids_equal_oracle": sum(r["ids_equal_oracle"] for r in runs),
         "slots_equal_oracle": sum(r["slots_equal_oracle"] for r in runs),
         "argmax_equal": sum(x["argmax_equal"] for x in slots),
         "full_vocab_top1_is_oracle_letter": sum(x["full_vocab_top1_is_oracle_letter"] for x in slots),
         "max_abs_dp": max((x["max_abs_dp"] for x in slots), default=None),
         "mean_of_run_mean_abs_dp": float(np.mean([r["mean_abs_dp"] for r in runs])) if runs else None,
         "mean_of_run_mean_of_slot_means": float(np.mean([r["mean_of_slot_means"] for r in runs])) if runs else None,
         "finite_all": all(x["finite"] for x in slots) and all(x.get("finite_full", True) for x in slots),
         "probs_app_equal_recomputed": sum(x["probs_app_equal_recomputed"] for x in slots),
         "mac": {"slots": len(mac), "argmax_equal": sum(m["argmax_equal"] for m in mac),
                 "max_abs_dp": max((m["max_abs_dp"] for m in mac), default=None),
                 "median_abs_dp": float(np.median([m["max_abs_dp"] for m in mac])) if mac else None,
                 "letter_logits_bit_equal": sum(m["letter_logits_bit_equal"] for m in mac),
                 "full_logits_bit_equal": sum(m.get("full_logits_bit_equal", False) for m in mac),
                 "full_logits_compared": sum("full_logits_bit_equal" in m for m in mac),
                 "full_logits_max_abs_diff": max((m["full_logits_max_abs_diff"] for m in mac if "full_logits_max_abs_diff" in m), default=None),
                 "full_vocab_top1_equal": sum(m["full_vocab_top1_equal"] for m in mac),
                 "ids_equal_runs": sum(bool(r["ids_equal_mac"]) for r in runs),
                 "tower_embeds_equal_runs": sum(bool(r["tower_embeds_sha256_equal_mac"]) for r in runs if r["arm"] != "text"),
                 "tower_runs": sum(r["arm"] != "text" for r in runs),
                 "tower_embeds_max_abs_diff": max((r["tower_embeds_max_abs_diff_mac"] for r in runs
                                                   if r["tower_embeds_max_abs_diff_mac"] is not None), default=None),
                 "pixels_equal_runs": sum(bool(r["pixels_equal_mac"]) for r in runs if r["arm"] != "text")},
         "decoded_rgb_equal_meta_runs": sum(bool(r["decoded_rgb_equal_meta"]) for r in runs if r["arm"] != "text"),
         "decode_paths": sorted({r["decode_path"] for r in runs if r["decode_path"]})}
    if slots and "cold_full_logits_bit_equal" in slots[0]:
        s["cold_full_logits_bit_equal"] = sum(x.get("cold_full_logits_bit_equal", False) for x in slots)
    worst = max(slots, key=lambda x: x["max_abs_dp"]) if slots else None
    if worst:
        s["worst"] = {k: worst[k] for k in ("id", "arm", "t", "max_abs_dp", "probs", "probs_oracle")}
    walls = [r["wall_from_file_s"] * 1e3 for r in runs]
    s["wall_ms"] = {"median": float(np.median(walls)) if walls else None, "min": min(walls, default=None),
                    "max": max(walls, default=None)}
    return s


def bar_checks(s: dict, resets: dict) -> dict:
    n = s["slots_expected"]
    return {"all_runs": s["runs"] == s["runs_expected"], "slots": s["slots"] == n,
            "ids_equal_oracle": s["ids_equal_oracle"] == s["runs_expected"],
            "slots_equal_oracle": s["slots_equal_oracle"] == s["runs_expected"],
            "argmax": s["argmax_equal"] == n, "full_vocab_top1": s["full_vocab_top1_is_oracle_letter"] == n,
            "max_abs_dp": s["max_abs_dp"] is not None and s["max_abs_dp"] <= BAR["max_abs_dp"],
            "mean_of_run_mean_abs_dp": s["mean_of_run_mean_abs_dp"] is not None
                and s["mean_of_run_mean_abs_dp"] <= BAR["mean_of_run_mean_abs_dp"],
            "finite": s["finite_all"], "probs_recomputed": s["probs_app_equal_recomputed"] == s["slots"],
            "reset_bit_equal": bool(resets) and all(resets.values())}


# --------------------------------------------------------------------------- #
# loads, times, device series
# --------------------------------------------------------------------------- #
def load_rows(result: dict, tag: str) -> list[dict]:
    st = result.get("stages", {})
    rows = []

    def mem(m):
        return {"peak_footprint_mb": m.get("peak_footprint_mb"), "min_available_mb": m.get("min_available_mb"),
                "footprint_mb_start": m.get("footprint_mb_start"), "available_mb_start": m.get("available_mb_start")}

    l1 = st.get("load1", {})
    if "tower_g256_alone" in l1:
        t = l1["tower_g256_alone"]
        rows.append({"run": tag, "step": "load1 (a) g256 tower alone", "wall_s": t.get("wall_s"), "tower_s": t.get("load_s"),
                     **mem(t.get("memory", {}))})
    for key, label in (("decoder_alone", "load1 (b) decoder alone"), ("decider_g256", "load1 (c) decider g256")):
        if key in l1:
            d = l1[key]
            rows.append({"run": tag, "step": label, "wall_s": d.get("wall_s"), "tokenizer_s": d.get("tokenizer_s"),
                         "decoder_s": d.get("decoder_s"), "tower_s": d.get("tower_s"), **mem(d.get("memory", {}))})
    for k, label in (("load_g448", "load_g448 decider g448"), ("load2", "load2 decider g256 (same process)")):
        d = st.get(k, {}).get("decider")
        if d:
            rows.append({"run": tag, "step": label, "wall_s": d.get("wall_s"), "tokenizer_s": d.get("tokenizer_s"),
                         "decoder_s": d.get("decoder_s"), "tower_s": d.get("tower_s"), **mem(d.get("memory", {}))})
    rows_cache = {"load1_cache_mb": {k: l1.get(k, 0) / 1e6 for k in ("cache_bytes_before", "cache_bytes_after_tower_alone",
                                                                      "cache_bytes_after_decoder_alone", "cache_bytes_after_decider")},
                  "load_g448_cache_mb": {k: st.get("load_g448", {}).get(k, 0) / 1e6 for k in ("cache_bytes_before", "cache_bytes_after")}}
    for r in rows:
        r["cache"] = rows_cache
    return rows


def bench_rows(result: dict, tag: str) -> list[dict]:
    out = []
    for stage in ("bench_g256", "bench_g448"):
        for key, row in result.get("stages", {}).get(stage, {}).get("rows", {}).items():
            dec = [d for d in row.get("decisions", []) if not d.get("warmup")]
            walls = [d["wall_from_file_s"] * 1e3 for d in dec]
            out.append({"run": tag, "row": key, "decisions_timed": len(dec),
                        "wall_ms_median": float(np.median(walls)) if walls else None,
                        "wall_ms_min": min(walls, default=None), "wall_ms_max": max(walls, default=None),
                        "wall_ms_each": walls, "start_offsets_s": [d["t_start_s"] for d in dec],
                        "tower_ms_median": float(np.median([d["tower_s"] * 1e3 for d in dec if d.get("tower_s") is not None]))
                            if any(d.get("tower_s") is not None for d in dec) else None,
                        "decoder_ms_median": float(np.median([d["decoder_s"] * 1e3 for d in dec])) if dec else None,
                        "warmup_decision_ms": next((d["wall_from_file_s"] * 1e3 for d in row.get("decisions", []) if d.get("warmup")), None),
                        "thermal_start": row.get("thermal_start"), "thermal_end": row.get("thermal_end"),
                        "thermal_each": [d.get("thermal") for d in dec],
                        "battery_start": row.get("battery_start"), "battery_end": row.get("battery_end"),
                        "wait_nominal": row.get("wait_nominal"), "timed_runs_end_s": row.get("summary", {}).get("timed_runs_end_s"),
                        "letter_logits_equal_e2e": row.get("summary", {}).get("letter_logits_equal_e2e")})
    return out


def device_series(result: dict) -> dict:
    st = result.get("stages", {})
    out = {"e2e_timeline_columns": ["t_s", "thermal", "battery_level", "battery_state", "footprint_mb", "available_mb"]}
    for arm in ARMS:
        e = st.get(E2E_STAGES[arm], {})
        out[f"e2e_{arm}_timeline"] = e.get("timeline", [])
        m = e.get("memory", {})
        out[f"e2e_{arm}_memory"] = {k: m.get(k) for k in ("peak_footprint_mb", "min_available_mb", "duration_s", "samples")}
    return out


def crash_record(run_dir: Path) -> dict:
    """What the phone's crash reports say (run dir crash/*.ips): exception, termination, the faulting thread's frames."""
    out = []
    for f in sorted((run_dir / "crash").glob("*.ips")):
        raw = f.read_text()
        first, rest = raw.split("\n", 1)
        hdr = json.loads(first)
        rec = {"file": str(f), "sha256": sha256_file(f), "bug_type": hdr.get("bug_type"), "timestamp": hdr.get("timestamp"),
               "os_version": hdr.get("os_version")}
        try:
            body = json.loads(rest)
        except json.JSONDecodeError:
            lines = [ln for ln in rest.splitlines() if ln.split(":", 1)[0].strip() in
                     ("Event", "Action taken", "Writes", "Writes limit", "Duration", "Memory size")]
            rec["text_summary"] = lines
            out.append(rec)
            continue
        imgs = body.get("usedImages", [])
        ft = body.get("faultingThread")
        frames = []
        if ft is not None and ft < len(body.get("threads", [])):
            th = body["threads"][ft]
            rec["faulting_thread"] = {"index": ft, "queue": th.get("queue"), "name": th.get("name")}
            for fr in th.get("frames", [])[:28]:
                img = imgs[fr["imageIndex"]] if fr.get("imageIndex") is not None and fr["imageIndex"] < len(imgs) else {}
                frames.append(f"{img.get('name', '?')} {fr.get('symbol', '?')}")
        rec.update({"exception": body.get("exception"), "termination": body.get("termination"), "asi": body.get("asi"),
                    "proc_launch": body.get("procLaunch"), "capture_time": body.get("captureTime"),
                    "model_code": body.get("modelCode"), "frames": frames})
        out.append(rec)
    return {"reports": out}


def memory_tsv_record(run_dir: Path) -> dict:
    """memory.tsv per sampler label: readings, peak footprint, least available, the last reading, and the full series."""
    p = run_dir / "memory.tsv"
    if not p.exists():
        return {}
    rows = [ln.split("\t") for ln in p.read_text().splitlines()[1:] if ln.strip()]
    by = {}
    for t_app, label, t, fp, av, th in rows:
        by.setdefault(label, []).append([float(t_app), float(t), float(fp), float(av), th])
    out = {"path": str(p), "sha256": sha256_file(p), "columns": ["t_app_s", "t_s", "footprint_mb", "available_mb", "thermal"]}
    for label, xs in by.items():
        out[label] = {"readings": len(xs), "peak_footprint_mb": max(x[2] for x in xs),
                      "least_available_mb": min(x[3] for x in xs), "first": xs[0], "last": xs[-1], "series": xs}
    return out


def stopped_record(run_dir: Path) -> dict:
    """A run that did not reach its e2e stages: where it stopped, and why."""
    result = json.loads((run_dir / "result.json").read_text())
    st = result.get("stages", {})
    l1 = st.get("load1", {})
    return {"run_dir": str(run_dir), "run_id": result.get("run_id"), "status": result.get("status"),
            "device": result.get("device"), "config": result.get("config"), "options": result.get("options"),
            "stage_order": result.get("stage_order"), "assets": {k: st.get("assets", {}).get(k) for k in (
                "md5sums_listed", "missing", "md5_mismatch", "md5_checked", "md5_deferred", "free_gb")},
            "load1_partial": {k: l1.get(k) for k in ("step", "partial", "free_gb_before", "footprint_mb_before",
                                                      "available_mb_before", "battery_start", "cache_bytes_before",
                                                      "cache_bytes_after_tower_alone", "tower_g256_alone", "thermal")},
            "result_log": (run_dir / "result.log").read_text().splitlines(), "run_out": (run_dir / "run.out").read_text().splitlines(),
            "memory": memory_tsv_record(run_dir), "crash": crash_record(run_dir)}


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def fmt(v, f="{:.6f}"):
    return f.format(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else "-"


def tables(rec: dict) -> str:
    lines = []
    cold = rec["cold"]
    lines.append("Table 1: device e2e vs the fp32 oracle and vs the Mac Swift JIT (cold run)")
    lines.append("| arm | runs | slots | argmax | full-vocab top-1 | max abs dp | mean (r4) | mean (slot means) | "
                 "Mac argmax | Mac abs dp max | Mac abs dp median | letter logits = Mac | tower = Mac |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in (*ARMS, "all"):
        s = cold["per_arm"][arm] if arm != "all" else cold["summary"]
        m = s["mac"]
        tower = f"{m['tower_embeds_equal_runs']}/{m['tower_runs']}" if m["tower_runs"] else "-"
        lines.append(f"| {arm} | {s['runs']} | {s['slots']} | {s['argmax_equal']} | {s['full_vocab_top1_is_oracle_letter']} | "
                     f"{fmt(s['max_abs_dp'])} | {fmt(s['mean_of_run_mean_abs_dp'])} | {fmt(s['mean_of_run_mean_of_slot_means'])} | "
                     f"{m['argmax_equal']}/{m['slots']} | {fmt(m['max_abs_dp'])} | {fmt(m['median_abs_dp'], '{:.2e}')} | "
                     f"{m['letter_logits_bit_equal']}/{m['slots']} | {tower} |")
    lines.append("")
    lines.append("Table 2a: loads (USB power throughout)")
    lines.append("| run | step | wall s | decoder s | tower s | peak footprint MB | least available MB |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rec["loads"]:
        dec = r.get("decoder_s") or {}
        tw = r.get("tower_s")
        tws = ", ".join(f"{k} {v:.2f}" for k, v in sorted(tw.items())) if isinstance(tw, dict) else fmt(tw, "{:.2f}")
        lines.append(f"| {r['run']} | {r['step']} | {fmt(r['wall_s'], '{:.2f}')} | {fmt(dec.get('total'), '{:.2f}')} | {tws or '-'} | "
                     f"{fmt(r['peak_footprint_mb'], '{:.0f}')} | {fmt(r['min_available_mb'], '{:.0f}')} |")
    lines.append("")
    lines.append("Table 2b: one decision (wall from the image file; 1 warm-up + 5 back to back, after a rest; USB power)")
    lines.append("| run | row | median ms | min-max ms | tower ms | decoder ms | thermal | battery | waited for nominal s |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for b in rec["bench"]:
        b0, b1 = b.get("battery_start") or {}, b.get("battery_end") or {}
        wn = b.get("wait_nominal") or {}
        lines.append(f"| {b['run']} | {b['row']} | {fmt(b['wall_ms_median'], '{:.1f}')} | {fmt(b['wall_ms_min'], '{:.1f}')}-"
                     f"{fmt(b['wall_ms_max'], '{:.1f}')} | {fmt(b['tower_ms_median'], '{:.1f}')} | {fmt(b['decoder_ms_median'], '{:.1f}')} | "
                     f"{b['thermal_start']} -> {b['thermal_end']} | {fmt(b0.get('level', -1) * 100, '{:.0f}')}->"
                     f"{fmt(b1.get('level', -1) * 100, '{:.0f}')} % {b1.get('power', '?')} | {fmt(wn.get('waited_s'), '{:.0f}')} |")
    return "\n".join(lines)


def score_one(run_dir: Path, orc: dict, mac: dict, mac_dump: Path, cold_logits=None) -> dict:
    sc = score_run_set(run_dir, orc, mac, mac_dump, cold_logits)
    result = sc["result"]
    n_slots = sum(len(o["slot_idx"]) for o in orc.values())
    summary = summarize(sc["slots"], sc["runs"], len(orc), n_slots)
    per_arm = {}
    for arm in ARMS:
        o_arm = {k: v for k, v in orc.items() if k[1] == arm}
        per_arm[arm] = summarize([x for x in sc["slots"] if x["arm"] == arm], [r for r in sc["runs"] if r["arm"] == arm],
                                 len(o_arm), sum(len(o["slot_idx"]) for o in o_arm.values()))
    st = result.get("stages", {})
    resets = {p: bool(st.get(f"reset_{p}", {}).get("bit_equal")) for p in ("g256", "g448") if f"reset_{p}" in st}
    checks = bar_checks(summary, resets)
    return {"run_dir": str(run_dir), "run_id": result.get("run_id"), "status": result.get("status"),
            "app_pass": result.get("pass"), "stage_verdicts": result.get("summary"), "device": result.get("device"),
            "device_end": result.get("device_end"), "config": result.get("config"), "options": result.get("options"),
            "summary": summary, "per_arm": per_arm, "resets": resets,
            "reset_detail": {p: {k: st.get(f"reset_{p}", {}).get(k) for k in ("run", "bit_equal", "slot_logits_max_abs_diff")}
                             for p in resets},
            "load2_check_bit_equal_e2e": st.get("load2", {}).get("check_bit_equal_e2e"),
            "checks": checks, "result": "PASS" if all(checks.values()) else "FAIL",
            "assets": {k: st.get("assets", {}).get(k) for k in ("md5sums_listed", "missing", "md5_mismatch", "md5_checked",
                                                                  "md5_deferred", "free_gb", "decoder_bytes", "tower_g256_bytes",
                                                                  "tower_g448_bytes")},
            "md5_stage": {k: st.get("md5", {}).get(k) for k in ("files", "bytes", "md5_mismatch", "seconds", "pass")},
            "warmup": {k: (st.get("warmup", {}).get("run") or {}).get(k) for k in ("wall_from_file_s", "seconds", "thermal")},
            "series": device_series(result), "memory_tsv": str(run_dir / "memory.tsv"),
            "slots_detail": sc["slots"], "runs_detail": sc["runs"], "_logits": sc["logits"]}


def staged_md5() -> dict:
    """The staged model files' md5 (MD5SUMS, what the phone checked against) against the export files they clone."""
    sums = {}
    if (STAGE_DIR / "MD5SUMS").exists():
        for line in (STAGE_DIR / "MD5SUMS").read_text().splitlines():
            h, rel = line.split(" ", 1)
            sums[rel.strip()] = h
    files = {}
    for rel, src in SOURCES.items():
        files[rel] = {"md5_staged": sums.get(rel), "md5_source": md5_file(src) if src.exists() else None, "source": str(src)}
        files[rel]["equal"] = files[rel]["md5_staged"] == files[rel]["md5_source"]
    return {"stage_dir": str(STAGE_DIR), "md5sums_sha256": sha256_file(STAGE_DIR / "MD5SUMS")
            if (STAGE_DIR / "MD5SUMS").exists() else None, "files": files}


def aot_record(path: Path) -> dict:
    """The decoder's iPhone AOT compile (round 9 D): files and bytes, and the compile log's last lines."""
    files = sorted(p for p in path.rglob("*") if p.is_file())
    log = LANE / "logs" / "r9_aot_h19p.log"
    tail = [ln for ln in log.read_text().splitlines() if ln.strip()][-6:] if log.exists() else []
    return {"path": str(path), "bytes": sum(p.stat().st_size for p in files), "files": len(files),
            "top_level": sorted(p.name for p in path.iterdir()) if path.exists() else [],
            "stats_json": json.loads((path / "stats.json").read_text()) if (path / "stats.json").exists() else None,
            "compile_log": str(log), "compile_log_tail": tail}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cold", required=True, help="device run dir of the cold run (fresh container)")
    ap.add_argument("--warm", help="device run dir of the warm run (same container)")
    ap.add_argument("--aot-dir", help="the decoder's h19p .aimodelc (round 9 D: compile result and size)")
    ap.add_argument("--note", action="append", default=[], help="a line for the transcript's notes (repeatable)")
    ap.add_argument("--stopped", action="append", default=[],
                    help="a device run dir that stopped before any decision (kept in the transcript as stopped_runs)")
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--tables", help="write the tables (markdown) here too")
    args = ap.parse_args()

    orc = oracle_runs()
    mac, mac_dump, mac_json = mac_runs()
    staged = staged_md5()
    aot = aot_record(Path(args.aot_dir)) if args.aot_dir else None
    cres = json.loads((Path(args.cold) / "result.json").read_text())
    if not any(cres.get("stages", {}).get(E2E_STAGES[a], {}).get("runs") for a in ARMS):
        # the run stopped before any decision: record where and why, nothing to score
        st = stopped_record(Path(args.cold))
        rec = {"schema": "coreai-decider-vision-iphone-gate/1", "result": "STOPPED",
               "gate": "iPhone 18 Pro: the shipped set (decoder int8mix_pf16 + towers g256 / g448 fp16w32, JIT .aimodel "
                       "specialized on the phone) through the DeciderVision library",
               "stopped": st, "staged_md5": staged, "aot_h19p": aot,
               "bar": {**BAR, "source": "readout_gate_vision.BAR (round 4, unchanged)"},
               "oracle": {"path": str(ORACLE), "sha256": sha256_file(ORACLE)},
               "mac_swift_jit": {"path": str(MAC_JIT), "sha256": sha256_file(MAC_JIT)},
               "power_note": "the iPhone 18 Pro is on USB power for every number (battery state charging / full)",
               "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        Path(args.transcript).parent.mkdir(parents=True, exist_ok=True)
        Path(args.transcript).write_text(json.dumps(rec, indent=1) + "\n")
        mem = st["memory"]
        print(f"STOPPED run {st['run_id']}: stages {st['stage_order']}, load1 step {st['load1_partial'].get('step')}")
        for label in [k for k in mem if k not in ("path", "sha256", "columns")]:
            m = mem[label]
            print(f"  memory {label}: {m['readings']} readings, peak footprint {m['peak_footprint_mb']:.1f} MB, least available "
                  f"{m['least_available_mb']:.1f} MB, last {m['last']}")
        for r in st["crash"]["reports"]:
            print(f"  crash {Path(r['file']).name}: bug_type {r['bug_type']} {r.get('exception') or ''} "
                  f"{(r.get('termination') or {}).get('indicator', '')} {r.get('text_summary', '')}")
            for fr in r.get("frames", [])[8:16]:
                print(f"     {fr}")
        print(f"staged md5 = source: {all(v['equal'] for v in staged['files'].values())} | aot {aot and aot.get('bytes')} | "
              f"transcript {args.transcript}: STOPPED")
        return 4
    cold = score_one(Path(args.cold), orc, mac, mac_dump)
    warm = score_one(Path(args.warm), orc, mac, mac_dump, cold_logits=cold["_logits"]) if args.warm else None

    loads = load_rows(cres, "cold")
    bench = bench_rows(cres, "cold")
    if warm:
        wres = json.loads((Path(args.warm) / "result.json").read_text())
        loads += load_rows(wres, "warm")
        bench += bench_rows(wres, "warm")
    for x in (cold, warm):
        if x:
            x.pop("_logits", None)
    rec = {"schema": "coreai-decider-vision-iphone-gate/1",
           "gate": "iPhone 18 Pro: the shipped set (decoder int8mix_pf16 + towers g256 / g448 fp16w32, JIT .aimodel "
                   "specialized on the phone) through the DeciderVision library, vs the author's fp32 oracle and the Mac "
                   "Swift JIT of the same assets",
           "oracle": {"path": str(ORACLE), "sha256": sha256_file(ORACLE)},
           "mac_swift_jit": {"path": str(MAC_JIT), "sha256": sha256_file(MAC_JIT), "label": mac_json.get("label"),
                             "assets": mac_json.get("assets"), "environment": mac_json.get("environment")},
           "bar": {**BAR, "source": "readout_gate_vision.BAR (round 4, unchanged)", "slots": "all 108, equal the oracle's",
                   "argmax": "all", "full_vocab_top1": "the oracle's argmax letter on every slot",
                   "mean_definition": "mean over runs of the run's mean |dp| over all its (slot, option) pairs (round 4); "
                                      "mean_of_run_mean_of_slot_means shown beside it",
                   "reset": "each phase's first run re-run at its end, full slot logits bit-equal"},
           "staged_md5": staged, "aot_h19p": aot,
           "cold": cold, "warm": warm, "loads": loads, "bench": bench,
           "stopped_runs": [stopped_record(Path(p)) for p in args.stopped],
           "notes": args.note,
           "power_note": "the iPhone 18 Pro is on USB power for every number (battery state charging / full)"}
    ok = cold["result"] == "PASS" and (warm is None or warm["result"] == "PASS")
    rec["result"] = "PASS" if ok else "FAIL"
    rec["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t = tables(rec)
    rec["tables_markdown"] = t
    Path(args.transcript).parent.mkdir(parents=True, exist_ok=True)
    Path(args.transcript).write_text(json.dumps(rec, indent=1) + "\n")
    if args.tables:
        Path(args.tables).write_text(t + "\n")
    print(t)
    for tag, x in (("cold", cold), ("warm", warm)):
        if x:
            print(f"{tag} {x['run_id']}: {x['result']} checks {x['checks']} resets {x['resets']} "
                  f"load2 check {x['load2_check_bit_equal_e2e']}"
                  + (f" | cold full logits bit-equal {x['summary'].get('cold_full_logits_bit_equal')}/{x['summary']['slots']}"
                     if tag == "warm" else ""))
    print(f"staged md5 = source: {all(v['equal'] for v in staged['files'].values())} | transcript {args.transcript}: {rec['result']}")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
