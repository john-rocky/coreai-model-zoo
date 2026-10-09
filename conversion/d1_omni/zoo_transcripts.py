#!/usr/bin/env python3
"""Write models/d1-omni-600m/gate-d1-omni-600m-*.json: the gate transcripts of the d1-omni-600M Core AI port.

    PY=~/code/coreai/coreai-models/.venv/bin/python
    $PY conversion/d1_omni/zoo_transcripts.py            # write the 9 transcripts (refuses to replace a different file
                                                         # without --force)
    $PY conversion/d1_omni/zoo_transcripts.py --check    # rebuild, compare, run the controls; write
                                                         # <work>/results/zoo_fixtures_check.json; exit 1 on FAIL

Each transcript is a summary of the lane's gate records ($ZOO_WORK_ROOT/_d1_omni/results/..., "the lane"): what was
gated, the source files (path in the lane, bytes, sha256), the bar, the status, and the summary numbers. The numbers are
computed here from the source files' per-row (per-image, per-clip, per-call) values, the way the gate scripts compute
them (conversion/d1_omni/_metrics.py, timing_run.py `stats`), and checked against the summary each source file wrote;
the per-row values stay in the lane files the sources list names. No model runs here.

  eager            the re-authored modules in fp32 torch on the CPU against the publisher's code (decision graph,
                   L4096's key blocks, vision tower, audio tower and mel), the host rows and resize in NumPy, the
                   activation range against fp16
  runtime-decide   the shipped decision graph (fp16, stripped, AOT h16c) at L64 ... L4096 on the Mac GPU against the
                   publisher's fp32 model, and the stripped vs unstripped comparison
  runtime-vision   the vision graph's prefix and the image rows end to end (vision fp16 -> decision fp16)
  runtime-audio    the audio graph's prefix per clip bucket and the audio rows end to end, routed by the round-10
                   buckets and by host.ALL_BUCKETS (L64 / L128 included)
  swift            the Swift host (apps/D1Omni): token rows, parity against the Python runtime (AOT and the .aimodel),
                   the media path, the L1024 probe, the small buckets
  strip            the debug-location strip of every shipped bundle (op counts, bytes, hashes, the byte scans) and the
                   compiles of the stripped bundles
  timing-mac       time per decision on the Mac GPU in measurement windows (Python runtime and Swift host), medians
                   recomputed from the per-call samples
  forms            every form measured on the way (precision x compute unit x bucket): its gate and its time
  iphone           round 11 on the iPhone 18 Pro (the gate app apps/D1OmniGate): parity of the .aimodel (JIT) and
                   the h19p compiles on a subset of the fixture, one decision's ms recomputed from the samples, the
                   loads, the L4096 compile killed at the per-process memory limit, the Neural Engine attempts, the
                   hold and the transfer (round 14)

Records the HF repository does not publish are named nowhere here: an item that names one (a row id, a per-source
aggregate of a source with no public record) is dropped by stage_hf.Scrub, and local paths become tokens. --check also
re-runs zoo_fixtures.py --check, a control for each check (a mutated copy must fail), and the scans for a local path, an
unpublished id and the measured-only material.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import repo_root, work_path  # noqa: E402
import host  # noqa: E402
import stage_hf  # noqa: E402
import zoo_fixtures  # noqa: E402

MODEL_ID = "LiquidAI/d1-omni-600M"
MODEL_SHA = "414f8d6438174f5b2133a9c21a478fc42625e308"
WORK = work_path("_d1_omni")
OUT_DIR = repo_root() / "models" / "d1-omni-600m"
CHECK_OUT = WORK / "results" / "zoo_fixtures_check.json"
NAMES = ("eager", "runtime-decide", "runtime-vision", "runtime-audio", "swift", "strip", "timing-mac", "forms", "iphone")
LANE = "$ZOO_WORK_ROOT/_d1_omni (the lane: every source path below is relative to it)"
SHIP_BAR = {"name": "FACTS §7 (the ship bar)", "argmax": "equal to the oracle's on every row whose oracle top-2 margin is "
            "above 0.02 (near ties reported apart)", "max_abs_dp": 0.02, "mean_row_max_abs_dp": 0.002,
            "oracle": "the publisher's model, AutoModel.from_pretrained(trust_remote_code=True), CPU, fp32: text rows after "
                      "the config.json temperature, image and audio rows the raw softmax"}
FP32_BAR = {"name": "round 2's eager bar (fp32 against fp32)", "argmax": "every row, near ties included",
            "max_abs_dp": 2e-5, "max_abs_dlogit": 1e-3,
            "basis": "2 x the fp32 oracle's own max |dp| against the publisher's code run in float64 (eager_check.json "
                     "bar_basis), the supervisor's ruling of 2026-10-08 round 2"}
SMALL = host.SMALL_BUCKETS
ALL = host.ALL_BUCKETS


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


class Book:
    """The source files one transcript reads, recorded with their bytes and sha256."""

    def __init__(self, root: Path = WORK):
        self.root = root
        self.files: dict[str, dict] = {}

    def note(self, rel: str) -> Path:
        path = self.root / rel
        if rel not in self.files:
            self.files[rel] = {"file": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        return path

    def load(self, rel: str):
        return json.loads(self.note(rel).read_text())

    def sources(self) -> list[dict]:
        return [self.files[k] for k in sorted(self.files)]


# =========================================================================== numbers from rows
def recompute(rows: list[dict]) -> dict:
    """_metrics.summarize's aggregates and the ship-bar verdict from per-row records (argmax_equal, near_tie,
    max_abs_dp, max_abs_dlogit, finite)."""
    n = len(rows)
    finite = [r for r in rows if r.get("finite", True)]
    clear = [r for r in rows if not r["near_tie"]]
    ties = [r for r in rows if r["near_tie"]]
    dps = [r["max_abs_dp"] for r in rows]
    out = {"rows": n, "finite_rows": len(finite), "argmax_equal": sum(bool(r["argmax_equal"]) for r in rows),
           "non_near_tie": {"rows": len(clear), "argmax_equal": sum(bool(r["argmax_equal"]) for r in clear)},
           "near_tie": {"rows": len(ties), "argmax_equal": sum(bool(r["argmax_equal"]) for r in ties)},
           "max_abs_dp": max(dps) if dps else None,
           "mean_row_max_abs_dp": float(np.mean(dps)) if dps and len(finite) == n else None,
           "max_abs_dlogit": max((r["max_abs_dlogit"] for r in rows), default=None)}
    verdict = {"argmax_non_near_tie": all(r["argmax_equal"] for r in clear),
               "max_abs_dp": out["max_abs_dp"] is not None and out["max_abs_dp"] <= SHIP_BAR["max_abs_dp"],
               "mean_row_max_abs_dp": out["mean_row_max_abs_dp"] is not None
               and out["mean_row_max_abs_dp"] <= SHIP_BAR["mean_row_max_abs_dp"],
               "finite": len(finite) == n}
    out["ship_bar"] = {"verdict": verdict, "status": "PASS" if all(verdict.values()) else "FAIL"}
    return out


def agrees(mine: dict, theirs: dict | None, keys=("rows", "argmax_equal", "max_abs_dp", "mean_row_max_abs_dp",
                                                   "max_abs_dlogit")) -> dict:
    """Recomputed numbers against the summary the source file wrote: {"equal": bool, "different": {key: [mine, file]}}."""
    diff = {}
    if theirs is None:
        return {"equal": False, "different": {"summary": "missing in the source"}}
    for k in keys:
        a, b = mine.get(k), theirs.get(k)
        if isinstance(a, dict):
            for kk in a:
                if a[kk] != (b or {}).get(kk):
                    diff[f"{k}.{kk}"] = [a[kk], (b or {}).get(kk)]
        elif a != b:
            diff[k] = [a, b]
    if "non_near_tie" in theirs and "non_near_tie" in mine:
        for kk in ("rows", "argmax_equal"):
            if mine["non_near_tie"][kk] != theirs["non_near_tie"][kk]:
                diff[f"non_near_tie.{kk}"] = [mine["non_near_tie"][kk], theirs["non_near_tie"][kk]]
    return {"equal": not diff, "different": diff}


def stats(values) -> dict:
    """timing_run.py `stats`: numpy median / 10th / 90th percentile of the per-decision samples."""
    a = np.asarray(values, dtype=np.float64)
    return {"n": int(a.size), "median": float(np.median(a)), "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90))}


def status_of(*parts) -> str:
    return "PASS" if all(p == "PASS" or p is True for p in parts) else "FAIL"


# =========================================================================== eager
def t_eager() -> dict:
    b = Book()
    eager = b.load("results/eager_check.json")
    mine = recompute([dict(r, finite=True) for r in eager["rows"]])
    file_sum = eager["summary"]
    fp32_verdict = {"argmax": mine["argmax_equal"] == mine["rows"],
                    "max_abs_dp": mine["max_abs_dp"] <= FP32_BAR["max_abs_dp"],
                    "max_abs_dlogit": mine["max_abs_dlogit"] <= FP32_BAR["max_abs_dlogit"]}
    near = [r for r in eager["rows"] if r["near_tie"]]
    decision = {
        "what": "d1_omni_model.py (the decision graph's module: bidirectional LFM2 trunk + decision head, an explicit "
                "mask and key-padding input, fp32) on the CPU against the oracle, every fixture row of round 2 (text "
                "rows and the media rows with the oracle's own prefix), at each row's bucket",
        "rows": mine["rows"], "argmax_equal": mine["argmax_equal"],
        "near_tie": {"rows": len(near), "argmax_equal": sum(r["argmax_equal"] for r in near)},
        "max_abs_dp": mine["max_abs_dp"], "max_abs_dlogit": mine["max_abs_dlogit"],
        "bar": FP32_BAR, "verdict": fp32_verdict, "status": status_of(*fp32_verdict.values(), eager["pass"]),
        "recomputed_vs_file": agrees(mine, {"rows": file_sum["rows"], "argmax_equal": file_sum["argmax_equal"],
                                            "max_abs_dp": file_sum["max_abs_dp"], "max_abs_dlogit": file_sum["max_abs_dlogit"]},
                                     keys=("rows", "argmax_equal", "max_abs_dp", "max_abs_dlogit")),
        "bar_basis": {k: eager["bar_basis"][k] for k in ("rule", "ref_vs_pub64_max_abs_dp", "mine_vs_pub64_max_abs_dp",
                                                          "mine64_vs_pub64_max_abs_dp", "twice_ref_vs_pub64",
                                                          "pure_float64_same_rope_length_max_abs_dlogit",
                                                          "publisher_padded_to_bucket_max_abs_dp",
                                                          "publisher_batch_partners_max_abs_dp")},
        "pad_content": {"bit_identical": eager["verdict"]["b_pad_content_bit_identical"],
                        "what": "pad ids and pad / non-prefix prefix_embeds replaced (5 forms on 20 rows): real-position "
                                "scores bit-identical"},
        "bucket_change": {"marker_max_abs_d": eager["c_bucket"]["marker_max_abs_d"],
                          "bit_identical_rows": eager["c_bucket"]["bit_identical_rows"],
                          "what": "the same rows at two bucket lengths (256 <-> 512, 2048 <-> 4096)"},
        "gqa_repeat_vs_broadcast": {k: v for k, v in eager["d_gqa"].items() if not isinstance(v, list)},
        "red_arms": {"what": "mutations of the graph's masks on the media rows: each must move the marker logits above "
                             "1e-3 on every row",
                     "arms": {k: {kk: v[kk] for kk in ("max_abs_dp", "max_abs_dlogit", "min_abs_dlogit",
                                                       "rows_over_logit_bar")} for k, v in eager["e_mutations"].items()},
                     "caught_every_row": eager["verdict"]["e_mutations_over_logit_bar_every_row"]},
    }

    bs = b.load("results/block_softmax_check.v2.json")
    block = {"what": "L4096 attention in two blocks of 2,048 keys (d1_omni_model.KEY_BLOCK) against the plain softmax: "
                     "float64 = the same function; fp32 against the oracle on the rows of 2,048 / 4,096 positions",
             "status": bs["status"], "bar": bs["bar"],
             "float64_max_abs_d_markers": bs["full"]["same_function_float64"]["max_abs_d_markers"],
             "toy_float64_max_abs_d_scores": bs["toy"]["max_abs_d_scores_float64"],
             "blocked_vs_reference": {k: v for k, v in bs["full"]["blocked_vs_reference"].items()
                                      if not isinstance(v, (list, dict))},
             "rows_by_bucket": bs["full"]["rows_by_bucket"]}

    ve = b.load("results/vision_eager.run2.json")
    vmin = min(i["prefix"]["cos"] for i in ve["images"])
    vision = {"what": "d1_omni_vision.py (SigLIP2 NaFlex tower + projector, one crop -> 256 rows, fp32) on the CPU "
                      "against the publisher's vision code, 16 images / 62 crops",
              "status": ve["status"], "bars": ve["bars"],
              "prefix_min_cos": vmin, "prefix_min_cos_file": ve["summary"]["prefix_min_cos"],
              "recomputed_equal": vmin == ve["summary"]["prefix_min_cos"],
              **{k: ve["summary"][k] for k in ("images", "crops", "rows_equal_reference", "pixel_values_bit_equal",
                                               "pos_numpy_bit_equal", "prefix_max_abs", "float64_mine_vs_publisher_max_abs",
                                               "fp32_floor_publisher_max_abs", "unshuffle_bit_equal", "red_arm_caught")}}

    ae = b.load("results/audio_eager.json")
    amin = min(c["prefix"]["cos"] for c in ae["clips"])
    audio = {"what": "d1_omni_audio.py (subsampling + 17 FastConformer layers + adapter + residual, clip buckets 5 / 10 / "
                     "20 / 30 s, fp32) on the CPU against the publisher's audio code, 22 clips",
             "status": ae["status"], "bars": ae["bars"], "prefix_min_cos": amin,
             "prefix_min_cos_file": ae["summary"]["prefix_min_cos"], "recomputed_equal": amin == ae["summary"]["prefix_min_cos"],
             **{k: ae["summary"][k] for k in ("clips", "rows_equal_reference", "host_lengths_equal", "prefix_max_abs",
                                              "float64_mine_vs_publisher_max_abs", "fp32_floor_publisher_max_abs")}}
    mel = b.load("results/audio_mel.json")
    mel_items = mel["clips"] + [c for c in (mel.get("synthetic") or []) if isinstance(c, dict)
                                and "same_function_numpy64_vs_torch64_max_abs" in c]
    mel_same = max(c["same_function_numpy64_vs_torch64_max_abs"] for c in mel_items)
    audio["mel"] = {"what": "mel_host.py: mel_torch = the publisher's front end bit for bit; mel_numpy (the Swift host's "
                            "specification, float64) = mel_torch run in float64",
                    "status": mel["status"], "bars": mel["bars"],
                    "same_function_max_abs": mel_same, "same_function_max_abs_file": mel["summary"]["same_function_max_abs"],
                    "recomputed_equal": mel_same == mel["summary"]["same_function_max_abs"],
                    **{k: mel["summary"][k] for k in ("clips", "torch_bit_equal_publisher", "frames_equal",
                                                      "numpy_clips_le_1e-4", "fp32_floor_max_abs")}}

    enc = b.load("results/encode_rows.v3.json")
    rz = b.load("results/resize_numpy_gate.json")
    crops = [c for i in rz["images"] for c in i["crops"]]
    host_np = {"rows": {"what": "host.py request_rows (the publisher's prompt.py copied, the tokenizer without the "
                                "post-processor) against the publisher's prompt.encode and system_one_batch, fixture v3",
                        "encode_rows_equal": enc["checks"]["encode_rows_equal"],
                        "dispatch": {k: enc["checks"]["dispatch"][k] for k in ("requests", "answers", "answers_equal",
                                                                               "usage_equal")}},
               "resize": {"what": "host.resize_uint8_antialias_numpy (torchvision's float32 antialias kernel, round half "
                                  "to even) against torchvision 0.24 on this Mac, every crop of 16 images",
                          "status": rz["status"], "crops": len(crops), "bit_equal": sum(c["bit_equal"] for c in crops),
                          "file_bit_equal": rz["summary"]["bit_equal"],
                          "recomputed_equal": sum(c["bit_equal"] for c in crops) == rz["summary"]["bit_equal"]}}

    ab = b.load("results/absmax.json")
    absmax = {"what": "the largest activation of the fp32 module over every round-2 row against fp16's largest value",
              "max_abs": ab["scan"]["largest_real_10"][0]["max"], "point": ab["scan"]["largest_real_10"][0]["point"],
              "fp16_max": ab["scan"]["fp16_max"], "headroom": ab["scan"]["fp16_max"] / ab["scan"]["largest_real_10"][0]["max"],
              "rows": ab["scan"]["rows"]}

    checks = [decision["status"], block["status"], vision["status"], vision["recomputed_equal"], audio["status"],
              audio["recomputed_equal"], audio["mel"]["status"], audio["mel"]["recomputed_equal"],
              decision["recomputed_vs_file"]["equal"], host_np["resize"]["status"], host_np["resize"]["recomputed_equal"],
              enc["checks"]["encode_rows_equal"].split("/")[0] == enc["checks"]["encode_rows_equal"].split("/")[1]]
    return {"what": "The re-authored modules in fp32 torch on the CPU against the publisher's code before any export "
                    "(rounds 1-6): the decision graph, its L4096 key blocks, the vision and audio towers and the mel front "
                    "end, the host's rows and resize in NumPy, and the activation range against fp16.",
            "lane": LANE, "sources": b.sources(), "status": status_of(*checks),
            "decision_graph": decision, "decision_graph_L4096_blocks": block, "vision_tower": vision,
            "audio_tower": audio, "host_numpy": host_np, "activation_range": absmax}


# =========================================================================== runtime: decision
def bucket_gate(b: Book, rel: str) -> dict:
    doc = b.load(rel)
    mine = recompute(doc["rows"])
    sets = {}
    for r in doc["rows"]:
        sets[r.get("set", "native")] = sets.get(r.get("set", "native"), 0) + 1
    c = doc["compiled"]
    return {"source": rel, "status": doc["status"], "rows": mine["rows"], "rows_by_set": sets,
            "argmax_equal": mine["argmax_equal"], "non_near_tie": mine["non_near_tie"], "near_tie": mine["near_tie"],
            "max_abs_dp": mine["max_abs_dp"], "mean_row_max_abs_dp": mine["mean_row_max_abs_dp"],
            "max_abs_dlogit": mine["max_abs_dlogit"], "ship_bar": mine["ship_bar"],
            "recomputed_vs_file": agrees(mine, doc["summary"]),
            "wrong_pairing_control": {k: doc["wrong_pairing_control"][k] for k in ("status", "must_be", "caught",
                                                                                   "paired_rows", "rows_over_dp_bar")},
            "repeat_drift": {k: doc["repeat_drift"][k] for k in ("repeats_per_row", "max_abs_scores", "rows_nonzero")},
            "compiled": {"aimodelc": c["aimodelc"], "architecture": c["architecture"], "bytes": c["bytes"],
                         "resources_bin_bytes": c["resources_bin_bytes"], "main_hash": c["hashes"].get("main.hash"),
                         "ane_regions": c["ane_regions"]},
            "specialization_options": doc["specialization_options"], "load_s": doc["load_s"],
            "first_call_s": doc["first_call_s"]}


def t_runtime_decide() -> dict:
    b = Book()
    buckets = {}
    for L in ALL:
        g = bucket_gate(b, f"results/ship_runtime_decide_fp16_L{L}_gpu.json")
        cmp_rel = "results/small_runtime_compare.json" if L in SMALL else "results/ship_runtime_compare.json"
        pair = b.load(cmp_rel)["pairs"][f"decide-fp16-L{L}"]
        g["stripped_vs_unstripped"] = {"source": cmp_rel, "status": pair["status"],
                                       "rows_compared": pair["rows"]["paired"],
                                       "logits_and_p_equal": pair["rows"]["logits_and_p_equal"],
                                       "max_abs_dlogit": pair["rows"]["max_abs_dlogit"]}
        buckets[str(L)] = g
    checks = []
    for g in buckets.values():
        checks += [g["status"], g["ship_bar"]["status"], g["recomputed_vs_file"]["equal"],
                   g["wrong_pairing_control"]["caught"], g["stripped_vs_unstripped"]["status"] == "SAME"]
    return {"what": "The shipped decision graph (fp16, debug locations stripped, AOT h16c) on the Mac GPU through the "
                    "Python Core AI runtime (coreai-core 1.0.0b2, SpecializationOptions preferring the GPU), against the "
                    "oracle: per bucket every fixture row whose bucket it is (native), shorter rows padded into it (pad) "
                    "and media rows with the oracle's prefix (prefix); 3 calls per row (repeat drift); a wrong-pairing "
                    "control that must fail; the same rows on the unstripped bundle (round 3-4, round 12).",
            "lane": LANE, "sources": b.sources(), "bar": SHIP_BAR, "status": status_of(*checks),
            "device": "Apple M4 Max, macOS 27.0 (26A428), coreai-build 3600.83.1 (--architecture h16c)",
            "buckets": buckets}


# =========================================================================== runtime: media
def e2e_arms(b: Book, rel: str, arm_ok: str, arm_ref: str, arm_control: str) -> dict:
    doc = b.load(rel)
    out = {"source": rel, "status": doc["status"], "rows": doc["rows"], "arms": {}}
    for arm in (arm_ok, *(a for a in doc["arms"] if a not in (arm_ok, arm_control)), arm_control):
        mine = recompute(doc["arms"][arm])
        by_bucket = {}
        for r in doc["arms"][arm]:
            by_bucket.setdefault(str(r["bucket"]), []).append(r)
        out["arms"][arm] = {"rows": mine["rows"], "argmax_equal": mine["argmax_equal"],
                            "non_near_tie": mine["non_near_tie"], "max_abs_dp": mine["max_abs_dp"],
                            "mean_row_max_abs_dp": mine["mean_row_max_abs_dp"], "max_abs_dlogit": mine["max_abs_dlogit"],
                            "ship_bar": mine["ship_bar"], "recomputed_vs_file": agrees(mine, doc["summaries"].get(arm)),
                            "by_bucket": {L: {k: recompute(rs)[k] for k in ("rows", "argmax_equal", "max_abs_dp",
                                                                            "mean_row_max_abs_dp")}
                                          for L, rs in sorted(by_bucket.items(), key=lambda x: int(x[0]))}}
    out["control_must_fail"] = doc["control_must_fail"]
    out["wrong_pairing_oracle_swap"] = {k: doc["wrong_pairing_oracle_swap"][k] for k in ("status", "must_be", "caught",
                                                                                         "rows_over_dp_bar")}
    out["repeat_drift_max"] = doc["repeat_drift_max"]
    out["decision_graphs"] = {L: {"aimodelc": d["aimodelc"], "main_hash": d["hashes"]["main.hash"]}
                              for L, d in doc["decide"].items()}
    ok = out["arms"][arm_ok]
    out["verdict"] = status_of(doc["status"], ok["ship_bar"]["status"], ok["recomputed_vs_file"]["equal"],
                               out["arms"][arm_control]["ship_bar"]["status"] == "FAIL",
                               out["wrong_pairing_oracle_swap"]["caught"])
    return out


def t_runtime_vision() -> dict:
    b = Book()
    pre = b.load("results/ship_runtime_vision_fp16_gpu.json")
    imgs = pre["images"]
    prefix = {"source": "results/ship_runtime_vision_fp16_gpu.json", "status": pre["status"], "images": len(imgs),
              "crops": sum(i["crops"] for i in imgs), "min_cos": min(i["prefix"]["cos"] for i in imgs),
              "min_row_cos": min(i["prefix"]["min_row_cos"] for i in imgs),
              "max_abs": max(i["prefix"]["max_abs"] for i in imgs), "drift_max": max(i["drift"] for i in imgs),
              "host_pixels_equal_reference": sum(bool(i["host_pixels_equal_reference"]) for i in imgs),
              "what": "the vision graph (fp16, stripped, AOT h16c) on the Mac GPU against the publisher's fp32 vision "
                      "prefix, every crop of 16 images (3 public, 13 measurement-only), 3 calls each"}
    prefix["recomputed_equal"] = (prefix["min_cos"] == pre["summary"]["min_cos"] and
                                  prefix["min_row_cos"] == pre["summary"]["min_row_cos"] and
                                  prefix["max_abs"] == pre["summary"]["max_abs"])
    e2e = e2e_arms(b, "results/ship_runtime_vision_e2e_fp16.json", "vision_bundle_prefix", "reference_prefix",
                   "control_other_image_prefix")
    e2e["what"] = ("46 image rows: the vision graph's prefix -> the decision graph fp16 (buckets 256 / 512 / 2048) on the "
                   "Mac GPU, against the oracle (raw softmax, no temperature); arms: the shipped prefix, the publisher's "
                   "prefix in the same decision graph (reference), another image's prefix (control, must fail)")
    pairs = b.load("results/ship_runtime_compare.json")["pairs"]
    strip = {k: pairs[k]["status"] for k in ("vision-fp16", "vision-e2e")}
    return {"what": "The vision graph (fp16, stripped, AOT h16c) on the Mac GPU: its prefix per crop, and the image rows end "
                    "to end through the fp16 decision graph.", "lane": LANE, "sources": b.sources(), "bar": SHIP_BAR,
            "status": status_of(prefix["status"], prefix["recomputed_equal"], e2e["verdict"],
                                *(s == "SAME" for s in strip.values())),
            "prefix": prefix, "end_to_end": e2e, "stripped_vs_unstripped": strip}


def t_runtime_audio() -> dict:
    b = Book()
    buckets = {}
    for s in (5, 10, 20, 30):
        rel = f"results/ship_runtime_audio_fp16_{s}s_gpu.json"
        doc = b.load(rel)
        clips = doc["clips"]
        row = {"source": rel, "status": doc["status"], "clips": len(clips),
               "min_cos": min(c["prefix"]["cos"] for c in clips), "min_row_cos": min(c["prefix"]["min_row_cos"] for c in clips),
               "max_abs": max(c["prefix"]["max_abs"] for c in clips), "drift_max": max(c["drift"] for c in clips)}
        row["recomputed_equal"] = (row["min_cos"] == doc["summary"]["min_cos"] and row["max_abs"] == doc["summary"]["max_abs"]
                                   and row["min_row_cos"] == doc["summary"]["min_row_cos"])
        buckets[f"{s}s"] = row
    ship = e2e_arms(b, "results/ship_runtime_audio_e2e_fp16.json", "audio_bundle_prefix_numpy_mel", "reference_prefix",
                    "control_next_clip_prefix")
    ship["what"] = ("46 audio rows routed by the round-10 buckets (256 / 512): the audio graph's prefix (mel from the "
                    "NumPy host = the Swift host's specification, and from the publisher's torch front end) -> the "
                    "decision graph fp16, against the oracle (raw softmax); control = the next clip's prefix")
    small = e2e_arms(b, "results/small_runtime_audio_e2e_fp16.json", "audio_bundle_prefix_numpy_mel", "reference_prefix",
                     "control_next_clip_prefix")
    small["what"] = ("the same 46 rows routed by host.ALL_BUCKETS (the shipped set): 9 rows of 128 positions or fewer go "
                     "to L64 / L128")
    pairs = b.load("results/ship_runtime_compare.json")["pairs"]
    strip = {k: pairs[k]["status"] for k in ("audio-fp16-5s", "audio-fp16-10s", "audio-fp16-20s", "audio-fp16-30s",
                                             "audio-e2e")}
    return {"what": "The audio graph (fp16, stripped, AOT h16c, clip buckets 5 / 10 / 20 / 30 s) on the Mac GPU: its prefix "
                    "per clip, and the audio rows end to end through the fp16 decision graph.", "lane": LANE,
            "sources": b.sources(), "bar": SHIP_BAR,
            "status": status_of(*(r["status"] for r in buckets.values()), *(r["recomputed_equal"] for r in buckets.values()),
                                ship["verdict"], small["verdict"], *(s == "SAME" for s in strip.values())),
            "prefix": buckets, "end_to_end_round10_buckets": ship, "end_to_end_all_buckets": small,
            "stripped_vs_unstripped": strip}


# =========================================================================== Swift
def t_swift() -> dict:
    b = Book()
    rows_gate = b.load("results/swift_rows_gate.json")
    rows = {"what": "d1omni rows (Swift: swift-transformers' tokenizer, the prompt rules of host.py, unicodeScalars) "
                    "against the Python host on every row of fixtures v1 / v2 / v3: ids, markers, masks, qtype",
            "acceptance": rows_gate["acceptance"],
            "versions": {v: {"rows": d["rows"]["swift"], "python_rows": d["rows"]["python"],
                             "ids_sha256_equal": d["fields_equal"]["ids_sha256"]} for v, d in rows_gate["versions"].items()},
            "control": {k: rows_gate["control"][k] for k in ("what", "rows_flagged", "status")}}
    ship = b.load("results/ship_swift_parity.json")
    kinds = {}
    for kind in ("aot", "jit"):
        doc = b.load(f"results/ship_swift_parity_{kind}.json")
        n_bit = sum(bool(r["python_bit_equal"]) for r in doc["rows"])
        kinds[kind] = {"text_rows": len(doc["rows"]), "python_bit_equal": n_bit,
                       "file_python_bit_equal_rows": doc["summary"]["python_bit_equal_rows"],
                       "recomputed_equal": n_bit == doc["summary"]["python_bit_equal_rows"],
                       "drift_max": max(r["drift_marker_logits"] for r in doc["rows"]),
                       "by_bucket": {str(L): sum(r["bucket"] == L for r in doc["rows"]) for L in host.BUCKETS}}
    text = {"what": "the 470 text-path rows of the round-2 fixture (459 text + 11 media rows with the oracle's prefix) "
                    "through the stripped decision graphs (buckets 256-4096): AOT h16c and the .aimodel specialized by "
                    "the Swift runtime, against the Python runtime on the same AOT (bit for bit) and the oracle",
            "per_kind": kinds, "facts_bar": {k: ship["per_kind"][k]["text"]["facts_bar"] for k in ("aot", "jit")},
            "argmax_equal": ship["per_kind"]["aot"]["text"]["argmax_equal"],
            "max_abs_dp": ship["per_kind"]["aot"]["text"]["max_abs_dp"],
            "mean_row_max_abs_dp": ship["per_kind"]["aot"]["text"]["mean_row_max_abs_dp"],
            "wrong_pairing_control": ship["per_kind"]["aot"]["text"]["wrong_pairing_control"],
            "jit_vs_aot": ship["text_jit_vs_aot"]}
    media = {}
    for mode, rel in (("image", "results/ship_swift_media_image.json"), ("audio", "results/ship_swift_media_audio.json")):
        doc = b.load(rel)
        sw = doc["e2e"]["swift"]
        media[mode] = {"source": rel, "status": doc["status"], "rows": sw["rows"], "argmax_equal": sw["argmax_equal"],
                       "max_abs_dp": sw["max_abs_dp"], "mean_row_max_abs_dp": sw["mean_row_max_abs_dp"],
                       "rows_bit_equal_python": doc["e2e"].get("rows_bit_equal_python",
                                                               doc["e2e"].get("rows_bit_equal_python_numpy_mel")),
                       "control_next_item_prefix": doc["e2e"]["control_next_item_prefix"],
                       "wrong_pairing_oracle_swap": doc["e2e"]["wrong_pairing_oracle_swap"], "jit_vs_aot": doc["jit_vs_aot"]}
        for key in ("decode", "graph_inputs", "prefix", "samples", "mel", "masks"):
            if key in doc:
                v = doc[key]
                media[mode][key] = {k: x for k, x in v.items() if not isinstance(x, (list, dict))} if isinstance(v, dict) else v
    small = b.load("results/swift_parity_L64_128.json")
    sk = {}
    for kind in ("aot", "jit"):
        doc = b.load(f"results/small_swift_parity_{kind}.json")
        n_bit = sum(bool(r["python_bit_equal"]) for r in doc["rows"])
        sk[kind] = {"rows": len(doc["rows"]), "python_bit_equal": n_bit,
                    "by_bucket": {str(L): sum(r["bucket"] == L for r in doc["rows"]) for L in SMALL}}
    small_out = {"what": "the 342 text rows whose bucket under host.ALL_BUCKETS is 64 or 128, through the stripped L64 / "
                         "L128 graphs: AOT and the .aimodel, against the Python runtime (bit for bit) and the oracle. The 9 "
                         "audio rows of 128 positions or fewer were not run through the Swift host at L64 / L128",
                 "status": small["status"], "per_kind": sk,
                 "argmax_equal": small["aot"]["argmax_equal"], "max_abs_dp": small["aot"]["max_abs_dp"],
                 "mean_row_max_abs_dp": small["aot"]["mean_row_max_abs_dp"],
                 "wrong_pairing_control": small["aot"]["wrong_pairing_control"]["status"], "jit_vs_aot": small["jit_vs_aot"]}
    probe = b.load("results/ship_swift_l1024_probe.json")
    l1024 = {"what": "no fixture row has 513-1024 positions: one request (a public long record's state cut to its first 34 "
                     "lines, 3 rows of 961-1009 positions) through `d1omni ask` against the Python runtime's response",
             "status": probe["status"], "response_equal_python_bytes": {k: v["response_equal_python_bytes"]
                                                                        for k, v in probe["swift"].items()}}
    ask = b.load("results/swift_media_ask.json")
    checks = [ship["status"], kinds["aot"]["recomputed_equal"], kinds["jit"]["recomputed_equal"],
              kinds["aot"]["python_bit_equal"] == kinds["aot"]["text_rows"], kinds["jit"]["python_bit_equal"] == kinds["jit"]["text_rows"],
              media["image"]["status"], media["audio"]["status"], small["status"],
              sk["aot"]["python_bit_equal"] == sk["aot"]["rows"], sk["jit"]["python_bit_equal"] == sk["jit"]["rows"],
              probe["status"], rows_gate["acceptance"]["status_all"]]
    return {"what": "The Swift host (apps/D1Omni: library D1Omni, CLI d1omni; Release; the system CoreAI framework, GPU "
                    "preferred) on the Mac against the Python host and runtime and the oracle.", "lane": LANE,
            "sources": b.sources(), "bar": SHIP_BAR, "status": status_of(*checks),
            "device": "Apple M4 Max, macOS 27.0 (26A428), Xcode 27.0 RC, Apple Swift 6.4, swift-transformers 1.3.3",
            "token_rows": rows, "text": text, "media": media, "small_buckets": small_out, "l1024_probe": l1024,
            "ask": {"status": ask.get("status"), "what": "README's `d1omni ask --image / --audio` and three failure paths "
                                                         "(a WAV as an image, a FLAC as audio, a missing file: exit 1 and a "
                                                         "message)"}}


# =========================================================================== strip
def t_strip() -> dict:
    b = Book()
    bundles = {}
    for rel in ("bundles/d1-omni-600m/macos-ship/manifest.json", "bundles/d1-omni-600m/macos-ship-small/manifest.json"):
        for name, e in b.load(rel)["bundles"].items():
            bundles[name] = {k: e[k] for k in ("status", "bundle", "bytes_before", "bytes_after", "main_hash_before",
                                               "main_hash_after", "ops", "ops_equal", "inspect_equal",
                                               "local_paths_before", "local_paths_after")}
    aot = {}
    for target in ("ship-h16c", "ship-h19p"):
        for m in ("manifest.json", "manifest.small.json"):
            doc = b.load(f"compiled/{target}/{m}")
            for name, e in doc["bundles"].items():
                aot.setdefault(target, {})[name] = {"bytes": e["bytes"], "resources_bin_bytes": e["resources_bin_bytes"],
                                                    "main_hash": e["hashes"].get("main.hash"), "ane_regions": e["ane_regions"],
                                                    "status": e["status"],
                                                    "files_with_local_paths": e["files_with_local_paths"]}
    b.note("logs/r13_staging_grep.txt")
    grep = (WORK / "logs" / "r13_staging_grep.txt").read_text().splitlines()
    # the scan's needle is the macOS users-folder prefix: written out it would trip this transcript's own path scan
    scans = [line.replace("/Users/", "<users-folder prefix>") for line in grep if line.startswith(("grep ", "strings "))]
    checks = [all(e["status"] == "PASS" and e["ops_equal"] and e["inspect_equal"] and e["local_paths_after"] == 0
                  for e in bundles.values()),
              all(e["status"] == "COMPILED" and e["files_with_local_paths"] == 0 for t in aot.values() for e in t.values()),
              all(line.rstrip().endswith((": 0 files", ": 0 lines")) for line in scans) and len(scans) == 5]
    return {"what": "Every shipped bundle with its MLIR debug locations removed (coreai-torch strip_debug_info, "
                    "strip_ship.py): the op count and `coreai-build inspect` summary before and after, the bytes and "
                    "main.hash before and after, the byte scan for the exporting machine's paths; the compiles of the "
                    "stripped bundles (h16c for the Mac, h19p for the iPhone 18 Pro, never loaded on a Mac); the scan of "
                    "the Hugging Face staging (every file, text and binary). The gates on the stripped bundles equal the "
                    "unstripped ones (runtime-decide / -vision / -audio: stripped_vs_unstripped).", "lane": LANE,
            "sources": b.sources(), "status": status_of(*checks), "bundles": bundles, "aot": aot,
            "staging_scan": {"file": "logs/r13_staging_grep.txt", "lines": scans}}


# =========================================================================== timing
def window(doc: dict) -> dict:
    lock = doc.get("lock_at_start") or {}
    return {"label": doc.get("label") or doc.get("expected_lock_label"), "started": doc.get("started"),
            "finished": doc.get("finished"), "contended": doc.get("contended"),
            "lock_was_ours": doc.get("lock_is_ours"), "lock_at_start": lock.get("content") if isinstance(lock, dict) else lock}


def t_timing_mac() -> dict:
    b = Book()
    out = {"what": "Time per decision on the Mac GPU (Apple M4 Max, macOS 27.0 26A428) in machine-wide measurement "
                   "windows (quiet_hold.py: no other GPU job in the window). One decision = the graph inputs, the graph "
                   "call(s) (media graph and decision graph), the output copy, the host's marker read, temperature and "
                   "softmax; tokenizing, decoding and the mel are apart. 5 warm-up rounds, then 30 decisions per form, forms "
                   "interleaved. Every median, p10 and p90 here is recomputed from the per-decision samples (numpy).",
           "lane": LANE, "windows": {}, "python_runtime": {}, "swift_host": {}}
    runs = {"run1": "results/timing/run1_main.json", "run2": "results/timing/run2_main.json"}
    win = {"run1": "results/timing/run1_window.json", "run2": "results/timing/run2_window.json"}
    for r, rel in win.items():
        out["windows"][r] = window(b.load(rel))
    rank = b.load("results/timing/ranking.json")
    mains = {r: b.load(rel) for r, rel in runs.items()}
    eq = []
    py = {}
    for w, wl in rank["workloads"].items():
        rows = []
        for row in wl["rows"]:
            per = {}
            for r in ("run1", "run2"):
                samples = mains[r]["workloads"][w]["forms"][row["form"]]["samples_ms"]
                per[r] = stats(samples)
            score = (per["run1"]["median"] + per["run2"]["median"]) / 2
            same = abs(score - row["score_ms"]) <= 1e-9 * max(1.0, score) and all(
                per[r]["median"] == row[r]["median"] for r in ("run1", "run2"))
            eq.append(same)
            rows.append({"form": row["form"], "score_ms": score, "run1": per["run1"], "run2": per["run2"],
                         "parity": row["parity"]["status"], "ship_candidate": row["ship_candidate"],
                         "recomputed_equal": same})
        py[w] = {"case": wl["case"], "forms": rows}
    out["python_runtime"]["round7_two_windows"] = {"rule": "results/timing/ranking_rule.md (+ addendum 1): score = the mean "
                                                           "of the two windows' medians", "workloads": py}
    b.note("results/timing/ranking_rule.md")
    # round 12: one window, the stripped buckets incl. L64 / L128
    r12 = b.load("results/timing/ranking_r12.json")
    m12 = b.load("results/timing/run5_r12_main.json")
    out["windows"]["run5_r12"] = window(b.load("results/timing/run5_r12_window.json"))
    py12 = {}
    for w, wl in r12["workloads"].items():
        rows = []
        for row in wl["rows"]:
            st = stats(m12["workloads"][w]["forms"][row["form"]]["samples_ms"])
            same = st["median"] == row["score_ms"] and st["p10"] == row["stats_ms"]["p10"] and st["p90"] == row["stats_ms"]["p90"]
            eq.append(same)
            rows.append({"form": row["form"], "score_ms": st["median"], "stats_ms": st, "parity": row["parity"]["status"],
                         "ship_candidate": row["ship_candidate"], "recomputed_equal": same})
        py12[w] = {"case": wl["case"], "forms": rows}
    out["python_runtime"]["round12_one_window"] = {"rule": "results/timing/ranking_rule_r12.md: score = the window's "
                                                            "median", "workloads": py12}
    b.note("results/timing/ranking_rule_r12.md")
    # Swift host: round 8 (text, two windows), round 9 (media, one window)
    r8 = b.load("results/timing/ranking_r8.json")
    swift8 = {}
    s8 = {"run1": b.load("results/timing/run3_swift_run1.json"), "run2": b.load("results/timing/run3_swift_run2.json")}
    for r in ("run1", "run2"):
        out["windows"][f"swift_{r}"] = window(b.load(f"results/timing/run3_swift_{r}_window.json"))
    for w, forms in r8["workloads"].items():
        rows = []
        for form, v in forms.items():
            if not form.endswith(("swift-aot", "swift-jit")):
                rows.append({"form": form, "score_ms": v["score"], "note": "the Python runtime on the same AOT in the "
                                                                           "same windows (control)"})
                continue
            per = {r: stats(s8[r]["workloads"][w]["forms"][form]["samples_ms"]) for r in ("run1", "run2")}
            score = (per["run1"]["median"] + per["run2"]["median"]) / 2
            same = abs(score - v["score"]) <= 1e-9 * score and all(per[r]["median"] == v[r]["median"] for r in per)
            eq.append(same)
            rows.append({"form": form, "score_ms": score, "run1": per["run1"], "run2": per["run2"],
                         "jit_aot_bit_equal": v["run1"].get("jit_aot_bit_equal"), "candidate": v["candidate"],
                         "recomputed_equal": same})
        swift8[w] = rows
    r9 = b.load("results/timing/ranking_r9.json")
    s9 = b.load("results/timing/run4_swift_media.json")
    out["windows"]["swift_media"] = window(b.load("results/timing/run4_swift_media_window.json"))
    swift9 = {}
    for w, wl in r9["workloads"].items():
        rows = []
        for form, v in wl["forms"].items():
            if form.endswith("-python"):
                rows.append({"form": form, "median_ms": v["median"], "note": "the Python runtime (control)"})
                continue
            st = stats(s9["workloads"][w]["forms"][form]["samples_ms"])
            same = st["median"] == v["median"]
            eq.append(same)
            rows.append({"form": form, "median_ms": st["median"], "stats_ms": st, "parts_median_ms": v["parts"],
                         "jit_aot_logits_bit_equal": v.get("jit_aot_logits_bit_equal"), "candidate": v["candidate"],
                         "recomputed_equal": same})
        swift9[w] = {"forms": rows, "preprocess_ms": wl["preprocess_ms"]}
    out["swift_host"] = {"round8_text_two_windows": swift8, "round9_media_one_window": swift9,
                         "loads_round8": r8["loads"], "rule_round8": "results/timing/ranking_rule_r8.md",
                         "rule_round9": "results/timing/ranking_rule_r9.md"}
    b.note("results/timing/ranking_rule_r8.md")
    b.note("results/timing/ranking_rule_r9.md")
    # the ship form per workload (the supervisor's ship decision of 2026-10-08: fp16 graphs, decision buckets 64 ... 4096)
    ship = {}
    for w, wl in py12.items():
        best = [r for r in wl["forms"] if r["ship_candidate"]]
        ship[w] = {"form": best[0]["form"], "ms": best[0]["score_ms"], "p10": best[0]["stats_ms"]["p10"],
                   "p90": best[0]["stats_ms"]["p90"], "window": "run5_r12"} if best else None
    for w in ("W3", "W4", "W5L"):
        best = [r for r in py[w]["forms"] if r["ship_candidate"]]
        f = min(best, key=lambda r: r["score_ms"])
        ship[w] = {"form": f["form"], "ms": f["score_ms"], "run1": f["run1"], "run2": f["run2"],
                   "windows": ["run1", "run2"]}
    out["ship_form_per_workload"] = ship
    out["sources"] = b.sources()
    out["status"] = status_of(*eq)
    return out


# =========================================================================== forms
def gate_brief(b: Book, rel: str) -> dict:
    """A runtime_check.py record (rows) or an e2e record (arms): status and the bar's numbers, recomputed."""
    doc = b.load(rel)
    if "arms" in doc:
        arm = next(a for a in doc["arms"] if "bundle_prefix" in a)
        mine = recompute(doc["arms"][arm])
        return {"source": rel, "status": doc["status"], "kind": "end to end, " + arm, "rows": mine["rows"],
                "argmax_equal": mine["argmax_equal"], "non_near_tie": mine["non_near_tie"],
                "max_abs_dp": mine["max_abs_dp"], "mean_row_max_abs_dp": mine["mean_row_max_abs_dp"],
                "recomputed_vs_file": agrees(mine, doc["summaries"].get(arm))}
    mine = recompute(doc["rows"])
    out = {"source": rel, "status": doc["status"], "kind": f"{doc.get('compute')} {doc.get('precision')} L{doc.get('seq_len')}",
           "rows": mine["rows"], "argmax_equal": mine["argmax_equal"], "non_near_tie": mine["non_near_tie"],
           "max_abs_dp": mine["max_abs_dp"], "mean_row_max_abs_dp": mine["mean_row_max_abs_dp"],
           "max_abs_dlogit": mine["max_abs_dlogit"], "recomputed_vs_file": agrees(mine, doc["summary"]),
           "repeat_drift_max": doc["repeat_drift"]["max_abs_scores"],
           "compiled": {"bytes": doc["compiled"]["bytes"], "resources_bin_bytes": doc["compiled"]["resources_bin_bytes"],
                        "ane_regions": doc["compiled"]["ane_regions"]}}
    return out


def t_forms() -> dict:
    b = Book()
    rank = b.load("results/timing/ranking.json")
    gates, timing = {}, {}
    for w, wl in rank["workloads"].items():
        for row in wl["rows"]:
            rel = row["parity"]["json"]
            if rel not in gates:
                gates[rel] = gate_brief(b, rel)
            timing.setdefault(row["form"], {})[w] = {"score_ms": row["score_ms"], "parity_json": rel,
                                                     "ship_candidate": row["ship_candidate"]}
    for L in (512, 1024, 2048, 4096):  # gated precisions that the round-7 windows timed under other names
        for p in ("wfp16", "fp16"):
            rel = f"results/runtime_{p}_L{L}_gpu.json"
            if rel not in gates:
                gates[rel] = gate_brief(b, rel)
    export = {}
    for folder in sorted((WORK / "bundles" / "d1-omni-600m" / "macos").iterdir()):
        mf = folder / "provenance" / "export-manifest.json"
        if not mf.exists():
            continue
        m = b.load(str(mf.relative_to(WORK)))
        export[folder.name] = {"status": m.get("status"), "bytes": m.get("bundle_bytes") or m.get("bytes"),
                               "ops": m.get("ops") or m.get("op_count"), "measure_only": m.get("measure_only", False)}
    ab = b.load("results/absmax.json")
    torch_fp16 = {p: {k: ab[f"eager_{p}"][k] for k in ("rows", "argmax_equal", "argmax_equal_non_near_tie", "near_ties",
                                                        "max_abs_dp", "mean_of_row_max_abs_dp", "max_abs_dlogit")}
                  for p in ("fp16", "wfp16")}
    probes = {}
    for name in ("vision_fp16", "audio_fp16_10s"):
        doc = b.load(f"results/ane_probe_{name}.json")
        probes[name] = {"regions": doc["verdict"]["regions"], "parity": doc["verdict"]["parity"],
                        "by_decision": doc["verdict"].get("by_decision")}
    for p in ("fp16", "wfp16"):
        doc = b.load(f"results/aot_{p}_L256_ane.json")
        probes[f"decision_{p}_L256"] = {"regions": doc["ane"]["regions"], "ir_bytes": doc["ane"]["ir_bytes"]}
    checks = [g["recomputed_vs_file"]["equal"] for g in gates.values()]
    ship_gates = [g for rel, g in gates.items() if rel in ("results/runtime_fp16_L256_gpu.json",
                                                            "results/runtime_fp16_L4096_gpu.json",
                                                            "results/vision_e2e_fp16.run4.json",
                                                            "results/audio_e2e_fp16.run4.json")]
    checks += [g["status"] for g in ship_gates] + [len(ship_gates) == 4]
    return {"what": "Every form measured on the way to the ship form, on the Mac (M4 Max, macOS 27.0 26A428): precision "
                    "(fp32 reference, wfp16 = fp16 weights with fp32 compute, fp16 = fp16 weights and compute, int8 = every "
                    "decision linear int8 per block of 32 in an fp16 frame) x compute unit (GPU, Neural Engine) x bucket. "
                    "Each gated form keeps its gate (recomputed from its rows) and, where it was timed, its round-7 score "
                    "(the mean of two windows' medians; timing-mac has the samples). The CPU torch gates of fp16 / wfp16 "
                    "are the eager references; the export manifests keep the CPU export gate of each bundle. Forms that "
                    "fail the bar are kept with their numbers. status: every recomputation equals its record's summary "
                    "and the ship form's gates (fp16 decision L256 / L4096, vision and audio end to end) pass; a form "
                    "whose own status is FAIL is a measured form, not a failure of this transcript.", "lane": LANE,
            "sources": b.sources(), "bar": SHIP_BAR,
            "status": status_of(*checks), "runtime_gates": gates, "timed": timing, "export_bundles": export,
            "torch_cpu_eager": torch_fp16, "neural_engine_compiles": probes,
            "ship_form": "fp16 for all three graphs, decision buckets 64 / 128 / 256 / 512 / 1024 / 2048 / 4096 on the GPU "
                         "(the supervisor's decision, 2026-10-08, from the speed ladder: the fastest form that passes)"}


# =========================================================================== iPhone (round 11)
def close(a, b, tol: float = 1e-9) -> bool:
    return a == b or (a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(b)))


def phone_rows(rows: list[dict]) -> dict:
    """The gate app's per-row records (r11_results.py: max |dp| and argmax recomputed from the probability bits against
    the oracle) -> the ship-bar aggregates and the comparison with the Mac's Swift bits."""
    n = len(rows)
    clear = [r for r in rows if not r["near_tie"]]
    ties = [r for r in rows if r["near_tie"]]
    dps = [r["max_abs_dp"] for r in rows]
    out = {"rows": n, "rows_by_mode": {m: sum(r["mode"] == m for r in rows) for m in ("text", "image", "audio")},
           "rows_by_bucket": {str(L): sum(r["bucket"] == L for r in rows) for L in sorted({r["bucket"] for r in rows})},
           "non_near_tie": len(clear), "argmax_equal_non_near_tie": sum(bool(r["argmax_equal"]) for r in clear),
           "near_tie": len(ties), "argmax_equal_near_tie": sum(bool(r["argmax_equal"]) for r in ties),
           "max_abs_dp": max(dps), "mean_row_max_abs_dp": float(np.mean(dps)),
           "ids_markers_bucket_ok": sum(bool(r["ids_equal"] and r["markers_equal"] and r["bucket_equal"]) for r in rows),
           "finite": sum(bool(r["finite"]) for r in rows),
           "mac": {"logits_bit_equal": sum(bool(r["mac_logits_bit_equal"]) for r in rows),
                   "probs_bit_equal": sum(bool(r["mac_probs_bit_equal"]) for r in rows),
                   "max_abs_dp": max(r["mac_max_abs_dp"] for r in rows),
                   "max_abs_dlogit": max(r["mac_max_abs_dlogit"] for r in rows)}}
    verdict = {"argmax_non_near_tie": out["argmax_equal_non_near_tie"] == out["non_near_tie"],
               "max_abs_dp": out["max_abs_dp"] <= SHIP_BAR["max_abs_dp"],
               "mean_row_max_abs_dp": out["mean_row_max_abs_dp"] <= SHIP_BAR["mean_row_max_abs_dp"],
               "finite": out["finite"] == n, "ids_markers_bucket": out["ids_markers_bucket_ok"] == n}
    out["ship_bar"] = {"verdict": verdict, "status": "PASS" if all(verdict.values()) else "FAIL"}
    return out


def phone_agrees(mine: dict, theirs: dict) -> dict:
    """The recomputed aggregates against the summary r11_results.py wrote."""
    diff = {}
    for k in ("rows", "non_near_tie", "argmax_equal_non_near_tie", "near_tie", "argmax_equal_near_tie", "max_abs_dp",
              "mean_row_max_abs_dp", "ids_markers_bucket_ok"):
        if not close(mine[k], theirs.get(k)):
            diff[k] = [mine[k], theirs.get(k)]
    for k in ("logits_bit_equal", "probs_bit_equal", "max_abs_dp"):
        if not close(mine["mac"][k], theirs.get("mac", {}).get(k)):
            diff[f"mac.{k}"] = [mine["mac"][k], theirs.get("mac", {}).get(k)]
    if (mine["ship_bar"]["status"] == "PASS") != bool(theirs.get("bar_pass")):
        diff["bar_pass"] = [mine["ship_bar"]["status"], theirs.get("bar_pass")]
    return {"equal": not diff, "different": diff}


def t_iphone() -> dict:
    b = Book()
    jit = b.load("results/iphone_parity_jit.json")
    aot = b.load("results/iphone_parity_aot.json")
    bench = b.load("results/iphone_bench.json")
    ane = b.load("results/iphone_ane.json")
    fixtures = json.loads(zoo_fixtures.pinned(*zoo_fixtures.FIXTURES).read_text())
    public = {r["id"] for r in fixtures["records"] if r["public"] is True}
    subset = b.load("device/fixtures/subset.json")
    sub_ids = sorted({r["id"] for r in subset["records"]})
    # parity: the shipped graphs, each .aimodel specialized on the phone (JIT) and the h19p compiles (AOT)
    parity = {}
    for kind, doc in (("jit", jit), ("aot", aot)):
        main, l64 = phone_rows(doc["rows"]), phone_rows(doc["rows_l64"])
        parity[kind] = {"what": doc["what"], "rows": main, "recomputed_vs_file": phone_agrees(main, doc["summary"]),
                        "rows_l64": l64, "recomputed_vs_file_l64": phone_agrees(l64, doc["summary_l64"]),
                        "control": {k: doc["app_verdicts"]["parity"]["control"][k]
                                    for k in ("status", "must_be", "caught", "paired_rows", "rows_over_dp_bar")}}
    same = 0  # the L64 rows are the same keys again at L64: compare each list with its own
    for part in ("rows", "rows_l64"):
        jit_bits = {r["key"]: (r["logits_bits"], r["probs_bits"]) for r in jit[part]}
        same += sum(jit_bits.get(r["key"]) == (r["logits_bits"], r["probs_bits"]) for r in aot[part])
    parity["aot_vs_jit"] = {"rows": len(aot["rows"]) + len(aot["rows_l64"]), "logits_and_p_bit_equal": same}
    long_jit = jit["app_verdicts"]["long"]
    parity["jit"]["l4096"] = {"rows": sum(r["bucket"] == 4096 for r in jit["rows"]), "pass": long_jit["pass"],
                              "control_caught": long_jit["control"]["caught"]}
    # the host's arrays against the Mac's Swift host (the same files): everything before a graph is bit-equal
    imgs, clips = jit["images"], jit["clips"]
    crops = [c for i in imgs for c in i["crops"]]
    host_arrays = {"rgb": sum(i["rgb_equal_mac"] for i in imgs), "images": len(imgs),
                   "crop_u8": sum(c["crop_u8_equal_mac"] for c in crops), "crops": len(crops),
                   "vision_inputs": sum(c[k] for c in crops for k in ("pixel_values_equal_mac", "pos_embed_equal_mac",
                                                                      "patch_mask_equal_mac", "unshuffle_index_equal_mac")),
                   "vision_inputs_total": 4 * len(crops),
                   "samples": sum(c["equal_mac"]["samples"] for c in clips), "mel": sum(c["equal_mac"]["mel"] for c in clips),
                   "masks": sum(c["equal_mac"][k] for c in clips for k in ("mask_f", "mask_f2", "mask_f4", "mask_t")),
                   "clips": len(clips)}
    graph_outputs = {"vision_output": sum(c["output_equal_mac"] for c in crops), "crops": len(crops),
                     "audio_output": sum(c["equal_mac"]["output"] for c in clips), "clips": len(clips),
                     "prefix": sum(i["prefix_equal_mac"] for i in imgs) + sum(c["equal_mac"]["prefix"] for c in clips),
                     "prefixes": len(imgs) + len(clips)}
    host_ok = (host_arrays["rgb"] == len(imgs) and host_arrays["crop_u8"] == len(crops)
               and host_arrays["vision_inputs"] == 4 * len(crops) and host_arrays["samples"] == len(clips)
               and host_arrays["mel"] == len(clips) and host_arrays["masks"] == 4 * len(clips))
    # loads: cold = the first load of a graph into the app's Core AI cache (cache bytes added), wall seconds
    loads = {"jit": [ld for ld in jit["loads"]], "aot": [ld for ld in aot["loads"]]}
    for w in bench["workloads"].values():
        for ld in w["loads"]:
            loads[ld["kind"]].append(ld)
    load_s = {}
    for kind, items in loads.items():
        cold = [ld["wall_s"] for ld in items if ld["cache_bytes_added"] > 0]
        warm = [ld["wall_s"] for ld in items if ld["cache_bytes_added"] == 0]
        load_s[kind] = {"cold": {"loads": len(cold), "min_s": min(cold), "max_s": max(cold)},
                        "warm": {"loads": len(warm), "min_s": min(warm), "max_s": max(warm)}}
    # one decision's ms (the Mac's `d1omni time` definition), recomputed from the 20 timed samples
    ms, bench_ok = {}, []
    for w, x in bench["workloads"].items():
        forms = {}
        for form, f in x["forms"].items():
            s = stats(f["samples_ms"])
            agree = close(s["median"], f["median_ms"]) and close(s["p10"], f["p10_ms"]) and close(s["p90"], f["p90_ms"])
            forms[form] = {"n": s["n"], "median_ms": s["median"], "p10_ms": s["p10"], "p90_ms": s["p90"],
                           "warmup": len(f["warmup_ms"]), "parts_median_ms": f["parts_median_ms"],
                           "thermal_of_timed": f["thermal_of_timed"], "recomputed_vs_file": agree}
            bench_ok.append(agree and s["n"] == bench["rules"]["timed"])
        bench_ok.append(x["output_check"]["status"] == "PASS")
        ms[w] = {"what": x["what"], "record": x["record"], "mode": x["mode"], "positions": x["positions"],
                 "prefix_rows": x["prefix_rows"], "buckets": x["buckets"], "interleaved": x["interleave"],
                 "forms": forms, "output_check": x["output_check"],
                 "series_s": [s["duration_s"] for s in x["series"]],
                 "thermal_every_decision": sorted({d["thermal"] for d in x["decisions"]}),
                 "battery": {"states": x["battery_states"], "levels": x["battery_levels"]}}
    # the L4096 compile: loaded, then killed at its first call (the app's 100 ms memory monitor and the JetsamEvent)
    mem_rel, jet_rel = "device/runs/r11-210603/memory.tsv", "device/runs/r11-210603/crash/JetsamEvent-2026-10-08-210652.ips"
    mem_lines = b.note(mem_rel).read_text().strip().splitlines()
    head = mem_lines[0].split("\t")
    last = dict(zip(head, mem_lines[-1].split("\t")))
    before = dict(zip(head, mem_lines[-2].split("\t")))
    jet_text = b.note(jet_rel).read_text()
    jet = json.loads(jet_text.split("\n", 1)[1])
    proc = next(p for p in jet["processes"] if p.get("name") == "D1OmniGate")
    aot_l4096 = next(ld for ld in aot["loads"] if ld["what"] == "decide L4096")
    l4096_aot = {"what": "the h19p compile of decide-fp16-L4096: it loaded, and its first call was killed (jetsam); the "
                         "rest of the phone's runs went on in a second launch. Not shipped (the HF repository's ios-h19p/ "
                         "leaves it out; a phone runs the .aimodel, which passed above)",
                 "load_wall_s": aot_l4096["wall_s"],
                 "last_samples": {"before": {k: before[k] for k in ("t_app_s", "label", "footprint_mb", "available_mb")},
                                  "last": {k: last[k] for k in ("t_app_s", "label", "footprint_mb", "available_mb")}},
                 "jetsam": {"largest_process": jet.get("largestProcess"), "reason": proc["reason"],
                            "rpages": proc["rpages"], "page_size": jet["memoryStatus"]["pageSize"],
                            "os": jet["build"]},
                 "jit_l4096_peak_footprint_mb": next(g["memory"]["peak_footprint_mb"] for g in jit["groups"]
                                                     if g["group"] == "L4096")}
    # the Neural Engine: the Neural Engine compiles fail to load; the .aimodel with the Neural Engine preferred loads
    au, de = ane["parts"]["audio"], ane["parts"]["decision"]
    neural_engine = {
        "what": "not shipped: the h19p compiles made with --preferred-compute neural-engine (audio 10 s, decision L256) "
                "and the .aimodel loaded with the Neural Engine preferred; placement not confirmed (no ANE region "
                "entry in the app's cache)",
        "options": ane["options"],
        "h19p_neural_engine_compiles": {"audio": au["failed_attempts_other_launches"][0]["load"]["error"],
                                        "decision": de["failed_attempts_other_launches"][0]["load"]["error"],
                                        "error_type": au["failed_attempts_other_launches"][0]["load"]["error_detail"]["reflecting"]},
        "aimodel_neural_engine_preferred": {
            "audio": {"load_wall_s": au["load"]["wall_s"], "ane_region_entries": au["ane_region_entries"],
                      "rows": phone_rows(au["rows"]), "vs_gpu_jit": au["rows_vs_gpu_jit"],
                      "W5_ms": {k: {kk: v[kk] for kk in ("n", "median_ms", "p10_ms", "p90_ms", "parts_median_ms")}
                                for k, v in au["w5"]["forms"].items()}},
            "decision_l256": {"load_wall_s": de["load"]["wall_s"], "ane_region_entries": de["ane_region_entries"],
                              "rows": phone_rows(de["rows"]), "vs_gpu_jit": de["rows_vs_gpu_jit"],
                              "drift_marker_logits_max": de["max_drift_marker_logits"],
                              "ms": {w: {k: {kk: v[kk] for kk in ("n", "median_ms", "p10_ms", "p90_ms")}
                                         for k, v in x["forms"].items()} for w, x in de["bench"].items()}}},
        "note": "each pair (Neural Engine preferred / GPU) ran alternating in one series; the GPU values of these series "
                "are higher than the bench's (W1 GPU 23.85 ms here, 20.12 ms in the bench at L256)"}
    # the conditions of every timed decision and every parity row: thermal state, series length, warm-up and timed counts
    ane_series = [au["w5"]] + list(de["bench"].values())
    conditions = {
        "thermal": {"bench_decisions": sorted({d["thermal"] for x in bench["workloads"].values() for d in x["decisions"]}),
                    "parity_rows": sorted({r["thermal"] for doc in (jit, aot) for r in doc["rows"] + doc["rows_l64"]}),
                    "neural_engine": sorted({r["thermal"] for r in au["rows"] + de["rows"]}
                                            | {d["thermal"] for x in ane_series for d in x["decisions"]})},
        "series_max_s": max(s["duration_s"] for x in [*bench["workloads"].values(), *ane_series] for s in x["series"]),
        "timed_per_form": sorted({f["n"] for x in [*bench["workloads"].values(), *ane_series] for f in x["forms"].values()}),
        "warmup_per_form": sorted({len(f["warmup_ms"]) for x in [*bench["workloads"].values(), *ane_series]
                                   for f in x["forms"].values()}),
        "power": sorted({f"{s} at {lv:.2f}" for x in [*bench["workloads"].values(), *ane_series]
                         for s in x["battery_states"] for lv in x["battery_levels"]})}
    # the phone's hold and the transfer
    hold_text = b.note("device/hold.log").read_text()
    taken = re.findall(r"^\[([0-9-]+ [0-9:]+)\] hold taken: ", hold_text, re.M)
    released = re.findall(r"^\[([0-9-]+ [0-9:]+)\] hold released \(held (\d+) s\)", hold_text, re.M)
    install = b.load("device/install_20261008-210148.json")
    device = {k: bench["device"][k] for k in ("machine", "hw_model", "os", "os_build", "coreai_architecture",
                                              "physical_memory_gb", "battery_level", "battery_state", "power_source",
                                              "low_power_mode", "available_mb", "free_gb", "thermal")}
    app = {"host": "apps/D1Omni built for iOS 27 inside the gate app apps/D1OmniGate",
           "launches": [{k: r[k] for k in ("run_id", "status", "started", "finished")}
                        | {"build": b.load(f"device/runs/{r['run_id']}/result.json")["build"]["configuration"],
                           "stages": r["stage_order"]} for r in bench["runs"]],
           "note": "the first launch was killed (jetsam) at the first call of the L4096 compile, so its result file "
                   "stays 'running'; the second launch ran the stages left, the third the W3 bench and the Neural "
                   "Engine runs"}
    checks = [parity["jit"]["rows"]["ship_bar"]["status"], parity["jit"]["rows_l64"]["ship_bar"]["status"],
              parity["aot"]["rows"]["ship_bar"]["status"], parity["aot"]["rows_l64"]["ship_bar"]["status"],
              parity["jit"]["recomputed_vs_file"]["equal"], parity["jit"]["recomputed_vs_file_l64"]["equal"],
              parity["aot"]["recomputed_vs_file"]["equal"], parity["aot"]["recomputed_vs_file_l64"]["equal"],
              parity["jit"]["control"]["caught"], parity["aot"]["control"]["caught"],
              parity["aot_vs_jit"]["logits_and_p_bit_equal"] == parity["aot_vs_jit"]["rows"],
              parity["jit"]["l4096"]["pass"] is True, parity["jit"]["l4096"]["control_caught"] is True,
              host_ok, *bench_ok]
    return {"what": "The shipped fp16 graphs on the iPhone 18 Pro through the Swift host (apps/D1Omni) in the gate app "
                    "apps/D1OmniGate (round 11, 2026-10-08): parity on a subset of the fixture with each .aimodel "
                    "specialized on the phone (JIT) and with the h19p compiles (AOT), one decision's ms, the loads, "
                    "the L4096 compile's memory limit and the Neural Engine attempts. The per-row values are in the "
                    "sources; the numbers here are recomputed from them and checked against each file's own summary.",
            "lane": LANE, "sources": b.sources(), "bar": SHIP_BAR, "status": status_of(*checks), "device": device,
            "app": app,
            "subset": {"records": len(sub_ids), "records_not_published": sum(i not in public for i in sub_ids),
                       "rows": parity["jit"]["rows"]["rows"], "rows_l64": parity["jit"]["rows_l64"]["rows"],
                       "what": "48 records of the fixture (text, image and audio); the rows of the records the HF "
                               "repository does not publish are counted, not named"},
            "conditions": conditions,
            "parity": parity, "host_arrays_equal_mac": host_arrays, "graph_outputs_equal_mac": graph_outputs,
            "load_s": load_s, "bench": {"rules": bench["rules"], "what": bench["what"], "workloads": ms},
            "l4096_h19p": l4096_aot, "neural_engine": neural_engine,
            "hold": {"taken": taken, "released": [{"at": a, "held_s": int(s)} for a, s in released],
                     "held_s_total": sum(int(s) for _, s in released), "timezone": "JST"},
            "transfer": {k: install[k] for k in ("staged_files", "staged_bytes", "push_s", "push_mb_per_s", "check",
                                                 "repushed_files", "bad_after_checks")}}


BUILDERS = {"eager": t_eager, "runtime-decide": t_runtime_decide, "runtime-vision": t_runtime_vision,
            "runtime-audio": t_runtime_audio, "swift": t_swift, "strip": t_strip, "timing-mac": t_timing_mac,
            "forms": t_forms, "iphone": t_iphone}


# =========================================================================== assemble, scrub, check
def scrubber() -> stage_hf.Scrub:
    """stage_hf's rule: the ids of the records the fixture does not publish, and the per-source aggregates of a source
    with no public record (and the card's measured-only image and clip)."""
    fixtures = json.loads(zoo_fixtures.pinned(*zoo_fixtures.FIXTURES).read_text())
    unpublished = {r["id"] for r in fixtures["records"] if r["public"] is not True}
    public_sources = {r["source"] for r in fixtures["records"] if r["public"] is True}
    groups = ({r["source"] for r in fixtures["records"]} - public_sources) | {"card/image", "card/audio"}
    return stage_hf.Scrub(unpublished, groups)


def build_all() -> dict[str, dict]:
    scrub = scrubber()
    out = {}
    for name, fn in BUILDERS.items():
        doc = fn()
        doc = {"schema": "d1-omni-gate-transcript/1", "transcript": name, "model": {"hf_id": MODEL_ID, "revision": MODEL_SHA}}\
            | doc | {"written_by": "conversion/d1_omni/zoo_transcripts.py"}
        clean = scrub.json(doc)
        out[name] = clean
    return out


def text_of(doc: dict) -> str:
    return json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def path_of(name: str) -> Path:
    return OUT_DIR / f"gate-d1-omni-600m-{name}.json"


def scans(texts: dict[str, str]) -> dict:
    """The published files must not hold a local path, the user name, an unpublished id or the
    measured-only material; round 14 adds the measured-only speech corpus's name and the tilde-slash home shorthand
    (stage_hf's --final needles)."""
    scrub = scrubber()
    home, user = str(Path.home()), Path.home().name
    out = {}
    for name, t in texts.items():
        hits = []
        if "/Users/" in t or home in t:
            hits.append("local path")
        if user in t:
            hits.append("user name")
        if stage_hf.MEASURED_ONLY_CORPUS.search(t.encode()):
            hits.append("the measured-only speech corpus")
        if stage_hf.TILDE in t:
            hits.append("home shorthand")
        if scrub.bad(t):
            hits.append("unpublished id or measured-only material")
        out[name] = hits
    return out


def control_transcript(names_to_docs: dict[str, dict]) -> dict:
    """A copy of one source (the L256 runtime gate) with one row's max_abs_dp raised past the bar and its argmax flipped:
    the recomputation must disagree with that file's own summary and the bar must fail."""
    rel = "results/ship_runtime_decide_fp16_L256_gpu.json"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / rel).parent.mkdir(parents=True)
        doc = json.loads((WORK / rel).read_text())
        bad = copy.deepcopy(doc)
        row = next(r for r in bad["rows"] if not r["near_tie"])
        row["max_abs_dp"] = 0.5
        row["argmax_equal"] = False
        (root / rel).write_text(json.dumps(bad))
        g = bucket_gate(Book(root), rel)
    caught = (not g["recomputed_vs_file"]["equal"]) and g["ship_bar"]["status"] == "FAIL"
    return {"what": "one row of a copy of " + rel + " moved past the bar (max |dp| 0.5, argmax flipped)",
            "recomputed_vs_file_equal": g["recomputed_vs_file"]["equal"], "ship_bar": g["ship_bar"]["status"],
            "status": "FAIL (as it must)" if caught else "PASS (the check cannot fail: broken)", "caught": caught}


def control_fixtures() -> dict:
    """A copy of the staging's reference/ with one record, one oracle question and one clip changed: the fixtures check
    must FAIL and name all three."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        shutil.copytree(zoo_fixtures.STAGING / "reference", root / "reference")
        r = json.loads((root / "reference" / "records.json").read_text())
        r["records"][5]["note"] = (r["records"][5].get("note") or "") + " x"
        (root / "reference" / "records.json").write_text(json.dumps(r))
        o = json.loads((root / "reference" / "oracle.json").read_text())
        o["records"][3]["questions"][0]["probs"][0] += 1e-7
        (root / "reference" / "oracle.json").write_text(json.dumps(o))
        p = root / "reference" / "audio" / "aud_02.wav"
        data = bytearray(p.read_bytes())
        data[-1] ^= 1
        p.write_bytes(bytes(data))
        res = zoo_fixtures.check(zoo_fixtures.build(), root)
    named = [res["records"]["different_or_missing"], res["rows"]["different_or_missing"], res["media"]["different"]]
    caught = res["status"] == "FAIL" and all(len(x) == 1 for x in named)
    return {"what": "a copy of the staging's reference/ with one record's note, one oracle probability (+1e-7) and one "
                    "byte of one clip changed", "status": res["status"], "named": named, "caught": caught}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true", help="replace transcripts that differ from the build")
    args = parser.parse_args()
    docs = build_all()
    texts = {n: text_of(d) for n, d in docs.items()}
    if not args.check:
        hits = {n: h for n, h in scans(texts).items() if h}
        if hits:
            raise SystemExit(f"a transcript would hold: {hits}")
        for n, t in texts.items():
            p = path_of(n)
            if p.exists() and p.read_text() != t and not args.force:
                raise SystemExit(f"{p} differs from the build: pass --force to replace it")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for n, t in texts.items():
            path_of(n).write_text(t)
        print(json.dumps({n: {"bytes": len(t.encode()), "sha256": hashlib.sha256(t.encode()).hexdigest(),
                              "status": docs[n]["status"]} for n, t in texts.items()}, indent=1))
        return 0
    # --check
    files = {}
    for n, t in texts.items():
        p = path_of(n)
        on_disk = p.read_text() if p.exists() else None
        sources_ok = all((WORK / s["file"]).exists() and sha256_file(WORK / s["file"]) == s["sha256"]
                         for s in docs[n]["sources"])
        files[n] = {"file": str(p.relative_to(repo_root())), "exists": on_disk is not None,
                    "equals_build": on_disk == t, "sha256": hashlib.sha256(t.encode()).hexdigest(),
                    "status": docs[n]["status"], "sources": len(docs[n]["sources"]), "sources_sha256_current": sources_ok}
    fixtures = zoo_fixtures.check(zoo_fixtures.build())
    scan = scans(texts | {"fixtures": zoo_fixtures.OUT.read_text()}
                 | {f: (OUT_DIR / f).read_text() for f in ("README.md", "recipe.toml")}
                 | {"knowledge": (repo_root() / "knowledge" / "d1-omni-port.md").read_text()})
    controls = {"transcript_recompute": control_transcript(docs), "fixtures_check": control_fixtures()}
    ok = (all(f["exists"] and f["equals_build"] and f["status"] == "PASS" and f["sources_sha256_current"]
              for f in files.values()) and fixtures["status"] == "PASS" and not any(scan.values())
          and all(c["caught"] for c in controls.values()))
    result = {"schema": "d1-omni-zoo-check/1",
              "what": "round 13 / 14: the zoo files of models/d1-omni-600m/ against their sources: each gate transcript (the "
                      "iPhone's since round 14) equals a "
                      "fresh build from the lane's records (its numbers recomputed from the per-row values and equal to "
                      "each record's own summary), its sources' sha256 are current, its gate passes; the fixture file "
                      "equals its build and the HF staging's reference/; no file holds a local path, the user name, an "
                      "unpublished id or the measured-only material; each check fails on a mutated copy",
              "status": "PASS" if ok else "FAIL", "transcripts": files, "fixtures": fixtures, "scans": scan,
              "controls": controls,
              "scanned": "the 9 transcripts, the fixture file, README.md (the card), recipe.toml and knowledge/d1-omni-port.md"}
    CHECK_OUT.write_text(json.dumps(scrubber().paths(result), indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "transcripts": {n: [f["equals_build"], f["status"],
                                                                      f["sources_sha256_current"]] for n, f in files.items()},
                      "fixtures": fixtures["status"], "scans": scan,
                      "controls": {k: v["status"] for k, v in controls.items()}}, indent=1, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
