#!/usr/bin/env python3
"""Swift host gate: the `clef-flash` CLI (apps/ClefFlash) against the author's fp32 oracle and the Python reference of
the same Core AI assets.

The Swift side runs the whole decision itself — the request's JSON parsed with its literals kept and rendered the way
Python's json.dumps renders it, the prompt and its spans with swift-transformers' tokenizer, ImageIO decode + Pillow's
integer bicubic + the patches, the tower, the decoder's chunk order, the head arrays and the bucket head, the float32
softmax, the response — and writes one JSON per fixture pass (`clef-flash fixture`). This script prepares the image
rows the CLI cannot make (native grids), runs the model-free checks, and scores the passes.

  prep     the oracle's fp32 image rows as raw float32 files + manifest.json, for the runs fed from files:
           native (no tower graph exists at a native grid; readout_gate.py fed the decoder the same rows) and, for
           the decoder-and-head cross-check, every g256 / g448 run  -> L/swift/embeds/<set>/
  render   the canonical JSON renderer (CLI render-test) vs Python: every JSON value of the fixture and held-out
           requests and every option rendering, edge cases (float literals, escapes, key order by code point, raw
           literals json.dumps never writes, duplicate keys) and 200,000 doubles (repr and round(x, 4))
  pixels   the CLI's `preprocess` tiles and patches vs host.preprocess (Pillow) and the oracle's pixel_values
  score    one asset's fixture pass + held-out pass (+ the oracle-rows pass) -> transcript:
           G1 ids and spans = the oracle's on every run (253: fixture 213 = text 172 + g256 14 + g448 14 + native 13,
              held-out 40), rope start / amount, the response's model / usage
           G2 the bar (readout_gate.BAR, unchanged): argmax per question = the oracle's on every question whose oracle
              top-2 margin is above 0.02 (near-ties listed apart), max |dp| <= 0.02 over every option, mean over runs
              of the run's mean |dp| <= 0.002, all runs, finite, the pass's reset re-run bit-equal
           G3 Swift vs Python, same assets: hidden sha256 vs the Python decoder's rows (readout shards; text and
              native runs have identical inputs), logits vs the head graph on the Python decoder's rows (head_h2_*),
              p vs the author's fp32 head on them (readout_*), the softmax and the response recomputed in Python from
              the Swift logits (decide.response), the response vs decide.py's (24 requests, same tower path); the
              oracle-rows pass isolates the decoder + head on the g256 / g448 runs
  negative the negative control: two `clef-flash ask --trace` files, the base request and the same request with one word
           of one question changed -> ids / spans vs the oracle (the changed one must go red) and how far p moves
  towers   the Swift tower outputs a gate pass dumped vs the Python runtime's output of the same tower AOT asset on
           host.preprocess's patches (the tower path decide.py runs)
  timing   the timed passes (_time_mac.sh) -> L/results/swift_timing.json

Run from the worktree root (shared venv, offline):
    HF_HOME=~/code/coreai/_clefflash/hf HF_HUB_OFFLINE=1 ../coreai-models/.venv/bin/python \\
        conversion/clef_flash/gate_swift.py score --asset fp16_aot --fixture .../fixture.json --heldout .../heldout.json \\
        [--oracle-rows .../oracle_rows.json] --transcript ~/code/coreai/_clefflash/results/swift_gate_fp16_aot.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import host  # noqa: E402
from _paths import record_path, work_path  # noqa: E402

LANE = work_path("_clefflash")
BIN = LANE / "swift" / ".build" / "release" / "clef-flash"
PKG = REPO / "apps" / "ClefFlash"
BAR = {"max_abs_dp": 0.02, "mean_of_run_mean_abs_dp": 0.002, "near_tie_top2_margin": 0.02}   # readout_gate.BAR
SETS = {"fixture": {"fixtures": LANE / "fixtures", "oracle": LANE / "oracle"},
        "heldout": {"fixtures": LANE / "fixtures" / "heldout", "oracle": LANE / "oracle" / "heldout"}}
SKIP = {("photo_01", "native"): "40 x 30 = 1,200 image rows > the graph's 1,024-row buffer (as in readout_gate.py)"}
# Python references of the same assets (round 4 / 5 transcripts)
PY = {"fp16": {"fixture": {"readout": "readout_fp16_pf64_full.json", "h2": "head_h2_fixture.json"},
               "heldout": {"readout": "readout_heldout_fp16_pf64.json", "h2": "head_h2_heldout.json"}},
      "int8mix": {"fixture": {"readout": "readout_int8mix_pf64.json", "h2": None},
                  "heldout": {"readout": "readout_heldout_int8mix_pf64.json", "h2": None}}}
DECIDE = {"fixture": "decide_e2e_fixture.json", "heldout": "decide_e2e_heldout.json"}
PY_RESULTS = {"fp16_aot": ("fp16_pf64", "AOT h16c .aimodelc, SpecializationOptions.default"),
              "fp16_jit": ("fp16_pf64", "the .aimodel specialized by Swift (GPU preferred + expectFrequentReshapes)"),
              "int8mix_aot": ("int8mix_pf64", "AOT h16c .aimodelc, SpecializationOptions.default"),
              "int8mix_jit": ("int8mix_pf64", "the .aimodel specialized by Swift (GPU preferred + expectFrequentReshapes)")}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def package_record() -> dict:
    files = sorted(p for p in PKG.rglob("*") if p.is_file() and ".build" not in p.parts and p.name != "Package.resolved")
    rec = {"path": str(PKG), "files_sha256": {str(p.relative_to(PKG)): sha256_file(p) for p in files}}
    resolved = PKG / "Package.resolved"
    if resolved.exists():
        rec["resolved"] = {p["identity"]: p["state"].get("version") or p["state"].get("revision")
                           for p in json.loads(resolved.read_text()).get("pins", [])}
    if BIN.exists():
        rec["binary"] = {"path": str(BIN), "sha256": sha256_file(BIN), "bytes": BIN.stat().st_size}
    return rec


def oracle_rows(set_name: str) -> dict:
    doc = json.loads((SETS[set_name]["oracle"] / "records_oracle.json").read_text())
    assert doc["complete"], "oracle incomplete"
    return {(r["id"], r["arm"]): r for r in doc["rows"]}


def records(set_name: str) -> dict:
    doc = json.loads((SETS[set_name]["fixtures"] / "records.json").read_text())
    return {r["id"]: r for r in doc["records"]}


# --------------------------------------------------------------------------- prep
def cmd_prep(args) -> int:
    out_root = LANE / "swift" / "embeds"
    summary = {}
    for set_name in SETS:
        rows = oracle_rows(set_name)
        d = out_root / set_name
        d.mkdir(parents=True, exist_ok=True)
        man = {"what": "the oracle's fp32 tower rows (oracle npz image_embeds) as raw little-endian float32 [N, 4096], "
                       "fed to `clef-flash fixture --embeds` (native: no tower at that grid; g256 / g448: the decoder + "
                       "head alone, the readout_gate.py inputs)", "runs": {}}
        for (rid, arm), o in sorted(rows.items()):
            if arm == "text" or (rid, arm) in SKIP:
                continue
            z = np.load(SETS[set_name]["oracle"] / "npz" / f"{rid}__{arm}.npz")
            e = np.ascontiguousarray(z["image_embeds"], dtype="<f4")
            hw = [int(v) for v in o["merged_hw"]] if arm == "native" else [host.tile_grid(int(arm[1:]))] * 2
            assert e.shape == (hw[0] * hw[1], 4096), (rid, arm, e.shape, hw)
            f = d / f"{rid}__{arm}.f32"
            e.tofile(f)
            man["runs"][f"{rid}:{arm}"] = {"file": str(f), "grid": hw, "sha256": hashlib.sha256(e.tobytes()).hexdigest()}
        (d / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
        summary[set_name] = len(man["runs"])
        print(f"{set_name}: {len(man['runs'])} image-row files -> {d}")
    print(json.dumps(summary))
    return 0


# --------------------------------------------------------------------------- render
def py_render(v) -> str:
    return host.render(v)


def render_inputs() -> dict:
    """Every JSON value the prompt renders (fixture + held-out), edge cases, raw literals, doubles."""
    values, where = [], []
    for set_name in SETS:
        for rid, rec in records(set_name).items():
            req = rec["request"]
            values.append(req["state"])
            where.append(f"{set_name}/{rid}/state")
            for qid, q in req["questions"].items():
                ins = q.get("instructions")
                values.append(qid if ins is None or ins == "" else ins)
                where.append(f"{set_name}/{rid}/{qid}/instructions")
                for oid, desc in host.question_options(q):
                    sem = {"option_id": oid}
                    if desc is not None:
                        sem["description"] = desc
                    values.append(sem)
                    where.append(f"{set_name}/{rid}/{qid}/option {oid}")
    edge = [
        1250, 1250.0, 0.0, -0.0, 1e-05, 0.0001, 1e16, 1e15, 1234567890123456.0, 12345678901234567.0, 1.5e300,
        5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 0.1, 0.3, 1 / 3, 100.0, -12.4, 3.97, 8.01, 1e22,
        1e-7, 123456789, -0, 10 ** 30, -(10 ** 25), True, False, None, "", "plain", 'quote " and \\ backslash / slash',
        "tab\tnewline\ncr\rbs\bff\f", "ctrl \x00\x01\x1f\x7f end", "unicode é 日本 😀   ́e", [], {}, [[]], [{}],
        {"b": 1, "a": 2, "B": 3, "é": 4, "é": 5, "aa": 6, "a": 7, "Z": 8, "😀": 9, "ÿ": 10},
        {"nested": {"z": [1, 2.5, {"y": None, "x": True}], "a": "s"}, "list": [3.0, -1e-05, "x"]},
        {"option_id": "x", "description": {"covers": "parts", "example": "a hinge"}},
    ]
    raw = ['1.50', '1E5', '-0', '-0.0', '1e400', '-1e400', '1.0e-5', '123456789012345678901234567890', '0.1e1',
           '{"a": 1, "a": 2}', '{"b": 0, "a": {"d": 1.10, "c": [1e2, 2E-3]}}', 'NaN', 'Infinity', '-Infinity',
           '"\\u00e9\\ud83d\\ude00\\/"', '[1, 1.0, 1.00, 1e0, 10e-1]', '{"\\u0000": 1, " ": 2}']
    rng = random.Random(6)
    doubles = []
    for _ in range(100_000):                         # probabilities: float32 values in [0, 1]
        doubles.append(float(np.float32(rng.random())))
    for _ in range(50_000):                          # scores: sums of k * p in double
        doubles.append(rng.random() * rng.choice([1, 2, 3, 4, 6, 9]))
    for _ in range(50_000):                          # any double
        doubles.append(struct.unpack("<d", struct.pack("<Q", rng.getrandbits(63)))[0] * rng.choice([1, -1]))
    doubles += [0.5, 0.25, 0.00005, 0.00015, 0.00025, 0.12345, 0.99995, 2.675, 1.0000500000000001, 0.3]
    doubles = [d for d in doubles if np.isfinite(d)]
    return {"values": values + edge, "where": where + [f"edge/{i}" for i in range(len(edge))], "raw": raw,
            "doubles": doubles}


def cmd_render(args) -> int:
    work = LANE / "swift" / "checks"
    work.mkdir(parents=True, exist_ok=True)
    inp = render_inputs()
    src = work / "render_in.json"
    src.write_text(json.dumps({"values": inp["values"], "raw": inp["raw"], "doubles": inp["doubles"]}, ensure_ascii=False))
    out = work / "render_out.json"
    subprocess.run([str(BIN), "render-test", "--in", str(src), "--out", str(out)], check=True)
    got = json.loads(out.read_text())
    want_render = [py_render(v) for v in inp["values"]]
    want_canon = [json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=True) for v in inp["values"]]
    want_raw = [json.dumps(json.loads(t), ensure_ascii=False, separators=(",", ":"), sort_keys=True) for t in inp["raw"]]
    want_repr = [repr(d) for d in inp["doubles"]]
    want_round = [repr(round(d, 4)) for d in inp["doubles"]]

    def diff(name, a, b, labels=None):
        bad = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        return {"n": len(b), "equal": len(b) - len(bad) if len(a) == len(b) else 0, "count_match": len(a) == len(b),
                "first_differences": [{"i": i, "where": labels[i] if labels else None, "swift": a[i], "python": b[i]}
                                      for i in bad[:8]]}

    rep = {"schema": "clef-flash-swift-render/1", "generated_at": now(),
           "what": "PythonJSON (apps/ClefFlash Renderer.swift) through `clef-flash render-test` vs CPython "
                   f"{sys.version.split()[0]} json.dumps(ensure_ascii=False, separators=(',', ':'), sort_keys=True), "
                   "repr(float), round(x, 4)",
           "render": diff("render", got["render"], want_render, inp["where"]),
           "canonical": diff("canonical", got["canonical"], want_canon, inp["where"]),
           "raw_literals": diff("raw", got["raw"], want_raw, inp["raw"]),
           "float_repr": diff("repr", got["repr"], want_repr), "round4": diff("round4", got["round4"], want_round),
           "values": len(inp["values"]), "request_values": len(inp["where"]) - 0, "doubles": len(inp["doubles"])}
    rep["pass"] = all(rep[k]["equal"] == rep[k]["n"] for k in ("render", "canonical", "raw_literals", "float_repr", "round4"))
    path = LANE / "results" / "swift_render_check.json"
    path.write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    for k in ("render", "canonical", "raw_literals", "float_repr", "round4"):
        print(f"{k}: {rep[k]['equal']}/{rep[k]['n']}", rep[k]["first_differences"][:2] if rep[k]["first_differences"] else "")
    print(f"{'PASS' if rep['pass'] else 'FAIL'} -> {path}")
    return 0 if rep["pass"] else 1


# --------------------------------------------------------------------------- pixels
def cmd_pixels(args) -> int:
    from PIL import Image

    rep = {"schema": "clef-flash-swift-pixels/1", "generated_at": now(), "sets": {}}
    ok = True
    for set_name in SETS:
        img_dir = SETS[set_name]["fixtures"] / "images"
        out = LANE / "swift" / "pixels" / set_name
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(BIN), "preprocess", "--images", str(img_dir), "--out-dir", str(out)], check=True,
                       capture_output=True)
        tiles = json.loads((out / "preprocess.json").read_text())["tiles"]
        # the oracle's pixel_values per (image, arm)
        rows = oracle_rows(set_name)
        recs = records(set_name)
        oracle_pv = {}
        for (rid, arm), o in rows.items():
            if arm in ("g256", "g448"):
                oracle_pv[(recs[rid]["image_files"][0], arm)] = SETS[set_name]["oracle"] / "npz" / f"{rid}__{arm}.npz"
        res = []
        for t in tiles:
            tile = 256 if t["grid"] == "g256" else 448
            im = Image.open(img_dir / t["image"])
            rgb = np.asarray(im.convert("RGB"))
            pil = np.asarray(im.convert("RGB").resize((tile, tile), Image.Resampling.BICUBIC))
            got = np.fromfile(out / t["resized_file"], np.uint8).reshape(tile, tile, 3)
            d = np.abs(got.astype(np.int16) - pil.astype(np.int16))
            patches = host.preprocess(img_dir / t["image"], tile)
            r = {"image": t["image"], "grid": t["grid"], "size_in": t["size_in"], "decode_path": t["decode_path"],
                 "decoded_rgb_equal_pillow": t["decoded_rgb_sha256"] == hashlib.sha256(rgb.tobytes()).hexdigest(),
                 "tile_max_abs_level_vs_pillow": int(d.max()), "tile_values_differing": int((d > 0).sum()),
                 "patches_sha256_equal_host": t["patches_sha256"] == hashlib.sha256(patches.tobytes()).hexdigest()}
            key = (t["image"], t["grid"])
            if key in oracle_pv:
                pv = np.load(oracle_pv[key])["pixel_values"].astype(np.float32)
                r["patches_sha256_equal_oracle_pixel_values"] = (
                    t["patches_sha256"] == hashlib.sha256(np.ascontiguousarray(pv).tobytes()).hexdigest())
            res.append(r)
        s = {"tiles": len(res), "decoded_rgb_equal_pillow": sum(r["decoded_rgb_equal_pillow"] for r in res),
             "tiles_bit_equal_pillow": sum(r["tile_max_abs_level_vs_pillow"] == 0 for r in res),
             "max_abs_level": max(r["tile_max_abs_level_vs_pillow"] for r in res),
             "patches_equal_host_preprocess": sum(r["patches_sha256_equal_host"] for r in res),
             "oracle_tiles": sum("patches_sha256_equal_oracle_pixel_values" in r for r in res),
             "patches_equal_oracle_pixel_values": sum(r.get("patches_sha256_equal_oracle_pixel_values", False) for r in res)}
        ok &= (s["tiles_bit_equal_pillow"] == s["tiles"] == s["patches_equal_host_preprocess"] == s["decoded_rgb_equal_pillow"]
               and s["patches_equal_oracle_pixel_values"] == s["oracle_tiles"])
        rep["sets"][set_name] = {"summary": s, "tiles": res}
        print(set_name, json.dumps(s))
    rep["pass"] = bool(ok)
    path = LANE / "results" / "swift_pixels.json"
    path.write_text(json.dumps(rep, indent=1) + "\n")
    print(f"{'PASS' if ok else 'FAIL'} -> {path}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- score
def py_refs(scheme: str, set_name: str) -> dict:
    """(id, arm) -> {readout run, h2 run, hidden fp16 rows of the Python decoder}."""
    ref = PY[scheme][set_name]
    ro = json.loads((LANE / "results" / ref["readout"]).read_text())
    procs = {p["shard"]: p for p in ro["processes"]} if "processes" in ro else {}
    h2 = json.loads((LANE / "results" / ref["h2"]).read_text()) if ref["h2"] else None
    h2_runs = {(r["id"], r["arm"]): r for r in h2["runs"] if r.get("variant", "base") == "base"} if h2 else {}
    out = {}
    npz_cache: dict = {}
    for r in ro["runs"]:
        if r.get("variant", "base") != "base":
            continue
        npz = r.get("npz") or procs[r["shard"]]["npz"]
        out[(r["id"], r["arm"])] = {"readout": r, "h2": h2_runs.get((r["id"], r["arm"])), "npz": npz, "key": r["npz_key"]}
    return {"runs": out, "readout_file": ref["readout"], "readout_sha256": sha256_file(LANE / "results" / ref["readout"]),
            "h2_file": ref["h2"], "bundle": ro.get("bundle"), "_npz": npz_cache}


def hidden_of(refs: dict, key) -> np.ndarray:
    e = refs["runs"][key]
    cache = refs["_npz"]
    if e["npz"] not in cache:
        cache[e["npz"]] = np.load(e["npz"])          # lazy: one member is read per call
    return cache[e["npz"]][f"{e['key']}__hidden"]


def score_pass(sw: dict, set_name: str, scheme: str, kind: str, decide_ref: dict | None) -> dict:
    import decide
    from clef_head import question_probs

    rows = oracle_rows(set_name)
    recs = records(set_name)
    expected = sorted(k for k in rows if k not in SKIP)
    runs = {(r["id"], r["arm"]): r for r in sw["runs"]}
    refs = py_refs(scheme, set_name)
    dump = Path(sw["dump_dir"]) if sw.get("dump_dir") else None
    qrows, run_rows, g1_bad = [], [], []
    for key in expected:
        o = rows[key]
        r = runs.get(key)
        if r is None:
            g1_bad.append(f"{key[0]}/{key[1]}: missing")
            continue
        # G1: ids, spans, rope, response model / usage
        ids_ok = r["ids"] == o["ids"]
        spans_ok = [[q["question_span"], q["option_spans"], q["option_ids"]] for q in r["questions"]] == \
                   [[q["question_span"], q["option_spans"], q["option_ids"]] for q in o["questions"]]
        if key[1] == "text":
            rope_ok = r["rope_shift_start"] == host.NO_SHIFT and r["rope_shift_amount"] == 0
        else:
            n = len([t for t in o["ids"] if t == host.IMAGE_PAD])
            rope_ok = (r["rope_shift_start"] == o["token_offset"] + 1 + n and r["rope_shift_amount"] == o["rope_shift_amount"]
                       if "rope_shift_amount" in o else r["rope_shift_start"] == o["token_offset"] + 1 + n)
        resp = r["response"]
        meta_ok = resp["model"] == o["systemone_response"]["model"] and resp["usage"] == o["systemone_response"]["usage"]
        if not (ids_ok and spans_ok and rope_ok and meta_ok):
            g1_bad.append(f"{key[0]}/{key[1]}: ids {ids_ok} spans {spans_ok} rope {rope_ok} model/usage {meta_ok}")
        # G2: per question vs the oracle
        logits = np.asarray(r["logits"], np.float32)
        probs_sw = [np.asarray(p, np.float32) for p in r["probabilities"]]
        layout, a = [], 0
        for q in r["questions"]:
            layout.append((a, a + len(q["option_ids"])))
            a += len(q["option_ids"])
        probs_py = question_probs(logits, layout)               # NumPy's float32 softmax of the Swift logits
        softmax_equal = all(np.array_equal(x.view(np.uint32), y.view(np.uint32)) for x, y in zip(probs_sw, probs_py))
        req = recs[key[0]]["request"]
        resp_py = decide.response(req, r["questions"], probs_sw, len(r["ids"]))
        response_equal = resp_py == resp and list(resp_py["answers"]) == list(resp["answers"])
        ref = refs["runs"].get(key)
        h2q = (ref or {}).get("h2")
        deltas, qs = [], []
        for qi, (q, oq, p) in enumerate(zip(r["questions"], o["questions"], probs_sw)):
            po = np.asarray(oq["probs"], np.float64)
            pd = np.asarray(p, np.float64)
            dp = np.abs(pd - po)
            deltas.append(dp)
            item = {"id": key[0], "arm": key[1], "question_id": q["question_id"], "type": q["type"], "n_options": len(po),
                    "argmax": int(pd.argmax()), "argmax_oracle": oq["argmax_index"],
                    "argmax_equal": int(pd.argmax()) == oq["argmax_index"], "near_tie": bool(oq["near_tie"]),
                    "oracle_top2_margin": oq["top2_margin"], "max_abs_dp": float(dp.max()), "mean_abs_dp": float(dp.mean()),
                    "max_abs_dlogit": float(np.abs(logits[layout[qi][0]:layout[qi][1]].astype(np.float64)
                                                   - np.asarray(oq["logits"], np.float64)).max()),
                    "probs": [float(v) for v in p], "probs_oracle": [float(v) for v in po]}
            if ref is not None:
                pr = np.asarray(ref["readout"]["questions"][qi]["probs"], np.float64)
                item["vs_python_author_head_max_abs_dp"] = float(np.abs(pd - pr).max())
            if h2q is not None:
                ph = np.asarray(h2q["questions"][qi]["probs"], np.float64)
                item["vs_python_head_graph_max_abs_dp"] = float(np.abs(pd - ph).max())
            qs.append(item)
        flat = np.concatenate(deltas)
        rr = {"id": key[0], "arm": key[1], "source": r.get("source"), "tokens": len(r["ids"]), "calls": r["calls"],
              "bucket": r["bucket"][0], "image_rows_from": r["image_rows_from"],
              "max_abs_dp": float(flat.max()), "mean_abs_dp": float(flat.mean()),
              "finite": bool(r["hidden_finite"] and r["logits_finite"]),
              "argmax_equal_non_near_tie": all(q["argmax_equal"] for q in qs if not q["near_tie"]),
              "softmax_recomputed_bit_equal": softmax_equal, "response_recomputed_equal": response_equal,
              "ids_equal_oracle": ids_ok, "spans_equal_oracle": spans_ok}
        # G3: vs the Python decoder's rows and the head graph on them
        if ref is not None:
            hp = hidden_of(refs, key)
            rr["hidden_sha256_equal_python"] = r["hidden_sha256"] == hashlib.sha256(np.ascontiguousarray(hp).tobytes()).hexdigest()
            if dump and r.get("hidden_dump") and not rr["hidden_sha256_equal_python"]:
                hs = np.fromfile(dump / r["hidden_dump"], np.float16).reshape(-1, 4096)
                rr["hidden_max_abs_diff_python"] = float(np.abs(hs.astype(np.float32) - hp.astype(np.float32)).max())
            rr["vs_python_author_head_max_abs_dp"] = max(q["vs_python_author_head_max_abs_dp"] for q in qs)
            if h2q is not None:
                lh = np.asarray(h2q["logits"], np.float32)
                rr["logits_bit_equal_python_head_graph"] = bool(lh.shape == logits.shape
                                                                and np.array_equal(lh.view(np.uint32), logits.view(np.uint32)))
                rr["logits_max_abs_diff_python_head_graph"] = float(np.abs(lh.astype(np.float64) - logits).max())
                rr["vs_python_head_graph_max_abs_dp"] = max(q["vs_python_head_graph_max_abs_dp"] for q in qs)
        if decide_ref and key in decide_ref:
            dr = decide_ref[key]["response"]
            rr["response_equal_decide_py"] = dr == resp
            rr["response_max_abs_diff_decide_py"] = decide.compare_answers(resp, dr)["max_abs_diff"]
        qrows += qs
        run_rows.append(rr)
    far = [q for q in qrows if not q["near_tie"]]
    near = [q for q in qrows if q["near_tie"]]
    worst = max(run_rows, key=lambda x: x["max_abs_dp"]) if run_rows else None
    s = {"runs": len(run_rows), "expected_runs": len(expected), "questions": len(qrows),
         "questions_non_near_tie": len(far), "argmax_equal_non_near_tie": sum(q["argmax_equal"] for q in far),
         "near_tie_questions": len(near), "argmax_equal_near_tie": sum(q["argmax_equal"] for q in near),
         "max_abs_dp": max((q["max_abs_dp"] for q in qrows), default=None),
         "mean_of_run_mean_abs_dp": float(np.mean([x["mean_abs_dp"] for x in run_rows])) if run_rows else None,
         "max_abs_dlogit_vs_oracle": max((q["max_abs_dlogit"] for q in qrows), default=None),
         "worst_run": {"id": worst["id"], "arm": worst["arm"], "max_abs_dp": worst["max_abs_dp"]} if worst else None,
         "finite_all": all(x["finite"] for x in run_rows),
         "reset_bit_equal": bool(sw.get("reset_check", {}).get("hidden_bit_equal") and sw["reset_check"].get("logits_bit_equal")),
         "ids_equal_oracle": sum(x["ids_equal_oracle"] for x in run_rows),
         "spans_equal_oracle": sum(x["spans_equal_oracle"] for x in run_rows),
         "softmax_recomputed_bit_equal_runs": sum(x["softmax_recomputed_bit_equal"] for x in run_rows),
         "response_recomputed_equal_runs": sum(x["response_recomputed_equal"] for x in run_rows),
         "near_ties": [{k: q[k] for k in ("id", "arm", "question_id", "oracle_top2_margin", "argmax_equal", "probs", "probs_oracle")}
                       for q in near]}
    checks = {"all_runs": s["runs"] == s["expected_runs"] and not g1_bad,
              "ids_spans_rope_usage": not g1_bad,
              "argmax_non_near_tie": s["argmax_equal_non_near_tie"] == s["questions_non_near_tie"],
              "max_abs_dp": s["max_abs_dp"] is not None and s["max_abs_dp"] <= BAR["max_abs_dp"],
              "mean_of_run_mean_abs_dp": s["mean_of_run_mean_abs_dp"] is not None
              and s["mean_of_run_mean_abs_dp"] <= BAR["mean_of_run_mean_abs_dp"],
              "finite": s["finite_all"], "reset_bit_equal": s["reset_bit_equal"]}

    def vs(sel):
        sel = list(sel)
        if not sel:
            return {"runs": 0}
        out = {"runs": len(sel)}
        if any("hidden_sha256_equal_python" in x for x in sel):
            out["hidden_bit_equal_runs"] = sum(x.get("hidden_sha256_equal_python", False) for x in sel)
            out["hidden_max_abs_diff_where_unequal"] = max((x.get("hidden_max_abs_diff_python", 0.0) for x in sel), default=None)
        if any("vs_python_author_head_max_abs_dp" in x for x in sel):
            out["max_abs_dp_vs_python_author_head"] = max(x.get("vs_python_author_head_max_abs_dp", 0.0) for x in sel)
        if any("logits_bit_equal_python_head_graph" in x for x in sel):
            out["logits_bit_equal_runs_python_head_graph"] = sum(x.get("logits_bit_equal_python_head_graph", False) for x in sel)
            out["max_abs_dp_vs_python_head_graph"] = max(x.get("vs_python_head_graph_max_abs_dp", 0.0) for x in sel)
            out["logits_max_abs_diff_python_head_graph"] = max(x.get("logits_max_abs_diff_python_head_graph", 0.0) for x in sel)
        if any("response_equal_decide_py" in x for x in sel):
            out["response_equal_decide_py"] = f"{sum(x.get('response_equal_decide_py', False) for x in sel)}/" \
                                              f"{sum('response_equal_decide_py' in x for x in sel)}"
            out["response_max_abs_diff_decide_py"] = max(x.get("response_max_abs_diff_decide_py", 0.0) for x in sel)
        return out

    by_arm = {a: vs(x for x in run_rows if x["arm"] == a) for a in ("text", "g256", "g448", "native")}
    per_arm = {}
    for a in ("text", "g256", "g448", "native"):
        sel = [q for q in qrows if q["arm"] == a]
        rr = [x for x in run_rows if x["arm"] == a]
        if rr:
            per_arm[a] = {"runs": len(rr), "questions": len(sel),
                          "argmax_equal_non_near_tie": f"{sum(q['argmax_equal'] for q in sel if not q['near_tie'])}/"
                                                       f"{sum(not q['near_tie'] for q in sel)}",
                          "max_abs_dp": max(q["max_abs_dp"] for q in sel),
                          "mean_of_run_mean_abs_dp": float(np.mean([x["mean_abs_dp"] for x in rr]))}
    return {"set": set_name, "fixture_json": sw.get("_path"), "label": sw.get("label"), "assets": sw.get("assets"),
            "binary_sha256": sw.get("environment", {}).get("binary_sha256")
            or "not in the pass (built before the CLI recorded it; L/swift/binary_sha256.txt line 1)",
            "load_first": sw.get("load_first"), "reset_check": sw.get("reset_check"),
            "coreai_cache": {k: sw.get(k) for k in ("coreai_cache_before_load", "coreai_cache_after_load", "coreai_cache_end")},
            "skipped_by_swift": sw.get("skipped"), "skipped_by_gate": {f"{a}/{b}": w for (a, b), w in SKIP.items() if (a, b) in rows},
            "summary": s, "checks": checks, "result": "PASS" if all(checks.values()) else "FAIL", "g1_failures": g1_bad,
            "per_arm": per_arm, "vs_python": {"all": vs(run_rows), "by_arm": by_arm,
                                              "python_readout": refs["readout_file"], "python_readout_sha256": refs["readout_sha256"],
                                              "python_head_graph": refs["h2_file"], "python_bundle": refs["bundle"]},
            "runs": run_rows, "questions": qrows}


def score_oracle_rows(docs: list[dict], scheme: str) -> dict:
    """The g256 / g448 runs fed the oracle's rows: hidden and logits vs the Python decoder + head on the same rows."""
    out = {}
    all_runs = [r for sw in docs for r in sw["runs"]]
    for set_name in SETS:
        refs = py_refs(scheme, set_name)
        sel = [r for r in all_runs if (r["id"], r["arm"]) in refs["runs"]
               and r.get("embeds_file") and f"/embeds/{set_name}/" in r["embeds_file"]]
        rows = []
        for r in sel:
            key = (r["id"], r["arm"])
            ref = refs["runs"][key]
            hp = hidden_of(refs, key)
            row = {"id": key[0], "arm": key[1],
                   "hidden_bit_equal_python": r["hidden_sha256"] == hashlib.sha256(np.ascontiguousarray(hp).tobytes()).hexdigest()}
            if ref["h2"] is not None:
                lh = np.asarray(ref["h2"]["logits"], np.float32)
                lg = np.asarray(r["logits"], np.float32)
                row["logits_bit_equal_python_head_graph"] = bool(np.array_equal(lh.view(np.uint32), lg.view(np.uint32)))
            pr = [np.asarray(q["probs"], np.float64) for q in ref["readout"]["questions"]]
            row["max_abs_dp_vs_python_author_head"] = float(max(np.abs(np.asarray(p, np.float64) - q).max()
                                                                for p, q in zip(r["probabilities"], pr)))
            rows.append(row)
        has_h2 = any("logits_bit_equal_python_head_graph" in x for x in rows)
        out[set_name] = {"runs": len(rows), "hidden_bit_equal": sum(x["hidden_bit_equal_python"] for x in rows),
                         "logits_bit_equal_head_graph": (sum(x["logits_bit_equal_python_head_graph"] for x in rows) if has_h2
                                                         else "n/a (no Python head-graph run on this decoder's rows)"),
                         "max_abs_dp_vs_python_author_head": max((x["max_abs_dp_vs_python_author_head"] for x in rows), default=None),
                         "rows": rows}
    return out


def decide_refs(set_name: str) -> dict:
    p = LANE / "results" / DECIDE[set_name]
    if not p.exists():
        return {}
    return {(r["id"], r["arm"]): r for r in json.loads(p.read_text())["runs"]}


def cmd_score(args) -> int:
    scheme = args.asset.split("_")[0]
    kind = args.asset.split("_")[1]
    tpath = Path(args.transcript)
    rec = {"schema": "clef-flash-swift-gate/1", "asset": args.asset, "asset_note": PY_RESULTS[args.asset][1],
           "gate": "Swift host (apps/ClefFlash, clef-flash CLI) vs the author's fp32 oracle, and vs the Python reference "
                   "of the same assets", "bar": {**BAR, "source": "readout_gate.BAR (unchanged)"},
           "package": package_record(), "generated_at": now()}
    passes = {}
    for set_name, path in (("fixture", args.fixture), ("heldout", args.heldout)):
        if not path:
            continue
        sw = json.loads(Path(path).read_text())
        sw["_path"] = record_path(Path(path).resolve())
        dref = decide_refs(set_name) if scheme == "fp16" else None
        passes[set_name] = score_pass(sw, set_name, scheme, kind, dref)
        p = passes[set_name]
        s = p["summary"]
        print(f"{args.asset} {set_name} {p['result']}: runs {s['runs']}/{s['expected_runs']} argmax {s['argmax_equal_non_near_tie']}/"
              f"{s['questions_non_near_tie']} near-tie {s['argmax_equal_near_tie']}/{s['near_tie_questions']} max|dp| "
              f"{s['max_abs_dp']:.6f} mean {s['mean_of_run_mean_abs_dp']:.3e} reset {s['reset_bit_equal']} ids {s['ids_equal_oracle']} "
              f"spans {s['spans_equal_oracle']} softmax= {s['softmax_recomputed_bit_equal_runs']} response= {s['response_recomputed_equal_runs']}")
        for k, ok in p["checks"].items():
            if not ok:
                print(f"   check {k}: FAIL")
        for f in p["g1_failures"][:8]:
            print("   ", f)
        print("   vs python:", json.dumps(p["vs_python"]["all"]))
    rec["passes"] = passes
    if args.oracle_rows:
        docs = [json.loads(Path(p).read_text()) for p in args.oracle_rows]
        rec["oracle_rows_pass"] = {"fixture_json": [record_path(Path(p).resolve()) for p in args.oracle_rows],
                                   "what": "g256 / g448 runs fed the oracle's fp32 rows (the readout_gate.py inputs): the "
                                           "Swift decoder + head vs the Python decoder + head on identical inputs",
                                   "reset_checks": [d.get("reset_check") for d in docs],
                                   **score_oracle_rows(docs, scheme)}
        for k, v in rec["oracle_rows_pass"].items():
            if isinstance(v, dict) and "runs" in v:
                print(f"oracle rows {k}: runs {v['runs']} hidden bit-equal {v['hidden_bit_equal']} logits bit-equal "
                      f"{v['logits_bit_equal_head_graph']} max|dp| vs author head {v['max_abs_dp_vs_python_author_head']}")
    if args.same_as:
        # another asset's passes of the same runs (e.g. the JIT specialization vs the AOT compile of the same .aimodel)
        other = {}
        for path in args.same_as:
            for r in json.loads(Path(path).read_text())["runs"]:
                other[(r["id"], r["arm"])] = r
        mine = [r for set_name, path in (("fixture", args.fixture), ("heldout", args.heldout)) if path
                for r in json.loads(Path(path).read_text())["runs"]]
        both = [(r, other[(r["id"], r["arm"])]) for r in mine if (r["id"], r["arm"]) in other]
        rec["same_as"] = {
            "other_passes": [record_path(Path(p).resolve()) for p in args.same_as], "runs_compared": len(both),
            "hidden_bit_equal": sum(a["hidden_sha256"] == b["hidden_sha256"] for a, b in both),
            "logits_bit_equal": sum(np.array_equal(np.asarray(a["logits"], np.float32).view(np.uint32),
                                                   np.asarray(b["logits"], np.float32).view(np.uint32)) for a, b in both),
            "image_rows_bit_equal": f"{sum(a.get('image_rows_sha256') == b.get('image_rows_sha256') for a, b in both if 'image_rows_sha256' in a)}/"
                                    f"{sum('image_rows_sha256' in a for a, _ in both)}",
            "responses_equal": sum(a["response"] == b["response"] for a, b in both)}
        print("same as other passes:", json.dumps(rec["same_as"]))
    if passes:
        allq = [q for p in passes.values() for q in p["questions"]]
        allr = [r for p in passes.values() for r in p["runs"]]
        far = [q for q in allq if not q["near_tie"]]
        near = [q for q in allq if q["near_tie"]]
        rec["combined"] = {"runs": len(allr), "questions_non_near_tie": len(far),
                           "argmax_equal_non_near_tie": sum(q["argmax_equal"] for q in far),
                           "near_tie_questions": len(near), "argmax_equal_near_tie": sum(q["argmax_equal"] for q in near),
                           "max_abs_dp": max(q["max_abs_dp"] for q in allq),
                           "result": "PASS" if all(p["result"] == "PASS" for p in passes.values()) else "FAIL"}
    tpath.parent.mkdir(parents=True, exist_ok=True)
    tpath.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    print(f"transcript: {tpath}")
    return 0


# --------------------------------------------------------------------------- negative control
def cmd_negative(args) -> int:
    base = json.loads(Path(args.base).read_text())
    neg = json.loads(Path(args.changed).read_text())
    o = oracle_rows("fixture")[(args.id, "text")]

    def vs_oracle(t):
        return {"ids_equal": t["ids"] == o["ids"],
                "spans_equal": [[q["question_span"], q["option_spans"]] for q in t["questions"]]
                == [[q["question_span"], q["option_spans"]] for q in o["questions"]]}

    qs = []
    for qa, qb, pa, pb in zip(base["questions"], neg["questions"], base["probabilities"], neg["probabilities"]):
        pa, pb = np.asarray(pa, np.float64), np.asarray(pb, np.float64)
        qs.append({"question_id": qa["question_id"], "max_abs_dp": float(np.abs(pa - pb).max()),
                   "argmax": [int(pa.argmax()), int(pb.argmax())], "p_base": pa.tolist(), "p_changed": pb.tolist(),
                   "question_span": [qa["question_span"], qb["question_span"]]})
    rec = {"schema": "clef-flash-swift-negative/1", "generated_at": now(), "run": f"{args.id}/text",
           "base_trace": record_path(Path(args.base).resolve()), "changed_trace": record_path(Path(args.changed).resolve()),
           "changed_request": neg.get("request"), "tokens": [base["tokens"], neg["tokens"]],
           "base_vs_oracle": vs_oracle(base), "changed_vs_oracle": vs_oracle(neg), "questions": qs,
           "max_abs_dp": max(q["max_abs_dp"] for q in qs)}
    rec["red"] = (rec["base_vs_oracle"]["ids_equal"] and rec["base_vs_oracle"]["spans_equal"]
                  and not rec["changed_vs_oracle"]["ids_equal"] and not rec["changed_vs_oracle"]["spans_equal"]
                  and rec["max_abs_dp"] > 0)
    Path(args.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("tokens", "base_vs_oracle", "changed_vs_oracle", "max_abs_dp", "red")}))
    for q in qs:
        print(f"  {q['question_id']}: max|dp| {q['max_abs_dp']:.4f} argmax {q['argmax']} span {q['question_span']}")
    return 0 if rec["red"] else 1


# --------------------------------------------------------------------------- towers
def cmd_towers(args) -> int:
    import asyncio
    import inspect

    import coreai.runtime as rt

    async def maybe(x):
        return await x if inspect.isawaitable(x) else x

    docs = [json.loads(Path(p).read_text()) for p in args.passes]
    towers = docs[0]["assets"]["towers"]
    rows = []

    async def go():
        fns = {}
        for g, path in towers.items():
            m = await maybe(rt.AIModel.load(path, rt.SpecializationOptions.default()))
            fns[g] = await maybe(m.load_function(m.function_names[0]))
        for sw in docs:
            dump = Path(sw["dump_dir"])
            fx = Path(sw["images_dir"])
            for r in sw["runs"]:
                if r.get("image_rows_from") != "tower" or not r.get("embeds_dump"):
                    continue
                tile = 256 if r["arm"] == "g256" else 448
                patches = host.preprocess(fx / r["image"], tile)
                res = await maybe(fns[r["arm"]](inputs={"patches": rt.NDArray(np.ascontiguousarray(patches))}))
                py = np.asarray(res["image_embeds"].numpy(), np.float32)
                sw_e = np.fromfile(dump / r["embeds_dump"], np.float32).reshape(py.shape)
                rows.append({"id": r["id"], "arm": r["arm"], "image": r["image"],
                             "bit_equal": bool(np.array_equal(py.view(np.uint32), sw_e.view(np.uint32))),
                             "max_abs_diff": float(np.abs(py.astype(np.float64) - sw_e).max()),
                             "patches_sha256_equal": r["patches_sha256"] == hashlib.sha256(patches.tobytes()).hexdigest()})

    asyncio.run(go())
    rep = {"schema": "clef-flash-swift-towers/1", "generated_at": now(), "towers": towers,
           "what": "Swift VisionTower output (dumped by the gate pass) vs the Python runtime (coreai.runtime, "
                   "SpecializationOptions.default) on host.preprocess (Pillow) patches, same AOT asset",
           "runs": len(rows), "bit_equal": sum(x["bit_equal"] for x in rows),
           "patches_equal": sum(x["patches_sha256_equal"] for x in rows),
           "max_abs_diff": max((x["max_abs_diff"] for x in rows), default=None), "rows": rows}
    out = Path(args.out)
    out.write_text(json.dumps(rep, indent=1) + "\n")
    print(f"towers: {rep['runs']} runs, bit-equal {rep['bit_equal']}, patches equal {rep['patches_equal']}, "
          f"max |d| {rep['max_abs_diff']} -> {out}")
    return 0 if rep["bit_equal"] == rep["runs"] else 1


# --------------------------------------------------------------------------- timing
def cmd_timing(args) -> int:
    run_dir = Path(args.run_dir)
    lock = json.loads((run_dir / "gpu_lock.json").read_text())
    out = {"schema": "clef-flash-swift-timing/1", "run_dir": str(run_dir), "generated_at": now(),
           "gpu_lock": {k: lock.get(k) for k in ("state", "waited_s", "file_existed", "file_bytes_before", "file_after",
                                                 "started", "finished")},
           "passes": []}
    by_bundle: dict = {}
    for p in lock["passes"]:
        js = Path(p["json"])
        if not js.exists():
            out["passes"].append({"bundle": p["bundle"], "pass": p["pass"], "exit": p["exit"], "missing": True})
            continue
        sw = json.loads(js.read_text())
        runs = sw["runs"]
        rows = []
        for r in runs:
            sec = r["seconds"]
            calls = r["call_ms"]
            rows.append({"id": r["id"], "arm": r["arm"], "tokens": len(r["ids"]), "calls": r["calls"], "bucket": r["bucket"][0],
                         "wall_from_file_ms": r["wall_from_file_s"] * 1e3,
                         "stages_ms": {k: v * 1e3 for k, v in sec.items()},
                         "decoder_ms_per_call_median": float(np.median(calls)), "decoder_call_ms": calls,
                         "image_file_decode_ms": r["image_file_decode_s"] * 1e3})
        rec = {"bundle": p["bundle"], "pass": p["pass"], "exit": p["exit"], "process_s": p["s"],
               "demoted_samples": p.get("demoted_samples"), "label": sw.get("label"),
               "load_first": sw.get("load_first"), "load_reload": sw.get("load_reload"),
               "warmup": sw.get("warmup"), "coreai_cache_before_load": sw.get("coreai_cache_before_load", {}).get("bytes"),
               "binary_sha256": sw.get("environment", {}).get("binary_sha256"),
               "reset_check": sw.get("reset_check"), "runs": rows}
        gate = LANE / "swift" / "gate" / f"{p['bundle'].replace('clef_flash_decode_', '').split('_')[0]}_aot" / "fixture.json"
        if gate.exists():
            g = {(r["id"], r["arm"]): r for r in json.loads(gate.read_text())["runs"]}
            same = [r["hidden_sha256"] == g[(r["id"], r["arm"])]["hidden_sha256"] and r["logits"] == g[(r["id"], r["arm"])]["logits"]
                    and r["response"] == g[(r["id"], r["arm"])]["response"] for r in runs if (r["id"], r["arm"]) in g]
            rec["outputs_equal_gate_pass"] = {"gate_pass": str(gate), "runs": len(same), "equal": sum(same)}
        out["passes"].append(rec)
        by_bundle.setdefault(p["bundle"], []).append(rec)
    summary = {}
    for b, recs in by_bundle.items():
        per_run: dict = {}
        for rec in recs:
            for r in rec["runs"]:
                per_run.setdefault((r["id"], r["arm"]), []).append(r)
        table = []
        for (rid, arm), rs in per_run.items():
            w = [x["wall_from_file_ms"] for x in rs]
            st = {k: [x["stages_ms"].get(k, 0.0) for x in rs] for k in rs[0]["stages_ms"]}
            pass_medians = [float(np.median([x["wall_from_file_ms"] for x in rec["runs"] if (x["id"], x["arm"]) == (rid, arm)]))
                            for rec in recs]
            table.append({"id": rid, "arm": arm, "tokens": rs[0]["tokens"], "calls": rs[0]["calls"], "bucket": rs[0]["bucket"],
                          "wall_ms_all": w, "wall_ms_median": float(np.median(w)), "wall_ms_min": float(min(w)),
                          "wall_ms_max": float(max(w)), "wall_ms_median_per_pass": pass_medians,
                          "stages_ms_median": {k: float(np.median(v)) for k, v in st.items()},
                          "decoder_ms_per_call_median": float(np.median([c for x in rs for c in x["decoder_call_ms"]]))})
        loads = [{"pass": rec["pass"], "first": rec["load_first"], "reload": rec["load_reload"],
                  "warmup_decision": rec["warmup"][0] if rec.get("warmup") else None} for rec in recs]
        calls = [c for rec in recs for r in rec["runs"] for c in r["decoder_call_ms"]]
        heads = {}
        for rec in recs:
            for r in rec["runs"]:
                heads.setdefault(r["bucket"], []).append(r["stages_ms"]["head"])
        summary[b] = {"passes": len(recs), "runs": sorted(table, key=lambda t: (t["arm"] != "text", t["tokens"])), "loads": loads,
                      "decoder_ms_per_call": {"n": len(calls), "median": float(np.median(calls)), "min": float(min(calls)),
                                              "max": float(max(calls))},
                      "head_ms_by_bucket_median": {k: float(np.median(v)) for k, v in sorted(heads.items())},
                      "state_reset_ms_median": float(np.median([r["stages_ms"]["state_reset"] for rec in recs for r in rec["runs"]])),
                      "outputs_equal_gate_pass": [rec.get("outputs_equal_gate_pass") for rec in recs]}
    out["summary"] = summary
    path = Path(args.out)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"timing -> {path}")
    for b, s in summary.items():
        print(b)
        for t in s["runs"]:
            print(f"  {t['id']}/{t['arm']}: T {t['tokens']} calls {t['calls']} {t['bucket']} wall {t['wall_ms_median']:.0f} ms "
                  f"({t['wall_ms_min']:.0f}-{t['wall_ms_max']:.0f}; pass medians {[round(x) for x in t['wall_ms_median_per_pass']]}) "
                  f"decoder/call {t['decoder_ms_per_call_median']:.1f} ms")
        print("  decoder ms/call", s["decoder_ms_per_call"], "head", s["head_ms_by_bucket_median"],
              "state reset", round(s["state_reset_ms_median"], 2), "outputs = gate", s["outputs_equal_gate_pass"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    sub.add_parser("render")
    sub.add_parser("pixels")
    a = sub.add_parser("score")
    a.add_argument("--asset", required=True, choices=sorted(PY_RESULTS))
    a.add_argument("--fixture")
    a.add_argument("--heldout")
    a.add_argument("--oracle-rows", nargs="+")
    a.add_argument("--same-as", nargs="+", help="another asset's pass JSONs over the same runs: bit-equality counts")
    a.add_argument("--transcript", required=True)
    n = sub.add_parser("negative")
    n.add_argument("--base", required=True)
    n.add_argument("--changed", required=True)
    n.add_argument("--id", default="own_t01")
    n.add_argument("--out", default=str(LANE / "results" / "swift_negative_control.json"))
    w = sub.add_parser("towers")
    w.add_argument("--passes", nargs="+", required=True)
    w.add_argument("--out", default=str(LANE / "results" / "swift_towers.json"))
    t = sub.add_parser("timing")
    t.add_argument("--run-dir", required=True)
    t.add_argument("--out", default=str(LANE / "results" / "swift_timing.json"))
    args = ap.parse_args()
    return {"prep": cmd_prep, "render": cmd_render, "pixels": cmd_pixels, "score": cmd_score, "negative": cmd_negative,
            "towers": cmd_towers, "timing": cmd_timing}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
