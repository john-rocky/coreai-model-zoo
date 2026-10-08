#!/usr/bin/env python3
"""The device gate's fixture subset and its two references, from the lane's files (stdlib only; ./_stage.sh runs it).

    /usr/bin/python3 -I _fixtures.py <lane dir> <out dir>        # e.g. ~/code/coreai/_d1_omni ~/code/coreai/_d1_omni/device/fixtures

Writes into <out dir>:
  subset.json      the records the gate answers and the rows it scores: 60 text rows (card 7 + semif 20 + tv4 10 +
                   own 19 + long_3400 4, buckets 256 and 4096), 9 image rows (img_01..03, buckets 256 and 2048), 9 audio
                   rows (aud_01..03, the 10 s clip bucket, bucket 256); each row's ids, markers, bucket, positions, P
  oracle_slim.json per row the publisher's fp32 oracle as the Mac gates read it: probabilities (reported order),
                   argmax index, top-2 margin, near tie, type, K
  mac_ref.json     per row the Mac's Swift run of the same shipping graphs (round 10: apps/D1Omni on macos-ship, the JIT
                   .aimodel and the AOT h16c .aimodelc, bit-equal to each other and to the Python runtime): the marker
                   logits and the probabilities as float32 bits; per image and clip the sha256 of every array the host
                   makes (the decoded RGB, each crop and its four vision inputs, the samples, the mel, the four masks),
                   of each media graph output and of the prefix — the Python host's dump, which round 10 found bit-equal
                   to the Swift host on the Mac for these six items (asserted here from the Swift parity files)
                   rows_l64: the 18 text rows of at most 64 positions on the decision graph L64 (round 12's shipping
                   bucket, macos-ship-small; the Mac's Swift run of round 12, JIT = AOT h16c = the Python runtime)
  bench.json       the timed workloads W1..W5 (the ladder's rows; W1 / W2 on L64 and L256) and the series rules
  media/           the three PNG and the three WAV files, copied

Text rows come from the text oracle's fixture (records.v1.json, ref/records_ref.json, results/ship_swift_ref), media rows
from the media oracle's fixture (records.json, ref/records_ref_{images,audio}.json, results/ship_swift_ref_media).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

TEXT_RULE = {
    "card": "every text-mode row of the 5 card records (card_text x3, card_batch_00, card_batch_01, card_cats, card_audio)",
    "semif": "the first 20 rows of semif_authored144 in fixture order",
    "tv4": "the first 10 bucket-256 rows of transfer_v4_dev_head (MMLU) in fixture order (tv4_009 is bucket 512)",
    "own": "the bucket-256 text rows of own_t01, own_t03, own_t04, own_t05 (text states) and own_j01, own_j04 (JSON "
           "states; own_j01/size is bucket 512)",
    "long": "the 4 rows of long_3400 (3,449..3,497 positions, bucket 4096)",
}
OWN = ["own_t01", "own_t03", "own_t04", "own_t05", "own_j01", "own_j04"]
CARD = ["card_text", "card_batch_00", "card_batch_01", "card_cats", "card_audio"]
IMAGES = ["img_01", "img_02", "img_03"]
CLIPS = ["aud_01", "aud_02", "aud_03"]
EXPECT = {"card": 7, "semif": 20, "tv4": 10, "own": 19, "long": 4, "image": 9, "audio": 9}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load(p: Path):
    return json.loads(p.read_text())


def src(p: Path) -> dict:
    return {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}


def key(r: dict) -> str:
    return f"{r['id']}/{r['qid']}/{r['mode']}"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    lane, out = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser()
    R = lane / "results"
    files = {
        "fixture_text": lane / "fixtures" / "records.v1.json",
        "fixture_media": lane / "fixtures" / "records.json",
        "python_text": R / "ship_swift_ref" / "swift_ref.json",
        "python_media": R / "ship_swift_ref_media" / "index.json",
        "swift_text_jit": R / "ship_swift_parity_jit.json",
        "swift_text_aot": R / "ship_swift_parity_aot.json",
        "swift_media_jit": R / "ship_swift_media_parity_jit.json",
        "swift_media_aot": R / "ship_swift_media_parity_aot.json",
        "swift_gate": R / "ship_swift_parity.json",
        "swift_l64_jit": R / "small_swift_parity_jit.json",
        "swift_l64_aot": R / "small_swift_parity_aot.json",
    }
    for f in files.values():
        if not f.is_file():
            print(f"missing: {f}")
            return 1
    sources = {k: src(p) for k, p in files.items()}
    fx_text = {r["id"]: r for r in load(files["fixture_text"])["records"]}
    fx_media = {r["id"]: r for r in load(files["fixture_media"])["records"]}
    order_text = {r: i for i, r in enumerate(fx_text)}
    py_text = load(files["python_text"])
    py_media = load(files["python_media"])
    assert py_text["fixtures"]["sha256"] == sources["fixture_text"]["sha256"], "swift_ref was not made from records.v1.json"
    assert py_media["fixtures"]["sha256"] == sources["fixture_media"]["sha256"], "the media dump was not made from records.json"
    assert py_text["ship"] and py_media["ship"], "the references are not the shipping (stripped) graphs'"
    gate = load(files["swift_gate"])

    # ---- the text rows (Python runtime rows of round 10, text mode)
    trows = [r for r in py_text["rows"] if r["mode"] == "text"]
    trows.sort(key=lambda r: (order_text[r["id"]], r["qid"]))
    # keep the question order of each record's request (the rows of one record follow its questions)
    qorder = {rid: {q: i for i, q in enumerate(fx_text[rid]["request"]["questions"])} for rid in fx_text}
    trows.sort(key=lambda r: (order_text[r["id"]], qorder[r["id"]][r["qid"]]))
    picked: dict[str, list] = {"card": [], "semif": [], "tv4": [], "own": [], "long": []}
    for r in trows:
        if r["id"] in CARD:
            picked["card"].append(r)
        elif r["source"] == "semif_authored144" and len(picked["semif"]) < 20:
            picked["semif"].append(r)
        elif r["source"] == "transfer_v4_dev_head" and r["bucket"] == 256 and len(picked["tv4"]) < 10:
            picked["tv4"].append(r)
        elif r["id"] in OWN and r["bucket"] == 256:
            picked["own"].append(r)
        elif r["id"] == "long_3400":
            picked["long"].append(r)
    for g, rs in picked.items():
        assert len(rs) == EXPECT[g], (g, len(rs))

    # ---- the media rows (the Python host's media rows of round 10: images "python", audio "python_numpy_mel")
    mrows = {g: [] for g in ("image", "audio")}
    for r in py_media["rows"]:
        if r["id"] in IMAGES and r["arm"] == "python":
            mrows["image"].append(r)
        elif r["id"] in CLIPS and r["arm"] == "python_numpy_mel":
            mrows["audio"].append(r)
    for g in mrows:
        mrows[g].sort(key=lambda r: (r["id"], list(fx_media[r["id"]]["request"]["questions"]).index(r["qid"])))
        assert len(mrows[g]) == EXPECT[g], (g, len(mrows[g]))

    # ---- the Mac's Swift rows (JIT and AOT): bit-equal to each other and to the Python rows
    def swift_rows(path: Path) -> dict:
        return {key(r): r for r in load(path)["rows"]}

    st_jit, st_aot = swift_rows(files["swift_text_jit"]), swift_rows(files["swift_text_aot"])
    sm_jit_doc, sm_aot_doc = load(files["swift_media_jit"]), load(files["swift_media_aot"])
    sm_jit = {key(r): r for r in sm_jit_doc["rows"] if r["python_arm"] in ("python", "python_numpy_mel")}
    sm_aot = {key(r): r for r in sm_aot_doc["rows"] if r["python_arm"] in ("python", "python_numpy_mel")}

    subset_rows, oracle_rows, mac_rows = [], {}, {}
    groups = [(g, rs, "text") for g, rs in picked.items()] + [(g, rs, g) for g, rs in mrows.items()]
    for group, rs, _ in groups:
        for r in rs:
            k = key(r)
            sj, sa = (st_jit, st_aot) if r["mode"] == "text" else (sm_jit, sm_aot)
            assert k in sj and k in sa, f"no Mac Swift row for {k}"
            for s in (sj[k], sa[k]):
                assert s["python_bit_equal"] and s["logits_bits"] == r["logits_bits"] and s["probs_bits"] == r["probs_bits"], k
                assert s["bucket"] == r["bucket"] and s["positions"] == r["positions"], k
            assert sj[k]["logits_bits"] == sa[k]["logits_bits"] and sj[k]["probs_bits"] == sa[k]["probs_bits"], k
            subset_rows.append({"key": k, "group": group, "id": r["id"], "qid": r["qid"], "mode": r["mode"],
                                "source": r["source"], "type": r["type"], "K": r["K"], "bucket": r["bucket"],
                                "positions": r["positions"], "prefix_len": r.get("prefix_len", 0) or 0,
                                "ids": r["ids"], "ids_sha256": r["ids_sha256"], "markers": r["markers"],
                                "calibrate": r["calibrate"]})
            probs = r["oracle_probs"]
            oracle_rows[k] = {"type": r["type"], "K": r["K"], "probs": probs, "argmax_index": r["argmax_index"],
                              "top2_margin": r["top2_margin"], "near_tie": r["near_tie"],
                              "logits_raw": r.get("oracle_logits_raw")}
            mac_rows[k] = {"logits_bits": sj[k]["logits_bits"], "probs_bits": sj[k]["probs_bits"],
                           "python_logits_bits": r["logits_bits"], "python_probs_bits": r["probs_bits"],
                           "jit_equal_aot": True, "python_equal": True}

    # ---- the rows of at most 64 positions on L64 (round 12: host.ALL_BUCKETS, the smallest bucket that holds the row)
    l64_jit, l64_aot = swift_rows(files["swift_l64_jit"]), swift_rows(files["swift_l64_aot"])
    for d in (load(files["swift_l64_jit"]), load(files["swift_l64_aot"])):
        assert d["fixtures"] == str(files["fixture_text"]), d["fixtures"]
    rows_l64 = {}
    for r in subset_rows:
        if r["mode"] != "text" or r["positions"] > 64:
            continue
        k = r["key"]
        assert k in l64_jit and k in l64_aot, f"no L64 Swift row for {k}"
        sj, sa = l64_jit[k], l64_aot[k]
        assert sj["bucket"] == sa["bucket"] == 64 and sj["python_bit_equal"] and sa["python_bit_equal"], k
        assert sj["logits_bits"] == sa["logits_bits"] and sj["probs_bits"] == sa["probs_bits"], k
        assert sj["positions"] == r["positions"], k
        rows_l64[k] = {"logits_bits": sj["logits_bits"], "probs_bits": sj["probs_bits"], "bucket": 64,
                       "jit_equal_aot": True, "python_equal": True}
    assert len(rows_l64) == 18, len(rows_l64)

    # ---- media items: the Python dump's array hashes (bit-equal to the Mac's Swift host, asserted from its parity)
    def by_id(items):
        return {x["id"]: x for x in items}

    img_py, clip_py = by_id(py_media["images"]), by_id(py_media["clips"])
    img_sw = {d: by_id(doc["images"]) for d, doc in (("jit", sm_jit_doc), ("aot", sm_aot_doc))}
    clip_sw = {d: by_id(doc["clips"]) for d, doc in (("jit", sm_jit_doc), ("aot", sm_aot_doc))}
    media_images, media_clips, mac_images, mac_clips = [], [], {}, {}
    (out / "media").mkdir(parents=True, exist_ok=True)
    for i in IMAGES:
        p = img_py[i]
        crops = []
        for k, c in enumerate(p["crops"]):
            f = c["files"]
            for d in ("jit", "aot"):
                s = img_sw[d][i]["crops"][k]
                assert all(s[n]["bit_equal"] for n in ("crop_u8", "pixel_values", "pos_embed", "patch_mask", "unshuffle_index")), (i, k, d)
                assert s["graph_output_sha256"] == f["graph_output_sha256"] == s["python_graph_output_sha256"], (i, k, d)
            crops.append({"k": k, "shape_hw": f["crop"]["shape"][1:], "crop_u8": f["crop"]["sha256"],
                          "pixel_values": f["pixel_values"]["sha256"], "pos_embed": f["pos_embed"]["sha256"],
                          "patch_mask": f["patch_mask"]["sha256"], "unshuffle_index": f["unshuffle_index"]["sha256"],
                          "output": f["graph_output_sha256"]})
        for d in ("jit", "aot"):
            assert img_sw[d][i]["rgb"]["bit_equal"] and img_sw[d][i]["prefix"]["bit_equal"], (i, d)
        mac_images[i] = {"rgb": p["files"]["rgb"]["sha256"], "prefix": p["files"]["prefix"]["sha256"], "crops": crops,
                         "px_wh": img_sw["jit"][i]["px_wh"], "prefix_rows": p["files"]["prefix"]["shape"][0]}
        name = Path(p["file"]).name
        shutil.copyfile(lane / p["file"], out / "media" / name)
        media_images.append({"id": i, "file": f"media/{name}", "file_sha256": sha256_file(out / "media" / name),
                             "px_wh": img_sw["jit"][i]["px_wh"], "crops": len(crops),
                             "prefix_rows": p["files"]["prefix"]["shape"][0]})
    for c in CLIPS:
        p = clip_py[c]
        f = p["files"]
        for d in ("jit", "aot"):
            s = clip_sw[d][c]
            assert all(s[n]["bit_equal"] for n in ("samples", "mel_vs_numpy", "mask_f", "mask_f2", "mask_f4", "mask_t",
                                                   "prefix_vs_numpy")), (c, d)
            assert s["graph_output_sha256"] == f["graph_output_numpy_sha256"], (c, d)
        mac_clips[c] = {"samples": f["samples"]["sha256"], "mel": f["mel_numpy"]["sha256"], "mask_f": f["mask_f"]["sha256"],
                        "mask_f2": f["mask_f2"]["sha256"], "mask_f4": f["mask_f4"]["sha256"],
                        "mask_t": f["mask_t"]["sha256"], "output": f["graph_output_numpy_sha256"],
                        "prefix": f["prefix_numpy"]["sha256"], "bucket_s": p["bucket_s"], "frames": p["frames"],
                        "prefix_rows": p["prefix_rows"]}
        name = Path(p["file"]).name
        shutil.copyfile(lane / p["file"], out / "media" / name)
        assert sha256_file(out / "media" / name) == p["file_sha256"], c
        media_clips.append({"id": c, "file": f"media/{name}", "file_sha256": p["file_sha256"], "samples": p["samples"],
                            "bucket_s": p["bucket_s"], "frames": p["frames"], "prefix_rows": p["prefix_rows"]})

    # ---- the records the gate answers (request as written), text from v1, media from v3
    records = []
    for rid in dict.fromkeys(r["id"] for r in subset_rows if r["mode"] == "text"):
        records.append({"id": rid, "kind": "text", "source": fx_text[rid]["source"], "request": fx_text[rid]["request"]})
    for rid in IMAGES:
        records.append({"id": rid, "kind": "image", "source": fx_media[rid]["source"], "request": fx_media[rid]["request"],
                        "images": [m["file"] for m in media_images if m["id"] == rid]})
    for rid in CLIPS:
        records.append({"id": rid, "kind": "audio", "source": fx_media[rid]["source"], "request": fx_media[rid]["request"],
                        "audio": next(m["file"] for m in media_clips if m["id"] == rid)})

    counts = {g: sum(1 for r in subset_rows if r["group"] == g) for g in EXPECT}
    buckets = {}
    for r in subset_rows:
        buckets.setdefault(r["mode"], {}).setdefault(str(r["bucket"]), 0)
        buckets[r["mode"]][str(r["bucket"])] += 1
    out.mkdir(parents=True, exist_ok=True)
    subset = {"schema": "d1omni-gate-subset/1", "rule": TEXT_RULE | {
                  "image": "every image-mode row of img_01, img_02, img_03 (one, one and seven crops; img_03's rows are "
                           "bucket 2048)",
                  "audio": "every audio-mode row of aud_01, aud_02, aud_03 (9.6 / 8.8 / 8.7 s, the 10 s clip bucket)"},
              "l64": {"rule": "every subset text row of at most 64 positions, again on the decision graph L64 (host.ALL_BUCKETS: "
                             "the smallest bucket that holds the row); its round-10 bucket stays the bit reference of mac_ref.rows",
                      "keys": sorted(rows_l64)},
              "counts": counts, "rows_total": len(subset_rows), "buckets_by_mode": buckets,
              "decision_buckets": sorted({r["bucket"] for r in subset_rows}), "audio_buckets_s": [10],
              "records": records, "rows": subset_rows, "images": media_images, "clips": media_clips,
              "sources": sources}
    oracle = {"schema": "d1omni-gate-oracle-slim/1",
              "bar": {"name": "FACTS §7", "argmax": "every row whose oracle top-2 margin is above 0.02 (near ties apart)",
                      "max_abs_dp": 0.02, "mean_row_max_abs_dp": 0.002,
                      "mean_definition": "the mean over rows of each row's max |dp| (conversion/d1_omni/_metrics.py)"},
              "rows": oracle_rows,
              "sources": {k: sources[k] for k in ("python_text", "python_media")}}
    mac = {"schema": "d1omni-gate-mac-ref/1",
           "what": "the Mac's Swift host (apps/D1Omni, round 10, macos-ship) on the same graphs: marker logits and "
                   "probabilities as float32 bits (JIT = AOT h16c = the Python runtime, every row); the arrays of the "
                   "host's media path as sha256 of their bytes (row-major, little-endian)",
           "swift_gate": {"status": gate.get("status"), "summary": gate.get("summary")},
           "rows": mac_rows, "rows_l64": rows_l64, "images": mac_images, "clips": mac_clips,
           "sources": {k: sources[k] for k in sources if k.startswith("swift_")}}
    bench = {"schema": "d1omni-gate-bench/1",
             "workloads": [
                 {"id": "W1", "what": "one question, short state", "record": "card_text", "qids": ["refund"], "mode": "text",
                  "buckets": [64, 256]},
                 {"id": "W2", "what": "three questions in one pass (3 calls)", "record": "card_text",
                  "qids": ["refund", "team", "urgency"], "mode": "text", "buckets": [64, 256]},
                 {"id": "W3", "what": "one question over a 3.4k-token state", "record": "long_3400", "qids": ["tension_10"],
                  "mode": "text", "buckets": [4096], "interleave": False},
                 {"id": "W4", "what": "one question about a 384 px image (1 crop)", "record": "img_01", "qids": ["room"],
                  "mode": "image", "buckets": [256]},
                 {"id": "W5", "what": "one question about 10 s of audio", "record": "aud_01", "qids": ["topic"],
                  "mode": "audio", "buckets": [256]}],
             "order": ["W1", "W2", "W4", "W5", "W3"],
             "interleave": "the forms of a workload alternate per round in one process; W3 (L4096) runs one form after the "
                           "other (two L4096 graphs at once would hold two of the largest working sets on the phone)",
             "one_decision": "graph inputs -> call(s) -> scores copied -> marker gather + temperature + float32 softmax "
                             "(the Mac's `d1omni time`); W4 / W5 first run the media graph on inputs prepared before the "
                             "series and cut the P prefix rows; tokenizing and media preprocessing are timed apart",
             "series": {"warmup_first": 3, "warmup_later": 1, "timed": 20, "max_series_s": 18, "rest_s": 10,
                        "forms_alternate": "per round, even rounds JIT then AOT, odd rounds AOT then JIT"}}
    for name, doc in (("subset.json", subset), ("oracle_slim.json", oracle), ("mac_ref.json", mac), ("bench.json", bench)):
        (out / name).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"out": str(out), "counts": counts, "rows": len(subset_rows), "buckets_by_mode": buckets,
                      "records": len(records), "images": [m["id"] for m in media_images],
                      "clips": [m["id"] for m in media_clips]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
