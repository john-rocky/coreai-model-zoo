#!/usr/bin/env python3
"""The Python host's media path on the shipping fp16 graphs, dumped as the reference the Swift host's media path
(apps/D1Omni, round 9) is judged against: the decoded image, every crop and the vision graph's four inputs, the image
prefix; the clip's samples, the mel and the four masks, the audio prefix; every media row's decision.

    PY=~/code/coreai/coreai-models/.venv/bin/python
    $PY conversion/d1_omni/host_dump_media.py            # -> <work>/results/swift_ref_media/ (never replaced)
    $PY conversion/d1_omni/host_dump_media.py --ship     # round 10: the stripped bundles (macos-ship/, their AOT in
                                                         # compiled/ship-h16c/) -> <work>/results/ship_swift_ref_media/

Images (16: img_01..03, imgm_01..12 and the card's cats, the image records of the references): PIL `convert("RGB")`
(rgb.npy, uint8 [h, w, 3]); per crop k of vision.preprocess() the torchvision crop (crop<k>.npy, uint8 [3, h, w]) and
the four graph inputs of host.crop_inputs (crop<k>_<name>.npy: pixel_values / pos_embed [1,1024,768] float32,
patch_mask [1,1024] float32, unshuffle_index [256,4] int32), with the NumPy forms (host.crop_pixels_numpy,
host.position_embeddings_numpy) asserted bit-equal to them; the prefix (prefix.npy [P, 1024] float32) from the vision
fp16 AOT .aimodelc (compiled/vision-fp16-h16c-r7), each crop's first (ph/2)(pw/2) rows in crop order, two calls each
(drift recorded).
Audio (16: card_audio, aud_01..aud_15): the samples as the card and the oracle read them (soundfile, int16:
samples.npy); at the clip's bucket (host.audio_bucket_for) the graph inputs with the NumPy mel (mel_numpy.npy [128, F],
the Swift host's spec: mel_host.mel_numpy, float64 cast to float32) and the torch mel (mel_torch.npy, the publisher's
code = the Python host), the four masks (mask_f / mask_f2 / mask_f4 / mask_t.npy); the prefix of each mel
(prefix_numpy.npy, prefix_torch.npy [P, 1024]) from the audio fp16 AOT of the bucket (compiled/audio-fp16-<sec>s-h16c-r7).
Rows (46 image + 46 audio): host.request_rows in the media mode (asserted equal to the oracle's ids / markers /
positions) -> host.graph_inputs at the row's bucket with the prefix -> the decision fp16 AOT of the bucket -> marker
logits and probabilities (bits too), two calls; an audio row once per mel (numpy = the Swift spec, torch = the Python
host). Every AOT is loaded with SpecializationOptions.from_preferred_compute_unit_kind(gpu), never default().

index.json lists every file with its sha256, shape and dtype; the rows carry the oracle's numbers (the publisher's fp32
model with its own media prefix) for the FACTS §7 bar.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import hf_snapshot, work_path  # noqa: E402

import host  # noqa: E402

WORK = work_path("_d1_omni")
OUT = WORK / "results" / "swift_ref_media"
MACOS = WORK / "bundles" / "d1-omni-600m" / "macos"
COMPILED = WORK / "compiled"
DEC_INPUTS = ("input_ids", "prefix_embeds", "pad_mask", "prefix_mask", "keep_right", "qtype_onehot")
REFERENCES = {"main": "records_ref.json", "images": "records_ref_images.json", "audio": "records_ref_audio.json"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def f32_bits(values) -> list[int]:
    return [int(v) for v in np.asarray(values, dtype=np.float32).view(np.uint32)]


def save(folder: Path, name: str, array: np.ndarray) -> dict:
    a = np.ascontiguousarray(array)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / f"{name}.npy", a, allow_pickle=False)
    return {"file": f"{folder.name}/{name}.npy", "sha256": sha256_array(a), "shape": list(a.shape), "dtype": str(a.dtype)}


def aot_for(source: Path) -> tuple[Path, dict]:
    """The Mac AOT .aimodelc (gpu, h16c) compiled from `source`, its files checked against the manifest's sha256."""
    found = []
    for d in sorted(COMPILED.iterdir()):
        m = d / "provenance" / "aot-manifest.json"
        if not m.exists():
            continue
        aot = json.loads(m.read_text())
        if (aot.get("status") == "COMPILED" and aot.get("preferred_compute") == "gpu" and aot.get("architecture") == "h16c"
                and Path(aot["source"]["folder"]) == source and (d / aot["aimodelc"]).exists()):
            found.append((d / aot["aimodelc"], aot))
    if len(found) != 1:
        raise SystemExit(f"{len(found)} AOT assets for {source}: {[str(f[0]) for f in found]}")
    aimodelc, aot = found[0]
    for item in aot.get("files", []):
        if sha256_file(aimodelc / item["path"]) != item["sha256"]:
            raise SystemExit(f"{aimodelc / item['path']} changed since its compile")
    return aimodelc, aot


def oracle_entries(mode: str) -> list[tuple[dict, str]]:
    """The oracle's media requests of one mode with their reference file's name, in the order vision_check.py /
    audio_check.py use: images = img_01..03, imgm_01..12 (records_ref_images.json), then card_cats (records_ref.json);
    audio = card_audio, aud_01..03 (records_ref.json), then aud_04..aud_15 (records_ref_audio.json)."""
    order = ("images", "main") if mode == "image" else ("main", "audio")
    out, seen = [], set()
    for key in order:
        ref = json.loads((WORK / "ref" / REFERENCES[key]).read_text())
        for entry in ref["records"]:
            if entry["mode"] == mode and entry["id"] not in seen:
                seen.add(entry["id"])
                out.append((entry, REFERENCES[key]))
    return out


def row_doc(rid: str, mode: str, entry: dict, q: dict, r: host.Row, L: int, markers: list[int], inputs: dict,
            z: np.ndarray, p: np.ndarray, drift: float, prefix_file: str, arm: str) -> dict:
    return {"id": rid, "qid": q["qid"], "mode": mode, "arm": arm, "source": entry["source"], "type": q["type"], "K": q["K"],
            "prefix_len": r.prefix_len, "positions": r.positions, "bucket": L, "markers": r.markers, "graph_markers": markers,
            "ids": r.ids, "ids_sha256": hashlib.sha256(json.dumps(r.ids).encode()).hexdigest(),
            "inputs_sha256": {k: sha256_array(inputs[k]) for k in DEC_INPUTS}, "prefix_file": prefix_file,
            "calibrate": r.calibrate, "question": {"type": r.question.type, "instructions": r.question.instructions,
                                                   "criteria": r.question.criteria},
            "logits": [float(v) for v in z], "logits_bits": f32_bits(z), "probs": [float(v) for v in p],
            "probs_bits": f32_bits(p), "drift_marker_logits": drift,
            "oracle_logits_raw": q["logits_raw"], "oracle_probs": q["probs"], "argmax_index": q["argmax_index"],
            "near_tie": q["near_tie"], "top2_margin": q["top2_margin"]}


async def dump() -> dict:
    import coreai.runtime as rt
    import soundfile as sf
    from PIL import Image

    import mel_host

    tok = host.RawTokenizer(Path(hf_snapshot(host.MODEL_ID, revision=host.MODEL_SHA)) / "tokenizer.json")
    host.check_token_ids(tok)
    snapshot = Path(hf_snapshot(host.MODEL_ID, revision=host.MODEL_SHA))
    table = host.load_position_table(snapshot / "model.safetensors")
    bundle_table = np.fromfile(MACOS / "vision-fp16" / "position_table.f32", dtype="<f4").reshape(16, 16, 768)
    assert np.array_equal(table, bundle_table), "the vision bundle's position_table.f32 is not the checkpoint's table"
    filters = np.fromfile(MACOS / "audio-fp16-10s" / mel_host.FILTERBANK_FILE, dtype="<f4").reshape(128, 257)
    assert np.array_equal(filters, mel_host.slaney_filterbank(n_fft=mel_host.N_FFT, n_mels=mel_host.FEATURES))
    fixtures_path = WORK / "fixtures" / "records.json"
    fixtures = {r["id"]: r for r in json.loads(fixtures_path.read_text())["records"]}
    opts =rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    loads, fns, aots = {}, {}, {}

    async def load(key: str, aimodelc: Path, aot: dict):
        t0 = time.perf_counter()
        m = await rt.AIModel.load(aimodelc, opts)
        fns[key] = (m, m.load_function("main"))
        loads[key] = {"aimodelc": str(aimodelc), "load_s": time.perf_counter() - t0,
                      "main_hash": (aimodelc / "main.hash").read_bytes().hex() if (aimodelc / "main.hash").exists() else None}
        aots[key] = {"folder": aot["source"]["folder"], "aimodelc": aimodelc.name, "hashes": aot.get("hashes")}
        print(f"loaded {key} {aimodelc.name} in {loads[key]['load_s']:.2f} s", flush=True)

    await load("vision", *aot_for(MACOS / "vision-fp16"))
    for sec in host.AUDIO_CLIP_SECONDS:
        await load(f"audio_{sec}s", *aot_for(MACOS / f"audio-fp16-{sec}s"))

    async def decision(r: host.Row, prefix: np.ndarray) -> tuple:
        L = host.bucket_for(r.positions)
        key = f"decide_L{L}"
        if key not in fns:
            await load(key, *aot_for(MACOS / f"fp16-L{L}"))
        inputs, markers = host.graph_inputs(r, L, prefix)
        outs = []
        for _ in range(2):
            res = await fns[key][1]({k: rt.NDArray(inputs[k]) for k in DEC_INPUTS})
            outs.append(np.array(res["scores"].numpy(), copy=True).reshape(-1))
        z = outs[0][markers]
        p = host.probabilities_from_logits(z, r.question, r.calibrate)
        drift = float(np.max(np.abs(outs[1][markers].astype(np.float64) - z)))
        return L, markers, inputs, z, p, drift

    images, clips, rows = [], [], []
    # ------------------------------------------------------------------ images
    for entry, ref_name in oracle_entries("image"):
        rid = entry["id"]
        record = fixtures[rid]
        path = WORK / record["media"]["images"][0]
        folder = OUT / rid
        with Image.open(path) as pil:
            pil.load()
            rgb_hwc = np.asarray(pil.convert("RGB"), dtype=np.uint8)
            crops_tv = host.image_crop_pixels(pil)
            crops_np = host.crop_pixels_numpy(host.rgb_uint8(pil))
            size = pil.size
        files = {"rgb": save(folder, "rgb", rgb_hwc)}
        crop_docs, parts, drift = [], [], 0.0
        for k, (crop, crop_np) in enumerate(zip(crops_tv, crops_np)):
            assert np.array_equal(crop, crop_np), f"{rid} crop {k}: the NumPy resize differs from torchvision"
            x = host.crop_inputs(crop, table)
            x_np = host.crop_inputs(crop_np, table, numpy_positions=True)
            for name in host.VISION_INPUT_NAMES:
                assert np.array_equal(x[name], x_np[name]), f"{rid} crop {k}: {name} of the NumPy forms differs"
            cf = {"crop": save(folder, f"crop{k}", crop)}
            for name in host.VISION_INPUT_NAMES:
                cf[name] = save(folder, f"crop{k}_{name}", x[name])
            runs = [np.asarray((await fns["vision"][1]({n: rt.NDArray(np.ascontiguousarray(x[n]))
                                                         for n in host.VISION_INPUT_NAMES}))["prefix"].numpy(),
                               dtype=np.float32).copy() for _ in range(2)]
            drift = max(drift, float(np.max(np.abs(runs[1].astype(np.float64) - runs[0]))))
            out_rows = runs[0].reshape(host.VISION_TOKENS, -1)
            parts.append(out_rows[:x["tokens"]])
            cf["graph_output_sha256"] = sha256_array(runs[0])
            crop_docs.append({"k": k, "size_hw": list(crop.shape[1:]), "grid": list(x["grid"]), "tokens": x["tokens"],
                              "files": cf})
        prefix = np.concatenate(parts, axis=0)
        assert prefix.shape[0] == host.image_prefix_length([size]), (rid, prefix.shape)
        files["prefix"] = save(folder, "prefix", prefix)
        hrows = host.request_rows(tok, record["request"].get("state"), record["request"]["questions"], "image",
                                  host.image_prefix_length([size]))
        by_q = {r.qid: r for r in hrows}
        n_rows = 0
        for q in entry["questions"]:
            r = by_q[q["qid"]]
            if r.ids != q["ids"] or r.markers != q["markers"] or r.positions != q["positions"]:
                raise SystemExit(f"{rid}/{q['qid']}: host row differs from the oracle's")
            L, markers, inputs, z, p, d = await decision(r, prefix)
            rows.append(row_doc(rid, "image", entry, q, r, L, markers, inputs, z, p, d, files["prefix"]["file"], "python"))
            n_rows += 1
        images.append({"id": rid, "reference": ref_name, "public": record["public"], "file": str(path.relative_to(WORK)),
                       "file_sha256": sha256_file(path), "px_wh": list(size), "prefix_rows": int(prefix.shape[0]),
                       "crops": crop_docs, "files": files, "vision_drift": drift, "rows": n_rows})
        print(f"{rid}: {len(crop_docs)} crops, P {prefix.shape[0]}, {n_rows} rows, vision drift {drift}", flush=True)
    # ------------------------------------------------------------------ audio
    for entry, ref_name in oracle_entries("audio"):
        rid = entry["id"]
        record = fixtures[rid]
        path = WORK / record["media"]["audio"]
        samples, rate = sf.read(str(path), dtype="int16")
        assert rate == host.SAMPLE_RATE and samples.ndim == 1, (path, rate, samples.shape)
        folder = OUT / rid
        files = {"samples": save(folder, "samples", samples)}
        sec = host.audio_bucket_for(len(samples))
        x_np = host.audio_inputs(samples, sec, numpy_mel=True)
        x_t = host.audio_inputs(samples, sec)
        for name in ("mask_f", "mask_f2", "mask_f4", "mask_t"):
            assert np.array_equal(x_np[name], x_t[name])
            files[name] = save(folder, name, x_np[name])
        files["mel_numpy"] = save(folder, "mel_numpy", x_np["mel"][0])
        files["mel_torch"] = save(folder, "mel_torch", x_t["mel"][0])
        key, p_rows, steps = f"audio_{sec}s", x_np["prefix_rows"], x_np["steps"]
        prefixes, drift = {}, 0.0
        for arm, x in (("numpy", x_np), ("torch", x_t)):
            runs = [np.asarray((await fns[key][1]({n: rt.NDArray(np.ascontiguousarray(x[n]))
                                                    for n in host.AUDIO_INPUT_NAMES}))["prefix"].numpy(),
                               dtype=np.float32).copy() for _ in range(2)]
            drift = max(drift, float(np.max(np.abs(runs[1].astype(np.float64) - runs[0]))))
            prefixes[arm] = runs[0].reshape(steps, -1)[:p_rows]
            files[f"prefix_{arm}"] = save(folder, f"prefix_{arm}", prefixes[arm])
            files[f"graph_output_{arm}_sha256"] = sha256_array(runs[0])
        mel_d = np.abs(x_np["mel"].astype(np.float64) - x_t["mel"])
        hrows = host.audio_request_rows(tok, record["request"].get("state"), record["request"]["questions"], samples)
        by_q = {r.qid: r for r in hrows}
        n_rows = 0
        for q in entry["questions"]:
            r = by_q[q["qid"]]
            if r.ids != q["ids"] or r.markers != q["markers"] or r.positions != q["positions"]:
                raise SystemExit(f"{rid}/{q['qid']}: host row differs from the oracle's")
            for arm in ("numpy", "torch"):
                L, markers, inputs, z, p, d = await decision(r, prefixes[arm])
                rows.append(row_doc(rid, "audio", entry, q, r, L, markers, inputs, z, p, d, files[f"prefix_{arm}"]["file"],
                                    f"python_{arm}_mel"))
            n_rows += 1
        clips.append({"id": rid, "reference": ref_name, "public": record["public"], "file": str(path.relative_to(WORK)),
                      "file_sha256": sha256_file(path), "samples": int(len(samples)), "bucket_s": sec,
                      "frames": int(x_np["frames"]), "prefix_rows": int(p_rows), "steps": int(steps), "files": files,
                      "mel_numpy_vs_torch_max_abs": float(mel_d.max()),
                      "mel_numpy_vs_torch_bit_equal_elems": int((x_np["mel"] == x_t["mel"]).sum()),
                      "audio_drift": drift, "rows": n_rows})
        print(f"{rid}: {len(samples)} samples, bucket {sec} s, P {p_rows}, mel numpy vs torch {mel_d.max():.3e}, "
              f"{n_rows} rows", flush=True)
    assert all(fns.values())
    return {"images": images, "clips": clips, "rows": rows, "loads": loads, "aots": aots,
            "fixtures": {"path": str(fixtures_path), "sha256": sha256_file(fixtures_path)},
            "position_table_sha256": sha256_array(table), "filterbank_sha256": sha256_array(filters),
            "specialization_options": " ".join(str(opts).split())}


def main() -> int:
    import argparse
    import platform
    from importlib import metadata

    global OUT, MACOS, COMPILED
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ship", action="store_true", help="round 10: the stripped bundles (macos-ship/) and their AOT "
                        "(compiled/ship-h16c/) -> results/ship_swift_ref_media/")
    args = parser.parse_args()
    if args.ship:
        OUT = WORK / "results" / "ship_swift_ref_media"
        MACOS = WORK / "bundles" / "d1-omni-600m" / "macos-ship"
        COMPILED = WORK / "compiled" / "ship-h16c"
    index = OUT / "index.json"
    if index.exists():
        raise SystemExit(f"{index} exists: never replaced")
    t0 = time.perf_counter()
    doc = asyncio.run(dump())
    doc = {"schema": "d1-omni-swift-ref-media/1", "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "what": __doc__.split("\n\n")[0].strip(), "ship": args.ship, "macos": str(MACOS), "compiled": str(COMPILED),
           **doc, "seconds": time.perf_counter() - t0,
           "references": {k: {"file": f"ref/{v}", "sha256": sha256_file(WORK / "ref" / v)} for k, v in REFERENCES.items()},
           "code_sha256": {f: sha256_file(HERE / f) for f in ("host_dump_media.py", "host.py", "mel_host.py")},
           "environment": {"python": platform.python_version(), "platform": platform.platform(),
                           **{p: metadata.version(p) for p in ("coreai-core", "numpy", "torch", "torchvision", "pillow",
                                                               "soundfile")}}}
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(doc['images'])} images, {len(doc['clips'])} clips, {len(doc['rows'])} rows, {doc['seconds']:.1f} s -> {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
