#!/usr/bin/env python3
"""d1-omni-600M's vision graph (d1_omni_vision.py) against the publisher's tower and projector, and the image rows end
to end through the decision graph.

    python3 conversion/d1_omni/vision_check.py --stage eager                            # venv-d1, CPU fp32
    python3 conversion/d1_omni/vision_check.py --stage runtime --compute gpu \\
        --vision <work>/compiled/vision-wfp16-h16c \\
        --decide 256=<work>/compiled/wfp16-L256-h16c --decide 512=... --decide 2048=...  # shared venv

Images: every image record of the round 5 reference (ref/records_ref_images.json: img_01..03, imgm_01..12) and the
card's cats (ref/records_ref.json, card_cats), each with its reference npz (sha256 checked): the publisher's
preprocess() output, the tower's last_hidden_state, the prefix and (round 5) the resized position tables.

Stage eager (CPU, the fp32 module, venv-d1 = the reference's environment):
  rows  host.image_request_rows on the image file == the reference's ids / markers / positions (P = the prefix rows)
  (i)   host patchify (host.image_crops_inputs on the image file) == preprocess()'s pixel_values, bit for bit; the
        patch masks and grids equal
  (ii)  host.position_embeddings == Siglip2VisionEmbeddings.resize_positional_embeddings (transformers 5.19, the
        publisher's call) bit for bit on every crop, and == the table the reference hooked (round 5 npz);
        host.position_embeddings_numpy (torch's float32 kernel written out: what a Swift host copies) bit for bit as
        well (bar <= 1e-5); a float64 filter is recorded beside it
  (iii) tower: D1Vision.tower on the host's inputs vs last_hidden_state on the crop's patches: max |d|, cos
  (iv)  prefix: D1Vision on every crop, each crop's rows concatenated, vs the reference prefix: max |d|, cos, min row
        cos. Bar: cos >= 0.999999. The launch's max |d| <= 1e-3 is recorded, not the bar: it lies below the fp32
        rounding floor of this tower (the publisher's own fp32 prefix is up to 0.12 from its float64 run, round 5)
  (vi)  float64: D1Vision in float64 vs the publisher's Vision (vision.py on transformers 5.19) in float64 on the same
        inputs, every image: max |d| <= 1e-6 (the same function); recorded beside it, each one's fp32 floor (its fp32
        run vs its float64 run)
  (v)   unshuffle: host.unshuffle_index + the row gather == the publisher's vision.Projector (its reshape / permute;
        linear_1 / linear_2 set to identity, the same gelu between) bit for bit on random tensors, every grid seen
        and some edge grids
  red arm: the 16x16 table not resized (its 256 rows repeated in raster order over the crop) must move the prefix
  out of the bar. Recorded: the absmax of the tower's residual stream (fp16 headroom, memory #19).
Stage runtime (Core AI Python runtime, the AOT .aimodelc, explicit compute unit, never default()):
  (i)   the vision bundle's prefix on the host's inputs vs the reference prefix, per image: max |d|, cos, min row cos;
        drift = max |d| between --repeats calls
  (ii)  end to end: every image row (45 + card_cats) with this prefix -> host.graph_inputs at the row's bucket -> the
        wfp16 decision bundle of that bucket -> marker logits -> probabilities vs the reference (_metrics.SHIP_BAR,
        FACTS §7); beside it, the same rows with the reference's own prefix through the same bundles (what the
        decision graph alone moves), and a control with another image's prefix (the next image's, its rows cut or
        repeated to the row's P), which must FAIL
Cosines are float64 (a float32 reduction over 1e6 elements is not exact to 1e-6).

Output: results/vision_eager.json; results/vision_<precision>_gpu.json and results/vision_e2e_<precision>.json (an
existing file is never replaced: a .runN name is used; --results-prefix is prepended to the runtime names).
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import hf_snapshot, work_path  # noqa: E402

import host  # noqa: E402

WORK = work_path("_d1_omni")
REFERENCE_SHA256 = "e7dba6f44d0452a6aea3e401746d417503056d6f468ff72b56c5511883436c5f"         # round 2
REFERENCE_IMAGES_SHA256 = "681d2d10eef35ecb8980a3caf93d922a77a17e88ee58795468e78b915ac91b08"  # round 5
FIXTURES_IMAGES_SHA256 = "07ea38a2c0f3d8da4afe4dd4ae79e083c46404cb8ab0ba150b9c0a3fca83ae62"
PREFIX_BAR = {"cos": 0.999999}
LAUNCH_MAX_ABS = 1e-3      # the launch's expectation for the fp32 prefix max |d|: recorded (below the fp32 floor)
FLOAT64_BAR = 1e-6         # D1Vision float64 vs the publisher's Vision float64, max |d| of the prefix
NUMPY_POS_BAR = 1e-5
EXTRA_GRIDS = [(2, 2), (2, 8), (8, 2), (2, 32), (32, 2), (24, 42), (42, 24), (32, 32)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = path
    for i in range(2, 100):
        if not out.exists():
            break
        out = path.with_name(f"{path.stem}.run{i}{path.suffix}")
    out.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    return out


def environment(packages) -> dict:
    from importlib import metadata

    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"date": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds"),
            "python": platform.python_version(), "executable": sys.executable, "packages": versions,
            "argv": sys.argv, "pid": os.getpid(), "macos": platform.mac_ver()[0]}


def compare(got, want) -> dict:
    """max |d|, mean |d|, cos and the lowest row cos, in float64."""
    g = np.asarray(got, dtype=np.float64).reshape(len(got), -1)
    w = np.asarray(want, dtype=np.float64).reshape(len(want), -1)
    d = np.abs(g - w)
    rows = (g * w).sum(-1) / (np.linalg.norm(g, axis=-1) * np.linalg.norm(w, axis=-1))
    cos = float(g.ravel() @ w.ravel() / (np.linalg.norm(g) * np.linalg.norm(w)))
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "cos": cos, "min_row_cos": float(rows.min()),
            "min_row": int(rows.argmin()), "rows": int(len(g)), "bit_equal": bool(np.array_equal(g, w))}


def passes(stats: dict, bar: dict = PREFIX_BAR) -> bool:
    return stats["cos"] >= bar["cos"] and stats["max_abs"] <= bar.get("max_abs", float("inf"))


# --------------------------------------------------------------------------- the images and their references
def load_images() -> tuple[list[dict], dict]:
    """[{id, file, public, entry (the reference's image-mode request), npz arrays, npz sha256}] in reference order,
    card_cats last; the reference files checked against their pinned sha256."""
    from _fixtures import fixtures_path  # records.json moved to v3 in round 6; v2 is kept byte for byte beside it

    paths = {"images": WORK / "ref" / "records_ref_images.json", "main": WORK / "ref" / "records_ref.json",
             "fixtures": fixtures_path(FIXTURES_IMAGES_SHA256)}
    pinned = {"images": REFERENCE_IMAGES_SHA256, "main": REFERENCE_SHA256, "fixtures": FIXTURES_IMAGES_SHA256}
    for key, path in paths.items():
        if sha256_file(path) != pinned[key]:
            raise SystemExit(f"{path} is not the pinned file ({pinned[key][:8]}…)")
    refs = {k: json.loads(paths[k].read_text()) for k in ("images", "main")}
    fixtures = {r["id"]: r for r in json.loads(paths["fixtures"].read_text())["records"]}
    out = []
    for key, rid_filter in (("images", None), ("main", "card_cats")):
        ref = refs[key]
        for entry in ref["records"]:
            if entry["mode"] != "image" or (rid_filter and entry["id"] != rid_filter):
                continue
            meta = ref["npz"][entry["id"]]
            npz_path = WORK / "ref" / "npz" / f"{entry['id']}.npz"
            if sha256_file(npz_path) != meta["sha256"]:
                raise SystemExit(f"{npz_path} differs from the reference's npz")
            with np.load(npz_path) as z:
                arrays = {k: z[k] for k in z.files}
            record = fixtures[entry["id"]]
            out.append({"id": entry["id"], "file": WORK / record["media"]["images"][0], "public": record["public"],
                        "record": record, "entry": entry, "npz": arrays,
                        "npz_file": f"ref/npz/{entry['id']}.npz", "npz_sha256": meta["sha256"]})
    info = {k: {"path": str(paths[k]), "sha256": pinned[k]} for k in paths}
    return out, info


def float64_filter(table: np.ndarray, ph: int, pw: int) -> np.ndarray:
    """The antialiased bilinear filter in float64 (PIL's form; the first host form, recorded for comparison)."""
    def weights(n_in, n_out):
        scale = n_in / n_out
        support = max(1.0, scale)
        w = np.zeros((n_out, n_in))
        for i in range(n_out):
            centre = (i + 0.5) * scale
            lo, hi = max(0, int(centre - support + 0.5)), min(n_in, int(centre + support + 0.5))
            taps = np.arange(lo, hi)
            v = np.clip(1.0 - np.abs((taps + 0.5 - centre) / support), 0.0, None)
            w[i, lo:hi] = v / v.sum()
        return w
    t = np.asarray(table, dtype=np.float64)
    out = np.einsum("ia,jb,abd->ijd", weights(t.shape[0], ph), weights(t.shape[1], pw), t)
    return out.reshape(ph * pw, -1)


def snapshot_dir() -> Path:
    return Path(hf_snapshot(host.MODEL_ID, revision=host.MODEL_SHA))


def load_publisher_vision(snapshot: Path):
    """The snapshot's vision.py as a module (its Projector is plain torch; Vision is never built here)."""
    spec = importlib.util.spec_from_file_location("d1_publisher_vision", snapshot / "vision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- stage eager
def eager(args) -> int:
    import torch
    from PIL import Image
    from torch.nn import functional as F
    from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionEmbeddings

    import d1_omni_vision as dv

    t_start = time.perf_counter()
    torch.set_grad_enabled(False)
    snapshot = snapshot_dir()
    if sha256_file(snapshot / "vision.py") != host.COPIED_FROM["vision.py"]:
        raise SystemExit("vision.py is not the pinned revision")
    images, ref_info = load_images()
    model, load_record = dv.load_d1_vision(snapshot, "fp32", verify_sha256=True)
    model64, _ = dv.load_d1_vision(snapshot, "fp32")
    model64.as_float64()
    table = model.position_table.numpy()
    publisher = load_publisher_vision(snapshot)
    config = json.loads((snapshot / "config.json").read_text())
    from safetensors import safe_open
    state = {}
    with safe_open(str(snapshot / "model.safetensors"), framework="pt") as f:  # as from_pretrained loads them
        for key in f.keys():
            if key.startswith("vision."):
                state[key[len("vision."):].replace("tower.vision_model.", "tower.")] = f.get_tensor(key)
    publisher64 = publisher.Vision(config["vision_config"], config["projector_hidden_size"],
                                   config["text_config"]["hidden_size"])
    publisher64.load_state_dict(state, strict=True)
    publisher64 = publisher64.double().eval()
    host_table = host.load_position_table(snapshot / "model.safetensors")
    if not np.array_equal(table, host_table):
        raise SystemExit("host.load_position_table differs from the module's table")
    flat = model.position_table.reshape(-1, table.shape[-1]).contiguous()  # the checkpoint's [256, 768] layout
    tok = host.RawTokenizer(snapshot / "tokenizer.json")
    host.check_token_ids(tok)

    def publisher_positions(ph: int, pw: int) -> np.ndarray:
        pe = flat.reshape(16, 16, -1)  # position_embedding.weight.reshape(16, 16, -1), as Siglip2VisionEmbeddings does
        out = Siglip2VisionEmbeddings.resize_positional_embeddings(pe, torch.tensor([[ph, pw]]), max_length=1024)
        return out[0, :ph * pw].numpy()

    def run(crop: dict, pos: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
        t = {k: torch.from_numpy(np.ascontiguousarray(crop[k])) for k in host.VISION_INPUT_NAMES}
        if pos is not None:
            t["pos_embed"] = torch.from_numpy(np.ascontiguousarray(pos))
        hidden = model.tower(t["pixel_values"], t["pos_embed"], t["patch_mask"])
        prefix = model.project(hidden, t["unshuffle_index"])
        # the residual stream's absmax over the real patches (the same loop as D1Vision.tower, checked equal)
        n = int(crop["patch_mask"].sum())
        mask = ((1.0 - t["patch_mask"]) * dv.MASK_VALUE).reshape(1, 1, 1, dv.MAX_PATCHES)
        x = model._lin(t["pixel_values"], model.patch_embedding) + t["pos_embed"]
        peak = float(x[0, :n].abs().max())
        for layer in model.layers:
            x = x + model._attention(layer.self_attn, model._ln(layer.layer_norm1, x), mask)
            peak = max(peak, float(x[0, :n].abs().max()))
            x = x + model._lin(F.gelu(model._lin(model._ln(layer.layer_norm2, x), layer.mlp.fc1), approximate="tanh"),
                               layer.mlp.fc2)
            peak = max(peak, float(x[0, :n].abs().max()))
        assert torch.equal(model._ln(model.post_layernorm, x), hidden)
        return hidden[0].numpy(), prefix[0].numpy(), peak

    records, worst = [], {}
    grids_seen = set()
    for img in images:
        t0 = time.perf_counter()
        z = img["npz"]
        with Image.open(img["file"]) as pil:
            pil.load()
            crops = host.image_crops_inputs(pil, table)
            rows = host.image_request_rows(tok, img["record"]["request"]["state"], img["record"]["request"]["questions"],
                                           [pil])
            prefix_publisher64 = publisher64([pil])[0].numpy()
        entry = img["entry"]
        rows_equal = [r.qid == q["qid"] and r.ids == q["ids"] and r.markers == q["markers"] and
                      r.positions == q["positions"] and r.prefix_len == entry["prefix"] for r, q in
                      zip(rows, entry["questions"], strict=True)]
        assert len(crops) == z["pixel_values"].shape[0], (img["id"], len(crops))
        crop_records, prefix_parts, red_parts, peaks = [], [], [], []
        for k, crop in enumerate(crops):
            ph, pw = crop["grid"]
            n = ph * pw
            grids_seen.add((ph, pw))
            assert [ph, pw] == z["spatial_shapes"][k].tolist(), (img["id"], k)
            pos_host = crop["pos_embed"][0, :n]
            pos_pub = publisher_positions(ph, pw)
            pos_np = host.position_embeddings_numpy(table, ph, pw)
            pos_f64 = float64_filter(table, ph, pw)
            rec = {"crop": k, "grid": [ph, pw], "tokens": crop["tokens"],
                   "pixel_values_bit_equal": bool(np.array_equal(crop["pixel_values"][0], z["pixel_values"][k])),
                   "patch_mask_equal": bool(np.array_equal(crop["patch_mask"][0] > 0, z["pixel_attention_mask"][k] > 0)),
                   "pos_bit_equal_publisher_call": bool(np.array_equal(pos_host, pos_pub)),
                   "pos_numpy_bit_equal": bool(np.array_equal(pos_np, pos_pub)),
                   "pos_numpy_max_abs": float(np.max(np.abs(pos_np.astype(np.float64) - pos_pub))),
                   "pos_float64_filter_max_abs": float(np.max(np.abs(pos_f64 - pos_pub)))}
            if "pos_resized" in z:
                rec["pos_bit_equal_hooked"] = bool(np.array_equal(pos_host, z["pos_resized"][z["pos_index"][k]][:n]))
            hidden, prefix, peak = run(crop)
            peaks.append(peak)
            rec["tower"] = compare(hidden[:n], z["tower_last_hidden_state"][k, :n])
            rec["tower_absmax"] = peak
            prefix_parts.append(prefix[:crop["tokens"]])
            red = np.resize(table.reshape(-1, table.shape[-1]), (n, table.shape[-1]))  # not resized: raster rows
            red_pos = np.zeros_like(crop["pos_embed"])
            red_pos[0, :n] = red
            red_parts.append(run(crop, red_pos)[1][:crop["tokens"]])
            crop_records.append(rec)
        prefix = np.concatenate(prefix_parts)
        assert prefix.shape == z["prefix"].shape, (img["id"], prefix.shape, z["prefix"].shape)
        stats = compare(prefix, z["prefix"])
        parts64 = []
        for crop in crops:
            t = {k: torch.from_numpy(np.ascontiguousarray(crop[k])) for k in host.VISION_INPUT_NAMES}
            parts64.append(model64(**t)[0, :crop["tokens"]].numpy())
        prefix64 = np.concatenate(parts64)
        float64 = {"mine64_vs_publisher64": compare(prefix64, prefix_publisher64),
                   "floor_publisher32_vs_publisher64": compare(z["prefix"], prefix_publisher64),
                   "floor_mine32_vs_mine64": compare(prefix, prefix64)}
        red_stats = compare(np.concatenate(red_parts), z["prefix"])
        record = {"id": img["id"], "public": img["public"], "file": str(img["file"].relative_to(WORK)),
                  "px": list(Image.open(img["file"]).size), "crops": len(crops), "prefix_rows": int(prefix.shape[0]),
                  "rows_equal_reference": f"{sum(rows_equal)}/{len(rows_equal)}",
                  "prefix": stats, "prefix_pass": passes(stats),
                  "prefix_max_abs_le_launch_1e-3": stats["max_abs"] <= LAUNCH_MAX_ABS,
                  "float64": float64, "float64_pass": float64["mine64_vs_publisher64"]["max_abs"] <= FLOAT64_BAR,
                  "red_arm_prefix": red_stats,
                  "red_arm_caught": not passes(red_stats), "tower_absmax": max(peaks),
                  "npz": img["npz_file"], "npz_sha256": img["npz_sha256"], "seconds": time.perf_counter() - t0,
                  "crops_detail": crop_records}
        records.append(record)
        print(f"{img['id']}: {len(crops)} crops, P {prefix.shape[0]}, rows {record['rows_equal_reference']}, pixel "
              f"{sum(c['pixel_values_bit_equal'] for c in crop_records)}/{len(crops)} pos "
              f"{sum(c['pos_bit_equal_publisher_call'] for c in crop_records)}/{len(crops)} numpy "
              f"{max(c['pos_numpy_max_abs'] for c in crop_records):.2e}, tower cos "
              f"{min(c['tower']['cos'] for c in crop_records):.9f} max {max(c['tower']['max_abs'] for c in crop_records):.2e}, "
              f"prefix cos {stats['cos']:.9f} max {stats['max_abs']:.2e} min-row {stats['min_row_cos']:.9f}, float64 "
              f"{float64['mine64_vs_publisher64']['max_abs']:.1e} (floors {float64['floor_publisher32_vs_publisher64']['max_abs']:.2e}"
              f" / {float64['floor_mine32_vs_mine64']['max_abs']:.2e}), red cos "
              f"{red_stats['cos']:.6f}, absmax {record['tower_absmax']:.1f} ({record['seconds']:.1f} s)", flush=True)

    # (v) the unshuffle order against the publisher's Projector
    publisher = load_publisher_vision(snapshot)
    channels = 8
    projector = publisher.Projector(channels, 4 * channels, 4 * channels)
    with torch.no_grad():
        for linear in (projector.linear_1, projector.linear_2):
            linear.weight.copy_(torch.eye(4 * channels))
            linear.bias.zero_()
    generator = torch.Generator().manual_seed(0)
    unshuffle = []
    for ph, pw in sorted(grids_seen | set(EXTRA_GRIDS)):
        x = torch.randn(1, ph, pw, channels, generator=generator)
        theirs = projector(x)[0]
        tokens = (ph // 2) * (pw // 2)
        index = torch.from_numpy(host.unshuffle_index(ph, pw)[:tokens].astype(np.int64))
        gathered = x.reshape(ph * pw, channels)[index.reshape(-1)].reshape(1, tokens, 4 * channels)
        mine = projector.linear_2(F.gelu(projector.linear_1(gathered)))[0]
        unshuffle.append({"grid": [ph, pw], "tokens": tokens, "bit_equal": bool(torch.equal(mine, theirs)),
                          "seen_in_images": (ph, pw) in grids_seen})

    crops_all = [c for r in records for c in r["crops_detail"]]
    summary = {
        "images": len(records), "crops": len(crops_all), "grids": sorted(list(g) for g in grids_seen),
        "rows_equal_reference": f"{sum(int(r['rows_equal_reference'].split('/')[0]) for r in records)}/"
                                f"{sum(int(r['rows_equal_reference'].split('/')[1]) for r in records)}",
        "pixel_values_bit_equal": f"{sum(c['pixel_values_bit_equal'] for c in crops_all)}/{len(crops_all)}",
        "patch_mask_equal": f"{sum(c['patch_mask_equal'] for c in crops_all)}/{len(crops_all)}",
        "pos_bit_equal_publisher_call": f"{sum(c['pos_bit_equal_publisher_call'] for c in crops_all)}/{len(crops_all)}",
        "pos_bit_equal_hooked": f"{sum(c.get('pos_bit_equal_hooked', False) for c in crops_all)}/"
                                f"{sum('pos_bit_equal_hooked' in c for c in crops_all)}",
        "pos_numpy_bit_equal": f"{sum(c['pos_numpy_bit_equal'] for c in crops_all)}/{len(crops_all)}",
        "pos_numpy_max_abs": max(c["pos_numpy_max_abs"] for c in crops_all),
        "pos_float64_filter_max_abs": max(c["pos_float64_filter_max_abs"] for c in crops_all),
        "tower_min_cos": min(c["tower"]["cos"] for c in crops_all),
        "tower_max_abs": max(c["tower"]["max_abs"] for c in crops_all),
        "prefix_min_cos": min(r["prefix"]["cos"] for r in records),
        "prefix_min_row_cos": min(r["prefix"]["min_row_cos"] for r in records),
        "prefix_max_abs": max(r["prefix"]["max_abs"] for r in records),
        "prefix_worst_image": min(records, key=lambda r: r["prefix"]["cos"])["id"],
        "prefix_max_abs_le_launch_1e-3": f"{sum(r['prefix_max_abs_le_launch_1e-3'] for r in records)}/{len(records)}",
        "float64_mine_vs_publisher_max_abs": max(r["float64"]["mine64_vs_publisher64"]["max_abs"] for r in records),
        "fp32_floor_publisher_max_abs": max(r["float64"]["floor_publisher32_vs_publisher64"]["max_abs"] for r in records),
        "fp32_floor_mine_max_abs": max(r["float64"]["floor_mine32_vs_mine64"]["max_abs"] for r in records),
        "unshuffle_bit_equal": f"{sum(u['bit_equal'] for u in unshuffle)}/{len(unshuffle)}",
        "red_arm_caught": f"{sum(r['red_arm_caught'] for r in records)}/{len(records)}",
        "red_arm_max_cos": max(r["red_arm_prefix"]["cos"] for r in records),
        "red_arm_min_max_abs": min(r["red_arm_prefix"]["max_abs"] for r in records),
        "tower_absmax": max(r["tower_absmax"] for r in records),
        "tower_absmax_fp16_headroom": 65504.0 / max(r["tower_absmax"] for r in records),
    }
    verdict = {
        "rows": all(r["rows_equal_reference"].split("/")[0] == r["rows_equal_reference"].split("/")[1] for r in records),
        "pixel_values": all(c["pixel_values_bit_equal"] and c["patch_mask_equal"] for c in crops_all),
        "pos_publisher_call": all(c["pos_bit_equal_publisher_call"] for c in crops_all),
        "pos_hooked": all(c.get("pos_bit_equal_hooked", True) for c in crops_all),
        "pos_numpy": summary["pos_numpy_max_abs"] <= NUMPY_POS_BAR,
        "prefix_cos": all(r["prefix_pass"] for r in records),
        "float64_same_function": all(r["float64_pass"] for r in records),
        "unshuffle": all(u["bit_equal"] for u in unshuffle),
        "red_arm": all(r["red_arm_caught"] for r in records),
    }
    status = "PASS" if all(verdict.values()) else "FAIL"
    doc = {"status": status, "verdict": verdict, "stage": "eager (CPU fp32, D1Vision vs the publisher's reference)",
           "bars": {"prefix": PREFIX_BAR, "float64_max_abs": FLOAT64_BAR, "pos_numpy_max_abs": NUMPY_POS_BAR,
                    "launch_prefix_max_abs_recorded": LAUNCH_MAX_ABS,
                    "why_max_abs_is_recorded": "the fp32 rounding floor of this tower is above it: the publisher's own "
                                               "fp32 prefix vs its float64 run (fp32_floor_publisher_max_abs)",
                    "bit_equal": "rows, pixel_values, patch masks, position tables (publisher call and hooked), "
                                 "unshuffle order"},
           "reference": ref_info, "weights": load_record, "summary": summary, "unshuffle": unshuffle,
           "seconds": time.perf_counter() - t_start,
           "code_sha256": {f: sha256_file(HERE / f) for f in ("vision_check.py", "d1_omni_vision.py", "host.py")},
           "environment": environment(("torch", "torchvision", "transformers", "numpy", "pillow", "safetensors")),
           "images": records}
    out = write_new(WORK / "results" / "vision_eager.json", doc)
    print(json.dumps({"status": status, "verdict": verdict, **summary}, indent=1))
    print("->", out)
    return 0 if status == "PASS" else 1


# --------------------------------------------------------------------------- stage runtime
def runtime(args) -> int:
    import coreai.runtime as rt
    from PIL import Image

    from _metrics import SHIP_BAR, row_record, summarize, wrong_pairing

    t_start = time.perf_counter()
    snapshot = snapshot_dir()
    images, ref_info = load_images()
    table = host.load_position_table(snapshot / "model.safetensors")

    def options(compute: str):
        if compute == "cpu_only":
            return rt.SpecializationOptions.cpu_only()
        kind = rt.ComputeUnitKind.gpu() if compute == "gpu" else rt.ComputeUnitKind.neural_engine()
        return rt.SpecializationOptions.from_preferred_compute_unit_kind(kind)

    def compiled(directory: Path) -> tuple[Path, dict]:
        aot = json.loads((directory / "provenance" / "aot-manifest.json").read_text())
        if aot.get("status") != "COMPILED" or not aot.get("aimodelc_kept", True):
            raise SystemExit(f"{directory}: no compiled bundle")
        aimodelc = directory / aot["aimodelc"]
        for item in aot["files"]:
            if sha256_file(aimodelc / item["path"]) != item["sha256"]:
                raise SystemExit(f"{aimodelc / item['path']} changed since its compile")
        return aimodelc, aot

    vision_dir = args.vision.resolve()
    vision_aimodelc, vision_aot = compiled(vision_dir)
    precision = vision_aot["precision"]
    decide = {}
    for spec in args.decide:
        length, directory = spec.split("=", 1)
        decide[int(length)] = compiled(Path(directory).resolve())

    async def main_async():
        opts, dopts = options(args.compute), options(args.decide_compute or args.compute)
        t0 = time.perf_counter()
        vmodel = await rt.AIModel.load(vision_aimodelc, opts)
        vfn = vmodel.load_function("main")
        load_s = {"vision": time.perf_counter() - t0}
        dfns = {}
        for length, (aimodelc, aot) in sorted(decide.items()):
            t0 = time.perf_counter()
            m = await rt.AIModel.load(aimodelc, dopts)
            dfns[length] = (m, m.load_function("main"), aot)
            load_s[f"decide_L{length}"] = time.perf_counter() - t0

        async def vision_call(inputs: dict) -> np.ndarray:
            out = await vfn({k: rt.NDArray(np.ascontiguousarray(inputs[k])) for k in host.VISION_INPUT_NAMES})
            return np.array(out["prefix"].numpy(), copy=True)

        # (i) the prefix of every image, --repeats times
        prefixes, image_records = {}, []
        for img in images:
            z = img["npz"]
            with Image.open(img["file"]) as pil:
                pil.load()
                crops = host.image_crops_inputs(pil, table)
            same_pixels = all(np.array_equal(c["pixel_values"][0], z["pixel_values"][k]) for k, c in enumerate(crops))
            runs = []
            for _ in range(args.repeats):
                parts = [(await vision_call(c)).reshape(host.VISION_TOKENS, -1)[:c["tokens"]] for c in crops]
                runs.append(np.concatenate(parts))
            drift = max(float(np.max(np.abs(r.astype(np.float64) - runs[0]))) for r in runs[1:])
            stats = compare(runs[0], z["prefix"])
            prefixes[img["id"]] = runs[0]
            image_records.append({"id": img["id"], "public": img["public"], "crops": len(crops),
                                  "prefix_rows": int(runs[0].shape[0]), "host_pixels_equal_reference": same_pixels,
                                  "prefix": stats, "prefix_pass": passes(stats), "drift": drift,
                                  "finite": bool(np.isfinite(runs[0]).all())})
            print(f"{img['id']}: P {runs[0].shape[0]} cos {stats['cos']:.9f} max {stats['max_abs']:.3e} min-row "
                  f"{stats['min_row_cos']:.9f} drift {drift:.1e}", flush=True)

        # (ii) end to end through the decision graph of each row's bucket
        rows, hrows, own, theirs = [], [], [], []
        for img in images:
            entry = img["entry"]
            for q in entry["questions"]:
                question = host.as_question(img["record"]["request"]["questions"][q["qid"]])
                hrow = host.Row(question=question, ids=q["ids"], markers=q["markers"], calibrate=q["calibrate"],
                                prefix_len=q["prefix"], mode="image", max_len=0, qid=q["qid"])
                assert hrow.positions == q["positions"]
                rows.append({"id": img["id"], "qid": q["qid"], "mode": "image", "source": entry["source"],
                             "type": q["type"], "K": q["K"], "positions": q["positions"], "near_tie": q["near_tie"],
                             "top2_margin": q["top2_margin"], "argmax_index": q["argmax_index"],
                             "calibrate": q["calibrate"], "bucket": q["bucket"],
                             "question": {"type": question.type, "instructions": question.instructions,
                                          "criteria": question.criteria},
                             "oracle": {"logits_raw": q["logits_raw"], "probs": q["probs"]}})
                hrows.append(hrow)
                own.append(prefixes[img["id"]])
                theirs.append(np.asarray(img["npz"]["prefix"], dtype=np.float32))
        # the control: the next image's prefix (cyclic), its rows cut or repeated to the row's P
        order = [img["id"] for img in images]
        control = []
        for hrow, row in zip(hrows, rows):
            donor = order[(order.index(row["id"]) + 1) % len(order)]
            rows_donor = prefixes[donor]
            control.append((donor, np.resize(rows_donor, (hrow.prefix_len, rows_donor.shape[1])).astype(np.float32)))

        async def decide_call(hrow, prefix, length):
            inputs, markers = host.graph_inputs(hrow, length, prefix)
            fn = dfns[length][1]
            out = await fn({k: rt.NDArray(inputs[k]) for k in ("input_ids", "prefix_embeds", "pad_mask", "prefix_mask",
                                                                "keep_right", "qtype_onehot")})
            scores = np.array(out["scores"].numpy(), copy=True).reshape(-1)
            return scores[markers], scores[:hrow.positions]

        arms = {"vision_bundle_prefix": [], "reference_prefix": [], "control_other_image_prefix": []}
        drifts = []
        for hrow, row, mine, ref, (donor, wrong) in zip(hrows, rows, own, theirs, control):
            length = host.bucket_for(hrow.positions)
            if length not in dfns:
                raise SystemExit(f"no decision bundle for bucket {length} (row {row['id']}/{row['qid']})")
            outs = [await decide_call(hrow, mine, length) for _ in range(args.repeats)]
            drift = max(float(np.max(np.abs(o[1].astype(np.float64) - outs[0][1]))) for o in outs[1:])
            drifts.append(drift)
            rec = row_record(row, outs[0][0])
            rec.update(bucket=length, repeat_drift_scores=drift)
            arms["vision_bundle_prefix"].append(rec)
            rec = row_record(row, (await decide_call(hrow, ref, length))[0])
            rec.update(bucket=length)
            arms["reference_prefix"].append(rec)
            rec = row_record(row, (await decide_call(hrow, wrong, length))[0])
            rec.update(bucket=length, donor=donor)
            arms["control_other_image_prefix"].append(rec)
        assert vmodel is not None and dfns
        return load_s, image_records, rows, arms, drifts

    load_s, image_records, rows, arms, drifts = asyncio.run(main_async())
    summaries = {name: summarize(records) for name, records in arms.items()}
    for name, records in arms.items():
        summaries[name]["by_bucket"] = {str(b): {k: s[k] for k in ("rows", "argmax_equal", "max_abs_dp",
                                                                     "mean_row_max_abs_dp", "max_abs_dlogit")}
                                        for b in sorted({r["bucket"] for r in records})
                                        if (s := summarize([r for r in records if r["bucket"] == b]))}
    shipped = summaries["vision_bundle_prefix"]["ship_bar"]["status"]
    control = summaries["control_other_image_prefix"]["ship_bar"]["status"]
    pairing = wrong_pairing(rows, arms["vision_bundle_prefix"])
    prefix_summary = {
        "images": len(image_records), "min_cos": min(r["prefix"]["cos"] for r in image_records),
        "min_row_cos": min(r["prefix"]["min_row_cos"] for r in image_records),
        "max_abs": max(r["prefix"]["max_abs"] for r in image_records),
        "worst_image": min(image_records, key=lambda r: r["prefix"]["cos"])["id"],
        "drift_max": max(r["drift"] for r in image_records), "repeats": args.repeats,
        "host_pixels_equal_reference": f"{sum(r['host_pixels_equal_reference'] for r in image_records)}/"
                                       f"{len(image_records)}",
        "bar_eager": PREFIX_BAR, "eager_bar_pass": f"{sum(r['prefix_pass'] for r in image_records)}/{len(image_records)}"}
    common = {"stage": "runtime (Mac, Core AI Python runtime, AOT .aimodelc)", "compute": args.compute,
              "decide_compute": args.decide_compute or args.compute,
              "precision": precision, "vision": {"folder": str(vision_dir), "aimodelc": vision_aimodelc.name,
                                                 "bytes": vision_aot["bytes"],
                                                 "resources_bin_bytes": vision_aot["resources_bin_bytes"],
                                                 "hashes": vision_aot["hashes"]},
              "load_s": load_s, "reference": ref_info,
              "code_sha256": {f: sha256_file(HERE / f) for f in ("vision_check.py", "host.py", "_metrics.py")},
              "environment": environment(("coreai-core", "numpy", "torch", "torchvision", "pillow"))}
    prefix_doc = {**common, "status": "PASS" if all(r["finite"] for r in image_records) and prefix_summary["drift_max"]
                  == 0.0 else "RECORDED", "summary": prefix_summary, "images": image_records,
                  "note": "the bar for the shipped forms is end to end (vision_e2e_<p>.json); the eager prefix bar "
                          "is shown for reference"}
    e2e_doc = {**common, "status": "PASS" if shipped == "PASS" and control == "FAIL" else "FAIL",
               "bar": SHIP_BAR, "decide": {str(k): {"folder": str(Path(v[0]).parent), "aimodelc": Path(v[0]).name,
                                                    "precision": v[1]["precision"], "seq_len": v[1]["seq_len"],
                                                    "hashes": v[1]["hashes"]} for k, v in sorted(decide.items())},
               "rows": len(rows), "repeat_drift_max": max(drifts), "summaries": summaries,
               "control_must_fail": control, "wrong_pairing_oracle_swap": pairing,
               "arms": arms, "seconds": time.perf_counter() - t_start}
    p_out = write_new(WORK / "results" / f"{args.results_prefix}vision_{precision}_{args.compute}.json", prefix_doc)
    e_out = write_new(WORK / "results" / f"{args.results_prefix}vision_e2e_{precision}.json", e2e_doc)
    s = summaries["vision_bundle_prefix"]
    r = summaries["reference_prefix"]
    print(json.dumps({"prefix": prefix_summary, "e2e": {
        "status": e2e_doc["status"], "rows": len(rows), "argmax": f"{s['argmax_equal']}/{s['rows']}",
        "non_near_tie": s["non_near_tie"], "max_abs_dp": s["max_abs_dp"], "mean": s["mean_row_max_abs_dp"],
        "max_abs_dlogit": s["max_abs_dlogit"], "reference_prefix_max_abs_dp": r["max_abs_dp"],
        "reference_prefix_mean": r["mean_row_max_abs_dp"], "control": control,
        "control_max_abs_dp": summaries["control_other_image_prefix"]["max_abs_dp"],
        "wrong_pairing": pairing["status"], "drift": max(drifts), "load_s": load_s}}, indent=1))
    print("->", p_out, e_out)
    return 0 if e2e_doc["status"] == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["eager", "runtime"], required=True)
    ap.add_argument("--compute", choices=["cpu_only", "gpu", "neural_engine"], default="gpu")
    ap.add_argument("--decide-compute", choices=["cpu_only", "gpu", "neural_engine"], default=None,
                    help="runtime: the decision bundles' compute unit (default: --compute); an ANE probe of the %s "
                         "bundle keeps the decision graph on the GPU" % "vision")
    ap.add_argument("--vision", type=Path, help="runtime: <work>/compiled/vision-<precision>-h16c")
    ap.add_argument("--decide", action="append", default=[], help="runtime: <L>=<work>/compiled/<dir> (repeat)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--results-prefix", default="", help="runtime: prepended to the results files' names "
                    "(round 10: ship_runtime_ for the stripped bundles)")
    args = ap.parse_args()
    if args.stage == "eager":
        return eager(args)
    if not args.vision or not args.decide:
        ap.error("--stage runtime needs --vision and --decide")
    if args.repeats < 2:
        ap.error("at least 2 repeats")
    return runtime(args)


if __name__ == "__main__":
    raise SystemExit(main())
