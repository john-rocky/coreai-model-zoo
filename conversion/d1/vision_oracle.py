#!/usr/bin/env python3
"""Tower oracle: transformers 5.19's own LFM2-VL image path per crop, on d1-3B's checkpoint (or the toy snapshot).

The model (default): LiquidAI/d1-3B @ da1fe36a's `model.safetensors` is loaded by transformers' own
`Lfm2VlForConditionalGeneration.from_pretrained(<snapshot>, dtype=float32)` on the CPU (the whole model: transformers
has no loader for the tower alone; the language model is loaded and never runs). Its load report is kept: transformers'
missing / unexpected / mismatched keys by part, and each of the checkpoint's 441 tower keys
(`model.vision_tower.vision_model.*` 437 + `model.multi_modal_projector.*` 4) against the loaded tensor of its
transformers 5 name (the `vision_model.` segment dropped: lfm2_vl_tower.hf_name's rule) bit for bit (BF16 widened to
fp32 is exact). transformers >= 5 only: 4.57 adds a projector LayerNorm this checkpoint does not have
(knowledge/lfm2.5-vl-port.md).

The toy (`--toy`, round 3b): the toy snapshot (`lfm2_vl_tower.py write-toy`: config.json + model.safetensors laid out
like d1-3B's checkpoint, the tensors BF16 under `model.vision_tower.vision_model.*` / `model.multi_modal_projector.*`,
+ name_table.json), loaded by the same class, every vision and projector tensor checked bit for bit against the file
under the name table's transformers name. The toy file has no language model: that part stays at its init and never
runs.

Inputs are the processor's, through `vision_host.py` (bit-equal to transformers 5.19's processor on these pictures,
round 2b): every crop of the 12 fixture pictures (`fixtures/image_records.json`) and of the 6 random pictures of
test_vision_host.py (numpy default_rng(0) integers in its size order; sha256 checked against vision_host.json). Per
crop, as the processor lays it out — pixel_values [1, 1024, 768] zero-padded, pixel_attention_mask [1, 1024],
spatial_shapes [[h, w]] — through `model.model.get_image_features` -> the crop's image_embeds [h w / 4, text_hidden]
fp32. Per picture, its crops in one call cut to their largest real patch count (the provider's runner,
`_image_inputs`) as well, compared with the per-crop rows. The first crop runs again after the loop (bit-equal).

Each crop's npz also holds the host's four tower inputs (`vision_host.tower_inputs` with the loaded model's position
table [256, d] fp32 = the table export_vision.py ships as host/position_embedding.safetensors; its sha256 is in the
record): the contract `gate_tower.py` feeds the graph, plus the mask, the grid and the crop's place. The transcript
records the load report, the attention implementation, per crop the rows' sha256, absmax, seconds and the
batch-vs-crop difference.

    cd conversion/d1
    source $ZOO_WORK_ROOT/_d1_3b/venv-oracle/bin/activate                  # transformers 5.19
    HF_HUB_OFFLINE=1 python vision_oracle.py                               # the model, 1 thread
    python vision_oracle.py --toy                                          # the toy (round 3b), 4 threads

-> the model: $ZOO_WORK_ROOT/_d1_3b/oracle/images/<picture>/<crop>.npz, oracle/images/oracle.json;
   the toy:   $ZOO_WORK_ROOT/_d1_3b/oracle_toy_vision/<picture>/<crop>.npz, oracle_toy_vision/oracle.json
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

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402

import vision_host as vh  # noqa: E402
from _paths import hf_snapshot, work_path  # noqa: E402

LANE = work_path("_d1_3b")
OUT = LANE / "oracle" / "images"
TOY_OUT = LANE / "oracle_toy_vision"
MODEL = {"hf_id": "LiquidAI/d1-3B", "revision": "da1fe36a861f24690f27f622dca1d8688503d113"}
RANDOM_SIZES = [(384, 384), (384, 256), (640, 480), (1024, 768), (1600, 1200), (2048, 1536)]   # test_vision_host.py
RANDOM_SEED = 0
VISION_PREFIX = "model.vision_tower.vision_model."     # the checkpoint's names (tf 4.57, as saved)
PROJECTOR_PREFIX = "model.multi_modal_projector."
POSITION_HF = "model.vision_tower.embeddings.position_embedding.weight"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_name(key: str) -> str:
    """A checkpoint key -> transformers 5.19's Lfm2VlForConditionalGeneration name (lfm2_vl_tower.hf_name: the
    `vision_model.` segment dropped by Siglip2VisionModel's PrefixChange, nothing else renamed)."""
    return key.replace("model.vision_tower.vision_model.", "model.vision_tower.", 1)


def crop_name(ci: int, c: vh.Crop) -> str:
    return f"crop{ci:02d}_{c.kind}" + (f"_r{c.row}c{c.col}" if c.kind == "tile" else "")


def pictures(fixtures: Path, host_json: Path) -> list[tuple[str, str, np.ndarray, dict]]:
    """(id, kind, capped RGB, provenance) for the 12 fixture pictures and the 6 random ones, in that order."""
    fx = json.loads(fixtures.read_text())
    out = []
    for r in fx["records"]:
        if r["source"] != "own_image":
            continue
        path = fixtures.parent / r["images"][0]
        out.append((r["id"], "fixture", vh.cap_pixels(vh.to_rgb(path)),
                    {"path": str(path), "png_sha256": sha256_file(path)}))
    want = {r["id"]: r.get("rgb_sha256") for r in json.loads(host_json.read_text())["records"] if r["kind"] == "random"}
    rng = np.random.default_rng(RANDOM_SEED)
    for w, h in RANDOM_SIZES:
        arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        rid = f"random_{w}x{h}"
        got = sha256_bytes(arr.tobytes())
        if want.get(rid) != got:
            raise SystemExit(f"{rid}: rgb sha256 {got} != vision_host.json's {want.get(rid)}")
        out.append((rid, "random", vh.cap_pixels(arr), {"rgb_sha256": got, "seed": RANDOM_SEED}))
    return out


def load_toy(snap: Path):
    """transformers' own loader on the toy snapshot, and the name check against the file."""
    import torch
    from safetensors import safe_open
    from transformers import Lfm2VlForConditionalGeneration

    model = Lfm2VlForConditionalGeneration.from_pretrained(str(snap), dtype=torch.float32).eval()
    names = json.loads((snap / "name_table.json").read_text())
    sd = model.state_dict()
    rows, bad = [], []
    with safe_open(str(snap / "model.safetensors"), framework="pt") as f:
        file_keys = sorted(f.keys())
        for row in names["keys"]:
            ck, hn = row["checkpoint"], row["hf_Lfm2VlForConditionalGeneration"]
            t = f.get_tensor(ck)
            have = sd.get(hn)
            same = have is not None and have.dtype == torch.float32 and torch.equal(have, t.float())
            rows.append({"checkpoint": ck, "hf": hn, "file_dtype": str(t.dtype).replace("torch.", ""), "bit_equal": same})
            if not same:
                bad.append(ck)
    vision_keys = sorted(k for k in sd if k.startswith(("model.vision_tower.", "model.multi_modal_projector.")))
    report = {"loader": "transformers Lfm2VlForConditionalGeneration.from_pretrained(dtype=float32)",
              "file_keys": len(file_keys), "name_table_keys": len(names["keys"]),
              "vision_projector_tensors_in_model": len(vision_keys),
              "model_tensors_not_in_name_table": sorted(set(vision_keys) - {r["hf"] for r in rows}),
              "bit_equal": sum(r["bit_equal"] for r in rows), "not_equal": bad,
              "attn_implementation": {"vision": model.config.vision_config._attn_implementation,
                                      "model": model.config._attn_implementation},
              "position_table_key": POSITION_HF}
    if bad or report["model_tensors_not_in_name_table"] or len(rows) != len(file_keys):
        raise SystemExit(f"the loaded tensors differ from the file: {json.dumps(report)[:2000]}")
    table = sd[POSITION_HF].numpy().astype(np.float32)
    snapshot = {"path": str(snap), "config_sha256": sha256_file(snap / "config.json"),
                "model_safetensors_sha256": sha256_file(snap / "model.safetensors"),
                "name_table_sha256": sha256_file(snap / "name_table.json")}
    return model, table, report, snapshot


def _part(k: str) -> str:
    return ("vision" if k.startswith("model.vision_tower.") else "projector" if k.startswith(PROJECTOR_PREFIX)
            else "language" if "language_model" in k else "other")


def load_checkpoint(snap: Path):
    """transformers' own loader on d1-3B's snapshot (fp32, CPU) and the load report (module docstring)."""
    import torch
    from safetensors import safe_open
    from transformers import Lfm2VlForConditionalGeneration

    ck = snap / "model.safetensors"
    t0 = time.perf_counter()
    model, info = Lfm2VlForConditionalGeneration.from_pretrained(str(snap), dtype=torch.float32, output_loading_info=True)
    model.eval()
    load_s = time.perf_counter() - t0
    sd = model.state_dict()

    def parts(keys) -> dict:
        out: dict = {}
        for k in keys:
            out.setdefault(_part(str(k)), []).append(str(k))
        return {p: sorted(v) for p, v in sorted(out.items())}

    rows, bad = [], []
    with safe_open(str(ck), framework="pt", device="cpu") as f:
        keys = list(f.keys())  # noqa: SIM118
        tower = [k for k in keys if k.startswith((VISION_PREFIX, PROJECTOR_PREFIX))]
        for k in tower:
            hn = hf_name(k)
            t = f.get_tensor(k)
            have = sd.get(hn)
            same = (have is not None and have.dtype == torch.float32 and tuple(have.shape) == tuple(t.shape)
                    and torch.equal(have, t.float()))
            rows.append({"checkpoint": k, "hf": hn, "shape": list(t.shape), "file_dtype": str(t.dtype).replace("torch.", ""),
                         "bit_equal": bool(same)})
            if not same:
                bad.append(k)
    model_tower = sorted(k for k in sd if k.startswith(("model.vision_tower.", "model.multi_modal_projector.")))
    missing = sorted(str(x) for x in info.get("missing_keys") or [])
    unexpected = sorted(str(x) for x in info.get("unexpected_keys") or [])
    mismatched = sorted(str(x) for x in info.get("mismatched_keys") or [])
    by_part = {p: sum(1 for k in keys if _part(hf_name(k)) == p) for p in ("vision", "projector", "language", "other")}
    report = {
        "loader": "transformers Lfm2VlForConditionalGeneration.from_pretrained(<snapshot>, dtype=float32, "
                  "output_loading_info=True)",
        "load_seconds": round(load_s, 1),
        "transformers": {"missing_keys": parts(missing), "unexpected_keys": parts(unexpected),
                         "mismatched_keys": mismatched, "error_msgs": [str(e) for e in info.get("error_msgs") or []],
                         "n_missing": len(missing), "n_unexpected": len(unexpected), "n_mismatched": len(mismatched),
                         "n_missing_tower": sum(1 for k in missing if _part(k) in ("vision", "projector")),
                         "n_unexpected_tower": sum(1 for k in unexpected if _part(k) in ("vision", "projector"))},
        "checkpoint": {"file": str(ck), "keys": len(keys), "keys_by_part": by_part},
        "tower_keys": len(rows), "tower_keys_by_part": {p: sum(1 for r in rows if _part(r["hf"]) == p)
                                                        for p in ("vision", "projector")},
        "tower_bit_equal": sum(r["bit_equal"] for r in rows), "tower_not_equal": bad,
        "model_tower_tensors": len(model_tower),
        "model_tower_tensors_not_in_checkpoint": sorted(set(model_tower) - {r["hf"] for r in rows}),
        "check": "every model.vision_tower.vision_model.* / model.multi_modal_projector.* checkpoint tensor (BF16) "
                 "widened to fp32 == the loaded tensor of its transformers 5 name (torch.equal)",
        "attn_implementation": {"vision": model.config.vision_config._attn_implementation,
                                "model": model.config._attn_implementation},
        "position_table_key": POSITION_HF,
        "projector_layer_norm": model.model.multi_modal_projector.layer_norm is not None,
    }
    if (bad or report["model_tower_tensors_not_in_checkpoint"] or report["transformers"]["n_missing_tower"]
            or report["transformers"]["n_unexpected_tower"] or len(rows) != 441 or report["projector_layer_norm"]):
        raise SystemExit(f"the loaded tower differs from the checkpoint: {json.dumps(report)[:3000]}")
    table = sd[POSITION_HF].numpy().astype(np.float32)
    snapshot = {"path": str(snap), "hf_id": MODEL["hf_id"], "revision": MODEL["revision"],
                "config_sha256": sha256_file(snap / "config.json"),
                "model_safetensors_bytes": ck.stat().st_size, "model_safetensors_sha256": sha256_file(ck),
                "model_safetensors_blob": os.path.basename(os.path.realpath(ck))}
    return model, table, report, snapshot


def image_features(model, ti: dict):
    import torch

    gh, gw = ti["grid"]
    return model.model.get_image_features(
        pixel_values=torch.from_numpy(ti["patches"][None]),
        spatial_shapes=torch.tensor([[gh, gw]], dtype=torch.int64),
        pixel_attention_mask=torch.from_numpy(ti["mask"][None])).pooler_output[0]


def run(args) -> int:
    import torch
    import transformers

    threads = args.threads if args.threads else (4 if args.toy else 1)
    torch.set_num_threads(threads)
    out_dir = Path(args.out_dir) if args.out_dir else (TOY_OUT if args.toy else OUT)
    record_path = out_dir / "oracle.json"
    if record_path.exists():
        raise SystemExit(f"{record_path} exists: the oracle is never overwritten")
    t0 = time.time()
    if args.toy:
        model, table, load, snap_rec = load_toy(Path(args.snapshot) if args.snapshot else TOY_OUT / "toy_snapshot")
    else:
        snap = Path(args.snapshot) if args.snapshot else Path(hf_snapshot(MODEL["hf_id"], revision=MODEL["revision"]))
        model, table, load, snap_rec = load_checkpoint(snap)
    t_loaded = time.time()
    print(f"loaded ({t_loaded - t0:.1f} s): {json.dumps({k: v for k, v in load.items() if k != 'transformers'})[:600]}",
          flush=True)
    pics = pictures(Path(args.fixtures), Path(args.host_json))
    host = json.loads(Path(args.host_json).read_text())
    want_crops = {r["id"]: sum(len(p["crops"]) for p in r["pictures"]) for r in host["records"]
                  if r["kind"] in ("fixture", "random")}
    hidden = model.config.text_config.hidden_size
    rec_pics, n_crops, first = [], 0, None
    with torch.no_grad():
        for pid, kind, rgb, prov in pics:
            p = vh.plan(*rgb.shape[:2])
            crops_u8 = vh.crop_images(rgb, p)
            if len(p.crops) != want_crops[pid]:
                raise SystemExit(f"{pid}: {len(p.crops)} crops != vision_host.json's {want_crops[pid]}")
            single, items = [], []
            for ci, (c, u8) in enumerate(zip(p.crops, crops_u8)):
                ti = vh.tower_inputs(u8, table)
                gh, gw = ti["grid"]
                tc = time.perf_counter()
                emb = image_features(model, ti).numpy().astype(np.float32)
                secs = time.perf_counter() - tc
                if emb.shape != (ti["n_tokens"], hidden):
                    raise SystemExit(f"{pid}/{ci}: image_embeds {emb.shape}")
                if first is None:
                    first = (pid, ci, ti, emb.copy())
                single.append(emb)
                name = crop_name(ci, c)
                path = out_dir / pid / f"{name}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    raise SystemExit(f"{path} exists: the oracle is never overwritten")
                np.savez_compressed(path, image_embeds=emb, patches=ti["patches"], pos_table=ti["pos_table"],
                                    key_bias=ti["key_bias"], unshuffle_idx=ti["unshuffle_idx"], mask=ti["mask"],
                                    grid=np.array([gh, gw], np.int32), n_patches=np.int32(ti["n_patches"]),
                                    n_tokens=np.int32(ti["n_tokens"]), crop_u8=u8)
                item = {"crop": name, "kind": c.kind, "row": c.row, "col": c.col, "grid": [gh, gw],
                        "n_patches": ti["n_patches"], "n_tokens": ti["n_tokens"], "npz": str(path),
                        "image_embeds_sha256": sha256_bytes(emb.tobytes()),
                        "image_embeds_absmax": float(np.abs(emb).max()),
                        "crop_u8_sha256": sha256_bytes(np.ascontiguousarray(u8).tobytes())}
                if not args.toy:
                    item["seconds"] = round(secs, 3)
                items.append(item)
            # the provider's call: every crop of the picture at once, cut to the largest real patch count
            pv = np.stack([vh.crop_pixels(u8)[0] for u8 in crops_u8])
            mk = np.stack([vh.crop_pixels(u8)[1] for u8 in crops_u8])
            n_cut = int(mk.sum(1).max())
            tb = time.perf_counter()
            batch = model.model.get_image_features(
                pixel_values=torch.from_numpy(np.ascontiguousarray(pv[:, :n_cut])),
                spatial_shapes=torch.tensor([list(c.grid) for c in p.crops], dtype=torch.int64),
                pixel_attention_mask=torch.from_numpy(np.ascontiguousarray(mk[:, :n_cut]))).pooler_output
            batch_s = time.perf_counter() - tb
            for it, a, b in zip(items, single, batch):
                b = b.numpy().astype(np.float64)
                it["provider_batch_cut_max_abs"] = float(np.abs(a.astype(np.float64) - b).max())
            n_crops += len(items)
            entry = {"id": pid, "kind": kind, "size_capped": [int(rgb.shape[1]), int(rgb.shape[0])],
                     "rows": p.rows, "cols": p.cols, "n_crops": len(items), "provider_cut": n_cut,
                     "n_image_tokens": int(sum(i["n_tokens"] for i in items)), "crops": items, **prov}
            if not args.toy:
                entry["provider_batch_seconds"] = round(batch_s, 3)
            rec_pics.append(entry)
            print(f"{pid}: {len(items)} crops, {sum(i['n_tokens'] for i in items)} tokens, batch-vs-crop max|d| "
                  f"{max(i['provider_batch_cut_max_abs'] for i in items):.2e}"
                  + ("" if args.toy else f", {sum(i['seconds'] for i in items):.1f} s + batch {batch_s:.1f} s"),
                  flush=True)
        again = None
        if not args.toy:   # the first crop once more: bit-equal
            pid, ci, ti, emb = first
            got = image_features(model, ti).numpy().astype(np.float32)
            again = {"crop": f"{pid}/{ci}", "bit_equal": bool(np.array_equal(got, emb))}
            if not again["bit_equal"]:
                raise SystemExit(f"the first crop re-run differs: max|d| {float(np.abs(got - emb).max())}")
    doc = {"schema": "d1-toy-vision-oracle/1" if args.toy else "d1-vision-oracle/1",
           "what": __doc__.splitlines()[0],
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "env": {"python": platform.python_version(), "torch": torch.__version__,
                   "transformers": transformers.__version__, "numpy": np.__version__, "threads": threads},
           "snapshot": snap_rec,
           "load": load,
           "inputs": {"fixtures": {"path": args.fixtures, "sha256": sha256_file(Path(args.fixtures))},
                      "vision_host_json": {"path": args.host_json, "sha256": sha256_file(Path(args.host_json))},
                      "vision_host_py_sha256": sha256_file(HERE / "vision_host.py"),
                      "random": f"numpy default_rng({RANDOM_SEED}).integers(0, 256, (h, w, 3), uint8) over {RANDOM_SIZES}"},
           "path": "per crop: model.model.get_image_features(pixel_values [1, 1024, 768] padded, spatial_shapes [[h, w]], "
                   "pixel_attention_mask [1, 1024]) -> pooler_output[0] [h w / 4, text_hidden] fp32 (CPU)",
           "n_pictures": len(rec_pics), "n_crops": n_crops,
           "n_crops_vision_host_json": int(sum(want_crops.values())),
           "provider_batch_cut_max_abs": max(i["provider_batch_cut_max_abs"] for r in rec_pics for i in r["crops"]),
           "pictures": rec_pics, "seconds": round(time.time() - t0, 1)}
    if not args.toy:
        secs = [i["seconds"] for r in rec_pics for i in r["crops"]]
        doc.update({"position_table": {"shape": list(table.shape), "dtype": "float32",
                                       "sha256": sha256_bytes(np.ascontiguousarray(table).tobytes()),
                                       "what": "the loaded model's " + POSITION_HF + " (fp32, C order): the host's "
                                               "table the npz pos_table rows were resized from"},
                    "rerun_first_crop": again, "load_seconds": round(t_loaded - t0, 1),
                    "seconds_per_crop": {"n": len(secs), "p50": float(np.median(secs)), "max": float(max(secs)),
                                         "total": round(float(sum(secs)), 1)},
                    "image_embeds_absmax": max(i["image_embeds_absmax"] for r in rec_pics for i in r["crops"])})
    record_path.write_text(json.dumps(doc, indent=1) + "\n")
    lr = (f"names {load['bit_equal']}/{load['file_keys']} bit-equal" if args.toy else
          f"tower keys {load['tower_bit_equal']}/{load['tower_keys']} bit-equal, missing (tower) "
          f"{load['transformers']['n_missing_tower']}, unexpected (tower) {load['transformers']['n_unexpected_tower']}")
    print(f"{n_crops} crops ({doc['n_crops_vision_host_json']} in vision_host.json), {lr}, attention "
          f"{load['attn_implementation']}, batch-vs-crop max|d| {doc['provider_batch_cut_max_abs']:.2e}, "
          f"{doc['seconds']} s -> {record_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--toy", action="store_true", help="the toy snapshot (round 3b) instead of d1-3B's checkpoint")
    ap.add_argument("--snapshot", help="default: the pinned HF snapshot (the toy: oracle_toy_vision/toy_snapshot)")
    ap.add_argument("--fixtures", default=str(LANE / "fixtures" / "image_records.json"))
    ap.add_argument("--host-json", default=str(LANE / "results" / "vision_host.json"))
    ap.add_argument("--out-dir", help="default: oracle/images (the toy: oracle_toy_vision)")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (default: 1 for the model, 4 for the toy)")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
