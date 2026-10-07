#!/usr/bin/env python3
"""Toy oracle for the exact vision tower: transformers 5.19's own LFM2-VL image path on the toy snapshot, per crop.

The toy snapshot (`lfm2_vl_tower.py write-toy`: config.json + model.safetensors laid out like d1-3B's checkpoint, the
tensors BF16 under `model.vision_tower.vision_model.*` / `model.multi_modal_projector.*`, + name_table.json) is loaded
by transformers' own `Lfm2VlForConditionalGeneration.from_pretrained` (fp32; its key conversion drops the
`vision_model.` segment), and every vision and projector tensor it holds is checked bit for bit against the file under
the name table's transformers name. The toy file has no language model: that part stays at its init and never runs.

Inputs are the processor's, through `vision_host.py` (bit-equal to transformers 5.19's processor on these pictures,
round 2b): every crop of the 12 fixture pictures (`fixtures/image_records.json`) and of the 6 random pictures of
test_vision_host.py (numpy default_rng(0) integers in its size order; sha256 checked against vision_host.json). Per
crop, as the processor lays it out — pixel_values [1, 1024, 768] zero-padded, pixel_attention_mask [1, 1024],
spatial_shapes [[h, w]] — through `model.model.get_image_features` -> the crop's image_embeds [h w / 4, text_hidden]
fp32. Per picture, its crops in one call cut to their largest real patch count (the provider's runner,
`_image_inputs`) as well, compared with the per-crop rows.

Each crop's npz also holds the host's four tower inputs (`vision_host.tower_inputs` with the toy's position table: the
contract `gate_tower.py` feeds the graph), the mask, the grid and the crop's place. The transcript records the name
check, the attention implementation, per crop the rows' sha256 and the batch-vs-crop difference.

    cd conversion/d1
    source $ZOO_WORK_ROOT/_d1_3b/venv-oracle/bin/activate && python vision_toy_oracle.py     # transformers 5.19

-> $ZOO_WORK_ROOT/_d1_3b/oracle_toy_vision/<picture>/<crop>.npz, oracle_toy_vision/oracle.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
from _paths import work_path  # noqa: E402

LANE = work_path("_d1_3b")
OUT = LANE / "oracle_toy_vision"
RANDOM_SIZES = [(384, 384), (384, 256), (640, 480), (1024, 768), (1600, 1200), (2048, 1536)]   # test_vision_host.py
RANDOM_SEED = 0


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


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


def load_model(snap: Path):
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
              "position_table_key": "model.vision_tower.embeddings.position_embedding.weight"}
    if bad or report["model_tensors_not_in_name_table"] or len(rows) != len(file_keys):
        raise SystemExit(f"the loaded tensors differ from the file: {json.dumps(report)[:2000]}")
    table = sd["model.vision_tower.embeddings.position_embedding.weight"].numpy().astype(np.float32)
    return model, table, report


def run(args) -> int:
    import torch
    import transformers

    torch.set_num_threads(args.threads)
    snap = Path(args.snapshot)
    out_dir = Path(args.out_dir)
    record_path = out_dir / "oracle.json"
    if record_path.exists():
        raise SystemExit(f"{record_path} exists: the oracle is never overwritten")
    t0 = time.time()
    model, table, load = load_model(snap)
    pics = pictures(Path(args.fixtures), Path(args.host_json))
    host = json.loads(Path(args.host_json).read_text())
    want_crops = {r["id"]: sum(len(p["crops"]) for p in r["pictures"]) for r in host["records"]
                  if r["kind"] in ("fixture", "random")}
    rec_pics, n_crops = [], 0
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
                feats = model.model.get_image_features(
                    pixel_values=torch.from_numpy(ti["patches"][None]),
                    spatial_shapes=torch.tensor([[gh, gw]], dtype=torch.int64),
                    pixel_attention_mask=torch.from_numpy(ti["mask"][None])).pooler_output
                emb = feats[0].numpy().astype(np.float32)
                if emb.shape != (ti["n_tokens"], model.config.text_config.hidden_size):
                    raise SystemExit(f"{pid}/{ci}: image_embeds {emb.shape}")
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
                items.append({"crop": name, "kind": c.kind, "row": c.row, "col": c.col, "grid": [gh, gw],
                              "n_patches": ti["n_patches"], "n_tokens": ti["n_tokens"], "npz": str(path),
                              "image_embeds_sha256": sha256_bytes(emb.tobytes()),
                              "image_embeds_absmax": float(np.abs(emb).max()),
                              "crop_u8_sha256": sha256_bytes(np.ascontiguousarray(u8).tobytes())})
            # the provider's call: every crop of the picture at once, cut to the largest real patch count
            pv = np.stack([vh.crop_pixels(u8)[0] for u8 in crops_u8])
            mk = np.stack([vh.crop_pixels(u8)[1] for u8 in crops_u8])
            n_cut = int(mk.sum(1).max())
            batch = model.model.get_image_features(
                pixel_values=torch.from_numpy(np.ascontiguousarray(pv[:, :n_cut])),
                spatial_shapes=torch.tensor([list(c.grid) for c in p.crops], dtype=torch.int64),
                pixel_attention_mask=torch.from_numpy(np.ascontiguousarray(mk[:, :n_cut]))).pooler_output
            for it, a, b in zip(items, single, batch):
                b = b.numpy().astype(np.float64)
                it["provider_batch_cut_max_abs"] = float(np.abs(a.astype(np.float64) - b).max())
            n_crops += len(items)
            rec_pics.append({"id": pid, "kind": kind, "size_capped": [int(rgb.shape[1]), int(rgb.shape[0])],
                             "rows": p.rows, "cols": p.cols, "n_crops": len(items), "provider_cut": n_cut,
                             "n_image_tokens": int(sum(i["n_tokens"] for i in items)), "crops": items, **prov})
            print(f"{pid}: {len(items)} crops, {sum(i['n_tokens'] for i in items)} tokens, batch-vs-crop max|d| "
                  f"{max(i['provider_batch_cut_max_abs'] for i in items):.2e}", flush=True)
    doc = {"schema": "d1-toy-vision-oracle/1",
           "what": __doc__.splitlines()[0],
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "env": {"python": platform.python_version(), "torch": torch.__version__,
                   "transformers": transformers.__version__, "numpy": np.__version__, "threads": args.threads},
           "snapshot": {"path": str(snap), "config_sha256": sha256_file(snap / "config.json"),
                        "model_safetensors_sha256": sha256_file(snap / "model.safetensors"),
                        "name_table_sha256": sha256_file(snap / "name_table.json")},
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
    record_path.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{n_crops} crops ({doc['n_crops_vision_host_json']} in vision_host.json), names {load['bit_equal']}/"
          f"{load['file_keys']} bit-equal, attention {load['attn_implementation']}, batch-vs-crop max|d| "
          f"{doc['provider_batch_cut_max_abs']:.2e}, {doc['seconds']} s -> {record_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", default=str(OUT / "toy_snapshot"))
    ap.add_argument("--fixtures", default=str(LANE / "fixtures" / "image_records.json"))
    ap.add_argument("--host-json", default=str(LANE / "results" / "vision_host.json"))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--threads", type=int, default=4)
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
