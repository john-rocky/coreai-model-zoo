#!/usr/bin/env python3
"""Gate: the Swift host's pixel path (apps/D1 `ImagePixels.swift`, through `d1-pixels-test`) against vision_host.py.

The Python side is the specification `vision_host.py` (gated bit for bit against transformers 5.19's processor and
torch 2.9.1 in round 2b): a picture opened as the card's `load_image` does (Pillow, `ImageOps.exif_transpose`,
`convert("RGB")`), `cap_pixels`, `plan`, `crop_images`, then `tower_inputs(crop, table)` per crop = patches [1024, 768]
f32, pos_table [1024, d] f32, key_bias [1024] f32, unshuffle_idx [256, 4] i32. The Swift side is the same picture file
through ImageIO and `D1Pixels`. Pictures: the 12 drawn fixture PNGs, the 6 random PNGs test_vision_host.py draws
(`make_fixture_images.py --random`), and the EXIF pairs (`--exif`: a JPEG tagged Orientation 6 + the PNG of the pixels
Pillow decodes from it). The position table is the toy tower's (`export_vision.py --toy`, d 96; the model's is d 1152
through the same code, checked on its own below).

    expected   the Python side -> K/swift/pixels/expected/<file name>/*.{f32,i32,u8} + index.json (sha256 per file),
               tied to round 2b (the fixture's PNG sha256 and random RGB sha256 in vision_host.json, every crop's kind
               and grid) and to torch itself (torchvision's uint8 resize of every crop, Siglip2's position-table resize
               of every grid); the probe pictures for the decoder (alpha, gray, palette, 16-bit, CMYK, the 8 EXIF
               orientations as PNG and TIFF), the EXIF pictures as JPEGs at 4:4:4 / 4:2:2 / 4:2:0, and a random
               [256, 1152] table for the wide position check
    swift      d1-pixels-test (--bin) over the same pictures: the main run with --stages, the negative controls
               (--resize float, --resize pillow, --positions unfused, also on the wide table), --decode-only on the
               probes and the JPEGs, --pos-only on the wide table
    score      per crop: patches / key_bias / unshuffle_idx / pos_table bit for bit (pos_table: max |d| when not), the
               stages (decoded and capped uint8, each crop's uint8 after the resize: max level and count; normalized
               f32); the PNG pictures must be bit-equal everywhere; the EXIF JPEGs record ImageIO's decode against
               libjpeg's (levels, and per chroma subsampling) and must agree on orientation (plan and grids as the PNG
               twin's, pixel correlation >= 0.999 with the twin); the probes whose layout the decoder reads directly
               must equal Pillow; the wide table must equal vision_host on every grid; every negative control must go
               red; a flipped byte must go red
    all        expected + swift + score

    cd conversion/d1
    source $ZOO_WORK_ROOT/_d1_3b/venv-oracle/bin/activate && python gate_swift_pixels.py all \
        --bin $ZOO_WORK_ROOT/_d1_3b/swift/.build-3d/release/d1-pixels-test
    # the binary: swift build -c release --package-path apps/D1 --scratch-path $ZOO_WORK_ROOT/_d1_3b/swift/.build-3d

-> $ZOO_WORK_ROOT/_d1_3b/results/r3d_swift_pixels.json, $ZOO_WORK_ROOT/_d1_3b/swift/pixels/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
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
PIX = LANE / "swift" / "pixels"
EXPECTED = PIX / "expected"
TABLE = LANE / "exports" / "toy_vision" / "d1_toy_vision_fp32" / "host" / "position_embedding.safetensors"
PICTURE_DIRS = [LANE / "fixtures" / "images", LANE / "fixtures" / "images_random", LANE / "fixtures" / "images_exif"]
NEG_DIRS = [LANE / "fixtures" / "images", LANE / "fixtures" / "images_random"]
VISION_HOST_JSON = LANE / "results" / "vision_host.json"
TRANSCRIPT = LANE / "results" / "r3d_swift_pixels.json"
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
CORR_BAR = 0.999
WIDE_DIM, WIDE_SEED = 1152, 0
# every grid the fixture makes, round 2b's extra grids, odd and lopsided ones (vision_resize_check.json), the extremes
WIDE_GRIDS = [(24, 24), (16, 24), (24, 16), (26, 36), (36, 26), (32, 32), (18, 54), (18, 18), (16, 16), (8, 24),
              (64, 16), (16, 64), (2, 452), (13, 9), (10, 50), (2, 512), (512, 2), (1, 1), (17, 15), (40, 26)]
RUNS = {   # swift output dir -> d1-pixels-test arguments (after --out)
    "swift": ["--images", *map(str, PICTURE_DIRS), "--table", str(TABLE), "--stages"],
    "swift_neg_resize_float": ["--images", *map(str, NEG_DIRS), "--table", str(TABLE), "--stages", "--resize", "float"],
    "swift_neg_resize_pillow": ["--images", *map(str, NEG_DIRS), "--table", str(TABLE), "--stages", "--resize", "pillow"],
    "swift_neg_pos_unfused": ["--images", *map(str, NEG_DIRS), "--table", str(TABLE), "--positions", "unfused"],
    "swift_decode": ["--decode-only", "--images", str(PIX / "decode_probes")],
    "swift_decode_jpeg": ["--decode-only", "--images", str(PIX / "jpeg_subsampling")],
    "swift_pos_wide": ["--pos-only", "--table", str(PIX / "pos_wide_table.safetensors"),
                       "--grids", ",".join(f"{h}x{w}" for h, w in WIDE_GRIDS)],
    "swift_neg_pos_wide_unfused": ["--pos-only", "--table", str(PIX / "pos_wide_table.safetensors"),
                                   "--grids", ",".join(f"{h}x{w}" for h, w in WIDE_GRIDS), "--positions", "unfused"],
}


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def pictures(dirs) -> list[Path]:
    out = []
    for d in dirs:
        out += sorted(p for p in Path(d).iterdir() if p.suffix.lower() in EXTS)
    return out


def load_rgb(path: Path) -> np.ndarray:
    """The card's `load_image` on a local file: Pillow open, `ImageOps.exif_transpose`, `convert("RGB")`."""
    from PIL import Image, ImageOps

    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"), dtype=np.uint8)


def load_table(path: Path) -> np.ndarray:
    from safetensors.numpy import load_file

    t = load_file(str(path))["position_embedding"]
    assert t.dtype == np.float32 and t.ndim == 2 and t.shape[0] == 256, (t.dtype, t.shape)
    return t


def write_raw(arr: np.ndarray, d: Path, name: str) -> dict:
    b = np.ascontiguousarray(arr).tobytes()
    (d / name).write_bytes(b)
    return {"path": str((d / name).relative_to(d.parent)), "sha256": sha256(b), "bytes": len(b)}


# --------------------------------------------------------------------------- expected
def torch_checks(capped: np.ndarray, p: vh.Plan, crops: list[np.ndarray], table: np.ndarray) -> dict:
    """vision_host vs torch on this picture: torchvision's uint8 resize of each crop, Siglip2's table resize."""
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import functional as tvF
    from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionEmbeddings

    t = torch.from_numpy(np.ascontiguousarray(capped)).permute(2, 0, 1)
    levels, resized = [], {}
    for c, mine in zip(p.crops, crops):
        if c.resize_to not in resized:
            resized[c.resize_to] = tvF.resize(t, list(c.resize_to), interpolation=InterpolationMode.BICUBIC,
                                              antialias=True).permute(1, 2, 0).numpy()
        y0, x0, y1, x1 = c.box
        levels.append(int(np.abs(resized[c.resize_to][y0:y1, x0:x1].astype(int) - mine.astype(int)).max()))
    pe = torch.from_numpy(table).reshape(16, 16, -1)
    pos = []
    for gh, gw in sorted({c.grid for c in p.crops}):
        hf = Siglip2VisionEmbeddings.resize_positional_embeddings(pe, torch.tensor([[gh, gw]]), max_length=1024)[0].numpy()
        mine = vh.pos_table(table, gh, gw)
        pos.append({"grid": [gh, gw], "bit_equal": bool(np.array_equal(hf[:gh * gw].view(np.uint32), mine.view(np.uint32)))})
    return {"resize_max_level": levels, "pos_table": pos}


def make_probes(out: Path) -> list[dict]:
    """Small pictures for the decoder: layouts Pillow and ImageIO may read differently, and the 8 EXIF orientations
    on lossless files (PNG eXIf, TIFF tag 274), from a seeded random 40 x 24 picture."""
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, (24, 40, 3), dtype=np.uint8)
    a = rng.integers(0, 256, (24, 40), dtype=np.uint8)
    a[0, :8] = 0                                   # fully transparent pixels keep their color in Pillow's RGB
    g16 = (rng.integers(0, 65536, (24, 40), dtype=np.uint32)).astype(np.uint16)
    made = {
        "rgb.png": Image.fromarray(rgb),
        "rgba.png": Image.fromarray(np.dstack([rgb, a])),
        "l.png": Image.fromarray(rgb[..., 0]),
        "la.png": Image.fromarray(np.dstack([rgb[..., 0], a])),
        "p.png": Image.fromarray(rgb).quantize(64),
        "i16.png": Image.fromarray(g16),
        "cmyk.jpg": Image.fromarray(rgb).convert("CMYK"),
    }
    want = {"rgb.png": "RGB", "rgba.png": "RGBA", "l.png": "L", "la.png": "LA", "p.png": "P", "i16.png": "I;16",
            "cmyk.jpg": "CMYK"}
    assert {k: v.mode for k, v in made.items()} == want, {k: v.mode for k, v in made.items()}
    rows = []
    for name, im in made.items():
        kw = {"quality": 95} if name.endswith(".jpg") else {}
        im.save(out / name, **kw)
        rows.append({"file": name, "mode": im.mode})
    ptr = Image.fromarray(rgb).quantize(64)
    ptr.save(out / "p_trns.png", transparency=3)
    rows.append({"file": "p_trns.png", "mode": "P + tRNS (index 3 transparent)"})
    for o in range(1, 9):
        ex = Image.Exif()
        ex[0x0112] = o
        for ext in ("png", "tif"):
            Image.fromarray(rgb).save(out / f"o{o}.{ext}", exif=ex.tobytes())
            rows.append({"file": f"o{o}.{ext}", "mode": "RGB", "exif_orientation": o})
    return rows


def make_jpeg_subsampling(out: Path) -> list[dict]:
    """The EXIF fixture's stored pictures again as JPEGs with each chroma subsampling (quality 95, Orientation 6):
    where ImageIO's decode and libjpeg's part ways."""
    from PIL import Image

    import make_fixture_images as mfi

    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in mfi.EXIF_SOURCES:
        fn, seed = mfi.IMAGES[src][0], mfi.IMAGES[src][1]
        stored = fn(seed).im.transpose(Image.Transpose.ROTATE_270)
        ex = Image.Exif()
        ex[0x0112] = mfi.EXIF_ORIENTATION
        for code, tag in ((0, "444"), (1, "422"), (2, "420")):
            name = f"{src.split('_')[0]}_{tag}.jpg"
            stored.save(out / name, quality=mfi.JPEG_QUALITY, subsampling=code, exif=ex.tobytes())
            rows.append({"file": name, "subsampling": tag, "source": src})
    return rows


def cmd_expected(args) -> dict:
    import PIL
    import torch
    import torchvision
    import transformers

    t0 = time.time()
    if EXPECTED.exists():
        shutil.rmtree(EXPECTED)
    EXPECTED.mkdir(parents=True)
    table = load_table(TABLE)
    vhj = json.loads(VISION_HOST_JSON.read_text())
    ref = {r["id"]: r for r in vhj["records"]}
    index = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "table": {"path": str(TABLE), "sha256": sha256(TABLE.read_bytes()), "dim": int(table.shape[1])},
             "env": {"python": platform.python_version(), "numpy": np.__version__, "pillow": PIL.__version__,
                     "torch": torch.__version__, "torchvision": torchvision.__version__,
                     "transformers": transformers.__version__},
             "pictures": []}
    ties = {"png_sha256": [], "rgb_sha256": [], "crops": [], "torch_resize_zero": [], "torch_pos_bit_equal": []}
    for path in pictures(PICTURE_DIRS):
        pid = path.name
        d = EXPECTED / pid
        d.mkdir()
        rgb = load_rgb(path)
        capped = vh.cap_pixels(rgb)
        p = vh.plan(*capped.shape[:2])
        crops = vh.crop_images(capped, p)
        files = {"decoded": write_raw(rgb, d, "decoded.u8")}
        if capped.shape != rgb.shape:
            files["capped"] = write_raw(capped, d, "capped.u8")
        crop_rows = []
        for k, (c, u8) in enumerate(zip(p.crops, crops)):
            ti = vh.tower_inputs(u8, table)
            cf = {"patches": write_raw(ti["patches"], d, f"c{k}_patches.f32"),
                  "key_bias": write_raw(ti["key_bias"], d, f"c{k}_key_bias.f32"),
                  "unshuffle_idx": write_raw(ti["unshuffle_idx"], d, f"c{k}_unshuffle_idx.i32"),
                  "pos_table": write_raw(ti["pos_table"], d, f"c{k}_pos_table.f32"),
                  "crop": write_raw(u8, d, f"c{k}_crop.u8"),
                  "normalized": write_raw(vh.normalize(u8), d, f"c{k}_normalized.f32")}
            crop_rows.append({"k": k, "kind": c.kind, "row": c.row, "col": c.col, "size": list(c.size),
                              "grid": list(c.grid), "n_patches": c.n_patches, "n_tokens": ti["n_tokens"], "files": cf})
        tc = torch_checks(capped, p, crops, table)
        entry = {"id": pid, "path": str(path), "file_sha256": sha256(path.read_bytes()),
                 "decoded": [int(rgb.shape[1]), int(rgb.shape[0])], "capped": [int(capped.shape[1]), int(capped.shape[0])],
                 "plan": {"rows": p.rows, "cols": p.cols, "thumb": list(p.thumb), "tokens": vh.n_image_tokens([p])},
                 "crops": crop_rows, "files": files, "torch": tc}
        stem = path.stem
        if stem in ref:       # round 2b's record of the same picture
            r = ref[stem]
            if "png_sha256" in r:
                ties["png_sha256"].append(r["png_sha256"] == entry["file_sha256"])
            if "rgb_sha256" in r:
                ties["rgb_sha256"].append(r["rgb_sha256"] == sha256(rgb.tobytes()))
            want = [(c["kind"], tuple(c["grid"])) for pc in r["pictures"] for c in pc["crops"]]
            ties["crops"].append(want == [(c["kind"], tuple(c["grid"])) for c in crop_rows])
            entry["round2b_record"] = stem
        ties["torch_resize_zero"].append(all(v == 0 for v in tc["resize_max_level"]))
        ties["torch_pos_bit_equal"].append(all(g["bit_equal"] for g in tc["pos_table"]))
        index["pictures"].append(entry)
        print(f"expected {pid}: {len(crop_rows)} crops {[c['grid'] for c in crop_rows]}", flush=True)
    # the EXIF PNG twins hold exactly what Pillow decodes from their JPEG: their expected files must be the same
    twins = []
    by_id = {e["id"]: e for e in index["pictures"]}
    for e in index["pictures"]:
        if e["id"].endswith(".jpg") and e["id"][:-4] + ".png" in by_id:
            png = by_id[e["id"][:-4] + ".png"]
            same = all(a["files"][f]["sha256"] == b["files"][f]["sha256"] for a, b in zip(e["crops"], png["crops"])
                       for f in a["files"]) and len(e["crops"]) == len(png["crops"])
            twins.append({"jpg": e["id"], "png": png["id"], "python_inputs_identical": same})
    # the decoder probes and the wide table
    probes = make_probes(PIX / "decode_probes")
    jpeg_rows = make_jpeg_subsampling(PIX / "jpeg_subsampling")
    rng = np.random.default_rng(WIDE_SEED)
    wide = rng.standard_normal((256, WIDE_DIM)).astype(np.float32)
    from safetensors.numpy import save_file

    save_file({"position_embedding": wide}, str(PIX / "pos_wide_table.safetensors"))
    n18 = [e for e in index["pictures"] if e.get("round2b_record")]
    index["ties"] = {
        "round2b_pictures": len(n18), "round2b_crops": sum(len(e["crops"]) for e in n18),
        "png_sha256_equal": f"{sum(ties['png_sha256'])}/{len(ties['png_sha256'])}",
        "random_rgb_sha256_equal": f"{sum(ties['rgb_sha256'])}/{len(ties['rgb_sha256'])}",
        "crops_kind_grid_equal": f"{sum(ties['crops'])}/{len(ties['crops'])}",
        "torch_resize_all_zero": f"{sum(ties['torch_resize_zero'])}/{len(ties['torch_resize_zero'])}",
        "torch_pos_bit_equal": f"{sum(ties['torch_pos_bit_equal'])}/{len(ties['torch_pos_bit_equal'])}",
        "exif_twins": twins,
    }
    index["probes"] = probes
    index["jpeg_subsampling"] = jpeg_rows
    index["pos_wide"] = {"table": f"standard normal [256, {WIDE_DIM}] float32, numpy default_rng({WIDE_SEED})",
                         "path": str(PIX / "pos_wide_table.safetensors"), "grids": [list(g) for g in WIDE_GRIDS]}
    index["seconds"] = round(time.time() - t0, 1)
    (EXPECTED / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    print("expected:", json.dumps(index["ties"]), f"{index['seconds']} s", flush=True)
    return index


# --------------------------------------------------------------------------- swift
def cmd_swift(args) -> dict:
    binary = Path(args.bin).expanduser()
    if not binary.is_file():
        raise SystemExit(f"no binary at {binary}")
    runs = {}
    for name, argv in RUNS.items():
        out = PIX / name
        if out.exists():
            shutil.rmtree(out)
        t0 = time.time()
        cp = subprocess.run([str(binary), *argv, "--out", str(out)], capture_output=True, text=True)
        runs[name] = {"argv": argv, "exit": cp.returncode, "seconds": round(time.time() - t0, 2),
                      "stderr": cp.stderr[-2000:], "stdout_tail": cp.stdout[-600:]}
        print(f"swift {name}: exit {cp.returncode} {runs[name]['seconds']} s", flush=True)
        if cp.returncode != 0:
            print(cp.stderr[-2000:], file=sys.stderr)
    runs["binary"] = {"path": str(binary), "sha256": sha256(binary.read_bytes()), "bytes": binary.stat().st_size}
    (PIX / "swift_runs.json").write_text(json.dumps(runs, indent=1) + "\n")
    return runs


# --------------------------------------------------------------------------- score
def raw(root: Path, f: dict, dtype) -> np.ndarray:
    return np.fromfile(root / f["path"], dtype=dtype)


def same_bytes(a: Path, b: Path) -> bool:
    return a.read_bytes() == b.read_bytes()


def level_diff(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"shape": [list(a.shape), list(b.shape)], "max_level": None, "n_diff": None}
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {"max_level": int(d.max()) if d.size else 0, "n_diff": int((d > 0).sum()), "n": int(d.size),
            "mean_abs": float(d.mean()) if d.size else 0.0}


def float_diff(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"shape": [list(a.shape), list(b.shape)], "bit_equal": False, "max_abs": None}
    bit = bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))
    with np.errstate(invalid="ignore"):
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    finite = np.isfinite(a) & np.isfinite(b)
    return {"bit_equal": bit, "max_abs": float(d[finite].max()) if finite.any() else 0.0,
            "n_diff": int((a.view(np.uint32) != b.view(np.uint32)).sum())}


def compare_picture(e: dict, s: dict, sroot: Path, stages: bool = True) -> dict:
    """One picture: Swift's files (under sroot) against the expected ones. A file the expected side has and Swift's
    run did not write counts as unequal (`stages` false: a run without --stages, whose stage files are not read)."""
    row = {"id": e["id"], "layout": s.get("layout"), "exact": s.get("exact"), "orientation": s.get("orientation"),
           "decoded": [e["decoded"], s.get("decoded")], "capped": [e["capped"], s.get("capped")]}
    row["plan_equal"] = (s.get("plan", {}).get("rows"), s.get("plan", {}).get("cols"), s.get("plan", {}).get("thumb"),
                         s.get("plan", {}).get("tokens")) == (e["plan"]["rows"], e["plan"]["cols"], e["plan"]["thumb"],
                                                              e["plan"]["tokens"])
    ecrops, scrops = e["crops"], s.get("crops", [])
    row["crops_equal"] = [(c["kind"], c["row"], c["col"], c["size"], c["grid"], c["n_tokens"]) for c in ecrops] == \
                         [(c["kind"], c["row"], c["col"], c["size"], c["grid"], c["n_tokens"]) for c in scrops]
    row["sizes_equal"] = e["decoded"] == s.get("decoded") and e["capped"] == s.get("capped")
    sf = s.get("files", {})
    if "decoded" in sf:
        row["stage_decoded"] = level_diff(raw(EXPECTED, e["files"]["decoded"], np.uint8),
                                          raw(sroot, sf["decoded"], np.uint8))
    elif stages:
        row["stage_decoded"] = {"missing": True, "max_level": None}
    if stages and ("capped" in e["files"] or "capped" in sf):
        row["stage_capped"] = (level_diff(raw(EXPECTED, e["files"]["capped"], np.uint8), raw(sroot, sf["capped"], np.uint8))
                               if "capped" in e["files"] and "capped" in sf else {"missing": True, "max_level": None})
    crows = []
    for ec, sc in zip(ecrops, scrops):
        ef, scf = ec["files"], sc["files"]
        c = {"k": ec["k"], "kind": ec["kind"], "grid": ec["grid"]}
        for name in ("patches", "key_bias", "unshuffle_idx"):
            c[name] = same_bytes(EXPECTED / ef[name]["path"], sroot / scf[name]["path"])
        if "pos_table" in scf:
            c["pos_table"] = same_bytes(EXPECTED / ef["pos_table"]["path"], sroot / scf["pos_table"]["path"])
            if not c["pos_table"]:
                c["pos_table_diff"] = float_diff(raw(EXPECTED, ef["pos_table"], np.float32),
                                                 raw(sroot, scf["pos_table"], np.float32))
        else:
            c["pos_table"], c["pos_table_missing"] = False, True
        if not c["patches"]:
            c["patches_diff"] = float_diff(raw(EXPECTED, ef["patches"], np.float32), raw(sroot, scf["patches"], np.float32))
            c["patches_diff"]["max_level"] = (round(c["patches_diff"]["max_abs"] * 127.5, 3)
                                              if c["patches_diff"]["max_abs"] is not None else None)
        if "crop" in scf:
            c["stage_crop"] = level_diff(raw(EXPECTED, ef["crop"], np.uint8), raw(sroot, scf["crop"], np.uint8))
        elif stages:
            c["stage_crop"] = {"missing": True, "max_level": None}
        if "normalized" in scf:
            c["stage_normalized_bit_equal"] = same_bytes(EXPECTED / ef["normalized"]["path"],
                                                         sroot / scf["normalized"]["path"])
        elif stages:
            c["stage_normalized_bit_equal"] = False
        crows.append(c)
    row["crops"] = crows
    row["n_crops"] = [len(ecrops), len(scrops)]
    keys = ["patches", "key_bias", "unshuffle_idx", "pos_table"]
    row["all_bit_equal"] = bool(row["plan_equal"] and row["crops_equal"] and row["sizes_equal"] and crows
                                and len(ecrops) == len(scrops)
                                and all(c[k] for c in crows for k in keys)
                                and (not stages or all(c["stage_crop"]["max_level"] == 0
                                                       and c["stage_normalized_bit_equal"] for c in crows))
                                and (not stages or row["stage_decoded"]["max_level"] == 0)
                                and row.get("stage_capped", {}).get("max_level", 0) == 0)
    return row


def exif_check(e_jpg: dict, s_jpg: dict, e_png: dict, sroot: Path) -> dict:
    """ImageIO's JPEG decode with the orientation applied, against the PNG twin (= Pillow's decode, transposed)."""
    a = raw(sroot, s_jpg["files"]["decoded"], np.uint8)
    b = raw(EXPECTED, e_png["files"]["decoded"], np.uint8)
    out = {"jpg": e_jpg["id"], "png": e_png["id"], "orientation_read": s_jpg.get("orientation"),
           "stored": s_jpg.get("stored"), "decoded": s_jpg.get("decoded"), "png_size": e_png["decoded"],
           "plan_as_png": (s_jpg.get("plan", {}).get("rows"), s_jpg.get("plan", {}).get("cols"),
                           s_jpg.get("plan", {}).get("thumb")) == (e_png["plan"]["rows"], e_png["plan"]["cols"],
                                                                   e_png["plan"]["thumb"]),
           "grids_as_png": [c["grid"] for c in s_jpg.get("crops", [])] == [c["grid"] for c in e_png["crops"]]}
    if a.shape == b.shape:
        out["decode_vs_pillow"] = level_diff(b, a)
        out["corr"] = float(np.corrcoef(a.astype(np.float64), b.astype(np.float64))[0, 1])
        w, h = e_png["decoded"]
        bb = b.reshape(h, w, 3)
        # the twin = the stored picture turned 90 degrees clockwise. Turned the other way it is the twin turned 180;
        # left as stored it is the twin turned 90 counter-clockwise (same size only when square: else the plan differs)
        wrong = {"turned_counter_clockwise": np.rot90(bb, 2)}
        if w == h:
            wrong.update({"tag_ignored": np.rot90(bb, 1), "turned_twice": np.rot90(bb, -1)})
        out["corr_wrong_orientations"] = {k: float(np.corrcoef(a.astype(np.float64), v.reshape(-1).astype(np.float64))[0, 1])
                                          for k, v in wrong.items()}
    else:
        out["decode_vs_pillow"] = {"shape": [list(a.shape), list(b.shape)]}
        out["corr"] = None
    # the tower's inputs from ImageIO's pixels against Pillow's (the JPEG's own expected files)
    pd = []
    for ec, sc in zip(e_jpg["crops"], s_jpg.get("crops", [])):
        d = float_diff(raw(EXPECTED, ec["files"]["patches"], np.float32), raw(sroot, sc["files"]["patches"], np.float32))
        pd.append({"k": ec["k"], "grid": ec["grid"], "patches_max_level": round(d["max_abs"] * 127.5, 3)
                   if d["max_abs"] is not None else None, "patches_n_diff": d.get("n_diff"),
                   "pos_key_unshuffle_bit_equal": all(same_bytes(EXPECTED / ec["files"][f]["path"], sroot / sc["files"][f]["path"])
                                                      for f in ("pos_table", "key_bias", "unshuffle_idx"))})
    out["tower_inputs_vs_pillow"] = pd
    out["pass"] = bool(out["orientation_read"] == 6 and out["plan_as_png"] and out["grids_as_png"]
                       and out["corr"] is not None and out["corr"] >= CORR_BAR
                       and all(c["pos_key_unshuffle_bit_equal"] for c in pd))
    return out


def negative(name: str, expected: dict, what: str) -> dict:
    root = PIX / name
    idx = json.loads((root / "index.json").read_text())
    by = {p["id"]: p for p in idx["pictures"]}
    n_crop, red_crops, worst_level, worst_pos, n_pics = 0, 0, 0, 0.0, 0
    by_grid: dict[str, list[int]] = {}
    for e in expected["pictures"]:
        if e["id"] not in by:
            continue
        n_pics += 1
        r = compare_picture(e, by[e["id"]], root, stages="--stages" in idx.get("argv", []))
        for c in r["crops"]:
            n_crop += 1
            if what == "resize":
                bad = not c["patches"]
                worst_level = max(worst_level, c.get("stage_crop", {}).get("max_level") or 0)
            else:
                bad = not c["pos_table"]
                worst_pos = max(worst_pos, c.get("pos_table_diff", {}).get("max_abs") or 0.0)
            red_crops += bad
            g = by_grid.setdefault("x".join(map(str, c["grid"])), [0, 0])
            g[0] += bool(bad)
            g[1] += 1
    out = {"run": name, "pictures": n_pics, "crops": n_crop, "crops_red": red_crops, "red": red_crops > 0,
           "red_by_grid": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_grid.items())}}
    if what == "resize":
        out["max_crop_level"] = worst_level
    else:
        out["pos_table_max_abs"] = worst_pos
    return out


def cmd_score(args) -> dict:
    t0 = time.time()
    expected = json.loads((EXPECTED / "index.json").read_text())
    sroot = Path(args.swift_out) if args.swift_out else PIX / "swift"
    sidx = json.loads((sroot / "index.json").read_text())
    sby = {p["id"]: p for p in sidx["pictures"]}
    rows = [compare_picture(e, sby[e["id"]], sroot) if e["id"] in sby else {"id": e["id"], "missing": True}
            for e in expected["pictures"]]
    eby = {e["id"]: e for e in expected["pictures"]}
    exif_dir = str(PICTURE_DIRS[2])
    png18 = [r for r in rows if r["id"].lower().endswith(".png") and str(Path(eby[r["id"]]["path"]).parent) != exif_dir]
    exif_png = [r for r in rows if r["id"].lower().endswith(".png") and str(Path(eby[r["id"]]["path"]).parent) == exif_dir]
    exif = [exif_check(eby[j], sby[j], eby[j[:-4] + ".png"], sroot) for j in sorted(sby) if j.endswith(".jpg")]
    crops18 = [c for r in png18 for c in r.get("crops", [])]

    def count(rs, key):
        return f"{sum(bool(c.get(key)) for c in rs)}/{len(rs)}"

    summary = {
        "png_pictures_bit_equal": f"{sum(r.get('all_bit_equal', False) for r in png18)}/{len(png18)}",
        "png_crops": len(crops18),
        "patches_bit_equal": count(crops18, "patches"), "key_bias_bit_equal": count(crops18, "key_bias"),
        "unshuffle_idx_bit_equal": count(crops18, "unshuffle_idx"), "pos_table_bit_equal": count(crops18, "pos_table"),
        "pos_table_max_abs": max((c.get("pos_table_diff", {}).get("max_abs") or 0.0) for c in crops18) if crops18 else None,
        "stage_decoded_zero": f"{sum(r.get('stage_decoded', {}).get('max_level') == 0 for r in png18)}/{len(png18)}",
        "stage_capped_zero": f"{sum(r.get('stage_capped', {}).get('max_level') == 0 for r in png18 if 'stage_capped' in r)}/"
                             f"{sum('stage_capped' in r for r in png18)}",
        "stage_crop_zero": f"{sum(c.get('stage_crop', {}).get('max_level') == 0 for c in crops18)}/{len(crops18)}",
        "stage_normalized_bit_equal": count(crops18, "stage_normalized_bit_equal"),
        "plan_equal": f"{sum(r.get('plan_equal', False) for r in png18)}/{len(png18)}",
        "exif_png_twins_bit_equal": f"{sum(r.get('all_bit_equal', False) for r in exif_png)}/{len(exif_png)}",
        "exif_jpg_pass": f"{sum(x['pass'] for x in exif)}/{len(exif)}",
    }
    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "expected_index": str(EXPECTED / "index.json"), "swift_index": str(sroot / "index.json"),
              "table": expected["table"], "ties_round2b_and_torch": expected["ties"], "summary": summary,
              "pictures": rows, "exif": exif}
    runs_path = PIX / "swift_runs.json"
    if runs_path.exists():
        report["swift_runs"] = json.loads(runs_path.read_text())
    # negative controls (each must go red)
    neg = []
    for name, what in (("swift_neg_resize_float", "resize"), ("swift_neg_resize_pillow", "resize"),
                       ("swift_neg_pos_unfused", "pos")):
        if (PIX / name / "index.json").exists():
            neg.append(negative(name, expected, what))
    # the comparison itself can fail: one flipped byte in a copy of a Swift patches file
    e0, s0 = expected["pictures"][0], sby[expected["pictures"][0]["id"]]
    b = bytearray((sroot / s0["crops"][0]["files"]["patches"]["path"]).read_bytes())
    b[len(b) // 3] ^= 0x01
    flipped = bytes(b) == (EXPECTED / e0["crops"][0]["files"]["patches"]["path"]).read_bytes()
    report["negative"] = {"runs": neg, "flipped_byte_still_equal": flipped,
                          "red": f"{sum(n['red'] for n in neg)}/{len(neg)}"}
    wneg = PIX / "swift_neg_pos_wide_unfused"
    if (wneg / "index.json").exists():
        table = load_table(PIX / "pos_wide_table.safetensors")
        wrows = []
        for g in json.loads((wneg / "index.json").read_text())["grids"]:
            gh, gw = g["grid"]
            mine = np.fromfile(wneg / g["file"]["path"], dtype=np.float32).reshape(gh * gw, -1)
            d = float_diff(vh.pos_table(table, gh, gw), mine)
            wrows.append({"grid": [gh, gw], "red": not d["bit_equal"], "max_abs": d["max_abs"], "n_diff": d["n_diff"]})
        report["negative"]["pos_wide_unfused"] = {"dim": WIDE_DIM, "grids": wrows,
                                                  "grids_red": f"{sum(r['red'] for r in wrows)}/{len(wrows)}",
                                                  "max_abs": max(r["max_abs"] for r in wrows)}
    # decoder probes: ImageIO + D1Pixels.decode against Pillow (exif_transpose, convert RGB)
    probes = []
    droot = PIX / "swift_decode"
    if (droot / "index.json").exists():
        didx = {p["id"]: p for p in json.loads((droot / "index.json").read_text())["pictures"]}
        for pr in expected.get("probes", []):
            f = pr["file"]
            s = didx.get(f)
            if s is None:
                probes.append({**pr, "missing": True})
                continue
            mine = raw(droot, s["files"]["decoded"], np.uint8)
            ref = load_rgb(PIX / "decode_probes" / f).reshape(-1)
            ld = level_diff(ref, mine)
            probes.append({**pr, "layout": s["layout"], "exact": s["exact"], "orientation_read": s["orientation"],
                           "size": s["decoded"], "bit_equal": bool(ref.shape == mine.shape and np.array_equal(ref, mine)),
                           **{k: ld.get(k) for k in ("max_level", "n_diff", "n", "shape") if k in ld}})
    report["decode_probes"] = probes
    jrows = []
    jroot = PIX / "swift_decode_jpeg"
    if (jroot / "index.json").exists():
        jidx = {p["id"]: p for p in json.loads((jroot / "index.json").read_text())["pictures"]}
        for jr in expected.get("jpeg_subsampling", []):
            s = jidx[jr["file"]]
            ld = level_diff(load_rgb(PIX / "jpeg_subsampling" / jr["file"]).reshape(-1),
                            raw(jroot, s["files"]["decoded"], np.uint8))
            jrows.append({**jr, "orientation_read": s["orientation"], **ld})
    report["jpeg_subsampling"] = jrows
    # the wide position table (d 1152) on 20 grids: Swift against vision_host and torch
    wide = []
    wroot = PIX / "swift_pos_wide"
    if (wroot / "index.json").exists():
        import torch
        from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionEmbeddings

        table = load_table(PIX / "pos_wide_table.safetensors")
        pe = torch.from_numpy(table).reshape(16, 16, -1)
        widx = json.loads((wroot / "index.json").read_text())
        for g in widx["grids"]:
            gh, gw = g["grid"]
            mine = np.fromfile(wroot / g["file"]["path"], dtype=np.float32).reshape(gh * gw, -1)
            ref = vh.pos_table(table, gh, gw)
            row = {"grid": [gh, gw], "vs_vision_host": float_diff(ref, mine)}
            if gh * gw <= 1024:
                hf = Siglip2VisionEmbeddings.resize_positional_embeddings(pe, torch.tensor([[gh, gw]]), max_length=1024)
                row["vs_torch"] = float_diff(hf[0].numpy()[:gh * gw], mine)
            wide.append(row)
    report["pos_wide"] = {"dim": WIDE_DIM, "grids": wide,
                          "bit_equal_vision_host": f"{sum(r['vs_vision_host']['bit_equal'] for r in wide)}/{len(wide)}",
                          "bit_equal_torch": f"{sum(r['vs_torch']['bit_equal'] for r in wide if 'vs_torch' in r)}/"
                                             f"{sum('vs_torch' in r for r in wide)}"}
    # pass / fail
    fails = []
    if len(png18) != 18 or any(not r.get("all_bit_equal", False) for r in png18):
        fails.append("png_bit_equal")
    if any(not r.get("all_bit_equal", False) for r in exif_png):
        fails.append("exif_png_twins")
    if not exif or any(not x["pass"] for x in exif):
        fails.append("exif_orientation")
    wide_neg = report["negative"].get("pos_wide_unfused")
    if len(neg) != 3 or any(not n["red"] for n in neg) or flipped or not wide_neg or wide_neg["grids_red"].startswith("0/"):
        fails.append("negative_controls")
    if (len(probes) != len(expected.get("probes", [])) or any(p.get("missing") for p in probes)
            or any(not p.get("bit_equal", False) for p in probes if p.get("exact"))):
        fails.append("decode_probes_exact_layouts")
    if len(jrows) != len(expected.get("jpeg_subsampling", [])) or not jrows:
        fails.append("jpeg_subsampling_record")
    if len(wide) != len(WIDE_GRIDS) or any(not r["vs_vision_host"]["bit_equal"] for r in wide):
        fails.append("pos_wide")
    ties = expected["ties"]
    if not (ties["round2b_pictures"] == 18 and ties["round2b_crops"] == 64
            and all(v.split("/")[0] == v.split("/")[1] for k, v in ties.items() if isinstance(v, str))
            and all(t["python_inputs_identical"] for t in ties["exif_twins"])):
        fails.append("expected_ties")
    report["fails"] = fails
    report["pass"] = not fails
    report["seconds"] = round(time.time() - t0, 1)
    out = Path(args.transcript)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(summary))
    print("exif:", [(x["jpg"], x["orientation_read"], x["plan_as_png"], round(x["corr"] or 0, 6),
                     x["decode_vs_pillow"].get("max_level")) for x in exif])
    print("negative:", [(n["run"], n["crops_red"], n["crops"]) for n in neg], "flipped byte equal:", flipped)
    print("probes:", [(p["file"], p.get("layout"), p.get("bit_equal")) for p in probes])
    print("pos wide:", report["pos_wide"]["bit_equal_vision_host"], "torch", report["pos_wide"]["bit_equal_torch"])
    print("PASS" if not fails else f"FAIL: {fails}", flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["expected", "swift", "score", "all"])
    ap.add_argument("--bin", default=str(LANE / "swift" / ".build-3d" / "release" / "d1-pixels-test"))
    ap.add_argument("--swift-out", default=None, help="the main run's output dir (default K/swift/pixels/swift)")
    ap.add_argument("--transcript", default=str(TRANSCRIPT))
    args = ap.parse_args()
    if args.command in ("expected", "all"):
        cmd_expected(args)
    if args.command in ("swift", "all"):
        cmd_swift(args)
    if args.command in ("score", "all"):
        return 0 if cmd_score(args)["pass"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
