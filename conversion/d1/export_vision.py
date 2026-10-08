#!/usr/bin/env python3
"""Export the d1 vision tower in its exact form (any crop's grid as inputs) to a Core AI bundle, and AOT-compile it.

The graph is `lfm2_vl_tower.Lfm2VlTowerExact` (contract in that file's header): one function `main`, every shape
static, no state:

    patches [1024, 768], pos_table [1024, d], key_bias [1024]   (fp16 for fp16, fp32 otherwise)
    unshuffle_idx [256, 4] int32                                  ->  image_embeds [256, text_hidden] (the same float dtype)

dtypes (`--dtype`):

    fp16      every weight and the math fp16
    fp16w32   the weights stored fp16 and read through a .float() cast; the math, the inputs and the output fp32
              (export_qwen38vl_pipelined.fp16_storage_fp32_compute: the decider-2b-vision tower's ship form, for a
              tower whose fp16 math misses; the checkpoint's BF16 weights fit fp16 storage)
    fp32      weights and math fp32

`--attention matmul` (default) is the bare matmul-softmax chain with the additive key_bias; `sdpa` is the coreai SDPA
composite with the bool mask key_bias == 0 (name suffix `_sdpa`).

The bundle is `<out-dir>/vision/<name>/`, `<name>` = `d1_3b_vision_<dtype>` (the toy's `d1_toy_vision_<dtype>` under
`toy_vision/`):

    <name>.aimodel
    metadata.json                        kind `vision-tower`: the graph's inputs and output (shape, dtype, meaning), the
                                         host's preprocessing in short (K/results/vision_rules.md (b)-(h)), how
                                         pos_table / key_bias / unshuffle_idx are made, the three crop kinds, how the
                                         rows become the decoder's image slots, the source and the load report
    host/position_embedding.safetensors  the checkpoint's 16 x 16 position table, fp32 [256, d]: the host resizes it per
                                         crop (vision_host.pos_table); it is not a graph weight
    LICENSE                              the snapshot's LFM Open License v1.0, verbatim

`--aot` compiles it for the Mac GPU into `<out-dir>/vision_aotc/<name>.h16c.aimodelc` (`coreai-build compile
--platform macOS --preferred-compute gpu --architecture h16c`, without --expect-frequent-reshapes: every input is
static, and a fixed-shape graph compiled with it has crashed before). The record gives the asset's main.hash, which
names the entry the Python runtime makes under ~/Library/Caches/coreai-cache/<build>/python/ when it loads it.

`--toy` runs the same path on the toy snapshot (`lfm2_vl_tower.py write-toy`, laid out like the checkpoint, BF16,
random weights; `vision_oracle.py --toy` is its transformers oracle). Without `--toy` it needs model.safetensors
(`vision_oracle.py` is the model's oracle). The record also says how many checkpoint values fp16 storage does not hold
exactly (`fp16_storage`: BF16 values outside fp16's normal range) and the compiled asset's file sizes
(`aot.files`, `aot.resources_bin_bytes`: what an fp16w32 asset really stores).

    cd conversion/d1
    export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer PY=<coreai-models venv>/bin/python
    ~/code/standup/tools/quiet/quiet_wait.py --max-wait 3600 -- \\
        $PY export_vision.py --toy --dtype fp32 --aot --record $ZOO_WORK_ROOT/_d1_3b/results/<json>
    $PY export_vision.py --dtype fp16w32 --aot --record <json>         # the model: needs model.safetensors
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import export_decoder as ed  # noqa: E402
from _paths import work_path  # noqa: E402

LANE = work_path("_d1_3b")
TOY_SNAPSHOT = LANE / "oracle_toy_vision" / "toy_snapshot"
NAME_PREFIX, TOY_PREFIX = "d1_3b_vision", "d1_toy_vision"
AOT_FLAGS = ["--platform", "macOS", "--preferred-compute", "gpu", "--architecture", "h16c"]


# --------------------------------------------------------------------------- metadata
def graph_block(model, dtype: str) -> dict:
    import lfm2_vl_tower as T

    fl = "float16" if dtype == "fp16" else "float32"
    d, n, t = model.vcfg.hidden_size, model.n_patches, model.n_tokens
    return {
        "function": "main",
        "inputs": {
            "patches": [[n, model.vcfg.patch_dim], fl],
            "pos_table": [[n, d], fl],
            "key_bias": [[n], fl],
            "unshuffle_idx": [[t, 4], "int32"],
        },
        "outputs": {"image_embeds": [[t, model.text_hidden], fl]},
        "states": {},
        "meaning": {
            "patches": "the crop's patches row-major over its (h, w) patch grid, [y][x][c] inside a 16 x 16 patch "
                       "(channel fastest), (x - 127.5) / 127.5 in float32; rows h * w .. 1023 zero",
            "pos_table": "host/position_embedding.safetensors [256, d] as [16, 16, d], resized to (h, w) by "
                         "F.interpolate(bilinear, align_corners=False, antialias=True) in float32, row-major [h * w, d]; "
                         "rows h * w .. 1023 zero (transformers writes row 0 there; those rows never reach a real one)",
            "key_bias": "0 for rows 0 .. h * w - 1, -inf after: added to every query's attention scores"
                        + (" (the sdpa form reads key_bias == 0 as its attend mask)" if model.attention == "sdpa" else ""),
            "unshuffle_idx": "row k = i * (w / 2) + j of the (h / 2, w / 2) merged grid: [2i w + 2j, 2i w + 2j + 1, "
                             "(2i + 1) w + 2j, (2i + 1) w + 2j + 1] (channel m * d + c of the 4 d-vector comes from "
                             "patch m); rows k >= h w / 4 are [0, 0, 0, 0]",
            "image_embeds": "rows 0 .. h w / 4 - 1 = the crop's image tokens in merged-grid row-major order (the "
                            "projector's output, no LayerNorm); rows after are padding and are discarded",
        },
        "math": "x = patch_embedding(patches) + pos_table; %d x [x += attn(LN1 x) (scores = q k^T / sqrt(%d) + key_bias, "
                "softmax over keys); x += fc2(gelu_tanh(fc1(LN2 x)))]; post_layernorm; gather the 4 rows of "
                "unshuffle_idx -> [%d, %d]; linear_2(gelu_exact(linear_1(.)))"
                % (model.vcfg.num_hidden_layers, model.vcfg.head_dim, t, 4 * d),
        "attention": model.attention,
        "input_names": list(T.INPUT_NAMES), "output_names": list(T.OUTPUT_NAMES),
    }


def host_block() -> dict:
    """The host's side of the contract, from vision_host.py (gated bit for bit against transformers 5.19, round 2b)."""
    return {
        "spec": "conversion/d1/vision_host.py (sections 1-8; test_vision_host.py gates it against the provider's code and "
                "transformers 5.19's Lfm2VlProcessor: ids, pixel_values bit for bit, the position table, the unshuffle)",
        "picture": "decoded with its EXIF orientation applied, RGB; over 1024 x 1024 pixels it is first scaled by "
                   "s = sqrt(1024^2 / (w h)) to (int(w s), int(h s)) with Pillow BICUBIC (the provider's cap_pixels)",
        "crops": {
            "rule": "F = 32; round_f(v) = round(v / F) * F (half to even). One crop when max(16, round_f(h)) * "
                    "max(16, round_f(w)) <= 524,288, else tiles. smart size (h_bar, w_bar) = (round_f(h), round_f(w)) "
                    "floored at 32, scaled down by sqrt(h w / 262,144) (floor to F) when over 262,144 px, up by "
                    "sqrt(65,536 / (h w)) (ceil to F) when under 65,536 px",
            "single": "the picture resized to (h_bar, w_bar): grid (h_bar / 16, w_bar / 16) <= 1024 patches, both even",
            "tile": "the (cols, rows) of 2 <= c r <= 10 closest to the aspect w / h (vision_host.TARGET_RATIOS order, "
                    "ties to the later ratio when w h > 0.5 * 512^2 * c r); the picture resized to (512 rows, 512 cols) "
                    "and cut into 512 x 512 tiles, row-major: grid 32 x 32 = 1024 patches, 256 tokens each",
            "thumbnail": "after the tiles: the picture (before the tile resize) at the smart size, as a single crop",
            "order": "tile (1, 1), (1, 2), .., (rows, cols), thumbnail; a picture has at most 11 crops and 2,810 image "
                     "tokens up to an aspect of 4:1 (vision_grid_table.json)",
        },
        "resize": "torch's uint8 antialiased bicubic (transformers 5's processor is the torchvision backend): per axis "
                  "int16 fixed-point weights at the axis' own precision, width pass then height pass, uint8 between "
                  "(vision_host.torch_bicubic, bit-exact to torch 2.9.1) — not Pillow's 22-bit filter",
        "normalize": "(x - 127.5) / 127.5 in float32",
        "per_crop_inputs": "vision_host.tower_inputs(crop_u8, position_table) -> patches, pos_table, key_bias, "
                           "unshuffle_idx",
        "tokens": "<|image_start|> + (single: <image> x h w / 4 | tiles: per tile <|img_row_r_col_c|> + <image> x 256, "
                  "then <|img_thumbnail|> + <image> x thumbnail tokens) + <|image_end|>; ids <image> 124907, "
                  "<|img_row_r_col_c|> 124908 + 10 (r - 1) + (c - 1), <|img_thumbnail|> 125008, <|image_start|> 125009, "
                  "<|image_end|> 125010",
        "decoder": "the k-th <image> of the row (counted over every picture and crop in text order) is sent as id "
                   "V + k (V = 128,000) and reads row k of the decoder's image_embeds = the crops' image_embeds rows "
                   "concatenated in crop order (each crop's first h w / 4 rows), fp16",
    }


def write_metadata(out_dir: Path, name: str, model, dtype: str, toy: dict | None, load: dict) -> dict:
    meta = {
        "schema": "d1-vision-tower/1", "kind": "vision-tower", "name": name, "assets": {"main": f"{name}.aimodel"},
        "dtype": {"name": dtype,
                  "weights": {"fp16": "float16", "fp16w32": "float16 storage, read through a float32 cast",
                              "fp32": "float32"}[dtype],
                  "math": "float16" if dtype == "fp16" else "float32",
                  "inputs_output": "float16" if dtype == "fp16" else "float32"},
        "graph": graph_block(model, dtype),
        "config": load.get("config"),
        "host": host_block(),
        "host_files": {"host/position_embedding.safetensors": "the checkpoint's model.vision_tower.vision_model."
                                                              "embeddings.position_embedding.weight, fp32 [256, d]"},
        "source": {"model_definition": "torch (conversion/d1/lfm2_vl_tower.py: Lfm2VlTowerExact, the overlay's "
                                       "Lfm2VlVisionEncoder with the grid as inputs)",
                   "hf_model_id": ed.MODEL["hf_id"], "hf_revision": ed.MODEL["revision"], "license": ed.MODEL["license"],
                   "weights": ("random (the toy: NOT the model's weights)" if toy else
                               {"file": "model.safetensors", **ed.WEIGHTS["model.safetensors"],
                                "keys": "model.vision_tower.vision_model.* + model.multi_modal_projector.*"})},
    }
    if toy:
        meta["toy"] = toy
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return meta


# --------------------------------------------------------------------------- export
def bundle_name(args) -> str:
    if args.name:
        return args.name
    return f"{TOY_PREFIX if args.toy else NAME_PREFIX}_{args.dtype}" + ("" if args.attention == "matmul" else
                                                                         f"_{args.attention}")


def load_model(args):
    import lfm2_vl_tower as T

    if args.toy:
        snap = Path(args.toy_snapshot)
        model = T.Lfm2VlTowerExact.toy(args.toy_seed, snap, attention=args.attention)
        nt = json.loads((snap / "name_table.json").read_text())
        files = {k: ed.sha256_file(snap / k) for k in ("config.json", "model.safetensors", "name_table.json")}
        if files["model.safetensors"] != nt["files"]["model.safetensors"]:
            raise SystemExit(f"{snap}/model.safetensors differs from its name table's sha256")
        toy = {"what": "a toy: lfm2_vl_tower's toy config (vision hidden %d, %d layers, %d heads; projector -> %d) with "
                       "seeded random weights stored BF16 like the checkpoint; NOT the model"
                       % (model.vcfg.hidden_size, model.vcfg.num_hidden_layers, model.vcfg.num_attention_heads,
                          model.text_hidden),
               "seed": args.toy_seed, "snapshot": str(snap), "snapshot_sha256": files}
    else:
        snap = ed.snapshot()
        if ed.sha256_file(snap / "config.json") != ed.CONFIG_SHA256:
            raise SystemExit(f"{snap}/config.json differs from the pinned revision's")
        ck = snap / "model.safetensors"
        if not ck.exists():
            raise SystemExit(f"no {ck}: the 6.2 GB checkpoint is not downloaded")
        got = {"bytes": ck.stat().st_size, "sha256": ed.sha256_file(ck)}
        if got != {k: ed.WEIGHTS["model.safetensors"][k] for k in ("bytes", "sha256")}:
            raise SystemExit(f"{ck} differs from the pinned LFS object: {got}")
        model = T.Lfm2VlTowerExact.from_hf(str(snap), attention=args.attention)
        bad_cfg = {k: (model.load_report["config"][k], v) for k, v in T.EXPECTED_VISION_CONFIG.items()
                   if model.load_report["config"][k] != v}
        if bad_cfg:
            raise SystemExit(f"vision config differs: {bad_cfg}")
        toy = None
    rep = model.load_report
    if (rep["unmapped_checkpoint_keys"] or rep["unexpected_keys"] or rep["missing_module_tensors"]
            or rep["shape_mismatches"] or rep["layout_differs"] or rep["host_position_table"] is None):
        raise SystemExit(f"load mismatch: {json.dumps(rep)}")
    return model, toy


def eager_check(model, typed, dtype: str) -> dict:
    """The typed module against the fp32 one on one random crop (26 x 36 patches), eager, before the export."""
    import numpy as np
    import torch

    import lfm2_vl_tower as T
    import vision_host as vh

    rng = np.random.default_rng(7)
    crop = rng.integers(0, 256, (26 * 16, 36 * 16, 3), dtype=np.uint8)
    ti = vh.tower_inputs(crop, model.host_position_table.numpy())
    args = [torch.from_numpy(ti[k]) for k in T.INPUT_NAMES]
    n = ti["n_tokens"]
    with torch.no_grad():
        ref = model(*args)[:n].double()
        got = typed(*[a.to(T.input_dtype(dtype)) if a.is_floating_point() else a for a in args])[:n].double()
    return {"crop": "random 26 x 36 patches (numpy default_rng(7))", "tokens": n,
            "max_abs_vs_fp32_eager": float((got - ref).abs().max()),
            "cos_vs_fp32_eager": float((got.flatten() @ ref.flatten()) / (got.norm() * ref.norm())),
            "finite": bool(torch.isfinite(got).all())}


def fp16_storage(model) -> dict:
    """How many of the (fp32, BF16-valued) weights fp16 storage changes: overflow to inf, flush to zero, or a rounded
    subnormal. 0 changed = fp16 storage holds every checkpoint value exactly (fp16w32 computes with the fp32 weights)."""
    import torch

    n = changed = to_inf = to_zero = 0
    worst = 0.0
    for _, p in model.named_parameters():
        w = p.detach().float()
        h = w.half().float()
        diff = h != w
        n += w.numel()
        changed += int(diff.sum())
        to_inf += int(torch.isinf(h).logical_and(torch.isfinite(w)).sum())
        to_zero += int((h == 0).logical_and(w != 0).sum())
        if diff.any():
            worst = max(worst, float((h - w).abs()[diff].max()))
    return {"values": n, "changed": changed, "overflow_to_inf": to_inf, "flushed_to_zero": to_zero,
            "max_abs_change": worst, "absmax": float(max(p.detach().abs().max() for _, p in model.named_parameters()))}


def export(args, out_dir: Path, name: str) -> dict:
    import torch
    from toy_graph_check import count_ops

    import lfm2_vl_tower as T
    from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai

    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists: an export never overwrites a bundle (remove it first, on purpose)")
    disk = {"before_load": ed.disk_free()}
    t0 = time.monotonic()
    base, toy = load_model(args)
    storage = fp16_storage(base)
    model = T.typed(base, args.dtype)
    fp32_params = sorted(n for n, p in model.named_parameters() if p.dtype == torch.float32)
    fp16_params = sorted(n for n, p in model.named_parameters() if p.dtype == torch.float16)
    eager = eager_check(base, model, args.dtype)
    spec = model.build_export_spec(T.input_dtype(args.dtype))
    t_loaded = time.monotonic()
    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    print(f"exporting the exact vision tower ({args.dtype}, attention {args.attention}, {name}) ...", flush=True)
    prog = export_to_coreai(model, spec["reference_inputs"], dynamic_shapes=spec["dynamic_shapes"],
                            input_names=spec["input_names"], output_names=spec["output_names"],
                            state_names=spec["state_names"], externalize_modules=specs)
    t_converted = time.monotonic()
    prog.optimize()
    t_exported = time.monotonic()
    ops = count_ops(prog)
    print(f"converted in {t_converted - t_loaded:.1f}s, optimized in {t_exported - t_converted:.1f}s, "
          f"{ops.get('ops')} ops", flush=True)
    out_dir.mkdir(parents=True)
    import coreai.runtime as rt
    from safetensors.torch import save_file

    aimodel = out_dir / f"{name}.aimodel"
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    t_saved = time.monotonic()
    (out_dir / "host").mkdir()
    table = out_dir / "host" / "position_embedding.safetensors"
    save_file({"position_embedding": base.host_position_table.float().contiguous()}, str(table))
    meta = write_metadata(out_dir, name, model, args.dtype, toy, base.load_report)
    lic = ed.copy_verbatim(ed.snapshot(), out_dir, {"LICENSE": ed.LICENSE_SHA256})
    disk["after_save"] = ed.disk_free()
    mlirb = aimodel / "main.mlirb"
    rep = base.load_report
    rec = {"bundle": str(out_dir), "name": name, "aimodel": str(aimodel), "toy": toy, "dtype": args.dtype,
           "attention": args.attention, "load_report": rep,
           "load_summary": {"checkpoint_keys": rep["checkpoint_keys_under_prefixes"], "expected_keys": rep["expected_keys"],
                            "unread": len(rep["unmapped_checkpoint_keys"]) + len(rep["unexpected_keys"]),
                            "missing": len(rep["missing_module_tensors"]), "shape_mismatches": len(rep["shape_mismatches"]),
                            "layout_differs": len(rep["layout_differs"]), "host_position_table": rep["host_position_table"]},
           "fp16_storage": storage,
           "spec": {"input_names": list(spec["input_names"]), "output_names": list(spec["output_names"]),
                    "reference_inputs": {k: [list(v.shape), str(v.dtype).replace("torch.", "")]
                                         for k, v in spec["reference_inputs"].items()}, "dynamic": "none"},
           "graph": meta["graph"], "fp32_params": len(fp32_params), "fp16_params": len(fp16_params),
           "parameters": int(sum(p.numel() for p in base.parameters())),
           "parametrized": sorted({n.split(".parametrizations.")[0] for n, _ in model.named_parameters()
                                   if ".parametrizations." in n})[:4],
           "eager_check": eager, "ops": ops,
           "seconds": {"load": t_loaded - t0, "export": t_converted - t_loaded, "optimize": t_exported - t_converted,
                       "save": t_saved - t_exported, "total": time.monotonic() - t0},
           "du_aimodel": ed.du(aimodel), "du_bundle": ed.du(out_dir),
           "aimodel_files": {str(p.relative_to(aimodel)): p.stat().st_size for p in sorted(aimodel.rglob("*")) if p.is_file()},
           "main_mlirb": {"bytes": mlirb.stat().st_size, "sha256": ed.sha256_file(mlirb)},
           "host_table": {"file": str(table), "sha256": ed.sha256_file(table),
                          "shape": list(base.host_position_table.shape)},
           "metadata_sha256": ed.sha256_file(out_dir / "metadata.json"), "license_sha256": lic["LICENSE"], "disk": disk}
    print(f"bundle ready: {out_dir} ({rec['du_aimodel']}, main.mlirb {rec['main_mlirb']['bytes']:,} B, eager "
          f"{args.dtype} vs fp32 max|d| {eager['max_abs_vs_fp32_eager']:.2e}, total {rec['seconds']['total']:.1f}s; "
          f"load {rec['load_summary']}; fp16 storage changes {storage['changed']} of {storage['values']:,} values)",
          flush=True)
    return rec


def aot_compile(aimodel: Path, out_dir: Path) -> tuple[Path, float, dict]:
    target = out_dir / f"{aimodel.stem}.h16c.aimodelc"
    if not os.environ.get("DEVELOPER_DIR"):
        sys.exit("set DEVELOPER_DIR to the Xcode 27 RC (its Metal toolchain carries coreai-build)")
    cb = subprocess.run(["xcrun", "-f", "coreai-build"], capture_output=True, text=True)
    if cb.returncode != 0 or not cb.stdout.strip():
        sys.exit("xcrun -f coreai-build failed:\n" + cb.stderr)
    if target.exists():
        sys.exit(f"{target} exists: an AOT compile never overwrites an asset (remove it first, on purpose)")
    out_dir.mkdir(parents=True, exist_ok=True)
    disk = {"before": ed.disk_free()}
    cmd = [cb.stdout.strip(), "compile", str(aimodel), "--output", str(out_dir), *AOT_FLAGS]
    print(" ".join(cmd), flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    secs = time.monotonic() - t0
    disk["after"] = ed.disk_free()
    info = {"coreai_build": cb.stdout.strip(), "command": cmd, "disk": disk, "returncode": proc.returncode,
            "stdout_tail": proc.stdout.splitlines()[-20:], "stderr_tail": proc.stderr.splitlines()[-40:]}
    if proc.returncode != 0 or not target.exists():
        info["failed"] = True
        return target, secs, info
    mh = target / "main.hash"
    info["main_hash_hex"] = mh.read_bytes().hex() if mh.exists() else None
    info["files"] = {str(p.relative_to(target)): p.stat().st_size for p in sorted(target.rglob("*")) if p.is_file()}
    info["resources_bin_bytes"] = sum(v for k, v in info["files"].items() if k.endswith("resources.bin"))
    info["runtime_cache_entry"] = str(ed.coreai_cache_dir() / info["main_hash_hex"]) if info["main_hash_hex"] else None
    stats = target / "stats.json"
    info["stats"] = json.loads(stats.read_text()) if stats.exists() else None
    return target, secs, info


def main() -> None:
    import lfm2_vl_tower as T

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dtype", default="fp16w32", choices=list(T.DTYPES))
    ap.add_argument("--attention", default="matmul", choices=list(T.ATTENTIONS))
    ap.add_argument("--out-dir", default=str(LANE / "exports"),
                    help="bundles go to <out-dir>/vision/<name>/ (the toy's to toy_vision/), AOT assets to "
                         "<out-dir>/vision_aotc/ (toy_vision_aotc/)")
    ap.add_argument("--name", help="override the generated bundle directory and asset name")
    ap.add_argument("--toy", action="store_true", help="the toy snapshot through the same path (no checkpoint)")
    ap.add_argument("--toy-seed", type=int, default=T.TOY_SEED)
    ap.add_argument("--toy-snapshot", default=str(TOY_SNAPSHOT), help="--toy: lfm2_vl_tower.py write-toy's directory")
    ap.add_argument("--skip-export", action="store_true", help="reuse the saved .aimodel (with --aot)")
    ap.add_argument("--aot", action="store_true", help="compile the .aimodel for the Mac GPU (h16c, no efr)")
    ap.add_argument("--record", help="write the export / AOT record JSON here (never overwritten)")
    args = ap.parse_args()
    name = bundle_name(args)
    sub = "toy_vision" if args.toy else "vision"
    out_dir = Path(args.out_dir) / sub / name
    aot_dir = Path(args.out_dir) / f"{sub}_aotc"
    if args.record and Path(args.record).exists():
        sys.exit(f"{args.record} exists: records are never overwritten")
    record: dict = {"name": name, "dtype": args.dtype, "attention": args.attention, "toy": args.toy,
                    "toy_seed": args.toy_seed if args.toy else None, "hf_id": ed.MODEL["hf_id"],
                    "revision": ed.MODEL["revision"], "argv": sys.argv[1:], "pid": os.getpid(), "started": ed.now(),
                    "script_sha256": ed.sha256_file(Path(__file__).resolve()),
                    "module_sha256": ed.sha256_file(HERE / "lfm2_vl_tower.py")}

    def save_record() -> None:
        if args.record:
            Path(args.record).parent.mkdir(parents=True, exist_ok=True)
            Path(args.record).write_text(json.dumps(record, indent=1) + "\n")

    if not args.skip_export:
        record["export"] = export(args, out_dir, name)
        save_record()
    if args.aot:
        aimodelc, secs, info = aot_compile(out_dir / f"{name}.aimodel", aot_dir)
        record["aot"] = {"aimodelc": str(aimodelc), "flags": AOT_FLAGS, "seconds": secs, **info}
        if info.get("failed"):
            record["finished"] = ed.now()
            save_record()
            print("\n".join(info["stderr_tail"]), file=sys.stderr)
            sys.exit(f"coreai-build failed (exit {info['returncode']}) after {secs:.1f} s; record {args.record}")
        record["aot"].update({"du_aimodelc": ed.du(aimodelc), "digest": ed.tree_digest(aimodelc)})
        print(f"asset: {aimodelc} (compile {secs:.1f} s, {record['aot']['du_aimodelc']}, "
              f"{record['aot']['digest']['bytes']:,} B, resources.bin {info['resources_bin_bytes']:,} B, "
              f"main.hash {info.get('main_hash_hex')})", flush=True)
    record["finished"] = ed.now()
    save_record()
    if args.record:
        print(f"record: {args.record}")


if __name__ == "__main__":
    main()
