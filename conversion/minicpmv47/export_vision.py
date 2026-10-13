#!/usr/bin/env python3
"""Export the MiniCPM-V-4.7-1B vision tower (one 448x448 thumbnail -> 64 LLM rows) to a Core AI bundle.

The graph is `minicpmv47_vision.VisionExport` over `MiniCPMV47Vision.bake_constants(32)` (round 2: the
position ids, the window index and the 2x2 merge order are int32 buffers, the window merger is one
batched attention, no data-dependent op reaches torch.export):

    pixel_values [1, 3, 448, 448] -> image_features [64, 1024]      (fixed shape, no state)

Three forms, all from the checkpoint's 459 tower + merger tensors:

    fp16      weights fp16, math fp16, pixel_values / image_features fp16
    fp16w32   weights stored fp16 and read through a `.float()` cast, math and the int32 buffers fp32,
              pixel_values / image_features fp32 (`export_qwen38vl_pipelined.fp16_storage_fp32_compute`,
              the form decider-2b-vision ships its towers in, `--vision-dtype fp16w32`)
    int8lin   every nn.Linear int8 per-block-16 `symmetric_with_clipping` (weight only; block 16 because
              the encoder MLP's fc2 in_features 4304 = 16 x 269 is not divisible by 32), LayerNorm /
              the patch Conv2d / the position Embedding fp16, math fp16, I/O fp16
              (MiniCPM-V-4.6's `vision_quant_config`)

Exported as `export_minicpmv46_vision.py` exports 4.6's tower: `export_to_coreai(..., dynamic_shapes=None,
externalize_modules=[])`, `prog.optimize()`. Bundle `<exports>/minicpmv47_vision_<mode>/` = `<name>.aimodel`
+ `metadata.json` (kind vision-encoder, I/O, grid, dtype, the host preprocess). `--record` writes the export
record (quantized-module counts, seconds, per-file sha256 of the .aimodel).

    QW coreai-models/.venv/bin/python conversion/minicpmv47/export_vision.py fp16w32 --record <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _paths import exports_dir, hf_snapshot  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")

HF_ID = "openbmb/MiniCPM-V-4.7-1B"
REVISION = "984a89aafcd5160164f4f0d04b4be0f4eec03cb8"
SIDE = 448
GRID = 32


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(path: Path) -> dict:
    files = sorted(p for p in path.rglob("*") if p.is_file())
    per = {str(p.relative_to(path)): sha256_file(p) for p in files}
    tree = hashlib.sha256("".join(f"{k}\0{v}\n" for k, v in per.items()).encode()).hexdigest()
    return {"bytes": sum(p.stat().st_size for p in files), "tree_sha256": tree,
            "files": {k: {"bytes": (path / k).stat().st_size, "sha256": v} for k, v in per.items()}}


def vision_quant_config() -> dict:
    """int8 per-block-16 symmetric_with_clipping on nn.Linear weights; LayerNorm / Conv2d / Embedding stay
    fp16 (MiniCPM-V-4.6 `export_minicpmv46_vision.vision_quant_config`, verbatim)."""
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {"weight": {
                "dtype": "int8", "qscheme": "symmetric_with_clipping",
                "granularity": {"type": "per_block", "block_size": 16, "axis": 1}}},
            "op_input_spec": None, "op_output_spec": None,
        },
        "module_type_configs": {
            "torch.nn.modules.normalization.LayerNorm": None,
            "torch.nn.modules.conv.Conv2d": None,
            "torch.nn.modules.sparse.Embedding": None,
        },
    }


def export(args, out_dir: Path, name: str) -> dict:
    import torch
    import torch.nn.utils.parametrize as P
    from minicpmv47_vision import MiniCPMV47Vision, VisionExport

    from coreai_models.export.macos import export_to_coreai

    snap = Path(hf_snapshot(HF_ID, revision=REVISION))
    dtype = torch.float32 if args.mode == "fp16w32" else torch.float16
    t0 = time.monotonic()
    core = MiniCPMV47Vision.from_checkpoint(snap, dtype).bake_constants(GRID)
    model = VisionExport(core).eval()
    report = core.load_report
    print(f"[load] {json.dumps(report)}", flush=True)
    quant = None
    if args.mode == "fp16w32":
        from export_qwen38vl_pipelined import fp16_storage_fp32_compute

        model = fp16_storage_fp32_compute(model)
        params = [(n, p) for n, p in model.named_parameters()]
        quant = {"form": "fp16 storage, fp32 compute (parametrization .float())",
                 "parametrized_tensors": sum(1 for m in model.modules() for k in getattr(m, "parametrizations", {})),
                 "stored_dtypes": sorted({str(p.dtype) for _, p in params}),
                 "buffers": {n: str(b.dtype) for n, b in model.named_buffers()}}
        print(f"[fp16w32] {quant['parametrized_tensors']} parameters stored fp16 behind a .float() cast; "
              f"buffers {quant['buffers']}", flush=True)
    t_loaded = time.monotonic()
    pv = torch.zeros(1, 3, SIDE, SIDE, dtype=dtype)
    if args.mode == "int8lin":
        from coreai_models.export.compression import quantize_pytorch_model

        print("[quant] nn.Linear int8 per-block-16 symmetric_with_clipping; LayerNorm / Conv2d / Embedding fp16 ...",
              flush=True)
        model = quantize_pytorch_model(model, (pv,), None, vision_quant_config())
        lins = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
        q8 = [n for n, m in lins if P.is_parametrized(m, "weight")]
        fp = [n for n, m in lins if not P.is_parametrized(m, "weight")]
        others = [n for n, m in model.named_modules()
                  if isinstance(m, (torch.nn.LayerNorm, torch.nn.Conv2d, torch.nn.Embedding))
                  and P.is_parametrized(m, "weight")]
        quant = {"linear": vision_quant_config()["global_config"]["op_state_spec"]["weight"],
                 "excluded_types": sorted(vision_quant_config()["module_type_configs"]),
                 "int8_linear_modules": len(q8), "fp16_linear_modules": fp,
                 "quantized_non_linear_modules": others, "seconds": time.monotonic() - t_loaded}
        print(f"[quant] int8 linears {len(q8)}, fp16 linears {len(fp)} {fp[:4]}, quantized non-linear "
              f"{len(others)}, {quant['seconds']:.0f}s", flush=True)
        if fp or others or len(q8) != 170:
            sys.exit(f"int8lin: the quantized set differs from the recipe: {json.dumps(quant)}")
    t_quantized = time.monotonic()

    print(f"[export] vision ({args.mode}) -> Core AI ...", flush=True)
    prog = export_to_coreai(model, {"pixel_values": pv}, dynamic_shapes=None, input_names=("pixel_values",),
                            output_names=("image_features",), state_names=None, externalize_modules=[])
    print("[export] optimizing ...", flush=True)
    prog.optimize()
    t_exported = time.monotonic()

    import coreai.runtime as rt

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aimodel = out_dir / f"{name}.aimodel"
    md = rt.AIModelAssetMetadata()   # constructor kwargs are dropped: set attributes
    md.author = "openbmb (weights); community Core AI port"
    md.license = "Apache-2.0"
    md.model_description = (f"MiniCPM-V-4.7-1B vision tower, 448x448 thumbnail -> 64 rows, {args.mode} "
                            f"(Community port - NOT an Apple model)")
    print(f"[save] {aimodel} ...", flush=True)
    prog.save_asset(aimodel, md)
    t_saved = time.monotonic()
    io_dtype = "float32" if args.mode == "fp16w32" else "float16"
    meta = {
        "metadata_version": "0.2", "kind": "vision-encoder", "name": name,
        "assets": {"main": f"{name}.aimodel"},
        "vision": {
            "input": {"name": "pixel_values", "shape": [1, 3, SIDE, SIDE], "dtype": io_dtype},
            "output": {"name": "image_features", "shape": [64, 1024], "dtype": io_dtype,
                       "rows": "row-major 8x8 merged grid = the decoder's image_embeds rows (cast to fp16)"},
            "grid": {"patch": 14, "patches": [GRID, GRID], "after_window_merger": [16, 16], "merged": [8, 8]},
            "dtype": args.mode,
            "weights": {"fp16": "fp16, math fp16", "fp16w32": "stored fp16, math fp32",
                        "int8lin": "nn.Linear int8 per-block-16 symmetric_with_clipping, the rest fp16, math fp16"}[args.mode],
            "host_preprocess": "RGB -> Pillow BICUBIC resize to 448x448 (aspect not kept) -> x/255 -> (x - 0.5)/0.5 "
                               "-> CHW [1,3,448,448] (conversion/minicpmv47/host.py preprocess)",
        },
        "source": {"model_definition": "torch", "hf_model_id": HF_ID, "hf_revision": REVISION},
        "author_note": "Community port - NOT an Apple model",
        "compilation": {"date": datetime.now(timezone.utc).isoformat(), "targets": []},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    rec = {"bundle": str(out_dir), "name": name, "aimodel": str(aimodel), "load_report": report,
           "io_dtype": io_dtype, "quantization": quant,
           "seconds": {"load": t_loaded - t0, "quantize": t_quantized - t_loaded,
                       "export_optimize": t_exported - t_quantized, "save": t_saved - t_exported,
                       "total": time.monotonic() - t0},
           "aimodel_digest": tree_digest(aimodel), "metadata_sha256": sha256_file(out_dir / "metadata.json")}
    print(f"[done] {out_dir}: .aimodel {rec['aimodel_digest']['bytes']:,} B tree sha256 "
          f"{rec['aimodel_digest']['tree_sha256']} ({rec['seconds']['total']:.0f}s)", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["fp16", "fp16w32", "int8lin"])
    ap.add_argument("--out-root", default=str(exports_dir()))
    ap.add_argument("--record", help="write the export record JSON here")
    args = ap.parse_args()
    name = f"minicpmv47_vision_{args.mode}"
    import importlib.metadata as md

    import torch
    record = {"schema": "minicpmv47-export-vision/1", "mode": args.mode, "name": name, "hf_id": HF_ID,
              "revision": REVISION, "export": export(args, Path(args.out_root) / name, name),
              "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                           **{p: md.version(p) for p in ("coreai-core", "coreai-torch", "coreai-opt")},
                           "platform": platform.platform()},
              "script": {"path": "conversion/minicpmv47/export_vision.py", "sha256": sha256_file(Path(__file__))},
              "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        Path(args.record).write_text(json.dumps(record, indent=1) + "\n")
        print(f"[record] {args.record}", flush=True)


if __name__ == "__main__":
    main()
