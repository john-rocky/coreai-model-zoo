#!/usr/bin/env python3
"""Export d1-omni-600M's vision graph (one crop: the SigLIP2 NaFlex tower and the 2x2 unshuffle projector) as a Core AI
bundle.

    python3 conversion/d1_omni/export_vision.py --precision fp32
    python3 conversion/d1_omni/export_vision.py --precision wfp16
    python3 conversion/d1_omni/export_vision.py --precision fp16

Run with the shared export venv (coreai-models/.venv: torch 2.9.0, torchvision 0.24.0, coreai-torch 0.4.1,
coreai-core 1.0.0b2). The graph is d1_omni_vision.D1Vision (plain torch, no transformers), its weights the pinned
snapshot's model.safetensors (sha256 verified on load).

Order: load_d1_vision(snapshot, precision, verify_sha256=True) -> torch.export with one crop's inputs
(host.image_crops_inputs of img_01, a padded crop) -> run_decompositions(get_decomp_table()) -> the export gate ->
TorchConverter (one function, main) -> optimize -> save_asset.

Export gate: the decomposed program on every crop of every image of the round 5 reference (img_01..03, imgm_01..12
and card_cats: 62 crops; the host's inputs, asserted equal to the publisher's preprocess output in the reference npz)
against the reference prefix (cos, max |d|, min row cos, in float64), and against the eager module of the same
precision (what export + decomposition changed; on every crop, or on the first crop of each image at fp16, which
torch runs slowly on the CPU). fp32 stops before conversion unless cos >= 0.999999 on every image and the program
equals the eager module; wfp16 / fp16 are measured and converted (the end-to-end runtime gate decides).

Layout ($ZOO_WORK_ROOT/_d1_omni/bundles/d1-omni-600m/macos/vision-<precision>/, built in a staging folder and renamed
at the end; an existing folder is never replaced):
  d1_omni_vision_<precision>.aimodel   the bundle, function main
  metadata.json                       the vision contract a host reads (inputs, output, crops, the position table)
  position_table.f32                  the checkpoint's 16x16x768 position table, float32 little-endian, row-major
  reference.json                      the gate's images and crops (grid, tokens) with the reference npz and sha256
  provenance/export-manifest.json     bytes, op counts, seconds, environment, input hashes
  provenance/export-gate.json         every image of the export gate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import hf_snapshot, work_path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import d1_omni_vision as dv  # noqa: E402
import host  # noqa: E402
from export_decide import (AUTHOR, FAMILY, LICENSE, element_type, environment, file_inventory,  # noqa: E402
                           parameter_bytes, sha256_file, verify_source, write_json)
from vision_check import PREFIX_BAR, compare, load_images  # noqa: E402

WORK = work_path("_d1_omni")
INPUT_NAMES = list(host.VISION_INPUT_NAMES)
OUTPUT_NAMES = ["prefix"]
EXAMPLE_IMAGE = "img_01"
CODE_FILES = ("export_vision.py", "d1_omni_vision.py", "host.py", "vision_check.py")


def bundle_dir(precision: str) -> Path:
    return WORK / "bundles" / FAMILY / "macos" / f"vision-{precision}"


def bundle_name(precision: str) -> str:
    return f"d1_omni_vision_{precision}.aimodel"


def as_tensors(crop: dict) -> dict:
    return {k: torch.from_numpy(np.ascontiguousarray(crop[k])) for k in INPUT_NAMES}


def prefix_of(out) -> np.ndarray:
    if isinstance(out, (tuple, list)):
        assert len(out) == 1, len(out)
        out = out[0]
    return out.detach().to(torch.float32).numpy().reshape(dv.MAX_TOKENS, -1).copy()


def image_crops(images: list[dict], table: np.ndarray) -> list[list[dict]]:
    """The host's crop inputs of every image, each asserted equal to the reference's preprocess output."""
    from PIL import Image

    out = []
    for img in images:
        with Image.open(img["file"]) as pil:
            pil.load()
            crops = host.image_crops_inputs(pil, table)
        z = img["npz"]
        for k, crop in enumerate(crops):
            if not np.array_equal(crop["pixel_values"][0], z["pixel_values"][k]):
                raise SystemExit(f"{img['id']} crop {k}: host pixel_values differ from the reference npz")
        out.append(crops)
    return out


def export_gate(program, model, images, crops_by_image, eager_all: bool) -> tuple[list[dict], dict]:
    module = program.module()
    records = []
    for img, crops in zip(images, crops_by_image):
        t0 = time.perf_counter()
        parts, eager_diffs = [], []
        for k, crop in enumerate(crops):
            with torch.no_grad():
                got = prefix_of(module(**as_tensors(crop)))
                if eager_all or k == 0:
                    eager = prefix_of(model(**as_tensors(crop)))
                    eager_diffs.append({"crop": k, "max_abs": float(np.max(np.abs(got[:crop["tokens"]].astype(np.float64)
                                                                                  - eager[:crop["tokens"]]))),
                                        "bit_equal": bool(np.array_equal(got[:crop["tokens"]], eager[:crop["tokens"]]))})
            parts.append(got[:crop["tokens"]])
        prefix = np.concatenate(parts)
        stats = compare(prefix, img["npz"]["prefix"])
        record = {"id": img["id"], "crops": len(crops), "prefix_rows": int(prefix.shape[0]), "prefix": stats,
                  "cos_pass": stats["cos"] >= PREFIX_BAR["cos"], "finite": bool(np.isfinite(prefix).all()),
                  "vs_eager": {"crops_compared": len(eager_diffs), "max_abs": max(d["max_abs"] for d in eager_diffs),
                               "bit_equal": sum(d["bit_equal"] for d in eager_diffs), "detail": eager_diffs},
                  "seconds": time.perf_counter() - t0}
        records.append(record)
        print(f"  {img['id']}: {len(crops)} crops P {prefix.shape[0]} cos {stats['cos']:.9f} max {stats['max_abs']:.3e} "
              f"min-row {stats['min_row_cos']:.9f}; vs eager {record['vs_eager']['max_abs']:.1e} "
              f"({record['vs_eager']['bit_equal']}/{len(eager_diffs)} bit) {record['seconds']:.1f} s", flush=True)
    summary = {"images": len(records), "crops": sum(r["crops"] for r in records),
               "min_cos": min(r["prefix"]["cos"] for r in records),
               "min_row_cos": min(r["prefix"]["min_row_cos"] for r in records),
               "max_abs": max(r["prefix"]["max_abs"] for r in records),
               "worst_image": min(records, key=lambda r: r["prefix"]["cos"])["id"],
               "cos_pass": f"{sum(r['cos_pass'] for r in records)}/{len(records)}",
               "finite": all(r["finite"] for r in records),
               "vs_eager_max_abs": max(r["vs_eager"]["max_abs"] for r in records),
               "vs_eager_bit_equal": f"{sum(r['vs_eager']['bit_equal'] for r in records)}/"
                                     f"{sum(r['vs_eager']['crops_compared'] for r in records)}"}
    return records, summary


def convert(program, description: str, out: Path) -> dict:
    from coreai.runtime import AIModelAssetMetadata
    from coreai_torch import TorchConverter

    t0 = time.perf_counter()
    converted = (TorchConverter()
                 .add_exported_program(exported_program=program, input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
                                       entrypoint_name="main")
                 .to_coreai())
    to_coreai_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    converted.optimize()
    optimize_s = time.perf_counter() - t0
    module = converted._mlir_module
    assert module.operation.verify()
    ops, result_types = Counter(), Counter()

    def inspect(operation):
        ops[operation.name] += 1
        for result in operation.results:
            result_types[element_type(str(result.type))] += 1
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    inspect(child.operation)

    inspect(module.operation)
    metadata = AIModelAssetMetadata()  # the constructor ignores keyword arguments: set the fields
    metadata.author, metadata.license, metadata.model_description = AUTHOR, LICENSE, description
    assert (metadata.author, metadata.license, metadata.model_description) == (AUTHOR, LICENSE, description)
    t0 = time.perf_counter()
    converted.save_asset(out, metadata)
    return {"op_counts": dict(sorted(ops.items())), "ops_total": sum(ops.values()),
            "result_element_types": dict(sorted(result_types.items())),
            "seconds": {"to_coreai": to_coreai_s, "optimize": optimize_s, "save": time.perf_counter() - t0}}


def io_contract() -> dict:
    return {
        "inputs": {
            "pixel_values": {"dtype": "float32", "shape": [1, 1024, 768],
                             "values": "the crop's 16x16 patches row-major over (ph, pw), each [py][px][c], "
                                       "(x - 127.5) / 127.5; 0.0 past ph*pw"},
            "pos_embed": {"dtype": "float32", "shape": [1, 1024, 768],
                          "values": "position_table resized to (ph, pw), bilinear + antialias (host.position_embeddings"
                                    " / position_embeddings_numpy), row-major; 0.0 past ph*pw"},
            "patch_mask": {"dtype": "float32", "shape": [1, 1024], "values": "1.0 on the ph*pw patches, 0.0 after"},
            "unshuffle_index": {"dtype": "int32", "shape": [256, 4],
                                "values": "token (i, j) = row i * (pw/2) + j: patches 2i*pw+2j, 2i*pw+2j+1, "
                                          "(2i+1)*pw+2j, (2i+1)*pw+2j+1; 0 past (ph/2)(pw/2) (host.unshuffle_index)"}},
        "outputs": {"prefix": {"dtype": "float32", "shape": [1, 256, 1024],
                               "read": "rows [0, (ph/2)(pw/2)): the crop's prefix embeddings"}},
    }


def bundle_metadata(name: str, precision: str) -> dict:
    return {
        "metadata_version": "0.2",
        "kind": "encoder",
        "vision": {
            "role": "d1-omni image prefix: one crop per call", "precision": precision,
            "functions": {"main": "main"}, **io_contract(),
            "max_patches": dv.MAX_PATCHES, "max_tokens": dv.MAX_TOKENS, "patch": dv.PATCH,
            "crops": "vision.preprocess(): an image whose rounded area exceeds 2 x 512^2 is cut into the 512 px tiles "
                     "of the closest grid (2 to 10 tiles, row-major) plus a thumbnail, else the thumbnail alone; each "
                     "crop resized by torchvision (bilinear, antialias) from the RGB image (host.layout / image_crops)",
            "prefix": "each crop's first (ph/2)(pw/2) output rows, crops in order, images in request order; this is "
                      "the decision graph's prefix_embeds [0, P)",
            "position_table": {"file": "position_table.f32", "shape": [16, 16, 768], "dtype": "float32",
                               "byte_order": "little", "layout": "row-major (row, column, channel)",
                               "resize": "per crop to (ph, pw): bilinear, align_corners false, antialias (torch's "
                                         "float32 kernel: width pass then height pass, fused multiply-add per tap; "
                                         "host.position_embeddings_numpy)"},
            "host_reference": "conversion/d1_omni/host.py (image_crops_inputs, vision_prefix, image_request_rows)",
        },
        "source": {"hf_model_id": host.MODEL_ID, "hf_revision": host.MODEL_SHA, "license": LICENSE},
        "assets": {"main": name},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--precision", choices=dv.PRECISIONS, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_grad_enabled(False)
    out_dir = bundle_dir(args.precision)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists: an existing bundle folder is never replaced")
    stage = out_dir.parent / f".staging-{out_dir.name}-{os.getpid()}"
    stage.mkdir(parents=True)
    name = bundle_name(args.precision)
    print(f"pid {os.getpid()} staging {stage}", flush=True)

    snapshot = Path(hf_snapshot(host.MODEL_ID, revision=host.MODEL_SHA))
    source = verify_source(snapshot)
    images, ref_info = load_images()
    t0 = time.perf_counter()
    model, load_record = dv.load_d1_vision(snapshot, args.precision, verify_sha256=True)
    load_s = time.perf_counter() - t0
    table = model.position_table.numpy()
    if not np.array_equal(table, host.load_position_table(snapshot / "model.safetensors")):
        raise SystemExit("the module's position table differs from host.load_position_table")
    crops_by_image = image_crops(images, table)
    example_index = next(i for i, img in enumerate(images) if img["id"] == EXAMPLE_IMAGE)
    example = crops_by_image[example_index][0]
    print(f"images {len(images)}, crops {sum(len(c) for c in crops_by_image)}; example {EXAMPLE_IMAGE} crop 0 grid "
          f"{example['grid']} ({example['tokens']} tokens)", flush=True)
    weights = parameter_bytes(model)
    from coreai_torch import get_decomp_table

    t0 = time.perf_counter()
    exported = torch.export.export(model, args=(), kwargs=as_tensors(example))
    export_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    program = exported.run_decompositions(get_decomp_table())
    decompose_s = time.perf_counter() - t0
    print(f"torch.export {export_s:.1f} s, decompositions {decompose_s:.1f} s", flush=True)

    t0 = time.perf_counter()
    records, summary = export_gate(program, model, images, crops_by_image, eager_all=args.precision != "fp16")
    gate_s = time.perf_counter() - t0
    if args.precision == "fp32":
        ok = summary["cos_pass"].split("/")[0] == summary["cos_pass"].split("/")[1] and summary["finite"] and \
             summary["vs_eager_bit_equal"].split("/")[0] == summary["vs_eager_bit_equal"].split("/")[1]
    else:
        ok = summary["finite"]
    status = "PASS" if ok and summary["cos_pass"].split("/")[0] == summary["cos_pass"].split("/")[1] else (
        "MEASURED" if args.precision != "fp32" else "FAIL")
    print(f"export gate {status} in {gate_s:.0f} s: {json.dumps(summary)}", flush=True)
    gate_record = {"status": status, "stage": "export (torch-exported + decomposed program, CPU, before conversion)",
                   "precision": args.precision, "bar": {**PREFIX_BAR, "program_equals_eager": "fp32: bit for bit"},
                   "bar_rule": ("fp32 stops before conversion if this misses" if args.precision == "fp32" else
                                "measured, converted whatever it says; the end-to-end runtime gate decides"),
                   "reference": ref_info, "example": f"{EXAMPLE_IMAGE} crop 0", "summary": summary, "seconds": gate_s,
                   "environment": environment(("torch", "torchvision", "coreai-torch", "coreai-core", "numpy",
                                               "safetensors", "pillow")),
                   "images": records}
    write_json(stage / "provenance" / "export-gate.json", gate_record)
    if args.precision == "fp32" and status != "PASS":
        raise SystemExit(f"fp32 export gate FAIL before conversion (record {stage / 'provenance' / 'export-gate.json'})")

    description = (f"d1-omni-600M vision graph (SigLIP2 NaFlex tower + 2x2 unshuffle projector), {args.precision}, "
                   f"one crop of up to 1024 patches -> 256 prefix rows, function main; source "
                   f"{host.MODEL_ID}@{host.MODEL_SHA}")
    converted = convert(program, description, stage / name)
    gathers = {k: v for k, v in converted["op_counts"].items() if "gather" in k or "index" in k or "embedding" in k}
    print(f"converted {converted['seconds']} ops {converted['ops_total']}; gather-like ops {gathers}", flush=True)
    (stage / "position_table.f32").write_bytes(np.ascontiguousarray(table, dtype="<f4").tobytes())
    metadata = bundle_metadata(name, args.precision)
    metadata["vision"]["precision_detail"] = {
        "weights": "fp32" if args.precision == "fp32" else "fp16",
        "compute": {"fp32": "fp32", "wfp16": "fp32 (each fp16 weight cast at its use)",
                    "fp16": "fp16 (LayerNorm, the attention softmax and the projector's gelu in fp32)"}[args.precision]}
    metadata["vision"]["position_table"]["sha256"] = sha256_file(stage / "position_table.f32")
    write_json(stage / "metadata.json", metadata)
    reference = {"schema": "d1-omni-vision-reference/1", "model": host.MODEL_ID, "revision": host.MODEL_SHA,
                 "reference": ref_info,
                 "oracle": "the publisher's Vision (vision.py on transformers 5.19, sdpa) in fp32 on the CPU, hooked "
                           "inside D1OmniModel.probabilities(); prefix = its output for the request's one image",
                 "images": [{"id": img["id"], "public": img["public"], "file": str(Path(img["file"]).relative_to(WORK)),
                             "npz": img["npz_file"], "npz_sha256": img["npz_sha256"],
                             "prefix_rows": int(img["npz"]["prefix"].shape[0]),
                             "crops": [{"grid": list(c["grid"]), "tokens": c["tokens"]} for c in crops]}
                            for img, crops in zip(images, crops_by_image)]}
    write_json(stage / "reference.json", reference)

    files = file_inventory(stage / name)
    bundle_bytes = sum(f["bytes"] for f in files)
    record = {
        "status": status, "model": host.MODEL_ID, "model_sha": host.MODEL_SHA, "bundle": name,
        "format": "JIT .aimodel (function main)", "intended_runtime": "macos", "aot": False,
        "precision": args.precision, **io_contract(), "files": files, "bytes": bundle_bytes,
        "module_weights": weights, "bytes_over_module_parameter_bytes": bundle_bytes / weights["parameters"],
        "op_counts": converted["op_counts"], "ops_total": converted["ops_total"], "gather_like_ops": gathers,
        "result_element_types": converted["result_element_types"],
        "seconds": {"weights_load": load_s, "torch_export": export_s, "decompositions": decompose_s,
                    "export_gate": gate_s, **converted["seconds"], "total": time.perf_counter() - started},
        "export_gate": {"status": status, "record": "provenance/export-gate.json", **summary},
        "weights_record": load_record, "source_sha256": source,
        "reference_json_sha256": sha256_file(stage / "reference.json"),
        "metadata_json_sha256": sha256_file(stage / "metadata.json"),
        "position_table_sha256": sha256_file(stage / "position_table.f32"),
        "code_sha256": {f: sha256_file(HERE / f) for f in CODE_FILES},
        "runtime_gate": "NOT RUN — aot_vision.py, then vision_check.py --stage runtime",
        "environment": environment(("torch", "torchvision", "coreai-torch", "coreai-core", "numpy", "safetensors")),
    }
    write_json(stage / "provenance" / "export-manifest.json", record)
    os.rename(stage, out_dir)
    print(json.dumps({"status": status, "folder": str(out_dir), "bundle": name, "bytes": bundle_bytes,
                      "module_parameter_bytes": weights["parameters"], "ops": converted["ops_total"],
                      "seconds": {k: round(v, 1) for k, v in record["seconds"].items()}}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
