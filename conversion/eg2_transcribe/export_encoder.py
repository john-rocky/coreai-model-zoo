#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.9.0",
#     "coreai-torch==0.4.1",
#     "coreai-core==1.0.0b2",
#     "coreai-opt==0.2.1",
#     "torchao==0.17.0",
#     "safetensors>=0.7.0",
#     "numpy>=2.2",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
# Community port — NOT an Apple or Google model.
"""Export the EmbeddingGemma 2 audio encoder to a static Core AI ``.aimodel`` (one frame bucket per bundle).

    features  [1, Tb, 128] + half_mask [1, Tb/2]  ->  soft_tokens [1, Tb/4, 512]

Forms (``--form``):
    fp32     float32 weights, float32 arithmetic (the correctness baseline)
    fp16w32  every parameter stored in float16, read through a float32 cast; float32 arithmetic
    fp16     float16 weights and arithmetic, float16 inputs and output, max-abs RMSNorm (the Neural Engine form)
    int8     every Linear weight int8, symmetric, per block of 32 along the input axis (coreai-opt, weight
             only, dequantized in-graph); the other parameters float32; float32 arithmetic
    fp16sf   fp16 with the subsampling LayerNorms' channel mean folded into the conv weights (W - mean_k W)
    fp16mm   fp16 with every RMSNorm's mean of squares taken as a matmul with a constant 1/D column
             (the two variants are exact rewrites of fp16; they test where the Neural Engine loses precision)

Steps: re-author from model.safetensors (eg2_audio_coreai.py) -> torch.export -> coreai decompositions ->
gate the decomposed graph in torch against the fixture soft tokens (4 clips) -> TorchConverter -> optimize ->
save_asset -> optionally ``coreai-build compile`` (macOS, h16c) for the GPU and the Neural Engine, counting
ANE regions by unique name, and ``coreai-build inspect``. Writes ``<out>/<form>/export_<form>_T<Tb>.json``.

Run with the shared Core AI venv:
    <venv>/bin/python conversion/eg2_transcribe/export_encoder.py --form fp32 --bucket 3000 --aot gpu,neural-engine
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eg2_audio_coreai as E  # noqa: E402
from _common import (BUCKETS, MODEL_ID, MODEL_REVISION, compared_tokens, exports_dir, fixtures_dir,  # noqa: E402
                     load_clips, model_dir, token_stats, verify_checkpoint, window_input, write_json)

FORMS = ("fp32", "fp16w32", "fp16", "int8", "fp16sf", "fp16mm")
FP16_FORMS = ("fp16", "fp16sf", "fp16mm")
GATE_CLIPS = ("1089-134686-0002", "1221-135766-0001", "2094-142345-0001", "2300-131720-0000")  # 2 with n % 4 == 2
LICENSE = "Apache-2.0 + Gemma Prohibited Use Policy"


class Wrapper(torch.nn.Module):
    """Named inputs for the converter: features, half_mask -> soft_tokens."""

    def __init__(self, enc: torch.nn.Module):
        super().__init__()
        self.enc = enc

    def forward(self, features: torch.Tensor, half_mask: torch.Tensor) -> torch.Tensor:
        return self.enc(features, half_mask)


def int8_config() -> dict:
    """Weight-only int8 on nn.Linear modules only (the global default would also try the rank-1 norm weights)."""
    return {
        "execution_mode": "eager",
        "global_config": None,
        "module_type_configs": {
            "torch.nn.modules.linear.Linear": {
                "op_state_spec": {"weight": {"dtype": "int8", "qscheme": "symmetric",
                                             "granularity": {"type": "per_block", "block_size": 32, "axis": 1}}},
                "op_input_spec": None, "op_output_spec": None,
            },
        },
    }


def build(form: str, mdir: Path, bucket: int) -> tuple[torch.nn.Module, torch.dtype]:
    enc = E.load_encoder(mdir, norm="maxabs" if form in FP16_FORMS else "exact")
    io_dtype = torch.float16 if form in FP16_FORMS else torch.float32
    if form == "fp16sf":
        enc = E.fold_subsample_mean(enc)
    elif form == "fp16mm":
        enc = E.use_matmul_meansq(enc)
    if form in FP16_FORMS:
        enc = enc.half()
    elif form == "fp16w32":
        enc = E.store_parameters_fp16(enc)
    model = Wrapper(enc).eval()
    if form == "int8":
        from coreai_opt.common import ExportBackend
        from coreai_opt.quantization import Quantizer, QuantizerConfig

        example = example_inputs(bucket, io_dtype)
        q = Quantizer(model, QuantizerConfig.from_dict({"quantization_config": int8_config()}))
        prepared = q.prepare(example_inputs=(example["features"], example["half_mask"]), dynamic_shapes=None)
        model = q.finalize(prepared, backend=ExportBackend.CoreAI).eval()
    return model, io_dtype


def example_inputs(bucket: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {"features": torch.zeros(1, bucket, E.N_MELS, dtype=dtype),
            "half_mask": torch.ones(1, (bucket + 1) // 2, dtype=dtype)}


def clip_inputs(clip: dict, bucket: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    x, nv = window_input(clip["features"], bucket)
    return {"features": torch.from_numpy(x)[None].to(dtype), "half_mask": E.half_mask_for(nv, bucket, dtype)}


def gate_torch(module, clips: list[dict], bucket: int, dtype: torch.dtype) -> dict:
    rows = []
    with torch.no_grad():
        for c in clips:
            k = compared_tokens(c, bucket)
            out = module(**clip_inputs(c, bucket, dtype))
            out = (out[0] if isinstance(out, (tuple, list)) else out)[0].float().numpy()
            rows.append({"id": c["id"], **token_stats(out[:k], c["ref"][:k])})
    return {"clips": rows, "cos_min": min(r["cos_min"] for r in rows),
            "cos_mean_min": min(r["cos_mean"] for r in rows), "max_abs": max(r["max_abs"] for r in rows),
            "nonfinite": sum(r["nonfinite"] for r in rows)}


def op_counts(program) -> dict:
    counts: Counter = Counter()

    def walk(op):
        counts[op.name] += 1
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    walk(child.operation)

    module = program._mlir_module
    assert module.operation.verify()
    walk(module.operation)
    return dict(sorted(counts.items()))


def coreai_build() -> str:
    env = dict(os.environ)
    if "DEVELOPER_DIR" not in env:
        xcodes = sorted(Path("/Applications").glob("Xcode-27*.app"))
        if xcodes:
            env["DEVELOPER_DIR"] = str(xcodes[-1] / "Contents" / "Developer")
    return subprocess.run(["xcrun", "-f", "coreai-build"], capture_output=True, text=True, check=True,
                          env=env).stdout.strip()


def ane_regions(bundle: Path) -> list[str]:
    """Unique ANE region names in a compiled bundle (a region shows up as a directory and nested files)."""
    names = set()
    for p in bundle.rglob("*"):
        m = re.search(r"([A-Za-z0-9_.-]*_ANE_region_\d+)", p.name)
        if m:
            names.add(re.sub(r"\.(mlir\.)?bc$", "", m.group(1)))
    return sorted(names)


def du_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def aot_compile(cb: str, src: Path, out: Path, compute: str, arch: str) -> dict:
    shutil.rmtree(out, ignore_errors=True)
    argv = [cb, "compile", str(src), "--output", str(out), "--platform", "macOS",
            "--preferred-compute", compute, "--architecture", arch]
    t0 = time.time()
    r = subprocess.run(argv, capture_output=True, text=True)
    log = (r.stdout + r.stderr).strip().splitlines()
    rec = {"argv": " ".join(argv[1:]), "returncode": r.returncode, "seconds": round(time.time() - t0, 1),
           "log_tail": log[-15:], "ane_validation_messages": sum("ane_validation_message" in s for s in log)}
    if r.returncode == 0 and out.exists():
        regions = ane_regions(out)
        rec.update({"bundle": str(out), "bytes": du_bytes(out), "ane_regions": len(regions),
                    "ane_region_names": regions[:64],
                    "files": sorted(str(p.relative_to(out)) for p in out.iterdir())})
    return rec


def inspect(cb: str, bundle: Path, dest: Path) -> dict:
    r = subprocess.run([cb, "inspect", str(bundle), "--ops", "--compute", "--storage", "--json"],
                       capture_output=True, text=True)
    dest.write_text(r.stdout if r.returncode == 0 else r.stdout + r.stderr)
    return {"returncode": r.returncode, "json": str(dest)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--form", choices=FORMS, required=True)
    ap.add_argument("--bucket", type=int, choices=BUCKETS, required=True, help="frames per window (1000 = 10 s)")
    ap.add_argument("--aot", default="", help="comma list of gpu,neural-engine: also compile h16c AOT bundles")
    ap.add_argument("--arch", default="h16c", help="AOT architecture (h16c = M4-class Mac)")
    ap.add_argument("--aot-only", action="store_true",
                    help="compile the bundle a previous run saved (exports are not byte-reproducible)")
    ap.add_argument("--model-dir")
    ap.add_argument("--fixtures")
    ap.add_argument("--out-dir", help="default: <work>/exports; the bundle goes to <out>/<form>/")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    mdir = model_dir(args.model_dir)
    digest = verify_checkpoint(mdir)
    out_dir = exports_dir(args.out_dir) / args.form
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"eg2_audio_{args.form}_T{args.bucket}"
    bundle = out_dir / f"{name}.aimodel"
    rec_path = out_dir / f"export_{name}.json"
    if args.aot_only:
        record = json.loads(rec_path.read_text())
        assert Path(record["bundle"]) == bundle and bundle.exists(), bundle
        t0 = time.time()
        compile_all(args, bundle, name, out_dir, record)
        record["aot_only_seconds"] = round(time.time() - t0, 1)
        write_json(rec_path, record)
        return
    clips = [c for c in load_clips(fixtures_dir(args.fixtures)) if c["id"] in GATE_CLIPS]
    record: dict = {"form": args.form, "bucket_frames": args.bucket, "tokens": args.bucket // 4,
                    "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "safetensors_sha256": digest},
                    "software": {"python": platform.python_version(), "torch": torch.__version__},
                    "inputs": {"features": [1, args.bucket, E.N_MELS], "half_mask": [1, (args.bucket + 1) // 2]},
                    "outputs": {"soft_tokens": [1, args.bucket // 4, 512]}}
    for pkg in ("coreai-core", "coreai-torch", "coreai-opt", "torchao"):
        try:
            from importlib.metadata import version
            record["software"][pkg] = version(pkg)
        except Exception:  # noqa: BLE001
            record["software"][pkg] = None

    t0 = time.time()
    model, io_dtype = build(args.form, mdir, args.bucket)
    record["io_dtype"] = str(io_dtype).replace("torch.", "")
    record["rmsnorm"] = "max-abs rescaled (fp16)" if args.form in FP16_FORMS else "exact"
    if args.form == "fp16mm":
        record["rmsnorm"] += ", mean of squares as a matmul with a constant 1/D column"
    record["subsample_layernorm"] = ("channel mean folded into the conv weights" if args.form == "fp16sf"
                                     else "plain")
    record["eager_gate"] = gate_torch(model, clips, args.bucket, io_dtype)
    print(f"[{name}] eager gate cos_min {record['eager_gate']['cos_min']:.9f} ({time.time() - t0:.0f} s)", flush=True)

    from coreai.runtime import AIModelAssetMetadata
    from coreai_torch import TorchConverter, get_decomp_table

    with torch.no_grad():
        exported = torch.export.export(model, args=(), kwargs=example_inputs(args.bucket, io_dtype))
        decomposed = exported.run_decompositions(get_decomp_table())
    record["decomposed_gate"] = gate_torch(decomposed.module(), clips, args.bucket, io_dtype)
    print(f"[{name}] decomposed gate cos_min {record['decomposed_gate']['cos_min']:.9f} "
          f"({time.time() - t0:.0f} s)", flush=True)

    t1 = time.time()
    program = TorchConverter().add_exported_program(
        exported_program=decomposed, input_names=["features", "half_mask"], output_names=["soft_tokens"]
    ).to_coreai()
    record["convert_s"] = round(time.time() - t1, 1)
    t1 = time.time()
    program.optimize()
    record["optimize_s"] = round(time.time() - t1, 1)
    record["op_counts"] = op_counts(program)
    meta = AIModelAssetMetadata()
    meta.author = "coreai-model-zoo community port; weights Google DeepMind (Gemma Team)"
    meta.license = LICENSE
    meta.model_description = (f"EmbeddingGemma 2 audio encoder (Gemma 4 audio tower + embed_audio): log-mel "
                              f"features [1,{args.bucket},128] + half_mask -> soft tokens [1,{args.bucket // 4},512]; "
                              f"{args.form}; source {MODEL_ID}@{MODEL_REVISION[:8]}")
    assert meta.author.startswith("coreai-model-zoo") and meta.license == LICENSE and meta.model_description
    shutil.rmtree(bundle, ignore_errors=True)          # save_asset does not overwrite
    t1 = time.time()
    program.save_asset(bundle, meta)
    record["save_s"] = round(time.time() - t1, 1)
    record["bundle"] = str(bundle)
    record["bytes"] = du_bytes(bundle)
    record["main_hash"] = (bundle / "main.hash").read_bytes().hex() if (bundle / "main.hash").exists() else None
    print(f"[{name}] saved {bundle} {record['bytes'] / 1e6:.1f} MB ({time.time() - t0:.0f} s)", flush=True)
    del program, decomposed, exported

    compile_all(args, bundle, name, out_dir, record)
    record["seconds"] = round(time.time() - t0, 1)
    write_json(rec_path, record)
    print(json.dumps({k: record[k] for k in ("form", "bucket_frames", "bytes", "seconds")}))


def compile_all(args, bundle: Path, name: str, out_dir: Path, record: dict) -> None:
    cb = coreai_build()
    record["coreai_build"] = subprocess.run([cb, "--version"], capture_output=True, text=True).stdout.strip()
    record["inspect_source"] = inspect(cb, bundle, out_dir / f"inspect_{name}.json")
    record.setdefault("aot", {})
    for compute in [c for c in args.aot.split(",") if c]:
        tag = {"gpu": "gpu", "neural-engine": "ane"}[compute]
        aot = out_dir / f"{name}_{tag}_{args.arch}.aimodelc"
        rec = aot_compile(cb, bundle, aot, compute, args.arch)
        if rec["returncode"] == 0:
            rec["inspect"] = inspect(cb, aot, out_dir / f"inspect_{name}_{tag}_{args.arch}.json")
        record["aot"][tag] = rec
        print(f"[{name}] AOT {compute}: rc {rec['returncode']} {rec['seconds']} s, "
              f"ANE regions {rec.get('ane_regions')}, {rec.get('bytes', 0) / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
