#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.9.0",
#     "coreai-torch==0.4.1",
#     "coreai-core==1.0.0b2",
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
"""Cut the EmbeddingGemma 2 audio encoder at a stage: float16 stage prefixes, float32 fronts, float16 bodies.

Stages in graph order: ``sub`` (the two stride-2 conv blocks + input_proj), ``layer0`` .. ``layer11`` (the output
of that conformer layer), ``head`` (output_proj + embed_audio, i.e. the soft tokens). "After stage s" means every
op up to and including s, so the prefix after ``head`` is the whole encoder.

Parts (``--part``), one static graph per frame bucket Tb:
    prefix  features + half_mask -> the hidden after s [1, Tb/4, 1024] (head: soft tokens [1, Tb/4, 512]);
            float16 weights, arithmetic and I/O, built like the fp16 form of export_encoder.py (max-abs RMSNorm,
            plain subsampling LayerNorm). ``attrib`` runs them to see on which stage each backend's float16
            error grows.
    front   features + half_mask -> hidden after s [1, Tb/4, 1024]; float32 weights, arithmetic and I/O (GPU)
    body    hidden after s [1, Tb/4, 1024] float16 -> soft_tokens [1, Tb/4, 512] float16; built like the fp16
            form (the Neural Engine part)

Two-bundle form: call the front (GPU), cast its ``hidden`` to float16 on the host, call the body (Neural Engine).
Every op after the subsampling is causal or per token, so the cut carries no state: the body's leading k rows
depend only on the leading k rows of ``hidden``. ``gate_encoder.py --front --body`` gates a pair and
``bench_encoder.py`` times it (form ``split<s>``).

Subcommands:
    reference  torch float32 hidden after every stage (optionally the torch float16 one too) for fixture clips,
               after checking that body(front(x)) and the head prefix reproduce the whole float32 encoder.
               Writes an npz (keys <clip>/<stage>/fp32 and <clip>/<stage>/torch_fp16, valid rows only) + json.
    export     one part: torch.export -> coreai decompositions -> torch check on one clip (decomposed vs eager,
               and vs the float32 reference) -> TorchConverter -> optimize -> save_asset -> optional
               coreai-build AOT (macOS h16c) for gpu / neural-engine with the ANE regions counted by unique name.
    warm       load one bundle with its backend and run one call, no numbers: the Neural Engine's initial load of
               an AOT bundle specializes for minutes, later loads are fast.
    attrib     run the prefix bundles on the ANE (AOT), the GPU (AOT) and the CPU (JIT, cpu_only) on fixture
               clips; per stage, per-token cosine and max |delta| against the float32 reference -> json.

Run with the shared Core AI venv (torch 2.9, coreai-torch, coreai-core); GPU / ANE runs inside a measurement window:
    <venv>/bin/python conversion/eg2_transcribe/split_encoder.py reference --bucket 1000 --clips attrib --torch-fp16
    <venv>/bin/python conversion/eg2_transcribe/split_encoder.py export --part prefix --stage layer4 --bucket 1000 \\
        --aot gpu,neural-engine
    <venv>/bin/python conversion/eg2_transcribe/split_encoder.py export --part front --stage layer4 --bucket 3000 --aot gpu
    <venv>/bin/python conversion/eg2_transcribe/split_encoder.py export --part body --stage layer4 --bucket 3000 \\
        --aot neural-engine
    <venv>/bin/python conversion/eg2_transcribe/split_encoder.py warm --bundle <x>_ane_h16c.aimodelc --backend ane
    <venv>/bin/python conversion/eg2_transcribe/split_encoder.py attrib --reference <npz> --out <json>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eg2_audio_coreai as E  # noqa: E402
import export_encoder as X  # noqa: E402
from _common import (BUCKETS, MODEL_ID, MODEL_REVISION, compared_tokens, exports_dir, fixtures_dir,  # noqa: E402
                     load_clips, model_dir, token_stats, verify_checkpoint, window_input, write_json)

N_LAYERS = 12
HIDDEN = 1024
STAGES = ("sub",) + tuple(f"layer{i}" for i in range(N_LAYERS)) + ("head",)
PREFIX_STAGES = ("sub", "layer0", "layer2", "layer4", "layer6", "layer8", "layer10", "layer11", "head")
PARTS = ("prefix", "front", "body")
# The clips whose near-silent tokens miss the per-token minimum on every fp16 Neural Engine / CPU form (round 3 of
# this port: the same positions on every backend and form), with those tokens; and two control clips.
SENSITIVE = {"1188-133604-0001": [217, 218], "121-121726-0001": [52, 53], "121-121726-0000": [160],
             "1995-1826-0000": [108]}
CONTROL = ("1089-134686-0002", "2300-131720-0000")
CHECK_CLIP = "121-121726-0001"        # export's torch check: a sensitive clip, 592 frames (fits every bucket)
NAME_RE = re.compile(r"eg2_(prefix|front|body)_([a-z0-9]+)_(fp16|fp32)_T(\d+)")


def layers_through(stage: str) -> int:
    """Conformer layers inside "after stage": sub -> 0, layerK -> K + 1, head -> 12."""
    if stage == "sub":
        return 0
    if stage == "head":
        return N_LAYERS
    k = int(stage.removeprefix("layer"))
    assert 0 <= k < N_LAYERS and stage == f"layer{k}", stage
    return k + 1


class Front(nn.Module):
    """features [B, Tb, 128] + half_mask [B, Tb/2] -> the hidden after ``stage`` (head: the soft tokens)."""

    def __init__(self, enc: E.Eg2AudioEncoder, stage: str):
        super().__init__()
        self.sub0, self.sub1, self.input_proj = enc.sub0, enc.sub1, enc.input_proj
        self.layers = nn.ModuleList(list(enc.layers)[: layers_through(stage)])
        self.head = stage == "head"
        if self.head:
            self.output_proj = enc.output_proj
            self.pre_projection_norm = enc.pre_projection_norm
            self.embedding_projection = enc.embedding_projection
        self.register_buffer("edge_bias", enc.edge_bias.clone())

    def forward(self, features: torch.Tensor, half_mask: torch.Tensor) -> torch.Tensor:
        x = self.sub1(self.sub0(features.unsqueeze(1)) * half_mask[:, None, :, None])
        b, c, t4, f = x.shape
        x = self.input_proj(x.permute(0, 2, 3, 1).reshape(b, t4, f * c))
        edge_bias = self.edge_bias[:, :t4]
        for layer in self.layers:
            x = layer(x, edge_bias)
        if self.head:
            x = self.embedding_projection(self.pre_projection_norm(self.output_proj(x)))
        return x


class Body(nn.Module):
    """hidden after ``stage`` [B, T4, 1024] -> soft tokens [B, T4, 512]."""

    def __init__(self, enc: E.Eg2AudioEncoder, stage: str):
        super().__init__()
        assert stage != "head", "the body after head is empty"
        self.layers = nn.ModuleList(list(enc.layers)[layers_through(stage):])
        self.output_proj = enc.output_proj
        self.pre_projection_norm = enc.pre_projection_norm
        self.embedding_projection = enc.embedding_projection
        self.register_buffer("edge_bias", enc.edge_bias.clone())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        edge_bias = self.edge_bias[:, : hidden.shape[1]]
        x = hidden
        for layer in self.layers:
            x = layer(x, edge_bias)
        return self.embedding_projection(self.pre_projection_norm(self.output_proj(x)))


def run_stages(enc: E.Eg2AudioEncoder, features: torch.Tensor, half_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """One forward of the whole encoder keeping the hidden after every stage ([T4, D] each, batch 1)."""
    keep = {}
    x = enc.sub1(enc.sub0(features.unsqueeze(1)) * half_mask[:, None, :, None])
    b, c, t4, f = x.shape
    x = enc.input_proj(x.permute(0, 2, 3, 1).reshape(b, t4, f * c))
    keep["sub"] = x[0]
    edge_bias = enc.edge_bias[:, :t4]
    for i, layer in enumerate(enc.layers):
        x = layer(x, edge_bias)
        keep[f"layer{i}"] = x[0]
    keep["head"] = enc.embedding_projection(enc.pre_projection_norm(enc.output_proj(x)))[0]
    return keep


def build_part(part: str, stage: str, mdir: Path) -> tuple[nn.Module, torch.dtype]:
    """prefix / body: the fp16 build (max-abs RMSNorm, plain subsampling LayerNorm, every tensor float16);
    front: float32 with the exact RMSNorm."""
    fp16 = part in ("prefix", "body")
    enc = E.load_encoder(mdir, norm="maxabs" if fp16 else "exact")
    assert enc.input_proj.weight.shape[0] == HIDDEN
    mod = Body(enc, stage) if part == "body" else Front(enc, stage)
    if fp16:
        mod = mod.half()
    return mod.eval(), (torch.float16 if fp16 else torch.float32)


def part_name(part: str, stage: str, bucket: int) -> str:
    return f"eg2_{part}_{stage}_{'fp32' if part == 'front' else 'fp16'}_T{bucket}"


def part_paths(split_dir: Path, part: str, stage: str, bucket: int, arch: str = "h16c",
               head_fallback: bool = True) -> dict[str, Path]:
    """Source .aimodel and AOT bundles of a part. Prefixes live in <split>/prefix/. The head prefix is the whole fp16
    encoder: when it was not exported here, export_encoder.py's fp16 bundles (<exports>/fp16/) are the same graph."""
    d = split_dir / "prefix" if part == "prefix" else split_dir
    name = part_name(part, stage, bucket)
    paths = {"src": d / f"{name}.aimodel", "gpu": d / f"{name}_gpu_{arch}.aimodelc",
             "ane": d / f"{name}_ane_{arch}.aimodelc"}
    if head_fallback and part == "prefix" and stage == "head" and not paths["src"].exists():
        fp16 = split_dir.parent / "fp16"
        name = f"eg2_audio_fp16_T{bucket}"
        paths = {"src": fp16 / f"{name}.aimodel", "gpu": fp16 / f"{name}_gpu_{arch}.aimodelc",
                 "ane": fp16 / f"{name}_ane_{arch}.aimodelc"}
    return paths


def split_dir_of(arg: str | None) -> Path:
    return Path(arg).expanduser() if arg else exports_dir(None) / "split"


def select_clips(fx: Path, which: str) -> list[dict]:
    clips = load_clips(fx)
    if which == "all":
        return clips
    by_id = {c["id"]: c for c in clips}
    ids = list(SENSITIVE) + list(CONTROL) if which == "attrib" else which.split(",")
    return [by_id[i] for i in ids]


def bucket_inputs(clip: dict, bucket: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, int]:
    x, nv = window_input(clip["features"], bucket)
    return torch.from_numpy(x)[None].to(dtype), E.half_mask_for(nv, bucket, dtype), compared_tokens(clip, bucket)


def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)


# ---------------------------------------------------------------------------------------------------- reference
def reference(args) -> None:
    mdir = model_dir(args.model_dir)
    digest = verify_checkpoint(mdir)
    clips = select_clips(fixtures_dir(args.fixtures), args.clips)
    torch.set_num_threads(args.threads)
    t0 = time.time()
    f32 = E.load_encoder(mdir, norm="exact")
    f16 = E.load_encoder(mdir, norm="maxabs").half().eval() if args.torch_fp16 else None
    keep = STAGES if args.stages == "all" else tuple(args.stages.split(","))
    arrays, rows = {}, []
    with torch.no_grad():
        for c in clips:
            feats, hm, k = bucket_inputs(c, args.bucket, torch.float32)
            h32 = run_stages(f32, feats, hm)
            for s in keep:
                arrays[f"{c['id']}/{s}/fp32"] = h32[s][:k].numpy().astype(np.float32)
            row = {"id": c["id"], "n_frames": c["n_frames"], "k": k,
                   "head_fp32_vs_fixture": token_stats(h32["head"][:k].numpy(), c["ref"][:k])}
            if f16 is not None:
                h16 = run_stages(f16, feats.half(), hm.half())
                for s in keep:
                    arrays[f"{c['id']}/{s}/torch_fp16"] = h16[s][:k].float().numpy()
                row["head_torch_fp16_vs_fixture"] = token_stats(h16["head"][:k].float().numpy(), c["ref"][:k])
            rows.append(row)
            print(f"{c['id']}: k={k} head fp32 vs fixture cos min {row['head_fp32_vs_fixture']['cos_min']:.9f}"
                  + (f", torch fp16 {row['head_torch_fp16_vs_fixture']['cos_min']:.6f}" if f16 is not None else "")
                  + f" ({time.time() - t0:.0f} s)", flush=True)
        # the cut is exact: body(front(x)) and the head prefix give the whole encoder, bit for bit in float32
        comp = []
        for c in clips[:2]:
            feats, hm, k = bucket_inputs(c, args.bucket, torch.float32)
            whole = f32(feats, hm)
            for s in args.split_stages.split(","):
                y = Body(f32, s)(Front(f32, s)(feats, hm))
                comp.append({"id": c["id"], "stage": s, "max_abs_vs_whole": float((y - whole).abs().max())})
            y = Front(f32, "head")(feats, hm)
            comp.append({"id": c["id"], "stage": "head prefix", "max_abs_vs_whole": float((y - whole).abs().max())})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    write_json(out.with_suffix(".json"), {
        "what": "torch float32 hidden after every stage of the re-authored encoder (eg2_audio_coreai.py, exact "
                "RMSNorm, CPU)" + (" and of the float16 build (max-abs RMSNorm, every tensor float16, CPU)"
                                    if f16 is not None else "") + "; valid rows only",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "safetensors_sha256": digest},
        "bucket_frames": args.bucket, "stages": list(keep), "npz": str(out),
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "threads": torch.get_num_threads()},
        "composition_check": comp, "composition_exact": all(r["max_abs_vs_whole"] == 0.0 for r in comp),
        "clips": rows, "seconds": round(time.time() - t0, 1)})
    print(f"composition max|d| {max(r['max_abs_vs_whole'] for r in comp):.3e}; wrote {out} ({len(arrays)} arrays)")


# ------------------------------------------------------------------------------------------------------- export
def example_inputs(part: str, bucket: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    if part == "body":
        return {"hidden": torch.zeros(1, E.n_tokens(bucket), HIDDEN, dtype=dtype)}
    return X.example_inputs(bucket, dtype)


def export(args) -> None:
    mdir = model_dir(args.model_dir)
    digest = verify_checkpoint(mdir)
    torch.set_num_threads(args.threads)
    if args.part == "body" and args.stage == "head":
        raise SystemExit("the body after head is empty")
    paths = part_paths(split_dir_of(args.split_dir), args.part, args.stage, args.bucket, args.arch,
                       head_fallback=False)
    name = part_name(args.part, args.stage, args.bucket)
    bundle = paths["src"]
    out_dir = bundle.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / f"export_{name}.json"
    if args.aot_only:
        record = json.loads(rec_path.read_text())
        assert Path(record["bundle"]) == bundle and bundle.exists(), bundle
        t0 = time.time()
        X.compile_all(args, bundle, name, out_dir, record)
        record["aot_only_seconds"] = round(time.time() - t0, 1)
        write_json(rec_path, record)
        return

    t0 = time.time()
    model, io_dtype = build_part(args.part, args.stage, mdir)
    in_names = ["hidden"] if args.part == "body" else ["features", "half_mask"]
    out_name = "soft_tokens" if (args.part == "body" or args.stage == "head") else "hidden"
    record: dict = {
        "part": args.part, "stage": args.stage, "layers_in_front": layers_through(args.stage),
        "bucket_frames": args.bucket, "tokens": E.n_tokens(args.bucket),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "safetensors_sha256": digest},
        "software": {"python": platform.python_version(), "torch": torch.__version__},
        "io_dtype": str(io_dtype).replace("torch.", ""),
        "rmsnorm": "max-abs rescaled (fp16)" if io_dtype == torch.float16 else "exact",
        "subsample_layernorm": "plain",
        "inputs": {n: list(v.shape) for n, v in example_inputs(args.part, args.bucket, io_dtype).items()},
        "outputs": {out_name: [1, E.n_tokens(args.bucket), 512 if out_name == "soft_tokens" else HIDDEN]}}
    for pkg in ("coreai-core", "coreai-torch", "coreai-opt"):
        try:
            from importlib.metadata import version
            record["software"][pkg] = version(pkg)
        except Exception:  # noqa: BLE001
            record["software"][pkg] = None

    # torch check on one clip: the float32 reference of the same stage (the body's input is the float32 front)
    clip = select_clips(fixtures_dir(args.fixtures), CHECK_CLIP)[0]
    feats, hm, k = bucket_inputs(clip, args.bucket, torch.float32)
    with torch.no_grad():
        ref = run_stages(E.load_encoder(mdir, norm="exact"), feats, hm)
    if args.part == "body":
        kw = {"hidden": ref[args.stage][None].to(io_dtype)}
        target = ref["head"][:k].numpy()
    else:
        kw = {"features": feats.to(io_dtype), "half_mask": hm.to(io_dtype)}
        target = ref[args.stage][:k].numpy()
    from coreai.runtime import AIModelAssetMetadata
    from coreai_torch import TorchConverter, get_decomp_table

    with torch.no_grad():
        eager = model(**kw)[0, :k].float().numpy()
        exported = torch.export.export(model, args=(), kwargs=example_inputs(args.part, args.bucket, io_dtype))
        decomposed = exported.run_decompositions(get_decomp_table())
        dec = decomposed.module()(**kw)[0, :k].float().numpy()
    record["torch_check"] = {"clip": clip["id"], "tokens": k,
                             "decomposed_vs_eager": token_stats(dec, eager),
                             "decomposed_vs_fp32_reference": token_stats(dec, target)}
    if args.part == "body" or args.stage == "head":
        record["torch_check"]["decomposed_vs_fixture"] = token_stats(dec, clip["ref"][:k])
    tc = record["torch_check"]
    print(f"[{name}] torch check: decomposed vs eager cos min {tc['decomposed_vs_eager']['cos_min']:.9f}, vs fp32 "
          f"reference {tc['decomposed_vs_fp32_reference']['cos_min']:.6f} ({time.time() - t0:.0f} s)", flush=True)

    t1 = time.time()
    program = TorchConverter().add_exported_program(
        exported_program=decomposed, input_names=in_names, output_names=[out_name]).to_coreai()
    record["convert_s"] = round(time.time() - t1, 1)
    t1 = time.time()
    program.optimize()
    record["optimize_s"] = round(time.time() - t1, 1)
    record["op_counts"] = X.op_counts(program)
    meta = AIModelAssetMetadata()
    meta.author = "coreai-model-zoo community port; weights Google DeepMind (Gemma Team)"
    meta.license = X.LICENSE
    meta.model_description = (f"EmbeddingGemma 2 audio encoder, {args.part} of the cut after {args.stage} "
                              f"({record['io_dtype']}): {', '.join(in_names)} -> {out_name}; bucket {args.bucket} "
                              f"frames; source {MODEL_ID}@{MODEL_REVISION[:8]}")
    shutil.rmtree(bundle, ignore_errors=True)          # save_asset does not overwrite
    t1 = time.time()
    program.save_asset(bundle, meta)
    record["save_s"] = round(time.time() - t1, 1)
    record["bundle"] = str(bundle)
    record["bytes"] = X.du_bytes(bundle)
    record["main_hash"] = (bundle / "main.hash").read_bytes().hex() if (bundle / "main.hash").exists() else None
    print(f"[{name}] saved {bundle} {record['bytes'] / 1e6:.1f} MB ({time.time() - t0:.0f} s)", flush=True)
    del program, decomposed, exported
    X.compile_all(args, bundle, name, out_dir, record)
    record["seconds"] = round(time.time() - t0, 1)
    write_json(rec_path, record)
    print(json.dumps({k2: record.get(k2) for k2 in ("part", "stage", "bucket_frames", "bytes", "seconds")}
                     | {"ane_regions": {t: a.get("ane_regions") for t, a in record.get("aot", {}).items()}}))


# ------------------------------------------------------------------------------------------- runtime (warm, attrib)
def runtime_options(rt, backend: str):
    if backend == "cpu":
        return rt.SpecializationOptions.cpu_only()
    if backend == "gpu":
        return rt.SpecializationOptions.default()
    return rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.neural_engine())


def parse_bundle(bundle: Path) -> tuple[str, str, str, int]:
    """-> (part, stage, dtype, bucket); export_encoder.py's fp16 bundle reads as the head prefix."""
    m = NAME_RE.match(bundle.name)
    if m:
        return m.group(1), m.group(2), m.group(3), int(m.group(4))
    m = re.match(r"eg2_audio_(fp16[a-z]*|fp32)_T(\d+)", bundle.name)     # fp16 and its variants fp16sf / fp16mm
    if m and m.group(1) != "fp16w":
        return "prefix", "head", m.group(1)[:4], int(m.group(2))
    raise SystemExit(f"cannot read part / stage / bucket from {bundle.name}")


def zero_inputs(rt, part: str, dtype: str, bucket: int) -> dict:
    dt = np.float16 if dtype == "fp16" else np.float32
    if part == "body":
        return {"hidden": rt.NDArray(np.zeros((1, E.n_tokens(bucket), HIDDEN), dtype=dt))}
    return {"features": rt.NDArray(np.zeros((1, bucket, E.N_MELS), dtype=dt)),
            "half_mask": rt.NDArray(np.ones((1, (bucket + 1) // 2), dtype=dt))}


async def warm(args) -> None:
    import coreai.runtime as rt

    bundle = Path(args.bundle).expanduser().resolve()
    part, stage, dtype, bucket = parse_bundle(bundle)
    t0 = time.perf_counter()
    model = await rt.AIModel.load(bundle, runtime_options(rt, args.backend))
    fn = model.load_function("main")
    load_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    await fn(zero_inputs(rt, part, dtype, bucket))
    print(json.dumps({"bundle": bundle.name, "backend": args.backend, "ane_regions": len(X.ane_regions(bundle))
                      if bundle.suffix == ".aimodelc" else None, "load_s": round(load_s, 2),
                      "first_call_s": round(time.perf_counter() - t1, 3), "finished": time.strftime("%H:%M:%S"),
                      "note": "warm-up only (outside a measurement window); not a timing"}), flush=True)


def stage_row(out: np.ndarray, ref: np.ndarray, clip_id: str) -> dict:
    cos = cos_rows(out, ref)
    sens = [p for p in SENSITIVE.get(clip_id, []) if p < len(cos)]
    other = np.delete(cos, sens) if sens else cos
    order = np.argsort(cos)[:5]
    return {"id": clip_id, "n": int(len(cos)), "cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
            "cos_min_other_tokens": float(other.min()),
            "sensitive": [{"pos": int(p), "cos": float(cos[p])} for p in sens],
            "worst5": [{"pos": int(i), "cos": float(cos[i])} for i in order],
            "max_abs": float(np.abs(out.astype(np.float64) - ref).max()), "ref_max_abs": float(np.abs(ref).max()),
            "nonfinite": int((~np.isfinite(out)).sum())}


def summarize(rows: list[dict]) -> dict:
    sens = [s["cos"] for r in rows for s in r["sensitive"]]
    ctrl = [r["cos_min"] for r in rows if r["id"] in CONTROL]
    return {"cos_min_sensitive_tokens": min(sens) if sens else None,
            "cos_min_all_tokens": min(r["cos_min"] for r in rows),
            "cos_min_other_tokens": min(r["cos_min_other_tokens"] for r in rows),
            "cos_min_control_clips": min(ctrl) if ctrl else None,
            "max_abs": max(r["max_abs"] for r in rows), "ref_max_abs": max(r["ref_max_abs"] for r in rows),
            "nonfinite": sum(r["nonfinite"] for r in rows), "clips": rows}


async def attrib(args) -> None:
    import coreai.runtime as rt

    split_dir = split_dir_of(args.split_dir)
    ref = np.load(args.reference)
    clips = select_clips(fixtures_dir(args.fixtures), args.clips)
    stages = args.stages.split(",")
    backends = args.backends.split(",")
    result = {"what": "per-stage float16 error of the encoder on the Mac backends: the prefix bundle that ends at each "
                      "stage (float16, built like the fp16 form), per-token cosine and max |delta| of its output "
                      "against the torch float32 hidden of the same stage",
              "bucket_frames": args.bucket, "reference": str(args.reference),
              "sensitive_tokens": SENSITIVE, "control_clips": list(CONTROL),
              "backends": {"ane": "h16c AOT .aimodelc, from_preferred_compute_unit_kind(neural_engine)",
                           "gpu": "h16c AOT .aimodelc, default()", "cpu": "JIT .aimodel, cpu_only()",
                           "torch_fp16": "the same float16 build in torch eager on the CPU (from the reference npz)"},
              "label": args.label, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stages": {}}
    raw = {}
    for stage in stages:
        paths = part_paths(split_dir, "prefix", stage, args.bucket)
        result["stages"][stage] = st = {"bundles": {k: str(v) for k, v in paths.items()}}
        for be in backends:
            bundle = paths["src" if be == "cpu" else be]
            if not bundle.exists():
                st[be] = {"error": f"missing {bundle}"}
                continue
            part, _, dtype, bucket = parse_bundle(bundle)
            assert bucket == args.bucket and dtype == "fp16", bundle
            t0 = time.perf_counter()
            model = await rt.AIModel.load(bundle, runtime_options(rt, be))
            fn = model.load_function("main")
            load_s = time.perf_counter() - t0
            out_name = list(fn.desc.output_names)[0]
            rows = []
            for c in clips:
                x, nv = window_input(c["features"], args.bucket)
                k = compared_tokens(c, args.bucket)
                hm = np.zeros((1, (args.bucket + 1) // 2), dtype=np.float16)
                hm[0, : (nv + 1) // 2] = 1
                res = await fn({"features": rt.NDArray(np.ascontiguousarray(x[None].astype(np.float16))),
                                "half_mask": rt.NDArray(hm)})
                out = res[out_name].numpy().astype(np.float32)[0][:k]
                raw[f"{c['id']}/{stage}/{be}"] = out
                rows.append(stage_row(out, ref[f"{c['id']}/{stage}/fp32"], c["id"]))
            st[be] = {"bundle": str(bundle), "ane_regions": len(X.ane_regions(bundle))
                      if bundle.suffix == ".aimodelc" else None, "load_s": round(load_s, 2), **summarize(rows)}
            print(f"{stage:8s} {be:4s} sens min {st[be]['cos_min_sensitive_tokens']:.6f} all min "
                  f"{st[be]['cos_min_all_tokens']:.6f} other {st[be]['cos_min_other_tokens']:.6f} ctrl "
                  f"{st[be]['cos_min_control_clips']:.6f} max|d| {st[be]['max_abs']:.3e} "
                  f"(ref max {st[be]['ref_max_abs']:.1f}) load {load_s:.1f} s", flush=True)
            del fn, model
        if f"{clips[0]['id']}/{stage}/torch_fp16" in ref:
            rows = [stage_row(ref[f"{c['id']}/{stage}/torch_fp16"], ref[f"{c['id']}/{stage}/fp32"], c["id"])
                    for c in clips]
            st["torch_fp16"] = summarize(rows)
    cols = backends + (["torch_fp16"] if any("torch_fp16" in s for s in result["stages"].values()) else [])
    result["curve"] = [{"stage": s, **{f"{b}_{q}": result["stages"][s].get(b, {}).get(f"cos_min_{q}")
                                       for b in cols for q in ("sensitive_tokens", "other_tokens")},
                        **{f"{b}_max_abs": result["stages"][s].get(b, {}).get("max_abs") for b in cols}}
                       for s in stages]
    result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    outputs = Path(args.out).with_name(Path(args.out).stem + "_outputs.npz")
    np.savez(outputs, **raw)
    result["outputs_npz"] = str(outputs)
    write_json(Path(args.out), result)
    print(f"wrote {args.out} and {outputs}")


def propagate(args) -> None:
    """Carry every prefix output through the rest of the encoder in torch float32 and compare the soft tokens with
    the fixture: the error each backend has made by the end of a stage, measured on the encoder's output (the
    hidden-state cosine of an intermediate stage can stay near 1 while the soft tokens it leads to do not)."""
    mdir = model_dir(args.model_dir)
    verify_checkpoint(mdir)
    torch.set_num_threads(args.threads)
    res = json.loads(Path(args.attrib).read_text())
    raw, ref = np.load(res["outputs_npz"]), np.load(res["reference"])
    clips = select_clips(fixtures_dir(args.fixtures), args.clips)
    f32 = E.load_encoder(mdir, norm="exact")
    t4 = E.n_tokens(res["bucket_frames"])
    cols = [b for b in ("ane", "gpu", "cpu") if any(f"{clips[0]['id']}/{s}/{b}" in raw for s in res["stages"])]
    sources = {b: (raw, "{}/{}/" + b) for b in cols}
    sources["torch_fp16"] = (ref, "{}/{}/torch_fp16")
    sources["fp32_reference"] = (ref, "{}/{}/fp32")       # sanity: the reference itself carried through
    t0 = time.time()
    with torch.no_grad():
        for stage, st in res["stages"].items():
            body = None if stage == "head" else Body(f32, stage).eval()
            for name, (npz, pattern) in sources.items():
                rows = []
                for c in clips:
                    key = pattern.format(c["id"], stage)
                    if key not in npz:
                        break
                    h = npz[key]
                    k = h.shape[0]
                    if body is not None:
                        x = torch.zeros(1, t4, h.shape[1])
                        x[0, :k] = torch.from_numpy(h.astype(np.float32))
                        y = body(x)[0, :k].numpy()
                    else:
                        y = h
                    rows.append(stage_row(y, c["ref"][:k], c["id"]))
                if len(rows) == len(clips):
                    st.setdefault(name, {})["propagated_fp32_rest"] = summarize(rows)
            print(f"{stage:8s} " + "  ".join(
                f"{n} {st[n]['propagated_fp32_rest']['cos_min_sensitive_tokens']:.6f}" for n in sources
                if "propagated_fp32_rest" in st.get(n, {})) + f" ({time.time() - t0:.0f} s)", flush=True)
    names = [n for n in sources if any("propagated_fp32_rest" in s.get(n, {}) for s in res["stages"].values())]
    res["propagated_what"] = ("each prefix output (valid rows, zero rows after: every op after a stage is causal or "
                              "per token) carried through the remaining stages in torch float32; soft tokens vs the "
                              "fixture reference; fp32_reference = the torch float32 hidden carried the same way")
    res["curve_propagated"] = [{"stage": s, **{f"{n}_{q}": res["stages"][s].get(n, {}).get(
        "propagated_fp32_rest", {}).get(f"cos_min_{q}") for n in names for q in ("sensitive_tokens", "other_tokens")}}
                               for s in res["stages"]]
    write_json(Path(args.attrib), res)
    print(f"updated {args.attrib}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference", help="torch float32 (+ float16) hidden after every stage -> npz")
    r.add_argument("--bucket", type=int, choices=BUCKETS, required=True)
    r.add_argument("--clips", default="attrib", help="attrib (4 sensitive + 2 control), all, or a comma list of ids")
    r.add_argument("--torch-fp16", action="store_true", help="also keep the torch float16 build's hiddens")
    r.add_argument("--stages", default="all", help="stages kept in the npz: all, or a comma list (e.g. sub,layer4)")
    r.add_argument("--split-stages", default="sub,layer4,layer8", help="stages whose cut the composition check runs")
    r.add_argument("--out", required=True, help="npz path (a json with the same stem is written next to it)")
    r.add_argument("--threads", type=int, default=4)
    e = sub.add_parser("export", help="export one part (and AOT-compile it)")
    e.add_argument("--part", choices=PARTS, required=True)
    e.add_argument("--stage", choices=STAGES, required=True)
    e.add_argument("--bucket", type=int, choices=BUCKETS, required=True, help="frames per window (1000 = 10 s)")
    e.add_argument("--aot", default="", help="comma list of gpu,neural-engine: also compile h16c AOT bundles")
    e.add_argument("--arch", default="h16c", help="AOT architecture (h16c = M4-class Mac)")
    e.add_argument("--aot-only", action="store_true", help="compile the bundle a previous run saved")
    e.add_argument("--threads", type=int, default=4)
    w = sub.add_parser("warm", help="the initial load of one bundle + one call, no numbers")
    w.add_argument("--bundle", required=True)
    w.add_argument("--backend", choices=("ane", "gpu", "cpu"), required=True)
    a = sub.add_parser("attrib", help="per-stage error of the prefix bundles on the ANE / GPU / CPU -> json")
    a.add_argument("--reference", required=True, help="npz from `reference --bucket <Tb> --clips attrib`")
    a.add_argument("--bucket", type=int, choices=BUCKETS, default=1000)
    a.add_argument("--clips", default="attrib")
    a.add_argument("--stages", default=",".join(PREFIX_STAGES))
    a.add_argument("--backends", default="ane,gpu,cpu")
    a.add_argument("--label", default="", help="the measurement window (recorded in the json)")
    a.add_argument("--out", required=True, help="json; the raw outputs go to <stem>_outputs.npz next to it")
    p = sub.add_parser("propagate", help="carry attrib's prefix outputs through the rest in torch float32")
    p.add_argument("--attrib", required=True, help="the json written by attrib (updated in place)")
    p.add_argument("--clips", default="attrib")
    p.add_argument("--threads", type=int, default=4)
    for q in (r, e, a, p):
        q.add_argument("--model-dir")
        q.add_argument("--fixtures")
    for q in (e, a):
        q.add_argument("--split-dir", help="default: <work>/exports/split")
    args = ap.parse_args()
    if args.cmd == "reference":
        reference(args)
    elif args.cmd == "export":
        export(args)
    elif args.cmd == "warm":
        asyncio.run(warm(args))
    elif args.cmd == "propagate":
        propagate(args)
    else:
        asyncio.run(attrib(args))


if __name__ == "__main__":
    main()
