#!/usr/bin/env python3
"""Export d1-omni-600M's audio graph (one clip bucket: the subsampling, the 17-layer conformer, the adapter and the
residual) as a Core AI bundle.

    python3 conversion/d1_omni/export_audio.py --precision fp32 --sec 10      # the reference
    python3 conversion/d1_omni/export_audio.py --precision wfp16 --sec 5      # 5 / 10 / 20 / 30
    python3 conversion/d1_omni/export_audio.py --precision fp16 --sec 30

Run with the shared export venv (coreai-models/.venv: torch 2.9.0, coreai-torch 0.4.1, coreai-core 1.0.0b2, soundfile).
The graph is d1_omni_audio.D1Audio (plain torch), its weights the pinned snapshot's model.safetensors (sha256 verified on
load).

Order: load_d1_audio(snapshot, sec, precision, verify_sha256=True) -> torch.export with one clip's inputs
(host.audio_inputs of EXAMPLE[sec]) -> run_decompositions(get_decomp_table()) -> the export gate -> TorchConverter (one
function, main) -> optimize -> save_asset.

Export gate: the decomposed program on every clip of audio_check.load_clips whose own bucket is this one (the host's
inputs, torch mel) against the reference prefix (the oracle npz, or the publisher's Audio run here for the measured edge
clips; cos, max |d|, min row cos, in float64), and against the eager module of the same precision (what export +
decomposition changed; every clip, or the first clip at fp16, which torch runs slowly on the CPU). fp32 stops before
conversion unless cos >= 0.999999 on every clip and the program equals the eager module; wfp16 / fp16 are measured and
converted (the end-to-end runtime gate decides).

Layout ($ZOO_WORK_ROOT/_d1_omni/bundles/d1-omni-600m/macos/audio-<precision>-<sec>s/, built in a staging folder and
renamed at the end; an existing folder is never replaced):
  d1_omni_audio_<precision>_<sec>s.aimodel   the bundle, function main
  metadata.json                             the audio contract a host reads (inputs, output, bucket, mel, masks)
  mel_filters_128x257_f32.bin               the Slaney mel filterbank (mel_host.filterbank_bytes), float32 LE [128, 257]
  reference.json                            the gate's clips (bucket, P) with the reference npz and sha256
  provenance/export-manifest.json           bytes, op counts, seconds, environment, input hashes
  provenance/export-gate.json               every clip of the export gate
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

import d1_omni_audio as da  # noqa: E402
import host  # noqa: E402
import mel_host  # noqa: E402
from audio_check import PREFIX_BAR, load_clips, load_publisher_audio, publisher_audio_model  # noqa: E402
from export_decide import (AUTHOR, FAMILY, LICENSE, element_type, environment, file_inventory,  # noqa: E402
                           parameter_bytes, sha256_file, verify_source, write_json)
from vision_check import compare  # noqa: E402

WORK = work_path("_d1_omni")
INPUT_NAMES = list(host.AUDIO_INPUT_NAMES)
OUTPUT_NAMES = ["prefix"]
EXAMPLE = {5: "aud_04", 10: "aud_01", 20: "aud_08", 30: "aud_10"}
CODE_FILES = ("export_audio.py", "d1_omni_audio.py", "host.py", "mel_host.py", "audio_check.py")


def bundle_dir(precision: str, sec: int) -> Path:
    return WORK / "bundles" / FAMILY / "macos" / f"audio-{precision}-{sec}s"


def bundle_name(precision: str, sec: int) -> str:
    return f"d1_omni_audio_{precision}_{sec}s.aimodel"


def as_tensors(inputs: dict) -> dict:
    return {k: torch.from_numpy(np.ascontiguousarray(inputs[k])) for k in INPUT_NAMES}


def prefix_of(out, rows: int) -> np.ndarray:
    if isinstance(out, (tuple, list)):
        assert len(out) == 1, len(out)
        out = out[0]
    return out.detach().to(torch.float32).numpy().reshape(-1, host.AUDIO_PREFIX_HIDDEN)[:rows].copy()


def export_gate(program, model, clips: list[dict], eager_all: bool) -> tuple[list[dict], dict]:
    module = program.module()
    records = []
    for i, clip in enumerate(clips):
        t0 = time.perf_counter()
        inputs, p = clip["inputs"], clip["inputs"]["prefix_rows"]
        with torch.no_grad():
            got = prefix_of(module(**as_tensors(inputs)), p)
            eager = prefix_of(model(**as_tensors(inputs)), p) if eager_all or i == 0 else None
        stats = compare(got, clip["reference_prefix"])
        record = {"id": clip["id"], "kind": clip["kind"], "samples": int(len(clip["samples"])), "prefix_rows": p,
                  "prefix": stats, "cos_pass": stats["cos"] >= PREFIX_BAR["cos"], "finite": bool(np.isfinite(got).all()),
                  "reference": clip["reference_kind"], "seconds": None}
        if eager is not None:
            record["vs_eager"] = {"max_abs": float(np.max(np.abs(got.astype(np.float64) - eager))),
                                  "bit_equal": bool(np.array_equal(got, eager))}
        record["seconds"] = time.perf_counter() - t0
        records.append(record)
        print(f"  {clip['id']}: P {p} cos {stats['cos']:.9f} max {stats['max_abs']:.3e} min-row "
              f"{stats['min_row_cos']:.9f}" + (f"; vs eager {record['vs_eager']['max_abs']:.1e} "
                                               f"({'bit' if record['vs_eager']['bit_equal'] else 'diff'})"
                                               if eager is not None else "") + f" {record['seconds']:.1f} s", flush=True)
    compared = [r for r in records if "vs_eager" in r]
    summary = {"clips": len(records), "clip_ids": [r["id"] for r in records],
               "min_cos": min(r["prefix"]["cos"] for r in records),
               "min_row_cos": min(r["prefix"]["min_row_cos"] for r in records),
               "max_abs": max(r["prefix"]["max_abs"] for r in records),
               "worst_clip": min(records, key=lambda r: r["prefix"]["cos"])["id"],
               "cos_pass": f"{sum(r['cos_pass'] for r in records)}/{len(records)}",
               "finite": all(r["finite"] for r in records),
               "vs_eager_max_abs": max(r["vs_eager"]["max_abs"] for r in compared),
               "vs_eager_bit_equal": f"{sum(r['vs_eager']['bit_equal'] for r in compared)}/{len(compared)}"}
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


def io_contract(sec: int) -> dict:
    s = host.audio_bucket_shapes(sec)
    return {
        "inputs": {
            "mel": {"dtype": "float32", "shape": [1, host.AUDIO_FEATURES, s["F"]],
                    "values": "the clip's normalised log-mel (mel_host: waveform() + MelFrontend), columns t >= frames "
                              "0.0, zero-padded to F"},
            "mask_f": {"dtype": "float32", "shape": [1, s["F"]], "values": "1.0 on t < l0 = frames (n // 160)"},
            "mask_f2": {"dtype": "float32", "shape": [1, s["F2"]], "values": "1.0 on t < l1 = (l0 - 1) // 2 + 1"},
            "mask_f4": {"dtype": "float32", "shape": [1, s["F4"]], "values": "1.0 on t < l2 = (l1 - 1) // 2 + 1"},
            "mask_t": {"dtype": "float32", "shape": [1, s["T"]], "values": "1.0 on t < P = l3 = (l2 - 1) // 2 + 1"}},
        "outputs": {"prefix": {"dtype": "float32", "shape": [1, s["T"], host.AUDIO_PREFIX_HIDDEN],
                               "read": "rows [0, P): the clip's prefix embeddings"}},
    }


def bundle_metadata(name: str, precision: str, sec: int) -> dict:
    s = host.audio_bucket_shapes(sec)
    return {
        "metadata_version": "0.2",
        "kind": "encoder",
        "audio": {
            "role": "d1-omni audio prefix: one clip per call", "precision": precision,
            "functions": {"main": "main"}, **io_contract(sec),
            "bucket": {"seconds": sec, **{k: s[k] for k in ("F", "F2", "F4", "T")},
                       "holds": f"clips of n samples with 1 + n // 160 <= {s['F']} after waveform()'s cut and pad "
                                f"(host.audio_bucket_for: the smallest of {list(host.AUDIO_CLIP_SECONDS)} s that holds it)",
                       "max_prefix_rows": host.audio_prefix_length(sec * host.SAMPLE_RATE + host.HOP - 1)},
            "waveform": "16 kHz mono; int16 / 32768 or float32 as is; cut to 480,000 samples (30 s); zero-padded to 8,000 "
                        "(0.5 s); no resampling (the publisher's waveform())",
            "mel": {"spec": "mel_host.py (module docstring): preemphasis 0.97; center=True constant-padded STFT, n_fft 512, "
                            "hop 160, Hann(400, periodic=False) centred in 512; power (sqrt(re^2+im^2))^2; Slaney mel "
                            "128 (mel_filters_128x257_f32.bin); log(x + 2^-24); per mel row over the frames t < n // 160: "
                            "(x - mean) / (std(ddof 1) + 1e-5); 0.0 after",
                    "python": "mel_host.mel_torch (the publisher's code, bit for bit)",
                    "swift_form": "mel_host.mel_numpy (the same steps in NumPy, float64)"},
            "filterbank": {"file": mel_host.FILTERBANK_FILE, "shape": [128, 257], "dtype": "float32",
                           "byte_order": "little", "layout": "row-major (mel row, FFT bin)",
                           "made_by": "the publisher's slaney_filterbank() (librosa's Slaney mel, 0 to 8 kHz)"},
            "prefix": "the graph's first P rows (P = host.audio_prefix_length(n)); this is the decision graph's "
                      "prefix_embeds [0, P) with max_len = min(15360, 16384 - P) and state None -> {}",
            "host_reference": "conversion/d1_omni/host.py (audio_inputs, audio_prefix, audio_request_rows) and mel_host.py",
        },
        "source": {"hf_model_id": host.MODEL_ID, "hf_revision": host.MODEL_SHA, "license": LICENSE},
        "assets": {"main": name},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--precision", choices=da.PRECISIONS, required=True)
    parser.add_argument("--sec", type=int, choices=da.CLIP_SECONDS, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_grad_enabled(False)
    out_dir = bundle_dir(args.precision, args.sec)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists: an existing bundle folder is never replaced")
    stage = out_dir.parent / f".staging-{out_dir.name}-{os.getpid()}"
    stage.mkdir(parents=True)
    name = bundle_name(args.precision, args.sec)
    print(f"pid {os.getpid()} staging {stage}", flush=True)

    snapshot = Path(hf_snapshot(host.MODEL_ID, revision=host.MODEL_SHA))
    source = verify_source(snapshot)
    clips, ref_info = load_clips()
    clips = [c for c in clips if host.audio_bucket_for(len(c["samples"])) == args.sec]
    publisher = load_publisher_audio(snapshot)
    pub32 = publisher_audio_model(snapshot, publisher)[0] if any(c["kind"] == "synthetic" for c in clips) else None
    for clip in clips:
        clip["inputs"] = host.audio_inputs(clip["samples"], args.sec)
        if clip["kind"] == "fixture":
            clip["reference_prefix"], clip["reference_kind"] = clip["npz"]["prefix"], "oracle npz"
        else:
            clip["reference_prefix"] = pub32(clip["samples"])[0].numpy()
            clip["reference_kind"] = "the publisher's Audio, run here"
    t0 = time.perf_counter()
    model, load_record = da.load_d1_audio(snapshot, args.sec, args.precision, verify_sha256=True)
    load_s = time.perf_counter() - t0
    example = next(c for c in clips if c["id"] == EXAMPLE[args.sec])
    print(f"clips {len(clips)} in the {args.sec} s bucket {model.shapes}; example {example['id']} (P "
          f"{example['inputs']['prefix_rows']})", flush=True)
    weights = parameter_bytes(model)
    from coreai_torch import get_decomp_table

    t0 = time.perf_counter()
    exported = torch.export.export(model, args=(), kwargs=as_tensors(example["inputs"]))
    export_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    program = exported.run_decompositions(get_decomp_table())
    decompose_s = time.perf_counter() - t0
    print(f"torch.export {export_s:.1f} s, decompositions {decompose_s:.1f} s", flush=True)

    t0 = time.perf_counter()
    records, summary = export_gate(program, model, clips, eager_all=args.precision != "fp16")
    gate_s = time.perf_counter() - t0
    all_cos = summary["cos_pass"].split("/")[0] == summary["cos_pass"].split("/")[1]
    if args.precision == "fp32":
        bit = summary["vs_eager_bit_equal"].split("/")
        ok = all_cos and summary["finite"] and bit[0] == bit[1]
        status = "PASS" if ok else "FAIL"
    else:
        status = "PASS" if all_cos and summary["finite"] else "MEASURED"
    print(f"export gate {status} in {gate_s:.0f} s: {json.dumps(summary)}", flush=True)
    gate_record = {"status": status, "stage": "export (torch-exported + decomposed program, CPU, before conversion)",
                   "precision": args.precision, "bucket_s": args.sec,
                   "bar": {**PREFIX_BAR, "program_equals_eager": "fp32: bit for bit"},
                   "bar_rule": ("fp32 stops before conversion if this misses" if args.precision == "fp32" else
                                "measured, converted whatever it says; the end-to-end runtime gate decides"),
                   "reference": ref_info, "example": example["id"], "summary": summary, "seconds": gate_s,
                   "environment": environment(("torch", "coreai-torch", "coreai-core", "numpy", "safetensors",
                                               "soundfile")),
                   "clips": records}
    write_json(stage / "provenance" / "export-gate.json", gate_record)
    if args.precision == "fp32" and status != "PASS":
        raise SystemExit(f"fp32 export gate FAIL before conversion (record {stage / 'provenance' / 'export-gate.json'})")

    shapes = model.shapes
    description = (f"d1-omni-600M audio graph (8x subsampling + 17-layer FastConformer + adapter + residual), "
                   f"{args.precision}, clip bucket {args.sec} s (mel [1,128,{shapes['F']}] -> prefix [1,{shapes['T']},1024]), "
                   f"function main; source {host.MODEL_ID}@{host.MODEL_SHA}")
    converted = convert(program, description, stage / name)
    print(f"converted {converted['seconds']} ops {converted['ops_total']}", flush=True)
    (stage / mel_host.FILTERBANK_FILE).write_bytes(mel_host.filterbank_bytes())
    metadata = bundle_metadata(name, args.precision, args.sec)
    metadata["audio"]["precision_detail"] = {
        "weights": "fp32" if args.precision == "fp32" else "fp16 (every Linear / Conv weight and bias, pos_bias_u / v)",
        "compute": {"fp32": "fp32", "wfp16": "fp32 (each fp16 weight cast at its use)",
                    "fp16": "fp16 (LayerNorm, the attention softmax, the BatchNorm affine, both GELUs and the output "
                            "in fp32)"}[args.precision],
        "constants_fp32": "the relative-position table [1, 2T-1, 512] and the folded BatchNorm scale / shift"}
    metadata["audio"]["filterbank"]["sha256"] = sha256_file(stage / mel_host.FILTERBANK_FILE)
    write_json(stage / "metadata.json", metadata)
    reference = {"schema": "d1-omni-audio-reference/1", "model": host.MODEL_ID, "revision": host.MODEL_SHA,
                 "reference": ref_info, "bucket_s": args.sec,
                 "oracle": "the publisher's Audio (audio.py) in fp32 on the CPU, hooked inside D1OmniModel.probabilities()"
                           " (fixture clips); the measured edge clips: the same module run by the export",
                 "clips": [{"id": c["id"], "kind": c["kind"], "public": c["public"], "file": c.get("file"),
                            "made": c.get("made"), "samples": int(len(c["samples"])), "frames": c["inputs"]["frames"],
                            "prefix_rows": c["inputs"]["prefix_rows"], "npz": c.get("npz_file"),
                            "npz_sha256": c.get("npz_sha256")} for c in clips]}
    write_json(stage / "reference.json", reference)

    files = file_inventory(stage / name)
    bundle_bytes = sum(f["bytes"] for f in files)
    record = {
        "status": status, "model": host.MODEL_ID, "model_sha": host.MODEL_SHA, "bundle": name,
        "format": "JIT .aimodel (function main)", "intended_runtime": "macos", "aot": False,
        "precision": args.precision, "bucket_s": args.sec, "shapes": shapes, **io_contract(args.sec), "files": files,
        "bytes": bundle_bytes, "module_weights": weights,
        "bytes_over_module_parameter_bytes": bundle_bytes / weights["parameters"],
        "op_counts": converted["op_counts"], "ops_total": converted["ops_total"],
        "result_element_types": converted["result_element_types"],
        "seconds": {"weights_load": load_s, "torch_export": export_s, "decompositions": decompose_s,
                    "export_gate": gate_s, **converted["seconds"], "total": time.perf_counter() - started},
        "export_gate": {"status": status, "record": "provenance/export-gate.json", **summary},
        "weights_record": load_record, "source_sha256": source,
        "reference_json_sha256": sha256_file(stage / "reference.json"),
        "metadata_json_sha256": sha256_file(stage / "metadata.json"),
        "filterbank_sha256": sha256_file(stage / mel_host.FILTERBANK_FILE),
        "code_sha256": {f: sha256_file(HERE / f) for f in CODE_FILES},
        "runtime_gate": "NOT RUN — aot_audio.py, then audio_check.py --stage runtime",
        "environment": environment(("torch", "coreai-torch", "coreai-core", "numpy", "safetensors", "soundfile")),
    }
    write_json(stage / "provenance" / "export-manifest.json", record)
    os.rename(stage, out_dir)
    print(json.dumps({"status": status, "folder": str(out_dir), "bundle": name, "bytes": bundle_bytes,
                      "module_parameter_bytes": weights["parameters"], "ops": converted["ops_total"],
                      "seconds": {k: round(v, 1) for k, v in record["seconds"].items()}}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
