#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.9.0",
#     "coreai-torch==0.4.1",
#     "coreai-core==1.0.0b2",
#     "safetensors>=0.7.0",
#     "numpy>=2.2",
#     "tokenizers>=0.22",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
"""Stage 3: export Julia-1 (mmBERT-small 144M + typed decision head) as one static Core AI bundle.

    python3 export_julia.py --window 1024 --dtype fp32
    python3 export_julia.py --window 1024 --dtype wfp16     # fp16 weight storage (rounded), fp32 compute
    python3 export_julia.py --window 512  --dtype fp32

One .aimodel, batch 1, static window S, one function:
  main  input_ids [1,S] int32, attention_mask [1,S] int32, qtype_onehot [1,3] float32
        -> token_logits [1,S] float32

Order: the authoring records for the window must PASS — the fp32 graph with its negative controls, and
for wfp16 its own record too. The graph is re-authored from the raw safetensors (`_julia_model.py`), never
from transformers. The function is torch-exported and decomposed with coreai_torch's table, and the
decomposed program runs the same row gate BEFORE conversion (every row, or every --rows-stride-th row plus every SUBSET and
window-filling row; the runtime gate then runs every row on the bundle). Then TorchConverter -> optimize -> save_asset.
`--measure-only` converts even when that gate fails; the manifest then says FAIL, never a candidate.

Layout mirrors the HF repo: <exports>/julia-1/macos/<dtype>-s<S>/ with the bundle, tokenizer/ (the
checkpoint's tokenizer.json + tokenizer_config.json, unmodified), metadata.json (the `decision` block: the
graph contract and the host recipe), reference.json (this window's rows: ids, markers, the publisher's
raw logits) and provenance/ (export-manifest.json with file hashes and op counts, export-gate.json).
"""
import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (CLS_ID, HEAD_LENGTH, MASK_ID, MODEL_ID, MODEL_SHA, OPTION_TOKENS, PAD_ID, SEP_ID,  # noqa: E402
                     environment, export_root, file_inventory, hashes, load_rows, oracle_dir, results_dir,
                     row_inputs, sha256_of, subset, verify_hashes, verify_source, write_json)
from _gate_metrics import POLICY, evaluate_row, summarize  # noqa: E402
from _julia_host import gather_markers  # noqa: E402
from _julia_model import MainGraph, load_julia  # noqa: E402

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
MAIN_INPUTS = ["input_ids", "attention_mask", "qtype_onehot"]
MAIN_OUTPUTS = ["token_logits"]


def export_program(model, window: int, example_row: dict):
    kwargs = {k: torch.from_numpy(v) for k, v in row_inputs(example_row, window).items()}
    with torch.no_grad():
        return torch.export.export(MainGraph(model).eval(), args=(), kwargs=kwargs).run_decompositions(get_decomp_table())


def gate_program(program, rows, window) -> tuple[list[dict], dict]:
    """The decomposed program through the deployed pipeline on every row, before any conversion."""
    module = program.module()
    records = []
    for row in rows:
        t0 = time.perf_counter()
        with torch.no_grad():
            token_logits = module(**{k: torch.from_numpy(v) for k, v in row_inputs(row, window).items()})
        record = evaluate_row(row, gather_markers(token_logits[0].numpy(), row["markers"]))
        record["ms"] = (time.perf_counter() - t0) * 1000
        records.append(record)
    return records, summarize(records)


def convert(program, description: str, out: Path) -> dict:
    converted = (TorchConverter()
                 .add_exported_program(exported_program=program, input_names=MAIN_INPUTS, output_names=MAIN_OUTPUTS,
                                       entrypoint_name="main")
                 .to_coreai())
    t0 = time.perf_counter()
    converted.optimize()
    optimize_seconds = time.perf_counter() - t0
    module = converted._mlir_module
    assert module.operation.verify()
    counts: Counter = Counter()

    def inspect(operation):
        counts[operation.name] += 1
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    inspect(child.operation)

    inspect(module.operation)
    metadata = AIModelAssetMetadata()
    metadata.author = "mlboydaisuke (coreai-model-zoo); weights Supersonic Labs (Julia-1)"
    metadata.license = "Apache-2.0"
    metadata.model_description = description
    t0 = time.perf_counter()
    converted.save_asset(out, metadata)
    return {"op_counts": dict(sorted(counts.items())), "optimize_seconds": optimize_seconds,
            "save_seconds": time.perf_counter() - t0}


def io_contract(window: int) -> dict:
    return {
        "inputs": {"input_ids": {"dtype": "int32", "shape": [1, window], "padding": f"right, PAD {PAD_ID}"},
                   "attention_mask": {"dtype": "int32", "shape": [1, window], "values": "1 real token, 0 padding"},
                   "qtype_onehot": {"dtype": "float32", "shape": [1, 3], "order": ["choice", "score", "noul"]}},
        "outputs": {"token_logits": {"dtype": "float32", "shape": [1, window], "read": "at the option marker positions"}},
    }


def bundle_metadata(window: int, bundle_name: str, precision: str) -> dict:
    return {
        "metadata_version": "0.2",
        "kind": "encoder",
        "decision": {
            "head": "encoder", "layout": "julia", "window": window, "head_max_len": HEAD_LENGTH[window],
            "option_text_tokens": OPTION_TOKENS, "strict_encoding": True,
            "cls_token_id": CLS_ID, "sep_token_id": SEP_ID, "pad_token_id": PAD_ID, "mask_token_id": MASK_ID,
            "functions": {"main": "main"}, **io_contract(window),
            "options": ("the descriptions as given: choice = the criteria mapping's values in order (answer = its id), "
                        "score = the rubric in order, noul = [criteria.false, criteria.true] or the literal "
                        "['false', 'true'] when a noul question has no criteria"),
            "sequence": ("[CLS 2] tok('{type} question: {instructions}') [SEP 1] ([MASK 4] tok(' ' + option)[:48])... "
                         "[SEP 1] tok(state) [SEP 1]; state = json.dumps(state, ensure_ascii=False) when not text; "
                         "strict: a row that does not fit, or text holding the literal '<mask>', is refused, never "
                                         "cut or rewritten (the publisher's non-strict builder turns '<mask>' into ' ')"),
            "readout": "softmax of the raw marker logits (T = 1); choice = argmax id, score = sum(i * p_i), noul = p[true]",
            "source_temperature": [1.0, 1.0, 1.0], "calibration": None,
            "precision": precision,
        },
        "source": {"hf_model_id": MODEL_ID, "hf_revision": MODEL_SHA},
        "assets": {"main": bundle_name},
        "language": {"tokenizer": f"{MODEL_ID}/tokenizer", "vocab_size": 256000, "max_context_length": window},
    }


def reference(rows, window) -> dict:
    keep = ("row_id", "set", "type", "keys", "gold", "ids", "markers", "publisher_logits", "marker_logits")
    return {"schema": "julia-coreai-reference/1", "model": MODEL_ID, "model_sha": MODEL_SHA, "window": window,
            "head_max_len": HEAD_LENGTH[window], "temperature": 1.0,
            "answers": ("publisher_logits = the publisher's FastEngine.logits, one question per forward, CPU fp32 "
                        "(the run that reproduces 426 / 542 / 483 on typed-decisions)"),
            "tensor": "marker_logits = the publisher's model on the same ids right-padded to this window, batch 1, CPU fp32",
            "rows_sha256": sha256_of(oracle_dir() / "rows.json"),
            "rows": [{k: row.get(k) for k in keep} for row in rows]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", type=int, required=True, choices=[512, 1024])
    parser.add_argument("--dtype", choices=["fp32", "wfp16"], default="fp32")
    parser.add_argument("--measure-only", action="store_true",
                        help="convert even if the export gate fails; the manifest says FAIL and why")
    parser.add_argument("--rows-stride", type=int, default=1,
                        help="gate every n-th row before conversion (plus every SUBSET and window-filling row); "
                             "the runtime gate runs every row on the bundle")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    started = time.perf_counter()
    source = verify_source()

    authoring_paths = [results_dir() / f"authoring_fp32_s{args.window}.json"]
    if args.dtype == "wfp16":
        authoring_paths.append(results_dir() / f"authoring_wfp16_s{args.window}.json")
    for path in authoring_paths:
        authoring = json.loads(path.read_text())
        if authoring["status"] != "PASS" and not args.measure_only:
            raise SystemExit(f"{path.name} is {authoring['status']}: {authoring['failures']} — the authoring gate must pass first")
        verify_hashes(authoring["input_hashes"])
    fp32_record = json.loads(authoring_paths[0].read_text())
    assert fp32_record["negative_controls_run"] and all(c["caught"] for c in fp32_record["negative_controls"].values()), \
        "the fp32 authoring record must carry caught negative controls"

    rows = load_rows(args.window)
    name = f"julia1_{args.dtype}_s{args.window}"
    out_dir = export_root() / "macos" / f"{args.dtype}-s{args.window}"
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_dir} exists; pass --overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    bundle = out_dir / f"{name}.aimodel"

    model = load_julia(source, args.window, precision=args.dtype)
    t0 = time.perf_counter()
    program = export_program(model, args.window, rows[0])
    export_seconds = time.perf_counter() - t0
    print(f"torch.export + decomposition {export_seconds:.0f} s", flush=True)
    keep = set(subset(args.window))
    gated = [r for i, r in enumerate(rows) if i % args.rows_stride == 0 or r["row_id"] in keep or r["set"] == "fill"]
    records, summary = gate_program(program, gated, args.window)
    gate_status = "PASS" if summary["answer_status"] == summary["tensor_status"] == "PASS" else "FAIL"
    print(f"export gate {gate_status}: {summary['rows']} rows, argmax {summary['argmax_identical']}, "
          f"max dp {summary['max_probability_error']:.2e}, marker {summary['max_marker_abs_error']:.2e}, "
          f"typed {json.dumps(summary.get('typed_accuracy'))}", flush=True)
    gate_record = {"status": gate_status, "stage": "export (torch-exported + decomposed, before conversion)",
                   "window": args.window, "precision": args.dtype, "policy": POLICY, "rows_stride": args.rows_stride,
                   "rows_gated": len(gated), "rows_in_window": len(rows), "summary": summary,
                   "environment": environment(), "rows": records}
    write_json(out_dir / "provenance" / "export-gate.json", gate_record)
    if gate_status != "PASS" and not args.measure_only:
        raise SystemExit(f"export gate failed before conversion: {summary['answer_failures'][:5]} {summary['tensor_failures'][:5]}")

    description = (f"Julia-1 decision encoder (mmBERT-small + typed head); {args.dtype}; S={args.window}; "
                   f"function main; source {MODEL_ID}@{MODEL_SHA}")
    t0 = time.perf_counter()
    converted = convert(program, description, bundle)
    convert_seconds = time.perf_counter() - t0
    print(f"converted + optimized + saved {convert_seconds:.0f} s -> {bundle}", flush=True)

    (out_dir / "tokenizer").mkdir()
    for file in TOKENIZER_FILES:
        shutil.copy2(source / "tokenizer" / file, out_dir / "tokenizer" / file)
    write_json(out_dir / "metadata.json", bundle_metadata(args.window, bundle.name, args.dtype))
    write_json(out_dir / "reference.json", reference(rows, args.window))

    record = {
        "status": gate_status, "measure_only": bool(args.measure_only and gate_status != "PASS"), "model": MODEL_ID,
        "model_sha": MODEL_SHA, "bundle": bundle.name, "intended_runtime": "macos",
        "format": "JIT .aimodel (portable; one function main)", "aot": False, "precision": args.dtype,
        "window": args.window, **io_contract(args.window), "files": file_inventory(bundle),
        "op_counts": converted["op_counts"],
        "seconds": {"torch_export": export_seconds, "convert_total": convert_seconds,
                    "optimize": converted["optimize_seconds"], "save": converted["save_seconds"],
                    "total": time.perf_counter() - started},
        "torch_export_gate": {"status": gate_status, "record": "provenance/export-gate.json", "rows_stride": args.rows_stride,
                              "rows_in_window": len(rows),
                              **{k: summary.get(k) for k in ("rows", "argmax_identical", "max_probability_error",
                                                             "max_marker_abs_error", "typed_accuracy")}},
        "authoring_records": [str(p) for p in authoring_paths],
        "runtime_gate": "NOT RUN — gate_julia_runtime.py <this folder> --compute cpu_only|gpu",
        "environment": environment(),
        "input_hashes": hashes([*authoring_paths, oracle_dir() / "rows.json", Path(__file__),
                                Path(__file__).parent / "_julia_model.py", Path(__file__).parent / "_julia_host.py",
                                Path(__file__).parent / "_gate_metrics.py"]),
    }
    record["bytes"] = sum(f["bytes"] for f in record["files"])
    write_json(out_dir / "provenance" / "export-manifest.json", record)
    print(json.dumps({k: record[k] for k in ("status", "bundle", "precision", "window", "bytes")} | {"seconds": record["seconds"]}, indent=1))
    return 0 if gate_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
