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
"""Stage 3: export laya multilingual (mmBERT-base 322M + typed decision head) as one static Core AI bundle.

    python3 export_laya.py --window 256 --dtype fp32  --target macos
    python3 export_laya.py --window 256 --dtype wfp16 --target macos      # fp16 weight storage, fp32 compute
    python3 export_laya.py --window 256 --dtype wfp16 --target ios        # the same portable JIT .aimodel, copied
    python3 export_laya.py --window 256 --dtype fp16  --target macos --measure-only   # the fp16 recipe, for the record

One multifunction .aimodel, batch 1, static window S:
  main  input_ids [1,S] int32, attention_mask [1,S] int32, qtype_onehot [1,3] float32
        -> token_logits [1,S] float32, pooled_cls [1,768] float32
  act   pooled_cls [1,768] float32, feats [1,4] float32 -> act_logits [1,2] float32 (fp32 in every variant)

Precisions: fp32; wfp16 = fp16 weight storage (the checkpoint is F16, so exact) with fp32 compute,
half the bytes; fp16 = the fp16 recipe (fp16 compute, RoPE application and softmax in fp32), which
misses the answer bar in torch — `--measure-only` builds it anyway, marked FAIL, never a candidate.

Order: the authoring records for the window must PASS — the fp32 graph with its negative controls,
and for wfp16 its own record too (rows, two-tier layer gate, pad isolation). The graph is re-authored
from the raw safetensors (`_laya_model.py`), never from
transformers. Both functions are torch-exported and decomposed with coreai_torch's table, and the
decomposed programs run the same 201-row gate BEFORE conversion (fp32 / wfp16: answer + tensor gated;
fp16: answer gated, tensor bar reported). Then TorchConverter (main + act) -> optimize -> save_asset.

`--target ios` does not convert again: it copies the already-gated macos variant byte for byte (the
portable JIT bundle runs on the phone as it is; the h18p AOT is a later stage) and records its hashes.
An iOS-targeted folder is never loaded on the Mac.

Layout mirrors the HF repo: <exports>/laya-multilingual/<target>/<dtype>-s<S>/ with the bundle,
tokenizer/ (the checkpoint's tokenizer.json + tokenizer_config.json, unmodified), metadata.json (the
`decision` block the kit reads), reference.json (this window's 201 rows + the official batch-1 tensors)
and provenance/ (export-manifest.json with file hashes and op counts, export-gate.json with every row).
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
from _common import (CLS_ID, HEAD_MAX_LEN, HIDDEN, MASK_ID, MODEL_ID, MODEL_SHA, OPTION_TEXT_TOKENS, PAD_ID,  # noqa: E402
                     SEP_ID, SOURCE_MAX_LEN, SUBFOLDER, environment, export_root, file_inventory, fixtures_dir,
                     hashes, load_calibration, load_rows, oracle_dir, results_dir, row_inputs, sha256_of,
                     verify_hashes, verify_source, write_json)
from _gate_metrics import POLICY, evaluate_row, summarize  # noqa: E402
from _laya_host import act_features, gather_markers  # noqa: E402
from _laya_model import ActGraph, FP16_RECIPE, MainGraph, load_laya  # noqa: E402

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
MAIN_INPUTS = ["input_ids", "attention_mask", "qtype_onehot"]
MAIN_OUTPUTS = ["token_logits", "pooled_cls"]
ACT_INPUTS = ["pooled_cls", "feats"]
ACT_OUTPUTS = ["act_logits"]


def export_programs(model, window: int, example_row: dict):
    x = row_inputs(example_row, window)
    main_kwargs = {k: torch.from_numpy(v) for k, v in x.items()}
    act_kwargs = {"pooled_cls": torch.zeros(1, HIDDEN, dtype=torch.float32), "feats": torch.zeros(1, 4, dtype=torch.float32)}
    with torch.no_grad():
        main = torch.export.export(MainGraph(model).eval(), args=(), kwargs=main_kwargs).run_decompositions(get_decomp_table())
        act = torch.export.export(ActGraph(model).eval(), args=(), kwargs=act_kwargs).run_decompositions(get_decomp_table())
    return main, act


def gate_programs(main, act, rows, window, config, oracle) -> tuple[list[dict], dict]:
    """The decomposed programs through the deployed pipeline on every row, before any conversion."""
    main_module, act_module = main.module(), act.module()
    records = []
    for index, row in enumerate(rows):
        t0 = time.perf_counter()
        with torch.no_grad():
            token_logits, pooled = main_module(**{k: torch.from_numpy(v) for k, v in row_inputs(row, window).items()})
            marker = gather_markers(token_logits[0].numpy(), row["marker_positions"])
            feats = act_features(marker)
            act_logits = act_module(pooled_cls=pooled, feats=torch.from_numpy(feats))[0].numpy()
        k = row["K"]
        record = evaluate_row(row, marker, act_logits, config, tensor_reference={
            "marker_logits": oracle["marker_logits"][index, :k], "act_logits": oracle["act_logits"][index]})
        record["ms"] = (time.perf_counter() - t0) * 1000
        record["pooled_cls_max_abs_vs_oracle"] = float(np.max(np.abs(pooled[0].numpy().astype(np.float64) - oracle["pooled_cls"][index])))
        records.append(record)
    return records, summarize(records)


def convert(main, act, description: str, out: Path) -> dict:
    program = (TorchConverter()
               .add_exported_program(exported_program=main, input_names=MAIN_INPUTS, output_names=MAIN_OUTPUTS,
                                     entrypoint_name="main")
               .add_exported_program(exported_program=act, input_names=ACT_INPUTS, output_names=ACT_OUTPUTS,
                                     entrypoint_name="act")
               .to_coreai())
    t0 = time.perf_counter()
    program.optimize()
    optimize_seconds = time.perf_counter() - t0
    module = program._mlir_module
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
    metadata.author = "mlboydaisuke (coreai-model-zoo); weights Convai Innovations (laya)"
    metadata.license = "Apache-2.0"
    metadata.model_description = description
    t0 = time.perf_counter()
    program.save_asset(out, metadata)
    return {"op_counts": dict(sorted(counts.items())), "optimize_seconds": optimize_seconds,
            "save_seconds": time.perf_counter() - t0}


def io_contract(window: int) -> dict:
    return {
        "inputs": {"input_ids": {"dtype": "int32", "shape": [1, window], "padding": f"right, PAD {PAD_ID}"},
                   "attention_mask": {"dtype": "int32", "shape": [1, window], "values": "1 real token, 0 padding"},
                   "qtype_onehot": {"dtype": "float32", "shape": [1, 3], "order": ["choice", "score", "noul"]}},
        "outputs": {"token_logits": {"dtype": "float32", "shape": [1, window], "read": "at the option marker positions"},
                    "pooled_cls": {"dtype": "float32", "shape": [1, HIDDEN], "read": "input of the act function"}},
        "act": {"inputs": {"pooled_cls": {"dtype": "float32", "shape": [1, HIDDEN]},
                           "feats": {"dtype": "float32", "shape": [1, 4],
                                     "order": ["top1", "top1_minus_top2", "entropy_over_ln_max_k_2", "max_k_2_over_255"],
                                     "from": "softmax of the RAW marker logits (no temperature)"}},
                "outputs": {"act_logits": {"dtype": "float32", "shape": [1, 2], "class_0": "direct answer"}}},
    }


def bundle_metadata(window: int, bundle_name: str, source_config: dict) -> dict:
    calibration = load_calibration()
    return {
        "metadata_version": "0.2",
        "kind": "encoder",
        "decision": {
            "head": "encoder", "layout": "laya", "window": window, "head_max_len": HEAD_MAX_LEN,
            "source_max_len": SOURCE_MAX_LEN, "option_text_tokens": OPTION_TEXT_TOKENS,
            "cls_token_id": CLS_ID, "sep_token_id": SEP_ID, "pad_token_id": PAD_ID, "mask_token_id": MASK_ID,
            "functions": {"main": "main", "act": "act"},
            **io_contract(window),
            "source_temperature": source_config["temperature"],
            "temperature": calibration["temperature"],
            "temperature_by_options": calibration["temperature_by_options"],
            "calibration_provenance": ("litert-community/Laya-Multilingual-LiteRT laya_ml_calibration.json "
                                       "(4415 examples, six balanced fits; choice:6-10 fixed at T=1)"),
        },
        "source": {"hf_model_id": MODEL_ID, "hf_revision": MODEL_SHA, "subfolder": SUBFOLDER},
        "assets": {"main": bundle_name},
        "language": {"tokenizer": f"{MODEL_ID}/{SUBFOLDER}/tokenizer", "vocab_size": 256000, "max_context_length": window},
    }


def reference(rows, window, oracle, oracle_hash: str) -> dict:
    keep = ("row_id", "fixture_id", "question_id", "family", "language", "primary_language", "question", "qtype", "K",
            "sequence_length", "sequence_ids", "marker_positions", "probabilities", "act_probability", "official_answer",
            "temperature_bucket", "state_right_truncated")
    out_rows = []
    for index, row in enumerate(rows):
        entry = {k: row[k] for k in keep}
        entry["oracle"] = {"marker_logits": [float(v) for v in oracle["marker_logits"][index, :row["K"]]],
                           "act_logits": [float(v) for v in oracle["act_logits"][index]]}
        out_rows.append(entry)
    return {"schema": "coreai-encoder-fixtures/1", "model": MODEL_ID, "model_sha": MODEL_SHA, "subfolder": SUBFOLDER,
            "window": window, "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {},
            "answers": "the official laya 0.3.4 Agent.predict, batched per fixture, max_len = window (frozen LiteRT-lane fixture)",
            "oracle": "the official DecisionModel on each row alone, batch 1, right-padded to this window, CPU fp32",
            "fixture_sha256": sha256_of(fixtures_dir() / f"ml_rows_s{window}.json"), "oracle_outputs_sha256": oracle_hash,
            "rows": out_rows}


def copy_for_ios(args) -> int:
    source = export_root() / "macos" / f"{args.dtype}-s{args.window}"
    manifest = json.loads((source / "provenance" / "export-manifest.json").read_text())
    assert manifest["status"] == "PASS" and manifest["intended_runtime"] == "macos", "export and gate the macos variant first"
    out_dir = export_root() / "ios" / f"{args.dtype}-s{args.window}"
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_dir} exists; pass --overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for name in (manifest["bundle"], "tokenizer", "metadata.json", "reference.json", "provenance/export-gate.json"):
        src = source / name
        (out_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, out_dir / name)
    bundle = out_dir / manifest["bundle"]
    files = file_inventory(bundle)
    assert files == manifest["files"], "the copy differs from the gated macos bundle"
    record = {k: manifest[k] for k in ("status", "model", "model_sha", "bundle", "format", "precision", "window",
                                       "fp32_islands", "inputs", "outputs", "act", "files", "bytes", "op_counts",
                                       "torch_export_gate")}
    record.update(intended_runtime="ios", aot=False, copied_from=str(Path("macos") / source.name),
                  copied_from_manifest_sha256=sha256_of(source / "provenance" / "export-manifest.json"),
                  runtime_gate="NOT RUN on the Mac by rule — the device gate runs this folder on the iPhone",
                  environment=environment())
    write_json(out_dir / "provenance" / "export-manifest.json", record)
    print(json.dumps({"status": record["status"], "copied_from": record["copied_from"], "bytes": record["bytes"]}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", type=int, required=True, choices=[256, 512])
    parser.add_argument("--dtype", choices=["fp32", "wfp16", "fp16"], default="fp32")
    parser.add_argument("--fp32-islands", default=",".join(FP16_RECIPE),
                        help="fp16 only: the parts kept in fp32 (default: the recipe — RoPE application and softmax)")
    parser.add_argument("--target", choices=["macos", "ios"], default="macos")
    parser.add_argument("--measure-only", action="store_true",
                        help="convert even if the export gate fails; the manifest says FAIL and why")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.target == "ios":
        return copy_for_ios(args)
    torch.set_num_threads(4)
    started = time.perf_counter()
    source = verify_source()

    authoring_paths = [results_dir() / f"authoring_fp32_s{args.window}.json"]
    if args.dtype == "wfp16":
        authoring_paths.append(results_dir() / f"authoring_wfp16_s{args.window}.json")
    for path in authoring_paths:
        authoring = json.loads(path.read_text())
        if authoring["status"] != "PASS":
            raise SystemExit(f"{path.name} is {authoring['status']}: {authoring['failures']} — the authoring gate must pass first")
        verify_hashes(authoring["input_hashes"])
    fp32_record = json.loads(authoring_paths[0].read_text())
    assert fp32_record["negative_controls_run"] and all(c["caught"] for c in fp32_record["negative_controls"].values()), \
        "the fp32 authoring record must carry caught negative controls"

    oracle_record = json.loads((results_dir() / "oracle.json").read_text())
    outputs = oracle_dir() / f"outputs_s{args.window}.npz"
    oracle_hash = oracle_record["output_hashes"][str(outputs)]
    verify_hashes({str(outputs): oracle_hash})
    with np.load(outputs) as data:
        oracle = {k: data[k] for k in data.files}
    rows = load_rows(args.window)
    agent_config = json.loads((source / "rl_agent_config.json").read_text())
    config = {"temperature": agent_config["temperature"], "temperature_by_options": agent_config["temperature_by_options"]}
    assert config["temperature"] == [1.0, 1.0, 1.0] and not config["temperature_by_options"]

    name = f"laya_ml_{args.dtype}_s{args.window}"
    out_dir = export_root() / args.target / f"{args.dtype}-s{args.window}"
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_dir} exists; pass --overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    bundle = out_dir / f"{name}.aimodel"

    islands = tuple(sorted(filter(None, args.fp32_islands.split(",")))) if args.dtype == "fp16" else ("all",)
    model = load_laya(source, args.window, precision=args.dtype, fp32_islands=islands if args.dtype == "fp16" else ())
    t0 = time.perf_counter()
    main_program, act_program = export_programs(model, args.window, rows[0])
    export_seconds = time.perf_counter() - t0
    print(f"torch.export + decomposition {export_seconds:.0f} s", flush=True)
    records, summary = gate_programs(main_program, act_program, rows, args.window, config, oracle)
    gate_status = summary["answer_status"] if args.dtype == "fp16" else (
        "PASS" if summary["answer_status"] == summary["tensor_status"] == "PASS" else "FAIL")
    print(f"export gate {gate_status}: argmax {summary['argmax_identical']}/{summary['choice_score_rows']}, "
          f"max dp {summary['max_probability_error']:.2e}, act p {summary['max_act_probability_error']:.2e}, "
          f"marker {summary['max_marker_abs_error']:.2e}, act rel {summary['max_act_relative_error']:.2e} "
          f"(tensor {summary['tensor_status']})", flush=True)
    gate_record = {"status": gate_status, "stage": "export (torch-exported + decomposed, before conversion)",
                   "window": args.window, "precision": args.dtype, "fp32_islands": list(islands), "policy": POLICY,
                   "tensor_bar": "reported (fp16: the answer level is the ship bar)" if args.dtype == "fp16" else "gated",
                   "summary": summary, "environment": environment(), "rows": records}
    write_json(out_dir / "provenance" / "export-gate.json", gate_record)
    if gate_status != "PASS" and not args.measure_only:
        raise SystemExit(f"export gate failed before conversion: {summary['answer_failures'][:5]} {summary.get('tensor_failures', [])[:5]}")

    description = (f"laya multilingual decision encoder (mmBERT-base + typed head); {args.dtype}; S={args.window}; "
                   f"functions main + act; source {MODEL_ID}@{MODEL_SHA}/{SUBFOLDER}")
    t0 = time.perf_counter()
    converted = convert(main_program, act_program, description, bundle)
    convert_seconds = time.perf_counter() - t0
    print(f"converted + optimized + saved {convert_seconds:.0f} s -> {bundle}", flush=True)

    (out_dir / "tokenizer").mkdir()
    for file in TOKENIZER_FILES:
        shutil.copy2(source / "tokenizer" / file, out_dir / "tokenizer" / file)
    write_json(out_dir / "metadata.json", bundle_metadata(args.window, bundle.name, config))
    write_json(out_dir / "reference.json", reference(rows, args.window, oracle, oracle_hash))

    record = {
        "status": gate_status, "measure_only": bool(args.measure_only and gate_status != "PASS"), "model": MODEL_ID, "model_sha": MODEL_SHA, "subfolder": SUBFOLDER, "bundle": bundle.name,
        "intended_runtime": args.target, "format": "JIT .aimodel (portable; multifunction main + act)", "aot": False,
        "precision": args.dtype, "fp32_islands": list(islands), "act_precision": "fp32", "window": args.window,
        **io_contract(args.window), "files": file_inventory(bundle), "op_counts": converted["op_counts"],
        "seconds": {"torch_export": export_seconds, "convert_total": convert_seconds,
                    "optimize": converted["optimize_seconds"], "save": converted["save_seconds"],
                    "total": time.perf_counter() - started},
        "torch_export_gate": {"status": gate_status, "record": "provenance/export-gate.json",
                              **{k: summary.get(k) for k in ("argmax_identical", "choice_score_rows", "max_probability_error",
                                                             "max_act_probability_error", "max_marker_abs_error",
                                                             "max_act_relative_error", "tensor_status")}},
        "authoring_records": [str(p) for p in authoring_paths],
        "runtime_gate": "NOT RUN — gate_laya_runtime.py <this folder> --compute cpu_only|gpu|neural_engine",
        "environment": environment(),
        "input_hashes": hashes([*authoring_paths, outputs, Path(__file__), Path(__file__).parent / "_laya_model.py",
                                Path(__file__).parent / "_laya_host.py", Path(__file__).parent / "_gate_metrics.py"]),
    }
    record["bytes"] = sum(f["bytes"] for f in record["files"])
    write_json(out_dir / "provenance" / "export-manifest.json", record)
    print(json.dumps({k: record[k] for k in ("status", "bundle", "precision", "window", "bytes")} | {"seconds": record["seconds"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
