#!/usr/bin/env python3
"""Export d1-omni-600M's decision graph (trunk + head, one question row, static length L) as a Core AI bundle.

    python3 conversion/d1_omni/export_decide.py --precision fp32  --seq-len 256
    python3 conversion/d1_omni/export_decide.py --precision wfp16 --seq-len 512
    python3 conversion/d1_omni/export_decide.py --precision fp16  --seq-len 4096
    python3 conversion/d1_omni/export_decide.py --precision int8  --seq-len 256
    python3 conversion/d1_omni/export_decide.py --precision fp16  --seq-len 64     # round 12: L64 / L128

Run with the shared export venv (coreai-models/.venv: torch 2.9.0, coreai-torch 0.4.1, coreai-core 1.0.0b2,
coreai-opt 0.2.1). The graph is d1_omni_model.D1Decide (plain torch, no transformers), its weights the pinned
snapshot's model.safetensors (sha256 verified on load).

Order: load_d1_decide(snapshot, L, precision, verify_sha256=True) [int8: quantize] -> torch.export with one gate
row's inputs (host.graph_inputs) -> run_decompositions(get_decomp_table()) -> the export gate -> TorchConverter
(one function, main) -> optimize -> save_asset.

Rows (ref/records_ref.json, the publisher's fp32 model on every fixture row; an image or audio row takes the
reference's own prefix from ref/npz). Each row is tagged with its set:
  native   every reference row whose bucket is L (at L64 / L128, round 12, the bucket is read from host.ALL_BUCKETS:
           a row of <= 64 positions is native at 64, of 65..128 at 128; at L >= 256 from host.BUCKETS, as before)
  pad      PAD_ROWS rows of the smaller buckets, padded to L (the bucket is invariant to padding): from the nearest
           smaller non-empty bucket first, all of its text / image rows if they fit, else that many at even spacing
           in reference order; then the next bucket down. fp16 at L >= 2048 takes PAD_ROWS_FP16_LONG (torch computes
           fp16 slowly on the CPU, about a minute a row at 4096)
  prefix   for the runtime gate only: the 10 audio-prefix rows and card_cats (image prefix), padded to L, when they
           are not native or pad rows already
The export gate runs native + pad; reference.json carries all three (runtime_check.py runs them all). At L=256 the
sets are round 3's: the 436 native rows (audio included), nothing else.

The export gate runs the decomposed program on its rows and compares it with the publisher's fp32 model, and with
the eager module of the same precision (what export + decomposition changed; the eager module is also compared with
the reference). fp32 is held to round 2's eager bar (_metrics.FP32_BAR: argmax on every row, max |dp| <= 2e-5,
marker logits <= 1e-3) and stops before conversion if it misses (--measure-only converts anyway). wfp16 / fp16 /
int8 are measured against the ship bar (_metrics.SHIP_BAR) and converted whatever it says: the runtime gate decides;
the manifest carries the verdict.

Precision (d1_omni_model.set_precision): fp32; wfp16 = fp16 weight storage, fp32 compute (each weight cast at its
use); fp16 = fp16 compute with RMSNorm, LayerNorm, softmax and the scorer in fp32; int8 = the fp16 frame with every
trunk and head linear weight-only int8, symmetric, per block of 32 input channels (coreai-opt's eager quantizer
through coreai_models.export.compression.quantize_pytorch_model, as conversion/export_lfm25vl_pipelined.py runs it);
the embedding, the norms and the scorer stay as in fp16 (d1_omni_model.set_quant_mode).

Attention: above d1_omni_model.KEY_BLOCK (2048) keys the graph computes each attention in key blocks sharing one
max (d1_omni_model, "Key blocks"); the manifest records the graph's softmax / reduce op counts as the evidence.

Layout (the lane's work dir, not the shared exports dir): $ZOO_WORK_ROOT/_d1_omni/bundles/d1-omni-600m/macos/
<precision>-L<L>/ holding
  d1_omni_decide_<precision>_L<L>.aimodel   the bundle, function main
  metadata.json                            the decision contract a host reads (inputs, token ids, temperatures)
  tokenizer/                               the snapshot's tokenizer.json + tokenizer_config.json, unmodified
  reference.json                           this bucket's rows: question, ids, markers, prefix npz, the oracle
  provenance/export-manifest.json          bytes, op counts, seconds, environment, input hashes
  provenance/export-gate.json              every row of the export gate
The folder is built in a staging folder next to it and renamed at the end; an existing folder is never replaced.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import hf_snapshot, work_path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import d1_omni_model as dm  # noqa: E402
import host  # noqa: E402
from _fixtures import fixtures_path as fixtures_for  # noqa: E402
from _metrics import FP32_BAR, SHIP_BAR, row_record, summarize  # noqa: E402

WORK = work_path("_d1_omni")
FAMILY = "d1-omni-600m"
REFERENCE_SHA256 = "e7dba6f44d0452a6aea3e401746d417503056d6f468ff72b56c5511883436c5f"  # ref/records_ref.json (round 2)
INPUT_NAMES = ["input_ids", "prefix_embeds", "pad_mask", "prefix_mask", "keep_right", "qtype_onehot"]
OUTPUT_NAMES = ["scores"]
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
SEQ_LENS = (*host.SMALL_BUCKETS, 256, 512, 1024, 2048, 4096)  # 4096: 2,048-key blocks (d1_omni_model.KEY_BLOCK)
PRECISIONS = (*dm.PRECISIONS, "int8")
PAD_ROWS = 20
PAD_ROWS_FP16_LONG = 5
ATTENTIONS = 8  # 6 trunk + 2 head
INT8_BLOCK = 32
LICENSE = "LFM Open License v1.0"
AUTHOR = "mlboydaisuke (coreai-model-zoo); weights Liquid AI (d1-omni-600M)"
CODE_FILES = ("export_decide.py", "d1_omni_model.py", "host.py", "_metrics.py")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n")


def file_inventory(root: Path) -> list[dict]:
    return [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in sorted(root.rglob("*")) if p.is_file()]


def _command(argv: list[str]) -> str:
    try:
        return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable ({type(error).__name__})"


def environment(packages=("torch", "coreai-torch", "coreai-core", "coreai-opt", "numpy", "safetensors",
                          "tokenizers")) -> dict:
    from importlib import metadata

    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"date": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds"),
            "machine": _command(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "macos_version": _command(["sw_vers", "-productVersion"]),
            "macos_build": _command(["sw_vers", "-buildVersion"]),
            "python": platform.python_version(), "executable": sys.executable, "packages": versions,
            "torch_threads": torch.get_num_threads(), "argv": sys.argv, "pid": os.getpid()}


def bundle_dir(precision: str, seq_len: int) -> Path:
    return WORK / "bundles" / FAMILY / "macos" / f"{precision}-L{seq_len}"


def bundle_name(precision: str, seq_len: int) -> str:
    return f"d1_omni_decide_{precision}_L{seq_len}.aimodel"


def verify_source(snapshot: Path) -> dict:
    """The publisher's files this graph and host.py were written from, and the tokenizer, at the pinned revision."""
    seen = {}
    for name, digest in dm.SOURCE_SHA256.items():
        seen[name] = sha256_file(snapshot / name)
        if seen[name] != digest:
            raise SystemExit(f"{name} is not the pinned revision {dm.MODEL_SHA}: {seen[name]}")
    seen["tokenizer.json"] = sha256_file(snapshot / "tokenizer.json")
    if seen["tokenizer.json"] != host.TOKENIZER_SHA256:
        raise SystemExit(f"tokenizer.json is not the pinned file: {seen['tokenizer.json']}")
    seen["tokenizer_config.json"] = sha256_file(snapshot / "tokenizer_config.json")
    return seen


def pad_row_keys(candidates: dict[int, list], seq_len: int, n: int) -> list:
    """The pad set (module docstring): n rows of the buckets below seq_len, nearest bucket first; a bucket's rows all
    if they fit, else that many at even spacing (first and last included) in reference order."""
    picked = []
    for bucket in sorted((b for b in candidates if b < seq_len), reverse=True):
        need, rows = n - len(picked), candidates[bucket]
        if need <= 0:
            break
        if len(rows) <= need:
            picked += rows
        elif need == 1:
            picked.append(rows[0])
        else:
            index = [round(i * (len(rows) - 1) / (need - 1)) for i in range(need)]
            assert len(set(index)) == need, index
            picked += [rows[i] for i in index]
    return picked


def native_buckets(seq_len: int) -> tuple[int, ...]:
    """The bucket set a row's native bucket is read from: the shipped set (host.BUCKETS), or the set with round 12's
    small buckets (host.ALL_BUCKETS) when seq_len is one of them (a row of 47 positions is native at L64 there)."""
    return host.ALL_BUCKETS if seq_len in host.SMALL_BUCKETS else host.BUCKETS


def load_rows(seq_len: int, tok, n_pad: int) -> tuple[list[dict], list[host.Row], list[np.ndarray | None], dict]:
    """The gate rows of bucket seq_len (module docstring: native, pad, prefix): the reference.json entry, the host
    Row (ids and markers asserted equal to the reference's on every reference row) and the media prefix (from the
    reference's npz, sha256 checked). The reference's own bucket field is the shipped set's (host.BUCKETS); a row's
    native bucket is read from native_buckets(seq_len)."""
    buckets = native_buckets(seq_len)
    ref_path = WORK / "ref" / "records_ref.json"
    if sha256_file(ref_path) != REFERENCE_SHA256:
        raise SystemExit(f"{ref_path} is not round 2's reference ({REFERENCE_SHA256[:8]}…)")
    ref = json.loads(ref_path.read_text())
    if ref["model"]["revision"] != host.MODEL_SHA:
        raise SystemExit("the reference was made from another revision")
    fixtures_path = fixtures_for(ref["fixtures"]["sha256"])  # the version the reference read (_fixtures.py)
    fixtures = {r["id"]: r for r in json.loads(fixtures_path.read_text())["records"]}
    every = {}  # (id, qid, mode) -> (row dict, host Row, prefix), in reference order
    for entry in ref["records"]:
        req = fixtures[entry["id"]]["request"]
        built = host.request_rows(tok, req["state"], req["questions"], entry["mode"], entry["prefix"])
        prefix, npz = None, None
        if entry["mode"] != "text":
            meta = ref["npz"][entry["id"]]
            path = WORK / "ref" / "npz" / f"{entry['id']}.npz"
            if sha256_file(path) != meta["sha256"]:
                raise SystemExit(f"{path} differs from the reference's npz")
            with np.load(path) as z:
                prefix = np.asarray(z["prefix"], dtype=np.float32).copy()
                if int(z["prefix_len"]) != entry["prefix"] or prefix.shape != (entry["prefix"], 1024):
                    raise SystemExit(f"{path}: prefix {prefix.shape} != P {entry['prefix']}")
            npz = {"file": f"ref/npz/{entry['id']}.npz", "sha256": meta["sha256"]}
        for row, q in zip(built, entry["questions"], strict=True):
            if not (row.qid == q["qid"] and row.ids == q["ids"] and row.markers == q["markers"]
                    and row.prefix_len == q["prefix"] and row.positions == q["positions"]):
                raise SystemExit(f"host rows differ from the reference at {entry['id']}/{q['qid']}")
            if host.bucket_for(row.positions) != q["bucket"]:
                raise SystemExit(f"{entry['id']}/{q['qid']}: bucket {host.bucket_for(row.positions)} != {q['bucket']}")
            small = {"ship_bucket": q["bucket"]} if buckets != host.BUCKETS else {}
            every[(entry["id"], q["qid"], entry["mode"])] = ({
                "id": entry["id"], "qid": q["qid"], "mode": entry["mode"], "source": entry["source"],
                "public": entry["public"], "native": entry["native"], "type": q["type"], "K": q["K"],
                "calibrate": q["calibrate"], "temperature_key": q["temperature_key"], "T": q["T"],
                "question": {"type": row.question.type, "instructions": row.question.instructions,
                             "criteria": row.question.criteria},
                "prefix_len": row.prefix_len, "prefix_npz": npz, "ids": q["ids"], "markers": q["markers"],
                "positions": row.positions, "bucket": seq_len,
                "native_bucket": host.bucket_for(row.positions, buckets), **small,
                "near_tie": q["near_tie"], "top2_margin": q["top2_margin"], "argmax_index": q["argmax_index"],
                "gold": q.get("gold"),
                "oracle": {"logits_raw": q["logits_raw"], "probs": q["probs"], "probs_raw": q["probs_raw"]}},
                row, prefix)
    native = [k for k, v in every.items() if v[0]["native_bucket"] == seq_len]
    candidates = {}
    for k, v in every.items():
        if v[0]["native_bucket"] < seq_len and k[2] in ("text", "image"):
            candidates.setdefault(v[0]["native_bucket"], []).append(k)
    pad = pad_row_keys(candidates, seq_len, n_pad)
    taken = set(native) | set(pad)
    prefix_rows = [k for k, v in every.items() if k not in taken and v[0]["positions"] <= seq_len
                   and (k[2] == "audio" or k == ("card_cats", "cats", "image"))]
    rows, hrows, prefixes = [], [], []
    for name, keys in (("native", native), ("pad", pad), ("prefix", prefix_rows)):
        for k in keys:
            row, hrow, prefix = every[k]
            rows.append({**row, "set": name})
            hrows.append(hrow)
            prefixes.append(prefix)
    info = {"path": str(ref_path), "sha256": REFERENCE_SHA256, "fixtures_sha256": ref["fixtures"]["sha256"],
            "written": ref["written"], "rows_total": sum(len(e["questions"]) for e in ref["records"]),
            "sets": {"native": len(native), "pad": len(pad), "prefix": len(prefix_rows)},
            "native_buckets": list(buckets),
            "rows_longer_than_L": sum(v[0]["positions"] > seq_len for v in every.values()),
            "pad_rows_requested": n_pad,
            "pad_rule": "nearest smaller non-empty bucket first; all its text / image rows if they fit, else that many "
                        "at even spacing in reference order (first and last included); then the next bucket down",
            "prefix_rule": "audio-prefix rows and card_cats not already native / pad, padded to L (runtime gate only)"}
    return rows, hrows, prefixes, info


def as_tensors(inputs: dict) -> dict:
    return {k: torch.from_numpy(inputs[k]) for k in INPUT_NAMES}


def scores_of(out) -> np.ndarray:
    if isinstance(out, (tuple, list)):
        assert len(out) == 1, len(out)
        out = out[0]
    return out.detach().numpy().reshape(-1).copy()


def export_gate(program, model, rows, hrows, prefixes, seq_len: int) -> tuple[list[dict], dict]:
    """The decomposed program on every native / pad row against the oracle, and against the eager module (same
    precision); the eager module against the oracle as well."""
    module = program.module()
    records, eager_records = [], []
    for row, hrow, prefix in zip(rows, hrows, prefixes):
        if row.get("set", "native") not in ("native", "pad"):
            continue
        inputs, markers = host.graph_inputs(hrow, seq_len, prefix)
        t0 = time.perf_counter()
        with torch.no_grad():
            scores = scores_of(module(**as_tensors(inputs)))
            eager = scores_of(model(**as_tensors(inputs)))
        n = row["positions"]
        record = row_record(row, scores[markers])
        record["set"] = row.get("set", "native")
        record["max_abs_dscore_vs_eager"] = float(np.max(np.abs(scores[:n].astype(np.float64) - eager[:n])))
        record["bit_identical_vs_eager"] = bool(np.array_equal(scores[:n], eager[:n]))
        record["seconds"] = time.perf_counter() - t0
        records.append(record)
        eager_records.append(row_record(row, eager[markers]))
        print(f"  {len(records)} {row['id']}/{row['qid']} ({record['set']}, {n} pos) |dp| {record['max_abs_dp']:.2e} "
              f"vs eager {record['max_abs_dscore_vs_eager']:.1e} {record['seconds']:.1f} s", flush=True)
    summary = summarize(records)
    summary["vs_eager"] = {"max_abs_dscore_real_positions": max(r["max_abs_dscore_vs_eager"] for r in records),
                           "bit_identical_rows": sum(r["bit_identical_vs_eager"] for r in records)}
    eager_summary = summarize(eager_records)
    summary["eager_vs_reference"] = {k: eager_summary[k] for k in (
        "rows", "argmax_equal", "non_near_tie", "near_tie", "max_abs_dp", "mean_row_max_abs_dp", "max_abs_dlogit",
        "max_abs_dp_row", "fp32_bar", "ship_bar")}
    summary["by_set"] = {name: {k: s[k] for k in ("rows", "argmax_equal", "non_near_tie", "near_tie", "max_abs_dp",
                                                   "mean_row_max_abs_dp", "max_abs_dlogit")}
                         for name in ("native", "pad")
                         if (s := summarize([r for r in records if r["set"] == name]) if any(
                             r["set"] == name for r in records) else None)}
    return records, summary


def element_type(type_text: str) -> str:
    m = re.search(r"x([a-z]+[0-9]+)>$", type_text) or re.search(r"<([a-z]+[0-9]+)>$", type_text)
    return m.group(1) if m else "other"


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


def io_contract(seq_len: int) -> dict:
    L = seq_len
    return {
        "inputs": {
            "input_ids": {"dtype": "int32", "shape": [1, L], "values": "0 on [0, P), the row's ids on [P, P+n), 0 (<|pad|>) after"},
            "prefix_embeds": {"dtype": "float32", "shape": [1, L, 1024], "values": "the image / audio prefix on [0, P), 0.0 elsewhere (finite)"},
            "pad_mask": {"dtype": "float32", "shape": [1, L], "values": "1.0 on [0, P+n), 0.0 after"},
            "prefix_mask": {"dtype": "float32", "shape": [1, L], "values": "1.0 on [0, P), 0.0 after"},
            "keep_right": {"dtype": "float32", "shape": [1, L], "values": "0.0 at P-1 when P > 0, 1.0 elsewhere"},
            "qtype_onehot": {"dtype": "float32", "shape": [1, 3], "order": ["choice", "score", "noul"]}},
        "outputs": {"scores": {"dtype": "float32", "shape": [1, L], "read": "at P + markers (the <|mask|> positions)"}},
    }


def bundle_metadata(seq_len: int, name: str, config: dict) -> dict:
    return {
        "metadata_version": "0.2",
        "kind": "encoder",
        "decision": {
            "head": "encoder", "layout": "d1-omni", "seq_len": seq_len, "buckets": [seq_len],
            "functions": {"main": "main"}, **io_contract(seq_len),
            "token_ids": dict(host.TOKEN_IDS),
            "delimiters": {"state": "<|reserved_7|>", "q": "<|reserved_8|>", "opt": "<|reserved_9|>",
                           "opt_end": "<|reserved_10|>", "decide": "<|reserved_11|>", "marker": "<|mask|>"},
            "temperatures": config["temperatures"],
            "temperature_rule": ("text rows: marker logits / temperatures[f'{type}:{2|3-5|6-10|11+}' by option count] "
                                 "if present else temperatures[type]; image and audio rows: no temperature"),
            "noul_order": ["false", "true"], "noul_reported": ["yes", "no"],
            "max_length": config["max_length"], "image_text_length": config["image_text_length"],
            "audio_text_length": config["audio_text_length"], "min_text_positions": host.MIN_TEXT_POSITIONS,
            "prefix_hidden": 1024,
            "host_reference": "conversion/d1_omni/host.py (request_rows, graph_inputs, probabilities_from_logits)",
        },
        "source": {"hf_model_id": host.MODEL_ID, "hf_revision": host.MODEL_SHA, "license": LICENSE},
        "assets": {"main": name},
        "language": {"tokenizer": "tokenizer/", "vocab_size": config["text_config"]["vocab_size"],
                     "max_context_length": seq_len},
    }


def parameter_bytes(model: torch.nn.Module) -> dict:
    by_dtype, buffers_by_dtype = Counter(), Counter()
    for p in model.parameters():
        by_dtype[str(p.dtype).replace("torch.", "")] += p.numel() * p.element_size()
    for b in model.buffers():
        buffers_by_dtype[str(b.dtype).replace("torch.", "")] += b.numel() * b.element_size()
    return {"parameters_by_dtype": dict(by_dtype), "parameters": sum(by_dtype.values()),
            "buffers_by_dtype": dict(buffers_by_dtype), "buffers": sum(buffers_by_dtype.values())}


def int8_config(block: int = INT8_BLOCK) -> dict:
    """coreai-opt eager config: weight-only int8, symmetric, per block of `block` input channels, on every nn.Linear
    called as a module (quant mode) and on the head's in-projection; nothing by default (global_config None: the
    embedding, type_emb, the norms and the depthwise filter run in the root module's context); the scorer off by
    name. The spec is vision_quant_config's (conversion/export_lfm25vl_pipelined.py). A name config reaches the
    module's children too (the head's self_attn config decides its out_proj), so it names both of its weights."""
    def spec(*states: str) -> dict:
        weight = {"dtype": "int8", "qscheme": "symmetric_with_clipping",
                  "granularity": {"type": "per_block", "block_size": block, "axis": 1}}
        return {"op_state_spec": {s: weight for s in states}, "op_input_spec": None, "op_output_spec": None}
    return {"execution_mode": "eager", "global_config": None,
            "module_type_configs": {"torch.nn.modules.linear.Linear": spec("weight")},
            "module_name_configs": {r"head\.head\.layers\.\d+\.self_attn": spec("in_proj_weight", "weight"),
                                    r"head\.scorer\.\d+": None}}


def quantize_int8(model: dm.D1Decide, example: dict) -> tuple[dm.D1Decide, dict]:
    """The fp16 frame in quant mode -> coreai-opt eager int8 (prepare, finalize for Core AI) -> the finalized module,
    and what was quantized (every trunk / head linear: 92 + 8 weights, asserted)."""
    import copy

    import torch.nn.utils.parametrize as P
    from coreai_models.export.compression import quantize_pytorch_model

    model.set_quant_mode()
    config = int8_config()
    t0 = time.perf_counter()
    quantized = quantize_pytorch_model(model, tuple(example[k] for k in INPUT_NAMES), None, copy.deepcopy(config))
    seconds = time.perf_counter() - t0
    weights, int8_elements, scale_bytes, scale_dtypes = [], 0, 0, Counter()
    for name, module in quantized.named_modules():
        if not P.is_parametrized(module):
            continue
        for param, chain in module.parametrizations.items():
            for p in chain:
                data, scale = getattr(p, "quantized_data", None), getattr(p, "scale", None)
                if data is None:
                    continue
                weights.append({"name": f"{name}.{param}", "shape": list(data.shape), "dtype": str(data.dtype),
                                "scale_shape": list(scale.shape), "scale_dtype": str(scale.dtype)})
                int8_elements += data.numel()
                scale_bytes += scale.numel() * scale.element_size()
                scale_dtypes[str(scale.dtype)] += 1
    expected = sorted(f"{n}.weight" for n, m in quantized.named_modules()
                      if isinstance(m, torch.nn.Linear) and not n.startswith("head.scorer")) + sorted(
        f"{n}.in_proj_weight" for n, m in quantized.named_modules() if isinstance(m, dm._HeadAttention))
    got = sorted(w["name"] for w in weights)
    if got != sorted(expected) or any(w["dtype"] != "torch.int8" for w in weights):
        raise SystemExit(f"int8: quantized {len(got)} weights, expected {len(expected)}: "
                         f"missing {sorted(set(expected) - set(got))[:5]}, extra {sorted(set(got) - set(expected))[:5]}")
    return quantized, {"config": config, "frame": "fp16 (set_precision('fp16') + set_quant_mode())",
                       "tool": "coreai_models.export.compression.quantize_pytorch_model (coreai-opt eager quantizer, "
                               "finalize for Core AI)",
                       "weights": len(weights), "int8_elements": int8_elements, "int8_bytes": int8_elements,
                       "scale_bytes": scale_bytes, "scale_dtypes": dict(scale_dtypes), "seconds": seconds,
                       "not_quantized": "embed_tokens, type_emb, every RMSNorm / LayerNorm, the depthwise filters, "
                                        "the scorer (LayerNorm, Linear 1024x1024, Linear 1024x1)",
                       "detail": weights}


def attention_form(op_counts: dict, seq_len: int) -> dict:
    """The converted graph's attention ops: 8 softmax ops at L <= KEY_BLOCK, none above it (the key blocks)."""
    blocked = seq_len > dm.KEY_BLOCK
    form = {"key_block": dm.KEY_BLOCK, "blocked": blocked, "blocks": -(-seq_len // dm.KEY_BLOCK) if blocked else 1,
            "attentions": ATTENTIONS, "softmax_ops": op_counts.get("coreai.softmax", 0),
            "reduce_and_exp_ops": {k: v for k, v in op_counts.items()
                                   if any(s in k for s in ("reduce", "exp", "max", "divide"))}}
    form["expected_softmax_ops"] = 0 if blocked else ATTENTIONS
    form["status"] = "PASS" if form["softmax_ops"] == form["expected_softmax_ops"] else "FAIL"
    return form


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--precision", choices=PRECISIONS, required=True)
    parser.add_argument("--seq-len", type=int, choices=SEQ_LENS, required=True)
    parser.add_argument("--pad-rows", type=int, default=None,
                        help=f"rows of the smaller buckets padded into this one (default {PAD_ROWS}; "
                             f"{PAD_ROWS_FP16_LONG} for fp16 at L >= 2048)")
    parser.add_argument("--measure-only", action="store_true",
                        help="fp32: convert even if the export gate misses its bar (the manifest says FAIL)")
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_grad_enabled(False)
    n_pad = args.pad_rows if args.pad_rows is not None else (
        PAD_ROWS_FP16_LONG if args.precision == "fp16" and args.seq_len >= 2048 else PAD_ROWS)
    out_dir = bundle_dir(args.precision, args.seq_len)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists: an existing bundle folder is never replaced")
    stage = out_dir.parent / f".staging-{out_dir.name}-{os.getpid()}"
    stage.mkdir(parents=True)
    name = bundle_name(args.precision, args.seq_len)
    print(f"pid {os.getpid()} staging {stage}", flush=True)

    snapshot = Path(hf_snapshot(host.MODEL_ID, revision=host.MODEL_SHA))
    source = verify_source(snapshot)
    config = json.loads((snapshot / "config.json").read_text())
    tok = host.RawTokenizer(snapshot / "tokenizer.json")
    host.check_token_ids(tok)
    rows, hrows, prefixes, ref_info = load_rows(args.seq_len, tok, n_pad)
    modes = Counter(r["mode"] for r in rows)
    sets = Counter(r["set"] for r in rows)
    print(f"rows {len(rows)} {dict(modes)} {dict(sets)}", flush=True)

    t0 = time.perf_counter()
    frame = "fp16" if args.precision == "int8" else args.precision
    model, load_record = dm.load_d1_decide(snapshot, args.seq_len, frame, verify_sha256=True)
    load_s = time.perf_counter() - t0
    example = next(i for i, r in enumerate(rows) if r["mode"] == "text" and r["set"] in ("native", "pad"))
    inputs, _ = host.graph_inputs(hrows[example], args.seq_len, None)
    quantization = None
    if args.precision == "int8":
        model, quantization = quantize_int8(model, as_tensors(inputs))
        print(f"int8: {quantization['weights']} weights, {quantization['int8_bytes']:,} B int8 + "
              f"{quantization['scale_bytes']:,} B scales in {quantization['seconds']:.1f} s", flush=True)
    weights = parameter_bytes(model)
    from coreai_torch import get_decomp_table

    t0 = time.perf_counter()
    exported = torch.export.export(model, args=(), kwargs=as_tensors(inputs))
    export_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    program = exported.run_decompositions(get_decomp_table())
    decompose_s = time.perf_counter() - t0
    print(f"torch.export {export_s:.1f} s, decompositions {decompose_s:.1f} s (example {rows[example]['id']}/"
          f"{rows[example]['qid']})", flush=True)

    t0 = time.perf_counter()
    records, summary = export_gate(program, model, rows, hrows, prefixes, args.seq_len)
    gate_s = time.perf_counter() - t0
    bar = FP32_BAR if args.precision == "fp32" else SHIP_BAR
    status = summary["fp32_bar" if args.precision == "fp32" else "ship_bar"]["status"]
    print(f"export gate {status} ({bar['name']}) in {gate_s:.0f} s: argmax {summary['argmax_equal']}/{summary['rows']} "
          f"(non-near-tie {summary['non_near_tie']['argmax_equal']}/{summary['non_near_tie']['rows']}), "
          f"max |dp| {summary['max_abs_dp']:.3e}, mean {summary['mean_row_max_abs_dp']:.3e}, "
          f"max |dlogit| {summary['max_abs_dlogit']:.3e}; vs eager {summary['vs_eager']['max_abs_dscore_real_positions']:.3e} "
          f"({summary['vs_eager']['bit_identical_rows']}/{len(records)} bit-identical); eager vs reference max |dp| "
          f"{summary['eager_vs_reference']['max_abs_dp']:.3e} mean {summary['eager_vs_reference']['mean_row_max_abs_dp']:.3e}",
          flush=True)
    gate_record = {"status": status, "stage": "export (torch-exported + decomposed program, CPU, before conversion)",
                   "precision": args.precision, "seq_len": args.seq_len, "bar": bar,
                   "bar_rule": ("fp32 stops before conversion if this misses" if args.precision == "fp32" else
                                "measured, converted whatever it says; the runtime gate decides"),
                   "reference": ref_info, "example_row": f"{rows[example]['id']}/{rows[example]['qid']}",
                   "summary": summary, "seconds": gate_s, "environment": environment(), "rows": records}
    write_json(stage / "provenance" / "export-gate.json", gate_record)
    if args.precision == "fp32" and status != "PASS" and not args.measure_only:
        raise SystemExit(f"fp32 export gate FAIL before conversion: {summary['fp32_bar']['verdict']} "
                         f"(record {stage / 'provenance' / 'export-gate.json'})")

    description = (f"d1-omni-600M decision graph (bidirectional LFM2 trunk + decision head), {args.precision}, "
                   f"one question row, L={args.seq_len}, function main; source {host.MODEL_ID}@{host.MODEL_SHA}")
    converted = convert(program, description, stage / name)
    attention = attention_form(converted["op_counts"], args.seq_len)
    print(f"converted {converted['seconds']} ops {converted['ops_total']}; attention {attention['status']}: softmax "
          f"{attention['softmax_ops']} (expected {attention['expected_softmax_ops']}), "
          f"{attention['reduce_and_exp_ops']}", flush=True)
    if attention["status"] != "PASS":
        raise SystemExit(f"the converted graph's attention is not the form L={args.seq_len} calls for: {attention} "
                         f"(staging folder left at {stage})")

    (stage / "tokenizer").mkdir()
    for file in TOKENIZER_FILES:
        shutil.copy2(snapshot / file, stage / "tokenizer" / file)
    metadata = bundle_metadata(args.seq_len, name, config)
    metadata["decision"]["precision"] = {
        "name": args.precision, "frame": frame,
        "weights": {"fp32": "fp32", "wfp16": "fp16", "fp16": "fp16",
                    "int8": "int8 per block of 32 input channels on the trunk / head linears, fp16 elsewhere"}[args.precision],
        "compute": "fp16 (RMSNorm, LayerNorm, softmax, scorer fp32)" if frame == "fp16" else "fp32"}
    metadata["decision"]["attention"] = {k: attention[k] for k in ("key_block", "blocked", "blocks")}
    write_json(stage / "metadata.json", metadata)
    reference = {"schema": "d1-omni-decide-reference/1", "model": host.MODEL_ID, "revision": host.MODEL_SHA,
                 "seq_len": args.seq_len, "reference": ref_info,
                 "oracle": "the publisher's D1OmniModel (transformers 5.19, trust_remote_code) in fp32 on the CPU, "
                           "probabilities() once per request; logits_raw = the head's marker logits before the "
                           "temperature; probs = the reported distribution (temperature for text rows, noul as "
                           "[yes, no])",
                 "prefix_npz_root": "$ZOO_WORK_ROOT/_d1_omni (the round 2 reference's npz)",
                 "rows": rows}
    write_json(stage / "reference.json", reference)

    files = file_inventory(stage / name)
    bundle_bytes = sum(f["bytes"] for f in files)
    record = {
        "status": status, "measure_only": bool(args.measure_only and status != "PASS"),
        "model": host.MODEL_ID, "model_sha": host.MODEL_SHA, "bundle": name, "format": "JIT .aimodel (function main)",
        "intended_runtime": "macos", "aot": False, "precision": args.precision, "seq_len": args.seq_len,
        **io_contract(args.seq_len), "files": files, "bytes": bundle_bytes,
        "precision_frame": frame, "quantization": quantization, "attention_form": attention,
        "module_weights": weights,
        "bytes_over_module_parameter_bytes": bundle_bytes / weights["parameters"],
        "bytes_over_module_state_bytes": bundle_bytes / (weights["parameters"] + weights["buffers"]),
        "op_counts": converted["op_counts"], "ops_total": converted["ops_total"],
        "result_element_types": converted["result_element_types"],
        "seconds": {"weights_load": load_s, "torch_export": export_s, "decompositions": decompose_s,
                    "export_gate": gate_s, **converted["seconds"], "total": time.perf_counter() - started},
        "export_gate": {"status": status, "record": "provenance/export-gate.json",
                        **{k: summary[k] for k in ("rows", "argmax_equal", "non_near_tie", "near_tie", "max_abs_dp",
                                                   "mean_row_max_abs_dp", "max_abs_dlogit", "vs_eager",
                                                   "eager_vs_reference", "by_set")}},
        "rows": {"total": len(rows), "by_mode": dict(modes), "by_set": dict(sets), "pad_rows_requested": n_pad},
        "weights_record": load_record,
        "source_sha256": source,
        "reference_json_sha256": sha256_file(stage / "reference.json"),
        "metadata_json_sha256": sha256_file(stage / "metadata.json"),
        "code_sha256": {f: sha256_file(HERE / f) for f in CODE_FILES},
        "runtime_gate": "NOT RUN — aot_decide.py, then runtime_check.py <compiled dir> --compute cpu_only|gpu|neural_engine",
        "environment": environment(),
    }
    write_json(stage / "provenance" / "export-manifest.json", record)
    os.rename(stage, out_dir)
    print(json.dumps({"status": status, "folder": str(out_dir), "bundle": name, "bytes": bundle_bytes,
                      "module_parameter_bytes": weights["parameters"],
                      "ratio": round(record["bytes_over_module_parameter_bytes"], 4), "ops": converted["ops_total"],
                      "seconds": {k: round(v, 1) for k, v in record["seconds"].items()}}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
