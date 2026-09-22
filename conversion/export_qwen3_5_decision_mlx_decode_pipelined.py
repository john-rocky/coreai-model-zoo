#!/usr/bin/env python3
"""Export the pinned Jev-Style Qwen3.5-2B Decision MLX model to Core AI.

Download chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16 at the pinned
revision, verify its checkpoint and tokenizer SHA256, then invert the MLX
layout with mlx_to_hf_qwen3_5.py. The resulting local snapshot is passed to the
unchanged frozen Qwen3.5 loader through a private offline Hub cache. No source
checkpoint, installed loader, or shared cache entry is patched.

The graph, loop-free S=1 path, four recurrent/KV states, and quantization are
the shipped export_qwen3_5_decode_pipelined.py recipe. int8hu clones the tied
embedding table into lm_head before block-32 quantization; --head-sym uses
plain symmetric absmax for that head. fp16 is the reference. The output is
[1,1,248320] logits. Gather the space-prefixed letter IDs for the listed options
and apply softmax at temperature 1; the checkpoint already folds calibration
into its final norm. Prompts end at Answer: with no BOS and no chat template.

Requires the frozen coreai-models Qwen3.5 overlay and the Swift extra-states
patch. Run engines with COREAI_CHUNK_THRESHOLD=1. _bundle.py is the standard
metadata helper; source.hf_revision and the decision block document readout.

Run from the pinned export environment (coreai-torch 0.4.1, transformers 4.57.6):
  HF_HUB_DISABLE_XET=1 python export_qwen3_5_decision_mlx_decode_pipelined.py \
      int8hu --head-sym --out-dir exports
  HF_HUB_DISABLE_XET=1 python export_qwen3_5_decision_mlx_decode_pipelined.py \
      fp16 --out-dir exports

Use --name for an exact bundle directory override; --max-ctx defaults to 4096.
No text generation or probability readout is baked into the exported graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from _bundle import head_quant_spec, write_bundle_metadata
from mlx_to_hf_qwen3_5 import HF_ID, REVISION, convert_snapshot, sha256, write_json

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.qwen3_5 import (
    DECODE_STATE_NAMES,
    Qwen3_5StatefulForCausalLM,
    build_decode_state,
)
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16


def palettization_config(n_bits: int = 8, group: int = 32) -> dict:
    """int8 k-means recipe (EXACT top-1, but 256-entry LUT is slow on the GPU)."""
    spec = {
        "n_bits": n_bits,
        "granularity": {"type": "per_grouped_channel", "axis": 0, "group_size": group},
        "enable_per_channel_scale": False,
    }
    return {
        "global_config": {"op_state_spec": {"weight": spec}},
        "module_name_configs": {r".*lm_head$": None, r".*conv1d$": None},
    }


def linear_quant_config(dtype: str = "int8") -> dict:
    """Weight-only linear int8 per-block-32 - scale-multiply dequant, no LUT.
    Embedding/conv/norms excluded; lm_head excluded by name (tied table stays
    fp16; use mode int8hu to quantize an untied head)."""
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {
                "weight": {
                    "dtype": dtype,
                    "qscheme": "symmetric_with_clipping",
                    "granularity": {"type": "per_block", "block_size": 32, "axis": 1},
                }
            },
            "op_input_spec": None,
            "op_output_spec": None,
        },
        "module_type_configs": {
            "coreai_models.primitives.macos.sdpa.SDPA": None,
            "coreai_models.primitives.macos.rope.RoPE": None,
            "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
            "coreai_models.primitives.macos.rms_norm.RMSNormPlusOne": None,
            "coreai_models.primitives.macos.rms_norm.RMSNormGated": None,
            "torch.nn.modules.sparse.Embedding": None,
            "torch.nn.modules.conv.Conv1d": None,
        },
        "module_name_configs": {r".*lm_head$": None},
    }



def prepare_converted_snapshot(args: argparse.Namespace) -> None:
    """Re-exec against a private offline Hub view; never patch the frozen loader.

    Hugging Face resolves environment constants when imported. Therefore the
    conversion/download coordinator execs a fresh interpreter with HF_HUB_CACHE
    and HF_HUB_OFFLINE already set. Only this view is changed; the published
    MLX snapshot stays intact. Its original HF ID remains in bundle metadata.
    """
    if args.hf_id != HF_ID or args.revision != REVISION:
        raise ValueError("This checked inverse conversion requires its pinned HF ID and revision")
    if args._converted_cache is not None:
        report = json.loads(Path(args._conversion_report).read_text())
        assert report["status"] == "PASS"
        assert report["source_hf_id"] == HF_ID and report["source_revision"] == REVISION
        assert sha256(report["output"]) == report["output_sha256"]
        assert sha256(report["config"]) == report["config_sha256"]
        assert Path(os.environ["HF_HUB_CACHE"]).resolve() == Path(args._converted_cache).resolve()
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        return

    from huggingface_hub import snapshot_download

    work = Path(args.work_dir or Path(args.out_dir) / ".qwen3_5_decision_source").resolve()
    work.mkdir(parents=True, exist_ok=True)
    snapshot = Path(snapshot_download(HF_ID, revision=REVISION, max_workers=2))
    assert snapshot.name == REVISION, (snapshot, REVISION)
    print(f"verified pinned snapshot path: {snapshot}", flush=True)
    report = convert_snapshot(snapshot, work / "converted")
    fingerprint = hashlib.sha256((report["output_sha256"] + report["config_sha256"]).encode()).hexdigest()[:40]
    cache = work / "hub"
    repo = cache / ("models--" + HF_ID.replace("/", "--"))
    local_snapshot = repo / "snapshots" / fingerprint
    local_snapshot.mkdir(parents=True, exist_ok=True)
    for name in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        destination = local_snapshot / name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(work / "converted" / name)
    (repo / "refs").mkdir(exist_ok=True)
    (repo / "refs" / "main").write_text(fingerprint)
    write_json(work / "converted-cache.json", {
        "status": "PASS", "hf_id": HF_ID, "upstream_revision": REVISION,
        "source_snapshot": str(snapshot), "hf_hub_cache": str(cache),
        "local_snapshot_id": fingerprint, "local_snapshot_is_upstream_revision": False,
        "snapshot": str(local_snapshot), "backing_directory": str(work / "converted"),
        "source_cache_modified": False, "frozen_loader_modified": False,
        "required_environment": {"HF_HUB_CACHE": str(cache), "HF_HUB_OFFLINE": "1"},
    })
    environment = dict(os.environ, HF_HUB_CACHE=str(cache), HF_HUB_OFFLINE="1", HF_HUB_DISABLE_XET="1")
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
               "--_converted-cache", str(cache), "--_conversion-report", str(work / "converted" / "mlx_to_hf.json")]
    print(f"converted 320 tensors: 18 convolution transposes, 61 norm offsets; reloading in private offline cache", flush=True)
    os.execvpe(sys.executable, command, environment)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="int8hu", choices=["fp16", "int8hu"])
    ap.add_argument("--hf-id", default=HF_ID)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--name", default=None, help="Exact bundle directory name")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--work-dir", default=None, help="Converted snapshot and private cache; defaults beneath --out-dir")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--head-quant", default="block32", choices=["block32", "block16", "block8", "perchan"])
    ap.add_argument("--head-sym", action="store_true", help="int8hu: plain symmetric absmax head; ship configuration")
    ap.add_argument("--_converted-cache", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_conversion-report", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    # The numerical graph and quantization below are the shipped Qwen3.5 recipe.
    args.num_layers = None
    args.state_dict_prefix = "model.language_model."
    name = f"qwen3_5_2b_decision_decode_{args.mode}"
    if args.mode == "int8hu" and (args.head_quant != "block32" or args.head_sym):
        name += f"_{args.head_quant}" + ("_sym" if args.head_sym else "")
    name = args.name or name
    prepare_converted_snapshot(args)

    print(f"loading {args.hf_id} fp16 ...")
    try:
        model = Qwen3_5StatefulForCausalLM.from_hf_memory_efficient(
            args.hf_id, max_context_length=args.max_ctx, target_dtype=DTYPE,
            hf_config_attr="text_config", num_layers=args.num_layers,
            hf_state_dict_prefix=args.state_dict_prefix)
    except AttributeError:
        model = Qwen3_5StatefulForCausalLM.from_hf_memory_efficient(
            args.hf_id, max_context_length=args.max_ctx, target_dtype=DTYPE,
            hf_config_attr=None, num_layers=args.num_layers,
            hf_state_dict_prefix=args.state_dict_prefix)
    model.eval()
    cfg = model.config

    n_lin = 0
    for layer in model.model.layers:
        if not layer.is_full:
            layer.linear_attn.use_loopfree_step = True
            n_lin += 1
    print(f"loop-free single-step enabled on {n_lin} linear layers")

    # Decode trace: S=1 static query, dynamic full-length positions, dynamic KV seq.
    trace_past = 64
    input_ids = torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32)
    position_ids = torch.arange(trace_past + 1, dtype=torch.int32).unsqueeze(0)
    state = build_decode_state(cfg, max_seq_len=TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)

    reference_inputs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "k_cache": state["k_cache"],
        "v_cache": state["v_cache"],
        "conv_state": state["conv_state"],
        "rec_state": state["rec_state"],
    }
    seq_pos = torch.export.Dim("seq_pos", min=2, max=args.max_ctx - 1)
    k_seq = torch.export.Dim("k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    v_seq = torch.export.Dim("v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    dynamic_shapes = {
        "input_ids": None,  # static [1, 1] - no scan, no while_loop
        "position_ids": {1: seq_pos},
        "k_cache": {KVCache.seq_len_dim(): k_seq},
        "v_cache": {KVCache.seq_len_dim(): v_seq},
        "conv_state": None,
        "rec_state": None,
    }

    if args.mode == "int8":
        from coreai_models.export.compression import palettize_pytorch_model

        print("palettizing (int8 k-means group-32, lm_head/conv1d excluded) ...")
        model = palettize_pytorch_model(
            model, tuple(reference_inputs.values()), palettization_config())
    elif args.mode in ("int8lin", "int8hu", "int4lin"):
        from coreai_models.export.compression import quantize_pytorch_model

        cfg_q = linear_quant_config("int4" if args.mode == "int4lin" else "int8")
        if args.mode == "int8hu":
            # Provenance for the published bundles: the ones named `*_perchan_sym` contain
            # per-block-32 heads. An earlier version of this script parsed --head-quant
            # without applying it, so both arms of that A/B were block32 — byte-identical
            # sizes confirm it, and every published number stands.
            # knowledge/pipelined-engine.md records the bisect.
            cfg_q["module_name_configs"] = {
                r".*lm_head$": head_quant_spec(args.head_quant, args.head_sym)}
            model.lm_head.weight = torch.nn.Parameter(
                model.lm_head.weight.detach().clone())
        print(f"quantizing (linear int8 per-block-32, mode={args.mode}) ...")
        model = quantize_pytorch_model(
            model, tuple(reference_inputs.values()), dynamic_shapes, cfg_q)

    # The loop-free path never calls the GatedDeltaUpdate composite; externalizing
    # the class would mark the (uncalled) submodules and then fail to find them in
    # the traced program. RMSNorm/SDPA stay fused composites for the GPU delegate.
    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    print("exporting decode-only graph to Core AI dialect ...")
    prog = export_to_coreai(
        model,
        reference_inputs,
        dynamic_shapes=dynamic_shapes,
        input_names=("input_ids", "position_ids"),
        output_names=("logits",),
        state_names=DECODE_STATE_NAMES,
        externalize_modules=specs,
    )
    print("optimizing ...")
    prog.optimize()

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    import coreai.runtime as rt

    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...")
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())

    write_bundle_metadata(
        out_dir, name, args.hf_id, cfg.vocab_size, args.max_ctx,
        revision=args.revision,
        extra={"decision": {"head": "lm", "readout": "decision-function letters",
                            "temperature": 1, "labels": "space-prefixed A-Z"}},
    )
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(args.hf_id).save_pretrained(out_dir / "tokenizer")
    print(f"bundle ready: {out_dir}")
    print(f"run: COREAI_CHUNK_THRESHOLD=1 llm-benchmark --model {out_dir} -p 128 -g 256 -n 3")


if __name__ == "__main__":
    main()
