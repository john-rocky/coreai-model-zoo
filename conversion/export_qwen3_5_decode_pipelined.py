"""Export a DECODE-ONLY (S=1, loop-free) Qwen3.5 bundle for the Core AI pipelined engine.

This is the fast path: one dynamic-KV bundle that rides Apple's `coreai-pipelined`
GPU engine (`CoreAILanguageModel` / `EngineFactory`) instead of a hand-rolled
per-token run loop. Measured on M4 Max (p=128 g=256, release `llm-benchmark`,
`COREAI_CHUNK_THRESHOLD=1`):

    int8lin (ship) 204 tok/s @ 1.0 GB   - 3.5x the zoo's custom-kernel CLI (58.5)
    fp16           175.8 tok/s @ 1.4 GB
    int8 k-means   113 tok/s  @ 1.0 GB  - 256-entry LUT gather is slow on the GPU
                                          delegate; per-block linear dequant wins

Why decode-only: the full dynamic graph's GatedDeltaUpdate while_loop does not
lower on the MPSGraph GPU delegate ('scf.while' region type mismatch, beta).
This export removes the loop entirely: every linear-attention layer takes the
loop-free single-step path (`use_loopfree_step=True`, numerically identical at
S=1) and `input_ids` is STATIC [1,1]. position_ids and the KV seq dim stay
dynamic, so the engine's growing KV cache works. Prefill runs as pipelined S=1
steps - set `COREAI_CHUNK_THRESHOLD=1` (prompt tok/s ~ decode tok/s).

Numerics gate (int8lin): 16/16 teacher-forced single-step top-1 vs the fp32 HF
oracle + HF-cache-seeded decode step, and token-for-token == fp16-GPU greedy.

Requires the qwen3_5 model overlay on `coreai-models` (see conversion/README.md)
plus the pipelined-engine extra-states patch (apps/coreai-pipelined-extra-states.patch)
on the Swift side to RUN the bundle: the engine carries the SSM conv/rec states
as fixed-shape extra states.

Run:  python export_qwen3_5_decode_pipelined.py [fp16|int8|int8lin|int8hu] \
          [--hf-id Qwen/Qwen3.5-0.8B] [--out-dir exports]

Modes: fp16 - baseline; int8 - k-means g32 palettization (slow on GPU, kept for
comparison); int8lin - per-block-32 linear int8, THE ship config; int8hu -
int8lin + untied int8 lm_head (clones the tied embed table first - the eager
quantizer silently skips shared parameters).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from _bundle import head_quant_spec, write_bundle_metadata

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="int8lin",
                    choices=["fp16", "int8", "int8lin", "int8hu", "int4lin"])
    ap.add_argument("--hf-id", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--name", help="override the generated bundle directory and asset name")
    ap.add_argument("--revision", default=None, help="source revision recorded as metadata source.hf_revision")
    ap.add_argument("--extra-metadata", type=Path, default=None, help="JSON object merged into the root bundle metadata")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--head-quant", default="block32",
                    choices=["block32", "block16", "block8", "perchan"],
                    help="int8hu only: lm_head weight granularity (ship=block32; perchan is BROKEN on the beta GPU delegate)")
    ap.add_argument("--head-sym", action="store_true",
                    help="int8hu only: plain symmetric (absmax, no clipping) for the head")
    ap.add_argument("--num-layers", type=int, default=None,
                    help="debug: truncated-layer export (engine-contract de-risk)")
    args = ap.parse_args()
    extra_metadata = json.loads(args.extra_metadata.read_text()) if args.extra_metadata is not None else None
    if extra_metadata is not None and not isinstance(extra_metadata, dict):
        ap.error("--extra-metadata must contain a JSON object")

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_decode_{args.mode}"
    if args.mode == "int8hu" and (args.head_quant != "block32" or args.head_sym):
        name += f"_{args.head_quant}" + ("_sym" if args.head_sym else "")
    if args.num_layers is not None:
        name += f"_l{args.num_layers}"

    if args.name:
        name = args.name

    print(f"loading {args.hf_id} fp16 ...")
    try:
        model = Qwen3_5StatefulForCausalLM.from_hf_memory_efficient(
            args.hf_id, max_context_length=args.max_ctx, target_dtype=DTYPE,
            hf_config_attr="text_config", num_layers=args.num_layers)
    except AttributeError:
        model = Qwen3_5StatefulForCausalLM.from_hf_memory_efficient(
            args.hf_id, max_context_length=args.max_ctx, target_dtype=DTYPE,
            hf_config_attr=None, num_layers=args.num_layers)
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
        revision=args.revision, extra=extra_metadata,
    )
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(args.hf_id).save_pretrained(out_dir / "tokenizer")
    print(f"bundle ready: {out_dir}")
    print(f"run: COREAI_CHUNK_THRESHOLD=1 llm-benchmark --model {out_dir} -p 128 -g 256 -n 3")


if __name__ == "__main__":
    main()
