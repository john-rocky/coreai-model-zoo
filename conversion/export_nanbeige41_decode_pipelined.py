"""Export a decode-pipelined Nanbeige4.1/4.2-3B bundle for the Core AI engine.

Nanbeige4.1-3B is a textbook plain-Llama dense model (`model_type: "llama"`,
GQA 20q/4kv head_dim128, SwiGLU, RoPE theta 70M, NO QK-norm / NO bias), so it
rides the standard pure-attention KV-only decode path (no conv/recurrent state,
no loop-free-linear gymnastics — unlike the qwen3_5 hybrid sibling this is
trimmed from). The one non-trivial bit is the BIG UNTIED head (vocab 166144,
`tie_word_embeddings: false`): ~0.85 GB in fp16, so the ship modes quantize it
(`int8hu` / `int4hu`, `--head-sym` absmax — big-vocab heads are fat-tailed and
`symmetric_with_clipping` craters outlier rows; absmax gates clean).

Arch parity is proven on CPU (`_smoke/test_nanbeige_parity.py`: top-1 6/6,
cosine 1.0, Δ=0 vs the native HF `LlamaForCausalLM` oracle). After exporting,
the FIRST on-device check is the ANE-smoke: a
plain-Llama GQA is the most ANE-likely shape, but the stock static engine + the
untied big head must be confirmed to place on ANE, not GPU-fallback — that is
the whole reason this model (over a hybrid/MoE one) was picked.

Modes:  fp16        - baseline
        int8lin     - body int8 per-block-32, head fp16
        int8hu      - int8lin + head int8 (absmax w/ --head-sym)   [quality ship]
        int4lin     - body int4 per-block-32, head fp16
        int4hu      - int4 body + head int8 (absmax)               [iPhone ship ~2 GB]

Run:  cd ~/code/coreai/coreai-models && .venv/bin/python \
          ../coreai-models-community/conversion/export_nanbeige41_decode_pipelined.py \
          int4hu --head-sym
      # smoke first:  ... int8hu --head-sym --num-layers 4
GPU exclusivity: grab the community-repo _GPU_LOCK; this is GPU-gated behind any
running diffgemma / coder-next export.

Nanbeige4.2-3B (`model_type: "nanbeige"`) reuses the same physical Llama blocks
twice, with a norm after each pass and disjoint 22-layer cache ranges. Its pinned
release baseline is `int8hu --head-sym --static-ids`; int4 remains conditional on
the identical quality gates. The 4.1 defaults above are intentionally unchanged.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from _bundle import head_quant_spec, write_bundle_metadata

from coreai_models.export._constants import (
    KEY_CACHE_NAME,
    QUANT_TRACE_OFFSET,
    QUANT_TRACE_QUERY_LEN,
    TRACE_KV_CACHE_SEQ_LEN,
    VALUE_CACHE_NAME,
)
from coreai_models.export.macos import export_to_coreai
from coreai_models.models.macos.llama import LlamaForCausalLM
from coreai_models.models.macos.nanbeige import (
    NanbeigeForCausalLM,
    create_cache_tensors as create_nanbeige_cache_tensors,
)
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8") -> dict:
    """Weight-only linear int8/int4 per-block-32 (scale-multiply dequant, no LUT).
    Norms/embedding/SDPA/RoPE excluded; lm_head excluded by name (stays fp16 unless
    a *hu mode adds an explicit head spec below)."""
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
            "torch.nn.modules.sparse.Embedding": None,
        },
        "module_name_configs": {r".*lm_head$": None},
    }


def build_kv_reference(cfg, max_ctx: int, static_ids: bool = False):
    """KV-only reference inputs + dynamic shapes.

    static_ids=False: dynamic input_ids (mirrors export.macos._build_reference_inputs) — allows
      multi-token prefill in one call. FAST on Mac, but on the iPhone pipelined engine (chunkThreshold=1,
      every step S=1) it pays a per-step input_ids RE-specialization that is pathological on a big model
      (~37 s/step cold on the 4.3 GB int8hu — the 900 s device probe never finished the first 24-tok run).
    static_ids=True: input_ids fixed [1,1] (the qwen3_5 loop-free device pattern). chunkThreshold=1 feeds
      S=1 anyway, so no prefill loss; eliminates the input_ids respec. position_ids + KV stay dynamic.
      Ship `_s1` for the device."""
    if static_ids:
        input_ids = torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32)
        position_ids = torch.arange(65, dtype=torch.int32).unsqueeze(0)  # trace_past 64 + 1
        ids_dyn = None
        pos_dyn = {1: torch.export.Dim("seq_pos", min=2, max=max_ctx - 1)}
    else:
        input_ids = torch.randint(1, cfg.vocab_size, (1, QUANT_TRACE_QUERY_LEN), dtype=torch.int32)
        position_ids = torch.arange(
            QUANT_TRACE_QUERY_LEN + QUANT_TRACE_OFFSET, dtype=torch.int32).unsqueeze(0)
        ids_dyn = {1: torch.export.Dim("seq_ids", max=max_ctx - 2)}
        pos_dyn = {1: torch.export.Dim("seq_pos", min=QUANT_TRACE_QUERY_LEN, max=max_ctx - 1)}

    saved = cfg.max_position_embeddings
    cfg.max_position_embeddings = TRACE_KV_CACHE_SEQ_LEN
    if cfg.model_type == "nanbeige":
        k_cache, v_cache = create_nanbeige_cache_tensors(cfg, dtype=DTYPE)
    else:
        k_cache, v_cache = KVCache.create_cache_tensors(cfg, dtype=DTYPE)
    cfg.max_position_embeddings = saved

    reference_inputs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "k_cache": k_cache,
        "v_cache": v_cache,
    }
    dynamic_shapes = {
        "input_ids": ids_dyn,
        "position_ids": pos_dyn,
        "k_cache": {KVCache.seq_len_dim(): torch.export.Dim("k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=max_ctx)},
        "v_cache": {KVCache.seq_len_dim(): torch.export.Dim("v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=max_ctx)},
    }
    return reference_inputs, dynamic_shapes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="int4hu",
                    choices=["fp16", "int8lin", "int8hu", "int4lin", "int4hu"])
    ap.add_argument("--hf-id", default="Nanbeige/Nanbeige4.1-3B")
    ap.add_argument("--revision", help="immutable Hugging Face checkpoint revision")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--head-quant", default="block32",
                    choices=["block32", "block16", "block8", "perchan"])
    ap.add_argument("--head-sym", action="store_true",
                    help="*hu modes: plain symmetric (absmax, no clipping) for the 166144-vocab head")
    ap.add_argument("--num-layers", type=int, default=None, help="debug: truncated-layer export")
    ap.add_argument("--static-ids", action="store_true",
                    help="fix input_ids at [1,1] (qwen3_5 device pattern) — REQUIRED for fast iPhone "
                         "decode (dynamic-ids respecializes per step, ~37 s/step cold on the 4.3 GB bundle)")
    args = ap.parse_args()

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_decode_{args.mode}"
    if args.mode in ("int8hu", "int4hu") and (args.head_quant != "block32" or args.head_sym):
        name += f"_{args.head_quant}" + ("_sym" if args.head_sym else "")
    if args.static_ids:
        name += "_s1"
    if args.num_layers is not None:
        name += f"_l{args.num_layers}"

    from transformers import AutoConfig

    source_config = AutoConfig.from_pretrained(args.hf_id, revision=args.revision)
    model_classes = {"llama": LlamaForCausalLM, "nanbeige": NanbeigeForCausalLM}
    try:
        model_class = model_classes[source_config.model_type]
    except KeyError:
        raise ValueError(
            f"unsupported model_type {source_config.model_type!r}; expected llama or nanbeige"
        ) from None

    print(f"loading {args.hf_id} fp16 (memory-efficient) ...", flush=True)
    model = model_class.from_hf_memory_efficient(
        args.hf_id,
        revision=args.revision,
        max_context_length=args.max_ctx,
        target_dtype=DTYPE,
        num_layers=args.num_layers,
    )
    model.eval()
    cfg = model.config
    print(f"hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
          f"q/kv={cfg.num_attention_heads}/{cfg.num_key_value_heads} vocab={cfg.vocab_size} "
          f"tied={cfg.tie_word_embeddings}", flush=True)

    reference_inputs, dynamic_shapes = build_kv_reference(cfg, args.max_ctx, static_ids=args.static_ids)

    if args.mode != "fp16":
        from coreai_models.export.compression import quantize_pytorch_model

        base = "int4" if "int4" in args.mode else "int8"
        cfg_q = linear_quant_config(base)
        if args.mode in ("int8hu", "int4hu"):
            cfg_q["module_name_configs"][r".*lm_head$"] = head_quant_spec(args.head_quant, args.head_sym)
            # the eager quantizer skips shared params; clone so an (untied) head is quantized in-place
            model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())
        print(f"quantizing (linear {base} per-block-32, mode={args.mode}) ...", flush=True)
        model = quantize_pytorch_model(
            model, tuple(reference_inputs.values()), dynamic_shapes, cfg_q)

    print("exporting decode graph to Core AI dialect ...", flush=True)
    prog = export_to_coreai(
        model,
        reference_inputs,
        dynamic_shapes=dynamic_shapes,
        input_names=("input_ids", "position_ids"),
        output_names=("logits",),
        state_names=(KEY_CACHE_NAME, VALUE_CACHE_NAME),
    )
    print("optimizing ...", flush=True)
    prog.optimize()

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    import coreai.runtime as rt

    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())

    write_bundle_metadata(out_dir, name, args.hf_id, cfg.vocab_size, args.max_ctx,
                          revision=args.revision, mode=args.mode)
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(args.hf_id, revision=args.revision).save_pretrained(
        out_dir / "tokenizer"
    )
    print(f"bundle ready: {out_dir}", flush=True)
    print(f"run: COREAI_CHUNK_THRESHOLD=1 llm-benchmark --model {out_dir} -p 128 -g 256 -n 3", flush=True)


if __name__ == "__main__":
    main()
