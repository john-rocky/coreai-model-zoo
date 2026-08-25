"""Export a decode-pipelined S1-mini bundle for the Core AI engine.

S1-mini by Superwhisper is a Qwen3-0.6B finetune that does ONE job: rewrite a raw
ASR transcript as clean written text (fillers dropped, false starts resolved,
punctuation/casing applied, spoken numbers/dates/currency rendered in written
form). It is not a chat model; it is steered by a control line at the top of the
user turn. That makes it the natural post-processor to sit behind the zoo's ASR
models (Parakeet / Qwen3-ASR / Nemotron) on the same device.

Arch: plain dense `qwen3` (model_type `qwen3`, byte-identical shape to
Qwen/Qwen3-0.6B) — 28 layers, hidden 1024, GQA 16q/8kv head_dim 128, SwiGLU
3072, QK-norm, RoPE theta 1e6, NO bias, `tie_word_embeddings: true`. It rides
the stock `coreai_models.models.macos.qwen3` definition unchanged (fused qkv +
fused qk_norm are handled by that class's `_mutate_state_dict`), on the standard
pure-attention KV-only decode path — no conv/recurrent state.

⚠️ NO `*hu` (head-quant) mode here, unlike the untied-head ports. The head is
TIED to the 151936x1024 embedding, and the eager quantizer skips shared params —
so quantizing the head means untying it first, which ADDS a tensor instead of
shrinking one:
    tied fp16 embed+head            = 311 MB   <- what this script ships
    fp16 embed + int8 untied head   = 311 + 156 = 467 MB
Untying is a pure loss at every bit width. The embedding itself stays fp16 by
the standard recipe (`torch.nn.modules.sparse.Embedding: None`), so on int4 it
is ~59% of the bundle; shrinking it is an embedding-quantization question, not a
head one, and is deliberately out of scope here.

Modes:  fp16     - baseline / control              (~1.2 GB)
        int8lin  - body int8 per-block-32          (~0.76 GB)  [quality ship]
        int4lin  - body int4 per-block-32          (~0.53 GB)  [phone ship, gate first]

Run:  cd ~/code/coreai/coreai-models && .venv/bin/python \
          ../coreai-models-community/conversion/export_s1_mini_decode_pipelined.py \
          int8lin
      # smoke first:  ... int8lin --num-layers 4

Gate:  python3 ../coreai-models-community/conversion/coreai_gate.py \
           exports/s1_mini_decode_int8lin superwhisper/s1-mini --arch qwen3 -n 16

License note: Apache 2.0 plus an ADDITIONAL TERM — any distribution or product
integration must keep identifying the model as "S1-mini" by "Superwhisper" with
that exact capitalization. The bundle name and every card must say so.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from _bundle import write_bundle_metadata

from coreai_models.export._constants import (
    KEY_CACHE_NAME,
    QUANT_TRACE_OFFSET,
    QUANT_TRACE_QUERY_LEN,
    TRACE_KV_CACHE_SEQ_LEN,
    VALUE_CACHE_NAME,
)
from coreai_models.export.macos import export_to_coreai
from coreai_models.models.macos.qwen3 import Qwen3ForCausalLM
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8") -> dict:
    """Weight-only linear int8/int4 per-block-32 (scale-multiply dequant, no LUT).

    Norms/embedding/SDPA/RoPE excluded; `lm_head` excluded by name — here that
    exclusion is load-bearing rather than a default, because the head IS the
    embedding (see the module docstring)."""
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
    """KV-only reference inputs + dynamic shapes (verbatim from the dense-Llama port).

    static_ids=False: dynamic input_ids — multi-token prefill in one call, fast on Mac.
    static_ids=True:  input_ids fixed [1,1], the loop-free device pattern; kills the
      per-step input_ids respecialization that the iPhone pipelined engine
      (chunkThreshold=1) otherwise pays every step. Cheap here (0.6 B), but export
      both and measure rather than assuming."""
    if static_ids:
        input_ids = torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32)
        position_ids = torch.arange(65, dtype=torch.int32).unsqueeze(0)  # trace_past 64 + 1
        ids_dyn = None
        pos_dyn = {1: torch.export.Dim("seq_pos", min=2, max=max_ctx - 1)}
    else:
        input_ids = torch.randint(1, cfg.vocab_size, (1, QUANT_TRACE_QUERY_LEN), dtype=torch.int32)
        position_ids = torch.arange(
            QUANT_TRACE_QUERY_LEN + QUANT_TRACE_OFFSET, dtype=torch.int32
        ).unsqueeze(0)
        ids_dyn = {1: torch.export.Dim("seq_ids", max=max_ctx - 2)}
        pos_dyn = {1: torch.export.Dim("seq_pos", min=QUANT_TRACE_QUERY_LEN, max=max_ctx - 1)}

    saved = cfg.max_position_embeddings
    cfg.max_position_embeddings = TRACE_KV_CACHE_SEQ_LEN
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
        "k_cache": {
            KVCache.seq_len_dim(): torch.export.Dim(
                "k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=max_ctx
            )
        },
        "v_cache": {
            KVCache.seq_len_dim(): torch.export.Dim(
                "v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=max_ctx
            )
        },
    }
    return reference_inputs, dynamic_shapes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="int8lin",
                    choices=["fp16", "int8lin", "int4lin"])
    ap.add_argument("--hf-id", default="superwhisper/s1-mini")
    ap.add_argument("--revision", help="checkpoint revision, recorded in metadata "
                                       "(the overlay loader reads the default branch)")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=4096,
                    help="the card recommends <=1000-token inputs and chunking beyond that; "
                         "4096 leaves room for the control line + a long transcript + its rewrite")
    ap.add_argument("--num-layers", type=int, default=None, help="debug: truncated-layer export")
    ap.add_argument("--static-ids", action="store_true",
                    help="fix input_ids at [1,1] (loop-free device pattern)")
    args = ap.parse_args()

    name = f"s1_mini_decode_{args.mode}"
    if args.static_ids:
        name += "_s1"
    if args.num_layers is not None:
        name += f"_l{args.num_layers}"

    from transformers import AutoConfig

    source_config = AutoConfig.from_pretrained(args.hf_id, revision=args.revision)
    if source_config.model_type != "qwen3":
        raise ValueError(
            f"unsupported model_type {source_config.model_type!r}; this script is the plain "
            "dense qwen3 route"
        )

    print(f"loading {args.hf_id} fp16 (memory-efficient) ...", flush=True)
    model = Qwen3ForCausalLM.from_hf_memory_efficient(
        args.hf_id,
        max_context_length=args.max_ctx,
        target_dtype=DTYPE,
        num_layers=args.num_layers,
    )
    model.eval()
    cfg = model.config
    print(f"hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
          f"q/kv={cfg.num_attention_heads}/{cfg.num_key_value_heads} vocab={cfg.vocab_size} "
          f"tied={cfg.tie_word_embeddings}", flush=True)
    assert cfg.tie_word_embeddings, (
        "an untied S1-mini checkpoint would need a head-quant mode this script "
        "deliberately does not have"
    )

    reference_inputs, dynamic_shapes = build_kv_reference(
        cfg, args.max_ctx, static_ids=args.static_ids
    )

    if args.mode != "fp16":
        from coreai_models.export.compression import quantize_pytorch_model

        base = "int4" if "int4" in args.mode else "int8"
        print(f"quantizing (linear {base} per-block-32, mode={args.mode}) ...", flush=True)
        model = quantize_pytorch_model(
            model, tuple(reference_inputs.values()), dynamic_shapes, linear_quant_config(base)
        )

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
    print(f"run: COREAI_CHUNK_THRESHOLD=1 llm-benchmark --model {out_dir} -p 128 -g 256 -n 3",
          flush=True)


if __name__ == "__main__":
    main()
