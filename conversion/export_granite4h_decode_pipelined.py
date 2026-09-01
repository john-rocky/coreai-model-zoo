"""Export a DECODE-ONLY (S=1) Granite 4.0-H bundle for the Core AI pipelined engine.

Granite 4.0-H (IBM) is a Mamba2 + attention hybrid — the first SSM-scan
architecture on the pipelined path. The dense "H" decoders interleave a few
GQA NoPE attention layers into a Mamba2 stack (1b: 36 mamba + 4 attention;
350m: 28 + 4). At S=1 the Mamba2 selective scan collapses to a single
recurrence step (state = state*dA + dt*B*x), so the decode graph is loop-free
the same way qwen3.5's GDN step is — no while_loop, lowers on the MPSGraph
GPU delegate. State = growing KV (4 attn layers) + TWO fixed-shape extra
states (stacked conv columns + stacked SSM states), exactly the
coreai-pipelined-extra-states.patch budget (<=2).

input_ids is STATIC [1,1]; position_ids and the KV seq dim stay dynamic, so
`EngineFactory` classifies the bundle as dynamic -> pipelined engine. Prefill
runs as pipelined S=1 steps: set `COREAI_CHUNK_THRESHOLD=1` (prompt tok/s ~
decode tok/s).

Numerics gate: 16/16 teacher-forced single-step top-1 vs the fp32 HF oracle +
HF-cache-seeded decode step (run _smoke/test_granite_decode_oracle_gate.py),
on an oracle whose top-2 margin is >= 0.1 at every position.

Requires the granite4h model overlay on `coreai-models` (see
conversion/README.md) plus the pipelined-engine extra-states patch on the
Swift side to RUN the bundle.

Run:  python export_granite4h_decode_pipelined.py \
          [fp16|int8lin|int8hu|int4lin|int4hu|int4lin8] \
          [--hf-id ibm-granite/granite-4.0-h-1b] [--out-dir exports]

Modes: fp16 - baseline; int8lin - per-block-32 linear int8 (the qwen3.5 ship
recipe: scale-multiply dequant, no LUT); int8hu - int8lin + untied int8
lm_head (clones the tied embed table first — the eager quantizer silently
skips shared parameters; use `--head-quant block32 --head-sym` = the qwen3.5
ship shape, absmax per-block-32: big-vocab heads are fat-tailed and the default
clipping qscheme crushes outlier rows). int4lin / int4hu - the same two shapes
at 4 bits (`--int4-block` sets the block, default 32); int4lin8 - int4
everywhere except a rescue set kept at int8 per-block-32 (default: the Mamba
mixer in/out projections, the int4-sensitive class on LFM2/qwen3.5;
`--rescue-regex` widens it for bisects). Embedding/conv1d/norms stay fp16;
lm_head stays fp16 except in the *hu modes. Ship guidance: 1b -> int8lin
(gate 16/16, ~32% faster); 350m -> fp16 (int8 is numerically marginal there
AND no faster — the model is overhead-bound, not bandwidth-bound, at that
size).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from _bundle import head_quant_spec, write_bundle_metadata

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.granite4h import (
    DECODE_STATE_NAMES,
    Granite4HForCausalLMStateful,
    build_decode_state,
)
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8", rescue_int8_regex: str | None = None,
                        block: int = 32) -> dict:
    """Weight-only linear per-block-N — scale-multiply dequant, no LUT.
    Embedding/conv/norms (incl. the gated Mamba output norm) excluded by type;
    the tied lm_head excluded by name. Unlike LFM2.5 the attention projections
    quantize cleanly here (NoPE, no q/k norm gains to amplify fp16 noise).
    ``rescue_int8_regex`` keeps the matching modules at int8 per-block-32 while
    the rest take ``dtype`` (the int4 rescue lever: same linear scheme, 8-bit)."""
    def spec(d: str, b: int = 32) -> dict:
        return {
            "op_state_spec": {
                "weight": {
                    "dtype": d,
                    "qscheme": "symmetric_with_clipping",
                    "granularity": {"type": "per_block", "block_size": b, "axis": 1},
                }
            },
            "op_input_spec": None,
            "op_output_spec": None,
        }

    name_configs: dict = {r".*lm_head$": None}
    if rescue_int8_regex:
        name_configs[rescue_int8_regex] = spec("int8")
    return {
        "execution_mode": "eager",
        "global_config": spec(dtype, block),
        "module_type_configs": {
            "coreai_models.primitives.macos.sdpa.SDPA": None,
            "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
            "coreai_models.models.macos.granite4h.GraniteGatedRMSNorm": None,
            "torch.nn.modules.sparse.Embedding": None,
            "torch.nn.modules.conv.Conv1d": None,
        },
        "module_name_configs": name_configs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="int8lin",
                    choices=["fp16", "int8lin", "int8hu",
                             "int4lin", "int4hu", "int4lin8"])
    ap.add_argument("--hf-id", default="ibm-granite/granite-4.0-h-1b")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--head-quant", default="block32",
                    choices=["block32", "block16", "block8", "perchan"],
                    help="int8hu only: lm_head weight granularity (ship=block32; perchan is BROKEN on the beta GPU delegate)")
    ap.add_argument("--head-sym", action="store_true",
                    help="int8hu only: plain symmetric (absmax, no clipping) for the head")
    ap.add_argument("--int4-block", type=int, default=32,
                    help="int4* only: linear weight block size (default 32)")
    ap.add_argument("--rescue-regex", default=None,
                    help="int4lin8 only: modules kept at int8 per-block-32 "
                         "(default: the Mamba mixer in/out projections)")
    args = ap.parse_args()

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_decode_{args.mode}"
    if args.mode in ("int8hu", "int4hu") and (args.head_quant != "block32" or args.head_sym):
        name += f"_{args.head_quant}" + ("_sym" if args.head_sym else "")
    if args.mode.startswith("int4") and args.int4_block != 32:
        name += f"_b{args.int4_block}"

    print(f"loading {args.hf_id} fp16 ...")
    model = Granite4HForCausalLMStateful.from_hf(args.hf_id, target_dtype=DTYPE)
    model.eval()
    cfg = model.config
    print(f"{cfg.num_hidden_layers} layers ({cfg.num_mamba_layers} mamba / "
          f"{cfg.num_attn_layers} attention), hidden={cfg.hidden_size}, "
          f"vocab={cfg.vocab_size}")

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
        "input_ids": None,  # static [1, 1] — single recurrence step, no scan
        "position_ids": {1: seq_pos},
        "k_cache": {KVCache.seq_len_dim(): k_seq},
        "v_cache": {KVCache.seq_len_dim(): v_seq},
        "conv_state": None,  # fixed-shape extra states
        "rec_state": None,
    }

    if args.mode != "fp16":
        from coreai_models.export.compression import quantize_pytorch_model

        dtype = "int8" if args.mode in ("int8lin", "int8hu") else "int4"
        # int4lin8 = int4 everywhere EXCEPT the rescue set, which stays int8.
        # Default rescue = the Mamba mixer in/out projections: on LFM2.5 and
        # qwen3.5 the mixer-projection class is the int4-sensitive path.
        rescue = None
        if args.mode == "int4lin8":
            rescue = args.rescue_regex or r".*mamba\.(in_proj|out_proj)$"
        block = args.int4_block if dtype == "int4" else 32
        cfg_q = linear_quant_config(dtype, rescue_int8_regex=rescue, block=block)
        if args.mode in ("int8hu", "int4hu"):
            # Untie the head (the eager quantizer skips shared params) and
            # quantize ONLY it on top of int8lin — the rest of the exclusion
            # list stays untouched.
            cfg_q["module_name_configs"][r".*lm_head$"] = head_quant_spec(
                args.head_quant, args.head_sym)
            model.lm_head.weight = torch.nn.Parameter(
                model.lm_head.weight.detach().clone())
        print(f"quantizing (linear {dtype} per-block-{block}, mode={args.mode}"
              f"{', mamba mixer rescued to int8' if rescue else ''}) ...")
        model = quantize_pytorch_model(
            model, tuple(reference_inputs.values()), dynamic_shapes, cfg_q)

    # No GatedDeltaUpdate modules in Granite 4.0-H — drop its externalize spec
    # (the exporter would otherwise look for uncalled submodules).
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

    write_bundle_metadata(out_dir, name, args.hf_id, cfg.vocab_size, args.max_ctx)
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(args.hf_id).save_pretrained(out_dir / "tokenizer")
    print(f"bundle ready: {out_dir}")
    print(f"run: COREAI_CHUNK_THRESHOLD=1 llm-benchmark --model {out_dir} -p 128 -g 256 -n 3")


if __name__ == "__main__":
    main()
