"""FLAGSHIP QUALITY-SAFE byte-reduction stack: Qwen3.6-35B-A3B decode with (a) routed experts on
``gather_qmm`` AND (b) the fused **FP4 (E2M1)** matvec on the dense path (lm_head + full-attention
q/o). This is the quality-safe twin of ``export_qwen3_6_moe_dense_int4km_decode_pipelined.py``: SAME
¼-byte weight reads (~2x decode bandwidth, measured fp4 == int4km speed in ``_dense_fp4_microbench.py``)
but E2M1 numerics instead of int4 k-means -> int8-level quality (fp4 ≈ int8 perplexity),
fixing int4's flagship quality cliff (the Nanbeige/RTN degradation).

vs the shipped ``export_qwen3_6_moe_metal_decode_pipelined.py`` (experts sym8, dense int8, int8 head):
  * routed experts:  sym8 -> km4 / km8 / sym8   (per ``mode``)
  * lm_head:         int8 head -> **fp4**
  * attn q/o (full): int8 -> **fp4**             (k/v_proj stay fp16 -> N=512, small-N never pays)

The fp4 matvec is a hand-rolled DECODE-bandwidth kernel (¼ weight bytes) — NOT TensorOps ``matmul2d``
(the A19-refuted *prefill* lever) — so it runs on the Mac GPU today and needs no OS27.

GATE before any claim: on-device decode tok/s up (fp4 == int4km bandwidth) AND multi-token reasoning
token-match toward the fp16/int8 baseline (fp4 should track int8, UNLIKE int4 which flips ~12/41 tokens).

Run:  cd ~/code/coreai/coreai-models && .venv/bin/python \
          ../coreai-models-community/conversion/export_qwen3_6_moe_dense_fp4_decode_pipelined.py \
          int4km --hf-id Qwen/Qwen3.6-35B-A3B
      (``mode`` selects the ROUTED-expert scheme; the dense path is always fp4.)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from _bundle import save_tokenizer, write_bundle_metadata

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS
from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels
from coreai_models.models.macos.gemma4_metal_mlp_fp4 import (
    build_fused_fp4_kernel,
    metalize_dense_fp4,
)
from coreai_models.models.macos.moe_metal import metalize_moe
from coreai_models.models.macos.qwen3_5_moe import (
    DECODE_STATE_NAMES,
    Qwen3_5MoeStatefulForCausalLM,
    build_decode_state,
)
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8") -> dict:
    """int8 per-block-32 for the non-routed-expert weights, but ALSO keep lm_head + full-attn q/o
    as fp16 (excluded) so the fp4 kernel can quantize them itself below."""
    block_2d = {
        "dtype": dtype,
        "qscheme": "symmetric_with_clipping",
        "granularity": {"type": "per_block", "block_size": 32, "axis": 1},
    }
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {"weight": block_2d},
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
            "coreai_models.primitives.macos.switch.SwitchLinear": None,
        },
        "module_name_configs": {
            r".*mlp\.gate$": None,
            r".*shared_expert_gate$": None,
            r".*lm_head$": None,                                # fp4'd below
            r".*self_attn\.(q_proj|k_proj|v_proj|o_proj)$": None,  # q/o fp4'd below; k/v stay fp16
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="int4km", choices=["sym8", "int8km", "int4km"],
                    help="routed-expert scheme (dense path is always fp4)")
    ap.add_argument("--hf-id", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--num-layers", type=int, default=None, help="debug: truncated-layer export")
    ap.add_argument("--no-quant-mmap", action="store_true")
    args = ap.parse_args()

    scheme = {"sym8": "sym8", "int8km": "km8", "int4km": "km4"}[args.mode]
    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_decode_{args.mode}_gather_dense_fp4"
    if args.num_layers is not None:
        name += f"_l{args.num_layers}"

    print(f"loading {args.hf_id} fp16 ...", flush=True)
    model = Qwen3_5MoeStatefulForCausalLM.from_hf_memory_efficient(
        args.hf_id, max_context_length=args.max_ctx, target_dtype=DTYPE, num_layers=args.num_layers)
    model.eval()
    cfg = model.config

    n_lin = 0
    for layer in model.model.layers:
        if not layer.is_full:
            layer.linear_attn.use_loopfree_step = True
            n_lin += 1
    print(f"loop-free single-step on {n_lin} linear layers; E={cfg.num_experts}/top{cfg.num_experts_per_tok}, "
          f"hidden={cfg.hidden_size}, vocab={cfg.vocab_size}", flush=True)

    trace_past = 64
    input_ids = torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32)
    position_ids = torch.arange(trace_past + 1, dtype=torch.int32).unsqueeze(0)
    state = build_decode_state(cfg, max_seq_len=TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)
    reference_inputs = {
        "input_ids": input_ids, "position_ids": position_ids,
        "k_cache": state["k_cache"], "v_cache": state["v_cache"],
        "conv_state": state["conv_state"], "rec_state": state["rec_state"],
    }
    seq_pos = torch.export.Dim("seq_pos", min=2, max=args.max_ctx - 1)
    k_seq = torch.export.Dim("k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    v_seq = torch.export.Dim("v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    dynamic_shapes = {
        "input_ids": None, "position_ids": {1: seq_pos},
        "k_cache": {KVCache.seq_len_dim(): k_seq}, "v_cache": {KVCache.seq_len_dim(): v_seq},
        "conv_state": None, "rec_state": None,
    }

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # 1) int8 the non-routed-expert weights (routed experts + lm_head + attn q/o excluded -> fp16)
    from coreai_models.export.compression import quantize_pytorch_model
    cfg_q = linear_quant_config("int8")
    quant_mmap = None
    if not args.no_quant_mmap:
        quant_mmap = out_dir / "_quant_mmap"
        quant_mmap.mkdir()
    print("quantizing non-expert linears (int8; experts + lm_head + attn q/o kept fp16) ...", flush=True)
    model = quantize_pytorch_model(
        model, tuple(reference_inputs.values()), dynamic_shapes, cfg_q,
        mmap_dir=str(quant_mmap) if quant_mmap else None)

    # 2) routed MoE -> gather_qmm (reads only top-k)
    print(f"metalizing routed MoE -> gather_qmm {args.mode} ...", flush=True)
    moe_kernel = metalize_moe(model, scheme=scheme)

    # 3) dense path -> fused FP4 (lm_head + full-attn q/o); share ONE fp4 kernel
    dense_kernel = build_fused_fp4_kernel(name="qwen36_dense_fp4")
    n_dense = metalize_dense_fp4(model, dense_kernel)
    print(f"metalized dense fp4: {n_dense} matvecs (lm_head + full-attn q/o)", flush=True)

    # 4) export with BOTH kernels embedded
    specs = [s for s in _EXTERNALIZE_SPECS
             if s.composite_op_name not in ("gated_delta_update", "gather_mm")]
    print("exporting decode-only graph (gather_qmm + dense fp4 embedded) ...", flush=True)
    prog = export_to_coreai_with_kernels(
        model, reference_inputs=reference_inputs, custom_kernels=[moe_kernel, dense_kernel],
        dynamic_shapes=dynamic_shapes, input_names=("input_ids", "position_ids"),
        output_names=("logits",), state_names=DECODE_STATE_NAMES, externalize_modules=specs)
    print("optimizing ...", flush=True)
    prog.optimize()

    import coreai.runtime as rt
    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    write_bundle_metadata(out_dir, name, args.hf_id, cfg.vocab_size, args.max_ctx)
    save_tokenizer(args.hf_id, out_dir, via_transformers=False)
    if quant_mmap is not None:
        shutil.rmtree(quant_mmap, ignore_errors=True)
    print(f"bundle ready: {out_dir}")
    print(f"run: COREAI_CHUNK_THRESHOLD=1 llm-benchmark --model {out_dir} -p 128 -g 256 -n 3")


if __name__ == "__main__":
    main()
