"""Export the GLM-Image AR generator (GLM-4-9B visual-token decoder) to Core AI.

The AR half of zai-org/GLM-Image. Decode-graph contract (bespoke image-pipeline
loop, not the chat engine):

  (input_ids [1,s], position_ids [1,total] ramp, cos [1,s,64], sin [1,s,64],
   keyCache, valueCache) -> logits [1,s,16512]

mrope is pre-computed on the host (cos/sin are graph inputs). See the model def
coreai_models.models.macos.glm_image_ar.

Run (coreai-models venv, from conversion/):
  python export_glm_image_ar.py [int8lin|int8hu|fp16] \
      --vle-dir <snapshot>/vision_language_encoder --out-dir exports
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from _bundle import head_quant_spec

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.glm_image_ar import (
    PIPELINED_STATE_NAMES,
    GlmImageARForCausalLM,
)

DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8") -> dict:
    """Weight-only linear per-block-32 (the qwen3.5 ship recipe). Embedding /
    composite ops / lm_head excluded (head stays fp16 unless int8hu)."""
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
                    choices=["fp16", "int8lin", "int8hu"])
    ap.add_argument("--vle-dir", required=True,
                    help="local GLM-Image vision_language_encoder folder")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--skip-dynamic", action="store_true",
                    help="skip the dynamic-query engine bundle")
    ap.add_argument("--skip-s1", action="store_true",
                    help="skip the static S=1 gate bundle")
    args = ap.parse_args()

    name = f"glm_image_ar_decode_{args.mode}"

    print(f"loading GLM-Image AR decoder ({args.mode}) fp16 ...", flush=True)
    model = GlmImageARForCausalLM.from_pretrained_dir(args.vle_dir, target_dtype=DTYPE)
    cfg = model.config

    spec = model.build_export_spec(DTYPE, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN)

    if args.mode in ("int8lin", "int8hu"):
        from coreai_models.export.compression import quantize_pytorch_model

        cfg_q = linear_quant_config("int8")
        if args.mode == "int8hu":
            # Optional here, unlike the text models: this head is vision_vocab=16512, well
            # short of the fat tail the big-vocab rule exists for.
            cfg_q["module_name_configs"] = {r".*lm_head$": head_quant_spec("block32", True)}
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        print(f"quantizing ({args.mode}) ...", flush=True)
        model = quantize_pytorch_model(
            model, tuple(spec["reference_inputs"].values()),
            spec["dynamic_shapes"], cfg_q)

    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    import coreai.runtime as rt

    variants = []
    if not args.skip_dynamic:
        variants.append((name, spec))
    if not args.skip_s1:
        variants.append((f"{name}_s1", model.build_export_spec(
            DTYPE, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, trace_query=1)))

    for vname, vspec in variants:
        print(f"exporting decoder graph ({vname}) ...", flush=True)
        prog = export_to_coreai(
            model,
            vspec["reference_inputs"],
            dynamic_shapes=vspec["dynamic_shapes"],
            input_names=vspec["input_names"],
            output_names=vspec["output_names"],
            state_names=PIPELINED_STATE_NAMES,
            externalize_modules=specs,
        )
        print("optimizing ...", flush=True)
        prog.optimize()

        out_dir = Path(args.out_dir) / vname
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        aimodel = out_dir / f"{vname}.aimodel"
        print(f"saving {aimodel} ...", flush=True)
        prog.save_asset(aimodel, rt.AIModelAssetMetadata())
        print(f"bundle ready: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
