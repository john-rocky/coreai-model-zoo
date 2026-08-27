"""Export Qwen3-VL (2B) for the Core AI pipelined engine — the zoo's first VLM.

Two artifacts per run:

* ``<name>/`` — the TEXT DECODER LanguageBundle (``.aimodel`` + metadata +
  tokenizer) on the unmodified pipelined-engine contract: dynamic-query
  ``input_ids``/``position_ids``, ONE KV pair, logits out. The multimodal
  state rides the static-input hook (apps/coreai-pipelined-static-inputs.patch):
  ``image_embeds [N,h]``, ``deepstack_embeds [3N,h]``, ``rope_shift_start [1]``,
  ``rope_shift_amount [1]``. With zero embeds and shift_start=1<<30 the graph
  is a plain Qwen3 text decoder (llm-benchmark runs it out of the box).
  Host contract: rewrite ``<|image_pad|>`` ids to ``V + slot``; after an image
  at merged grid HxW set shift_start = img_start + N, shift_amount =
  N - max(H, W). Pure attention -> native chunked prefill, no extra states.

* ``<name>_vision/`` — the fixed-grid VISION ENCODER ``.aimodel``:
  ``patches [n_patch, 1536] -> (image_embeds [N, 2048],
  deepstack_embeds [3N, 2048])``, fp16. Run once per image, write the outputs
  into the decoder's static-input buffers.

Numerics gating: torch ladder (_smoke/test_qwen3vl_torch_ladder.py) PASSED
16/16 + per-layer cos 1.000 before this script existed; the .aimodel gate is
_smoke/test_qwen3vl_aimodel_gate.py.

Run:  python export_qwen3_vl_pipelined.py [fp16|int8lin|int8hu] \
          [--hf-id Qwen/Qwen3-VL-2B-Instruct] [--out-dir exports] [--skip-vision]

Modes follow the qwen3.5 recipe: int8lin = per-block-32 linear int8 body;
int8hu = + untied int8 head, absmax symmetric (big-vocab fat-tail rule:
clipping crushes outlier head rows). Vision stays fp16 in all modes.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from _bundle import head_quant_spec, write_bundle_metadata

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.qwen3_vl import (
    PIPELINED_STATE_NAMES,
    Qwen3VLPipelinedForCausalLM,
    Qwen3VLVisionEncoder,
)

DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8") -> dict:
    """Weight-only linear per-block-32 (the qwen3.5 ship recipe)."""
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


def export_vision(args, name: str) -> None:
    out_dir = Path(args.out_dir) / f"{name}_vision"
    print(f"loading vision tower ({args.hf_id}) ...")
    vis = Qwen3VLVisionEncoder.from_hf(
        args.hf_id, target_dtype=DTYPE, grid_h=args.grid, grid_w=args.grid)
    vcfg = vis.vcfg
    patch_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size ** 2
    patches = torch.zeros(vis.n_patches, patch_dim, dtype=DTYPE)

    print("exporting vision graph ...")
    prog = export_to_coreai(
        vis,
        {"patches": patches},
        dynamic_shapes={"patches": None},
        input_names=("patches",),
        output_names=("image_embeds", "deepstack_embeds"),
        state_names=(),
        externalize_modules=[
            s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"
        ],
    )
    prog.optimize()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    import coreai.runtime as rt

    prog.save_asset(out_dir / f"{name}_vision.aimodel", rt.AIModelAssetMetadata())
    print(f"vision ready: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="int8lin",
                    choices=["fp16", "int8lin", "int8hu"])
    ap.add_argument("--hf-id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--grid", type=int, default=14,
                    help="merged vision grid side (14 = 448x448 input, 196 tokens)")
    ap.add_argument("--skip-vision", action="store_true")
    ap.add_argument("--skip-decoder", action="store_true")
    ap.add_argument("--skip-dynamic", action="store_true",
                    help="skip the dynamic-query engine bundle")
    ap.add_argument("--skip-s1", action="store_true",
                    help="skip the static S=1 gate bundle")
    ap.add_argument("--prefill-chunk", type=int, default=0,
                    help="also export a MULTIFUNCTION bundle (suffix _pfN): "
                         "'main' = static S=1 decode + 'prefill' = static S=N "
                         "chunk, weights shared. Works around the MPSGraph "
                         "GPURegionRuntime crash on dynamic-ids graphs.")
    args = ap.parse_args()

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_decode_{args.mode}"

    if not args.skip_vision:
        export_vision(args, short)
    if args.skip_decoder:
        return

    print(f"loading {args.hf_id} text decoder fp16 ...")
    model = Qwen3VLPipelinedForCausalLM.from_hf(
        args.hf_id, target_dtype=DTYPE, grid_h=args.grid, grid_w=args.grid)
    cfg = model.config

    spec = model.build_export_spec(
        DTYPE, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN)

    if args.mode in ("int8lin", "int8hu"):
        from coreai_models.export.compression import quantize_pytorch_model

        cfg_q = linear_quant_config("int8")
        if args.mode == "int8hu":
            cfg_q["module_name_configs"] = {r".*lm_head$": head_quant_spec("block32", True)}
            model.lm_head.weight = torch.nn.Parameter(
                model.lm_head.weight.detach().clone())
        print(f"quantizing ({args.mode}) ...")
        model = quantize_pytorch_model(
            model, tuple(spec["reference_inputs"].values()),
            spec["dynamic_shapes"], cfg_q)

    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    import coreai.runtime as rt
    from transformers import AutoTokenizer

    # Two graphs from the same (quantized) weights:
    #   dynamic query  -> the engine ship bundle (chunked prefill + decode)
    #   static S=1     -> the python oracle-gate bundle (suffix _s1); the
    #                     python runtime cannot run dynamic-shaped outputs.
    variants = []
    if not args.skip_dynamic:
        variants.append((name, spec))
    if not args.skip_s1:
        variants.append((f"{name}_s1", model.build_export_spec(
            DTYPE, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, trace_query=1)))

    for vname, vspec in variants:
        print(f"exporting decoder graph ({vname}) ...")
        prog = export_to_coreai(
            model,
            vspec["reference_inputs"],
            dynamic_shapes=vspec["dynamic_shapes"],
            input_names=vspec["input_names"],
            output_names=vspec["output_names"],
            state_names=PIPELINED_STATE_NAMES,
            externalize_modules=specs,
        )
        print("optimizing ...")
        prog.optimize()

        out_dir = Path(args.out_dir) / vname
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        aimodel = out_dir / f"{vname}.aimodel"
        print(f"saving {aimodel} ...")
        prog.save_asset(aimodel, rt.AIModelAssetMetadata())
        write_bundle_metadata(out_dir, vname, args.hf_id, cfg.vocab_size, args.max_ctx)
        AutoTokenizer.from_pretrained(args.hf_id).save_pretrained(out_dir / "tokenizer")
        print(f"bundle ready: {out_dir}")

    if args.prefill_chunk:
        from coreai_models.export.macos import export_to_coreai_multifunction

        # coreai-torch bug workaround: the externalize pipeline exports each
        # composite submodule with fallback Dim(min=1) for dims whose parent
        # expression has no var_to_range entry (constants / derived exprs).
        # A static S=64 causal SDPA generates a `key_seq >= 64` guard inside
        # the submodule, which min=1 violates. The submodule body must stay
        # DYNAMIC (its MLIR call sites are parametric — pinning it static
        # fails optimize() with "input types did not match callee
        # signature"), so the fix is to keep the Dims and retry the export
        # with the min/max bounds torch itself suggests in the
        # ConstraintViolation message.
        import re as _re

        # coreai-torch 0.4.2 moved _torch_export_module converter -> externalize
        # (same name/signature); patch whichever module carries it.
        import coreai_torch.converter as _ct_conv
        if not hasattr(_ct_conv, "_torch_export_module"):
            import coreai_torch.externalize as _ct_conv
        from torch.export import Dim as _Dim

        _orig_export_module = _ct_conv._torch_export_module

        def _export_module_with_retry(prep):
            for _attempt in range(3):
                try:
                    return _orig_export_module(prep)
                except Exception as e:  # noqa: BLE001 — retry only on suggested fixes
                    fixes = {
                        name: (int(mn), int(mx) if mx else None)
                        for name, mn, mx in _re.findall(
                            r"(\w+) = Dim\('\w+', min=(\d+)(?:, max=(\d+))?\)", str(e))
                    }
                    if not fixes:
                        raise
                    print(f"[externalize-retry] {prep.name}: {fixes}")
                    rebuilt: dict[str, object] = {}

                    def _remap(dims):
                        if dims is None:
                            return None
                        out = {}
                        for j, d in dims.items():
                            name = getattr(d, "__name__", None)
                            if name not in fixes:
                                out[j] = d
                                continue
                            if name not in rebuilt:
                                mn, mx = fixes[name]
                                kwargs = {"min": mn}
                                if mx is not None:
                                    kwargs["max"] = mx
                                rebuilt[name] = _Dim(name, **kwargs)
                            out[j] = rebuilt[name]
                        return out

                    prep.dynamic_shapes = tuple(
                        _remap(dims) for dims in prep.dynamic_shapes)
            return _orig_export_module(prep)

        _ct_conv._torch_export_module = _export_module_with_retry

        pf = args.prefill_chunk
        vname = f"{name}_pf{pf}"
        print(f"exporting multifunction decoder (main S=1 + prefill S={pf}) ...")
        entries = [
            ("main", model.build_export_spec(
                DTYPE, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN,
                trace_query=1)),
            ("prefill", model.build_export_spec(
                DTYPE, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN,
                trace_query=pf, static_ids=True)),
        ]
        prog = export_to_coreai_multifunction(model, entries, externalize_modules=specs)
        print("optimizing ...")
        prog.optimize()

        out_dir = Path(args.out_dir) / vname
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        aimodel = out_dir / f"{vname}.aimodel"
        print(f"saving {aimodel} ...")
        prog.save_asset(aimodel, rt.AIModelAssetMetadata())
        write_bundle_metadata(out_dir, vname, args.hf_id, cfg.vocab_size, args.max_ctx,
                              functions=("main", "prefill"))
        AutoTokenizer.from_pretrained(args.hf_id).save_pretrained(out_dir / "tokenizer")
        print(f"bundle ready: {out_dir}")


if __name__ == "__main__":
    main()
