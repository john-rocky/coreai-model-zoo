"""Export the Qwen3.8-27B VISION path: fixed-grid tower + embeddings-input decoder.

Three artifacts (the text bundle `qwen3_8_27b_decode_int8hu_block32_sym` ships
unchanged next to these — combined release):

* ``qwen3_8_27b_vision_fp16/`` — the 458M vision tower as a one-shot fp16
  ``.aimodel``: ``patches [1024, 1536] -> image_embeds [256, 5120]`` at the
  canonical 512x512 grid (32x32 patches, 2x2 merge). Host preprocessing spec:
  ``_smoke/qwen38vl_preprocess.py`` (gated vs the HF processor).

* ``qwen3_8_27b_vl_decode_<mode>_pf16/`` — the text decoder as an
  EMBEDDINGS-INPUT multifunction bundle (`Qwen3_5VLStatefulEmbeds`): "main" =
  static S=1 decode, "prefill" = static S=16 chunk (chunked GDN scan — see the
  PF constant below for why 16 and not 32), shared weights. Interleaved mRoPE from three host-fed position planes; host contract
  in ``_smoke/qwen38vl_host.py``. int8hu = the text bundle's ship recipe
  (per-block-32 linear int8 body + absmax-sym int8 untied head).

* ``embed_tokens.safetensors`` (fp16, inside the decoder bundle) — the host
  gathers text-token rows from this table and splices tower rows at
  ``<|image_pad|>`` before calling the graph (the graph holds no embed table).

Gating: python suite gates only (`llm-runner` cannot bind an embeddings input)
— `_smoke/test_qwen38vl_tower_gate.py --stage aimodel` and
`_smoke/test_qwen38vl_suite_gate.py`. Never gate numerics via raw
`AIModel.load(...gpu())` on the multi-GB graph.

Run:  python export_qwen38vl_pipelined.py [fp16|int8hu] \
          [--hf-id Qwen/Qwen3.8-27B] [--out-dir exports] [--skip-vision]
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import torch
from _bundle import head_quant_spec, write_bundle_metadata
from export_qwen3_5_decode_pipelined import linear_quant_config

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import (
    _EXTERNALIZE_SPECS,
    export_to_coreai,
    export_to_coreai_multifunction,
)
from coreai_models.models.macos.qwen3_5 import DECODE_STATE_NAMES, Qwen3_5VLStatefulEmbeds
from coreai_models.models.macos.qwen3_5_vision import Qwen3_5VisionEncoder

DTYPE = torch.float16
# PF chunk 16, not 32: the in-graph doubling-inverse runs in fp16 on the GPU
# delegate, and its worst-case intermediate growth is ~C(PF-1, PF/2-1) — ~6e3 at
# PF=16 (inside fp16's 65504) vs ~3e8 at PF=32. PF=32 passed the 6-case suite
# but collapses CONTENT-DEPENDENTLY on real images (weak-decay image spans;
# reproduced 2026-08-15 on two Pexels photos — "!" spam from the first token).
PF = 16
GRID = 16  # default merged grid side: 16x16 = 256 tokens = a 512x512 tile
# --grid-h/--grid-w override it: the tower authoring is generic over the grid,
# so a portrait document page can bake its own aspect instead of being squashed
# into the square tile (OvisOCR2 ships 28x40 = 1120 tokens = 896x1280).


def export_vision(args) -> None:
    name = f"{args.name}_vision_fp16"
    out_dir = Path(args.out_dir) / name
    print(f"loading vision tower ({args.hf_id}) ...")
    vis = Qwen3_5VisionEncoder.from_hf(
        args.hf_id, target_dtype=DTYPE, grid_h=args.grid_h, grid_w=args.grid_w)
    patch_dim = (vis.vcfg["in_channels"] * vis.vcfg["temporal_patch_size"]
                 * vis.vcfg["patch_size"] ** 2)
    patches = torch.zeros(vis.n_patches, patch_dim, dtype=DTYPE)

    print("exporting vision graph ...")
    prog = export_to_coreai(
        vis,
        {"patches": patches},
        dynamic_shapes={"patches": None},
        input_names=("patches",),
        output_names=("image_embeds",),
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

    prog.save_asset(out_dir / f"{name}.aimodel", rt.AIModelAssetMetadata())
    print(f"vision ready: {out_dir}")


def _install_externalize_dim_retry() -> None:
    """coreai-torch bug workaround (verbatim from the Qwen3-VL pf64 export):
    the externalize pipeline exports composite submodules with fallback
    Dim(min=1); a static S=PF causal SDPA generates a `key_seq >= PF` guard the
    fallback violates. Retry with the bounds torch suggests."""
    import coreai_torch.converter as _ct_conv
    from torch.export import Dim as _Dim

    # coreai-torch 0.4.2 moved _torch_export_module converter -> externalize
    # (same name/signature); patch whichever module carries it.
    if not hasattr(_ct_conv, "_torch_export_module"):
        import coreai_torch.externalize as _ct_conv

    _orig = _ct_conv._torch_export_module

    def _with_retry(prep):
        for _ in range(3):
            try:
                return _orig(prep)
            except Exception as e:  # noqa: BLE001 — retry only on suggested fixes
                fixes = {
                    name: (int(mn), int(mx) if mx else None)
                    for name, mn, mx in re.findall(
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

                prep.dynamic_shapes = tuple(_remap(d) for d in prep.dynamic_shapes)
        return _orig(prep)

    _ct_conv._torch_export_module = _with_retry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="int8hu", choices=["fp16", "int8hu"])
    ap.add_argument("--hf-id", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--name", default="qwen3_8_27b",
                    help="bundle name stem: <name>_vision_fp16 / <name>_vl_decode_...")
    ap.add_argument("--grid-h", type=int, default=GRID,
                    help="merged grid rows (each merged token = 32x32 px)")
    ap.add_argument("--grid-w", type=int, default=GRID,
                    help="merged grid cols (each merged token = 32x32 px)")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--skip-vision", action="store_true")
    ap.add_argument("--skip-decoder", action="store_true")
    ap.add_argument("--no-prefill", action="store_true",
                    help="export ONLY the S=1 'main' function. One static shape means the AOT "
                         "needs no --expect-frequent-reshapes, which is what the flag's ~4.7x "
                         "size inflation buys back; the host then ingests the prompt at S=1.")
    ap.add_argument("--num-layers", type=int, default=None,
                    help="debug: truncated-layer export (engine-contract de-risk)")
    args = ap.parse_args()

    if not args.skip_vision:
        export_vision(args)
    if args.skip_decoder:
        return

    suffix = "int8hu_block32_sym" if args.mode == "int8hu" else "fp16"
    name = (f"{args.name}_vl_decode_{suffix}_s1" if args.no_prefill
            else f"{args.name}_vl_decode_{suffix}_pf{PF}")
    if args.num_layers is not None:
        name += f"_l{args.num_layers}"

    print(f"loading {args.hf_id} text decoder fp16 (embeds variant) ...")
    model = Qwen3_5VLStatefulEmbeds.from_hf_memory_efficient(
        args.hf_id, max_context_length=args.max_ctx, target_dtype=DTYPE,
        hf_config_attr="text_config", num_layers=args.num_layers)
    model.eval()
    cfg = model.config

    n_lin = 0
    for layer in model.model.layers:
        if not layer.is_full:
            layer.linear_attn.use_loopfree_chunk = True
            n_lin += 1
    print(f"loop-free chunked GDN enabled on {n_lin} linear layers "
          f"(static S=1 entry short-circuits to the single step)")

    # The host-side embed table, BEFORE quantization touches anything.
    embed_fp16 = model.model.embed_tokens.weight.detach().clone()

    spec_main = model.build_vl_export_spec(
        DTYPE, args.max_ctx, query_len=1, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN)

    if args.mode == "int8hu":
        from coreai_models.export.compression import quantize_pytorch_model

        cfg_q = linear_quant_config("int8")
        cfg_q["module_name_configs"] = {r".*lm_head$": head_quant_spec("block32", True)}
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        print("quantizing (linear int8 per-block-32 + absmax-sym int8 head) ...")
        model = quantize_pytorch_model(
            model, tuple(spec_main["reference_inputs"].values()),
            spec_main["dynamic_shapes"], cfg_q)

    _install_externalize_dim_retry()
    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    entries = [
        ("main", model.build_vl_export_spec(
            DTYPE, args.max_ctx, query_len=1, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN)),
    ] if args.no_prefill else [
        ("main", model.build_vl_export_spec(
            DTYPE, args.max_ctx, query_len=1, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN)),
        ("prefill", model.build_vl_export_spec(
            DTYPE, args.max_ctx, query_len=PF, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN)),
    ]
    print("exporting decode-only decoder (main S=1, no prefill) ..." if args.no_prefill
          else f"exporting multifunction decoder (main S=1 + prefill S={PF}) ...")
    prog = export_to_coreai_multifunction(model, entries, externalize_modules=specs)
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
    write_bundle_metadata(out_dir, name, args.hf_id, cfg.vocab_size, args.max_ctx,
                          functions=("main",) if args.no_prefill else ("main", "prefill"))

    from safetensors.torch import save_file

    save_file({"embed_tokens.weight": embed_fp16.contiguous()},
              str(out_dir / "embed_tokens.safetensors"))
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(args.hf_id).save_pretrained(out_dir / "tokenizer")
    print(f"bundle ready: {out_dir}")
    print(f"state names: {DECODE_STATE_NAMES}")
    print("gate: _smoke/test_qwen38vl_suite_gate.py (python runtime, _GPU_LOCK held)")


if __name__ == "__main__":
    main()
