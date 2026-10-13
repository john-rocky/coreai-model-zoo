#!/usr/bin/env python3
"""Export the MiniCPM-V-4.7-1B VL decoder (S=1 decode, canvas M-RoPE in the graph) to a Core AI bundle.

The graph is `minicpmv47_vl_pipelined.MiniCPMV47VLPipelinedForCausalLM` (contract in its header):

    input_ids [1,1] i32 (static; image row k = V + k), position_ids [1,seq] i32 (dynamic),
    image_embeds [64,1024] fp16, rope_shift_start [1] i32, rope_shift_amount [1] i32
    + keyCache / valueCache (dynamic sequence dim) / convState / recState -> logits [1,1,248094]

Built as `decider_vision/export_decoder.py` builds its S=1 bundle: every linear-attention layer on
the loop-free single step (`use_loopfree_step`), static S=1 ids, dynamic position and KV dims,
gated_delta_update left out of the externalized composites, `prog.optimize()`. Two modes:

    fp16     the reference: every weight fp16
    int8hu   body linears int8 per-block-32 `symmetric_with_clipping` (SDPA / RoPE / the three
             RMSNorms / Embedding / Conv1d excluded, MiniCPM-V-4.6's `linear_quant_config`) +
             the checkpoint's own untied `lm_head.weight` int8 per-block-32 plain `symmetric`
             (absmax, `_bundle.head_quant_spec("block32", True)`). 4.7's head is untied in the
             checkpoint, so there is nothing to clone; the embed table stays fp16.

`--prefill-chunk S` (round 5, S = 16; `decider_vision/export_decoder.py --prefill-chunk`) makes the
asset multifunction (`export_to_coreai_multifunction`, constants shared): "main" = the S=1 decode
graph above, "prefill" = static input_ids [1, S] with the same static image inputs and the logits
of the chunk's LAST position ([1, 1, V], `last_token_only`; a no-op slice in "main"). Every
linear-attention layer then also gets `use_loopfree_unroll` (the overlay's
`_gated_delta_step_unroll`: S single steps unrolled in-graph, fp32 inside the call, no doubling
inverse; at S=1 the one unrolled step is the single-step formula), and the static-S causal SDPA's
externalize guard is retried with torch's suggested bounds
(`export_qwen38vl_pipelined._install_externalize_dim_retry`). The bundle name gets `_pf<S>`. The
canvas planes are per token from (ids, position, start, amount), so a chunk may cut the image block
anywhere. Host order: "prefill" on ids[c : c + S] with position_ids 0..c+S-1 while a whole chunk
fits, then "main" one token at a time (`metadata.json` `vlm.prefill_order`).

Bundle `<exports>/minicpmv47_vlm_decode_<mode>/` = `<name>.aimodel` + `metadata.json` (kind llm,
the language block, the `vlm` block with the host contract) + `tokenizer/` (the pinned snapshot's
tokenizer.json, tokenizer_config.json, chat_template.jinja, generation_config.json, verbatim,
sha256-checked against the snapshot). `--record` writes the export record (load report, quant
counts, seconds, per-file sha256 of the .aimodel, tokenizer sha256).

    QW coreai-models/.venv/bin/python conversion/minicpmv47/export_decoder.py int8hu --record <json>
    QW coreai-models/.venv/bin/python conversion/minicpmv47/export_decoder.py int8hu --prefill-chunk 16 --record <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _paths import exports_dir, hf_snapshot  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")

HF_ID = "openbmb/MiniCPM-V-4.7-1B"
REVISION = "984a89aafcd5160164f4f0d04b4be0f4eec03cb8"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")
STOP_IDS = [248044, 248046]
PROMPT_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(path: Path) -> dict:
    """sha256 per file of a directory asset + one digest over the sorted listing."""
    files = sorted(p for p in path.rglob("*") if p.is_file())
    per = {str(p.relative_to(path)): sha256_file(p) for p in files}
    tree = hashlib.sha256("".join(f"{k}\0{v}\n" for k, v in per.items()).encode()).hexdigest()
    return {"bytes": sum(p.stat().st_size for p in files), "tree_sha256": tree,
            "files": {k: {"bytes": (path / k).stat().st_size, "sha256": v} for k, v in per.items()}}


def linear_quant_config(dtype: str = "int8") -> dict:
    """Weight-only linear int8 per-block-32 (MiniCPM-V-4.6 `export_minicpmv46_vlm_pipelined.py`):
    SDPA / RoPE / norms / Embedding / Conv1d excluded; lm_head gets its own spec in int8hu."""
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


def vlm_block(n_image: int, hidden: int, vocab: int) -> dict:
    return {
        "image_embeds": {"shape": [n_image, hidden], "dtype": "float16", "binding": "static",
                         "rows": "tower image_features rows 0..63 (row-major 8x8 merged grid); zero for a text-only prompt"},
        "rope_shift_start": {"shape": [1], "dtype": "int32", "binding": "static"},
        "rope_shift_amount": {"shape": [1], "dtype": "int32", "binding": "static"},
        "image_token_base": vocab,
        "image_ids": "the 64 <|image_pad|> (248056) between <image> (248078) and </image> (248079) become V + k, k = 0..63",
        "canvas_rule": "transformers 5.19.0 MiniCPMV4_7Model.get_rope_index (canvas M-RoPE, interleaved [11,11,10]) for one "
                       "thumbnail-only 448x448 image, derived in the graph from (ids, position, start, amount): "
                       "O = start - amount - 9; p < O -> (p,p,p); <image> -> (O, O-1, O-1); row k -> (O, O+k//8, O+k%8); "
                       "</image> -> (O, O+8, O+8); other tokens in [O, start) -> (O, O, O); p >= start -> (p-amount) x3",
        "host_start_amount": "start = first index after </image> and the marker tokens that follow it (198, <image>, "
                             "</image>, <slice>, </slice>); amount = start - (i0 + 9), i0 = index of <image>; "
                             "text-only: start = 1 << 30, amount = 0 (minicpmv47_vl_pipelined.host_static_inputs)",
        "image_block": "<image_id>0</image_id><image> + <|image_pad|> x64 + </image> (processor with max_slice_nums=1; "
                       "a 448x448 input is thumbnail-only)",
        "tower": "minicpmv47_vision_<mode> (pixel_values [1,3,448,448] -> image_features [64,1024])",
    }


def export(args, out_dir: Path, name: str) -> dict:
    import torch
    import torch.nn.utils.parametrize as P
    from _bundle import head_quant_spec, write_bundle_metadata
    from minicpmv47_vl_pipelined import N_IMAGE, MiniCPMV47VLPipelinedForCausalLM

    from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
    from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai, export_to_coreai_multifunction

    snap = Path(hf_snapshot(HF_ID, revision=REVISION))
    dtype = torch.float16
    chunk = args.prefill_chunk
    t0 = time.monotonic()
    print(f"[load] {snap.name} text decoder + untied head, fp16 ...", flush=True)
    model = MiniCPMV47VLPipelinedForCausalLM.from_checkpoint(snap, target_dtype=dtype)
    report = model.load_report
    n_lin = 0
    for layer in model.model.layers:
        if not layer.is_full:
            layer.linear_attn.use_loopfree_step = True
            if chunk:
                # Precedence over the single step whenever the traced S is a static int: the
                # prefill trace unrolls `chunk` steps, the S=1 main trace unrolls one.
                layer.linear_attn.use_loopfree_unroll = True
            n_lin += 1
    gdn = "unrolled step scan (S=1 main: one step)" if chunk else "loop-free single step"
    print(f"[load] {json.dumps(report)}; {gdn} on {n_lin} linear layers", flush=True)
    cfg = model.config
    spec = model.build_export_spec(dtype, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, query_len=1)
    spec_pf = None
    if chunk:
        model.last_token_only = True  # prefill logits [1, 1, V]; a no-op slice at S=1
        spec_pf = model.build_export_spec(dtype, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN,
                                          query_len=chunk)
    t_loaded = time.monotonic()

    quant = None
    if args.mode == "int8hu":
        from coreai_models.export.compression import quantize_pytorch_model

        cfg_q = linear_quant_config("int8")
        cfg_q["module_name_configs"] = {r".*lm_head$": head_quant_spec("block32", True)}
        if model.lm_head.weight is model.model.embed_tokens.weight:
            sys.exit("int8hu: lm_head is tied to embed_tokens; 4.7's checkpoint head is untied")
        print("[quant] body linear int8 per-block-32 symmetric_with_clipping + lm_head int8 per-block-32 "
              "symmetric (absmax); SDPA/RoPE/RMSNorm*/Embedding/Conv1d fp16 ...", flush=True)
        model = quantize_pytorch_model(model, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"], cfg_q)
        lins = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
        q8 = [n for n, m in lins if P.is_parametrized(m, "weight")]
        fp = [n for n, m in lins if not P.is_parametrized(m, "weight")]
        emb_q = P.is_parametrized(model.model.embed_tokens, "weight")
        conv_q = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Conv1d) and P.is_parametrized(m, "weight")]
        quant = {"linear": linear_quant_config("int8")["global_config"]["op_state_spec"]["weight"],
                 "excluded_types": sorted(linear_quant_config("int8")["module_type_configs"]),
                 "lm_head": head_quant_spec("block32", True)["op_state_spec"]["weight"],
                 "lm_head_untied_in_checkpoint": True, "lm_head_cloned": False,
                 "int8_linear_modules": len(q8), "fp16_linear_modules": fp,
                 "lm_head_int8": "lm_head" in q8, "embed_tokens_quantized": bool(emb_q),
                 "conv1d_quantized": conv_q, "seconds": time.monotonic() - t_loaded}
        print(f"[quant] int8 linears {len(q8)} (lm_head {'in' if 'lm_head' in q8 else 'NOT in'}), fp16 linears "
              f"{len(fp)} {fp[:4]}, embed quantized {bool(emb_q)}, conv1d quantized {len(conv_q)}, "
              f"{quant['seconds']:.0f}s", flush=True)
        if "lm_head" not in q8 or fp or emb_q or conv_q:
            sys.exit(f"int8hu: the quantized set differs from the recipe: {json.dumps(quant)}")
    t_quantized = time.monotonic()

    # The loop-free path never calls the GatedDeltaUpdate composite; externalizing the class would
    # mark the uncalled submodules and then fail to find them in the traced program.
    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    if chunk:
        from export_qwen38vl_pipelined import _install_externalize_dim_retry

        _install_externalize_dim_retry()
        print(f"[export] multifunction decoder (main S=1 + prefill S={chunk}) ...", flush=True)
        prog = export_to_coreai_multifunction(
            model, [("main", spec), ("prefill", spec_pf)], externalize_modules=specs)
    else:
        print("[export] S=1 decode graph ...", flush=True)
        prog = export_to_coreai(
            model, spec["reference_inputs"], dynamic_shapes=spec["dynamic_shapes"],
            input_names=spec["input_names"], output_names=spec["output_names"],
            state_names=spec["state_names"], externalize_modules=specs)
    print("[export] optimizing ...", flush=True)
    prog.optimize()
    t_exported = time.monotonic()

    import coreai.runtime as rt

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aimodel = out_dir / f"{name}.aimodel"
    md = rt.AIModelAssetMetadata()   # constructor kwargs are dropped: set attributes
    md.author = "openbmb (weights); community Core AI port"
    md.license = "Apache-2.0"
    md.model_description = (f"MiniCPM-V-4.7-1B VL decoder, S=1 decode"
                            f"{f' + S={chunk} prefill' if chunk else ''}, canvas M-RoPE in-graph, {args.mode} "
                            f"(Community port - NOT an Apple model)")
    print(f"[save] {aimodel} ...", flush=True)
    prog.save_asset(aimodel, md)
    t_saved = time.monotonic()

    language_extra = {"tokenizer_dir": "tokenizer", "stop_token_ids": STOP_IDS,
                      "thinking": "off: the prompt ends '" + PROMPT_TAIL.replace("\n", "\\n") + "' "
                                  "(chat_template enable_thinking=False)",
                      "static_inputs": ["image_embeds", "rope_shift_start", "rope_shift_amount"],
                      "image_tokens_max": N_IMAGE}
    vlm = vlm_block(N_IMAGE, cfg.hidden_size, cfg.vocab_size)
    if chunk:
        language_extra["prefill_chunk"] = chunk
        vlm["prefill_order"] = (f"fresh zero states; while the prompt has a whole chunk left at cursor c, 'prefill' on "
                                f"input_ids[c : c + {chunk}] with position_ids 0..c+{chunk - 1} (logits = the chunk's last "
                                f"position), c += {chunk}; then 'main' one token at a time with position_ids 0..c; the "
                                f"same static inputs on both functions")
    write_bundle_metadata(
        out_dir, name, HF_ID, cfg.vocab_size, args.max_ctx, revision=REVISION, mode=args.mode,
        functions=("main", "prefill") if chunk else ("main",),
        language_extra=language_extra,
        extra={"vlm": vlm, "author_note": "Community port - NOT an Apple model"},
    )
    tdir = out_dir / "tokenizer"
    tdir.mkdir()
    tok = {}
    for fn in TOKENIZER_FILES:
        shutil.copyfile(snap / fn, tdir / fn)        # follows the cache symlink: the blob's bytes
        tok[fn] = {"bundle": sha256_file(tdir / fn), "snapshot": sha256_file(snap / fn)}
    tok_ok = all(v["bundle"] == v["snapshot"] for v in tok.values())
    if not tok_ok:
        sys.exit(f"tokenizer files differ from the snapshot: {json.dumps(tok)}")
    rec = {"bundle": str(out_dir), "name": name, "aimodel": str(aimodel), "load_report": report,
           "loopfree_linear_layers": n_lin, "trace_kv_len": TRACE_KV_CACHE_SEQ_LEN, "max_ctx": args.max_ctx,
           "functions": ["main", "prefill"] if chunk else ["main"], "prefill_chunk": chunk,
           "gdn_scan": "unroll" if chunk else "step", "last_token_only": bool(chunk), "quantization": quant,
           "seconds": {"load": t_loaded - t0, "quantize": t_quantized - t_loaded,
                       "export_optimize": t_exported - t_quantized, "save": t_saved - t_exported,
                       "total": time.monotonic() - t0},
           "aimodel_digest": tree_digest(aimodel),
           "metadata_sha256": sha256_file(out_dir / "metadata.json"),
           "tokenizer_sha256": tok, "tokenizer_verbatim_from_snapshot": tok_ok}
    print(f"[done] {out_dir}: .aimodel {rec['aimodel_digest']['bytes']:,} B tree sha256 "
          f"{rec['aimodel_digest']['tree_sha256']} ({rec['seconds']['total']:.0f}s)", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["fp16", "int8hu"])
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--prefill-chunk", type=int, default=None,
                    help="also export a static-S 'prefill' function (S = this, e.g. 16) next to 'main' in one "
                         "multifunction asset; the name gets _pf<S>")
    ap.add_argument("--out-root", default=str(exports_dir()))
    ap.add_argument("--record", help="write the export record JSON here")
    args = ap.parse_args()
    if args.prefill_chunk is not None and args.prefill_chunk < 2:
        ap.error("--prefill-chunk must be >= 2 (S=1 is 'main')")
    name = f"minicpmv47_vlm_decode_{args.mode}" + (f"_pf{args.prefill_chunk}" if args.prefill_chunk else "")
    out_dir = Path(args.out_root) / name
    import coreai_models
    import torch
    record = {"schema": "minicpmv47-export-decoder/1", "mode": args.mode, "name": name, "hf_id": HF_ID,
              "revision": REVISION, "export": export(args, out_dir, name),
              "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                           "coreai_models": getattr(coreai_models, "__version__", None), "platform": platform.platform()},
              "script": {"path": "conversion/minicpmv47/export_decoder.py", "sha256": sha256_file(Path(__file__))},
              "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    import importlib.metadata as md
    for p in ("coreai-core", "coreai-torch", "coreai-opt"):
        record["versions"][p] = md.version(p)
    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        Path(args.record).write_text(json.dumps(record, indent=1) + "\n")
        print(f"[record] {args.record}", flush=True)


if __name__ == "__main__":
    main()
