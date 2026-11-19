#!/usr/bin/env python3
"""Export the clef-flash decoder (final-norm hidden at every position) to a Core AI bundle, one static-S function, and AOT-compile it.

The graph is `qwen3_5_clef_decoder.Qwen3_5ClefDecoder` — decider-2b-vision's ids-input Qwen3.5 VL
decoder on the 9B config with no vocabulary head:

    input_ids [1,S] i32 (static), position_ids [1,seq] i32 (dynamic), image_embeds [1024,4096],
    image_rc [1024,2] i32, rope_shift_start [1] i32, rope_shift_amount [1] i32
    + keyCache / valueCache (dynamic sequence dim) / convState / recState -> hidden [1,S,4096]

One function, `main` (export_to_coreai's default entry point), at static S = `--prefill-chunk`
(16 by default; there is no S=1 function: the head reads every position of one prefill). Every
linear-attention layer runs `use_loopfree_unroll` (the overlay's `_gated_delta_step_unroll`: the S
single steps unrolled in-graph, fp32 inside the call, no doubling inverse), gated_delta_update is
left out of the externalized composites (the loop-free path never calls it), and the static-S
causal SDPA's externalize guard is retried with torch's suggested bounds
(`export_qwen38vl_pipelined._install_externalize_dim_retry`). Modes:

    fp16               the reference
    int8lin            every decoder linear int8 per block of 32 (symmetric_with_clipping, weight
                       only); the embedding table, conv1d and norms stay fp16 (there is no lm_head
                       in this graph: the untied table stays on the host)
    int8mix --fp16-layers I,J,..
                       int8lin with every linear of decoder layers I, J, .. left fp16 (excluded by
                       name); the layers go into the bundle metadata (`compression.fp16_layers`)
    fp16 --variant fp16attn32
                       a diagnostic, not a ship form: the q/k/v/o linears and the SDPA of the 8
                       full-attention layers compute in fp32 (`ATTN_FP32_HOW`, metadata `variant`)

Both int8 modes write `compression` (scheme, linear spec, what stays fp16, the fp16 layers) into
the bundle metadata.

The bundle is `<out-dir>/bundles/<name>/` = `<name>.aimodel` + `metadata.json` + `tokenizer/`, with
`<name>` = `clef_flash_decode_<mode>_pf<S>`. `metadata.json` is `_bundle.write_bundle_metadata`'s
with `kind` rewritten to `decision-backbone` (the graph returns hidden states, not logits) and two
top-level blocks: `decision` (how a host turns the hidden rows into typed decisions: the prompt
layout, the spans, the author's head files, the lexical table, the per-question softmax, the
SystemOne-compatible response shape) and `vision` (how a host feeds an image). `tokenizer/` is the
pinned snapshot's tokenizer files, copied verbatim (sha256 checked).

`--aot` compiles the `.aimodel` ahead of time for the Mac GPU into
`<out-dir>/bundles_aotc/<name>.h16c.aimodelc` (`coreai-build compile --platform macOS
--preferred-compute gpu --architecture h16c --expect-frequent-reshapes`; the Python runtime's JIT
mis-executes hybrids above 0.8B). The gate is `readout_gate.py`.

    cd conversion/clef_flash
    HF_HOME=$ZOO_WORK_ROOT/_clefflash/hf HF_HUB_OFFLINE=1 \\
      DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer \\
      <coreai-models venv>/bin/python export_decoder.py fp16 --prefill-chunk 16 --aot --record <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _paths import hf_snapshot, record_argv, work_path  # noqa: E402

os.environ.setdefault("HF_HOME", str(work_path("_clefflash", "hf")))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

HF_ID = "Cloudflare/clef-flash"
REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"
JSM_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
VISION_START, VISION_END, IMAGE_PAD, PAD_ID = 248053, 248054, 248056, 248044
PREFIX_TOKENS, SUFFIX_TOKENS = 36, 18
TOWER_GRIDS = (("g256", 8), ("g448", 14))
AOT_FLAGS = ["--platform", "macOS", "--preferred-compute", "gpu", "--architecture", "h16c",
             "--expect-frequent-reshapes"]
DISK = "/System/Volumes/Data"


def du(path: Path) -> str:
    return subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True).stdout.split()[0]


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
    return {"bytes": sum(p.stat().st_size for p in files), "tree_sha256": tree, "files": per}


def disk_free() -> dict:
    """Free space on the data volume and the two caches the export and the compiler grow."""
    import glob

    free = shutil.disk_usage(DISK).free
    build = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    cc = Path.home() / "Library/Caches/coreai-cache" / build / "python"
    scratch = sorted(glob.glob("/private/var/folders/*/*/T/com.apple.MetalPerformanceShadersGraph"))
    caches = {"coreai_cache_python": du(cc) if cc.exists() else None,
              "mpsgraph_scratch": {p: du(Path(p)) for p in scratch}}
    return {"time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "free_bytes": free, "free_gib": round(free / 2**30, 1), "caches": caches}


def linear_quant_config(dtype: str = "int8") -> dict:
    """Weight-only linear int8 per-block-32, decider_vision/export_decoder.py's recipe: SDPA / RoPE /
    norms / Embedding / Conv1d excluded. This graph has no lm_head (the parent's slot is an
    Identity), so nothing is excluded by name."""
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
        "module_name_configs": {},
    }


ATTN_FP32_HOW = ("every full-attention layer's q_proj / k_proj / v_proj / o_proj and its SDPA run in fp32: the "
                 "linears keep fp16 weights with an fp32 view (export_qwen38vl_pipelined.fp16_storage_fp32_compute), "
                 "each wrapped module casts its tensor inputs to fp32 and its output back to fp16; the residual "
                 "stream, the norms, RoPE and the KV cache stay fp16 (the SDPA reads the fp16 cache, cast up)")


def attn_fp32_compute(model) -> list[str]:
    """--variant fp16attn32 (a diagnostic, not a ship form): see ATTN_FP32_HOW. Returns the wrapped
    module names."""
    import torch
    from export_qwen38vl_pipelined import fp16_storage_fp32_compute

    class Fp32Compute(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module):
            super().__init__()
            self.inner = fp16_storage_fp32_compute(inner)

        def forward(self, *xs: torch.Tensor) -> torch.Tensor:
            return self.inner(*(x.float() for x in xs)).half()

    wrapped = []
    for i, layer in enumerate(model.model.layers):
        if not layer.is_full:
            continue
        attn = layer.self_attn
        for n in ("q_proj", "k_proj", "v_proj", "o_proj", "sdpa"):
            setattr(attn, n, Fp32Compute(getattr(attn, n)))
            wrapped.append(f"model.layers.{i}.self_attn.{n}")
    return wrapped


def readout_text(chunk: int) -> str:
    c = chunk
    return (f"fresh zero states; a row of T ids runs as ceil(T / {c}) calls of 'main' (static S = {c}): call k "
            f"gets ids[{c}k : {c}k + {c}] with position_ids 0..{c}k+{c - 1}; the last call is padded with "
            f"<|endoftext|> ({PAD_ID}) and the hidden rows of the padded positions are discarded (causal: "
            f"they cannot reach a real position). The hidden rows [1, {c}, 4096] of every call, concatenated "
            f"and cut to T, are the backbone's final-norm last_hidden_state [T, 4096] the head reads.")


def bundle_extra(n_image_max: int, vocab: int, chunk: int) -> dict:
    """Top-level metadata blocks: how a host turns the hidden rows into decisions, and how it feeds an image."""
    return {
        "decision": {
            "output": "hidden [1, S, 4096] per call: the final-norm hidden state at every position (no "
                      "vocabulary head in the graph)",
            "readout": readout_text(chunk),
            "prompt": {
                "source": "the author's encode_record() (joint_schema_model.py at the pinned revision, sha256 "
                          f"{JSM_SHA256}); every piece tokenized on its own without special tokens, then "
                          "concatenated (conversion/clef_flash/host.py build_ids)",
                "layout": "prefix + [image block] + state + schema + suffix",
                "prefix": "<|im_start|>system\\n{system prompt}<|im_end|>\\n<|im_start|>user\\nSTATE:\\n",
                "prefix_tokens": PREFIX_TOKENS,
                "system_prompt": "Read the complete state and schema. Decide every field jointly. Each answer "
                                 "must be exactly one of that field's allowed options.",
                "image_block": f"at index {PREFIX_TOKENS} (right after the prefix): <|vision_start|> "
                               f"({VISION_START}), N <|image_pad|> ({IMAGE_PAD}) sent to the graph as ids V+k "
                               f"(k = 0..N-1 row-major over the merged grid, V = {vocab}), <|vision_end|> "
                               f"({VISION_END}), '\\n'; N + 3 ids",
                "state": "render(state): a string as is, anything else compact JSON (sorted keys, ensure_ascii "
                         "off); cut so the whole row fits max_length",
                "schema": "'\\n\\nSCHEMA FIELDS:\\n', then per question i (1-based) '\\nFIELD {i}\\nID: {id}\\nTYPE: "
                          "{type}\\nINSTRUCTION: ' + render(instructions or id) + '\\nALLOWED OPTIONS:\\n' + per "
                          "option j (1-based) 'OPTION {j}: ' + render({option_id, description}) + '\\n', then "
                          "'END FIELD\\n'",
                "suffix": "\\n<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\nJOINT SCHEMA "
                          "DECISIONS:",
                "suffix_tokens": SUFFIX_TOKENS,
                "last_token": "':' (25)",
                "add_special_tokens": False,
                "chat_template": None,
                "max_length_author": 16384,
                "max_tokens_this_graph": "max_context_length (the row and its last padded call must fit)",
            },
            "spans": {
                "question": "[start, end) of the render(instructions or id) tokens of the question",
                "option": "[start, end) of the render({option_id, description}) tokens of each option",
                "ids": "spans index the processor-form ids (N x <|image_pad|>), the same positions as the graph's",
            },
            "option_order": {"noul": ["true", "false"], "choice": "option ids sorted as strings",
                             "score": "levels 0..n-1"},
            "question_types": {"noul": 0, "choice": 1, "score": 2},
            "noul_default_criteria": {"true": "The proposition is true or the answer is yes.",
                                      "false": "The proposition is false or the answer is no."},
            "head": {
                "files": ["joint_head.safetensors", "joint_head_config.json"],
                "code": f"JointSchemaHead in the checkpoint's joint_schema_model.py (sha256 {JSM_SHA256})",
                "reads": "last_hidden_state [T, 4096] at every position; LayerNorm(4096) -> memory; the global "
                         "vector = the last token; question / option vectors = the mean of the LayerNorm-ed "
                         "rows over each span",
                "lexical": "lm_head.weight rows (untied [248320, 4096], model-00001-of-00004.safetensors) "
                           "averaged over each option span's ids; the table is not in this graph",
                "dtype_of_record": "fp32 (the gate reads the graph's fp16 hidden through the fp32 head)",
            },
            "softmax": "per question, fp32, over that question's options (temperature 1)",
            "response": {
                "shape": "SystemOne-compatible: {model, answers: {question_id: answer}, usage: {input_tokens: T, "
                         "output_tokens: 0}}",
                "noul": "{type: 'noul', noul: p(true)}",
                "choice": "{type: 'choice', choice: argmax option id, confidence: its p, probabilities: {id: p}}",
                "score": "{type: 'score', score: sum(level * p), confidence: max p, legend: {level: "
                         "criterion}, probabilities: {level: p}}",
                "rounding": "4 decimals",
            },
        },
        "vision": {
            "n_image_max": n_image_max,
            "image_token_base": vocab,
            "image_embeds": "tower rows 0..N-1 (N = H*W merged tokens) as fp16, rows N.. zero",
            "image_rc": "image_rc[k] = (k // W, k % W) for k < N, rows N.. zero",
            "rope_shift_start": f"{PREFIX_TOKENS + 1} + H*W (the <|vision_end|> index; the image block starts "
                                f"at {PREFIX_TOKENS})",
            "rope_shift_amount": "H*W - max(H, W)",
            "text_only": {"rope_shift_start": 1 << 30, "rope_shift_amount": 0,
                          "image_embeds": "zero", "image_rc": "zero"},
            "towers": [{"name": f"clef_flash_{arm}_vision_fp16w32", "merged_grid": [g, g],
                        "tile": 32 * g, "patches": [4 * g * g, 1536], "image_embeds": [g * g, 4096]}
                       for arm, g in TOWER_GRIDS],
            "host_preprocess": "RGB -> Pillow BICUBIC resize to (32*grid) x (32*grid), aspect not kept -> /255 -> "
                               "(x - 0.5) / 0.5 -> merge-block-major patchify, the frame repeated at both "
                               "temporal slots (conversion/clef_flash/host.py preprocess)",
        },
    }


def export(args, out_dir: Path, name: str) -> dict:
    import torch
    from _bundle import save_tokenizer, write_bundle_metadata
    from qwen3_5_clef_decoder import INTENTIONALLY_UNREAD, OUTPUT_NAMES, Qwen3_5ClefDecoder

    from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
    from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai

    dtype = torch.float16
    chunk = args.prefill_chunk
    disk = {"before_load": disk_free()}
    t0 = time.monotonic()
    print(f"loading {args.hf_id} text decoder fp16 (no lm_head) ...", flush=True)
    model = Qwen3_5ClefDecoder.from_hf(args.hf_id, target_dtype=dtype, max_context_length=args.max_ctx,
                                       n_image_max=args.n_image_max)
    report = model.load_report
    if (report["unread_checkpoint_keys"] or report["module_tensors_not_in_checkpoint"]
            or report["intentionally_unread_keys"] != list(INTENTIONALLY_UNREAD)
            or report["module_has_lm_head_weight"]):
        sys.exit(f"load mismatch: {json.dumps(report)}")
    n_lin = 0
    for layer in model.model.layers:
        if not layer.is_full:
            layer.linear_attn.use_loopfree_step = True
            layer.linear_attn.use_loopfree_unroll = True   # the traced S is a static int: S steps unrolled
            n_lin += 1
    print(f"unrolled step scan on {n_lin} linear layers; load {json.dumps(report)}", flush=True)
    cfg = model.config
    spec = model.build_export_spec(dtype, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, query_len=chunk)
    if tuple(spec["output_names"]) != tuple(OUTPUT_NAMES):
        sys.exit(f"export spec output names {spec['output_names']} != {OUTPUT_NAMES}")
    t_loaded = time.monotonic()
    disk["after_load"] = disk_free()

    variant = None
    if args.variant == "fp16attn32":
        variant = {"name": "fp16attn32", "fp32_compute_modules": attn_fp32_compute(model),
                   "how": ATTN_FP32_HOW}
        print(f"fp16attn32: {len(variant['fp32_compute_modules'])} modules compute in fp32", flush=True)

    quant: dict | None = None
    if args.mode == "int8lin":
        import torch.nn.utils.parametrize as P
        from coreai_models.export.compression import quantize_pytorch_model

        cfg_q = linear_quant_config("int8")
        print("quantizing (linear int8 per-block-32; embedding, conv1d, norms fp16) ...", flush=True)
        model = quantize_pytorch_model(
            model, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"], cfg_q)
        lin = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
        int8 = [n for n, m in lin if P.is_parametrized(m, "weight")]
        fp16 = [n for n, m in lin if not P.is_parametrized(m, "weight")]
        other = sorted({type(m).__name__ for n, m in model.named_modules()
                        if P.is_parametrized(m) and not isinstance(m, torch.nn.Linear)})
        if fp16 or other or not int8 or any(not n.startswith("model.layers.") for n in int8):
            sys.exit(f"int8lin: not every decoder linear (and nothing else) is int8: fp16 {fp16[:8]}, "
                     f"other quantized types {other}")
        # The quantizer rewrites the config it was given (dtype strings become torch dtypes): record a
        # fresh copy.
        quant = {"linear": linear_quant_config("int8")["global_config"]["op_state_spec"]["weight"],
                 "excluded_types": sorted(linear_quant_config("int8")["module_type_configs"]),
                 "fp16_layers": [], "int8_linear_modules": len(int8), "fp16_linear_modules": fp16,
                 "int8_params": int(sum(next(p for p in m.parametrizations["weight"]
                                             if hasattr(p, "quantized_data")).quantized_data.numel()
                                        for n, m in lin if n in set(int8))),
                 "lm_head": "none in the graph (the untied table stays on the host)",
                 "seconds": time.monotonic() - t_loaded}
        print(f"quantized in {quant['seconds']:.0f}s: {len(int8)} int8 linears ({quant['int8_params']:,} params)",
              flush=True)
    elif args.mode == "int8mix":
        import torch.nn.utils.parametrize as P
        from coreai_models.export.compression import quantize_pytorch_model

        keep = sorted(set(args.fp16_layers))
        cfg_q = linear_quant_config("int8")
        for i in keep:  # fullmatch on the module name, applied to every linear of the layer
            cfg_q["module_name_configs"][rf"model\.layers\.{i}\..*"] = None
        print(f"quantizing (linear int8 per-block-32, layers {keep} fp16) ...", flush=True)
        model = quantize_pytorch_model(
            model, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"], cfg_q)
        body = [(n, m) for n, m in model.named_modules()
                if isinstance(m, torch.nn.Linear) and n.startswith("model.layers.")]
        int8 = [n for n, m in body if P.is_parametrized(m, "weight")]
        fp16 = [n for n, m in body if not P.is_parametrized(m, "weight")]
        wrong = [n for n in int8 if int(n.split(".")[2]) in keep] + \
                [n for n in fp16 if int(n.split(".")[2]) not in keep]
        if wrong or not fp16:
            sys.exit(f"int8mix: quantized set differs from the request (layers {keep}): {wrong[:8]}")
        quant = {"linear": linear_quant_config("int8")["global_config"]["op_state_spec"]["weight"],
                 "excluded_types": sorted(linear_quant_config("int8")["module_type_configs"]),
                 "fp16_layers": keep, "fp16_layer_patterns": [rf"model\.layers\.{i}\..*" for i in keep],
                 "int8_linear_modules": len(int8), "fp16_linear_modules": fp16,
                 "int8_params": int(sum(next(p for p in m.parametrizations["weight"]
                                             if hasattr(p, "quantized_data")).quantized_data.numel()
                                        for n, m in body if n in set(int8))),
                 "fp16_params": int(sum(m.weight.numel() for n, m in body if n in set(fp16))),
                 "lm_head": "none in the graph (the untied table stays on the host)",
                 "seconds": time.monotonic() - t_loaded}
        print(f"quantized in {quant['seconds']:.0f}s: {len(int8)} int8 linears, {len(fp16)} fp16 "
              f"({quant['fp16_params']:,} params)", flush=True)
    t_quantized = time.monotonic()

    # The loop-free path never calls the GatedDeltaUpdate composite; externalizing the class
    # would mark the uncalled submodules and then fail to find them in the traced program.
    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    from export_qwen38vl_pipelined import _install_externalize_dim_retry

    _install_externalize_dim_retry()
    print(f"exporting the hidden-output decoder (one function 'main', static S={chunk}) ...", flush=True)
    prog = export_to_coreai(
        model,
        spec["reference_inputs"],
        dynamic_shapes=spec["dynamic_shapes"],
        input_names=spec["input_names"],
        output_names=spec["output_names"],
        state_names=spec["state_names"],
        externalize_modules=specs,
    )
    t_converted = time.monotonic()
    print(f"converted in {t_converted - t_quantized:.0f}s; optimizing ...", flush=True)
    prog.optimize()
    t_exported = time.monotonic()
    print(f"optimized in {t_exported - t_converted:.0f}s", flush=True)
    disk["after_export"] = disk_free()

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    import coreai.runtime as rt

    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    t_saved = time.monotonic()
    disk["after_save"] = disk_free()
    language_extra = {"prefill_chunk": chunk,
                      "static_inputs": ["image_embeds", "image_rc", "rope_shift_start", "rope_shift_amount"],
                      "image_tokens_max": args.n_image_max,
                      "output": f"hidden [1, {chunk}, {cfg.hidden_size}] fp16, every position"}
    extra = bundle_extra(args.n_image_max, cfg.vocab_size, chunk)
    if args.mode in ("int8lin", "int8mix"):
        extra["compression"] = {
            "scheme": args.mode,
            "linear": "int8 per-block-32 symmetric_with_clipping (weight only)",
            "excluded": "fp16: the embedding table, the GDN conv1d, every RMSNorm (RMSNorm, RMSNormPlusOne, "
                        "RMSNormGated), SDPA, RoPE, the GDN A_log / dt_bias",
            "excluded_types": quant["excluded_types"],
            "int8_linear_modules": quant["int8_linear_modules"],
            "fp16_layers": quant["fp16_layers"],
            "fp16_linear_modules": quant["fp16_linear_modules"],
            "head": "none in the graph (the untied lm_head table stays on the host)",
        }
    if variant:
        extra["variant"] = variant
    write_bundle_metadata(
        out_dir, name, args.hf_id, cfg.vocab_size, args.max_ctx, revision=args.revision, mode=args.mode,
        functions=("main",), language_extra=language_extra, extra=extra,
    )
    # The graph returns hidden states, not logits: a generation loop must not pick this bundle up as an llm.
    meta_path = out_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["kind"] = "decision-backbone"
    meta_path.write_text(json.dumps(meta, indent=2))
    # Verbatim copy of the pinned snapshot's tokenizer files (the ids were gated on these bytes).
    save_tokenizer(args.hf_id, out_dir, via_transformers=False)
    snap = Path(hf_snapshot(args.hf_id, revision=args.revision))
    tok = {f.name: sha256_file(f) for f in sorted((out_dir / "tokenizer").iterdir())}
    tok_verbatim = all(sha256_file(snap / n) == h for n, h in tok.items())
    mlirb = aimodel / "main.mlirb"
    rec = {"bundle": str(out_dir), "name": name, "aimodel": str(aimodel), "load_report": report,
           "loopfree_unrolled_linear_layers": n_lin, "trace_kv_len": TRACE_KV_CACHE_SEQ_LEN,
           "max_ctx": args.max_ctx, "n_image_max": args.n_image_max,
           "functions": ["main"], "query_len": chunk, "gdn_scan": "unroll", "output_names": list(OUTPUT_NAMES),
           "quantization": quant, "variant": variant,
           "seconds": {"load": t_loaded - t0, "quantize": t_quantized - t_loaded,
                       "export": t_converted - t_quantized, "optimize": t_exported - t_converted,
                       "export_optimize": t_exported - t_quantized, "save": t_saved - t_exported,
                       "total": time.monotonic() - t0},
           "du_aimodel": du(aimodel), "du_bundle": du(out_dir),
           "aimodel_files": {str(p.relative_to(aimodel)): p.stat().st_size for p in sorted(aimodel.rglob("*"))
                             if p.is_file()},
           "main_mlirb": {"bytes": mlirb.stat().st_size, "sha256": sha256_file(mlirb)},
           "metadata_sha256": sha256_file(meta_path), "kind": meta["kind"],
           "tokenizer_sha256": tok, "tokenizer_verbatim_from_snapshot": tok_verbatim, "disk": disk}
    if not tok_verbatim:
        sys.exit(f"tokenizer files differ from the snapshot {snap}: {json.dumps(tok)}")
    print(f"bundle ready: {out_dir} ({rec['du_aimodel']}, main.mlirb {rec['main_mlirb']['bytes']:,} B sha256 "
          f"{rec['main_mlirb']['sha256']}, total {rec['seconds']['total']:.0f}s)", flush=True)
    return rec


def aot_compile(aimodel: Path, out_dir: Path) -> tuple[Path, float, dict]:
    target = out_dir / f"{aimodel.stem}.h16c.aimodelc"
    if not os.environ.get("DEVELOPER_DIR"):
        sys.exit("set DEVELOPER_DIR to the Xcode 27 RC (its Metal toolchain carries coreai-build)")
    cb = subprocess.run(["xcrun", "-f", "coreai-build"], capture_output=True, text=True)
    if cb.returncode != 0 or not cb.stdout.strip():
        sys.exit("xcrun -f coreai-build failed:\n" + cb.stderr)
    if target.exists():
        shutil.rmtree(target)
    out_dir.mkdir(parents=True, exist_ok=True)
    disk = {"before": disk_free()}
    cmd = [cb.stdout.strip(), "compile", str(aimodel), "--output", str(out_dir), *AOT_FLAGS]
    print(" ".join(cmd), flush=True)
    t0 = time.monotonic()
    subprocess.run(cmd, check=True)
    secs = time.monotonic() - t0
    disk["after"] = disk_free()
    if not target.exists():
        sys.exit(f"coreai-build produced no {target}")
    return target, secs, {"coreai_build": cb.stdout.strip(), "command": cmd, "disk": disk}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="fp16", choices=["fp16", "int8lin", "int8mix"])
    ap.add_argument("--fp16-layers", type=lambda s: [int(x) for x in s.split(",") if x != ""],
                    help="int8mix only: comma list of decoder layer indices whose linears stay fp16")
    ap.add_argument("--variant", choices=["fp16attn32"],
                    help="fp16 only, a diagnostic: fp16attn32 = the full-attention linears and SDPA in fp32 "
                         "(the name becomes clef_flash_decode_fp16attn32_pf<S>)")
    ap.add_argument("--hf-id", default=HF_ID)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--out-dir", default=str(work_path("_clefflash", "exports")),
                    help="bundles go to <out-dir>/bundles/<name>/, AOT assets to <out-dir>/bundles_aotc/")
    ap.add_argument("--name", help="override the generated bundle directory and asset name")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--n-image-max", type=int, default=1024)
    ap.add_argument("--prefill-chunk", type=int, default=16,
                    help="static S of the one function 'main' (the name gets _pf<S>); S=1 is not built")
    ap.add_argument("--skip-export", action="store_true", help="reuse the saved .aimodel")
    ap.add_argument("--aot", action="store_true", help="compile the .aimodel for the Mac GPU (h16c)")
    ap.add_argument("--record", help="write the export / AOT record JSON here")
    args = ap.parse_args()
    if (args.mode == "int8mix") != bool(args.fp16_layers):
        ap.error("--fp16-layers is required by int8mix and applies to it only")
    if args.prefill_chunk < 2:
        ap.error("--prefill-chunk must be >= 2 (this graph has no S=1 function)")
    if args.variant and args.mode != "fp16":
        ap.error("--variant applies to fp16 only")

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = args.name or f"{short}_decode_{args.variant or args.mode}_pf{args.prefill_chunk}"
    out_dir = Path(args.out_dir) / "bundles" / name
    aot_dir = Path(args.out_dir) / "bundles_aotc"
    record: dict = {"mode": args.mode, "name": name, "hf_id": args.hf_id, "revision": args.revision,
                    "prefill_chunk": args.prefill_chunk, "argv": record_argv(sys.argv[1:]), "pid": os.getpid(),
                    "started": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "script_sha256": sha256_file(Path(__file__).resolve()),
                    "module_sha256": sha256_file(HERE / "qwen3_5_clef_decoder.py"),
                    "parent_sha256": sha256_file(HERE.parent / "decider_vision" / "qwen3_5_vl_pipelined.py")}

    def save_record() -> None:
        if args.record:
            Path(args.record).parent.mkdir(parents=True, exist_ok=True)
            Path(args.record).write_text(json.dumps(record, indent=1) + "\n")

    if not args.skip_export:
        record["export"] = export(args, out_dir, name)
        save_record()
    if args.aot:
        aimodelc, secs, info = aot_compile(out_dir / f"{name}.aimodel", aot_dir)
        record["aot"] = {"aimodelc": str(aimodelc), "flags": AOT_FLAGS, "seconds": secs, **info,
                         "du_aimodelc": du(aimodelc), "digest": tree_digest(aimodelc)}
        print(f"asset: {aimodelc} (compile {secs:.1f} s, {record['aot']['du_aimodelc']}, "
              f"{record['aot']['digest']['bytes']:,} B, tree sha256 {record['aot']['digest']['tree_sha256']})",
              flush=True)
    record["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
    save_record()
    if args.record:
        print(f"record: {args.record}")


if __name__ == "__main__":
    main()
