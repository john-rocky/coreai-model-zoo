#!/usr/bin/env python3
"""Export the ids-input decider-2b-vision decoder to a Core AI LanguageBundle (S=1 decode, optionally + an S=16 prefill chunk) and smoke it.

The graph is `qwen3_5_vl_pipelined.Qwen3_5VLPipelinedForCausalLM` — the Qwen3.5 hybrid text
decoder with the embed table in the graph and the image riding static inputs:

    input_ids [1,1] i32 (static), position_ids [1,seq] i32 (dynamic), image_embeds [256,2048],
    image_rc [256,2] i32, rope_shift_start [1] i32, rope_shift_amount [1] i32
    + keyCache / valueCache (dynamic sequence dim) / convState / recState -> logits [1,1,248320]

Built the way `export_qwen3_5_decode_pipelined.py` builds the text decode bundle: every
linear-attention layer on the loop-free single step, static S=1 ids, dynamic position and KV
dims, gated_delta_update left out of the externalized composites. Three modes:

    fp16               the reference
    int8hu --head-sym  block-32 int8 linears (symmetric_with_clipping) plus a block-32 int8
                       vocabulary head cloned from the tied embedding (plain symmetric = absmax);
                       embeddings, conv1d and norms stay fp16, so the text-table gather and the
                       image rows are untouched
    int8lin            the same int8 body, and the head stays the tied fp16 embedding table
                       (`lm_head` excluded by name, nothing cloned) — int8hu minus the int8 head
    int8mix --fp16-layers I,J,..
                       int8lin with every linear of decoder layers I, J, .. left fp16 (excluded
                       by name, the head's mechanism); the layers go into the bundle metadata
                       (`compression.fp16_layers`). The layers come from `int8_bisect_torch.py`.

`--prefill-chunk S` (S = 16) makes the asset multifunction (`export_to_coreai_multifunction`,
constants shared): "main" = the S=1 decode graph above, "prefill" = static input_ids [1, S]
with the same static image inputs and the logits of the chunk's LAST position ([1, 1, V]).
Every linear-attention layer then runs `use_loopfree_unroll` (the overlay's
`_gated_delta_step_unroll`: S single steps unrolled in-graph, fp32 inside the call, no
doubling inverse; at S=1 the one unrolled step is the single-step formula), and the static-S
causal SDPA's externalize guard is retried with torch's suggested bounds
(`export_qwen38vl_pipelined._install_externalize_dim_retry`). The bundle name gets `_pf<S>`.

The bundle is `<out-dir>/bundles/<name>/` = `metadata.json` + `tokenizer/` + `<name>.aimodel`, with
`<name>` = `decider_2b_vision_decode_<mode>` (`..._int8hu_block32_sym` for the ship head), the
exporter convention. `metadata.json` carries the decision readout (`decision`) and the image
contract (`vision`) next to the language block; `tokenizer/` is the pinned snapshot's tokenizer
files, copied verbatim.

`--aot` compiles the `.aimodel` ahead of time for the Mac GPU into
`<out-dir>/bundles_aotc/<name>.h16c.aimodelc` (`coreai-build compile --platform macOS
--preferred-compute gpu --architecture h16c --expect-frequent-reshapes`; the Python runtime's JIT
mis-executes hybrids above 0.8B). `--smoke <row>:<arm>` (compiles too) loads the `.aimodelc` with
`SpecializationOptions.default()`, prints the function's input / output / state descriptors, and
runs one fixture row at S=1 from fresh zero states, reading the letter probabilities at every slot
next to the fp32 oracle's. A look ahead, not a gate — the gate is `readout_gate_vision.py`.
`--smoke-red` re-runs the row with the image rows zeroed and/or the (row, col) table swapped, to
show the compiled graph reads its image inputs.

    HF_HOME=~/code/coreai/_decider2bv/hf HF_HUB_OFFLINE=1 \\
      DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer \\
      python export_decoder.py int8hu --head-sym --aot --record <json>
    ... python export_decoder.py int8lin --prefill-chunk 16 --aot --record <json>
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
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _paths import hf_snapshot, work_path  # noqa: E402

os.environ.setdefault("HF_HOME", str(work_path("_decider2bv", "hf")))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

HF_ID = "Mapika/decider-2b-vision"
REVISION = "863e290863655f1d6b69324d77d09ac972d21609"
ORACLE = work_path("_decider2bv", "oracle")
VISION_START, VISION_END, IMAGE_PAD = 248053, 248054, 248056
LETTER_IDS = list(range(32, 42))
TOWER_GRIDS = (("g256", 8), ("g448", 14))
AOT_FLAGS = ["--platform", "macOS", "--preferred-compute", "gpu", "--architecture", "h16c",
             "--expect-frequent-reshapes"]


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


def linear_quant_config(dtype: str = "int8") -> dict:
    """Weight-only linear int8 per-block-32, `export_qwen3_5_decode_pipelined.py`'s recipe verbatim:
    SDPA / RoPE / norms / Embedding / Conv1d excluded, lm_head excluded by name here and given its
    own spec by int8hu."""
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


def readout_text(prefill_chunk: int | None) -> str:
    if not prefill_chunk:
        return ("fresh zero states, S=1 through the whole row; at every slot step, fp32 softmax "
                "(T=1) over logits[32 : 32 + number of options]")
    c = prefill_chunk
    return (f"fresh zero states; per slot s (ascending): while s - cursor + 1 >= {c}, 'prefill' on "
            f"ids[cursor : cursor + {c}] with position_ids 0..cursor+{c - 1} (its logits are the slot's "
            f"when cursor + {c - 1} == s), cursor += {c}; then 'main' one token at a time up to and "
            f"including s (the logits at cursor == s are the slot's). At a slot, fp32 softmax (T=1) over "
            f"logits[32 : 32 + number of options]. The last slot is the row's last token.")


def bundle_extra(n_image_max: int, vocab: int, prefill_chunk: int | None = None) -> dict:
    """Top-level metadata blocks: how a host reads a decision, and how it feeds an image."""
    return {
        "decision": {
            "letters": list("ABCDEFGHIJ"),
            "letter_ids": LETTER_IDS,
            "letter_form": "bare A..J, no leading space",
            "max_options": len(LETTER_IDS),
            "slot": {"token": 318, "rule": "token 318 (' (') preceded by 25 (':') with 15666 ('Answer') "
                                           "among the 5 tokens before it; one slot per question"},
            "temperature": 1.0,
            "readout": readout_text(prefill_chunk),
            "prompt": {
                "source": "the author's build() (decider/prompt.py at the pinned revision), no shuffling; "
                          "the text is decoded and re-encoded as VisionDecisionModel.prepare() does",
                "text": "'Context:\\n' + context (first 1536 tokens), then per question: "
                        "'\\n\\nQuestion{ k}: {text}\\nOptions:' + '\\n({letter}) {option}' per option + "
                        "'\\nAnswer{ k}: (' (' k' = the 1-based number, only when there are several questions)",
                "max_ctx_tokens": 1536,
                "add_special_tokens": False,
                "bos": None,
                "chat_template": None,
                "image_prefix": f"<|vision_start|> ({VISION_START}), N ids V+k (k = 0..N-1 row-major over the "
                                f"merged grid, V = {vocab}), <|vision_end|> ({VISION_END}), in front of the text",
            },
        },
        "vision": {
            "n_image_max": n_image_max,
            "image_token_base": vocab,
            "image_embeds": "tower rows 0..N-1 (N = H*W merged tokens) as fp16, rows N.. zero",
            "image_rc": "image_rc[k] = (k // W, k % W) for k < N, rows N.. zero",
            "rope_shift_start": "1 + H*W (the <|vision_end|> index)",
            "rope_shift_amount": "H*W - max(H, W)",
            "text_only": {"rope_shift_start": 1 << 30, "rope_shift_amount": 0,
                          "image_embeds": "zero", "image_rc": "zero"},
            "towers": [{"name": f"decider_2b_vision_{arm}_vision_fp16w32", "merged_grid": [g, g],
                        "tile": 32 * g, "patches": [4 * g * g, 1536], "image_embeds": [g * g, 2048]}
                       for arm, g in TOWER_GRIDS],
            "host_preprocess": "RGB -> Pillow BICUBIC resize to (32*grid) x (32*grid), aspect not kept -> "
                               "/255 -> (x - 0.5) / 0.5 -> merge-block-major patchify, the frame repeated at "
                               "both temporal slots (conversion/decider_vision/host.py preprocess)",
        },
    }


def export(args, out_dir: Path, name: str) -> dict:
    import torch
    from _bundle import head_quant_spec, save_tokenizer, write_bundle_metadata
    from qwen3_5_vl_pipelined import Qwen3_5VLPipelinedForCausalLM

    from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
    from coreai_models.export.macos import (
        _EXTERNALIZE_SPECS,
        export_to_coreai,
        export_to_coreai_multifunction,
    )

    dtype = torch.float16
    chunk = args.prefill_chunk
    t0 = time.monotonic()
    print(f"loading {args.hf_id} text decoder fp16 ...", flush=True)
    model = Qwen3_5VLPipelinedForCausalLM.from_hf(
        args.hf_id, target_dtype=dtype, max_context_length=args.max_ctx, n_image_max=args.n_image_max)
    report = model.load_report
    if report["unread_checkpoint_keys"] or report["module_tensors_not_in_checkpoint"]:
        sys.exit(f"load mismatch: {json.dumps(report)}")
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
    print(f"{gdn} on {n_lin} linear layers; load {json.dumps(report)}", flush=True)
    cfg = model.config
    spec = model.build_export_spec(dtype, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN)
    spec_pf = None
    if chunk:
        model.last_token_only = True  # prefill logits [1, 1, V]; a no-op slice at S=1
        spec_pf = model.build_export_spec(dtype, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN,
                                          query_len=chunk)
    t_loaded = time.monotonic()

    quant: dict | None = None
    if args.mode == "int8lin":
        from coreai_models.export.compression import quantize_pytorch_model

        cfg_q = linear_quant_config("int8")  # lm_head excluded by name: the tied fp16 table
        print("quantizing (linear int8 per-block-32, head = the tied fp16 embedding) ...", flush=True)
        model = quantize_pytorch_model(
            model, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"], cfg_q)
        tied = model.lm_head.weight is model.model.embed_tokens.weight
        if not tied:  # the size would show a second fp16 table; the numerics would not move
            print("WARNING int8lin: lm_head is no longer the embedding table after quantization", flush=True)
        # The quantizer rewrites the config it was given (dtype strings become torch dtypes): record a
        # fresh copy, as int8hu does.
        quant = {"linear": linear_quant_config("int8")["global_config"]["op_state_spec"]["weight"],
                 "excluded_types": sorted(linear_quant_config("int8")["module_type_configs"]),
                 "lm_head": None, "lm_head_untied": False,
                 "lm_head_is_embed_tokens_after_quantization": tied,
                 "seconds": time.monotonic() - t_loaded}
        print(f"quantized in {quant['seconds']:.0f}s", flush=True)
    elif args.mode == "int8mix":
        import torch.nn.utils.parametrize as P
        from coreai_models.export.compression import quantize_pytorch_model

        keep = sorted(set(args.fp16_layers))
        cfg_q = linear_quant_config("int8")
        for i in keep:  # fullmatch on the module name, applied to every linear of the layer
            cfg_q["module_name_configs"][rf"model\.layers\.{i}\..*"] = None
        print(f"quantizing (linear int8 per-block-32, layers {keep} fp16, head = the tied fp16 "
              f"embedding) ...", flush=True)
        model = quantize_pytorch_model(
            model, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"], cfg_q)
        tied = model.lm_head.weight is model.model.embed_tokens.weight
        if not tied:
            print("WARNING int8mix: lm_head is no longer the embedding table after quantization", flush=True)
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
                 "lm_head": None, "lm_head_untied": False,
                 "lm_head_is_embed_tokens_after_quantization": tied,
                 "seconds": time.monotonic() - t_loaded}
        print(f"quantized in {quant['seconds']:.0f}s: {len(int8)} int8 linears, {len(fp16)} fp16 "
              f"({quant['fp16_params']:,} params)", flush=True)
    elif args.mode == "int8hu":
        from coreai_models.export.compression import quantize_pytorch_model

        cfg_q = linear_quant_config("int8")
        cfg_q["module_name_configs"] = {r".*lm_head$": head_quant_spec(args.head_quant, args.head_sym)}
        # The eager quantizer silently skips a shared parameter: untie the head from the embed
        # table first, so the head is quantized and the table stays fp16.
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        print(f"quantizing (linear int8 per-block-32, head {args.head_quant} "
              f"{'symmetric' if args.head_sym else 'symmetric_with_clipping'}) ...", flush=True)
        model = quantize_pytorch_model(
            model, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"], cfg_q)
        quant = {"linear": linear_quant_config("int8")["global_config"]["op_state_spec"]["weight"],
                 "excluded_types": sorted(linear_quant_config("int8")["module_type_configs"]),
                 "lm_head": head_quant_spec(args.head_quant, args.head_sym)["op_state_spec"]["weight"],
                 "lm_head_untied": True, "seconds": time.monotonic() - t_loaded}
        print(f"quantized in {quant['seconds']:.0f}s", flush=True)
    t_quantized = time.monotonic()

    # The loop-free path never calls the GatedDeltaUpdate composite; externalizing the class
    # would mark the uncalled submodules and then fail to find them in the traced program.
    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    if chunk:
        from export_qwen38vl_pipelined import _install_externalize_dim_retry

        _install_externalize_dim_retry()
        print(f"exporting multifunction decoder (main S=1 + prefill S={chunk}) ...", flush=True)
        prog = export_to_coreai_multifunction(
            model, [("main", spec), ("prefill", spec_pf)], externalize_modules=specs)
    else:
        print("exporting S=1 decode graph ...", flush=True)
        prog = export_to_coreai(
            model,
            spec["reference_inputs"],
            dynamic_shapes=spec["dynamic_shapes"],
            input_names=spec["input_names"],
            output_names=spec["output_names"],
            state_names=spec["state_names"],
            externalize_modules=specs,
        )
    print("optimizing ...", flush=True)
    prog.optimize()
    t_exported = time.monotonic()

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    import coreai.runtime as rt

    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    t_saved = time.monotonic()
    language_extra = {"image_tokens_max": args.n_image_max,
                      "static_inputs": ["image_embeds", "image_rc", "rope_shift_start", "rope_shift_amount"]}
    if chunk:
        language_extra["prefill_chunk"] = chunk
    extra = bundle_extra(args.n_image_max, cfg.vocab_size, chunk)
    if args.mode == "int8mix":
        extra["compression"] = {
            "scheme": "int8mix",
            "linear": "int8 per-block-32 symmetric_with_clipping (weight only)",
            "fp16_layers": quant["fp16_layers"],
            "fp16_linear_modules": quant["fp16_linear_modules"],
            "head": "tied fp16 embedding table",
        }
    write_bundle_metadata(
        out_dir, name, args.hf_id, cfg.vocab_size, args.max_ctx, revision=args.revision, mode=args.mode,
        functions=("main", "prefill") if chunk else ("main",),
        language_extra=language_extra,
        extra=extra,
    )
    # Verbatim copy of the pinned snapshot's tokenizer files (the ids were gated on these bytes).
    save_tokenizer(args.hf_id, out_dir, via_transformers=False)
    snap = Path(hf_snapshot(args.hf_id, revision=args.revision))
    tok = {f.name: sha256_file(f) for f in sorted((out_dir / "tokenizer").iterdir())}
    tok_verbatim = all(sha256_file(snap / n) == h for n, h in tok.items())
    mlirb = aimodel / "main.mlirb"
    rec = {"bundle": str(out_dir), "name": name, "aimodel": str(aimodel), "load_report": report,
           "loopfree_linear_layers": n_lin, "trace_kv_len": TRACE_KV_CACHE_SEQ_LEN, "max_ctx": args.max_ctx,
           "functions": ["main", "prefill"] if chunk else ["main"], "prefill_chunk": chunk,
           "gdn_scan": "unroll" if chunk else "step", "last_token_only": bool(chunk),
           "quantization": quant,
           "seconds": {"load": t_loaded - t0, "quantize": t_quantized - t_loaded,
                       "export_optimize": t_exported - t_quantized, "save": t_saved - t_exported,
                       "total": time.monotonic() - t0},
           "du_aimodel": du(aimodel), "du_bundle": du(out_dir),
           "main_mlirb": {"bytes": mlirb.stat().st_size, "sha256": sha256_file(mlirb)},
           "tokenizer_sha256": tok, "tokenizer_verbatim_from_snapshot": tok_verbatim}
    if not tok_verbatim:
        sys.exit(f"tokenizer files differ from the snapshot {snap}: {json.dumps(tok)}")
    print(f"bundle ready: {out_dir} ({rec['du_aimodel']}, main.mlirb sha256 {rec['main_mlirb']['sha256']}, "
          f"total {rec['seconds']['total']:.0f}s)", flush=True)
    return rec


def aot_compile(aimodel: Path, out_dir: Path) -> tuple[Path, float]:
    target = out_dir / f"{aimodel.stem}.h16c.aimodelc"
    if not os.environ.get("DEVELOPER_DIR"):
        sys.exit("set DEVELOPER_DIR to the Xcode 27 RC (its Metal toolchain carries coreai-build)")
    cb = subprocess.run(["xcrun", "-f", "coreai-build"], capture_output=True, text=True)
    if cb.returncode != 0 or not cb.stdout.strip():
        sys.exit("xcrun -f coreai-build failed:\n" + cb.stderr)
    if target.exists():
        shutil.rmtree(target)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    subprocess.run([cb.stdout.strip(), "compile", str(aimodel), "--output", str(out_dir), *AOT_FLAGS],
                   check=True)
    if not target.exists():
        sys.exit(f"coreai-build produced no {target}")
    return target, time.monotonic() - t0


def smoke(args, aimodelc: Path) -> dict:
    import asyncio

    import numpy as np
    import torch
    import coreai.runtime as rt
    from huggingface_hub import snapshot_download
    from qwen3_5_vl_pipelined import host_static_inputs
    from transformers import AutoConfig

    from coreai_models.models.macos import qwen3_5 as q

    rid, arm = args.smoke.split(":")
    fx = json.loads((ORACLE / "fixture_oracle.json").read_text())
    row = next(r for r in fx["rows"] if r["id"] == rid and r["arm"] == arm)
    npz = np.load(ORACLE / "npz" / f"{rid}__{arm}.npz")

    raw = AutoConfig.from_pretrained(snapshot_download(args.hf_id, allow_patterns=["config.json"]))
    cfg = q.qwen3_5_config_from_hf(raw.text_config, args.max_ctx, None)
    g = row["grid_thw"]
    hw = None if g is None else (g[1] // 2, g[2] // 2)
    ids, rc, start, amount = host_static_inputs(row["ids"], hw, cfg.vocab_size, IMAGE_PAD, VISION_START,
                                                args.n_image_max)
    emb = np.zeros((args.n_image_max, cfg.hidden_size), np.float16)
    if hw is not None:
        emb[: hw[0] * hw[1]] = npz["image_embeds"].astype(np.float16)

    def nd(a):
        return rt.NDArray(np.ascontiguousarray(a))

    def dsc(d):
        return {"shape": [int(x) for x in d.shape], "dtype": str(d.dtype).split(".")[-1]}

    out: dict = {"row": rid, "arm": arm, "asset": str(aimodelc),
                 "host": {"start": int(start[0]), "amount": int(amount[0]), "grid": hw}}

    async def run():
        t0 = time.monotonic()
        model = await rt.AIModel.load(aimodelc, rt.SpecializationOptions.default())
        fn = model.load_function("main")
        out["load_seconds"] = time.monotonic() - t0
        d = fn.desc
        out["descriptor"] = {
            "function": d.name,
            "inputs": {n: dsc(d.input_descriptor(n)) for n in d.input_names},
            "outputs": {n: dsc(d.output_descriptor(n)) for n in d.output_names},
            "states": {n: dsc(d.state_descriptor(n)) for n in d.state_names},
        }
        print(json.dumps(out["descriptor"], indent=1), flush=True)
        async def run_row(emb_in, rc_in):
            st = q.build_decode_state(cfg, max_seq_len=args.max_ctx, dtype=torch.float16)
            state = {n: nd(st[k].numpy()) for n, k in
                     zip(q.DECODE_STATE_NAMES, ("k_cache", "v_cache", "conv_state", "rec_state"))}
            static = {"image_embeds": nd(emb_in), "image_rc": nd(rc_in),
                      "rope_shift_start": nd(start.numpy()), "rope_shift_amount": nd(amount.numpy())}
            read = {}
            t1 = time.monotonic()
            for t in range(len(ids)):
                o = await fn(inputs={"input_ids": nd(np.array([[int(ids[t])]], np.int32)),
                                     "position_ids": nd(np.arange(t + 1, dtype=np.int32)[None]), **static},
                             state=state)
                if t in row["slot_idx"]:
                    read[t] = o["logits"].numpy()[0, -1].astype(np.float32).copy()
            return read, time.monotonic() - t1

        def letter_probs(lg, n):
            letters = lg[LETTER_IDS[0]:LETTER_IDS[0] + n].astype(np.float64)
            p = np.exp(letters - letters.max())
            return letters, p / p.sum()

        read, secs = await run_row(emb, rc.numpy())
        out["steps"] = len(ids)
        out["step_seconds_total"] = secs
        slots = []
        for s, (t, n) in enumerate(zip(row["slot_idx"], row["nopts"])):
            lg = read[t]
            letters, p = letter_probs(lg, n)
            po = np.asarray(row["probs"][s], dtype=np.float64)
            slots.append({"t": t, "nopts": n, "finite": bool(np.isfinite(lg).all()),
                          "full_vocab_argmax_id": int(lg.argmax()),
                          "letter_logits": letters.tolist(), "letter_logits_oracle": row["letter_logits"][s],
                          "probs": p.tolist(), "probs_oracle": row["probs"][s],
                          "argmax": int(p.argmax()), "argmax_oracle": row["argmax"][s],
                          "max_abs_dp": float(np.abs(p - po).max())})
            print(f"slot t={t}: probs {np.round(p, 5).tolist()} oracle {np.round(po, 5).tolist()} "
                  f"max|dp| {slots[-1]['max_abs_dp']:.2e} full-vocab argmax {slots[-1]['full_vocab_argmax_id']}",
                  flush=True)
        out["slots"] = slots

        # Red arms on the compiled graph: host-side input changes only, so a graph that ignored
        # its image inputs would show no movement here.
        reds = {"embeds_zero": (np.zeros_like(emb), rc.numpy()),
                "rc_swap": (emb, rc.numpy()[:, [1, 0]].copy())}
        out["red_arms"] = {}
        for name in [x for x in (args.smoke_red or "").split(",") if x]:
            e_in, rc_in = reds[name]
            r_read, _ = await run_row(e_in, rc_in)
            arm = []
            for s, (t, n) in enumerate(zip(row["slot_idx"], row["nopts"])):
                _, p = letter_probs(r_read[t], n)
                base = np.asarray(slots[s]["probs"])
                arm.append({"t": t, "probs": p.tolist(), "max_abs_dp_vs_base": float(np.abs(p - base).max()),
                            "argmax": int(p.argmax()), "base_argmax": int(base.argmax()),
                            "argmax_changed": int(p.argmax()) != int(base.argmax())})
                print(f"red {name} slot t={t}: probs {np.round(p, 5).tolist()} max|dp| vs base "
                      f"{arm[-1]['max_abs_dp_vs_base']:.4f} argmax {arm[-1]['base_argmax']}->{arm[-1]['argmax']}",
                      flush=True)
            out["red_arms"][name] = arm

    asyncio.run(run())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="fp16", choices=["fp16", "int8hu", "int8lin", "int8mix"])
    ap.add_argument("--fp16-layers", type=lambda s: [int(x) for x in s.split(",") if x != ""],
                    help="int8mix only: comma list of decoder layer indices whose linears stay fp16")
    ap.add_argument("--head-quant", default="block32", choices=["block32", "block16", "block8"],
                    help="int8hu only: lm_head weight granularity (ship = block32)")
    ap.add_argument("--head-sym", action="store_true",
                    help="int8hu only: plain symmetric (absmax, no clipping) for the head (ship)")
    ap.add_argument("--hf-id", default=HF_ID)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--out-dir", default=str(work_path("_decider2bv", "exports")),
                    help="bundles go to <out-dir>/bundles/<name>/, AOT assets to <out-dir>/bundles_aotc/")
    ap.add_argument("--name", help="override the generated bundle directory and asset name")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--n-image-max", type=int, default=256)
    ap.add_argument("--prefill-chunk", type=int, default=None,
                    help="add a static-S 'prefill' function (last-position logits, unrolled GDN scan) "
                         "next to 'main' in one multifunction asset; the name gets _pf<S>")
    ap.add_argument("--skip-export", action="store_true", help="reuse the saved .aimodel")
    ap.add_argument("--aot", action="store_true", help="compile the .aimodel for the Mac GPU (h16c)")
    ap.add_argument("--smoke", help="<row>:<arm> from the fixture oracle, e.g. r14:g256 (compiles first)")
    ap.add_argument("--smoke-red", help="comma list of red arms run after the smoke row on the compiled "
                                        "graph: embeds_zero, rc_swap")
    ap.add_argument("--record", help="write the export / AOT / smoke record JSON here")
    args = ap.parse_args()
    if args.mode != "int8hu" and (args.head_sym or args.head_quant != "block32"):
        ap.error("--head-quant / --head-sym apply to int8hu only")
    if (args.mode == "int8mix") != bool(args.fp16_layers):
        ap.error("--fp16-layers is required by int8mix and applies to it only")

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_decode_{args.mode}"
    if args.mode == "int8hu" and (args.head_quant != "block32" or args.head_sym):
        name += f"_{args.head_quant}" + ("_sym" if args.head_sym else "")
    if args.prefill_chunk:
        if args.prefill_chunk < 2:
            ap.error("--prefill-chunk must be >= 2 (S=1 is 'main')")
        if args.smoke:
            ap.error("--smoke drives 'main' only; gate a prefill bundle with readout_gate_vision.py")
        name += f"_pf{args.prefill_chunk}"
    name = args.name or name
    out_dir = Path(args.out_dir) / "bundles" / name
    aot_dir = Path(args.out_dir) / "bundles_aotc"
    record: dict = {"mode": args.mode, "name": name, "hf_id": args.hf_id, "revision": args.revision,
                    "prefill_chunk": args.prefill_chunk}
    if not args.skip_export:
        record["export"] = export(args, out_dir, name)
    if args.aot or args.smoke:
        aimodelc, secs = aot_compile(out_dir / f"{name}.aimodel", aot_dir)
        record["aot"] = {"aimodelc": str(aimodelc), "flags": AOT_FLAGS, "seconds": secs,
                         "du_aimodelc": du(aimodelc), "digest": tree_digest(aimodelc)}
        print(f"asset: {aimodelc} (compile {secs:.1f} s, {record['aot']['du_aimodelc']}, "
              f"tree sha256 {record['aot']['digest']['tree_sha256']})", flush=True)
        if args.smoke:
            record["smoke"] = smoke(args, aimodelc)
    if args.record:
        Path(args.record).write_text(json.dumps(record, indent=1) + "\n")
        print(f"record: {args.record}")


if __name__ == "__main__":
    main()
