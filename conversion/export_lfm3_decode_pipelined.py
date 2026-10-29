"""Export a DECODE-ONLY (S=1) LFM3 bundle for the Core AI pipelined engine.

LFM3 (LiquidAI; 1.7B: 13 short-conv + 6 full-attention + 5 sliding-attention
layers) is LFM2 plus a sliding-window layer type, a per-head sigmoid output gate
on attention, a zero-tail partial RoPE on the full layers, sandwich norms and a
width-7 short conv — see the header of `coreai_models/models/macos/lfm3.py`.
Still no recurrent scan, so the decode graph is loop-free. State = ONE growing
KV pair for all attention layers (full and sliding; the window is the SDPA
composite's own mask over the full-length KV) + ONE fixed-shape conv state
[13, 1, hidden, 6], inside the pipelined engine's <=2 extra-state budget.

input_ids (or inputs_embeds) is STATIC [1,1]; position_ids and the KV seq dim
stay dynamic, so `EngineFactory` classifies the bundle as dynamic -> pipelined
engine. Prefill runs as pipelined S=1 steps: set `COREAI_CHUNK_THRESHOLD=1`.
AOT for the Mac: `xcrun coreai-build compile <name>.aimodel --output <dir>
--platform macOS --preferred-compute gpu --architecture h16c
--expect-frequent-reshapes`, then point metadata.json `assets.main` at the
`.aimodelc` (a real directory, not a symlink).

Run:  python export_lfm3_decode_pipelined.py [fp16|int8lin|int8hu|int4lin|int4lin8|int4hu] \
          [--src LiquidAI/LFM3-1.7B | <local checkpoint dir>] [--revision REV] [--out-dir DIR]
      Multimodal checkpoints (the LFM3-VL / -Omni / -Audio decoders):
          --key-prefix model.language_model. --config-json <repo>/config.json --config-key text_config
          --key-prefix lfm.model.            --config-json <repo>/config.json --config-key lfm
          --embeds        query = inputs_embeds [1, 1, hidden] instead of input_ids
          --emit-hidden   ONE graph returns logits AND the final-norm hidden [1, 1, hidden]
      The same as a function: `export_lfm3_decode(src, out_dir, mode=..., ...)`.

Modes: fp16 - baseline; int8lin - per-block-32 linear int8 (scale-multiply
dequant, no LUT); int8hu - int8lin + untied int8 lm_head (`--head-quant block32
--head-sym` = the ship shape for big-vocab heads); int4lin - per-block-32 linear
int4; int4lin8 - int4 with the rescue set (default: the conv-mixer projections)
kept int8; int4hu - int4lin8 + the untied int8 lm_head of int8hu (mixed int4:
`--rescue-regex` names every body FC that stays int8, e.g. only the MLP w1 / w3
at int4 = `--rescue-regex '.*(feed_forward\\.down_proj|conv\\.(in_proj|out_proj))$'`).
In every quantized mode the embedding, conv1d, norms and the attention
q/k/v/o/gate projections stay high precision (the attention set is the
GPU-delegate precision path, fp32 — lfm3.py IMPORTANT #2); lm_head stays fp16
except in int8hu and int4hu.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from _bundle import head_quant_spec, write_bundle_metadata
from _paths import exports_dir

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.lfm3 import (
    build_pipelined_export_spec,
    lfm3_from_hf,
    resolve_lfm3_source,
)

DTYPE = torch.float16
MODES = ("fp16", "int8lin", "int8hu", "int4lin", "int4lin8", "int4hu")
HEAD_MODES = ("int8hu", "int4hu")  # untied int8 lm_head on top of the body recipe


def linear_quant_config(dtype: str = "int8", rescue_int8_regex: str | None = None,
                        block: int = 32) -> dict:
    """Weight-only linear per-block — scale-multiply dequant, no LUT.
    Embedding/conv/norms excluded by type; lm_head excluded by name (tied
    table stays fp16 — the eager quantizer skips shared parameters anyway).
    The attention q/k/v/o projections AND gate_proj are excluded too: they are
    the GPU-delegate precision-critical path, kept fp32 (lfm3.py IMPORTANT
    #2). The MLP + conv-mixer linears (the bulk of the weights) quantize.
    ``rescue_int8_regex`` keeps the matching modules at int8 per-block-32
    while the rest take ``dtype``."""
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

    name_configs: dict = {
        r".*lm_head$": None,
        r".*self_attn\.(q_proj|k_proj|v_proj|o_proj|gate_proj)$": None,
    }
    if rescue_int8_regex:
        name_configs[rescue_int8_regex] = spec("int8")
    return {
        "execution_mode": "eager",
        "global_config": spec(dtype, block),
        "module_type_configs": {
            "coreai_models.primitives.macos.sdpa.SDPA": None,
            "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
            "torch.nn.modules.sparse.Embedding": None,
            "torch.nn.modules.conv.Conv1d": None,
        },
        "module_name_configs": name_configs,
    }


def save_tokenizer(src: str, dest: Path, revision: str | None = None,
                   chat_template: str | None = None) -> None:
    """Copy the source tokenizer files into the bundle VERBATIM, with
    ``chat_template`` (a path) in place of the source `chat_template.jinja`.

    Verbatim, not `AutoTokenizer.save_pretrained`: LFM3 declares
    `tokenizer_class: "TokenizersBackend"`, which a 4.x `AutoTokenizer`
    refuses (knowledge/lfm2.5-2.6b-port.md), and a re-serialized config is not
    the file the Swift tokenizer was checked against. The vendor template does
    not parse on the CM pin of swift-jinja (2.3.2) and CoreAILanguageModel then
    falls back to a plain encode without a word — hence the override: a
    rewrite that renders the vendor's strings on transformers' jinja2 and on
    swift-jinja (lfm3 r3a, `_lfm3/r3a/template/CHANGES.md`). `src` is a local
    checkpoint dir or an HF id.
    """
    from _bundle import _TOKENIZER_PATTERNS, is_tokenizer_file

    snap = Path(src)
    if not snap.is_dir():
        from huggingface_hub import snapshot_download

        snap = Path(snapshot_download(src, revision=revision, allow_patterns=_TOKENIZER_PATTERNS))
    dest.mkdir(parents=True, exist_ok=True)
    for f in sorted(snap.iterdir()):
        if f.is_file() and is_tokenizer_file(f.name):
            shutil.copy2(f, dest / f.name)
    if chat_template:
        shutil.copy2(chat_template, dest / "chat_template.jinja")
    if not (dest / "chat_template.jinja").exists():
        print("  WARNING: no chat_template.jinja in the bundle tokenizer")


def export_lfm3_decode(
    src: str,
    out_dir: str | Path | None = None,
    *,
    mode: str = "fp16",
    revision: str | None = None,
    key_prefix: str = "model.",
    config: dict | str | None = None,
    config_key: str | None = None,
    embeds: bool = False,
    emit_hidden: bool = False,
    max_ctx: int = 4096,
    name: str | None = None,
    head_quant: str = "block32",
    head_sym: bool = False,
    rescue_regex: str | None = None,
    int4_block: int = 32,
    tokenizer_src: str | None = None,
    chat_template: str | None = None,
    hf_id: str | None = None,
) -> Path:
    """Export one decode bundle (`<out_dir>/<name>/`: .aimodel, metadata.json,
    tokenizer/) and return its directory.

    ``src``: HF id or local checkpoint dir; ``key_prefix`` / ``config`` /
    ``config_key`` pick the decoder inside a multimodal checkpoint (lfm3.py
    ``lfm3_from_hf``). ``tokenizer_src`` defaults to ``src`` (a VL / Omni /
    Audio checkpoint carries its own tokenizer); pass ``""`` to skip it.
    ``chat_template`` (a path) replaces the source `chat_template.jinja`.
    ``hf_id`` names the source in metadata.json (default: ``src`` when it is
    not a local dir).
    """
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    if name is None:
        short = (hf_id or src).rstrip("/").rsplit("/", 1)[-1].lower()
        short = short.replace(".", "_").replace("-", "_")
        name = f"{short}_decode_{mode}" + ("_embeds" if embeds else "") + ("_hidden" if emit_hidden else "")
        if mode in HEAD_MODES and (head_quant != "block32" or head_sym):
            name += f"_{head_quant}" + ("_sym" if head_sym else "")

    print(f"loading {src} (prefix {key_prefix!r}{', config ' + config_key if config_key else ''}) fp16 ...")
    model = lfm3_from_hf(src, key_prefix=key_prefix, config=config, config_key=config_key,
                         target_dtype=DTYPE, stateful=True, embeds=embeds,
                         emit_hidden=emit_hidden, revision=revision)
    cfg = model.config
    print(f"{cfg.num_hidden_layers} layers ({cfg.layer_types.count('full_attention')} full / "
          f"{cfg.layer_types.count('sliding_attention')} sliding / {cfg.num_conv_layers} conv), "
          f"window {cfg.sliding_window}, vocab {cfg.vocab_size}")

    # Decode trace: S=1 static query, dynamic full-length positions, dynamic KV seq.
    spec = build_pipelined_export_spec(
        cfg, target_dtype=DTYPE, max_context_length=max_ctx,
        trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, trace_query=1, trace_past=64,
        embeds=embeds, emit_hidden=emit_hidden)

    if mode != "fp16":
        from coreai_models.export.compression import quantize_pytorch_model

        dtype = "int8" if mode in ("int8lin", "int8hu") else "int4"
        rescue = (rescue_regex or r".*conv\.(in_proj|out_proj)$") if mode in ("int4lin8", "int4hu") else None
        block = int4_block if dtype == "int4" else 32
        cfg_q = linear_quant_config(dtype, rescue_int8_regex=rescue, block=block)
        if mode in HEAD_MODES:
            # Untie the head (the eager quantizer skips shared params) and
            # quantize it at int8 on top of the body recipe.
            cfg_q["module_name_configs"][r".*lm_head$"] = head_quant_spec(head_quant, head_sym)
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        print(f"quantizing (linear {dtype} per-block-{block}, mode={mode}"
              f"{f', rescued to int8: {rescue}' if rescue else ''}"
              f"{', head int8 ' + head_quant + (' sym' if head_sym else '') if mode in HEAD_MODES else ''}) ...")
        model = quantize_pytorch_model(
            model, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"], cfg_q)

    # No GatedDeltaUpdate in LFM3 — drop its externalize spec. RMSNorm/SDPA
    # stay fused composites for the GPU delegate.
    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    print(f"exporting decode-only graph to Core AI dialect (inputs {spec['input_names']}, "
          f"outputs {spec['output_names']}) ...")
    prog = export_to_coreai(
        model,
        spec["reference_inputs"],
        dynamic_shapes=spec["dynamic_shapes"],
        input_names=spec["input_names"],
        output_names=spec["output_names"],
        state_names=spec["state_names"],
        externalize_modules=specs,
    )
    print("optimizing ...")
    prog.optimize()

    out = (Path(out_dir) if out_dir else exports_dir()) / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    import coreai.runtime as rt

    aimodel = out / f"{name}.aimodel"
    print(f"saving {aimodel} ...")
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())

    local_src = Path(src).is_dir()
    meta_id = hf_id or (Path(src).name if local_src else src)
    language_extra = {}
    if embeds:
        language_extra["inputs_embeds"] = [1, 1, cfg.hidden_size]
    extra = {"lfm3": {"inputs": list(spec["input_names"]), "outputs": list(spec["output_names"]),
                      "states": list(spec["state_names"]),
                      "kv_slots": cfg.attention_layer_ids, "conv_slots": cfg.conv_layer_ids,
                      "sliding_window": cfg.sliding_window}}
    write_bundle_metadata(out, name, meta_id, cfg.vocab_size, max_ctx, revision=revision,
                          weights=str(src) if local_src else None, mode=mode,
                          language_extra=language_extra or None, extra=extra)
    tok_src = src if tokenizer_src is None else tokenizer_src
    if tok_src:
        save_tokenizer(tok_src, out / "tokenizer", revision=None if Path(tok_src).is_dir() else revision,
                       chat_template=chat_template)
    print(f"bundle ready: {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="fp16", choices=MODES)
    ap.add_argument("--src", default="LiquidAI/LFM3-1.7B",
                    help="HF id or local checkpoint dir (config.json + *.safetensors)")
    ap.add_argument("--revision", default=None, help="HF revision to pin (HF id only)")
    ap.add_argument("--hf-id", default=None, help="source id written to metadata.json")
    ap.add_argument("--key-prefix", default="model.",
                    help="decoder key prefix: model. (1.7B), model.language_model. (VL/Omni), lfm.model. (Audio)")
    ap.add_argument("--config-json", default=None, help="config JSON (default: <src>/config.json)")
    ap.add_argument("--config-key", default=None, help="sub-dict of the config: text_config | lfm")
    ap.add_argument("--embeds", action="store_true", help="inputs_embeds [1,1,hidden] instead of input_ids")
    ap.add_argument("--emit-hidden", action="store_true", help="also output the final-norm hidden")
    ap.add_argument("--tokenizer-src", default=None, help="tokenizer source (default: --src; '' = none)")
    ap.add_argument("--chat-template", default=None,
                    help="a .jinja carried as the bundle's chat_template.jinja in place of the source's")
    # Default through _paths, not a bare relative "exports".
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--head-quant", default="block32",
                    choices=["block32", "block16", "block8", "perchan"],
                    help="int8hu / int4hu only: lm_head weight granularity (ship=block32; perchan is BROKEN on the beta GPU delegate)")
    ap.add_argument("--head-sym", action="store_true",
                    help="int8hu / int4hu only: plain symmetric (absmax, no clipping) for the head")
    ap.add_argument("--rescue-regex", default=None,
                    help="int4lin8 / int4hu only: the int8-rescue module regex (default = the conv-mixer projections)")
    ap.add_argument("--int4-block", type=int, default=32)
    args = ap.parse_args()

    config = args.config_json
    if config is None and args.config_key and not Path(args.src).is_dir():
        config = str(Path(resolve_lfm3_source(args.src, args.revision)) / "config.json")
    export_lfm3_decode(
        args.src, args.out_dir, mode=args.mode, revision=args.revision, key_prefix=args.key_prefix,
        config=config, config_key=args.config_key, embeds=args.embeds,
        emit_hidden=args.emit_hidden, max_ctx=args.max_ctx, name=args.name,
        head_quant=args.head_quant, head_sym=args.head_sym, rescue_regex=args.rescue_regex,
        int4_block=args.int4_block, tokenizer_src=args.tokenizer_src,
        chat_template=args.chat_template, hf_id=args.hf_id,
    )
    print("run: COREAI_CHUNK_THRESHOLD=1 llm-benchmark --model <bundle> -p 128 -g 256 -n 3")


if __name__ == "__main__":
    main()
