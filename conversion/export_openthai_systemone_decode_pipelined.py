#!/usr/bin/env python3
"""Export OpenThai-SystemOne's 256-way slot head as a Core AI decode bundle.

The graph is the Qwen3.5 S=1 recipe in export_qwen3_5_decode_pipelined.py.
This checkpoint supplies the text tower under model.*, resizes its input
embedding to 248,339 tokens, and replaces the LM head with a biased 256-way
slot head. The remote root config is read as JSON; its text_config is passed
to the Qwen3.5 overlay without importing the checkpoint's remote code.

The embedding keeps its full input vocabulary. Bundle language.vocab_size is
256 because the sequential engine allocates its logits buffer from that field.
The decision block records the answer token, abstain slot, and checkpoint
temperatures. Apply temperature, mask invalid slots, and normalize outside
this graph; it returns raw logits [1, 1, 256] and never generates a decision
label as text. See conversion/slot/README.md for the readout contract.

Modes: fp16 is the reference; int8lin quantizes linears per block of 32 and
leaves embeddings, conv1d, norms, and the biased slot head in fp16. Both keep
the original loop-free S=1 path and four decode states. COREAI_CHUNK_THRESHOLD=1
is required when running through the Core AI engines. The Qwen3.5 authoring
overlay and the Swift extra-states patch are the same dependencies as the
Qwen3.5 decode exporter. _bundle.py is the shared zoo metadata tail.

Run: python export_openthai_systemone_decode_pipelined.py int8lin \
         --out-dir exports --max-ctx 4096
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
from _bundle import write_bundle_metadata
from huggingface_hub import snapshot_download
from safetensors import safe_open

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.qwen3_5 import (
    DECODE_STATE_NAMES,
    Qwen3_5StatefulForCausalLM,
    build_decode_state,
    qwen3_5_config_from_hf,
)
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16
HF_ID = "iapp/OpenThai-SystemOne"
REVISION = "f3709948b5e3cc9606a57e74ba62b7a639d17dd3"



def load_text_config(snapshot, max_context_length=4096):
    # (a) The checkpoint's remote root config is not the Qwen text config.
    raw = json.loads((Path(snapshot) / 'config.json').read_text())
    text = raw['text_config']
    assert text['vocab_size'] == 248339 and text['tie_word_embeddings'] is False
    assert raw['n_slots'] == 256 and raw['abstain_slot'] == 255 and raw['answer_token_id'] == 248082
    return qwen3_5_config_from_hf(SimpleNamespace(**text), max_context_length)


def load_slot_model(snapshot, max_context_length=4096, target_dtype=torch.float16):
    snapshot = Path(snapshot).resolve()
    cfg = load_text_config(snapshot, max_context_length)
    model = Qwen3_5StatefulForCausalLM(cfg, model_device='meta')
    model.to(dtype=target_dtype)
    # (b) Retain the frozen loader's safetensors loop with prefix=model.; no HF auto-config.
    sd, shapes, head, temp = {}, {}, {}, None
    prefix = 'model.'
    files = sorted(snapshot.glob('*.safetensors'))
    assert len(files) == 1
    for path in files:
        with safe_open(path, framework='pt', device='cpu') as f:
            for key in f.keys():
                shapes[key] = list(f.get_slice(key).get_shape())
                if key == 'log_temperature':
                    temp = f.get_tensor(key).float().exp()
                if key.startswith('slot_head.'):
                    head[key] = f.get_tensor(key).to(target_dtype)
                if not key.startswith(prefix):
                    continue
                local = 'model.' + key[len(prefix):]
                sd[local] = f.get_tensor(key).to(target_dtype)
    assert len(shapes) == 323
    missing, unexpected = model.load_state_dict(sd, assign=True, strict=False)
    assert missing == ['lm_head.weight'] and not unexpected, (missing, unexpected)
    assert shapes['model.embed_tokens.weight'] == [248339, 1024]
    assert head['slot_head.weight'].shape == (256, 1024)
    assert head['slot_head.bias'].shape == (256,)
    # (c) The absent LM head is replaced before the original loader's final meta check.
    model.lm_head = torch.nn.Linear(1024, 256, bias=True, dtype=target_dtype)
    model.lm_head.load_state_dict({'weight':head['slot_head.weight'], 'bias':head['slot_head.bias']})
    assert model.model.embed_tokens.weight.shape == (248339, 1024)
    assert torch.equal(model.lm_head.weight,head['slot_head.weight'])
    assert torch.equal(model.lm_head.bias,head['slot_head.bias'])
    model.model.reset_buffers()
    assert not [n for n,p in model.named_parameters() if p.is_meta]
    assert temp is not None and torch.isfinite(temp).all()
    temperatures = dict(zip(('choice','score','noul'), temp.tolist()))
    print(f"loaded {len(shapes)} tensors from {snapshot.name}; prefix=model.; "
          f"embedding={list(model.model.embed_tokens.weight.shape)}; "
          f"slot_head={list(model.lm_head.weight.shape)} + bias")
    print(f"temperature_by_type={temperatures}")
    return model, temperatures


def linear_quant_config(dtype: str = "int8") -> dict:
    """Weight-only linear int8 per-block-32 - scale-multiply dequant, no LUT.
    Embedding/conv/norms and the biased slot head remain fp16."""
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
                    choices=["fp16", "int8lin"])
    ap.add_argument("--hf-id", default=HF_ID)
    ap.add_argument("--revision", default=REVISION, help="Hugging Face commit or ref")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=4096)
    args = ap.parse_args()

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_decode_{args.mode}"
    print(f"loading {args.hf_id} fp16 ...")
    snapshot = Path(snapshot_download(
        args.hf_id, revision=args.revision, ignore_patterns=["assets/*"], max_workers=8))
    resolved_revision = snapshot.name
    if re.fullmatch(r"[0-9a-f]{40}", args.revision):
        assert resolved_revision == args.revision, (resolved_revision, args.revision)
    print(f"snapshot={snapshot}; resolved_revision={resolved_revision}")
    model, temperatures = load_slot_model(
        snapshot, max_context_length=args.max_ctx, target_dtype=DTYPE)
    head_weight, head_bias = model.lm_head.weight.detach().clone(), model.lm_head.bias.detach().clone()
    model.eval()
    cfg = model.config

    n_lin = 0
    for layer in model.model.layers:
        if not layer.is_full:
            layer.linear_attn.use_loopfree_step = True
            n_lin += 1
    print(f"loop-free single-step enabled on {n_lin} linear layers")

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
        "input_ids": None,  # static [1, 1] - no scan, no while_loop
        "position_ids": {1: seq_pos},
        "k_cache": {KVCache.seq_len_dim(): k_seq},
        "v_cache": {KVCache.seq_len_dim(): v_seq},
        "conv_state": None,
        "rec_state": None,
    }

    if args.mode == "int8lin":
        from coreai_models.export.compression import quantize_pytorch_model

        print("quantizing (linear int8 per-block-32; slot head excluded) ...")
        model = quantize_pytorch_model(
            model, tuple(reference_inputs.values()), dynamic_shapes, linear_quant_config())

    # int8lin excludes the replaced lm_head by its unchanged module name.
    assert re.fullmatch(r".*lm_head$", "lm_head")
    assert model.lm_head.weight.dtype == DTYPE and model.lm_head.bias.dtype == DTYPE
    assert torch.equal(model.lm_head.weight, head_weight) and torch.equal(model.lm_head.bias, head_bias)

    # The loop-free path never calls the GatedDeltaUpdate composite; externalizing
    # the class would mark the (uncalled) submodules and then fail to find them in
    # the traced program. RMSNorm/SDPA stay fused composites for the GPU delegate.
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

    # Runtime logits buffers use the 256-way output width; input vocabulary stays 248339.
    write_bundle_metadata(out_dir, name, args.hf_id, 256, args.max_ctx, revision=resolved_revision,
        extra={"decision": {"head": "slot", "n_slots": 256, "abstain_slot": 255,
            "answer_token_id": 248082, "temperature_by_type": temperatures,
            "layout": "openthai_systemone"}})
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(snapshot, local_files_only=True).save_pretrained(out_dir / "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(out_dir / "tokenizer", local_files_only=True)
    assert tokenizer.encode("<|ts_answer|>", add_special_tokens=False) == [248082]
    assert len(json.loads((out_dir / "tokenizer" / "tokenizer.json").read_text())["added_tokens"]) == 295
    assert len(tokenizer) == 248339
    print(f"bundle ready: {out_dir}")
    print(f"run: COREAI_CHUNK_THRESHOLD=1 llm-benchmark --model {out_dir} -p 128 -g 256 -n 3")


if __name__ == "__main__":
    main()
