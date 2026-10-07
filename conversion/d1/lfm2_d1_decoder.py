# d1 decoder: LiquidAI/d1-3B's LFM2 text stack returning the final-norm hidden state at every
# position of a static-S chunk (no vocabulary head).
#
# Community port — NOT an Apple model.
#
# LiquidAI/d1-3B (LFM Open License v1.0) is LFM2.5-VL-3B post-trained for single-pass decisions:
# the provider's `prompt.readout` reads the log-softmax at the row's last position (the answer
# slot) at a few option token ids, max-pools each option's forms and softmaxes over the options.
# A softmax over options is unchanged by the vocabulary log-sum-exp, so the host needs the logits
# of those ids only: logit = h_slot . E[id], with E the tied embedding table (the checkpoint has no
# lm_head key). So the graph is the overlay's LFM2.5-VL text decoder (`Lfm2VlPipelinedForCausalLM`,
# coreai_models/models/macos/lfm2_vl.py: ids input with image tokens as extension ids V + slot, the
# static `image_embeds` input, three states) with three changes and nothing else:
#
#   (a) no `lm_head`: the parent's head slot is an Identity, so its forward returns the final-norm
#       hidden at EVERY query position ([1, S, 2048]; `last_token_only` is refused). The tied
#       [128000, 2048] table stays in the graph only as the embedding the ids gather from, and a
#       weight-only int8 pass that targets linears never sees it.
#   (b) `from_hf` is the parent's loader (`load_lfm2_vl_state_dict(..., "model.language_model.")`,
#       the four attention projections kept fp32 on an fp16 load), then the head slot is reset to an
#       Identity (the loader re-ties `lm_head.weight`, which on an Identity registers an alias) and
#       every checkpoint key under the prefix is recorded against the module's tensors
#       (`load_report`; the parent's load_state_dict is strict=False).
#   (c) `build_export_spec`: the parent's spec at a static query length S (`trace_query=S,
#       static_ids=True`) with the output renamed `hidden`.
#
# Graph (static S; states mutate in place):
#
#   inputs  input_ids     [1, S]        int32  static; ids < V are tokens, V + slot reads image_embeds[slot]
#           position_ids  [1, seq]      int32  dynamic; the cache ramp 0..seq-1 (offset = seq - S)
#           image_embeds  [2816, 2048]  fp16   static; zeros for a text row
#   states  keyCache / valueCache [8, 1, 8, ctx, 64] (ctx dynamic), convState [22, 1, 2048, 2]
#   output  hidden        [1, S, 2048]  final-norm hidden at every position
#
# A row of T tokens runs as ceil(T / S) calls from fresh zero states; call c gets ids[cS : cS + S]
# with position_ids 0..cS+S-1, the last call is padded with <|pad|> (124893) and the padded
# positions' outputs are discarded (causal, so they cannot reach a real position). The host reads
# the hidden row of the row's last real token (the answer slot) — `conversion/d1/host.py`.
#
# Image rows (N_IMAGE_TOKENS = 2,816 = 11 x 256): the k-th <image> of a row, counted over every
# picture and every crop in text order, is sent as V + k and reads image_embeds[k]; rows 0..n-1 are
# the crops' tower rows (export_vision.py's bundle: each crop's first h * w / 4 rows) concatenated in
# crop order, rows n..N-1 zero, the same buffer bound to every call of the row. One picture needs at
# most 2,810 rows (10 tiles + a thumbnail, aspect up to 4:1 after the provider's cap_pixels;
# vision_host.py, K/results/vision_grid_table.json), so N holds any one picture. A host refuses, before
# any graph call, a request whose pictures need more than N rows together and a row over the position
# bound (ceil(T / S) * S <= max_ctx - 1: 4,080 tokens at S = 16, max_ctx 4,096; host.graph_context_check).
# A larger N costs a call no measurable time (the bound buffer is not copied per call:
# K/results/r3b_image_rows_cost.json, a toy at the model's width).
from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from coreai_models.models.macos.lfm2 import DECODE_STATE_NAMES, Lfm2Config, build_decode_state
from coreai_models.models.macos.lfm2_vl import (
    Lfm2VlPipelinedForCausalLM,
    _text_state_dict,
    lfm2_vl_configs_from_dict,
)

PREFILL_CHUNK = 16
N_IMAGE_TOKENS = 2816     # 11 x 256 >= one picture's 2,810 image tokens
PAD_ID = 124893            # <|pad|> (config pad_token_id); never read: its rows are dropped
INPUT_NAMES = ("input_ids", "position_ids", "image_embeds")
OUTPUT_NAMES = ("hidden",)
CHECKPOINT_PREFIX = "model.language_model."

# config.json `text_config` of LiquidAI/d1-3B @ da1fe36a (= LFM2.5-VL-3B's), as the authoring config reads it
EXPECTED_TEXT_CONFIG = {
    "num_hidden_layers": 30, "num_full_layers": 8, "num_conv_layers": 22, "hidden_size": 2048, "ff_dim": 10752,
    "vocab_size": 128000, "rope_theta": 1e6, "tie_embedding": True, "head_dim": 64, "num_attention_heads": 32,
    "num_key_value_heads": 8, "conv_L_cache": 3, "conv_state_width": 2, "norm_eps": 1e-5,
    "full_attention_layers": [2, 5, 9, 13, 17, 21, 24, 27],
}

__all__ = ["CHECKPOINT_PREFIX", "EXPECTED_TEXT_CONFIG", "INPUT_NAMES", "Lfm2D1Decoder", "N_IMAGE_TOKENS", "OUTPUT_NAMES",
           "PAD_ID", "PREFILL_CHUNK", "checkpoint_key_map", "text_config", "text_config_record"]


def text_config(raw: dict) -> Lfm2Config:
    """The authoring text config from the raw config.json (the overlay's own parser)."""
    return lfm2_vl_configs_from_dict(raw)[1]


def text_config_record(cfg: Lfm2Config) -> dict:
    """The fields EXPECTED_TEXT_CONFIG names, read off an authoring config."""
    return {"num_hidden_layers": cfg.num_hidden_layers, "num_full_layers": cfg.num_full_layers,
            "num_conv_layers": cfg.num_conv_layers, "hidden_size": cfg.hidden_size, "ff_dim": cfg.ff_dim,
            "vocab_size": cfg.vocab_size, "rope_theta": cfg.rope_theta, "tie_embedding": cfg.tie_embedding,
            "head_dim": cfg.head_dim, "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads, "conv_L_cache": cfg.conv_L_cache,
            "conv_state_width": cfg.conv_state_width, "norm_eps": cfg.norm_eps,
            "full_attention_layers": [i for i in range(cfg.num_hidden_layers) if cfg.is_full(i)]}


def checkpoint_key_map(keys: list[str]) -> dict[str, str]:
    """Checkpoint keys under `model.language_model.` -> the module's state-dict names (the parent loader's rename:
    prefix stripped, HF's w1 / w3 / w2 to gate_proj / up_proj / down_proj, `model.` prepended)."""
    out = {}
    for k in keys:
        if k.startswith(CHECKPOINT_PREFIX):
            (out[k],) = _text_state_dict({k.removeprefix(CHECKPOINT_PREFIX): torch.empty(0)}, fp32_attn_proj=False)
    return out


class Lfm2D1Decoder(Lfm2VlPipelinedForCausalLM):
    """The overlay's LFM2.5-VL text decoder without a vocabulary head; contract in the header."""

    def __init__(self, config: Lfm2Config, n_image_tokens: int = N_IMAGE_TOKENS) -> None:
        super().__init__(config, n_image_tokens=n_image_tokens)
        del self.lm_head                      # the tied table is the embedding; no head in the graph
        self.lm_head = torch.nn.Identity()

    def forward(
        self,
        input_ids: torch.Tensor,     # [1, S] int32
        position_ids: torch.Tensor,  # [1, seq] int32 cache ramp
        image_embeds: torch.Tensor,  # [N, hidden]
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> torch.Tensor:
        """-> hidden [1, S, hidden]: the final-norm hidden state at every query position."""
        if self.last_token_only:
            raise ValueError("the d1 decoder returns the final-norm hidden at every position only")
        return super().forward(input_ids, position_ids, image_embeds, k_cache, v_cache, conv_state)

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_hf(
        cls,
        hf_id_or_dir: str,
        target_dtype: torch.dtype = torch.float16,
        n_image_tokens: int = N_IMAGE_TOKENS,
        fp32_attn_proj: bool = True,
    ) -> "Lfm2D1Decoder":
        """The parent's loader on d1-3B's `model.language_model.*` weights (round 2), the head slot reset to an
        Identity, and `load_report`: every checkpoint key under the prefix against the module's tensors."""
        import glob
        import os

        from safetensors import safe_open

        from coreai_models.models.macos.lfm2_vl import _snapshot

        model = super().from_hf(hf_id_or_dir, target_dtype=target_dtype, n_image_tokens=n_image_tokens,
                                fp32_attn_proj=fp32_attn_proj)
        model.lm_head = torch.nn.Identity()
        model.eval()
        model_dir = _snapshot(hf_id_or_dir)
        ckpt, files = [], []
        for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            files.append(os.path.basename(path))
            with safe_open(path, framework="pt", device="cpu") as f:
                ckpt += list(f.keys())  # noqa: SIM118
        mapped = checkpoint_key_map(ckpt)
        own = set(model.state_dict(keep_vars=True).keys())
        model.load_report = {
            "checkpoint_files": files,
            "checkpoint_keys": len(ckpt),
            "checkpoint_keys_under_prefix": len(mapped),
            "module_tensors": len(own),
            "unread_checkpoint_keys_under_prefix": sorted(k for k, v in mapped.items() if v not in own),
            "module_tensors_not_in_checkpoint": sorted(own - set(mapped.values())),
            "checkpoint_has_lm_head_key": any("lm_head" in k for k in ckpt),
            "tie_embedding": bool(model.config.tie_embedding),
            "module_has_lm_head_weight": any(k.startswith("lm_head.") for k in own),
            "meta_params": [n for n, p in model.named_parameters() if p.is_meta],
            "fp32_params": sorted(n for n, p in model.named_parameters() if p.dtype == torch.float32),
            "dtype": str(target_dtype).replace("torch.", ""),
        }
        return model

    # -- export -------------------------------------------------------------

    def build_export_spec(
        self,
        target_dtype: torch.dtype,
        max_context_length: int,
        trace_kv_len: int,
        query_len: int = PREFILL_CHUNK,
        trace_past: int = 64,
    ) -> dict:
        """Static-S entrypoint: the parent's spec with `input_ids` [1, S] static (`static_ids=True`, so the
        position dim admits a first call at seq = S), `position_ids` and the KV sequence dim dynamic, `image_embeds`
        and the conv state fixed-shape, and the output named `hidden`."""
        if query_len < 2:
            raise ValueError("this graph has no S=1 function: query_len must be >= 2")
        if trace_past + query_len > trace_kv_len:
            raise ValueError(f"trace_past + S = {trace_past + query_len} exceeds trace_kv_len {trace_kv_len}")
        spec = super().build_export_spec(target_dtype, max_context_length, trace_kv_len, trace_query=query_len,
                                         trace_past=trace_past, static_ids=True)
        if tuple(spec["input_names"]) != INPUT_NAMES or tuple(spec["state_names"]) != tuple(DECODE_STATE_NAMES):
            raise ValueError(f"parent spec names changed: {spec['input_names']} / {spec['state_names']}")
        spec["output_names"] = OUTPUT_NAMES
        return spec


# --------------------------------------------------------------------------- the round-1 scout (no weights)
def _dims(dyn: dict) -> dict:
    out = {}
    for name, spec in dyn.items():
        if spec is None:
            out[name] = None
            continue
        out[name] = {str(axis): {"name": getattr(d, "__name__", str(d)), "min": getattr(d, "min", None),
                                 "max": getattr(d, "max", None)} for axis, d in spec.items()}
    return out


def scout(config_json: Path, header_json: Path | None = None, max_ctx: int = 4096, chunk: int = PREFILL_CHUNK) -> dict:
    """The decoder module against d1-3B's config with no weights: the config asserts, the state shapes at max_ctx,
    the module on the meta device (forward signature, parameter names and shapes), the export spec's names and
    dynamic dims, and — when the safetensors header is given — every `model.language_model.*` key mapped onto the
    module's tensors with its shape."""
    from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN

    raw = json.loads(Path(config_json).read_text())
    cfg = text_config(raw)
    got = text_config_record(cfg)
    mismatch = {k: (got[k], v) for k, v in EXPECTED_TEXT_CONFIG.items() if got[k] != v}
    if mismatch:
        raise AssertionError(f"text config differs from the expected d1-3B text config: {mismatch}")
    with torch.device("meta"):
        state = build_decode_state(cfg, max_seq_len=max_ctx, dtype=torch.float16)
        model = Lfm2D1Decoder(cfg, n_image_tokens=N_IMAGE_TOKENS)
    spec = model.build_export_spec(torch.float16, max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, query_len=chunk)
    params = {n: list(p.shape) for n, p in model.named_parameters()}
    out = {
        "config_json": str(config_json),
        "text_config": got,
        "text_config_asserts": {k: "ok" for k in EXPECTED_TEXT_CONFIG},
        "image_token_id": raw.get("image_token_id"),
        "decode_state_at_max_ctx": {"max_ctx": max_ctx,
                                    **{k: list(v.shape) for k, v in state.items()},
                                    "state_names": list(DECODE_STATE_NAMES)},
        "module": {"class": f"{Lfm2D1Decoder.__module__}.{Lfm2D1Decoder.__qualname__}",
                   "parent": f"{Lfm2VlPipelinedForCausalLM.__module__}.{Lfm2VlPipelinedForCausalLM.__qualname__}",
                   "forward_signature": str(inspect.signature(model.forward)),
                   "lm_head": type(model.lm_head).__name__,
                   "n_image_tokens": model.n_image_tokens,
                   "parameters": len(params),
                   "parameter_count": int(sum(p.numel() for p in model.parameters())),
                   "embed_tokens": params["model.embed_tokens.weight"]},
        "export_spec": {"query_len": chunk, "trace_kv_len": TRACE_KV_CACHE_SEQ_LEN, "max_ctx": max_ctx,
                        "input_names": list(spec["input_names"]), "output_names": list(spec["output_names"]),
                        "state_names": list(spec["state_names"]),
                        "reference_inputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")}
                                             for k, v in spec["reference_inputs"].items()},
                        "dynamic_shapes": _dims(spec["dynamic_shapes"])},
    }
    if header_json is not None:
        header = json.loads(Path(header_json).read_text())
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        mapped = checkpoint_key_map(list(tensors))
        shape_bad = [k for k, m in mapped.items() if m in params and list(tensors[k]["shape"]) != params[m]]
        out["checkpoint_header_check"] = {
            "header_json": str(header_json),
            "checkpoint_tensors": len(tensors),
            "dtypes": sorted({v["dtype"] for v in tensors.values()}),
            "under_prefix": len(mapped),
            "mapped_to_module": sum(1 for m in mapped.values() if m in params),
            "unread_under_prefix": sorted(k for k, m in mapped.items() if m not in params),
            "module_params_not_in_checkpoint": sorted(set(params) - set(mapped.values())),
            "shape_mismatches": shape_bad,
            "other_prefixes": sorted({".".join(k.split(".")[:2]) for k in tensors if not k.startswith(CHECKPOINT_PREFIX)}),
            "lm_head_key": [k for k in tensors if "lm_head" in k],
        }
    return out


if __name__ == "__main__":   # the round-1 scout: python lfm2_d1_decoder.py <config.json> [header.json] --out <json>
    import argparse

    ap = argparse.ArgumentParser(description="d1 decoder module against the d1-3B config (no weights)")
    ap.add_argument("config_json")
    ap.add_argument("header_json", nargs="?", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rec = scout(Path(args.config_json), Path(args.header_json) if args.header_json else None)
    Path(args.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("export_spec",)}, indent=1)[:4000])
