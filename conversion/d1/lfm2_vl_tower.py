# d1 vision tower, exact form: LiquidAI/d1-3B's SigLIP2-NaFlex tower and LFM2-VL projector as one graph
# whose inputs carry everything a crop's grid decides, so one graph serves every crop the processor makes.
#
# Community port — NOT an Apple model.
#
# The overlay's `Lfm2VlVisionEncoder` (coreai_models/models/macos/lfm2_vl.py) bakes one patch grid: the
# position-table resize is a load-time constant, there is no padding mask and the pixel-unshuffle is a
# reshape. d1's processor gives every picture its own grid — one crop at the picture's aspect (<= 1024
# patches, both sides even), or 512 x 512 tiles (32 x 32 patches) plus a thumbnail at the one-crop rule
# (conversion/d1/vision_host.py, `K/results/vision_rules.md` (b)-(d)) — so this module takes the grid's
# three consequences as inputs and is otherwise the same mathematics:
#
#   inputs  patches        [1024, 768]   the crop's patches row-major, [y][x][c] inside a patch (channel
#                                        fastest), (x - 127.5) / 127.5; rows past the crop's h * w patches 0
#           pos_table      [1024, d]     the checkpoint's 16 x 16 position table resized to the crop's (h, w)
#                                        patches (bilinear, antialias, fp32; vision_host.pos_table), row-major;
#                                        rows past h * w 0 (HF writes row 0 there; those rows are never read)
#           key_bias       [1024]        0 for a real patch, -inf for padding: added to every query's scores,
#                                        so a padded patch is never attended to
#           unshuffle_idx  [256, 4] i32  merged token k = i * (w / 2) + j concatenates patch rows
#                                        [2i w + 2j, 2i w + 2j + 1, (2i + 1) w + 2j, (2i + 1) w + 2j + 1]
#                                        (HF's pixel_unshuffle; vision_host.unshuffle_index); rows k >= h w / 4
#                                        point at patch 0
#   output  image_embeds   [256, text_hidden]   rows 0 .. h w / 4 - 1 are the crop's tokens, the rest discarded
#
#   x = patch_embedding(patches) + pos_table
#   per layer: x = x + attn(LN1(x)), attn scores = q k^T / sqrt(head_dim) + key_bias, softmax over keys
#              x = x + fc2(gelu_tanh(fc1(LN2(x))))
#   x = post_layernorm(x); u = gather(x, unshuffle_idx) -> [256, 4 d] (channel m * d + c from patch m)
#   image_embeds = linear_2(gelu_exact(linear_1(u)))          (no projector LayerNorm on this checkpoint)
#
# A padded query row is computed and dropped; a real query never attends to a padded key, so the real rows
# equal HF's `get_image_features` (the pad rows' content does not matter). The attention is either the bare
# matmul-softmax chain with the additive key_bias (`matmul`, exact on the GPU up to at least 3,072 keys:
# knowledge/clef-flash-port.md "Attention over 4,032 keys") or the coreai SDPA composite with the bool mask
# key_bias == 0 (`sdpa`; the composite reads a mask as bool). Checkpoint keys (tf 4.57 names, as saved):
# `model.vision_tower.vision_model.*` (transformers 5's Siglip2VisionModel drops the `vision_model.`
# segment when it loads them: conversion_mapping.py `Siglip2VisionModel` -> `CLIPVisionModel` ->
# PrefixChange("vision_model")) and `model.multi_modal_projector.*`. The position table is not a graph
# weight: the host resizes it per crop (`host_position_table`, shipped next to the graph).
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.models.macos.lfm2_vl import (
    Lfm2VlVisionConfig,
    _snapshot,
    _VisionMLP,
    lfm2_vl_configs_from_dict,
    load_lfm2_vl_state_dict,
)
from coreai_models.primitives.macos.sdpa import SDPA

N_PATCHES = 1024           # processor max_num_patches = max(256 * 2^2, (512 / 16)^2)
N_TOKENS = 256             # N_PATCHES / downsample^2
VISION_PREFIX = "model.vision_tower.vision_model."
PROJECTOR_PREFIX = "model.multi_modal_projector."
POSITION_KEY = VISION_PREFIX + "embeddings.position_embedding.weight"
HOST_POSITION_TABLE = "host: position table (resized per crop by the host; bundle host/position_embedding.safetensors)"
INPUT_NAMES = ("patches", "pos_table", "key_bias", "unshuffle_idx")
OUTPUT_NAMES = ("image_embeds",)
ATTENTIONS = ("matmul", "sdpa")
DTYPES = ("fp16", "fp16w32", "fp32")

# config.json of LiquidAI/d1-3B @ da1fe36a: `vision_config` and the projector fields, as this module reads them
EXPECTED_VISION_CONFIG = {
    "hidden_size": 1152, "intermediate_size": 4304, "num_hidden_layers": 27, "num_attention_heads": 16,
    "num_channels": 3, "patch_size": 16, "num_patches": 256, "layer_norm_eps": 1e-6, "hidden_act": "gelu_pytorch_tanh",
    "vision_use_head": False, "downsample_factor": 2, "projector_hidden_size": 2048, "projector_hidden_act": "gelu",
    "projector_bias": True, "projector_use_layernorm": False, "text_hidden_size": 2048,
}

# The toy: a small SigLIP2 + projector at the checkpoint's patch size, position grid and activations, with the toy
# decoder's text width (toy_graph_check.toy_config: hidden 64), written as a snapshot laid out like the checkpoint
TOY_SEED = 0
TOY_VISION_CONFIG = {
    "attention_dropout": 0.0, "hidden_act": "gelu_pytorch_tanh", "hidden_size": 96, "intermediate_size": 384,
    "layer_norm_eps": 1e-06, "model_type": "siglip2_vision_model", "num_attention_heads": 4, "num_channels": 3,
    "num_hidden_layers": 3, "num_patches": 256, "patch_size": 16, "vision_use_head": False,
}
TOY_TEXT_HIDDEN = 64
TOY_TEXT_CONFIG = {   # toy_graph_check.toy_config in config.json form (ids folded into the toy's 256 tokens)
    "hidden_size": TOY_TEXT_HIDDEN, "block_dim": TOY_TEXT_HIDDEN, "conv_dim": TOY_TEXT_HIDDEN, "intermediate_size": 128,
    "block_auto_adjust_ff_dim": False, "num_hidden_layers": 3, "layer_types": ["conv", "full_attention", "conv"],
    "num_attention_heads": 4, "num_heads": 4, "num_key_value_heads": 2, "vocab_size": 256,
    "max_position_embeddings": 4096, "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2,
}
TOY_SCALES = {"linear_weight": "N(0, 1) / sqrt(in_features)", "linear_bias": "0.02 N(0, 1)",
              "layer_norm_weight": "1 + 0.1 N(0, 1)", "layer_norm_bias": "0.05 N(0, 1)",
              "position_embedding": "0.5 N(0, 1)"}

__all__ = ["ATTENTIONS", "DTYPES", "EXPECTED_VISION_CONFIG", "HOST_POSITION_TABLE", "INPUT_NAMES", "Lfm2VlTowerExact",
           "N_PATCHES", "N_TOKENS", "OUTPUT_NAMES", "PROJECTOR_PREFIX", "TOY_SEED", "VISION_PREFIX",
           "checkpoint_key_map", "checkpoint_layout", "hf_name", "toy_checkpoint", "typed", "vision_config_record",
           "write_toy_snapshot"]


# --------------------------------------------------------------------------- names
def checkpoint_layout(vcfg: Lfm2VlVisionConfig) -> list[tuple[str, tuple[int, ...], str]]:
    """Every checkpoint tensor of the tower and the projector: (key, shape, kind), in a fixed order."""
    d, i, pdim = vcfg.hidden_size, vcfg.intermediate_size, vcfg.patch_dim
    out = [(VISION_PREFIX + "embeddings.patch_embedding.weight", (d, pdim), "linear_weight"),
           (VISION_PREFIX + "embeddings.patch_embedding.bias", (d,), "linear_bias"),
           (POSITION_KEY, (vcfg.num_patches, d), "position_embedding")]
    for n in range(vcfg.num_hidden_layers):
        p = f"{VISION_PREFIX}encoder.layers.{n}."
        out += [(p + "layer_norm1.weight", (d,), "layer_norm_weight"), (p + "layer_norm1.bias", (d,), "layer_norm_bias")]
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            out += [(p + f"self_attn.{proj}.weight", (d, d), "linear_weight"),
                    (p + f"self_attn.{proj}.bias", (d,), "linear_bias")]
        out += [(p + "layer_norm2.weight", (d,), "layer_norm_weight"), (p + "layer_norm2.bias", (d,), "layer_norm_bias"),
                (p + "mlp.fc1.weight", (i, d), "linear_weight"), (p + "mlp.fc1.bias", (i,), "linear_bias"),
                (p + "mlp.fc2.weight", (d, i), "linear_weight"), (p + "mlp.fc2.bias", (d,), "linear_bias")]
    out += [(VISION_PREFIX + "post_layernorm.weight", (d,), "layer_norm_weight"),
            (VISION_PREFIX + "post_layernorm.bias", (d,), "layer_norm_bias")]
    f2 = vcfg.downsample_factor ** 2
    out += [(PROJECTOR_PREFIX + "linear_1.weight", (vcfg.projector_hidden_size, d * f2), "linear_weight"),
            (PROJECTOR_PREFIX + "linear_1.bias", (vcfg.projector_hidden_size,), "linear_bias"),
            (PROJECTOR_PREFIX + "linear_2.weight", (vcfg.text_hidden_size, vcfg.projector_hidden_size), "linear_weight"),
            (PROJECTOR_PREFIX + "linear_2.bias", (vcfg.text_hidden_size,), "linear_bias")]
    return out


def checkpoint_key_map(keys: list[str]) -> dict[str, str]:
    """Checkpoint keys under the two prefixes -> this module's names (HOST_POSITION_TABLE for the position table)."""
    out = {}
    for k in keys:
        if k == POSITION_KEY:
            out[k] = HOST_POSITION_TABLE
        elif k.startswith(VISION_PREFIX + "embeddings.patch_embedding."):
            out[k] = "patch_embedding." + k.rsplit(".", 1)[1]
        elif k.startswith(VISION_PREFIX + "encoder.layers."):
            out[k] = "layers." + k.removeprefix(VISION_PREFIX + "encoder.layers.")
        elif k.startswith(VISION_PREFIX + "post_layernorm."):
            out[k] = k.removeprefix(VISION_PREFIX)
        elif k.startswith(PROJECTOR_PREFIX):
            out[k] = k.removeprefix(PROJECTOR_PREFIX)
    return out


def hf_name(key: str, base_model: bool = False) -> str:
    """A checkpoint key -> transformers 5.19's module name (Lfm2VlForConditionalGeneration, or Lfm2VlModel when
    `base_model`): the `vision_model.` segment dropped (PrefixChange), nothing else renamed."""
    name = key.replace("model.vision_tower.vision_model.", "model.vision_tower.", 1)
    return name.removeprefix("model.") if base_model else name


def vision_config_record(raw: dict, vcfg: Lfm2VlVisionConfig) -> dict:
    """The fields EXPECTED_VISION_CONFIG names, read off config.json and the overlay's parsed vision config."""
    v = raw["vision_config"]
    return {"hidden_size": vcfg.hidden_size, "intermediate_size": vcfg.intermediate_size,
            "num_hidden_layers": vcfg.num_hidden_layers, "num_attention_heads": vcfg.num_attention_heads,
            "num_channels": vcfg.num_channels, "patch_size": vcfg.patch_size, "num_patches": vcfg.num_patches,
            "layer_norm_eps": vcfg.layer_norm_eps, "hidden_act": v.get("hidden_act"),
            "vision_use_head": bool(v.get("vision_use_head", True)), "downsample_factor": vcfg.downsample_factor,
            "projector_hidden_size": vcfg.projector_hidden_size, "projector_hidden_act": raw.get("projector_hidden_act"),
            "projector_bias": vcfg.projector_bias, "projector_use_layernorm": bool(raw.get("projector_use_layernorm")),
            "text_hidden_size": vcfg.text_hidden_size}


# --------------------------------------------------------------------------- the module
class _ExactAttention(nn.Module):
    """SigLIP2 attention over every patch row with an additive per-key bias (0 = attend, -inf = padding)."""

    def __init__(self, vcfg: Lfm2VlVisionConfig, attention: str) -> None:
        super().__init__()
        if attention not in ATTENTIONS:
            raise ValueError(f"attention {attention!r} not in {ATTENTIONS}")
        self.num_heads = vcfg.num_attention_heads
        self.head_dim = vcfg.head_dim
        self.scale = self.head_dim ** -0.5
        d = vcfg.hidden_size
        self.q_proj = nn.Linear(d, d, bias=True)
        self.k_proj = nn.Linear(d, d, bias=True)
        self.v_proj = nn.Linear(d, d, bias=True)
        self.out_proj = nn.Linear(d, d, bias=True)
        self.attention = attention
        if attention == "sdpa":
            self.sdpa = SDPA(scale=self.scale, is_causal=False)

    def forward(self, x: torch.Tensor, key_bias: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        shape = (1, n, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)              # [1, heads, n, hd]
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        if self.attention == "sdpa":
            out = self.sdpa(q, k, v, attn_mask=(key_bias == 0).reshape(1, 1, 1, n))
        else:
            scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale + key_bias
            out = torch.matmul(torch.softmax(scores, dim=-1), v)
        out = out.transpose(1, 2).reshape(n, self.num_heads * self.head_dim)
        return self.out_proj(out)


class _ExactBlock(nn.Module):
    def __init__(self, vcfg: Lfm2VlVisionConfig, attention: str) -> None:
        super().__init__()
        eps = vcfg.layer_norm_eps
        self.layer_norm1 = nn.LayerNorm(vcfg.hidden_size, eps=eps)
        self.self_attn = _ExactAttention(vcfg, attention)
        self.layer_norm2 = nn.LayerNorm(vcfg.hidden_size, eps=eps)
        self.mlp = _VisionMLP(vcfg)                                 # fc2(gelu_tanh(fc1(x)))

    def forward(self, x: torch.Tensor, key_bias: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x), key_bias)
        return x + self.mlp(self.layer_norm2(x))


class Lfm2VlTowerExact(nn.Module):
    """LFM2-VL vision tower + projector for any crop the processor makes; contract in the header."""

    def __init__(self, vcfg: Lfm2VlVisionConfig, text_hidden: int | None = None, n_patches: int = N_PATCHES,
                 n_tokens: int = N_TOKENS, attention: str = "matmul") -> None:
        super().__init__()
        f = vcfg.downsample_factor
        text_hidden = vcfg.text_hidden_size if text_hidden is None else text_hidden
        if text_hidden != vcfg.text_hidden_size:
            raise ValueError(f"text_hidden {text_hidden} != the config's {vcfg.text_hidden_size}")
        if n_tokens * f * f != n_patches:
            raise ValueError(f"{n_tokens} tokens x {f * f} != {n_patches} patches")
        self.vcfg = vcfg
        self.text_hidden = text_hidden
        self.n_patches, self.n_tokens = n_patches, n_tokens
        self.attention = attention
        self.patch_embedding = nn.Linear(vcfg.patch_dim, vcfg.hidden_size, bias=True)
        self.layers = nn.ModuleList([_ExactBlock(vcfg, attention) for _ in range(vcfg.num_hidden_layers)])
        self.post_layernorm = nn.LayerNorm(vcfg.hidden_size, eps=vcfg.layer_norm_eps)
        self.linear_1 = nn.Linear(vcfg.hidden_size * f * f, vcfg.projector_hidden_size, bias=vcfg.projector_bias)
        self.linear_2 = nn.Linear(vcfg.projector_hidden_size, text_hidden, bias=vcfg.projector_bias)
        # the checkpoint's 16 x 16 position table, fp32: the host's, not a graph weight (a plain attribute, so
        # neither export nor a dtype cast touches it)
        self.host_position_table: torch.Tensor | None = None
        self.load_report: dict | None = None

    def forward(
        self,
        patches: torch.Tensor,        # [n_patches, patch_dim]
        pos_table: torch.Tensor,      # [n_patches, hidden]
        key_bias: torch.Tensor,       # [n_patches]
        unshuffle_idx: torch.Tensor,  # [n_tokens, 4] int32
    ) -> torch.Tensor:
        """-> image_embeds [n_tokens, text_hidden]."""
        x = self.patch_embedding(patches) + pos_table
        for layer in self.layers:
            x = layer(x, key_bias)
        x = self.post_layernorm(x)
        u = x.index_select(0, unshuffle_idx.reshape(-1)).reshape(self.n_tokens, -1)
        return self.linear_2(F.gelu(self.linear_1(u)))

    # -- export -------------------------------------------------------------

    def build_export_spec(self, input_dtype: torch.dtype) -> dict:
        """Every shape static: one function, no state."""
        d = self.vcfg.hidden_size
        ref = {"patches": torch.zeros(self.n_patches, self.vcfg.patch_dim, dtype=input_dtype),
               "pos_table": torch.zeros(self.n_patches, d, dtype=input_dtype),
               "key_bias": torch.zeros(self.n_patches, dtype=input_dtype),
               "unshuffle_idx": torch.zeros(self.n_tokens, 4, dtype=torch.int32)}
        return {"reference_inputs": ref, "dynamic_shapes": {k: None for k in ref}, "input_names": INPUT_NAMES,
                "output_names": OUTPUT_NAMES, "state_names": ()}

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_hf(cls, hf_id_or_dir: str, attention: str = "matmul", n_patches: int = N_PATCHES,
                n_tokens: int = N_TOKENS) -> "Lfm2VlTowerExact":
        """The checkpoint's `model.vision_tower.vision_model.*` + `model.multi_modal_projector.*` in fp32 (the
        overlay's loader, bf16 -> fp32 exact), every key accounted for in `load_report`; the position table goes
        to `host_position_table`. Cast with `typed()`."""
        raw, vsd = load_lfm2_vl_state_dict(hf_id_or_dir, VISION_PREFIX, torch.float32)
        _, psd = load_lfm2_vl_state_dict(hf_id_or_dir, PROJECTOR_PREFIX, torch.float32)
        vcfg, _ = lfm2_vl_configs_from_dict(raw)
        sd = {VISION_PREFIX + k: t for k, t in vsd.items()} | {PROJECTOR_PREFIX + k: t for k, t in psd.items()}
        model = cls(vcfg, attention=attention, n_patches=n_patches, n_tokens=n_tokens).float()
        names = checkpoint_key_map(list(sd))
        out = {names[k]: t for k, t in sd.items() if k in names and names[k] != HOST_POSITION_TABLE}
        own = {n: tuple(p.shape) for n, p in model.state_dict().items()}
        shape_bad = sorted(n for n, t in out.items() if n in own and tuple(t.shape) != own[n])
        missing, unexpected = model.load_state_dict(out, strict=False, assign=True)
        table = sd.get(POSITION_KEY)
        model.host_position_table = None if table is None else table.float().contiguous()
        model.eval()
        layout = {k: s for k, s, _ in checkpoint_layout(vcfg)}
        model.load_report = {
            "snapshot": str(_snapshot(hf_id_or_dir)),
            "checkpoint_keys_under_prefixes": len(sd),
            "expected_keys": len(layout),
            "unmapped_checkpoint_keys": sorted(k for k in sd if k not in names),
            "unexpected_keys": sorted(unexpected), "missing_module_tensors": sorted(missing),
            "shape_mismatches": shape_bad,
            "layout_differs": sorted(k for k in set(layout) ^ set(sd)) + sorted(
                k for k in layout if k in sd and tuple(sd[k].shape) != tuple(layout[k])),
            "host_position_table": None if table is None else list(table.shape),
            "config": vision_config_record(raw, vcfg),
            "attention": attention,
        }
        return model

    @classmethod
    def toy(cls, seed: int = TOY_SEED, snapshot_dir: str | Path | None = None, attention: str = "matmul",
            template_config: str | Path | None = None) -> "Lfm2VlTowerExact":
        """The toy through `from_hf`: its snapshot (config.json + model.safetensors laid out like the checkpoint) is
        written first when `snapshot_dir` does not hold one (`template_config` = the real config.json)."""
        snap = Path(snapshot_dir)
        if not (snap / "model.safetensors").exists():
            if template_config is None:
                raise FileNotFoundError(f"no toy snapshot in {snap} and no template config.json to write one")
            write_toy_snapshot(snap, seed, template_config)
        meta = json.loads((snap / "name_table.json").read_text())
        if int(meta["toy"]["seed"]) != seed:
            raise ValueError(f"{snap} holds the seed-{meta['toy']['seed']} toy, not seed {seed}")
        return cls.from_hf(str(snap), attention=attention)


def typed(model: Lfm2VlTowerExact, dtype: str) -> Lfm2VlTowerExact:
    """fp32 (as loaded), fp16 (every weight and the math fp16), fp16w32 (weights stored fp16 and read through a
    .float() cast, the math fp32: export_qwen38vl_pipelined.fp16_storage_fp32_compute)."""
    if dtype not in DTYPES:
        raise ValueError(f"dtype {dtype!r} not in {DTYPES}")
    m = copy.deepcopy(model).float().eval()
    m.host_position_table = model.host_position_table
    m.load_report = model.load_report
    if dtype == "fp16":
        return m.half()
    if dtype == "fp16w32":
        from export_qwen38vl_pipelined import fp16_storage_fp32_compute
        return fp16_storage_fp32_compute(m)
    return m


def input_dtype(dtype: str) -> torch.dtype:
    """The graph's float inputs and output: fp16 for fp16, fp32 otherwise."""
    return torch.float16 if dtype == "fp16" else torch.float32


# --------------------------------------------------------------------------- the toy snapshot
def toy_vcfg() -> Lfm2VlVisionConfig:
    v = TOY_VISION_CONFIG
    return Lfm2VlVisionConfig(hidden_size=v["hidden_size"], intermediate_size=v["intermediate_size"],
                              num_hidden_layers=v["num_hidden_layers"], num_attention_heads=v["num_attention_heads"],
                              num_channels=v["num_channels"], patch_size=v["patch_size"], num_patches=v["num_patches"],
                              layer_norm_eps=v["layer_norm_eps"], downsample_factor=2,
                              projector_hidden_size=TOY_TEXT_HIDDEN, projector_bias=True,
                              projector_use_layernorm=False, text_hidden_size=TOY_TEXT_HIDDEN)


def bf16_round(a: np.ndarray) -> np.ndarray:
    """float32 -> the nearest bfloat16 value (ties to even), kept as float32 (finite inputs)."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    u = (u + (((u >> 16) & 1) + 0x7FFF)) & np.uint32(0xFFFF0000)
    return u.astype(np.uint32).view(np.float32)


def toy_checkpoint(seed: int = TOY_SEED) -> dict[str, np.ndarray]:
    """The toy's tensors under the checkpoint's names, drawn by NumPy (version-stable) in layout order and rounded
    to bfloat16 values like the checkpoint's (so fp16 storage holds them exactly, as it holds the model's), float32."""
    rng = np.random.default_rng(seed)
    out = {}
    for key, shape, kind in checkpoint_layout(toy_vcfg()):
        z = rng.standard_normal(shape)
        if kind == "linear_weight":
            a = z / math.sqrt(shape[1])
        elif kind == "linear_bias":
            a = 0.02 * z
        elif kind == "layer_norm_weight":
            a = 1.0 + 0.1 * z
        elif kind == "layer_norm_bias":
            a = 0.05 * z
        else:
            a = 0.5 * z
        out[key] = np.ascontiguousarray(bf16_round(a.astype(np.float32)))
    return out


def toy_config_json(template: dict) -> dict:
    """The real config.json with the toy's vision config, projector width and text sizes (ids folded into the toy's
    vocabulary: pad 0, bos 1, eos 2; the image token id kept, as get_image_features never reads it)."""
    cfg = json.loads(json.dumps(template))
    cfg.pop("auto_map", None)                     # the toy is transformers' own Lfm2VlForConditionalGeneration
    cfg["vision_config"] = dict(TOY_VISION_CONFIG)
    cfg["projector_hidden_size"] = TOY_TEXT_HIDDEN
    cfg["text_config"].update(TOY_TEXT_CONFIG)
    cfg["text_config"]["full_attn_idxs"] = None
    for k in ("pad_token_id", "bos_token_id", "eos_token_id"):
        cfg[k] = TOY_TEXT_CONFIG[k]
    cfg["dtype"] = "float32"
    cfg["text_config"]["dtype"] = "float32"
    cfg["vision_config"]["dtype"] = "float32"
    return cfg


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def name_table(vcfg: Lfm2VlVisionConfig, shapes: dict[str, tuple[int, ...]] | None = None) -> list[dict]:
    """Per checkpoint key: shape, this module's name, transformers 5.19's names, the overlay encoder's name."""
    names = checkpoint_key_map([k for k, _, _ in checkpoint_layout(vcfg)])
    rows = []
    for key, shape, kind in checkpoint_layout(vcfg):
        overlay = (None if key == POSITION_KEY else names[key])
        rows.append({"checkpoint": key, "shape": list(shapes[key] if shapes else shape), "kind": kind,
                     "exact_module": names[key], "hf_Lfm2VlForConditionalGeneration": hf_name(key),
                     "hf_Lfm2VlModel": hf_name(key, base_model=True),
                     "overlay_Lfm2VlVisionEncoder": overlay if key != POSITION_KEY else
                     "baked: _init_positional_constants (fixed grid)"})
    return rows


def write_toy_snapshot(out_dir: str | Path, seed: int, template_config: str | Path) -> dict:
    """config.json + model.safetensors (the checkpoint's names and dtype, BF16) + name_table.json; never overwrites."""
    from safetensors.torch import save_file

    out = Path(out_dir)
    if (out / "model.safetensors").exists() or (out / "config.json").exists():
        raise FileExistsError(f"{out} already holds a toy snapshot: it is never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    template = json.loads(Path(template_config).read_text())
    cfg = toy_config_json(template)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    tensors = toy_checkpoint(seed)
    bf16 = {k: torch.from_numpy(v).to(torch.bfloat16) for k, v in tensors.items()}
    if not all(np.array_equal(bf16[k].float().numpy(), v) for k, v in tensors.items()):
        raise AssertionError("a toy value is not a bfloat16 value")
    save_file(bf16, str(out / "model.safetensors"), metadata={"format": "pt"})
    vcfg = toy_vcfg()
    doc = {"schema": "d1-vision-name-table/1",
           "rule": {"checkpoint": "tf 4.57 names as saved: model.vision_tower.vision_model.* and "
                                  "model.multi_modal_projector.*",
                    "transformers_5": "Siglip2VisionModel loads through CLIPVisionModel's PrefixChange('vision_model') "
                                      "(transformers/conversion_mapping.py, 'Siglip2VisionModel': 'CLIPVisionModel' and "
                                      "'CLIPVisionModel': [PrefixChange(prefix_to_remove='vision_model')]): "
                                      "model.vision_tower.vision_model.X -> model.vision_tower.X; the projector is "
                                      "not renamed",
                    "exact_module": "embeddings.patch_embedding.* -> patch_embedding.*, encoder.layers.* -> layers.*, "
                                    "post_layernorm.* as is, the projector's linear_1 / linear_2 as is; "
                                    "embeddings.position_embedding.weight -> the host's table"},
           "keys": name_table(vcfg),
           "toy": {"seed": seed, "draw": "numpy.random.default_rng(seed).standard_normal per key in this table's "
                                         "order, scaled per kind, rounded to bfloat16 (ties to even) and stored BF16 "
                                         "like the checkpoint", "scales": TOY_SCALES,
                   "vision_config": TOY_VISION_CONFIG, "projector_hidden_size": TOY_TEXT_HIDDEN,
                   "text_config": TOY_TEXT_CONFIG,
                   "sha256": {k: sha256_array(v) for k, v in tensors.items()},
                   "template_config_sha256": hashlib.sha256(Path(template_config).read_bytes()).hexdigest()},
           "files": {"config.json": hashlib.sha256((out / "config.json").read_bytes()).hexdigest(),
                     "model.safetensors": hashlib.sha256((out / "model.safetensors").read_bytes()).hexdigest()}}
    (out / "name_table.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


# --------------------------------------------------------------------------- the scout (no weights)
def scout(config_json: Path, header_json: Path | None = None) -> dict:
    """The module against d1-3B's config with no weights: the config asserts, the module on the meta device, the
    export spec, and — with the safetensors header — every checkpoint key under the two prefixes mapped onto the
    module (or the host's table) with its shape, and the layout this file expects equal to the header's."""
    raw = json.loads(Path(config_json).read_text())
    vcfg, _ = lfm2_vl_configs_from_dict(raw)
    got = vision_config_record(raw, vcfg)
    bad = {k: (got[k], v) for k, v in EXPECTED_VISION_CONFIG.items() if got[k] != v}
    if bad:
        raise AssertionError(f"vision config differs from the expected d1-3B one: {bad}")
    with torch.device("meta"):
        model = Lfm2VlTowerExact(vcfg)
    params = {n: list(p.shape) for n, p in model.named_parameters()}
    spec = model.build_export_spec(torch.float16)
    out = {"config_json": str(config_json), "vision_config": got, "asserts": {k: "ok" for k in EXPECTED_VISION_CONFIG},
           "module": {"class": f"{Lfm2VlTowerExact.__module__}.{Lfm2VlTowerExact.__qualname__}",
                      "parameters": len(params), "parameter_count": int(sum(p.numel() for p in model.parameters())),
                      "head_dim": vcfg.head_dim},
           "export_spec": {k: [list(v.shape), str(v.dtype).replace("torch.", "")]
                           for k, v in spec["reference_inputs"].items()},
           "output": [[model.n_tokens, model.text_hidden]]}
    if header_json is not None:
        header = json.loads(Path(header_json).read_text())
        tensors = {k: v for k, v in header.items() if k.startswith((VISION_PREFIX, PROJECTOR_PREFIX))}
        names = checkpoint_key_map(list(tensors))
        layout = {k: list(s) for k, s, _ in checkpoint_layout(vcfg)}
        out["checkpoint_header_check"] = {
            "header_json": str(header_json), "tensors_under_prefixes": len(tensors),
            "dtypes": sorted({v["dtype"] for v in tensors.values()}),
            "mapped_to_module": sum(1 for k, n in names.items() if n in params),
            "mapped_to_host_table": [k for k, n in names.items() if n == HOST_POSITION_TABLE],
            "unmapped": sorted(k for k in tensors if k not in names),
            "module_params_not_in_checkpoint": sorted(set(params) - set(names.values())),
            "shape_mismatches": sorted(k for k, n in names.items() if n in params and list(tensors[k]["shape"]) != params[n]),
            "layout_equal_header": layout == {k: list(v["shape"]) for k, v in tensors.items()},
            "hf_tf519_names_example": {k: hf_name(k) for k in list(tensors)[:3]},
        }
    return out


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="d1 exact vision tower: the scout (no weights) and the toy snapshot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scout", help="the module against d1-3B's config.json and safetensors header")
    s.add_argument("config_json")
    s.add_argument("header_json", nargs="?", default=None)
    s.add_argument("--out", required=True)
    w = sub.add_parser("write-toy", help="the toy snapshot: config.json + model.safetensors + name_table.json")
    w.add_argument("--seed", type=int, default=TOY_SEED)
    w.add_argument("--template-config", required=True, help="the real config.json (d1-3B @ da1fe36a)")
    w.add_argument("--out", required=True, help="a new directory")
    args = ap.parse_args()
    if args.cmd == "scout":
        rec = scout(Path(args.config_json), Path(args.header_json) if args.header_json else None)
        if Path(args.out).exists():
            sys.exit(f"{args.out} exists: never overwritten")
        Path(args.out).write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k != "export_spec"}, indent=1)[:3000])
    else:
        doc = write_toy_snapshot(args.out, args.seed, args.template_config)
        print(f"toy snapshot (seed {args.seed}): {args.out} — {len(doc['keys'])} tensors, model.safetensors sha256 "
              f"{doc['files']['model.safetensors'][:16]}")
