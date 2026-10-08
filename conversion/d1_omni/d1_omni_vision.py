#!/usr/bin/env python3
"""Static, plain-PyTorch authoring graph for d1-omni-600M's vision path: one crop through the SigLIP2 NaFlex tower and
the 2x2 pixel-unshuffle projector, out as that crop's prefix embeddings.

Re-authored from the publisher's `vision.py` (LiquidAI/d1-omni-600M @ 414f8d64, sha256 in d1_omni_model.SOURCE_SHA256)
and the class it instantiates, transformers 5.19's `Siglip2VisionModel` (sdpa attention), from the tensor names of
model.safetensors; it imports neither transformers nor the publisher's code.

  main  pixel_values [1,1024,768] float32     the crop's 16x16 patches, row-major over the (ph, pw) patch grid, each
                                              patch [py][px][c] (channel fastest), RGB in [-1, 1]; 0.0 past ph*pw
        pos_embed [1,1024,768] float32        the 16x16 position table resized to (ph, pw) (bilinear, antialias, fp32),
                                              row-major; 0.0 past ph*pw
        patch_mask [1,1024] float32           1.0 on the crop's ph*pw patches, 0.0 after
        unshuffle_index [256,4] int32         token t = (i, j), t = i * (pw/2) + j: the patches 2i*pw + 2j,
                                              2i*pw + 2j+1, (2i+1)*pw + 2j, (2i+1)*pw + 2j+1; 0 past (ph/2)(pw/2)
        -> prefix [1,256,1024] float32        rows [0, (ph/2)(pw/2)) are the crop's prefix; the rest are not read

The host (host.py: image_crops_inputs, vision_prefix) cuts the image into the publisher's crops (vision.preprocess:
the 512 px tiles of a large image, then the thumbnail), builds these four inputs per crop, and concatenates each
crop's first (ph/2)(pw/2) rows in crop order; images in request order.

What the checkpoint runs (vision.py and Siglip2VisionModel, read in full):
  - embeddings: patch_embedding = Linear(768, 768) over the flattened patches; + the 16x16 position table
    (position_embedding [256, 768]) resized per crop to (ph, pw) with F.interpolate(bilinear, align_corners=False,
    antialias=True) in fp32 on the CPU (resize_positional_embeddings; pad rows get the first resized row)
  - 12 encoder layers, pre-LN (eps 1e-6): h += out_proj(attention(layer_norm1(h))); h += fc2(gelu_tanh(fc1(
    layer_norm2(h)))); 12 heads of 64, scale 64**-0.5, bidirectional, the pad patches masked as keys
  - post_layernorm (vision_use_head false: no pooling head)
  - projector (vision.Projector): the crop's [1, ph, pw, 768] hidden states pixel-unshuffled by 2 (the reshape /
    permute pair above = the gather by unshuffle_index), linear_1 [2048, 3072] -> gelu (exact erf) -> linear_2
    [1024, 2048]

How this graph departs from that code, and why each departure is exact:
  - the position resize is on the host, not in the graph: it depends only on the crop's (ph, pw), and the host does
    it with the same call (F.interpolate, fp32 CPU). Pad rows hold 0.0 instead of the first resized row: a pad
    patch is masked as a key and its own output is never read, so the real rows do not depend on it.
  - attention is matmul -> softmax (fp32) -> matmul with an additive mask, MASK = -1e4 on pad keys (exp(-1e4 + ...)
    = 0 in fp32 and fp16, as with sdpa's boolean mask); the scale 0.125 is a power of two. 1,024 keys: the plain
    chain (the Mac GPU's 4,032-key limit, d1_omni_model "Key blocks", is not reached).
  - the unshuffle is a row gather (F.embedding of the hidden states by unshuffle_index) and a reshape to 4 x 768:
    the same four rows in the same order as the publisher's reshape / permute / reshape / permute, so the same
    numbers (vision_check.py proves the order bit for bit against the publisher's Projector).

Precision (`set_precision`): fp32 is the reference. wfp16 = fp16 weight storage with fp32 compute (each weight cast
at its use); fp16 = fp16 compute with every LayerNorm, the attention softmax and the projector's gelu in fp32.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

import d1_omni_model as dm

MAX_PATCHES = 1024       # vision.MAX_PATCHES: patches per crop, padded
MAX_TOKENS = 256         # MAX_PATCHES / 4: prefix rows per crop
PATCH = 16
MASK_VALUE = -1.0e4
PRECISIONS = ("fp32", "wfp16", "fp16")
# checkpoint prefix -> this module's prefix; the position table is the host's (POSITION_KEY)
KEY_MAP = (("vision.tower.vision_model.embeddings.patch_embedding.", "patch_embedding."),
           ("vision.tower.vision_model.encoder.layers.", "layers."),
           ("vision.tower.vision_model.post_layernorm.", "post_layernorm."),
           ("vision.projector.", "projector."))
POSITION_KEY = "vision.tower.vision_model.embeddings.position_embedding.weight"
CHECKPOINT_KEYS = 201    # 197 tower (the position table included) + 4 projector
DERIVED = {"head_dim": 64, "patch_dim": 768, "position_grid": 16, "projector_in": 3072, "projector_hidden": 2048,
           "out": 1024}


def derived(config: dict) -> dict:
    v = config["vision_config"]
    return {"head_dim": v["hidden_size"] // v["num_attention_heads"],
            "patch_dim": v["num_channels"] * v["patch_size"] ** 2,
            "position_grid": int(round(v["num_patches"] ** 0.5)),
            "projector_in": v["hidden_size"] * 4, "projector_hidden": config["projector_hidden_size"],
            "out": config["text_config"]["hidden_size"]}


def validate_config(config: dict) -> None:
    """The whole config.json, key for key (d1_omni_model.validate_config: vision_config included), and the vision
    sizes this graph is written for."""
    dm.validate_config(config)
    got = derived(config)
    v = config["vision_config"]
    problems = [f"{k}: {got[k]} != {DERIVED[k]}" for k in DERIVED if got[k] != DERIVED[k]]
    if v["hidden_act"] != "gelu_pytorch_tanh" or v["vision_use_head"] or v["patch_size"] != PATCH:
        problems.append(f"hidden_act {v['hidden_act']} / vision_use_head {v['vision_use_head']} / patch {v['patch_size']}")
    if got["position_grid"] ** 2 != v["num_patches"]:
        problems.append("num_patches is not a square")
    if problems:
        raise ValueError("Unsupported d1-omni vision configuration:\n  " + "\n  ".join(problems))


# --------------------------------------------------------------------------- modules (names = checkpoint names)
class _Attention(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(d, d) for _ in range(4))


class _MLP(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, hidden), nn.Linear(hidden, d)


class _Layer(nn.Module):
    def __init__(self, d: int, hidden: int, eps: float):
        super().__init__()
        self.layer_norm1, self.layer_norm2 = nn.LayerNorm(d, eps=eps), nn.LayerNorm(d, eps=eps)
        self.self_attn = _Attention(d)
        self.mlp = _MLP(d, hidden)


class _Projector(nn.Module):
    def __init__(self, d_in: int, hidden: int, out: int):
        super().__init__()
        self.linear_1, self.linear_2 = nn.Linear(d_in, hidden), nn.Linear(hidden, out)


class D1Vision(nn.Module):
    """Tower + projector for one crop (the module docstring's `main`). Parameter names are the checkpoint's under
    KEY_MAP; the position table is held for the host (`position_table`), not used by forward."""

    def __init__(self, config: dict):
        super().__init__()
        v = config["vision_config"]
        d = v["hidden_size"]
        self.hidden, self.heads = d, v["num_attention_heads"]
        self.head_dim = d // self.heads
        self.patch_embedding = nn.Linear(v["num_channels"] * v["patch_size"] ** 2, d)
        self.layers = nn.ModuleList(_Layer(d, v["intermediate_size"], v["layer_norm_eps"])
                                    for _ in range(v["num_hidden_layers"]))
        self.post_layernorm = nn.LayerNorm(d, eps=v["layer_norm_eps"])
        self.projector = _Projector(4 * d, config["projector_hidden_size"], config["text_config"]["hidden_size"])
        self.out = config["text_config"]["hidden_size"]
        self.precision, self.compute_dtype = "fp32", torch.float32
        self.float_dtype = torch.float32  # LayerNorm / softmax / projector gelu / output: fp32 in every export precision
        self.position_table = None  # [16, 16, 768] fp32, set by load_weights (a plain attribute: not a graph weight)

    # ---- precision
    def set_precision(self, precision: str) -> "D1Vision":
        """fp32 / wfp16 / fp16 (module docstring). LayerNorm weights stay fp32 tensors in every mode."""
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        storage = torch.float32 if precision == "fp32" else torch.float16
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                continue
            for param in module.parameters(recurse=False):
                param.data = param.data.to(storage)
        self.precision = precision
        self.compute_dtype = torch.float16 if precision == "fp16" else torch.float32
        return self

    def as_float64(self) -> "D1Vision":
        """A diagnostic, never exported: every weight and every op in float64 (the fp32-kept ops included)."""
        self.double()
        self.precision, self.compute_dtype, self.float_dtype = "float64", torch.float64, torch.float64
        return self

    # ---- pieces (each weight is cast to the activation dtype at its use: a no-op in fp32)
    @staticmethod
    def _lin(x: torch.Tensor, module: nn.Linear) -> torch.Tensor:
        return F.linear(x, module.weight.to(x.dtype), module.bias.to(x.dtype))

    def _ln(self, norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        f = self.float_dtype
        y = F.layer_norm(x.to(f), norm.normalized_shape, norm.weight.to(f), norm.bias.to(f), norm.eps)
        return y.to(x.dtype)

    def _attention(self, attn: _Attention, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n, heads, hd = x.shape[1], self.heads, self.head_dim
        q = self._lin(x, attn.q_proj).reshape(1, n, heads, hd).transpose(1, 2)
        k = self._lin(x, attn.k_proj).reshape(1, n, heads, hd).transpose(1, 2)
        v = self._lin(x, attn.v_proj).reshape(1, n, heads, hd).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)).to(self.float_dtype) * (hd ** -0.5) + mask
        y = torch.matmul(torch.softmax(scores, dim=-1).to(v.dtype), v)
        return self._lin(y.transpose(1, 2).reshape(1, n, heads * hd), attn.out_proj)

    def tower(self, pixel_values, pos_embed, patch_mask) -> torch.Tensor:
        """The NaFlex tower through post_layernorm: [1, 1024, 768] in the compute dtype (pad rows are not read)."""
        dtype = self.compute_dtype
        x = self._lin(pixel_values.to(dtype), self.patch_embedding) + pos_embed.to(dtype)
        mask = ((1.0 - patch_mask.to(self.float_dtype)) * MASK_VALUE).reshape(1, 1, 1, MAX_PATCHES)
        for layer in self.layers:
            x = x + self._attention(layer.self_attn, self._ln(layer.layer_norm1, x), mask)
            x = x + self._lin(F.gelu(self._lin(self._ln(layer.layer_norm2, x), layer.mlp.fc1), approximate="tanh"),
                              layer.mlp.fc2)
        return self._ln(self.post_layernorm, x)

    def project(self, hidden: torch.Tensor, unshuffle_index: torch.Tensor) -> torch.Tensor:
        """[1, 1024, 768] -> the gather by unshuffle_index -> [1, 256, 3072] -> linear_1 -> gelu (erf, fp32) ->
        linear_2 -> [1, 256, out] float32."""
        rows = F.embedding(unshuffle_index.reshape(1, 4 * MAX_TOKENS), hidden.reshape(MAX_PATCHES, self.hidden))
        u = self._lin(rows.reshape(1, MAX_TOKENS, 4 * self.hidden), self.projector.linear_1)
        u = F.gelu(u.to(self.float_dtype)).to(u.dtype)
        return self._lin(u, self.projector.linear_2).to(self.float_dtype)

    def forward(self, pixel_values, pos_embed, patch_mask, unshuffle_index):
        for name, t, shape in (("pixel_values", pixel_values, (1, MAX_PATCHES, self.patch_embedding.in_features)),
                               ("pos_embed", pos_embed, (1, MAX_PATCHES, self.hidden)),
                               ("patch_mask", patch_mask, (1, MAX_PATCHES)),
                               ("unshuffle_index", unshuffle_index, (MAX_TOKENS, 4))):
            if tuple(t.shape) != shape:
                raise ValueError(f"{name}: expected {shape}, got {tuple(t.shape)}")
        return self.project(self.tower(pixel_values, pos_embed, patch_mask), unshuffle_index)


# --------------------------------------------------------------------------- weights
def map_key(checkpoint_key: str) -> str | None:
    """A vision checkpoint key -> this module's key; None for the position table (the host's)."""
    if checkpoint_key == POSITION_KEY:
        return None
    for source, target in KEY_MAP:
        if checkpoint_key.startswith(source):
            return target + checkpoint_key[len(source):]
    raise KeyError(f"unexpected vision key {checkpoint_key!r}")


def load_weights(model: D1Vision, safetensors_path: str | Path, verify_sha256: bool = False) -> dict:
    """Strict load of the vision tensors (fp32) from model.safetensors (trunk, head and audio skipped and counted);
    the position table goes to model.position_table [16, 16, 768]."""
    from safetensors import safe_open

    path = Path(safetensors_path)
    if path.stat().st_size != dm.WEIGHTS["bytes"]:
        raise ValueError(f"{path}: {path.stat().st_size} bytes, expected {dm.WEIGHTS['bytes']}")
    if verify_sha256:
        import hashlib
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 24), b""):
                digest.update(block)
        if digest.hexdigest() != dm.WEIGHTS["sha256"]:
            raise ValueError(f"{path}: sha256 {digest.hexdigest()} != {dm.WEIGHTS['sha256']}")
    state, table, skipped, parameters, keys = {}, None, 0, 0, 0
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if not key.startswith("vision."):
                skipped += 1
                continue
            tensor = handle.get_tensor(key)
            if tensor.dtype != torch.float32:
                raise ValueError(f"{key}: {tensor.dtype}, the checkpoint is float32")
            keys += 1
            parameters += tensor.numel()
            target = map_key(key)
            if target is None:
                table = tensor
            else:
                state[target] = tensor
    if keys != CHECKPOINT_KEYS or parameters != dm.PARAMETERS["vision"] or table is None:
        raise ValueError(f"vision keys {keys} / parameters {parameters}: expected {CHECKPOINT_KEYS} / "
                         f"{dm.PARAMETERS['vision']} with the position table")
    model.load_state_dict(state, strict=True)
    side = int(round(table.shape[0] ** 0.5))
    model.position_table = table.reshape(side, side, -1).contiguous()
    return {"tensors": len(state), "position_table": list(table.shape), "skipped_other": skipped,
            "parameters": parameters, "graph_parameters": sum(p.numel() for p in model.parameters()),
            "sha256_verified": verify_sha256}


def build(config: dict) -> D1Vision:
    validate_config(config)
    return D1Vision(config)


def load_d1_vision(snapshot_dir: str | Path, precision: str = "fp32",
                   verify_sha256: bool = False) -> tuple[D1Vision, dict]:
    """The pinned snapshot's config.json + model.safetensors -> the eval-mode vision graph module."""
    snapshot = Path(snapshot_dir)
    model = build(json.loads((snapshot / "config.json").read_text()))
    record = load_weights(model, snapshot / dm.WEIGHTS["file"], verify_sha256)
    model.eval().requires_grad_(False)
    return model.set_precision(precision), record


def _self_check() -> None:  # python3 d1_omni_vision.py: config + module structure against the pinned files
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _paths import hf_snapshot  # noqa: E402

    snapshot = Path(hf_snapshot(dm.MODEL_ID, revision=dm.MODEL_SHA))
    config: dict[str, Any] = json.loads((snapshot / "config.json").read_text())
    validate_config(config)
    with torch.device("meta"):
        model = build(config)
    print(f"config ok; graph parameters {sum(p.numel() for p in model.parameters()):,} "
          f"(+ the 16x16x768 position table on the host); {len(model.state_dict())} tensors")


if __name__ == "__main__":
    _self_check()
