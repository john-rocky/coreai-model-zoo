#!/usr/bin/env python3
"""Invert the pinned Jev-Style Qwen3.5-2B MLX checkpoint into HF text weights.

MLX stores the 18 ``layers.*.linear_attn.conv1d.weight`` tensors as [6144, 4, 1];
Torch Conv1d expects [6144, 1, 4], so transpose axes 1 and 2. MLX's RMSNorm uses
a direct scale, while Qwen3.5's Torch RMSNormPlusOne multiplies by ``1 + weight``.
Subtract one from 48 input/post-attention norms, 12 full-attention q_norm/k_norm
weights, and the final model norm: 61 tensors. The 18 gated linear_attn.norm
weights already use the same convention and MUST remain unchanged.

Compute and store both transformed groups in fp32. BF16 spacing near 1.0 is
2**-7: the author's weights already carry that rounding, and storing the offset
back in bf16 would add another rounding. All other tensors retain their source
dtype (including the 18 fp32 A_log tensors). Rename language_model.model.* to
model.language_model.*; leave the tied lm_head absent. Preserve the calibration
already folded into the final norm; the probability readout temperature is 1.

This checked tool targets the revision declared below. It verifies the published
checkpoint and tokenizer SHA256 values before opening any tensor, writes every
tensor's before/after key, shape, and dtype to OUT/mlx_to_hf.json, and flattens the
source text_config into a qwen3_5_text config with tie_word_embeddings=true.

Usage: python mlx_to_hf_qwen3_5.py --snapshot PATH --out converted
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HF_ID = "chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16"
REVISION = "69a91b6778df7d11b9e13e0f7d4c6a5aa6ad1bc5"
SOURCE_SHA256 = "95fc80a9c0bb7dbecd29896d67b6de1ba3306870dc7018bb4a240c9d63d00649"
TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
PREFIX_IN = "language_model.model."
PREFIX_OUT = "model.language_model."
NORM_ENDINGS = (
    ".input_layernorm.weight", ".post_attention_layernorm.weight",
    ".self_attn.q_norm.weight", ".self_attn.k_norm.weight",
)


def sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, document: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_snapshot(snapshot: str | Path) -> dict:
    snapshot = Path(snapshot).resolve()
    hashes = {}
    for name, expected in (("model.safetensors", SOURCE_SHA256),
                           ("tokenizer.json", TOKENIZER_SHA256)):
        actual = sha256(snapshot / name)
        if actual != expected:
            raise ValueError(f"{name}: expected sha256 {expected}, got {actual}")
        hashes[name] = actual
    return hashes


def convert_snapshot(snapshot: str | Path, out: str | Path) -> dict:
    """Verify, invert, and save the pinned layout; return the written audit report."""
    started = time.monotonic()
    snapshot, out = Path(snapshot).resolve(), Path(out).resolve()
    if out == snapshot:
        raise ValueError("--out must differ from the read-only source snapshot")
    hashes = verify_snapshot(snapshot)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "RUNNING", "source_hf_id": HF_ID, "source_revision": REVISION,
        "source_snapshot": str(snapshot), "source_sha256": SOURCE_SHA256,
        "source_hashes": hashes, "source_integrity_verified_before_tensor_read": True,
        "output": str(out / "model.safetensors"), "config": str(out / "config.json"),
        "tensors": [], "transposes": [], "offsets": [], "unchanged_gated_norms": [],
        "frozen_loader_modified": False,
    }
    write_json(out / "mlx_to_hf.json", report)
    state = {}
    with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as source:
        keys = list(source.keys())
        assert len(keys) == 320, len(keys)
        for key in keys:
            assert key.startswith(PREFIX_IN), key
            new_key = PREFIX_OUT + key[len(PREFIX_IN):]
            original = source.get_tensor(key)
            assert bool(torch.isfinite(original).all()), key
            converted, operation, roundtrip_error = original, "unchanged", None
            if key.endswith(".linear_attn.conv1d.weight"):
                assert list(original.shape) == [6144, 4, 1], (key, original.shape)
                converted = original.float().transpose(1, 2).contiguous()
                assert torch.equal(converted.transpose(1, 2), original.float())
                operation, roundtrip_error = "transpose_1_2_fp32", 0.0
                report["transposes"].append(new_key)
            elif key == PREFIX_IN + "norm.weight" or key.endswith(NORM_ENDINGS):
                converted = original.float() - 1.0
                operation = "subtract_1_fp32"
                roundtrip_error = float((converted + 1.0 - original.float()).abs().max())
                report["offsets"].append(new_key)
            elif key.endswith(".linear_attn.norm.weight"):
                report["unchanged_gated_norms"].append(new_key)
            state[new_key] = converted.contiguous()
            entry = {
                "before_key": key, "after_key": new_key,
                "before_shape": list(original.shape), "after_shape": list(converted.shape),
                "before_dtype": str(original.dtype).removeprefix("torch."),
                "after_dtype": str(converted.dtype).removeprefix("torch."),
                "operation": operation, "finite": True,
            }
            if roundtrip_error is not None:
                entry["inverse_roundtrip_max_abs_error"] = roundtrip_error
            report["tensors"].append(entry)
    assert len(report["transposes"]) == 18
    assert len(report["offsets"]) == 61
    assert len(report["unchanged_gated_norms"]) == 18
    assert not any("lm_head" in key for key in state)
    assert Counter(str(value.dtype) for value in state.values()) == {
        "torch.bfloat16": 223, "torch.float32": 97,
    }
    checkpoint = out / "model.safetensors"
    temporary = checkpoint.with_suffix(".safetensors.tmp")
    save_file(state, str(temporary), metadata={"format": "pt"})
    temporary.replace(checkpoint)
    raw_config = json.loads((snapshot / "config.json").read_text())
    config = dict(raw_config["text_config"])
    config["tie_word_embeddings"] = True
    assert config["model_type"] == "qwen3_5_text"
    write_json(out / "config.json", config)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        destination = out / name
        if destination.is_symlink():
            destination.unlink()
        shutil.copy2(snapshot / name, destination)
    report.update(
        status="PASS", tensor_count=len(state), transpose_count=18, offset_count=61,
        unchanged_gated_norm_count=18,
        before_dtype_counts=dict(Counter(row["before_dtype"] for row in report["tensors"])),
        after_dtype_counts=dict(Counter(row["after_dtype"] for row in report["tensors"])),
        output_sha256=sha256(checkpoint), output_size_bytes=checkpoint.stat().st_size,
        config_sha256=sha256(out / "config.json"),
        config_size_bytes=(out / "config.json").stat().st_size,
        source_config_sha256=sha256(snapshot / "config.json"),
        config_policy="exact source text_config flattened; tie_word_embeddings true; no vision_config, no second temperature",
        wall_seconds=time.monotonic() - started,
    )
    write_json(out / "mlx_to_hf.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    report = convert_snapshot(args.snapshot, args.out)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("tensors", "transposes", "offsets", "unchanged_gated_norms")}, indent=2))


if __name__ == "__main__":
    main()
