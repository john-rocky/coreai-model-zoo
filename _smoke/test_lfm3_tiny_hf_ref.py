#!/usr/bin/env python3
"""test_lfm3_tiny_hf_ref.py CODE_DIR OUT_DIR — the HF side of the LFM3 tiny parity smoke (oracle venv).

Step 1 of 2 (step 2 = test_lfm3_tiny_parity.py in the export venv). Runs in an interpreter with transformers >= 5
(the LFM3 lane used transformers 5.20.2, torch 2.14.0); CODE_DIR = any LFM3 checkpoint dir holding
modeling_lfm3.py / configuration_lfm3.py / config.json (e.g. a snapshot of LiquidAI/LFM3-1.7B). Builds Lfm3ForCausalLM from
CODE_DIR's own modeling_lfm3.py with a 6-layer config (conv, conv, full, conv, sliding, conv; head_dim 16 so
the full layers' zero-tail = 4 rotated + 4 unrotated frequencies; window 8), randomizes every weight so the
norms, the gate and the conv are non-trivial, and writes:

  OUT_DIR/plain/{config.json, model.safetensors}   keys `model.*` (the 1.7B layout, tied head dropped)
  OUT_DIR/audio/{config.json, model.safetensors}   the same tensors under `lfm.model.*` + a decoy tower,
                                                   config under the `lfm` key (the Audio-2B layout)
  OUT_DIR/ref.npz   ids [N], logits [N, V] and last_hidden [N, H] of one fp32 eager forward

Set HF_MODULES_CACHE to a scratch dir if the remote-code copy should not land in the shared HF modules cache.
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

LAYER_TYPES = ["conv", "conv", "full_attention", "conv", "sliding_attention", "conv"]
N = 45  # > 5 windows of 8


def main(hf_dir: str, out_dir: str) -> None:
    torch.set_grad_enabled(False)
    cfg = AutoConfig.from_pretrained(
        hf_dir, trust_remote_code=True,
        vocab_size=512, hidden_size=64, intermediate_size=96, num_hidden_layers=6,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, sliding_window=8,
        layer_types=LAYER_TYPES, max_position_embeddings=4096,
        pad_token_id=3, bos_token_id=1, eos_token_id=2,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).float().eval()
    g = torch.Generator().manual_seed(1)
    for name, p in model.named_parameters():
        if name.endswith("norm.weight"):
            p.copy_(1.0 + 0.3 * torch.randn(p.shape, generator=g))
        elif name.endswith("conv.conv.weight"):
            p.copy_(0.5 * torch.randn(p.shape, generator=g))
        elif name.endswith("embed_tokens.weight"):
            p.copy_(torch.randn(p.shape, generator=g))
        elif p.dim() == 2:
            p.copy_(torch.randn(p.shape, generator=g) / p.shape[1] ** 0.5)
    ids = torch.randint(4, 512, (1, N), generator=g)
    out = model(input_ids=ids, output_hidden_states=False)
    hidden = model.model(input_ids=ids).last_hidden_state
    # the head is tied: logits must be hidden @ embed^T
    assert torch.allclose(out.logits, hidden @ model.model.embed_tokens.weight.T, atol=1e-5)

    sd = {k: v.contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    raw = cfg.to_dict()
    for k in ("auto_map", "architectures", "_attn_implementation_autoset", "transformers_version"):
        raw.pop(k, None)
    os.makedirs(os.path.join(out_dir, "plain"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "audio"), exist_ok=True)
    save_file(sd, os.path.join(out_dir, "plain", "model.safetensors"))
    with open(os.path.join(out_dir, "plain", "config.json"), "w") as f:
        json.dump(raw, f, indent=1)
    audio = {"lfm." + k: v for k, v in sd.items()}
    audio["encoder.lfm.lfm.layers.0.operator_norm.weight"] = torch.ones(64)  # decoy tower
    audio["lfm.depth_embeddings.0.embedding_norm.weight"] = torch.ones(64)  # outside lfm.model.
    save_file(audio, os.path.join(out_dir, "audio", "model.safetensors"))
    with open(os.path.join(out_dir, "audio", "config.json"), "w") as f:
        json.dump({"lfm": raw, "codebooks": 8}, f, indent=1)
    np.savez(os.path.join(out_dir, "ref.npz"), ids=ids[0].numpy(),
             logits=out.logits[0].float().numpy(), last_hidden=hidden[0].float().numpy())
    print(f"tiny written: {len(sd)} tensors, N={N}, logits absmax {out.logits.abs().max():.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
