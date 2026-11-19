# Community port — NOT an Apple model.
"""End-to-end torch-ladder capstone: wire ALL verified overlays into the VoxCPM AR loop and check
the produced latents match the oracle ``latent_pred`` (so the host orchestration itself is correct,
not just the individual bundles). Uses oracle prefill inputs + captured z noise; derives everything
else (dit_hidden, pred_feat, curr_embed, KV continuation) with our own modules.

Reads oracle_ref.npz from $SCRATCH: set SCRATCH to the folder that holds it (unset: voxcpm_scratch
under the system temp folder).
"""
from __future__ import annotations

import glob
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from minicpm4 import build_kv_state, load_backbone  # noqa: E402
from feat_decoder import load_feat_decoder  # noqa: E402
from feat_encoder import load_feat_encoder, load_fsq, load_linear  # noqa: E402

SCRATCH = os.environ.get("SCRATCH") or os.path.join(tempfile.gettempdir(), "voxcpm_scratch")
REF = np.load(os.path.join(SCRATCH, "oracle_ref.npz"))
D = torch.float32
BUF = 64


def cos(a, b):
    a = a.detach().float().reshape(-1)
    b = torch.tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    snap = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--VoxCPM-0.5B/snapshots/*")))[-1]
    ck = torch.load(snap + "/pytorch_model.bin", map_location="cpu", weights_only=True)
    sd = ck.get("state_dict", ck)

    base = load_backbone(sd, "base_lm.", 24, 73448, BUF, D)
    res = load_backbone(sd, "residual_lm.", 6, 0, BUF, D)
    cfm = load_feat_decoder(sd, 4, D)
    enc = load_feat_encoder(sd, 4, D)
    fsq = load_fsq(sd, D)
    enc2lm = load_linear(sd, "enc_to_lm_proj.", 1024, 1024, D)
    lm2dit = load_linear(sd, "lm_to_dit_proj.", 1024, 1024, D)
    res2dit = load_linear(sd, "res_to_dit_proj.", 1024, 1024, D)

    kb, vb = build_kv_state(base.cfg, BUF, D)
    kr, vr = build_kv_state(res.cfg, BUF, D)

    n_steps = sum(1 for k in REF.files if k.startswith("feat_dec.out__"))
    Tprefill = REF["prefill_base.in_inputs_embeds__0"].shape[1]

    with torch.inference_mode():
        # prefill (no-prompt: enc_outputs unchanged by fsq since text_mask=1; use oracle combined embed)
        enc_out = base.prefill(torch.tensor(REF["prefill_base.in_inputs_embeds__0"], dtype=D), kb, vb)
        lm_hidden = enc_out[:, -1, :]
        res_out = res.prefill(torch.tensor(REF["prefill_res.in_inputs_embeds__0"], dtype=D), kr, vr)
        residual_hidden = res_out[:, -1, :]

        prefix_cond = torch.zeros(1, 2, 64, dtype=D)  # feat[:, -1] = zeros (no prompt)
        feats = []
        for i in range(n_steps):
            dit_hidden = lm2dit(lm_hidden) + res2dit(residual_hidden)
            z = torch.tensor(REF[f"randn__{i}"], dtype=D)
            pred = cfm(dit_hidden, prefix_cond.transpose(1, 2).contiguous(), z)  # [1,64,2]
            pred_feat = pred.transpose(1, 2)                                      # [1,2,64]
            curr = enc2lm(enc(pred_feat.unsqueeze(1)))                            # [1,1,1024]
            feats.append(pred_feat.unsqueeze(1))                                  # [1,1,2,64]
            prefix_cond = pred_feat
            pos = torch.tensor([Tprefill + i], dtype=torch.int32)
            lm_hidden = base.decode(curr, pos, kb, vb)[:, 0, :]
            lm_hidden = fsq(lm_hidden)
            residual_hidden = res.decode((lm_hidden + curr[:, 0, :]).unsqueeze(1), pos, kr, vr)[:, 0, :]

        latents = torch.cat(feats, dim=1)                       # [1,T,2,64]
        b, T, P, Dd = latents.shape
        latents = latents.permute(0, 3, 1, 2).reshape(b, Dd, T * P)  # b d (t p) = [1,64,12]

    c = cos(latents, REF["latent_pred__0"])
    ma = float(np.abs(latents.numpy() - REF["latent_pred__0"]).max())
    print("=== END-TO-END AR loop (all overlays wired, oracle z) ===")
    print(f"  produced latents {tuple(latents.shape)} vs oracle latent_pred")
    print(f"  cos = {c:.6f}   max|Δ| = {ma:.2e}   {'GATE PASS' if c >= 0.999 else 'GATE FAIL'}")
    sys.exit(0 if c >= 0.999 else 1)


if __name__ == "__main__":
    main()
