# Community port — NOT an Apple model.
"""Parity gate: the Core AI Gemma4MtpDrafter torch module vs the REAL Section-11 tflite.

Consumes gemma4e2b_drafter_ref_cases.npz (dumped by litertlm-convert/scripts/
gemma4_drafter_ref_cases.py from the tflite on identical inputs). The torch module
runs fp32 with dequantized weights; the tflite quantizes activations to int8
internally, so the gate is argmax agreement + logits/proj cosine, not id-exactness.

PASS: argmax >= 22/24, mean logits cos >= 0.99, mean proj cos >= 0.99.

Run (coreai-models checkout): .venv/bin/python ../coreai-models-community/_smoke/check_gemma4_mtp_drafter_parity.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from coreai_models.models.macos.gemma4_mtp_drafter import (
    FULL_SLOT,
    SLIDING_SLOT,
    Gemma4MtpDrafter,
    load_transplant,
)

EXTRACT = os.path.expanduser("~/code/litertlm-convert/out/gemma4e2b_extract")
N_SLOTS = 15
HD_MAX = 512
MAX_SEQ = 256


class StubEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = None

    def forward(self, input_ids):
        return self.value.reshape(1, 1, -1)


def main():
    model = Gemma4MtpDrafter()
    load_transplant(model, EXTRACT, dtype=torch.float32, fp_head=True)
    model.embed_tokens = StubEmbed()
    model.eval()

    z = np.load(f"{EXTRACT}/gemma4e2b_drafter_ref_cases.npz")
    s_k13, s_v13, s_k14, s_v14 = z["kv_scales"]

    n = 0
    agree = 0
    lcos, pcos = [], []
    while f"c{n}_pos" in z:
        pos = int(z[f"c{n}_pos"][0])
        seq = pos + 1
        k = torch.zeros(N_SLOTS, 1, 1, MAX_SEQ, HD_MAX)
        v = torch.zeros(N_SLOTS, 1, 1, MAX_SEQ, HD_MAX)
        k[SLIDING_SLOT, 0, 0, :seq, :256] = torch.from_numpy(
            z[f"c{n}_k13"].astype(np.float32) * s_k13)
        v[SLIDING_SLOT, 0, 0, :seq, :256] = torch.from_numpy(
            z[f"c{n}_v13"].astype(np.float32) * s_v13)
        k[FULL_SLOT, 0, 0, :seq, :] = torch.from_numpy(
            z[f"c{n}_k14"].astype(np.float32) * s_k14)
        v[FULL_SLOT, 0, 0, :seq, :] = torch.from_numpy(
            z[f"c{n}_v14"].astype(np.float32) * s_v14)

        model.embed_tokens.value = torch.from_numpy(z[f"c{n}_emb"])
        hidden = torch.from_numpy(z[f"c{n}_hidden"]).reshape(1, 1, -1)
        position_ids = torch.arange(seq, dtype=torch.int32).unsqueeze(0)
        ids = torch.zeros(1, 1, dtype=torch.int32)

        with torch.no_grad():
            logits, proj = model(ids, hidden, position_ids, k, v)
        lg = logits[0, 0].numpy()
        pj = proj[0, 0].numpy()
        ref_l = z[f"c{n}_logits"]
        ref_p = z[f"c{n}_proj"]

        am_t, am_r = int(lg.argmax()), int(ref_l.argmax())
        c_l = float(np.dot(lg, ref_l) / (np.linalg.norm(lg) * np.linalg.norm(ref_l)))
        c_p = float(np.dot(pj, ref_p) / (np.linalg.norm(pj) * np.linalg.norm(ref_p)))
        agree += am_t == am_r
        lcos.append(c_l)
        pcos.append(c_p)
        print(f"case {n:2d}: seq={seq:3d} argmax {am_t}{'==' if am_t == am_r else '!='}{am_r}"
              f"  cos_logits={c_l:.5f} cos_proj={c_p:.5f}")
        n += 1

    print(f"\nargmax {agree}/{n}  mean cos logits {np.mean(lcos):.5f}  proj {np.mean(pcos):.5f}")
    ok = agree >= n - 2 and np.mean(lcos) >= 0.99 and np.mean(pcos) >= 0.99
    print("PARITY", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
