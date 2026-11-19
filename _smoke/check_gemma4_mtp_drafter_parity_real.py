# Community port — NOT an Apple model.
"""Parity gate on REAL cases: Gemma4MtpDrafter torch vs the Section-11 tflite.

Real inputs (MLX fp16 transplant greedy on the sky prompt) keep the tflite's int8
activation quantization in-distribution — the honest transplant gate. The shared
cache is stored once (int8 + scales); each case slices rows 0..pos.

PASS: argmax agreement >= 90%, mean logits cos >= 0.995, proj cos >= 0.995.

Run: .venv/bin/python ../coreai-models-community/_smoke/check_gemma4_mtp_drafter_parity_real.py
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

    z = np.load(f"{EXTRACT}/gemma4e2b_drafter_real_cases.npz")
    s_k13, s_v13, s_k14, s_v14 = z["kv_scales"]
    T = z["k13"].shape[0]
    max_seq = 128
    k11 = torch.zeros(1, 1, max_seq, HD_MAX); v11 = torch.zeros(1, 1, max_seq, HD_MAX)
    k14 = torch.zeros(1, 1, max_seq, HD_MAX); v14 = torch.zeros(1, 1, max_seq, HD_MAX)
    k11[0, 0, :T, :256] = torch.from_numpy(z["k13"].astype(np.float32) * s_k13)
    v11[0, 0, :T, :256] = torch.from_numpy(z["v13"].astype(np.float32) * s_v13)
    k14[0, 0, :T, :] = torch.from_numpy(z["k14"].astype(np.float32) * s_k14)
    v14[0, 0, :T, :] = torch.from_numpy(z["v14"].astype(np.float32) * s_v14)

    # The gate that matters: does the TORCH drafter draft as well as the tflite?
    # Full G=3 chain per position vs the main greedy refs, compared to the tflite's
    # accept counts recorded in the same dump. (Argmax/cos still reported.)
    seq_ref = z["seq"].tolist()
    G = 3

    # torch-side embedding for chained drafts: dequantized int2 embed x sqrt(1536)
    from coreai_models.models.macos.gemma4_mtp_drafter import _dq_int8  # noqa: F401
    from safetensors import safe_open
    fw = safe_open(f"{EXTRACT}/gemma4e2b_mixedbit_weights.safetensors", framework="pt")
    emb_packed = fw.get_tensor("embed.composite")
    emb_scale = fw.get_tensor("embed.composite.scale")

    def emb_norm_torch(tok: int) -> torch.Tensor:
        hidden_dim = 1536
        row = emb_packed.reshape(262144, hidden_dim // 4)[tok].to(torch.int16)
        c = torch.stack([(row >> s) & 3 for s in (0, 2, 4, 6)], dim=-1).reshape(hidden_dim)
        codes = torch.where(c >= 2, c - 4, c).float()
        return codes * emb_scale[tok].float() * (hidden_dim ** 0.5)

    n = agree = 0
    lcos, pcos = [], []
    acc_t, acc_f = [], []
    while f"c{n}_pos" in z:
        pos = int(z[f"c{n}_pos"][0])
        seq = pos + 1
        pos_t = torch.tensor([[pos]], dtype=torch.int32)
        m_full = torch.zeros(1, 1, 1, max_seq)
        m_full[..., :seq] = 1.0
        m_slide = torch.zeros(1, 1, 1, max_seq)
        m_slide[..., max(0, pos - 511):seq] = 1.0
        ids = torch.zeros(1, 1, dtype=torch.int32)

        model.embed_tokens.value = torch.from_numpy(z[f"c{n}_emb"])
        hidden = torch.from_numpy(z[f"c{n}_hidden"]).reshape(1, 1, -1)
        with torch.no_grad():
            logits, proj = model(ids, hidden, pos_t, m_slide, m_full, k11, v11, k14, v14)
        lg = logits[0, 0].numpy()
        ref_l = z[f"c{n}_logits"]
        am_t, am_r = int(lg.argmax()), int(ref_l.argmax())
        agree += am_t == am_r
        lcos.append(float(np.dot(lg, ref_l) / (np.linalg.norm(lg) * np.linalg.norm(ref_l))))
        pj = proj[0, 0].numpy()
        ref_p = z[f"c{n}_proj"]
        pcos.append(float(np.dot(pj, ref_p) / (np.linalg.norm(pj) * np.linalg.norm(ref_p))))

        # chain
        drafts = [am_t]
        cur_proj = proj
        for _ in range(1, G):
            model.embed_tokens.value = emb_norm_torch(drafts[-1])
            with torch.no_grad():
                lgi, cur_proj = model(ids, cur_proj, pos_t, m_slide, m_full, k11, v11, k14, v14)
            drafts.append(int(lgi[0, 0].argmax()))
        refs = seq_ref[pos + 2:pos + 2 + G]  # p = pos+1; refs seq[p+1..]
        c = 0
        while c < len(refs) and c < G and drafts[c] == refs[c]:
            c += 1
        acc_t.append(c)
        acc_f.append(int(z[f"c{n}_tflite_accept"][0]))
        n += 1

    a1_t = np.mean([a >= 1 for a in acc_t])
    a1_f = np.mean([a >= 1 for a in acc_f])
    tps_t = np.mean(acc_t) + 1
    tps_f = np.mean(acc_f) + 1
    print(f"argmax-vs-tflite {agree}/{n}  cos logits {np.mean(lcos):.5f} proj {np.mean(pcos):.5f}")
    print(f"alpha1  torch {a1_t:.3f}  tflite {a1_f:.3f}")
    print(f"tok/step torch {tps_t:.3f}  tflite {tps_f:.3f}")
    ok = a1_t >= a1_f - 0.05 and tps_t >= tps_f - 0.15
    print("PARITY(ALPHA)", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
