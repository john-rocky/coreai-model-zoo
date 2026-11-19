# Community port — NOT an Apple model.
"""Gate LocEnc + FSQ + enc_to_lm_proj vs oracle. Also verify enc_to_lm_proj(LocEnc)==base_step input
(closes the audio->LM feedback loop)."""
from __future__ import annotations

import glob
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from feat_encoder import load_feat_encoder, load_fsq, load_linear  # noqa: E402

SCRATCH = os.environ.get("SCRATCH") or os.path.join(tempfile.gettempdir(), "voxcpm_scratch")
REF = np.load(os.path.join(SCRATCH, "oracle_ref.npz"))
D = torch.float32


def cos(a, b):
    a = a.detach().float().reshape(-1)
    b = torch.tensor(b, dtype=torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    snap = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--VoxCPM-0.5B/snapshots/*")))[-1]
    ck = torch.load(snap + "/pytorch_model.bin", map_location="cpu", weights_only=True)
    sd = ck.get("state_dict", ck)
    enc = load_feat_encoder(sd, 4, D)
    fsq = load_fsq(sd, D)
    enc2lm = load_linear(sd, "enc_to_lm_proj.", 1024, 1024, D)

    cs = []
    print("=== LocEnc (feat_encoder) ===")
    n_enc = sum(1 for k in REF.files if k.startswith("feat_enc.out__"))
    for i in range(n_enc):
        x = torch.tensor(REF[f"feat_enc.in_0__{i}"], dtype=D)
        with torch.inference_mode():
            out = enc(x)
        c = cos(out, REF[f"feat_enc.out__{i}"])
        cs.append(c)
        tag = "prefill" if i == 0 else f"step{i-1}"
        print(f"  {tag:8s} in={tuple(x.shape)} cos={c:.6f} {'OK' if c>=0.999 else 'FAIL'}")

    print("=== FSQ ===")
    n_fsq = sum(1 for k in REF.files if k.startswith("fsq.out__"))
    for i in range(n_fsq):
        h = torch.tensor(REF[f"fsq.in_0__{i}"], dtype=D)
        with torch.inference_mode():
            out = fsq(h)
        c = cos(out, REF[f"fsq.out__{i}"])
        cs.append(c)
        print(f"  fsq[{i}] cos={c:.6f} {'OK' if c>=0.999 else 'FAIL'}")

    print("=== feedback loop: enc_to_lm_proj(LocEnc(pred_feat)) == base_step.in_0 ? ===")
    # decode step i feeds curr_embed = enc_to_lm_proj(LocEnc(pred_feat_i)); LocEnc input is feat_enc step i+1
    n_steps = sum(1 for k in REF.files if k.startswith("base_step.in_0__"))
    for i in range(n_steps):
        x = torch.tensor(REF[f"feat_enc.in_0__{i+1}"], dtype=D)  # step decode encoder call (idx 0 = prefill)
        with torch.inference_mode():
            curr = enc2lm(enc(x))
        c = cos(curr.reshape(-1), REF[f"base_step.in_0__{i}"])
        cs.append(c)
        print(f"  loop[{i}] cos={c:.6f} {'OK' if c>=0.999 else 'FAIL'}")

    lo = min(cs)
    print(f"\n>>> min cos = {lo:.6f} -> {'GATE PASS' if lo>=0.999 else 'GATE FAIL'}")
    sys.exit(0 if lo >= 0.999 else 1)


if __name__ == "__main__":
    main()
