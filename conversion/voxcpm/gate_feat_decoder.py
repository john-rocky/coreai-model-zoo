# Community port — NOT an Apple model.
"""Gate the CFM feat_decoder (LocDiT + euler loop) vs the oracle, with host-supplied z noise."""
from __future__ import annotations

import glob
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from feat_decoder import load_feat_decoder  # noqa: E402

SCRATCH = os.environ.get("SCRATCH") or os.path.join(tempfile.gettempdir(), "voxcpm_scratch")
REF = np.load(os.path.join(SCRATCH, "oracle_ref.npz"))
DTYPE = torch.float32


def cos(a, b):
    a = a.detach().float().reshape(-1)
    b = torch.tensor(b, dtype=torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    snap = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--VoxCPM-0.5B/snapshots/*")))[-1]
    ck = torch.load(snap + "/pytorch_model.bin", map_location="cpu", weights_only=True)
    sd = ck.get("state_dict", ck)
    m = load_feat_decoder(sd, n_layers=4, dtype=DTYPE)

    n = sum(1 for k in REF.files if k.startswith("feat_dec.out__"))
    cs, maxabs = [], []
    print("=== feat_decoder CFM (10-step euler, cfg 2.0, host z) ===")
    for i in range(n):
        mu = torch.tensor(REF[f"feat_dec.in_mu__{i}"], dtype=DTYPE)
        cond = torch.tensor(REF[f"feat_dec.in_cond__{i}"], dtype=DTYPE)
        z = torch.tensor(REF[f"randn__{i}"], dtype=DTYPE)
        with torch.inference_mode():
            out = m(mu, cond, z)
        ref = REF[f"feat_dec.out__{i}"]
        c = cos(out, ref)
        ma = (out.detach().numpy() - ref).__abs__().max()
        cs.append(c); maxabs.append(float(ma))
        print(f"  step[{i}] cos={c:.6f}  max|Δ|={ma:.2e}  {'OK' if c >= 0.999 else 'FAIL'}")
    lo = min(cs)
    print(f"\n>>> min cos = {lo:.6f}  max|Δ|={max(maxabs):.2e}  -> {'GATE PASS' if lo >= 0.999 else 'GATE FAIL'}")
    sys.exit(0 if lo >= 0.999 else 1)


if __name__ == "__main__":
    main()
