# Community port — NOT an Apple model.
"""Gate the AudioVAE decoder (weight_norm folded) vs the oracle: oracle latent -> wav."""
from __future__ import annotations

import glob
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from audio_vae import load_audio_vae  # noqa: E402

SCRATCH = os.environ.get("SCRATCH") or os.path.join(tempfile.gettempdir(), "voxcpm_scratch")
REF = np.load(os.path.join(SCRATCH, "oracle_ref.npz"))
D = torch.float32


def cos(a, b):
    a = torch.as_tensor(a, dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(b, dtype=torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    snap = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--VoxCPM-0.5B/snapshots/*")))[-1]
    ck = torch.load(snap + "/audiovae.pth", map_location="cpu", weights_only=True)
    sd = ck.get("state_dict", ck)
    m = load_audio_vae(sd, D)

    z = torch.tensor(REF["vae_dec.in_0__0"], dtype=D)   # [1,64,12]
    with torch.inference_mode():
        wav = m(z)                                       # [1,1,7680]
    ref = REF["vae_dec.out__0"]
    c = cos(wav.numpy(), ref)
    ma = float(np.abs(wav.numpy() - ref).max())
    rms = float(np.sqrt(((wav.numpy() - ref) ** 2).mean()))
    print("=== AudioVAE decoder (weight_norm folded, deterministic) ===")
    print(f"  in {tuple(z.shape)} -> wav {tuple(wav.shape)}  vs oracle {ref.shape}")
    print(f"  raw-wav cos = {c:.6f}   max|Δ| = {ma:.2e}   rmse = {rms:.2e}")
    ok = c >= 0.999
    print(f"  -> {'GATE PASS' if ok else 'GATE FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
