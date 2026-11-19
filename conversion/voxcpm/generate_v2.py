# Community port — NOT an Apple model.
"""VoxCPM2 (2B) real-text speech generation: my exportable overlays vs the official model, zero-shot.

Drives the official `VoxCPM2Model` (oracle) on a real sentence, records the per-step CFM noise + the
natural stop length, then runs MY self-contained `VoxCPM2Pipeline` replaying the same noise for the same
number of steps. Decodes both latent tracks to 48kHz audio (mine via my AudioVAE overlay, official via
its VAE), reports raw + magnitude-spectrogram correlation, and writes both .wav for listening.

  coreai-models/.venv/bin/python generate_v2.py "Your sentence here."
"""
from __future__ import annotations

import glob
import os
import struct
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "_ref_v2"))

from pipeline_v2 import VoxCPM2Pipeline, snap  # noqa: E402
from gate_v2_e2e_torch import RandnRecorder, cos  # noqa: E402
from voxref.model.voxcpm2 import VoxCPM2Model  # noqa: E402

DTYPE = torch.float32
SR = 48000


def write_wav(path, wav, sr=SR):
    x = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2").tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF"); f.write(struct.pack("<I", 36 + len(pcm))); f.write(b"WAVE")
        f.write(b"fmt "); f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data"); f.write(struct.pack("<I", len(pcm))); f.write(pcm)


def magspec(w):
    win = torch.hann_window(1024)
    return torch.stft(w, 1024, 256, window=win, return_complex=True).abs().reshape(-1)


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "On device speech synthesis running on Apple silicon."
    print(f"[text] {text!r}")

    print("[load] official VoxCPM2Model (cpu, fp32) ...")
    m = VoxCPM2Model.from_local(snap(), optimize=False, training=False, device="cpu")
    m = m.to(DTYPE).eval()
    m.config.dtype = "float32"
    m.base_lm.setup_cache(1, m.config.max_length, torch.device("cpu"), DTYPE)
    m.residual_lm.setup_cache(1, m.config.max_length, torch.device("cpu"), DTYPE)

    # zero-shot inputs (mirror _generate)
    ids = torch.LongTensor(m.text_tokenizer(text))
    text_token = torch.cat([ids, torch.tensor([m.audio_start_token])]).unsqueeze(0)
    T = text_token.shape[1]
    text_mask = torch.ones(1, T, dtype=torch.int32)
    feat = torch.zeros(1, T, 4, 64, dtype=DTYPE)
    feat_mask = torch.zeros(1, T, dtype=torch.int32)

    with torch.inference_mode(), RandnRecorder() as rec:
        torch.manual_seed(0)
        feat_pred, _ = m.inference(text_token, text_mask, feat, feat_mask,
                                   min_len=2, max_len=2000, inference_timesteps=10, cfg_value=2.0)
    zs = rec.draws
    N = len(zs)
    print(f"[oracle] {N} AR steps ({N * 4 / 6.25 / 4:.2f}s audio)  latents {tuple(feat_pred.shape)}")

    pipe = VoxCPM2Pipeline(dtype=DTYPE, buf=max(512, T + N + 8))
    with torch.inference_mode():
        my_lat = pipe.generate(text, zs=zs)             # replay same noise, same step count
        my_wav = pipe.vae(my_lat.to(DTYPE)).reshape(-1)
        off_wav = m.audio_vae.decode(feat_pred.to(DTYPE)).reshape(-1)

    n = min(len(my_wav), len(off_wav))
    lat_c = cos(my_lat, feat_pred) if my_lat.shape == feat_pred.shape else float("nan")
    raw = cos(my_wav[:n], off_wav[:n])
    mag = cos(magspec(my_wav[:n]), magspec(off_wav[:n]))

    sp = (os.environ.get("SCRATCH")
          or os.path.join(tempfile.gettempdir(), "voxcpm_scratch"))
    os.makedirs(sp, exist_ok=True)
    mp, op = os.path.join(sp, "voxcpm2_mine.wav"), os.path.join(sp, "voxcpm2_official.wav")
    write_wav(mp, my_wav.numpy()); write_wav(op, off_wav.numpy())

    def stats(w):
        a = w.numpy()
        return f"rms={np.sqrt((a**2).mean()):.3f} peak={np.abs(a).max():.3f} len={len(a)/SR:.2f}s"
    print(f"[mine]     {stats(my_wav)}")
    print(f"[official] {stats(off_wav)}")
    print(f"[match] latents cos={lat_c:.6f}  raw wav cos={raw:.6f}  magspec cos={mag:.6f}")
    print(f"[wav] {mp}\n      {op}")


if __name__ == "__main__":
    main()
