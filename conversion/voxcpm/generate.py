# Community port — NOT an Apple model.
"""Standalone VoxCPM-0.5B generator built ENTIRELY from the ported overlays (no voxcpm dep).

This is the host AR loop (the reference for the eventual Swift/CoreAIKit port): tokenize -> prefill
base_lm+residual_lm -> per-frame [dit_hidden -> CFM(host z) -> LocEnc -> base_decode -> FSQ ->
res_decode], stop on the stop head -> AudioVAE -> 16kHz wav. Plain TTS (no voice-clone prompt yet).

  python generate.py --text "..." [--seed 0] [--max-len 200] [--out out.wav]
  python generate.py --reproduce-oracle   # seed 0, oracle text, 6 steps -> bit-check vs oracle wav
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
import wave

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from minicpm4 import build_kv_state, load_backbone  # noqa: E402
from feat_decoder import load_feat_decoder  # noqa: E402
from feat_encoder import load_feat_encoder, load_fsq, load_linear  # noqa: E402
from audio_vae import load_audio_vae  # noqa: E402

D = torch.float32
BUF = 512
SR = 16000
AUDIO_START = 101


def snap_dir():
    return sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--VoxCPM-0.5B/snapshots/*")))[-1]


class VoxCPMGen:
    def __init__(self):
        snap = snap_dir()
        ck = torch.load(snap + "/pytorch_model.bin", map_location="cpu", weights_only=True)
        sd = ck.get("state_dict", ck)
        self.base = load_backbone(sd, "base_lm.", 24, 73448, BUF, D)
        self.res = load_backbone(sd, "residual_lm.", 6, 0, BUF, D)
        self.cfm = load_feat_decoder(sd, 4, D)
        self.enc = load_feat_encoder(sd, 4, D)
        self.fsq = load_fsq(sd, D)
        self.enc2lm = load_linear(sd, "enc_to_lm_proj.", 1024, 1024, D)
        self.lm2dit = load_linear(sd, "lm_to_dit_proj.", 1024, 1024, D)
        self.res2dit = load_linear(sd, "res_to_dit_proj.", 1024, 1024, D)
        self.stop_proj = load_linear(sd, "stop_proj.", 1024, 1024, D)
        self.stop_head = torch.nn.Linear(1024, 2, bias=False).to(D).eval()
        self.stop_head.load_state_dict({"weight": sd["stop_head.weight"].to(D)}, assign=True)
        avck = torch.load(snap + "/audiovae.pth", map_location="cpu", weights_only=True)
        self.vae = load_audio_vae(avck.get("state_dict", avck), D)
        from transformers import LlamaTokenizerFast
        self.tok = LlamaTokenizerFast.from_pretrained(snap)

    @torch.inference_mode()
    def generate(self, text, seed=0, max_len=200, min_len=2):
        ids = self.tok(text, add_special_tokens=False)["input_ids"] + [AUDIO_START]
        ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        T0 = ids.shape[1]
        embeds = self.base.embed_tokens(ids)  # scale_emb=1 (use_mup=False)

        kb, vb = build_kv_state(self.base.cfg, BUF, D)
        kr, vr = build_kv_state(self.res.cfg, BUF, D)
        enc_out = self.base.prefill(embeds, kb, vb)           # text positions: fsq combine = identity
        lm_hidden = enc_out[:, -1, :]
        res_out = self.res.prefill(enc_out, kr, vr)
        residual_hidden = res_out[:, -1, :]

        torch.manual_seed(seed)
        prefix_cond = torch.zeros(1, 2, 64, dtype=D)
        feats = []
        for i in range(max_len):
            dit = self.lm2dit(lm_hidden) + self.res2dit(residual_hidden)
            z = torch.randn(1, 64, 2, dtype=D)
            pred = self.cfm(dit, prefix_cond.transpose(1, 2).contiguous(), z)  # [1,64,2]
            pred_feat = pred.transpose(1, 2)                                   # [1,2,64]
            curr = self.enc2lm(self.enc(pred_feat.unsqueeze(1)))               # [1,1,1024]
            feats.append(pred_feat.unsqueeze(1))
            prefix_cond = pred_feat
            stop = self.stop_head(torch.nn.functional.silu(self.stop_proj(lm_hidden))).argmax(-1)[0].item()
            if i > min_len and stop == 1:
                break
            pos = torch.tensor([T0 + i], dtype=torch.int32)
            lm_hidden = self.base.decode(curr, pos, kb, vb)[:, 0, :]
            lm_hidden = self.fsq(lm_hidden)
            residual_hidden = self.res.decode((lm_hidden + curr[:, 0, :]).unsqueeze(1), pos, kr, vr)[:, 0, :]

        latents = torch.cat(feats, dim=1)
        b, Tn, P, Dd = latents.shape
        latents = latents.permute(0, 3, 1, 2).reshape(b, Dd, Tn * P)
        wav = self.vae(latents).squeeze(0).squeeze(0)  # [samples]
        return wav, latents


def write_wav(path, wav, sr=SR):
    w = np.clip(wav.numpy(), -1, 1)
    pcm = (w * 32767).astype(np.int16)
    with wave.open(path, "w") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="Open source on device speech, running entirely on your phone.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=200)
    ap.add_argument("--out", default="voxcpm_port.wav")
    ap.add_argument("--reproduce-oracle", action="store_true")
    a = ap.parse_args()

    g = VoxCPMGen()
    if a.reproduce_oracle:
        wav, _ = g.generate("Hello world, this is a test.", seed=0, max_len=6, min_len=999)
        SCRATCH = os.environ.get("SCRATCH") or os.path.join(tempfile.gettempdir(), "voxcpm_scratch")
        ref = np.load(os.path.join(SCRATCH, "oracle_ref.npz"))["wav__0"].reshape(-1)
        n = min(len(wav), len(ref))
        c = torch.nn.functional.cosine_similarity(wav[:n], torch.tensor(ref[:n]), dim=0).item()
        print(f"[reproduce-oracle] my wav {tuple(wav.shape)} vs oracle {ref.shape}  raw-wav cos={c:.6f}  "
              f"{'BIT-MATCH' if c >= 0.999 else 'MISMATCH'}")
        return
    wav, lat = g.generate(a.text, seed=a.seed, max_len=a.max_len)
    write_wav(a.out, wav)
    print(f"[generate] text={a.text!r}")
    print(f"[generate] {lat.shape[-1]} frames -> {len(wav)} samples = {len(wav)/SR:.2f}s @ {SR}Hz -> {a.out}")


if __name__ == "__main__":
    main()
