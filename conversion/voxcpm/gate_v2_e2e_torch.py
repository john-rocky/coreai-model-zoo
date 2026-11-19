# Community port — NOT an Apple model.
"""End-to-end torch parity gate for VoxCPM2 (2B): my exportable overlays, wired in the v2 AR dataflow,
vs the OFFICIAL `VoxCPM2Model._inference` (the true oracle, full official model loaded from the real
checkpoint).

The v2 dataflow is the risky, never-before-verified part (vs v1):
  * mu = CONCAT(lm_to_dit_proj(lm_h), res_to_dit_proj(res_h))  -> 2048 -> two 1024 DiT tokens
  * residual input = fusion_concat_proj(cat(fsq(lm_h), curr_embed))   (v1 ADDED them)
  * FSQ latent 512 (v1 256); patch_size 4 (v1 2)

Oracle = official `m.inference(...)` with torch.randn recorded per diffusion step. Mine = the same AR
loop transcribed against the bundle-I/O contract, using my MiniCPM4Backbone (base 28L + residual 8L
no-rope), my CFMDecoder(LocDiTV2 12L), my LocEnc 12L, and the official host-glue Linears (plain
matmuls, identical in export). z replayed so only the trajectory differs by overlay fp error.

Pass = cos >= 0.99 on the produced latents (v1's e2e capstone landed 0.999269; AR feedback amplifies
per-step fp error in max|Δ| but cos stays high — orchestration correctness, not bit-exactness).

  coreai-models/.venv/bin/python gate_v2_e2e_torch.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "_ref_v2"))

from einops import rearrange  # noqa: E402

from minicpm4 import load_backbone, build_kv_state  # noqa: E402
from feat_decoder_v2 import load_feat_decoder_v2  # noqa: E402
from feat_encoder_v2 import load_feat_encoder_v2  # noqa: E402
from voxref.model.voxcpm2 import VoxCPM2Model  # noqa: E402

DTYPE = torch.float32
PATCH = 4
FEAT = 64
T_TEXT = 7      # fixed pseudo-text length
N_STEPS = 8     # fixed AR steps (no early stop)
BUF = 64


def snap() -> str:
    return sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--VoxCPM2/snapshots/*")))[-1]


def cos(a, b) -> float:
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


class RandnRecorder:
    """Record every torch.randn draw (the only stochastic input is the CFM z)."""
    def __init__(self):
        self.draws = []
        self._orig = torch.randn

    def __enter__(self):
        def rec(*a, **k):
            out = self._orig(*a, **k)
            self.draws.append(out.detach().clone())
            return out
        torch.randn = rec
        return self

    def __exit__(self, *exc):
        torch.randn = self._orig


def main():
    sd_path = snap() + "/model.safetensors"
    from safetensors.torch import load_file
    sd = load_file(sd_path)
    cfg = json.load(open(snap() + "/config.json"))
    lm = cfg["lm_config"]
    short_factor = lm["rope_scaling"]["short_factor"]

    # ---- official model (fp32, cpu) = oracle -------------------------------
    print("[load] official VoxCPM2Model (cpu, fp32) ...")
    m = VoxCPM2Model.from_local(snap(), optimize=False, training=False, device="cpu")
    m = m.to(DTYPE).eval()
    m.config.dtype = "float32"
    # caches were allocated bf16 in __init__ (config dtype); re-allocate fp32 to match the casted model
    dev = torch.device("cpu")
    m.base_lm.setup_cache(1, m.config.max_length, dev, DTYPE)
    m.residual_lm.setup_cache(1, m.config.max_length, dev, DTYPE)

    # fixed zero-shot inputs
    torch.manual_seed(7)
    text_token = torch.randint(10, 70000, (1, T_TEXT), dtype=torch.long)
    text_mask = torch.ones(1, T_TEXT, dtype=torch.int32)
    feat = torch.zeros(1, T_TEXT, PATCH, FEAT, dtype=DTYPE)
    feat_mask = torch.zeros(1, T_TEXT, dtype=torch.int32)

    with torch.inference_mode():
        with RandnRecorder() as rec:
            torch.manual_seed(0)
            feat_pred, generated_feat = m.inference(
                text_token, text_mask, feat, feat_mask,
                min_len=10_000, max_len=N_STEPS, inference_timesteps=10, cfg_value=2.0,
            )
    zs = rec.draws
    print(f"[oracle] latents {tuple(feat_pred.shape)}  recorded {len(zs)} z-draws "
          f"(expect {N_STEPS}, each {tuple(zs[0].shape)})")

    # ---- my overlays -------------------------------------------------------
    my_base = load_backbone(sd, "base_lm.", 28, 73448, BUF, DTYPE,
                            hidden_size=lm["hidden_size"], intermediate_size=lm["intermediate_size"],
                            num_attention_heads=lm["num_attention_heads"],
                            num_key_value_heads=lm["num_key_value_heads"],
                            head_dim=lm["kv_channels"], short_factor=short_factor, no_rope=False)
    my_res = load_backbone(sd, "residual_lm.", 8, 0, BUF, DTYPE,
                           hidden_size=lm["hidden_size"], intermediate_size=lm["intermediate_size"],
                           num_attention_heads=lm["num_attention_heads"],
                           num_key_value_heads=lm["num_key_value_heads"],
                           head_dim=lm["kv_channels"], short_factor=short_factor, no_rope=True)
    my_cfm = load_feat_decoder_v2(sd, short_factor)
    my_enc = load_feat_encoder_v2(sd, short_factor)

    # host glue: reuse the official Linears (plain matmuls, identical in export)
    embed_tokens = m.base_lm.embed_tokens
    enc_to_lm = m.enc_to_lm_proj
    lm_to_dit = m.lm_to_dit_proj
    res_to_dit = m.res_to_dit_proj
    fusion = m.fusion_concat_proj
    fsq = m.fsq_layer

    def to_dit_cond(prev):                 # [1,P,D] -> [1,D,P]
        return prev.transpose(1, 2).contiguous()

    with torch.inference_mode():
        # ---- prefill (zero-shot: combined = text_embed) ----
        text_embed = embed_tokens(text_token) * 1.0          # scale_emb=1 (use_mup False)
        combined = text_embed                                 # feat_mask 0
        kb, vb = build_kv_state(my_base.cfg, BUF, DTYPE)
        enc_outputs = my_base.prefill(combined, kb, vb)       # [1,T,2048]
        lm_hidden = enc_outputs[:, -1, :]                     # text-masked => no FSQ
        zero_feat_embed = torch.zeros_like(enc_outputs)       # feat_mask 0
        res_in = fusion(torch.cat((enc_outputs, zero_feat_embed), dim=-1))
        krb, vrb = build_kv_state(my_res.cfg, BUF, DTYPE)
        res_outputs = my_res.prefill(res_in, krb, vrb)
        res_hidden = res_outputs[:, -1, :]                    # [1,2048]

        prefix_cond = feat[:, -1, ...]                        # [1,4,64] zeros
        lat = []
        for i in range(N_STEPS):
            dit_hidden = torch.cat((lm_to_dit(lm_hidden), res_to_dit(res_hidden)), dim=-1)   # [1,2048]
            pred = my_cfm(dit_hidden, to_dit_cond(prefix_cond), zs[i]).transpose(1, 2)        # [1,4,64]
            curr_embed = enc_to_lm(my_enc(pred.unsqueeze(1)))                                 # [1,1,2048]
            lat.append(pred.unsqueeze(1))
            prefix_cond = pred
            pos = torch.tensor([T_TEXT + i], dtype=torch.int32)
            lm_hidden = my_base.decode(curr_embed[:, 0:1, :], pos, kb, vb).reshape(1, -1)     # [1,2048]
            lm_hidden = fsq(lm_hidden)
            res_input = fusion(torch.cat((lm_hidden, curr_embed[:, 0, :]), dim=-1))
            res_hidden = my_res.decode(res_input.unsqueeze(1), pos, krb, vrb).reshape(1, -1)

        my_lat = rearrange(torch.cat(lat, dim=1), "b t p d -> b d (t p)", p=PATCH)            # [1,64,32]

    c = cos(my_lat, feat_pred)
    md = (my_lat - feat_pred).abs().max().item()
    print(f"\n[latents] cos={c:.6f}  max|Δ|={md:.4f}  (my={tuple(my_lat.shape)})")

    # ---- full chain: latents -> AudioVAE -> 48kHz wav, vs official VAE ----
    from audio_vae_v2 import load_audio_vae_v2
    _vae_ck = torch.load(snap() + "/audiovae.pth", map_location="cpu", weights_only=True)
    my_vae = load_audio_vae_v2(_vae_ck.get("state_dict", _vae_ck))
    with torch.inference_mode():
        off_wav = m.audio_vae.decode(feat_pred.to(DTYPE)).reshape(-1)   # official latents -> official VAE
        my_wav = my_vae(my_lat.to(DTYPE)).reshape(-1)                   # my latents -> my VAE

    def magspec(w):
        win = torch.hann_window(1024)
        return torch.stft(w, 1024, 256, window=win, return_complex=True).abs().reshape(-1)

    n = min(len(off_wav), len(my_wav))
    raw = cos(my_wav[:n], off_wav[:n])
    mag = cos(magspec(my_wav[:n]), magspec(off_wav[:n]))
    sp = os.environ.get("SCRATCH", os.path.join(tempfile.gettempdir(),
                                   "voxcpm_scratch"))
    os.makedirs(sp, exist_ok=True)
    my_wav.numpy().astype("float32").tofile(os.path.join(sp, "voxcpm2_e2e_mine.f32"))
    off_wav.numpy().astype("float32").tofile(os.path.join(sp, "voxcpm2_e2e_official.f32"))
    print(f"[wav 48kHz] off={tuple(off_wav.shape)} my={tuple(my_wav.shape)}  raw cos={raw:.6f}  magspec cos={mag:.6f}")
    print(f"[wav] wrote voxcpm2_e2e_{{mine,official}}.f32 to scratchpad")

    ok = c >= 0.99 and mag >= 0.99
    print(f"\n>>> {'GATE PASS' if ok else 'GATE FAIL'}  (latents cos={c:.4f}, magspec cos={mag:.4f})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
