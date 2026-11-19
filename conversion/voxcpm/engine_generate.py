# Community port — NOT an Apple model.
"""Full ON-ENGINE VoxCPM pipeline: wire the 5 exported .aimodel bundles into the host AR loop and
synthesize speech entirely on the Core AI engine. The small glue (embed lookup, lm/res->dit projs,
FSQ, stop head) runs host-side (numpy/torch) — exactly what the Swift/CoreAIKit host will do.

  python engine_generate.py --reproduce-oracle   # seed0 oracle text -> engine wav, spectral vs oracle
  python engine_generate.py --text "..." [--seed 3] [--max-len 220] [--out engine.wav]
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import os
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

ART = Path(__file__).resolve().parent / "artifacts"
SCRATCH = os.environ.get("SCRATCH") or os.path.join(tempfile.gettempdir(), "voxcpm_scratch")
NL_B, NL_R, NKV, HD, CL = 24, 6, 2, 64, 512
SR = 16000
AUDIO_START = 101
PREFILL_T = 9  # baked prefill length of the exported backbone bundles


def na(x):
    import coreai.runtime as rt
    return rt.NDArray(np.ascontiguousarray(x))


async def call(fn, inputs, state=None, tries=4):
    """Engine calls intermittently raise CoreAIError 3 under back-to-back multi-model load; retry."""
    last = None
    for t in range(tries):
        try:
            return await (fn(inputs=inputs, state=state) if state is not None else fn(inputs=inputs))
        except RuntimeError as e:
            last = e
            await asyncio.sleep(0.05 * (t + 1))
    raise last


class Host:
    """host-side glue weights (fp16)."""

    def __init__(self, sd):
        g = lambda k: sd[k].to(torch.float16)
        self.embed = g("base_lm.embed_tokens.weight")            # [V,1024]
        self.lm2dit_w, self.lm2dit_b = g("lm_to_dit_proj.weight"), g("lm_to_dit_proj.bias")
        self.res2dit_w, self.res2dit_b = g("res_to_dit_proj.weight"), g("res_to_dit_proj.bias")
        self.fsq_iw, self.fsq_ib = g("fsq_layer.in_proj.weight"), g("fsq_layer.in_proj.bias")
        self.fsq_ow, self.fsq_ob = g("fsq_layer.out_proj.weight"), g("fsq_layer.out_proj.bias")
        self.sp_w, self.sp_b = g("stop_proj.weight"), g("stop_proj.bias")
        self.sh_w = g("stop_head.weight")

    def fsq(self, h):
        h = torch.tanh(F.linear(h, self.fsq_iw, self.fsq_ib))
        h = torch.round(h * 9) / 9
        return F.linear(h, self.fsq_ow, self.fsq_ob)

    def dit(self, lm, res):
        return F.linear(lm, self.lm2dit_w, self.lm2dit_b) + F.linear(res, self.res2dit_w, self.res2dit_b)

    def stop(self, lm):
        return int(F.linear(F.silu(F.linear(lm, self.sp_w, self.sp_b)), self.sh_w).argmax(-1)[0].item())


async def load_one(name):
    """Load one bundle -> (model, fn). Keep the model ref so we can free it (too many co-resident
    GPU-specialized models -> CoreAIError 3; the host loads/frees in phases, like CoreAIKit will)."""
    import coreai.runtime as rt
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    d = ART / name
    m = await rt.AIModel.load(str(d / f"{d.name}.aimodel"), gpu)
    return m, m.load_function("main")


def new_state(nl):
    import coreai.runtime as rt
    return {"keyCache": rt.NDArray(np.zeros((nl, 1, NKV, CL, HD), np.float16)),
            "valueCache": rt.NDArray(np.zeros((nl, 1, NKV, CL, HD), np.float16))}


async def generate(host, ids, seed, max_len, min_len):
    assert len(ids) == PREFILL_T, f"this bundle bakes prefill_len={PREFILL_T}; got {len(ids)}"
    emb = host.embed[torch.tensor(ids)].unsqueeze(0).to(torch.float16)  # [1,T,1024]
    sb, sr = new_state(NL_B), new_state(NL_R)

    # --- phase 1: prefill (load -> seed shared KV state -> FREE; state buffers persist) ---
    bp_m, bp = await load_one("voxcpm_base_fp16_prefill_t9")
    enc = torch.tensor((await call(bp, {"inputs_embeds": na(emb.numpy())}, state=sb))["hidden"].numpy())
    lm_h = enc[:, -1, :]
    del bp, bp_m
    rp_m, rp = await load_one("voxcpm_res_fp16_prefill_t9")
    res_h = torch.tensor((await call(rp, {"inputs_embeds": na(enc.numpy())}, state=sr))["hidden"].numpy())[:, -1, :]
    del rp, rp_m

    # --- phase 2: decode loop (base_decode + res_decode + fd + fe co-resident) ---
    bd_m, bd = await load_one("voxcpm_base_fp16_decode_cl512")
    rd_m, rd = await load_one("voxcpm_res_fp16_decode_cl512")
    fd_m, fd = await load_one("voxcpm_feat_decoder_fp16")
    fe_m, fe = await load_one("voxcpm_feat_encoder_fp16")
    torch.manual_seed(seed)
    cond = torch.zeros(1, 64, 2, dtype=torch.float16)
    feats = []
    for i in range(max_len):
        dit = host.dit(lm_h, res_h)
        z = torch.randn(1, 64, 2).to(torch.float16)
        pred = torch.tensor((await call(fd, {"mu": na(dit.numpy()), "cond": na(cond.numpy()),
                                             "z": na(z.numpy())}))["pred_feat"].numpy())  # [1,64,2]
        pred_feat = pred.transpose(1, 2)                              # [1,2,64]
        curr = torch.tensor((await call(fe, {"pred_feat": na(pred_feat.unsqueeze(1).numpy())}))["curr_embed"].numpy())
        feats.append(pred_feat.unsqueeze(1))
        cond = pred_feat
        if i > min_len and host.stop(lm_h) == 1:
            break
        pos = np.array([PREFILL_T + i], np.int32)
        bd_o = await call(bd, {"inputs_embeds": na(curr.numpy()), "pos": na(pos)}, state=sb)
        lm_h = host.fsq(torch.tensor(bd_o["hidden"].numpy())[:, 0, :])
        rd_o = await call(rd, {"inputs_embeds": na((lm_h + curr[:, 0, :]).unsqueeze(1).numpy()),
                               "pos": na(pos)}, state=sr)
        res_h = torch.tensor(rd_o["hidden"].numpy())[:, 0, :]
    del bd, rd, fd, fe, bd_m, rd_m, fd_m, fe_m

    lat = torch.cat(feats, dim=1)
    b, T, P, Dd = lat.shape
    lat = lat.permute(0, 3, 1, 2).reshape(b, Dd, T * P)              # [1,64,2T]
    # --- phase 3: vocoder (bakes T=12 frames; chunk latents into 12-frame windows) ---
    voc_m, voc = await load_one("voxcpm_vocoder_fp16_t12")
    wavs = []
    for s in range(0, lat.shape[-1], 12):
        chunk = lat[:, :, s:s + 12]
        if chunk.shape[-1] < 12:
            chunk = F.pad(chunk, (0, 12 - chunk.shape[-1]))
        vr = await call(voc, {"latents": na(chunk.to(torch.float16).numpy())})
        wavs.append(torch.tensor(vr["wav"].numpy()).reshape(-1)[: chunk.shape[-1] * 640])
    del voc, voc_m
    return torch.cat(wavs), lat


def magspec(x, n=512, hop=128):
    x = torch.as_tensor(x, dtype=torch.float32)
    return torch.stft(x, n_fft=n, hop_length=hop, window=torch.hann_window(n), return_complex=True).abs()


def write_wav(path, wav):
    pcm = (np.clip(wav.numpy(), -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "w") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR); f.writeframes(pcm.tobytes())


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="On device speech synthesis, running entirely on your iPhone.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=220)
    ap.add_argument("--out", default="voxcpm_engine.wav")
    ap.add_argument("--reproduce-oracle", action="store_true")
    a = ap.parse_args()

    snap = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--VoxCPM-0.5B/snapshots/*")))[-1]
    ck = torch.load(snap + "/pytorch_model.bin", map_location="cpu", weights_only=True)
    host = Host(ck.get("state_dict", ck))
    from transformers import LlamaTokenizerFast
    tok = LlamaTokenizerFast.from_pretrained(snap)

    if a.reproduce_oracle:
        ids = tok("Hello world, this is a test.", add_special_tokens=False)["input_ids"] + [AUDIO_START]
        wav, _ = await generate(host, ids, seed=0, max_len=6, min_len=999)
        ref = np.load(os.path.join(SCRATCH, "oracle_ref.npz"))["wav__0"].reshape(-1)
        n = min(len(wav), len(ref))
        A, B = magspec(wav[:n]).reshape(-1), magspec(ref[:n]).reshape(-1)
        spec = F.cosine_similarity(A, B, dim=0).item()
        raw = F.cosine_similarity(wav[:n], torch.tensor(ref[:n]), dim=0).item()
        print(f"[engine reproduce-oracle] raw-wav cos={raw:.5f}  MAGSPEC corr={spec:.5f}  "
              f"-> {'PERCEPTUAL MATCH' if spec>=0.99 else 'CHECK'}")
        return
    ids = tok(a.text, add_special_tokens=False)["input_ids"] + [AUDIO_START]
    if len(ids) != PREFILL_T:
        print(f"NOTE: text tokenizes to {len(ids)} (bundle bakes prefill_len={PREFILL_T}); "
              f"re-export with --prefill-len {len(ids)} for arbitrary text. Using oracle-len text for demo.")
        return
    wav, lat = await generate(host, ids, a.seed, a.max_len, 2)
    write_wav(a.out, wav)
    print(f"[engine generate] {lat.shape[-1]} frames -> {len(wav)} samp = {len(wav)/SR:.2f}s -> {a.out}")


if __name__ == "__main__":
    asyncio.run(main())
