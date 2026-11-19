# Community port — NOT an Apple model.
"""Export the remaining feed-forward bundles (no state) + engine-gate each vs oracle:
  feat_encoder : LocEnc + enc_to_lm_proj : pred_feat[1,1,2,64] -> curr_embed[1,1,1024]
  vocoder      : AudioVAE decoder       : latents[1,64,T] -> wav[1,1,640T]

  python export_feedforward.py [--which feat_encoder|vocoder] [--mode fp16|fp32] [--vae-frames 12]
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coreai_models.export.macos as _macos  # noqa: E402
from coreai_models.export.macos import export_to_coreai  # noqa: E402
from feat_encoder import load_feat_encoder, load_linear  # noqa: E402
from audio_vae import load_audio_vae  # noqa: E402

_macos._EXTERNALIZE_SPECS = [s for s in _macos._EXTERNALIZE_SPECS
                             if s.composite_op_name not in {"scaled_dot_product_attention", "rope"}]
ART = Path(__file__).resolve().parent / "artifacts"
SCRATCH = os.environ.get("SCRATCH") or os.path.join(tempfile.gettempdir(), "voxcpm_scratch")


def _du(p):
    return subprocess.run(["du", "-sh", str(p)], capture_output=True, text=True).stdout.split()[0]


def _save(prog, out_dir):
    import coreai.runtime as rt
    prog.optimize()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aim = out_dir / f"{out_dir.name}.aimodel"
    prog.save_asset(aim, rt.AIModelAssetMetadata())
    return aim


def cos(a, b):
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


class FeatEncWrap(nn.Module):
    """LocEnc + enc_to_lm_proj -> curr_embed (the per-step base_lm feedback input)."""

    def __init__(self, enc, enc2lm):
        super().__init__()
        self.enc = enc
        self.enc2lm = enc2lm

    def forward(self, pred_feat):  # [1,1,2,64]
        return self.enc2lm(self.enc(pred_feat))


class VocoderWrap(nn.Module):
    """forward arg name must match the export ref-dict key ('latents')."""

    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, latents):
        return self.dec(latents)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=["feat_encoder", "vocoder"])
    ap.add_argument("--mode", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--vae-frames", type=int, default=12)
    a = ap.parse_args()
    DT = torch.float16 if a.mode == "fp16" else torch.float32
    NP = np.float16 if a.mode == "fp16" else np.float32
    REF = np.load(os.path.join(SCRATCH, "oracle_ref.npz"))

    snap = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--VoxCPM-0.5B/snapshots/*")))[-1]

    import coreai.runtime as rt
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())

    if a.which == "feat_encoder":
        ck = torch.load(snap + "/pytorch_model.bin", map_location="cpu", weights_only=True)
        sd = ck.get("state_dict", ck)
        enc = load_feat_encoder(sd, 4, DT)
        enc2lm = load_linear(sd, "enc_to_lm_proj.", 1024, 1024, DT)
        m = FeatEncWrap(enc, enc2lm).eval()
        ref = {"pred_feat": torch.zeros(1, 1, 2, 64, dtype=DT)}
        prog = export_to_coreai(m, ref, dynamic_shapes=None,
                                input_names=("pred_feat",), output_names=("curr_embed",), state_names=None)
        ddir = ART / f"voxcpm_feat_encoder_{a.mode}"
        aim = _save(prog, ddir)
        print(f"  -> {ddir} ({_du(aim)})", flush=True)
        fn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
        n = sum(1 for k in REF.files if k.startswith("base_step.in_0__"))
        cs = []
        for i in range(n):
            x = REF[f"feat_enc.in_0__{i+1}"].astype(NP)  # decode-step encoder input (idx0=prefill)
            res = await fn(inputs={"pred_feat": rt.NDArray(np.ascontiguousarray(x))})
            c = cos(res["curr_embed"].numpy(), REF[f"base_step.in_0__{i}"])
            cs.append(c)
            print(f"  step[{i}] cos={c:.6f} {'OK' if c>=0.99 else 'FAIL'}")
        lo = min(cs)
    else:
        avck = torch.load(snap + "/audiovae.pth", map_location="cpu", weights_only=True)
        m = VocoderWrap(load_audio_vae(avck.get("state_dict", avck), DT)).eval()
        Tf = a.vae_frames
        ref = {"latents": torch.zeros(1, 64, Tf, dtype=DT)}
        prog = export_to_coreai(m, ref, dynamic_shapes=None,
                                input_names=("latents",), output_names=("wav",), state_names=None)
        ddir = ART / f"voxcpm_vocoder_{a.mode}_t{Tf}"
        aim = _save(prog, ddir)
        print(f"  -> {ddir} ({_du(aim)})", flush=True)
        fn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
        z = REF["vae_dec.in_0__0"].astype(NP)  # [1,64,12]
        res = await fn(inputs={"latents": rt.NDArray(np.ascontiguousarray(z))})
        wav = res["wav"].numpy()
        rawc = cos(wav, REF["vae_dec.out__0"])
        peak = float(np.abs(wav).max())
        print(f"  raw-wav cos={rawc:.6f}  peak={peak:.3f}  (peak~0 => ConvTranspose-zeros trap)")
        lo = rawc
    print(f"\n>>> {a.which} engine vs oracle: cos={lo:.6f} ({a.mode}) -> {'GATE PASS' if lo>=0.99 else 'GATE FAIL'}")
    sys.exit(0 if lo >= 0.99 else 1)


if __name__ == "__main__":
    asyncio.run(main())
