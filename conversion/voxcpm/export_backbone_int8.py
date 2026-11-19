# Community port — NOT an Apple model.
"""Export an int8 weight-only DECODE bundle for a VoxCPM MiniCPM4 backbone (base_lm or residual_lm).
Halves the backbone size (the ship/iPhone-memory driver) with ~fp16 quality. Weight-only symmetric
per-channel int8 via coreai-opt PT2E (no activation calibration), then the same static-KV export as
export_backbone.py. Only the decode bundle is produced — prefill runs through it (prefill-via-decode).

  python export_backbone_int8.py --which base   # -> artifacts/voxcpm_base_int8_decode_cl512
  python export_backbone_int8.py --which res

The base gate reads oracle_ref.npz from $SCRATCH: set SCRATCH to the folder that holds it (unset: voxcpm_scratch
under the system temp folder).
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coreai_models.export.macos as _macos  # noqa: E402
from coreai_models.export.macos import export_to_coreai  # noqa: E402
from coreai_models.export.compression import quantize_pytorch_model  # noqa: E402
from minicpm4 import build_kv_state, load_backbone  # noqa: E402

_macos._EXTERNALIZE_SPECS = [s for s in _macos._EXTERNALIZE_SPECS
                             if s.composite_op_name not in {"scaled_dot_product_attention", "rope"}]
ART = HERE / "artifacts"
DT, H, CL = torch.float16, 1024, 512

# Weight-only symmetric per-channel int8 (coreai-opt PT2E, no activation/calibration).
INT8_CFG = {"execution_mode": "eager", "global_config": {
    "op_state_spec": {"weight": {"dtype": "int8", "qscheme": "symmetric",
                                 "granularity": {"type": "per_channel", "axis": 0}}},
    "op_input_spec": None, "op_output_spec": None}}


class DecodeWrap(nn.Module):
    def __init__(self, bb):
        super().__init__()
        self.bb = bb

    def forward(self, inputs_embeds, pos, k_cache, v_cache):
        return self.bb.decode(inputs_embeds, pos, k_cache, v_cache)


def cos(a, b):
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="base", choices=["base", "res"])
    ap.add_argument("--cache-len", type=int, default=CL)
    a = ap.parse_args()

    snap = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--VoxCPM-0.5B/snapshots/*")))[-1]
    sd = torch.load(snap + "/pytorch_model.bin", map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    nl, pref, vocab = (24, "base_lm.", 73448) if a.which == "base" else (6, "residual_lm.", 0)
    bb = load_backbone(sd, pref, nl, vocab, a.cache_len, DT).to(DT).eval()
    kc, vc = build_kv_state(bb.cfg, a.cache_len, DT)

    inp = (torch.zeros(1, 1, H, dtype=DT), torch.tensor([9], dtype=torch.int32), kc.clone(), vc.clone())
    dyn = {"inputs_embeds": None, "pos": None, "k_cache": None, "v_cache": None}
    print(f"[{a.which}] int8 weight-only quantize ...", flush=True)
    qm = quantize_pytorch_model(DecodeWrap(bb).eval(), inp, dyn, dict(INT8_CFG))

    ref = {"inputs_embeds": torch.zeros(1, 1, H, dtype=DT), "pos": torch.tensor([9], dtype=torch.int32),
           "k_cache": kc.clone(), "v_cache": vc.clone()}
    prog = export_to_coreai(qm, ref, dynamic_shapes=None, input_names=("inputs_embeds", "pos"),
                            output_names=("hidden",), state_names=("keyCache", "valueCache"))
    prog.optimize()
    out = ART / f"voxcpm_{a.which}_int8_decode_cl{a.cache_len}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    import coreai.runtime as rt
    aim = out / f"{out.name}.aimodel"
    prog.save_asset(aim, rt.AIModelAssetMetadata())
    du = subprocess.run(["du", "-sh", str(out)], capture_output=True, text=True).stdout.split()[0]
    print(f"saved {aim} ({du})", flush=True)

    # gate: prefill-via-decode vs oracle (base only — res prefill input is the base output sequence)
    if a.which == "base":
        import engine_generate as eg
        REF = np.load(os.path.join(
            os.environ.get("SCRATCH") or os.path.join(tempfile.gettempdir(), "voxcpm_scratch"),
            "oracle_ref.npz"))
        emb = torch.tensor(REF["prefill_base.in_inputs_embeds__0"]).to(DT)
        gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
        fn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
        st = eg.new_state(nl)
        outs = []
        for t in range(emb.shape[1]):
            o = await eg.call(fn, {"inputs_embeds": eg.na(emb[:, t:t + 1, :].numpy()),
                                   "pos": eg.na(np.array([t], np.int32))}, state=st)
            outs.append(torch.tensor(o["hidden"].numpy())[:, 0, :])
        seq = torch.stack(outs, dim=1)
        c = cos(seq[:, -1], REF["prefill_base.out__0"][:, -1])
        print(f">>> int8 base decode-prefill last-pos cos vs oracle = {c:.6f} "
              f"({'GATE PASS' if c >= 0.99 else 'GATE FAIL'})")


if __name__ == "__main__":
    asyncio.run(main())
