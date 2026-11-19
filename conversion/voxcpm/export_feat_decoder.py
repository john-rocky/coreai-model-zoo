# Community port — NOT an Apple model.
"""Export the feat_decoder CFM (LocDiT + unrolled 10-step euler) as ONE feed-forward .aimodel
(no state): (mu[1,1024], cond[1,64,2], z[1,64,2]) -> pred_feat[1,64,2], and engine-gate vs oracle.

  python export_feat_decoder.py [--mode fp16|fp32]

Reads oracle_ref.npz from $SCRATCH: set SCRATCH to the folder that holds it (unset: voxcpm_scratch
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coreai_models.export.macos as _macos  # noqa: E402
from coreai_models.export.macos import export_to_coreai  # noqa: E402
from feat_decoder import load_feat_decoder  # noqa: E402

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


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fp16", choices=["fp16", "fp32"])
    a = ap.parse_args()
    DT = torch.float16 if a.mode == "fp16" else torch.float32
    NP = np.float16 if a.mode == "fp16" else np.float32

    snap = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--VoxCPM-0.5B/snapshots/*")))[-1]
    ck = torch.load(snap + "/pytorch_model.bin", map_location="cpu", weights_only=True)
    sd = ck.get("state_dict", ck)
    m = load_feat_decoder(sd, 4, DT).eval()

    ref = {"mu": torch.zeros(1, 1024, dtype=DT),
           "cond": torch.zeros(1, 64, 2, dtype=DT),
           "z": torch.zeros(1, 64, 2, dtype=DT)}
    print(f"[feat_decoder] export ({a.mode}) ...", flush=True)
    prog = export_to_coreai(m, ref, dynamic_shapes=None,
                            input_names=("mu", "cond", "z"), output_names=("pred_feat",),
                            state_names=None)
    ddir = ART / f"voxcpm_feat_decoder_{a.mode}"
    aim = _save(prog, ddir)
    print(f"  -> {ddir} ({_du(aim)})", flush=True)

    # engine gate
    import coreai.runtime as rt
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    mdl = await rt.AIModel.load(str(aim), gpu)
    fn = mdl.load_function("main")
    REF = np.load(os.path.join(SCRATCH, "oracle_ref.npz"))
    n = sum(1 for k in REF.files if k.startswith("feat_dec.out__"))
    cs = []
    for i in range(n):
        res = await fn(inputs={
            "mu": rt.NDArray(np.ascontiguousarray(REF[f"feat_dec.in_mu__{i}"].astype(NP))),
            "cond": rt.NDArray(np.ascontiguousarray(REF[f"feat_dec.in_cond__{i}"].astype(NP))),
            "z": rt.NDArray(np.ascontiguousarray(REF[f"randn__{i}"].astype(NP)))})
        c = cos(res["pred_feat"].numpy(), REF[f"feat_dec.out__{i}"])
        cs.append(c)
        print(f"  step[{i}] cos={c:.6f} {'OK' if c>=0.99 else 'FAIL'}")
    lo = min(cs)
    print(f"\n>>> feat_decoder engine vs oracle: min cos={lo:.6f} ({a.mode}) -> "
          f"{'GATE PASS' if lo>=0.99 else 'GATE FAIL'}")
    sys.exit(0 if lo >= 0.99 else 1)


if __name__ == "__main__":
    asyncio.run(main())
