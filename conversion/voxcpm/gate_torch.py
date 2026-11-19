# Community port — NOT an Apple model.
"""Torch-ladder gate: MiniCPM4 overlay (base_lm 24L + residual_lm 6L) vs the VoxCPM oracle.

Loads the VoxCPM-0.5B checkpoint into the standalone overlay, replays the oracle's captured
prefill input + per-step decode inputs (continuous ``inputs_embeds`` + positions), and checks
cosine similarity of every hidden against ``oracle_ref.npz``. Pass = cos >= 0.999 everywhere.

Reads oracle_ref.npz from $SCRATCH: set SCRATCH to the folder that holds it (unset: voxcpm_scratch
under the system temp folder).
"""
from __future__ import annotations

import glob
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from minicpm4 import build_kv_state, load_backbone  # noqa: E402

SCRATCH = os.environ.get("SCRATCH") or os.path.join(tempfile.gettempdir(), "voxcpm_scratch")
REF = np.load(os.path.join(SCRATCH, "oracle_ref.npz"))
BUF = 64
DTYPE = torch.float32


def cos(a: torch.Tensor, b: np.ndarray) -> float:
    a = a.detach().float().reshape(-1)
    b = torch.tensor(b, dtype=torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def load_sd():
    snap = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--VoxCPM-0.5B/snapshots/*")))[-1]
    ck = torch.load(snap + "/pytorch_model.bin", map_location="cpu", weights_only=True)
    return ck.get("state_dict", ck)


def gate_backbone(sd, name, prefix, n_layers, vocab, prefill_in, prefill_out, step_in, step_pos, step_out):
    print(f"\n=== {name} ({n_layers}L, vocab={vocab}) ===")
    m = load_backbone(sd, prefix, n_layers, vocab, BUF, DTYPE)
    k_cache, v_cache = build_kv_state(m.cfg, BUF, DTYPE)

    # prefill
    pin = torch.tensor(REF[prefill_in], dtype=DTYPE)
    with torch.inference_mode():
        pout = m.prefill(pin, k_cache, v_cache)
    c = cos(pout, REF[prefill_out])
    print(f"  prefill  q={pin.shape[1]:2d}  cos={c:.6f}  {'OK' if c >= 0.999 else 'FAIL'}")
    results = [c]

    # decode steps
    n_steps = sum(1 for k in REF.files if k.startswith(step_out + "__"))
    for i in range(n_steps):
        emb = torch.tensor(REF[f"{step_in}__{i}"], dtype=DTYPE).reshape(1, 1, -1)
        pos = torch.tensor(REF[f"{step_pos}__{i}"], dtype=torch.int32).reshape(1)
        with torch.inference_mode():
            out = m.decode(emb, pos, k_cache, v_cache)
        c = cos(out, REF[f"{step_out}__{i}"])
        results.append(c)
        print(f"  decode[{i}] pos={int(pos.item()):2d}  cos={c:.6f}  {'OK' if c >= 0.999 else 'FAIL'}")
    return results


def main():
    sd = load_sd()
    all_c = []
    all_c += gate_backbone(sd, "base_lm", "base_lm.", 24, 73448,
                           "prefill_base.in_inputs_embeds__0", "prefill_base.out__0",
                           "base_step.in_0", "base_step.in_1", "base_step.out")
    all_c += gate_backbone(sd, "residual_lm", "residual_lm.", 6, 0,
                           "prefill_res.in_inputs_embeds__0", "prefill_res.out__0",
                           "res_step.in_0", "res_step.in_1", "res_step.out")
    lo = min(all_c)
    print(f"\n>>> min cos = {lo:.6f}  ->  {'GATE PASS' if lo >= 0.999 else 'GATE FAIL'}")
    sys.exit(0 if lo >= 0.999 else 1)


if __name__ == "__main__":
    main()
