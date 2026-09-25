"""Write the host constants a Swift (or any non-Python) host needs next to the three bundles.

  host/rope_axis{0,1,2}_{cos,sin}.f32   per-axis RoPE tables, fp32 little-endian, row-major
                                        [9216, P_a] with P = (8, 28, 28): row r = position r - 1024,
                                        i.e. positions -1024..8191 — the range ``QwenImage21Rope``
                                        itself tabulates (``arange(8192)`` + ``arange(1024)`` negated)
  host/scheduler.json                   the sampler constants of ``qi21_sched.py`` + the step count

The tables are ``qi21_host._axis_table`` (the same ``torch.polar(1, outer(pos, inv_freq))`` expression
the DiT gates use) evaluated once over the whole position range. A host builds a token's 64 cos/sin
pairs by concatenating ``axis0[f + 1024] | axis1[h + 1024] | axis2[w + 1024]`` for its (frame, height,
width) position — see ``host_contract.md`` for how the positions are assigned.

Gates (all must be bit-exact, max|d| = 0):
  1. table lookups == ``qi21_host.rope_tables(L, H, W)`` for several (L, H, W), including the
     largest text length (512) and image grid (64x64) the DiT graph accepts;
  2. the tables == ``QwenImage21Rope.freqs`` of diffusers main (run with ``.venv-qi21``; skipped with
     a note when diffusers is not importable);
  3. ``scheduler.json`` reproduces ``qi21_sched.schedule`` (sigmas and timesteps) for 256² / 512² / 1024².

Run (from conversion/qwenimage21/; ``.venv-qi21`` runs gate 2 as well):
  ~/code/coreai/coreai-models/.venv-qi21/bin/python make_host_consts.py --out <staging>/host
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from qi21_host import AXES_DIM, THETA, _axis_table, rope_positions, rope_tables  # noqa: E402
from qi21_sched import SCHED_CFG, calculate_shift, schedule  # noqa: E402

POS_MIN, POS_MAX = -1024, 8191          # QwenImage21Rope: neg_index -1024..-1, pos_index 0..8191
STEPS = 40                              # the pipeline default (num_inference_steps)
SHAPES = [(18, 16, 16), (18, 32, 32), (33, 64, 64), (512, 64, 64), (8, 8, 8), (20, 16, 32), (9, 4, 4)]


def build_tables() -> dict[str, np.ndarray]:
    pos = torch.arange(POS_MIN, POS_MAX + 1, dtype=torch.long)
    out = {}
    for a, dim in enumerate(AXES_DIM):
        c, s = _axis_table(pos, dim, THETA)
        out[f"rope_axis{a}_cos"] = np.ascontiguousarray(c.numpy(), dtype="<f4")
        out[f"rope_axis{a}_sin"] = np.ascontiguousarray(s.numpy(), dtype="<f4")
    return out


def lookup(tables: dict[str, np.ndarray], L: int, H: int, W: int):
    """What the host does: gather one row per axis and concatenate -> [1,L,64] / [1,N,64] cos and sin."""
    pos = [p.numpy() - POS_MIN for p in rope_positions(L, H, W)]
    cos = np.concatenate([tables[f"rope_axis{a}_cos"][pos[a]] for a in range(3)], axis=-1)[None]
    sin = np.concatenate([tables[f"rope_axis{a}_sin"][pos[a]] for a in range(3)], axis=-1)[None]
    return cos[:, :L], sin[:, :L], cos[:, L:], sin[:, L:]


def gate_lookup(tables) -> bool:
    ok = True
    for L, H, W in SHAPES:
        ref = [t.numpy() for t in rope_tables(L, H, W)]
        got = lookup(tables, L, H, W)
        d = max(float(np.abs(g.astype(np.float64) - r).max()) for g, r in zip(got, ref))
        same = all(np.array_equal(g, r) for g, r in zip(got, ref))
        ok &= same
        print(f"[gate 1] L {L:3d} H {H:3d} W {W:3d}: lookup vs qi21_host.rope_tables max|d| {d:.1e} "
              f"{'bit-exact' if same else 'DIFFERS'}", flush=True)
    return ok


def gate_diffusers(tables) -> bool | None:
    try:
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Rope
    except Exception as e:  # base venv: no diffusers main
        print(f"[gate 2] skipped: diffusers QwenImage21Rope not importable ({type(e).__name__})", flush=True)
        return None
    rope = QwenImage21Rope(theta=THETA, axes_dim=list(AXES_DIM))
    idx = torch.arange(POS_MIN, POS_MAX + 1)          # the reference indexes negatives from the end
    ok = True
    for a in range(3):
        ref = rope.freqs[a][idx]
        for part, r in (("cos", ref.real), ("sin", ref.imag)):
            same = np.array_equal(tables[f"rope_axis{a}_{part}"], r.numpy())
            ok &= same
            print(f"[gate 2] axis {a} {part} [{r.shape[0]}, {r.shape[1]}] vs QwenImage21Rope.freqs: "
                  f"{'bit-exact' if same else 'DIFFERS'}", flush=True)
    return ok


def scheduler_json() -> dict:
    keys = ("base_image_seq_len", "max_image_seq_len", "base_shift", "max_shift", "shift_terminal",
            "time_shift_type", "num_train_timesteps")
    d = {k: SCHED_CFG[k] for k in keys}
    d["steps"] = STEPS
    return d


def gate_scheduler(cfg: dict) -> bool:
    """Re-derive the schedule from the json values alone and compare with qi21_sched.schedule."""
    import math
    ok = True
    for size in (256, 512, 1024):
        n = (size // 16) ** 2
        mu = calculate_shift(n, cfg["base_image_seq_len"], cfg["max_image_seq_len"], cfg["base_shift"],
                             cfg["max_shift"])
        s = np.linspace(1.0, 1 / cfg["steps"], cfg["steps"]).astype(np.float32)
        e = math.exp(mu)
        s = e / (e + (1 / s - 1) ** 1.0)        # time_shift_type "exponential", sigma 1.0
        one_minus = 1 - s
        s = (1 - one_minus / (one_minus[-1] / (1 - cfg["shift_terminal"]))).astype(np.float32)
        ts = (s * np.float32(cfg["num_train_timesteps"])).astype(np.float32)
        ref_sig, ref_ts, ref_mu = schedule(cfg["steps"], n)
        same = np.array_equal(s, ref_sig[:-1]) and np.array_equal(ts, ref_ts) and mu == ref_mu
        ok &= same
        print(f"[gate 3] {size}²: N {n} mu {mu:.10f} sigma_1 {s[0]:.8f} sigma_40 {s[-1]:.8f}: "
              f"{'bit-exact' if same else 'DIFFERS'} vs qi21_sched.schedule", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the host/ dir of the staged repo")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = build_tables()
    ok1 = gate_lookup(tables)
    ok2 = gate_diffusers(tables)
    cfg = scheduler_json()
    ok3 = gate_scheduler(cfg)
    if not (ok1 and ok2 is not False and ok3):
        print("[host] FAIL — nothing written", flush=True)
        return 1
    for name, arr in tables.items():
        arr.tofile(out / f"{name}.f32")
        print(f"[host] {name}.f32  [{arr.shape[0]}, {arr.shape[1]}] fp32 LE  {arr.nbytes} bytes", flush=True)
    (out / "scheduler.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"[host] scheduler.json {cfg}", flush=True)
    print(f"[host] PASS (gate 2 {'ran' if ok2 is not None else 'skipped'}) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
