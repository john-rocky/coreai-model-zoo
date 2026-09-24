"""Gate: the re-authored graph (n3d_model.N3DGraph, fp32 eager, T=541) against transformers' per-step
chunk logits, on every captured low-latency step of both fixtures. The step's `chunk_input_embeds`
[L, 512] fill rows [0, L) of `packed`; rows [L, 541) are padding, filled two ways: zeros, and seeded
N(0, 1) noise. Compared: logits[:L*8] vs the captured `chunk_logits` [L*8, 8].

    PASS: every step, both paddings: max|Δlogit| <= 1e-4 and corr >= 0.999999

Negative controls (noise padding, on a subset of steps that holds L < 541, L = 541 and the first
compressed steps): RoPE off / key mask off / padding rows not zeroed after proj. Each must FAIL the
same bar on at least one step; a control that stays green means the gate cannot see that defect.

Run in the export venv (plain torch, no transformers):

    ~/code/coreai/coreai-models/.venv/bin/python gate_reauthor.py [--fixtures diarization_example,test_multispk]

Per-step numbers go to _work/gate_reauthor.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from n3d_model import HID, SUB, T_STREAM, load_checkpoint  # noqa: E402

WORK = HERE / "_work"
MAX_ABS, MIN_CORR = 1e-4, 0.999999
NOISE_SEED = 1234


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel() - a.mean(dtype=np.float64)
    b = b.astype(np.float64).ravel() - b.mean(dtype=np.float64)
    return float((a @ b) / np.sqrt((a @ a) * (b @ b)))


def pack(emb: np.ndarray, T: int, pad: str, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    L = emb.shape[0]
    packed = np.zeros((1, T, HID), np.float32)
    if pad == "noise":
        rng = np.random.default_rng(NOISE_SEED + step)
        packed[0, L:] = rng.standard_normal((T - L, HID)).astype(np.float32)
    packed[0, :L] = emb
    valid = np.zeros((1, T), np.float32)
    valid[0, :L] = 1.0
    return torch.from_numpy(packed), torch.from_numpy(valid)


def run_step(graph, d, i, pad):
    emb = d[f"s{i:03d}_inputs_embeds"]
    ref = d[f"s{i:03d}_chunk_logits"]
    L = emb.shape[0]
    packed, valid = pack(emb, graph.T, pad, i)
    with torch.inference_mode():
        out = graph(packed, valid)[0, : L * SUB].numpy()
    return float(np.abs(out - ref).max()), corr(out, ref), L


def control_steps(d) -> list[int]:
    n = int(d["n_steps"])
    L = d["L"]
    comp = np.nonzero(d["compressed_after"])[0]
    pick = {0, 1, n - 1}
    full = np.nonzero(L == T_STREAM)[0]
    if full.size:
        pick.add(int(full[0]))
    if comp.size:
        pick.update({int(comp[0]), min(int(comp[0]) + 1, n - 1)})
    short = np.nonzero(L < T_STREAM)[0]
    if short.size:
        pick.add(int(short[short.size // 2]))
    return sorted(pick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default="diarization_example,test_multispk")
    ap.add_argument("--safe-ln", action="store_true",
                    help="gate the SafeLayerNorm variant (n3d_model.SafeLayerNorm); numbers go to gate_reauthor_safeln.json")
    args = ap.parse_args()

    t_start = time.time()
    graph, _ = load_checkpoint(T=T_STREAM, safe_ln=args.safe_ln)
    controls = {
        "rope_off": load_checkpoint(T=T_STREAM, verify_sha=False, use_rope=False)[0],
        "key_mask_off": load_checkpoint(T=T_STREAM, verify_sha=False, use_key_mask=False)[0],
        "no_pad_zeroing": load_checkpoint(T=T_STREAM, verify_sha=False, zero_pad_rows=False)[0],
    }
    print(f"[weights] strict load OK (sha256 verified), T={T_STREAM}, safe_ln={args.safe_ln}", flush=True)

    report = {"bar": {"max_abs": MAX_ABS, "min_corr": MIN_CORR}, "fixtures": {}}
    all_ok, controls_red = True, {k: False for k in controls}
    for name in args.fixtures.split(","):
        d = np.load(WORK / f"chunk_io_{name}_ll.npz")
        n = int(d["n_steps"])
        rows = []
        for i in range(n):
            row = {"step": i, "L": int(d["L"][i]), "compressed_after": bool(d["compressed_after"][i])}
            for pad in ("zero", "noise"):
                mx, c, _ = run_step(graph, d, i, pad)
                row[pad] = {"max_abs": mx, "corr": c, "pass": mx <= MAX_ABS and c >= MIN_CORR}
            rows.append(row)
        ok = all(r[p]["pass"] for r in rows for p in ("zero", "noise"))
        all_ok &= ok
        summary = {p: {"worst_max_abs": max(r[p]["max_abs"] for r in rows),
                       "min_corr": min(r[p]["corr"] for r in rows),
                       "n_fail": sum(not r[p]["pass"] for r in rows)} for p in ("zero", "noise")}
        L = d["L"]
        comp = np.nonzero(d["compressed_after"])[0]
        print(f"[{name}] {n} steps (L=541: {int((L == T_STREAM).sum())}, first compressed after step "
              f"{int(comp[0]) if comp.size else 'never'}, L range {int(L.min())}..{int(L.max())})")
        for p in ("zero", "noise"):
            s = summary[p]
            print(f"  pad={p:5s} worst max|Δlogit| {s['worst_max_abs']:.3e}  min corr {s['min_corr']:.9f}  "
                  f"fail {s['n_fail']}/{n}")

        ctl = {}
        for cname, cg in controls.items():
            crow = []
            for i in control_steps(d):
                mx, c, Li = run_step(cg, d, i, "noise")
                crow.append({"step": i, "L": Li, "max_abs": mx, "corr": c,
                             "fail": not (mx <= MAX_ABS and c >= MIN_CORR)})
            red = any(r["fail"] for r in crow)
            controls_red[cname] |= red
            ctl[cname] = crow
            print(f"  control {cname:15s} " + " ".join(
                f"s{r['step']}(L{r['L']}):{r['max_abs']:.2e}/{r['corr']:.6f}{'R' if r['fail'] else 'g'}"
                for r in crow) + f" -> {'RED' if red else 'GREEN (instrument blind)'}")
        report["fixtures"][name] = {"steps": rows, "summary": summary, "controls": ctl}

    report["pass"] = bool(all_ok)
    report["controls_red"] = controls_red
    report["seconds"] = round(time.time() - t_start, 1)
    report["safe_ln"] = bool(args.safe_ln)
    (WORK / ("gate_reauthor_safeln.json" if args.safe_ln else "gate_reauthor.json")).write_text(json.dumps(report, indent=1))
    print(f"controls red: {controls_red}")
    print(f"gate_reauthor: {'PASS' if all_ok and all(controls_red.values()) else 'FAIL'} "
          f"({report['seconds']} s)")
    raise SystemExit(0 if all_ok and all(controls_red.values()) else 1)


if __name__ == "__main__":
    main()
