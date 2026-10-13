#!/usr/bin/env python3
"""MiniCPM-V-4.7-1B oracle margins: the fp32 oracle's top-1 and softmax top-2 gap at every generated step.

The engine gates of round 3 free-run greedy and judge the first divergence from the oracle's tokens
by the rule of `conversion/coreai_gate.py`: a divergence passes only where the oracle's softmax
top-2 gap (p_top1 - p_top2) is < 0.1 (a knife edge). `oracle.npz` keeps the full logits of the last
prompt position only (P1..P4) or of the last 48 positions (P0), so the gap at generated step 2 and
later is not in it. This script adds it, teacher-forced, from the same model, weights and inputs
the oracle used:

  per prompt, ONE forward over prompt + the oracle's generated ids (`<P>__gen_ids`),
  transformers 5.19.0 `MiniCPMV4_7ForConditionalGeneration`, fp32, CPU, one thread, eager, offline;
  image prompts get the oracle's own pixels (`<P>__pixel_values`, the processor's packing),
  `<P>__target_sizes`, and `mm_token_type_ids` = the prompt's + 0 for every generated token;
  `rope_deltas` reset before every forward (ROUND1 trap). Generated step k is read at position
  S - 1 + k (S = prompt length): float64 softmax over the full vocabulary, top-1 / top-2 ids and
  probabilities, gap = p1 - p2.

Checks written next to the numbers: the forward's top-1 equals the oracle's greedy token at every
step (self-consistency), the last prompt position's logits against `<P>__logits` (P1..P4) / the
48 logits rows against `P0__logits` (P0), the merger output against `<P>__image_features`, and the
rotary planes the text model received against `<P>__planes` + `<P>__gen_rotary_planes`.

Writes `<oracle>/margins.npz` (`<P>__margins`, `__top1`, `__top2`, `__p1`, `__p2`, `__positions`,
`__gen_ids`; P0 also `P0__gap_all` / `P0__top1_all` / `P0__top2_all` over all 67 positions) and
`<oracle>/margins.json` (method, per-prompt step and knife-edge counts, checks, versions).

    K/.venv-oracle/bin/python -I conversion/minicpmv47/oracle_margins.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _paths import hf_snapshot, work_path  # noqa: E402
import oracle_minicpmv47 as O  # noqa: E402  (sets HF_HUB_OFFLINE=1)

import numpy as np  # noqa: E402

KNIFE_EDGE = 0.1
PROMPTS = ("P0", "P1", "P2", "P3", "P4")


def softmax_top2(logits) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """[n, V] logits -> top-1 ids, top-2 ids, p1, p2 (float64 softmax over the full vocabulary)."""
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(axis=-1, keepdims=True)
    order = np.argsort(-p, axis=-1, kind="stable")[:, :2]
    rows = np.arange(p.shape[0])
    return order[:, 0], order[:, 1], p[rows, order[:, 0]], p[rows, order[:, 1]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle-dir", default=str(work_path("_minicpmv47", "oracle")))
    args = ap.parse_args()

    import torch

    torch.set_num_threads(O.THREADS)
    torch.set_num_interop_threads(O.THREADS)
    oracle_dir = Path(args.oracle_dir)
    z = np.load(oracle_dir / "oracle.npz")
    snap = Path(hf_snapshot(O.HF_ID, revision=O.REVISION))
    t_all = time.perf_counter()
    model, load_info, load_secs = O.load_model(snap, torch)
    rot = {}
    hook_rot = model.model.language_model.rotary_emb.register_forward_pre_hook(
        lambda mod, a: rot.__setitem__("planes", a[1].detach().clone()))
    merged = {}
    hook_mrg = model.model.merger.register_forward_hook(
        lambda mod, a, out: merged.__setitem__("x", torch.cat(out, 0).detach().clone()))
    print(f"[load] {load_secs:.1f}s threads {torch.get_num_threads()} / {torch.get_num_interop_threads()}", flush=True)

    arrays: dict[str, np.ndarray] = {}
    record: dict[str, dict] = {}
    for P in PROMPTS:
        gen = z[f"{P}__gen_ids"].astype(np.int64)
        if P == "P0":
            full = z["P0__input_ids"].astype(np.int64)
            S = int(z["P0__prompt_len"])
            assert np.array_equal(full[S:], gen), "P0: input_ids = prompt + gen_ids"
            kw = {}
        else:
            prompt = z[f"{P}__input_ids"].astype(np.int64)
            S = len(prompt)
            full = np.concatenate([prompt, gen])
            mm = np.concatenate([z[f"{P}__mm_token_type_ids"], np.zeros(len(gen), np.int64)])
            kw = {"pixel_values": torch.from_numpy(z[f"{P}__pixel_values"]),
                  "target_sizes": torch.from_numpy(z[f"{P}__target_sizes"]),
                  "mm_token_type_ids": torch.from_numpy(mm)[None]}
        L = len(full)
        model.model.rope_deltas = None
        rot.clear()
        merged.clear()
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(input_ids=torch.from_numpy(full)[None], use_cache=False, **kw).logits[0].float().numpy()
        secs = time.perf_counter() - t0
        pos = np.arange(S - 1, S - 1 + len(gen))
        top1, top2, p1, p2 = softmax_top2(logits[pos])
        gap = p1 - p2
        knife = [int(k) for k in np.nonzero(gap < KNIFE_EDGE)[0]]
        rec = {"prompt_tokens": S, "sequence_tokens": L, "steps": len(gen), "forward_seconds": secs,
               "knife_edge_steps": knife, "knife_edge_count": len(knife),
               "knife_edge_gaps": [float(gap[k]) for k in knife],
               "min_gap": float(gap.min()), "min_gap_step": int(gap.argmin()),
               "self_consistency_top1_eq_gen": int((top1 == gen).sum()),
               "self_consistency_mismatch_steps": [int(k) for k in np.nonzero(top1 != gen)[0]],
               "gen_ends_with_stop": bool(len(gen) and int(gen[-1]) in O.STOP_IDS)}
        planes = rot["planes"][:, 0].numpy().astype(np.int64)            # [3, L]
        if P == "P0":
            ref = z["P0__logits"]                                          # positions 19..66
            rp = z["P0__logits_positions"]
            rec["logits_rows_vs_oracle_max_abs"] = float(np.abs(logits[rp] - ref).max())
            rec["logits_rows_vs_oracle_bit_equal"] = bool(np.array_equal(logits[rp], ref))
            rec["planes_equal_oracle"] = bool(np.array_equal(planes, z["P0__planes"]))
            t1a, t2a, p1a, p2a = softmax_top2(logits)
            arrays.update({"P0__gap_all": p1a - p2a, "P0__top1_all": t1a, "P0__top2_all": t2a})
            rec["argmax_all_equal_oracle"] = int((t1a == z["P0__argmax_all"]).sum())
        else:
            ref = z[f"{P}__logits"]
            rec["last_prompt_logits_vs_oracle_max_abs"] = float(np.abs(logits[S - 1] - ref).max())
            rec["last_prompt_logits_vs_oracle_bit_equal"] = bool(np.array_equal(logits[S - 1], ref))
            feats = merged["x"].numpy()
            rec["image_features_vs_oracle_max_abs"] = float(np.abs(feats - z[f"{P}__image_features"]).max())
            rec["image_features_bit_equal"] = bool(np.array_equal(feats, z[f"{P}__image_features"]))
            g = z[f"{P}__gen_rotary_planes"]                               # prefill S + (n_gen - 1) steps
            rec["planes_prompt_equal_oracle"] = bool(np.array_equal(planes[:, :S], z[f"{P}__planes"]))
            rec["planes_equal_generate_rotary"] = bool(np.array_equal(planes[:, :g.shape[1]], g))
        arrays.update({f"{P}__margins": gap, f"{P}__top1": top1, f"{P}__top2": top2, f"{P}__p1": p1,
                       f"{P}__p2": p2, f"{P}__positions": pos, f"{P}__gen_ids": gen})
        record[P] = rec
        print(f"[{P}] S={S} L={L} steps {len(gen)} knife-edge {len(knife)} {knife} min gap {rec['min_gap']:.4f}"
              f"@{rec['min_gap_step']} self-consistency {rec['self_consistency_top1_eq_gen']}/{len(gen)} "
              f"({secs:.1f}s)", flush=True)
    hook_rot.remove()
    hook_mrg.remove()

    out_npz = oracle_dir / "margins.npz"
    np.savez(out_npz, **arrays)
    meta = {
        "schema": "minicpmv47-oracle-margins/1",
        "purpose": "per generated step: the fp32 oracle's top-1 and softmax top-2 gap, for the first-divergence "
                   "rule of conversion/coreai_gate.py (a divergence passes only where the gap < 0.1)",
        "method": {
            "forward": "one teacher-forced forward per prompt over prompt + oracle.npz <P>__gen_ids",
            "model": "transformers MiniCPMV4_7ForConditionalGeneration, fp32, CPU, eager, offline",
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "image_inputs": "oracle.npz <P>__pixel_values (the processor's packing [1,3,14,14336]) + "
                            "<P>__target_sizes through the model's own tower (no image_features injected); "
                            "mm_token_type_ids = <P>__mm_token_type_ids + 0 per generated token",
            "rope_deltas": "reset to None before every forward",
            "readout": "generated step k at position S-1+k; float64 softmax over the full vocabulary; "
                       "gap = p_top1 - p_top2",
            "knife_edge": f"gap < {KNIFE_EDGE}",
        },
        "summary": {P: {"steps": r["steps"], "knife_edge": r["knife_edge_count"]} for P, r in record.items()},
        "prompts": record,
        "load_seconds": load_secs, "loading_info": load_info,
        "oracle_npz_sha256": O.sha256(oracle_dir / "oracle.npz"),
        "npz": {"path": str(out_npz), "sha256": O.sha256(out_npz), "keys": sorted(arrays)},
        "versions": O.versions(torch),
        "source": {"hf_id": O.HF_ID, "revision": O.REVISION, "snapshot": str(snap)},
        "script": {"path": "conversion/minicpmv47/oracle_margins.py", "sha256": O.sha256(Path(__file__))},
        "wall_seconds": time.perf_counter() - t_all,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (oracle_dir / "margins.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[done] {out_npz} summary {json.dumps(meta['summary'])} wall {meta['wall_seconds']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
