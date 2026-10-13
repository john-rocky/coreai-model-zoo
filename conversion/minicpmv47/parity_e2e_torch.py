#!/usr/bin/env python3
"""fp32 torch end to end: the re-authored vision tower -> the VL decoder module vs the HF fp32 oracle (P1..P4).

Nothing from the oracle enters the model except the pixels and the token ids: the tower
(`minicpmv47_vision.MiniCPMV47Vision`, baked path (b), from the checkpoint) turns
`oracle.npz <P>__pixel_values` (unpacked to [1, 3, 448, 448]) into `image_embeds` [64, 1024], and
`minicpmv47_vl_pipelined.MiniCPMV47VLPipelinedForCausalLM` (round 1, from the checkpoint) takes the
processor ids through `host_static_inputs` (64 <|image_pad|> -> V + k, `rope_shift_start` /
`rope_shift_amount` by the canvas rule).

  logits   one forward over the whole prompt ([1, S], fresh zero states sized S + 8, GDN on the
           reference path): the last position vs `<P>__logits` (cosine in float64, max |d|) and its
           ordered top-5 vs `<P>__top5_ids[-1]`; for scale, the same forward with the oracle's own
           tower rows (`<P>__image_features`) and the difference the tower swap alone makes.
  greedy   then S = 1 stateful steps on the same states (position ramp 0..p): the argmax fed back,
           min(8, len(<P>__gen_ids)) tokens vs the oracle's greedy `<P>__gen_ids`.
  planes   the module's in-graph (T, H, W) planes on the prompt and on the rows the greedy loop fed
           vs the planes the HF rotary received (`<P>__planes`, `<P>__gen_rotary_planes`) — round 1's gate.
  Bars (round-2 launch): top-5 4/4, cos >= 0.9999, greedy 4/4, planes 100 %.

    QW coreai-models/.venv/bin/python -I conversion/minicpmv47/parity_e2e_torch.py [--threads 1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _paths import hf_snapshot, work_path  # noqa: E402
from parity_vl_torch import logit_cmp, plane_cmp  # noqa: E402  (round 1's gate functions)

HF_ID = "openbmb/MiniCPM-V-4.7-1B"
REVISION = "984a89aafcd5160164f4f0d04b4be0f4eec03cb8"
PROMPTS = ("P1", "P2", "P3", "P4")
BAR_COS = 0.9999
MAX_GREEDY = 8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(args) -> dict:
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    from minicpmv47_vision import GRID, MiniCPMV47Vision, unpack_patches
    from minicpmv47_vl_pipelined import MiniCPMV47VLPipelinedForCausalLM, host_static_inputs

    from coreai_models.models.macos import qwen3_5 as q

    snap = hf_snapshot(HF_ID, revision=REVISION)
    z = np.load(Path(args.oracle_dir) / "oracle.npz")
    t0 = time.perf_counter()
    tower = MiniCPMV47Vision.from_checkpoint(snap, torch.float32).bake_constants(GRID)
    vlm = MiniCPMV47VLPipelinedForCausalLM.from_checkpoint(snap, torch.float32)
    load_seconds = time.perf_counter() - t0
    cfg = vlm.config
    V = cfg.vocab_size
    print(f"[load] tower {tower.load_report['checkpoint_tower_merger_keys']} tensors, decoder "
          f"{vlm.load_report['checkpoint_text_keys']} tensors, {load_seconds:.1f}s", flush=True)

    def planes(ids_row, start, amount) -> np.ndarray:
        ids = ids_row.reshape(1, -1).to(torch.int32)
        p = torch.arange(ids.shape[1], dtype=torch.int32).unsqueeze(0)
        is_img = ids >= V
        slot = (ids - V).clamp(0, vlm.n_image - 1)
        t, h, w = vlm._rope_planes(ids, is_img, slot, p, start, amount)
        return torch.stack([t[0], h[0], w[0]]).numpy().astype(np.int64)

    out = {"prompts": {}}
    for P in PROMPTS:
        rec = {}
        pv = unpack_patches(torch.from_numpy(z[f"{P}__pixel_values"]), GRID, GRID)
        t0 = time.perf_counter()
        with torch.inference_mode():
            emb = tower(pv)                                              # [64, 1024], baked path
        rec["tower_seconds"] = time.perf_counter() - t0
        emb_hf = torch.from_numpy(z[f"{P}__image_features"]).float()
        e64, h64 = emb.double(), emb_hf.double()
        cos_rows = (e64 * h64).sum(-1) / (e64.norm(dim=-1) * h64.norm(dim=-1))
        rec["tower_vs_oracle"] = {"cos_min": float(cos_rows.min()), "max_abs_diff": float((e64 - h64).abs().max())}

        ids = torch.from_numpy(z[f"{P}__input_ids"])
        mapped, start, amount = host_static_inputs(ids)
        S = int(ids.numel())
        gen_ref = z[f"{P}__gen_ids"].tolist()
        n_gen = min(MAX_GREEDY, len(gen_ref))
        rec.update({"tokens": S, "start": int(start[0]), "amount": int(amount[0]), "greedy_tokens": n_gen})

        st = q.build_decode_state(cfg, max_seq_len=S + MAX_GREEDY, dtype=torch.float32)
        state = (st["k_cache"], st["v_cache"], st["conv_state"], st["rec_state"])
        x_ids = mapped.reshape(1, -1).to(torch.int32)
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = vlm(x_ids, torch.arange(S, dtype=torch.int32).unsqueeze(0), emb, start, amount, *state)[0]
        rec["prompt_forward_seconds"] = time.perf_counter() - t0
        ref_last = torch.from_numpy(z[f"{P}__logits"]).float()[None]
        rec["logits_last"] = logit_cmp(logits[-1:].float(), ref_last)
        rec["top5_last_equal_oracle_top5_ids"] = bool(
            torch.topk(logits[-1], 5).indices.tolist() == z[f"{P}__top5_ids"][-1].tolist())
        rec["top5_all_positions_equal_oracle"] = int(
            (torch.topk(logits, 5, dim=-1).indices.numpy() == z[f"{P}__top5_ids"]).all(-1).sum())

        # greedy on the same states: S = 1 steps, position ramp 0..p
        gen = [int(torch.argmax(logits[-1]))]
        t0 = time.perf_counter()
        with torch.inference_mode():
            for i in range(1, n_gen):
                p = S + i - 1                                            # position of the fed token
                step = vlm(torch.tensor([[gen[-1]]], dtype=torch.int32),
                           torch.arange(p + 1, dtype=torch.int32).unsqueeze(0), emb, start, amount, *state)
                gen.append(int(torch.argmax(step[0, -1])))
        rec["greedy_seconds"] = time.perf_counter() - t0
        rec["greedy"] = {"ids": gen, "oracle": gen_ref[:n_gen], "equal": gen == gen_ref[:n_gen],
                         "first_mismatch": next((i for i, (a, b) in enumerate(zip(gen, gen_ref)) if a != b), None)}

        # planes: prompt, and the rows the greedy loop fed (prompt + gen[:-1])
        rec["planes_prompt"] = plane_cmp(planes(mapped, start, amount), z[f"{P}__planes"], ids.numpy())
        fed = torch.cat([mapped.long(), torch.tensor(gen[:-1], dtype=torch.long)])
        g_want = z[f"{P}__gen_rotary_planes"][:, : fed.numel()]
        rec["planes_generate"] = plane_cmp(planes(fed, start, amount), g_want, fed.numpy())

        # for scale: the same prompt forward with the oracle's own tower rows
        st2 = q.build_decode_state(cfg, max_seq_len=S, dtype=torch.float32)
        with torch.inference_mode():
            logits_hf = vlm(x_ids, torch.arange(S, dtype=torch.int32).unsqueeze(0), emb_hf, start, amount,
                            st2["k_cache"], st2["v_cache"], st2["conv_state"], st2["rec_state"])[0]
        rec["logits_last_with_oracle_tower_rows"] = logit_cmp(logits_hf[-1:].float(), ref_last)
        rec["logits_last_mine_vs_oracle_rows_max_abs"] = float((logits[-1] - logits_hf[-1]).abs().max())

        ll = rec["logits_last"]
        print(f"[{P}] S={S} tower cos min {rec['tower_vs_oracle']['cos_min']:.10f} | last top5 "
              f"{'=' if rec['top5_last_equal_oracle_top5_ids'] else 'NO'} cos {ll['cos_min']:.10f} max|d| "
              f"{ll['max_abs_diff']:.3e} (ref absmax {ll['ref_absmax']:.2f}; oracle rows: cos "
              f"{rec['logits_last_with_oracle_tower_rows']['cos_min']:.10f} max|d| "
              f"{rec['logits_last_with_oracle_tower_rows']['max_abs_diff']:.3e}) | greedy {n_gen}: "
              f"{'=' if rec['greedy']['equal'] else 'NO'} {gen} | planes {rec['planes_prompt']['equal']}/"
              f"{rec['planes_prompt']['positions']} + {rec['planes_generate']['equal']}/"
              f"{rec['planes_generate']['positions']} | {rec['prompt_forward_seconds']:.1f}s + "
              f"{rec['greedy_seconds']:.1f}s", flush=True)
        out["prompts"][P] = rec
    out["load_seconds"] = load_seconds
    out["load_report"] = {"tower": tower.load_report, "decoder": vlm.load_report}
    return out


def versions() -> dict:
    import importlib.metadata as md

    import torch
    v = {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
         "platform": platform.platform(), "threads": torch.get_num_threads(),
         "interop_threads": torch.get_num_interop_threads()}
    for pkg in ("transformers", "coreai-core", "coreai-torch", "coreai-opt"):
        try:
            v[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            v[pkg] = None
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--oracle-dir", default=str(work_path("_minicpmv47", "oracle")))
    ap.add_argument("--out", default=str(work_path("_minicpmv47", "oracle", "parity_e2e_torch.json")))
    args = ap.parse_args()
    t_all = time.perf_counter()
    res = run(args)
    pr = res["prompts"]
    top5_ok = all(pr[P]["top5_last_equal_oracle_top5_ids"] and pr[P]["logits_last"]["top5_equal"] == 1 for P in PROMPTS)
    cos_ok = all(pr[P]["logits_last"]["cos_min"] >= BAR_COS for P in PROMPTS)
    greedy_ok = all(pr[P]["greedy"]["equal"] for P in PROMPTS)
    planes_ok = all(pr[P][k]["equal"] == pr[P][k]["positions"] for P in PROMPTS
                    for k in ("planes_prompt", "planes_generate"))
    summary = {
        "top5_equal_last_position": f"{sum(pr[P]['top5_last_equal_oracle_top5_ids'] for P in PROMPTS)}/{len(PROMPTS)}",
        "cos_min_last_position": min(pr[P]["logits_last"]["cos_min"] for P in PROMPTS),
        "max_abs_diff_last_position": max(pr[P]["logits_last"]["max_abs_diff"] for P in PROMPTS),
        "greedy_equal": f"{sum(pr[P]['greedy']['equal'] for P in PROMPTS)}/{len(PROMPTS)}",
        "greedy_tokens": {P: pr[P]["greedy_tokens"] for P in PROMPTS},
        "planes": {P: f"{pr[P]['planes_prompt']['equal'] + pr[P]['planes_generate']['equal']}/"
                      f"{pr[P]['planes_prompt']['positions'] + pr[P]['planes_generate']['positions']}" for P in PROMPTS},
        "tower_cos_min": min(pr[P]["tower_vs_oracle"]["cos_min"] for P in PROMPTS),
        "with_oracle_tower_rows_cos_min": min(pr[P]["logits_last_with_oracle_tower_rows"]["cos_min"] for P in PROMPTS),
        "tower_swap_logits_max_abs": max(pr[P]["logits_last_mine_vs_oracle_rows_max_abs"] for P in PROMPTS),
    }
    record = {
        "schema": "minicpmv47-parity-e2e-torch/1",
        "purpose": "re-authored tower (baked path) -> round-1 VL decoder module, fp32 CPU, vs the HF fp32 oracle: "
                   "last-position logits, greedy, planes",
        "result": "PASS" if top5_ok and cos_ok and greedy_ok and planes_ok else "FAIL",
        "gates": {"top5": top5_ok, "cos": cos_ok, "greedy": greedy_ok, "planes": planes_ok},
        "bars": {"top5_equal_last_position": "4/4", "cos_min": BAR_COS,
                 "greedy": f"min({MAX_GREEDY}, len(gen_ids)) tokens equal, 4/4", "planes": "100 %"},
        "summary": summary,
        "modules": {"tower": {"path": "conversion/minicpmv47/minicpmv47_vision.py",
                              "sha256": sha256(HERE / "minicpmv47_vision.py"), "path_used": "baked (b), grid 32"},
                    "decoder": {"path": "conversion/minicpmv47/minicpmv47_vl_pipelined.py",
                                "sha256": sha256(HERE / "minicpmv47_vl_pipelined.py")}},
        "script": {"path": "conversion/minicpmv47/parity_e2e_torch.py", "sha256": sha256(Path(__file__))},
        **res,
        "oracle_npz_sha256": sha256(Path(args.oracle_dir) / "oracle.npz"),
        "versions": versions(),
        "wall_seconds": time.perf_counter() - t_all,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
    print(f"[parity_e2e_torch] {record['result']} {json.dumps(summary)}", flush=True)
    print(f"[parity_e2e_torch] -> {args.out}")


if __name__ == "__main__":
    main()
