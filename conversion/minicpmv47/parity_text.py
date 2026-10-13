#!/usr/bin/env python3
"""fp32 torch parity: the overlay's Qwen3.5 text decoder vs the MiniCPM-V-4.7-1B oracle (P0).

The text backbone of MiniCPM-V-4.7-1B is `qwen3_5_text` (24 layers, [linear x3, full] x6). This
script builds the overlay's `Qwen3_5ForCausalLM` (prefill, no KV, 1-D positions, GDN on the
reference while_loop path) from a hand-written `Qwen3_5Config` — the overlay venv's transformers
4.57.6 has no `minicpmv4_7`, so no AutoConfig — and reads the weights by hand from the pinned
safetensors: `model.language_model.*` -> `model.*`, and the checkpoint's own `lm_head.weight`
(4.7's head is untied; nothing is tied here). Every checkpoint text tensor must land on exactly
one module tensor and none may be left on the meta device.

Then one forward over the oracle's P0 row (prompt + 48 greedy tokens, `oracle.npz P0__input_ids`)
with positions 0..L-1, and the logits of the last 48 positions against the oracle's
(`P0__logits`, HF fp32 CPU one thread): ordered top-5 equal, argmax equal, cosine (float64) and
max |d| per position. Over all L positions: argmax and top-5 against the oracle's argmax/top-5.
Bars (round-1 launch): top-5 48/48, argmax 48/48, min cos >= 0.9999.

Red arm (same hidden states, no extra forward): the LM head replaced by the tied embedding table
(the 4.6 convention) must lose top-5 / argmax — the comparison sees a wrong head. The weight-level
distance between the two tables is recorded next to it (`head_vs_embed_weights`).

    QW coreai-models/.venv/bin/python conversion/minicpmv47/parity_text.py [--threads 1]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import hf_snapshot, work_path  # noqa: E402

HF_ID = "openbmb/MiniCPM-V-4.7-1B"
REVISION = "984a89aafcd5160164f4f0d04b4be0f4eec03cb8"
TEXT_PREFIX = "model.language_model."
BAR_COS = 0.9999
N_POS = 48


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_config():
    """MiniCPM-V-4.7-1B text_config (config.json), written out: no AutoConfig in this venv."""
    from coreai_models.models.macos.qwen3_5 import Qwen3_5Config

    lt = (["linear_attention"] * 3 + ["full_attention"]) * 6
    return Qwen3_5Config(
        hidden_size=1024, num_hidden_layers=24, vocab_size=248094, intermediate_size=3584,
        rms_norm_eps=1e-6, tie_word_embeddings=False, head_dim=256,
        num_attention_heads=8, num_key_value_heads=2, attn_output_gate=True,
        partial_rotary_factor=0.25, rope_theta=1e7,
        linear_num_key_heads=16, linear_num_value_heads=16,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
        full_attention_interval=4, layer_types=lt, mrope_section=(11, 11, 10),
    )


def load_text_weights(model, snap: Path, torch) -> dict:
    """model.language_model.* -> model.*, lm_head.weight as is; fp32. Returns the load report."""
    from safetensors import safe_open

    sd, table, skipped = {}, [], 0
    with safe_open(str(snap / "model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():  # noqa: SIM118
            if k.startswith(TEXT_PREFIX):
                local = "model." + k[len(TEXT_PREFIX):]
            elif k == "lm_head.weight":
                local = k
            else:
                skipped += 1                     # model.vision_tower.*, model.merger.*
                continue
            t = f.get_tensor(k)
            sd[local] = t.to(torch.float32)
            table.append([k, local, list(t.shape), str(t.dtype)])
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    model.model.reset_buffers()
    on_meta = [n for n, p in list(model.named_parameters()) + list(model.named_buffers()) if p.is_meta]
    own = dict(model.named_parameters())
    equal = sum(bool(torch.equal(own[local], sd[local])) for _, local, _, _ in table if local in own)
    return {"checkpoint_text_tensors": len(table), "checkpoint_other_tensors_skipped": skipped,
            "module_parameters": len(own), "missing": list(missing), "unexpected": list(unexpected),
            "on_meta_after_load": on_meta, "loaded_equal_checkpoint": equal,
            "lm_head_is_embed": bool(model.lm_head.weight is model.model.embed_tokens.weight),
            "weight_name_table": table}


def compare(mine, ref) -> dict:
    """mine / ref: [N, V] float32 logits at the same positions."""
    import torch

    m64, r64 = mine.double(), ref.double()
    cos = (m64 * r64).sum(-1) / (m64.norm(dim=-1) * r64.norm(dim=-1))
    t5m = torch.topk(mine, 5, dim=-1).indices
    t5r = torch.topk(ref, 5, dim=-1).indices
    top5_eq = (t5m == t5r).all(-1)
    top5_set = [set(a.tolist()) == set(b.tolist()) for a, b in zip(t5m, t5r)]
    arg_eq = t5m[:, 0] == t5r[:, 0]
    dmax = (m64 - r64).abs().amax(-1)
    first_bad = int((~top5_eq).nonzero()[0, 0]) if (~top5_eq).any() else None
    return {"positions": int(mine.shape[0]), "top5_equal": int(top5_eq.sum()),
            "top5_set_equal": int(sum(top5_set)), "argmax_equal": int(arg_eq.sum()),
            "cos_min": float(cos.min()), "cos_mean": float(cos.mean()),
            "max_abs_diff": float(dmax.max()), "mean_of_max_abs_diff": float(dmax.mean()),
            "ref_absmax": float(r64.abs().max()), "first_top5_mismatch_row": first_bad,
            "per_position": [{"cos": float(c), "max_abs_diff": float(d), "top5": t.tolist(), "top5_ref": u.tolist()}
                             for c, d, t, u in zip(cos, dmax, t5m, t5r)]}


def versions(torch) -> dict:
    import importlib.metadata as md

    import coreai_models
    import transformers
    out = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
           "numpy": np.__version__, "platform": platform.platform()}
    for pkg in ("coreai-core", "coreai-torch", "coreai-opt"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    src = Path(coreai_models.__file__).resolve().parent
    out["overlay_qwen3_5_py"] = {"path": str(src / "models/macos/qwen3_5.py"),
                                 "sha256": sha256(src / "models/macos/qwen3_5.py")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--oracle-dir", default=str(work_path("_minicpmv47", "oracle")))
    ap.add_argument("--out", default=str(work_path("_minicpmv47", "oracle", "parity_text.json")))
    args = ap.parse_args()

    import torch

    torch.set_num_threads(args.threads)
    from coreai_models.models.macos.qwen3_5 import Qwen3_5ForCausalLM

    oracle_dir = Path(args.oracle_dir)
    z = np.load(oracle_dir / "oracle.npz")
    meta = json.loads((oracle_dir / "oracle.json").read_text())
    snap = Path(hf_snapshot(HF_ID, revision=REVISION))

    cfg = build_config()
    t0 = time.perf_counter()
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(cfg)
    report = load_text_weights(model, snap, torch)
    model.eval()
    load_secs = time.perf_counter() - t0
    assert not report["missing"] and not report["unexpected"] and not report["on_meta_after_load"], \
        {k: report[k] for k in ("missing", "unexpected", "on_meta_after_load")}
    assert not report["lm_head_is_embed"]
    print(f"[load] {report['checkpoint_text_tensors']} text tensors, {report['loaded_equal_checkpoint']} equal, "
          f"{report['checkpoint_other_tensors_skipped']} other skipped, {load_secs:.1f}s", flush=True)

    ids = torch.from_numpy(z["P0__input_ids"]).long()[None]          # [1, L]
    L = int(ids.shape[1])
    pos = torch.arange(L)[None]
    rows = torch.from_numpy(z["P0__logits_positions"]).long()
    assert rows.tolist() == list(range(L - N_POS, L)), rows
    ref = torch.from_numpy(z["P0__logits"]).float()                   # [48, V]
    t0 = time.perf_counter()
    with torch.inference_mode():
        hidden = model.model(ids, pos)                                 # [1, L, h], final norm applied
        logits = model.lm_head(hidden)[0].float()                      # [L, V]
        tied = (hidden[0] @ model.model.embed_tokens.weight.T).float()  # red arm: the 4.6 tied head
    fwd_secs = time.perf_counter() - t0

    main_cmp = compare(logits[rows], ref)
    red_cmp = compare(tied[rows], ref)
    with torch.inference_mode():   # how far the untied head is from the embedding table
        head, emb = model.lm_head.weight, model.model.embed_tokens.weight
        d = head - emb
        row_cos = torch.nn.functional.cosine_similarity(head, emb, dim=-1)
        head_vs_embed = {"max_abs_diff": float(d.abs().max()), "mean_abs_diff": float(d.abs().mean()),
                         "rel_frobenius": float(d.norm() / emb.norm()), "row_cos_min": float(row_cos.min()),
                         "identical_rows": int((d.abs().amax(-1) == 0).sum()), "rows": int(head.shape[0])}
    arg_all = logits.argmax(-1).numpy()
    t5_all = torch.topk(logits, 5, dim=-1).indices.numpy()
    all_positions = {"positions": L,
                     "argmax_equal_oracle": int((arg_all == z["P0__argmax_all"]).sum()),
                     "top5_equal_oracle": int((t5_all == z["P0__top5_ids"]).all(-1).sum())}
    passed = (main_cmp["top5_equal"] == N_POS and main_cmp["argmax_equal"] == N_POS
              and main_cmp["cos_min"] >= BAR_COS)
    red = red_cmp["top5_equal"] < N_POS
    record = {
        "schema": "minicpmv47-parity-text/1",
        "purpose": "overlay Qwen3_5ForCausalLM (fp32, CPU, prefill, 1-D positions, untied head) vs the HF fp32 "
                   "oracle on P0 (prompt + 48 greedy tokens), last 48 positions",
        "result": "PASS" if passed and red else "FAIL",
        "bars": {"top5_equal": f"{N_POS}/{N_POS}", "argmax_equal": f"{N_POS}/{N_POS}", "cos_min": BAR_COS},
        "summary": {k: main_cmp[k] for k in main_cmp if k != "per_position"},
        "all_positions": all_positions,
        "red_arm_tied_head": {**{k: red_cmp[k] for k in red_cmp if k != "per_position"}, "red": bool(red)},
        "head_vs_embed_weights": head_vs_embed,
        "per_position": main_cmp["per_position"],
        "input": {"oracle_npz_sha256": sha256(oracle_dir / "oracle.npz"), "prompt_tokens": int(z["P0__prompt_len"]),
                  "sequence_tokens": L, "logits_positions": [L - N_POS, L - 1],
                  "oracle_versions": meta["versions"], "oracle_threads": meta["oracle"]["threads"]},
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(cfg).items()},
        "load": {k: v for k, v in report.items() if k != "weight_name_table"},
        "weight_name_table": report["weight_name_table"],
        "threads": torch.get_num_threads(), "dtype": "float32", "load_seconds": load_secs, "forward_seconds": fwd_secs,
        "versions": versions(torch),
        "script": {"path": "conversion/minicpmv47/parity_text.py", "sha256": sha256(Path(__file__))},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
    s = record["summary"]
    print(f"[parity_text] {record['result']}: top5 {s['top5_equal']}/{N_POS} argmax {s['argmax_equal']}/{N_POS} "
          f"cos min {s['cos_min']:.9f} mean {s['cos_mean']:.9f} max|d| {s['max_abs_diff']:.3e} "
          f"(ref absmax {s['ref_absmax']:.2f}); all {L} positions argmax {all_positions['argmax_equal_oracle']}/{L} "
          f"top5 {all_positions['top5_equal_oracle']}/{L}; forward {fwd_secs:.1f}s", flush=True)
    r = record["red_arm_tied_head"]
    print(f"[parity_text] red arm (tied head): top5 {r['top5_equal']}/{N_POS} argmax {r['argmax_equal']}/{N_POS} "
          f"cos min {r['cos_min']:.4f} -> {'RED' if r['red'] else 'NOT RED'}", flush=True)
    h = record["head_vs_embed_weights"]
    print(f"[parity_text] lm_head vs embed_tokens: max|d| {h['max_abs_diff']:.3e} rel fro {h['rel_frobenius']:.3e} "
          f"row cos min {h['row_cos_min']:.6f} identical rows {h['identical_rows']}/{h['rows']}", flush=True)
    print(f"[parity_text] -> {args.out}")


if __name__ == "__main__":
    main()
