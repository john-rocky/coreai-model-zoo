"""Gate: the re-authored text encoder (``qi21_text.py``) with the REAL weights, fp32 on CPU, vs the oracle.

``oracle/<tag>/enc_input_ids.i32`` [1,Lfull] -> ``load_qi21_text(..., dtype=float32)`` -> hidden
[1,Lfull,4096], compared over ALL tokens with ``enc_hidden_full.f32`` (the last decoder layer's
output before the final RMSNorm). Bar: corr >= 0.999999 and max|d| / max|ref| <= 1e-4.
Also: ``prompt_embeds.f32`` must equal ``enc_hidden_full[:, drop_idx:]`` bit for bit, and the
re-author's ``[:, drop_idx:]`` slice is scored against ``prompt_embeds`` (what the DiT reads).

Negative controls, each must be RED at the bar (same weights, one deliberate bug):
  (a) q_norm / k_norm removed   (b) RoPE theta 1e4 instead of 5e6   (c) final RMSNorm applied.

``--save`` writes the fp32 output to ``_work/torch_fp32_text/<tag>/hidden.f32`` (the engine gate
scores the bundle against it too).

Run (coreai-models base venv — no transformers import; ~31 GB RAM):
  python parity_text_oracle.py --oracle oracle/256 --save
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import hf_snapshot  # noqa: E402
from qi21_text import RMSNorm, load_final_norm_weight, load_qi21_text, text_config  # noqa: E402

MODEL = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
BAR_CORR, BAR_REL_MAX = 0.999999, 1e-4


def score(out, ref) -> dict:
    a = np.asarray(out, np.float64).reshape(-1, np.shape(ref)[-1])
    b = np.asarray(ref, np.float64).reshape(-1, np.shape(ref)[-1])
    d = a - b
    tok_corr = [float(np.corrcoef(a[i], b[i])[0, 1]) for i in range(a.shape[0])]
    tok_rel = [float(np.linalg.norm(d[i]) / np.linalg.norm(b[i])) for i in range(a.shape[0])]
    maxd, max_ref = float(np.abs(d).max()), float(np.abs(b).max())
    return dict(corr=float(np.corrcoef(a.ravel(), b.ravel())[0, 1]), maxd=maxd, max_ref=max_ref,
                rel_max=maxd / max_ref, rel_l2=float(np.linalg.norm(d) / np.linalg.norm(b)),
                min_tok_corr=min(tok_corr), min_tok_corr_at=int(np.argmin(tok_corr)),
                max_tok_rel=max(tok_rel), max_tok_rel_at=int(np.argmax(tok_rel)),
                nan=int(np.isnan(a).sum()))


def passes(s: dict) -> bool:
    return s["nan"] == 0 and s["corr"] >= BAR_CORR and s["rel_max"] <= BAR_REL_MAX


def line(name: str, s: dict) -> str:
    return (f"{name:<28} corr {s['corr']:.9f}  max|d| {s['maxd']:.3e}  max|d|/max|ref| {s['rel_max']:.2e}  "
            f"|d|/|ref| {s['rel_l2']:.2e}  min tok corr {s['min_tok_corr']:.9f} (tok {s['min_tok_corr_at']})  "
            f"max tok |d|/|ref| {s['max_tok_rel']:.2e} (tok {s['max_tok_rel_at']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default=str(HERE / "oracle" / "256"))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--text-encoder-dir", default=None, help="default: the pinned HF snapshot's text_encoder/")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    od = Path(args.oracle)
    meta = json.load(open(od / "meta.json"))
    Lfull, drop, L = int(meta["Lfull"]), int(meta["drop_idx"]), int(meta["L"])
    ids = torch.from_numpy(np.fromfile(od / "enc_input_ids.i32", "<i4").reshape(1, Lfull).copy())
    ref = np.fromfile(od / "enc_hidden_full.f32", "<f4").reshape(1, Lfull, -1)
    pe = np.fromfile(od / "prompt_embeds.f32", "<f4").reshape(1, L, -1)
    assert L == Lfull - drop and pe.shape[-1] == ref.shape[-1]
    pe_is_slice = bool(np.array_equal(pe, ref[:, drop:]))
    print(f"[parity-text] oracle {od.name}: Lfull {Lfull} drop_idx {drop} L {L} |ref| max {np.abs(ref).max():.1f}; "
          f"prompt_embeds == enc_hidden_full[:, {drop}:] bit-exact: {pe_is_slice}", flush=True)
    print(f"[parity-text] ids {ids[0].tolist()}", flush=True)

    tdir = Path(args.text_encoder_dir or Path(hf_snapshot(MODEL, revision=REVISION)) / "text_encoder")
    t0 = time.time()
    model = load_qi21_text(tdir, dtype=torch.float32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[parity-text] loaded {tdir} fp32 in {time.time() - t0:.0f}s ({n_params / 1e9:.3f} B params, "
          f"{len(model.layers)} layers, theta {model.rope_theta:g}), torch threads {torch.get_num_threads()}",
          flush=True)

    rows = {}
    with torch.no_grad():
        t1 = time.time()
        out = model(ids).numpy()
        sec = time.time() - t1
        rows["reauthor"] = score(out, ref)
        rows["reauthor"]["sec"] = sec
        rows["reauthor_slice_vs_prompt_embeds"] = score(out[:, drop:], pe)
        ok = passes(rows["reauthor"]) and passes(rows["reauthor_slice_vs_prompt_embeds"]) and pe_is_slice
        print(line("re-author vs enc_hidden_full", rows["reauthor"]) + f"  ({sec:.1f}s)  "
              f"{'PASS' if passes(rows['reauthor']) else 'FAIL'}", flush=True)
        print(line(f"[:, {drop}:] vs prompt_embeds", rows["reauthor_slice_vs_prompt_embeds"]) + "  "
              f"{'PASS' if passes(rows['reauthor_slice_vs_prompt_embeds']) else 'FAIL'}", flush=True)
        if args.save:
            sd = HERE / "_work" / "torch_fp32_text" / od.name
            sd.mkdir(parents=True, exist_ok=True)
            np.ascontiguousarray(out, "<f4").tofile(sd / "hidden.f32")
            print(f"[parity-text] saved {sd / 'hidden.f32'}", flush=True)

        controls = {}
        if not args.no_controls:
            # (c) the final RMSNorm the pipeline neutralises — applied on top of the correct output
            norm = RMSNorm(model.hidden_size, text_config(tdir)["rms_norm_eps"])
            norm.weight.data = load_final_norm_weight(tdir).float()
            controls["c_final_norm_applied"] = score(norm(torch.from_numpy(out)).numpy(), ref)
            # (b) RoPE theta 1e4 (the Qwen2 default) instead of 5e6
            theta = model.rope_theta
            model.set_rope(1e4, model.max_len)
            controls["b_rope_theta_1e4"] = score(model(ids).numpy(), ref)
            model.set_rope(theta, model.max_len)
            # (a) q_norm / k_norm removed
            saved = [(l.self_attn.q_norm, l.self_attn.k_norm) for l in model.layers]
            for l in model.layers:
                l.self_attn.q_norm, l.self_attn.k_norm = nn.Identity(), nn.Identity()
            controls["a_no_qk_norm"] = score(model(ids).numpy(), ref)
            for l, (qn, kn) in zip(model.layers, saved):
                l.self_attn.q_norm, l.self_attn.k_norm = qn, kn
            # restored model must reproduce the first run exactly
            again = model(ids).numpy()
            restored_exact = bool(np.array_equal(again, out))
            for key in ("a_no_qk_norm", "b_rope_theta_1e4", "c_final_norm_applied"):
                s = controls[key]
                s["red"] = not passes(s)
                s["red_at_corr_0.999"] = s["corr"] < 0.999
                print(line(f"control {key}", s) + f"  {'RED' if s['red'] else 'NOT RED'} "
                      f"({'red' if s['red_at_corr_0.999'] else 'NOT red'} at corr 0.999)", flush=True)
            print(f"[parity-text] model restored after controls reproduces the run bit-exactly: {restored_exact}",
                  flush=True)
            ok = ok and all(c["red"] for c in controls.values()) and restored_exact

    print(f"\n[parity-text] {'PASS' if ok else 'FAIL'} (bar: corr >= {BAR_CORR}, max|d|/max|ref| <= {BAR_REL_MAX}, "
          f"all {len(controls)} controls red)", flush=True)
    js = Path(args.json or HERE / "_work" / f"parity_text_oracle_{od.name}.json")
    js.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(oracle=str(od), Lfull=Lfull, drop_idx=drop, prompt_embeds_is_slice=pe_is_slice,
                   bar=dict(corr=BAR_CORR, rel_max=BAR_REL_MAX), rows=rows, controls=controls, ok=ok),
              open(js, "w"), indent=2)
    print(f"[parity-text] -> {js}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
