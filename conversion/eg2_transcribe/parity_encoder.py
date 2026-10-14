#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.9.0",
#     "safetensors>=0.7.0",
#     "numpy>=2.2",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
# Community port — NOT an Apple or Google model.
"""Torch fp32 checks of the re-authored audio encoder, before any export.

    parity   the re-authored graph (eg2_audio_coreai.py, float32, CPU) against the fixture soft tokens
             (transformers 5.19.0 fp32 sdpa), per token, on the 20 test-clean clips, in five input layouts:
             the clip's own length; right-zero-padded to 3000 frames without and with the half mask; padded or
             cut to 1000 frames without and with the half mask. The no-mask rows test the claim that a static
             bucket needs no mask input. Writes <exports>/reauthor_parity.json.
    absmax   float16 headroom: the largest |x| per layer of the residual stream, the attention logits (before
             the softcap), the FFN intermediate, the conv / attention outputs, and every RMSNorm input and
             mean square, in float32, over the 20 clips (own length and padded to 3000) and the opening 10 minutes
             of the long clip as 20 windows of 3000 frames. Writes <exports>/absmax.json.

Run with the shared Core AI venv (torch 2.9, safetensors):
    <venv>/bin/python conversion/eg2_transcribe/parity_encoder.py parity
    <venv>/bin/python conversion/eg2_transcribe/parity_encoder.py absmax
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eg2_audio_coreai as E  # noqa: E402
from _common import (MODEL_ID, MODEL_REVISION, exports_dir, fixtures_dir, load_clips, model_dir,  # noqa: E402
                     token_stats, verify_checkpoint, window_input, write_json)

FP16_MAX = 65504.0


def run(enc, x: np.ndarray, half_mask: torch.Tensor | None = None) -> np.ndarray:
    with torch.no_grad():
        return enc(torch.from_numpy(x)[None], half_mask)[0].numpy()


def parity(args) -> None:
    mdir, fx = model_dir(args.model_dir), fixtures_dir(args.fixtures)
    digest = verify_checkpoint(mdir)
    enc = E.load_encoder(mdir, norm="exact")
    clips = load_clips(fx)
    rows, t0 = [], time.time()
    for c in clips:
        f, n, nt = c["features"], c["n_frames"], c["n_tokens"]
        row = {"id": c["id"], "n_frames": n, "n_tokens": nt, "n_frames_mod4": n % 4}
        out = run(enc, f)
        assert out.shape[0] == nt, (c["id"], out.shape)
        row["own_length"] = token_stats(out, c["ref"])
        for bucket in (3000, 1000):
            x, nv = window_input(f, bucket)
            k = min(nt, E.n_tokens(nv))
            out = run(enc, x)
            row[f"pad{bucket}_nomask"] = token_stats(out[:k], c["ref"][:k])
            row[f"pad{bucket}_nomask"]["last_token_cos"] = token_stats(out[k - 1:k], c["ref"][k - 1:k])["cos_min"]
            row[f"pad{bucket}_nomask"]["all_but_last"] = token_stats(out[:k - 1], c["ref"][:k - 1])
            out = run(enc, x, E.half_mask_for(nv, bucket))
            row[f"pad{bucket}_halfmask"] = token_stats(out[:k], c["ref"][:k])
            row[f"pad{bucket}_halfmask"]["last_token_cos"] = token_stats(out[k - 1:k], c["ref"][k - 1:k])["cos_min"]
            row[f"pad{bucket}_cut"] = nv < n
        rows.append(row)
        print(f"{c['id']}: n={n} (mod4 {n % 4}) own {row['own_length']['cos_min']:.9f} "
              f"pad3000 {row['pad3000_nomask']['cos_min']:.9f} / mask {row['pad3000_halfmask']['cos_min']:.9f} "
              f"pad1000 {row['pad1000_nomask']['cos_min']:.9f} / mask {row['pad1000_halfmask']['cos_min']:.9f}",
              flush=True)

    def agg(key: str, sub: str | None = None) -> dict:
        st = [(r["id"], r[key][sub] if sub else r[key]) for r in rows]
        worst = min(st, key=lambda t: t[1]["cos_min"])
        return {"clips": len(st), "tokens": int(sum(s["n"] for _, s in st)),
                "cos_mean_of_means": float(np.mean([s["cos_mean"] for _, s in st])),
                "cos_min": worst[1]["cos_min"], "cos_min_clip": worst[0],
                "max_abs": max(s["max_abs"] for _, s in st)}

    summary = {key: agg(key) for key in ("own_length", "pad3000_nomask", "pad3000_halfmask",
                                         "pad1000_nomask", "pad1000_halfmask")}
    summary["pad3000_nomask_all_but_last"] = agg("pad3000_nomask", "all_but_last")
    summary["pad1000_nomask_all_but_last"] = agg("pad1000_nomask", "all_but_last")
    bar = 0.999999
    mod2 = [r for r in rows if r["n_frames_mod4"] == 2]
    claim = {
        "statement": "right zero padding to a static bucket without any mask input gives the reference soft "
                     "tokens for every valid token",
        "holds_except": "the last token of clips with n_frames % 4 == 2 (the 3x3 stride-2 conv at half-frame "
                        "ceil(n/2) reads the last real frame; upstream zeroes that half-frame with its mask)",
        "clips_n_mod4_eq_2": [{"id": r["id"], "n_frames": r["n_frames"],
                               "last_token_cos_pad3000_nomask": r["pad3000_nomask"]["last_token_cos"],
                               "last_token_cos_pad3000_halfmask": r["pad3000_halfmask"]["last_token_cos"],
                               "all_but_last_cos_min_pad3000_nomask": r["pad3000_nomask"]["all_but_last"]["cos_min"],
                               "n_tokens": r["n_tokens"]} for r in mod2],
        "other_clips_last_token_cos_min_pad3000_nomask": min(
            [r["pad3000_nomask"]["last_token_cos"] for r in rows if r["n_frames_mod4"] != 2] or [None]),
    }
    result = {
        "what": "re-authored EmbeddingGemma 2 audio encoder (eg2_audio_coreai.py, float32, CPU eager) vs the "
                "fixture soft tokens (transformers 5.19.0, fp32, sdpa, CPU), per token",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "safetensors_sha256": digest, "dir": str(mdir)},
        "fixtures": str(fx), "bar_cos_min": bar,
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "threads": torch.get_num_threads()},
        "summary": summary,
        "pass_own_length": summary["own_length"]["cos_min"] >= bar,
        "pass_pad3000_halfmask": summary["pad3000_halfmask"]["cos_min"] >= bar,
        "pass_pad3000_nomask": summary["pad3000_nomask"]["cos_min"] >= bar,
        "pass_pad3000_nomask_all_but_last": summary["pad3000_nomask_all_but_last"]["cos_min"] >= bar,
        "mask_claim": claim, "seconds": round(time.time() - t0, 1), "clips": rows,
    }
    out = exports_dir(args.out_dir) / "reauthor_parity.json"
    write_json(out, result)
    for k, v in summary.items():
        print(f"{k:32s} cos_min {v['cos_min']:.9f}  mean {v['cos_mean_of_means']:.9f}  max|d| {v['max_abs']:.3e}")
    print(f"wrote {out}")


def absmax(args) -> None:
    mdir, fx = model_dir(args.model_dir), fixtures_dir(args.fixtures)
    verify_checkpoint(mdir)
    enc = E.load_encoder(mdir, norm="exact")
    clips = load_clips(fx)
    long_feats = np.load(fx / "long-speckled-band" / "features.npy", mmap_mode="r")
    sets = {}
    t0 = time.time()
    for name in ("clips_own_length", "clips_pad3000", "long_first_10min_w3000"):
        E._PROBE = {}
        if name == "clips_own_length":
            for c in clips:
                run(enc, c["features"])
        elif name == "clips_pad3000":
            for c in clips:
                run(enc, window_input(c["features"], 3000)[0])
        else:
            for w in range(20):
                run(enc, np.ascontiguousarray(long_feats[w * 3000:(w + 1) * 3000], dtype=np.float32))
        sets[name] = dict(E._PROBE)
        print(f"{name}: {len(E._PROBE)} probes, {time.time() - t0:.0f} s", flush=True)
    E._PROBE = None
    probes = sorted(set().union(*[s.keys() for s in sets.values()]))
    overall = {p: max(s.get(p, 0.0) for s in sets.values()) for p in probes}

    def kind(p: str) -> str:
        if p.endswith(".norm_meansq"):
            return "rmsnorm_mean_square"
        if p.endswith(".norm_in"):
            return "rmsnorm_input"
        if ".residual" in p:
            return "residual"
        if p.endswith(".logit"):
            return "attention_logit_pre_softcap"
        if p.endswith(".ffn_mid"):
            return "ffn_intermediate_post_silu"
        return p.split(".")[-1]

    per_layer: dict[str, dict] = {}
    for p, v in overall.items():
        layer = p.split(".")[0]
        k = kind(p)
        per_layer.setdefault(layer, {})
        per_layer[layer][k] = max(per_layer[layer].get(k, 0.0), v)
    # float16 headroom: a value must stay below 65504; an RMSNorm input is squared element-wise before the mean
    headroom = {}
    for layer, d in per_layer.items():
        h = {}
        for k, v in d.items():
            if v <= 0:
                continue
            h[k] = FP16_MAX / (v * v) if k == "rmsnorm_input" else FP16_MAX / v
        headroom[layer] = h
    flat = [(layer, k, v) for layer, h in headroom.items() for k, v in h.items()]
    tight = sorted(flat, key=lambda r: r[2])[:12]
    under2 = [r for r in flat if r[2] < 2.0]
    conclusion = (
        f"{len(under2)} quantities under 2x float16 headroom (worst: {tight[0][0]} {tight[0][1]} "
        f"{tight[0][2]:.3g}x)" + (" -> the all-fp16 graph needs the max-abs RMSNorm; fp16w32 is the main "
                                   "candidate" if under2 else " -> plain float16 arithmetic fits"))
    result = {"what": "float32 absmax per layer and float16 headroom (65504 / value; RMSNorm input: 65504 / value^2)",
              "inputs": {"clips_own_length": "20 test-clean fixture clips at their own length",
                         "clips_pad3000": "the same clips right-zero-padded to 3000 frames (padding tokens included)",
                         "long_first_10min_w3000": "long-speckled-band features rows 0..60000 as 20 windows of 3000"},
              "per_layer_absmax": per_layer, "per_layer_headroom": headroom,
              "tightest12": [{"layer": a, "quantity": b, "headroom": c} for a, b, c in tight],
              "under_2x": [{"layer": a, "quantity": b, "headroom": c} for a, b, c in under2],
              "per_input_set": sets, "conclusion": conclusion, "seconds": round(time.time() - t0, 1)}
    out = exports_dir(args.out_dir) / "absmax.json"
    write_json(out, result)
    for a, b, c in tight:
        print(f"{a:28s} {b:32s} headroom {c:.3g}x  (absmax {per_layer[a][b]:.4g})")
    print(conclusion)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("what", choices=("parity", "absmax"))
    ap.add_argument("--model-dir", help="google/embeddinggemma-2 snapshot (default: see _common.py)")
    ap.add_argument("--fixtures", help="reader fixtures directory (default: see _common.py)")
    ap.add_argument("--out-dir", help="where the json goes (default: <work>/exports)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    parity(args) if args.what == "parity" else absmax(args)


if __name__ == "__main__":
    main()
