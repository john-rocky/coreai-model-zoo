"""Greedy reference decode with PrismML's own MLX runtime, straight from the PQ2_0 GGUF.

Runs in a venv with the pack's pinned stack (mlx==0.32.0, mlx-lm==0.31.3) plus PrismML's gguf-py
(knows type 142). Uses the loader bundled in the MLX pack (`runtime.load`), which reads the
GGUF, transcodes PQ2_0 -> MLX affine 2-bit, applies the sign+Hadamard transform before each
folded matmul and the inverse after the embedding lookup — the token gate's yardstick.

    <mlx-venv>/bin/python _smoke/bonsai/mlx_reference.py --runtime <pack>/runtime \
        --gguf-py <llama.cpp-prism>/gguf-py --tokenizer <bundle>/tokenizer --out mlx_ref.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True, help="the MLX pack's runtime/ dir")
    ap.add_argument("--gguf-py", required=True)
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--prompt-file", default=None, help="read the prompt text from a file instead")
    ap.add_argument("--chat", action="store_true", help="apply the chat template")
    ap.add_argument("--new", type=int, default=16)
    ap.add_argument("--out", default="mlx_ref.json")
    args = ap.parse_args()
    if args.prompt_file:
        args.prompt = Path(args.prompt_file).read_text()

    sys.path.insert(0, str(Path(args.runtime).resolve()))
    import mlx.core as mx
    from runtime import load
    from transformers import AutoTokenizer

    gguf = args.gguf
    if gguf is None:
        from huggingface_hub import hf_hub_download
        gguf = hf_hub_download("prism-ml/Ternary-Bonsai-2-27B-gguf", "Ternary-Bonsai-2-27B-PQ2_0.gguf")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.chat:
        enc = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                      add_generation_prompt=True, tokenize=True)
    else:
        enc = tok(args.prompt)
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc     # transformers 5 returns a mapping
    ids = [int(i) for i in (ids[0] if isinstance(ids[0], (list, tuple)) else ids)]
    print(f"[mlx] prompt ids ({len(ids)}): {ids}", flush=True)

    t0 = time.perf_counter()
    model, info, _ = load(gguf, args.gguf_py)
    print(f"[mlx] loaded in {time.perf_counter() - t0:.0f}s: {info}", flush=True)

    cache = model.make_cache()
    seq = list(ids)
    steps = []
    out = []
    t0 = time.perf_counter()
    # token-by-token, mirroring the S=1 walk the Core AI bundle does
    for step in range(len(ids) + args.new - 1):
        x = mx.array([[seq[step]]], dtype=mx.int32)
        y = model(x, cache=cache)                       # mlx_lm TextModel returns logits
        if y.shape[-1] == info["config"]["hidden_size"]:  # (hidden only if a bare core was passed)
            y = model.lm_head(y)
        logits = np.asarray(y[0, -1].astype(mx.float32))
        top = np.argsort(logits)[::-1][:5]
        steps.append({"step": step, "token": int(seq[step]), "argmax": int(top[0]),
                      "top5": [int(i) for i in top], "top5_logits": [float(logits[i]) for i in top],
                      "margin": float(logits[top[0]] - logits[top[1]])})
        if step >= len(ids) - 1:
            nxt = int(top[0])
            out.append(nxt)
            seq.append(nxt)
            print(f"  -> {nxt} {tok.decode([nxt])!r} (margin {steps[-1]['margin']:.3f})", flush=True)
    dt = time.perf_counter() - t0
    print(f"[mlx] {len(steps)} steps in {dt:.1f}s ({len(steps) / dt:.1f} tok/s); GENERATION: {tok.decode(out)!r}", flush=True)
    Path(args.out).write_text(json.dumps({"prompt": args.prompt, "chat": args.chat, "ids": ids,
                                          "generated": out, "steps": steps}, indent=1))
    print(f"[mlx] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
