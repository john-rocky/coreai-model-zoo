#!/usr/bin/env python3
"""test_lfm3_ane_tiny.py OUT_DIR — the LFM3 static iOS builder (CM models/ios/lfm3.py) against the macOS authoring
module (models/macos/lfm3.py, line 1 PASS vs HF fp32) on a random 5-layer LFM3-shaped checkpoint (conv, conv, full,
conv, sliding; export_lfm3_ane_static.write_tiny_checkpoint), both fp32 on the CPU. Tens of seconds; exits 1 on a miss.

The builder is walked the way Apple's StaticShapeEngine (CoreAILanguageModels, main 7359dbc) calls a static bundle:
q in {8, 16, 64} with the q-aligned re-run of the batch, context buckets 256 / 512, position_ids / in_step /
causal_mask as the runner fills them (sentinel -40000, padding columns all-sentinel), padding rows = token 0. A
300-token sequence = 280 prompt tokens (prefill chunks of 64 + the decode graph's tail) + 20 one-token steps: the
conv history is read back across every chunk boundary and every re-run, the window (256) starts holding keys back
at position 256, the bucket switches 256 -> 512. Each position's logits come from the call that first processed it.

  block    history read as one 8-position block per row (the export's default)
  column   the LFM2.5 builder's read: one one-hot per history column
  stock    act="stock" (sigmoid / SiLU instead of the tanh forms)
Negative controls (must be RED): history="none" (the conv history written to the KV rows, never read back); no
window (the sliding layers get the plain causal mask; red only past position 256).
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "conversion"))
from export_lfm3_ane_static import write_tiny_checkpoint  # noqa: E402

from coreai_models.models.ios import lfm3 as LI  # noqa: E402
from coreai_models.models.macos import lfm3 as LM  # noqa: E402

TOL = 1e-4
QS, CTXS, SENTINEL = (8, 16, 64), (256, 512), -40000.0


def inplace_slice_update(x, update, begin, end):
    """The eager body of coreai::mutable_slice_update without its whole-cache clone."""
    sl = tuple(slice(int(b), int(e)) for b, e in zip(begin.tolist(), end.tolist(), strict=False))
    x[sl] = update
    return x


def engine_logits(model, tokens, n_prompt):
    """StaticShapeEngine.inference over the prompt, then one token per step; logits of every position."""
    kc = torch.zeros(model.config.num_hidden_layers, 1, model.cache_width, 1, max(CTXS))
    vc = torch.zeros_like(kc)
    rows, processed = {}, 0
    for end_tok in [n_prompt] + list(range(n_prompt + 1, len(tokens) + 1)):
        cur = processed
        while cur < end_tok:
            remaining = end_tok - cur
            q = max(QS) if remaining > max(QS) else min(x for x in QS if x >= remaining)
            ctx = min(c for c in CTXS if c > cur)
            start = (cur // q) * q
            n_in = min(start + q - 1, end_tok - 1) - start + 1
            ids = torch.zeros(1, q, dtype=torch.int32)
            ids[0, :n_in] = torch.tensor(tokens[start:start + n_in], dtype=torch.int32)
            mask = torch.full((1, ctx, 1, q), SENTINEL)
            for i in range(n_in):
                mask[0, :min(start + i, ctx - 1) + 1, 0, i] = 0
            pos = (start + torch.arange(q, dtype=torch.int32))[None]
            out = model(ids, pos, torch.tensor([start], dtype=torch.int32), mask, kc[..., :ctx], vc[..., :ctx])
            for p in range(cur, start + n_in):
                rows[p] = out[0, 0, p - start].numpy()
            cur = start + n_in
        processed = cur
    return np.stack([rows[p] for p in range(len(tokens))])


def main(out: str) -> int:
    torch.set_grad_enabled(False)
    LI.mutable_slice_update = inplace_slice_update
    ckpt = write_tiny_checkpoint(Path(out) / "tiny_ckpt", seed=0)
    rng = np.random.default_rng(0)
    tokens = [int(t) for t in rng.integers(3, 1024, 300)]

    ref_model = LM.lfm3_from_hf(str(ckpt), target_dtype=torch.float32, stateful=False)
    ids = torch.tensor(tokens, dtype=torch.int32)[None]
    want = ref_model(ids, torch.arange(len(tokens), dtype=torch.int32)[None])[0].numpy()
    print(f"tiny: {ref_model.config.layer_types}, window {ref_model.config.sliding_window}, {len(tokens)} tokens")

    fails = []
    window = LI.sliding_window_mask
    for name, kw, red in (("block", {"history": "block"}, False), ("column", {"history": "column"}, False),
                          ("stock act", {"act": "stock"}, False), ("history none", {"history": "none"}, True),
                          ("no window", {}, True)):
        LI.sliding_window_mask = (lambda mask, w, s: mask) if name == "no window" else window
        m = LI.lfm3_ios_from_hf(str(ckpt), dtype=torch.float32, disable_embedding_quantization=True,
                                max_context_length=max(CTXS), **kw)
        got = engine_logits(m, tokens, 280)
        LI.sliding_window_mask = window
        d = np.abs(got - want).max(-1)
        top1 = bool((got.argmax(-1) == want.argmax(-1)).all())
        ok = float(d.max()) <= TOL and top1
        if red:
            status = "RED (expected)" if not ok else "GREEN (control did not bite!)"
            if ok:
                fails.append(name)
        else:
            status = "PASS" if ok else "FAIL"
            if not ok:
                fails.append(name)
        print(f"  {name:14s} max|d| {d.max():.3e} (positions >= 256: {d[256:].max():.3e})  "
              f"top1 {'all' if top1 else 'MISS'}  {status}")
    print("RESULT", "FAIL " + ",".join(fails) if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.getcwd(), "_lfm3_ane_tiny")))
