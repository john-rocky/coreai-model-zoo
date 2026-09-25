"""Check (transformers side): for text-only input the interleaved 3-axis M-RoPE IS plain 1D RoPE.

Runs the HF ``Qwen3VLForConditionalGeneration`` (bf16, CPU, the pipeline's call: ``attention_mask``
+ ``mm_token_type_ids``, final norm neutralised by the same forward hook) on the oracle ids:

  A  no ``position_ids`` (what the pipeline does: the model builds ``arange`` itself)
  B  explicit 3-axis ``position_ids`` = ``arange(L)`` on T, H and W          -> must equal A exactly
  C  control: H axis shifted by +7 (T and W stay ``arange``)                 -> must differ from A
  R  the rotary module's fp32 cos/sin for the 3-axis ``arange`` vs ``qi21_text.rope_tables``
     (the 1D tables the graph uses)                                          -> must be bit-identical

Also records the HF bf16 hidden vs the fp32 oracle (context for the engine's bf16 numbers) and
saves it to ``_work/hf_bf16_text/<tag>/hidden.f32``.

Run (qi21 venv — transformers 5.17; ~18 GB RAM):
  ~/code/coreai/coreai-models/.venv-qi21/bin/python check_mrope_hf.py --oracle oracle/256
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
sys.path.insert(0, str(HERE.parent))
from _paths import hf_snapshot  # noqa: E402
from qi21_text import rope_inv_freq, rope_tables  # noqa: E402

MODEL = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"


def stats(a, b) -> dict:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return dict(maxd=float(np.abs(a - b).max()), corr=float(np.corrcoef(a, b)[0, 1]),
                rel_l2=float(np.linalg.norm(a - b) / np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default=str(HERE / "oracle" / "256"))
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = ap.parse_args()
    import transformers
    from transformers import Qwen3VLForConditionalGeneration

    od = Path(args.oracle)
    meta = json.load(open(od / "meta.json"))
    Lfull = int(meta["Lfull"])
    ids = torch.from_numpy(np.fromfile(od / "enc_input_ids.i32", "<i4").reshape(1, Lfull).astype(np.int64))
    ref = np.fromfile(od / "enc_hidden_full.f32", "<f4").reshape(1, Lfull, -1)
    tdir = Path(hf_snapshot(MODEL, revision=REVISION)) / "text_encoder"
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    t0 = time.time()
    te = Qwen3VLForConditionalGeneration.from_pretrained(str(tdir), dtype=dt).eval()
    tm = te.model.language_model
    print(f"[mrope] transformers {transformers.__version__} torch {torch.__version__}: loaded {args.dtype} in "
          f"{time.time() - t0:.0f}s, attn {te.config._attn_implementation}, rope {tm.rotary_emb.rope_type} "
          f"mrope_section {tm.rotary_emb.mrope_section}", flush=True)

    kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=torch.zeros_like(ids),
              output_hidden_states=True)
    ar = torch.arange(Lfull)
    pos_b = ar.view(1, 1, -1).expand(3, 1, -1).contiguous()
    pos_c = torch.stack([ar, ar + 7, ar]).view(3, 1, -1)

    def run(**extra):
        h = tm.norm.register_forward_hook(lambda module, a, o: a[0])
        try:
            with torch.no_grad():
                return te(**kw, **extra).hidden_states[-1].float().numpy()
        finally:
            h.remove()

    hA = run()
    hB = run(position_ids=pos_b)
    hC = run(position_ids=pos_c)
    sAB, sAC = stats(hB, hA), stats(hC, hA)
    exact_ab = bool(np.array_equal(hA, hB))
    print(f"[mrope] B (explicit 3-axis arange) vs A (no position_ids): max|d| {sAB['maxd']:.3e}  "
          f"bit-exact {exact_ab}  {'PASS' if exact_ab else 'FAIL'}", flush=True)
    print(f"[mrope] C control (H axis +7) vs A: max|d| {sAC['maxd']:.3e}  corr {sAC['corr']:.6f}  "
          f"|d|/|ref| {sAC['rel_l2']:.3e}  {'RED' if sAC['maxd'] > 0 else 'NOT RED'}", flush=True)

    with torch.no_grad():
        cos3, sin3 = tm.rotary_emb(torch.zeros(1, Lfull, 1, dtype=torch.float32), pos_b)
    theta = float(tm.config.rope_parameters["rope_theta"])
    cos1, sin1 = rope_tables(Lfull, tm.config.head_dim, theta)
    exact_r = bool(torch.equal(cos3[0], cos1) and torch.equal(sin3[0], sin1))
    print(f"[mrope] rotary fp32 cos/sin (3-axis arange) vs qi21_text 1D tables: max|d| cos "
          f"{float((cos3[0] - cos1).abs().max()):.3e} sin {float((sin3[0] - sin1).abs().max()):.3e}  "
          f"bit-exact {exact_r}  {'PASS' if exact_r else 'FAIL'}", flush=True)
    print(f"[mrope] inv_freq HF vs qi21_text bit-exact: "
          f"{bool(torch.equal(tm.rotary_emb.inv_freq.float(), rope_inv_freq(tm.config.head_dim, theta)))}  "
          f"(theta {theta:g}, config value {tm.config.rope_parameters['rope_theta']!r})", flush=True)

    sO = stats(hA, ref)
    print(f"[mrope] context: HF {args.dtype} (CPU) vs fp32 oracle: corr {sO['corr']:.6f}  max|d| {sO['maxd']:.3e}  "
          f"|d|/|ref| {sO['rel_l2']:.3e}", flush=True)
    sd = HERE / "_work" / f"hf_{args.dtype}_text" / od.name
    sd.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(hA, "<f4").tofile(sd / "hidden.f32")
    ok = exact_ab and sAC["maxd"] > 0 and exact_r
    json.dump(dict(oracle=str(od), dtype=args.dtype, B_vs_A=sAB, B_equals_A=exact_ab, C_vs_A=sAC,
                   rotary_tables_bit_exact=exact_r, hf_vs_oracle=sO, ok=ok),
              open(HERE / "_work" / f"check_mrope_hf_{od.name}_{args.dtype}.json", "w"), indent=2)
    print(f"[mrope] {'PASS' if ok else 'FAIL'}  (saved {sd / 'hidden.f32'})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
