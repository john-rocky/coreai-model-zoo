"""fp32 reference hiddens for more prompts, from transformers (the oracle's code path, encoder only).

For each prompt of ``gate_tokenize_hf.PROMPTS``: the pipeline's processor call -> HF
``Qwen3VLForConditionalGeneration`` fp32 on CPU with the final norm neutralised by the same hook
-> ``hidden_states[-1]`` [1,Lfull,4096]. Saved to ``_work/ref_hf_fp32_text/<key>/{ids.i32,hidden.f32}``.
The apple prompt must reproduce ``oracle/256/enc_hidden_full.f32`` bit for bit.

Run (qi21 venv; ~36 GB RAM):
  ~/code/coreai/coreai-models/.venv-qi21/bin/python ref_text_hf.py
"""
from __future__ import annotations

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
from gate_tokenize_hf import PROMPTS  # noqa: E402
from qi21_tokenize import MODEL, REVISION  # noqa: E402


def main():
    from diffusers import QwenImage21Pipeline
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    snap = Path(hf_snapshot(MODEL, revision=REVISION))
    proc = Qwen3VLProcessor.from_pretrained(str(snap / "processor"))
    pipe = QwenImage21Pipeline(scheduler=None, vae=None, text_encoder=None, processor=proc, transformer=None)
    t0 = time.time()
    te = Qwen3VLForConditionalGeneration.from_pretrained(str(snap / "text_encoder"), dtype=torch.float32).eval()
    tm = te.model.language_model
    print(f"[ref-hf] fp32 loaded in {time.time() - t0:.0f}s, attn {te.config._attn_implementation}", flush=True)
    out = {}
    for key, p in PROMPTS.items():
        mi = proc(text=[pipe.prompt_template_t2i.format(" " if not p else p)], padding=True, padding_side="left",
                  return_tensors="pt")
        h = tm.norm.register_forward_hook(lambda module, a, o: a[0])
        try:
            with torch.no_grad():
                hid = te(input_ids=mi.input_ids, attention_mask=mi.attention_mask,
                         mm_token_type_ids=mi.mm_token_type_ids, output_hidden_states=True).hidden_states[-1]
        finally:
            h.remove()
        d = HERE / "_work" / "ref_hf_fp32_text" / key
        d.mkdir(parents=True, exist_ok=True)
        np.ascontiguousarray(mi.input_ids.numpy().astype("<i4")).tofile(d / "ids.i32")
        a = hid.float().numpy()
        np.ascontiguousarray(a, "<f4").tofile(d / "hidden.f32")
        out[key] = dict(Lfull=int(a.shape[1]), max_abs=float(np.abs(a).max()))
        print(f"[ref-hf] {key:<8} Lfull {a.shape[1]}  |h| max {np.abs(a).max():.1f}  -> {d}", flush=True)
    ora = np.fromfile(HERE / "oracle" / "256" / "enc_hidden_full.f32", "<f4")
    same = bool(np.array_equal(np.fromfile(HERE / "_work" / "ref_hf_fp32_text" / "apple" / "hidden.f32", "<f4"), ora))
    print(f"[ref-hf] apple == oracle/256/enc_hidden_full.f32 bit-exact: {same}", flush=True)
    json.dump(dict(prompts=out, apple_equals_oracle=same), open(HERE / "_work" / "ref_hf_fp32_text" / "meta.json", "w"),
              indent=2)
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
