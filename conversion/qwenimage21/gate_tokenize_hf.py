"""Gate: ``qi21_tokenize.QI21Tokenizer`` (tokenizers only) vs the pipeline's own processor call.

The reference side is the real ``QwenImage21Pipeline`` object (built around the snapshot's
``Qwen3VLProcessor``; no weights), so its ``prompt_template_t2i`` and ``_drop_idx`` are the
pipeline's, not a copy. Per prompt, the ids come from exactly the call
``_get_qwen_prompt_embeds`` makes: ``processor(text=[template.format(p or " ")], padding=True,
padding_side="left", return_tensors="pt").input_ids``.

Pass = every prompt's ids identical, the host drop_idx == the pipeline's (14), and the apple
prompt's ids == ``oracle/256/enc_input_ids.i32``.

Run (qi21 venv — transformers 5.17 + diffusers main):
  ~/code/coreai/coreai-models/.venv-qi21/bin/python gate_tokenize_hf.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import hf_snapshot  # noqa: E402
from qi21_tokenize import MODEL, REVISION, QI21Tokenizer  # noqa: E402

PROMPTS = {
    "apple": "a red apple on a wooden table, studio lighting",
    "empty": "",
    "sticker": ("This is an RGBA image with transparency. A cute cartoon dragon sticker. "
                "The image has alpha channel and the background is transparent."),
}


def main():
    import transformers
    from diffusers import QwenImage21Pipeline
    from transformers import Qwen3VLProcessor

    snap = Path(hf_snapshot(MODEL, revision=REVISION))
    proc = Qwen3VLProcessor.from_pretrained(str(snap / "processor"))
    pipe = QwenImage21Pipeline(scheduler=None, vae=None, text_encoder=None, processor=proc, transformer=None)
    host = QI21Tokenizer(snap / "processor" / "tokenizer.json")
    print(f"[tok-gate] transformers {transformers.__version__}; processor {type(proc).__name__} / "
          f"{type(proc.tokenizer).__name__}; pipeline drop_idx {pipe._drop_idx}, host drop_idx {host.drop_idx}; "
          f"template identical to the pipeline's: {host_template_matches(pipe)}", flush=True)

    ok = host.drop_idx == pipe._drop_idx == 14 and host_template_matches(pipe)
    rows = {}
    for key, p in PROMPTS.items():
        text = pipe.prompt_template_t2i.format(" " if not p else p)
        mi = proc(text=[text], padding=True, padding_side="left", return_tensors="pt")
        ref = mi.input_ids[0].tolist()
        got = host.encode(p)
        same = got == ref and bool(mi.attention_mask.all())
        rows[key] = dict(prompt=p, Lfull=len(ref), L=len(ref) - pipe._drop_idx, identical=same,
                         first_diff=next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), None),
                         host_len=len(got))
        ok = ok and same
        print(f"[tok-gate] {key:<8} Lfull {len(ref):3d} (L {len(ref) - pipe._drop_idx:3d})  host == processor: {same}"
              + ("" if same else f"  host {got}\n  ref  {ref}"), flush=True)

    oracle_ids = np.fromfile(HERE / "oracle" / "256" / "enc_input_ids.i32", "<i4").tolist()
    same_oracle = host.encode(PROMPTS["apple"]) == oracle_ids
    ok = ok and same_oracle
    print(f"[tok-gate] apple host ids == oracle/256/enc_input_ids.i32: {same_oracle}", flush=True)
    print(f"[tok-gate] {'PASS' if ok else 'FAIL'}", flush=True)
    json.dump(dict(drop_idx=dict(host=host.drop_idx, pipeline=pipe._drop_idx), rows=rows,
                   apple_equals_oracle=same_oracle, ok=ok),
              open(HERE / "_work" / "gate_tokenize_hf.json", "w"), indent=2)
    return 0 if ok else 1


def host_template_matches(pipe) -> bool:
    from qi21_tokenize import TEMPLATE_T2I
    return pipe.prompt_template_t2i == TEMPLATE_T2I


if __name__ == "__main__":
    sys.exit(main())
