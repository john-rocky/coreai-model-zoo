"""Gate: ``qi21_tokenize.QI21Tokenizer`` (tokenizers only) vs the pipeline's own processor call.

The reference side is the real ``QwenImage21Pipeline`` object (built around the snapshot's
``Qwen3VLProcessor``; no weights), so its ``prompt_template_t2i`` and ``_drop_idx`` are the
pipeline's, not a copy. Per prompt, the ids come from exactly the call
``_get_qwen_prompt_embeds`` makes: ``processor(text=[template.format(p or " ")], padding=True,
padding_side="left", return_tensors="pt").input_ids``.

Pass = every prompt's ids identical, the host drop_idx == the pipeline's (14), and the apple
prompt's ids == ``oracle/256/enc_input_ids.i32``.

``--snapshot`` points both sides at another checkpoint of the same pipeline (processor/ for the
reference, processor/tokenizer.json for the host unless ``--tokenizer-json`` overrides it);
``--oracle`` names the oracle whose ``enc_input_ids.i32`` the apple prompt must reproduce.

Run (qi21 venv — transformers 5.17 + diffusers main):
  ~/code/coreai/coreai-models/.venv-qi21/bin/python gate_tokenize_hf.py
Qwen-Image-2.1-Turbo (``.venv-qi21t``, diffusers 85c9fa4):
  ~/code/coreai/coreai-models/.venv-qi21t/bin/python gate_tokenize_hf.py --snapshot <Turbo snapshot> \\
      --oracle oracle/turbo256 --json <out>.json
"""
from __future__ import annotations

import argparse
import hashlib
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None, help="pipeline snapshot dir (default: the pinned base snapshot)")
    ap.add_argument("--tokenizer-json", default=None, help="host tokenizer.json (default: <snapshot>/processor/)")
    ap.add_argument("--oracle", default=str(HERE / "oracle" / "256"))
    ap.add_argument("--json", default=str(HERE / "_work" / "gate_tokenize_hf.json"))
    args = ap.parse_args()
    import diffusers
    import transformers
    from diffusers import QwenImage21Pipeline
    from transformers import Qwen3VLProcessor

    snap = Path(args.snapshot or hf_snapshot(MODEL, revision=REVISION))
    tj = Path(args.tokenizer_json or snap / "processor" / "tokenizer.json")
    proc = Qwen3VLProcessor.from_pretrained(str(snap / "processor"))
    pipe = QwenImage21Pipeline(scheduler=None, vae=None, text_encoder=None, processor=proc, transformer=None)
    host = QI21Tokenizer(tj)
    print(f"[tok-gate] snapshot {snap}\n[tok-gate] host tokenizer.json {tj} (sha256 "
          f"{hashlib.sha256(tj.read_bytes()).hexdigest()[:16]})", flush=True)
    print(f"[tok-gate] transformers {transformers.__version__}, diffusers {diffusers.__version__}; processor "
          f"{type(proc).__name__} / {type(proc.tokenizer).__name__}; pipeline drop_idx {pipe._drop_idx}, host drop_idx "
          f"{host.drop_idx}; template identical to the pipeline's: {host_template_matches(pipe)}", flush=True)

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

    od = Path(args.oracle)
    oracle_ids = np.fromfile(od / "enc_input_ids.i32", "<i4").tolist()
    meta = json.load(open(od / "meta.json"))
    same_oracle = (host.encode(PROMPTS["apple"]) == oracle_ids and meta["prompt"] == PROMPTS["apple"]
                   and int(meta["drop_idx"]) == host.drop_idx)
    ok = ok and same_oracle
    print(f"[tok-gate] apple host ids == {od.name}/enc_input_ids.i32 (and meta prompt / drop_idx): {same_oracle}",
          flush=True)
    print(f"[tok-gate] {'PASS' if ok else 'FAIL'}", flush=True)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(snapshot=str(snap), tokenizer_json=str(tj), transformers=transformers.__version__,
                   diffusers=diffusers.__version__, oracle=str(od),
                   drop_idx=dict(host=host.drop_idx, pipeline=pipe._drop_idx), rows=rows,
                   apple_equals_oracle=same_oracle, ok=ok),
              open(args.json, "w"), indent=2)
    print(f"[tok-gate] -> {args.json}", flush=True)
    return 0 if ok else 1


def host_template_matches(pipe) -> bool:
    from qi21_tokenize import TEMPLATE_T2I
    return pipe.prompt_template_t2i == TEMPLATE_T2I


if __name__ == "__main__":
    sys.exit(main())
