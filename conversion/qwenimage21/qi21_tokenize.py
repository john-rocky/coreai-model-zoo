"""Host tokenizer for the Qwen-Image-2.1 text-encoder graph — the ``tokenizers`` library only.

Reproduces what ``QwenImage21Pipeline._get_qwen_prompt_embeds`` does before the text encoder,
text-to-image, batch 1, without transformers:

  prompt ("" -> " ", Qwen has no BOS)  ->  t2i chat template  ->  ``processor/tokenizer.json``
  ->  ``input_ids [1, Lfull]``  ->  graph  ->  ``hidden[:, drop_idx:]`` = what the DiT reads.

``drop_idx`` is NOT a constant: it is the token count of the system part of the template
(``<|im_start|>system\\n<sys prompt><|im_end|>\\n``), the same thing the pipeline measures with
``apply_chat_template`` on the system message. ``gate_tokenize_hf.py`` checks both against the
real processor.

The graph's L axis is dynamic, so the host passes exactly ``Lfull`` tokens. ``encode_padded``
right-pads with ``<|endoftext|>`` (151643) for a fixed-L graph; attention is causal, so the
first ``Lfull`` outputs do not depend on the padding.

Which ``tokenizer.json``: the argument, else ``$QI21_TOKENIZER_JSON``, else the pinned base
snapshot's ``processor/tokenizer.json``. Qwen-Image-2.1-Turbo ships its own (same vocab / merges /
added tokens; only offset and decoder flags differ) — pass ``--snapshot <Turbo>`` or
``--tokenizer-json`` (``gate_tokenize_hf.py`` gates it against the Turbo processor).

  python qi21_tokenize.py "a red apple on a wooden table, studio lighting"
  python qi21_tokenize.py --snapshot <Turbo snapshot> "a red apple on a wooden table, studio lighting"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
TOKENIZER_JSON_ENV = "QI21_TOKENIZER_JSON"


def tokenizer_json_path(tokenizer_json: str | Path | None = None, snapshot: str | Path | None = None) -> Path:
    """``tokenizer_json``, else ``<snapshot>/processor/tokenizer.json``, else ``$QI21_TOKENIZER_JSON``, else
    the pinned base snapshot's ``processor/tokenizer.json``."""
    if tokenizer_json:
        return Path(tokenizer_json)
    if snapshot:
        return Path(snapshot) / "processor" / "tokenizer.json"
    if os.environ.get(TOKENIZER_JSON_ENV):
        return Path(os.environ[TOKENIZER_JSON_ENV])
    sys.path.insert(0, str(HERE.parent))
    from _paths import hf_snapshot
    return Path(hf_snapshot(MODEL, revision=REVISION)) / "processor" / "tokenizer.json"

SYS_PROMPT = "Comprehend and analyze the provided prompt."
SYS_PART = f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n"
TEMPLATE_T2I = SYS_PART + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
PAD_TOKEN = "<|endoftext|>"


class QI21Tokenizer:
    def __init__(self, tokenizer_json: str | Path | None = None):
        from tokenizers import Tokenizer
        self.path = tokenizer_json_path(tokenizer_json)
        self.tok = Tokenizer.from_file(str(self.path))
        self.tok.no_padding()
        self.tok.no_truncation()
        self.sys_ids = self._ids(SYS_PART)
        self.drop_idx = len(self.sys_ids)
        self.pad_id = self.tok.token_to_id(PAD_TOKEN)

    def _ids(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def encode(self, prompt: str) -> list[int]:
        """Token ids of the t2i template around ``prompt`` (length Lfull, no padding)."""
        ids = self._ids(TEMPLATE_T2I.format(prompt if prompt else " "))
        assert ids[:self.drop_idx] == self.sys_ids, "system part does not tokenize as a prefix"
        return ids

    def encode_np(self, prompt: str) -> np.ndarray:
        """``[1, Lfull]`` int32 — the dynamic-L graph's ``input_ids``."""
        return np.asarray([self.encode(prompt)], dtype=np.int32)

    def encode_padded(self, prompt: str, L: int) -> tuple[np.ndarray, int]:
        """``([1, L]`` int32 right-padded with 151643, Lfull) — for a fixed-L graph."""
        ids = self.encode(prompt)
        assert len(ids) <= L, f"prompt needs {len(ids)} tokens > graph L {L}"
        out = np.full((1, L), self.pad_id, dtype=np.int32)
        out[0, :len(ids)] = ids
        return out, len(ids)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts", nargs="*")
    ap.add_argument("--tokenizer-json", default=None)
    ap.add_argument("--snapshot", default=None, help="use <snapshot>/processor/tokenizer.json")
    a = ap.parse_args()
    t = QI21Tokenizer(tokenizer_json_path(a.tokenizer_json, a.snapshot))
    print(f"tokenizer.json: {t.path}")
    for p in a.prompts or ["a red apple on a wooden table, studio lighting"]:
        ids = t.encode(p)
        print(f"Lfull {len(ids)}  drop_idx {t.drop_idx}  L {len(ids) - t.drop_idx}  pad {t.pad_id}\n{ids}")
