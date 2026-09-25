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

  python qi21_tokenize.py "a red apple on a wooden table, studio lighting"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"

SYS_PROMPT = "Comprehend and analyze the provided prompt."
SYS_PART = f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n"
TEMPLATE_T2I = SYS_PART + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
PAD_TOKEN = "<|endoftext|>"


class QI21Tokenizer:
    def __init__(self, tokenizer_json: str | Path | None = None):
        from tokenizers import Tokenizer
        if tokenizer_json is None:
            sys.path.insert(0, str(HERE.parent))
            from _paths import hf_snapshot
            tokenizer_json = Path(hf_snapshot(MODEL, revision=REVISION)) / "processor" / "tokenizer.json"
        self.tok = Tokenizer.from_file(str(tokenizer_json))
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
    t = QI21Tokenizer()
    for p in sys.argv[1:] or ["a red apple on a wooden table, studio lighting"]:
        ids = t.encode(p)
        print(f"Lfull {len(ids)}  drop_idx {t.drop_idx}  L {len(ids) - t.drop_idx}  pad {t.pad_id}\n{ids}")
