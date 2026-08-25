"""Stage + upload the S1-mini Core AI bundle to HF. USER-GATED — run only when asked.

Uploads ONLY the int8lin bundle. int4lin is smaller and faster and passes the same 16/16
fp32 oracle gate, and it corrupts digits ($23,450 -> $2,345) — publishing it beside the
ship shape would put a broken normalizer one click away from someone who reads "smaller".

    coreai-models/.venv/bin/python conversion/_s1_mini_hf_upload.py
"""
import os
import shutil

os.environ["HF_HUB_DISABLE_XET"] = "1"
from pathlib import Path  # noqa: E402

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

REPO = "mlboydaisuke/S1-mini-CoreAI"
SRC = "superwhisper/s1-mini"
BUNDLE = Path.home() / "code/coreai/coreai-models/exports/s1_mini_decode_int8lin"
COMM = Path(__file__).resolve().parent.parent
STAGE = Path("/tmp/s1_mini_hf")

CARD = """---
license: other
license_name: s1-mini-license
license_link: LICENSE
library_name: coreai
pipeline_tag: text-generation
base_model: superwhisper/s1-mini
base_model_relation: quantized
language: [en]
tags: [core-ai, coreaikit, asr, speech-to-text, text-normalization,
       inverse-text-normalization, punctuation, truecasing, dictation,
       post-processing, on-device, apple, qwen3]
---

# S1-mini by Superwhisper — Core AI

[**S1-mini**](https://huggingface.co/superwhisper/s1-mini) by **Superwhisper** converted to
Apple **Core AI**, running fully on-device on iPhone and Mac.

S1-mini is a 0.6B **text normalizer for speech-to-text output**. Give it a raw ASR
transcript and it returns clean written text: fillers removed, false starts and
self-corrections resolved to whatever the speaker landed on, punctuation and capitalization
applied, and spoken numbers, dates, times, currency and email addresses rendered in written
form. It is not a chat model — it does one job, steered by a control line at the top of the
input.

That makes it the piece an on-device dictation stack is usually missing. Core AI already has
ASR (Parakeet, Nemotron-3.5-ASR-Streaming, Whisper, Qwen3-ASR); this is the post-processor
that turns a raw transcript into text a person would actually send, with nothing leaving the
device.

> **Naming.** The upstream license adds a term to Apache-2.0: any use, distribution or
> product integration must keep identifying this model as **"S1-mini"** by **"Superwhisper"**,
> with that exact capitalization, whatever the surrounding product is called. See
> [`LICENSE`](LICENSE).

## Contents

`gpu-pipelined/s1_mini_decode_int8lin/` — decode bundle for the Core AI pipelined engine,
**759 MB**. Body int8 per-block-32; the head is tied to the embedding and stays fp16 (see
*Quantization* below). Runs unchanged on macOS and iOS — no AOT compile needed.

## Measured

| | decode | prefill | numerics |
|---|---:|---:|---|
| **M4 Max** (Mac Studio, macOS 27.0) | **268.4 tok/s** | **4161 tok/s** | 16/16 token-exact vs the fp32 HF oracle |
| **iPhone 17 Pro** (A19 Pro, Release) | **62.4 tok/s** | **69.0 tok/s** | 276/276 + 27/27 token-exact vs the Mac engine |

`device == Mac == fp32 HF`. Load 0.2–1.0 s on device. The iPhone numbers are cold and
reproduced across two runs; **under sustained load expect about half** (34.9 prefill / 30.5
decode after back-to-back 1024-token generations, restored by seven idle minutes — thermal,
not a regression). A dictation post-processor runs repeatedly, so plan against the sustained
number.

### Task quality

The conversion gate above is a free-run continuation, which on a single-task model measures
the base language prior and very little of the task. So this port also gates the model in its
own input format, across the card's three control axes, against the released weights run
through `transformers`: **13/14**, the one miss being punctuation
(`$23,450 and` for `$23,450, and`).

## ⚠️ iPhone ceiling: prompt + generated must stay under 1024 tokens

Measured: a 611-token transcript whose rewrite runs 603 tokens produced **413 tokens on
device, every one token-identical to the Mac**, then stopped at absolute position exactly
**1024**. Truncation, not corruption.

This is shipped engine behaviour — `CoreAIPipelinedEngine` caps iOS growing-KV capacity at
1024 (`1024 - processed - prompt.count`) and throws `contextLengthExceeded` when a prompt
leaves no budget, guarding the iOS compiler's miscompilation of growing-KV specializations at
seq ≥ 2048. `1024 − 611 = 413`, exactly the measurement.

**Chunk input to roughly ≤450–500 tokens** so prompt + rewrite clears the cap. macOS has no
such cap.

## Prompt format — `enable_thinking=False` is mandatory

The system prompt and the control line are part of the trained input format. Leave thinking
on and the model emits an empty `<think>` block and stops: every call returns the empty
string, which reads like a working pipeline producing nothing.

```
<|im_start|>system
You are a text normalizer for speech-to-text transcripts. The input begins with a control line specifying the styling, structure, and context settings; clean the transcript to match those settings and output only the cleaned text.<|im_end|>
<|im_start|>user
[Styling: semi-formal] [Structure: prose] [Context: general]
<raw transcript><|im_end|>
<|im_start|>assistant
<think>

</think>

```

`Styling` ∈ `casual` / `semi-casual` / `semi-formal` / `formal` ·
`Structure` ∈ `prose` / `lists` · `Context` ∈ `general` / `email`.
All three axes are independent and every combination was trained.

Example, `[Styling: semi-formal] [Structure: prose] [Context: general]`:

| in | out |
|---|---|
| `so um i need to like send the the report by uh friday no wait make that thursday` | `So I need to send the report by Thursday.` |
| `the invoice came to twenty three thousand four hundred and fifty dollars and it's due on march third twenty twenty six` | `The invoice came to $23,450 and it's due on March 3, 2026.` |
| `um` | *(empty string)* |

## Quantization

- **Body int8**, per-block-32 `symmetric_with_clipping`; norms, RoPE, SDPA and the embedding
  stay full precision.
- **No head quantization, on purpose.** The head is *tied* to the 151936×1024 embedding, and
  the eager quantizer skips shared params — so quantizing it means untying it first, which
  *adds* a tensor rather than shrinking one: tied fp16 embed+head is 311 MB, while fp16 embed
  + int8 untied head is 311 + 156 = 467 MB. Untying is a pure loss at every bit width.
- **int4 is a measured no-go and is not published here.** It is 549 MB, decodes faster, and
  passes the *same* 16/16 fp32 oracle gate — and it corrupts digits: `$23,450` → `$2,345`,
  `107` → `177`, and it drops "tomorrow" from a time normalization. For a model whose job
  includes inverse text normalization of money and counts, that closes it. Only the
  task-format gate sees this; the continuation gate is blind to it.

## Reproduce

Exporter, gates, card and port notes live in the
[Core AI model zoo](https://github.com/john-rocky/coreai-model-zoo):
[`models/s1-mini/`](https://github.com/john-rocky/coreai-model-zoo/tree/main/models/s1-mini),
[`conversion/export_s1_mini_decode_pipelined.py`](https://github.com/john-rocky/coreai-model-zoo/blob/main/conversion/export_s1_mini_decode_pipelined.py),
[`knowledge/s1-mini-port.md`](https://github.com/john-rocky/coreai-model-zoo/blob/main/knowledge/s1-mini-port.md).

```bash
python3 conversion/zoo_convert.py show s1-mini
python3 conversion/zoo_convert.py run  s1-mini
```

## Credits

Model: **S1-mini** by **Superwhisper** ([superwhisper.com](https://superwhisper.com)).
Core AI conversion: the Core AI model zoo.
"""


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    dest = STAGE / "gpu-pipelined" / "s1_mini_decode_int8lin"
    dest.parent.mkdir(parents=True)
    shutil.copytree(BUNDLE, dest)
    (STAGE / "README.md").write_text(CARD)
    # Source LICENSE (Apache-2.0 + the naming term) and config.json travel with the bundle:
    # a reader must be able to see the terms and the source shape without leaving the repo.
    for name in ("LICENSE", "config.json"):
        shutil.copy(hf_hub_download(SRC, name), STAGE / name)

    total = sum(f.stat().st_size for f in STAGE.rglob("*") if f.is_file())
    print(f"staged {STAGE}  ({total / 1e6:.0f} MB)")
    for f in sorted(STAGE.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(STAGE)}  {f.stat().st_size / 1e6:.1f} MB")

    api = HfApi()
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=REPO, folder_path=str(STAGE), repo_type="model",
                      commit_message="S1-mini by Superwhisper — Core AI int8lin bundle")
    print(f"uploaded -> https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
