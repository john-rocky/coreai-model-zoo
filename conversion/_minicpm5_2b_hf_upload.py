"""Stage + upload the MiniCPM5-2B Core AI int8 bundle to HF. USER-GATED — run only when asked.

Mirrors the published 1B repo layout (mlboydaisuke/MiniCPM5-1B-CoreAI): `int8/` holds the
FM-format dynamic bundle (metadata.json + <name>.aimodel + tokenizer/), the source LICENSE
(Apache-2.0) and a `config.json` descriptor sit at the root, README.md is the card.

    coreai-models/.venv/bin/python conversion/_minicpm5_2b_hf_upload.py [--stage-only]
"""
import json
import os
import re
import shutil
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # classic LFS by default; HF_HUB_DISABLE_XET=0 opts into chunked XET
from pathlib import Path  # noqa: E402

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

REPO = "mlboydaisuke/MiniCPM5-2B-CoreAI"
SRC = "openbmb/MiniCPM5-2B"
EXPORT = Path.home() / "code/coreai/coreai-models/exports/minicpm5-2b-b32"
BUNDLE = EXPORT / "minicpm5_2b_minicpm5_int8sym_b32_dynamic"   # what coreai.llm.export wrote
PUBLISHED_NAME = "minicpm5_2b_int8"                            # what the repo calls it (as the 1B)
COMM = Path(__file__).resolve().parent.parent
STAGE = Path("/tmp/minicpm5_2b_hf")
# The source repo ships no LICENSE file (cardData says apache-2.0); the published 1B repo carries the
# Apache-2.0 text, so the 2B repo copies that same file — one license text for the family.
LICENSE_FROM = ("mlboydaisuke/MiniCPM5-1B-CoreAI", "LICENSE")

CARD = """---
license: apache-2.0
base_model: openbmb/MiniCPM5-2B
pipeline_tag: text-generation
library_name: coreai
tags:
- coreai
- core-ai
- coreml
- apple
- on-device
- iphone
- metal
base_model_relation: quantized
---

Core AI is Apple's on-device ML runtime in iOS 27 / macOS 27 and the successor to Core ML: PyTorch models are exported with Apple's `coreai-torch` (LLMs: `coreai.llm.export`) into `.aimodel` bundles that run on the GPU or the Neural Engine, e.g. Qwen3-8B 4-bit decodes at 94 tok/s on an M4 Max GPU, MLX 90 under the same protocol ([apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench), macOS 27 beta, 2026-06).

<!-- gen-cards:devicemark begin (managed by scripts/gen-cards + tools/devicemark_row.py — edit cards.json, not this block) -->
This model has no row on [DeviceMark](https://devicemark.github.io/), the on-device LLM leaderboard.
<!-- gen-cards:devicemark end -->

# MiniCPM5-2B — Core AI (int8 block-32, runs on iPhone)

Apple **Core AI** (`.aimodel`) conversion of [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) —
OpenBMB's 2.5B on-device LLM (released 2026-09-06, the 1B's 42-layer sibling) with **hybrid
Think / No-Think reasoning**, native tool calling and **128K** context; OpenBMB reports it as
2B-class open-source SOTA (LiveCodeBench v6 69.1, AIME 2026 86.5, BFCL v4 66.6, SWE-bench Verified
46.4 on their card). Runs fully on-device on **iPhone** and Apple Silicon Macs (GPU, pipelined engine).

Part of the community Core AI model zoo: **https://github.com/john-rocky/coreai-model-zoo**

<!-- gen-cards:use-it begin id=minicpm5-2b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — run the kit's task op on this model
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let tldr = try await CoreAI.summarize(text, options: .model("minicpm5-2b"))
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [ChatDemo runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo)
(GUI + CLI, one app for every chat model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/ChatDemo/ChatDemo.xcodeproj
# → Run, then pick "MiniCPM5 2B" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/ChatDemo
swift run chat-cli --model minicpm5-2b --prompt "What can you do, offline?"
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let chat = try await ChatSession(catalog: "minicpm5-2b")
let reply = try await chat.respond(to: prompt)
// reply: the answer, generated fully on-device
```

The take-home is [`Examples/ChatDemo/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/main/Examples/ChatDemo/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; the CLI is an argument shell over it, and
the GUI drives the same `ChatSession` across turns for its transcript.
Multi-turn? Hold the `ChatSession` and call `respond(to:)` per turn — it keeps the
conversation history; `streamResponse(to:)` yields tokens as they decode.

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` → product **CoreAIKit**
- Info.plist: none needed
- Entitlements: none on Mac; iPhone needs `com.apple.developer.kernel.increased-memory-limit` (the 2.7 GB cold specialization passes the default jetsam limit)
- First run downloads the model — 2.7 GB (Mac) / 2.7 GB (iPhone) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Measured

| | decode | prefill | numerics | size |
|---|---:|---:|---|---:|
| **iPhone 17 Pro** (A19 Pro, `PipelinedBench`, Release) | **22.4 tok/s** | 27.3 tok/s | **24/24 + 24/24 token-exact** vs HF fp32 (nat + oracle, the margin-clean alphabet prompt); engine ready 28.9 s cold | **2.7 GB** |
| **M4 Max** (macOS 27, `llm-benchmark`) | **127.6 tok/s** | 2654 tok/s | **16/16 token-exact** vs the fp32 oracle (margin-aware gate, min margin 0.925) | |

Free-run check (4 prompts × 30 greedy tokens vs fp32 HF, `verify_minicpm5.py`): **3/4 exact**; the one miss is a name at fp32 probability 0.2126 vs 0.2065 (`Emma`/`Lily`, top-2 margin 0.006) — a tie any precision may flip. The fp16 control export scores 4/4, and the **per-channel** int8 sibling of this bundle scored 2/4 with a real 0.245-margin flip (`,`→` and`), which is why this repo ships per-**block-32** scales instead (same recipe, three YAML lines; see *Quantization*).

⚠️ **iPhone context cap: prompt + generated tokens must stay under 1024.** The bundle declares a 131072
dynamic KV, and the shipped `CoreAIPipelinedEngine` caps iOS growing-KV capacity at 1024 (its guard
against the iOS compiler miscompiling growing-KV specializations at seq ≥ 2048) — so a phone
conversation truncates at absolute position 1024. Chunk or trim the history on iOS; macOS has no cap.

Same recipe as the published [MiniCPM5-1B](https://huggingface.co/mlboydaisuke/MiniCPM5-1B-CoreAI)
(int8 66.8 tok/s on the same phone) with one YAML changed — **per-block-32 scales instead of
per-channel**. Measured on the Mac before picking it (`llm-benchmark`, 512p/1024g): int8
per-channel 25.6 tok/s, fp16 80.0, int8 per-block-32 **127.6**. Per-channel int8 lowers to a
slow dequant path on the Mac GPU; block-32 lands on the fast quantized-matmul path, 5× the
per-channel decode and 1.6× fp16's, at +155 MB. On the phone the two decode the same
(bandwidth-bound), so block-32 wins on both.

## Quantization

Weight-only **symmetric int8, per-block-32** (a scale per 32-wide block along the input dim; no
clipping), applied as a torch pre-export pass via `coreai-opt`; SDPA / RoPE / RMSNorm stay full
precision. The 1B ships the per-channel version of the same config.

```bash
uv run coreai.llm.export openbmb/MiniCPM5-2B --experimental --compute-precision float16 \\
  --compression-config minicpm5_int8sym_b32.yaml
# minicpm5_int8sym_b32.yaml: quantization_config → op_state_spec.weight = {dtype: int8,
#   qscheme: symmetric, granularity: {type: per_block, block_size: 32}}
```

## Conversion notes

- **`llama → mistral` remap.** MiniCPM5-2B's `model_type` is `llama` (a plain `LlamaForCausalLM`:
  42 layers × hidden 2048, GQA 16:2, `head_dim` 128, RoPE θ 5e6, untied 130560-vocab head); the
  stock exporter has no `llama` graph family, but Mistral's builder is architecturally identical for
  this config (GQA, no qkv bias, no qk-norm, explicit `head_dim` honored). One-line remap in the
  model registry — the same line that ships the 1B.
- **Chat EOS.** Base `eos_token` is `</s>`, but the chat template ends turns with `<|im_end|>`
  (id 130073). The bundle's tokenizer `eos_token` is set to `<|im_end|>` (as Qwen ships) so
  generation halts cleanly — checked through the engine with the chat template applied: the model
  thinks, answers, and stops (124-token reply, 131.6 tok/s short-context on the M4 Max, on the
  published bundle).
- **Dynamic-shape bundle** → the Core AI pipelined engine (the iPhone path); a static iOS export
  routes to the static-shape engine instead, which this FM-format bundle doesn't target. The 2.67 GB
  single-file bundle cold-specializes on the phone in 28.9 s (no AOT); that step needs the
  increased-memory entitlement and ~3 GB of free phone storage.
- **Thinking.** The model thinks by default (`<think>…</think>` before the answer); pass
  `enable_thinking=False` through the chat template for a direct answer. Give generation a
  generous budget (the kit caps at 4096) — the think trace alone can run several hundred tokens.

## Run

```swift
import FoundationModels
import CoreAILanguageModels
let model = try await CoreAILanguageModel(resourcesAt: int8BundleURL)   // …/int8
let session = LanguageModelSession(model: model)
print(try await session.respond(to: "Explain on-device AI in one sentence."))
```

Or in the zoo's **CoreAIChat** app / the kit's **ChatDemo** (Model → "MiniCPM5 2B").

## Reproduce

Exporter, gate, card and port notes live in the
[Core AI model zoo](https://github.com/john-rocky/coreai-model-zoo):
[`models/minicpm5-2b/`](https://github.com/john-rocky/coreai-model-zoo/tree/main/models/minicpm5-2b),
[`conversion/export_minicpm5.py`](https://github.com/john-rocky/coreai-model-zoo/blob/main/conversion/export_minicpm5.py),
[`knowledge/minicpm5-1b.md`](https://github.com/john-rocky/coreai-model-zoo/blob/main/knowledge/minicpm5-1b.md).

```bash
python3 conversion/zoo_convert.py show minicpm5-2b
python3 conversion/zoo_convert.py run  minicpm5-2b
```

## Credits

Model: **MiniCPM5-2B** by **OpenBMB** ([openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B), Apache-2.0).
Core AI conversion: the Core AI model zoo.
"""

DESCRIPTOR = {
    "model_type": "coreai-aimodel",
    "format": "aimodel",
    "framework": "Apple Core AI (iOS 27 / macOS 27)",
    "repository": "https://github.com/john-rocky/coreai-model-zoo",
    "note": "Converted .aimodel bundles for Apple's Core AI framework. See README.md for the bundle layout and run instructions.",
}


def stage() -> None:
    assert not re.search(r"__[A-Z0-9_]+__", CARD), "fill the measured numbers before staging"
    if STAGE.exists():
        shutil.rmtree(STAGE)
    dest = STAGE / "int8"
    dest.mkdir(parents=True)
    # bundle: rename <export-name>.aimodel -> minicpm5_2b_int8.aimodel and point metadata at it
    src_asset = next(BUNDLE.glob("*.aimodel"))
    shutil.copytree(src_asset, dest / f"{PUBLISHED_NAME}.aimodel")
    shutil.copytree(BUNDLE / "tokenizer", dest / "tokenizer")
    meta = json.loads((BUNDLE / "metadata.json").read_text())
    meta["name"] = PUBLISHED_NAME
    meta["assets"]["main"] = f"{PUBLISHED_NAME}.aimodel"
    (dest / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    inner = dest / f"{PUBLISHED_NAME}.aimodel" / "metadata.json"
    if inner.exists():
        im = json.loads(inner.read_text())
        if "name" in im:
            im["name"] = PUBLISHED_NAME
            inner.write_text(json.dumps(im, indent=2) + "\n")
    (STAGE / "README.md").write_text(CARD)
    (STAGE / "config.json").write_text(json.dumps(DESCRIPTOR, indent=2) + "\n")
    shutil.copy(hf_hub_download(*LICENSE_FROM), STAGE / "LICENSE")
    total = sum(f.stat().st_size for f in STAGE.rglob("*") if f.is_file())
    print(f"staged {STAGE}  ({total / 1e6:.0f} MB)")
    for f in sorted(STAGE.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(STAGE)}  {f.stat().st_size / 1e6:.1f} MB")


def upload() -> None:
    api = HfApi()
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=REPO, folder_path=str(STAGE), repo_type="model",
                      commit_message="MiniCPM5-2B — Core AI int8 bundle (dynamic, pipelined engine)")
    print(f"uploaded -> https://huggingface.co/{REPO}")


if __name__ == "__main__":
    stage()
    if "--stage-only" not in sys.argv:
        upload()
