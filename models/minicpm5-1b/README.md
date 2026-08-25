# MiniCPM5-1B — Core AI

[🤗 mlboydaisuke/MiniCPM5-1B-CoreAI](https://huggingface.co/mlboydaisuke/MiniCPM5-1B-CoreAI) · Apache-2.0 · base [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)

OpenBMB's 1.08B on-device LLM (hybrid Think / No-Think reasoning, 128K context, 1B-class open-source SOTA), converted to Apple **Core AI** and running fully on-device on iPhone via the pipelined engine.

<!-- gen-cards:use-it begin id=minicpm5-1b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — run the kit's task op on this model
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let tldr = try await CoreAI.summarize(text, options: .model("minicpm5-1b"))
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [ChatDemo runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo)
(GUI + CLI, one app for every chat model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/ChatDemo/ChatDemo.xcodeproj
# → Run, then pick "MiniCPM5 1B" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/ChatDemo
swift run chat-cli --model minicpm5-1b --prompt "What can you do, offline?"
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let chat = try await ChatSession(catalog: "minicpm5-1b")
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
- Entitlements: none needed
- First run downloads the model — 1.1 GB (Mac) / 1.1 GB (iPhone) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## On-device (iPhone 17 Pro, A19 Pro — `PipelinedBench`, random 128-tok prompt, greedy)

| bundle | decode | prefill | quality | size |
|---|---:|---:|---|---:|
| **`int8/`** (ship) | **66.8 tok/s** | 68.0 tok/s | **lossless** (24/24 token-exact vs HF fp32) | **1.0 GB** |

int8 is **~2.2× faster than fp16** on iPhone (decode is memory-bandwidth-bound → half the weight read ≈ double throughput) at **no quality cost**. So int8 strictly dominates fp16 here.

## Conversion

- **`llama → mistral` remap** — MiniCPM5-1B is a plain `LlamaForCausalLM`; the stock exporter has no `llama` graph family, but the Mistral builder is architecturally identical (GQA, no qkv bias, no qk-norm, explicit `head_dim`). One-line remap in the model registry.
- **int8** — weight-only symmetric per-channel (absmax, no clipping; SDPA/RoPE/RMSNorm full precision) via `coreai.llm.export … --compression-config` with a `quantization_config` (coreai-opt torch pre-export). Same recipe family as the zoo's `sym8`.
- **Chat EOS** — base `eos_token` is `</s>`, but the chat template ends turns with `<|im_end|>` (130073); the bundle's tokenizer `eos_token` is set to `<|im_end|>` (as Qwen ships) so generation halts cleanly.
- **Dynamic-shape bundle** → the pipelined engine (the iPhone path). Runs unchanged on macOS and iOS.

## Run

In the zoo's **CoreAIChat** app (Model → "MiniCPM5 1B"), or via Foundation Models:

```swift
import FoundationModels
import CoreAILanguageModels
let model = try await CoreAILanguageModel(resourcesAt: int8BundleURL)
let session = LanguageModelSession(model: model)
print(try await session.respond(to: "Explain on-device AI in one sentence."))
```
