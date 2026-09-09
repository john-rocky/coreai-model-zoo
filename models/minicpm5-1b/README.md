# MiniCPM5-1B — Core AI

[🤗 mlboydaisuke/MiniCPM5-1B-CoreAI](https://huggingface.co/mlboydaisuke/MiniCPM5-1B-CoreAI) · Apache-2.0 · base [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)

OpenBMB's 1.08B on-device LLM (hybrid Think / No-Think reasoning, 128K context, 1B-class open-source SOTA), converted to Apple **Core AI** and running fully on-device on iPhone via the pipelined engine.

> **2026-09-09 — the bundle was rebuilt (int8 per-block-32) and the HF revision moved.** The
> per-channel int8 bundle published as `5ad650f` had dead LM-head rows from vocab id ~65024 up,
> `<|im_end|>` (130073) included, so a chat turn never halted and any high-id token was lost; its
> "24/24 token-exact" was real and blind to it (the checked continuation never left the low vocab).
> Measurements and the gate that catches it: [`knowledge/minicpm5-1b.md`](../../knowledge/minicpm5-1b.md)
> (2026-09-09 section). Pin the new revision or newer.

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

## Measured (rebuilt bundle, 2026-09-09)

| | decode | prefill | numerics | size |
|---|---:|---:|---|---:|
| **iPhone 17 Pro** (A19 Pro — `PipelinedBench`, random 128-tok prompt, greedy, Release; medians of 5 trials over 2 launches: decode 63.6 / 61.2 / 48.4 / 62.7 / 61.7) | **61.7 tok/s** | 65.6 tok/s | **24/24 token-exact** vs HF fp32 on the alphabet prompt (min fp32 top-2 margin 0.841) **+ 6/6 including the stop** on the no-think turn `1+1=?`; engine ready 7.3 s cold, 0.2 s warm | **1.1 GB** |
| **M4 Max** (macOS 27, `llm-benchmark`, 512p/1024g, 5 trials) | **246.6 tok/s** | 6649 tok/s | **16/16 token-exact** vs the fp32 oracle (`cli/coreai_verify.py`, min margin 0.913, [`gate-minicpm5-1b.json`](gate-minicpm5-1b.json)) + **6/6 and stops** on `--chat no-think --prompt "1+1=?"` ([`gate-minicpm5-1b-stop.json`](gate-minicpm5-1b-stop.json)) | |

The per-channel bundle it replaces, on the same Mac the same night: 53.7 tok/s decode, and **FAIL** on the
stop gate (5/6, then runs to the cap against the oracle's 0.80-margin `<|im_end|>`) and on the Think-mode halt check
(400-token cap). The rebuilt bundle's Think-mode `1+1=?` halts after 171 tokens; the 2B after 190.

⚠️ **iPhone context cap: prompt + generated tokens must stay under 1024** (the shipped pipelined engine caps iOS growing-KV
capacity at 1024). Trim or chunk the history on the phone; macOS has no cap.

## Conversion

- **`llama → mistral` remap** — MiniCPM5-1B is a plain `LlamaForCausalLM`; the stock exporter has no `llama` graph family, but the Mistral builder is architecturally identical (GQA, no qkv bias, no qk-norm, explicit `head_dim`). One-line remap in the model registry.
- **int8 per-block-32** — weight-only symmetric (a scale per 32-wide block along the input dim, no clipping; SDPA/RoPE/RMSNorm full precision) via `coreai.llm.export … --compression-config minicpm5_int8sym_b32.yaml` (coreai-opt torch pre-export) — the same YAML as the 2B. The per-channel sibling (`minicpm5_int8sym.yaml`) is the arm that produced the dead head rows and must not ship; fresh per-channel exports on the current toolchain reproduce them exactly, while per-block-32, the CLI's int4 default and fp16 are clean on the same teacher-forced probes.
- **Chat EOS** — base `eos_token` is `</s>`, but the chat template ends turns with `<|im_end|>` (130073); the bundle's tokenizer `eos_token` is set to `<|im_end|>` (as Qwen ships). Necessary, not sufficient: the gate now checks that the engine actually stops where the fp32 oracle stops.
- **Dynamic-shape bundle** → the pipelined engine (the iPhone path). Runs unchanged on macOS and iOS.

## Gate

```bash
python3 cli/coreai_verify.py <bundle> -n 16                                                   # alphabet, margin-clean parity
python3 cli/coreai_verify.py <bundle> --chat no-think --prompt "1+1=?" -n 16 --must-stop-within 16   # the stop is a gated step
python3 cli/coreai_verify.py <bundle> --chat think    --prompt "1+1=?" -n 0  --must-stop-within 400  # Think-mode halt check
```

A cross-model form of the stop gate — `--prompt "Reply with only the number: 1+1=?"` — reaches `2` `<|im_end|>` at step 1
on both the 1B (min margin 0.315) and the 2B (0.992); the plain `1+1=?` is rejected on the 2B (two positions below the
0.1 floor) because the 2B answers it at length.

## Run

In the zoo's **CoreAIChat** app (Model → "MiniCPM5 1B"), or via Foundation Models:

```swift
import FoundationModels
import CoreAILanguageModels
let model = try await CoreAILanguageModel(resourcesAt: int8BundleURL)
let session = LanguageModelSession(model: model)
print(try await session.respond(to: "Explain on-device AI in one sentence."))
```
