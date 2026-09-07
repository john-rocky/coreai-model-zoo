# MiniCPM5-2B — Core AI

[🤗 mlboydaisuke/MiniCPM5-2B-CoreAI](https://huggingface.co/mlboydaisuke/MiniCPM5-2B-CoreAI) · Apache-2.0 · base [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)

OpenBMB's 2.5B on-device LLM (released 2026-09-06; hybrid Think / No-Think reasoning, native tool calling, 128K context; OpenBMB's card reports 2B-class open-source SOTA — LiveCodeBench v6 69.1, AIME 2026 86.5, BFCL v4 66.6, SWE-bench Verified 46.4), converted to Apple **Core AI** and running fully on-device on iPhone via the pipelined engine. The [MiniCPM5-1B](../minicpm5-1b/README.md) recipe with one YAML changed: same `LlamaForCausalLM` family (42 layers × hidden 2048 instead of 24 × 1536), same chat-EOS fix, int8 with **per-block-32** scales instead of per-channel — which turned out to matter (below).

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
| **iPhone 17 Pro** (A19 Pro — `PipelinedBench`, random 128-tok prompt, greedy, Release) | **22.4 tok/s** | 27.3 tok/s | **24/24 + 24/24 token-exact** vs HF fp32 (nat + oracle, the margin-clean alphabet prompt); engine ready 28.9 s cold | **2.7 GB** |
| **M4 Max** (macOS 27 — `llm-benchmark`, 512p / 1024g) | **127.6 tok/s** | 2654 tok/s | **16/16 token-exact** vs the fp32 oracle (margin-aware gate, min margin 0.925) | |

Free-run check (4 prompts × 30 greedy tokens vs fp32 HF, `verify_minicpm5.py`): **3/4 exact**; the one miss is a name at fp32 probability 0.2126 vs 0.2065 (`Emma`/`Lily`, top-2 margin 0.006) — a tie any precision may flip. The fp16 control export scores 4/4, and the **per-channel** int8 sibling of this bundle scored 2/4 with a real 0.245-margin flip (`,`→` and`), which is why this repo ships per-**block-32** scales instead (same recipe, three YAML lines; see *Quantization*).

⚠️ **iPhone context cap: prompt + generated tokens must stay under 1024.** The bundle declares a 131072
dynamic KV, and the shipped `CoreAIPipelinedEngine` caps iOS growing-KV capacity at 1024 (its guard
against the iOS compiler miscompiling growing-KV specializations at seq ≥ 2048) — so a phone
conversation truncates at absolute position 1024. Chunk or trim the history on iOS; macOS has no cap.

Three bundles were measured on the Mac before picking this one (`llm-benchmark`, 512p/1024g, [`bench-mac-llm-benchmark.json`](bench-mac-llm-benchmark.json)): int8 **per-channel** (the 1B's yaml) 25.6 tok/s, fp16 80.0, int8 **per-block-32** **127.6**. Per-channel int8 lowers to a slow dequant path on the Mac GPU; block-32 lands on the fast quantized-matmul path, 5× the per-channel decode and 1.6× fp16's, at +155 MB. On the phone the two int8 shapes decode the same (bandwidth-bound: 22.4 vs 22.7 tok/s), so block-32 wins everywhere.

Gate transcript: [`gate-minicpm5-2b.json`](gate-minicpm5-2b.json) (`cli/coreai_verify.py`, fp32 oracle, margin-aware).

## Conversion

- **`llama → mistral` remap** — MiniCPM5-2B is a plain `LlamaForCausalLM` (GQA 16:2, `head_dim` 128, RoPE θ 5e6, no scaling, untied 130560-vocab head); the stock exporter has no `llama` graph family, but the Mistral builder is architecturally identical (GQA, no qkv bias, no qk-norm, explicit `head_dim`). The one-line remap that ships the 1B, untouched.
- **int8, per-block-32** — weight-only symmetric int8 with a scale per 32-wide block along the input dim (`minicpm5_int8sym_b32.yaml`: `granularity: {type: per_block, block_size: 32}`, axis resolved per module by the quantizer; no clipping; SDPA/RoPE/RMSNorm full precision) via `coreai.llm.export … --compression-config` (coreai-opt torch pre-export). The 1B's per-channel yaml (`minicpm5_int8sym.yaml`) is the `--qconfig` default of the wrapper; on the 2B it flipped one 0.245-margin greedy token that fp16 reproduces exactly and decoded 5× slower on the Mac GPU, so the 2B names the block-32 yaml in its recipe.
- **Chat EOS** — base `eos_token` is `</s>`, but the chat template ends turns with `<|im_end|>` (130073); the bundle's tokenizer `eos_token` is set to `<|im_end|>` (as Qwen ships) so generation halts cleanly. Declared in [`verify.toml`](verify.toml). Checked end to end through the engine: `llm-runner --prompt "Explain on-device AI in one sentence."` (chat template applied) thinks, answers, and stops at `<|im_end|>` — 124 tokens, 131.6 tok/s short-context on the M4 Max, on the published bundle.
- **Dynamic-shape bundle** → the pipelined engine (the iPhone path). Runs unchanged on macOS and iOS; no AOT needed — the 2.67 GB single-file bundle cold-specializes on the phone in 28.9 s, then the cache persists. That specialization needs the increased-memory entitlement and ~3 GB of free phone storage (a full phone fails it with `No space left on device`).
- **Thinking** — on by default (`<think>…</think>` first); `enable_thinking=False` in the chat template for a direct answer. The think trace alone can run several hundred tokens, so cap generation generously (the kit uses 4096).
- **Not ported (yet): `openbmb/MiniCPM5-2B-DSpark`** — a 5-layer, 324M draft model (7 draft tokens per pass, `num_target_layers` 42) trained for exact pairing with this checkpoint. A dense target with an official drafter is the clean spec-decode test bed the zoo's n-gram work asked for; see [`../../knowledge/spec-decode-ngram-dense.md`](../../knowledge/spec-decode-ngram-dense.md).

## Run

In the zoo's **CoreAIChat** app (Model → "MiniCPM5 2B"), the kit's ChatDemo, or via Foundation Models:

```swift
import FoundationModels
import CoreAILanguageModels
let model = try await CoreAILanguageModel(resourcesAt: int8BundleURL)
let session = LanguageModelSession(model: model)
print(try await session.respond(to: "Explain on-device AI in one sentence."))
```

## Reproduce

```bash
python3 conversion/zoo_convert.py show minicpm5-2b
python3 conversion/zoo_convert.py run  minicpm5-2b        # export_minicpm5.py --hf-id openbmb/MiniCPM5-2B --qconfig minicpm5_int8sym_b32.yaml
python3 cli/coreai_verify.py <bundle> -n 16 --transcript models/minicpm5-2b/gate-minicpm5-2b.json
```
