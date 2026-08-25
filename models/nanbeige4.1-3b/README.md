# Nanbeige4.1-3B (text decoder) — Core AI

Plain-Llama dense decoder (Nanbeige LLM Lab): **32 layers**, GQA **20 q / 4 kv heads, head_dim 128**,
hidden 2560, SwiGLU intermediate 10496, **vocab 166144 (untied lm_head)**, RMSNorm eps 1e-5, RoPE
θ=70M, context 262144 — **no QK-norm, no qkv/mlp bias** (the textbook Llama shape; `model_type: "llama"`,
3.93B total / ~3B non-embedding backbone). Source: `Nanbeige/Nanbeige4.1-3B` (Apache-2.0). A reasoning /
agentic model whose first-party card claims it beats Qwen3-4B and rivals Qwen3-32B / Qwen3-30B-A3B
(LiveCodeBench-Pro-Easy 81.4 vs 40.2, AIME 2026-I 87.4, GPQA 83.8) — a 32B-class reasoner at 3.93B,
running on an iPhone.

**⬇️ Converted `.aimodel` bundle (ready to run):
[mlboydaisuke/Nanbeige4.1-3B-CoreAI](https://huggingface.co/mlboydaisuke/Nanbeige4.1-3B-CoreAI)** —
`gpu-pipelined/nanbeige4_1_3b_decode_int8hu_block32_sym_s1/` (full LanguageBundle incl. tokenizer).

The first **plain-Llama** model on the [pipelined-engine fast path](../../knowledge/pipelined-engine.md):
it reuses `qwen3.py` MINUS the q/k-norm (qwen3 already has a bias-free fused QKV), so the body is the
existing overlay with one norm removed — see `models/macos/llama.py`. Pure-attention, KV-only state
(no conv / recurrent), so it needs no engine patch beyond the base stack.

<!-- gen-cards:use-it begin id=nanbeige4.1-3b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — run the kit's task op on this model
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let tldr = try await CoreAI.summarize(text, options: .model("nanbeige4.1-3b"))
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [ChatDemo runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo)
(GUI + CLI, one app for every chat model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/ChatDemo/ChatDemo.xcodeproj
# → Run, then pick "Nanbeige4.1 3B" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/ChatDemo
swift run chat-cli --model nanbeige4.1-3b --prompt "What can you do, offline?"
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let chat = try await ChatSession(catalog: "nanbeige4.1-3b")
let reply = try await chat.respond(to: prompt)
// reply: the answer, generated fully on-device
```

Also runs behind **Apple's FoundationModels API** — CoreAIKit's [`KitLanguageModel`](https://github.com/john-rocky/coreai-kit#works-with-apples-foundationmodels-api) plugs this bundle into the system `LanguageModelSession`; capabilities (tool calling, guided generation) auto-detect per model.

The take-home is [`Examples/ChatDemo/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/main/Examples/ChatDemo/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; the CLI is an argument shell over it, and
the GUI drives the same `ChatSession` across turns for its transcript.
Multi-turn? Hold the `ChatSession` and call `respond(to:)` per turn — it keeps the
conversation history; `streamResponse(to:)` yields tokens as they decode.

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` → product **CoreAIKit**
- Info.plist: none needed
- Entitlements (iOS): `com.apple.developer.kernel.increased-memory-limit`
- First run downloads the model — 4.4 GB (Mac) / 4.4 GB (iPhone) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Measured (macOS + iOS 27 beta, release builds, p=128 g=256, `COREAI_CHUNK_THRESHOLD=1`)

| config | bundle | prefill tok/s | decode tok/s | numerics |
|---|---:|---:|---:|---|
| **int8hu --head-sym (ship), M4 Max** | **4.3 GB** | **114.9** | **114.5** | engine ≡ fp32-HF oracle (raw greedy → "Paris"); reasoning coherent (trick "17 sheep, all but 9" → reasons to 9) |
| **int8hu --head-sym --static-ids (`_s1`, ship), iPhone 17 Pro** | 4.3 GB | **16.6** | **15.9** | **nat 24/24 + oracle 24/24 — token-identical to the M4 Max GPU reference (Paris / Tokyo + full continuation)** |

- **Loads on iPhone 17 Pro**: cold GPU specialization `engine ready 53.5 s`, device free 51 GB, **no
  jetsam / no std::bad_alloc** — the largest bundle we have run on the pipelined bench (4.58 GB payload).
- **`--static-ids` is REQUIRED for the device.** The generic dynamic-`input_ids` export is fast on the
  Mac but on the iPhone pipelined engine (chunkThreshold=1, every step S=1) it pays a **per-step
  input_ids re-specialization** that is pathological on a 4.3 GB model (~37 s/step cold; the 900 s
  probe never finished the first 24-token run). Fixing `input_ids` at `[1,1]` (the qwen3.5 loop-free
  device pattern; `--static-ids` → `_s1` bundle) eliminates it — chunkThreshold=1 feeds S=1 anyway, so
  no prefill loss — and the device numerics complete 24/24.
- **The untied 166144-vocab head** is ~0.85 GB; quantize it absmax per-block-32 int8 (`--head-sym`,
  plain `symmetric`). `symmetric_with_clipping` craters big-vocab heads (the documented qwen lever).

### int4: NO-GO — int8 is this reasoning model's floor

`int4hu` (body int4 per-block-32 + int8 head) is 2.9 GB and 169 tok/s on the Mac, and its **raw
single-token greedy still returns "Paris"** — but multi-token **reasoning CRATERS**: the same
"17 sheep, all but 9 run away" trick collapses to a wrong "17" with a repetition loop and Chinese
drift. The single-token probe is **misleading for a reasoning model** — you must check multi-token.
This is the non-QAT-int4 structural cliff (same wall as qwen3.5 / LFM2.5; needs QAT). **Palettized
(k-means) int4 does not rescue it** either — for non-QAT weights the cliff is the scheme-independent
property, and on the GPU-pipelined path the LUT dequant is *slower* than linear besides. int8hu ships.

### ANE: right architecture class, wrong size

Plain-dense is the one class that *could* ride the ANE (where the LUT-friendly palettized weights run
native-fast, unlike on the GPU). But the ANE sweet spot is the **~0.6–1B** rung (tied head): a 0.6B
fully-palettized model rides the ANE blazing. At **3.93B + a 166144 untied head** Nanbeige overruns
the ANE working set, so it ships **GPU-pipelined** like the rest of the dense line. The ANE-blazing
target is a 1B plain-dense model, not this one.

## Numerics gating

- Parity ladder (fp32 eager vs native HF `LlamaForCausalLM` oracle, no trust_remote_code): teacher-forced
  top-1 **24/24, cosine 1.000000, max-abs-logit Δ = 0** (`_smoke/test_nanbeige_parity.py`, `USE_HF_IMPL=true`).
- Engine gate: raw-token greedy on the int8hu bundle reproduces the fp32 oracle's first token ("Paris");
  reasoning output coherent and correct.
- Device gate: iPhone greedy sequences **24/24 token-identical** to the Mac reference on both fixed
  prompts (`_smoke/gen_nanbeige_device_ref_tokens.py`). Reasoning models drift on a bare prompt after the
  answer — the **first token is the anchor** (Paris 9965 / Tokyo 20150) and the full 24 still matched here.

## Convert it yourself

```bash
cd coreai-models   # with the plain-Llama overlay (models/macos/llama.py) in place
# device ship (REQUIRED static [1,1] for fast iPhone decode):
.venv/bin/python ../coreai-models-community/conversion/export_nanbeige41_decode_pipelined.py \
    int8hu --head-sym --static-ids
COREAI_CHUNK_THRESHOLD=1 ./.build/out/Products/Release/llm-benchmark \
    --model exports/nanbeige4_1_3b_decode_int8hu_block32_sym_s1 -p 128 -g 256 -n 3
```

Run contract: `COREAI_CHUNK_THRESHOLD=1` before engine creation; the bundle's `input_ids` is static
`[1,1]`, so every prefill token is fed as an S=1 step (never call `engine.warmup()` — warm with a
1-token generate; `llm-runner` needs `--warmup exact --warmup-length 1`).

## License

Model weights and conversion code: **Apache-2.0** (Nanbeige LLM Lab upstream; the conversion code in this
repo is BSD-3-Clause). Redistribution retains the upstream notices.
