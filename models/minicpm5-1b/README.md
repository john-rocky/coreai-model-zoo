# MiniCPM5-1B — Core AI

[🤗 mlboydaisuke/MiniCPM5-1B-CoreAI](https://huggingface.co/mlboydaisuke/MiniCPM5-1B-CoreAI) · Apache-2.0 · base [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)

OpenBMB's 1.08B on-device LLM (hybrid Think / No-Think reasoning, 128K context, 1B-class open-source SOTA), converted to Apple **Core AI** and running fully on-device on iPhone — on the **GPU** (int8, pipelined engine) and, since 2026-09-15, on the **Neural Engine** (Apple's stock static iOS export, 8-bit palettized, AOT h18p; `ios-ane-h18p/` on HF).

> **2026-09-09 — the bundle was rebuilt (int8 per-block-32) and the HF revision moved.** The
> per-channel int8 bundle published as `5ad650f` had dead LM-head rows from vocab id ~65024 up,
> `<|im_end|>` (130073) included, so a chat turn never halted and any high-id token was lost; its
> "24/24 token-exact" was real and blind to it (the checked continuation never left the low vocab).
> Measurements and the gate that catches it: [`knowledge/minicpm5-1b.md`](../../knowledge/minicpm5-1b.md)
> (2026-09-09 section). Pin the new revision or newer.

<!-- gen-cards:use-it begin id=minicpm5-1b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

**New to Core AI? [Start with CoreAIKit 0.7.1](https://github.com/john-rocky/coreai-kit#readme).** Follow its requirements and first-run steps for `qwen3-0.6b`, then open the same release's [ChatDemo](https://github.com/john-rocky/coreai-kit/tree/0.7.1/Examples/ChatDemo). The README records the tested OS/SDK and download size; model and device coverage is stated per example.

⚡ **One line** — run the kit's task op on this model
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let tldr = try await CoreAI.summarize(text, options: .model("minicpm5-1b"))
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/0.7.1/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [ChatDemo runner](https://github.com/john-rocky/coreai-kit/tree/0.7.1/Examples/ChatDemo)
(GUI + CLI, one app for every chat model in the catalog):

```bash
git clone --branch 0.7.1 --depth 1 https://github.com/john-rocky/coreai-kit
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
open -a /Applications/Xcode-27.0.0-RC.app coreai-kit/Examples/ChatDemo/ChatDemo.xcodeproj
# → Run, then pick "MiniCPM5 1B" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/ChatDemo
swift run -c release chat-cli --model minicpm5-1b --prompt "What can you do, offline?"
```

Use Xcode build **27A266a** from the release's `.xcode-pin`; adjust the app path if your installation is named differently.

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let id = "minicpm5-1b"
let chat: ChatSession
if id == "qwen3-0.6b" {
    // Freeze the release starter; other selections retain the live catalog's
    // model-specific dispatch (including paired Gemma bundles).
    guard let model = ModelCatalog.builtin.entry(id: id)?.modelID else {
        throw CoreAIKitError.modelNotAvailableOnPlatform(id: id)
    }
    chat = try await ChatSession(model: model)
} else {
    chat = try await ChatSession(catalog: "minicpm5-1b")
}
let reply = try await chat.respond(to: prompt)
// reply: the answer, generated fully on-device
```

The take-home is [`Examples/ChatDemo/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/0.7.1/Examples/ChatDemo/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; the CLI is an argument shell over it, and
the GUI drives the same `ChatSession` across turns for its transcript.
Multi-turn? Hold the `ChatSession` and call `respond(to:)` per turn — it keeps the
conversation history; `streamResponse(to:)` yields tokens as they decode.

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` (exact **0.7.1**) → product **CoreAIKit**
- Info.plist: none needed
- Entitlements: none needed
- First run downloads the model — ~1,159 MB (Mac) / ~1,159 MB (iPhone) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Measured (rebuilt bundle, 2026-09-09)

| | decode | prefill | numerics | size |
|---|---:|---:|---|---:|
| **iPhone 17 Pro** (A19 Pro — `PipelinedBench`, random 128-tok prompt, greedy, Release; medians of 5 trials over 2 launches: decode 63.6 / 61.2 / 48.4 / 62.7 / 61.7) | **61.7 tok/s** | 65.6 tok/s | **24/24 token-exact** vs HF fp32 on the alphabet prompt (min fp32 top-2 margin 0.841) **+ 6/6 including the stop** on the no-think turn `1+1=?`; engine ready 7.3 s cold, 0.2 s warm | **1.1 GB** |
| **M4 Max** (macOS 27, `llm-benchmark`, 512p/1024g, 5 trials) | **246.6 tok/s** | 6649 tok/s | **16/16 token-exact** vs the fp32 oracle (`cli/coreai_verify.py`, min margin 0.913, [`gate-minicpm5-1b.json`](gate-minicpm5-1b.json)) + **6/6 and stops** on `--chat no-think --prompt "1+1=?"` ([`gate-minicpm5-1b-stop.json`](gate-minicpm5-1b-stop.json)) | |
| **iPhone 17 Pro, Neural Engine** (`ios-ane-h18p/`, **8-bit** palettized g32, static graphs, Apple llm-benchmark method p512 / g1024 n5, two back-to-back runs) | **69.6 / 58.6 tok/s** (per-trial 71.3 → 57.4, thermal) | 2710 / 2364 tok/s | **PASS 3/3** on the device gate vs HF fp32: teacher-forced 24/24 + 10/10 (`Say hello.`, stop at margin 0.909) + 16/16, free-run token-exact incl. the stop — [`gate-minicpm5-1b-ane-device.json`](gate-minicpm5-1b-ane-device.json); Apple's default **4-bit** preset FAILS the same fixture ([`gate-minicpm5-1b-ane-device-4bit-FAIL.json`](gate-minicpm5-1b-ane-device-4bit-FAIL.json)); footprint 1.5 GB, cold load 38 s / warm 0.05 s | **1.3 GB** |

Same-day interleaved A/B on the phone (one app embedding both bundles, p128 / g256 n5, order ANE-GPU-ANE-GPU, [`bench-iphone-ane-vs-gpu-2026-09-15.json`](bench-iphone-ane-vs-gpu-2026-09-15.json)): **Neural Engine 8-bit 76.8 / 76.8 tok/s decode, 3899 / 3865 prefill** vs **GPU int8 (this repo's `int8/`, pipelined engine) 67.1 / 64.4 decode, 2359 / 2270 prefill** — the two arms read the same weight bytes per token (8-bit LUT vs int8), so on the 1B the ANE is +17 % decode and +70 % prefill at equal precision.

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
- **Neural Engine bundle (`ios-ane-h18p/`, 2026-09-15)** — Apple's stock static iOS export, `export_minicpm5.py --ios-ane --qconfig minicpm5_pal8_g32.yaml`: k-means **8-bit** palettization (group 32; embeddings int8; static graphs `prompt_opt`/`extend` × {256, 512, 1024, 2048, 4096} × {8, 16, 64}) then `xcrun coreai-build compile … --platform iOS --preferred-compute neural-engine --architecture h18p` → 31/31 ANE regions. Why 8-bit: Apple's default `4bit_weight_palettized_group32` passes the alphabet and long-prompt sweeps but flips two margin-clear steps on a chat turn (`' How'`→`' 😊'` at 0.593, `' today'`→`'?'` at 0.995 — fluent, stops, not the fp32 answer), and the same two flips survive with fp16 embeddings, so the loss is in the 4-bit body/head; 8 bits are clean on every step. The 2B passes at 4 bits. `ios-static/` is the same export before AOT (portable IR). Tool and rules: [`../../knowledge/minicpm5-1b.md`](../../knowledge/minicpm5-1b.md) §2026-09-15.
- **2026-09-16 fidelity probe on a longer chat prompt** ("Explain in a short paragraph why the sky is blue.", no-think, greedy, identical prompt ids, fp32 `transformers` oracle 87 tokens): both bundles stay on the fp32 answer to the floor — the **8-bit ANE bundle matches 21 tokens** and the **int8 GPU bundle 6 tokens** before each diverges on a 0.051 near-tie (below the 0.1 floor); no margin-clear flip on either — [`fidelity-probe-sky-2026-09-16.json`](fidelity-probe-sky-2026-09-16.json). Speeds on that run: ANE 73.2 tok/s, GPU 62.3 tok/s decode (200 tokens, iOS 27.0 24A437). The 2B's 4-bit ANE bundle does **not** pass this probe (see its card).

## Gate

```bash
python3 cli/coreai_verify.py <bundle> -n 16                                                   # alphabet, margin-clean parity
python3 cli/coreai_verify.py <bundle> --chat no-think --prompt "1+1=?" -n 16 --must-stop-within 16   # the stop is a gated step
python3 cli/coreai_verify.py <bundle> --chat think    --prompt "1+1=?" -n 0  --must-stop-within 400  # Think-mode halt check
# Neural Engine bundle: python3 conversion/zoo_convert.py run minicpm5-1b-ane, then the on-device gate (AneGateRunner, knowledge/minicpm5-1b.md §2026-09-15)
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
