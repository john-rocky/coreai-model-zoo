# MiniCPM5-2B — Core AI

[🤗 mlboydaisuke/MiniCPM5-2B-CoreAI](https://huggingface.co/mlboydaisuke/MiniCPM5-2B-CoreAI) · Apache-2.0 · base [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)

OpenBMB's 2.5B on-device LLM (released 2026-09-06; hybrid Think / No-Think reasoning, native tool calling, 128K context; OpenBMB's card reports 2B-class open-source SOTA — LiveCodeBench v6 69.1, AIME 2026 86.5, BFCL v4 66.6, SWE-bench Verified 46.4), converted to Apple **Core AI** and running fully on-device on iPhone — on the **GPU** (int8, pipelined engine) and, since 2026-09-15, on the **Neural Engine** (Apple's stock static iOS export, AOT h18p; `ios-ane-h18p/` on HF — **6-bit palettized since 2026-09-16**, replacing the 4-bit export that was not fp32-faithful on long answers). The [MiniCPM5-1B](../minicpm5-1b/README.md) recipe with one YAML changed: same `LlamaForCausalLM` family (42 layers × hidden 2048 instead of 24 × 1536), same chat-EOS fix, int8 with **per-block-32** scales instead of per-channel — which turned out to matter (below).

<!-- gen-cards:use-it begin id=minicpm5-2b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

**New to Core AI? [Start with CoreAIKit 0.7.3](https://github.com/john-rocky/coreai-kit#readme).** Follow its requirements and first-run steps for `qwen3-0.6b`, then open the same release's [ChatDemo](https://github.com/john-rocky/coreai-kit/tree/0.7.3/Examples/ChatDemo). The README records the tested OS/SDK and download size; model and device coverage is stated per example.

⚡ **One line** — run the kit's task op on this model
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let tldr = try await CoreAI.summarize(text, options: .model("minicpm5-2b"))
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/0.7.3/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [ChatDemo runner](https://github.com/john-rocky/coreai-kit/tree/0.7.3/Examples/ChatDemo)
(GUI + CLI, one app for every chat model in the catalog):

```bash
git clone --branch 0.7.3 --depth 1 https://github.com/john-rocky/coreai-kit
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
open -a /Applications/Xcode-27.0.0-RC.app coreai-kit/Examples/ChatDemo/ChatDemo.xcodeproj
# → Run, then pick "MiniCPM5 2B" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/ChatDemo
swift run -c release chat-cli --model minicpm5-2b --prompt "What can you do, offline?"
```

Use Xcode build **27A266a** from the release's `.xcode-pin`; adjust the app path if your installation is named differently.

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let id = "minicpm5-2b"
let chat: ChatSession
if id == "qwen3-0.6b" {
    // Freeze the release starter; other selections retain the live catalog's
    // model-specific dispatch (including paired Gemma bundles).
    guard let model = ModelCatalog.builtin.entry(id: id)?.modelID else {
        throw CoreAIKitError.modelNotAvailableOnPlatform(id: id)
    }
    chat = try await ChatSession(model: model)
} else {
    chat = try await ChatSession(catalog: "minicpm5-2b")
}
let reply = try await chat.respond(to: prompt)
// reply: the answer, generated fully on-device
```

The take-home is [`Examples/ChatDemo/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/0.7.3/Examples/ChatDemo/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; the CLI is an argument shell over it, and
the GUI drives the same `ChatSession` across turns for its transcript.
Multi-turn? Hold the `ChatSession` and call `respond(to:)` per turn — it keeps the
conversation history; `streamResponse(to:)` yields tokens as they decode.

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` (exact **0.7.3**) → product **CoreAIKit**
- Info.plist: none needed
- Entitlements: none on Mac; iPhone needs `com.apple.developer.kernel.increased-memory-limit` (the 2.7 GB cold specialization passes the default jetsam limit)
- First run downloads the model — ~2,685 MB (Mac) / ~2,685 MB (iPhone) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Measured

| | decode | prefill | numerics | size |
|---|---:|---:|---|---:|
| **iPhone 17 Pro** (A19 Pro — `PipelinedBench`, random 128-tok prompt, greedy, Release) | **22.4 tok/s** | 27.3 tok/s | **24/24 + 24/24 token-exact** vs HF fp32 (nat + oracle, the margin-clean alphabet prompt); engine ready 28.9 s cold | **2.7 GB** |
| **M4 Max** (macOS 27 — `llm-benchmark`, 512p / 1024g) | **127.6 tok/s** | 2654 tok/s | **16/16 token-exact** vs the fp32 oracle (margin-aware gate, min margin 0.925) | |
| **iPhone 17 Pro, Neural Engine** (`ios-ane-h18p/` since 2026-09-16: **6-bit palettized g8**, static graphs; p128 / g256, 60 s of trials after the thermal state returned to `fair`) | **38.5 tok/s** (flat over the minute) | 1679 tok/s | **GSM8K 200: 173 correct (86.5 %) on the phone vs 172 (86.0 %) for the fp32 checkpoint** — [`gsm8k-200-2026-09-17.json`](gsm8k-200-2026-09-17.json); 4-prompt token gate vs HF fp32: natural 24/24, chat 8/8 incl. the stop, long 16/16, the 109-token free-form answer exact for 33 tokens then a flip on a 0.168-margin step — [`gate-minicpm5-2b-ane-6bit-device.json`](gate-minicpm5-2b-ane-6bit-device.json); footprint 2.6 GB, **first load 1406 s** (ANE program build), then 0.2 s | **2.5 GB** |
| iPhone 17 Pro, Neural Engine — the **2026-09-15 4-bit g32 bundle, replaced** (Apple llm-benchmark method p512 / g1024 n5, two back-to-back runs) | 48.0 / 38.2 tok/s (per-trial 50.4 → 36.5, thermal); ~55 at p128 / g256 | 1858 / 1494 tok/s | **GSM8K 200: 131 correct (65.5 %), −20.5 pt** — [`gsm8k-200-2026-09-17.json`](gsm8k-200-2026-09-17.json); it passed the 3-prompt token gate ([`gate-minicpm5-2b-ane-device.json`](gate-minicpm5-2b-ane-device.json)) but diverges from fp32 at step 0 of a long answer ([`fidelity-probe-sky-2026-09-16.json`](fidelity-probe-sky-2026-09-16.json)); footprint 2.0 GB | 1.4 GB |

Same-day comparison of the shipped 6-bit ANE bundle and `int8/` (thermal-recovered 60 s runs, [`bench-iphone-ane-6bit-vs-gpu-2026-09-16.json`](bench-iphone-ane-6bit-vs-gpu-2026-09-16.json)): **Neural Engine 6-bit 38.5 tok/s decode, prefill 1679** vs **GPU int8 23.8 (26.0 → 21.6 as the phone reached `serious`), prefill 948** — 1.6× at 6 vs 8 weight bits. The 2026-09-15 interleaved A/B of the replaced 4-bit bundle (p128 / g256 n5, order ANE-GPU-ANE-GPU, [`bench-iphone-ane-vs-gpu-2026-09-15.json`](bench-iphone-ane-vs-gpu-2026-09-15.json)): **Neural Engine 4-bit 55.4 / 51.5 tok/s decode, 1954 / 1793 prefill** vs **GPU int8 (this repo's `int8/`, pipelined engine) 22.8 / 20.6 decode, 894 / 715 prefill** — 2.4× decode, but the ANE bundle also reads half the weight bytes per token (4-bit vs int8), so this is "what ships vs what ships", not ANE-vs-GPU at equal precision (the 1B, 8-bit vs int8, is +17 %). The p512 / g1024 row above is the Apple llm-benchmark default and is not the `PipelinedBench` protocol.

Free-run check (4 prompts × 30 greedy tokens vs fp32 HF, `verify_minicpm5.py`): **3/4 exact**; the one miss is a name at fp32 probability 0.2126 vs 0.2065 (`Emma`/`Lily`, top-2 margin 0.006) — a tie any precision may flip. The fp16 control export scores 4/4, and the **per-channel** int8 sibling of this bundle scored 2/4 with a real 0.245-margin flip (`,`→` and`), which is why this repo ships per-**block-32** scales instead (same recipe, three YAML lines; see *Quantization*).

⚠️ **iPhone context cap: prompt + generated tokens must stay under 1024.** The bundle declares a 131072
dynamic KV, and the shipped `CoreAIPipelinedEngine` caps iOS growing-KV capacity at 1024 (its guard
against the iOS compiler miscompiling growing-KV specializations at seq ≥ 2048) — so a phone
conversation truncates at absolute position 1024. Chunk or trim the history on iOS; macOS has no cap.

Three bundles were measured on the Mac before picking this one (`llm-benchmark`, 512p/1024g, [`bench-mac-llm-benchmark.json`](bench-mac-llm-benchmark.json)): int8 **per-channel** (the 1B's yaml) 25.6 tok/s, fp16 80.0, int8 **per-block-32** **127.6**. Per-channel int8 lowers to a slow dequant path on the Mac GPU; block-32 lands on the fast quantized-matmul path, 5× the per-channel decode and 1.6× fp16's, at +155 MB. On the phone the two int8 shapes decode the same (bandwidth-bound: 22.4 vs 22.7 tok/s), so block-32 wins everywhere.

Gate transcript: [`gate-minicpm5-2b.json`](gate-minicpm5-2b.json) (`cli/coreai_verify.py`, fp32 oracle, margin-aware).

### JevBench public 231 (Mac, 2026-09-24)

| easy 48 | standard 72 | hard 111 | ECE hard | p50 | p95 | hard max |
|---:|---:|---:|---:|---:|---:|---:|
| 0.979 | 0.708 | 0.459 | 0.241 | 0.14 s | 1.19 s | 1.8 s |

The benchmark's own harness ([fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench) `2fa63fa`, v1.4.0, `typesafe` adapter) ran the 231 public items against coreai-kit `adbc755` `decide-cli serve` with the `int8/` bundle, one question per request. This chat model was asked zero-shot under the kit's JSON decision prompt, with the catalog's calibration temperature 2.93 (fit on SemIf perturbations108). Accuracy per tier and the hard tier's ECE are JevBench's own scoring (argmax of the returned probabilities); p50 and p95 are per-request latency over all 231 requests, hard max the maximum over the hard tier. Latency was measured without an exclusive GPU window (contended), with this model's server running alone. JevBench's published scores (Intelligence and the rest) are chance-corrected over 534 items, sealed ones included, and are not comparable to these accuracies.

**iPhone 17 Pro (2026-09-24)**

| | easy 48 | standard 72 | hard 111 |
|---|---:|---:|---:|
| accuracy | 0.979 | 0.708 | 0.459 |
| p50 | 0.22 s | 0.17 s | 0.63 s |
| p95 | 0.25 s | 0.25 s | 3.49 s |
| p50 / p95 over | 48 rows, nominal | 72 rows, nominal | 111 rows, nominal |

The phone (iOS 27.0 24A437) received the same request bodies as the Mac run, one question per request. A headless harness app answered each with coreai-kit 0.7.1, through the call the kit's System One server makes. It ran the same `int8/` bundle as the Mac run, not `ios-ane-h18p/`. Every bundle file on the phone matched the Hub revision by hash. Every answer's argmax equals the Mac run's. p50 and p95 are the kit's time per request. A nominal row started and ended with the phone on its battery at thermal state nominal.

## Conversion

- **`llama → mistral` remap** — MiniCPM5-2B is a plain `LlamaForCausalLM` (GQA 16:2, `head_dim` 128, RoPE θ 5e6, no scaling, untied 130560-vocab head); the stock exporter has no `llama` graph family, but the Mistral builder is architecturally identical (GQA, no qkv bias, no qk-norm, explicit `head_dim`). The one-line remap that ships the 1B, untouched.
- **int8, per-block-32** — weight-only symmetric int8 with a scale per 32-wide block along the input dim (`minicpm5_int8sym_b32.yaml`: `granularity: {type: per_block, block_size: 32}`, axis resolved per module by the quantizer; no clipping; SDPA/RoPE/RMSNorm full precision) via `coreai.llm.export … --compression-config` (coreai-opt torch pre-export). The 1B's per-channel yaml (`minicpm5_int8sym.yaml`) is the `--qconfig` default of the wrapper; on the 2B it flipped one 0.245-margin greedy token that fp16 reproduces exactly and decoded 5× slower on the Mac GPU, so the 2B names the block-32 yaml in its recipe.
- **Chat EOS** — base `eos_token` is `</s>`, but the chat template ends turns with `<|im_end|>` (130073); the bundle's tokenizer `eos_token` is set to `<|im_end|>` (as Qwen ships) so generation halts cleanly. Declared in [`verify.toml`](verify.toml). Checked end to end through the engine: `llm-runner --prompt "Explain on-device AI in one sentence."` (chat template applied) thinks, answers, and stops at `<|im_end|>` — 124 tokens, 131.6 tok/s short-context on the M4 Max, on the published bundle.
- **Dynamic-shape bundle** → the pipelined engine (the iPhone path). Runs unchanged on macOS and iOS; no AOT needed — the 2.67 GB single-file bundle cold-specializes on the phone in 28.9 s, then the cache persists. That specialization needs the increased-memory entitlement and ~3 GB of free phone storage (a full phone fails it with `No space left on device`).
- **Neural Engine bundle (`ios-ane-h18p/`, 6-bit since 2026-09-16)** — Apple's stock static iOS export: `coreai.llm.export openbmb/MiniCPM5-2B --platform iOS --compression-config minicpm5_pal6_g8.yaml --max-context-length 4096` (k-means **6-bit, group 8** — the shape of Apple's own iOS preset for Qwen3-1.7B, [`conversion/minicpm5_pal6_g8.yaml`](../../conversion/minicpm5_pal6_g8.yaml); embeddings int8; static graphs `prompt_opt`/`extend` × {256, 512, 1024, 2048, 4096} × {8, 16, 64}), then `xcrun coreai-build compile … --platform iOS --preferred-compute neural-engine --architecture h18p` → 31/31 ANE regions, 2.5 GB (weights 1.70 GB). Same `llama → mistral` remap, otherwise Apple's iOS Mistral builder untouched. Loads through the unmodified Apple main `StaticShapeEngine` (`EngineFactory` auto-detects the chunked-static structure). `ios-static/` is the same export before AOT — the portable IR, compile it yourself for another chip. **Gate** on the phone (fp32 oracle, teacher-forced sweep + free-run incl. the stop, 4 prompts — the 2026-09-16 fixture adds a 109-token free-form answer, "Explain in a short paragraph why the sky is blue."): natural 24/24, chat 8/8, long 16/16, the long answer exact for 33 tokens then a flip where fp32 is nearly tied (margin 0.168; 5 of 109 steps differ) — [`gate-minicpm5-2b-ane-6bit-device.json`](gate-minicpm5-2b-ane-6bit-device.json). Not token-exact to the floor, but a different animal from the replaced 4-bit bundle, which diverges at step 0 with fp32 certain (margin 1.0) and gets the physics wrong ([`fidelity-probe-sky-2026-09-16.json`](fidelity-probe-sky-2026-09-16.json)). **Task accuracy (GSM8K test, first 200, 0-shot CoT, greedy, no-think, 640-token cap; the litertlm-convert prompt and scoring):** fp32 checkpoint 172/200, **this 6-bit bundle on the phone 173/200**, the same recipe applied to the fp32 weights on a Mac 172/200, the `int8/` recipe 171/200, the replaced 4-bit bundle 131/200 — [`gsm8k-200-2026-09-17.json`](gsm8k-200-2026-09-17.json). The token gate alone had let the 4-bit bundle ship; the task gate is now part of the recipe (the Mac simulation of a recipe predicts the phone within 0.5 pt, so it runs before any export). **First load builds the ANE programs: 1406 s on an iPhone 17 Pro**, 0.2 s afterwards (the cache lives in the app container; an iOS update invalidates it). **What else was tried (2026-09-16)**: Apple's mixed 4/8 shape (4-bit g32 + the nine most 4-bit-sensitive layers at 8-bit, [`conversion/minicpm5_mixed_4bit_8bit_g32.yaml`](../../conversion/minicpm5_mixed_4bit_8bit_g32.yaml)) runs but flips the chat turn at step 5 and the long answer at step 9 — [`gate-minicpm5-2b-ane-mixed48g32-FAIL.json`](gate-minicpm5-2b-ane-mixed48g32-FAIL.json); the 4-bit g8 preset (and the mixed shape on a g8 base) cannot be compiled for the ANE (`coreai-build` reports `ANECCompileOffline … CompilationFailure` and silently emits 0 ANE regions); an 8-bit shape is exact in an fp32 simulation of the recipe but the 8-bit 2B (2.16 GB of weights) loads and never returns its first token — the boundary on this phone is between 1.70 and 2.16 GB. An fp32 simulation of each recipe with coreai-opt's own k-means reproduced every device verdict, so these are properties of the recipes, not the chip. Details: [`../../knowledge/ane-vs-gpu-iphone-2026-09.md`](../../knowledge/ane-vs-gpu-iphone-2026-09.md) §9. The 1B sibling ships at **8-bit** palettization ([MiniCPM5-1B](../minicpm5-1b/README.md)), where it is fp32-faithful; at 2B the faithful 8-bit does not fit the ANE path.
- **Thinking** — on by default (`<think>…</think>` first); `enable_thinking=False` in the chat template for a direct answer. The think trace alone can run several hundred tokens, so cap generation generously (the kit uses 4096).
- **Not ported (yet): `openbmb/MiniCPM5-2B-DSpark`** — a 5-layer, 324M draft model (7 draft tokens per pass, `num_target_layers` 42) trained for exact pairing with this checkpoint. A dense target with an official drafter is the clean spec-decode test bed the zoo's n-gram work asked for; see [`../../knowledge/spec-decode-ngram-dense.md`](../../knowledge/spec-decode-ngram-dense.md).

## Run

In the zoo's **CoreAIChat** app (Model → "MiniCPM5 2B"), the kit's ChatDemo, or via Foundation Models. The Neural Engine bundle (`ios-ane-h18p/`) loads through the same `EngineFactory`/`CoreAILanguageModel` path on iPhone 17-class devices (h18p); it is not in the kit catalog yet:

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
python3 conversion/zoo_convert.py run  minicpm5-2b-ane    # export_minicpm5.py --hf-id openbmb/MiniCPM5-2B --ios-ane --qconfig minicpm5_pal6_g8.yaml  (stock static export, 6-bit g8 + AOT h18p neural-engine)
# device gate for the static bundle: AneGateRunner (fp32-oracle fixture, teacher-forced + free-run) — knowledge/minicpm5-1b.md §2026-09-15
```
