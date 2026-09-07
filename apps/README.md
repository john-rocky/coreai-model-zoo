# On-device Core AI sample apps (iOS 27 / macOS 27)

> **These are engine showcases, not starter examples.** The apps here exist for models that
> need a hand-tuned backend the generic path can't drive yet (custom Metal kernels, the
> `coreai-models` patch stack: BitCPM ternary, RWKV-7, LLaDA diffusion decode, spec-decode, …)
> — and they are the device-verification bench behind the zoo's published numbers. If you just
> want to **run a model** or copy working integration code, start from the
> [CoreAIKit examples ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples)
> (per-capability runners: chat, transcribe, vision, …) and the model's card in
> [`zoo/`](../zoo/).

SwiftUI sample apps that run models **on device** via Core AI. The chat apps are verified
greedy-exact (top-1 vs the HF eager reference) on an iPhone 17 Pro running the iOS 27 beta:

| App | Model | Decode (iPhone 17 Pro) |
|---|---|---|
| [`CoreAIChat/`](CoreAIChat/) | **Gemma 4 E2B** (text) with a GPU / ANE / ⚡ engine segment, **plus the ⚡pipelined model menu**: Qwen3.5-0.8B / Qwen3.5-2B / LFM2.5-1.2B / Granite-4.0-H-1B (one spec-parameterized `PipelinedBackend` drives them all), plus **vision modes** Qwen3-VL 2B/4B + Gemma 4 E2B VL (photo attach) | **Gemma GPU 22 tok/s** (int4-k-means kernels) / **ANE 6** (int8 chunks) / **Gemma ⚡ 32.7 chat-surface** (int4lin, PLE table as a static graph input) / **Qwen ⚡ 69.7–74.0** (benchmark; 62–67 chat-surface, int8hu ship bundle) / **LFM ⚡ ~40** / **Granite ⚡ ~31** — the ⚡ set rides Apple's `coreai-pipelined` engine, zero custom kernels |
| [`QwenChatFast/`](QwenChatFast/) | **Qwen3.5-0.8B** (hybrid linear+full attention) — static-shape loop-free decode, fused int8 Metal kernels + GPU argmax head, q16 chunked prefill, host-managed KV + SSM conv/rec state | **GPU 42.5–45.4 tok/s** decode · **147 tok/s** prefill (int8 kernels, ctx 2048; `QWEN_KIND=fp16` selects the previous fp16 path, 27.7) |

Measured numbers, bundle sizes, and per-config caveats live in the zoo cards:
[`models/gemma4-e2b/README.md`](../models/gemma4-e2b/README.md) · [`models/qwen3.5/README.md`](../models/qwen3.5/README.md).

### On-device agent — CoreAIAgent

[`CoreAIAgent/`](CoreAIAgent/) — **MiniCPM5-2B** (int8 block-32) behind FoundationModels'
`LanguageModelSession` with three Swift `Tool`s (EventKit calendar read, EventKit reminder
creation, device status), on the zoo's `ZooFMProvider` + `MiniCPMDialect`. The flow is gated on
the Mac with fixed data (`zoo-fm-gate … agent`); device run pending. See its README.

### macOS chat — CoreAIChatMac

A desktop chat app on Apple's **stock runtime** with an in-app download catalog
(`ModelCatalog.swift`) spanning two families, both loaded by one `ChatEngine` via the
`coreai-sequential` variant:

- **Official-recipe** bundles (`coreai.llm.export`, unmodified): Qwen3 0.6B/4B/8B,
  Gemma 3 4B/12B IT, Mistral 7B v0.3, gpt-oss 20B (harmony output → a "Thinking" section).
- **Zoo community ports** (engine patches / custom Metal kernels): Qwen3.6-35B-A3B,
  Qwen3.6-27B, GLM-4.7-Flash, LFM2.5-8B-A1B, Gemma 4 12B/31B.

See [`CoreAIChatMac/README.md`](CoreAIChatMac/README.md).

### Image generation

| App | Model | Image @ 4 steps |
|---|---|---|
| [`CoreAIImageGen/`](CoreAIImageGen/) | **FLUX.2 klein 4B** (text→image, **macOS**) — [HF bundle](https://huggingface.co/mlboydaisuke/FLUX.2-klein-4B-CoreAI) | macOS 1024 ≈ 17.4 s |

Runs on Apple's stock `CoreAIDiffusionPipeline` — **no model-code port**; any
`coreai.diffusion.export` bundle (FLUX.2 / SD3 / SD) drops in. It needs **no `coreai-models`
patch stack** (the diffusion runtime is unmodified), so its build is self-contained — see
[`CoreAIImageGen/README.md`](CoreAIImageGen/). The hosted FLUX.2 model is macOS-only (4B
overruns a 12 GB iPhone's memory limit); the iOS build runs smaller bundles (Stable
Diffusion 0.9B) loaded via **Local…**.

### Video generation

| App | Model | Video @ 8 steps |
|---|---|---|
| [`CoreAIVideo/`](CoreAIVideo/) | **LTX-Video 2B distilled** (text→video, **macOS**) — [card](../models/ltxvideo/README.md) · [HF bundle](https://huggingface.co/mlboydaisuke/LTX-Video-2B-CoreAI) | macOS 512×768 × 49f ≈ 14 s |

The zoo's first **video** app — a SwiftUI front-end over a resident Python backend that runs all
three nets (T5 + DiT + causal video VAE) as Core AI bundles; only LTX's FlowMatch sampler is on
host. Mac-host pattern (like `TripoSplatMac`): the app launches the proven pipeline and plays the
result. See [`CoreAIVideo/README.md`](CoreAIVideo/).

### 3D generation

| App | Model | Gen @ 20 steps |
|---|---|---|
| [`TripoSplatMac/`](TripoSplatMac/) | **TripoSplat** (single image → 3D Gaussian splats, **macOS**) — [HF bundle](https://huggingface.co/mlboydaisuke/TripoSplat-CoreAI) · [conversion](../conversion/triposplat) | macOS GPU ≈ 1 min |

The zoo's first **3D** app — a SwiftUI front-end over the Python backend that runs the four heavy
nets (DINOv3 ViT-H + Flux2-VAE + flow-matching DiT + Gaussian decoder) as Core AI bundles; octree
sampling stays on host. Same Mac-host pattern as `CoreAIVideo`. In-app preview is a SceneKit point
cloud; export the `.splat` and open it in [MetalSplatter](https://github.com/scier/MetalSplatter)
for true splat rendering — including on iPhone / iPad / Vision Pro, since generation is macOS-only.
See [`TripoSplatMac/README.md`](TripoSplatMac/).

### Audio (understanding · transcription · speech)

| App | Models | Device |
|---|---|---|
| [`coreai-audio/`](coreai-audio/) | **Understand:** Qwen2.5-Omni-3B Thinker · **Transcribe:** Whisper large-v3-turbo *·* Qwen3-ASR-1.7B *·* Parakeet-TDT-0.6B (selectable) · **Voice:** VoxCPM-0.5B · **Speak:** Kokoro-82M | iPhone 17 Pro + M4 Max |

Four tabs, all on-device. **Understand** → *"what do you hear?"*: a Whisper-style audio encoder (run
once per clip) feeds the Qwen2.5-3B decoder on the ⚡pipelined engine; the audio embeds ride one
`EngineOptions.staticInputBuffers` input (`<|AUDIO|>` ids rewritten to `vocab+slot`, no rope shift —
TMRoPE collapses to 1-D), and a Swift vDSP log-mel front end (bit-exact with the HF extractor, cos
1.0) does the waveform→features step. iPhone uses the **AOT** decoder (`.aimodelc`, h18p) so the
3.9 GB graph dodges the on-device JIT jetsam. **Transcribe** turns speech → text with a choice of three
ASR models: Whisper large-v3-turbo (stock-runtime fixed-128 graph via `KitWhisperModel`, 100 langs,
auto-detect), Qwen3-ASR-1.7B (the zoo's first ASR, via `KitASRModel`, 52 langs), or Parakeet-TDT-0.6B
(the zoo's first transducer / TDT, via `KitParakeetModel`, 25 EU langs, iPhone 47.9× real-time).
**Voice** /
**Speak** are diffusion (VoxCPM) and StyleTTS2 (Kokoro) text-to-speech. See
[`coreai-audio/README.md`](coreai-audio/).

### Depth estimation

| App | Model | Device |
|---|---|---|
| [`CoreAIDepth/`](CoreAIDepth/) | **Depth Anything 3** (monocular depth, the zoo's first depth model) — [HF bundle](https://huggingface.co/mlboydaisuke/Depth-Anything-3-CoreAI) | iPhone 17 Pro + M4 Max |

Two modes — a still photo from the library, or live camera depth at a few FPS. Reuses CoreAIKit's
`DepthEstimator` (DINOv2 ViT-S + DPT head, 504² monocular, small fp16 ≈ 54 MB, engine cos 1.000000
vs torch, ~15 ms/frame on an M4 Max GPU). Depth is rendered with the DA3 colormap (inverse-depth,
percentile-normalized, Spectral — far = red, near = blue). See [`CoreAIDepth/README.md`](CoreAIDepth/)
and [`models/depth-anything-3/README.md`](../models/depth-anything-3/README.md).

### Vision-language (VLM)

| App | Model | Device |
|---|---|---|
| [`MiniCPMVisualIntel/`](MiniCPMVisualIntel/) | **MiniCPM-V 4.6** (sub-2B VLM — photo + question → answer) — [HF bundle](https://huggingface.co/mlboydaisuke/MiniCPM-V-4.6-CoreAI) | iPhone 17 Pro |

Photo-attach visual Q&A on the strongest tiny VLM. (Qwen3-VL and Gemma 4 E2B VL are also reachable
from CoreAIChat's vision modes.)

### Document OCR

| App | Model | Device |
|---|---|---|
| [`CoreAIOCR/`](CoreAIOCR/) | **Unlimited-OCR** (3B-A0.5B MoE — document → structured markdown: tables → HTML, formulas → LaTeX) — [HF bundle](https://huggingface.co/mlboydaisuke/Unlimited-OCR-CoreAI) | macOS |

Drop a document image, get markdown. Stock `coreai.runtime`, no engine patch (DeepEncoder vision
`.aimodel` + an `inputs_embeds`-driven decoder). See [`CoreAIOCR/README.md`](CoreAIOCR/).

### Segmentation

| App | Model | Device |
|---|---|---|
| [`CoreAISegment/`](CoreAISegment/) | **SAM 3** (text-prompt image segmentation — type `the red car`) — Apple stock `CoreAIImageSegmenter` | iPhone + macOS |

Pick an image, type what to segment. Stock runtime, no patch. (The zoo's own **RF-DETR-Seg** runs via
the CoreAIKit [`DetectCamera`](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DetectCamera)
example.) See [`CoreAISegment/README.md`](CoreAISegment/).

### Super-resolution

| App | Model | Device |
|---|---|---|
| [`CoreAIUpscale/`](CoreAIUpscale/) | **AdcSR ×4** (one-step diffusion-GAN SR, CVPR 2025 — the zoo's first SR) — [HF bundle](https://huggingface.co/mlboydaisuke/AdcSR-CoreAI) | iPhone + macOS |

Pick a photo, get a 4× sharper version on-device — one step (pruned SD2.1 UNet + half VAE decoder), no
iterative denoising. See [`CoreAIUpscale/README.md`](CoreAIUpscale/).

### Transcription (standalone)

| App | Model | Device |
|---|---|---|
| [`CoreAITranscribe/`](CoreAITranscribe/) | **Whisper large-v3-turbo** (speech → text, 100 languages) — [HF bundle](https://huggingface.co/mlboydaisuke/whisper-large-v3-turbo-CoreAI-official) | iPhone + macOS |

A single-model transcriber (also available as a tab in [`coreai-audio`](coreai-audio/) alongside
Qwen3-ASR and Parakeet). Stock runtime, one 30 s window. See [`CoreAITranscribe/README.md`](CoreAITranscribe/).

## Model delivery

On first launch each app offers an **in-app download** of the published `.aimodel` set from the
Hugging Face repos (editable URL field, defaults to the zoo's repos). Files stream into a staging
directory and a bundle is renamed into place only when ALL of its files are complete — Core AI's
specialization cache is content-keyed, and a partially-present bundle poisons it
([`knowledge/swift-runtime.md`](../knowledge/swift-runtime.md)). Weights are NOT bundled into the
.app (GB-class; Apple's guidance for large models is download-then-specialize off the interactive
flow — WWDC26 session 326). For development iteration you can still sideload with
`xcrun devicectl device copy to --domain-type appDataContainer …` — the apps check the same
locations. Shared implementation: [`AppShared/ModelDownloader.swift`](AppShared/ModelDownloader.swift).

## Build

Requires the **Xcode 27 beta** and [xcodegen](https://github.com/yonaskolb/XcodeGen).

```bash
# 1. The apps build against Apple's `CoreAIShared` Swift library (AIModel / InferenceFunction /
#    NDArray) and, for CoreAIChat's Qwen ⚡pipelined mode, the `CoreAILM` engine stack.
#    Clone coreai-models AT THIS REPO'S ROOT and apply the patch stack IN ORDER
#    (CoreAIShared product export → pipelined-engine extra states for SSM conv/rec caches →
#    pipelined-engine per-token inputs for host-gathered tables like Gemma 4's PLE rows →
#    pipelined-engine static inputs for in-graph gather tables bound as constant buffers;
#    the Qwen apps only need the first two, but the stack is additive — apply it whole):
git clone https://github.com/apple/coreai-models
git -C coreai-models apply ../apps/coreai-shared-product.patch \
                           ../apps/coreai-pipelined-extra-states.patch \
                           ../apps/coreai-pipelined-per-token-inputs.patch \
                           ../apps/coreai-pipelined-static-inputs.patch

# 2. tokenizer.json is not committed (tens of MB). Fetch it from the upstream model repo into
#    the app's Resources/tokenizer/ (tokenizer_config + chat template are already there):
#      CoreAIChat:   https://huggingface.co/google/gemma-4-E2B-it  (accept the Gemma terms)
#      QwenChatFast: https://huggingface.co/Qwen/Qwen3.5-0.8B

# 3. Generate + build (set DEVELOPMENT_TEAM in project.yml, or pick a team in Xcode > Signing):
export DEVELOPER_DIR=/path/to/Xcode-beta.app/Contents/Developer
cd apps/CoreAIChat            # or apps/QwenChatFast
xcodegen generate
xcodebuild -project CoreAIChat.xcodeproj -scheme CoreAIChat -configuration Release \
  -sdk iphoneos -destination 'generic/platform=iOS' -derivedDataPath build \
  -allowProvisioningUpdates build
xcrun devicectl device install app --device <udid> \
  build/Build/Products/Release-iphoneos/CoreAIChat.app
```

⚠️ Benchmark **Release** builds only — Debug-build tok/s under-reads by 2–3× (host-side work
dominates in Debug; e.g. 10.3 vs 30.4 tok/s on the same artifact).

## Engine notes

- The engines are plain Swift over `CoreAIShared`: gemma = `Gemma4ChatEngine` (GPU
  metal-kernel monolith / ANE 6-chunk host-cache backends behind a `GemmaMode` picker; mode
  switch frees one model set before loading the other), qwen = `FastEngine` (static-shape
  single-step graph; the 4 state arrays are host-managed, and NDArray views are function-local
  `inout` — class-property state trips MutableViews lifetime checks).
- CoreAIChat's **⚡ modes** are different: one spec-parameterized `PipelinedBackend` hands the
  whole generation to Apple's `coreai-pipelined` engine (`EngineFactory` over the
  `gpu-pipelined/` LanguageBundle — async non-blocking encode, on-GPU argmax, on-device KV
  growth), so it consumes a token stream instead of stepping per token. Contract for the S=1
  bundles: set `COREAI_CHUNK_THRESHOLD=1` before engine creation (prefill = pipelined S=1
  steps) and never call `engine.warmup()` (it warms query length 256, which the static `[1,1]`
  graph rejects — a 1-token generate after load is the warmup). The **Gemma ⚡** spec
  additionally binds the two PLE table files (from the shared `gemma4_gather_raw/` download)
  as owned `storageModeShared` MTLBuffers via `EngineOptions.staticInputBuffers` (the
  static-inputs engine patch; owned beats mmap on iOS — buffer-backing economics in
  [`../knowledge/pipelined-engine.md`](../knowledge/pipelined-engine.md)), and stops on
  gemma's `<end_of_turn>` (106) on top of the tokenizer's eos.
- Headless device-probe hooks ride env vars (`GEMMA_*` / `QWEN_*` — see `CoreAIChatApp.swift` and
  the engine sources); launching from the home screen uses the published release configuration.
- Engine-first workflow: validate the graph on the Mac CLI first (`swift run coreai-run`, raw
  token ids in/out vs a conversion oracle), UI second — see [`../swift/`](../swift/).
- The conversion side (how the `.aimodel` bundles are produced) is
  [`../conversion/`](../conversion/); the gotchas it hit are in [`../knowledge/`](../knowledge/).
