# CoreAI-Model-Zoo

[![CoreAIKit](https://img.shields.io/github/v/tag/john-rocky/coreai-kit?label=CoreAIKit)](https://github.com/john-rocky/coreai-kit/releases)
[![HF downloads](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjohn-rocky%2Fcoreai-assets%2Fmain%2Fbadge%2Fhf-downloads.json)](https://huggingface.co/mlboydaisuke)

Core AI is Apple's on-device ML runtime in iOS 27 / macOS 27 and the successor to Core ML. This repo is a community catalog of more than 90 open models converted to it, ready to download from Hugging Face. It is not affiliated with Apple. Each model's output was compared with the original before it shipped, and each comes with a conversion recipe. You can try them in the CoreAI Zoo app or add one to your own Swift app.

<p align="center">
  <img width="260" alt="CoreAIChat screen recording" src="https://github.com/user-attachments/assets/999dbd95-45b5-468f-b1a8-34112ee3b74d" />
</p>
<p align="center"><i>A zoo model answering on an iPhone, in a sample app from <a href="apps/">apps/</a>.</i></p>

PyTorch models are exported with Apple's `coreai-torch` into `.aimodel` bundles that run on the GPU or the Neural Engine. On the June 2026 betas, Qwen3-8B in 4-bit decoded at 94 tok/s on an M4 Max GPU, against 90 for MLX under the same protocol ([apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench)). Qwen3.5-0.8B decoded at 69.7 to 74.0 tok/s on an iPhone 17 Pro GPU.

## Try it

To try a model without building anything, get CoreAI Zoo for [iPhone](https://apps.apple.com/us/app/coreai-zoo/id6780135339) or [Mac](https://github.com/john-rocky/coreai-model-zoo/releases/latest). The iPhone app needs iOS 27, and the Mac app needs macOS 27 on Apple silicon. Both OS versions are out now, and models download inside the app.

To use one in your own Swift app, add the package `https://github.com/john-rocky/coreai-kit` in Xcode. [CoreAIKit](https://github.com/john-rocky/coreai-kit) is this catalog's Swift package, and it loads a model by its catalog id:

```swift
import CoreAIKit

guard let modelID = ModelCatalog.builtin.entry(id: "qwen3-0.6b")?.modelID else {
    throw CoreAIKitError.modelNotAvailableOnPlatform(id: "qwen3-0.6b")
}
let chat = try await ChatSession(model: modelID)
let reply = try await chat.respond(to: "What is the capital of Japan?")
print(reply)
```

The first download is about 352 MB on a Mac and 456 MB on an iPhone. See [CoreAIKit's quickstart](https://github.com/john-rocky/coreai-kit#quickstart) for the exact version and the tested Xcode build. There is more on [the first run](docs/first-model.md), including the ChatDemo sample, text to speech and Apple's FoundationModels.

## Models

Start with Qwen3 0.6B for chat or Whisper large-v3-turbo for speech to text. Many model cards include the Swift code to load the model.

| Task | Models |
|---|---|
| Chat, 2B and smaller | [Qwen3 0.6B](models/qwen3-0.6b-official/README.md), [Qwen3.5-0.8B](models/qwen3.5/README.md), [Qwen3 1.7B](models/qwen3-1.7b-official/README.md), [Qwen3.5-2B](models/qwen3.5/README.md), [Gemma 4 E2B](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI), [Gemma 4 E2B with hand-written Metal kernels](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI), [LFM2.5-1.2B-Instruct](models/lfm2.5/README.md), [LFM3-1.7B](models/lfm3-1.7b/README.md), [Granite 4.0-H 1B / 350M](models/granite-4.0-h/README.md), [RWKV-7 Goose 1.5B](models/rwkv7-goose/README.md), [MiniCPM5-1B](models/minicpm5-1b/README.md), [MiniCPM5-2B](models/minicpm5-2b/README.md), [Youtu-LLM-2B](models/youtu/README.md) |
| Chat, larger | [Gemma 4 E4B](https://huggingface.co/mlboydaisuke/gemma-4-E4B-CoreAI), [LFM2.5-2.6B](models/lfm2.5-2.6b/README.md), [LFM2.5-8B-A1B](models/lfm2.5-8b-a1b-moe/README.md), [Nanbeige4.1-3B](models/nanbeige4.1-3b/README.md), [Nanbeige4.2-3B](models/nanbeige4.2-3b/README.md), [Qwen3 4B](models/qwen3-4b-official/README.md), [Qwen3.5-4B](models/qwen3.5-4b/README.md), [Nemotron-3-Nano 4B](models/nemotron-3-nano/README.md), [Agents-A1-4B](models/agents-a1-4b/README.md), [BitCPM-8B](https://huggingface.co/mlboydaisuke/BitCPM-8B-CoreAI), [LLaDA-8B dLLM](https://huggingface.co/mlboydaisuke/LLaDA-8B-dLLM-CoreAI) |
| Chat, Mac only | [Qwen3.6-35B-A3B](models/qwen3.6/README.md), [Qwen3.6-27B](models/qwen3.6-27b/README.md), [Qwen3.8-27B](models/qwen3.8-27b/README.md), [Ornith-1.0-9B](models/ornith-1.0-9b/README.md), [GLM-4.7-Flash](models/glm-4.7-flash/README.md), [Gemma 4 12B](models/gemma4-12b/README.md), [Gemma 4 31B](models/gemma4-31b/README.md), [Bonsai 2 27B](models/bonsai2-27b/README.md), [Muse-Glimmer-30B](models/muse-glimmer-30b/README.md) |
| Vision-language | [Qwen3-VL](models/qwen3-vl/README.md), [Holo2-4B](https://huggingface.co/mlboydaisuke/Holo2-4B-CoreAI), [MiniCPM-V 4.6](https://huggingface.co/mlboydaisuke/MiniCPM-V-4.6-CoreAI), [MiniCPM-V 4.7-1B](models/minicpm-v-4.7-1b/README.md), [WeVisOne-VL-2B](models/wevisone-vl-2b/README.md), [LFM2.5-VL-450M](models/lfm2.5-vl/README.md), [North-Micro-Vision](models/north-micro-vision/README.md), [LFM2.5-VL-3B](models/lfm2.5-vl/README.md), [Gemma 4 E2B vision (VL)](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI), [BitVLA](https://huggingface.co/mlboydaisuke/BitVLA-CoreAI) |
| Documents and OCR | [Unlimited-OCR](https://huggingface.co/mlboydaisuke/Unlimited-OCR-CoreAI), [GLM-OCR](models/glm-ocr/README.md), [MinerU2.5-Pro](models/mineru/README.md), [ColModernVBERT](models/colmodernvbert/README.md) |
| Speech to text | [Whisper large-v3-turbo](models/whisper-large-v3-turbo/README.md), [Qwen3-ASR-1.7B](models/qwen3-asr/README.md), [Fun-ASR-Nano-2512](models/funasr-nano/README.md), [EmbeddingGemma 2 Transcribe](models/embeddinggemma-2-transcribe/README.md), [Parakeet-TDT-0.6B](models/parakeet/README.md), [Parakeet-TDT-0.6B-v2](models/parakeet-v2/README.md), [Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/mlboydaisuke/Nemotron-3.5-ASR-Streaming-CoreAI), [S1-mini](models/s1-mini/README.md) |
| Speaker diarization and audio understanding | [Streaming Sortformer 4-spk v2](models/sortformer-diar/README.md), [Nemotron-3-Diarization](models/nemotron-3-diarization/README.md), [Qwen2.5-Omni-3B Audio](https://huggingface.co/mlboydaisuke/Qwen2.5-Omni-3B-Audio-CoreAI) |
| Text to speech | [Kokoro-82M](https://huggingface.co/mlboydaisuke/Kokoro-82M-CoreAI), [VoxCPM-0.5B](https://huggingface.co/mlboydaisuke/VoxCPM-0.5B-CoreAI), [VoxCPM2 2B](https://huggingface.co/mlboydaisuke/VoxCPM2-CoreAI), [VibeVoice-Realtime-0.5B](models/vibevoice/README.md), [pocket-tts](models/pocket-tts/README.md), [Audio8-TTS-Preview-0.6b](models/audio8-tts/README.md) |
| Music and stems | [Stable Audio Open Small](https://huggingface.co/mlboydaisuke/Stable-Audio-Open-Small-CoreAI), [Mel-Band RoFormer](models/melband-roformer/README.md) |
| Embeddings and ranking | [EmbeddingGemma 300M](https://huggingface.co/mlboydaisuke/embeddinggemma-300m-CoreAI), [Qwen3-Embedding 0.6B](https://huggingface.co/mlboydaisuke/Qwen3-Embedding-0.6B-CoreAI), [Qwen3-Reranker 0.6B](https://huggingface.co/mlboydaisuke/Qwen3-Reranker-0.6B-CoreAI), [Granite-Embedding-97M-Multilingual-R2](models/granite-embedding-97m/README.md) |
| Routing, scoring and yes-no decisions | [decider-0.8b](models/decider-0.8b/README.md), [decider-2b-vision](models/decider-2b-vision/README.md), [clef-flash](models/clef-flash/README.md), [Kev-0.8B](models/kev-0.8b/README.md), [Kev-4B](models/kev-4b/README.md), [d1-3B](models/d1-3b/README.md), [OpenThai-SystemOne](models/openthai-systemone/README.md), [laya multilingual](models/laya-multilingual/README.md), [Julia-1](models/julia-1/README.md), [d1-omni-600M](models/d1-omni-600m/README.md), [APUS-Decision-v1-4B](models/apus-decision-v1-4b/README.md), [Qwen3.5-2B-Decision](models/qwen3.5-2b-decision/README.md), [System One scorer — Qwen3.5-4B](models/system-one-scorer-4b/README.md) |
| Classification and extraction | [Shieldstral-1.0-3B](models/shieldstral/README.md), [GLiNER2-PII](models/gliner2-pii/README.md), [GLiNER2.5-Decide](models/gliner25-decide/README.md) |
| Detection, depth, upscaling and 3D | [RF-DETR nano/small/medium/large](models/rf-detr/README.md), [RF-DETR-Seg nano→2xlarge](https://huggingface.co/mlboydaisuke/RF-DETR-CoreAI), [YOLOX-S](models/yolox/README.md), [AdcSR ×4](https://huggingface.co/mlboydaisuke/AdcSR-CoreAI), [Depth Anything 3](models/depth-anything-3/README.md), [TripoSplat](https://huggingface.co/mlboydaisuke/TripoSplat-CoreAI), [V-JEPA 2](https://huggingface.co/mlboydaisuke/VJEPA2-ViTL-SSv2-CoreAI) |
| Image and video generation | [LTX-Video 2B distilled](https://huggingface.co/mlboydaisuke/LTX-Video-2B-CoreAI), [FLUX.2 klein 4B](https://huggingface.co/mlboydaisuke/FLUX.2-klein-4B-CoreAI), [GLM-Image](models/glm-image/README.md), [Z-Image-Turbo](models/z-image-turbo/README.md), [RGBA-Image-2.1](models/rgba-image-2.1/README.md), [RGBA-Image-2.1-Turbo](models/rgba-image-2.1-turbo/README.md) |
| Time series | [TimesFM 2.5 200M](models/timesfm/README.md), [Timestral-209M](models/timestral/README.md) |

See [MODELS.md](MODELS.md) for downloads, licenses and measured speeds.

## Convert or rebuild a model

To port a new model, follow [`PORTING.md`](PORTING.md) from a Hugging Face checkpoint to a verified `.aimodel` on an iPhone. To rebuild a published model from its recipe, see [Rebuild a bundle](docs/rebuild.md).

My notes on how the runtime behaves, with measured numbers, are in [`knowledge/`](knowledge/) and on [the website](https://john-rocky.github.io/coreai-model-zoo/). [The Art of Core AI](https://john-rocky.github.io/the-art-of-core-ai/) is a free book built from them. If you hit an error, search for its text in [the error index](knowledge/coreai-error-index.md), or [open a conversion clinic issue](https://github.com/john-rocky/coreai-model-zoo/issues/new?template=conversion-clinic.yml).

The other entry points and the repository layout are in [Where to start](docs/start-here.md).

## Agent Skills

This repo ships a plugin with two skills for coding agents. An agent without plugin support can read [`AGENTS.md`](AGENTS.md) instead.

| Skill | Description |
| --- | --- |
| `reproduce-a-zoo-model` | Choose a model from `models/index.json`, rebuild its published bundle with `zoo_convert.py` and its prerequisites, and check it with `zoo_verify.py`. |
| `port-a-model-to-the-zoo` | Port a **new** model: compare it with the original at every step, choose a compression, test it on a device, and publish it. |

Apple's own [`coreai-skills`](https://github.com/apple/coreai-models) cover authoring, export and compression. You can install both.

### Install

#### Claude Code

```
/plugin marketplace add https://github.com/john-rocky/coreai-model-zoo
/plugin install coreai-zoo-skills@coreai-model-zoo
```

#### Codex CLI

```
codex plugin marketplace add https://github.com/john-rocky/coreai-model-zoo
```

then `/plugins` → `coreai-model-zoo` → `coreai-zoo-skills` → Install.

#### Gemini CLI

From a local clone of this repo:

```
gemini extensions install /path/to/coreai-model-zoo/skills
```

## Contributing

Anyone can port a model. The port stays under your own Hugging Face account with your name on it, and [`CONTRIBUTING.md`](CONTRIBUTING.md) lists what I accept. To help without code, run the Bench tab in [CoreAI Zoo](https://apps.apple.com/us/app/coreai-zoo/id6780135339) and submit the result. Your device then appears in [`BENCHMARKS.md`](BENCHMARKS.md).

## Community

### Community ports

These came from other people and stay under their own Hugging Face accounts:

| Port | Contributor | PR |
|---|---|---|
| [**pocket-tts**](models/pocket-tts/README.md): Kyutai's streaming text to speech, 7.8× real-time on an iPhone 17 Pro Max | [Rahul Rachuri](https://github.com/RahulRachuri) | [#12](https://github.com/john-rocky/coreai-model-zoo/pull/12) |
| [**Parakeet-TDT-0.6B-v2**](models/parakeet-v2/README.md): English speech to text, 175.3× real-time on an iPhone 17 Pro Max | [Rahul Rachuri](https://github.com/RahulRachuri) | [#13](https://github.com/john-rocky/coreai-model-zoo/pull/13) |
| [**Bonsai 2 27B**](models/bonsai2-27b/README.md): PrismML's ternary Qwen3.8-27B, 23.6 tok/s on an M4 Pro | [Rahul Rachuri](https://github.com/RahulRachuri) | [#37](https://github.com/john-rocky/coreai-model-zoo/pull/37) |
| [**Nanbeige4.2-3B**](models/nanbeige4.2-3b/README.md): a looped Llama, and the first community port here | [Vadim Smirnov](https://github.com/ukint-vs) | [#6](https://github.com/john-rocky/coreai-model-zoo/pull/6) |

Thank you both. Each of these ports also found a bug in this repo.

<!-- bench-thanks begin (managed by scripts/aggregate_bench.py — do not edit by hand) -->
### Community benchmarks

People who ran the Bench tab of the CoreAI Zoo app on their own devices:

- **[@jarrodpr](https://github.com/jarrodpr)** — iPhone 18 Pro Max (A20 Pro, `iPhone19,7`): [`qwen3.5-2b`](https://github.com/john-rocky/coreai-model-zoo/issues/38), [`minicpm5-1b`](https://github.com/john-rocky/coreai-model-zoo/issues/35)
- **[@Lancelotbronner](https://github.com/Lancelotbronner)** — MacBook Pro 16-inch 2024 (M4 Pro, `Mac16,7`): [`qwen3-0.6b`](https://github.com/john-rocky/coreai-model-zoo/issues/24), [`qwen3-4b`](https://github.com/john-rocky/coreai-model-zoo/issues/31), [`qwen3.5-0.8b`](https://github.com/john-rocky/coreai-model-zoo/issues/25), [`qwen3.5-2b`](https://github.com/john-rocky/coreai-model-zoo/issues/27), [`minicpm5-1b`](https://github.com/john-rocky/coreai-model-zoo/issues/33), [`minicpm5-2b`](https://github.com/john-rocky/coreai-model-zoo/issues/32), [`qwen3.6-27b`](https://github.com/john-rocky/coreai-model-zoo/issues/26), [`qwen3.8-27b`](https://github.com/john-rocky/coreai-model-zoo/issues/34)
- **[@LinkedWrong](https://github.com/LinkedWrong)** — iPhone 17 (A19, `iPhone18,3`): [`minicpm5-1b`](https://github.com/john-rocky/coreai-model-zoo/issues/30)
- **[@minato-devv](https://github.com/minato-devv)** — iPhone 17 Pro (A19 Pro, `iPhone18,1`): [`minicpm5-2b`](https://github.com/john-rocky/coreai-model-zoo/issues/29)

The results are in [`BENCHMARKS.md`](BENCHMARKS.md). To add your device, use the Bench tab of [CoreAI Zoo](https://apps.apple.com/us/app/coreai-zoo/id6780135339).
<!-- bench-thanks end -->

### Built with the zoo

- **[RF-DETR](https://github.com/roboflow/rf-detr)**, Roboflow's real-time detection and segmentation model. `RFDETR.export(format="coreai")`, added in 1.11.0, was ported from the zoo's [RF-DETR conversion](models/rf-detr/README.md) ([pull request](https://github.com/roboflow/rf-detr/pull/1521)).
- **[Archon](https://github.com/VM451/archon-swift)**, a local-first Swift SDK for AI features on Apple platforms. Its `CoreAIModelZooCatalog` reads the zoo's [`models/index.json`](models/index.json) and lists the recipes marked `verified` ([source](https://github.com/VM451/archon-swift/blob/86af1356a9f2029635e6a676385da251c40163a6/Sources/ArchonModels/CoreAIModelZooCatalog.swift)).
- **[Slurper](https://github.com/TrevorS/slurper)**, a macOS CLI that transcribes YouTube or audio files and splits them into stems. Its Core ML conversion script for Mel-Band RoFormer follows the zoo's [recipe](models/melband-roformer/README.md) ([source](https://github.com/TrevorS/slurper/blob/eab5f8b064338f1bf9cd63fc6683852929a5146d/scripts/convert_melband_roformer.py)).

To add yours, open a [showcase issue](https://github.com/john-rocky/coreai-model-zoo/issues/new?template=showcase.yml) with a name, a link and one line.

## Maintainer

I'm [Daisuke Majima (MLBoy)](https://github.com/john-rocky). My converted models live at [huggingface.co/mlboydaisuke](https://huggingface.co/mlboydaisuke). I also run [devicemark](https://devicemark.github.io/), an on-device LLM leaderboard that measures these ports.

## License

BSD-3-Clause ([`LICENSE`](LICENSE)). Re-authored model code derives from Apple's BSD-3-Clause `coreai_models` and retains its notices. Model weights follow their own licenses (see each Hugging Face repo).
