# CoreAI-Model-Zoo

[![CoreAIKit](https://img.shields.io/github/v/tag/john-rocky/coreai-kit?label=CoreAIKit)](https://github.com/john-rocky/coreai-kit/releases)
[![HF downloads](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjohn-rocky%2Fcoreai-assets%2Fmain%2Fbadge%2Fhf-downloads.json)](https://huggingface.co/mlboydaisuke)
[![CI](https://github.com/john-rocky/coreai-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/john-rocky/coreai-kit/actions/workflows/ci.yml)
[![Nightly device gate](https://github.com/john-rocky/coreai-kit/actions/workflows/nightly-gate.yml/badge.svg)](https://github.com/john-rocky/coreai-kit/actions/workflows/nightly-gate.yml)
[![GA validation](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjohn-rocky%2Fcoreai-assets%2Fmain%2Fbadge%2Fga-validation.json)](models/_INVENTORY.md#tier-0-load-check)

**Converted models + conversion recipes** for Apple **Core AI** (`.aimodel`, iOS 27 / macOS 27):
every model here is downloadable, gated against the original model before it ships, and carries
the recipe that produced it in [`models/<model>/recipe.toml`](models/) — rerun it
(`zoo_convert.py run <name>`), check the published bundle against its source model
(`zoo_verify.py`), or adapt it for your own port. **Don't take the gates on faith — they are run
here, and everything needed to re-run them is published.** Each card states what was measured on
which hardware (iPhone tier is device-measured; the large models are Mac-only and say so) and how
strong that model's parity proof is, because it differs per model.
Where the shipped configuration could not be recovered from the repository, the recipe says so
rather than guessing. Model ports are open to everyone — the catalog serves
community ports from the contributor's own Hugging Face namespace, credited by name
([who](#community-ports)).
Successor to [`CoreML-Models`](https://github.com/john-rocky/CoreML-Models).

**The `from_pretrained` of Core AI** — one line, any zoo model, via
[**CoreAIKit**](https://github.com/john-rocky/coreai-kit) (SPM):

```swift
let chat = try await ChatSession(catalog: "qwen3.5-2b")   // downloads once, then cached
let reply = try await chat.respond(to: "What can you do, offline?")
```

Same gesture for every capability — `KitTranscriber(catalog: "whisper-large-v3-turbo")` is
speech-to-text in 3 lines ([card](models/whisper-large-v3-turbo/README.md)). Each model's card carries the
complete copy-paste snippet and its integration checklist. Every row below also links a
ready-to-build app — in this repo's [`apps/`](apps) or a
[CoreAIKit example](https://github.com/john-rocky/coreai-kit/tree/main/Examples) (marked ↗).

**When Apple's FoundationModels built-in model isn't enough, keep your session code and
swap the model — one line.**
`LanguageModelSession(model: try await KitLanguageModel(model: .qwen3_0_6B))` gives you the
same system session — `Tool` calling, `@Generable` guided generation, transcripts — backed
by any zoo chat model
([how](https://github.com/john-rocky/coreai-kit#when-foundationmodels-isnt-enough)).
Zero-dependency alternative: every bundle loads with Apple's own
`CoreAILanguageModel(resourcesAt:)` as-is; this repo's
[`ZooFMProvider`](swift/Sources/ZooFMProvider) adds streaming tool calling on top (incl. LFM's
native dialect) — engineering notes in [`knowledge/fm-provider.md`](knowledge/fm-provider.md).

## Quickstart — running a model on your device

New here? You'll have a model answering on-device in a few minutes (needs Xcode 27 + a Mac or an
iPhone/iPad on iOS/macOS 27):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/ChatDemo/ChatDemo.xcodeproj   # Run, then pick a model in the picker
```

The app downloads the model on first pick (cached after), then runs it fully offline. **Start
small for the fastest first run:** `Qwen3-0.6B` (454 MB) or `Qwen3.5-2B` on iPhone;
any of the Mac-only rows on a Mac. Prefer the terminal? `swift run chat-cli --model qwen3-0.6b
--prompt "Hello"` from `Examples/ChatDemo`. To drop a model into **your own** app, copy the
snippet from that model's card — it's the same `catalog:` one-liner shown above.

## Rebuild a bundle — the conversion recipes

Every published bundle carries the configuration that produced it in
`models/<model>/recipe.toml`, and one command runs it:

```bash
python3 conversion/zoo_convert.py list                    # what can be rebuilt
python3 conversion/zoo_convert.py show  qwen3.5-0.8b      # the command + everything it needs
python3 conversion/zoo_convert.py run   qwen3.5-0.8b --dry-run
python3 conversion/zoo_convert.py run   qwen3.5-0.8b      # do it
```

`show` prints four kinds of prerequisite, and the run-time ones matter as much as the rest — an
export that skips them still succeeds, and the bundle then misbehaves inside the app:

| line | means |
|---|---|
| `overlay` | the interpreter needs `coreai_models` with [`conversion/overlay/`](conversion/overlay/) applied — `zoo_convert.py doctor` checks it |
| `needs` | something the export cannot run without: a checkpoint download, a gather-table dump, a package patch |
| `runtime` | what the **app** needs to run the result: an engine patch from [`apps/`](apps/), an environment variable such as `COREAI_CHUNK_THRESHOLD=1` |
| `device` | the AOT compile step for the iPhone bundle |
| `uv` | this script declares its own dependencies — no venv, no overlay, nothing to install |

Ports whose exporter is self-contained need no setup at all:

```bash
uv run conversion/export_da3.py --variant small --dtype float16 --res 504
```

Paths never assume a machine: `python3 conversion/_paths.py` prints where downloads, exports and
the Hugging Face cache resolve, and `ZOO_WORK_ROOT` / `ZOO_EXPORTS` / `ZOO_CODE_ROOT` /
`HF_HUB_CACHE` move them.

**A recipe marked `unverified` refuses to run** without `--force`. It means the repository does
not record which configuration produced the published bundle, and it prints the exact question
it cannot answer. Running it anyway yields *a* bundle, not *the* bundle.

**Do not expect a checksum match.** Conversion is not byte-deterministic here: the same recipe
run twice on the same machine produces bundles that differ from each other (measured:
`main.mlirb` by 7 bytes, `main.hash` entirely). A rebuild is judged by the gates the export
runs — and by checking the published artifact itself:

```bash
python3 conversion/zoo_verify.py mlboydaisuke/Gemma-4-12B-CoreAI     # one repo
python3 conversion/zoo_verify.py --all --json models/_VERIFY.json    # the whole catalog, minutes
```

That compares a bundle's tokenizer, chat template, context length and declared precision against
the source model it names in its own `metadata.json` — no oracle, no device, no weights.

**The numerical check is a different command.** `zoo_verify.py` checks that a bundle is described
correctly; it does not check that the bundle still computes the right thing. That is
`conversion/coreai_gate.py`, which rebuilds the reference model in fp32 and compares a greedy
decode token for token:

```bash
python3 conversion/coreai_gate.py <bundle-dir> Qwen/Qwen3.5-2B --revision <sha> \
    --transcript models/qwen3.5-2b/gate.json
```

It runs outside this working tree — point `--runner` / `ZOO_LLM_RUNNER` at your `llm-runner` and
`--python` / `ZOO_CONVERT_PYTHON` at an interpreter that has the export overlay, and it tells you
which one is missing rather than failing obscurely. It currently covers the decode
architectures in `--arch` (the oracle is a per-architecture transcription of that model's export,
so it grows one port at a time, not all at once).

`--transcript` is the part worth publishing: the pinned revision, the exact `input_ids`, both
sides' generated tokens, the tie margins, and the verdict. **Rebuilding the oracle is expensive;
re-running the engine side against a published transcript is not** — it needs the bundle,
`llm-runner`, and the recorded `input_ids`, and the output must match `engine.gen_text`.

New ports publish one. For models ported before the flag existed,
`conversion/backfill_gate_transcripts.py` prints what can still be gated and, with `--run`, gates
it — preferring the **published** bundle out of the CoreAIKit cache over a local export, so the
transcript describes the bytes apps download at the revision the catalog pins. Nothing
reconstructs a transcript from a card's prose after the fact: a model that cannot be re-gated
keeps a card that says what was gated, without a file claiming to prove it.

Results land in [`models/_INVENTORY.md`](models/_INVENTORY.md); [`models/index.json`](models/index.json)
is the same catalog machine-readable, which is where an agent should start. Each recipe entry
there carries `status` (does the repository record what produced this bundle) and
`gate_transcript` (is the numerical check against the original published, and where) — two
different questions, kept as two fields so neither has to be inferred from prose.

## Models

### Community ports

Ported by people outside this repo, published under their own Hugging Face namespace, and
carrying their name on the card, on the row below, and in the release notes:

| Port | Contributor | PR |
|---|---|---|
| [**pocket-tts**](models/pocket-tts/README.md) — Kyutai's streaming TTS; the zoo's **first Kyutai model** and **first Mimi conversion**, 7.8× real-time on iPhone 17 Pro Max at 169 MB | [Rahul Rachuri](https://github.com/RahulRachuri) | [#12](https://github.com/john-rocky/coreai-model-zoo/pull/12) |
| [**Parakeet-TDT-0.6B-v2**](models/parakeet-v2/README.md) — the English-only sibling of the v3 port; 175.3× real-time over 66 minutes of audio on iPhone 17 Pro Max, 20376/20376 tokens exact | [Rahul Rachuri](https://github.com/RahulRachuri) | [#13](https://github.com/john-rocky/coreai-model-zoo/pull/13) |
| [**Nanbeige4.2-3B**](models/nanbeige4.2-3b/README.md) — looped Llama, 22 physical blocks × 2 passes; the zoo's **first community port** | [Vadim Smirnov](https://github.com/ukint-vs) | [#6](https://github.com/john-rocky/coreai-model-zoo/pull/6) |

Thank you both — each of these arrived gated, and each one found a bug in this repo on the way
in. [`CONTRIBUTING.md`](CONTRIBUTING.md) is the path to the next row.

| Model | Download (`.aimodel`) | Run in app | License |
|---|---|---|---|
| [**Qwen3.5-0.8B**](models/qwen3.5/README.md) | [🤗 qwen3.5-0.8B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Apache-2.0 |
| [**Qwen3.5-2B**](models/qwen3.5/README.md) | [🤗 qwen3.5-2B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-2B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Apache-2.0 |
| [**Qwen3.6-35B-A3B**](models/qwen3.6/README.md) (MoE, Mac-only) | [🤗 Qwen3.6-35B-A3B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3.6-35B-A3B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Apache-2.0 |
| [**Qwen3.6-27B**](models/qwen3.6-27b/README.md) (dense, Mac-only) | [🤗 Qwen3.6-27B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3.6-27B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Apache-2.0 |
| [**Qwen3.8-27B**](models/qwen3.8-27b/README.md) (dense **VLM**, Mac-only — the Qwen3.8 generation's only open compact model, text ported on release day + full vision path: fp16 ViT tower, embeddings-input mRoPE decoder, S=16 chunked prefill at 86 tok/s) | [🤗 Qwen3.8-27B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3.8-27B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Apache-2.0 |
| [**Ornith-1.0-9B**](models/ornith-1.0-9b/README.md) (zoo's first **agentic-coding** model — self-scaffolding coder, Qwen3.5 arch, DeepReinforce; Mac-only, 48 tok/s int8 / 59 int4) | [🤗 Ornith-1.0-9B-CoreAI](https://huggingface.co/mlboydaisuke/Ornith-1.0-9B-CoreAI) | [CoreAIChatMac](apps/CoreAIChatMac) | MIT |
| [**GLM-4.7-Flash**](models/glm-4.7-flash/README.md) (MoE + MLA, Mac-only — zoo's first MLA) | [🤗 GLM-4.7-Flash-CoreAI](https://huggingface.co/mlboydaisuke/GLM-4.7-Flash-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | MIT |
| 🔧 **Gemma 4 E2B** (text, incl. official-QAT int4) | [🤗 gemma-4-E2B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Gemma |
| **Gemma 4 E2B ⚡raw-Metal** (hand-written mixed-bit int2/int4 kernels, ~55 tok/s on iPhone 17 Pro = LiteRT-LM parity, lossless) | `raw-metal/` in [🤗 gemma-4-E2B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI) | [CoreAIChat](apps/CoreAIChat) | Gemma |
| 🔧 **Gemma 4 E4B** (text, official-QAT int4) | [🤗 gemma-4-E4B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E4B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Gemma |
| 🔧 [**Gemma 4 12B**](models/gemma4-12b/README.md) (dense, Mac-only — custom flash-decode kernel) | [🤗 Gemma-4-12B-CoreAI](https://huggingface.co/mlboydaisuke/Gemma-4-12B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Gemma |
| 🔧 [**Gemma 4 31B**](models/gemma4-31b/README.md) (dense, Mac-only — custom flash-decode kernel) | [🤗 Gemma-4-31B-CoreAI](https://huggingface.co/mlboydaisuke/Gemma-4-31B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Gemma |
| [**LFM2.5-1.2B-Instruct**](models/lfm2.5/README.md) | [🤗 LFM2.5-1.2B-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-1.2B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | LFM Open License v1.0 |
| [**LFM2.5-2.6B**](models/lfm2.5-2.6b/README.md) (reasoning; int4lin clears the cliff this family usually hits) | [🤗 LFM2.5-2.6B-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-2.6B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | LFM Open License v1.0 |
| [**LFM2.5-8B-A1B**](models/lfm2.5-8b-a1b-moe/README.md) (MoE, custom `gather_qmm` kernel — first iPhone MoE) | [🤗 LFM2.5-8B-A1B-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-8B-A1B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | LFM Open License v1.0 |
| [**Granite 4.0-H 1B / 350M**](models/granite-4.0-h/README.md) | [🤗 granite-4.0-h-CoreAI](https://huggingface.co/mlboydaisuke/granite-4.0-h-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) (1B) · 350M: [CoreAIChat](apps/CoreAIChat) | Apache-2.0 |
| [**Nanbeige4.1-3B**](models/nanbeige4.1-3b/README.md) (dense reasoning/agentic, iPhone — 32B-class @ 3.93B) | [🤗 Nanbeige4.1-3B-CoreAI](https://huggingface.co/mlboydaisuke/Nanbeige4.1-3B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Apache-2.0 |
| [**Nanbeige4.2-3B**](models/nanbeige4.2-3b/README.md) (looped Llama: 22 physical blocks × 2 passes, 44 KV layers; int8 46.4 tok/s on M4 Max; port by [Vadim Smirnov](https://github.com/ukint-vs)) | [🤗 Nanbeige4.2-3B-CoreAI](https://huggingface.co/ukint-vs/Nanbeige4.2-3B-CoreAI/tree/5864ec7a5581940958e58354a6b6c46c8f06891e) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Apache-2.0 |
| [**MiniCPM5-1B**](models/minicpm5-1b/README.md) (1B-class on-device LLM, hybrid Think/No-Think, 128K, OpenBMB) | [🤗 MiniCPM5-1B-CoreAI](https://huggingface.co/mlboydaisuke/MiniCPM5-1B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Apache-2.0 |
| [**Youtu-LLM-2B**](models/youtu/README.md) (dense **MLA** — zoo's first **iPhone** MLA & first **dense** MLA; DeepSeek-V2-style latent-KV attention at 1.96B with an absorbed flash-decode kernel, reasoning + agentic; Tencent) | [🤗 Youtu-LLM-2B-CoreAI](https://huggingface.co/mlboydaisuke/Youtu-LLM-2B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Other (youtu-llm) |
| **BitCPM-8B** (zoo's first **1.58-bit ternary** LLM — every weight is {-1,0,+1}; MiniCPM4-8B arch, custom 2-bit packed-GEMM Metal kernel; 8B running in ~2.1 GB on iPhone GPU; OpenBMB) | [🤗 BitCPM-8B-CoreAI](https://huggingface.co/mlboydaisuke/BitCPM-8B-CoreAI) | [CoreAIChat](apps/CoreAIChat) ‡ | Apache-2.0 |
| **LLaDA-8B dLLM** (zoo's first **diffusion LLM** — masked-diffusion decode: fills a canvas of `[MASK]` tokens **in parallel**, not left-to-right AR; bidirectional LLaMA-dense 8B, [d3LLM](https://huggingface.co/d3LLM/d3LLM_LLaDA)-distilled; int4 ~4.9 GB, Mac) | [🤗 LLaDA-8B-dLLM-CoreAI](https://huggingface.co/mlboydaisuke/LLaDA-8B-dLLM-CoreAI) | [DiffuseChat ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DiffuseChat) | Other |
| **BitVLA** (zoo's first **Vision-Language-Action / robotics** model + first ternary multimodal — image+instruction → 7-DoF robot action; **1.58-bit** BitNet-2B LLM + BitSigLIP vision, shared ternary kernel; runs on iPhone GPU; arXiv 2506.07530) | [🤗 BitVLA-CoreAI](https://huggingface.co/mlboydaisuke/BitVLA-CoreAI) | [CoreAIChat](apps/CoreAIChat) ‡ | MIT |
| [**Qwen3-VL**](models/qwen3-vl/README.md) (vision-language) | [🤗 2B](https://huggingface.co/mlboydaisuke/Qwen3-VL-2B-CoreAI) · [4B](https://huggingface.co/mlboydaisuke/Qwen3-VL-4B-CoreAI) · [8B](https://huggingface.co/mlboydaisuke/Qwen3-VL-8B-CoreAI) | [VLChat ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/VLChat) | Apache-2.0 |
| **Holo2-4B** (GUI-grounding / computer-use VLM — screenshot + instruction → click coordinates; Qwen3-VL-4B backbone, H Company; zoo's first computer-use model) | [🤗 Holo2-4B-CoreAI](https://huggingface.co/mlboydaisuke/Holo2-4B-CoreAI) | [VLChat ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/VLChat) | Apache-2.0 |
| **MiniCPM-V 4.6** (vision-language, sub-2B — strongest tiny VLM) | [🤗 MiniCPM-V-4.6-CoreAI](https://huggingface.co/mlboydaisuke/MiniCPM-V-4.6-CoreAI) | [VLChat ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/VLChat) | Apache-2.0 |
| [**LFM2.5-VL-450M**](models/lfm2.5-vl/README.md) (vision-language, **658 MB** — the smallest VLM here; SigLIP2-NaFlex + the shipped LFM2 decoder, 112 tok/s on iPhone 17 Pro) | [🤗 LFM2.5-VL-450M-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-VL-450M-CoreAI) | [VLChat ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/VLChat) | LFM Open License v1.0 |
| [**North-Micro-Vision**](models/north-micro-vision/README.md) (vision-language, 2.4B, **11 languages incl. Japanese** — Cohere; iPhone 24/24 token-exact vs fp32) | [🤗 North-Micro-Vision-CoreAI](https://huggingface.co/mlboydaisuke/North-Micro-Vision-CoreAI) | [VLChat ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/VLChat) | Apache-2.0 |
| [**LFM2.5-VL-3B**](models/lfm2.5-vl/README.md) (vision-language, Mac-only — the detail tier of the same family; int4 costs it nothing, unlike the 450M) | [🤗 LFM2.5-VL-3B-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-VL-3B-CoreAI) | [VLChat ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/VLChat) | LFM Open License v1.0 |
| [**Shieldstral-1.0-3B**](models/shieldstral/README.md) (**safety classifier** — you write the policy in plain language at call time; ships as ONE forward → `probs[1,2]`, no decode loop; 12 languages, **123.6 ms/verdict** on a Mac and **371.9 ms on an iPhone 17 Pro**, 9/9 vs fp32 at int4 on both) | [🤗 Shieldstral-CoreAI](https://huggingface.co/mlboydaisuke/Shieldstral-CoreAI) | — | Apache-2.0 |
| 🔧 **Gemma 4 E2B vision (VL)** (image+text) | `vl/` in [🤗 gemma-4-E2B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI) | [CoreAIChat](apps/CoreAIChat) | Gemma |
| **Unlimited-OCR** (document OCR → markdown: tables→HTML, formulas→LaTeX; zoo's first doc-OCR — **stock runtime, no patch**, flat-latency R-SWA) | [🤗 Unlimited-OCR-CoreAI](https://huggingface.co/mlboydaisuke/Unlimited-OCR-CoreAI) | [ReadDoc ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ReadDoc) | MIT |
| [**GLM-OCR**](models/glm-ocr/README.md) (document OCR → Markdown; GLM-4.V small **0.9B**, single-pass, tables→Markdown; iPhone + Mac, ~4 s/page) | [🤗 GLM-OCR-CoreAI](https://huggingface.co/mlboydaisuke/GLM-OCR-CoreAI) | [ReadDoc ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ReadDoc) | MIT |
| [**MinerU2.5-Pro**](models/mineru/README.md) (whole-page document parsing → structured Markdown; zoo's first **whole-page auto-structuring** — 2-stage layout + per-region recognition in one stock Qwen2-VL **1.2B**, tables→`<table>` HTML; Mac) | [🤗 MinerU2.5-Pro-CoreAI](https://huggingface.co/mlboydaisuke/MinerU2.5-Pro-CoreAI) | [ReadDoc ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ReadDoc) | Apache-2.0 |
| **Qwen2.5-Omni-3B Audio** (audio *understanding* — describes sounds, not a transcript; iPhone + Mac, zoo's first audio model) | [🤗 Qwen2.5-Omni-3B-Audio-CoreAI](https://huggingface.co/mlboydaisuke/Qwen2.5-Omni-3B-Audio-CoreAI) | [AudioChat ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/AudioChat) | Apache-2.0 |
| [**Whisper large-v3-turbo**](models/whisper-large-v3-turbo/README.md) (speech→text — 100 languages, auto-detect; stock runtime, iPhone AOT + Mac) | [🤗 whisper-large-v3-turbo-CoreAI-official](https://huggingface.co/mlboydaisuke/whisper-large-v3-turbo-CoreAI-official) | [Transcribe ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Transcribe) | MIT |
| [**Qwen3-ASR-1.7B**](models/qwen3-asr/README.md) (speech→text — the zoo's first ASR; AuT encoder + Qwen3 decoder, 52 languages; iPhone + Mac) | [🤗 Qwen3-ASR-1.7B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3-ASR-1.7B-CoreAI) | [Transcribe ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Transcribe) | Apache-2.0 |
| [**Parakeet-TDT-0.6B**](models/parakeet/README.md) (speech→text — zoo's first **transducer / TDT (RNN-T)**; NVIDIA FastConformer + LSTM predictor + joint, 3 graphs + host greedy loop, 25 EU languages; iPhone 47.9× real-time) | [🤗 Parakeet-TDT-0.6B-CoreAI](https://huggingface.co/mlboydaisuke/Parakeet-TDT-0.6B-CoreAI) | [Transcribe ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Transcribe) | CC-BY-4.0 |
| [**Parakeet-TDT-0.6B-v2**](models/parakeet-v2/README.md) (speech→text — the **English-only** sibling of the v3 port; same FastConformer TDT, vocab 1025; **iPhone 17 Pro Max 175.3× real-time** over 66 min of audio at 20376/20376 tokens exact; port by [Rahul Rachuri](https://github.com/RahulRachuri)) | [🤗 parakeet-tdt-0.6b-v2-coreai](https://huggingface.co/rahulrachuri/parakeet-tdt-0.6b-v2-coreai) | [parakeet-swift ↗](https://github.com/RahulRachuri/parakeet-swift) (kit enrollment pending) | CC-BY-4.0 |
| **Nemotron 3.5 ASR Streaming 0.6B** (speech→text — the zoo's first **STREAMING ASR**: live-mic transcription in 320 ms chunks, cache-aware FastConformer + pure RNN-T with explicit KV/conv cache I/O; 40 locales in one checkpoint via a run-time language input, punctuation built in, any-length audio) | [🤗 Nemotron-3.5-ASR-Streaming-CoreAI](https://huggingface.co/mlboydaisuke/Nemotron-3.5-ASR-Streaming-CoreAI) | [coreai-audio](apps/coreai-audio) | OpenMDW-1.1 |
| [**Streaming Sortformer 4-spk v2**](models/sortformer-diar/README.md) (speaker diarization — the zoo's first **"who spoke when"**, up to 4 speakers; NeMo core as a Core AI graph + Swift host streaming loop + AOSC speaker-cache compression; pairs with any zoo ASR for a **diarized transcript**; iPhone + Mac, 100% activity-agree vs NeMo) | [🤗 Streaming-Sortformer-Diar-CoreAI](https://huggingface.co/mlboydaisuke/Streaming-Sortformer-Diar-CoreAI) | [coreai-audio](apps/coreai-audio) | CC-BY-4.0 |
| [**S1-mini** by **Superwhisper**](models/s1-mini/README.md) (ASR **text normalizer** — the zoo's first **post-ASR** model: raw transcript → clean written text, fillers and false starts resolved, punctuation and casing applied, spoken numbers/dates/currency/emails written out; a Qwen3-0.6B finetune on the stock graph, steered by a control line. Completes the dictation path behind Parakeet / Nemotron-ASR, all on-device; **iPhone 17 Pro 62.4 tok/s, 276/276 token-exact**, chunk input to ~450–500 tok) | [🤗 S1-mini-CoreAI](https://huggingface.co/mlboydaisuke/S1-mini-CoreAI) | — (kit enrollment pending) | Apache-2.0 + naming clause |
| **Kokoro-82M** (text-to-speech — zoo's first TTS; StyleTTS2 + iSTFTNet, 28 English voices, runs on any text) | [🤗 Kokoro-82M-CoreAI](https://huggingface.co/mlboydaisuke/Kokoro-82M-CoreAI) | [Speak ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Speak) | Apache-2.0 |
| **VoxCPM-0.5B** (text-to-speech — diffusion TTS: MiniCPM4 LM + LocDiT flow-matching + AudioVAE; iPhone + Mac, int8 LM) | [🤗 VoxCPM-0.5B-CoreAI](https://huggingface.co/mlboydaisuke/VoxCPM-0.5B-CoreAI) | [Speak ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Speak) | Apache-2.0 |
| **VoxCPM2 2B** (text-to-speech — 2B successor at 48 kHz: MiniCPM4 28L LM + LocDiT-12L flow-matching + 48 kHz AudioVAE; iPhone + Mac, int8 LM) | [🤗 VoxCPM2-CoreAI](https://huggingface.co/mlboydaisuke/VoxCPM2-CoreAI) | [Speak ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Speak) | Apache-2.0 |
| [**VibeVoice-Realtime-0.5B**](models/vibevoice/README.md) (text → **multi-speaker dialogue** — the zoo's first **multi-speaker / podcast-style** TTS: dual Qwen2.5 LM (4L context + 20L speech) + **next-token diffusion** head (DPMSolver++ v-pred, CFG) + causal-conv acoustic VAE, 24 kHz; host turn-switching for N-speaker conversations, pairs with Sortformer for a **generate → diarize** loop; iPhone + Mac, all-fp16, **10.6 tok/s ≈ 1.4× real-time** on iPhone 17 Pro) | [🤗 VibeVoice-Realtime-0.5B-CoreAI](https://huggingface.co/mlboydaisuke/VibeVoice-Realtime-0.5B-CoreAI) | [coreai-audio](apps/coreai-audio) | MIT |
| [**pocket-tts**](models/pocket-tts/README.md) (text-to-speech — Kyutai's 109M streaming TTS, the zoo's **first Kyutai model** and **first Mimi conversion**: AR flow-matching LM over Mimi latents + one-step flow decoder + streaming Mimi decoder, 8 voices, takes text directly so there is no G2P layer to degrade; iPhone 17 Pro Max **7.8× real-time at 169 MB peak**; port by [Rahul Rachuri](https://github.com/RahulRachuri)) | [🤗 pocket-tts-coreai](https://huggingface.co/rahulrachuri/pocket-tts-coreai/tree/ad989309a5781c403113f9653f04a7d27c642c21) | [pocket-tts-swift ↗](https://github.com/RahulRachuri/pocket-tts-swift) (kit enrollment pending) | CC-BY-4.0 |
| [**Mel-Band RoFormer**](models/melband-roformer/README.md) (song → **vocals + instrumental stems** — the zoo's first **source separation**: band-split mel RoFormer with **STFT/iSTFT folded into the graph** as constant DFT matmuls, so the host only frames and overlap-adds; iPhone + Mac, fp16, **6.5× real-time** on iPhone 17 Pro) | [🤗 MelBandRoformer-Vocal-CoreAI](https://huggingface.co/mlboydaisuke/MelBandRoformer-Vocal-CoreAI) | [coreai-audio](apps/coreai-audio) | MIT |
| **Stable Audio Open Small** (text → **music / audio** — the zoo's first **generative audio**; latent diffusion: T5 encoder + DiT (8-step rectified-flow) + Oobleck VAE, ~11s 44.1 kHz stereo; fp16 ~1 GB, **~0.4 s / 11 s on M4 Max ≈ 30× real-time**; Stability AI + Arm) | [🤗 Stable-Audio-Open-Small-CoreAI](https://huggingface.co/mlboydaisuke/Stable-Audio-Open-Small-CoreAI) | [Music ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Music) | Stability Community |
| **V-JEPA 2** (ViT-L, SSv2 — the zoo's first **world model**: Meta's self-supervised video encoder (JEPA, predicts in representation space) + action-recognition head, 174 physical-interaction classes; 16-frame clip → action, fp16 ~675 MB, **~160 ms/clip on M4 Max**; Meta AI, MIT) | [🤗 VJEPA2-ViTL-SSv2-CoreAI](https://huggingface.co/mlboydaisuke/VJEPA2-ViTL-SSv2-CoreAI) | [ActionCamera ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ActionCamera) | MIT |
| **EmbeddingGemma 300M** (text embeddings — on-device RAG / semantic search) | [🤗 embeddinggemma-300m-CoreAI](https://huggingface.co/mlboydaisuke/embeddinggemma-300m-CoreAI) | [DocChat ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DocChat) | Gemma |
| **Qwen3-Embedding 0.6B** (multilingual text embeddings, last-token pooling + MRL) | [🤗 Qwen3-Embedding-0.6B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3-Embedding-0.6B-CoreAI) | [DocChat ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DocChat) | Apache-2.0 |
| **Qwen3-Reranker 0.6B** (cross-encoder reranker — yes/no relevance score) | [🤗 Qwen3-Reranker-0.6B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3-Reranker-0.6B-CoreAI) | [DocChat ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DocChat) | Apache-2.0 |
| [**ColModernVBERT**](models/colmodernvbert/README.md) (visual document retrieval — late-interaction MaxSim over page *images*, no OCR; zoo's first multi-vector retriever) | [🤗 ColModernVBERT-CoreAI](https://huggingface.co/mlboydaisuke/ColModernVBERT-CoreAI) | [DocSearch ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DocSearch) | MIT |
| [**GLiNER2-PII**](models/gliner2-pii/README.md) (information extraction / NER — the zoo's first **NER / schema-driven extraction** & first **DeBERTa-v3** port; zero-shot PII detection + redaction, any label set at call time; mDeBERTa-v3 fused graph + Swift host collator/decode, iPhone + Mac, byte-identical to GLiNER2) | [🤗 GLiNER2-PII-CoreAI](https://huggingface.co/mlboydaisuke/GLiNER2-PII-CoreAI) | [InfoExtract ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/InfoExtract) | Apache-2.0 |
| [**RF-DETR nano/small/medium/large**](models/rf-detr/README.md) (object detection, no NMS) | [🤗 RF-DETR-CoreAI](https://huggingface.co/mlboydaisuke/RF-DETR-CoreAI) | [DetectCamera ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DetectCamera) | Apache-2.0 |
| **RF-DETR-Seg nano→2xlarge** (instance segmentation, 6 sizes) | [🤗 RF-DETR-CoreAI](https://huggingface.co/mlboydaisuke/RF-DETR-CoreAI) | [DetectCamera ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DetectCamera) | Apache-2.0 |
| [**YOLOX-S**](models/yolox/README.md) (object detection — dense anchor-free, host NMS) | [🤗 YOLOX-CoreAI](https://huggingface.co/mlboydaisuke/YOLOX-CoreAI) | [DetectCamera ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DetectCamera) | Apache-2.0 |
| **AdcSR ×4** (super-resolution — zoo's first; one-step diffusion-GAN, on-device) | [🤗 AdcSR-CoreAI](https://huggingface.co/mlboydaisuke/AdcSR-CoreAI) | [UpscaleDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/UpscaleDemo) | Apache-2.0 + OpenRAIL++ |
| [**Depth Anything 3**](models/depth-anything-3/README.md) (monocular depth — zoo's first depth model; small + base, fp16/fp32) | [🤗 Depth-Anything-3-CoreAI](https://huggingface.co/mlboydaisuke/Depth-Anything-3-CoreAI) | [DepthCamera ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DepthCamera) | Apache-2.0 |
| **TripoSplat** (single image → **3D Gaussian splats** — the zoo's first 3D; DINOv3 ViT-H + 20-step flow-matching DiT + octree sampler + Gaussian decoder, Mac GPU ~1 min; `.ply`/`.splat` → RealityKit / [MetalSplatter](https://github.com/scier/MetalSplatter); VAST) | [🤗 TripoSplat-CoreAI](https://huggingface.co/mlboydaisuke/TripoSplat-CoreAI) | [TripoSplatMac](apps/TripoSplatMac) | MIT |
| **LTX-Video 2B distilled** (text → **video** — the zoo's first video model; T5-XXL + 8-step flow-matching DiT + causal video VAE, host FlowMatch sampler; 512×768×49f ~14 s Mac GPU; Lightricks) | [🤗 LTX-Video-2B-CoreAI](https://huggingface.co/mlboydaisuke/LTX-Video-2B-CoreAI) | [CoreAIVideo](apps/CoreAIVideo) | Other (LTXV) |
| **FLUX.2 klein 4B** (text → **image** + **in-context editing** — the zoo's first image-generation & editing model; step-distilled flow-matching DiT (4 steps, guidance 1.0) + Qwen3 text encoder, 1024²; native **in-context edit** — add/replace/combine while keeping the subject, unlike strength-based SDEdit — plus **multi-reference** compose, both exported as edit-sequence transformers (output latent T=0 concatenated with reference tokens T=10·i); int4, Mac; Black Forest Labs) | [🤗 FLUX.2-klein-4B-CoreAI](https://huggingface.co/mlboydaisuke/FLUX.2-klein-4B-CoreAI) | [CoreAIImageGen](apps/CoreAIImageGen) | Apache-2.0 |
| [**GLM-Image**](models/glm-image/README.md) (text → **image** — the zoo's first **AR + diffusion hybrid**; a 9B GLM-4 AR model *samples the image as discrete visual prior tokens* like an LLM (~36 tok/s), then a 7B flow-matching DiT denoises conditioned on them + 16ch VAE; composition from the AR, texture from the DiT; 1024² native + 512² fast, int8, Mac; ZhipuAI) | [🤗 GLM-Image-CoreAI](https://huggingface.co/mlboydaisuke/GLM-Image-CoreAI) | [CoreAIImageGen](apps/CoreAIImageGen) | MIT |
| [**Z-Image-Turbo**](models/z-image-turbo/README.md) (text → **image** — a 6B Single-Stream DiT (S3-DiT): Qwen3-4B text encoder → 34-block DiT (8-step FlowMatchEuler + CFG) → 16ch VAE; photoreal by default. **One graph covers 256²/512²/1024² and any prompt length** (dynamic image + caption axes, ~5–9 % cost). bf16 and **near-lossless** — PSNR 42.6 dB vs the fp32 reference; 18 s @512² / 70 s @1024² on M4 Max. fp16 NaNs this model, so Mac-only: AOT will not take a bf16 module; Alibaba Tongyi-MAI) | [🤗 Z-Image-Turbo-CoreAI](https://huggingface.co/mlboydaisuke/Z-Image-Turbo-CoreAI) | [CoreAIImageGen](apps/CoreAIImageGen) | Apache-2.0 |
| [**TimesFM 2.5 200M**](models/timesfm/README.md) (**time-series forecasting** — the zoo's first forecasting foundation model; decoder-only patched transformer, any univariate series → 128-step point + 10-quantile forecast; one stateless graph + host RevIN/flip DSP, fp16 ~463 MB, **~14 ms/forecast** M4 Max / **~25 ms iPhone 17 Pro** device-verified; Google) | [🤗 TimesFM-2.5-200M-CoreAI](https://huggingface.co/mlboydaisuke/TimesFM-2.5-200M-CoreAI) | [Forecast ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Forecast) | Apache-2.0 |

▸ **Run in app** — apps in [`apps/`](apps) live in this repo; **↗** links a
[CoreAIKit example app](https://github.com/john-rocky/coreai-kit/tree/main/Examples); **‡** = app
wiring in progress. **🔧** = the bundle still carries coreai-torch 0.4.0-era IR, which every
OS 27 build from beta 2 on refuses at load (measured through macOS 26A5416b, 2026-09-04); it
is queued for the in-place `strip_debug_info` repair (see the
[recovery note](#recovery-note--the-coreai-torch-040-incident) at the end).
Full app list: [`apps/README.md`](apps/README.md).

**GA validation** (the badge at the top) is staged, not assumed: on the day the release OS
lands, the most-downloaded bundles are downloaded and loaded through the runtime on that build
(`python3 conversion/zoo_smoke.py --top 20`), and the rest of the catalog follows over the next
week. The badge counts only bundles checked on the current release build; per-bundle status and
the build that checked it are in [`models/_INVENTORY.md`](models/_INVENTORY.md#tier-0-load-check).
A bundle that fails there is repaired or re-exported, not relabelled.

### Built with the zoo

Third-party apps running zoo models. Built something? Open a
[showcase issue](https://github.com/john-rocky/coreai-model-zoo/issues/new?template=showcase.yml)
— a name, a link, and one line is all it takes. *Your app here.*

### Most downloaded

<img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/charts/hf-top.svg" alt="Most downloaded zoo models this month" width="700">

*(auto-updated weekly from Hugging Face download counts)*

### Decode throughput (tok/s, greedy; output top-1 exact vs the Hugging Face reference)

| | iPhone 17 Pro · GPU | iPhone 17 Pro · ANE | M4 Max · GPU |
|---|---|---|---|
| **Qwen3.5-0.8B** | **71.9** | 14.7 | **210** |
| **Qwen3.5-2B** | **29** | — | **161** |
| **LFM2.5-1.2B** | **45.4** | — | **276.5** |
| **LFM2.5-2.6B** (reasoning; int8hu / int4lin 2.0 GB) | not measured | — | **116.7** / **139.2** |
| **Granite 4.0-H 1B** | **36.3** | — | **136.5** |
| **Nanbeige4.1-3B** | **15.9** | — | **114.5** |
| **Nanbeige4.2-3B** (22 physical / 44 executed+cache layers; int8) | pending | — | **46.4** |
| **S1-mini** (Superwhisper, ASR text normalizer, int8lin — device 276/276 + 27/27 token-exact; ⚠️ iOS cap: prompt+gen < 1024 tok) | **62.4** | — | **268.4** |
| **MiniCPM5-1B** (OpenBMB, int8 — 24/24 exact vs HF) | **66.8** | — | 59.4 |
| **Youtu-LLM-2B** (dense MLA, int8 — 16/16 device ≡ Mac ≡ HF) | **~19** (in-app ~24) | — | **102.8** |
| **FastContext-1.0-4B** (repo-exploration agent, 4bit — AOT h18p; ANE inference unsupported) | **20.4** | ✗ | — |
| **BitCPM-8B** (1.58-bit ternary, OpenBMB — custom 2-bit packed-GEMM kernel; AOT h18p; ~2.1 GB resident; token-exact 3/3 vs ref) | **17** | ✗ | **62.7** |
| **Gemma 4 E2B** | **30.3** (QAT 30.7) | 6 | **77.0** (QAT 78.9) |
| **Gemma 4 E4B** (official QAT) | **15.1** | — | **55.8** |
| **Gemma 4 E2B VL** (image+text, official QAT) | **25.5** | — | **82.4** |
| **MiniCPM-V 4.6** (vision-language, sub-2B) | **53.4** | — | **224.3** |
| **LFM2.5-VL-450M** (vision-language, int8lin — image buffer bound; vision encode 33.6 ms/image on iPhone, 18.0 on Mac) | **112.0** | — | **387.2** |
| **LFM2.5-VL-3B** (vision-language, int4 on iPhone / int8 on Mac; vision 75.7 ms/image) | **19.3–22.8** | — | **105.3** |
| **North-Micro-Vision** (vision-language 2.4B, int8lin — image bound; vision 83.4 ms/image on Mac) | **18.2** | — | **118.6** |
| **Qwen3.6-35B-A3B** (MoE, 35B/~3B active, Mac-only) | — | — | **64.9** † |
| **Qwen3.6-27B** (dense, Mac-only) | — | — | **15.9** |
| **Qwen3.8-27B** (dense VLM, Mac-only; vision 111 ms/image, VLM prefill 86.0 tok/s via pf16) | — | — | **15.7** |
| **GLM-4.7-Flash** (MoE + MLA, 30B/~3B active, Mac-only) | — | — | **52.4** † |
| **Gemma 4 12B** (dense, Mac-only) | — | — | **23** int8 / **33** int4 ‡ |
| **Gemma 4 31B** (dense, Mac-only) | — | — | **17.2** int4 ‡ |

Measured on the iOS 27 / macOS 27 beta, Apple's `coreai-pipelined` GPU engine, zero custom
kernels (ANE column + **†**/**‡** excepted). **†** = MoE bundle using the custom
[`gather_qmm`](knowledge/compute-units-and-authoring.md) Metal kernel (reads only the routed
experts). **‡** = dense bundle whose full/global-attention SDPA is a custom flash-decode Metal
kernel — the stock MPSGraph SDPA crashes on the ≥16-head × 512 Q (a GPU scratch-heap overflow,
[apple/coreai-models#27](https://github.com/apple/coreai-models/issues/27)), so these models are
**unrunnable without it**. Prefill, sizes, per-model caveats, and the Mac-only big models: [`models/`](models/).

<p align="center">
  <img width="380" alt="CoreAIChat screen recording" src="https://github.com/user-attachments/assets/999dbd95-45b5-468f-b1a8-34112ee3b74d" />
</p>
<p align="center"><i>CoreAIChat (<a href="apps/">apps/</a>) — the zoo's models running on-device on iPhone.</i></p>

## Start here

- **You're a coding agent** (or you're pointing one here) → [**`AGENTS.md`**](AGENTS.md) — the
  porting contract in one file: why conversion isn't conversion, the two gates, the traps agents
  hit, and what isn't an agent's call. No install needed. One fetch for everything here:
  [**`llms.txt`**](https://john-rocky.github.io/coreai-model-zoo/llms.txt).
- **Looking something up about the runtime itself** →
  [**john-rocky.github.io/coreai-model-zoo**](https://john-rocky.github.io/coreai-model-zoo/) —
  the knowledge base as a page per topic. Apple documents the API surface well; what these notes
  add is what the runtime *does* when you run it — the thresholds, the failure modes, the
  measured numbers — which is not written down anywhere. Plain HTML, so a reader without JS, a
  search crawler and a coding agent all get the same words.
- **Reading start to finish rather than looking one thing up** →
  [**The Art of Core AI**](https://john-rocky.github.io/the-art-of-core-ai/) — a free book built
  from these same measurements ([Japanese edition](https://zenn.dev/mlboydaisuke/books/coreai-textbook)).
- **Surveying the whole Core AI ecosystem**, not just this catalog →
  [**awesome-core-ai**](https://github.com/john-rocky/awesome-core-ai) — Apple's own tooling,
  other people's converters and runtimes, sample apps, benchmarks, and learning material.
- **Try the app** (iOS 27 / macOS 27 beta; the model downloads in-app):
  - **Demo app, no build** → Mac: [**.dmg**](https://github.com/john-rocky/coreai-model-zoo/releases/download/mac-v1.0/CoreAI-Zoo-for-Mac.dmg) (notarized, runs the Mac-only bundles) · iPhone: [**CoreAIChat on TestFlight**](https://testflight.apple.com/join/bK4P7xby)
  - **Build it** → [`apps/`](apps/) — Xcode 27 beta + xcodegen, the `coreai-models` patch stack + `tokenizer.json`
- **Use a model in your own app** → add [**CoreAIKit**](https://github.com/john-rocky/coreai-kit)
  (SPM) and load the catalog id; the model's card has the complete snippet + a 5-line
  integration checklist (golden example: [`models/whisper-large-v3-turbo/README.md`](models/whisper-large-v3-turbo/README.md)).
  Engine-level deep-dive: [`knowledge/swift-runtime.md`](knowledge/swift-runtime.md)
- **Port a model, end to end** → [**`PORTING.md`**](PORTING.md) — the complete walk from HF
  checkpoint to a verified `.aimodel` on iPhone (oracle → export → gates → device → publish),
  with a vision and an LLM worked example. Start here to contribute a port.
- **Rebuild a published bundle** → [`models/<model>/recipe.toml`](models/) via
  `python3 conversion/zoo_convert.py run <name>` — see [Rebuild a bundle](#rebuild-a-bundle--the-conversion-recipes)
- **Convert a model** (export API + gotchas) → [`knowledge/conversion-guide.md`](knowledge/conversion-guide.md)
- **Compress** → [`knowledge/compression.md`](knowledge/compression.md)
- **Make it fast** → [`knowledge/custom-metal-kernels.md`](knowledge/custom-metal-kernels.md) · [`knowledge/performance-ceiling.md`](knowledge/performance-ceiling.md)
- **Known beta issue** (in-graph KV-write crash; workarounds + the input-mask escape) → [`knowledge/coreai-beta-mpsgraph-kvwrite-bug.md`](knowledge/coreai-beta-mpsgraph-kvwrite-bug.md) — FB23024751 / [apple/coreai-models#5](https://github.com/apple/coreai-models/issues/5)

## Repository layout

| Dir | What |
|---|---|
| [**coreai-kit** ↗](https://github.com/john-rocky/coreai-kit) | (sibling repo) The Swift package that runs this zoo: 1-line `catalog:` APIs (`ChatSession`, `KitTranscriber`, …), model download + cache, and per-kind example apps in [`Examples/`](https://github.com/john-rocky/coreai-kit/tree/main/Examples) — the cards' ▶️ / 💻 doors point there. |
| [`models/`](models/) | One directory per model — the card, the `recipe.toml` that reproduces its published bundles, and the generated [`_INVENTORY.md`](models/_INVENTORY.md) / [`index.json`](models/index.json). Laid out like [`apple/coreai-models`](https://github.com/apple/coreai-models). |
| [`skills/`](skills/) | Agent skills — install them and your coding agent can pick a model, reproduce its bundle, and verify the result. |
| [`knowledge/`](knowledge/) | Verified notes on the framework: conversion, compression, stateful KV, custom Metal kernels, AOT, compute-unit rules, the Swift runtime. |
| [`conversion/`](conversion/) | Re-authored models + convert / verify / compress scripts (PyTorch → `.aimodel`). |
| [`cli/`](cli/) | `coreai export` / `doctor` / `verify` — find a model's export route, lint an artifact against the failure patterns in [`knowledge/`](knowledge/), gate it against an HF oracle. [`DOCTOR_RULES.md`](cli/DOCTOR_RULES.md) is the table those checks come from: 64 ways a conversion succeeds and the result is quietly wrong. |
| [`swift/`](swift/) | `CoreAIRunner` — a Swift package that drives `.aimodel` LLM bundles, including architectures beyond the standard runtime. |
| [`apps/`](apps/) | **Engine showcases** — apps for models that need a hand-tuned backend (custom Metal kernels, patch stack: BitCPM, RWKV-7, LLaDA, …) and the device-verification bench behind the published numbers. Want to *just run a model*? Use the [kit examples ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples) instead. |

## Agent Skills

This repo ships a plugin so a coding agent can use the zoo without being told how: pick a model
from the catalog, reproduce its published bundle with the recorded recipe, and verify the
result against the model it came from. With no plugin mechanism at all, point the agent at
[`AGENTS.md`](AGENTS.md) — the same contract, zero install.

| Skill | Description |
| --- | --- |
| `reproduce-a-zoo-model` | Choose a model from `models/index.json`, rebuild its published bundle with `zoo_convert.py` and its prerequisites, and check it with `zoo_verify.py`. |
| `port-a-model-to-the-zoo` | Port a **new** model: the oracle-first method, the two gates, compression choices, the device tier, and what a finished port must ship. |

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

```
gemini extensions install /path/to/coreai-model-zoo/skills
```

Apple's own [`coreai-skills`](https://github.com/apple/coreai-models) covers the toolchain
itself (authoring, export, compression). These two cover this catalog; install both.

## Contributing

Every model in the table is a conversion recipe anyone could have written — and some are, by
[people who are not me](#community-ports). [Nanbeige4.2-3B](models/nanbeige4.2-3b/README.md)
([PR #6](https://github.com/john-rocky/coreai-model-zoo/pull/6), by
[@ukint-vs](https://github.com/ukint-vs)) is the reference for what a contributed port looks
like, and [pocket-tts](models/pocket-tts/README.md)
([PR #12](https://github.com/john-rocky/coreai-model-zoo/pull/12), by
[@RahulRachuri](https://github.com/RahulRachuri)) is the reference for how far the gates can be
taken — an ASR round trip and a 302-sentence sweep, because tensor cosine can pass while the
audio is unintelligible. The shape of both: pinned checkpoint + immutable bundle revision, the
overlay/recipe/gates in the PR, the bundle published under the **contributor's own HF
namespace** (you keep ownership and credit), and the maintainer runs the iPhone hardware gate
for you if you don't have the device.

- **Port a model** — [`PORTING.md`](PORTING.md) walks the whole path (oracle → export → gates →
  publish); PRs welcome.
- **Conversion requests** — a model you'd like to see here? [Open an issue](https://github.com/john-rocky/coreai-model-zoo/issues/new) with the Hugging Face link and what you'd use it for.
- **No code needed** — run the Bench tab in [CoreAIChat (TestFlight)](https://testflight.apple.com/join/bK4P7xby) and submit the result: your device becomes a row in [`BENCHMARKS.md`](BENCHMARKS.md).

## Recovery note — the coreai-torch 0.4.0 incident

Every model broken by the `coreai-torch` 0.4.0 debug-location issue has been re-published:
re-converted with 0.4.1 (verification gates re-run), or repaired in place with
[`strip_debug_info`](https://github.com/apple/coreai-torch/issues/44) (debug locations removed,
weights byte-identical). Catalog-served apps just re-download. The few 🔧 rows above are the
remaining re-export queue. If you have your own 0.4.0-converted assets, `strip_debug_info`
fixes them in minutes — no re-conversion needed. (One port, FastContext-1.0-4B, was retired
instead of recovered: Microsoft removed its upstream weights on 2026-06-30, so it cannot be
rebuilt.) Details: [`knowledge/coreai-torch-041-ir-incident.md`](knowledge/coreai-torch-041-ir-incident.md).

## Maintainer

[**Daisuke Majima (MLBoy)**](https://github.com/john-rocky) — the converted models live at
[huggingface.co/mlboydaisuke](https://huggingface.co/mlboydaisuke); the same maintainer runs
[devicemark](https://devicemark.github.io/) (on-device LLM leaderboard, where zoo ports are
measured) and wrote the Japanese textbook
[The Art of Core AI](https://zenn.dev/mlboydaisuke/books/coreai-textbook), drawn from these ports.

## License

BSD-3-Clause ([`LICENSE`](LICENSE)). Re-authored model code derives from Apple's BSD-3-Clause
`coreai_models` and retains its notices. Model weights follow their own licenses (see each
Hugging Face repo).
