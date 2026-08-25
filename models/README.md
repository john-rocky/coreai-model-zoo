# Models

One directory per model, laid out like [`apple/coreai-models`](https://github.com/apple/coreai-models)
so that an agent primed on Apple's repo recognises this one without new instructions:

```
models/<model>/
  README.md      the card — architecture, parity numbers, measured speeds, lessons
  recipe.toml    the configuration that produced each published bundle (our addition)
  verify.toml    expected config, where this port deliberately deviates from its source
```

Ready-to-run bundles are on **Hugging Face** — one best verified configuration per platform ×
compute unit. Every card links its source checkpoint, its exporter, and its parity numbers.

Two files here are generated, never hand-edited: [`_INVENTORY.md`](_INVENTORY.md) (every
published repo, with downloads and verification verdicts) and [`index.json`](index.json) (the
same thing machine-readable — start there if you are an agent).

## Reproduce a bundle

```bash
python3 conversion/zoo_convert.py list                  # what can be reproduced
python3 conversion/zoo_convert.py show qwen3.5-0.8b     # the command + its prerequisites
python3 conversion/zoo_convert.py doctor                # is this venv wired up?
python3 conversion/zoo_convert.py run  qwen3.5-0.8b     # do it
```

Scripts that can run without any setup say so: `show` prints a `uv` line for the ones that
declare their own dependencies inline (PEP 723, the same mechanism Apple's `models/*/export.py`
use), so `uv run conversion/export_da3.py --variant small --dtype float16 --res 504` is the
whole story for those ports. The rest import re-authored model code and need the overlay
environment — `zoo_convert.py doctor` checks it.

Unlike Apple's repo, several families share one exporter (Qwen3.5's drives Ornith,
Qwen3.6-27B and Qwen3.8-27B too), so the export scripts stay in [`../conversion/`](../conversion/) and
`recipe.toml` names the one to run. A recipe marked `status = "unverified"` will not run
without `--force`: the repo does not record which configuration produced the published bundle,
and a wrong recipe is worse than a missing one.

Two fields name the two ends of a conversion, and they are easy to confuse:
**`hf_repo`** is the repo the recipe publishes **to**; **`source_hf_id`** is the checkpoint it
converts **from**. Ports whose upstream is not on the Hub carry **`source_repo`** instead
(RF-DETR, YOLOX, AdcSR, TripoSplat, LTX-Video are released on GitHub). Recording the source is
what lets a tool answer "how do I convert *this* model" from a Hugging Face id, rather than
only "how was *that* bundle produced" from a recipe name.

## Check a published bundle

```bash
python3 conversion/zoo_verify.py mlboydaisuke/qwen3.5-0.8B-CoreAI   # one repo
python3 conversion/zoo_verify.py --all --json models/_VERIFY.json   # the whole catalog

python3 cli/coreai_doctor.py exports/my_bundle --profile iphone     # known failure patterns
python3 cli/coreai_verify.py exports/my_bundle -n 16                # vs an HF oracle
```

Tier 1 compares a bundle's tokenizer, chat template, context length and declared precision
against the source repository named in its own `metadata.json` — no oracle, no device, no
weights, so the whole catalog checks in minutes.

[`cli/`](../cli/) asks a different question of the same bundle: not "does it match its
source" but "does it match what the runtime will do with it", from the bundle's own files.
The two are complementary — a port can copy its source perfectly and still ship a value the
chat template never emits. See [`cli/DOCTOR_RULES.md`](../cli/DOCTOR_RULES.md).

## Catalog

| Card | Family | Download | Status |
|---|---|---|---|
| [`qwen3.5.md`](qwen3.5/README.md) | Qwen3.5 (hybrid linear+full attn) | [🤗 qwen3.5-0.8B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreAI) | 0.8B + 2B, top-1 exact vs HF |
| [`gemma4-e2b.md`](gemma4-e2b/README.md) | Gemma 4 (multimodal; text decoder) | [🤗 gemma-4-E2B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI) | 8/8 exact vs HF |
| [`gemma4-vl.md`](gemma4-vl/README.md) | Gemma 4 E2B vision (image+text→text, 2nd VLM) | `vl/` in [🤗 gemma-4-E2B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI) | margin-ruled exact vs fp32 HF; **82.4 tok/s M4 Max / 25.5 iPhone 17 Pro** (pipelined VLM rider) |
| [`lfm2.5.md`](lfm2.5/README.md) | LFM2.5 (conv + full-attn hybrid, LiquidAI) | [🤗 LFM2.5-1.2B-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-1.2B-CoreAI) | 1.2B, oracle gate 16/16, **276.5 tok/s M4 Max / 44.1–46.6 iPhone (int8 + absmax int8 head)** (pipelined) |
| [`lfm2.5-2.6b.md`](lfm2.5-2.6b/README.md) | LFM2.5 2.6B (same hybrid, 30L; reasoning model) | [🤗 LFM2.5-2.6B-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-2.6B-CoreAI) | oracle gate 16/16 on int8hu **and** int4lin; **116.7 tok/s M4 Max int8hu / 139.2 int4lin (2.0 GB)**, no iPhone measurement yet |
| [`lfm2.5-vl.md`](lfm2.5-vl/README.md) | LFM2.5-VL **450M + 3B** (SigLIP2-NaFlex tower + the LFM2 decoder) — the **smallest VLM here**, 658 MB, and its 3.9 GB detail tier | [🤗 LFM2.5-VL-450M-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-VL-450M-CoreAI) | vision cos 0.999996 + 48/48 token-exact vs fp32 HF; **387.2 tok/s M4 Max** (text core) / **112.0 iPhone 17 Pro** (image bound, nat 16/16 + image oracle 24/24) + vision **33.6 ms/image iPhone / 18.0 Mac** (device cos 0.999995); int8lin 7/9 suite cases (fp16 baseline 8/9), **int4 = 0/9, no-go**. 3B (Mac-only, AOT past the iOS 2 GiB wall): vision 75.7 ms/image, text core **105.3 tok/s**, suite 7/9 at int8 **and int4** |
| [`north-micro-vision.md`](north-micro-vision/README.md) | North-Micro-Vision-Instruct (Cohere, 2.4B) — a **Qwen3-VL tower reused verbatim** + a parallel-block Cohere decoder; 11 languages | [🤗 North-Micro-Vision-CoreAI](https://huggingface.co/mlboydaisuke/North-Micro-Vision-CoreAI) | fp32 ladder cos 1.000000 at every seam; suite **9/9 (338/338 tokens)** at int8; **145.3/118.6 tok/s M4 Max**, **21.5/18.2 iPhone 17 Pro with image oracle 24/24**; vision 83.4 ms/image; int4 = 0/9 no-go |
| [`shieldstral.md`](shieldstral/README.md) | Shieldstral-1.0-3B (Mistral) — a **policy-conditioned safety classifier**, shipped as a stateless graph rather than a decoder | [🤗 Shieldstral-CoreAI](https://huggingface.co/mlboydaisuke/Shieldstral-CoreAI) | `ministral3 == Mistral + YARN` measured at cos **1.000000** / |ΔP| 0.00000; **9/9 verdicts vs fp32** at fp16, int8lin AND int4lin; **123.6 ms/verdict (S=256) / 232.5 ms (S=512)** on M4 Max and **371.9 / 624.7 ms on an iPhone 17 Pro** (9/9 there too), 2.53 GB; quantization buys size, not speed |
| [`granite-4.0-h.md`](granite-4.0-h/README.md) | Granite 4.0-H (Mamba2 + attn hybrid, IBM) | [🤗 granite-4.0-h-CoreAI](https://huggingface.co/mlboydaisuke/granite-4.0-h-CoreAI) | 1b + 350m, oracle gate 16/16, **136.5 tok/s M4 Max / 35.4–37.1 iPhone 17 Pro (int8 head)** (pipelined, first SSM-scan rider) |
| [`nemotron-3-nano.md`](nemotron-3-nano/README.md) | Nemotron-3-Nano 4B (Mamba2 + attn + MLP hybrid, NVIDIA) — zoo's **2nd SSM-scan rider, 1st non-Granite Mamba2** | [🤗 Nemotron-3-Nano-4B-CoreAI](https://huggingface.co/mlboydaisuke/Nemotron-3-Nano-4B-CoreAI) | 4B int8hu 4.3 GB, AOT h18p, **16.0 tok/s iPhone 17 Pro (cooled) / 85.2 tok/s M4 Max** (raw AIModel), nat 24/24 + oracle 24/24; no custom kernel |
| [`minicpm5-1b.md`](minicpm5-1b/README.md) | MiniCPM5-1B (plain Llama dense, OpenBMB) | [🤗 MiniCPM5-1B-CoreAI](https://huggingface.co/mlboydaisuke/MiniCPM5-1B-CoreAI) | 1.08B int8, **lossless** (24/24 token-exact vs HF fp32), **66.8 tok/s iPhone 17 Pro** (pipelined, llama→mistral remap) |
| [`fastcontext.md`](fastcontext/README.md) | FastContext-1.0-4B-SFT (repo-exploration agent, Qwen3-4B arch, Microsoft) | [🤗 FastContext-1.0-4B-CoreAI](https://huggingface.co/mlboydaisuke/FastContext-1.0-4B-CoreAI) | 4B 4bit, parity **23/24 argmax (ppl 1.41)** vs HF, **20.4 tok/s decode / 22.1 prefill iPhone 17 Pro** (AOT h18p GPU; zoo's first stock-arch + first 4B-class iPhone LLM) |
| [`youtu.md`](youtu/README.md) | Youtu-LLM-2B (dense DeepSeek-MLA, Tencent) — zoo's **first iPhone MLA + first dense MLA** (absorbed latent-KV + flash-decode kernel shared with GLM-4.7-Flash) | [🤗 Youtu-LLM-2B-CoreAI](https://huggingface.co/mlboydaisuke/Youtu-LLM-2B-CoreAI) | 1.96B int8, **token-exact vs HF fp32** (naive+absorbed 0 flips; int8 engine **16/16 device ≡ Mac ≡ HF**), **102.8 tok/s M4 Max / ~19 iPhone 17 Pro** |
| [`rf-detr.md`](rf-detr/README.md) | RF-DETR + RF-DETR-Seg (detection / instance segmentation, Roboflow) | [🤗 RF-DETR-CoreAI](https://huggingface.co/mlboydaisuke/RF-DETR-CoreAI) | det ×4 + seg ×6 fp32, gated cpu+gpu (mask IoU 1.000), **8.6–59.1 ms/frame M4 Max GPU** |
| [`depth-anything-3.md`](depth-anything-3/README.md) | Depth Anything 3 (monocular depth, DINOv2+DPT, ByteDance) — zoo's **first depth model** | [🤗 Depth-Anything-3-CoreAI](https://huggingface.co/mlboydaisuke/Depth-Anything-3-CoreAI) | small + base, fp16/fp32, engine cos 1.000000 (cpu+gpu) / vs official mean r 0.98, **54 MB · 65.7 FPS M4 Max GPU (small fp16)** |
| [`qwen3-embedding.md`](qwen3-embedding/README.md) | Qwen3-Embedding (multilingual text embedder, last-token pooling + MRL, Alibaba) | [🤗 Qwen3-Embedding-0.6B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3-Embedding-0.6B-CoreAI) | 0.6B fp16, torch ladder exact + engine gate cos 0.999998, **25–45 ms/embedding M4 Max GPU** |
| [`qwen3-reranker.md`](qwen3-reranker/README.md) | Qwen3-Reranker (cross-encoder reranker, yes/no logit score, Alibaba) | [🤗 Qwen3-Reranker-0.6B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3-Reranker-0.6B-CoreAI) | 0.6B fp16, torch ladder exact (P(yes) Δ=0) + engine gate Δ<5e-4, **45.7 ms/score M4 Max GPU** |
| [`holo2.md`](holo2/README.md) | Holo2-4B (GUI-grounding / computer-use VLM, Qwen3-VL-4B backbone, H Company) | [🤗 Holo2-4B-CoreAI](https://huggingface.co/mlboydaisuke/Holo2-4B-CoreAI) | 4B int8lin + fp16 vision, parity **vision cos 0.9999 / decoder 16/16** vs fp32 HF; rides the Qwen3-VL pipeline; zoo's **first GUI-grounding / computer-use model** |
| [`colmodernvbert.md`](colmodernvbert/README.md) | ColModernVBERT (visual document retriever, late-interaction/MaxSim, ModernBERT+SigLIP2) — zoo's **first visual retriever + first late-interaction model** | [🤗 ColModernVBERT-CoreAI](https://huggingface.co/mlboydaisuke/ColModernVBERT-CoreAI) | 250M, query + doc encoders fp16/fp32, engine per-token cosine **1.000000** (fp32) / ≥0.99999 (fp16), MaxSim == `processor.score` exactly, single-tile retrieval 3/3 |
| [`yolox.md`](yolox/README.md) | YOLOX-S (single-stage anchor-free detector, YOLO-family, Megvii) — zoo's **first YOLO / single-stage detector** (CNN counterpart to RF-DETR; needs host NMS) | [🤗 YOLOX-CoreAI](https://huggingface.co/mlboydaisuke/YOLOX-CoreAI) | 8.97M fp32, gated cpu+gpu (head cos **1.000000**, detections IoU **1.000**), **4.80 ms · 208 FPS M4 Max GPU · ~22 ms iPhone 17 Pro GPU** (device-verified live in DetectCamera) |
| [`parakeet.md`](parakeet/README.md) | Parakeet-TDT-0.6B (FastConformer transducer / TDT, NVIDIA) — zoo's **first transducer / TDT (RNN-T family)** ASR (3 graphs + host greedy loop, not an LLM) | [🤗 Parakeet-TDT-0.6B-CoreAI](https://huggingface.co/mlboydaisuke/Parakeet-TDT-0.6B-CoreAI) | 600M, encoder fp16 + predict/joint fp32, **77/77 token-exact e2e** vs HF (GPU enc cos 0.999995); **iPhone 17 Pro: 14.84 s clip → 0.31 s (47.9× real-time)**, AOT encoder load 3.9 s; 25 EU langs |
| [`parakeet-v2`](parakeet-v2/README.md) | Parakeet-TDT-0.6B-**v2** (same FastConformer TDT, English-only; vocab 1025 / blank 1024) — ported by [Rahul Rachuri](https://github.com/RahulRachuri); NVIDIA ships v2 as a `.nemo` only, so the HF-layout checkpoint is itself a conversion (two fixes to transformers' `convert_nemo_to_hf.py`) | [🤗 parakeet-tdt-0.6b-v2-coreai](https://huggingface.co/rahulrachuri/parakeet-tdt-0.6b-v2-coreai) | 3 bundles, **82/82 token-exact e2e** vs HF `ParakeetForTDT` (GPU enc cos 0.999998); **iPhone 17 Pro Max 175.3× real-time** over 3989.9 s of audio, 152/152 chunks and 20376/20376 tokens exact, load 0.27 s; M4 Pro 291× peak |
| [`nemotron-asr-streaming.md`](nemotron-asr-streaming/README.md) | Nemotron 3.5 ASR Streaming 0.6B (cache-aware FastConformer + pure RNN-T, NVIDIA) — zoo's **first STREAMING ASR** (live mic, 320 ms chunks, 40 locales via a run-time language input, punctuation built in, any-length) | [🤗 Nemotron-3.5-ASR-Streaming-CoreAI](https://huggingface.co/mlboydaisuke/Nemotron-3.5-ASR-Streaming-CoreAI) | 600M, conformer fp16 (two 12-layer AOT halves) + predict/joint fp32, **99/99 token-exact e2e** vs HF streaming; **iPhone 17 Pro: 53 ms/chunk = 6.0× real-time**, cached load ~4 s; OpenMDW-1.1 (commercial OK) |
| [`whisper-large-v3-turbo.md`](whisper-large-v3-turbo/README.md) | Whisper large-v3-turbo (speech→text, OpenAI) — official-recipe artifact + fixed-128 autoregressive decode, stock runtime; **the golden card** (▶️ runner / 💻 snippet / checklist) | [🤗 whisper-large-v3-turbo-CoreAI-official](https://huggingface.co/mlboydaisuke/whisper-large-v3-turbo-CoreAI-official) | 809M fp16, **token-for-token exact** vs HF greedy; 0.18 s/token M4 Max (first step 0.68 s); Mac + iPhone (AOT); 100 langs, auto-detect |
| [`ltxvideo.md`](ltxvideo/README.md) | LTX-Video 2B distilled (text→video flow-matching DiT, Lightricks) — zoo's **first VIDEO model** (T5 + DiT + causal video VAE; 8-step host FlowMatch sampler) | [🤗 LTX-Video-2B-CoreAI](https://huggingface.co/mlboydaisuke/LTX-Video-2B-CoreAI) | 3 nets converted, per-net cos **1.000000** + DiT 8/8 real-step cos 1.000000; **DiT fp16 + VAE fp16 + T5 bf16 = 13.5 G**; **512×768×49f in ~14 s M-series GPU**, coherent photoreal video; Mac-first |
| [`triposplat.md`](triposplat/README.md) | TripoSplat (single image → 3D Gaussian splats, VAST) — zoo's **first 3D model** | [🤗 TripoSplat-CoreAI](https://huggingface.co/mlboydaisuke/TripoSplat-CoreAI) | 5 nets converted, each converted-vs-eager cos **1.000000**; DiT 20-step flow sampler + octree resampling host-side; **~1 min/image Mac GPU**; `.ply`/`.splat` out |
| [`ornith-1.0-9b.md`](ornith-1.0-9b/README.md) | Ornith-1.0-9B (agentic coding / self-scaffolding, Qwen3.5 hybrid arch, DeepReinforce) — zoo's **first agentic-coding model** | HF upload pending (user-gated) | 9B, eager gate **24/24 exact** vs fp32 oracle (fp16 / int8hu / **int4lin — family-first clean int4**) + engine greedy **12/12 ≡ oracle**, **48.3 tok/s** int8hu ship / **58.9** int4lin option, M4 Max (pipelined, zero new export code; Mac ship) |
| [`nanbeige4.2-3b.md`](nanbeige4.2-3b/README.md) | Nanbeige4.2-3B (22 shared physical Llama layers × 2 passes), ported by [Vadim Smirnov](https://github.com/ukint-vs) | [🤗 Nanbeige4.2-3B-CoreAI](https://huggingface.co/ukint-vs/Nanbeige4.2-3B-CoreAI/tree/5864ec7a5581940958e58354a6b6c46c8f06891e) | int8hu 4.59 GiB, fp32 + engine gates pass, **46.4 tok/s M4 Max**; 44 KV cache layers; **iPhone 17 Pro device-gated: 24/24 token-exact ×2 runs, 8.5/6.4 tok/s settled** — zoo's **first community-contributed model** |
| [`s1-mini.md`](s1-mini/README.md) | **S1-mini** by **Superwhisper** (ASR text normalizer, Qwen3-0.6B finetune) — the zoo's **first ASR post-processor**: raw transcript → clean written text (fillers, false starts, punctuation, inverse text normalization). Completes the dictation path behind Parakeet / Nemotron-ASR | [🤗 S1-mini-CoreAI](https://huggingface.co/mlboydaisuke/S1-mini-CoreAI) | int8lin 759 MB, oracle gate **16/16** + **task gate 13/14** vs released weights, **268.4 decode / 4161 prefill tok/s M4 Max**; **int4 = no-go — passes the same 16/16 oracle and corrupts digits** ($23,450→$2,345), caught only by the task gate. **iPhone 17 Pro: 62.4 decode / 69.0 prefill tok/s, 276/276 + 27/27 token-exact vs Mac, PB_G=1024 gate passes** — with a hard iOS ceiling of prompt+generated < 1024 tokens (shipped GrowingKVCache guard; chunk input to ~450–500) |
| [`kokoro-82m.md`](kokoro-82m/README.md) | Kokoro-82M (StyleTTS2 + iSTFTNet text-to-speech, hexgrad) — zoo's **first TTS** (3 graphs + host DSP, non-autoregressive 24 kHz) | [🤗 Kokoro-82M-CoreAI](https://huggingface.co/mlboydaisuke/Kokoro-82M-CoreAI) | 82M fp32 (~335 MB), spectral gate corr **0.999** vs patched torch; full utterance **~0.75 s M4 Max CPU**; 28 English voices |
| [`vjepa2.md`](vjepa2/README.md) | V-JEPA 2 ViT-L SSv2 (self-supervised video world model + action head, Meta) — zoo's **first video understanding model** | [🤗 VJEPA2-ViTL-SSv2-CoreAI](https://huggingface.co/mlboydaisuke/VJEPA2-ViTL-SSv2-CoreAI) | 375M fp16 (~708 MB), engine cos **0.999996** top-5 identical + semantic motion gate; **~160 ms/clip M4 Max GPU · ~0.34 s iPhone 17 Pro (AOT)** |
| [`unlimited-ocr.md`](unlimited-ocr/README.md) | Unlimited-OCR 3B-A0.5B MoE (document OCR → structured markdown, baidu) — zoo's **first doc-OCR**, stock runtime | [🤗 Unlimited-OCR-CoreAI](https://huggingface.co/mlboydaisuke/Unlimited-OCR-CoreAI) | vision fp16 + R-SWA MoE decoder sym8 (~4.5 GB), decode 0 flips / 9 steps vs fp32 oracle, **flat 12.7 ms/token (~79 tok/s) M4 Max** |
| [`glm-ocr.md`](glm-ocr/README.md) | GLM-OCR 0.9B (document OCR, GLM-4.V small — zhipu) — zoo's **2nd doc-OCR** (`image_embeds` + M-RoPE decode) | [🤗 GLM-OCR-CoreAI](https://huggingface.co/mlboydaisuke/GLM-OCR-CoreAI) | vision fp16 + GLM decoder int8, CogViT tower + GLM-4 decoder on the rope-shift rider |
| [`mineru.md`](mineru/README.md) | MinerU2.5-Pro 1.2B (whole-page document parsing → structured markdown, opendatalab) — zoo's **first whole-page auto-structuring** doc-OCR (2-stage layout + per-region recognition in one stock Qwen2-VL) | [🤗 MinerU2.5-Pro-CoreAI](https://huggingface.co/mlboydaisuke/MinerU2.5-Pro-CoreAI) | vision fp16 + Qwen2 int8lin, torch ladder **706/706** vs HF. **Single-pass** (768 grid) on iPhone 17 Pro ~4 s/page (chunked prefill). **2-stage** (1036² layout + 768 recognition, tables → `<table>` HTML via OTSL) byte-identical to reference in the ReadDoc Mac app |
| [`llada-8b.md`](llada-8b/README.md) | LLaDA-8B d3LLM (masked-diffusion LLM, GSAI-ML/d3LLM) — zoo's **first diffusion LLM** (parallel canvas denoising, host semi-AR loop) | [🤗 LLaDA-8B-dLLM-CoreAI](https://huggingface.co/mlboydaisuke/LLaDA-8B-dLLM-CoreAI) | int4 blk32 + int8 head **4.9 GB** (S=256), layer/logits cos ≈1.0 + temp-0 decode gate vs official; **~38–40 tok/s M4 Max (NFE 11)**; Mac |
| [`flux2-klein`](flux2-klein/README.md) | text → image + in-context editing; int4, Mac (BFL) | [🤗 FLUX.2-klein-4B-CoreAI](https://huggingface.co/mlboydaisuke/FLUX.2-klein-4B-CoreAI) | card adopted from the model page |
| [`voxcpm`](voxcpm/README.md) | tokenizer-free TTS, iPhone + Mac (OpenBMB) — kit `voxcpm-0.5b` | [🤗 VoxCPM-0.5B-CoreAI](https://huggingface.co/mlboydaisuke/VoxCPM-0.5B-CoreAI) | card adopted from the model page |
| [`voxcpm2`](voxcpm2/README.md) | tokenizer-free TTS, second generation — kit `voxcpm2-2b` | [🤗 VoxCPM2-CoreAI](https://huggingface.co/mlboydaisuke/VoxCPM2-CoreAI) | card adopted from the model page |
| [`pocket-tts`](pocket-tts/README.md) | pocket-tts (Kyutai) — the zoo's **first Kyutai model** and **first Mimi conversion**: AR flow-matching LM over Mimi latents + one-step flow decoder + streaming Mimi decoder, 8 voices, no G2P layer; ported by [Rahul Rachuri](https://github.com/RahulRachuri) | [🤗 pocket-tts-coreai](https://huggingface.co/rahulrachuri/pocket-tts-coreai/tree/ad989309a5781c403113f9653f04a7d27c642c21) | 5 bundles, every graph cos 1.000000 on `cpu_only` and `gpu` (Mimi bit-identical), ASR round-trip **1.38 % WER** on a 148-word paragraph and 4.27 % over a 302-sentence sweep; **iPhone 17 Pro Max 7.8× real-time fp16, 169 MB peak, load 0.4 s**; M4 Pro 8.5× |
| [`stable-audio-open-small`](stable-audio-open-small/README.md) | text → music/audio, 13× realtime (Stability AI) — kit `stable-audio-open-small` | [🤗 Stable-Audio-Open-Small-CoreAI](https://huggingface.co/mlboydaisuke/Stable-Audio-Open-Small-CoreAI) | card adopted from the model page |
| [`qwen2.5-omni-audio`](qwen2.5-omni-audio/README.md) | audio understanding (Omni thinker's audio tower) — kit `qwen2.5-omni-3b-audio` | [🤗 Qwen2.5-Omni-3B-Audio-CoreAI](https://huggingface.co/mlboydaisuke/Qwen2.5-Omni-3B-Audio-CoreAI) | card adopted from the model page |
| [`rwkv7-goose`](rwkv7-goose/README.md) | attention-free RNN, constant-memory decode | [🤗 RWKV7-Goose-1.5B-CoreAI](https://huggingface.co/mlboydaisuke/RWKV7-Goose-1.5B-CoreAI) | card adopted from the model page |
| [`qwen3.5-4b`](qwen3.5-4b/README.md) | the Qwen3.5 family's Mac-class dense size | [🤗 qwen3.5-4B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-4B-CoreAI) | card adopted from the model page |
| [`adcsr`](adcsr/README.md) | single-step ×4 image super-resolution — kit `adcsr-x4` | [🤗 AdcSR-CoreAI](https://huggingface.co/mlboydaisuke/AdcSR-CoreAI) | card adopted from the model page |

## The matrix (every meaningful platform × compute-unit cell, greedy, top-1 vs HF)

<!-- Mac column RELEASE-VERIFIED 2026-06-10 (R2, ondevice/MACOS_RELEASE_README.md).
     qwen static iOS GPU 27.7 (ctx 2048, release config) = 2026-06-10 RELEASE-build device
     measurement (ctx-256 export measured 30.4).
     gemma4 iOS GPU 22 + ANE 6 = 2026-06-10 hands-on re-measure in the RELEASE chat app
     (int4km monolith; instrumented run 22.5, core 39ms / head 2ms — the earlier 17.7 was the
     AOT-harness number; the Release-confirm TODO is resolved). -->

| | macOS GPU (M4 Max) | iOS GPU | iOS ANE |
|---|---|---|---|
| **Gemma 4 E2B** | ✅ 8/8 · 56.6–59.0 tok/s (int8 kernels) | ✅ 8/8 · **22 tok/s** (int4-k-means kernels) | ✅ 8/8 · 6 tok/s (int8 chunks) |
| **Qwen3.5 0.8B** | ✅ 8/8 · 58.5 (int8 dynamic) | ✅ **27.7** (fp16 static, ctx 2048) / 12.5 (int8 dynamic) | ✅ 14.7 (int8 dynamic); static ✗ this beta (fp16 SSM recurrence) |

macOS ANE is intentionally out of scope (the runtime auto-prefers GPU on Mac for these
structures, and the Mac GPU dominates it anyway).

Parity is measured against the Hugging Face eager reference (cosine + top-1 argmax on a fixed
prompt): conversion on macOS, then re-verified end-to-end on-device (iPhone 17 Pro, iOS 27 beta).
Device numbers are int8, greedy, prompt "What is the capital of France?" / "The capital of France
is" → "Paris".
