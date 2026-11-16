# VibeVoice-1.5B → Core AI — PLAN (2026-07-06)

zoo's **first multi-speaker / dialogue (podcast) TTS**. Selected via `/next-target`; precheck 🟢 GREEN.
Rubric = `strategy/SHIP_RUBRIC.md`.

**target**: HF [`microsoft/VibeVoice-1.5B`](https://huggingface.co/microsoft/VibeVoice-1.5B) — **MIT**,
non-gated, ~5.4 GB BF16, ~3B params. Synthesizes **up to 4 speakers, up to 90 min** dialogue, 24 kHz,
EN/ZH only. Variants: `VibeVoice-Realtime-0.5B` (streaming; iPhone-friendly), `VibeVoice-ASR` (who/when/what — different model).

## Why (START gate)
GAP ✅ (multi-speaker/podcast gen not in Apple stock) · EDGE 🟡→OK (obscure CoreML/GGUF ports exist,
but not pipelined/ANE/app-integrated — same edge shape as VoxCPM/dots) · FIRST ✅ (zoo's 4 TTS are all
single-utterance) · DEVICE ✅ (~2 GB int4-trunk/fp16-heads) · QUALITY 🟢 (VoxCPM/dots quant split).
**Do NOT claim "first on-device"** — ship as "zoo's first multi-speaker/dialogue TTS, app-integrated,
paired with the shipped Sortformer diarization (generate a conversation → diarize it)".

## Architecture (from config.json)
- **decoder** = `qwen2` hidden 1536 / ffn 8960 / 28 layers = **stock Qwen2.5-1.5B** (same backbone as
  dots.tts → export ~free, reuse scaffolding).
- **diffusion_head** = DDPM cosine, 20 inference steps, 4 layers, hidden 1536, ffn_ratio 3.0 (~123M).
- **acoustic_tokenizer** = VAE dim 64, causal, decoder ratios [8,5,5,4,2,2] → 24 kHz (~340M).
- **semantic_tokenizer** = VAE dim 128, causal, encoder depths 3-3-3-3-3-3-8 (~340M).
- next-token diffusion: the LM predicts a latent per 7.5 Hz frame; the diffusion head denoises it to
  the acoustic latent; the acoustic VAE decoder → 24 kHz audio.

## Export boundary (maps 1:1 onto conversion/{voxcpm,dots_tts})
1. **Qwen2.5-1.5B decoder** — stock export, int4; host AR loop (reuse dots).
2. **diffusion head** (4-layer DDPM) — graph; run the DDPM sampler in host (VoxCPM sampler-capture idiom). fp16.
3. **acoustic VAE decoder** — vocoder graph (AudioVAE/BigVGAN idiom). fp16.
4. **semantic tokenizer** — conditioning input side; export what generation needs.
5. **host** — AR loop + DDPM sampling + **multi-speaker turn conditioning (NEW = the differentiator)**
   + Swift host (VoxCPM/dots `StatefulGraphModel` / streaming / iPhone AOT h18p / coreai-audio Voice tab).

## RETARGET → VibeVoice-Realtime-0.5B (2026-07-06)
Current GitHub repo dropped the 1.5B non-streaming `generate`; the 0.5B streaming model runs out of
the box (`VibeVoiceStreamingForConditionalGenerationInference` + `VibeVoiceStreamingProcessor` + 25
voice-preset `.pt` in `_code/demo/voices/streaming_model/`). 0.5B = newer / smaller / streaming /
iPhone-ideal. **Oracle GREEN**: Mac MPS RTF 1.19x, `_oracle_out/*.wav`. env = `_oracle/.venv` (uv,
py3.11, torch2.12.1, transformers==4.51.3, `-e ./_code`). ⚠️ patch `weights_only=False` for voice `.pt`.

## Export boundary — CAPTURED (capture_arch.py, 0.5B streaming)
| module | params | I/O (measured) | plan |
|---|---|---|---|
| `model.language_model` (Qwen2.5 h896) | 195.8M | → hidden `(1,T,896)` | context LM, int4 |
| `model.tts_language_model` | 434.4M | → hidden `(1,T,896)` | speech AR LM (trunk), int4 |
| `model.prediction_head` (diffusion) | 42.1M | `(noisy[2,64], t[2])` → `(2,64)` | 5-step DDPM in host, fp16 (2 = CFG cond+uncond) |
| `model.acoustic_tokenizer.decoder` | 343.7M | `(1,64,1)` → `(1,1,3200)` | vocoder 7.5 Hz→24 kHz (3200/frame), fp16 |
| `model.tts_eos_classifier` | 0.8M | `(1,896)` → `(1,1)` | EOS decision |
- acoustic **encoder** (343M) NOT exported — reference audio is pre-encoded into the voice `.pt`.
- inputs: `input_ids`(main), `tts_lm_input_ids`, `tts_text_ids`, `speech_input_mask`; voice prefill
  cache keys `{lm, tts_lm, neg_lm, neg_tts_lm}` (cond + CFG-negative). speech token = 3200 samples.
- **exact VoxCPM/dots idiom** (LM×2 + diffusion head + VAE decoder + host AR/DDPM loop), ~1GB fp16.

## Plan (standard port order)
0. ✅ precheck GREEN. ✅ oracle GREEN (Mac). ✅ arch trace captured (export boundary above).
1. ✅ **reference oracle** — `oracle.py` golden per-stage fixtures (artifacts/oracle_ref.npz + oracle.wav,
   5-step DDPM, en-Frank_man, 30 latents / 3.73 s / 24 kHz). `oracle_decode_ref.py` (non-stream golden,
   proved non-stream==stream cos 1.0), `oracle_lm_ref.py` (fresh-prefill LM hiddens).
2. ✅ **plain-torch re-author** — torch_overlays.py (diffusion head, connector), decoder_ref.py (causal
   conv VAE), backbone.py (Qwen2 static-KV, adapted from dots). All cos=1.0 vs oracle in fp32.
3. ✅ **Core AI export + engine gate (ALL 5 graphs pass ≥0.9999):**
   * diffusion head fp16  `export_diffusion_head.py` — engine cos **0.999999** (80 M). ⚠️ fp16-SENSITIVE:
     pure-torch fp16 collapses to 0.79 (RMSNorm/adaLN reductions) but Core AI keeps them fp32 → host
     DDPM reference MUST run fp32.
   * connector fp16 — cos **1.000000** (1.7 M).
   * decoder fp16  `export_decoder.py` — cos **1.000004** (656 M, largest). Whole-seq non-streaming.
     quant-tolerant. ⚠️ dynamic-T single graph hits macOS-27 dynamic-query JIT (garbage off-ref shape /
     ANE W>65536) → ship needs AOT + `expectFrequentReshapes=true` (also req. for iPhone). fixed-T works.
   * main LM int8  `export_backbone.py` — decode(q=1) cos **0.999995** (61 M, 4L norm=Identity).
   * tts LM int8 — decode(q=1) cos **0.999995** (303 M, 20L norm=real). ONE q=1 graph/LM serves the whole
     loop (causal: q=W window == W q=1 steps); tts graph reused for the pos + neg (CFG) KV streams.
4. ✅ **host E2E** (`host_e2e.py`, coreai venv): generate() replicated with all 5 engines. torch backend
   wav cos **1.000009** (loop logic proven); engine backend (fp16 LMs) latent min 0.999198, **wav vs oracle
   STREAMED cos 0.999473** (3.73 s, rms/peak identical). DDPM/DPMSolver sampler, `.pt` voice-cache KV
   seeding (main 72 / tts 253 / negtts 1), CFG, EOS, connector feedback, whole-seq decode all validated.
   Deterministic gate = feed oracle randn{i} noise. Host non-engine: embed_tokens lookup, tts_input_types,
   eos_classifier (dump_e2e_seed.py bridges the two venvs). ⚠️ **int8 tts LM DIVERGES in the AR feedback
   loop (min 0.187, early EOS) — fp16 LMs required for the speech feedback path** (int8 main = 0.977, also
   fp16). **ship = all-fp16 ≈1.42 GB** (+ embed_tokens 272 MB host lookup).
   ✅ **multi-speaker dialogue** (`host_multispeaker.py`, the differentiator) = host turn-switching: each
   "Speaker N:" turn generated with its own single-speaker .pt seed + fresh noise, concatenated. 2-speaker
   9.7 s conversation, all on Core AI engines. No multi-speaker-prefill / acoustic-encoder needed. Pairs
   with shipped Sortformer (generate → diarize).
5. ✅ **Swift host + iPhone device gate PASS.** `ondevice/VibeVoiceRunner` (Mac cos 0.999) +
   `coreai-audio` `VibeVoiceSelfTest.swift` (raw CoreAI stateful KV + Swift DPMSolver++). AOT h18p
   (`artifacts_ios/`), sideload (`sideload_ios.sh`) + `VIBEVOICE_SELFTEST=1`. **iPhone 17 Pro:
   `[VV] gate vs golden cos=0.992889 PASS`, 6 graphs load 2.5 s (warm), 24 latents / 3.20 s audio in
   2.3 s = 10.3 tok/s ≈ 1.4× realtime.** (GUI print() misses devicectl --console on the cold first run;
   read the result from `VV_RESULT=<container>/Documents/vv_result.txt` via `devicectl copy from`.)
6. ✅ **shipped 2026-07-23** — HF [`mlboydaisuke/VibeVoice-Realtime-0.5B-CoreAI`](https://huggingface.co/mlboydaisuke/VibeVoice-Realtime-0.5B-CoreAI)
   (5 macOS `.aimodel` + 5 iOS h18p `.aimodelc` + 25 voice presets + host embedding table +
   `device_bundle/`), zoo card `models/vibevoice/README.md` + README row, knowledge
   `knowledge/vibevoice-multispeaker-tts.md`. Upload script = `conversion/_vibevoice_hf_upload.py`.
   Re-gated end-to-end on the day of the ship (the earlier bundles were lost to a disk sweep):
   all 5 graphs re-exported, host E2E wav cos **0.999479**, iPhone 17 Pro **cos 0.998308, 10.6 tok/s**.
   ⚠️ **`expectFrequentReshapes` must be OFF on iOS** — with the hint the runtime skips the AOT
   specialization and compiles on device, segfaulting in the MPSGraph AICode compiler (fixed-shape
   graphs only here). Also: `export_decoder.py --tframes 64` now tiles the 30-latent golden instead of
   clamping T, so the T the device wants is actually the T that gets exported.

⚠️ HF DL needs `HF_HUB_DISABLE_XET=1`. env = `coreai-models/.venv`.

## Differentiator (sharpen before ship)
Multi-speaker is zoo-unique. Pair with the **shipped Sortformer diarization**: a "generate a
conversation → diarize who-said-what" loop entirely inside coreai-audio, all on-device. EN/ZH only →
lives beside dots.tts (multilingual single-utterance): VibeVoice = multi-speaker dialogue, dots = 24-lang.
