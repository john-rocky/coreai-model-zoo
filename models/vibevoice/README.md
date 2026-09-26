# VibeVoice-Realtime-0.5B — Core AI

The zoo's first **multi-speaker / dialogue (podcast-style)** TTS.
[`microsoft/VibeVoice-Realtime-0.5B`](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)
(MIT, EN/ZH) as five Core AI graphs plus a host generate loop — iPhone and Mac, all fp16 (the iPhone rows
below are the h18p AOT bundles on a 17 Pro).

> **Not a "first on-device" claim.** Other CoreML/GGUF VibeVoice ports exist. What's new here is the
> zoo's first *multi-speaker* TTS, app-integrated, and the other half of a **generate → diarize**
> loop with the shipped [Streaming Sortformer](../sortformer-diar/README.md) diarizer: synthesize a
> conversation, then have the zoo tell you who spoke when.

## Use it

```swift
import CoreAIKit

let dialogue = try await KitDialogue(catalog: "vibevoice-realtime-0.5b")
let (audio, turns) = try await dialogue.perform("""
    Speaker 1: Did you know this runs entirely on the phone?
    Speaker 2: No cloud at all? That is wild.
    """)
// audio.samples @ 24 kHz; turns[i].voice tells you which preset spoke
// one line, one voice: try await dialogue.speak("Hello from Core AI.")
```

25 voice presets ship with the model (`dialogue.voices`). Free text needs no torch and no
reference audio: the Qwen2.5 tokenizer plus an mmapped fp16 embedding table, with the voice
arriving as a pre-computed prefill KV cache.

## How it works

Dual Qwen2.5 LM — a **4-layer** text context LM (norm = Identity) and a **20-layer** speech trunk —
emit one latent per 7.5 Hz frame. A 4-layer adaLN **diffusion head** denoises that latent (DDPM
cosine, v-prediction, DPMSolver++ 5 steps, CFG 1.5), the connector feeds it back as the next input
embedding, and a causal-conv **acoustic VAE decoder** renders 3200 samples (24 kHz) per frame. This
is "next-token diffusion": the LM predicts *what to say* in latent space, the head makes it audio.

Five graphs + a host loop:

| graph | shape | note |
|---|---|---|
| main LM | q=1 decode, KV-stateful, cache 512 | 114 MB |
| tts LM | q=1 decode, KV-stateful, cache 512 | 569 MB — one graph serves the positive **and** the CFG-negative stream |
| diffusion head | `(noisy[2,64], t[2]) → [2,64]` | 80 MB, the 2 = cond + uncond |
| connector | `latent[1,1,64] → embed[1,1,896]` | 1.7 MB |
| acoustic decoder | `latents[1,64,T] → audio[1,1,3200·T]` | 656 MB, whole-sequence |

The host runs the AR loop, the DDPM sampler, the EOS classifier, the token-embedding lookup, and —
the differentiator — **multi-speaker turn switching**: each `Speaker N:` turn is generated from its
own voice preset with fresh noise, then the turns are concatenated. No multi-speaker prefill and no
acoustic *encoder* are needed; the 25 upstream voice presets already carry a pre-computed prefill KV.

## Traps worth knowing

- **fp16 only.** int8 LMs diverge inside the speech feedback loop (latent min cos 0.187, early EOS).
  int8 main LM alone is 0.977 — still not good enough. The diffusion head is fp16-sensitive in the
  other direction: pure-torch fp16 collapses to cos 0.79 (RMSNorm/adaLN reductions), while Core AI
  keeps those in fp32 — so the *host reference* must run fp32 to compare fairly.
- **Do not request `expectFrequentReshapes` on iOS.** Every graph here is fixed-shape. Asking for the
  hint makes the runtime skip the AOT specialization and compile on device, which **segfaults inside
  the MPSGraph AICode compiler** (`getInitializedAICodeBytecodeWithPayloadPrefix`) with no error
  message. Dropping the hint: 6 graphs load in 2.6 s and the gate passes.

## Verification

| gate | result |
|---|---|
| diffusion head / connector, engine vs oracle | cos **0.999999** / **1.000000** |
| acoustic decoder (T=30 / T=64) vs non-stream golden | cos **1.000004** / **1.000005** |
| main LM / tts LM decode, engine vs torch | cos **0.999999** / **0.999996** |
| Python E2E on all 5 engines vs the upstream streamed wav | latent min cos **0.999198**, wav **0.999479** |
| **iPhone 17 Pro** (A19 Pro, AOT h18p, GPU) vs the golden | cos **0.998308** |

On device: 6 graph loads in **2.6 s** (warm), **24 latents / 3.20 s of audio in 2.3 s = 10.6 tok/s
≈ 1.4× real-time**.

Reproduce — [`conversion/vibevoice`](../../conversion/vibevoice) (`KICKOFF.md` has the full ladder):

```bash
PY=coreai-models/.venv/bin/python
$PY conversion/vibevoice/export_diffusion_head.py --dtype fp16      # head + connector
$PY conversion/vibevoice/export_backbone.py --mode fp16 --which both
$PY conversion/vibevoice/export_decoder.py --tframes 64 --dtype fp16
$PY conversion/vibevoice/host_e2e.py --backend engine --lm-mode fp16   # E2E gate
$PY conversion/vibevoice/host_multispeaker.py                          # dialogue demo
bash conversion/vibevoice/sideload_ios.sh <device-udid>                 # then VIBEVOICE_SELFTEST=1
```

- 🤗 [VibeVoice-Realtime-0.5B-CoreAI](https://huggingface.co/mlboydaisuke/VibeVoice-Realtime-0.5B-CoreAI)
  — `macos/` and `ios/` each hold the 5 JIT `.aimodel` (every iPhone generation specializes them on its first
  load), `ios-h18p/` the 5 `.h18p.aimodelc` compiled for the iPhone 17 Pro (that phone only; they sat in `ios/`
  until Hub revision `b4480f87`, 2026-09-26) + voice presets + host embedding table. On the iPhone 18 Pro (h19p)
  the five JIT graphs loaded in 0.04–1.78 s on a fresh install (decoder 1.78 s, ttslm 1.36 s) and in 0.01–0.50 s on
  relaunch (load-only check, 2026-09-26).
- Base model: [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) (MIT).
