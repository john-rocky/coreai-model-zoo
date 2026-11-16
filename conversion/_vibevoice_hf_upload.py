"""Stage + upload the VibeVoice-Realtime-0.5B Core AI bundles to HF.

Stages the 5 macOS fp16 `.aimodel`, their iOS AOT `.h18p.aimodelc`, the voice presets, the host
embedding table and the compact device self-test bundle into /tmp/vibevoice_hf, then uploads to
mlboydaisuke/VibeVoice-Realtime-0.5B-CoreAI.

    coreai-models/.venv/bin/python conversion/_vibevoice_hf_upload.py
"""
import os, shutil
os.environ["HF_HUB_DISABLE_XET"] = "1"          # xet stalls/panics on these bundles
from pathlib import Path
import numpy as np
from safetensors.torch import load_file
from huggingface_hub import HfApi
from _paths import hf_snapshot

REPO = "mlboydaisuke/VibeVoice-Realtime-0.5B-CoreAI"
HERE = Path(__file__).resolve().parent
CONV = HERE / "vibevoice"
ART, IOS = CONV / "artifacts", CONV / "artifacts_ios"
VOICES = CONV / "_code" / "demo" / "voices" / "streaming_model"
SNAP = hf_snapshot("microsoft/VibeVoice-Realtime-0.5B", "model.safetensors")
STAGE = Path("/tmp/vibevoice_hf")
GRAPHS = ["vibevoice_mainlm_fp16_decode_cl512", "vibevoice_ttslm_fp16_decode_cl512",
          "vibevoice_diffusion_head_fp16", "vibevoice_connector_fp16", "vibevoice_decoder_fp16_t64"]

CARD = '''---
license: mit
library_name: coreai
pipeline_tag: text-to-speech
base_model: microsoft/VibeVoice-Realtime-0.5B
language: [en, zh]
tags: [core-ai, coreaikit, tts, multi-speaker, dialogue, podcast, diffusion, on-device, apple]
---

# VibeVoice-Realtime-0.5B — Core AI

[`microsoft/VibeVoice-Realtime-0.5B`](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)
(MIT) converted to **Apple Core AI** — the [zoo](https://github.com/john-rocky/coreai-model-zoo)'s
first **multi-speaker / dialogue (podcast-style)** TTS. iPhone (AOT) + Mac, all-fp16.

Not a "first on-device VibeVoice" claim — other CoreML/GGUF ports exist. What this is: the zoo's
first multi-speaker TTS, app-integrated, and the other half of a **generate -> diarize** loop with
the zoo's [Streaming Sortformer](https://huggingface.co/mlboydaisuke/Streaming-Sortformer-Diar-CoreAI)
diarizer.

## Architecture

Dual Qwen2.5 LM (**4-layer** text context LM, norm = Identity + **20-layer** speech trunk) ->
per-frame **next-token diffusion** (4-layer adaLN head, DDPM cosine, v-prediction, DPMSolver++
5-step, CFG 1.5) -> causal-conv acoustic VAE decoder (7.5 Hz latent -> 24 kHz, 3200 samples/frame).
The LM predicts one latent per frame; the diffusion head denoises it; the VAE decoder renders audio.
Multi-speaker output is **host turn-switching**: each `Speaker N:` turn is generated from its own
voice preset and the turns are concatenated — no multi-speaker prefill, no acoustic encoder.

## Contents

| path | what |
|---|---|
| `macos/vibevoice_mainlm_fp16_decode_cl512.aimodel` | context LM, q=1 decode, KV-stateful, cache 512 (114 MB) |
| `macos/vibevoice_ttslm_fp16_decode_cl512.aimodel` | speech trunk, same shape; also drives the CFG-negative stream (569 MB) |
| `macos/vibevoice_diffusion_head_fp16.aimodel` | prediction head, `(noisy[2,64], t[2]) -> [2,64]` (80 MB) |
| `macos/vibevoice_connector_fp16.aimodel` | acoustic connector, `latent[1,1,64] -> embed[1,1,896]` (1.7 MB) |
| `macos/vibevoice_decoder_fp16_t64.aimodel` | acoustic VAE decoder, `latents[1,64,64] -> audio[1,1,204800]` (656 MB) |
| `ios/*.h18p.aimodelc` | the same five, AOT-compiled for A19 (h18p), GPU |
| `voices/*.pt` | 25 upstream voice presets (EN/ZH/…): pre-computed prefill KV, so no acoustic encoder is shipped |
| `coreai_host/voices/<name>/` | the same 25 presets as flat fp16 blobs a Swift host reads directly (no torch) |
| `coreai_host/glue/` | type embeddings, EOS classifier, DPMSolver++ schedule, Qwen2.5 tokenizer |
| `coreai_host/embed/embed_tokens_fp16.bin` | `(151936, 896)` fp16 embedding table, mmapped for the host token lookup |
| `device_bundle/` | compact host inputs + `golden.f32` for the on-device self-test |

**fp16 is required.** int8 LMs diverge inside the speech feedback loop (min cos 0.187, early EOS);
the diffusion head is fp16-sensitive too (pure-torch fp16 collapses to 0.79 — Core AI keeps the
RMSNorm/adaLN reductions in fp32, so the host DDPM reference must run fp32).

**Fixed shapes only.** Every graph is static (q=1 decode, fixed-T decoder), so the runtime must
**not** be given the `expectFrequentReshapes` hint on iOS: it makes the runtime skip the AOT
specialization and compile on device, which segfaults inside the MPSGraph AICode compiler.

## Gates

| gate | result |
|---|---|
| diffusion head / connector, engine vs oracle | cos **0.999999** / **1.000000** |
| acoustic decoder (T=30 / T=64), engine vs non-stream golden | cos **1.000004** / **1.000005** |
| main LM / tts LM decode, engine vs torch | cos **0.999999** / **0.999996** |
| Python E2E on all 5 engines vs the upstream streamed wav | latent min cos **0.999198**, wav cos **0.999479** |
| **iPhone 17 Pro** (A19 Pro, AOT h18p, GPU) vs the golden | cos **0.998308** |

On device: 6 graph loads in **2.6 s** (warm), **24 latents / 3.20 s of audio in 2.3 s = 10.6 tok/s
~ 1.4x real-time**.

## Use it

```swift
import CoreAIKit

let dialogue = try await KitDialogue(catalog: "vibevoice-realtime-0.5b")
let (audio, turns) = try await dialogue.perform("""
    Speaker 1: Did you know this runs entirely on the phone?
    Speaker 2: No cloud at all? That is wild.
    """)
```

Swift host reference: `ondevice/VibeVoiceRunner` (Mac) and `VibeVoiceSelfTest.swift` in the zoo's
[coreai-audio](https://github.com/john-rocky/coreai-model-zoo/tree/main/apps/coreai-audio) app —
raw Core AI stateful-KV loop + a Swift DPMSolver++ sampler. Python host + conversion recipe:
[`conversion/vibevoice`](https://github.com/john-rocky/coreai-model-zoo/tree/main/conversion/vibevoice)
(`host_e2e.py` = the full generate loop, `host_multispeaker.py` = the dialogue demo).

Base model: [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) (MIT).
EN/ZH. *Community port — not an Apple model.*
'''


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "macos").mkdir(parents=True)
    (STAGE / "ios").mkdir(parents=True)
    for g in GRAPHS:
        shutil.copytree(ART / g / f"{g}.aimodel", STAGE / "macos" / f"{g}.aimodel")
        shutil.copytree(IOS / f"{g}.h18p.aimodelc", STAGE / "ios" / f"{g}.h18p.aimodelc")
    shutil.copytree(VOICES, STAGE / "voices")
    shutil.copytree(CONV / "device_bundle", STAGE / "device_bundle")

    # Swift-host assets (pack_voice_presets.py): flat voice caches + glue + tokenizer. Everything
    # CoreAIKit's KitDialogue resolves lives under coreai_host/ — ModelStore downloads subtrees,
    # so the embedding table goes in a directory of its own, not at the repo root.
    host = CONV / "voices_coreai"
    if host.exists():
        shutil.copytree(host, STAGE / "coreai_host")
    (STAGE / "coreai_host" / "embed").mkdir(parents=True, exist_ok=True)

    # host token lookup: the LMs take embeddings, so a free-text host needs the embedding table
    emb = load_file(SNAP)["model.language_model.embed_tokens.weight"].to(dtype=None).float().numpy()
    emb.astype(np.float16).tofile(STAGE / "coreai_host" / "embed" / "embed_tokens_fp16.bin")
    print(f"embed_tokens {emb.shape} -> fp16 {(emb.size * 2) / 1e6:.0f} MB")

    (STAGE / "README.md").write_text(CARD)
    total = sum(f.stat().st_size for f in STAGE.rglob("*") if f.is_file()) / 1e9
    print(f"staged {total:.2f} GB in {STAGE}")

    api = HfApi()
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=REPO, folder_path=str(STAGE),
                      commit_message="VibeVoice-Realtime-0.5B Core AI: 5 macOS .aimodel + iOS h18p AOT + voice presets")
    print(f"uploaded -> https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
