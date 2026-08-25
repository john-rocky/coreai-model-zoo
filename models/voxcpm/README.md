# VoxCPM-0.5B — Core AI (on-device, iPhone + Mac)

[OpenBMB's **VoxCPM-0.5B**](https://huggingface.co/openbmb/VoxCPM-0.5B) converted to Apple's **Core AI** engine, running **fully on-device** — iPhone (Apple Neural Engine / GPU, AOT-compiled) and Apple-silicon Mac. No network, no server.

VoxCPM is not a classic vocoder TTS: it pairs a **MiniCPM4 language-model backbone** with a **LocDiT flow-matching diffusion** head and an **AudioVAE**, generating speech through a continuous (token-rate) diffusion loop. This repo ships the whole stack as Core AI model bundles plus the small host-side glue the runtime needs.

- **Output:** 16 kHz mono
- **License:** Apache-2.0 (commercial-friendly), inherited from the base model
- **Quantization:** weight-only **int8** on the two LM backbones (the size driver); the diffusion decoder, feature encoder, and AudioVAE stay **fp16** — the continuous-feedback path is quantization-sensitive (the same split [mlx-community/VoxCPM2](https://huggingface.co/collections/mlx-community/voxcpm) uses).

<!-- gen-cards:use-it begin id=voxcpm-0.5b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
![VoxCPM 0.5B demo](https://huggingface.co/mlboydaisuke/VoxCPM-0.5B-CoreAI/resolve/main/demo.gif)
*VoxCPM 0.5B on iPhone 17 Pro — the zoo's coreai-audio app, real speed.*

## Use it

⚡ **One line** — this model is the default behind the kit's task op
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let audio = try await CoreAI.speak(text)
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [Speak runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Speak)
(GUI + CLI, one app for every text-to-speech model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/Speak/Speak.xcodeproj
# → Run, then pick "VoxCPM 0.5B" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/Speak
swift run speak-cli --model voxcpm-0.5b --text "Hello from Core AI." --output hello.wav
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let speaker = try await KitSpeaker(catalog: "voxcpm-0.5b")
let audio = try await speaker.synthesize(text)
// audio.samples: 16 kHz mono PCM in [-1, 1] — play it or write a WAV
```

The take-home is [`Examples/Speak/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/main/Examples/Speak/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; the CLI is an argument shell over it, and
the GUI drives the same `KitSpeaker(catalog:)` and plays the samples.
Live playback? `synthesizeStreaming(_:onChunk:)` hands you ~0.5 s chunks as they decode,
so audio starts before the whole clip exists. The WAV container is your app's territory
(the runner ships a 20-line writer).

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` → product **CoreAIKit**
- Info.plist: none needed
- Entitlements: none needed
- First run downloads the model — 1.4 GB (Mac) / 1.7 GB (iPhone) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Contents

| Path | What |
|------|------|
| `macos/voxcpm_base_int8_decode_cl512/` | LM backbone (MiniCPM4, 24L), int8, static-KV decode — JIT `.aimodel` for Mac |
| `macos/voxcpm_res_int8_decode_cl512/` | Residual LM (6L), int8 |
| `macos/voxcpm_base_int8_prefill_t32/` | LM backbone q=32 batched prefill — seeds the KV cache in one pass for fast time-to-first-audio, int8 |
| `macos/voxcpm_res_int8_prefill_t32/` | Residual LM q=32 batched prefill, int8 |
| `macos/voxcpm_feat_decoder_fp16/` | LocDiT CFM diffusion decoder (10-step euler + CFG, unrolled), fp16 |
| `macos/voxcpm_feat_encoder_fp16/` | LocEnc + projection (per-frame feedback embed), fp16 |
| `macos/voxcpm_vocoder_fp16_t12/` | AudioVAE decoder (DAC-style, 640× upsample), fp16 |
| `ios/*.h18p.aimodelc/` | The same bundles (5 + the 2 int8 prefill), AOT-compiled for iOS (h18p) |
| `voxcpm_host_glue/` | Token-embedding table + dit/FSQ/stop-head weights (run host-side via Accelerate) |
| `tokenizer/` | Llama tokenizer (`tokenizer.json` + config) |

A q=32 batched-prefill bundle **is** shipped, for fast time-to-first-audio: it seeds the KV cache in a single pass instead of looping the decode bundle once per text token (costly on the bandwidth-bound A19). Text longer than 32 tokens falls back to the bit-identical prefill-via-decode loop, so length stays unbounded.

## Usage

Easiest path is the **[coreai-model-zoo](https://github.com/john-rocky/coreai-model-zoo)** `coreai-audio` app (the "Voice" tab) and **[CoreAIKit](https://github.com/john-rocky/coreai-kit)**:

```swift
import CoreAIKit

let tts = try await VoxCPMTTS(paths: .standard(artifactsRoot: modelRoot))   // macOS (.aimodel)
// let tts = try await VoxCPMTTS(paths: .aot(root: modelRoot, arch: "h18p")) // iOS (.aimodelc)
let pcm = try await tts.synthesize("On device speech synthesis, running entirely on your iPhone.")
// pcm: [Float] @ 16 kHz mono

// Or stream — get each ~0.48 s chunk as it is generated (first chunk emitted at ~0.43 s). On iPhone
// RTF sits near 1.0, so pre-roll ~2 chunks (~1 s) before playback for smooth, gapless audio
// (perceived first audio ~0.9 s — still ~5x faster than waiting ~4 s for the whole clip):
let stats = try await tts.synthesizeStreaming(text) { chunk in player.play(chunk) }
```

The conversion scripts and the Swift host are in the zoo (`conversion/voxcpm/`) and CoreAIKit.

## Notes

- Plain TTS (fixed speaker). VoxCPM's voice-cloning branch is a follow-on.
- Per-step quality is fp16-equivalent (int8 LM cos > 0.999 vs the fp32 reference); whole-utterance output is natural speech.
- Community port — not an official Apple model.

## Acknowledgements

[OpenBMB / VoxCPM](https://huggingface.co/openbmb/VoxCPM-0.5B). Built on Apple's Core AI.

---

**⬇️ Download:** [🤗 mlboydaisuke/VoxCPM-0.5B-CoreAI](https://huggingface.co/mlboydaisuke/VoxCPM-0.5B-CoreAI) — this card and the
model page are the same document; `scripts/gen-cards` keeps the *Use it* block in sync.
Reproduce it: `python3 conversion/zoo_convert.py show voxcpm-0.5b` prints the command and its
prerequisites; [`recipe.toml`](recipe.toml) is the record.
