# VoxCPM2 2B — Core AI (on-device, 48 kHz)

[OpenBMB **VoxCPM2** (2B)](https://huggingface.co/openbmb/VoxCPM2) converted to **Apple Core AI**, running
fully **on-device** on iPhone (A19 Pro / iPhone 17 Pro) and Mac — no network. The 2B, 48 kHz successor to
[VoxCPM-0.5B-CoreAI](https://huggingface.co/mlboydaisuke/VoxCPM-0.5B-CoreAI).

A tokenizer-free diffusion TTS: a **MiniCPM4 28-layer** text-semantic LM + an **8-layer residual** acoustic
LM drive a **12-layer LocDiT** flow-matching diffusion head, decoded by a **48 kHz AudioVAE**. Five Core AI
bundles + a few host-side projections.

<!-- gen-cards:use-it begin id=voxcpm2-2b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — run the kit's task op on this model
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let audio = try await CoreAI.speak(text, options: .model("voxcpm2-2b"))
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [Speak runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Speak)
(GUI + CLI, one app for every text-to-speech model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/Speak/Speak.xcodeproj
# → Run, then pick "VoxCPM2 2B" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/Speak
swift run speak-cli --model voxcpm2-2b --text "Hello from Core AI." --output hello.wav
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let speaker = try await KitSpeaker(catalog: "voxcpm2-2b")
let audio = try await speaker.synthesize(text)
// audio.samples: 48 kHz mono PCM in [-1, 1] — play it or write a WAV
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
- First run downloads the model — 4.7 GB (Mac) / 5.7 GB (iPhone) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## What's inside

| dir | contents |
|---|---|
| `macos/` | JIT `.aimodel` bundles (Mac): int8 base/res decode + prefill, fp16 feat_decoder / feat_encoder / vocoder |
| `ios/` | AOT `.aimodelc` bundles (iOS `h18p`, GPU): same five + the two int8 prefill bundles |
| `voxcpm2_host_glue/` | embed table + projections / FSQ-512 / stop-head / fusion (`.bin` + manifest) |
| `tokenizer/` | the VoxCPM2 tokenizer (Llama fast) |

The backbone LMs are **weight-only int8** (the size driver); the diffusion + VAE stay **fp16** (the
continuous-feedback path is quant-sensitive — same split mlx-community uses).

## On-device numbers (iPhone 17 Pro, int8 + prefill + streaming)

- **RTF 1.19**, **first-audio 0.65 s**, 48 kHz, ~4.9 GB resident (increased-memory entitlement).
- Streaming starts after the first ~0.65 s; the 2B is ~4× the 0.5B, so RTF sits just above realtime.

## Use it

Runs through **[coreai-kit](https://github.com/john-rocky/coreai-kit)** `VoxCPM2TTS`, wired into the
**[coreai-model-zoo](https://github.com/john-rocky/coreai-model-zoo)** `coreai-audio` app ("Voice 2B" tab).
Conversion + gates + export scripts: `coreai-model-zoo/conversion/voxcpm/` (`*_v2.py`).

```swift
let tts = try await VoxCPM2TTS(paths: .standard(artifactsRoot: root, lm: .int8))
let wav = try await tts.synthesize("On device speech synthesis, running entirely on your iPhone.") // 48 kHz Float PCM
```

## Verification

Reimplemented in exportable Core AI overlays and gated end-to-end against the official model: backbone /
feat_decoder / feat_encoder **cos 1.0**, full chain **magspec 0.996**; every exported bundle engine-gated
**cos ≥ 0.9999**.

## License

Apache-2.0 (commercial OK), inherited from [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2).
Not affiliated with OpenBMB or Apple. Community port.

---

**⬇️ Download:** [🤗 mlboydaisuke/VoxCPM2-CoreAI](https://huggingface.co/mlboydaisuke/VoxCPM2-CoreAI) — this card and the
model page are the same document; `scripts/gen-cards` keeps the *Use it* block in sync.
Reproduce it: `python3 conversion/zoo_convert.py show voxcpm2-2b` prints the command and its
prerequisites; [`recipe.toml`](recipe.toml) is the record.
