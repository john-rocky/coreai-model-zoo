# Qwen2.5-Omni-3B Audio Understanding — Core AI

[Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)'s **Thinker** converted to Apple
**Core AI** (`.aimodel` / `.aimodelc`, iOS 27 / macOS 27) for **on-device audio *understanding*** —
the model describes the **sounds** it hears (events, texture, emotion, music), it is **not** a
transcriber. *"I hear a loud hissing sound."* · *"…a continuous sine wave sound."* · *"…a series of
beeps."*

Part of the [CoreAI-Model-Zoo](https://github.com/john-rocky/coreai-model-zoo). **Device-verified on
iPhone 17 Pro (A19 Pro) and M4 Max.**

<!-- gen-cards:use-it begin id=qwen2.5-omni-3b-audio (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — this model is the default behind the kit's task op
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let scene = try await CoreAI.describeAudio(audioURL)
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [AudioChat runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/AudioChat)
(GUI + CLI, one app for every audio-understanding model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/AudioChat/AudioChat.xcodeproj
# → Run, then pick "Qwen2.5-Omni 3B Audio" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/AudioChat
swift run audiochat-cli --model qwen2.5-omni-3b-audio --audio sample.wav --prompt "What do you hear?"
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit
import FoundationModels

let audio = try await KitAudioModel(catalog: "qwen2.5-omni-3b-audio")
try await audio.attach(samples: AudioFile.pcm16kMono(audioURL))  // clip → encoder buffer
let session = LanguageModelSession(model: audio)
let reply = try await session.respond(to: question)
// reply.content: what the model heard, described fully on-device
```

The take-home is [`Examples/AudioChat/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/main/Examples/AudioChat/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; the CLI is an argument shell over it, and
the GUI drives the same `KitAudioModel(catalog:)` behind a `LanguageModelSession`.
Live mic? `MicRecorder` (kit API) captures 16 kHz mono `[Float]` — attach that instead.
One clip per session; attach a new clip to ask about different audio.

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` → product **CoreAIKit**
- Info.plist: `NSMicrophoneUsageDescription` — only if you record
- Entitlements: none needed (macOS)
- First run downloads the model — 5.5 GB (Mac) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## What's here

Two models, run as a pair on the **coreai-pipelined** GPU engine:

| path | what | size |
|---|---|---|
| `gpu-pipelined/qwen2_5_omni_3b_thinker_int8lin_n750_s1/` | text decoder (Qwen2.5-3B int8lin, S=1) — **macOS** | 3.9 GB |
| `gpu-pipelined/qwen2_5_omni_3b_audio_encoder_fp16_k15.aimodel/` | Whisper-style audio encoder (fp16, K=15 ≈ 30 s) — both platforms | 1.2 GB |
| `ios/qwen2_5_omni_3b_thinker_n750_ios/` | text decoder **AOT** (`.aimodelc`, iPhone 17 Pro / h18p) | 4.5 GB |

The decoder's audio embeds ride **one static-input buffer** (`audio_embeds [750,2048]`); the prompt's
`<|AUDIO|>` placeholders carry extension ids `vocab + slot` the graph gathers. TMRoPE collapses to
1-D for audio+text, so positions are engine-native (no rope-shift inputs). iPhone uses the **AOT**
decoder so the 3.9 GB graph dodges the on-device JIT jetsam; the AOT weights mmap as clean pages, so
it loads comfortably (≈5.9 GB headroom after load on a 12 GB device, with the
`increased-memory-limit` entitlement).

## Use it

The [`coreai-audio`](https://github.com/john-rocky/coreai-model-zoo/tree/main/apps/coreai-audio) app
(record from the mic / choose a file / demo clip → "what do you hear?"), or
[CoreAIKit](https://github.com/john-rocky/coreai-kit):

```swift
let model = try await KitAudioModel(model: .qwen2_5Omni3B)   // downloads decoder + encoder
try await model.attach(samples: pcm16kMono)                  // mel → encoder → static buffer
let answer = try await LanguageModelSession(model: model).respond(to: "What do you hear?")
```

The 16 kHz log-mel front end is Whisper-large-v3 (Accelerate/vDSP), bit-exact with the HF feature
extractor (gated cos 1.0). Any clip is decoded to 16 kHz mono, ≤ ~30 s.

## Conversion / numerics

Conversion code + gates:
[`conversion`](https://github.com/john-rocky/coreai-model-zoo) (`export_qwen2_5_omni_thinker.py` /
`export_qwen2_5_omni_audio.py`). Decoder int8lin gates top-1-exact vs the fp32 HF oracle; the
encoder static rework is cos 1.0 vs eager (GPU 0.99999); the Swift vDSP mel is cos 1.0 vs the HF
extractor. iPhone greedy matches the Mac content (white-noise → "I hear a loud hissing sound.").

## License

Apache-2.0 (inherits [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)). A community
conversion — not affiliated with Alibaba or Apple.

---

**⬇️ Download:** [🤗 mlboydaisuke/Qwen2.5-Omni-3B-Audio-CoreAI](https://huggingface.co/mlboydaisuke/Qwen2.5-Omni-3B-Audio-CoreAI) — this card and the
model page are the same document; `scripts/gen-cards` keeps the *Use it* block in sync.
Reproduce it: `python3 conversion/zoo_convert.py show qwen2.5-omni-3b-audio` prints the command and its
prerequisites; [`recipe.toml`](recipe.toml) is the record.
