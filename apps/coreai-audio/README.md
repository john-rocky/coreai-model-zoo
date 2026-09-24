# coreai-audio — on-device audio *understanding* + *transcription* + *speech* (iOS + macOS)

Four tabs, all fully on-device:

- **Understand** — record from the mic, choose a file, or use the demo clip, then ask *"what do you
  hear?"*: a local **Qwen2.5-Omni Thinker** describes the **sounds** (events, texture, emotion), not
  a transcript. *"I hear a loud hissing sound."* · *"I hear a man speaking in English."*
- **Transcribe** — speech → **text**, with a choice of three Core AI ASR models (segmented picker):
  - **Whisper large-v3-turbo** — the Apple-recipe export on the **stock runtime** (fixed-128-token
    decoder window), via CoreAIKit's `KitWhisperModel`. 100 languages, ≤30 s, auto language-detect.
    Downloads from [🤗 whisper-large-v3-turbo-CoreAI-official](https://huggingface.co/mlboydaisuke/whisper-large-v3-turbo-CoreAI-official)
    on both platforms. See [`knowledge/whisper-asr-fixed-decode.md`](../../knowledge/whisper-asr-fixed-decode.md).
  - **Qwen3-ASR-1.7B** — the zoo's first ASR (AuT encoder + Qwen3 decoder on the pipelined engine),
    via `KitASRModel`. 52 languages, ≤30 s. See [`models/qwen3-asr/README.md`](../../models/qwen3-asr/README.md).
  - **Parakeet-TDT-0.6B** — the zoo's first **transducer / TDT (RNN-T family)** (NVIDIA FastConformer
    encoder + LSTM predictor + joint, 3 graphs driven by a host greedy TDT loop), via
    `KitParakeetModel`. 25 EU languages, ≤30 s; **iPhone 17 Pro 47.9× real-time**. See
    [`models/parakeet/README.md`](../../models/parakeet/README.md).
  - **Diarize — who said what** (toggle): **Nemotron-3-Diarization** (NVIDIA, OpenMDW-1.1) finds who
    spoke when for up to **8 speakers**, then the chosen ASR transcribes each turn →
    *"Speaker 1 [0.3–4.1s]: …"*. One fp16 graph runs on the GPU. The Swift package
    [`NemotronDiarizer`](../../conversion/nemotron3_diar/swift) does the rest: the log-mel, 0.72 s
    streaming chunks with 0.32 s look-ahead, and the speaker cache. On a 97.6 s clip it matches
    transformers fp32 on **99.9987 %** of the frame × speaker decisions (1 of 78,072 differs). That
    holds on the M4 Max GPU and on the iPhone 17 Pro GPU. A chunk takes **15.4–15.9 ms** on the M4 Max
    and **30.1 ms** on the iPhone. The iPhone numbers come from the gate app
    [`apps/N3DGate`](../N3DGate), which runs the same package and graph. For the transcript, each 10 ms
    frame goes to its most probable speaker above 0.5. One speaker's turns at most 0.48 s apart merge
    into one. See [`models/nemotron-3-diarization`](../../models/nemotron-3-diarization/README.md).
    Without its bundle the toggle uses **Streaming Sortformer 4-spk v2** (NVIDIA, CC-BY-4.0, 4 speakers;
    byte-gated **100% speaker-activity agreement** vs NeMo, see
    [`conversion/sortformer_diar/HANDOFF.md`](../../conversion/sortformer_diar/HANDOFF.md)).
- **Voice** — **VoxCPM-0.5B** diffusion text-to-speech (MiniCPM4 LM + LocDiT flow-matching +
  AudioVAE), streaming int8. See [🤗 VoxCPM-0.5B-CoreAI](https://huggingface.co/mlboydaisuke/VoxCPM-0.5B-CoreAI).
- **Speak** — **Kokoro-82M** (StyleTTS2 + iSTFTNet) text-to-speech on Core AI: pick a voice and a
  phrase, hear it spoken. Three `.aimodel` bundles (predictor / prosody / vocoder) on the CPU
  compute unit + the host DSP (alignment + hn-nsf source) in Swift; ~0.7 s/utterance, magspec-corr
  0.999 vs the PyTorch reference. See [`models/kokoro-82m/README.md`](../../models/kokoro-82m/README.md). The demo
  phrases are phonemized ahead of time (host-side G2P), so this build needs no MLX/espeak; the three
  bundles come from [🤗 Kokoro-82M-CoreAI](https://huggingface.co/mlboydaisuke/Kokoro-82M-CoreAI)
  (`KokoroAssets/` ships the voices + tokenizer; drop the `.aimodel` there or in Documents).

Device-verified on **iPhone 17 Pro** (A19 Pro) and **M4 Max** (TTS verified on M4 Max).

## How it works

Built on **[CoreAIKit](https://github.com/john-rocky/coreai-kit)** (`KitAudioModel`) — a local audio
model behind FoundationModels' `LanguageModel`, on the **coreai-pipelined** GPU engine:

```swift
let model = try await KitAudioModel(model: id)        // downloads decoder + encoder from HF
try await model.attach(samples: pcm16kMono)           // mel → audio encoder → static buffer
let answer = try await LanguageModelSession(model: model).respond(to: "What do you hear?")
```

- **Audio encoder** (Whisper-style, 1.2 GB fp16) — run once per clip → `audio_embeds [750,2048]`.
- **Text decoder** (Qwen2.5-3B, 3.9 GB int8) — the audio embeds ride **one static-input buffer**;
  the prompt's `<|AUDIO|>` placeholders carry extension ids `vocab + slot` the graph gathers. No
  rope-shift inputs (TMRoPE collapses to 1-D for audio+text).
- **Mel front end** — Whisper-large-v3 log-mel in Accelerate/vDSP, bit-exact with the HF feature
  extractor (cos 1.0). Mic capture uses `AVAudioRecorder`; clips are capped to ~6 s for snappy
  on-device prefill.

iPhone downloads the **AOT** decoder (`.aimodelc`) so the 3.9 GB graph dodges the on-device JIT
jetsam (the AOT weights mmap as clean pages → comfortable headroom). Needs the
`com.apple.developer.kernel.increased-memory-limit` entitlement.

**Transcribe** rides CoreAIKit's ASR models. Whisper is one stateless graph (no LLM engine) driven
through `GraphModel`; the same `mel_filters.f32` powers it (bit-exact with the HF `mel_filters_128.npy`),
so no extra resource ships:

```swift
let whisper = try await KitWhisperModel(model: .largeV3Turbo)   // downloads .aimodel + tokenizer
let result  = try await whisper.transcribe(samples: pcm16kMono) // -> Transcription(language, text)
// or: let asr = try await KitASRModel(model: .qwen3ASR1_7B); try await asr.transcribe(samples:)
// or: let par = try await KitParakeetModel(model: .parakeetTDT); try await par.transcribe(samples:)
```

## Run

```sh
cd apps/coreai-audio
xcodegen generate
open coreai-audio.xcodeproj   # set your team, then Run (Release) on iPhone or Mac
```

First launch **downloads ~5 GB** from
[🤗 Qwen2.5-Omni-3B-Audio-CoreAI](https://huggingface.co/mlboydaisuke/Qwen2.5-Omni-3B-Audio-CoreAI)
(`ios/` AOT decoder on iPhone, `gpu-pipelined/` on macOS, shared encoder). **Load model** →
**Record** / **Choose…** / **Demo** → **Ask**.

Audio understanding here is GPU-pipelined (an ANE static-shape rework for lower power is a follow-up).
Any clip is decoded to 16 kHz mono.

The app does not download the Diarize bundle. On macOS, `Sources/N3DAssets` is a git-ignored symlink to
a folder with the files of
[🤗 Nemotron-3-Diarization-CoreAI](https://huggingface.co/mlboydaisuke/Nemotron-3-Diarization-CoreAI).
On iPhone, sideload the same files to `Library/Application Support/N3DAssets/`; the app loads the
`.h18p.aimodelc` graph there. The headless check runs both diarizers against their reference outputs and
exits 0 only when both pass. `N3D_GOLDEN` is the folder `conversion/nemotron3_diar/export_golden.py`
writes; `N3D_FIXTURES` holds `test_multispk_16k.wav` and `diarization_example_16k.wav`.

```sh
DIARIZE_SELFTEST=1 DIAR_RESULT=/tmp/d.txt N3D_GOLDEN=<golden folder> N3D_FIXTURES=<wav folder> \
  DYLD_FRAMEWORK_PATH=<app>/Contents/Frameworks <app>/Contents/MacOS/coreai-audio
```
