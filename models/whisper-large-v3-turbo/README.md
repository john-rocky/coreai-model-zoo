# Whisper large-v3-turbo — Core AI

[`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo)
(809M, MIT) — OpenAI's multilingual speech-to-text: **100 languages, automatic language
detection**, ≤30 s window per decode. Runs fully on device on the **stock** Core AI runtime
(no engine patch), **token-for-token identical** to the PyTorch greedy reference.

Bundle: [🤗 mlboydaisuke/whisper-large-v3-turbo-CoreAI-official](https://huggingface.co/mlboydaisuke/whisper-large-v3-turbo-CoreAI-official)
— `macos/` and `ios/` each hold the fp16 JIT `.aimodel` (~1.6 GB; every iPhone generation specializes it on its
first load: iPhone 18 Pro 4.37 s, first call 2.11 s, 0.26 s on relaunch, 2026-09-26), `ios-h18p/` the bundle
compiled for the iPhone 17 Pro (~3.2 GB, that phone only; it sat in `ios/` until Hub revision `171153c6`,
2026-09-26, and comes from a separate export of 2026-06-28). Catalog id: **`whisper-large-v3-turbo`** (its
pin still points at the previous layout until the next kit release).

<!-- gen-cards:use-it begin id=whisper-large-v3-turbo (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
![Whisper large-v3-turbo demo](https://huggingface.co/mlboydaisuke/whisper-large-v3-turbo-CoreAI-official/resolve/main/demo.gif)
*Whisper large-v3-turbo on iPhone 17 Pro — the zoo's coreai-audio app, real speed.*

## Use it

**New to Core AI? [Start with CoreAIKit 0.7.3](https://github.com/john-rocky/coreai-kit#readme).** Follow its requirements and first-run steps for `qwen3-0.6b`, then open the same release's [ChatDemo](https://github.com/john-rocky/coreai-kit/tree/0.7.3/Examples/ChatDemo). The README records the tested OS/SDK and download size; model and device coverage is stated per example.

⚡ **One line** — this model is the default behind the kit's task op
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let text = try await CoreAI.transcribe(audioURL)
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/0.7.3/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [Transcribe runner](https://github.com/john-rocky/coreai-kit/tree/0.7.3/Examples/Transcribe)
(GUI + CLI, one app for every speech-to-text model in the catalog):

```bash
git clone --branch 0.7.3 --depth 1 https://github.com/john-rocky/coreai-kit
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
open -a /Applications/Xcode-27.0.0-RC.app coreai-kit/Examples/Transcribe/Transcribe.xcodeproj
# → Run, then pick "Whisper large-v3-turbo" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/Transcribe
swift run -c release transcribe-cli --model whisper-large-v3-turbo --audio sample.wav
```

Use Xcode build **27A266a** from the release's `.xcode-pin`; adjust the app path if your installation is named differently.

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let transcriber = try await KitTranscriber(catalog: "whisper-large-v3-turbo")
let samples = try AudioFile.pcm16kMono(url)  // any wav/m4a/mp3 → 16 kHz mono Float
let result = try await transcriber.transcribe(samples: samples)
// result.text, result.language ("en", "ja", … auto-detected)
```

The take-home is [`Examples/Transcribe/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/0.7.3/Examples/Transcribe/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; both the runner's GUI and its CLI call it.
Recording? `MicRecorder` (kit API) captures mic audio as 16 kHz mono `[Float]` — the record
button and permission prompt are your app's own chrome.

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` (exact **0.7.3**) → product **CoreAIKit**
- Info.plist: `NSMicrophoneUsageDescription` — only if you record
- Entitlements (iOS): `com.apple.developer.kernel.increased-memory-limit`
- First run downloads the model — ~1,623 MB (Mac) / ~3,235 MB (iPhone) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Measured (M4 Max, GPU)

Greedy decode vs the HF PyTorch reference (`generate`, greedy):

| Metric | Value |
|---|---|
| Transcript | **token-for-token identical** to PyTorch greedy |
| First step (compile + warmup) | 0.68 s |
| Per token (steady state) | **0.18 s** |

The fixed window caps one decode at 128 tokens — enough for a 30 s window; chunk longer audio
into 30 s segments (the runner and `KitWhisperModel` do this for you).

## How the bundle works (why it's not the stock recipe verbatim)

Apple's official `models/whisper/export.py` traces a **single** decode step
(`decoder_input_ids [1,1]`, no KV cache) — that graph can't be driven autoregressively. This
bundle is the same recipe traced at a **fixed 128-token decoder window**: pad the buffer, read
the logits at the real last position (causal attention ignores the padding, so the read is
exact), constant shape so MPSGraph compiles **once** (a dynamic-length export recompiles every
step → ~15 s/token; this is 0.18 s/token). Full write-up:
[`knowledge/whisper-asr-fixed-decode.md`](../../knowledge/whisper-asr-fixed-decode.md) · recipe:
[`conversion/export_whisper_fixed.py`](../../conversion/export_whisper_fixed.py) · hashes + export
environment: the [HF card](https://huggingface.co/mlboydaisuke/whisper-large-v3-turbo-CoreAI-official).

Driving the raw graph yourself (no kit)? [`apps/CoreAITranscribe`](../../apps/CoreAITranscribe/)
implements this exact decode loop + the log-mel frontend from scratch (macOS + iOS, file or mic)
— the reference if you're porting to another language or runtime.

## Related

- Same runner, same 3 lines — change the id: [`qwen3-asr.md`](../qwen3-asr/README.md) (52 languages,
  LLM-decoder ASR) · [`parakeet.md`](../parakeet/README.md) (fastest, 25 EU languages)
- Streaming ASR (live mic, any length):
  [Nemotron 3.5 ASR Streaming](https://huggingface.co/mlboydaisuke/Nemotron-3.5-ASR-Streaming-CoreAI)
- Official-recipe artifacts + protocol: [`official/`](../../official/README.md)
