# Parakeet-TDT-0.6B — Core AI

The zoo's **first transducer / TDT (RNN-T family) model** — a different ASR architecture from the
attention decoders (Whisper, Qwen3-ASR). [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
(cc-by-4.0, 600M) transcribes ≤~29 s clips in **25 European languages**. It runs as **three
stateless `.aimodel` graphs + a host-driven greedy loop** — no LLM runtime, just CoreAIKit's
`GraphModel`.

English only? [**parakeet-v2**](../parakeet-v2/README.md) is the same architecture trained on
English alone (vocab 1025, blank 1024), ported by
[Rahul Rachuri](https://github.com/RahulRachuri) and exported by these same two scripts through
`--hf-id`.

A **token-and-duration transducer (TDT)**: a **FastConformer encoder** streams acoustic frames, an
**LSTM predictor** carries the text state, and a **joint network** decides, at each (frame, token)
pair, both the next token and **how many frames to skip** (the "duration" head — the TDT speedup
over vanilla RNN-T). Blanks advance time; non-blank tokens advance the predictor. Fulfills
apple/coreai-models issue **#7**.

<!-- gen-cards:use-it begin id=parakeet-tdt-0.6b-v3 (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
![Parakeet-TDT 0.6B v3 demo](https://huggingface.co/mlboydaisuke/Parakeet-TDT-0.6B-CoreAI/resolve/main/demo.gif)
*Parakeet-TDT 0.6B on iPhone 17 Pro — the zoo's coreai-audio app, real speed.*

## Use it

⚡ **One line** — run the kit's task op on this model
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let text = try await CoreAI.transcribe(audioURL, options: .model("parakeet-tdt-0.6b-v3"))
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [Transcribe runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Transcribe)
(GUI + CLI, one app for every speech-to-text model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/Transcribe/Transcribe.xcodeproj
# → Run, then pick "Parakeet-TDT 0.6B v3" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/Transcribe
swift run transcribe-cli --model parakeet-tdt-0.6b-v3 --audio sample.wav
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let transcriber = try await KitTranscriber(catalog: "parakeet-tdt-0.6b-v3")
let samples = try AudioFile.pcm16kMono(url)  // any wav/m4a/mp3 → 16 kHz mono Float
let result = try await transcriber.transcribe(samples: samples)
// result.text (25 EU languages)
```

The take-home is [`Examples/Transcribe/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/main/Examples/Transcribe/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; both the runner's GUI and its CLI call it.
Recording? `MicRecorder` (kit API) captures mic audio as 16 kHz mono `[Float]` — the record
button and permission prompt are your app's own chrome.

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` → product **CoreAIKit**
- Info.plist: `NSMicrophoneUsageDescription` — only if you record
- Entitlements: none needed (macOS)
- First run downloads the model — 1.3 GB (Mac) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Pipeline

```
16 kHz mono ──(log-mel, host: preemphasis→STFT→slaney mel→log→per-utt norm)──▶ mel[1,128,2885]
  1. encoder.aimodel : mel[1,128,2885] f16 ─▶ enc_proj[1,361,640]   (FastConformer + 1024→640 projector)
  host greedy TDT loop over the 361 encoder frames:
  2. predict.aimodel : token[1,1] i32 · h,c[2,1,640] ─▶ dec_out[1,640] · h',c'   (embedding→2-LSTM→proj)
  3. joint.aimodel   : dec_out[1,640] · enc_frame[1,640] ─▶ token_logits[1,8193] · dur_logits[1,5]
     token = argmax(token_logits) ; dur = [0,1,2,3,4][argmax(dur_logits)]
     if token==blank(8192) && dur==0 : dur = 1     # forward-progress guard
     frame += dur ; if token!=blank : emit(token) and step the predictor (advance LSTM only on non-blank)
  host: detokenize the emitted ids (BPE + Metaspace) ─▶ transcript
```

The encoder is baked at a **fixed 30 s bucket (2885 mel frames → 361 encoder frames)**; the host
pads/trims every clip to it. Trailing silence just makes the loop emit blanks, so no attention/conv
masking is needed — the simple static graph is correct.

### Graph contracts

```
encoder  in  mel[1,128,2885] f16            out enc_proj[1,361,640]     (GPU gate cos 0.999995 vs fp32)
predict  in  token[1,1] i32 · h[2,1,640] f32 · c[2,1,640] f32  out dec_out[1,640] · h_out · c_out
joint    in  dec_out[1,640] f32 · enc_frame[1,640] f32          out token_logits[1,8193] · dur_logits[1,5]
```

Constants: blank **8192**, vocab **8193**, durations **[0,1,2,3,4]**, decoder_start 8192, 16 kHz.

## Numerics gate (end-to-end, on-engine + in Swift)

The whole pipeline was gated **token-for-token** vs the HF `ParakeetForTDT` reference on a
LibriSpeech clip: **77/77 exact** —

> "With her white paint and her scarlet smokestack, the Inverashiel, one of the two small steamers
> that during the summer months plied up and down the loch, and incidentally carried on
> communication between Inverashiel and Cryonon."

Re-authored in plain torch from `model.safetensors` (encoder eager cos 1.000000, 710 tensors),
exported, and gated on Core AI **GPU**. Then the **Swift** path (`KitParakeetModel`: Accelerate mel
→ the three graphs → host TDT loop → swift-transformers detokenize) reproduced the same 77 tokens
**token-exact on Mac GPU** — the engine is a CoreAIKit drop-in, no Python at inference.

## Speed (on-device, iPhone 17 Pro / iOS 27, Release, GPU)

Device-verified end-to-end (the AOT-compiled encoder bundle, token-exact): a **14.84 s** clip
transcribes in **0.31 s** (warm; 0.40 s cold) — **47.9× real-time**. Encoder bundle **load 3.9 s**
(one-time, the `.h18p.aimodelc` mmaps its precompiled MPSGraph; the un-compiled 1.2 GB `.aimodel`
JIT-specializes for minutes/stalls on-device — see lesson 4). The whole transducer (encoder one-shot
+ a host greedy loop over 361 frames) is why decode is near-instant.

## The port in four lessons

1. **The joint activation is ReLU, not the encoder's SiLU.** The joint uses the top-level
   `config.hidden_act` (relu); wiring the encoder's silu there silently garbles the transcript.
2. **`ParakeetFeatureExtractor` always per-utterance normalizes.** Despite a `do_normalize` arg
   (dead code in transformers 5.12.1), the extractor *always* zero-means/unit-vars the log-mel per
   channel. The encoder was gated on the normalized features, so the Swift mel must normalize too —
   and the bucket is filled by **silence-padding the audio** (the trailing-silence frames are
   normalized in place), **not** by padding the mel with a constant. Padding the mel with raw zeros
   instead makes the decoder hallucinate extra tokens over the tail.

   **The bucket length is therefore baked into the feature values, so it is not a free knob.**
   [Rahul Rachuri](https://github.com/RahulRachuri) exported the encoder at L=2880 instead of
   2885 while porting [v2](../parakeet-v2/README.md): it gates clean per-token (cos mean
   0.999996) and still fails end to end, 131/152 chunks. The normalization spans the whole
   padded window, so a mel produced at one bucket and truncated to another is not the mel that
   bucket's pipeline would have seen. Changing L means regenerating the golden, not reslicing it.
3. **fp16 encoder, fp32 decoder, all on GPU.** The 24-layer FastConformer ships fp16 (GPU cos
   0.999995; CPU fp16 is noisier, min ~0.95). The small predictor/joint stay fp32.
4. **iOS ships the AOT-compiled encoder; the tokenizer is retagged for swift-transformers.** The
   1.2 GB encoder's on-device JIT specialization stalls (minutes / no progress), so iPhone loads a
   `coreai-build compile … --platform iOS --architecture h18p --preferred-compute gpu` `.aimodelc`
   (2.3 GB, embeds the precompiled MPSGraph) — load drops to ~3.9 s. The small predict/joint JIT
   fine and ship as portable `.aimodel`. Separately, swift-transformers rejects
   `tokenizer_class:"ParakeetTokenizer"` (`unsupportedTokenizer`); the bundle retags it to
   `"PreTrainedTokenizer"` (→ BPETokenizer; decode is driven by tokenizer.json's Metaspace decoder,
   so it stays exact).

## ⬇️ Bundle

**[mlboydaisuke/Parakeet-TDT-0.6B-CoreAI](https://huggingface.co/mlboydaisuke/Parakeet-TDT-0.6B-CoreAI)**
— `parakeet_encoder_float16_L2885.aimodel` (fp16, 1.2 GB) + `parakeet_predict_float32.aimodel`
(49 MB) + `parakeet_joint_float32.aimodel` (21 MB) + `tokenizer.json`/`tokenizer_config.json` +
`mel_filters_128x257_f32.bin`. cc-by-4.0. CoreAIKit drop-in: `KitParakeetModel`
(`transcribe(samples:onPartial:)`). In the **coreai-audio** app's Transcribe tab as "Parakeet
TDT 0.6B". For iPhone, AOT-compile the encoder to `.h18p.aimodelc` and sideload the bundle into
`Documents/Models/Parakeet/` (the app prefers it over the Hub download).

Convert yourself — [`conversion/parakeet/`](../../conversion/parakeet/), two venvs (an isolated
transformers-5.x for the golden, the main coreai-torch venv for export/gate; `_GPU_LOCK` before any
Mac-GPU run):
- `gen_oracle.py` — golden oracle from `ParakeetForTDT` (`--seconds 16 --pad-seconds 14` → the 30 s
  bucket); also asserts the hand-rolled TDT loop == `model.generate()`.
- `export_encoder.py` — re-author the FastConformer (Transformer-XL rel-pos, depthwise conv,
  BatchNorm fold) + projector from safetensors → fp16 `.aimodel`, per-token cosine gate vs golden.
- `export_decoder.py` — re-author the embedding + 2-layer LSTM + joint (ReLU) → fp32 `.aimodel`s.
- `gate_e2e.py` — full pipeline (mel → encoder → host TDT loop), token-exact vs the golden.
- `gate_mel_swift.py` / `mel_swift_sim.py` / `diff_swift_mel.py` — pin the Swift mel recipe: a manual
  cos/sin-DFT reimplementation (what `ParakeetMelPreprocessor.swift` runs) gated token-exact e2e and
  diffed against the golden `input_features`. This is what caught the normalization bug (lesson 2).
- `_parakeet_hf_upload.py` — stage + upload the bundle (also retags `tokenizer_class`, lesson 4).
