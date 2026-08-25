# Qwen3-ASR-1.7B — Core AI

The zoo's **first ASR (speech-to-text) model**. [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
(Apache-2.0) transcribes ≤30 s clips in 52 languages (auto language-ID, dialects, and
song/music). It completes the on-device **speech stack** on one runtime alongside Qwen2.5-Omni
(audio *understanding*) and Kokoro (text-to-speech): **understand + speak + transcribe**.

It is a **thinker**: a ~300M **AuT audio encoder** feeds a **standard Qwen3 text decoder**
(28L / 2048 / GQA 16:8 / head_dim 128 / QK-RMSNorm / tied head / vocab 151936). The AuT's own
AED decoder is dropped — the encoder output is spliced into the Qwen3 embedding stream and the
LLM does the transcription. (Distinct from FluidAudio's legacy **Core ML + ANE** Qwen3-ASR — this
is the first on the new **Core AI** runtime and a CoreAIKit drop-in.)

<!-- gen-cards:use-it begin id=qwen3-asr-1.7b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — run the kit's task op on this model
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let text = try await CoreAI.transcribe(audioURL, options: .model("qwen3-asr-1.7b"))
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [Transcribe runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Transcribe)
(GUI + CLI, one app for every speech-to-text model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/Transcribe/Transcribe.xcodeproj
# → Run, then pick "Qwen3-ASR 1.7B" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/Transcribe
swift run transcribe-cli --model qwen3-asr-1.7b --audio sample.wav
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let transcriber = try await KitTranscriber(catalog: "qwen3-asr-1.7b")
let samples = try AudioFile.pcm16kMono(url)  // any wav/m4a/mp3 → 16 kHz mono Float
let result = try await transcriber.transcribe(samples: samples)
// result.text, result.language (52 languages)
```

The take-home is [`Examples/Transcribe/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/main/Examples/Transcribe/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; both the runner's GUI and its CLI call it.
Recording? `MicRecorder` (kit API) captures mic audio as 16 kHz mono `[Float]` — the record
button and permission prompt are your app's own chrome.

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` → product **CoreAIKit**
- Info.plist: `NSMicrophoneUsageDescription` — only if you record
- Entitlements: none needed (macOS)
- First run downloads the model — 3.1 GB (Mac) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Pipeline

```
16 kHz mono ──(Whisper log-mel, host)──▶ mel[128, frames]   (zero-pad to 30 chunks = 3000 frames)
  1. encoder.aimodel : input_features[1,128,3000] · attn_bias[1,390,390]
                       ─▶ audio_embeds[390, 2048]            (host trims to the clip's N rows)
  host: build the ASR prompt; rewrite the N <|audio_pad|> ids to V + slot (0..N-1)
  2. decoder.aimodel : input_ids (audio ids = V+slot) · position_ids · audio_embeds[N,2048] (KV state)
                       ─▶ logits ─ greedy ─▶ {Language}<asr_text>{transcript}
  host: split at <asr_text> (151704); the text after it is the transcript
```

The **AuT encoder** is a fixed-shape K=30 graph (HF's data-dependent `cu_seqlens` windowed
attention re-expressed as one static `attn_bias [1, S, S]`, S = K·13 = 390). The host zero-pads the
mel to whole 100-frame chunks and trims the `[390, 2048]` output to the clip's real token count N
(a per-chunk sum: full chunks ×13 + the partial last chunk). Audio rides the decoder on **one
static `audio_embeds` input** via the id-space trick (audio tokens carry `vocab + slot`; the graph
`index_select`s row `slot`). MRoPE is a **no-op** for ASR (`get_rope_index` returns 3 identical
position copies) → the decoder is plain 1-D Qwen3 RoPE, `qwen3.py` reused verbatim.

### Graph contracts

```
encoder  in  input_features[1,128,3000] f16 · attn_bias[1,390,390] f16 (0 in-window&valid / −65504 else)
         out audio_embeds[390,2048] f16            (host trims to N; engine cos 0.999997 vs fp32)
decoder  in  input_ids[1,1] i32 (audio = V+slot) · position_ids · audio_embeds[390,2048] (static buffer)
         out logits[1,V]                           KV state: keyCache/valueCache
```

The shippable decoder is the **static-[1,1]-query (`_s1`)** twin driven by the high-level Core AI
engine (the Qwen-Omni audio path), with `audio_embeds` bound as a `StaticInputBuffer`. The engine
manages the KV cache and positions, so prefill runs as pipelined S=1 steps and decode is flat.

## Numerics gate (end-to-end, on-engine)

The whole pipeline was gated **token-for-token** against the official `qwen_asr` greedy output on a
Japanese clip: **17/17 exact** — `language Japanese<asr_text>音声認識のテストです。今日はいい天気ですね。`
+ EOS. Encoder engine cos **0.999997**; decoder int8hu first-token = golden. Measured on M4 Max
(macOS 27 beta, Core AI **GPU**): encoder one-shot, prefill ~0.4 s, decode flat **~45–57 ms/tok**
(~18–24 tok/s). ≈2.9 GB on disk (decoder int8hu 2.3 GB + encoder fp16 0.6 GB).

## The port in four lessons

1. **A fully-static decode graph beats the per-token wedge.** Driving a decoder with a *growing*
   `position_ids[1,p+1]` re-specializes the graph every token and each specialization probes a
   failing ANE compile → effective hang. The correctness gate instead uses two fixed-shape graphs
   (prefill q_len=Sp; decode q=1 with `pos` a runtime VALUE → `mutable_slice_update` at slot `pos`
   + full-buffer read + explicit causal mask `j≤pos`) that compile **once** → flat decode. For ship,
   the high-level engine drives the standard dynamic decoder and manages this itself.
2. **The engine-native RoPE op mishandles a baked-constant `position_ids`.** In a fully-static
   graph, externalizing `rope` with `position_ids = arange(Sp)` gives a *silently wrong* rotation
   that grows with position (layer-0 attention cos 0.69 vs **1.0** decomposed); the model runs but
   ignores the audio (first token `\n`). Fix: **drop `rope` and `scaled_dot_product_attention` from
   the externalize set** for static graphs (RoPE decomposes bit-exact; SDPA must decompose to take
   the explicit mask). RMSNorm/GatherMM stay externalized.
3. **Variable length = a dynamic-q prefill + K=30 encoder.** The encoder is baked once at K=30
   (host pads mel, trims to N). The prefill is dynamic in prompt length Sp and audio count N → the
   engine specializes once per length and **caches on disk** (a one-time ~1 s warmup per new length,
   not the per-token wedge). The decode graph is length-agnostic.
4. **`encoder` fp16 NaNs in eager, not on the engine.** The −65504 window mask × fp16 matmul
   overflows in plain torch; gate the encoder eager in **fp32** and on-engine in **fp16** (the
   exported graph is correct — engine cos 0.999997).

## ⬇️ Bundle

**[mlboydaisuke/Qwen3-ASR-1.7B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3-ASR-1.7B-CoreAI)**
— `qwen3_asr_1.7b_decode_int8hu_n390_s1/` (decoder, 2.3 GB) + `qwen3_asr_1.7b_audio_encoder_fp16_k30.aimodel`
(0.6 GB) + tokenizer. Apache-2.0. CoreAIKit drop-in: `KitASRModel` (`attach(samples:)` →
`transcribe()`).

Convert yourself: [`conversion/qwen3_asr/`](../../conversion/qwen3_asr/) —
`export_encoder.py --chunks 30` (AuT encoder) + `export_decoder.py --n-audio 390` (decoder), gated
end-to-end by `gate_static.py`. The 0.6B variant falls out of the same scripts (config swap:
d896 / 18L / 14h, FFN 3584, output_dim 1024).
