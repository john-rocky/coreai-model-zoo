# Nemotron 3.5 ASR Streaming 0.6B — Core AI

The zoo's **first streaming ASR** — transcription that keeps up with the microphone instead of
waiting for a clip. [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
(**OpenMDW-1.1 — commercial use OK**, 600M) is a **cache-aware FastConformer + pure-RNNT
transducer**: one checkpoint covers **40 locales** (the language is a run-time one-hot input,
`auto` gives built-in language ID) with **punctuation and capitalization built in**. On Core AI it
runs as **six stateless graphs + a host greedy loop**, consuming audio in **320 ms chunks** with
explicit encoder caches — so latency is constant whether you stream 10 seconds or an hour, and
offline clips of ANY length go through the same pipeline (no 30 s bucket like the other three
ASR engines).

## Why streaming is a different export problem

The offline encoder attends over the whole utterance. The streaming variant is *cache-aware*: at
each 320 ms step the conformer must see exactly (a) the new 4 encoder frames and (b) a bounded
window of the past — an attention **KV sliding window of 56 frames** and the **left tails of every
causal convolution**. NeMo/HF keep that state in Python objects; for a static Core AI graph it all
becomes explicit I/O:

```
16 kHz mic ──(host log-mel: preemphasis→STFT→slaney mel→log; NO normalization)──▶ mel chunks (25 then 32 frames)
  1. stream_pre_first / stream_pre : mel (+3 conv2d caches) ─▶ embeds[1,4,1024] + caches     (9 MB fp16)
  2. stream_conformer_a : x · neg_mask[1,1,4,60] · k/v_cache[12,8,56,128] · conv_cache[12,1024,8]
                          ─▶ x + updated caches                       (layers 0-11, 605 MB fp16)
  3. stream_conformer_b : x · one_hot[1,128] · neg_mask · caches
                          ─▶ enc_proj[1,4,640] + updated caches       (layers 12-23 + prompt
                                                                       fusion + projector, 615 MB fp16)
  host greedy RNN-T over the 4 new frames:
  4. predict.aimodel : token[1,1] i32 · h,c[2,1,640] ─▶ dec_out[1,640] · h',c'               (61 MB fp32)
  5. joint.aimodel   : dec_out · enc_frame[1,640] ─▶ token_logits[1,13088]                   (34 MB fp32)
     blank(13087) → next frame ; token → emit + step predictor ; 10 symbols/frame cap
```

Design choices that made it a clean static graph:
- **Right-aligned fixed 56-slot KV window.** In steady state the chunked-limited attention window
  is exactly cache(56)+chunk(4)=60 keys, all visible — so no in-graph masking logic; the host just
  passes an additive `neg_mask` input that hides the not-yet-filled slots during the first 14
  chunks (all zeros afterwards). Right-alignment keeps relative-position distances invariant, so
  the Transformer-XL rel-pos table bakes in as one constant.
- **The conformer ships in two 12-layer halves.** A single 24-layer AOT bundle (2.4 GB
  `resources.bin`) fails to LOAD on-device with an instant POSIX-2 — bisected on-device: the
  same topology at 1 and 12 layers loads fine, every file byte-verified, and Parakeet's
  single-I/O 2.4 GB bundle loads — so the trigger is a big multi-I/O AOT bundle. Splitting
  keeps each compiled half at ~1.1 GB (device-proven) for one extra ~1 ms call per chunk.
  The tiny subsampling stem (what differs between the first chunk and the rest — HF's
  `init_pad`) ships as two more 9 MB graphs.
- **The mel frontend is streaming-exact by construction.** Frame *t* depends only on samples
  `[160t−200, 160t+200)`: the 56-zero margins of the 400-in-512 Hann window absorb both the
  stream-start pad (HF `center=True` first chunk) and the per-chunk preemphasis boundary — so the
  Swift frontend emits a frame the moment its last sample arrives, independent of mic packet size.
  (And unlike Parakeet, Nemotron **never normalizes** the mel.)

## Numerics gate

Golden = the HF *streaming* path itself (chunked `use_cache=True` forwards + the chunk-generator
`generate()`), which was first shown to reproduce the offline forward bit-for-near (cos 1.0000000)
and token-exactly. Then every stage was gated against it:

- streaming re-author (torch): max|Δ| 9.2e-6 vs the HF caches/chunks, per-layer.
- Core AI engine (Mac GPU, fp16): per-frame cos mean 0.999980 → **99/99 token-exact** end-to-end.
- Swift mel frontend: fed in 100 ms packets → **99/99 token-exact** again (packet-size independent).

> "With her white paint and her scarlet smoke stack, the inver is shiel, one of the two small
> steamers that during the summer months plied up and down the lock…"

## Speed

| | per 320 ms chunk (warm) | RTF | load |
|---|---|---|---|
| M4 Max (GPU, JIT) | ~26 ms | 0.082 (12.2× real-time) | 1.7 s |
| iPhone 17 Pro (GPU, AOT h18p) | ~53 ms | 0.167 (6.0× real-time) | ~4 s¹ |

¹ The very first load after install is slow (~52 s: the two 1.1 GB AOT halves specialize their
MPSGraph on the device's GPU once); that result is cached, so every subsequent load is ~4 s. The
conformer
reads its full ~1.2 GB of fp16 weights every chunk, so streaming costs a steady ~4 GB/s of
memory bandwidth — comfortable headroom for an all-day live-caption session (warm 53 ms/chunk
vs 320 ms of audio = 6× real-time). The shipped `lookahead=3` = 320 ms model latency; the
checkpoint also supports 80 ms / 560 ms / 1.12 s variants (re-export with one parameter).

## In the app

**coreai-audio → Transcribe → "Nemotron Streaming 0.6B" → Live** — the transcript grows while you
speak (`MicStreamer` AVAudioEngine tap → `NemotronStreamSession.feed`). The same engine also
transcribes files of any length. vs. Apple's stock `SpeechAnalyzer`: open weights, fully offline,
40 locales in one model with run-time switching, tunable latency, OpenMDW-1.1 commercial use.

## ⬇️ Bundle

**[mlboydaisuke/Nemotron-3.5-ASR-Streaming-CoreAI](https://huggingface.co/mlboydaisuke/Nemotron-3.5-ASR-Streaming-CoreAI)**
— platform subtrees: `macos/` and `ios/` hold the six JIT graphs + tokenizer (every iPhone generation
specializes them on its first load), `ios-h18p/` the conformer halves compiled for the iPhone 17 Pro beside the
four small JIT graphs (that phone only; `ios/` carried them until Hub revision `73c45366`, 2026-09-26). The
iPhone rows above are the h18p AOT halves. On the iPhone 18 Pro (h19p) the JIT halves loaded in 1.37 / 1.44 s the
first time an app container held no specialization of them, ran a first call in 0.93 / 0.32 s and loaded in
0.44 / 0.46 s on relaunch (load-only check, 2026-09-26).
CoreAIKit drop-in: `KitNemotronModel` (`makeSession(language:)` for live, `transcribe(samples:)`
for files). OpenMDW-1.1, LICENSE included.

Convert yourself — [`conversion/nemotron_asr/`](../../conversion/nemotron_asr/), two venvs (isolated
transformers-5.13-dev for the oracles, the main coreai-torch venv for export/gate):
- `gen_oracle.py` / `gen_oracle_streaming.py` — offline + streaming goldens from
  `Nemotron3_5AsrForRNNT` (the streaming one drives chunked `use_cache=True` and records the
  per-chunk embeds/enc_proj and cache tensors, and asserts streaming == offline).
- `export_encoder.py` — offline encoder (the L1485 full-utterance gate that de-risked the
  re-author: chunked-limited attention mask, causal convs, LayerNorm conv module, prompt one-hot).
- `export_encoder_streaming.py` — the cache-explicit streaming re-author + 4-graph export
  (pre_first / pre / conformer_a / conformer_b) + per-chunk engine gate. `export_decoder.py` — predictor + joint (pure RNNT, no duration head).
- `gate_e2e_streaming.py` — mel chunks → engine caches → host RNNT → token-exact.
- `gate_mel_swift_streaming.py` — the Swift frontend spec (packet-driven incremental mel), gated
  token-exact e2e.
- `sideload_ios.sh` / `_nemotron_hf_upload.py` — device push + Hub staging.
