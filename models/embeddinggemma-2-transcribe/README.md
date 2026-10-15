Core AI is Apple's on-device ML runtime in iOS 27 / macOS 27 and the successor to Core ML: PyTorch models are exported with Apple's `coreai-torch` (LLMs: `coreai.llm.export`) into `.aimodel` bundles that run on the GPU or the Neural Engine, e.g. Qwen3-8B 4-bit decodes at 94 tok/s on an M4 Max GPU, MLX 90 under the same protocol ([apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench), macOS 27 beta, 2026-06).

# EmbeddingGemma 2 Transcribe — Core AI

[🤗 mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI](https://huggingface.co/mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI) (uploaded 2026-10-14, revision `a4b7895d`) · Apache-2.0, Gemma Prohibited Use Policy · source [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2/tree/914f7f89142e33e77833254d9c9b90c3cef7303b) (revision `914f7f89`) · reader weights [mlboydaisuke/EmbeddingGemma-2-Transcribe-Reader](https://huggingface.co/mlboydaisuke/EmbeddingGemma-2-Transcribe-Reader)

## What it is

English speech to text from the audio encoder of Google's EmbeddingGemma 2. The frozen encoder turns 16 kHz audio
into 25 soft tokens per second in the model's text-embedding space; a 19.6 M-parameter reader (reader v2), trained on
the 960 h of LibriSpeech, maps each token to a 768-d unit vector, and the graph reads the closest of 200,000 English
words (cosine) or blank.
Greedy CTC on the host (merge repeats, drop blanks) gives the transcript. There is no language model, no punctuation
and no casing.

This is an independent reproduction of the method behind Shahir's *Embedding Gemma Transcribe* Mac app (eg2.haku.sh),
described in a post on 2026-10-13; that app's code and weights are not public. The encoder conversion and the reader
here are this port's own, the reader trained from scratch (reader v2: test-clean WER 4.21 %, test-other 10.42 %,
greedy, no language model; reader v1, trained on train-clean-100 alone: 6.81 % and 15.10 %). The Android build on
LiteRT ([mlboydaisuke/EmbeddingGemma-2-Transcribe-LiteRT](https://huggingface.co/mlboydaisuke/EmbeddingGemma-2-Transcribe-LiteRT))
ships its own copy of the reader; that repository's card says which reader version it carries.

Two static graphs, one 30 s window each (bytes of the JIT `.aimodel`; the iPhone 18 Pro AOT compiles are 587,306,390
and 346,673,956):

| Bundle | Graph | Inputs → outputs | Bytes |
|---|---|---|---:|
| `eg2_audio_fp16_T3000` | the Gemma 4 audio tower (12-layer conformer) + `embed_audio`, fp16 | `features [1, 3000, 128]`, `half_mask [1, 1500]` → `soft_tokens [1, 750, 512]` | 586,857,301 |
| `eg2_reader_v2_fp16_T750` | reader v2 (6 transformer layers) + the 200,000 × 768 word table + argmax, fp16 | `soft_tokens [1, 750, 512]`, `token_mask [1, 750]` → `top1_index [1, 750]` (int32), `top1_logit [1, 750]`, `blank_logit [1, 750]` | 346,600,873 |

## Contract

- **Log-mel (host)**: transformers' `Gemma4AudioFeatureExtractor`: 16 kHz mono, 160 zeros in front, 20 ms frames every
  10 ms, periodic Hann window, |FFT| of 512, 128 HTK mel bins over 0–8 kHz without normalization (the 257 × 128
  float64 matrix is `mel_filters.bin` in the HF repo), `log(mel + 0.001)`. Only frames whose window is all real audio
  count: n_frames = ⌊(samples − 161) / 160⌋ + 1 (2,999 for 30 s); rows from n_frames on are 0.
- **`half_mask`**: 1 on the leading ⌈n_frames / 2⌉ of its 1,500 entries, 0 after. Without it a window whose n_frames %
  4 == 2 gets a wrong last token (cosine 0.85–0.88): the second stride-2 convolution reads the half frame.
- **Encoder**: causal apart from the one-frame look-ahead of the 3 × 3 subsampling convolutions; each attention query
  sees itself and the 11 tokens before it. Tokens = ⌈⌈n_frames / 2⌉ / 2⌉, 25 per second.
- **Windows for long audio**: 30 s windows (3,000 frames, 750 tokens), each starting 96 tokens (3.84 s) before the
  end of the previous one; a window keeps only its new tokens and drops its last token (the last token of a 30 s cut
  misses one input frame); the last window keeps through the end. Step = 653 tokens. A window needs 83 tokens of left
  context before its tokens match a continuous run (per-token cosine mean ≥ 0.999 and min ≥ 0.99).
- **Reader**: `token_mask` is 1 on the valid tokens. Per token, `top1_index` is the argmax over the 200,000 words of
  cos(h, table), `top1_logit` = 216.351 × that cosine (reader v2's learned scale), `blank_logit` the blank score. The
  word table and the rotary positions are constants of the graph.
- **Host read-out**: blank where `blank_logit > top1_logit` (strict: a tie reads the word), else `top1_index`; merge
  repeats, drop blanks; map ids through `vocab.txt` (line number = index, blank = 200,000).
- **Calls**: two calls of the same bundle's function issued together return wrong outputs (encoder: max |Δ| 8.04 against
  the sequential pass). The encoder of window i + 1 and the reader of window i can run together (two bundles): the
  labels stay identical to the sequential pass.

## Measured

Mac: M4 Max 128 GB, macOS 27.0 (26A428), Core AI Python runtime (coreai-core 1.0.0b2); ms = median of 8
process-level repetitions × 20 calls. iPhone 18 Pro (iPhone19,2), iOS 27.2 (24B5099f), Swift Core AI runtime, on USB
power, thermal state nominal; ms = median of rounds 1–7 of 8 rounds × 20 calls (round 0 = cold in brackets). ×RT =
audio seconds / wall-clock seconds. The long recording is *The Adventure of the Speckled Band* (LibriVox, 3,257.4 s,
125 windows), scored against the Project Gutenberg text (9,889 words; the LibriVox intro and outro are in the audio
only).

| Device | Workload | Form | Time | ×RT | Parity |
|---|---|---|---|---:|---|
| Mac M4 Max | encoder, one 30 s window | fp16, GPU | 40.02 ms | 750 | 20/20 clips pass, per-token cosine min 0.99645 |
| Mac M4 Max | encoder, 30 s windows, 2 in flight | second form: front fp32 GPU ∥ body fp16 Neural Engine | 30.96 ms per window | 969 | 20/20 clips pass, per-token cosine min 0.99671; soft tokens bit-identical to one window at a time |
| Mac M4 Max | encoder, one 30 s window | second form, one window at a time | 57.40 ms (front 29.13 + host cast 0.58 + body 27.75) | 523 | 20/20 clips pass, per-token cosine min 0.99671 |
| Mac M4 Max | reader v2 + table, one 30 s window | fp16, GPU | 23.85 ms | 1,258 | fixture transcripts 20/20 = PyTorch, test-clean (200 utterances) WER 3.633 % (PyTorch 3.655 %) |
| Mac M4 Max | 3,257.4 s recording | encoder GPU ∥ reader v2 Neural Engine, 2 windows in flight | 6.327 s | 515 | WER 3.80 % (float32 PyTorch path on the same windows: 3.81 %) |
| Mac M4 Max | 3,257.4 s recording | encoder GPU → reader v2 GPU, one window at a time | 8.928 s | 365 | WER 3.82 % |
| Mac M4 Max | 3,257.4 s recording, the app | encoder GPU ∥ reader v2 Neural Engine, 2 windows in flight (the app's Mac default) | 5.946 s | 548 | WER 3.82 % (378 of 9,889 words; float32 PyTorch path 377); fixture transcripts 19/20 = PyTorch (one near-tie between two words) |
| iPhone 18 Pro | encoder, one 30 s window | fp16, GPU, AOT h19p | 59.60 ms (58.49) | 503 | 20/20 clips pass, per-token cosine min 0.99626 |
| iPhone 18 Pro | encoder, one 30 s window | fp16, GPU, JIT `.aimodel` | 59.88 ms (56.81) | 501 | outputs bit-identical to the AOT bundle |
| iPhone 18 Pro | reader v2 + table, one 30 s window | fp16, GPU, AOT h19p | 34.76 ms (34.28) | 863 | through the app: fixture transcripts 20/20 = PyTorch, long-recording WER 3.81 %; reader v1's graph in the same launch 33.05 ms |
| iPhone 18 Pro | 3,257.4 s recording, the app | encoder GPU → reader v2 GPU, one window at a time (the app's iPhone default) | 12.011 s | 271 | WER 3.81 % (377 of 9,889 words; float32 PyTorch path 377); fixture transcripts 20/20 = PyTorch |

- The Mac rows ran the AOT h16c compile of each graph; `macos/` in the HF repo holds the JIT `.aimodel`, which the Mac
  specializes when it loads it (encoder 40.50 ms, reader v2 23.99 ms, outputs identical to the AOT compile).
- The bundles in the HF repo are the measured ones with their MLIR debug locations removed (coreai-torch
  `strip_debug_info`; the export recorded the converting machine's file paths there). The operations are unchanged:
  on the Mac each stripped bundle returns outputs bit-identical to the gated one on the 20 fixture clips (the encoder
  with the GPU preferred, [`provenance/strip/compare_mac.json`](provenance/strip/compare_mac.json); reader v2 with the
  GPU and with the Neural Engine preferred, as the JIT `.aimodel` and as its AOT h16c compile,
  [`provenance/reader-v2/strip/compare_mac_reader_v2.json`](provenance/reader-v2/strip/compare_mac_reader_v2.json)),
  and the `ios-h19p/` compiles of the stripped graphs hold the same weights file (`resources.bin`, byte for byte) as
  the compiles of the gated graphs. The app rows and the iPhone reader row ran the HF repo's bundles (`macos/` on the
  Mac, `ios-h19p/` on the iPhone); the iPhone encoder rows ran the compiles of the unstripped graph.
- The app rows: the eg2-transcribe app (Release build, Swift Core AI runtime) on the 125 windows, the log-mel computed
  before the clock starts (decode and log-mel add 0.66 + 0.05 s on the Mac, 0.51 + 0.16 s on the iPhone); the iPhone
  pass started and ended at thermal state nominal (the pass before it, started at thermal state fair: 12.042 s).
- The Mac's Neural Engine: the fp16 encoder runs there as one region at 85.21 ms but misses the per-token bar at
  near-silent tokens (min 0.888). Cut in two after layer 7 (a float32 front on the GPU, a float16 body on the Neural
  Engine) it passes (min 0.99671) and runs 30.96 ms per window with two windows in flight (×969); that two-bundle form
  is the second encoder form (next point). The iPhone's Neural Engine runs none of the 30 s graphs: with the Neural
  Engine preferred they are placed on the GPU, the encoder's Neural Engine AOT h19p fails to load
  (`failedToSpecialize`), and the reader's (reader v1's 4-layer graph) is ended by jetsam at 3.77 GB under the
  default memory limit.
- **The second encoder form** (the rows marked *second form*) is for running the encoder alone over long audio on a
  Mac. The HF repo has it in `macos-encoder-split7-30s/`, a folder at the repo root that the app does not download:
  the encoder cut in two after layer 7, as two JIT `.aimodel` bundles. The front (subsampling + layers 0–7, float32,
  loaded with the GPU preferred) maps `features [1, 3000, 128]` + `half_mask [1, 1500]` to `hidden [1, 750, 1024]`;
  the host casts `hidden` to float16; the body (layers 8–11 + `embed_audio`, float16, loaded with the Neural Engine
  preferred) maps it to `soft_tokens [1, 750, 512]`. Issue the front of window i + 1 together with the body of window
  i (two bundles; never two calls of one bundle's function at once): 20 windows in a row then take 30.96 ms per
  window against 40.04 ms for the one-bundle encoder in the same measurement window, while one window alone takes
  57.40 ms. It is not for the iPhone: on the iPhone 18 Pro the body is placed on the GPU, and two windows in flight
  take 144.0–144.6 ms per window against 58.4–60.9 ms for the one-bundle fp16 encoder in the same launch. On the 20
  fixture clips the published bundles (their MLIR debug locations removed) return hidden states and soft tokens
  bit-identical to the gated `.aimodel` bundles and to the AOT h16c compile the rows were measured on
  ([`provenance/encoder-split7/compare_mac.json`](provenance/encoder-split7/compare_mac.json)); recipe
  `embeddinggemma-2-transcribe-macos-encoder-split7`.
- Not measured: a launch of the app right after an install, the reader's own gate on the iPhone (the 200 test-clean
  utterances), other iPhones (the JIT `.aimodel` specializes on each; the AOT bundle is h19p only), languages other
  than English, noisy or conversational speech.
- Records: [`provenance/`](provenance/) holds the gate, timing, pipeline, strip and app records behind these rows (the
  work directory's paths replaced by `<work>`; `SOURCES.tsv` gives each original's sha256): reader v2's in
  [`provenance/reader-v2/`](provenance/reader-v2/), reader v1's in `reader/`, `pipeline/` and `app/`; and
  [`gate-embeddinggemma-2-transcribe.json`](gate-embeddinggemma-2-transcribe.json) summarizes the encoder, reader,
  pipeline and strip ones (reader v1 and v2).

## Parity

- **Encoder bar**: per-token cosine against transformers 5.19.0 float32 soft tokens (the default attention; `eager`
  inverts the audio mask in that version) on 20 LibriSpeech test-clean clips: every clip mean ≥ 0.999 and min ≥ 0.99,
  no non-finite value, two passes with drift 0, and a red arm (`half_mask` all ones must break the last token of the
  3 clips with n_frames % 4 == 2; it does on every row).
- **float16 and near-silent tokens**: the tokens that fail on the Neural Engine sit in near-silence (log-mel
  −6.91 = log 0.001 in every bin). In PyTorch, rounding only the input features to float16 already moves them to
  0.996; moving 12 of the 115,584 inputs of one clip by one float16 step puts one token anywhere from 0.972 to
  0.99996 (7 of 24 trials below 0.99). The GPU passes, and its per-token minimum is a number, not a margin.
- **Task bar for the app path**: a float16 path fed by another log-mel implementation is judged by what it writes:
  fixture transcripts ≥ 19/20 equal to the PyTorch path and long-recording WER ≤ PyTorch + 0.1 pt; the extractor itself
  must stay within 1e-5 of transformers' and reach per-token cosine ≥ 0.99999 through the float32 encoder.
- **Reader bar**: against the reader's own float32 PyTorch forward and greedy labels on the same windows, the 20 fixture
  utterances and the 200 leading test-clean utterances (44,162 tokens): test-clean WER ≤ PyTorch + 0.2 pt and fixture
  transcripts ≥ 19/20 for float16. Two red arms: `token_mask` all ones changes 5 of 10 transcripts (reader v1: 7), and
  the second half of the valid soft tokens set to 0 changes the labels of 10 of 10 utterances. The labels that differ
  from PyTorch are near-ties (reader v2 on the GPU: 2 of 44,162, a blank and a word within 0.054 and 0.005 logits of
  each other in float32), and that PyTorch path gives the same transcripts as the release's decoding on all 220
  utterances.
- **Pipeline**: WER on the long recording next to the float32 PyTorch path on the same windows.

## Levers

| Lever | Result |
|---|---|
| quant | fp16 ships. Mac GPU, 30 s: fp16 40.02 ms, fp32 42.47, fp16 weights with fp32 arithmetic 48.17, int8 (Linear, per block of 32, weights only) 42.80 ms; int8 misses the per-token bar (min 0.929) and compiles to the fp32 size (1,174 MB). On the iPhone 18 Pro the fp32 encoder (AOT h19p) takes 188.53 ms against 60.08 ms for fp16 in one launch (3.1×). Reader v1: fp16 22.45 ms against fp32 28.05 ms, table 307 MB against 614 MB. Reader v2 ships in fp16 too (23.85 ms); its residual stream reaches 1,160 (v1: 60.5), 56× below the float16 limit. |
| chunk | 30 s windows. Mac GPU 10 s 17.87 ms vs 30 s 40.02 ms; with the 96-token overlap that is ×345 vs ×654 of new audio per call. iPhone 20.87 vs 59.88 ms (×295 vs ×437 with the overlap). Reader v1 9.38 ms (10 s) vs 22.45 ms (30 s); reader v2 is built for the 30 s window only. |
| aot | JIT and AOT run at the same speed with the same outputs (Mac encoder 40.50 vs 40.22 ms, reader v2 23.99 vs 23.85 ms; iPhone encoder 59.88 vs 59.60 ms, reader v1's graph 32.97 vs 32.99 ms with the interim reader weights). AOT saves the phone's on-device specialization (the encoder in a fresh app container: JIT load 2.94 s and 10.1 s for the opening call, AOT 0.91 s and 0.20 s). fp32 runs on the phone only as AOT: its JIT aborts in MPSGraph's on-device canonicalization, the AOT h19p loads in 1.26 s. |
| compute-unit | Mac: encoder GPU, reader on the Neural Engine beside it (reader v2: 49.98 ms there vs 23.85 ms on the GPU; the two run side by side, 6.33 s vs 11.33 s one window at a time, and 7.40 s with both bundles on the GPU). The all-fp16 encoder on the Neural Engine misses the per-token bar; the two-bundle cut passes (above). iPhone: GPU only at 30 s (above). |
| state | n-a: the metric is throughput on recordings (× real time); a streaming form splits the same audio into smaller calls for latency and does not raise throughput. |

## Reproduce

[`conversion/eg2_transcribe/`](../../conversion/eg2_transcribe/README.md) re-authors the encoder from
`model.safetensors` and puts the reader and its table into one graph; `recipe.toml` here names the commands:

```bash
PY=<venv>/bin/python    # coreai-core 1.0.0b2, coreai-torch 0.4.1, torch 2.9.0
$PY conversion/eg2_transcribe/parity_encoder.py parity
$PY conversion/eg2_transcribe/export_encoder.py --form fp16 --bucket 3000 --aot gpu,neural-engine
$PY conversion/eg2_transcribe/export_reader.py --form fp16 --tokens 750 --aot gpu,neural-engine
$PY conversion/eg2_transcribe/gate_encoder.py --bundle <exports>/fp16/eg2_audio_fp16_T3000_gpu_h16c.aimodelc --backend gpu
$PY conversion/eg2_transcribe/gate_reader.py gate --bundle <exports>/reader_v2/fp16/eg2_reader_v2_fp16_T750_gpu_h16c.aimodelc --backend gpu
$PY conversion/eg2_transcribe/strip_ship.py strip && $PY conversion/eg2_transcribe/strip_ship.py aot
```

Lessons from the port: [`knowledge/embeddinggemma-2-transcribe-port.md`](../../knowledge/embeddinggemma-2-transcribe-port.md).

## App

The `ios/` directory of the eg2-transcribe repository is a SwiftUI app for iPhone and Mac that downloads this model
folder (`pipeline.json`, `reader_config.json`, `vocab.txt`, `mel_filters.bin`, `SHA256SUMS` and the platform's bundle
folder: `ios-<arch>/` when the phone's chip has one, else `ios/`; `macos/` on the Mac), computes the log-mel in Swift
with a float64 FFT (within 4.77e-7 of transformers'), runs the windows through the two bundles and shows the
transcript. When the folder's fingerprint (`SHA256SUMS` and every bundle file) differs from the one it loaded last, it
clears its Core AI cache before loading. Its defaults: on the Mac the reader on the Neural Engine with two windows in
flight, on the iPhone both bundles on the GPU one window at a time.

With reader v2 on the 3,257.4 s recording (the Mac app row above): the Mac default takes 5.946 s (×548), WER 3.82 %
(378 of 9,889 words; float32 PyTorch path 377), fixture transcripts 19/20 = PyTorch (one near-tie between two words);
with both bundles on the GPU, 7.403 s with two windows in flight and 8.056 s one at a time (WER 3.83 %, fixture 20/20).
From launch to the transcript of a 10.4 s clip on the Mac: 8.49 s right after the app cleared its Core AI cache (it
found the new folder's fingerprint; loading the reader for the Neural Engine takes 5.59 s of it) and 0.57 s from the
cache. On the iPhone 18 Pro (thermal state nominal, on USB power) the default takes 12.011 s (×271), WER 3.81 % (377),
fixture transcripts 20/20 = PyTorch (reader v1, revision `a4b7895d`: 11.631 s, WER 5.77 %); from launch to the
transcript, 1.40 s right after the cache clear (that launch began at thermal state serious) and 0.55 s from the cache.
A launch right after an install: not yet measured.

## Licenses

- The bundles: Apache-2.0. EmbeddingGemma 2 (Google DeepMind) is Apache-2.0; deployments must adhere to the
  [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
- Reader v2 is trained on LibriSpeech train-clean-100, train-clean-360 and train-other-500 (CC BY 4.0); the word
  table holds EmbeddingGemma 2 text embeddings of the words of the LibriSpeech lexicon (public domain).
- Graph reference: transformers' `modeling_gemma4.py` / `modeling_embedding_gemma2.py` (Apache-2.0). The banded
  attention, the folded q / k scales and the max-abs RMSNorm follow the published design of EverPaper's Core AI port
  ([EverPaper/embeddinggemma-2-coreai](https://huggingface.co/EverPaper/embeddinggemma-2-coreai) v1.2, Apache-2.0);
  no code was copied.
- Gate data: LibriSpeech test-clean (CC BY 4.0); the long recording is LibriVox (public domain) with the Project
  Gutenberg #1661 text.
