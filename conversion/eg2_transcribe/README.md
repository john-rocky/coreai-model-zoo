# EmbeddingGemma 2 Transcribe → Core AI (audio encoder + word reader)

`google/embeddinggemma-2` (revision `914f7f89`, Apache-2.0; deployments follow the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)) turns 16 kHz speech into 25 soft
tokens per second in its text space: the Gemma 4 audio tower (a 12-layer conformer) followed by `embed_audio`
(RMSNorm without scale + Linear 1536→512). A reader trained on LibriSpeech (eg2-transcribe `reader/`, released as
[mlboydaisuke/EmbeddingGemma-2-Transcribe-Reader](https://huggingface.co/mlboydaisuke/EmbeddingGemma-2-Transcribe-Reader))
maps each soft token to a 768-d unit vector and reads the closest of 200,000 words or blank.

This directory re-authors the audio path in plain PyTorch from `model.safetensors`, puts the reader and its word
table into a second graph, exports both to static Core AI bundles, gates every form on the Mac CPU, GPU and Neural
Engine, times them, and strips and compiles the shipping pair. Card:
[`../../models/embeddinggemma-2-transcribe/README.md`](../../models/embeddinggemma-2-transcribe/README.md); lessons:
[`../../knowledge/embeddinggemma-2-transcribe-port.md`](../../knowledge/embeddinggemma-2-transcribe-port.md); bundles:
[mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI](https://huggingface.co/mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI).

## Graph contracts

Encoder, one static graph per frame bucket `Tb` (10 ms frames; `Tb = 3000` = a 30 s window, `1000` = 10 s):

| name | shape | dtype | meaning |
|---|---|---|---|
| `features` | `[1, Tb, 128]` | fp32 (fp16 form: fp16) | log-mel frames of the HF `Gemma4AudioFeatureExtractor`; rows ≥ n_frames are 0 |
| `half_mask` | `[1, Tb/2]` | same as `features` | 1 on the leading ⌈n_frames/2⌉ rows, 0 after |
| `soft_tokens` | `[1, Tb/4, 512]` | same as `features` | the leading ⌈⌈n_frames/2⌉/2⌉ rows are valid |

n_frames = ⌊(samples − 161) / 160⌋ + 1 (2999 for 480,000 samples). One mask is enough: after the stride-2
subsampling every op is causal or per token (12-key local attention, left-padded depthwise conv), so right padding
reaches no valid token except through the 3×3 conv look-ahead. Without `half_mask`, a clip with n_frames % 4 == 2
gets a wrong last token (cos 0.847–0.880 on the 3 such fixture clips); with it, every token matches
(`parity_encoder.py parity`). For long audio, cut 30 s windows with ≥ 96 tokens (3.84 s) of left overlap, keep only
the new tokens, and drop each window's last token (it reads one frame past a 30 s cut); the last window keeps the end.

Reader + word table, one static graph per window of T tokens (T = 750 for a 30 s encoder window, 250 for 10 s):

| name | shape | dtype | meaning |
|---|---|---|---|
| `soft_tokens` | `[1, T, 512]` | fp32 (fp16 form: fp16) | one window of the encoder's soft tokens; the rows after the valid ones are masked |
| `token_mask` | `[1, T]` | same as `soft_tokens` | 1 on the valid tokens, 0 after |
| `top1_index` | `[1, T]` | int32 | argmax over the 200,000 words of cos(h_t, table_w) |
| `top1_logit` | `[1, T]` | same as `soft_tokens` | scale · the largest cosine (scale 91.218 for reader v1) |
| `blank_logit` | `[1, T]` | same as `soft_tokens` | |

Host read-out: label_t = blank if `blank_logit_t > top1_logit_t`, else `top1_index_t` (strict, so a tie reads the word,
as the reader's `greedy_labels` does); merge repeats, drop blanks, map ids through `vocab.txt`. The table (fp16
`[200000, 768]`, re-normalized in fp32) and the rotary cos / sin of positions 0..T−1 are constants of the graph;
padded keys get −1e4 before the softmax. An utterance longer than T tokens goes in as consecutive windows of at most T
tokens; their labels are concatenated before the collapse.

## Forms

| form | weights / arithmetic | bundle | notes |
|---|---|---|---|
| encoder `fp32` | fp32 / fp32 | 1,174 MB | the correctness baseline |
| encoder `fp16w32` | fp16 stored, cast in-graph / fp32 | 588 MB | the casts stay in the graph |
| encoder `fp16` (ships) | fp16 / fp16, fp16 I/O | 587 MB | every RMSNorm divides by its row max-abs before squaring (the residual stream reaches 383; its square overflows fp16); the two subsampling LayerNorms stay plain |
| encoder `int8` | Linear weights int8 symmetric per-block 32 (coreai-opt, weight only) / fp32 | 332 MB (compiled GPU bundle 1,174 MB) | fails the per-token bar |
| encoder `fp16sf`, `fp16mm` | fp16 with one exact rewrite each (subsampling LayerNorm mean folded into the conv weights; RMSNorm mean of squares as a matmul) | 587 MB | Neural Engine experiments; both fail there |
| reader `fp32` | reader + table, fp32 | 668 MB | |
| reader `fp16` (ships) | reader + table, fp16 (inputs, weights, table, arithmetic, logits; int32 index) | 334 MB | every norm and the final L2 divide by the row's max \|x\| up front; LayerNorm written out (coreai-torch lowers `layer_norm` on an fp16 input with fp32 statistics) |
| reader `h-out`, `h-out-fp16` | the reader body alone: → `h [1, T, 768]` (L2-normalized) + `blank_logit`; the table product and argmax on the host (`host_argmax`) | 53.5 / 26.8 MB | the read-out comparison |

## Bars

- **Encoder**: 20 LibriSpeech test-clean clips (5–17 s), per-token cosine against transformers 5.19.0 fp32 soft tokens
  (the default sdpa attention: `eager` inverts the audio mask in 5.19.0); PASS = every clip mean ≥ 0.999 and min ≥
  0.99, no non-finite value. Two passes (drift must be 0) and a red arm: `half_mask` all ones must break the last token
  of the 3 clips with n_frames % 4 == 2.
- **Reader**: the 20 fixture utterances and the 200 leading test-clean utterances (44,162 tokens, 4,542 + 541 words)
  against the reader's own torch fp32 forward + `greedy_labels` on the same windows. fp32 forms: all 20 fixture
  transcripts identical and ≥ 99.9 % of the test-clean tokens; fp16 forms: test-clean WER ≤ torch + 0.2 pt and ≥ 19/20
  fixture transcripts. Two passes (label drift 0) and two red arms: `token_mask` all ones on the 10 shortest fixture
  utterances must change transcripts, and the second half of the valid soft tokens set to 0 must change the labels of
  ≥ 9 of 10 (`redarm_zerohalf_*.json`; with reader v1 the `token_mask` arm no longer moves a label at T = 250).
- **Pipeline**: the long fixture's WER against its Gutenberg text, next to the torch fp32 path on the same windows,
  two passes with identical labels.
- **App path** (another log-mel implementation in front of the fp16 encoder): the extractor within 1e-5 of the HF one
  and per-token cos ≥ 0.99999 through the fp32 encoder; then the task — fixture transcripts ≥ 19/20 equal to torch and
  long-recording WER ≤ torch + 0.1 pt. The fp16 encoder's per-token minimum is reported, not used as a bar there:
  ±1 ulp on 12 inputs moves one near-silent token between 0.972 and 0.99996.

## Measured on the Mac (M4 Max 128 GB, macOS 27.0 26A428, coreai-core 1.0.0b2, coreai-build 3600.83.1)

Time = median of 8 process-level reps × 20 calls on the h16c AOT bundle, forms compared inside one quiet measurement
window; ×RT = window audio / wall time.

Encoder, 30 s and 10 s windows:

| form | GPU 30 s | GPU 10 s | gate (GPU) | Neural Engine 30 s | gate (ANE) |
|---|---|---|---|---|---|
| fp16 | **40.02 ms (×750)** | 17.87 ms (×560) | PASS, min 0.99645 (10 s: 0.99255) | 85.21 ms (×352), 1 region | FAIL, min 0.888 |
| fp32 | 42.47 ms (×706) | **17.74 ms (×564)** | PASS, min 0.999999999 | 0 regions (GPU fallback) | — |
| fp16w32 | 48.17 ms (×623) | 21.20 ms (×472) | PASS, min 0.9999989 | 0 regions | — |
| int8 | 42.80 ms (×701) | 17.73 ms (×564) | FAIL, min 0.929 | 0 regions | — |
| EverPaper v1.2 asset (comparison) | 42.50 ms (×706) | 18.95 ms (×528, 1024 bucket) | PASS, min 0.9968 | 93.18 ms (×322), 5 regions | FAIL, min 0.905 |

Reader v1 + table, 30 s (750 tokens) and 10 s (250 tokens) windows:

| form | GPU 30 s | GPU 10 s | gate (GPU, 30 s) | Neural Engine 30 s | gate (ANE, 30 s) |
|---|---|---|---|---|---|
| fp16 | **22.45 ms (×1,336)** | **9.38 ms (×1,066)** | PASS: fixture 19/20, test-clean WER 5.878 % = torch, 2 of 44,162 labels differ | 45.55 ms (×659), 2 ANE regions | PASS: fixture 20/20, WER 5.900 % (+0.022 pt) |
| fp32 | 28.05 ms (×1,070) | 11.81 ms (×847) | PASS: every label identical | 0 ANE regions | — |
| h-out-fp16 + host read-out | 2.97 + 124.9 ms (×235) | — | PASS: fixture 19/20, WER = torch | 7.91 + 122.6 ms (×230), 1 ANE region | PASS: fixture 20/20, WER = torch |

The long recording (The Adventure of the Speckled Band, LibriVox, 3,257.4 s; 125 windows of 3000 log-mel rows stepping
653 tokens; encoder fp16 GPU → reader → host labels in one process; WER against the Gutenberg #1661 text, 9,889 words,
the LibriVox intro and outro being in the audio only):

| reader | wall, loading excluded | with loading both bundles | WER |
|---|---|---|---|
| fp16 GPU, one window at a time | 8.093 s (×402.5) | 9.444 s (×345) | 5.794 % |
| fp32 GPU, one window at a time | 8.728 s (×373) | 10.432 s (×312) | 5.794 % |
| fp16 Neural Engine, one window at a time | 10.708 s (×304) | 12.754 s (×255) | 5.804 % |
| fp16 GPU, 2 windows in flight | 7.640 s (×426) | — | 5.794 % |
| **fp16 Neural Engine, 2 windows in flight** | **5.681 s (×573)** | — | 5.804 % |
| torch fp32 (CPU), same windows | — | — | 5.784 % |

- **The Neural Engine and the encoder.** The fp16 graph lands on the ANE as one region only after the subsampling
  LayerNorms are plain: their max-abs variant makes `ANECCompileOffline` fail (`CompilationFailure`) and
  `coreai-build` still exits 0 with a GPU-only bundle — count `*_ANE_region_*` names, do not trust the exit code. Every
  all-fp16 form (this one and EverPaper's) misses the per-token minimum on the ANE on the same near-silent tokens
  (log-mel −6.91 in every bin) while clip means stay ≥ 0.9990; the GPU runs the same fp16 graph within the bar.
- **The reader's time is the table**: the body alone takes 2.97 ms on the GPU, the 200,000 × 768 product and argmax
  the other ≈ 19.5 ms; the same read-out in numpy takes 124.9 ms, so it stays in the graph. On the ANE the reader +
  table is 2 regions without weights (they stay in the MPSGraph package), half the GPU's speed, 7.0 GB peak footprint.
- **2 windows in flight**: the encoder of window i+1 issued together with the reader of window i keeps the labels
  identical to the one-at-a-time pass; with the reader on the Neural Engine the GPU encoder and the ANE reader run side
  by side (8.5 GB peak footprint). Two calls of the **same** function issued together return wrong outputs (the fp16
  encoder: max |Δ| 8.04 against the sequential pass): never overlap one function.
- **JIT vs AOT**: the same per-call time and identical outputs (encoder 40.50 vs 40.22 ms, reader 22.68 vs 22.51 ms);
  AOT saves the load-time specialization. A Neural Engine AOT bundle's initial load specializes for minutes (encoder
  116–240 s), later loads 0.01 s: `split_encoder.py warm` / `gate_reader.py warm` do it outside a measurement.
- **CPU (`cpu_only()`)**: reader fp32 178.9 ms, fp16 234.7 ms per 30 s window, both pass. In fp16 the CPU returns
  `top1_logit` = 0.0 at every token whose largest cosine is negative (blank tokens); the index is right and the blank
  logit still wins, so the labels do not change. The GPU and the ANE return the negative value.
- **int8** does not pay on the Mac: the per-token minimum fails (adjacent token pairs in 8 of 20 clips) and the
  compiled GPU bundle is fp32-sized (the dequantization folds into constants).

## Measured on the iPhone 18 Pro (iPhone19,2, iOS 27.2 24B5099f)

With a gate app on the Swift Core AI runtime (not in this directory), USB powered, thermal state nominal; the same 20
clips and 220 reader utterances, judged on the Mac with this directory's functions; ms = median of rounds 1–7 of 8
rounds × 20 calls (round 0 in brackets).

| bundle | time | gate |
|---|---|---|
| encoder fp16, GPU, AOT h19p (`coreai-build compile --platform iOS --preferred-compute gpu --architecture h19p`) | 59.60 ms (58.49), ×503 | PASS 20/20, min 0.99626 |
| encoder fp16, JIT `.aimodel` (the Mac's file), GPU | 59.88 ms (56.81), ×501 | outputs bit-identical to the AOT bundle |
| encoder fp32, AOT h19p | 188.53 ms (188.46), ×159 | PASS, min 0.9999999972; its JIT `.aimodel` aborts in the load (MPSGraph canonicalization) |
| reader fp16, GPU, AOT h19p, with the interim reader checkpoint | 32.99 ms (32.77), ×909 | PASS; reader v1 on the phone, through the app: fixture 20/20 transcripts = torch, long-recording WER 5.774 % |

At 30 s nothing runs on the phone's Neural Engine: with the Neural Engine preferred the graphs are placed on the GPU
(outputs bit-identical to the GPU route), the encoder's neural-engine AOT h19p fails with `failedToSpecialize`, and
the reader's is ended by jetsam at 3.77 GB (default memory limit).

## Two-bundle form: float32 front on the GPU + float16 body on the Neural Engine

Every all-fp16 form misses the per-token minimum on the Neural Engine (above). `split_encoder.py` measures where the
error comes from and cuts the encoder after a stage s into two static bundles per window length:

| bundle | input | output | precision | compiled for |
|---|---|---|---|---|
| front_s | `features` [1, Tb, 128] + `half_mask` [1, Tb/2], fp32 | `hidden` [1, Tb/4, 1024] fp32 | fp32 weights and arithmetic | h16c GPU AOT |
| body_s | `hidden` [1, Tb/4, 1024] fp16 | `soft_tokens` [1, Tb/4, 512] fp16 | the fp16 form's build (max-abs RMSNorm) | h16c Neural Engine AOT, 1 ANE region |

The host casts the front's `hidden` to fp16 and calls the body. "After stage s" includes s: `sub` = the two
stride-2 conv blocks + `input_proj`, `layerK` = conformer layers 0..K. The cut carries no state (every op after the
subsampling is causal or per token), and body(front(x)) equals the whole encoder bit for bit in fp32 torch.

Where the error comes from (`split_encoder.py attrib` / `propagate`, 6 clips at 10 s): each fp16 prefix ending at a
stage, run on the ANE, the GPU and the CPU, then carried through the remaining stages in torch fp32. The ANE's
output at the near-silent tokens is already at cos 0.933 after the subsampling + layer 0 (GPU 0.9986, torch fp16
0.9975) and at 0.889 after layer 6; the hidden-state cosine of the same prefixes stays ≥ 0.99987 up to layer 8 and
hides it. The near-silent tokens amplify small early perturbations even when the rest runs in fp32.

| cut after | front (GPU, fp32) | body (ANE, fp16) | gate 30 s | gate 10 s | 30 s: front + cast + body | 30 s, 2 in flight | 10 s / 10 s 2 in flight |
|---|---|---|---|---|---|---|---|
| `sub` | 4 MB | 613 MB, 12 layers + head | FAIL, min 0.974 | FAIL, min 0.969 | 2.5 + 0.5 + 81.0 = 84.3 ms | 81.8 ms | 20.9 / 19.6 ms |
| `layer4` | 488 MB | 359 MB, 7 layers + head | FAIL, min 0.988 | FAIL, min 0.979 | 19.0 + 0.6 + 47.8 = 67.4 ms | 49.4 ms | 20.7 / 12.2 ms |
| `layer7` | 778 MB | 207 MB, 4 layers + head | **PASS, min 0.9967** | **PASS, min 0.9980** | 29.1 + 0.6 + 27.7 = 57.4 ms | **31.0 ms (×969)** | 20.6 / **12.8 ms (×783)** |
| `layer8` | 874 MB | 157 MB, 3 layers + head | **PASS, min 0.9988** | **PASS, min 0.9974** | 32.4 + 0.6 + 20.9 = 53.9 ms | 33.9 ms (×886) | 19.4 / 14.2 ms |
| one bundle, fp16 GPU (anchor) | | | PASS, min 0.9964 | PASS, min 0.9926 | 40.0 ms (×750) | wrong outputs | 18.0 ms / wrong outputs |
| one bundle, fp16 ANE (anchor) | | | FAIL, min 0.888 | FAIL, min 0.888 | 84.9 ms (×353) | 84.3 ms | 20.3 / 20.1 ms |

"2 in flight" issues the front of window i+1 together with the body of window i (`--pipeline`); the gate runs the
clips in that order too and gets soft tokens bit-identical to the sequential pass. The cut after layer 7 balances
the two engines (front 29.1 ms on the GPU, body 27.7 ms on the ANE), so a stream of 30 s windows runs at 31.0 ms per
window against 40.0 ms for the one-bundle GPU form measured in the same window. One window alone is slower (57.4 ms).
On the iPhone 18 Pro the fp32 front alone takes 124.4 ms per 30 s window (its GPU runs fp32 at 3.1× the fp16 time) and
the body is placed on the GPU, so the pair (149 ms) is slower than the one-bundle fp16 encoder there. The layer-7
pair at 30 s is published as a second encoder form for the Mac only, in `macos-encoder-split7-30s/` at the root of the
HF repo (not under `macos/`, which the app downloads; not in `ios/` or `ios-h19p/`): the two JIT `.aimodel` exports
with their MLIR debug locations removed by `strip_ship.py`'s `strip_one` (the two bundles added to its `SOURCES`),
whose hidden states and soft tokens on the Mac are bit-identical to the gated exports and to their h16c AOT compiles
on the 20 fixture clips (`provenance/encoder-split7/` in the HF repo).

Two all-fp16 rewrites aimed at the ANE error (exact in real arithmetic, `export_encoder.py --form fp16sf | fp16mm`)
do not help: folding the subsampling LayerNorm's channel mean into the conv weights leaves the ANE at min 0.896
(and drops the GPU to 0.946); taking each RMSNorm's mean of squares as a matmul breaks every clip on the ANE (min
0.404) while the GPU still passes (0.996).

## Reproduce

Environment: the shared Core AI venv (torch 2.9.0, coreai-torch 0.4.1, coreai-core 1.0.0b2, coreai-opt 0.2.1 and
torchao 0.17.0 for the int8 form, safetensors, numpy); `jiwer` for the `score` steps (any venv); Xcode 27 with
`coreai-build` for the AOT compiles. Locations come from `_common.py` (environment variable, else a default under
`../_paths.py`):

- `EG2_MODEL_DIR` — a `google/embeddinggemma-2` snapshot at `914f7f89` (`model.safetensors` is checked by sha256).
- `EG2_FIXTURES` — the eg2-transcribe `reader/fixtures` (`test-clean-20/<id>/{features,soft_tokens}.npy`,
  `long-speckled-band/features.npy`, `MANIFEST.json`), made with transformers 5.19.0 (default attention).
- `EG2_READER_DIR` — the eg2-transcribe `reader/` directory: `train/model.py` and `train/decode.py`, `vocab.txt`,
  `word_table.npy`, and the release's `reader.pt` + `reader_config.json` (the last four are also in the Reader repo
  on the Hub).
- `EG2STT_CACHE` — the soft-token cache with `test-clean/index.tsv` (the reader gate's 200 test-clean utterances).
- `EG2STT_WORK` — exports and records (`<work>/exports/…`).

Run the GPU / Neural Engine steps alone on the machine: another job on the GPU changes the numbers and, at times, the
results. In order:

```bash
PY=<venv>/bin/python
# 1. encoder: torch parity (+ the mask claim) and the fp16 headroom
$PY conversion/eg2_transcribe/parity_encoder.py parity
$PY conversion/eg2_transcribe/parity_encoder.py absmax
# 2. export (each export gates its torch-exported, decomposed graph on 4 clips before it saves)
$PY conversion/eg2_transcribe/export_encoder.py --form fp16 --bucket 3000 --aot gpu,neural-engine
$PY conversion/eg2_transcribe/reader_coreai.py parity          # the reader graph vs the reader's torch path
$PY conversion/eg2_transcribe/export_reader.py --form fp16 --tokens 750 --aot gpu,neural-engine
# 3. runtime gates
$PY conversion/eg2_transcribe/gate_encoder.py --bundle <work>/exports/fp16/eg2_audio_fp16_T3000_gpu_h16c.aimodelc --backend gpu
$PY conversion/eg2_transcribe/gate_reader.py gate --bundle <work>/exports/reader/fp16/eg2_reader_fp16_T750_gpu_h16c.aimodelc --backend gpu
$PY conversion/eg2_transcribe/gate_reader.py warm --bundle <work>/exports/reader/fp16/eg2_reader_fp16_T750_ane_h16c.aimodelc --backend ane
$PY conversion/eg2_transcribe/gate_reader.py gate --bundle <work>/exports/reader/fp16/eg2_reader_fp16_T750_ane_h16c.aimodelc --backend ane
<venv with jiwer>/bin/python conversion/eg2_transcribe/gate_reader.py score --gate <work>/exports/reader/fp16/gate_gpu_T750.json
# 4. timing (forms interleaved in one window; --cold-first deletes each bundle's cache entry before rep 0)
$PY conversion/eg2_transcribe/bench_encoder.py prepare
$PY conversion/eg2_transcribe/bench_encoder.py ab --label gpu30 --config fp32:3000:gpu --config fp16:3000:gpu --reps 8 --cold-first
$PY conversion/eg2_transcribe/bench_pipeline.py prepare
$PY conversion/eg2_transcribe/bench_pipeline.py ab --label reader30 --config fp32:750:gpu --config fp16:750:gpu --reps 8 --cold-first
# 5. the long recording end to end (torch path, then the bundles, with and without 2 windows in flight)
$PY conversion/eg2_transcribe/bench_pipeline.py torch-long
$PY conversion/eg2_transcribe/bench_pipeline.py long --reader <work>/exports/reader/fp16/eg2_reader_fp16_T750_ane_h16c.aimodelc --backend ane --overlap
# 6. the shipping pair: strip the debug locations, compile for the iPhone 18 Pro, compare on the Mac
$PY conversion/eg2_transcribe/strip_ship.py strip
$PY conversion/eg2_transcribe/strip_ship.py aot
$PY conversion/eg2_transcribe/strip_ship.py compare
```

Other forms: `export_encoder.py --form fp32 | fp16w32 | int8 | fp16sf | fp16mm --bucket 3000 | 1000`,
`export_reader.py --form fp32 | h-out | h-out-fp16 --tokens 750 | 250`, `--checkpoint <run>/best_model.pt` for a
training checkpoint instead of the release. The two-bundle form:

```bash
$PY conversion/eg2_transcribe/split_encoder.py reference --bucket 3000 --clips all --stages layer7 --out <split>/ref_hidden_layer7_T3000.npz
$PY conversion/eg2_transcribe/split_encoder.py export --part front --stage layer7 --bucket 3000 --aot gpu
$PY conversion/eg2_transcribe/split_encoder.py export --part body --stage layer7 --bucket 3000 --aot neural-engine
$PY conversion/eg2_transcribe/split_encoder.py warm --bundle <split>/eg2_body_layer7_fp16_T3000_ane_h16c.aimodelc --backend ane
$PY conversion/eg2_transcribe/gate_encoder.py --front <split>/eg2_front_layer7_fp32_T3000_gpu_h16c.aimodelc \
    --body <split>/eg2_body_layer7_fp16_T3000_ane_h16c.aimodelc --backend ane --ref-hidden <split>/ref_hidden_layer7_T3000.npz --pipeline
$PY conversion/eg2_transcribe/bench_encoder.py ab --label split30 --json-dir <split> --config splitlayer7:3000:gpu+ane \
    --config fp16:3000:gpu --config fp16:3000:ane --reps 8 --pipeline
```

Exports are not byte-reproducible (the converter names resources at random): keep a gated bundle.

## The Core AI cache and a re-exported AOT bundle

An AOT `.aimodelc`'s `main.hash` is the hash of the program without its weights, and the Core AI cache entry of a
process is `~/Library/Caches/coreai-cache/<OS build>/<process name>/<main.hash in hex>`. A bundle re-exported with new
weights (a new reader checkpoint) and compiled to the same path keeps its `main.hash`, and the next load returns the
previous weights' specialization: the gates reproduce the old numbers to the decimal. Delete the entry before the next
load (for the venv's `python`):

```bash
rm -rf "$HOME/Library/Caches/coreai-cache/$(sw_vers -buildVersion)/python/$(xxd -p -c 32 <bundle>.aimodelc/main.hash)"
```

A JIT `.aimodel`'s `main.hash` is sha256(`main.mlirb`) and changes with the weights; `strip_ship.py strip` gives each
copy a new `main.hash` too. `bench_*.py --cold-first` deletes each bundle's own entry before rep 0.

## Strip and ship

coreai-torch 0.4.1 records each op's source location, so an exported `main.mlirb` names the converting machine's
files (absolute paths), and every AOT bundle compiled from it does too; `grep -rIl` skips binary files and misses them
(`grep -rl -a` finds them). `strip_ship.py` (no inference in `strip` / `aot`):

- `strip` — `AIModelAsset.load(bundle).program`, one program object, → coreai-torch `strip_debug_info` → the same
  object's `save_asset(out, the source's metadata)`; checks: op counts of source, stripped and saved-then-reloaded
  equal, `coreai-build inspect` summaries equal, author / license / description kept, `main.hash` = sha256(`main.mlirb`)
  and new, no home-directory path or user name in any byte. Out: `<work>/exports/ship/macos/<name>.aimodel`.
- `aot` — `coreai-build compile <copy> --platform iOS --preferred-compute gpu --architecture h19p`; checks: exit 0,
  `sourceHash` = the copy's sha256(`main.mlirb`), no ANE region, the same byte scan; recorded beside them: which files
  equal the h19p compile of the unstripped source. Out: `<work>/exports/ship/ios-h19p/<name>.aimodelc`. Never load an
  iOS bundle on the Mac.
- `compare` — the Core AI runtime on the Mac: each gated bundle and its stripped copy as JIT `.aimodel`, the 20 fixture
  clips: encoder (GPU preferred) and reader (GPU and Neural Engine preferred); PASS = every output bit-identical, plus
  two controls that must differ.

Results (2026-10-14; macOS 27.0 26A428, coreai-torch 0.4.1, coreai-core 1.0.0b2, coreai-build 3600.83.1):

- `strip` — encoder `main.hash` 3cce8936… → 330ff806…, 4,559 ops equal (source, stripped, saved and reloaded),
  inspect summaries equal, 18 → 0 local-path hits; reader 6fbf8f27… → f7e5b2b8…, 465 ops, 12 → 0. A few seconds each.
- `aot` — 587,306,390 and 334,027,356 bytes, 0 ANE regions, 0 local-path hits. Against the h19p compiles of the gated
  bundles, `resources.bin` (the weights) and `manifest.plist` are byte-identical; `main-h19p.mlirb`,
  `specialized_model_0.mpsgraph`, `main.hash`, `metadata.json` and `stats.json` differ.
- `compare` — SAME: the encoder (GPU preferred) and the reader (GPU preferred, and Neural Engine preferred) return
  bit-identical outputs on 20 of 20 calls each (max |Δ| 0); the controls differ (19 of 19 next-clip pairs in each
  comparison, and one float16 step on one encoder input changes the output). 14 s in one quiet window.

The records go to `<work>/exports/ship/provenance/` (`strip_<name>.json`, `aot_h19p_<name>.json`, `compare_mac.json`).

## Files

- `eg2_audio_coreai.py` — the re-authored encoder, the checkpoint loader (q / k scales folded into the projections
  and their clip bounds), the fp16w32 parametrization.
- `parity_encoder.py` — torch parity and the fp16 headroom measurement.
- `export_encoder.py` — torch.export → Core AI, metadata, AOT compiles, ANE region count, `coreai-build inspect`.
- `gate_encoder.py` — the runtime gate (CPU / GPU / ANE, AOT or `--jit`), with the red arm;
  `--contract everpaper` gates the EverPaper asset the same way.
- `bench_encoder.py` — timing inputs, per-process timing, the interleaved A/B driver; `split<s>` configs time the
  two-bundle form, `--pipeline` adds the two-in-flight pass.
- `split_encoder.py` — stage prefixes (fp16), fronts (fp32) and bodies (fp16) of a cut encoder; the torch stage
  reference, the per-stage ANE / GPU / CPU attribution and its propagation to the soft tokens; `warm` for the
  Neural Engine's initial load.
- `reader_coreai.py` — the reader + table export graph (`ReaderGraph`), the host read-out (`host_labels`,
  `host_argmax`, `collapse`, word Levenshtein), the gate inputs; `parity` (graph vs the reader's torch path, the
  reference npz), `absmax`, `info`.
- `export_reader.py` — torch.export → Core AI for the reader forms, the torch gate of the eager and decomposed graphs,
  AOT compiles, ANE region count; `--random-arch b|c` as a converter check.
- `gate_reader.py` — the runtime gate of a reader bundle (CPU / GPU / ANE, AOT or `--jit`) with the bars, drift and
  red arm; `warm` (a Neural Engine bundle's initial load), `score` (jiwer).
- `bench_pipeline.py` — reader timing (`prepare`, `run`, `ab`), the long recording end to end (`long`), its torch
  path (`torch-long`), `cross` (one path's encoder windows through another reader), `score` (jiwer).
- `strip_ship.py` — the shipping pair: `strip`, `aot` (h19p), `compare` (Mac runtime, stripped vs gated).
- `_common.py` — paths, fixtures, metrics.

## Credits

Weights: EmbeddingGemma 2 by the Gemma Team, Google DeepMind (Apache-2.0). Graph reference: Hugging Face
transformers `modeling_gemma4.py` / `modeling_embedding_gemma2.py` (Apache-2.0). The banded attention, the folded
q / k scales and the max-abs RMSNorm follow the published design of EverPaper's Core AI port
([EverPaper/embeddinggemma-2-coreai](https://huggingface.co/EverPaper/embeddinggemma-2-coreai) v1.2, Apache-2.0;
the max-abs idea credited there to the FluidInference Core ML port); no code was copied. The comparison rows run
EverPaper's published audio asset unchanged. Reader: trained on LibriSpeech train-clean-100 (CC BY 4.0); the word table
is EmbeddingGemma 2 text embeddings of the LibriSpeech lexicon's words (Gemma Prohibited Use Policy applies); the gate
utterances are LibriSpeech test-clean (CC BY 4.0); the long recording is LibriVox (public domain) with the Project
Gutenberg #1661 text.
