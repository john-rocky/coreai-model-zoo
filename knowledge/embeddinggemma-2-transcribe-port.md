# EmbeddingGemma 2 Transcribe: a frozen audio encoder and a trained word reader on Core AI

Lessons from porting the audio path of [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2)
(revision `914f7f89`, Apache-2.0) to Core AI as an English speech-to-text pipeline: the Gemma 4 audio tower +
`embed_audio` as one static graph, and a reader trained here (v2: 19.6 M parameters, v1: 13.3 M; CTC over a
200,000-word cosine table) as a second graph with the table inside. One fact per line. Card:
[`models/embeddinggemma-2-transcribe`](../models/embeddinggemma-2-transcribe/README.md); code and gates:
[`conversion/eg2_transcribe`](../conversion/eg2_transcribe/README.md); records: the `provenance/` folder of
[mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI](https://huggingface.co/mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI);
the reader's release note: `RELEASE.md` of
[mlboydaisuke/EmbeddingGemma-2-Transcribe-Reader](https://huggingface.co/mlboydaisuke/EmbeddingGemma-2-Transcribe-Reader).

## Contract

- Encoder graph: `features [1, Tb, 128]` + `half_mask [1, Tb/2]` → `soft_tokens [1, Tb/4, 512]`; the shipped bucket is Tb = 3000 (30 s).
- Frames: n_frames = ⌊(samples − 161) / 160⌋ + 1 (160 zeros in front, frames of 321 cut to 320, only frames whose window is all audio); 480,000 samples give 2,999 frames.
- Tokens: ⌈⌈n_frames / 2⌉ / 2⌉, 25 per second (40 ms each).
- `half_mask` is the only mask a right-padded window needs: after the stride-2 subsampling every op is causal or per token, and only the 3 × 3 convolution's one-frame look-ahead reaches a valid token.
- Without `half_mask`, a window with n_frames % 4 == 2 gets a wrong last token (cosine 0.85–0.88): the second convolution reads the half frame ⌈n_frames / 2⌉.
- Attention is local and causal: each query sees itself and the 11 tokens before it (12 keys) in every layer; `attention_context_left = 13` gives a past horizon of 12 keys including the query, not 13 chunks of context.
- A window's start lacks left context: 83 tokens (3.32 s) of overlap bring every token to per-token cosine mean ≥ 0.999 and min ≥ 0.99 against a continuous run; the pipeline uses 96 (3.84 s) and keeps only the new tokens.
- The last token of a 30 s cut misses one input frame and comes out wrong (mean cosine 0.487 against the continuous run): drop it, or pass one token (640 samples) of audio past the window.
- Reader graph: `soft_tokens [1, 750, 512]` + `token_mask [1, 750]` → `top1_index` (int32), `top1_logit` = scale × max cosine, `blank_logit`; the 200,000 × 768 table and the rotary positions are graph constants.
- Host rule: blank where `blank_logit > top1_logit` (strict, a tie reads the word, as the reader's own greedy decoder does); merge repeats, drop blanks.

## Authoring

- Re-authored from `model.safetensors`: 2 subsampling conv blocks + `input_proj`, 12 conformer layers, `output_proj` 1024 → 1536, then `embed_audio` (RMSNorm without scale, Linear 1536 → 512).
- Banded attention: logit = q_t · (k_{t−d} + relk_d) for d = 0..11, with relk_d = `relative_k_proj(sinusoid(d))` a per-layer constant; no block / unfold tensors of rank 5.
- Keys left of the clip start get an additive −1e4 after the logit softcap (a constant of the graph, not an input).
- The query scale (head_dim^−0.5 / ln 2 × softplus(per_dim_scale)) and the key scale (ln(1 + e) / ln 2) fold into `q_proj` / `k_proj` and into their clip bounds: clamp(Wx, lo, hi) · s = clamp(sWx, s · lo, s · hi).
- The clipped linears keep their min / max ops; the ±1e10 "gradient clipping" clamps are dropped (no-ops at inference).
- float16 headroom: the residual stream reaches 383.4 (layer 10); its element-wise square overflows float16 (65,504) while the mean of squares peaks at 426 → every RMSNorm divides its row by the row's max |x| (floored) before squaring.
- The two subsampling LayerNorms (over the channel axis of NCHW) stay plain in float16: their inputs stay ≤ 51, and their max-abs variant makes the h16c Neural Engine compile fail.
- The float16 encoder takes and returns float16; the float32-arithmetic forms (fp32, fp16w32, int8) keep float32 I/O.
- coreai-torch 0.4.1 lowers `layer_norm` on a float16 input with float32 statistics: the reader's float16 graph writes LayerNorm out by hand (mean, centring, max-abs, rsqrt) to stay all-float16.
- The 200,000-way argmax converts in coreai-torch 0.4.1 (reduce_max + argmax → int32); a top-k fallback is not needed.
- coreai-opt int8 with a global weight spec tries to quantize the rank-1 norm weights (`ValueError: axis 1 is out of bounds for tensor of rank 1`); configure `module_type_configs` for `torch.nn.Linear` only.
- Exports are not byte-reproducible (the converter names resources at random): keep the gated bundle, do not re-export to "the same" bytes.

## Parity

- Reference: transformers 5.19.0, float32, the default attention (sdpa).
- transformers 5.19.0 with `attn_implementation="eager"` inverts the audio tower's mask (`masked_fill(mask.logical_not(), -1e9)` applied to a float mask): soft-token cosine mean 0.54 against the default; fixed on transformers main from 22cfdfa.
- A SentenceTransformer self-check that loads both sides with `eager` agrees with itself at cosine 1.0 and does not catch the inversion.
- Encoder bar: per clip, per-token cosine mean ≥ 0.999 and min ≥ 0.99 on 20 LibriSpeech test-clean clips; red arm = `half_mask` all ones must break the last token of the 3 clips with n_frames % 4 == 2.
- The float16 failure point is near-silent tokens (log-mel −6.91 = log 0.001 in every bin), the same tokens on every backend: Mac Neural Engine min 0.888, CPU (BNNS) 0.909, EverPaper's Neural Engine asset 0.905, GPU 0.99645 (pass).
- In PyTorch, rounding only the input features to float16 already puts those tokens at 0.996; one layer alone in float16 (layer 6) gives 0.979: input sensitivity, not one wrong op.
- Moving 12 of the 115,584 inputs of one clip by one float16 step puts one token anywhere from 0.972 to 0.99996 (7 of 24 trials below 0.99): an fp16 path's per-token minimum is a number, not a margin.
- The app path is judged by its task: fixture transcripts ≥ 19/20 equal to the PyTorch path and long-recording WER ≤ PyTorch + 0.1 pt; the log-mel extractor is gated on its own (≤ 1e-5 from transformers', per-token cosine ≥ 0.99999 through the float32 encoder).
- Reader float16 vs its float32 PyTorch forward: 2 of 44,162 labels differ on the Mac GPU; the WER on 200 test-clean utterances is equal (5.878 %).
- The CPU (`cpu_only`) float16 reader returns `top1_logit = 0.0` wherever the true maximum cosine is negative (blank tokens; the labels do not change); the GPU and the Neural Engine return the negative value.
- A red arm can lose its teeth when the weights change: with reader v1, `token_mask` all ones changes no label at T = 250 (the reader learned to ignore zero padding) → add an arm that breaks the input by construction (the second half of the valid tokens set to 0 changes 10/10 utterances).

## Neural Engine

- Mac (h16c): the float16 encoder forms 1 ANE region once the subsampling LayerNorms are plain; with their max-abs variant `ANECCompileOffline` fails and `coreai-build` still exits 0 with a GPU-only bundle.
- float32 arithmetic (fp32, fp16w32, int8 weight-only) forms 0 ANE regions: a silent GPU fallback; count the `*_ANE_region_*` names in the compiled bundle.
- Mac Neural Engine encoder: 85.21 ms per 30 s window against 40.02 ms on the GPU, and it misses the per-token bar at the near-silent tokens.
- Stage attribution puts the ANE-specific error in the subsampling and layer 0 (cosine 0.933 after layer 0 against 0.9986 on the GPU), amplified later at near-silent tokens; hidden-state cosine stays ≥ 0.99987 to layer 8 and hides it.
- A float32 front on the GPU + a float16 body on the Neural Engine passes when cut after layer 7 (min 0.99671) or layer 8 (0.99880), not after the subsampling (0.974) or layer 4 (0.988).
- With two windows in flight (front of window i + 1 with the body of window i) the layer-7 pair runs 30.96 ms per window (×969); one window alone takes 57.40 ms.
- The layer-7 pair is published as a second encoder form, Mac only, for running the encoder alone on long audio: `macos-encoder-split7-30s/` at the root of the HF repo, not under `macos/`, because swift-transformers' `snapshot(matching: ["macos/*"])` matches with `fnmatch(…, 0)`, whose `*` also crosses `/`, and the app would download a folder it never reads.
- Two exact float16 rewrites do not lift the Neural Engine's minimum: the subsampling LayerNorm mean folded into the conv weights (0.896), RMSNorm's mean of squares as a matmul (0.404, every clip).
- Small graphs probed alone on the Neural Engine can err where the same ops inside the full graph do not (relative error 13 % against 0.1 %): check an op-level probe against the full graph before reading it.
- The reader + table in float16 forms 2 ANE regions that hold no weights (the weights stay in the MPSGraph package): 45.55 ms against 22.45 ms on the GPU, peak footprint 7.0 GB against 1.0 GB.
- The Mac's Neural Engine pays off for the reader beside the GPU encoder: encoder GPU ∥ reader ANE, two windows in flight, 5.68 s against 10.68 s one window at a time for a 3,257.4 s recording.
- A Neural Engine AOT bundle's initial load on the Mac specializes for minutes (encoder 116–240 s), later loads 0.01 s: warm it outside a measurement window.
- iPhone 18 Pro (h19p) at 30 s runs no graph of this port on the Neural Engine: Neural-Engine-preferred JIT bundles are placed on the GPU (outputs bit-identical to the GPU route).
- The encoder's Neural Engine AOT h19p (1 region, offline compile exit 0) is refused at load with `failedToSpecialize`.
- The reader's Neural Engine AOT h19p loads in 0.88 s and jetsam then ends the app at 3.77 GB (the default per-process limit, no increased-memory-limit entitlement).
- The 10 s encoder does run on the phone's Neural Engine (15.81 ms), with outputs bit-identical to the Mac Neural Engine's: the same arithmetic on A19 Pro and M4 Max, the same failing tokens.
- Judge placement by comparing outputs with the GPU route: a Neural-Engine-preferred bundle placed on the GPU still uses less CPU per call than a GPU-preferred one.

## Speed

- Mac M4 Max, one 30 s window: encoder fp16 GPU 40.02 ms (×750), reader fp16 GPU 22.45 ms (×1,336).
- Mac whole transcriber: 3,257.4 s in 5.681 s (×573) with the encoder on the GPU and the reader on the Neural Engine, two windows in flight; 8.093 s (×402.5) one window at a time on the GPU.
- iPhone 18 Pro: encoder fp16 GPU AOT h19p 59.60 ms (×503); reader 32.99 ms (timed with interim weights in the same graph).
- JIT `.aimodel` and AOT run at the same speed with the same outputs (Mac 40.50 vs 40.22 ms, iPhone 59.88 vs 59.60 ms); AOT saves the phone's on-device specialization (load 2.94 s + 10.1 s opening call → 0.91 s + 0.20 s).
- float32 costs 1.06× float16 on the Mac GPU (42.47 vs 40.02 ms) and 3.1× on the iPhone GPU (188.53 vs 60.08 ms).
- int8 weight-only (Linear, per block of 32) compiles to the float32 size (1,174 MB): the dequantization folds into float32 constants; no gain in speed or download.
- The 200,000 × 768 table product + argmax is about 19.5 of the reader's 22.45 ms; in numpy on the host it takes 124.9 ms → keep it in the graph.
- 10 s windows lose to 30 s once the 96-token overlap is counted (Mac ×345 vs ×654 of new audio per call).
- The iPhone's GPU slows as the load goes on even at thermal state nominal (56.8 → 60.9 ms over 8 rounds; long-recording passes 11.53 → 13.45 s): alternate the arms, start each pass at nominal, and do not settle a few-percent difference in one launch.
- Two stages in flight help on the Mac (GPU + Neural Engine) and not on the iPhone, where both calls share the one GPU (stage medians 59 / 35 ms alone, 91–97 / 48–53 ms together).

## Reader

- The reader learns what a frozen projection cannot: soft-token span mean → `embedding_projection` → nearest word gives top-1 0.013 (5,000-word vocabulary, 100 utterances), span audio through SentenceTransformer gives 0.41.
- Objective: CTC over 200,000 words + blank, word logits = scale × cosine(h, table) with a learned scale (exp of a learned log, 91.218 at v1, 216.351 at v2) against a fixed table of EmbeddingGemma 2 text embeddings (prompt `task: search result | query: `).
- Data (v1): LibriSpeech train-clean-100 without the utterances that hold an out-of-vocabulary word (26,422 utterances, 92.8 h); soft tokens computed once in float32, stored as float16.
- v1: 4-layer transformer from scratch, 8 epochs on one M4 Max (2.72 h active), dev-clean WER 6.57 %, test-clean 6.81 %, test-other 15.10 % (greedy, no language model).
- CTC over 200,001 classes without moving [T, B, 200001] to the CPU: run `ctc_loss` on the columns of the utterance's words + blank only, compute the normalization over all 200,001 separately and correct the gradient by exp(log p); loss and gradient match the full CTC to 2e-15 in float64.
- Learn the scale through its log: AdamW moves a raw scale only by about the learning rate per step.
- MPS's caching allocator grows without bound on large buffers of changing shape (57 → 99 GB by step 200): pad to fixed lengths, cap with `torch.mps.set_per_process_memory_fraction`, call `empty_cache()`; a 10-step timing run does not show it.
- After a dev decode, Metal allocations outside torch's pool stay until the process ends: run each epoch in its own process.

### Reader v2 (the 960 h set, 6 layers)

- v2: the same body with 6 layers (19,573,762 parameters) on the three LibriSpeech training sets (961 h; 890 h without the utterances that hold an out-of-vocabulary word), CTC as v1, 9 epochs with early stopping on dev-clean (best epoch 7) on one A100: test-clean 4.21 %, test-other 10.42 % (v1 6.81 / 15.10 %). The 4-layer body trained the same way reached 3.88 % on dev-clean against 3.75 % for 6 layers.
- On an A100, TF32 matmuls train 2.4× faster per token than float32 (29 min per 960 h epoch); decode dev, test and fixtures in float32, and record which matmul mode trained the weights (`reader_config.json` → `weights.format`).
- Batching the encoder (16 utterances of similar length per call, right-padded) keeps the soft tokens at per-token cosine ≥ 0.999999994 against one at a time; the 961 h took 80.6 min (×715) on the A100.
- The export reads the depth from `reader_config.json` (`architecture.layers`, absent = 4): reader v1's config rebuilds v1's graph op for op (465 ops, the same `coreai-build inspect` distribution), v2's graph has 639. The bundle name carries the release's version (`eg2_reader_v2_fp16_T750`), so two readers never share a path; the AOT `main.hash` differs anyway, because the graph differs.
- v2's residual stream reaches 1,160 (v1: 60.5); with every norm dividing by its row's max |x| first, the float16 graph keeps a ×56 headroom and passes: Mac GPU fixture 20/20, test-clean (200 utterances) WER 3.633 % against 3.655 % for PyTorch float32 (2 of 44,162 labels differ); the Neural Engine 20/20 and 3.721 %.
- Every label the float16 graph changes is a near-tie, and one variable at a time names the rounding: a blank and a word 0.005 logits apart in float32 land on either side of a float16 step (0.0156 at logits 16–32) on the GPU, while the float16 table alone and the float16 graph in PyTorch keep the word; an exact float16 tie on the Neural Engine is read as the word by the strict host rule (`blank > top1`); one word ⇄ word tie (top-1 − top-2 cosine 1.2e-5) flips with the float16 table alone.
- With the learned scale at 216 (v1: 91), `top1_logit` moves by up to 8.6 (GPU) and 20.5 (Neural Engine) at blank tokens, where the best word's cosine is small and its choice arbitrary: judge the labels, not the logits.

## Runtime traps

- An AOT `.aimodelc`'s `main.hash` is the hash of the program without its weights, and the Core AI cache is keyed by it: a re-exported AOT bundle with new weights at the same path loads the previous weights' specialization (gates reproduced the old WER to the decimal) until the cache entry is deleted.
- A JIT `.aimodel`'s `main.hash` is sha256(`main.mlirb`) and changes with the weights; an app that replaces its model should clear its own Core AI cache when the model's sha changes (a new folder name does not change the key).
- Two calls of one function issued together return wrong outputs (encoder on the GPU: max |Δ| 8.04 against the sequential pass): one call per function at a time; two different bundles in flight are fine.
- Swift: an AOT `.aimodelc` opened through a symbolic link fails with `failedToSpecialize`; a JIT `.aimodel` does not care → resolve the path.
- iPhone 18 Pro: the 1.17 GB float32 JIT `.aimodel` aborts 2–3 s into the load (SIGABRT, `operator new` in MPSGraph's `CanonicalizeMatMulTransposeConstantRHS` → `foldTransposeOp`); the AOT h19p of the same graph loads in 1.26 s.
- A process-list poll can report an app alive for 900 s after it crashed: read the device's crash reports newer than the launch.
- An exported `main.mlirb` names the converting machine's source files (absolute paths) in its debug locations, and every AOT bundle compiled from it carries them; `grep -rIl` skips binary files → strip with coreai-torch `strip_debug_info` on one program object and save that object (operations unchanged, new `main.hash`, outputs bit-identical on the Mac).
- `coreai-build compile --preferred-compute neural-engine` exits 0 when the ANE compile fails: read the region names, not the exit code.
- The Swift SDK's `NDArray(scalars:shape:)` walks a Sequence (about 38 ns per element, 29 ms for a 768,000-element hidden state): fill `NDArray(shape:scalarType:)` through `mutableView` instead (0.8 ms).
- An iOS AOT bundle loaded on a Mac can wedge the GPU stack: compile h19p on the Mac, never load it there.

## Extractor port

- `Gemma4AudioFeatureExtractor` takes the magnitude |rFFT| (not the power), periodic Hann, 160 zeros in front, 128 HTK mel bins over 0–8 kHz without normalization, `log(mel + 0.001)`.
- vDSP's float32 FFT puts the log-mel up to 3.8e-4 from the extractor's on quiet bins; a float64 FFT with |X| rounded to float32 gives 4.77e-7, bit-identical on the Mac and the iPhone, 0.05 s (Mac) / 0.16 s (iPhone) for 54 minutes of audio.
- Ship the mel filters as constants: the 257 × 128 float64 matrix of `mel_filter_bank(257, 128, 0, 8000, 16000, norm=None, mel_scale="htk")`; rounded to float32 it moves the log-mel by 4.8e-7 and the fp16 encoder's minimum cosine from 0.99645 to 0.99573.
- Frame count: ⌊(n − 161) / 160⌋ + 1, not ⌊(n − 321) / 160⌋ + 1 (the 160 zeros come in front of the 321-sample frames); the latter is one frame short on every utterance (features max |Δ| inf, cosine min 0.44 in the gate).
- The Python extractor needs a list (`[wav]`; a bare 1-D array fails) and, for a continuous long recording, `max_length=None, truncation=False` (the default cuts at 480,000 samples).
