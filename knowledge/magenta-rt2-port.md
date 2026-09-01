# Magenta RealTime 2 (small) → Core ML / ANE — port learnings (NOT shipped)

**Outcome: not shipped, not dead — parked with the first Core AI blocker identified.**

An earlier draft of this doc called it DROPPED on the strength of this repo's music-gen gate
([`music-generation-stable-audio.md`](music-generation-stable-audio.md), "check whether MLX / CoreML
already ship it on iPhone"). That gate was applied to the wrong stack: the existing public port
([`mattmireles/magenta-realtime-2-iphone`](https://github.com/mattmireles/magenta-realtime-2-iphone),
Apache-2.0, HF weights cc-by-4.0) is **Core ML**, and the App Store app (*Murmur*, id6776807467) is a
product, not a stack. **No Core AI port of MRT2 exists**, and §7 shows Core ML and Core AI are not
interchangeable on this graph — 3.83 ms against 10.85 ms for the identical model on the identical
phone. So "someone already did it in Core ML" does not answer whether it works here.

What is established: real-time on an iPhone holds on **both** runtimes. Core ML 3.83 ms and Core AI
10.85 ms for the frame graph, against a 40 ms budget, on a model Google itself scopes to Apple
Silicon **Macs** (their own line is correct for the *GPU* — our Core ML GPU numbers on the phone are
29.42 + 17.04 = 46.5 ms, over budget). What is not established: any of it with real weights — every
"ours" number here is a shape-and-dispatch stub. The transferable findings are §2 (sampler), §3
(codec), §4 (quantization) and §7 (placement).

Model: `google/magenta-realtime-2`, `mrt2_small` — 230M-param Depthformer + SpectroStream codec,
continuous 25 Hz steerable music generation. Weights cc-by-4.0, code Apache-2.0, `gated: false`.

---

## 1. The head-to-head — measured twice, and the first answer was wrong

iPhone 17 Pro (A19 Pro), iOS 27, Core ML, `.cpuAndNeuralEngine` pinned, thermal nominal, one harness
for both sides (`_mrt2/probe/devbench/`, generic feature provider built from each model's own
`inputDescriptionsByName`, predictions that throw are reported rather than timed).

| per 40 ms frame | prior port (fp16) | ours, REAL weights | ours, earlier stub |
|---|---|---|---|
| LLM half | 13.46 (temporal 8.00 + depth 5.46) | **10.95** (p99 12.21) | 3.80 |
| SpectroStream decoder | 0.48 (fp16, 25/call) | **1.01** (fp16 + 2-stage rescale, 25/call) | 8.29 → 1.93 |
| | | ~~6.22~~ (fp32, before §3's rescale) | |
| **frame total** | 13.94 | **11.96** | 12.08 |

Both fit the 40 ms budget — ours at 3.35× real time — on a model Google scopes to Apple Silicon
**Macs**. Their line is correct for the *GPU*: our own Core ML GPU numbers on the phone are
29.42 + 17.04 = 46.5 ms, over budget. The ANE is what breaks it.

**The stub was 2.9× optimistic and an earlier version of this doc drew a conclusion from it.**
Everything in the "stub" column above came from a shape-and-dispatch mock with random weights that
also skipped `primer_hybrid`'s second norm, the attention sinks, and the real embedding gather. On
the real weights our LLM half leads by **1.23×, not the 3.6× that mock implied**, and we are
**behind overall** because our decoder is fp32 where theirs is fp16 (§3). Treat any latency measured
on a stub as an upper bound on speed and nothing else.

## 2. What we had that they did not: the depth loop's sampler

MRT2 emits 12 RVQ tokens per frame through a strictly sequential depth loop — step *k+1* consumes
the **embedding of the token sampled at step k** (`depthformer.py:715-727`), so sampling must live in
the graph. The natural authoring is `argmax(logits + gumbel)` then a gather into the embedding
table. The prior port does exactly that (`exporters/convert_depth_body_rollout.py`: "Gumbel-max over
a static top-k set (host supplies noise) … token-embedding feedback is an in-graph gather").

**The ANE cannot run `argmax` or `gather`, so each of the 12 steps forces a partition crossing.**
Measured on M4 Max, same stub, same harness, only the sampler formulation changing:

| depth-loop sampler | ANE | GPU |
|---|---|---|
| `argmax` + in-graph `gather` (the prior port's shape) | **45.76 ms** | 22.17 |
| `argmax` + `arange == idx` one-hot | 53.26 | 25.87 |
| no sampler at all (floor) | 10.99 | 15.91 |
| **`onehot = (noisy >= max(noisy))`, normalised, then matmul** | **11.53** | 26.78 |
| … + per-codebook vocab slicing | **7.48** | 14.95 |
| … + 4-bit LUT palettization | **4.13** | 17.27 |

The rewrite is **exact, not an approximation**: gumbel-max *is* argmax, and `x >= max(x)` is the same
one-hot vector; `onehot @ E` is the gather. The integer index is never needed inside the graph —
the codec consumes the RVQ *embedding*. Cost: ~34 ms/frame on M4 Max, and it is the difference
between their 5.46 ms depth rollout and our 3.79 ms whole frame.

Caveat we did not close: the reference also applies `top_k=40` at temperature 1.3. `>= max` is exact
gumbel-max **without** top-k. The prior port keeps top-k by baking a static top-k set at export. If
you take this technique, settle top-k against the reference distribution rather than assuming.

Not novel vs them: **per-codebook vocab slicing.** `rvq_index` is a compile-time constant in the
unrolled loop and the reference masks logits to `[6+1024k, 6+1024(k+1))`, so each step needs only its
own 1024 rows of `to_logits [768,12294]` and of the embedding table. Their `gumbel_noise` input is
`[12, 1024]` — they already slice.

---

## 3. What they had that we did not: the codec

Two decisions, both theirs, both right, both measured:

1. **Batch the decoder.** Their `SpectroStreamDecoder` takes `decoder_embeddings [1, 25, 256]` — one
   second of audio per call — for 12.00 ms, i.e. 0.48 ms/frame. Ours took one frame per call at
   8.29 ms. Batching ours to 25 frames took it to 1.93 ms/frame.
2. **Keep the iSTFT out of the graph.** Core ML has no FFT op, so we authored the inverse STFT as an
   explicit 481×960 DFT matmul plus a Python-unrolled overlap-add. At one frame it merely cost more;
   **at 25 frames the model would not load on device at all** (`LOAD FAILED … .cpuAndNeuralEngine`;
   the 8-frame build exceeded the watchdog during ANE compile). Their decoder returns
   `decoder_stft [1, 96, 480, 4]` and leaves iSTFT/overlap-add to the host.

Even at matched batching and matched scope (25 frames, no iSTFT) theirs is 0.48 ms/frame against our
1.93 — a 4× gap that is ours to explain, not theirs: our decoder is an approximated stub
(19.6M params against the real 35.7M with `channel_splits=2` weight sharing).

**Rule to carry:** for a streaming codec on the ANE, batch the call and put the spectral inverse on
the host. Do not put an unrolled overlap-add in the graph.

### fp16 needs an exact rescale, and both of its constants must be measured

The decoder's activations reach ~1e7 inside the grouped section while fp16 tops out at 65504, so a
plain `compute_precision=FLOAT16` conversion saturates mid-stack and returns a **finite, correctly
shaped, meaningless** result — cos 0.035, with the final output back at ~1.1e4 so nothing looks
wrong at the boundary. It is not a few outliers either: at decoder_5, 127 of 128 channels and the
99.9th percentile are over.

The fix (the prior port's `apply_fp16_safe_rescale`, reimplemented here as
`SpectroStreamDecoder.rescale_for_fp16`) divides the stream by S across the hot region and multiplies
back at the exit:

- **Insert the scale only at top-level serial boundaries.** Scaling inside a residual branch mixes
  scales at the add and destroys the function.
- **Weights untouched, every bias in the region × 1/S** — a bias adds at output scale. The shortcut
  convs' biases count.
- **ELU must be replaced, not rescaled.** A constant factor does not commute with ELU. Use
  `relu(y) + (α/S)·(exp(clamp(y, max=0)·S) − 1)`, which is exactly `elu(S·y)/S`; the clamp keeps the
  exponent ≤ 0 so `exp` cannot overflow, and if an fp16 multiply saturates a large negative to −inf
  then `exp(−inf)=0` lands on the mathematical limit −α/S. (Retuning ELU's α to α/S instead changes
  the exponent slope and is *not* equivalent.)

Both constants have to come from your own profile, not from the prior port's:

- **S.** Theirs is 128 for a measured peak of 2.65e6. Ours peaks at 1.0e7 on random codes and
  **1.54e7 on real generated ones**, where 128 leaves the stream at ~1.2e5 — still over — and cos
  stalls at 0.983. Profile on real content: random draws understate the peak by 1.5×.
- **Where the exit goes.** Ending the region after the last hot block is wrong: decoder_6's *native*
  magnitude is 5.5e5, so multiplying back there overflows immediately. The symptom is a plateau —
  cos stuck at 0.988 no matter how large S gets — which reads like "S is too small" and is not.
  The region has to extend through the output conv, whose output is ~1.1e4. This is what the
  reference's `restore_margin` is for: advance the exit until the boundary's native absmax is safe.

- **One S cannot serve both ends, and the two requirements point opposite ways.** Raising S buys
  overflow margin in the deep units but drives the *early* ones toward fp16 subnormals, and ANE
  quality falls monotonically with it — swept on real exports: S=512 → 49.3 dB, 1024 → 48.9,
  2048 → 42.6, 4096 → 40.2, 8192 → 34.2. "Bigger S is safer" is wrong.
  So use a **two-stage schedule**: 1/1024 from decoder_1, an extra ×1/8 (→1/8192) before decoder_5,
  restored after the output conv. The step is a power of two, so it stays bit-exact. Measured
  peaks that motivate the split: decoder_1..4 reach 1.05e6, decoder_5..6 reach 1.54e7.

Result: fp32 SNR 115.3 dB (against 115.4 dB untouched — the transform is exact), and converted at
fp16, **cos 0.999994 on ANE / 0.999999 on GPU, finite on both, 72 MB instead of 143 MB, and
1.01 ms/frame on device instead of 6.22**. Against a flat S=512 the two-stage schedule measures the
same quality and the same speed with **35× overflow headroom instead of 2.2×** — it buys robustness
on unseen content, not milliseconds.

---

## 4. Quantization: 4-bit works, but not where the parameters are

The prior port reports six 4/6/8-bit PTQ variants failing parity and ships fp16/fp32 only (~523 MB).
That reproduces here — with the cause located. Method: palettize the real checkpoint, load it back
into the **MLX reference**, and replay the fp32 model's own sampled trajectory through it, so both
models see an identical history and every logit difference is the weights alone
(`_mrt2/probe/parity_int4.py`, 25 frames = 372 depth steps).

The metric has to be distributional, not deterministic: this model samples at temperature 1.3 with
top-k 40, so what matters is how often two models draw the **same token** — `1 − TV` between the
top-k-truncated distributions. A bit-exact gate fails the int8 Google itself ships.

| recipe | logit cos mean / min | greedy argmax | sampling agreement |
|---|---|---|---|
| MLX affine int8-g64 (**what Google ships**) | 0.99992 / 0.99935 | 91.1% | **0.945** |
| 4-bit LUT g16, all 207 M matmul params | 0.9832 / 0.736 | 38.7% | **0.513** |
| 4-bit LUT g16 **+ per-channel scale** | 0.9865 / 0.739 | 35.8% | **0.513** |
| 4-bit LUT g16, **temporal body only** (182.5 M) | 0.9988 / 0.962 | 81.2% | **0.880** |
| temporal 4-bit + the other 24.6 M at 8-bit LUT | 0.9987 / 0.957 | 79.0% | **0.861** |

- **Weight SNR is not the lever.** Per-channel scale raised the worst tensor 15.16 → 16.80 dB and
  moved sampling agreement by *nothing*. Do not tune the palettizer; 4 bits does not carry these
  logits.
- **The damage is not where the parameters are.** The temporal body is 182.5 M of 207 M and takes
  4 bits nearly free; the ~25 M outside it — depth body, the 768×12294 logits head, the decoder
  embedding, the conditioning encoder — is what collapses it.
- Resulting bundle ≈238 MB (temporal int4 91 MB + sensitive 25 M int8 + embeddings/codec fp16)
  against the prior port's ~523 MB fp16 set.
- **Grouping axis trap:** group along the *output-channel* axis, and for attention
  `output_projection` kernels shaped `[d_model, heads, dim_per_head]` that axis is **axis 0**, not
  the last. Getting it wrong cost 4.25 dB on the worst tensor (10.91 → 15.16 dB).

---

## 5. Facts about the model worth not re-deriving

- **`MagentaRT2ModelBase` is `base`, not `small`.** Reading base defaults gives 20 layers / d=3072 /
  window 25 — all wrong. `small` (`mlx/model.py:490-500`): temporal 12L × d=1024 × ffn 4096
  (non-gated, tanh-GELU, bias), 8 heads × 128; depth 2L × d=768 × ffn 3072, 6 heads × 128, no sinks;
  window 41 both self and cross; **NoPE**; `primer_hybrid` norm (two RMSNorms per residual branch);
  learned `per_dim_scale` = `1.442695041 · (1/√128) · softplus(w)`, not `1/√d`; one learned
  attention sink per temporal attention whose key is **pre-divided by the query-scale vector**
  (`sequence_layers/mlx/attention.py:336`).
- **The conditioning "encoder" has no transformer body** — `body=sl.Identity.Config()`
  (`model.py:362`). Embedding + LayerNorm only.
- **All state is constant.** `mrt2_small_state.safetensors` is 165 tensors / 8.66 MB: 48 ×
  `(1,41,8,128)` bf16 KV, 24 × `(1,41)` bool masks, 27 × `(1,)` int32 ring pointers, 31 float32 codec
  conv/OLA buffers. No growing cache, no prefill wall, batch 1 (CFG arrives as discretized
  conditioning *tokens*, not a batch expansion).
- **The on-device weight set is one file.** `checkpoints/mrt2_small.safetensors` already contains the
  SpectroStream decoder and RVQ codebook under `params/soundstream/*`, byte-identical (SHA-256) to
  the standalone `decoder.safetensors`/`quantizer.safetensors`. Summing them triple-counts 277 MB.
- **The 418 MB MusicCoCa text encoder does not need to ship.** Style reaches the transformer as 12
  int32 RVQ codes per frame; upstream already supports injecting raw codes and precomputed
  embeddings, and prompt blending happens in *embedding space* before quantization. Only free-form
  runtime text is lost.
- **The shipped `.mlxfn` is int8 group-64, not bf16.** 244.09 MB (229.7 M × 1.0625 B) + 142.74 MB
  fp32 codec + 67.11 MB fp32 codebook = 453.94 MB against a 455,654,550 B file. Reading
  455 MB ÷ 229.7 M ≈ 2 B/param and concluding bf16 forgets the codec and codebook live in the same
  file.
- **`mlx==0.31.1` or the `.mlxfn` will not load** (`RuntimeError: [import_function] Invalid string
  size` on 0.32.x — the version is a literal string at byte 8 of the file).
- **MLX takes the first traced signature regardless of input shape.** The `.mlxfn`'s second signature
  (`forced_tokens`, the depth-skipping path) is unreachable through `mx.import_function`; measuring
  the depth loop "by subtraction" silently returns byte-identical audio and a 0.00 ms delta.

## 6. Method notes

- **Google's Mac-only line is correct — for the GPU.** Our own Core ML GPU numbers on the phone:
  frame 29.42 ms + codec 17.04 ms = 46.5 ms, over the 40 ms budget. MLX and their C++ engine can
  only use the GPU. The ANE is what breaks it.
- **The `.mlxfn` frame is kernel-launch-bound on the Mac**, not compute-bound: 3306 graph nodes at
  ~3.0 µs/dispatch ≈ 9.95 ms of the 10.8 ms frame; two concurrent processes each still ran at
  ~10.7 ms (≈29% GPU utilisation). This is why Google's own matrix lists a fanless M1 Air as
  real-time, and why MLX timings do not transfer to a Core ML port.
- **Count ANE residency, never infer it.** `MLComputePlan` on device for our frame graph: 11412 ops,
  4303 assigned to the ANE, **0 GPU, 0 CPU**.
- **Benchmark serially.** An overlapping palettization run inflated one sweep here and every number
  had to be retaken. The first reference figure (25.2 ms/frame) was taken on a busy machine and was
  2.3× pessimistic; quiet it is 10.8 ms.
- **A non-UI iOS app is SIGKILLed at ~25 s of wall clock**, and the first ANE compile of a 110 MB
  graph alone takes 13–14 s. Run one config per launch and roll out partial stats; sustained/thermal
  testing needs a real UIKit app. What we did get: median held at 3.84 ms through 1200 consecutive
  frames (48 s of audio in 4.6 s), thermal nominal → fair.
- `MLComputePlan.load_from_path` on an `.mlpackage` aborts the process with a C++ exception and takes
  buffered stdout with it. Flush, or run it out-of-process.

---

## 7. Core AI vs Core ML on the same graph, same phone — placement, not submissions

The zoo ships **Core AI**, and the prior public port is **Core ML**, so "a Core ML port exists" does
not settle whether MRT2 works on this stack. It was exported both ways and measured back to back.

Export recipe (worked first try, 13 s, 452.1 MB `.aimodel`): `coreai_torch.TorchConverter()` +
`add_pytorch_module(..., input_names, output_names, entrypoint_name="frame")` +
`register_custom_torch_lowering` + `to_coreai()` + `optimize()` + `save_asset`, copying
`conversion/pocket-tts/export_flowlm.py:144-215`. AOT:
`xcrun coreai-build compile <x>.aimodel --platform iOS --preferred-compute neural-engine --architecture h18p --output <dir>`.
Exporter kept at `_mrt2/probe/export_mrt2_coreai.py`.

iPhone 17 Pro, iOS 27, same frame graph, same harness (`devbench/coreai_bench.swift`, output arrays
hoisted out of the timed loop), 100 iterations:

| runtime | median | p99 | load |
|---|---|---|---|
| **Core ML, `.cpuAndNeuralEngine`** | **3.83 ms** | 4.51 | 13,198 ms |
| Core AI, `preferredComputeUnitKind: .neuralEngine` | 10.85 | 13.68 | 26 ms |
| Core AI, `.gpu` | 11.99 | 12.76 | 1,132 ms |
| Core AI, `.cpu` | 10.77 | 11.36 | 3,177 ms |

**Three findings, and the first one corrects a reading that looks like a win.**

1. **`ANE_region` count = 1 is not necessarily good news.** The bundle compiles to exactly one
   `*_ANE_region_0_0.mlir.bc`, which reads like the ideal (one region = one submission = Core ML's
   delivery model). It is the opposite. The weights say where the graph actually went:
   `main-h18p-delegates/MPSGraph/.../resources.bin` = **450.9 MB** against an ANE region weight blob
   of **48.8 MB** — about **10% of the model is in the ANE region**, and that region is carved
   *inside* the MPSGraph package. One region because almost nothing was placed there. Count the
   region **bytes**, not just the regions; `find … -name '*ANE_region_*.bc.weights' -exec ls -la` next
   to the MPSGraph `resources.bin` is the check.
2. **The compute-unit preference moved nothing.** `.neuralEngine` 10.85, `.cpu` 10.77, `.gpu` 11.99
   — indistinguishable, and a 226M-param frame plainly does not run in 10.77 ms on the CPU. Read
   this as confirmation of the documented rule rather than as evidence on its own:
   `SpecializationOptions(preferredComputeUnitKind:)` is a *preference over the same allowed set*,
   not a restriction — `cpu_only()` is the exact one, and it is the parity option, not a performance
   option (`AGENTS.md`, "Timing with `cpu_only()`"). So three equal numbers are the expected
   behaviour of a preference and prove nothing by themselves. **The evidence that the graph is not
   on the ANE is the weight split in (1)**, not this triple.
3. **So the Core AI penalty here is NOT the per-region submission tax.** That mechanism —
   25 submissions, ~1.125 ms inter-submission gap, ~2.3 ms IOSurface round trip, ~48 ms/pass — is
   the recorded Parakeet-encoder finding and it needs many regions to bite. With one region it
   cannot apply, and Core AI is still 2.8× slower. The cause is placement: Core ML puts this graph
   on the ANE (`MLComputePlan`: 4303 ops attributed ANE, 0 GPU, 0 CPU, 7109 unattributed) and Core
   AI does not.

Core AI does win load decisively — 26 ms AOT against Core ML's 13.2 s on-device compile — which is
the "it buys load and not throughput" line, confirmed on a second, very different model.

**What this means for a Core AI port of MRT2:** the first problem is not reducing regions, it is
getting the graph onto the ANE at all. The same authoring that Core ML places 100% on the ANE (BC1S,
1×1 Conv2d, per-head SDPA, −40000 mask, fp16, static shapes, KV as readonly I/O) is only ~10%
accepted by Core AI's partitioner. Suspects visible in the compiled stats: `gather_nd` ×195,
`reduce` ×1434 / `reduce_sum` ×1227, and Float32 still present in `computeTypes` despite an fp16
model. Isolating which of those breaks placement is the next experiment, and it is cheap — the
export takes 13 s and the region-byte check is a `find`.

## 7b. Core AI with the REAL weights: the codec runs, the frame graph aborts

§7's Core AI numbers were the random-weight stub. Re-exported from the gated modules
(`_mrt2/probe/export_mrt2_coreai_real.py`), AOT-compiled `--platform iOS --preferred-compute
neural-engine --architecture h18p`, iPhone 17 Pro:

| asset | ANE regions | load | run |
|---|---|---|---|
| codec, fp16 + 2-stage rescale, 25 frames/call | 2 | 158 ms | **22.40 ms = 0.90 ms/frame** |
| frame, fp16 | **46** | 973 ms (succeeds) | **SIGABRT on the first run** |

The codec is fine — and *faster* than its Core ML twin (0.90 vs 1.01 ms/frame) with a 12× shorter
load (158 ms vs 1856 ms), which is the "Core AI buys load" line again. The frame graph loads and
then aborts, reproducibly, under `.neuralEngine`, `.gpu` **and** `.cpu` preferences alike, so it is
not a placement problem. Isolated by bracketing each stage with a flushed print: open → load →
allocate inputs → **first run → abort**. A C++ abort in the runtime is not a Swift throw, so `try?`
never sees it; without the stage prints SIGABRT is all you get.

**The stub does not predict this.** The same graph shape with random weights and three
simplifications compiled to **1** ANE region and ran at 10.85 ms; the real one compiles to **46** and
aborts. A simplified mock cannot be used to predict Core AI partitioning, let alone whether the
runtime will accept the graph.

**Bisected to a hard boundary: 11 temporal layers run, 12 abort.** Probes at
`_mrt2/probe/probe_coreai_abort.py`, each exported → `coreai-build compile` → device:

| probe | ANE regions | first run |
|---|---|---|
| depth loop only (12 unrolled RVQ steps) | 56 | **runs** |
| temporal, 1 layer | 9 | runs |
| temporal, 6 layers | 49 | runs |
| temporal, 9 layers | — | runs |
| temporal, **11 layers** | — | **runs** |
| temporal, **12 layers** | 53 | **ABORT** |
| full frame (12 temporal + depth) | 46 | **ABORT** |

What that rules out:
- **Not the region count.** The 56-region depth graph runs; the 46-region full frame aborts.
  This repo's Core AI story so far explains the runtime's *slowness* by region count; whether a graph
  *runs at all* is a different axis.
- **Not the asset size.** The 452 MB random-weight stub ran; the 442 MB real graph aborts.
- **Not a per-layer construct.** Attention sinks, streaming cross-attention and the 41-slot K/V
  inputs are all present at 1 layer and at 11, and both run.

It is depth of the temporal stack, and the cliff sits exactly at this model's own layer count.
Still unexplained: what resource 12 layers exhausts that 11 does not.

**Latencies across these probes are not comparable** — 6 layers measured 30.41 ms and 11 layers
8.48 ms, a 3.6× inversion, with load times swinging 0.6 s → 19.5 s → 8.1 s for growing graphs.
Only the binary (aborts / does not) is trustworthy here. Load time growing into the tens of seconds
is a second, separate problem from the abort: the codec asset loads in 158 ms, so AOT is working in
general and something about the deep temporal graph defeats it.

### The split runs — and is 10x too slow

Splitting the temporal stack 6+6 (`_mrt2/probe/export_frame_split.py`, seam = one 1024-d vector)
clears the abort: both halves load and run. It does not clear the budget.

| Core AI, iPhone 17 Pro, ANE | regions | load | median | min |
|---|---|---|---|---|
| frame part A (embed + layers 0-5) | 49 | 445 ms | 68.27 ms | 48.81 |
| frame part B (layers 6-11 + depth loop) | 50 | 1758 ms | 48.57 ms | 46.38 |
| **frame total** | | | **116.84 ms** | 95.19 |
| codec, 25 frames/call | 2 | 158 ms | 22.40 ms (0.90/frame) | |

Against a 40 ms budget and against **10.95 ms for the same frame in Core ML**. Iteration counts here
are small (10 and 5) and this session's Core AI probe latencies have been erratic, but the *minimum*
observed values already sum to 95 ms, so measurement noise does not reach the budget.

**Consequence for shipping:** the Core ML pipeline runs end to end at 3.34x real time (§1). The
Core AI pipeline does not make real time for this model: its codec is excellent (0.90 ms/frame,
faster than the Core ML twin), but the frame graph aborts as one piece and, split so it runs, costs
~10x what Core ML costs. A zoo entry claiming a real-time Core AI music generator would be claiming
something this port cannot currently do.

## 8. The port proper — frame-0 gate PASSES token-exact

`_mrt2/probe/mrt2_torch.py` re-authors mrt2_small in plain PyTorch from
`checkpoints/mrt2_small.safetensors` (AGENTS.md: a port is a re-author gated against the oracle, not
a conversion). It loads **229.7M params — exactly the checkpoint's `params/depthformer` count**, so
every weight is consumed and none invented.

Gate harness, two interpreters because the venvs are disjoint (the mlx env has no torch):
`gate_mlx_capture.py` drives the MLX reference for one frame from a fresh state, records the 12
depth-step logits and the tokens it sampled, and writes `oracle/gate_frame0.npz`;
`gate_torch_compare.py` replays those tokens teacher-forced through the torch model. Any difference
is the re-authoring, not sampling.

**Result, at matched fp32 precision: cos 0.99920–0.99988 — every step clears the 0.999 bar.**
Greedy argmax agrees on 9/12, and all three disagreements are near-ties in the *reference*:

| step | reference top1−top2 margin | our max\|Δ\| on those two | outcome |
|---|---|---|---|
| 3 | **0.0005** | 0.054 | flipped |
| 7 | 0.3097 | 0.3057 | flipped (on the boundary) |
| 11 | 0.0941 | 0.2501 | flipped |
| the other 9 | 0.17 – 1.47 | 0.03 – 0.13 | kept, margin 4–26× the delta |

On ±30 soft-capped logits the reference's own top two are 5e-4 apart at step 3. Any implementation
difference flips that; nothing distinguishes a correct port from an incorrect one there. Where the
reference is decided, we agree. **Gate the cosine, and gate token-exactness only where the margin
exceeds the numerical delta** — the same objection this doc raises against the prior port's
bit-exact gate (§4), now applied to our own numbers.

### The two bugs, both in the conditioning, both silent

The signature that found them: cos rising monotonically with depth step (0.28 at step 0 → 0.92 at
step 11). Step 0's only input is the temporal output; later steps are increasingly pinned by the
forced correct tokens. That points upstream of the depth loop every time.

1. **`MultiChannelEmbedding` is one table with per-channel vocabularies laid end to end.** Channel
   *c*'s token *t* is row `offset[c] + t`, `offsets = cumsum([0] + vocabs[:-1])`
   (`magenta_rt/mlx/transformer.py:~205-215`). Indexing with the raw token reads the wrong row for
   every channel after the first. Layout from `input_configs`: mulan 12×1031 → `[12372,768]`;
   pianoroll 128×11, drums 1×9, cfg 2×47, cfg_drums 1×15 → 1526 rows, and the table is `[1536,256]`
   because `round_num_embeddings_to_multiple_of_128=True`. `num_reserved_embeddings` defaults to 0
   here, so the "reserved tokens are not offset" branch does **not** apply — do not add it.
2. **The conditioning embedder is `sl.Parallel` with `combination=CombinationMode.MEAN` over two
   branches, and the MusicCoCa branch SUMS its RVQ levels** (`model.py:49-100`, `268-307`):
   - mulan branch: `x + arange(12)*1031` → `Embedding(12*1031, 768)` → **`mx.sum(axis=-2)`** →
     `Dense(768→256, no bias)`. It is an RVQ *dequantizer*: residual levels are summed. Averaging
     them costs a factor of 12 on that branch.
   - regular branch: `MultiChannelEmbedding(reduction_fn=mean)` over its 132 channels.
   - the two branch outputs are then **averaged**. A single mean over all 144 channels weights
     MusicCoCa 12/144 = 8.3% instead of 50% — i.e. dilutes the style conditioning six-fold.

   Together these took the conditioning source from cos 0.45 → 0.56 → **0.99935**, and the frame
   from token-exact 0/12 → 2/12 → **12/12**.

Verified correct along the way and not worth re-checking: the frame-token path (embedding ×√1024,
fp32 mean over the 12 codebooks) at cos 0.999952; `primer_hybrid` = `y + post_norm(branch(pre_norm(y)))`
per `sl.Residual.Config([pre, branch, post])` (`transformer.py:504-517`); the shipped initial state
has all 41 ring-mask slots **False** and all write pointers 0, so a fresh frame really does attend to
nothing but the sink and the current step.

### The SpectroStream decoder — gated at cos 1.000000000

`_mrt2/probe/mrt2_codec_torch.py`, 35.7M conv params (matching the analytic 35,684,546) + a 3.1M
12-level codebook. Scope stops before the inverse STFT, per §3. Reference captured stage by stage
with `gate_codec_capture.py` (the decoder is a 7-layer `sl.Serial`, so each top-level layer can be
applied in turn and its intermediate saved — that turns a guess into a bisect):

```
[0] ExpandDims       (1,25,1,256)     [4] Residual         (1,50,5,1024)   decoder_0
[1] Residual         (1,25,1,2560)    [5] ParallelChannels (1,100,480,4)   decoder_1..6 + out, x2 groups
[2] Reshape          (1,25,5,512)     [6] Lookahead        (1,96,480,4)
[3] Residual         (1,25,5,512)
```

Final: **cos 1.000000000, rel-L2 1.7e-6, max|Δ| 2.2e-2** on `(1,96,480,4)`. RVQ dequantize alone is
cos 1.000000 — the levels are SUMMED, as in the conditioning encoder.

Five things that all produce correct shapes and plausible output while being wrong:

1. **Transpose-conv kernels must be flipped on both spatial axes.** JAX's `conv_transpose` flips the
   kernel (true mathematical transposition); MLX's and torch's do not. The reference loader
   compensates (`spectrostream/load_weights.py:103-113`,
   `kernel = kernel[::-1, ::-1, :, :]`). Without the flip the whole stack runs and lands at
   **cos 0.53** — the single most expensive hour of this port.
2. **Kernel axes are (time, freq), not (freq, time).** `kernel_size[0]` is time and `[1]` is freq
   (`modeling.py:45` derives the freq pad from `kernel_size[1]`; `resample_kernel_size =
   (max(3, 2·stride_t), max(3, 2·stride_f))` names `conv2dtranspose_4x3` = t-stride 2, f-stride 1).
3. **`decoder_0` does not upsample in frequency.** The sequence is 1,2,2,2,3,2,2 (product 96:
   5 bins → 480). Using 2 there gives 960 bins and every downstream shape still checks out.
4. **The reshape is (freq outer, channels inner)** — `sl.Reshape([input_bins, output_channels])` on a
   `[B,T,F,C]` tensor. Splitting the 2560 axis the other way transposes the spectrogram silently.
5. **Layer mode ≠ step mode for the transpose conv.** In layer mode it is conv → trim → bias, and the
   causal time trim is `(0, ek − stride)` — cut from the **right**
   (`convolution.py:598-619`). The streaming step path instead skips the trim, overlap-adds, and
   applies bias *after* the add; copying that into a batched decode is wrong.

Plus two ordinary ones worth stating: the input layer is
`Residual([conv1x1_first], shortcut=[conv1x1_b1, ELU, conv1x1_b2])` — there is an activation
**between** b1 and b2; and `input_layers_residual_unit` runs `conv2d_3x3_a` **before** `conv2d_3x3`.
Finally `Lookahead(4)` (decoder_lookahead=1 × time stride 4) drops the **first** four output frames,
so T code frames yield 4·(T−1) spectrogram frames — 25 in, 96 out, not 100.

### What is still open

- ~~cos sits just under 0.999 on six steps~~ **RESOLVED: it was the reference's bf16 weights.**
  The trap is that `compute_dtype`/`param_dtype` do not control them —
  `magenta_rt/mlx/load_weights.py:516-517` calls `convert_to_bf16(mrt_sampler.depthformer)`
  **unconditionally** at the end of loading, so a "float32 reference" built through the config is
  still a bf16 one. Neutralise that function too (`load_weights.convert_to_bf16 = lambda m: None`)
  and cos moves 0.99904 → 0.99957 mean, every step over 0.999. Note the correct fix was to raise the
  REFERENCE to fp32, not to lower our port to bf16 — matching the dtype does not match the
  accumulation order, and casting our model to bf16 moved the mean by 2e-6.
- **Frame 0 does not exercise the ring buffer at all** (every history slot is masked off). Any bug
  in the ring/shift bookkeeping — scatter at `t mod 41` for self-attention, concat-and-slice for
  cross-attention — is invisible in this gate. Extending it to N frames with a populated window is
  the next step and the larger remaining risk.
- Both halves are now re-authored and gated; what is NOT done is exporting *these* (the Core AI and
  Core ML artifacts measured in §1 and §7 are still the random-weight stub), the host-side iSTFT in
  NumPy, and an end-to-end audio comparison.

### Probing the reference, for whoever continues

Hooking by module path wastes time: the tree is `MagentaRT2Sampler` → dict-like children, and
`.decoder`, `Encoder.layer` and `EncoderDecoder.encode` are either absent or off the streaming path.
Hook module-level functions in `depthformer.py` instead — `_mean_in_f32` fires exactly once per frame
on the frame-token path. `sl.Serial.step` also works but fires **158 times per frame** because every
`CheckpointName` wrapper is a Serial; identify the one you want by its children
(`Serial(n=4: Parallel, Identity, Identity, LayerNormalization)` is the conditioning encoder,
144 → 256). Identifying a call by position alone produced a wrong bisect here. And mlx bf16 arrays do
not cross the numpy buffer protocol — `.astype(mx.float32)` first, or you get a PEP 3118 error.

## 9. Where this stands, and the next experiment

The artifacts are at `~/code/coreai/_mrt2/` (reference env with a working MLX oracle, the Core ML
stubs, the palettizer, the parity harness, the device bench app) and `~/code/coreai/MRT2_KICKOFF.md`.
The one action worth taking without reopening the port is offering §2 upstream: it is a 3.5×
speedup on the half of the pipeline the prior port owns, their repo is Apache-2.0 and active, and
they have the validated end-to-end pipeline we do not.
