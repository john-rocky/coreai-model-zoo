# Nemotron-3-Diarization: an 8-speaker streaming Sortformer ported from transformers

Lessons from porting [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization)
(OpenMDW-1.1, 99.2M parameters) to Core AI: streaming speaker diarization for up to 8 speakers at 10 ms.
The export boundary follows the 4-speaker port ([`sortformer-speaker-diarization.md`](sortformer-speaker-diarization.md)):
the per-chunk network is the graph, and the stateful streaming algorithm lives in the host. The reference
changed, though. Here it is transformers `Nemotron3DiarizationForAudioFrameClassification` in fp32, not
NeMo, and the cache rules come from it. Code and gates: [`conversion/nemotron3_diar`](../conversion/nemotron3_diar);
card: [`models/nemotron-3-diarization`](../models/nemotron-3-diarization/README.md).

## 1. Take the cache rules from transformers, not from the 4-speaker port

The host ports `Nemotron3DiarizationSpeakerCache.update` / `_compress` line by line. Read against the
4-speaker NeMo port, three parts differ:

- **Silence rows.** One per speaker, holding the learned `silence_embeds` parameter. A `-inf` pick and the
  `+inf` silence pad both land on it, and indices wrap by N + 1 in the speaker-major flat scores. The
  4-speaker port pads three rows per speaker and fills those picks with a running mean of silent frames.
- **A FIFO.** 264 rows, at least 222 of them popped into the cache at a time (offline: 40 and 300). The
  4-speaker configuration had none.
- **The per-speaker budget.** 264 / 8 − 1 = 32 rows, so the same rates give strong boost 24, weak boost 48
  and minimum positives 16 (4-speaker: 44 → 33 / 66 / 22).

One rule is the same in both and easy to get wrong: until the first compression, each step re-estimates
the cache's probabilities; after it, the stored ones are used.

The graph boundary moved as well. The cache stores the *input* rows of the encoder: 8 stacked mel frames
through the 1024→512 projection. The host computes those itself, so the graph returns logits only. The
4-speaker graph had to return its embeddings as a second output.

## 2. One graph for three streaming modes: left-pack the rows, pass `valid`

The input is `packed [1, T, 512]` with the real rows at [0, L) and `valid [1, T]`. The attention key mask
is `(1 − valid) · (−1e4)`. Because the rows start at 0, each real row sits at the position transformers
gives it (positions restart every chunk there). T = 264 cache + 264 FIFO + 9 chunk + 4 look-ahead = 541
covers low latency, and the two lower-latency modes need fewer rows, so one bundle serves all three. Offline
needs its own bundle: 264 + 40 + 340 + 40 = 684.

## 3. Zero the padding rows before the Conv1d

A key mask keeps padding out of attention, but the 8× sub-pixel upsampler is a Conv1d with kernel 3 and
padding 1. At row L−1 it reads row L, where an unpadded sequence has the conv's implicit zero. The graph
multiplies the projection's output by `valid`. The gate fills the padding with random values to prove it:
without the multiply, max |Δlogit| is 0.865–8.86 on padded chunks. At L = 541, with no padding, the
control passes, as it should.

transformers has the same hazard offline. The feature extractor's extra centered frame becomes an embed
row that is masked as a key but not zeroed before the conv, so the last 8 output frames carry its garbage.
The host drops that frame instead of masking it. The gates skip the last 16 frames, as transformers'
integration test does.

## 4. Bake RoPE in as constants

RoPE covers all 64 head dims (theta 10000). With left-packed rows the positions are always 0..T−1, so the
cos / sin tables are registered buffers and the graph does no rotary arithmetic. The negative control
with RoPE removed misses by max |Δlogit| 10.1–40.9.

## 5. Keep float32 I/O on the fp16 bundle

A thin wrapper casts the inputs to fp16 and the logits back to float32 inside the graph. The host and
every gate then load the fp16 or the fp32 bundle without a code change. Do not name the wrapped module
`graph`. torch.export's unlift resolves `graph.*` against `GraphModule.graph` and fails with
`references nonexistent attribute model of graph`.

## 6. Compression can hinge on 2 ulp: score in float64

transformers computes the compression scores in float32. In 3 of the 16 compressions of the two test clips,
its own top-k choice is 2 ulp (1.19e-7) from the boundary. A NumPy float32 host broke one of those ties
the other way (97.6 s, low latency, the second compression), and the cache diverged from there on.
The host now scores in float64 from the float32 probabilities. Fed transformers' own scores, the host's
selection matches all 16 compressions; with its float64 scores it matches 15. The 16th (very low latency,
97.6 s) is a case where transformers' float32 order and the true order disagree.

Two consequences:

- Do not gate on step-exact cache equality. Even the fp32 eager graph leaves the reference's trajectory in
  very low latency (99.9167 %). Gate the closed-loop agreement, plus teacher-forced compressions.
- A Swift port has to match the summation order. NumPy pools the 8 frames left to right and sums the
  8 speakers pairwise, ((0+1)+(2+3))+((4+5)+(6+7)). libm's `expf`, `log` and `logf` equal NumPy's.
  With both orders copied, 35 teacher-forced cache updates are bit-identical.

## 7. fp16 on the Neural Engine: the error is not LayerNorm overflow

The final LayerNorm sees |x| up to 1,216. In 52 of the 64 LayerNorms, a row's sum of squared deviations
exceeds 65,504, the fp16 maximum. A rewrite that scales x by 1/64 before the statistics keeps every
square small. The input LayerNorm stays as it was: its eps / 64² underflows in fp16, and a zero row would
become 0 / 0.

It changed little. The per-chunk Neural Engine error went from max |Δlogit| 0.194 to 0.163, and the
agreement from 99.9970 % to 99.9965 %. The 97.6 s closed loop stayed below the 99.9 % bar (99.7746 % →
99.8271 %), and neither version produced NaN or inf. The GPU runs the same fp16 graph at 0.076 and passes
without the rewrite. The cause of the Neural Engine error was not found. The iPhone 17 Pro's Neural Engine
(h18p compile, iOS 27.0) gives the same result: 99.7758 % on the 97.6 s clip and 100 % on 21.5 s, at 31.5 ms
per chunk against the GPU's 30.1 ms, so the GPU bundle ships for the iPhone too.

## 8. The reference's own fp32 STFT sets the mel bar

transformers' `torch.stft` in fp32 lands 1.82e-4 from a float64 STFT, on near-silent bins (log-mel ≈ −15).
NumPy's fp32 rFFT on the same windowed frames is 1.04e-6 away. A host mel cannot beat the reference's
rounding, so the bar became: max 5e-4, fewer than 0.1 % of cells above 1e-4, and logits within 1e-4 of
transformers' downstream. Measured: 1.81e-4, 28 of 1,249,280 cells, and logits within 5.15e-5.

The window matters at this scale. NumPy's Hann and `torch.hann_window(400, periodic=False)` differ by
1 ulp in 9 of 400 entries. The port ships torch's bits as a constant (`hann_window_400.f32le`).

## 9. In Swift, sum the DFT in Double

The Swift mel first ran the DFT as a float32 matmul of the windowed frames against cos / sin tables. It
drifted 1.18e-4 from NumPy's rFFT, and on the 97.6 s clip that flipped compression decisions: 1–45
elements differed from the Python host on the same GPU bundle. The fix sums in Double (`vDSP_mmulD`),
reduces each angle exactly as (k · j) mod 512, and rounds re / im to float32 the way NumPy's complex64 rFFT
does. The mel became bit-identical to NumPy, and the Swift loop's logits bit-identical to the Python loop's.
It costs 12 ms on 97.6 s of audio (0.015 → 0.027 s).

## 10. Count ANE regions by id after an AOT compile

`xcrun coreai-build compile --preferred-compute neural-engine` exits 0 even when the ANE compiler rejects
the graph ([`ane-quality-gate.md`](ane-quality-gate.md)). Count the distinct `ANE_region_<i>_<j>` ids in the
compiled bundle. A plain file count
triples: one region shows up as a `.bc` directory, its `.bc.weights` and a `.mlir.bc`. The MPSGraph
manifest's `mps.fullyPlacedOnANE` and `mps.noGPUActivity` flags confirm full placement. This graph
compiles to one region, fully placed, for both iOS h18p and macOS h16c.
