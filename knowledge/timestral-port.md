# Timestral-209M: a patch transformer for time series on Core AI, GPU and Neural Engine

> 2026-10-22. mistralai/Timestral-209M (revision `9874d25`, Modified MIT License with a revenue condition; reference
> code `mistralai/timestral` @ `a534bdb`, Apache-2.0) forecasts 21 quantiles from a series cut into 32-step patches:
> d 768, 12 heads, 24 blocks (time attention, and every fourth block a variate attention), a monotone quantile head.
> The port is one univariate series, no covariates, one decode block (≤ 1,024 steps): the graph is the transformer
> over `[context patches ; forecast queries]` plus the head, the host does the released `forecast()`'s scaling,
> refine frame, cap and symmetric averaging in NumPy. Card: [`models/timestral/README.md`](../models/timestral/README.md).
> Scripts: [`conversion/timestral/`](../conversion/timestral/). Records outside the repo are marked "lane" and live
> under `$ZOO_WORK_ROOT/_timestral/`. Mac M4 Max, macOS 27.0 26A428, Xcode 27.0 RC, coreai-torch 0.4.1,
> coreai-core 1.0.0b2; iPhone 18 Pro, iOS 27.2 24B5099f.

Each section below is one finding, with the measurement it rests on.

## A finite additive mask is not a mask when the scores are large

TimesFM's port in this zoo passes an additive mask with a finite fill (−10⁴), which is harmless there. Timestral's
attention scores |q·k/8| reach 242,875 in block 1 on the fixture (125,775 in block 22, 80,771 in block 4), so
−10⁴ added to a masked score leaves it far above the allowed ones: the re-authored graph missed the released module
by max |Δ| 0.929 against a bar of 0.129, on every window, not just the padded ones (lane
`gates/w1_additive_mask_1e4/l1_torch_L64_h32.json`). The fix is a 1 / 0 mask as an input and
`where(mask, scores, fill)` in the graph (fill −10³⁰ in fp32, −65,504 in fp16): max |Δ| 1.44 × 10⁻⁴.

Measure the scores before choosing the mask form: `gate_core.py` records the per-block absmax of the allowed and the
masked scores. `coreai-torch`'s SDPA composite turns a bool mask into `−1e4 × ~mask` in its decomposition
(`_utils.py`), yet on the Mac GPU the bool-mask SDPA graph returned the float-mask (−10³⁰) graph's numbers in every
digit (lane `r3/gates/e2e_L64_h32_r2_fp32-qkv4-sdpab-gpu.json`), so the decomposition is not what runs there. Other
units were not checked; the shipped graphs use the explicit `where`.

## The fp16 overflow was in the attention scores

Per-layer absmax in fp32 over every fixture token (pads included): residual ≤ 9,534, Linear outputs ≤ 6,207 — inside
fp16 — RMSNorm's sum of squares up to 1.65 × 10⁸ (`out_norm`) and the attention scores up to 2.4 × 10⁵ — outside.
The sum of squares is an authoring problem with exact fixes (LayerNorm over `[x, −x]`, or x / 2⁷ with eps / 4⁷); the
scores are not. fp16 compute on the GPU saturates at ±65,504 (MASE 6,573 times the oracle's); fp16 compute with the
scores and the softmax in fp32 (`fp16-attn32`) passes L3 (worst group ×1.0006 on the Mac GPU, ×1.0015 on the
iPhone 18 Pro's); fp16 weights with fp32 compute passes too (×1.0004). Lane `gates/l1_torch_L64_h32.json` `absmax`,
`gates/e2e_main_fp16-layernorm_gpu.json`, `gates/e2e_main_fp16-attn32_gpu.json`.

## fp16 compute is 2.4 times faster than fp32 on the iPhone 18 Pro's GPU, not on the M4 Max's

The same graph, R = 2, warm medians. iPhone 18 Pro (Swift app): in one launch the `h19p` compiles took 15.19 ms
(fp16-attn32) and 36.52 ms (fp32), 2.4 times; the `.aimodel` forms took 14.76 ms (fp16-attn32, a launch a minute
earlier), 36.29 ms (fp32) and 44.20 ms (fp16 weights + fp32 compute). M4 Max: 11.54 / 11.29 / 13.44 ms (Swift tool);
11.94 / 11.69 / 13.82 ms (Python runtime, AOT). At 8,192 steps the M4 Max's GPU gains 5 % from fp16 compute (24.73 vs
25.93 ms, Swift, one launch), which made fp16-attn32 its shipped long-context form. fp16 storage with fp32 compute is
the slowest form on both devices: it halves the download and buys no speed. Lane `device/pull/d-ms-s1-a/run.json`,
`d-ms-s1-b/run.json`, `device/mac/out/m-timing-L64h32r2/run.json`, `timing/timing_run1.json`,
`stage/r6/swift/out/r6-ms-long/run.json`.

## On the Neural Engine, the near miss was the sigmoid and SiLU; written through tanh the graph passes

The initial fp16 graph on the Neural Engine (one region, R = 1) passed 21 of 22 fixture groups and missed one by
0.09 % (MASE ×1.0109 against the line's 1.010; overall ×1.0014), with the same numbers, digit for digit, on the M4 Max
and the iPhone 18 Pro.
What it was not: the attention scores (an exact key shift that brought blocks 4, 19 and 22 inside fp16 made it
×1.0116), the RMSNorm form (x / 2⁷ instead of LayerNorm: ×1.0109), the Linear kernel (1×1 conv: identical output).
A bisect by contiguous cuts — blocks [0, k) on the Neural Engine in fp16, the rest on the GPU in fp32, the host
passing the residual — put the miss in the patch embedding: with k = 0 (the embedding alone) the group was already
×1.0101, and adding blocks moved it little. The embedding is `Linear → SiLU → Linear` plus a residual Linear; alone on
the Neural Engine it was 3.4 % off. With every sigmoid gate and SiLU written as `0.5 · tanh(x / 2) + 0.5` (exact) the
embedding was 0.19 % off and the whole graph passed L3 as one region (worst group ×1.00517). Lane `r3/gates/
e2e_split_k*.json`, `emb_sig.json`, `emb_tanh.json`, `e2e_L64_h32_r1_fp16p7-qkv4-tanh-conv-ane.json`.

The graph's fp16 scores stay finite on the Neural Engine where the same graph on the GPU saturates (why is not
isolated): the Neural Engine form is right on the Neural Engine only. Judge where it ran by its output against the same
bundle on the GPU (3,043,858 of 3,053,568 elements differ, max |Δ| 65,529), not by a region's name.

## On the iPhone 18 Pro the Neural Engine form has the lowest latency in every bucket; ship its `h19p` compile

The tanh form's `h19p` Neural Engine compiles (one region each) loaded on the iPhone 18 Pro in 0.81–1.25 s and
returned the M4 Max Neural Engine's numbers digit for digit in all three buckets, R = 1 and R = 2. Against the phone's
best GPU form in the same launch: 9.44 vs 14.58 ms at 2,048 + 1,024 steps (fp16-attn32, −35 %), 8.42 vs 13.90 ms at
2,048 + 128 (−39 %), 33.86 vs 40.13 ms at 8,192 + 1,024 (−16 %). On the Mac the 8,192-step bucket stays on the GPU
(26.23 vs 35.92 ms). Rows: on the phone one R = 2 call is faster at 96 and 68 tokens and two R = 1 calls at 288; on
the Mac two calls are faster at 96 and 288 tokens and one call at 68. The same `.aimodel` specialized on the phone
for the Neural Engine is within 1 % of the compile once warm (7 % faster for R = 2 at 288 tokens), but the six tanh
`.aimodel` forms took 21.8–59.1 s to load (30–59 s for the three shipped ones) in a launch that held several bundles
(1.18–1.50 s alone; the cause is not isolated), against 0.87–1.09 s for the compiles in the same launches. Lane `device/pull/r4-ms-{main,short,long}/run.json`, `device/scored_r4/vs_mac.json`.

## The Mac ships its own forms, measured through Swift

The zoo ships `.aimodel` files and the Mac runs them through the Swift runtime, so the Mac's forms were chosen there
(one launch per bucket, the forms alternated): the Neural Engine form R = 1 × 2 at 9.46 ms beats R = 2 at 9.96 ms (5 %)
at 2,048 + 1,024 steps, R = 2 at 8.24 ms leads at 2,048 + 128, and the GPU's fp16-attn32 at 24.73 ms leads at 8,192 +
1,024 (the Neural Engine form 35.31). Within a bucket the `.aimodel` and its `h16c` compile are within 0.4 %. A
`.aimodel` loaded for the Neural Engine on the Mac returned its compile's outputs bit for bit (three forms), so it ran
on the Neural Engine. The initial load of a Neural Engine compile on the Mac took 36–53 s; later loads, with the app's
cache entry removed each time, took 0.14–0.54 s for the compile and 4.7–9.3 s for the `.aimodel` (what makes the initial
one slow was not isolated). Lane `stage/r6/ms_swift.md`, `stage/r6/gates/score_r6-*-jit-ane.json`.

## A rebuilt export is a new file: check it by its outputs

Exporting the same form twice with the same code gives a different `main.hash`: the `main.mlirb` differs by a few
bytes in length and in some thousand bytes (fp32 L256: 809,314,684 vs 809,314,685 bytes, 6,294 bytes differ). The
compiles differ too (`resources.bin` has the same size and other contents) while their `main.hash` stays the same, the
cache-name trap below. What stays is the computation: each of 10 forms rebuilt from `conversion/timestral/` returned
the measured bundle's outputs bit for bit on every fixture window of its bucket on the Mac (Neural Engine and GPU),
and a deliberately different compile (sigmoid instead of tanh) did not (2,954,725 of 3,053,568 elements). So a
rebuild is checked by running both on the fixture, not by hashing files. Lane `stage/r6/hashcmp.txt`, `repro.txt`,
`aotcmp.txt`, `gates/score_r6-*.json`.

## R = 2 compiles for the Neural Engine only with 1×1-conv projections

Every fp16 graph with the two rows of symmetric averaging in one call aborted the Neural Engine compiler:
`Assertion failed: (false), function SetDimensionOrderHint, file ZinIrTensor.cpp, line 986.` (exit −6, macOS `h16c`
and iOS `h19p`). Cut down to one block, only the attention sublayer fails; the MLP, the variate sublayer, the patch
embedding and the head compile at R = 2. Heads as rank 5, rank 4 (Q / K / V as three Linears) or rank 3 (rows folded
into heads) all abort. With every Linear as a 1×1 `Conv2d` over (R, C, 1, S) the R = 2 graph compiles to one region,
and conv and Linear give bit-identical outputs on the Neural Engine (R = 1 and R = 2 too). Lane `r3/aot/
*_r2_*-h16c-ane.aot.json`, `coreai-build.log` beside each.

## An AOT compile's cache entry name ignores the weights

The runtime caches a compiled bundle under its `main.hash`, and for an AOT compile that hash covers the graph, not the
weights. Two bundles of the same shape with different weights — single blocks of the same kind, or the 8-bit and the
6-bit k-means compressions of one graph — got the same `main.hash` (also for `h19p`); loading the second in the same
cache lifetime ran the earlier bundle's compiled graph and weights, with finite, plausible output. A control arm caught it: the
same per-block chain on the GPU in fp32, which had to pass, missed by 23.26. Before a run that loads several bundles,
list their compiled `main.hash` files and split duplicates into separate processes with the entry removed in
between; cut by contiguous ranges (different shapes) instead of single blocks. Lane
`r3/gates/blockerr_perblock_r1.json`, `r3/gates/l1_chain_fp32_gpu.json`, `r3/aot/*.aot.json` `hash_files`.

## Symmetric averaging in one R = 2 call

The released forecaster runs the series and its negation and averages them. With one variate per series the variate
attention has one key, so its softmax is 1 and the sublayer is `wO(sigmoid(gate(x)) ⊙ Wv(x))`; the two series in one
call are two groups whose variate mask is the diagonal, so the same holds at R = 2. The mirror's graph input is the
series' input with the value half negated (the float64 statistics run once). One R = 2 call vs two R = 1 calls: on
the iPhone 18 Pro's GPU 44.89 vs 54.02 ms (fp16w32), on the M4 Max's GPU 4.8–5.1 ms faster; on the Neural Engine R = 1 × 2
is the faster form at 2,048 + 1,024 steps (9.914 vs 10.195 ms) and R = 2 at 2,048 + 128 (8.442 vs 9.300 ms). Lane
`gates/e2e_main_fp16w32_gpu_r1split.json`, `device/pull/d-ms-s2-rows/run.json`, `r3/ms/ms_A4.json`, `r3/ms/ms_C3.json`.

## One decode block, cut at the bucket's horizon

A forecast token reads the valid context and the forecast tokens up to itself, never one after it, and the refine
frame of horizon patch j uses only patches before j. So the leading ⌈H / 32⌉ patches of a 32-patch graph are a
smaller-horizon graph's (up to the order of float sums), and one graph per context bucket serves every horizon up to
1,024 steps; L2 passed on horizons 24 to 720 from the h = 32 graph. A 4-patch graph for horizons ≤ 128 saves 11 % on
the Mac GPU (10.27 vs 11.59 ms, fp32) and 23 % on the iPhone's fp32 GPU graph, and on the Mac's Neural Engine it is the
quickest form for that workload. Lane `timing/timing_run2.json`, `device/pull/d-ms-s2-bucket/run.json`.

## int8: the cliff is the patch embedding and the head

int8 per block of 32 (symmetric, weights only) on all 144 Linears misses L3 (×1.018); the same int8 model in torch fp32
misses by the same amount (L1 0.6745 vs 0.6744 on the GPU), so it is the quantization, not a kernel. Leaving the
patch embedding and the head (6 Linears) unquantized passes (×1.0020 with fp32 compute, ×1.0015 with fp16-attn32) at a
250 / 222 MB `.aimodel` instead of 809 / 405 MB. The Mac AOT expands the int8 weights to dense ones (`resources.bin`
unchanged) and runs at the unquantized speed; on the iPhone 18 Pro the 222 MB `.aimodel` passes too (×1.0032) but runs
12 % slower than the fp16-attn32 `.aimodel` (16.46 vs 14.69 ms), and its `h19p` compile folds to dense fp16 (405 MB)
at the fp16 speed: a download-size lever only. k-means 8-bit per tensor misses (×1.055) and keeps a LUT that runs 1.2
times slower; 6-bit per grouped channel misses far (×1.86). Lane `r3/gates/e2e_L64_h32_r2_*-int8b32*-gpu.json`,
`l1_t_int8.json`, `r3/ms/ms_A4.json`, `device/pull/r4-ms-quant/run.json`.

## Swift, Python, Mac and iPhone: the same numbers where they should be

The Swift tool on the Mac returned the Python engine's numbers window for window on 20 runs (every window's L1 and every
group's MASE and CRPS equal), and the JIT `.aimodel` and the AOT compile gave identical outputs on every form tried
(7 on the Mac; on the iPhone 18 Pro every GPU and Neural Engine form). A Swift port of the host (CoreAIKit's
`TimestralHost`) returned the Python host's forecasts bit for bit: graph inputs, graph outputs and all 21 quantiles of
265 forecasts on five bundles, which took NumPy's own float32 cos / sin kernel and its pairwise summation order in
Swift. On the Neural Engine the iPhone 18 Pro returned the M4 Max's numbers digit for digit (every window's L1, every
group's MASE and CRPS; the r1 near-miss form and all six tanh forms), and every `h19p` Neural Engine compile loaded on
the phone. GPU outputs differ between the two devices in most elements (fp32 max |Δ| 9.6 × 10⁻⁵, fp16-attn32 0.086)
with the same verdicts. Lane `device/scored/summary.json`, `device/scored_r4/summary.json`,
`sample/parity/score_swift.json`.

## The work is minutes; plan around exclusive time

On this graph an export takes 3–5 s, an AOT compile 0.5–4 s, an end-to-end gate over 71 windows 1–3 s after the load
(a Neural Engine compile's initial load adds up to 18 s; gates sharing the Neural Engine with other jobs took minutes),
and a timing run of 40 rounds a few seconds per form. A whole round of exports, compiles, gates and timings fits in a
few minutes of a quiet machine; here the wait for one (45 to 77 minutes per window) set the pace. Pack each window
with a resumable job list that skips finished results, and run the device work (iPhone) beside it.
