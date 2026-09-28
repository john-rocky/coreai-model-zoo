# decider-2b-vision: an image-input decision model on Core AI

> 2026-09-28 / 29. Mapika/decider-2b-vision (revision `863e290`, Apache-2.0) answers lettered questions
> about an image by the letter probabilities at each answer slot of one pass. The port is a fixed-grid
> tower from shipped code plus one new decoder module (ids in, image rows as a static input, M-RoPE in
> the graph). Card: [`models/decider-2b-vision/README.md`](../models/decider-2b-vision/README.md).
> Transcripts below are in `models/decider-2b-vision/`; records outside the repo are marked "lane" and
> live under `$ZOO_WORK_ROOT/_decider2bv/`.

## What was reused and what is new

| part | source | change |
|---|---|---|
| vision tower | overlay `qwen3_5_vision.Qwen3_5VisionEncoder` (the Qwen3.8-27B vision path), `conversion/export_qwen38vl_pipelined.py --skip-decoder` | grid 8×8 / 14×14 by flag; new `--vision-dtype fp16w32` |
| decoder | overlay `Qwen3_5StatefulForCausalLM` + `_mrope_freq_masks` + `forward_stateful_core(cos_sin=…)` + `_gated_delta_step_unroll` | new subclass `conversion/decider_vision/qwen3_5_vl_pipelined.py` |
| host | none fit: the zoo's NumPy `resize_antialias` is 14.6 / 17.3 levels off Pillow | new `conversion/decider_vision/host.py` |
| readout gates | the decider-0.8b / letter gates' method (AOT `.aimodelc`, fresh states, softmax at the slots) | new scripts in `conversion/decider_vision/` |
| Swift host | the low-level runtime pattern of `knowledge/swift-runtime.md` | new package `apps/DeciderVision` |

The overlay already had a Qwen3.5 VL decoder, `Qwen3_5VLStatefulEmbeds`, but it takes host-built
embeddings and three host-fed M-RoPE planes, so the 1 GB embedding table ships beside the graph. The
new module is the Qwen3-VL engine contract on the Qwen3.5 hybrid instead.

## The decoder contract, and why `image_rc`

`input_ids` carry the image as ids V + k (V = 248,320). In the graph an id ≥ V reads row id − V of
`image_embeds [256, 2048]`; any other id reads the embedding table. Position planes come from the ids
and the ramp: an image token with slot k at position p has (t, h, w) = (p − k, p − k + row, p − k + col),
and text after the image gets p − `rope_shift_amount` on all planes. A text-only row sends zero image
rows and `rope_shift_start` = 2³⁰, and the graph then equals the overlay's plain text decoder bit for
bit (467/467 steps, `gate-decider-2b-vision-torch-parity.json` P2).

The torch-parity transcript records the sha256 of the module file it ran (`7f7fd569…`), the file
the bundles were exported from.

`image_rc [256, 2]` holds (row, col) per image token. Qwen3-VL's module computes `slot // W` with a
baked W; the table instead lets one bundle take the 8×8 and 14×14 towers and any rectangle up to 256
tokens. The fixture's `native` arm (the processor's own grids, some non-square) passed through the
same decoder: 35/35 runs.

## Gate positions by plane equality, not by |Δp|

Position mistakes barely move this fixture's probabilities. In fp32 torch on r14 / r16 at `g256`:
row/column swapped in `image_rc` moves p by ≤ 0.0004, 1-D positions by ≤ 0.0012, planes collapsed to
the text position by ≤ 0.0006 — no argmax changes (`gate-decider-2b-vision-torch-parity.json` P4). A
|Δp| bar of 0.02 would pass wrong positions here. The author's contract is full M-RoPE, so the decoder
derives all three planes and the gate checks them for equality.

The equality check: the decoder module's planes (fp32 torch, the code the graph is exported from)
equal the planes the author's rotary received (captured by a hook in the oracle) on
111/111 runs (P1), and `host.build_ids` reproduces the
ids, slots and planes on 111/111 runs with two tokenizers (`test_host.py`). Its negative controls go
red: `rope_shift_amount` + 1 on 105/105 image runs, H and W swapped on 16/16 non-square runs (lane
`logs/r2_test_host.log`).

## The fp16 tower fails; fp16 storage with fp32 math passes

Against the author's fp32 tower on the 29 fixture images (`gate-decider-2b-vision-tower.json`):

| tower on the Mac GPU (AOT h16c) | worst image cosine | worst row cosine | rows below 0.999 | encode, contended |
|---|---:|---:|---:|---:|
| fp16, `g256` | 0.99635 | 0.849 | 97 / 1,856 | 17.7 ms |
| fp16, `g448` | 0.99409 | 0.528 | 405 / 5,684 | 49.8 ms |
| fp16w32, `g256` | 0.99999992 | 0.9999975 | 0 | 22.9 ms |
| fp16w32, `g448` | 0.99999997 | 0.9999957 | 0 | 62.8 ms |

The failing rows have about half the median reference norm (3.19 vs 6.09 at `g256`, 3.22 vs 5.50 at
`g448`; `g3_isolation_pooled`), and flat image regions are not the cause (40 of 97 failing rows at
`g256` are flat tokens, as are 730 of 1,856 rows overall). CPU torch in fp16 misses as well (55 of 58
image-grid pairs below row cosine 0.999), so the cause is the graph's fp16 arithmetic, not the GPU or
the compiler; the residual stream's absmax peaks at block 23 on every image, 4,698–6,231 (lane
`logs/r2_isolate_fp16.json`). Rounding the weights and the baked positional constants to fp16 while
computing in fp32 still misses on 7 of 58 (worst row 0.978, same record).

The fp32 tower passes, and so does fp16w32: parameters stored fp16 and read through a `.float()`
cast, the math and the baked positional buffers in fp32. It is the Fun-ASR encoder's form
(`conversion/funasr_nano/export_encoder.py`), copied into the Qwen3.8 exporter as `--vision-dtype
fp16w32`. It keeps the fp16 size (660 MB vs 659 MB at `g256`; fp32 is 1.32 GB).

The fp16 tower in front of the int8hu decoder raised the end-to-end max |Δp| from 0.0214 to 0.0285
(`gate-decider-2b-vision-e2e.json`).

## Resize in Pillow's pass order

The processor resizes with Pillow's BICUBIC (`resample: 3`). The all-NumPy resize earlier VLM ports
used (`_smoke/lfm25vl_preprocess.resize_antialias`, through `_smoke/qwen38vl_preprocess.py`) runs the
vertical pass first in float64. It lands up to 14.6 levels (`g256`) and 17.3 levels (`g448`) away from
Pillow's pixels. Pillow runs the horizontal pass first and rounds the intermediate to uint8. A NumPy
copy of that order, `host.resize_bicubic`, lands 0 / 2 levels away; through its patches the fp16w32
tower keeps image cosine ≥ 0.9999935 and row cosine ≥ 0.99971 on every image
(`gate-decider-2b-vision-tower.json` G1b, G1c, `g4c_fp16w32`).

The Swift host copies that order: its tiles equal `host.resize_bicubic` bit for bit on 58/58 tiles,
and differ from Pillow by at most 2 levels on 2 of 58 (`gate-decider-2b-vision-swift.json` S2). The
two `g448` runs whose tiles differ are the only runs where Swift and Python disagree at all (S3
`vs_python_attribution`). A JPEG decoded by ImageIO is not guaranteed to equal Pillow's libjpeg decode;
the fixture is PNG, where the decoded RGB equals Pillow's byte for byte.

## int8: the body's error, spread over the early layers

| decoder | fixture max \|Δp\| (bar 0.02) | worst run |
|---|---:|---|
| fp16 | 0.0101 | r26 `g448` |
| int8hu: int8 body + int8 head | 0.0206 | r29 `g256` |
| int8lin: int8 body, fp16 head | 0.0217 | r29 `g256` |
| int8mix: layers 0, 2, 5 fp16 | 0.0094 | r26 `g448` |

(`gate-decider-2b-vision-readout-*.json`.) The fp16 head does not help (int8lin), so the error is in the
body's linears. `conversion/decider_vision/int8_bisect_torch.py` evaluates configurations in fp32 torch
with the exporter's own int8 weights read back through the finalized module's dequantization, on 4
fixture runs (r29 `g256`, r26 `g448`, r28 `g256`, r20 `g256`). Lane `logs/r6_bisect.json`:

- all int8: r29 0.0221; layers 0–11 fp16: ≤ 0.0023 on all four; layers 12–23 fp16: r29 0.0202.
- MLP linears alone int8: r29 0.0162; Gated DeltaNet projections alone int8: r26 0.0221, above the
  all-int8 0.0093 — the errors partly cancel; full-attention linears alone int8: ≤ 0.0071.
- one fp16 layer at a time: the best, layer 5, leaves r29 at 0.0174.
- rule (fixed before the search): r29 ≤ 0.010, no run worse than all-int8, at most 4 layers, the smallest
  set. {0, 2, 5} gives r29 0.0097 and was chosen; {0, 5, 8} also passed.

The set was chosen on the fixture. The held-out gate below is what tested it.

**coreai-opt 0.2.1 does not clip.** `symmetric_with_clipping` gives codes in −127..127 on all 186
quantized modules (plain `symmetric`: −128..127), and on `layers.3.mlp.down_proj` every one of its
393,216 blocks keeps its absmax: dequantized block absmax = 127 × fp16(absmax / 127) within 1.1e-4
(codes: lane `scratch/r6/int8_clipping.json`; the block check read the dequantized weights in
`scratch/r6/int8_clipping.pt` against the source safetensors on 2026-09-29). Plain `symmetric` left
r29 at 0.0178 in the bisect, so the scheme is not the lever.

## A held-out gate after choosing on the fixture

The fixture chose the layer set, so the test set had to be one no round had looked at: the 250 photos
of the grid-price table, the first rows of visual7w (150) and vsr (100) in The Cauldron — the two
subsets the author's `decider/vision/data.py` marks `held_out=True`. `heldout_prepare.py` rebuilds them
with the author's filter and `thumbnail((768, 768))` and matches each to the grid-price record; it
needs only the processor, not the model. The images stay in the lane directory; the transcript holds
row indices and numbers only.

Bars were written before any result: ids = the processor's (500/500), pixels within 2 levels
(500/500, all exact), argmax equal on every run with author top-2 margin ≥ 0.02 (near-ties listed
apart), max |Δp| ≤ 0.02, mean of run means ≤ 0.002. Ship rule: int8mix ships if it meets all; no layer
is re-chosen on these data. Result (`gate-decider-2b-vision-heldout.json`): int8mix 497/497 + 3/3
near-ties, max 0.0146, mean 0.00144; fp16 reference 497/497 + 3/3, max 0.0131, mean 0.00061.

Read the subsets apart before quoting the mean: on vsr alone int8mix's mean is 0.00276 / 0.00274
(`g256` / `g448`), fp16's 0.00107 / 0.00105. Red arms checked that the bars can fail: an edited
question and a moved pixel fail bars 1–2, and pairing each photo's `g448` output with the author's
`g256` reference fails bars 3–5 (233/247 argmax, max 0.69).

## Two functions: chunked prefill costs no accuracy, and grows the AOT by 3.8 GB

`prefill` runs 16 tokens per call with the Gated DeltaNet recurrence unrolled in the graph (fp32
inside the call), next to the S=1 `main`. Against the same bundle read one token at a time, the chunk
order changes no argmax (157/157) and moves p by ≤ 0.0058 (int8mix), 0.0019 (fp16) and 0.0032 (int8lin);
full logits are never bit-equal (0/157), as expected for a different summation order
(`chunk_vs_s1` in the readout transcripts). Contended Python gate, int8mix, median row: 0.305 s chunked
vs 1.68 s at S=1 for `g256` (143 tokens), 0.526 vs 3.23 s for `g448`.

The `.aimodel` shares the weights between the two functions (int8lin 2,477,654,906 B with `main`
only, 2,478,952,184 B with both). The AOT compile does not: 6,241,920,985 B vs 10,010,705,679 B for
the same pair, 11,297,479,003 B for fp16_pf16. This port ships no AOT asset.

## Swift specializes the `.aimodel` correctly; the Python gates stayed on AOT

The Swift CLI loaded the ship decoder's `.aimodel` with `SpecializationOptions(preferredComputeUnitKind:
.gpu)` + `expectFrequentReshapes` and passed the fixture: 108/108 argmax, max |Δp| 0.0103 (S4). Cold
specialization 14.35 s, 10.34 s from the cache; the MPSGraph scratch directory stayed at 96 KB. This
port never ran the Python runtime's JIT on the decoder: decider-0.8b measured it returning all-zero
logits on 26A428 ([`decider-0.8b-port.md`](decider-0.8b-port.md)), so every Python gate here loads the
AOT `.aimodelc` with `SpecializationOptions.default()`.

The Python runtime leaks one IOSurface per call, so each gate process takes at most 14 runs plus a
re-run of its first run, which must reproduce its slot logits bit for bit.

## Swift host notes

- `InferenceFunction.MutableViews` borrows what it holds until `run` returns. `DecisionDecoder` keeps
  the four states and the logits buffer in a class property and moves them into locals for a row's
  whole pass, because a class property's access scope does not cover that borrow (the source's own
  comment; no failure of the other form is recorded in the transcripts).
- The states are allocated once at `max_context_length` (4,096) and zeroed per row (≈ 60 MB fp16);
  zeroing takes 0.53 ms (the timed passes in `gate-decider-2b-vision-swift.json` D).
- The load checks the contract: input, state and output names, shapes and types of both functions and
  the tower, and the letters' ids through the tokenizer.

## iPhone 18 Pro: device JIT works, the memory entitlement is not optional

The phone runs the same `.aimodel` files the Mac runs, specialized on the device by the Core AI
runtime; no AOT asset is involved (the Mac AOT of the two-function decoder is 10.2 GB, far past
the iOS load wall). The gate app (`apps/DeciderVisionGate`, the FunASRGate pattern, one target for
iOS and macOS) links the same `apps/DeciderVision` library, so the phone and the Mac run one code
path; the app carries `com.apple.developer.kernel.increased-memory-limit`.

- Without the entitlement the decoder's cold specialization dies with `std::bad_alloc` (SIGABRT in
  `MPSGraphExecutable specializeWithDevice`, ~5 s in) at the default limit — about 3.5 GB
  available at launch; the tower alone loads fine (0.69 s). With it, 6.4 GB is available and the
  peak footprint through the whole gate stays under 543 MB (the weights are mapped). The
  pipelined-engine note recorded the same failure for 2B-class bundles; this is the second time.
  A new bundle id needs the capability registered on its App ID (Xcode's "+ Capability" →
  Increased Memory Limit, once, with the account signed in); an entitlements file alone makes
  automatic signing fail with "not found and could not be included in profile".
- Numerics on the phone = the Mac's: 108/108 argmax against the oracle (max |Δp| 0.0091, mean
  0.00027, both the cold and the warm run), 108/108 against the Mac Swift JIT run of the same
  files (max |Δp| 0.0015; letter logits bit-equal on 28/108 slots — the phone's specialization
  fuses differently, which is the fp16 noise the Qwen3-VL port also saw). The warm re-run
  reproduced the cold run's slot logits bit for bit.
- Load: decoder 21.6 s cold (14.7 s of it the model's specialization) and 7.9 s from the
  runtime's cache; a tower 0.6 / 0.5 s. The cache for the three graphs is 6.7 GB on the phone —
  budget the container for it. The first decision after a cold load costs 8.1 s more (remaining
  shapes), 0.98 s after a warm load.
- A decision: `g256` 754–785 ms, `g448` 1,415–1,439 ms, text 725–735 ms (medians of 5, USB
  power, battery 80 %, thermal nominal except the cold `g448` burst at fair). About 2.3× the M4
  Max. Sideloading the 3.7 GB of assets: 317 s to push, 1,042 s to pull back for the md5 check.
- Harness trap: an md5 stage that keeps an 8 MB read buffer per file alive until the stage ends
  left the process at a 4.37 GB footprint at exit — survivable only because of the entitlement.
  Fixed in `GateSupport.swift` (a chunk-local autorelease pool: 4,024 → 41 MB on the Mac);
  `apps/FunASRGate` carries the same `md5Hex`.

Transcript: `models/decider-2b-vision/gate-decider-2b-vision-iphone.json` (the stopped run without
the entitlement is kept in it).

## Numbers of record

- Swift, Release, AOT int8mix_pf16, machine-wide GPU lock taken at 0 % GPU: decision 0.322 / 0.321 s
  (`g256`), 0.584 / 0.579 s (`g448`), 0.284 / 0.286 s (text), two passes; tower 22.0 / 61.3 ms; `main`
  11.6 ms per call, `prefill` 25.2 ms per 16 tokens.
- What a fixed grid costs, with the author's code alone (`conversion/decider_vision/grid_price.py`,
  lane `price/grid_price.json`): visual7w 150 accuracy 0.940 native / 0.913 `g256` / 0.940 `g448`
  (argmax = native 0.947 / 0.980); vsr 100 0.770 / 0.740 / 0.770 (0.910 / 0.940). The 256×240 game
  frames get the 64-token grid from the processor itself, so `native` = `g256` there (7/7 rows).
