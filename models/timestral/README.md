# Timestral-209M — Core AI

[🤗 mlboydaisuke/Timestral-209M-CoreAI](https://huggingface.co/mlboydaisuke/Timestral-209M-CoreAI) (revision `5d1713f`) ·
weights under Mistral AI's Modified MIT License, which has a revenue condition ([Licence](#licence)) · source
[mistralai/Timestral-209M](https://huggingface.co/mistralai/Timestral-209M/tree/9874d252ab0aae16f7f7c999f98f98d6ad7d8365)
(revision `9874d25`) · reference code
[mistralai/timestral](https://github.com/mistralai/timestral/tree/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff) (`a534bdb`,
Apache-2.0)

This is Mistral AI's Timestral-209M, a time-series foundation model, as Core AI `.aimodel` graphs converted with
Apple's `coreai-torch`; on the Neural Engine of an iPhone 18 Pro one forecast from a 2,048-step history takes 9.44 ms
of graph time, the host's scaling not included (2026-10-22).

Give it one series, up to 8,192 past values with NaN where a value is missing, and a horizon of up to 1,024 steps.
It returns 21 quantiles per step (0.01, 0.05, 0.1 … 0.9, 0.95, 0.99) with the released `forecast()`'s defaults:
symmetric averaging (the series and its mirror, averaged), the advancing refine frame (the released code's per-patch
re-denormalization of the block, `_refine_block`), the cap (the history's range widened by 10⁴ times the running
standard deviation at its last step) and the non-negative clip.

## What this port covers

- One univariate series without covariates. The released model also forecasts several variates together and takes
  past-only and known-future covariates; these graphs do not.
- A horizon of up to 1,024 steps, one decode block. The released forecaster goes further block by block, appending
  each block's median to the context; that loop is not here.
- A context of up to 8,192 steps, in buckets of 2,048 and 8,192. The released forecaster reads up to 32,768; a longer
  series would have to be cut to its latest 8,192 steps, which gives a different forecast and was not gated.

## Bundle

One graph per fixed input shape (a bucket of context and horizon patches): on each device, the graph variant that
passed its parity check (see [Parity](#parity)) with the lowest latency. In the names, L = context patches of 32 steps
(L64 = 2,048 steps, L256 = 8,192), h = horizon patches (h4 = 128 steps, h32 = 1,024) and R = rows per call (R = 2 runs
the series and its mirror together; R = 1 × 2 is two R = 1 calls). Pick the smallest bucket that holds the series:
L64 h4 when the context is at most 2,048 steps and the horizon at most 128; L64 h32 when the context is at most 2,048
steps and the horizon up to 1,024; L256 h32 when the context is 2,049–8,192 steps.

| Folder | Bundle | Form | Serves | Size |
|---|---|---|---|---:|
| `macos/` | `timestral_209m_L64_h32_r1_fp16p7-qkv4-tanh-conv.aimodel` | Neural Engine form, R = 1 (the series and its mirror in two calls) | context ≤ 2,048 steps, horizon ≤ 1,024 | 404.7 MB |
| `macos/`, `ios/` | `timestral_209m_L64_h4_r2_fp16p7-qkv4-tanh-conv.aimodel` | Neural Engine form, R = 2 (one call) | context ≤ 2,048 steps, horizon ≤ 128 | 404.7 MB |
| `macos/` | `timestral_209m_L256_h32_r2_fp16-attn32.aimodel` | GPU: fp16 compute, attention scores and softmax in fp32, R = 2 | context ≤ 8,192 steps, horizon ≤ 1,024 | 404.7 MB |
| `ios/` | `timestral_209m_L64_h32_r2_fp16p7-qkv4-tanh-conv.aimodel` | Neural Engine form, R = 2 | context ≤ 2,048 steps, horizon ≤ 1,024 | 404.8 MB |
| `ios/` | `timestral_209m_L256_h32_r1_fp16p7-qkv4-tanh-conv.aimodel` | Neural Engine form, R = 1 | context ≤ 8,192 steps, horizon ≤ 1,024 | 404.7 MB |
| `ios-h19p/` | the three `ios/` names as `.h19p.aimodelc` | the `ios/` bundles compiled for the iPhone 18 Pro's Neural Engine | the same | 405.7–405.8 MB each |

The `ios/` and `ios-h19p/` bundles, and the L64 h4 bundle that `macos/` shares with `ios/`, are the bytes measured on
the iPhone 18 Pro. The other two `macos/` bundles are rebuilt with [`conversion/timestral/`](../../conversion/timestral/);
their outputs equal the measured bundles' bit for bit ([gate ladder](#gate-ladder), step 6).

`fp16p7-qkv4-tanh-conv` names the Neural Engine form: fp16 with RMSNorm on x / 2⁷ (p7), Q, K and V as three
projections (qkv4), sigmoid and SiLU written through tanh, and every linear layer as a 1×1 convolution (conv). `h19p`
and `h16c` are the ahead-of-time compile targets of the iPhone 18 Pro and of the M4 Max used here. A `.aimodel` is
specialized at load (the runtime compiles it for the device when it loads it); a `.aimodelc` is that step done in
advance.

The Mac and the iPhone run different forms where a different form measured quicker: on the M4 Max two R = 1 calls
beat one R = 2 call by 5 % at 2,048 steps and the GPU beats the Neural Engine at 8,192 steps; on the iPhone 18 Pro it
is the other way round.

An iPhone 18 Pro loads the `ios-h19p/` compile in about 1 s. On the same phone the `ios/` `.aimodel`, specialized at
load for the Neural Engine, returns the compile's numbers bit for bit, but its load took 30–59 s in a launch that
held several bundles (1.2–1.5 s when it held that bundle alone; the cause was not isolated). Other iPhones were not
measured. On the Mac a Neural Engine `.aimodel` loaded in 4.7–9.3 s with the app's cache entry removed; the initial
load of a Neural Engine compile there took 36–53 s, and what makes it slow was not isolated.

The Neural Engine form must run on the Neural Engine: its attention scores are fp16, and the same graph on the GPU
saturates at ±65,504. Load it with the Neural Engine preferred and judge where it ran by its output, not by the
region names inside the compiled bundle.

The repository also holds the source's `config.json` and `LICENSE`, unchanged.

## Use it

**CoreAIKit** — the [Forecast example](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Forecast): paste
numbers or pass a CSV file; the kit picks the bucket from the series and the horizon.

```bash
git clone https://github.com/john-rocky/coreai-kit
cd coreai-kit/Examples/Forecast
swift run forecast-cli --model timestral-209m --series sales.csv --horizon 48 --band 90
```

```swift
import CoreAIKit

let timestral = try await KitTimestralForecaster(catalog: "timestral-209m")
let q = try await timestral.forecast(series, horizon: 96)     // series: [Float], NaN = a missing value
q.median, q.quantile(0.1), q.quantile(0.9)                    // 96 values each
```

Each graph runs on the compute unit its form needs, with no unit passed: the Neural Engine form on the Neural Engine,
the Mac's 8,192-step GPU form on the GPU. Asking for the GPU for a Neural Engine graph (`--compute gpu`,
`computeUnits: .gpu`) stops with an error before loading, because there its fp16 attention overflows; an output that
reaches fp16's limit also stops with an error instead of returning a wrong forecast. A series longer
than the largest bucket keeps its most recent 8,192 points (`q.details` says how many were read); an empty series, one
with no observed value, or a horizon outside 1…1,024 is refused with the reason.

**Python** — the reference host is a NumPy copy of the released forecaster's host steps (with the re-authored fp32
graph its overall MASE and CRPS are within 2 × 10⁻⁷ of the oracle's). This example loads the compiled form; the Swift
runtime loads the `.aimodel` directly (the Mac numbers below). Compile the bundle for the Mac's Neural Engine once,
then forecast (from [`conversion/timestral/`](../../conversion/timestral/), in the Python environment listed under
`needs` in [`recipe.toml`](recipe.toml)):

```bash
python3 aot_timestral.py --aimodel macos/timestral_209m_L64_h32_r1_fp16p7-qkv4-tanh-conv.aimodel \
    --arch h16c --preferred-compute neural-engine --out-root compiled
```

```python
import engines, host_forecast as H

graph = engines.CoreAIGraph("compiled/timestral_209m_L64_h32_r1_fp16p7-qkv4-tanh-conv-h16c-ane/"
                            "timestral_209m_L64_h32_r1_fp16p7-qkv4-tanh-conv.h16c.aimodelc",
                            "neural_engine", "float16", rows=1)
q = H.forecast(graph, series, prediction_length=96, bucket=H.Bucket(64, 32), mode="split")  # (96, 21)
median, p10, p90 = q[:, 10], q[:, 2], q[:, 18]
```

`series` is a 1-D array of at most 2,048 values for this bucket (NaN = missing); columns 10, 2 and 18 of `q` are the
levels 0.5, 0.1 and 0.9 of the 21. This host checks only the bucket and the horizon (an `AssertionError` otherwise): an
empty series raises an `IndexError`, and a series with no observed value gets the released forecaster's degenerate
forecast.

## Pipeline

```
series (≤ 8,192 steps, NaN = missing) ──host──▶ patches, RoPE tables, mask ──graph──▶ quantiles ──host──▶ forecast

host (pre):  NaN → missing · left-pad to whole patches · causal running mean / std in float64
             · loc / scale = the statistics at each patch's last step · asinh((x − loc) / scale) per value
             → [values 32, missing 32] per patch
             · the mirror −x as a second row · RoPE cos / sin for positions from the earliest observed patch
             · the attention mask (1 = attend) for the bucket
graph:       patch embedding → [context ; h forecast queries] → 24 blocks (time attention with partial RoPE
             and a sigmoid gate; every fourth block also the variate sublayer) → out_norm → head
             → monotone quantiles (R, h·32, 21) in the normalized space of the last context patch
host (post): sinh · denormalize with the last patch's statistics · the advancing refine frame (float64)
             · the cap · average the series with its flipped mirror · clip at 0 when the history is non-negative
```

Only the network (patch embedding, 24 blocks, head) runs on Core AI. The host is
[`host_forecast.py`](../../conversion/timestral/host_forecast.py) (NumPy, float64 where the released code uses float64).

## Graph contract

```
function "main"   (one bucket = L context patches, h horizon patches, R rows)
  input  "patches"    [R, L, 64]          per patch: 32 asinh-normalized values, 32 missing flags
  input  "cos"        [L+h, 64]           partial RoPE (32 of 64 dims rotate), θ 10,000
  input  "sin"        [L+h, 64]
  input  "attn_mask"  [1, 1, L+h, L+h]    1 = attend, 0 = masked
  output "quantiles"  [R, h·32, 21]       monotone, normalized space
dtype: float16 (the Neural Engine form, fp16-attn32) or float32 (fp32)
```

The mask is an input selected with `where(mask, scores, fill)` inside the graph. Timestral's attention scores reach
|q·k/8| ≈ 2.4 × 10⁵ on the fixture, so an additive finite mask (TimesFM's −10⁴ in this zoo) does not mask them. With
one variate per series the variate attention has one key and is a gated V projection, also when the series and its
mirror ride in one R = 2 call.

## Measured

**iPhone 18 Pro** (iPhone19,2), iOS 27.2 (24B5099f), [`apps/TimestralGate`](../../apps/TimestralGate/README.md) (Swift),
2026-10-22. Graph time per forecast = the Core AI call(s) only (one R = 2 call or two R = 1 calls; the host's
normalization, refine and averaging are not in these numbers), median of 8 rounds × 4 forecasts (3 at 8,192 steps)
[range of the round medians], the forms
alternated in one launch, thermal state nominal throughout, on USB power. Inputs: an Air Quality window of 2,048
steps (horizon 168), a BOOM window of 2,048 steps (horizon 96) for h ≤ 128, an Air Quality window of 8,192 steps.

| Bucket | Shipped form | ms | Other forms in the same launch |
|---|---|---:|---|
| context ≤ 2,048, horizon ≤ 1,024 | Neural Engine form R = 2, `h19p` compile | 9.44 [9.27–9.65] | the same `.aimodel` specialized on the phone 9.53; R = 1 × 2 10.19; GPU fp16-attn32 14.58 |
| context ≤ 2,048, horizon ≤ 128 | Neural Engine form R = 2, `h19p` compile | 8.42 [8.29–8.65] | `.aimodel` 8.43; R = 1 × 2 9.74; GPU fp16-attn32 13.90; GPU fp32 27.86 |
| context ≤ 8,192, horizon ≤ 1,024 | Neural Engine form R = 1 × 2, `h19p` compile | 33.86 [33.45–35.06] | `.aimodel` 34.14; R = 2 compile 37.36, `.aimodel` 34.74; GPU fp16-attn32 40.13 |

On the phone's GPU, fp16 compute is 2.4 times faster than fp32 (15.19 vs 36.52 ms at 2,048 steps, both `h19p`
compiles in one launch); on the M4 Max's GPU the two are within 5 % (below).

**Mac** — Apple M4 Max (Mac16,9), macOS 27.0 (26A428), the Swift runtime (`apps/TimestralGate`'s Mac tool) with the
`.aimodel` specialized at load, 2026-10-22, on the rebuilt bundles ([gate ladder](#gate-ladder), step 6). Graph time
per forecast, median of 11 rounds × 8 forecasts [range of the round medians], the forms alternated in one process per
bucket, the same input windows as the iPhone's.

| Bucket | Shipped form | ms | Other forms in the same launch |
|---|---|---:|---|
| context ≤ 2,048, horizon ≤ 1,024 | Neural Engine form R = 1 × 2, `.aimodel` | 9.46 [9.42–9.63] | its `h16c` compile 9.48; R = 2 9.96; GPU fp32 11.19; GPU fp16-attn32 11.41 |
| context ≤ 2,048, horizon ≤ 128 | Neural Engine form R = 2, `.aimodel` | 8.24 [8.16–8.29] | its `h16c` compile 8.22; GPU fp32 9.84 |
| context ≤ 8,192, horizon ≤ 1,024 | GPU fp16-attn32 R = 2, `.aimodel` | 24.73 [24.65–24.88] | its `h16c` compile 24.67; GPU fp32 25.93; Neural Engine form R = 1 × 2 35.31 |

## Parity

The oracle is the released `TimestralForecaster.forecast()` with its defaults (fp32, CPU, the reference code at
`a534bdb`), one call per window, on a fixture of 84 windows in 22 groups: UCI Air Quality, Datadog BOOM and UCI
Electricity windows in their benchmarks' standard settings (horizons 24 to 720, contexts of 2,048 and 8,192 steps
and one full daily history), plus 12 synthetic series (short, leading and inner gaps, constant, level 10⁶, negative,
intermittent, 8,192 steps). Three pass bars (L1–L3), fixed before any Core AI number existed:

- **L1** (the graph alone, fp32): max |Δ| against the released module's output on the same inputs ≤ 0.129 (context
  ≤ 2,048) and ≤ 5.2 × 10⁻⁵ (8,192), twice the released fp32's own distance from float64.
- **L2** (fp32 forms, host + graph): MASE and CRPS within 0.1 % of the oracle's, in every group and overall.
- **L3** (fp16, Neural Engine and quantized forms): MASE and CRPS at most 1.010 times the oracle's in every group and
  overall, no NaN or inf, all 21 quantiles in order.

| Shipped bundle | Windows | MASE / CRPS, ratio ours / oracle (1.000 = equal) | Worst group | Bar | Run on |
|---|---:|---|---|---|---|
| Neural Engine form, context ≤ 2,048 (R = 1 and R = 2: identical outputs) | 71 | 0.99800 / 1.00051 | 1.00517 (synthetic, 100 steps) | L3 PASS | M4 Max, iPhone 18 Pro |
| Neural Engine form, horizon ≤ 128 | 39 | 0.99782 / 1.00081 | 1.00517 | L3 PASS | M4 Max, iPhone 18 Pro |
| Neural Engine form, context ≤ 8,192 | 13 | 0.99924 / 0.99992 | 1.00021 (BOOM, 8,192 steps) | L3 PASS | M4 Max, iPhone 18 Pro |
| GPU fp16-attn32, context ≤ 8,192 | 13 | 0.99997 / 1.00002 | 1.00020 | L3 PASS | M4 Max |

On the iPhone 18 Pro the Neural Engine returns the Mac's numbers digit for digit (every window's L1, every group's
MASE / CRPS); on each machine the `.aimodel` specialized on the device and the compile, and R = 1 and R = 2, are bit
for bit the same. The fp16 forms are judged on L3; their L1 distances are above the fp32 bar (the Neural Engine form
0.988 at 2,048 steps), as expected of fp16 arithmetic through 24 blocks. The fp32 GPU graph passes L2 (overall MASE
×0.99999998 and CRPS ×0.99999999 at 8,192 steps, every group within 2.1 × 10⁻⁷) and is not shipped: at 8,192 steps
on the Mac it is 5 % slower than fp16-attn32; at 2,048 steps the Neural Engine form is quicker than both.

### Gate ladder

1. Re-authored graph vs the released module, torch fp32, the series and its mirror: max |Δ| 1.4 × 10⁻⁴ (71 windows)
   and 2.1 × 10⁻⁵ (13 windows at 8,192 steps). The red arm (a deliberately wrong variant, run to show the check can
   turn red), the fused wQKV read without reordering its rows (stored as (qkv, head_dim, heads)), misses by 23.5.
2. NumPy host + re-authored graph vs the oracle on all 84 windows: L2 PASS (MASE ×1.0000002, CRPS ×1.0000001); the
   red arm's MASE is 5.8 times the oracle's.
3. Core AI fp32 on the Mac GPU: L1 1.8 × 10⁻⁴ and L2 PASS on every bucket (71 + 13 + 39 windows).
4. The shipped forms: the table above.
5. The iPhone 18 Pro reaches the Mac's L1 / L3 verdicts on every form gated there (34 runs), and its Neural Engine
   returns the Mac's numbers digit for digit. The Swift tool on the Mac returns the Python engine's numbers window for
   window (20 runs), and on the Mac the Neural Engine `.aimodel` returns its compile's numbers bit for bit.
6. The iOS bytes in this repository (`ios/`, `ios-h19p/`, and the L64 h4 bundle that `macos/` shares with `ios/`) are
   the bundles measured on the iPhone 18 Pro above. The other two `macos/` bundles were rebuilt from
   [`conversion/timestral/`](../../conversion/timestral/), as were the bundles behind the Mac table. An export is not
   byte-reproducible (the same command twice gives a different `main.hash`), so every rebuilt bundle was run again on
   every window of its bucket on the Mac: for all 10 forms rebuilt, the outputs equal, bit for bit, the outputs of the
   bundles the gate and iPhone numbers above were measured on, and the bars hold. A tanh and a sigmoid Neural Engine
   compile, run the same way, differ in 2,954,725 of 3,053,568 elements: the comparison can fail.

## Lessons

Details and evidence in [`knowledge/timestral-port.md`](../../knowledge/timestral-port.md).

- A finite additive mask is not a mask when scores reach 2.4 × 10⁵. The mask is an input and the graph selects with
  `where`.
- fp16 compute overflows on the GPU through the attention scores, not through RMSNorm. fp16 compute with fp32 scores
  and softmax passes L3 and is the phone's quickest GPU form (2.4 times fp32 in one launch); on the Mac it buys 5 % at
  8,192 steps
  and nothing at 2,048.
- On the Neural Engine the graph initially missed L3 by 0.09 % on one group. The cause was the Neural Engine's sigmoid /
  SiLU (the patch embedding alone was 3.4 % off); writing them through tanh passes. An R = 2 graph compiles for the
  Neural Engine only with 1×1-conv projections. The Neural Engine form has the lowest latency of the passing forms
  on the iPhone 18 Pro in every bucket (9.44 ms against 14.58 ms for its best GPU form at 2,048 steps).
- int8 per block of 32 on every Linear misses L3 (×1.018); leaving the patch embedding and the head unquantized passes
  at a 222–250 MB `.aimodel`, but a compile expands the weights back to dense ones and the speed does not improve
  (on the iPhone the `.aimodel` ran 12 % slower than fp16). k-means 8-bit and 6-bit miss.

## Reproduce

[`recipe.toml`](recipe.toml) names the export of every bundle; [`conversion/timestral/`](../../conversion/timestral/)
lists the stages (fixture, oracle, the three lines, compile, timing, the device gate) in order. The device harness is
[`apps/TimestralGate`](../../apps/TimestralGate/README.md).

## Licence

The weights are Mistral AI's, under the Modified MIT License below, included unchanged as `LICENSE` in the bundle's
repository. Its second condition covers the Model and any derivatives, modifications or combined works, these graphs
included: you may not exercise any right under it if the global consolidated monthly revenue of your company (or of
your employer) exceeded $20 million, or its equivalent in another currency, in the preceding month. The reference
code (`mistralai/timestral`) is Apache-2.0; the conversion code in this repository is under the repository's licence
(BSD 3-Clause, `LICENSE` at the repository root).

```
Modified MIT License 

Attribution notice: 2026 - Mistral AI

Permission is hereby granted, free of charge, to any person obtaining a copy of the weights of this model and associated documentation files (the “Model”), to deal in the Model without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Model, and to permit persons to whom the Model is furnished to do so, subject to the following conditions: 

1. The above attribution notice and this permission notice shall be included in all copies or substantial portions of the Model. 
2. You are not authorized to exercise any rights under this license if the global consolidated monthly revenue of your company (or that of your employer) exceeds $20 million (or its equivalent in another currency) for the preceding month. This restriction in (b) applies to the Model and any derivatives, modifications, or combined works based on it, whether provided by Mistral AI or by a third party. You may contact Mistral AI (sales@mistral.ai) to request a commercial license, which Mistral AI may grant you at its sole discretion, or choose to use the Model on Mistral AI's hosted services available at https://mistral.ai/.

THE MODEL IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL MISTRAL AI BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE MODEL OR THE USE OR OTHER DEALINGS IN THE MODEL.
```
