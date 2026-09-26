# TimesFM 2.5 200M — Core AI

The zoo's **first time-series forecasting foundation model** — a brand-new capability class next to
the LLMs, VLMs, ASR, detectors and generators. [`google/timesfm-2.5-200m-transformers`](https://huggingface.co/google/timesfm-2.5-200m-transformers)
(Apache-2.0, 200M) takes **any univariate series and returns a 128-step point + 10-quantile
forecast**, entirely on device — the private-by-default answer for sensor, energy, demand, health
and finance series that should never leave the phone. Neither Apple's stock stack nor a real MLX
path ships a time-series FM, so this is green-field on Apple silicon.

TimesFM is a **decoder-only transformer over time-series *patches*** (32 points → 1 token): the
familiar LLM machinery — RoPE, RMSNorm **sandwich-norm** (4 norms/layer, Gemma-style), **QK-norm**,
a **learnable per-dim attention scale** `softplus(s)·log2(e)/√d`, plain (non-gated) MLP — but with
numeric patches in and quantile forecasts out. It runs as **one stateless `.aimodel` graph + a host
DSP wrapper** — no LLM runtime, just CoreAIKit's `GraphModel`.

## Use it

▶️ **Run it (source)** — the [Forecast runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Forecast)
(GUI + CLI):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/Forecast/Forecast.xcodeproj
# → Run, paste/np-load a series, get a fan-chart forecast

# agents / headless (macOS):
cd coreai-kit/Examples/Forecast
swift run forecast-cli --model timesfm-2.5-200m --csv series.csv --horizon 128
```

💻 **Build with it** — the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let forecaster = try await KitForecaster(catalog: "timesfm-2.5-200m")
let out = try await forecaster.forecast(series)     // [Float] of any length ≤ 2048
// out.mean       — 128-step point forecast
// out.quantiles  — 128 × 10 (deciles 0.1…0.9 + mean), for fan charts / prediction intervals
```

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` → product **CoreAIKit**
- Info.plist / entitlements: none
- First run downloads the model — ~463 MB (fp16) — then loads from the local cache
- One bundle covers every context length ≤ 2048; the host front-pads + masks shorter series
- Measure in Release

## Pipeline

```
1-D series (any length ≤ 2048) ──host DSP──▶ patch tokens ──graph──▶ projections ──host DSP──▶ forecast

host (pre):  truncate→front-pad to 2048 · global RevIN (mean/std) · per-patch causal Welford RevIN
             · patch (32) + mask → tok_in[1,64,64] · RoPE cos/sin · causal+padding mask
graph:       input_ff_layer → 20 decoder layers (RoPE · QK-norm · learnable scale · sandwich-norm
             · plain MLP) → output_projection_point[1,64,1280] + output_projection_quantiles[1,64,10240]
             ↳ run TWICE (on +normalized and −normalized) for flip-invariance
host (post): un-RevIN projections (per-patch) · flip-combine · continuous-quantile head
             · denorm (global) · positivity clamp → mean[128] + quantiles[128,10]
```

Only the transformer is on the Core AI engine; every normalization/flip/quantile step is
deterministic host arithmetic (identical Mac ↔ device).

## Numbers

| | value |
|---|---|
| params / dtype | 200M / fp16 (bundle 463 MB) |
| context / horizon | 2048 (64 patches of 32) / 128 steps, 10 quantiles |
| **Mac GPU (M4 Max)** | **~7 ms/graph → ~14 ms per 128-step forecast** (flip = 2 calls) |
| **iPhone 17 Pro (AOT h18p GPU)** | **~25 ms/forecast warm** (54 ms cold, ~0.8 s very-first-run = one-time shader compile), **device-verified in-app** |
| layout since Hub revision `2a0801a1` (2026-09-26) | `ios/` = the JIT `.aimodel` (same bytes as the root graph; every iPhone generation specializes it on its first load), `ios-h18p/` = the h18p AOT bundle above (iPhone 17 Pro only). On the iPhone 18 Pro (h19p) the JIT graph loaded in 1.54 s the first time an app container held no specialization of it, ran a first call in 0.75 s and loaded in 0.44 s on relaunch (load-only check, 2026-09-26). |
| parity vs HF fp32 oracle | graph **cos 1.0000000**; end-to-end fp16 **cos 0.9999999**, values match to 2–3 dp (incl. front-padded short context); **iPhone in-app == Mac to 3 dp** |

## Gate ladder (vs `TimesFm2_5ModelForPrediction` fp32)

1. Re-authored `TimesFmCore` vs HF captured projections — **cos 1.0000000**, MAE ~1e-6.
2. Independent host DSP + core vs HF final forecast — **cos 1.0000000**, rel ~1e-8.
3. Core AI export fp32 (CPU) — **cos 1.0000000**; fp16 (CPU/GPU) — **cos ≥ 0.99998**.
4. End-to-end host DSP + fp16 GPU engine vs HF — **cos 0.9999999** across sine / trend / random-walk
   / padded-short.
5. **iPhone 17 Pro, in-app (`KitForecaster`, AOT h18p)** — forecast **matches the Mac reference to 3
   decimals** (Δ ≤ 0.001, fp16 GPU rounding); **~25 ms warm** per 128-step forecast.

Conversion: [`conversion/timesfm/`](../../conversion/timesfm/) (`export_timesfm.py`, `timesfm_core.py`,
`host_forecast.py`, `oracle.py`, `gate_core.py`, `e2e_engine_gate.py`). Knowledge:
[`knowledge/timesfm-port.md`](../../knowledge/timesfm-port.md).
