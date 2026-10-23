# timestral/ — Timestral-209M (mistralai/Timestral-209M) → Core AI

The recipe behind [`models/timestral/`](../../models/timestral/README.md). Timestral-209M is a patch-based time-series
foundation model (32 steps per patch, d 768, 24 blocks, 21 quantiles). This port covers one univariate series without
covariates and a horizon of at most 1,024 steps (one decode block): the graph is the transformer over
`[context patches ; forecast queries]` plus the quantile head, and the host does everything the released
`TimestralForecaster.forecast()` does around it, in NumPy.

Each stage is a script gated against the one before. Run them from this directory; the export resolves the checkpoint
and the output directory through [`../_paths.py`](../_paths.py) (the pinned revision in the Hugging Face cache,
`<ZOO_EXPORTS>/timestral`), the other scripts take their paths as arguments.

| Stage | Script | Gate |
|---|---|---|
| 0 weights | `huggingface-cli download mistralai/Timestral-209M --revision 9874d252ab0aae16f7f7c999f98f98d6ad7d8365` | `model.safetensors` sha256 `1456effb9927e3daa8c7234f10497e3424233869245a9c72b8ecebaf7ad2b7fa` (the export records it) |
| 1 fixture | `python3 fixture_build.py --raw <raw> --out <fixtures>` (numpy + pyarrow) | 84 windows in 22 groups: UCI Air Quality, Datadog BOOM and UCI Electricity windows in their benchmarks' standard settings, plus 12 synthetic series (short, leading and inner NaN, constant, level 1e6, negative, intermittent, 8,192 steps); sources, licences and window rules in the fixture's `RECIPE.md` |
| 2 oracle | `oracle.py --weights <dir> --fixtures <fixtures> --repo <timestral checkout>`, in the released code's own environment (`github.com/mistralai/timestral` @ a534bdb, `uv sync --frozen`) | the released `forecast()` with every default (symmetric averaging, non-negative clip, cap, one block), fp32 CPU, one call per window; writes `<group>.oracle_q.npy`; a second pass over window 0 of each group must be bit-equal |
| 3 L1 (torch) | `PYTHONPATH=<timestral>/src python3 gate_core.py --weights <dir> --fixtures <fixtures> --bucket 64,32 --max-context-steps 2048 --out l1_torch_L64_h32.json --save-ref ref_L64_h32.npz` (and `--bucket 256,32 --min-context-steps 2049`) | the re-authored graph vs the released module up to its head, on the same rows `[s, -s]`: max \|Δ\| ≤ bar = 2 × (released fp32 vs released `.double()`), monotone; the red arm (the fused wQKV read without reordering its rows) must fail; per-layer absmax of every Linear output, residual and RMSNorm sum of squares (the fp16 budget) |
| 4 export | `python3 export_timestral.py --form <form> --bucket L,h --rows R` | records main.hash, op counts, and that the IR holds no home path |
| 5 compile | `python3 aot_timestral.py --aimodel <x.aimodel or bundle name> --arch h16c\|h19p --preferred-compute gpu\|neural-engine [--out-root <dir>]` | macOS (`h16c`, this Mac) or iPhone 18 Pro (`h19p`, compiled and counted only: an iOS bundle never runs on a Mac); Neural Engine regions counted by name |
| 6 L1 (engine) | `python3 engine_l1.py --ref ref_L64_h32.npz --bundle <x.aimodelc> --unit gpu\|neural_engine\|cpu --dtype float32\|float16 --rows R --out l1.json` | the same bar on the Core AI runtime (recorded for the fp16 forms, whose line is L3) |
| 7 L2 / L3 | `python3 e2e_gate.py --fixtures <fixtures> --engine coreai --bundle-main <x.aimodelc> [--bundle-long <y.aimodelc>] --unit … --dtype … --rows R --mode batched\|split --level l2\|l3 --out e2e.json` (`--engine torch --st <safetensors>` for the re-authored graph; `--red qkv` for the red arm) | host + graph vs the oracle on every window: L2 (fp32) = MASE and CRPS within 0.1 % of the oracle in every group and overall; L3 (fp16, Neural Engine, quantized) = MASE and CRPS ≤ 1.010 × the oracle in every group and overall, no NaN / inf, every forecast monotone |
| 8 Mac ms | `python3 timing_mac.py --forms forms.json --fixtures <fixtures> --out t.json` | forms alternated in one process, 40 rounds, the 8 warm-up rounds dropped, median [min–max]; every timed forecast compared with the oracle |
| 9 device | `device_pack.py --fixtures … --refs … --out <inputs>` → [`apps/TimestralGate`](../../apps/TimestralGate/README.md) → `device_score.py --run <pulled run> …` | L4: the device's L1 / L3 verdicts equal the Mac's for the same form; JIT vs AOT and the placement (the Neural Engine request vs the same bundle on the GPU) bit by bit |

`host_forecast.py` is the host (prepare, bucket inputs, finish a pass, the full forecast) and `engines.py` the graph
callables it runs (the torch core, or a Core AI bundle at a fixed row count).

## Graph contract (one bucket = L context patches, h horizon patches, R rows)

| Tensor | Shape | Content |
|---|---|---|
| `patches` in | [R, L, 64] | per context patch: asinh-normalized values (32) and the missing indicator (32); a shorter context is left-padded with unobserved patches |
| `cos`, `sin` in | [L+h, 64] | partial RoPE tables (32 of 64 dims rotate) for positions `arange(L+h)` minus the index of the earliest observed patch |
| `attn_mask` in | [1, 1, L+h, L+h] | 1 = attend, 0 = masked: the context is causal over valid keys, each forecast token reads every valid context patch and the forecast tokens up to itself |
| `quantiles` out | [R, h·32, 21] | monotone quantiles at 0.01, 0.05, 0.1 … 0.9, 0.95, 0.99, in the normalized space of the last context patch |

Rows: symmetric averaging forecasts the series and its mirror. An R = 2 graph takes both in one call, an R = 1 graph in
two. With one variate per series the variate attention has one key and reduces to a gated V projection, also at R = 2.
Horizon patches 0 to `ceil(H / 32) − 1` are the forecast for H steps: a forecast token reads nothing after itself.

Buckets measured: L64 h32 (context ≤ 2,048 steps, horizon ≤ 1,024), L64 h4 (horizon ≤ 128), L256 h32 (context ≤ 8,192).
`host_forecast.forecast` refuses a context longer than its bucket. A caller can cut a longer series to its latest
L × 32 steps, but the released forecaster reads up to 32,768 steps, so such a forecast is not the released one and
was not gated here.

## Forms (`--form`)

| Form | io | What computes in what |
|---|---|---|
| `fp32` | float32 | everything fp32 |
| `fp16-attn32` | float16 | fp16 compute; the attention scores and softmax in fp32 (the scores reach \|q·k/8\| ≈ 2.4e5, past fp16's 65,504); RMSNorm as LayerNorm([x, −x]) |
| `fp16w32` | float32 | fp16 weights, fp32 compute |
| `fp16-layernorm` | float16 | fp16 everywhere: overflows on the GPU (kept as the counterexample) |
| `ane` | float16 | the Neural Engine form ([`timestral_ane.py`](timestral_ane.py)): RMSNorm on x / 2⁷, Q / K / V as three Linears with rank-4 heads, sigmoid and SiLU through tanh, every Linear as a 1×1 Conv2d; fp16 scores, right on the Neural Engine only (the same graph saturates on the GPU). `--sigmoid` builds its control without the tanh rewrite |

The attention mask is a 1 / 0 input selected with `where(mask, scores, fill)` in the graph. An additive finite mask
(TimesFM's −1e4) does not mask scores of this size.

## Environments

- Stages 0–1: any Python with numpy and pyarrow for the fixture.
- Stage 2: the released code's own environment (Python 3.12, torch 2.14.1, numpy 2.5.4 from its `uv.lock`).
- Stages 3–9: the zoo export venv (torch 2.9.0, coreai-torch 0.4.1, coreai-core 1.0.0b2, numpy 2.3.5, safetensors
  0.7.0); stage 3 also imports the released `timestral` package from its source tree (`PYTHONPATH`). The graph imports
  neither.
