"""Stage + upload the TimesFM 2.5 200M Core AI bundle to HF.

Stages the ctx-2048 fp16 .aimodel (host front-pads shorter contexts) + the Python host-DSP
reference (timesfm_core.py / host_forecast.py) + an Apache-2.0 model card into /tmp/timesfm_hf,
then uploads to mlboydaisuke/TimesFM-2.5-200M-CoreAI.

    coreai-models/.venv/bin/python conversion/_timesfm_hf_upload.py
"""
import os, shutil
os.environ["HF_HUB_DISABLE_XET"] = "1"
from pathlib import Path
from huggingface_hub import HfApi

REPO = "mlboydaisuke/TimesFM-2.5-200M-CoreAI"
HERE = Path(__file__).resolve().parent
CONV = HERE / "timesfm"
# built by: coreai-models/.venv/bin/python conversion/timesfm/export_timesfm.py \
#           --st <model.safetensors> --dtype float16 --ctx 2048 --no-verify
BUNDLE = CONV / "exports" / "timesfm_2p5_200m_ctx2048_fp16.aimodel"
STAGE = Path("/tmp/timesfm_hf")

CARD = """---
license: apache-2.0
library_name: coreai
pipeline_tag: time-series-forecasting
base_model: google/timesfm-2.5-200m-transformers
tags: [core-ai, coreaikit, timesfm, time-series, forecasting, on-device, apple]
---

# TimesFM 2.5 200M — Core AI

[`google/timesfm-2.5-200m-transformers`](https://huggingface.co/google/timesfm-2.5-200m-transformers)
(Apache-2.0, 200M) converted to **Apple Core AI** `.aimodel` — the
[zoo](https://github.com/john-rocky/coreai-models-community)'s **first time-series forecasting
foundation model**. A decoder-only patched transformer: feed it any univariate series, get a
**128-step point + 10-quantile forecast**, entirely on device.

TimesFM is a **decoder-only transformer over time-series *patches*** (32 points/patch), with the
familiar LLM stack — RoPE, RMSNorm sandwich-norm, QK-norm, a learnable per-dim attention scale — but
numeric patches in and quantile forecasts out. The zoo port runs it as **one stateless Core AI graph
+ a host DSP wrapper** (RevIN normalization, flip-invariance, continuous-quantile head): no LLM
runtime, just CoreAIKit's `GraphModel`.

## Contents

- `timesfm_2p5_200m_ctx2048_fp16.aimodel` — the transformer graph (fp16, ~463 MB). Fixed context
  **2048** (64 patches); **shorter series are front-padded + masked by the host**, so one bundle
  covers every context length ≤ 2048.
  Inputs `tok_in[1,64,64]`, `cos/sin[1,64,80]`, `attn_bias[1,1,64,64]` →
  outputs `proj_point[1,64,1280]`, `proj_q[1,64,10240]`.
- `host/` — the Python host-DSP reference (`timesfm_core.py`, `host_forecast.py`): patching,
  two-level RevIN (global + per-patch causal Welford), flip-invariance (2 graph calls on ±input),
  continuous-quantile head, denormalization, positivity clamp. This is the exact spec the Swift
  `Forecaster` follows.

## Gates (vs the HF `TimesFm2_5ModelForPrediction` fp32 oracle)

- Re-authored graph vs HF projections: **cos 1.0000000** (MAE ~1e-6).
- Independent host DSP + graph vs HF final forecast: **cos 1.0000000** (rel ~1e-8).
- Core AI **fp16** graph, Mac GPU: **cos ≥ 0.99998**; end-to-end forecast **cos 0.9999999**,
  values match HF to 2–3 decimals — including a front-padded short-context case.
- **iPhone 17 Pro, in-app (`KitForecaster`, AOT h18p): device forecast == Mac to 3 decimals**
  (Δ ≤ 0.001, fp16 GPU rounding).
- Mac GPU **~7 ms/graph → ~14 ms per 128-step forecast** (flip = 2 calls);
  **iPhone 17 Pro ~25 ms warm** (54 ms cold). iOS h18p AOT: clean, device-verified.

## Use (Python, Core AI runtime)

```python
import numpy as np, torch, coreai.runtime as rt, asyncio
from host_forecast import forecast          # host/host_forecast.py
from timesfm_core import EngineCore          # thin engine adapter (see host/)

CFG = dict(patch=32, horizon=128, hidden=1280, layers=20, heads=16,
           head_dim=80, inter=1280, q=9, oql=1024, eps=1e-6)
model = asyncio.run(rt.AIModel.load("timesfm_2p5_200m_ctx2048_fp16.aimodel",
                                    rt.SpecializationOptions.from_preferred_compute_unit_kind(
                                        rt.ComputeUnitKind.gpu())))
core = EngineCore(model.load_function("main"), torch.float16)
series = torch.tensor(my_1d_series, dtype=torch.float32)     # any length ≤ 2048
mean_pred, full_pred = forecast(core, series, ctx_len=2048, cfg=CFG)   # (128,), (128,10)
```

## Use (CoreAIKit, Swift)

```swift
let forecaster = try await KitForecaster(catalog: "timesfm-2.5-200m")
let out = try await forecaster.forecast(series)          // [Float] → point + quantiles
// out.mean (128-step), out.quantiles (128 × 10)
```

Base model: TimesFM 2.5 (Google Research). Core AI export: coreai-model-zoo. Apache-2.0.
"""


def main() -> None:
    if not BUNDLE.exists():
        raise SystemExit(f"missing bundle {BUNDLE} — build it with export_timesfm.py --ctx 2048")
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "host").mkdir(parents=True)
    shutil.copytree(BUNDLE, STAGE / BUNDLE.name)
    for f in ["timesfm_core.py", "host_forecast.py"]:
        shutil.copy(CONV / f, STAGE / "host" / f)
    (STAGE / "README.md").write_text(CARD)
    print("staged:", STAGE)
    for p in sorted(STAGE.rglob("*")):
        if p.is_file():
            print("  ", p.relative_to(STAGE))

    api = HfApi()
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(STAGE), repo_id=REPO, repo_type="model")
    print("uploaded ->", f"https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
