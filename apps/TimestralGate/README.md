# TimestralGate

The device gate of the Timestral-209M Core AI port (`conversion/timestral`): a headless iPhone app and the same code as
a Mac command-line tool. It runs one graph bundle — a JIT `.aimodel` specialized at load, or an AOT `.aimodelc` — on the
fixture windows packed by `conversion/timestral/device_pack.py`, writes the raw outputs, and times bundles against each
other in one process. It gives no verdict: `conversion/timestral/device_score.py` scores a pulled run on the Mac with
the port's own measures (L1 against the released module's fp32 output, L3 = MASE and CRPS of the full forecast against
the released forecaster).

## The graph it drives

`patches [R, L, 64]`, `cos [L+h, 64]`, `sin [L+h, 64]`, `attn_mask [1, 1, L+h, L+h]` (1 = attend) →
`quantiles [R, h·32, 21]`, one input dtype (float32, or float16 for the fp16-compute forms). A window has two rows,
the series and its mirror (symmetric averaging): an R = 2 graph takes both in one call, an R = 1 graph in two calls.
Everything outside the graph (scaling, the refine frame, the cap, the averaging) stays on the Mac in NumPy
(`host_forecast.py`); this app moves only the graph.

## Layout

| Path | What |
|---|---|
| `Sources/Gate/Gate.swift` | the modes, shared by both targets |
| `Sources/iOS/App.swift` | the iPhone app: one run per launch from the launch arguments, screen kept on |
| `Sources/macOS/main.swift` | the Mac tool (macOS bundles only; it refuses an `.aimodelc` for another architecture) |
| `_build.sh ios\|mac` | `xcodegen generate` + `xcodebuild` Release into `_work/` |
| `_install.sh <udid> <push list>` | installs the app, pushes files one by one, pulls every file back and compares md5 |
| `_run.sh <udid> <tag> <args…>` | launches with a tag, follows `Documents/out/<tag>/log.txt`, pulls the run |

On the phone: bundles under `Documents/models/`, packs under `Documents/inputs/L<L>_h<h>/`, runs under
`Documents/out/<tag>/` (`log.txt` whose last line is `DONE …` or `ERROR: …`, `run.json`, `pass<k>/quantiles.f32.bin`).

## Modes

| `--mode` | What it does |
|---|---|
| `gate` | one `--bundle` / `--compute` on every window of its bucket's pack, `--passes` times (pass k compared with pass 1 bit for bit) |
| `timing` | every `--config <bundle>@<compute>` loaded once (`--cold`: cache entry deleted first) and its first forecast timed; then `--rounds` rounds A/B/C… (odd rounds reversed) of `--warm` + `--calls` forecasts on `--window`, stopped at `--budget` s |
| `warm` | every `--config` loaded and run once |
| `probe` | device, OS, thermal state, power, free disk, and what `models/` and `inputs/` hold |
| `thermal` | waits for the nominal thermal state (`--cap` s) |

`--compute` is `gpu`, `ane` (`preferredComputeUnitKind: .neuralEngine`), `cpu`, `cpuonly` or `default`. A request for
the Neural Engine never fails at load: a graph it cannot place runs elsewhere, so placement is judged by comparing the
output with the same bundle's GPU output, not by the request.

## Example

```sh
./_build.sh mac
_work/dd_mac/Build/Products/Release/TimestralGateMac --root /tmp/tg --inputs <inputs> --mode gate \
  --bundle <exports>/timestral_209m_L64_h32_r2_fp16w32.aimodel --compute gpu --tag mac-fp16w32 --cold
python3 ../../conversion/timestral/device_score.py --run /tmp/tg/out/mac-fp16w32 --inputs <inputs> \
  --fixtures <fixtures> --refs <dir of ref_*.npz> --out score.json
```
