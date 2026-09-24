# N3DGate: the iPhone gate for Nemotron-3-Diarization

A headless iPhone app that runs the Mac self-test of [`conversion/nemotron3_diar`](../../conversion/nemotron3_diar)
on the phone. The streaming host, the golden files and the metrics are the same code: the app links the
conversion's Swift package by path (`NemotronDiarizer` + `N3DGateSupport`). The gate starts at launch with no
UI input. It writes `Documents/n3d_gate/result.json` (rewritten after every stage) and `result.log`.

Stages, each on a sideloaded AOT h18p bundle:

| stage | bundle | runs (agreement@0.5 ≥ 99.9 % vs transformers fp32) |
|---|---|---|
| `gpu_streaming` | `n3d_streaming_float16.h18p.aimodelc` (preferred gpu) | 97.6 s and 21.5 s, low latency |
| `gpu_offline` | `n3d_offline_float16.h18p.aimodelc` (preferred gpu) | 97.6 s and 21.5 s, offline (last 16 frames left out) |
| `ane_streaming` | `ane/n3d_streaming_float16.h18p.aimodelc` (preferred neural-engine; optional) | same as `gpu_streaming` |

Each stage also records the first and second load time, the first call, the app footprint and a bench.
The bench is 5 warm-up calls and 100 timed calls of the graph, then one timed 97.6 s run. The thermal state
is logged before and after every step, with the device model (`utsname.machine`) and the OS build. Streaming
runs are also compared with the Mac GPU's logits for the same bundle, and the ANE runs with the GPU stage's.

## Steps (Mac, then the phone)

1. `./_build.sh`: xcodegen + `xcodebuild` Release for generic iOS. The `.app` path goes to `_work/n3dgate_app_path.txt`.
2. In `conversion/nemotron3_diar`, the AOT bundles: `aot_compile.py` and `aot_compile.py --bundle _work/artifacts/n3d_offline_float16.aimodel --tag offline --targets ios_gpu`.
3. `./_stage.sh`: gathers the sideload set into `_work/device_stage/N3DAssets/` (578 MB) and writes `MD5SUMS`.
4. Unlock the iPhone and connect it by USB. Get its id from `xcrun devicectl list devices`.
5. If another session holds the phone (`~/code/coreai/ondevice/.device_hold`), wait.
6. `./_gate.sh <udid>` runs steps 7 and 8.
7. `./_install.sh <udid>` installs the app and pushes `N3DAssets/` to `Library/Application Support/`. It pulls the files back and re-pushes any whose md5 differs.
8. `./_run.sh <udid>` launches with no console and polls `result.json` until it is done. Results land in `_work/device_runs/<run id>/`.
9. Exit codes: 0 = every stage passed, 3 = a stage failed, 1 = no result, 2 = the device is busy or held.
10. Reruns: `./_run.sh <udid> '"N3D_STAGES":"gpu"'` (knobs are in `Sources/GateRunner.swift`). Reprint a result: `./_run.sh --summary <result.json>`.

`_work` is `conversion/nemotron3_diar/_work` (git-ignored). Never load an `.h18p` bundle on a Mac: the host
refuses one by name.
