# DecideGate: the iPhone gate for GLiNER2.5-Decide

A headless iPhone app that runs the Mac self-test of [`conversion/gliner25_decide`](../../conversion/gliner25_decide)
on the phone. The graph host, the fixtures and the metrics are the same code: the app links the conversion's Swift
package by path (`DecideGraph` + `DecideGateSupport`). The gate starts at launch with no UI input. It writes
`Documents/decide_gate/result.json` (rewritten after every stage) and `result.log`.

Stages, each on a sideloaded AOT bundle (fp16, preferred gpu) compiled for the phone's own Core AI architecture:
the app loads `gliner25-decide_float16_<tag>_m32.<arch>.aimodelc`, `<arch>` = `AIModel.deviceArchitectureName`
(`h19p` on the iPhone 18 Pro, iPhone19,2; `h18p` on the iPhone 17 Pro). A bundle compiled for another architecture
fails at load with `incompatibleCompiledAssetArchitecture`.

| stage | bundle | cases (gliner2 2.0.0 fp32 oracle) |
|---|---|---|
| `gpu_s256` | `gliner25-decide_float16_s256_m32.<arch>.aimodelc` | README examples 21 + fast-decisions 340 (≤ 256 tokens): 606 tasks |
| `gpu_s512` | `gliner25-decide_float16_s512_m32.<arch>.aimodelc` | the same + 93 fast-decisions rows of 257–431 tokens: 787 tasks |

A stage passes when every task's decision equals the oracle's (single label: softmax argmax; multi label: sigmoid
≥ the task's threshold, none above → argmax) and every logits row is finite. Each stage also records the first
and second load time, the first call, the app footprint, max|Δlogit| and max|Δprob| against the oracle, the
distance to the Mac GPU's logits for the same fp16 bundle, and a bench of 5 warm-up calls and 100 timed calls on
fixture inputs. The thermal state is logged around every step, with the device model (`utsname.machine`), the OS
build and the Core AI architecture name.

## Steps (Mac, then the phone)

1. In `conversion/gliner25_decide`: `aot_compile.py --tag s256` and `aot_compile.py --tag s512` (iOS h18p and h19p,
   macOS h16c, gpu preferred, no expect-frequent-reshapes: the shapes are fixed).
2. `./_build.sh`: xcodegen + `xcodebuild` Release for generic iOS. The `.app` path goes to `_work/decidegate_app_path.txt`.
3. `./_stage.sh`: gathers the sideload set into `_work/device_stage/DecideAssets/` (1.8 GB) and writes `MD5SUMS`:
   the bundles of `DECIDE_ARCHS` (default `h19p`), the fixtures and the Python Mac GPU gate from `DECIDE_DATA`
   (default `~/code/coreai/_gliner25_decide`). A push only adds to the container, so at launch the app deletes any
   `gliner25-decide_*.aimodelc` that `MD5SUMS` does not list (bundles of an earlier stage).
4. Unlock the iPhone and connect it by USB. Get its ids from `xcrun devicectl list devices`.
5. `DECIDE_SESSION=<name> DECIDE_ALLOWED_DEVICES=<udid>,<coredevice id> ./_gate.sh <udid>` takes the device hold
   (`~/code/coreai/ondevice/.device_hold`; while another session holds it, waits up to 60 min), runs steps 6 and 7,
   and releases the hold on any exit. `_work/device_runs/hold.log` records every take, wait and release.
6. `./_install.sh <udid>` installs the app and pushes `DecideAssets/` to `Library/Application Support/`. It pulls the
   files back and re-pushes any whose md5 differs. It runs only under this lane's hold.
7. `./_run.sh <udid>` launches with no console and polls `result.json` until it is done. Results land in
   `_work/device_runs/<run id>/`.
8. Exit codes: 0 = every stage passed, 3 = a stage failed, 1 = no result, 2 = the device is busy, held or refused.
9. Reruns: `DECIDE_SKIP_INSTALL=1 ./_gate.sh <udid> '"DECIDE_STAGES":"s512"'` (knobs are in
   `Sources/GateRunner.swift`). Reprint a result: `./_run.sh --summary <result.json>`.

`_work` is `conversion/gliner25_decide/_work` (git-ignored). Never load an iPhone bundle (`.h18p`, `.h19p`) on a Mac: the host
refuses one by path.
