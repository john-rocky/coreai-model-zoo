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

`DECIDE_BUNDLE_KIND` picks another bundle for the same stages: `jit` loads `gliner25-decide_float16_<tag>_m32.aimodel`
(the JIT bundle, as `macos/` ships it: the phone specializes it at the first load), `mixed` loads
`gliner25-decide_float16_<tag>_m32.mixed.aimodel` (the JIT files with another architecture's AOT files beside them:
`main-h18p.mlirb`, `main-h18p-delegates/`, `stats.json`, the shape of GLiNER2-PII-CoreAI's `ios/` bundle). Stage it
with `DECIDE_KINDS=aot,jit,mixed ./_stage.sh`.

A stage passes when every task's decision equals the oracle's (single label: softmax argmax; multi label: sigmoid
≥ the task's threshold, none above → argmax) and every logits row is finite. Each stage also records the first
and second load time, the first call, the app footprint, max|Δlogit| and max|Δprob| against the oracle, the
distance to the Mac GPU's logits for the same fp16 bundle, and a bench of 5 warm-up calls and 100 timed calls on
fixture inputs. The thermal state is logged around every step, with the device model (`utsname.machine`), the OS
build, the Core AI architecture name and the battery. Each load and the first call run under a 100 ms memory
sampler on its own thread (peak footprint, least `os_proc_available_memory`, a 1 s series, a progress line every
10 s), and the container's caches are sized before load 1, after load 1, after the first call, after load 2 and at
the end (`Library/Caches/coreai-cache` = the container half of Core AI's specialization cache; the system half is out
of the app's reach).

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
   `Sources/GateRunner.swift`); a new app build on assets already there: `DECIDE_SKIP_PUSH=1 ./_gate.sh <udid>`.
   Reprint a result: `./_run.sh --summary <result.json>`.
   A cold first load: `DECIDE_FRESH=1 ./_gate.sh <udid> ...` uninstalls first (the data container goes with the app,
   the container's Core AI cache with it), then installs and pushes everything again. A run that does not end
   `done` leaves the phone's crash-log listing, and copies of today's DecideGate / jetsam reports, in the run directory.
10. The phone slows down as it heats (the thermal state goes from nominal to fair within a minute of back-to-back
    calls). For a nominal-state bench: `'"DECIDE_WAIT_NOMINAL":"600","DECIDE_BENCH_FIRST":"1"'` waits (5 s steps, up
    to 600 s) for nominal before each stage's bench and runs the bench before the case loop.

## Load-only mode: any other bundle

The same app measures whether the phone loads a bundle of another model at all, and at what cost. Stage the bundles
with no GLiNER2.5-Decide bundle: `DECIDE_KINDS=none DECIDE_EXTRA="<label>=<path>,..." ./_stage.sh` clones each path (a
`.aimodel` or `.aimodelc` directory, as shipped) to `DecideAssets/extra/<label>/`; `DECIDE_EXTRA_JIT="<label>=<path>"`
stages only the JIT files (`main.mlirb`, `main.hash`, `metadata.json`) of a `.aimodel` that also carries AOT files. Then
run with `'"DECIDE_LOAD_ONLY":"<label>,..."'`: one stage `load_<label>` per label in the order given, and no GLiNER
stage.

Each stage loads the bundle with `AIModel(contentsOf:options:)`, preferred gpu, and the function `main`, under the
memory sampler (load 1: wall seconds, peak footprint, least available memory, the container's Core AI cache before and
after); calls `main` once with every input all zeros (skipped, and said so, when an input has a dynamic shape, is not
an array, or the function has states); drops the model and loads it again (load 2). A load that fails records the
error (`error_detail`: type, NSError domain and code, `String(reflecting:)`); a load that kills the app leaves the
stage's partial record in `result.json`, its `step` naming the step that was running, and `./_run.sh` copies the
crash logs. `./_run.sh --summary` prints one line per load-only stage.

`_work` is `conversion/gliner25_decide/_work` (git-ignored). Never load an iPhone bundle (`.h18p`, `.h19p`) on a Mac: the host
refuses one by path.
