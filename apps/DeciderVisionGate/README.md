# DeciderVisionGate: the device gate for decider-2b-vision

A headless app that runs the decider-2b-vision port ([`conversion/decider_vision`](../../conversion/decider_vision))
through the Swift read-out [`apps/DeciderVision`](../DeciderVision) on sideloaded JIT `.aimodel` assets, over the round-1
fixture the port was gated on (35 image rows at g256 and g448, 6 text rows: 76 runs, 108 answer slots), on the iPhone or
on the Mac. It links the library by path, so the decision path is exactly the one an app gets from `VisionDecider`:
ImageIO decode, Pillow-order bicubic resize, the fp16w32 vision tower, the author's prompt with swift-transformers'
tokenizer, the slot rule, the 2-function decoder in its chunk order, the letter softmax. The gate starts at launch with
no UI input. It writes `result.json` (rewritten after every stage and every 5 runs), `result.log` and `memory.tsv`
(every 100 ms memory reading, written as taken, so a killed process leaves its series): in `Documents/dv_gate/` on the
iPhone, in `DV_OUT` on the Mac. `dump/` beside them holds each run's full fp16 slot logits and tower output.

What it loads (staged by `_stage.sh`, the shipped set):

| path in the assets directory | what |
|---|---|
| `decoder/` | the decoder LanguageBundle: `metadata.json`, `decider_2b_vision_decode_int8mix_pf16.aimodel` (functions `main` S=1 and `prefill` S=16, int8 per-block-32 linears with layers 0 / 2 / 5 fp16), `tokenizer/` |
| `towers/decider_2b_vision_{g256,g448}_vision_fp16w32.aimodel/` | the fixed-grid vision towers, fp16 weights and fp32 compute: patches f32 `[4 G², 1536]` -> image_embeds f32 `[G², 2048]`, G = 8 / 14 |
| `fixtures/` | `rows.json`, `meta.json`, `images/`, `oracle_slim.json` (the author's fp32 oracle per run: ids, slots, probabilities, argmax), `mac_ref.json` (the Mac's Swift JIT read-out of the same assets, round 8, with the sha256 of each run's slot logits) |

All three graphs are JIT `.aimodel` bundles: the phone specializes them at the first load and caches the result
(`Library/Caches/coreai-cache/<OS build>/…`: 6.71 GB for the three graphs on the iPhone 18 Pro, run `20260929-080203`).

## Stages

The library fixes a `VisionDecider`'s towers when it is made, so each grid gets its own decider and the previous one is
dropped first: the peak is the decoder plus one tower.

| stage | what it records |
|---|---|
| `assets` | every file in `MD5SUMS` present, md5 of every file up to 16 MB (the model files wait for `md5`, so the cold load does not start on a warm file cache) |
| `load1` | stops first when the volume has less than `DV_MIN_FREE_GB` (default 8) free. Then (a) the g256 tower alone (its cold specialization), dropped; (b) `VisionDecider` with no tower = the tokenizer and the decoder (its cold specialization), dropped; (c) `VisionDecider` with the g256 tower, both now cached = the decider of the g256 phase. Each step under the 100 ms memory sampler (peak footprint, least `os_proc_available_memory`, the whole series), the Core AI cache sized between the steps |
| `warmup` | one g256 decision (`DV_WARMUP`, default r01): the first call after the load |
| `e2e_g256` | the 35 image rows at g256: per run the ids, slots, letter logits (fp16 bit patterns), probabilities, argmax and full-vocabulary top-1 against the oracle and the Mac, the sha256 of the decoded / resized pixels, patches, tower output and slot logits, every step's time; thermal, battery, footprint and headroom every 20 s |
| `reset_g256` | the phase's first run again: its full slot logits bit-equal (the states are zeroed per row) |
| `bench_g256` | `DV_BENCH_G256` (default r23, a 256x240 frame): rest `DV_BENCH_REST` s (default 60), wait up to `DV_WAIT_NOMINAL` s (default 300) for the thermal state nominal, then 1 warm-up and `DV_BENCH_RUNS` (default 5) decisions back to back, each with its start offset (the iPhone 18 Pro's GPU slows after about 20 s of back-to-back work) |
| `load_g448` | the g256 decider dropped, `VisionDecider` with the g448 tower: the decoder's warm reload and the g448 tower's cold load, apart |
| `e2e_g448`, `e2e_text` | the 35 image rows at g448, then the 6 text rows on the same decider |
| `reset_g448` | the phase's first run (the first g448 run) again, bit-equal |
| `bench_g448` | `DV_BENCH_G448` (default r23) at g448, then `DV_BENCH_TEXT` (default t06), as `bench_g256` |
| `load2` | the decider dropped and the g256 decider made again in the same process (all warm), then the warm-up row once more: bit-equal to its e2e run |
| `md5` | md5 of the model files |
| `load_aot` | not in the default list: `DV_DECODER_AOT` (an iPhone AOT `.aimodelc` staged under `aot/`) with `SpecializationOptions.default`, then one text decision |

The decoder is specialized GPU-preferred with frequent reshapes (its position ramp and the prefill width change per call),
the towers GPU-preferred: the Mac CLI's `--asset jit` options. The app passes when every stage passes; the bar itself (the
round-4 bar on all 108 slots) is scored on the Mac by
[`conversion/decider_vision/gate_iphone.py`](../../conversion/decider_vision/gate_iphone.py), which recomputes every
probability from the fp16 letter logits and compares the dumps with the Mac's.

## Signing and the memory limit

The target asks for `com.apple.developer.kernel.increased-memory-limit` (`DeciderVisionGate.entitlements`, iOS only).
Without it the decoder's cold JIT specialization aborts on the iPhone 18 Pro: run `20260929-034405` (iOS 27.0 24A437,
default limit: `os_proc_available_memory` 3.53 GB at launch) died 5 s into load1 (b) with `std::bad_alloc` thrown from
MPSGraph's `BumpMmapResourceAllocator::allocateResource` while `CanonicalizeMatMulTransposeConstantRHS` folded a
transposed constant (SIGABRT, thread `MPSGraphExecutable_queue`). The 100 ms sampler's last reading (5.0 s into the
step; the crash report was captured 6.0 s after launch) was 345 MB of footprint and 3.20 GB available; the size of the
request that failed is not in the report. The same failure of 2B-class cold specializations at the default limit is
recorded in `knowledge/pipelined-engine.md`.

The bundle id `com.daisukemajima.decidervisiongate` is new, and the only profile of team MFN25KNUGJ that lists the
iPhone 18 Pro and fits a new id is the wildcard one, which cannot carry the entitlement: `./_build.sh` fails with
"No Accounts: Add a new account in Accounts settings" and "Provisioning profile "iOS Team Provisioning Profile: *"
doesn't include the Increased Memory Limit capability" until the App ID has the capability. To register it once: open
`DeciderVisionGate.xcodeproj` (generated by `xcodegen generate` or `./_build.sh`) in Xcode with the team's account
signed in and build it for the iPhone (⌘B); after that `./_build.sh` signs with the downloaded profile.
`./_build.sh --no-iml` builds without the entitlement (the wildcard profile): the build the stopped run used. With the
entitlement (runs `20260929-080203` cold and `20260929-081408` warm) `os_proc_available_memory` read 6.43 GB at launch,
the decoder's cold specialization took 21.6 s with a peak footprint of 417 MB, and every stage passed.

## Steps

1. `./_build.sh` (generic iOS, Release; `--no-iml` without the entitlement) and `./_build.sh --mac` (macOS arm64,
   Release). The `.app` paths go to `_work/app_path.txt` and `_work/app_path_mac.txt`. The script puts `../DeciderVision/Package.resolved` back if the
   package resolution rewrote it.
2. `./_stage.sh` gathers the assets into `_work/device_stage/DeciderVisionAssets/` (3.7 GB, APFS clones) and writes
   `MD5SUMS`. `DV_LANE` (default `~/code/coreai/_decider2bv`) is the port's working directory.
3. The Mac: `./_run_mac.sh` runs the macOS build on the stage directory under the machine-wide GPU lock
   (`~/code/coreai/_GPU_LOCK`). Results land in `_work/mac_runs/<run id>/`.
4. The phone: `DV_SESSION=<name> DV_ALLOWED_DEVICES=<udid>,<coredevice id> ./_gate.sh <udid>` takes the device hold
   (`~/code/coreai/ondevice/.device_hold`; while another session holds it, waits up to 60 min; a JSON hold is never removed
   while its keeper pid lives), installs the app, pushes `DeciderVisionAssets/` to `Library/Application Support/`, pulls it
   back and re-pushes any file whose md5 differs, launches the app with no console, polls `result.json` until it is done,
   pulls `dump/`, and releases the hold on any exit. `DV_ALLOWED_DEVICES` is required: the scripts refuse any other phone.
5. A cold first load: `DV_FRESH=1 ./_gate.sh <udid>` uninstalls first (the data container goes with the app, and the
   container's Core AI cache with it). A warm rerun on what is installed: `DV_SKIP_INSTALL=1 ./_gate.sh <udid>`. Knobs go
   in as JSON members, e.g. `./_gate.sh <udid> '"DV_STAGES":"assets,load1"'` (all of them are listed at the top of
   `Sources/GateRunner.swift`). The poll cap is `DV_CAP` polls of 10 s (default 270).
6. Exit codes: 0 = the gate passed, 3 = a stage failed, 1 = no result, 2 = the device is busy, held or refused. A run that
   does not end `done` leaves the phone's crash-log listing, and copies of today's DeciderVisionGate / jetsam reports, in
   the run directory. Reprint a result: `./_run.sh --summary <result.json>`.

`_work/` is git-ignored. The scripts never pass `--remove-existing-content` (it wipes the whole app container) or
`--console`, and the app refuses to open an iPhone AOT bundle (`.h18p.`, `.h19p.`) on a Mac.
