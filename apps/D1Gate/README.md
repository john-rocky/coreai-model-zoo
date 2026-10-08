# D1Gate: the device gate for d1-3B

A headless app that runs the d1-3B port ([`conversion/d1`](../../conversion/d1)) through the Swift host
[`apps/D1`](../D1) on sideloaded assets, over the fixture the port was gated on (361 text records / 393 questions, the
12 picture records / 24 questions, round 4's 5 red arms), on the iPhone or on the Mac. It links the library by path, so
the decision path is exactly the one an app gets from `D1Decider`: host.py's request checks, text and rows, the option
table, the pictures decoded and cut on the device (`D1Pixels`), the vision tower once per crop, the decoder `main` in its
S = 16 calls from zero states (or the shared prefix), the float64 option readout, the answers. The gate starts at launch
with no UI input. It writes `result.json` (rewritten when a stage starts, after every stage and every 5 records),
`result.log` (every line with the thermal state and the battery) and `memory.tsv` (every memory reading, written as
taken, at least once a second while the app lives, so a killed process leaves its series and a still file means a frozen
app): in `Documents/d1_gate/` on the iPhone, in `D1_OUT` on the Mac. `p_ref_jit.json` / `p_ref_aot.json` beside them
keep every e2e row's p bits, hidden sha256 and row checks across launches.

What it loads (staged by `_stage.sh`):

| path in the assets directory | what |
|---|---|
| `decoder/` | the bundle `d1_3b_decode_int8mlp_pf16`: `metadata.json`, `d1_3b_decode_int8mlp_pf16.aimodel` (`main`, S = 16, the MLP linears int8 per block of 32, final-norm hidden output), `tokenizer/`, `head/option_rows.{json,safetensors}` |
| `tower/` | the bundle `d1_3b_vision_fp16w32`: `metadata.json`, `d1_3b_vision_fp16w32.aimodel` (static, fp16 weights, fp32 compute), `host/position_embedding.safetensors` |
| `aot/d1_3b_decode_int8mlp_pf16.h19p.aimodelc` | the iPhone 18 Pro's AOT of the decoder (`coreai-build compile --platform iOS --architecture h19p --preferred-compute gpu --expect-frequent-reshapes`), pushed apart (`_stage.sh --aot`, `MD5SUMS_AOT`) |
| `fixtures/` | `requests.json` (373 records), `images/` (the 12 PNG files), `oracle_slim.json` (the provider's fp32 oracle per question: row ids, readout groups, keys, probabilities of the question's own row and of the API path, argmax, top-2 margin), `mac_ref.json` (the Mac's read-out of the same decoder asset, AOT h16c: hidden sha256 and p bits per question — round 5c's readout gate for the text, round 8's `decide.py run` for the pictures), `red_arms.json`, `bench.json` |

## Stages

| stage | what it records |
|---|---|
| `assets` | every file in `MD5SUMS` present, md5 of every file up to 16 MB (the model files wait for `md5`), the free space (stops below `D1_MIN_FREE_GB`, default 12) |
| `load_jit` | three steps under the 100 ms memory sampler, the Core AI cache sized around each: the text side (`D1Decider`: tokenizer, option table), the tower alone (`D1Tower` on its `.aimodel`, GPU preferred, no frequent reshapes: its specialization when cold), the decoder (`D1Decider.loadGraph(.jit)`: the bundle's `.aimodel`, GPU preferred with `expectFrequentReshapes`, the tower again from its cache) with the library's split (`AIModel`, `main`, the state allocation, the tower) |
| `load_aot` | the same with the decoder's `D1_AOT` (default the h19p `.aimodelc`) and `SpecializationOptions.default`; the asset's MPSGraph package files and `asset_efr` (a `specialized_model_*` beside the original = compiled with `--expect-frequent-reshapes`) |
| `load_tower_aot` | the tower's AOT (`D1_TOWER_AOT`, default `aot/d1_3b_vision_fp16w32.h19p.aimodelc`) alone under the sampler with the cache sized around it; the crops of `D1_TOWER_RECORD` (default `img01_shapes_384x384`) through it twice (bit-equal re-run) and through the tower's `.aimodel`; per crop the output's sha256 against the Mac's (`mac_ref.json`), max \|d\| / cosine / bit-different values against the JIT's and against the Mac's values (`D1_TOWER_REF`, default `aot/tower_ref/<record>.f32` when staged); the outputs go to `tower_out_<record>_<aot\|jit>.f32` (the Mac run's file is the phone's reference) |
| `probe_noefr` | on the decider in use (the decoder AOT without `--expect-frequent-reshapes`, which specializes once per new position length): `D1_PROBE_RECORD` (`card_refund`) `D1_PROBE_PASSES` (2) times, direct; every call's ms with its position length (a row's call c binds (c + 1) S positions) and whether it was new, the container's bytes written and the memory around each pass, p bits and hidden sha256 against the first pass, the oracle and the Mac |
| `warm` | one decision (`card_refund`): the first calls after the load, scored, not in the bar |
| `red` | round 4's five red arms: every perturbed request against its base record's questions on this graph; an arm is red when an argmax moves on a non-near-tie question, max \|dp\| > 0.02 or the mean of its rows' mean \|dp\| > 0.002; the graph's \|dp\| against the provider's on the same rows |
| `e2e_images` | the picture records: each file decoded and cut here, every crop through the tower, the decoder on the image rows; per question the ids (picture slots mapped back to `<image>`) / groups / keys against the oracle, p against the oracle (both forms) and the Mac's run (p bits, \|dp\|, hidden sha256), per crop the tower output's sha256 and ms; a record of 2+ questions also shared (bit-equal to direct) |
| `e2e_fixture` | the text records, the same; a request the host refuses: its text, then each valid question alone. `D1_SKIP_DONE=1` leaves out the records `p_ref_<kind>.json` already holds (a later launch finishes the fixture); `union_summary` is the bar over every launch's rows |
| `reset` | the first e2e record of the process again (its pictures too): hidden rows and p bit-equal |
| `bench` | per item (`bench.json`: round 7's `one_question`, `three_shared`, `three_direct`, `state_3_4k`, `image_384px`): rest `D1_BENCH_REST` s (60), wait up to `D1_WAIT_NOMINAL` s (300) for the thermal state nominal, 1 warm-up and 5 decisions, `state_3_4k` resting 30 s before each (one decision runs past the 20 s after which the 18 Pro's GPU slows); every decision with its start offset, every call's ms, the tower's ms, thermal, battery, footprint; `latency_ms` = the tower calls and image rows + the graph calls + the readout (decide.py's) |
| `e2e_aot` | on the decider in use: the fixture's first `D1_AOT_LIMIT` (60) text records, scored as e2e and against `p_ref_jit.json` |
| `bench_aot` | `bench.json`'s `bench_aot` items on the decider in use |
| `delete` | `D1_DELETE`: paths under the assets directory, `cache:<hex>` (this app's Core AI cache entries of that hash), or `tmp:<name>` (a directory under the app's tmp, the iPhone only: MPSGraph's scratch) |
| `md5` | md5 of the files `assets` left (`MD5SUMS`), and of every file of any other list in `D1_MD5SUMS` (comma-separated, e.g. `MD5SUMS,MD5SUMS_AOT`) |

The launch line records `os_proc_available_memory` and whether the build carries
`com.apple.developer.kernel.increased-memory-limit`, read from the executable's code signature (`LC_CODE_SIGNATURE`) and
from the embedded profile. The disk guard (iPhone): the volume's free-space reading holds still within a launch, so the gate
keeps the launch's reading and subtracts what the container has written since (`Library/Caches` with the Core AI cache,
`tmp` with MPSGraph's scratch), statfs's reading beside it; below `D1_MIN_FREE_GB` + `D1_DISK_MARGIN_GB` (8) the loops stop
as at the deadline and `result.json` carries `disk_stop`.

A stage writes its start into `result.json` before it runs. A launch that finds `result.json` still `running` records
the stage the previous launch died in (a crash or a jetsam kill) and skips that stage if it is planned again
(`D1_RETRY_DIED=1` overrides): a configuration that died is not retried by accident. `D1_DEADLINE_S` (seconds after the
launch) stops the e2e and bench loops before a record or an item that would start within `D1_RESERVE_S` (40) of it and
skips every later stage but `reset`, `md5` and `delete` (`complete: false` in `result.json`). The bar (the port's,
unchanged: argmax on every question whose oracle top-2 margin is above 0.02, max |dp| <= 0.02, mean of the questions'
mean |dp| <= 0.002, plus the ids and finite hidden rows) is computed in the app and again on the Mac from the p bits in
`result.json`.

## Signing and the memory limit

`D1Gate.entitlements` asks for `com.apple.developer.kernel.increased-memory-limit`; the target takes it only when the build
passes `D1_ENTITLEMENTS_FILE` (`D1_ENTITLED=1 ./_build.sh`, iOS), otherwise the app has no entitlements and runs at the
default memory limit. The wildcard profile of team MFN25KNUGJ (the one that signs a new bundle id without an account) cannot
carry the key: the entitled build fails with `Provisioning profile "iOS Team Provisioning Profile: *" doesn't include the
Increased Memory Limit capability.` The same build with `-allowProvisioningUpdates` (`D1_PROVISIONING_UPDATES=1`) let
Xcode's account add the capability to the explicit App ID `com.daisukemajima.d1gate` and fetch
`iOS Team Provisioning Profile: com.daisukemajima.d1gate` (2026-10-08, the iPhone 18 Pro listed); later builds, with or
without the key, sign with that profile from the Mac. The app records at launch `os_proc_available_memory` and whether its
code signature and embedded profile carry the key, then the memory through every load and every second of the run. On the
iPhone 18 Pro (iOS 27.2) the key moved `os_proc_available_memory` at launch from 3,529 MB to 6,432 MB.

## Steps

1. `./_build.sh` (generic iOS, Release, no entitlements), `D1_ENTITLED=1 ./_build.sh` (iOS with `D1Gate.entitlements`;
   `D1_PROVISIONING_UPDATES=1` adds `-allowProvisioningUpdates`) and `./_build.sh --mac` (macOS arm64, Release), each
   after any Mac measurement window closes (`timing` in `~/code/coreai/_GPU_LOCK`). The `.app` paths go to
   `_work/app_path.txt` (the last iOS build; `app_path_noiml.txt` / `app_path_iml.txt` name each) and
   `_work/app_path_mac.txt`; a failed build keeps its signing lines in `_work/xcodebuild_<tag>_signing_errors.txt`. The
   script puts `../D1/Package.resolved` back if package resolution rewrote it, and compares the md5 of every other file of
   `../D1` before and after.
2. `./_stage.sh` gathers the assets into `_work/device_stage/D1Assets/` (4.4 GB, APFS clones) and writes `MD5SUMS`;
   `./_stage.sh --aot [<x.h19p…aimodelc>...] [--file <src> aot/<rel>]...` stages iPhone AOT assets apart (by default the
   decoder's efr AOT) with `MD5SUMS_AOT` (`D1_AOT_STAGE` names another `_work/device_stage_aot*/D1Assets`). `D1_LANE`
   (default `~/code/coreai/_d1_3b`) is the port's working directory.
3. The Mac: `./_run_mac.sh` runs the macOS build on the stage directory with the Mac's h16c assets (`D1_AOT`, the tower's
   beside its bundle), 20 text records and 3 picture records, only while no measurement window is open — it never takes
   the lock, and a window that opens while it runs stops the app until it closes; `./_run_mac.sh --red` runs the text
   records with one oracle row's probabilities reversed, and the bar must fail. Results land in `_work/mac_runs/`.
4. The phone: `D1_SESSION=<name> D1_ALLOWED_DEVICES=<udid>,<coredevice id> ./_gate.sh <udid> [JSON members]` takes the
   device hold (`~/code/coreai/ondevice/.device_hold`: the file absent, and the last JSON holder's keeper pid gone, for
   60 s; a JSON hold is never removed while its keeper pid lives), installs the app, pushes `D1Assets/` to
   `Library/Application Support/`, pulls it back and re-pushes any file whose md5 differs, launches the app with no
   console and a deadline that ends the window within `D1_FRAME_S` (1,800 s) of the take, polls `result.json`, and
   releases the hold on any exit. With `D1_HOLD_SCRIPT=<name>` the scripts also accept this lane's hold in the JSON form
   other lanes' queue tools respect (`{"pid": <a live keeper>, "script": "<name>", ...}`, written and removed by that
   keeper), for slots under a kept hold. `D1_ALLOWED_DEVICES` is required: the scripts refuse any other phone. Knobs go in as
   JSON members, e.g. `./_gate.sh <udid> '"D1_STAGES":"load_jit,warm,bench,e2e_fixture,reset","D1_SKIP_DONE":"1"'`.
   `D1_SKIP_INSTALL=1` runs again on what is installed; `D1_INSTALL_ONLY=1 D1_SKIP_PUSH=1 D1_SKIP_VERIFY=1` installs the
   app alone (`D1_APP` picks the build) when the assets stayed on the phone (the app's `assets` / `md5` stages then read
   them there). A still `memory.tsv` for `D1_FREEZE_S` (180) s while the app's process lives = frozen: `_run.sh`
   terminates it by pid and keeps what it wrote. Three polls in a row with no file coming back make `_run.sh` look for
   this run's crash report (a crash can leave a second process the pulls trip on), and whatever ended the poll, an app
   process still there is terminated before the window closes.
5. The AOT asset: `D1_INSTALL_ONLY=1 D1_SKIP_APP=1 D1_STAGE_DIR=_work/device_stage_aot/D1Assets
   D1_PUSH_ONLY=aot/d1_3b_decode_int8mlp_pf16.h19p.aimodelc,MD5SUMS_AOT ./_gate.sh <udid>` (sizes checked from the
   phone's listing), then the stages `md5,load_aot,warm,e2e_aot,bench_aot` with `D1_MD5SUMS=MD5SUMS_AOT`, then `delete`
   with `D1_DELETE=aot/d1_3b_decode_int8mlp_pf16.h19p.aimodelc,cache:<its main.hash>`.
6. Exit codes: 0 = the gate passed, 3 = a stage failed, 1 = no result, 2 = the device is busy, held or refused. A run that
   does not end `done` leaves the phone's crash-log listing, and copies of today's D1Gate / jetsam reports, in the run
   directory. Reprint a result: `./_run.sh --summary <result.json>`.

`_work/` is git-ignored. The scripts never pass `--remove-existing-content` (it wipes the whole app container) or
`--console`, never uninstall, and the app refuses to open an iPhone AOT bundle (`.h18p.`, `.h19p.`) on a Mac.

## Runs (2026-10-08)

The Mac (M4 Max, macOS 27.0 26A428, the h16c AOT of the same decoder and the tower): `mac-r8-2` scored 20 text records
and 3 picture records (40 questions) with every hidden row and every p bit equal to the Mac's own references (round 5c's
readout gate, round 8's `decide.py run`), the red arms all red, shared = direct, the reset bit-equal; `mac-r8-2-red`
(one oracle row reversed) failed the bar as it must.

The iPhone 18 Pro (iOS 27.2 24B5099f, h19p, no entitlement: 3,529 MB available at launch): the tower's `.aimodel` loads
(JIT 1.23 s cold, Core AI cache +853 MB, peak footprint 299 MB), the decoder does not. `r8s1-194626`: its `.aimodel`'s
on-device JIT aborts 1.2–1.3 s into the load (`std::bad_alloc` from `operator new` in MPSGraph's
`RuntimeCanonicalizationPass` → `foldTransposeOp` → `BumpMmapResourceAllocator::allocateResource`, SIGABRT).
`r8s3-195423`: the h19p AOT with frequent reshapes (8.74 GB) ends in SIGSEGV in the on-device compile for delegates
(`MPSGraphAICodeCompilerDelegate getInitializedAICodeBytecodeWithPayloadPrefix` ← `CompileForDelegates`), the stack
KevGate recorded for Kev-4B's 15.6 GB AOT. No jetsam; neither configuration was retried. The transfer checks held (the
pull-back md5 of 33 files, the AOT's sizes and the app's md5 of its 8 files).

The iPhone 18 Pro again, with increased-memory-limit (6,432 MB available at launch), on USB power at 80 % charging:

- `r9a1-225551`: the tower's AOT h19p (static, 853 MB) loads in 1.83 s and its crop output equals the tower JIT's bit for
  bit (against the Mac's values max |d| 3.7e-6, cosine 1.0000000000). The decoder's `.aimodel` now specializes on the phone:
  25.3 s at the first load (Core AI cache +4,779 MB, peak footprint 276 MB), 3.6 s warm. All 417 questions (393 text,
  24 pictures) pass the bar against the provider's oracle (argmax 416/416 + near-tie 1/1, max |dp| 0.0180, mean 0.00120;
  against the Mac's p: argmax 417/417, no bit equality, max |dp| 0.0044), the five red arms are red, shared = direct, the
  reset is bit-equal.
- `r9a3-231403` (bench, every timed p equal to the gate's): `one_question` 125.5 ms (thermal fair after the 300 s wait),
  `three_shared` 382.7 ms, `three_direct` 464.5 ms, `state_3_4k` 9,300.7 ms (217 calls), `image_384px` 930.1 ms (the
  tower's crop 421 ms); about 40.5 ms per S = 16 call.
- `r9a2-230611`: the decoder's AOT without frequent reshapes (3.70 GB, the original graph only) loads in 8.3 s, then
  specializes on the phone for every new position length (11.2 / 16.5 / 37.6 / 59.8 s for 16 / 32 / 48 / 64 positions), and a
  call right after such a specialization can return a wrong row (card_refund `team` max |dp| 0.376, right on the second
  pass); 4 of the 5 red arms; the picture records ended in SIGTRAP inside CoreAIRuntime under `D1Decoder.runShared`. Not
  retried; the delete-only launch `r9a2d-231251` removed the AOT assets and their cache entries.
- `r9a4-233426`: round 8's efr AOT (8.74 GB) loads with the key (17.4 s, cache +8,743 MB; without it: the SIGSEGV above), the
  first call after the load 9.7 s, then 55.9 ms per call; the fixture's first 20 questions pass (max |dp| 0.0075), p not
  bit-equal to the JIT's (max |dp| 0.0013).
