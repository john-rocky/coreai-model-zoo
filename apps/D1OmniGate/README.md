# D1OmniGate: the device gate for d1-omni-600M

A headless app that runs the d1-omni-600M port ([`conversion/d1_omni`](../../conversion/d1_omni)) through the Swift host
[`apps/D1Omni`](../D1Omni) on sideloaded assets, over a subset of the fixture the port was gated on, on the iPhone or on
the Mac. It links the library by path, so the decision path is exactly the one an app gets from `D1Omni`: the
publisher's prompt rows, the decision graph of the row's bucket, the vision and audio graphs on media the host decodes
and cuts on the device, the readout. The gate starts at launch with no UI input. It writes `result.json` (rewritten when
a stage starts, after every stage, every 10 rows and every bench series), `result.log` (every line with the thermal
state and the battery) and `memory.tsv` (the footprint every 100 ms from launch to exit, written as taken, so a killed
process leaves its series and a file that stops growing means a frozen app): in `Documents/d1omni_gate/` on the iPhone,
in `D1_OUT` on the Mac.

What it loads (staged by `_stage.sh` into `$D1_WORK/assets/D1OmniAssets/`, default `D1_WORK` = the lane's
`~/code/coreai/_d1_omni/device`):

| path in the assets directory | what |
|---|---|
| `jit/<name>/` | the shipping bundles after `strip_ship.py` (the bytes of the Hugging Face `ios/` folder): `metadata.json` and the `.aimodel` of `fp16-L64`, `fp16-L256` (with `tokenizer/`), `fp16-L2048`, `fp16-L4096`, `vision-fp16` (with `position_table.f32`), `audio-fp16-10s` (with `mel_filters_128x257_f32.bin`) |
| `aot/<name>/<stem>.h19p.aimodelc` | the iPhone 18 Pro's AOT of each (`coreai-build compile --platform iOS --architecture h19p --preferred-compute gpu`, `aot_ship.py --target h19p`) |
| `ane/<name>/<stem>.h19p.aimodelc` | the Neural Engine AOT of `fp16-L256` (52 regions) and `audio-fp16-10s` (64 regions) (`aot_ship.py --target h19p --compute neural-engine`) |
| `fixtures/` | `subset.json` (48 records, 78 rows: 60 text, 9 image, 9 audio; the 18 text rows of at most 64 positions again on L64), `oracle_slim.json` (the publisher's fp32 oracle per row), `mac_ref.json` (the Mac's Swift run of the same graphs, rounds 10 and 12: marker logits and probabilities as float32 bits, and the sha256 of every array of the media path), `bench.json` (W1..W5), `media/` (3 PNG, 3 WAV) — written by `_fixtures.py` from the lane's files |

## Stages

| stage | what it records |
|---|---|
| `assets` | every file in `MD5SUMS` present, md5 of every file up to 16 MB (the model files wait for `md5`), the free space |
| `parity_jit` | per group of graphs that live together (L64 + L256 + vision + audio, then L2048 + vision), a fresh `D1Omni` on the `.aimodel` files (specialized on the device): each graph's load (wall, `AIModel`, `main`, the Core AI cache before and after, the memory), then the rows of the group: ids / markers / bucket against the publisher's, p against the oracle (FACTS §7) and logits / p against the Mac (bits); per image and clip every array of the media path (decoded RGB, crops and their four inputs, samples, mel, masks, each graph output, the prefix) against the Mac's sha256; the rows of at most 64 positions also on L64; the control (each row against the next row of its class) must FAIL. The group's instance is dropped before the next loads |
| `parity_aot` | the same on the h19p `.aimodelc` of each graph (`D1_AOT_DIR`), and every row against `parity_jit`'s |
| `bench` | W1 / W2 (L64 and L256), W4, W5 (`bench.json`): per workload fresh instances of every form (JIT and AOT x the workload's buckets), alternating per round in one process; wait up to `D1_WAIT_NOMINAL` s (60) for the thermal state nominal, then series of at most 18 s with 10 s rests (the 18 Pro's GPU slows after 20 s of back-to-back calls): 3 warm-up rounds in the first series, 1 in each later one, until 20 rounds are timed; a series ends early when the thermal state turns serious, and the next waits in 60 s steps (at most 300 s); every decision with its start offset, ms (media / inputs / graph / readout), thermal, battery and footprint; the outputs checked on every call (JIT = AOT bits per bucket, the oracle) |
| `long` | the graph L4096 apart (its working set is 4-6 GB of footprint on the Mac, over the phone's default memory limit of about 3.5 GB: a jetsam kill there must not take the other stages with it): the 4 long rows on JIT, then on AOT, then W3 with the forms one after the other |
| `ane` | the Neural Engine AOTs (`D1_ANE_DIR`): their regions, load, then aud_01..03's prefix on the ANE audio graph against the GPU's, the 9 audio rows on the GPU decision graph against the oracle, W5 with the ANE audio graph against W5 on the GPU in the same series; the 56 bucket-256 text rows on the ANE decision graph (2 calls each: the bar's values and the drift) and W1 / W2 against the GPU in the same series |
| `md5` | md5 of the model files `assets` left |

A stage writes its start into `result.json` before it runs. A launch that finds `result.json` still `running` records
the stage the previous launch died in (a crash or a jetsam kill) and skips that stage if it is planned again
(`D1_RETRY_DIED=1` overrides). `D1_DEADLINE_S` (seconds after the launch) skips a stage that cannot finish and stops a
loop early (the record says so).

## Signing and the memory limit

The bundle id `com.daisukemajima.d1omnigate` is new, and the only profile of team MFN25KNUGJ that lists the iPhone 18 Pro
and fits a new id is the wildcard one, which cannot carry `com.apple.developer.kernel.increased-memory-limit`
([KevGate](../KevGate/README.md#signing-and-the-memory-limit)). The app has no entitlements and runs at the default
memory limit; it records `os_proc_available_memory` every 100 ms.

## Steps

1. `./_build.sh` (generic iOS, Release) and `./_build.sh --mac` (macOS arm64, Release), each through
   `~/code/standup/tools/quiet/quiet_wait.py --` (a full-CPU build disturbs another lane's Mac timing window). Derived
   data in `$D1_WORK/build/`; the `.app` paths go to `$D1_WORK/app_path.txt` and `app_path_mac.txt`. The script puts
   `../D1Omni/Package.resolved` back if package resolution rewrote it.
2. `./_stage.sh` checks every graph against its `provenance/strip.json` and every AOT against its `aot-manifest.json`,
   runs `_fixtures.py`, clones the assets into `$D1_WORK/assets/D1OmniAssets/` and writes `MD5SUMS` and
   `stage_manifest.json`; `./_stage.sh --mac` stages the Mac's own AOT apart for the dry run.
3. The Mac: `./_run_mac.sh [--ane]` runs the macOS build on the stage directory (JIT from the stage, AOT = the Mac's
   h16c; the app refuses an iPhone AOT on a Mac) and checks that every row's logits and probabilities are bit-equal to
   `mac_ref.json`, every media array equal, the controls FAIL and the bench outputs PASS (`mac_check.json`). It takes no
   lock: run it through `quiet_wait.py --` and the lane's `r7_yield.py`.
4. The phone: `D1_SESSION=<name> D1_ALLOWED_DEVICES=<udid>,<coredevice id> ./_gate.sh <udid>` takes the device hold
   (`~/code/coreai/ondevice/.device_hold`: absent, no keeper pid alive, and no other process naming the phone, for 60 s
   in a row; a JSON hold is never removed while its keeper pid lives), records `devicectl device info details`,
   installs the app, pushes `D1OmniAssets/` to `Library/Application Support/`, checks every file's size from one
   recursive listing (re-pushing what differs; the app md5-checks the bytes), launches the app with a deadline that
   ends the 30 min window in time, polls `result.json` until it is done (a still `memory.tsv` while the process lives =
   frozen: terminated and recorded), launches once more for the stages a killed launch left, and releases the hold on
   any exit. `D1_ALLOWED_DEVICES` is required: the scripts refuse any other phone.
5. Exit codes: 0 = the gate passed, 3 = a stage failed, 1 = no result, 2 = the device is busy, held or refused.
   Reprint a result: `./_run.sh --summary <result.json>`.

The scripts never pass `--remove-existing-content` (it wipes the whole app container) or `--console`, never uninstall.

## Runs (2026-10-08, iPhone 18 Pro, iOS 27.2 24B5099f, on USB power)

The Mac dry run `mac-20261008-205644` (macOS 27.0, the h16c AOT): every one of the 78 rows and the 18 L64 rows
bit-equal to the Mac's Swift runs of rounds 10 and 12 (JIT and AOT), every media array equal, the controls FAIL.

The phone, one hold (21:01:48, 405 s) and a second one for what the first could not do (21:13:51, 86 s); assets pushed
once (8,047,720,825 B in 254 s, 31.7 MB/s, every size equal; the md5 stage then matched every model file on the phone):

| run | stages | result |
|---|---|---|
| `r11-210603` | assets, parity_jit, parity_aot, bench, long | parity PASS on JIT and AOT (74 rows + 18 L64 rows; AOT = JIT bit for bit), bench W1 / W2 (L64, L256), W4, W5; long: the JIT L4096 rows PASS (peak footprint 2,778 MB), then the AOT L4096's first call took the footprint to 3,306 MB and the system killed the app (JetsamEvent, per-process-limit) |
| `r11-210811` | long (skipped: died there), ane, md5 | the Neural Engine AOT `.aimodelc` (h19p, `--preferred-compute neural-engine`) of both graphs: `CoreAIDelegates.AIModelError.failedToSpecialize`; md5 PASS |
| `r11b-211352` | bench W3 (JIT), ane (`D1_ANE_SOURCE=jit,aot`) | W3 on the JIT L4096; the `.aimodel` specialized on the phone with the Neural Engine preferred loads: audio rows PASS, decision rows FAIL (as on the Mac) |

One decision, median of 20 (the forms alternating per round in one process, thermal nominal throughout):

| workload | JIT | AOT h19p |
|---|---:|---:|
| W1, L64 / L256 | 10.79 / 20.12 ms | 10.70 / 20.15 ms |
| W2, L64 / L256 | 32.52 / 60.60 ms | 32.07 / 60.64 ms |
| W3, L4096 | 758.23 ms | killed at the default memory limit |
| W4 (vision 37.3 ms + L256) | 57.67 ms | 57.72 ms |
| W5 (audio 10 s 6.7 ms + L256) | 27.02 ms | 27.30 ms |
| W5 with the audio graph on the Neural Engine (`.aimodel`) | 59.98 ms (audio graph 39.5 ms) | — |

The phone's probabilities are not the Mac's bits (its GPU's arithmetic): max |dp| against the Mac 0.0040, against the
oracle 0.0059 (FACTS §7 PASS). The host's own arrays are: the decoded images, every crop and its four vision inputs,
the samples, the mel and the four masks are bit-equal to the Mac's on all six media items. Numbers and per-row values:
`~/code/coreai/_d1_omni/results/iphone_{parity_jit,parity_aot,bench,ane}.json` (lane directory, not in the repository).
