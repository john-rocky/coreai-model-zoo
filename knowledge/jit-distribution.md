# JIT distribution: one `.aimodel` serves every iPhone generation

Measured 2026-09-26 on an iPhone 18 Pro (iPhone19,2, iOS 27.0 24A437, Core AI architecture `h19p`) with
`apps/DecideGate` in its load-only mode (`DECIDE_LOAD_ONLY`: `AIModel(contentsOf:options:)` with the GPU preferred, the
function `main`, one call on all-zero inputs, then a second load; no memory entitlement; thermal state nominal
throughout). The bundles are the shipped Hub bytes (for each JIT bundle, `main.hash` equals the SHA-256 of its
`main.mlirb`; an AOT package's `main.hash` is that of its `main-<arch>.mlirb`).

## The question

An AOT bundle (`.aimodelc`, `xcrun coreai-build compile --platform iOS --architecture <arch>`) runs on one device
architecture only: `h18p` is the iPhone 17 Pro, `h19p` the 18 Pro, and a mismatch is refused at load
(`incompatibleCompiledAssetArchitecture(device: "h19p", asset: ["h18p"])`, no fallback, no on-device compile —
[coreai-error-index.md](coreai-error-index.md)). Shipping AOT therefore means one artifact per device generation
and a re-compile whenever a new phone appears. The JIT form (`.aimodel`: `main.mlirb` + `main.hash` +
`metadata.json`) is one artifact for every device, at the price of an on-device specialization on the first load.
The question was how large a graph the phone specializes itself, and what that first load costs.

## Measured

| bundle (Hub bytes) | size | first load after install | app footprint peak | first call | load on relaunch | call on relaunch | cache written | result |
|---|---|---|---|---|---|---|---|---|
| Nemotron-3-Diarization `n3d_streaming_float16.aimodel` | 198 MB | 0.78 s | 64 MB | 1.16 s | 0.16 s | 79 ms | +198 MB | loads |
| Laya-Multilingual `ios/wfp16-s256/…aimodel` (already shipped as JIT) | 645 MB | 0.54 s | 64 MB | 1.08 s | 0.50 s | 72 ms | +645 MB | loads |
| GLiNER2-PII, the JIT files of its `ios/` bundle alone | 611 MB | 0.75 s | 92 MB | 1.42 s | (not taken) | — | +639 MB | loads |
| whisper-large-v3-turbo `macos/…fixed128.aimodel` | 1,617 MB | 4.37 s | 78 MB | 2.11 s | 0.26 s | 317 ms | +1,750 MB | loads |
| Nemotron-3-Diarization `n3d_streaming_float16.h18p.aimodelc` (the iOS form the kit picked) | 198 MB | refused in 2 ms | — | — | — | — | 0 | `incompatibleCompiledAssetArchitecture` |
| GLiNER2-PII `ios/…aimodel` as shipped (JIT files + `main-h18p.mlirb` + `main-h18p-delegates/` + `stats.json`) | 1,474 MB | refused in 2 ms | — | — | — | — | 0 | `incompatibleCompiledAssetArchitecture` |

Runs `20260926-162943` (fresh install: the app and its container removed first, so no container cache),
`163212` (relaunch), `163809` (the PII JIT files, then the shipped PII bundle right after); raw under
`conversion/gliner25_decide/_work/device_runs/` and `~/code/coreai/_jit_distribution/results/` (lane data).

## What the numbers say

- **A 1.6 GB static graph specializes on the phone.** The whisper encoder-decoder (fp16, fixed 128-token window)
  took 4.4 s on the first load and 2.1 s on the first call, then 0.3 s per load on every relaunch. Nothing crashed.
  The kit's comment that "the on-device JIT aborts on the 1.6 GB graph" (`ModelID.whisperLargeV3Turbo`) does not
  hold on the 18 Pro; it was never recorded as a measurement in this repository. The known on-device JIT failures
  ([coreai-error-index.md](coreai-error-index.md): `std::bad_alloc` on SAM 3, `LLVM ERROR: Failed to allocate
  mmap'd buffer` on the Gemma 4 E2B decoder) are the ~2 GB-of-constants decoders and a dynamic-shape segmenter;
  the LLM-class bundles keep their AOT form ([aot-and-specialization.md](aot-and-specialization.md), the 4B wall).
- **The app's own memory barely moves while the phone specializes.** The app's `phys_footprint` peaked at
  64–92 MB during the loads (whisper: 78 MB, then 177 MB on the first call) and `os_proc_available_memory` never
  fell below 3.4 GB. Where the compile's memory is accounted (another process, or memory the footprint does not
  count) was not isolated; what is measured is that the app did not approach its own limit. The result lands in
  the app container's Core AI cache (`Library/Caches/coreai-cache/<OS build>/…`) at about the size of the IR
  (whisper: 1.75 GB for a 1.62 GB `main.mlirb`), keyed by `main.hash`, and the next launch read it.
- **The first load pays once per install and OS build** (one relaunch measured; the cache path carries the OS
  build, so an OS update should compile again — not measured). With the cache present, `AIModel(contentsOf:)`
  returns in milliseconds and the load time moves into `loadFunction(named:)` (0.16–0.50 s here); the first call
  drops from 1.1–2.1 s to 72–317 ms.
- **A directory that carries the JIT files next to another architecture's AOT files is not a fallback.** The
  shipped GLiNER2-PII `ios/` bundle (JIT IR plus `h18p` delegates, the layout `coreai-build compile` leaves when it
  writes into the `.aimodel` directory) is refused on the 18 Pro exactly like a pure `h18p` `.aimodelc` — the same
  error in 2 ms, so the IR beside the delegates is not used as a fallback (the runtime's internal order is not
  observed, only that outcome). It loaded only when the container already
  held the specialization of the same `main.hash` (0.08 s, right after the JIT-only copy had been loaded). This
  corrects [gliner25-decide.md](gliner25-decide.md) §7 arm C, whose "the runtime ignored the delegates" was that
  warm-cache case — arm B had written the cache minutes earlier under the same hash.

## The distribution rule this repository follows from here

1. **Non-LLM graphs (encoders, vision, audio, TTS parts, classifiers with static shapes) ship as `.aimodel` only.**
   The `ios/` subtree holds the same JIT IR as `macos/` (the Hub stores one copy of identical content), so the
   kit's platform default (`resolvedPath` = `ios` on iPhone) keeps working and every generation of phone
   specializes the graph itself on its first load. Laya-Multilingual has shipped this way since 2026-09-23.
2. **AOT stays for the LLM-class bundles** (the pipelined decoders, ~2 GB of constants and up, dynamic shapes),
   one `ios-<arch>/` subtree per device generation, never inside `ios/`.
3. **An AOT bundle that still exists for a non-LLM graph moves to `ios-<arch>/`** (`ios-h18p/…h18p.aimodelc`), so a
   caller who has measured that the AOT form is worth its second on a given phone can pin it by path; it is not the
   default.
4. **A loader picks the AOT form only when its architecture is the device's** (`AIModel.deviceArchitectureName`,
   the `<arch>` in `<name>.<arch>.aimodelc`); otherwise the `.aimodel`. "AOT wins over JIT when both are present"
   is wrong on a phone of another generation.
5. **Cards and catalogs name the JIT form for iOS**; an `ios` variant that points at a `.h18p.aimodelc` promises a
   bundle that loads on one phone generation only.

Which shipped repositories were in the AOT-only iOS form on 2026-09-26 is read from `models/index.json` (`bundles`
containing `.h18p.aimodelc` or `ios-h18p/`); the fix per repository is the layout above.

## Not measured

The iPhone 17 Pro (`h18p`) was not available; the on-device JIT of the whisper-size graph on that phone, and
whether an AOT bundle buys back more than the ~1 s it does on the 18 Pro there, are open. The Mac is not the
question: it has always JIT-compiled these graphs.
