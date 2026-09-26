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

### The rest of the zoo's non-LLM iOS graphs, same phone, same mode (runs `20260926-185357`, `185441`, `191440`, `191523`)

Every JIT graph that replaced an h18p bundle in `ios/` on 2026-09-26, plus the two repositories still to switch,
loaded and ran once on the iPhone 18 Pro from a fresh container; nothing crashed and the thermal state stayed
nominal. One measurement each; the graphs that hold state (the VoxCPM / VoxCPM2 / VibeVoice language-model
graphs) were loaded only.

| repository (JIT graph) | size | first load | first call | load on relaunch |
|---|---|---|---|---|
| TimesFM-2.5-200M | 463 MB | 1.54 s | 0.75 s | 0.44 s |
| VJEPA2-ViTL-SSv2 | 708 MB | 2.22 s | 1.19 s (app peak 244 MB) | 0.56 s |
| Nemotron-3.5-ASR-Streaming conformer a / b | 605 / 615 MB | 1.37 / 1.44 s | 0.93 / 0.32 s | 0.44 / 0.46 s |
| VibeVoice-Realtime-0.5B decoder / ttslm / mainlm / head / connector | 688 / 597 / 120 / 84 / 2 MB | 1.78 / 1.36 / 0.20 / 0.17 / 0.04 s | 1.18 s / state / state / 0.42 s / 0.09 s | 0.08 / 0.50 / 0.07 / 0.06 / 0.01 s |
| VoxCPM-0.5B base decode / prefill / feat decoder / feat encoder / res ×2 / vocoder | 360 / 360 / 129 / 122 / 90 / 54 MB | 1.13 / 1.19 / 1.06 / 0.17 / 0.19 / 0.09 s | state / state / 0.76 s / 0.18 s / state / 0.18 s | 0.27 / 0.27 / 0.10 / 0.08 / 0.06 / 0.04 s |
| VoxCPM2 base prefill / decode (int8 static LM, 1.32 GB each) / feat decoder / feat encoder / res ×2 / vocoder | 1,324 / 1,324 / 426 / 420 / 378 / 92 MB | 3.49 / 3.20 / 5.64 / 1.17 / 0.79 / 0.16 s | state / state / 1.74 s / 0.65 s / state / 0.68 s | 1.25 / 0.97 / 0.36 / 0.30 / 0.31 / 0.06 s |
| Granite-Embedding-97M `macos/` graphs fp32 s128 / s512, w8 s128 / s512 | 390 / 390 / 305 / 306 MB | 1.01 / 0.40 / 0.63 / 0.33 s | 0.62 / 0.09 / 0.13 / 0.09 s | 0.26 / 0.26 / 0.21 / 0.21 s |

The Granite graphs are the macOS export; its iPhone gate had run on a separate iOS export compiled to h18p, so
these four are load-and-call checked on the phone, not re-gated for embedding parity. The two 1.32 GB VoxCPM2
language-model graphs are the largest int8 static graphs the phone has specialized here.

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
3. **An AOT bundle that still exists for a non-LLM graph moves to `ios-<arch>/`** (`ios-h18p/…h18p.aimodelc`), as a
   complete subtree: the same relative names as `ios/` (a loader that opens a fixed name, such as
   `KitWhisperModel`'s `…fixed128.aimodel`, must find it there too) plus the tokenizer, manifests and host files
   beside it. Since CoreAIKit PR #59 (2026-09-26) an iPhone whose `AIModel.deviceArchitectureName` matches an
   `ios-<arch>/` on the Hub downloads that subtree by default and falls back to `ios/`; a kit built before it, and
   any other phone, take `ios/`.
4. **A loader picks the AOT form only when its architecture is the device's** (`AIModel.deviceArchitectureName`,
   the `<arch>` in `<name>.<arch>.aimodelc`); otherwise the `.aimodel`. "AOT wins over JIT when both are present"
   is wrong on a phone of another generation.
5. **Cards and catalogs name the JIT form for iOS**; an `ios` variant that points at a `.h18p.aimodelc` promises a
   bundle that loads on one phone generation only.

Which shipped repositories were in the AOT-only iOS form on 2026-09-26 is read from `models/index.json` (`bundles`
containing `.h18p.aimodelc` or `ios-h18p/`) plus the Hub listing of every `ios/*.aimodel`: a directory named
`.aimodel` can hold `main-h18p.mlirb` + `main-h18p-delegates/` and no `main.mlirb` at all (whisper-large-v3-turbo's
`ios/`, 3.2 GB), which is the AOT form under a JIT name and refused the same way. The fix per repository is the
layout above.

## Not measured

The iPhone 17 Pro (`h18p`) was not available; the on-device JIT of the whisper-size graph on that phone, and
whether an AOT bundle buys back more than the ~1 s it does on the 18 Pro there, are open. The Mac is not the
question: it has always JIT-compiled these graphs.
