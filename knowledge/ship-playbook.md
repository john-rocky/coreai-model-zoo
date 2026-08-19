# Ship playbook — converted `.aimodel` → CoreAIKit → device → published

The repeatable flow for taking a model from "converted" to "shipped on iPhone, measured, and
published," validated end-to-end on **Parakeet-TDT-0.6B in one session** (convert → gate → Swift
engine → app → device 47.9× real-time → HF → zoo card → post). The per-topic docs go deep on each
stage; this is the **runbook + the cross-cutting traps** that don't live in any single one.

Deep docs per stage: ML convert → [`conversion-guide.md`](conversion-guide.md); target/precision →
[`compute-units-and-authoring.md`](compute-units-and-authoring.md); Swift runtime →
[`swift-runtime.md`](swift-runtime.md); `LanguageModelSession` → [`fm-provider.md`](fm-provider.md);
on-device AOT → [`aot-and-specialization.md`](aot-and-specialization.md).

## Conventions (every session)
- **Mac-GPU is exclusive** — `echo <tag> > ~/code/coreai/_GPU_LOCK` before any python/AOT GPU run,
  `rm` after; run GPU work solo (the beta driver kernel-panics under parallel GPU load).
- **Two venvs** — an *isolated* env for the HF golden (some archs need a newer `transformers` than the
  export env); the *main* `coreai-models/.venv` for export + gating. Don't cross-contaminate.
- **`git add` EXPLICIT paths only** (repos carry unrelated WIP). **Never commit** models/weights/
  `.aimodel`/`.aimodelc`/build files/large `.npz`. **No "claude"** in messages or committer.
- **push / HF upload / social posts are USER-GATED.** Build the iOS app on the device.

## Stages

**1. Convert + gate the ML.** Re-author in plain torch from `model.safetensors`, export with
`export_to_coreai`, gate **token-exact (or per-token cosine)** vs a saved HF golden (`oracle.npz`).
Don't move on until the `.aimodel` matches the golden on GPU. (conversion-guide.md)

**2. Pin host pre/post-processing in NumPy BEFORE writing Swift.** Anything the host computes (mel,
image norm, detok, samplers) — reimplement the *exact algorithm the Swift will run* (e.g. a manual
cos/sin-DFT, not `torch.stft`) in NumPy, gate it **token-exact end-to-end**, and diff vs the golden
features. This is where Parakeet's hidden normalization bug surfaced — see traps. Scripts:
`gate_mel_swift.py` / `mel_swift_sim.py` / `diff_swift_mel.py` in `conversion/parakeet/`.

**3. CoreAIKit engine.** New `Kit<X>Model` in `coreai-kit/Sources/CoreAIKit/<X>/`: load each graph
with `GraphModel(contentsOf:computeUnits:.gpu)`, tokenizer via `AutoTokenizer.from(modelFolder:)`,
host loop in Swift. Mirror the closest sibling's public surface (`KitWhisperModel` / `KitASRModel` /
`VoxCPM2TTS`). Bundle fixed matrices (mel filterbanks) as a target resource. **`swift build
--target CoreAIKit`** to typecheck (catches `await`/`convenience init`/API misuse cheaply).

**4. App wiring.** Add the engine to the view model (enum case + load + run branch). For models with a
big graph, add an **iOS sideload override**: if `Documents/Models/<X>/` holds the bundle, load it
(`init(bundleAt:)`); else Hub-download. The picker/UI usually needs no change.

**5. Headless self-test bench (the perf number).** An env-gated entrypoint (`<X>_SELFTEST=1`, launched
from `App.init()` via `Task.detached`) that: resolves the bundle (sideloaded on device, else local
artifacts on Mac), loads a clip, times **load + N transcribe/generate runs** (run 1 cold, rest warm),
computes RTF (`audio_sec ÷ run_sec`), and writes `Documents/<x>_selftest_result.txt` + NSLog. Run it
on **Mac first** (must be token-exact) before the device.

**6. On-device ship.** Build the app for the device (`xcodebuild -destination 'platform=iOS,id=<UDID>'
-configuration Release`), `devicectl device install app`, `… process launch`. If a **1 GB+ graph's
on-device JIT stalls**, AOT-compile it and sideload (see traps). Run the self-test on device:
```
xcrun devicectl device copy to   --device <UDID> --domain-type appDataContainer \
      --domain-identifier <bid> --source <file> --destination "Documents/Models/<X>/<name>"
xcrun devicectl device process launch --device <UDID> -e '{"<X>_SELFTEST":"1"}' <bid>
xcrun devicectl device copy from --device <UDID> --domain-type appDataContainer \
      --domain-identifier <bid> --source "Documents/<x>_selftest_result.txt" --destination /tmp/r.txt
```

**7. Publish (USER-GATED).** HF upload (`conversion/_<x>_hf_upload.py`: stage → `upload_folder`;
**patch `tokenizer_class`** at stage time). `zoo/<x>.md` (pipeline, graph contracts, on-device speed,
"lessons", convert-yourself) + a `models/README.md` row + a **root `README.md`** row (don't forget this
one). Commit (explicit paths). Draft the X post with the measured RTF — **post is the user's.**

## Cross-cutting traps (each cost real time)
- **Gate before you port — don't trust handoff/kickoff specs.** Parakeet's notes said "no mel
  normalization, pad the mel with a constant"; the real `ParakeetFeatureExtractor` *always*
  per-utterance normalizes (the `do_normalize` arg is dead code) and the bucket must be filled by
  **silence-padding the audio** (zero-padding the mel makes the decoder hallucinate). Caught only
  because step 2 gated the NumPy mel e2e first.
- **1 GB+ graph → AOT, not JIT.** On-device JIT specialization of a big static graph stalls for
  minutes / OOMs. `xcrun coreai-build compile <m>.aimodel --output <dir> --platform iOS
  --architecture h18p --preferred-compute gpu --min-deployment-version 27.0` → `<m>.h18p.aimodelc`
  (embeds a precompiled MPSGraph; ~2× the `.aimodel` size). Arch tracks the **device-identifier major
  version** (iPhone 17 Pro = `iPhone18,1` → `h18p`; M-series Mac → `h16c`), NOT the marketing name.
  Small graphs (≤~50 MB) JIT fine — ship them as portable `.aimodel`. (aot-and-specialization.md)
- **swift-transformers rejects unknown `tokenizer_class`** (`unsupportedTokenizer`). Retag the
  bundle's `tokenizer_config.json` to a registered class (`PreTrainedTokenizer` → `BPETokenizer`);
  decode is driven by `tokenizer.json`'s decoder, so it stays exact. Do it in the upload script.
- **Release device build re-clones packages from a fresh derived-data dir** (MisakiSwift pulls
  `mlx-swift` whose `mlx` submodule clone looks "stuck" — 0% CPU on `git index-pack`). **Reuse the
  Debug build's `-derivedDataPath`** so packages are already resolved.
- **`devicectl` facts:** `install` preserves the app data container (sideloaded files survive a
  reinstall); env vars need the `-e '{"K":"V"}'` JSON flag; copy individual files (bulk can
  false-succeed). `GraphModel`/`AIModel(contentsOf:)` load `.aimodel` *or* `.aimodelc`.
- **A crash with no log is usually the log's fault.** `print` under `devicectl … --console` goes to a
  block-buffered stdout, and the buffer dies with the process — so a SIGSEGV inside a load call reads
  as a crash with zero output. Write diagnostics with `fputs(…, stderr)` + `fflush`: one line before
  the suspect call localizes it. Piping `--console` through `tail` hides everything until EOF too;
  use `head` or redirect to a file.
- **`devicectl` must come from the Xcode whose SDK matches the device OS.** After an iOS update the
  stable Xcode's `devicectl` hung with no output and no error while the phone was healthy;
  `DEVELOPER_DIR=/Applications/Xcode-<beta>.app/…/Developer` fixed it immediately. `list devices`
  also returns **simulators with the same marketing name** — select on
  `connectionProperties.transportType` (`wired`/`localNetwork`), never on the model string.
- **There is no simulator path for a Core AI app.** The simulator SDK ships no `CoreAI` framework, so
  any simulator destination fails at `Unable to resolve module dependency: 'CoreAI'`. To see a screen
  without a device, render the SwiftUI view offscreen with `ImageRenderer` at the phone's point size.
- **A single on-device run measures specialization, not throughput.** The first call after load
  specializes the graphs: Parakeet v2 on an iPhone 17 Pro measured **7.2× realtime on the first pass
  and 67.8× on the next**, same clip, same process. Anything that shows one number — a demo app as
  much as a bench — must discard a warmup pass or it reports a figure no workload ever sees.
- **`Bundle.module` / cross-file symbols show as SourceKit errors in-editor** until a real build
  regenerates the resource accessor — `swift build` is the source of truth, not the squiggles.
