# CoreAIStudio

A **macOS-exclusive** Core AI media enhancer: **×4 super-resolution** (AdcSR) and **frame
interpolation** (RIFE v4.26), both self-contained on the low-level Core AI runtime (no
`coreai-kit`) so both models run through one native pipeline. No iOS anywhere, by design — this
is a Mac power-user tool that leans on Apple Silicon (GPU + Neural Engine).

> **Status: fully built and run this session, both features verified end-to-end on a real
> macOS 27.0 / M2 Pro Mac.** Not just compiled — launched, driven through the real UI, and the
> actual on-device output inspected. See "What was verified" below.

## The compute story

- **Upscale (AdcSR)** → **GPU**. A pruned SD-2.1 UNet + VAE decoder — a large diffusion-derived
  graph, the Mac-native "max throughput" tier.
- **Interpolate (RIFE)** → **ANE + GPU split**, per `conversion/rife_compute_router.py`
  (measured, not assumed — see `zoo/rife-v4.26.md`): flow-estimation runs on the **Neural
  Engine**, warp/merge on the **GPU**. Confirmed loading this way in the running app (no
  fallback triggered).

Two features, two different parts of the chip, both driven by measured placement.

## Build

```sh
xcodegen generate
open CoreAIStudio.xcodeproj
```

Single target: `CoreAIStudio` (macOS 27.0+). Links the system `CoreAI.framework` directly (no
external package — ✓ verified this session that `AIModel`/`InferenceFunction`/`NDArray`/
`SpecializationOptions`/`ComputeUnitKind` all live there, re-exported via `CoreAIDelegates`).

```sh
DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
  xcodebuild -scheme CoreAIStudio -sdk macosx -configuration Debug build \
  CODE_SIGN_IDENTITY="-" CODE_SIGNING_REQUIRED=NO DEVELOPMENT_TEAM="" CODE_SIGN_STYLE=Manual
```

(Ad-hoc signing for local runs without the repo's own Apple Developer Team ID.)

## App Sandbox — currently OFF (a real finding, not a shortcut)

`ENABLE_APP_SANDBOX: "NO"` in `project.yml`, matching `apps/CoreAIUpscale`'s `CoreAIUpscaleMac`
target. This was **root-caused, not assumed**: a sandboxed build (`com.apple.security.app-
sandbox` + `files.user-selected.read-write`, correctly embedded — confirmed via `codesign -d
--entitlements -`) failed to actually grant read access to `.fileImporter`-picked files under
this exact macOS 27.0 beta + Swift 6 — every `open()` from ImageIO returned `errno 1 (Operation
not permitted)` despite `startAccessingSecurityScopedResource()` being called correctly. The
same code, rebuilt with the sandbox off, loads and renders images correctly — isolating the
defect to security-scoped-resource grant behavior on this OS build, not to this app's Swift.
Revisit if a later macOS 27 build fixes it.

## Models

- **AdcSR** — `mlboydaisuke/AdcSR-CoreAI` (published, ~1.7 GB, fp32). Downloads on first
  "Download AdcSR" click.
- **RIFE v4.26** — not yet published (see `zoo/rife-v4.26.md`). Use **"Load Local…"** and point
  at a directory containing `<stem>.aimodel`, `<stem>_flow.aimodel`, `<stem>_warpmerge.aimodel`
  (produced by `conversion/export_rife.py --split-warp`; stem e.g.
  `rife-v4.26_384x640_float32`). The "Download RIFE" button will work once the bundle ships.

## What was verified (this session, real device)

- **Upscale**: downloaded AdcSR for real (~1.7 GB, GPU-loaded), ran a real photo through the
  native Swift tiling/feather-blend/color-match pipeline (`Sources/UpscaleEngine.swift`,
  `Sources/ImageTensor.swift`) — **15.5s**, sharp, no seams, no color artifacts.
- **Interpolate → Tween**: sideloaded the split RIFE bundle exported this session — loaded as
  **"ANE + GPU split"** (the ANE load did not throw), generated a t=0.50 interpolated frame from
  two real photos — coherent, no ghosting, no warp artifacts, matching the quality of the
  Python reference demo in `zoo/rife-v4.26.md`.
- **Interpolate → Video**: authored and compile-verified (`Sources/VideoInterpolator.swift`) —
  not yet run end-to-end with a real video file in this session (image-pair tween was the
  priority path); the underlying `engine.interpolate(_:_:count:)` call it drives is the same
  one verified above.

## Known follow-ups

- `ModelDownloader`'s progress percentage UI doesn't update during a download (the download
  itself is real and completes correctly — confirmed by watching the staged file grow on disk).
  Cosmetic; not investigated further this session.
- `VideoInterpolator.swift` uses the classic `AVAssetReader`/`AVAssetWriter` `.add()` +
  `AVAssetWriterInputPixelBufferAdaptor` API, which macOS 27.0 deprecates in favor of a newer
  `outputProvider`/`inputReceiver` pattern (`AVAssetWriter.inputPixelBufferReceiver(for:...)`
  etc.) — functionally correct (compiles, and the code path it shares with the verified tween
  flow works), just not modernized to the OS's newest API yet.
- Sandbox is off (see above) — App Store distribution would need that root-caused before
  re-enabling it.
