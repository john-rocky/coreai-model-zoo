# CoreAIImageGen

A minimal **image-generation** app for Core AI diffusion bundles. FLUX.2 runs on Apple's stock
`CoreAIDiffusionPipeline` ([apple/coreai-models](https://github.com/apple/coreai-models), pinned to
commit `97a14be`); GLM-Image and Z-Image-Turbo run bespoke host loops on the Core AI runtime. One
SwiftUI codebase, two targets:

| Target | Platform | Hosted models | Notes |
|---|---|---|---|
| `CoreAIImageGenMac` | macOS 27 | FLUX.2 klein 4B (int8, fp16), GLM-Image (512, 1024), Z-Image-Turbo (512, 1024) | desktop split-view UI; in-app download |
| `CoreAIImageGen` | iOS 27 | — (load via **Local…**) | builds; not run on a device for the 2026-10-10 port |

## FLUX.2 klein 4B

The Mac catalog offers two folders of
**[mlboydaisuke/FLUX.2-klein-4B-CoreAI](https://huggingface.co/mlboydaisuke/FLUX.2-klein-4B-CoreAI)**,
both exported with apple/coreai-models `97a14be` and pinned to Hub revision `039db98c`:

| Entry | Folder | Files the app downloads | Bytes | Modes |
|---|---|---|---|---|
| FLUX.2 klein 4B int8 (default) | `macos-int8/` | 19 | 7,608,928,899 | Text → Image, Edit |
| FLUX.2 klein 4B fp16 | `macos-fp16/` | 16 | 14,090,986,305 | Text → Image |

int8 is per-block 32 on the text encoder and the transformer; its transformer carries the
image-to-image entrypoints that Edit runs. fp16 has no image-to-image transformer, so the app leaves
out its VAE encoder and does not offer Edit. The half-size VAEs of `macos-int8/` (512 and tiled
decoding) are not downloaded: the Mac app decodes at 1024×1024.

FLUX.2 [klein] is step-distilled: 4 steps at guidance 1.0. Above 1.0 the pipeline adds an unconditional
pass per step and mixes the two (classifier-free guidance), at twice the compute.

**Why FLUX.2 is macOS-only:** the earlier int4 export already overran the per-process memory limit of a
12 GB iPhone (iPhone 17 Pro, traced on 2026-06-15); the 2026-10-10 folders are larger. The iOS app loads
bundles via **Local…**.

## Edit (in-context editing)

Pick a reference image and write an instruction (*"make the bicycle yellow, keep everything else"*).
The output keeps the subject, the layout and the background and applies the instruction — unlike a
strength-based img2img, which re-renders the whole frame.

This is the stock pipeline's FLUX.2 image-to-image (`PipelineConfiguration.startingImage` +
`referenceGrid: .full`): the reference is VAE-encoded to clean latent tokens, appended after the noise
tokens and marked `T = 10` on the RoPE time axis. The output starts from pure noise and attends to the
reference tokens at every step; only the noise tokens are denoised. Mechanism:
[knowledge/flux2-in-context-editing.md](../../knowledge/flux2-in-context-editing.md).

One reference per edit. Any size or aspect ratio works: the app letterboxes the reference into the
model's 1024×1024 square and crops the result back to the reference's aspect ratio.

## What the stock runtime removed (2026-10-10)

Until 2026-10-10 the app pinned the `flux2-in-context-edit` branch of `john-rocky/coreai-models` and ran
the 2026-07-20 int4 files at the Hub repo root, edit transformers included. The port to stock
apple/coreai-models `97a14be` drops three things:

- **2-reference compose** (`Transformer_edit_2ref`) was a feature of that branch. The stock runtime has no
  multi-reference path; Edit takes one reference.
- **Strength-based Image → Image** is gone. The stock FLUX.2 image-to-image is the reference-token method
  above and does not use `strength`, so it is the Edit tab.
- **Stable Diffusion and SD3 bundles** no longer load via Local…: `97a14be` has no
  `StableDiffusionPipeline` or `SD3Pipeline`. Its diffusion pipeline (`FlowTransformerPipeline`) reads
  FLUX.2 and Sana Sprint bundles.

Files the earlier version downloaded stay in `~/Documents/FLUX.2-klein-4B/` (macOS) and can be deleted.

## Measured (2026-10-10)

Mac Studio (M4 Max, 128 GB), macOS 27.0 (26A428), this app's Release build, hands-off runs, one launch per
row, all in one window with no other GPU job. Prompt *"a red bicycle leaning against a blue wooden door on a
narrow cobblestone street, soft morning light, photograph"*, seed 42. **Image** = Generate pressed → image
returned (text encoding, every step, VAE decode). **Load** = Download & Load pressed → Ready, with the files
on disk and the Core AI cache built by an earlier launch. **Peak** = the process's peak memory footprint.

| Model | Mode | Steps | Image (s) | Load (s) | Peak (MiB) |
|---|---|---|---|---|---|
| FLUX.2 klein 4B int8 | Text → Image | 4 | 11.60, 11.59 | 9.5 | 10,950 |
| FLUX.2 klein 4B int8 | Edit (the image above as the reference) | 4 | 24.39, 24.31 | 1.6¹ | 13,481 |
| FLUX.2 klein 4B fp16 | Text → Image | 4 | 10.93 | 18.3 | 8,055 |
| GLM-Image 512 | Text → Image | 20 (guidance 1.5) | 77.48 | 25.4 | — |
| Z-Image-Turbo 512 | Text → Image | 8 | 17.39 | 28.5 | — |

¹ The launch right after the int8 text-to-image one, on the same files.

- The int8 and fp16 images are byte-identical (PNG SHA-256) to the ones coreai-samples' `CoreAIImageGenMac`
  makes from the same folder, prompt and seed: same bundle, same runtime commit, same pixels.
- Edit instruction: *"Make the bicycle bright yellow. Keep the blue door, the street and the light the
  same."* The bicycle turned yellow; the door, the walls, the street and the light stayed.
- Stop, pressed during an 8-step run, finished the step in progress (3 of 8), went back to Ready and did not
  keep the unfinished image.

## Build & run

```bash
brew install xcodegen
cd apps/CoreAIImageGen
xcodegen generate
# macOS:
open CoreAIImageGen.xcodeproj          # run the CoreAIImageGenMac scheme (Release)
# iOS (device):
xcodebuild -project CoreAIImageGen.xcodeproj -scheme CoreAIImageGen -configuration Release \
  -sdk iphoneos -destination 'generic/platform=iOS' -derivedDataPath build-ios \
  -allowProvisioningUpdates build
xcrun devicectl device install app --device <udid> \
  build-ios/Build/Products/Release-iphoneos/CoreAIImageGen.app
```

`project.yml` pulls `apple/coreai-models` straight from GitHub at `97a14be` (no patch stack) and
`swift-transformers` for its Hub downloader.

## Model delivery

On macOS **Download & Load** fetches the picked model and loads it; a model already on disk loads
without a download.

- **FLUX.2**: swift-transformers' `HubApi`, one file at a time, from the pinned revision. A folder counts
  as on disk only when every file has its size at that revision, so the runtime never opens a partial
  bundle; an interrupted download keeps the finished files and the next try fetches the rest. Files land
  under `~/Documents/FLUX.2-klein-4B-CoreAI/`.
- **GLM-Image, Z-Image-Turbo**: the shared [`AppShared/ModelDownloader`](../AppShared/ModelDownloader.swift)
  (range-chunked parallel download, cross-launch resume, atomic placement — a partial bundle never
  poisons the content-keyed coreai-cache) for the `.aimodel` directory bundles + tokenizer; the few tiny
  root files come down with a plain resolve GET (the HF tree API only enumerates directories).

Download, load and generation hold a user-initiated activity, so macOS does not App-Nap the app while
its window is hidden. On iOS the screen is kept awake during the multi-GB transfer; for a big set, stay
on Wi-Fi and keep the app foregrounded.

## Hands-off runs

For checks and timings the Mac app takes launch arguments (`Sources/Autoplay.swift`) and presses the
same controls a person would:

```bash
CoreAIImageGenMac.app/Contents/MacOS/CoreAIImageGenMac -ApplePersistenceIgnoreState YES \
  -autoplay 1 -model int8 -prompt "a red bicycle" -seed 42 -runs 2 -out ~/imagegen-run -quit 1
```

`-model` takes `int8`, `fp16`, `glm512`, `glm1024`, `zimage512` or `zimage1024`; `-reference <image>` runs
Edit with the prompt as the instruction; `-local <folder>` loads a folder instead; `-stopAfterStep <n>`
presses Stop. Each run writes its images and `result-<epoch>.json` (load and generation seconds, image
hashes, peak memory footprint) to `-out`.

## Notes

- On macOS the FLUX.2 models stay loaded between images (`lazyModelLoading: false`); on iOS each stage
  loads on demand and is released after it.
- Stop finishes the denoising step in progress, then returns; the unfinished image is not kept.
