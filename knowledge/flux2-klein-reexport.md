# FLUX.2 klein 4B on Apple's stock exporter: 15 forms against an fp32 gate

> 2026-10-10. The July 2026 export of `mlboydaisuke/FLUX.2-klein-4B-CoreAI` stopped loading on apple/coreai-models
> 97a14be ("unsupported metadata_version '0.1'"). This note records the re-export with that commit's own exporter
> (`uv run coreai.diffusion.export flux2-klein-4b --platform macOS`, no zoo code): every exporter lever, a
> per-component gate against fp32 PyTorch, and the two forms that shipped. Card:
> [`models/flux2-klein/README.md`](../models/flux2-klein/README.md). Apps:
> [CoreAIImageGenMac](https://github.com/john-rocky/coreai-samples/tree/main/CoreAIImageGenMac) (coreai-samples) and
> [`apps/CoreAIImageGen`](../apps/CoreAIImageGen). Mac Studio M4 Max (128 GB), macOS 27.0 26A428, Xcode 27.0
> 27A266a; apple/coreai-models 97a14be, coreai-torch 0.4.3, coreai-core 1.0.0b3, coreai-opt 0.3.0, torch 2.9.0,
> diffusers 0.37.1. Source weights: `black-forest-labs/FLUX.2-klein-4B` at `e7b7dc27`. The gate scripts and window
> logs are not in this repository.

## What was measured

Fifteen forms of one model: twelve exports, an AOT compile, a 16-file subset, and the Neural Engine. Each export got
its own `--output-dir`. A component gate compared each export against fp32 PyTorch. The sample app timed every form
it could open: 1024×1024, 4 steps, guidance 1.0, seed 42, two prompts.

"Image" is the app's own clock: prompt encoding, 4 steps and the VAE decode. It is the median of 4 runs after a
relaunch (2 per prompt). A run counted only if no other GPU job ran during it ("Timing on a shared machine" below).

## Levers and results

| form | export flags (all with `--platform macOS`) | image (s) | first load (s) | load after relaunch (s) | peak (GiB) | gate (i) / (ii) / (iii) | gate |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | no flags: the registry default, int4 per-block 32, multi-function | 11.02 | 6.70 | 1.13 | 10.7–10.8 | 0.65–0.75 / 0.9675 / 65.5 dB | FAIL |
| 2 | `--compression none` | 11.06 | 242.7 | 91.4–126.5 | 12.0–12.2 | 0.99966–0.99995 / 0.999987 / 65.5 dB | PASS |
| 3 | `--compression 8bit` (int8 per channel) | 14.88 | 19.5 | 1.55 | 10.7–10.8 | 0.981–0.987 / 0.99955 / 65.5 dB | FAIL |
| 4 | `--single-function --resolution 1024` | 11.02 | 9.93 | 0.97 | 7.8 | = form 1 (bit-equal outputs) | FAIL |
| 5 | form 4 + `--low-memory` | 11.03 | 9.38 | 0.93 | 7.7–7.8 | = form 1 | FAIL |
| 6 | `--compute-precision float32` | 16.36 | 14.7 | 1.4–1.5 | 21.0–23.3 | 0.65–0.75 / 0.9661 / 95.3 dB | FAIL |
| 8 | form 1, only the 16 files the app downloads | 11.03 | 13.3 | 1.12 | 9.5–9.6 | = form 1 | FAIL |
| 10 | `--compression 4bit-asym` | 11.30 | 7.15 | 1.22–1.27 | 13.4–13.6 | 0.68–0.78 / 0.9742 / 65.5 dB | FAIL |
| 11 | `--compute-precision bfloat16` | 11.19 | 13.4 | 1.18 | 11.3–11.5 | 0.66–0.76 / 0.9692 / 56.6 dB | FAIL |
| 13 | JSON: text encoder unquantized, transformer int8 per channel | 14.80 | 28.0 | 1.9 | 10.6–10.7 | 0.99966–0.99995 / 0.99955 / 65.5 dB | PASS |
| 14 | JSON: text encoder unquantized, transformer int8 per-block 32 | 11.59 | 11.2 ¹ | 1.87 | 10.7–10.8 | 0.99966–0.99995 / 0.99981 / 65.5 dB | PASS |
| **15** | JSON: int8 per-block 32 on both (`macos-int8/`) | **11.60** | 20.4 | 1.7–1.8 | 10.8–10.9 | 0.9928–0.9948 / 0.99981 / 65.5 dB | PASS (0.99) |
| **16** | `--compression none --single-function --resolution 1024` (`macos-fp16/`) | **10.88** | 27.9 | 2.14–2.16 | 9.2–9.4 | 0.99966–0.99995 / 0.999987 / 65.5 dB | PASS (0.999) |

¹ Its files were in the page cache from a discarded run 70 s earlier.

Forms 16 and 15 shipped. Form 16 is the fastest form that passes. Form 15 is the only passing form under an 8 GB
download: its text-to-image files are 7.54 GB, form 16's 14.09 GB. Form 7 (AOT) and form 9 (the Neural Engine) have
no app row; their sections below say why.

One Transformer call (the `main` function, 1024², first-step inputs), median of 5 after a warm-up:

| how | ms |
| --- | ---: |
| GPU, JIT `.aimodel` (form 1) | 2,485.2 |
| GPU, AOT h16c `.aimodelc` (form 7) | 2,552.4 (+2.7 %) |
| `.neuralEngine` requested (form 1) | 2,489.2 (it ran on the GPU) |
| `SpecializationOptions.cpuOnly` (form 4) | 7,816.4 (median of 3) |

## The gate: fp32 PyTorch, one component at a time

The reference is diffusers' `Flux2KleinPipeline` in float32 on MPS, run through apple/coreai-models' own export
wrappers (`coreai_models.diffusion.flux2`). Its floor against the same code on the CPU: text-encoder cosine
0.9999999996, transformer 0.99999999996, VAE 92.5 dB.

The inputs are built the way the Swift pipeline builds them: the chat template with `enable_thinking=False`, padding
to 512 tokens with `<|endoftext|>`, `torch.randn((1, 128, 64, 64))` from seed 42 packed to [1, 4096, 128], and the
position ids `img_ids` [0, h, w, 0] and `txt_ids` [0, 0, 0, s].

| check | what is compared | bar |
| --- | --- | --- |
| (i) text encoder | cosine of each of the 512 token vectors, averaged per prompt, 4 prompts | ≥ 0.999 unquantized, ≥ 0.99 quantized |
| (ii) transformer | cosine of the first step's output [1, 4096, 128] | same |
| (iii) VAE decoder | PSNR of the 8-bit image decoded from the reference's final latent | ≥ 35 dB |
| (iv) end to end | PSNR of the app's seed-42 image against the fp32 diffusers image | recorded, no bar |

- **The bar sits on the mean, not the minimum.** Even the unquantized form has a padding token at cosine 0.867 in the
  fourth prompt. A bar on the minimum would fail every form.
- **The gate can fail.** On form 1 it went red three times: the text encoder's output against another prompt's
  reference (0.459), the transformer at timestep 0.5 against the 1.0 reference (0.145), and the VAE fed the latent plus
  N(0, 0.5²) noise (29.4 dB).
- **The app computes what the gate measures.** The bundle's components, chained through the same CLI with the Swift
  pipeline's tokens, noise and schedule, give an image 47.4 dB from the app's.
- **The noise matches, not the bits.** `TorchRandomSource(seed: 42)` gives `torch.randn`'s sequence within 1.03e-6
  (Swift runs Box-Muller in Double), so (iv) is measurable.
- **The timesteps differ.** diffusers passes 967.384, 908.144 and 767.2. Swift's `DiscreteFlowScheduler` truncates
  them to 967, 908 and 767. In fp32 the two schedules give images 44.8 dB apart, so (iv) keeps a reference for each.
- **Same graph, same seed, same bytes.** The sample app, apple/coreai-models' `diffusion-runner`, the card's Swift
  example, `apps/CoreAIImageGen` and a folder downloaded from the Hub all wrote byte-identical PNGs.

## int4 fails; the text encoder breaks at the padding

The registry gives `flux2-klein-4b` the `4bit` preset (int4 `symmetric_with_clipping`, per-block 32, on the text
encoder and the transformer) and calls it recommended. The CLI's own default is `none`. On this checkpoint int4 fails
both (i) and (ii). It draws a picture, but not the one fp32 draws: the seed-42 image is 15.9 dB from fp32 (31.0 dB
unquantized), and the door and the bicycle move.

The text encoder breaks at the padding. The prompts fill 15 to 65 of the 512 positions. The transformer has no mask
input, so it reads all 512 rows. Mean token cosine against fp32 on the first prompt (34 real tokens):

| text encoder weights | real tokens | padding tokens |
| --- | --- | --- |
| fp16 | 1.0000 | 0.9999 |
| int8 per-block 32 | 0.9999 | 0.9923 |
| int8 per channel | 0.9997 | 0.9800 |
| int4 per-block 32 | 0.9805 | 0.6578 |

Over the four prompts, int4's padding tokens read 0.64–0.72. Why quantization error gathers at the padding positions
was not isolated.

- int4 also fails with `4bit-asym` (transformer 0.9742), bfloat16 compute (0.9692) and float32 compute (0.9661). The
  weight error dominates.
- The transformer passes at int8. Per-block 32 gives 0.99981 and 11.59 s; per channel gives 0.99955 and 14.80–14.88 s.
- `--compression` accepts a JSON config. coreai-opt's `module_name_configs` is a full-match regex, so a config can keep
  the text encoder (`model.model.*` and `model.lm_head` inside the wrapper) out of quantization. Forms 13 and 14 do
  that. Form 15's JSON is the `4bit` preset with dtype int8, as printed in the card.
- int4 and int8 stay quantized after specialization: the cached `resources.bin` is about the size of `main.mlirb`, and
  the 8 functions share it.

## fp16 needs `--single-function`

Unquantized, the multi-function transformer specializes each of its 8 functions with its own copy of the fp16
weights. The cache entry's `resources.bin` is 62,008,597,216 bytes, 8.0 times the 7.75 GB `main.mlirb`. The first
load took 242.7 s. Loads after a relaunch took 91.4 and 126.5 s.

`--single-function --resolution 1024` writes one asset per graph. The specialized transformer is 7.75 GB, the first
load 27.9 s. The image-to-image graph becomes its own 7.75 GB asset, `Transformer_img2img_full.aimodel`.
`macos-fp16/` leaves it out, so that folder is text-to-image only: an image-to-image request stops with "this bundle
has no img2img transformer". With int4 the switch changes nothing but the peak, 2.9 GiB lower.

`SpecializationOptions.cpuOnly` has the same cost: BNNS expands int4 to fp16, 62.0 GB for the 8-function
transformer. Its outputs are also worse than the GPU's (transformer 0.947, VAE 29.3 dB). Test the CPU on a
single-function export.

## The Neural Engine: no-go

- `xcrun coreai-build compile … --architecture h16c --preferred-compute neural-engine` failed for five graphs (the int4
  text encoder, transformer and VAE decoder; the fp16 transformer and text encoder) with `ANECCompileOffline() failed:
  OSStatus=0, aneCompileStatus=1 … ErrorList = (CompilationFailure)`. It exited 0 and wrote a GPU-only `.aimodelc` with
  no `*_ANE_region_*` entry. Count the regions; the exit code says nothing.
- Asking for `.neuralEngine` in Swift does not fail. The int4 transformer and VAE decoder and the fp16 transformer run
  on the GPU: bit-equal output, the same time. The text encoder gets an ANE region and comes out slower and further
  from fp32: int4 608 ms against 218 ms on the GPU; fp16 cosine 0.901 against 0.99994, after a 442 s specialization.
- The fp16 VAE decoder's `.neuralEngine` specialization sat at 0 % CPU for more than 2.5 minutes and was stopped.
- The Swift pipeline hard-codes `SpecializationOptions(preferredComputeUnitKind: .gpu)`
  (`CoreAIDiffusionModelFunction.swift`). Any compute-unit test needs its own caller; this one used a small CLI around
  `AIModel(contentsOf:options:)` and `loadFunction(named:)`.

## AOT does not pay here

An h16c AOT compile (`--preferred-compute gpu`, 5.4 s per graph) gave a slower call: transformer 2,552.4 ms against
2,485.2 ms JIT, text encoder 234.5 against 218.0 ms. Its outputs differ from the JIT's (text encoder max |Δ| 10.4),
and its gate result is the same.

The app cannot open it anyway. `FlowTransformerPipeline` resolves `<name>.aimodel` only. An AOT asset renamed to
`.aimodel` fails `AIModel(contentsOf:)` with `failedToSpecialize` while it has no cache entry. Load the `.aimodelc`
once and the renamed copy loads too, so a test run after that is a false pass. Delete the entry before testing.

## Tiled decode: measured, not shipped

`DecodeResolution.tiled` runs the half-size VAE over a 3×3 grid of 64² latent tiles. It opens
`VAEDecoder_half.aimodel` by name, which a `--single-function` export without `--low-memory` does not have. The test
folder took that one asset from a `--compression none` export; VAEs are never quantized. The sample opens `.auto`,
which picks the full decoder whenever `VAEDecoder` exists.

Measured with apple/coreai-models' `diffusion-runner`, one image per process, loads included:

| decode | lazy model loading | image (s) | decode (s) | peak footprint (bytes) |
| --- | --- | ---: | ---: | ---: |
| full | on (the runner's default) | 12.74 | 0.79 | 5,475,883,680 |
| tiled | on | 13.23 | 1.52 | 5,473,131,168 |
| full | off (every model stays loaded) | 12.45 | 0.74 | 6,183,775,608 |
| tiled | off | 13.12 | 1.45 | 5,553,270,088 |

With lazy loading the peak is in another stage, so tiling saves nothing. With every model loaded it saves 630 MB
(10.2 %) for 0.67 s. The tiled image is 26.3 dB from the full decode and 25.0 dB from fp32 (full: 31.0 dB). No
memory class here needs it.

## What the app reads

- `loadResources()` loads `VAEEncoder` whenever the folder has one, even for text-to-image. A full export folder (22
  or 25 files) therefore peaks 0.9–1.2 GiB above the 16 files text-to-image needs, with the same images.
- The sample downloads 16 files of either folder. `apps/CoreAIImageGen` downloads 19 files of `macos-int8/`: the 16
  plus `VAEEncoder` for its Edit tab. The half-size VAEs are for `.half` and `.tiled`, which a Mac app opening
  `.auto` never reads.
- Both apps skip `VAEEncoder` in `macos-fp16/`. With it present, `supportsImageToImage` reports true for a folder
  that has no image-to-image transformer.
- Edit in `apps/CoreAIImageGen` is the stock reference-token image-to-image, one reference, 24.39 and 24.31 s per
  1024² edit: [`flux2-in-context-editing.md`](flux2-in-context-editing.md), "On the stock runtime".

## Traps

- **A second export into the same `--output-dir` keeps the first one's weights.** The exporter skips any asset that
  exists and logs one INFO line (`pipeline.py`, "Skipping …: exists (use --overwrite)"). Give each form its own
  directory; the bundle lands in `<output-dir>/FLUX.2-klein-4B/`.
- **`--compression` takes a preset name or a JSON string, not a path**: `--compression "$(cat int8-block32.json)"`.
  The presets are `none`, `4bit`, `4bit-asym` and `8bit`. `--help` points at `--list-presets`, which does not exist.
- **`--components` cannot be combined with `--platform`.**
- **The text encoder is not Qwen3-8B.** It is the Qwen3 inside the FLUX.2 klein repository: hidden size 2560, 36
  layers, the shape of Qwen3-4B. The wrapper reads hidden states 9, 18 and 27, so the export drops the last 9 layers
  (6.2 GB at fp16).
- **diffusers will not load from a component-only download with `HF_HUB_OFFLINE=1`** ("not cached locally"). Run the
  reference online.
- **The specialization cache is per process name**, under
  `~/Library/Caches/coreai-cache/<OS build>/<process or bundle id>/<sha256 of main.mlirb>/`. The runner, the card's
  Swift example and the app each kept their own 14 GB fp16 copy. The sample and `apps/CoreAIImageGen` share the bundle
  id `com.coreai.imagegenmac` and the process name `CoreAIImageGenMac`: they share one cache, and `pkill -x` or `open
  -b` reaches both. Launch by path and stop by pid.
- **Hold an activity in the app.** A Mac app whose window is hidden is App Napped. With the screen locked, the sample's
  Hub download fell to 0.06–0.16 MB/s while curl pulled 4.9–6.4 MB/s; holding
  `ProcessInfo.beginActivity(options: .userInitiated, reason:)` brought it to 8.1–13 MB/s (2026-10-10). Both apps
  hold one around download, load and generation.
- **`/usr/bin/time -l` counts mapped weight files.** Its maximum resident set size read 15.7–21.9 GB where the peak
  memory footprint read 5.5–6.2 GB, and once 173 GB on a 128 GB Mac (not explained). Compare footprints.
- **The Hub's tree API pages its listing** (`Link: …; rel="next"`). Without following it, a check of the 82-file repo
  counted 28 root files.

## Timing on a shared machine

Other work shared this Mac's GPU. A headless Chromium with Metal ANGLE rendered without taking the GPU lock.

- Every timed window sampled GPU utilization and that Chromium's activity once a second. A run during which Chromium
  was active (at first: its GPU process alive; later: 5 % of a core or more) was discarded and retried. Ten runs were
  discarded this way.
- Waiting for zero Chromium GPU processes did not work: an idle one sat at 0 % GPU for minutes. Wait on GPU
  utilization instead, before taking the lock.
- A retried cold run is no longer cold: the first attempt left the files in the page cache (fp16 first load 32.8 s
  discarded, 16.3 s retried). `purge` needs sudo.

## Not explained

- **The fp16 load after a relaunch split in two.** Right after a cold load it took 2.07–2.16 s (3 runs). Later the
  same day it took 17.7–24.4 s (6 runs), from the downloaded folder and from the local one alike. int8 took 1.47–1.82
  s every time. The cached `resources.bin` reads at 5.4–5.9 GB/s with `dd`.
- **First loads disagree.** Right after the download, int8 loaded in 8.75 s and fp16 in 34.06 s. Before the upload the
  same int8 graphs took 21.9 and 20.4 s.
- Why quantization error gathers at the text encoder's padding positions.
- The seed-42 images of another stock export of this model (`dushandz/FLUX.2-klein-4B-CoreAI`, apple/coreai-models
  e282dbdd) are 21.4 and 23.0 dB from this one's int4 form: same composition, different details. The source of the
  difference was not traced.

## Numbers of record

- `macos-fp16/` (form 16): 10.88 s per 1024² image, 4 steps (median of 4, 10.86–10.92 s); gate (i) 0.99966–0.99995,
  (ii) 0.999987, (iii) 65.5 dB; 31.0 dB from fp32 end to end. Text-to-image reads 16 files, 14,090,986,305 bytes.
- `macos-int8/` (form 15): 11.60 s (11.57–11.64 s); gate (i) 0.9928–0.9948, (ii) 0.99981, (iii) 65.5 dB; 23.3 dB end
  to end. Text-to-image reads 16 files, 7,540,032,209 bytes.
- Through the sample's Download & Load on Wi-Fi: int8 in 2,686 s, fp16 in 2,900 s; all 16 files' SHA-256 matched each
  folder's `MANIFEST.json`.
- The export: form 16 in 39.8 s, form 15 in 110.8 s; peak RSS 24.2–42.4 GB over the twelve exports.
