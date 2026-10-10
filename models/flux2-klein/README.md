# FLUX.2 klein 4B — Core AI

> **2026-10-10 re-export, two folders.** `macos-fp16/` and `macos-int8/` of the HF repo were exported with
> apple/coreai-models 97a14be and load with its diffusion pipeline. The files at the repo root are the 2026-07-20
> export: current apple/coreai-models stops on them with "unsupported metadata_version". They stay for
> [`apps/CoreAIImageGen`](../../apps/CoreAIImageGen), which pins a fork of that time.

[Black Forest Labs' **FLUX.2 [klein] 4B**](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
converted to **Core AI** for image generation on a Mac (macOS 27). It runs on Apple's diffusion pipeline in
[apple/coreai-models](https://github.com/apple/coreai-models).

FLUX.2 [klein] is step-distilled: 4 denoising steps at guidance 1.0 give a 1024×1024 image. It pairs a
flow-matching diffusion transformer (DiT, 25 blocks) with a Qwen3 text encoder.

Mac only. Both folders are `--platform macOS` exports. The earlier int4 export already peaked at about 6.5 GB on an
iPhone 17 Pro, over the ~6.1 GB a 12 GB iPhone gives one app ([`apps/CoreAIImageGen`](../../apps/CoreAIImageGen)).

## Files

| Folder | Weights | Runs | Files | Bytes |
| --- | --- | --- | --- | --- |
| `macos-fp16/` | fp16, one asset per graph | text-to-image | 19 | 14,159,882,995 |
| `macos-int8/` | int8 per-block 32 on the text encoder and the transformer, fp16 VAEs | text-to-image, image-to-image | 25 | 7,777,120,667 |
| repo root (2026-07-20) | int4 per-block 32, plus the in-context edit transformers | `apps/CoreAIImageGen` only | 34 | 10,854,216,650 |

Text-to-image reads 16 files of a folder: 14,090,986,305 bytes from `macos-fp16/`, 7,540,032,209 bytes from
`macos-int8/`. The VAE encoders and the half-size VAEs are only for image-to-image and tiled decoding.

`macos-fp16/` leaves out the image-to-image transformer (`Transformer_img2img_full.aimodel`, 7.75 GB). Image-to-image
with it stops with "this bundle has no img2img transformer". `macos-int8/` has one transformer with all eight
entrypoints, image-to-image included.

Each folder has a `MANIFEST.json`: the size and SHA-256 of every file, the export command, and the gate result.

## Usage

### Sample apps

- [CoreAIImageGenMac](https://github.com/john-rocky/coreai-samples/tree/main/CoreAIImageGenMac) (coreai-samples) runs
  `macos-fp16/` or `macos-int8/` on unmodified apple/coreai-models 97a14be: pick one, press Download & Load, type a
  prompt, press Generate.
- [`apps/CoreAIImageGen`](../../apps/CoreAIImageGen) runs the root files on the `john-rocky/coreai-models` fork,
  including the **Edit** tab below.

### Swift

Add apple/coreai-models at commit `97a14be` as a Swift package (product `CoreAIDiffusion`). `folder` is a downloaded
`macos-fp16/` or `macos-int8/`.

```swift
import CoreAIDiffusionPipeline

let pipeline = try await FlowTransformerPipeline(from: folder, mode: .auto)
try await pipeline.loadResources()
let configuration = PipelineConfiguration(
    prompt: "a red bicycle leaning against a blue wooden door", seed: 42,
    stepCount: 4, guidanceScale: 1.0, guidanceMode: .distilled, lazyModelLoading: false)
let result = try await pipeline.generateImages(configuration: configuration) { _ in true }
let image = result.images.first  // CGImage, 1024×1024
```

### Command line

apple/coreai-models' own runner, from a checkout at `97a14be`:

```bash
swift run -c release diffusion-runner --model path/to/macos-fp16 \
  --prompt "a red bicycle leaning against a blue wooden door" --steps 4 --seed 42 --output bicycle.png
```

## How it was converted

Apple's stock exporter, no zoo code ([`recipe.toml`](recipe.toml): `flux2-klein-4b-fp16`, `flux2-klein-4b-int8`):

```bash
git clone https://github.com/apple/coreai-models && cd coreai-models
git checkout 97a14be40bb1c5b95badb08b8532530443d62d26
```

`macos-fp16/` is `exports/fp16/FLUX.2-klein-4B/` without `Transformer_img2img_full.aimodel`:

```bash
uv run coreai.diffusion.export flux2-klein-4b --platform macOS \
  --compression none --single-function --resolution 1024 --output-dir exports/fp16
```

`macos-int8/` is `exports/int8/FLUX.2-klein-4B/`, made with the exporter's own 4bit preset changed to dtype int8:

```bash
cat > int8-block32.json <<'EOF'
{"execution_mode": "eager",
 "global_config": {"op_state_spec": {"weight": {"dtype": "int8", "qscheme": "symmetric_with_clipping", "granularity": {"type": "per_block", "block_size": 32}}}, "op_input_spec": null, "op_output_spec": null},
 "module_type_configs": {"diffusers.models.normalization.RMSNorm": null, "transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm": null, "transformers.models.gemma2.modeling_gemma2.Gemma2RMSNorm": null, "transformers.models.umt5.modeling_umt5.UMT5LayerNorm": null}}
EOF
uv run coreai.diffusion.export flux2-klein-4b --platform macOS \
  --compression "$(cat int8-block32.json)" --output-dir exports/int8
```

`--compression` takes a preset name or a JSON string, not a file path. Give each export its own `--output-dir`: the
exporter skips any asset that already exists in its folder, so a second export there keeps the first one's weights.
The source was `black-forest-labs/FLUX.2-klein-4B` at `e7b7dc27`. The root files came from the same CLI with its
defaults in 2026-07 (int4), plus the fork's edit transformers (`flux2-klein-4b-edit` in `recipe.toml`).

## Measured

M4 Max Mac Studio (128 GB), macOS 27.0 (26A428), apple/coreai-models 97a14be, GPU, 2026-10-10. Each export folder was
opened in CoreAIImageGenMac with Local…: 1024×1024, 4 steps, guidance 1.0, seed 42, two prompts. "Image" is the median
of 4 runs after a relaunch, with no other GPU work running.

Parity is checked per component against fp32 PyTorch (diffusers' `Flux2KleinPipeline` through apple/coreai-models'
own export wrappers), on the inputs the Swift pipeline builds: (i) text encoder, mean cosine over the 512 tokens, 4
prompts; (ii) transformer, cosine of the first step's output; (iii) VAE decoder, PSNR of the 8-bit image. The bar is
0.999 / 0.999 / 35 dB without quantization and 0.99 / 0.99 / 35 dB with it. The last column compares the seed-42 image
with the fp32 diffusers image.

| Weights | Image (s) | First load (s) | Load after relaunch (s) | Peak memory (GiB) | (i) / (ii) / (iii) | Parity | Image vs fp32 (dB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp16 (`macos-fp16/`) | 10.88 | 27.9 | 2.1 | 9.2–9.4 | 0.99966–0.99995 / 0.999987 / 65.5 dB | PASS (0.999) | 31.0 |
| int8 per-block 32 (`macos-int8/`) | 11.60 | 20.4 | 1.7–1.8 | 10.8–10.9 | 0.9928–0.9948 / 0.99981 / 65.5 dB | PASS (0.99) | 23.3 |
| fp16 text encoder + int8 transformer (not published) | 11.59 | 11.2 ¹ | 1.9 | 10.7–10.8 | 0.99966–0.99995 / 0.99981 / 65.5 dB | PASS (0.99) | 29.3 |
| int4 per-block 32 (the exporter's default for this model) | 11.02 | 6.7 | 1.1 | 10.7–10.8 | 0.65–0.75 / 0.9675 / 65.5 dB | FAIL | 15.9 |

¹ Its files were already in memory from a discarded run 70 s earlier; the other first loads read them from disk.

The int4 form draws a picture, but not the one fp32 draws. Its seed-42 image is 15.9 dB from the fp32 image, and the
layout changes: the door and the bicycle move. The quantized text encoder loses the padding positions that the
transformer reads (cosine 0.64–0.72 there). That is why the new folders are fp16 and int8.

fp16 is exported with `--single-function`. Without it, Core AI specializes the unquantized transformer once per
entrypoint: 62.0 GB of cache and a 243 s first load. Everything runs on the GPU. Compiling the transformer, the text
encoder or the VAE decoder for the Neural Engine fails (`ANECCompileOffline() … CompilationFailure`).

## The text encoder

The text encoder is the Qwen3 model inside the FLUX.2 klein repo: hidden size 2560, 36 layers, the shape of
Qwen3-4B. Earlier versions of this card said 8B; that was wrong. The pipeline reads hidden states 9, 18 and 27, so the
export drops the last 9 layers.

## In-context editing (repo root, 2026-07-20)

The repo root also has **`Transformer_edit.aimodel`**, **`Transformer_edit_512.aimodel`** and
**`Transformer_edit_2ref.aimodel`** for FLUX.2's in-context editing. You give a reference image and an instruction —
*"add a red wizard hat, keep everything else the same"* — and only the instructed change is applied while the
subject, pose, and background are preserved. This is different from strength-based image-to-image (SDEdit), which
re-renders the whole frame.

It is the same DiT graph exported at a longer sequence: the output latent (time index `T=0`) concatenated with the
reference image's latent tokens (`T=10`), so the transformer attends to the reference while denoising the output.
`Transformer_edit_2ref.aimodel` takes two references (`T=10`, `T=20`). Running them needs a runtime that drives this
path: [`apps/CoreAIImageGen`](../../apps/CoreAIImageGen) pins the `john-rocky/coreai-models` fork with
`Flux2Pipeline.editImages` and exposes it as the **Edit** tab. These files are int4 and use the 2026-07-20 layout, so
apple/coreai-models 97a14be does not load them. Mechanism:
[knowledge/flux2-in-context-editing.md](../../knowledge/flux2-in-context-editing.md).

int4, ~25 s for a 1024 edit and ~43 s for a two-reference edit on a Mac GPU (4 steps, guidance 1.0; numbers from
the earlier card, not measured again).

## License

Apache 2.0, inherited from the base model
[black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).
The converted weights are redistributed under the same terms, with attribution to
Black Forest Labs.

---

**⬇️ Download:** [🤗 mlboydaisuke/FLUX.2-klein-4B-CoreAI](https://huggingface.co/mlboydaisuke/FLUX.2-klein-4B-CoreAI) — the
model page carries the same text. Reproduction: [`recipe.toml`](recipe.toml) — `flux2-klein-4b-fp16` and
`flux2-klein-4b-int8` are `verified` (the gate above); the root files' `flux2-klein-4b` records the 2026-07 run, and
`flux2-klein-4b-edit` is `unverified` (see [`../_INVENTORY.md`](../_INVENTORY.md), "Needs owner input").

