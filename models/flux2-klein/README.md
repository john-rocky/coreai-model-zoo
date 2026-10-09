# FLUX.2 klein 4B — Core AI

> **Note (2026-10-10):** the HF export (2026-07-20) loads only with the apple/coreai-models diffusion pipeline of that time; on current main (97a14be and later) it stops with "unsupported metadata_version". A re-export with the current exporter is planned. Until then, the coreai-samples app CoreAIImageGenMac (https://github.com/john-rocky/coreai-samples/tree/main/CoreAIImageGenMac) pins dushandz/FLUX.2-klein-4B-CoreAI, a community export made with the current recipe.

[Black Forest Labs' **FLUX.2 [klein] 4B**](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
converted to **Core AI** for on-device image generation on Apple Silicon (macOS 27+),
running on Apple's official diffusion runtime in
[apple/coreai-models](https://github.com/apple/coreai-models).

FLUX.2 [klein] is step-distilled: **4 denoising steps at guidance 1.0** produce a full
1024×1024 image. It pairs a 4B flow-matching diffusion transformer (DiT) with an 8B
Qwen3 text encoder.

> **macOS only.** At 4B the peak footprint (~6.5 GB — the text encoder stays resident
> through the transformer) exceeds a 12 GB iPhone's ~6.1 GB per-process memory limit, even
> with the transformer AOT-compiled. Use a smaller diffusion model (e.g. Stable Diffusion
> 0.9B) for on-device iOS image generation.

## Components

| Component | Description |
| --- | --- |
| `Transformer.aimodel` | Flow-matching DiT (25 blocks), 1024×1024 |
| `TextEncoder.aimodel` | Qwen3 text encoder (hidden states 9 / 18 / 27) |
| `VAEDecoder.aimodel` | Latent → 1024×1024 RGB image |
| `VAEEncoder.aimodel` | 1024×1024 RGB image → latent (image-to-image / editing) |
| `Transformer_edit.aimodel` | In-context edit DiT — 1024, sequence 8192 (output + 1 reference) |
| `Transformer_edit_512.aimodel` | In-context edit DiT — 512, sequence 2048 |
| `Transformer_edit_2ref.aimodel` | Two-reference edit DiT — 1024, sequence 12288 (output + 2 references) |
| `Transformer_edit_2ref_512.aimodel` | Two-reference edit DiT — 512, sequence 3072 |
| `tokenizer/`, `pipeline.json`, `vae_bn_*.npy` | Sidecar assets (auto-loaded) |

Weights are 4-bit quantized (int4, per-block, block size 32); compute precision
float16. The full bundle is **4.0 GB** — Transformer 2.0 GB · TextEncoder 1.8 GB ·
VAE 0.16 GB.

## Usage

### Sample app (easiest)

[**CoreAIImageGen** (macOS)](https://github.com/john-rocky/coreai-model-zoo/tree/main/apps/CoreAIImageGen)
— run the `CoreAIImageGenMac` scheme, tap **Download & Load**, type a prompt, **Generate**.

### Swift

```swift
import CoreAIDiffusionPipeline

let pipeline = try await Flux2Pipeline(from: modelURL)
let config = PipelineConfiguration(
    prompt: "a photo of a cat",
    stepCount: 4,
    guidanceScale: 1.0,
    schedulerType: .discreteFlow
)
let result = try await pipeline.generateImages(configuration: config) { _ in true }
let image = result.images.first!
```

### Command line (zoo reference tool)

```bash
swift run -c release diffusion-runner \
  --model path/to/FLUX.2-klein-4B \
  --prompt "a photo of a cat" --steps 4 --guidance-scale 1.0
```

## In-context editing

Beyond text-to-image and image-to-image, this bundle ships **`Transformer_edit.aimodel`** for
FLUX.2's native **in-context editing**. You give a reference image and an instruction —
*"add a red wizard hat, keep everything else the same"* — and only the instructed change is
applied while the subject, pose, and background are preserved. This is different from
strength-based image-to-image (SDEdit), which re-renders the whole frame.

It is the same DiT graph exported at a longer sequence: the output latent (time index `T=0`)
concatenated with the reference image's latent tokens (`T=10`), so the transformer attends to
the reference while denoising the output. The reference tokens are kept clean each step and
their predictions are discarded. Running it needs a runtime that drives this path
(`Flux2Pipeline.editImages`) — the zoo's **CoreAIImageGen** app exposes it as the **Edit** tab.
The stock apple/coreai-models runtime does text- and image-to-image only.

int4, ~25 s for a 1024 edit on a Mac GPU (4 steps, guidance 1.0).

### Multi-reference

**`Transformer_edit_2ref.aimodel`** takes **two** reference images at once — each concatenated at
its own time index (`T=10`, `T=20`) — so the instruction can combine them: *"put the subject from
the first image into the scene from the second image."* Same mechanism, longer sequence (12288).
`editImages(referenceImages:)` selects the 1- or 2-reference transformer by the number of images.
int4, ~43 s for a 1024 two-reference edit on a Mac GPU.

## How it was converted

```bash
uv run coreai.diffusion.export flux2-klein-4b --platform macOS
# in-context edit transformers
uv run coreai.diffusion.export flux2-klein-4b --components transformer_edit transformer_edit_512
```

## Performance

M4 Max (128 GB): **~17 s** for a 4-step 1024×1024 image (cold model load + 4 denoising
steps + VAE decode). The distilled 4-step schedule means no negative prompt / CFG is
needed (guidance 1.0).

## License

Apache 2.0, inherited from the base model
[black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).
The converted weights are redistributed under the same terms, with attribution to
Black Forest Labs.

---

**⬇️ Download:** [🤗 mlboydaisuke/FLUX.2-klein-4B-CoreAI](https://huggingface.co/mlboydaisuke/FLUX.2-klein-4B-CoreAI) — this card and the
model page are the same document; `scripts/gen-cards` keeps the *Use it* block in sync.
Reproduction: [`recipe.toml`](recipe.toml) records what is known, but its `status` is
`unverified` — see [`../_INVENTORY.md`](../_INVENTORY.md), "Needs owner input".
