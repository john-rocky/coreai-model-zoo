# AdcSR ×4 Super-Resolution — Core AI

On-device **×4 super-resolution** with **AdcSR** ([Adversarial Diffusion Compression](https://github.com/Guaishou74851/AdcSR),
CVPR 2025) converted for Apple's **Core AI** stack. AdcSR compresses the one-step diffusion model
[OSEDiff](https://github.com/cswry/OSEDiff) into a small **diffusion-GAN**: a pruned Stable
Diffusion 2.1 UNet + a half-size VAE decoder, run in **one forward pass** — no iterative denoising,
no prompt, no noise — so it is fast and small enough to run fully on-device, including iPhone.

<!-- gen-cards:use-it begin id=adcsr-x4 (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — this model is the default behind the kit's task op
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let big = try await CoreAI.upscale(image)
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [UpscaleDemo runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/UpscaleDemo)
(pick a photo, upscale it ×4 on-device):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/UpscaleDemo/UpscaleDemo.xcodeproj
# → Run, pick a photo — the app loads AdcSR ×4 (the catalog's superResolution entry) automatically

# agents / headless (macOS):
cd coreai-kit/Examples/UpscaleDemo
swift run upscale-cli --model adcsr-x4 --image sample_small.png --output big.png
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKitVision

let resolver = try await SuperResolver(catalog: "adcsr-x4")
let image = try ImageFile.load(imageURL)  // any image file → CGImage + EXIF orientation
let upscaled = try await resolver.upscale(image.cgImage)
// upscaled: CGImage — 4× the input's pixels
```

The take-home is [`Examples/UpscaleDemo/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/main/Examples/UpscaleDemo/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; the CLI is an argument shell over it, and
the GUI runs the same resolver on the photo you pick.
Big photos? Inputs are tiled and feather-blended internally; `maxInputSide` (default 512)
caps the input first so a full-res phone photo can't produce a gigapixel result.

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` → product **CoreAIKitVision**
- Info.plist: none needed
- Entitlements: none needed
- First run downloads the model — 1.7 GB (Mac) / 1.7 GB (iPhone) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## What it is

- **fp32, ~1.7 GB.** Output matches the torch reference (cosine 1.000012). fp32 because the
  pruned SD-2.1 UNet's attention/group-norm overflow in fp16 (NaN on smooth tiles).
- **Image → image, one step.** Input a low-resolution tile, get a 4× tile back. No text, no noise.
- **456 M parameters** (pruned SD-2.1 UNet + half VAE decoder).
- The graph outputs the **raw** SR; AdcSR's per-image color-match is applied host-side by
  `SuperResolver` after tiling (baking it per-tile blows up uniform tiles).

## I/O contract (per tile)

- **input:** `lr` `[1,3,128,128]` in `[-1,1]` (a low-resolution tile).
- **output:** `sr` `[1,3,512,512]` in `[-1,1]` (×4), with the reference's per-image color-match baked in.

## Usage (CoreAIKit)

```swift
import CoreAIKitVision

let sr = try await SuperResolver(model: .adcsrX4)   // downloads this repo on first use
let big = try await sr.upscale(cgImage)             // ×4; tiles any-size input + feather-blends
```

`SuperResolver` splits any-size input into overlapping 128-px LR windows, runs each, and blends
(and caps very large inputs so the result stays a reasonable size).

## License & attribution

- **AdcSR** (method + the pruning/training code): Apache-2.0 — Bingchen Li et al., *Adversarial
  Diffusion Compression for Real-World Image Super-Resolution*, CVPR 2025.
- **Weights** are derived from **Stable Diffusion 2.1** (via OSEDiff) and therefore carry the
  **CreativeML Open RAIL++-M** license — commercial use is permitted under its use-based
  restrictions, the same license under which Apple distributes Stable Diffusion for Core ML.

This Core AI conversion inherits both. See `LICENSE` (Apache-2.0, AdcSR) and the SD-2.1 OpenRAIL++-M
terms.

---

**⬇️ Download:** [🤗 mlboydaisuke/AdcSR-CoreAI](https://huggingface.co/mlboydaisuke/AdcSR-CoreAI) — this card and the
model page are the same document; `scripts/gen-cards` keeps the *Use it* block in sync.
Reproduce it: `python3 conversion/zoo_convert.py show adcsr-x4` prints the command and its
prerequisites; [`recipe.toml`](recipe.toml) is the record.
