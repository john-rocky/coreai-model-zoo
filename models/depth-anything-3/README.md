# Depth Anything 3 (small / base / mono-large) — Core AI

**The zoo's first depth model** — monocular (single-image) depth estimation running as a
single static `.aimodel` on every Apple compute unit. A port of ByteDance's
[Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3)
([`depth-anything/DA3-SMALL`](https://huggingface.co/depth-anything/DA3-SMALL) and friends,
Apache-2.0): a DINOv2 ViT backbone + DPT-style dense head that predicts a relative depth map
(and a confidence map) from one RGB image. Host post-processing is a colormap — nothing else.

DA3 is an **"any-view"** model (1 → many views). Fed a **single view (S = 1)** it is a monocular
depth estimator: the alternating cross-view attention collapses to self-attention, the
reference-view reorder is statically dead (it needs S ≥ threshold), and the camera token is a
fixed parameter. We export only the depth path (**backbone + head → depth, depth_conf**); the
camera decoder, ray head and sky post-processing are dropped — the ray branch is dead-code
eliminated by `optimize()` because only `depth`/`depth_conf` are graph outputs.

| variant | backbone | params | head | output |
|---|---|---|---|---|
| small | DINOv2 ViT-S | 34.3M | DualDPT | depth + confidence |
| base | DINOv2 ViT-B | 135.4M | DualDPT | depth + confidence |
| mono-large | DINOv2 ViT-L | 334.2M | DPT | depth (pure monocular) |

<!-- gen-cards:use-it begin id=depth-anything-3-small (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — this model is the default behind the kit's task op
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let map = try await CoreAI.estimateDepth(in: image)
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [DepthCamera runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DepthCamera)
(live camera depth, one app for every depth model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/DepthCamera/DepthCamera.xcodeproj
# → Run, then pick "Depth Anything 3 Small" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/DepthCamera
swift run depth-cli --model depth-anything-3-small --image sample.jpg --output depth.png
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKitVision

let estimator = try await DepthEstimator(catalog: "depth-anything-3-small")
let image = try ImageFile.load(imageURL)  // any image file → CGImage + EXIF orientation
let depth = try await estimator.estimateDepth(for: image.cgImage)
// depth: DepthMap — .cgImage() renders it, .values are the raw floats
```

The take-home is [`Examples/DepthCamera/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/main/Examples/DepthCamera/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; the CLI is an argument shell over it, and
the GUI runs the same estimator on every camera frame (`CameraFeed`, ~10 lines).
Live camera? `CameraFeed` (kit API) streams frames — feed each one to
`estimateDepth(for:)`; the camera permission prompt is your app's own chrome.

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` → product **CoreAIKitVision**
- Info.plist: `NSCameraUsageDescription` — only for the live camera; the snippet needs none
- Entitlements: none needed
- First run downloads the model — 0.1 GB (Mac) / 0.1 GB (iPhone) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Graph contract

```
image [1, 3, 504, 504] RGB in [0, 1]   (ImageNet mean/std folded in-graph)
  └─ da3-<variant>_<dtype>.aimodel ─▶ depth [1, 504, 504]      (relative, exp-activated)
                                       depth_conf [1, 504, 504] (small/base only)
host: colormap for display (DA3 default: inverse-depth → percentile 2–98 → Spectral)
```

### Input: square preprocessing contract

The bundle is a **fixed square 504 × 504** graph. The host: resize the RGB image to 504 × 504
(cv2 `INTER_AREA`), feed it as **raw [0, 1]** (the ImageNet normalization is folded into the graph),
run, then **resize the depth map back to the original H × W**. Depth is relative, so the brief
aspect squash during inference is recovered by the resize-back.

- **Faithfulness:** the shipped model vs the official DA3 viewer is **mean r ≈ 0.98** across diverse
  aspect ratios (square inputs are **r = 1.000**), which is within DA3's **own** resolution
  sensitivity — its 504-vs-518 outputs differ by r ≈ 0.975–0.984. The deviation is at the model's
  intrinsic floor, not a port artifact.
- **Bit-exactness:** the engine is **cos 1.000000 vs torch at any fixed input shape** — square and
  non-square alike — verified on diverse images. So a per-aspect **non-square** bundle is also
  possible for exact-match deployments; the single square bundle + resize-back is the simplest
  contract and is what ships.
- **Why 504 (= 36 × 14):** it matches DA3's default `process_res`, so the model runs at its native
  operating point. The pos-embed bicubic interpolation is over fixed sizes, so it folds to a
  constant at export (no runtime bicubic). (518 = 37 × 14, the DINOv2 native grid with no
  interpolation at all, also exports and is ~5% faster.)

## The matrix (M4 Max GPU, greedy, vs the HF eager fp32 reference)

| variant · dtype | size | parity (cos / scale-aligned) | M4 Max GPU |
|---|---|---|---|
| small · fp32 | 105 MB | cos 1.000000, relmax 3e-5 (cpu+gpu) | 17.7 ms · **56.5 FPS** |
| **small · fp16** | **54.5 MB** | cos 1.000000, relmax 7e-3 | **15.2 ms · 65.7 FPS** |
| base · fp32 | 402 MB | cos 1.000000, relmax 4e-5 | 43.4 ms · 23.0 FPS |
| base · fp16 | 202 MB | cos 1.000000, relmax 1e-2 | 37.7 ms · 26.5 FPS |
| mono-large · fp32 | 1.34 GB | cos 1.000000, relmax 2e-4 | 118.3 ms · 8.5 FPS |

Parity = cosine + scale-aligned relative error of the engine depth map vs the torch fp32 oracle on
real images, at the fixed square resolution. fp32 gates bit-clean on **cpu and gpu**; fp16 (depth
is robust) holds cos 1.000000 with ≤ ~1% per-pixel error. **`small · fp16` is the iPhone hero**
(54.5 MB, 65.7 FPS on M4 Max). mono-large is depth-only (no confidence) and Mac-tier.

## Conversion

`conversion/export_da3.py` (one script, all variants, both DualDPT and DPT heads). Two import-time
patches, both numerically identical: (1) RoPE's frequency-table length is `int(positions.max())+1`
— a Python int from a traced tensor → a data-dependent guard under `torch.export`; the grid is
static so we bake the length as a constant, and we make the RoPE / position caches recompute (the
originals memoize tensors that `torch.export` poisons with fakes). (2) the DPT pos-embed UV grid is
hard-cast to fp32; we cast it back to the feature dtype so fp16 export is dtype-uniform.

```
python export_da3.py --variant small --dtype float16 --res 504 \
    --ckpt <DA3-SMALL dir> --da3-src <depth_anything_3 src> \
    --verify-images img1.jpg,img2.jpg --unit gpu
```

No platform-bug workarounds were needed for the square path — bilinear upsample, 2D RoPE, SDPA and
ConvTranspose all lower as-is on the GPU delegate.

**Source:** [Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3) ·
[`depth-anything/DA3-SMALL`](https://huggingface.co/depth-anything/DA3-SMALL) · Apache-2.0.
