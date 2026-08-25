# V-JEPA 2 ViT-L (SSv2) — Core AI

[`facebook/vjepa2-vitl-fpc16-256-ssv2`](https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2)
(375M, MIT) — Meta's **video world model**: a self-supervised video encoder (JEPA — learns by
predicting in representation space, not pixels) with an attentive pooler + Something-Something-v2
**action-recognition head** (174 physical-interaction classes: pushing, lifting, covering,
rolling…). A 16-frame clip goes in, an action label comes out — the zoo's **first video
understanding model**, a category with no MLX port and no Apple stock path.

Bundle: [🤗 mlboydaisuke/VJEPA2-ViTL-SSv2-CoreAI](https://huggingface.co/mlboydaisuke/VJEPA2-ViTL-SSv2-CoreAI)
— macOS (~708 MB fp16 JIT) + iOS (~1.4 GB, AOT-precompiled for iPhone), `labels.json`
alongside. Catalog id: **`vjepa2-vitl-ssv2`**.

<!-- gen-cards:use-it begin id=vjepa2-vitl-ssv2 (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — this model is the default behind the kit's task op
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let actions = try await CoreAI.recognizeAction(videoAt: videoURL)
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [ActionCamera runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ActionCamera)
(live camera action recognition, one app for every video model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/ActionCamera/ActionCamera.xcodeproj
# → Run, then pick "V-JEPA 2 ViT-L (SSv2)" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/ActionCamera
swift run action-cli --model vjepa2-vitl-ssv2 --video sample.mp4
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKitVision

let recognizer = try await ActionRecognizer(catalog: "vjepa2-vitl-ssv2")
let actions = try await recognizer.classify(videoAt: videoURL, topK: 3)
// actions: ranked [Prediction] — .label ("Pushing [something] from left to right"),
// .probability; 174 SSv2 classes, fully on-device
```

The take-home is [`Examples/ActionCamera/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/main/Examples/ActionCamera/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; the CLI is an argument shell over it, and
the GUI classifies a rolling 16-frame clip from `CameraFeed`.
Live camera? Keep the last 16 `CameraFeed` frames and call `classify(frames:)` — other
frame counts are uniformly resampled to 16. The bundled `sample.mp4` is a synthetic
clip (a hand pushing a block); point `--video` at real footage for real results.

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` → product **CoreAIKitVision**
- Info.plist: `NSCameraUsageDescription` — only for the live camera; the snippet needs none
- Entitlements: none needed
- First run downloads the model — 0.7 GB (Mac) / 0.7 GB (iPhone) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Measured

| Platform | Latency (16-frame clip) |
|---|---|
| M4 Max, GPU | **~150–180 ms** (load 0.15 s) |
| iPhone 17 Pro (AOT h18p) | **~0.34 s** warm |

Parity: engine vs the HF fp16 reference **cos 0.999996, top-5 identical**; plus a semantic
gate — a synthetic square moving up vs down flips the top labels
(`Moving [something] up` ↔ `…down`), proving motion understanding on-engine with no dataset
(`conversion/vjepa2/gate_semantic.py`).

## Shape of the port

One stateless graph: `pixel_values_videos [1,16,3,256,256] → logits [1,174]`
(`VJEPA2ForVideoClassification`, direct export). Host does the preprocessing: 16 frames
uniform-sampled, aspect-fill 256², RGB 0..1, ImageNet mean/std — the model does **not**
normalize internally. One overlay was needed: 3D RoPE's no-op `squeeze(-1)` on a non-1 dim
maps to `ShrinkDims` and fails — patched out, math unchanged (see
[`conversion/vjepa2/export_fp16.py`](../../conversion/vjepa2/export_fp16.py) and
[`knowledge/video-world-models-vjepa2.md`](../../knowledge/video-world-models-vjepa2.md)).

## Run

Live camera → rolling 16-frame clip → label: the kit's
[ActionCamera runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ActionCamera)
(GUI + `action-cli`), or the zoo's bespoke [coreai-video](../../apps/coreai-video) app
(iPhone-verified).
