# RF-DETR (nano / small / medium / large) — Core AI

**The zoo's first object detector**, and the answer to
[apple/coreai-models#14](https://github.com/apple/coreai-models/issues/14):
Roboflow's [RF-DETR](https://github.com/roboflow/rf-detr) — the real-time
detection transformer that broke 60 mAP on COCO — running as a single static
`.aimodel` on every Apple compute unit. **DETR family = no NMS**: host
post-processing is one sigmoid and a top-k.

Architecture (rfdetr 1.7.1): windowed-attention **DINOv2 ViT-S backbone** →
single-scale P4 projector → **two-stage proposal head** (top-300 of HW
anchors) → **deformable-attention decoder** (multi-scale deformable sampling,
2 points/head, iterative box refinement, `bbox_reparam`). All four sizes share
the architecture — only resolution, decoder depth and window count change:

| variant | input | params | dec layers | `.aimodel` |
|---|---|---|---|---|
| nano | 384² | 30.5M | 2 | 108 MB |
| small | 512² | 32.1M | 3 | 115 MB |
| medium | 576² | 33.7M | 4 | 121 MB |
| large | 704² | 33.9M | 4 | 122 MB |

<!-- gen-cards:use-it begin id=rf-detr (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — this model is the default behind the kit's task op
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let boxes = try await CoreAI.detect(inImageAt: url)
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [DetectCamera runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DetectCamera)
(real-time object detection on the zero-copy camera path):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/DetectCamera/DetectCamera.xcodeproj
# → Run, then pick "Nano" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/DetectCamera
swift run detect-cli --model rf-detr --image Resources/gate_image.jpg
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKitVision

let detector = try await KitDetector(catalog: "rf-detr")
let image = try ImageFile.load(imageURL)  // any image file → CGImage + EXIF orientation
let detections = try await detector.detect(in: image.cgImage)
// detections: [Detection] — label, score, normalized box (top-left origin)
```

The take-home is [`Examples/DetectCamera/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/main/Examples/DetectCamera/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; the CLI is an argument shell over it, and
the GUI runs the same detector per camera frame on a zero-copy pixel-buffer fast path.
Real time? Use `detect(in: CVPixelBuffer)` — vImage scales the frame with no CGImage
round-trip; `CameraFeed` (kit API) streams the buffers.

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
input  "image"  [1, 3, R, R]  float32, RGB in [0, 1]   (ImageNet mean/std folded in-graph)
output "dets"   [1, 300, 4]   cxcywh, normalized to [0, 1]
output "labels" [1, 300, 91]  raw logits; column index = ORIGINAL COCO id (0 unused, 1=person … 17=cat … 90)
```

Host decode: `score = sigmoid(labels)`, take top-k over the flattened
query×class plane, gather boxes — done. No anchors, no NMS, no letterboxing
(plain square resize, matching the upstream recipe).

## Measured (macOS 27 / iOS 27 beta, GPU, median)

| variant | M4 Max | M4 Max CPU | iPhone 17 Pro | iPhone end-to-end FPS |
|---|---|---|---|---|
| nano | **8.6 ms** (~116 FPS) | 27.1 ms | **~25 ms** | **33–39** |
| small | **12.0 ms** (~83 FPS) | 44.9 ms | — | — |
| medium | **14.8 ms** (~68 FPS) | 56.5 ms | **56–63 ms** | **15–17** |
| large | **19.1 ms** (~52 FPS) | 86.1 ms | — | — |

iPhone numbers are live-camera measurements from the DetectCamera example app
(Release, zero-copy capture pipeline: AVCaptureVideoPreviewLayer display +
hardware-scaled 32BGRA data buffers + vImage preprocessing overlapped with GPU
inference; 60 fps capture). Peak measured 39.6 FPS ≈ the nano model ceiling.
Sustained max-load throughput drops on a hot chassis (thermal); one-time
on-device specialization ~5 s on first load. The on-device gate reproduces the Mac fp32 oracle detections with
boxes within 1e-3 normalized and scores within 0.01 (medium) / 0.04 (nano) —
the documented GPU h16c noise class. `neural_engine` preference loads and
gates clean but measures ≈ GPU — the gather-heavy deformable decoder keeps the
graph on the GPU delegate.

**Ship dtype is fp32.** It gates bit-clean everywhere, and fp16 only bought 7%
latency (14.8 → 13.7 ms medium) while introducing near-tie detection noise
(score swings up to ~0.04 on duplicate predictions). Detection ≠ LLM: there is
no memory-bandwidth-bound decode loop to feed, so weight bytes barely matter.

Numerics gate: per real COCO image, every confident (>0.3) torch-fp32
detection must have a same-class IoU≥0.75 partner within 2e-3 score in the
`.aimodel` output (set-based matching — DETR emits near-duplicates whose ranks
swap under noise, so positional top-k compare overflags). **All four variants:
cpu AND gpu, 4 images each, worst-IoU ≥ 0.999, zero misses.**

<p align="center"><img src="https://huggingface.co/mlboydaisuke/RF-DETR-CoreAI/resolve/main/demo_coco_cats.jpg" width="420" alt="RF-DETR medium on Core AI — cats 0.94, remotes 0.93/0.86, couch 0.53"></p>

## ⬇️ Bundles

**[mlboydaisuke/RF-DETR-CoreAI](https://huggingface.co/mlboydaisuke/RF-DETR-CoreAI)** —
`rfdetr-{nano,small,medium,large}_float32.aimodel`. Apache-2.0 (upstream code
and COCO-pretrained weights are Apache-2.0).

Convert yourself: [`conversion/export_rf_detr.py`](../../conversion/export_rf_detr.py)
(`pip install rfdetr==1.7.1`, torch ≤ 2.11 to match coreai-torch 0.4.0).

Live-camera reference app: **DetectCamera** in
[CoreAIKit](https://github.com/john-rocky/coreai-kit) (`Examples/DetectCamera` —
`CameraFeed` + `ObjectDetector`, box overlay, in-app Hub download; the whole ML
surface is two calls).

## RF-DETR-Seg (instance segmentation)

The segmentation family converts with the same recipe — the seg head
(bilinear-resize → depthwise-conv blocks → query·feature einsum) needed zero
extra workarounds. Third output `masks [1, Q, R/4, R/4]`: per-query FULL-FRAME
logit planes at stride 4; host does sigmoid > 0.5 (no per-box ROI plumbing).

| variant | input | queries | M4 Max GPU | `.aimodel` |
|---|---|---|---|---|
| seg-nano | 312² | 100 | **10.7 ms** | 123 MB |
| seg-small | 384² | 100 | **12.4 ms** | 123 MB |
| seg-medium | 432² | 200 | **18.3 ms** | 130 MB |
| seg-large | 504² | 200 | **22.5 ms** | 131 MB |
| seg-xlarge | 624² | 300 | **38.5 ms** | 139 MB |
| seg-2xlarge | 768² | 300 | **59.1 ms** | 141 MB |

All six gate on CPU and GPU: box set-match + **binary-mask IoU = 1.000** on
stable scenes. Gate-design finding: on busy scenes the deepest models flip 1–2
near-tie proposals between any two numerically-different executions — the
PyTorch reference itself flips **10 confident queries under 1e-4 input noise**
on the same image (measured, seg-xlarge / COCO 397133) — so the gate matches
identity by class + IoU + score ± 0.05 with a ≤ 2-flip budget, and compares
masks only for queries confident on both sides. The detection analog of the
LLM argmax margin rule, now needed on real images, not just noise probes.

<p align="center"><img src="https://huggingface.co/mlboydaisuke/RF-DETR-CoreAI/resolve/main/demo_seg_coco_cats.jpg" width="420" alt="RF-DETR-Seg nano on Core AI"></p>

## Split deployment & the ANE question (measured)

`export_rf_detr.py --split` emits `rfdetr-<v>_backbone.aimodel` (pure ViT) +
`rfdetr-<v>_head.aimodel` (deformable decoder, position encodings baked) —
the chain gates **bit-exact** vs the monolith, and the bundles are on the Hub
under `split/`. Purpose: per-stage compute units, i.e. the ANE-friendly
backbone on `.neuralEngine` while the gather-heavy head stays on `.gpu`
(`ObjectDetector(backboneAt:headAt:)` in CoreAIKit).

Measured on iPhone 17 Pro, iOS 27 beta: **the runtime does not engage the ANE
for these graphs** — the monolith under `.neuralEngine` preference falls back
wholesale, and even the pure-ViT backbone executes on the GPU delegate
(identical detection fingerprint to the GPU run, no ANE-compile pause, timing
equal at the same thermal state; the split costs ~5 ms of two-graph handoff).
GPU monolith stays the ship config. The split exists for the day ANE
placement starts working — the prize is thermal: the GPU throttles nano
25 → 75–103 ms at `thermalState=serious`, and the ANE does the same work at a
fraction of the power.

## The port in one lesson: deformable attention vs the Core AI stack

`grid_sample` has no Core AI lowering, but rfdetr 1.7.1 already ships a
gather-based bilinear fallback (their MPS path) — we force it on at export.
What actually fought back was four platform bugs, each pinned with a minimal
repro (details + repros in
[knowledge/conversion-guide.md](../../knowledge/conversion-guide.md#detection-transformers)):

1. **`aten.arange` with float args aborts the converter** (`bad_optional_access`).
   `gen_sineembed_for_position(…, d_model / 2)` passes dim as a *float*. Repro:
   `torch.arange(8.0)` in any graph. Fix: precompute `dim_t` as a Python-list
   constant (also deletes the runtime arange/pow/floordiv chain).
2. **int64-comparison bool chains make the runtime clobber unrelated live
   buffers.** The bilinear's `((ix0 >= 0) & (ix0 < W)).to(float)` corrupted the
   decoder LayerNorm output *two ops upstream* — even when that tensor is a
   graph output; `clone()`/`contiguous()` guards don't help. Fix: compute
   in-bounds masks in pure float arithmetic
   (`1 - (x - x.clamp(lo, hi)).abs().clamp(max=1)` is an exact 0/1 mask on
   integer-valued floats).
3. **`aten.floor`/`trunc`/`ceil` lower to IDENTITY on the GPU delegate**
   (CPU is fine; `round` rounds ties away instead of to-even). And the two
   obvious workarounds also fail: `div(x, 1, floor)` folds to identity, and
   `float→long→float` roundtrips get cast-cancelled by the converter (so does
   on CPU!). The one floor that survives every unit:
   **`torch.div(x * 2.0, 2.0, rounding_mode="floor")`** (×2/2 is a power-of-two
   scale — exact in fp).
4. **`torch._assert` on a data-dependent equality** trips
   GuardOnDataDependentSymNode under torch-2.11 non-strict export — no-op it
   for static-shape export.

Everything else — two-stage top-k + gather, windowed DINOv2, 16 SDPA blocks,
the iterative refinement loop — converted and gated clean on the first try
once those four were out of the way.
