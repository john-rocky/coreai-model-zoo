# YOLOX-S — Core AI

The zoo's **first single-stage anchor-free detector** and first **YOLO-family**
model — the lean CNN counterpart to the transformer-based
[`rf-detr`](../rf-detr/README.md) (the zoo's first detector). Megvii's
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0) running as a
single static `.aimodel` on every Apple compute unit. Where RF-DETR is **NMS-free**
(DETR set prediction), YOLOX is the classic **dense** detector: the host does
`score = obj · cls` + **per-class NMS** — the other half of the on-device detection
story.

Architecture (yolox_s.pth, 0.1.1rc0): **CSPDarknet** backbone (Focus space-to-depth
stem + SPP) → **PAFPN** neck → **decoupled** anchor-free head (separate cls / reg /
obj branches per FPN level). 8.97M params, 640² input, three strides 8/16/32 →
**8400 anchors** (80² + 40² + 20²). Upstream COCO accuracy is **40.5 AP** (Megvii's
reported number).

<!-- gen-cards:use-it begin id=yolox-s (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — run the kit's task op on this model
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let boxes = try await CoreAI.detect(inImageAt: url, options: .model("yolox-s"))
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [DetectCamera runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DetectCamera)
(real-time object detection on the zero-copy camera path):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/DetectCamera/DetectCamera.xcodeproj
# → Run, then pick "YOLOX" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/DetectCamera
swift run detect-cli --model yolox-s --image Resources/gate_image.jpg
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKitVision

let detector = try await KitDetector(catalog: "yolox-s")
let image = try ImageFile.load(imageURL)  // any image file → CGImage + EXIF orientation
let detections = try await detector.detect(in: image.cgImage)
// detections: [Detection] — label, score, normalized box (top-left origin)
```

The take-home is [`Examples/DetectCamera/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/main/Examples/DetectCamera/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; the CLI is an argument shell over it, and
the GUI runs the same detector per camera frame on a zero-copy pixel-buffer fast path.
YOLOX is a dense detector — `KitDetector` runs the obj·cls threshold + per-class NMS
host-side; the DETR family needs none. Same `detect(in:)` either way.

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` → product **CoreAIKitVision**
- Info.plist: `NSCameraUsageDescription` — only for the live camera; the snippet needs none
- Entitlements: none needed
- First run downloads the model — 0.0 GB (Mac) / 0.0 GB (iPhone) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Graph contract

```
input  "image"  [1, 3, 640, 640]  float32   BGR, 0-255, letterboxed (pad value 114,
                                             top-left aligned) — YOLOX-native, NO /255, NO mean/std
output "preds"  [1, 8400, 85]      float32   [cx, cy, w, h, obj, cls_0 .. cls_79]
                                             box DECODED to 640-space pixels; obj + 80 cls SIGMOID-ed
```

Grid + stride decode and the obj/cls sigmoids are **in-graph**
(`decode_in_inference=True`). Host decode: `score = obj · max_class`, threshold
(0.25), **per-class NMS** (IoU 0.45), then unletterbox the survivors (`box /= ratio`)
back to original-image pixels.

## Numerics gate (vs torch-fp32 oracle, COCO 000000039769)

Two ways, on cpu AND gpu: (1) raw head-tensor cosine + max|Δ| over the whole
`[1, 8400, 85]` output; (2) end-to-end detection set-match after the full host
post-process (every confident reference detection finds a same-class partner,
IoU ≥ 0.6, score within tol).

| dtype · unit | head cosine | max\|Δ\| | detection set-match |
|---|---|---|---|
| **fp32 · cpu** | **1.000000** | 4.6e-3 | **6/6, worst-IoU 1.000** |
| **fp32 · gpu** | **1.000000** | 6.3e-3 | **6/6, worst-IoU 1.000** |
| fp16 · cpu | 0.999990 | 45 (decoded px) | 6/6, worst-IoU 0.980 |
| fp16 · gpu | 1.000000 | 6.0 (decoded px) | 6/6, worst-IoU 0.997 |

The fp32 engine is **bit-clean** end to end. The fp16 max\|Δ\| is large only in
*absolute* terms because it lands on decoded box coordinates (≤ 640 px) and the
`exp(w,h)` — cosine stays ≥ 0.99999 and every detection still matches; it is the
documented detection near-tie noise class.

Detections (fp32, both units, identical): **cat 0.96 · cat 0.95 · remote 0.86 ·
remote 0.86 · bed 0.70 · couch 0.54**.

## Measured (macOS 27 beta, M4 Max, median of 80, weights resident)

| dtype · unit | median | best | throughput | `.aimodel` |
|---|---|---|---|---|
| **fp32 · GPU** | **4.80 ms** | 4.65 ms | **208 FPS** | 36.2 MB |
| fp32 · CPU | 57.4 ms | 46.6 ms | 17.4 FPS | 36.2 MB |
| fp16 · GPU | 5.21 ms | 4.22 ms | 192 FPS | 18.1 MB |

**Ship dtype is fp32** — the same call as RF-DETR. Detection has no
bandwidth-bound decode loop, so halving the weights to fp16 buys **nothing** on
the GPU (5.21 ms median is *slower* than fp32's 4.80, within noise) while adding
detection noise (worst-IoU 1.000 → 0.980). At 36 MB the size is a non-issue. fp16
is exported (`--dtype float16`) only for experiments.

> 4.80 ms / **208 FPS** on the M4 Max GPU makes YOLOX-S the zoo's fastest detector
> — a plain CNN with no gather-heavy deformable decoder maps almost perfectly onto
> the GPU delegate (RF-DETR nano 384² is 8.6 ms / 116 FPS by comparison).

**On-device (iPhone 17 Pro, iOS 27 beta, Release, GPU, live camera in DetectCamera):**
**~22 ms median inference** (≈45 model FPS), **35–40 FPS end-to-end**, thermal nominal —
measured with **Low Power Mode ON** (throttled; full-power is faster). The first load
specializes the static graph on-device in **~2.6 s** (no AOT needed). The on-device gate
reproduces the Mac fp32 oracle **exactly — 6/6 identical detections** (cat 0.96/0.96,
remote 0.86/0.86, bed 0.71, couch 0.54). The `.neuralEngine` preference also loads and
runs at ~21 ms (a plain CNN, unlike RF-DETR's gather-heavy head, so the ANE path is in
play).

## The port in one line: a clean CNN, no workarounds

Unlike RF-DETR (four platform bugs around deformable attention — float-arange
abort, int64-cmp buffer clobber, GPU floor=identity, data-dependent assert),
YOLOX is a **plain convolutional graph** — Focus strided-slice stem, SPP maxpools,
SiLU, the decoupled head, and the in-graph grid/stride decode all converted and
gated clean on the **first try**, no `coreai-torch` workarounds. The only friction
was *packaging*: build the model straight from `yolox.models` (avoiding
`yolox.exp`, which drags in OpenCV) under a one-line `cv2` stub, so the export env
stays OpenCV-free.

## ⬇️ Bundle

**[mlboydaisuke/YOLOX-CoreAI](https://huggingface.co/mlboydaisuke/YOLOX-CoreAI)** —
`yolox-s_float32.aimodel` (36 MB, fp32). Apache-2.0. CoreAIKit drop-in: `YOLOXDetector`
(`detect(in: pixelBuffer)` → `[Detection]`); the **DetectCamera** app downloads it on first
run. Gated fp32-clean on Mac and **device-verified on iPhone 17 Pro** (live in DetectCamera —
6/6 gate detections match the Mac oracle).

Convert yourself: [`conversion/export_yolox.py`](../../conversion/export_yolox.py) —
`--variant s --yolox-repo <YOLOX checkout> --weights yolox_s.pth`, gated
end-to-end by `--verify-image <img> --unit {cpu,gpu}`. The script also maps the
rest of the family (`nano` / `tiny` 416² · `m` / `l` / `x` 640²); `s` is the
canonical baseline. Needs `pip install loguru` and a
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) checkout + the
[`yolox_s.pth`](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth)
release weights; torch ≤ 2.11 to match coreai-torch 0.4.0.
