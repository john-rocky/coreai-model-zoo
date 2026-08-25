# MiniCPM-V-4.6 (1.3B, vision-language) — Core AI

**The strongest sub-2B open VLM, on-device.** A Core AI port of
[`openbmb/MiniCPM-V-4.6`](https://huggingface.co/openbmb/MiniCPM-V-4.6): image + text → text,
end-to-end on the GPU via the [pipelined-engine fast path](../../knowledge/pipelined-engine.md) —
no engine changes beyond the published static-inputs patch. Pick a photo, ask, stream the answer.

Architecture (`model_type: minicpmv4_6`): a **SigLIP So400m vision tower** (980px / patch 14 /
27 layers, gelu-tanh) with a **window-attention insert-merger @ layer 6** (2×2) + a **downsample-MLP
merger** (2×2) → ÷16 = **64 visual tokens** per 448px slice, spliced (`masked_scatter`) into the
text embeddings at `<image>` positions; and a **Qwen3.5-hybrid text backbone** (`qwen3_5_text`:
0.8B, 24 L, GatedDeltaNet linear-attn ×3 : full-attn ×1, head_dim 256, vocab 248094, **tied head**,
plain 1D positions). The backbone reuses the zoo's **existing `qwen3_5.py` overlay verbatim** (the
qwen3.6 hybrid); only the SigLIP tower + mergers were authored fresh.

**⬇️ Converted `.aimodel` bundles:
[mlboydaisuke/MiniCPM-V-4.6-CoreAI](https://huggingface.co/mlboydaisuke/MiniCPM-V-4.6-CoreAI)** —
**recommended (optimized)** `gpu-pipelined/minicpmv46_vlm_decode_int8hu/` (int8 body + untied **int8 head** →
**+48% decode** on iPhone 17 Pro) + `gpu-pipelined/minicpmv46_vision_int8lin/` (**int8** SigLIP, ~0.6 GB, half) ;
original `…_int8lin` decoder + fp16 `minicpmv46_vision` kept for compatibility. Apache-2.0.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c4baa524-5217-4bb3-a23f-b0acd6249bd4" width="300" alt="MiniCPM-V 4.6 on iPhone — a fridge photo becomes recipe ideas, fully on-device in CoreAIChat">
</p>
<p align="center"><em>Fridge photo → recipe ideas, fully on-device on an iPhone 17 Pro (CoreAIChat).</em></p>

<!-- gen-cards:use-it begin id=minicpm-v-4.6 (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — run the kit's task op on this model
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let caption = try await CoreAI.caption(imageAt: url, options: .model("minicpm-v-4.6"))
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [VLChat runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/VLChat)
(GUI + CLI, one app for every vision-language model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/VLChat/VLChat.xcodeproj
# → Run, then pick "MiniCPM-V 4.6" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/VLChat
swift run vlchat-cli --model minicpm-v-4.6 --image sample.jpg --prompt "What is in this image?"
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit
import FoundationModels

let vlm = try await KitVisionModel(catalog: "minicpm-v-4.6")
let session = LanguageModelSession(model: vlm)
let image = try ImageFile.load(imageURL)  // any image file → CGImage + EXIF orientation
let reply = try await session.respond(to: Prompt {
    prompt
    Attachment(image.cgImage, orientation: image.orientation)
})
// reply.content: the answer about the image, generated fully on-device
```

The take-home is [`Examples/VLChat/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/main/Examples/VLChat/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; the CLI is an argument shell over it, and
the GUI drives the same `KitVisionModel(catalog:)` behind a `LanguageModelSession`.
Multi-turn about the same image? Hold the `LanguageModelSession` and call `respond(to:)`
per turn. The photo picker / file chooser is your app's own chrome — `ImageFile.load`
(kit API) turns any image file into model input.

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` → product **CoreAIKit**
- Info.plist: `NSPhotoLibraryUsageDescription` — only if you use PhotosPicker
- Entitlements (iOS): `com.apple.developer.kernel.increased-memory-limit`
- First run downloads the model — 2.1 GB (Mac) / 2.1 GB (iPhone) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## How a VLM rides a text-only engine

The pipelined engine knows nothing about images. The whole multimodal state rides the
**static-input hook** (`apps/coreai-pipelined-static-inputs.patch`) + an id-space trick — the graph
stays `ids + positions → logits`:

- The host runs the vision encoder ONCE per image (resize 448, normalize `x/127.5−1`, CHW
  `[1,3,448,448]`) and writes `image_embeds [64,1024]` into one owned MTLBuffer the engine binds
  on every step (~0.13 MB).
- The prompt's `<|image_pad|>` ids (id 248056) are rewritten to **extension ids** `V + slot`
  (slot 0..63). In-graph: `embed = ids < V ? table[ids] : image_embeds[ids − V]`.
- **Positions are plain 1D** — no M-RoPE, no rope-shift. So this is the Qwen3-VL static-buffer
  recipe **minus deepstack and minus the M-RoPE machinery** (MiniCPM-V-4.6 dropped the perceiver
  resampler too; the connector is plain 2×2 merges + MLP).
- With zero embeds and no `V+slot` ids the decoder **is** a plain qwen3.5-hybrid text LLM — same
  bundle, no image required.

The backbone is the qwen3.5 **hybrid**, so the engine carries the SSM **conv + recurrent** states
alongside KV (the `expectFrequentReshapes` / extra-states path; cf. granite / lfm2). The vision
tower is a separate plain `.aimodel` with ALL positional work (bucketized pos-embed, window index
+ inverse) baked as constants for the fixed grid: `pixel_values [1,3,448,448] → image_features [64,1024]`.

## Measured (macOS 27 / iOS 27 beta, release, p=128 g=256, `COREAI_CHUNK_THRESHOLD=1`)

| config | bundle | platform | prefill | decode | numerics |
|---|---:|---|---:|---:|---|
| **VLM in-app A/B (same conditions)** | ~1.0→1.2 GB | iPhone 17 Pro | 50.6→70.4 | int8lin 51.5 → int8hu **70.0 (+36%)** | CoreAIChat, cool, back-to-back; the VLM bundle binds image_embeds each step (dilutes the head gain vs text core) |
| int8 head A/B (text core) | ~1.0 GB | iPhone 17 Pro | 47.8→69.6 | 46.1→**68.1 (+48%)** | PipelinedBench text core, nat+oracle 24/24 — head effect without image_embeds |
| text core int8 | ~1.0 GB | iPhone 17 Pro | 53.3 | **53.4** | **nat 24/24 + oracle 24/24** (engine ≡ python ≡ HF) |
| text core int8 | ~1.0 GB | M4 Max | 225.1 | **224.3** | `llm-benchmark` p128 g256 n3 (qwen3.5-0.8B class; VLM bundle decodes ~the same — llm-benchmark can't feed the image buffer so the text core is the Mac proxy) |
| vision encoder int8 | ~0.6 GB | Mac | — | ≈fp16 (compute-bound) | per-token cos **0.99980** vs fp32-HF |

**Optimization notes (2026-06-25).** ① **int8 head** (untie + quantize the 248k-vocab lm_head, block-32
symmetric/absmax) → **+48% decode** on iPhone (the fp16 head is ~half the per-token read). ② **int8 vision**
halves the encoder (0.6 GB); the encode is *compute-bound* so this is a size win, not speed — the ~2.7 s
first-image latency is the SigLIP graph's **cold compile**, so warm it with a dummy encode at load (then the
first photo is ~tens of ms). ③ **Chunked prefill** via a custom **fp32 gated-delta Metal kernel** is
Mac-validated (chunk=63, cos 0.9998, ~21.5× prefill) but **not yet shipped on-device**: the stock pipelined
engine binds dynamic-query bundles to S=1 and its multi-token path mis-computes for this 4-state GDN+kernel
bundle — a runtime-specialization gap (a host-side prefill loop is correct at S=1 but hits the same S>1 wall).

- **Gated end-to-end**: fp32-torch ladder EXACT (vision image_features cos 1.000000; full overlay
  logits cos 1.00004, top-5 identical to HF) → fp16/int8 `.aimodel` (Mac GPU) → **engine ≡ python**
  ("The capital of France is" → " Paris", 5/5) → **full VLM on engine** (vision.aimodel + decoder →
  correct image caption) → **iPhone 17 Pro** (text 24/24; image → accurate grounded description).
- Real-photo example (iPhone, kakigōri): *"a bowl of shaved ice ... chunks of mango ... a dark blue
  saucer ... a menu or a book, hinting at a café ... a wooden table"* — accurate, fully on-device.
- ~1.5 GB resident (vision fp16 + int8 decoder) = iPhone increased-memory jetsam-safe; cold spec ~3–5 s.
- The image numerics "fork" vs the fp32 rollout is a single near-tie (`" This"` vs `"\nThe"`) that
  reconverges — the fp16/int8-noise class (cf. granite's nat rollout), not a path error.
- `head_dim 256` dodges the Gemma4-12B decode scratch-heap; tied head (no untied-head step). The
  qwen3.5-hybrid GatedDeltaNet runs via ANE-compile-fail → GPU-fallback (benign; cf. granite / qwen3.6).

## Convert / verify

```
# vision encoder (fixed 448 grid; bakes window-index/argsort/bucketize as constants)
python conversion/export_minicpmv46_vision.py
# VLM decoder (input_ids -> logits + image_embeds static buffer; qwen3_5 hybrid core)
python conversion/export_minicpmv46_vlm_pipelined.py int8lin
# standalone text decode core (input_ids -> logits, in-graph embed + tied head)
python conversion/export_minicpmv46_decode_pipelined.py int8lin
# head-split core variant (inputs_embeds -> hidden, embed/head on the front-end)
python conversion/export_minicpmv46_core_decode_pipelined.py int8lin
```

The backbone reuses `coreai-models/.../macos/qwen3_5.py` verbatim; the vision tower re-authoring is
faithful to transformers `minicpmv4_6` (oracle + ladders in `_smoke/`). The fp32 oracle uses
synthetic deterministic pixels so model parity is decoupled from preprocessing (real images use the
repo's resize-448 + mean/std 0.5 normalization, mirrored on the Swift host).

## Try it

`apps/CoreAIChat` has a **MiniCPM-V 4.6 mode with a photo picker** (the 8th model in the picker), and
there's a standalone `MiniCPMVLM` app — pick an image, ask about it, stream the answer. The vision
tower runs once per attached image; each turn re-prefills (S=1).
