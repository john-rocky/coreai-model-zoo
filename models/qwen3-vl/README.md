# Qwen3-VL (2B / 4B / 8B, vision-language) — Core AI

**The zoo's first VLM family on Core AI**: image + text → text, end-to-end on
the GPU via the [pipelined-engine fast path](../../knowledge/pipelined-engine.md) —
no engine changes beyond the published static-inputs patch.

Architecture (`Qwen/Qwen3-VL-{2B,4B,8B}-Instruct`): a **pure-attention Qwen3
text decoder** (vocab 151 936, head_dim 128, 8 kv heads, SwiGLU, rope θ 5e6) +
a **ViT vision tower** (patch 16, temporal-2 duplicated frame, 2×2 spatial
merge → decoder hidden) with **DeepStack** (three merger outputs added to the
decoder hidden state at image positions after decoder layers 0/1/2). Positions
are **interleaved M-RoPE** — 3D (t,h,w), section [24,20,20].

The 4B and 8B drop in on the **same recipe** — the model overlay and exporter
are fully config-driven, so the only deltas are configuration:

| size | decoder | head | ViT (depth · hidden · deepstack) |
|---|---|---|---|
| **2B** | 28 L · h2048 · 16 q-heads · SwiGLU 6144 | tied fp16 | 24 · 1024 · [5,11,17] |
| **4B** | 36 L · h2560 · 32 q-heads · SwiGLU 9728 | tied fp16 | 24 · 1024 · [5,11,17] |
| **8B** | 36 L · h4096 · 32 q-heads · SwiGLU 12288 | **untied** | **27 · 1152** · [8,16,24] |

4B reuses the 2B path verbatim. 8B needed exactly one loader change — its head
is **untied** (a separate `lm_head.weight`), so the checkpoint remap routes
that tensor to the top-level module instead of under `model.`; its larger ViT
(depth 27, head_dim 72) is otherwise absorbed by config.

**⬇️ Converted `.aimodel` bundles:
[mlboydaisuke/Qwen3-VL-2B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3-VL-2B-CoreAI)** —
`gpu-pipelined/qwen3_vl_2b_instruct_decode_int8hu_s1/` (text decoder
LanguageBundle, ship config) + `gpu-pipelined/qwen3_vl_2b_instruct_vision/`
(fixed-grid vision encoder, fp16). Apache-2.0. The **4B**
([🤗 Qwen3-VL-4B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3-VL-4B-CoreAI))
and **8B**
([🤗 Qwen3-VL-8B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3-VL-8B-CoreAI))
bundles convert from the same recipe (`--hf-id Qwen/Qwen3-VL-4B-Instruct` /
`8B-Instruct`) and ship the same int8hu config.

<p align="center"><img src="https://huggingface.co/mlboydaisuke/Qwen3-VL-2B-CoreAI/resolve/main/demo.gif" width="300" alt="CoreAIChat Qwen3-VL demo on iPhone 17 Pro"></p>

<!-- gen-cards:use-it begin id=qwen3-vl-2b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
![Qwen3-VL 2B demo](https://huggingface.co/mlboydaisuke/Qwen3-VL-2B-CoreAI/resolve/main/demo.gif)
*Qwen3-VL 2B on iPhone 17 Pro — in the zoo's CoreAIChat app, real speed.*

## Use it

⚡ **One line** — this model is the default behind the kit's task op
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let caption = try await CoreAI.caption(imageAt: url)
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [VLChat runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/VLChat)
(GUI + CLI, one app for every vision-language model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/VLChat/VLChat.xcodeproj
# → Run, then pick "Qwen3-VL 2B" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/VLChat
swift run vlchat-cli --model qwen3-vl-2b --image sample.jpg --prompt "What is in this image?"
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit
import FoundationModels

let vlm = try await KitVisionModel(catalog: "qwen3-vl-2b")
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
- First run downloads the model — 3.3 GB (Mac) / 3.3 GB (iPhone) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## How a VLM rides a text-only engine

The pipelined engine knows nothing about images. The whole multimodal state
rides the **static-input hook** (`apps/coreai-pipelined-static-inputs.patch`)
plus an id-space trick — the graph stays `ids + positions → logits`:

- The host runs the vision encoder ONCE per image (448×448 → 196 merged
  tokens) and writes `image_embeds [196,2048]` + `deepstack_embeds [588,2048]`
  into two owned MTLBuffers the engine binds on every encode (~3.2 MB).
- The prompt's `<|image_pad|>` ids are rewritten to **extension ids**
  `V + slot` (slot 0..195). In-graph: `embed = ids < V ? table[ids] :
  image_embeds[ids - V]`; the three DeepStack adds gather
  `deepstack_embeds[k·196 + slot]` at image positions.
- **M-RoPE is derived in-graph from (ids, position) alone.** Image tokens
  self-locate: `slot = ids - V`, image start `s0 = pos - slot`, then
  `t = s0, h = s0 + slot//14, w = s0 + slot%14`. Text tokens use `pos -
  shift` where `shift = amount·(pos ≥ start)` — two `[1] i32` static inputs
  the host sets per conversation (an image consumes only max(H,W)=14 rope
  positions, so post-image text shifts by 196−14=182). The interleaved
  layout is three constant 0/1 masks over head_dim mixing three standard
  RoPE rotations (frequency j: j%3==1 → h, j%3==2 → w, else t, j<60).
- With zero embeds and `start = 1<<30` the decoder **is** a plain Qwen3 text
  LLM — same bundle, no images required.

KV cache stays the engine's native pair (pure attention, no extra states).
The vision tower is a separate plain `.aimodel` with ALL positional work
(bilinear pos-embed interpolation, 2D rotary) baked as constants for the
fixed grid: `patches [784,1536] → (image_embeds, deepstack_embeds)`.

## Measured (macOS 27 beta / iOS 27 beta, release, p=128 g=256, `COREAI_CHUNK_THRESHOLD=1`)

All three ship the **int8hu** config (int8lin per-block-32 body + untied absmax
int8 head, the `_s1` static-query bundle; the dynamic twin crashes the engine —
see below). Decode is bandwidth-bound, so tok/s scales ~inversely with bundle
size across the family.

| size · config | bundle | platform | prefill tok/s | decode tok/s | numerics |
|---|---:|---|---:|---:|---|
| **2B int8hu** | 2.3 GB | M4 Max | **191.0** | **187.6** | A 211-tok multimodal sweep 4/4 + B decode 16/16 + D HF-seeded (cos 0.99995) vs fp32-HF; engine ≡ python 24/24 |
| **2B int8hu**, iPhone 17 Pro (settled, unlocked) | 2.3 GB | iPhone | **33.5–34.6** | **33.3 typical (32.9–33.8; one thermal-dip 27.5)** | **nat 24/24 + multimodal oracle 24/24 × 8 runs — token-identical to M4 Max. ~92% of the naive BW ceiling** |
| **4B int8hu** | 4.7 GB | M4 Max | **93.3** | **92.2** | torch ladder ALL PASS vs fp32-HF (positions exact, vision cos 1.000, 36/36 layers cos 1.000, prefill top-1, decode 16/16); **engine ≡ python 24/24** (211-tok multimodal prompt) |
| **4B int8hu**, iPhone 17 Pro | 4.7 GB | iPhone | 10–15 | **14.0 cool → ~8.5 sustained** | **nat 24/24 + multimodal oracle 24/24 × 3 device runs** (token-identical to M4 Max). Thermally limited: the 4.7 GB/tok read peaks ~14 from a cool start, throttles to ~8.5 under sustained decode (cold spec 52.7 s, warm load 8–9 s) |
| **8B int8hu** (Mac-only) | 8.7 GB | M4 Max | **54.4** | **54.3** | torch ladder ALL PASS vs fp32-HF incl. untied head + depth-27 ViT (vision cos 1.0001, 36/36 layers cos 1.000, decode 16/16); **engine ≡ python 24/24** (211-tok multimodal prompt) |
| 2B int8lin (tied fp16 head) | 2.0 GB | M4 Max | 165.7 | 162.9 | same gate set PASS (D cos 0.99996) |
| vision encoder fp16 | 0.77 / 0.79 / 1.1 GB | both | — | 59–79 ms/image (Mac, 2B) | cos ≥ 0.9997 embeds / deepstack vs fp32-HF, all sizes |

- **8B is Mac-only**: its 8.7 GB decoder exceeds the iPhone increased-memory
  jetsam ceiling (~6.4 GB class). **4B (4.7 GB) runs on iPhone 17 Pro** — numerics
  24/24 every run, but decode is **thermally limited**: ~14 tok/s from a cool
  start, settling to ~8.5 under sustained load (2B at 2.3 GB holds 33 steady; the
  4B's 2× per-token read both halves the ceiling and heats faster). Needs the
  increased-memory entitlement for the 4.7 GB cold spec.
- 4B/8B are gated end-to-end like 2B: fp32-torch ladder (architecture EXACT) →
  **engine ≡ python 24/24** on the real 211-token house-scene prompt (the
  int8hu-through-runtime check — the engine reproduces the `_s1` bundle's greedy
  tokens token-for-token; both produce a correct house caption). The int8hu
  quantization spec is byte-identical to the 2B recipe and the head rule is
  **vocab-dependent, not size-dependent** (all three share vocab 151 936).
- The big-vocab head rule ([compression-reference](../../knowledge/compression-reference.md))
  holds at vocab 151 936: absmax `symmetric` int8 head gates clean, +15% on Mac
  over int8lin.
- Cold GPU specialization on device 12.3 s (no AOT needed), warm load 0.6–5 s;
  needs the increased-memory entitlement (2.3 GB class). TTFT for a 211-token
  image prompt ≈ 6 s on iPhone (S=1 prefill), ~1.1 s on M4 Max.
- A LOCKED iPhone screen collapses GPU priority for devicectl-launched runs
  (decode 0.5–13 tok/s, oscillating) — bench with the screen unlocked and awake;
  numerics still pass 24/24 even throttled.
- End-to-end sanity (full `.aimodel` pipeline, Mac): the oracle scene caption
  forks from the fp32-HF rollout only at token 41/51 (fp16-greedy class); a
  fresh snowman scene is described correctly ("three stacked, white, circular
  shapes").
- The decoder also ships a **dynamic-query twin** (no `_s1` suffix) that would
  enable true chunked prefill; it exports and gates in torch but crashes the
  engine at generate on this beta (`NSArrayM nil insert` — first dynamic-ids
  graph on this path). S=1 prefill ≈ decode tok/s, so TTFT for a 211-token
  image prompt is ~1.1 s on M4 Max regardless.

## Convert / verify

```
# decoder (fp16 / int8lin / int8hu) + _s1 gate twin + vision encoder.
# --hf-id selects the size; the recipe is identical across 2B / 4B / 8B.
python conversion/export_qwen3_vl_pipelined.py int8hu                       # 2B (default)
python conversion/export_qwen3_vl_pipelined.py int8hu --hf-id Qwen/Qwen3-VL-4B-Instruct
python conversion/export_qwen3_vl_pipelined.py int8hu --hf-id Qwen/Qwen3-VL-8B-Instruct
# fp32-HF torch ladder (position formula, vision cos, per-layer cos, decode top-1).
# Capture the oracle with the matching size, then run the ladder:
QWEN3VL_MID=Qwen/Qwen3-VL-4B-Instruct QWEN3VL_REF=_smoke/qwen3_vl_4b_ref.pt \
    python _smoke/qwen3vl_capture_ref.py            # py312 venv (transformers w/ qwen3_vl)
QWEN3VL_MID=Qwen/Qwen3-VL-4B-Instruct QWEN3VL_REF=_smoke/qwen3_vl_4b_ref.pt \
    python _smoke/test_qwen3vl_torch_ladder.py      # coreai-models venv
```

The torch re-authoring lives in
`coreai-models/python/src/coreai_models/models/macos/qwen3_vl.py` (model
overlay; see conversion/README.md). The position formula is verified EXACT
against HF `get_rope_index` before any conversion; the full gate ladder is
fp32 torch → fp16 `.aimodel` (Mac GPU) → engine → device.

## Try it

`apps/CoreAIChat` has a **Qwen3-VL mode with a photo picker**: pick an image,
ask about it. The vision tower runs once per attached image (~60-80 ms-class);
each turn re-prefills (S=1), so a 211-token image prompt streams its first
token in ~1 s on an M4 Max-class GPU.
