# Holo2-4B — Core AI

[🤗 mlboydaisuke/Holo2-4B-CoreAI](https://huggingface.co/mlboydaisuke/Holo2-4B-CoreAI) · Apache-2.0 · base [Hcompany/Holo2-4B](https://huggingface.co/Hcompany/Holo2-4B)

H Company's **computer-use / GUI-grounding** VLM: given a screenshot + an instruction
("click the submit button") it predicts the **click coordinates / locates the UI element**
(SOTA UI localization). Built on the **Qwen3-VL-4B** backbone, converted to Apple **Core AI**.
The zoo's **first GUI-grounding / computer-use model**, and a worked example of riding an existing
zoo pipeline: Holo2-4B is byte-identical to Qwen3-VL-4B, so the conversion is the stock
`export_qwen3_vl_pipelined.py` with `--hf-id Hcompany/Holo2-4B` — no model-code changes.
Catalog id: **`holo2-4b`**.

<!-- gen-cards:use-it begin id=holo2-4b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

**New to Core AI? [Start with CoreAIKit 0.7.1](https://github.com/john-rocky/coreai-kit#readme).** Follow its requirements and first-run steps for `qwen3-0.6b`, then open the same release's [ChatDemo](https://github.com/john-rocky/coreai-kit/tree/0.7.1/Examples/ChatDemo). The README records the tested OS/SDK and download size; model and device coverage is stated per example.

▶️ **Run it (source)** — the [VLChat runner](https://github.com/john-rocky/coreai-kit/tree/0.7.1/Examples/VLChat)
(GUI + CLI, one app for every vision-language model in the catalog):

```bash
git clone --branch 0.7.1 --depth 1 https://github.com/john-rocky/coreai-kit
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
open -a /Applications/Xcode-27.0.0-RC.app coreai-kit/Examples/VLChat/VLChat.xcodeproj
# → Run, then pick "Holo2 4B" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/VLChat
swift run -c release vlchat-cli --model holo2-4b --image screenshot.png --prompt "Localize an element on the GUI image according to my instructions and output a click position as Click(x, y) with x num pixels from the left edge and y num pixels from the top edge. Instruction: click the Submit button."
```

Use Xcode build **27A266a** from the release's `.xcode-pin`; adjust the app path if your installation is named differently.

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit
import FoundationModels

let vlm = try await KitVisionModel(catalog: "holo2-4b")
let session = LanguageModelSession(model: vlm)
let image = try ImageFile.load(imageURL)  // any image file → CGImage + EXIF orientation
let reply = try await session.respond(to: Prompt {
    prompt
    Attachment(image.cgImage, orientation: image.orientation)
})
// reply.content: "Click(x, y)" in 0-1000-normalized coordinates for a grounding prompt,
// or a plain answer for a normal question - all generated on-device
```

The take-home is [`Examples/VLChat/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/0.7.1/Examples/VLChat/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; the CLI is an argument shell over it, and
the GUI drives the same `KitVisionModel(catalog:)` behind a `LanguageModelSession`.
Holo2 is a GUI-grounding model: feed a screenshot and H Company's localization prompt
(see the card's grounding section) and it returns `Click(x, y)` in 0-1000-normalized
coordinates — multiply by `imageSize / 1000` for pixels. It also answers free-form
questions like its Qwen3-VL base.

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` (exact **0.7.1**) → product **CoreAIKit**
- Info.plist: `NSPhotoLibraryUsageDescription` — only if you use PhotosPicker
- Entitlements (iOS): `com.apple.developer.kernel.increased-memory-limit`
- First run downloads the model — ~5,484 MB (Mac) / ~5,484 MB (iPhone) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Grounding prompt (the model's real job)

Holo2 answers free-form questions about an image like any VLM, but its specialty is
localization: give it a screenshot and H Company's localization prompt, and it returns a
click point.

```
Localize an element on the GUI image according to my instructions and output a click
position as Click(x, y) with x num pixels from the left edge and y num pixels from the
top edge. Instruction: click the Submit button.
```

The reply is `Click(x, y)` in **0–1000-normalized coordinates** (Qwen-VL convention):
multiply by `imageWidth / 1000` and `imageHeight / 1000` for pixels. Verified through the
kit path on a synthetic 800×600 settings screen: `Click(511, 841)` → (409, 505) px, dead
center of the Submit button at (400, 505).

## Parity (vs fp32 HF oracle, Core AI GPU engine)

| stage | metric |
|---|---|
| **vision** (`holo2_4b_vision`) | image-embeds cos **0.999983**, deepstack cos **0.999989** — PASS |
| **decoder** (`holo2_4b_decode_int8lin_s1`) | S=1 sweep **4/4**, **16/16** decode steps token-exact, HF-seeded match — PASS |

## Contents
- `gpu-pipelined/holo2_4b_decode_int8lin_s1/` — decode bundle (static query=1, per-block-32 int8
  linear body). Rides Apple's `coreai-pipelined` GPU engine and **specializes on-device — no AOT**
  needed (the static decode graph is cheap to specialize, unlike a dense 4B *dynamic* bundle).
- `gpu-pipelined/holo2_4b_vision/` — fixed-grid vision encoder `.aimodel` (fp16): `patches
  [784,1536] -> (image_embeds [196,2560], deepstack [3,196,2560])`. Run once per image.

## Conversion

- **Stock Qwen3-VL pipeline.** `coreai-models/.venv/bin/python conversion/export_qwen3_vl_pipelined.py
  int8lin --hf-id Hcompany/Holo2-4B` → decoder (+ `_s1` gate twin) + vision. text hidden 2560 /
  36 layers / 8 KV / head_dim 128 / vocab 151936; vision `qwen3_vl` depth 24.
- **Why Holo2 (not Holo1.5 / Holo-3.1):** Holo evolved 1.5 (Qwen2.5-VL) → 2 (Qwen3-VL) → 3.1
  (Qwen3.5-VL). Holo2 is the generation whose backbone matches the zoo's shipped Qwen3-VL pipeline,
  so it drops in with an HF-id swap; Holo-3.1 would need a new `qwen3_5_vision` tower.
- int8lin body, fp16 vision. (int8hu — untied absmax int8 head — is the optional head-quality upgrade,
  same as the other pipelined riders.)

## Run

In the zoo's **CoreAIChat** app: pick **Holo2 4B**, attach a screenshot, and ask where an element
is / what to click — it grounds the instruction to the image and returns the location. Rides the
same on-device path as [`qwen3.5`](../qwen3.5/README.md)'s VLM siblings (Qwen3-VL decoder + vision tower).
