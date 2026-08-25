# Unlimited-OCR (3B-A0.5B MoE) — Core AI document OCR

**On-device document → structured markdown, end-to-end on Core AI.** A port of
[`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR) (MIT, tops OmniDocBench v1.6 at
93.92): drop a document image, get **markdown** — tables as HTML (`<table><tr><td>…`), formulas as
**LaTeX**, reading order, and `<|det|>` layout boxes. Japanese + English + multilingual. The zoo's
first on-device doc-OCR — the RAG ingestion layer (a page becomes structured text you can index).

Unlike the zoo's other VLMs, this rides the **stock `coreai.runtime` directly** — **no engine
patch, no static-input-buffer hook**. The decoder is driven on `inputs_embeds`, so it's a pure-export
port that the runtime can run as-is.

**⬇️ Converted `.aimodel` bundles:
[mlboydaisuke/Unlimited-OCR-CoreAI](https://huggingface.co/mlboydaisuke/Unlimited-OCR-CoreAI)** —
`vision/` (DeepEncoder, fp16, 762 MB) + `decoder/` (R-SWA MoE decoder, sym8, 3.2 GB, two functions
`prefill`+`decode` sharing one weight set) + `assets/` (embedding table + arrangement constants) +
`tokenizer/`. MIT. Catalog id: **`unlimited-ocr`**.

<!-- gen-cards:use-it begin id=unlimited-ocr (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — run the kit's task op on this model
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let markdown = try await CoreAI.read(documentAt: url, options: .model("unlimited-ocr"))
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [ReadDoc runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ReadDoc)
(GUI + CLI, one app for every document-OCR model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/ReadDoc/ReadDoc.xcodeproj
# → Run, then pick "Unlimited-OCR" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/ReadDoc
swift run readdoc-cli --model unlimited-ocr --image sample.png
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let reader = try await KitDocReader(catalog: "unlimited-ocr")
let markdown = try await reader.read(imageAt: imageURL)
// markdown: the document as structured text — tables as <table>/<tr>/<td>,
// <|det|> layout boxes, reading order — fully on-device
```

The take-home is [`Examples/ReadDoc/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/main/Examples/ReadDoc/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; the CLI is an argument shell over it, and
the GUI drives the same `KitDocReader(catalog:)` on the image you pick.
One `read(imageAt:)` call per page; chunk a PDF into page images first. The output keeps
the model's structural markup (tables as HTML, formulas as LaTeX, `<|det|>` boxes) —
strip or render it as your app prefers.

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` → product **CoreAIKit**
- Info.plist: none needed
- Entitlements: none needed
- First run downloads the model — 4.5 GB (Mac) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

<p align="center"><em>macOS app <code>apps/CoreAIOCR</code>: drop a document → structured markdown,
fully on-device. Reads a Japanese invoice (table + totals) and an English paper (abstract + LaTeX
equation + table) verbatim.</em></p>

## Architecture

- **DeepEncoder** (DeepSeek-OCR-derived, frozen): `image [1,3,640,640]` → **SAM-ViT** → **CLIP-ViT**
  (conditioned on SAM) → bridge `cat(clip[:,1:], sam.flatten)` → linear projector 2048→1280 →
  **100 visual tokens** `[1,100,1280]` (Base mode, image_size 640; token count scales with size).
  Exported as one fp16 `.aimodel`; all bicubic pos-embed interpolations baked to constants.
- **Decoder**: a **DeepseekV2 R-SWA MoE** — 12 L / hidden 1280 / **plain MHA** (10 heads × 128, no
  MLA) / **R-SWA** (every token attends the global prefix `[0,Lm)` ∪ a 128-token sliding tail) /
  greedy-softmax **MoE** (64 experts top-6 + a non-gated shared expert; layer 0 dense).

## The novel piece — flat-latency via a static-shape decode graph

R-SWA's promise is constant-memory, flat-latency decode. The trap: a growing `position_ids` / a
dynamic KV slice makes the runtime **recompile the Metal shader every step** — the
[Qwen3-Coder-Next freeze](../../knowledge/coreai-vs-mlx-speed.md), which on Metal 4 / macOS 26 actually
**faults** on the second distinct shape. The fix is to make the decode graph **fully static**:

- inputs are `inputs_embeds [1,1,1280]` + **`pos [1]`** (the absolute position as a runtime *value*);
- the KV cache write uses a **data-driven offset** (`mutable_slice_update` with `begin` built from the
  `pos` tensor) — verified to lower + run with no recompile across offset values;
- the fetch reads the **whole fixed StaticKVCache buffer** and applies the R-SWA visibility mask
  `(j≤i)&((j<Lm)|(j>i−W))` over `[0, buf_len)` — math-identical to a gather, but no dynamic slice.

No tensor shape ever changes → the engine compiles **once** → decode is **flat ~12.7 ms/token**
(`max/median 1.22×`, no freeze). The MoE FFN runs the `sym8` [`gather_qmm`](../../knowledge/compute-units-and-authoring.md)
kernel (reads only the routed 6/64 experts). The kickoff's dynamic constant-window *gather* was
math-proven (cos 1.0) but un-exportable: `sym_max` has no lowering, and the dynamic narrow recompiles.

**Prefill** is a separate static graph (`prefill` function, q_len = Lm, batched `gather_qmm`) that
writes the global prefix `[0,Lm)` in one pass; both functions share the same sym8 weights + KV state
in **one bundle**. SDPA is **not** externalized (the engine SDPA can't take a runtime mask), so it
lowers as plain ops; RMSNorm stays externalized.

## Pipeline (host side; see `assets/recipe.json`)

```
image → pad to 640², normalize mean=std=0.5
      → vision .aimodel                                  → [1,100,1280]
      → arrange: view 10×10 + image_newline per row (110) + view_seperator (111)
      → scatter into embed_tokens([BOS, <image>×111, "document parsing."])  → prefix [1,115,1280]
      → decoder prefill(prefix) → step0; then decode(token_embed, pos) loop → tokens
      → detokenize (keep special tokens; no_repeat_ngram=35)                → markdown
```

## Measured (M4 Max, GPU, stock `coreai.runtime`)

| stage | bundle | latency | numerics |
|---|---:|---|---|
| vision encoder (fp16) | 762 MB | per-image (one-shot) | per-token cos **1.000000** vs fp32-HF |
| decoder prefill (sym8, q=115) | 3.2 GB | ~63–81 ms (one-shot) | step-0 logit cos 0.99992 |
| decoder decode (sym8, q=1) | (same bundle) | **flat 12.7 ms/token (~79 tok/s)** | **0 flips / 9 sampled steps incl. steady-state** vs fp32 oracle |
| full image→markdown | — | 386 tok / ~8 s | fp32-oracle-identical structured OCR (`<table><tr><td>`, LaTeX, all numbers) |

`sym8` (symmetric per-block-32 int8) is the MoE quality floor; consistent sym8 throughout gives 0
argmax flips (a couple early-decode steps dip cos to 0.9989 — the sym8 prefix — but the output is
identical).

## Use / reproduce

- **App**: [`apps/CoreAIOCR`](../../apps/CoreAIOCR) — a macOS app driving the stock runtime directly
  (`AIModel` + `InferenceFunction.run(inputs:states:outputViews:)` with `MutableViews` for the KV
  cache; vision via `CoreAIKitVision.GraphModel`). Drop an image → markdown.
- **Convert**: [`conversion/unlimited_ocr/`](../../conversion/unlimited_ocr) — vision + decoder export,
  the arrangement/asset generator, and an end-to-end Python pipeline (the app's reference).
- **Knowledge**: [`knowledge/unlimited-ocr-rswa-static-decode.md`](../../knowledge/unlimited-ocr-rswa-static-decode.md).

## Notes

- **Appropriate input**: clean single-page documents (invoice / paper / report / table / formula),
  ~square or portrait, legible when fit to 640². Dense small-text scans (newspaper) want a tiled
  `crop_mode` vision export (not included; Base mode only).
- Per-config prefix length `Lm` is baked (here 115 = 111 visual + BOS + 3 prompt tokens).
- License: **MIT**. *Community port — not affiliated with Apple or baidu.*
