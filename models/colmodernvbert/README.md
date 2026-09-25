# ColModernVBERT (visual document retriever) — Core AI

**The zoo's first visual document retriever and first late-interaction (ColBERT / MaxSim)
multi-vector model.** A port of [`ModernVBERT/colmodernvbert`](https://huggingface.co/ModernVBERT/colmodernvbert)
(MIT, *ModernVBERT: Towards Smaller Visual Document Retrievers*,
[arXiv:2510.01149](https://arxiv.org/abs/2510.01149)): a compact **250M** encoder = a
**ModernBERT-150M bidirectional text encoder** + a **SigLIP2 vision encoder** (pixel-shuffle ×4),
with a `custom_text_proj` Linear(768→128) head that emits a **per-token L2-normalized 128-d
multi-vector**. There is no OCR and no pooling: a text query and a page *image* are each encoded
into token-level vectors and scored by **late interaction (MaxSim)** —
`score = Σ_q max_d ⟨E_q, E_d⟩` — so tables, charts and complex layouts are matched as pictures.

This completes the on-device RAG stack next to the text [`qwen3-embedding.md`](../qwen3-embedding/README.md)
(text→text dense) and [`qwen3-reranker.md`](../qwen3-reranker/README.md) (cross-encoder):
**embed → rerank → visual-retrieval**, all on device.

It is an **encoder, not a generator** — one bidirectional forward per graph, no KV cache, no LM
head, no sampling — exported / run like the other encoders (plain `.aimodel` via `AIModel.run`).
Two graphs (two encoders, shared backbone), one per modality.

<!-- gen-cards:use-it begin id=colmodernvbert (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

**New to Core AI? [Start with CoreAIKit 0.7.3](https://github.com/john-rocky/coreai-kit#readme).** Follow its requirements and first-run steps for `qwen3-0.6b`, then open the same release's [ChatDemo](https://github.com/john-rocky/coreai-kit/tree/0.7.3/Examples/ChatDemo). The README records the tested OS/SDK and download size; model and device coverage is stated per example.

▶️ **Run it (source)** — the [DocSearch runner](https://github.com/john-rocky/coreai-kit/tree/0.7.3/Examples/DocSearch)
(visual page search over bundled sample pages; the GUI (iPhone) adds tiled where-it-matched highlights):

```bash
git clone --branch 0.7.3 --depth 1 https://github.com/john-rocky/coreai-kit
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
open -a /Applications/Xcode-27.0.0-RC.app coreai-kit/Examples/DocSearch/DocSearch.xcodeproj
# → Run, then pick "ColModernVBERT" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/DocSearch
swift run -c release docsearch-cli --model colmodernvbert --query "monthly revenue trend"
```

Use Xcode build **27A266a** from the release's `.xcode-pin`; adjust the app path if your installation is named differently.

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKitEmbeddings

let retriever = try await VisualDocumentRetriever(
    catalog: "colmodernvbert")
var corpus: [VisualDocumentRetriever.PageEmbedding] = []
for url in pages {
    corpus.append(try await retriever.encode(page: ImageFile.load(url).cgImage))
}
let hits = try await retriever.retrieve(query: query, over: corpus, topK: pages.count)
// hits: pages ranked by MaxSim, best match first — no OCR, pages are matched as pictures
```

The take-home is [`Examples/DocSearch/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/0.7.3/Examples/DocSearch/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; the CLI is an argument shell over it, and
the GUI drives the same `VisualDocumentRetriever(catalog:)` with tiled per-page encoding.
Encode your corpus once and keep the `PageEmbedding`s — scoring a query is then host-side
MaxSim, no model call per page. `encodeTiled(page:)` localizes *where* a query matched.

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` (exact **0.7.3**) → product **CoreAIKitEmbeddings**
- Info.plist: `NSPhotoLibraryUsageDescription` — only if you use PhotosPicker to import pages
- Entitlements: none needed
- First run downloads the model — ~743 MB (Mac) / ~743 MB (iPhone) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Graph contracts

```
query: input_ids [1,32] i32, attention_mask [1,32] i32
   └─ colmodernvbert-query_<dtype>_s32_static.aimodel ─▶ query_embeddings [1,32,128]

doc:   pixel_values [1,1,3,512,512], pixel_attention_mask [1,1,512,512] i32
   └─ colmodernvbert-doc_<dtype>_s89_static.aimodel   ─▶ doc_embeddings [1,89,128]

host: MaxSim over the real query tokens × the 89 doc tokens (matmul → max → sum).
```

Per-token L2-norm + `attention_mask` masking are baked in-graph. MaxSim is a tiny host op.

### Query input

Right-pad the tokenized query to the **32-token** grid; slice back to the real length before
MaxSim. Queries are short, so ModernBERT's sliding-window(128) layers see the whole sequence
(= full attention) — no windowing on this path.

### Document input — single 512px tile (v1)

The doc graph is a **single 512×512 "global image" tile**. The 89-token sequence (CLS + image
markers + **64 `<image>` placeholders** + SEP) is **baked as a graph constant**, so the only
runtime inputs are the pixels. Host preprocessing mirrors Idefics3: resize longest edge ≤ 512,
pad to 512×512, ×1/255, normalize mean/std = 0.5, and supply `pixel_attention_mask` (1 real / 0
pad). 89 tokens < 128 → again full attention.

> **Single-tile v1.** Lightweight and iPhone-friendly, and accurate on typical pages. The model's
> full high-resolution mode (split the page into several 512px tiles + the global image, 800+ doc
> tokens, which engages real sliding-window attention) is a planned follow-up for dense
> small-print documents.

## Parity (Core AI engine vs. PyTorch `colpali_engine`, M4 Max GPU)

Per-token cosine of the 128-d multi-vectors:

| encoder | float32 | float16 |
|---|---|---|
| query | min/mean **1.000000** | min 0.999997 / mean 0.999999 |
| doc | min/mean **1.000000** | min 0.999994 / mean 0.999998 |

End-to-end: host **MaxSim reproduces `processor.score` exactly** (max |Δ| = 0.0000), the engine
ranking matches PyTorch on every clear-margin query, and the single-tile engine retrieves the
intended page **3/3** on a rendered-text corpus.

## Ship variants

`float16` (ship) and `float32`, per encoder:

| bundle | size |
|---|---|
| `colmodernvbert-query_float16_s32_static.aimodel` | 298 MB |
| `colmodernvbert-doc_float16_s89_static.aimodel` | 407 MB |
| `colmodernvbert-query_float32_s32_static.aimodel` | 595 MB |
| `colmodernvbert-doc_float32_s89_static.aimodel` | 813 MB |

iPhone footprint (both fp16 encoders) ≈ **705 MB**.

## Conversion notes (porting traps)

- **`inputs_merger` → cat-splice.** The stock image/text merger uses data-dependent ops
  (`num_image_tokens.sum()`, a `torch_compilable_check` assert, a boolean masked-assign) that
  create unbacked symints under `torch.export`. With a constant `input_ids` the 64 image tokens
  are a contiguous block, so we splice via `cat([text_before | image_embeds | text_after])`. An
  index/scatter merge instead lowers to `mps.scatter_nd`, which the GPU delegate **rejects** on
  rank-3 data (`3 != 2 + 0`).
- **`get_image_features` → reshape patch mask.** The stock path derives the vision patch mask
  from `pixel_attention_mask` with `aten.unfold` (**unsupported** by `coreai_torch`) and filters
  all-zero padding images with data-dependent boolean indexing. For the single real tile we
  compute the patch mask with a reshape (`[N,ph,16,pw,16].sum((2,4))>0`, numerically identical)
  and drop the filter.
- **`pixel_attention_mask` must be int32.** An int64 mask input fails at inference
  (`CoreAIError 3`); int32 is clean.
- **fp16 RoPE.** After casting the module to fp16, restore rotary `inv_freq`/`cos`/`sin` buffers
  to fp32 (θ = 160000's smallest frequency is an fp16 subnormal) — then fp16 ≈ fp32.

Both static patches are validated against the stock model (static-vs-stock per-token cosine
1.000000) before export.

Conversion: [`conversion/export_colmodernvbert.py`](../../conversion/export_colmodernvbert.py)
(`--phase query|doc`, `--dtype float16|float32`). Gates:
[`_smoke/gate_colmodernvbert_query_engine.py`](../../_smoke/gate_colmodernvbert_query_engine.py),
[`_smoke/gate_colmodernvbert_doc_engine.py`](../../_smoke/gate_colmodernvbert_doc_engine.py),
[`_smoke/gate_colmodernvbert_retrieval.py`](../../_smoke/gate_colmodernvbert_retrieval.py).
Download: [🤗 ColModernVBERT-CoreAI](https://huggingface.co/mlboydaisuke/ColModernVBERT-CoreAI).
