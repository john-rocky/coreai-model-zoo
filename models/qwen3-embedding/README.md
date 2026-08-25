# Qwen3-Embedding-0.6B — Core AI

The open, **multilingual, instruction-aware, Matryoshka** text embedder running as a single
static `.aimodel`, completing the on-device RAG stack: **embed → (rerank) → generate**, all
local and private. [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
(Apache-2.0) is the instruction-aware / MRL complement to the already-shipped
[EmbeddingGemma-300m](https://huggingface.co/mlboydaisuke/embeddinggemma-300m-CoreAI).

> **Corrected 2026-08-25.** This card used to call it "current open SOTA-class for its size on
> multilingual MTEB (incl. Japanese)". Measured here on three retrieval collections under one
> protocol, it is **last of the three embedders this repo can run**, on English and on Japanese:
>
> | collection | Qwen3-Embedding-0.6B | EmbeddingGemma-300m | Nemotron-3-Embed-1B |
> |---|---|---|---|
> | NanoSciFact (en, 50 q) | 0.687 | **0.864** | 0.765 |
> | JaQuAD (ja, 250 q) | 0.570 | **0.621** | 0.616 |
> | MIRACL-ja (ja, 250 q, hard negatives) | 0.792 | 0.825 | **0.862** |
>
> nDCG@10; every gap against Qwen3-Embedding is separated at 95% by a paired bootstrap. An
> independent evaluation reaches the same ordering on Japanese
> ([sionic-ai Nano-BEIR](https://huggingface.co/blog/sionic-ai/eval-sionic-nano-beir):
> EmbeddingGemma 0.590 leads Japanese, and the two are within 0.004 on English aggregate).
> Reproduce with `_smoke/compare_embedders_retrieval.py`; the numbers are in
> `_smoke/results/embedder_retrieval_2026-08-25.json`.
>
> What still stands, and is why this model stays in the catalog: it is the only one of the three
> that takes a **task instruction** in the query, and the only one with **Matryoshka** truncation
> (1024 → 32 dims, re-normalize on the host, zero ranking flips measured). Those are capabilities
> the other two do not have. The retrieval-accuracy claim was the part that was not measured.

**This is an encoder, not a generator** — one forward over the (right-padded) input returns one
pooled vector. No autoregressive loop, no KV cache, no LM head, no sampling. It runs like the
vision encoders: a plain `.aimodel` via raw `AIModel.run`, **not** the pipelined generate engine.

Architecture (`model_type: qwen3`): standard **Qwen3-0.6B-Base** backbone — hidden 1024, 28
layers, 16 query / 8 KV heads (GQA), head_dim 128, per-head q/k-norm, SwiGLU, RoPE θ=1e6, full
causal attention, vocab 151669, tied embeddings. The embedding head is **last-token (EOS)
pooling → L2-normalize**, both baked into the graph, so one call returns a ready unit vector.

## Graph contract

```
input  "input_ids"       [1, S]    int32   right-padded to the grid S (pad id 151643)
input  "attention_mask"  [1, S]    int32   1 over real tokens, 0 over padding
output "embedding"       [1, 1024] fp16    L2-normalized; MRL-truncatable to 32–1024
```

Shipped grid **S = 512** (covers typical RAG chunks). The grid is an export-time choice
(`--seq-len`); a smaller grid is proportionally faster for short queries (see below).

**Host recipe** (mirror exactly; everything else is in-graph):
- **Query** → prepend `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:`. **Document** → no prefix. (Asymmetric instruction prefix; host-side only.)
- Tokenize, **right-pad** to S (truncate longer text at S). Last-token pooling under the causal
  mask is right-pad safe — real tokens never attend to trailing pads, so the pooled state is
  identical with or without padding.
- Run → 1024-d unit vector. Similarity = cosine = plain dot product (vectors are unit-norm).
- **MRL**: to shrink, slice the first D dims (32 ≤ D ≤ 1024) and **re-L2-normalize on the host** —
  no separate export per dim.

## Measured (macOS 27 beta, M4 Max GPU, fp16, warm)

| grid S | ms / embedding | embeddings/s | cold load |
|---|---|---|---|
| 256 | **25.0 ms** | ~40 | 1.9 s |
| 512 | **44.6 ms** | ~22 | 1.6 s |

The fixed grid computes all S positions regardless of real length, so latency scales ~linearly
with S — pick the smallest grid that covers your text (256 for queries / short chunks, 512 for
longer passages). Embeddings are cosine-sensitive but **not** bandwidth-bound like LLM decode, so
fp16 is the ship dtype: it gates essentially bit-exact (below) at half the fp32 footprint.

## Numerics gate

**Torch ladder** (`conversion/export_qwen3_embedding.py`, vs the official `sentence-transformers`
pipeline, fp32): per-text embedding cosine **1.000000** (padded grid and unpadded both exact);
the converted graph reproduces the official model's **retrieval order** (each query's top-1 doc
matches the official model, clear top1−top2 margins 0.32–0.79); and **MRL** truncation to
512 / 256 / 128 + renorm preserves every ranking (zero flips). Near-ties below a 0.05 cosine
margin are reported, not gated (the argmax-margin rule, applied to retrieval).

**Engine gate** (`_smoke/gate_qwen3emb_engine.py`, `.aimodel` on the GPU delegate vs the torch
reference): end-to-end (host tokenize → `AIModel.run` → embedding) cosine **0.999998**, output
norm ≈ 1.000 (in-graph normalize confirmed), retrieval order identical. fp16 does not overflow
(unlike Gemma3 — no NaN).

## ⬇️ Bundle

**[mlboydaisuke/Qwen3-Embedding-0.6B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3-Embedding-0.6B-CoreAI)**
— `qwen3-embedding-0.6b_float16_s512_static.aimodel` (~1.1 GB) + `reference.json` (torch
reference embeddings + cosines for the parity test) + `tokenizer/`. Apache-2.0.

Convert yourself: [`conversion/export_qwen3_embedding.py`](../../conversion/export_qwen3_embedding.py)
(`uv run conversion/export_qwen3_embedding.py --dtype float16 --seq-len 512 --output-dir out`).

## CoreAIKit (Swift)

Runs out of the box with [CoreAIKit](https://github.com/john-rocky/coreai-kit) — it downloads this
repo on first use and applies the query/document prompts in-process:

```swift
import CoreAIKitEmbeddings

let embedder = try await TextEmbedder(model: .qwen3Embedding0_6B, prompts: .qwen3Embedding)
let query = try await embedder.embed(query: "What is the capital of Japan?")
let doc   = try await embedder.embed(document: "Tokyo is the capital and largest city of Japan.")
let score = TextEmbedder.cosineSimilarity(query, doc)   // unit vectors → dot product = cosine
```

## The port in one lesson: an encoder, and two dtype traps

1. **No generate engine.** Pooling (last-token) + L2-normalize live *in the graph*, so the host
   just feeds `input_ids`/`attention_mask` and reads a unit vector — the
   `sentence-transformers` module chain (`Transformer → Pooling(last-token) → Normalize`) traces
   straight through, exactly like the EmbeddingGemma export.
2. **Load fp32, not the bf16 checkpoint.** The checkpoint is bf16; tracing it under
   `autocast(fp16)` makes Qwen3's RMSNorm fp32-roundtrip (`hidden_states.to(float32)`) collide
   with autocast's `_assert_tensor_metadata` (expected fp16, got bf16/fp32). Fix: load the model
   in **fp32** (`model_kwargs={"torch_dtype": float32}`) — clean reference — then…
3. **`module.half()`, not `autocast`, for the fp16 graph.** Casting the module to true fp16 and
   tracing plainly sidesteps the autocast assert entirely; Qwen3's RMSNorm still upcasts to fp32
   internally, so the norm stays numerically safe. (EmbeddingGemma could autocast because Gemma3's
   RMSNorm is shaped differently; Qwen3 cannot.)
