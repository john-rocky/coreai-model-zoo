# Compression (palettization & quantization)

**TL;DR for LLM decoders: int8 k-means palettization is the floor that stays exact when applied
across the whole transformer; whole-model int4 degrades. SELECTIVE 4-bit works: k-means int4 on
the FFN + lm_head only (attention/embeddings kept ≥int8/fp16) measured top-1 exact and is the
shipping iPhone-GPU config (via custom fused kernels — see
[`custom-metal-kernels.md`](custom-metal-kernels.md)).**

## int8 k-means > whole-model int4 (for these models)

Across Gemma 4 E2B and Qwen3.5, linear int4 and k-means int4 both flip next-token argmax vs the
HF reference; **int8 k-means palettization reproduces HF top-1 exactly** at ~half the fp16 size.

- k-means fits a per-group lookup table to the actual weight clusters → tracks non-uniform weight
  distributions far better than symmetric per-block int4 at the same bit width.
- **Finer groups are the main int4 lever** (group32 → group8 helps), but still don't reach exact.
  Per-channel scale is marginal or harmful.
- Sensitivity is broadly distributed; for Gemma 4 the **gate/up MLP projections must be int8** for
  exactness (keeping them at 4-bit caps accuracy regardless of other layers).
- k-means palettizes **`F.linear`/`F.conv` weights only**, so RMSNorm/RoPE params stay full
  precision automatically — exactly what you want given their wide range.

Recommended LLM recipe: **int8 k-means, group 32, all projections**; keep tied lm_head + 1-D conv
(SSM) full precision. Sizes seen: Gemma 4 E2B core 7.0 GB fp32 → 3.5 GB fp16 → **1.9 GB int8**;
Qwen3.5-0.8B **969 MB**, -2B **2.2 GB** (fp16 embed + int8 transformer, single bundle).

## Palettization × stateful export composes

Palettizing the **stateful** decode core (mutable KV/SSM state + dynamic prefill+decode graph)
works: read the export spec (reference inputs / dynamic shapes / state names) from the ORIGINAL
model first (the finalized palettized model loses that method), palettize, then drive
`export_to_coreai` with that spec. Verified top-1-exact for both Gemma 4 (dual-KV) and Qwen3.5
(hybrid 4-state).

## Embedding tables (the on-device memory problem)

Big-vocab models have huge embedding tables (Gemma 4's per-layer table is 9.4 GB fp32). For device:
- The decode **core** keeps these tables OUT of the graph (gathered on a front-end).
- The front-end gather table compresses with **plain int8 per-row dequant-gather**
  (`q_table[ids].to(fp16) * scale[ids]`) — k-means is `F.linear`-only so it doesn't apply to a
  gather, and the iOS palettized-embedding custom op doesn't lower on macOS. int4 gather has no
  clean path today; **int8 is the practical floor** for embedding gather too.

## Per-channel int8 on a big-vocab LM head: use per-block-32

Weight-only symmetric int8 with **per-channel** scales (`granularity: per_channel, axis: 0`) on
MiniCPM5's 130560-row untied head produced bundles whose rows from vocab id ~65024 up score ~0
through the engine — `<|im_end|>` among them, so chat turns never ended — while rows below match
fp32 to ~0.01 and a low-vocab parity gate reads 24/24. Reproduced bit-for-bit by a fresh export on
`coreai-torch` 0.4.1 / `coreai-opt` 0.2.1; the same YAML with `granularity: {type: per_block,
block_size: 32}` has no dead rows, and on the Mac GPU it also lands on the fast quantized-matmul path
(1B: 246.6 vs 53.7 tok/s decode; 2B: 127.6 vs 25.6). Which component cuts the rows is not
established. Ship per-block-32 for int8 heads of this size, and gate the stop token
([`minicpm5-1b.md`](minicpm5-1b.md), 2026-09-09 section).

## Via the CLI

`coreai.llm.export <model> --compression int8` routes a new macOS int8 k-means preset through the
decode-core signature (palettizes the *extracted* core, not the `input_ids→logits` forward) for
models that expose `export_core()`. Other models keep the standard quantization path.
