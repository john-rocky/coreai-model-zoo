# Porting a text embedder: what is different, and how the benchmark lies

Written 2026-08-25 from the Nemotron-3-Embed-1B port, against the two embedders this
repo already published. An embedder is not a small LLM, and three of the habits that
work for decoders quietly produce the wrong answer here.

## The product is the ranking, so gate the ranking

Every embedder passes a cosine gate. `cos ≥ 0.999` against the source is necessary and
says almost nothing about whether the model still retrieves the right document — an
embedding can move well inside that tolerance and still reorder results. Gate the
argmax and its margin, not only the cosine. `_smoke/gate_nemotron_embed_bundle.py` is
the shape: cosine, determinism across two runs, and query→document argmax with margin.

For the same reason, compression is gated on retrieval, not on cosine. That is cheap:
`KMeansPalettizer.prepare()` returns a model whose forward already carries the
compression, so the whole comparison runs on torch before any bundle exists.

## int4 fails on the interval, not on the mean

MIRACL-ja, 250 queries, 8000 documents with hard negatives, paired bootstrap against
the embeddinggemma-300m already in the catalog:

| | nDCG@10 | Δ vs incumbent | 95% CI | |
|---|---|---|---|---|
| fp32 | 0.8623 | +0.0377 | [+0.0181, +0.0578] | clear |
| **int8 k-means** | 0.8618 | +0.0372 | [+0.0173, +0.0576] | clear |
| int4 | 0.8452 | +0.0206 | [−0.0008, +0.0425] | **not separated** |

int8 costs 0.0005 — `compression.md`'s "int8 k-means is the floor that stays exact",
now confirmed on an encoder rather than a decoder. int4 still *looks* ahead by +0.021,
and reading the mean alone would ship it. The interval crosses zero, so at int4 the
claim the port exists to make no longer holds. top1 falls 0.784 → 0.740 as well, and
top-1 is what a RAG pipeline hands to the generator.

**A candidate's own size estimate is not a plan.** This one was scoped as "int4 layers
+ int8 embedding, ~700 MB". Measured, that configuration cannot carry the reason to
ship the model.

## k-means does not touch `nn.Embedding`, and for an embedder that is not a detail

Palettization is `F.linear`/`F.conv` only. A big-vocab embedder carries most of its
weight outside those: Nemotron's tied 131072×2048 table is **24% of the parameters**.
int8 k-means alone landed at **1947 MB** — 872 MB of int8 layers plus 1074 MB of
untouched fp32 table — over this repo's ~1.5 GB AOT shipping rule, and emitting
`Incompatible element type for ANE: expected fp16 … si8`, because an fp32 table cannot
be placed on the ANE at all.

Storing the table fp16 (the **fp16 embed + int8 transformer** config this repo already
ships for Qwen3.5) gives 1410 MB, halves the ANE errors, and costs *nothing measurable*
— the bundle gate is identical to six decimal places. That is not luck: the checkpoint
is bf16, 8 mantissa bits, and fp16 has 10 with the values well inside its range, so
fp16 represents them exactly. **Whenever the source is bf16, an fp16 table is free.**

Predict the bundle size from the parameter split before exporting. Both times the
prediction was wrong here, the arithmetic explained it immediately.

## Quality is the model's, not the port's

Measure retrieval on the native runtime (torch / sentence-transformers). Quality is an
attribute of the artifact; the port's job is to preserve it, and the port gate measures
that separately. Mixing them means a compression regression and a conversion bug are
the same number.

Each model runs **its own** documented recipe — prefixes, pooling, dimension are part
of a model, not knobs to equalize. Hold the corpus, queries, qrels, metric and sequence
budget identical instead.

## How a retrieval benchmark lies

Three ways, all found while building `_smoke/compare_embedders_retrieval.py`, and all
three made the numbers look *better*:

- **The corpus collapses to its own answer key.** A subsetting helper that kept only
  judged documents took NanoSciFact from 2919 to 55. Every model scored above 0.94 with
  recall@10 of exactly 1.0000. A benchmark that has lost its distractors does not look
  broken — it looks like all the models are excellent.
- **Hard negatives get filtered out as "not relevant".** In
  `MIRACLRetrievalHardNegatives` the score-0 rows are ~6.5k of 8.3k and they are the
  entire point. Dropping them and sampling distractors uniformly pinned everything above
  0.93 — and compressed a real +0.038 difference into +0.011 with a CI spanning zero.
  **That inverted a ship decision**, not just the precision of one.
- **Ceiling effects are invisible to a size check.** 8000 documents for 578 judged ones
  is 14×; any ratio guard passes it. Difficulty and size are different properties.

A guard for this must separate *"this script shrank the corpus"* (a bug — refuse) from
*"the dataset ships small"* (a weak discriminator — say so and continue). Conflating
them rejects standard collections: JaQuAD is natively 6.3× because it is QA-derived.

**What actually caught two of the three was not a guard.** It was the score being higher
than the published value for a model someone else had already measured. Check a new
harness against a number from the literature before trusting it — an independent
evaluation reproducing your ordering is worth more than any self-test.

## Incidental, but it will cost someone an hour

- `AIModel.load` and `InferenceFunction.__call__` are **coroutines**. Calling them
  synchronously returns a coroutine object and fails three lines later with an
  `AttributeError` naming neither.
- A bf16 checkpoint exported without an explicit cast produces a **bf16 graph output**,
  and `numpy`'s DLPack has no bf16: the run reaches its last line and dies there, which
  reads exactly like "this model will not port".
- Read `NDArray` with `.numpy()`. `np.asarray()` on it yields an object array.
