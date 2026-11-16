# Cross-runtime quality benchmarking: how to not measure your own harness

Written 2026-07-17, after a Gemma-4-E2B GSM8K comparison (Core AI / MLX / LiteRT-LM)
produced a Core AI "quality win" that was entirely an artifact of the harness.

This is a checklist for any *quality* comparison across runtimes. Speed benchmarks have
their own traps; this doc is about accuracy.

## The failure, concretely

Scores about to be published: Core AI 80% vs MLX ~20%. Both numbers were meaningless.

1. **The arms ran the model in different modes.** Gemma-4 has a configurable thinking
   mode. HF's `apply_chat_template` defaults to **thinking ON**. The same template
   rendered by swift-transformers (what `llm-runner` uses) comes out **thinking OFF**,
   and `llm-runner` exposes no flag to turn it on. One arm did chain-of-thought, the
   other answered directly — and the delta was about to be reported as runtime quality.

2. **The token budget truncated the thinking arm.** Thinking-ON Gemma-4-E2B spends ~250
   tokens reasoning before it answers; a GSM8K item needs **419–479 tokens**. The budget
   was **512** — right at the cliff. Easy items fit; hard ones were cut off mid-thought,
   and the answer extractor then scraped a stray number out of the reasoning text.
   Measured: same build, same weights → **~20% at 512, correct when given room.**

**A truncated reasoning arm is indistinguishable from a bad model.** Nothing in the log
says "truncated" — you get a confident wrong number.

The two defects interacted so as to hide each other: the arm we had *handicapped*
(Core AI, thinking off) was the one that *scored well*, because direct answers fit in
512. The harness manufactured the result we would have liked.

3. **Provenance.** Three of the four numbers in the table (bf16 92 / LiteRT 88 /
   MLX 78) had no stored report and no recorded budget or mode. Inherited numbers are
   not measurements. If you cannot re-run it, do not cite it.

## Checklist before believing any cross-runtime quality number

- **Same checkpoint.** Not "both int4" — the same file. See `##bits-are-not-a-spec`.
- **Same mode.** Thinking/reasoning defaults differ *per template renderer*, not just
  per model. Verify by grepping the raw generations for the thinking marker
  (`<|channel>thought` on Gemma-4) — do not trust the template source.
- **Budget ≥ 2× the observed worst case.** Measure the worst case first (generate a few
  items, count tokens). Never set the budget from the *typical* length.
- **Check the truncation rate explicitly.** Count generations that hit `max_tokens`
  without emitting the answer marker. If it is not ~0, the score is a budget artifact.
- **Probe-item parity before the full run.** One item through every arm; compare prompt
  token count, output token count, and answer. Ours: Core AI 76→195, MLX 75→197, both
  correct. If the probe doesn't match, the run won't mean anything.
- **Store a report per run** with `n`, `max_tokens`, mode, checkpoint, and the per-item
  preds. A number without its conditions is not reusable.

## Bits are not a spec

"int4" named three different products in this comparison. Google publishes four QAT
checkpoints for Gemma-4 and they are not interchangeable:

| variant | what it is | who uses it |
|---|---|---|
| Unquantized QAT (Q4_0) | half-precision weights from the QAT pipeline, "for custom downstream compilation and research" | Core AI, our MLX build |
| Mobile-optimized (**wNa8o8**) | "targeted **2-bit decoding layers**, optimized **KV caches**, and **static activations**" | LiteRT-LM `.litertlm` |
| GGUF (Q4_0) | ready-to-deploy | llama.cpp etc. |
| Compressed Tensors (w4a16) | vLLM | server |

The mobile variant is a **co-designed weights+runtime package**, not a bit-width. It
differs on three axes at once (2-bit layers → fewer bytes/token; optimized KV cache →
less traffic *and* smaller footprint; int8 activations → a different arithmetic path).
Comparing it to a generic Q4_0 build and calling the delta "runtime speed" credits the
engine with what is substantially the checkpoint's doing.

The naive bandwidth sanity check cannot rescue you here: Gemma-4 *gathers* its PLE, so
model size ≠ bytes/token (MLX at 181.9 tok/s × 3.3 GB = 600 GB/s would exceed the M4
Max's 546 GB/s peak — proof that no arm reads its whole file per token).

**To build a matched pair:** compile every arm from the *unquantized QAT* checkpoint
yourself, at the same block size. For MLX:

```
mlx_lm.convert --hf-path <qat-q4_0-unquantized> --mlx-path <out> -q --q-bits 4 --q-group-size 32
```

matching Core AI's int4lin per-block-32. Then weights, recipe, and block size are equal
and the runtime is the only variable.

## Ops notes

- HF python downloads stall (xet). Use `curl -C -` against
  `https://huggingface.co/<repo>/resolve/main/<file>`; check `x-linked-size` for the
  real size. `HF_HUB_DISABLE_XET=1` also helps.
- The QAT unquantized checkpoint's big blob is often already in the HF cache — hardlink
  it and `curl` only the small configs rather than re-downloading 10 GB.
- `llm-runner` on a gemma4 `tbl` bundle needs `--raw-dir <ple dump>` (PLE static inputs)
  and `COREAI_CHUNK_THRESHOLD=1` (the S=1 graph cannot take a multi-token prompt), plus
  `--warmup exact --warmup-length 1` (the default warmup prefills 256 → fatal on S=1).
- A bundle with no `chat_template` anywhere silently falls back to **raw completion**.
  `--apply-chat-template` defaults to true and does *not* warn when there is nothing to
  apply. Check `tokenizer/tokenizer_config.json` for a `chat_template` key and for
  `chat_template.jinja` next to it; if absent, the model never sees turn markers.
