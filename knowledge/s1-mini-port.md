# S1-mini port — a task model, and the gate that could not see it

**S1-mini** by **Superwhisper** (Apache-2.0 + a naming term) is a Qwen3-0.6B finetune that
rewrites raw ASR transcripts as clean written text. Card: [`models/s1-mini/`](../models/s1-mini/README.md).
Exporter: [`conversion/export_s1_mini_decode_pipelined.py`](../conversion/export_s1_mini_decode_pipelined.py).
Task gate: [`_smoke/gate_s1_mini_task.py`](../_smoke/gate_s1_mini_task.py).

## Reuse: the whole port is one exporter and one gate branch

`model_type: qwen3`, shape byte-identical to `Qwen/Qwen3-0.6B`. The stock
`coreai_models.models.macos.qwen3` class already handles both non-obvious pieces —
q/k/v fusion and q_norm/k_norm fusion, in `_mutate_state_dict` — so no model code was
written. The export script is the plain dense KV-only decode template with
`Qwen3ForCausalLM` swapped in, and `coreai_gate.py` gained a `qwen3` arch branch of the
same shape as its `nanbeige`/llama one.

`ARCH` in `coreai_gate.py` is scanned **in insertion order** by `detect_arch`, so the new
generic `"qwen3"` key goes **last**: put it first and every `qwen3.5` / `qwen3_6_moe`
bundle silently routes to the dense branch and gates against the wrong oracle. The
S1-mini bundle name contains neither "qwen" nor "3", so it needs an explicit
`ALIASES` entry (`s1_mini` / `s1-mini` → `qwen3`) rather than substring luck.

## The finding: a passing conversion gate said nothing about the task

Both int8lin and int4lin passed `coreai_gate.py` **16/16 token-exact vs the fp32 oracle**
on the standard free-run continuation prompt. int4lin is nevertheless unshippable: over ten
number-heavy transcripts in the model's own input format it produced `$2,345` for `$23,450`
and `177` for `107`, and dropped "tomorrow" from a time normalization.

This is not a criticism of the conversion gate — it measures conversion fidelity, and it
measured it correctly. It is a statement about **what a conversion gate's green means for a
task model**. A free-run continuation exercises the base model's language prior. A
task-tuned 0.6B is almost all task; the prior is the part quantization damages last. So on
this class of model the continuation gate is close to blind to the thing being shipped.

The rule that follows: **for a single-task model, a conversion gate is necessary and never
sufficient — pair it with a gate that speaks the model's own input format.** The zoo's
multimodal ports already do this (LFM2.5-VL's 9-case suite, North-Micro-Vision's 9/9); the
same discipline belongs on a text model whose task is narrow.

## Fixtures: the upstream card's examples do not reproduce

Measured 2026-08-25. The released weights reproduce **9 of 14** strings printed on the
upstream card, under a prompt verified **byte-identical** to the literal one the card
documents (checked before concluding anything — the alternative explanation was a bug in
the harness, and it was ruled out first). Misses are stylistic: a retained `So`/`Hmm`
opener, `March 3rd` for `March 3`, and the card's own `Structure: lists` example staying
prose. The card's table appears to predate the released checkpoint.

So the task gate's verdict runs against **the released weights via `transformers`**, and
card agreement is an informational column. The general trap: an upstream card's example
outputs read like fixtures and are not — they are a snapshot of some checkpoint, usually
undated. Verify they reproduce on the weights you are converting *before* adopting them,
or a faithful conversion fails a gate that was never measuring conversion.

## Non-levers

- **Head quantization.** The head is tied to the 151936×1024 embedding and the eager
  quantizer skips shared params, so an `int8hu`-style mode has to untie first — which adds
  a 156 MB int8 head *beside* the 311 MB fp16 embedding instead of replacing it. Pure loss
  at every bit width. On a tied-embedding model, `hu` is not a smaller variant of the same
  idea; it is the wrong idea. The embedding is ~59% of the int4 bundle, and shrinking it is
  an embedding-quantization question that this port does not open.
- **`--static-ids`.** Exists to kill per-step input_ids respecialization on big bundles. At
  0.6B the respecialization is cheap and the workload is prefill-heavy (the input is a whole
  transcript), so batched prefill is worth far more: 4161 tok/s default vs 344 tok/s at
  `COREAI_CHUNK_THRESHOLD=1`. Revisit only with a device measurement in hand.
- **int4.** Faster (296 vs 268 tok/s decode) and 210 MB smaller, and it passes the oracle
  gate. It corrupts digits. For a model whose job includes inverse text normalization of
  money and counts, that closes it.

## The iPhone ceiling — measured, then explained from the source

Device gate (iPhone 17 Pro, PipelinedBench Release, 2026-08-25): numerics **276/276 + 27/27
token-exact vs the Mac engine**, the mandatory dynamic-KV `PB_G=1024` run passes, **62.4
decode / 69.0 prefill tok/s** cold, no AOT needed. `device == Mac == fp32 HF`.

Then the interesting part. `PB_G=1024` drives the long KV shape with a *random* prompt and
never gates its output, so it does not answer whether the model still speaks up there. Run
the model's own workload instead — a 611-token transcript whose Mac rewrite runs 603 tokens,
1214 total — and the device produces **413 tokens, every one identical to the Mac**, then
stops at absolute position exactly **1024**.

Truncation, not corruption. And the cause was read out of the source rather than inferred
from the shape of the number: `CoreAIPipelinedEngine.swift` carries `#if os(iOS)` +
`GrowingKVCache` → `iosDynamicKVCapacityCap = 1024`, capping `maxTokens` to
`1024 - processed - prompt.count` and throwing `contextLengthExceeded` when a prompt leaves
no budget. `1024 − 611 = 413` — the arithmetic matches the measurement exactly. It guards
the iOS compiler's miscompilation of growing-KV specializations at seq ≥ 2048
(apple/coreai-models#124), so truncating is the guard doing its job.

Two things generalize:

- **A `PB_G` trial and a long real-workload case are different checks.** The trial proves
  the engine survives the shape; only the workload case proves the output survives it. This
  port needed both, and the second is what found the ceiling.
- **A shipped guard is a product constraint, not just a safety net.** Any dynamic-KV model
  on iOS inherits the 1024 cap. For a chat model that is a long conversation; for a
  normalizer whose input is a whole transcript it lands inside the *first* call. S1-mini's
  own card recommends inputs up to ~1000 tokens — which on iPhone leaves no room for the
  rewrite, and ≥1024 throws outright. Chunk to roughly ≤450–500 input tokens. Mac is
  uncapped and clean at every shape.

Thermal, recorded because a dictation post-processor runs repeatedly: back-to-back
1024-token generations took the same bundle to 34.9/30.5 tok/s, and seven idle minutes
restored 69.0/62.4. Cold is the published number because that is the shared protocol; the
sustained number is the one to plan capacity against.

## Chat template

Qwen3's, unchanged, and `enable_thinking=False` is **mandatory**: with thinking on the model
emits an empty `<think>` block and stops, so every case returns the empty string. That failure
is uniform and silent — a pipeline that looks like it runs and produces nothing. The trained
prefix is `<|im_start|>assistant\n<think>\n\n</think>\n\n` (two newlines inside the block, two
after it). The user turn is a control line, a newline, then the raw transcript:
`[Styling: casual|semi-casual|semi-formal|formal] [Structure: prose|lists] [Context: general|email]`.
The system prompt is part of the trained format and must be sent verbatim.

## Licensing

Apache-2.0 plus an ADDITIONAL TERM: any use, distribution or product integration must keep
identifying the model as **"S1-mini"** by **"Superwhisper"**, exact capitalization, whatever
the product is called. Redistribution of converted weights is permitted; renaming is not.
Every card, bundle name and app string has to carry the name.
