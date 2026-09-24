# decider-0.8b: a System One decision model on Core AI — the probability readout is the gate

> 2026-09-21. Mapika/decider-0.8b (Qwen3.5-0.8B-Base fine-tune, Apache-2.0) answers typed
> questions by the probability of option-letter tokens at an answer slot; it never generates
> text. The port is the Qwen3.5-0.8B ship recipe with the HF id swapped — the work was proving
> that the **probabilities** the bundle yields equal the author's, and finding where the Swift
> stack can and cannot read them. Card: [`models/decider-0.8b/README.md`](../models/decider-0.8b/README.md).
> Converted and gated by Codex (gpt-6-astra) under a supervising Claude session; the facts below
> are its run record, re-checked by the supervisor from the raw JSON.

## The readout, and how it was gated

The author's `Decider.system_one(state, questions)` plans one independent row per question
(`Context:\n<state>\n\nQuestion: …\nOptions:\n(A) …\nAnswer: (`), takes the logits at the last
token, restricts them to the label ids (A..Z then 229 single-token pairs, 255 labels), and
applies `softmax(logits / 1.03)`. Isolated Score levels become one yes/no row each. That row
function is what the Core AI bundle must reproduce, so the gate is probability parity, not a
greedy transcript:

- 44 fixture rows from 13 synthetic requests, built and scored through the author's unchanged
  `decider/` package in fp32 on the CPU (`conversion/decider/oracle_decider.py`; the file
  carries ids, slot, label ids, logits and probabilities). Minimum top-2 margin 0.51: the argmax
  gate needs no near-tie exemption.
- Bundle side, S=1 steps with fresh zero states per row and full-length position_ids, last
  step's fp16 logits gathered at the label ids, same softmax: fp16 build max |Δp| 0.0050 /
  mean of row means 0.00018; **int8hu ship 0.0084 / 0.00067**, argmax 44/44 on both, the
  full-vocabulary argmax is a label on every row, and row 1 re-run at the end reproduces its
  logits exactly. Ship bar = 44/44 + max ≤ 0.02 + mean ≤ 0.002 (four times the fp16 floor).
- Engine side, the Swift `coreai-pipelined` engine's first greedy token on the same ids equals
  the oracle label on 44/44 (`conversion/decider/engine_argmax_decider.py`).
- The fp16 floor is real: the graph emits float16 logits (the pipelined engine requires it),
  and fp16 spacing at |logit| 16–32 is 0.0156. A |Δp| ≤ 1e-3 target would be unreachable by
  construction on this graph.

## What the Swift stack can read today

- `CoreAIPipelinedEngine` samples on the GPU and throws on `includeLogits`. `CoreAISequentialEngine`
  returns logits but requires exactly two states; this hybrid carries four (KV + conv + rec).
  So the Swift engine gives the argmax only. The fp16 logits already sit in the pipelined
  engine's `decodeLogitsBuffers`; the smallest change is a completion-synchronized
  read-last-logits primitive — the recommendation of
  [`decider-systemone-op-design.md`](decider-systemone-op-design.md), which also specifies the
  `CoreAI.systemOne` op, the prompt-builder port, a tokenizer parity contract on the 44 rows,
  and the cost model (S=1 prefill: a request costs `rows × (state + question tokens)` steps).
- The frozen fork's pipelined engine caps the iOS growing KV cache at 1,024 tokens
  (`CoreAIPipelinedEngine.swift`, the `GrowingKVCache` limit). Measured on the iPhone 17 Pro
  (2026-09-21, AOT h18p GPU, PipelinedBench rows mode, sideloaded file by file with md5 round
  trips): 43/43 rows under the cap emit the oracle's label, load 2.5 s; the 1,965-token row is
  rejected with `contextLengthExceeded(0, 1024)` — `models/decider-0.8b/gate-decider-0.8b-iphone-argmax.json`.

- That cap is the pipelined engine's, not the phone's: through the kit's logits engine (`TypedDecisions`, `decide-cli parity` run inside an app on the iPhone 17 Pro, 2026-09-23) the same 255-option row, 1,965 tokens, answered in 69.9 s with the fp32 argmax, 44/44 rows — `models/decider-0.8b/gate-decider-0.8b-iphone-parity.json`.
## Python runtime traps on macOS 27.0 (26A428)

- **The GPU JIT of this 0.8B graph is wrong.** Loading the `.aimodel` with
  `SpecializationOptions.from_preferred_compute_unit_kind(gpu())` logged
  `MTL4CommandQueueErrorDomain error 1` on every forward and returned all-zero logits — the
  signature previously seen only above ~1B. Fix: `coreai-build compile … --platform macOS
  --preferred-compute gpu --architecture h16c --expect-frequent-reshapes`, then load the
  `.aimodelc` with `SpecializationOptions.default()`. Both bundles then pass; scratch growth 0.
- **One IOSurface per call leaks.** S=1 stepping dies after roughly 25,000 calls in one
  process (`NDArray+SharedStorage.swift:108: Failed to allocate storage for NDArray … int32
  [1, 1] … ioSurface`), deterministically at the same call. Split long fixture sets across
  processes (≤ 15 prompts of ~150 tokens each worked; 44 decider rows are ~5,000 steps and
  fit in one). Re-running the first prompts in a fresh process reproduced their logits exactly.
- The exporter assumed a nested `text_config`; the decider checkpoint stores a flat
  `qwen3_5_text` config. `export_qwen3_5_decode_pipelined.py` now retries the loader with
  `hf_config_attr=None` on `AttributeError`. Weights are under `model.language_model.*` as in
  the Qwen/Qwen3.5-0.8B checkpoint, so nothing else changed.

## The same letter readout on two shipped bundles (SemIf's authored144, Mac GPU)

SemIf (TheoLeeCJ/SemIf, MIT) scores criteria the same way — last-position logits restricted to
the option letters, plain softmax — and publishes per-row predictions for
Qwen3-0.6B / MiniCPM5-2B / Qwen3.5-4B. The shipped Core AI bundles of two of them were read
the same way (AOT `.aimodelc`, Python runtime, SemIf's own rendered prompts and evaluator, no
conversion):

| shipped bundle | published bf16 balanced acc. | Core AI Mac GPU int8 | argmax vs reference | max / mean \|Δp\| | engine argmax vs readout |
|---|---:|---:|---:|---:|---:|
| MiniCPM5-2B-CoreAI `int8/` | 0.686 | 0.681 | 141/144 vs the fp32 CPU oracle and vs bf16 (the 3 flips are near-ties: 0.50/0.50, 0.49/0.46, 0.56/0.43) | 0.115 / 0.009 (vs fp32) | 143/144 (the miss is the 0.50/0.50 tie) |
| qwen3.5-4B-CoreAI `gpu-pipelined-b2/` | 0.813 | 0.821 | 143/144 vs bf16 (flip 0.46/0.49) | 0.078 / 0.007 | 143/144 |

Balanced accuracy = SemIf's `evaluate.py` on the same 144 gold rows for every column. The
Qwen3.5-4B reference is SemIf's published bf16 run (its MLX reproduction was not available);
the MiniCPM5-2B reference is a CPU fp32 re-run of SemIf's scorer plus the published bf16 rows.
Facts only — the columns are the same fixture, the same model, a different device.

## Measurement notes

- Ship bundle throughput on M4 Max (Release `llm-benchmark`, p128/g256, 2 launches × 3
  trials): decode 193.6 tok/s median (186.5–197.0), prefill 226.4, load 1.5 s cold / 0.2 s warm;
  GPU free of other Core AI work, one CPU-bound job from another lane present.
- `coreai_gate.py` (the zoo's language-model gate) passes 16/16 on the ship bundle with the
  alphabet prompt — the fine-tune still continues text, though the API never asks it to.
