# decider-0.8b — Core AI

[🤗 mlboydaisuke/decider-0.8b-CoreAI](https://huggingface.co/mlboydaisuke/decider-0.8b-CoreAI) · Apache-2.0 · source [Mapika/decider-0.8b](https://huggingface.co/Mapika/decider-0.8b) (revision `1ea5412`) · base Qwen/Qwen3.5-0.8B-Base

A **System One decision model**: it reads a state (text or JSON) and a set of typed questions —
Choice (2–255 options), Score (2–10 described levels), Noul (probability of yes) — and returns a
probability for every option from the **letter logits at an answer slot**. It never generates
text. Mapika fine-tuned Qwen3.5-0.8B-Base for this readout (one epoch over 1.47M examples, per
the author's card) and ships it behind the same `POST /v1/systemone` shape as the larger
decider-2b. The author's own numbers, quoted from the source card and not re-measured here:
in-task accuracy 0.776, held-out 0.707, calibrated with temperature 1.03.

This is the zoo's first decision model: the value is the **calibrated probability**, so the gate
below is a probability-parity gate against the author's fp32 inference code, not a token match.
The bundle is the Qwen3.5-0.8B ship recipe with the HF id swapped (`int8hu --head-sym`: linear
int8 per block of 32 with an absmax-symmetric int8 head, decode-only loop-free S=1 graph on the
pipelined GPU engine), 1.34 GB, context 4,096.

## Readout contract

Every question is one independent row in the author's `state_first` layout:

```
Context:\n<state>\n\nQuestion: <question>\nOptions:\n(A) <option>\n(B) <option>...\nAnswer: (
```

- The answer slot is the **last token** of the row; the next-token logits at that position are
  the readout. No chat template, no BOS, no generated token.
- Labels come from the bundle's tokenizer: `A`..`Z`, then the first 229 two-letter strings that
  encode as **one token** (255 labels; `AA` = 5840, `AB` = 1803, …). Only the first `nopts`
  labels are read; `p = softmax(logits[labels[:nopts]] / 1.03)`.
- A Score question with the checkpoint's `isolated_levels = true` becomes one yes/no row per
  level (`<question>\nProposed answer: <level>\nDoes the proposed answer fit?`); the level
  probabilities are the normalized yes-mass. `neutralize_none = false`: option strings are used
  unchanged.
- Rows must fit the bundle's 4,096-token context. A 255-option row is ≥ 1,275 tokens by
  construction; the fixture's is 1,965.

`conversion/decider/oracle_decider.py` builds the fixture rows through the author's unchanged
`decider/` package (downloaded from the checkpoint) and records the fp32 probabilities:
[`fixtures-decider-0.8b.json`](fixtures-decider-0.8b.json) — 13 requests, 44 rows (23 Choice
rows with 3–10 options, 9 Noul, 10 isolated Score rows from 2 Score questions, one 11-option
row, one 255-option row), every row's ids, slot, label ids, fp32 slot logits and probabilities,
and the author's `system_one` API output for each request (assembled answers equal the
row-level probabilities, 13/13). Minimum oracle top-2 margin 0.51 — no near-ties, so the argmax
gate has no exemptions.

## Measured (Apple M4 Max, macOS 27.0 26A428, 2026-09-21)

| | fp16 build (reference) | **int8hu --head-sym (ship)** |
|---|---:|---:|
| letter argmax = fp32 oracle | 44/44 | **44/44** |
| full-vocabulary argmax is one of the row's labels | 44/44 | 44/44 |
| max \|Δp\| over all option probabilities | 0.0050 | **0.0084** |
| mean of per-row mean \|Δp\| | 0.00018 | **0.00067** |
| Swift pipelined engine, first greedy token = oracle label | 44/44 | **44/44** |
| state reset proof (row 1 re-run, logits bit-identical) | yes | yes |

Ship bar: argmax 44/44 with no exemption, max |Δp| ≤ 0.02 and mean of row means ≤ 0.002 — four
times the fp16 build's floor. The fp16 floor is the graph's own fp16 logits (the pipelined engine
requires a float16 `logits` output), not conversion error.

Two paths produce those rows, because the pipelined engine samples on the GPU and exposes no
logits:

- **Probabilities**: the bundle is AOT-compiled (`coreai-build compile … --platform macOS
  --preferred-compute gpu --architecture h16c --expect-frequent-reshapes`) and the `.aimodelc`
  is driven S=1 through the Core AI Python runtime with fresh zero states per row —
  `conversion/decider/readout_gate_decider.py`, transcript
  [`gate-decider-0.8b-readout.json`](gate-decider-0.8b-readout.json). AOT is required, not an
  optimization: on 26A428 the Python runtime's JIT of this graph logged
  `MTL4CommandQueueErrorDomain error 1` on every forward and returned all-zero logits.
- **Engine argmax**: Release `llm-runner --raw-tokens <row ids> --max-tokens 1 --temperature 0.0
  --inference-engine-variant coreai-pipelined --warmup off` (`COREAI_CHUNK_THRESHOLD=1`) must
  emit the oracle's label string — `conversion/decider/engine_argmax_decider.py`, transcript
  [`gate-decider-0.8b-engine.json`](gate-decider-0.8b-engine.json).

The zoo's language-model gate also passes on the ship bundle — `coreai_gate.py`, prompt "The
alphabet begins A, B, C, D, E, F,", **16/16** token-exact vs the fp32 overlay oracle
([`gate-decider-0.8b.json`](gate-decider-0.8b.json)): the fine-tune still speaks, which the
System One API never asks of it.

**Throughput** (ship bundle, Release `llm-benchmark`, p=128 g=256, `coreai-pipelined`,
`COREAI_CHUNK_THRESHOLD=1`, 2 launches × 3 trials): decode **193.6 tok/s** median (186.5–197.0),
prefill 226.4 (201.6–236.6), load 1.5 s cold / 0.2 s warm. No other Core AI work was on the GPU;
a CPU-bound job from another lane ran on the same machine during the measurement. Because
prefill is S=1 on this graph, a System One request costs about `rows × (state + question
tokens) / decode rate` — ten independent questions over a 300-token state are ~3,500 steps.

### JevBench public 231 (Mac, 2026-09-24)

| easy 48 | standard 72 | hard 111 | ECE hard | p50 | p95 | hard max |
|---:|---:|---:|---:|---:|---:|---:|
| 1.000 | 0.833 | 0.414 | 0.320 | 1.69 s | 29.10 s | 90.6 s |

The benchmark's own harness ([fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench)
`2fa63fa`, v1.4.0, `typesafe` adapter) ran the 231 public items against coreai-kit `adbc755`
`decide-cli serve` with the ship bundle, one question per request. Accuracy per tier and the hard
tier's ECE are JevBench's own scoring (argmax of the returned probabilities); p50 and p95 are
per-request latency over all 231 requests, hard max the maximum over the hard tier. Latency was
measured without an exclusive GPU window (contended), with this model's server running alone; the
graph's prefill is S=1, and at that kit commit the state was prefilled again for every question.
JevBench's published scores (Intelligence and the rest) are chance-corrected over 534 items, sealed
ones included, and are not comparable to these accuracies.

**iPhone 17 Pro (2026-09-24)**

| | easy 48 | standard 72 | hard 20 |
|---|---:|---:|---:|
| accuracy | 1.000 | 0.833 | 0.300 |
| p50 | 1.33 s | 2.26 s | 8.04 s |
| p95 | 1.80 s | 4.21 s | 10.05 s |
| p50 / p95 over | 48 rows, nominal | 72 rows, hot | 20 rows, nominal |

The phone (iOS 27.0 24A437) received the same request bodies as the Mac run, one question per
request. A headless harness app answered each with coreai-kit 0.7.1, through the call the kit's
System One server makes. Every bundle file on the phone matched the Hub revision by hash. Every
answer's argmax equals the Mac run's. The hard column is the middle 20 of the 111 hard items by
state length. The Mac run scored 0.300 on the same 20. p50 and p95 are the kit's time per request:
the state's prefill plus the decision. A nominal row started and ended with the phone on its
battery at thermal state nominal. A hot row started or ended at fair or worse. The standard tier
ran on the charger. The easy latency is from a second easy run on the battery, 48/48 again.

## Swift side

CoreAIKit 0.7.0 reads this model as the catalog decision model `decider-0.8b`
(`Decision.Format.decider`): the kit renders the model's own prompt form, reads the answer slot
at the card's temperature and returns each option's probability. In Swift (`import CoreAIOps`),
`CoreAI.decide(state, questions, options: .model("decider-0.8b"))` asks typed questions of one
state, and `CoreAI.systemOne(json:)` answers a `/v1/systemone` request as a client sent it (with
`"model": "decider-0.8b"`), the typed answers beside the wire object. Over HTTP, in the hosted
API's forms: `brew install john-rocky/tap/systemone && systemone serve --model decider-0.8b`, or
`decide-cli serve --model decider-0.8b` in the kit's `Examples/Decide`. The calls, the other
decision models and the kit's measurements are in
[System One, on device](https://github.com/john-rocky/coreai-kit/blob/0.7.0/docs/SYSTEM_ONE.md).

**iPhone 17 Pro (iOS 27.0, GPU, pipelined engine, AOT h18p, 2026-09-21):** the same 44 rows through a
PipelinedBench rows mode (raw ids in, one greedy token out) — **43/43 rows that fit emit the
oracle's label**, engine load 2.5 s; the 255-option row (1,965 tokens) is rejected with
`InferenceRuntimeError.contextLengthExceeded(0, 1024)` because the frozen fork's pipelined engine
caps the iOS growing KV cache at 1,024 tokens
([`gate-decider-0.8b-iphone-argmax.json`](gate-decider-0.8b-iphone-argmax.json)).

**iPhone 17 Pro through coreai-kit (iOS 27.0, GPU, the kit's logits engine, 2026-09-23):** the same 44 rows
through `decide-cli parity` run inside an app, with the kit's `Decision.Format.decider` and its 255-option
label table — tokens **44/44**, slots and labels **44/44**, option argmax **44/44** against the author's fp32
readout, max |Δp| **0.0092**, mean 0.0010, at the card's temperature 1.03; the 255-option row (1,965 tokens)
answered in **69.9 s** — the 1,024-token limit above is the pipelined engine's growing cache, not the
phone's. Median **2,078 ms** per question (all rows of a score question summed) with the phone at thermal
state "serious" ([`gate-decider-0.8b-iphone-parity.json`](gate-decider-0.8b-iphone-parity.json)).

## ⬇️ Bundle

[mlboydaisuke/decider-0.8b-CoreAI](https://huggingface.co/mlboydaisuke/decider-0.8b-CoreAI)
`gpu-pipelined/decider_0_8b_decode_int8hu_block32_sym/` — `.aimodel` (main.mlirb 1,309,263,719 B,
sha256 `2ab6d715…aaf4`), `metadata.json`, `tokenizer/`. Runs on the pipelined engine with the
zoo's [`apps/coreai-pipelined-extra-states.patch`](../../apps/coreai-pipelined-extra-states.patch)
(the hybrid's conv/rec states) and `COREAI_CHUNK_THRESHOLD=1`, like every Qwen3.5 bundle here.

## Reproduce

```bash
# export (recipe.toml): the Qwen3.5 exporter with the HF id swapped; the decider checkpoint
# stores a flat qwen3_5_text config, so the loader falls back from text_config to the root.
python3 conversion/zoo_convert.py run decider-0.8b

# fixtures + fp32 oracle through the author's own decider/ package (uv-managed env, CPU, ~5 min)
uv run conversion/decider/oracle_decider.py --out models/decider-0.8b/fixtures-decider-0.8b.json

# probability gate: AOT h16c + Python runtime (overlay interpreter, DEVELOPER_DIR = Xcode 27 RC)
python3 conversion/decider/readout_gate_decider.py exports/decider_0_8b_decode_int8hu_block32_sym \
    models/decider-0.8b/fixtures-decider-0.8b.json --transcript models/decider-0.8b/gate-decider-0.8b-readout.json

# engine argmax gate: Release llm-runner from the patched fork
python3 conversion/decider/engine_argmax_decider.py exports/decider_0_8b_decode_int8hu_block32_sym \
    models/decider-0.8b/fixtures-decider-0.8b.json --runner <fork>/.build/release/llm-runner \
    --transcript models/decider-0.8b/gate-decider-0.8b-engine.json
```

Port notes: [`knowledge/decider-0.8b-port.md`](../../knowledge/decider-0.8b-port.md).

## License

Source Apache-2.0 (Mapika/decider-0.8b); the bundle inherits it. The author's `decider/`
inference code is used by the oracle script at gate time and is not part of the bundle.
