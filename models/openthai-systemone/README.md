Core AI is Apple's on-device ML runtime in iOS 27 / macOS 27 and the successor to Core ML: PyTorch models are exported with Apple's `coreai-torch` (LLMs: `coreai.llm.export`) into `.aimodel` bundles that run on the GPU or the Neural Engine, e.g. Qwen3-8B 4-bit decodes at 94 tok/s on an M4 Max GPU, MLX 90 under the same protocol ([apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench), macOS 27 beta 26A5353q, 2026-06-11).

# OpenThai-SystemOne — Core AI

[🤗 mlboydaisuke/OpenThai-SystemOne-CoreAI](https://huggingface.co/mlboydaisuke/OpenThai-SystemOne-CoreAI) · Apache-2.0 · source [iapp/OpenThai-SystemOne](https://huggingface.co/iapp/OpenThai-SystemOne/tree/f3709948b5e3cc9606a57e74ba62b7a639d17dd3) (revision `f3709948`) · base Qwen/Qwen3.5-0.8B-Base

A **Thai + English System One decision model**: it reads a state and typed questions — Choice,
Score or Noul — and returns option probabilities. The Qwen3.5 text tower was continued-pretrained
on Thai; its language-model head was replaced by a **256-way biased slot head**. The readout is
at `<|ts_answer|>`. It never generates text.

This is the zoo's second decision model. The graph uses the Qwen3.5 decode-only, loop-free S=1
recipe with three changes: the `model.*` weight prefix, a 248,339-row embedding, and the slot
head in the `lm_head` position. **int8lin is the ship bundle; fp16 is published beside it as
the reference.** Both have a 4,096-token context and emit `logits` with shape `[1, 1, 256]`.

## Readout contract

Each question is one independent row in the author's layout, with no chat template or BOS:

```text
<|ts_state|> <state>
<|ts_q|><|ts_choice|> <instructions>
<|ts_opt_0|> <name>: <description>
<|ts_opt_1|> <name>
<|ts_answer|>
```

The author's encoded sequence has a newline after `<|ts_answer|>`; the answer slot is
`len(ids_full) - 2`. The bundle consumes the prefix through the answer token, id **248082**,
and reads the last call's 256 logits. Removing the trailing newline changes the fp32 oracle
logits by at most **0.000018597** across the fixtures (tolerance 0.0001).

- Choice options retain request order. Score uses `<|ts_score|>` and options `i: <level>`,
  with 2–10 levels. Noul uses `<|ts_noul|>` and slots `0 = no`, `1 = yes`, with the supplied
  false/true descriptions when present.
- For `k` options, divide all slot logits by the question type's temperature, mask slots
  `k..254` to negative infinity, and softmax over all 256 slots. Return
  `p_options = p_full[:k] / sum(p_full[:k])`; retain `p_full[255]` as abstain. Slot indices
  are **not vocabulary token ids**. Choice supports up to 255 options.
- Temperatures are read from `exp(log_temperature)` in the checkpoint: **choice 1.058534,
  score 1.043141, noul 1.006767**. They differ from the author's v0.3 card, which quotes
  **choice 1.055, score 1.008, noul 1.047**. Metadata carries the tensor-derived values at
  full precision.
- The author's API exposes abstain for **Choice only**. Score exposes probabilities and
  confidence; Noul exposes the probability of yes. Confidence is one minus normalized
  entropy. The pinned client does not round; its Choice/Score assembly performs a second
  fp32 renormalization. All fixtures use `permutations=1`.

The [fixture](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/openthai-systemone/fixtures-openthai-systemone.json)
contains 18 requests: 48 kit rows (24 Choice, 14 Noul, 10 Score) and two zoo rows with 40 and
255 options. It covers Thai, English, mixed text, dict states and list states. The bundle and
the kit answer one question per row. The independent-row fp32 oracle assembly equals the
author's single-question API exactly for **18/18 requests** (50/50 question calls). The author's
one-pass API places several questions in one causal sequence; its answers to later questions
can differ from independent rows: max **|Δp| 0.375453** on these requests. That comparison is
recorded as API behavior, not a conversion gate.

## Measured (Apple M4 Max GPU, macOS 27.0 26A428, 2026-09-23)

| | fp16 (reference) | int8lin (ship) |
|---|---:|---:|
| option argmax = author's fp32 oracle | 50/50 | 50/50 |
| argmax on oracle margin ≥ 0.02 | 49/49 | 49/49 |
| max \|Δp\| over option probabilities | 0.005059 | 0.020813 |
| mean of per-row mean \|Δp\| | 0.000225 | 0.000659 |
| max \|Δabstain\| | 0.016488 | 0.018837 |
| Swift pipelined first token = decoded raw-slot argmax | 50/50 | 50/50 |
| Swift sequential first token = decoded raw-slot argmax | 50/50 | 50/50 |
| state reset, row 1 logits bit-identical | yes | yes |

The gate requires option-argmax agreement on every row with oracle margin ≥ 0.02, finite
logits and the state-reset proof. Probability and abstain errors are recorded. The only row
below that margin is `r18-slot` (0.009739); it agrees on both bundles. The largest int8lin
probability difference is `r05-dry`, a two-option Noul row with oracle margin 0.061681.

Probabilities come from an AOT h16c GPU asset loaded through the Core AI Python runtime with
`SpecializationOptions.default()`, fresh zero states per row and full `position_ids` at each
S=1 step. The engine check uses Release `llm-runner`, raw ids, one greedy token, and
`COREAI_CHUNK_THRESHOLD=1`. It compares `tokenizer.decode([raw256_argmax])` against the same
bundle's unmasked Python readout; that diagnostic string is not a decision answer. Transcripts:
[fp16 readout](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/openthai-systemone/gate-openthai-systemone-readout-fp16.json),
[int8lin readout](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/openthai-systemone/gate-openthai-systemone-readout-int8lin.json),
[fp16 engines](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/openthai-systemone/gate-openthai-systemone-engine-fp16.json),
[int8lin engines](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/openthai-systemone/gate-openthai-systemone-engine-int8lin.json).

**Throughput**, int8lin, Release `llm-benchmark`, p=128 / g=256, two launches × three trials
per engine, `COREAI_CHUNK_THRESHOLD=1`; median (range):

| Engine | prefill proxy, tok/s | decode, tok/s | load per launch, s |
|---|---:|---:|---:|
| coreai-pipelined | 252.7 (243.8–258.6) | 250.8 (244.5–253.8) | 1.415 / 0.167 |
| coreai-sequential | 197.1 (196.1–201.3) | 194.8 (192.6–197.7) | 0.172 / 0.166 |

This is a **prefill-rate proxy**: the graph is S=1, and synthetic generation throughput is
not decision latency. The benchmark samples ids from the metadata's 256-wide output range.
No other Core AI, Python or Swift engine job appeared in the before/after process snapshots
(`contended: false`). Load is measured per launch, excluding warmup. The frozen Swift tag's
benchmark needed a local CLI option to select `EngineOptions.variant`; the trial loop was
unchanged. [Trials, load times and environment](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/openthai-systemone/llm-benchmark.json).

## Through the kit

**Measured through coreai-kit**, using its sequential engine and tokenizer: `decide-cli parity`
matched tokens 50/50, answer slots 50/50 and option argmax 50/50 on both bundles, including the
40- and 255-option rows. int8lin max |Δp| was **0.0226** (`r05-dry`), mean **0.0009**, and max
|Δabstain| **0.0222**; fp16 was **0.0051** (`r05-task`), **0.0003**, and **0.0165** respectively.
These are kit measurements supplied by the supervisor, separate from the Python-runtime
table above. Median int8lin wall time per fixture question was **354 ms** over the 50 rows, two to
three questions per state (a question on a new state pays for the whole state); the 255-option row (1,449 tokens, S=1 prefill) took **7.2 s**. A
three-question Thai ticket took **351 / 316 / 429 ms** for its 57-, 63- and 83-token rows; the
recurrent hybrid cannot rewind mid-sequence, so every row is prefilled from its first token (0
tokens reused).

On SemIf's authored144 — 144 English rows with three options, SemIf's gold labels and unchanged
`benchmarks/evaluate.py` — int8lin on the Mac GPU, measured through coreai-kit, scored
**109/144** raw and **0.7249 mean family balanced accuracy**. The kit README reports **0.681**
for MiniCPM5-2B int8 and **0.821** for Qwen3.5-4B int8 on the same rows and evaluator.
[Kit measurement record](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/openthai-systemone/measurements-coreai-kit.json).
**iPhone 17 Pro** (iOS 27.0 24A437, the same int8lin bundle sideloaded into the kit's ModelStore, sha256 equal to
the Hub revision, 2026-09-23, a headless harness that runs the kit's own `decide-cli parity` / `oracle` inside an
app; thermal state "serious" throughout): `parity` matched tokens **50/50**, answer slots **50/50** and option
argmax **50/50**, the 40- and 255-option rows included (the 1,449-token row fits this bundle's context); max |Δp|
**0.0210**, mean **0.0009**, max |Δabstain| **0.0213**. Median wall time per fixture question **1,893 ms** (Mac
354 ms); the 255-option row **40.8 s** (Mac 7.2 s). On SemIf's authored144 through `oracle` on the phone: **109/144**
raw and **0.7249** mean family balanced accuracy — the Mac's figures exactly — at **2,074 ms** median per decision.
Cooled to thermal state "nominal" (2026-09-24, `decide-cli bench --repeat 3`, a 111-token state and eight questions):
**1,527 ms per decision** with the state shared, 1,713 from scratch, 13.5 s for the state and its eight (0 tokens
reused). Load 6.6 s. Records: the SemIf rows in the standup record, and
[`gate-openthai-systemone-iphone-parity.json`](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/openthai-systemone/gate-openthai-systemone-iphone-parity.json).

## Bundle

[mlboydaisuke/OpenThai-SystemOne-CoreAI](https://huggingface.co/mlboydaisuke/OpenThai-SystemOne-CoreAI)
contains both LanguageBundles, each with `.aimodel`, `metadata.json` and `tokenizer/`:

| Path under `gpu-pipelined/` | role | bundle bytes | `main.mlirb` bytes |
|---|---|---:|---:|
| `openthai_systemone_decode_int8lin/` | ship | 1,068,353,811 | 1,039,655,099 |
| `openthai_systemone_decode_fp16/` | reference | 1,534,777,239 | 1,506,078,533 |

int8lin quantizes the linears per block of 32; the biased slot head, embeddings, conv1d and
norms stay fp16. `language.vocab_size = 256` describes the logits width because the sequential
engine allocates its output buffer from it. The input tokenizer still contains **248,339
tokens, including all 295 added tokens**. The `decision` metadata carries the slot count,
abstain slot, answer token, temperatures and layout. The source `config.json` is retained for
provenance.

Use the zoo's [extra-states runtime patch](https://github.com/john-rocky/coreai-model-zoo/blob/main/apps/coreai-pipelined-extra-states.patch)
for the hybrid's KV, conv and recurrent states, and `COREAI_CHUNK_THRESHOLD=1`. Both pipelined
and sequential engines were checked with Release tools from fork tag `0.2.4-zoo` (`f7a75ec`).
The [recipe](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/openthai-systemone/recipe.toml)
records the source revision and each graph's SHA-256.

## Reproduce

Run from the zoo checkout with the overlay environment; the oracle uses its own uv-managed
environment. `DEVELOPER_DIR` must select Xcode 27 for the Core AI tools.

```bash
python3 conversion/zoo_convert.py run openthai-systemone
python3 conversion/zoo_convert.py run openthai-systemone-fp16

uv run conversion/slot/oracle_slot.py \
    --out models/openthai-systemone/fixtures-openthai-systemone.json

python3 conversion/slot/readout_gate_slot.py \
    exports/openthai_systemone_decode_int8lin \
    models/openthai-systemone/fixtures-openthai-systemone.json \
    --transcript models/openthai-systemone/gate-openthai-systemone-readout-int8lin.json

python3 conversion/slot/engine_argmax_slot.py \
    exports/openthai_systemone_decode_int8lin \
    models/openthai-systemone/fixtures-openthai-systemone.json \
    --readout models/openthai-systemone/gate-openthai-systemone-readout-int8lin.json \
    --runner <fork>/.build/release/llm-runner \
    --engine pipelined --engine sequential \
    --transcript models/openthai-systemone/gate-openthai-systemone-engine-int8lin.json
```

Repeat the readout and engine commands with `fp16` paths for the reference. The
[exporter](https://github.com/john-rocky/coreai-model-zoo/blob/main/conversion/export_openthai_systemone_decode_pipelined.py)
downloads the pinned snapshot itself. [Gate instructions](https://github.com/john-rocky/coreai-model-zoo/blob/main/conversion/slot/README.md)
and [port notes](https://github.com/john-rocky/coreai-model-zoo/blob/main/knowledge/openthai-systemone-port.md)
record the oracle dependencies and runtime contract.

## License

Source Apache-2.0 (`iapp/OpenThai-SystemOne`); the bundles inherit it. The pinned source
snapshot has no license file, so `LICENSE` contains the canonical
[Apache License 2.0 text](https://www.apache.org/licenses/LICENSE-2.0.txt). The author's
inference files are downloaded by the oracle at gate time and are not included in the bundles.
