# Kev-0.8B — Core AI

[🤗 mlboydaisuke/Kev-0.8B-CoreAI](https://huggingface.co/mlboydaisuke/Kev-0.8B-CoreAI) · Apache-2.0 · source [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b/tree/bf75a6a8848ea6960ff2ed108d9ed44c2941174f) (tag `v1.0`, commit `bf75a6a`) · base [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/tree/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68) (revision `dc7cdfe`)

A **decision model**. Give it a state (text, or a JSON object or array) and typed questions: `noul`
(yes / no), `choice` (named options) or `score` (ordered levels). It returns a probability for every
option of every question. It never generates text. Requests and responses use the SystemOne-compatible
request shape: `{model, state, questions}` in, `{model, answers, usage}` out, one typed decision per
question.

Jared Palmer trained it as a rank-16 LoRA adapter and a pointer head on Qwen3.5-0.8B-Base (24 layers:
18 Gated DeltaNet and 6 full attention, hidden size 1,024). The author's card: "It reads one document
(the *state*) and a set of typed questions about it, and returns a calibrated probability distribution
over the options supplied with each question, in a single forward pass and without generating text."
Each question is its own row: the state, then the question and its options. The head scores each
option's closing token against the question's last token. The author's card reports benchmark
results; none of them are re-measured here.

This port merges the adapter into the base with the author's own merge script and exports the text
backbone as one Core AI graph. The graph takes 128 token ids per call and returns the final-norm hidden
state at every position; it has no vocabulary head. Each Gated DeltaNet layer runs its recurrence in an
fp32 Metal kernel, one GPU dispatch per layer per call. Weights are fp16 (1.51 GB). The host runs the
pointer head from the author's head weights. The gate is probability parity with the author's own fp32
code, on every option of every question, on a fixture of 434 questions and on a held-out set of 130.

It runs on the Mac GPU and on an iPhone 18 Pro. On both, the shipped `.aimodel` is specialized where it
runs.

## Readout contract

One row per question, every row from zeroed states. The ids are the author's `kev.model.encode` in its
row form (`rows_of`), with the base model's tokenizer:

```
[<|fim_prefix|>] + render(state) + [<|fim_middle|>] + render(instructions)
  + for each option: [<|box_start|>] + option_text + [<|box_end|>]
  + [<|fim_suffix|>]
```

- Delimiters, from the base tokenizer: `<|fim_prefix|>` 248060 (state), `<|fim_middle|>` 248061
  (question), `<|box_start|>` 248049 / `<|box_end|>` 248050 (an option), `<|fim_suffix|>` 248062
  (decide). Pad = eos = `<|endoftext|>` 248044; no bos. The `kev-4b` repository still carries Qwen3-era
  `added_tokens.json` with other ids; the author's loader reads the tokenizer from the base, and so
  does this port.
- User text is tokenized without special tokens after `<|name|>` is rewritten to `<¦name¦>`
  (`user_tokens`), so a request can never produce a delimiter.
- `render()` (`kev.api.to_record`): a string as is; an object or an array as `key: value` lines (`- `
  for array items, nesting indented by two spaces), values as Python's `str()`. Options: `noul` =
  `no[: <criteria.false>]`, `yes[: <criteria.true>]`, in that order; `choice` = `name` or
  `name: <description>` in criteria order; `score` = the levels in order. Keys: `choice` = the criteria
  names, `noul` = `false`, `true`, `score` = `"0"`..`"n-1"`.
- The head reads the hidden state at `<|fim_suffix|>` (the row's last token) and at every
  `<|box_end|>`: `z_k = ((W_k h_opt_k + b_k) · (W_q h_decide + b_q)) / 16`, `p = softmax(z / T)` within
  the question, T = 2.3510958125672174 (the author's calibration, stored in `head.pt`). `W_q`, `W_k`
  are `[256, 1024]` with biases, fp32.
- Response (`kev.api.to_answers`): `noul` → p(true); `choice` → the argmax key, a confidence
  `(p_max − 1/K) / (1 − 1/K)` and every p; `score` → Σ level · p, the legend, every p and a confidence
  `max(0, 1 − E|level − mode| / D)`; values rounded to 4 decimals. `usage` = `{input_tokens,
  output_tokens}` with the author's counting (the packed request; the tokens of `json.dumps(answers)`).
- Length: a row holds at most **3,968 tokens**. The last call is padded to 128 tokens and must end at or
  below the graph's position bound of 4,095. The author's server accepts states of 65,536 tokens; here a
  longer row is refused before any call.
- Shared prefix (optional, exact): every row of a request starts with the same state tokens, so the
  state's whole calls (⌊Ls / 128⌋ of them) can run once, the four states be copied per question, and each row
  continue from there. Every row's hidden state equals its direct run bit for bit (below).

`conversion/kev/oracle_kev.py` runs the author's package unchanged (`Checkpoint.load("cpu", fp32)`,
`kev.model.admit`, the row-form `forward`, one CPU thread) and records the reference:
[`fixtures-kev-0.8b.json`](fixtures-kev-0.8b.json). The fixture is 384 records and 434 questions:
lines 1–60 of the author's transfer-v4 development file (`evals/v4/transfer-v4/development.jsonl`,
tag `kev-1.0`; MMLU, 4 options), records 1–20 of each of its seven other sources (140, `choice` and
`noul`), its `score` records 1–20, the 144 `choice` items of SemIf authored144, and 20 records
written for the port (70 questions: `noul`, `choice` with 2–10 options, `score` with 2–7 levels,
JSON states, three states of 1,431–1,746 tokens, one record of 8 questions). The longest row is 1,802
tokens. Fourteen questions have an oracle top-2 margin of 0.02 or less (near-ties) and are listed apart.
The held-out set is the next 130 transfer-v4 records (mmlu 61–100, records 21–30 of each other source,
`score` records 21–40), with its own oracle run.

## Core AI shape

`conversion/kev/qwen3_5_kev_decoder.py`: the overlay's stateful Qwen3.5 text decoder
(`Qwen3_5StatefulForCausalLM`) on the merged weights, the vocabulary head replaced by the identity, the
output the final-norm hidden state at every position. One function, `main`, at a static 128 tokens. Each
Gated DeltaNet layer runs its recurrence in the overlay's fp32 Metal chunk kernel
(`qwen3_5_gdn_metal`): one GPU dispatch per layer per call, the recurrent state kept in fp32 through the
call.

| | name | shape, type |
|---|---|---|
| inputs | `input_ids` | [1, 128] int32 |
| | `position_ids` | [1, seq] int32, the ramp 0 .. seq − 1 |
| states | `keyCache`, `valueCache` | [6, 1, 2, ctx, 256] fp16, ctx up to 4,096 |
| | `convState`, `recState` | [18, 1, 6144, 3], [18, 1, 16, 128, 128] fp16 |
| output | `hidden` | [1, 128, 1024] fp16, every position |

Per row of T ids, from zeroed states: ⌈T / 128⌉ calls, call k with ids[128k : 128k + 128] and position_ids
0..128k + 127. The last call is padded with `<|endoftext|>` (248044) and its padded rows are dropped. The
rows of every call, cut to T, are the backbone's final-norm `last_hidden_state [T, 1024]`. The bundle's
`metadata.json` says `kind: decision-backbone` and `language.prefill_chunk: 128`, and carries this order,
the row layout, the delimiter ids, the render rules, the head formula and the response shape.

**Host.** Builds the rows, runs the calls, converts the hidden rows at the readout positions to float64,
applies the head (`head/head.safetensors`, `head/kev_head.json`) and the per-question softmax in float64,
rounds p to fp32 once, and writes the response. `conversion/kev/decide.py` is the Python reference;
[`apps/Kev`](../../apps/Kev/) is the Swift one. With `shared: true` the state's whole calls run once. A
prepared state (`prepare(state:)`, then `decide(prepared:questionsJSON:)`) runs the same calls in two steps,
so questions that arrive later skip the state.

No engine is involved: the hosts drive the low-level runtime (`AIModel` + `loadFunction`, four zeroed
states per row). No runtime patch.

## Measured (Apple M4 Max, macOS 27.0 26A428, 2026-10-03 – 10-04)

The bar, fixed before any graph ran: the argmax equal to the oracle's on every question whose oracle
top-2 margin is above 0.02 (near-ties listed apart), max |Δp| ≤ 0.02 over every option of every
question, the mean over rows of each row's mean |Δp| ≤ 0.002, and every process re-running its opening row
bit for bit. The Python gates load the AOT `.aimodelc` (`coreai-build compile … --platform macOS
--preferred-compute gpu --architecture h16c --expect-frequent-reshapes`) with
`SpecializationOptions.default()`.

### Before any graph: the merge and fp32 torch

- The merged checkpoint (`scripts/merge_lora_checkpoint.py`, tag `kev-1.0`; fp32, 320 tensors, 186 of
  them adapted, `model.safetensors` sha256 `b5ebf92a…`) answers through the author's own loader as the
  adapter checkpoint does: 434/434 questions, logits bit-equal. `W + (B @ A) × 2` reproduces three
  merged tensors bit for bit (the adapter moves them by 2.8–3.6 %). The base alone moves p by up to
  0.789 (argmax 25/53), the base's Gated DeltaNet projections put back by up to 0.282 (46/53): the
  check can fail ([`gate-kev-0.8b-merge.json`](gate-kev-0.8b-merge.json)).
- The decoder module in fp32 on the CPU, driven like the graph with the recurrence unrolled, read through
  the author's head: 434/434 argmax (the 14 near-ties included), max |Δp| 3.5e-6, lowest per-position
  hidden cosine 0.99999999999 on the 21 rows the oracle keeps. It equals the overlay's plain text decoder
  bit for bit (3 rows) ([`gate-kev-0.8b-torch-parity.json`](gate-kev-0.8b-torch-parity.json)).
- The Metal kernel has no torch values: its `torch_defn` returns zeros of the right shapes. The GPU gate
  below is its parity check.

### The graph alone on the Mac GPU

The oracle's ids in, the fp16 hidden rows read through the author's fp32 head:

| set | questions | argmax (margin > 0.02) | near-ties agreeing | max \|Δp\| | mean of row means | bar |
|---|---:|---:|---:|---:|---:|---|
| **fixture** | 434 | **420/420** | **13/14** | **0.0124** | **0.00096** | **PASS** |
| held out | 130 | 125/125 | 5/5 | 0.0058 | 0.00083 | PASS |

The near-tie that flips is a SemIf item with an oracle margin of 0.0032; it moves by 0.0026. The worst
row is a SemIf item (0.0124). Every value is finite and every process's re-run is bit-equal. Two red arms
move p on the same graph: a state swapped for another record's moves 4 of 5 argmaxes (max |Δp| 0.905); a
grammatical "not" in five yes / no questions moves 2 of 5 (0.759). Transcript:
[`gate-kev-0.8b-readout.json`](gate-kev-0.8b-readout.json).

### From the request: Python and Swift

- `conversion/kev/host.py`, without the author's package, rebuilds every oracle row from the raw
  request (ids, `<|fim_suffix|>` / `<|box_end|>` indices, keys): 434/434 and 130/130 rows with two
  tokenizer implementations. `decide.py` on the AOT graph: hidden rows bit-equal to the gate's on 434/434
  and 130/130 rows ([`gate-kev-0.8b-host.json`](gate-kev-0.8b-host.json)).
- The Swift host ([`apps/Kev`](../../apps/Kev/), Release): ids from the raw request 434/434 and 130/130;
  on the same AOT asset every row's hidden state and every p equal the Python reference's bit for bit
  (434/434), every answer set's `json.dumps` and usage byte for byte (384/384), and the bar is the gate's
  (fixture 0.0124 / 0.00096). With the shared prefix every row's hidden state and p equal the direct run's
  (434/434). Its text half equals CPython on 2,787 rendered values, 599 requests (21 refused with the
  same messages), 203,635 doubles (`str`, `repr`, `round`) and 1,093 `json.dumps` texts. One word changed
  in one question turns its record red.
- JIT: the `.aimodel` specialized by the Swift runtime (GPU preferred, `expectFrequentReshapes`) equals
  the AOT asset bit for bit on 25 rows of 10 records. The specialization took 3.34 s (4.05 s with the
  tokenizer and the head); with the runtime's cache warm a load took 0.69 s
  ([`gate-kev-0.8b-swift.json`](gate-kev-0.8b-swift.json)).

### Time per decision on the Mac

Swift, Release CLI, the AOT asset, in one machine-wide GPU lock window (2026-10-04 12:13–12:27 JST). Two
processes per graph counted only when no other GPU job ran and the one-minute load average was at most 12;
each made 10 decisions per item after one warm-up, and the table gives the median (p10–p90) over the 20.
The earlier upload's graph (below, under Bundle) ran in the same window:

| request | tokens | calls | ms | p10–p90 | earlier upload's graph, ms |
|---|---:|---:|---:|---|---:|
| one question | 94 | 1 | 29.9 | 29.6–30.1 | 100.1 |
| one question | 380 | 3 | 89.2 | 88.7–90.4 | 404.4 |
| one question | 1,518 | 12 | 357.0 | 355.4–359.3 | 1,600.1 |
| one question | 1,802 | 15 | 447.9 | 446.6–449.8 | 1,897.6 |
| five questions on one 137-token state, each row from zero | 240 | 10 | 296.6 | 294.6–298.5 | 845.7 |
| the same five questions, the state run once (shared) | 240 | 6 | 186.4 | 185.4–187.7 | 327.7 |
| eight questions on that state, each row from zero | 320 | 16 | 474.2 | 472.2–475.9 | 1,377.0 |
| the same eight questions, shared | 320 | 9 | 278.8 | 277.2–281.7 | 461.0 |
| four questions on one 1,477-token state, each row from zero | 1,622 | 49 | 1,455.9 | 1,453.7–1,460.4 | 6,366.1 |
| the same four questions, shared | 1,622 | 16 | 485.4 | 483.8–489.4 | 1,749.0 |
| a follow-up question on a prepared 137-token state | — | — | 30.3 | — | 34.1 |
| a follow-up question on a prepared 1,477-token state | — | — | 31.5 | — | 51.5 |

Preparing the two states took 34.1–34.7 ms and 335.9–339.6 ms (the two processes' medians). In the
lock window of 2026-10-04 09:43–10:45 JST the `.aimodel` specialized by Swift, the shipped path, took 29.7 ms for the
94-token question and 186.5 ms for the five questions shared, against 30.0 and 189.1 ms for the AOT asset;
the Python reference took 33.4 and 207.1 ms. Transcript: [`gate-kev-0.8b-timing-mac.json`](gate-kev-0.8b-timing-mac.json).

### iPhone 18 Pro (iOS 27.0 24A437, Core AI arch h19p, 2026-10-04)

Through [`apps/KevGate`](../../apps/KevGate/), a headless gate app on the Swift host, Release, without the
increased-memory-limit entitlement (3,529 MB available at launch). The phone was on USB power, the
battery at 80 % and charging; every bench item started at thermal state nominal. Every p the app wrote
was re-scored on the Mac from its bit patterns.

- **Load.** The `.aimodel` specialized on the phone (GPU preferred, `expectFrequentReshapes`) took 4.26 s cold,
  3.60 s of it in `AIModel(contentsOf:)`, with +2,502,056,463 B of runtime cache and a peak footprint of 233 MB
  (an earlier KevGate build, the same day). With the cache warm, the gate run below loaded it in 1.50 s.
- **Gate** (run `20261004-140048`, a later KevGate build):

| set | questions | argmax (margin > 0.02) | near-ties agreeing | max \|Δp\| | mean of row means | bar |
|---|---:|---:|---:|---:|---:|---|
| fixture | 434 | 420/420 | 13/14 | 0.0110 | 0.00097 | PASS |
| held out | 130 | 125/125 | 5/5 | 0.0059 | 0.00083 | PASS |

  The shared prefix equals the direct run on all 20 multi-question records, and the re-run of the opening
  record is bit-equal. No hidden row equals the Mac's (a different GPU): their p differ by at most 0.0026,
  with every argmax equal (434/434). The fixture pass peaked at a 389 MB footprint.
- **Bench** (60 s of rest before each item, one warm-up, then 5 decisions; `latency_ms` = state resets,
  graph calls and the head). The earlier upload's graph ran in the same session:

| request | tokens | ms | earlier upload's graph, ms |
|---|---:|---:|---:|
| one question | 94 | 37.8 | 156.7 |
| one question | 380 | 110.6 | 627.9 |
| one question | 1,518 | 444.1 | 2,500.9 |
| one question | 1,802 | 559.1 | 2,977.8 |
| five questions on one 137-token state, each row from zero / shared | 240 | 361.9 / 223.2 | 1,331.4 / 507.4 |
| eight questions on that state, each row from zero / shared | 320 | 583.2 / 333.4 | 2,167.4 / 723.3 |
| four questions on one 1,477-token state, each row from zero / shared | 1,622 | 1,822.2 / 613.9 | 10,061.6 / 2,769.8 |
| the five questions on a prepared state (the prepare: 38.8 / 211.6 ms) | 240 | 183.4 | 295.5 |
| the four questions on a prepared state (the prepare: 415.3 / 2,451.4 ms) | 1,622 | 196.5 | 342.4 |

  An earlier session on the same phone, with an earlier KevGate build, gave 36.5 ms for the 94-token question
  against 157.1 ms for the earlier upload's graph. Transcript: [`gate-kev-0.8b-iphone.json`](gate-kev-0.8b-iphone.json).

Kev-4B on the phone: see the [Kev-4B card](../kev-4b/README.md).

## Forms measured

Every form below was exported, compiled and gated the same way. The times come from different windows, so
compare forms only within one window. Full records: [`gate-kev-0.8b-forms.json`](gate-kev-0.8b-forms.json).

Mac (Swift Release CLI, the 94-token question, median of the counted processes):

| form | 94-token decision, ms | window (2026-10-04, JST) | fixture max \|Δp\| | note |
|---|---:|---|---:|---|
| Gated DeltaNet unrolled, S = 16 (the earlier upload) | 100.1 | 12:13–12:28 | 0.0117 | |
| unrolled, S = 32 | 139.0 | 09:43–10:45 | 0.0118 | |
| unrolled, S = 64 | — | — | 0.0100 | not timed in a lock window (68.06 ms per call on a shared GPU) |
| in-graph chunk scan, S = 16 | — | — | 0.0135 | not timed in a lock window (10.90 ms per call on a shared GPU) |
| in-graph chunk scan, S = 32 | — | — | — | fails: non-near-tie argmax 112/127, max \|Δp\| 0.125, non-finite rows; breaks in fp32 torch too |
| Metal kernel, S = 16 / 32 / 64 | 63.6 / 42.7 / 41.5 | 09:43–10:45 | 0.0115 / 0.0134 / 0.0115 | |
| **Metal kernel, S = 128 (this release)** | **29.9** | 12:13–12:28 | **0.0124** | |
| Metal kernel, query length 2..512, host calls of at most 512 ids | 25.2 | 12:13–12:28 | 0.0112 | not shipped: memory grows (below) |
| the same graph, host calls of at most 256 / 128 ids | 25.1 / 25.0 | 12:13–12:28 | 0.0060 / 0.0060 (130 rows) | not shipped |
| int8, every linear (unrolled S = 16) | — | — | 0.0431 | fails the bar (Precision) |
| any form on the Neural Engine | — | — | — | not run: the recurrence's fp32 state is not an ANE element type ([`qwen3.5-static-ane.md`](../../knowledge/qwen3.5-static-ane.md)) |

iPhone 18 Pro (KevGate bench, the 94-token question):

| form | ms | session (2026-10-04, JST) | fixture max \|Δp\| |
|---|---:|---|---:|
| unrolled, S = 16 (the earlier upload) | 156.7 | from 11:56 | 0.0102 |
| Metal kernel, S = 16 / 32 / 64 | 92.4 / 58.5 / 49.3 | from 10:29 | 0.0110 / 0.0133 / 0.0098 |
| **Metal kernel, S = 128 (this release)** | **37.8** | from 11:56 | **0.0110** |
| Metal kernel, query length 2..512, host calls of at most 512 ids | 32.6 | from 11:56 | gate stopped by memory (below) |

**Several short questions on one state.** For these, the S = 64 graph takes less time than S = 128. In
the Mac's 09:43–10:45 JST window, five questions on one 137-token state, shared, took 145.0 ms with S = 64 and
189.1 ms with S = 128; one 94-token question took 41.5 and 30.0 ms. On the iPhone (the session from 10:29 JST)
the same pairs took 180.3 against 224.4 ms and 49.3 against 36.5 ms. The S = 64 bundle is not in this
release: it would need a new export and a new gate.

**Why the dynamic-length graph does not ship.** The graph whose query length is dynamic (2..512) passed the
gate on the Mac and decided the 94-token question in 25.2 ms. A process that keeps changing its call length
keeps growing, though. Cycling four call lengths (128, 256, 384, 512) three times added 467.1, 166.9 and
168.5 MB per lap in Swift and 460.7, 169.5 and 169.6 MB in Python. Alternating between two lengths did not
grow after the opening calls; bringing in a third added 19.1 MB, then 1.3 MB. On the iPhone the gate run
reached 3,497 MB with 43 MB available after 439 calls and stopped writing. A 2-second sleep gave back part of
the memory and loading the function again gave back none; creating the `AIModel` again gave it back. Records:
[`gate-kev-0.8b-forms.json`](gate-kev-0.8b-forms.json); notes:
[`knowledge/kev-port.md`](../../knowledge/kev-port.md).

## Precision

**int8 was measured and is not shipped.** It was measured on the unrolled S = 16 graph, the earlier upload;
the Metal-kernel graphs were not measured in int8. int8 per block of 32 over every decoder linear
(`symmetric_with_clipping`, weights only; the embedding table, the Gated DeltaNet conv1d and every norm
fp16) fails the bar:

| decoder (unrolled S = 16) | `main.mlirb` bytes | fixture max \|Δp\| / mean | held out max \|Δp\| / mean | bar |
|---|---:|---:|---:|---|
| fp16 | 1,506,481,909 | 0.0117 / 0.00097 | 0.0057 / 0.00081 | PASS |
| int8lin: every linear int8 | 1,040,063,350 | 0.0431 / 0.00326 | 0.0273 / 0.00294 | FAIL |
| int8mix: int8 except layers 0–11 | 1,273,276,920 | 0.0089 / 0.00103 | 0.0043 / 0.00082 | PASS |

int8lin breaks the bar on 10 fixture rows (above 0.01: 51 rows; fp16: 1). An fp32 torch bisect with the
exporter's own int8 weights (`conversion/kev/int8_bisect_torch.py`, 60 rows) reproduces it (0.0420
against the graph's 0.0431). Its rule, written before any result, asked for the worst row below 0.010
with at most half of the linear weights kept fp16; no set met it (the best: four kinds of linear kept
fp16, 45.5 % of the weights, 0.0141). Layers 0–11 kept fp16, exactly half and outside the rule's search,
give 0.0046 on the bisect rows; that is int8mix. It keeps 85 % of fp16's bytes, and it was chosen on the
fixture, so it does not ship ([`gate-kev-0.8b-int8.json`](gate-kev-0.8b-int8.json)).

**What the compiled asset holds.** `--expect-frequent-reshapes` adds an fp16 copy of every linear weight
to the compiled asset: for the unrolled graph the iPhone h19p asset is 2,505,674,625 bytes with it and
1,506,400,620 without, and this release's Mac h16c asset is 2,502,047,194 bytes against a 1,505,385,733-byte
`main.mlirb`. On a toy (an embedding and two linears) compiled with a fixed input length, the int8
version's `resources.bin` is 12,582,936 bytes on macOS and iOS, with and without the flag, the bytes of
the fp16 toy compiled without it, against an int8 `main.mlirb` of 10,623,080 bytes. An int8 `.aimodel`
downloads smaller; the compiled asset is not smaller by the same amount
([`../kev-4b/gate-kev-4b-int8.json`](../kev-4b/gate-kev-4b-int8.json)).

## ⬇️ Bundle

[mlboydaisuke/Kev-0.8B-CoreAI](https://huggingface.co/mlboydaisuke/Kev-0.8B-CoreAI), one folder under
`gpu-pipelined/`:

| file | what | bytes | sha256 |
|---|---|---:|---|
| `kev_0_8b_decode_fp16_metal_pf128.aimodel/main.mlirb` | the decoder, fp16 | 1,505,385,733 | `19d5a480…2c3f684e` |
| `metadata.json` | `kind: decision-backbone`, the readout contract, the gate | 12,590 | `6e50e41a…47282227` |
| `tokenizer/tokenizer.json` | the base model's, verbatim | 12,807,196 | `fe000e3e…d50d2927` |
| `tokenizer/tokenizer_config.json` | the base model's, verbatim | 16,712 | `e611fbcc…c47885de` |
| `head/head.safetensors` | the pointer head: q / k weight `[256, 1024]` and bias, fp32 | 2,099,576 | `12038b02…00893a42` |
| `head/kev_head.json` | head size, scale, temperature, delimiter ids, provenance | 2,046 | `73d51070…9d5c9d7c` |

The repository root carries the base model's `config.json`, the adapter's `adapter_config.json` and the
Apache-2.0 `LICENSE`, verbatim. `SHA256SUMS` lists every file. Its Hub revision is
[`3792e4b`](https://huggingface.co/mlboydaisuke/Kev-0.8B-CoreAI/tree/3792e4bf4ff6337afaa774b2a229580bcc0493e5) (2026-10-04).

**The earlier upload.** Hub revision
[`7793533`](https://huggingface.co/mlboydaisuke/Kev-0.8B-CoreAI/tree/7793533783de7c8f916c1cba1144a561fb70b321)
held another graph of this port, `kev_0_8b_decode_fp16_pf16`: the Gated DeltaNet recurrence unrolled step by
step, 16 tokens per call. It passed the same gate (fixture max |Δp| 0.0117, held out 0.0057). In the Mac
window above it took 100.1 ms for the 94-token question against 29.9 ms for this release, and on the iPhone
156.7 ms against 37.8 ms.

No AOT asset ships: the Swift runtime specializes the `.aimodel` correctly on the Mac and on the phone (the
JIT rows above). To compile one for the Mac anyway:

```bash
xcrun coreai-build compile kev_0_8b_decode_fp16_metal_pf128.aimodel --output aot --preferred-compute gpu \
    --platform macOS --architecture h16c --expect-frequent-reshapes
```

## Use it

Swift, with the [`Kev`](../../apps/Kev/) package (macOS 27 / iOS 27; the system CoreAI framework,
Accelerate and swift-transformers' tokenizer), on a download of the repository:

```swift
import Kev

let bundle = URL(filePath: "Kev-0.8B-CoreAI/gpu-pipelined/kev_0_8b_decode_fp16_metal_pf128")
let kev = try await KevDecider(bundle: bundle)              // asset: nil = the .aimodel, specialized here (GPU, frequent reshapes)
let response = try await kev.decide(requestJSON: requestData, shared: true)   // shared: the state's whole calls run once
print(PythonFormat.dumps(response, asciiOnly: false))
// {"model": "kev_0_8b_decode_fp16_metal_pf128", "answers": {"team": {"type": "choice", "choice": "billing", "confidence": …,
//  "probabilities": {…}}, "urgent": {"type": "noul", "noul": …}}, "usage": {"input_tokens": …, "output_tokens": …}, "latency_ms": …}

// questions that arrive later, on the same state
let prepared = try await kev.prepare(state: try JSONParser.parse(stateData))       // the state's whole calls, once
let later = try await kev.decide(prepared: prepared, questionsJSON: questionsData)  // only the questions' rows run
```

The same from the Mac CLI (`swift build -c release --package-path apps/Kev` builds `kev`):

```bash
kev run --bundle Kev-0.8B-CoreAI/gpu-pipelined/kev_0_8b_decode_fp16_metal_pf128 --asset jit \
    --request req.json --shared --out resp.json
```

A request, in the SystemOne-compatible request shape:

```json
{"model": "kev-0.8b",
 "state": "I was charged twice for my last order. Please refund one of the charges.",
 "questions": {
   "team": {"type": "choice", "instructions": "Which team should handle this?",
            "criteria": {"billing": "Charges and refunds", "shipping": "Deliveries", "returns": "Exchanges"}},
   "urgent": {"type": "noul", "instructions": "Does this need a reply today?"}}}
```

Python: `conversion/kev/decide.py run --model kev-0.8b --bundle <folder> --aimodelc <asset> --request
req.json [--shared] --out resp.json` is the gates' own read-out with `coreai.runtime`. It loads an AOT
`.aimodelc` (compile one with the command above); the Python gates never used the Python runtime's JIT.

## Reproduce

Environment: the zoo overlay venv (coreai-core 1.0.0b2, coreai-torch 0.4.1, torch 2.9.0, transformers
4.57.6), Xcode 27.0 RC. The author's code (the oracle and the merge) runs in its own venv built from the
author's `uv.lock` at tag `kev-1.0` (torch 2.8.0, transformers 5.17.0, peft 0.21.0). The steps, in order,
with every flag, are in [`conversion/kev/README.md`](../../conversion/kev/README.md).

```bash
# the merged checkpoint, with the author's script (tag kev-1.0)
(cd $ZOO_WORK_ROOT/_kev/kev-src && python scripts/merge_lora_checkpoint.py \
    --lora jaredpalmer/kev-0.8b@788ddbdd65715bb03a56788c822f6c632c9a551d --out $ZOO_WORK_ROOT/_kev/merged/kev-0.8b-v1.0)
# the decoder (--aot adds the h16c .aimodelc the Python gates load)
python conversion/kev/export_decoder.py fp16 --gdn-scan metal --prefill-chunk 128 --aot
# the published bundle: the gated .aimodel bytes and their metadata, without exporting again
python conversion/kev/export_decoder.py fp16 --gdn-scan metal --prefill-chunk 128 --metadata-only \
    --from-aimodel <bundles>/kev_0_8b_decode_fp16_metal_pf128/kev_0_8b_decode_fp16_metal_pf128.aimodel \
    --bundle-dir <ship>/kev_0_8b_decode_fp16_metal_pf128
```

Recipe: [`recipe.toml`](recipe.toml). Port notes: [`knowledge/kev-port.md`](../../knowledge/kev-port.md).

## Other formats

On the Hub (2026-10-04), not run here:

- Core ML: [`FluidInference/kev-0.8b-coreml`](https://huggingface.co/FluidInference/kev-0.8b-coreml), a
  different runtime (Core ML packages for iOS 18 / macOS 15, fp16 on the GPU): a fused multifunction
  package that packs a request's questions into one call, row packages, and the pointer head.
- ONNX: [`midudev/kev-0.8b-ONNX`](https://huggingface.co/midudev/kev-0.8b-ONNX) (ONNX Runtime Web, int4
  weights with the Gated DeltaNet projections and MLPs int8, the pointer head as `head.bin`).
- GGUF: [`ggml-org/Kev-0.8B-GGUF`](https://huggingface.co/ggml-org/Kev-0.8B-GGUF) ("a decision model,
  to be used via `/v1/systemone` API", per its card).
- RKLLM: [`ShiWarai/kev-0.8b-rknn`](https://huggingface.co/ShiWarai/kev-0.8b-rknn) (Rockchip RK3588,
  w8a8, the pointer head in NumPy on the CPU, rows up to 320 tokens).
- Merged weights: [`ayan4m1/kev-0.8b-merged`](https://huggingface.co/ayan4m1/kev-0.8b-merged).
- MLX: the author's repository serves Kev through MLX on Apple silicon.

No other Core AI conversion of Kev was listed on the Hub on 2026-10-04.

## License

Apache-2.0: the adapter and the head (jaredpalmer/kev-0.8b) and the base (Qwen/Qwen3.5-0.8B-Base); the
bundle inherits it, and the Hugging Face repository carries the base's `LICENSE`. The author's package
runs only in the oracle and the merge, at gate time; it is not part of the bundle. The fixture file
carries its own terms: its 144 SemIf authored144 items are MIT (github.com/TheoLeeCJ/SemIf at
`ca3ba65f`), with their copyright and permission notice; the transfer-v4 records are references to the
author's file (line and row hashes), not its text; of the 20 records written for the port, 11 are
published and 9 are withheld (an invented name in each was found in use on the web), keeping their ids
and numbers.

## Limits

From the [author's card](https://huggingface.co/jaredpalmer/kev-0.8b): English only; text generation,
chat, tool-call routing and fully automated consequential decisions about people are out of scope; the
temperature was fitted on the author's development rows (the card explains how to measure and refit it on
one's own data). This port adds:

- A row holds at most 3,968 tokens, while the author's server accepts states up to 65,536.
- A process's opening call is slower. On the Mac the opening 128-token call of each Python gate process
  took 211–1,188 ms, against a median of 29.9 ms for its later calls. On the phone the opening decision
  took 66.9 ms, against 37.8 ms in the bench of the same build.
- If you export this decoder with a dynamic query length yourself, keep the call lengths to one or two
  values. A process that cycles more lengths grows its memory until the `AIModel` is created again (measured
  on the Mac and the iPhone; which runtime layer keeps the memory is not isolated).
