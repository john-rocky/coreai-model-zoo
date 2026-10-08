Core AI is Apple's on-device ML runtime in iOS 27 / macOS 27 and the successor to Core ML: PyTorch models are exported with Apple's `coreai-torch` (LLMs: `coreai.llm.export`) into `.aimodel` bundles that run on the GPU or the Neural Engine, e.g. Qwen3-8B 4-bit decodes at 94 tok/s on an M4 Max GPU, MLX 90 under the same protocol ([apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench), macOS 27 beta, 2026-06).

# d1-3B — Core AI

[🤗 mlboydaisuke/d1-3B-CoreAI](https://huggingface.co/mlboydaisuke/d1-3B-CoreAI) · LFM Open License v1.0 · source [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B/tree/da1fe36a861f24690f27f622dca1d8688503d113) (revision `da1fe36`) · base [LiquidAI/LFM2.5-VL-3B](https://huggingface.co/LiquidAI/LFM2.5-VL-3B)

A **decision model** that also reads pictures. Give it a state (text, a JSON value, pictures, or a mix) and named
questions: `noul` (yes / no), `choice` (named options) or `score` (ordered levels). It returns a probability for every
option of every question. It never generates text. Requests and responses use the System One shape:
`{state, questions}` in, `{answers, usage}` out, one typed answer per question.

Liquid AI post-trained it from LFM2.5-VL-3B: a SigLIP2 vision encoder (27 layers, width 1,152) and the LFM2 hybrid
decoder (30 layers: 22 short-convolution and 8 grouped-query attention, hidden size 2,048, a 128,000-token vocabulary).
The provider's card: "You give it a state (text, JSON, images, or a mix) and a set of questions. It returns calibrated,
typed answers in one forward pass with zero output tokens." It reads the logits of a few option tokens at the prompt's
last position. The provider's card reports benchmark results and speeds; none of them are re-measured here.

This port exports the text decoder as one Core AI graph. The graph takes 64 token ids per call and returns the
final-norm hidden state at every position; it has no vocabulary head. The host reads the last position against the
tied embedding rows of the question's option tokens (a 2,134-row table ships beside the graph), keeps each option's
highest logit and takes a softmax over the options: the provider's readout. Pictures go through a second graph, the
vision tower, one call per crop. The gate is probability parity with the provider's own fp32 code, on every option of
every question: 393 fixture questions, 120 held-out ones and 24 questions about pictures.

Two decoders ship, with the same contract: the Mac runs the fp16 decoder (a 5.56 GB `.aimodel`), the iPhone one whose
MLP linears are int8 per block of 32 (3.70 GB). Both keep the attention projections fp32. On both devices the shipped
`.aimodel` is specialized where it runs. On the iPhone 18 Pro the app needs the
`com.apple.developer.kernel.increased-memory-limit` entitlement: without it, neither the decoder's on-device
specialization nor its ahead-of-time asset loads.

**Why two decoders.** On the phone, the int8mlp decoder at 64 tokens a call passes the bar and decides one question in
48.0 ms; each other form tried there is slower, misses the bar or does not load (Forms measured). On the Mac, the same
`.aimodel` specializes to a graph up to 25.7 % slower than its AOT asset, with other bits. The fp16 `.aimodel`
specializes to its AOT asset's rows bit for bit, at most 0.7 % slower, and keeps more room under the bar (fixture max
|Δp| 0.0039 against 0.0197).

## Readout contract

One row per question, every row from zeroed states. The ids are the provider's row form (`prompt.py`, `runner.py` in the
checkpoint) with the checkpoint's tokenizer:

```
<|startoftext|> <|im_start|>user\n [the pictures' token runs] state block \nQUESTION:\n
  question block <|im_end|>\n<|im_start|>assistant\n
```

- Special ids: `<|startoftext|>` 124894 (written as text; the tokenizer adds no BOS), `<|im_start|>` 124899,
  `<|im_end|>` 124900, `<|pad|>` 124893, `<image>` 124907.
- The state block: a string as is, any other JSON value as `json.dumps(state, ensure_ascii=False, indent=2)`, each
  followed by two newlines; a null state leaves the block and the `QUESTION:` line out.
- The question block: `choice` = the instructions, `Options:`, one line `code description` per option and `Reply with
  the option code only.`; `noul` = the instructions (with `Yes: …` / `No: …` lines when criteria are given) and `Reply
  with yes or no only.`; `score` = the instructions, one line `k level` per level and `Reply with a single digit 0-K
  only.` with K the last level's index.
- Option codes: the labels themselves when every label is one letter, else `A`, `B`, … up to 26 options, else `00`,
  `01`, …. A code that is not one free token takes the next free single-token alias, up to 1,639 options.
- The readout reads the hidden state at the row's last token. Each option has a group of ids: a choice's code, plus `
  code` with its leading space when that is one token; for `noul` the single-token forms of yes / Yes / YES against no /
  No / NO; for `score` the digits `0`..`K`. `z[id] = h · E[id]` in float64, with `E` the tied embedding rows
  (`head/option_rows.safetensors`: 2,134 ids × 2,048, fp32); an option's score is its group's highest `z`, and p is the
  softmax over the options. The provider takes a log-softmax over the whole vocabulary before the readout; that constant
  cancels in a softmax over the options (on random full-vocabulary logits the two agree within 1.75e-7).
- The response: `noul` → P(yes); `choice` → the argmax label, its p as `confidence`, every p; `score` → Σ k · p_k, the
  argmax p as `confidence`, every p and the legend; floats not rounded. `usage` = `{input_tokens, output_tokens: 0}`,
  counted as the provider counts (several questions: the shared prefix once, then each question's suffix).
- Pictures: each picture becomes one crop at its own aspect (sides multiples of 32, 64–256 image tokens) or 512 × 512
  tiles plus a thumbnail, after the provider's one-megapixel cap. A 384 × 384 picture is one crop of 144 image tokens.
  The prompt's `<image>` becomes the picture's run (`<|image_start|>` 125009 … `<|image_end|>` 125010, a tile marked
  `<|img_row_r_col_c|>` = 124908 + 10(r − 1) + (c − 1), the thumbnail `<|img_thumbnail|>` 125008), and the graph receives
  the k-th `<image>` of the row as id 128,000 + k, which reads row k of `image_embeds`.
- Length: a row holds at most 4,032 tokens. Its calls end at ⌈T / 64⌉ · 64 positions, which must stay at or below the
  graph's position bound of 4,095. The provider's card gives a context length of 32,768 tokens; here a longer row is
  refused before any call. One picture needs at most 2,810 image tokens, and a request whose pictures need more than
  the graph's 2,816 image rows together is refused too.
- A request is also refused whole when one of its readout ids is not in the option table (a choice of one-letter labels
  outside A–Z / a–z, such as `é` or `α`), and when a question breaks the provider's schema (no `instructions`, a score of more
  than 10 levels).

`conversion/d1/oracle_d1.py` runs the provider's code unchanged (`AutoModel` with `trust_remote_code`, fp32, CPU, one
thread) and records the reference: [`fixtures-d1-3b.json`](fixtures-d1-3b.json). The fixture is 361 records and 394
questions, 393 of them answerable (one question has no instructions, so the provider refuses its request). 356 records
come from the fixture of a LiteRT port of Kev: 60 MMLU rows of the Kev suite's transfer-v4 file, 20 rows from each of
six other sources, 20 score rows, the 144 SemIf authored144 items and 12 records written for that port. Three are the
model card's example requests, and two are long JSON states written for this port (1,464 and 3,419 tokens of state).
One question has an oracle top-2 margin of 0.02 or less (a near-tie) and is scored apart. The held-out set is 120 later
transfer-v4 records with their own oracle run. Twelve pictures drawn for this port carry 24 questions.

## Core AI shape

`conversion/d1/lfm2_d1_decoder.py`: the overlay's LFM2.5-VL decoder (`Lfm2VlPipelinedForCausalLM`) on d1-3B's language
weights, the vocabulary head replaced by the identity, the output the final-norm hidden state at every position. One
function, `main`, at a static 64 tokens.

| | name | shape, type |
|---|---|---|
| inputs | `input_ids` | [1, 64] int32 |
| | `position_ids` | [1, seq] int32, the ramp 0 .. seq − 1 |
| | `image_embeds` | [2816, 2048] fp16, the request's image rows (zero for a text row) |
| states | `keyCache`, `valueCache` | [8, 1, 8, ctx, 64] fp16, ctx up to 4,096 |
| | `convState` | [22, 1, 2048, 2] fp16 |
| output | `hidden` | [1, 64, 2048] fp16, every position |

Per row of T ids, from zeroed states: ⌈T / 64⌉ calls, call c with ids[64c : 64c + 64] and position_ids 0..64c + 63. The
last call is padded with `<|pad|>` (124893) and its padded rows are dropped. The bundle's `metadata.json` says `kind:
decision-backbone` and `language.prefill_chunk: 64`, and carries this order, the prompt, the option rules, the readout
formula, the refusals and the image-row contract.

`conversion/d1/lfm2_vl_tower.py`: the vision tower (SigLIP2 and the projector) with every shape static, one call per
crop. Its weights are stored fp16 and computed in fp32; the fp16-math tower missed the tower gate (lowest row cosine
0.9807) and is not shipped.

| | name | shape, type |
|---|---|---|
| inputs | `patches` | [1024, 768] fp32: the crop's 16 × 16 patches, (x − 127.5) / 127.5, zero after h · w |
| | `pos_table` | [1024, 1152] fp32: the 16 × 16 position table, resized to the crop's grid by the host |
| | `key_bias` | [1024] fp32: 0 for a patch, −inf for padding |
| | `unshuffle_idx` | [256, 4] int32: the 2 × 2 merge |
| output | `image_embeds` | [256, 2048] fp32: the crop's image tokens in its leading h · w / 4 rows |

**Host.** Builds the rows, decodes and cuts the pictures (the processor's uint8 bicubic resize from torch, written out),
runs the tower per crop and writes the image rows, runs the decoder's calls, converts the last position's hidden row to
float64, applies the readout in float64, and writes the response. `conversion/d1/decide.py` is the Python reference;
[`apps/D1`](../../apps/D1/) is the Swift one. With `shared` the state's whole calls run once and every question
continues from a copy of the states. A prepared state (`prepare(state:)`, then `decide(prepared:questionsJSON:)`) runs
the same calls in two steps, so questions that arrive later skip the state.

No engine is involved: the hosts drive the low-level runtime (`AIModel` + `loadFunction`, three zeroed states per row).
No runtime patch.

## Measured (Apple M4 Max, macOS 27.0 26A428, 2026-10-08 – 10-09)

The bar, fixed before any graph ran: the argmax equal to the oracle's on every question whose oracle top-2 margin is
above 0.02 (near-ties scored apart), max |Δp| ≤ 0.02 over every option of every question, the mean over rows of each
row's mean |Δp| ≤ 0.002, finite rows, and every process re-running its opening row bit for bit. The Python gates load
the AOT `.aimodelc` (`coreai-build compile … --platform macOS --preferred-compute gpu --architecture h16c
--expect-frequent-reshapes`; the tower without the last flag) with `SpecializationOptions.default()`.

### Before any graph: fp32 torch

- The decoder module in fp32 on the CPU, driven like the graph and read through the host's readout: 84/84 questions of
  56 records argmax-equal, max |Δp| 2.4e-6, lowest per-position hidden cosine 0.9999999998 on the 14 rows the oracle
  keeps. Chunks of 16, 32 and 64 tokens give the same hidden rows bit for bit, and chunked against one forward pass the
  rows differ by at most 5.2e-5. Its static form (below) equals the shipped form bit for bit on three rows
  ([`gate-d1-3b-torch-parity.json`](gate-d1-3b-torch-parity.json)).
- The tower graph against transformers 5.19's own `get_image_features` in fp32, on 64 crops of 18 pictures: every row's
  cosine 0.99999998 or more, max |Δ| 6.9e-5, a re-run bit-equal
  ([`gate-d1-3b-images.json`](gate-d1-3b-images.json), `tower_gate`).

### The graph alone on the Mac GPU

The oracle's rows in, each decoder's hidden row at the last position read through the host's readout:

| decoder | set | questions | argmax (margin > 0.02) | near-ties agreeing | max \|Δp\| | mean of row means | bar |
|---|---|---:|---:|---:|---:|---:|---|
| **fp16 (the Mac's)** | **fixture** | 393 | **392/392** | **1/1** | **0.0039** | **0.00018** | **PASS** |
| | pictures | 24 | 24/24 | — | 0.00049 | 0.00005 | PASS |
| **int8mlp (the iPhone's)** | **fixture** | 393 | **392/392** | **1/1** | **0.0197** | **0.00126** | **PASS** |
| | held out | 120 | 118/118 | 2/2 | 0.0140 | 0.00101 | PASS |
| | pictures | 24 | 24/24 | — | 0.0024 | 0.00023 | PASS |

The held-out set checks the int8 choice, made on the fixture. The int8mlp decoder's worst row is a SemIf item
(`semif_33d6e5da58f0fee2490d`, its `answer` question, oracle p 0.548 / 0.293 / 0.159): 0.0197 here, 0.0186 on the
phone. Across the chunk widths and the two machines the same question lands between 0.0180 and 0.0223; the phone's
32-token graph puts it at 0.0223, over the bar, and does not ship. Five red arms move p on each graph as they move it on
the oracle; on the int8mlp graph: one word changed in a question (max |Δp| 0.977), a grammatical "not" in two yes / no
questions (0.958 and 0.035), and a state swapped for another record's (0.840 and 0.951), each within 0.0017 of the
oracle's change. Every value is finite and every process's re-run is bit-equal. Transcripts:
[`gate-d1-3b-readout.json`](gate-d1-3b-readout.json),
[`gate-d1-3b-heldout.json`](gate-d1-3b-heldout.json), [`gate-d1-3b-images.json`](gate-d1-3b-images.json),
[`gate-d1-3b-red.json`](gate-d1-3b-red.json).

The released bundles are these exports with their debug locations stripped (Bundle, below). On the Mac the stripped
bundles give every row of the gates above bit for bit: the fixture with the red arms (406 rows for each decoder), the
held-out set (120 rows), the pictures (24 rows and 40 tower crops for each decoder) and the tower gate (64 crops)
([`gate-d1-3b-strip.json`](gate-d1-3b-strip.json)).

### From the request: Python and Swift

- `conversion/d1/host.py`, without the provider's package, rebuilds every oracle row from the raw request (the ids, the
  answer slot, the option groups and keys): 393/393 questions with two tokenizer implementations, and on 361/361 records
  the provider's request ids, answers and response body.
- The Swift host ([`apps/D1`](../../apps/D1/), Release) builds the same ids from the raw request (393/393, and 2,000
  stress texts), cuts every fixture picture into the Python reference's tower inputs bit for bit (64 crops of 18 PNG
  pictures), and on the same AOT assets gives every hidden row, p and response of the Python reference bit for bit: the
  fp16 decoder at 16 tokens a call on 393 text and 24 picture rows, the int8mlp decoder at 64 on 51 rows. With the
  shared prefix every record's rows equal the direct run (360 of 360 accepted records).
- JIT: the fp16 `.aimodel` specialized by the Swift runtime gives the AOT asset's rows bit for bit (at 16 tokens a call
  on 54 rows; at 64, the Mac's decoder, on all 9 timed rows). The int8mlp `.aimodel`'s JIT does not (0 of 9 timed rows; max
  |Δp| 0.00046 against the AOT asset), so the Mac does not ship it. JPEG pictures decode differently in ImageIO and
  libjpeg (by up to 57 levels with 4:2:0 chroma); PNG pictures match exactly
  ([`gate-d1-3b-swift.json`](gate-d1-3b-swift.json)).

### Time per decision on the Mac

Swift, Release CLI, inside the machine-wide GPU lock (2026-10-09, 08:04–08:08 JST): the shipped (stripped) Mac
decoder's `.aimodel` specialized by the `d1` process (the shipped path: GPU preferred, `expectFrequentReshapes`; the
tower's `.aimodel` the same way), and the same bundle's AOT asset as a control. Two processes per form and request, counted only when no other
GPU job ran and the one-minute load average stayed at or below 12; each made 20 decisions after one warm-up, and the
table gives the median (p10–p90) over the 40. A decision is the picture's tower call and image rows, the graph calls and
the readout; building the rows and decoding the picture are not in it.

| request | input tokens | calls | ms | p10–p90 | AOT asset, ms |
|---|---:|---:|---:|---|---:|
| one question (the card's refund question) | 39 | 1 | 34.3 | 34.0–34.5 | 34.0 |
| three questions on that state, shared | 116 | 3 | 102.2 | 102.1–102.8 | 101.9 |
| the same three, each row from zero | 116 | 3 | 102.1 | 101.9–102.8 | 102.1 |
| one question on a 3,470-token state | 3,470 | 55 | 1,888.0 | 1,886.1–1,890.5 | 1,881.8 |
| one question on a 384 × 384 picture (1 crop, 144 image tokens) | 186 | 3 | 202.3 | 201.8–203.2 | 201.7 |

The specialized graph gives the AOT asset's hidden rows and p bit for bit on every timed question, so the AOT gates
above hold for it. Specializing it took 9.52 s (`AIModel`) + 0.69 s (function) in the window's opening process and
wrote 10,368,248 KiB of runtime cache; the tower's took 2.05 s and 845,612 KiB. The tower's crop is 100.2 ms of the
picture decision. At 64 tokens a call the three questions share no whole call, so shared and direct run the same calls.
Transcript: [`gate-d1-3b-timing-mac.json`](gate-d1-3b-timing-mac.json).

### iPhone 18 Pro (iOS 27.2 24B5099f, Core AI arch h19p, 2026-10-09)

Through [`apps/D1Gate`](../../apps/D1Gate/), a headless gate app on the Swift host, Release, with
`com.apple.developer.kernel.increased-memory-limit` (6,432 MB available at launch; 3,530 MB without the entitlement).
The phone was on USB power, the battery at 80 % and charging; every bench item started at thermal state nominal. Every
p the app wrote was re-scored on the Mac from its bit patterns. These runs used the int8mlp bundle as exported, before its
debug locations were stripped (Bundle, below); the stripped bundle has not run on the phone yet.

- **Load.** The `.aimodel` specialized on the phone (GPU preferred, `expectFrequentReshapes`), the tower's `.aimodel`
  the same way: 28.1 s cold, 20.8 s of it in `AIModel(contentsOf:)`, with +4,779 MB of runtime cache and a peak
  footprint of 270 MB during the load. With the cache warm a load took 3.6 s.
- **Gate** (run `r10a1c-012336`):

| set | questions | argmax (margin > 0.02) | near-ties agreeing | max \|Δp\| | mean of row means | bar |
|---|---:|---:|---:|---:|---:|---|
| fixture | 393 | 392/392 | 1/1 | 0.0186 | 0.00125 | PASS |
| pictures | 24 | 24/24 | — | 0.0021 | 0.00023 | PASS |

  The red arms are red (5/5), the shared prefix equals the direct run on all 23 multi-question records, and the re-run
  of the opening record is bit-equal. No hidden row equals the Mac's (a different GPU): their p differ by at most
  0.0037, with every argmax equal (417/417). The run's footprint peaked at 547 MB.
- **Bench** (60 s of rest before each item, one warm-up, then 5 decisions; the 3,470-token state with 30 s of rest
  before each decision):

| request | ms | 5 runs |
|---|---:|---|
| one question | 48.0 | 47.3–50.3 |
| three questions on that state, shared / each row from zero | 146.7 / 145.2 | 142.7–150.2 / 143.0–150.9 |
| one question on a 3,470-token state | 2,794.4 | 2,781.6–2,802.4 |
| one question on a 384 × 384 picture (the tower's crop: 418.7) | 570.0 | 568.5–575.6 |

Without the entitlement neither decoder form loads on this phone: the on-device specialization dies with
`std::bad_alloc` while it folds a weight transpose, and the ahead-of-time asset (8.74 GB) with a `SIGSEGV` in the
delegate compile, both within two seconds. Transcript: [`gate-d1-3b-iphone.json`](gate-d1-3b-iphone.json).

## Forms measured

Every form below was exported, compiled and gated the same way. The times come from different windows, so compare forms
only within one window. Full records: [`gate-d1-3b-forms.json`](gate-d1-3b-forms.json).

Mac (Swift Release CLI, median of the counted processes):

| form | one question, ms | 3,470-token state, ms | window | fixture max \|Δp\| | note |
|---|---:|---:|---|---:|---|
| fp16, S = 16 / 32 / 64 | 61.8 / 50.1 / 34.1 | 4,446.2 / 2,727.9 / 1,875.0 | 1 | 0.0046 / 0.0058 / 0.0039 | |
| int8mlp, S = 16 / 32 / 64 | 61.9 / 50.3 / 34.0 | 4,680.8 / 2,734.1 / 1,882.6 | 1 | 0.0187 / 0.0184 / 0.0197 | S = 64: the iPhone's decoder |
| int8mlp per block of 16, S = 16 | 62.4 | 4,543.3 | 1 | 0.0156 | |
| fp16, S = 16, JIT | 61.9 | 4,701.7 | 1 | = AOT, bit for bit | |
| int8mlp, S = 128 | 55.5 | 1,572.4 | 2 | 0.0174 | the S = 64 control in window 2: 34.6 / 1,876.2 |
| int8mlp, static form, S = 64 / 16 | 40.3 / 76.8 | 2,193.9 / 5,536.9 | 2 | 0.0189 / 0.0162 | |
| fp16, static form, S = 64 | 40.4 | 2,196.3 | 2 | 0.0026 | |
| int8mlp, S = 64, JIT | 42.8 | 2,358.9 | 3 | not gated: other bits than its AOT asset | the AOT control in window 3: 34.0 / 1,877.0 |
| fp16, S = 64, JIT, as exported | 34.2 | 1,888.5 | 4 | = AOT, bit for bit | the AOT control in window 4: 34.1 / 1,878.1 |
| **fp16, S = 64, JIT, debug locations stripped (the Mac's shipped path)** | **34.3** | **1,888.0** | 5 | **= AOT, bit for bit** | the AOT control in window 5: 34.0 / 1,881.8 |
| int8lin: every MLP and conv-projection linear int8 | — | — | — | 0.0227 | fails the bar |
| int8lin per block of 16 | — | — | — | 0.0205 | fails the bar |
| int8mix: int8lin with layers 2, 3, 6, 8, 9, 10, 12 fp16 | — | — | — | 0.0203 | fails the bar |
| int4lin: the same linears int4 | — | — | — | 0.668 | fails: argmax 373/392 |
| any form on the Neural Engine | — | — | — | — | no-go (below) |

Windows: 1 = 2026-10-08 18:16–20:01 JST, 2 = 2026-10-09 02:01–02:21, 3 = 07:04–07:08, 4 = 07:15–07:20, 5 = 08:04–08:08.
S = the tokens a call.

iPhone 18 Pro (D1Gate, with the entitlement):

| form | one question, ms | gate max \|Δp\| | note |
|---|---:|---:|---|
| int8mlp, S = 16, the phone's JIT | 125.5 | 0.0180 | 3 calls of 40.2 ms; thermal fair |
| int8mlp, S = 32, the phone's JIT | — | 0.0223 | one question over the bar: not timed |
| **int8mlp, S = 64, the phone's JIT (this release)** | **48.0** | **0.0186** | |
| int8mlp, S = 16, AOT h19p with `--expect-frequent-reshapes` (8.74 GB) | — | 0.0075 (20 questions) | 55.9 ms a call against the JIT's 40.5 |
| int8mlp, S = 16, AOT h19p without it (3.70 GB) | — | — | no-go: a call right after a new position length is specialized can return a wrong row, then `SIGTRAP` |
| int8mlp, static form, S = 64: AOT h19p and JIT | — | 0.0208 | one question over the bar; the two give the same rows bit for bit |

**Long states.** S = 128 decides the 3,470-token state in 1,572.4 ms, 16.2 % faster than S = 64 in the same window, but
one question takes 55.5 ms against 34.6 (+60.5 %): a call of 128 tokens costs 56.2 ms against 34.1 for 64. Its asset is
not in this release.

**The static form.** `conversion/d1/lfm2_d1_static.py` gives each call its own positions and keeps the caches at a
fixed 4,096 slots with the attention mask built in the graph. It has no dynamic dimension, so an AOT compile without
`--expect-frequent-reshapes` specializes it once (5.56 GB: the compile folds the int8 weights into fp16) and nothing is
specialized again at run time. It is slower per call (a 64-token call 39.9 against 34.1 ms on the Mac), and on the phone
it misses the bar on one question, so it does not ship.

**The Neural Engine.** Compiled with `--preferred-compute neural-engine`, the dynamic decoder gets no Neural Engine
region and runs on the GPU bit for bit. At a fixed position length the runtime builds one Neural Engine region, 4.70 GB
of IR, the size of its 134 MLP and conv-projection linears in fp16, and the Neural Engine compiler fails on it every
time (`MLIR MPS to ANEC conversion failed`, its nonbonded phase). The fp16w32 tower computes in fp32, which the Neural
Engine does not take; the fp16-math tower runs there but misses the tower bar (lowest row cosine 0.9755).

## Precision

The Mac's decoder is fp16 with the attention projections in fp32. The iPhone's decoder quantizes the MLP's linears alone
(90 linears, per block of 32, symmetric with clipping, weights only); the conv-mixer projections, the embedding table,
the conv1d and the norms stay fp16 and the four attention projections of each full-attention layer fp32. int8 over
every MLP and conv-projection linear (int8lin) breaks the bar at 0.0227. An fp32 torch instrument with the exporter's
own int8 weights reproduces it (0.0235 on the bisect rows), so the error comes from the int8 weights. No layer set met
the bisect's rule, written before any result. Keeping the conv projections fp16 passes: 0.0187 on the fixture and 0.0137
on the held-out set at 16 tokens a call, 0.0197 and 0.0140 at 64.

| decoder, S = 64, as exported | `main.mlirb` bytes | compiled with `--expect-frequent-reshapes` | static form, compiled once |
|---|---:|---:|---:|
| fp16 | 5,562,548,035 | 10,600,779,479 | 5,562,784,605 |
| int8mlp | 3,704,646,757 | 8,742,889,020 | 5,562,787,469 |

**What the compiled asset holds.** `--expect-frequent-reshapes` adds an fp16 copy of every linear to the compiled asset,
and a compile that specializes once folds the int8 weights into fp16 constants (its `stats.json` still counts Int8
elements). An int8 `.aimodel` downloads 1.86 GB smaller than the fp16 one. Compiled with `--expect-frequent-reshapes` it
stays 1.86 GB smaller (8.74 against 10.60 GB); compiled once in the static form, both are 5.56 GB. The phone's JIT keeps
a cache of 4,779 MB for the int8mlp `.aimodel`.

## ⬇️ Bundle

[mlboydaisuke/d1-3B-CoreAI](https://huggingface.co/mlboydaisuke/d1-3B-CoreAI), three folders under `gpu-pipelined/`: the
Mac's decoder, the iPhone's decoder and the tower. The exporter writes every op's source location into `main.mlirb`,
local paths included; each folder is its exported bundle with those locations stripped (`conversion/d1/strip_bundle.py`,
the `_s` in its name), the ops and the weights unchanged. Each `metadata.json` records the strip (`strip`: both
`main.mlirb` sha256).

| file | what | bytes | sha256 |
|---|---|---:|---|
| `d1_3b_decode_fp16_pf64_s/d1_3b_decode_fp16_pf64_s.aimodel/main.mlirb` | the Mac's decoder, fp16 | 5,562,302,268 | `e54d5d41…f4ecbc5c` |
| `d1_3b_decode_fp16_pf64_s/metadata.json` | `kind: decision-backbone`, the readout contract, the strip record | 12,329 | `f9e4ad70…e313029a` |
| `d1_3b_decode_int8mlp_pf64_s/d1_3b_decode_int8mlp_pf64_s.aimodel/main.mlirb` | the iPhone's decoder, int8mlp | 3,704,370,745 | `1836fba8…8079473f` |
| `d1_3b_decode_int8mlp_pf64_s/metadata.json` | the same contract, its `compression` block, the strip record | 15,005 | `71d8739f…6051e32b` |
| `<decoder>/tokenizer/tokenizer.json` | the checkpoint's, verbatim | 17,905,750 | `8096ecb9…8349fcee` |
| `<decoder>/tokenizer/tokenizer_config.json` | the checkpoint's, verbatim | 507 | `ef6770d1…45feb0ea` |
| `<decoder>/tokenizer/chat_template.jinja` | the checkpoint's, verbatim | 5,436 | `86f47704…22f61cca` |
| `<decoder>/head/option_rows.safetensors` | the option table: 2,134 ids × 2,048, fp32 | 17,491,440 | `e43cfaee…ad948522` |
| `<decoder>/head/option_rows.json` | the table's ids and their strings | 76,692 | fp16 `133b9d71…6c1a5acf`, int8mlp `41bb7fc3…57ab8bba` |
| `d1_3b_vision_fp16w32_s/d1_3b_vision_fp16w32_s.aimodel/main.mlirb` | the tower, fp16 weights / fp32 math | 852,915,980 | `358b7de0…59591161` |
| `d1_3b_vision_fp16w32_s/metadata.json` | `kind: vision-tower`, the crop contract | 6,956 | `1d84a4c0…0b71f669` |
| `d1_3b_vision_fp16w32_s/host/position_embedding.safetensors` | the 16 × 16 position table, fp32 | 1,179,744 | `d0f11651…088a4a0e` |

The two decoder folders hold the same `tokenizer/` files and option table; their `option_rows.json` differ only in the
time they were written. The repository root carries the checkpoint's `config.json` and the LFM Open License v1.0
`LICENSE`, verbatim; each bundle folder carries the same `LICENSE`. `SHA256SUMS` lists every file.

No AOT asset ships: the Swift runtime specializes each `.aimodel` where it runs (the JIT rows above). To compile the
Mac's decoder ahead of time anyway (10.60 GB with the flag's fp16 copies):

```bash
xcrun coreai-build compile d1_3b_decode_fp16_pf64_s.aimodel --output aot --preferred-compute gpu \
    --platform macOS --architecture h16c --expect-frequent-reshapes
```

## Use it

Swift, with the [`D1`](../../apps/D1/) package (macOS 27 / iOS 27; the system CoreAI framework, Accelerate and
swift-transformers' tokenizer), on a download of the repository:

```swift
import D1

let root = URL(filePath: "d1-3B-CoreAI/gpu-pipelined")
#if os(macOS)
let decoder = root.appending(path: "d1_3b_decode_fp16_pf64_s")     // the Mac's decoder
#else
let decoder = root.appending(path: "d1_3b_decode_int8mlp_pf64_s")  // the iPhone's: the app needs increased-memory-limit
#endif
let d1 = try await D1Decider(bundle: decoder)
try await d1.loadGraph(asset: .jit, tower: root.appending(path: "d1_3b_vision_fp16w32_s"), towerAsset: .jit)
                                                         // .jit = each .aimodel, specialized here
let body = try await d1.decide(requestJSON: requestData, shared: true)                // shared: the state's calls once
let seen = try await d1.decide(requestJSON: requestData, images: [pictureURL])        // the k-th file = the k-th <image>
print(PythonFormat.dumps(body, indent: 2, asciiOnly: false))

// questions that arrive later, on the same state
let prepared = try await d1.prepare(state: try JSONParser.parse(stateData))           // the state's whole calls, once
let later = try await d1.decide(prepared: prepared, questionsJSON: questionsData)    // only the questions' rows run
```

The same from the Mac CLI (`swift build -c release --package-path apps/D1` builds `d1`):

```bash
d1 decide --bundle d1-3B-CoreAI/gpu-pipelined/d1_3b_decode_fp16_pf64_s --asset jit \
    --tower d1-3B-CoreAI/gpu-pipelined/d1_3b_vision_fp16w32_s --tower-asset jit \
    --request req.json --shared --out resp.json
```

A request (the model card's example):

```json
{"state": "I was charged twice this month, please refund one of them.",
 "questions": {
   "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
   "team": {"type": "choice", "instructions": "Which team should handle this?",
            "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                         "fraud": "Suspected unauthorised use"}},
   "urgency": {"type": "score", "instructions": "How urgent is this?",
               "criteria": ["Can wait", "Today", "Blocking the customer now"]}}}
```

A request with pictures adds `"images": ["photo.png"]` (files relative to `--images`, by default the request's folder).
An iOS app needs `com.apple.developer.kernel.increased-memory-limit` in its entitlements (above).

Python: `conversion/d1/decide.py run --bundle <folder> --request req.json [--tower <tower folder>] [--shared] --out
resp.json` is the gates' own read-out with `coreai.runtime`. It loads an AOT `.aimodelc` (compile one with the command
above); the Python gates never used the Python runtime's JIT.

## Reproduce

Environment: the zoo overlay venv (coreai-core 1.0.0b2, coreai-torch 0.4.1, coreai-opt 0.2.1, torch 2.9.0, transformers
4.57.6), Xcode 27.0 RC. The provider's code (the oracle) runs in its own venv with transformers 5.19.0. The steps, in
order, with every flag, are in [`conversion/d1/README.md`](../../conversion/d1/README.md).

```bash
cd conversion/d1; K=$ZOO_WORK_ROOT/_d1_3b; S=$K/exports/bundles_ship
# the bundles as exported (--aot adds the h16c .aimodelc the Python gates load)
python export_decoder.py fp16 --prefill-chunk 64 --aot        # the Mac's decoder
python export_decoder.py int8mlp --prefill-chunk 64 --aot     # the iPhone's decoder
python export_vision.py --dtype fp16w32 --aot
# the shipped bundles: each copied to a new name with the export's debug locations (local paths) stripped
for b in bundles/d1_3b_decode_fp16_pf64 bundles/d1_3b_decode_int8mlp_pf64 vision/d1_3b_vision_fp16w32; do
  python strip_bundle.py $K/exports/$b $S/$(basename $b)_s --aot
done
# the gates on the shipped bundles: the fixture with the red arms (each decoder), the held-out set (the int8 decoder),
# a picture record end to end
python readout_gate.py run $S/d1_3b_decode_fp16_pf64_s --red --transcript <fixture gate>.json
python readout_gate.py run $S/d1_3b_decode_int8mlp_pf64_s --oracle $K/oracle/heldout/records_oracle.json \
    --tag heldout_int8mlp_pf64_s --transcript <held-out gate>.json
python decide.py run --bundle $S/d1_3b_decode_fp16_pf64_s --tower $S/d1_3b_vision_fp16w32_s \
    --request <picture record request>.json --out <response>.json --trace <trace>.json
# the Mac timing window: the shipped .aimodel's JIT against its AOT asset (the Release CLI takes the GPU lock itself)
D1_TOWER=$S/d1_3b_vision_fp16w32_s \
D1_FORMS="fp16_pf64_s=$S/d1_3b_decode_fp16_pf64_s jit_fp16_pf64_s=$S/d1_3b_decode_fp16_pf64_s@jit" ../../apps/D1/_time_mac.sh
```

A rebuild does not reproduce `main.mlirb` byte for byte (each export names its weight resources at random), so check
it with the gates (`readout_gate.py`, `decide.py`), not with sha256. Recipe: [`recipe.toml`](recipe.toml). Port notes:
[`knowledge/d1-port.md`](../../knowledge/d1-port.md).

## License

LFM Open License v1.0 (`license: other`, `license_name: lfm1.0`): the weights and the provider's code
(LiquidAI/d1-3B); the bundles inherit it, and the Hugging Face repository carries the full `LICENSE`. The licence grants
commercial use only to a user or legal entity below 10 million US dollars of annual revenue (its Section 5). Changes in
this repository's bundles, as its Section 4(b) asks: the checkpoint's language and vision weights were re-exported as
Core AI graphs (`.aimodel`, the export's debug locations stripped): two decoders and the tower. The decoders' vocabulary head is removed (its tied rows for the
readout ids ship as `head/option_rows`), the iPhone decoder's MLP linears are quantized to int8 per block of 32, and the
other weights are stored as fp16 (the decoders' attention projections as fp32); every bundle's `metadata.json` records
the details. The provider's code runs only in the oracle, at gate time; it is not part of the bundles.

The fixture file carries its own terms. Its 144 SemIf authored144 items are MIT (github.com/TheoLeeCJ/SemIf at
`ca3ba65f`), with their copyright and permission notice. The transfer-v4 records are references to the Kev suite's file
(line and row hashes), not its text; the held-out set the same. The 12 records written for the LiteRT Kev port keep
only their ids and numbers: they name invented people and companies that were never checked against real ones. The
model card's three example requests and the two long states written for this port carry their text. The 12 pictures
are CC0-1.0 and are not stored here: `conversion/d1/make_fixture_images.py` redraws them (their sha256 are in the file).

## Limits

From the [provider's card](https://huggingface.co/LiquidAI/d1-3B): it is not a chat model and does not write text. This
port adds:

- A row holds at most 4,032 tokens, while the provider's card gives a context length of 32,768.
- Loading an `.aimodel` with a cold runtime cache specializes it: 10.2 s for the Mac's decoder, 28.1 s for the iPhone's
  on the phone. With the cache warm a load takes 0.7 s on the Mac and 3.6 s on the phone.
- Measured on one iPhone 18 Pro with iOS 27.2 beta (24B5099f) and one M4 Max; other iPhones and Macs were not measured.
- The oracle is the provider's code in fp32 on the CPU; a bf16 run of the provider's code was not compared, and no
  form here runs on the CPU alone.
- JPEG pictures decode with ImageIO in the Swift host and with libjpeg in the provider's Python path; the pixels can
  differ (by up to 57 levels with 4:2:0 chroma). PNG pictures give the same pixels.
