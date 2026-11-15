# Kev-4B — Core AI

[🤗 mlboydaisuke/Kev-4B-CoreAI](https://huggingface.co/mlboydaisuke/Kev-4B-CoreAI) · Apache-2.0 · source [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b/tree/6cfce5c2fa4b4bd64026336ab649c5ca78857d52) (tag `v1.0`, commit `6cfce5c`) · base [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base/tree/1001bb4d826a52d1f399e183466143f4da7b741b) (revision `1001bb4`)

A **decision model**. Give it a state (text, or a JSON object or array) and typed questions: `noul`
(yes / no), `choice` (named options) or `score` (ordered levels). It returns a probability for every
option of every question. It never generates text. Requests and responses use the SystemOne-compatible
request shape: `{model, state, questions}` in, `{model, answers, usage}` out, one typed decision per
question.

Jared Palmer trained it as a rank-16 LoRA adapter and a pointer head on Qwen3.5-4B-Base (32 layers:
24 Gated DeltaNet and 8 full attention, hidden size 2,560). The author's card: "It is intended for
developers who classify, route, triage or check documents and who need probabilities that can be
thresholded, for example to send uncertain cases to human review." Each question is its own row: the
state, then the question and its options. The head scores each option's closing token against the
question's last token. The author's card reports benchmark results; none of them are re-measured here.

This port merges the adapter into the base with the author's own merge script and exports the text
backbone as one Core AI graph. The graph takes 128 token ids per call and returns the final-norm hidden
state at every position; it has no vocabulary head. Each Gated DeltaNet layer runs its recurrence in an
fp32 Metal kernel, one GPU dispatch per layer per call. Weights are fp16 (8.41 GB). The host runs the
pointer head from the author's head weights. The gate is probability parity with the author's own fp32
code, on every option of every question, on a fixture of 434 questions and on a held-out set of 130.

It runs on the Mac GPU. This release is for the Mac: the one load of an iPhone AOT asset of an earlier
graph on an iPhone 18 Pro crashed (below). [Kev-0.8B](../kev-0.8b/README.md), the same port at 0.8B, is
measured on the iPhone 18 Pro.

## Readout contract

The contract is Kev-0.8B's, with this checkpoint's head and temperature; the full description is in the
[Kev-0.8B card](../kev-0.8b/README.md#readout-contract). In short:

```
[<|fim_prefix|>] + render(state) + [<|fim_middle|>] + render(instructions)
  + for each option: [<|box_start|>] + option_text + [<|box_end|>]
  + [<|fim_suffix|>]
```

- One row per question, zeroed states, positions 0..L − 1. Delimiters from the base tokenizer: 248060,
  248061, 248049 / 248050, 248062; pad = eos = 248044; no bos. The 4B base's `tokenizer.json` is
  byte-identical to the 0.8B base's; its `tokenizer_config.json` differs in the chat template's thinking
  default, which this readout does not use. The `kev-4b` repository still carries Qwen3-era
  `vocab.json`, `merges.txt` and `added_tokens.json` (other delimiter ids); the author's loader reads the
  tokenizer from the base, and so does this port.
- The head: `z_k = ((W_k h_opt_k + b_k) · (W_q h_decide + b_q)) / 16`, `p = softmax(z / T)` within the
  question, T = 2.406050072164233 (`head.pt`); `W_q`, `W_k` are `[256, 2560]` with biases, fp32.
- A row holds at most **3,968 tokens**. The last call is padded to 128 tokens and must end at or below
  the graph's position bound of 4,095.
- Shared prefix (optional, exact): the state's whole 128-token calls run once and the four states are
  copied per question. Every row's hidden state equals its direct run bit for bit (below).

The fixture and the held-out set are Kev-0.8B's (434 and 130 questions), with this checkpoint's own fp32
oracle run (the author's package, one CPU thread): [`fixtures-kev-4b.json`](fixtures-kev-4b.json). It has
twelve near-ties on the fixture and four on the held-out set.

## Core AI shape

The same module as Kev-0.8B, `conversion/kev/qwen3_5_kev_decoder.py`, with no code change: the overlay's
stateful Qwen3.5 text decoder on the merged weights, no vocabulary head. One function, `main`, at a
static 128 tokens. Each Gated DeltaNet layer runs its recurrence in the overlay's fp32 Metal chunk kernel
(`qwen3_5_gdn_metal`): one GPU dispatch per layer per call, the recurrent state kept in fp32 through the
call.

| | name | shape, type |
|---|---|---|
| inputs | `input_ids` | [1, 128] int32 |
| | `position_ids` | [1, seq] int32, the ramp 0 .. seq − 1 |
| states | `keyCache`, `valueCache` | [8, 1, 4, ctx, 256] fp16, ctx up to 4,096 |
| | `convState`, `recState` | [24, 1, 8192, 3], [24, 1, 32, 128, 128] fp16 |
| output | `hidden` | [1, 128, 2560] fp16, every position |

Per row of T ids, from zeroed states: ⌈T / 128⌉ calls, the last padded with `<|endoftext|>` (248044) and
its padded rows dropped. `metadata.json` says `kind: decision-backbone` and `language.prefill_chunk: 128`,
and carries the readout contract. The host (`conversion/kev/decide.py`, [`apps/Kev`](../../apps/Kev/)) is
Kev-0.8B's: the head in float64 from `head/head.safetensors`, p rounded to fp32 once, the
SystemOne-compatible response, the shared prefix and the prepared state. No engine, no runtime patch.

## Measured (Apple M4 Max, macOS 27.0 26A428, 2026-10-03 – 10-04)

The bar is Kev-0.8B's, fixed before any graph ran: the argmax equal to the oracle's on every question
whose oracle top-2 margin is above 0.02 (near-ties listed apart), max |Δp| ≤ 0.02 over every option of
every question, the mean over rows of each row's mean |Δp| ≤ 0.002, every process's re-run bit-equal.
The Python gates load the AOT `.aimodelc` (h16c, `--expect-frequent-reshapes`) with
`SpecializationOptions.default()`; their times are on a GPU shared with other work.

### Before any graph: the merge and fp32 torch

- The merged checkpoint (`scripts/merge_lora_checkpoint.py`, tag `kev-1.0`; fp32, four shards of
  16.8 GB, 426 tensors, 248 of them adapted) answers through the author's loader as the adapter
  checkpoint does: 434/434 questions, logits bit-equal. `W + (B @ A) × 2` reproduces three merged
  tensors bit for bit (the adapter moves them by 1.4–2.0 %)
  ([`gate-kev-4b-merge.json`](gate-kev-4b-merge.json)).
- The decoder module in fp32 on the CPU, driven like the graph with the recurrence unrolled, read through
  the author's head: 434/434 argmax (the 12 near-ties included), max |Δp| 6.6e-6, lowest per-position
  hidden cosine 0.9999999965 on 21 rows; the module's loader reads all 426 keys and leaves none; it equals
  the overlay's plain text decoder bit for bit (3 rows, one of 1,518 tokens)
  ([`gate-kev-4b-torch-parity.json`](gate-kev-4b-torch-parity.json)).
- The Metal kernel has no torch values: its `torch_defn` returns zeros of the right shapes. The GPU gate
  below is its parity check.

### The graph alone on the Mac GPU

| set | questions | argmax (margin > 0.02) | near-ties agreeing | max \|Δp\| | mean of row means | bar |
|---|---:|---:|---:|---:|---:|---|
| **fixture** | 434 | **422/422** | **10/12** | **0.0153** | **0.00077** | **PASS** |
| held out | 130 | 126/126 | 3/4 | 0.0096 | 0.00069 | PASS |

The near-ties that flip have oracle margins of 0.0005–0.0024 and move by at most 0.0029. The worst
fixture row is a SemIf item (0.0153), the worst held-out row a `score` item (0.0096). The lowest
per-position hidden cosine on the 21 recorded rows is 0.9921. Every value is finite and every process's
re-run is bit-equal. The red arms move p: a swapped state moves 5 of 5 argmaxes (max |Δp| 0.988), a
grammatical "not" 3 of 5 (0.957). Transcript: [`gate-kev-4b-readout.json`](gate-kev-4b-readout.json).

### From the request: Python and Swift

- `conversion/kev/host.py`, without the author's package, rebuilds every oracle row from the raw request
  (ids, `<|fim_suffix|>` / `<|box_end|>` indices, keys): 434/434 and 130/130 rows with two tokenizer
  implementations. `decide.py` on the AOT graph: hidden rows bit-equal to the gate's on 434/434 and
  130/130 rows, and the shared prefix equal to the direct run on every row
  ([`gate-kev-4b-host.json`](gate-kev-4b-host.json)).
- The Swift host ([`apps/Kev`](../../apps/Kev/), Release): the ids from the raw request equal the
  oracle's on 434/434 and 130/130 rows. On the same AOT asset every row's hidden state equals the Python
  gate's bit for bit (434/434 and 130/130), and its float64 head gives p within 4.5e-7 of the gate's fp32
  head. The bar is the gate's (fixture 0.0153 / 0.00077, held out 0.0096 / 0.00069). With the shared
  prefix every row's hidden state and p equal the direct run's (434/434 and 130/130).
- **JIT = AOT.** The `.aimodel` specialized by the Swift runtime (GPU preferred,
  `expectFrequentReshapes`) gives the AOT asset's hidden state and p bit for bit on 25 rows of 10 records.
  The specialization took 16.6 s in `AIModel(contentsOf:)` (18.4 s with the tokenizer and the head)
  and left a 15.55 GB entry in the runtime's cache; a later load took 1.74 s
  ([`gate-kev-4b-swift.json`](gate-kev-4b-swift.json)).

### Time per decision on the Mac

Swift, Release CLI, the AOT asset, in one machine-wide GPU lock window (2026-10-04 14:05–14:30 JST). Two
processes per graph counted only when no other GPU job ran and the one-minute load average was at most 12;
each made 6 decisions per item after one warm-up, and the table gives the median (p10–p90) over the 12.
The earlier unrolled graph (16 tokens per call) ran in the same window:

| request | tokens | calls | ms | p10–p90 | unrolled S = 16, ms |
|---|---:|---:|---:|---|---:|
| one question | 94 | 1 | 100.2 | 99.3–101.2 | 269.5 |
| one question | 380 | 3 | 298.0 | 296.1–304.5 | 1,075.4 |
| one question | 1,518 | 12 | 1,189.0 | 1,182.9–1,194.9 | 4,341.9 |
| one question | 1,802 | 15 | 1,488.1 | 1,484.3–1,498.5 | 5,200.2 |
| five questions on one 137-token state, each row from zero | 240 | 10 | 986.7 | 983.0–990.2 | 2,367.9 |
| the same five questions, the state run once (shared) | 240 | 6 | 615.6 | 613.3–624.5 | 894.7 |
| eight questions on that state, each row from zero | 320 | 16 | 1,578.7 | 1,574.2–1,591.6 | 3,776.0 |
| the same eight questions, shared | 320 | 9 | 920.0 | 916.9–930.6 | 1,262.4 |
| four questions on one 1,477-token state, each row from zero | 1,622 | 49 | 4,851.7 | 4,839.2–4,870.6 | 17,087.7 |
| the same four questions, shared | 1,622 | 16 | 1,614.5 | 1,607.5–1,647.8 | 4,713.7 |

Loading the AOT asset took 9.90 / 9.91 s at the start of each process and 1.30 / 1.22 s when loaded
again. The two processes ended at 0.89 and 0.95 GB of footprint. Transcript:
[`gate-kev-4b-timing-mac.json`](gate-kev-4b-timing-mac.json).

### iPhone 18 Pro: not in this release

An earlier graph of this port (the recurrence unrolled, 16 tokens per call) compiled for the iPhone 18
Pro's GPU (h19p, `--expect-frequent-reshapes`) is 15,557,759,365 bytes. Loaded once in the headless gate
app (iOS 27.0 24A437, `.default` options, no increased-memory-limit entitlement), the app died inside
`AIModel(contentsOf:)` with `EXC_BAD_ACCESS (SIGSEGV)` in the on-device compile for delegates
(`-[MPSGraphAICodeCompilerDelegate getInitializedAICodeBytecodeWithPayloadPrefix:delegateId:]`). Its last
memory sample, 0.32 s in, read a footprint of 175.9 MB with 3,364.1 MB available. The cause is not
isolated and the load was not retried; the graph of this release was not loaded on the phone
([`../kev-0.8b/gate-kev-0.8b-iphone.json`](../kev-0.8b/gate-kev-0.8b-iphone.json)
`round8.device.summary.load_4b`, `round8.load_4b_memory_samples`).

## Forms measured

Every form below was exported and compiled the same way and read against the same oracle: the full gate,
or the 130-row subset where the table says so. The times come from different windows, so compare forms only
within one window. Full records: [`gate-kev-4b-forms.json`](gate-kev-4b-forms.json).

Mac (Swift Release CLI, the 94-token question, median of the counted processes):

| form | 94-token decision, ms | window (2026-10-04, JST) | fixture max \|Δp\| | note |
|---|---:|---|---:|---|
| Gated DeltaNet unrolled, S = 16 | 269.5 | 14:06–14:30 | 0.0154 | |
| unrolled, S = 32 / 64 | — | — | 0.0078 / 0.0070 (130 rows) | not timed in a lock window (104.8 / 174.1 ms per call on a shared GPU) |
| Metal kernel, S = 32 | — | — | 0.0074 (130 rows) | not timed in a lock window (48.4 ms per call on a shared GPU) |
| Metal kernel, S = 64 | 118.0 | 14:06–14:30 | 0.0170 | five questions shared: 435.9 ms (S = 128: 615.6) |
| **Metal kernel, S = 128 (this release)** | **100.2** | 14:06–14:30 | **0.0153** | |
| Metal kernel, query length 2..512, host calls of at most 512 ids | 83.3 | 14:06–14:30 | 0.0132 | not shipped: memory grows (below) |
| in-graph chunk scan | — | — | — | not run at 4B; on Kev-0.8B it fails from S = 32 |
| int8, every linear (unrolled S = 16) | — | — | 0.0601 | fails the bar (Precision) |
| any form on the Neural Engine | — | — | — | not run: the recurrence's fp32 state is not an ANE element type ([`qwen3.5-static-ane.md`](../../knowledge/qwen3.5-static-ane.md)) |

**S = 64 for several short questions on one state.** In the same window, five questions on one
137-token state, shared, took 435.9 ms with S = 64 and 615.6 ms with S = 128; one 94-token question took
118.0 and 100.2 ms. This release ships S = 128.

**Why the dynamic-length graph does not ship.** It passed the gate and decided the 94-token question in
83.3 ms in the same window. Its two timing processes ended at 6.31 and 6.32 GB of footprint, against 0.89
and 0.95 GB for S = 128. On Kev-0.8B the same form keeps growing while the call length keeps changing,
until the `AIModel` is created again ([Kev-0.8B card](../kev-0.8b/README.md#forms-measured)).

## Precision

**int8 was measured and is not shipped.** It was measured on the unrolled S = 16 graph; the Metal-kernel
graphs were not measured in int8. int8 per block of 32 over every decoder linear fails the bar:

| decoder (unrolled S = 16) | fp16 kept | `main.mlirb` bytes | fixture max \|Δp\| / mean | held out max \|Δp\| / mean | bar |
|---|---:|---:|---|---|---|
| fp16 | 100 % | 8,414,007,636 | 0.0154 / 0.00078 | 0.0109 / 0.00072 | PASS |
| int8lin: every linear int8 (`symmetric_with_clipping`) | 0 % | 5,068,103,201 | 0.0601 / 0.00192 | 0.0285 / 0.00169 | FAIL |
| asymmetric int8, `in_proj_qkv` + `out_proj` + `v_proj` fp16 | 21.7 % | 5,882,832,315 | 0.0175 / 0.00135 | 0.0277 / 0.00123 | FAIL |
| asymmetric int8, `in_proj_qkv` + `out_proj` fp16 | 21.2 % | 5,863,831,482 | 0.0174 / 0.00138 | not run | — |

The two asymmetric sets are the ones an fp32 torch bisect chose under a rule written before any result
(at most 25 % of the linear weights fp16, the target worst row ≤ 0.006 met by none; the two best by mean
taken). The one with `v_proj` fp16 passes the fixture and fails the held-out set on one `score` row, the row that also
breaks int8lin and is fp16's held-out worst. A set built before the rule's last stage (layers 0–3 and
`out_proj` fp16) fails both (0.0272, 0.0293). The quantizer's `qscheme` accepts `symmetric`,
`asymmetric` and `symmetric_with_clipping`; block 16 and an int8 embedding table also compile
([`gate-kev-4b-int8.json`](gate-kev-4b-int8.json)).

**What the compiled asset holds.** `--expect-frequent-reshapes` adds an fp16 copy of every linear weight:
for the unrolled graph the h19p asset of the fp16 bundle is 15,557,759,365 bytes with it and 8,413,701,860
without, and the int8 sets above compile to 13,026,670,589 and 13,007,670,689 bytes with it. This
release's Mac h16c asset is 15,552,786,962 bytes against an 8,412,500,461-byte `main.mlirb`. Without the
flag the runtime specializes the unrolled graph again for every new position length: 9.6–14.6 s per new
length on the Mac GPU (calls at a length already seen: 46.4 ms median, 33 rows); an int8 asset without the
flag had not finished specializing its opening length after 12 minutes. A Mac gate worker on the efr fp16
asset of the unrolled graph showed up to 16.6 GB rss, up to 16.4 GB of it clean mapped file, and a
phys_footprint of at most 1.15 GB.

## ⬇️ Bundle

[mlboydaisuke/Kev-4B-CoreAI](https://huggingface.co/mlboydaisuke/Kev-4B-CoreAI), one folder under
`gpu-pipelined/`:

| file | what | bytes | sha256 |
|---|---|---:|---|
| `kev_4b_decode_fp16_metal_pf128.aimodel/main.mlirb` | the decoder, fp16 | 8,412,500,461 | `da529dd4…413eb17f` |
| `metadata.json` | `kind: decision-backbone`, the readout contract, the gate | 12,643 | `e641e44f…c9039b08` |
| `tokenizer/tokenizer.json` | the base model's, verbatim | 12,807,196 | `fe000e3e…d50d2927` |
| `tokenizer/tokenizer_config.json` | the base model's, verbatim | 16,713 | `3891e840…b6b9f89c` |
| `head/head.safetensors` | the pointer head: q / k weight `[256, 2560]` and bias, fp32 | 5,245,304 | `36c392f7…54157106` |
| `head/kev_head.json` | head size, scale, temperature, delimiter ids, provenance | 2,038 | `5ff071ff…cfba7416` |

The repository root carries the base model's `config.json`, the adapter's `adapter_config.json` and the
Apache-2.0 `LICENSE`, verbatim. `SHA256SUMS` lists every file. Its Hub revision is
[`ee978c7`](https://huggingface.co/mlboydaisuke/Kev-4B-CoreAI/tree/ee978c72ad20f98d8548b5ef5a28e7e7e1102389) (2026-10-04).

No AOT asset ships: the Swift runtime's specialization of the `.aimodel` equals the AOT asset bit for bit
(above). To compile one for the Mac anyway (15.55 GB, the flag is required):

```bash
xcrun coreai-build compile kev_4b_decode_fp16_metal_pf128.aimodel --output aot --preferred-compute gpu \
    --platform macOS --architecture h16c --expect-frequent-reshapes
```

## Use it

Swift, with the [`Kev`](../../apps/Kev/) package (macOS 27; the system CoreAI framework, Accelerate and
swift-transformers' tokenizer), on a download of the repository:

```swift
import Kev

let bundle = URL(filePath: "Kev-4B-CoreAI/gpu-pipelined/kev_4b_decode_fp16_metal_pf128")
let kev = try await KevDecider(bundle: bundle)              // the .aimodel, specialized here (GPU, frequent reshapes)
let response = try await kev.decide(requestJSON: requestData, shared: true)   // shared: the state's whole calls run once

// questions that arrive later, on the same state
let prepared = try await kev.prepare(state: try JSONParser.parse(stateData))       // the state's whole calls, once
let later = try await kev.decide(prepared: prepared, questionsJSON: questionsData)  // only the questions' rows run
```

Loading the bundle specializes the graph once (16.6 s on the M4 Max) and caches it; later loads read the cache.
The Mac CLI: `kev run --bundle Kev-4B-CoreAI/gpu-pipelined/kev_4b_decode_fp16_metal_pf128 --asset jit
--request req.json --shared --out resp.json` (`swift build -c release --package-path apps/Kev`). The
request shape and the Python read-out (`conversion/kev/decide.py run --model kev-4b …`, AOT assets only)
are as in the [Kev-0.8B card](../kev-0.8b/README.md#use-it).

## Reproduce

Environment: the zoo overlay venv (coreai-core 1.0.0b2, coreai-torch 0.4.1, torch 2.9.0, transformers
4.57.6), Xcode 27.0 RC; the author's code (the oracle and the merge) in its own venv from the author's
`uv.lock` at tag `kev-1.0`. Order and flags: [`conversion/kev/README.md`](../../conversion/kev/README.md).

```bash
(cd $ZOO_WORK_ROOT/_kev/kev-src && python scripts/merge_lora_checkpoint.py \
    --lora jaredpalmer/kev-4b@591dcb5bd6d05eb0b5131ea6608f93f10243335c --out $ZOO_WORK_ROOT/_kev/merged/kev-4b-v1.0)
# the decoder (--aot adds the h16c .aimodelc the Python gates load)
python conversion/kev/export_decoder.py fp16 --model kev-4b --gdn-scan metal --prefill-chunk 128 --aot
# the published bundle: the gated .aimodel bytes and their metadata, without exporting again
python conversion/kev/export_decoder.py fp16 --model kev-4b --gdn-scan metal --prefill-chunk 128 --metadata-only \
    --from-aimodel <bundles>/kev_4b_decode_fp16_metal_pf128/kev_4b_decode_fp16_metal_pf128.aimodel \
    --bundle-dir <ship>/kev_4b_decode_fp16_metal_pf128
```

Recipe: [`recipe.toml`](recipe.toml). Port notes: [`knowledge/kev-port.md`](../../knowledge/kev-port.md).

## Other formats

On the Hub (2026-10-04), not run here:

- ONNX: [`midudev/kev-4b-ONNX`](https://huggingface.co/midudev/kev-4b-ONNX) (ONNX Runtime Web, int4
  weights, the pointer head as `head.bin`);
  [`onnx-community/kev-4b-ONNX`](https://huggingface.co/onnx-community/kev-4b-ONNX) (Transformers.js,
  `q4` / `q4f16`, the pointer head in the graph; its card names Qwen/Qwen3-4B-Base as the base, the
  Qwen3 generation of Kev-4B before Kev 1.0).
- GGUF: [`ggml-org/Kev-4B-GGUF`](https://huggingface.co/ggml-org/Kev-4B-GGUF) ("a decision model, to be
  used via `/v1/systemone` API", per its card).
- MLX: [`RoderickQiu/kev-4b-mlx-8bit`](https://huggingface.co/RoderickQiu/kev-4b-mlx-8bit) (merged, 8-bit,
  `head.pt` fp32) and [`aselea/Kev-4B-MLX-Serve-8bit`](https://huggingface.co/aselea/Kev-4B-MLX-Serve-8bit)
  (for mlx-serve, the head as `kev_head.safetensors`); the author's repository also serves Kev through
  MLX on Apple silicon.

No other Core AI conversion of Kev was listed on the Hub on 2026-10-04.

## License

Apache-2.0: the adapter and the head (jaredpalmer/kev-4b) and the base (Qwen/Qwen3.5-4B-Base); the bundle
inherits it, and the Hugging Face repository carries the base's `LICENSE`. The author's package runs only
in the oracle and the merge, at gate time. The fixture file's terms are Kev-0.8B's: SemIf authored144
under MIT with its notice, transfer-v4 records by reference, 11 of the 20 records written for the port
published and 9 withheld (an invented name in each was found in use on the web).

## Limits

From the [author's card](https://huggingface.co/jaredpalmer/kev-4b): English only; text generation, chat
and fully automated consequential decisions about people are out of scope; questions that need facts
that are neither in the state nor general knowledge are out of scope; the temperature was fitted on the
author's development rows (the card explains how to measure and refit it). This port adds:

- A row holds at most 3,968 tokens, while the author's server accepts states up to 65,536.
- The release is for the Mac.
- A process's opening call is slower. On the Mac the opening 128-token call of each Python gate process
  took 1,121–9,881 ms, against a median of 99.0 ms for its later calls.
- If you export this decoder with a dynamic query length yourself, keep the call lengths to one or two
  values. A process that cycles more lengths grows its memory until the `AIModel` is created again
  (measured on Kev-0.8B on the Mac and the iPhone; which runtime layer keeps the memory is not isolated).
