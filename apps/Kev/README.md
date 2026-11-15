# Kev: the Swift host of Kev-0.8B and Kev-4B

A Swift package that answers a SystemOne request — a state and typed questions (`noul`, `choice`, `score`) — with the
probabilities of [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) and
[jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b) on Core AI, on the system CoreAI framework, Accelerate
and swift-transformers' tokenizer only. The library `Kev` builds for macOS 27 and iOS 27; the executable `kev` is the
Mac CLI. `conversion/kev/host.py` is the specification every file here copies, and `conversion/kev/gate_swift.py`
checks the copy against the author's fp32 oracle and against the Python reference on the same assets. It reads both
kinds of bundle the port has made: a static-S graph (every call S tokens, `language.prefill_chunk`) and the dynamic-S
graph of round 14 (any call length up to the graph's cap, `language.query_len_range`), whose call lengths the bundle's
metadata fixes for the host (see "How a row becomes calls" below). The published Kev-0.8B bundle is static:
`kev_0_8b_decode_fp16_metal_pf128`, 128 tokens per call, the Gated DeltaNet recurrence in an fp32 Metal kernel. The
dynamic-S graph was measured and does not ship: a process that keeps changing its call length keeps growing its memory
([`knowledge/kev-port.md`](../../knowledge/kev-port.md)).

```swift
import Kev

let kev = try await KevDecider(bundle: bundleDir, asset: aimodelcURL)   // asset: nil = the bundle's .aimodel (JIT)
_ = try await kev.warmUp()                                              // a dynamic-S bundle only: every call length once
let response = try await kev.decide(requestJSON: requestData)           // shared: true = one prefill of the state
print(PythonFormat.dumps(response, asciiOnly: false))
// the fixture's tv4_000 on Kev-0.8B's earlier upload (kev_0_8b_decode_fp16_pf16, static S = 16; AOT, Mac):
// {"model": "kev_0_8b_decode_fp16_pf16", "answers": {"answer": {"type": "choice", "choice": "c", "confidence": 0.2474,
//  "probabilities": {"a": 0.1476, "b": 0.2051, "c": 0.4355, "d": 0.2117}}}, "usage": {"input_tokens": 94,
//  "output_tokens": 74}, "latency_ms": ...}
```

`trace(request:shared:)` returns the rows (ids, `<decide>` / `</opt>` indices, keys), the hidden rows the head read,
every option's logit and float32 probability, the answers as `json.dumps` writes them, the response and every step's
time. The response's `model` is the bundle's name (the author's server echoes the request's; set `modelName` to
change it).

## What each part copies

| file | the contract it reproduces (`conversion/kev/host.py`) |
|---|---|
| `JSONValue.swift` | the request as written: members in order, number literals kept (`1250` ≠ `1250.0` ≠ `1.25e3` in the text the model reads); `json.loads`' duplicate-key and NaN / Infinity rules (from apps/ClefFlash) |
| `PythonFormat.swift` | Python's `str()` of a value (`True`, an int's digits, a float's repr, `inf`), `round(x, n)`, `json.dumps` with its defaults (`", "` / `": "`, ensure_ascii), `str.lstrip`'s whitespace |
| `Request.swift` | `validate_request` (the author's pydantic decisions), `render`, `option_text`, the option order and the keys, `to_record` |
| `Encoder.swift` | `user_tokens` (`<\|name\|>` → `<¦name¦>` on code points), the packed encoding and the serving limits (`admit`'s messages), the rows; `KevGraphShape` (`graph_shape`, `plan`, `call_lengths`, `padded_end`, `shared_prefix_plan`): the call plan, the graph limit (4,080 tokens at q = 16), the shared prefix; the delimiter ids looked up by text and checked against `metadata.json` |
| `Decoder.swift` | `main` (static S, or the dynamic query length): four states allocated once at `max_context_length` and zeroed per row, the plan's calls (the last padded with `<\|endoftext\|>`), one hidden buffer per call length, hidden `[T, d]` fp16; the shared prefix (the state's first ⌊Ls / q⌋·q tokens once, the four states copied per question); `warmUp` (every call length once) |
| `Head.swift` | the pointer head in float64 (`q = Wq h + bq`, `k = Wk h + bk`, `z = k · q × 1/16`), the softmax in float64 with NumPy's pairwise sum, p rounded to Float once; `head.safetensors` read here |
| `SystemOne.swift` | `to_answers` (`choice_confidence`, `score_confidence`, Python 3.12's compensated `sum`), the response body and its usage |
| `KevDecider.swift` | the glue, in decide.py's order; the bundle's metadata read and checked at load |

## Contract checks at load

An asset that differs fails at load, not in a probability:

- `metadata.json`: `language.prefill_chunk` S, or `language.query_len_range` [qmin, cap] with `query_len_call_max` L
  (absent = cap) and `query_len_multiple` q (absent = 1) — qmin ≤ L ≤ cap, L a multiple of q, q = 1 or ≥ qmin —;
  `max_context_length`, `vocab_size`, `decision.row` (the five delimiter ids and the pad id), `decision.head` (the files
  and the scale);
- the tokenizer: each delimiter's id by token text equals `metadata.json`'s, and the token text encodes to that id alone;
- the head: `kev_head.json` (scale = 1 / √head_dim = `metadata.json`'s scale, the temperature, the hidden size) and
  `head.safetensors` (q / k weight `[head_dim, d]` and bias `[head_dim]`, F32);
- the decoder's `main`: inputs exactly `input_ids [1, S]` int32 (a dynamic-S bundle: `[1, -1]`) and `position_ids
  [1, -1]` int32, the output `hidden [1, S, d]` float16 (dynamic: `[1, -1, d]`; d from `kev_head.json`), the four states
  float16 (KV rank 5 with one dynamic axis, conv rank 4, recurrent rank 5).

## How a row becomes calls

`KevGraphShape` (host.py `plan`) cuts a row of T ids into pieces of L ids (the longest call the host makes), the
remainder last, and pads the last piece with `<|endoftext|>` up to the next multiple of q; the padded positions' hidden
rows are dropped (causal: they cannot reach a real position). A call of c ids after p earlier ids gets `position_ids`
0 ..< p + c. A static-S bundle is the case L = q = S: ⌈T / S⌉ calls of S. With the shared prefix, the state's first
k = ⌊Ls / q⌋·q tokens run once (no pad: k and L are multiples of q) and every question's rest runs from a copy of the
states, positions continuing at k. On a static-S bundle these are the direct run's calls, so the hidden rows are equal
bit for bit; on a dynamic-S bundle the calls are cut at other places, so the shared rows differ from the direct ones in
their last bits (p within about 1e-3; both are gated against the author's oracle).

A prepared state answers questions that arrive later on the same state without running the state again:

```swift
let prepared = try await kev.prepare(state: stateJSON)                         // the state's first k tokens, once
let r1 = try await kev.decide(prepared: prepared, questionsJSON: questionsData)  // only the questions' branches run
```

It is the shared prefix split in two (the same calls), so its hidden rows and p equal a `shared: true` request's bit for
bit, one question at a time or several. The value holds one copy of the four states (about 60 MB for Kev-0.8B and
160 MB for Kev-4B at the 4,096 context); the caller keeps it and drops it, nothing is cached behind the call.

On a dynamic-S bundle (the port measured `kev_0_8b_decode_fp16_metal_dyn512`, L = 512 and q = 16; it does not ship) a
process makes calls of at most 32 lengths (16, 32, …, 512). Such a process grows its memory with each change of call
length once it has used more than a few lengths, and only a new `AIModel` gives the memory back; a static-S bundle
makes one call length and stays flat. `warmUp()` runs each once from zero states and returns each one's ms. On the Mac only the
first call of a process was slow, 150-170 ms (1.3 s in the process that made the AOT asset's runtime cache entry); every
other length's first call cost what its later calls cost. `warmUp()` takes that first call out of the first request.
`KevDecider(…, callMax:, multiple:)` overrides L and q within the graph's range (nil = the bundle's metadata). The graph
limit is the padded end: ⌈T / q⌉·q ≤ max_context_length − 1, so T ≤ 4,080 at q = 16.

## Build and run (Mac)

```sh
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer   # CoreAI.framework is in the Xcode 27 SDK
# $ZOO_WORK_ROOT: the work root, by default the parent directory of this repository (conversion/_paths.py)
swift build -c release --package-path apps/Kev --scratch-path $ZOO_WORK_ROOT/_kev/swift/.build
BIN=$ZOO_WORK_ROOT/_kev/swift/.build/release/kev
B=$ZOO_WORK_ROOT/_kev/exports/ship/kev_0_8b_decode_fp16_metal_pf128     # the published Kev-0.8B bundle
$BIN run --bundle $B --request req.json [--shared] --out resp.json [--trace trace.json] [--asset aot|jit|<path>]
$BIN fixture --bundle $B --records records.json [--shared] --out out.json [--ids a,b] [--dump-hidden <dir>]
$BIN rows --bundle $B --records records.json --out rows.json [--oracle records_oracle.json]   # the tokenizer alone
$BIN render-test --in values.json --out out.json [--bundle $B]                                 # the text alone
$BIN time --bundle $B --records records.json --model-label kev-0.8b --slot 0 --out p0.json      # one timing process
$BIN warm --bundle $B [--asset aot|jit] --out w.json [--passes 2]     # every call length's first and later calls
# run / fixture / time: --warm (warmUp after the load); run / fixture / time / warm / rows: --call-max L --multiple q
# fixture --shared --prepared: each record's state also prepared once and checked against the shared prefix
# time --prepared: adds a prepared state's two costs (the prepare, then one question)
# iOS: the library alone
xcodebuild -scheme Kev -destination 'generic/platform=iOS' -derivedDataPath $ZOO_WORK_ROOT/_kev/swift/dd-ios build
```

`--asset aot` (default) loads `<bundle>/../../bundles_aotc/<name>.h16c.aimodelc` with `SpecializationOptions.default`,
as the Python gates do; `--asset jit` specializes the bundle's `.aimodel` here, GPU preferred with
`expectFrequentReshapes` (the exporter's AOT flags). The library takes either URL. `_time_mac.sh` times a fixed set of
decisions under the machine-wide GPU lock (round 5's protocol, `conversion/kev/timing.py`).

## Measured (M4 Max, macOS 27.0 26A428, Xcode 27.0 RC, Release, AOT)

Round 7, on the earlier upload's graphs (`kev_0_8b_decode_fp16_pf16` and Kev-4B's, static S = 16):

- Against the author's fp32 oracle, every record from its raw request (fixture 434 questions + held-out 130): the ids,
  rows, `<decide>` / `</opt>` indices and keys of all 564 rows of each checkpoint; the readout gate's bar —
  Kev-0.8B argmax 420/420 + 125/125 outside near-ties, max |dp| 0.0117 / 0.0057, mean 0.00097 / 0.00081; Kev-4B
  422/422 + 126/126, 0.0154 / 0.0109, 0.00078 / 0.00072.
- Against the Python reference of the same assets (`conversion/kev/decide.py`): every row's hidden rows bit for bit,
  every p bit for bit, every answer set's `json.dumps` byte for byte, and the shared prefix equal to the direct run on
  every row. The `.aimodel` specialized here (Kev-0.8B) equals the AOT asset bit for bit on 10 records.
- A decision, in one `_GPU_LOCK` window (`_time_mac.sh`; `latency_ms` = state resets + graph calls + head, median of
  20): Kev-0.8B 112 ms for one question of 94 tokens, 1.60 s at 1,518 tokens, 1.92 s at 1,802; five questions on a
  137-token state 342 ms with the shared prefix (880 ms direct). Kev-4B 264 ms, 4.21 s, 5.01 s; 863 ms shared (2.25 s
  direct). The graph runs 17-19 ms per 16-token call on Kev-0.8B and 44 ms on Kev-4B; the Python reference measured in
  the same window is within 8% of these. Load (decoder, the runtime's cache warm): 0.2-0.6 s and 1.2-9.5 s.
- Transcripts: `$ZOO_WORK_ROOT/_kev/results/swift_gate_kev-{0.8b,4b}.json`, `swift_render_test.json`,
  `swift_jit_kev-0.8b.json`, `longrow_kev-{0.8b,4b}.json`, `swift_timing_r7.json`.

Kev-0.8B's published bundle `kev_0_8b_decode_fp16_metal_pf128` (the GDN scan as an fp32 Metal kernel, a static 128
tokens per call; round 11's gates, round 15's timing window):

- Against the author's fp32 oracle, from the raw requests: argmax 420/420 + 125/125 outside near-ties (near-ties 13/14 +
  5/5), max |dp| 0.0124 / 0.0058, mean 0.00096 / 0.00083; the shared prefix equal to the direct run on all 434 rows.
- Against the Python reference of the same AOT asset: every row's ids, hidden sha256, p bits and answer bytes (434 rows).
  The `.aimodel` specialized here equals the AOT asset bit for bit on 25 rows; the first load specialized it in 3.34 s.
- A decision (`_time_mac.sh`, two clean processes, median of 20): 29.9 ms for one question of 94 tokens, 357.0 ms at
  1,518 tokens, 186.4 ms for five questions on a 137-token state shared; a 137-token state prepared in 34.1–34.7 ms,
  then 30.3 ms per question. Transcripts: `r11_swift_gate_K128.json`, `r11_swift_jit_K128.json`, `r15_timing_0_8b.json`.

Kev-0.8B's dynamic-S bundle `kev_0_8b_decode_fp16_metal_dyn512`, measured and not shipped (the GDN scan as an fp32
Metal kernel, call lengths 2..512 in the graph, L = 512, q = 16; the same machine, round 15):

- Against the author's fp32 oracle, from the raw requests: argmax 420/420 + 125/125 outside near-ties (near-ties 12/14
  + 5/5), max |dp| 0.0112 / 0.0062, mean 0.00096 / 0.00087; with the shared prefix 420/420, 0.0128, 0.00095. The shared
  p differ from the direct p by 0.0022 at most (0.00085 on the 70 questions of the port's own records). Four rows near
  the graph limit: max |dp| 0.0017.
- Against the Python reference of the same assets: all 564 rows' ids, every hidden row's sha256, every p's bits and
  every answer set's bytes, direct and shared, each on the calls `host.plan` makes. A prepared state's hidden rows and p
  equal the shared prefix's bit for bit (434 rows, all questions together and one at a time). The `.aimodel` specialized
  here equals the AOT asset bit for bit on 25 rows. On `kev_0_8b_decode_fp16_pf16` the host gives round 7's numbers.
- A decision in one `_GPU_LOCK` window (two clean processes per form: no other GPU job, load average ≤ 12; median of 20;
  the Python reference on the same bundle: 29.5 ms and 136.7 ms):

| decision | dyn512, AOT | dyn512, JIT | static S = 128, Metal scan (round 11), AOT | `pf16` (static S = 16), AOT |
|---|---:|---:|---:|---:|
| one question, 94 tokens | 25.4 ms | 25.2 ms | 29.9 ms | 100.1 ms |
| one question, 1,518 tokens | 297 ms | 297 ms | 357 ms | 1,600 ms |
| five questions on a 137-token state, shared / direct | 111 / 189 ms | 112 / 189 ms | 186 / 297 ms | 328 / 846 ms |
| a 137-token state prepared: the prepare, then one question | 34.3 + 14.2 ms | 34.7 + 14.2 ms | 34.4 + 30.3 ms | 137.8 + 34.1 ms |
| a 1,477-token state prepared: the prepare, then one question | 295 + 18.7 ms | 296 + 18.8 ms | 338 + 31.5 ms | 1,543 + 51.5 ms |

- Load, the runtime's cache warm: 0.2 s. The first load makes a 2.50 GB cache entry: the `.aimodel` specialized here
  3.2-3.3 s, the AOT asset 2.1 s.
- Transcripts: `$ZOO_WORK_ROOT/_kev/results/r15_gate_0_8b.json` (with `r15_swift_gate_D512L512q16.json` and
  `r15_swift_jit_D512L512q16.json`), `r15_timing_0_8b.json`.

## Notes

- `InferenceFunction.MutableViews` borrows what it holds until `run` returns: `KevDecoder` keeps the four states and
  the hidden buffer in a class property and moves them into locals for a pass (the ClefFlash / DeciderVision form).
- The states are allocated once at `max_context_length` (4,096) and zeroed per row: about 60 MB of fp16 for Kev-0.8B
  and 160 MB for Kev-4B; the shared prefix keeps one more copy of them.
- A row longer than the graph limit is refused before any call (`KevError.graphLimit`: the padded end ⌈T / q⌉·q must stay
  at or below 4,095, so 3,968 tokens on the published S = 128 bundles and 4,080 at q = 16); the author's serving limits (a
  65,536-token state, a 73,728-token row) are checked too (`KevError.contextOverflow`).
- The CLI lives in `Sources/kev-cli`: on a case-insensitive volume `Sources/kev` would be the library's `Sources/Kev`.
