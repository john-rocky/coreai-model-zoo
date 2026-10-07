# D1: the Swift host of d1-3B

A Swift package that answers a System One request — a state and named questions (`noul`, `choice`, `score`) — the
way [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B)'s own code does, on the system frameworks (CoreAI,
Accelerate) and swift-transformers' tokenizer. The library `D1` builds for macOS 27 and iOS 27; the executable `d1` is
the Mac CLI. `conversion/d1/host.py` (the text, the ids, the readout, the response) and `conversion/d1/vision_host.py`
(a picture's crop plan and token run) are the specification every file here copies; `conversion/d1/gate_swift.py`
checks the copy against them and against HF `tokenizers` on the checkpoint's `tokenizer.json`, exactly: the same text,
the same ids, the same readout bits, the same response bytes.

The text side and the graph side are here. `D1Decider.loadGraph` loads the decoder bundle's graph (and a tower bundle
for pictures) and checks both against their `metadata.json`; `decide` then runs request → rows → graph → readout →
response, the order of `conversion/d1/decide.py`, the Python reference it copies. A decider loaded without a graph
builds the rows and stops with `D1Error.graphNotWired`; `D1Decider.response(requestJSON:slotHidden:)` runs the readout
and the answers from slot hidden rows the caller supplies. A picture's pixels (decode, the processor's resize, patches,
the position table) are not computed here yet: `D1Vision` plans the crops and the image-token run from the picture's
size, and the four tower inputs of each crop are read from files (`D1TowerInputs`, what `vision_host.tower_inputs`
writes).

```swift
import D1

let d1 = try await D1Decider(bundle: bundleDir)              // metadata.json, tokenizer/, head/option_rows checked
try await d1.loadGraph(asset: .aot)                           // the decoder (.aimodelc; .jit = the bundle's .aimodel)
let body = try await d1.decide(requestJSON: requestData)      // the System One response
print(PythonFormat.dumps(body, indent: 2, asciiOnly: false))

let rows = try d1.rows(requestJSON: requestData)              // the text side alone: one row per question
let kept = try await d1.prepare(state: state)                 // a state run once ...
let later = try await d1.decide(prepared: kept, questionsJSON: questionsData)   // ... and questions on it later
```

## What each part copies

| file | the contract it reproduces |
|---|---|
| `JSONValue.swift` | the request as written: members in order, number literals kept (`1250` and `1250.0` render differently), `json.loads`' duplicate-key, NaN / Infinity and 4,300-digit int rules (from apps/Kev, itself from apps/ClefFlash) |
| `PythonFormat.swift` | `json.dumps(state, ensure_ascii=False, indent=2)` (the state block) and `json.dumps` with its defaults, `repr` / `str` / `round(x, n)` of a float, `str.strip`, `str.isalpha`, `repr(str)` (the option-table refusal's text) |
| `Request.swift` | host.py §1–2: `validate_question` / `validate_request` (the same accept / refuse decisions and texts), `state_block`, `prefix_text`, `question_block`, `suffix_text`, `option_codes`, the keys |
| `Encoder.swift` | host.py §3–4: the ids of `tokenizers`' `encode(text, add_special_tokens=False)`, `aliases` with the fallback pool, `readout_groups`, `build_question` (row, slot), `build_request` (one question = its row; several = the Tree, trunk and branches encoded apart), `shared_prefix`, `option_table_check`, `graph_context_check` |
| `Vision.swift` | vision_host.py §1, 2, 5, 6: `cap_size`, `smart_size`, `is_too_large`, the 26 target ratios and the tie-break, `plan`, `image_tokens`, `prompt_ids` (the text cut at each `<image>`), `extension_ids` (the k-th `<image>` → 128,000 + k) |
| `Readout.swift` | host.py §5–6: z = h · E[id] in float64, the group max, the softmax with Python 3.12's `sum()`, `answer`, `response`; the option table |
| `D1Decider.swift` | the glue in host.py's order; the bundle's `metadata.json` and `head/option_rows.{json,safetensors}` read and checked at load; the graph's side of decide.py (`D1.build` / `decide` / `prepare` / `decide_prepared`): the rows in the graph's ids, the readout groups, the state's stable tokens Ls, direct / shared / prepared, the trace |
| `Decoder.swift` | the decoder graph on the low-level runtime (apps/Kev's `KevDecoder`, itself apps/ClefFlash's): `AIModel` + `loadFunction("main")`, the descriptor checked against `metadata.json` `language.contract`, the three states allocated once at `max_context_length` and zeroed per row, the image rows written once per request and bound to every call, S-id calls with `position_ids` 0 ..< p + S, the pad, the shared prefix and the prepared state |
| `Tower.swift` | the vision tower bundle (apps/ClefFlash's `VisionTower`): its contract from the tower's `metadata.json` `graph`, one call per crop, the float inputs cast to an fp16 tower's type, each crop's first h w / 4 rows cast to float16 into the image rows; `D1TowerInputs` reads a crop's four inputs from raw files |
| `D1BLAS` (C) | the readout's products through the BLAS calls NumPy makes for `E @ h` (Accelerate's new interface, ILP64; operands on page-aligned copies) |

## The tokenizer: swift-transformers plus three steps

The ids must be `tokenizers`' for this `tokenizer.json`. swift-transformers loads the file and runs the BPE merges;
two things it does differently are done here instead (`Encoder.swift`):

- **The Split pre-tokenizer on code points.** swift-transformers matches the Split regex with
  `String.range(of:options:.regularExpression)`, which snaps every match to grapheme clusters (`".\r\n"` becomes `"."`
  and `"\r\n"`; a combining mark stays on its base). The host cuts each text with `NSRegularExpression` on UTF-16
  offsets, as ICU and `tokenizers`' Oniguruma match code points. Open upstream: huggingface/swift-transformers PR #398.
- **`ignore_merges`.** The checkpoint's BPE model sets `ignore_merges: true`: a pre-token that is a vocabulary entry is
  that entry, before any merge. swift-transformers always runs the merges, and some entries are not rebuilt by them
  (`"\tpubli"` is one entry, three pieces through the merges). The host looks each piece up first. Open upstream:
  huggingface/swift-transformers PR #397.
- **The added tokens first.** The 124 added tokens are cut out of the text before the regex, leftmost and longest
  first, on the raw text (none strips or normalizes): a state holding `<|im_end|>` reads as that token, as in the
  provider's code.

Each piece goes to swift-transformers through a second tokenizer built from the same `tokenizer.json` with the
pre-tokenizer set to ByteLevel alone, so swift-transformers sees one BPE word at a time. Resolved here: swift-transformers
1.3.4 (`c21fdcde3903`, `from: "1.3.3"`). When PRs #397 and #398 are in a release, swift-transformers' own ids
(`D1Tokenizer.plainTokens`, `d1 rows --plain`) should equal the host's on every row; `gate_swift.py ids` reports that
count, and the three steps can then go.

## Contract checks at load

- `tokenizer.json`: no normalizer; the pre-tokenizer is Sequence[Split(the LFM2 regex, Isolated, not inverted),
  ByteLevel(no prefix space, no regex)]; a BPE model without byte fallback or dropout; no added token with lstrip /
  rstrip / single_word / normalized. Anything else is `D1Error.contract`: the host reproduces this pipeline and no other.
- The special tokens: `<|startoftext|>` 124894, `<|im_start|>` 124899, `<|im_end|>` 124900, `<|pad|>` 124893,
  `<image>` 124907, `<|image_start|>` 125009, `<|image_end|>` 125010, `<|img_thumbnail|>` 125008 and the 100
  `<|img_row_r_col_c|>` = 124908 + 10(r − 1) + (c − 1): each in the added vocabulary with that id and encoded alone to
  `[id]`.
- A bundle: `metadata.json` `kind` = `decision-backbone`, `decision.prompt.special` equal to the tokenizer's ids,
  `vision.image_token.id` = 124907; `head/option_rows.json` and `option_rows.safetensors` hold the same ascending ids,
  as many as `decision.option_table.n`, rows `[n, hidden]` fp32.
- A request whose readout ids are not all in the option table is refused whole, before any graph call, with host.py's
  text (`questions.<name>: the token id <N> of label '<label>' is not in the option table`).

## Arithmetic

The readout is host.py's, in the same precision and order: the slot's hidden row and the option rows widened to
float64, `E @ h` through Accelerate's `cblas_dgemv` (one id: `0.0 + cblas_ddot`) — the calls NumPy makes when it is
built against Accelerate — then the group max, `exp(s − max)` with the C library's `exp`, Python 3.12's compensated
`sum()`, and the division. `cblas_ddot` sums in an order that depends on where its operands start (below a 256-byte
boundary the last bits move), so the operands are copied to page-aligned buffers first, as NumPy's arrays of these
sizes start on a page. The response's floats are not rounded and print as Python's `repr`.

## The graph

The decoder bundle (`conversion/d1/export_decoder.py`) holds one static-S function `main`: `input_ids [1, S]`,
`position_ids [1, -1]` and `image_embeds [N, d]` in, the final-norm hidden state `hidden [1, S, d]` of every position
out, and three states (`keyCache` / `valueCache` with a dynamic sequence axis, `convState`). Every name, shape and
type of that contract is read from `metadata.json` `language.contract`, and the loaded function's descriptor must equal
it; S, `max_context_length`, d, N and the pad id come from the same file. A different graph fails at load.

- **A row.** The states are allocated once, with the sequence axis at `max_context_length`, and zeroed at the start of
  every row. A row of T ids runs as ceil(T / S) calls; call c gets ids cS ..< cS + S with `position_ids` 0 ..< cS + S,
  the last call padded with `<|pad|>`, and the padded positions' rows are dropped. The readout reads the row's last
  position.
- **The image rows.** One `image_embeds` buffer per decoder: a request with pictures writes every crop's first h w / 4
  tower rows (crops in order, pictures in text order, cast to float16) at its top and zero after; a text request leaves
  it zero and nothing is rewritten. The k-th `<image>` of a row is sent as 128,000 + k and reads row k.
- **Shared and prepared.** Every row of a request starts with the same prefix (BOS, the user turn, the pictures, the state
  block, `"\nQUESTION:\n"`). Its stable tokens Ls are the prefix's ids without the ids of its last pre-token: that piece
  (`":\n"`) can merge with the question's first characters (`":\n\n"` is one token), nothing before it can. Shared runs
  the first k = floor(Ls / S) · S ids once, copies the three states and runs every row's rest from a copy, positions
  continuing at k: on a static-S graph these are the direct run's calls with the direct run's inputs, so the hidden rows
  are the same bits. `prepare(state:)` runs those k ids once and keeps the states; `decide(prepared:questionsJSON:)`
  answers questions on a copy of them later (a row that does not start with the kept ids runs whole). Pictures in a
  prepared state are not supported here.
- **AOT and JIT.** `loadGraph(asset: .aot)` loads `<bundles>_aotc/<name>.h16c.aimodelc` with
  `SpecializationOptions.default`; `.jit` lets the runtime specialize the bundle's `.aimodel` here, GPU preferred with
  `expectFrequentReshapes` (the exporter's AOT flags), and keeps the result under
  `~/Library/Caches/coreai-cache/<OS build>/<process name>/`. The tower's JIT is GPU preferred without frequent reshapes
  (its AOT flags).
- **A toy bundle.** A bundle with a `toy` block in `metadata.json` (a 256-id vocabulary with random weights, for the
  gates) gets every real id folded (id % V; 128,000 + k → V + k) and its pad from `toy.pad_folded`; its readout groups
  are given by the caller (`groups:`) or folded the same way.
- **The trace.** `trace(request:)` returns every row's ids in the graph's vocabulary, the slot, the hidden rows, the
  logits and p, the response and each call's time; `gate_swift.py` compares them with `decide.py`'s on the same asset.

## Build and run (Mac)

```bash
# $ZOO_WORK_ROOT: the lane work root, by default the parent directory of this repository (conversion/_paths.py)
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
swift build -c release --package-path apps/D1 --scratch-path $ZOO_WORK_ROOT/_d1_3b/swift/.build
D1=$ZOO_WORK_ROOT/_d1_3b/swift/.build/release/d1
SNAP=<the HF snapshot of LiquidAI/d1-3B, or a bundle's tokenizer/>

$D1 rows --records records.json --tokenizer $SNAP --option-ids option_rows.json --out rows.json [--plain]
$D1 render-test --in values.json --tokenizer $SNAP --out out.json
$D1 encode-test --texts texts.json --tokenizer $SNAP --out out.json [--plain]
$D1 image-plan --sizes sizes.json --out out.json
$D1 image-rows --records image_records.json --tokenizer $SNAP --out out.json
$D1 readout-test --in cases.json --head <bundle>/head --hidden hidden.f32 --out out.json
$D1 answers-test --in cases.json --out out.json
$D1 bundle-check --bundle <bundle> --out out.json

# the graph (the Mac GPU)
$D1 decide --bundle <bundle> [--asset aot|jit] [--tower <tower bundle> --tower-inputs <dir>] --request req.json \
    [--shared] [--groups groups.json] --out resp.json [--trace trace.json] [--reps N] [--warm]
$D1 fixture --bundle <bundle> --records records.json [--groups groups.json] --arms direct,shared [--asset aot|jit] \
    [--tower <tower bundle> --tower-inputs <dir> [--zero-image-control]] [--dump-hidden <dir>] --out pass.json
$D1 prepare-test --bundle <bundle> --records records.json [--groups groups.json] --out out.json

# the gates (the lane's venv: tokenizers, NumPy, safetensors; the graph sections after conversion/d1/decide.py's records)
cd conversion/d1 && python gate_swift.py all        # -> $ZOO_WORK_ROOT/_d1_3b/results/r3a_swift_text.json
~/code/standup/tools/quiet/quiet_wait.py -- python gate_swift.py graph   # -> results/r3c_swift_graph.json
```

`decide` answers one request (`images` in the request names pictures by file; their crops' inputs come from the
`--tower-inputs` directory's `manifest.json`) and writes the response as `json.dumps(response, indent=2,
ensure_ascii=False)`; `fixture` answers every record from its raw request, direct and shared, a refused request with
its text and each of its questions alone, and re-runs the first record at the end; `prepare-test` checks prepare +
decide(prepared) against shared. `_time_mac.sh` is the timing window's driver (apps/Kev's, not run in round 3c).

`rows` validates each question alone (as `test_host.py` does) and then the whole request with the option table;
`--plain` adds swift-transformers' own ids of every row. `readout-test` reads an option table as a bundle holds it and
slot hidden rows from a raw little-endian float32 file. The iOS target is the library alone.

## Notes

- Text is compared, cut and edited on code points, never on Swift's grapheme clusters (`"_"` → `" "` in a label,
  `str.split("<image>")`, the regex pieces), and strings are decoded with `String(validating:as:)`: Foundation's
  `String(bytes:encoding:)` drops a U+FEFF at the start of a run as a byte order mark.
- `str.isalpha` and `str.isprintable` come from `Unicode.Scalar.Properties.generalCategory`, the same definition as
  CPython's; the two can differ on code points whose category changed between their Unicode versions.
