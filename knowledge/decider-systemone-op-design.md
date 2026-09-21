# `CoreAI.systemOne(state:questions:options:)` — design note (2026-09-21)

Design only, written at the end of the decider-0.8b port (Codex gpt-6-astra under supervision; the zoo's
card is [`models/decider-0.8b/README.md`](../models/decider-0.8b/README.md)). Nothing here is
implemented or compiled. The fixture file it keeps referring to is
[`models/decider-0.8b/fixtures-decider-0.8b.json`](../models/decider-0.8b/fixtures-decider-0.8b.json)
(the 44 rows with ids, slot, label ids and fp32 oracle probabilities); other `results/…` paths name the
port's run record, which is not in this repository.


Design only, 2026-09-21. No Swift op or logits patch was implemented or compiled in this round. The recommendation is to add a completion-synchronized **read-last-logits operation to the existing pipelined engine**, then put the author's deterministic prompt and answer assembly around it. The fallback is an N-state low-level Core AI runner, compiled and validated in a later implementation run.

Source references are read-only. CoreAIKit was read with `git show main:<path>` at `f3c33b12b982fb17111d1baeedb0088d5c9e2c43`; the zoo was read the same way at `347393ede35fd25e9e59203dba562e5ee4d268bb`. Runtime line references below use this run's frozen `coreai-models-src` at fork `397b337e234474a191c0bd96ac9ef71c4f808a3d`. Author line references use the unchanged files under `oracle/decider/`, from HF revision `1ea54127d3bd52f6d753d9257b32a6380b873907`. The full fixture is **44 rows across 13 requests**: the approved wide row added a thirteenth request to the original twelve.

## 1. API and kit integration

The API consumes the same state/question shape as the author's `POST /v1/systemone` and returns typed probabilities without generating an answer token. Its default contract is `independent=true`, `state_first`, and isolated Score levels, matching this checkpoint and the validated fixtures. `CoreAI.swift:3–6` describes anchored operations as fixing their own prompt and output contract, and `CoreAI.swift:71–80` provides `OpOptions.model(_:)`. Follow those conventions with proposed catalog id `decider-0.8b`; that id and the op are not currently installed.

```swift
// Interface sketch; Codable adapters and initializers are omitted.
public enum SystemOneState: Sendable {
    case text(String)
    case json(JSONValue)
}
public indirect enum JSONValue: Sendable {
    case null, bool(Bool), number(JSONNumber), string(String)
    case array([JSONValue]), object([JSONMember]) // retain insertion order
}
public struct JSONNumber: Sendable { public let sourceLexeme: String }
public struct JSONMember: Sendable {
    public let key: String
    public let value: JSONValue
}
public struct ChoiceCriterion: Sendable {
    public let name: String
    public let description: JSONValue? // nil or JSON null means name only
}
public enum SystemOneQuestion: Sendable {
    case choice(instructions: JSONValue, criteria: [ChoiceCriterion])
    case score(instructions: JSONValue, levels: [JSONValue], isolated: Bool = true)
    case noul(instructions: JSONValue, falseCriterion: JSONValue?, trueCriterion: JSONValue?)
}
public struct SystemOneQuestionEntry: Sendable {
    public let id: String
    public let question: SystemOneQuestion
}
public struct SystemOneQuestions: Sendable {
    public let entries: [SystemOneQuestionEntry] // encodes as an ordered id -> spec object
}
public enum SystemOneAnswer: Sendable {
    case choice(ChoiceAnswer)
    case score(ScoreAnswer)
    case noul(Double) // P(yes), not Bool and not a thresholded decision
}
public struct ChoiceAnswer: Sendable {
    public let choice: String
    public let confidence: Double
    public let certainty: Double
    public let probabilities: [String: Double]
}
public struct ScoreAnswer: Sendable {
    public let score: Double // expected zero-based level, rounded to 2 places
    public let confidence: Double
    public let certainty: Double
    public let legend: [String: String]
    public let probabilities: [String: Double]
    public let levelFit: [String: Double]? // present for isolated Score
    public let fitMass: Double?
}
public struct SystemOneResponse: Sendable {
    public let model: String
    public let answers: [String: SystemOneAnswer]
    public let usage: SystemOneUsage
}
public struct SystemOneUsage: Sendable {
    public let inputTokens: Int // author's unique-prefix accounting
    public let outputTokens: Int // always 0
}
extension CoreAI {
    public static func systemOne(
        state: SystemOneState,
        questions: SystemOneQuestions,
        options: OpOptions = OpOptions()
    ) async throws -> SystemOneResponse
}
```

Wire adapters preserve the author's `type`, `instructions`, `criteria` and result key spellings, including `level_fit`, `fit_mass` and `input_tokens`. They accept Choice's ordered criteria map or a list of option names, Score's ordered levels or numerically sorted legend keys, and the `bool` alias for `noul`. Question ids are response keys only; they never enter the prompt. Validate unique ids and criterion names, 2–255 Choice options, 2–10 Score levels, and nonempty rendered instructions before loading a model. Retain JSON object order through parsing and rendering rather than routing through an unordered Swift `Dictionary`.

The v1 op fixes the validated defaults instead of exposing `independent=false` or `schema_first`; a future explicit API can add those distinct prompt contracts. Per-question `isolated=false` retains the author's list-form Score behavior. `.model("decider-0.8b")` selects a compatible System One catalog entry, not an arbitrary text model: require tokenizer, readout metadata and a logits-capable execution path. Return `model: "decider-0.8b-v1"` for this checkpoint, as `infer.py:78` derives it, while the catalog id stays an implementation detail.

Add a dedicated actor-owned scoring session to the kit's catalog residency machinery. Reuse the conventions in `CoreAI.swift:222–265`: share a model load, serialize calls on one model, pin its residency while running, and release on completion or error. This op must submit raw token ids directly; `ChatSession`'s chat template, detokenization, sampling and free-text retry behavior do not implement this contract. Drain before reset or cancellation cleanup so an unfinished GPU write cannot mutate the next row's states.

## 2. Author prompt builder and assembly port

The following maps every consequential source block in the validated state-first path to the Swift port. Match the source statements and their ordering; rendering differences change token ids.

| Author source lines | Required Swift behavior |
|---|---|
| `systemone.py:14–15` `_txt` | Return strings unchanged; serialize other JSON values with Python `json.dumps(..., ensure_ascii=False)` semantics. Preserve Unicode and object order, with default `", "` and `": "` separators. The source describes JSON as compact, but does **not** pass compact separators. |
| `systemone.py:18–30` `annotate_indices` | Recurse through objects and arrays. For an array of at least 8 elements, turn a nonobject element into `{"_index": i, "value": annotatedValue}`; prepend `_index` to object elements and then overlay their recursively annotated fields. An existing `_index` field wins, following Python's `{... , **object}` order. Short arrays recurse without inserting indices. |
| `systemone.py:33–36` `render_state` | Pass a String through unchanged. Otherwise apply array indexing and serialize with `_txt`'s JSON rules. No path evaluation, key sorting, trimming or prompt escaping. |
| `systemone.py:39–43` `render_question` | Default `type` to `choice`. Resolve instructions from `instructions`, then `question`, then `""`; resolve criteria from `criteria`, then `options`. Render instructions through `_txt`; reject an empty result. |
| `systemone.py:44–49` Choice | Turn an option-name list into an ordered name→null map. Check 2–255 entries. Preserve names in insertion order. Render each option as just its name when its criterion is null or the empty string, otherwise `name + ": " + _txt(criterion)`. |
| `systemone.py:50–55` Score | A legend map is sorted by numeric key, then treated as an ordered level list. Check 2–10 levels. Names become zero-based integers; list-form options are `"i: " + _txt(level)`. Original numeric keys do not become the score scale. |
| `systemone.py:56–63` Noul and shared fields | Resolve `false` and `true` criteria (including boolean-key compatibility in the Python-facing adapter). Options are `no`, `yes`, or `no: <description>`, `yes: <description>`. Names are false/true in that order. Normalize `bool` to `noul`; retain rendered Score legend; default per-question `isolated` to true. |
| `systemone.py:67–79` `strip_level_number` / `isolated_rows` | Strip only a leading match of `^\s*-?\d+\s*:\s*` from a level. For each level in order, render `<question>\nProposed answer: <level>\nDoes the proposed answer fit?` with options `["no", "yes"]`. A level sees neither its numerical label nor its neighbors. |
| `systemone.py:89–98` `plan_rows` | Iterate question ids in request order. When global isolation and per-question isolation are enabled for Score, append one row per level and index `(id, "iso", start, count)`; otherwise append one row and `(id, "list", start, 1)`. |
| `infer.py:141–159` `system_one` / `_Keep` | Use state-first, independent rows and checkpoint isolation. Call the renderers, then `plan_rows`; wrap every planned row in a separate `Example(context, [Q(text, options, gold=0)])`. `_Keep.shuffle` does nothing and `_Keep.sample` returns the first `k` elements. There is no label shuffle. |
| `prompt.py:111–128` `build` prefix and options | Encode **one piece** `"Context:\n" + renderedState`, with special tokens disabled; take its first `max_ctx_tokens` ids. Start the output with that prefix. Preserve option order. Validated API widths never exceed `max_options=255`, so the training-time gold/abstain retention branch is not used. |
| `prompt.py:129–133` narrow row | Since independent rows contain one question, use the unnumbered head `"\n\nQuestion: <text>\nOptions:"` and tail `"\nAnswer: ("`. For at most 10 options, concatenate head, all `"\n(A) <text>"` … lines, and tail; encode this entire question block as **one piece**. |
| `prompt.py:134–139`, `27–35` wide row | For more than 10 options, encode the head separately. For every option append `encode("\n(")`, the single label id, and `encode(") " + optionText)`; append separately encoded tail. Do not tokenize the decoded wide prompt as one string. Cache option suffix encodings by tokenizer identity and text if useful. |
| `prompt.py:140–145` row metadata | Append question-piece ids to context ids; record `slot = ids.count - 1`, `nopts`, identity permutation and gold index 0. Gold is fixture/training metadata, never a supplied answer token. |
| `prompt.py:38–55`, `148–152` label table | Enumerate A..Z followed by AA..AZ, BA..BZ, … in lexical nested-loop order. Encode each candidate without special tokens, retain only candidates encoding to one id, and stop at 255 distinct ids. This tokenizer gives A..Z plus the first **229 single-token** pairs. Assert the first ten ids are those for A..J. Store strings and ids together; do not assume every pair survives or that ids are contiguous. |
| `model.py:16–25`; `infer.py:169–174` slot readout | Author fp32 logits are the slot hidden state projected onto the 255 label head rows, with positions beyond `nopts` masked to negative infinity. Core AI already returns a full vocabulary fp16 vector: gather the first `nopts` label ids, promote to Float32, and apply stable softmax at temperature **1.03**. This is equivalent to masking unused labels; full-vocabulary softmax is wrong. Keep raw probabilities until assembly. |
| `systemone.py:82–86`, `101–110` isolated assembly | Gather `pYes[j] = rowProbabilities[start+j][1]`. Let mass be their sum, or `1e-9` if zero, and level probabilities be `pYes / mass`. Feed these to `format_answer`, then add per-level fit values and fit mass rounded to four places. List rows go directly to `format_answer`. |
| `systemone.py:113–116` `certainty` | Compute `H = -sum(p * log(p))` over positive probabilities; return `max(0, 1 - H/log(n))`, or 1 for a single option. Certainty is normalized entropy, distinct from confidence. |
| `systemone.py:119–129` `format_answer` | Slice to the original option count and renormalize by the sum, using 1 when the sum is zero. Argmax breaks ties in favor of the first option. Choice returns name, maximum probability, certainty and name→probability map. Noul returns only `type: noul` and `noul: P(yes)`. Score returns `sum(i*p[i])` rounded to two places, confidence, certainty, zero-based string-keyed legend/probabilities; other numeric outputs use four places. Match Python rounding, including ties, rather than formatting and reparsing locale-dependent strings. |
| `systemone.py:132–139`; `infer.py:175–176` usage | Report longest-common-prefix length plus the sum of each row's remaining length, even before a Swift prefix cache exists; this is the author's API accounting, not measured work. Report `output_tokens=0`. Internal diagnostics may separately count executed steps. |

The two token limits describe different APIs: `prompt.build` and `decide_batch` default to **1,536 context-prefix tokens** (`prompt.py:111`, `infer.py:85`), while `system_one` explicitly supplies **32,768** (`infer.py:141,158–159`). For state-first this slice includes the `Context:\n` header; question/options/slot tokens are appended afterward. Neither number is this bundle's usable capacity. The exported graph has a **4,096-token total context**: the port can construct the author's row first, then must reject a row longer than 4,096 before inference. Do not silently change the author truncation policy to fit a long state. The approved fixture's 255-option row is 1,965 tokens; all other fixture rows satisfy their 1,024-token cap. Those caps are test construction rules, not the op's general API limits.

`neutralize_none=false` is significant: keep all option strings, including abstain wording, unchanged. Do not apply `infer.py:29–40`'s replacement with `not listed here`. `abstain_below` is not part of `system_one` assembly. No additional confidence threshold belongs in the port.

Propose a versioned `metadata.json` `extra.system_one` object carrying `version: "0.8b-v1"`, `temperature: 1.03`, `temperature_schema_first: 1.03`, `neutralize_none: false`, `max_options: 255`, `max_state_tokens: 32768`, `schema_first: false`, `isolated_levels: true`, `layout: "state_first"`, and a tokenizer/label-table fingerprint. These values come from the pinned `decider_config.json`, whose file hash is in `results/download.json`. This is a future metadata/schema change: the two measured bundles' metadata remains unchanged, and the kit must not silently default to temperature 1.0 when the contract metadata is missing.

## 3. Tokenizer parity contract

Swift must reproduce **every id**, slot and selected label id in `models/decider-0.8b/fixtures-decider-0.8b.json`, not merely an equivalent decoded string. Load the tokenizer embedded in the selected LanguageBundle, verify its revision/file fingerprints, and reconstruct its 255-entry label table. The reference contains the complete table; A..L map to ids 32..43 and the wide fixture exercises two-letter labels. This checkpoint has no BOS: disable automatically added BOS/EOS and other special tokens. Do not use a chat template.

Encoding boundaries are part of the contract. State-first context and question block are separate encodes. Narrow question head/options/tail are one encode; wide head, opening marker, individual label id, each option suffix and tail use the exact split described above. The apparent text `Answer: (` is not an instruction to concatenate an answer or retokenize everything together. The final existing prompt token is the slot whose next-token label logits are read.

The JSON renderer must match Python whitespace, Unicode escaping, string escapes, number formatting and insertion order. Keeping a numeric source lexeme alone is insufficient: parse it to the author's integer/float semantics and emit the Python-compatible representation; test `1`, `1.0`, negative zero and exponent spellings. Reject unsupported nonfinite numbers. For this run, the 44 stored id vectors are the authoritative conformance tests. Additional synthetic serialization tests should cover long arrays and existing `_index` fields, nested Unicode, empty descriptions, numerical Score-map ordering, ten versus eleven options, and duplicate/invalid inputs before execution. Preserve exact reference fixtures; add tests separately.

The full request set is 13 requests with 23 Choice rows of 3–10 options (4 have 6–10), 9 Noul rows, 10 isolated rows from 2 Score questions, one 11-option row and one 255-option row. Source: `models/decider-0.8b/fixtures-decider-0.8b.json` composition, with row identities and first-43 preservation proven by `results/round2-summary.json` and `results/oracle-wide.json`.

## 4. Readout paths and recommendation

The available evidence separates engine-generated argmax from probability readout. The Release pipelined engine produced the exact oracle label on **44/44 rows for each bundle** (`results/engine_argmax_{fp16,int8hu}.json`). The current engine explicitly throws for `includeLogits` (`CoreAIPipelinedEngine.swift:108–112`); its API exposes no logits. This limitation is source-established, while the generation and Python readout paths were actually measured. The sequential engine's constructor requires exactly two states (`CoreAISequentialEngine.swift:112–125`), so its existing logits API cannot load this four-state hybrid.

The kit already has the right distinction between engine capabilities and consumers. `ConstrainedLoop.swift:44–65` requests `InferenceOptions(maxTokens: 1, includeLogits: true)` and takes `output.logits`; `KitExecutor.swift:119–132` throws if `engine.supportsLogits` is false and its constrained path delegates to that loop at lines 376–393. System One needs a slot distribution and no sampler/grammar. Reuse a capability-aware logits primitive, but do not route it through grammar-guided generation or interpret a sampled label as a probability distribution.

**Recommendation: add a dedicated `scoreLastToken(inputIDs:)` / read-last-logits primitive to the pipelined engine and use it in the kit.** It is the smallest change to the engine already carrying the hybrid state buffers and already tested for argmax here. It must return an owned Float32 array (or owned selected-label vector), copied from the completed step's fp16 logits, without sampling or advancing one extra token. Required mechanics:

1. Acquire exclusive session ownership, settle all prior work, and reset all four states per independent row. Keep `COREAI_CHUNK_THRESHOLD=1`, the existing extra-states bindings, and the full position vector `[0, …, t]` with shape `[1,t+1]` at step `t`. Feed `input_ids` as static `[1,1]`.
2. Encode the supplied prompt tokens through the existing S=1 prefill/encode path. Capture the buffer, shape, offset and step identity for the **final supplied prompt token**. `decodeLogitsBuffers` already allocates fp16 vocabulary buffers (`:844–852`), but the existing code selects `logits.metalBuffer` when supplied prompt tokens are nonempty and a rotating `decodeLogitsBuffers[step % pipelineDepth]` only for autonomous decode (`:1207–1210`, `:1266–1270`). Copy the actual final-prompt buffer, or deliberately route that final S=1 step to a dedicated readout buffer; reading the last decode ring entry blindly is wrong.
3. Await the inference stream's completion before CPU access. The existing source documents why merely committing on the same Metal queue can race the stream's pending submission and uses `await computeStream.currentWorkCompleted()` (`:1296–1306`). Keep the ring slot owned until the copy has completed, propagate command errors, and release the session only afterward. Never let another row overwrite the source during a read.
4. Read fp16 values, promote to Float32, gather labels and softmax at 1.03. Full-vocabulary copies are simplest for parity and diagnostics; later selected-label gather is an optimization requiring its own parity evidence. Do not call `generate(maxTokens: 1)` and then perform another decode step: the slot distribution belongs to the final prompt token, before any generated label is consumed.
5. Use an explicit scoring capability. Existing generation should keep its present semantics; do not globally advertise arbitrary `includeLogits` support until that separate contract is implemented. Cancellation must drain/settle and reset, and a failed read must not fall back to a guessed distribution.

**Fallback: compile and adapt the zoo's draft N-state low-level runner**, if the engine scoring patch cannot be integrated. `HybridCoreAIEngine.swift:3–6` explicitly says “DRAFT” and “NOT yet compiled”; its `:83–95` descriptor-driven allocation and `:117–129` state binding/readback are a design precedent, not a verified runtime. Adapt `generateGreedy`'s current whole-prompt `forward(promptTokens)` (`:132–140`) to a loop of single-token forwards: this exported graph's `input_ids` is static `[1,1]`. Its existing full-length `position_ids` construction (`:107–113`) is the right convention. Allocate `keyCache`, `valueCache`, `convState`, `recState` at the graph's declared dtypes/capacity, zero each row, run all prompt tokens, and read only the last `[1,1,248320]` fp16 logits. Bind the named descriptors explicitly and implement safe public NDArray helpers if this stays outside the language-model module. Compile, load and numerically gate it before treating it as usable; no fallback implementation or device claim is made here.

The currently proven probability path remains `python-gpu-aotc`: AOT h16c `.aimodelc` loaded with `SpecializationOptions.default()`, independent state dictionaries, S=1 steps, full positions and exact first-row reset proof. Both bundles passed 44/44 label argmax and 44/44 full-vocabulary argmax-is-label checks. On macOS 27.0 build 26A428, the Python JIT `.aimodel` path produced `MTL4CommandQueueErrorDomain error 1` and zero logits; the AOT path fixed that observed failure (`results/readout_fp16_round1_failed.json`, `results/aot.json`, `results/readout_{fp16,int8hu}.json`). This says nothing yet about whether an uncompiled Swift N-state fallback needs AOT; test its actual load path in the implementation run.

The supervisor accepted **fp16 max/mean |Δp| = 0.0050 / 0.00018** and **int8hu = 0.0084 / 0.00067**, where mean is the **mean of per-row means**. These are quoted Round 2 results, not new measurements. The ship bar is 44/44 letter argmax, maximum ≤0.02 and that mean ≤0.002 versus the author's fp32 oracle. The fp16 value is an empirical floor for this fixture/run; it is not a universal bound on fp16 rounding.

## 5. Cost model and prefix sharing

The actual work is the sum of independent row lengths. With S=1 prefill, its rate is of the same order as decode, not a batched transformer-prefill rate. If a state prefix is `S` tokens and row `i` contributes `Q_i` question/options/slot tokens, fresh-state scoring costs approximately

`steps = sum(S + Q_i) = N*S + sum(Q_i)` and `seconds ≈ steps / decode_tok_s`.

Score expands into one row per isolated level, so count planned rows rather than only request questions. This is an arithmetic estimate excluding load, tokenization, reset, buffer copying and scheduling; it is not a System One latency measurement. `usage.input_tokens` uses the author's logical prefix accounting even when executed work is larger.

Piece A measured the ship bundle on **Apple M4 Max GPU**, Release fork `397b337` plus the Xcode 27 RC initializer fix, `coreai-pipelined`, with `COREAI_CHUNK_THRESHOLD=1`, 128 prompt tokens, 256 generated tokens and 3 trials. Median prefill was **117.7925 tok/s** and median decode **109.1245 tok/s**. The run is **contended: true / measured on a shared machine**: other processes exceeded 10% CPU in the before/after snapshots. Evidence: `results/llm-benchmark.json`, `results/llm-benchmark-ps-before.txt`, `results/llm-benchmark-ps-after.txt` and `logs/llm-benchmark.log`. These figures support a rough cost model, not an isolated performance claim.

On that measured Mac decode rate, a **300-token state × 10 independent questions** incurs `3,000 / 109.1245373 = 27.49 seconds` for repeated state tokens alone. Add `sum(Q_i) / 109.1245373` for question/options/slot tokens. If each suffix is assumed to be 50 tokens, `(3,000 + 500) / 109.1245373 = 32.07 seconds`; with a future complete-state prefix fork, `(300 + 500) / 109.1245373 = 7.33 seconds` plus snapshot/copy overhead. These are explicitly derived estimates, not executed request timings.

For the family-only phone illustration, the zoo's `models/qwen3.5/README.md` at `347393e`, lines 180–187, reports **69.7–74.0 tok/s on iPhone 17 Pro** for the shipped Qwen3.5-0.8B family graph. Those are **the family's published measurements, not Decider measurements**. Ten independent questions over a 300-token state repeat `300 × 10 = 3,000` state-token steps: `3,000 / 74.0 = 40.54 s` to `3,000 / 69.7 = 43.04 s`, plus question tokens and overhead. At an illustrative 50 suffix tokens per row, the total is 3,500 steps, giving `47.30–50.22 s`. Fifty suffix tokens is an assumption for arithmetic, not a measured fixture length.

Prefix sharing can change the model to `S + sum(Q_i)` plus snapshot/copy overhead. With the same illustrative 50-token suffix, `300 + 10 × 50 = 800` steps instead of 3,500; the phone-family arithmetic is `800 / 74.0 = 10.81 s` to `800 / 69.7 = 11.48 s`, before state-copy overhead. This is a proposed optimization estimate.

The zoo's **`apps/coreai-prefix-cache.patch`** is the kit integration lever for cache ownership, retained-prefix length and full-sequence versus suffix feed conventions. It is **not by itself an implementation of sharing for this hybrid**: its lines 119–125 reject recurrent extra states, and the frozen pipelined engine's `supportsRewind` is false when `extraStates` is nonempty (`:1695–1699`). Rewinding KV offsets does not restore the convolution/recurrent scan. Extend the contract with a completed-prefix checkpoint containing both KV and **all** recurrent/conv state plus counters, then clone/restore it independently for every question suffix. Keep prefix snapshots immutable and keyed by exact prefix token ids, model, tokenizer and configuration.

The author's reference is `Engine.score_shared` (`oracle/decider/engine.py:136–155`): find the common prefix without swallowing the slot, compute it once, fork the cache with `cache.reorder_cache(...)`, run independent suffixes and read the corresponding slots. `infer.py:161–162` selects it for multiple state-first rows. That cache includes attention KV and delta-net conv/recurrent states. Implement fresh-row parity first; enable prefix forks only after proving equivalent row probabilities and independence across request order.

## 6. Implementation-run test plan

1. **Tokenizer before GPU:** port rendering/planning and compare all 44 complete id vectors, slot indices, identity permutations, option counts and label ids against `models/decider-0.8b/fixtures-decider-0.8b.json`. Require exact equality, including the 11-option rendering transition and the 1,965-token/255-option row. Verify no BOS and separate encoding boundaries. Keep all existing fixture inputs and reference values unchanged.
2. **Swift readout on both assets:** run the Swift path on fp16 first, then ship int8hu, using the validated 44 rows and fresh four-state zero initialization. Save raw slot logits and label probabilities. Require finite, nonconstant logits, full-vocabulary argmax in the row's label set on 44/44, and letter argmax equal to `models/decider-0.8b/fixtures-decider-0.8b.json` on **44/44 with no exemption**. Repeat the first row at the end and require identical logits. The ship int8hu gate is `max |Δp| ≤ 0.02` and `mean_of_row_means |Δp| ≤ 0.002`; report both that mean and the option-weighted mean with explicit names. Do not substitute the existing engine label-only gate for this test.
3. **API assembly:** compare row-based `assemble` and `CoreAI.systemOne` for all 13 requests, including each isolated Score's probabilities, expected score, legend, `level_fit` and `fit_mass`; verify confidence/certainty definitions and Python-compatible rounding. Direct probability-vector unit tests should assert exact assembly against the Python formatter. Against model inference, preserve the probability tolerance above and report any final rounded-field differences rather than pretending fp16 output is bitwise fp32 API output. Noul must be P(yes), not a Boolean.
4. **Completion and ownership:** exercise back-to-back calls, different request orders, concurrent callers serialized by catalog id, cancellation during prefill/readout, and reset after a command failure. Verify no stale ring read and no extra generated-token forward; observe output-token usage 0. Test multi-row final-logit selection separately from autonomous decode ring selection.
5. **Limits and errors:** test missing contract metadata, incompatible catalog model, 1/256 Choice options, 1/11 Score levels, malformed JSON, duplicate ids, empty instructions, 4,096-token boundary and over-capacity rejection. Test the author's 1,536 and 32,768 context-prefix truncation policies in CPU-only builder tests without suggesting the bundle executes 32,768 tokens.
6. **Optional prefix optimization afterward:** compare each forked result to fresh-state execution, alternate question order, verify all four checkpointed states are restored, and require the same numerical/argmax gate. Record actual step reduction and end-to-end latency separately; published family rates and this document's arithmetic do not constitute those measurements. The low-level fallback, if chosen after integration failure, must pass this same suite before exposure through the kit.

No tolerance changes, phone results, catalog release or publication are authorized by this design. The implementation run should use this document plus the immutable run evidence and the owner's selected integration branch.
