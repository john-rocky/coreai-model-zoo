# Zoo models behind Foundation Models' `LanguageModelSession`

> Verified 2026-06-11 (macOS 27 beta, M4 Max): a zoo pipelined bundle backs Apple's standard
> `LanguageModelSession` **with zero new code** — `CoreAILanguageModel(resourcesAt: bundleDir)`
> is the entire integration (Qwen3.5-0.8B int8: load 3.7 s, first turn 0.41 s, multi-turn OK).
> The one capability Apple's adapter lacks is **tool calling**; a ~200-line own `LanguageModel`
> conformance added it, and the full round trip — model emits a call, the framework runs the
> Swift `Tool`, the model answers grounded on the result — worked on the first run, hybrid
> 4-state Qwen included. Source session: WWDC26 339 "Bring an LLM provider to the Foundation
> Models framework". Apple's adapter source ships in `coreai-models`
> (`swift/Sources/CoreAILanguageModels/LanguageModel/`).

Why this matters: every app written against the Foundation Models framework — the same
`session.respond(to:)` / `streamResponse` / `Tool` / `@Generable` API as Apple's built-in model
— can switch to a zoo model by changing one line. Anthropic and Google announced FM provider
packages for Claude/Gemini; MLX has `MLXLanguageModel` (`ml-explore/mlx-swift-lm`); Hugging Face
ships `AnyLanguageModel`. Local Core AI models join that ecosystem through the path below.

## The protocol in 60 seconds (macOS/iOS 27)

Two pieces (verified against the 27-beta `FoundationModels.swiftinterface`):

```swift
protocol LanguageModel: Sendable {
    associatedtype Executor: LanguageModelExecutor where Self == Executor.Model
    var capabilities: LanguageModelCapabilities { get }   // .vision/.guidedGeneration/.reasoning/.toolCalling
    var executorConfiguration: Executor.Configuration { get }
}
protocol LanguageModelExecutor: Sendable {
    associatedtype Configuration: Hashable, Sendable      // per-session executor cache KEY
    init(configuration: Configuration) throws
    func prewarm(model: Model, transcript: Transcript)    // careful: default no-op exists
    nonisolated(nonsending) func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: Model,
        streamingInto channel: LanguageModelExecutorGenerationChannel) async throws
}
```

The session hands the executor the **full transcript on every `respond`** (entries:
`instructions / prompt / toolCalls / toolOutput / response / reasoning`), plus
`enabledToolDefinitions`, an optional `schema`, and `generationOptions`. The executor streams
events back: `.response(action: .appendText(...))`, `.reasoning(...)`,
`.toolCalls(action: .toolCall(id:name:action: .appendArguments(json)))`, `.updateUsage`,
`.updateMetadata`. One-shot `respond` is just collected streaming. KV reuse across turns is
the executor's job (diff the new transcript against the one you saved; invalidate at the
divergence point) — nobody does it for you.

## Quick start — Apple's adapter, zoo bundle

```swift
import CoreAILanguageModels   // product "CoreAILM" of the coreai-models package
import FoundationModels

setenv("COREAI_CHUNK_THRESHOLD", "1", 1)   // BEFORE engine creation (decode-only S=1 bundles)

let model = try await CoreAILanguageModel(resourcesAt: bundleDirURL)  // LanguageBundle dir
let session = LanguageModelSession(model: model, instructions: "You are a helpful assistant.")
let answer = try await session.respond(to: "Why is the sky blue?")
```

Requirements:

- A **LanguageBundle dir** (`metadata.json` + `.aimodel` + `tokenizer/`) — every bundle this zoo
  publishes is one.
- The **patched `coreai-models` package** for the non-standard architectures: the same
  extra-states / per-token-inputs / static-inputs patch stack the chat app already needs (see
  [`pipelined-engine.md`](pipelined-engine.md) and `apps/*.patch`). Plain-attention bundles run
  on the unpatched upstream; Qwen3.5 (hybrid GDN), LFM2.5, Granite (SSM), Gemma 4 tbl do not.
- `EngineFactory` auto-picks the engine from the bundle structure — pipelined for dynamic GPU
  bundles, sequential, or static-shape/ANE.

What you get for free from Apple's adapter: UTF-8-safe incremental detokenization, `<think>` /
`<|reasoning_start|>` auto-detection routed to `.reasoning` transcript entries, chat templating
via the bundle tokenizer, greedy default + temperature override.

## What works / what doesn't (today's beta, verified)

| Surface | Status |
|---|---|
| Plain chat, streaming, multi-turn | ✅ via Apple's adapter, zero code |
| Reasoning models (`<think>`) | ✅ routed to `Transcript.reasoning` entries automatically |
| Tool calling | ❌ in Apple's adapter (tool entries skipped, capability never declared) → ✅ `ZooFMProvider` (multi-call, streaming parse, **per-model dialect** — Hermes + LFM native) |
| Guided generation (`@Generable`, schema) | ⚠️ only when `engine.supportsLogits` — **GPU-pipelined engines sample on-GPU and return `false`**, so every zoo pipelined bundle lacks `.guidedGeneration`; the sequential engine has it. `ZooFMProvider` throws `unsupportedCapability` on schema requests |
| `session.prewarm()` | ❌ silent no-op for Core AI models (see trap 1) → ✅ `ZooFMProvider` (real 1-token generate + reset) |
| Usage accounting (`.updateUsage`) | ❌ placeholder in Apple's adapter → ✅ `ZooFMProvider` (per-turn, summed into `session.usage`, `cachedTokenCount` on KV reuse) |
| KV reuse across turns | ❌ Apple's adapter resets + re-prefills everything. `ZooFMProvider` implements the append-only fast path — measured on LFM2.5-1.2B int8 (turns ended by token cap): turn 2 reused 97 cached tokens and prefilled 18, per-turn latency flat at ~0.33 s instead of growing with history. Structural limits: the engine over-generates past EOS into the cache (see note above) and thinking models' templates strip historic `<think>` blocks the cache still contains — so EOS-ended/thinking turns still reset (measured: ~2.3-2.7 s turn-2 settle on the default 512-token budget). The real fix is engine-side (stop-at-break + KV truncate) |

## Tool calling with an own conformance

Apple's adapter doesn't do tools, but the protocol + the zoo engines have everything needed.
A minimal conformance that reuses the same public pieces (`CoreAIRunner(from:).makeInferenceEngine()`
+ `LanguageBundle.loadTokenizer()`) and speaks the Qwen/Hermes tool dialect:

1. Declare `capabilities = [.toolCalling]`.
2. In `respond`, render the transcript to ChatML yourself. Advertise
   `request.enabledToolDefinitions` in the system message inside `<tools>…</tools>` (each as
   `{"type":"function","function":{name, description, parameters: <JSONEncoder'd
   GenerationSchema>}}`); replay past `toolCalls` entries as assistant
   `<tool_call>{json}</tool_call>` turns and `toolOutput` entries as user-role
   `<tool_response>…</tool_response>` turns.
3. Generate (`engine.generate(with:samplingConfiguration:inferenceOptions:)`, break on
   `tokenizer.eosTokenId`), split a leading `<think>` block into a `.reasoning` event, then
   if the output contains `<tool_call>` parse `{"name", "arguments"}` and send
   `.toolCalls(action: .toolCall(id: UUID().uuidString, name: name,
   action: .appendArguments(argsJSON, tokenCount: n)))`; otherwise send the text as
   `.response`.
4. That's all — the **framework** parses the arguments against the tool's `@Generable` schema,
   executes the Swift `Tool`, appends the `toolOutput` entry, and calls `respond` again; your
   executor replays it and the model answers grounded.

Verified transcript on Qwen3.5-0.8B int8 (greedy, one shot):

```
instructions → prompt → reasoning → toolCall get_weather({"city":"Tokyo"})
             → toolOutput get_weather → response   (turn 4.6 s incl. both respond calls)
```

This recipe is packaged as the **`ZooFMProvider`** library in this repo's
[`swift/`](../swift/README.md) package: streaming incremental `<tool_call>`/`<think>` parse
(tags straddling token deltas are caught; text streams the moment it decodes), multi-call
turns (consecutive `.toolCalls` events coalesce into ONE transcript entry with N calls — and
the framework executes all of them before re-responding), usage events with
`cachedTokenCount`, `toolCallingMode` honoring (`.disallowed` drops the tools block,
`.required` renders a must-call instruction), and a working `prewarm`.

Two beta behaviors the packaged executor encodes (verified macOS 27.0 beta):

- **Don't send WWDC-339-style upfront usage/metadata.** A `.response(updateUsage:)` event on a
  turn that ends in tool calls materializes an EMPTY `Response` transcript entry. Send
  metadata + usage once at end of turn, attached to the entry kind the turn produced.
- **Breaking the token stream does not stop the pipelined engine.** It generates to
  `maxTokens` in the background and those post-EOS tokens land in the KV cache; the next
  `engine.reset()` blocks on them (and its internal drain traps after ~5 s — big slow models
  beware). The packaged executor pumps the stream through a task it can settle on the next
  respond instead of breaking the engine stream directly.

## Tool-calling dialects (per model family)

A model emits tool calls in the format it was *fine-tuned* on, and an in-context instruction
will not override that prior (trap 9). So tool calling can't share one renderer/parser across
families — each needs its own. `ZooFMProvider` factors this into a `PromptDialect`:

```swift
public protocol PromptDialect: Sendable {
    var name: String { get }
    var toolCallOpen: String { get }      // stream markers delimiting a call block
    var toolCallClose: String { get }
    func render(transcript:tools:requireToolCall:) -> String          // whole prompt, framing included
    func parseToolCalls(_ body: String, tools:) throws -> [ParsedToolCall]   // a block may hold N calls
}
```

The dialect owns the **whole** render (not just the tool block) because families differ in
framing too, not only call syntax. `ZooLanguageModel` auto-selects by probing the tokenizer
vocab (`defaultDialect(probing:)`); pass `dialect:` to override.

Three dialects ship, each verified against the bundle's own `chat_template.jinja` (render the
template with jinja2 and diff against the Swift output — the template is the spec):

| | Hermes (Qwen3.5, the default) | LFM (LFM2.5) | MiniCPM (MiniCPM5-1B/2B) |
|---|---|---|---|
| tools advertised | system `<tools>{json}…</tools>` block | system `List of tools: [{json}, …]` text | system `<tools>{json}…</tools>` block + the template's "Tool usage guidelines" paragraph |
| call syntax | `<tool_call>\n{"name","arguments"}\n</tool_call>` | `<\|tool_call_start\|>[fn(a="x"), fn2(n=3)]<\|tool_call_end\|>` (pythonic) | `<function name="fn"><param name="p">v</param></function>` XML, one block per call, CDATA for `<`/`&`/newline values |
| result replay | user-role `<tool_response>…</tool_response>` | `tool`-role `<\|tool_response_start\|>…<\|tool_response_end\|>` | user-role `<tool_response>…</tool_response>` |
| framing | ChatML `<\|im_start\|>` | ChatML `<\|im_start\|>` | ChatML `<\|im_start\|>`; `thinking: .off` renders `<think>\n\n</think>\n\n` (`enable_thinking=false`) |
| parse | JSON object (or array of objects) | tolerant pythonic scanner | regex over `<param>` elements, values typed against the tool's schema |

MiniCPM5-2B on the gate (int8 block-32, greedy, macOS 27 beta): `tools` PASS (two parallel calls
in one turn, grounded answer), `toolchain` PASS, and the `agent` scenario — the
`apps/CoreAIAgent` flow with fixed data: read tomorrow's calendar → "remind me 15 minutes before
the first one" → a `create_reminder` at 09:45 → device status — PASS with thinking on (cap 220)
and off (cap 120; prompts 541 / 701 / 789 tokens, so three turns fit the 1024-token iOS KV cap).

**Probe trap (fixed 2026-09-08):** `defaultDialect(probing:)` tested vocab membership with
`convertTokenToId(_:) != nil`, but some tokenizer models answer an unknown token with the unk id
instead of nil — MiniCPM5's does, so every LFM special token "existed" and the 2B was routed to
the LFM dialect (it still emitted its native XML; nothing parsed). The probe now also requires
`id != tokenizer.unknownTokenId`. MiniCPM5's own markers are single vocab tokens (`<function`,
`<param`, `</param>`, `</function>`), which is what its probe keys on; note its vocab also carries
`<tool_call>` tokens, so that Hermes probe cannot separate the two.

The LFM parser must be **tolerant**: the model emits half-mangled argument lists (single
quotes, bare/unquoted values, Python `True`/`None`, nested containers, truncated tails). It
salvages per-call (a broken call is skipped, the rest of the block still executes) and maps a
lone positional argument onto a single-parameter tool via the schema. The replay path sorts
kwargs so re-rendered calls are byte-stable (the KV fast path's prefix match depends on it).

Recon for two more families (templates read; dialects not yet built): **granite-4.0** uses
Hermes tool *syntax* but `<|start_of_role|>…<|end_of_role|>…<|end_of_text|>` framing (so it
needs its own dialect, not Hermes); **gemma4** is fully custom and non-JSON
(`<|tool_call>call:name{key:value}<tool_call|>`, a `<|"|>` quote token, `<|channel>thought`
for reasoning).

## Traps

1. **`prewarm` has a default no-op extension.** Implement `prewarm(model:transcript:)`
   *exactly* — implement `prewarm(transcript:)` and it compiles but is never called. Apple's
   own adapter has this today, which is why `session.prewarm()` does nothing for Core AI
   models: do your own warm-up (a 1-token generate after load).
2. **`request.enabledToolDefinitions`** is the property; `enabledTools` is only the
   memberwise-init label.
3. **`Configuration` is the executor cache key.** The session stores executors keyed by your
   Hashable `Configuration` — key it by bundle identity (+ anything that changes behavior).
   Apple keys by `(modelIdentifier, samplingConfig)`.
4. **`COREAI_CHUNK_THRESHOLD=1`** before engine creation for decode-only S=1 bundles, and
   never call `engine.warmup()` with the default query length on them (warms S=256, which the
   S=1 graph rejects) — same contract as [`pipelined-engine.md`](pipelined-engine.md).
5. **Pipelined ⇒ no `.guidedGeneration`.** Don't declare it without logits; schema requests
   on a pipelined bundle can't be honored (approximate-or-throw rule: throw
   `LanguageModelError.unsupportedCapability`).
6. **Multi-turn re-prefill tax.** Until an executor implements transcript diffing, budget
   ~decode-speed × history-tokens per turn on decode-only bundles (measured: turn 1 = 0.41 s,
   turn 2 = 2.8 s on the 0.8B with a 3-entry history + hidden thinking).
7. **Thinking is invisible in `response.content`** — it lands as `.reasoning` transcript
   entries. A "hanging" first response is usually the model thinking.
8. **Small `maximumResponseTokens` + a thinking model = no response at all.** If the cap cuts
   generation mid-`<think>`, the turn produces only reasoning events and the session throws
   "ended without producing a response". Budget caps with thinking headroom (or use a
   non-thinking model).
9. **Tool-prompt dialects don't transfer — render+parse each model's NATIVE format.** LFM2.5
   ignores in-context Hermes `<tool_call>`-JSON instructions and emits its trained
   special-token dialect (`<|tool_call_start|>[fn(arg=…)]<|tool_call_end|>`, pythonic) — the
   training prior wins over the prompt. So tool calling per model family needs its own
   rendering + stream-marker detection + call parsing. `ZooFMProvider` packages this as a
   `PromptDialect` protocol (see next section); Hermes (Qwen3.5) and LFM are implemented,
   picked automatically by probing the tokenizer vocab.

## Sources

WWDC26 339 "Bring an LLM provider to the Foundation Models framework" (protocol, executor
lifecycle, transcript-diff guidance, custom segments/metadata) and 241 "What's new in the
Foundation Models framework"; the `FoundationModels.swiftmodule` interface in the macOS 27
beta SDK (signatures above were read from it, not from docs); Apple's adapter source in
`coreai-models` `swift/Sources/CoreAILanguageModels/LanguageModel/CoreAILanguageModel.swift`.
