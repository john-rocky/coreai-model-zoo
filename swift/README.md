# Swift package — `ZooFMProvider` + `CoreAIRunner`

Two libraries in one package:

| Product | What | Status |
|---|---|---|
| [`ZooFMProvider`](Sources/ZooFMProvider/) | Zoo bundles behind FoundationModels' `LanguageModelSession`, with **tool calling**, streaming, usage events, and append-only KV reuse — the capabilities Apple's `CoreAILanguageModel` adapter doesn't implement | ✅ verified on macOS 27 beta (Qwen3.5-0.8B int8) |
| [`CoreAIRunner`](Sources/CoreAIRunner/) | Self-contained N-state engine on the low-level `CoreAI` system framework only (no `coreai-models` dependency) | ⚠️ DRAFT — authored on macOS 26.6, not yet compiled |

## ZooFMProvider

```swift
import FoundationModels
import ZooFMProvider

let model = try await ZooLanguageModel(resourcesAt: bundleDir)   // any zoo LanguageBundle
let session = LanguageModelSession(model: model, tools: [WeatherTool()])
let answer = try await session.respond(to: "What's the weather in Tokyo?")
```

The same `session.respond` / `streamResponse` / `Tool` / `@Generable` API as Apple's built-in
model. The framework executes your Swift `Tool` and the model answers grounded on the result.
Tool calls are rendered and parsed in each model's **native dialect** (picked automatically by
probing the tokenizer vocab): Qwen3.5 speaks Hermes `<tool_call>` JSON, LFM2.5 speaks its
pythonic `<|tool_call_start|>[fn(arg=…)]<|tool_call_end|>` special-token form, MiniCPM5 speaks
`<function name=…><param name=…>…</param></function>` XML (`MiniCPMDialect`, with a `thinking: .off`
option for the 1024-token iOS budget) — an in-context
format instruction does not override a model's training prior, so each family gets its own
`PromptDialect`. `<think>` blocks stream to `Transcript.reasoning` entries; `session.usage`
reports prompt/generated token counts including KV-cache reuse (`cachedTokenCount`).

**Requirements** — same as [`../apps/`](../apps/README.md): clone `coreai-models` at this repo's
root and apply the four-patch stack (the package depends on it by path; hybrid-architecture
bundles need the patches at runtime too):

```bash
cd ..   # repo root
git clone https://github.com/apple/coreai-models
git -C coreai-models apply apps/coreai-shared-product.patch \
                           apps/coreai-pipelined-extra-states.patch \
                           apps/coreai-pipelined-per-token-inputs.patch \
                           apps/coreai-pipelined-static-inputs.patch
cd swift
export DEVELOPER_DIR=/path/to/Xcode-beta.app/Contents/Developer
swift build -c release --product zoo-fm-gate
```

A path dependency was chosen over a URL dependency deliberately: SPM-managed checkouts are
immutable, so the patch stack — which the flagship hybrid bundles (Qwen3.5, LFM2.5, Granite)
require — could not be applied to one. Without the clone at the repo root, `swift build` fails
at manifest resolution; that is the documented trade-off.

Verification harness (macOS):

```bash
swift run -c release zoo-fm-gate <bundle-dir> chat        # plain-chat regress, streamed deltas
swift run -c release zoo-fm-gate <bundle-dir> tools       # two-tool round trip
swift run -c release zoo-fm-gate <bundle-dir> multiturn   # per-turn latency + KV reuse
swift run -c release zoo-fm-gate <bundle-dir> agent [cap] # apps/CoreAIAgent flow with fixed data (ZOO_FM_NOTHINK=1 for MiniCPM5 without its trace)
ZOO_FM_DEBUG=1 ...                                        # log KV fast-path / reset decisions
```

Verification harness scenarios also include `toolchain` (a dependent two-tool sequence) on top
of the three above.

Known limits (see [`../knowledge/fm-provider.md`](../knowledge/fm-provider.md) for the full
gap table): no `.guidedGeneration` on pipelined bundles (on-GPU sampling exposes no logits —
schema requests throw `unsupportedCapability`); one session per model instance at a time. Tool
dialects: Hermes (Qwen3.5) and LFM2.5 ship; granite-4.0 (Hermes syntax, `<|start_of_role|>`
framing) and gemma4 (custom non-JSON) are recorded as follow-ups.

## CoreAIRunner (draft)

- `Sources/CoreAIRunner/HybridCoreAIEngine.swift` — generic **N-state** engine (Apple's
  `CoreAISequentialEngine` is hard-coded to 2 states; Qwen3.5/Gemma 4 need 4), fixed-capacity,
  greedy generate. Drives the all-in-one stateful `.aimodel` (`input_ids,position_ids → logits` +
  N in-place states).
- `Sources/CoreAIRunner/NDArrayHelpers.swift` — self-contained NDArray fill/read (no dependency
  on Apple's CoreAILanguageModels module).
- `Sources/coreai-run/` — a minimal CLI to validate the engine on macOS (feeds raw token ids,
  greedy-decodes, prints) before the iOS app.

⚠️ Authored against the exact API used by Apple's `CoreAISequentialEngine.swift`, not yet
compiled. Note: Apple's `CoreAILanguageModels` module also declares a type named `CoreAIRunner`
— don't import both modules in one file, or qualify uses.

See [`../knowledge/swift-runtime.md`](../knowledge/swift-runtime.md) for the API + per-model
runtime contracts, and [`../apps/CoreAIChat/`](../apps/CoreAIChat/) for the iOS chat app that
embeds the same patterns.
