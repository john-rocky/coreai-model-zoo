# LFM3-1.7B — Core AI

LiquidAI's [LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) on Apple's Core AI: a 1.7B model that
reasons before it answers and calls tools in XML, in two bundles with a 4,096-token context. For the Mac, one
int8 `.aimodel` on the `coreai-pipelined` GPU engine that prefills a prompt 64 tokens at a time; for the iPhone
18 Pro, a Neural Engine build with fixed input shapes, compiled in advance for that phone's chip (`h19p`).
CoreAIKit's `ChatSession` and Apple's `LanguageModelSession` drive both.

Source checkpoint: revision `ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7` (LFM Open License v1.0).

**⬇️ Converted bundles:
[mlboydaisuke/LFM3-1.7B-CoreAI](https://huggingface.co/mlboydaisuke/LFM3-1.7B-CoreAI)** —
`gpu-pipelined/lfm3_1_7b_decode_int8hu_block32_sym_pf64/` (Mac, 2.72 GB: the `.aimodel`, `metadata.json` and the
tokenizer with its chat template) and `ios-h19p/` (iPhone 18 Pro, 2.16 GB: the `.aimodelc` compiled for its Neural
Engine, `metadata.json` and the same tokenizer), with the upstream LICENSE beside them.

## What it is

24 layers = 13 short-conv mixers (depthwise causal conv, kernel 7) + 11 GQA attention layers — 6 full, 5 with
a 256-token sliding window — hidden 2048, MLP 7680, 16 q / 4 kv heads of 128, vocab 128 000, tied head. Each
attention layer scales its heads by a sigmoid gate before the output projection; the full layers rotate half
of each head (θ 1e6), the sliding ones all of it (θ 1e4); every block is sandwich-normed. There is no
recurrent scan, so the decode graph is loop-free. In the Mac graph the state is one KV pair for the 11 attention
layers and one fixed conv state `[13, 1, 2048, 6]`; the window is applied inside the graph, and the host passes
positions only. What is new against LFM2.5, and how each part was gated:
[`knowledge/lfm3-1.7b-port.md`](../../knowledge/lfm3-1.7b-port.md).

The Mac bundle is `int8hu`: the MLP and conv-mixer linears int8 per-block-32, the head untied and int8
per-block-32 symmetric, the attention projections (q, k, v, o and the gate) and the q / k norms in fp32,
everything else fp16. The `.aimodel` holds two functions over the same weights: `main`, the one-token decode
graph, and `prefill`, a static 64-token query over the same positions, KV and conv state. The pipelined engine
finds `prefill` by name and sends the prompt through it 64 tokens at a time while more than 64 tokens remain; the
last 1–64 tokens of the prompt and the decode go through `main`.

The iPhone bundle is the same 24 layers rebuilt the way Apple's iOS Neural Engine path requires (fixed shapes
throughout; coreai-models' `models/ios/lfm3.py`, shipped in the zoo's overlay under `conversion/overlay/`): each
conv layer's input history kept in its own rows of the KV cache (the 6-column history read as one 8-position
block), the sliding window made from the engine's causal mask, the output gate and the MLP's SiLU in a
tanh form, one cos / sin table for both rotary types; 8-bit k-means palettized weights in groups of 32 (the
attention gate fp16) and the tied head as an int8 table. Its functions cover contexts of 256 to 4,096 tokens at
queries of 8, 16 and 64. Compiled for the iPhone 18 Pro (`h19p`, `--preferred-compute neural-engine`), each of its
30 transformer functions gets a Neural Engine region; the `.aimodelc` is 2.15 GB, 1.48 GB of it weights in
`resources.bin`.

## Speed

**Mac** (M4 Max, macOS 27.0): one measurement window on 2026-10-31, the same 1,024 prompt ids for every runtime,
256 tokens generated greedily without a stop, one process per run (its warm-up + one timed request), 8 rounds with
the order rotated; median of the 8 runs, the range in brackets. This bundle on CoreAIKit's path is the published
`.aimodel`, compiled on first load, run by the coreai-models version CoreAIKit pins: the path an app that uses
CoreAIKit takes (`llm-benchmark` built from the coreai-models fork's tag `0.2.8-zoo`); the other rows are
LiquidAI's own MLX and GGUF files.

| runtime | weights | prefill tok/s | first token | decode tok/s | peak footprint |
|---|---|---:|---:|---:|---:|
| **Core AI, this bundle, the kit's path** | int8hu, chunked prefill | **1,234** [1,217–1,246] | **0.83 s** | **169.5** [159.5–171.1] | 0.54 GB |
| Core AI, this bundle compiled ahead of time (h16c), the same engine | the same | 1,023 [1,008–1,035] | 1.00 s | 105.5 [103.8–106.6] | 2.09 GB |
| MLX (mlx-vlm 0.7.11) | `LFM3-1.7B-MLX-8bit` | 4,198 [4,172–4,211] | 0.24 s | 189.8 [176.8–192.0] | 3.04 GB |
| MLX | `LFM3-1.7B-MLX-4bit` | 4,236 [4,161–4,249] | 0.24 s | 286.9 [246.8–290.2] | 2.40 GB |
| MLX | `LFM3-1.7B-MLX-bf16` | 4,405 [4,381–4,416] | 0.23 s | 113.1 [49.7–114.8] | 4.51 GB |
| llama.cpp (b12001, Metal) | `LFM3-1.7B-GGUF` Q8_0 | 4,079 [4,051–4,104] | 0.25 s | 180.2 [172.3–183.6] | 0.40 GB |
| llama.cpp | Q4_K_M | 3,901 [3,855–3,911] | 0.26 s | 242.1 [219.8–244.9] | 0.40 GB |
| llama.cpp | F16 | 4,215 [4,079–4,234] | 0.24 s | 111.1 [57.8–114.1] | 0.40 GB |

MLX and llama.cpp decode faster than this bundle at 8 and at 4 bits and reach the first token 3.2–3.6× sooner;
every row gives transformers bf16's verdicts on the 8 questions below. Peak footprint is what each process was
charged (llama.cpp maps the GGUF, MLX holds its arrays, Core AI maps the specialized asset), not the weights' size.
What this port adds is a Neural Engine bundle for the iPhone and tool calls through Apple's `LanguageModelSession`.
Compiled ahead of time the Mac bundle is slower (second row), so it ships as the `.aimodel`.

The chunked prefill is what buys the first token: in earlier windows (2026-10-30, the `.aimodel` compiled ahead of
time, `llm-benchmark` built from an older coreai-models engine than the kit's tag) the same weights with `main` only
reached it in 9.44 s against this bundle's 0.99 s at about the same decode speed, and a 3,712-token prompt decoded
at 93.1 tok/s.

**iPhone 18 Pro** (iOS 27.2): `ios-h19p/` on the fixed-shape (`static-shape`) engine of the coreai-models version
CoreAIKit pins (a test app around that engine), a 768-token prompt and 256 generated tokens, 2 launches × 3 trials;
median, the range in brackets.

| prefill tok/s | decode tok/s | a 768 + 256 turn | peak footprint | thermal state | load |
|---:|---:|---:|---:|---|---|
| **590.6** [583.7–592.7] | **47.17** [46.28–47.22] | 6.71 s | 2.82 GB | nominal at the start and end of every run | in this app: 34.9 s the first time on the phone, then 4.4 s or less |

CoreAIKit's test app, with the bundle at its own path, loaded it in 32.4 s the first time, then in 3.7 s and 0.4 s.

The turn column is computed from the medians (768 / prefill + 255 / decode), not timed. Through
`ChatSession(catalog:)` on the same phone, the kit's own counters gave a median decode of 50.0 tok/s over the
thinking-on answers and 0.11 s to the first token of a short prompt.

## Accuracy — what was checked

Against transformers 5.20.2 on the same weights. Mac rows: M4 Max, macOS 27.0; iPhone rows: iPhone 18 Pro
(iPhone19,2), iOS 27.2. Transcripts are in this directory.

| check | how | result |
|---|---|---|
| the re-authored module | PyTorch fp32 vs transformers fp32 on 17 sequences (8 questions with thinking off and on, a 1,648-token document): one call per sequence, S=1, and the engine's own walk | max\|Δlogit\| ≤ 2.61e-3, top-1 2682 / 2682 |
| the graph in fp32 | the fp32 export on the GPU, teacher-forced, the same 17 | 2.20e-3, top-1 2682 / 2682 |
| the graph in fp16 | the same | max 0.353, top-1 2676 / 2682 — under the strict bar; the module run in fp16 in PyTorch shows the same width, so it is the fp16 activations, not the conversion |
| **the Mac bundle** | the same, vs transformers fp32, the `.aimodel` compiled ahead of time for the Mac GPU and the prompts fed as the engine feeds them (64-token chunks through `prefill`, the rest through `main`) | **top-1 2639 / 2682 = 98.40 %** (bar ≥ 93 %); through `main` alone 2637 / 2682 (1027 / 1035 on the short sequences, 1610 / 1647 on the document) |
| the chunked prefill | the same bundle, chunked vs `main` alone | the logits at the 49 chunk boundaries agree to cos ≥ 0.9999922; `main` alone equals the `main`-only build at all 2,683 positions |
| 8 short questions, thinking off | greedy; right = the reply holds the expected answer (42, Tokyo, cold, 7, merci, 56, 0.9, blue) | 8 / 8, the verdicts of transformers bf16; 7 of 8 replies word for word |
| token gate | `conversion/coreai_gate.py`, 16 greedy tokens vs the fp32 oracle | 16 / 16 (alphabet); 16 / 16 on a 215-token prompt that rides three chunks; 8 / 8 then `<\|im_end\|>` (17 + 25, the `main`-only build) |
| stop token | "Reply with only the number: 1+1=?", thinking off | stops on `<\|im_end\|>` after 2 tokens |
| thinking on | the 8 questions with thinking on, greedy, a 3,584-token budget, 3 runs of fresh processes, on the `main`-only build (these prompts never reach `prefill`, and `main` gives the same values) | every run 8 / 8: `</think>` closes after 40–237 tokens of reasoning (median 56) and the answer after it is right; the three runs answer alike |
| a tool call | one tool returning a fresh random value per run, through CoreAIKit's `KitLanguageModel` and the zoo's `ZooLanguageModel` behind Apple's `LanguageModelSession`, 3 runs each | 6 / 6: the XML call parsed into `lookup_secret(key: "alpha")`, run by the framework, the value in the answer verbatim, the reasoning kept out of it |
| the published bytes | the `.aimodel` re-saved without its source locations (a new `main.hash`), the checks above run again on it | teacher-forced values equal to the export's at all 2,683 positions, chunked and through `main` alone; 8 / 8; 16 / 16 (alphabet and the 215-token prompt); stops after 2 tokens; the tool call 6 / 6; CoreAIKit's catalog path answers with thinking on and off and runs the tool |
| the kit's path on the Mac | the published `.aimodel` specialized on load by the kit's pinned engine (`llm-runner` from the fork's tag `0.2.8-zoo`), and CoreAIKit's `ChatSession(catalog:)` / `KitLanguageModel` on it (the teacher-forced pass above was not repeated on this path) | 8 / 8 with the replies of the ahead-of-time build; 16 / 16 (alphabet and the 215-token prompt); the stop-token prompt stops after 6 tokens (`1+1=2`; transformers fp32 stops after 2, at a step where its own top-2 gap is 0.005); thinking on 3 launches × 8 / 8, thinking off 8 / 8, the tool round trip 3 / 3, a 3-turn chat |
| **the iPhone bundle** | `ios-h19p/` on the kit's static-shape engine, teacher-forced over the replies of the 17 sequences (721 steps, up to 64 a sequence after its prompt) vs transformers fp32 | **top-1 692 / 721 = 95.98 %** (bar ≥ 93 %; every step's fp32 top-2 gap ≥ 1e-3); a poisoned fixture fails, as it must |
| against Apple's engine | the same 18 prompts through `StaticShapeEngine` of apple/coreai-models' main on an iPhone 17 Pro (the same weights compiled for h18p) | the same tokens on all 18, teacher-forced and generated |
| iPhone: 8 short questions, the stop token | thinking off, greedy | 8 / 8, the verdicts of transformers bf16; the stop-token prompt stops after 6 tokens |
| iPhone: CoreAIKit's catalog path | `ChatSession(catalog: "lfm3-1.7b")` → the `ios` variant → `ios-h19p/` (copied into the kit's model store at the path a download would use, so no download ran) → the static-shape engine, 6 launches | thinking on: 3 launches × 8 / 8, `</think>` inside the budget and the reasoning out of the answer; thinking off: 8 / 8 verdicts; a tool round trip through `KitLanguageModel` 3 / 3 with the value verbatim; "What is 17 + 25? Answer briefly." → "17 + 25 = 42."; a 3-turn chat recalls its first turn, every turn on the stop token |

On the iPhone, with thinking off, 1 of the 8 questions ("How do you say 'thank you' in French?") still opens
`<think>` and reasons for 68 tokens before its answer, which gives "Merci"; transformers bf16 and the Mac bundle
answer it directly. Apple's engine on the iPhone 17 Pro does the same, so this comes from the Neural Engine export
itself, not from CoreAIKit's engine.

## Use it

Needs a CoreAIKit build that supports LFM3 (0.7.3 does not). The catalog entry has a macOS variant (the Mac bundle
on the pipelined engine) and an iOS variant (`ios`, on the static-shape engine).

```swift
import CoreAIKit

let chat = try await ChatSession(catalog: "lfm3-1.7b")
for try await event in await chat.streamResponse(to: "What is 17 + 25? Answer briefly.") {
    switch event {
    case .thinking(let delta): print(delta, terminator: "")   // the reasoning comes first
    case .response(let delta): print(delta, terminator: "")   // then the answer
    default: break
    }
}
```

Thinking is on by default — the chat template ends the prompt with `<think>` — and `maxResponseTokens` covers
the reasoning and the answer together. To answer without it:

```swift
var configuration = ChatSession.Configuration()
configuration.enableThinking = false
let chat = try await ChatSession(catalog: "lfm3-1.7b", configuration: configuration)
let reply = try await chat.respond(to: "What is 17 + 25? Answer briefly.")
```

LiquidAI's own note on that switch: it "can be inconsistent, especially for reasoning-heavy tasks"
([model card](https://huggingface.co/LiquidAI/LFM3-1.7B)).

On an iPhone the kit takes `ios-<arch>/` for the architecture Core AI reports for the phone
(`AIModel.deviceArchitectureName`: `h19p` on the iPhone 18 Pro) and falls back to `ios/`. The repo has `ios-h19p/`
only, so on any other iPhone (the iPhone 17 Pro, which reports `h18p`, included) the kit finds no bundle and throws
`variantNotFound` before it downloads anything. Both apps that ran the iPhone checks carried the
`com.apple.developer.kernel.increased-memory-limit` entitlement (peak footprint 2.82 GB in the gate app, up to 2.94 GB
through `ChatSession`); a run without it was not tried.

## Tool calling

LFM3 calls tools in XML, one block per call:

```
<tool_call>
<function=lookup_secret>
<parameter=key>
alpha
</parameter>
</function>
</tool_call>
```

CoreAIKit's `KitLanguageModel` and the zoo's `ZooLanguageModel` (`swift/Sources/ZooFMProvider`) render the
tool definitions the way the chat template does, parse these blocks (values typed by the tool's schema) and
hand them to Apple's `LanguageModelSession`, which runs the tool and asks again:

```swift
import CoreAIKit
import FoundationModels

struct LookupSecretTool: Tool {
    let name = "lookup_secret"
    let description = "Returns the secret value stored under a key in the user's vault."
    let secret: String
    @Generable
    struct Arguments {
        @Guide(description: "Vault key")
        var key: String
    }
    func call(arguments: Arguments) async throws -> String { secret }
}

let entry = try await ModelCatalog.entry(forID: "lfm3-1.7b", expecting: .chat)
guard let id = entry.modelID else { throw CoreAIKitError.modelNotAvailableOnPlatform(id: entry.id) }
let model = try await KitLanguageModel(model: id, engineVariant: .pipelined)
let session = LanguageModelSession(
    model: model, tools: [LookupSecretTool(secret: "GUZF-Z2TT")], instructions: "You are a helpful assistant.")
let answer = try await session.respond(
    to: "Use the lookup_secret tool to read the value stored under the key \"alpha\", then tell me that value.",
    options: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 2048))
// answer.content: The value stored under the key "alpha" is **GUZF-Z2TT**.
```

Run 3 times through each provider on the Mac with a fresh value the model cannot guess, and once more through the
catalog as above: every run called `lookup_secret` with `key: "alpha"` and answered with the value verbatim, the
reasoning in its own transcript entry. On the iPhone 18 Pro the same flow with `engineVariant: .staticShape` (the
iOS variant's engine, `entry.resolvedEngine`) did the same 3 times out of 3.

## Known limits

- **The Mac bundle cannot resume a conversation from its middle.** The conv state is not indexed by position, so
  the engine cannot rewind its conv state; a new conversation starts from zero state. `ChatSession`
  and the FoundationModels providers re-render the history every turn and, when the engine refuses the rewind,
  prefill it again from zero state — a three-turn chat recalled its first turn through both `ChatSession` and the
  zoo's provider. Each turn costs a full prefill of the conversation. The iPhone bundle keeps the conv history in
  the KV cache: in the 3-turn chat on the phone the static-shape engine reused the earlier turns' prompt tokens
  (32, then 70).
- **On the Mac the tail of a prompt goes one token at a time.** The engine always runs the last prompt token
  through the one-token `main` function, so a prompt's final 1–64 tokens run through `main` at about the decode
  speed; a 1,024-token prompt is 15
  chunks and then 64 single steps, about half of its first-token time.
- **4,096 tokens.** Both bundles are exported with a 4,096-token context; the checkpoint itself allows more.
- **The iPhone bundle is for the iPhone 18 Pro only.** The iPhone 17 Pro has no bundle here: the same static form
  compiled for it (h18p) passed the same checks on Apple's engine (a 768 + 256 turn in 10.02 s), but CoreAIKit's
  path has not run on that phone. The Mac bundle is not for iPhones: on an iPhone 17 Pro these weights built with `main`
  only and compiled ahead of time for its GPU decoded at 9.25 tok/s with the phone at thermal state serious (that
  compile holds fp16 copies of the int8 linears); on an iPhone 18 Pro the `main`-only `.aimodel`, specializing on
  the phone (device JIT), had not built its engine 300 s after launch (the cause is not isolated); and on iOS the
  pipelined engine — the Mac bundle's engine, not the iPhone bundle's — caps prompt + answer at 1,024 tokens. Details:
  [the port notes](../../knowledge/lfm3-1.7b-port.md#iphone).
- **Two compiles of one graph in one app.** On a Mac, Core AI named the
  compile-cache entry of an ahead-of-time compile after its graph and not its weights, so a second compile of the same
  graph with other weights ran with the first one's weights (not checked on an iPhone). An app that switches between
  such compiles should delete that cache entry first; the Mac bundle is a JIT `.aimodel`, whose cache entry name
  covers its weights.
- **fp16 activations.** Teacher-forced over a 1,648-token document, the fp16 graph changes the top-1 at 6
  positions where transformers' own top-2 gap is 0.003–0.061, which the fp32 graph keeps; the int8 bundle's
  98.40 % (98.32 % through `main` alone) includes that floor.
- **The chat template is rewritten, and renders the same string.** The vendor template does not parse in
  swift-jinja 2.3.2 (the pin of `apple/coreai-models`) and renders tool definitions differently in 2.3.6; the
  bundle's template has eight rewrites that render the same string as the vendor's under transformers. When it
  cannot parse a template, `CoreAILanguageModel` in `apple/coreai-models` (and in the zoo's fork of it) falls back
  to the messages' text joined by newlines, without a chat template, and logs one line that is off by default —
  check the rendered prompt when you swap the tokenizer.

## Reproduce

`recipe.toml` holds both exports: `python3 conversion/zoo_convert.py run --force lfm3-1.7b` runs its six steps (the
chat template; the Mac export and its strip; the static export, its strip, and the compile for the iPhone 18 Pro's
Neural Engine into `ios-h19p/`). `conversion/export_lfm3_decode_pipelined.py` is the Mac exporter,
`conversion/export_lfm3_ane_static.py` the static one, `conversion/ios_device_bundle.py` lays out the iPhone
bundle, and `conversion/coreai_gate.py --arch lfm3` is the token gate.

```bash
coreai-models/.venv/bin/python conversion/export_lfm3_decode_pipelined.py int8hu \
    --head-quant block32 --head-sym --prefill-chunk 64 \
    --src LiquidAI/LFM3-1.7B --revision ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7 \
    --hf-id LiquidAI/LFM3-1.7B --max-ctx 4096 --chat-template <the bundle's chat_template.jinja>
python3 conversion/coreai_gate.py exports/lfm3_1_7b_decode_int8hu_block32_sym_pf64 LiquidAI/LFM3-1.7B \
    --revision ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7 --arch lfm3 -n 16
```

The export needs the zoo's overlay of `apple/coreai-models` (`conversion/overlay/`; the LFM3 modules are in its
`files/`), and `conversion/lfm3_chat_template.py` writes the bundle's chat template from the vendor's. The
exported `main.mlirb` records the export machine's source paths; the published `.aimodel` is the export
re-saved after coreai-torch's `strip_debug_info` (`conversion/d1/strip_bundle.py` does this for a bundle whose
`assets.main` is the `.aimodel`), which is a new `main.hash`, so the gates above were run again on it. A
re-export does not reproduce that `main.hash`: each export draws new names for its externalized graphs and
weight resources (on 2026-10-30 four re-exports from a fresh clone matched the published `main.mlirb` section
by section except for those names and the lengths that encode them, with the same 3,971 ops). The iPhone bundle
is compiled from its stripped `.aimodel` with `xcrun coreai-build compile <name>.aimodel --platform iOS
--preferred-compute neural-engine --architecture h19p` and is checked on the phone only: an iOS bundle is never
loaded on a Mac. Running the Mac bundle needs the pipelined engine with one extra state and
`COREAI_CHUNK_THRESHOLD=1` (`main` takes one token; the engine feeds `prefill` whatever the threshold);
`llm-runner` takes `--inference-engine-variant coreai-pipelined --warmup off`.

## License

The weights are under the **LFM Open License v1.0**
([LICENSE](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE)): Apache-2.0-style grants, with
§5: commercial use by a legal entity at or above US$10 M annual revenue is not licensed (the threshold does
not apply to a qualified non-profit's non-commercial or research use). The bundle on Hugging Face ships that
LICENSE file. The conversion code in this repo stays BSD-3-Clause. Not affiliated
with Apple or LiquidAI.
