# LFM3-1.7B — Core AI

LiquidAI's [LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) on Apple's Core AI: a 1.7B model that
reasons before it answers and calls tools in XML, as one int8 `.aimodel` on the `coreai-pipelined` GPU engine
with a 4,096-token context that prefills a prompt 64 tokens at a time, driven from CoreAIKit's `ChatSession` or
Apple's `LanguageModelSession`.

Source checkpoint: revision `ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7` (LFM Open License v1.0).

**⬇️ Converted bundle:
[mlboydaisuke/LFM3-1.7B-CoreAI](https://huggingface.co/mlboydaisuke/LFM3-1.7B-CoreAI)** —
`gpu-pipelined/lfm3_1_7b_decode_int8hu_block32_sym_pf64/` (2.72 GB: the `.aimodel`, `metadata.json` and the
tokenizer with its chat template), with the upstream LICENSE beside it.

## What it is

24 layers = 13 short-conv mixers (depthwise causal conv, kernel 7) + 11 GQA attention layers — 6 full, 5 with
a 256-token sliding window — hidden 2048, MLP 7680, 16 q / 4 kv heads of 128, vocab 128 000, tied head. Each
attention layer scales its heads by a sigmoid gate before the output projection; the full layers rotate half
of each head (θ 1e6), the sliding ones all of it (θ 1e4); every block is sandwich-normed. There is no
recurrent scan, so the decode graph is loop-free. Its state is one KV pair for the 11 attention layers and
one fixed conv state `[13, 1, 2048, 6]`; the window is applied inside the graph, and the host passes
positions only. What is new against LFM2.5, and how each part was gated:
[`knowledge/lfm3-1.7b-port.md`](../../knowledge/lfm3-1.7b-port.md).

The bundle is `int8hu`: the MLP and conv-mixer linears int8 per-block-32, the head untied and int8 per-block-32
symmetric, the attention projections (q, k, v, o and the gate) and the q / k norms in fp32, everything else fp16.
The `.aimodel` holds two functions over the same weights: `main`, the one-token decode graph, and `prefill`, a
static 64-token query over the same positions, KV and conv state. The pipelined engine finds `prefill` by name and
sends the prompt through it 64 tokens at a time while more than 64 tokens remain; the last 1–64 tokens of the
prompt and the decode go through `main`.

## Speed

Mac (M4 Max, macOS 27.0): the `.aimodel` compiled for the Mac GPU (`coreai-build compile … --architecture h16c
--expect-frequent-reshapes`), `llm-benchmark` with 256 generated tokens, one process per run (its warm-up + one
timed trial), alternated with the `main`-only build of the same weights in the same quiet windows; median of the
runs, the range in brackets. Measured on the export before its source locations were stripped (the same ops).

| prompt | build | prefill tok/s | first token | decode tok/s | peak footprint | runs |
|---:|---|---:|---:|---:|---:|---:|
| 1,024 | **this bundle** | **1,030** [952–1,033] | **0.99 s** | **94.9** [93.1–95.9] | 2.08 GB | 8 |
| 1,024 | `main` only | 108 [105–109] | 9.44 s | 95.5 [94.8–96.1] | 0.42 GB | 9 |
| 3,712 | this bundle | 1,713 [1,697–1,718] | 2.17 s | 93.1 [89.9–94.3] | 2.13 GB | 8 |

No iOS bundle in this release: the iPhone runs are under [Known limits](#known-limits).

## Accuracy — what was checked

Against transformers 5.20.2 on the same weights; Mac (M4 Max, macOS 27.0). Transcripts are in this
directory.

| check | how | result |
|---|---|---|
| the re-authored module | PyTorch fp32 vs transformers fp32 on 17 sequences (8 questions with thinking off and on, a 1,648-token document): one call per sequence, S=1, and the engine's own walk | max\|Δlogit\| ≤ 2.61e-3, top-1 2682 / 2682 |
| the graph in fp32 | the fp32 export on the GPU, teacher-forced, the same 17 | 2.20e-3, top-1 2682 / 2682 |
| the graph in fp16 | the same | max 0.353, top-1 2676 / 2682 — under the strict bar; the module run in fp16 in PyTorch shows the same width, so it is the fp16 activations, not the conversion |
| **this bundle** | the same, vs transformers fp32, the prompts fed as the engine feeds them (64-token chunks through `prefill`, the rest through `main`) | **top-1 2639 / 2682 = 98.40 %** (bar ≥ 93 %); through `main` alone 2637 / 2682 (1027 / 1035 on the short sequences, 1610 / 1647 on the document) |
| the chunked prefill | the same bundle, chunked vs `main` alone | the logits at the 49 chunk boundaries agree to cos ≥ 0.9999922; `main` alone equals the `main`-only build at all 2,683 positions |
| 8 short questions, thinking off | greedy; right = the reply holds the expected answer (42, Tokyo, cold, 7, merci, 56, 0.9, blue) | 8 / 8, the verdicts of transformers bf16; 7 of 8 replies word for word |
| token gate | `conversion/coreai_gate.py`, 16 greedy tokens vs the fp32 oracle | 16 / 16 (alphabet); 16 / 16 on a 215-token prompt that rides three chunks; 8 / 8 then `<\|im_end\|>` (17 + 25, the `main`-only build) |
| stop token | "Reply with only the number: 1+1=?", thinking off | stops on `<\|im_end\|>` after 2 tokens |
| thinking on | the 8 questions with thinking on, greedy, a 3,584-token budget, 3 runs of fresh processes, on the `main`-only build (these prompts never reach `prefill`, and `main` gives the same values) | every run 8 / 8: `</think>` closes after 40–237 tokens of reasoning (median 56) and the answer after it is right; the three runs answer alike |
| a tool call | one tool returning a fresh random value per run, through CoreAIKit's `KitLanguageModel` and the zoo's `ZooLanguageModel` behind Apple's `LanguageModelSession`, 3 runs each | 6 / 6: the XML call parsed into `lookup_secret(key: "alpha")`, run by the framework, the value in the answer verbatim, the reasoning kept out of it |
| the published bytes | the `.aimodel` re-saved without its source locations (a new `main.hash`), the checks above run again on it | teacher-forced values equal to the export's at all 2,683 positions, chunked and through `main` alone; 8 / 8; 16 / 16 (alphabet and the 215-token prompt); stops after 2 tokens; the tool call 6 / 6; CoreAIKit's catalog path answers with thinking on and off and runs the tool |

## Use it

Needs a CoreAIKit build that supports LFM3 (0.7.3 does not). The catalog entry is for macOS: this release has
no iOS bundle.

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

Run 3 times through each provider with a fresh value the model cannot guess, and once more through the catalog
as above: every run called `lookup_secret` with `key: "alpha"` and answered with the value verbatim, the
reasoning in its own transcript entry.

## Known limits

- **One conversation per state.** The conv state is not indexed by position, so the engine refuses to rewind
  into the middle of a conversation; a new conversation starts from zero state. `ChatSession` and the
  FoundationModels providers re-render the history every turn and, when the engine refuses the rewind, prefill
  it again from zero state — a three-turn chat recalled its first turn through both `ChatSession` and the zoo's
  provider. Each turn costs a full prefill of the conversation.
- **The tail of a prompt goes one token at a time.** The engine keeps the last prompt token for `main`, so a
  prompt's final 1–64 tokens run through `main` at about the decode speed; a 1,024-token prompt is 15 chunks and
  then 64 single steps, about half of its first-token time.
- **4,096 tokens.** The bundle is exported with a 4,096-token context; the checkpoint itself allows more.
- **No iOS bundle in this release.** On an iPhone 17 Pro (iOS 27.2; a 768-token prompt, 256 generated tokens),
  these weights built with `main` only and compiled ahead of time for its GPU gave the Mac's tokens on 8 questions
  but decoded at 9.25 tok/s with the phone at thermal state serious; a static Neural Engine export (not in this
  release) decoded at 32.86 tok/s, but thinking, tool calls and CoreAIKit were not checked on it. On an iPhone 18 Pro
  the `main`-only `.aimodel`, specializing on the phone (device JIT), had not built its engine 300 s after launch
  (the cause is not isolated). On iOS the pipelined engine caps prompt + answer at 1,024 tokens — tight for a
  model that thinks first. Details: [the port notes](../../knowledge/lfm3-1.7b-port.md#iphone).
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

`recipe.toml` holds the export: `python3 conversion/zoo_convert.py run --force lfm3-1.7b` runs its three steps
(the chat template, the export, the strip). `conversion/export_lfm3_decode_pipelined.py` is the exporter and
`conversion/coreai_gate.py --arch lfm3` the token gate.

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
by section except for those names and the lengths that encode them, with the same 3,971 ops). Running the
bundle needs the pipelined engine with one extra state and `COREAI_CHUNK_THRESHOLD=1`
(`main` takes one token; the engine feeds `prefill` whatever the threshold); `llm-runner` takes
`--inference-engine-variant coreai-pipelined --warmup off`.

## License

The weights are under the **LFM Open License v1.0**
([LICENSE](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE)): Apache-2.0-style grants, with
§5: commercial use by a legal entity at or above US$10 M annual revenue is not licensed (the threshold does
not apply to a qualified non-profit's non-commercial or research use). The bundle on Hugging Face ships that
LICENSE file. The conversion code in this repo stays BSD-3-Clause. Not affiliated
with Apple or LiquidAI.
