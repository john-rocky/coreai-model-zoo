# LFM3-1.7B — Core AI

LiquidAI's [LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) on Apple's Core AI: a 1.7B model that
reasons before it answers and calls tools in XML, as one int8 `.aimodel` on the `coreai-pipelined` GPU engine
with a 4,096-token context, driven from CoreAIKit's `ChatSession` or Apple's `LanguageModelSession`.

Source checkpoint: revision `ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7` (LFM Open License v1.0).

**⬇️ Converted bundle:
[mlboydaisuke/LFM3-1.7B-CoreAI](https://huggingface.co/mlboydaisuke/LFM3-1.7B-CoreAI)** —
`gpu-pipelined/lfm3_1_7b_decode_int8hu_block32_sym/` (2.72 GB: the `.aimodel`, `metadata.json` and the
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

The bundle is `int8hu`: the MLP and conv-mixer linears int8 per-block-32, the head untied and int8
per-block-32 symmetric, the attention projections (q, k, v, o and the gate) and the q / k norms in fp32,
everything else fp16.

## Speed

<!-- r2 の数字待ち: the Mac row = the speed ladder's ship row (llm-benchmark, prefill 1024 / decode 256, ctx 2048) once r2 closes; iPhone stays "not measured" until a device run -->
Mac: not published yet. iPhone: not measured.

## Accuracy — what was checked

Against transformers 5.20.2 on the same weights; Mac (M4 Max, macOS 27.0). Transcripts are in this
directory.

| check | how | result |
|---|---|---|
| the re-authored module | PyTorch fp32 vs transformers fp32 on 17 sequences (8 questions with thinking off and on, a 1,648-token document): one call per sequence, S=1, and the engine's own walk | max\|Δlogit\| ≤ 2.61e-3, top-1 2682 / 2682 |
| the graph in fp32 | the fp32 export on the GPU, teacher-forced, the same 17 | 2.20e-3, top-1 2682 / 2682 |
| the graph in fp16 | the same | max 0.353, top-1 2676 / 2682 — under the strict bar; the module run in fp16 in PyTorch shows the same width, so it is the fp16 activations, not the conversion |
| **this bundle** | the same, vs transformers fp32 | **top-1 2637 / 2682 = 98.32 %** (bar ≥ 93 %): 1027 / 1035 on the short sequences, 1610 / 1647 on the document |
| 8 short questions, thinking off | greedy; right = the reply holds the expected answer (42, Tokyo, cold, 7, merci, 56, 0.9, blue) | 8 / 8, the verdicts of transformers bf16; 7 of 8 replies word for word |
| token gate | `conversion/coreai_gate.py`, 16 greedy tokens vs the fp32 oracle | 16 / 16 (alphabet); 8 / 8 then `<\|im_end\|>` (17 + 25) |
| stop token | "Reply with only the number: 1+1=?", thinking off | stops on `<\|im_end\|>` after 2 tokens |
| thinking on | the 8 questions with thinking on, greedy, a 3,584-token budget, 3 runs of fresh processes | every run 8 / 8: `</think>` closes after 40–237 tokens of reasoning (median 56) and the answer after it is right; the three runs answer alike |
| a tool call | one tool returning a fresh random value per run, through CoreAIKit's `KitLanguageModel` and the zoo's `ZooLanguageModel` behind Apple's `LanguageModelSession`, 3 runs each | 6 / 6: the XML call parsed into `lookup_secret(key: "alpha")`, run by the framework, the value in the answer verbatim, the reasoning kept out of it |
| the published bytes | the `.aimodel` re-saved without its source locations (a new `main.hash`), the checks above run again on it | teacher-forced values equal to the export's at all 2,683 positions (top-1 2637 / 2682); 8 / 8; 16 / 16 and 8 / 8; stops after 2 tokens |

## Use it

Needs a CoreAIKit release whose catalog lists `lfm3-1.7b` (0.7.3 does not). Mac only until an iPhone run has
been measured.

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
    model: model, tools: [LookupSecretTool(secret: "ABGX-LQLE")], instructions: "You are a helpful assistant.")
let answer = try await session.respond(
    to: "Use the lookup_secret tool to read the value stored under the key \"alpha\", then tell me that value.",
    options: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 2048))
// answer.content: The value stored under the key "alpha" is **ABGX-LQLE**.
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
- **4,096 tokens.** The bundle is exported with a 4,096-token context; the checkpoint itself allows more.
- **iPhone: not measured, no iOS variant.** On iOS the pipelined engine caps the dynamic KV at 1,024 tokens,
  so a turn there is prompt + answer ≤ 1,024 — tight for a model that thinks first.
- **fp16 activations.** Teacher-forced over a 1,648-token document, the fp16 graph changes the top-1 at 6
  positions where transformers' own top-2 gap is 0.003–0.061, which the fp32 graph keeps; the int8 bundle's
  98.32 % includes that floor.
- **The chat template is rewritten, not changed.** The vendor template does not parse in swift-jinja 2.3.2
  (the pin of `apple/coreai-models`) and renders tool definitions differently in 2.3.6; the bundle's template
  has eight rewrites that render the same string as the vendor's under transformers. A Core AI stack that
  cannot parse a template falls back to the raw prompt without saying so — check the rendered prompt when you
  swap the tokenizer.

## Reproduce

`recipe.toml` holds the export; `conversion/export_lfm3_decode_pipelined.py` is the exporter and
`conversion/coreai_gate.py --arch lfm3` the token gate.

```bash
coreai-models/.venv/bin/python conversion/export_lfm3_decode_pipelined.py int8hu \
    --head-quant block32 --head-sym \
    --src LiquidAI/LFM3-1.7B --revision ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7 \
    --hf-id LiquidAI/LFM3-1.7B --max-ctx 4096 --chat-template <the bundle's chat_template.jinja>
python3 conversion/coreai_gate.py exports/lfm3_1_7b_decode_int8hu_block32_sym LiquidAI/LFM3-1.7B \
    --revision ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7 --arch lfm3 -n 16
```

The export needs the LFM3 authoring module (`coreai_models/models/macos/lfm3.py`) on the `coreai-models`
checkout. The exported `main.mlirb` records the export machine's source paths; the published `.aimodel` is the
export re-saved after coreai-torch's `strip_debug_info` (`conversion/d1/strip_bundle.py` does this for a
bundle whose `assets.main` is the `.aimodel`), which is a new `main.hash`, so the gates above were run again
on it. Running the bundle needs the pipelined engine with one extra state and `COREAI_CHUNK_THRESHOLD=1` (S=1
prefill); `llm-runner` takes `--inference-engine-variant coreai-pipelined --warmup off`.

## License

The weights are under the **LFM Open License v1.0**
([LICENSE](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE)): Apache-2.0-style grants, with
§5: commercial use by a legal entity at or above US$10 M annual revenue is not licensed (the threshold does
not apply to a qualified non-profit's non-commercial or research use). The bundle on Hugging Face ships that
LICENSE file. The conversion code in this repo stays BSD-3-Clause. Not affiliated
with Apple or LiquidAI.
