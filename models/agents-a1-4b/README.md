# Agents-A1-4B — Apple Core AI (`.aimodel`)

The [source card](https://huggingface.co/InternScience/Agents-A1-4B) describes the model as “Agents‑A1 is a long-horizon agentic model from InternScience”. Its declared license is `apache-2.0`.
This bundle converts the **TEXT tower** of the `Qwen3_5ForConditionalGeneration` checkpoint. Its `text_config` equals Qwen/Qwen3.5-4B, so the zoo's 4B recipe applies unchanged with an HF-id swap.

> [!NOTE]
> Exported with `coreai-core 1.0.0b2` for the Mac-class `coreai-pipelined` GPU engine on macOS 27. The canonical bundle path is `gpu-pipelined-b2/agents_a1_4b_decode_int8hu_block32_sym/`.

## Bundles

- **`gpu-pipelined-b2/agents_a1_4b_decode_int8hu_block32_sym/`**: **5,770,816,352 bytes** (5.771 GB, decimal), 11 files.
- `int8hu --head-sym`: transformer int8 per block of 32, with an untied vocabulary head quantized using per-block-32 absmax int8.
- Full LanguageBundle: `.aimodel`, `metadata.json`, and `tokenizer/`. Single-step input shape `[1,1]`; vocabulary 248320; exported maximum context 4096 tokens.

## Measured

Apple M4 Max GPU, macOS 27.0, Release tools. Correctness uses greedy `coreai-pipelined`, `COREAI_CHUNK_THRESHOLD=1`, and warmup off; the reference oracle runs fp32 on CPU.

| Check | Result |
|---|---|
| France raw prompt: `The capital of France is` | 16/16 tokens exact against fp32 CPU ([transcript](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/agents-a1-4b/gate-agents-a1-4b.json)) |
| Alphabet raw prompt: `The alphabet begins A, B, C, D, E, F,` | 16/16 tokens exact against fp32 CPU ([transcript](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/agents-a1-4b/gate-agents-a1-4b-alphabet.json)) |
| Chat no-think, `1+1=?` | `1 + 1 = 2`; 8/8 IDs exact including EOS 248046 (the [transcript](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/agents-a1-4b/gate-agents-a1-4b-chat-nothink.json) records the 7 tokens before EOS) |
| Tool calls, no-think | 3/3 in the template's XML format: `get_weather(Tokyo, celsius)`, `convert_currency(250, USD, EUR)`, `search_web("Swift 6.3 release notes")`; stopped at 40 / 55 / 31 tokens, with nothing after each closing tag |
| Swift-side template | Template applied without error; `</think>` followed by `Canberra`; stopped at 201 tokens |
| Eight fixed sanity questions, no-think | 8/8 expected-substring hits and 8/8 EOS stops |
| First / cached load (llm-runner) | 3.377 s / 0.807 s |

Decode speed, `llm-benchmark` Release, `-p 128 -g 256 -n 3`, no other GPU job on the machine (M4 Max, macOS 27.0 26A428):

| Trial | Prompt tok/s | Generation tok/s |
|---|---:|---:|
| 1 | 82.1 | 72.3 |
| 2 | 82.9 | 72.3 |
| 3 | 82.9 | 72.4 |
| Mean | 82.6 | 72.3 |

Prompt tok/s is the pipelined S=1 prefill (`COREAI_CHUNK_THRESHOLD=1`), which is why it sits near the decode rate.

## Run (macOS)

Use the [zoo's engine patch stack](https://github.com/john-rocky/coreai-model-zoo): `apps/coreai-shared-product.patch` followed by `apps/coreai-pipelined-extra-states.patch`. From the downloaded `gpu-pipelined-b2/` directory, run a Release build:

```bash
COREAI_CHUNK_THRESHOLD=1 llm-benchmark --model agents_a1_4b_decode_int8hu_block32_sym -p 128 -g 256 -n 3
```

- Set `COREAI_CHUNK_THRESHOLD=1` **before engine creation** so prefill uses pipelined single-token steps.
- **Never call `engine.warmup()`**: it requests a query shape the static `[1,1]` graph rejects. A one-token generation after load can warm the engine.
- Use **Release** builds for timing.

The original chat template is preserved:

- Without a system message, it injects an **Intern-A1 deep research assistant** persona and the hard-coded **`Current date: 2026-07-14`**. An app should pass its own system message with the real date.
- Thinking is **on by default**. `enable_thinking=False` renders the no-think prefix.
- Tool calls use an XML `<function=…>` block, **not JSON**, inside `<tool_call>`. The required shape is:

```text
<tool_call>
<function=…>
<parameter=…>…</parameter>
</function>
</tool_call>
```

- The chat stop token is **`<|im_end|>` (248046)**. The `text_config.eos_token_id` value **248044** is not the chat stop.

## iPhone

No iPhone bundle is provided. The 4B-class graph exceeds on-device GPU specialization and needs ahead-of-time (h18p) compilation.

## Reproduce

The [unchanged exporter](https://github.com/john-rocky/coreai-model-zoo/blob/main/conversion/export_qwen3_5_decode_pipelined.py) and [recipe](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/agents-a1-4b/recipe.toml) specify the conversion:

```bash
python3 conversion/zoo_convert.py show agents-a1-4b
python3 conversion/export_qwen3_5_decode_pipelined.py int8hu --head-sym --hf-id InternScience/Agents-A1-4B
```

Gate transcripts: [gate-agents-a1-4b.json](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/agents-a1-4b/gate-agents-a1-4b.json), [gate-agents-a1-4b-alphabet.json](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/agents-a1-4b/gate-agents-a1-4b-alphabet.json), and [gate-agents-a1-4b-chat-nothink.json](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/agents-a1-4b/gate-agents-a1-4b-chat-nothink.json). Runtime and quantization details are in the zoo's [pipelined-engine notes](https://github.com/john-rocky/coreai-model-zoo/blob/main/knowledge/pipelined-engine.md).

---

**⬇️ Download:** [🤗 mlboydaisuke/Agents-A1-4B-CoreAI](https://huggingface.co/mlboydaisuke/Agents-A1-4B-CoreAI) — this card and the
model page are the same document. Reproduce it: `python3 conversion/zoo_convert.py show agents-a1-4b` prints the
command and its prerequisites; [`recipe.toml`](recipe.toml) is the record. Conversion and gates were run by
Codex (gpt-6-astra) under supervision; the transcripts above are its evidence.
