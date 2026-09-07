# Nanbeige4.2-3B (text decoder) — Core AI

[`Nanbeige/Nanbeige4.2-3B`](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) is a recurrent Llama decoder:
**22 physical transformer blocks execute twice**, with an RMS norm after each pass and separate KV history.
The Core AI model therefore executes 44 layer passes and uses **44 KV-cache layers**, while storing and
quantizing only one copy of the 22 physical blocks.

Source checkpoint: `5ff54fb7ed86ce8e216d78bff5417ab9981de3d4` (Apache-2.0).
Ported by [Vadim Smirnov (@ukint-vs)](https://github.com/ukint-vs); tracked in
[`john-rocky/coreai-model-zoo#5`](https://github.com/john-rocky/coreai-model-zoo/issues/5).

**⬇️ Verified int8 LanguageBundle:
[ukint-vs/Nanbeige4.2-3B-CoreAI](https://huggingface.co/ukint-vs/Nanbeige4.2-3B-CoreAI/tree/5864ec7a5581940958e58354a6b6c46c8f06891e)**

- immutable bundle revision: `5864ec7a5581940958e58354a6b6c46c8f06891e`
- path: `gpu-pipelined/nanbeige4_2_3b_decode_int8hu_block32_sym_s1`
- format: `int8hu --head-sym --static-ids`, complete tokenizer included
- producer: `coreai-core 1.0.0b2`

## Runtime contract

The bundle uses the existing `coreai-pipelined` static-S=1 GPU path:

```text
input_ids, position_ids, k_cache, v_cache → logits
```

K/V states have shape `[44, 1, 8, max_context, 128]`. Pass 0 uses cache slots 0–21 and pass 1 uses 22–43.
Quantization visits 111 physical linear modules; recurrent execution does not duplicate serialized weights.
The bundled vendor chat template supports both `enable_thinking=true` and `enable_thinking=false`.

## Verified

| Check | Result |
|---|---|
| Bundle | **4.59 GiB** (`4,815,288 KiB`), int8 |
| Float32 parity | Full and cached logits pass at `rtol=1e-4`, `atol=1e-4`; identical 32-token greedy continuation |
| Int8 authoring gate | Prompt top-1 8/8, greedy 32/32, cosine 0.9997768 |
| Int8 engine gate | Token-exact vs fp32 on two factual prompts and the `9.11` versus `9.8` reasoning smoke; deterministic rerun |
| M4 Max Release, prompt 128 / generation 256 / 3 runs | **47.37 prefill / 46.35 decode tok/s** |
| Maximum context tested | **4,096 tokens**: 29.83 prefill / 32.80 decode tok/s, 9.17 GiB peak RSS, zero swaps |
| iOS 27 GPU AOT `h18p` | Compile pass with Xcode 27; generated package source hash matches the published model |
| iPhone 17 Pro `h18p` hardware | **PASS (2026-07-24)**: nat 24/24 + oracle 24/24 greedy tokens identical to the M4 Max engine reference (itself token-exact vs the fp32 oracle), reproduced on a second full run; cold-specialization load 31.7 s (warm 10.8 s), no jetsam |
| iPhone 17 Pro speed (p=128 g=256, S=1 chunking, ×2 trials) | first run 6.9 prefill / 5.7 decode → settled **8.5 prefill / 6.4 decode tok/s** — the two-pass loop reads the 22 shared blocks twice per token (~2× weight traffic), so a bandwidth-bound phone lands near half a same-size single-pass model; Mac-class GPUs are this model's sweet spot |

The Mac benchmark used macOS 27.0, Xcode 27 beta 4, Core AI runtime `aff0bb2`, AC power, and High Power Mode.
The checkpoint config SHA-256 is
`f6cb15b22847664f3a6049dc4b58fdd10f1650d112ac99a1da3d051f17c2ca19`.

Int4 and mixed int4/int8 candidates are not published because they failed the same multi-token quality gates
required of int8. Detailed architecture, parity, quantization, and optimization results are in the
[support report](../../knowledge/nanbeige4.2-coreai-support.md).

## Convert and verify

From a checkout at the commit pinned in `conversion/overlay/BASE`, apply the Python overlay, then run:

```sh
python3 ../coreai-model-zoo/conversion/zoo_convert.py run nanbeige4.2-3b --dry-run
python3 ../coreai-model-zoo/conversion/export_nanbeige41_decode_pipelined.py \
  int8hu --head-sym --static-ids \
  --hf-id Nanbeige/Nanbeige4.2-3B \
  --revision 5ff54fb7ed86ce8e216d78bff5417ab9981de3d4

python3 ../coreai-model-zoo/conversion/coreai_gate.py \
  exports/nanbeige4_2_3b_decode_int8hu_block32_sym_s1 \
  Nanbeige/Nanbeige4.2-3B \
  --revision 5ff54fb7ed86ce8e216d78bff5417ab9981de3d4 \
  --arch nanbeige -n 24
```

Reproduce the isolated official-checkpoint parity gate with:

```sh
python3 ../coreai-model-zoo/_smoke/verify_nanbeige42_checkpoint.py \
  --official-python /path/to/nanbeige-oracle/bin/python
```

The advertised 262K context is not claimed; the published bundle is verified to 4K.

<!-- gen-cards:use-it begin id=nanbeige4.2-3b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

▶️ **Run it (source)** — the [ChatDemo runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo)
(GUI + CLI, one app for every chat model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/ChatDemo/ChatDemo.xcodeproj
# → Run, then pick "Nanbeige4.2 3B" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/ChatDemo
swift run chat-cli --model nanbeige4.2-3b --prompt "What can you do, offline?"
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let chat = try await ChatSession(catalog: "nanbeige4.2-3b")
let reply = try await chat.respond(to: prompt)
// reply: the answer, generated fully on-device
```

The take-home is [`Examples/ChatDemo/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/main/Examples/ChatDemo/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; the CLI is an argument shell over it, and
the GUI drives the same `ChatSession` across turns for its transcript.
Multi-turn? Hold the `ChatSession` and call `respond(to:)` per turn — it keeps the
conversation history; `streamResponse(to:)` yields tokens as they decode.

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` → product **CoreAIKit**
- Info.plist: none needed
- Entitlements (iOS): `com.apple.developer.kernel.increased-memory-limit`
- First run downloads the model — 4.7 GB (Mac) / 4.7 GB (iPhone) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## License

Model weights: **Apache-2.0** (Nanbeige LLM Lab). Conversion code: BSD-3-Clause.
