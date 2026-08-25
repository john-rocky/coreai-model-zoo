# S1-mini by Superwhisper — Core AI

[🤗 mlboydaisuke/S1-mini-CoreAI](https://huggingface.co/mlboydaisuke/S1-mini-CoreAI) · iPhone-verified · Apache-2.0 **+ naming clause** · base [superwhisper/s1-mini](https://huggingface.co/superwhisper/s1-mini)

**S1-mini** by **Superwhisper** is a 0.6B text normalizer for speech-to-text output: it takes a
raw ASR transcript and rewrites it as clean written text — fillers removed, false starts and
self-corrections resolved to whatever the speaker landed on, punctuation and capitalization
applied, and spoken numbers, dates, times, currency and email addresses rendered in written
form. It is not a chat model; it does one job, steered by a control line at the top of the input.

That makes it the piece the zoo's ASR models did not have. Parakeet, Parakeet-v2 and
Nemotron-3.5-ASR-Streaming all produce raw transcripts; **S1-mini** is the on-device
post-processor that turns one into text a person would actually send — the whole
dictation path, microphone to finished sentence, with nothing leaving the device.

> **Naming.** The upstream license adds a term to Apache-2.0: any use, distribution or
> product integration must keep identifying this model as **"S1-mini" by "Superwhisper"**,
> with that exact capitalization, whatever the surrounding product is called.

<!-- gen-cards:use-it begin id=s1-mini (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — this model is the default behind the kit's task op
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let clean = try await CoreAI.tidyTranscript(rawTranscript)
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [Tidy runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Tidy)
(GUI + CLI, the three control axes as pickers):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/Tidy/Tidy.xcodeproj
# → Run, then pick "S1-mini by Superwhisper" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/Tidy
swift run tidy-cli --model s1-mini --text "so um i need to like send the the report by uh friday no wait make that thursday"
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let tidier = try await KitTextNormalizer(catalog: "s1-mini")
// Long input is cut at word boundaries into ~450-token chunks and the rewrites stitched:
// on iPhone the engine caps prompt + generated at 1024 tokens, so a whole meeting
// transcript passed in one call would stop mid-sentence.
let result = try await tidier.normalize(transcript)
// result: the transcript as written text — English only; filler-only input returns ""
```

The take-home is [`Examples/Tidy/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/main/Examples/Tidy/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; both the runner's GUI and its CLI call it.
Cleaning transcripts repeatedly? Keep the `KitTextNormalizer` loaded and call
`normalize(_:)` per transcript — the 796 MB load is what you are avoiding.

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` → product **CoreAIKit**
- Info.plist: none needed
- Entitlements: none needed
- First run downloads the model — 0.8 GB (Mac) / 0.8 GB (iPhone) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Bundles

Measured on an **M4 Max Mac Studio (128 GB, macOS 27.0 / 26A5416b)**, `llm-benchmark`,
128-token prompt / 256 generated, n=3, default chunking.

| bundle | decode | prefill | conversion gate | task gate | size |
|---|---:|---:|---|---|---:|
| **`int8lin/`** (ship) | **268.4 tok/s** | **4161 tok/s** | **16/16 token-exact vs fp32 oracle** | **13/14** vs the released weights | **759 MB** |
| `int4lin/` (**no-go**) | 296.0 tok/s | 5558 tok/s | 16/16 token-exact vs fp32 oracle | 10/14 — **corrupts digits** | 549 MB |

### On-device (iPhone 17 Pro, A19 Pro — `PipelinedBench`, Release, greedy)

| | decode | prefill | numerics | load |
|---|---:|---:|---|---:|
| **`int8lin/`** | **62.4 tok/s** | **69.0 tok/s** | **276/276 + 27/27 token-exact vs the Mac engine** | 0.2–1.0 s |

`device == Mac == fp32 HF`. The 276-token case is a real 323-token dictation transcript and
its rewrite, not a synthetic continuation; the 27-token case is the currency/date one int4
corrupted. Both include EOS, so the halt is gated too. No AOT needed — the 759 MB JIT
`.aimodel` loads directly. The mandatory dynamic-KV `PB_G=1024` run passes.

Cold numbers, the protocol every other card here uses, reproduced across two runs
(69.0/62.4 and 71.3/64.7). **Sustained, expect about half**: after back-to-back 1024-token
generations the same bundle measured 34.9 prefill / 30.5 decode, and seven minutes idle
restored it — thermal, not a regression. A dictation post-processor runs repeatedly, so
that is the number to plan capacity against.

### iPhone ceiling: prompt + generated must stay under 1024 tokens

A 611-token transcript whose Mac rewrite runs 603 tokens (1214 total) produced **413 tokens
on device, every one token-identical to the Mac**, and then stopped at absolute position
exactly **1024**. Truncation, not corruption.

The cause is shipped engine behaviour, not a bench artifact — `CoreAIPipelinedEngine.swift`
carries `#if os(iOS)` + `GrowingKVCache` → `iosDynamicKVCapacityCap = 1024`, capping
`maxTokens` to `1024 - processed - prompt.count` and throwing `contextLengthExceeded` when a
prompt leaves no budget. The arithmetic matches the measurement exactly: `1024 − 611 = 413`.
It guards the iOS compiler's miscompilation of growing-KV specializations at seq ≥ 2048
(apple/coreai-models#124), so truncating is the guard working.

**Practical rule.** The upstream card recommends inputs up to ~1000 tokens. On iPhone a
1000-token prompt leaves almost no room for the rewrite, and a ≥1024-token prompt throws.
**Chunk to roughly ≤450–500 input tokens** so prompt + rewrite clears the cap. On Mac there
is no such cap; the 4096 export is clean at every shape.

Evidence: [`device/gate.json`](device/gate.json) — every run's verdict lines, plus the Mac and device token sequences for the case above, so the exact-prefix claim and the 1024 arithmetic are both re-checkable without the device.

### int4 is a measured no-go, and the continuation gate could not see it

Both bundles pass the standard oracle gate token-for-token. int4 still breaks the model's
actual job. Over ten number-heavy transcripts against the bf16 reference:

| input | int8lin | int4lin |
|---|---|---|
| `…twenty three thousand four hundred and fifty dollars…` | `$23,450` ✅ | **`$2,345`** ❌ |
| `we sold one hundred and seven units…` | `107` ✅ | **`177`** ❌ |
| `let's meet at half past two tomorrow uh actually make it three fifteen p m` | `3:15pm tomorrow` ✅ | `3:15pm` (drops *tomorrow*) ❌ |

int8lin reproduced 9/10 of that set exactly (its one miss is punctuation: `$23,450 and` for
`$23,450, and`); int4lin reproduced 7/10, and two of the three misses are wrong digits (the
third, `$1,999`/`$2,150` → `$19.99`/`$21.50`, is arguably the *better* reading and is not
counted against it). For a model whose job includes inverse text normalization of money and
counts, a wrong digit is not a style difference. **int8lin carries the quality claim.**
Full probe: [`int4-number-probe.json`](int4-number-probe.json).

The lesson generalizes past this port: a free-run continuation gate ("the alphabet begins A,
B, C…") measures conversion fidelity and nothing about task capability. Both bundles scored
16/16 on it. The damage only appears when the gate speaks the model's own input format —
which is why this port ships a second gate that does.

## Parity

- **Conversion** — `conversion/coreai_gate.py`, greedy, 16 tokens, vs the fp32 eager oracle:
  **PASS token-for-token** for int8lin and int4lin. Transcripts:
  [`gate-s1-mini-int8lin.json`](gate-s1-mini-int8lin.json), [`gate-s1-mini-int4lin.json`](gate-s1-mini-int4lin.json).
  The same gate on a deliberately truncated 4-layer build returns FAIL at token 0 (margin
  0.938), so the green is a result and not a gate that cannot go red.
- **Task** — `_smoke/gate_s1_mini_task.py`, 14 cases across the card's three control axes
  (`Styling` ×4, `Structure` ×2, `Context` ×2, plus the six-row Examples table and the
  filler-only → empty-string case), greedy, `enable_thinking=False`.
  Verdict is against the **released weights** run through `transformers`, not against the
  card's printed strings — see below. Records: [`task-gate-int8lin.json`](task-gate-int8lin.json)
  (13/14), [`task-gate-int4lin.json`](task-gate-int4lin.json) (10/14).

### The model card's example outputs do not fully reproduce

Measured 2026-08-25: the released `superwhisper/s1-mini` weights reproduce **9 of the 14**
strings printed on the upstream card, under a prompt verified byte-identical to the literal
one the card documents. The misses are stylistic — a retained `So`/`Hmm` opener,
`March 3rd` where the card says `March 3`, and the card's own `Structure: lists` example
staying prose. The card's table appears to predate the released checkpoint. Gating on it
would fail every faithful conversion, so it is recorded as an informational column and the
verdict runs against the weights that actually shipped.

## Conversion

- **Stock `qwen3` graph, no new model code.** S1-mini is a Qwen3-0.6B finetune and its config
  is byte-identical in shape to `Qwen/Qwen3-0.6B` — 28 layers, hidden 1024, GQA 16q/8kv
  head_dim 128, SwiGLU 3072, QK-norm, RoPE θ=1e6, no bias. It rides
  `coreai_models.models.macos.qwen3` unchanged (that class already fuses q/k/v and q_norm/k_norm
  in `_mutate_state_dict`) on the plain KV-only decode path.
- **No `*hu` head-quant mode, on purpose.** The head is **tied** to the 151936×1024 embedding.
  Quantizing it means untying it first, and the eager quantizer skips shared params — so
  untying *adds* a tensor rather than shrinking one: tied fp16 embed+head is 311 MB, while
  fp16 embed + int8 untied head is 311 + 156 = 467 MB. Untying is a pure loss at every bit
  width. The embedding stays fp16 by the standard recipe, so on int4 it is ~59% of the
  bundle; shrinking that is an embedding-quantization question, not a head one.
- **Dynamic input_ids, not `--static-ids`.** The loop-free `[1,1]` pattern exists to kill
  per-step respecialization on big bundles. At 0.6B the respecialization is cheap and the
  workload is prefill-heavy — the input is a whole transcript — so batched prefill (4161 vs
  344 tok/s at `COREAI_CHUNK_THRESHOLD=1`) is worth far more than it costs. Revisit only
  with a device measurement in hand.

## Prompt format

The system prompt and the control line are part of the trained input format, and
`enable_thinking=False` is mandatory:

```
<|im_start|>system
You are a text normalizer for speech-to-text transcripts. The input begins with a control line specifying the styling, structure, and context settings; clean the transcript to match those settings and output only the cleaned text.<|im_end|>
<|im_start|>user
[Styling: semi-formal] [Structure: prose] [Context: general]
<raw transcript><|im_end|>
<|im_start|>assistant
<think>

</think>

```

`Styling` ∈ {casual, semi-casual, semi-formal, formal}, `Structure` ∈ {prose, lists},
`Context` ∈ {general, email}. Leave thinking on and the model emits an empty `<think>` block
and stops — every case returns the empty string, which looks like a working pipeline
producing nothing.

## Reproduce

```bash
cd ~/code/coreai/coreai-models && .venv/bin/python \
  ../coreai-models-community/conversion/export_s1_mini_decode_pipelined.py int8lin

python3 ../coreai-models-community/conversion/coreai_gate.py \
  exports/s1_mini_decode_int8lin superwhisper/s1-mini --arch qwen3 -n 16

python3 ../coreai-models-community/_smoke/gate_s1_mini_task.py exports/s1_mini_decode_int8lin
```

See [`knowledge/s1-mini-port.md`](../../knowledge/s1-mini-port.md) for the port notes.
