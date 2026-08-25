# Nemotron-3-Nano 4B (text decoder) — Core AI

Mamba2 + attention + MLP hybrid decoder (NVIDIA): the 4B's `hybrid_override_pattern`
`M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-` gives 42 blocks = **21 Mamba2 mixers**
(selective-scan SSM: 96 heads × d_head 80, d_state 128, 8 groups, kernel-4 depthwise conv,
**grouped** gated RMSNorm) + **17 dense MLPs** (`up → relu² → down`, no gate branch) +
**4 GQA attention layers** (40 q / 8 kv heads, head_dim 128, **NoPE**, no q/k norm, no biases),
hidden 3136, vocab 131 072, **untied head**. One mixer per block — a mamba block carries no
second MLP branch, unlike Granite 4.0-H. Source: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`.

**⬇️ Converted `.aimodel` bundle:
[mlboydaisuke/Nemotron-3-Nano-4B-CoreAI](https://huggingface.co/mlboydaisuke/Nemotron-3-Nano-4B-CoreAI)** —
`gpu-pipelined/` (Mac, JIT) + `ios-h18p/` (iPhone, AOT), both int8hu, tokenizer included.

**The zoo's second SSM-scan architecture, and the first Mamba2 that isn't Granite.** Same
enabler: **at S=1 the Mamba2 selective scan is a single recurrence step** (`state = state*dA +
dt*B*x; y = (state·C) + D*x` — the HF `use_precomputed_states` branch), so the decode-only
graph is loop-free and lowers on the MPSGraph GPU delegate. State = growing KV for the 4
attention layers + two fixed-shape stacks (conv columns `[21,1,9728,3]`, SSM state
`[21,1,96,80,128]` = 41 MB) — the same `(convState, recState)` shape-class as granite4h and
qwen3.5, inside the extra-states patch budget (≤2). **No custom Metal kernel** — but not because
one loses. A hand-written fused scan measures **3–8% faster** than the stock graph on
granite-4.0-h-350m (paired A/B, both arms in one process, 8 interleaved reps, median 1.03–1.08x).
That margin does not pay for the fusion barrier a `metal4_kernel` op puts in the graph, its grid
and shape constraints, and a second artifact to maintain. At S=1 the scan is a single recurrence
step, so the plain torch graph is already the right answer.

(Two earlier versions of this card said the kernel measured *slower* — first 1.24x, then, after a
"correction", that a fp32 cast at the kernel boundary explained it. Both were artifacts of unpaired
single-shot timings on a machine that drifts 10–15% run to run: the same configuration read anywhere
from 0.89x to 1.19x. The lesson worth keeping is the protocol, not the kernel: pair the arms in one
process, interleave, repeat, report the median and the spread.)

A 4B graph cannot specialize on-device, so the iPhone bundle is **AOT-compiled** for `h18p`
(the FastContext lesson). `CoreAIShared.ModelBundle` reads `metadata.json` **at the model dir**,
so after `coreai-build compile` you must rewrite `assets.main` to `<name>.h18p.aimodelc`.

<!-- gen-cards:use-it begin id=nemotron-3-nano-4b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — run the kit's task op on this model
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let tldr = try await CoreAI.summarize(text, options: .model("nemotron-3-nano-4b"))
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [ChatDemo runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo)
(GUI + CLI, one app for every chat model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/ChatDemo/ChatDemo.xcodeproj
# → Run, then pick "Nemotron-3-Nano 4B" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/ChatDemo
swift run chat-cli --model nemotron-3-nano-4b --prompt "What can you do, offline?"
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let chat = try await ChatSession(catalog: "nemotron-3-nano-4b")
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
- Entitlements: iOS needs `com.apple.developer.kernel.increased-memory-limit` — a 4.3 GB bundle is past the default jetsam limit
- First run downloads the model — 4.6 GB (Mac) / 4.6 GB (iPhone) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Measured

Numerics, port vs `transformers` fp32 (`nemotron_parity.py`): per-step logits `rel 2e-7 … 4e-7`
over the prompt, and 8 greedy tokens **token-identical**. The exported int8hu bundle reproduces
the fp32 oracle's top-1 on the GPU at a margin-clean position.

| where | prefill | decode | numerics |
|---|---:|---:|---|
| **iPhone 17 Pro (A19 Pro), AOT h18p, cooled** | **16.3** | **16.0 tok/s** | **nat 24/24 + oracle 24/24 on every run** |
| iPhone 17 Pro, back-to-back trials | 15.1 → 13.1 | 12.0 → 10.5 | (thermal, monotonic) |
| M4 Max GPU (raw `AIModel` calls, **not** `llm-benchmark`) | — | 85.2 tok/s | top-1 == fp32 oracle |
| M4 Max GPU, fp16 control | — | 49.6 tok/s | — |

Bundle 4.29 GiB · `engine ready` 18.9 s cold / 6.9–9.8 s warm · **no jetsam**, 9.0 GB device free
after. int8hu is 1.72× the fp16 decode, against a 1.77× weight-size ratio — clean bandwidth scaling.

- **It is bandwidth-saturated, and the per-token read is *not* the bundle size.** The 0.77 GiB
  embedding table is a one-row gather, not a matmul. Subtracting it leaves 3.52 GiB = 3.78 GB per
  token, so the ~60 GB/s ceiling is 15.9 tok/s — and the cooled device reads 16.0. (Granite is
  tied-embedding, so *its* head **is** the embedding table and the naive full-bundle estimate holds
  there. Do not reuse that estimate on an untied-head model.)
- The Mac figures are raw `AIModel` calls: `llm-benchmark`/`llm-runner` currently die at dyld on
  this toolchain (`FoundationModels.LanguageModelExecutorGenerationChannel.send` missing), so no
  pipelined-engine Mac number is quoted. For scale, the same raw harness reads 132 tok/s on
  granite-4.0-h-350m where its shipped pipelined bundle reads 191.

## 4-bit is boxed in on three sides

Dropping to 4 bits would lift the per-token read to ~2.4 GiB and the ceiling to ~23 tok/s. Every
4-bit scheme in the tree was gated (teacher-forced top-1 vs the fp32 oracle at the 33 *margin-clean*
prompt positions — oracle top-2 gap ≥ 0.1; near-ties are decided by fp16 noise either way). All
three fail, **for three different reasons**.

| scheme | quality | weights | device decode |
|---|---:|---:|---:|
| **int8 sym-clip b32 + absmax int8 head (ship)** | **33/33** | 4.29 GiB | **16.0 tok/s** |
| int4 **symmetric** clip, block-32 / block-16 | 27/33 · 29/33 | 2.83 · 3.01 GiB | — |
| int4 **asymmetric**, block-64 / block-32 | 30/33 · 31/33 | 2.78 · 2.92 GiB | — |
| **int4 asymmetric, block-16** | **33/33** | 3.10 GiB | **3.0–3.5 tok/s** |
| int4 **k-means** (the `int4km` kernel's format) | 22/33 | 2.64 GiB | — |
| int4 k-means, kernel-eligible weights only | 23/33 | 3.64 GiB | — |

1. **Symmetric int4 is fast but wrong.** A 2-layer GPU probe times it at 2.20 ms/step against
   int8's 2.31 — *cheaper* than int8. No block size rescues the numerics.
2. **Asymmetric int4 is right but slow.** Block-16 is the only 4-bit scheme that gates 33/33, and
   its greedy continuation is token-identical to int8hu's over 48 tokens (device: nat 24/24 +
   oracle 24/24 PASS). But the zero-point path costs ~0.42 ms per layer — ~18 ms over 42 layers —
   and lands at **3.0–3.5 tok/s, 4.6× slower than int8** while reading 1.45× fewer bytes. A pure
   dequant-throughput loss.
3. **k-means int4 is wrong here, and mostly inapplicable.** The format the fused `int4km` kernel
   reads shares one 16-entry codebook across 32 rows × K columns — **no scale along K**. That was
   8/8 exact on gemma4; on Nemotron-H it is the *worst* of the three (22/33), below linear int4.
   And the kernel needs `K % 256 == 0`, while `hidden_size = 3136 = 256·12 + 64` — so every
   projection whose input axis is the hidden dim (`in_proj`, `up_proj`, `q/k/v`, `lm_head`) is
   ineligible. Only 35% of the weight bytes could use it at all.

So the remaining lever has a very specific shape: a **fused asymmetric-int4 matvec kernel**. The
tree has none — `int4km` uses a LUT precisely to avoid the affine path, and that LUT is what costs
the quality here. **int8 at 16.0 tok/s is the ship shape.**

## Gotchas

- transformers **5.12.1** loads the 4B natively (`_pattern_to_list` maps `"-" → "mlp"`); the
  `KeyError: '-'` on 5.5.0 is fixed upstream, no patch needed.
- Drop the `gated_delta_update` **and** `rope` externalize specs — Nemotron-H calls neither, and
  the exporter otherwise hunts for submodules that never run.
- The head is **already untied** (`tie_word_embeddings=false`), so unlike granite there is nothing
  to clone before quantizing it. It still needs absmax, not clipping: a fat-tailed 131k-vocab head.
- Gate the **last margin-clean** position, not the last prompt token — the oracle's top-2 gap at the
  final token is 0.013, a coin flip that produced one spurious FAIL before this was fixed.
- The first device run after install under-reads (~15%) and a 4 GB model throttles hard. Let it cool
  before quoting a number.
