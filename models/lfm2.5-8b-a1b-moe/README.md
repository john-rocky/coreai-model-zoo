# LFM2.5-8B-A1B (sparse-MoE text decoder) — Core AI

Conv + full-attention **MoE** hybrid decoder (LiquidAI): 24 layers = **18 short-conv mixers**
(depthwise causal conv, kernel 3, B/C/x gating) + **6 GQA attention layers** (32 q / 8 kv heads,
head_dim 64, per-head q/k RMSNorm, full-dim RoPE θ=5e6), hidden 2048, vocab 128 000, tied head.
The first two layers are dense (SwiGLU, intermediate 7168); **every layer after is a 32-expert
top-4 sparse MoE** (expert MLP intermediate 1792, sigmoid routing + selection-only expert bias,
no shared expert). 8.3B total / **~1.5B active per token**. Source: `LiquidAI/LFM2.5-8B-A1B`.

A custom Metal kernel fixes how MoE decode reads memory, making this practical to ship. **Shipped
Mac-only** (the clean `sym8` int8 bundle): the int4 bundle that fits the iPhone was *validated to
run on device* (first MoE on the phone) but is **not shipped** — non-QAT int4 doesn't hold quality
(see below).

<!-- gen-cards:use-it begin id=lfm2.5-8b-a1b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — run the kit's task op on this model
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let tldr = try await CoreAI.summarize(text, options: .model("lfm2.5-8b-a1b"))
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [ChatDemo runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo)
(GUI + CLI, one app for every chat model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/ChatDemo/ChatDemo.xcodeproj
# → Run, then pick "LFM2.5-8B-A1B (MoE)" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/ChatDemo
swift run chat-cli --model lfm2.5-8b-a1b --prompt "What can you do, offline?"
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let chat = try await ChatSession(catalog: "lfm2.5-8b-a1b")
let reply = try await chat.respond(to: prompt)
// reply: the answer, generated fully on-device
```

Also runs behind **Apple's FoundationModels API** — CoreAIKit's [`KitLanguageModel`](https://github.com/john-rocky/coreai-kit#works-with-apples-foundationmodels-api) plugs this bundle into the system `LanguageModelSession`; capabilities (tool calling, guided generation) auto-detect per model.

The take-home is [`Examples/ChatDemo/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/main/Examples/ChatDemo/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; the CLI is an argument shell over it, and
the GUI drives the same `ChatSession` across turns for its transcript.
Multi-turn? Hold the `ChatSession` and call `respond(to:)` per turn — it keeps the
conversation history; `streamResponse(to:)` yields tokens as they decode.

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` → product **CoreAIKit**
- Info.plist: none needed
- Entitlements: none needed (macOS)
- First run downloads the model — 9.0 GB (Mac) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## The `gather_qmm` kernel — the MoE-decode fix

MoE FFN decode normally lowers through the `SwitchGLU`/`GatherMM` composite, whose matmul — as
lowered — reads **all 32 experts' weights every token** even though only the top-4 are routed. So
MoE decode is over-read-bound, not active-param-bound: plain int8 here was bandwidth-saturated at
**39 tok/s** reading its full 8.8 GB/token.

The fix ([`models/macos/moe_metal.py`](../../../coreai-models/python/src/coreai_models/models/macos/moe_metal.py),
a `coreai_torch.TorchMetalKernel`) takes the routed expert indices as a kernel **input** and reads
**only the 4 routed experts'** weight slabs (`QP[w, n, e]` with `e = IDX[slot]` — an indexed load,
so the other 28 experts are never fetched). `MetalSwitchGLU` is a drop-in for `SwitchGLU`;
`metalize_moe(model, scheme)` swaps every MoE layer. The kernel owns its weight format, with four
schemes: **`sym8`** (symmetric-linear int8, per-K-block-32 scale — the CLEAN one, matches the
shipped int8-linear quality), `km8`/`km4` (k-means int8/int4), `aff4` (affine int4). All read 1
byte (or nibble)/weight, so all hit the same ~141 tok/s; they differ only in quality (see below).
The kernel is GPU-only by construction, so the old GatherMM→ANE raw-load abort can't occur, and it
composes with the stateful KV+conv decode graph.

## Measured (macOS + iOS 27 beta, release builds, p=128 g=256, `COREAI_CHUNK_THRESHOLD=1`)

| config | bundle | decode tok/s | quality (vs fp32) | verdict |
|---|---:|---:|---|---|
| int8 linear (GatherMM dense over-read), M4 Max | 8.8 GB | 39.2 | clean | the over-read baseline |
| **sym8-gather (Mac), M4 Max** — **SHIPPED** | **8.8 GB** | **140.4** | **clean (+1 flip/41 = fp16 ceiling)** | **Mac ship: 3.6× AND clean** |
| int8km-gather (Mac), M4 Max | 8.4 GB | 141.0 | +5 flips/41 | superseded by sym8 (k-means lossier) |
| int4km-gather, M4 Max / **iPhone 17 Pro** — *not shipped* | **4.7 GB** | 162.7 / **31.3** | +12 flips/41 (degraded) | *validated to run* on iPhone, but NOT clean |

**Honest bottom line:** the gather kernel gives the 3.6× speed on both int8 and int4. With the
**symmetric-linear int8 kernel (`sym8`) the Mac bundle is BOTH 3.6× faster AND clean** (matches
the shipped int8-linear quality) — **this is what ships.** The **iPhone** needs int4 for size
(8.8 GB int8 won't fit even with the memory-limit entitlement), and **non-QAT int4 is a hard
quality wall**: the int4 bundle was *validated to run on device* (first MoE on the phone, ~32
tok/s) but is measurably degraded, so it is **not shipped** (rebuild it locally if you want it).

## Quality — fp32-oracle margin-rule gate (honest)

Teacher-forced over a 41-token factual paragraph (S=1, accumulating KV+conv = the decode path).
A top-1 disagreement vs the fp32 oracle counts as a **REAL flip** only if the oracle's logit gap
to the token the model picked is **≥ 0.1** (below = statistical tie). The kernel itself is
**bit-exact** vs the dequantized reference — the cost is purely the **expert weight quantization**:

| scheme | real flips / 41 | introduced beyond fp16 | worst fp32 margin | read |
|---|---:|---:|---:|---|
| fp16 (ceiling) | 1 | — | 0.11 | fp16 ≈ fp32 |
| **sym8** (linear int8) | 2 | **1** | 0.34 | **CLEAN — at the fp16 ceiling** ✅ |
| int8km (k-means int8) | 6 | 5 | 1.18 | lossier than sym8 → superseded |
| int4 aff (affine int4) | 12 | 11 | — | degraded |
| int4 km (k-means int4) | 13 | 12 | 7.81 | degraded |

Read honestly:
- **sym8 is clean** — 1 introduced flip out of 41 (margin 0.34), i.e. at the fp16 ceiling, the same
  quality as the shipped int8-linear bundle. This is the quality+speed result; **it is what the Mac
  bundle ships.** (The earlier int8km / "fp16-faithful" claim was wrong — k-means int8 introduces
  5 flips; corrected by this gate.)
- **int4 is a wall.** TWO independent 4-bit schemes — k-means (+12) and affine-block-32 (+11) —
  both land at ~12 introduced flips with large margins. Non-QAT int4 cannot reach clean here;
  that needs QAT int4 weights (LiquidAI ships none). The int4 bundle stays grammatical and
  on-topic (no broken grammar) and is the only thing that fits the phone, but it is **not a
  quality-equivalent model** — published as the compact / on-device variant with this caveat.
- On a bare prompt the *base model itself* greedy-degenerates into repetition (present in fp16
  too) — use the chat template + sampling.

## Bundles

**⬇️ [mlboydaisuke/LFM2.5-8B-A1B-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-8B-A1B-CoreAI)** —
ships the upstream LFM Open License v1.0 LICENSE file. **Shipped: `gpu-pipelined/..._sym8_gather/`
(8.8 GB, Mac, clean)** only. The iPhone int4 bundle is NOT on HF (degraded — see above); rebuild
it locally if you want the on-device variant.

Convert with
[`conversion/export_lfm2_moe_metal_decode_pipelined.py`](../../conversion/export_lfm2_moe_metal_decode_pipelined.py)
(`sym8` = clean Mac ship; `int4km`/`aff4` = iPhone-jetsam-safe but degraded). Run with
`COREAI_CHUNK_THRESHOLD=1` (the decode graph's `input_ids` is static `[1,1]`; prefill runs as S=1
pipelined steps via `llm-benchmark`).
