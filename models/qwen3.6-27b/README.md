# Qwen3.6-27B (text decoder) — Core AI

The **Mac-class dense companion** to [Qwen3.6-35B-A3B](../qwen3.6/README.md). Source:
`Qwen/Qwen3.6-27B` (an image-text-to-text checkpoint; this card is the **text decoder**).

Where the 35B-A3B is a sparse MoE, the 27B is the **same Qwen3.5 hybrid decoder run
dense** — no experts, no router, just the proven token mixers at scale. 64 layers on a
3:1 interleave of **GatedDeltaNet** linear-attention mixers and **gated full attention**:

- full attention: head_dim 256, **GQA 24 query / 4 KV heads**, partial mRoPE θ=1e7, a
  swish output gate;
- linear (GatedDeltaNet): **48 value heads over 16 key heads** (GVA — each k/q head is
  shared across **three** value heads, vs the 35B's two), short causal conv + delta rule;
- every FFN is a plain dense `MLP(17408)` (no MoE);
- untied 248320-vocab `lm_head`.

**27B parameters, all dense → the entire model is read per token.** Unlike the 35B-A3B
(≈3B active), there is no sparsity to hide behind, so this is a true 27B-class decode: the
quality of a large dense model, at the memory-bandwidth speed that implies on a Mac.

**Why it runs on this Mac today:** head_dim 256 keeps the full-attention Q buffer small, so
it side-steps the macOS-27-beta MPSGraph decode-heap bug that currently blocks the
head_dim-512 Gemma 4 12B. The qwen3.5 family already runs on this engine.

**⬇️ Converted `.aimodel` bundle:** `qwen3_6_27b_decode_int8hu_block32_sym/` (28 GB,
full LanguageBundle incl. tokenizer; decode-only loop-free for the
[pipelined engine](../../knowledge/pipelined-engine.md)). *HF upload user-gated.*

<!-- gen-cards:use-it begin id=qwen3.6-27b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — run the kit's task op on this model
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let tldr = try await CoreAI.summarize(text, options: .model("qwen3.6-27b"))
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [ChatDemo runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo)
(GUI + CLI, one app for every chat model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/ChatDemo/ChatDemo.xcodeproj
# → Run, then pick "Qwen3.6-27B (dense)" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/ChatDemo
swift run chat-cli --model qwen3.6-27b --prompt "What can you do, offline?"
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let chat = try await ChatSession(catalog: "qwen3.6-27b")
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
- First run downloads the model — 28.0 GB (Mac) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Measured (macOS 27 beta, M4 Max 128 GB, release `llm-benchmark`, `COREAI_CHUNK_THRESHOLD=1`)

| config | bundle | prefill tok/s | decode tok/s | numerics (teacher-forced vs bf16 HF oracle) |
|---|---:|---:|---:|---|
| **int8 linear per-block-32 + untied absmax int8 head (`int8hu --head-sym`) = SHIP** | **28 GB** | **15.8** | **15.9** | int8 == full precision at every confident position (see below) |

**Numerics** — 27B fp32 would need ~111 GB RAM (27.8B × 4 B), unsafe alongside the OS on a
128 GB host, so the oracle is the checkpoint's native **bf16**; the gate is teacher-forced
single-step argmax under the oracle-margin≥0.1 rule (the rule absorbs bf16↔fp16 rounding on
sub-0.1-margin top-2 ties). The result is cleaner than the 35B-A3B's:

- **int8 adds zero confident-margin flips over full precision.** Both `int8hu` and an fp16
  full-precision control score 15/16 vs the bf16 oracle, and they fail the **same** position
  (pos 4, margin 0.50): fp16 flips there **byte-identically to int8** (both pick token 26 vs
  the oracle's 11, both cos ≈ 0.998). So pos 4 is a **bf16-oracle-resolution artifact**, not an
  int8 defect — the position where bf16↔fp16 rounding itself moves the argmax. The only
  int8-vs-fp16 difference anywhere is one sub-0.1-margin tie (pos 14, margin 0.062), which the
  rule absorbs. There is **no int8-specific confident flip to attribute** — the absmax int8
  head and the int8 body are both clean.
- A real-engine smoke (`llm-runner` greedy on the oracle's prompt) reproduces the oracle's
  greedy continuation **token-for-token for 6 tokens** before drifting at a sub-0.1-margin
  tie; free-running greedy is not a valid gate on this tie-dense probe (the teacher-forced
  gate above is), but the exact 6-token prefix is a strong coherence signal, and a chat-prompt
  generation is fluent.

**int4 is a borderline speed/size option — not the quality ship.** A linear int4 per-block-32
gate (head kept fp16) also scores 15/16 under the rule, but *unlike int8* it pays a real
fidelity cost: it flips a **high-confidence** position (pos 0, margin 0.688) that fp16 and int8
both get right, at cos 0.985, and its per-position cosine is systematically lower across the
board (~0.985–0.999 vs int8's 0.998–0.99998). It is far better than the 35B-A3B's int4
(9–10/16 NO-GO — the shared hybrid body tolerates 4-bit better when dense), so int4 is *usable*
for a ~14 GB bundle at roughly double the decode rate, but it is a measurable quality drop. We
ship **int8hu** as the default; int4 is one export flag away (`int4lin`) for callers who want
size/speed over the last bit of fidelity.

A **mixed-precision** middle ground was tested and rejected: MLP/FFN linears (~63 % of the
per-token read) at int4-asymmetric with attention / GatedDeltaNet / head / edge-layers kept at
int8. Keeping the mixers at int8 *does* repair int4lin's high-confidence flip — confirming the
attention/GDN path, not the FFN bulk, is the 4-bit-sensitive part — but the int4 MLP then
introduces its **own** confident flip at a different (middle-layer) position that keeping the
edge layers at int8 does not fix. The result lands in the same borderline-quality class as plain
int4lin while being slower (~23 vs ~30 tok/s) and larger (~19 vs ~14 GB), so there is no
quality-preserving speedup between int8 (clean, 15.9 tok/s) and int4 (borderline, ~30): int8 is
the quality ship and int4 the size/speed option, with nothing useful in between.

- **Dense means no MoE speed trick:** decode rate is set by reading the whole 27B per token.
  At int8 that is ~28 GB/token → 15.9 tok/s, which is **~87 % of the M4 Max memory-bandwidth
  ceiling** — this model is bandwidth-bound, as a dense 27B at int8 must be. (Contrast the
  35B-A3B, which decodes *faster* than this despite more parameters, because only ~3B are
  active per token.)
- The untied 248320-vocab head uses the **absmax `symmetric` per-block-32** rule — clipping
  corrupts big-vocab heads (see the [qwen3.5 card](../qwen3.5/README.md)); `int8hu` quantizes it at
  zero fidelity cost over keeping it fp16.
- Prefill ≈ decode because prefill runs as pipelined S=1 steps (`COREAI_CHUNK_THRESHOLD=1`).
- **Mac-only:** at 28 GB int8 the bundle is far past the iPhone jetsam limit.

## How to reproduce

```bash
cd coreai-models   # with the qwen3_5 model overlay (see ../conversion)
# convert (CPU-side; ~54 GB fp16 load + in-RAM int8 quantize, fits a 128 GB host)
.venv/bin/python ../coreai-models-community/conversion/export_qwen3_5_decode_pipelined.py \
    int8hu --head-sym --hf-id Qwen/Qwen3.6-27B
# bench (needs the coreai-pipelined extra-states engine patch + COREAI_CHUNK_THRESHOLD=1)
COREAI_CHUNK_THRESHOLD=1 .build/release/llm-benchmark \
    --model exports/qwen3_6_27b_decode_int8hu_block32_sym -p 64 -g 128 -n 3
```

Model overlay: `models/macos/qwen3_5.py` (the shared Qwen3.5 hybrid decoder — the GVA
query/key head-repeat is config-driven, so the 27B's 48v/16k ratio needs no new code; the
loader picks up the untied `lm_head.weight` from the checkpoint root). **Decode-only
loop-free** because the GatedDeltaNet `while_loop` doesn't lower on the GPU delegate. No MoE
files — that is the 35B-A3B's `qwen3_5_moe.py`; this dense port reuses the base directly.
