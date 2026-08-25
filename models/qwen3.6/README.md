# Qwen3.6-35B-A3B (text decoder) — Core AI

The **first MoE model on the zoo**, and the answer to Apple's most-requested model
(`coreai-models` issue #1). Source: `Qwen/Qwen3.6-35B-A3B` (an image-text-to-text
checkpoint; this card is the **text decoder**).

Architecturally this is **Qwen3.5's hybrid decoder + a sparse Mixture-of-Experts FFN**:
the token mixers are exactly [Qwen3.5](../qwen3.5/README.md) — 40 layers on a 3:1 interleave of
**GatedDeltaNet** linear-attention mixers and **gated full attention** (head_dim 256, GQA
16/2, partial mRoPE θ=1e7) — but here the GatedDeltaNet runs **32 value heads over 16 key
heads** (GVA: each k/q head is shared across two value heads) and every FFN is a
**256-expert top-8 MoE** with a shared expert:

- router `Linear(2048, 256)` → fp32 softmax over all 256 → top-8 → renormalize by the top-8 sum,
- experts via Apple's `SwitchGLU` / `GatherMM` composite (the data-dependent expert gather),
- an always-on shared `MLP(512)` scaled by a per-token `sigmoid(shared_expert_gate(x))`,
- output = sparse-sum + gated-shared.

**35B parameters, ~3B active per token** → a frontier-quality model that decodes at
**Mac-class speed** because only 8 of 256 experts fire per token.

**⬇️ Converted `.aimodel` bundle:** `qwen3_6_35b_a3b_decode_sym8_gather/` (35 GB, **the
`gather_qmm` kernel build — 2.1× faster, same clean int8 quality**; full LanguageBundle incl.
tokenizer; decode-only loop-free for the [pipelined engine](../../knowledge/pipelined-engine.md)).
Convert with [`conversion/export_qwen3_6_moe_metal_decode_pipelined.py`](../../conversion/export_qwen3_6_moe_metal_decode_pipelined.py) — recipe: [`recipe.toml`](recipe.toml).

<!-- gen-cards:use-it begin id=qwen3.6-35b-a3b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — run the kit's task op on this model
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let tldr = try await CoreAI.summarize(text, options: .model("qwen3.6-35b-a3b"))
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [ChatDemo runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo)
(GUI + CLI, one app for every chat model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/ChatDemo/ChatDemo.xcodeproj
# → Run, then pick "Qwen3.6-35B-A3B (MoE)" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/ChatDemo
swift run chat-cli --model qwen3.6-35b-a3b --prompt "What can you do, offline?"
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let chat = try await ChatSession(catalog: "qwen3.6-35b-a3b")
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
- First run downloads the model — 35.0 GB (Mac) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Measured (macOS 27 beta, M4 Max 128 GB, release `llm-benchmark`, `COREAI_CHUNK_THRESHOLD=1`)

| config | bundle | prefill tok/s | decode tok/s | numerics (teacher-forced vs bf16 HF oracle) |
|---|---:|---:|---:|---|
| **`sym8` gather kernel (reads 8/256 experts) = SHIP** | **35 GB** | **65.3** | **64.9** | **CLEAN — 0 introduced flips/18 vs fp16** (sym8 = the int8lin recipe via a bit-exact gather) |
| int8 linear (GatherMM, reads all 256 experts) | 35 GB | 30.8 | 30.9 | 14/16 (1 flip, pos 6 margin 0.31) — same int8 quality, but the dense over-read |
| fp16 control (4 layers) | 8 GB | 167 | 175 | — (truncated; engine-path smoke) |

**Numerics in full** (the 35B is too large for an fp32 oracle on 128 GB, so the oracle is
the checkpoint's native **bf16**; gate = teacher-forced single-step argmax under the
oracle-margin≥0.1 rule):

- **fp16 full-scale eager tracks the bf16 oracle** at every confident position; its only
  argmax disagreements are sub-0.2-margin positions where bf16↔fp16 rounding itself flips
  the top-2 (pos 12, margin 0.188 — a three-way bf16/fp16/int8 disagreement, i.e. an
  oracle-resolution artifact, not a port defect). So the loader (packed-expert split,
  untied head, GVA) and the MoE math are correct.
- **int8hu adds exactly ONE flip** vs full precision (pos 6, margin 0.31, cos 0.991) —
  body-attributable: `int8lin` (fp16 head) gives the **byte-identical** pos-6 result, so
  the absmax int8 head is clean and `int8hu` ships with no fidelity cost over `int8lin`.
  This is the practical int8 ceiling for a 35B MoE; free-running engine greedy is NOT a
  valid gate here (the probe prompt has many sub-0.1-margin ties that drift under any
  rounding).

- **31 tok/s for a 35B-class model on a Mac** — this is the "frontier quality on your
  desk" slot. Comfortably above reading speed; the active-parameter count (≈3B), not the
  35B total, sets the decode rate.
- The MoE rides Apple's `SwitchGLU`/`GatherMM` exactly like Apple's own `gpt_oss`/`qwen3_moe`,
  so the **expert int8 quantization uses the documented 4-D SwitchLinear override**
  (`block_size [1,1,1,32]`, `axis: None` — Apple's `export/presets.py` MoE recipe, int4→int8);
  the **router and shared-expert scalar gate stay fp16** (quantizing the router can flip
  discrete expert selection for ~0.1 % of the bytes). The untied 248K-vocab head uses the
  **absmax `symmetric` per-block-32** rule (clipping corrupts big-vocab heads — see the
  qwen3.5 card).
- Prefill ≈ decode because prefill runs as pipelined S=1 steps (`COREAI_CHUNK_THRESHOLD=1`).
- Mac-only: at 35 GB int8 the bundle is far past the iPhone jetsam limit. This is the
  64/128 GB-Mac flagship; a smaller MoE (LFM2-8B-A1B class) is the iPhone-MoE story.

## Speed: the expert-gather 2× is now CLOSED (`gather_qmm` kernel)

**Update:** the ~2× MoE-expert-gather penalty below is FIXED. A custom
[`gather_qmm`](../../knowledge/compute-units-and-authoring.md) Metal kernel
(`models/macos/moe_metal.py`) reads only the 8 routed experts (8/256) instead of `GatherMM`'s
dense all-256 read — **decode 30.9 → 64.9 tok/s (2.1×), and CLEAN** (the `sym8` scheme = the same
symmetric-linear int8 recipe, read via a bit-exact gather: 0 introduced flips/18 vs fp16). That is
exactly the "~2× (MoE expert-gather)" term predicted below, now realized. The remaining gap to MLX
4-bit is the **int8-vs-int4 bytes** term — which is a hard wall here (int4 fails this model's
numerics, below). The original accounting, for the record:

30.9 tok/s (the GatherMM path) is **~4× slower than MLX 4-bit on the same M4 Max**
(`mlx-community/Qwen3.6-35B-A3B-4bit`: 125 tok/s, measured), characterized as:

- The 4× decomposes into **~2× (int8 vs 4-bit bytes)** and **~2× (the MoE expert-gather)** —
  the gather 2× is now closed by `gather_qmm` (above).
- **int4 does not survive this model's numerics.** The attention/GatedDeltaNet body can't
  take 4-bit: full int4 teacher-forced gates **9/16 (symmetric) / 10/16 (asymmetric)** vs
  int8's 14/16 — asymmetric's usual +3-5 dB doesn't transfer to this LLM's mixer. Only a
  mixed recipe (experts int4, body int8) reaches a borderline 12/16. So the easy "4-bit 2×"
  is not free here, unlike Apple's QAT-int4 Gemma models.
- The other ~2× is structural: Apple's `GatherMM` composite **gathers then runs a DENSE
  matmul** with no active-experts-only read — it over-reads the 256-expert tensor, sitting
  at ~25 % of the memory-bandwidth ceiling, whereas MLX's custom-Metal `gather_qmm` reads
  only the 8 routed experts (~50 %). Apple's own MoE models use the same `GatherMM`, so
  this is not a porting mistake — it's the maturity of the Core AI GPU MoE path.
- **Core AI is not generally slow:** on a *dense* model the same pipelined engine is ~2×
  MLX (qwen3-0.6b-4bit: 1,150 tok/s). The gap is specific to the MoE expert-gather.
- **Direct int8-vs-int4 confirmation (LFM2.5-8B-A1B, the first working Core-AI int4 MoE):**
  int4 had no usable number here (it fails this model's numerics), so the "~2× bytes" term
  above was inferred from the MLX-4bit comparison, not measured on Core AI. The smaller
  LFM2.5-8B-A1B MoE *does* produce a (degraded but running) int4 bundle, and the direct
  measurement is sharper than expected: **int8 = 39 tok/s** (8.8 GB, 345 GB/s ≈ full-read
  BW-saturated) vs **int4 = 170 tok/s** (5.0 GB, 848 GB/s effective — above physical BW, so
  int4 is *not* full-reading). The over-read tax scales **super-linearly** with dtype (~4×,
  not ~2×): at int8 the `GatherMM` dense read saturates bandwidth, at int4 it doesn't. This
  reinforces the diagnosis — the bottleneck is the dense over-read, independent of the quant
  *scheme* (so int8 block/clipping/per-channel tweaks don't move it) — and quantifies the
  prize the deferred custom `gather_qmm` kernel would capture. (Measured on a validated
  LFM2.5-8B-A1B MoE port; non-QAT int4 also flips structural tokens, so int8 stays the floor.)

The real fix is a custom Metal gather-matmul kernel (Core AI exposes `TorchMetalKernel` →
`coreai.metal4_kernel`), but that API is beta-experimental and the integration with the
pipelined decode path is unverified — a high-variance multi-day spike we are **deliberately
deferring until the Core AI MoE path / kernel API matures**. So this card ships at frontier
*quality* with honest decode *speed*, and we expect the number to rise on OS/runtime
updates without any change to the bundle. For a "fast on Mac today" model, the dense Gemma /
Qwen ports are the better fit; this is the frontier-capability slot.

## A Core AI GPU bug this port surfaced (worked around at the engine surface)

A MoE decode graph raw-loaded via `AIModel.load(from_preferred_compute_unit_kind(.gpu))`
makes MPSGraph try to lower the 256-expert `GatherMM` to the **Neural Engine**, which
fails pathologically: a 2-layer slice aborts with
`MPSGraphANEUtils.mm:939: failed assertion 'ANE compilation writeToFile failed!'`, and a
4-layer slice **explodes the MPSGraph temp dir past 100 GB** (it crashed the host once).
Bisection (`_smoke/isolate_moe_ane_blowup.py`, single-layer random-weight probes under a
hard temp guard): a **dense** MLP layer loads in 0.2 s / 1 GB temp; swapping in the
**256-expert MoE** triggers the ANE assertion. So it's the `GatherMM`→ANE path, not the
GatedDeltaNet or the states.

**The real pipelined engine never hits it.** It loads `dynamic` models with
`SpecializationOptions(preferredComputeUnitKind: .gpu)` **plus `expectFrequentReshapes =
true`** (`ModelStructure.swift`), and that flag steers the graph off the ANE path. The
Python runtime binding doesn't expose `expectFrequentReshapes`, so raw Python load can't
replicate it — but the engine is the deployment path anyway, and Apple's own gpt_oss-20B
MoE runs the same way (78 tok/s). Proof: `llm-benchmark` on the real bundle is rc=0, peak
temp **1 GB**, 30.9 tok/s. Filed as Core AI feedback (ANE assertion / temp blowup on a
raw-loaded GPU-preferred GatherMM graph).

## How to reproduce

The published bundle is the `sym8` **gather-kernel** build. Its recipe is
[`recipe.toml`](recipe.toml) (`zoo_convert.py show qwen3.6-35b-a3b`) — note the one open
question recorded there: `--head-sym` changes the lm_head spec but does not appear in the
bundle name, so which head configuration shipped is not recoverable from the artifact.

```bash
cd coreai-models   # with the qwen3_5 + qwen3_5_moe model overlay (see ../conversion)
# convert the SHIPPED bundle (gather kernel; ~70 GB fp16 load, mmap quantize; ~35 GB bundle)
.venv/bin/python ../coreai-models-community/conversion/export_qwen3_6_moe_metal_decode_pipelined.py \
    sym8
# the pre-gather-kernel build, kept for the comparison table above (NOT what is published):
.venv/bin/python ../coreai-models-community/conversion/export_qwen3_6_decode_pipelined.py \
    int8hu --head-sym
# bench (needs the coreai-pipelined-extra-states engine patch + COREAI_CHUNK_THRESHOLD=1)
COREAI_CHUNK_THRESHOLD=1 .build/release/llm-benchmark \
    --model exports/qwen3_6_35b_a3b_decode_int8hu_block32_sym -p 64 -g 128 -n 3
```

Model overlay: `models/macos/qwen3_5_moe.py` (MoE FFN on `SwitchGLU` + the packed-expert
checkpoint loader) on top of the `qwen3_5.py` hybrid decoder (which gained the GVA
query/key head-repeat for this model). Same engine patch stack as the other pipelined
riders; **decode-only loop-free** because the GatedDeltaNet `while_loop` doesn't lower on
the GPU delegate.
