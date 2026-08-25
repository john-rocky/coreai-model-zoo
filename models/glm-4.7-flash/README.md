# GLM-4.7-Flash (text decoder) — Core AI

The **first model on the zoo with Multi-head Latent Attention (MLA)** — the
DeepSeek-V3 attention family Core AI had no port of — and a strong **local-coding**
model for 64/128 GB Macs. Source: `zai-org/GLM-4.7-Flash` (MIT, `model_type
glm4_moe_lite`).

Architecturally this is **MLA attention on all 47 layers + a sparse
Mixture-of-Experts FFN** (DeepSeek-V3 style):

- **MLA** (heads 20, `q_lora_rank` 768, `kv_lora_rank` 512, `qk_nope` 192 +
  decoupled `qk_rope` 64 = head_dim 256, `v_head_dim` 256, RoPE θ=1e6,
  **interleaved**). Authored in the **naive / materialized** form: the latent
  caches are projected up to per-head q/k/v, decoupled RoPE is applied to the
  64-dim rope slice, and the full per-head `[20, 256]` Q/K/V run through Apple's
  standard `SDPA` composite (scale `256**-0.5`). The KV cache stores the
  materialized per-head K/V (full-MHA size). *The absorbed-MLA form that keeps
  only the `[512]` latent in cache is a follow-up.*
- **MoE FFN** (layers 1–46; layer 0 is a dense `MLP(10240)`,
  `first_k_dense_replace=1`): 64 routed experts, top-4, plus **one non-gated
  shared expert**. Routing is `noaux_tc` which — at `n_group == topk_group == 1`
  — reduces exactly to **sigmoid scoring + selection-only bias correction**: the
  top-4 indices come from `sigmoid(logits) + e_score_correction_bias`, the
  gathered weights come from the **raw sigmoid**, then `norm_topk_prob` (÷sum)
  × `routed_scaling_factor` (1.8). Experts ride Apple's `SwitchGLU` / `GatherMM`
  composite (the data-dependent expert gather), the shared `MLP(1536)` is added
  directly (no sigmoid gate, unlike Qwen3.5-MoE).

**~30B parameters, ~3B active per token** → a strong local-coder that decodes at
**Mac-class speed** because only 4 of 64 experts fire per token.

**⬇️ Converted `.aimodel` bundle:** `glm_4_7_flash_decode_sym8_gather/` (30 GB, **the
`gather_qmm` kernel build — 2.6× faster, same clean int8 quality**; full LanguageBundle incl.
tokenizer; decode-only loop-free for the [pipelined engine](../../knowledge/pipelined-engine.md)).
Convert with [`conversion/export_glm47_moe_metal_decode_pipelined.py`](../../conversion/export_glm47_moe_metal_decode_pipelined.py).

<!-- gen-cards:use-it begin id=glm-4.7-flash (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — run the kit's task op on this model
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let tldr = try await CoreAI.summarize(text, options: .model("glm-4.7-flash"))
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [ChatDemo runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo)
(GUI + CLI, one app for every chat model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/ChatDemo/ChatDemo.xcodeproj
# → Run, then pick "GLM-4.7-Flash (MoE+MLA)" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/ChatDemo
swift run chat-cli --model glm-4.7-flash --prompt "What can you do, offline?"
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let chat = try await ChatSession(catalog: "glm-4.7-flash")
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
- First run downloads the model — 30.0 GB (Mac) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Measured (macOS 27 beta, M4 Max 128 GB, release `llm-benchmark`, `COREAI_CHUNK_THRESHOLD=1`)

| config | bundle | prefill tok/s | decode tok/s | numerics |
|---|---:|---:|---:|---|
| **`sym8` gather kernel (reads 4/64 experts) = SHIP** | **30 GB** | **53.6** | **52.4** | **CLEAN — 0 introduced flips/18 vs fp16** (sym8 = the int8lin recipe via a bit-exact gather) |
| int8 linear (GatherMM, reads all 64 experts) | 30 GB | 20.5 | 20.3 | engine ≡ eager-int8 (exact 32-token greedy); fp16 ≡ fp32 oracle 16/16 — same int8 quality, dense over-read |
| int8hu @ -p 64 -g 128 (lower context) | 30 GB | 20.5 | 20.5 | — (decode is context-stable; set by per-token weight read, not KV growth) |
| int8hu, 2-layer control | 1.6 GB | 420 | 502 | — (engine-path smoke / scratch-heap pre-check) |

**52 tok/s for a 30B-class local coder on a Mac** (was 20.3 before the `gather_qmm`
kernel — **2.6×**). The [`gather_qmm`](../../knowledge/compute-units-and-authoring.md)
Metal kernel reads only the 4 routed experts (4/64) instead of `GatherMM`'s dense
all-64 read, at the **same clean int8 quality** (`sym8` = the shipped int8lin recipe
via a bit-exact gather: 0 introduced flips/18 vs fp16). The rate is flat across
context (the per-token weight read dominates, not the KV cache). GLM is still a bit
slower than [Qwen3.6-35B-A3B](../qwen3.6/README.md) (64.9 tok/s, also gather'd) for an
architectural reason: GLM runs **full MLA attention with a materialized full-MHA KV
cache on all 47 layers** (per-token: q_a→q_b `768→5120`, kv_a→kv_b `512→8960`
up-projections + a 20-head × 256-dim SDPA), whereas Qwen3.6 is a 3:1 hybrid that runs
cheap GatedDeltaNet linear-attention on ¾ of its layers. The **absorbed-MLA** form
(cache only the `[512]` latent, fold the up-projection into Q) is the remaining
decode/​KV-memory lever — the planned follow-up now that the expert-gather is solved.

## Numerics

The 30B model is ~120 GB in fp32 — too large for an fp32-resident oracle on a
128 GB host — so the parity ladder gates the **fp16 port vs the fp32 HF oracle**
(fp16-vs-fp32 rounding lowers cosines by ~1e-3, but a real logic bug collapses
them to <0.9, so structural errors are still caught). Results on a 16-token probe:

- **RoPE** cos/sin **1.000000** (the interleaved decoupled-RoPE convention,
  verified bit-exact vs HF — the non-interleaved path differs by attention-score
  ~25, so the convention is load-bearing).
- **Isolated MLA block** cosine **1.000000** (maxdiff 1.2e-4) — the naive
  materialized MLA is correct.
- **Isolated MoE block** cosine **1.000000** (maxdiff 1.0e-4) — `noaux_tc`
  routing + non-gated shared expert + `SwitchGLU` experts + the per-expert→
  SwitchLinear loader are correct.
- **Full prefill**: all 47 layers ≥0.9995, logits cosine 0.999977, **top-1
  16/16**; stateful decode (HF-seeded and self-seeded) cosine 1.000000 with
  matching argmax; teacher-forced S=1 sweep 16/16.

**int8 end-to-end** (raw `AIModel.load` aborts on the MoE→ANE `GatherMM` path —
see the [Qwen3.6 card](../qwen3.6/README.md) — so the int8 gate drives the REAL engine):
the **coreai-sequential** engine (GPU, greedy) reproduces the **eager-int8 CPU
greedy continuation token-for-token** over 32 tokens from the probe prompt —
`". This process,—known as word embedding—is crucial for tasks like sentiment
analysis, machine translation, and question answering. However"` — a coherent,
on-topic continuation. So **engine ≡ python** at int8, and the int8 quantization
(linear per-block-32 body + absmax int8 head + 4-D SwitchLinear experts) holds
quality.

## Notes

- The MoE rides Apple's `SwitchGLU`/`GatherMM` exactly like Apple's own
  `gpt_oss`/`qwen3_moe`, so the **expert int8 quantization uses the documented
  4-D SwitchLinear override** (`block_size [1,1,1,32]`, `axis: None`); the
  **router stays fp16** (quantizing it can flip discrete expert selection for
  ~0.1 % of the bytes). The untied 154 880-vocab head uses the **absmax
  `symmetric` per-block-32** rule (clipping corrupts big-vocab heads).
- Prefill ≈ decode because prefill runs as pipelined S=1 steps
  (`COREAI_CHUNK_THRESHOLD=1`).
- **Mac-only**: at 30 GB int8 + a full-MHA materialized KV cache (~8 GB at 8 K
  ctx) the footprint is far past the iPhone jetsam limit. This is the 64/128 GB-Mac
  local-coder slot.
- The MoE expert-gather speed caveat is the same as the
  [Qwen3.6-35B-A3B card](../qwen3.6/README.md): Apple's `GatherMM` over-reads the dense
  expert tensor, so int8 decode sits well below the per-token-active-bytes ideal;
  the deferred fix is a custom Metal `gather_qmm` kernel. int4 is not the free 2×
  it looks (non-QAT int4 flips structural tokens on these LLM mixers); int8 is the
  ship floor.

## How to reproduce

The published bundle is the `sym8` **gather-kernel** build. Its recipe is
[`recipe.toml`](recipe.toml) (`zoo_convert.py show glm-4.7-flash`) — with the same open
question as qwen3.6-35B: `--head-sym` changes the lm_head spec without changing the bundle
name, so the shipped head configuration is not recoverable from the artifact.

```bash
cd coreai-models   # with the glm4_moe_lite model overlay (see ../conversion)
# convert the SHIPPED bundle (gather kernel; ~60 GB fp16 load, mmap quantize; ~30 GB bundle)
.venv/bin/python ../coreai-models-community/conversion/export_glm47_moe_metal_decode_pipelined.py \
    sym8
# the pre-gather-kernel build, kept for the comparison table above (NOT what is published):
.venv/bin/python ../coreai-models-community/conversion/export_glm47_decode_pipelined.py \
    int8hu --head-sym
# bench (pipelined engine + COREAI_CHUNK_THRESHOLD=1)
COREAI_CHUNK_THRESHOLD=1 .build/release/llm-benchmark \
    --model exports/glm_4_7_flash_decode_int8hu_block32_sym -p 64 -g 128 -n 3
```

Model overlay: `models/macos/glm4_moe_lite.py` (MLA naive-materialized attention +
`noaux_tc` MoE on `SwitchGLU` + the per-expert checkpoint loader). **Decode-only
loop-free** for the pipelined engine; drive token-by-token with the
`coreai-sequential` engine variant (the default / `coreai-pipelined` variants feed
a prefill chunk that the static-`[1,1]` graph can't take).
