# Gemma 4 12B (text decoder) — Core AI

Gemma 4 12B **dense** text decoder on the pipelined-engine fast path, ported 2026-06-12
**directly from Google's QAT release**
[`google/gemma-4-12B-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized)
(bf16 weights *trained* for q4_0 rounding = per-block-32 absmax symmetric int4 — exactly
the `int4lin --lin-sym` recipe, so **int4 ≈ bf16 quality by design**). A Mac-class flagship
companion to the Qwen3.6-35B-A3B MoE: a 12B dense model that reads ~7 GB/token at int4 and
decodes at interactive speed on an M4 Max.

**⬇️ Converted `.aimodel` bundles:
[mlboydaisuke/Gemma-4-12B-CoreAI](https://huggingface.co/mlboydaisuke/Gemma-4-12B-CoreAI)**
— `gpu-pipelined/` decode-only **int8lin** (14 GB, verified-clean) and **int4linsym** (8.2 GB,
faster 4-bit), both built with the **custom flash-decode SDPA kernel** (`--metal-sdpa`) that lets
the 12B run on the engine at all (see *Throughput* below).

> **First Core AI runtime for a ≥16-head × head_dim-512 full-attention model.** The stock
> MPSGraph SDPA crashes on this layout (a GPU scratch-heap `ViewOp` overflow); the ship bundles
> replace that op on the full layers with a custom Metal flash-decode kernel. Use the `_msdpa`
> bundles — the plain (non-kernel) bundles still crash on the current engine.

<!-- gen-cards:use-it begin id=gemma-4-12b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — run the kit's task op on this model
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let tldr = try await CoreAI.summarize(text, options: .model("gemma-4-12b"))
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [ChatDemo runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo)
(GUI + CLI, one app for every chat model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/ChatDemo/ChatDemo.xcodeproj
# → Run, then pick "Gemma 4 12B" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/ChatDemo
swift run chat-cli --model gemma-4-12b --prompt "What can you do, offline?"
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let chat = try await ChatSession(catalog: "gemma-4-12b")
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
- First run downloads the model — 14.7 GB (Mac) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Architecture (config + checkpoint verified) — clean dense, no PLE

Top `model_type: gemma4_unified` (the text decoder of the multimodal wrapper). Unlike the
on-device **E2B/E4B** siblings, the 12B is a **clean dense interleaved-attention Gemma
decoder with NONE of the PLE / AltUp / Laurel / MoE / KV-sharing / double-wide-MLP
machinery** (`hidden_size_per_layer_input: 0`, `enable_moe_block: false`,
`num_kv_shared_layers: 0`, `use_double_wide_mlp: false`; no `per_layer`/`altup`/`laurel`
weight keys). It is much closer to `gemma3_text` than to E2B.

- **48 layers**, hidden 3840, intermediate 15360, **16 heads**, vocab 262144, softcap 30.0.
- **5:1 sliding:full** (`layer_types`; full attention every 6th layer — idx 5/11/…/47).
- **Dual head_dim**: sliding 256 / full `global_head_dim` 512.
- **Dual KV-head count via `attention_k_eq_v: true`** — the 12B-specific wrinkle: sliding
  layers use `num_key_value_heads` 8 with a real `v_proj`; full layers use
  `num_global_key_value_heads` **1**, carry **no `v_proj`**, and set value = the *raw*
  `k_proj` output (pre-norm/pre-RoPE) followed by a scale-free V RMSNorm.
- Attention scale **1.0** (QK-norm bounds magnitudes), per-head Q/K RMSNorm + scale-free V
  RMSNorm, a learned per-layer scalar (`layer_scalar`), dual RoPE (sliding θ 1e4 full-rotate;
  full θ 1e6 "proportional" — first 64 of 256 freq pairs rotate, rest NoPE), tied embeddings.

The port adds two small overlay files (`gemma4_dense_text.py` + `gemma4_dense_pipelined.py`);
the existing E2B `gemma4_text.py`/`gemma4_pipelined.py` are PLE/KV-sharing-specific and don't
fit the dense 12B.

### One growing KV pair — stock engine, no patch

The pipelined engine grows exactly ONE KV state pair, so the dual sliding/full attention
rides a single `[48, 1, 8, S, 512]` cache (`n_kv_max` 8, `hd_max` 512): every layer owns a
slot (no sharing), sliding layers zero-pad head_dim 256→512, full layers zero-pad KV-heads
1→8, and both slice back to their real shape on read (the padded region is never read).
Sliding layers attend over the LINEAR cache with SDPA's window mask. `embed_tokens` + the
tied `lm_head` + final softcap live in-graph — **exactly 2 states, so the bundle loads on the
stock `CoreAIPipelinedEngine` with no engine patch** (the simplest zoo decode-only path).

## Verification (the ladder)

| check | result |
|---|---|
| Block parity vs HF gemma4 eager (random weights, fp32) | **max\|Δ\| = 0, cos = 1.0, argmax match** (k_eq_v / dual RoPE / softcap / RMSNorm / layer_scalar all bit-exact) |
| Pipelined unified-KV core ≡ stateless eager (pure torch) | **max\|Δ\| 3.7e-6, 0 flips** over the S=1 sweep |
| Tiny-config full lowering → GPU engine ≡ eager | **16/16 argmax**, min cos 0.999997 |
| fp32 oracle (real 12B, chat-templated) | coherent — "The capital of France is Paris." |
| **`int8lin` — EAGER teacher-forced gate vs fp32 oracle (margin ≥ 0.1)** | **35/35 exact, cos ~0.9999 → PASS** |
| `int4lin` — same gate | 27–28/35 exact, cos ~0.98 (4-bit precision loss; see below) |

**int8 is the verified-clean ship; int4 is the faster 4-bit option.** The `int8lin`
35/35-exact pass proves the graph/overlay is bug-free. `int4lin` is *not* near-lossless on
this checkpoint — it shows real ~2 % logit error (cos ~0.98) that flips a few
moderate-margin **structural / "thought-channel" tokens** (the actual answer
"The capital of France is Paris" matches with high margin). This is inherent 4-bit
precision loss of the same class as MLX's own 4-bit — *not* a bug (int8 is exact) — and
unlike the near-lossless E2B/E4B QAT int4, most likely because coreai's `symmetric` int4
grid (`[-7, 7]`) differs from the q4_0 `[-8, 7]` grid the QAT weights were trained for.
Gate numerics via the **eager teacher-forced** path; the GPU raw `AIModel.load` path falls
to an fp16/ANE compile (uniform cos 0.98 + a spurious NaN) that does *not* reflect the
engine's real numerics.

## The engine blocker — and the custom-kernel bypass

The full 48-layer bundle hits an **MPSGraph memory-planning bug**: the stock SDPA lowering of a
**full-attention** layer's Q tensor (`[1, 16, 1, 512]` fp16 = 16 heads × `global_head_dim` 512 =
16 384 B) materialises a `ViewOp` that overflows MPSGraph's ~208 KB GPU decode scratch heap, and
the engine aborts at the **first** decode token:

```
allocateMTLBufferFromMTLHeap: offset 204544 + size 32768 exceeds heap total 229376
GPUMemrefOps.mm:687: failed assertion 'Failed to acquire the source buffer for the ViewOp'
```

Sliding-layer Q (16 × 256 = 8 KB) fits, and **E2B/E4B full layers (8 heads × 512 = 8 KB) also
fit** — only the 12B's **16-head × 512** full Q tips the heap over. It is invariant to every
model-side change (KV pad↔replicate, `.contiguous()`, SDPA variant), because they all keep
MPSGraph's SDPA op. Filed as [apple/coreai-models#27](https://github.com/apple/coreai-models/issues/27).

**The fix (this is what ships):** replace the SDPA *op* on the full layers with a **custom Metal
flash-decode kernel** (`models/macos/gemma4_dense_metal_sdpa.py`, registered via
`export_to_coreai_with_kernels`, enabled with `--metal-sdpa`). It removes the offending ViewOp
entirely. The full (global) layers carry a single global KV head, and a q=1 decode attends every
grown key, so the kernel is a mask-free GQA-16:1 flash decode (scale 1.0, fp32 online softmax,
one SIMD-group per query head); sliding layers keep MPSGraph SDPA (their window mask is unchanged).
With it, **Gemma 4 12B runs on the stock pipelined engine for the first time.**

## Throughput (M4 Max, `--metal-sdpa`, `-p 128 -g 256 -n 3`)

| bundle | decode | prefill | size | quality |
|---|---|---|---|---|
| **`int8lin_msdpa_g8`** (clean ship) | **~23 tok/s** | 27.5 tok/s | 14 GB | engine greedy == fp32 oracle ("The capital of France is Paris.") |
| `int4linsym_msdpa_g8` (fast option) | **~33 tok/s** | 43.4 tok/s | 8.2 GB | answers correctly ("Paris") but 4-bit-lossy phrasing (see int4 caveat) |

**Two flash-decode kernels** (both numerically identical — same engine token-match):

- the **simple** kernel (one SIMD-group per query head) is near the int8 weight-bandwidth ceiling
  at short context but its serial KV scan degrades at long context (the full/global layers read
  *every* grown key — no window);
- the **higher-occupancy** kernel (`--split-g 8`, the default, bundle suffix `_g8`) splits each
  head's KV scan across 8 cooperating SIMD-groups (FlashDecoding-style) to keep the GPU busy.

Measured int8 decode vs generated tokens (the global layers' KV scan growing with context):

| gen tokens | simple | `g8` (default) |
|---|---|---|
| 64 | 28.2 | 28.6 |
| 256 | 22.2 | 23.0 |
| 1024 | 17.5 | **20.3** (+16%) |

Decode is otherwise weight-bandwidth-bound (int8 14 GB ÷ ~400 GB/s ≈ 28.6 tok/s ceiling, hit at
short context). MLX's own 4-bit bench stays blocked by `mlx_lm` not yet supporting `gemma4_unified`.

## Reproduce

```
# overlay (on the coreai-models checkout): gemma4_dense_text.py + gemma4_dense_pipelined.py
#   + gemma4_dense_metal_sdpa.py (the full-layer flash-decode SDPA kernel)
cd coreai-models
# clean ship (int8) — --metal-sdpa bypasses the scratch-heap crash; --split-g 8 (the default) is
# the higher-occupancy kernel (use --split-g 0 for the simple 1-SIMD-group kernel):
.venv/bin/python ../coreai-models-community/conversion/export_gemma4_12b_decode_pipelined.py \
    int8lin --metal-sdpa --max-ctx 4096
# faster 4-bit option:
.venv/bin/python ../coreai-models-community/conversion/export_gemma4_12b_decode_pipelined.py \
    int4lin --lin-sym --metal-sdpa --max-ctx 4096
# engine-truth gate (greedy vs fp32 oracle; run SOLO — parallel python-GPU = MTL4CommandQueueError):
.venv/bin/python ../_smoke/engine_tokenmatch_gemma4_12b.py \
    exports/gemma4_12b_qat_decode_int8lin_msdpa_g8
# bench:
COREAI_CHUNK_THRESHOLD=1 .build/release/llm-benchmark \
    --model exports/gemma4_12b_qat_decode_int8lin_msdpa_g8 -p 128 -g 256 -n 3
```

> Without `--metal-sdpa` the bundle still exports and is numerically clean, but **crashes on the
> current engine** at the first decode token (the full-attention scratch-heap `ViewOp` bug). The
> truncated `--num-layers N` bundles do *not* reproduce the crash (the heap only overflows under
> the full 48-layer graph's pressure) — probe the full bundle via the real `llm-runner`, not raw
> `AIModel.load(gpu)` (which ANE-falls-back on the 14 GB graph and can't run the GPU-only kernel).
