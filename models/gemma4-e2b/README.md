# Gemma 4 E2B (text decoder) — Core AI

Gemma 4 E2B multimodal model; this card is the **text decoder** (`model_type` `gemma4` /
`gemma4_text`). Source: `google/gemma-4-E2B-it`.

**⬇️ Converted `.aimodel` bundles (ready to run):
[mlboydaisuke/gemma-4-E2B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI)** —
one best verified set per category: `ios-gpu/` (int4-kmeans kernels, 22 tok/s), `ios-ane/`
(6 chunks, fp16-hardened, 8/8), `macos/` (int8 kernels, 56.6–59 tok/s) + shared `ios-frontend/`.

Signature features (all handled in the re-authored model): 35 layers, dual head_dim (sliding 256 /
full 512), attention **scale = 1.0** (QK-norm bounds magnitudes), per-head Q/K RMSNorm + scale-free
V RMSNorm, **KV-sharing** (last 20 layers reuse a producer's K/V), double-wide MLP on shared
layers, **Per-Layer Embeddings** (gated per-layer skip), dual RoPE (sliding θ=1e4 full / full θ=1e6
proportional), final logit softcap `tanh(z/30)·30`. RMSNorm multiplies by `weight` directly.

<!-- gen-cards:use-it begin id=gemma-4-e2b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — run the kit's task op on this model
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let tldr = try await CoreAI.summarize(text, options: .model("gemma-4-e2b"))
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [ChatDemo runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo)
(GUI + CLI, one app for every chat model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/ChatDemo/ChatDemo.xcodeproj
# → Run, then pick "Gemma 4 E2B" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/ChatDemo
swift run chat-cli --model gemma-4-e2b --prompt "What can you do, offline?"
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let chat = try await ChatSession(catalog: "gemma-4-e2b")
let reply = try await chat.respond(to: prompt)
// reply: the answer, generated fully on-device
```

Also runs behind **Apple's FoundationModels API** — CoreAIKit's [`KitGemmaModel`](https://github.com/john-rocky/coreai-kit#works-with-apples-foundationmodels-api) plugs this bundle into the system `LanguageModelSession`; capabilities (tool calling, guided generation) auto-detect per model.

The take-home is [`Examples/ChatDemo/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/main/Examples/ChatDemo/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; the CLI is an argument shell over it, and
the GUI drives the same `ChatSession` across turns for its transcript.
Multi-turn? Hold the `ChatSession` and call `respond(to:)` per turn — it keeps the
conversation history; `streamResponse(to:)` yields tokens as they decode.

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` → product **CoreAIKit**
- Info.plist: none needed
- Entitlements: none on Mac; iPhone needs `com.apple.developer.kernel.increased-memory-limit`
- First run downloads the model — 4.9 GB (Mac) / 4.9 GB (iPhone) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## On-device status (iOS/macOS 27 beta — measured, greedy, 8/8 top-1 vs HF)

<!-- Mac numbers RELEASE-VERIFIED by R2 2026-06-10 (ondevice/MACOS_RELEASE_README.md);
     iOS numbers re-measured 2026-06-10 hands-on in the RELEASE chat app: GPU 22 tok/s
     (int4km monolith; instrumented 22.5, core 39ms / head 2ms — the earlier 17.7 was the
     AOT-harness number) + ANE 6 tok/s. The Release-confirm TODO is resolved. -->

| | macOS GPU (M4 Max) | iOS GPU (iPhone 17 Pro) | iOS ANE (iPhone 17 Pro) |
|---|---|---|---|
| Correctness | ✅ 8/8 exact | ✅ 8/8 exact | ✅ 8/8 exact |
| Decode | **56.6–59.0 tok/s** (Swift e2e; core ~70 tok/s) | **22 tok/s** | **6 tok/s** |
| Path | host-cache fixed-shape + custom Metal kernels (fused-**int8** FFN + head+argmax) | same kernel family at **int4 k-means** (1.3 GB core), monolith + host KV | 6-chunk host-cache, fp32-safe ANE authoring, on-ANE argmax head |

- "What is the capital of France?" → "The capital of France is **Paris**." on every cell.
- macOS ANE is intentionally out of scope (GPU dominates on a Mac; the runtime auto-prefers GPU
  for this structure).
- The ANE number is **energy-play honest**: it is capped by the 262144-vocab head (→ int4/pruned
  head, separate lever) and the host-cache KV re-feed (→ the beta KV-write bug,
  [`../knowledge/coreai-beta-mpsgraph-kvwrite-bug.md`](../../knowledge/coreai-beta-mpsgraph-kvwrite-bug.md);
  an in-graph escape — write-mask-as-input blend — is Mac-GPU-proven, device test pending) — see
  [`../knowledge/performance-ceiling.md`](../../knowledge/performance-ceiling.md).

### Pipelined-engine fast path (2026-06-11 — the post-kernel configs)

Gemma 4 also rides Apple's `coreai-pipelined` engine (zero custom kernels) once the giant
per-layer-embedding table gets an engine hook — two variants, conversion in
[`../conversion/export_gemma4_decode_pipelined.py`](../../conversion/export_gemma4_decode_pipelined.py),
engine patches in [`../apps/`](../../apps/), full method + traps in
[`../knowledge/pipelined-engine.md`](../../knowledge/pipelined-engine.md):

| config (int4-linear weights, oracle 8/8 everywhere) | M4 Max decode / prefill | iPhone 17 Pro decode / prefill |
|---|---|---|
| `int4lin` — PLE rows as a **per-token input** (host mmap provider) | 70.9 / 85.3 | 26.5 / **40.5** (AOT h18p) |
| `int4lin --tbl` — PLE table as a **static graph input** (in-graph gather) | **77.0 / 87.1** | **30.3 / 38.9** (AOT h18p, owned buffers + memory entitlement) |

- **`--tbl` is the fastest decode on BOTH platforms** (Mac +8.6%, iPhone +14% over the
  provider config; +30–36% over the kernel CLI above on Mac). The decode-vs-prefill gap
  closes because no token ever round-trips to the CPU.
- iPhone trade-offs: the static tables are 2.35 GB of OWNED (dirty) memory — needs the
  `increased-memory-limit` entitlement (peak footprint 4.4 GB vs the ~6.4 GB entitled limit) —
  and statically-bound bytes pay a small per-encode residency tax (prefill 38.9 vs the
  provider's 40.5). The provider config is the lighter/steadier choice (clean mmap, no
  entitlement). **Measure on a SETTLED device**: a just-unlocked phone under-reads ~35%
  (19.8 vs 30.3 ten minutes apart); buffer-mode traps in the knowledge page.
- Both beat the kernel monolith row above on device (22–24), with on-device KV growth
  and on-GPU argmax for free.

#### QAT weights (2026-06-11): int4 quality now design-guaranteed

The same two configs re-exported from Google's official QAT release
[`google/gemma-4-E2B-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized)
— bf16 weights **trained for q4_0 rounding** (q4_0 = per-block-32 absmax symmetric int4,
i.e. exactly this int4-linear recipe class). Google: the QAT checkpoints "*preserv[e]
similar quality to bfloat16*", and the unquantized variant is published precisely for
"*custom downstream compilation*". This upgrades the int4 claim from "PTQ that happens to
gate 8/8" to **int4 ≈ bf16 by design** — and it is the one int4 route that doesn't depend
on the model winning the int4-tolerance lottery (qwen3.5 ✗ / LFM2.5 ✗ / gemma4 ✓).

| `gemma4_e2b_qat_decode_…` (oracle regenerated from the QAT checkpoint, margins ≥ 1.97) | M4 Max decode / prefill | iPhone 17 Pro decode / prefill |
|---|---|---|
| `int4lin` (provider) | 74.7 / 89.6 | — (AOT compiled, untested) |
| `int4lin --tbl` | **78.9 / 89.6** | **30.7 / 36.7** (settled; hf-oracle 8/8) |

- Speed is unchanged vs the PTQ bundles (same bytes, same graph) — **the QAT deliverable
  is the quality guarantee**, not throughput. All four bundles gate 8/8 (python GPU +
  engine path); on device the 8 HF-anchored oracle tokens are exact (the 24-token
  Mac-engine determinism check forks once at a post-`<end_of_turn>` filler position —
  fp16 noise territory, granite precedent: judge by the gate).
- A `--lin-sym` probe (plain absmax, the literal q4_0 grid) also gates 8/8 at identical
  speed (72.5 / 90.6) — clipping vs absmax doesn't matter on gemma4 QAT weights; the
  proven clipping recipe stays the default.
- QAT checkpoints **prune the dead shared-layer KV projections** (k_proj/v_proj/k_norm on
  the 20 KV-shared layers — never used, those layers read the producer's cache slot); the
  overlay loader tolerates the pruned layout. The PLE table and the oracle are
  checkpoint-derived — both were regenerated from the QAT weights (a swapped checkpoint
  is a different oracle).
- **Chat-surface (CoreAIChat ⚡ mode, `--tbl` config, 200-token turn): decode 32.7 /
  prefill 44.2 tok/s** on a settled iPhone 17 Pro — the app binds the two PLE table files
  it already downloads for the kernel modes (`ios-frontend/gemma4_gather_raw/`) as owned
  `staticInputBuffers` (~2.35 GB dirty; generation footprint ~3.4 GB, ~3 GB headroom under
  the entitled limit). First load in a container ingests the ~2 GB AOT executable into the
  content-keyed cache (engine load ~11 s; ~6 s warm) and the first process pays the
  executable page-in on its first prefill (~13 tok/s once) — later turns run at full
  speed. The ingest can invalidate sibling models' cached specializations in the same
  container (one wipe + re-spec cycle — see the run contract in
  [`../knowledge/pipelined-engine.md`](../../knowledge/pipelined-engine.md)).

### What it took (the interesting parts)

1. **Fixed-shape host-cache decode** — the beta crashes on any data-indexed in-graph KV write
   (FB23024751 / [apple/coreai-models#5](https://github.com/apple/coreai-models/issues/5)), so KV
   caches are plain I/O: in-graph `cat`, masked SDPA, host writes the new column back. 8/8,
   and it unblocked Mac GPU + device GPU + device ANE with one core.
2. **Custom Metal kernels (GPU)** — fused int8 dequant-LUT matvec for the FFN and the
   262144-vocab head (+ two-level in-kernel argmax, no logit readback). MSL embedded in the
   `.aimodel` (WWDC 325), still 100% Core AI — and it **survives AOT** (`coreai-build` →
   `.aimodelc`, device output bit-identical). Mac decode 13 → 27 → ~57 tok/s. On the iPhone the
   same kernels at **int4 k-means** buy another ~1.5× (device is bandwidth-bound where the Mac
   is ALU-bound) → 22 tok/s.
3. **ANE fp16 exactness** — two root causes, both fixed default-on (GPU-equiv ~2e-8):
   composite RMSNorm overflows fp16 `mean(x²)` on gemma4's large activations → `[x,-x]`
   LayerNorm trick; `nn.Linear` accumulates fp16 → Conv2d 1×1 (fp32 MAC on the conv engine).
   `.float()` casts are no-ops on the ANE.
4. **Chunking for the ANE** — the 35-layer monolith OOMs the on-device first-run compile →
   6 chunks (≤8 layers, **bucket/ctx 64**; a bucket-512 re-export passes on the Mac but its
   on-device first ANE compile still jetsams — long-ANE-ctx waits on the levers below). AOT
   (`xcrun coreai-build compile`) was measured as the un-chunk lever: the un-chunked `.aimodelc`
   now **loads on the device ANE (no compile-OOM)** but is jetsam'd at the first inference —
   load ✅ / run ❌ — and the chunk graphs themselves SIGSEGV the AOT compiler (beta bug), so the
   shipped ANE set stays chunked
   ([`../knowledge/aot-and-specialization.md`](../../knowledge/aot-and-specialization.md)).

## Conversion status (macOS, vs HF eager — 8-token canonical prompt)

- Eager parity: argmax exact; logits maxdiff 1.4e-4.
- Decode core: fp32 (7.0 GB) / **fp16 8/8 (3.5 GB)** / **int8 k-means 8/8 (1.9 GB)**.
- **Stateful** dual-KV core + **ring-buffer** sliding cache (constant ~6 MB sliding KV vs growing) —
  single-pass + incremental match stateless; ring is argmax-exact incl. wrap.
- Tied **head** + softcap as its own unit: fp16 (768 MB) / int8 (392 MB), 8/8.
- Full pipeline (front-end gather → core → head → argmax): **8/8 + top-5 vs HF**, fp16 and int8.
- int4 cannot reach exact (best 6/8); int8 is the floor. gate/up projections must be int8.

## On-device artifacts (int8, 3-stage)

| Stage | Bundle | I/O |
|---|---|---|
| Front-end gather | `gemma4_e2b_frontend_int8` (2.6 GB) | `input_ids → inputs_embeds, per_layer_inputs` — on device this becomes a Swift **mmap gather** (`Gemma4Gather`), the tables never enter process memory |
| Decode core | host-cache fixed-shape core, int8 (+ Metal-kernel FFN variant for GPU) ~1.8–2.0 GB | `inputs_embeds, per_layer_inputs, position_ids, masks, KV-cache I/O → hidden, new K/V columns` |
| Head | `gemma4_e2b_int8_head` (392 MB) / GPU: fused head+argmax kernel (388 MB) | `hidden → logits` (tied lm_head + softcap) / GPU: `hidden → (value,index) partials` |

Full int8 set ~4.9 GB on disk; runs in budget on an iPhone 17 Pro (mmap front-end keeps the
resident footprint flat). Dual-KV state names (when the stateful path returns post-beta):
`slidingKeyCache/slidingValueCache/fullKeyCache/fullValueCache`. Flow details:
[`../knowledge/swift-runtime.md`](../../knowledge/swift-runtime.md).

## Reproduce

Re-authored decoder + stateful + ring + head + front-end gather in `conversion/`. CLI:
`coreai.llm.export gemma-4-e2b --compression int8` (core); `convert_head.py int8` (head);
`export_gemma4_frontend.py` (front-end gather). Fixed-shape host-cache core + chunked ANE export +
Metal-kernel variants ship alongside (`export_gemma4_hostcache*.py`, `export_gemma4_metal.py`,
`export_gemma4_head_kernel.py`). **E4B is ported** (config-verified clean DENSE — no MoE,
contrary to an earlier note here): same pipelined path, zero model-code changes, see
[`gemma4-e4b.md`](../gemma4-e4b/README.md).

Reference CoreML (NOT Core AI) throughput for scale: Gemma4-E2B ~34 tok/s on iPhone 17 Pro
(stateful KV + pruned head + AOT — the stack Core AI reaches once the beta KV-write bug lifts).
