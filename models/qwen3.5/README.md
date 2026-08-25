# Qwen3.5 (text decoder) — Core AI

Hybrid linear + full-attention decoder (Mamba/SSM-style linear mixer + periodic full attention,
gated-delta, partial mRoPE, MTP head). Source: `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`
(image-text-to-text checkpoints; this card is the **text decoder**).

**⬇️ Converted `.aimodel` bundles (ready to run):
[mlboydaisuke/qwen3.5-0.8B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreAI)** —
`gpu-pipelined/` (**ship `int8hu --head-sym` decode-only loop-free: iPhone 69.7–74.0 / Mac
210 tok/s**, full bundle for the pipelined engine; the fp16-head `int8lin` comparison point
is 50.3–51.5 / 204), `ios-gpu/` (static ctx-2048: **int8 fused-kernel monolith
42.5–45.4 tok/s** + its q16 chunked-prefill companion **147 tok/s prefill**, plus the previous
fp16 monolith 27.7), `ios-ane/` + `macos/` (dynamic int8, 14.7 / 58.5 tok/s).

To our knowledge this is the first hybrid linear-attention (SSM) LLM running on Core AI —
including on the iPhone **Neural Engine**.

<!-- gen-cards:use-it begin id=qwen3.5-2b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — run the kit's task op on this model
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let tldr = try await CoreAI.summarize(text, options: .model("qwen3.5-2b"))
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [ChatDemo runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo)
(GUI + CLI, one app for every chat model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/ChatDemo/ChatDemo.xcodeproj
# → Run, then pick "Qwen3.5 2B" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/ChatDemo
swift run chat-cli --model qwen3.5-2b --prompt "What can you do, offline?"
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let chat = try await ChatSession(catalog: "qwen3.5-2b")
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
- Entitlements (iOS): `com.apple.developer.kernel.increased-memory-limit`
- First run downloads the model — 2.9 GB (Mac) / 2.9 GB (iPhone) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## On-device status (iOS/macOS 27 beta — measured, greedy)

<!-- Mac numbers RELEASE-VERIFIED by R2 2026-06-10 (ondevice/MACOS_RELEASE_README.md);
     dynamic device numbers 2026-06-09; static fp16 numbers 2026-06-10 PM; int8-kernel +
     chunked-prefill numbers 2026-06-10 evening (ondevice/_qwen_ios_fast_RESULTS.md
     §Session Q — Q1/Q2). All iOS numbers are RELEASE builds. -->

Two engines, both 100% Core AI:

| Engine | macOS GPU (M4 Max) | iOS GPU (iPhone 17 Pro) | iOS ANE (iPhone 17 Pro) |
|---|---|---|---|
| **dynamic** (int8, 969 MB — shipped app) | **58.5 tok/s** · 8/8 | 12.5 · 8/8 | **14.7 · 8/8** |
| **static fixed-shape** (fp16 host-cache monolith, **ctx 2048**) | — | 27.7 tok/s · exact | ✗ numerics (this beta) |
| **static + fused int8 Metal kernels** (ctx 2048, GPU argmax head) | — | **42.5–45.4 tok/s** · exact ← release config | GPU-only (custom kernels) |
| **+ chunked prefill** (q16 blocks, int8 LUT companion graph) | — | **prefill 147 tok/s** (185-tok prompt: 4.2 s → 1.26 s; decode unchanged) | — |
| **decode-only loop-free × pipelined engine** (int8 linear per-block-32, dynamic KV) | **204 tok/s** · 16/16 oracle | **50.3–51.5 tok/s** · 24/24 ≡ Mac-GPU (beats the int8v3 kernels; warm load 0.2 s) | — |

- "The capital of France is" → " Paris." on every shipped cell (static: Mac-GPU 8/8 vs HF +
  on-device greedy correct; the int8-kernel monolith is ALSO Mac-GPU 8/8 EXACT and produced
  byte-identical device output to the fp16 path on a 185-token prompt).
- **The int8 fused-kernel monolith is ~3× the dynamic path and ~1.6× the fp16 static path**
  (42.5–45.4 vs 27.7): the device GPU is weight-bandwidth-bound (~1.5 GB of fp16 weights per
  token), so fused dequant-in-matvec Metal kernels (k-means LUT, fp32 accumulate — embedded in
  the `.aimodel`, 100% Core AI) halve the stream to ~760 MB/token. The 248320-token tied head
  runs as a fused matvec + two-level **GPU argmax** (greedy; the host reduces vocab/8 partials).
- **int4 does NOT survive on this model** (PyTorch gate, k-means group-32: head 3/8, MLP 0/8,
  SSM 2/8 vs the HF oracle) — unlike gemma4, whose FFN+head take int4 k-means at 8/8. int8 is
  the qwen sweet spot; 4-bit would need finer codebooks (kernel-level change).
- **Chunked prefill (q16 blocks)**: a prefill-only companion graph consumes the prompt 16 tokens
  per pass (SSM scan unrolled in-graph, fp32 recurrence; KV/conv/rec states handed off to the
  decode graph; FULL blocks only — the remainder uses q=1 decode). 185-token prompt: 4.2 s →
  **1.26 s (147 tok/s, 6.8 ms/token)**, decode untouched, output identical. q32 measured NOT
  faster (blocks become unroll/SDPA-bound, and the q=1 remainder grows). The companion is int8
  (MPSGraph LUT — fp16 prefill + the decode monolith together exceed the app memory budget).
- **Context capacity is nearly free on this architecture**: growing the fixed bucket 8×
  (256 → 2048) costs only ~9% (30.4 → 27.7 tok/s) — only the 6 full-attention layers carry
  growing KV; the 18 SSM layers' states are fixed-size. Hybrid models suit fixed-shape export
  unusually well. Dynamic stays as the unbounded-context fallback.
- Within the dynamic engine the **ANE beats the GPU** (14.7 vs 12.5) — a hybrid SSM decoding on a
  phone NPU, exactly. The *static* ANE variant is blocked **this beta**: the loop-free SSM
  recurrence needs fp32 accumulation, the ANE executes fp16-only, there is no blessed SSM
  composite to externalize, and custom kernels are GPU-only. Re-test each new beta.
- The int8-kernel monolith is still weight-bandwidth-bound (~21 ms/tok ≈ 760 MB int8 stream at
  ~36 GB/s — the same kernel bandwidth measured on gemma4); the rest is KV/SDPA I/O and
  dispatch. Both former "next levers" (int8 kernels, chunked prefill) are now shipped.
- ⚠️ **Benchmark release builds only** (Mac AND iOS): a debug binary measures the Mac at ~19
  tok/s (vs 58.5) and the static iOS path at ~10 (vs ~30) — per-token host work dominates
  unoptimized Swift.

### The device unlock: loop-free decode

The gated-delta recurrence is a sequential scan (`torch.ops.higher_order.while_loop` via Apple's
`GatedDeltaUpdate` composite). It converts and runs on the **Python** runtime, but the `scf.while`
doesn't lower on either **Swift** delegate. The fix: at `query_len == 1` the scan is one step —
a loop-free single-step recurrence (`use_loopfree_step`) is **bit-identical** (maxdiff 0.0) to the
composite at s=1. Decode runs the loop-free graph; prefill runs as chunked s=1 steps (zero-state
OK). That makes the hybrid architecture device-deployable today, with no converter patch.

### Pipelined-engine fast path: 204 tok/s on macOS (3.5× the custom-kernel CLI)

The loop-free trick unlocks more than hand-rolled device loops: a **decode-only
export** (every linear layer on `use_loopfree_step`, `input_ids` STATIC `[1,1]`,
position_ids + KV seq still dynamic) contains no `scf.while` at all, lowers
cleanly on the MPSGraph GPU delegate, and — because it keeps the dynamic-KV
shape — rides Apple's **`coreai-pipelined` engine** (`EngineFactory` →
async non-blocking encode, on-GPU argmax sampling, on-device KV growth) instead
of a synchronous per-token `fn.run()` loop. Prefill runs as pipelined S=1 steps:
set `COREAI_CHUNK_THRESHOLD=1` (prompt tok/s ≈ decode tok/s).

Quantization sweep (M4 Max, p=128 g=256, release `llm-benchmark`):

| config | bundle | decode tok/s | numerics |
|---|---:|---:|---|
| fp16 | 1.4 GB | 175.8 | == torch fp16, 24/24 greedy |
| int8 k-means g32 | 1.0 GB | 113.2 | EXACT — but the 256-entry LUT gather is slow on the GPU delegate |
| int8 linear per-block-32 (`int8lin`) | 1.0 GB | 204.1 | token-for-token == fp16-GPU; 16/16 single-step top-1 vs the fp32 HF oracle + HF-cache-seeded decode step |
| + untied int8 lm_head, block-32 with clipping (`int8hu`) | 1.3 GB | 201.4 | 16/16 PASS but no Mac win — shelved at first; the 2B later showed "no Mac win" ≠ "no device win" |
| **+ untied per-block-32 absmax int8 head (ship)** — `int8hu --head-sym` | 1.3 GB | **210.0** | 16/16 + decode step; greedy rollouts token-identical to int8lin on both fixed prompts |

**The head config is the iPhone lever**: on the Mac the head matvec is largely
hidden (+3%), but on the bandwidth-bound phone the fp16 head was **54% of the
~0.95 GB per-token read** — quantizing it took the device from 50.3–51.5 to
**69.7–74.0 tok/s (+40%)**. Use absmax `symmetric` for the head (the clipping
qscheme corrupts big-vocab heads — discovered on the 2B, see the 2B section).

Notes: k-means LUTs aren't slow per se (official qwen3-0.6b runs int4-km g8 at
1000+ tok/s) — the 256-entry int8 LUT is the slow case, 16-entry int4 is fast;
per-block *linear* dequant (scale multiply, no LUT) is what unlocks int8 here.
int4-km g8 fails the oracle gate on this model even with the official-recipe
int8 per_tensor rescue (MLP-only and early-layer variants both tried; the
per_tensor rescue is coarser than int4-g8 for the SSM in_projs) — int8 is the
qwen3.5 floor, matching the earlier g32 finding.
Judge quant quality by single-step top-1 vs the HF oracle or fp16-GPU sequence
match — greedy rollouts fork from CPU-fp16 after ~17 tokens on natural prompts
(fp16 noise, not a bug).

To RUN the bundle you need the engine to carry the SSM conv/rec states: Apple's
pipelined engine hard-requires exactly 2 states as shipped. The
extra-states patch ([`../apps/coreai-pipelined-extra-states.patch`](../../apps/coreai-pipelined-extra-states.patch))
adds up to 2 fixed-shape extra states (zero-filled at init, bound every encode,
zeroed on `reset()`). App-side warmup must use `queryLength=1` (the S=1 bundle;
the engine default warms shape 256 — `llm-runner` needs
`--warmup exact --warmup-length 1`; `llm-benchmark` is fine, it warms via a
real trial).

**iPhone 17 Pro (measured 2026-06-11, Release build, one-shot benchmark app on
the patched engine): the ship config (int8 + per-block-32 absmax int8 head)
decodes 69.7–74.0 tok/s, 24/24 token-identical to the Mac-GPU sequences on
both fixed prompts — +40% over the fp16-head int8lin (50.3–51.5) and ~1.6×
the shipped int8v3 fused-kernel monolith (42.5–45.4)** with zero custom
kernels. The loop-free graph lowers cleanly on the iOS MPSGraph GPU delegate
(no execute crash); warm load 2.9 s (int8lin: cold 4.8 s / warm 0.2 s).
Caveat: pipelined prefill is S=1 (≈ decode, 72–74 tok/s), so for very long
prompts the static q16 chunked-prefill companion (147 tok/s) still wins TTFT —
chunkwise-parallel GDN prefill is the open fix. fp16×pipelined was correctly
predicted to lose (~44 ceiling) and was not run.

Export: [`../conversion/export_qwen3_5_decode_pipelined.py`](../../conversion/export_qwen3_5_decode_pipelined.py)
(ship: `int8hu --head-sym`; `int8lin` fp16-head default;
`fp16` / `int8` k-means for comparison).

### Qwen3.5-2B on the same fast path: 161 tok/s on M4 Max, 28–30 on iPhone

The 2B is architecturally identical (24 layers = 18 linear + 6 full attention, hidden
2048), so the decode-only loop-free export and the extra-states patch carry over with
zero new work — same script, `--hf-id Qwen/Qwen3.5-2B`. Measured (M4 Max, p=128 g=256,
release `llm-benchmark`, `COREAI_CHUNK_THRESHOLD=1`):

| config | bundle | prefill tok/s | decode tok/s | numerics |
|---|---:|---:|---:|---|
| **int8lin + per-block-32 absmax int8 head (ship)** — `int8hu --head-sym` | 2.9 GB | 161.2 | **160.8** | 16/16 single-step top-1 vs the fp32 HF oracle + HF-cache-seeded decode step; greedy rollouts token-identical to int8lin |
| int8 linear per-block-32, fp16 head (`int8lin`) | 2.3 GB | 127.0 | 127.2 | 16/16 + decode step (cos 0.99999) |
| fp16 | 3.5 GB | 91.2 | 90.9 | 16/16 + decode step (cos 0.999999) |
| + int8 head, block-32 **with clipping** (`int8hu`) | 2.8 GB | 151.1 | 159.1 | **FAILS the gate (10/16)** — see below |
| int4 **linear** per-block-32 (`int4lin`) | 1.7 GB | 156.2 | 156.0 | **FAILS the gate (10/16 AND the HF-seeded decode step**, top-1 561≠220, cos 0.9955**)** |

**The head was the lever, and the killer was the clipping, not the bits**: the
2B's fp16 tied head is ~1.0 GB of the ~2.4 GB per-token read, and quantizing it
is worth +26% — but with the default `symmetric_with_clipping` qscheme the
248 K-vocab head flips 6/16 oracle top-1s, with a tell-tale signature (one
position craters to cos 0.62 while its neighbors sit at 0.999x — outlier head
rows getting clipped, not uniform noise; the 0.8B's smaller head tolerated the
same recipe). Switching the head to plain **absmax `symmetric`** fixes it: per-block-32 +
symmetric gates 16/16 at the same ~161 tok/s.

**Granularity correction (2026-06-11)**: the original A/B bisect believed it
compared per-channel (axis 0) against per-block-32 — but the export script
parsed the granularity flag without applying it, so BOTH probes were
per-block-32 (byte-identical bundle sizes confirmed it; per-channel scales
would be 0.5 MB vs ~30 MB). What the bisect actually proved is that the
clipping was the killer. REAL per-channel axis-0 int8 was tested later the
same day and is **broken on the macOS-27-beta GPU delegate** (garbage logits,
cos ~0 vs torch, reproduced in a minimal head-only graph at multiple vocab
shapes, sym and clipping alike — torch-level numerics are 16/16, so it is a
delegate lowering bug, not a quantization problem). **Ship shape = per-block-32
+ absmax symmetric** (`int8hu --head-sym`; block32 is the default granularity).
The HF bundles published under the historical `*_perchan_sym` name contain
per-block-32 heads and are kept under that name — every published number was
measured on per-block-32 + symmetric and stands.

The `int4lin` row closes the int4 question **for both quantizer families**: the
0.8B failed int4 as k-means (g32, and g8 + int8 rescue variants); the 2B now
also fails int4 as *linear* per-block-32 — and unlike `int8hu` it fails even
the clean cache-seeded single step, i.e. the damage is in the transformer (SSM
in_projs), not just the head. **int8 is the qwen3.5 floor regardless of
quantizer** (contrast Gemma 4, which ships int4 at oracle 8/8 — quantization
sensitivity is a property of this GDN hybrid, not of the toolchain). It is not
even faster than `int8hu` (156 vs 159): int4-linear dequant underuses bandwidth
on this delegate, matching the gemma4 int4lin observation.

**iPhone 17 Pro (device-measured 2026-06-11): the ship config decodes
28–30 tok/s** (4 trials over 2 runs: 29.5/28.3/30.2/25.8, prefill ≈ same at
S=1) with perfect numerics (nat + oracle 24/24, token-identical to the Mac-GPU
engine) — **at or above the CoreML 2B port (~27)**, and the same bundle does
161 on the Mac. The earlier fp16-head `int8lin` measured 19.1–21.3 (decode is
fully bandwidth-bound; dropping the 1.0 GB fp16 head read is what closed the
gap). Phone requirements: the
`com.apple.developer.kernel.increased-memory-limit` entitlement (cold GPU
specialization `std::bad_alloc` without it) and ~4 GB of free disk for the
specialization cache (ship bundle: cold spec 22.3 s, warm load 5.6 s). Trap
for big bundles: failed cold specializations leave partial caches that eat
disk and turn later attempts into `NSPOSIXErrorDomain code=2` at engine
create — uninstall the app to reclaim.

## Conversion status (macOS, vs HF eager)

- Prefill + stateful decode: **cosine 1.0 / top-1 100%** (fp32).
- **fp16**: top-1 exact; multi-token generation stable (the gated-delta recurrent state is a leaky
  integrator, so fp16 rounding decays — no mixed-precision SSM state needed; only fix was casting
  RoPE cos/sin to the query dtype).
- **int8 k-means palettization**: top-1 EXACT (prefill + decode). int4 degrades (not shippable).

## On-device artifacts (int8, all-in-one stateful bundle)

`input_ids, position_ids → logits` with **4 in-place states** `(keyCache, valueCache, convState,
recState)`; fp16 embed + tied lm_head in-graph, int8 transformer (conv1d full precision);
`last_token_only` → logits `[1,1,vocab]`.

| Size | Bundle | Parity |
|---|---|---|
| 0.8B | **969 MB** | prefill + decode top-1 exact (Mac + iPhone GPU/ANE) |
| 2B | **2.2 GB** | prefill + decode top-1 exact (Mac) |

The app pushes the bundle into its sandbox (`devicectl device copy to --domain-type
appDataContainer`) rather than bundling it. Runtime contract + state shapes:
[`../knowledge/swift-runtime.md`](../../knowledge/swift-runtime.md) and
[`../knowledge/stateful-kv-cache.md`](../../knowledge/stateful-kv-cache.md). Apple's Swift engine is
2-state; this model needs the generic N-state runner (`swift/`).

## Reproduce

Re-authored decoder + hybrid 4-state export live in `conversion/` (port of `coreai_models`'s macOS
authoring path + a `qwen3_5_text` registry entry). On-device bundle: `conversion/export_qwen3_5.py
[0.8b|2b]` (int8 k-means stateful); decode graph `export_qwen3_5_decode.py [--int8]`; **pipelined
fast path `conversion/export_qwen3_5_decode_pipelined.py [int8lin]`** (+ the Swift engine patch
`apps/coreai-pipelined-extra-states.patch`, run with `COREAI_CHUNK_THRESHOLD=1`). HF reference
oracle generated separately (a transformers build with `qwen3_5`).

The static iOS exports (workspace scripts, to be consolidated here): host-cache monolith
`export_qwen3_5_hostcache.py --num-chunks 1 --max-ctx 2048 [--kind int8v3]` (fp16 or fused-kernel
+ GPU-argmax head; `--verify` = chained greedy 8/8 vs the HF oracle on the Mac GPU) and the
prefill companion `export_qwen3_5_prefill.py --chunk 16 --max-ctx 2048 --int8 --verify`
(oracle + chunked-vs-q=1 parity gates). Fused kernels are the generic gemma4 set
(`knowledge/custom-metal-kernels.md`) wrapped over qwen's Linears (MLP, attn q/o, SSM
in/z/out, tied head).

Reference CoreML (NOT Core AI) device throughput for scale: 0.8B ~48 tok/s, 2B ~27 tok/s on iPhone
17 Pro.
