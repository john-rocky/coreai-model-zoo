# LFM2.5-1.2B (text decoder) — Core AI

Conv + full-attention hybrid decoder (LiquidAI): 16 layers = **10 short-conv mixers**
(depthwise causal conv, kernel 3, B/C/x gating, no activation) + **6 GQA attention layers**
(32 q / 8 kv heads, head_dim 64, per-head q/k RMSNorm, full-dim RoPE θ=1e6), hidden 2048,
MLP 8192 (SwiGLU, auto-adjusted from 12288), vocab 65 536, tied head.
Source: `LiquidAI/LFM2.5-1.2B-Instruct`.

**⬇️ Converted `.aimodel` bundle (ready to run):
[mlboydaisuke/LFM2.5-1.2B-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-1.2B-CoreAI)** —
`gpu-pipelined/lfm2_5_1_2b_instruct_decode_int8lin/` (full LanguageBundle incl. tokenizer;
ships the upstream LFM Open License v1.0 LICENSE file).

**The first non-Qwen architecture on the [pipelined-engine fast path](../../knowledge/pipelined-engine.md)**
— and the easiest ride so far: LFM2 has **no recurrent scan at all** (the conv mixer is a
3-tap causal conv), so the decode graph is loop-free *by construction*, and the only extra
state is one fixed-shape conv buffer `[10, 1, 2048, 2]` — well inside the
extra-states patch budget (≤2). No engine changes needed beyond the existing patch stack.

<!-- gen-cards:use-it begin id=lfm2.5-1.2b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — run the kit's task op on this model
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let tldr = try await CoreAI.summarize(text, options: .model("lfm2.5-1.2b"))
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [ChatDemo runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo)
(GUI + CLI, one app for every chat model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/ChatDemo/ChatDemo.xcodeproj
# → Run, then pick "LFM2.5 1.2B" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/ChatDemo
swift run chat-cli --model lfm2.5-1.2b --prompt "What can you do, offline?"
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let chat = try await ChatSession(catalog: "lfm2.5-1.2b")
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
- Entitlements: none needed
- First run downloads the model — 1.6 GB (Mac) / 1.6 GB (iPhone) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Measured (macOS + iOS 27 beta, release builds, p=128 g=256, `COREAI_CHUNK_THRESHOLD=1`)

| config | bundle | prefill tok/s | decode tok/s | numerics |
|---|---:|---:|---:|---|
| fp16 (+fp32 attn proj), M4 Max | 2.2 GB | 162.8 | **162.1** | 16/16 oracle gate, cos ≥ 0.999998 |
| **int8 linear per-block-32 (ship), M4 Max** | **1.5 GB** | **253.3** | **253.3** | **16/16 oracle gate + HF-seeded decode step (cos ≥ 0.99992); engine path ≡ python-GPU 24/24 greedy** |
| **+ untied per-block-32 absmax int8 head (`int8hu --head-sym`) = SHIP, M4 Max** | 1.6 GB | 277.8 | **276.5** | **16/16 oracle gate + decode step; greedy token-identical to int8lin on both fixed prompts (python runtime AND release `llm-runner`)** |
| **int8hu --head-sym, iPhone 17 Pro** (PipelinedBench, n=3×2, settled) | 1.6 GB | 44.2–46.6 | **44.1–46.6** | **nat 24/24 + oracle 24/24 on ALL 3 runs — token-identical to the M4 Max GPU sequences** |
| **int8lin, iPhone 17 Pro** (PipelinedBench, n=2×2) | 1.5 GB | 39.2–39.4 | **38.0–39.6** | **nat 24/24 + oracle 24/24 on BOTH runs — token-identical to the M4 Max GPU sequences** |
| int8lin, iPhone 17 Pro, chat app (CoreAIChat LFM mode, 200-tok story) | 1.5 GB | 30.7 | **35.8** | coherent instruct output via the bundle chat template |

- **253 tok/s on a 1.2B** beats our qwen3.5-0.8B pipelined result (204) on a model 1.5× larger.
- **The qwen head lever replicates here**: untie the tied lm_head and quantize it absmax
  per-block-32 `symmetric` (clipping corrupts big-vocab heads; REAL per-channel axis-0 is
  broken on this beta GPU delegate — see the qwen3.5 card). The fp16 head was ~19% of the
  per-token read: **+9% on the Mac (253→276.5), +15–20% on iPhone (38.0–39.6 →
  44.1–46.6, typical settled 44.4–46.4)** — the predicted 43–46 window, ~94–98% of the
  naive BW ceiling (~60 GB/s ÷ ~1.27 GB/token ≈ 47); warm engine load 0.3 s, cold
  specialization 4.5 s. The Mac-visible win (qwen-0.8B's was Mac-flat) comes from the
  bigger head share at hidden 2048/vocab 65536.
- Prefill ≈ decode because prefill runs as pipelined S=1 steps (`COREAI_CHUNK_THRESHOLD=1`).
- `llm-runner` (the real engine surface, on-GPU argmax sampling): 262 tok/s in a short run,
  warm load 2.7 s, output token-for-token identical to the python-runtime rollout (24/24).
- **iPhone: ~39 tok/s ≈ ~87% of the naive BW ceiling** (~60 GB/s ÷ ~1.4 GB/token ≈ 45) —
  a much higher fraction than qwen-0.8B realized (~60%); at ~1.4 GB/token the device is
  effectively memory-bandwidth saturated. Cold GPU specialization 6.8 s, warm load 1.6 s
  (no AOT needed). Cross-device determinism held again: iPhone greedy sequences were
  24/24 token-identical to Mac-GPU on both fixed prompts.
- **Chat app integrated**: CoreAIChat ([`../apps/CoreAIChat`](../../apps/CoreAIChat)) has an
  LFM picker segment next to Gemma GPU/ANE and Qwen — both pipelined modes share one
  `PipelinedBackend` (spec-parameterized: bundle dir, HF subpath, warmup token); mode
  switches free the previous model set first. Headless hook: `GEMMA_ENGINE=lfm2`.

## Two GPU-delegate findings this port surfaced (both worked around in the model file)

Both are macOS-27-beta MPSGraph GPU-delegate behaviors, found by bisection with single-layer
probes; the workarounds live in the re-authored `models/macos/lfm2.py`:

1. **Chained fixed-shape state writes are silently dropped.** Per-layer
   `SSMState.update_states` calls compile to a read_handle → slice_update → write_handle
   round trip per conv layer on the *same* state handle. With more than one round trip the
   GPU delegate drops them ALL — the state buffer simply stays zero (the compiled IR is
   correct and token-chained; a single-slot graph works; a 3-slot repro fails). Decode then
   silently runs with no conv history: position 0 is right, everything after is garbage.
   **Workaround: layers return their new conv columns and the model issues ONE fused
   full-state `slice_update` per step.** (qwen3.5's 18-slot graph happens not to trigger
   this — pattern-dependent.)
2. **fp16 attention projections compute ~1.3% relative error under a dynamic-shape graph**
   (the fused attention-prologue matmul accumulates in fp16; the *same* matmul in a static
   graph measures 0.07%). LFM2.5's large q/k-norm gains (|k| up to ~14) amplify this and it
   compounds across the 16-layer stack into garbage logits (full-stack cos 0.71 vs eager).
   **Workaround: the four attention projections (q/k/v/out) keep fp32 weights** (cast in/out
   around the matmul) → layer-level cos 1.000000, +~126 MB. Conv-mixer and MLP matmuls
   measure clean in fp16 and stay fp16/int8.

## int8lin recipe (this model)

Per-block-32 **linear** int8 (scale-multiply dequant, no LUT — 256-entry LUT gathers are
slow on this delegate, see the qwen3.5 card). Quantized: MLP w1/w3/w2 + conv-mixer
in/out_proj (the bulk of the 1.17B params). Excluded: embedding (tied, fp16), depthwise
conv1d, norms, lm_head, **and the four attention projections** (the precision-critical
path above — quantizing them flips near-tie argmaxes: 14/16).

### int4lin: NO-GO (bisected 2026-06-11) — int8 is this model's floor

Decode is BW-saturated on device, so int4 (the remaining decode lever) was probed with the
gemma4-verified **linear** per-block recipe (`int4lin` / `int4lin8` modes in the conversion
script, kept for reproducibility; bundles deleted):

| probe | bundle | gate | note |
|---|---:|---|---|
| int4lin g32 (pure) | 1.0 GB | 14/16 FAIL | **314 tok/s on M4 Max (+24%)** — the speed was real |
| + conv-mixer int8 rescue (`int4lin8`) | 1.1 GB | 15/16 FAIL | fixes the mid-position flip; oracle pos 1 still flips |
| + layers-0/1 MLP int8 rescue | 1.1 GB | 15/16 FAIL | pos 1 unchanged |
| g16 blocks + conv rescue | 1.1 GB | 15/16 FAIL | cos floor 0.90→0.95 but argmax still a special token — and **97.6 tok/s: per-block-16 scales are a 3.2× slow class on this delegate** |

The surviving failure is a **short-context regime flip** (oracle position 1 = two tokens of
context, fp32 top-2 margin 0.52 — a real miss, not a near-tie): it resists every surgical
rescue, and recovering it would need ~all-MLP int8, which is int8lin again. Same conclusion
as qwen3.5 (both k-means AND linear int4 fail there): **int4 tolerance is a model property;
gemma4 passes, qwen3.5 and LFM2.5 don't.** int8lin stays the ship config.

## Numerics gating

- Tier A parity ladder (fp32 eager vs HF oracle): conv block / attn block / full prefill /
  stateful decode + teacher-forced S=1 sweep — all cosine 1.000, 16/16 top-1.
- GPU gate (the [pipelined-engine](../../knowledge/pipelined-engine.md) shipping gate):
  teacher-forced S=1 sweep over the 16 oracle positions on a fresh state + an
  HF-cache-seeded decode step, top-1 vs the fp32-HF oracle. **16/16 + decode for both
  bundles.**
- Oracle-validity note: the oracle prompt must not contain near-tie positions. Our first
  prompt had two positions with top-2 logit margins of 0.012–0.014 (statistical ties that
  even healthy int8 noise flips, and that fp16 passes only by luck). Gate prompts on a
  **fp32-oracle top-2 margin ≥ 0.1 at every position** (ours: ≥ 0.40) — computable before
  any bundle exists, so it selects the measuring instrument, not the result.

## Convert it yourself

```bash
cd coreai-models   # with the lfm2 model overlay (models/macos/lfm2.py) in place
.venv/bin/python ../coreai-models-community/conversion/export_lfm2_decode_pipelined.py int8lin
# fastest decode (+9% Mac): int8lin + untied absmax per-block-32 int8 head
.venv/bin/python ../coreai-models-community/conversion/export_lfm2_decode_pipelined.py int8hu --head-sym
COREAI_CHUNK_THRESHOLD=1 ./.build/release/llm-benchmark \
    --model exports/lfm2_5_1_2b_instruct_decode_int8lin -p 128 -g 256 -n 3
```

Run contract (same as qwen3.5): Swift package patches
[`../apps/coreai-shared-product.patch`](../../apps/coreai-shared-product.patch) +
[`../apps/coreai-pipelined-extra-states.patch`](../../apps/coreai-pipelined-extra-states.patch)
applied on a fresh `coreai-models` clone; `COREAI_CHUNK_THRESHOLD=1` before engine creation;
never call `engine.warmup()` on the S=1 bundle (warm with a 1-token generate;
`llm-runner` needs `--warmup exact --warmup-length 1`).

## License

The **model weights** are under the **LFM Open License v1.0** (LiquidAI; `lfm1.0`):
Apache-style grants, but **Commercial Use is licensed only below a US$10 M annual-revenue
threshold** (entities at/above it are not licensed; 501(c)(3)-class non-profits exempt for
non-commercial/research use). Redistribution must retain notices and include the license —
any converted-bundle upload ships the upstream `LICENSE` file alongside. The conversion
*code* in this repo stays BSD-3-Clause.
