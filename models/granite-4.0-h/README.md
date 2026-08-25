# Granite 4.0-H 1B / 350M (text decoder) — Core AI

Mamba2 + attention hybrid decoder (IBM, dense "H" variants): the 1b is 40 layers =
**36 Mamba2 mixers** (selective-scan SSM: 48 heads × d_head 64, d_state 128, kernel-4
depthwise conv, gated output RMSNorm) + **4 GQA attention layers** (12 q / 4 kv heads,
**NoPE** — no positional embedding, fixed config scale, no q/k norm), hidden 1536,
shared SwiGLU MLP 4096, vocab 100 352, tied head, mup-style scalar multipliers
(embedding ×12, residual ×0.22, logits ÷6). The 350m: 32 layers = 28 + 4, hidden 768.
Source: `ibm-granite/granite-4.0-h-1b`, `ibm-granite/granite-4.0-h-350m`.

**⬇️ Converted `.aimodel` bundles (ready to run):
[mlboydaisuke/granite-4.0-h-CoreAI](https://huggingface.co/mlboydaisuke/granite-4.0-h-CoreAI)** —
`gpu-pipelined/granite_4_0_h_1b_decode_int8lin/` (the ship config, Mac + iPhone) +
`gpu-pipelined/granite_4_0_h_350m_decode_fp16/` (full LanguageBundles incl. tokenizer;
Apache-2.0 LICENSE included).

**The first SSM-scan architecture on the [pipelined-engine fast path](../../knowledge/pipelined-engine.md).**
The enabler is the same observation that unlocked qwen3.5's GDN: **at S=1 the Mamba2
selective scan is a single recurrence step** (`state = state*dA + dt*B*x;
y = (state·C) + D*x` — the HF `use_precomputed_states` branch), so the decode-only
graph is loop-free and lowers on the MPSGraph GPU delegate. State = growing KV for the
4 attention layers + two fixed-shape stacks (conv columns `[36,1,conv_dim,3]`, SSM
state `[36,1,48,64,128]`) — exactly the extra-states patch budget (≤2), the same
`(convState, recState)` shape-class as qwen3.5. No engine changes; neither LFM2.5
GPU-delegate workaround was needed here (per-layer `SSMState` writes compile fine —
the multi-write drop is pattern-dependent — and NoPE attention without q/k-norm gains
shows no fp16-matmul amplification; the SSM step itself computes in fp32 in-graph).

<!-- gen-cards:use-it begin id=granite-4.0-h-1b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — run the kit's task op on this model
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let tldr = try await CoreAI.summarize(text, options: .model("granite-4.0-h-1b"))
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [ChatDemo runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo)
(GUI + CLI, one app for every chat model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/ChatDemo/ChatDemo.xcodeproj
# → Run, then pick "Granite 4.0-H 1B" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/ChatDemo
swift run chat-cli --model granite-4.0-h-1b --prompt "What can you do, offline?"
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let chat = try await ChatSession(catalog: "granite-4.0-h-1b")
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
- First run downloads the model — 1.6 GB (Mac) / 1.8 GB (iPhone) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Measured (macOS 27 beta, M4 Max, release builds, p=128 g=256, `COREAI_CHUNK_THRESHOLD=1`)

| config | bundle | prefill tok/s | decode tok/s | numerics |
|---|---:|---:|---:|---|
| **1b int8 linear per-block-32 (ship)** | **1.6 GB** | **136.7** | **136.5** | **16/16 oracle gate + HF-seeded decode step — on a margin-clean oracle (min top-2 margin 0.137, decode 2.53)** |
| **1b int8lin, iPhone 17 Pro** (PipelinedBench, n=2×2) | 1.6 GB | 30.1–32.2 | **30.8 avg r1 (26.4–31.3)** | **nat 24/24 + oracle 24/24 on BOTH runs — token-identical to the M4 Max GPU sequences** |
| **1b + untied per-block-32 absmax int8 head (`int8hu --head-sym`) = device SHIP** | 1.8 GB | 134.9 | 134.2 | 16/16 oracle gate + decode step; oracle greedy ≡ int8lin, natural forks at +7 (post-`<\|end_of_text\|>` filler — fp16-noise class). **Mac-flat (−1.7%)** — but see the iPhone row |
| **1b int8hu --head-sym, iPhone 17 Pro** (PipelinedBench, n=3×2, settled) | 1.8 GB | 35.1–37.0 | **35.4–37.1 typical (one thermal-dip trial 31.7)** | **nat 24/24 + oracle 24/24 on ALL 3 runs — token-identical to the M4 Max GPU sequences. +17–21% over int8lin's 30.2–31.3: the third "Mac no-win ≠ device no-win" confirmation (head ≈ 10% of per-token read on the BW-saturated surface; ~88% of the ~41 naive ceiling)** |
| 1b fp16 (control) | 2.8 GB | 103.7 | 103.6 | 16/16 oracle gate + decode step |
| 350m fp16 (ship at this size) | 0.66 GB | 193.2 | 191.1 | 16/16 oracle gate + decode step; engine path ≡ torch 24/24 greedy ×2 prompts; `llm-runner` 217.9 tok/s short-run |
| 350m int8 (lin/b8/sel/mix) | — | — | 185.8–186.6 | **not shipped**: gate FAILs *and* no speed win (see below) |

- Prefill ≈ decode because prefill runs as pipelined S=1 steps (`COREAI_CHUNK_THRESHOLD=1`).
- int8lin buys the 1b **+32%** (BW-bound, as expected). The 350m is *overhead-bound*:
  int8 made it *slower* (185.8 vs 191.1) — at ~0.7 GB/token the GPU isn't waiting on
  weights, so dequant work is pure cost. Ship rule of thumb: quantize for speed only
  when the model is big enough to be bandwidth-bound on your device.
- **iPhone: ~31 tok/s ≈ ~84% of the naive BW ceiling** (~60 GB/s ÷ 1.6 GB ≈ 37) — like
  LFM2.5 (~87%), the device is effectively memory-bandwidth saturated; the SSM scan costs
  nothing extra at S=1. Cold GPU specialization 5.7 s / warm load 1.9 s (no AOT needed,
  no increased-memory entitlement issue at 1.6 GB). One trial dipped to 26.4 decode
  (install-adjacent/thermal — the knowledge-page caveat); typical trials 30.2–31.3.
  Cross-device determinism held again: all four numerics checks (2 runs × 2 prompts)
  were 24/24 token-identical to the M4 Max GPU rollouts.

## Why the 350m int8 gate fails (and why that's two separate findings)

1. **Real block-32 sensitivity in the shared MLP.** Torch fake-quant attribution
   (`_smoke/probe_granite_quant_attrib.py`) shows per-block-32 int8 on
   `shared_mlp.output_linear` alone flips sweep positions whose oracle margins are
   *solid* (0.57, 0.17) — genuine quantization damage, not ties. Per-block-8 on the
   MLP (rest at 32) repairs the sweep.
2. **The decode-step check was a statistical tie on this prompt.** The 350m fp32
   oracle's decode top-2 margin is **+0.0102** (and one sweep position sits at 0.045)
   — below the **margin ≥ 0.1 oracle-validity rule** (see the LFM2.5 card): healthy
   int8 noise flips it regardless of recipe (block-8-all lands at +0.0093 — a coin
   toss). The same prompt on the 1b yields margins 0.137–2.95, which is why the 1b
   gates cleanly with the *standard* recipe, no rescues.

Since int8 is also no faster at 350m, the variants were dropped rather than re-gated
on a new prompt: **350m ships fp16, 1b ships int8lin.**

## int8lin recipe (1b)

Per-block-32 **linear** int8 (scale-multiply dequant, no LUT). Quantized: Mamba2
in/out projections, shared-MLP input/output, attention q/k/v/o (NoPE attention
quantizes cleanly here — no LFM-style exclusion needed). Excluded: embedding (tied,
fp16), depthwise conv1d, all norms (incl. the gated Mamba output norm), lm_head.

## Numerics gating

- Tier A parity ladder (authored fp32 vs HF eager oracle, `_smoke/test_granite_ladder.py`):
  Mamba2 mixer stepped S=1 (output + final conv/SSM state), attention mixer + KV
  content, full-model teacher-forced sweep — 16/16 + HF-seeded decode step, both sizes.
- GPU gate (`_smoke/test_granite_decode_oracle_gate.py`, the
  [pipelined-engine](../../knowledge/pipelined-engine.md) shipping gate): teacher-forced
  S=1 sweep on a fresh state + HF-cache-seeded decode step vs the fp32-HF oracle.
  **16/16 + decode for 1b int8lin, 1b fp16, 350m fp16.**
- Engine-path agreement: 24-token greedy rollouts (fixed random + natural prompts),
  python-GPU vs torch reference — 24/24 on the 350m; 1b run + `llm-runner` warm-load
  numbers to follow.

## Convert it yourself

```bash
cd coreai-models   # with the granite4h model overlay (models/macos/granite4h.py) in place
.venv/bin/python ../coreai-models-community/conversion/export_granite4h_decode_pipelined.py int8lin
# fastest device decode (+17-21% iPhone): int8lin + untied absmax per-block-32 int8 head
.venv/bin/python ../coreai-models-community/conversion/export_granite4h_decode_pipelined.py int8hu --head-sym
COREAI_CHUNK_THRESHOLD=1 ./.build/release/llm-benchmark \
    --model exports/granite_4_0_h_1b_decode_int8lin -p 128 -g 256 -n 3
```

`--hf-id ibm-granite/granite-4.0-h-350m` exports the 350m on the same path (use
`fp16` there). Run contract (same as qwen3.5): Swift package patches
[`../apps/coreai-shared-product.patch`](../../apps/coreai-shared-product.patch) +
[`../apps/coreai-pipelined-extra-states.patch`](../../apps/coreai-pipelined-extra-states.patch)
applied on a fresh `coreai-models` clone; `COREAI_CHUNK_THRESHOLD=1` before engine
creation; never call `engine.warmup()` on the S=1 bundle (warm with a 1-token
generate; `llm-runner` needs `--warmup exact --warmup-length 1`).

## License

Model weights: **Apache-2.0** (IBM Granite). The conversion *code* in this repo stays
BSD-3-Clause.
