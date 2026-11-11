# Raw-Metal loop playbook — when leaving the engine pays, and when it doesn't

Distilled from the Gemma-4-E2B raw-Metal port (2026-07, sessions 1–4 — the worked
example is `knowledge/gemma4-raw-metal-port.md`; per-lever details in
`GEMMA4_METAL_LOOP_STATE.md` in the coreai work tree). This doc generalizes: what
actually made the loop fast, which models the approach transfers to, and the physics
that bound prefill and losslessness. Read this BEFORE proposing a raw loop for a new
model.

## 1. The kernel/glue decomposition (the core lesson)

A custom kernel runs at the same speed wherever it is launched — the GPU does not
care who encoded the dispatch. What a raw loop buys is the **surroundings**:

| config (Gemma-4-E2B, iPhone 17 Pro) | decode tok/s | effective BW |
|---|---|---|
| engine + custom int2 kernel (mixed-bit transplant) | 36.5 | 28.6 GB/s |
| same formats, raw loop, before device tuning | 47.7 | 37.4 GB/s |
| raw loop, A19-tuned | 55–56.2 | 43.8 GB/s |

The 36.5 row already had the custom kernel. The +31% to 47.7 came from deleting the
glue, not from kernel math:

- **Cross-seam fusion rights** — inside the engine your kernel plugs into fixed graph
  seams; you cannot fold the neighboring residual/rmsnorm into your prologue or merge
  q/k/v across ops. Owning both sides of every seam took Gemma 4 from 452 to 253
  dispatches/token (A19 issue tax is ~8 µs/dispatch ON the GPU timeline — Mac ~3 µs,
  which is why Mac gains less from fusion).
- **Layout freedom** — pack weights in kernel-native layout at export (scales/biases
  placement, interleaved words); zero runtime conversion tax.
- **GPU-resident token chain** — on-GPU argmax writes the next token id where the next
  step's embed-gather reads it; the CPU never touches the token loop; steps batch 8
  per command buffer, 3 CBs in flight.

Rule of thumb: reach for an in-engine custom kernel when ONE op's math is the
problem; reach for a raw loop when the problem is **between** the ops.

## 2. When a raw loop pays (all three, ideally)

1. **Exotic quantization** the engine has no kernels for (int2 / ternary / mixed-bit
   QAT / table formats). This was most of Gemma 4's win: the engine's expressible
   path left ~35% of achievable bandwidth on the floor.
2. **Exotic per-token structure** with materialization taxes: PLE-style table
   gathers, MoE expert gathers. The decisive trick is **fusing the gather into the
   consuming matvec so the gathered matrix is never materialized** (read
   expert-strided weights in place). Through a graph seam this is usually impossible
   — the framework wants a real tensor between gather and matmul; that missing fused
   primitive is exactly the `gather_qmm` frontier.
3. **Bandwidth-bound decode.** Compute-bound work (diffusion GEMMs, dLLM full-canvas
   forwards, vision towers) is where MPSGraph/TensorOps already win — a hand loop
   loses there.

When NOT:
- **Plain dense int4/int8** — the engine is already at the byte floor
  (`reference: dense = Core AI ≥ MLX`); expected gain +0–10%, not worth permanent
  kernel ownership. EXCEPTION: audit first — see the measured 3B iPhone gap in §5.
- **MLA latent attention** — measured non-win (staged cross-head kernel: 0.93× at
  ≤2K ctx, 1.12× only at 4K). Do not re-propose outside a long-context niche.
- Anything compute-bound.

**Audit before proposing** (the never-blame-the-model rule): effective GB/s =
bytes-per-token ÷ measured tok/s on device, compared against what clean streams
achieve on the same silicon (A19: mixed/gemma workloads ~44; clean int8 dense has
measured into the 60s). If the engine is already near the ceiling, there is no glue
to harvest.

## 3. Prefill physics and the losslessness trade

Prefill throughput is set by **chunk width M** (weights read once per M tokens):
783 MB/token at M=1 → 196 MB at M=4 → 49 MB at M=16 → ~6 MB at M=128 (LiteRT's
chunk). Three regimes:

| route | prefill (iPhone, Gemma 4) | guarantee |
|---|---|---|
| S=1 sequential | 71–72 tok/s | bit-exact |
| S=4 chunks (verify-lane reuse, zero new kernels) | 94.7 (Mac: 441 = 3.6×) | **bit-exact** (proven: S1 token gate PASS with chunks live) |
| M=8/16, K-order-fixed tiles (new kernels) | est. 200–400 | **bit-exact preservable** — keep each output scalar's K-dot in the S=1 accumulation order, stage weights across the M columns |
| M=64/128 GEMM tiles (simdgroup_matrix / TensorOps) | thousands | **downgrade**: tile reductions change accumulation order → near-tie tokens can fork |

The M=64+ "quality cost" is NOT a benchmark-measurable degradation — it is a
different sample from the same quality class (the fp16 oracle is itself one arbitrary
accumulation order, and every other runtime already lives there). What it costs is
the *provability*: you fall back from "bit-identical to S=1" to the zoo's standard
near-tie fusion-numerics class gate. Prefill feeds KV, so one fork can reroute a
whole answer — different, not worse. Decide per model whether the lossless badge is
worth more than the prefill headroom; a split posture ("decode bit-exact, prefill
quality-class") is technically honest and possible.

On A19 specifically, S=4 gains were bounded (+31% vs Mac's 3.6×) because the reused
verify lane is non-fused (~450 dispatches/chunk × 8 µs) and its kernels were
M4-tuned — fusing the wide lane is worth roughly as much as widening it.

## 4. What we did and didn't take from LiteRT (attribution discipline)

Borrowed: the **weights** (Google's QAT mixed-bit checkpoint, values unchanged),
the **existence proof**
(their 44.6 GB/s said the bar was reachable — without it, 36.5 would have read as
"the silicon's limit"), and **model wiring facts** decoded from their released
graphs (PLE handling, MTP drafter seams, drafter-off default). NOT borrowed: kernel
implementations — the decisive levers (constant-memory byte-LUT int2 decode, fusion
folds, per-device R/G tiles) came from on-device per-kernel probes. Their runtime
still beats ours on per-token dispatch count; we caught up by other means.

Measurement lesson from the same-afternoon interleaved A/B: a single historical
number is not decision-grade — LiteRT's own fresh decode spread ±6 tok/s within one
afternoon on one device. Interleave, same afternoon, settled fresh trial1, medians,
pre-committed decision rule.

## 5. Expansion candidates (state of 2026-07-16, audit-first in every case)

| rank | target | expected | why |
|---|---|---|---|
| 1 | **Gemma 4 E4B** | iPhone 15.1 → ~35 (2.3×) | same QAT mixed-bit format — kernels + pack machinery reuse nearly whole; the engine's static-table form doesn't fit the device, so this is an unlock, not just a speedup |
| 2 | **Gemma 4 E2B VL decode** | iPhone 25.5 → ~50 (2×) | same weights as the shipped raw loop; only host work (inject vision embeddings in place of embed-gather at image positions). Zero new kernels |
| 3 | **Llama-3.2-3B / SmolLM3-3B (iPhone dense)** | 19–20 → MLX-parity 34–37 | a MEASURED engine gap, not a hypothesis — MLX proves the silicon does ~35 at 3B; find why the engine loses 45% first (may be an engine fix, no raw loop needed) |
| 4 | **gpt-oss MoE (Mac)** | close the gather_qmm frontier | zero-materialization expert-gather matvec (§2.2) is the direct prescription |
| 5 | **BitCPM-8B ternary (iPhone)** | audit-gated; potential "8B at usable speed on phone" | ternary unpack is the same ALU-bound family the constant-mem LUT fixed for int2 |

Non-candidates: hybrids with standard quant (Qwen3.5/LFM2.5-1.2B/Granite — audit
first, expect only +15–25% from dispatch glue), Gemma 12B/31B Mac (already
half-custom, Mac issue tax is low), MLA (measured non-win), dLLM/image/audio-gen
(compute-bound). The strongest NON-zoo adjacency: LFM2 8B MoE QAT-int4 (official QAT
exists + MoE gather + would be the iPhone's first 8B MoE — the PLE gather machinery
is structurally a mini-MoE already).

## 6. Non-negotiable discipline (inherited, bit twice each)

- `mathMode .safe` AND literal op sequences — fp16 FMA contracting forks near-ties
  even under .safe when you "clean up" an expression. Copy reference kernel bodies,
  widen indices only.
- Gates are the only proof: oracle chain (delegate → python raw loop → Swift), token
  gate at every step, KV byte-compare for prefill work, settled-fresh-trial1 for
  every tok/s claim.
- Kernels version with the HOST (app/kit resources), never next to the weights on HF.
- One canonical engine source, copies synced by `cp` — never hand-fork.
