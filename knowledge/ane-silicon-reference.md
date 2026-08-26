# ANE silicon reference — the reverse-engineered map (arXiv:2606.22283)

> Foundation note (external-reference distillation). Source: S. H. Bryngelson, *Apple Neural Engine:
> Architecture, Programming, and Performance*, arXiv:2606.22283v1 (June 2026), CC BY 4.0 — a 235-page
> reverse-engineered account (static decompile of runtime/compiler/kernel-driver/firmware + live
> measurement), with a companion open-source direct runtime **ANEForge**
> (github.com/comp-physics/ANEForge, arXiv:2606.17090). Every claim in the paper carries a
> measured / decompile-derived / predicted label (its Appendix E), and chapter numbers below point into it.
>
> **Scope guard — read before quoting a number.** All figures are the paper's, taken on **M1/M2/M5
> Macs over the *direct* (below-the-framework, unentitled) route**, not our Core AI-route benches.
> A-series values are decompile-derived or predicted, never iPhone-measured. The direct route is
> private API: measurement/research only, not shippable. Our own protocol-matched measurements
> ([`performance-ceiling.md`](performance-ceiling.md), [`coreai-vs-mlx-speed.md`](coreai-vs-mlx-speed.md))
> stay the authority for what our stack does; this note is the map of mechanisms, ceilings, and traps
> underneath. Marks: **[M]** silicon-measured, **[P]** predicted/derived, **(our read)** our inference.

## Compression: what streams vs what folds, per generation (ch 7)

Two outcomes hide under "compression": a form that **streams** crosses DRAM in compressed bytes and
is reconstructed to fp16 at the multiplier input (bandwidth win); a form that **folds** is expanded
to dense fp16 in DRAM before dispatch (disk-size win only). The split is a per-chip HAL feature-byte
table, not a property of the op.

| Form | M1 / A13 | M2 / A14 | A15+ | M5 / A17-class | Measured speedup |
|---|---|---|---|---|---|
| int4 LUT (palettization) | **stream** | stream | stream | stream | M1 2.37× [M]; M5 1.6–1.8× [M] |
| structured sparsity | **stream** | stream | stream | stream | M1 1.55–1.64× at 0.43× bytes [M] |
| int8 per-tensor/channel | fold | **stream** | stream | stream | M2: 0.85× (2k-wide) → 0.52× (8k-wide) latency vs fp16 [M] |
| blockwise affine | fold | fold | **stream [P]** | stream [M] | A15 floor is read off gate bytes, *not* silicon-confirmed |

- **Palettization is the only dense form that streams on every generation** — the mechanism under
  our "ANE = palettization" rule. There is **no int4 arithmetic lane**: a 4-bit value is always a
  palette index into a 16-entry fp16 codebook. (The coreai-torch blockwise-int4 → ANE SIGSEGV we hit
  is a separate framework-route compiler trap; both facts stand.)
- int8 on M1 is a size-only saving (folds to fp16 in DRAM, 1.0× latency). From A14 it dispatches as
  int8 and halves the weight stream.
- Separate from weight *storage*: the array has a double-int8 **compute** mode — int8 arithmetic runs
  ~1.4–2× the fp16 rate (ch 9) [M]. W8A8 is a compute lever, not just a bandwidth one.
- fp8 (e4m3) exists in the element-type catalog but the datapath is **H18 (A18)-only** [P].
- Energy: on a bandwidth-bound matmul (M2), int4 stream = **0.41× fp16 energy** at equal work; power
  stays ~2–2.6 W across formats, latency does the work (ch 10) [M].

## Decode vs encoder verdict, with the mechanism (ch 9, 11, 14)

- **Decode is bandwidth- + dispatch-bound.** A per-layer-dispatched decoder issues ~40–50 dispatches
  per token at a ~0.23 ms/eval floor (M1); a single-output-row matmul reaches ~5.9 GFLOP/s — the
  array never fills. GPU at batch 16: **2.7× faster, 4.6× more energy-efficient** than the engine [M].
- **int8 weights do not speed a dispatch-bound decoder: 0.99× vs fp16** [M]. "Batching, not
  quantization, is the control that moves serving throughput."
- Weight-stream bandwidth: **M1 ~51 GB/s → M5 ~145 GB/s vs GPU ~230 GB/s** [M]. On current silicon
  the ANE/GPU decode gap is structurally ≲1.6× before dispatch effects — which is why our Core
  AI-route ANE≈GPU decode parity with per-model sign flips is plausible (our read; cf.
  [`coreai-vs-mlx-speed.md`](coreai-vs-mlx-speed.md)).
- **Encoders/vision are the engine's side**: single-sentence encoder 4.4× faster than GPU; GPU
  throughput crossover at batch ~23 (encoder block) / ~6 (self-attention block); vision convolution
  never crosses on either axis, batch 1–256 (3.6–5.7× speed, 6–10× energy) [M]. The measured form of
  the prefill/vision-tower-on-ANE, decode-on-GPU split.
- Chunked/batched prefill: 2.3–5.9× latency reduction, bit-equal to serial [M]. Speculative decoding
  on a TinyLlama hybrid: 35 → up to 128 tok/s (repetitive) / 42–56 (factual), output-identical [M].
- Per-projection fp16 placement of a hybrid decoder (ch 14, M1) — placement follows *position*
  (what is downstream), not per-op error:

  | Survives fp16 on-engine | Fails fp16 (kept in wider precision off-engine) |
  |---|---|
  | query, key, gate, up, output-embedding | value, output projection, **down-projection** (~3% err at contraction 5632 → flips greedy argmax); fp16 residual stream across deep stacks |

- Resident KV cache = compile cache as input+output, then alias the output buffer onto the input
  buffer; host sends only the new row + slot one-hot per step (the mechanism our stateful KV path
  rides; cf. [`stateful-kv-cache.md`](stateful-kv-cache.md)).

## Hard caps and traps (ch 4, 14, 19)

- ⚠️ **~128 loaded programs per process** (next load fails `GetANEFModel: must re-compile`) and
  **127 in-flight requests per program** [M]. Bounds multi-shape / many-function serving.
- ⚠️ **Width-offset slice saturates to ±inf above 4094 on A13/A14-generation parts** (M1/M2
  measured): a last-axis slice with nonzero start offset routes through a ×16 Q.4 crop-DMA that
  clamps at fp16 max (65504/16 = 4094). Zero start offset avoids the path entirely; A15+ takes a
  plain-fp16 route [M]. Debug pointer for inf/NaN that only reproduces on older devices.
- **On-chip working set 2 MB (M1) → 4.72 MB (M5)** [M]. Crossing it flips a layer from
  compute-bound to bandwidth-bound in one step (12 → 4.8 TFLOP/s on the probe matmul). First thing
  to tune; the root number for chunk/tile sizing (our read).
- **Dispatch floor ~0.23 ms/eval (M1)**; fusing 32 layers into one program amortizes to ~6.3 µs/layer,
  batching 512 to ~1.5 µs/sample [M]. Fusion > everything else below the compute roof.
- ⚠️ **After a failed ANE compile, wait ~15 s** before the next one: a failure restarts the shared
  compiler service, and a burst of failures faster than the restart interval stalls unrelated
  compiles [M].
- Axis extents: 16384 per axis through A15 (channel axis 65536 already on M2); A16 raises the limit
  to 65536 [M/P]. A 150k-class vocab exceeds every generation's cap → the logits split in our
  recipes is permanent, not a workaround (our read).
- Dynamic-weight convolution (weight as runtime tensor) is closed above batch 1 — batch ≥2 crashes
  the compiler service [M].
- Symbolic/flexible shapes never reach the engine on the direct route (entitled loader feature);
  one program per concrete shape, so pad / bucket / recompile [M].

## Family map and capability staircase (ch 12, 34; App. D)

- **M(n) = H(n+12)**: M1=H13 … M5=H17. **A17 Pro-class and the M5 are the same H17s 16-core
  target.** One compiler binary builds all 28 targets; per-target data tables differ. A program
  compiled for its floor generation runs on everything above it.
- Staircase (each generation adds one thing, then stops):
  **A13** base set incl. fused attention, softmax, layer norm, all reductions →
  **A14** texture-engine samplers (resize / crop-resize / resample / affine / hardware gather
  on-engine; M1 has none) + cross-die addressing →
  **A15** native sin/cos, dropout/random, global argmin/argmax →
  **A16** axis limit 16384→65536, fp16 kernel-width ceiling 13→15 →
  **A17/A18 add no operation** — core count only. "A13→A16 was the last capability expansion."
- Compiler targets include 32-core (H17c) and 64-core (H17d, Ultra-class) A17-generation parts; fp8
  datapath is H18-only [P].
- Cross-chip fp16 divergence is ≤1 ULP on three of four axes; the only finite→infinity axis is the
  slice saturation above. M1 and M2 are numerical twins; drift appears only at M5 [M].

## Power and bench-protocol facts (ch 10) — rules for a fair ANE bench

- **Idle is rail-off (~0 W)**, not clock-gated. Consequences: ⚠️ the first call after a **>5 s idle
  gap pays ~260 ms re-wake** (~123× steady p50); warmup is a ~3-call ramp (7.6 → 4.6 → ~2.15 ms
  steady on the probe) [M]. A bench must warm up, then keep sub-second cadence — or knowingly charge
  the re-wake to the first token.
- **No thermal throttle** on M1 over 3.5 min sustained compute-bound load: flat ~8.1 TFLOP/s at
  ~5.5 W, drift <3–4% [M]. (Engine exposes power + on/off only — no frequency/voltage telemetry.)
- Engine leads GPU on energy on every substantial class even where it loses on speed: conv stack
  **14.5× (M1) / 12.9× (M2) / 13× (M5)** GFLOP/s/W; ViT-B/16 forward 10.5× mJ/inference; large-4096
  GEMM 4.4 W vs GPU 32.5 W while the GPU is ~2× faster [M]. Power scales with utilization: ~0.9 W at
  the dispatch floor → ~4–6 W compute-bound.
- These are the protocol roots for a tokens/joule instrument: subtract idle (engine idle is a true
  zero), report warmup separately, pin cadence, and expect the energy edge to be widest on
  compute-bound encoder/vision work, narrowest at dispatch-bound decode (our read).

## What this does NOT change

- Zoo strategy stays stock GPU-pipelined for decode ([`pipelined-engine.md`](pipelined-engine.md));
  the paper's verdict table reinforces it with measurements.
- ANEForge is a Mac-side research instrument at most — unentitled private API, version-fragile, not
  App Store material.
- Adjacent literature if a tokens/joule lane opens: the paper cites concurrent ANE LLM work
  ("Orion", 2026), on-device LLM roofline benchmarking (Bi 2026), and sustained mobile NPU-vs-GPU
  LLM studies (Tummalapalli 2026).
