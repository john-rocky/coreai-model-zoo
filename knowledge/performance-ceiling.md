# Core AI performance ceiling — what's actually tapped (honest)

> ⚠️ **2026-06-11 CORRECTION — read [`pipelined-engine.md`](pipelined-engine.md) first.** This
> page's ceiling (and the "MLX gap is structural" verdict below) was measured on a hand-rolled
> per-token `fn.run()` loop. That was the LOOP's ceiling, not Core AI's: Apple's
> `coreai-pipelined` engine runs the same weights ~3.5× faster (qwen3.5 58.5 → 204 tok/s on
> M4 Max; ~2× MLX) with zero custom kernels. The per-kernel findings below (what to kernelize,
> AOT, int8 floor, ANE-as-energy) still hold — they just apply to the hand-rolled path, which
> is now only the right path when a model can't ride the engine (e.g. Gemma 4's PLE).
>
> Foundation note so future work doesn't chase wins that aren't there. The point: **within Core AI, on the
> Mac GPU, on-device LLM decode is near its ceiling** — and the remaining levers are enablement/energy, not
> dramatic throughput. Evidence: this project's gemma4 E2B macOS measurements
> + WWDC 325 + `skills/.../gpu_rules.md`.

## General principles (reusable, not just this project)
1. **Custom Metal kernels beat MPSGraph ONLY for big memory-bound matmuls** (FFN, the 262144-vocab head).
   - Per-op kernelization of small ops is **futile** (measured: attn q/k/v/o int8 kernels were *slower*; any
     single op-class ≤1.3 ms — MPSGraph already handles small ops well).
   - **Whole-layer fusion does NOT beat MPSGraph either — CLOSED by measure-first** (SDPA fold 3.6× SLOWER,
     glue fold within-noise). Don't re-attempt it. See [`custom-metal-kernels.md`](custom-metal-kernels.md).
   - The **MLX gap (~2×) is STRUCTURAL**, not closeable by more kernel polishing: it decomposes into kernel
     coverage/efficiency (~2×, only FFN+head are custom; attention stays MPSGraph) × quant (~1.5–2×, MLX
     4-bit vs our int8) × host/framework/OS-runtime tax (~1.3×). ~80% of MLX is the in-design ceiling; the
     last ~15–25% is real OS-runtime tax you don't control.
2. **AOT (`coreai-build`) helps FIRST-RUN latency + per-shape re-specialization — NOT steady-state decode
   tok/s.** Once a shape is specialized + cached, the compute is identical. The throughput fix for the
   re-specialization tax is **fixed-shape buckets** (pad → 1 compile → reuse; ~60–80× collapse per shape);
   AOT removes the *first*-compile-per-bucket cost and the device first-run wait. So AOT = UX/enablement.
   See [`aot-and-specialization.md`](aot-and-specialization.md).
3. **int8 is the practical exactness floor for these LLMs** (int4 — linear and k-means — flips the next-token
   argmax; gate/up MLP must be int8). So quantization won't buy a big extra bandwidth win past int8 without a
   quality hit. See [`compression-reference.md`](compression-reference.md).
4. **The ANE's differentiated value is ENERGY, not peak tok/s.** Its throughput target (~34 tok/s class for a
   small LLM, and only after stateful-KV + int4 head + AOT) is *below* the Mac GPU's ~55. Pick ANE for
   battery/always-on/thermals, GPU for peak speed. See [`compute-units-and-authoring.md`](compute-units-and-authoring.md).

## Project evidence (gemma4 E2B, macOS, M4 Max)
- host-cache fixed-shape GPU decode (8/8 exact) + **Lever A** (FFN fused-int8 kernel) + **Lever #3** (fused
  int8 GPU head+argmax) → **core ~14.2 ms / ~70.5 tok/s, Swift end-to-end ~55 tok/s**.
- **Lever B (whole-layer fusion) CLOSED** by measure-first (above). int8-MPSGraph FFN was a *regression* vs
  fp16-MPSGraph (the LUT-dequant traffic > byte savings for a matvec); the custom fused-int8 kernel wins by
  reading only uint8 indices + a tiny codebook.
- Efficiency: int8 core 2.0 GB / 15.7 ms ≈ 127 GB/s ≈ **23% of M4 Max peak (~546 GB/s)** ⇒
  **efficiency/dispatch-bound, not bandwidth-bound** (≈1000 Metal dispatches/token; MLX fuses the whole layer
  to ~0). That dispatch overhead is the structural gap, and layer-fusion (the only thing that would close it)
  doesn't beat MPSGraph here.

## Decision implication
- For **peak speed**, Core AI on the Mac GPU is **near its ceiling**; going materially faster means leaving
  Core AI (MLX / llama.cpp hand-tuned Metal) — which forfeits only the ANE path. *(Correction 2026-07-24:
  an earlier revision said it also forfeits Foundation Models integration — false; the `LanguageModel`
  protocol is public and MLX plugs in. See coreai-vs-mlx-speed.md §5.)*
- So Core AI's real differentiation is **on-device privacy + ANE GPU-exclusivity and (thin, ~+8.5% vs
  MLX-GPU) energy edge + the official zero-code FM adapter**, *not* winning a GPU tok/s race — and note
  guided generation needs logits, which the GPU-pipelined fast path doesn't expose (coreai-vs-mlx-speed.md
  §5.3). Optimize for the right axis.
- Practical: **ship the GPU path at current numbers**; treat ANE as an energy play (parked on the Apple
  KV-write fix); use AOT for first-run UX + (for gemma4-iOS) to potentially **un-chunk**, not for throughput.
