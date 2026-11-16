# TensorOps impact across the zoo + the pure custom-Metal-kernel win (survey, 2026-07-01)

> Applied companion to [`tensorops-quantized-kernels.md`](tensorops-quantized-kernels.md) (the WWDC26
> §330 foundation) and [`custom-metal-kernels.md`](custom-metal-kernels.md). Five read-only passes over
> every shipped/in-flight zoo model's impl (`models/macos/*.py`), measured stats, and zoo cards.
> **Nothing here is device-measured for TensorOps yet — all speedups marked ⚗️ are estimates** anchored
> on Apple's MLX-on-M5 numbers (prefill/TTFT **3.33–4.06×** on compute-bound matmul via the neural
> accelerator; decode **1.19–1.27×** = bandwidth delta only).

## The one-line finding

**TensorOps' big win is the *non-autoregressive / one-shot compute-bound forward* — diffusion, dLLM,
and encoders. The autoregressive LLM *decode* path (the zoo's historical focus) is bandwidth-bound and
gets ~nothing.** Two structural reasons LLM decode/prefill is closed to TensorOps today:
1. **Decode is BW-bound** — int8lin already reads near ½ device bandwidth; the neural accelerator adds ~1.2×.
2. **Current LLM export is pipelined S=1** → every step is a **matvec `[1,K]×[K,N]`**, not the 2D
   multi-token `matmul2d` TensorOps accelerates. A large-chunk prefill re-export would be needed (speculative).

## Cross-model impact table (⚗️ estimates)

### TIER-0 / ULTRA — diffusion-LLM
| Model | Why | Est. impact | TensorOps construct |
|---|---|---|---|
| **LLaDA-8B (dLLM)** | Non-AR masked diffusion: **full 32-layer bidirectional forward every step, NO KV, 185 ms/forward**; hot path is 100% matmul, multiplied across ~20–30 steps. Zero KV-skip escape = purest compute-bound case in the zoo. int4 already; numerics gate loose (text coherence, not token-exact). A19 device on hand. | **3–4× end-to-end** (128-tok gen 4.6 s → ~1.2 s) | `matmul2d` (int4 native dequant) + FlashAttention (bidirectional SDPA) |

### TIER-1 / HIGH — image/audio diffusion + encoders (one-shot, compute-bound, no decode tail)
| Model | Why | Est. impact | Construct |
|---|---|---|---|
| **FLUX.2 klein 4B** | DiT 25 blocks × 4 steps; ~3.5 s of 4.25 s/step is DiT matmul. int4 already. | step 1.2–1.8× (~17 s → 11–13 s) | matmul2d(int4) + VAE conv |
| **MiniCPM-V vision tower** | SigLIP ViT, compute-bound one-shot; **measured device latency (warm 42–82 ms)**; int8 shipped = greenfield. (LLM decode tail = separate, BW-bound, not a target.) | warm 1.5–2× | matmul2d(int8/fp16) |
| **Qwen2.5-Omni audio encoder** | 32-layer Whisper-style enc, static shape, fp16, ZERO custom kernels; conv frontend + attention. | enc 3–4× (0.18 s → ~0.06 s) | matmul2d + Conv |
| **Qwen3-ASR AuT encoder** | 24-layer audio enc, windowed attn + conv frontend, fp16. | enc 3–4× | matmul2d + Conv |
| **Whisper encoder** | 32-layer enc, one-shot compute-bound (decoder = BW-bound, skip). | enc 3–4× | matmul2d + FlashAttention |
| **Qwen3-VL / Gemma4-VL towers** | Same compute-bound ViT profile as MiniCPM (device unmeasured). | warm 1.5–2× | matmul2d |

### TIER-2 / MED
| Model | Caveat |
|---|---|
| **Stable Audio 341M** | 8-step DiT multiplies the win, but small model = small absolute (50 ms → 25–35 ms/step). |
| **VoxCPM2 diffusion (LocDiT) + vocoder** | 12L×10-step diffusion is compute-bound but **can't be quantized** (quality); vocoder is a one-shot conv tail. |
| **Depth Anything 3 / RF-DETR / Unlimited-OCR vision / ColModernVBERT doc** | **ViT backbone benefits**, but ConvTranspose heads (depth/SR), deformable-gather head (RF-DETR ~40%), and fp32 paths do **not** → partial win (~1.5–2×). |

### TIER-3 / LOW · out of scope
| Group | Why not |
|---|---|
| **All LLM decode** (Qwen3.5/3.6, Gemma4, Nanbeige, MiniCPM5, LFM, all MoE) | Decode BW-bound (~1.2×); current export S=1 matvec ⇒ matmul2d doesn't apply. |
| **MoE** (Qwen3.6-35B, LFM-8B) | Win already captured by `gather_qmm`; expert gather is not a TensorOps op. |
| **Absorbed-MLA (GLM-4.7)** | A **custom-kernel-architecture** problem (cross-head staging), not TensorOps — see below. |
| **Ternary** (BitCPM/BitNet/BitVLA) | Ternary {-1,0,+1} is **not a TensorOps dtype** (int4+ only). |
| **SinSR / AdcSR** | Conv-heavy, fp16-optimized / fp32-locked — not matmul. |
| **Embedding / Reranker** | Small text encoders, already fast on stock GPU. |

### Orthogonal lever (size, not speed)
**Native int4/fp4 dequant re-test** (OS 27 E2M1 / E8M0 block scales). The zoo's int4 collapse was a
*hand-rolled dequant* numerics failure; HW dequant is a different numerics path. "One conversion +
PSNR gate" on FLUX / LLaDA / a MoE could unlock smaller footprints (→ iPhone fit) without QAT.

## THE pure custom-Metal-kernel win (no TensorOps, no new Apple HW) — Absorbed-MLA cross-head staging

This is the genuine hand-rolled-MSL win on the table, independent of the neural accelerator / OS 27.
Source: `MLA_KERNEL_BREAKTHROUGH.md`, `ABSORBED_MLA_STATE.md`.

- **What**: a 2-pass MQA-staged flash-decode for Multi-head Latent Attention. Today's kernel
  (`mla_metal_sdpa.py`) is per-head, so each of H heads re-reads the shared latent → `H·S·576` global
  reads = **0.78× (slower than naive), crashes at ctx≥512**. The lever: stage each KV tile **once in
  threadgroup memory**, all H heads read from tg-mem ⇒ `S·576` reads (≈ H≈20× less latent traffic).
- **Why it matters**: (1) pure zoo kernel craft, ships on current Mac/A19 — no waiting on Apple HW;
  (2) **write-once, accelerate a whole class** — the kernel is config-driven and already proven on GLM
  (H20) *and* DeepSeek-V2-Lite (H16) dims; (3) it's the **moat for the 2026 frontier**, which is all
  **MoE+MLA** (GLM-4.7/5.x, DeepSeek V2/V3/V4, Kimi K2.x, Mistral 3 Large) — a class dense porting
  structurally can't reach. MoE is already solved by `gather_qmm`; **MLA is the one open lever left.**
- **Payoff**: GLM-4.7-Flash reaches dense-model speed at long context + ~17.8× KV shrink (576 vs 10240
  elems/token) = on-device long-context. Forecast (⚗️ calibrated roofline): ~neutral ≤1K, **~1.8–2.0×
  @4K, ~2.5–2.9× @8K**.
- **Status / plan**: math proven (fold cos 0.99999987), kernel correct-but-per-head. 4-step plan in
  `MLA_KERNEL_BREAKTHROUGH.md`: Step 0 combine cache to one `[.,576]` state (fixes ctx≥512), Step 1 the
  2-pass cross-head-staged kernel (main `tg=(32,H)` + merge), Step 2 int8 the W_UK/W_UV lifts, Step 3
  one 60 GB export + bench at ctx∈{128…8192}. **Ship gate: ≥ naive across ctx, ≥1.5× @≥4K, token-match
  vs GLM oracle.** Don't claim a win until the bench shows one.

## Recommended order
1. **Absorbed-MLA staging kernel** (pure custom-Metal, frontier moat) — the headline win; half-built.
2. **LLaDA-8B TensorOps `matmul2d`** (highest TensorOps multiplier; profile the 185 ms forward for
   matmul-boundedness first, then drop one TensorOps layer and compare on A19).
3. **MiniCPM-V vision tower** — only encoder with a measured device baseline; clean TensorOps pilot.
