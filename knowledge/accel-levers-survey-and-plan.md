# Accel levers — industry survey + zoo maximal-optimization plan (2026-07-01)

> **Companion docs**: [`tensorops-zoo-impact-and-kernel-wins.md`](tensorops-zoo-impact-and-kernel-wins.md)
> (per-model TensorOps impact table + the absorbed-MLA win) and
> [`tensorops-quantized-kernels.md`](tensorops-quantized-kernels.md) (WWDC26 §330 foundation).
> **Purpose**: run the four levers (custom kernel / TensorOps / spec-decode / quantization) as
> **PARALLEL work streams**; they share CoreAIChat, `_GPU_LOCK`, and the single A19 device.
> **All speedups marked ⚗️ are
> targets/estimates** (industry numbers are cited; nothing below is zoo-device-measured yet).

---

## PART 1 — Industry survey (top-3 × 4 categories, 2025–2026, cited)

### ① Custom GPU kernels
1. **FlashAttention v1→v4** — fused IO-aware attention. FA3: 75% Hopper util + FP8; FA4: 1605 TFLOP/s
   on Blackwell (~2.7× over Triton). Universal (PyTorch SDPA, vLLM, SGLang, TRT-LLM). [FA3](https://tridao.me/blog/2024/flash3/) · [FA4](https://lambda.ai/blog/flashattention-4-gives-the-nvidia-blackwell-platform-its-most-optimized-attention-kernel-yet)
2. **PagedAttention (vLLM)** — paged KV cache, 2–4× serving throughput, <4% KV waste. [paper](https://arxiv.org/abs/2309.06180)
3. **Marlin→Machete** — fused dequant+GEMM, ~4× FP16×INT4 to batch 16–32 (AWQ ~10.9× in vLLM). [Marlin](https://github.com/IST-DASLab/marlin)
- Frontier 2026: FA4, **FP4 GEMM (NVFP4/MXFP4)**, MoE grouped-GEMM (DeepGEMM/DeepEP).

### ② Hardware matmul accel (= TensorOps' industry analog)
1. **FP8 + Transformer Engine (Hopper)** — DeepSeek-V3 trained in FP8, ~2 PFLOPS. [DeepSeek-V3](https://arxiv.org/pdf/2412.19437)
2. **FP4 / NVFP4·MXFP4 (Blackwell)** — 4× over FP8, ~7× GEMM over Hopper, ≤1% accuracy drop. **gpt-oss ships native MXFP4; DeepSeek-V4 ships FP4 experts.** [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
3. **Marlin→Machete in-kernel dequant GEMM** — the direct software analog of Apple `matmul2d`.
- **Apple TensorOps / M5-A19**: `matmul2d` auto-dequant (int4/int8/fp8/fp4) on the neural accelerator.
  **MLX-on-M5: prefill 3.33–4.06×, decode +19–27%.** Third-party `cider` already gets **1.2–1.9×
  prefill via INT8 TensorOps on M5**. [Apple MLX/M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) · [cider](https://github.com/Mininglamp-AI/cider)
- **Headline 2025–2026 = FP4 on Blackwell.**

### ③ Speculative decoding (all lossless)
1. **EAGLE-3** — trained draft head fusing 3-layer features + tree verify. **3–5× (4.8× on 70B code),
   accept 0.80–0.88.** Pretrained heads exist for **Qwen3 (1.7B–235B)**, Llama, DeepSeek-distill, etc.
   **Requires training a head (cheap: ~1–2 days / 8×3090)**; Red Hat `Speculators` standardizes Qwen3
   EAGLE-3 → vLLM. [EAGLE-3](https://arxiv.org/pdf/2503.01840) · [Speculators](https://blog.vllm.ai/2025/12/13/speculators-v030.html)
2. **MTP (DeepSeek-V3/V3.2 built-in)** — self-draft head shipped in the checkpoint, ~1.8×, >80% accept,
   free at deploy. [MTP](https://deepwiki.com/deepseek-ai/DeepSeek-V3/4.4-multi-token-prediction-(mtp))
3. **N-gram / prompt-lookup** — **training-free**, 2–4× on input-grounded tasks (RAG/code/structured),
   ~0 on free chat. [vLLM n-gram](https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/)
- Vanilla off-the-shelf draft (no training) = ~2–3× baseline, the fallback when no EAGLE head exists.

### ④ Quantization
1. **GGUF k-quants (Q4_K_M)** — PTQ, most-downloaded, Apple-Silicon-friendly (Gemma official QAT ships
   as q4_0 GGUF). [GGUF/AWQ/MLX 2026](https://www.digitalapplied.com/blog/gguf-vs-awq-vs-gptq-vs-mlx-llm-quantization-formats-2026)
2. **AWQ** — PTQ, production GPU int4 default, 0.5–1.5% PPL at int4; **AWQ > GPTQ is the 2026 verdict.**
3. **FP4 (MXFP4/NVFP4)** — the defining 2025–26 trend, near-FP8 quality (≤1%); gpt-oss/DeepSeek-V4/Llama-405B. [gpt-oss MXFP4](https://github.com/openai/gpt-oss)
- **Quality champion at 4-bit = official QAT (Gemma q4_0, PPL −54%).** [Gemma QAT](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/)
- **HONEST NUANCE (decides a zoo bet)**: AWQ/GPTQ vs naive RTN at **4-bit / gs128 / large model** is a
  **surprisingly small gap (<1pt PPL)** — AWQ's edge is at 3-/2-bit and small models. So "AWQ rescues
  Qwen3.6 int4" is weak. **The real 4-bit-quality answers are (a) official QAT, (b) FP4 (E2M1)** — and
  Apple TensorOps natively dequants fp4 on A19 (OS27). [QuaRot RTN-vs-GPTQ-by-size](https://arxiv.org/pdf/2404.00456)

**Industry's three strongest cards right now**: **FP4** (quant×HW), **EAGLE-3** (spec-decode),
**FlashAttention** (kernel).

---

## PART 2 — The plan: 4 parallel streams

Each stream = "apply the lever to the models where the survey says it's effective, and optimize those
to the metal." Effectiveness mapping is grounded in the 5-bucket zoo analysis (see companion doc):
**TensorOps wins on non-AR/one-shot compute-bound forwards (diffusion/dLLM/encoders) + LLM prefill;
spec-decode wins on LLM decode; FP4/QAT solves the int4-cliff/size; the pure-MSL MLA kernel is the
frontier moat.**

### Stream A — PURE custom Metal kernel: Absorbed-MLA cross-head staging  ★ moat, no Apple-HW dep
- **Why effective**: MoE+MLA is the 2026 frontier (GLM-4.7/5.x, DeepSeek V2/V3/V4, Kimi K2, Mistral 3
  Large) that dense porting can't reach; MoE is solved by `gather_qmm`, **MLA is the last open lever**.
  Pure hand-rolled MSL, current Mac/A19 — no TensorOps/OS27 dependency.
- **Targets (ranked)**: GLM-4.7-Flash (own port + oracle) → DeepSeek-V2-Lite (cheap validation, kernel
  dims already proven H16) → GLM-5.x / DeepSeek-V3/V4 / Kimi / Mistral-3 (same config-driven kernel).
- **Maximal opt**: cross-head threadgroup staging (read latent `S·576` once, not `H·S·576`) + 2-pass
  split-K flash-decode + int8 the W_UK/W_UV lifts. Plan = `MLA_KERNEL_BREAKTHROUGH.md` Steps 0→3.
- **First de-risk (cheap)**: Step 0 (combine cache to one `[.,576]` state) + Step 1 (2-pass staged
  kernel) validated in `_smoke/test_mla_absorbed_kernel.py` at GLM **and** DeepSeek dims (maxdiff ≤1e-3,
  **no 60 GB load**). Only then the one 60 GB export + bench.
- **Gate**: staged ≥ naive across ctx, **≥1.5× @≥4K**, token-match vs GLM oracle. If it can't beat naive
  even @8K → cross-tg merge lost it; record and stop (don't ship a non-win).
- **Resource**: Mac GPU (`_GPU_LOCK`, GPU SOLO). Status today: math proven, kernel correct-but-per-head
  (0.78×).

### Stream B — TensorOps `matmul2d` on compute-bound forwards (diffusion/dLLM + encoders)  ★ greenfield, biggest multiplier
- **Why effective**: these run a compute-bound matmul forward many times / one-shot with NO BW-bound
  decode tail; the A19/M5 neural accelerator gives 3.3–4.06× on exactly this (Apple MLX/M5). Zoo has
  ZERO TensorOps kernels — this stream also **builds the reusable TensorOps kernel scaffolding**.
- **Targets (ranked)**: **LLaDA-8B dLLM (ULTRA — full 32-layer bidirectional forward every step, no KV,
  185 ms/forward, int4 already, loose numerics gate)** → FLUX.2 DiT (25 blocks × 4 steps) → MiniCPM-V
  SigLIP tower (only encoder with a measured device baseline) → Qwen2.5-Omni / Qwen3-ASR / Whisper
  encoders → Stable Audio DiT.
- **Maximal opt (stack)**: TensorOps `matmul2d` (native dequant) + TensorOps FlashAttention via
  cooperative tensors, hitting the neural accelerator. fp16 first → then int8/**fp4** native dequant.
- **First de-risk**: profile LLaDA's 185 ms forward on A19 to confirm **matmul-bound** (not the
  KV-fill/unroll trap that hits full-attention LLM prefill — LLaDA has no KV, so low risk). Then write
  ONE TensorOps `matmul2d` layer, swap it in, A/B on A19. Reference impl pattern: `cider` (M5 INT8
  TensorOps). Core AI integration = same `TorchMetalKernel` path (Sam3 demo, see custom-metal-kernels.md).
- **Gate**: per-forward latency ↓ with cos/PSNR parity; for LLaDA, end-to-end gen wall-clock.
- **Resource**: A19 device (iPhone 17 Pro, APPLE-BENCH) + Xcode 27 / Metal Toolchain 27 + OS 27 beta
  (fp4/fp8 = OS27; int4/int8 = OS26 point update).

### Stream C — Speculative decoding on the flagship LLMs  ★ most bankable decode 爆速
- **Why effective**: the ONLY lever that beats the decode bandwidth wall (verify K tokens/forward).
  Industry: EAGLE-3 3–5×. Apple stack + zoo both LACK it (a flagged gap).
- **Targets (ranked)**: Qwen3.6-27B dense (15.9 t/s = where decode speed is most needed) and Qwen3.6-35B-A3B
  → other shipped LLMs. Draft sources: (1) **n-gram/prompt-lookup** (training-free, instant, 2–4× on
  code/RAG/structured) as the zero-cost first win; (2) **vanilla draft** = shipped qwen3.5-0.8B / Qwen3-0.6B
  (no training, ~2×); (3) **EAGLE-3 head** (train via Red Hat `Speculators`, Qwen3 supported, 3–5×).
- **Maximal opt**: EAGLE-3 trained head + tree/verify-forward in the pipelined engine.
- **First de-risk (the gating prereq)**: **confirm the pipelined engine can do a verify-forward (S=K
  batch)** — it does chunked prefill, so feasible, but draft→verify→rollback wiring is new ENGINE work,
  not a kernel. Prove n-gram + vanilla-draft first (no training) to validate the verify path; then EAGLE-3.
- **Gate**: decode tok/s ↑ with output distribution preserved (lossless verify). Acceptance-rate
  measured per domain (code high, chat lower).
- **Resource**: engine work (Swift/pipelined). EAGLE-3 head training = external GPU box (not the Mac/A19).
- **Note**: this is engine+algorithm, not a single Metal kernel — sequence it where engine bandwidth exists.

### Stream D — Quantization: FP4-via-TensorOps + QAT-int4 (int4-cliff / iPhone-fit)  ★ size + the int4 answer
- **Why effective**: the zoo's int4 collapse is a non-QAT property; **the industry answer is FP4 (E2M1,
  near-FP8 quality) and official QAT**. AWQ over RTN at 4-bit/large is small (don't rely on it). FP4 is
  natively dequanted by TensorOps on A19 (OS27) — so this stream depends on Stream B's TensorOps path.
- **Targets (ranked)**: Qwen3.6 (35B-A3B MoE → size/iPhone; 27B dense) and **LFM2.5-8B-A1B** (the
  "iPhone-first 8B MoE" target) → FLUX/LLaDA (FP4 for size+the int4 re-test).
- **Maximal opt**: (a) **FP4 (E2M1) via TensorOps native dequant** — "one conversion + PSNR gate" — the
  no-training path that might hold quality where int4 RTN craters; (b) **coreai-opt QAT-int4-LINEAR**
  (official QAT, the training path) feeding the existing `gather_qmm` int4-LINEAR kernel for MoE.
- **First de-risk (cheap)**: convert Qwen3.6 / LFM-8B to fp4 (and a coreai-opt QAT-int4 pipe-cleaner on a
  shipped 3B first), run the existing int4 gather path, **gate on multi-token reasoning** (Nanbeige
  lesson: single-token survives but reasoning craters — test long chains, not just "Paris").
- **Gate**: quality (reasoning multi-token) ≥ bar AND smaller/faster than shipped int8.
- **Resource**: Mac GPU for export/gate (`_GPU_LOCK` for bench), A19 for fp4 runtime (needs Stream B).

---

## PART 3 — Cross-stream coordination + conventions

- **`_GPU_LOCK`** at `coreai-models-community/` root before any Mac-GPU run; **GPU SOLO** (concurrent CPU
  export/quant is safe). Streams A and D both want Mac GPU — serialize via the lock.
- **Single A19 device** (iPhone 17 Pro, id `A6F3E849`) = APPLE-BENCH. Streams B and D (fp4 runtime) share
  it — serialize device runs.
- **Dependency**: Stream D's FP4 runtime depends on **Stream B building the TensorOps fp4 `matmul2d`
  path first.** Start B before D's fp4 work. A and C are independent.
- **OS/HW gating**: int4/int8 TensorOps = OS26 point update; **fp4/fp8 = OS27**. Confirm device OS before
  fp4 work.
- **iOS large graphs need AOT** (`coreai-build compile --platform iOS`); device JIT SIGABRTs on big graphs.
  ⚠️ Never run iOS bundles on the Mac llm-runner (GPU/ANE
  wedge → reboot).
- **Bench truth = `ondevice/PipelinedBench`** (`com.coreai.pipelinedbench`), not chat UIs.
- English code+UI; don't commit coreml bundles or build files.
  Don't claim a win until the bench shows one.

---

## PART 4 — First steps per stream
- **Stream A (MLA kernel)**: read `MLA_KERNEL_BREAKTHROUGH.md` → do Step 0 + Step 1 in
  `_smoke/test_mla_absorbed_kernel.py` (no 60 GB load).
- **Stream B (TensorOps)**: **✅ GATE DONE 2026-07-01 — LLaDA forward on A19 is matmul-bound (S-scaling
  1.79×, ~80–89% compute; baseline S128 warm_min 385.8ms/med 450ms, S256 692.1/754ms).** GO to the kernel.
  **Kernel phase:**
  1. **M4 numerical-validation prototype** of a `TorchMetalKernel` wrapping `matmul2d` for
     `half activations × int4b_format weights (block-32 fp16 scales) → half`. Two unknowns to resolve
     FIRST: (a) can coreai-torch compile the embedded MSL at **`-std=metal4.1`**? (blockwise scale plane
     `metal::tensor_blockwise` needs `__HAVE_TENSOR_MULTIPLANE__` = 4.1; matmul2d + uniform int4 = 4.0);
     (b) build `tensor`/`tensor_blockwise` operands from the auto-generated raw buffer-pointer signature
     inside `helper_src`/`src` (the `tensor_inline`-from-raw-pointer path). Validate cos/PSNR vs a torch
     matmul on M4 (correctness is HW-independent; speedup only shows on A19).
  2. **Graft ONE FFN matmul** of the LLaDA forward to the kernel → **re-export on a macOS-26.4 box**
     (⚠️ NOT this Mac — 27 mis-converts) → AOT h18p → **A/B on A19** against the
     baselines already on device (`llada_s128`, `llada_s256`) via `ondevice/_llada_device_probe.sh`.
  - Facts/paths: API + compile facts = `tensorops-quantized-kernels.md` (§"The real matmul2d API");
    header = `iPhoneOS27.0.sdk/…/MetalPerformancePrimitives.framework/Headers/MPPTensorOpsMatMul2d.h`;
    validated test kernel = scratchpad `mm_test.metal`; harness = `PipelinedBench` PB_LLADA mode;
    integration ref = `custom-metal-kernels.md`; LLaDA quant = body int4 block-32 + int8 head.
    Projected win: TensorOps 3×@matmul → forward 2.1–2.4× → device generation ~2–3× (Amdahl floor = the
    ~79.5ms fixed term).
- **Stream C (spec-decode)**: confirm pipelined-engine verify-forward (S=K) feasibility → n-gram +
  vanilla qwen3.5-0.8B draft on Qwen3.6 (no training) → EAGLE-3 head via Speculators.
- **Stream D (quant/FP4)**: coreai-opt PTQ-`w8` pipe-cleaner on a shipped 3B → fp4 convert Qwen3.6/LFM-8B
  → existing int4 gather → multi-token reasoning gate. (Blocks on Stream B for fp4 runtime.)

**Ranked by confidence-of-win**: C (spec-decode, technique proven) > A (MLA, half-built, real risk) >
B (TensorOps, greenfield but Apple-measured 3–4×) > D (FP4 quality unproven on these models). **Ranked by
kernel novelty**: A (pure MSL moat) > B (neural-accelerator) > D > C (engine/algo).
