# Core AI vs MLX — where Core AI is faster, where it's slower, and *why*

A structured database of every measured Core AI–vs–MLX decode comparison we have, plus the
causal decomposition of the gap. All LLM rows are **same M4 Max, same protocol** as
`mlx-lm benchmark` (Apple's `llm-benchmark` is explicitly modeled on it): 512 prompt /
1024 generation / 5 trials, release build. MLX side = `mlx-lm 0.31.3`, `mlx-community` 4-bit.

## 1. The database (decode tok/s)

| # | Model | Arch class | Core AI | MLX | CA/MLX | Winner | Engine path | Dominant factor |
|---|---|---|---:|---:|---:|---|---|---|
| 1 | qwen3-0.6b | dense | **484** | 432 | 1.12 | **CA +12%** | pipelined | dispatch-bound, not BW-bound → MLX 4-bit edge doesn't cash in |
| 2 | qwen3-4b | dense | 145.4 | 145.8 | 1.00 | tie | pipelined | — |
| 3 | qwen3-8b | dense | **94.1** | 90.0 | 1.05 | **CA +5%** | pipelined | — |
| 4 | gemma3-4b-it | dense | **141.5** | 136.3 | 1.04 | **CA +4%** | pipelined | — |
| 5 | gemma3-12b-it | dense | 55.0 | 55.1 | 1.00 | tie | pipelined | biggest dense → BW starts to matter → MLX 4-bit pulls even |
| 6 | mistral-7b-v0.3 | dense | **101.7** | 97.5 | 1.04 | **CA +4%** | pipelined | — |
| 7 | gpt-oss-20b | **MoE** | 78.1 | **100.2** | 0.78 | **MLX +28%** | pipelined, stock GatherMM | **expert dispatch: GatherMM reads ALL experts/token (over-read-bound)** |
| 8 | Qwen3.6-35B-A3B | MoE (256e/top-8) | 30.9 | ~55–70 | ~0.5 | MLX | stock GatherMM | 32× expert over-read |
| 8b| Qwen3.6-35B-A3B | MoE + **gather_qmm kernel** | **64.9** | ~55–70 | ~1.0 | **tie/CA** | custom Metal sym8 gather | **kernel reads only routed experts → gap closes** |
| 9 | LFM2.5-8B-A1B | MoE (32e/top-4) | 39 → **141** | — | — | (3.6× self) | stock → gather_qmm | same over-read fix |
| 10| GLM-4.7-Flash | **MoE + MLA** | 20.3 → **52.4** | — | — | (2.6× self) | stock → gather_qmm | MoE fixed by kernel; **MLA on all 47 layers keeps it < qwen3.6** |
| 11| Qwen3-Coder-Next-80B-A3B | MoE (512e) | ~24 | "MLX-competitive" | ~1.0 | tie | gather_qmm | BW-bound on 79GB cold weight, not GDN |
| 13| **Muse-Glimmer-30B** (text tower) | dense | **27.4** | 27.4 | 1.00 | tie | pipelined | **largest dense measured — the tie the curve predicts.** Vendor's own ExecuTorch-Metal build (MLX-native) is 24.0 = **14% behind BOTH** |
| 12| Qwen3-ASR-1.7B (audio) | dense decoder, **ANE** | WhisperKit-ANE | **MLX 2.6×** | — | **MLX** | ANE (CoreML) | ANE = energy-not-speed; MLX-GPU wins raw tok/s + WER (1.52 vs 1.71) |

Row 13 protocol differs and is stated rather than folded in: 192 generated tokens, greedy,
batch 1, **interleaved CA/ET/MLX with a 45 s cooldown between every run**, MLX side is
`mlx-vlm` 0.6.13 (`mlx-lm` 0.31.3 does not know `muse_glimmer`). Interleaving is not optional
at this size — a block-ordered first attempt had ExecuTorch decaying 23.5 → 17.4 tok/s inside
its own block, which measures the order rather than the engine. Bytes: Core AI int4hu 16.35 GB,
MLX 4-bit 18 GB, ExecuTorch k-quant-17G 17.9 GB.

**The new fact in row 13 is not the tie — it is the third column.** ExecuTorch's `metal`
backend is MLX-native by its own README, so a vendor's shipped on-device artifact running 14%
behind the runtime it wraps is the finding. "The vendor ships it, so it is the fast path" does
not hold; measure it.

Sources: rows 1–7 `apple-models-bench.md` (head-to-head matrix).

## 2. The one-line answer

**The difference is operator/architecture coverage on the engine — NOT the core engine.**
On standard **dense** transformers Core AI's pipelined engine ties or beats MLX. Core AI only
loses where the model uses an op-class the stock engine lowers *naively*:

- **MoE** → `SwitchGLU`/`GatherMM` lowers to a **dense matmul over ALL experts every token**
  (32× over-read for 256-expert top-8). MLX has real sparse expert gather. → MLX +28% on gpt-oss.
- **MLA attention** (GLM/DeepSeek) → naive materialized form does big per-token up-projections +
  full SDPA on every layer; the absorbed/latent-KV form that MLX-class runtimes exploit is hard
  to kernelize (the `c_kv` must be threadgroup-staged across heads, which our kernel didn't do).

When we replace the naive lowering with a **custom Metal kernel** (`gather_qmm`, reads only the
routed experts), the MoE gap closes: **Qwen3.6-35B-A3B 30.9 → 64.9 ≈ MLX**. So the gap was
dispatch/over-read, not the engine.

## 3. Factor decomposition (why a gap exists *at all* when it does)

The historical "MLX is ~2× faster, structural" verdict was measured on a **hand-rolled per-token
`fn.run()` loop** (~11% of BW peak, ~1000 Metal dispatches/token). That was the *loop's* ceiling,
not Core AI's. Apple's **`coreai-pipelined` engine** runs the same weights ~3.5× faster (qwen3.5
58.5 → 204 tok/s, ~2× MLX) with zero custom kernels — which is why the dense rows above tie/win.

The remaining gap, where it survives, decomposes into three independent multipliers:

| Factor | Size | Helps MLX when… | Notes |
|---|---|---|---|
| **Kernel coverage / dispatch** | ~2× | the model has uncovered op-classes (MoE gather, MLA) | dense is covered by the pipelined engine; MoE/MLA aren't → that's the whole gap |
| **Quantization byte-class** | ~1.5–2× | **bandwidth-bound** (big models, long ctx) | MLX = 4-bit affine g64; CA ships int8 (int4 flips argmax for non-QAT). Only pays off once BW-bound — see gemma3-12b tie vs 0.6b +12% win |
| **Host / framework / OS-runtime tax** | ~1.3× | always (uncontrollable) | the irreducible ~15–25% you don't own |

**Decision rule that predicts the winner:**
1. **Dense + pipelined engine** → Core AI ≥ MLX (Apple's tuned MPSGraph + async pipelined
   scheduling, and it isn't paying a Python-loop tax). The smaller / less BW-bound the model,
   the bigger Core AI's win (0.6b +12%); the bigger the model, the more MLX's 4-bit erases it (12b tie).
2. **MoE (sparse expert)** → Core AI loses on stock lowering (over-read), reaches **parity** with a
   custom gather kernel, but does **not** beat MLX — MLX's sparse dispatch is already good.
3. **MLA / exotic attention** → Core AI loses; the structural kernel (absorbed-MLA latent staging)
   is unsolved, so it stays below even Core AI's own dense models.
4. **ANE / iPhone** → not a raw-tok/s contest — and now measured (iPhone 17 Pro,
   DeepSeek-R1-1.5B, matched 4-bit bytes ANE 0.97 / GPU 0.95 / MLX 0.95 GB, cold short-chat
   median-of-3; `litertlm-convert/reports/coreai-ane-gpu-parity-addendum.md`):
   decode **ANE 83.3 / CA-GPU 75.9 / MLX 73.0 tok/s**; energy **ANE 6,144 / MLX 5,662 /
   CA-GPU 4,506 tok per 1% battery**. The ANE-vs-GPU delta sign-flips across sibling models →
   **throughput parity**, not an ANE speed win. And the ANE *energy* edge over MLX-GPU is only
   **~+8.5%** (it's +36% over CA's own GPU — MLX's GPU path is energy-efficient); the robust
   ANE win is **GPU exclusivity** (UI/rendering don't contend). MLX DOES run on iPhone (GPU,
   via mlx-swift) — *correction 2026-07-24: an earlier note here claimed it "can't run on
   ANE/iPhone at all"; only the ANE is closed to MLX*.
   (Foundation Models integration is NOT an exclusive: the `LanguageModel` protocol is public
   and MLX plugs in via `MLXLanguageModel` — see fm-provider.md; CA's edge is only the official
   zero-code adapter. And the ANE LLM path is static-shape + palettization-required +
   host-driven — no pipelined loop exists on ANE, see compute-units-and-authoring.md.)

## 4. Takeaway for porting decisions

- Porting a **dense** model to Core AI: expect tie-or-win vs MLX for free on the pipelined engine.
- Porting a **MoE**: budget a `gather_qmm` custom Metal kernel up front, or you ship at ~0.5–0.78×
  MLX. With the kernel you reach parity (the ceiling), not a win.
- Porting **MLA**: parity/win is not currently reachable; ship for coverage/quality, not speed,
  until absorbed-MLA latent-staging lands.
- The genuine, non-speed differential is **thinner than this doc previously claimed** — see §5.
  It reduces to: ANE **GPU exclusivity + a thin energy edge** (conditional: static+palettized,
  host-driven loop), AOT first-launch *control*, and the official zero-code FM adapter.
  NOT "iPhone reach" (mlx-swift runs on iPhone GPU too); NOT "FM integration" (the protocol is
  public and MLX plugs in); NOT "no bundled runtime" (see §5.1). Optimize for that axis — or
  for ecosystem positioning — not for beating MLX on tok/s.

## 5. Audited non-speed differentials (2026-07-24)

Claims in earlier revisions of this doc, re-checked against artifacts. Net: the technical
differential for *LLM execution* is thin; most of the advertised deployment gap was illusory.

### 5.1 "OS-resident runtime / nothing to bundle" — HALF-FALSE

Only the **graph compiler + executor** (`CoreAI.framework`) is OS-resident. The LLM runtime —
`EngineFactory`, the `coreai-pipelined` engine, `LanguageBundle`, on-GPU sampling, KV growth —
is Swift code from `coreai-models` that **you compile into the app** (proof: we patch it —
`apps/coreai-pipelined-extra-states.patch`; you can't patch an OS framework). MLX draws the
same line one layer lower: it bundles its tensor framework + kernels on top of OS-resident
Metal/MPS. The binary-size delta is tens of MB against GB-class model assets — noise.

OS-residency is also **double-edged** (all incidents verified in this repo): beta seed-to-seed
ABI churn kills TestFlight launches (`FoundationModels` must be weak-linked);
the OS reclaims model assets into 6-byte zeroed stubs; the
~O(p²) prefill scratch lives in the closed compiler and **cannot be fixed app-side** (MLX's
stack is fully open — every layer is fixable). "The OS owns it" = "you can't repair it."

### 5.2 AOT startup control — real, but mostly self-remediation

Cold GPU specialization is Core AI's own cost (0.8B ≈ 4.8 s, 2.3 GB ≈ 29 s on iPhone;
pipelined-engine.md). AOT / `AIModelCache` gives *control over* that first-run cost; MLX's
runtime kernel JIT is light enough that it never had the problem. Deterministic first-launch
is a genuine product knob, but don't sell it as an advantage *over MLX*.

### 5.3 Reverse differential: logits / guided generation favor MLX

FM guided generation (`@Generable`) needs engine logits, and the GPU-pipelined fast path
**does not expose logits** (fm-provider.md). MLX exposes logits trivially → structured
generation, logprobs tooling, and sampler experiments are *easier* on MLX than on Core AI's
fast path.

### 5.4 What each side genuinely keeps

- **Core AI**: ANE access (throughput parity, ~+8.5% energy vs MLX-GPU, **GPU exclusivity** —
  the one structural fact MLX can never reach); AOT first-launch control; official zero-code
  FM adapter; closed compiler improves with OS updates (double-edged, §5.1).
- **MLX**: fully-OSS stack (every layer fixable); no conversion step → new-arch turnaround in
  days; mature 4-bit affine quant (CA ships int8 — int4 flips argmax for non-QAT); free logits
  (§5.3); no O(p²) prefill-scratch wall.
