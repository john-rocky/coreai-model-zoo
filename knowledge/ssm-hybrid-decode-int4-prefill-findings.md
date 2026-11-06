# SSM-hybrid decode, int4, and chunked prefill on Core AI — session findings (2026-07-10/11)

Cross-cutting lessons from porting Nemotron-3-Nano-4B (Mamba2 hybrid) and probing int4 on it and
on LFM2.5. The per-model tables live in `models/nemotron-3-nano/README.md`, `zoo/lfm2.5*.md`, and the
working notes in `~/code/mamba-metal-scan/coreai/*_RESULTS.md`. This file is the transferable part.

---

## 1. Measurement protocol — the most expensive lesson

A single stock-vs-kernel A/B, run **once per arm in separate processes**, has ~10–15% run-to-run
spread on this M4 Max. A 5% effect cannot survive that. In one day I published, from the *same*
configuration, both "the kernel is 1.5x slower" and "no — a fp32 boundary cast explains it, the
kernel wins 1.07x." Both were noise. The same configs, timed unpaired and once each, read 0.89x…1.19x.

**Rule.** To A/B two graphs: load both in one process, time them **alternately, ≥8 interleaved
reps**, report the **median AND spread** of the paired ratio. On device (can't co-load), alternate
`PB_MODEL` in an **ABBA quartet** so linear thermal drift cancels: ratio = (B₂+B₃)/(A₁+A₄).
Two-point slopes (L=1, L=10) and single-shot verdicts confirm whatever you already believe.

Corollary: **suspect your own harness before the platform.** Two "platform bugs" this session were
mine (an un-externalized SDPA, §2; a dtype cast at a kernel boundary, §3), and one "quality wall"
was a scheme choice (§5).

---

## 2. An un-externalized SDPA deadlocks `AIModel.load(gpu)` after `optimize()`

`export_to_coreai` defaults `externalize_modules=None` → the default composite specs
(SDPA/RMSNorm/RoPE). `export_to_coreai_with_kernels` defaults to `()` → **disabled**. Mismatch them
and any attention graph hangs `AIModel.load(gpu)` after `optimize()` — with or without a Metal
kernel (2×2: kernel+externalize loads 0.2 s, stock+no-externalize hangs). Distinct from the known
`optimize()`-hangs-on-huge-attention (conversion-guide.md); here optimize finishes in 0.8 s and the
*load* hangs.

**Rule.** Pass the **same** `externalize_modules=` list to both arms of any kernel A/B. Also: passing
a `rope` spec to a NoPE model (granite, Nemotron) costs ~0.87 ms/token — drop specs the model never uses.

---

## 3. The `blockwise_shift_scale` fast path, and a fused affine-int4 matvec kernel

`coreai.blockwise_shift_scale(data, scale, shift, …)` is the dequant op for symmetric **and**
asymmetric weight quant — identical operands. **The backend only fuses it into the matmul when the
shift is all-zero.** Measured (2-layer probe): symmetric int4 2.20 ms/step (cheaper than int8's
2.31), asymmetric int4 3.09, asymmetric with the shift buffer zeroed 1.88. A nonzero shift costs
~0.6 ms/layer → on Nemotron-4B, asymmetric int4 decodes at **3.0–3.5 tok/s vs int8's 16.0**. This is
*why* the shipped `int4km` kernel uses a k-means LUT: a codebook has no shift, so it never leaves
the fast path (and pays with quality — see §5).

The zero point factors out of the inner product, so it need not fall off the fast path:
`y = (q·s)x − (s⊙z) @ blocksum(x)`, where `blocksum(x)` is shared across the rows a lane serves.
As separate graph ops that correction is a dispatch/Linear (~0.3 ms) and cancels the win; **inside a
fused matvec it is one fma per block**. Built `int4a_kernel.py`: `acc[r] += s*(dot − z*xsum)`, 8
unsigned nibbles/uint32/lane, offset-binary Z, 32 rows/threadgroup, a lane guard so **K needs only
K%8==0** (serves hidden=3136, which int4km's K%256 cannot — 65% of Nemotron's weights).

Result: GPU vs the affine grid rel 3.6e-4. It runs the asymmetric grid at the symmetric fast path's
efficiency: **3.3× the stock asymmetric path on Mac, ~5× on device.** But over int8 it is only **+3%
Mac / +11.8% device** (paired ABBA), because the nibble-unpack ALU caps it at 50 GB/s vs int8's 61
GB/s (memory bus). On an A19 the unpack does **not** hide — the ALU is weaker in the same proportion
as the memory, so the ratio is preserved. Next levers: cheaper unpack
(`as_type<float>(0x4B000000|nib) − 8388608.0f` is 2 ops vs shift+mask+cvt = 3; vectorize the loop),
and Z as uint8 (0.750 → 0.6875 B/weight).

---

## 4. A decode-step SSM scan kernel does not pay

Nemotron/Granite Mamba2 decode is S=1: `state = state*dA + dt*B*x` (elementwise) and `y = (state·C)`
(a d_state-wide sum). No loop, nothing to fuse. Paired A/B (§1 protocol), granite-350m 32 layers:
a hand-written fused scan is **3–8% faster** than the plain torch graph (median 1.03–1.08x over
variants), not slower. Occupancy is not the lever (a 32×-more-threads simd-parallel scan bought
nothing; the 1536-thread version was *faster*). Dispatch count is not it either (two kernels/layer
were cheaper than one).

**But 3–8% does not pay** for the `metal4_kernel` fusion barrier, its grid/shape constraints, and a
second artifact to maintain. Ship the plain torch graph — for *that* reason, not "the optimizer
wins". The optimizer does not beat a hand kernel here; it beat my measurement error.

The genuine SSM win is **prefill**, not decode (§7).

---

## 5. int4 quality is model-specific, RTN cannot clear it, GPTQ is the untried lever

Teacher-forced top-1 vs the fp32 oracle at margin-clean positions (top-2 gap ≥ 0.1; near-ties are
fp16 noise either way). Every scheme below is **round-to-nearest** (RTN).

| model | what carries it | what doesn't |
|---|---|---|
| Nemotron-H 4B | **zero point** (sym b32 27/33 → **aff b16 33/33**) | block size below 32 |
| LFM2.5-1.2B | finer block + **conv-mixer excluded** (aff b8 + int8 cm → 32/33) | the zero point (aff *worse* than sym) |
| LFM2.5-8B-A1B experts | nothing RTN clears it; **gate_up is the fragile matrix** (int8 gate_up → 32/35) | — |
| all three | — | **k-means is worst everywhere** (Nemotron 22/33; LFM2.5-8B int4km +12/41) |

So there is no universal int4 recipe. **Bisect by module, not by bit-width** (Nemotron: uniform;
LFM2.5: conv-mixer; 8B-A1B: gate_up — and gate_up is 2/3 of the expert bytes, so the fragile matrix
is the one you most need at int4). **Watch the byte budget while you rescue**: a config that finally
gates at 0.81 B/weight is not worth shipping over int8 at 1.06.

RTN int4 flips 6–11 margin-clean positions on the two MoE-relevant models regardless of grid. The
untried lever — never used in this zoo — is **error-compensated PTQ (GPTQ/AWQ)**: weight-only, not
QAT, pushes each weight's rounding error into the not-yet-quantized weights, and is what
mlx-community's 4-bit checkpoints use. The fused affine-int4 kernel (§3) already runs that grid at
int8 speed, so if GPTQ clears the quality the runtime side is done.

**4-bit is a fit lever, not a speed lever.** On a bandwidth-bound decode it buys ~1.2–1.35×. Its real
job is putting an 8B-A1B MoE (int4 ≈ 6.0 GB) under the iPhone's ~6.1 GB app budget where int8 (8.5 GB)
won't go.

---

## 6. The decode bandwidth model, and one correction

Decode is bandwidth-bound: `tok/s = effective_GB/s ÷ bytes_read_per_token`. On the iPhone 17 Pro,
int8 Nemotron-4B measured 60.9 GB/s — the A19's memory bus.

**Correction that bit me:** the per-token read is *not* the bundle size. The embedding table is a
one-row **gather**, not a matmul, so it is not read per token — subtract it. Nemotron int8: 3.79 GB
(not 4.29), ceiling 15.9 tok/s, measured 16.0. The naive "bundle ÷ bandwidth" only happens to work on
a **tied**-embedding model, where the lm_head *is* the embedding table and does get read every token
(granite). Check whether the head is tied before trusting the estimate.

---

## 7. Chunked prefill — the real SSM win, exact and kernel-free

`COREAI_CHUNK_THRESHOLD=1` prefills **one token at a time**, so prefill tok/s ≈ decode tok/s and the
whole weight set is read once per prompt token. On device: a 512-token prompt costs **34.1 s** before
the first token, vs 12.5 s to stream a 200-token answer — **prefill is 73% of a 512-in/200-out turn.**
The int4a decode work moves that total by 4%.

At S=q the time-axis dependency is real but factors. Single-chunk SSD form (q ≤ chunk_size=128, state
h_in): `y_l = Σ_{s≤l} exp(A_l−A_s)(C_l·B_s) x̃_s + exp(A_l)(C_l·h_in) + D·x_l`, `x̃_s = dt_s·x_s`.
Every term is a matmul or elementwise — **no kernel**, same lesson as §4 from the other side. Gated in
fp64 against the *same mixer stepped q times*: out rel **7.2e-15** (`nemotron_prefill.py`).

M4 Max, fp16, same graph: q=1 20.2 ms (49.6 tok/s) → **q=64 95.9 ms (667 tok/s), 13.7×**; q=128 751.
The ~20 ms intercept is the weight read, the slope is the chunk's compute. Prefill reads 85 GB/s of a
270 GB/s bus — it is **compute-bound**, so quantization barely helps it and the device number will
ride the A19's FLOPs (est. ~150–250 tok/s → 512-token TTFT ~2–3 s vs today's 34 s). PipelinedBench has
a `PB_CHUNKED` host loop for the companion graph; chunks beyond chunk_size need only a decay-weighted
carry of h_out, no graph change. On an iPhone 17 Pro this form did not run: the companion graph (AOT
`h18p`) logged `ANECompile FAILED` and `MTL4CommandQueueError`, then a GPU fault rebooted the phone (2026-07-11).
The device estimate above is unmeasured.

---

## 8. What is the mobile-defactor shape

Decode ceiling = the byte floor; no architecture beats it, they only change the numerator of
`bytes/token`. Against the measured 60.9 GB/s:

- **Sparsity (MoE) is the biggest single lever.** 4B dense int4 → 21.7 tok/s; 8B-A1B int4 → 45.6
  tok/s (+110%) *and* a bigger, smarter model. The MoE wall is **RAM, not bandwidth**: 8B int4 ≈
  6.0 GB fits the ~6.1 GB app budget; int8 8.5 GB does not. This is 4-bit's real purpose.
- **Recurrence (Mamba2 / GDN / MLA / DSA) is long-context insurance, not the main lever.** It buys
  nothing at 128 tokens (16 tok/s either way); it stops KV read from dominating at 32k (hybrid −16%
  vs dense −55%). GDN currently out-ships Mamba2 (zoo: 6 GDN vs 2 Mamba2). MLA/DSA attack the same KV
  term by compression / top-k.
- **MTP multiplies everything** — k drafts verified on one weight read. transformers ships a trained
  MTP head only on `deepseek_v4` and `nemotron_h` (config), and our 4B checkpoint has no MTP weights.

Target that stacks the levers: **sparse (MoE) + 4-bit + linear/SSM hybrid + MTP, ~8B-total/1B-active.**
Nearest real checkpoints: **LFM2.5-8B-A1B** (short-conv hybrid + MoE, already runs int4km on iPhone at
4.7 GB / 31.3 tok/s — quality-gated out, see §5), Qwen3-Next (GDN+MoE, no phone-size variant),
Nemotron-3-Nano-30B-A3B (Mamba2+MoE, too big).
