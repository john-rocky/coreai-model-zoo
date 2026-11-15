# Stream D — Quantization (FP4 + QAT-int4): feasibility, first result, and plan

Companion to `accel-levers-survey-and-plan.md` (Part 2, Stream D). This doc records the
toolchain reality and the **first de-risk result**: a faithful FP4-vs-int4 quality gate.

---

## TL;DR (the bankable result)

On the pipe-cleaner **LFM2.5-1.2B-Instruct** (same family as the real target LFM2.5-8B-A1B),
weight-only quant, per-block(32), lm_head kept fp16:

| scheme | weight rel-err | ppl (reasoning corpus) | Δppl vs fp16 |
|--------|---------------:|-----------------------:|-------------:|
| fp16   | 0.0000 | 3.384 | +0.0% |
| int8   | 0.0055 | 3.389 | +0.1% |
| **int4** (sym + MSE-clip) | 0.0949 | 3.729 | **+10.2%** |
| **fp4** (E2M1, e8m0 scale) | 0.1173 | 3.416 | **+1.0%** |

**The headline (and the whole point of Stream D):** fp4 has *higher* raw weight error than
int4, yet its model-level degradation is **~10× smaller** and nearly matches int8. Weight-RMSE
is a misleading proxy — uniform int4 wins on average reconstruction but catastrophically
mis-quantizes the outlier channels that drive the forward pass; FP4's exponent preserves that
dynamic range. This empirically confirms the survey's claim that **the zoo's int4 collapse is a
property of int4 RTN, not of 4-bit**, and that **FP4 (E2M1) is "the int4 answer."**

Reproduce: `.venv/bin/python coreai-models-community/conversion/quant_fp4/fp4_vs_int4_quality_gate.py`
(MPS for the forward; FP4 fake-quant runs on CPU — torchao's fp4 kernel is not MPS-safe).

---

## 1. Toolchain reality (what exists today)

### FP4 conversion — EXISTS in `coreai-opt`
- Dtype `torch.float4_e2m1fn_x2` (E2M1, OCP MX) is in `SUPPORTED_DTYPES`
  (`coreai-optimization/src/coreai_opt/quantization/spec/spec.py:387`).
- Weight-only, 2D weights, **PerBlockGranularity block_size=32**, scale_dtype must be
  `float8_e8m0fnu` (or None → resolved to e8m0). FP4 *activation* quant is not exportable.
  (`quantization/_export_utils.py:32,310-345`; `_graph/_prepare_for_export.py:288-345`.)
- e8m0 scale formula (`spec.py:155`): `scale = 2^(floor(log2(max_abs)) - target_max_pow2)`,
  `target_max_pow2 = 2` for E2M1 (8 for FP8-E4M3, 15 for FP8-E5M2).
- Fake-quant numerics = torchao `f32_to_f4_unpacked` / `f4_unpacked_to_f32`
  (`quantization/spec/fake_quantize.py:706-714`) — **our standalone harness reproduces this
  bit-identically** (verified `max|coreai_opt − ours| = 0.0` via eager Quantizer cross-check).
- Export packs to `Float4Tensor` (uint8, 2 vals/byte) — `coreai-torch/.../_compression/_floatx.py`.

### FP4 RUNTIME — the gap = **Stream B dependency**
- FP4 weight dequant via custom op is **not yet supported in the runtime core path**
  (`coreai-torch/tests/compression/test_custom_layers.py` skip note). So FP4 *runs* on device
  only once **Stream B builds the TensorOps native fp4 `matmul2d` dequant** (A19/M5, **OS27**).
- gpt-oss MXFP4 can already be *loaded* (`coreai-models/.../test_macos_models.py`), proving the
  Float4Tensor plumbing, but not GPU/ANE dequant for arbitrary models.

> **📅 Update 2026-07-01 — the OS half of the gate is now satisfied.** macOS 27 dev **beta 2**
> (`26A5368g`, 2026-06-22) is shipping and installed here (`Darwin 27.0.0`); the OS27 TensorOps
> runtime (fp4/fp8 `matmul2d`, MX·E8M0 scale plane, coop-tensor matmul input) is **present on
> device**, and Metal Toolchain v27.1 already compiles `half × metal_fp4_e2m1_format → half`
> (see `tensorops-quantized-kernels.md` §"The real `matmul2d` API"). ⇒ the FP4-runtime blocker is
> now **purely Stream B kernel work, not OS availability**. ⚠️ Still convert on **macOS 26.4** (the
> `coreai-core` wheel mis-converts on 27); 27 = runtime / AOT only.

### int4/int8 MoE gather — PRODUCTION today (Mac GPU, no OS27)
- `coreai-models/python/src/coreai_models/models/macos/moe_metal.py`: `gather_qmm_int4aff`
  (affine int4, block 32, **better than k-means for expert distributions**), `int4km`,
  `int8sym`, `int8km`. `metalize_moe` / `from_hf_streaming_metal_moe` convert HF MoE →
  `MetalSwitchGLU`. This is the existing int4-LINEAR kernel that **QAT-int4 (D2) feeds**.

### QAT — EXISTS in `coreai-opt`
- `QATSchedule`, `quantizer.step()`, `quantizer.train()` context
  (`quantization/quantizer.py:34,524-592`). Presets `w4`/`w4_per_block`/`w8` are PTQ; QAT =
  same int4 spec + a schedule + a short fine-tune.

### Convert invocation
- `python -m coreai.llm.export <model> --compression 4bit --platform macOS`
  (`coreai-models/.../llm/export.py`). macOS default `4bit` = int4 per-block-32
  symmetric_with_clipping axis=1. MoE switch-linear preset uses 4D block `[1,1,1,32]`.
- Programmatic: `get_preset("4bit")` → `quantize_pytorch_model(model, inputs, cfg)` →
  `TorchConverter().add_pytorch_module(...).to_coreai()`.

### Env
- `/Users/USER/code/coreai/.venv` — torch 2.11.0 (`float4_e2m1fn_x2` present),
  torchao 0.17.0, `coreai_opt` importable. (coreai-models/.venv is torch 2.9 — too old for
  the cpp extensions; use root `.venv`.)

---

## 2. The de-risk experiment (D1: FP4 quality gate)

Script: `conversion/quant_fp4/fp4_vs_int4_quality_gate.py`. Fake-quantizes all decoder
`nn.Linear` weights (lm_head kept fp16) four ways and measures degradation with **no device and
no Stream B** (weights are dequantized back to fp32, so the model runs as a normal fp32 graph):

- **fp4**: E2M1 via torchao kernel + e8m0 `2^(floor(log2 amax)−2)` scale — bit-identical to
  coreai-opt's `_DefaultFakeQuantizeImpl`.
- **int4 / int8**: symmetric-with-clipping, MSE-optimal clip over
  `_SYM_CLIPS=(1.0,…,0.7)` per block — matches shipped `symmetric_with_clipping` / `moe_metal`.
- Metrics: aggregate weight rel-err; perplexity on a reasoning corpus; greedy exact-match on
  12 arithmetic word-problems (the multi-token reasoning gate).

Result = the TL;DR table above.

### Why int4 uses MSE-optimal clip here
That is the *favorable* (shipped) int4, so the +10.2% is a **conservative, best-case int4**;
vanilla int4 RTN would be worse and the fp4 gap larger. The fp4 win is therefore not an artifact
of a weak int4 baseline.

---

## 3. Caveats (honest)
1. **N=1, 1.2B.** One pipe-cleaner model. The result must be confirmed on the real targets
   (LFM2.5-8B-A1B MoE, Qwen3.6) where int4 craters harder.
2. **Reasoning-accuracy saturated** (11/12 for *all* schemes incl. fp16): at 1.2B on easy
   problems the int4 cliff does not flip the final answer. Perplexity is the discriminating
   signal here; a harder reasoning set (or a bigger model) is needed to show answer-flips.
3. **Quality only.** This gate validates the *payoff*; FP4 cannot yet *run* on device (Stream B
   / OS27). The point is to prove the payoff is real **before** investing in the runtime.
4. lm_head fp16 and block_size 32 held constant across schemes (apples-to-apples).

---

## 4. Plan / next steps (in order)

**D1-next — confirm on the real targets (Mac, `_GPU_LOCK` for MPS):**
- LFM2.5-8B-A1B (the "iPhone-first 8B MoE" target). NOTE: handle the **MoE expert weight
  layout** — experts may be fused 3D `SwitchLinear`, not plain `nn.Linear`; extend
  `target_linears` to quantize expert slabs (block along K), mirroring `moe_metal` `[1,1,1,32]`.
- Qwen3.6-27B dense / 35B-A3B (download needed). Use a **harder reasoning set** so the
  generation gate (not just ppl) shows the int4 answer-flip.

**D2 — coreai-opt QAT-int4 pipe-cleaner (the training path):**
- `QuantizerConfig` from `w4_per_block(32)` + `QATSchedule`; `prepare(example_inputs)` →
  short fine-tune in `quantizer.train()` → `finalize()`. Gate: QAT-int4 recovers ppl vs PTQ-int4
  toward int8. Feeds the existing `gather_qmm` int4-LINEAR MoE kernel — **ships on OS26, no
  Stream B / OS27 needed** (unlike fp4). This is the nearer-term shippable lever.

**Runtime (blocked on Stream B):**
- Once Stream B has the TensorOps fp4 `matmul2d` dequant on A19 (OS27), wire the Float4Tensor
  export through it and bench on `ondevice/PipelinedBench`. Gate: smaller/faster than shipped
  int8 AND quality ≥ bar.

**Coordination:** Mac GPU serialized with Stream A via `_GPU_LOCK` (use an **absolute** lock
path — a relative path in a trap removes the wrong file). fp4 runtime shares the A19 with Stream
B. OS gating: int4/int8 = OS26; fp4/fp8 = OS27.

**Strategic read:** two shippable products fall out — (a) **QAT-int4** (OS26, today's kernel,
near-term) and (b) **FP4** (OS27 + Stream B, the quality win shown above). D2 is the faster ship;
D1's fp4 result is what justifies the Stream B investment.

---

## 5. Start-here (next session) — read this first

**0. Orient.** Read memory `project_quant_d_port` + §1 and §4 above. Check no other session
   (Stream A) is on the Mac GPU: `ls coreai-models-community/_GPU_LOCK`. Stream C (spec-decode)
   is in a *separate* session — do not touch its files.

**1. First move (recommended): confirm D1 on the real iPhone target — LFM2.5-8B-A1B (MoE).**
   - The result so far is N=1 at 1.2B. Re-run the gate at the actual 8B-A1B target where int4
     craters harder — the fp4 gap should *widen*.
   - **Blocker to fix first:** `target_linears()` only catches `nn.Linear`. LFM2.5-8B-A1B experts
     are likely a fused `SwitchLinear`/3D expert tensor — inspect `model.named_modules()` and
     extend the harness to fake-quant the expert weight slabs (block along K, mirroring
     `moe_metal` `[1,1,1,32]`). If experts are skipped the gate is not representative.
   - Use a **harder reasoning set** (the 12 easy problems saturated at 11/12 for all schemes at
     1.2B) so the *generation* gate shows an int4 answer-flip, not just the ppl gap.
   - Run: `.venv/bin/python coreai-models-community/conversion/quant_fp4/fp4_vs_int4_quality_gate.py --model ~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-8B-A1B/snapshots/*`
     under an **absolute-path** `_GPU_LOCK`.

**2. Then D2 — QAT-int4 pipe-cleaner (the near-term SHIP; OS26, needs NO Stream B):**
   - `cfg = QuantizerConfig.presets.w4_per_block(block_size=32)`; add a `QATSchedule`;
     `prepared = Quantizer(model, cfg).prepare(example_inputs)`; short fine-tune inside
     `quantizer.train()`; `quantizer.finalize()`. Gate: QAT-int4 ppl recovers vs PTQ-int4 toward
     int8. Feeds the existing `gather_qmm` int4-LINEAR MoE kernel → **ships today on OS26.**

**3. Gotchas (all hit this session):**
   - torchao's fp4 kernel is **NOT MPS-safe** → run fake-quant on **CPU** (keep `originals` on
     CPU), model forward on MPS. Already handled in the harness.
   - Env = root `/Users/USER/code/coreai/.venv` (torch 2.11 + torchao 0.17 + coreai_opt).
     **NOT** `coreai-models/.venv` (torch 2.9 — too old, skips cpp extensions).
   - `_GPU_LOCK` = **absolute path**. A relative path inside a shell `trap` removes the wrong
     file (we left a stale lock this way). Verify holder pid is dead before reclaiming.
   - push / HF / model-card = **USER-GATED**. Never `git add -A` (parallel sessions). Delete
     `__pycache__` before finishing.

**4. Blocked / parallel:** fp4 **on-device runtime** needs **Stream B** (TensorOps fp4
   `matmul2d` dequant, A19/**OS27**). Do not attempt fp4 device bench until B lands. **Device OS is
   now confirmed = macOS/iOS 27 (dev beta 2, `26A5368g`, 2026-06-22)** — the OS gate is satisfied,
   so the only remaining blocker is B's kernel. Everything in §5.1–5.2 is Mac-only and needs neither
   Stream B nor OS27.
