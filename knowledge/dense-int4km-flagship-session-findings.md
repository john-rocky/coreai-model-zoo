# Dense-path int4km coverage + flagship Qwen3.6-35B — session findings & methods (2026-07-01)

> Companion to [[flagship-full-tuning-stack]] (the plan) and [[spec-decode-design]] (Axis-2). This doc
> records the **measured results, methods, and hard lessons** of the dense-path-coverage lever (kernel
> "Axis-1"), from de-risk to the flagship 2.18× measurement, so it's fully reproducible later.

## 1. The lever (what & why)

Flagship MoE decode is weight-bandwidth-bound. The shipped `metalize_moe` (gather_qmm) kernelizes ONLY the
routed-expert FFN; the **dense path stays on MPSGraph** (lm_head + attn q/o + shared expert, int8 or fp16).
**Lever = wire the proven fused int4km matvec (`gemma4_metal_mlp.build_fused_int4km_kernel` /
`gemma4_metal_attn_int4km.MetalInt4KMLinear`, gemma4-measured 2.9× int8 / +1.43× int4km, 8/8 EXACT,
AOT-surviving) into that dense path.** lm_head is the single biggest per-token matvec (vocab×hidden);
attn q/o are the next; k/v stay fp16 (N small — "small-N matvecs never pay", the Mac lesson).

## 2. Method — the export variants (reproducible)

- **LFM2.5-8B-A1B**: `conversion/export_lfm2_moe_dense_int4km_decode_pipelined.py <int8km|int4km>`
  (committed bab5fa7). = baseline gather_qmm + `metalize_dense_int4km(model, kernel)` on `model.lm_head`
  and full-attn `self_attn.{q_proj,out_proj}`.
- **Qwen3.6-35B-A3B**: `conversion/export_qwen3_6_moe_dense_int4km_decode_pipelined.py int4km`.
  = experts int4km (gather_qmm km4) + dense int4km on `model.lm_head` + full-attn `self_attn.{q_proj,o_proj}`
  (21 matvecs: lm_head + 10 full-attn layers × q/o). Key: add lm_head + attn q/o to the **int8-quant skip
  list** (keep fp16) so `MetalInt4KMLinear` can palettize them itself.
- **BUG FIXED (needed for both)**: lfm2/qwen attention forwards read `self.q_proj.weight.dtype` to cast
  the input; `MetalInt4KMLinear` had no `.weight` → export failed. Fix = added a `weight` property
  returning the codebook `cb` to `gemma4_metal_attn_int4km.MetalInt4KMLinear` (isolated with a toy repro:
  tie/head were innocent; the attn `.weight.dtype` access was the culprit).

## 3. Results (measured)

| evidence | number | method / file |
|---|---|---|
| lm_head int4km **per-op** vs fp16 @vocab=248K (flagship's exact shape) | **2.77×** | `ondevice/_dense_int4km_microbench.py` (single-op benches are round-trip-floor-confounded at small N; lm_head is big enough to dominate) |
| byte-audit ceiling (config-only) | Qwen3.6 **~1.97×**, GLM-4.7 ~1.34× | `ondevice/_flagship_dense_coverage_audit.py` (GLM cross-checks to its known 3.58 GB/tok) |
| **LFM-8B on-device (A19, PipelinedBench)** decode | **1.23× sustained / 1.43× avg** | thermally-matched PB_N=6; baseline int4-experts+dense-fp16 vs #2 int4-experts+dense-int4 |
| LFM-8B on-device **quality** | **PASS** (33/48 greedy match, coherent, 25+17=42 correct) | reasoning prompt via `oraclePrompt`; the numerics degenerate-prompt "1/24" was an artifact |
| **Flagship Qwen3.6-35B (Mac M4 Max GPU)** decode | **2.18×** (baseline 2.79 → #2 6.08 tok/s) | `ondevice/_qwen36_mac_bench.py`; #2 (experts+dense int4) vs shipped baseline (experts+dense int8, `qwen3_6_..._sym8_gather`) |

**The flagship 2.18× exceeds the ~1.97× audit projection** — and per-step fixed overhead *compresses* the
ratio, so the true byte-read win is ≥2.18×. This beats the prior gather_qmm result (Qwen3.6 2.1× over the
32× over-read; [Qiita](https://qiita.com/john-rocky/items/d687d2c6fcc4f3e70a82)) because #2 stacks
experts int8→int4 AND dense int8→int4 on top of gather_qmm.

## 4. Hard limits & methods learned (the expensive lessons)

- **⚠️ HF download slowness = `hf_xet` bug, NOT a rate-limit.** Symptom: starts ~14 MB/s then stalls
  (esp. near 99%). **Fix = `HF_HUB_DISABLE_XET=1`** (plain HTTP, stable full speed). HF's actual
  rate-limit is a 5-min fixed window (header `ratelimit-policy: fixed window;resolvers;q=12000;w=300`),
  quota 12000 req/5min — nowhere near hit. Restarting `snapshot_download` repeatedly LOSES `.incomplete`
  progress; let one run finish. [xet-core #789 / huggingface_hub #3580].
- **⚠️ On-device (A19) model-size ceiling ≈ 5-6 GB (int4-8B-class).** LFM-8B (5 GB) runs; **Qwen3.6-35B
  int4 (18 GB) → `signal 9` (jetsam OOM)** on the iPhone 17 Pro's ~12 GB RAM (killed during the ~26-min
  cold compile). Flagship 35B **cannot run on the phone** — bench it on the Mac.
- **✅ Mac-GPU method for these decode bundles (macOS 27):** raw `rt.AIModel.load(aimodel,
  SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu()))`. It **spews
  `ANECCompile() FAILED / MLIR MPS to ANEC conversion failed` (dozens) — these are NON-FATAL**: MPSGraph
  falls back to GPU and runs. Earlier I killed a run on the first ANE error (wrong call). The Mac has the
  RAM the phone lacks. There is **no GPU-only spec** (`allowed_compute_unit_kinds` is a read-only
  property; only `default`/`cpu_only`/`from_preferred` exist), so you can't suppress the ANE attempts.
- **⚠️ Absolute tok/s from the quick Mac driver are ~10× too slow** (ANE-retry + per-step
  re-specialization overhead): 2.79/6.08 are NOT real speeds (the real Qwen3.6-35B is tens of tok/s via
  the proper engine). **Only the RATIO (2.18×) is valid** (both bundles share the same slow path).

## 5. Quality — the honest state

- **The dense-int4km lever itself is quality-safe** (LFM-8B: coherent, 25+17=42 correct, 33/48 greedy).
- **Flagship int4 degrades quality** — the shipped-recipe article already documents "non-QAT int4 quality
  degradation / ~12 token flip / 41" on Qwen3.6. This session's flagship greedy check was **inconclusive**
  (the quick Mac driver produced garbage for BOTH int8 and int4 = a driver bug, not int4). So the 2.18×
  is a **speed win at a known int4 quality cost**.
- **The quality-safe path to the flagship speedup = FP4 (E2M1) or QAT-int4** (the "int4 answer" —
  int4-speed at int8-quality; fp4 numerics de-risked: fp4 ≈ int8 perplexity).
  **CORRECTION (per [`tensorops-quantized-kernels.md`](tensorops-quantized-kernels.md) §"A19 DEVICE A/B"
  + §"fp8/fp4"):** FP4's flagship win is DECODE-bandwidth (¼ the weight bytes, like this int4 lever), which
  **rides an fp4/int4 MATVEC kernel (like `int4km`), NOT the TensorOps `matmul2d`** — so it is **NOT
  blocked** by the A19 refutation of the matmul2d *prefill* speed lever (default MPSGraph already ≈6 TFLOP/s
  there; the "3–4× prefill" was an M5 claim that doesn't hold on A19). fp8/fp4 matvec is **UNVERIFIED
  on-device** (numerics de-risked, on-device speed/integration not measured yet). Concretely: **swap the
  int4km matvec for an fp4-E2M1 matvec in this same dense-coverage lever** → same ~2× decode bandwidth win,
  int8-like quality. That + QAT-int4 (OS26-shippable) is Stream D's work.

## 6. Also-measured this session
- **⚠️ CORRECTION — prefill/FlashAttention is A19-specific, OPEN on Mac.** The claim below over-generalized
  from A19. On **M4 Max, MPSGraph prefill SDPA is only ~22% of the fp16 peak (~6.4 TFLOP/s @ S=4096)** —
  i.e. ~78% HEADROOM, NOT near-ceiling. A custom prefill kernel (simdgroup_matrix / fused FlashAttention)
  on Mac is **UNTESTED** (only the MPSGraph baseline was measured; no custom Mac prefill kernel was run).
  The A19 matmul2d refutation (`tensorops-quantized-kernels.md` §"A19 DEVICE A/B") is A19-only — the doc
  itself says "M5 desktop may show the gain." So **Mac prefill FlashAttention = a genuine open lever** (do
  the de-risk on a clean Mac-GPU window: matmul2d/simdgroup_matrix vs MPSGraph matmul at S² shapes, then a
  fused FlashAttention if the matrix path beats the default). De-risk scaffold: `_tensorops_proto/m4_speed_ab.py`.
- **The below applies to A19 (iPhone) only**: MPSGraph prefill
  SDPA is compute-bound (~22% of M4 fp16 peak, clean S² scaling, no crash); scalar q=1-decode kernels
  can't beat a matrix-tuned baseline. Needs simdgroup_matrix / cooperative-tensor. `ondevice/_prefill_sdpa_baseline.py`.

## 7. Reproduce (commands)
```
# 1. download the fp16 source (disable xet!)
HF_HUB_DISABLE_XET=1 python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.6-35B-A3B')"
# 2. export #2 (full Axis-1) and the baseline
cd coreai-models && .venv/bin/python ../coreai-models-community/conversion/export_qwen3_6_moe_dense_int4km_decode_pipelined.py int4km
.venv/bin/python ../coreai-models-community/conversion/export_qwen3_6_moe_metal_decode_pipelined.py sym8
# 3. bench on the Mac GPU (phone jetsams the 35B) — ANE errors are non-fatal
.venv/bin/python ../ondevice/_qwen36_mac_bench.py exports/qwen3_6_35b_a3b_decode_int4km_gather_dense_int4km
.venv/bin/python ../ondevice/_qwen36_mac_bench.py exports/qwen3_6_35b_a3b_decode_sym8_gather
# LFM-8B on-device A/B (fits the phone): PipelinedBench + ondevice deploy script, PB_N=6, reasoning oraclePrompt
```
bundles are coreml (NOT committed); `ondevice/` isn't a git repo (scripts live there, documented here).
Nothing pushed to HF.

---

## 8. FP4 (E2M1) dense matvec — the quality-safe twin (built + measured, 2026-07-02)

Executing §5's plan: swap the int4km dense matvec for an **fp4-E2M1** matvec — same ¼ weight bytes
(~2× decode bandwidth), E2M1 numerics → int8-level quality (fixes int4's flagship cliff). This is a
DECODE-bandwidth hand-rolled matvec, NOT TensorOps `matmul2d` (the A19-refuted *prefill* lever), so it
runs on the Mac GPU today and needs no OS27.

**Files (additive, isolated):**
- Kernel + quant + `MetalFP4Linear` + `metalize_dense_fp4`:
  `coreai_models/models/macos/gemma4_metal_mlp_fp4.py` (twin of `gemma4_metal_mlp.py`'s int4km).
- Export: `conversion/export_qwen3_6_moe_dense_fp4_decode_pipelined.py` (int4km export, dense→fp4).
- Microbench: `ondevice/_dense_fp4_microbench.py`.

**What differs from int4km (everything else identical — packing 8 codes/uint32, R/SGY tiling, dispatch):**
the dequant. int4km gathers a per-output-group 16-entry k-means codebook; fp4 maps the 4-bit code
through the FIXED universal E2M1 grid (16 constants staged in tg memory) × a per-K-block **e8m0**
power-of-2 scale (block 32). So the kernel is the AFFINE-int4 structure (scale along K) minus the bias.

**Numerics — bit-identical to the de-risked fp4:** `quantize_fp4_e2m1` uses e8m0 scale
`2^(floor(log2|amax|)-2)` + torchao `f32_to_f4_unpacked`; reconstruction `max|W_mine − torchao_fp4| =
0.0`. The Metal kernel matches its torch reference `cosKern = 1.0000` at every flagship shape.

**Speed (per-op q=1 decode, Mac M4 Max, random weights, `_dense_fp4_microbench.py`):**

| shape | K | N | fp16 ms | int4km ms | fp4 ms | fp4/fp16 | **fp4/int4km** | cos(fp4,fp16) |
|---|---|---|---|---|---|---|---|---|
| q_proj | 2048 | 4096 | 0.482 | 0.509 | 0.524 | 0.92× | 0.97× | 0.9926 |
| o_proj | 4096 | 2048 | 0.470 | 0.512 | 0.502 | 0.94× | 1.02× | 0.9919 |
| **lm_head** | 2048 | 248320 | 3.358 | 1.972 | 1.951 | **1.72×** | **1.01×** | 0.9930 |

**Result: fp4 == int4km speed (1.01× at the flagship lm_head shape)** — the ¼-byte decode-bandwidth win
is fully inherited. (Small shapes sit under the ~0.35 ms round-trip floor, so <1× vs fp16 there — the Mac
ALU-bound regime, same as int4km; lm_head is the real signal. Absolute fp16-relative numbers are noisy
run-to-run per §4; the same-run fp4-vs-int4km equivalence is the robust claim.)

**Key kernel lesson:** a naive `const float FP4[16]` indexed by a runtime code spills to stack on Apple
GPUs and made fp4 ~1.4× SLOWER than int4km (0.70× at lm_head). **Staging the 16-entry grid into
threadgroup memory** (as int4km does its codebook) turns the lookup into a fast tg-mem gather → fp4 back
to int4km speed. Extra fp4 cost vs int4km = the per-K-block scale read (~12% traffic), absorbed.

**Quality:** inherited from the de-risked fp4 (fp4 ≈ int8 perplexity); the
kernel reproduces those numerics bit-for-bit, so the flagship dense path gets int8-level quality where
int4km flips ~12/41 tokens.

**Gate status:** kernel correctness + fp4==int4km speed ✅ (microbench, real coreai export + Mac GPU).
Wiring `metalize_dense_fp4` ✅ (synthetic + real-weight truncated run through metalize). Full 35B
export/decode-bench = the remaining crank (source `Qwen/Qwen3.6-35B-A3B` is local; baselines
`..._int4km_gather_dense_int4km` and `..._sym8_gather` already in `exports/`). Reproduce:
```
cd coreai-models && .venv/bin/python \
  ../coreai-models-community/conversion/export_qwen3_6_moe_dense_fp4_decode_pipelined.py int4km
# bench fp4 vs the existing int4km (isolates dense fp4-vs-int4km) and sym8 (fp4-vs-shipped):
.venv/bin/python ../ondevice/_qwen36_mac_bench.py exports/qwen3_6_35b_a3b_decode_int4km_gather_dense_fp4
```
⚠️ truncated `--num-layers 2` export fails with a `state_names 4 vs graph 2` mismatch (the first layers
are linear-attn → no KV state); NOT fp4-related (int4km export shares it). Use the full model.

### 8b. QUALITY — the honest result: **fp4 does NOT beat int4km on the flagship dense path**

The premise ("swap int4km→fp4 for int8-level quality") came from the fp4-vs-int4-**RTN** de-risk
(fp4 +1.0% vs int4-RTN +10.2% ppl). But the flagship dense bundle uses int4-**km**
(k-means, `palettize_grouped` n_bits=4), which is already outlier-robust. Measured on the REAL flagship
weights (`conversion/quant_fp4/flagship_dense_fp4_vs_int4km_quality.py`, no forward — weights loaded from
the safetensors shards; lm_head top-1 flip on embed-rows-through-final-RMSNorm hidden proxy, S=512):

| | int8km | int4km | fp4 (e8m0) | fp4 (fp16 scale) |
|---|---|---|---|---|
| lm_head weight rel-err | 0.032 | 0.124 | 0.115 | 0.102 |
| **lm_head top-1 flip vs fp16** | **32/512** | **104/512** | **104/512** | 117/512 |

**fp4 ≈ int4km in quality (identical 104/512 flip for e8m0; fp16-scale fp4 has LOWER weight error yet
MORE flips — weight-RMSE is a misleading proxy in both directions).** Neither 4-bit scheme approaches
int8km (32 flips). **Conclusion: fp4 gives no quality advantage over int4km on the dense path** — same
¼-byte speed (§8), same ~4-bit quality. The int4 flagship cliff is a 4-bit-capacity gap vs int8, NOT an
RTN-vs-fp4 issue (k-means already fixed the RTN part). So **int4km→fp4 is a no-op** (nothing gained).

**The actual quality-safe paths to the flagship dense speedup** (unchanged from §5's other options):
(a) keep the dense path **int8** (int8km: 6% flip, ~half the byte win), or (b) **QAT-int4** (train the
4-bit to recover — the real "int4 answer", OS26-shippable). fp4's genuine
edge is only when the neural accelerator dequants it for free (TensorOps) — which is decode-irrelevant
(BW-bound) and A19-refuted for prefill. ⚠️ Caveat: flip measured on a hidden-state PROXY (embed rows via
RMSNorm), not real decode hidden; real-hidden confirmation needs a 35B forward or an on-device LFM-8B run.
But the proxy-free weight rel-err (fp4 ≈ int4km, both 3.6× int8km) already carries the conclusion.
