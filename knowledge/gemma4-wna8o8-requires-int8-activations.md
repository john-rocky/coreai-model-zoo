# Gemma-4 E2B mobile QAT (wNa8o8): the weights require their int8 activation path

**Measured 2026-07-17. This is a ship-blocking finding for `gemma4-metal`.**

Google's mobile QAT weights (`google/gemma-4-E2B-it-qat-mobile-transformers`, bit-identical
to the `.litertlm`) **lose roughly half their reasoning accuracy when run with fp16
activations** — which is what every one of our engines does. Running them at *higher*
arithmetic precision than they were trained for is what breaks them.

## The measurement

Same weights, same 100 GSM8K questions, same prompt/greedy/extractor, same day:

| runtime | activations | GSM8K |
|---|---|--:|
| **LiteRT-LM** (the `.litertlm`, as shipped) | **int8 static** | **86.0%** |
| **Core AI** (our mixed-bit transplant) | fp16 | **48.0%** |

For reference, on the *other* official checkpoint (`-qat-q4_0-unquantized` → uniform int4,
no activation quantization in its recipe) the same Core AI runtime scores **88.0%**, and MLX
on the same checkpoint scores 87.0%. So the ~48% is specific to wNa8o8-on-fp16, not to
Core AI and not to quantization in general.

## It is not a bug in our port

Three **independent** fp16 implementations of these weights produce the *same wrong answers*
on the 12 questions where Core AI failed and LiteRT succeeded:

| question | gold | Core AI | MLX oracle | raw-Metal | LiteRT |
|---|--:|--:|--:|--:|--:|
| q0 | 18 | 26 | **26** | **26** | ✅ |
| q3 | 540 | 180 | **180** | **180** | ✅ |
| q5 | 64 | 48 | **48** | **48** | ✅ |
| q10 | 366 | 246 | **246** | **246** | ✅ |
| total | | 0/12 | 2/12 | 3/12 | **12/12** |

Not "similarly bad" — *identically* wrong. A shared bug across the Core AI export, the MLX
oracle (`litertlm-convert/scripts/gemma4_mixedbit_oracle.py`, a plain mlx_lm dequant+load),
and the raw-Metal kernel chain (`_metal_loop/p1_chain.py`) is not a credible explanation.
The weights themselves are verified: the official checkpoint is bit-exact against our
extraction (`max|Δ|=0.0`, 312/313 tensors byte-identical — see
`gemma4-litertlm-to-official-migration.md`).

## Why higher precision hurts — the mechanism (**PROVEN by fake-quant, 2026-07-17**)

int8 **static** activation quantization clamps activations into a learned, fixed range.
QAT trains the weights *with that clamp in the loop*, so the clamp is not a precision loss
to be recovered — **it is a learned outlier suppressor the model depends on**. Run the same
weights with fp16 activations and the clamp disappears, outliers propagate, and error
compounds across reasoning steps.

This predicts exactly what we see: single-hop recall is fine ("What is the capital of
France?" → Paris; "24/3" → 8), multi-step arithmetic collapses.

The official checkpoint ships `input_activation_scale`, `output_activation_scale`, and
`k/v_cache_scale`. **Our pipeline reads none of them.** They are not optional metadata.

### Proven twice over: two independent fake-quant harnesses (2026-07-17)

Two sessions independently added fake-quant int8 static activations
(`clamp(round(x/s),-128,127)*s`, scales read from the official checkpoint) to the pure-fp16
MLX oracle — no kernels, no other change — and re-ran the same 12
Core-AI-wrong/LiteRT-right questions. Neither harness knew of the other; both flip the
result, and one of them ran ablations to locate the mechanism:

| activations | 12-q set | GSM8K n=100 |
|---|--:|--:|
| fp16, no clamp (both harnesses reproduce the original oracle run **per-answer exactly**) | 2/12 | **48%** |
| ablation: KV-cache clamp only | 0/12 | — |
| ablation: linear in/out clamps only | 9/12 | 88% |
| harness A — linear + KV clamps (checkpoint scales only) | 10/12 | **89%** |
| harness B semantics — + the TFLite-only `per_layer_model_projection` quant (verified op-by-op vs the Section10 decode graph; A's `--ple-proj` reproduces B's 11/12 exactly — cross-validated) | 11/12 | 86% |
| LiteRT (control, true static-int8 graph) | 12/12 | 86% |

The four signature wrong answers all flip to correct (q0 26→18✓, q3 180→540✓, q5 48→64✓,
q10 246→366✓); recall stays intact. The ablations put the learned suppressor at the
**linear boundaries**: KV quantization alone recovers nothing (it even loses the two
questions fp16 got right — noise without the suppressor), while the linear clamps carry 9
of the recovered points. The residual misses (q13; q5 under harness A) are emulation noise —
fake-quant is not bit-exact int8 arithmetic; at n=100 the KV and PLE-projection refinements
are within noise (86–89%).

**The n=100 column completes the argument quantitatively.** The oracle at fp16 scores 48% —
exactly matching Core AI's 48% on the same weights. Adding nothing but the clamp takes it
to 86–89%, i.e. **the clamp recovers the entire fp16 gap**, landing at/above the true int8
path's 86.0 — consistent, since fake-quant keeps matmul interiors in fp16 (strictly more
precise than real int8 arithmetic). The fp16 degradation *is* the missing static activation
quantization; there is no second mechanism left to find. (Numbers re-scored 2026-07-20
after a scoring-normalization audit — '26.00'-style answers were previously marked wrong;
LiteRT's own 85.0 rose to 86.0 in the same audit.)

Implication for a raw-Metal int8 path: **implement the per-linear static activation quant
first** — that is where the quality lives; add KV-cache int8 for the remaining margin (it is
also the memory win). Harnesses: `litertlm-convert/scripts/gemma4_fq_oracle_gsm8k.py`
(B — doubles as the reference implementation of the semantics below) and
`gemma4_fakequant_gsm8k.py` (A — ablation flags `--baseline/--no-kv/--no-linear`, plus
`--all N` for full-set scoring; per-mode JSONs in `reports/parity/fakequant_gsm8k_*.json`).

### The exact semantics (read from the .litertlm Section10 decode graph, op-by-op)

Checkpoint scalars match the graph's quantization parameters (21/21 spot-checked on
layers 0/1). The recipe any port must implement:

- **All activation scales are per-tensor scalars, zero-point 0** (weight scales are
  per-channel).
- **Quantization happens at linear boundaries only**: fp32 → quantize with
  `input_activation_scale` → int8×int-weight matmul → requantize the result to int8 with
  `output_activation_scale` → dequantize → fp32. Norms, residuals, GELU, RoPE, softmax and
  the final logit softcap all run in float. q/k/v share one input quantize (their three
  input scales are equal); gate/up likewise.
- **`k/v_cache_scale` is int8 KV-cache storage quantization**: K is quantized after
  k_norm+RoPE, V after value_norm, at cache-write time; attention math runs in float on
  dequantized values. `v_cache_scale` = 6/127 for every layer; `k_cache_scale` is learned
  per layer.
- **`lm_head` activation scales are 0 in the checkpoint = unquantized** (float activations
  into the int2 head, then softcap) — matches the graph.
- Two gotchas visible only in the TFLite graph, not the checkpoint:
  `per_layer_model_projection` is also activation-quantized (s_in=0.038878,
  s_out=0.002353) but the checkpoint carries no scales for it (its weight is even plain
  bf16 there); and `value_norm` is a real RMS norm with weight ≡ 1.0.

## Why every gate we had missed this

`g4loop --gate` — the raw-Metal ship engine's only correctness check — is blind twice over:

1. **It tests 3 prompts**, and all three are single-hop recall: *"Why is the sky blue?"*,
   *"What is the capital of France?"*, *"Explain photosynthesis in one sentence."*
   The defect only shows in multi-step reasoning. These prompts cannot see it.
2. **It compares against `oracle_refs.json`** — generated by the fp16 MLX oracle, i.e. a
   reference carrying the identical defect. "3/3 EXACT vs oracle" proves conformance to a
   degraded reference, not quality.

Every gate in the transplant chain has this shape (P3's "quality gate — 3/3 PASS, id-exact
vs the P2 MLX oracle" included). **They are equivalence gates, not quality gates.** An
equivalence gate cannot detect a defect its reference shares — that is the whole lesson.

`g4loop` also has no free-prompt input at all (`--gate` and `--bench` only), so nobody
*could* have run a task benchmark against it before shipping.

## Consequences

- **`gemma4-metal` ships a model that is ~half as good at reasoning as it looks.** The
  +24% Mac speed and the iPhone LiteRT-parity claims stand; the quality does not.
  *(Resolved 2026-07-18 — the engine now implements the int8 activation path; see
  "Implemented" below. 48 → 73 vs LiteRT's 85.)*
- **The byte-floor argument was incomplete.** "783 MB/token vs 2.0 GB → must be faster" left
  quality out. Measured on the Core AI standard runtime, mixed-bit loses on *both*: decode
  70.6 vs 75.9 tok/s (int2 unpack eats the bandwidth saving) and 48.0% vs 88.0%.
- **"Just publish the int4 weights" does not transfer a mobile QAT model.** The wNa8o8
  weights are half of a co-designed weights+runtime product. This
  bounds what any third-party port of these weights can achieve.

## What to do

1. **Give `g4loop` a free-prompt input.** An engine whose quality cannot be measured cannot
   be ship-gated.
2. **Replace the gate prompts with multi-step reasoning**, and gate on *task accuracy vs a
   trusted reference* (bf16, or LiteRT itself), never against an oracle built from the same
   fp16 path.
3. **Re-decide the ship.** Uniform int4 from `-qat-q4_0-unquantized` scores 88.0% and is
   *faster* on the Core AI runtime (75.9 vs 70.6). Mixed-bit's case rests entirely on the
   raw-Metal engine's speed, and now has to pay for it in quality.
4. **If mixed-bit is still wanted, implement the int8 static activation path** — the exact
   semantics are pinned down above ("The exact semantics"), and the fake-quant harness
   doubles as the reference implementation. Until then these weights are mis-used by
   construction.

## Implemented (2026-07-18): the int8 activation path is now in the raw-Metal engine

"What to do" items 1, 2 and 4 are done. Fake-quant at every linear boundary
(`s * clamp(rint(x * (1/s)), -128, 127)`, per-tensor checkpoint scales, input post-norm +
output at store) plus int8 KV-cache storage quant (K after k_norm+RoPE, V after value_norm),
in the shared MSL kernels both the python harness and the Swift engine compile.

| runtime, same weights | GSM8K-100 | decode tok/s (M4 Max) |
|---|--:|--:|
| raw-Metal fp16 activations (before) | 48 | 157 |
| raw-Metal + int8-activation fake-quant | **73** | **140** |
| LiteRT-LM (true int8 arithmetic, control) | 85 | — |

The learned clamp restores +25 points at ~11% decode cost. The residual gap vs LiteRT is
the **fake-quant method itself**, not the port: three independent fake-quant
implementations (Metal reciprocal-mul 73, Metal divide 75, MLX oracle 79) sit within
binomial noise of each other at n=100, while LiteRT's true int8 arithmetic (int8×int8 →
int32 accumulate → requant) scores 85 above the whole family. Emulating the clamp in fp
arithmetic recovers most but not all of what the QAT training baked in — the per-op
int32-accumulate/requant rounding is itself part of the trained numerics. The remaining
quality lever is therefore real int8 matvecs, which is also the speed upside wNa8o8
exists for.

Two transferable engineering notes from the port:

- **A token-exact gate is not a logit-exact gate.** The python and Swift engines compile
  the same MSL through different compilers (torch `mps.compile_shader` = fast math,
  MTLLibrary `.safe`). A bare `exp()` in the SDPA kernel lowered differently — 1-ulp
  context differences on ~0.05% of lanes that token gates never surfaced, because fp16
  argmax margins absorb 1 ulp. The int8 activation grid makes argmax near-ties common
  enough that the dormant difference started forking tokens. Namespace every
  transcendental (`metal::precise::`) in cross-compiler MSL.
- **Quantize with `x * (1/s)`, not `x / s`, in hot loops.** An inner-loop divide cost 31%
  of decode; the reciprocal multiply is also what TFLite's own quantize kernels do.

## Reproduce

```bash
# LiteRT (int8 activations) — the control
python scripts/parity_gsm8k.py --which int4 \
  --litertlm ~/code/litertlm-convert/src_models/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm \
  --n 100 --max-tokens 2048 --greedy      # 86.0%
# NOTE: litert-mac-verify's --max-tokens is TOTAL context, not a generation budget, and
# undersizing it CORRUPTS output rather than truncating (25-token input at 30 → garbage).

# Core AI (fp16 activations), same weights
python scripts/parity_gsm8k.py --which coreai \
  --bundle .../exports/gemma4_e2b_mixedbit_decode --n 100 --max-tokens 1024   # 48.0%
```
See `cross-runtime-quality-benchmarking.md` for the protocol these runs use and the harness
defects that had to be fixed first (thinking-mode mismatch, budget truncation).
