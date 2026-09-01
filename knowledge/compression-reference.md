# coreai-optimization (`coreai-opt`) reference — quantization & palettization

> Foundation note (API reference). Complements [`compression.md`](compression.md) (this project's LLM-specific
> empirical notes). Relevant to the **int4 / vocab-pruned head** lever in the ANE-later plan and to shrinking
> deployable assets. Sources: `coreai-optimization/README.md`, `docs/src/{introduction,quantization,palettization,utils}/*`,
> `skills/.../model-compression-exploration/references/{compression_patterns,size_estimation}.md`.

## Where it plugs in
```
PyTorch model → coreai-opt (compress) → finalize() → torch.export(run_decompositions(get_decomp_table()))
              → cast_to_16_bit_precision → coreai_torch.TorchConverter → .optimize() → save_asset() → .aimodel
```
Every compressor output is itself a PyTorch model (validate/finetune/export it). Lifecycle:
`Quantizer/KMeansPalettizer(model, config)` → `prepare(example_inputs)` → optional `calibration_mode()` /
`training_mode()` (QAT) → `finalize(backend=ExportBackend.CoreAI)`.

## Quantization (weights ±activations)
- **dtypes**: INT2/4/8 (signed+unsigned), FP4_E2M1, FP8_E4M3FN/E5M2 (limited Core AI support).
- **granularity**: per-tensor / per-channel (axis 0 for Linear/Conv/Embedding) / **per-block** (`block_size`,
  e.g. 32 along the in-features axis). Finer = better quality, more scale overhead.
  ⚠️ per-channel (axis-0) int8 Linear weights are **broken on the macOS-27-beta MPSGraph GPU
  delegate** — torch-level numerics are clean but the lowered matmul returns garbage (minimal
  head-only repro 2026-06-11, multiple shapes, sym and clipping alike); use per-block-32 there
  (see [pipelined-engine.md](pipelined-engine.md)).
- **scheme**: symmetric vs asymmetric. At int8 the gap is small (~1.5 dB); at int4 asymmetric gains +3–5 dB,
  and `symmetric_with_clipping` can add +7 dB.
  ⚠️ **That ordering does not hold on stacked MoE expert weights** — measured on Ling-3.0-tiny (128 experts,
  `[1,128,512,1536]` stacks, 2026-08-21), only the expert qscheme varied, everything else identical:
  | expert qscheme | size | meanKL | top-1 vs fp32 |
  |---|---|---|---|
  | `symmetric_with_clipping` (the default everywhere) | 4.56 GiB | 3.68e-2 | 91.0% |
  | `asymmetric` | 4.66 GiB | 3.43e-2 | 92.2% |
  | **`symmetric`** (plain absmax) | **4.56 GiB** | 3.45e-2 | **92.7%** |
  Plain absmax wins **at identical size and with no extra storage**; asymmetric ties it on KL but must carry a
  zero-point per block (+0.125 bits/param at int4 block-32). Not an int4 artefact either — the same swap at int8
  took meanKL 3.76e-3 → 3.37e-3 and oracle cos 0.998999 → **0.999684**, i.e. from failing the `cos ≥ 0.999` gate
  to passing it. The likely cause is the one already written down for the LM head in `conversion/_bundle.py:
  head_quant_spec`: clipping craters fat-tailed outlier rows, and a 128-expert stack is fat-tailed for the same
  reason a 157 k-row vocabulary is. **Treat the big-vocab-head rule as a big-expert-stack rule too**, and re-check
  it on the other MoE ports, which have all been left at the clipping default.
- **workflows**: data-free weight-only PTQ (seconds; good ≥8-bit, sometimes 4–6) → calibration (≈128 samples,
  needed for activation ranges) → QAT (full training; the only way to recover ≤4-bit).
- **modes**: graph (torchao PT2E, default; needs `torch.export`-able model; best for weight+activation) vs
  eager (`__torch_function__`; weight-only or when graph fails; supports dynamic control flow).
- **config**: `QuantizerConfig` → `module_type_configs`/`module_name_configs` override `global_config`
  (name > type > global). No-arg default = **W_INT8_A_INT8**. Presets: `QuantizerConfig.presets.w8()`,
  `.w4()` (int4 per-block 32). `.without(nn.LayerNorm, "model.lm_head")` to skip layers.

## Palettization (k-means LUT, weights only)
- **`n_bits ∈ {1,2,3,4,6,8}`**, LUT = `2^n_bits` centroids; each weight → index into LUT.
- **scalar** (1-D k-means, default) vs **vector** (`cluster_dim>1`; effective bpw = n_bits/cluster_dim).
- **granularity**: per-tensor vs **per-grouped-channel** (`group_size`). **Per-channel (group_size=1) basically
  always wins**; at per-channel, k-means beats quantization by ~15–19 dB at both 8-bit and 4-bit. Per-tensor
  palettization can be *worse* than per-channel quantization.
- **`lut_qspec`**: quantize the LUT centroids to int8 → enables W_INT8-A_INT8 execution (a fp LUT forces fp ops).
- **sensitivity-based k-means** (SqueezeLLM): weight clustering by per-weight importance from calibration grads.
- vector k-means is **non-deterministic** — seed numpy+torch before each `prepare()` (and `num_workers=1`).

## Mixed precision & joint compression
- **Mixed precision** (`utils/mixed_precision.md`): per-layer bit-widths from a layer-sensitivity sweep
  (compress one layer at a time, score by PSNR), then walk least-loss-first until a target avg-bitwidth is met.
- **Joint** (`utils/joint_compression.md`): palettize weights **first** (with int8 `lut_qspec`), then quantize
  activations on the palettized model. **Finalizable to the Core AI backend only.**
- ⚠️ When compressing a **stateful** decode core, read the export spec (reference inputs, dynamic_shapes,
  state_names) from the ORIGINAL model — the finalized model loses those methods.

## The LM head + embeddings (biggest tensors; the ANE-later lever)
- **Head** = vocab × hidden (e.g. 262144 × 1536) — largest single tensor, high sensitivity, needs **per-row
  (per-output-channel)** scales for matmul efficiency.
- **Embeddings** (and gemma4-style per-layer tables) can be multiple GB → gathered on a **front-end**, kept OUT
  of the decode-core graph.
- This project's measured floor: **int8 k-means (group 32, all projections) stays argmax-exact; int4 flips the
  next token** (both linear-int4 and k-means-int4). Gate/up MLP must be int8. Keep tied lm_head + 1-D conv (SSM)
  full precision for exactness. Embedding gather = **plain int8 per-row dequant-gather** (`q[ids].fp16*scale[ids]`);
  k-means is `F.linear`-only so it can't palettize a gather; **int4 gather has no clean macOS path → int8 is the
  embedding floor**. (So an int4 head needs a *kernel* path, not coreai-opt's F.linear quantizer — ties back to
  [`custom-metal-kernels.md`](custom-metal-kernels.md): the fused-int8 head+argmax kernel.)

## Pitfalls
- **Silent skips (divisibility)**: per-block quant / per-grouped-channel palettization silently skip layers whose
  dim isn't divisible by the block/group → those layers stay uncompressed. Check divisibility before trusting a size.
- **Silent skips (op registry) — the expensive one.** Both compressors find weights by intercepting the ops in
  their registry (`F.linear`, `F.conv*`, `F.embedding`, `matmul`, …). **`SwitchLinear` is not one of them**: it
  routes through `coreai_torch.composite_ops.GatherMM`, so a MoE model's expert weights are invisible unless the
  config names them explicitly. Measured 2026-08-21 on a real `[1,128,512,1536]` stack: `palettize_pytorch_model`
  reported 100.7 M params "palettized" in **0.0 s**, weight bit-identical afterwards, 99,649 distinct values in the
  first 100 k — versus a same-recipe `nn.Linear` that came back with `parametrizations.weight.0.{indices,lut}` and 48.
  The *quantizer* path is safe only because `presets.py` carries `_TORCH_MOE_SWITCH_LINEAR_4BIT`, a `module_state_spec`
  with 4-D `block_size=[1,1,1,32]`; the palettization recipes in `ondevice/export_qwen3_5*.py` carry no equivalent.
  On a model like Ling-3.0-tiny (experts = 88% of 7.9 B) this ships a "4-bit" bundle at ~14 GiB.
  **Never accept a compression run without a per-tensor coverage table** — the cosine will not tell you, and it lies
  in the reassuring direction: skipping the experts *improves* it (0.99988 vs 0.99604 on a 2-layer repro).
- **`RMSNormGated` crashes the stock macOS `4bit` preset.** `presets.py:_TORCH_MODULE_EXCLUSIONS` excludes `RMSNorm`
  and `RMSNormPlusOne` but not `RMSNormGated`, whose `weight` is rank 1 — `per_block block_size=32 axis=1` on a rank-1
  tensor raises `ValueError: axis 1 is out of bounds`, the hard path, not the graceful block-size skip. Reproduced on
  the bare primitive, so it hits **qwen3.5 / granite4h / nemotron_h** too; they ship via palettization, whose exclusion
  set differs, which is why nothing had taken the quantizer path on a gated norm before. Add it (and any subclass —
  matching is by exact class) to `module_type_configs` yourself.
- **Boundary layers** (first/last) are high-error — skipping them can add up to +9 dB; always ablate.
- **`‖ΔW‖/‖W‖` is not a layer-sensitivity proxy** — at least not on a stacked-expert MoE. Ranking Ling-3.0-tiny's 23
  MoE layers by forward KL vs by relative weight-space error agreed on **0/23** positions (top-6 overlap 0/6): the
  weight-space error was flat to three decimals across every layer (0.0970–0.0983) because every expert stack
  quantizes equally well. Sensitivity there is a function of *depth* (7.8× from first MoE layer to last), which a
  weight-space statistic cannot see. Budget for the forward sweep; don't substitute the cheap number.
- **MPSGraph's fused SDPA takes ONE head_dim.** Not a compression pitfall but it surfaces at the same
  stage (first engine run of a converted bundle), so it belongs next to them. A model whose `qk_head_dim`
  differs from its `v_head_dim` fails to lower:
  `'mps_spi.sdpa' op failed: query and value must have matching inner dimension but have 192 and 128`
  (Ling-3.0-tiny, 2026-08-21: qk 192 = 128 nope + 64 rope, v 128). GLM-4.7-Flash never hit it because it
  is 256/256. **Fix: store V in the KV cache zero-padded to `qk_head_dim` and slice the extra dims off the
  SDPA output** — exact, and it keeps the fused kernel. Costs KV: Ling went 60 → 72 KB/token. `models/macos/
  mla_metal_sdpa.py` records the same constraint for ABSORBED MLA and rejects padding there for a different
  reason — 576 > 512 trips the ViewOp overflow — so check your padded width against 512 before assuming this
  fix transfers.
- **`_ANECompiler : ANECCompile() FAILED ... MLIR MPS to ANEC conversion failed` is usually noise.** MPSGraph
  probes the ANE, fails, falls back to the GPU, and the run completes. Expected for blockwise int4, which the
  ANE cannot take at all (it wants palettization). Do not chase it on a GPU-targeted bundle.
- **int4 puts a MoE router on a knife edge — do not verify such a model mid-stream.** Measured on
  Ling-3.0-tiny (128 experts, group-limited top-8), controlled, one process: perturbing the hidden state
  by a relative **1e-5** flips **54 of 128 expert slots** across 11 of 16 MoE layers and drops the final
  hidden to cos 0.875, where the SAME model in fp16 flips **zero**. The cliff sits between 1e-6 and 1e-5,
  well under fp16's ~1e-3, so any two implementations differ enough to cross it. End-to-end comparisons
  are unaffected (engine-vs-torch logits 0.999997; engine single bundle vs engine 3-slice chain
  0.9999994) because each run's router sees its own numbers throughout. **Splicing a reference tensor
  into the middle of a run measures nothing** — it looks exactly like a broken conversion. If you need to
  localise a fault in a quantized MoE, compare end-to-end outputs of self-consistent runs and bisect by
  re-exporting, not by mixing tensors from two implementations.
- graph-mode export fails on dynamic control flow → fall back to eager for weight-only.

## Theoretical size
```
weight/index bytes = numel * n_bits/8           # int4 = 0.5 B/elem, int8 = 1 B/elem
scale bytes        = n_groups * 2 (fp16)         # n_groups per granularity (per-tensor=1, per-channel=shape[axis], per-block=ceil(dim/B)*…)
zero_point bytes   = n_groups * n_bits/8         # asymmetric only
lut bytes          = 2^n_bits * n_luts * 2       # palettization
total ≈ Σ(above) + uncompressed (biases, fp embeds, skipped layers)
avg_bitwidth = Σ(numel_i * bits_i) / Σ numel_i
```
Sizes hit in this project: gemma4 E2B core 7.0 GB fp32 → 3.5 GB fp16 → **1.9 GB int8**; qwen3.5-0.8B **969 MB**.
