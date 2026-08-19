# Compute units (ANE / GPU / CPU) & on-device authoring rules

> Foundation note: the empirical do's/don'ts for making a model run correctly + fast on each compute unit.
> The single most important framing: **iOS/ANE = static-shape, BC1S, Conv2d, per-head, fp16-only**;
> **macOS/GPU = dynamic-shape, standard layout, fused, custom kernels**. These are Apple's two first-class modes.
> Sources: `coreai-models/skills/.../model-authoring/references/{neural_engine_rules,gpu_rules,common_issues}.md`,
> `.../working-with-coreai/{SKILL.md,references/guidance.md}`, and the official primitives
> `coreai-models/python/src/coreai_models/primitives/{ios,macos}/` + `export/{ios,macos}.py`.

## The three compute units
| | ANE (Neural Engine) | GPU | CPU (BNNS) |
|---|---|---|---|
| Best for | energy-efficient inference, fixed shapes, iOS foreground | large models, dynamic shapes, batch, max throughput | validation, fallback |
| Shapes | **fully static** (one fn per shape config) | dynamic OK | any |
| Layout | **BC1S** `(B, C, 1, S)` | standard `(B,S,D)` / `(B,H,S,D)` | any |
| Projections | **1×1 Conv2d** (Conv engine accumulates fp32) | `nn.Linear`, fused QKV | any |
| Attention | **per-head, sequential** (no fused SDPA) | **fused native SDPA** (all heads) | either |
| KV cache | **readonly functional I/O** (host writes), seq on **dim 4** | **stateful** (`mutable_slice_update`), seq on **dim 3** | — |
| Custom MSL kernels | **NO** (fixed ops only) | **YES** (`TorchMetalKernel`) | no |
| Precision | **fp16 only** (no fp32 literals/intermediates) | fp16 weights, fp32 intermediates OK | fp32/fp16 |

> The "ANE can't run custom MSL" row is *why* the GPU speed track exists — see
> [`custom-metal-kernels.md`](custom-metal-kernels.md) (project memory: `project_ane_vs_gpu_premise`).

## ANE authoring rules (the high-leverage ones)
- **BC1S layout** `(B, C, 1, S)`; all matmuls as 1×1 Conv2d. `neural_engine_rules.md:43-65,92-109`.
- **Conv2d not nn.Linear** — Linear falls back off-ANE; Conv2d maps to the conv engine **and accumulates in
  fp32** (the fix for fp16 matmul drift over many layers). `neural_engine_rules.md:92-109`.
- **No fp32 anywhere** — a single Python float literal (`1.0`) creates an f32 buffer and breaks ANE residency.
  Use `torch.ones(1, dtype=x.dtype)`. `.float()` is a **no-op on the ANE** (MPSGraph drops the cast). To get
  fp32 accumulation you must use an op the hardware accumulates in fp32 (Conv engine, LayerNorm kernel).
  `neural_engine_rules.md:120-134`, `common_issues.md:49-52`.
- **RMSNorm trap**: composite RMSNorm computes `mean(x²)` in fp16 → **overflows** large activations. Use the
  `[x,-x]` LayerNorm trick (`LayerNorm([x,-x]) == RMSNorm`, and the ANE runs LayerNorm with an fp32-accumulating
  hardware kernel). (This project's gemma4 fix; same root cause as Conv2d.)
- **Per-head SDPA** via einsum `bchq,bkhc->bkhq` (no reshape copies). `primitives/ios/sdpa.py:35-80`.
- **Causal mask**: shape `(1, key, 1, query)` (transposed vs GPU), masked value **`-40000.0` not `-inf`**
  (ANE softmax mishandles IEEE −inf). `neural_engine_rules.md:357-372`, `common_issues.md:12-15`.
- **RoPE as input**: precompute cos/sin outside the graph, pass as 4D `(1, head_dim, 1, S)` (in-graph
  `gather_nd` makes rank-3 → ANE rejects). `neural_engine_rules.md:375-379`.
- **KV cache = readonly I/O**: model concats past+new, returns new K/V; host writes the cache. **Cache the
  post-RoPE key** (`key_rope`), not raw — else stale keys → PSNR ~20 dB. `neural_engine_rules.md:382-427`.
- **Last dim aligned to 64 B / power-of-2**; **rank ≤ 5**; strides/dilations factored into 2s and 3s; large
  kernels decomposed (`k = k1+k2-1`). `neural_engine_rules.md:19-40,147-239`.
- **Chunked prefill** (S_q=64) for long prompts — fp16 per-token decode drifts ~5–10 dB/50 tokens.
  `neural_engine_rules.md:451-465`.

## What following every ANE rule actually buys — measured, on a non-LLM encoder

The rules above are Apple's. Until now nothing here recorded what happens when someone follows
*all* of them on a model that is not an LLM. [Rahul Rachuri](https://github.com/RahulRachuri)
re-authored the 600M Parakeet FastConformer encoder on the iOS/ANE track — BC1S between blocks,
every `nn.Linear` a 1×1 `Conv2d`, per-head sequential attention, the mel image transposed so
**time** is the width axis (the GPU track's freq-last subsampling ends at width 16, under the 64 B
granule), the two fp32 literals folded into weights, BatchNorm folded into the depthwise conv, and
`relative_k_proj(pos_emb)` baked at build time since it depends only on constants. Same fp16
weights, no quantization, no approximation. It gates: eager fp32 per-token cos mean 1.000000
against the GPU track's own re-author, engine cos 0.999833 on ANE against the fp32 oracle.

**It does not buy throughput.** iPhone 17 Pro, iOS 27, Release, AOT `h18p`, L=2885 fp16, warm
median of 20 passes, runs interleaved (a locked screen caps the GPU, so order must not decide it):

| encoder / unit | warm median | bundle | load (cold / warm) |
|---|---:|---:|---:|
| GPU-authored, `gpu` (what ships) | **113.5 / 113.9 / 114.8 ms** | 1.1 GB | 4.9 s / 1.3 s |
| GPU-authored, `neural_engine` | 180.7 / 197.9 ms | 2.3 GB | 65.7 s / 2.2 s |
| ANE-authored, `neural_engine` | 204.5 / 207.4 ms | 2.2 GB | 11.9 s / 0.59 s |

Both ANE routes lose to the GPU one by ~1.7×, and the ANE-authored graph is not faster on the ANE
than simply compiling the GPU-authored graph for it — the two ANE rows are within noise.

**It buys load.** Compiled with `--preferred-compute neural-engine`, the GPU-authored graph forms
**49 ANE regions**; the ANE-authored one forms **25**. That halving does not show up as steady-state
speed; it shows up as **65.7 s → 11.9 s cold and 2.2 s → 0.59 s warm**, which for a 1 GB encoder is
the difference between a cold start you can ship and one you cannot. Count regions with
`find <.aimodelc> -name '*ANE_region_*.mlir.bc' | wc -l`.

Measure this on a phone, not a Mac: the same bundles JIT-loaded on an M4 Max give 49.2 ms on gpu
against 183.9 ms ANE-authored on ane, a 3.7× gap against 1.8× on the phone. Harness:
`ENCBENCH_SELFTEST` in [`apps/coreai-audio`](../apps/coreai-audio) times any single-input
`.aimodelc` and reports load, first call and warm median separately.

### Why it buys load and not throughput: 25 regions are 25 submissions

Instruments `ane-hw-intervals` traces of the same encoder, one phone, same fp16 weights, same
authoring — Core AI against a Core ML conversion of the same network, exported and measured by
[Rahul Rachuri](https://github.com/RahulRachuri), who published the finding himself ([write-up](https://rachuri.me/blog/parakeet-apple-silicon/), [traces](https://gist.github.com/RahulRachuri/6761fdb6eb940bd25e4b55926925fbb4)):

| per encoder pass | Core AI, ANE-authored | Core ML, converted |
|---|---:|---:|
| ANE submissions | **25.0** | **1.0** |
| mean submission | 5.906 ms | 150.4 ms |
| ANE busy | 147.6 ms | 150.4 ms |
| median inter-submission gap | 1.125 ms | 0.569 ms |
| ANE idle across the timed window | 28.7 % | 0.4 % |

**Both routes give the ANE the same work — 147.6 ms against 150.4 ms, within 2 %. What differs is
delivery.** Core ML hands the hardware the whole graph as one job and holds 99.6 % residency. Core
AI issues one submission per conformer layer and spends ~48 ms per pass in the round trips between
them; that is where the idle 28.7 % goes. Rahul ties the gap size to the ~2.3 ms IOSurface
round-trip reported in arXiv 2603.06728, paid 25 times instead of once.

**The submission count is the region count.** 25 regions measured statically in the bundle is
exactly what the runtime issues at inference, which makes the 49 → 25 halving above one mechanism
rather than two coincidences: half the regions to specialize is the load win, and 25 submissions
instead of 1 is why there is no throughput win. It is also the ceiling on this track — an ANE
authoring pass can remove regions, but nothing available to us fuses the graph into one submission.

OS control: 26A5388g and 26A5406e are identical here — 25.0 submissions per pass, 5.922 against
5.906 ms mean. Nothing in beta 5 fuses the graph. The exports carry interval timings and one model
filename; the full `.trace` bundles are not published.

## Never express a preference over a heterogeneous allowed set

`SpecializationOptions(preferredComputeUnitKind: .cpu)` and `.gpu` declare the **same**
`allowedComputeUnitKinds` — `[cpu, gpu, neuralEngine]` — and differ only in which unit is
preferred. That preference is a partitioning input, and on some graphs it places a region on a
different unit and silently changes the numbers. `cpu_only()` collapses the allowed set to one
unit and is exact. It scales with fragmentation rather than model size, and the blast radius runs
from rounding-scale to anti-correlated. Full evidence, from two unrelated models, in
[`pocket-tts-port.md`](pocket-tts-port.md#preferredcomputeunitkind-cpu-is-a-partitioning-hazard-not-a-compute-unit-choice).

Two consequences: **use `cpu_only()` for parity work, never preferred `.cpu`**, and note that
CoreAIKit's `GraphModel(computeUnits:)` exposes only `.neuralEngine / .gpu / .cpu`, so `cpuOnly`
is not currently expressible through the kit.

## GPU authoring rules
- Standard layout, `nn.Linear`, **fused QKV** (`gpu_rules.md:132-154`).
- **Native fused SDPA** `F.scaled_dot_product_attention(...)` (`gpu_rules.md:50-65`, `primitives/macos/sdpa.py:13-28`).
- **Stateful KV** via `register_buffer` + `mutable_slice_update`, cache `[n,B,H_kv,max_S,D]` seq dim 3
  (`gpu_rules.md:189-258`, `primitives/macos/cache.py:12-54`). ⚠️ The **data-indexed** write SIGSEGVs the
  WWDC26 beta on GPU+ANE — use a shape-symint index or host-cache (see `coreai-beta-mpsgraph-kvwrite-bug.md`).
- IEEE `-inf` mask is fine on GPU; dynamic shapes + control flow OK; custom Metal kernels available.
- **MoE**: `SwitchLinear` + composite `GatherMM` (cast expert idx to uint16). `gpu_rules.md:262-276`.
  ⚠️ **Decode speed**: `GatherMM` gathers then runs a DENSE matmul — it does NOT read only the
  routed experts, so MoE decode is over-read-bound, not active-param-bound (Qwen3.6-35B-A3B
  int8 sits at ~25% of BW; see `models/qwen3.6/README.md`). The over-read traffic scales *super-linearly*
  with weight dtype: on LFM2.5-8B-A1B (the first direct Core-AI int4-vs-int8 MoE measurement)
  int8 decode = 39 tok/s (8.8 GB bundle, 345 GB/s ≈ full-read BW-saturated) vs int4 = 170 tok/s
  (5.0 GB; 848 GB/s effective > physical BW ⇒ int4 is NOT full-reading). So dropping a MoE to
  int4 buys ~4× decode here, not the ~2× the byte ratio predicts — but non-QAT int4 flips
  structural tokens (broken grammar), so int8 stays the quality floor. Engine-load only:
  raw `AIModel.load(.gpu)` of a MoE graph aborts (GatherMM→ANE); the pipelined engine's
  `expectFrequentReshapes` steers it off ANE (`models/qwen3.6/README.md`).
  ✅ **FIXED — the `gather_qmm` custom Metal kernel landed** (`models/macos/moe_metal.py`,
  2026-06-13; `ondevice/_gather_qmm_RESULTS.md`). A `coreai_torch.TorchMetalKernel` matvec
  takes the routed expert indices as a kernel INPUT and reads ONLY the top-k experts' weight
  slabs (`QP[w,n,e]`, `e = IDX[slot]` — indexed global load; the other E−k experts are never
  fetched). `MetalSwitchGLU` is a drop-in for `SwitchGLU`; `metalize_moe(model, nbits)` swaps
  every MoE layer. Reads weights k-means-palettized in-file (int8km 256-entry / int4km
  16-entry, reusing the gemma4 FFN kernel's multi-row + tg-codebook structure). Key enabler:
  rank-3 DSL buffer indexing + a data-dependent gather both lower+run on the GPU (probe
  `_moe_kernel_probe.py`). Result on LFM2.5-8B-A1B M4 Max: int8 MoE decode **39 → 141 tok/s
  (3.6×)** (the over-read removed by reading 4/32 experts), and an int4km bundle at **4.7 GB**
  (iPhone-jetsam-safe) / 162.7 tok/s that **RUNS on the iPhone 17 Pro A19 Pro GPU at ~32 tok/s
  decode = the zoo's first iPhone MoE on hardware**. Numerics: kernel == "select-from-all"
  bit-for-bit; composes with the stateful KV+conv decode graph (GPU-only by construction, so the
  old GatherMM→ANE raw-load abort can't occur). **Quality (fp32-oracle margin-rule gate, 41-tok
  paragraph; kernel bit-exact, so the quant SCHEME is the lever): `sym8` (symmetric-LINEAR int8,
  per-K-block-32 scale = the shipped int8-linear recipe) = CLEAN, +1 flip/41 at the fp16 ceiling AND
  same 140 tok/s → the Mac ship (quality AND speed). k-means int8 = +5 (lossier — don't use). int4
  is a WALL: two 4-bit schemes (k-means +12, affine-block-32 +11) both ~12 flips/41 with large
  margins → non-QAT int4 can't reach clean (needs QAT weights). So int8 → use `sym8` not k-means;
  int4 = compact/runs-on-iPhone but degraded. (An earlier "fp16-faithful" claim was WRONG — held
  only on a degenerate loop-y prompt; the gather kernel's QUALITY is set by the expert quant
  scheme, not the gather.)** `llm-benchmark` drives the bundle; `llm-runner`'s gen path hard-asserts
  on the 3-state (KV+conv) layout (CLI limit).
  ⚠️ **REFINEMENT — "sym8 not k-means" holds for top-k≥4, REVERSES for top-1** (ZAYA1-8B,
  `ZAYA1_8B_CCA_VALIDATED_UNSHIPPED.md`, 2026-06-22). The sym8-wins result was measured on top-4
  (LFM) / top-8 (Qwen3.6) MoE, where each token's FFN output is a weighted sum of k experts so
  expert-quant error AVERAGES (~/√k) and even crude linear int8 survives. **ZAYA is top-1 of 16:
  one token → one expert, error NOT averaged → `sym8` (linear) collapses** (engine skips the
  reasoning block + emits `<pad>`; diverges from fp16 at token 1), while **`km8` (k-means int8,
  256-entry codebook) recovers fp16 quality** (matched fp16 29 tokens token-exact). So: **top-k≥4 →
  sym8; top-1/low-k → km8** (k-means fits outlier expert weights that linear int8 clips). Two km8
  gotchas hit on the way: (a) **`moe_metal.py` `_proj` had a km8 bug** — `k_pad = qp.shape[2] *
  (4 if sym8 else 8)` lumped km8 (4 bytes/uint32, like sym8) with km4 (8 nibbles) → K_pad=2K einsum
  mismatch; FIX = `(4 if scheme in ("sym8","km8") else 8)` (km8 was unusable zoo-wide before this).
  (b) `MetalSwitchGLU`'s eager torch path is unreliable (garbage on MPS) — judge schemes ONLY via a
  real export + engine run, never eager-MPS. Best-quality fallback when even km8 is risky and the
  model is Mac-only: **skip metalize entirely → plain fp16 SwitchGLU / dense GatherMM** (runs on the
  pipelined engine, no ANE-abort; ZAYA 27 tok/s vs km8 49, zero quant loss = the Mac quality ceiling).
- **Memory-efficient load** (7B+): meta-device init + `load_state_dict(assign=True)` + per-layer streaming. `gpu_rules.md:279-297`.

## macOS vs iOS export (the official split)
`export/pipeline.py` picks dynamic (macOS) vs static (iOS).
| | iOS (`export/ios.py`) | macOS (`export/macos.py`) |
|---|---|---|
| Shapes | static buckets (query `[8,16,64]` × cache `256,512,1024,…`) | dynamic `torch.export.Dim` |
| KV | `state_names` + `in_step` data-tensor write + IOSurface/interleave | `state_names` + shape-symint offset |
| Engine | `CoreAIStaticShapeEngine` (host owns KV NDArray, passes state views each step) | `CoreAIPipelinedEngine` (GPU) |
| Target | Neural Engine | GPU |

## Runtime compute-unit selection — auto-derived from STRUCTURE, preferred-not-forced, overridable
The official runtime does NOT hard-pin a compute unit; `export/ios.py`/`export/macos.py` bake none. Instead the
Swift runtime probes the model's **structure** and derives a *preference* (`CoreAIShared/Runtime/ModelStructure.swift:57-66`):
- **`chunkedStatic`** (the iOS recipe: chunked + static shapes) → `SpecializationOptions(preferredComputeUnitKind: .neuralEngine)`
- **`dynamic`** (single `main`, the macOS recipe) → `.gpu` + `expectFrequentReshapes`

`PreparedModel.prepare(at:)` → `probeStructure` → `AIModel(contentsOf: url, options:)` (`:137-141`). Notes:
- It's a **preference, not a lock** — the compiler places ops; AOT `--preferred-compute` **defaults to `none`**
  (compiler decides), and a "compiles but runs on CPU" case needs an explicit `--preferred-compute neural-engine`
  (`common_issues.md:109-112`). So "iOS ⇒ ANE" is the *default tendency*, not a guarantee.
- The axis is **structure, not literally iOS**: static/chunked ⇒ ANE-preferred, dynamic ⇒ GPU-preferred.
- **Overridable**: `EngineFactory` takes an `EngineOptions.variant` override; the low-level path accepts your own
  `SpecializationOptions`; AOT chooses with `--preferred-compute gpu|neural-engine|none`. So ANE is selectable, not forced.

## Verification gates (PSNR)
`working-with-coreai/SKILL.md:94-99`, `guidance.md:145-153`:
- re-authored vs source (fp16): **> 70 dB** (investigate < 60)
- compiled vs torch (fp16): **≥ 40–50 dB**
- 4-bit palettized: **~40 dB** (investigate < 30)

**Localize divergence with REAL inputs** — degenerate constant-input probes lie (they said an ANE chunk was
exact when real inputs showed it diverged from layer 1). This project's hardest-won ANE lesson.

## Decision guidance
- **ANE** when energy/battery + predictable shapes + model fits (iOS ~2 GB) + single-token latency matters.
- **GPU** when large (7B+), batch, dynamic shapes, or you need custom kernels / max throughput. macOS default.
- **CPU** for debugging/fallback only.
- This project's call: **GPU now** (custom kernels, beta-robust) **+ ANE later** (when the KV-write bug lifts +
  int4 head + AOT). (Project memory: `project_ane_vs_gpu_premise`.)
