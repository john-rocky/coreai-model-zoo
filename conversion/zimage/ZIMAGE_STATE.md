# Z-Image-Turbo → Core AI port — STATE

**Model:** Tongyi-MAI/Z-Image-Turbo (6B, Apache-2.0). Single-Stream Diffusion
Transformer (S3-DiT) T2I. Qwen3-4B text encoder → S3-DiT (8-step FlowMatchEuler,
CFG) → 16ch AutoencoderKL VAE. First serious diffusion T2I aimed at **iPhone**
(FLUX.2 / GLM-Image are Mac-only). Arch fully reverse-engineered from the
device-proven LiteRT/Android port at `~/downloads/depthanything-android/zimage`
(scripts vendored in `ref_litert/`).

## ✅ Done + validated (2026-07-09, this session, Mac / 512px bring-up)

- **Oracle** (`capture_oracle.py` → `oracle/`): fp32 diffusers run, teacher-forcing
  bins (per-step latent/adaln/velocity cond+uncond, caps, noise0, sigmas) +
  `image_ref.png` (photoreal apple). The ground truth for every gate.
- **DiT graph = `NativeZDiT` (`zimage_dit_native.py`), bf16.** Wraps the STOCK
  diffusers blocks (native SDPA + native RMSNorm); the ONLY change is a real
  (export-clean) interleaved RoPE processor replacing `view_as_complex`. Host does
  patchify / RoPE / pad-mask / unpatchify (`zimage_host.py`).
  - torch parity (`parity_dit_torch.py` via DeployDiT, `zimage_dit.py`): **corr 1.000000**.
  - engine parity all 16 step×branch (`engine_parity_dit.py`): **corr ≥ 0.99972, no NaN**.
- **VAE graph** (`export_vae.py`, fp32): Core AI decoder, scalar unscale z/0.3611+0.1159.
- **E2E** (`pipeline_engine.py`): Core AI DiT (bf16) + Core AI VAE → **PSNR 35.21 dB
  vs oracle**, photoreal apple (`oracle/engine_image.png`). Latent trajectory corr
  0.9999→0.9981.
- **int8lin DiT** (`export_dit.py int8lin`, bf16 activations / int8 weights): **6.1 GB**
  (vs 11 GB bf16), **e2e PSNR 34.77 dB** — quantization barely degrades, faithful image.
  This is the **ship artifact**. int8 config must exclude `diffusers...RMSNorm` (1D
  weight → per-block axis-1 quant fails on rank-1).
- **Text encoder** (`export_encoder.py`, Qwen3-4B, bf16 6.6 GB): graph = inputs_embeds +
  additive causal∧non-padding 4D mask → `hidden_states[-2]`. Host: chat-template
  tokenize → right-pad to L=64 → `embed_tokens` gather → mask. Padding keys are masked,
  so valid outputs are pad-length independent. **penultimate vs oracle cap corr 0.999984**
  (large max|d| is Qwen outlier dims × bf16 — harmless).
- **FULL Core AI pipeline** (encoder + DiT + VAE), real prompt→image (no oracle replay):
  novel prompts render correctly ("fluffy orange cat on a blue velvet sofa",
  "lighthouse on a rocky cliff at sunset") → `oracle/engine_prompt.png`.

## ⚡ Speed / precision trade-off (M4 Max, 512px, 8 steps, CFG = 16 DiT forwards)

| stack | size (enc+DiT) | s/forward | e2e denoise | PSNR vs oracle |
|---|---|---|---|---|
| **bf16** | 6.6 + 11 GB | **0.88 s** | **14.2 s** | **38.67 dB** |
| int8lin | 3.5 + 6.1 GB | 2.35 s | 37.7 s | 35.30 dB |

**int8lin is 2.7× SLOWER than bf16.** Root cause established by sweep (6-layer graphs,
`bench_dit.py` — speed is numerics-independent so NaN configs are still measurable):

| config | s/fwd (6L) |
|---|---|
| bf16 | **0.220** |
| int8lin + bf16 activations | 0.585 |
| int8lin + **fp16** activations | 0.585 |
| int8lin per_channel / per_tensor | 0.571 / 0.570 |

Activation dtype and granularity are **irrelevant**. The DiT forward is **compute-bound**
(1056 tokens × 6B), so weight-only int8 is a pure loss: the graph dequantizes weights back
to 16-bit and runs the same matmul — strictly more work than bf16. Weight-only int8 wins
only on **bandwidth-bound** shapes (LLM decode, S=1) and on footprint.

⇒ **bf16 is the Mac ship default** (faster AND better). int8 = size-only trade.
`default()` spec == GPU spec (same time), so this is not a placement issue.
**The only lever that could make int8 fast here is a true int8×int8 quantized matmul
(TensorOps, WWDC26 330) — which is exactly the "diffusion/encoder" case.** Untested here;
note `knowledge/tensorops-quantized-kernels.md` says matmul2d prefill was *negated* on A19.

## 🧩 One graph for every resolution AND every prompt length

`export_dit.py --dyn-cap --dyn-img` makes the image-token and caption axes dynamic.
Cost is **~4.5–9 %** (6-layer: 0.222 static → 0.232 dyn; full: 0.89 → 0.97 s/fwd),
so there is no reason to ship static-shape buckets (each would duplicate 11 GB).

**`--dyn-cap` is not optional for real prompts:** the pipeline pads each caption to a
multiple of `SEQ_MULTI_OF` (=32) and the inner pad tokens are real attention context, so
**cond and uncond generally have different `n_cap`** (e.g. a 45-token prompt → cond 64,
uncond `""` → 32). A static cap=32 graph simply cannot run that pair. (The LiteRT README
flags the same thing: each branch needs its own context RoPE + cap-pad mask.)

| run (full dyn graph, bf16) | s/fwd | denoise | PSNR vs fp32 ref |
|---|---|---|---|
| 512, short prompt (cap 32/32) | 0.97 | 15.5 s | 39.47 dB |
| 512, long prompt (cap **64/32**) | 1.03 | 16.6 s | 27.66 dB* |
| **1024** (n_img 4096, cap 32/32) | 4.02 | 64.2 s | **42.39 dB** |

\* 27.66 dB is the texture-heavy oil-painting case — the images are visually identical
(composition/light/subject match; only high-frequency net/sack detail differs). PSNR is
not comparable across prompts: a smooth apple scores ~40 dB at the same relative error.

**VAE stays per-size** (`export_vae.py --size 512|1024`): a dynamic latent H/W trips
`Constraints violated` (the decoder derives W from H). It runs once per image and is
~0.3 GB, so per-size bundles are the right call — don't fight it.

## 📱 iPhone: the dtype is decided by AOT, and it is fp32-activations

`xcrun coreai-build compile` has **no flag to set input element types** (only
`--expect-frequent-reshapes`). It feeds f16 to the module, so:

- **bf16 module → AOT fails**: `Incompatible element type for parameter at index 0, mlir
  module expected element type bf16 but received f16`. bf16 is a Mac-only escape hatch.
- **fp16 module → Z-Image NaNs** at sampler step 2 (same as 512px; depth-driven, not
  token-driven — it NaNs at 256px too, with a healthy latent |max|=4.1).
- **Hand-rolled max-norm RMS/LayerNorm made it worse** (NaN from step 0 in-graph), even
  though it is fp32-exact in torch (corr 1.000000). Hand-rolled norms fare as badly in the
  Core AI graph as the hand-rolled attention did. Do not retry this.
- **int8 weights + fp32 activations: clean, and the FASTEST int8 config.** This is exactly
  the device-proven LiteRT recipe ("INTEGER-int8 graph" + `GpuOptions(precision = FP32)`).

| @256px, full depth | s/fwd (M4 Max) | NaN | e2e PSNR | AOT |
|---|---|---|---|---|
| bf16 (no quant) | 0.287 | no | 41.93 | ❌ |
| int8 + bf16 act | 0.678 | no | 36.79 | ❌ |
| int8 + fp16 act | 0.69 | **step 2** | 8.54 | ✅ |
| int8 + fp16 act + max-norm | 0.48 | **step 0** | 8.54 | ✅ |
| **int8 + fp32 act** | **0.37** | **no** | **32.77** | ✅ (untested) |

Also settled: the int8 penalty is a **constant ~2.4–2.7× multiplier**, not a fixed
per-weight cost — it does NOT get relatively worse at low resolution (measured, an earlier
prediction to the contrary was wrong). And **guidance=0 renders fine at 256px**, halving
the work (16 → 8 forwards).

### Estimate (NOT measured — device is truth)
0.37 s/fwd × (A19 ≈ 5–8× slower than M4 Max GPU) ≈ 1.9–3.0 s/fwd →
**30–47 s at g=1 (16 fwd), 15–24 s at g=0 (8 fwd)** for a 256×256 image.
Footprint: DiT 6.5 GB + encoder 3.5 GB, **sequential load** (cannot co-reside);
~10 GB download. The same author's LiteRT/Android port needed ~27 min at 256px on a
Pixel 8a, so this would be a large improvement — if it holds on device.

### ✅ AOT works — and it settles the activation dtype (6-layer probes, h18p)

| graph | source `.aimodel` | AOT `.aimodelc` | errors |
|---|---|---|---|
| int8 + **fp16** act | 1.8 GB | **1.8 GB** (compression kept) | 0 |
| int8 + fp32 act | 1.9 GB | **8.7 GB** (weights expanded to fp32) | 0 |

**MPSGraph constant-folds the weight dequantization when activations are fp32**, baking
fp32 weights into `…/mpsExecutable.mpsgraphpackage/resources.bin`. That is *also* why the
fp32 graph was the fastest int8 config (0.37 vs 0.68 s/fwd — the dequant is gone) and why
it is unusable on a phone (a full DiT would be ~24 GB). With fp16/bf16 activations the
weights stay int8 and are dequantized at run time.

Sanity anchor: the zoo's shipped iOS bundle `glm_ocr_decode_int8hu_s1` is 764 MB as both
`.aimodel` and `.aimodelc` — same `int8lin` quant config as ours, but **fp16 activations**.
(`int8hu` = `int8lin` + untied int8 head; the schemes are identical.)
Placement makes no difference: `--preferred-compute gpu|none|neural-engine` all land on the
MPSGraph delegate (no ANE regions) with identical size — this DiT cannot go on ANE.

### 📱 DEVICE VERDICT (iPhone 17 Pro, A19 — throwaway fp16 probe app, 2026-07-10)

Every fp16 conclusion above was drawn on the Mac. The device says the same thing, and adds a
second wall. Both are now measured, not inferred.

**1. fp16 dies identically on A19.** The exact step-2 inputs that NaN on the Mac
(`dump_step2_inputs.py` writes them as fp16 bins) were replayed on the phone through the
same 6-layer int8+fp16 AOT bundle:

| | nan | forward |
|---|---|---|
| Mac | 18432 / 18432 | — |
| **iPhone 17 Pro** | **18432 / 18432** | 1.14 s |

The hope that A19's kernels or accumulate precision might differ is **refuted**. And iOS AOT
takes fp16 only (bf16 modules are rejected), so there is no dtype left.

**2. A 2 GiB runtime load wall.** `AIModel(contentsOf:)` fails with `NSPOSIXErrorDomain 2`
(ENOENT, empty userInfo) once `resources.bin` exceeds 2 GiB. Device bracket:

| bundle | resources.bin | first 16 bytes readable | load |
|---|---|---|---|
| `glm_ocr_decode_int8hu_s1` (known-good) | 0.80 GB | yes | ✅ 0.0 s |
| Z-Image, 6 layers | **1.96 GB** | yes | ✅ 4.6 s |
| Z-Image, 16 layers | **3.92 GB** | yes | ❌ ENOENT |
| Z-Image, full depth | 6.22 GB | yes | ❌ ENOENT |

The files are intact (readable, correct size via `devicectl info files`) — the wall is in the
runtime. **`coreai-build compile` happily produces the 6.2 GB bundle with 0 errors**, so
*compiles* ≠ *loads*. So even a fixed fp16 would still need the DiT split into ≤2 GiB chunks —
which is exactly why the LiteRT/Android port shipped ~866 MB chunks, and matches
the Nemotron ASR port's ">2 GB multi-IO AOT" split.

Device speed, for the record: 1.14 s/forward for 10 of 34 blocks at 256px → ~3.9 s/forward at
full depth → ~62 s per image (16 forwards), ~31 s at `guidance=0`.

⇒ **iPhone is closed with this toolchain.** The only exits are outside our control: Apple
accepting bf16 AOT entry points (bf16 numerics are already proven exact — but 6.1 GB still
needs chunking), or an fp16 fix at the matmul-accumulation level.

### ⛔ The Mac-side fp16 blocker (why bf16 is the only working dtype)
Everything now reduces to a single problem. AOT needs fp16 (bf16 rejected); fp16 NaNs at
sampler step 2 (depth-driven, also at 256px). Solve it and iPhone is live: full DiT = 6.1 GB
AOT bundle, ~0.69 s/fwd on M4 Max @256 → **est. 28–44 s at g=0 (8 fwd) on A19**.

#### What has been TRIED and does NOT fix it (all verified)

| attempt | fp32 exactness | fp16 engine |
|---|---|---|
| hand-rolled max-norm RMSNorm/LayerNorm (`_swap_norms`) | corr 1.000000 | **worse** — NaN from step 0 |
| `rescale_residual(K=64)` — residual 6.7e3 → 104 | corr 1.000000 | NaN at step 2 (unchanged) |
| `rescale_fp16_safe(C=32)` — w2/to_out 3.1e5 → 9.8e3 | corr 1.000000 | NaN at step 2 (unchanged) |
| both together (K=64, C=32) | corr 1.000000 | NaN at step 2 (unchanged) |

Both rescalings are output-exact weight transforms (scale-invariance of the RMSNorm that
follows each rescaled tensor, with `eps /= scale^2`), and together they leave **zero**
fp32 module outputs above 65504 (measured: peak 1.96e4 at step 2, 515 modules hooked).
The engine still NaNs — identically, at the same step, with the same PSNR 8.54.

#### What the NaN actually is
- **Input-dependent, not call-count**: the same step-0 inputs run 6× in a row are clean;
  step-2 inputs fed as the *very first* call NaN immediately.
- Not explained by any module *output* exceeding fp16 range.
⇒ The overflow is almost certainly **inside a matmul accumulation** (w2 sums 10240 terms;
SDPA's q·kᵀ sums 128) or inside SDPA's softmax — places a per-module fp32 hook cannot see,
and which weight rescaling only helps proportionally (partial sums can dwarf the result
when there is cancellation).

#### Activation quantization (the LiteRT/Android recipe) — TRIED, FAILED
The Android README is explicit: it ships **INTEGER-int8** (int×int) and says *"the
weight-only-FLOAT path hangs/overflows on the GPU delegate, so it is NOT used here"* —
i.e. exactly the `int8lin` path we use. So `op_input_spec` (activation quant → true
int8×int8 matmul) was the obvious untried lever: it would explain the fp16 overflow, the
2.4–2.7× slowdown, and the AOT fp32 weight expansion in one stroke.

`export_dit.py --act-quant` implements it (`op_input_spec`, static, calibrated on the real
oracle inputs — per-step latent/adaln/caption). Result:

| config | export | numerics (fp16, 6-layer) |
|---|---|---|
| `op_input_spec` on `global_config` (`"*"`) | ✅ | **NaN from step 0** (quantizes the inputs of *every* op — softmax, mul, add) |
| `op_input_spec` scoped to `torch.nn.Linear` | ✅ | **NaN from step 0** |

Both are *worse* than plain fp16 (which is clean at step 0 and NaNs at step 2). Activation
quant as exposed here does not reproduce LiteRT's integer path — most likely coreai-opt
emits fake-quant nodes that still execute in fp16 (so the overflow survives and quant noise
is added), and/or per-tensor symmetric scales are hopeless against Z-Image's outliers.
AOT size (gate 2) was never reached.

#### Score so far: four hypotheses, four failures
1. hand-rolled max-norm norms → NaN from step 0
2. residual rescale (K) → unchanged
3. w2/to_out rescale (C) → unchanged
4. both together → unchanged
5. activation quant (global, then Linear-scoped) → NaN from step 0

Nothing here has moved the fp16 NaN by one bit. **Stop guessing.** The only honest next
step is the engine-side depth bisect (`--layers 1,2,4,8,…` on fp16, step-2 inputs) to find
the first block that NaNs, and then inspect that block's internals — accepting that the
toolchain may expose no accumulate-precision knob, in which case iPhone needs either
bf16 AOT support from Apple (bf16 numerics are already proven clean) or the LiteRT-style
chunked FP32-compute design.

## 🔑 Hard-won gotchas (non-obvious)

1. **beta3 toolchain.** OS on macOS 27 beta3 (26A5378j). b1/coreai-torch 0.4.0 FAILS
   to LOAD every bundle (`Failed to convert to versioned IR / cannot unwrap empty
   odiec_module_t`). Fix = bump to `coreai-core 1.0.0b2 + coreai-torch 0.4.1 +
   coreai-opt 0.2.1` (pulls torch 2.11). This was the real cause of two red herrings
   (looked like hand-rolled-attention op-hostility).
2. **fp16 overflows Z-Image → use bf16.** The S3-DiT residual reaches ~4400; fp16
   activations NaN on late/some sampler steps (input-specific, not a repeat-call bug).
   bf16 (range ±3e38) fixes it cleanly, 16-bit. Native SDPA is more robust than the
   hand-rolled softmax matmul (which NaNs at step 0). **Ship DiT = NativeZDiT bf16.**
3. **Caption padding.** `_pad_with_ids` pads cap to a multiple of SEQ_MULTI_OF (=32);
   pad tokens are REAL attention context (the pipeline does NOT effectively mask them).
   So n_cap = round_up(L, 32) must match the pipeline exactly; truncating to valid L
   drops corr to ~0.994. build_dit_inputs mirrors the model's own padding.
4. **DiT conditions on encoder PENULTIMATE hidden** (`hidden_states[-2]`).
5. **CFG is negated:** `noise_pred = -(pos + g*(pos-neg))` (guidance=1 default, turbo).
6. **VAE fp32** (fp16 → NaN, GLM lesson).

## Commands (coreai-models venv, from conversion/zimage/)
```
PY=~/code/coreai-models/.venv/bin/python
$PY capture_oracle.py --size 512 --steps 8 --guidance 1.0
$PY export_dit.py bf16 --size 512 --cap 32 --arch native     # DiT bundle (11 GB bf16)
$PY export_vae.py --size 512                                  # VAE bundle
$PY engine_parity_dit.py exports/zimage_dit_512_cap32_full_native_bf16/*.aimodel --arch native --dtype bf16
$PY pipeline_engine.py  exports/zimage_dit_512_cap32_full_native_bf16/*.aimodel --arch native --dtype bf16 \
     --vae exports/zimage_vae_512_fp32/*.aimodel
```

## ⏭ Pending
- **iPhone**: AOT h18p (`xcrun coreai-build compile`), sequential component load
  (enc + DiT cannot co-reside), 256–384px, likely guidance=0. Nothing device-tested yet.
- **Encoder** is still fixed L=64 (`export_encoder.py --L`) → prompts over ~64 tokens need
  a larger L or a dynamic L axis (same `Dim` trick as the DiT should apply; untested).
- **int8×int8 TensorOps matmul** — the only route to a fast *and* small DiT. Untested.
- Ship: HF bundle+card, zoo conversion/knowledge/README, app CoreAIImageGen tab.

## Operational notes (bit us this session)

- **AOT scratch is enormous.** Compiling a 6.5 GB `.aimodel` for h18p filled a 60 GB-free
  volume and had to be killed; a 1.9 GB graph used ~10 GB. Free ≥ 60 GB first, and run the
  compile behind a disk watchdog.
- **The reference engine loads the fp32 transformer (~24 GB) just for host prep**
  (patchify / RoPE / unpatchify are weight-free; only `t_embedder` has weights). Dumping
  `t_embedder` + `t_scale` next to the bundle and instantiating the transformer from config
  would cut engine startup from minutes to seconds. Worth doing before an app port.
- `pipeline_engine.py` / the parity scripts now load **components** (transformer fp32,
  tokenizer, `embed_tokens` in bf16, VAE only for `--vae torch`) instead of the whole fp32
  `ZImagePipeline` (~40 GB).

## Ship artifact recommendation (Mac)
`zimage_dit_512_cap32_full_native_bf16_dyncap_dynimg` (11 GB, any res / any prompt len)
+ `zimage_encoder_seq64_full_bf16` (6.6 GB) + `zimage_vae_{512,1024}_fp32`.
int8lin bundles exist and are half the size but 2.7× slower — keep them only for the
memory-constrained (iPhone) path.
