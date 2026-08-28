# Conversion guide (PyTorch → `.aimodel`)

Re-author the model with `coreai_models` primitives (so it lowers cleanly), export via
`torch.export`, convert with `TorchConverter`, optimize, save. Verify numerically against the
Hugging Face reference (cosine / top-1 argmax) before trusting a bundle.

## Canonical API (gotchas burned in)

```python
import torch, shutil
from pathlib import Path
from coreai_torch import TorchConverter, get_decomp_table
import coreai.runtime as rt

ep = torch.export.export(m.eval(), args=(), kwargs=inputs).run_decompositions(get_decomp_table())
prog = (TorchConverter()
        .add_exported_program(exported_program=ep, input_names=[...], output_names=[...])
        .to_coreai())
prog.optimize()
shutil.rmtree(out, ignore_errors=True)                 # save_asset will NOT overwrite
prog.save_asset(Path(out), rt.AIModelAssetMetadata())  # Path (not str); minimum_os defaults to v27

model = await rt.AIModel.load(Path(out), rt.SpecializationOptions.cpu_only())  # async; cpu_only for parity
fn = model.load_function("main")                       # SYNC; model.function_names is a list attr
res = await fn({"x": rt.NDArray(np_or_tensor)})        # __call__ is ASYNC -> dict[str, NDArray]
```

## Gotchas that cost real time

- **`save_asset` won't overwrite** — `shutil.rmtree(out)` first.
- **`devicectl copy to` flattens a `.aimodelc`, and the symptom lies.** Pointing `--source` at the
  `.aimodelc` directory copies its *contents* into the destination, dropping the `.aimodelc` level.
  The runtime then cannot select the AOT specialization and reports **`failedToSpecialize`
  (`CoreAIDelegates.AIModelError` code 1)** — which reads like a bad compile. Every
  `SpecializationOptions` mode fails identically, so A/B-ing the load options teaches nothing.
  Push so the name survives: `--destination Documents/models/<dir>/<name>.h18p.aimodelc`.
  Two more from the same hour: `--destination` is relative to the **appDataContainer root**, not
  `Documents` (a push to `models/…` returned `EXIT=0` with a plausible printed path and landed
  nothing); and a real 2.15 GiB transfer takes ~77 s flat / ~311 s nested — verify by listing the
  device, never by exit code.
- **`coreai-build` lives inside the Metal Toolchain cryptex, not in Xcode.** `xcrun -f coreai-build`
  resolves to `…/cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v<ver>/Metal.xctoolchain/usr/bin/`.
  So it is present or absent depending on which Xcode `xcode-select` points at — an Xcode whose
  Metal Toolchain predates it simply has no `coreai-build`, and `aimodelc` there answers the
  misleading `error: Core AI requires the Metal Toolchain` even when `xcrun metal` works fine in
  the same shell. `DEVELOPER_DIR` does **not** fix it; only `sudo xcode-select -s <Xcode>` does.
  Diagnose with `xcodebuild -showComponent MetalToolchain` (prints the toolchain identifier per
  Xcode) rather than believing the error text.
- **`--expect-frequent-reshapes` is usually mandatory for a decoder, and it costs ~1.5 GB per
  exported function.** The reshapes come from `position_ids` growing by one every call
  (`shape=[1, -1]`), not from a multifunction split — so no build escapes them. Without efr the
  AOT is byte-for-byte the source `.aimodel` and buys nothing: the runtime re-specialises per
  shape (hundreds of `ANECCompile() FAILED`) and is compile-bound. Measured on OvisOCR2's int8hu
  decoder: `main`+`prefill` + efr = 3.6 GB (0 re-specialisations), `main` only + efr = 2.1 GB,
  either one without efr = 764 MB and unusable. **The lever is the function count, not the flag** —
  dropping a `prefill` function is what buys the 1.5 GB, at the cost of ingesting the prompt at
  S=1 (3.7× slower prefill). Budget the result against the iOS load wall (2.39 GiB ✅ / 3.92 GB ❌)
  and the ~1.5 GB ship rule. The opposite trap still holds: efr on a genuinely **fixed-shape**
  graph SIGSEGVs on iOS — decide per graph.
- **`AIModel.load` is async; `load_function` is sync; calling the function is async.** Mixing these
  up is the most common first error.
- **Use `cpu_only()` for numeric parity checks** — h16c (fp16-compute) on GPU/ANE produces larger
  hidden-state diffs on high-magnitude activations (still argmax-exact, but noisier maxdiff).
  But **switch to `SpecializationOptions.default()` (GPU/ANE auto) for any real run** — it's far
  faster (TripoSplat DiT: 24.2s→2.6s/call, ~9.3x, cos vs cpu still 1.000000). `coreai_kit.run`
  defaults to cpu_only, so apps/benchmarks that copy it silently run on CPU — override it.
- **Unsupported ATen ops surface at `add_exported_program` validate time**, not at runtime. e.g.
  `aten.remainder.Scalar` (tensor modulo) is unsupported — compute it with `floor`/`where` or a
  scalar symint instead. Either register a lowering (`register_torch_lowering`) or avoid the op.
- **In-place state writes** (KV cache via `slice_update`) need `remove_functionalization(ep)` after
  `run_decompositions`, before converting. Without it the mutation is dropped.
- **Composite externalization** (`ExternalizeSpec` for RMSNorm/RoPE/SDPA/...) marks ops by *class*.
  If your export unit holds submodules of that class which are NOT in the traced graph (e.g. a
  front-end norm kept as an attribute), externalizing fails ("custom op not found"). Opt out by
  exposing `coreai_externalize_specs = ()` on the module, or pass an empty externalize list.
- **fp16 conversion**: keep RoPE `inv_freq` a plain fp32 attribute (NOT a buffer) so `.half()`
  can't underflow the small frequencies to zero. Cast RoPE cos/sin to the query dtype.
- **Truncating layers for smoke tests**: keep architectural regions (KV-shared / double-wide MLP)
  aligned with the checkpoint's real layout, or weights won't load.
- **No complex ops** (`torch.polar`, `view_as_complex`/`view_as_real`, complex `*`). Rewrite complex
  RoPE as real cos/sin: rope returns `stack([cos,sin],-1)`; apply = `(x_re·cos−x_im·sin,
  x_re·sin+x_im·cos)`. (TripoSplat DiT `RePo3DRotaryEmbedding`/`apply_rotary_emb`.)
- **`F.normalize` drops the eps denom clamp** → near-zero-norm vectors blow up (~1e13). Input-
  dependent, so it hides at small seq len and surfaces at large. Replace with explicit
  `x*rsqrt(mean(x²)+eps)` (RMS) or `x*rsqrt(sum(x²)+eps)` (L2). (TripoSplat `MultiHeadRMSNorm`.)
- **Constant-folding sin/cos of LARGE args is low-precision** (cos collapses to ~0.5). If a positional
  embed is computed in-graph from a *constant* (fixed Sobol/grid → sin/cos of arg ~1e5) it gets folded
  wrong. Precompute it in torch and bake as `register_buffer(..., persistent=False)` (plain data, no
  foldable trig). Runtime/input-driven sin/cos is fine. (TripoSplat DiT `pos_emb_cache`.)
- **`prog.optimize()` can hang** on big attention graphs (TripoSplat DiT: 24 blocks × ~12k-token attn
  → >90 min, ~64 GB RAM) while the conversion itself is ~7 s. Skip it: `convert(..., optimize=False)`
  then gate with a manual `run()` (note `verify()` forces optimize=True). On-device AOT `coreai-build`
  optimizes anyway.
- **Bisect divergence at TRUE scale**: when the full model gates low but sub-modules pass at small
  seq len, emit intermediate activations as extra outputs and convert ONCE at real shapes — the eps
  and constant-fold bugs above are emergent at scale and invisible in small-L probes.
- **int8 for a generic (non-LM) model**: `palettize_pytorch_model(model, ex, cfg)` (coreai_models.
  export.compression; n_bits=8, per_grouped_channel axis0 group32) then a PLAIN
  `torch.export`+`TorchConverter` (optimize=False) — **no `export_to_coreai`/externalize needed**;
  palettize auto-skips small output layers. Run as a FILE (it spawns 32 workers re-importing
  __main__). TripoSplat: all nets cos>0.9998, ~3.7x smaller. Note: **on GPU int8 is NOT faster**
  (weights dequant to fp16 for compute) — the win is size/memory + fitting on ANE/iPhone, not speed.
  **WARNING: per-net cos>0.999 does NOT guarantee output quality.** On TripoSplat int8 cos was 0.9998
  per net yet generated splats visibly desaturated (RGB std 0.275->0.172, colors collapse to gray) —
  quant error in conditioning/decoder compounds through the 20-step sampler. For generative/perceptual
  models gate int8 by VISUAL end-to-end output, not cosine; prefer **fp16** (half size, ~lossless)
  when cos is high but the render degrades.
- **fp16 weights (half size, ~lossless on GPU)**: load the model `.half()`, wrap so graph io stays
  fp32 (cast in→half, out→float), convert(optimize=False). Bundle is ~0.5x. **Gotcha (inverse of
  int8): per-net CPU cos can look BAD (0.9 range) because forcing fp16 COMPUTE overflows
  attention/softmax/layernorm on CPU — but on GPU the e2e is IDENTICAL to fp32** (GPU runs fp32
  bundles at fp16 compute anyway, and weights are fp16-native). So **gate fp16 on GPU/visual, not
  CPU cos** — the opposite failure mode from int8. (TripoSplat: int8 desaturates, fp16 is clean.)
- **No-op `squeeze(dim)` aborts the converter**: `x.squeeze(1)` is a torch no-op when dim 1 ≠ 1, but
  coreai-torch lowers `squeeze(dim)` to a HARD shrink and aborts (`dimension to be shrunk must have
  size 1, got N`). Guard it so the trace resolves it away: `if x.shape[1]==1: x = x.squeeze(1)`.
  (LTX-Video DiT `BasicTransformerBlock`.)
- **T5-XXL (and T5-family) encoders OVERFLOW in fp16** → washed-out / degraded generative output even
  though the bundle converts. Use **bf16** (same exponent range as fp32, still half size) or fp32 for
  T5; other nets (DiT/VAE) are fp16-clean. `FP16IO(model, torch.bfloat16)`. (LTX-Video text encoder.)
- **`AIModel.load(path, None)` trips `RuntimeError: MPSGraph Unresolved symbol (prepare/initialize)`**
  on the GPU path — pass an EXPLICIT `SpecializationOptions.default()` (GPU) or `.cpu_only()`, never
  `None`. (`coreai_kit.run` passes `None` for the non-cpu branch; override it in a persistent runner.)
- **Keep the `AIModel` reference alive** in a persistent multi-call runner — storing only the
  `load_function` lets the model get GC'd and the function then returns GARBAGE (no crash, just wrong
  output → looks like a conversion bug). Hold `self.models[name] = m`.
- **Gating a stochastic diffusion/sampler port**: end-to-end pixel cosine vs a reference is dominated
  by **sampler variance, not conversion error** — two legit torch runs (MPS vs CPU, same seed) differ
  at cos ≈ 0.93. Gate by **per-step** net cos (capture the real rollout inputs from a torch run, feed
  the bundle, compare each step → expect 1.000000) PLUS a visual check. (LTX-Video DiT: 8/8 steps cos
  1.000000 while e2e pixel cos is 0.93 = identical to torch-vs-torch device variance.)

## Verify, don't trust

Compare to an HF eager reference: cosine ≈ 1.0 and **top-1 argmax match** on a fixed prompt is the
real pass criterion. A large hidden-state maxdiff alone is usually h16c compute noise, not a bug —
let argmax/top-5 decide. Re-run parity after any OS/toolchain bump (the runtime is OS-coupled).

## Detection transformers

Found by the RF-DETR port (zoo's first detector, [models/rf-detr/README.md](../models/rf-detr/README.md)). Four
platform bugs, each pinned by a minimal repro; all four bite ANY model, not just detection —
detection transformers just happen to use the trigger ops (sine position embeds, bilinear
sampling masks, coordinate floors).

1. **`aten.arange` with float start/end/step aborts the converter** — C++
   `bad_optional_access`, no Python error. Repro: any graph containing `torch.arange(8.0)`
   (`torch.arange(8)` is fine; the *dtype* doesn't matter, the *argument types* do). DETR-class
   models hit it via `gen_sineembed_for_position(…, d_model / 2)` — a float dim. Fix: precompute
   the `dim_t` vector as a Python-list constant (kills the runtime arange/floordiv/pow chain too).
2. **int64-comparison bool chains clobber unrelated live buffers at runtime.** A chain like
   `((ix0 >= 0) & (ix0 < W)).to(float)` on `.long()` tensors makes a *different*, still-live fp
   tensor (two ops upstream; even a graph OUTPUT) read back garbage/NaN once the subgraph
   executes. Deterministic, unit-independent (CPU too); `clone()` / `contiguous()` barriers do
   NOT protect the victim; skipping `optimize()` doesn't either. Diagnosis pattern: a tensor is
   provably computed right (its other consumer is exact) but reads wrong later → buffer-liveness
   bug, hunt the comparison chain. Fix: compute 0/1 masks in float arithmetic —
   `1 - (x - x.clamp(lo, hi)).abs().clamp(max=1)` is exact on integer-valued floats.
3. **`aten.floor` / `trunc` / `ceil` lower to IDENTITY on the GPU delegate** (CPU correct;
   `round` rounds ties away-from-zero instead of to-even). Two natural workarounds also fail:
   `div(x, 1, rounding_mode="floor")` simplifies to identity, and `float→long→float` roundtrips
   are cast-cancelled by the converter **dropping truncation semantics (CPU too)**. The floor
   that survives every unit: `torch.div(x * 2.0, 2.0, rounding_mode="floor")` — floor-div with
   divisor ≠ 1 lowers correctly, and ×2/2 is a power-of-two scale (exact in fp). Corollary: the
   "compute remainder with floor" advice above must use THIS floor on GPU paths.
4. **`torch._assert` on data-dependent comparisons breaks torch.export non-strict**
   (GuardOnDataDependentSymNode, torch 2.11) — ironically added by upstreams *for* export
   compatibility. For static-shape exports the check is vacuous: no-op `torch._assert` around
   `torch.export.export` and restore after.

Detection-gate design note: gate detector numerics with **set-based matching** (per confident
oracle detection: same class, IoU ≥ 0.75, score within tolerance), not positional top-k
compare — DETR-family models emit near-duplicate predictions whose ranks swap under fp16/h16c
noise, and positional compare overflags healthy conversions. Random-noise inputs flip two-stage
top-k proposal selection at near-ties (the detection analog of the LLM argmax margin rule) —
use real images for the gate, noise only as an informational probe.

## Publishing & fetching artifacts (HF Xet)

Hugging Face now stores large LFS files in **Xet** content-addressed storage, and this bites both
download and upload of multi-GB `.aimodel` bundles:

- **`HF_HUB_DISABLE_XET=1` does NOT bypass Xet** — the `resolve/main/<file>` URL still 302-redirects
  to the Xet bridge (`cas-bridge.xethub.hf.co`), which reconstructs cold files server-side and can
  crawl or stall (the "stuck at a few MB" symptom).
- **`curl -C -` (resume) HANGS the Xet bridge** — the resume request sends a `Range:` header that the
  bridge stalls on (0 bytes forever). A **plain GET** of the same URL streams fine (measured 13 MB/s).
  If you must curl, no `-C -`; add `--retry`/`--speed-limit` to re-roll onto a warm edge.
- **Best for a flaky link: native `HF_XET_HIGH_PERFORMANCE=1 hf download <repo>`** — it fetches the
  reconstruction in parallel chunks and **resumes at chunk granularity** (from `~/.cache/huggingface/xet/`),
  so a dropped connection continues instead of restarting from 0 (which a single curl stream does).
- **Don't mix Xet and non-Xet attempts** for the same file: the cache snapshot symlink can end up
  pointing at a **sparse, incomplete** blob from the killed attempt (apparent size right, `du` actual
  size small) while the good bytes land in a differently-hashed blob. `from_pretrained` then fails to
  load it. Fix: repoint the symlink to the complete blob, or wipe `~/.cache/huggingface/hub/models--<org>--<name>`
  and re-download cleanly.
- **Upload:** `hf upload-large-folder <repo> . --repo-type=model` is **resumable** — kill it and re-run,
  it continues (only un-committed files re-upload). Uploads are fast when the chunks are already in the
  local Xet cache from a prior download.


## Device-integration traps (detection/CV apps; from the RF-DETR device work, added 2026-07-14)

- **`.aimodel` directories cannot be embedded in an app bundle** — the installer misreads the
  extension-suffixed root dir as a nested bundle → "invalid bundle". Ship via download (Hub) or
  sideload to Documents instead.
- **Never name a folder `Resources/` at the iOS bundle root** — CodeSign fails with
  "code object is not signed at all (embedded.mobileprovision)".
- **Benchmark only at `thermalState == .nominal`**: a day of device use silently degrades a 25 ms
  model to 58–103 ms (thermal saturation, not your app). Record thermal/lowPower alongside STATS;
  cool-down between runs.
- **ANE status for CV (iOS 27 beta)**: CV graphs silently fall back to GPU even with an ANE preference
  and a pure-ViT split backbone (fingerprint-identical outputs, zero ANE-compile wait). GPU monolith is
  the fastest deployment; the split-deploy infra (`export_rf_detr.py --split`, backbone/head chain
  bit-exact) stands ready for when the runtime honors ANE.
