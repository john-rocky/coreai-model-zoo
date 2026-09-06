# Core AI error index — every exact error string this project has hit, verbatim, with the verified cause

An agent that hits an error searches the exact string. This page exists so that search lands
somewhere: **every H2 below, apart from the two closing sections, is one error string, verbatim as
the log printed it**, from Core AI's runtime, `coreai-torch`, `coreai-build`, or the Swift engines
in `apple/coreai-models`. Under each:
when it appears, the cause this project verified, the fix, the evidence, and the OS / toolchain it
was seen on.

Three rules, so the page stays true:

- **Only strings this project observed in its own runs.** Nothing here is copied from Apple's docs
  or from hearsay. The evidence line says what is behind each entry: `log:` a file kept on the
  maintainer's machine (`~/code/coreai/…`) or in this repo (`cli/logs/…`); `issue:` the report this
  project filed against `apple/coreai-models` or `apple/coreai-torch`, which carries the log
  excerpt; `record:` the note or state file that captured the string when the raw log was not kept.
- **"Not isolated" means exactly that.** Where the cause was never pinned down, the entry says
  "Not isolated" and gives only what was measured. No entry states a cause that was not established.
- **Silent failures are not here.** A conversion that succeeds and produces wrong numbers prints no
  string; those live in [`cli/DOCTOR_RULES.md`](../cli/DOCTOR_RULES.md) and `coreai doctor`
  reads them. `coreai doctor` findings link back into this page by anchor.

The string in the heading is the search key; the block under it is the fuller signature to match
against. Lines and line numbers inside Apple's frameworks (`GPUMemrefOps.mm:687` / `:700` / `:707`,
`NDArrayDescriptor.swift:139` / `:136`) move between builds — match on the message, not the number.

The entries are grouped by the stage at which the string appears: conversion, `coreai-build`, load /
specialization, execution, the Swift engines, and the tools around the runtime.

---

**Conversion time (`coreai-torch`, `coreai-opt`, `torch.export`)**

## Unsupported ATen op: sym_max

The converter has no lowering for the op; `add_exported_program` rejects the program at validate
time, before anything runs.

- **When:** `torch.sym_max` used to clamp a dynamic narrow start (`max(seq_len − W, 0)`) traced
  under `torch.export` and was refused by the converter (Unlimited-OCR's R-SWA decode). The same
  class, message not recorded verbatim: `aten.remainder.Scalar` (tensor modulo — the only op that
  blocked the YOLO26 port), `aten.atan2` (Kokoro), `aten.var.correction` (AdcSR's `.std()`).
- **Verified cause:** no lowering exists for the op in `coreai-torch` 0.4.x. Unsupported ops
  surface at `add_exported_program`, not at runtime.
- **Fix:** rewrite the op. `sym_max` — do not clamp a dynamic start at all; make the decode graph
  fully static (data-driven `mutable_slice_update` from a `pos` tensor, full-buffer masked read).
  `remainder` — `x - s * torch.div(x, s, rounding_mode="floor")` (integer dtypes; on GPU paths use
  the `×2/2` floor from the detection-transformer notes). `atan2` — half-angle
  `2·atan(y / (√(x²+y²) + x))`. `var` — remove the in-graph color match. Or register a lowering
  with `register_torch_lowering`.
- **Evidence:** record: [`unlimited-ocr-rswa-static-decode.md`](unlimited-ocr-rswa-static-decode.md)
  (What did NOT work), [`conversion-guide.md`](conversion-guide.md) (Gotchas), [`kokoro-tts.md`](kokoro-tts.md),
  [`adcsr-super-resolution.md`](adcsr-super-resolution.md); issue: the `remainder` note in
  [apple/coreai-torch#66](https://github.com/apple/coreai-torch/issues/66). `coreai doctor`: `SRC-REMAINDER`.
- **OS · toolchain:** `coreai-torch` 0.4.0–0.4.2.

## dimension to be shrunk must have size 1, got N

The converter aborts on a `squeeze(dim)` that is a no-op in PyTorch.

- **When:** `x.squeeze(1)` where `x.shape[1] ≠ 1` (a torch no-op) — LTX-Video's
  `BasicTransformerBlock`; V-JEPA 2's `rotate_queries_or_keys` calling `emb_sin.squeeze(-1)` on a dim
  of size D/2, where the same failure printed as `Operation creation failed` at `VJEPA2RopeAttention`.
- **Verified cause:** `coreai-torch` lowers `squeeze(dim)` to a hard `ShrinkDims`, which requires
  the dim to be 1.
- **Fix:** guard it so the trace resolves it away — `if x.shape[1] == 1: x = x.squeeze(1)` — or
  patch the reference function with the squeeze removed (math unchanged; `conversion/vjepa2/export_fp16.py`).
  Rule: a converter failure on a torch no-op is a patch-the-reference case, not a rewrite case.
- **Evidence:** record: [`conversion-guide.md`](conversion-guide.md) (Gotchas),
  [`video-world-models-vjepa2.md`](video-world-models-vjepa2.md). `coreai doctor`: `SRC-SQUEEZE-DIM`.
- **OS · toolchain:** `coreai-torch` 0.4.x.

## Operation creation failed

The same no-op-squeeze abort as above, as V-JEPA 2 printed it. See
[dimension to be shrunk must have size 1, got N](#dimension-to-be-shrunk-must-have-size-1-got-n).

## bad_optional_access

A C++ exception out of the converter, with no Python error, on `torch.arange` with float arguments.

- **When:** any graph containing `torch.arange(8.0)` — `torch.arange(8)` is fine; the dtype does
  not matter, the argument types do. DETR-class models hit it through
  `gen_sineembed_for_position(…, d_model / 2)`, a float dim.
- **Verified cause:** `aten.arange` with a float start / end / step is not handled by the converter
  (minimal repro in the issue).
- **Fix:** precompute the `dim_t` vector as a Python-list constant, which also removes the runtime
  `arange` / `floordiv` / `pow` chain.
- **Evidence:** issue: [apple/coreai-torch#8](https://github.com/apple/coreai-torch/issues/8);
  record: [`conversion-guide.md`](conversion-guide.md) (Detection transformers, 1).
  `coreai doctor`: `SRC-ARANGE-FLOAT`.
- **OS · toolchain:** `coreai-torch` 0.4.x (RF-DETR port, July 2026).

## ValueError: axis 1 is out of bounds

`coreai-opt`'s stock macOS `4bit` preset dies on a gated RMSNorm.

- **When:** quantizing a model that carries `RMSNormGated` (qwen3.5, granite4h, nemotron_h,
  Ling-3.0-tiny) through the quantizer path. Models that ship via palettization never hit it because
  that exclusion set differs.
- **Verified cause:** `presets.py:_TORCH_MODULE_EXCLUSIONS` excludes `RMSNorm` and `RMSNormPlusOne`
  but not `RMSNormGated`, whose `weight` is rank 1; `per_block block_size=32 axis=1` on a rank-1
  tensor takes the hard error path, not the graceful block-size skip. Reproduced on the bare primitive.
- **Fix:** add `RMSNormGated` — and any subclass, matching is by exact class — to
  `module_type_configs` yourself.
- **Evidence:** record: [`compression-reference.md`](compression-reference.md) (Pitfalls);
  `~/code/coreai/LING_3_0_TINY_STATE.md`.
- **OS · toolchain:** `coreai-opt` 0.2.1, August 2026.

## RuntimeError: macOS quantization preset provided, but platform is iOS

`coreai.llm.export` refuses a macOS quantization preset on an iOS export.

- **When:** `coreai.llm.export … --platform iOS --compression 4bit`.
- **Verified cause:** the CLI couples the quantization scheme to the platform by design: the iOS
  (static) export ships palettized weights, the macOS (dynamic) export linear INT4.
- **Fix:** apply linear INT4 at the MLIR level after export with the same `quantize_weights`
  primitive the diffusion pipeline uses (script in the issue). Then read
  [ANECCompileOffline() failed](#aneccompileoffline-failed-osstatus0-anecompilestatus1) before
  targeting the ANE with it.
- **Evidence:** issue: [apple/coreai-models#55](https://github.com/apple/coreai-models/issues/55) (Reproduction).
- **OS · toolchain:** `coreai-models` June 2026, `coreai-opt` 0.2.0.

## torch.fx.experimental.symbolic_shapes.ConstraintViolationError: Constraints violated (d_73)!

`torch.export` inside the `coreai-models` export pipeline refuses a dynamic dim because a guard
in the traced graph contradicts the `Dim` range the pipeline declared.

```
torch.fx.experimental.symbolic_shapes.ConstraintViolationError: Constraints violated (d_128)! For more information, run with TORCH_LOGS="+dynamic".
  - Not all values of d_128 = L['key'].size()[2] in the specified range satisfy the generated guard 64 <= L['key'].size()[2] and L['key'].size()[2] <= IntInfinity()
```

- **When:** (a) exporting a static-S>1 verify graph for spec-decode with SDPA *externalized*
  (`d_73`); (b) MinerU's `pf64` chunked-prefill export, where the guard was `64 <= key.size()[2]`
  (`d_128`).
- **Verified cause (a):** the externalized SDPA's lower-right causal mask emits a `k_len ≥ S` guard,
  and the externalize pipeline's auto-`Dim` (`min=2`) violates it at static S>1.
  **(b): Not isolated** — the export later shipped, but the record does not say what changed.
- **Fix (a):** do not externalize SDPA in that export; a decomposed in-graph SDPA builds the
  identical mask from plain ops (at S≤17 the unfused cost is noise against the weight read).
- **Evidence:** record: [`spec-decode-hybrid-verify-design.md`](spec-decode-hybrid-verify-design.md);
  log: `~/code/coreai/_mineru_export_pf64.log` (2026-07-05).
- **OS · toolchain:** `coreai-torch` 0.4.0, torch 2.9.0.

## IndexError: shape[3]

`replace_avg_pool2d` throws on an unbatched 1-D pool.

- **When:** `AvgPool1d` applied to an unbatched `[C, L]` tensor (Qwen2.5-Omni's audio encoder).
- **Verified cause:** the converter's `replace_avg_pool2d` indexes a fourth dim that a `[C, L]`
  input does not have.
- **Fix:** `reshape(N, 2, d).mean(1)` — the pool axis is even, so it is bit-identical.
- **Evidence:** record: [`qwen2.5-omni-audio-understanding.md`](qwen2.5-omni-audio-understanding.md).
- **OS · toolchain:** `coreai-torch` 0.4.0, June 2026.

## SIGSEGV in coreai-pre-compilation-rewrite

`program.optimize()`, the Python runtime's `AIModel.load`, and `xcrun coreai-build compile` all
segfault on the same graph: the pass is shared by every compile path.

```
faulthandler: coreai/_compiler/_transforms/passes.py:261 apply_passes_sync  <- coreai/authoring/asset.py:230 optimize
xcrun coreai-build compile <x>.aimodel --platform macOS --preferred-compute gpu   ->  SIGSEGV (139)
```

- **When:** the SinSR diffusion U-Net (a swin U-Net). Reproduced on a 3.09 M-parameter conv-only
  U-Net with `attention_resolutions=[]`.
- **Verified cause:** the swin block's `calculate_mask()` runs at forward time for the no-shift
  block (`shift_size == 0`) and emits a degenerate all-zero constant-mask subgraph that the
  `coreai-pre-compilation-rewrite` pass segfaults on. Every hand-rolled piece (ResBlock, skip-cat,
  WindowAttention, shifted SwinBlock) converts fine; only the real `BasicLayer` no-shift path crashes.
- **Fix:** an output-identical mask workaround (the mask is constant; build it outside the forward),
  Mac-validated. AdcSR avoids it entirely because its pruned SD-2.1 has no swin stage.
- **Evidence:** record: `~/code/coreai/SINSR_COREAI_STATE.md` (RESOLVED 2026-06-22, and the
  compile-path probes), [`adcsr-super-resolution.md`](adcsr-super-resolution.md).
- **OS · toolchain:** `coreai-torch` 0.4.0, `coreai-core` 1.0.0b1, MetalToolchain v27.1.5194
  (2026-06-22). Not re-tested on 0.4.2.

---

**`xcrun coreai-build compile`**

## error: Core AI requires the Metal Toolchain

`aimodelc` answers this even though `xcrun metal` works in the same shell.

- **When:** the Xcode that `xcode-select` points at has a Metal Toolchain that predates
  `coreai-build`.
- **Verified cause:** `coreai-build` lives inside the Metal Toolchain cryptex
  (`…/cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v<ver>/Metal.xctoolchain/usr/bin/`), not in
  Xcode, so it is present or absent depending on the selected Xcode. `DEVELOPER_DIR` does not fix
  it.
- **Fix:** `sudo xcode-select -s <Xcode with the newer Metal Toolchain>`. Diagnose with
  `xcodebuild -showComponent MetalToolchain` rather than believing the error text.
- **Evidence:** record: [`conversion-guide.md`](conversion-guide.md) (Gotchas).
- **OS · toolchain:** Xcode 27 betas, 2026-07.

## GPU::anePreCompileBinary

`coreai-build compile` runs ~5 min at 100 % CPU, then dies with `SIGSEGV` (exit 139), no diagnostic,
no `.aimodelc`.

```
Exception:  EXC_BAD_ACCESS (SIGSEGV) — KERN_INVALID_ADDRESS
Crashing thread: MPSGraphExecutable_queue
  0  libobjc.A.dylib                    objc_release
  1  MetalPerformanceShadersGraph_host  GPU::anePreCompileBinary(MPSGraphExecutable*, llvm::SmallVectorImpl<mlir::…>)
  2  MetalPerformanceShadersGraph_host  BaseModuleRef::compileAndLoadANE()
```

- **When:** (a) a static-shape (iOS) LLM program with linear blockwise-INT4 weights
  (`blockwise_shift_scale`) compiled `--preferred-compute neural-engine`; the byte-for-byte
  identical structure with palettized weights (`lut_to_dense`) compiles to 31 ANE regions.
  (b) June 2026: the six host-cache Gemma 4 chunk graphs, ~0.9 s in, both archs, with
  `ANECompilerOffline::~ANECompilerOffline → objc_release` on the stack, while the 35-layer monolith
  from the same authoring compiled fine.
- **Verified cause (a):** the ANE pre-compiler cannot legalize the linear-INT4 static program and
  segfaulted instead of failing. **Status: the crash is fixed** — the exact repro re-run 2026-08-27
  exits 0 three times out of three; the same configuration now fails silently instead, see the next
  entry. **(b): Not isolated**, and not re-tested after the fix.
- **Fix:** none needed for the crash on current builds. For the ANE, palettize (the control).
- **Evidence:** issue: [apple/coreai-models#55](https://github.com/apple/coreai-models/issues/55)
  (crash report excerpt, retest); record: [`aot-and-specialization.md`](aot-and-specialization.md)
  (Status / caveats, the chunk-graph line).
- **OS · toolchain:** crashed on macOS 27.0 26A5353q, `coreai-build` 3600.67.5.8.1
  (MetalToolchain v27.1.5194.15); clean on 26A5416b, `coreai-build` 3600.82.1 (v27.1.5237.12).

## ANECCompileOffline() failed: OSStatus=0, aneCompileStatus=1

`coreai-build` prints an `Error:` block, exits 0, and writes a GPU-only `.aimodelc`.

```
Error:

 ANECCompileOffline() failed: OSStatus=0, aneCompileStatus=1, statusdict={
    CompiledInputSourceFileName = ".../extend_1024_16_..._ANE_region_1_0.bc.mlir";
    ErrorList =     (
    );
    NetworkStatusList =     (
    );
}
```

- **When:** the linear-INT4 static LLM from the entry above, `--preferred-compute neural-engine
  --architecture h18p`, on the build that fixed the segfault. The block sits inside ~160–525 KB of
  MLIR warnings; the only other trace is six `failed: ANE regionCall op not found` warnings.
- **Verified cause:** ANE compilation of the linear-INT4 program fails internally and the tool does
  not surface it: exit 0, empty `ErrorList`, and a bundle whose `main-h18p-delegates/` holds only
  `MPSGraph`. The palettized control compiled back-to-back lands 31 ANE regions. Whether linear
  INT4 can be legalized for the ANE at all is a separate, unasked question.
- **Fix:** detect the fallback yourself — `find <x>.aimodelc -name '*ANE_region*' | wc -l` is 0 on
  the failing bundle — and palettize for the ANE. Open upstream as a missing failure signal.
- **Evidence:** issue: [apple/coreai-models#205](https://github.com/apple/coreai-models/issues/205)
  (open), full stderr logs in the linked gist; 4/4 reproductions 2026-08-27/28.
- **OS · toolchain:** macOS 27.0 26A5416b, `coreai-build` 3600.82.1, `coreai-torch` 0.4.2,
  `coreai-core` 1.0.0b2.

---

**Load and specialization (`AIModel.load`, AOT `.aimodelc`, the on-device specializer)**

## LLVM ERROR: cannot unwrap empty odiec_module_t

Every `.aimodel` converted with `coreai-torch` 0.4.0 stops loading on OS 27 beta 2 and later. The
abort — at `AIModel.load` and at `coreai-build compile` alike — is preceded by the two lines that
name the real problem:

```
loc(fused<{call_stack = ["_empty_nn_module_stack_from_metadata_hook$1"], identifiers = ["sym_size_int_28"]}>[...]): error: expected AICode versioned location, got: loc(fused<...>)
error: Failed to convert to versioned IR
LLVM ERROR: cannot unwrap empty `odiec_module_t`
```

- **When:** loading or AOT-compiling an asset whose `metadata.json` has **no `producer` field**
  (0.4.1+ writes `"producer": "coreai-core 1.0.0b2"`). Beta 1 (26A5353q / 24A5355q) loads it;
  beta 2 (26A5368g) on refuses it. Apple's beta 5 release notes list the incident (177008303) as
  fixed; a load on 26A5416b (2026-09-04) and a `coreai-build compile` on the same asset still abort
  with this signature. A release note is not a measurement.
- **Verified cause:** 0.4.0 baked PyTorch stack traces into the IR as MLIR `fused` locations and
  the beta-2 compiler no longer parses that nested form (Apple,
  [apple/coreai-torch#37](https://github.com/apple/coreai-torch/issues/37)).
- **Fix:** strip the debug locations in place — `strip_debug_info` from
  [apple/coreai-torch#44](https://github.com/apple/coreai-torch/issues/44), weights byte-identical,
  minutes per model, verified on 40 zoo bundles (on a beta-2+ machine the asset must be *read* with
  a `coreai-core` 1.0.0b1 wheel first; recipe in the note) — or re-export with 0.4.1+. `.aimodelc`
  cannot be stripped; recompile from a fixed `.aimodel`. Things that do not work: `coreai-build
  package`, pinning `coreai-core` back, re-AOT with a newer toolchain.
- **Also recorded with this string, not re-isolated against the cause above:** loading a
  custom-Metal-kernel `.aimodel` low-level on device (`AIModel(contentsOf:)`) in June 2026 — the
  AOT-compiled `.aimodelc` loads ([`bitvla-1.58bit-vla.md`](bitvla-1.58bit-vla.md)); a hand-rolled
  multi-head view/transpose/matmul attention in a DiT failing "the versioned-IR pass", where the
  SDPA composite passes ([`zimage-port.md`](zimage-port.md)).
- **Evidence:** log: [`cli/logs/case-a-ground-truth-load-abort.txt`](../cli/logs/case-a-ground-truth-load-abort.txt)
  (in this repo), `~/code/coreai/_mtp_mac/smoke1.log`; the measured builds in
  [`models/_SMOKE.json`](../models/_SMOKE.json); issues apple/coreai-torch#37, #44; record:
  [`coreai-torch-041-ir-incident.md`](coreai-torch-041-ir-incident.md). `coreai doctor`:
  `IR-040-DEBUG-LOC` (fingerprints the missing `producer`, severity by host build).
- **OS · toolchain:** refused on every OS 27 build from 26A5368g through 26A5416b; `coreai-build`
  3600.75.3–3600.82.1.

## NSPOSIXErrorDomain Code=2 "No such file or directory"

`ENOENT` at engine creation or `AIModel.load`, on a file that is right there. Six verified causes,
none of them a missing model file.

```
ERROR Error Domain=NSPOSIXErrorDomain Code=2 "No such file or directory"
App terminated due to signal 15.
```

- **(a) A partial specialization cache from an earlier failed cold specialization.** A ≥2 GB
  bundle's cold GPU specialization died with [`std::bad_alloc`](#libcabi-terminating-due-to-uncaught-exception-of-type-stdbad_alloc-stdbad_alloc)
  (run 1); every later attempt on the same container returned `Code=2` (runs 2–5). The partial
  e-cache ate ~3.5 GB of device disk, and the chain is an out-of-disk ENOENT, not a payload
  problem. **Fix:** wipe `Library/Caches/coreai-cache` from the app before engine creation
  (CoreAIChat's `GEMMA_CLEAR_SPEC_CACHE=1`, device-verified: the same specialization then completed
  in 20.9 s). Last resort: uninstall, reinstall, retry with ≥~4 GB free. Log:
  `~/code/coreai/ondevice/_pipelined_dev_q2b_r1.log` … `_r5.log`, `_qwen_prefill_dev_q8_r1/r2.log`,
  `_q16_r1/r2.log`; record: [`pipelined-engine.md`](pipelined-engine.md) (Run contract).
- **(b) A load attempt against a partially copied `.aimodel` poisons the content-keyed cache.**
  Once a half-pushed file's load fails mid-specialize, every later load of that model errors
  `Code=2` even after the copy completes or the file is renamed (2026-06-10). **Fix:** after every
  multi-GB push, list the destination and confirm `main.mlirb` is full-size before launching
  anything that loads it. Record: [`swift-runtime.md`](swift-runtime.md).
- **(c) A macOS-tagged IR on an iPhone.** It has no iOS delegates to load. Same for Apple's own
  recipes: an uncompiled `.aimodel` fails at engine load; iOS needs `coreai-build compile
  --platform iOS … --architecture h18p` and `metadata.json` `assets.main` pointed at the
  `.aimodelc`. Log: `~/code/coreai/ondevice/AppleBenchRunner/_device_fastcontext_cold2.log`
  (the FastContext run the note summarizes; the log itself shows only the ENOENT); record:
  [`aot-and-specialization.md`](aot-and-specialization.md) (The 4B wall),
  [`apple-models-bench.md`](apple-models-bench.md).
- **(d) A static iOS bundle from `coreai.llm.export --platform iOS` in FM format.** It is detected
  as chunked-static and routed to the static-shape engine, which expects `extend_*` /
  `load_embeddings` functions the bundle does not provide → `Code=2` at engine create. **Fix:** ship
  the dynamic (macOS-default) FM-format bundle; it routes to the pipelined engine on both platforms.
  Log: `~/code/coreai/_mc5_launch2.log`, `_mc5_launch3.log` (MiniCPM5, 2026-06-26); record:
  [`minicpm5-1b.md`](minicpm5-1b.md) (§3).
- **(e) An AOT `.aimodelc` compiled with `--expect-frequent-reshapes` loaded low-level with
  `expectFrequentReshapes = true`.** Fails `POSIX Code=2`; with `= false` it loads. Record:
  [`bitvla-1.58bit-vla.md`](bitvla-1.58bit-vla.md) (§5).
- **(f) A near-full device.** The AOT load stages the precompiled MPSGraph package into
  `Library/Caches/coreai-cache` (~3 GB for a 3 GB bundle); ENOSPC leaves a partial stage that
  pollutes the content-keyed cache → `Code=2` on the next launch. `devicectl` has no file-remove;
  reset is uninstall → reinstall → re-copy. Record: [`bitcpm-ternary-1.58bit.md`](bitcpm-ternary-1.58bit.md).
- **OS · toolchain:** iOS 27 betas, June–August 2026.

## failedToSpecialize

`CoreAIDelegates.AIModelError` code 1 at load. It reads like a bad compile; every time it was
isolated here, the compile was fine.

- **(a) `devicectl copy to` flattened the `.aimodelc`.** Pointing `--source` at the `.aimodelc`
  directory copies its *contents* into the destination, dropping the `.aimodelc` level; a
  `--destination Documents/models/<name>` without the extension does the same. The runtime then
  cannot select the AOT specialization. Every `SpecializationOptions` mode fails identically, so an
  A/B of the load options teaches nothing — and a control bundle with no custom kernel failed the
  same way, which is what kept "the ternary kernel doesn't work on iOS" from being written down.
  **Fix:** push so the name survives — `--destination Documents/models/<dir>/<name>.h18p.aimodelc` —
  and verify by listing the device, never by exit code. Record:
  [`conversion-guide.md`](conversion-guide.md) (Gotchas), [`ovisocr2-port.md`](ovisocr2-port.md),
  [`ternary-chunked-prefill.md`](ternary-chunked-prefill.md) (§6).
- **(b) A symlinked bundle path in Swift.** `AIModel(contentsOf:)` does not follow symlinks to the
  arch-specific delegate dir (`main-h16c-delegates`); the Python runtime does. **Fix:**
  `url.resolvingSymlinksInPath()`. Record: [`glm-image-port.md`](glm-image-port.md). `coreai doctor`:
  `ASSET-SYMLINK`.
- **(c) Forcing `preferredComputeUnitKind: .gpu` on an AOT `.aimodelc`.** That re-specializes the
  baked graph on device (JIT) and wedges the 9 GB GLM-Image graphs; GUI apps hit it reliably.
  **Fix:** load AOT bundles with `SpecializationOptions.default()` or `.cpuOnly`. Record:
  [`glm-image-port.md`](glm-image-port.md). `coreai doctor`: `AOTC-LOAD-OPTIONS`.
- **(d) An asset compiled `--preferred-compute none`** fails to specialize CPU-only on device,
  while the `gpu` and `neural-engine` compiles of the same graph load. Not isolated further. Issue:
  the closing note of [apple/coreai-torch#66](https://github.com/apple/coreai-torch/issues/66).
- **OS · toolchain:** iOS 27 betas 24A5380h–24A5418b, `coreai-build` 3600.75.3–3600.82.1.

## invalidCompiledModel

The `LanguageBundle` / `llm-runner` face of an AOT bundle the runtime will not accept; the raw
`AIModel.load` face is `CoreAIDelegates.AIModelError error 3`.

- **(a) The wrong `--architecture` for the device.** Arch names track the device-identifier major
  version, not the marketing name: iPhone 17 Pro is `iPhone18,1` → `h18p`, and an `h17p` bundle
  pushed to it fails with this string (validated 2026-06-10). On an M4 Max (`Mac16,x`) only `h16c`
  loads; `h16s`, `h16g`, `h17*` all raise in the Python runtime. `coreai-build compile` exits 0
  for **any** requested arch — a successful compile validates nothing; only a device load does.
  Record: [`aot-and-specialization.md`](aot-and-specialization.md) (Architecture names). `coreai
  doctor` describes the rule in `cli/logs/export-routing-cases.txt`.
- **(b) The macOS AOT-load "regression" of June 2026** ([apple/coreai-models#27](https://github.com/apple/coreai-models/issues/27),
  Bug 2): every `.aimodelc` compiled for macOS failed to load with `AIModelError 3` while the same
  runtime loaded iOS `h18p` bundles. Those bundles were compiled `--architecture h16s`; the
  measurement above that only `h16c` loads on an M4 Max came later. Apple closed the issue
  2026-08-26 as not reproducible on a recent beta. Not re-measured here since.
- **(c)** A note attributes an AOT wall on the spec-decode verify graphs to Apple's known issue
  177729331 ("AOT compilation might fail unexpectedly for certain models") — that is the note's
  attribution, not a measurement ([`spec-decode-c1-handoff-2026-07-03.md`](spec-decode-c1-handoff-2026-07-03.md)).
- **OS · toolchain:** macOS 27.0 26A5353q, `coreai-build` 3600.67.5.8.1; iOS 27 beta, 2026-06-10.

## LLVM ERROR: Failed to allocate mmap'd buffer:

The on-device specializer dies on a graph carrying ~2 GB of constants.

```
creating engine (first run pays cold GPU specialization)...
LLVM ERROR: Failed to allocate mmap'd buffer: 
App terminated due to signal 6.
```

- **When:** the Gemma 4 E2B decode bundle in provider mode (PLE rows fed per token from an
  mmapped host table) loaded as a plain `.aimodel` on an iPhone 17 Pro; engine creation dies in
  the cold GPU specialization.
- **Verified:** the same graph AOT-compiled runs (numerics 24/24 ≡ Mac GPU, 26.5 tok/s). The
  notes attribute the crash to the graph's ~2 GB of constants; that attribution was not tested
  separately.
- **Fix:** `coreai-build compile … --platform iOS --architecture h18p --expect-frequent-reshapes`
  and point `assets.main` at the `.aimodelc`. The AOT path is the Gemma ship path on iOS.
- **Evidence:** log: `~/code/coreai/ondevice/_gemma_ple_dev_r1.log` (2026-06-11); record:
  [`pipelined-engine.md`](pipelined-engine.md) (What fits next), `~/code/coreai/GEMMA_KIT_STATE.md`.
- **OS · toolchain:** iOS 27 beta 1, `coreai-torch` 0.4.0.

## RuntimeError: MPSGraph Unresolved symbol (prepare/initialize)

The Python runtime's `AIModel.load(path, None)` on the GPU path.

- **When:** `coreai_kit.run` passes `None` for the non-CPU branch; anything that copies it hits this.
- **Verified:** passing an explicit `SpecializationOptions.default()` (GPU) or `.cpu_only()` fixes
  it. Why `None` fails inside the runtime: **Not isolated.**
- **Fix:** never pass `None`; override it in a persistent runner.
- **Evidence:** record: [`conversion-guide.md`](conversion-guide.md) (Gotchas).
- **OS · toolchain:** macOS 27 beta, `coreai-core` 1.0.0b1–b2.

## Error occurred when loading ANE module

An FP16 asset aborts the process on ANE load instead of returning an error; `AIModel.load` never
returns, so nothing the caller writes can catch it.

```
Error = Error Domain=com.apple.appleneuralengine Code=53
  "createProgramInstanceForModel:...: Program load failed — no memory (transient; retry under lower
   memory pressure) (underlying=0x1)" UserInfo={_ANEErrorUnderlyingStatus=1, _ANEErrorLoadStage=4}

MPSGraphExecutable.mm:3543: failed assertion `Error occurred when loading ANE module:
  Error Domain=MPSGraph Code=-1 "MPSGraphExecutable_Project.h:510:: could not load module from
  MPSGraphPackage"'
```

- **When:** some FP16 `.aimodel` assets under the default specialization (an FP16 YOLO26-pose export
  at 640, a smaller detection export at 32, and a 12 MB reducer). The FP32 versions of the same
  graphs load and run on the ANE; several other FP16 exports of similar size do too.
- **Verified:** the same asset loads and runs with `SpecializationOptions.cpu_only()`, so it is the
  ANE program specifically. The message calls the condition transient; it reproduces on every run,
  in a fresh process, on an idle machine. Why the ANE program fails to load: **Not isolated.**
- **Fix:** none. Workarounds: ship FP32 for the ANE, or load `cpu_only`. Open upstream as an
  abort-instead-of-error.
- **Evidence:** issue: [apple/coreai-torch#67](https://github.com/apple/coreai-torch/issues/67)
  (open; model-free reproducer).
- **OS · toolchain:** macOS 27.0 beta 6 (26A5416b), `coreai-torch` 0.4.2, `coreai-core` 1.0.0b2.

## _ANECompiler : ANECCompile() FAILED

Usually noise. MPSGraph probes the ANE, fails, and falls back to the GPU; the run completes. The
full line, with its usual companion:

```
Error Domain=com.apple.appleneuralengine.compiler Code=1 "_ANECompiler : ANECCompile() FAILED"
... MLIR MPS to ANEC conversion failed
```

- **When it is noise:** any GPU-targeted bundle loaded with the default or GPU-preferred options on
  a Mac — dozens of lines for blockwise-int4 weights (the ANE takes palettization, not blockwise
  int4), and for custom-Metal-kernel graphs (a GPU-only kernel cannot be ANE-placed). It does not
  appear on the iOS AOT path where `--preferred-compute gpu` is explicit. There is no GPU-only
  spec in the Python runtime (`allowed_compute_unit_kinds` is read-only), so the probe cannot be
  suppressed. A run killed on the first of these lines was killed for nothing.
- **When it is not noise:** the int8 *dynamic* decoder graph of Qwen2.5-Omni cannot compile on the
  ANE at all (its fixed-shape encoder does, at cos ~0.99 → byte-identical text); and a raw Python
  `AIModel.load` of a dynamic-shape S=1 decode graph on macOS 27 routes to `ANECCompile`, fails
  this way, then wedges in repeated [`MTL4CommandQueueErrorDomain error 1`](#the-operation-couldnt-be-completed-mtl4commandqueueerrordomain-error-1)
  (MinerU: kill it, the 90 s watchdog will not). That wedge is the Python binding not exposing
  `expectFrequentReshapes`, which the Swift engine sets to steer such graphs off the ANE path.
- **Fix:** ignore it on GPU bundles. For a dynamic graph in the Python runtime, AOT-compile with
  `--expect-frequent-reshapes` and load the `.aimodelc` with `SpecializationOptions.default()`.
- **Evidence:** log: `~/code/coreai/_gemma12b_gate_gpu.log`, `_ling3_dev_gen.log`,
  `_bitvla_engine_gate.log`, `_mineru_decode_gate.log`, `leaderboard/results/_qwen4b_ref.log`;
  record: [`compression-reference.md`](compression-reference.md) (Pitfalls),
  [`dense-int4km-flagship-session-findings.md`](dense-int4km-flagship-session-findings.md),
  [`ternary-chunked-prefill.md`](ternary-chunked-prefill.md), [`mineru-port.md`](mineru-port.md),
  [`qwen2.5-omni-audio-understanding.md`](qwen2.5-omni-audio-understanding.md).
- **OS · toolchain:** macOS 27 betas 1–6.

## connection to service named com.apple.ANECompilerService

A 4B-class ANE bundle static-loads, then its first inference dies.

```
2026-06-27 03:30:01.851 AppleBenchRunner[5279:1373185] Error = Error Domain=com.apple.appleneuralengine Code=16 "compileAsNeededAndLoadCachedModel:...: file not found" ...
engine loaded in 518.421s
warmup trial...
2026-06-27 03:35:18.635 AppleBenchRunner[5279:1377845] Error = Error Domain=NSCocoaErrorDomain Code=4097 "connection to service named com.apple.ANECompilerService" ...
2026-06-27 03:35:18.635 AppleBenchRunner[5279:1377845] ANE compile failed!
LLVM ERROR: IO failure on output stream: No space left on device
```

- **When:** FastContext-1.0-4B (Qwen3-4B) as an ANE static bundle on an iPhone 17 Pro: 31 ANE
  regions, ~518 s cold load, then the warmup inference.
- **Verified:** the GPU AOT bundle is the only on-device path at this size. The cause of the
  ANECompilerService failure: **Not isolated** — the same log ends in a disk-full `LLVM ERROR`,
  so which of the two killed the run is not established. The `Code=16 "file not found"` at load
  is also unexplained; the load went on to succeed.
- **Fix:** ship 4B-class models as GPU `.aimodelc` (`--preferred-compute gpu --architecture h18p`).
- **Evidence:** log: `~/code/coreai/ondevice/AppleBenchRunner/_device_fastcontext_cold3.log`;
  record: [`aot-and-specialization.md`](aot-and-specialization.md) (The 4B wall).
- **OS · toolchain:** iOS 27 beta, 2026-06-27.

## Unable to use cached specializations and original module not available

An MPSGraph assertion (signal 6) at load, on a model that loaded yesterday.

- **When:** right after a GB-class AOT bundle (the Gemma 4 `--tbl` `.aimodelc`, ~2 GB executable)
  was ingested into the content-keyed `coreai-cache` of the same app container, the Gemma 4 ANE
  chunk set in that container started dying at load with this assertion.
- **Verified cause:** the cache state after the ingest; the in-app wipe restores every model.
  Why the ingest invalidates a sibling's cached specialization: **Not isolated.**
- **Fix:** the cache wipe (`GEMMA_CLEAR_SPEC_CACHE=1`); every model pays one cold re-specialization
  (ANE chunks 53.8 s). Rule of thumb: after adding a multi-GB bundle next to other specialized
  models, expect one wipe + re-spec cycle.
- **Evidence:** record: [`pipelined-engine.md`](pipelined-engine.md) (Run contract).
- **OS · toolchain:** iOS 27 beta, June 2026.

## MPSGraphAICodeCompilerDelegate getInitializedAICodeBytecodeWithPayloadPrefix:

A segfault inside the MPSGraph AICode compiler at `AIModel(contentsOf:options:)`, with no error
string and no partial output.

```
EXC_BAD_ACCESS (SIGSEGV) … MPSGraphAICodeCompilerDelegate getInitializedAICodeBytecodeWithPayloadPrefix:
  → Compiler_coreAI.compile(moduleBytecode:to:with:) → libODIECompiler … CompileForDelegates
```

- **(a) iOS: `expectFrequentReshapes = true` at load on a fixed-shape AOT graph.** The hint is a
  request for a reshape-tolerant specialization; on a graph whose shapes are all static the runtime
  discards the AOT specialization and compiles on device, which segfaults here. Device-validated
  2026-07-23 on VibeVoice's five fixed-shape graphs: `= true` → SIGSEGV on the first graph;
  `= false` → all six loads in 2.6 s, gate PASS. Compiling the bundle *with*
  `--expect-frequent-reshapes` does not make the runtime hint safe; the load-time option is what
  matters. **Fix:** the hint only where shapes really change (dynamic query length, bucketed
  prefill); static S=1 decode and fixed-T vocoders load without it. Record:
  [`aot-and-specialization.md`](aot-and-specialization.md) (`expectFrequentReshapes` on a
  FIXED-shape graph), [`vibevoice-multispeaker-tts.md`](vibevoice-multispeaker-tts.md),
  `~/code/coreai/LING_3_0_TINY_STATE.md` (the iOS slice). `coreai doctor`: `AOTC-LOAD-OPTIONS`.
- **(b) Mac, Python runtime: default-options `AIModel.load` of an fp16 asset** segfaults in the
  GPU-delegate JIT (`CompileForDelegates`); an explicit `SpecializationOptions(.gpu)` — which the
  Swift `GraphModel` always passes — is clean. Cause: **Not isolated.** Record:
  [`adcsr-super-resolution.md`](adcsr-super-resolution.md).
- **OS · toolchain:** iOS 27 beta 3–4 (July 2026), `coreai-build` 3600.75.3; macOS 27 beta.

## libc++abi: terminating due to uncaught exception of type std::bad_alloc: std::bad_alloc

A C++ allocation failure the runtime does not turn into an error. Two verified sources.

```
creating engine (first run pays cold GPU specialization)...
libc++abi: terminating due to uncaught exception of type std::bad_alloc: std::bad_alloc
App terminated due to signal 6.
```

- **(a) Cold GPU specialization of a ≥~2 GB bundle on an iPhone without the
  `com.apple.developer.kernel.increased-memory-limit` entitlement.** It dies at the default jetsam
  limit; with the entitlement the same specialization completes (the 2.3 GB Qwen3.5-2B bundle:
  29.1 s cold, 3.0 s warm). The failed attempt leaves the partial cache described under
  [`NSPOSIXErrorDomain Code=2`](#nsposixerrordomain-code2-no-such-file-or-directory) (a). Log:
  `~/code/coreai/ondevice/_pipelined_dev_q2b_r1.log`, `_qwen_prefill_dev_q8_r1.log`,
  `_qwen_prefill_dev_q16_r1.log`; record: [`pipelined-engine.md`](pipelined-engine.md) (Run
  contract). `coreai doctor`: `IPHONE-MEMORY-ENTITLEMENT`.
- **(b) JIT-compiling a large graph on iOS.** `ImageSegmenter(resourcesAt:)` on the hosted
  float16 SAM 3 bundle: MPSGraph constant-folds a transpose for matmul canonicalization and
  `BumpMmapResourceAllocator::allocateResource` throws `bad_alloc` — not a clean jetsam (that would
  be `EXC_RESOURCE`), and not reliably fixed by the entitlement. **Fix:** AOT-compile on the Mac
  (`--platform iOS --preferred-compute gpu --architecture h18p --expect-frequent-reshapes`) and
  ship the `.aimodelc`; the device mmaps the precompiled package with no JIT spike. Record:
  [`sam3-promptable-segmentation.md`](sam3-promptable-segmentation.md) (§4),
  [`swift-runtime.md`](swift-runtime.md).
- **OS · toolchain:** iOS 27 betas, June–July 2026.

## 'mps_spi.copy_discarding_constraints' op input must have tensor constraints

MPSGraph refuses to lower a custom Metal kernel's reshape at engine compile.

```
error: 'mps_spi.copy_discarding_constraints' op input must have tensor constraints   (op_id ~48, early)
```

- **When:** a `TorchMetalKernel` bundle exported with *dynamic* `input_ids`, so a prefill runs at
  S>1 and the kernel's `x.reshape(s, k)` produces a dynamic-row tensor.
- **Verified cause:** the kernel is M=1 (single-row matvec) and the dynamic-row reshape into it
  cannot be constrained by MPSGraph.
- **Fix:** export `--static-ids` — `input_ids` pinned to `[1, 1]`, `position_ids` + KV dynamic —
  and run the prefill as pipelined S=1 steps under `COREAI_CHUNK_THRESHOLD=1`. With S=1 the kernel
  compiles, survives AOT, and runs.
- **Evidence:** record: [`bitcpm-ternary-1.58bit.md`](bitcpm-ternary-1.58bit.md),
  [`ternary-chunked-prefill.md`](ternary-chunked-prefill.md).
- **OS · toolchain:** macOS 27 beta, `coreai-torch` 0.4.0–0.4.1.

## error: 'mps_spi.sdpa' op failed: query and value must have matching inner dimension but have 192 and 128

MPSGraph's fused SDPA takes one head_dim. The first engine run of the converted bundle dies at
lowering, followed by the assertion in the next entry.

```
error: 'mps_spi.sdpa' op failed: query and value must have matching inner dimension but have 192 and 128
.../MPSGraphExecutable.mm:2300: failed assertion `Error: AICode -> MPS lowering failed'
```

- **When:** a model whose `qk_head_dim` differs from its `v_head_dim` — Ling-3.0-tiny's MLA
  (qk 192 = 128 nope + 64 rope, v 128). GLM-4.7-Flash never hit it because it is 256/256.
- **Verified cause:** the fused kernel's constraint. `models/macos/mla_metal_sdpa.py` records the
  same constraint for absorbed MLA.
- **Fix:** store V in the KV cache zero-padded to `qk_head_dim` and slice the extra dims off the
  SDPA output — exact (gate 8 stayed at cos 1.000000000), keeps the fused kernel, costs KV (Ling
  60 → 72 KB/token). Check the padded width against 512 first: 576 trips the
  [ViewOp overflow](#failed-to-acquire-the-source-buffer-for-the-viewop), which is why the absorbed
  case rejects padding.
- **Evidence:** log: `~/code/coreai/_ling3_engine_verify.log`; record:
  [`compression-reference.md`](compression-reference.md) (Pitfalls), `~/code/coreai/LING_3_0_TINY_STATE.md`.
- **OS · toolchain:** macOS 27 beta, 2026-08-21, `coreai-torch` 0.4.2.

## Error: AICode -> MPS lowering failed

`MPSGraphExecutable.mm:2300: failed assertion` at GPU load. The line under it, or above it, names
the op; two triggers verified here.

- **(a) The SDPA head_dim mismatch** in the previous entry.
- **(b) The Core AI SDPA composite over a large prefill query block.** The LFM2-Audio detokenizer
  backbone at S=384 prefill with `SDPA(is_causal, window_size)` aborts at GPU load with this
  assertion and at AOT with a `libODIECompiler` NSException; `window=0` full-causal crashes too, so
  it is not the sliding window. The composite is fine for S=1 decode; the blow-up is large
  query-length attention. **Two escapes:** rewrite as raw matmul-softmax with an explicit additive
  mask (`(q@kᵀ)·scale + mask → softmax → @v`, GQA via `repeat_interleave`; GPU JIT then cos
  1.000000 / 68 dB fp16), or keep the query block small — pocket-tts prefills through a 16-token
  window and never triggers it. Record: [`lfm2audio-port.md`](lfm2audio-port.md),
  [`pocket-tts-port.md`](pocket-tts-port.md). (The LFM2-Audio note spells the string `AICode→MPS`;
  the log prints `AICode -> MPS`.)
- **OS · toolchain:** macOS 27 betas, June–August 2026.

---

**Execution (MPSGraph runtime assertions, Metal)**

## Failed to acquire the source buffer for the ViewOp

MPSGraph plans a fixed per-encode scratch heap; one allocation past it aborts the encode. The two
lines always come together, and the numbers in the first one are the diagnosis:

```
allocateMTLBufferFromMTLHeap: offset 198400 + size 16384 exceeds heap total 212992
.../MPSRuntime/Operations/GPUMemrefOps.mm:687: failed assertion `Failed to acquire the source buffer for the ViewOp'
App terminated due to signal 6.
```

(`GPUMemrefOps.mm:687` on the June builds, `:700` in July, `:707` on 24A5418b.)

- **(a) Gemma 4 12B on Mac, first decode token.** Bisected: `--num-layers 5` (all sliding, head_dim
  256) runs; `--num-layers 6` (adds the first full-attention layer, 16 heads × head_dim 512)
  crashes. The failing buffer is exactly `[1, 16, 1, 512]` fp16 = 16384 B, the full layer's
  `q_proj` output; it scales with the number of full layers and overflows the ~208 KB decode heap,
  invariant to every graph-side change (pad↔replicate, `.contiguous()`, HF vs vanilla SDPA). E2B /
  E4B full layers (8 × 512 = 8 KB) fit. **Fix (verified 2026-06-13):** replace the full layers'
  MPSGraph SDPA with a custom flash-decode Metal kernel; the 12B then runs on the stock pipelined
  engine, greedy-exact against the fp32 oracle. Apple reproduced the overflow on an M2 in July and
  closed the issue 2026-08-26 as not reproducible on a recent macOS beta; the Mac 12B case has not
  been re-measured here since. Log: `~/code/coreai/_gemma12b_llmrunner.log`,
  `_gemma12b_bench_int8.log`, `_gemma12b_bench_int4.log`, `_gemma12b_int4_coherence.log`
  (2026-06-12); issue: [apple/coreai-models#27](https://github.com/apple/coreai-models/issues/27);
  record: `~/code/coreai/GEMMA4_12B_STATE.md`.
- **(b) Gemma 4 E2B `tbl` on iOS, first encode.** The in-graph PLE table gather plus the VL splice
  overflow the same heap (`offset 188160 + size 32768 exceeds heap total 212992`; an earlier build
  reported `heap total 180224`). Model-side squeezing is blind: a 36 KB-smaller dequant chain
  crashed at the byte-identical offset because the savings land in a different heap region.
  **Fix:** move the gather off-graph — provider mode (`PerTokenInputProvider` PLE rows) with
  `image_embeds` still a static buffer; that is the VL device ship. Log:
  `~/code/coreai/ondevice/_gemma_ple_dev_tbl_r1.log` (2026-06-11), `_gemma_ple_dev_vl1.log` …
  `_vl3.log` (2026-06-12); record: [`pipelined-engine.md`](pipelined-engine.md) (iOS per-encode
  scratch-heap ceiling), `~/code/coreai/GEMMA4VL_STATE.md`.
- **(c) Any multi-token (S>1) prefill of Gemma 4 E2B on iOS.** The iOS heap is 145920 B and the
  overflowing allocation scales with query width: chunk 64 → `offset 512 + size 196608` (the
  64·1536·2 hidden buffer), chunk 32 → `offset 98816 + size 98304` (two 32·1536·2 buffers), chunk
  16 clears those and a ~560 KB attention intermediate overflows instead. Only S=1 stays under, so
  a 1024-token prompt degrades to per-token processing. A Qwen3-VL-2B multifunction bundle with a
  static S=64 prefill runs clean on the same phone and build, so it is per-graph under-sizing, not a
  blanket S>1 limit. **Status:** still reproduces byte-identically on 24A5418b with a fresh
  `coreai-torch` 0.4.2 export re-AOT'd with `coreai-build` 3600.82.1 (2/2 runs, 2026-08-27). Open
  as [apple/coreai-models#201](https://github.com/apple/coreai-models/issues/201). Log:
  `~/code/coreai/ondevice/_gemma4_pf32_recheck_2026-08-27_r1.log`, `_r2.log`.
- **(d) Absorbed MLA, 4 layers, Mac** — even the manual einsum attention (not MPSGraph SDPA)
  materializes a 576-dim intermediate ViewOp that overflows the decode heap (`offset 240896 + size
  6144 exceeds heap total 245760`), which is what justified the absorbed-MLA flash-decode kernel.
  Record: `~/code/coreai/ABSORBED_MLA_STATE.md`.
- **(e) MTP spec-decode on iOS** (Gemma 4 mixed-bit + drafter): `offset 1111040 + size 61440
  exceeds heap total 1146880` after 96 clean rounds. Not isolated beyond the signature. Log:
  `~/code/coreai/ondevice/_pipelined_dev_mtp_r1.log`, `_r2.log` (2026-07-03).
- **(f) Alternating entrypoints on iOS** (ternary multifunction bundle, seq → chunk order): the
  second function to run in a process aborts here; the other order aborts in the
  [next entry](#gpumemrefopsmm159-failed-to-resolve-dynamic-dimensions-for-memrefalloc). Both work
  alone; macOS alternates them freely. **Not isolated.** Record:
  [`ternary-chunked-prefill.md`](ternary-chunked-prefill.md) (§6).
- **OS · toolchain:** macOS 27.0 26A5353q (June), iOS 27 24A5380h → 24A5418b; `coreai-build`
  3600.67.5.8.1 → 3600.82.1.

## GPUMemrefOps.mm:159: Failed to resolve dynamic dimensions for memref.alloc

The chunk → seq face of alternating entrypoints on iOS (see (f) above).

- **When:** the ternary multifunction bundle on iPhone 17 Pro: the S=32 chunk prefill runs
  (0.814 GB), then the first S=1 decode step in the same process aborts here. Seq → chunk order
  aborts in the ViewOp assertion instead. Each function alone runs; macOS alternates them freely.
- **Verified:** it is the switch, not either graph. A first guess — the S=1 entrypoint's
  `position_ids` `Dim min=2` rejecting a length-1 call at position 0 — was wrong (positions 0/1/2
  all run seq-first). Cause: **Not isolated.**
- **Fix:** none; the bundle cannot ship on iOS until the switch is understood. Do not quote device
  correctness for a two-arm bundle until both arms run in one process.
- **Evidence:** record: [`ternary-chunked-prefill.md`](ternary-chunked-prefill.md) (§6).
- **OS · toolchain:** iOS 27 beta, July 2026.

## MPSNDArray.mm:893: failed assertion [MPSNDArray, initWithBufferImpl:...] Error: buffer is not large enough.

Three verified causes here, one third-party report; the byte count after `Must be` tells them apart.

```
MPSNDArray.mm:893: failed assertion `[MPSNDArray, initWithBufferImpl:...] Error: buffer is not large enough.
Must be 128 bytes'
```

- **(a) A width-0 model output** (`return x[:, :0]`) on GPU or ANE — `Must be 128 bytes`, at run.
  CPU runs it. Neighbouring constructs (`cat` with a width-0 operand, `new_zeros(n, 0)` into a
  `cat`, indexing a width-0 tensor) are fine. **Fix:** do not emit the output. Issue:
  [apple/coreai-torch#68](https://github.com/apple/coreai-torch/issues/68) (open, model-free
  reproducer); record: [`coreai-zero-sized-dim-abort.md`](coreai-zero-sized-dim-abort.md).
- **(b) The pipelined engine sizes the logits buffer as `ceil(vocab/64)*64`.** BitCPM's vocab
  73448 (% 64 = 40) aborts at engine warm-up with `Must be 146944 bytes`. **Fix:** pad the head
  (gemma4 262144 / qwen3 151936 / Qwen3.6-27B 248320 are all clean). Record:
  [`ternary-chunked-prefill.md`](ternary-chunked-prefill.md) (§5).
- **(c) Two cache states with different last dims** (absorbed MLA: latent 512 + rope 64): the engine
  mis-sizes the smaller one at runtime context > ~256 — `Must be 12320768 bytes`, the rope cache
  `[47,1,1,2048,64]` at trace depth. **Fix:** one combined `[., 576]` state. Record:
  `~/code/coreai/ABSORBED_MLA_STATE.md`.
- **(d) Third-party, not reproduced here:** `Must be 64 bytes` from the `coreai-pipelined` engine
  on a stock `coreai.llm.export Qwen/Qwen3-0.6B` bundle on beta 2 / M5, sequential engine fine,
  invariant to quantization — posted as a comment on
  [apple/coreai-models#27](https://github.com/apple/coreai-models/issues/27). **Not isolated.**
- **OS · toolchain:** macOS 27 betas 2–6, June–August 2026.

## Pass failed: MPSCommonRuntimeCanonicalization

A zero-length `split` section, never used, aborts the process at load on GPU and ANE; CPU runs it.

```
MPSGraphExecutable.mm:4419: failed assertion `Error: Optimize Original Module MLIR pass manager failed
Pass failed: MPSCommonRuntimeCanonicalization
Pass failed: mlir::detail::OpToOpPassAdaptor'
```

- **When:** `a, b = x.split([x.shape[1], 0], dim=1)` with `b` dead. Nobody writes this on purpose:
  it arrives from postprocess code that sizes a section by subtraction — Ultralytics' detection
  head splits `[4, num_classes, extras]` and `extras` is 0 for plain detection models.
- **Verified cause:** the zero-length section specifically (model-free reproducer); the same graph
  with that section omitted runs on the Neural Engine and matches PyTorch to 3.1e-4 px.
- **Fix:** omit the zero-length section. When isolating, pin the compute unit explicitly —
  `SpecializationOptions.default()` can fall back to CPU, and a fallback reads as a pass.
- **Evidence:** issue: [apple/coreai-torch#68](https://github.com/apple/coreai-torch/issues/68)
  (open); record: [`coreai-zero-sized-dim-abort.md`](coreai-zero-sized-dim-abort.md).
- **OS · toolchain:** macOS 27.0 26A5416b, `coreai-torch` 0.4.2, `coreai-core` 1.0.0b2.

## EXC_BREAKPOINT (SIGTRAP, code 5)

The fixed-shape / ANE decode recipe — a KV column written in-graph with `slice_update` at a
runtime `in_step` index — converts fine and dies at the first execute on the WWDC26 betas.

- **Mac GPU:** `EXC_BREAKPOINT` (SIGTRAP, code 5), process exit 133; faulting-thread frames all in
  `CoreAIRuntime`. **iPhone GPU:** SIGSEGV at the first execute (the graph loads and specializes
  first). **iPhone ANE:** `MPSGraphExecutable.mm` → `optimizeOriginalModule` → "MLIR pass manager
  failed" (SIGABRT), which also corrupts the ANE compile cache (next load = ENOENT).
- **Verified cause:** the same attention block, same `slice_update`, same SDPA, exported three ways
  that differ only in the write index: a shape-symint `begin` runs; a runtime-tensor `begin` (dynamic
  or static shapes) traps. Further isolated: what crashes is deriving the write position in-graph from
  runtime data (`arange == in_step` crashes exactly like `slice_update`); a one-hot mask handed in as
  an *input* lowers and runs. Model-agnostic — every model shares `KVCache.update_and_fetch`, and
  Apple's own `KVCacheHandler` (`primitives/ios/cache.py`) uses the crashing form.
- **Fix:** the input-mask blend (`sl.copy_(sl * (1 - m) + col * m)`, mask built on the host;
  fixed shapes *and* Core AI states; 35-layer Gemma 4 E2B 8/8 greedy-exact on the beta Mac GPU), or
  the host-cache pattern (KV as plain I/O, `cat`-append, masked SDPA; runs on Mac GPU, iPhone GPU,
  iPhone ANE chunked). **Status:** Apple said fixed in macOS / Xcode beta 4 and closed the issue
  2026-09-02; not re-verified here on beta 4+.
- **Evidence:** issue: [apple/coreai-models#5](https://github.com/apple/coreai-models/issues/5),
  Apple Feedback FB23024751, repro gist linked there; record:
  [`coreai-beta-mpsgraph-kvwrite-bug.md`](coreai-beta-mpsgraph-kvwrite-bug.md). `coreai doctor`:
  `SRC-DATA-INDEXED-KV-WRITE`.
- **OS · toolchain:** macOS 27.0 26A5353q, Xcode 27A5194q, `coreai-torch` 0.4.0, `coreai-core`
  1.0.0b1 (beta 1).

## The operation couldn’t be completed. (MTL4CommandQueueErrorDomain error 1.)

A Metal 4 command buffer dies; the run either wedges or degenerates. Seen on the Mac and on the
iPhone.

```
	Error: 
	(null)
	The operation couldn’t be completed. (MTL4CommandQueueErrorDomain error 1.)
	<MTL3On4CommandBuffer: 0x7707400000>
Error: command buffer exited with error status.
```

- **(a) Two GPU jobs at once on the beta driver.** A concurrent Python GPU gate made an engine
  probe hang at 0 % CPU and then die with this. **Fix:** run GPU verification solo — hold
  `~/code/coreai/_GPU_LOCK`, check
  `pgrep` for other Python-GPU processes. Export (CPU lowering + quant) is safe to run
  concurrently; only engine load / gate / bench must be solo. Record: `~/code/coreai/GEMMA4_12B_STATE.md`.
- **(b) The Python runtime driving a dynamic-shape decoder per token.** Every step grows
  `position_ids` → a new MPSGraph shape → a re-specialization; on Metal 4 the *second* distinct
  shape faults (`Failed to import MPS module` + this error, Unlimited-OCR), or the rollout
  corrupts around step 25 (Qwen2.5-Omni). **Fix:** make the decode graph fully static (a `pos`
  tensor drives a data-driven `mutable_slice_update`; the engine then compiles once), or reload the
  bundle every 8 calls (the Python-held KV NDArrays survive the reload), or AOT-compile — the 4B
  hybrid decode bundle under Python JIT errored on every forward at 24,000 ms and degenerated to
  token 0 after ~25 tokens; the same bundle AOT-compiled ran at 44.4 ms/forward with zero errors.
  Record: [`unlimited-ocr-rswa-static-decode.md`](unlimited-ocr-rswa-static-decode.md),
  `~/code/coreai/QWEN2_5_OMNI_THINKER_STATE.md`, `~/code/coreai/SPEC35_HYBRID_S_WINDOW_STATE.md`,
  [`spec-decode-hybrid-verify-design.md`](spec-decode-hybrid-verify-design.md) (`--reload-every 3`).
- **(c) Long Python gates** — the per-step re-specialization (~10 s/step with a benign
  ANECCompile-fail → GPU fallback) eventually wedged the MTL4 queue; the Swift engine specializes
  once and has neither issue. Record: [`gemma4-mixedbit-qat-transplant.md`](gemma4-mixedbit-qat-transplant.md).
- **Evidence:** log: `~/code/coreai/_gemma12b_gate_gpu.log`, `_spec_27b_free.log`,
  `_bitvla_engine_gate.log`, `_mineru_decode_gate.log`, `_ling3_dev_gen.log`, and on the iPhone
  `~/code/coreai/ondevice/_pipelined_dev_cv_metal_k1.log`, `_ttft_hostloop32b.log`, `_spec_dev_spec2.log`.
- **OS · toolchain:** macOS 27 betas 1–6, iOS 27 betas.

## Failed to import MPS module

The second-distinct-shape fault of (b) above, as Unlimited-OCR saw it: the first shape compiles and
runs, the second recompile crashes. The fix is the fully static decode graph in
[`unlimited-ocr-rswa-static-decode.md`](unlimited-ocr-rswa-static-decode.md); an earlier reading of
this string as "the custom MoE kernel is broken" was wrong — the kernel runs fine once shapes are
static.

## Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)

The Mac GPU runs out of memory on a JIT-loaded `.aimodel`.

- **When:** the 32-layer ternary-kernel bundle loaded as a plain `.aimodel` (8 layers was fine).
- **Verified cause:** the JIT load; the same bundle routed through `coreai-build compile
  --platform macOS --architecture h16c --preferred-compute gpu --expect-frequent-reshapes` and
  loaded as `.aimodelc` with `SpecializationOptions.default()` runs.
- **Also seen** in the ZAYA-1 8B engine-gate logs (September 2026), where the cause was not isolated.
- **Evidence:** record: [`ternary-chunked-prefill.md`](ternary-chunked-prefill.md) (§5); log:
  `~/code/coreai/_zaya_engine_gate_long.log`, `_zaya_km8_long.log`, `_zaya_chat_engine.log`.
- **OS · toolchain:** macOS 27 betas, July–September 2026.

## CoreAIError 3

The call-time contract of an `InferenceFunction`: an NDArray whose dtype does not match the graph.
Also printed as `CoreAIRuntime error 3`.

- **(a)** fp32 NDArrays into an fp16 bundle (precision follows the traced dtype; the engine gate
  must pass fp16 — in Swift `fillNDArray(as: Float16.self)`, not `Float`). Record:
  [`music-generation-stable-audio.md`](music-generation-stable-audio.md).
- **(b)** int64 inputs where the graph takes int32 — engine int inputs must be int32. Record:
  [`esam3-port.md`](esam3-port.md), [`kokoro-tts.md`](kokoro-tts.md).
- **(c)** fp32 state NDArrays into an fp16 export — state dtype at runtime must match the export
  dtype (`CoreAIRuntime error 3` at the call). Record:
  [`rwkv7-recurrent-linear-attention-coreai.md`](rwkv7-recurrent-linear-attention-coreai.md).
- **OS · toolchain:** macOS 27 betas, June–August 2026.

## CoreAICompiler error 2

The CPU delegate cannot compile the graph (`cpu_only`); `CoreAICompiler error 3` is the same class
on other graphs. GPU compiles them.

- **When:** MinerU's dynamic-shape S=1 decode graph (`cpu_only()` fails fast with error 2 while the
  GPU load wedges in ANECCompile, above); Unlimited-OCR's sym8 Metal MoE graph; a Qwen3.5
  bundle whose GDN recurrence is a `while_loop` ("Compiler error 2" on the macOS 27 beta); the
  RWKV-7 recurrence graphs (`error 3`).
- **Verified cause where known:** a custom Metal kernel is GPU-only by construction; a `scf.while`
  region fails the delegates (next entry). For the plain dynamic decode graphs: **Not isolated.**
- **Fix:** the GPU compute unit (serialized under `_GPU_LOCK`).
- **Evidence:** record: [`mineru-port.md`](mineru-port.md),
  [`unlimited-ocr-rswa-static-decode.md`](unlimited-ocr-rswa-static-decode.md),
  [`rwkv7-recurrent-linear-attention-coreai.md`](rwkv7-recurrent-linear-attention-coreai.md);
  issue: the 2026-08-30 comment on [apple/coreai-models#212](https://github.com/apple/coreai-models/issues/212).
- **OS · toolchain:** macOS 27 betas, June–August 2026.

## 'scf.while' region type mismatch

The MPSGraph GPU delegate rejects a `torch.ops.higher_order.while_loop`.

- **When:** batched (S>1) prefill of a GatedDeltaNet hybrid (Qwen3.5 / 3.6 / 3.8): the recurrent
  scan exports as `while_loop`.
- **Verified cause:** the loop region; at S=1 the scan is a single step, and a loop-free single-step
  recurrence is numerically identical and removes `scf.while` from the graph entirely. That is why
  the zoo's GDN bundles are decode-only S=1 exports with the prompt run as pipelined S=1 steps.
- **Fix:** the loop-free S=1 export; a chunkwise-parallel (scan-free) prefill formulation would
  restore batched prefill and is unsolved here. Apple's known issue 177354777 ("inference might
  fail/crash for control-flow over dynamic-shape tensors") is the same class.
- **Evidence:** issue: the 2026-08-30 comment on [apple/coreai-models#212](https://github.com/apple/coreai-models/issues/212);
  record: [`pipelined-engine.md`](pipelined-engine.md) (The export trick). `coreai doctor`:
  `SRC-WHILE-LOOP`.
- **OS · toolchain:** macOS 27 betas.

---

**The Swift engines (`apple/coreai-models`: `llm-runner`, pipelined, sequential)**

## Shape at dimension 1 of 256 is not a valid substitution for source shape 1

A Swift trap while preparing the asset, before any token is bound. The `256` is a constant, not
your prompt length — that is the fingerprint.

```
⏳ Preparing AI asset from source...CoreAIRuntime/NDArrayDescriptor.swift:139: Fatal error: Shape at dimension 1 of 256 is not a valid substitution for source shape 1
```

(`NDArrayDescriptor.swift:136` on current `main`.)

- **(a) The stock pipelined engine on any bundle whose `main` emits logits as a static `[1, 1,
  vocab]`** (decode-only S=1 graphs — every zoo GDN bundle). `CoreAIPipelinedEngine.swift` builds
  `GrowingLogitsBuffer` with `initialCapacity: averageExpectedPromptSize` (256) and
  `TensorStorage+CoreAI.swift` resolves `[1, 256, vocab]` against the static descriptor. **Fix:** the
  fork's five-line `logitsSeqIsStatic` guard (john-rocky/coreai-models@9e5b605; rebased on current
  `main` as 948d5e9) plus `COREAI_CHUNK_THRESHOLD=1`. Upstream triaged it as a feature request
  outside the pipelined engine's scope ("designed for dynamic inputs"), pointed at the incoming
  `prefill` graph export (apple/coreai-models#211) as the likely path, and closed it 2026-09-04 —
  so plan on the fork guard. Log: `~/code/coreai/_zoo04_harness/bench212/qwen3_5_0_8b/unpatched.log`,
  `…/qwen3_5_4b/unpatched.log`, `_zoo04_harness/validation227-20260905/logs/before-s1.log`,
  `before-pipette27b.log`, `_nanbeige42_speed.log`, `_zaya_engine_gate.log`; issue:
  [apple/coreai-models#212](https://github.com/apple/coreai-models/issues/212); record:
  [`pipelined-engine.md`](pipelined-engine.md) (Run contract).
- **(b) `llm-runner`'s default warmup** submits a synthetic 256-token prefill that a static-S=1
  graph cannot serve; the known-good qwen3.6 decode bundle fails the same way, which proves it is
  the driver. **Fix:** `--warmup off`, or `--warmup exact --warmup-length 1`; `llm-benchmark` warms
  through a real trial and is safe; `coreai_gate.py` disables warmup deliberately; in an app, a
  1-token generate after load is the warmup. Never call `engine.warmup()` on an S=1 bundle.
  Record: [`lfm2.5-2.6b-port.md`](lfm2.5-2.6b-port.md), `~/code/coreai/GLM_4_7_FLASH_STATE.md`,
  [`coreai-torch-041-ir-incident.md`](coreai-torch-041-ir-incident.md) (Non-obvious things the gate encodes).
- **(c) The guard without `COREAI_CHUNK_THRESHOLD=1`:** construction passes and the first prompt
  traps with `… of 128 is not a valid substitution …` — the whole prompt was bound to the static
  `input_ids` because 128 tokens is under the default chunk threshold (1024). Log:
  `~/code/coreai/_zoo04_harness/bench212/qwen3_5_0_8b/patched-defaultchunk.log`,
  `_zoo04_harness/bench212-run2/bench212-fresh.log`.
- `coreai doctor`: `GRAPH-S1-RUN-CONTRACT`.
- **OS · toolchain:** every `coreai-models` revision from June through `de31ba5` (Xcode 27 beta 5),
  macOS 27 betas.

## invalidOutputType("Expected 2 states (KV cache), got 4: [keyCache, valueCache, convState, recState]")

Engine creation fails for every hybrid bundle on the stock engines.

- **When:** any 4-state hybrid (the qwen3.5 / 3.6 family, LFM2.5, Granite-4.0-H / Mamba2) through
  the factory's `coreai-sequential` variant, or through the stock `apple/coreai-models` v0.1.0
  pipelined engine that `coreai-kit`'s `Package.swift` pulled.
- **Verified cause:** the engine hard-requires exactly two states.
- **Fix:** the fork's extra-states patch relaxes it to "at least 2" with conv / recurrent state
  handling (`CoreAIPipelinedEngine.swift`); route hybrids to the patched `coreai-pipelined`
  variant. `COREAI_CHUNK_THRESHOLD=1` is read live at prefill time — set it for the pipelined load
  and restore the launch value before a sequential load in the same process. CoreAIChatMac's
  pattern: try sequential, catch this error, retry pipelined.
- **Evidence:** record: `~/code/coreai/SPOT_RAG_STATE.md`, [`pipelined-engine.md`](pipelined-engine.md)
  (Driving hybrid bundles from an APP). `coreai doctor`: `GRAPH-STATE-COUNT`.
- **OS · toolchain:** `coreai-models` v0.1.0 through July 2026.

---

**Around the runtime (disk, `devicectl`, `swift-transformers`, FoundationModels)**

## LLVM ERROR: IO failure on output stream: No space left on device

Core AI's compile paths write multi-GB scratch and cache; when the volume fills, the failure
surfaces here — or as something that looks like a bug. Four verified sources.

```
LLVM ERROR: IO failure on output stream: No space left on device
```

- **(a) `~/Library/Caches/coreai-cache` on the Mac** caches a multi-GB compiled graph *per
  shape*; a dynamic-shape decode loop in the Python runtime reached 423 GB and filled the data
  volume (a SIGSEGV with this text). **Fix:** `rm -rf ~/Library/Caches/coreai-cache` before long
  dynamic-shape runs (regenerable; shapes are shared across clips so the footprint is one-time).
  Record: [`qwen2.5-omni-audio-understanding.md`](qwen2.5-omni-audio-understanding.md).
- **(b) MPSGraph JIT scratch under `/private/var/folders/*/T/com.apple.MetalPerformanceShadersGraph`**
  is not cleaned on exit: it grew to 257 GB (one 0.8B S=9 JIT compile left 103 GB) and killed a run
  with this line. MPSGraph ignores `TMPDIR`. **Fix:** sweep dead-PID `mpsgraph-<pid>-…` directories.
  Record: `~/code/coreai/SPEC35_HYBRID_S_WINDOW_STATE.md`.
- **(c) On-device GPU specialization of a 4B palettized iOS IR** exhausts the iPhone's scratch
  disk mid-compile. **Fix:** AOT-compile per device class and ship the `.aimodelc`. Log:
  `~/code/coreai/ondevice/AppleBenchRunner/_device_fastcontext_cold3.log`,
  `~/code/coreai/ondevice/_coreaichat_qwen2b_r1.log`; record:
  [`aot-and-specialization.md`](aot-and-specialization.md) (The 4B wall).
- **(d) A `save_asset` on a full Mac volume** (the DiffusionGemma encoder export). Log:
  `~/code/coreai/_dg_sp128_export.log`.
- **Disguised forms:** the converter's `Indexing.swift: interleave must have rank (1)` and a
  runtime `No space left on device` are both temp-write failures — check `df -h
  /System/Volumes/Data` and the MPSGraph scratch dir before reading either as a bug. Record:
  [`lfm2audio-port.md`](lfm2audio-port.md).
- **OS · toolchain:** macOS 27 betas, iOS 27 betas, June–August 2026.

## unsupportedTokenizer

`swift-transformers` rejects an unregistered `tokenizer_class`.

- **When:** a bundle whose `tokenizer_config.json` names a class the Swift library does not know
  (`ParakeetTokenizer`).
- **Verified cause:** the class name is matched against `knownTokenizers`.
- **Fix:** retag to a registered class — `PreTrainedTokenizer` → `BPETokenizer`; decode is driven by
  `tokenizer.json`'s decoder, so it stays exact. Do it in the upload script.
- **Evidence:** record: [`ship-playbook.md`](ship-playbook.md) (Cross-cutting traps),
  `~/code/coreai/PARAKEET_STATE.md`. `coreai doctor`: `TOKENIZER-CLASS-UNREGISTERED`.
- **OS · toolchain:** swift-transformers, June 2026.

## GenerationError.decodingFailure

FoundationModels rejects what the model generated ("failed to parse generated content").

- **When (all on the kit engine behind `LanguageModelSession`):** small / thinking models emitting
  tool-call JSON, independent of the argument schema (verified with required, optional and empty
  `@Generable` arguments); `/no_think` on the qwen3 kit bundles, even the 0.6B on a tool-free
  respond; a thinking model whose token budget runs out inside `<think>`; a plain
  `LanguageModelSession(model:instructions:)` first respond where the profile path is solid.
- **Verified:** the tool-call rejection is independent of the argument schema (tested with
  required, optional and empty `@Generable` arguments). The other triggers are observations from
  the same sessions, not isolated further.
- **Fix:** make the model-decision channel guided generation (`@Generable`), not a tool; budget
  `maxTokens` for the thinking span; do not use `/no_think` as a mitigation on this stack.
- **Evidence:** record: [`dynamic-profiles-local-models.md`](dynamic-profiles-local-models.md),
  `~/code/coreai/DUAL_PROFILE_STATE.md`, `~/code/coreai/ENGINE_D1_STATE.md`.
- **OS · toolchain:** macOS 27 / iOS 27 betas, FoundationModels, June 2026.

## CoreDeviceError 4016

`devicectl` cannot install or launch: the device screen is locked. Set Auto-Lock to Never for a
device session. (`… powerAssertionTaken` in the same domain means the device was already going
down.) Record: [`bitvla-1.58bit-vla.md`](bitvla-1.58bit-vla.md), `~/code/coreai/SPEC35_HYBRID_S_WINDOW_STATE.md`.

---

## When there is no string

Some aborts print nothing useful. What was learned about each:

- **`App terminated due to signal 6.`** is `devicectl`'s report of any SIGABRT; the cause is the
  line above it (the ViewOp assertion, `bad_alloc`, the mmap'd-buffer error). If there is no line
  above it, `print` output died in a block-buffered stdout with the process: write diagnostics with
  `fputs(…, stderr)` + `fflush`, one line before each suspect call, and do not pipe `--console`
  through `tail`. Record: [`ship-playbook.md`](ship-playbook.md) (Cross-cutting traps).
- **A C++ abort in the runtime is not a Swift throw** — `try?` never sees it. Bracket load → allocate
  → first run with flushed prints. Magenta RealTime 2's real-weight frame graph loads (46 ANE
  regions) and aborts on the first run under every compute preference; bisected to a hard boundary
  (11 temporal layers run, 12 abort), not isolated further. Record:
  [`magenta-rt2-port.md`](magenta-rt2-port.md) (§7b).
- **`signal 9` with nothing else** on an iPhone is jetsam: an 18 GB int4 35B is killed during its
  ~26-min cold compile on ~12 GB of RAM. The on-device ceiling is ≈5–6 GB (int4 8B-class). Record:
  [`dense-int4km-flagship-session-findings.md`](dense-int4km-flagship-session-findings.md).
- **A SIGSEGV at `AIModel(contentsOf:options:)` with no text** on iOS is the
  `expectFrequentReshapes` hint on a fixed-shape graph, above.

## Maintaining this page

Add an entry when a run prints a string that is not here, in the section for the stage that
printed it; the H2 is the string verbatim, backticks dropped. Fill the cause only with what was
isolated; write "Not isolated" otherwise. Name the log file or the issue. When Apple fixes one, keep
the entry and add the build the fix was *measured* on — a release note is not a measurement.
`cli/coreai_doctor.py` links its rules here by anchor, and `cli/selftest.py` fails if an anchor
it links to disappears, so rename a heading only together with the doctor.
