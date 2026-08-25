# Core AI overview

**Core AI** is Apple's Core ML successor, announced at WWDC 2026 (iOS/macOS 27). It keeps the
"convert once, run on ANE/GPU/CPU" idea but replaces the `.mlpackage` + coremltools stack with a
new IR, compiler, and runtime.

## The three Apple repos (open) + the closed runtime

| Repo | Core ML analog | Role |
|---|---|---|
| `coreai-torch` | coremltools converter | PyTorch → Core AI IR. Entry: `TorchConverter().add_exported_program(...).to_coreai()`. Extension points: `register_torch_lowering`, `composite_ops` (SDPA, RoPE, RMSNorm, GatedDeltaUpdate, GatherMM), `ExternalizeSpec`. |
| `coreai-optimization` (`coreai-opt`) | coremltools.optimize | quant / palettization / pruning (torchao PT2E). |
| `coreai-models` | — | Apple's own model zoo + Swift runtime + agent skills. |

The **compiler + runtime are closed-source**, shipped as the `coreai-core` Python wheel (the
`coreai.runtime` module) + the OS Core AI framework (`CoreAI.framework`, on-device).

## The `.aimodel` bundle

A `.aimodel` is a **directory bundle**: `{metadata.json, main.mlirb, main.hash}` (the IR + a
manifest). It can hold multiple **functions** (entrypoints) and declares **states** (tensors the
graph mutates in place, surfaced via a `state=` API at runtime) — this is how KV caches live.

## The pipeline

```
PyTorch (re-authored model)
  → coreai-opt (optional compress: palettize / quantize)
  → coreai-torch TorchConverter → Core AI IR
  → .optimize() → save_asset() → .aimodel
  → [Python] coreai.runtime  (macOS, for convert/verify)
  → [Swift]  CoreAI.framework (on device, iOS/macOS 27)  ── AOT-compiled by `aimodelc`
```

- **macOS 26+ on Apple silicon is enough to convert + run (Python) + numerically verify.** The
  `coreai-core` wheels are tagged `macosx_26_0_arm64` (PyPI also carries `manylinux_2_34_x86_64`
  Linux wheels — convert-on-Linux untested by us). Verified on macOS 26.5.2 (2026-08-26,
  ultralytics CI run 32907390789): coreai-torch 0.4.2 exported + loaded + validated 8 YOLO task
  models, detect mAP identical to macOS 27.0 beta. On-device iOS / the Swift runtime need
  iOS/macOS 27; the AOT compiler (`aimodelc`, ships in Xcode 27) runs on macOS 26.4+ — see
  [`swift-runtime.md`](swift-runtime.md).
- Python runtime (sketch): `prog.save_asset(Path(out), rt.AIModelAssetMetadata())`;
  `model = await rt.AIModel.load(path, rt.SpecializationOptions.cpu_only())`;
  `fn = model.load_function("main")`; `res = await fn({"x": rt.NDArray(arr)}, state=...)`.

## Developer toolchain & app integration (WWDC 324/326)

- **Core AI Debugger** (separate macOS app) — visualize the converted graph, inspect intermediate tensor
  values, and **trace each op back to the Python source line** that introduced it. Plus an in-Xcode **debug
  gauge** (streaming Core AI activity) and **Core AI Instruments** for profiling inference + specialization.
- **AOT compile** — `xcrun coreai-build compile … → .aimodelc` (see
  [`aot-and-specialization.md`](aot-and-specialization.md)); **specialization** is the on-device first-run
  compile, managed via `AIModelCache` / `AIModel.specialize()`.
- **Foundation Models integration** — `CoreAILanguageModel` (from `coreai-models`'s Swift `CoreAILM`) plugs
  your own `.aimodel` into the **same `LanguageModelSession` API** as Apple's built-in on-device LLM: same
  `session.respond(to:)`, streaming, and **`@Generable` guided/structured generation**. So a custom model
  gets the Foundation Models ergonomics. (For non-standard architectures the high-level pipeline can't
  express — multi-state SSMs, dual-KV — drop to the low-level `CoreAI` framework; see
  [`swift-runtime.md`](swift-runtime.md).)
  *Update 2026-06-11, verified:* with the pipelined-engine patch stack, the non-standard bundles (hybrid
  Qwen3.5, SSM Granite/LFM) DO run behind `LanguageModelSession` too; note guided generation requires
  engine logits, which GPU-pipelined bundles don't expose — details, tool-calling recipe, and traps in
  [`fm-provider.md`](fm-provider.md).

## Why this repo exists

Apple's `coreai-models` zoo lags ~one generation (Qwen3 / Gemma 3, no VLM) and its Swift runtime
assumes standard `input_ids → logits` + single KV models. Newer architectures (hybrid
linear-attention SSMs, dual-KV + per-layer-embedding decoders, VLMs) need re-authoring + a runner
that handles non-standard states. That's what `conversion/`, `swift/`, and `zoo/` provide.
