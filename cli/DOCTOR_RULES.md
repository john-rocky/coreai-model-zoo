# The Core AI conversion traps that pass every gate

*Extracted 2026-07-31 from 80 notes in `coreai-models-community/knowledge/`, 107 conversion
scripts, the shipped Swift runtimes, and the 17 issues this project filed against
`apple/coreai-torch` and `apple/coreai-models`.*

A conversion that errors is cheap. You get a stack trace, you fix it, you move on.

The expensive class is the other one: `torch.export` succeeds, `TorchConverter` succeeds,
`save_asset` writes a bundle, the bundle loads, the model generates fluent text — and the
numbers are wrong, or the app never stops generating, or it works on your Mac and produces
garbage on the phone. Nothing in the log says so. You find out from a benchmark score you
can't explain, or from a user.

Every row below is one of those. Each is something that already happened here, with a
citation to the note or upstream issue that recorded it. The point of writing them as a
table is that a table can be executed: `coreai_doctor.py` next to this file implements the
rows marked ✅.

**How to read the severity column.** It answers "what happens if I ship this without
acting":

| | |
|---|---|
| **fatal** | it will not convert, will not load, or will not execute |
| **silent** | it converts, loads and runs — and the numbers are wrong |
| **runaway** | it runs and produces output, but the app-level behaviour is broken |
| **perf** | it works, and is slower or larger than it needs to be |
| **requires** | the artifact is fine; whatever drives it must do something specific |
| **info** | worth knowing before you ship |

---

## 1. The asset itself

One row here is true only on some OS 27 builds, and says so. A 0.4.0-era asset loads on OS 27 beta 1 and is refused at load by every build measured since — through macOS 26A5416b on 2026-09-04, both at `AIModel.load` and at `coreai-build compile`, although Apple's beta 5 release notes list the incident (177008303) as fixed. Doctor reads the host build from `sw_vers` and reports `fatal` on a build that refuses the artifact; `--host-build 26A5353q` judges against another build. A release build inherits nothing: `IR040_MEASURED_OK_FROM` in `coreai_doctor.py` names the first build a sweep measures to load such an asset, and stays unset until one does.

What a `.aimodel` or `.aimodelc` directory says about its own provenance. Cheap to check,
and it is where the single largest incident in this project's history lives.

| ✅ | ID | Symptom you experience | Detection | Severity | Source |
|---|---|---|---|---|---|
| ✅ | `IR-040-DEBUG-LOC` | Asset loaded fine on beta 1. On every OS 27 build since, every load and every `coreai-build compile` aborts with `expected AICode versioned location` → `Failed to convert to versioned IR` → `LLVM ERROR: cannot unwrap empty odiec_module_t`. `coreai-build inspect` still reads it perfectly, which makes it look recoverable. | `.aimodel/metadata.json` has no `producer` key. A 0.4.1+ asset carries `{"producer": "coreai-core 1.0.0b2", …}`; a 0.4.0 one carries only `assetVersion`. Audit a whole tree on that field alone. **Severity follows the host OS build**: `info` on beta 1, `fatal` from beta 2 on (measured through 26A5416b, 2026-09-04 — Apple's beta 5 note did not change it); flips on a release build only once a sweep measures one loading it. | fatal / info | [coreai-torch#37](https://github.com/apple/coreai-torch/issues/37), [#44](https://github.com/apple/coreai-torch/issues/44); `knowledge/coreai-torch-041-ir-incident.md`; `RECOVERY_STATUS.md`; `models/_SMOKE.json` |
| ✅ | `AOTC-STALE-TOOLCHAIN` | A compiled `.aimodelc` that worked on beta 2 fails to specialize on beta 3 and later. | `metadata.json` `producer: coreai-build-<ver>` is older than Xcode 27 beta 3's `3600.75.3` — the artifact was compiled by beta 2 or earlier, the class Apple 181264112 describes: **fatal**. A producer merely older than the *installed* toolchain is **info**, not a defect — every published h18p artifact was compiled on beta 3 and loads on every later device build; whether a release toolchain still specializes it is a device check. | fatal / info | `knowledge/coreai-torch-041-ir-incident.md` (Apple 181264112) |
| ✅ | `AOTC-NO-SOURCE-ASSET` | The cheap in-place repair does not apply, and you discover it during an incident. | A `.aimodelc` with no sibling `.aimodel`. `strip_debug_info` works on source assets only; compiled-only artifacts need a full re-export plus AOT recompile. | info | `RECOVERY_STATUS.md` ("NOT recoverable by strip") |
| ✅ | `ASSET-SYMLINK` | Loads from Python, `failedToSpecialize` from Swift. | The asset path (or its parent) is a symlink. `AIModel(contentsOf:)` does not follow symlinks, so it cannot resolve `main-<arch>-delegates`; `rt.AIModel.load` does. Every Python gate passes. | fatal | `GLM_IMAGE_KICKOFF.md` Swift Phase 3 |
| ✅ | `AOTC-LOAD-OPTIONS` | The AOT bundle you compiled specifically to avoid on-device compilation… compiles on device, and wedges. | Any `.aimodelc`. It must be loaded with `SpecializationOptions.default()` or `.cpuOnly()`. `init(preferredComputeUnitKind: .gpu)` re-specializes the baked graph. Separately, asking for `expectFrequentReshapes` at load time on a fixed-shape graph SIGSEGVs the on-device MPSGraph compiler — and compiling *with* `--expect-frequent-reshapes` does not make the runtime hint safe. | requires | `GLM_IMAGE_KICKOFF.md`; `knowledge/aot-and-specialization.md` |
| — | `AOTC-ARCH-NAME` | An `.aimodelc` fails to load on the device with `invalidCompiledModel`. | `--architecture` tracks the **device identifier** major version, not the marketing name: iPhone 17 Pro is `iPhone18,1` → `h18p`; an M-series Mac is `Mac16,x` → `h16c`. `coreai-build compile` exits 0 for *any* arch you name, so a successful compile validates nothing. Doctor reports the compiled arch as a note. | fatal | `knowledge/aot-and-specialization.md` (device-validated 2026-06-10) |
| — | `DEVICE-PARTIAL-COPY` | Every load of a model errors `NSPOSIXErrorDomain Code=2`, even after you re-copy the file, even after you rename it. | A load attempt against a partially-copied `.aimodel` permanently poisons the content-hash-keyed on-device specialization cache. After every multi-GB `devicectl` push, list the destination and confirm `main.mlirb` is full size *before* launching anything that loads it. | fatal | `knowledge/swift-runtime.md` |

## 2. The graph, read with `coreai-build inspect`

`xcrun coreai-build inspect <asset> --ops --json` returns states, IO shapes and the full
operation distribution in well under a second, even for a 4.6 GB asset — and it reads
assets that cannot be loaded. It is the cheapest structural check available and it is
almost never used.

| ✅ | ID | Symptom you experience | Detection | Severity | Source |
|---|---|---|---|---|---|
| ✅ | `GRAPH-FLOOR-IDENTITY` | Coordinate maths is subtly wrong on GPU and exactly right on CPU, from the same asset. | `operationDistribution` contains `floor`, `trunc` or `ceil`. On the GPU delegate all three execute as the **identity function**. The floor that survives every unit is `torch.div(x * 2.0, 2.0, rounding_mode="floor")`. | silent | [coreai-torch#10](https://github.com/apple/coreai-torch/issues/10) |
| ✅ | `GRAPH-ROUND-TIES` | A quantize path is off by one LSB against torch. | `round` in the op distribution: the GPU delegate rounds ties away-from-zero, torch rounds to even. | info | [coreai-torch#10](https://github.com/apple/coreai-torch/issues/10) |
| ✅ | `GRAPH-STATE-COUNT` | `invalidOutputType("Expected 2 states … got 4")` at engine-create, in the app, after everything passed on the CLI. | The function declares a number of states ≠ 2. Apple's `coreai-sequential` variant hard-requires exactly two; every 4-state hybrid (the whole qwen3.5 family, Granite) needs the pipelined engine plus the extra-states patch. | requires | `knowledge/pipelined-engine.md`; `knowledge/stateful-kv-cache.md` |
| ✅ | `GRAPH-S1-RUN-CONTRACT` | `NDArrayDescriptor` fatal on the first inference, or `Shape at dimension 1 of 256 is not a valid substitution for source shape 1`. | `input_ids` is static `[1,1]`. The engine's default warmup prefills 256 tokens, which an S=1 graph cannot serve. Needs `COREAI_CHUNK_THRESHOLD=1` set **before engine creation**, no `engine.warmup()`, and `--warmup exact --warmup-length 1` under `llm-runner`. | requires | `knowledge/pipelined-engine.md`; `knowledge/coreai-torch-041-ir-incident.md` |
| ✅ | `GRAPH-VOCAB-MISMATCH` | Occasional wrong glyphs; never an error. | `metadata.language.vocab_size` ≠ the graph's logits width. The engine sizes its sampler from metadata and indexes the real logits buffer. | silent | `knowledge/pipelined-engine.md` |
| ✅ | `GRAPH-TOKENIZER-OVERFLOW` | Special markers silently become `[UNK]`. | Highest tokenizer id ≥ logits width. Harmless when the overflow is a multimodal placeholder a text-only port never sees (Gemma-3's `<image_soft_token>` sits at 262144, one past the head); a real bug when those markers carry meaning, as GLiNER's `[P] [E] [SEP_TEXT]` do. | info | `knowledge/gliner2-pii.md` |
| ✅ | `GRAPH-CUSTOM-METAL-KERNEL` | An ANE deployment plan that cannot work. | `metal4_kernel` in the op distribution. Custom MSL is GPU-only by construction — the ANE runs fixed hardware ops. (It *does* survive AOT: the `.aimodelc` embeds the MSL and outputs are bit-identical.) | requires | `knowledge/custom-metal-kernels.md`; `knowledge/compute-units-and-authoring.md` |
| ✅ | `GRAPH-DYNAMIC-KV-IOS-2048` | Token-exact on Mac. On iPhone, soup from the first generated token, at full pipeline speed. Device gates pass because they cap generation at 256. | A dynamic-seq KV state plus `max_context_length ≥ 2048`. The iOS on-device compiler miscompiles this graph class once the **bound** KV seq dim reaches 2048 — however that shape is reached. Capacity ≤ 1024 is clean every time; `.fixedSize(4096)` is *worse*, because then every generation rides the broken shape; a full cache evict plus fresh compile still soups. | requires | [coreai-models#124](https://github.com/apple/coreai-models/issues/124); coreai-kit#5; the engine guard in `coreai-models` `bd8dcf7`; `NANBEIGE42_PR6_STATE.md` |

## 3. The bundle wrapper

| ✅ | ID | Symptom you experience | Detection | Severity | Source |
|---|---|---|---|---|---|
| ✅ | `BUNDLE-INCOMPLETE` | `LanguageBundle`/`EngineFactory` will not load it. | `kind: llm` without the full contract: `metadata.json` (`assets.main`, `language.{tokenizer,vocab_size,max_context_length,function_map}`) **and** a `tokenizer/` directory. A bare `.aimodel` directory is not a bundle. Only hold `kind: llm` to this — a `vision-encoder` or `llm-drafter` ships neither by design. | fatal | `knowledge/pipelined-engine.md` |
| ✅ | `BUNDLE-ASSET-MISSING` | Load fails on an otherwise complete-looking directory. | `assets.main` does not resolve on disk. | fatal | `knowledge/pipelined-engine.md` |
| ✅ | `CHAT-TEMPLATE-MISSING` | The model answers, badly, and your benchmark number is meaningless. | No `chat_template.jinja` and no `chat_template` key. `--apply-chat-template` defaults to true and **does not warn when there is nothing to apply** — the model silently runs raw completion and never sees turn markers. | runaway | `knowledge/cross-runtime-quality-benchmarking.md` |
| ✅ | `EOS-NOT-EMITTED-BY-TEMPLATE` | Every answer runs to `maxTokens`. The first snapshot is a wall of repeated `…Paris.<turn|>`. | The declared `eos_token` string never appears in the chat template, and the template does not reference the `eos_token` variable either. Swift's `StopSequences(for:)` stops only on `tokenizer.eosTokenId`, so a document-end eos means the app never sees the turn terminator. Gemma-4 declares `<eos>` (id 1) and ends turns with `<turn|>` (106); Gemma-2/3 use `<end_of_turn>`; ChatML models declare `<\|endoftext\|>` and end turns with `<\|im_end\|>`. | runaway | `GEMMA4_12B_STATE.md`; `knowledge/minicpm5-1b.md` |
| ✅ | `CHAT-TEMPLATE-ABSENT` | Nothing — usually. | The `info`-severity twin of the row above, for a bundle whose source model does not look instruction-tuned. An ASR, OCR or drafter decoder legitimately has no chat template; a chat model with none is the `runaway` above. The tool splits on the `source.hf_model_id` in the bundle metadata, which is a heuristic and is stated as one in the finding. | info | `knowledge/cross-runtime-quality-benchmarking.md` |
| ✅ | `BUNDLE-NOT-A-MANIFEST` | A lint that shouts about a file that never claimed to be a bundle. | `metadata.json` with no `metadata_version` and no `assets` is some other sidecar sharing the name (ship manifests for the audio and TTS ports do this). Lint the assets in the directory instead of holding it to a contract it never made. | info | `knowledge/pipelined-engine.md` |
| ✅ | `TOKENIZER-CLASS-UNREGISTERED` | `unsupportedTokenizer` — or worse, no error and wrong tokenization. | `tokenizer_class` (with a `Fast` suffix stripped) is not in swift-transformers' `knownTokenizers`. The strict path throws; **the non-strict fallback is BPE for everything**, which is silently wrong for a SentencePiece/Unigram or WordPiece model. `DebertaV2Tokenizer` → retag to `XLMRobertaTokenizer` to reach the real Unigram implementation. | fatal | `knowledge/ship-playbook.md`; `knowledge/gliner2-pii.md`; swift-transformers `Sources/Tokenizers/Tokenizer.swift` |
| ✅ | `IOS-LARGE-GRAPH-JIT` | On-device load stalls for minutes, or dies with `LLVM ERROR: No space left on device` / `Failed to allocate mmap'd buffer` / a `BumpMmapResourceAllocator` SIGABRT. | A GB-class portable `.aimodel` with no `.aimodelc` alongside. **This is graph-shaped, not size-shaped** — a 4.6 GB static-S=1 int8 pipelined bundle JITs on iPhone in ~54 s, while a 4B dense GPU bundle exhausts the scratch disk and a 2 GB-constants graph fails mmap allocation. Measure a cold specialization; AOT if it aborts. | requires | `knowledge/aot-and-specialization.md`; `knowledge/ship-playbook.md`; `NANBEIGE42_PR6_STATE.md` (the counter-case) |
| ✅ | `IPHONE-MEMORY-ENTITLEMENT` | Cold specialization dies with `std::bad_alloc` (SIGABRT); afterwards *every* attempt fails `NSPOSIXErrorDomain code=2`. | Asset ≥ ~2 GB. Needs `com.apple.developer.kernel.increased-memory-limit`. The follow-on ENOENT chain is partial e-caches, not a payload problem — carry a `Library/Caches/coreai-cache` wipe hook. | requires | `knowledge/pipelined-engine.md` |
| — | `BUNDLE-IN-APP-RESOURCES` | "Invalid bundle" at install; or CodeSign fails with "code object is not signed at all". | An `.aimodel` **directory** embedded in an app bundle — the installer misreads the extension-suffixed root as a nested bundle. And never name a folder `Resources/` at the iOS bundle root. Ship via download or sideload to Documents. | fatal | `knowledge/conversion-guide.md` |
| — | `SPEC-CACHE-CROSS-CONTAMINATION` | Adding one model breaks a *different*, already-working model with `Unable to use cached specializations and original module not available`. | Ingesting a multi-GB AOT bundle into a container can invalidate other models' cached specializations. Expect one wipe + re-spec cycle after adding one. | fatal | `knowledge/pipelined-engine.md` |

## 4. The source checkpoint, before you export anything

| ✅ | ID | Symptom you experience | Detection | Severity | Source |
|---|---|---|---|---|---|
| ✅ | `QAT-STATIC-ACTIVATION-SCALES` | Recall is fine ("capital of France" → Paris). Multi-step reasoning collapses: **GSM8K 48% where the vendor's own runtime scores 86% on the same weights**. Every gate passes. | The checkpoint carries `input_activation_scale` / `output_activation_scale` / `k_cache_scale` / `v_cache_scale` tensors. These weights were QAT-trained **with a static activation clamp in the loop** — the clamp is a learned outlier suppressor, not a precision loss to recover. Exporting them into an fp16-activation graph removes it. Read the tensor names from the safetensors headers; no weights needed. | silent | `knowledge/gemma4-wna8o8-requires-int8-activations.md` |
| ✅ | `EOS-SOURCE-MISMATCH` | See `EOS-NOT-EMITTED-BY-TEMPLATE` — this is the same defect, caught one stage earlier. | `generation_config.json` lists more `eos_token_id`s than `tokenizer_config.json`'s single `eos_token` resolves to (Gemma-4: `[1, 106, 50]` vs `<eos>`). Decide which id ends a *turn* and retag at export time, in `save_tokenizer`, after the verbatim HF copy. | runaway | `GEMMA4_12B_STATE.md` |
| ✅ | `CHECKPOINT-PREQUANTIZED` | You publish a comparison and it measures the checkpoint, not the runtime. | `config.json` carries a `quantization_config`. "int4" named three different products in one comparison here: unquantized-QAT, mobile `wNa8o8` (2-bit decode layers + optimized KV + static int8 activations), GGUF Q4_0, and w4a16. The mobile variant is a co-designed weights+runtime package, not a bit width. To build a matched pair, compile every arm from the same *unquantized* checkpoint at the same block size. | info | `knowledge/cross-runtime-quality-benchmarking.md` |
| ✅ | `QUANT-BLOCK-DIVISIBILITY` | The bundle is bigger than the recipe predicts and nothing said why. | A weight dim is not divisible by the block size. Per-block quantization and per-grouped-channel palettization **silently skip** layers whose dim does not divide, and ship them uncompressed. Check the size against the theoretical one. | silent | `knowledge/compression-reference.md` |
| ✅ | `TIED-EMBEDDINGS-SKIP-QUANT` | The head you thought you quantized is still fp16. | `tie_word_embeddings: true` — the eager quantizer silently skips tied weights. Clone the embedding table first. Then measure **on the phone**: the untied int8 head looked like a no-win on Mac three models in a row and was +17–40% on device, where it is a large share of the per-token read. | perf | `knowledge/pipelined-engine.md` |
| — | `HEAD-QSCHEME-CLIPPING` | 6/16 oracle top-1s flip, with one sweep position at cos 0.62 while its neighbours sit at 0.999x. | A big-vocab head quantized with `symmetric_with_clipping`: outlier head rows get clipped. Plain `symmetric` absmax gates 16/16 at identical speed. The transformer body is fine *with* clipping — this rule is specifically about fat-tailed embedding/head tables. | silent | `knowledge/pipelined-engine.md` |
| — | `PER-CHANNEL-INT8-GPU` | Bundle loads and runs; the quantized matmul returns garbage (`cos = nan`) while torch-level numerics gate 16/16. | Per-channel (axis-0) int8 Linear weights are broken on the macOS-27-beta MPSGraph GPU delegate, at multiple shapes, symmetric and clipping alike. Use per-block-32. | silent | `knowledge/compression-reference.md`; `knowledge/pipelined-engine.md` |
| — | `INT8-COSINE-IS-NOT-QUALITY` | Per-net cos 0.9998 and the generated images are visibly desaturated (RGB std 0.275 → 0.172). | For generative/perceptual models, quant error compounds through the sampler. Gate int8 by end-to-end *visual* output, not cosine. The inverse trap also exists: fp16 per-net **CPU** cos can look bad (0.9 range) because forcing fp16 compute overflows on CPU, while GPU e2e is identical to fp32. Gate int8 on CPU cos, gate fp16 on GPU. | silent | `knowledge/conversion-guide.md` |
| — | `T5-FP16-OVERFLOW` | The bundle converts; generative output is washed out. | T5-family text encoders overflow in fp16. Use bf16 (same exponent range, still half size) or fp32; DiT/VAE stay fp16-clean. | silent | `knowledge/conversion-guide.md` |
| — | `MOE-QUANT-SCHEME-BY-TOPK` | Expert-quantized MoE emits `<pad>` and skips reasoning blocks, diverging from fp16 at token 1. | The right int8 scheme depends on top-k: at k ≥ 4 expert error averages (~1/√k) and linear `sym8` survives and is faster; at **top-1** the error is not averaged and only k-means `km8` recovers fp16 quality. Non-QAT int4 is a wall at any k. | silent | `knowledge/compute-units-and-authoring.md` |

## 5. The PyTorch source, before you trace it

These are the `apple/coreai-torch` silent-corruption family plus the authoring traps that
produce a graph which lowers wrongly. All are detectable by reading the modelling code —
which for a `trust_remote_code` checkpoint means reading files you already downloaded.

| ✅ | ID | Symptom you experience | Detection | Severity | Source |
|---|---|---|---|---|---|
| ✅ | `SRC-CAST-ROUNDTRIP` | The bounded-floor idiom `(x + K).long().float() - K` compiles to the **identity function** — on every compute unit, CPU included. | `.long().float()` / `.int().float()` / a `torch.int64` → `torch.float` `.to()` pair. The converter cancels the cast pair and drops the truncation. | silent | [coreai-torch#9](https://github.com/apple/coreai-torch/issues/9) |
| ✅ | `SRC-FLOOR-ON-GPU` | Correct on CPU, wrong on GPU, same asset. | `torch.floor` / `trunc` / `ceil`. Check first whether the call is even on the traced path — drop-path and other training-only branches match the pattern and never reach the graph. | silent | [coreai-torch#10](https://github.com/apple/coreai-torch/issues/10) |
| ✅ | `SRC-FLOORDIV-ONE` | The natural in-graph workaround for the above is itself folded away. | `div(x, 1, rounding_mode="floor")` specifically. **A divisor other than 1 is fine** — `div(2x, 2, floor)` is the form that works, and flagging it would teach people to ignore the lint. | silent | [coreai-torch#10](https://github.com/apple/coreai-torch/issues/10) |
| ✅ | `SRC-INT64-BOOL-MASK` | A tensor two ops upstream — even a declared graph **output** — reads back garbage or NaN once an unrelated subgraph executes. Deterministic, CPU too. `clone()`/`contiguous()` barriers do not protect it; skipping `optimize()` does not help. | An int64-comparison → bool → float mask chain, e.g. `((ix0 >= 0) & (ix0 < W)).to(dtype)` — the standard in-bounds mask of a gather-based bilinear sampler. Diagnosis pattern: a tensor is *provably* computed right (another consumer sees exact values) but reads wrong later → buffer-liveness bug, hunt the comparison chain. Fix in float: `1 - (x - x.clamp(lo, hi)).abs().clamp(max=1)`. | silent | [coreai-torch#11](https://github.com/apple/coreai-torch/issues/11) |
| ✅ | `SRC-ARANGE-FLOAT` | The process aborts with `bad_optional_access was thrown in -fno-exceptions mode`. No Python traceback. | `torch.arange` with float start/end/step. DETR-family models reach it through `gen_sineembed_for_position(…, d_model / 2)` — the Python division makes the argument a float. It is the *argument* types that matter, not the dtype. | fatal | [coreai-torch#8](https://github.com/apple/coreai-torch/issues/8) |
| ✅ | `SRC-FP16-DECOMP-OVERFLOW` | Output collapses to 0 (or to inf/NaN) on the ANE past a threshold input. | `softplus`, `mish`, `logsumexp`, `logcumsumexp` are not in the preserved-op list, so PyTorch decomposes them into overflow-prone primitives: `log(1+exp(x))` overflows fp16 at x ≈ 10.4 on the ANE. Use the stable forms. | silent | [coreai-torch#21](https://github.com/apple/coreai-torch/issues/21) |
| ✅ | `SRC-OPTIMIZE-AXIS-MOVE` | `optimize()` changes the semantics of the expanded squared-distance expression; for equal-length inputs the output shape is still right and **no diagnostic is emitted**. | `sum(y**2, -1).unsqueeze(-2)` or `keepdim=True).transpose(-1,-2)` feeding a broadcast add. The axis move is dropped, so `‖y_i‖²` broadcasts where `‖y_j‖²` belongs. Compare against eager torch with `optimize()` **on**. | silent | [coreai-torch#49](https://github.com/apple/coreai-torch/issues/49) |
| ✅ | `SRC-SQUEEZE-DIM` | Converter aborts: `dimension to be shrunk must have size 1, got N`. | `x.squeeze(1)` is a no-op in torch when dim 1 ≠ 1; coreai-torch lowers it to a hard shrink. Guard it so the trace resolves it away. | fatal | `knowledge/conversion-guide.md` |
| ✅ | `SRC-COMPLEX-OPS` | Unsupported-op failure, usually from a complex-valued RoPE. | `torch.polar`, `view_as_complex`, `view_as_real`, complex multiply. Rewrite as real cos/sin. | fatal | `knowledge/conversion-guide.md` |
| ✅ | `SRC-REMAINDER` | Fails at `add_exported_program` validate time, not at runtime. | `aten.remainder` (tensor modulo) is unsupported. Compute it with a scalar symint plus `where()`. | fatal | `knowledge/conversion-guide.md`; `knowledge/stateful-kv-cache.md` |
| ✅ | `SRC-F-NORMALIZE` | Values blow up to ~1e13 — at large sequence lengths only. Small-L probes pass. | `F.normalize` loses its eps denominator clamp, so near-zero-norm vectors explode. It is input-dependent, which is why it hides. Write the norm explicitly. | silent | `knowledge/conversion-guide.md` |
| ✅ | `SRC-TORCH-ASSERT` | `GuardOnDataDependentSymNode` breaks non-strict export — added upstream *for* export compatibility. | `torch._assert` on a data-dependent comparison. For static-shape exports the check is vacuous: no-op it around the export call and restore. | fatal | `knowledge/conversion-guide.md` |
| ✅ | `SRC-WHILE-LOOP` | `'scf.while'` region type mismatch on the MPSGraph GPU delegate; on the macOS-27 beta the same bundle fails even `cpu_only`, so **"it verified on CPU earlier" proves nothing**. | `torch.ops.higher_order.while_loop` — any recurrent scan. Export at S=1 with a loop-free single-step recurrence (set the flag *before* quantization and tracing); at S=1 a scan is one step, so it is numerically identical. | fatal | `knowledge/pipelined-engine.md` |
| ✅ | `SRC-CHAINED-STATE-WRITES` | Position 0 is fine, everything after it decodes garbage. | More than one per-layer write to the same **fixed-shape** state handle. Each compiles to read_handle → slice_update → write_handle on one handle, and the GPU delegate drops all but one. Position 0 looks right because a fresh state *is* zero. Collect the slices and issue one fused full-state `slice_update` per step. The growing KV pair is exempt — its written values are re-read in-graph. | silent | `knowledge/pipelined-engine.md`; `knowledge/rwkv7-recurrent-linear-attention-coreai.md` |
| ✅ | `SRC-DATA-INDEXED-KV-WRITE` | Conversion succeeds. Mac GPU SIGTRAPs at the first execute; iPhone GPU SIGSEGVs after it loads and specializes; the iPhone ANE SIGABRTs and **corrupts the compile cache**, so the next load ENOENTs. | A KV write position derived in-graph from runtime data (the `in_step` index). Isolated to exactly that: swapping the begin-index source from a shape symint to a runtime tensor flips run → crash, everything else identical. Escape: hand the graph a host-built one-hot write mask and blend, `sl*(1-m) + col*m`. | fatal | `knowledge/coreai-beta-mpsgraph-kvwrite-bug.md`; [coreai-models#5](https://github.com/apple/coreai-models/issues/5); FB23024751; [coreai-torch#6](https://github.com/apple/coreai-torch/issues/6) |
| ✅ | `SRC-MISSING-DEFUNCTIONALIZE` | The state never updates and nothing complains. | `slice_update` present, `remove_functionalization(ep)` absent. In-place state writes need it after `run_decompositions` and before converting, or the mutation is dropped. | silent | `knowledge/conversion-guide.md` |
| — | `SRC-CONSTFOLD-TRIG` | A positional embedding is quietly wrong (cos collapses to ~0.5). | `sin`/`cos` of a *constant* large argument (a fixed Sobol/grid → arg ~1e5) gets constant-folded at low precision. Precompute in torch and bake as a non-persistent buffer. Runtime-driven trig is fine. | silent | `knowledge/conversion-guide.md` |
| — | `SRC-EXTERNALIZE-CLASS-MISMATCH` | `custom op not found` during externalization. | `ExternalizeSpec` marks ops by *class*. If the export unit holds submodules of that class that are not in the traced graph (a front-end norm kept as an attribute), externalizing fails. Opt out with `coreai_externalize_specs = ()`. | fatal | `knowledge/conversion-guide.md` |
| — | `SRC-ROPE-INVFREQ-BUFFER` | fp16 conversion underflows the small RoPE frequencies to zero. | Keep `inv_freq` a plain fp32 attribute, not a buffer, so `.half()` cannot reach it; cast cos/sin to the query dtype. | silent | `knowledge/conversion-guide.md` |
| — | `SRC-GC-DROPPED-MODEL` | The function returns **garbage** — no crash, just wrong output, which reads exactly like a conversion bug. | A persistent runner that stores only the `load_function` lets the `AIModel` get GC'd. Hold the model reference. | silent | `knowledge/conversion-guide.md` |
| — | `SRC-LOAD-OPTIONS-NONE` | `RuntimeError: MPSGraph Unresolved symbol (prepare/initialize)`. | `AIModel.load(path, None)` on the GPU path. Pass an explicit `SpecializationOptions.default()` or `.cpu_only()`, never `None`. Note `coreai_kit.run` defaults to `cpu_only`, so anything copied from it silently benchmarks CPU. | fatal | `knowledge/conversion-guide.md` |
| — | `SRC-OPTIMIZE-HANG` | `prog.optimize()` runs for 90+ minutes on ~64 GB while the conversion itself took 7 s. | Big attention graphs. Skip it (`optimize=False`) and gate manually — on-device AOT optimizes anyway. Note `verify()` forces `optimize=True`. | perf | `knowledge/conversion-guide.md` |
| — | `SRC-SDPA-EXTERNALIZE-BOUND` | `Constraints violated (d_20)` — "This is a coreai-torch bug. Please report it." | A static query length with a dynamic KV context: the externalize pass re-exports the SDPA submodule and reconstructs the key seq dim with an **unbounded** upper bound. Shipped models miss it because they keep the query dynamic too. | fatal | [coreai-torch#1](https://github.com/apple/coreai-torch/issues/1) |
| — | `SRC-HYBRID-DELTANET-DYNKV` | `EXC_BAD_ACCESS` in `mlir::FloatType::getWidth()` during MPSGraph JIT — exports and loads fine, segfaults on first execute. | **2 or more** `GatedDeltaUpdate` layers together with one attention layer reading a dynamic-length KV context. One DeltaNet layer runs; three stacked with no attention run; the combination crashes. | fatal | [coreai-torch#2](https://github.com/apple/coreai-torch/issues/2) |

## 6. The environment

| ✅ | ID | Symptom you experience | Detection | Severity | Source |
|---|---|---|---|---|---|
| ✅ | `ENV-CONVERTER-SHADOWED` | A correctly-pinned 0.4.1 environment silently exports 0.4.0 IR. You find out weeks later, on a device. | A `coreai_torch` source checkout as the working directory: its `coreai_torch.egg-info` takes `sys.path[0]` priority over the installed wheel. | silent | `knowledge/coreai-torch-041-ir-incident.md` |
| ✅ | `ENV-OVERLAY-MISSING` | Every recipe that re-authors a model dies on import. | The interpreter has `coreai_models` without `conversion/overlay/` applied. Same probe `zoo_convert.py doctor` runs; `coreai_doctor.py --env` carries it so the artifact lint is a superset of the environment lint rather than a second, competing `doctor`. | fatal | `conversion/overlay/README.md` |
| — | `ENV-TORCH-VERSION-DRIFT` | Every export dies at load with a torchvision circular import. | Letting a resolver bump `torch` past the pinned 2.9.0. | fatal | `knowledge/coreai-torch-041-ir-incident.md` |
| — | `ENV-GPU-CONCURRENCY` | Kernel panic. | The beta driver panics under parallel GPU load. Mac-GPU work is exclusive — take the lock, run solo. | fatal | `knowledge/ship-playbook.md` |

---

## What this is not: `conversion/zoo_verify.py`

The zoo already ships a bundle checker, and the two ask different questions. `zoo_verify`
compares a bundle against **the source repository it came from** — eos/bos, chat template,
context, dtype — and reports an undeclared deviation as `DIFF`. `coreai doctor` compares a
bundle against **what the runtime will do with it**, using only the bundle's own files.

Fidelity and fitness are not the same property, and the Gemma-3 case is the clean
demonstration: those bundles declared `eos_token: "<eos>"` and so did
`google/gemma-3-4b-it`, so a source comparison says `ok` — the port copied its source
perfectly. What it copied was a value the chat template never emits, which is a runaway in
any app that stops on `eosTokenId`. Doctor sees it because it reads the template instead of
the source. (In practice `zoo_verify` could not even get that far: Google's repos are gated,
so it correctly reports `skipped`.)

The inverse holds too — `zoo_verify` catches drift doctor cannot see, because doctor never
looks upstream. Both, not either.

## What a lint cannot catch

Doctor is static. Everything above is a pattern in a file. The defects that remain after a
clean run are the ones that only a measurement can see, and the notes are emphatic about
three of them:

**An equivalence gate cannot detect a defect its reference shares.** The `wNa8o8`
checkpoint scored 3/3 "EXACT vs oracle" while losing half its reasoning accuracy, because
the oracle was an fp16 implementation of the same weights and carried the identical defect.
Doctor can flag the *checkpoint* (it does), but nothing static can tell you that a passing
gate is passing for the wrong reason. Gate against something that does not share your
pipeline's assumptions.

**Prompts that cannot see the failure.** The same ship gate ran three prompts — "Why is the
sky blue?", "What is the capital of France?", "Explain photosynthesis in one sentence" — all
single-hop recall. The defect only appears in multi-step reasoning. Coverage of the input
space is not something a file can be inspected for.

**Budgets that cap below the failure.** The iOS KV miscompile reached a shipping chat app
because every device gate capped generation at 256 tokens and the bug needs 1024. A quality
comparison in the same period produced a 20%-vs-80% result that was entirely a 512-token
budget cutting off a reasoning arm mid-thought. Neither is visible in any artifact.

So the honest closing line of a clean doctor run is the one the tool prints: gate the bundle
against an HF oracle, teacher-forced top-1 over a prompt whose fp32 top-2 margin clears 0.1
at every position, 16/16 — and gate the **engine** path, not only the Python runtime.

---

## Coverage

64 rows: **45 implemented** (✅), 19 documented here only (—). Run
`python3 coreai_doctor.py --rules` for the machine-readable list, and `python3 selftest.py`
for the source-rule fixtures — which test both that each rule fires on its trigger *and*
that it stays quiet on the documented workaround.

The 19 unimplemented rows are not a backlog of equal items. Roughly half need something
doctor deliberately does not do — run the model, read weights, or know the app's build
settings — and belong to a `coreai verify` rather than a `coreai doctor`. The rest are
straightforward extensions of the scan surfaces already here.
