# Apple's `apple/coreai-models`: repo reference

A close read of Apple's own [`apple/coreai-models`](https://github.com/apple/coreai-models)
(BSD-3-Clause, ~1.4k stars) — the official Core AI model catalog, Python authoring primitives,
Swift runtime package, and agent-skill plugins. This repo (`coreai-model-zoo`) already depends
on it directly (SwiftPM, in three apps) and indirectly (a Python source overlay patched onto a
pinned checkout for conversion), and its `skills/` content is the primary distilled source behind
[`compute-units-and-authoring.md`](compute-units-and-authoring.md). This doc is the fuller
reference: verified directory tree (via the GitHub API tree, not the JS-rendered file browser),
the complete 22-model catalog cross-referenced against this zoo's own ports, the authoring-rule
content extracted in full from `skills/skills/model-authoring/references/*.md`, and every place
in this repo that consumes `apple/coreai-models`.

All facts below were verified against `repos/apple/coreai-models` at the `main` branch tip
fetched via `gh api repos/apple/coreai-models/git/trees/main?recursive=1` and per-file
`gh api .../contents/<path>` during this session (2026-07-23) — not scraped from the rendered
GitHub UI, which truncates deep trees. Cross-repo file:line citations point at this repo's
worktree (`coreai-model-zoo`) at the same commit range as `git status` at session start.

## Why this repo depends on it

Three consumption modes exist in this repo, verified by `grep -rn "apple/coreai-models"
--include="*.yml" --include="*.py"`:

1. **Direct SwiftPM dependency, unmodified.**
   - `apps/CoreAISegment/project.yml:13-15` — pins `url: https://github.com/apple/coreai-models`
     at `revision: d5804c8f6475e9e41aef08d8d2516ccf39fcbcc3`, consuming the `CoreAISegmentation`
     product (SAM3 support) with **no patch stack** — the comment at `project.yml:10-12` states
     the runtime "runs unmodified on the stock engine."
   - `apps/CoreAITranscribe/project.yml:13-15` — same `apple/coreai-models` URL, pinned at
     `revision: d5804c8f6475e9e41aef08d8d2516ccf39fcbcc3` (same commit as CoreAISegment).
2. **Forked SwiftPM dependency** (not upstream `apple/`, but a public fork tracking it):
   - `apps/CoreAIImageGen/project.yml:13-15` pulls `url: https://github.com/john-rocky/coreai-models`
     at `revision: bc48d3d07e03ce90a4a05d3264dca4c4c40ac81f` — the comment at `project.yml:10`
     says it's "Fork of apple/coreai-models @ 02a8edd + the in-context edit path
     (Flux2Pipeline.editImages" for FLUX.2 in-context editing (not upstream yet).
   - `apps/CoreAIUpscale/project.yml:11-13` explicitly notes it does *not* pin `coreai-models` at
     all, to avoid a package-name collision with `CoreAIImageGen`'s pin — SR has no runtime
     dependency on the package.
   - `apps/CoreAIChatMac/project.yml:9,12-13` pins `path: ../../coreai-models` — a **local
     patched fork** (not upstream) carrying the catalog's decode-pipelined-engine changes.
3. **Python source overlay** (`conversion/overlay/`) — the zoo's conversion scripts need
   re-authored model definitions (qwen3.5, qwen3.6-MoE, gemma4, GLM-4.7, LFM2.5, LLaDA, BitCPM,
   RWKV-7, MiniCPM, VL towers, …) that Apple's `coreai-models` does not ship, since it "takes no
   PRs and does not register newer models" (`conversion/overlay/README.md:3-4`). Rather than
   maintaining a parallel package, this repo layers a source overlay onto a **pinned upstream
   checkout**:
   - `conversion/overlay/BASE` pins `repo: https://github.com/apple/coreai-models.git` at
     `commit: b1cb71b8522d99408059fa0b98b8742171bcb0b8`.
   - `conversion/overlay/patches/python-overlay.patch` edits *tracked* upstream files (export
     pipeline, registries, `primitives/{macos,ios}/{cache,rope}.py`, small model fixes).
   - `conversion/overlay/files/python/src/coreai_models/...` adds *new* files (mostly
     `models/{macos,ios}/*.py`), verified present at
     `conversion/overlay/files/python/src/coreai_models/models/macos/gemma4_dense_metal_sdpa.py`
     and `.../models/ios/gemma4_blend.py` (each references an `apple/coreai-models` upstream
     issue number it works around — `#27` and `#5` respectively).
   - `conversion/overlay/apply.py` clones-and-checks (`read_base()`/`BASE`), applies the patch,
     copies `files/` in, then the workflow is `pip install -e python/` on the patched checkout —
     i.e. this repo runs a **superset of Apple's actual `coreai_models` Python package**, not a
     reimplementation. `conversion/overlay/README.md:36-43` notes the overlay's scope is
     Python-only; the Swift engine changes needed by some apps live as commits on the
     `john-rocky/coreai-models` fork instead (see mode 2 above), tracked separately.

Separately, this repo's export scripts pin **`coreai-torch`** (Apple's PyPI *converter* package —
a different repo, `apple/coreai-torch`, not to be confused with `apple/coreai-models`) as a PEP
723 inline dependency, e.g. `conversion/export_adcsr.py:1-9` declares
`"coreai-core==1.0.0b1"` / `"coreai-torch==0.4.0"`. So the two Apple packages play different
roles here: `coreai-torch` (PyPI) is consumed unmodified as the PyTorch→IR converter; the
`coreai-models` *Python* package is consumed as a patched/extended checkout via the overlay; the
`coreai-models` *Swift* package is consumed either stock (CoreAISegment/CoreAITranscribe) or
forked (CoreAIImageGen/CoreAIChatMac).

## Directory structure (verified tree)

Top level (from `Package.swift`, `README.md`, and the recursive tree):

```
.claude-plugin/         marketplace.json — registers the skills/ plugin for Claude Code
.github/                issue templates only (bug_report, model_request, workflow_feedback) — no PR template
models/                 model catalog: one dir per model/family, README.md + export.py (or shared CLI)
python/                 the coreai_models Python package (src/coreai_models/) + tests/
skills/                 agent-skill plugin (3 skills, Claude Code / Codex CLI / Gemini CLI installable)
swift/                  the coreai-models SwiftPM package: Sources/ (5 libraries + 5 CLI tools) + Tests/
LICENSE                 BSD 3-Clause
Package.swift            SwiftPM manifest (products below)
pyproject.toml, uv.lock  Python project (uv-managed)
```

`models/README.md` states: *"Only models listed in the catalog below or registered in the
[model registry](../python/src/coreai_models/model_registry.py) are supported."* — i.e. the
catalog is closed-world, not a loose collection of examples.

`python/src/coreai_models/` subpackages (verified via the recursive tree): `diffusion/`,
`export/` (`bundle.py`, `compiler.py`, `compression.py`, `ios.py`, `macos.py`, `metadata.py`,
`mlir_ops.py`, `pipeline.py`, `presets.py`), `llm/` (`eval.py`, `export.py`), `models/`
(`base.py`, `ios/` incl. a full `sam3/` subpackage, `macos/`, `registry.py`), `primitives/`
(`ios/`: `bidirectional_sdpa.py`, `cache.py`, `embedding.py`, `gelu.py`, `layer_norm.py`,
`mlp.py`, `quantization.py`, `rms_norm.py`, `rope.py`, `sdpa.py`; `macos/`: `cache.py`,
`cache_scatter.py`, `mlp.py`, `rms_norm.py`, `rope.py`, `sdpa.py`, `switch.py`), `segmentation/`,
`vlm/`, plus `model_registry.py` (top-level `LLM_PRESETS`/`DIFFUSION_PRESETS`) and `_hf.py`.

`swift/Sources/` targets (from `Package.swift`, verified): `CoreAILanguageModels`,
`CoreAIImageSegmenter`, `CoreAIObjectDetector`, `CoreAIShared`, `CoreAISpeech`,
`CoreAIDiffusionPipeline`, plus a `CXGrammar` C++ bridge target (links `xgrammar` for guided
generation) and five CLI executable targets under `Sources/Tools/`: `llm-runner`,
`image-segmenter`, `object-detector`, `diffusion-runner`, `speech-runner`, plus `llm-benchmark`
("based on mlx-lm benchmark" per the target comment).

## Model catalog (22 entries, `models/`)

Verified against `models/README.md`'s catalog section and each model dir's `README.md` first
line. Grouped exactly as Apple groups them.

### Language Models (LLMs) — 7 families, via `coreai.llm.export`
| Model | One-liner | This zoo's overlap |
|---|---|---|
| `gemma3` | Google's Gemma 3 (text) | Superseded in this zoo by Gemma 4 ports ([`gemma4-mixedbit-qat-transplant.md`](gemma4-mixedbit-qat-transplant.md), [`gemma4-ple-static-input-fm-stack.md`](gemma4-ple-static-input-fm-stack.md)) — one generation ahead of Apple's catalog |
| `gpt_oss` | OpenAI's GPT-OSS (MXFP4 MoE) | Measured by this zoo, unmodified recipe — see [`apple-models-bench.md`](apple-models-bench.md) (78.1 tok/s decode, 33.9 GB RSS) |
| `mistral` | Mistral AI's Mistral | Measured unmodified — `apple-models-bench.md` (101.7 tok/s, 4-bit) |
| `mixtral` | Mistral AI's Mixtral (MoE) | Not separately ported in this zoo |
| `qwen2` | Alibaba's Qwen2.5 | Superseded by this zoo's newer Qwen3/3.5/3.6 ports |
| `qwen3` | Alibaba's Qwen3 (dense) | Measured unmodified (0.6b/4b/8b) — `apple-models-bench.md`; this zoo ships far newer Qwen3.5/3.6 hybrid variants with the pipelined engine ([`pipelined-engine.md`](pipelined-engine.md)) |
| `qwen3_moe` | Alibaba's Qwen3 MoE | This zoo separately ships dense-int4km flagship MoE work ([`dense-int4km-flagship-session-findings.md`](dense-int4km-flagship-session-findings.md)) |

### Diffusion Models — via `coreai.diffusion.export`
| Model | One-liner | This zoo's overlap |
|---|---|---|
| `stable-diffusion` | SD 1.5 / 2.1 / 3.5 Medium | This zoo's AdcSR ([`adcsr-super-resolution.md`](adcsr-super-resolution.md)) is a *pruned SD-2.1* derivative for super-resolution, not text-to-image — different task, shared lineage |
| `flux2` | Black Forest Labs' FLUX.2 | This zoo ships FLUX.2 [klein] **in-context editing** ([`flux2-in-context-editing.md`](flux2-in-context-editing.md)) via the `john-rocky/coreai-models` fork's `flux2-in-context-edit` branch — functionality beyond Apple's stock text-to-image recipe |

### Vision-Language Models (VLMs) — via `coreai.vlm.export`
| Model | One-liner | This zoo's overlap |
|---|---|---|
| `vlm` (Qwen3-VL) | Text decoder + token embedding + vision encoder as one `.llmasset/` bundle | This zoo's OCR trio (GLM-OCR, MinerU, unlimited-OCR) targets whole-page/document VLM tasks Apple's recipe doesn't cover |

### Vision Models
| Model | One-liner | This zoo's overlap |
|---|---|---|
| `clip` | CLIP — joint image/text embeddings, zero-shot classification | Measured unmodified (fp16 on ANE: 3.68 ms, 1.7× GPU) — `apple-models-bench.md` |
| `depth-anything` | Depth Anything v3 — monocular depth + confidence + camera intrinsics/extrinsics | This zoo separately ports DA3 ([`depth-anything-3-monocular-depth.md`](depth-anything-3-monocular-depth.md)) with its own double-normalization-trap findings — same model family, independently re-authored |
| `edsr` | EDSR — fixed-integer-factor (2/3/4×) super-resolution | **Direct competitor** to this zoo's own super-resolution ports: AdcSR (one-step diffusion-GAN ×4, [`adcsr-super-resolution.md`](adcsr-super-resolution.md)) targets *quality* via a generative prior; EDSR is a plain feed-forward CNN — smaller/faster but lower fidelity than a diffusion-GAN approach. Worth benchmarking EDSR as a cheap fallback tier. |
| `efficient-sam` | EfficientSAM — ViT-Tiny MAE-pretrained encoder, promptable segmentation | This zoo ships the newer/larger SAM 3 ([`sam3-promptable-segmentation.md`](sam3-promptable-segmentation.md)) as its official segmenter (also **the exact model** `apps/CoreAISegment` pulls straight from `apple/coreai-models`'s own `sam3` recipe — see below) |
| `pvt` | PVT v2 — pyramid ViT backbone for dense prediction | Not used elsewhere in this zoo |
| `sam3` | SAM 3 — Meta's promptable image/video segmentation (text or visual prompts) | **This is the model `apps/CoreAISegment` ships.** Apple's own `models/sam3/README.md` notes the export targets iOS via BC1S restructuring, `Conv2d(1x1)` projections, fp16-safe primitives, rank-4 window attention, split into 3 independently-optimizable functions — i.e. Apple did the ANE-authoring work `neural_engine_rules.md` describes, for this exact model. This repo's own SAM3 notes are in [`sam3-promptable-segmentation.md`](sam3-promptable-segmentation.md) (float16≈float32 fidelity, gated-model licensing) |
| `yolo` (YOLOS) | Plain-ViT object detection (patches → object queries → boxes/logits) | Answers to `apple/coreai-models#14`; this zoo separately ports RF-DETR (`conversion/export_rf_detr.py:1`) as a competing detector, explicitly noting it "answers apple/coreai-models #14" |

### Audio Models
| Model | One-liner | This zoo's overlap |
|---|---|---|
| `clap` | CLAP — joint audio/text embeddings, zero-shot audio classification | Not ported in this zoo |
| `wav2vec2` | Self-supervised speech representations, fine-tuned for character-level ASR | This zoo's ASR entry is Whisper large-v3-turbo instead ([`whisper-asr-fixed-decode.md`](whisper-asr-fixed-decode.md)) |
| `whisper` | OpenAI Whisper ASR encoder-decoder | **Apple ships Whisper too** — this zoo's own Whisper port ([`whisper-asr-fixed-decode.md`](whisper-asr-fixed-decode.md)) independently solves the dynamic-decoder-recompile problem (fixed 128-token decoder window) that Apple's stock single-step `[1,1]` decoder graph "can't transcribe" per that doc's own framing — worth comparing recipes directly since both target the same upstream model |

### Text Models
| Model | One-liner | This zoo's overlap |
|---|---|---|
| `roberta` | BERT-improved transformer encoder (bigger batches/data, no NSP) | Not ported in this zoo |
| `t5` | Encoder-decoder, prefix-driven multitask (translation/summarization); FLAN-T5 supported | Not ported in this zoo |

Registry-driven LLM export also supports arbitrary compression presets per platform
(`models/README.md`): macOS defaults to `4bit` (INT4 weight-only, block 32); iOS defaults to
`4bit_weight_palettized_group32`, with `4bit_weight_palettized_group8` and `none` as
alternatives, and Embedding forced to 8-bit-per-tensor on all iOS presets. Custom mixed-precision
YAML recipes ship alongside individual model cards, e.g.
`models/qwen3/qwen3_0_6b_mixed_4bit_8bit.yaml`.

## Python authoring primitives (`python/src/coreai_models`)

Two *different* Apple packages are in play and this repo consumes them differently — see
"Why this repo depends on it" above for the split. This section covers the `coreai_models`
package specifically (primitives + export helpers), which this repo extends via the overlay
rather than reimplementing.

- **`primitives/ios/*`** — Neural-Engine-safe building blocks matching the BC1S contract:
  `bidirectional_sdpa.py`, `cache.py` (readonly KV I/O), `embedding.py` (`(V,1,D)` shape),
  `gelu.py`, `layer_norm.py`, `mlp.py`, `quantization.py`, `rms_norm.py`, `rope.py`
  (4D `(1,head_dim,1,S)` precomputed cos/sin), `sdpa.py` (per-head einsum).
- **`primitives/macos/*`** — GPU/CPU standard-layout equivalents: `cache.py` +
  `cache_scatter.py` (stateful `mutable_slice_update`-based KV), `mlp.py`, `rms_norm.py`,
  `rope.py`, `sdpa.py` (native fused SDPA), `switch.py` (MoE `SwitchLinear`/`SwitchGLU` via the
  `GatherMM` composite op).
- **`export/*`** — the actual pipeline: `ios.py` / `macos.py` (platform export paths),
  `compression.py` (preset resolution), `compiler.py`, `metadata.json` construction
  (`metadata.py`), `mlir_ops.py`, `bundle.py`, `presets.py`.
- **`models/{ios,macos}/*`** — per-architecture re-authored decoders (qwen2, qwen3, qwen3_moe,
  gemma3_text, gpt_oss, mistral, mixtral, qwen3_vl) plus a full `models/ios/sam3/` subpackage
  (`detr.py`, `fpn.py`, `image_encoder.py`, `mask_decoder.py`, `text_encoder.py`, a
  `primitives/{rope,window}.py` pair, and `sam3_reauthored.py`).
- **`model_registry.py`** — the `ModelPreset` table (`LLM_PRESETS`/`DIFFUSION_PRESETS`) that
  backs `coreai.llm.export <short-name>`; adding a model that fits the standard pipeline means
  adding an entry here, per `models/README.md`'s "Adding a Model" section.

This repo's `conversion/*.py` scripts do **not** reimplement these primitives from scratch for
models Apple already registers (qwen3, mistral, etc. — see `apple-models-bench.md`, which
benchmarks those recipes *unmodified*). For everything Apple hasn't registered, the
`conversion/overlay/` mechanism above extends this exact package via patch + new files rather
than forking it wholesale — e.g. `conversion/overlay/files/python/src/coreai_models/models/macos/gemma4_dense_metal_sdpa.py:4`
is explicitly framed as "THE BYPASS for the engine blocker" tracked against
`apple/coreai-models#27`.

`conversion-guide.md` in this repo documents the canonical `TorchConverter` API (from the
separate `coreai-torch` PyPI package, not `coreai_models`) that both Apple's own export scripts
and this repo's `conversion/*.py` scripts call into — see that file for the gotchas (async/sync
mixing, `save_asset` non-overwrite, in-place-state functionalization, etc.).

## The `skills/` directory (agent skills — the high-value content)

`skills/` is a **Claude Code / Codex CLI / Gemini CLI plugin** (`skills/.claude-plugin/plugin.json`,
`skills/.codex-plugin/plugin.json`, `skills/gemini-extension.json`) bundling three skills under
`skills/skills/`. Installable in Claude Code via `/plugin marketplace add
git@github.com:apple/coreai-models.git` then `/plugin install coreai-skills@coreai-models`
(per this repo's own README: `README.md:150-159` region references the same skill paths).

### `model-authoring` (`skills/skills/model-authoring/SKILL.md`)

Empirical, hardware-behavior-driven rules for authoring PyTorch models that compile/run
correctly on Neural Engine, GPU, or CPU. Three reference files, **verified in full** this
session (not just cited):

**`references/neural_engine_rules.md`** (479 lines) — the ANE contract in complete detail:
- **Hard limits**: max tensor rank 5; dtypes fp16/int8/int16 only (fp32 falls back off-ANE);
  fully static shapes (one exported function per shape config).
- **Memory alignment**: the *last* tensor axis is ANE "width" and must be 64-byte aligned. A
  singleton last axis pads to 64 bytes = **32× memory cost at fp16, 64× at int8** — never put a
  size-1 dim last.
- **BC1S**: `(B, S, D) → (B, D, 1, S)` via `permute(0,2,1).unsqueeze(2)`; multi-head
  `(B,H,S,D) → (B, H*D, 1, S)`.
- **Conv2d not Linear** for all projections (`nn.Linear` falls back off-chip); state-dict
  conversion is `linear.weight.unsqueeze(-1).unsqueeze(-1)`.
- **No fp32 literals anywhere** — even `x * 1.0` creates an f32 buffer; use
  `torch.ones(1, dtype=x.dtype)`.
- **Per-head attention only** — no fused SDPA; use the `einsum("bchq,bkhc->bkhq", ...)` pattern
  for zero-copy Q@K.
- **Causal mask**: shape `(1, key, 1, query)` — **transposed vs GPU** — and masked value
  `-40000.0`, never `float('-inf')` (ANE softmax mishandles IEEE -inf).
- **RoPE**: precompute cos/sin outside the graph, pass as 4D `(1, head_dim, 1, S)`; in-graph 2D
  table indexing produces 3D `gather_nd` output ANE rejects.
- **KV cache — readonly functional I/O pattern**: cache shape
  `[n_layers, B, H_kv*D, 1, max_S]`, sequence on **dim 4**; the model contains *no cache writes*
  — it receives full past K/V, concatenates, attends, and returns *new* K/V as outputs; Python
  writes them into the cache slots externally. **Critical trap**: must return post-RoPE
  `key_rope`, not raw `new_k` — caching pre-RoPE keys collapses PSNR to ~20 dB.
- **Chunked prefill**: `CHUNK=64`, offset = `chunk_start` not `chunk_end`; sequential
  per-token (`S_q=1`) prefill accumulates fp16 rounding error past ~50 tokens.
- **Layer-design levers requiring retraining**: conv stride must factor into 2s/3s only (equal
  strides 4/6/8/9/12/16/24/32; mixed strides pair 2 with 3/4/8/9); large-kernel decomposition
  `k_fused = k1+k2-1`; convolution fusion (only when no activation between them); dilated-conv
  factorization via the same 2/3-prime-factor rule; pooling stride 2 or 4 only.
- **Neural Engine functions**: one dynamic `torch.export` compiles into multiple static
  entrypoints per `(context_len, extend_num_tokens)` — `extend_{ctx}_{len}` (generation, returns
  logits+KV), `prompt_opt_{ctx}_{len}` (fast prefill, KV only), `gather_embeddings_{N}`.

**`references/gpu_rules.md`** (297 lines) — the GPU/CPU contract:
- Standard layouts throughout, `nn.Linear`, native fused `F.scaled_dot_product_attention`.
- **Fused QKV** into one `nn.Linear`; **fused Q/K-norm+RoPE** on the combined slice before
  splitting Q/K/V (fewer kernel launches).
- **MLP op order matters**: compute `up_proj` *before* `gate_proj` — "reversed from many
  reference implementations but yields better GPU utilization."
- **KV cache — stateful pattern**: shape `[n_layers, B, H_kv, max_S, D]`, sequence on **dim 3**,
  via the custom op `coreai::mutable_slice_update` (eager mutates in place; meta/fake returns a
  new same-shape tensor, satisfying `torch.export`'s functional semantics) — compile with
  `mutable_arg_action="hoistToArg"` in `LegalizeToCoreOptions`. Explicit warning: stateful
  transform APIs **reset state between inference calls** — don't use them for token generation.
- **MoE**: `SwitchLinear` (weight shape `(num_weight_sets, num_experts, out, in)`, batched
  gather+matmul via `coreai_torch.composite_ops.GatherMM`) + `SwitchGLU` (three `SwitchLinear`s
  + SwiGLU); expert indices cast to `uint16`.
- **Memory-efficient large-model loading**: meta-device init
  (`MyModel(config, device="meta")`) + `load_state_dict(..., assign=True)` + optional
  layer-at-a-time safetensors streaming for 7B+ models.

**`references/common_issues.md`** (176 lines) — a debugging lookup table; highlights not
already covered above:
- ANE SDPA PSNR 15–30 dB → almost always the causal-mask-orientation bug (transpose it).
- Input dtype error "does not match" → use `"si32"` not `"i32"` in the descriptor.
- Import error about input counts → filter `input_specs` to `USER_INPUT`/`BUFFER` kinds only
  (PARAMETER/CONSTANT_TENSOR get folded away by `run_decompositions()`).
- ANE MLP "3 invalid ops from `mps.swish`" → `F.silu(x)` lowers to a 3-op f32 round-trip; replace
  with `gate_pre * torch.sigmoid(gate_pre)`.
- M-RoPE PSNR ~18 dB → GPU pattern not reproduced exactly; match `cat([cos,cos],-1)` then `::2`.
- `post_init()` missing `rope_parameters` → patch `ROPE_INIT_FUNCTIONS["default"]` before
  instantiation (snippet included).
- Wrong logits on ANE → non-contiguous tensors; call `.contiguous()` on everything before
  `NDArray`.
- Compiles but runs on CPU → recompile with `--preferred-compute neural-engine`.
- `embed_tokens()` numpy conversion → needs `.detach()` first (`requires_grad=True` by default).
- `runner(**inputs)` not `runner(inputs_dict)` — `InferenceFunction.__call__` takes kwargs.
- Output dict key order is non-deterministic — identify K vs V by shape/MSE, never by index.

**Verification-gate table** (from `SKILL.md`, not the references): re-authored-vs-source >70 dB;
ANE-layout-vs-GPU-layout >70 dB; compiled-vs-torch ≥40 dB; post-4-bit-palettization ≥35 dB.

### `working-with-coreai` (`skills/skills/working-with-coreai/SKILL.md`, 199 lines)

The end-to-end deployment workflow skill: AUTHOR → COMPRESS → EXPORT (`coreai-torch`
`TorchConverter`) → COMPILE (`xcrun coreai-build compile`) → RUN (Swift `CoreAI` framework or
Python `coreai.runtime`). Notable content beyond what's in this repo's `conversion-guide.md`:
- **Onboarding protocol** — explicitly instructs the agent's *first response* to be a
  clarifying conversation (workload type, platform, model shape, priority: speed vs. energy vs.
  quality vs. reach), not code — and to check `coreai-models/models/` for an existing recipe
  before authoring from scratch.
- **PSNR verification table** (Python-runtime-level, complementing the authoring-skill's table):
  fp32 end-to-end >70 dB (investigate <60); fp16 on-device >50 dB (investigate <40); 4-bit
  palettized ~40 dB (investigate <30).
- **`references/guidance.md`** (69 lines) — platform sizing/use-case guidance: iOS models should
  stay under 2 GB and are foreground-workload-oriented (background execution is subject to iOS
  resource-management policy); macOS should leave ≥6 GB RAM headroom and handles both foreground
  and background/batch. iOS optimization = static shapes + int4/int8 linear-quant or 2/4/6/8-bit
  palettization; macOS optimization = dynamic shapes OK, int4 per-block quant recommended. Use
  `.default` specialization at runtime unless a specific compute unit is forced, in which case
  the model representation should be pre-aligned to it (ANE → static+palettized;
  GPU → linear-quant+non-chunked).

### `model-compression-exploration` (`skills/skills/model-compression-exploration/SKILL.md`, 191 lines)

A `coreai-opt` (`coreai_opt`, the **third** Apple package — the compression/palettization
library, distinct from `coreai-torch` and `coreai_models`) sweep-automation skill: given a model
+ reference data + quality metric from the user, it runs three experiment groups —
1a. channel-structured quant (6 configs: `{int8,int4} × {symmetric, asymmetric,
symmetric_with_clipping}`), 1b. block-structured quant (9 configs: `{16,32,128 block size} ×
{same 3 schemes}`, int4), 2. palettization (15 configs spanning 8/6/4-bit × per-tensor/grouped ×
per-channel-scale on/off) — then layer-skip refinement on the 95th/75th-percentile survivors, and
reports 5 Pareto-anchor configs per group with a PSNR-vs-compression-ratio scatter plot. Designed
to be parallelized as one subagent per group. Not directly cited elsewhere in this repo's
knowledge base as of this writing — worth pointing a future palettization-exploration task at it
directly rather than hand-rolling a sweep.

## Consumption model summary

| Layer | What this repo pulls | How | Pin example |
|---|---|---|---|
| Swift runtime (stock) | `CoreAISegmentation` (SAM3), `CoreAILM` (per README's product table) | SwiftPM `url:` + `revision:` | `apps/CoreAISegment/project.yml:14-15` → `d5804c8f6475e9e41aef08d8d2516ccf39fcbcc3` |
| Swift runtime (forked) | patched engine (pipelined-engine fixes, in-context edit) | SwiftPM `url:` to `john-rocky/coreai-models` fork, or local `path:` | `apps/CoreAIImageGen/project.yml:14-15` → fork `bc48d3d07e...`; `apps/CoreAIChatMac/project.yml:12-13` → `path: ../../coreai-models` |
| Python `coreai_models` pkg | primitives + export pipeline + extra model defs | overlay: pinned-checkout + patch + new files, `pip install -e python/` | `conversion/overlay/BASE` → `b1cb71b8522d99408059fa0b98b8742171bcb0b8` |
| Python `coreai-torch` pkg (separate repo, PyPI) | `TorchConverter`, composite ops | PEP 723 inline `dependencies = [...]`, unmodified | `conversion/export_adcsr.py:1-9` → `coreai-torch==0.4.0` |
| Agent skills | authoring rules, workflow guidance | manual extraction into this knowledge base (this doc + `compute-units-and-authoring.md`) — not installed as a live plugin in this repo | — |

The five SwiftPM library products, verified from `Package.swift` (not previously enumerated in
this repo's knowledge base): `CoreAILM` (→ `CoreAILanguageModels` target), `CoreAIDiffusion`
(→ `CoreAIDiffusionPipeline`), `CoreAISegmentation` (→ `CoreAIImageSegmenter`), `CoreAISpeech`,
`CoreAIObjectDetection` (→ `CoreAIObjectDetector`). Plus five CLI executables under
`Sources/Tools/`: `llm-runner`, `image-segmenter`, `object-detector`, `diffusion-runner`,
`speech-runner`, and a separate `llm-benchmark` target explicitly modeled on `mlx-lm benchmark`
(the tool `apple-models-bench.md`'s numbers were generated with).

## Contribution policy — verified current

`README.md`'s "Contributing" section states, verbatim, as of the `main` branch tip fetched this
session: **"We are not accepting code contributions at this time... We are not accepting pull
requests at launch while we learn how the community uses this project. If you open a pull
request, it will be closed."** GitHub Issues *are* open (bug reports, model requests, feature
suggestions via the templates in `.github/ISSUE_TEMPLATE/`). No separate `CONTRIBUTING.md`
exists in the tree — this text lives directly in `README.md`. This matches what
[`pipelined-engine.md:468`](pipelined-engine.md) already noted ("Apple's `coreai-models` repo is
**issues-only** (no PRs)") — **confirmed still accurate**, and is the reason this repo's own
`conversion/overlay/` and Swift-fork mechanisms exist instead of upstreaming fixes: there is
nowhere to upstream them to. License is BSD 3-Clause (`LICENSE`), matching this repo's own
license framing in `conversion/overlay/README.md:45-48` ("provided under the same terms as this
repository's LICENSE").

## Related reading in this knowledge base

- [`coreai-overview.md`](coreai-overview.md) — the three-repo Apple picture (`coreai-torch`,
  `coreai-optimization`, `coreai-models`) and why this project exists relative to Apple's zoo.
- [`compute-units-and-authoring.md`](compute-units-and-authoring.md) — the condensed ANE/GPU/CPU
  cheat-sheet distilled from the same skill references detailed in full above.
- [`apple-models-bench.md`](apple-models-bench.md) — the missing performance numbers for every
  official recipe (Apple ships zero benchmarks itself).
- [`conversion-guide.md`](conversion-guide.md) — this repo's own `TorchConverter` API notes
  (the `coreai-torch` PyPI package, used unmodified by both Apple's recipes and this repo's).
