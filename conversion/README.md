# Conversion

PyTorch → Core AI `.aimodel`: re-authored models + convert / verify / compress scripts.

**One-command entry point**: the configuration that produced each published bundle lives beside
its card in [`models/<model>/recipe.toml`](../models/) and runs through
[`zoo_convert.py`](zoo_convert.py) —
`python3 zoo_convert.py doctor` checks your venv + overlay wiring, then
`python3 zoo_convert.py run qwen3.5-0.8b`. The model authoring code the scripts import is
packaged in [`overlay/`](overlay/) (pinned apple/coreai-models base + patch + files).

**Was it published correctly?** [`zoo_verify.py`](zoo_verify.py) compares a published bundle's
tokenizer, chat template, context length and declared precision against the source repository it
names in its own `metadata.json` — no oracle, no device, no weights, so it runs over the whole
catalog in minutes: `python3 zoo_verify.py --all --json ../models/_VERIFY.json`. Results land in
[`../models/_INVENTORY.md`](../models/_INVENTORY.md) via `scripts/gen_inventory.py`.

**Does it still load on this OS?** [`zoo_smoke.py`](zoo_smoke.py) is the tier below that: it
reads each asset's producer stamp off the Hub, downloads the bundle, loads every `.aimodel` in it
through the Core AI runtime in a child interpreter, and runs the tier-1 checks — recording which
OS build and toolchain did so in `../models/_SMOKE.json`. `python3 zoo_smoke.py --top 20` is the
sweep an OS release day needs (`--dry-run` prints the plan with sizes first);
`--all --stamp-only` audits the producer of all 278 bundles in minutes without downloading any.
It never loads a compiled `.aimodelc` or an iOS-path bundle on a Mac; those are device work.

**No setup, where possible**: 14 scripts carry a PEP 723 inline dependency block, so
`uv run conversion/<script>.py ...` builds a throwaway environment and runs them — no venv, no
overlay. Every one of them is checked with `uv lock --script`. Scripts that import re-authored
model code (`coreai_models.models.*`) cannot work this way; they need the overlay environment.

**Paths**: no script hardcodes a home directory. Checkpoint downloads, exports, sibling checkouts
and the Hugging Face cache all resolve through [`_paths.py`](_paths.py) — run it
(`python3 _paths.py`) to print where they land, and set `ZOO_WORK_ROOT` / `ZOO_EXPORTS` /
`ZOO_CODE_ROOT` / `HF_HUB_CACHE` to move them.

**The tail every exporter shares**: writing `metadata.json`, specifying the `lm_head`
quantization, and copying the tokenizer in are the same job in every driver, and live in
[`_bundle.py`](_bundle.py). They used to be copy-pasted — 28 private copies of the metadata
writer, in ten variants — and the drift cost a `TypeError` at the end of a completed export and
a tokenizer fetch that could not retrieve a file its own copy step asked for. What a driver
*quantizes* stays in the driver: `linear_quant_config` is the recipe, not boilerplate.
[`_bundle_selftest.py`](_bundle_selftest.py) holds each driver's pre-extraction output and checks
the call sites on disk still produce it, byte for byte.

## How it relates to Apple's `coreai_models`

The re-authored decoders use `coreai_models` primitives (KVCache, RMSNorm, RoPE, SDPA, SSMState,
…) and the `coreai_models.export` pipeline. Apple's `coreai-models` does **not** take PRs and does
**not** register these newer models, so the model authoring + export wiring lives in **our fork /
overlay** of that package. Concretely, the additions are:

- `models/macos/qwen3_5.py`, `models/macos/qwen3_5_moe.py`, `models/macos/gemma4_text.py` — re-authored decoders (+ config shims).
- `models/registry.py` entries (`qwen3_5_text`, `gemma4_text`) + `model_registry.py` short-name
  presets + `export/{presets,metadata,macos,pipeline}.py` hooks (e.g. `export_core()` routing,
  macOS int8 palettization, multi-function front-end gather).

These are packaged as a pinned-base patch set + file overlay in [`overlay/`](overlay/) —
`python3 overlay/apply.py /path/to/coreai-models` on a fresh upstream clone reproduces the
conversion environment byte-for-byte (verified 2026-07-03). After new porting work on the live
checkout, refresh it with `overlay/regen.sh`.

## Scripts

Several families share one exporter (the Qwen3.5 script also drives Ornith and
Qwen3.6-27B), which is why the scripts live here rather than under `models/<model>/` as in
Apple's repo; each recipe names the script it runs.

- **FastContext-1.0-4B-SFT (STOCK — no re-authoring): `coreai.llm.export fastcontext-4b`** —
  Microsoft's Qwen3-4B-arch repo-exploration agent is byte-identical to `Qwen/Qwen3-4B`, so it
  rides the stock `coreai_models` `qwen3` graph unchanged (GQA, q/k-norm, tied embeddings all
  handled). The *only* additions are a `model_registry.py` short-name preset (`fastcontext-4b` —
  macOS 4bit + iOS palettized, both reuse the `qwen3-4b` recipe) + an `export/metadata.py` entry;
  no decoder code. macOS GPU 4-bit linear-INT4, parity **23/24 argmax (ppl 1.41)** vs HF fp16.
  On-device the 4B graph can't specialize, so ship the **AOT** bundle:
  `coreai-build compile exports/fastcontext_4b_dynamic/*.aimodel --platform iOS --preferred-compute gpu --architecture h18p`
  → the `gpu/` `.aimodelc` (see [`../knowledge/aot-and-specialization.md`](../knowledge/aot-and-specialization.md)
  and [`../models/fastcontext/README.md`](../models/fastcontext/README.md)). The zoo's first stock-architecture model —
  the template for "drop-in any HF model the stock exporter already supports."
- **Holo2-4B (STOCK Qwen3-VL drop-in): `export_qwen3_vl_pipelined.py int8lin --hf-id Hcompany/Holo2-4B`** —
  H Company's GUI-grounding / computer-use VLM is byte-identical to Qwen3-VL-4B, so the zoo's Qwen3-VL
  pipeline converts it with just an HF-id swap (no model code). Emits decoder + `_s1` gate twin + fp16
  vision tower. Parity (vs fp32 HF, GPU engine): **vision cos 0.9999, decoder S=1 4/4 + 16/16 decode
  steps token-exact**. The decode `_s1` is a STATIC graph → specializes on-device, no AOT (unlike a
  dense 4B *dynamic* bundle). Gate harness for tf-4.57: `_smoke/qwen3vl_capture_ref.py` +
  `test_qwen3vl_aimodel_gate.py` patched (rope_scaling/get_image_features-tuple/get_rope_index sig;
  `QWEN3VL_MID`/`QWEN3VL_REF`/`QWEN3VL_NLAYERS=36` envs). See [`../models/holo2/README.md`](../models/holo2/README.md). The
  VLM analogue of the FastContext stock-drop-in template.
- **S1-mini (STOCK dense qwen3 drop-in): `export_s1_mini_decode_pipelined.py int8lin`** —
  Superwhisper's ASR text normalizer is a Qwen3-0.6B finetune, byte-identical in shape to
  `Qwen/Qwen3-0.6B`, so it rides the stock `coreai_models.models.macos.qwen3` graph with no
  model code (that class already fuses q/k/v and q_norm/k_norm). Deliberately **no `*hu`
  mode**: the head is TIED to the 151936×1024 embedding, and untying it to quantize ADDS
  156 MB beside the fp16 embedding instead of replacing it — a pure loss at every bit width.
  int8lin 759 MB, **268.4 decode / 4161 prefill tok/s M4 Max**, oracle gate 16/16.
  **int4lin is a measured no-go** and this is the interesting part: it passes the SAME 16/16
  oracle gate and still corrupts digits (`$23,450`→`$2,345`, `107`→`177`). A free-run
  continuation gate is blind to a single-task model's task, so this port ships a second gate
  that speaks the model's own input format — `_smoke/gate_s1_mini_task.py`, 14 cases across
  the card's control axes, verdict against the released weights (the upstream card's own
  printed examples reproduce only 9/14 and cannot be used as fixtures).
  See [`../knowledge/s1-mini-port.md`](../knowledge/s1-mini-port.md) and
  [`../models/s1-mini/README.md`](../models/s1-mini/README.md).
- Gemma 4: `convert.py` / `convert_palettize.py` (int8 `all8`) / `convert_stateful*.py` (stateful +
  ring) / `convert_head.py` / `check_pipeline.py` / `verify_*` — the full convert+verify harness.
- Qwen3.5: parity ladder + fp16/int8 + head-split + stateful-palettize harnesses.
- On-device export (kept artifacts): `export_qwen3_5.py [0.8b|2b]`, `export_gemma4_frontend.py`.
- **Qwen3.5 pipelined fast path (in this dir): `export_qwen3_5_decode_pipelined.py`** —
  decode-only loop-free bundles for Apple's `coreai-pipelined` GPU engine. Ship config for
  BOTH sizes is `int8hu --head-sym` (per-block-32 **absmax** int8 head — clipping corrupts
  big-vocab heads; per-channel axis-0 is BROKEN on the beta GPU delegate, and the historical
  `*_perchan_sym` bundle names actually contain per-block-32 heads — see the qwen3.5 card):
  0.8B **210 tok/s M4 Max / 69.7–74.0 iPhone 17 Pro**
  (fp16-head int8lin: 204 / 50.3–51.5; custom-kernel CLI was 58.5); `--hf-id Qwen/Qwen3.5-2B`
  → 2B **161 / 28–30** (int8lin: 127 / 19–21). Needs the Swift engine patch
  `../apps/coreai-pipelined-extra-states.patch` and `COREAI_CHUNK_THRESHOLD=1` at run time.
- **LFM2.5 pipelined (in this dir): `export_lfm2_decode_pipelined.py [fp16|int8lin|int8hu]`** —
  the first non-Qwen rider: LiquidAI's conv+attention hybrid, decode-only S=1 (loop-free by
  construction — no scan anywhere), **253 tok/s int8lin / 276.5 with the int8 head
  (`int8hu --head-sym`) / 162 fp16 on M4 Max**, oracle gate 16/16 (all three). Model overlay: `models/macos/lfm2.py` on the `coreai-models` checkout — it bakes in
  two macOS-27-beta GPU-delegate workarounds (fused single conv-state write; fp32 attention
  projections). Same engine patch + `COREAI_CHUNK_THRESHOLD=1` run contract. See
  [`../models/lfm2.5/README.md`](../models/lfm2.5/README.md).
  **The same script converts LFM2.5-2.6B** — `--hf-id LiquidAI/LFM2.5-2.6B` and nothing else; the
  module reads `config.json` generically, so 30 layers (22 conv / 8 attn) and a 10752 MLP come for
  free → **116.7 tok/s int8hu / 139.2 int4lin (2.0 GB)**, oracle gate 16/16 both. That checkpoint is
  transformers-v5 era and needed two silent fixes that now live in the script: RoPE theta read from
  `rope_parameters` as well as the flat key, and `save_tokenizer()` for a
  `tokenizer_class: "TokenizersBackend"` that 4.x cannot resolve (it raises *after* the .aimodel is
  written). See [`../models/lfm2.5-2.6b/README.md`](../models/lfm2.5-2.6b/README.md) and
  [`../knowledge/lfm2.5-2.6b-port.md`](../knowledge/lfm2.5-2.6b-port.md).
- **LFM2.5-VL (in this dir): `export_lfm25vl_pipelined.py [int8lin] [--vision-mode fp16]`** — one
  run emits both halves of the zoo's smallest VLM (658 MB): a fixed-grid **SigLIP2-NaFlex** tower
  (`patches [1024,768] → image_embeds [256,1024]`, 181 MB fp16, **18.0 ms/image** on M4 Max,
  **33.6 ms on an iPhone 17 Pro**) and the
  LFM2 decoder above with an `image_embeds` static input and extension ids `V+slot` (477 MB
  int8lin, **112.0 tok/s decode on iPhone 17 Pro** via the `ios-h18p` AOT variant). The decoder is the *same* module — the VL checkpoint keeps it under
  `model.language_model.` — so the new overlay code (`models/macos/lfm2_vl.py`) is the tower, the
  projector, and the splice. `--text-core` exports the decoder with no image input: **387.2 tok/s
  M4 Max**, and the only way to benchmark or `coreai_gate.py` this port, because `llm-runner`
  cannot bind the VLM bundle's image buffer. int4 is a no-go here (0/9 suite cases). Host
  preprocessing is gated in NumPy first (`_smoke/lfm25vl_preprocess.py`).
  **`--hf-id LiquidAI/LFM2.5-VL-3B` converts the 3B** with no code change (vision 75.7 ms/image,
  text core 105.3 tok/s, suite 7/9 at int8 **and** int4 — the opposite of the 450M's int4 cliff);
  it is Mac-only because its AOT `resources.bin` is 3.13 GiB against the iOS 2 GiB load wall.
  Two host details are per-checkpoint: `resample` (450M BILINEAR / 3B BICUBIC) and whether the
  tokenizer prepends BOS (the 3B's does not, and without it the model answers ' F, F, F, F'). See
  [`../models/lfm2.5-vl/README.md`](../models/lfm2.5-vl/README.md) and
  [`../knowledge/lfm2.5-vl-port.md`](../knowledge/lfm2.5-vl-port.md).
- **Shieldstral (in this dir): `export_shieldstral.py [int4lin] [--seq-len 512]`** — Mistral's 3B
  safety model as a **stateless classifier graph**: `(input_ids[1,S], attention_mask[1,S]) ->
  probs[1,2] = softmax([no, yes])`. No KV cache, no loop, and the head is **two rows** of the tied
  embedding (the full 131k-row head is 805 MB a classifier never reads). Same archetype as
  `export_qwen3_reranker.py`. The conversion venv (4.57.6) **cannot load this checkpoint**, so the
  oracle runs on transformers git main and the exporter's premise — `ministral3` is Mistral +
  YARN — is gated at cos 1.000000 / |ΔP| 0.00000 rather than assumed. **9/9 verdicts vs fp32** at
  every precision; **232.5 ms/verdict at S=512, 123.6 ms at S=256** on an M4 Max, **624.7 / 371.9 ms on an iPhone 17 Pro** (9/9 on device, 2.336 GiB AOT), 2.53 GB. See
  [`../models/shieldstral/README.md`](../models/shieldstral/README.md).
- **North-Micro-Vision (in this dir): `export_northmv_pipelined.py [int8lin]`** — Cohere's 2.4B
  multilingual VLM. The vision half needed **no code**: its tower is the Qwen3-VL visual encoder
  at SigLIP2-SO400M dimensions, so `vision_encoder_from_hf` is a config shim over the existing
  `Qwen3VLVisionEncoder` (zero missing keys, every seam cos 1.000000). The new module
  (`models/macos/cohere_compass.py`) is the decoder: parallel Cohere block, mean-subtracting
  LayerNorm, `SSSF x 7` layer types where the full-attention layers have **no positional
  encoding**, logit_scale 0.25, 262k tied vocab. **145.3/118.6 tok/s M4 Max, 21.5/18.2 on an
  iPhone 17 Pro with the image oracle 24/24**; suite 9/9 at int8, 0/9 at int4 (not published).
  The oracle needs transformers git main. See [`../models/north-micro-vision/README.md`](../models/north-micro-vision/README.md).
- **Gemma 4 E2B / E4B pipelined fast path (in this dir): `export_gemma4_decode_pipelined.py [int4lin]`** —
  decode-only S=1 bundle whose per-layer-embedding rows arrive as a per-token INPUT (the 9.4 GB
  PLE table stays a host mmap): in-graph embed + softcapped head, ONE unified padded KV pair,
  oracle 8/8, **70.9 tok/s decode on M4 Max** (+20-25% over the int4km-kernel CLI, zero custom
  kernels; int4-LINEAR per-block — eager-palettized k-means LUTs measure 2.25× slower at the
  same bytes). Needs the full patch stack incl.
  `../apps/coreai-pipelined-per-token-inputs.patch`, `COREAI_CHUNK_THRESHOLD=1`, and a
  `PerTokenInputProvider` that dequants the int8 PLE row dump per token. Add **`--tbl`** to
  export the variant whose PLE table is a STATIC graph input instead (in-graph gather; no
  provider, no per-token decode wait — **77.0 tok/s on M4 Max**, the best Mac gemma4 config;
  needs `../apps/coreai-pipelined-static-inputs.patch` + an app that binds the two dump files
  via `EngineOptions.staticInputBuffers` — buffer-mode traps in
  [`../knowledge/pipelined-engine.md`](../knowledge/pipelined-engine.md)).
  **`--hf-id` swaps the checkpoint**: Google's official QAT releases
  (`google/gemma-4-{E2B,E4B}-it-qat-q4_0-unquantized`) ride the same script — bundle names
  gain `_qat`, E2B-QAT measures 74.7/78.9 (provider/tbl) and **E4B (42L, 2 KV heads, dense
  — no MoE, zero model-code changes) 53.2/55.8**, all oracle 8/8; q4_0 IS per-block-32
  absmax int4, so these bundles carry Google's "≈ bf16" QAT quality claim. Regenerate the
  PLE dump (`--out`) and the oracle (`gen_gemma4_prompt.py --tag`) from the same
  checkpoint; `--lin-sym` exports the literal-q4_0-grid (absmax) variant (measured: same
  gate, same speed). See [`../models/gemma4-e4b/README.md`](../models/gemma4-e4b/README.md).
- **Granite 4.0-H pipelined (in this dir): `export_granite4h_decode_pipelined.py [fp16|int8lin|int8hu]`** —
  the first Mamba2/SSM-scan rider: at S=1 the selective scan is a single recurrence step
  (loop-free, no while_loop), states = KV (4 attn layers) + conv/SSM stacks (= the ≤2
  extra-states budget). 1b int8lin **136.5 tok/s** / fp16 103.6 on M4 Max, oracle gate
  16/16 (`int8hu --head-sym` also gates 16/16 but is Mac-flat at 134.2 — device re-test
  pending, the qwen "Mac no-win ≠ device no-win" pattern); `--hf-id ibm-granite/granite-4.0-h-350m` exports the 350m (ship fp16 there, 191
  tok/s — int8 fails the gate at that scale and is no faster). Model overlay:
  `models/macos/granite4h.py`. Same engine patch + `COREAI_CHUNK_THRESHOLD=1` run contract.
  See [`../models/granite-4.0-h/README.md`](../models/granite-4.0-h/README.md).
- **Qwen3-VL 2B pipelined — the first VLM (in this dir): `export_qwen3_vl_pipelined.py [fp16|int8lin|int8hu]`** —
  emits the text-decoder bundle (+ a `_s1` static-query twin that carries the python
  oracle gates) AND the fixed-grid vision encoder `.aimodel`. Multimodal state rides the
  static-inputs patch: image/deepstack embeds as rewritable owned buffers, image tokens
  as extension ids `V+slot`, interleaved M-RoPE derived in-graph from (ids, pos) + two
  `[1] i32` shift inputs. int8hu **187.6 tok/s decode** on M4 Max, multimodal oracle
  gates 4/4+16/16+HF-seeded vs fp32-HF; iPhone numerics 24/24 (text AND image prompts).
  Model overlay: `models/macos/qwen3_vl.py`. See [`../models/qwen3-vl/README.md`](../models/qwen3-vl/README.md).
- **Gemma 4 E2B VISION pipelined — the second VLM (in this dir): `export_gemma4_vl_pipelined.py [fp16|int8lin|int4lin] [--lin-sym] [--tbl]`** —
  the Qwen3-VL rider recipe on the shipped gemma4 decoder: a fixed-grid SigLIP-class vision
  tower (48×48 patches = 768×768 square → 256 soft tokens, checkpoint-calibrated activation
  clamps) + ONE new `image_embeds [280,1536]` static input. Image span is CAUSAL on E2B
  (verified vs the fp32 mask dump); PLE rows for image steps gather the PAD row, so the PLE
  tables stay byte-identical to the text ship. Ship = `int4lin --lin-sym` (QAT q4_0 = absmax
  grid — clipping flips real-margin argmaxes at a 272-token horizon): Mac `--tbl` **95.2
  prefill / 82.4 decode tok/s**; iPhone provider mode **41.2 / 25.5** (the tbl gather overflows
  the iOS ~208 KB per-encode MPSGraph scratch heap — an engine bug, second reproducer).
  Model overlay: `models/macos/gemma4_vision.py` + the `Gemma4VLPipelined*` subclasses in
  `models/macos/gemma4_pipelined.py`. See [`../models/gemma4-vl/README.md`](../models/gemma4-vl/README.md).
- **Unlimited-OCR — document OCR, zoo's first doc-OCR, on the STOCK runtime (no patch): [`unlimited_ocr/`](unlimited_ocr)** —
  baidu/Unlimited-OCR (3B-A0.5B MoE, MIT) → fp16 DeepEncoder vision `.aimodel` + a sym8 DeepseekV2
  **R-SWA** MoE decoder (unified `prefill`+`decode` bundle). Driven on `inputs_embeds` directly, so
  no static-input patch. The novel piece = a **fully-static decode graph** (data-driven KV write +
  full fixed-buffer R-SWA mask, `pos [1]` as a value not a shape) → no per-step recompile (a growing
  shape *faults* on Metal 4) → **flat 12.7 ms/token**. Image→markdown (tables→HTML, formulas→LaTeX);
  arrangement assets shipped raw for host-side assembly. App: `apps/CoreAIOCR` (drives the stock
  runtime via `InferenceFunction.MutableViews`). See [`../models/unlimited-ocr/README.md`](../models/unlimited-ocr/README.md)
  + [`../knowledge/unlimited-ocr-rswa-static-decode.md`](../knowledge/unlimited-ocr-rswa-static-decode.md).
- **GLM-OCR — doc-OCR (GLM-4.V small): `export_glm_ocr_pipelined.py [fp16|int8lin|int8hu] [--grid-h H --grid-w W]`** —
  zai-org/GLM-OCR (0.9B, MIT) → CogViT vision `.aimodel` (fp16) + a GLM text decoder (int8hu) on the
  pipelined rope-shift rider (`image_embeds` + `rope_shift_start`/`rope_shift_amount`). GLM ChatML
  (`[gMASK]<sop>…<|assistant|>`), single-pass `Text Recognition:`, tables → Markdown. App: `KitGlmOcrReader`
  in `Examples/ReadDoc`. See [`../models/glm-ocr/README.md`](../models/glm-ocr/README.md)
  + [`../knowledge/glm-ocr-port.md`](../knowledge/glm-ocr-port.md).
- **MinerU2.5-Pro — whole-page auto-structuring doc-OCR (stock Qwen2-VL): `export_mineru_pipelined.py [fp16|int8lin] [--grid-h H --grid-w W] [--prefill-chunk 64]`** —
  opendatalab/MinerU2.5-Pro (1.2B, Apache-2.0) → Qwen2-VL ViT vision `.aimodel` (fp16) + Qwen2-0.5B
  int8lin decoder on the same rider. **Two grids**: 768 (32×24 portrait) recognition + 1036² (37×37
  square) layout for the 2-stage pipeline (`Layout Detection:` → per-region recognition → `json2md`,
  tables → `<table>` HTML via OTSL). `--prefill-chunk 64` = `pf64` multifunction chunked prefill. App:
  `KitMineruReader.readStructured` in `Examples/ReadDoc`. See [`../models/mineru/README.md`](../models/mineru/README.md)
  + [`../knowledge/mineru-port.md`](../knowledge/mineru-port.md).
- **Qwen3.6-35B-A3B pipelined — the first MoE (in this dir): `export_qwen3_6_decode_pipelined.py [int8lin|int8hu]`** —
  Qwen3.5's hybrid decoder + a 256-expert top-8 sparse-MoE FFN (+ shared expert), 40 layers,
  GVA GatedDeltaNet (32 value / 16 key heads). Experts ride Apple's `SwitchGLU`/`GatherMM`;
  the 4-D expert weights quantize with the documented SwitchLinear override
  (`block_size [1,1,1,32]`), router + shared-expert gate stay fp16, head is absmax int8.
  `int8hu --head-sym` = 35 GB bundle, **30.9 tok/s decode on M4 Max** (35B-A3B, ~3B active);
  fp16 full-scale eager ≡ bf16 HF oracle (margin rule), int8 eager teacher-forced gate.
  Mac-only (35 GB > iPhone jetsam). Model overlay: `models/macos/qwen3_5_moe.py` (MoE FFN +
  packed-expert loader) on top of `qwen3_5.py` (which gained the GVA head-repeat). **NOTE:
  raw `AIModel.load(gpu)` aborts on the MoE→ANE path (`ANE compilation writeToFile failed!`
  / 100 GB temp blowup); the real engine's `expectFrequentReshapes` avoids it — run via
  `llm-benchmark`/`llm-runner`, not raw load.** See [`../models/qwen3.6/README.md`](../models/qwen3.6/README.md).
- **Qwen3.6-27B (dense) pipelined — reuse the qwen3.5 script: `export_qwen3_5_decode_pipelined.py int8hu --head-sym --hf-id Qwen/Qwen3.6-27B`** —
  the **dense** Mac-class companion to the 35B-A3B: the *same* Qwen3.5 hybrid decoder (3:1
  GatedDeltaNet + gated full attention, head_dim 256) run **without MoE**, 64 layers, GVA
  GatedDeltaNet at **48 value / 16 key heads** (ratio 3 — the head-repeat is config-driven, no
  new code) and an untied 248320-vocab head (the loader picks it up from the checkpoint root).
  `int8hu --head-sym` = **28 GB bundle, 15.9 tok/s decode on M4 Max** (~87 % of the bandwidth
  ceiling — a *dense* 27B reads the whole model per token, unlike the 35B-A3B's ~3B active).
  int8 == full precision at every confident position (teacher-forced vs bf16 oracle; the lone
  confident oracle disagreement is an fp16-identical bf16 artifact). Mac-only (28 GB > iPhone
  jetsam). No MoE files — reuses `models/macos/qwen3_5.py` directly. See
  [`../models/qwen3.6-27b/README.md`](../models/qwen3.6-27b/README.md).
- **Qwen3.8-27B pipelined — the same script, next generation: `export_qwen3_5_decode_pipelined.py int8hu --head-sym --hf-id Qwen/Qwen3.8-27B`** —
  the Qwen3.8 flagship generation's only open compact model (dense 27B VLM checkpoint; this
  is its text decoder), architecturally **byte-identical to Qwen3.6-27B** (same config, same
  851-key text weight map) — **zero new export code**, ported on release day. `int8hu
  --head-sym` = **28 GB bundle, 15.7 tok/s decode / 16.2 prefill on M4 Max** (bandwidth-bound
  dense 27B, matches the 3.6 within noise). Gate: fp16 control **16/16 exact** vs the bf16
  oracle (cleaner than 3.6 — no bf16 artifact), int8hu 15/16 with zero confident flips (the
  miss is a 0.061-margin tie). `mtp.*` deliberately skipped (GDN-hybrid spec-decode caps at
  ~1.2–1.3× so the MTP head isn't worth a graph). Release-day HF CDN crawl routed via
  sha256-verified ModelScope — see
  [`../knowledge/qwen3.8-27b-port.md`](../knowledge/qwen3.8-27b-port.md) and
  [`../models/qwen3.8-27b/README.md`](../models/qwen3.8-27b/README.md).
- **Qwen3.8-27B VISION path — `export_qwen38vl_pipelined.py int8hu`** — the qwen3.5 family's
  first vision authoring (`models/macos/qwen3_5_vision.py`, no deepstack): one run emits the
  **fp16 ViT tower** (0.9 GB, `patches [1024,1536] → image_embeds [256,5120]` at a baked
  512×512 grid) and the **embeddings-input VLM decoder** (28 GB int8hu, multifunction "main"
  S=1 + "prefill" S=16 chunked GDN — **86.0 prefill tok/s vs 16.2** for the S=1 text bundle;
  S=32 passed the suite then collapsed on real photos: fp16 doubling-inverse overflow, see the
  port note)
  plus the host-side `embed_tokens.safetensors`. Interleaved mRoPE from three host-fed
  position planes (host contract: `_smoke/qwen38vl_host.py`, asserted vs oracle-captured
  positions). Not engine-drivable; the python driver needs the AOT h16c compile (JIT asserts
  in MPSGraph's ANE region pass). Gates: tower **cos 1.000000 vs an fp32 tower reference**
  (a bf16 full-model oracle is NOT a valid tower target), eager fp16 mixed text+image
  **32/32**, full-chain int8hu suite **5/6 token-exact** (140/144; the miss a 0.055-margin
  tie).
- **Ornith-1.0-9B (agentic coding) pipelined — reuse the qwen3.5 script: `export_qwen3_5_decode_pipelined.py int8hu --head-sym --hf-id deepreinforce-ai/Ornith-1.0-9B --max-ctx 8192`** —
  DeepReinforce's self-scaffolding agentic coder is a **stock Qwen3.5 hybrid decoder**
  (`model_type qwen3_5`, 32 layers, GVA 32v/16k, GQA 16/4 hd256, untied 248320 head at the
  checkpoint root, `model.visual.*` skipped, no MTP weights) — **zero new export code**.
  `int8hu --head-sym` = **9.8 GB bundle, 48.3 tok/s decode / 48.5 prefill on M4 Max** (ship);
  `int4lin` = **7.5 GB, 58.9 tok/s (+22%)** and — a family first — ALSO gates 24/24 exact
  (0.8B/2B int4 NO-GO, 27B borderline; short-context gate only, see the card).
  Gate: fp32-oracle (margin-validated, min 0.205) eager teacher-forced **24/24 exact** for
  fp16, int8hu AND int4lin; release `llm-runner` greedy on raw prompt ids **12/12 ≡ oracle** (both bundles).
  Oracle/gate scripts: [`../_smoke/gen_ornith9b_ref.py`](../_smoke/gen_ornith9b_ref.py) +
  [`../_smoke/test_ornith9b_eager_gate.py`](../_smoke/test_ornith9b_eager_gate.py).
  Mac ship (9.8 GB > iPhone jetsam). See [`../models/ornith-1.0-9b/README.md`](../models/ornith-1.0-9b/README.md).
- **Gemma 4 12B (dense) pipelined (in this dir): `export_gemma4_12b_decode_pipelined.py [int4lin|int8lin|fp16] [--lin-sym] [--metal-sdpa]`** —
  the 12B-class **clean dense** Gemma 4 (`gemma4_unified`): no PLE/AltUp/Laurel/MoE/KV-sharing,
  48 layers, dual head_dim 256/512, dual KV-head count via `attention_k_eq_v` (full layers = 1 KV
  head, no `v_proj`, value = raw k_proj). Both attention shapes ride ONE growing KV pair
  (`[48,1,8,S,512]`, sliding padded 256→512, full padded 1→8 heads), so the bundle loads on the
  **stock pipelined engine — no engine patch** (2 states). In-graph embed + tied head + softcap.
  **`--metal-sdpa` is required to RUN:** the full layers' 16-head × 512 Q (16 KB fp16) overflows
  MPSGraph's SDPA scratch heap ([apple/coreai-models#27](https://github.com/apple/coreai-models/issues/27)),
  so the full layers' SDPA op is swapped for a custom flash-decode Metal kernel
  (`models/macos/gemma4_dense_metal_sdpa.py`) — the first Core AI runtime for a ≥16-head × 512
  full-attention model. Ship = `int8lin --metal-sdpa` (verified-clean, M4 Max 22.2 tok/s decode) +
  `int4lin --lin-sym --metal-sdpa` (faster 4-bit, 33.0 tok/s). Overlays:
  `models/macos/gemma4_dense_{text,pipelined,metal_sdpa}.py`. Gate via
  `_smoke/engine_tokenmatch_gemma4_12b.py`, run SOLO (parallel python-GPU → MTL4CommandQueueError).
  See [`../models/gemma4-12b/README.md`](../models/gemma4-12b/README.md).
- **Gemma 4 31B (dense) pipelined (same script): `export_gemma4_12b_decode_pipelined.py int4lin --lin-sym --metal-sdpa --hf-id google/gemma-4-31B-it-qat-q4_0-unquantized`** —
  the **frontier dense** twin of the 12B (60 layers, hidden 5376, 32 heads, **4 global KV heads**,
  same dual head_dim 256/512). Reuses the 12B overlay verbatim — the `--metal-sdpa` kernel does
  block GQA over the unified cache (`repeat_interleave` replication: correct for the 31B's 4 global
  heads, a no-op for the 12B's 1). Same #27 scratch-heap bypass. Ship = `int4lin --lin-sym`
  (q4_0 QAT, ~19 GB Mac-only, M4 Max 17.2 tok/s decode). See [`../models/gemma4-31b/README.md`](../models/gemma4-31b/README.md).

- **RF-DETR (object detection, in this dir): `export_rf_detr.py --variant {nano|small|medium|large}`** —
  the zoo's first detector ([apple/coreai-models#14](https://github.com/apple/coreai-models/issues/14)):
  single static graph, `image [1,3,R,R]` RGB [0,1] → 300 cxcywh boxes + 91 COCO-id logit
  columns, **no NMS**. fp32 ship (fp16 = +7% speed, near-tie noise). Patches rfdetr 1.7.1 at
  import: constant-dim_t sine embed (float-arange converter abort), bool-free + floor-safe
  bilinear (int64-cmp buffer clobber; GPU floor=identity → `div(2x,2,floor)`), `torch._assert`
  no-op. Set-based detection gate vs torch fp32 (near-tie flip budget ≤2 — the reference
  itself flips confident queries under 1e-4 input noise on busy scenes). Also:
  `--variant seg-nano…seg-2xlarge` = RF-DETR-Seg instance segmentation (6 sizes, masks
  [1,Q,R/4,R/4], gated mask-IoU 1.000) and `--split` = backbone/head bundles for per-stage
  compute units. `pip install rfdetr==1.7.1`, torch ≤ 2.11.
  See [`../models/rf-detr/README.md`](../models/rf-detr/README.md).

- **YOLOX (single-stage anchor-free detection, in this dir): `export_yolox.py --variant s`** —
  the zoo's first YOLO-family / dense detector (CNN counterpart to RF-DETR's DETR):
  static graph `image [1,3,640,640]` BGR 0-255 letterboxed → `preds [1,8400,85]`
  (grid+stride DECODED cxcywh pixels + obj + 80 sigmoid class scores, all in-graph),
  host does `score=obj·cls` + **per-class NMS** (the DETR family needs none). A plain
  conv graph — Focus strided-slice stem, SPP maxpools, decoupled head, in-graph decode
  — converts with **zero coreai-torch workarounds** (unlike RF-DETR's four); the only
  trick is a one-line `cv2` stub so the model builds from `yolox.models` without OpenCV.
  fp32 ship (fp16 is no faster on GPU and adds near-tie noise — same call as RF-DETR):
  M4 Max GPU **4.80 ms / 208 FPS**, head cosine **1.000000** + detections IoU **1.000**
  cpu+gpu vs torch fp32. `--variant {nano,tiny,s,m,l,x}`, `--verify-image <img> --unit
  {cpu,gpu}` gates end-to-end. Needs a
  [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) checkout + `yolox_s.pth` +
  `pip install loguru`, torch ≤ 2.11. See [`../models/yolox/README.md`](../models/yolox/README.md).

- **Kokoro-82M (text-to-speech, the zoo's first TTS, in this dir): `export_kokoro.py`** —
  StyleTTS2 + iSTFTNet cut into **three** fixed-bucket `.aimodel` bundles around the
  data-dependent duration→alignment expansion: `predictor` (ids → duration/d/t_en),
  `prosody` (+ host alignment → asr/F0/N), `vocoder` (+ host hn-nsf source `har` →
  audio). Bundles are voice-independent (`ref_s` input); token/frame lengths are
  buckets (default 128/512), host-padded + trimmed; G2P (misaki) and the source STFT
  are host-side. The six bidirectional LSTMs become **masked unrolls** (fused nn.LSTM
  leaks pads → corrupts prosody), the 58 AdaIN InstanceNorms get a **frame-masked**
  norm (pad frames poison the L-axis stats), the hn-nsf source's STFT runs on the
  **host** (2π phase flip at the F0→0 boundary on the engine), and weight_norm MUST be
  folded (old hook-based → `module.weight` is random until a forward fires; the manual
  conv stand-ins read it). `ConvTranspose1d`/`conv_transpose1d` → zero-insertion +
  conv1d (symbolic length / all-zeros on the engine). Run on the **CPU** compute unit
  (unrolled LSTM ~8 ms). Spectral gate (`--verify`): magspec-corr 0.999 vs torch
  (waveform 0.98 = bounded pad-boundary effect). `pip install kokoro misaki soundfile`,
  torch ≤ 2.11. See [`../models/kokoro-82m/README.md`](../models/kokoro-82m/README.md).

- **pocket-tts (Kyutai text-to-speech, the zoo's first Kyutai model and first Mimi
  conversion, in [`pocket-tts/`](pocket-tts/)): `export_flowlm.py`, `export_flow_decoder.py`,
  `export_mimi_decoder.py`** — an AR flow-matching LM over Mimi latents at 12.5 Hz, a
  one-step flow decoder, and a streaming Mimi decoder to 24 kHz PCM. The flow-LM ships as
  **two functions over one shared rank-5 KV state**: `prefill` at a static width of 16
  tokens applied as a sliding window, and `step`. That windowing is also why the
  large-prefill SDPA lowering crash from the LFM2-Audio notes never fires here — the query
  block never gets big enough to stress the composite. `S_MAX = 512` is **derived** inside
  `flowlm_graphs.py` from the model's own generation bound (162 voice + 50 text + 234
  frames = 446) rather than configured; a fixture-sized 256 silently truncated real
  sentences and only an end-to-end sweep saw it. The Mimi decoder folds the rescale and the
  k=1 quantizer in-graph (`--fold-quantizer`) and holds its 12 streaming-state tensors as
  in-graph Core AI state (`--graph-state`), never reset across chunks. **Fixed-capacity KV
  caches for this family must be zero-initialised, never NaN-initialised**: upstream fills
  with `float("NaN")` and gets away with it by slicing the unwritten tail off before
  attention, which a fixed-capacity graph cannot do — a masked SDPA still multiplies V by a
  zero weight, so `0 * NaN = NaN` poisons the cache, silently. Each exporter gates its graph
  against the oracle capture (`gen_oracle.py --tag orc_a`) on eager, `cpu_only` and `gpu`
  before writing a bundle. Ported by [Rahul Rachuri](https://github.com/RahulRachuri). See
  [`../models/pocket-tts/README.md`](../models/pocket-tts/README.md).

## Reproduce (env)

Convert/verify needs the `coreai-core` + `coreai-torch` + `coreai-opt` Python env (macOS; the
`coreai-core` wheel is OS-coupled — re-verify after any OS bump). The HF reference oracles need a
transformers build with the target models. CLI: `coreai.llm.export <model> [--compression int8]`.

## License

The re-authored code derives from Apple's **BSD-3-clause** `coreai_models` and retains its
notices. This repo is licensed **BSD-3-Clause** (see the root [`LICENSE`](../LICENSE)).
