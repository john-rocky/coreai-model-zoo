# Core AI knowledge base

Hard-won, verified notes on Apple's Core AI (iOS/macOS 27) — what the docs don't spell out.

Published as a page per topic at **[john-rocky.github.io/coreai-model-zoo](https://john-rocky.github.io/coreai-model-zoo/)**,
so each note has a URL that can be cited. Apple documents the API surface; these notes cover the
part that is not written down anywhere — what the runtime does when you run it, the thresholds,
and the failure modes. Plain HTML, so a reader without JS, a search crawler and a coding agent
all get the same words.
[`llms.txt`](https://john-rocky.github.io/coreai-model-zoo/llms.txt) indexes every one of them.
For the long-form version of the same material, read
[The Art of Core AI](https://john-rocky.github.io/the-art-of-core-ai/) — a free book.

## Orientation
- [`undocumented-answers.md`](undocumented-answers.md) — **the questions Apple's docs leave open**,
  with measured answers: the AOT threshold, the iOS-only dynamic-KV miscompile at seq ≥2048, whether
  a 4B fits on the ANE, what the chunk threshold really dials, and why a hand-written kernel forces
  a single-token export.
- [`coreai-overview.md`](coreai-overview.md) — what Core AI is, the 3 Apple repos, the `.aimodel`
  format, the PyTorch → `.aimodel` → Swift-runtime pipeline.
- [`conversion-guide.md`](conversion-guide.md) — converting a PyTorch model to `.aimodel`: the
  canonical `TorchConverter` API + the gotchas that cost real time.
- [`ship-playbook.md`](ship-playbook.md) — **the end-to-end runbook**: converted `.aimodel` →
  CoreAIKit Swift engine → app → on-device (AOT + sideload + headless self-test for RTF) → publish
  (HF + zoo + post). The stage checklist + cross-cutting traps (gate-before-port, JIT→AOT, tokenizer
  retag), validated shipping Parakeet in one session.
- [`compute-units-and-authoring.md`](compute-units-and-authoring.md) — **ANE vs GPU vs CPU**: the
  static/BC1S/Conv2d/per-head/fp16 ANE rules vs the dynamic/fused/custom-kernel GPU rules, the
  macOS↔iOS export split, and the PSNR verification gates. Read this to choose a target.
- [`performance-ceiling.md`](performance-ceiling.md) — **reality check**: where Core AI LLM decode tops out
  (Mac GPU near its ceiling, MLX gap structural, fusion closed), what AOT does/doesn't do, and why ANE is an
  energy play not a speed one. Read before chasing a "dramatic speed" win.

## GPU-now track (speed)
- [`pipelined-engine.md`](pipelined-engine.md) — **read this first for decode speed**: riding Apple's
  `coreai-pipelined` engine = 3.5× over a hand-rolled per-token loop with ZERO custom kernels
  (qwen3.5: Mac 204 tok/s, iPhone 50.3–51.5). The decode-only loop-free export, the extra-states
  engine patch, the chunk=1 / warmup-256 traps, LUT-vs-linear int8, oracle gating, and what
  fits/doesn't (Gemma 4's PLE doesn't — yet).
- [`int8-head-and-decode-measurement.md`](int8-head-and-decode-measurement.md) — the **int8 LM head**
  decode lever (untie + absmax int8, ~half the per-token read) and **how to measure the win honestly**:
  controlled-bench vs in-app, VLM `image_embeds` dilution (+48% text core → +36% VLM), thermal-robust
  ratios, and the SwiftUI O(n²) re-decode that masquerades as a slow model.
- [`fm-provider.md`](fm-provider.md) — **zoo models behind Apple's `LanguageModelSession`**
  (WWDC 339): `CoreAILanguageModel(resourcesAt:)` = the whole integration (verified, incl.
  hybrid/SSM bundles on the patched pipelined engine), plus the own-conformance recipe that
  adds **tool calling** (verified round trip) and the protocol traps (dead prewarm, no guided
  generation on pipelined, re-prefill tax).
- [`custom-metal-kernels.md`](custom-metal-kernels.md) — `TorchMetalKernel` (WWDC 325): the API, the
  register-then-add order, MSL embedded in the `.aimodel`, GPU-only, what to (and not to) kernelize.
  Still the tool when a model CAN'T ride the pipelined engine (e.g. Gemma 4).
- [`rwkv7-recurrent-linear-attention-coreai.md`](rwkv7-recurrent-linear-attention-coreai.md) — the
  **no-KV playbook** for pure-recurrent / linear-attention LLMs (RWKV-7, Mamba-2, gated-delta-net):
  the matrix-state decode recurrence lowers to **standard ops** (no custom kernel); **O(1) fixed-size
  states, no KV cache** wired via `SSMState` + fused end-of-step writes; the **pipelined engine can't
  drive a no-KV model** (positional keyCache/valueCache) → a custom backend binding states BY NAME
  (like BitVLA); the recurrence-protecting quant recipe (`int8keepproj`); teacher-forced gating.
  Shipped RWKV-7 1.5B, iPhone 17 Pro 25.2 tok/s.
- [`tensorops-quantized-kernels.md`](tensorops-quantized-kernels.md) — the layer UNDER custom kernels
  (WWDC 330): TensorOps quantized matmul (int4/int8 = OS 26; fp4/fp8/int2 + E8M0 scale planes = OS 27),
  cooperative tensors, the FlashAttention recipe, and the M5/A19 GPU **neural accelerator** — the
  compute-bound/prefill lever hand-rolled MSL can't reach.

- [`prefix-cache-kv-reuse.md`](prefix-cache-kv-reuse.md) — **cross-turn KV reuse**: turn-2 TTFT
  0.23 s vs 23.3 s = **101× at a 4k context**, proven lossless (greedy token-identical). The
  orthogonal speed lever nobody's shipping on-device yet.
- [`spec-decode-design.md`](spec-decode-design.md) — **speculative decoding** on the pipelined decode
  bundles: the only lever past the decode bandwidth wall (verify K tokens per forward); design +
  feasibility. GDN-hybrid (Qwen3.5/3.6) static-S verify companion:
  [`spec-decode-hybrid-verify-design.md`](spec-decode-hybrid-verify-design.md).
- [`spec-decode-ngram-dense.md`](spec-decode-ngram-dense.md) — the **dense** case, measured:
  training-free n-gram drafting at **1.34–1.96× lossless** on a shipped 30B bundle, no
  re-export and no drafter. Verify cost is a **staircase** (S≤3 free, S=4–8 ~1.47×, S≥9 ~2.3×),
  so K belongs at the top of a plateau — the wrong K turns the same drafter into a slowdown.
- [`tensorops-zoo-impact-and-kernel-wins.md`](tensorops-zoo-impact-and-kernel-wins.md) — applied
  survey: where TensorOps quantized kernels and pure custom-Metal wins actually pay across the zoo.

## Benchmarks & comparisons
- [`apple-models-bench.md`](apple-models-bench.md) — measured numbers for Apple's own
  `coreai-models` export recipes — **the README Apple didn't write** (21 recipes, zero official
  numbers).
- [`coreai-vs-mlx-speed.md`](coreai-vs-mlx-speed.md) — every measured Core AI–vs–MLX decode
  comparison (same M4 Max, same protocol) + the **causal decomposition** of the gap: where Core AI
  wins, where it structurally can't, and why.

## ANE-later track (when the beta KV-write bug lifts + int4 head + AOT)
- [`aot-and-specialization.md`](aot-and-specialization.md) — specialization, `AIModelCache` /
  `AIModel.specialize()`, and AOT compile (`xcrun coreai-build compile` → `.aimodelc`,
  `--preferred-compute neural-engine`). The first-run-latency mitigation path.
- [`compression-reference.md`](compression-reference.md) — `coreai-opt` quantization & palettization
  API reference (int4/int8, granularity, mixed-precision, joint); the LM-head/embedding lever.
- [`coreai-beta-mpsgraph-kvwrite-bug.md`](coreai-beta-mpsgraph-kvwrite-bug.md) — the data-indexed
  in-graph KV write SIGSEGV (FB23024751 / apple#5): platform-agnostic (GPU too), host-cache workaround.
- [`coreai-ane-partition-cost.md`](coreai-ane-partition-cost.md) — an op the ANE cannot run (`topk`
  is the usual one) charges a **fixed** cost, not one that scales with its work: cutting k 30× buys
  nothing. Count boundary crossings, not ops. Self-contained reproducer (apple/coreai-torch#66).
- [`coreai-zero-sized-dim-abort.md`](coreai-zero-sized-dim-abort.md) — a 0-length `split` section
  or a width-0 output converts fine and then **aborts the process** on GPU and ANE (CPU runs it).
  Arrives from generic postprocess code that sizes a section by subtraction. Also: pin the compute
  unit explicitly when isolating — `SpecializationOptions.default()` can fall back to CPU and a
  fallback reads as a pass (apple/coreai-torch#68).
- [`porting-text-embedders.md`](porting-text-embedders.md) — an embedder is not a small LLM:
  gate the **ranking**, not the cosine; k-means skips `nn.Embedding`, which is 24% of a big-vocab
  embedder; fp16 tables are free when the checkpoint is bf16; int4 fails on the **interval**. Plus
  the three ways a retrieval benchmark lies, all of which make the numbers look better.
- [`coreai-torch-042-lowering-changes.md`](coreai-torch-042-lowering-changes.md) — which of the
  nine semantic lowering changes in coreai-torch 0.4.2 can reach a shipped bundle, decided by
  converting the same minimal module under both versions, diffing the graph, and running the ones
  that differ. **Exactly one moves values** (integer true-divide: `7/3` was `2.0`); fp16 batch norm
  rewrites the graph and changes nothing, because the runtime picks the kernel for a composite.
  Graph-identical proves output-identical; graph-different proves nothing until you run it.

## Agent & app layer (FM framework / App Intents / Evaluations / security)
- [`spotlight-rag-third-party.md`](spotlight-rag-third-party.md) — running Apple's WWDC26
  `SpotlightSearchTool` (local RAG as one Tool) behind a **third-party zoo model** via
  `KitLanguageModel`: only `.toolCalling` is needed (not guided generation), the tool returns
  metadata-not-body (hydrate with a companion `fetch_note` tool), guidance level is a token gate,
  and the thinking-model `/no_think` mitigation. Verified example: `coreai-kit/Examples/SpotlightChat`.
- [`dynamic-profiles-local-models.md`](dynamic-profiles-local-models.md) — WWDC26 `DynamicProfile`
  (242) routing between **two local zoo models** (0.6B triage ↔ 4B expert) in one
  `LanguageModelSession`, fully on-device/airplane-mode — the config Apple's on-device↔PCC demo
  doesn't show. The body-purity rule, switch re-prefill cost, two-resident-model footprint, and
  why the model-decision channel must be guided-gen (not a tool) on the stock engine. Example:
  `agent-demos/DualProfileChat`.
- [`visual-intelligence-third-party-model.md`](visual-intelligence-third-party-model.md) — running
  YOUR own model (CLIP / RF-DETR) behind the system **Visual Intelligence** camera/screenshot search
  (WWDC26 297): `IntentValueQuery` + `SemanticContentDescriptor`, model-agnostic by construction (no
  model param, no capability, no entitlement), and the real gate — running a model in the query's
  background-launch memory budget. Example: `coreai-kit/Examples/VisualIntel`.
- [`agentic-security-checklist.md`](agentic-security-checklist.md) — pre-ship checklist for
  on-device LLM agent apps (WWDC 347+343): indirect prompt injection, the Lethal Trifecta,
  `.onToolCall`/`.historyTransform` guardrails, App-Intents risk-based confirmation +
  `authenticationPolicy` + `OwnershipProvidingEntity`.
- [`evaluations-framework.md`](evaluations-framework.md) — Apple's Evaluations framework
  (WWDC 298/299/335) mapped to this project's oracle/margin gates; the `disallowed`-trajectory
  injection test; a Vault-style on-device eval suite.

## Runtime & decode internals
- [`compression.md`](compression.md) — this project's LLM-specific empirical compression notes
  (int8 floor, per-subsystem sensitivity); pairs with `compression-reference.md`.
- [`stateful-kv-cache.md`](stateful-kv-cache.md) — stateful decode export, dual/hybrid KV state,
  the sliding-window ring buffer, the dynamic prefill+decode graph.
- [`swift-runtime.md`](swift-runtime.md) — the Core AI Swift API, driving `.aimodel` from Swift,
  non-standard architectures, macOS/Xcode 27 setup (incl. running Xcode 27 beta without sudo).

## Model port notes (per-architecture lessons)
- [`muse-glimmer-port.md`](muse-glimmer-port.md) — Meta's 30B VLM text tower (NoPE full layers,
  gated attention, weight-less Q/K norm). Two traps that generalize: `_mutate_state_dict` never
  runs on the shared slice (nested text tower + untied head loads to garbage), and
  `hidden_states[-1]` is the final norm, not the last layer. Also the EDGE record for a port
  taken on knowing it is Mac-only against an already-converted MLX build.
- [`bitcpm-ternary-1.58bit.md`](bitcpm-ternary-1.58bit.md) — **1.58-bit ternary** MiniCPM4-8B: the
  zoo's first sub-int8 packed-GEMM Metal kernel; an 8B running in ~2.1 GB on the iPhone GPU.
- [`bitvla-1.58bit-vla.md`](bitvla-1.58bit-vla.md) — 1.58-bit **Vision-Language-Action** (robotics):
  image + instruction → 7-DoF actions, fully on-device.
- [`s1-mini-port.md`](s1-mini-port.md) — S1-mini (Superwhisper), a Qwen3-0.6B ASR text
  normalizer, and the port's real finding: **a passing conversion gate says nothing about a
  task model's task.** int4 scored the same 16/16 token-exact oracle as int8 and still
  corrupted digits. Also: why a tied head makes `*hu` the wrong idea rather than a smaller
  one, and why an upstream card's example outputs are not fixtures until you check they
  reproduce on the released weights (here: 9/14).
- [`minicpm5-1b.md`](minicpm5-1b.md) — the clean-LlamaForCausalLM recipe done end-to-end (hybrid
  Think/No-Think, untied head, 128K) — the most reusable conversion template in the zoo.
- [`youtu-mla-port.md`](youtu-mla-port.md) — dense **DeepSeek-style MLA at 2B on iPhone**: latent-KV
  attention with an absorbed flash-decode kernel.
- [`diffusion-llms-dllm.md`](diffusion-llms-dllm.md) — masked-**diffusion** LLMs (LLaDA): parallel
  canvas denoising, bidirectional attention, no KV cache — and how that maps to Core AI graphs.
- [`gliner2-pii.md`](gliner2-pii.md) — DeBERTa-v3 (disentangled attention) NER / schema-driven
  zero-shot extraction; the on-device PII-redaction model.
- [`glm-ocr-port.md`](glm-ocr-port.md) / [`mineru-port.md`](mineru-port.md) /
  [`unlimited-ocr-rswa-static-decode.md`](unlimited-ocr-rswa-static-decode.md) — the document-OCR
  trio: a Glm4v variant, whole-page parsing on stock Qwen2-VL, and an R-SWA MoE on the **stock
  runtime** with static-shape stateful decode.
- [`ovisocr2-port.md`](ovisocr2-port.md) — **in progress, nothing published**. OvisOCR2 tops
  OmniDocBench v1.6 (96.58) over the zoo's own MinerU2.5-Pro (95.69) and is config-identical to
  Qwen3.5-0.8B, so it is a weights swap on shipped authoring. Carries the port's two transferable
  findings: the **first non-square vision grid** through `qwen3_5_vision.py` (gated exact in fp32,
  a 4-row fp16 tail that is *not* the blank margin), and the **stop token living in
  `tokenizer_config.json`, not `config.json`** — on the config value the bundle never terminates.
- [`video-world-models-vjepa2.md`](video-world-models-vjepa2.md) — V-JEPA 2: a self-supervised
  **video world model** as on-device action classification (16-frame clips).
- [`gemma4-mixedbit-qat-transplant.md`](gemma4-mixedbit-qat-transplant.md) — extracting Google's
  mobile **mixed-bit QAT** weights and transplanting them into Core AI bundles.
- [`gemma4-ple-static-input-fm-stack.md`](gemma4-ple-static-input-fm-stack.md) — Gemma 4's
  **per-layer-embedding table** as a static graph input, loaded behind FoundationModels.
- [`timesfm-port.md`](timesfm-port.md) — TimesFM 2.5: the zoo's first **time-series forecasting**
  foundation model (stateless graph + host RevIN DSP).
- [`esam3-port.md`](esam3-port.md) — EfficientSAM3: a **dropped** port (device-verified but redundant
  vs the official SAM 3) — kept for what transferred.
- [`shieldstral-port.md`](shieldstral-port.md) — Shieldstral-1.0-3B: why a model that answers
  with **one token** should not ship as a decoder (bake the tail, keep two rows of the head), how
  to port a checkpoint your **export venv cannot load** — build the oracle in the venv that
  supports it natively, then *measure* the substitute (`ministral3` = Mistral + YARN, cos
  1.000000) instead of forcing a config — and the two numbers that invert decode-loop intuition:
  quantization buys **size, not speed**, and a verdict costs the **grid, not the text** — the same
  |ΔP| on an M4 Max and an iPhone 17 Pro, to four decimals.
- [`north-micro-vision-port.md`](north-micro-vision-port.md) — Cohere's `cohere_compass`: how to
  notice in a minute that a new checkpoint's **vision tower is one you already have** (load it
  strict=False and print missing/unexpected), the four decoder details that run fine when done
  wrong (parallel block, mean-subtracting LayerNorm, a quarter of the layers with **no**
  positional encoding, logit_scale), and the iOS load wall measured 2.39 GiB further out than
  this repo believed.
- [`lfm2.5-vl-port.md`](lfm2.5-vl-port.md) — LFM2.5-VL (450M + 3B): **SigLIP2 NaFlex** (a Linear patch embedding
  over host-flattened patches, a 16×16 position grid resized per image), what baking a square grid
  costs a NaFlex model, the antialiased resize a GPU "bilinear" is not — and why judging a
  compressed VLM needs an **fp16 baseline**, because greedy decoding turns any near-tie into a
  different tail — and how the same family flips verdict on int4 between two sizes (450M 0/9,
  3B 7/9), plus the two per-checkpoint host details (resample filter, and a tokenizer that
  does not prepend BOS) that read as a broken port.
- [`lfm2.5-2.6b-port.md`](lfm2.5-2.6b-port.md) — LFM2.5-2.6B, where the port was a config change and
  the work was surviving a **transformers-v5-era checkpoint**: a RoPE key that moved, a tokenizer
  class that does not exist yet, and a 4.x modeling file that silently ignores a config flag and
  would certify a wrong oracle as PASS.
- [`qwen3.8-27b-port.md`](qwen3.8-27b-port.md) — Qwen3.8-27B, a **generation bump that was a
  verified weight swap** (config + weight map byte-identical to 3.6-27B), and what a
  **release-day download** actually costs: HF CDN at ~0.15 MB/s per connection with a ~2.7 MiB/s
  per-IP ceiling, XET stalling live, and the fix — sha256-verified ModelScope through aria2c at
  ~20 MB/s, seeded into the hub cache by LFS oid. Phase 2 adds the **vision path** and four
  durable lessons: a bf16 full-model oracle is not a valid tower target (gate vs an fp32 tower
  ref); the chunked GDN scan's fp16 doubling-inverse overflows content-dependently at S=32 — it
  passed the oracle suite then collapsed on the next real photo, so ship the chunk the dtype
  can prove (S=16) and gate chunked-vs-S=1 token equality; multifunction bundles can assert in MPSGraph's ANE region
  pass under JIT (AOT h16c is the python-driver path); capture the oracle's mRoPE positions with
  a hook instead of re-deriving them.

## Image generation & editing (diffusion)
- [`zimage-port.md`](zimage-port.md) — Z-Image-Turbo, a 6B Single-Stream DiT text-to-image — and why
  it ships Mac-only.
- [`glm-image-port.md`](glm-image-port.md) — GLM-Image: the zoo's first **AR + diffusion hybrid**
  (9B GLM-4 AR writes visual prior tokens → 7B flow-matching DiT renders).
- [`flux2-in-context-editing.md`](flux2-in-context-editing.md) — FLUX.2 [klein] **instruction
  editing + multi-reference composition** with no separate editing model and no ControlNet.

## Audio (the new modality — model-specific port notes)
- [`qwen2.5-omni-audio-understanding.md`](qwen2.5-omni-audio-understanding.md) — Qwen2.5-Omni's
  **Thinker** as on-device **audio understanding** (describes sounds, *not* ASR): de-dynamizing the
  Whisper-style encoder (ragged `cu_seqlens` chunk attention → one batched fixed attention, bit-exact),
  TMRoPE collapsing to 1-D, a bit-exact vDSP mel (DFT-as-matmul), audio embeds on a static buffer, and
  the two payoffs — **AOT clean-mmap weights dodge the iOS jetsam dirty limit** (4.5 GB decoder, 5930 MB
  free), and a **fixed-shape encoder runs on the ANE** (0.99 cos → byte-identical text) where the
  dynamic decoder can't.
- [`whisper-asr-fixed-decode.md`](whisper-asr-fixed-decode.md) — Whisper large-v3-turbo as on-device
  **ASR** (the zoo's first official speech-to-text): why the official recipe's single-step `[1,1]`
  decoder graph can't transcribe, why a dynamic-length re-export recompiles every step (~15 s/token),
  and the fix — a **fixed 128-token decoder window** (pad, read logits at the real last position;
  causal attention ignores the padding; constant shape compiles once → 0.18 s/token, token-exact).
  Plus the Swift log-mel frontend (n_fft 400 = DFT-as-matmul) and single-`main`→GPU routing.
- [`kokoro-tts.md`](kokoro-tts.md) — Kokoro-82M (StyleTTS2 + iSTFTNet), the zoo's first **text-to-speech**:
  3 bundles cut at the one data-dependent length + host DSP, the `weight_norm`-random-init bug
  (non-determinism → suspect weight loading), variable length via **masked-unrolled bi-LSTM +
  frame-masked InstanceNorm + host STFT**, the Core AI op rewrites (ConvTranspose1d/iSTFT → zero-insert
  + conv1d), and why an unrolled-LSTM model runs **faster on the CPU than the GPU**.
- [`adcsr-super-resolution.md`](adcsr-super-resolution.md) — AdcSR, the zoo's first **super-resolution**
  (one-step diffusion-GAN, pruned SD-2.1 + half VAE decoder): why fp16 NaNs (SD-2.1 attention overflow,
  `upcast_attention` ignored by the SDPA processor) so **fp32 ships**, the **global** (not per-tile)
  color-match, host-side tiling with an input-size cap, and the CoreGraphics `bytesPerRow`/no-y-flip traps.
- [`sam3-promptable-segmentation.md`](sam3-promptable-segmentation.md) — SAM 3 as on-device **text-prompt
  segmentation** (the zoo's first official segmenter): the segmenter *bundle* format + one-call official
  `CoreAIImageSegmenter` runtime, float16≈float32 fidelity (Δ≤1e-4) verified with the `image-segmenter`
  CLI, redistributing a **gated** model under the SAM License (ship the LICENSE), and the platform box-origin
  trap when compositing masks.
- [`depth-anything-3-monocular-depth.md`](depth-anything-3-monocular-depth.md) — Depth Anything 3, the
  zoo's first **depth** model (DINOv2 ViT + DPT head, single static square graph): why the input is
  **raw [0,1]** (ImageNet norm folded in-graph) and the **double-normalization trap** that faked a
  cos-0.9 "engine bug" / "noise" / "non-square bug" for a day; engine **cos 1.000000 at any shape**;
  the squish + resize-back contract and why r≈0.98 vs official **is** faithful (= the model's own
  504-vs-518 variance); the RoPE-const / cache-free / pos-embed-dtype export patches; fp16 via
  `.half()` not autocast. Sample: [`scripts/depth_anything_3_sample.py`](scripts/depth_anything_3_sample.py).

- [`voxcpm-tts.md`](voxcpm-tts.md) — VoxCPM-0.5B, the zoo's first **diffusion TTS**: a family of
  graphs (LM + diffusion + VAE + vocoder) resolved as one catalog model.
- [`pocket-tts-port.md`](pocket-tts-port.md) — pocket-tts (Kyutai), the zoo's first **Kyutai model**
  and first **Mimi conversion**: why a fixed-capacity KV cache must be **zero-initialised, never
  NaN-initialised** (`0 * NaN = NaN` survives a masked SDPA, and a fixed graph cannot slice the
  unwritten tail off the way upstream does); `preferredComputeUnitKind: .cpu` as a **partitioning
  hazard rather than a compute-unit choice** (`.cpu` and `.gpu` declare the same allowed set and
  differ only in preference — verified on two unrelated models, blast radius from rounding-scale
  to anti-correlated); deriving `S_MAX` from the model's own generation bound instead of the
  fixtures; windowed prefill as a **second escape** from the large-query SDPA lowering crash; and
  gating TTS with an **ASR round trip** because tensor cosine passes on unintelligible audio.
- [`chatterbox-port.md`](chatterbox-port.md) — Chatterbox: the zoo's first **zero-shot voice-cloning
  TTS** and first multi-network port.
- [`music-generation-stable-audio.md`](music-generation-stable-audio.md) — Stable Audio Open:
  text→music latent diffusion (T5 cond + DiT + VAE) on-device.
- [`sortformer-speaker-diarization.md`](sortformer-speaker-diarization.md) — streaming **speaker
  diarization**: export only the neural core, port the streaming loop + AOSC speaker cache to the
  host.
- [`vibevoice-multispeaker-tts.md`](vibevoice-multispeaker-tts.md) — VibeVoice, the zoo's first
  **multi-speaker / dialogue TTS**: dual-LM **next-token diffusion**, why the speech feedback loop
  forces fp16, and why the `expectFrequentReshapes` hint must be off on iOS.
- [`melband-source-separation.md`](melband-source-separation.md) — Mel-Band RoFormer, the zoo's first
  **source separation**: complex → real arithmetic, and folding STFT/iSTFT into the graph as constant
  DFT matmuls so the on-device host needs no FFT.

Primary official sources behind these notes: the open repos (`coreai-torch`, `coreai-optimization`,
`coreai-models` incl. its agent skills), the WWDC26 talks **324 / 325 / 326 / 330** (verbatim transcripts in
`ondevice/_wwdc{324,325,326,330}_transcript.txt`), and `developer.apple.com/core-ai/`. Verified against
Hugging Face references (convert + numeric parity); on-device iOS 27 notes marked where still in bring-up.
