# Qwen-Image-2.1 → Core AI (Mac GPU) — STATE / handoff

**Model:** [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (rev `790c926`,
33.1 GB bf16). Unified T2I + editing, **RGBA-native** output. Qwen3-VL-8B text encoder
(17.5 GB, vision tower unused for T2I) → **7B single-stream block-causal DiT** (32 blocks,
14.2 GB) → 64-channel 3D-causal VAE (1.35 GB, 4-channel RGBA out). 40-step
FlowMatchEuler, no CFG by default. **Licence = Qwen RESEARCH LICENSE (2026-09-20):
non-commercial only** (§2), redistribution allowed with the Agreement copy + change notice +
"Notice" file text (§3), "Built with Qwen" (§4b), "Qwen" not the primary name (§4c).
Started 2026-09-25. Working tree = this dir; bundles → `_paths.exports_dir()`.

## Gates answered first (AGENTS.md §"Decide whether to port at all")

- **GAP ✓** — Apple's stock `coreai.diffusion.export` handles `sd` / `sd3` / `flux2` only
  (`coreai_models/diffusion/models.py`, read 2026-09-25). No Qwen-Image type.
- **EDGE (one sentence)** — the Core AI port is the only way to run Qwen-Image-2.1 from a
  Swift app on Apple's own runtime, at full bf16 fidelity (the two MLX packs on the Hub —
  ddalcu/mlx-serve 8-bit, JoyFusionAI/mflux 8-bit — are quantized, Python/zig-hosted, and
  one is unreleased), and it puts the first RGBA-native T2I model into the zoo's
  CoreAIImageGen app next to FLUX.2 klein and Z-Image. **Speed vs MLX is a measurement, not a
  claim**: MLX 8-bit numbers on the Hub are M1 Max 768² 6.7 s/step (mflux) and M1 Pro 1024²
  23 s/step (mlx-serve); the Z-Image precedent (6B DiT bf16, 4096 tokens 4.4 s/fwd on M4 Max)
  says a bf16 Core AI DiT lands in the same class. If the measured 1024² s/step is worse than
  the MLX pack on the same Mac, that is a "worse-MLX" flag for the user, not a reason to
  silently stop.

## Corrections to the launch brief (found by reading the code)

1. **Latents are NOT 2×2-packed.** `QwenImage21Pipeline._pack_latents` is a plain spatial
   flatten: one DiT token per **16×16 px** tile (256² → 16×16 = **256 tokens**, 512² → 1024,
   1024² → 4096, 2048² default → 16384). The "2×2 group" in the transformer is the VLM
   image-slot expansion (`_IMG_TOKENS_PER_SLOT = 4`: every `<|image_pad|>` slot stands for
   4 latent tokens) and is invisible at the graph boundary for T2I.
2. `hidden_states[-1]` is the **last decoder layer output before the final RMSNorm** — the
   pipeline neutralises the norm with a forward hook because transformers ≥5.0 ties
   `hidden_states[-1]` to the normalised `last_hidden_state`. The export takes the residual
   stream after layer 36 directly.
3. `coreai-opt 0.2.1` pins `safetensors<=0.7.0`; diffusers main needs `safetensors>=0.8.0`.
   **They cannot share a venv.** Split: `.venv-qi21` (diffusers main @4295ee3 + transformers
   5.17 + coreai-torch, for the oracle and the VAE export) / base `.venv` (coreai-torch +
   coreai-opt, for the re-authored DiT + encoder exports and the quantization stage). The DiT
   and the text encoder are therefore **re-authored in plain torch** (key-compatible with the
   checkpoints), not wrappers over diffusers/transformers modules as Z-Image was.

## Architecture → graph mapping (T2I, batch 1)

| Piece | Reference | Graph contract |
| --- | --- | --- |
| Text encoder | Qwen3-VL-8B text stack: 36 L / h 4096 / GQA 32:8 / hd 128 / MLP 12288 / q_norm+k_norm / RoPE θ 5e6 (mrope-interleaved collapses to plain RoPE for text-only) | `qi21_text.py` re-author. `input_ids [1,L]` (**dynamic L 16..512** as shipped; causal ⇒ valid outputs pad-content-independent) → `hidden [1,L,4096]` (no final norm). Host: t2i template → tokenize → slice `[drop_idx:Lv]`. bf16 weights, **fp32 compute** (w16a32, see the numerics decision), fp32 boundary. |
| DiT | 32 single-stream blocks, ONE shared modulation Linear (4096→16384, rows t and t=0), ZeroCenterRMSNorm text proj, SwiGLU ×3, 3-axis interleaved RoPE (16/56/56), block-causal attention (text causal, image bidirectional, image sees all text) | `qi21_dit.py` re-author. `img_tokens [1,N,64]`, `txt_feats [1,L,4096]`, `timestep [1]`, `txt_cos/sin [1,L,64]`, `img_cos/sin [1,N,64]` → `vel [1,N,64]`. Attention = two SDPA calls per block (text→text causal, image→all) = exactly the reference's segment decomposition; no mask input. Prefix KV cache dropped (full recompute each step, mathematically identical). bf16, fp32 boundary, both axes dynamic. |
| VAE dec | Wan-style 3D causal VAE, 1 frame, 64 ch → 4 ch RGBA, `latents*std+mean` un-normalisation | diffusers-main class, wrapper bakes un-normalisation, fp32, per size. |
| Sampler | FlowMatchEuler, sigmas linspace(1, 1/40, 40), dynamic exponential shift μ(N), shift_terminal 0.02 | host (recorded oracle sigmas first; numpy twin gated after). |

## Log


- 13:10 weights download started by the launch session (hf_transfer, ~6 MB/s) → **done 14:54**.
- 13:4x–14:52 **Opus r1** (`coreai-d8`, session e3d0f8da): parity_dit_torch (random 2-layer, fp32)
  max|Δ| 7.2e-7 / corr 1.0 / 4 negative controls red / RoPE tables bit-exact / KV on-off exact;
  fixed `TimeTextEmbed.freqs` (buffer → fp32 attribute; `.to(bf16)` had rounded it, sinusoid
  corr 0.958); 2-layer probes exported + engine-checked; **full 32-layer bf16 export 14:51–14:52
  (14.23 GB, 31 s convert, peak RAM 14.9 GB)**.
  Remaining r1 stages (real-weight fp32 parity, engine parity, bench) were run by the supervisor (15:0x).
- 14:54 oracle 256² (2 min), 15:03 oracle 512² (6 min). Both reference images are clean apples.
- 13:2x env: `.venv-qi21` = uv, Python 3.11, torch 2.9.0, transformers 5.17.0, diffusers
  0.41.0.dev0 (main 4295ee3, tarball install — a git+ URL blocked on a uv git lock held by another process), coreai-torch 0.4.1 + coreai-core 1.0.0b2. coreai-opt excluded (see above).
- 13:3x scripts written: `capture_oracle.py` (fp32 CPU reference, all three boundaries).
- 13:5x `capture_oracle.py` smoke-tested end to end on a tiny random pipeline (diffusers
  test-suite configs + the real processor, `--snapshot`, 64², 3 steps → `oracle/tiny/`): all
  asserts hold (prompt_embeds == hidden[drop:], img_mask layout `[L×False | N/4×True]`, RGBA
  out). **KV cache on vs off gives identical reference tensors** (`oracle/tiny_nocache/`:
  vel max|Δ| ≤ 6e-7 on values ~1.5, fp32 rounding only) = the exported graph's full-prefix
  recompute is the reference computation, not an approximation.

## Text-encoder numerics decision (supervisor, 15:2x, on Opus r2's isolation)

The fp32 re-author is **bit-exact** with the oracle (32/32 tokens, max|Δ| 0). In plain bf16 the
first user-turn token (index 14, the first token the DiT reads) drops to per-token corr 0.968:
its residual grows to |h| ≈ 9,100 through layers 17–34 and is cancelled to ≈100 in the last two
layers, and bf16 cannot hold that cancellation (HF's own bf16 model: 0.9225 on the same token;
fp32 residual + norm statistics with bf16 matmuls: 0.9996; attention-only or MLP-only fp32 does
not help). Downstream, the fp32 DiT's velocity corr stays ≥ 0.99998 either way. **Decision:
GO for the fp32-residual variant (`_r32`) as the ship candidate; the plain bf16 graph is kept
as the measured control; the bar (every token corr ≥ 0.999) is unchanged.**

## Text-encoder host facts (processor only, `.venv-qi21`, 2026-09-25 13:4x)

- `_drop_idx` = **14** (the tokenised system message
  `<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n` =
  `[151644, 8948, 198, 1092, 30782, 408, 323, 23643, 279, 3897, 9934, 13, 151645, 198]`).
- t2i template lengths: apple prompt Lfull 32 → **L 18**; the empty prompt (" ") Lfull 23 → L 9;
  the README RGBA sticker prompt Lfull 47 → L 33. Tail is always
  `<|im_end|>\n<|im_start|>assistant\n` = `[151645, 198, 151644, 77091, 198]`.
- Processor = `Qwen2Tokenizer`, pad id 151643, `<|image_pad|>` 151655. It also emits
  `mm_token_type_ids` (all 0 for text) which the pipeline forwards to the encoder — text-only,
  so no effect on the hidden states (verified by the oracle's equality assert).
- `Qwen3VLProcessor` needs torchvision (installed 0.24.0 into `.venv-qi21`).

## Numbers

### Oracle (fp32 CPU, diffusers main 4295ee3, seed 1234, apple prompt, 40 steps, no CFG)

| size | N | L | s/forward (fp32 CPU, M4 Max, contended) | image | dir |
| --- | --- | --- | --- | --- | --- |
| 256² | 256 | 18 | 2.37 | RGBA, clean red apple on wood (checked by eye) | `oracle/256/` (14:56) |

| 512² | 1024 | 18 | 8.97 | RGBA, clean apple with reflection (checked by eye) | `oracle/512/` (15:03) |

Encoder hidden `|h|` max 4596 (Qwen outlier dims — bf16 range is fine, fp16 would not be).
`mu` = 0.5 at N=256 (= base_shift) and 0.539 at 512²; the exponential shift `exp(mu)` and the
terminal stretch apply at both sizes (256²: σ₁ = 0.98436, not the linspace 0.975 — corrected by
Opus r3 after this note first said "unshifted").

### DiT gates (supervisor re-ran every script, 2026-09-25 15:0x–15:2x, 256² oracle, 40 steps)

| gate | script | result | bar |
| --- | --- | --- | --- |
| re-author vs diffusers (random 2-layer, fp32) | `parity_dit_torch.py` | max\|Δ\| 7.2e-7, corr 1.0; controls a/b/c/d red; RoPE 4 sets max\|Δ\| 0; KV on/off exact | ≤1e-4 / ≥0.999999 / red |
| re-author, real weights, fp32 CPU, teacher-forced | `parity_dit_oracle.py --steps all --controls 0,20,39` | **40/40 corr 1.000000000**, max\|Δ\|/max\|ref\| ≤ 2.4e-4 (step 14; others ~1e-6); controls red at corr 0.897 / 0.837 / 0.9969 | ≥0.999999 / ≤1e-3 |
| Core AI bf16 full, `cpu_only()` | `engine_parity_dit.py --cpu-only --steps 0,20,39` | corr 0.99995 / 0.99998 / 0.99983, NaN 0 (4.2–4.9 s/fwd) | ≥0.999 |
| Core AI bf16 full, JIT `default()` and `preferred gpu` | same | **crash**: `ANERegion.mm:414 failed assertion … Code=-19` (2×) | — |
| Core AI bf16 full, AOT h16c gpu (plain) | same on `.aimodelc` | **same ANE crash** | — |
| **Core AI bf16 full, AOT h16c gpu `--expect-frequent-reshapes`** | same on `.aimodelc` (27 GB) | **40/40 PASS**, corr 0.999978 → 0.999854 (min at step 38), \|Δ\|/\|ref\| 0.4–1.7 %, NaN 0, **0.34 s/fwd** at N=256 (contended), first call 35.8 s | ≥0.999 / NaN 0 |

### DiT s/forward (AOT efr bundle, `default()`, GPU lock held, M4 Max, loadavg ~10, `bench_dit.py`, 15:15)

| size | N | L | first call | warm median ×5 | 40 steps (no CFG) |
| --- | --- | --- | --- | --- | --- |
| 256² | 256 | 40 | 2.94 s | **0.343 s** | 13.7 s |
| 512² | 1024 | 40 | 1.14 s | **1.104 s** | 44 s |
| 1024² | 4096 | 40 | 4.77 s | **4.757 s** | 190 s |

Same class as Z-Image-Turbo bf16 on this Mac (6B: 1.12 s @512², 4.36 s @1024²). The MLX packs'
public numbers are on M1-class Macs (mflux 8-bit 768² 6.7 s/step on M1 Max; mlx-serve 8-bit
1024² 23 s/step on M1 Pro) — **not comparable until measured on this machine with the same
protocol**; no claim is made either way.

### Text encoder (Opus r2 report 15:49; supervisor re-run of the engine gate in progress)

| gate | result |
| --- | --- |
| fp32 re-author vs oracle (32 tokens) | **bit-exact** (max\|Δ\| 0); controls a/b/c red (0.356 / 0.988 / 0.518); mrope collapse verified against HF (explicit 3-axis arange == unspecified, bit-exact) |
| host tokenizer (`qi21_tokenize.py`, `tokenizers` only) | 3 prompts (Lfull 32 / 23 / 47) identical to the processor, drop_idx 14 |
| engine, plain bf16 full (AOT efr) | token 14 corr 0.976 (others ≥ 0.99965) — **bar miss** |
| engine, r32 (fp32 residual + norm stats) full | oracle L=32: all tokens ≥ 0.999809, downstream DiT vel corr 1.000000; **but** L=64 pad → token 14 0.9974; empty prompt token 20 → 0.9938 (L=23) / 0.664 (L ≥ 48). Downstream DiT vel corr ≥ 0.99992 in every case |
| s/call (L=32 / 128) | bf16 0.063 / 0.148 s; r32 0.062 / 0.149 s; first call 15–16 s |
| w16a32 (bf16 weights, fp32 compute), 8-layer probe | rel err 4.1e-6 (r32 6.4e-3); .aimodelc 4.04 GiB (weights stay bf16 through AOT); s/call 2.9× / 1.8× of r32 |

| **w16a32 full (r2b, 16:03)** | oracle L=32 min token corr **0.999999999**; sweep 3 prompts × 7 lengths **21/21 ≥ 0.99999998**; pad content independence bit-exact (L=32 vs padded L=64 differ by rel 4.4e-6 = GPU path changes with length, accepted); downstream DiT vel corr 1.000000 (min over 3 prompts × 2 lengths 0.9999999999); s/call 0.181 / 0.271 s (L=32 / 128), first call 0.37 s, load 19.6 s; .aimodel 14.10 GiB, AOT efr .aimodelc **14.10 GiB** (no fp32 expansion); export convert 33 s + AOT 16 s, peak RSS 35.5 GB |

**Decision (supervisor, 15:5x): ship candidate = w16a32 full (r2b done 16:03, PASS).** The two
fragile tokens are the `<|im_start|>` of the user line (token 14: residual ≈ 9,100 cancelled to
≈ 100 in the last two layers) and, on the empty prompt, the `<|im_start|>` of the assistant line
(token 20: no massive activation, peak < 1000 — a different mechanism, per Opus r2's diagnosis);
no partial-fp32 variant survives all prompts and lengths, only full fp32 compute does. The encoder runs once per image, so 0.2–0.3 s/call is acceptable; the bundle stays bf16-sized.

### VAE + pipeline twin (Opus r3, 16:04–16:40; images checked by the supervisor)

| gate | result |
| --- | --- |
| VAE fp32 graph vs oracle (GPU, AOT efr, 4 ch RGBA) | 256²: corr 1.0000000, max\|Δ\| 1.35e-5; 512²: 1.29e-5; 1024² vs torch on a synthetic latent 2.33e-5. Per-size bundles 0.94 GiB each. `nearest-exact` lowers to gather_nd (no patch needed); 1-frame decode is bit-identical with/without feat_cache, `first_chunk=True` required |
| scheduler twin (`qi21_sched.py`) | sigmas / timesteps / 40 Euler steps bit-exact with the oracle at 256² and 512² |
| e2e 256² (3 bundles, prompt → RGBA) | 46.72 dB (46.7–50.4 over 5 perturbed runs), alpha max\|Δ\| 1/255, final latent corr 0.999987; encoder 1.15 s + DiT 40 steps 45 s (first 32 s, then 0.34 s/step) + VAE 0.16 s |
| e2e 512² | 35.49 dB via the encoder bundle; **spread over 6 runs 23.6–43.4 dB (4 ≥ 30)**; the two 24 dB runs are the same apple with one extra leaf on the stem — a semantic fork, not noise; alpha 2/255; teacher-forced DiT at 512² 40/40 corr ≥ 0.999844; bit-reproducible on re-run |
| RGBA sticker 512² (seed 42) | alpha min 0 / mean 126 / background corners 0.96/255; torch-decoded vs engine VAE max\|Δ\| 1.8e-5 |
| sensitivity of the fp32 reference (supervisor, `perturb_ref.py`) | prompt_embeds × (1 + 1e-5·N(0,1)) → **82 dB** at both 256² and 512², latent rel err ≤ 8e-6 at step 38: the fork is not an intrinsic instability of the fp32 model; it is the bf16 DiT's per-step error (0.4–1.8 %) crossing a decision boundary at 512² |

### The bf16 band of the model itself (supervisor, `ref_bf16.py`, diffusers main in bf16 on MPS, same seed/noise/prompt, 16:4x)

| size | bf16 diffusers (MPS) vs fp32 oracle | our bf16 Core AI engine vs fp32 oracle |
| --- | --- | --- |
| 256² | 43.47 dB, final latent corr 0.999947, latent rel err 1.0e-2 | 46.72 dB, 0.999987, 5e-3 |
| 512² | 33.19 dB, 0.999053, 4.3e-2 | 35.49 dB (spread 23.6–43.4 over perturbed runs), 0.999559 |

Speed of the same bf16 diffusers run on MPS: 0.47 s/step @256², 1.25 s/step @512² (incl.
Python overhead) vs the Core AI engine's 0.34 / 1.10 s (GPU lock). **Decision (16:5x): ship the
bf16 DiT.** The port sits inside — slightly above — the band the official bf16 pipeline itself
occupies against fp32; the 512² "leaf" fork is a property of bf16 inference of this model, not of
the port. An fp32-compute DiT (`w16a32`, 2–3× slower) stays a recipe option, not a shipped bundle.

### Quantization (Opus r4, interim 17:1x; supervisor decision)

- **DiT int8lin (per_block 32, bf16 compute): not shippable.** The efr AOT (the only GPU path)
  fails in the MLIR pass manager ("operand #0 does not dominate this use", at the first op that
  touches a graph input; same with bf16 I/O and with `--preferred-compute none`). Plain AOT
  compiles (2-layer probe 0.57 GiB, corr 0.999988 on GPU after an `ANECCompile() FAILED`
  fallback per new shape) but plain AOT of the full bf16 graph already died on the ANE region.
  Not pursued further: zimage measured weight-only int8 at 2.4–2.7× slower on a compute-bound DiT.
- **Encoder int8 (w8a32): not shippable.** Torch-side int8 weights alone already break the
  sink tokens (apple tok 14 corr 0.9946, empty-prompt tok 20 0.656 — the same cancellation that
  needed fp32 compute is sensitive to weight rounding), and the efr AOT folds the dequant into
  fp32 constants: `.aimodel` 8.44 GiB → `.aimodelc` **34.32 GiB** (larger than bf16's 14.1).
  int4 skipped. (coreai-opt 0.2.1 also cannot quantize the w16 custom Linear — `Tensor.float`
  has no aten mapping — so the fp32 `nn.Linear` build was used.)
- **Ship = bf16/fp32 only**: DiT bf16 14.23 GB (13.25 GiB) + encoder w16a32 14.10 GiB + VAE fp32
  0.94 GiB × 3. (r4 final 17:13: w8a32 engine gate min tok corr 0.9946, sweep 21/21 fail,
  downstream DiT still ≥ 0.99994 and e2e 256² 48.1 dB — the error is entirely the int8 weights;
  DiT int8lin efr AOT fails in `MPSMemrefAllocFusion` / "operand #0 does not dominate this use".)

**Runtime finding (durable):** on 26A428 the Python runtime's MPSGraph delegate forms an ANE
region inside this 32-layer bf16 graph and the ANE inference fails (`Code=-19`) — with JIT
`default()`, with `preferred gpu` (the allowed set cannot be narrowed from Python), and with a
plain h16c AOT. The 2-layer probe does not trigger it. `cpu_only()` is exact. The only working
GPU path is **AOT with `--expect-frequent-reshapes`** (same recipe as the 27B
`ANERegionFormationPass` wall in `~/code/coreai/SPEC38_VERIFY_LEVER_STATE.md`): 2 min 9 s to
compile, 27 GB (≈1.9× the bundle), delegates = MPSGraph only. Whether the Swift host
(`GraphModel(computeUnits: .gpu)`) hits the same region is an open question for the app round.
