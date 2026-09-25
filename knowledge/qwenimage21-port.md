# Qwen-Image-2.1 port — a 7B block-causal DiT and an RGBA VAE on the Mac GPU

[`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) (revision `790c926`, 33.1 GB
bf16, Qwen Research License: non-commercial) is a unified text-to-image and editing model with
RGBA output. A Qwen3-VL-8B text encoder (17.5 GB; the vision tower is unused for text-to-image)
conditions a 7B single-stream DiT (32 blocks, 14.2 GB). A 3D-causal VAE with 64 latent channels
decodes to four channels, RGBA. The pipeline default is 40 FlowMatch Euler steps without CFG.

Staged as [mlboydaisuke/RGBA-Image-2.1-CoreAI](https://huggingface.co/mlboydaisuke/RGBA-Image-2.1-CoreAI);
the licence keeps "Qwen" out of the primary name. Card: [`models/rgba-image-2.1/`](../models/rgba-image-2.1/README.md).
Scripts and the working state: `conversion/qwenimage21/` (`QWENIMAGE21_STATE.md`).

## Architecture → export mapping (text-to-image, batch 1)

| Piece | Reference | Graph |
| --- | --- | --- |
| Text encoder | Qwen3-VL-8B text stack: 36 layers, hidden 4096, GQA 32:8, head dim 128, MLP 12288, q_norm + k_norm, RoPE θ 5e6. With text only, the interleaved M-RoPE collapses to plain 1D RoPE | `qi21_text.py`, re-authored from the checkpoint. `input_ids [1,L]` int32 → `hidden [1,L,4096]`, the last layer's residual stream with no final norm; `embed_tokens` inside; dynamic L 16..512. bf16 weights, **fp32 compute**, fp32 boundary. The host slices `[drop_idx:]`. |
| DiT | 32 single-stream blocks; ONE modulation Linear shared by all blocks (4096 → 16384; rows for the sampled t and for t = 0); ZeroCenterRMSNorm text projection; SwiGLU ×3; 3-axis interleaved RoPE (16/56/56); block-causal attention (text causal, image bidirectional, image sees all text) | `qi21_dit.py`, re-authored with the checkpoint's parameter names. `img_tokens [1,N,64]`, `txt_feats [1,L,4096]`, `timestep [1]`, `txt_cos/sin [1,L,64]`, `img_cos/sin [1,N,64]` → `vel [1,N,64]`. Attention is two SDPA calls per block (text→text causal, image→all), the reference's own segment decomposition, so no mask input. The prefix KV cache is dropped: full recompute every step, mathematically identical. bf16, fp32 boundary, both axes dynamic (L 8..512, N 64..4096). |
| VAE decoder | Wan-style 3D-causal VAE, one frame, 64 ch → 4 ch RGBA, `latents * std + mean` un-normalisation | `qi21_vae.py`: the diffusers-main class in a wrapper that unpacks the tokens and un-normalises inside the graph. fp32, one graph per size (256 / 512 / 1024). |
| Sampler | FlowMatch Euler; sigmas `linspace(1, 1/40, 40)`, exponential shift with `μ(N)`, `shift_terminal` 0.02 | host: `qi21_sched.py` (numpy) |

## Three corrections to the first reading

1. **Latents are not 2×2-packed.** `QwenImage21Pipeline._pack_latents` is a plain spatial flatten:
   one DiT token per 16×16 px tile (256² → 256 tokens, 512² → 1024, 1024² → 4096, the 2048²
   default → 16384). The "2×2 group" in the transformer is the VLM image-slot expansion (every
   `<|image_pad|>` slot stands for 4 latent tokens), invisible at the graph boundary for
   text-to-image.
2. **`hidden_states[-1]` is the last decoder layer's output BEFORE the final RMSNorm.** The pipeline
   neutralises the norm with a forward hook, because transformers ≥ 5.0 ties `hidden_states[-1]` to
   the normalised `last_hidden_state`. The export takes the residual stream after layer 36 directly.
3. **Two venvs.** coreai-opt 0.2.1 pins `safetensors<=0.7.0`; diffusers main needs `>=0.8.0`. The
   oracle and the VAE export run in a venv with diffusers main (4295ee3) and transformers 5.17; the
   DiT and encoder exports and the quantization stage run in the base venv. That is why the DiT and
   the text encoder are re-authored in plain torch, key-compatible with the checkpoints, instead of
   wrapping the diffusers / transformers modules as the Z-Image port did.

## The Neural Engine region, and the one GPU path that runs

On 26A428 the Python runtime's MPSGraph delegate forms a Neural Engine region inside the 32-layer
bf16 DiT, and the ANE inference fails:

```
ANERegion.mm:414 failed assertion … Code=-19
```

It fails with JIT `default()`, with `preferred gpu` (the allowed set cannot be narrowed from
Python), and with a plain h16c AOT compile. The 2-layer probe does not trigger it, and `cpu_only()`
is exact. The GPU path that runs is **AOT with `--expect-frequent-reshapes`**, the same fix as the
Qwen3.8-27B multifunction bundle's `ANERegionFormationPass` assert
([`qwen3.8-27b-port.md`](qwen3.8-27b-port.md)):

```
xcrun coreai-build compile <bundle>.aimodel --output <dir> --platform macOS --architecture h16c \
    --preferred-compute gpu --expect-frequent-reshapes     # -> <dir>/<bundle>.h16c.aimodelc
```

The DiT takes 2 min 9 s to compile, the result is 27 GB (≈ 1.9× the bundle), and its delegates are
MPSGraph only. Load it with `SpecializationOptions.default()`. Every GPU gate of this port ran on
efr-compiled bundles. Whether the Swift host (`GraphModel(computeUnits: .gpu)`) hits the same
region is open (Follow-ups).

Lesson: a probe that passes does not clear the full-depth graph. Run the full graph on the path the
app will use before building on it.

## Text-encoder numerics: the `<|im_start|>` tokens need fp32 compute

The fp32 re-author is **bit-exact** with the oracle (32/32 tokens, max|Δ| 0). In plain bf16 the first
user-turn token (index 14, the `<|im_start|>` that opens the user line, the first token the DiT
reads) drops to per-token corr 0.968. Its residual grows to |h| ≈ 9,100 through layers 17–34 and is
cancelled to ≈ 100 in the last two layers; bf16 cannot hold that cancellation. transformers' own
bf16 model scores 0.9225 on the same token. fp32 residual + norm statistics with bf16 matmuls
reaches 0.9996 on CPU; attention-only or MLP-only fp32 does not help.

On the engine the same picture holds, and it moves with length and prompt:

| encoder variant | result |
| --- | --- |
| plain bf16 (AOT efr) | token 14 corr 0.976 (others ≥ 0.99965): misses the every-token 0.999 bar |
| r32: fp32 residual + norm statistics | oracle L = 32: all tokens ≥ 0.999809. But L = 64 padding → token 14 0.9974; the empty prompt's token 20 → 0.9938 (L = 23) and 0.664 (L ≥ 48) |
| **w16a32: bf16 weights stored, fp32 compute** | oracle L = 32 min token corr **0.999999999**; 3 prompts × 7 lengths: 21/21 ≥ 0.99999998; downstream DiT vel corr 1.000000 |

No partial-fp32 variant survives all prompts and lengths; only full fp32 compute does. The two
failing tokens are the `<|im_start|>` of the user line (token 14, the massive activation above) and,
on the empty prompt, the one of the assistant line (token 20). The DiT hardly notices either way:
its velocity corr stays ≥ 0.99998 even with the plain bf16 encoder. A per-token gate on the encoder
is what catches it.

w16a32 costs compute, not bytes. The converter keeps the bf16 constants and casts at run time, so
the `.aimodel` is 14.10 GiB and the efr `.aimodelc` is also 14.10 GiB (no fp32 expansion). Per call:
0.181 / 0.271 s at L = 32 / 128, against 0.062 / 0.149 s for r32. The encoder runs once per image.

### Quantization: no-go on both graphs

- **Encoder int8 (w8a32, per_block 32).** The int8 weights alone break the same tokens on the CPU,
  before any export: apple token 14 corr 0.9946, empty-prompt token 20 0.656. The cancellation
  that needed fp32 compute is also sensitive to weight rounding. The engine gate: min token corr
  0.9946, sweep 21/21 fail; downstream DiT still ≥ 0.99994 and e2e 256² 48.1 dB, so the whole error
  is the int8 weights. The efr AOT folds the dequant into fp32 constants: `.aimodel` 8.44 GiB →
  `.aimodelc` **34.32 GiB**, larger than the bf16-weight encoder's 14.1. int4 was skipped.
  coreai-opt 0.2.1 also cannot quantize the w16 custom Linear (`Tensor.float` has no aten
  mapping), so the fp32 `nn.Linear` build was used.
- **DiT int8lin (per_block 32, bf16 compute).** The efr AOT, the only GPU path, fails in the MLIR
  pass manager at the first op that touches a graph input, with bf16 I/O too and with
  `--preferred-compute none`:

  ```
  operand #0 does not dominate this use
  Pass failed: MPSMemrefAllocFusion
  ```

  A plain AOT compiles (2-layer probe 0.57 GiB, corr 0.999988 on the GPU after an
  `ANECCompile() FAILED` fallback per new shape), but the plain AOT of the full bf16 graph already
  died on the ANE region. Not pursued: Z-Image measured weight-only int8 at 2.4–2.7× slower on a
  compute-bound DiT.

Ship = bf16 / fp32 only: DiT bf16 14.23 GB (13.25 GiB), encoder w16a32 14.10 GiB, VAE fp32 0.94 GiB × 3.

## The bf16 band of the model itself

The official pipeline run in bf16 (diffusers main on MPS, same seed, noise and prompt), scored
against the fp32 oracle:

| size | bf16 diffusers (MPS) vs fp32 oracle | this port (bf16 Core AI) vs fp32 oracle |
| --- | --- | --- |
| 256² | 43.47 dB, final latent corr 0.999947, latent rel err 1.0e-2 | 46.72 dB, 0.999987, 5e-3 |
| 512² | 33.19 dB, 0.999053, 4.3e-2 | 35.49 dB (spread 23.6–43.4 over perturbed runs), 0.999559 |

The port sits in the band the official bf16 pipeline occupies against fp32, so the DiT ships bf16.
An fp32-compute DiT stays a recipe option, not a shipped bundle. Its speed is unmeasured; the
encoder's w16a32 probe ran 1.8–2.9× slower per call than r32.

**The 512² leaf fork.** Over 6 runs whose prompt embeddings differ at the 1e-5 level, 512² spans
23.6–43.4 dB (4 of 6 ≥ 30). The two 24 dB runs are the same apple with one extra leaf on the stem: a
semantic fork, not noise. The fp32 reference is not the unstable part: `prompt_embeds × (1 +
1e-5·N(0,1))` gives 82 dB at both 256² and 512², with latent rel err ≤ 8e-6 at step 38. The fork is
the bf16 DiT's per-step error (0.4–1.8 %) crossing a decision boundary at 512². Runs are
bit-reproducible: the same inputs give the same image.

Lesson: judge a bf16 port against the model's own bf16 band, and read the image before calling a
PSNR drop a regression.

## Numbers (M4 Max, macOS 27.0 26A428)

**Oracle** (fp32 CPU, diffusers main 4295ee3, seed 1234, prompt "a red apple on a wooden table,
studio lighting", 40 steps, no CFG): 2.37 s/forward at 256² (N 256, L 18), 8.97 s at 512² (N 1024).
`μ` = 0.5 at N = 256 and 0.539 at 512². The exponential shift and the terminal stretch apply at both
sizes (256²: σ₁ = 0.98436, not the linspace 0.975).

**DiT** (efr bundle, `default()`, GPU lock held, load average ~10, L = 40, `bench_dit.py`):

| size | N | first call | warm median ×5 | 40 steps (no CFG) |
| --- | --- | --- | --- | --- |
| 256² | 256 | 2.94 s | 0.343 s | 13.7 s |
| 512² | 1024 | 1.14 s | 1.104 s | 44 s |
| 1024² | 4096 | 4.77 s | 4.757 s | 190 s |

Speed against the MLX packs on the Hub (8-bit, published on M1-class Macs) is not measured on this
machine with the same protocol; no claim either way.

**Text encoder** (w16a32, efr): 0.181 / 0.271 s per call at L = 32 / 128, first call 0.37 s, load
19.6 s. Export: convert 33 s, AOT 16 s, peak RSS 35.5 GB.

**VAE** (fp32, efr, all 4 channels): corr 1.0000000, max|Δ| 1.35e-5 at 256² and 1.29e-5 at 512²;
2.33e-5 at 1024² against torch on a synthetic latent. 0.94 GiB per size. `nearest-exact` lowers to
`gather_nd`, so no upsample patch is needed. A one-frame decode is bit-identical with and without
the feature cache; `first_chunk=True` is required.

**End to end** (3 bundles, prompt → RGBA):

| size | white-composited RGB PSNR | alpha max\|Δ\| | final latent corr | time |
| --- | --- | --- | --- | --- |
| 256² | 46.72 dB (46.7–50.4 over 5 perturbed runs) | 1/255 | 0.999987 | encoder 1.15 s + DiT 40 steps 45 s (first 32 s, then 0.34 s/step) + VAE 0.16 s |
| 512² | 35.49 dB via the encoder bundle (23.6–43.4 over 6 runs, 4 ≥ 30) | 2/255 | 0.999559 | — |

Teacher-forced DiT at 512²: 40/40 corr ≥ 0.999844. RGBA sticker prompt at 512², seed 42: alpha min
0, mean 126, background corners 0.96/255; the engine VAE against a torch decode of the same latents:
max|Δ| 1.8e-5.

## Gates

Every script runs from `conversion/qwenimage21/`. Two oracles, `capture_oracle.py --size 256 --steps
40` and `--size 512` (fp32 CPU, diffusers main venv), record every boundary: encoder ids and hidden
states, per-step latents and velocities, the VAE input and output, the reference PNGs.

| gate | script | result | bar |
| --- | --- | --- | --- |
| DiT re-author vs diffusers (random 2 layers, fp32) | `parity_dit_torch.py` | max\|Δ\| 7.2e-7, corr 1.0; controls a/b/c/d red; RoPE tables max\|Δ\| 0 (4 sets); KV cache on/off exact | ≤ 1e-4 / ≥ 0.999999 / red |
| DiT re-author, real weights, fp32 CPU, teacher-forced | `parity_dit_oracle.py --steps all --controls 0,20,39` | 40/40 corr 1.000000000, max\|Δ\|/max\|ref\| ≤ 2.4e-4; controls red at 0.897 / 0.837 / 0.9969 | ≥ 0.999999 / ≤ 1e-3 |
| DiT bundle, `cpu_only()` | `engine_parity_dit.py --cpu-only --steps 0,20,39` | corr 0.99995 / 0.99998 / 0.99983, NaN 0 | ≥ 0.999 |
| DiT bundle, efr AOT, GPU | `engine_parity_dit.py` on the `.aimodelc` | 40/40, corr 0.999978 → 0.999854 (min at step 38), \|Δ\|/\|ref\| 0.4–1.7 %, NaN 0 | ≥ 0.999 / NaN 0 |
| encoder re-author, fp32 | `parity_text_oracle.py` | bit-exact, 32 tokens; controls a/b/c red (0.356 / 0.988 / 0.518) | ≥ 0.999999 |
| M-RoPE collapse | `check_mrope_hf.py` | explicit 3-axis arange == unspecified, bit-exact | exact |
| host tokenizer | `qi21_tokenize.py`, `gate_tokenize_hf.py` | 3 prompts (Lfull 32 / 23 / 47) identical to the processor, `drop_idx` 14 | identical |
| encoder bundle, efr AOT | `engine_parity_encoder.py --dit`, `sweep_encoder_L.py`, `downstream_prompts.py` | w16a32 rows above | every token ≥ 0.999 |
| VAE | `parity_vae_torch.py`, `engine_parity_vae.py` | VAE numbers above | ≥ 0.9999, ≤ 1e-2 |
| sampler | `qi21_sched.py` | sigmas, timesteps and 40 Euler steps bit-exact at 256² and 512² | ≤ 1e-6 |
| end to end | `pipeline_engine.py --oracle oracle/256` (and `oracle/512`) | table above | RGB ≥ 30 dB, alpha ≤ 2/255, final corr ≥ 0.999 |

`pipeline_engine.py --encoder oracle` feeds the oracle's own embeddings (isolates the encoder), and
`--embed-perturb 1e-5 --perturb-seed k` is the sensitivity probe behind the spread above.
`make_host_consts.py` writes the Swift host's RoPE tables and `scheduler.json` and checks them
bit-exact against `qi21_host.rope_tables`, diffusers' `QwenImage21Rope` and `qi21_sched`.

## Host side

The host owns tokenization, the RoPE tables, the sampler, and packing the noise. `qi21_tokenize.py`
needs only the `tokenizers` library: the t2i chat template, `drop_idx` computed from the system part
(14 here, never a constant), no BOS. `qi21_host.py` builds the tables: text token `i` → `(i, i, i)`,
image token `(y, x)` → `(L, y − (h − h//2), x − (w − w//2))`. `qi21_sched.py` is the scheduler twin.
The Hugging Face repo carries the tables as files and the whole contract as `host/host_contract.md`.

Feed the VAE the sampler's latent unchanged: the graph unpacks and un-normalises. Un-normalising
again on the host is the bug the Z-Image Swift port hit ([`zimage-port.md`](zimage-port.md)).

## Follow-ups

- **Swift host.** Does `GraphModel(computeUnits: .gpu)` run the DiT `.aimodel` just-in-time, or does
  it hit the same ANE region (then the app ships or builds the efr `.aimodelc`)? Untested. The
  CoreAIImageGen app has no Qwen-Image-2.1 pipeline yet.
- **iPhone** is out of scope for this port.
- **CFG.** The pipeline default is no CFG (`true_cfg_scale` 1.0). The graph runs one conditional
  forward per step; a negative-prompt branch has not been gated.
- **Editing.** Condition images, the VLM image path and the prefix KV cache are not in these graphs.
  Text-to-image only.
- **Resolution.** 256² / 512² / 1024² square. The DiT axis stops at 4096 image tokens, so the
  upstream 2048² default (16,384 tokens) needs a wider export.
