# RGBA-Image-2.1 — Core AI port of Qwen-Image-2.1

**Built with Qwen.**

**Non-commercial use only** — Qwen Research License (research or evaluation purposes).

**macOS 27 on Apple silicon only.** Five Core AI bundles, 32.41 GB (30.18 GiB) in total.

This is [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) (revision `790c926`)
converted to Core AI `.aimodel` graphs that run on the Mac GPU. A Qwen3-VL-8B text encoder (text
path only) conditions a 7B single-stream DiT (32 blocks, block-causal attention). A 64-channel VAE
decodes to four channels, RGBA. The sampler is the pipeline default: 40 FlowMatch Euler steps, no
CFG. Ask for a transparent background in the prompt and the fourth channel is a real alpha.

Sizes: 256², 512² and 1024², square. The DiT and the text encoder have dynamic axes (see Graph
contracts); the VAE has one graph per size. The upstream default of 2048² needs 16,384 image
tokens, outside this DiT's 64–4096-token axis.

## Bundle

[🤗 mlboydaisuke/RGBA-Image-2.1-CoreAI](https://huggingface.co/mlboydaisuke/RGBA-Image-2.1-CoreAI)

| file | what it is | size |
| --- | --- | --- |
| `qi21_dit_full_bf16_dyn_iofp32.aimodel` | the DiT: bf16 weights and compute, fp32 inputs and outputs | 14.23 GB (13.25 GiB) |
| `qi21_encoder_dynL_w16a32_ids_iofp32.aimodel` | the text encoder: bf16 weights, fp32 compute; token ids in, `embed_tokens` inside | 15.14 GB (14.10 GiB) |
| `qi21_vae_{256,512,1024}_fp32.aimodel` | the VAE decoder, fp32, one per size | 1.01 GB (0.94 GiB) each |
| `host/` | per-axis RoPE tables, `scheduler.json`, [`host_contract.md`](https://huggingface.co/mlboydaisuke/RGBA-Image-2.1-CoreAI/blob/main/host/host_contract.md) | |
| `tokenizer/` | the source's `processor/` tokenizer files, unchanged | |
| `config.json` | the source's `transformer/config.json`, unchanged | |
| `LICENSE`, `NOTICE` | the Qwen Research License and the change notice | |

No quantized variant ships. The int8 DiT does not compile on the one GPU path that works (next
paragraph). int8 weights break the text encoder's `<|im_start|>` tokens (Lessons, 2). Compiled, the int8
encoder is also 34.32 GiB; this one is 14.10 GiB.

**Compile before you run.** On macOS 27.0 (26A428) the DiT crashes in the Python runtime when its
`.aimodel` is compiled just-in-time. What runs is an ahead-of-time compile with
`--expect-frequent-reshapes` (step 2 below). The compiled copies need their own disk: 27 GB for
the DiT and 14.10 GiB for the encoder.

## Use it

There is no Swift app for this model yet. The Python engine
[`conversion/qwenimage21/pipeline_engine.py`](https://github.com/john-rocky/coreai-model-zoo/blob/main/conversion/qwenimage21/pipeline_engine.py)
runs the whole loop on the three bundles: tokenize, encode, 40 DiT steps, decode, write a PNG.

```
# 1. the bundles, and the tokenizer the engine reads from the source repo's processor/
hf download mlboydaisuke/RGBA-Image-2.1-CoreAI --local-dir RGBA-Image-2.1-CoreAI
hf download Qwen/Qwen-Image-2.1 --revision 790c92633540aa0cb11d9abf19eb46d861714758 --include "processor/*"

# 2. compile for the Mac GPU (once per bundle; the gates used --architecture h16c on an M4 Max)
cd RGBA-Image-2.1-CoreAI
for m in qi21_encoder_dynL_w16a32_ids_iofp32 qi21_dit_full_bf16_dyn_iofp32 qi21_vae_512_fp32; do
  xcrun coreai-build compile $m.aimodel --output aot/$m --platform macOS --architecture h16c \
      --preferred-compute gpu --expect-frequent-reshapes
done
A=$PWD/aot

# 3. generate, from a checkout of the zoo
#    (Python with coreai-core 1.0.0b2, torch, numpy, tokenizers, pillow)
cd /path/to/coreai-model-zoo/conversion/qwenimage21
python pipeline_engine.py \
    --prompt "This is an RGBA image with transparency. A cute cartoon dragon sticker. The image has alpha channel and the background is transparent." \
    --size 512 --seed 42 --tag dragon \
    --encoder $A/qi21_encoder_dynL_w16a32_ids_iofp32/qi21_encoder_dynL_w16a32_ids_iofp32.h16c.aimodelc \
    --dit $A/qi21_dit_full_bf16_dyn_iofp32/qi21_dit_full_bf16_dyn_iofp32.h16c.aimodelc \
    --vae $A/qi21_vae_512_fp32/qi21_vae_512_fp32.h16c.aimodelc
# -> _work/samples/dragon.png (RGBA) and dragon_rgb.png (composited on white)
```

The prompt above is the upstream card's recommended form for transparent images: "This is an RGBA
image with transparency. {subject}. The image has alpha channel and the background is
transparent." The noise is `torch.randn` on a CPU generator seeded with `--seed`, the same draw the
reference pipeline makes with a CPU generator and that seed.

To check the port against the fp32 reference, record the reference once and run the engine in
oracle mode. `capture_oracle.py` runs the diffusers-main pipeline in fp32 on the CPU, about 2 min
at 256². It needs the full `Qwen/Qwen-Image-2.1` snapshot and a venv with diffusers main 4295ee3
and transformers 5.17.

```
python capture_oracle.py --size 256 --steps 40        # -> oracle/256/
python pipeline_engine.py --oracle oracle/256 \
    --encoder $A/qi21_encoder_dynL_w16a32_ids_iofp32/qi21_encoder_dynL_w16a32_ids_iofp32.h16c.aimodelc \
    --dit $A/qi21_dit_full_bf16_dyn_iofp32/qi21_dit_full_bf16_dyn_iofp32.h16c.aimodelc \
    --vae $A/qi21_vae_256_fp32/qi21_vae_256_fp32.h16c.aimodelc
# prints the latent corr vs the reference after every step, then the RGBA and white-composited PSNR
```

## Which image model should I use?

The zoo's Mac text-to-image models are not ranked; pick by trade-off. RGBA-Image-2.1 writes RGBA
natively, so it fits images that need an alpha channel.

| | params | sampler | time @1024 | precision |
| --- | --- | --- | --- | --- |
| **FLUX.2 klein** | 4B | 4 steps, guidance-distilled (no CFG) | ~17 s | int4 |
| **Z-Image-Turbo** | 6B | 8 steps + CFG (16 forwards) | ~70 s | bf16, near-lossless |
| **GLM-Image** | 16B (9B AR + 7B DiT) | AR prior + 20-step DiT | ~208 s | int8 |
| **RGBA-Image-2.1** | 7B | 40 steps, no CFG | ~190 s | bf16 |

Times are the ones each card reports, on an M4 Max. For RGBA-Image-2.1 it is 40 DiT steps at the
warm median of 4.757 s per forward. It leaves out the encoder, the VAE and loading.

## Graph contracts

Every graph has one function, `main`, and fp32 inputs and outputs except `input_ids`.

| graph | inputs | output |
| --- | --- | --- |
| encoder | `input_ids [1,Lfull]` int32, `Lfull` 16..512 | `hidden [1,Lfull,4096]`: the last layer's residual stream, before the final norm |
| DiT | `img_tokens [1,N,64]`, `txt_feats [1,L,4096]`, `timestep [1]`, `txt_cos`/`txt_sin [1,L,64]`, `img_cos`/`img_sin [1,N,64]`; `L` 8..512, `N` 64..4096 | `vel [1,N,64]` |
| VAE | `latents_packed [1,N,64]`, the sampler's latent unchanged | `image [1,4,S,S]`, RGBA in [-1, 1] |

- **Prompt.** One tokenization of the text-to-image chat template around the prompt, no padding.
  The DiT reads `hidden[:, drop_idx:]`. `drop_idx` is the token count of the template's system
  part: 14 with this tokenizer. Compute it; do not hard-code it.
- **Sequence.** `[text L | image N]`, image tokens in raster order, one token per 16×16 px tile
  (256² → 256 tokens, 512² → 1024, 1024² → 4096). There is no 2×2 latent packing.
- **RoPE.** Text token `i` sits at `(i, i, i)`. Image token `(y, x)` sits at `(L, y − (h − h//2),
  x − (w − w//2))`. `host/` has the per-axis tables for positions −1024..8191.
- **Sampler.** Shifted sigmas: `μ` from the image-token count, exponential shift, terminal stretch
  to 0.02. The DiT's `timestep` input is `timesteps[i] / 1000` in fp32 (σ to within 1 ulp), and each step is `x += (σᵢ₊₁ − σᵢ)·v`.
- **VAE.** The graph unpacks the tokens and applies `latents·std + mean` itself. Feed it the raw
  sampler latent.

The full contract, with the formulas a Swift host needs:
[`host/host_contract.md`](https://huggingface.co/mlboydaisuke/RGBA-Image-2.1-CoreAI/blob/main/host/host_contract.md).

## Measured

M4 Max, macOS 27.0 (26A428). Bundles compiled ahead of time with `--expect-frequent-reshapes`,
run with `SpecializationOptions.default()`.

**DiT speed** (`bench_dit.py`, text L = 40, GPU lock held, load average ~10):

| size | image tokens | first call | s/forward (warm median of 5) | 40 steps |
| --- | --- | --- | --- | --- |
| 256² | 256 | 2.94 s | 0.343 s | 13.7 s |
| 512² | 1024 | 1.14 s | 1.104 s | 44 s |
| 1024² | 4096 | 4.77 s | 4.757 s | 190 s |

**Text encoder:** 0.181 s per call at 32 tokens and 0.271 s at 128 (warm median). The first call
takes 0.37 s; loading takes 19.6 s.

**One 256² image end to end:** encoder 1.15 s, DiT 40 steps 45 s (the first step 32 s, then
0.34 s per step), VAE 0.16 s.

**Memory.** A 512² run with all three compiled bundles loaded in one process (`pipeline_engine.py`,
free prompt): peak resident set 58.3 GB (`/usr/bin/time -l`), wall 108 s of which loading is 43 s,
the DiT 40 steps 63 s (first step 21 s, then 1.08 s), encoder 0.5 s, VAE 0.4 s.

**Fidelity** against the fp32 diffusers reference (prompt "a red apple on a wooden table, studio
lighting", seed 1234, the same noise, 40 steps):

| size | white-composited RGB PSNR | alpha max\|Δ\| | final latent corr |
| --- | --- | --- | --- |
| 256² | 46.72 dB (46.7–50.4 over 5 runs) | 1/255 | 0.999987 |
| 512² | 35.49 dB (23.6–43.4 over 6 runs, 4 of them ≥ 30 dB) | 2/255 | 0.999559 |

The ranges are over runs whose prompt embeddings differ at the 1e-5 level.

**The model's own bf16 band.** The official pipeline run in bf16 (diffusers main, MPS), scored
against the same fp32 reference:

| size | official pipeline in bf16 | this port |
| --- | --- | --- |
| 256² | 43.47 dB, final latent corr 0.999947 | 46.72 dB, 0.999987 |
| 512² | 33.19 dB, 0.999053 | 35.49 dB, 0.999559 |

**Gates:**

- DiT re-authored in plain PyTorch vs diffusers (fp32, 2 random layers): max|Δ| 7.2e-7. With the
  real weights in fp32, teacher-forced on all 40 steps: corr 1.000000000.
- DiT bundle (bf16, GPU), teacher-forced against the fp32 reference: 40/40 steps corr ≥ 0.999854
  at 256² (NaN 0) and ≥ 0.999844 at 512².
- Text encoder re-authored in fp32: bit-exact with the reference on all 32 tokens. The bundle: min
  per-token corr 0.999999999 on the reference prompt, and ≥ 0.99999998 over 3 prompts × 7 lengths.
  The host tokenizer gives the processor's ids on all 3 prompts, `drop_idx` 14.
- VAE bundles on all four channels: corr 1.0000000, max|Δ| 1.35e-5 at 256² and 1.29e-5 at 512²;
  2.33e-5 at 1024² against torch on a synthetic latent.
- Sampler: sigmas, timesteps and all 40 Euler steps bit-exact at 256² and 512².
- Transparency: the dragon prompt above at 512², seed 42: alpha min 0, mean 126; the background
  corners average 0.96/255.

## Lessons

1. **A 2-layer probe does not clear a 32-layer graph.** On 26A428 the Python runtime puts a Neural
   Engine region inside the 32-block bf16 DiT, and the ANE inference fails (`Code=-19`). It fails
   under JIT and under a plain AOT compile; the 2-layer probe never triggered it. The GPU path that
   runs is AOT with `--expect-frequent-reshapes`: 2 min 9 s to compile, 27 GB, MPSGraph delegates
   only. The int8 DiT does not compile with that flag (`Pass failed: MPSMemrefAllocFusion`).
2. **The encoder's `<|im_start|>` tokens need fp32 compute, not fp32 storage.** Token 14, the `<|im_start|>`
   that opens the user turn, is the first token the DiT reads. Its residual grows to |h| ≈ 9,100 in
   layers 17–34, and the last two layers cancel it to ≈ 100. bf16 cannot hold that cancellation:
   per-token corr 0.968 in torch bf16, 0.976 on the engine. An fp32 residual stream alone did not
   hold across prompts and lengths. Full fp32 compute over bf16-stored weights did: min token corr
   0.999999999 at the same 14.10 GiB. The DiT barely notices the bf16 error (velocity corr
   ≥ 0.99998); only a per-token gate catches it.
3. **Judge a bf16 port by the model's own bf16 band, and look at the image when PSNR drops.** At
   512², six runs whose prompt embeddings differ at the 1e-5 level span 23.6–43.4 dB. The two runs
   near 24 dB show the same apple with one extra leaf on the stem: a semantic fork, not noise. The
   fp32 reference stays at 82 dB under a 1e-5 perturbation, so the fork comes from the bf16 DiT's
   per-step error (0.4–1.8 %). The official pipeline in bf16 scores 33.19 dB at 512²; this port's
   35.49 dB is in that band.

Port notes, every gate and the dead ends:
[`knowledge/qwenimage21-port.md`](https://github.com/john-rocky/coreai-model-zoo/blob/main/knowledge/qwenimage21-port.md).
Scripts: [`conversion/qwenimage21/`](https://github.com/john-rocky/coreai-model-zoo/tree/main/conversion/qwenimage21).

## Licence

The weights are Qwen Materials under the **Qwen RESEARCH LICENSE AGREEMENT** (release date
2026-09-20), included unchanged as
[`LICENSE`](https://huggingface.co/mlboydaisuke/RGBA-Image-2.1-CoreAI/blob/main/LICENSE). A summary
follows; the Agreement is what binds.

- **§2, non-commercial only.** You may use, copy, modify and redistribute the Materials for
  research or evaluation purposes only. Commercial use needs a separate licence from Hangzhou
  Tongyi Laboratory Technology Co., Ltd.; §2(b) gives the contact.
- **§3, redistribution,** under three conditions: every recipient gets a copy of the Agreement
  (`LICENSE`); modified files carry a notice that they were changed (`NOTICE` lists every change);
  and copies keep the attribution text below in a "Notice" file (`NOTICE`, first line).
- **§4(b).** An AI model created from the Materials and made available displays "Built with Qwen"
  prominently in its documentation. This card does, at the top.
- **§4(c).** "Qwen" is not the primary name of a derivative. This port is named RGBA-Image-2.1;
  "Core AI port of Qwen-Image-2.1" is the descriptive use the Agreement permits.

[`NOTICE`](https://huggingface.co/mlboydaisuke/RGBA-Image-2.1-CoreAI/blob/main/NOTICE) begins with
the attribution text §3(c) requires:

> Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.
