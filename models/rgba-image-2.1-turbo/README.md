# RGBA-Image-2.1-Turbo — Core AI port of Qwen-Image-2.1-Turbo

- **Built with Qwen.**
- **Non-commercial use only** — Qwen Research License (research or evaluation purposes).
- **macOS 27 on Apple silicon only.** Six Core AI bundles, 39.97 GB (37.22 GiB) in total; a run loads
  five of them, 32.41 GB with the bf16 DiT or 25.73 GB with the int8 DiT.

RGBA-Image-2.1-Turbo is [`Qwen/Qwen-Image-2.1-Turbo`](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)
(revision `d65dbc9`) converted to Core AI `.aimodel` graphs with coreai-torch 0.4.1 and coreai-core
1.0.0b2. On an Apple M4 Max (macOS 27.0 26A428, measured 2026-10-11) the 7B DiT runs a 1024² step in
4.72 s, so an 8-step image takes about 38 s of DiT time.

A Qwen3-VL-8B text encoder (text path only) conditions a 7B single-stream DiT (32 blocks,
block-causal attention). A 64-channel VAE decodes to four channels, RGBA. The sampler is the
checkpoint's own: 8 FlowMatch Euler steps on a fixed sigma grid, no CFG. Ask for a transparent
background in the prompt and the fourth channel is a real alpha.

Sizes: 256², 512² and 1024², square. The DiT and the text encoder have dynamic axes (see Graph
contracts); the VAE has one graph per size. The upstream 1:1 preset of 2048² needs 16,384 image
tokens, outside this DiT's 64–4096-token axis.

## What changed from RGBA-Image-2.1

The graph code and the host code are those of
[RGBA-Image-2.1](https://huggingface.co/mlboydaisuke/RGBA-Image-2.1-CoreAI). The differences, and why
the rest is the same:

- **Sampler.** 8 steps on the grid the checkpoint carries (`sample_sigmas` in its `model_index.json`:
  1.0 … 0.414568), shift 1.0, no `μ`, no CFG. RGBA-Image-2.1 runs 40 steps on a shifted linspace.
- **DiT.** Re-exported from the Turbo weights with the same exporter: 297 bf16 tensors with the
  base checkpoint's names, shapes and dtypes. An int8 variant ships beside it.
- **Text encoder.** The Turbo text encoder is byte-identical to the base one (750 of 750 tensors).
  This repo carries RGBA-Image-2.1's encoder graph. Its export-time debug locations are stripped, so
  the file differs from RGBA-Image-2.1's by that alone.
- **VAE.** The Turbo `vae/` is the base fp32 VAE rounded to bf16 (238 of 238 tensors). This repo
  carries RGBA-Image-2.1's fp32 VAE graphs. Their export-time debug locations are stripped, so the
  files differ from RGBA-Image-2.1's by that alone. Their decode differs from the Turbo pipeline's by
  the rounding band in Measured.

The Turbo DiT is a distilled checkpoint. On the same noise and prompt, the fp32 Turbo pipeline's
8-step image and the fp32 base pipeline's 40-step image are 21.75 dB (256²) and 23.35 dB (512²)
apart (white-composited RGB PSNR).

## Bundle

[🤗 mlboydaisuke/RGBA-Image-2.1-Turbo-CoreAI](https://huggingface.co/mlboydaisuke/RGBA-Image-2.1-Turbo-CoreAI)

| file | what it is | size |
| --- | --- | --- |
| `qi21t_dit_full_bf16_dyn_iofp32.aimodel` | the DiT: bf16 weights and compute, fp32 inputs and outputs | 14.23 GB (13.25 GiB) |
| `qi21t_dit_full_int8lin_dyn_iofp32.aimodel` | the same DiT with int8 Linear weights (one bf16 scale per 32 inputs), bf16 compute | 7.56 GB (7.04 GiB) |
| `qi21_encoder_dynL_w16a32_ids_iofp32.aimodel` | the text encoder: bf16 weights, fp32 compute; token ids in, `embed_tokens` inside | 15.14 GB (14.10 GiB) |
| `qi21_vae_{256,512,1024}_fp32.aimodel` | the VAE decoder, fp32, one per size | 1.01 GB (0.94 GiB) each |
| `host/` | per-axis RoPE tables, `scheduler.json` (the 8-step grid), [`host_contract.md`](https://huggingface.co/mlboydaisuke/RGBA-Image-2.1-Turbo-CoreAI/blob/main/host/host_contract.md) | |
| `tokenizer/` | the source's `processor/` files, unchanged | |
| `config.json` | the source's `transformer/config.json`, unchanged | |
| `LICENSE`, `NOTICE` | the Qwen Research License and the change notice | |

The two DiT bundles have one interface; a run uses one of them. On every gate below the int8 DiT
stays inside the bar, at the bf16 DiT's speed (within 0.1 %). Compiled, the bf16 DiT is 26.52 GiB
and the int8 DiT 20.30 GiB.

**Compile before you run.** On macOS 27.0 (26A428) the Python runtime crashes on this DiT unless it
is compiled ahead of time with `--expect-frequent-reshapes` (step 2 below). A plain ahead-of-time
compile fails at the initial call (`ANERegion.mm:414 … Code=-19`), and so does a just-in-time load with
the default options or with the GPU preferred alone, in Python (2026-09-25, base weights) and in Swift
(2026-10-11). A Swift host that opens the `.aimodel` with `SpecializationOptions(preferredComputeUnitKind:
.gpu)` and `expectFrequentReshapes = true` runs it just in time, and its bf16 outputs equal the compiled
bundle's bit for bit (CoreAIImageGenMac, 2026-10-11). The compiled copies need their own disk:
26.52 GiB for the bf16 DiT or 20.30 GiB for the int8, and 14.10 GiB for the encoder.

## Use it

The Mac app [CoreAIImageGenMac](https://github.com/john-rocky/coreai-samples/tree/main/CoreAIImageGenMac)
(coreai-samples) runs this model with the runtime's `AIModel`, compiled just in time: type a prompt, pick
256 / 512 / 1024, Generate, and Save… writes the RGBA PNG; a 1024² image takes 39.8 s on the M4 Max by the
app's clock (2026-10-11, first load 82.5 s). The Python engine
[`conversion/qwenimage21/pipeline_engine.py`](https://github.com/john-rocky/coreai-model-zoo/blob/main/conversion/qwenimage21/pipeline_engine.py)
runs the whole loop on the three bundles: tokenize, encode, 8 DiT steps, decode, write a PNG.
`--turbo` selects the 8-step grid.

```
# 1. the bundles (the tokenizer is in tokenizer/)
hf download mlboydaisuke/RGBA-Image-2.1-Turbo-CoreAI --local-dir RGBA-Image-2.1-Turbo-CoreAI

# 2. compile for the Mac GPU (once per bundle; the gates used --architecture h16c on an M4 Max)
cd RGBA-Image-2.1-Turbo-CoreAI
for m in qi21_encoder_dynL_w16a32_ids_iofp32 qi21t_dit_full_bf16_dyn_iofp32 qi21_vae_256_fp32 qi21_vae_1024_fp32; do
  xcrun coreai-build compile $m.aimodel --output aot/$m --platform macOS --architecture h16c \
      --preferred-compute gpu --expect-frequent-reshapes
done
R=$PWD; A=$PWD/aot

# 3. generate, from a checkout of the zoo
#    (Python with coreai-core 1.0.0b2, torch, numpy, tokenizers, pillow)
cd /path/to/coreai-model-zoo/conversion/qwenimage21
python pipeline_engine.py --turbo \
    --prompt "This is an RGBA image with transparency. A cute cartoon dragon sticker. The image has alpha channel and the background is transparent." \
    --size 1024 --seed 42 --tag dragon \
    --tokenizer-json $R/tokenizer/tokenizer.json \
    --encoder $A/qi21_encoder_dynL_w16a32_ids_iofp32/qi21_encoder_dynL_w16a32_ids_iofp32.h16c.aimodelc \
    --dit $A/qi21t_dit_full_bf16_dyn_iofp32/qi21t_dit_full_bf16_dyn_iofp32.h16c.aimodelc \
    --vae $A/qi21_vae_1024_fp32/qi21_vae_1024_fp32.h16c.aimodelc
# -> _work/samples/dragon.png (RGBA) and dragon_rgb.png (composited on white)
```

For the int8 DiT, compile `qi21t_dit_full_int8lin_dyn_iofp32` the same way and pass its
`.aimodelc` as `--dit`. The prompt above is the upstream card's recommended form for transparent
images: "This is an RGBA image with transparency. {subject}. The image has alpha channel and the
background is transparent." The noise is `torch.randn` on a CPU generator seeded with `--seed`, the
same draw the reference pipeline makes with a CPU generator and that seed.

To check the port against the fp32 reference, record the reference once and run the engine in
oracle mode. `capture_oracle.py` runs the diffusers-main pipeline in fp32 on the CPU, 17 s of
generation at 256² on the M4 Max. It needs the full `Qwen/Qwen-Image-2.1-Turbo` snapshot and a venv
with diffusers main 85c9fa4 and transformers 5.17.

```
python capture_oracle.py --snapshot <Qwen-Image-2.1-Turbo snapshot dir> --model-id Qwen/Qwen-Image-2.1-Turbo \
    --size 256 --steps 8 --tag turbo256                    # -> oracle/turbo256/
python pipeline_engine.py --turbo --oracle oracle/turbo256 \
    --tokenizer-json $R/tokenizer/tokenizer.json \
    --encoder $A/qi21_encoder_dynL_w16a32_ids_iofp32/qi21_encoder_dynL_w16a32_ids_iofp32.h16c.aimodelc \
    --dit $A/qi21t_dit_full_bf16_dyn_iofp32/qi21t_dit_full_bf16_dyn_iofp32.h16c.aimodelc \
    --vae $A/qi21_vae_256_fp32/qi21_vae_256_fp32.h16c.aimodelc
# prints the latent corr vs the reference after every step, then the RGBA and white-composited PSNR
```

## Which image model should I use?

The zoo's Mac text-to-image models are not ranked; pick by trade-off. The two RGBA-Image models write
RGBA natively, so they fit images that need an alpha channel.

| | params | sampler | time @1024 | precision |
| --- | --- | --- | --- | --- |
| **FLUX.2 klein** | 4B | 4 steps, guidance-distilled (no CFG) | ~11 s | fp16 or int8 |
| **Z-Image-Turbo** | 6B | 8 steps + CFG (16 forwards) | ~70 s | bf16, near-lossless |
| **GLM-Image** | 16B (9B AR + 7B DiT) | AR prior + 20-step DiT | ~208 s | int8 |
| **RGBA-Image-2.1** | 7B | 40 steps, no CFG | ~190 s | bf16 |
| **RGBA-Image-2.1-Turbo** | 7B | 8 steps, distilled, no CFG | ~38 s | bf16 or int8 |

Times are the ones each card reports, on an M4 Max. For the two RGBA-Image models they are the DiT
steps at the warm median per forward: 40 × 4.757 s and 8 × 4.721 s. They leave out the encoder, the
VAE and loading.

## Graph contracts

Every graph has one function, `main`, and fp32 inputs and outputs except `input_ids`. Both DiT
bundles have the DiT row's contract.

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
- **Sampler.** The 8 sigmas in `host/scheduler.json`, then `0`. Shift 1.0 leaves them unchanged,
  and there is no `μ` and no terminal stretch, so the grid is the same at every size. The DiT's
  `timestep` input is `timesteps[i] / 1000` in fp32 (σᵢ bit for bit on this grid), and each step is
  `x += (σᵢ₊₁ − σᵢ)·v`.
- **VAE.** The graph unpacks the tokens and applies `latents·std + mean` itself. Feed it the raw
  sampler latent.

The full contract, with the formulas a Swift host needs:
[`host/host_contract.md`](https://huggingface.co/mlboydaisuke/RGBA-Image-2.1-Turbo-CoreAI/blob/main/host/host_contract.md).

## Measured

M4 Max (128 GiB), macOS 27.0 (26A428), coreai-core 1.0.0b2, 2026-10-11. Bundles compiled ahead of
time with `--expect-frequent-reshapes` (`--architecture h16c`), run with
`SpecializationOptions.default()`. Speed runs held the Mac's measurement window, which holds back
other sessions' heavy jobs. The same form timed in different windows moved by up to 1.6 %. The
timings were taken on the bundles before their debug locations were stripped; the published files
compute the same outputs bit for bit (Gates).

**DiT speed** (`bench_dit.py`, text L = 40, bf16 DiT):

| size | image tokens | initial call | s/forward (warm median of 5) | 8 steps |
| --- | --- | --- | --- | --- |
| 256² | 256 | 2.27 s | 0.341 s | 2.7 s |
| 512² | 1024 | 1.10 s | 1.097 s | 8.8 s |
| 1024² | 4096 | 4.72 s | 4.721 s | 37.8 s |

The int8 DiT against the bf16 one in the same process (`ab_dit.py`, L = 40, 8 alternating pairs per
size): 0.3409 / 1.0923 / 4.7808 s per forward against 0.3410 / 1.0926 / 4.7850 s.

**One image end to end** (`pipeline_engine.py --turbo`, bf16 DiT, inside the measurement window,
compile cache already warm):

| prompt, size | encoder load / call | DiT load / initial step / 8 steps | VAE load / call | total |
| --- | --- | --- | --- | --- |
| dragon sticker, 512² | 17.0 / 0.40 s | 2.0 / 3.20 / 10.9 s | 1.2 / 0.38 s | 32.0 s |
| dragon sticker, 1024² | 3.4 / 0.37 s | 2.0 / 6.63 / 39.6 s | 0.5 / 1.45 s | 47.3 s |
| apple, 1024² | 3.5 / 0.36 s | 1.9 / 6.64 / 39.3 s | 0.3 / 1.46 s | 47.0 s |

Load times vary between processes on this shared Mac: 1.9–43.3 s for the DiT and 2.5–55.8 s for
the encoder across the gate runs. The cold load of a new `.aimodelc` adds a compile-cache entry the
size of the compiled bundle (26 GB for the bf16 DiT, 20 GB for the int8).

**Memory.** Peak memory footprint 4.80 GB at 512² and 17.77–18.16 GB at 1024²
(`/usr/bin/time -l`). Its "maximum resident set size", 58.3 GB with the bf16 DiT and 51.6 GB with
the int8, counts the mapped weight pages of the encoder and the DiT.

**Fidelity** against the fp32 diffusers Turbo pipeline (prompt "a red apple on a wooden table,
studio lighting", seed 1234, the same noise, 8 steps):

| size | bf16 DiT: white-composited RGB PSNR / final latent corr | int8 DiT: the same | alpha max\|Δ\| |
| --- | --- | --- | --- |
| 256² | 43.79 dB / 0.999961 | 41.76 dB / 0.999944 | 1/255 |
| 512² | 42.45 dB / 0.999800 | 49.28 dB / 0.999976 | 1/255 |
| 1024² | 42.50 dB / 0.999926 | 39.32 dB / 0.999873 | 1/255 |

The reference decodes with the Turbo bf16 VAE file and the bundles carry the fp32 weights; the band
between the two (below) is inside these numbers. With the reference's prompt embeddings in place of
the encoder bundle, 256² scores 46.57 dB (bf16 DiT) and 42.56 dB (int8 DiT).

**The VAE band.** The same final latent decoded with the fp32 weights (these bundles) and with the
Turbo pipeline's bf16 copy:

| size | max\|Δ\| on [-1, 1] | uint8 PSNR | max levels, RGB / alpha | bundle vs a fp32 decode of its own weights |
| --- | --- | --- | --- | --- |
| 256² | 0.074 | 56.95 dB | 10 / 1 | max\|Δ\| 3.0e-5 |
| 512² | 0.106 | 58.95 dB | 14 / 1 | max\|Δ\| 3.2e-5 |
| 1024² | 0.105 | 58.15 dB | 13 / 1 | max\|Δ\| 4.2e-5 |

**Gates:**

- DiT re-authored in plain PyTorch, fp32, with the Turbo weights, teacher-forced on all 8 steps:
  min corr 0.999999999982 at 256² and 0.999999999997 at 512²; 9 of 9 broken variants fail the bar
  (256²).
- DiT bundles (GPU), teacher-forced against the fp32 reference, 8/8 steps, NaN 0: min corr 0.999971 /
  0.999940 / 0.999977 at 256² / 512² / 1024² (bf16) and 0.999960 / 0.999883 / 0.999965 (int8).
- Text encoder bundle on the Turbo reference: min per-token corr 0.999999998629, NaN 0. The host
  tokenizer gives the Turbo processor's ids, `drop_idx` 14.
- VAE bundles against a fp32 decode of their own weights: max|Δ| 3.0e-5 / 3.2e-5 / 4.2e-5, every
  channel corr ≥ 0.99999995.
- Sampler: sigmas, timesteps and all 8 Euler steps bit-exact at 256², 512² and 1024². RoPE tables
  bit-exact against diffusers main.
- Transparency: the dragon prompt above, seed 42: alpha min 0 and all four corner pixels 0 at 512²
  (alpha mean 102) and 1024² (alpha mean 82).
- Stripped bundles reproduce the unstripped outputs bit for bit. The published files have their
  export-time debug locations stripped; against the bundles they were made from, both DiTs (8
  teacher-forced steps at 256²), the encoder, the three VAEs and a 256² image end to end with each DiT
  give identical outputs.

**Other forms tried** (s/forward against the shipped bf16 DiT in the same window):

| form | result |
| --- | --- |
| int8 Linear weights, AOT with `--expect-frequent-reshapes` | compiles in 301 s, passes every gate, −0.02 / −0.02 / −0.09 % at 256² / 512² / 1024²: ships as the second DiT |
| image axis fixed at N = 4096 | −0.50 % at L = 40, +0.58 % at L = 18: the dynamic graph ships |
| `--preferred-compute neural-engine` | 0 Neural Engine regions; runs on the GPU, outputs equal bit for bit, +0.01 / 0.00 / +0.72 % |
| plain AOT, no `--expect-frequent-reshapes` | crashes at the initial call (`ANERegion.mm:414 … Code=-19`), dynamic and fixed image axis alike |
| text K/V computed once per image | the 32 text tokens between L = 8 and L = 40 cost 1.59 % of a forward at 512² and 0.51 % at 1024² |

## Lessons

1. **A probe's compile result does not carry to the full graph, either way.** The 2-layer int8 probe
   with random weights fails the `--expect-frequent-reshapes` compile (`Pass failed:
   MPSMemrefAllocFusion`), on 2026-09-25 and again on 2026-10-11. The full 32-layer int8 DiT with the
   Turbo weights compiles in 301 s and passes every gate. On 2026-09-25 it went the other way: the
   probe ran where the full bf16 DiT crashed. Judge compilation on the full graph.
2. **A sibling checkpoint can carry a bf16 copy of a component.** The Turbo `vae/` is the base fp32
   VAE rounded to bf16, so the fp32 bundles sit 0.074–0.106 from the Turbo pipeline's decode, 7–10×
   the 1e-2 bar. Gate a bundle against a fp32 decode of the weights it holds, and report the rounding
   band beside it. Alpha is near-constant on an opaque image (0.976–1.000), so its correlation drops
   to 0.966–0.991 under that band: report alpha in levels, not correlation.
3. **Read a dB gap between two DiT forms next to a nudge control.** The 8-step sampler turns a
   1.25e-5 relative change of the prompt embeddings into 29.03 dB (512²) and 32.08 dB (1024²)
   between dragon stickers. The int8 and bf16 stickers are 34.42 and 28.99 dB apart, the same order.
   78–88 % of the pixels whose alpha moves by more than 2 levels lie within 3 px of the silhouette:
   alpha max|Δ| on a sticker (227–255) measures the edge.
4. **`--preferred-compute neural-engine` can compile to 0 regions without an error.** This DiT
   compiled that way has no Neural Engine region and runs on the GPU, equal bit for bit to the gpu
   compile. Count the `*ANE_region*` entries before timing an ANE form.
5. **An exported `.aimodel` carries the export machine's paths.** coreai-torch writes every op's
   source location into `main.mlirb`, and `grep -I` skips that binary. coreai-torch's
   `strip_debug_info` removes them without touching the ops or the weights
   ([`strip_bundle.py`](https://github.com/john-rocky/coreai-model-zoo/blob/main/conversion/qwenimage21/strip_bundle.py));
   the result is a new asset, so gate it again.

The lessons of the base port (the Neural Engine region, the encoder's fp32 compute, the bf16 band):
[RGBA-Image-2.1](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/rgba-image-2.1/README.md#lessons).
Port notes, every gate and the dead ends:
[`knowledge/qwenimage21-port.md`](https://github.com/john-rocky/coreai-model-zoo/blob/main/knowledge/qwenimage21-port.md).
Scripts: [`conversion/qwenimage21/`](https://github.com/john-rocky/coreai-model-zoo/tree/main/conversion/qwenimage21).

## Licence

The weights are Qwen Materials under the **Qwen RESEARCH LICENSE AGREEMENT** (release date
2026-09-20), included unchanged as
[`LICENSE`](https://huggingface.co/mlboydaisuke/RGBA-Image-2.1-Turbo-CoreAI/blob/main/LICENSE). A
summary follows; the Agreement is what binds.

- **§2, non-commercial only.** You may use, copy, modify and redistribute the Materials for
  research or evaluation purposes only. Commercial use needs a separate licence from Hangzhou
  Tongyi Laboratory Technology Co., Ltd.; §2(b) gives the contact.
- **§3, redistribution,** under three conditions: every recipient gets a copy of the Agreement
  (`LICENSE`); modified files carry a notice that they were changed (`NOTICE` lists every change);
  and copies keep the attribution text below in a "Notice" file (`NOTICE`, line 1).
- **§4(b).** An AI model created from the Materials and made available displays "Built with Qwen"
  prominently in its documentation. This card does, at the top.
- **§4(c).** "Qwen" is not the primary name of a derivative. This port is named
  RGBA-Image-2.1-Turbo; "Core AI port of Qwen-Image-2.1-Turbo" is the descriptive use the Agreement
  permits.

[`NOTICE`](https://huggingface.co/mlboydaisuke/RGBA-Image-2.1-Turbo-CoreAI/blob/main/NOTICE) begins
with the attribution text §3(c) requires:

> Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.
