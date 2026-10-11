# Qwen-Image-2.1 → Core AI (RGBA-Image-2.1)

Text-to-image with RGBA output on the Mac GPU, from
[`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) at revision `790c926`. The
exporters and gates pin that revision; `capture_oracle.py` takes the newest local snapshot and
records its name in `meta.json`. Three graphs: the Qwen3-VL-8B text path, the 7B DiT, the VAE
decoder. The host runs the tokenizer, the RoPE tables and the sampler. Card:
[`models/rgba-image-2.1/`](../../models/rgba-image-2.1/README.md). Port notes:
[`knowledge/qwenimage21-port.md`](../../knowledge/qwenimage21-port.md). Working state and every
number: [`QWENIMAGE21_STATE.md`](QWENIMAGE21_STATE.md).

The same scripts port the distilled checkpoint
[`Qwen/Qwen-Image-2.1-Turbo`](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) (revision `d65dbc9`,
RGBA-Image-2.1-Turbo): the DiT is re-exported from its weights, the encoder and VAE bundles are
reused, and the sampler is its fixed 8-step grid. Card:
[`models/rgba-image-2.1-turbo/`](../../models/rgba-image-2.1-turbo/README.md); the commands are in
[Qwen-Image-2.1-Turbo](#qwen-image-21-turbo) below.

## Two venvs

coreai-opt 0.2.1 pins `safetensors<=0.7.0` and diffusers main needs `>=0.8.0`, so one venv cannot
hold both.

- **qi21 venv** (Python 3.11, torch 2.9.0, transformers 5.17.0, diffusers main 4295ee3,
  coreai-torch 0.4.1, coreai-core 1.0.0b2, torchvision for the processor): the oracle, the VAE
  export, and every script that imports diffusers or transformers.
- **base venv** (the coreai-models `.venv`: torch 2.9.0, coreai-torch 0.4.1, coreai-core 1.0.0b2,
  coreai-opt 0.2.1): the DiT and encoder exports, every engine gate, the quantization scripts.

`qi21_dit.py`, `qi21_text.py`, `qi21_host.py`, `qi21_sched.py` and `qi21_tokenize.py` import neither
diffusers nor transformers, so they load in both.

The Turbo checkpoint needs a diffusers main that reads its `sample_sigmas`. Its oracle ran in a
third venv, **qi21t** (Python 3.11.13, torch 2.9.0, torchvision 0.24.0, transformers 5.17.0, diffusers
main 85c9fa4, safetensors 0.9.0-rc.1, accelerate 1.15.0, no coreai packages): `capture_oracle.py`,
`gate_tokenize_hf.py`, `parity_vae_torch.py` and `make_host_consts.py --turbo`.

## Order

Run from this directory. Engine gates load the `.h16c.aimodelc` from the AOT step. Take the GPU
lock (`python3 ~/code/coreai-kit/scripts/with-gpu-lock.py -- <cmd>`) for timing runs, and when a
gate fails with other GPU work running.

| step | venv | command | what it proves or makes |
| --- | --- | --- | --- |
| 1 oracle | qi21 | `capture_oracle.py --size 256 --steps 40`, `--size 512` | fp32 CPU reference at every boundary → `oracle/<size>/` |
| 2 DiT | qi21 | `parity_dit_torch.py` | re-author == diffusers (random 2 layers, fp32), red controls, RoPE tables |
| | either | `parity_dit_oracle.py --oracle oracle/256 --steps all --controls 0,20,39` | real weights, fp32, teacher-forced |
| | base | `export_dit.py` | `qi21_dit_full_bf16_dyn_iofp32.aimodel` |
| | base | `engine_parity_dit.py <bundle> --oracle oracle/256 --steps all`, `bench_dit.py <bundle>` | bf16 bundle vs the oracle; s/forward |
| 3 encoder | base | `parity_text_oracle.py --oracle oracle/256 --save` | fp32 re-author == oracle, red controls |
| | qi21 | `check_mrope_hf.py --oracle oracle/256`, `gate_tokenize_hf.py` | M-RoPE collapses to 1D; host tokenizer == processor |
| | base | `export_encoder.py --aot --w16a32` | `qi21_encoder_dynL_w16a32_ids_iofp32.aimodel` + its efr compile |
| | base | `engine_parity_encoder.py <aimodelc> --dit`, `sweep_encoder_L.py <aimodelc>`, `downstream_prompts.py <aimodelc>`, `bench_encoder.py <aimodelc>` | every token vs the oracle; 3 prompts × 7 lengths; the DiT's sensitivity; s/call |
| 4 VAE | qi21 | `parity_vae_torch.py`; `export_vae.py --size {256,512,1024} --aot` | wrapper == pipeline decode; `qi21_vae_<size>_fp32.aimodel` |
| | base | `engine_parity_vae.py <aimodelc> --oracle oracle/256` | all 4 channels vs the oracle |
| 5 sampler | any | `qi21_sched.py` | sigmas, timesteps, Euler steps == oracle |
| 6 end to end | base | `pipeline_engine.py --oracle oracle/256` (and `oracle/512`) | image vs the reference; `--prompt "…" --size 512 --seed N` for a free prompt |
| 7 host constants | qi21 | `make_host_consts.py --out <repo>/host` | RoPE tables + `scheduler.json` for a Swift host, checked bit-exact |

`diag_text_bf16.py`, `diag_text_islands.py`, `probe_fp32_cost.py` and `ref_text_hf.py` are the
encoder-precision diagnosis behind the w16a32 choice. `quant_export_encoder.py`,
`quant_export_dit.py` and `quant_diag_w16_probe.py` are the quantization attempts; none ships for
RGBA-Image-2.1. `quant_export_dit.py` exports the int8 DiT that RGBA-Image-2.1-Turbo ships beside its
bf16 one.

## AOT: the only GPU path

JIT crashes on the 32-block DiT in the Python runtime (`ANERegion.mm:414 … Code=-19`), and so does a
plain AOT compile. Compile every bundle with `--expect-frequent-reshapes` and load the result with
`SpecializationOptions.default()`:

```
xcrun coreai-build compile <name>.aimodel --output <name>_aot_efr --platform macOS \
    --architecture h16c --preferred-compute gpu --expect-frequent-reshapes
# -> <name>_aot_efr/<name>.h16c.aimodelc
```

`export_encoder.py --aot` and `export_vae.py --aot` run this command after saving. The DiT was
compiled by hand with it (2 min 9 s, 27 GB).

## Qwen-Image-2.1-Turbo

The same order with the Turbo snapshot. `T` = the snapshot directory of `Qwen/Qwen-Image-2.1-Turbo`
at revision `d65dbc9a7e8f6b5479e33dee6030eaab2a906509`. Three flags point the scripts at it:

- `--transformer-dir $T/transformer` (`export_dit.py`, `quant_export_dit.py`, `parity_dit_oracle.py`,
  `engine_parity_encoder.py`): the Turbo DiT weights instead of the pinned base snapshot's.
- `--exports-sub qwenimage21_turbo` (`export_dit.py`, `quant_export_dit.py`): bundles go to
  `_paths.exports_dir()/qwenimage21_turbo/`, beside copies of the base encoder and VAE bundles.
- `--turbo` (`qi21_sched.py`, `make_host_consts.py`, `pipeline_engine.py`): the checkpoint's 8-step
  grid (`schedule_fixed`, no `μ`); `pipeline_engine.py --turbo` also reads the bundles from
  `qwenimage21_turbo/` and the tokenizer from `$T/processor/tokenizer.json`.

| step | venv | command | what it proves or makes |
| --- | --- | --- | --- |
| 1 oracle | qi21t | `capture_oracle.py --snapshot $T --model-id Qwen/Qwen-Image-2.1-Turbo --size 256 --steps 8 --tag turbo256` (and 512, 1024) | fp32 CPU reference on the 8-step grid → `oracle/turbo<size>/` |
| 2 sampler | any | `qi21_sched.py --turbo --oracle oracle/turbo256 oracle/turbo512 oracle/turbo1024 --snapshot $T` | sigmas, timesteps, 8 Euler steps == oracle; constants == the snapshot's |
| 3 tokenizer | qi21t | `gate_tokenize_hf.py --snapshot $T --oracle oracle/turbo256` | host ids == the Turbo processor, `drop_idx` 14 |
| 4 DiT | either | `parity_dit_oracle.py --transformer-dir $T/transformer --oracle oracle/turbo256 --steps all --controls 0,4,7` | real Turbo weights, fp32, teacher-forced, red controls |
| | base | `export_dit.py --transformer-dir $T/transformer --name qi21t_dit_full_bf16_dyn_iofp32 --exports-sub qwenimage21_turbo`, then the AOT command above | the bf16 DiT |
| | base | `quant_export_dit.py --transformer-dir $T/transformer --exports-sub qwenimage21_turbo --name qi21t_dit_full_int8lin_dyn_iofp32 --aot --torch-ref --ref-steps 0,4,7 --ref-oracle oracle/turbo256` | the int8 DiT + its efr compile; the int8 weights' own cost on the CPU |
| | base | `engine_parity_dit.py <aimodelc> --oracle oracle/turbo<size> --steps all`; `bench_dit.py <aimodelc>`; `ab_dit.py --a <A> --b <B>` | each DiT vs the oracle; s/forward; a paired same-process A/B of two forms |
| 5 encoder | base | `engine_parity_encoder.py <encoder aimodelc> --oracle oracle/turbo256 --dit --dit-steps 0,4,7 --transformer-dir $T/transformer --tokenizer-json $T/processor/tokenizer.json` | the reused encoder bundle on the Turbo reference, downstream through the Turbo DiT |
| 6 VAE | base + qi21t | `engine_parity_vae.py <aimodelc> --oracle oracle/turbo<size>`; `parity_vae_torch.py --synthetic <size> --latents oracle/turbo<size>/final_latents.f32 --snapshot <dir> [--fp64]`; `vae_band.py` | the reused fp32 VAE bundles against their own weights, and the band the Turbo bf16 VAE file puts between them and the oracle |
| 7 end to end | base | `pipeline_engine.py --turbo --oracle oracle/turbo<size>` (`--dit <int8 aimodelc>` for int8); `--turbo --prompt "…" --size 1024 --seed N` | image vs the reference; a free prompt |
| 8 host constants | qi21t | `make_host_consts.py --turbo --out <repo>/host` | RoPE tables + the fixed-grid `scheduler.json`, checked bit-exact |
| 9 ship copy | base | `strip_bundle.py <name>/<name>.aimodel <name>_s/<name>.aimodel --aot`, then the gates of steps 4–7 on the `_s` bundles | the published `.aimodel`: the export machine's paths (MLIR debug locations) stripped, ops and weights untouched, a `strip` record in `metadata.json`; outputs equal the unstripped bundles' bit for bit |
