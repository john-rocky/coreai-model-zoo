# coreai_models python overlay

The zoo's conversion scripts import re-authored model definitions and export-pipeline hooks
that Apple's `coreai-models` does not ship (it takes no PRs and does not register newer
models). This directory packages those additions so the zoo is **self-contained**: anyone can
reproduce the conversion environment from a pinned upstream checkout instead of needing our
working tree.

## Contents

- `BASE` — pinned upstream repo + commit the overlay applies to.
- `patches/python-overlay.patch` — the whole diff from the pinned base: the export pipeline
  (`export/{bundle,macos,metadata,pipeline,presets}.py`), registries (`model_registry.py`,
  `models/registry.py`), primitives (`primitives/{macos,ios}/{cache,rope}.py`), the diffusion
  pipeline (`diffusion/{components,flux2,pipeline}.py` — FLUX.2 in-context editing), the VLM
  export path, and the re-authored decoders themselves (qwen3.5 / qwen3.6-MoE / gemma4 /
  GLM-4.7 / LFM2.5 / LLaDA / BitCPM / RWKV-7 / Zaya / MiniCPM / omni / VL towers, …) plus the
  Metal-kernel variants. Research variants ride along (`gemma4_metal_mlp_fp4.py`, `*_int2.py`,
  `gemma4_mtp_drafter.py`) — the cards say which export is the shipping one.
- `files/` — package files that are still untracked in the source checkout. Most of what used
  to live here is now committed there, so it comes through the patch instead; regen.sh sorts
  that out automatically. `files/python/src/coreai_models/models/ios/lfm2.py` (2026-09-17) is the
  LFM2 / LFM2.5 builder for Apple's **static iOS (Neural Engine) path** — the conv layers keep their
  history inside the key/value cache rows so the unmodified Apple runner can drive the bundle. It is
  not wired into Apple's `models/registry.py`; call its `register()` before `coreai.llm.export`'s
  `main()` (`conversion/export_lfm25_ane_static.py` does exactly that).

  `files/python/src/coreai_models/models/macos/lfm3.py` and `models/ios/lfm3.py` (2026-10-30) are the LFM3
  decoder modules (LFM3-1.7B, and the language model of the LFM3-VL / -Omni / -Audio ports). They were copied in
  by hand, not by `regen.sh`, which would also have taken the source checkout's other work in progress. The macOS
  module is the decoder for the pipelined GPU engine: it imports `Lfm2ShortConv`, `_rotate_half`,
  `DECODE_STATE_NAMES` and `_MLP_KEY_MAP` from `models/macos/lfm2.py` (in the patch) and reads the safetensors
  itself, since transformers 4.57 has no lfm3; `conversion/export_lfm3_decode_pipelined.py` drives it. The iOS
  module is the static (Neural Engine) builder and imports the macOS one. Its `register()` does not make
  `coreai.llm.export --platform iOS` work for LFM3 (that path sizes the cache rows from n_kv × head_dim and needs a
  transformers config), so `conversion/export_lfm3_ane_static.py` loads it with `lfm3_ios_from_hf(...)` and calls
  `_export_ios_programs` / `_convert_to_coreai` (`export/ios.py`) with `build_ios_reference_inputs(...)`. Checked on
  a fresh clone of BASE after `apply.py`: both modules and every `coreai_models` file the two exporters load come
  from the applied tree, the zoo's three LFM3 tiny smokes pass, and `zoo_convert.py run lfm3-1.7b` exports the
  published graph (byte for byte but for the names each export draws at random and the lengths that encode them).
- `apply.py` — applies patch + files onto a pinned checkout (verifies the base commit).
- `regen.sh` — regenerates patch + files from a live checkout (run after new porting work).

## Use

```sh
git clone https://github.com/apple/coreai-models.git
git -C coreai-models checkout "$(awk -F': *' '/^commit:/{print $2}' BASE)"
python3 apply.py ./coreai-models
cd coreai-models && python3 -m venv .venv && . .venv/bin/activate && pip install -e python/
```

Then run any `conversion/export_*.py` with that venv (per-script flags: see
[`../README.md`](../README.md) and the model's zoo card).

## Scope

Python package only (`python/src/coreai_models/**`). The Swift engine changes that some zoo
apps need are **not** captured here — they live as commits on the public fork
[john-rocky/coreai-models](https://github.com/john-rocky/coreai-models) (e.g. `trimKVCache`,
pipelined-engine stop fix) and as per-app patches like
[`../../apps/coreai-pipelined-extra-states.patch`](../../apps/coreai-pipelined-extra-states.patch);
in-flight engine experiments are tracked by their stream state docs, not frozen here.

## License

The patch modifies and the new files derive idioms from Apple's `coreai-models`
(see upstream LICENSE); they are provided under the same terms as this repository's LICENSE.
