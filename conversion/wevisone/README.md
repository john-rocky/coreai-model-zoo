# wevisone — WeVisOne-VL-2B on Core AI

Scripts for [tencent/WeVisOne-VL-2B](https://huggingface.co/tencent/WeVisOne-VL-2B) (revision
`3dad9ce8bcb6fbe0aea945bf8860bd2cfc3663bf`) and its inference package
[Tencent/WeVisOne-VL](https://github.com/Tencent/WeVisOne-VL) (`40f735b8daed437ece982db9718e9c9856f52a79`). The
checkpoint, by WeChat Vision, Tencent Inc., is built upon Qwen3-VL: its text decoder and vision tower go through this
repository's Qwen3-VL exporter ([`../export_qwen3_vl_pipelined.py`](../export_qwen3_vl_pipelined.py), on the overlay's
`qwen3_vl_pipelined.py`), and the package's 9,511,460-parameter mask decoder (`mask_decoder.pt`) is restructured here
into two static graphs. Card: [`../../models/wevisone-vl-2b/README.md`](../../models/wevisone-vl-2b/README.md); the
Swift host that runs the four graphs in an app is CoreAIKit's `Examples/Cutout`. `<work>` below is the work directory
`$ZOO_WORK_ROOT/_wevisone/` (`conversion/_paths.py`); the conversion's working records live there, outside the
repository.

## The four graphs on Hugging Face

| graph | made by |
|---|---|
| `wevisone_vl_2b_decode_int8hu_pf64/` — the text decoder: `main` (S = 1) and `prefill` (S = 64) sharing the two KV states, outputs `logits` and `last_hidden` (the last layer's output before the final norm, what the package's `LastHiddenCapture` reads at each `<OBJ>`); linears int8 per block of 32 (body `symmetric_with_clipping`, the untied head absmax) | `mask/mask_hidden.py int8hu-pf64` (the exporter call of `export_wevisone.py` with the two-output decoder class) |
| `wevisone_vl_2b_vision_taps_g14.aimodel` — the vision tower for a 448 × 448 image (196 rows), fp16, with the mask decoder's four taps (ViT layers 5 / 11 / 17 / 23, row-major) as a third output | `mask/mask_vision_taps.py build --grid 14` |
| `wv_mask_a_g28_n16_fp16_hf.aimodel` — mask decoder part A: 16 query rows per call, the box Fourier encoding (`box_pe`) given by the host, fp16 | `mask/mask_build.py a --grid 28 --host-fourier --precision fp16` |
| `wv_mask_b_fp16.aimodel` — mask decoder part B: one call per match, `mask_logits [1, 1024, 1024]` fp32 | `mask/mask_build.py b --precision fp16` |

[`strip_stage.py`](strip_stage.py) then removes the MLIR debug locations (the exporting machine's paths) and lays out
the repository's `ios/` folder; `macos/` holds the same four compiled ahead of time for the Mac ([Stage](#stage)). Run
on the 2026-10-24 exports (`<work>/exports/r2/`), `strip_stage.py` gives the published `main.mlirb` of all four graphs
byte for byte. A new export of the same script gives the same weights and ops under other names: the exporter names
its externalized composites and resources anew on every export (`layer_norm_<8 letters>`, `<module>.sdpa_<8 hex>`,
`resource_<integer>`), so a re-export is compared with `coreai-build inspect` and the weights, not with a hash (the
tower exported again on 2026-10-25: the first 811,102,801 bytes of the stripped `main.mlirb`, which hold the weights,
equal, the last 12 KB of names differ, `inspect` and the op counts equal).

## Files

| file | what it does |
|---|---|
| `make_fixtures.py` | the fixture set: copies a shared base set's images (sha256 of the file and of its decoded pixels checked), keeps its rows, adds rows to reach a per-task floor; prompts from the package's `build_prompt` |
| `mask/make_fixtures_r2.py` | four segmentation / matting rows with several objects in one image (`fixtures_r2.json`), for the mask decoder's self-attention across `<OBJ>` rows |
| `oracle_wevisone.py` | the fp32 CPU oracle, the package's own transformers path, one thread: O1 (the author's form: the original image, the package's smart-resize) and O2 (the image resized to a fixed 448 × 448 or 512 × 512 first), greedy at repetition penalty 1.0, plus O2 at 1.05 and in bf16; per row the prompt ids, pixels, positions, vision outputs, 16 steps of full logits, and for segmentation / matting rows the mask decoder's inputs and outputs |
| `oracle_batch.py` | runs the oracle over every row and form, a few one-thread processes at a time, resumable |
| `oracle_index.py` | indexes the oracle runs into one `oracle.json` |
| `parity_torch_ladder.py` | the overlay's decoder and fixed-grid tower in fp32 torch against O2: M-RoPE positions at every position and step, vision cos, prompt-last logits, 16 teacher-forced steps, two red arms |
| `export_wevisone.py` | the Qwen3-VL exporter for this checkpoint: asserts the cache's `refs/main` is the pinned revision and runs offline, copies the checkpoint's own tokenizer files (the export venv's transformers 4.57 cannot load its v5-layout `tokenizer_config.json`), writes the provenance; decoder forms `dec-int8hu-pf64`, `dec-int8hu-s1`, `dec-fp16-s1`, `dec-fp16w32-s1`, `dec-int8lin-pf64`, towers `vis-fp16`, `vis-int8lin` |
| `aot_compile.py` | `coreai-build compile … --platform macOS --preferred-compute gpu --architecture h16c` (`--reshapes` adds `--expect-frequent-reshapes` for a decoder) and a record; the gates load these AOT assets |
| `gate_common.py` | the engine-gate pieces: S=1 decoder driver on fresh states, the host rule for image tokens and RoPE shifts, records |
| `gate_decoder.py` | one decoder bundle on the Mac GPU against O2: A (last four prompt positions), B (16-token free run), D (16 teacher-forced steps, cosine ≥ 0.9999 each), red arms, a reset re-run |
| `gate_prefill.py` | the `prefill` function (64-token chunks, as the Swift engine splits a prompt) against `main` and O2 |
| `chunked.py` | runs a per-row script over the rows in sequential processes (the Python runtime leaks an IOSurface per call: one process died after about 10,000 S=1 calls, another 27 rows into a 33-row run) and merges the records |
| `gate_report.py` | merges the gate records and writes the gate table |
| `suite_coreai.py` | Core AI answers for the structured-output suite: tower → decoder, free-run greedy, no repetition penalty |
| `suite.py` | the suite's scoring, task by task (box IoU ≥ 0.75, exact count, point within 2 % of the side, OCR CER) and the bars |
| `mask/mask_graph.py` | the mask decoder as graph A (once per response, 16 query rows, key padding in the self-attention) and graph B (once per match); `box_fourier_np`, the host's box encoding in NumPy |
| `mask/mask_torch_check.py` | the two graphs in torch against the package before any export: alpha per instance, padding invariance, a red arm, activation absmax |
| `mask/mask_build.py` | exports graph A or B in fp32, fp16w32 or fp16 |
| `mask/mask_coreai_run.py` | the two graphs on Core AI against the package in fp32 torch (alpha MAE ≤ 0.01 and IoU(α > 0.5) ≥ 0.95 per instance) |
| `mask/mask_vision_taps.py` | `build`: the tower with the taps as a third output; `run`: its taps and the alpha through them against the oracle's |
| `mask/mask_hidden.py` | the decoder with `last_hidden` as a second output (and a two-layer toy of the same shape) |
| `mask/mask_hidden_run.py` | the `<OBJ>` hidden states on Core AI against the oracle's, teacher-forced and during generation |
| `mask/mask_e2e.py` | photo → tower → decoder → boxes and hidden states → graph A + B → alpha, on Core AI parts only, against the oracle's alpha |
| `mask/mask_host.py` | Pillow's BICUBIC resize written out in NumPy (22-bit fixed point, two passes) and the mask decoder's input tensors; `check` compares it with Pillow on every fixture image |
| `host/check_boxpe.py` | the Swift host's box encoding (`cutout-cli boxpe`) against torch's fp32 arithmetic for every width a box with integer per-mille corners can have |
| `host/check_fixture.py` | the Swift host's fixture outputs (`cutout-cli fixture`) against the oracle and the Python chain: input tensors, answers, the first divergence, alpha |
| `strip_stage.py` | the four graphs without debug locations in the Hugging Face `ios/` layout, checked (op counts, `coreai-build inspect`, `main.hash`, a byte scan for local paths) |

## Environment

- **Oracle:** Python 3.12 with the package's pins minus vLLM, Triton and Gradio (not needed for the transformers
  backend): torch 2.11.0, torchvision 0.26.0, transformers 5.12.1, huggingface-hub 1.21.0, safetensors 0.8.0,
  numpy 2.2.6, Pillow 12.2.0, accelerate 1.14.0, qwen_vl_utils 0.0.14. The package is imported by path
  (`--pkg <a Tencent/WeVisOne-VL checkout at 40f735b8>`, the directory holding `wevisone_vl/`).
- **Exports, AOT and gates:** the zoo's overlay venv (coreai-core 1.0.0b2, coreai-torch 0.4.1, coreai-opt 0.2.1,
  torch 2.9.0, transformers 4.57.6) and Xcode 27.0 RC as `DEVELOPER_DIR` (its Metal toolchain carries `coreai-build`).
- **Weights:** the snapshot at `3dad9ce8…` in the Hugging Face cache, with `refs/main` naming it (the exporter's
  loaders read the default branch; `export_wevisone.py` asserts it), and `HF_HUB_OFFLINE=1`. The mask decoder's weights
  are the snapshot's `mask_decoder.pt` (`--sidecar`).
- **Fixtures:** not in this repository. The gates ran on 33 rows over 17 images (two seeded synthetic drawings, four
  NASA public-domain photos, and eleven images from the authors' repository — crops of its case-study figures and its
  demo photo — which are not cleared for redistribution), plus four rows with several objects; the images stay in
  `<work>/fixtures/`.

## Run (from the repository root)

`<oracle-venv>` is the oracle's Python, `<venv>` the overlay venv's. `QW`, `QH`, `QY` and `WATCH` in the scripts' own
usage lines are this workspace's wrappers (wait for the Mac's measurement window, hold it, yield it, watch the scratch
space and the memory); the command after them runs as it is.

Oracle (CPU, one thread per process; a row took about 6 s in fp32 alone and 17 s with four processes at once, and bf16
about three times as long):

```bash
<oracle-venv> -I conversion/wevisone/make_fixtures.py --base <shared>/fixtures.json --package <WeVisOne-VL> --out <work>/fixtures
<oracle-venv> -I conversion/wevisone/mask/make_fixtures_r2.py --fixtures-dir <work>/fixtures --package <WeVisOne-VL>
python3 conversion/wevisone/oracle_batch.py --python <oracle-venv> --pkg <WeVisOne-VL> --model <snapshot> \
    --fixtures <work>/fixtures/fixtures.json --out <work>/oracle --logs <work>/logs --hidden-ids syn_a_det,nasa_jungle_det
python3 conversion/wevisone/oracle_index.py --oracle <work>/oracle --fixtures <work>/fixtures/fixtures.json --out <work>/oracle/oracle.json
python3 conversion/wevisone/suite.py bar --oracle <work>/oracle --fixtures <work>/fixtures/fixtures.json
<venv> conversion/wevisone/parity_torch_ladder.py --oracle <work>/oracle --fixtures <work>/fixtures/fixtures.json --grid 14 \
    --out <work>/oracle/parity_torch_ladder_g14.json --hidden-ids syn_a_det,nasa_jungle_det     # and --grid 16
```

The decoder forms and the gates (Mac GPU, Python runtime on the AOT assets, one bundle per process):

```bash
<venv> conversion/wevisone/export_wevisone.py dec-int8hu-pf64 --grid 14 --out-root <exports> --record <json> --oracle <work>/oracle
<venv> conversion/wevisone/export_wevisone.py vis-fp16 --grid 14 --out-root <exports> --record <json>
<venv> conversion/wevisone/aot_compile.py <exports>/g14/wevisone_vl_2b_decode_int8hu_pf64 --aot-dir <work>/aot/g14 --reshapes --record <json>
<venv> conversion/wevisone/aot_compile.py <exports>/g14/wevisone_vl_2b_vision --aot-dir <work>/aot/g14 --record <json>
python3 conversion/wevisone/chunked.py --python <venv> --script gate_decoder.py --fixtures <work>/fixtures/fixtures.json \
    --size 11 --out <work>/oracle/gate_decoder_int8hu_g14.json -- <exports>/g14/wevisone_vl_2b_decode_int8hu_pf64 \
    --aot <aimodelc> --oracle <work>/oracle --grid 14 --reload-every 0
<venv> conversion/wevisone/gate_prefill.py <exports>/g14/wevisone_vl_2b_decode_int8hu_pf64 --aot <aimodelc> \
    --oracle <work>/oracle --fixtures <work>/fixtures/fixtures.json --grid 14 --out <work>/oracle/gate_prefill_int8hu_g14.json
python3 conversion/wevisone/chunked.py --python <venv> --script suite_coreai.py --fixtures <work>/fixtures/fixtures.json \
    --size 11 --out <work>/oracle/suite_coreai_g14.json -- <decoder bundle> --aot <decoder aimodelc> \
    --vision-aot <tower aimodelc> --oracle <work>/oracle --grid 14
python3 conversion/wevisone/gate_report.py --gate int8hu_g14=<json> --ladder g14=<json> --out <json> --md <work>/GATE.md
python3 conversion/wevisone/suite.py tables --oracle <work>/oracle --fixtures <work>/fixtures/fixtures.json \
    --coreai g14=<work>/oracle/suite_coreai_g14.json --out <json> --md <work>/SUITE.md
```

The mask path (`<pt>` = the snapshot's `mask_decoder.pt`; the exports' AOT assets by `aot_compile.py`):

```bash
<venv> conversion/wevisone/mask/mask_torch_check.py --pkg <WeVisOne-VL> --sidecar <pt> --answers <work>/oracle --out <json>
<venv> conversion/wevisone/mask/mask_build.py a --grid 28 --host-fourier --precision fp16 --pkg <WeVisOne-VL> --sidecar <pt> \
    --out-root <exports> --record <json>
<venv> conversion/wevisone/mask/mask_build.py b --precision fp16 --pkg <WeVisOne-VL> --sidecar <pt> --out-root <exports> --record <json>
<venv> conversion/wevisone/mask/mask_vision_taps.py build --grid 14 --out-root <exports> --record <json>
<venv> conversion/wevisone/mask/mask_hidden.py int8hu-pf64 --out-root <exports>/hidden --record <json>
<venv> conversion/wevisone/mask/mask_coreai_run.py --a28 <A aimodelc> --b <B aimodelc> --pkg <WeVisOne-VL> --sidecar <pt> \
    --answers <work>/oracle --forms O2-448 --out <json>
<venv> conversion/wevisone/mask/mask_vision_taps.py run --grid 14 --taps-aot <aimodelc> --r1-aot <the two-output tower aimodelc> \
    --pkg <WeVisOne-VL> --sidecar <pt> --answers <work>/oracle --out <json>
<venv> conversion/wevisone/mask/mask_hidden_run.py <exports>/hidden/int8hu-pf64/wevisone_vl_2b_decode_int8hu_pf64 --aot <aimodelc> \
    --r1-aot <the one-output decoder aimodelc> --answers <work>/oracle --grid 14 --out <json>
<venv> conversion/wevisone/mask/mask_e2e.py --vision <aimodelc> --dec-bundle <bundle> --dec <aimodelc> --a <aimodelc> --b <aimodelc> \
    --pkg <WeVisOne-VL> --answers <work>/oracle --fixtures <work>/fixtures/fixtures.json --fixtures <work>/fixtures/fixtures_r2.json --out <json>
<venv> -I conversion/wevisone/mask/mask_host.py check --fixtures <work>/fixtures/fixtures.json --fixtures <work>/fixtures/fixtures_r2.json --out <json>
```

The Swift host (CoreAIKit `Examples/Cutout`, `swift build -c release`; `cutout-cli` runs the app's request code on a
Mac against the AOT folder):

```bash
cutout-cli boxpe --out <dir>
<venv> -I conversion/wevisone/host/check_boxpe.py <dir> <json>
cutout-cli fixture --models <model folder> --rows <rows.json> --out <dir> --forms pad-fp16,s1-fp32
<venv> -I conversion/wevisone/host/check_fixture.py --fixture <dir> --forms s1-fp32,pad-fp16 --oracle <work>/oracle/O2-448 \
    --r2 <mask_e2e json> --rows <rows.json> --out <json>
```

### Stage

```bash
DEVELOPER_DIR=<Xcode 27 RC>/Contents/Developer <venv> -I -B conversion/wevisone/strip_stage.py \
    --exports <exports> --out <stage>/ios --record <work>/strip_stage.json
cd <stage>/ios
xcrun coreai-build compile wevisone_vl_2b_vision_taps_g14.aimodel --output ../macos --platform macOS --preferred-compute gpu --architecture h16c
xcrun coreai-build compile wv_mask_a_g28_n16_fp16_hf.aimodel --output ../macos --platform macOS --preferred-compute gpu --architecture h16c
xcrun coreai-build compile wv_mask_b_fp16.aimodel --output ../macos --platform macOS --preferred-compute gpu --architecture h16c
mkdir -p ../macos/wevisone_vl_2b_decode_int8hu_pf64
xcrun coreai-build compile wevisone_vl_2b_decode_int8hu_pf64/wevisone_vl_2b_decode_int8hu_pf64.aimodel \
    --output ../macos/wevisone_vl_2b_decode_int8hu_pf64 --platform macOS --preferred-compute gpu --architecture h16c --expect-frequent-reshapes
cp -c -R wevisone_vl_2b_decode_int8hu_pf64/metadata.json wevisone_vl_2b_decode_int8hu_pf64/tokenizer ../macos/wevisone_vl_2b_decode_int8hu_pf64/
```

The compile is run from inside the stripped folder with relative inputs; then a byte scan of `macos/` for local paths
(`LC_ALL=C grep -a -l -F -r` with the home directories' root, `/private/var/`, `/Volumes/` and `/var/folders/` as
fixed strings; a text-only grep skips the binary files) finds none. The
four compiles peaked at 34.4 GB of resident memory on an M4 Max, the decoder's taking about 140 s; compile it alone.

## What to watch for

- Load an AOT asset once per process. Each rebuild of an ahead-of-time asset in the same Python process kept about
  5.5–7 GB that never came back; 54 rebuilds in one gate run reached 110 GB resident, and the disk's free space went to
  zero before the process was killed. Without rebuilds, 300 calls moved the resident size from 9.51 to 9.60 GB. Pass
  `--reload-every 0` to `gate_decoder.py`: its default, 16, is the JIT-era advice to rebuild every 16 calls, and every
  recorded gate ran with 0.
- The mask decoder's box encoding (`FourierPE1D`, 2π·v·2^k for k up to 31) turns one fp32 ulp of an input into
  unrelated high-k features: graph A takes it as an input (`--host-fourier`), and the host computes it in the CPU's
  fp32 order, with torch's values for the 7 box widths where `logf` / `np.log` round differently (`mask/mask_graph.py`
  `box_fourier_np` does not carry them; the Swift host does).
- The tokenizer files are the checkpoint's, byte for byte; transformers 4.57 cannot read its `tokenizer_config.json`.
