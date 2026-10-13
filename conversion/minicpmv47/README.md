# minicpmv47 — MiniCPM-V-4.7-1B on Core AI

Scripts for [openbmb/MiniCPM-V-4.7-1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B)
(revision `984a89aafcd5160164f4f0d04b4be0f4eec03cb8`). The text backbone is the overlay's Qwen3.5 decoder and
the vision tower is MiniCPM-V-4.6's code; what 4.7 adds is canvas M-RoPE, which the decoder module derives
in-graph.

| file | what it does |
|---|---|
| `make_fixtures.py` | the oracle's images: one drawn sign (Pillow primitives, no font file) and two NASA photographs, each resized to 448x448 BICUBIC |
| `oracle_minicpmv47.py` | transformers' native `minicpmv4_7` in fp32 on the CPU, one thread: ids, pixel values, tower output, logits, and the M-RoPE planes the text rotary received (hooks), for P0 (text) and P1–P4 (image) |
| `parity_text.py` | the overlay's `Qwen3_5ForCausalLM` (hand-written config, untied head) against the oracle on P0 |
| `minicpmv47_vl_pipelined.py` | the decoder module: ids in, 64 image rows and `rope_shift_start / amount` as static inputs, canvas M-RoPE in the graph; `host_static_inputs` is the host side |
| `parity_vl_torch.py` | that module in fp32 torch against the oracle: planes on every position and generate step, two red arms, last-position logits |
| `minicpmv47_vision.py` | the vision tower re-authored from the checkpoint (SigLIP 27 layers + window merger after layer 6 + DownsampleMLP merger): `pixel_values [1,3,448,448] -> image_features [64,1024]`; a reference path in HF's order and a baked path (`bake_constants(32)`, int32 index buffers, one batched window attention) for export; `VisionExport` is the export wrapper |
| `dump_vision_stages.py` | the HF tower's stages on one prompt (embeddings, layers 0–6, window merger, layer 26, post-LN, merger) and transformers' own position ids / window index, for bisecting |
| `parity_vision.py` | the tower against the oracle's `image_features` on P1–P4, both paths, stage by stage on P1, and the reference path minus the baked path |
| `export_dryrun_vision.py` | `torch.export.export` of the baked wrapper (the call `export_to_coreai` makes; no conversion): op counts, forbidden data-dependent ops asserted absent |
| `parity_e2e_torch.py` | the tower's rows through the decoder module: last-position logits, greedy on the decode states, planes |
| `fp16_precheck.py` | activation absmax per stage of the tower and the decoder (and norm-input sums of squares) before an fp16 export; the tower in CPU fp16 / fp16w32 for scale |
| `host.py` | host preprocess (Pillow BICUBIC 448, (x/255 − 0.5)/0.5, CHW) and the decoder's static inputs; run as a script it gates the pixels against the oracle |
| `oracle_margins.py` | the oracle's top-1 and softmax top-2 gap at every generated step (one teacher-forced fp32 forward per prompt) — the knife-edge rule of `conversion/coreai_gate.py` needs it past step 1 |
| `export_decoder.py` | the decoder to a Core AI bundle: S=1 `main`, `fp16` or `int8hu` (body int8 per-block-32 + the untied head int8 per-block-32 absmax), `metadata.json` with the host contract, the tokenizer files verbatim |
| `export_vision.py` | the tower to a Core AI bundle in three forms: `fp16`, `fp16w32` (fp16 storage, fp32 math and I/O), `int8lin` (nn.Linear int8 per-block-16) |
| `aot_compile.py` | `coreai-build compile … --platform macOS --preferred-compute gpu --architecture h16c` (+ `--expect-frequent-reshapes` for the decoder) and a record: bytes, wall, ANE regions, `main.hash` |
| `gate_common.py` | the engine-gate pieces: S=1 decoder driver on fresh states, first-divergence rule, GPU-utilization sampler, window record |
| `gate_decoder.py` | G1 (P0 teacher-forced 48 positions + free-run) and G2 (P1–P4 free-run) on one decoder `.aimodelc`; red arms (amount+1, image rows zeroed, 1-D planes); reset re-run |
| `gate_vision.py` | G3: one tower form against the oracle's rows on P1–P4 (per-token cos), cold / warm ms — run inside a quiet window |
| `gate_e2e.py` | G4: image file → tower → decoder → answer on P1–P4, judged like G2 |
| `timing_decoder.py` | decoder ms per call (one process per bundle, or `--interleave` in one process) and the towers interleaved, with the window record |

## Environment

- **Oracle and fixtures:** Python 3.12, transformers 5.19.0, torch 2.14.1, torchvision 0.29.1, Pillow 12.3.0
  (`uv venv` + `uv pip install transformers==5.19.0 torch torchvision pillow safetensors numpy huggingface_hub`).
- **Parity:** the zoo's overlay venv (coreai-core 1.0.0b2, torch 2.9.0, transformers 4.57.6).
- **Weights:** the pinned snapshot in the Hugging Face cache, `HF_HUB_OFFLINE=1`. Outputs go to
  `$ZOO_WORK_ROOT/_minicpmv47/oracle/` (`conversion/_paths.py`).

## Run (from the repository root)

```bash
<oracle-venv>/bin/python -I conversion/minicpmv47/make_fixtures.py       # -> <lane>/oracle/fixtures
<oracle-venv>/bin/python -I conversion/minicpmv47/oracle_minicpmv47.py   # -> <lane>/oracle/oracle.{npz,json}
../coreai-models/.venv/bin/python conversion/minicpmv47/parity_text.py      # -> <lane>/oracle/parity_text.json
../coreai-models/.venv/bin/python conversion/minicpmv47/parity_vl_torch.py  # -> <lane>/oracle/parity_vl_torch.json
<oracle-venv>/bin/python -I conversion/minicpmv47/dump_vision_stages.py           # -> <lane>/oracle/vision_stages.{npz,json}
../coreai-models/.venv/bin/python -I conversion/minicpmv47/parity_vision.py        # -> <lane>/oracle/parity_vision.json
../coreai-models/.venv/bin/python -I conversion/minicpmv47/export_dryrun_vision.py # -> <lane>/oracle/vision_export_ops.txt
../coreai-models/.venv/bin/python -I conversion/minicpmv47/parity_e2e_torch.py     # -> <lane>/oracle/parity_e2e_torch.json
../coreai-models/.venv/bin/python -I conversion/minicpmv47/fp16_precheck.py        # -> <lane>/oracle/fp16_precheck.json
../coreai-models/.venv/bin/python -I conversion/minicpmv47/host.py                 # -> <lane>/oracle/host_preprocess.json
```

All run on one CPU thread (`--threads 1` is the default); each takes under a minute on an Apple silicon Mac
except `fp16_precheck.py` (about 70 s: the CPU fp16 tower is ~13 s per image). `dump_vision_stages.py`
must run before `parity_vision.py`, which reads its stages.

Export, AOT and the Mac GPU gates (bundles go to `$ZOO_EXPORTS/minicpmv47_*`, AOT assets to
`$ZOO_EXPORTS/minicpmv47_aotc/`; `DEVELOPER_DIR` = an Xcode 27 whose Metal toolchain carries `coreai-build`):

```bash
<oracle-venv>/bin/python -I conversion/minicpmv47/oracle_margins.py                   # -> <lane>/oracle/margins.{npz,json}
../coreai-models/.venv/bin/python -I -B conversion/minicpmv47/export_decoder.py int8hu --record <lane>/r3/export_decoder_int8hu.json
../coreai-models/.venv/bin/python -I -B conversion/minicpmv47/export_decoder.py fp16   --record <lane>/r3/export_decoder_fp16.json
../coreai-models/.venv/bin/python -I -B conversion/minicpmv47/export_vision.py int8lin --record <lane>/r3/export_vision_int8lin.json   # also fp16w32, fp16
../coreai-models/.venv/bin/python -I -B conversion/minicpmv47/aot_compile.py $ZOO_EXPORTS/minicpmv47_vlm_decode_int8hu --reshapes --record <lane>/r3/aot_minicpmv47_vlm_decode_int8hu.json
../coreai-models/.venv/bin/python -I -B conversion/minicpmv47/aot_compile.py $ZOO_EXPORTS/minicpmv47_vision_int8lin --record <lane>/r3/aot_minicpmv47_vision_int8lin.json
../coreai-models/.venv/bin/python -I -B conversion/minicpmv47/gate_decoder.py $ZOO_EXPORTS/minicpmv47_vlm_decode_int8hu --text --vl --red --out <lane>/r3/gate_decoder_int8hu.json
../coreai-models/.venv/bin/python -I -B conversion/minicpmv47/gate_vision.py $ZOO_EXPORTS/minicpmv47_vision_int8lin --out <lane>/r3/gate_vision_int8lin.json
../coreai-models/.venv/bin/python -I -B conversion/minicpmv47/gate_e2e.py $ZOO_EXPORTS/minicpmv47_vision_int8lin $ZOO_EXPORTS/minicpmv47_vlm_decode_int8hu --out <lane>/r3/gate_e2e_int8lin.json
../coreai-models/.venv/bin/python -I -B conversion/minicpmv47/timing_decoder.py --interleave --bundles <int8hu>,<fp16> --towers <int8lin>,<fp16w32>,<fp16> --out <lane>/r3/timing_run3.json
```

On the M4 Max each export takes 4–41 s and each AOT 3–37 s; a decoder gate ~20 s, a tower gate ~6 s. The gates load
only the AOT `.aimodelc` with `SpecializationOptions.default()` (the python runtime's JIT on the GPU is not used), one
bundle per process (the runtime leaks an IOSurface per call; each gate stays under 15 sequences).
