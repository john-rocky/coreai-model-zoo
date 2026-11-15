# clef_flash — clef-flash on Core AI

Export and gate scripts for [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash)
(revision `17f0b0ad64efb65d273590632833508766b2aae6`, Apache-2.0). Card:
[`models/clef-flash/README.md`](../../models/clef-flash/README.md); port notes:
[`knowledge/clef-flash-port.md`](../../knowledge/clef-flash-port.md); Swift host:
[`apps/ClefFlash`](../../apps/ClefFlash/).

The model answers typed questions (`noul`, `choice`, `score`) about a state (text, JSON or an image) in
one prefill: a Qwen3.5-9B backbone returns its final-norm hidden state at every position and a
121.8M-parameter joint schema head scores every option of every question. Requests and responses use
the SystemOne-compatible request shape. The port is a decoder graph that returns hidden rows, a
fixed-grid tower, the head as a graph and the untied `lm_head.weight` as a host table.

| file | what it does |
|---|---|
| `make_fixtures.py` | writes the fixture: 16 text states (3 of about 2,000 tokens), 12 JSON states, 10 images drawn with Pillow (CC0-1.0), 4 CC0 photographs (`--fetch-photos`, sha256-pinned), SemIf authored144 as 144 `choice` questions (MIT); `--heldout` writes the held-out set with a novelty check against the fixture |
| `oracle_clef.py` | the checkpoint's own `joint_schema_model.py` (path import, sha256 asserted) in fp32 on the CPU: `load_release_model()` + `systemone()` per record and image arm, with hooks; asserts the answers, the head re-run, the spans and the M-RoPE planes on every run |
| `grid_price.py` | what a fixed square grid costs against the processor's own grid, with the author's code alone |
| `host.py` | the host spec, NumPy + Pillow + the tokenizer: request → ids, spans, option order, the decoder's static inputs; image → tower patches |
| `test_host.py` | `host.py` against every oracle run: ids, image offset, spans, option order, static inputs, M-RoPE planes, `pixel_values` |
| `gate_tower.py` | the fixed-grid tower fed the host's patches: fp32 torch (`--stages torch`) and the AOT fp16w32 asset (`--stages aot`) against the oracle's tower rows |
| `qwen3_5_clef_decoder.py` | the decoder module: decider-2b-vision's ids-input Qwen3.5 VL decoder at 9B, no vocabulary head, the final-norm hidden state at every position, a 1,024-row image buffer |
| `parity_decoder_torch.py` | that module in fp32 torch against the oracle through the author's head: P1 every run, P2 the overlay's plain text decoder, P3 per layer, P4 red arms, P5 chunk widths, CPU vs MPS |
| `export_decoder.py` | the decoder bundle: `fp16`, `int8lin`, `int8mix --fp16-layers`; one function `main` at a static `--prefill-chunk` S; `--variant fp16attn32` (diagnostic); `--aot` the h16c `.aimodelc`; `--record` a JSON log |
| `readout_gate.py` | the decoder alone on the Mac GPU: the oracle's ids and fp32 image rows in, the fp16 hidden through the author's fp32 head, against the oracle; `merge`, `--compare-with`, `--subset`, `--oracle` |
| `int8_bisect_torch.py` | which layers carry the int8 error, in fp32 torch with the exporter's own int8 weights; `rule` writes the selection rule before any result |
| `clef_head.py` | `ClefHeadGraph`, the author's head for one record from ten tensors; `head_inputs()` (the host arrays), `lexical_rows()`, `question_probs()` |
| `export_head.py` | the head asset: `--shape bucket` (shipped: `t512` / `t1024` / `t2048` / `t4096`, Q 16, O 128) or `--shape dynamic` (measured, not shipped); `--weights fp16w32`; `--aot` |
| `gate_head.py` | `h0` the matrix form vs the author's head (eager); `h1` the AOT head graph on the oracle's hidden; `h2` a readout's fp16 hidden → host arrays → head graph, `--red` |
| `export_lm_head_table.py` | the untied `lm_head.weight` as raw fp16 for the host's gather, read back through a second path |
| `decide.py` | the Python reference read-out: `run` (request + image → response) and `check` (fixture requests vs the author's `systemone()` responses) |
| `gate_swift.py` | scores the Swift CLI (`apps/ClefFlash`): `prep`, `render`, `pixels`, `score`, `towers`, `negative`, `timing` |

## Environment

- **Export and gates:** the zoo's overlay venv (`conversion/overlay/`; coreai-core 1.0.0b2, coreai-torch 0.4.1,
  torch 2.9.0, transformers 4.57.6), called `$PY` below; Xcode 27.0 RC as `DEVELOPER_DIR`.
- **The author's code** (`make_fixtures.py`, `oracle_clef.py`, `grid_price.py`): each declares its dependencies
  (PEP 723), so `uv run` builds the environment; the gated runs used an equivalent venv (Python 3.12.11, torch
  2.9.0, torchvision 0.24.0, transformers 5.17.0, pillow 12.3.0). Without torchvision, transformers 5.17 switches
  the image processor to its PIL backend.
- **Weights:** the pinned snapshot in `HF_HOME` (the scripts default it to `$ZOO_WORK_ROOT/_clefflash/hf`), with
  `HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_XET=1`. Everything the scripts write outside the repository goes under
  `$ZOO_WORK_ROOT/_clefflash/` (`conversion/_paths.py`).

Run from `conversion/clef_flash`. `<work>` below is `$ZOO_WORK_ROOT/_clefflash`, `R=<work>/results`,
`G=<work>/logs`, `B=<work>/exports/bundles`, `H=<work>/exports/head/clef_flash_head_bucket_fp16w32`.

## 1. Oracle and fixture

```bash
uv run make_fixtures.py --fetch-photos           # once; then offline. -> <work>/fixtures
uv run oracle_clef.py                            # 214 runs, about 47 min on an M4 Max CPU -> <work>/oracle, <work>/results
uv run oracle_clef.py --arms g672,g896 --tag _large --independent img_01:g672,photo_01:g896
uv run grid_price.py --arms native,g256,g448,g672,g896 --out $R/grid_price_large.json
```

The oracle asserts, per run, that `systemone()`'s answers equal the hooked logits' answers, that the head
re-run on the hooked hidden state reproduces the logits, that the ids and spans equal a separate
`encode_record()`, that every span decodes to its rendered text and that the rotary planes equal the closed
form; five runs go again through an independent `model(collate_records(...))` call and must be bit-identical.
The image block starts at index 36 (the prefix's length) on every record. `photo_01` at `native` (1,200 image
rows) is outside the decoder gates: the graph's buffer holds 1,024.

## 2. Host and the decoder module in fp32 torch

```bash
$PY test_host.py                                                  # -> $R/test_host.json
$PY parity_decoder_torch.py --stages device                       # 5 runs on CPU and on MPS
$PY parity_decoder_torch.py --stages p1,p4,p5 --device cpu --conv-check-every 8
$PY parity_decoder_torch.py --stages p2                           # the overlay's plain text decoder
$PY parity_decoder_torch.py --merge --p1-device cpu               # -> $R/parity_decoder.json
```

The CPU path computes the depthwise conv as one `bmm` over the windows and checks it against `F.conv1d`
(`--conv-check-every`). The transcript is `models/clef-flash/gate-clef-flash-torch-parity.json`.

## 3. Towers

```bash
python ../export_qwen38vl_pipelined.py --hf-id Cloudflare/clef-flash --name clef_flash_g256 \
    --grid-h 8 --grid-w 8 --skip-decoder --vision-dtype fp16w32 --out-dir <work>/exports
python ../export_qwen38vl_pipelined.py --hf-id Cloudflare/clef-flash --name clef_flash_g448 \
    --grid-h 14 --grid-w 14 --skip-decoder --vision-dtype fp16w32 --out-dir <work>/exports
xcrun coreai-build compile <work>/exports/clef_flash_g256_vision_fp16w32/clef_flash_g256_vision_fp16w32.aimodel \
    --output <work>/exports/clef_flash_g256_vision_fp16w32_aotc --platform macOS --preferred-compute gpu --architecture h16c
#   (the same for g448)
$PY gate_tower.py --stages torch                                   # g256, g448 -> $R/tower_torch.json
$PY gate_tower.py --stages torch --arms g672,g896                  # the two larger grids, added to the same file
$PY gate_tower.py --stages aot                                     # -> $R/tower_aot.json
```

## 4. Decoder and the readout gate

```bash
$PY export_decoder.py fp16 --prefill-chunk 64 --aot --record $R/export_fp16_pf64.json      # the default bundle
$PY readout_gate.py run $B/clef_flash_decode_fp16_pf64 --subset s64 --red \
    --compare-with $R/readout_fp16_pf16.json --transcript $R/readout_fp16_pf64.json
$PY readout_gate.py run $B/clef_flash_decode_fp16_pf64 --subset s64rest --tag fp16_pf64_rest \
    --transcript $R/readout_fp16_pf64_rest.json
$PY readout_gate.py merge $R/readout_fp16_pf64.json $R/readout_fp16_pf64_rest.json \
    --compare-with $R/readout_fp16_pf16.json --transcript $R/readout_fp16_pf64_full.json
```

The bundle lands in `<work>/exports/bundles/<name>/`, the AOT asset in `<work>/exports/bundles_aotc/`. The S = 16
reference of the chunk-width comparison is `export_decoder.py fp16 --prefill-chunk 16 --aot` and `readout_gate.py
run $B/clef_flash_decode_fp16_pf16 --red`. Each gate process takes at most 40 runs plus a re-run of its first,
which must reproduce its hidden rows bit for bit (the Python runtime leaks one IOSurface per call). The GPU is
not locked, so the recorded times are contended. Loading an AOT asset in the Python runtime leaves an entry the
size of the asset in `~/Library/Caches/coreai-cache/<build>/python/<sha256 of main-h16c.mlirb>`; delete it after
the gate. Transcripts: `gate-clef-flash-readout-fp16_pf64.json`, `-readout-fp16_pf16.json`.

## 5. int8, and how the fp16 layers were chosen

```bash
$PY export_decoder.py int8lin --prefill-chunk 64 --aot --record $G/r4_export_int8lin_pf64.json
$PY readout_gate.py run $B/clef_flash_decode_int8lin_pf64 --red --compare-with $R/readout_fp16_pf64_full.json \
    --transcript $R/readout_int8lin_pf64.json                      # FAILS the bar (max |dp| 0.0748)
$PY int8_bisect_torch.py dump
$PY int8_bisect_torch.py rule --runs semif_1105577da4c8dad4609d:text,semif_86a15618ba4cb90a307e:text,\
semif_fb73a85d386cf59b0c51:text,own_j12:text,semif_8a4c3de28be95c105ca1:text,img_06:native --why "..."
$PY int8_bisect_torch.py run --part <work>/bisect/parts/part_a.json --candidates
$PY int8_bisect_torch.py run --part <work>/bisect/parts/part_b.json --informative \
    --sets "0,1,2,3,4,5,6,7,8;0,1,2,3,4,5,6,7,8,9;0,1,2,3,4,5,6,7,8,9,10"
$PY int8_bisect_torch.py merge --out $R/int8_bisect.json
$PY export_decoder.py int8mix --fp16-layers 0,1,2,3,4,5,6,7,8,9,10,11 --prefill-chunk 64 --aot \
    --record $G/r4_export_int8mix_pf64.json
$PY readout_gate.py run $B/clef_flash_decode_int8mix_pf64 --red --compare-with $R/readout_fp16_pf64_full.json \
    --transcript $R/readout_int8mix_pf64.json
```

Rule, written by `rule` before any bisect result: worst run ≤ 0.010, no run worse than its int8lin value, at most
six fp16 layers, the smallest set. It chose none; layers 0–11 come from the informative map (`--informative`,
outside the rule), the smallest contiguous fp16 prefix meeting the first two conditions. The held-out set (step 7)
is its test. Two diagnostics ran the same way: `export_decoder.py fp16 --variant fp16attn32 --prefill-chunk 64
--aot` (fp32 attention) and `export_decoder.py fp16 --prefill-chunk 128 --aot`, each through `readout_gate.py run
--subset list --runs-file $R/r4_runs_<…>.json --compare-with $R/readout_fp16_pf64_full.json`. Transcripts:
`gate-clef-flash-readout-int8lin_pf64.json`, `-readout-int8mix_pf64.json`, `-int8-bisect.json`, `-variants.json`.

## 6. The head and the lm_head table

```bash
$PY export_lm_head_table.py                                        # -> <work>/exports/host/lm_head_fp16.bin + .json
$PY gate_head.py h0 --out $R/head_h0.json
$PY export_head.py --shape bucket --weights fp16w32 --aot --record $G/r5_export_head_bucket.json
$PY gate_head.py h1 --head $H --out $R/head_h1.json
$PY gate_head.py h2 --head $H --readout $R/readout_fp16_pf64_full.json --red --out $R/head_h2_fixture.json
$PY decide.py check --runs own_t01:text,own_t03:text,own_t15:text,semif_a3f18f3a63d45345942b:text,own_j01:text,\
own_j08:text,own_j12:text,img_04:g448,img_06:g448,photo_02:g448,img_01:g256,img_08:g256 --out $R/decide_e2e_fixture.json
$PY decide.py run --request req.json [--image x.png --grid 448] --out resp.json [--trace trace.json]
```

`export_head.py --shape dynamic --weights fp16w32 --q-min 1 --aot` builds the dynamic-shape asset; with the AOT
flags above plus `--expect-frequent-reshapes` it aborts at its first call (`GPUMemrefOps.mm:164: failed assertion
'Failed to resolve dynamic dimensions for memref.alloc'`), so the bucket form ships. Above 2,048 keys `clef_head.py`
computes attention in key blocks (`KEY_CHUNK`): the plain chain compiled for the GPU is wrong from 4,032 keys.
`decide.py` loads AOT assets only: the decoder's `.aimodelc` (`--decoder-aot`), the towers' (`--tower-g256`,
`--tower-g448`) and the head's, which it reads from the head folder's `metadata.json` (`aot.aimodelc`).
Transcript: `gate-clef-flash-head.json`.

## 7. Held out

```bash
uv run make_fixtures.py --heldout --fetch-photos                  # -> <work>/fixtures/heldout
uv run oracle_clef.py --fixtures <work>/fixtures/heldout --out-dir <work>/oracle/heldout \
    --results-dir <work>/oracle/heldout --arms text,g256,g448 --independent ho_t01:text,ho_j02:text,ho_i01:g256,ho_p01:g448
$PY readout_gate.py run $B/clef_flash_decode_fp16_pf64 --oracle <work>/oracle/heldout --tag heldout_fp16_pf64 \
    --transcript $R/readout_heldout_fp16_pf64.json
$PY readout_gate.py run $B/clef_flash_decode_int8mix_pf64 --oracle <work>/oracle/heldout --tag heldout_int8mix_pf64 \
    --compare-with $R/readout_heldout_fp16_pf64.json --transcript $R/readout_heldout_int8mix_pf64.json
$PY gate_head.py h1 --head $H --oracle <work>/oracle/heldout --tag head_h1_heldout --out $R/head_h1_heldout.json
$PY gate_head.py h2 --head $H --oracle <work>/oracle/heldout --readout $R/readout_heldout_fp16_pf64.json \
    --tag head_h2_heldout --out $R/head_h2_heldout.json
```

The held-out set was written after the int8 layers were chosen and is not used to choose anything. Without
`--arms text,g256,g448`, `--results-dir` and `--independent`, `oracle_clef.py` would skip the text records,
overwrite the fixture's summary files and look for fixture record ids. Transcript: `gate-clef-flash-heldout.json`.

## 8. Swift

```bash
swift build -c release --package-path ../../apps/ClefFlash --scratch-path <work>/swift/.build   # DEVELOPER_DIR = Xcode 27 RC
BIN=<work>/swift/.build/release/clef-flash
$BIN check-ids --records <work>/fixtures/records.json --expected <work>/oracle/records_oracle.json \
    --tokenizer $B/clef_flash_decode_fp16_pf64/tokenizer --out <work>/swift/checks/check_ids_fixture.json
$PY gate_swift.py prep && $PY gate_swift.py render && $PY gate_swift.py pixels
$BIN fixture --assets <work>/exports --bundle-name clef_flash_decode_fp16_pf64 --decoder-asset aot \
    --records <work>/fixtures/records.json --images <work>/fixtures/images --arms text,g256,g448,native \
    --embeds <work>/swift/embeds/fixture/manifest.json --embeds-arms native \
    --dump-dir <work>/swift/gate/fp16_aot/fixture_dump --dump tiles,embeds,hidden --out <work>/swift/gate/fp16_aot/fixture.json
#   the same for fixtures/heldout -> heldout.json; --arms g256,g448 --embeds-arms g256,g448 -> *_rows.json;
#   --decoder-asset jit; --bundle-name clef_flash_decode_int8mix_pf64
$PY gate_swift.py score --asset fp16_aot --fixture .../fixture.json --heldout .../heldout.json \
    --oracle-rows .../fixture_rows.json .../heldout_rows.json --transcript $R/swift_gate_fp16_aot.json
$PY gate_swift.py score --asset fp16_jit --fixture ... --heldout ... --same-as <fp16_aot passes> --transcript ...
$PY gate_swift.py towers --passes .../fp16_aot/fixture.json .../fp16_aot/heldout.json
$BIN ask --assets <work>/exports --request req.json --out resp.json --trace trace.json
$PY gate_swift.py negative --base own_t01.trace.json --changed one_word.trace.json
../../apps/ClefFlash/_time_mac.sh && $PY gate_swift.py timing --run-dir <work>/swift/timing/<run id>
```

`--decoder-asset aot` loads the AOT assets with `SpecializationOptions.default`, as the Python gates do;
`--decoder-asset jit` specializes the `.aimodel` files (the decoder GPU-preferred with `expectFrequentReshapes`,
the head and the towers GPU-preferred). `_time_mac.sh` takes the machine-wide GPU lock before it times anything.
The runtime's cache for the Swift CLI is `~/Library/Caches/coreai-cache/<build>/clef-flash/`. Transcripts:
`gate-clef-flash-swift.json`, `gate-clef-flash-timing.json`.

## The published transcripts

`models/clef-flash/fixtures-clef-flash.json` and the twelve `gate-clef-flash-*.json` files were assembled from the
local files above: each names its source files with their bytes and sha256 and lists what it leaves out
(`trimmed`). The int8mix JIT-vs-AOT comparison in `gate-clef-flash-swift.json` (`jit_vs_aot`) was computed then from
the Swift pass files; nothing was re-run.
