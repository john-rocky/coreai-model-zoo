# decider_vision — decider-2b-vision on Core AI

Export and gate scripts for [Mapika/decider-2b-vision](https://huggingface.co/Mapika/decider-2b-vision)
(revision `863e290863655f1d6b69324d77d09ac972d21609`). Card:
[`models/decider-2b-vision/README.md`](../../models/decider-2b-vision/README.md); port notes:
[`knowledge/decider-2b-vision-port.md`](../../knowledge/decider-2b-vision-port.md).

| file | what it does |
|---|---|
| `make_fixture_images.py` | draws the fixture images from Pillow primitives (CC0-1.0) and writes `rows.json` / `meta.json` |
| `oracle_decider_vision.py` | the author's own `decider/vision.py` in fp32 on the CPU: ids, slots, M-RoPE planes, tower output and letter probabilities per run |
| `grid_price.py` | what a fixed 256×256 / 448×448 input costs against the processor's own grid, with the author's code alone |
| `host.py` | the host spec: Pillow-order bicubic, patches, the author's prompt, the slot rule, `rope_shift_start / amount` |
| `test_host.py` | `host.build_ids` against the oracle's ids, slots and planes on every run |
| `gate_tower.py` | host patches and the tower (fp32 torch, AOT fp16 / fp32 / fp16w32) against the author's fp32 tower |
| `qwen3_5_vl_pipelined.py` | the decoder module: Qwen3.5 hybrid, ids in, image rows as a static input, M-RoPE in the graph |
| `parity_decoder_torch.py` | that module in fp32 torch against the oracle (P1–P4) |
| `export_decoder.py` | the decoder LanguageBundle (`fp16`, `int8hu`, `int8lin`, `int8mix`; `--prefill-chunk 16` adds the `prefill` function; `--aot` the h16c `.aimodelc`) |
| `readout_gate_vision.py` | b1: the decoder alone on all 111 runs; b2: image file → host → tower → decoder on the 70 image runs |
| `int8_bisect_torch.py` | which layers carry the int8 error, in fp32 torch with the exporter's own int8 weights |
| `heldout_prepare.py`, `heldout_eval.py` | the 500 held-out photo runs: inputs from the author's processor, then the shipped path against the author's probabilities |
| `gate_swift.py` | scores the Swift CLI (`apps/DeciderVision`) against the oracle, the Python host and the Python read-out |

## Environment

- **Export and gates:** the zoo's overlay venv (`conversion/overlay/`; coreai-core 1.0.0b2, coreai-torch 0.4.1,
  coreai-opt 0.2.1, torch 2.9.0, transformers 4.57.6), Xcode 27.0 RC as `DEVELOPER_DIR`.
- **The author's code** (`oracle_decider_vision.py`, `grid_price.py`, `heldout_prepare.py`): transformers 5.17.0.
  Each script declares its dependencies (PEP 723), so `uv run` builds the environment; the gated runs used an
  equivalent venv (Python 3.12.11, torch 2.9.0, transformers 5.17.0).
- **Weights:** the pinned snapshot in `HF_HOME` (the scripts default it to `$ZOO_WORK_ROOT/_decider2bv/hf`), with
  `HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_XET=1`. Everything the scripts write outside the repository goes under
  `$ZOO_WORK_ROOT/_decider2bv/` (`conversion/_paths.py`).

Run everything from the repository root. `<lane>` below is `$ZOO_WORK_ROOT/_decider2bv`.

## 1. Oracle and fixture

```bash
python3 conversion/decider_vision/make_fixture_images.py                  # -> <lane>/fixtures
uv run conversion/decider_vision/oracle_decider_vision.py \
    --public-json models/decider-2b-vision/fixtures-decider-2b-vision.json  # -> <lane>/oracle + the public fixture
uv run conversion/decider_vision/grid_price.py                            # -> <lane>/price (The Cauldron shards, MPS fp32)
python conversion/decider_vision/test_host.py                             # host ids / slots / planes, 111/111
```

The oracle asserts, per run, the author's slot count, the M-RoPE planes against a closed form, and the letter logits
against the captured slot hidden state. The published fixture merges re-runs of game-frame requests made after
their contexts were changed to plain game descriptions (`--rows r23,…,r27 --merge`, then r23–r24 again).

## 2. Towers

```bash
python conversion/export_qwen38vl_pipelined.py --hf-id Mapika/decider-2b-vision --name decider_2b_vision_g256 \
    --grid-h 8 --grid-w 8 --skip-decoder --vision-dtype fp16w32 --out-dir <lane>/exports
python conversion/export_qwen38vl_pipelined.py --hf-id Mapika/decider-2b-vision --name decider_2b_vision_g448 \
    --grid-h 14 --grid-w 14 --skip-decoder --vision-dtype fp16w32 --out-dir <lane>/exports
# the Python gates load AOT assets, e.g. for the fp16w32 g256 tower:
xcrun coreai-build compile <lane>/exports/decider_2b_vision_g256_vision_fp16w32/decider_2b_vision_g256_vision_fp16w32.aimodel \
    --output <lane>/exports/decider_2b_vision_g256_vision_fp16w32_aotc --platform macOS --preferred-compute gpu --architecture h16c
python conversion/decider_vision/gate_tower.py        # -> models/decider-2b-vision/gate-decider-2b-vision-tower.json
```

`gate_tower.py` runs all three dtypes by default (`--variants fp16,fp32,fp16w32`); export `--vision-dtype fp16`
and `fp32` the same way to reproduce the whole transcript. It looks for `decider_2b_vision_<grid>_vision_aotc/`
(fp16) and `decider_2b_vision_<grid>_vision_<dtype>_aotc/` (fp32, fp16w32). Only fp16w32 ships.

## 3. Decoder

```bash
for i in 0 1 2; do python conversion/decider_vision/parity_decoder_torch.py --shard $i --nshards 3 --out-dir <dir> & done; wait
python conversion/decider_vision/parity_decoder_torch.py --merge --out-dir <dir> \
    --transcript models/decider-2b-vision/gate-decider-2b-vision-torch-parity.json
python conversion/decider_vision/export_decoder.py int8mix --fp16-layers 0,2,5 --prefill-chunk 16 --aot   # ship
python conversion/decider_vision/export_decoder.py fp16 --prefill-chunk 16 --aot                         # reference
```

The bundles land in `<lane>/exports/bundles/<name>/`, the AOT assets in `<lane>/exports/bundles_aotc/`. The
non-shipping decoders of the transcripts come from the same script: `fp16`, `int8hu --head-sym`, `int8lin`,
`int8lin --prefill-chunk 16`, each with `--aot`.

## 4. Readout gates

```bash
B=<lane>/exports/bundles
python conversion/decider_vision/readout_gate_vision.py b1 $B/decider_2b_vision_decode_int8mix_pf16 \
    --order chunk --also-s1 --red --work-dir <lane>/readout_r6 \
    --transcript models/decider-2b-vision/gate-decider-2b-vision-readout-int8mix_pf16.json
python conversion/decider_vision/readout_gate_vision.py b2 --bundles $B/decider_2b_vision_decode_int8mix_pf16 \
    --order chunk --append --work-dir <lane>/readout_r6 \
    --transcript models/decider-2b-vision/gate-decider-2b-vision-e2e.json
```

The same two commands with `decider_2b_vision_decode_fp16_pf16` write the reference's rows. The S=1 bundles ran
b1 with the default `--order s1`. Each gate process takes at most 14 runs plus a re-run of its first one (the
Python runtime leaks one IOSurface per call); the GPU is not locked, so the recorded times are contended.

## 5. How the fp16 layers were chosen

```bash
python conversion/decider_vision/int8_bisect_torch.py dump --qscheme clipping
python conversion/decider_vision/int8_bisect_torch.py run --part <dir>/part_a.json      # (split over processes)
python conversion/decider_vision/int8_bisect_torch.py merge --parts <dir> --out <lane>/logs/r6_bisect.json
```

Rule, fixed before the search: r29 |Δp| ≤ 0.010, no bisect run worse than all-int8, at most four layers, the
smallest set. It selected {0, 2, 5}.

## 6. Held out

```bash
uv run conversion/decider_vision/heldout_prepare.py                       # -> <lane>/heldout (the photos stay here)
python conversion/decider_vision/heldout_eval.py run \
    --bundles <lane>/exports/bundles/decider_2b_vision_decode_int8mix_pf16,<lane>/exports/bundles/decider_2b_vision_decode_fp16_pf16 \
    --transcript models/decider-2b-vision/gate-decider-2b-vision-heldout.json
```

The bars and the ship rule are in the script's docstring and in the transcript; they were fixed before any
held-out result.

## 7. Swift

```bash
cd apps/DeciderVision && swift build -c release --scratch-path <lane>/swift/.build && cd -
BIN=<lane>/swift/.build/release/decider-vision
$BIN fixture --bundle $B/decider_2b_vision_decode_int8mix_pf16 \
    --tower-g256 <lane>/exports/decider_2b_vision_g256_vision_fp16w32_aotc/decider_2b_vision_g256_vision_fp16w32.h16c.aimodelc \
    --tower-g448 <lane>/exports/decider_2b_vision_g448_vision_fp16w32_aotc/decider_2b_vision_g448_vision_fp16w32.h16c.aimodelc \
    --rows <lane>/fixtures/rows.json --images <lane>/fixtures/images --meta <lane>/fixtures/meta.json \
    --out <lane>/swift/aot/fixture_aot.json
# --asset jit with the towers' .aimodel files -> <lane>/swift/jit/fixture_jit.json (the JIT row)
python conversion/decider_vision/gate_swift.py score --aot <lane>/swift/aot/fixture_aot.json \
    --jit <lane>/swift/jit/fixture_jit.json --transcript models/decider-2b-vision/gate-decider-2b-vision-swift.json
apps/DeciderVision/_time_mac.sh                     # timed passes under the machine-wide GPU lock
```

`gate_swift.py ask` and `timing` add the side-by-side `ask` check and the timed passes to the same transcript.
