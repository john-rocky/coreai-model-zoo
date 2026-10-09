# d1_omni/ — LiquidAI d1-omni-600M → Core AI

The recipe behind [`models/d1-omni-600m/`](../../models/d1-omni-600m/). d1-omni-600M (`LiquidAI/d1-omni-600M`, revision
`414f8d64…`, LFM Open License v1.0) is an encoder-type decision model: a bidirectional LFM2.5-Encoder-350M trunk
(10 short convolutions + 6 GQA attentions, hidden 1024), a two-layer decision head, a SigLIP2 vision tower and a
17-layer FastConformer audio tower. One forward answers one typed question (choice / score / noul) over a state:
the host builds `<bos> <state> state <q> instructions (<opt> <mask> option </opt>)… <decide>`, the graph scores
every position, and the host reads the scores at the `<mask>` markers. Images and audio enter as a prefix of
embeddings in front of the text.

What ships (the supervisor's decision of 2026-10-08, from the speed ladder: the fastest form that passes the bar):
three graphs in fp16, their MLIR debug locations removed — the decision graph at seven static lengths (L = 64, 128,
256, 512, 1024, 2048, 4096; L4096 in 2,048-key blocks), the vision graph (one crop) and the audio graph (clip buckets
5 / 10 / 20 / 30 s) — as `.aimodel` bundles for macOS and iOS and as h19p AOT compiles for the iPhone 18 Pro (all but
L4096, whose compile exceeds the phone's per-process memory limit at its first call: a phone runs its `.aimodel`). Every
stage is gated against the publisher's own model in fp32 on the CPU; the transcripts are
`models/d1-omni-600m/gate-d1-omni-600m-*.json`, the lessons [`knowledge/d1-omni-port.md`](../../knowledge/d1-omni-port.md),
the Swift host [`apps/D1Omni`](../../apps/D1Omni/README.md). The iPhone numbers come from round 11, the device gate
[`apps/D1OmniGate`](../../apps/D1OmniGate/README.md) on the iPhone 18 Pro.

## Reproduce (the order of `models/d1-omni-600m/recipe.toml`)

Two environments. **venv-d1** reads the checkpoint with its own code (Python 3.12.11, torch 2.9.0, torchvision 0.24.0,
transformers 5.19.0, tokenizers 0.23.2, numpy 2.5.3, safetensors 0.8.0, huggingface_hub 1.33.0, pillow 12.3.0,
soundfile 0.14.0; the checkpoint needs transformers ≥ 5.15). **The zoo venv** (`coreai-models/.venv`: coreai-core
1.0.0b2, coreai-torch 0.4.1, torch 2.9.0, numpy 2.3.5) exports, strips and gates; the graphs are plain torch and read
`model.safetensors` themselves. Xcode 27.0 RC as `DEVELOPER_DIR` for `coreai-build` (`xcrun -f coreai-build`) and the
Swift build. The work folder is `$ZOO_WORK_ROOT/_d1_omni/` (`conversion/_paths.py`); every script writes new files and
refuses to replace one. Run from the repository root with `PYTHONDONTWRITEBYTECODE=1`; on a Mac shared with other
measurement windows, wrap each heavy step in `quiet_wait.py --`.

```bash
D1=$ZOO_WORK_ROOT/_d1_omni; PY=../coreai-models/.venv/bin/python; PD=$D1/venv-d1/bin/python
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer HF_HUB_DISABLE_XET=1
hf download LiquidAI/d1-omni-600M --revision 414f8d6438174f5b2133a9c21a478fc42625e308   # then HF_HUB_OFFLINE=1

# 0. the fixture and the oracle (venv-d1); each version of the fixture is kept (records.v1.json, records.v2.json) and
#    each oracle run reads the version it was made from
$PD conversion/d1_omni/make_fixtures.py                       # fixtures/records.json v3 (18b4ddff…), checked against v2
$PD conversion/d1_omni/reference_d1_omni.py                   # v1 (c0781185…): ref/records_ref.json + ref/npz/
$PD conversion/d1_omni/reference_d1_omni.py --only img_01 img_02 img_03 'imgm_*' --fixtures-sha256 \
    07ea38a2c0f3d8da4afe4dd4ae79e083c46404cb8ab0ba150b9c0a3fca83ae62 --out $D1/ref/records_ref_images.json
$PD conversion/d1_omni/reference_d1_omni.py --only 'aud_0[4-9]' 'aud_1[0-5]' --out $D1/ref/records_ref_audio.json   # v3
$PD conversion/d1_omni/encode_rows.py --out $D1/results/encode_rows.v3.json && $PD conversion/d1_omni/media_lengths.py

# 1. the torch modules against the oracle (venv-d1)
$PD conversion/d1_omni/eager_check.py && $PD conversion/d1_omni/block_softmax_check.py && $PD conversion/d1_omni/absmax_scan.py
$PD conversion/d1_omni/vision_check.py --stage eager
$PD conversion/d1_omni/audio_check.py --stage mel && $PD conversion/d1_omni/audio_check.py --stage eager

# 2. export (zoo venv): each export gates its torch program on the oracle rows of its bucket before converting
for L in 64 128 256 512 1024 2048 4096; do $PY conversion/d1_omni/export_decide.py --precision fp16 --seq-len $L; done
$PY conversion/d1_omni/export_vision.py --precision fp16
for S in 5 10 20 30; do $PY conversion/d1_omni/export_audio.py --precision fp16 --sec $S; done

# 3. strip the debug locations (the shipped bundles: macos-ship/, and macos-ship-small/ for L64 / L128)
$PY conversion/d1_omni/strip_ship.py && $PY conversion/d1_omni/strip_ship.py --only fp16-L64 fp16-L128

# 4. AOT: h16c for this Mac (the Python gates load it), h19p for the iPhone 18 Pro (never loaded on a Mac)
for T in h16c h19p; do $PY conversion/d1_omni/aot_ship.py --target $T && $PY conversion/d1_omni/aot_ship.py --target $T --only fp16-L64 fp16-L128; done

# 5. the Mac GPU gates (FACTS §7 against the oracle) and stripped = unstripped
C=$D1/compiled/ship-h16c
for L in 64 128 256 512 1024 2048; do $PY conversion/d1_omni/runtime_check.py $C/fp16-L$L --compute gpu --out $D1/results/ship_runtime_decide_fp16_L${L}_gpu.json; done
$PY conversion/d1_omni/runtime_check.py $C/fp16-L4096 --compute gpu --repeats 5 --out $D1/results/ship_runtime_decide_fp16_L4096_gpu.json
$PY conversion/d1_omni/vision_check.py --stage runtime --compute gpu --vision $C/vision-fp16 \
    --decide 256=$C/fp16-L256 --decide 512=$C/fp16-L512 --decide 2048=$C/fp16-L2048 --results-prefix ship_runtime_
A="--audio 5=$C/audio-fp16-5s --audio 10=$C/audio-fp16-10s --audio 20=$C/audio-fp16-20s --audio 30=$C/audio-fp16-30s"
$PY conversion/d1_omni/audio_check.py --stage runtime --compute gpu $A --decide 256=$C/fp16-L256 --decide 512=$C/fp16-L512 --results-prefix ship_runtime_
$PY conversion/d1_omni/audio_check.py --stage runtime --compute gpu $A --decide 64=$C/fp16-L64 --decide 128=$C/fp16-L128 \
    --decide 256=$C/fp16-L256 --decide 512=$C/fp16-L512 --buckets 64,128,256,512,1024,2048,4096 --results-prefix small_runtime_
$PY conversion/d1_omni/compare_ship.py && $PY conversion/d1_omni/compare_ship.py --small

# 6. the Swift host on the same graphs: bit-equal to the Python runtime, AOT and the .aimodel (recipe steps 41-54)
swift build -c release --package-path apps/D1Omni --scratch-path $D1/swift/.build
$PY conversion/d1_omni/gate_swift.py --ship pyref && $PY conversion/d1_omni/host_dump_media.py --ship
#    d1omni parity / parity-media --asset aot|jit (the recipe has the full lines), then:
$PY conversion/d1_omni/gate_swift.py --ship parity && $PY conversion/d1_omni/gate_swift.py --ship media && $PY conversion/d1_omni/gate_swift.py ship-summary
$PY conversion/d1_omni/gate_swift.py --small pyref    # d1omni parity on macos-ship-small, then:
$PY conversion/d1_omni/gate_swift.py --small parity

# 7. the Hugging Face folder (uploads nothing; README.md = the card, no ios-h19p/decide-fp16-L4096) and
#    models/d1-omni-600m/'s fixture and transcripts (the iPhone transcript reads round 11's device records)
$PY conversion/d1_omni/stage_hf.py --ship --final
$PY conversion/d1_omni/zoo_fixtures.py && $PY conversion/d1_omni/zoo_transcripts.py && $PY conversion/d1_omni/zoo_transcripts.py --check
```

| Stage | Gate | Transcript |
|---|---|---|
| 0–1 | the publisher's encode = host.py on every row; the fp32 module: argmax on every row, max \|Δp\| ≤ 2e-5, marker logits ≤ 1e-3; L4096's key blocks = the plain softmax in float64 (≤ 1e-12); the vision and audio towers cos ≥ 0.999999 in fp32 and ≤ 1e-6 in float64; the mel bit for bit (torch) / ≤ 1e-6 (NumPy, float64) | `gate-d1-omni-600m-eager.json` |
| 2 | the export program = the eager module on every row; fp16: the ship bar, measured (the runtime decides) | the bundles' `provenance/export-gate.json` |
| 3 | op counts and `coreai-build inspect` unchanged, no local path in any byte | `gate-d1-omni-600m-strip.json` |
| 4 | exit 0, no ANE region (GPU compiles), no local path | `gate-d1-omni-600m-strip.json` `aot` |
| 5 | FACTS §7 (argmax on every row whose oracle top-2 margin is above 0.02, max \|Δp\| ≤ 0.02, mean of the rows' max \|Δp\| ≤ 0.002) on every bucket and end to end; the controls FAIL; stripped = unstripped bit for bit | `-runtime-decide.json`, `-runtime-vision.json`, `-runtime-audio.json` |
| 6 | the Swift rows = host.py; marker logits = the Python runtime bit for bit, AOT and JIT; FACTS §7 | `gate-d1-omni-600m-swift.json` |
| 7 | no text or binary file with a local path, the user name, an unpublished record, the measured-only speech corpus's name or the tilde home shorthand (one allowed line: the Swift host's comment on the runtime's cache location); `ios/` = `macos/`; README.md = the card; the fixture file = the staging's `reference/` | the staging's `MANIFEST.json` `checks`, `<work>/_d1_omni/results/zoo_fixtures_check.json` |
| speed (not a recipe step) | `timing_run.py window` / `gate_swift.py window` under `quiet_hold.py`; every timed decision checked against the oracle row | `gate-d1-omni-600m-timing-mac.json`, `-forms.json` |
| iPhone (not a recipe step) | `apps/D1OmniGate` on the iPhone 18 Pro: FACTS §7 on 78 + 18 fixture rows with the `.aimodel` and the h19p compiles, AOT = JIT bit for bit, every timed decision checked | `gate-d1-omni-600m-iphone.json` |

The rest of this file: the graph contracts, the fixture, the reference, every script, and the notes of each round
(the numbers behind the transcripts).

## Graph contract (decision bundle, one question row, batch 1, static length L)

| Tensor | Shape | Dtype | Meaning |
|---|---|---|---|
| `input_ids` | [1,L] | int32 | 0 on the prefix positions, the row's ids on [P, P+n), 0 (`<\|pad\|>`) after |
| `prefix_embeds` | [1,L,1024] | float32 | the image / audio prefix on [0, P), 0.0 elsewhere |
| `pad_mask` | [1,L] | float32 | 1 on [0, P+n) |
| `prefix_mask` | [1,L] | float32 | 1 on [0, P) |
| `keep_right` | [1,L] | float32 | 0 at P−1 when P > 0 (the last media position does not read the first text token), else 1 |
| `qtype_onehot` | [1,3] | float32 | choice / score / noul |
| → `scores` | [1,L] | float32 | the scorer at every position; the host reads P + markers |

Text answers are divided by the per-type temperature of `config.json` (`noul:2` 1.666, `choice:3-5` 1.400, …)
before the softmax; image and audio answers are the plain softmax; a noul is scored as [false, true] and reported
as [yes, no].

## Vision graph contract (one crop, batch 1)

| Tensor | Shape | Dtype | Meaning |
|---|---|---|---|
| `pixel_values` | [1,1024,768] | float32 | the crop's 16 px patches row-major over (ph, pw), each [py][px][c]; (x − 127.5) / 127.5; 0.0 past ph·pw |
| `pos_embed` | [1,1024,768] | float32 | `position_table.f32` (16×16×768) resized to (ph, pw), bilinear + antialias, float32; 0.0 past ph·pw |
| `patch_mask` | [1,1024] | float32 | 1 on the ph·pw patches |
| `unshuffle_index` | [256,4] | int32 | token (i, j) → patches (2i,2j), (2i,2j+1), (2i+1,2j), (2i+1,2j+1); 0 past (ph/2)(pw/2) |
| → `prefix` | [1,256,1024] | float32 | rows [0, (ph/2)(pw/2)) are the crop's prefix |

An image's crops are the publisher's: a picture whose rounded area exceeds 2 × 512² is cut into the 512 px tiles of
the closest grid (row-major) plus a thumbnail, else the thumbnail alone (`host.layout` / `image_crops`); the prefix
is the crops' rows in order, images in request order, and goes to the decision graph's `prefix_embeds` [0, P).

## Audio graph contract (one clip, batch 1, one bundle per clip bucket)

| Tensor | Shape (5 / 10 / 20 / 30 s) | Dtype | Meaning |
|---|---|---|---|
| `mel` | [1,128,F], F = 501 / 1001 / 2001 / 3001 | float32 | the normalised log-mel (`mel_host`), columns t ≥ frames 0.0, zero-padded to F |
| `mask_f` | [1,F] | float32 | 1 on t < l0 = frames = n // 160 |
| `mask_f2` | [1,F2], 251 / 501 / 1001 / 1501 | float32 | 1 on t < l1 = (l0 − 1) // 2 + 1 |
| `mask_f4` | [1,F4], 126 / 251 / 501 / 751 | float32 | 1 on t < l2 = (l1 − 1) // 2 + 1 |
| `mask_t` | [1,T], 63 / 126 / 251 / 376 | float32 | 1 on t < P = l3 = (l2 − 1) // 2 + 1 |
| → `prefix` | [1,T,1024] | float32 | rows [0, P) are the clip's prefix |

The clip is the publisher's `waveform()`: 16 kHz mono, int16 / 32768, cut to 30 s, zero-padded to 0.5 s, no
resampling. The bucket is the smallest whose F holds the clip's 1 + n // 160 columns (`host.audio_bucket_for`); P
(`host.audio_prefix_length`) goes to the decision graph's `prefix_embeds` [0, P), with max_len = min(15360, 16384 − P)
and a `None` state sent as `{}`. The mel (`mel_host.py`): preemphasis 0.97; a centre-padded 512-point STFT, hop 160,
Hann(400, periodic=False) centred; power; Slaney mel 128 (`mel_filters_128x257_f32.bin`); log(x + 2^-24); per mel row
over the valid frames (x − mean) / (std(ddof 1) + 1e-5), 0.0 after.

## Fixtures (`fixtures/records.json`)

| Records | From | Public | Note |
|---|---|---|---|
| `card_text`, `card_batch_00/01` | the model card's examples | yes | gold null |
| `card_cats`, `card_audio` | the publisher card's example photo and speech clip | no | media whose licences require attribution: measured only |
| `tv4_000`..`tv4_059` | the Kev port's records (transfer-v4 development, MMLU) | yes | `provenance.line_sha256` added |
| `semif_<id>` ×144 | SemIf authored144 (MIT), mapped as the Kev port maps it | yes | `LICENSE-SemIf` |
| `own_*` ×20 | the Kev port's own records: 11 published + 9 withheld after its name screen | 11 | questions without `instructions` are dropped (5) |
| `tv4x_*` ×140, `tv4s_*` ×20 | the Kev port's other transfer-v4 slices | no | licences not open (tweet_eval, SciQ) |
| `long_3400` | own_L01's sorter log continued (no new names) | yes | 3,410 state tokens |
| `img_01`..`img_03` | questions written here over CC0 photographs and own FLUX.2 klein output (round 5) | yes | `images/manifest.json`; rewritten in v2 (v1: `records.v1.json`) |
| `imgm_01`..`imgm_12` | the same, the measured image rows | no | 8 CC0 + 4 FLUX |
| `aud_01`..`aud_03` | Kokoro-82M voice notes from scripts here | yes | 16 kHz mono int16 |
| `aud_04`..`aud_15` | the same, at lengths across the clip buckets (round 6) | yes | 0.37 s to 36.2 s; added in v3 (v2: `records.v2.json`) |
| `red_arm_000` | tv4_000 with "correctly" → "incorrectly" | no | a gate arm, not a fixture |

## The reference

`AutoModel.from_pretrained(<snapshot>, trust_remote_code=True, dtype=torch.float32)` — the checkpoint's own
`modeling_d1.D1OmniModel` under transformers 5.19 — on the CPU in fp32, through `probabilities(state, questions,
images, audio)` once per request: text mode for every record, and image or audio mode for the records that carry
media, with the questions of a request in one batch as the publisher's `_run` puts them. Hooks read the numbers
without touching the code: the head's raw marker logits ([B, K_max], -1e4 past a row's K), the trunk output, the
vision / audio prefixes, the NaFlex tower inputs and the log-mel. bf16 and MPS are never the reference. transformers
5.19 loads the checkpoint's `vision.tower.vision_model.*` (197 tensors) as `vision.tower.*` (Siglip2VisionModel has
no `vision_model` level); the loaded state equals the file tensor for tensor. The publisher's code is not exactly
batch- and length-invariant in fp32: one request's questions batched together sit up to 4.9e-6 (p) from the same
questions run one at a time, and its RoPE table changes in the last bit with the length it is built for (at 12
threads).

## Scripts

Run from the repo root with a private venv that satisfies the checkpoint's own requirement
(`transformers>=5.15`, torch 2.9.0, torchvision 0.24.0, tokenizers, safetensors, numpy, pillow, soundfile); paths
resolve through [`../_paths.py`](../_paths.py), the work dir is `$ZOO_WORK_ROOT/_d1_omni/`.

| Script | What it does | Gate |
|---|---|---|
| `make_fixtures.py [--fetch]` | writes `fixtures/records.json` (+ `LICENSE-SemIf`, `audio/script.json`) from pinned sources; round 5: copies or downscales the picked pictures into `images/` (+ `images/manifest.json`: origin, licence, resize, sha256); round 6: the twelve voice notes aud_04..aud_15 (`AUDIO_R6`; `K/scripts/synth_aud_r6.py` synthesises them with Kokoro-82M and appends them to `audio/manifest.json`) | every source sha256-pinned; Kev-derived requests asserted equal to the Kev records; every record of the previous version unchanged, in order (round 6: `records.v2.json`, 409 records; round 5 rewrote three of `records.v1.json`) |
| `_fixtures.py` | the fixture file a reference was made from, found by its sha256 (`records.json` or the kept `records.v<N>.json`) | — |
| `host.py` | the host reference: the publisher's `prompt.py` copied verbatim, the per-mode dispatch, graph inputs, temperature / softmax / noul flip, media prefix lengths; round 5: the vision graph's inputs per crop (`image_crops_inputs`: torchvision crops, `patchify`, `position_embeddings` and its NumPy form, `unshuffle_index`), `vision_prefix`, `image_request_rows`, `load_position_table`; round 6: the audio graph's inputs per clip (`audio_bucket_for`, `audio_bucket_shapes`, `audio_inputs`: the mel padded to the bucket and the four time masks), `audio_prefix`, `audio_request_rows`; round 9: `resize_uint8_antialias_numpy` (torchvision's uint8 resize as it runs without AVX2: float32 antialias kernel, rounded half to even), `rgb_uint8`, `crop_pixels_numpy`, `image_crops_inputs(numpy_resize=)`, and torch's tap-count clamp in `_aa_weights_f32` | `encode_rows.py`, `media_lengths.py`, `vision_check.py`, `audio_check.py`, `gate_swift.py resize` |
| `mel_host.py` | the audio front end on the host: `mel_torch` (the publisher's `waveform()` + `MelFrontend`, the same torch calls) and `mel_numpy` (the same seven steps in NumPy, float64: the form a Swift host copies), `slaney_filterbank` (copied), `filterbank_bytes` (`mel_filters_128x257_f32.bin`) | `audio_check.py --stage mel` |
| `encode_rows.py [--out F]` | every fixture row through host.py and through the publisher's `prompt.encode`, and every request through the publisher's `system_one_batch` with stub networks | ids / markers equal on every row; answers and usage equal |
| `media_lengths.py` | image and audio prefix lengths (the decision buckets must hold prefix + text) | host copies = the publisher's `layout` / `preprocess` / audio front end |
| `d1_omni_model.py` | the decision graph in plain PyTorch (`D1Decide`), `validate_config`, `load_weights` (checkpoint key table), `set_precision`, `set_quant_mode` (int8 export), the key-block attention above `KEY_BLOCK` = 2048 keys | `toy_module_check.py`, `block_softmax_check.py` |
| `toy_module_check.py` | `D1Decide` vs the publisher's `encoder.py` on two random small configurations; pad / batch / mask-value invariance; three negative controls; parameter coverage; full-size key and shape table | marker logits ≤ 1e-5, invariances bit-exact, controls > 1e-3 |
| `reference_d1_omni.py` | the reference (below): the publisher's model on every fixture request; per question the ids, raw marker logits, probabilities and answer; trunk outputs, media prefixes and tower inputs for 23 rows → `ref/records_ref.json`, `ref/npz/` (17 files), `results/ref_summary.json`. Round 5: `--only <ids> --out ref/records_ref_images.json` = the image records of the current fixtures, one npz per image (+ the resized position tables). Round 6: `--only 'aud_0[4-9]' 'aud_1[0-5]' --out ref/records_ref_audio.json` (+ the conformer's and the subsampling's outputs in each npz; a sweep of each record's questions over the next clip) | ids = host.py's on every row; `system_one` = host.py's `response` on every request; the head re-run on the saved trunk output returns the logits bit for bit; record 0 and a second whole run bit-identical; the red arm moves p by > 0.02 |
| `d1_omni_vision.py` | the vision graph in plain PyTorch (`D1Vision`: one crop → 256 prefix rows), `validate_config`, `load_weights` (the position table goes to the host), `set_precision` | `vision_check.py` |
| `d1_omni_audio.py` | the audio graph in plain PyTorch (`D1Audio`: one clip bucket, mel + four time masks → the prefix), `validate_config`, `load_weights` (the 17 BatchNorms folded into scale / shift), `set_precision`, `rebucket` | `audio_check.py` |
| `audio_check.py --stage mel` / `--stage eager` | mel: `mel_torch` and `mel_numpy` against the publisher's front end on 22 clips (16 fixture + 6 edge cases); eager: `D1Audio` in fp32 against the publisher's Audio on the same clips (subsampling, conformer, prefix, float64, the clip in every larger bucket, two red arms, the fp32 absmax) | mel_torch: bit for bit; mel_numpy in float64 = mel_torch in float64 ≤ 1e-6; prefix cos ≥ 0.999999, float64 ≤ 1e-6, pad ≤ 1e-5, red arms out of the bar |
| `vision_check.py --stage eager` | `D1Vision` in fp32 against the reference on every crop of 16 pictures: the host's rows, patches and position tables against the publisher's, the tower and the prefix, float64 against the publisher's Vision in float64, the unshuffle order against the publisher's Projector, a red arm | rows, patches, position tables (torch and NumPy), unshuffle: bit for bit; prefix cos ≥ 0.999999; float64 ≤ 1e-6; the red arm out of the bar |
| `eager_check.py` | `D1Decide` in fp32 with the checkpoint's weights, every row at its bucket length, against the reference; trunk outputs; pad content; bucket; GQA forms; three mutations on the prefix rows | argmax on every row, max \|Δp\| ≤ 2e-5, marker logits ≤ 1e-3; pad content bit-exact; each mutation moves every prefix row's logits past 1e-3 |
| `absmax_scan.py` | max \|x\| at every point of the fp32 graph on every row (residual stream, linears, MLP product, conv products, q·kᵀ, scorer input) against fp16's 65,504; the fp16 and wfp16 frames in torch eager on 60 rows | values: the precision of the export is decided on them |
| `block_softmax_check.py [--out F]` | the key-block attention (above 2,048 keys) against the plain form: random toy modules at L=256 (4 blocks) and 4096 (2 blocks) in float64 / fp32 / fp16; the checkpoint's module at L=4096 on 21 rows in fp32 (both forms against the reference) and on 8 of them in float64 | float64: blocked = plain to ≤ 1e-12 in the marker logits; fp32 blocked vs the reference: the eager bar; the fp32 blocked-vs-plain gap is recorded (fp32's own floor here, ~3e-5) |

The export and runtime scripts run with the zoo's shared export venv (`coreai-models/.venv`: torch 2.9.0,
coreai-torch 0.4.1, coreai-core 1.0.0b2, no transformers); the graph is plain torch and reads model.safetensors itself.

| Script | What it does | Gate |
|---|---|---|
| `_metrics.py` | the row-level readout shared by the export and runtime gates: marker logits → host probabilities → the reference; the two bars; the wrong-pairing control | — |
| `export_decide.py --precision fp32\|wfp16\|fp16\|int8 --seq-len 256\|512\|1024\|2048\|4096 [--pad-rows N]` | `load_d1_decide` (int8: the fp16 frame in quant mode, quantized by coreai-opt's eager quantizer) → `torch.export` → `run_decompositions(get_decomp_table())` → the export gate on the bucket's native rows and rows of the smaller buckets padded in → `TorchConverter` (function `main`) → `optimize` → `save_asset`; checks the converted graph's attention form (8 softmax ops at L ≤ 2048, none above); writes the bundle folder below | fp32: argmax on every row, max \|Δp\| ≤ 2e-5, marker logits ≤ 1e-3, else no conversion; wfp16 / fp16 / int8: the ship bar, measured |
| `aot_decide.py --precision P --seq-len L [--preferred-compute gpu\|neural-engine\|none]` | `coreai-build compile … --platform macOS --architecture h16c` (no `--expect-frequent-reshapes`: static graph); counts the `*_ANE_region_*` entries by unique name; records `resources.bin` bytes, `main.hash`, `coreai-build inspect` | exit code and region count recorded; a region-0 Neural Engine probe keeps only its manifest |
| `runtime_check.py <compiled dir> --compute gpu\|neural_engine\|cpu_only [--repeats N]` | the AOT `.aimodelc` on the Core AI runtime with explicit specialization options (never `default()`), every row of the bundle's reference.json `--repeats` times (3; 5 at 4096); `--trace` writes one JSON line per row as it goes | fp32: the fp32 bar on any unit; wfp16 / fp16 / int8: the ship bar; the wrong-pairing control must FAIL |
| `export_vision.py --precision fp32\|wfp16\|fp16` | `load_d1_vision` → `torch.export` (one padded crop) → `run_decompositions` → the export gate on 62 crops → `TorchConverter` (function `main`) → `optimize` → `save_asset`; writes `bundles/d1-omni-600m/macos/vision-<precision>/` with `position_table.f32` | fp32: cos ≥ 0.999999 on every picture and the program = the eager module bit for bit, else no conversion; wfp16 / fp16 measured |
| `aot_vision.py --precision P [--preferred-compute gpu\|neural-engine] [--tag T]` | `coreai-build compile … --platform macOS --architecture h16c` into `compiled/vision-<P>-h16c/` (gpu) or `-ane/` (round 7: the Neural Engine probe; `-<T>` appended with a tag) with aot_decide.py's manifest and the ANE regions counted by unique name | exit code; a region-0 probe keeps only its manifest |
| `vision_check.py --stage runtime --vision <dir> --decide <L>=<dir> …` | the vision AOT on the GPU over every crop (3 calls), then every image row end to end: the bundle's prefix → `host.graph_inputs` → the wfp16 decision AOT of the row's bucket; beside it the reference's prefix through the same bundles, and another picture's prefix as a control | the ship bar end to end; the control must FAIL |
| `aot_decide.py … --tag T` | compile a decision bundle again into `<dir>-<T>/` (round 5: the wfp16 L512 / L2048 AOTs round 4 removed; round 6: L512) | — |
| `export_audio.py --precision fp32\|wfp16\|fp16 --sec 5\|10\|20\|30` | `load_d1_audio` → `torch.export` (one clip of the bucket) → `run_decompositions` → the export gate on the bucket's clips → `TorchConverter` (function `main`) → `optimize` → `save_asset`; writes `bundles/d1-omni-600m/macos/audio-<precision>-<sec>s/` with `mel_filters_128x257_f32.bin` | fp32: cos ≥ 0.999999 on every clip and the program = the eager module bit for bit, else no conversion; wfp16 / fp16 measured |
| `aot_audio.py --precision P --sec S [--preferred-compute gpu\|neural-engine] [--tag T]` | `coreai-build compile … --platform macOS --architecture h16c` into `compiled/audio-<P>-<S>s-h16c/` (gpu) or `-ane/` (round 7: the Neural Engine probe; `-<T>` appended with a tag) with aot_decide.py's manifest | exit code; a region-0 probe keeps only its manifest |
| `audio_check.py --stage runtime --audio <sec>=<dir> … --decide <L>=<dir> …` | the audio AOTs on the GPU, each clip through the bundle of its bucket (3 calls, and once with the NumPy mel), then every audio row end to end: the bundle's prefix → `host.graph_inputs` → the wfp16 decision AOT of the row's bucket (256 / 512); beside it the reference's prefix through the same bundles, and the next clip's prefix as a control | the ship bar end to end (torch mel and NumPy mel); the control must FAIL |
| `vision_check.py` / `audio_check.py --stage runtime --compute neural_engine --decide-compute gpu …` | round 7: the media bundle's Neural Engine AOT on the Neural Engine while the decision bundles stay on the GPU (`--decide-compute`, default = `--compute`); any decision precision (round 7 also gated the fp16 decision bundles with every media bundle) | as above |
| `timing_run.py window --run run1\|run2` (under `quiet_hold.py d1d-r7-<run>`), `summarize` | round 7's speed measurement: in one process every form of every workload (W1 one question, W2 three questions, W3 a 3.4k-token state, W4 a 384 px image, W5 10 s of audio, W5s / W5L the audio buckets) loaded once with explicit compute units, the forms of a workload alternated (35 rounds, 5 warm-up), one decision = graph inputs → graph call(s) → copy → the host's softmax; preprocessing timed apart; every timed decision checked against the oracle row; each bundle again in a fresh process (load seconds, footprint); `summarize` ranks by the mean of the two windows' medians (the rule fixed before timing in `results/timing/ranking_rule.md`) | every timed decision: argmax and \|Δp\| ≤ 0.02 against the oracle row |
| `aot_ios.py [--only <names>]` | round 8: the ten ship bundles (decision fp16 L256–4096, vision fp16, audio fp16 5–30 s) compiled for the iPhone 18 Pro: `coreai-build compile … --platform iOS --min-deployment-version 27.0 --preferred-compute gpu --architecture h19p` into `compiled/ios-h19p/<name>/` (+ `manifest.json`); never loaded on the Mac | exit code, bytes, `resources.bin`, `main.hash`, ANE regions (0) |
| `gate_swift.py strings \| rows \| pyref \| parity \| window --run run1\|run2 [--cold] \| timing` | round 8: the Swift host (`apps/D1Omni`'s `d1omni`) against host.py, the publisher's encode and the oracle (`rows`); the Python runtime on the fp16 AOTs as the reference (`pyref` → `results/swift_ref/`); `d1omni parity` scored on AOT and JIT (`parity`); one measurement window under `quiet_hold.py d1d-r8-swift-<run>` (`window`: `d1omni time` with AOT and JIT alternated, then `timing_run.py main` on the same AOTs as the control); the ranking (`timing`, the rule in `results/timing/ranking_rule_r8.md`) | rows: every field equal; parity: FACTS §7, the wrong-pairing control FAILs, marker logits vs the Python runtime recorded bit for bit; timing: JIT = AOT bit-equal on every call |
| `host_dump_media.py` | round 9: the Python host's media path on the ship form as the Swift host's reference → `results/swift_ref_media/` (`index.json` + `.npy`): per image the PIL RGB, every crop and its four vision inputs (the NumPy forms asserted equal to torchvision / torch), the prefix from the vision fp16 AOT; per clip the int16 samples, the mel (NumPy = the Swift spec, and torch), the four masks, the prefix of each mel from the audio fp16 AOT of its bucket; every media row through the decision fp16 AOT of its bucket (two calls) | the host rows = the oracle's ids / markers; the bundles' position table and filterbank = the checkpoint's / slaney's |
| `gate_swift.py resize \| media \| media-window \| media-timing` | round 9: `host.resize_uint8_antialias_numpy` against torchvision on the 62 crops with three controls (`resize`, venv-d1 → `results/resize_numpy_gate.json`); `d1omni parity-media` (AOT and JIT) scored against the dump and the oracle (`media` → `results/swift_media_image.json`, `swift_media_audio.json`); the round-9 measurement window under `quiet_hold.py d1d-r9-swift-media` (`media-window`: `d1omni time --workloads W4,W5 --cold`, then `timing_run.py main` on the same AOTs) and its ranking (`media-timing`, the rule in `results/timing/ranking_rule_r9.md`) | resize: every crop bit-equal (else ≤ 1 level recorded); media: FACTS §7 end to end, the next item's prefix and the wrong pairing FAIL, inputs / prefixes / rows against the dump bit for bit (or max \|d\|) |
| `stage_hf.py` | round 8: the Hugging Face folder `hf_staging/d1-omni-600M-CoreAI/` (uploads nothing): `macos/` `ios/` (the same `.aimodel`) `ios-h19p/` (AOT) per graph with metadata, host files, tokenizer and provenance; the repository `metadata.json`, `tokenizer/`, `reference/` (public records only, their oracle rows, images and clips), `LICENSE`, `NOTICE.md`, a placeholder `README.md`, `MANIFEST.json` (every file's sha256) | every file checked against its manifest; no text file with a local path or an unpublished record; `ios/` = `macos/` byte for byte |
| `strip_ship.py [--only <names>]` | round 10: each ship bundle (`bundles/d1-omni-600m/macos/<name>/`) loaded (`AIModelAsset.load`), its debug locations removed (coreai-torch `strip_debug_info`: every location → unknown + an operation id), saved with the source's metadata into `bundles/d1-omni-600m/macos-ship/<name>/` (the folder's other files are APFS clones) + `provenance/strip.json` and `macos-ship/manifest.json` | op counts before = after = reloaded = the export manifest; `coreai-build inspect` summaries equal; author / licence / description equal; `main.hash` = sha256 of `main.mlirb`; no '/Users/', home directory or user name in any byte of the stripped `.aimodel` |
| `aot_ship.py --target h16c\|h19p [--only <names>]` | round 10: the stripped bundles compiled for this Mac (`compiled/ship-h16c/<name>/`) and the iPhone 18 Pro (`compiled/ship-h19p/<name>/`, never loaded on the Mac), `--preferred-compute gpu` | exit code, bytes, `resources.bin`, `main.hash`, ANE regions (0), no local path in any byte |
| `runtime_check.py … --out F`, `vision_check.py` / `audio_check.py --stage runtime … --results-prefix P` | round 10: the runtime gates of rounds 3–6 on the ship AOTs, written to `results/ship_runtime_*.json` (the unstripped files stay) | as before |
| `compare_ship.py` | round 10: every stripped gate against the unstripped one (decision L256–4096 rows, the vision and audio prefixes, the 46-row end-to-end arms) → `results/ship_runtime_compare.json` | SAME (every row's logits and p equal, every summary number equal); argmax or max \|Δp\| moved by more than 1e-6 stops the round |
| `gate_swift.py --ship pyref \| parity \| media`, `gate_swift.py ship-summary`; `host_dump_media.py --ship` | round 10: round 8 / 9's Swift parity on the stripped bundles (`macos-ship/`, AOT `compiled/ship-h16c/`) → `results/ship_swift_*`; `ship-summary` → `results/ship_swift_parity.json` | 562 rows (text 470 + image 46 + audio 46) bit-equal to the Python runtime, AOT and JIT; every Python and Swift row and media output bit-equal to the unstripped run |
| `host.SMALL_BUCKETS`, `host.ALL_BUCKETS` | round 12: the decision buckets L64 / L128 and the set with them (64, 128, 256, …, 4096). `host.BUCKETS` stays the shipped set and the default of `bucket_for` (every reference, gate, staging and the Swift host's folder scan recorded buckets with it); a host that ships the small buckets calls `bucket_for(positions, host.ALL_BUCKETS)` | `export_decide.py --seq-len 64\|128`, `runtime_check.py` |
| `export_decide.py --precision fp16 --seq-len 64\|128` | round 12: the decision graph at L64 / L128; a row's native bucket is read from `host.ALL_BUCKETS` at these lengths (≤ 64 positions → 64, 65–128 → 128), the reference's own bucket field (the shipped set's) is still asserted; `reference.json` records `native_buckets`; the pad set of L128 = 20 rows of ≤ 64 positions | the ship bar, measured |
| `aot_ship.py --target h19p --compute neural-engine --only audio-fp16-10s fp16-L256` | round 11: the stripped audio 10 s and decision L256 bundles compiled for the iPhone 18 Pro's Neural Engine (`compiled/ship-h19p-ane/<name>/`, `--preferred-compute neural-engine`; the regions counted by name: 64 and 52; never loaded on the Mac); the device gate is [`apps/D1OmniGate`](../../apps/D1OmniGate/README.md) | the iPhone's ANE: the phone refuses both (`failedToSpecialize`) |
| `strip_ship.py --only fp16-L64 fp16-L128`, `aot_ship.py --target h16c\|h19p --only fp16-L64 fp16-L128` | round 12: the small buckets stripped into `bundles/d1-omni-600m/macos-ship-small/` (not `macos-ship/`: the Swift host's `D1Omni.folders(macos:)` takes every `fp16-L<L>` folder of a directory, so they would move the short rows of every Swift run over `macos-ship/`), compiled into `compiled/ship-h16c/fp16-L<L>/` and `compiled/ship-h19p/fp16-L<L>/` (summaries `manifest.small.json`; the ship set's `manifest.json` untouched) | as round 10 |
| `runtime_check.py` (bucket set from `reference.json`), `compare_ship.py --small`, `audio_check.py --stage runtime … --buckets 64,128,256,512,1024,2048,4096` | round 12: the runtime gates of L64 / L128 (unstripped `compiled/fp16-L<L>-h16c/` and stripped `ship-h16c/`), compared (`results/small_runtime_compare.json`); the 46 audio rows end to end with the rows routed by `host.ALL_BUCKETS` (4 rows on L64, 5 on L128) → `results/small_runtime_audio_e2e_*.json` | FACTS §7; stripped = unstripped (SAME); the controls FAIL |
| `gate_swift.py --small pyref \| parity` | round 12: the Python runtime on the stripped L64 / L128 AOTs over the oracle rows they take (342) as the reference, `d1omni parity --bundle-dir macos-ship-small` (AOT and JIT) scored → `results/swift_parity_L64_128.json` | every row bit-equal to the Python runtime, AOT and JIT; FACTS §7; the control FAILs |
| `timing_run.py window --set r12 --run run5_r12 --lock-label d1d-r12` (under `quiet_hold.py d1d-r12`), `summarize --set r12` | round 12: round 7's harness on the stripped ship AOTs: W1 / W2 at L64 / L128 / L256, W5s (audio 5 s + L128 / L256), W5 (audio 10 s + L256); a stopped job of another lane is not waited on; one window ranked by its medians (`results/timing/ranking_rule_r12.md`) → `results/timing/run5_r12_*.json`, `ranking_r12.json`, `r12_table.md` | every timed decision: argmax and \|Δp\| ≤ 0.02 against the oracle row |
| `stage_hf.py --ship` | round 10: the Hugging Face folder from the stripped bundles and `ship-h19p`, with `reference/LICENSE-MMLU-MIT.txt`, the NOTICE lines for the strip and MMLU, and `host/` (the Swift host's sources + `host/README.md`: what it does, the JPEG rule) | as above, and every staged file, binary included, free of '/Users/', the home directory and the user name |
| `stage_hf.py --ship --with-small` | round 13: the same folder with the decision graph at L64 / L128 too (`macos-ship-small/`, their h16c / h19p compiles, round 12's gates): `metadata.json` lists the seven buckets and `bucket_for(positions, ALL_BUCKETS)`; NOTICE says seven lengths; `host/README.md` says which rows ran through which graph in Swift; each audio folder adds `provenance/e2e-gate-all-buckets.json`. Without `--with-small`, `--ship` stages round 10's folder unchanged | as above; the staged files of round 10 that the small buckets do not touch stay byte for byte |
| `zoo_fixtures.py [--check]` | round 13: `models/d1-omni-600m/fixtures-d1-omni-600m.json` (schema `coreai-d1-omni-fixtures/1`): the 237 public records of fixture v3, the MIT notices of the sources whose text they carry, the 18 media files by name and sha256, and one row per question (360) with ids, markers, the prefix length, the bucket (`ALL_BUCKETS`), the temperature and the oracle's numbers. Its own schema: coreai-kit's `decide-cli parity` routes every `coreai-encoder-fixtures*` file through laya's prompt builder | `--check`: the file = a build = the staging's `reference/` (records by sha256 of their canonical JSON, media by file sha256, rows by ids / markers / logits / p) |
| `zoo_transcripts.py [--check]` | round 13 (round 14: + `iphone`): `models/d1-omni-600m/gate-d1-omni-600m-{eager,runtime-decide,runtime-vision,runtime-audio,swift,strip,timing-mac,forms,iphone}.json`: per gate the source records (path, bytes, sha256), the bar, the status, and the summary numbers recomputed from the per-row / per-call values; items naming an unpublished record dropped (`stage_hf.Scrub`) | every recomputation = its record's own summary; `--check` → `results/zoo_fixtures_check.json`: each transcript = a build, sources' sha256 current, the fixtures check, scans for a local path / the user name / an unpublished id / the measured-only speech corpus / the home shorthand (round 14: also the card, the recipe and the knowledge note), and two controls that must FAIL |
| `stage_hf.py --ship --final` | round 14: the folder to upload: `README.md` = the card (`<work>/_d1_omni/card/README.md`, byte for byte), no `ios-h19p/decide-fp16-L4096` (metadata.json, NOTICE.md and `MANIFEST.json` say why), the paths outside the folder removed from `reference/` and the decision graphs' `runtime-gate.json` made relative, the measured-only clip's clause out of `reference/records.json`; every edit asserted and listed in `MANIFEST.json` `final` | the scans of `--ship` plus the measured-only speech corpus's name (any file), the tilde home shorthand (text files; one allowed line) and the shorthand with a home folder name (binary files) |

## Round notes

The numbers each round measured, in its words (the transcripts in `models/d1-omni-600m/` recompute the ones the card uses). Round 1 wrote the fixture and the host; round 11 is the iPhone round; round 13 wrote the zoo files and staged L64 / L128.

### Round 2 results (CPU, Mac M4 Max)

- Reference: 402 requests / 470 rows (every record in text mode and the 11 image / audio rows; the three planned
  pictures do not exist yet, so their 9 image rows are skipped); 13 near ties (top-2 margin ≤ 0.02).
- `D1Decide` fp32 vs the reference: argmax 470/470, max |Δp| 1.45e-5, max marker-logit |Δ| 8.6e-5; pad content
  bit-exact; the two GQA forms bit-identical; the mutations move the prefix rows' logits by 0.04 to 1.6.
- Why the |Δp| bar is 2e-5 and not 1e-5: the reference is itself 9.3e-6 (p) from the publisher's code run in
  float64, the graph 7.9e-6, so two fp32 computations of the same function cannot be held below their sum. With
  every fp32 step of both in float64 and the RoPE table built for the same length, the marker logits agree to
  1.2e-13: the graph computes the publisher's function.
- On saturated rows a wrong mask moves the logits but hardly the probabilities (keep_right all ones: |Δp| 0.0084,
  logits 0.04 to 0.74), so the prefix-row gates read the marker logits as well as p.
- Activation ranges (fp32, every row): the largest |x| is 2,069, the residual stream of one text position from
  layer 10 to 14 (a massive activation that layer 9's MLP writes), a 31.7× margin to fp16's 65,504; media prefix
  embeddings reach 529, q·kᵀ 190 in the trunk and 1,180 in the head's second layer, the MLP product 321. Nothing
  is non-finite.
- torch eager on 60 rows (13 of them near ties) against the reference: fp16 compute — argmax 57/60 (the three
  flips are near ties), max |Δp| 8.2e-3, mean of the row maxima 1.6e-3, no NaN / inf; wfp16 (fp16 weights, fp32
  compute) — argmax 60/60, max |Δp| 2.1e-3, mean 4.0e-4.

### Round 3 results (L=256, Mac M4 Max, macOS 27.0 26A428, coreai-build 3600.83.1)

Rows: every reference row whose prefix + text fits 256 positions, 436 = 426 text rows and 10 audio-prefix rows
(card_audio, aud_01..03, with the reference's own prefix). 12 of them are near ties (top-2 margin ≤ 0.02). Bars:
fp32 = round 2's eager bar; ship bar = argmax on every non-near-tie row, max |Δp| ≤ 0.02, mean of the row maxima
≤ 0.002. Every run's wrong-pairing control (each row judged against another same-shape row's reference) fails.

Bundles (`$ZOO_WORK_ROOT/_d1_omni/bundles/d1-omni-600m/macos/<precision>-L256/`: the `.aimodel`, `metadata.json`,
`tokenizer/`, `reference.json`, `provenance/`):

| Precision | Export gate (decomposed program, torch CPU) | Bundle bytes | Ops | AOT h16c `resources.bin` bytes | Neural Engine probe |
|---|---|---:|---:|---:|---|
| fp32 | argmax 436/436, max \|Δp\| 1.4e-5, mean 1.3e-6 | 1,523,328,898 | 1,193 | 1,523,057,800 | — |
| wfp16 | argmax 436/436, max \|Δp\| 2.0e-3, mean 3.1e-4 | 762,076,218 | 1,299 | 761,785,432 | 1 region, 5,746 B IR |
| fp16 | argmax 434/436 (2 near ties flip), max \|Δp\| 1.2e-2, mean 1.3e-3 | 761,849,972 | 1,315 | 761,554,008 | 52 regions, 1,287,141 B IR |

The decomposed programs equal the eager module bit for bit on all 436 rows at every precision. wfp16 keeps its
weights in fp16 through the AOT compile (`resources.bin` is half the fp32 one): the casts to fp32 are not folded.

Runtime (AOT `.aimodelc`, explicit specialization options, every row 3 times):

| Bundle | Units preferred | Verdict | argmax (non-near-tie / near tie) | max \|Δp\| | mean | max \|Δlogit\| | Repeat drift |
|---|---|---|---|---:|---:|---:|---|
| fp32 h16c | GPU | PASS, fp32 bar | 424/424, 12/12 | 1.09e-5 | 1.4e-6 | 1.0e-4 | 0 |
| wfp16 h16c | GPU | PASS, ship bar | 424/424, 12/12 | 1.99e-3 | 3.1e-4 | 2.9e-2 | 0 |
| fp16 h16c | GPU | PASS, ship bar | 424/424, 12/12 | 6.70e-3 | 9.8e-4 | 6.1e-2 | 0 |
| fp16 Neural Engine compile | Neural Engine | FAIL | 401/424, 8/12 | 0.95 | 4.3e-2 | 11.2 | 0 |
| wfp16 Neural Engine compile | Neural Engine | FAIL | 406/424, 12/12 | 0.73 | 3.4e-2 | 13.8 | 13.8 on 167 rows |

- The GPU computes the fp32 graph at fp32 precision: closer to the reference than torch on the CPU (1.09e-5 vs
  1.45e-5), and wfp16 on the GPU lands where wfp16 lands in torch (1.99e-3 vs 2.00e-3).
- fp16 on the GPU passes, with every near tie kept (torch on the CPU flipped two).
- With a Neural Engine preference the fp16 graph (52 regions) is deterministic and wrong: 238 of 436 rows are
  past 0.02, the audio-prefix rows worst (argmax 4 of 10). The wfp16 graph (1 region) changes from call to call on
  167 rows and matches the GPU on the rest (median marker-logit gap 3.6e-7) — the laya port saw the same.
- `cpu_only()` does not load an h16c AOT `.aimodelc` (compiled `--preferred-compute gpu`, or `none`, which gives
  the same bytes): `CoreAIDelegates.AIModelError error 1` (failedToSpecialize). The asset holds only the MPSGraph
  package. The fp32 GPU run is the runtime's reference instead.
- The first Neural Engine run of the wfp16 graph died at row 400+ with `MPSCommandBufferImageCache.mm:1220: failed
  assertion 'Internal error: Released a texture not in current cache frame.'` while another lane timed a GPU job on
  the same Mac; run alone it completed (the numbers above). The cause is not isolated.
- Load: 0.2–0.3 s on the GPU; 17.5 s (fp16) and 75.4 s (wfp16) the first time with a Neural Engine preference,
  0.005 s the next time the same asset loaded. Peak RSS 1.5 GiB (2.9 GiB for fp32). No milliseconds per call were
  measured in this round.

### Round 4 results (the bucket family, Mac M4 Max, macOS 27.0 26A428, coreai-build 3600.83.1)

Buckets 512 / 1024 / 2048 / 4096 in wfp16 and fp16 — export, AOT h16c, the Core AI runtime on the GPU — and int8 at
L=256. Each bundle's `reference.json` holds its rows, tagged by set: the native rows, whose bucket it is (512: 19,
card_cats' image prefix among them; 1024: none; 2048: 11; 4096: 4); 20 rows of the smaller buckets padded into it,
the nearest bucket first (5 for fp16 at 2048 and 4096, where torch computes fp16 slowly on the CPU: 46 s a row at
2048, 150 s at 4096); and, at the runtime gate only, the 10 audio-prefix rows (and card_cats where it is not in yet)
padded in. Bars as in round 3.

Key blocks at 4096. Above 2,048 keys every attention, trunk and head, runs in key blocks sharing one max: the plain
chain `softmax(q kᵀ) v` returns wrong values from 4,032 keys on this GPU (zoo `knowledge/clef-flash-port.md`). In
torch (`block_softmax_check.json`, `block_softmax_diag.json`) the blocked and the plain forms agree to 6.7e-14 in the
marker logits in float64: the same function. In fp32 they are up to 3.3e-5 apart, which is fp32's own floor on this
graph — the plain form sits 3.0e-5 from its float64 run, the blocked form 2.9e-5 — and the blocked form keeps the
eager bar against the reference (21 rows at L=4096: argmax 21/21, max |Δp| 4.8e-6, marker logits 3.1e-5). The
converted 4096 graphs have no `softmax` op and 16 `reduce_max` / `exp` / `reduce_sum` / `broadcasting_divide` and 8
`broadcasting_maximum` (8 attentions × 2 blocks); the graphs at L ≤ 2048 have the L=256 graph's op counts, op type by
op type.

Runtime (AOT `.aimodelc`, GPU preferred, every row 3 times, 5 at 4096):

| Bundle | Rows | Verdict | argmax (non-near-tie / near tie) | max \|Δp\| | mean | max \|Δlogit\| | Drift | Bundle bytes | Ops |
|---|---:|---|---|---:|---:|---:|---|---:|---:|
| wfp16 L512 | 49 | PASS | 46/46, 3/3 | 2.06e-3 | 3.4e-4 | 1.7e-2 | 0 | 762,207,305 | 1,299 |
| wfp16 L1024 | 30 | PASS | 29/29, 1/1 | 2.05e-3 | 3.1e-4 | 1.7e-2 | 0 | 762,469,259 | 1,299 |
| wfp16 L2048 | 41 | PASS | 40/40, 1/1 | 2.06e-3 | 3.2e-4 | 1.7e-2 | 0 | 762,993,741 | 1,299 |
| wfp16 L4096 | 34 | PASS | 33/33, 1/1 | 2.06e-3 | 3.1e-4 | 1.7e-2 | 0 (5 calls) | 764,087,671 | 1,481 |
| fp16 L512 | 49 | PASS | 46/46, 3/3 | 4.37e-3 | 9.5e-4 | 4.8e-2 | 0 | 761,915,494 | 1,315 |
| fp16 L1024 | 30 | PASS | 29/29, 1/1 | 7.13e-3 | 1.0e-3 | 4.6e-2 | 0 | 762,046,397 | 1,315 |
| fp16 L2048 | 26 | PASS | 25/25, 1/1 | 7.13e-3 | 9.9e-4 | 4.6e-2 | 0 | 762,308,740 | 1,315 |
| fp16 L4096 | 20 | PASS | 20/20, — | 3.77e-3 | 6.4e-4 | 4.5e-2 | 0 (5 calls) | 762,879,057 | 1,513 |
| int8 L256 | 436 | FAIL | 420/424, 9/12 | 9.74e-2 | 1.0e-2 | 0.80 | 0 | 468,960,581 | 1,529 |

- Every bucket lands near L=256's numbers (round 3: wfp16 1.99e-3 / 3.1e-4, fp16 6.70e-3 / 9.8e-4): padding a row
  into a longer bucket does not move it past the bar, and the blocked 4096 graphs do not change from call to call.
  The bundle grows with L only by the RoPE tables it holds as constants (wfp16: 762.2 MB at 512, 764.1 MB at 4096).
- The fp16 export gates in torch on the CPU sit further from the reference than the GPU does (mean 2.13e-3 at 1024 and
  2.59e-3 at 2048, over the ship bar on those small row sets), as at L=256: the runtime decides.
- int8 (weight-only, symmetric, per block of 32 input channels, on all 100 trunk and head linears; the embedding,
  norms and scorer as in fp16; coreai-opt's eager quantizer): torch on the CPU and the GPU agree (max |Δp| 9.76e-2 /
  9.74e-2, mean 1.02e-2 both) and both fail the ship bar, four non-near-tie rows flipping. The bundle is 0.47 GB
  (int8 312 MB, fp16 scales 20 MB, the fp16 embedding 134 MB), but its AOT compile's `resources.bin` is 761,554,008
  bytes, the fp16 AOT's size (with another sha256), though `coreai-build inspect` still counts the weights as Int8. Which layers carry the error is left for a later round's bisect.
- Load 0.2 s, first call 0.2–0.9 s, peak RSS 1.5 GiB in every run. No milliseconds per call were measured.

### Round 5 results (the vision graph, Mac M4 Max, macOS 27.0 26A428, coreai-build 3600.83.1)

Pictures (`images/manifest.json`; nothing generated or downloaded in this round). Two sieved pools, read only:
Wikimedia Commons hotel-room photographs whose licence template is CC0 (the decider-2b-vision lane's set, no person in
frame) and this zoo's own FLUX.2 klein 4B rooms (prompt, seed, hub revision in its manifest). Every file used was
looked at by eye at full size before its questions were written: no person, no legible text or logo, no existing
design (the 11 photographs with a burned-in date stamp, and those with a readable brand, sign, magazine or clock, were
left out). Public: `img_01` (a FLUX room at 384×384, 144 prefix rows), `img_02` (a CC0 room at 384×288, 108),
`img_03` (a CC0 twin room at 768×1024, tiled 2×3 + thumbnail, 1,770). Measured only: 4 CC0 at 1024 px (tiled, 1,770 /
1,783), 4 CC0 at 512 px (one crop, 176 / 192), 4 FLUX at 1024×1024 (2×2 + thumbnail, 1,280). Downscaling: Pillow
BICUBIC. Each record asks three questions (choice / noul / score: the room, beds, a lamp, a window, a suitcase,
brightness, tidiness); gold is the answer seen in the picture. The publisher's model agrees with that gold on 29 of 45
(it never answers "Bright": the four rooms seen as bright get Dark three times and Dim once) — a fact about the
model, not part of any gate.

Reference (`ref/records_ref_images.json`, 15 requests × text and image mode = 90 rows; `ref/npz/<id>.npz`, 699 MB):
the same checks as round 2 (ids, responses, hook points, determinism, a second run bit-identical); batch invariance
2.9e-6; the image red arm (img_01's questions over imgm_11's picture) moves p by 0.9999.

Eager (`vision_check.py --stage eager`, CPU fp32, 62 crops, 9 grid shapes): the host's rows, patches and position
tables equal the publisher's bit for bit; so does `position_embeddings_numpy`, which writes out torch's float32
antialias kernel (weights rounded to float32, the width pass then the height pass, one fused multiply-add per tap; a
float64 filter is up to 2.0e-5 off). The prefix: cos ≥ 0.99999999992 on every picture; max |d| 0.020 (img_02, where
the prefix reaches 413). That is fp32's rounding, not a different function: in float64, D1Vision and the publisher's
Vision agree to 2.7e-10, while each one's fp32 run sits up to 0.12 / 0.14 from its own float64 run. The unshuffle
order equals the publisher's Projector on 16 grids; feeding the 16×16 position table without the resize drops cos to
≤ 0.49. The tower's residual stream peaks at 895 (fp16 headroom 73×).

| Vision bundle | Bytes | Ops | GPU prefix: cos min / max \|Δ\| | End to end (wfp16 decision graph, 46 rows): argmax / max \|Δp\| / mean | Same rows, the reference's prefix | Control (another picture) |
|---|---:|---:|---|---|---|---|
| fp32 | 376,281,817 | 679 | 0.999999999 / 0.068 | 46/46 / 2.05e-3 / 2.5e-4 — PASS | 2.06e-3 / 2.5e-4 | FAIL (0.9998) |
| wfp16 | 188,477,299 | 754 | 0.999975 / 11.0 | 46/46 / 2.78e-3 / 4.8e-4 — PASS | 2.06e-3 / 2.5e-4 | FAIL (0.9998) |
| fp16 | 188,291,698 | 758 | 0.999974 / 10.7 | 46/46 / 6.83e-3 / 9.1e-4 — PASS | 2.06e-3 / 2.5e-4 | FAIL (0.9998) |

- The gather (`F.embedding` of the hidden states by `unshuffle_index`) lowers to one `coreai.gather_nd`: the graph
  stays one function. The AOT compiles in 0.7–1.5 s; `resources.bin` is the bundle's size (no cast folded wider).
- fp16 weights alone move the prefix: wfp16's cos falls to 0.999975 in torch and on the GPU alike (img_02, a few
  rows at 0.9985). The decisions barely notice: end to end, max |Δp| is 2.78e-3 (wfp16) and 6.83e-3 (fp16) against
  2.06e-3 for the same rows with the reference's own prefix (the decision graph's share). fp16 compute is closer to the reference on the GPU than in torch on the CPU
  (prefix cos 0.999974 vs 0.99938), as in rounds 3–4.
- Buckets: the image rows land in 256 (18), 512 (card_cats) and 2048 (27); the 2048 rows run the round-4 wfp16 L2048
  bundle with no change. Every call gives the same numbers 3 times (drift 0). No milliseconds were measured.

### Round 6 results (the audio graph, Mac M4 Max, macOS 27.0 26A428, coreai-build 3600.83.1)

Clips (`audio/manifest.json`; nothing downloaded). Twelve more voice notes made here with Kokoro-82M (the five voices in
the local cache, seed 0, each synthesised twice and required equal), English requests from a user to an assistant with
no names and no real person, company or product: aud_04 / 05 (2.8 / 2.9 s), aud_06 / 07 (8.1 / 8.5 s), aud_08 / 09
(15.9 / 13.7 s), aud_10 / 11 (21.5 / 21.0 s), aud_12 / 13 (28.0 / 28.5 s), aud_14 (one word, its speech kept: 0.37 s,
which the publisher pads to 0.5 s) and aud_15 (36.2 s, which it cuts to 30 s). Each asks aud_01's three questions
(topic / is something asked for / urgency); gold is what the script was written to carry. The publisher's model answers
"feedback" to the topic on every one of the 15 own clips (gold agreement 13 of 35 on the new rows) — a fact about the
model on these questions, not part of any gate. Fixtures v3 (`fixtures/records.json`, 18b4ddff…): v2's 409 records
unchanged, 12 added; v2 kept as `records.v2.json`.

Reference (`ref/records_ref_audio.json`, 12 requests × text and audio mode = 72 rows; `ref/npz/aud_04..15.npz`,
62 MB, with the conformer's and the subsampling's outputs): the same checks as rounds 2 and 5 (ids, responses, hook
points, determinism, a second run bit-identical); batch invariance 1.8e-5. aud_04's questions over aud_07's clip move p
by only 0.0034 (both answered "feedback" / "Right now" with near-certainty), while each record's questions over the next
clip move 27 of 36 rows by more than 0.02 (11 argmax changes): that next-clip swap is the end-to-end control below.
The audio rows land in decision buckets 256 (30, with the round-2 rows) and 512 (16: the clips from 15 s up).

Mel (`audio_check.py --stage mel`, 22 clips = the 16 fixture clips + 6 edge cases cut from them: exactly 0.5 s, the
largest clip of the 5 s bucket and one sample more, an odd length, exactly 30 s, 2 s of digital silence):
`mel_torch` equals the publisher's front end bit for bit on all 22 (and the publisher's mel equals the oracle's hooked
mel on the 16); a float-sample input equals the int16 input. `mel_numpy` in float64 equals `mel_torch` run in float64
to 4.5e-13: the same function. Against the publisher's float32 mel it is within 1e-4 on 14 of 22 clips and up to
3.1e-4 on speech, exactly as far as that float32 mel sits from its own float64 run (the low mel rows 0–3 read FFT
bins at 31–94 Hz that carry 1e-8 of a frame's power). Digital silence is the extreme case: every mel row has std 0, so
float64 gives 0.0, while the publisher's float32 mean of equal values rounds and the normalisation multiplies that by
1e5 (up to 0.16). The filterbank equals the publisher's bit for bit (`mel_filters_128x257_f32.bin`, sha256 bce5ec5f…).

Eager (`audio_check.py --stage eager`, CPU fp32, 22 clips): the host's rows equal the reference (46/46); the subsampling
output cos ≥ 0.99999999999995, the conformer output ≥ 0.9999999999996, the prefix ≥ 0.9999999999997 (max |d|
1.8e-5); in float64, D1Audio and the publisher's Audio agree to 3.1e-9 (the publisher's relative-position table
changes in the last bit with the clip length it is built for, 49 of 181k entries between 356 and 376 steps). Each clip in
every larger bucket: max |d| 8.8e-6 over 35 pairs. aud_14 (padded) and aud_15 (cut) match like the rest. Red arms:
the four masks set to 1 move every clip with pad steps out of the bar (cos ≤ 0.9962), dropping the BatchNorm affine
moves every clip (cos ≤ 0.80). The fp32 absmax is 1,403 (the subsampling's output; fp16 headroom 47×).

| Audio bundle | Bytes | Ops (cast) | GPU prefix cos min / max \|Δ\| | End to end (wfp16 decision graph): rows, argmax / max \|Δp\| / mean | Same, NumPy mel | Same rows, the reference's prefix | Control (next clip) |
|---|---:|---:|---|---|---|---|---|
| fp32 10 s | 449,661,094 | 2,273 (0) | 0.9999999999996 / 1.0e-5 (13 clips ≤ 10 s) | 24, 24/24 / 8.13e-4 / 1.09e-4 — PASS | 8.13e-4 / 1.09e-4 | 8.15e-4 / 1.09e-4 | FAIL (0.985) |
| wfp16 5 / 10 / 20 / 30 s | 226,117,475 / 226,375,540 / 226,887,540 / 227,399,542 | 2,501 (192) | 0.9999989 / 0.0126, 0.9999889 / 0.083, 0.9999937 / 0.052, 0.9999797 / 0.244 | 46, 46/46 / 3.88e-3 / 4.07e-4 — PASS | 3.89e-3 / 4.08e-4 | 2.39e-3 / 1.52e-4 | FAIL (0.985) |
| fp16 5 / 10 / 20 / 30 s | 225,187,123 / 225,316,174 / 225,572,188 / 225,828,193 | 2,560 (286) | 0.9999989 / 0.0125, 0.9999834 / 0.100, 0.9999936 / 0.053, 0.9999787 / 0.237 | 46, 46/46 / 3.69e-3 / 4.37e-4 — PASS | 3.76e-3 / 4.31e-4 | 2.39e-3 / 1.52e-4 | FAIL (0.985) |

- One function per bucket: 56 `coreai.conv2d` (the subsampling's five and the conformer's 3 × 17 Conv1d), 17 softmax,
  37 pad. The rel_shift (pad, reshape, slice) lowers as written. AOT h16c 0.9–3.0 s per bundle, no Neural Engine
  region, `resources.bin` the bundle's size; a larger bucket adds exactly its relative-position rows (fp32 in wfp16,
  fp16 in fp16).
- The prefix moves with fp16 weights (cos 0.99998 at worst, the 30 s bucket); the decisions barely do: end to end max
  |Δp| 3.9e-3 (wfp16) and 3.7e-3 (fp16) against 2.4e-3 for the same rows with the reference's prefix. The NumPy mel
  changes the end-to-end numbers in the fourth digit. Every call gives the same numbers 3 times (drift 0). No
  milliseconds were measured.

### Round 7 results (speed, Mac M4 Max, macOS 27.0 26A428, Core AI Python runtime, AOT h16c)

The speed table is the lane's speed ladder (`$ZOO_WORK_ROOT/_d1_omni/SPEED_LADDER.md`), built by `timing_run.py` in two
measurement windows (`quiet_hold.py`), every form of a workload alternated in one process, the ranking rule fixed
before the first window (`results/timing/ranking_rule.md`). One decision = graph inputs → graph call(s) → copy → the
host's softmax; preprocessing apart. In short (the mean of the two windows' medians):

- fp16 is the fastest decision graph on the GPU at every length (L256 16.8 ms, L4096 299 ms); fp32 is 17.6 ms; wfp16
  (fp16 weights, fp32 compute) is the slowest form of every graph (decision L256 21–25 ms, vision 24 ms vs 22, audio
  10 s 12.6 ms vs 8.6); int8 runs as fp16 (the same AOT bytes) and misses the bar.
- Per card column on the Mac: one question 16.8 ms, three questions 49.7 ms (3 calls), a 3.4k-token state 299 ms, a
  384 px image 38.3 ms (vision 21.7 + decision 16.6), 10 s of audio 25.0 ms (audio 8.6 + decision 16.3).
- The Neural Engine is slower than the GPU for every graph tried: decision L256 fp16 41.8 ms / wfp16 23.5 ms (both miss
  the bar), vision fp16 (40 ANE regions) 79–84 ms per crop (misses the bar on the mean, 3.1e-3), audio fp16 10 s (64
  regions) 28 ms per clip (passes, 24 rows).
- The audio clip buckets cost 7.5 / 8.6 / 11.9 / 15.5 ms (fp16, 5 / 10 / 20 / 30 s); the decision buckets 16.8 / 28.7 /
  55.2 / 112.4 / 299.9 ms (fp16, L256 … 4096).

### Round 8 results (the Swift host, Mac M4 Max, macOS 27.0 26A428, Xcode 27.0.0 RC, Swift 6.4)

- [`apps/D1Omni`](../../apps/D1Omni/README.md) (library `D1Omni`, CLI `d1omni`, swift-transformers 1.3.3): every row of
  fixture v1 / v2 / v3 (479 / 551 / 623 rows) has host.py's and the publisher's `prompt.encode`'s ids, markers, cuts,
  bucket and graph inputs; the publisher's raw marker logits through the Swift readout give host.py's p bit for bit
  and its responses byte for byte (`results/swift_rows_gate.json`).
- Parity on the ship form (the fp16 decision graphs: the Mac AOT h16c, and the `.aimodel` specialized by the Swift
  runtime at load), the 470 oracle rows (459 text + 11 with the oracle's media prefix): FACTS §7 PASS for both (argmax
  470/470, max |Δp| 0.0067, mean 0.00098); every row's marker logits and p equal the Python runtime's bit for bit, JIT
  = AOT on every row; the wrong-pairing control FAILs (`results/swift_parity_gate.json`). No fixture row reaches L1024:
  one request with three rows of 961–1,009 positions gave the Python runtime's response byte for byte from the AOT,
  the cold JIT and the warm JIT (`results/swift_l1024_probe.json`).
- The aot lever, two measurement windows (`results/timing/ranking_r8.json`, the rule in `ranking_rule_r8.md`), JIT /
  AOT: one question 16.14 / 16.16 ms, three questions 48.46 / 48.59 ms, a 3.4k-token state 301.21 / 300.84 ms, the
  outputs bit-equal on every timed call; the Python runtime on the same AOT in the same windows 16.67 / 49.55 / 301.50
  ms. Load (`AIModel` + `loadFunction`, L256 / L4096): cold AOT 2.00 / 2.43 s, JIT 1.93 / 2.39 s; a second launch AOT
  1.04 / 1.11 s, JIT 1.64 / 0.99 s; again in the same process 0.14–0.35 s. = On the Mac the `.aimodel` is enough. The
  Swift tokenizer takes 0.40 s on the 3.4k-token state (Python's `tokenizers`: 2.0 ms), more than the decision.
- iOS: the ten ship bundles compiled for the iPhone 18 Pro (`aot_ios.py`, h19p, 4.90 GB together, no ANE region),
  not loaded on a device yet.
- Hugging Face: `stage_hf.py` → `hf_staging/d1-omni-600M-CoreAI/` (318 files besides `MANIFEST.json`, 14.8 GB, every
  graph and media file an APFS clone), not uploaded. The Swift host answers from its `macos/` folders as from the work
  folders (two requests, the Python runtime's responses; `results/swift_staged_ask.json`). No text file holds a local
  path or names an unpublished record. The graphs themselves carry MLIR debug locations with the exporting machine's
  path to the module file (40 binary files: `main.mlirb` of every `.aimodel` in `macos/` and `ios/`, `main-h19p.mlirb`
  and `specialized_model_0.mpsgraph` of every AOT); stripping them changes the bundles' bytes and `main.hash`, a
  re-export decision left for later.

### Round 9 results (the Swift host's media path, Mac M4 Max, macOS 27.0 26A428, Xcode 27.0.0 RC, Swift 6.4)

- The resize: torchvision 0.24's `resize` of a uint8 tensor runs, on a CPU without AVX2 (this Mac:
  `torch.backends.cpu.get_cpu_capability()` = DEFAULT), as uint8 → float32 → `F.interpolate(bilinear,
  align_corners=False, antialias=True)` → `round_()` (half to even) → uint8; with AVX2 / AVX512 it takes a native
  fixed-point uint8 kernel instead (not measured here). `host.resize_uint8_antialias_numpy` writes the float32 path out
  (the round-5 weights and fused multiply-add passes, plus torch's tap-count clamp): 62/62 crops of the 16 fixture images
  bit-equal to torchvision (42 resized), and their patches equal the publisher's `preprocess()` output; a float64 filter
  differs on 40 crops, PIL's BILINEAR on 42 (both by 1 level), rounding half up on all 17 resized images
  (`results/resize_numpy_gate.json`).
- The Swift media path ([`apps/D1Omni`](../../apps/D1Omni/README.md): `ImagePreprocess`, `JPEGBaseline`,
  `AudioPreprocess`, `MediaGraphs`) against the Python host's dump on the ship form (`host_dump_media.py`; vision fp16,
  audio fp16 × 4 buckets, decision fp16), AOT and JIT alike: images — the decoded RGB 16/16 (15 PNG, four of them with
  an ICC profile, four with EXIF; one baseline 4:4:4 JPEG, read by `JPEGBaseline` = libjpeg's integer IDCT and colour
  tables: ImageIO's own JPEG decoder was 1 to 3 levels off PIL on 19 % of its bytes), the 62 crops, the four vision
  inputs and the 16 prefixes bit-equal; audio — the samples, the four masks and the 16 prefixes bit-equal, the mel
  equal but for 1 of 3,522,048 values (1 ulp: vDSP's FFT against NumPy's pocketfft; the rest of the mel follows NumPy's
  arithmetic: libm log and cos, Accelerate's dgemm called as NumPy's matmul calls it, NumPy's pairwise sums). The 92
  media rows' marker logits and p equal the Python runtime's bit for bit; end to end against the oracle (FACTS §7):
  images 46/46, max |Δp| 0.0062, mean 0.0011; audio 46/46, 0.0032, 0.00077; the next item's prefix and the wrong
  pairing FAIL; JIT = AOT on every media output and row (`results/swift_media_image.json`, `swift_media_audio.json`).
  `d1omni ask --image / --audio` answers img_01 and aud_01 with the Python host's responses byte for byte, and exits
  non-zero on a WAV as an image, a FLAC as audio and a missing file (`results/swift_media_ask.json`).
- Speed, one measurement window (`results/timing/ranking_r9.json`, the rule in `ranking_rule_r9.md`), JIT / AOT: W4 (a
  384 px image) 37.37 / 37.47 ms, W5 (9.6 s of audio) 24.48 / 24.78 ms; the Python runtime on the same AOTs in the same
  window 37.98 / 25.49 ms. Preprocessing apart: PNG decode 1.91 ms, crops 0.08 ms, patches + position table 0.73 ms;
  WAV 0.30 ms, mel + masks 1.80 ms; tokenize 0.35 / 3.58 ms. Cold loads: vision AOT 0.33 / JIT 0.60 s, audio 10 s
  0.39 / 1.40 s, decision L256 1.07 / 2.09 s.

### Round 10 results (the stripped ship bundles, Mac M4 Max, macOS 27.0 26A428, coreai-torch 0.4.1, coreai-build 3600.83.1)

- Why: each ship graph's `main.mlirb` carried MLIR debug locations naming the exporting machine's module file and
  folder (two strings per bundle; round 8 found them in 40 staged binaries: the `.aimodel` in `macos/` and `ios/`, and
  `main-h19p.mlirb` and `specialized_model_0.mpsgraph` of every AOT). A loader never reads them, but they published a
  local path.
- How: `strip_ship.py` loads each bundle (`AIModelAsset.load`), runs coreai-torch's `strip_debug_info` (every location
  becomes an unknown location with an operation id) and saves it with the source's metadata to
  `bundles/d1-omni-600m/macos-ship/<name>/` (the source folders are untouched). The operations did not change: the op
  counts before, after and on reload equal the export manifest's (decision 1,315, L4096 1,513, vision 758, audio
  2,560), and the `coreai-build inspect` summaries are equal. The local-path strings went from 2 to 0 per bundle, and
  the bundles lost 73–267 kB each (4,901,195,036 → 4,899,173,046 B together). A second strip of the vision bundle gave
  the same `main.mlirb`. Each `main.hash` changed (`provenance/strip.json` has both).
- The AOTs again (`aot_ship.py`): `compiled/ship-h16c/` (this Mac) and `compiled/ship-h19p/` (iPhone 18 Pro, not loaded
  anywhere). In all 20, `resources.bin` and `manifest.plist` equal the unstripped compiles' byte for byte (the same
  weights); `specialized_model_0.mpsgraph` shrank (L256: 368,766 → 210,881 B); no ANE region; no local path in any byte.
- The Python runtime gates on `ship-h16c` against the unstripped runs (`results/ship_runtime_compare.json`): all 12
  pairs SAME. Every row's marker logits and p are bit-equal (decision L256 / 512 / 1024 / 2048 / 4096: 436 / 49 / 30 /
  26 / 20 rows; the vision and audio end-to-end arms: 46 rows each), as are the vision and audio prefixes' statistics.
  Every summary number is unchanged, and the controls still FAIL.
- The Swift host on the stripped bundles (`results/ship_swift_parity.json`): 562 rows (text 470, image 46, audio 46),
  AOT and JIT, bit-equal to the Python runtime; FACTS §7 as in rounds 8–9 (text max |Δp| 0.0067, image 0.0062, audio
  0.0032). Every Python and Swift row, crop prefix and clip prefix equals the unstripped run's bit for bit. The L1024
  probe gives the Python runtime's response byte for byte (AOT, cold JIT, warm JIT).
- Hugging Face: `stage_hf.py --ship` → `hf_staging/d1-omni-600M-CoreAI/` (385 files besides `MANIFEST.json`, 14.8 GB,
  every graph and media file an APFS clone; round 8's folder kept as `.r8`), not uploaded. `macos/`, `ios/` and
  `ios-h19p/` hold the stripped bundles and their h19p compiles. New: `reference/LICENSE-MMLU-MIT.txt` (MMLU's MIT
  licence, for the `tv4_*` records), two NOTICE lines (the strip, MMLU), and in `host/` the Swift host's sources with
  `host/README.md` (what the host does, the JPEG rule). No staged file holds a local path or the user name: `grep -rIl`
  and `grep -rl -a` over all 386 files and `strings` over every graph binary find none.
- Speed was not measured again. The operations are the same and the compiled weights are byte-identical, so no
  speed change is expected; this expectation is not a measurement.

### Round 11 results (the iPhone 18 Pro, iOS 27.2 24B5099f, the gate app apps/D1OmniGate, on USB power)

- The device gate [`apps/D1OmniGate`](../../apps/D1OmniGate/README.md) runs the Swift host `apps/D1Omni` on the phone
  over 78 fixture rows (60 text, 9 image, 9 audio; the 18 text rows of at most 64 positions again on L64), with the
  stripped `.aimodel` files (specialized on the phone: JIT) and their h19p AOT `.aimodelc`. Its macOS build on this
  Mac reproduces the Mac's Swift runs of rounds 10 and 12 bit for bit on every row before the phone is touched.
- Parity on the phone: FACTS §7 PASS on JIT (78 rows: argmax 77/77 + the near tie, max |dp| 0.0059, mean 0.00090; L64
  rows 0.0015) and on AOT (74 rows; AOT = JIT bit for bit on every row). The phone's GPU does not give the Mac's bits
  (max |dp| against the Mac 0.0040); the host's arrays do: the decoded images, every crop and its four vision inputs,
  the samples, the mel and the four masks are bit-equal to the Mac's on all six media items.
- One decision on the phone (median of 20, JIT / AOT alternating in one process, thermal nominal throughout): W1 L64
  10.79 / 10.70 ms, L256 20.12 / 20.15; W2 L64 32.52 / 32.07, L256 60.60 / 60.64; W4 57.67 / 57.72 (vision 37.3 ms);
  W5 27.02 / 27.30 (audio 6.7 ms); W3 758.23 ms on JIT. First loads (JIT specializing on the phone): 0.4–1.8 s per graph.
- L4096 at the default memory limit: the JIT graph runs (peak footprint 2,778 MB); the AOT graph's first call took the
  footprint to 3,306 MB and the system killed the app (JetsamEvent, per-process-limit). The gate app cannot carry the
  increased-memory-limit entitlement (the team's wildcard profile).
- The Neural Engine: the h19p `.aimodelc` compiled with `--preferred-compute neural-engine` fails to load on the phone
  (`CoreAIDelegates.AIModelError.failedToSpecialize`, both graphs). The `.aimodel` specialized on the phone with the
  Neural Engine preferred loads: audio 10 s → 9 audio rows PASS, W5 59.98 ms (its audio graph 39.5 ms against the GPU's
  6.8 ms); decision L256 → 56 rows FAIL (argmax 33/55, as on the Mac), W1 155.9 ms. Whether those graphs ran on the
  Neural Engine is not confirmed from the app (its Core AI cache holds no region file).
- Lane records: `~/code/coreai/_d1_omni/results/iphone_{parity_jit,parity_aot,bench,ane}.json`, `ROUND11.md`.

### Round 12 results (the small decision buckets, Mac M4 Max, macOS 27.0 26A428, coreai-torch 0.4.1, coreai-build 3600.83.1)

- Which rows fit: of the oracle's 470 rows (`ref/records_ref.json`), 64 have ≤ 64 positions and 278 have 65–128, all text
  (the card's three questions are 47 / 56 / 51); no image row fits (the shortest is 136 positions) and 9 audio rows of
  `ref/records_ref_audio.json` do (aud_04, aud_05, aud_14: 34–98 positions). `results/r12_rowsets.json`.
- Export (`export_decide.py --precision fp16 --seq-len 64 / 128`): 1,315 ops and 8 softmax ops at both lengths (the
  same op count as L256–2048), 761.8 MB each. Export gate (CPU torch, the decomposed program =
  the eager module bit for bit): L64 64/64, max |Δp| 0.0073, mean 0.0012; L128 297/298 (the miss is a near tie),
  0.0120, 0.0014.
- Strip and AOT as round 10: `macos-ship-small/fp16-L{64,128}/` (op counts and `coreai-build inspect` unchanged, no
  local path), `compiled/ship-h16c/` and `compiled/ship-h19p/` (2.0–2.1 s each; `resources.bin` byte-equal across the
  unstripped h16c, the stripped h16c and the h19p compile; no ANE region; h19p never loaded on the Mac).
- Runtime gate on the GPU (Python runtime, AOT h16c): L64 64/64, max |Δp| 0.0058, mean 0.00095; L128 298/298 (the 8 near
  ties included; 278 native rows + 20 rows of ≤ 64 positions padded in), 0.0117, 0.0011; no drift over 3 repeats; the
  wrong-pairing controls FAIL. Stripped = unstripped bit for bit (`results/small_runtime_compare.json`: SAME).
- Audio end to end with the small buckets in the routing (`audio_check.py … --buckets` with L64 / L128 / L256 / L512):
  46/46, max |Δp| 0.0055, mean 0.00067; the 4 rows on L64 max 0.0021, the 5 on L128 0.0017; the 37 rows that stayed on
  L256 / L512 give round 10's numbers bit for bit (`results/r12_e2e_rows_check.json`).
- Swift host (round 9's binary, no source change: the buckets come from the folder names and metadata.json's
  `seq_len`): 342 rows (64 on L64, 278 on L128), AOT and JIT, bit-equal to the Python runtime; JIT = AOT; FACTS §7
  (`results/swift_parity_L64_128.json`).
- Speed (one window `d1d-r12`, 30 decisions per form after 5 warm-up rounds, the forms alternated in one process; no
  other GPU job ran; `results/timing/ranking_r12.json`): W1 L64 7.35 ms / L128 10.27 / L256 16.20; W2 21.97 / 30.91 /
  48.87; W5s audio 5 s + L128 17.72 / + L256 23.70; W5 audio 10 s + L256 24.94. The decision graph's call is 7.3 / 10.2 /
  16.1 ms at L64 / L128 / L256 (L512 was 28.7 in round 7). The ladder rows are in `results/timing/ladder_rows_r12.md`
  for the supervisor to merge; the Swift host was not timed at L64 / L128.


### Round 14 (the files to upload; nothing measured)

- `stage_hf.py --ship --final` → `hf_staging/d1-omni-600M-CoreAI/`: 452 files besides `MANIFEST.json`, 18.6 GB (APFS
  clones). Against round 13's folder: the 11 files of `ios-h19p/decide-fp16-L4096/` left out (767.6 MB; on the iPhone 18
  Pro that compile loaded and its first call was killed at the per-process memory limit, round 11), 22 files changed
  (README.md = the card, NOTICE.md, both metadata.json, host/README.md, the decision graphs' `runtime-gate.json` in
  `macos/` and `ios/`, three files of `reference/`), nothing else touched. The scans found nothing, the one allowed line
  aside (a comment of the Swift host on the runtime's cache location).
- A bare search for the tilde shorthand is no test in a binary file: the fp16 weights hold that byte pair 642,226 times
  in 52 files, and even a shape like `~/x…/x` appears in them by chance; binary files are scanned for the shorthand
  followed by a home folder name (0 hits) besides the users-folder prefix, the home directory and the user name.
- `zoo_transcripts.py` writes the iPhone transcript `gate-d1-omni-600m-iphone.json` from round 11's records (parity,
  AOT = JIT bits, the recomputed medians, loads, the L4096 compile's jetsam, the Neural Engine attempts, the hold and the
  transfer); the other eight rebuild byte for byte.
