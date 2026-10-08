# d1 — LiquidAI d1-3B on Core AI (work in progress)

Scripts for [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) (revision `da1fe36a861f24690f27f622dca1d8688503d113`,
LFM Open License v1.0), a decision model post-trained from LFM2.5-VL-3B. It reads a state (text, JSON, images) and named
questions (`noul`, `choice`, `score`) and returns a probability per option from one forward pass, reading the logits of a
few option tokens at the last position; it generates nothing. The provider's runtime ships with the checkpoint
(`modeling_d1.py`, `hybrid.py`, `lfm2_vl.py`, `runner.py`, `prompt.py`, `api.py`); every gate here compares with that code
run in fp32 on the CPU.

The port's shape: the overlay's LFM2.5-VL text decoder with no vocabulary head, returning the final-norm hidden state at
every position of a static number of tokens a call; the host gathers the tied embedding rows of the question's option
tokens and computes their logits, max-pools each option's forms and softmaxes over the options, exactly the provider's
readout (the vocabulary's log-sum-exp cancels in a softmax over options).

| file | what it does |
|---|---|
| `make_fixtures.py` | the fixture: the Kev fixture's records as they are (its tweet_offensive records and its red arm left out), the model card's example requests, two long JSON states written for the port; `red_arms.json` (a word arm, two grammatical "not" arms, two state swaps); `--heldout`: the Core AI Kev lane's held-out set in d1's form (`heldout.json`, its tweet_offensive records left out), checked against `records.json` |
| `host.py` | the host specification (`tokenizers`, NumPy, json): the request checks, the prompt text, option codes and readout groups, the row and the Tree split, the readout arithmetic, the answers and the response, the graph's static-S rows |
| `test_host.py` | `host.py` against the provider's code (through `oracle_d1.dry_run`) and transformers' tokenizer on every fixture question with two tokenizers; the tokenizer contract, the readout's equivalence on random full-vocabulary logits, a request table, negative controls, the fixture's lengths |
| `oracle_d1.py` | the provider's code unchanged (`AutoModel` + `trust_remote_code`, fp32, CPU): the API path and the row form per question, Tree against row, the final-norm hook and its proof, determinism; `--dry-run` runs the same code on a stand-in backbone (no weights); --images runs the picture records through the provider's image path: the API path (one question = the whole prompt in one plain pass, several = the Tree), each question's plain pass, the image features and the final-norm rows of the --hidden records |
| `lfm2_d1_decoder.py` | the decoder module: `Lfm2VlPipelinedForCausalLM` with an Identity head, hidden `[1, S, 2048]` out, `image_embeds [N_IMAGE_TOKENS, 2048]` in (the image-row contract in its header), the static-S export spec; run as a script it checks the module against the checkpoint's config and safetensors header with no weights |
| `toy_graph_check.py` | the decoder class at a toy config through export, `optimize`, `save_asset` and the Python runtime on the CPU, against its eager forward; the extension-id image rows |
| `export_decoder.py` | the bundle (round 3 with the weights): fp16 / int8lin / int8mix / int8mlp / int8conv / int4lin at block 32 or 16 (`--quant-block`), one static-S function `main`, the quantized set asserted equal to the mode's; `metadata.json` (`decision-backbone`, the contract, `decision` with the option table, `vision`, `compression`), `tokenizer/` and `LICENSE` verbatim, `head/` the option rows; `--aot` compiles for the Mac GPU (h16c); `--toy` runs the same path on a toy config with random weights |
| `export_option_rows.py` | the option table (round 3 with the weights): the tied embedding rows of every readout candidate id, bf16 to fp32, read one row at a time and checked by a second reader; `--check` holds every fixture readout id against the table and the host's refusals against their controls |
| `readout_gate.py` | the gate (round 3 against the oracle): the AOT graph on the Mac GPU (`SpecializationOptions.default()`), the slot's hidden row through the host's readout, against the oracle; at most 40 rows a process and a re-run of its first row; the red arms of `--arms` (default round 4's set) with an oracle pre-check on that set's own oracle; `red`, `merge`, `red-records` |
| `parity_decoder_torch.py` | the decoder module in fp32 torch driven as the graph runs (round 3 against the oracle): P1 every row through the host's readout, P2 the chunk order against one forward, P3 the chunk widths, P4 the red arms; `toy` runs all four on the toy |
| `compute_unit_probe.py` | the speed ladder's compute-unit and AOT-flag facts (round 6a): `ane` compiles the decoder and the tower with the Neural Engine preferred, counts the Neural Engine regions, loads them with that preference and runs a fixed subset against the GPU assets' records and the oracle (plus the GPU assets loaded the same way); `noefr` compiles the decoder without `--expect-frequent-reshapes` and times each call of a row whose position length grows, twice in a process and in two processes; the transcript opens with the rules |
| `int8_bisect_torch.py` | which layers carry int8lin's error: P1's fp32 instrument with each of the 134 int8lin linears holding its checkpoint weight or the exporter's own int8 weight (read back through the finalized module's dequantization), on the int8lin gate's worst rows; the selection rule written to a file before any result (`rule`, `dump`, `run`, `plan`, `merge`) |
| `timing.py` | decision latency for the card's five columns (four text requests and one 384 px picture) on decide.py's path, inside a measurement window its caller holds, with every item's p and hidden rows checked against the gates' rows; `--dry-run` prints the plan (rows, calls) |
| `vision_host.py` | the image path's host specification (NumPy + Pillow): the provider's `cap_pixels`, the processor's crop plan (one crop or tiles + a thumbnail), torch's uint8 bicubic resize, patches and mask, the image token run and the extension ids, the position-table resize, the unshuffle index, the tower's four inputs per crop |
| `test_vision_host.py` | `vision_host.py` against the provider's image path (`cap_pixels` → transformers 5.19's processor, the row / trunk split): ids, pixel_values bit for bit, spatial shapes, mask, the position table, the unshuffle, the tower contract on a small random SigLIP2, negative controls, and the grid table |
| `make_fixture_images.py` | the CC0 fixture pictures (drawn shapes, seeded) and their records; the card's example picture as a URL only |
| `lfm2_vl_tower.py` | the vision tower in its exact form: SigLIP2 + projector with the crop's grid as inputs (patches, pos_table, key_bias, unshuffle_idx → image_embeds), every shape static; `from_hf` with a load report, `typed` (fp16 / fp16w32 / fp32), the toy snapshot (`write-toy`) and the scout against the checkpoint's config and safetensors header (`scout`) |
| `vision_oracle.py` | the tower's oracle: transformers 5.19's own loader (`Lfm2VlForConditionalGeneration.from_pretrained`, fp32, CPU) and `get_image_features` on d1-3B's checkpoint (`--toy`: the toy snapshot), every crop of the fixture's and the random pictures, with the host's four inputs per crop; the load report holds each of the checkpoint's 441 tower tensors against the loaded one, and the position table's sha256 ties the oracle to the tower bundle |
| `export_vision.py` | the tower bundle: fp16 / fp16w32 / fp32, `metadata.json` (`vision-tower`: the inputs and the output, the host's rules in short), `host/position_embedding.safetensors`, `LICENSE`; `--aot` compiles for the Mac GPU (h16c, no `--expect-frequent-reshapes`); `--toy`; the record counts the checkpoint values that fp16 storage changes and lists the compiled asset's files (resources.bin: what an fp16w32 asset really stores) |
| `gate_tower.py` | the tower gate: the AOT asset on the Mac GPU, every oracle crop, against transformers' rows (cosine, lowest row cosine, max \|d\|), a re-run in a fresh process, and two negative controls (no padding mask, the unshuffle index transposed); on the model: fp32 and fp16w32 hold every crop's and every row's cosine and re-run bit for bit, fp16 is recorded and stays a candidate only when every row's cosine holds 0.999; the oracle is tied to the bundle (the position table, the checkpoint) before any GPU process |
| `readout_gate_vision.py` | the picture path end to end on the Mac GPU: a picture record's file → `vision_host` → the tower's AOT asset → `image_embeds` → the decoder's AOT asset → the host's readout, against the provider's fp32 oracle (`oracle_d1.py --images`) in its two forms (each question's plain pass, the API path); the host's ids, groups and keys checked against the oracle's; arms `zero` (image_embeds zero: must be red), `reversed` (the crops in reverse order) and `hf_rows` (the decoder alone on transformers' tower rows); the tower against transformers per crop, the hidden rows' per-position cosine, at most 700 decoder calls a process and a re-run of its first row |
| `gate_swift.py` | the Swift host `apps/D1` against `host.py` / `vision_host.py` / `tokenizers` / `decide.py`, bit for bit: the text side (`all`: the request checks and the rendered text, every row's ids and readout groups, a picture's crop plan and token run, the readout arithmetic and the answers, three negative controls and the bundle's contract checks) and the graph side (`graph`: every row's hidden rows, p and response against the readout gate and `decide.py` on the same assets, shared and prepared, JIT against AOT, the tower's outputs, the picture rows end to end; §4) |
| `gate_swift_pixels.py` | the Swift pixel path (`apps/D1/Sources/D1/ImagePixels.swift`, through the `d1-pixels-test` CLI) against `vision_host.tower_inputs`, bit for bit: 18 PNG pictures (64 crops) and two EXIF JPEG pairs, decode probes, the d = 1152 position table, three negative controls, the JPEG subsampling differences (`source $ZOO_WORK_ROOT/_d1_3b/venv-oracle/bin/activate && python gate_swift_pixels.py all`) |
| `make_fixture_images.py --random --exif` | the six random-size pictures of `test_vision_host.py` as PNG and two EXIF-orientation JPEG pairs (with their decoded PNG twins) |
| `decide.py` | the Python reference read-out on the graph (the AOT assets, the Mac GPU): a request (with pictures: through the tower bundle) to rows, the decoder's calls direct, shared (the state's stable tokens once, the states copied per question) or on a prepared state, the readout and the response; `check` answers every fixture record from its raw request against the readout gate's transcript of the same asset, `e2e` runs round 3b's picture rows through the tower and the decoder; the specification `apps/D1`'s graph side copies |

## Environment

- **The provider's code and transformers' own image path** (`oracle_d1.py`, `test_host.py`, `make_fixtures.py`,
  `test_vision_host.py`, `vision_oracle.py`): a private venv with transformers >= 5.14
  (the checkpoint's tokenizer class `TokenizersBackend` and the provider's imports need transformers 5):

  ```bash
  cd $ZOO_WORK_ROOT/_d1_3b
  uv venv --python 3.12 venv-oracle
  VIRTUAL_ENV=$PWD/venv-oracle uv pip install "transformers>=5.14,<6" "torch==2.9.*" torchvision pillow safetensors \
      "huggingface_hub>=0.34" numpy "tokenizers>=0.22"
  ```

  The six `.py` files of the snapshot are copied unchanged into `$ZOO_WORK_ROOT/_d1_3b/src/d1/` (with an empty
  `__init__.py`), where the scripts import the provider's `prompt` and `api` from.
- **The decoder module and the toy graph:** the zoo's overlay venv (coreai-core, coreai-torch, torch, transformers 4.x);
  the decoder reads `config.json` and the weights with the overlay's own loader, so the transformers version does not
  matter there.
- **AOT:** Xcode 27.0 RC as `DEVELOPER_DIR`; `xcrun coreai-build compile <name>.aimodel --output <dir> --platform macOS
  --preferred-compute gpu --architecture h16c --expect-frequent-reshapes` (`export_decoder.py --aot` runs it). The gate
  and the timing load only the `.aimodelc`, with `SpecializationOptions.default()`; the Python runtime's JIT is not used.
  Loading an AOT asset makes an entry named by the asset's `main.hash` under
  `~/Library/Caches/coreai-cache/<build>/python/`: remove your own entries by that name when you are done (an asset of
  the same graph reuses the entry).
- **A shared machine:** on the machine this lane ran on, a guard holds back commands that mention export, compile,
  gate, parity, readout, timing or oracle while a timing window is open (the machine-wide GPU lock, `_paths.gpu_lock()`,
  holds a `timing` label); the steps below run through a wrapper that waits for the window to close, and `timing.py`
  runs inside a window its caller holds (it reads the lock and never writes it). The private venv's name contains one
  of those words.
- **Checkpoint files:** `HF_HUB_DISABLE_XET=1`, then `HF_HUB_OFFLINE=1`. Name the files: `hf download --include` takes
  one pattern, and further words become file names.

  ```bash
  hf download LiquidAI/d1-3B config.json processor_config.json tokenizer_config.json tokenizer.json chat_template.jinja \
      README.md LICENSE .gitattributes api.py hybrid.py lfm2_vl.py modeling_d1.py prompt.py runner.py \
      --revision da1fe36a861f24690f27f622dca1d8688503d113
  ```

Run from `conversion/d1`. `K=$ZOO_WORK_ROOT/_d1_3b`; everything the scripts write outside the repository goes under it
(`conversion/_paths.py`).

## 1. Fixture and oracle

```bash
$K/venv-oracle/bin/python make_fixtures.py     # -> $K/fixtures/records.json, red_arms.json, LICENSE-SemIf-MIT.txt
$K/venv-oracle/bin/python test_host.py         # -> $K/results/test_host.json, tokenizer_check.json, render_ids.json,
                                               #    fixture_lengths.json
HF_HUB_DISABLE_XET=1 hf download LiquidAI/d1-3B model.safetensors \
    --revision da1fe36a861f24690f27f622dca1d8688503d113   # the checkpoint; check its sha256 against the LFS oid
HF_HUB_OFFLINE=1 $K/venv-oracle/bin/python oracle_d1.py --threads 1 \
    --hidden tv4_000,card_refund,long_34k,own_long_log_10,semif_a3f18f3a63d45345942b,tv4x_qnli_00,tv4s_00
                                               # -> $K/oracle/records_oracle.json, oracle/hidden/<id>.npz,
                                               #    $K/results/oracle_summary.json
```

One thread: on the CPU the oracle's rows come out bit-equal at one and four threads and not at twelve (a long row's
hidden rows move), and one thread is the fastest of the three. `oracle_d1.py` runs the provider's code as the card
loads it (`AutoModel` + `trust_remote_code`, the remote-code copy under `~/.cache/huggingface/modules`), so the question
objects come from that copy's `prompt` (`provider(engine)`); `oracle_summary.json` carries the load report
(transformers' missing / unexpected keys, and every language tensor of the checkpoint bit-equal to the loaded
parameter), and the oracle's header (`model.provider_code`) the remote-code files' sha256 against the snapshot's.

`make_fixtures.py` reads the Kev fixture from the lane's copy (`$K/fixtures/src/requests.json`, sha256 checked) and the
checkpoint's tokenizer to size the two long states. The fixture keeps the Kev records' gold keys (`"true"` / `"false"` for
a noul); d1 reports a noul's probabilities as `[yes, no]`, and `host.gold_key` maps the two. One copied question has no
`instructions` (`own_email_03` / `next_step`): the provider's `as_question` refuses it, and so does the host. The sciq,
emotion, qnli and paws records are for measurement only: their text does not go into a published fixture
(`sources.publication` in `records.json`).

The oracle asserts, per question, that the model's own `lm_head` applied to the final-norm hook's row at the answer slot,
minus its log-sum-exp, equals the row form's log-probabilities bit for bit; with `--hidden` it also keeps the rows and
checks the host's float64 gather against the fp32 logits. Record 0 runs again at the end and must be bit-equal.

The held-out set (round 5c) checks a choice made on `records.json` (an int8 mode, a layer set) on records nothing was
chosen on:

```bash
$PY make_fixtures.py --heldout                 # -> $K/fixtures/heldout.json (never overwritten)
HF_HUB_OFFLINE=1 $K/venv-oracle/bin/python oracle_d1.py --threads 1 --fixtures $K/fixtures/heldout.json \
    --out-dir $K/oracle/heldout --results-dir $K/oracle/heldout
```

`--heldout` reads the Core AI Kev lane's held-out file (`$ZOO_WORK_ROOT/_kev/fixtures/heldout.json`, sha256 pinned in
the script: transfer-v4 development records past the fixture's slices) and writes only `heldout.json`: its
tweet_offensive records left out as in `records.json`, the request's `model` key dropped, ids and gold kept, Kev's
provenance kept with that file's sha256 and each row's `_meta.id`. It asserts that no id and no transfer-v4 line is
shared with `records.json`, and lists in `summary.checks_against_records_json` the records whose request still renders
to a `records.json` row (a `permuted` mmlu row whose option order came out unchanged) and those that only reorder a
fixture record's options; a verdict on the held-out set can be read with and without the first kind. Give the oracle
its own `--results-dir`: the default is the fixture oracle's summary. The held-out text stays local, as the fixture's
measurement-only sources do.

## 2. The decoder: module, export and gate

```bash
$PY lfm2_d1_decoder.py <snapshot>/config.json <safetensors header json> --out $K/results/overlay_scout.json
$PY toy_graph_check.py --eager-only                # torch only
$PY toy_graph_check.py                             # export, the Python runtime on the CPU
```

The CPU-only runtime loads the decoder graph and refuses its first call (`inferenceFailed(-1)`): it does not run the
graph's dynamic dimensions (the position input and the KV sequence axis); the same module with every shape static runs.
The toy check therefore records the refusal and runs the module on the CPU as a chain of static graphs, one per call
position, the states carried between them. The shipped graph runs on the GPU, compiled ahead of time.

### Export, the option table and the gate on the toy

The same scripts run end to end with no checkpoint: `--toy` builds the decoder at the toy config with seeded random
weights, and the gate's toy oracle is that module's own fp32 forward over the fixture rows (their ids folded into the
toy vocabulary, random readout groups of the real groups' shape). A toy checks the instruments, not a model: its
readout does not follow the text, so the red arms are judged on the model (below).

```bash
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
$PY export_decoder.py fp16 --toy --prefill-chunk 16 --aot --record $K/results/<toy fp16 record>.json
$PY export_decoder.py int8lin --toy --prefill-chunk 16 --aot --record $K/results/<toy int8lin record>.json
$PY parity_decoder_torch.py toy --toy-oracle $K/oracle_toy/records_oracle.json \
    --table $K/exports/toy_bundles/d1_toy_decode_fp16_pf16/head --red --out $K/results/<toy parity>.json
$PY readout_gate.py run $K/exports/toy_bundles/d1_toy_decode_fp16_pf16 --toy-oracle $K/oracle_toy/records_oracle.json \
    --red --transcript $K/results/<toy gate fp16>.json
$PY readout_gate.py run $K/exports/toy_bundles/d1_toy_decode_int8lin_pf16 --toy-oracle $K/oracle_toy/records_oracle.json \
    --red --compare-with $K/results/<toy gate fp16>.json --transcript $K/results/<toy gate int8lin>.json
$K/venv-oracle/bin/python export_option_rows.py --check --self-test --out $K/results/<option rows check>.json
$PY timing.py run --dry-run
```

### Export and the gate on the model (needs `model.safetensors`)

```bash
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
$PY export_option_rows.py --write $K/exports/head --check --out $K/results/<option rows check>.json
$PY export_decoder.py fp16 --prefill-chunk 16 --aot --record $K/results/<fp16 record>.json
$PY readout_gate.py red-records                    # -> $K/fixtures/red_arms_records.json, round 1's arms for the oracle
HF_HUB_OFFLINE=1 $K/venv-oracle/bin/python oracle_d1.py --threads 1 --fixtures $K/fixtures/red_arms_records.json \
    --out-dir $K/oracle/red --results-dir $K/oracle/red
HF_HUB_OFFLINE=1 $K/venv-oracle/bin/python oracle_d1.py --threads 1 --fixtures $K/fixtures/red_arms_r4_records.json \
    --out-dir $K/oracle/red_r4 --results-dir $K/oracle/red_r4     # round 4's arms (the gate's default set)
for k in 0 1 2; do $PY parity_decoder_torch.py p1 --oracle <oracle> --hidden-npz $K/oracle/hidden \
    --shard $k --shards 3 --threads 1 & done; wait
$PY parity_decoder_torch.py p2 --threads 1; $PY parity_decoder_torch.py p3 --threads 1
$PY parity_decoder_torch.py p4 --threads 1
$PY parity_decoder_torch.py merge --oracle <oracle> --out $K/results/<parity>.json
$PY readout_gate.py run $K/exports/bundles/d1_3b_decode_fp16_pf16 --red --transcript $K/results/<gate fp16>.json
$PY decide.py check --bundle $K/exports/bundles/d1_3b_decode_fp16_pf16 --gate $K/results/<gate fp16>.json \
    --records <the multi-question records> --out $K/results/<check>.json --shared-out $K/results/<shared vs direct>.json
<window holder> $PY timing.py run --bundle $K/exports/bundles/d1_3b_decode_fp16_pf16 \
    --tower $K/exports/vision/d1_3b_vision_fp16w32 --gate-transcript $K/results/<gate fp16>.json \
    --e2e-transcript $K/results/<e2e gate fp16w32>.json --tag <tag>
```

`timing.py` runs decide.py's decisions (the picture item through the tower, its decoding and cutting once per process
outside the timed decision) and names its bundle's cache entry the interpreter's way: forms whose AOT assets share a
`main.hash` (int8mlp at block 32 and 16) are timed with their own entries, by the interpreter's file name
(`.venv/bin/python3.11` keeps its entries in `python3-11/`) or by moving one entry aside, never with whichever is there.

`parity_decoder_torch.py p1` runs every row of the oracle document it is given (`--oracle`, default
`$K/oracle/records_oracle.json`): a subset is the same layout with fewer records (per source the first records, plus
the records whose hidden rows the oracle kept, for the position cosine). P2, P3 and P4 read the whole oracle.

The gate's bar is fixed in `readout_gate.py` and written at the top of each transcript before the first GPU process:
argmax against the oracle on every question that is not a near-tie, max |dp| and the mean of the rows' mean |dp| over
every option, a bit-equal re-run of each process's first row, finite hidden rows. The red arms come first on the oracle:
an arm that does not move the oracle's probabilities by the gate's own bar is listed for replacement, the graph must be
red on every arm that moves the oracle, and the graph's change must equal the oracle's.

The arms are `--arms` (`run --red` and `red`): by default `$K/fixtures/red_arms_r4.json`, round 4's set, in which the
word arm moves the oracle (`$K/fixtures/red_arms.json`, round 1's set, stays as its record and is a toy bundle's
default). Their oracle is the `$K/oracle/*/records_oracle.json` whose fixture file names the arms file's sha256
(`--red-oracle` to name it); an oracle run on another arms file is refused. The transcript names the arms file and its
sha256.

### Quantized modes, the int8 bisect and int8mix (needs `model.safetensors`)

```bash
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
$PY export_decoder.py int8lin --prefill-chunk 16 --aot --record $K/results/<int8lin record>.json
$PY readout_gate.py run $K/exports/bundles/d1_3b_decode_int8lin_pf16 --red \
    --compare-with $K/results/<gate fp16>.json --transcript $K/results/<gate int8lin>.json
$PY export_decoder.py int4lin --prefill-chunk 16 --aot --record $K/results/<int4lin record>.json   # --quant-block 16
mv ~/Library/Caches/coreai-cache/<OS build>/python/<main.hash> <that path>__<mode>                  # see below
$PY readout_gate.py run $K/exports/bundles/d1_3b_decode_int4lin_pf16 --red \
    --compare-with $K/results/<gate fp16>.json --transcript $K/results/<gate int4lin>.json
# int8lin over the bar: which layers carry the error (fp32 torch, the exporter's own int8 weights)
$PY int8_bisect_torch.py rule --gate $K/results/<gate int8lin>.json        # $K/bisect/rule.json, before any result
$PY int8_bisect_torch.py dump
$PY int8_bisect_torch.py run --part $K/bisect/parts/part_a.json --configs all_int8,exact,only_mlp_int8,only_conv_int8,..
$PY int8_bisect_torch.py plan                                               # what the rule asks next
$PY int8_bisect_torch.py merge --out $K/results/<bisect>.json
$PY export_decoder.py int8mix --fp16-layers <the chosen layers> --prefill-chunk 16 --aot --record $K/results/<record>.json
$PY readout_gate.py run $K/exports/bundles/d1_3b_decode_int8mix_l<..>_pf16 --red \
    --compare-with $K/results/<gate fp16>.json --transcript $K/results/<gate int8mix>.json
# round 5c: the block size and the kind of linear
$PY export_decoder.py int8lin --quant-block 16 --prefill-chunk 16 --aot --record $K/results/<record>.json  # .._int8lin_b16_..
$PY export_decoder.py int8mlp --prefill-chunk 16 --aot --record $K/results/<record>.json   # the conv-mixer projections fp16
$PY export_decoder.py int8conv --prefill-chunk 16 --aot --record $K/results/<record>.json  # the MLP linears fp16: a map
$PY export_decoder.py int8mlp --quant-block 16 --prefill-chunk 16 --aot --record $K/results/<record>.json
# a mode that passes the fixture gate, on the held-out set (§1; no red arms: their base rows are fixture records)
$PY readout_gate.py run $K/exports/bundles/d1_3b_decode_fp16_pf16 --oracle $K/oracle/heldout/records_oracle.json \
    --tag heldout_fp16_pf16 --transcript $K/results/<gate fp16 held-out>.json
$PY readout_gate.py run $K/exports/bundles/<the mode's bundle> --oracle $K/oracle/heldout/records_oracle.json \
    --tag heldout_<mode> --compare-with $K/results/<gate fp16 held-out>.json --transcript $K/results/<gate held-out>.json
```

The quantizer prints two warnings and only these: `Tensor size 1 along axis 1 is not divisible by block size 32.
Skipping quantization.` (22 times, int8 and int4, and with `block size 16` at block 16) and `dynamic_shapes is only
supported in graph mode and will be ignored.` (coreai-opt's eager mode, which the weight-only recipe uses, does not read
input shapes); any other warning, or a quantized set other than the mode's, stops the export.

The held-out gate reads the held-out oracle (`--oracle`) and keeps its shards apart (`--tag`); the bar is the gate's
own. Which modes go to it and which one is the candidate is written down before the first int8 gate of a round, and
nothing is chosen on held-out numbers.

The AOT asset's `main-h16c.mlirb` holds the function's type and the source files' paths and sha256, not the weights:
the `main.hash` follows which linears are quantized, not the bit width or the block size (int8lin at block 32 and 16
and int4lin share one name, int8mlp at block 32 and 16 another), and the Python runtime names its cache entry by it
(`~/Library/Caches/coreai-cache/<OS build>/python/<main.hash>/`). Loading the second mode's asset while the first
mode's entry is there can run the first mode's graph. Before gating (or timing) another mode of the same graph,
move the entry aside, and after the run check that the entry's `manifest.plist` has the asset's own sha256.

`int8_bisect_torch.py` takes its rows from the int8lin gate (its worst rows, identical rows counted once) and writes
the rule before any result: the bound on the worst row, no row worse than all-int8, the most fp16 layers, the
smallest set, layers ranked by the mean over the rows. A set chosen outside the rule is only a candidate: the rows that
chose it cannot test it, and it needs a held-out set.

The option table covers every readout id the host's rules can produce for noul, score and ASCII-letter or positional
choice questions, up to the alias pool's limit; a request that would read an id outside it (a one-letter native label
in another script) is refused whole by the host (`host.py` section 3, `decision.option_table` in the metadata).

### Chunk widths and the compute-unit probes (round 6a)

The forms a speed ladder compares are made and gated here; their speed is measured in a timing window, not by these
commands. A chunk width is the bundle's static S (`--prefill-chunk`, the name's `_pf<S>`): a row runs as ⌈T / S⌉ calls
and the last call is padded, so a wider S makes fewer calls and pads more. The gate reads S from the bundle's
`metadata.json`; `--compare-with` the same mode's S = 16 transcript shows what the width alone moves (the fp16 graph
rounds in another order at another width; in fp32 torch the widths are bit-equal, `parity_decoder_torch.py p3`).

```bash
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
for S in 32 64; do
  $PY export_decoder.py fp16 --prefill-chunk $S --aot --record $K/results/<fp16 S record>.json
  $PY export_decoder.py int8mlp --prefill-chunk $S --aot --record $K/results/<int8mlp S record>.json
  $PY readout_gate.py run $K/exports/bundles/d1_3b_decode_fp16_pf$S --red \
      --compare-with $K/results/<gate fp16 S=16>.json --transcript $K/results/<gate fp16 S>.json
  $PY readout_gate.py run $K/exports/bundles/d1_3b_decode_int8mlp_pf$S --red \
      --compare-with $K/results/<gate int8mlp S=16>.json --transcript $K/results/<gate int8mlp S>.json
done
$PY compute_unit_probe.py ane --transcript $K/results/<ane probe>.json
$PY compute_unit_probe.py noefr --transcript $K/results/<noefr probe>.json
# the efr-less asset again, the Neural Engine preferred at load; a static tower that computes in fp16 (a diagnostic)
$PY compute_unit_probe.py noefr --options neural_engine --asset $K/exports/probe_noefr/d1_3b_decode_fp16_pf16.h16c.aimodelc \
    --processes 1 --tag _ane --transcript $K/results/<noefr ane probe>.json
$PY compute_unit_probe.py ane --graphs tower --tower $K/exports/vision/d1_3b_vision_fp16 \
    --tower-gate $K/results/<gate tower fp16>.json --no-control --tag _fp16diag --transcript $K/results/<diag>.json
```

Each S compiles to its own `main.hash` (the function's type holds S), but check the runtime cache entry before a gate
all the same (above). `compute_unit_probe.py ane` compiles the decoder (`--expect-frequent-reshapes`, as it ships) and
the tower (without it, as it ships) with `--preferred-compute neural-engine` (the flag's values are `gpu`,
`neural-engine` and `none`) into `$K/exports/probe_ane/`, counts the compiled asset's Neural Engine regions (the unique
`*_ANE_region_<n>` names in its tree), loads it with
`SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.neural_engine())` (`neural_engine` is a method:
call it) and runs a fixed subset of rows and crops against the GPU asset's own record, bit for bit, and against the
oracle. `noefr` compiles the decoder without `--expect-frequent-reshapes` into `$K/exports/probe_noefr/` and times the
calls of one row whose position length grows by S each call, twice in a process and in two processes. A probe asset
shares its source `.aimodel` with a shipped asset; the script renames a same-named runtime cache entry aside before it
loads the probe asset, renames the probe's own entry `<hash>__probe_r6a_<label>` after the run and puts the aside entry
back; remove the probe entries by name when you are done. Inside an entry the runtime keeps one directory per set of
specialization options (`default()` and a neural-engine preference get two), so loading a shipped asset with other
options adds a second copy to its entry; remove that directory by name too. The runtime's `Profiler` is not used: with
callbacks, `load_function(.., profiler=)` stops the process (SIGTRAP), and without them it records nothing.

Where the Neural Engine shows up: an AOT compile of the dynamic decoder makes no Neural Engine region whatever the
preference (its compiled graph holds no ANE message), while the static tower's compile tries it and leaves the
validation messages in its `*.mpsgraph`. An asset compiled without `--expect-frequent-reshapes` is specialized by the
runtime at each new position length, and that specialization does try the Neural Engine (with `default()` too): it
writes a region's IR and compiler options under `$TMPDIR/com.apple.MetalPerformanceShadersGraph/mpsgraph-<pid>-*`,
calls the Neural Engine compiler, prints its failure on stderr and runs the GPU. The probe reads both and lists the
worker's scratch; that scratch stays after the process exits (one region IR per new length, each about as large as
the MLP and conv-mixer linears' fp16 weights), so remove your own `mpsgraph-<pid>-*` directories by name after a probe.

## 3. Vision

A picture reaches the decoder in three steps, each with its own gate:

1. **The host** (`vision_host.py`): the picture becomes crops — one crop at the picture's aspect (at most 1024 patches,
   both sides even), or 512 × 512 tiles plus a thumbnail — and each crop becomes the tower's four inputs. The prompt's
   `<image>` becomes the picture's token run, and every `<image>` of the row becomes an extension id `V + k`, k counted
   over every picture and crop. `test_vision_host.py` holds all of it against the provider's code and transformers'
   processor.
2. **The tower** (`lfm2_vl_tower.py`, one graph for every crop): `patches [1024, 768]` + `pos_table [1024, 1152]` (the
   16 × 16 table resized to the crop's grid by the host) + `key_bias [1024]` (0 for a patch, -inf for padding) +
   `unshuffle_idx [256, 4]` → `image_embeds [256, 2048]`, of which the first h·w/4 rows are the crop's tokens. The
   bundle ships the position table next to the graph (`host/`). Three forms: fp16, fp16w32 (fp16 weights, fp32 math:
   the form a tower whose fp16 math misses ships in), fp32.
3. **The decoder's image rows**: `image_embeds [2816, 2048]` (N = `lfm2_d1_decoder.N_IMAGE_TOKENS`). One picture needs
   at most 2,810 rows, so N holds any one picture; the crops' rows are concatenated in crop order, the rest is zero,
   and the same buffer is bound to every call of the row. A request whose pictures need more than N rows together, and
   a row over the position bound, are refused by the host before any graph call (`vision` in the decoder's metadata).

### The tower on the toy

The toy is a small SigLIP2 + projector at the checkpoint's patch size, position grid and activations, with the toy
decoder's width; its snapshot is laid out like the checkpoint (the same tensor names, BF16), so the toy runs through the
same `from_hf` as the model and transformers loads it with its own loader. The oracle is transformers 5.19's
`get_image_features` on every crop of the fixture's 12 pictures and 6 random ones.

```bash
SNAP=<the pinned snapshot>   # hf_snapshot("LiquidAI/d1-3B", revision=...)
$PY lfm2_vl_tower.py scout $SNAP/config.json <safetensors header json> --out $K/results/<tower scout>.json
$PY lfm2_vl_tower.py write-toy --seed 0 --template-config $SNAP/config.json --out $K/oracle_toy_vision/toy_snapshot
source $K/venv-oracle/bin/activate && python vision_oracle.py --toy     # -> $K/oracle_toy_vision/<picture>/<crop>.npz
for d in fp32 fp16w32 fp16; do
  $PY export_vision.py --toy --dtype $d --aot --record $K/results/<toy tower export $d>.json
  $PY gate_tower.py run $K/exports/toy_vision/d1_toy_vision_$d --transcript $K/results/<toy tower gate $d>.json
done
```

`gate_tower.py` writes its bar into the transcript before the first GPU process: fp32 must match transformers within
fp32 rounding on every crop and reproduce itself bit for bit in a fresh process; fp16w32 must hold every crop's and
every row's cosine; fp16 is recorded. Both negative controls must miss the bar on every crop they can move.

### The tower on the model (needs `model.safetensors`)

transformers' own loader reads the checkpoint (the whole model, fp32, on the CPU; the language model is loaded and
never runs) and `get_image_features` gives every crop's rows; the npz carry the host's four inputs made with the loaded
position table, which is the table the bundle ships.

```bash
source $K/venv-oracle/bin/activate && HF_HUB_OFFLINE=1 python vision_oracle.py   # -> $K/oracle/images/<picture>/<crop>.npz
for d in fp16w32 fp16 fp32; do
  $PY export_vision.py --dtype $d --aot --record $K/results/<tower export $d>.json
  $PY gate_tower.py run $K/exports/vision/d1_3b_vision_$d --transcript $K/results/<tower gate $d>.json
done
```

On the model `gate_tower.py` holds fp32 and fp16w32 to every crop's and every row's cosine and a bit-for-bit re-run in a
fresh process; fp16 is recorded and stays a candidate only when every row's cosine holds. It ties the oracle to the
bundle (the position table and the checkpoint) before any GPU process. A decoder for another number of image rows takes
`export_decoder.py --n-image-tokens N` (default N_IMAGE_TOKENS; another N adds `_n<N>` to the name).

### The picture rows end to end (needs `model.safetensors`)

The provider's own image path is the oracle: one question is the whole prompt in one plain pass, several questions one
Tree over the processor's trunk; each question's plain pass is kept beside it.

```bash
source $K/venv-oracle/bin/activate && HF_HUB_OFFLINE=1 python oracle_d1.py --images --threads 1 \
    --hidden img01_shapes_384x384,img06_grid_1024x768,img12_small_300x300 \
    --summary $K/results/<image oracle summary>.json    # -> $K/oracle/records_oracle_images.json, oracle/images_hidden/
$PY readout_gate_vision.py run --tower $K/exports/vision/d1_3b_vision_fp16w32 \
    --decoder $K/exports/bundles/d1_3b_decode_fp16_pf16 --transcript $K/results/<e2e gate>.json
$PY decide.py run --bundle $K/exports/bundles/d1_3b_decode_fp16_pf16 --tower $K/exports/vision/d1_3b_vision_fp16w32 \
    --request <request JSON with "images": [paths]> --out <response>.json --trace <trace>.json
```

`readout_gate_vision.py` holds §2's bar against the oracle in both its forms; its `zero` arm (image_embeds zero, the
extension ids kept) must fail it, and its `hf_rows` arm runs the decoder on transformers' tower rows, which splits an
end-to-end error between the tower and the decoder. The card's example picture is a URL and is not fetched (its record
is skipped by the oracle and the gate).

## 4. The hosts: `decide.py` and the Swift host

`decide.py` is the whole read-out in Python on the AOT assets (the Mac GPU, `SpecializationOptions.default()`): the
request → `host.build_request` → one row per question (with pictures: `vision_host` → the tower bundle per crop → the
image rows, and each row's processor ids with `<image>` → 128,000 + k) → the decoder's S-id calls from zero states →
the slot's hidden row → `host.readout` on the bundle's option table → `host.response`. Shared runs the state's first
floor(Ls / S) · S ids once and every question's rest from a copy of the states: Ls is the prefix's ids without its last
pre-token (`":\n"`, the one piece a question's first characters can change), so every row starts with those ids, and on
a static-S graph the calls are the direct run's (the same hidden bits). `prepare` / `decide_prepared` split the same
calls in two. A toy bundle folds every real id into its vocabulary (round 2a's gate: id % V; 128,000 + k → V + k) and
reads the toy oracle's random groups (`--toy-oracle`).

The Swift host `apps/D1` copies it (`D1Decider.loadGraph`, `decide` / `trace` / `prepare`; `Decoder.swift`,
`Tower.swift`), and `gate_swift.py graph` compares the two on the same assets: every row's hidden rows (sha256 of the
fp16 bytes) against the readout gate's transcript and `decide.py check`, p bit for bit, the response byte for byte,
shared = direct and prepared = shared, the bundle's `.aimodel` specialized by Swift against the AOT asset, the tower's
outputs against `gate_tower.py`'s, and round 3b's picture rows end to end.

```bash
PY=<coreai-models venv>/bin/python Q="$HOME/code/standup/tools/quiet/quiet_wait.py --max-wait 3600 --"
K=$ZOO_WORK_ROOT/_d1_3b
$Q $PY decide.py run --bundle <bundle> --request req.json [--tower <tower bundle>] [--shared] --out resp.json --trace trace.json
$Q $PY decide.py check --bundle $K/exports/toy_bundles/d1_toy_decode_fp16_pf16_tbl2 \
    --toy-oracle $K/oracle_toy_tbl2/records_oracle.json --gate $K/results/<the readout gate's transcript> \
    --out $K/results/<check>.json --shared-out $K/results/<shared vs direct>.json
$Q $PY decide.py e2e --bundle $K/exports/toy_bundles/d1_toy_decode_fp16_n2816_pf16 \
    --tower $K/exports/toy_vision/d1_toy_vision_fp32 --eager --out $K/results/<e2e>.json
DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer \
    swift build -c release --package-path ../../apps/D1 --scratch-path $K/swift/.build
source $K/venv-oracle/bin/activate && $Q python gate_swift.py graph   # -> $K/results/r3c_swift_graph.json
```

`check` answers every fixture record from its raw request (a refused request: its text, and each of its questions
alone), direct and shared (prepared on the requests of two or more questions), at most 40 records a process with a
re-run of the process's first record at its end. `gate_swift.py graph` reads the Python records named in its header
(`PY_CHECK`, `PY_E2E`, the readout gates' and the tower gates' transcripts) and runs the Swift passes it does not find.
