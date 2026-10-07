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
| `make_fixtures.py` | the fixture: the Kev fixture's records as they are (its tweet_offensive records and its red arm left out), the model card's example requests, two long JSON states written for the port; `red_arms.json` (a word arm, two grammatical "not" arms, two state swaps) |
| `host.py` | the host specification (`tokenizers`, NumPy, json): the request checks, the prompt text, option codes and readout groups, the row and the Tree split, the readout arithmetic, the answers and the response, the graph's static-S rows |
| `test_host.py` | `host.py` against the provider's code (through `oracle_d1.dry_run`) and transformers' tokenizer on every fixture question with two tokenizers; the tokenizer contract, the readout's equivalence on random full-vocabulary logits, a request table, negative controls, the fixture's lengths |
| `oracle_d1.py` | the provider's code unchanged (`AutoModel` + `trust_remote_code`, fp32, CPU): the API path and the row form per question, Tree against row, the final-norm hook and its proof, determinism; `--dry-run` runs the same code on a stand-in backbone (no weights) |
| `lfm2_d1_decoder.py` | the decoder module: `Lfm2VlPipelinedForCausalLM` with an Identity head, hidden `[1, S, 2048]` out, the static-S export spec; run as a script it checks the module against the checkpoint's config and safetensors header with no weights |
| `toy_graph_check.py` | the decoder class at a toy config through export, `optimize`, `save_asset` and the Python runtime on the CPU, against its eager forward; the extension-id image rows |
| `export_decoder.py` | the bundle (round 3 with the weights): fp16 / int8lin / int8mix / int4lin, one static-S function `main`, the quantized set asserted equal to the recipe's; `metadata.json` (`decision-backbone`, the contract, `decision` with the option table, `vision`, `compression`), `tokenizer/` and `LICENSE` verbatim, `head/` the option rows; `--aot` compiles for the Mac GPU (h16c); `--toy` runs the same path on a toy config with random weights |
| `export_option_rows.py` | the option table (round 3 with the weights): the tied embedding rows of every readout candidate id, bf16 to fp32, read one row at a time and checked by a second reader; `--check` holds every fixture readout id against the table and the host's refusals against their controls |
| `readout_gate.py` | the gate (round 3 against the oracle): the AOT graph on the Mac GPU (`SpecializationOptions.default()`), the slot's hidden row through the host's readout, against the oracle; at most 40 rows a process and a re-run of its first row; the red arms with an oracle pre-check; `red`, `merge`, `red-records` |
| `parity_decoder_torch.py` | the decoder module in fp32 torch driven as the graph runs (round 3 against the oracle): P1 every row through the host's readout, P2 the chunk order against one forward, P3 the chunk widths, P4 the red arms; `toy` runs all four on the toy |
| `timing.py` | decision latency for the card's columns (round 3 on the model's bundle), inside a measurement window its caller holds; `--dry-run` prints the plan without a bundle |
| `decide.py` | (later) the Python reference read-out on the graph: a request to a response |

## Environment

- **The provider's code** (`oracle_d1.py`, `test_host.py`, `make_fixtures.py`): a private venv with transformers >= 5.14
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
HF_HUB_OFFLINE=1 $K/venv-oracle/bin/python oracle_d1.py --threads 1 --hidden tv4_000,card_refund,long_34k   # (round 2)
```

`make_fixtures.py` reads the Kev fixture from the lane's copy (`$K/fixtures/src/requests.json`, sha256 checked) and the
checkpoint's tokenizer to size the two long states. The fixture keeps the Kev records' gold keys (`"true"` / `"false"` for
a noul); d1 reports a noul's probabilities as `[yes, no]`, and `host.gold_key` maps the two. One copied question has no
`instructions` (`own_email_03` / `next_step`): the provider's `as_question` refuses it, and so does the host. The sciq,
emotion, qnli and paws records are for measurement only: their text does not go into a published fixture
(`sources.publication` in `records.json`).

The oracle asserts, per question, that the model's own `lm_head` applied to the final-norm hook's row at the answer slot,
minus its log-sum-exp, equals the row form's log-probabilities bit for bit; with `--hidden` it also keeps the rows and
checks the host's float64 gather against the fp32 logits. Record 0 runs again at the end and must be bit-equal.

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
$PY export_decoder.py fp16 --prefill-chunk 16 --aot --record $K/results/<fp16 record>.json
$PY export_decoder.py int8lin --prefill-chunk 16 --aot --record $K/results/<int8lin record>.json
$PY readout_gate.py red-records                    # -> $K/fixtures/red_arms_records.json, the arms for the oracle
HF_HUB_OFFLINE=1 $K/venv-oracle/bin/python oracle_d1.py --fixtures $K/fixtures/red_arms_records.json \
    --out-dir $K/oracle/red --results-dir $K/oracle/red
for k in 0 1 2; do $PY parity_decoder_torch.py p1 --shard $k --shards 3 --threads 1 & done; wait
$PY parity_decoder_torch.py p2; $PY parity_decoder_torch.py p3; $PY parity_decoder_torch.py p4
$PY parity_decoder_torch.py merge
$PY readout_gate.py run $K/exports/bundles/d1_3b_decode_fp16_pf16 --red --transcript $K/results/<gate fp16>.json
$PY readout_gate.py run $K/exports/bundles/d1_3b_decode_int8lin_pf16 --red \
    --compare-with $K/results/<gate fp16>.json --transcript $K/results/<gate int8lin>.json
<window holder> $PY timing.py run --bundle $K/exports/bundles/d1_3b_decode_fp16_pf16 \
    --gate-transcript $K/results/<gate fp16>.json --tag <tag>
```

The gate's bar is fixed in `readout_gate.py` and written at the top of each transcript before the first GPU process:
argmax against the oracle on every question that is not a near-tie, max |dp| and the mean of the rows' mean |dp| over
every option, a bit-equal re-run of each process's first row, finite hidden rows. The red arms come first on the oracle:
an arm that does not move the oracle's probabilities by the gate's own bar is listed for replacement, the graph must be
red on every arm that moves the oracle, and the graph's change must equal the oracle's.

The option table covers every readout id the host's rules can produce for noul, score and ASCII-letter or positional
choice questions, up to the alias pool's limit; a request that would read an id outside it (a one-letter native label
in another script) is refused whole by the host (`host.py` section 3, `decision.option_table` in the metadata).
