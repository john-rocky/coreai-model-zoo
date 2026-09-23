# letter — gates for OpenJev and plain decision-function letter readouts

These gates cover `apus-ailab/APUS-OpenJev-v1-4B` at revision
`65797c526c27c4d24f564333779162cd4a64328e`. This Qwen3.5-4B checkpoint retains its
tied language-model head. Its decision readout selects the A–P token logits at
the first output position after the author's compiled chat prompt, then applies
float32 softmax at T = 1. The author's API reports `calibrated: false`.

1. `oracle_letter.py` vendors the snapshot's five `openjet_runtime` files
   unchanged and verifies each SHA-256 against `evaluation/runtime-smoke.json`.
   It embeds 48 requests: 28 `choice`, 12 `noul`, and 8 `score_level`. Sixteen
   requests contain Chinese and 30 have structured state strings. Eight choice
   rows have 16 criteria, eight have 2–3, and twelve have 4–8. State lengths are
   34–66 tokens; compiled prompts are 124–1,672 tokens. Two rows marked
   `zoo_only` have 1,672 and 1,670 compiled tokens. Fresh generation calls the
   author's `OpenJet.from_pretrained(..., device="cpu", dtype="float32")`,
   `compile()`, and both `decide(..., effort="high")` and `decide(..., effort="low")`.
   Every high API probability must exactly equal independent fp32 softmax of
   its recorded label logits. Layer hooks only observe the 32 high / 16 low
   execution indices. The low-effort results are evidence only; that path needs
   a separate graph and is not exported here.
2. `readout_gate_letter.py` runs the exported full-depth bundle using the zoo's
   frozen overlay interpreter. It AOT-compiles for h16c GPU, or reuses an
   explicitly selected `--aot-asset`, and loads with
   `SpecializationOptions.default()`. Each prompt starts with four fresh zero
   states and supplies its entire compiled `ids` sequence as S=1 steps. The
   final fp16 `[1, 1, 248320]` logits are gathered at `label_ids` and softmaxed in
   float32. Each process handles at most 15 prompt evaluations including its
   reset, and at most 14,000 calls. Each wide row gets its own process. Every
   process repeats its first row, and a final process repeats the fixture's
   first row. The transcript includes full-vocabulary argmax text, finite
   values, raw-logit and probability errors, near ties, and the five worst rows.
3. `engine_argmax_letter.py` invokes a Release `llm-runner` once per row for each
   of `coreai-pipelined` and `coreai-sequential`, using `--raw-tokens`,
   `--max-tokens 1`, `--temperature 0.0`, and `--warmup off`. Expected text is
   `tokenizer.decode([full_vocab_argmax_id])` from the same bundle's Python
   readout. Exact agreement with the oracle's best label is counted separately.
   Stdout/stderr are retained for all 96 calls. The parser removes only the
   runner's exact framing and preserves generated whitespace. Mismatches are
   recorded without retries. This tests decoded text; different token IDs may
   decode to identical text.

The numerical readout gate requires matching option argmax on all rows with
oracle margin ≥ 0.02, finite/nonconstant logits, a label as full-vocabulary
argmax on at least 90% of rows, and bit-identical reset logits. It records max
|Δp| and the mean of row means. The provisional int8hu expectation of max |Δp|
≤ 0.05 is reported as `REVIEW_REQUIRED` when exceeded, without further tuning.
The accepted fixture has no near ties; its minimum top-two margin is 0.948419.
The engine gate requires every call to succeed and all emitted text to match.

The oracle's uv header pins Python 3.11, Torch 2.9.0, Transformers 5.16.1,
safetensors 0.8.0, huggingface_hub 1.32.0, numpy 2.3.5, and tokenizers 0.23.2.
The author's constructor explicitly requires Transformers 5.16.1. The readout
and engine scripts use the overlay environment: coreai-core 1.0.0b2,
coreai-torch 0.4.1, coreai-opt 0.2.1, Torch 2.9.0, Transformers 4.57.6,
safetensors 0.7.0, huggingface_hub 0.36.2, numpy 2.3.5, tokenizers 0.22.2, and
accelerate 1.15.0. Install `coreai_models` from the frozen fork Python tree
at `397b337e234474a191c0bd96ac9ef71c4f808a3d`; do not import a shared working tree.

From the zoo root, with a RUN-local cache and the frozen overlay interpreter:

```sh
export HF_HUB_DISABLE_XET=1
export COREAI_CHUNK_THRESHOLD=1
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
uv run conversion/letter/oracle_letter.py \
  --hf-id apus-ailab/APUS-OpenJev-v1-4B \
  --revision 65797c526c27c4d24f564333779162cd4a64328e \
  --out models/apus-openjev-v1-4b/fixtures-apus-openjev-v1-4b.json
python conversion/letter/readout_gate_letter.py \
  exports/apus_openjev_v1_4b_decode_int8hu_block32_sym \
  models/apus-openjev-v1-4b/fixtures-apus-openjev-v1-4b.json \
  --snapshot "$PINNED_SNAPSHOT" \
  --transcript models/apus-openjev-v1-4b/gate-apus-openjev-v1-4b-readout.json
python conversion/letter/engine_argmax_letter.py \
  exports/apus_openjev_v1_4b_decode_int8hu_block32_sym \
  models/apus-openjev-v1-4b/fixtures-apus-openjev-v1-4b.json \
  --readout models/apus-openjev-v1-4b/gate-apus-openjev-v1-4b-readout.json \
  --runner "$ZOO_LLM_RUNNER" \
  --transcript models/apus-openjev-v1-4b/gate-apus-openjev-v1-4b-engine.json
```

`--snapshot` can select a pinned local snapshot for the oracle too. Without it,
the oracle downloads that revision with at most eight workers. `--work-dir`
places vendored code and progress evidence (oracle), full logits and sessions
(readout), or raw token files and runner logs (engine). Defaults are beside the
output. `--deadline-epoch` bounds each command in Unix seconds; child processes
set their own alarm before exec, and the controller never signals a child.

To preserve an accepted oracle without rerunning numerical inference, add
`--replay-fixtures /path/to/accepted-fixtures.json` to the oracle command. This
mode vendors and verifies the author runtime, re-compiles every request,
checks all IDs and label boundaries, independently checks the saved fp32
softmax/API equality, and copies the fixture byte-for-byte. It writes
`oracle_replay.json` as an explicitly non-inference proof. Round 2 used this
mode; fresh CPU inference was accepted in round 1. The subsequent readout and
both-engine proofs were executed from this portable zoo layout.

The fixture schema is `coreai-letter-fixtures/1`; transcripts use
`coreai-letter-readout-gate/1` and `coreai-letter-engine-gate/1`, retaining the
decider/slot `rows`, `summary`, and `result` envelope. Their summary companion
uses `coreai-letter-gate-summary/1`. Evidence is written incrementally.

On macOS 27.0 (26A428), the Python GPU JIT path gives incorrect logits for this
graph, so the gate uses AOT. The runtime also leaks an IOSurface per call, which
is why processes are split. Gate wall times are correctness evidence only;
throughput was measured separately before these gates. Reference forms were
read from zoo `main:conversion/decider/README.md` and the read-only OpenThai
round-2 slot scripts.

## Plain-text decision-function form

`chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16`, revision
`69a91b6778df7d11b9e13e0f7d4c6a5aa6ad1bc5`, uses the same tied language-model
head but reads space-prefixed letter tokens: `" A"`, `" B"`, …, `" Z"`.
`oracle_decision_function.py` embeds 22 deterministic requests producing 58
rows: 34 choice, 12 bool (`["yes", "no"]`), and 12 score. Two Spanish rows are
marked `evidence_only`; two long rows are marked `zoo_only`. The source's
unchanged `jev_style_mlx.py` renders a plain prompt ending at `\n\nAnswer:` and
tokenizes with `add_special_tokens=False`. No BOS or chat template is added.
The final RMSNorm already contains calibration, so the restricted softmax
temperature stays 1. Bool reads p(yes); score reads the expected level index.

The fixture records two references on every row. `p_author` comes from the
unchanged author code with published MLX BF16 weights; `p_oracle` comes from
the inverse-MLX checkpoint loaded in fp32 by Transformers on the CPU. The
converted checkpoint transposes 18 depthwise-convolution weights and converts
61 direct RMSNorm scales to stored offsets in fp32; gated attention norms
retain their direct scale. The fixture links both converted-file hashes and
the conversion receipt. Model-family metadata continues to cite the original
author repository and revision.

The shared gates add two explicit flags, preserving their existing APUS
defaults: `--label-style space-prefixed --prompt-format decision-function`.
The defaults remain `plain` and `chat`; `--mode` still defaults to `int8hu`
and now also accepts `fp16`. Both forms use the fixture's complete `ids`
verbatim and never apply a template in the gate. The decision-function mode
also checks all prompt and label IDs against the bundle tokenizer. It reads
the original source's text-config attributes directly to avoid the pinned
Transformers 4.57.6 serialization error on the MLX config's absent vision
tower; the source config and exported graph are unchanged.

Decision-function acceptance requires finite values, every option argmax
with fp32-reference margin ≥ 0.02 to match, and exact full-vocabulary reset
logits. Full-vocabulary outputs outside the listed labels remain findings.
The provisional fp16 0.02 / int8hu 0.05 maximum probability-error expectations
are recorded without changing PASS. APUS's existing nonconstant/90% label
gate and `REVIEW_REQUIRED` policy remain its defaults. When `p_author` is
present, both readout and engine transcripts record it independently; it is
optional for other letter fixtures. The engine's required expected text is
always the same bundle's Python full-vocabulary argmax, preserving its leading
space. Both engines run every row for both modes.

From the zoo root, using a local cache and the pinned source snapshot:

```sh
uv run --python 3.11 conversion/letter/oracle_decision_function.py \
  --hf-id chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16 \
  --revision 69a91b6778df7d11b9e13e0f7d4c6a5aa6ad1bc5 \
  --snapshot "$PINNED_DECISION_SNAPSHOT" \
  --converted "$CONVERTED_DECISION" \
  --conversion-report "$DECISION_CONVERSION_REPORT" \
  --out models/qwen3.5-2b-decision/fixtures-qwen3.5-2b-decision.json \
  --work-dir "$DECISION_ORACLE_WORK"

python conversion/letter/readout_gate_letter.py \
  "$DECISION_FP16_BUNDLE" \
  models/qwen3.5-2b-decision/fixtures-qwen3.5-2b-decision.json \
  --mode fp16 --label-style space-prefixed --prompt-format decision-function \
  --snapshot "$PINNED_DECISION_SNAPSHOT" --work-dir "$DECISION_GATE_WORK" \
  --deadline-epoch "$GATE_DEADLINE" \
  --transcript models/qwen3.5-2b-decision/gate-qwen3.5-2b-decision-readout-fp16.json

python conversion/letter/engine_argmax_letter.py \
  "$DECISION_FP16_BUNDLE" \
  models/qwen3.5-2b-decision/fixtures-qwen3.5-2b-decision.json \
  --mode fp16 --label-style space-prefixed --prompt-format decision-function \
  --readout models/qwen3.5-2b-decision/gate-qwen3.5-2b-decision-readout-fp16.json \
  --runner "$ZOO_LLM_RUNNER" --work-dir "$DECISION_GATE_WORK" \
  --deadline-epoch "$GATE_DEADLINE" \
  --transcript models/qwen3.5-2b-decision/gate-qwen3.5-2b-decision-engine-fp16.json
```

Repeat the two gate commands for `--mode int8hu`, the int8hu bundle, and
`*-int8hu.json` transcript names. Use the same `--work-dir` for both modes and
engine checks so the two-identical-runtime-errors stop ledger is shared.
`--aot-asset /path/to/name.h16c.aimodelc` can reuse an existing GPU asset; this
round used the original accepted assets. Every evaluation still gets fresh
states, and the ≤15-evaluation/14,000-call caps include reset repetitions.

The decision oracle's uv header pins Python 3.11, mlx-lm 0.31.3, mlx 0.32.2,
Torch 2.9.0, Transformers 5.17.0, safetensors 0.8.0, huggingface_hub 1.32.0,
numpy 2.3.5 and tokenizers 0.23.2. `--author-python` and `--oracle-python` can
select separate installed environments; `--prepare-only` verifies embedded
inputs without running model inference. Gate environments retain the overlay
pins listed above.

## CC BY-NC 4.0 OpenJev: chat-template bare letters and calibrated noul

The weights of `openjev/openjev` at revision
`5ec9e5fd2f80a6fff386779b1e5ac7e389971889` are CC BY-NC 4.0. The unchanged
`helper/` serving code is Apache-2.0, as is the named base
`Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
This is the 27B OpenJev source model; it is distinct from the APUS model above.
The export is Mac-only and uses the author's unquantized bf16 helper readout as
its reference. There is no fp16 reference bundle and no fp32 second oracle.

`oracle_openjev.py` embeds the accepted 61 requests (61 rows, 51 distinct
composed prompts). It imports the source's `helper/shim.py` unchanged only
after checking SHA-256
`81a22f1b1b8912a465059207ef9f60b7c6c16b4de6372305d867efbe38a1987a`.
Every answer is produced by the helper's `answer_choice`, `answer_score`, or
`answer_noul`, with `shim.client.chat.completions.create` replaced by the
source's `shim_mlx.py` mechanism: `mlx_lm.load()` loads the pinned HF snapshot
as bf16 and computes the full-vocabulary log-softmax at the last prompt
position. The stand-in records the requested letter log-probabilities before
temperature adjustment. The dtype assertion removes the `mlx.core.` prefix
from `str(dtype)`; both names refer to the same MLX bfloat16 dtype.

The SERVE settings are `READOUT_T=0.85`, `READOUT_NOUL_T=1.829074`,
`READOUT_NOUL_BIAS=0`, `READOUT_TARGETED=1`, `READOUT_INSTR_STYLE=pyrepr`, and
`READOUT_PERMS=1`. Optional padding, compaction, layout and loop-breaking
transforms are off. Labels are the **bare** single tokens A–Z then a–z,
without a leading space; all 52 IDs must be unique. The helper renders one
user message as follows, with null descriptions rendered as an empty string:

```text
State:
{state}

Question: {instructions}
Options:
[A] {key_0}: {description_0}
[B] {key_1}: {description_1}
...

Answer with the letter of the best option only.
```

The snapshot's own chat template is applied with
`add_generation_prompt=True, enable_thinking=False`. The fixture stores the
complete resulting `ids`; the runtime always executes these IDs verbatim as
S=1 steps, and a separate tokenization check verifies that the bundle's
retained template and tokenizer reproduce them. `with_image` leaves string
states unchanged and serializes dict states with
`json.dumps(..., ensure_ascii=False)`, retaining the `text` key of a
`{"text": ...}` dict rather than unwrapping it.

Choice is softmax of the selected raw letter log-probabilities divided by
0.85. Score appends ` Rate along the ordered levels below (lowest first).`
to the instructions, labels options with their integer indices, and returns
Σ i·p(i). Noul uses yes/no options with the supplied true/false descriptions,
or `The statement is true.` / `The statement is false.` by default. After
`p_yes = p[0]`, it computes
`sigmoid(logit(clamp(p_yes, 1e-4, 1-1e-4)) / 1.829074 + 0)`.
The source API rounds probabilities and typed outputs to four decimals.
The fixture retains those API answers verbatim and records the helper's
unrounded restricted distribution separately; the readout gate compares
unrounded values, including the calibrated noul value and its absolute error.

The shared gates retain APUS's defaults (`plain`, `chat`, T=1, 32 layers,
15 evaluations including reset, 90% label-argmax floor). They also retain the
2B decision-function flags, its 24-layer config handling, fp16/int8hu modes,
space-prefixed labels, independent author-reference columns and acceptance
policy. OpenJev adds explicit bare labels and temperature, while `--template`
is an alias for the existing `--prompt-format`. This model's commands use
64 layers and at most six prompt evaluations per process, including reset;
long rows run alone. The OpenJev full-vocabulary label fraction is recorded
without a required floor (`--min-label-fraction 0`). Every oracle margin
≥0.02 must match, all logits must be finite and nonconstant, and resets must
be bit-identical. The max |Δp| ≤0.05 expectation is recorded without tuning;
exceedance uses the existing chat `REVIEW_REQUIRED` policy. Near ties, the five
worst rows, calibrated noul error, and peak process RSS are retained.

From the zoo root, after restoring the separately verified pinned source
weights if fresh oracle generation is needed:

```sh
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 COREAI_CHUNK_THRESHOLD=1
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
uv run --python 3.11 conversion/letter/oracle_openjev.py \
  --snapshot "$PINNED_OPENJEV_SNAPSHOT" \
  --out models/openjev-27b/fixtures-openjev-27b.json \
  --work-dir "$OPENJEV_ORACLE_WORK"
python conversion/letter/readout_gate_letter.py \
  "$OPENJEV_BUNDLE" models/openjev-27b/fixtures-openjev-27b.json \
  --mode int8hu --label-style bare --prompt-format chat --temperature 0.85 \
  --snapshot "$PINNED_OPENJEV_SNAPSHOT" --num-layers 64 \
  --max-prompt-evaluations 6 --min-label-fraction 0 \
  --aot-asset "$OPENJEV_AOT_ASSET" --work-dir "$OPENJEV_GATE_WORK" \
  --deadline-epoch "$GATE_DEADLINE" \
  --transcript models/openjev-27b/gate-openjev-27b-readout-int8hu.json
python conversion/letter/engine_argmax_letter.py \
  "$OPENJEV_BUNDLE" models/openjev-27b/fixtures-openjev-27b.json \
  --mode int8hu --label-style bare --prompt-format chat --temperature 0.85 \
  --readout models/openjev-27b/gate-openjev-27b-readout-int8hu.json \
  --runner "$ZOO_LLM_RUNNER" --work-dir "$OPENJEV_GATE_WORK" \
  --deadline-epoch "$GATE_DEADLINE" \
  --transcript models/openjev-27b/gate-openjev-27b-engine-int8hu.json
```

`--prepare-only` on the oracle checks the unchanged helper, all 61 embedded
requests, IDs, option boundaries and fixture floors without loading weights.
Write that output to a separate filename: its uniform dummy values are marked
`PREPARED` and are not oracle evidence. The accepted fixtures came from the
real bf16 inference in round 1. Round 2 validated the portable prepare-only
inputs against every accepted input field and reran the readout and engine
gates from this zoo layout. The weights had already been removed after round
1, so this shipping round does not claim a re-export. The exporter diff and
metadata contract are the reproduction evidence for the single successful
export. The throughput measurement is a separate final quiet-window step.

The oracle uv header pins the working Python 3.11 environment: mlx-lm 0.31.3,
mlx 0.32.2, Transformers 5.17.0, openai 3.16.2, httpx 0.28.1, numpy 2.3.5,
tokenizers 0.23.2, huggingface_hub 1.32.0 and safetensors 0.8.0. The readout
and engine environments keep the overlay pins and frozen Python fork above.
