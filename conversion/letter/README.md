# letter — gates for an OpenJev letter-readout model

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
