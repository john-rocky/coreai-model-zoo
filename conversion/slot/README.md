# slot — gates for a System One slot-head model

These gates cover `iapp/OpenThai-SystemOne` at revision
`f3709948b5e3cc9606a57e74ba62b7a639d17dd3`. The checkpoint reads 256 slot logits at
`<|ts_answer|>`; it does not generate an answer string. Its embedding has 248,339
rows, while the bundle's output width and `language.vocab_size` are 256.

1. `oracle_slot.py` downloads the pinned checkpoint and copies its inference files
   unchanged, with the author's `client.py` pinned to `5d04bcc`. The embedded 18
   requests produce 48 kit rows and two zoo rows with 40 and 255 options. Each row
   contains one question in request order, with `permutations=1`. The CPU fp32
   oracle checks the answer position, full versus truncated sequence causality,
   and exact agreement with the author's single-question API. It also records the
   one-pass API's answers to all questions of each request; that comparison is not
   a pass criterion.
2. `readout_gate_slot.py` uses the overlay interpreter, AOT-compiles h16c for GPU
   unless the selected AOT asset exists, and loads it with
   `SpecializationOptions.default()`. Every row starts with four zero states and
   uses S=1 input IDs and full-length position IDs. It reads the final 256 logits,
   divides by the row's checkpoint temperature, masks slots `k..254`, applies
   softmax, then renormalises the first `k` probabilities. Slot 255 is recorded as
   abstain. Each runtime process handles at most 30 rows and 18,000 calls; each
   wide row has a separate process. Every process repeats its first row, and a
   final process repeats the fixture's first row. PASS requires finite logits and
   probabilities, matching option argmax on every row whose oracle margin is at
   least 0.02, and identical reset logits. Probability and abstain errors are
   recorded without additional thresholds.
3. `engine_argmax_slot.py` checks a Release `llm-runner` against the same bundle's
   Python readout. Greedy samples all 256 raw slots, so the expected text is
   `tokenizer.decode([raw256_argmax])`, before temperature or masking. PASS
   requires every call to succeed and every exact emitted string to match. Slot
   IDs are byte vocabulary tokens when interpreted by the text runner; multiple
   bytes can decode to the same Unicode replacement character. This check proves
   emitted text agreement, not recovery of the emitted token ID. Mismatches are
   recorded once, without retries.

The oracle's uv header pins Python dependencies that resolved and executed:
Torch 2.9.0, Transformers 5.17.0, safetensors 0.8.0, huggingface_hub 1.32.0,
numpy 2.3.5, tokenizers 0.23.2, and pydantic 2.13.5. It uses the author's
non-CUDA reference kernels; do not install flash-linear-attention or causal-conv1d.
The readout uses the zoo overlay environment (coreai-core 1.0.0b2,
coreai-torch 0.4.1, coreai-opt 0.2.1, Torch 2.9.0, Transformers 4.57.6).

From the zoo root, with the fork's Release runner and overlay interpreter set:

```sh
export HF_HUB_DISABLE_XET=1
export COREAI_CHUNK_THRESHOLD=1
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
uv run conversion/slot/oracle_slot.py \
  --hf-id iapp/OpenThai-SystemOne \
  --revision f3709948b5e3cc9606a57e74ba62b7a639d17dd3 \
  --out models/openthai-systemone/fixtures-openthai-systemone.json
python conversion/slot/readout_gate_slot.py \
  exports/openthai_systemone_decode_int8lin \
  models/openthai-systemone/fixtures-openthai-systemone.json \
  --transcript models/openthai-systemone/gate-openthai-systemone-readout-int8lin.json
python conversion/slot/engine_argmax_slot.py \
  exports/openthai_systemone_decode_int8lin \
  models/openthai-systemone/fixtures-openthai-systemone.json \
  --readout models/openthai-systemone/gate-openthai-systemone-readout-int8lin.json \
  --runner "$ZOO_LLM_RUNNER" --engine pipelined --engine sequential \
  --transcript models/openthai-systemone/gate-openthai-systemone-engine-int8lin.json
```

Repeat the last two commands with the fp16 bundle and fp16 transcript names. A
single `--engine pipelined` or `--engine sequential` checks only that engine;
repeating the option combines both in one transcript. `--aot-dir`, `--work-dir`,
and `--deadline-epoch` allow an isolated run to select existing compiled assets,
place evidence, and bound execution. The oracle has `--deadline-unix`.

The fixture schema is `coreai-slot-fixtures/1`. Gate transcripts retain the
predecessor's `coreai-decider-readout-gate/1` and
`coreai-decider-engine-gate/1` envelopes (`rows`, `summary`, `result`,
`generated_at`) with `head: "slot"`. Readout rows use `option_argmax` instead of
`letter_argmax`, add `raw256_argmax`, `p_full`, and abstain errors, and retain
`full_vocab_argmax_id` as an alias for the raw256 argmax. Off-option raw argmax
is valid for a slot head and is not a failure. Engine summaries additionally
separate the selected engines. Evidence is written incrementally, with session
JSON and stdout logs beside the transcript unless `--work-dir` is specified.

AOT is required because the Python GPU JIT of this graph returns zero logits on
macOS 27.0 (26A428). The runtime's per-call IOSurface leak also requires process
splitting. Scratch usage is captured before and after every runtime process;
these scripts never delete runtime scratch. See
`../../knowledge/openthai-systemone-port.md` for the source and runtime findings.
