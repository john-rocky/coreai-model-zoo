Core AI is Apple's on-device ML runtime in iOS 27 / macOS 27 and the successor to Core ML: PyTorch models are exported with Apple's `coreai-torch` (LLMs: `coreai.llm.export`) into `.aimodel` bundles that run on the GPU or the Neural Engine, e.g. Qwen3-8B 4-bit decodes at 94 tok/s on an M4 Max GPU, MLX 90 under the same protocol ([apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench), macOS 27 beta 26A5353q, 2026-06-11).

# APUS-OpenJev-v1-4B — Core AI

[🤗 mlboydaisuke/APUS-OpenJev-v1-4B-CoreAI](https://huggingface.co/mlboydaisuke/APUS-OpenJev-v1-4B-CoreAI) · Apache-2.0 · source [apus-ailab/APUS-OpenJev-v1-4B](https://huggingface.co/apus-ailab/APUS-OpenJev-v1-4B/tree/65797c526c27c4d24f564333779162cd4a64328e), revision `65797c526c27c4d24f564333779162cd4a64328e` · base Qwen/Qwen3.5-4B

A decision model for browser-action selection, workflow routing and proposition judgments.
The source contains checkpoint-5949 merged BF16 weights from a Qwen3.5-4B fine-tune. Each
request supplies a state, instructions and criteria; the answer is a distribution over the
criteria from the next-token logits of letters A–P. The model retains its vocabulary LM head.
This bundle exports the 32-layer text path with the zoo's unchanged Qwen3.5-4B exporter,
`int8hu --head-sym`, and a 4,096-token context.

The author's [Frozen80 development-panel result](https://huggingface.co/apus-ailab/APUS-OpenJev-v1-4B/blob/65797c526c27c4d24f564333779162cd4a64328e/README.md)
is **66/80 (82.50%)** at full depth; its [merged evaluation](https://huggingface.co/apus-ailab/APUS-OpenJev-v1-4B/blob/65797c526c27c4d24f564333779162cd4a64328e/merged-evaluation.json)
records **61/80** at 16 layers. These are the author's measurements, not re-measured here.
The panel covers Browser, HelpSteer3, BoolQ, MNLI and attribute decisions; it is a reused
engineering panel, not a blind benchmark or browser success rate. The author reports
160/160 decision agreement before/after BF16 merging, while its original probability-delta
≤0.05 gate failed: max |Δp| was 0.061457 at 32 layers and 0.092269 at 16 layers.

## Decision contract

`jev.dynamic.prompt.v2` accepts `id`, `group_id`, a nonempty string `state`, nonempty
`instructions`, `primitive` (`choice`, `noul` or `score_level`) and 2–16 ordered criteria.
Each criterion has an `id` and a `description`. JSON states must be serialized as text.
For `k` criteria, labels are `"ABCDEFGHIJKLMNOP"[:k]`; on this tokenizer A–P have ids
**32–47**. The author checks each label is one nonspecial token at the compiled answer boundary.

The user turn is `"Shared state:\n" + state + "\n\n"`, followed by
`json.dumps({"criteria": [{"description": ..., "label": "A"}, ...], "instructions": ...,
"primitive": ...}, ensure_ascii=False, sort_keys=True)`, followed by
`"\nReturn only the selected letter: " + ", ".join(labels) + ".\nAnswer:"`.
Compile that as **one user message** with the source chat template,
`add_generation_prompt=True`, `enable_thinking=False`; encode the rendered text with
`add_special_tokens=False`. The answer slot is `len(ids)-1`, the last compiled token.
The author rejects more than 8,192 tokens and multimodal placeholders; this exported bundle
requires at most 4,096 tokens.

The following is the verbatim decoded compiled prompt for fixture `choice16_01` (484 tokens),
including its user turn and the closed thinking prefix. The two newlines after `</think>`
are part of the prompt; the fixture also stores the exact `tail_after_state` string.

```text
<|im_start|>user
Shared state:
{"goal": "find the lunar archive schedule", "history": "The page has just opened; no search has been submitted.", "loading": false, "page": "document search", "search_field": "empty", "visible_controls": ["search field", "Search button"]}

{"criteria": [{"description": "Return to the previous page without changing any data.", "label": "A"}, {"description": "Open the next page of the paginated results.", "label": "B"}, {"description": "Dismiss the visible informational banner with its close button.", "label": "C"}, {"description": "Save the completed form as a draft without submitting it.", "label": "D"}, {"description": "Submit the completed form once all required fields are valid.", "label": "E"}, {"description": "Fill the required field that is currently marked missing.", "label": "F"}, {"description": "Select the Details tab to view the requested additional information.", "label": "G"}, {"description": "Change the result ordering to most recent first.", "label": "H"}, {"description": "Expand the collapsed section containing the requested information.", "label": "I"}, {"description": "Download the plain text copy using the visible download link.", "label": "J"}, {"description": "Copy the displayed reference code using the adjacent copy button.", "label": "K"}, {"description": "Wait for the current loading indicator to finish before taking another action.", "label": "L"}, {"description": "Stop because the requested task is already complete and confirmed.", "label": "M"}, {"description": "Open the first relevant search result in the current tab.", "label": "N"}, {"description": "Enter the requested query in the empty search field.", "label": "O"}, {"description": "Submit the completed search field using the Search button.", "label": "P"}], "instructions": "Choose the next browser action that advances the stated goal. The search field must contain the goal query before a search can be submitted.", "primitive": "choice"}
Return only the selected letter: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P.
Answer:<|im_end|>
<|im_start|>assistant
<think>

</think>

```

For the full-depth path, take `logits[0, -1, label_ids]` as float32 and compute
`p = softmax(label_logits)` with **T = 1**. Map probabilities back to the criteria's ids in
request order. `choice` returns the argmax id; `noul` and `score_level` return
`yes_probability`. The author's response says **`calibrated: false`**: these are relative
candidate preferences, with no temperature calibration.

Both `noul` and `score_level` require these exact criteria; the proposition goes in
`instructions`:

```json
[{"id": "yes", "description": "The stated proposition is true."}, {"id": "no", "description": "The stated proposition is false."}]
```

`score_level` judges one proposition; it does not assemble an aggregate score. The author's
`effort="low"` executes 16 layers with an early-exit wrapper and a projection over candidate
rows. That path needs its own graph and is not exported. Low-effort logits and probabilities
are retained only as fixture evidence; the bundle and gate use `effort="high"` (32 layers).

The [fixture](fixtures-apus-openjev-v1-4b.json) has **48 requests / 48 rows**: 28 choice,
12 noul and 8 score_level; eight choice rows have 16 criteria, eight have 2–3 and twelve have
4–8. Sixteen requests contain Chinese and thirty have structured states serialized as text.
States are **34–66 tokens**, compiled rows **124–1,672 tokens**; two rows above 1,024 tokens
are marked `zoo_only`. Every author's high-effort API probability equals the recorded fp32
row probability (48/48). Minimum oracle top-2 margin is **0.948419**: this fixture has no
near ties and does not test decisions near a probability boundary.

## Measured (Apple M4 Max GPU, macOS 27.0 26A428, 2026-09-23)

| Check | int8hu --head-sym |
|---|---:|
| option argmax = author's fp32 full-depth oracle | 48/48 |
| argmax on oracle margin ≥ 0.02 | 48/48 |
| max \|Δp\| over option probabilities | 0.005302 |
| mean of per-row mean \|Δp\| | 0.000127 |
| max absolute label-logit difference | 0.236008 |
| full-vocabulary argmax is a row label | 48/48 |
| Swift pipelined first token = Python full-vocabulary argmax text | 48/48 |
| Swift sequential first token = Python full-vocabulary argmax text | 48/48 |
| Swift first token = oracle argmax letter, both engines | 96/96 |
| row-1 reset, logits bit-identical | yes |

These are the accepted round-one measurements. Probability errors are recorded against the
author's `openjet_runtime.OpenJet`, CPU fp32, transformers **5.16.1**, torch **2.9.0**.
The GPU readout uses an AOT **h16c** asset, `SpecializationOptions.default()`, fresh zero
states for each row and S=1 steps over the complete compiled ids. The output is fp16
`[1,1,248320]`; label gathering and softmax run in float32. Processes evaluate at most
15 prompts including resets; the two wide rows run separately. All readout values are finite.

The [readout transcript](gate-apus-openjev-v1-4b-readout.json) and
[engine transcript](gate-apus-openjev-v1-4b-engine.json) are the zoo-layout execution proofs;
the [summary](gate-apus-openjev-v1-4b.json) also records the kit parity facts.
The separate round-two [alphabet gate](gate-apus-openjev-v1-4b-alphabet.json) passed **16/16**
tokens on `The alphabet begins A, B, C, D, E, F,` against a CPU fp32 overlay oracle; both
sides produced ` G, H, I, J, K, L, M, N,`.
That transcript is separate from the letter-probability gate. The unmodified generic gate
reports `weights_pinned: false` because its loader has no revision argument; this invocation
used `HF_HUB_OFFLINE=1`, the pinned local `refs/main`, and three matching Hub LFS hashes.
The effective-pin receipts are described in the [port notes](../../knowledge/apus-openjev-v1-4b-port.md).

Throughput uses Release `llm-benchmark`, p=128 / g=256, two launches × three trials per engine,
`COREAI_CHUNK_THRESHOLD=1`, Xcode 27.0 (27A266a); median (range):

| Engine | S=1 prefill, tok/s | decode, tok/s | load per launch, s |
|---|---:|---:|---:|
| coreai-pipelined | 81.781 (81.166–81.870) | 80.379 (79.323–80.606) | 2.899 / 0.700 |
| coreai-sequential | 69.369 (69.257–69.524) | 68.493 (67.860–68.605) | 0.703 / 0.706 |

A decision costs one S=1 prefill of the **whole row**, about **15 ms per token** on the
sequential engine measured through coreai-kit below, so the decode number is a
**prefill-rate proxy**, not per-decision latency. Synthetic ids are sampled from the metadata's
248,320-token vocabulary. The before/after process snapshots contain no other engine workload
(`contended: false`); load is measured once per launch, excluding warmup. The benchmark source
copy only adds its engine-variant option and forwards `EngineOptions.variant`.
[Trials, load times and environment](llm-benchmark.json).

## Through the kit

**Measured through coreai-kit**, using `Decision.Format.sharedState`, the sequential engine
and the kit's own chat-template rendering: `decide-cli parity` on the 40 `choice` and `noul`
rows matched tokens **40/40** (byte-for-byte with the author's `compile()`), label slots
**40/40** and option argmax **40/40**, with max |Δp| **0.0055** and mean **0.0002**.
The 8 `score_level` rows were skipped: the kit has no primitive of that name. Its `noul` has
the same proposition semantics, but rendering `"primitive": "noul"` changes the author's
`"primitive": "score_level"` prompt. Median time was **2.8 s** per fixture question over
those 40 rows (**124–1,672 compiled tokens**, S=1 prefill); the 1,672-token row took **24 s**.

A two-question workflow request with a **20-token state**, a 3-criteria choice and a noul,
took **2,069 ms** for its 134-token row and **1,540 ms** for its 107-token row. It reused
**0 tokens**: the recurrent hybrid re-prefills every row, about **15 ms per token** on the
sequential engine. Answers were **close 0.999** and **refund asked P(yes) 0.000**.

On SemIf's authored144 — **144 English rows**, three options each, SemIf's gold labels and
its unchanged `benchmarks/evaluate.py` — the kit's rendering of each row as `choice` gave
**131/144** raw and **0.906 mean family balanced accuracy**, median **1.96 s** per decision.
On the same rows and evaluator, the kit README reports Qwen3.5-4B int8 zero-shot **0.821**,
MiniCPM5-2B int8 **0.681**, and the OpenThai run **0.725**. These are measurements on that
fixture and evaluator. [Supervisor-supplied kit record](measurements-coreai-kit.json).

## Bundle

The staged Hub path is
`gpu-pipelined-b2/apus_openjev_v1_4b_decode_int8hu_block32_sym/` under
[mlboydaisuke/APUS-OpenJev-v1-4B-CoreAI](https://huggingface.co/mlboydaisuke/APUS-OpenJev-v1-4B-CoreAI).
The LanguageBundle is **5,770,814,417 bytes**, 11 files: `.aimodel`, `metadata.json` and
`tokenizer/`. `main.mlirb` is **5,742,220,411 bytes**, SHA-256
`7b719b75f6782d60ff0e082230e464f041927ef7bbd3ffb5cfbd2a9352e32f23`.
Root `metadata.json` is 636 bytes, SHA-256
`48b133c07934b04dbe77248f6ef421b013c6582f4888123a6ff3214a37be0316`.

`int8hu --head-sym` uses block-32 int8 linears and an absmax-symmetric block-32 int8
vocabulary head cloned from the tied embedding; embeddings, conv1d and norms stay fp16.
Metadata retains vocabulary **248320**, maximum context **4096** and the exporter's
`compression: null`; the recipe records quantization. The source chat template is unchanged.

Use the [extra-states runtime patch](../../apps/coreai-pipelined-extra-states.patch) for KV,
conv and recurrent states, and set `COREAI_CHUNK_THRESHOLD=1` before engine creation.
Both engines were checked with Release tools from fork tag **0.2.4-zoo** (`f7a75ec`).
The readout uses AOT h16c on macOS; no iPhone result is claimed. The
[recipe](recipe.toml) records the source revision and exporter invocation.

## Reproduce

The [exporter](../../conversion/export_qwen3_5_decode_pipelined.py) and `_bundle.py` are
unchanged zoo `082fe55` copies. The frozen authoring overlay is fork `397b337`; export pins
include coreai-core **1.0.0b2**, coreai-torch **0.4.1**, coreai-opt **0.2.1**, torch **2.9.0**
and transformers **4.57.6**. The author oracle has its own transformers **5.16.1** environment.
Populate `HF_HOME` with all files at the source revision and confirm its cached `refs/main`
resolves to that revision before running the unchanged exporter offline.
`HF_HUB_DISABLE_XET=1`, a run-local cache and Xcode 27 are required for this recorded run.

```bash
HF_HUB_OFFLINE=1 python3 conversion/zoo_convert.py run apus-openjev-v1-4b
# Equivalent exporter invocation:
HF_HUB_OFFLINE=1 python3 conversion/export_qwen3_5_decode_pipelined.py int8hu --head-sym \
    --hf-id apus-ailab/APUS-OpenJev-v1-4B --out-dir exports

SNAPSHOT="$HF_HOME/hub/models--apus-ailab--APUS-OpenJev-v1-4B/snapshots/65797c526c27c4d24f564333779162cd4a64328e"
uv run conversion/letter/oracle_letter.py --snapshot "$SNAPSHOT" \
    --out models/apus-openjev-v1-4b/fixtures-apus-openjev-v1-4b.json

python3 conversion/letter/readout_gate_letter.py \
    exports/apus_openjev_v1_4b_decode_int8hu_block32_sym \
    models/apus-openjev-v1-4b/fixtures-apus-openjev-v1-4b.json \
    --snapshot "$SNAPSHOT" \
    --transcript models/apus-openjev-v1-4b/gate-apus-openjev-v1-4b-readout.json

python3 conversion/letter/engine_argmax_letter.py \
    exports/apus_openjev_v1_4b_decode_int8hu_block32_sym \
    models/apus-openjev-v1-4b/fixtures-apus-openjev-v1-4b.json \
    --readout models/apus-openjev-v1-4b/gate-apus-openjev-v1-4b-readout.json \
    --runner <fork>/.build/release/llm-runner \
    --transcript models/apus-openjev-v1-4b/gate-apus-openjev-v1-4b-engine.json
```

For the recorded round-two layout proof, the oracle command used `--replay-fixtures` to
validate and copy the accepted fixture byte-for-byte, and the readout used `--aot-asset` to
reuse the round-one h16c asset. The commands above describe fresh reproduction.
[Letter-gate instructions](../../conversion/letter/README.md) describe the pinned oracle,
AOT cache and two-engine check. [Port notes](../../knowledge/apus-openjev-v1-4b-port.md)
record runtime traps and fixture coverage.

## License

The source's `LICENSE` is Apache-2.0, **11,544 bytes**, SHA-256
`bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a`.
The staged bundle includes that file verbatim and retains its Qwen notice:
**Copyright 2026 Alibaba Cloud**. The source `config.json` is also retained verbatim.
[Source license](https://huggingface.co/apus-ailab/APUS-OpenJev-v1-4B/blob/65797c526c27c4d24f564333779162cd4a64328e/LICENSE).
The author's `openjet_runtime` is vendored from that pinned source for the oracle gate and
is not part of the LanguageBundle.
