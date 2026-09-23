License: **cc-by-nc-4.0** (Creative Commons Attribution-NonCommercial 4.0 International).

Core AI is Apple's on-device ML runtime in iOS 27 / macOS 27 and the successor to Core ML. Apple's `coreai-torch` exports PyTorch graphs to `.aimodel` bundles; this port is measured on the Mac GPU.

# System One scorer — Qwen3.5-4B — Core AI

[🤗 mlboydaisuke/system-one-qwen3.5-4b-scorer-CoreAI](https://huggingface.co/mlboydaisuke/system-one-qwen3.5-4b-scorer-CoreAI) · source [pngwn/system-one-qwen3.5-4b-scorer](https://huggingface.co/pngwn/system-one-qwen3.5-4b-scorer/tree/e6464dce15f013c2ef641593a85cc6afcdaea928) · revision `e6464dce15f013c2ef641593a85cc6afcdaea928` · base [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base/tree/1001bb4d826a52d1f399e183466143f4da7b741b) at `1001bb4d826a52d1f399e183466143f4da7b741b` (Apache-2.0).

For routing, classification or an ordered score, supply a state, a question and the allowed options. **pngwn/system-one-qwen3.5-4b-scorer** scores each `(state, question, option)` sequence with a scalar sequence-classification head, then returns the softmax of those scalars at **T = 1.75**. It does not generate an answer. This conversion merges its rank-16 LoRA into the base text tower and replaces the vocabulary LM head with `nn.Linear(2560, 1, bias=False)` loaded from `score.weight`.

The [author's pinned card](https://huggingface.co/pngwn/system-one-qwen3.5-4b-scorer/blob/e6464dce15f013c2ef641593a85cc6afcdaea928/README.md) reports **576 held-out test questions, accuracy 0.707 and ECE 0.044**, using T = 1.75 fitted on validation data; the raw head's test ECE is **0.135**. The author reports **112.3 ms per four-option question** with options batched together on the author's hardware (the training run is identified as `a100-large`). These are the author's measurements, not task accuracy, calibration or latency re-measured by this port.

## Merge and its proof

The adapter contains **200 LoRA A/B pairs** (400 tensors) and one scalar-head tensor. The ten target projection families are `o_proj`, `in_proj_qkv`, `up_proj`, `k_proj`, `gate_proj`, `in_proj_z`, `out_proj`, `down_proj`, `v_proj` and `q_proj`. `in_proj_a` and `in_proj_b` receive no LoRA update. Merge uses the author's `load_model(base, lora=False)` in CPU FP32, then `PeftModel.from_pretrained`; a separate copy is merged while the unmerged adapter object is retained for the canonical oracle. On three sequences, the maximum scalar difference is **3.8743019104e-06**, below 1e-3.

The [merge tool](../../conversion/merge_system_one_scorer.py) stores one safetensors file with `model.*` text-tower keys in BF16 (the base's storage dtype) and the scalar head in FP32, plus a flat `qwen3_5_text` config with `tie_word_embeddings: false`. The FP32 parity check precedes BF16 storage rounding; the bundle readout gate below covers the stored/converted path. [Exporter](../../conversion/export_system_one_scorer_decode_pipelined.py) downloads both pinned snapshots, performs the merge locally and exports the Qwen3.5 S=1 graph. The zoo oracle can independently compute the FP32 merged model's scores using the author's unchanged functions.

## Decision contract

For **each option separately**, encode this form using `add_special_tokens=False`, with no chat template or added BOS:

```text
State:
{state}

Question:
{question}

Option:
{option}
```

The author first tokenizes `"\n\nQuestion:\n" + question + "\n\nOption:\n" + option` as the tail. If the tail has **384 or more tokens**, keep its last 384 tokens and discard the state. Otherwise tokenize `"State:\n" + state`, keep its first `384 - len(tail_ids)` tokens and append the entire tail. Thus the state is cut from **its end**. The two pieces are encoded separately, as in the author's `encode`, rather than by tokenizing an already concatenated string. The source tokenizer already has pad = EOS = **248044**; padded CPU batches use right padding and last-non-pad pooling.

Start from fresh zero states for each option row. Process its `ids` in S=1 steps; at `slot = len(ids) - 1`, read the sole scalar `logits[0,0,0]` from the output shaped `[1,1,1]`. For a question with k options, compute `softmax([scalar_i / 1.75 for i in range(k)])` in caller option order. Choice uses the declared options; `noul` uses `["yes", "no"]` with yes first; Score uses the declared ordered levels. The training option cap of 16 is not a readout limit.

The [canonical fixture](fixtures-system-one-scorer-4b.json), schema **`coreai-scalar-fixtures/1`**, contains **20 requests / 48 questions / 280 option rows**. The 44 ordinary questions include 24 Choice, 10 noul and 10 Score, with six 9–16-option Choice questions, 16 descriptive Choice questions and two ten-level scores. Four additional `zoo_only` questions have 20, 24, 28 and 32 options. Six long states exercise truncation in twelve rows. Each compiled row is 24–384 tokens. The same fixture is available to the kit for its separate measurement round.

## Measured (Apple M4 Max GPU, macOS 27.0 26A428, 2026-09-23)

| Check on 48 questions / 280 option rows | fp16 reference | int8lin ship |
| --- | ---: | ---: |
| Question argmax = author's FP32 oracle | 48/48 | 48/48 |
| Agreement at oracle margin ≥ 0.02 | 46/46 | 46/46 |
| Max \|Δp\| | 0.003684332 | 0.010874180 |
| Mean of question mean \|Δp\| | 0.000208544 | 0.000833418 |
| Max / mean absolute scalar error | 0.064673901 / 0.014342549 | 0.097398758 / 0.021618439 |
| Finite scalars and probabilities | yes | yes |
| Fresh-state reset scalar, bit-identical | yes | yes |
| Release engine load calls completed | 4/4 | 4/4 |

These accepted measurements compare against the author's unchanged `system_one.py` `encode`, `score_options` and `softmax`, with the PEFT adapter model on CPU FP32 using eight threads. They measure conversion fidelity on synthetic English fixtures. The two near ties, `request-07-q03` (margin 0.006738307) and `request-08-q03` (0.004175941), also retain their argmax. The oracle scalars range from −14.593598366 to 7.658686638. The probability mean weights each question equally, after averaging over its options; scalar error averages the 280 option rows.

The table retains the accepted comparison against the **unmerged PEFT FP32 oracle**; the linked zoo-layout readout transcripts instead use the **freshly recomputed merged FP32 oracle**. Their max |Δp| values are **0.00368434458632** (fp16) and **0.0108744634178** (int8lin), differing from the table's accepted maxima by **1.23851191525e-08** and **2.83582027749e-07**, respectively (required ≤1e-6); both retain **48/48 question argmaxes**. The recomputed merged oracle differs from the canonical PEFT oracle by at most **3.3438205719e-05** scalar units and **2.63638415043e-06** probability; the canonical fixture remains byte-identical. [fp16 reference comparison](gate-system-one-scorer-4b-fp16.json) · [int8lin reference comparison](gate-system-one-scorer-4b.json).


Readout uses AOT **h16c**, `SpecializationOptions.default()`, fresh zero KV/conv/recurrent states for each row and every token in S=1. Workers execute at most 15 rows including reset checks; the first row is repeated within a process and in a fresh final process. [fp16 readout](gate-system-one-scorer-4b-readout-fp16.json) · [int8lin readout](gate-system-one-scorer-4b-readout-int8lin.json).

The [fp16 engine check](gate-system-one-scorer-4b-engine-fp16.json) and [int8lin engine check](gate-system-one-scorer-4b-engine-int8lin.json) load two rows per bundle with both Release `coreai-pipelined` and `coreai-sequential`, raw token ids, one output token, temperature 0 and warmup off. **The output width is 1, so the emitted id is always 0 and its decoded text is meaningless.** Completion and fresh loads are the check; this does not check option probabilities through the Swift engine or generated-text quality.

### S=1 throughput proxy

Release `llm-benchmark`, **int8lin only**, `-p 128 -g 256 -n 3`, two alternating launches per engine / six trials each, `COREAI_CHUNK_THRESHOLD=1`, Xcode 27.0 (27A266a). The Swift source copy is fork tag `0.2.4-zoo` (`f7a75ec`) with the three-line engine-variant option patch. The recorded inference window has **contended: false**; before/after process snapshots contain no other engine workload, and the previously untagged GPU lock was held only during this measurement.

| Engine | S=1 prefill tok/s, median (min–max) | Decode tok/s, median (min–max) | Load seconds, two launches |
| --- | ---: | ---: | ---: |
| coreai-pipelined | 89.347 (88.546–90.107) | 88.894 (87.169–89.510) | 0.740 / 0.659 |
| coreai-sequential | 76.039 (75.701–76.273) | 75.297 (74.705–75.531) | 0.725 / 0.671 |

This is an **S=1 prefill-rate proxy** over synthetic 128-token inputs, not per-question latency: every option requires processing its own entire row. With the one-wide output metadata the benchmark samples id 0 for its synthetic inputs and emits id 0 for its 256 decode steps. The generated tokens are meaningless as text; decode is recorded only as a rate. Load time is measured once per launch and excludes warmup. [All trials, process snapshots and interpretation](llm-benchmark.json).

## Through the kit

**Measured through coreai-kit** by the supervisor on **2026-09-23 09:05–09:25 JST**, in worktree `~/code/coreai-kit-models-wt` on branch `decision-models-4`, using **`Decision.Format.scalar`**, the sequential engine and the kit's own rendering of the author's rows. **All times in this section are contended:** the GPU was shared with two conversion runs. These supplied measurements were not re-derived by this conversion run.

`decide-cli parity` reads the canonical `coreai-scalar-fixtures/1` fixture. Both bundles matched **280/280 option-row token sequences and last-token slots**, and **48/48 question argmaxes** against its FP32 oracle. Question time sums all of that question's option rows.

| Bundle | Questions / option rows | Max / mean \|Δp\| | Median time per question, contended |
| --- | ---: | ---: | ---: |
| int8lin | 48 / 280 | 0.0110 / 0.0012 | 1,016 ms |
| fp16 | 48 / 280 | 0.0037 / 0.0004 | 1,556 ms |

The int8lin **32-option question took 11.2 s**. The kit resolves the scalar format from bundle metadata (`decision.head == "scalar"`, `temperature`, `max_len`), applies the author's row rendering and truncation, and softmaxes the rows' scalars at **1.75**. It lists up to **64 options**. Its catalog entry is `system-one-scorer-4b` (macOS only, `license: CC-BY-NC-4.0`, pinned to this repository's revision).

SemIf's authored144 has **144 English rows with three options each**, SemIf's gold labels and its unchanged `benchmarks/evaluate.py`. Rendering every option as one scalar row on int8lin gave **121/144 raw correct** and **0.844 mean family balanced accuracy**: evidence_interpretation **0.918**, rule_application **0.759**, candidate_selection **0.855**. Median decision time was **2,311 ms**, contended, for three rows totaling about **175 tokens**. For scale on the same rows and evaluator, the kit README reports Qwen3.5-4B int8 zero-shot **0.821**, MiniCPM5-2B int8 **0.681**, OpenThai-SystemOne **0.725**, APUS-OpenJev-v1-4B **0.906** and Qwen3.5-2B-Decision **0.798**. These are numbers on that evaluator, not a ranking.

For the three-question support ticket, the supplied state was:

> Customer message: I was charged twice for my order last week and nobody has replied. I want this fixed today.

| Question and declared options | Reported answer | Option rows / total tokens | Question time, contended |
| --- | --- | ---: | ---: |
| Which team: billing / shipping / technical | billing **0.950**, shipping **0.010**, technical **0.040** | 3 / 123 | 1,862 ms |
| Is the customer angry: yes / no | P(yes) **0.723** | 2 / 80 | 1,094 ms |
| How urgent: can wait / this week / today / right now | expected level **2.16**; today **0.586**, right now **0.301** | 4 / 163 | 2,198 ms |

The request reused **0 tokens**: every option row is prefilled from its first token on this decode-only graph. A choice costs one row per option. [Supervisor-supplied measurements](measurements-coreai-kit.json).

## Bundle

Both LanguageBundles are staged under `gpu-pipelined/` in [mlboydaisuke/system-one-qwen3.5-4b-scorer-CoreAI](https://huggingface.co/mlboydaisuke/system-one-qwen3.5-4b-scorer-CoreAI). `int8lin` ships; `fp16` is the reference.

| Bundle | Staged bytes | main.mlirb bytes | main.mlirb SHA256 |
| --- | ---: | ---: | --- |
| `gpu-pipelined/system_one_qwen3_5_4b_scorer_decode_int8lin` | 5,095,388,697 | 5,066,794,425 | `1dbd8310571e2a306fd0b00fae2a3fe252905b8267e183f865cea911bf3933f5` |
| `gpu-pipelined/system_one_qwen3_5_4b_scorer_decode_fp16` | 8,441,294,422 | 8,412,700,156 | `f07923c007f760dd8fcd11d8c755d097ecdeb4adc2c51bf1c306648eca4eff89` |

The full 32-layer text graph has 24 linear-attention layers and eight full-attention layers, with loop-free single-step enabled on the 24 linear-attention layers. `int8lin` uses block-32 int8 linear weights with the recorded symmetric-with-clipping quantizer; the scalar head, **248,320-row input embedding**, conv1d and norms stay FP16. The source tokenizer has **248,077 entries**, preserved as supplied. Root metadata's `language.vocab_size` is **1**, the output width, and does not resize the input embedding. `max_context_length` is 4096; the decision contract still limits every row to 384 tokens. Root metadata includes the pinned adapter revision and `decision` head/temperature/layout/base/license block; `compression: null` is preserved from the helper, while the recipe names the quantization.

The staged graphs, metadata and tokenizer files are byte-identical copies of the accepted exports. The separate int8lin re-export records metadata equality excluding only `compilation.date`, and records IR size/SHA256; serialization byte identity is not required. The independent re-export has **5,066,794,486 bytes**, SHA256 `e66b8678e24e6f9d3ccbf9e257aafc6bdd766c7b91b59fabbf33b461b981cf87`; its metadata equals the shipped metadata after excluding only `compilation.date`, and its merged tensor payload exactly equals the accepted merge. Use the [extra-states runtime patch](../../apps/coreai-pipelined-extra-states.patch) for KV/conv/recurrent state and set `COREAI_CHUNK_THRESHOLD=1` before engine creation. Mac GPU is the tested device; no phone or Neural Engine measurement is claimed.

## Reproduce

Export uses coreai-core 1.0.0b2, coreai-torch 0.4.1, coreai-opt 0.2.1, torch 2.9.0, transformers 4.57.6 and frozen fork `coreai_models` at `397b337`. Merge and oracle use their own environment: transformers 5.17.0, peft 0.21.0, torch 2.9.0, safetensors 0.8.0 and huggingface_hub 1.32.0. All model/cache/output paths remain local to the conversion workspace; downloading is limited to two workers.

From the zoo checkout, using the prepared adjacent environments:

```sh
export HF_HOME="$(pwd)/../cache/huggingface"
export HF_HUB_DISABLE_XET=1
export COREAI_CHUNK_THRESHOLD=1
../.venv/bin/python conversion/export_system_one_scorer_decode_pipelined.py int8lin \
  --merge-python "$(pwd)/../.venv-oracle/bin/python" --out-dir "$(pwd)/../reexport"
../.venv/bin/python conversion/export_system_one_scorer_decode_pipelined.py fp16 \
  --merge-python "$(pwd)/../.venv-oracle/bin/python" --out-dir "$(pwd)/../reexport"
```

The [recipe](recipe.toml) fixes both bundle names and the pinned source. The standalone merge CLI takes `--base-snapshot`, `--adapter-snapshot` and `--out`. [Scalar gate instructions](../../conversion/scalar/README.md) specify the embedded fixture builder, unchanged author functions, AOT/default-specialization readout and two-engine load check. [Port notes](../../knowledge/system-one-scorer-4b-port.md) explain the head replacement and runtime limits.

## License

This converted scorer is **cc-by-nc-4.0**. The base **Qwen/Qwen3.5-4B-Base** at `1001bb4d826a52d1f399e183466143f4da7b741b` is **Apache-2.0**; that base license does not replace the scorer's stated non-commercial license. The pinned adapter snapshot carries **no LICENSE file**. Its [card](https://huggingface.co/pngwn/system-one-qwen3.5-4b-scorer/blob/e6464dce15f013c2ef641593a85cc6afcdaea928/README.md) declares at line 2 and lines 94–95:

> license: cc-by-nc-4.0
> **The ticket component of the training data is CC-BY-NC-4.0**, so this model inherits a
> non-commercial restriction.

The staged `LICENSE` is the canonical [Creative Commons Attribution-NonCommercial 4.0 International legal text](https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt), **19,347 bytes**, SHA256 `41003d4a74749c0220e33dd415042164b5a1093ed401f36277234f772d22d3d0`. `adapter_config.json` and `base-config.json` are verbatim pinned-source files. This port changes the model by merging the LoRA, retaining its scalar head, exporting the text graph and quantizing the int8lin linears. The owner has selected shipment with **cc-by-nc-4.0** stated; publication is a separate owner action.
