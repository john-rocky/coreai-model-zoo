# OpenThai-SystemOne: slot probabilities on Core AI

> 2026-09-23. `iapp/OpenThai-SystemOne` uses a Qwen3.5-0.8B-Base text tower continued-pretrained
> on Thai and a biased slot head in place of the LM head. The conversion preserves the author's
> single-question probability readout. int8lin ships; fp16 is the reference. Card:
> [`models/openthai-systemone/README.md`](../models/openthai-systemone/README.md).

The conversion run is
`~/code/codex-conversions/2026-09-23/openthai-systemone-coreai/`; `results/`, `logs/` and
`scripts/` paths below are relative to that run. Fixture and gate links point to files staged
for this zoo. Source revision and license are recorded in `results/download.json`; checkpoint
configuration and model structure are in `results/hf-config-premise.json`,
`results/oracle_model.json` and `results/loader_int8lin.json`.

## What the checkpoint is

The source is `iapp/OpenThai-SystemOne` at
`f3709948b5e3cc9606a57e74ba62b7a639d17dd3`, Apache-2.0. Its root type is
`openthai_systemone`; `text_config` describes the Qwen3.5 text tower. The checkpoint has
24 layers: 18 linear-attention and 6 full-attention layers. Its hidden size is 1,024, its
embedding is `[248339, 1024]`, and the head tensors are `slot_head.weight [256, 1024]` plus
`slot_head.bias [256]`. There is no `lm_head` tensor. The graph emits `[1, 1, 256]` logits,
uses decode-only S=1 calls and has a 4,096-token context. Evidence:
`results/hf-config-premise.json`, `results/loader_{fp16,int8lin}.json`,
`results/export_{fp16,int8lin}.json` and `results/export-review.md`.

This is a decision readout: Choice, Score and Noul use slot indices at `<|ts_answer|>`.
The tokenizer's answer id is 248082; the abstain slot is 255. The input vocabulary has
248,339 entries, including 295 added tokens. `language.vocab_size = 256` in bundle metadata
is deliberately the **output width**: the sequential engine sizes its logits buffer from
that field. Tokenizer ids remain in the full input range. Evidence: both export manifests,
their embedded metadata, and
[`fixtures-openthai-systemone.json`](../models/openthai-systemone/fixtures-openthai-systemone.json).

## The three exporter differences

The baseline was zoo `082fe55`'s `conversion/export_qwen3_5_decode_pipelined.py`; the frozen
authoring tree was fork commit `397b337`. Both original and adapted hashes, and the exact
diff, are in `results/env.json` and `results/exporter.diff`. The zoo exporter is
[`conversion/export_openthai_systemone_decode_pipelined.py`](../conversion/export_openthai_systemone_decode_pipelined.py).

1. **Weight prefix.** The checkpoint keys start with `model.*`, not
   `model.language_model.*`. The exporter builds a config from the snapshot's `text_config`
   object, avoiding the root remote `AutoConfig`. Its local safetensors loop retains the
   frozen loader's construction and key mapping, with the changed prefix.
2. **Embedding rows.** Preserve `vocab_size = 248339` and `tie_word_embeddings = false` in
   the text config. The saved tokenizer retains all 295 added tokens. Random trace input
   ids are bounded by the input config, not the metadata's output width.
3. **Biased head.** The missing `lm_head.weight` is expected; the frozen loader's final
   meta-parameter check would otherwise reject the checkpoint. The exporter installs
   `nn.Linear(1024, 256, bias=True)` before that check and copies the slot tensors into it.
   int8lin uses the existing block-32 linear quantizer; its `.*lm_head$` exclusion keeps
   the slot head fp16, as it keeps embeddings, conv1d and norms fp16. The full fp16 mode
   uses the same graph. Temperature and masking are host readout operations.

The graph's layer, state and loop-free decode code is unchanged. Both exports logged
“loop-free single-step enabled on 18 linear layers.” Evidence: `results/export-review.md`,
`results/loader_{fp16,int8lin}.json`, `logs/export-fp16.log` and `logs/export-int8lin.log`.
The separate zoo exporter downloads the pinned source itself and incorporates the run's
loader wrapper. Its int8lin re-export completed in **47.482 s** on CPU, with peak child RSS
**4,484,366,336 bytes**. The new `main.mlirb` is **1,039,655,086 bytes**, SHA-256
`4169ffc381318694e15c03d7d8c1db080d11af8034c9f7eb6db82f64a50f7fa5`, versus the original
**1,039,655,099 bytes**, SHA-256
`0e58d55b8740710eee7bea8117aa8625a1061c142a63e03a1a79d47bf71ed9eb`. The **13-byte**
difference is within the round's permitted IR serialization variation; byte identity was
not expected. Metadata differs only in `compilation.date`: the original records
`2026-09-22T18:37:15.022124+00:00`, and the re-export records
`2026-09-22T19:31:22.977388+00:00`. All other fields are equal. The unchanged zoo `_bundle.py`
writes the current compilation time, so literal metadata equality is not met; both raw
files remain intact. Evidence: `results/round2/reexport.json` and
`logs/reexport-int8lin-round2.log`.

## Temperatures and the author's API

The checkpoint tensor, evaluated as fp32 `exp(log_temperature)`, is ordered Choice, Score,
Noul. Its values are **1.058534, 1.043141, 1.006767**, respectively. The source card's v0.3
line quotes **Choice 1.055, Score 1.008, Noul 1.047**. The exporter and the fixture use the
tensor values, with full fp32 precision retained in metadata and row temperatures. Evidence:
`results/oracle_model.json`, `results/oracle_contract_notes.md`, the fixture's
`temperature_exact_by_type`, and pinned source `README.md:207`.

For `k` options, the author's model divides its 256 raw slot logits by the question type's
temperature, masks slots at or above `k` except slot 255, and applies softmax. The client then
normalizes the first `k` probabilities; its Choice/Score `_decode_named` path performs a
second fp32 normalization. There is **no rounding** in the pinned client. Confidence is one
minus entropy normalized by `log(k)`. Only Choice exposes abstain; Score exposes probabilities
and confidence, and Noul exposes only the probability of yes. Evidence:
`results/oracle_contract_notes.md`, the fixture's `api_contract_note`, and the unchanged
author client at GitHub revision `5d04bcc` (`results/oracle_sources.json` records its hash).

The fixture contains **18 requests and 50 independent rows**: **48 kit rows** (24 Choice,
14 Noul, 10 Score) plus Choice rows with **40 and 255 options**. All use `permutations=1`.
Each author-encoded row ends with answer id **248082** and a newline, so the answer slot is
`len(ids_full)-2`. Truncating at the answer token changes the fp32 logits by at most
**0.000018596649**, below the **0.0001** causal-prefix tolerance. Independent-row assembly
equals the author's single-question API exactly for **18/18 requests**, comprising
**50/50 question calls**. Evidence: the fixture's `summary`, `rows` and `api_assembly`.

The author's one-pass API appends several questions to one causal sequence. Its later
questions can see earlier question text; the bundle and the kit instead answer one question
per row. On the same fixture requests the two API forms differ by a maximum
**|Δp| = 0.3754529953**. This is an API-layout finding, not a conversion failure. The one-pass
answers and each independent answer are retained beside one another in the fixture's
`api_assembly`; the maximum is in `summary.one_pass_max_abs_delta_p`.

## Oracle and runtime dependencies

The oracle runs the author's source files and client on CPU in fp32, with reference
gated-delta kernels. The working environment is Python **3.11.13**, torch **2.9.0**,
transformers **5.17.0**, safetensors **0.8.0**, huggingface_hub **1.32.0**, numpy **2.3.5**,
tokenizers **0.23.2**, and pydantic **2.13.5**. The original request for safetensors **0.7.0**
and huggingface_hub **0.36.2** could not resolve with transformers **5.17.0**, which requires
safetensors **≥0.8.0** and huggingface_hub **≥1.5.0,<2**. Only the oracle environment changed;
no flash-linear-attention or causal-conv1d package was installed. Evidence:
`results/env.json`, `results/oracle_env.json`, `logs/oracle-resolver-original-pins.log` and
[`conversion/slot/oracle_slot.py`](../conversion/slot/oracle_slot.py).

The export/runtime environment remains coreai-core **1.0.0b2**, coreai-torch **0.4.1**,
coreai-opt **0.2.1**, torch **2.9.0**, transformers **4.57.6**, safetensors **0.7.0**,
huggingface_hub **0.36.2**, numpy **2.3.5**, tokenizers **0.22.2**, and accelerate **1.15.0**.
The `coreai_models` import resolves inside the run's frozen `src/python` tree, from fork
commit `397b337`; no shared working tree was imported. Evidence: `results/env.json`.

## Probability and engine gates

Both bundles were AOT-compiled for **h16c GPU**, loaded with
`SpecializationOptions.default()`, and stepped with four fresh zero states per row and full
`position_ids` on every S=1 call. Processes hold at most **30 rows**; each wide row has its
own process. The first row is repeated at the end for a state-reset proof. Evidence:
[`readout gate`](../conversion/slot/readout_gate_slot.py),
[`fp16 transcript`](../models/openthai-systemone/gate-openthai-systemone-readout-fp16.json),
[`int8lin transcript`](../models/openthai-systemone/gate-openthai-systemone-readout-int8lin.json).

| Python runtime vs author's fp32 oracle | fp16 | int8lin |
|---|---:|---:|
| option argmax | 50/50 | 50/50 |
| margin ≥ 0.02 argmax | 49/49 | 49/49 |
| max \|Δp\| | 0.0050586462 | 0.0208126605 |
| mean of row mean \|Δp\| | 0.0002248096 | 0.0006594286 |
| max \|Δabstain\| | 0.0164879188 | 0.0188367814 |
| max absolute raw-logit error | 0.1001068950 | 0.1723060608 |
| repeated first row logits | identical | identical |

All logits were finite and nonconstant. The gate requires option argmax agreement where the
oracle margin is **≥0.02** and exact reset behavior; it records probability and abstain
errors. The only near-tie row is `r18-slot`, margin **0.0097392052**; both bundles agree on
it. The worst int8lin probability row, `r05-dry`, is a two-option Noul row with margin
**0.0616810024**, outside that near-tie category. Evidence: both readout transcripts and the
fixture. The accepted round-one records are `results/readout_{fp16,int8lin}.json`.

The Release Swift engine check uses the same input ids and takes one greedy token from all
**256 raw slot logits**. Its expected string is the saved tokenizer's decode of the same
bundle's Python `raw256_argmax`; it is not the option argmax. Both pipelined and sequential
engines match **50/50 per bundle**, **200/200 calls** total. Some raw slot indices decode to
a replacement character or whitespace; the parser preserves the string rather than stripping
it. Evidence: the
[`fp16 engine transcript`](../models/openthai-systemone/gate-openthai-systemone-engine-fp16.json),
[`int8lin engine transcript`](../models/openthai-systemone/gate-openthai-systemone-engine-int8lin.json)
and `results/engine_argmax_{fp16,int8lin}.json`. Tools were built Release from tag
**0.2.4-zoo**, commit `f7a75ec`, with Xcode **27.0 (27A266a)**; evidence:
`results/swift-build.json`.

The [decider runtime record](decider-0.8b-port.md#python-runtime-traps-on-macos-270-26a428)
documents the Python JIT failure on **26A428** (`MTL4CommandQueueErrorDomain error 1`, zero
logits) and an IOSurface leak near **25,000 calls per process**. This port used AOT from the
start and did not repeat that JIT experiment. Read-only MPSGraph scratch measurements were
**0 B before and after all 10 round-one runtime sessions**; nothing outside the run was
deleted. Evidence: `results/readout_{fp16,int8lin}.json` session records.

## Swift side

Measured through coreai-kit, the first tokenizer parity run matched **23/50 rows** because
swift-transformers applies the tokenizer's `Split` regex through Foundation string search,
whose match ranges snap to grapheme clusters. Thai letter runs retained their combining marks
and BPE merged differently: `กล่อง` became one token where the reference has `กล` + `่อง`.
The kit now cuts each text segment with the same regex through ICU on UTF-16, then encodes
each piece. Both bundles then match tokens **50/50**, slots **50/50** and option argmax
**50/50**, including the **40- and 255-option rows**. This is a kit/swift-transformers change;
the bundle tokenizer was unchanged. Evidence:
[`measurements-coreai-kit.json`](../models/openthai-systemone/measurements-coreai-kit.json),
supplied by the supervisor from its kit worktree and not re-derived in this run.

## Measurement notes

Device and build: **Apple M4 Max GPU**, macOS **27.0 (26A428)**, Xcode **27.0 (27A266a)**.
Round-one correctness work shared the GPU and supplies no throughput claim. The separate
int8lin benchmark ran with no other Core AI, Python or Swift engine job in the mandated
before/after process snapshots (`contended: false`). It used p=**128**, g=**256**,
**two launches × three trials per engine**, `COREAI_CHUNK_THRESHOLD=1`. The frozen benchmark
did not accept `--inference-engine-variant`; a copy of its source was patched only to expose
the option and pass `EngineOptions.variant`, then built Release. The original binaries were
left intact. Evidence: `results/round2/benchmark-engine-option.patch`,
`results/round2/benchmark-build.json`, `results/ps-before.txt`, `results/ps-after.txt`, and
[`llm-benchmark.json`](../models/openthai-systemone/llm-benchmark.json).

| int8lin engine | prefill proxy median (range), tok/s | decode median (range), tok/s | load per launch, seconds |
|---|---:|---:|---:|
| coreai-pipelined | 252.690 (243.793–258.566) | 250.776 (244.516–253.778) | 1.414586 / 0.167014 |
| coreai-sequential | 197.143 (196.114–201.258) | 194.829 (192.644–197.660) | 0.172054 / 0.165567 |

All trial and load values are in `llm-benchmark.json`. This is a **prefill-rate proxy** for
the S=1 graph: the benchmark's synthetic input ids come from the metadata's **256-wide**
output range, and synthetic generation does not measure a decision request. Load is one
measurement per launch and excludes warmup.

The supervisor's **measured through coreai-kit** results use the sequential engine and the
kit tokenizer. int8lin max/mean **|Δp| = 0.0226 / 0.0009**, max **|Δabstain| = 0.0222**;
fp16 **0.0051 / 0.0003**, max **|Δabstain| = 0.0165**. int8lin median wall time was
**354 ms/question** over the 50 fixture rows, two to three questions per state; the **255-option,
1,449-token** row took **7.2 s**. A **three-question** Thai ticket took **351 / 316 / 429 ms** for its
**57-, 63- and 83-token** rows, with **0** prefix tokens reused: the recurrent hybrid cannot rewind
mid-sequence, so every row is prefilled from its first token. (An earlier figure of 170 / 40 / 65 ms
with 79–81 reused tokens was measured on `minicpm5-2b` by mistake — the kit's `ask` command had
ignored `--bundle` — and is withdrawn.) Evidence:
[`measurements-coreai-kit.json`](../models/openthai-systemone/measurements-coreai-kit.json).

SemIf authored144, measured through coreai-kit on the Mac GPU with int8lin, gave **109/144**
raw and **0.7249 mean family balanced accuracy**. The **144 English rows**, **three options**
each, use SemIf's gold labels and unchanged `benchmarks/evaluate.py`. The kit README's values
on the same rows and evaluator are MiniCPM5-2B int8 **0.681** and Qwen3.5-4B int8 **0.821**.
Evidence: `measurements-coreai-kit.json`; these supervisor measurements were not repeated.
There is no phone gate for this port.

## Bundle provenance

int8lin `main.mlirb` is **1,039,655,099 bytes**, SHA-256
`0e58d55b8740710eee7bea8117aa8625a1061c142a63e03a1a79d47bf71ed9eb`; fp16 is
**1,506,078,533 bytes**, SHA-256
`f811a29cb5c2e6ff5ab5834e4ad288d04aa22c06791b4efcd1abb9c629a18e08`.
Whole LanguageBundles are **1,068,353,811** and **1,534,777,239 bytes**, respectively.
Evidence: `results/export_{int8lin,fp16}.json` and
[`recipe.toml`](../models/openthai-systemone/recipe.toml).

Hub staging retains the source `config.json` verbatim and the two original bundles. The
source snapshot has no license file; the staged `LICENSE` is the canonical Apache-2.0 text
from `https://www.apache.org/licenses/LICENSE-2.0.txt`, not the frozen Swift tree's BSD
license. Evidence: `results/round2/license-source.json`, `staging/MANIFEST.json` and
`staging/UPLOAD.md`. Publication and shared-tree folding are owner actions.
