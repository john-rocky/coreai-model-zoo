# APUS-OpenJev-v1-4B: full-depth letter readout

2026-09-23. Source `apus-ailab/APUS-OpenJev-v1-4B` at
`65797c526c27c4d24f564333779162cd4a64328e`, checkpoint-5949 merged BF16, Apache-2.0.
The [card](../models/apus-decision-v1-4b/README.md),
[recipe](../models/apus-decision-v1-4b/recipe.toml) and
[gate summary](../models/apus-decision-v1-4b/gate-apus-decision-v1-4b.json) are the zoo record.
Run evidence paths below are relative to
`~/code/codex-conversions/2026-09-23/apus-openjev-4b-coreai/`.

## HF-id swap

The text tower matches Qwen3.5-4B: 32 layers (24 linear attention, 8 full attention), hidden
size 2560, vocabulary 248320 and a tied vocabulary head. The checkpoint stores
`model.language_model.*` tensors and `model.visual.*`; no standalone `lm_head.weight` is
expected because the embedding supplies it. The default
`from_hf_memory_efficient(hf_config_attr="text_config")` loader applies unchanged and skips
vision. The zoo `qwen3.5-4b` / `agents-a1-4b` recipe therefore needs only the HF-id swap.

The exporter and `_bundle.py` are unchanged zoo `082fe55` copies; the frozen Python overlay
is fork `397b337`. Exporter SHA-256 is
`70f413b685a5537a62db084c73d1ca56de89f8b704b5d4d69d1f91ba393210a7`.
`int8hu --head-sym` exports S=1, context 4096, `[1,1,248320]` logits, and logs loop-free
single-step enabled on all 24 linear layers. The whole LanguageBundle is 5,770,814,417 bytes;
`main.mlirb` is 5,742,220,411 bytes with SHA-256
`7b719b75f6782d60ff0e082230e464f041927ef7bbd3ffb5cfbd2a9352e32f23`.
Evidence: `results/export_int8hu.json`, `results/export_resolution.json`,
`results/copied_scripts.json`. No re-export was required in round 2.

## Author oracle and prompt

The oracle vendors the source snapshot's `openjet_runtime` unchanged; all five runtime
file hashes match `evaluation/runtime-smoke.json`. The author asserts transformers
**5.16.1** exactly. The CPU fp32 oracle environment uses torch **2.9.0**, safetensors
**0.8.0**, huggingface_hub **1.32.0**, tokenizers **0.23.2**, numpy **2.3.5** and Python
**3.11.13**. The export/runtime environment separately retains transformers **4.57.6**,
coreai-core **1.0.0b2**, coreai-torch **0.4.1** and coreai-opt **0.2.1**.
Evidence: `results/env.json`, `results/oracle_env.json`, `results/oracle_vendor.json`.

`jev.dynamic.prompt.v2` puts `Shared state:` and the string state before a sorted JSON object
containing criteria descriptions with labels, instructions and primitive, then
`Return only the selected letter: A, B, ….\nAnswer:`. This is one user message under the
source chat template with `add_generation_prompt=True`, `enable_thinking=False`, encoded
without added special tokens. The fixture's `prompt_examples` preserves the decoded ids and
exact tails. Read the last compiled position, `slot = len(ids)-1`; A–P are token ids 32–47,
with single-token-at-boundary assertions for every row.

At full depth, gather the row's label logits as fp32 and softmax with **T = 1**. The author
says `calibrated: false`. Choice maps probabilities to criterion ids; noul and score_level
use the exact yes/no criteria, with the proposition in instructions. `score_level` judges
one proposition, not an aggregate score. The low-effort 16-layer candidate-row projection
requires its own graph and is not exported. Its fixture outputs are evidence only; it
agrees with the full-depth argmax on 38/48 rows. The source card's Frozen80 figures are
quoted as author measurements in the card; the merge's failed probability gate is distinct
from this conversion's comparison against the released merged checkpoint.

## Fixture and Mac gates

The [fixture](../models/apus-decision-v1-4b/fixtures-apus-decision-v1-4b.json) contains 48
requests / 48 rows: 28 choice (8 with 16 criteria, 8 with 2–3, 12 with 4–8), 12 noul and
8 score_level; 16 requests include Chinese and 30 contain structured states as text.
States are short: **34–66 tokens**. Compiled rows span **124–1,672 tokens**; two rows above
1,024 are `zoo_only`, lengthened through instructions. Minimum top-2 oracle margin is
**0.948419**, so there are **no near ties**. Every API probability equals its recorded
row probability, 48/48. This set tests numerical/readout parity, not accuracy on difficult
or ambiguous decisions. Evidence: `results/fixtures.json` and its summary.

The accepted round-one GPU readout has **48/48** option argmax agreement, max |Δp|
**0.005301987752318382**, mean of row means **0.000126608183026633**, max absolute label-logit
error **0.2360076904296875**, and **48/48** full-vocabulary argmaxes inside the label set.
The first row's logits reproduce bit-identically after reset. Each row has fresh zero
states and S=1 calls over the complete compiled ids. The Python runtime loads an AOT
**h16c GPU** asset with `SpecializationOptions.default()`; processes evaluate at most
15 prompts including resets, with each wide row separate. This follows the decider
record's Python GPU JIT failure and IOSurface leak on **26A428**; those failure experiments
were not repeated. Evidence: `results/readout_int8hu.json` and the
[zoo-layout readout transcript](../models/apus-decision-v1-4b/gate-apus-decision-v1-4b-readout.json).

Release `llm-runner` gets raw fixture ids, warmup off, one greedy token and
`COREAI_CHUNK_THRESHOLD=1`. Both engines match the Python full-vocabulary argmax's decoded
text **48/48** each (**96/96**); the same calls also match the oracle letter 96/96.
The parser preserves output whitespace. Evidence: `results/engine_argmax_int8hu.json` and
the [zoo-layout engine transcript](../models/apus-decision-v1-4b/gate-apus-decision-v1-4b-engine.json).
The [alphabet gate](../models/apus-decision-v1-4b/gate-apus-decision-v1-4b-alphabet.json)
passed 16/16 tokens against the CPU fp32 overlay oracle on the raw alphabet prompt. Its unchanged
generic tool writes `weights_pinned: false`; this run's offline `refs/main` resolves to the
pinned source and all three restored LFS hashes match. Effective-pin evidence is
`results/round2/alphabet-download.json` and `results/round2/alphabet-execution.json`.

## Swift and coreai-kit

The supervisor's measurements through **coreai-kit** use `Decision.Format.sharedState`, the
sequential engine and the kit's own chat-template rendering. The 40 choice/noul rows match
tokens **40/40**, label slots **40/40** and option argmax **40/40**, max |Δp| **0.0055**, mean
**0.0002**. Eight score_level rows are skipped because the kit has no such primitive;
substituting noul preserves proposition semantics but changes the primitive JSON and prompt.
Median fixture time is **2.8 s/question** over **124–1,672 tokens**; the 1,672-token row is
**24 s**. A two-question workflow with **20 state tokens**, a 3-criteria choice and noul,
takes **2,069 ms / 134 tokens** and **1,540 ms / 107 tokens**, with **0 tokens reused**;
the recurrent hybrid re-prefills each row, about **15 ms/token**. Answers: **close 0.999**;
**refund asked P(yes) 0.000**.

SemIf authored144, measured through coreai-kit: **144 English rows**, three options,
SemIf gold labels, unchanged `benchmarks/evaluate.py`, kit rendering as choice; **131/144**
raw, **0.906 mean family balanced accuracy**, median **1.96 s/decision**. The kit README's
same-row/evaluator figures are Qwen3.5-4B int8 zero-shot **0.821**, MiniCPM5-2B int8 **0.681**,
and the OpenThai run **0.725**. These supervisor measurements were not re-derived here;
see [kit record](../models/apus-decision-v1-4b/measurements-coreai-kit.json).

Trap: an earlier OpenThai `ask` measurement ignored `--bundle` and used the catalog default;
the command was fixed, and every figure above printed
`model: apus_decision_… format: sharedState`.

## Measurement notes

Apple M4 Max GPU, macOS **27.0 (26A428)**, Xcode **27.0 (27A266a)**, Release tools from fork
**0.2.4-zoo / f7a75ec**. Round-one correctness supplies no throughput claim. Round-two A
ran before all other round-two GPU work, with no other engine workload in the mandated
before/after process snapshots, `contended: false`; the GPU lock was read and never changed.
A source copy adds only the benchmark engine-variant option and forwards it to
`EngineOptions`. The patch matches the sibling's byte for byte. Evidence:
`results/round2/benchmark-build.json`, `results/round2/benchmark-engine-option.patch`,
`results/ps-before.txt`, `results/ps-after.txt`.

| Engine | S=1 prefill median (range), tok/s | decode median (range), tok/s | load per launch, s |
|---|---:|---:|---:|
| coreai-pipelined | 81.781 (81.166–81.870) | 80.379 (79.323–80.606) | 2.899 / 0.700 |
| coreai-sequential | 69.369 (69.257–69.524) | 68.493 (67.860–68.605) | 0.703 / 0.706 |

Protocol: p=128 / g=256, two launches × three trials per engine, `COREAI_CHUNK_THRESHOLD=1`;
load measured per launch excludes warmup. A decision costs one S=1 prefill of the whole row,
so decode throughput is a **prefill-rate proxy**; kit time above measures decisions. Trials:
[llm-benchmark.json](../models/apus-decision-v1-4b/llm-benchmark.json). No phone result is claimed.
