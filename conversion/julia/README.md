# julia/ — Julia-1 (SupersonicLabs/Julia-1) → Core AI

The recipe behind `models/julia-1/` and `mlboydaisuke/Julia-1-CoreAI`. Julia-1 is an encoder-type decision
model: an mmBERT-small encoder (22 layers, hidden 384, 256k vocabulary) with a typed decision head — the same
modules and forward as laya's `DecisionModel`, at mmBERT-small's width. One forward answers one typed question
(choice / score / noul) over a state: the host builds
`[CLS] <type> question: … [SEP] [MASK] option … [SEP] state [SEP]`, the graph scores every position, and the
host reads the logits at the option markers and takes their softmax at T = 1.

Each stage is a script, gated against the one before; run them in this order from the repo root (paths resolve
through [`../_paths.py`](../_paths.py); the work dir is `<ZOO_WORK_ROOT>/julia-1/`, bundles go to
`<ZOO_EXPORTS>/julia-1/`):

| Stage | Script | Gate |
|---|---|---|
| 0 oracle | `uv run --python 3.12 conversion/julia/oracle_julia.py` | the publisher's own `julia` package (FastEngine, CPU fp32, strict, max_length 1024, head_length 512, marker-only head off), one question per forward: must reproduce the publisher's typed-decisions CPU FP32 result (426/600 choice, 542/800 score, 483/600 noul) and every answer of its `scripts/reproduce_typed.py`; its 100 Julia-1-ONNX parity requests against the PyTorch logits it recorded; the window-filling rows; each row alone right-padded to each export window (the tensor reference); every hidden state of the SUBSET rows through SDPA and through eager attention |
| 0b named | `uv run --python 3.12 conversion/julia/oracle_named_julia.py` | the publisher's `predict(state=..., questions=...)` on the 400 test cases: the answer dictionaries a host must reproduce from text |
| 1a exactness | `uv run --python 3.12 conversion/julia/gate_julia_exactness.py --window 1024\|512` | the re-authored graph under the publisher's torch 2.14.0 vs the publisher's eager-attention path on the SUBSET rows: every hidden state within 1e-6 relative, marker logits within 1e-5 (measured 1.3e-7 / 1.9e-6) — the graph itself is exact |
| 1 authoring | `python3 conversion/julia/gate_julia_authoring.py --window 1024\|512 [--dtype wfp16] [--negative-controls] [--rejudge]` | the re-authored graph (zoo venv, torch 2.9.0) vs the oracle on every row of the window (answer + tensor gates), the two-tier layer gate on the SUBSET rows (1e-4 absolute to layer 9, 5e-4 relative above = twice the publisher's own SDPA-vs-eager distance), pad isolation, five mutations that must be caught; `--rejudge` re-applies the current bars to an existing record |
| 2 host | `python3 conversion/julia/gate_julia_host.py [--bundle <folder> --compute gpu]` | the NumPy host (`_julia_host.py`) rebuilds every oracle row from text (ids and markers identical), renders every named question the publisher's way, and reproduces its answer dictionaries from the same logits; with a bundle, text → answers end to end vs `engine.predict` |
| 3 export | `python3 conversion/julia/export_julia.py --window 1024\|512 --dtype fp32 [--rows-stride 4]` | requires the authoring records to PASS; torch-export + decomposition, the same row gate before conversion (every row, or every 4th plus every SUBSET and window-filling row), then one `.aimodel` with one function (`main`) |
| 4 runtime | `python3 conversion/julia/gate_julia_runtime.py <variant dir> --compute cpu_only\|gpu` | the bundle on the Core AI runtime: every row, repeat drift, wrong-pairing control, load + warm timings; gpu holds the machine-wide flock GPU lock and discards the result if another GPU job appears |
| fixture | `python3 conversion/julia/fixtures_julia.py [--check]` | writes `models/julia-1/fixtures-julia-1.json`: requests, token ids, markers and the publisher's raw logits for a sample of every row set |
| swift | `python3 conversion/julia/swift_pieces_julia.py <pieces.json>`, then `swiftc -O -target arm64-apple-macos27.0 swift/JuliaDecisions.swift swift/parity/main.swift -o julia-parity` and `./julia-parity <folder> <pieces.json> gpu\|cpu_only` | the Swift host on the fixture's strict rows, the tokenizer replaced by the publisher's token ids per text piece: ids and markers identical, argmax identical, max \|dp\| <= 1e-3 |
| transcript | `python3 conversion/julia/transcript_julia.py` | writes `models/julia-1/gate-julia-1.json`: every gate's summary with the sha256 of its full record |
| stage | `python3 conversion/julia/stage_hf_julia.py --variants macos/fp32-s1024 macos/fp32-s512` | builds `<exports>/julia-1/hf_stage/` (hard-linked gated folders, card, LICENSE, SHA256SUMS, STAGING.md); uploads nothing |

The host is `_julia_host.py` (NumPy) and `julia_coreai.py` (the Python reference host: a variant folder, text in,
the publisher's answer dictionaries out); `swift/JuliaDecisions.swift` is the same host in Swift over the system
CoreAI framework, with the tokenizer passed in.

## Graph contract (one question row, batch 1, static window S ∈ {512, 1024})

| Function | Tensor | Shape | Dtype |
|---|---|---|---|
| `main` in | `input_ids` | [1,S] | int32, right-padded with PAD 0 |
| `main` in | `attention_mask` | [1,S] | int32, 1 real / 0 pad |
| `main` in | `qtype_onehot` | [1,3] | float32, choice / score / noul |
| `main` out | `token_logits` | [1,S] | float32, read at the marker positions |

The checkpoint's `act_head` and its `temperature` buffer ([1, 1, 1]) are not exported: the publisher's inference
API reads neither (`forward(..., return_actions=False)`, no temperature anywhere, `inference-policy.json`
`calibration: null`).

## Environments

- Stages 0, 0b and 1a: their own environment from the script header (Python ≥ 3.11, torch 2.14.0, transformers 5.0.0
  — the publisher's pins, `transformers>=5.0,<5.1` — pyarrow for the parquet).
- Stages 1–4: the zoo venv (torch 2.9.0, coreai-torch 0.4.1, coreai-core 1.0.0b2, tokenizers); the graph imports
  neither transformers nor the publisher's package.
- The checkpoint is the pinned HF snapshot `SupersonicLabs/Julia-1@a85b1273…` (sha256-checked on every load);
  the data are `LocalLLaMA/typed-decisions@c76749ec…` (test parquet) and `SupersonicLabs/Julia-1-ONNX@82a2fadf…`
  (`parity-cases.json`), both sha256-pinned in `_common.py` and fetched over plain HTTP when missing.

## What the port found

- **The publisher's result reproduces exactly.** Its own `scripts/reproduce_typed.py`, run unchanged here, gives its
  CPU FP32 totals (426/600 choice, 542/800 score, 483/600 noul); the oracle's one-question-per-forward logits give
  the same answers on all 2,000 questions, and so does the bundle's argmax on the Mac GPU and CPU.
- **The host is where Julia differs from laya**: options are the criteria descriptions as given, the builder
  refuses a cut instead of cutting, T = 1, one function. A laya host gives Julia other rows and other answers.
- **The layer bar was re-measured, not reused.** laya's relative 2e-4 fails this checkpoint at the final norm, where
  the publisher's own SDPA and eager paths differ by 2.87e-4 (S=1024); the same rule gives 5e-4. The graph is exact
  under the publisher's torch; the residual under the zoo's torch 2.9.0 comes from the final LayerNorm over values
  near 4.9e3.
- **fp16 weight storage is not free on an F32 checkpoint**: every argmax holds, but |dp| reaches 0.023. fp32 ships.
