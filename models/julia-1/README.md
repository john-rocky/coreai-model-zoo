# Julia-1 — Core AI

[🤗 mlboydaisuke/Julia-1-CoreAI](https://huggingface.co/mlboydaisuke/Julia-1-CoreAI) · Apache-2.0 · source [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1/tree/a85b127321d580d65176c89ced8273f305745d85) (revision `a85b127`) · base jhu-clsp/mmBERT-small

This is Supersonic Labs' Julia-1, a 144M typed decision model, as a Core AI `.aimodel` converted with Apple's `coreai-torch`; on the GPU of an Apple M4 Max it takes 16.00 ms per decision at the 1,024-token window, tokenization not included (2026-09-30).

Give it a state (a message or a JSON record) and a question with 2 to 20 options you write: pick one (`choice`), place the state on an ordered rubric (`score`), or answer false or true (`noul`). It returns a probability for every option from one forward pass and generates no text.

## Graph contract

```
function "main"
  input  "input_ids"       [1, S]  int32   the row, right-padded with PAD 0
  input  "attention_mask"  [1, S]  int32   1 over real tokens, 0 over padding
  input  "qtype_onehot"    [1, 3]  fp32    choice / score / noul
  output "token_logits"    [1, S]  fp32    the scorer at every position, read at the option markers
S = 512 or 1024, one bundle per window; batch 1; fp32 weights and compute
```

Inside `main` is the mmBERT-small encoder in the ModernBERT layout: 22 layers, hidden 384, 6 heads × 64,
GLU MLP 1152, global attention on layers 0, 3, …, 21 and a sliding window of inclusive radius 64 elsewhere,
RoPE θ 160,000, vocabulary 256,000. After it come the type embedding, two pre-norm transformer layers (384
wide, 6 heads, ReLU FFN 1536) and the scorer (LayerNorm → Linear → GELU → Linear to one logit).

The checkpoint also holds an act head (Linear 388 → 256, GELU, Linear 256 → 2) and a temperature buffer of
[1, 1, 1]. Neither is in the graph, because the publisher's inference API reads neither: it calls
`forward(..., return_actions=False)`, uses no temperature, and `inference-policy.json` says `calibration: null`.

## Host recipe

The row and the readout are the publisher's (`julia/data.py` `sequence()`, `julia/typed.py`). `metadata.json`
in each folder records them with the window, the head budget and the token ids.

- Row: `[CLS 2] tok("{type} question: {instructions}") [SEP 1] ([MASK 4] tok(" " + option)[:48])… [SEP 1]
  tok(state) [SEP 1]`. Each piece is tokenized alone, without special tokens, by the checkpoint's
  `tokenizer.json`. A dict or list state becomes `json.dumps(state, ensure_ascii=False)`. An option's marker is
  the position of its `[MASK]`.
- Options: the criteria descriptions as given. `choice` uses the mapping's values in order, and the answer is
  the caller's id. `score` uses the rubric in order. `noul` uses `[criteria.false, criteria.true]`, or the
  literal words `false` / `true` when the question has no criteria.
- Readout: the softmax of the raw marker logits at T = 1; no calibration exists. `choice` = the winning id,
  `score` = the expected rubric index Σ i·p_i (not rounded), `noul` = p[true].
- Strict encoding: 2–20 options, each at most 48 tokens, and no literal `<mask>` in any text (the publisher's
  non-strict builder turns it into a space). A question that would need any cut is refused, never truncated.
- Head budget (question plus options): 512 tokens at S = 1024, the publisher's protocol, and 256 at S = 512.
  Every row that fits 512 gets the same ids with either budget (2,065 of 2,065). All 2,000 typed-decisions
  test questions fit 1,024 tokens (the longest is 607); 1,965 fit 512.

The option text is where Julia differs from [laya-multilingual](../laya-multilingual/README.md), the same
design on mmBERT-base with the same graph inputs. laya hosts render `label: description`, `level i: …` and
`false: …` / `true: …`, so a laya host builds other rows for Julia and gets other answers.

## Measured (Mac)

Apple M4 Max, macOS 27.0 build 26A428, coreai-build 3600.83.1, coreai-core 1.0.0b2, 2026-09-30. Each bundle
ran every row of its window: at S = 1024 the 2,000 typed-decisions test questions, the publisher's 100 parity
requests and 20 window-filling rows; at S = 512 the 1,965 questions that fit, the same 100 and 35
window-filling rows. The GPU rows ran alone (the machine-wide GPU lock held, no other GPU job seen).
`cpu_only` is the parity option; its milliseconds were taken while other jobs used the CPU and are reference
values. A question's time covers the NumPy inputs, the graph and the marker gather, not the tokenization.

| Variant | S | Compute | Rows | Argmax | Max \|Δp\| | Marker max \|Δ\| | Repeat drift | Load | Warm median |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 1024 | GPU | 2,120 | 2,120/2,120 | 8.58e-5 | 5.71e-4 | 0 | 641 ms | 16.00 ms |
| fp32 | 512 | GPU | 2,100 | 2,100/2,100 | 7.97e-5 | 6.17e-4 | 0 | 702 ms | 8.28 ms |
| fp32 | 1024 | cpu_only | 2,120 | 2,120/2,120 | 5.91e-5 | 7.58e-4 | 0 | 648 ms | 109.7 ms |
| fp32 | 512 | cpu_only | 2,100 | 2,100/2,100 | 5.91e-5 | 7.58e-4 | 0 | 621 ms | 81.0 ms |

The GPU median at S = 1024 is over 60 warm questions (p10 15.89 ms, p90 16.16 ms); at S = 512, p10 is 8.20 ms
and p90 8.42 ms. The question right after load took 601 ms at S = 1024 and 118 ms at S = 512. Repeat drift is
the largest change in the marker logits when a row runs 3 times (212 rows at S = 1024).

On the typed-decisions test split, the bundle's own argmax (GPU, S = 1024) gives the same three totals as the
publisher's CPU FP32 run. The split is `LocalLLaMA/typed-decisions` at `c76749e` (400 cases, 2,000
questions); the publisher's run used torch 2.14.0, transformers 5.0.0, max_length 1024, head_length 512 and
strict encoding.

| | choice | score | noul |
|---|---:|---:|---:|
| Supersonic Labs, PyTorch CPU FP32 | 426/600 | 542/800 | 483/600 |
| this bundle, Mac GPU, S = 1024 | 426/600 | 542/800 | 483/600 |

From text to answer, the Python host (`julia_coreai.JuliaCoreAI`, the S = 1024 bundle, GPU) gives the
publisher's `engine.predict` answers on all 2,000 named questions: argmax 2,000/2,000, max |Δp| 8.58e-5,
score and noul values within 8.6e-5, 0 questions refused, and 426/542/483 when re-scored. The 2,000 questions
took 32.9 s, tokenization included.

The Swift host (`JuliaDecisions.swift` over the system CoreAI framework, built with Xcode 27.0 RC) was
checked on the 200 strict rows of the fixture below. Its row builder gives the publisher's ids and markers on
200/200 rows, and its answers through the bundle match the publisher's: argmax 200/200, max |Δp| 4.3e-5 on the
GPU and 3.3e-5 with cpu_only. That check read the publisher's token ids per text piece from a table, so a
Swift tokenizer is not part of it.

## Numerics gate

The oracle is the publisher's own `julia` package at the pinned revision (torch 2.14.0, transformers 5.0.0,
CPU fp32, 4 threads, one question per forward). On this Mac it reproduces 426/600, 542/800 and 483/600, and
its 2,000 answers are identical to those of the publisher's `scripts/reproduce_typed.py`, run unchanged here.
On the publisher's 100 WebGPU parity requests (`parity-cases.json` in `SupersonicLabs/Julia-1-ONNX` at
`82a2fad`) it matches the PyTorch logits the publisher recorded: argmax 100/100, max |Δ logit| 1.03e-4 (theirs
in padded batches of 4, ours one per forward).

Bars at every stage: the argmax equal to the oracle's on every row, max |Δp| ≤ 1e-3 over the options at
T = 1, and marker logits within 1e-3. The rows include near-ties: 58 typed questions have a top probability
below 0.6, 46 a top-two gap below 0.2 and 9 below 0.05 (the smallest 0.0098). The window-filling rows have no
padding: the 35 questions over 512 tokens, cut to exactly 512 by the publisher's non-strict builder (head
budget 256), and 20 synthetic states (consecutive test states in one JSON list) cut to exactly 1,024.

- Authoring: the graph re-authored in plain PyTorch from `model.safetensors` (no transformers or `julia` code,
  all 170 tensors loaded strictly), run in the zoo venv with torch 2.9.0 on every row of each window.
  S = 1024: argmax 2,120/2,120, max |Δp| 6.56e-5, marker 7.95e-4, typed totals 426/542/483. S = 512: argmax
  2,100/2,100, max |Δp| 6.56e-5, marker 7.95e-4.
- Layer gate on the hidden states: absolute ≤ 1e-4 on the embeddings and layers 0–9 (max 4.8e-5 at S = 1024,
  3.5e-5 at S = 512); relative ≤ 5e-4 from layer 10 through the head (max 2.07e-4 and 3.42e-4).
- Negative controls, caught at both windows (S = 1024 / 512): all-global attention 5.09 / 3.59 at layer 1;
  all-local 3.72 / 4.20 at layer 0; radius 63 instead of 64, 1.29 / 1.45 at layer 1; padding ignored,
  29.5 / 29.4 at layer 0; no type embedding, caught at the head input (marker logits moved by 0.81 at
  S = 512). Random ids in every pad position leave every real position bit-identical (11 rows per window).
  On the bundle, judging each row against another row's outputs fails on 1,724 of 2,120 pairs (S = 1024).
- Export: the `torch.export` graph after `coreai_torch` decomposition, gated before conversion on every 4th
  row plus the layer-gate rows and every window-filling row. S = 1024: 552 rows, argmax 552/552, max |Δp|
  5.81e-5. S = 512: 560 rows, argmax 560/560, max |Δp| 6.56e-5.
- Runtime: the table above.
- Host: the NumPy host (tokenizers 0.23.0rc0 on the checkpoint's `tokenizer.json`) rebuilds all 2,155 oracle
  rows from text with identical ids and markers. It renders the 2,000 named questions the publisher's way and
  reproduces the publisher's 2,000 answer dictionaries from the same logits (max difference 8.9e-16).

The relative bar is twice the publisher's own SDPA-vs-eager distance on this checkpoint (2.87e-4 at the final
norm, S = 1024), rounded down; laya's 2e-4, tried before it, failed at the final norm. Under the publisher's
torch 2.14.0 the graph equals the publisher's eager-attention path to 1.3e-7 relative at every hidden state
and 1.9e-6 on the marker logits (13 rows per window).

`fixtures-julia-1.json` beside this card (543 KB) holds the 100 parity requests, every 20th typed question and
5 window-filling rows per window, each with the request, token ids, markers and the publisher's raw logits.

## Precision

fp32 ships: the checkpoint's F32 weights as published, computed in fp32. fp16 weight storage with fp32 compute
was measured at S = 1024 and fails the 1e-3 bar, so it does not ship. It keeps argmax 2,120/2,120 and the
typed totals 426/542/483, but max |Δp| is 0.0234 (over 1e-3 on 325 rows, over 1e-2 on 32); a score's expected
index moves by up to 0.029 and a noul's p[true] by up to 0.023.

Neural Engine: not used. With a Neural Engine preference the Mac returns the GPU's results bit for bit at the same speed (844 rows, 2026-09-30), and `coreai-build compile --preferred-compute neural-engine` produces no Neural Engine region for macOS (h16c) or the iPhone 18 Pro (h19p): the graph computes in fp32, which the Neural Engine does not take.

## Bundle

**[mlboydaisuke/Julia-1-CoreAI](https://huggingface.co/mlboydaisuke/Julia-1-CoreAI)** (revision `d1e943545c64e20e73a88ae1f890227c349e22ba`, 2026-09-30; every file's sha256 and size checked against the staging manifest after the upload). One folder per window, each self-contained: the bundle,
`tokenizer/` (the checkpoint's files, unmodified), `metadata.json` (the decision contract), `reference.json`
(the window's rows with token ids, markers and the publisher's logits) and `provenance/` (the export manifest
and the export and runtime gate records).

| Folder | Platform | Format | Bundle | Bytes |
|---|---|---|---|---:|
| `macos/fp32-s1024/` | macOS 27 | JIT `.aimodel` | `julia1_fp32_s1024.aimodel` | 581,765,544 |
| `macos/fp32-s512/` | macOS 27 | JIT `.aimodel` | `julia1_fp32_s512.aimodel` | 578,357,637 |

The whole folders are 627,543,671 and 623,850,816 bytes.

Convert yourself: [`conversion/julia/`](../../conversion/julia/README.md), the staged scripts in order;
`recipe.toml` beside this card names the commands.

## Use it

Python, with `conversion/julia/julia_coreai.py` (the reference host; it needs `coreai-core`, `numpy` and
`tokenizers`):

```python
import sys
sys.path.insert(0, "conversion/julia")                  # the folder with julia_coreai.py and _julia_host.py
from julia_coreai import JuliaCoreAI

julia = JuliaCoreAI("Julia-1-CoreAI/macos/fp32-s1024")    # a download of the HF repo; GPU by default
result = julia.predict(
    state="I was charged twice for the same order.",
    questions={
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "billing": "Billing and payment disputes",
                "shipping": "Shipping and delivery",
                "access": "Account access and login",
            },
        },
    },
)
print(result["answers"]["team"]["choice"])
print(result["answers"]["team"]["probabilities"])
```

The call and the returned dictionary follow the publisher's `engine.predict(state=..., questions=...)`.
`JuliaCoreAI(folder, compute="cpu_only")` loads the parity option.

Swift, with `conversion/julia/swift/JuliaDecisions.swift` added to your target (the system CoreAI framework,
macOS 27). It needs a tokenizer for the folder's `tokenizer/tokenizer.json`, passed in as `encode:`; with
swift-transformers:

```swift
import CoreAI
import Tokenizers   // swift-transformers

let folder = URL(filePath: "Julia-1-CoreAI/macos/fp32-s1024")   // a download of the HF repo
let tokenizer = try await AutoTokenizer.from(modelFolder: folder.appendingPathComponent("tokenizer"))
let julia = try await JuliaDecisions(folder: folder, encode: { text in
    tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
})
let answers = try await julia.predict(state: "I was charged twice for the same order.", questions: [
    ("team", .choice("Which team should handle this request?", [
        ("billing", "Billing and payment disputes"), ("shipping", "Shipping and delivery"),
        ("access", "Account access and login")])),
])
print(answers["team"]!.choice!)
```

`JuliaDecisions` loads the folder with `AIModel` itself; Julia-1 is not a CoreAIKit catalog model. A
structured state goes in as text, serialized the way Python's `json.dumps(state, ensure_ascii=False)` writes
it. The Swift check above did not run a tokenizer; `reference.json` and the fixture hold the ids a tokenizer
has to reproduce.

## Other runtimes

Not compared here: the publisher's ONNX export with a WebGPU adapter, `SupersonicLabs/Julia-1-ONNX`; MLX,
`zainmerchan/Julia-1-MLX`; GGUF, `andrelucas/Julia-1-GGUF`.

## License and limits

Apache-2.0, from the upstream card: its metadata declares it, and its README says "The model artifacts are
licensed under Apache 2.0." The upstream repository has no LICENSE file. The base, jhu-clsp/mmBERT-small, is
MIT. Julia-1 and its weights are the work of Supersonic Labs; these bundles carry the weights in fp32 as
published. The fixture's requests come from `LocalLLaMA/typed-decisions` and the Julia-1-ONNX
`parity-cases.json`, both Apache-2.0.

Not tested: iPhone (not measured), other Macs and OS builds, batched shapes, windows other than 512 and 1024,
the publisher's other benchmarks (not re-run here), and the publisher's `Router` for more than 20 options.
