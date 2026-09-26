# GLiNER2.5-Decide — Core AI

Zero-shot text classification: the caller names the tasks and their labels at call time, and one forward
answers all of them with a probability per label. [`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide)
(Apache-2.0) is a DeBERTa-v3-large encoder with a trained label head, 486M parameters with its
128k-token embedding table (340M on Fastino's card). Its classification path is one static Core AI graph
per sequence length; the tokenizer, the schema layout and the decision rule are the host's. The span,
record and counting heads of the checkpoint are not converted.

The graph is the classification half of [GLiNER2-PII](../gliner2-pii/README.md)'s fused graph, on the
English DeBERTa-v3 tokenizer instead of mDeBERTa's. Other conversions of this path exist as ONNX:
[onnx-community/GLiNER2.5-Decide-ONNX](https://huggingface.co/onnx-community/GLiNER2.5-Decide-ONNX) and
[nishparadox/gliner2.5-decide-onnx](https://huggingface.co/nishparadox/gliner2.5-decide-onnx).

- 🤗 [mlboydaisuke/GLiNER2.5-Decide-CoreAI](https://huggingface.co/mlboydaisuke/GLiNER2.5-Decide-CoreAI)
  (revision `7464c91`, 2026-09-26; mirror:
  [coreai-community/GLiNER2.5-Decide-CoreAI](https://huggingface.co/coreai-community/GLiNER2.5-Decide-CoreAI)).
- Oracle, export and gates: [`conversion/gliner25_decide_oracle.py`](../../conversion/gliner25_decide_oracle.py),
  [`conversion/export_gliner25_decide.py`](../../conversion/export_gliner25_decide.py); the Swift graph
  runner, AOT compile and staging: [`conversion/gliner25_decide/`](../../conversion/gliner25_decide/).
- iPhone gate app: [`apps/DecideGate`](../../apps/DecideGate).
- Swift host: CoreAIKit `TextClassifier` ([Examples/TextClassify ↗](https://github.com/john-rocky/coreai-kit/tree/gliner-decide/Examples/TextClassify)).
- Porting notes: [`knowledge/gliner25-decide.md`](../../knowledge/gliner25-decide.md).

## How it works

`gliner2` puts the tasks ahead of the text and reads the hidden state at each label's `[L]` marker:

```
( [P] intent ( [L] order_status [L] refund_request … ) ) [SEP_STRUCT] ( [P] urgency ( [L] low … ) ) [SEP_TEXT] the text, lowercased, word by word .
```

```
input_ids [1, S] int32   attention_mask [1, S] int32   label_idx [1, 32] int32   ->   logits [1, 32] float32
```

Inside the graph: `DebertaV2Model` from transformers 4.57.6 with the checkpoint's `encoder.*` tensors
loaded strict (390 tensors), the relative-position bucket table computed at export and stored as a
constant, an `index_select` of the `[L]` rows, and the head `Linear(1024, 2048) → ReLU → Linear(2048, 1)`
(the `classifier.*` tensors). Unused `label_idx` slots repeat the first `[L]` position. `S = 256` and
`S = 512` are exported from the same script; `MMAX = 32` labels per call covers every row of
fast-decisions (the largest label set is 28).

The host is `gliner2` 2.0.0's collate path: a `.` appended when the text ends without `.`, `!` or `?`;
words split with its regex (URL, e-mail, @handle, `\w+(?:[-_]\w+)*`, any other character), lowercased;
labels, task names, prompts and descriptions kept as written; every piece tokenized on its own with
`tokenizer.tokenize`, no `[CLS]` / `[SEP]`; markers emitted by id (they are added tokens outside the
Unigram vocabulary). Decision: softmax and argmax for a single-label task; a sigmoid per label and
`cls_threshold` (default 0.5) for a multi-label task, the best label alone when none passes; temperature
1. The host runs the smallest shape the tokens fit and drops words from the end when 512 is exceeded.

## Verification

Reference: `gliner2` 2.0.0, fp32, CPU, in its own venv (torch 2.12.1, transformers 4.57.6), on the
checkpoint at `7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6`. Every fixture case carries `classify_text`'s
own output next to the logits, and the two agree on all 787 decisions with probabilities equal to
2.3e-7. The fixture also matches the reference file published with
onnx-community/GLiNER2.5-Decide-ONNX (13 decisions, max |Δprob| 4.8e-7).

Fixture: the 21 `classify_text` examples of the model card, verbatim, and
[`fastino/fast-decisions`](https://huggingface.co/datasets/fastino/fast-decisions) (development
split, revision `1a33070`): per domain the first 20 rows whose collated input fits 256 tokens (340 rows,
580 decisions) and up to 10 rows that need 257–512 tokens (93 rows, 181 decisions). 454 texts, 787
decisions. Bar: every decision equal to the reference (single-label: the argmax; multi-label: the set at
the task's threshold). Logit and probability differences are reported without a bar.

| stage | texts / decisions | decisions equal | max \|Δlogit\| | max \|Δprob\| |
|---|---|---|---|---|
| re-authored graph, fp32, torch 2.9.0 CPU | 454 / 787 | 787 | 1.24e-5 | 2.5e-6 |
| fp16 S = 256, Mac GPU (M4 Max, macOS 27.0 26A428), JIT | 361 / 606 | 606 | 0.0140 | 0.0022 |
| fp16 S = 512, Mac GPU, JIT | 454 / 787 | 787 | 0.0184 | 0.0025 |
| fp16 S = 256 and 512, Mac GPU, AOT h16c | 361 / 606, 454 / 787 | 606, 787 | 0.0140, 0.0184 | 0.0022, 0.0025 |
| Swift `decide-selftest` and CoreAIKit `TextClassifier`, Mac GPU | 454 / 787 | 787 | 0.0140 | logits bit-identical to the Python run |
| fp16 S = 256, iPhone 18 Pro GPU, h19p | 361 / 606 | 606 | 0.0187 | 0.0023 |
| fp16 S = 512, iPhone 18 Pro GPU, h19p | 454 / 787 | 787 | 0.0187 | 0.0023 |

- The comparator goes red: `label_idx` shifted by one (the label's first sub-word instead of its `[L]`)
  changes 229 of 361 short texts' decisions (max |Δlogit| 13.6), in torch and in the engine alike.
- The stock forward, exported as is, fails: 604 of 606 decisions at S = 256, max |Δlogit| 0.73, every
  error on texts of 130 tokens or more. coreai-torch 0.4.1 lowers `aten.div.Tensor(int, int)` to an
  integer division (`broadcasting_divide` on `si32`), so DeBERTa's `make_log_bucket_position` puts every
  distance of 129 or more in bucket ±128. The shipped graph carries the table computed in torch; its
  fp32 output equals the stock forward's.
- Swift host: token ids bit-identical to `gliner2`'s on 454 / 454 texts (the word regex follows
  Python's `\w`, `\s` and `str.lower`, which ICU's classes do not match on `²` and on a final sigma);
  the Swift graph runner's Mac GPU logits equal the Python engine's bit for bit (4,376 / 4,376 at
  S = 256, 5,749 / 5,749 at S = 512); the decision rule reproduces NumPy's float64 softmax bit for bit.
- iPhone rows: iOS 27.0 (build 24A437), [`apps/DecideGate`](../../apps/DecideGate) runs `20260926-124014`
  and `20260926-125706`, 2026-09-26. The iPhone's logits are within 0.0156 of the Mac GPU's with every
  decision equal; runs repeat bit for bit.
- On the 340 short rows, one call per row with all of its heads at threshold 0.5, 63.45 % of decisions
  equal the gold label (macro over heads 64.07 %, all heads of a row right 49.71 %). This is a
  development-split number under this protocol; Fastino's 60.2 % is the held-out test split under
  theirs, and the two do not compare.

## Speed

Warm median over 100 calls after 5 warm-up calls, fixture inputs. Mac: M4 Max, macOS 27.0 (26A428),
another session on the GPU (contended). iPhone 18 Pro: iPhone19,2, iOS 27.0 (24A437), GPU, low power
off, phone rested 7 minutes, thermal state nominal before and after the bench (run `20260926-125706`).

| device | S = 256 | S = 512 | load first / second | first call | footprint after load |
|---|---|---|---|---|---|
| iPhone 18 Pro, h19p | 37.8 ms (p90 38.1) | 92.9 ms (p90 94.5) | 1.29 s / 0.14 s; 1.82 s / 0.84 s | 1.23 s; 0.41 s (cold) | 220 MB; 376 MB |
| Mac GPU, JIT `.aimodel` | 28.1 ms (p90 28.4) | 52.0 ms (p90 52.5) | 1.3 s / 0.01 s | 74 ms; 78 ms (specialization cache warm) | — |

A run of several hundred calls without a pause takes the iPhone from nominal to fair within a minute
and slows the calls: 92 → 151 ms across one 454-case loop at S = 512, and 84 ms at S = 256 for a run
started on a warm phone against 38 ms rested. After five minutes of rest the speed is back even while
the thermal state still reads fair; the cause was not isolated.

Bundles: `.aimodel` 873.0 MB (S = 256) and 874.6 MB (S = 512); `.h19p.aimodelc` 974.0 MB and
975.6 MB, 0 Neural Engine regions (GPU preferred). AOT compile 3–4 s each with coreai-build 3600.83.1.
Layout since Hub revision `820d4e90` (2026-09-26): `macos/` and `ios/` hold the two JIT `.aimodel` (the
iPhone 18 Pro specializes them itself: first load 1.48 / 2.32 s, 0.36 / 0.13 s on relaunch, the same decisions;
[`knowledge/gliner25-decide.md`](../../knowledge/gliner25-decide.md) §7), `ios-h19p/` the two
`.h19p.aimodelc` (iPhone 18 Pro only). Until then `ios/` carried the h19p bundles.

## Reproduce

`O` is the oracle venv (`uv venv --python 3.12`; gliner2 2.0.0, torch 2.12.1, transformers 4.57.6,
peft 0.21.0, accelerate 1.15.0), `E` the export venv (Python 3.11, torch 2.9.0, coreai-torch 0.4.1,
coreai-core 1.0.0b2). `HF_HOME` holds the checkpoint and the dataset.

```bash
$O conversion/gliner25_decide_oracle.py --work-dir <dir>                       # fixtures/, lengths, own-subset accuracy
$E conversion/export_gliner25_decide.py --fixtures <dir>/fixtures/readme21.json <dir>/fixtures/fast_decisions_s256.json \
     -S 256 --mmax 32 --dtype float16 --poison --output-dir <dir>/exports        # GATE-1/2, convert, GATE-3 on the Mac GPU
$E conversion/export_gliner25_decide.py --fixtures <dir>/fixtures/readme21.json <dir>/fixtures/fast_decisions_s256.json \
     <dir>/fixtures/fast_decisions_long.json -S 512 --mmax 32 --dtype float16 --poison --output-dir <dir>/exports
$E conversion/gliner25_decide/aot_compile.py --tag s256 --targets ios_gpu_h19p,mac_gpu   # and --tag s512
cd conversion/gliner25_decide/swift && swift build -c release                    # decide-selftest, DecideGate's package
cd ../../../apps/DecideGate && ./_build.sh && ./_stage.sh && ./_gate.sh <udid>   # the iPhone gate
$E conversion/gliner25_decide/stage_ship.py                                       # the Hugging Face layout + SHA256SUMS
```

`--relpos runtime` exports the stock forward (the negative evidence above). Swift needs
`DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer`.

## Lessons

- **Integer division in the converter.** coreai-torch 0.4.1 turns `int / int` (true division in torch)
  into an integer division, and DeBERTa's log-bucket positions depend on it. A gate on short texts
  passes; the error starts at 130 tokens. Bake input-independent tables in torch.
- **The iPhone 18 Pro is h19p.** `AIModel.deviceArchitectureName` says so, and an h18p bundle is refused
  at load (`incompatibleCompiledAssetArchitecture`). The 17 Pro is h18p. Compile per device.
- **A label costs about 4 tokens.** With 28 labels, half of the support-intent rows overflow 256
  tokens; two shapes (256 and 512) cover the dataset, and the host picks the smallest.
- **The oracle checks itself.** Every fixture case stores `classify_text`'s output beside the logits
  taken from the same forward; a host that reproduces the decision rule can be gated without the model.
- **Bit-identical Swift takes three things.** Python's `\w` / `\s` / `str.lower` rather than ICU's
  classes, `JSONDecoder` rather than `JSONSerialization` for the fixture numbers, and NumPy's pairwise
  summation order in the softmax.
- **Bench a rested phone, bench before the case loop.** The thermal state read at step boundaries
  is a coarse signal; the call time is the measurement.

## License

Apache-2.0, the source model's license. The published repo carries `LICENSE` and `NOTICE` (origin
revision, weight checksum, what was converted); keep both with any part of it you redistribute.
DeBERTa-v3-large is MIT.
