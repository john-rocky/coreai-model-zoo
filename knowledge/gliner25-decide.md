# GLiNER2.5-Decide: zero-shot classification as one DeBERTa-v3 graph

Lessons from porting the classification path of
[`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide) (Apache-2.0, a DeBERTa-v3-large
encoder with a label head) to Core AI. The caller names the labels at call time and gets one probability per label.
The graph is the encoder, a gather and the head; the tokenizer, the schema layout and the softmax / sigmoid are host
code. The reference is `gliner2` 2.0.0 in fp32. Code and gates: [`conversion/export_gliner25_decide.py`](../conversion/export_gliner25_decide.py),
[`conversion/gliner25_decide_oracle.py`](../conversion/gliner25_decide_oracle.py),
[`conversion/gliner25_decide`](../conversion/gliner25_decide); card: [`models/gliner25-decide`](../models/gliner25-decide/README.md).
The first GLiNER port here, extraction rather than classification, is [`gliner2-pii.md`](gliner2-pii.md).

## 1. Export only the classification path

`classify_text` lays every task out ahead of the text, runs the encoder once, and reads one hidden state per label:

```
( [P] intent ( [L] order_status [L] refund_request … ) ) [SEP_STRUCT] ( [P] urgency ( [L] low … ) ) [SEP_TEXT] the words .
```

The row at each `[L]` marker goes through the head (Linear 1024→2048, ReLU, Linear 2048→1) and becomes that label's
logit. The checkpoint also carries the span and count heads of entity extraction. Classification never calls them,
so the graph leaves them out:

```
input_ids [1, S] int32   attention_mask [1, S] int32   label_idx [1, 32] int32   ->   logits [1, 32] float32
```

`label_idx` holds the `[L]` positions of every task, concatenated in task order; unused slots repeat the first
position, a safe gather the host never reads. The host keeps each task's slice and applies `gliner2`'s rule: softmax
and argmax for a single-label task, a sigmoid per label and a threshold for a multi-label one, the best label alone
when none passes. Labels enter as tokens and positions, so one bundle answers any label set up to 32 labels per call,
the property [`gliner2-pii.md`](gliner2-pii.md) §2 found for extraction.

Two shapes ship, S = 256 and S = 512, and the host runs the smallest one the tokens fit. On the 1,700 rows of the
`fastino/fast-decisions` development split, 88.4 % fit 256 and all fit 512 (the longest is 431 tokens). The domain
with 28 labels fits 256 in 47 of its 100 rows: the labels take room in the same window as the text. Over the 454
fixture texts the schema costs about 4 tokens per label (median 3.6, counting each task's markers and name).

## 2. coreai-torch 0.4.1 turns `int / int` into integer division

DeBERTa-v3 maps each token distance to a relative-position bucket. transformers' `make_log_bucket_position` divides
the int64 distance by `mid` (128 here) with `/`, which is true division in PyTorch. coreai-torch 0.4.1 lowers that
`aten.div.Tensor(int, int)` to an integer divide and casts afterwards:

```
%3 = coreai.decomposable.broadcasting_divide %1, %2 : (tensor<1x8xsi32>, tensor<si32>) -> tensor<1x8xsi32>
%4 = coreai.cast %3 : tensor<1x8xsi32> to tensor<1x8xf32>
```

Reproduce it with a module that returns `x.to(torch.int64) / 128` and `x.to(torch.int64).float() / 128`, exported with
`torch.export`, `run_decompositions(get_decomp_table())` and `TorchConverter().to_coreai()`. For the inputs
1, 64, 127, 128, 129, 200, 255, 511, PyTorch returns 0.0078 … 3.9922. The engine returns 0, 0, 0, 1, 1, 1, 1, 3 for the
first output, on the GPU and on `cpu_only` alike; the explicit float division is exact. `print(prog)` after
`to_coreai()` shows the IR above.

In the bucket formula, log(129 / 128) becomes log(1) = 0, so every distance from 129 to 255 lands in bucket ±128.
At S = 256, 16,256 of the 65,536 table entries are wrong, by up to 64 buckets. Nothing reports it. Texts of 129
tokens or fewer are unaffected (max |Δlogit| 0.010, fp16 noise). From 130 tokens the error grows with length:
max |Δlogit| 0.40 at 130–191 tokens and 0.73 at 192–255. On the S = 256 fixture that flipped 2 of 606 decisions.
All 21 examples on the model card are shorter than 128 tokens, so a check on the card's examples alone passes.

The bucket table depends only on S, so the export computes it in PyTorch (`build_relative_position`), asserts it
equals `encoder.get_rel_pos`, stores it as a buffer and passes it to `DebertaV2Encoder.forward(relative_pos=…)`. In
fp32 PyTorch the output is unchanged (max |diff| 0.0 on the 8 longest fixture texts); the fp16 bundle then decides 606
of 606, max |Δlogit| 0.014.
`--relpos runtime` keeps the stock forward, as the negative evidence.

GLiNER2-PII (mDeBERTa-v3-base, S = 256) runs the same function. Its texts longer than 128 tokens have not been measured.

## 3. The iPhone 18 Pro is h19p

`AIModel.deviceArchitectureName` on the iPhone 18 Pro (`iPhone19,2`) is `h19p`. A bundle compiled with
`--architecture h18p`, the iPhone 17 Pro's target, fails to load there in 0.04 s:
`incompatibleCompiledAssetArchitecture(device: "h19p", asset: ["h18p"])`. Compile once per architecture, name the
result `<name>.<arch>.aimodelc`, and let the app pick its file by `deviceArchitectureName`. Both compiles take 3–4 s
and produce 974 MB for S = 256. The h19p bundle loaded in 1.29 s the first time after install.

## 4. The oracle

- **Its own environment.** `gliner2` 2.0.0 reads this checkpoint; the zoo venv's 1.3.2 does not. The pins live in the
  oracle script's PEP 723 header, so `uv run --python 3.12 conversion/gliner25_decide_oracle.py` rebuilds the
  environment. `gliner2` 2.0.0 imports `peft` unconditionally, so `peft` and `accelerate` are pinned too. CPU, fp32,
  one text per forward, so that batch padding plays no part.
- **Self-consistency.** The oracle computes its logits on the path `classify_text` takes internally (the encoder,
  `extract_embeddings_from_batch`, the head) and asserts that its decisions equal `classify_text`'s on every task:
  787 of 787, with bit-equal probabilities.
- **The text it sees.** `gliner2` appends `.` to a text that does not end in `.`, `!` or `?` (1.3.2 and 2.0.0 alike).
  The fixtures store the text after that, and the host must do the same. The shipped GLiNER2-PII host does not
  append it, so on such a text its input differs from `gliner2`'s by that one token.
- **An outside check.** `onnx-community/GLiNER2.5-Decide-ONNX` publishes a `reference.json` recorded with
  `gliner2` 2.0.0 on its author's machine. This oracle agrees on 6 of 6 token-id sequences and 13 of 13 decisions,
  max |Δprob| 4.8e-7. Only 12 of the card's 21 "Potential output" lines match the oracle; the card calls them potential
  results, and the outside check shows the difference is not the environment.

## 5. Three things a bit-exact Swift host needed

- **Python's `\w`, `\s` and `str.lower()`.** `gliner2` splits words with Python's `re`, where `\w` includes digits
  such as `²` and `\s` includes `\v`, `\x1c`–`\x1f` and `\x85`. ICU's `\w` splits `mm²` into `mm` and `²`. With the ICU
  class, 3 of the 454 fixture texts get different token ids; with `[\p{L}\p{N}_]` and Python's whitespace set, all
  454 match. Python lowercases a word-final `Σ` to `ς`; Swift's `lowercased()` gives `σ`. The shipped GLiNER2-PII host
  (`InformationExtractor`) uses the ICU `\w` and splits such text differently from `gliner2`.
- **`JSONDecoder`, not `JSONSerialization`.** `JSONSerialization` does not round every decimal to the nearest double:
  372 of 2,432 numbers read from the Python gate record came back off by an ulp. `JSONDecoder` reads all of them exactly.
- **NumPy's summation order.** The softmax denominator is `numpy.add.reduce`, a pairwise sum: a plain loop below 8
  elements, eight accumulators up to 128, halves above that. The Swift softmax sums in that order. Applied to the
  Python engine's logits, the Swift decision rule then gives the same decisions and bit-equal metrics on every task.

With these, CoreAIKit's `TextClassifier` produces `gliner2`'s token ids for all 454 fixture texts, and its Mac GPU
logits equal the Python engine's bit for bit (5,749 of 5,749).

## 6. Measure the phone after a rest

The same bundle and the same inputs ran at 92.9 ms per call (S = 512) on a phone that had rested, and at 147.6 ms
after about 50 s of continuous calls, with the thermal state at fair. At S = 256, a run that started at fair measured
84.4 ms against 38.1 ms. After a 300 s wait the phone was back at 91.7 ms while the thermal state still read fair, so
the label does not follow the slowdown one to one. The cause is not isolated.

What worked: rest the phone for 5 minutes or more, run the bench right after loading and before the long case loop
(`DECIDE_BENCH_FIRST=1` in `apps/DecideGate`), and record the thermal state before and after it. Waiting for the
label to return to nominal (`DECIDE_WAIT_NOMINAL`) can run to its cap while the phone is already fast again.
