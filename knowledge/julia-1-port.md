# Porting Julia-1 (mmBERT-small + a typed decision head): the laya path at half the width

Written 2026-09-30 from the port of `SupersonicLabs/Julia-1` (revision `a85b1273…`, 144,292,870 tensor
elements, all F32, Apache-2.0) to a static Core AI graph for macOS 27. Card and numbers:
[`models/julia-1/`](../models/julia-1/README.md); scripts: [`conversion/julia/`](../conversion/julia/README.md).
Julia-1 is laya's `DecisionModel` on mmBERT-small (the publisher's `from_pretrained` even reads laya's
`rl_agent_config.json`), so [`laya-multilingual-port.md`](laya-multilingual-port.md) applies throughout; this note
is what differed.

## What the checkpoint is, and what the graph leaves out

mmBERT-small: 22 ModernBERT layers, hidden 384, 6 heads × 64, GLU MLP 1152, global attention on layers 0, 3, …, 21,
an inclusive sliding radius of 64 elsewhere, RoPE θ 160,000 for both kinds, a 256k Gemma-style vocabulary. Then
`h + type_emb[qtype]`, two norm-first `nn.TransformerEncoderLayer(384, 6, 1536, relu)` blocks with a key-padding
mask, and a scorer (LayerNorm → Linear → GELU → Linear(384, 1)) read at the option markers. The graph is one
function, `main`: ids, mask and the one-hot type in, the scorer at every position out.

Two things in the checkpoint stay out of the graph. `act_head` (Linear(388, 256) → GELU → Linear(256, 2)) has
weights, but the publisher's inference API never calls it (`forward(..., return_actions=False)` in every engine
path). The `temperature` buffer is [1, 1, 1] and nothing reads it; `inference-policy.json` says
`calibration: null`. So the answer is the softmax of the raw marker logits at T = 1, and a laya host's
calibration step must not be carried over.

## The host is where Julia differs from laya

The row is laya's (`[CLS 2] <type> question: … [SEP 1] [MASK 4] option … [SEP 1] state [SEP 1]`, options cut at
48 tokens, the same head squeeze), but the option text is not. Julia's named questions feed the criteria
descriptions as given: a choice's mapping values (the answer is the caller's id), a score's rubric in order, a
noul's `[criteria.false, criteria.true]`, or the literal `false` / `true` when a noul has no criteria. laya
prefixes `label: `, `level i: ` and `false: ` / `true: ` and has default noul wording, so a laya host builds
different rows for Julia and gets different answers. The publisher's typed-decisions reproduction renders the
dataset's criteria the same way, and its README shows what the difference is worth: replacing only the noul
descriptions with the literal words drops noul from 483/600 to 391/600.

Julia's builder is also strict by default: a state, an option or a question that would be cut raises instead
(laya cuts the state). And the head budget depends on the window: the publisher's protocol is
`max_length 1024, head_length 512`, and its builder requires `head_length + 4 < max_length`, so a 512 window
takes `head_length 256` (the `load_model` default). Under strict encoding nothing is ever cut, so the budget only
decides which rows are accepted — every row that fits 512 gets the same ids at 256 as at 512 (2,065 of 2,065).

`_julia_host.py` (the `tokenizers` library on the checkpoint's `tokenizer.json`) rebuilds all 2,155 oracle rows
from text with identical ids and markers — under tokenizers 0.23.0rc0, where the publisher's run used 0.22.2 —
and reproduces the publisher's 2,000 `predict(state=..., questions=...)` answer dictionaries from the same logits.
The tokenizer file is byte-identical to laya's (sha256 `609d8f4c…`), so the Swift tokenizer lessons of the laya
note carry over unchanged.

## The oracle reproduced the publisher's CPU result, and a second lane's oracle bit for bit

The publisher's `scripts/reproduce_typed.py`, run unchanged on an M4 Max (torch 2.14.0, transformers 5.0.0, CPU
fp32, 4 threads), gives its published 426/600 choice, 542/800 score, 483/600 noul in 71 s. `oracle_julia.py` takes
the raw logits of the same engine one question per forward: the same three numbers, and all 2,000 answers equal to
that run's. The publisher's 100 WebGPU parity requests match the PyTorch logits it recorded to 1.03e-4 with
100/100 argmax (it ran them in padded batches of four). The LiteRT lane's independent oracle of the same pins
gives the same ids and the same logits on all 2,000 questions, bit for bit.

Two facts about transformers 5.0 a port runs into: ModernBERT refuses `set_attn_implementation("eager")` with a
warning and keeps SDPA, so the eager comparison needs `encoder.config._attn_implementation = "eager"` set directly
(the attention function is looked up from the config at every forward); and SDPA no longer returns exactly zero
for a fully masked sliding row (a pad position more than 64 past the last real token) under torch 2.14, where
laya observed zeros under torch 2.12. Pad positions are never read, so neither changes an answer.

## The layer bar, re-measured: the final LayerNorm under a massive activation

The largest value in the residual stream jumps from 25 at layer 10 to about 3.2e3 at layer 11 and 4.9e3 from
layer 18 (S=1024; laya's mmBERT-base reached 1.4e4 from layer 11). The publisher's own two attention paths
differ, relative to each state's magnitude, by up to 1.3e-4 (S=512) and 2.9e-4 (S=1024) at the final norm and
the head — above laya's 2e-4 bar, which was twice laya's own spread. The same rule re-measured here gives
2 × 2.87e-4 → 5e-4.

The graph is exact: under the publisher's torch 2.14.0 it equals the publisher's eager path to 1.3e-7 relative at
every state and 1.9e-6 on the marker logits (`gate_julia_exactness.py`). Under the zoo's torch 2.9.0 the encoder
layers stay as close to SDPA as eager is (2.1e-5 at layer 21), and the final LayerNorm adds the rest: 3.4e-4
relative at S=512 and 2.1e-4 at S=1024, 7.95e-4 on the marker logits: a different torch build normalizes a
vector holding values near 4.9e3 differently. The negative controls stay far outside the bar: all-global 3.6,
all-local 4.2, a radius of 63 1.45 and ignoring the padding 29 at layers 0–1 (S=512); dropping the type embedding
shows at the head input and moves the marker logits by 0.81.

## F32 weights: fp16 storage is not free here

laya's checkpoint is F16, so storing its weights in fp16 changed nothing. Julia's is F32, and rounding every weight
to fp16 (fp32 compute) keeps every argmax (2,120/2,120 at S=1024, the three typed-decisions totals unchanged) but
moves the probabilities: max |Δp| 0.023, over 1e-3 on 325 of 2,120 rows, a score's expected index by up to 0.029
and a noul's p[true] by up to 0.023. The embedding table alone differs by 4.4e-3 after its LayerNorm. The bundle
therefore ships fp32 (the published weights as they are).

## Where it runs: the Mac GPU, at fp32

Apple M4 Max, macOS 27.0 (26A428), coreai-build 3600.83.1, the JIT `.aimodel`, the GPU requested explicitly and
the machine-wide GPU lock held, every row of the window through the bundle: S=1024 argmax 2,120/2,120, max |Δp|
8.6e-5, **16.0 ms** per question warm (p10 15.9, p90 16.2; load 641 ms); S=512 argmax 2,100/2,100, max |Δp| 8.0e-5,
**8.3 ms** (load 702 ms). A question here is NumPy in, the graph, the marker gather — tokenization not included.
The typed-decisions totals re-scored from the bundle's own argmax are the publisher's 426 / 542 / 483 at S=1024,
on the GPU and on `cpu_only` alike. The Python host end to end (text in, the publisher's answer dictionaries
out) gives the publisher's answer on all 2,000 named questions at max |Δp| 8.6e-5, 32.9 s for the 2,000
including tokenization. The iPhone was not measured.

The Neural Engine does not take this graph. With a Neural Engine preference the Mac returns the GPU's results
bit for bit at the same speed (844 rows across both windows), and `coreai-build compile --preferred-compute
neural-engine` gives no Neural Engine region for macOS (h16c) or the iPhone 18 Pro (h19p): the only delegate is
MPSGraph and the compute type is Float32. A Neural Engine variant needs fp16 compute, which laya's recipe showed
misses a 1e-3 probability bar on that model; for Julia it is not built.

## The Swift host, checked without a Swift tokenizer

`swift/JuliaDecisions.swift` is the NumPy host in Swift over the system CoreAI framework, with the tokenizer
passed in as a closure. Its parity run replaces the tokenizer with a table of the publisher's token ids per text
piece (`swift_pieces_julia.py`), so what it checks is everything after the tokenizer: on the 200 strict fixture
rows the ids and markers equal the publisher's 200/200 and the bundle's answers agree with the publisher's logits
(argmax 200/200, max |Δp| 4.3e-5 on the GPU). One thing the SDK taught: `NDArrayDescriptor` has no public
initializer, so an input array is made from the function's declared input descriptor
(`descriptor.inputDescriptor(of:)`), as CoreAIKit's `GraphModel` does.
