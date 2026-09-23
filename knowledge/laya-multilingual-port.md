# Porting laya multilingual (mmBERT-base + a typed decision head): the encoder reads the whole question once

Written 2026-09-23 from the port of `convaiinnovations/laya` (subfolder `multilingual`, revision
`1c5edc17…`, 321,908,998 parameters, Apache-2.0) to a static Core AI graph for macOS 27 and iOS 27.
Card and numbers: [`models/laya-multilingual/`](../models/laya-multilingual/README.md); scripts:
[`conversion/laya/`](../conversion/laya/README.md). The ModernBERT lessons of the Granite embedder port
([`granite-embedding-97m-port.md`](granite-embedding-97m-port.md)) all apply — the same 129-key sliding
window, the same missing attention norm on layer 0 — and this note is what a decision head adds.

## What the checkpoint is

A decision model, not a generator: one forward pass over
`[CLS] <type> question: <instructions> [SEP] [MASK] option [MASK] option … [SEP] <state> [SEP]`
gives every position a logit; the options are read at their mask markers and softmaxed within the
question. mmBERT-base (22 layers, hidden 768, GELU-gated MLP 1152, RoPE θ 160,000 for both the global
layers 0, 3, …, 21 and the sliding ones, 256k Gemma-style vocabulary) is followed by a type embedding
(choice / score / noul), two norm-first `nn.TransformerEncoderLayer` blocks with a key-padding mask, a
scorer (LayerNorm → Linear → GELU → Linear(768, 1)) and an act/escalate head (Linear(772, 256) → GELU →
Linear(256, 2)) over the [CLS] state and four features of the raw option softmax. The publisher's package
(laya 0.3.4) is the host recipe: `build_sequence` for the row, `Agent.predict` for the readout. The graph
is exported as one bundle with two functions, `main` (encoder, type embedding, head, scorer, the [CLS]
state) and `act` (the act head, fp32); marker gathering, the act features and the temperature stay on
the host.

## The oracle reproduced the frozen fixture bit for bit

The publisher's own model (transformers 5.17, SDPA attention, CPU fp32) re-ran the LiteRT lane's 201
rows × 2 windows: `build_sequence` gives the same ids and markers 402/402, and the batched
`Agent.predict` gives the same answer dictionaries and the same raw marker and act logits with zero
difference. Each row alone, right-padded to its window (batch 1), sits at max |Δp| 2.4e-6 from the frozen
answers and 3.8e-6 marker logits from the LiteRT capture of the same model padded to 512. That batch-1 run
is the tensor reference every later stage gates against. Two facts about the publisher's code paths that
a port must reproduce or account for: SDPA returns exactly zero for a fully masked query row (a pad
position more than 64 past the last real token; 125/125 rows at 256, 201/201 at 512, torch 2.12.1) while
row n+63 is non-zero — the inclusive radius 64 confirmed on the official model itself; and the head
layers' fused eval fast path and the plain path are bit-identical on CPU fp32.

## A massive activation moves the per-layer bar

mmBERT carries a massive activation on the [CLS] position (dims 468/488/530/580/614) from layer 11:
|h| reaches 1.37e4 at S=256 and 1.40e4 at S=512, where one fp32 ulp is about 1e-3. The publisher's model
run with SDPA and run with eager attention — same weights, same library — differ by 8.8e-3 at layer 11
and 4.8e-2 at layers 15–17 over real positions, and by 5e-3 to 1.4e-2 in the head, while their marker
logits agree to 3.5e-5. So the publisher's own eager path does not meet an absolute 1e-4 per-layer bar
above layer 10, and neither can a reimplementation be held to it. Relative to each state's own magnitude
the two official paths differ by up to 1.2e-4 (final_norm and the head input at S=256; 1.7e-5 at S=512),
and the re-authored graph sits at 5.7e-5 (S=256) / 7.8e-6 (S=512) from the SDPA path, with marker logits
at 5.7e-5 and act relative 5.6e-6. The gate that ships is two-tier: absolute 1e-4 for the embeddings and
layers 0–9 (max measured 5.0e-5), relative 2e-4 — twice the reference's own spread — from layer 10
through the head. The negative controls still fail where they should: all-global 4.2 and window-63 1.24
at layer 1, all-local 11.1 and ignore-padding 133 at layer 0 (four to six orders over the absolute bar; at
the relative layers the smallest mutation signal is 2.9e-3, 14× the bar); no-type-embedding is invisible
to the encoder states by construction and is caught by the marker (2.84) and answer (|Δp| 0.099) gates.

## fp16 compute does not ship; fp16 storage does

The fp16 recipe (weights and activations in fp16, RoPE application and softmax in fp32) misses the answer
bar in torch at S=256 — argmax 81/81 but max |Δp| 3.45e-3 on 32 of 201 rows, marker 0.10, act relative
2.4e-3 — and keeping the residual stream and LayerNorm in fp32 too does not help (|Δp| 4.9e-3): the error
is the fp16 matmul activations themselves, amplified at the [CLS] massive activation (final-norm relative
error 7.5e-2). On the Core AI runtime it is worse still (CPU |Δp| 4.4e-2 with one argmax flip, GPU 5.8e-3,
Neural Engine 1.85e-2 with one flip). `wfp16` — fp16 weight storage with fp32 compute — is exact instead:
the checkpoint is stored in F16, so the values are the same, and coreai-torch 0.4.1 keeps a constant in
fp16 when the graph casts it to fp32 before use (bundle 645 MB against 1.29 GB, cpu_only outputs
bit-identical to the fp32 bundle on all 402 rows).

## Where it runs: the GPU, at fp32 precision

Mac M4 Max (26A428), GPU solo, one whole question (main + host + act): wfp16 12.1 / 19.5 ms warm median
at S=256 / 512 (load 570 / 589 ms), fp32 10.4 / 17.2 ms (load 853 / 982 ms); both at max |Δp| 4.5e-6,
argmax 81/81, repeat drift 0. The Mac GPU computes these graphs at fp32 precision (GPU-vs-CPU marker
logits 7.8e-5), so fp16 weight storage costs nothing in accuracy here. cpu_only (contended, reference
only): 45 / 86 ms. Through CoreAIKit on the same GPU: 11.5 ms per decision at S=256 with the state shared,
and the kit's rows are token- and marker-identical to the publisher's 201 rows at both windows.

With a Neural Engine preference the fp32 bundle returns the GPU's results bit for bit, the wfp16 bundle
returns results that change from run to run (repeat drift 90–212 on the marker logits, |Δp| up to 0.34,
one argmax flip on a repeat; reproduced through the kit: 168–183 of 201 rows within the bar over four
runs), and the fp16 recipe is deterministic but outside the bar at 42.9 / 67.7 ms. `coreai.runtime`
reports no placement, so these readings are inferred from the numbers. Compiled for h18p with
`--preferred-compute neural-engine`, the wfp16 graph yields one Neural Engine region (IR 5.7 KB) with the
rest of the graph in the MPSGraph package, while the fp16-compute recipe yields 47 regions (IR 1.1–1.3 MB):
the Neural Engine takes the fp16-compute graph and not the fp32-compute one, and the fp16-compute graph
is the one that misses the bar. The shipped bundles run on the GPU; the iPhone folder is the GPU compile
and is named for it (`ios-h18p/`); the kit refuses a Neural Engine preference for encoder bundles.
iPhone: not yet measured.

## The host contract, and what the Swift port got wrong first

`build_sequence` (laya/common.py) is copied step for step; the traps are in the language, not the
recipe. Swift String's `==` / `hasPrefix` are grapheme- and canonical-equivalence based where Python is
code-point literal: a wire-form option `id: description` whose description opens with a combining mark
rendered `k: k: …` until the comparison moved to `unicodeScalars` (10 of 3,000 differential cases).
swift-transformers' Metaspace tokenizer emits a lone "▁" for the empty string where HF emits nothing (an
empty state is legal). With those two fixed, the kit's rows match the publisher's 201 rows at both windows
token for token, and its readout matches the publisher's model on SemIf authored144 (144/144 argmax at
the same temperature, balanced accuracy 0.6114 — the model's own figure). Two costs the kit measured on
the Mac: the 256k-entry tokenizer is about 150 MB resident and about 1.0 s of the 1.1 s load, and
tokenizing a question's head and options is 2.3 ms per call, which a cache of recent questions removes
(a decision's wall clock 14.2 → 11.7 ms).
