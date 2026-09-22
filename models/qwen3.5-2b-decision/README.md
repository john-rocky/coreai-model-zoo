Core AI is Apple's on-device ML runtime in iOS 27 / macOS 27 and the successor to Core ML: PyTorch models are exported with Apple's `coreai-torch` (LLMs: `coreai.llm.export`) into `.aimodel` bundles that run on the GPU or the Neural Engine, e.g. Qwen3-8B 4-bit decodes at 94 tok/s on an M4 Max GPU, MLX 90 under the same protocol ([apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench), macOS 27 beta 26A5353q, 2026-06-11).

# Qwen3.5-2B-Decision — Core AI

[🤗 mlboydaisuke/Qwen3.5-2B-Decision-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3.5-2B-Decision-CoreAI) · Apache-2.0 · source [chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16/tree/69a91b6778df7d11b9e13e0f7d4c6a5aa6ad1bc5) · revision `69a91b6778df7d11b9e13e0f7d4c6a5aa6ad1bc5` · base Qwen/Qwen3.5-2B-Base

An English typed-decision model: give it a state, a question and declared options to obtain a probability for each option. The source, Jev-Style-Qwen3.5-2B-Decision, merges a LoRA into Qwen3.5-2B-Base and folds its calibration temperature into the final RMSNorm. It retains the tied vocabulary LM head. This port exports the 24-layer text tower, with 18 linear-attention and six full-attention layers, as an S=1 Core AI LanguageBundle with a 4,096-token context.

The [author's pinned card](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16/blob/69a91b6778df7d11b9e13e0f7d4c6a5aa6ad1bc5/README.md) reports **82.3% accuracy on five decision tasks** and **ECE 0.017 on 1,500 held-out examples**. These are the author's measurements, not re-measured by this conversion. Temperature is already in the published weights; readout uses **T = 1**.

## Decision contract

Use the author's exact plain-text form, ending at `Answer:`:

```text
You are a decision function. Read the state, then answer the question by choosing exactly one option.

[State]
{state}

[Question]
{question}

[Options]
A. {first option}
B. {second option}

Answer:
```

Encode with `add_special_tokens=False`, without a BOS or chat template. Every question starts from fresh state. The answer slot is `len(ids)-1`. Gather the final-position logits at the single-token IDs for **` A`, ` B`, …** with their leading spaces, up to the listed option count, and apply FP32 softmax over those logits only. The author computes the same scores as `embed_tokens.weight[label_ids] @ last_hidden_state` after the final norm. The tied LM head makes the bundle's `logits[label_ids]` equivalent.

Choice returns the option distribution. Bool uses options `["yes", "no"]` and returns p(yes). Score lists levels in order and returns Σ i·p(i), indexed from zero. The source permits up to 26 letters; this fixture covers Choice with 2–16 options and Score with 2–10 levels. The pass-through source chat template is preserved but unused on this path. The decision API reads probabilities; the one-token Swift check below is separate from the probability readout.

The [fixture](fixtures-qwen3.5-2b-decision.json) contains **22 requests / 58 rows**: 34 Choice, 12 Bool and 12 Score, including twelve Choice rows with 10–16 options, eight with descriptions longer than five words, and two ten-level scores. Two Spanish rows are `evidence_only`. Kit-sized prompts are 98–290 tokens; two `zoo_only` prompts contain 1,743/1,742 tokens. All states satisfy 30–500 tokens. Six actual author `decide` / `decide_bool` / `decide_score` calls on distinct requests exactly equal the row readouts.

## Inverse MLX conversion and its proof

This checkpoint requires more than a key-prefix change. The [inverse converter](../../conversion/mlx_to_hf_qwen3_5.py) maps `language_model.model.*` to `model.language_model.*`, transposes 18 `linear_attn.conv1d.weight` tensors from `[6144,4,1]` to `[6144,1,4]`, and subtracts one from 61 direct-scale RMSNorm weights: input/post-attention norms, q/k norms, and the final norm. The frozen Torch graph multiplies by `1 + weight`; the 18 gated `linear_attn.norm` scales already use the direct convention and stay unchanged. Transposes and offsets are computed and stored in FP32 to avoid a second BF16 rounding (spacing at 1.0 is 2⁻⁷). Other tensors retain their stored dtype, including the 18 FP32 `A_log` tensors. The tied head remains absent from the converted checkpoint.

The converter writes a flat `qwen3_5_text` config with tied embeddings, avoiding the observed transformers 4.57.6 serialization failure on the source's nested `vision_config: null`. The frozen loader and graph remain unchanged. The [self-contained exporter](../../conversion/export_qwen3_5_decision_mlx_decode_pipelined.py) downloads the pinned MLX source, converts a private local snapshot and uses the existing Qwen3.5 export/quantization path.

Oracle A is the author's unchanged `jev_style_mlx.py` with mlx-lm 0.31.3 / MLX 0.32.2, BF16 on Mac GPU. Oracle B loads the actual converted artifact into transformers 5.17.0's native text-only Qwen3.5 class, strictly with tied head, FP32 on CPU. A and B agree on **58/58 choices**, including **57/57** at B margin ≥ 0.02; max |Δp| is **0.011958122**, mean of row means **0.001554969**. This is the inverse-conversion proof, not a new task-accuracy or calibration evaluation.

## Measured (Apple M4 Max GPU, macOS 27.0 26A428, 2026-09-23)

| Check on all 58 rows | fp16 reference | int8hu ship |
| --- | ---: | ---: |
| Option argmax = FP32 B | 58/58 | 58/58 |
| Agreement at B margin ≥ 0.02 | 57/57 | 57/57 |
| Max \|Δp\| vs B | 0.005788386 | 0.007585824 |
| Mean of row mean \|Δp\| vs B | 0.000343644 | 0.000767530 |
| Option argmax = author MLX A | 58/58 | 58/58 |
| Max \|Δp\| vs A | 0.012892783 | 0.013182223 |
| Mean of row mean \|Δp\| vs A | 0.001582145 | 0.001594868 |
| Full-vocabulary argmax is a listed letter | 58/58 | 58/58 |
| Swift pipelined first text = Python full-vocabulary argmax text | 58/58 | 58/58 |
| Swift sequential first text = Python full-vocabulary argmax text | 58/58 | 58/58 |
| Fresh-state reset logits, bit-identical | yes | yes |

The accepted round-2 measurements use all 58 rows, including evidence-only and long rows. The zoo-layout scripts reproduce the argmax counts and max probability errors to 1e-6; their transcripts are [fp16 readout](gate-qwen3.5-2b-decision-readout-fp16.json), [int8hu readout](gate-qwen3.5-2b-decision-readout-int8hu.json), [fp16 engines](gate-qwen3.5-2b-decision-engine-fp16.json) and [int8hu engines](gate-qwen3.5-2b-decision-engine-int8hu.json). Both references remain in the fixture. The one near tie, `package_log__3_score`, has B margin **0.005873829** and also agrees. All logits and probabilities are finite.

Readout uses AOT **h16c GPU**, `SpecializationOptions.default()`, fresh zero states and S=1 steps through the entire prompt. Final logits are FP16 `[1,1,248320]`; label selection and softmax are FP32 at T=1. Each process has at most 15 evaluations including resets. Complete first-row logits are bit-identical within each process and in the final fresh process. Engine checks use both Release variants with raw tokens, one greedy token and warmup off; expected text is the same bundle's full-vocabulary Python argmax, preserving its leading space. This gives **232/232** exact comparisons across both bundles and engines.

### S=1 throughput proxy

Release `llm-benchmark`, int8hu only, `-p 128 -g 256 -n 3`, two alternating launches per engine / six trials each, `COREAI_CHUNK_THRESHOLD=1`, Xcode 27.0 (27A266a). The benchmark adds only the engine-variant option on a copy of fork tag 0.2.4-zoo (`f7a75ec`). **Contended: false** under the recorded before/after inference-job snapshots and untagged GPU lock. Swift compiler/build and Storage CPU activity were visible in the starting snapshot, with Storage/background desktop activity at the end; this does not claim that the whole system was idle. A prior contended attempt is retained separately; the table uses the completed quiet-window repeat.

| Engine | Prefill tok/s, median (min–max) | Decode tok/s, median (min–max) | Load seconds, two launches |
| --- | ---: | ---: | ---: |
| coreai-pipelined | 164.125 (161.425–165.146) | 159.555 (127.061–160.141) | 3.818 / 0.428 |
| coreai-sequential | 130.030 (129.453–130.781) | 127.278 (126.609–127.650) | 0.456 / 0.431 |

A decision costs one S=1 prefill of the **whole question row**, so these are **prefill-rate proxies**, not per-decision latencies. Synthetic decode does not represent text generation by this decision API. [Per-trial rates, launch load times and process snapshots](llm-benchmark.json).

## Through the kit

**Measured through coreai-kit** by the supervisor, using `Decision.Format.decisionFunction`, the sequential engine and the kit's own rendering of the author's plain-text prompt. No kit measurement was re-derived by this run.

`decide-cli parity` on all 58 fixture rows matched tokens **58/58**, label slots **58/58** and option argmax **58/58 on both bundles**. Bool maps to the kit's noul; Score maps to its ordered-level score. Against FP32 B, int8hu max |Δp| / mean were **0.0079 / 0.0017**; fp16 **0.0058 / 0.0008**. Median int8hu time was **986 ms per fixture question**; the two 1,500–2,000-token rows took up to **13.6 s**.

SemIf's authored144 uses 144 English rows, three options each, SemIf's gold labels and its unchanged `benchmarks/evaluate.py`. Rendering each row in the author's Choice form gave **114/144 raw**, **0.798 mean family balanced accuracy**, and **872 ms median per decision** on int8hu. For scale on the same rows/evaluator, the kit README reports Qwen3.5-4B int8 zero-shot **0.821**, MiniCPM5-2B int8 **0.681**, OpenThai **0.725** and APUS-OpenJev-v1-4B **0.906**. These are numbers on that evaluator, not a ranking.

On the author's chipmaker-news example, the kit's int8hu probabilities were **Business 0.684 / Science-Technology 0.309 / World 0.005 / Sports 0.002**; the author's MLX BF16 card quotes **0.70 / 0.29 / 0.005 / 0.002**. A three-question request on that sentence (Choice, yes/no, five-level Score) took **817 / 633 / 766 ms** for its **82 / 73 / 88-token** rows, with **0 tokens reused**. This recurrent hybrid re-prefills each row, about **10 ms per token** on the sequential engine. [Supervisor-supplied record](measurements-coreai-kit.json).

## Bundle

Both bundles are staged under `gpu-pipelined/` in [mlboydaisuke/Qwen3.5-2B-Decision-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3.5-2B-Decision-CoreAI). int8hu ships; fp16 is its reference.

| Bundle | Staged bytes | main.mlirb bytes | main.mlirb SHA256 |
| --- | ---: | ---: | --- |
| `gpu-pipelined/qwen3_5_2b_decision_decode_int8hu_block32_sym` | 3,046,578,446 | 3,017,991,690 | `2daf0c1522ab65fc6c124dc6d8ef80cca3c9341ae8a435636b27aeddc8aa4ed3` |
| `gpu-pipelined/qwen3_5_2b_decision_decode_fp16` | 3,793,065,369 | 3,764,478,641 | `d2c2bac89671def4253ce545dd2502bc115e0b5656305fbcedad9c38c5221558` |

Each LanguageBundle has `.aimodel`, `metadata.json` and `tokenizer/`. The staged IR and tokenizer files are the accepted originals. Root metadata is copied and supplemented with the pinned `source.hf_revision` and the decision readout block; original exports remain unchanged. Vocabulary is **248320**, compiled context **4096**, and the unchanged helper preserves `compression: null`; precision is specified by the recipe. The new exporter emits the same metadata contract.

`int8hu --head-sym` uses block-32 int8 linears and a symmetric block-32 vocabulary head cloned from the tied embedding. Embeddings, conv1d and norms remain fp16 in the exported graph. Use the [extra-states patch](../../apps/coreai-pipelined-extra-states.patch) for KV/conv/recurrent states and set `COREAI_CHUNK_THRESHOLD=1` before engine creation. Both engines were checked with Release tools from fork tag **0.2.4-zoo**. The port is measured on Mac GPU; no phone result is claimed.

## Reproduce

Use the frozen Core AI export environment: coreai-core 1.0.0b2, coreai-torch 0.4.1, coreai-opt 0.2.1, torch 2.9.0, transformers 4.57.6 and fork `coreai_models` at `397b337`. HF_HOME and all output/cache paths should be local to the conversion workspace; XET is disabled.

```sh
export HF_HUB_DISABLE_XET=1
export COREAI_CHUNK_THRESHOLD=1
python conversion/export_qwen3_5_decision_mlx_decode_pipelined.py int8hu --head-sym --out-dir exports
python conversion/export_qwen3_5_decision_mlx_decode_pipelined.py fp16 --out-dir exports
```

The [recipe](recipe.toml) pins the source and both bundle names. The standalone inverse converter accepts `--snapshot <pinned MLX snapshot> --out <converted snapshot>`. The [letter gate instructions](../../conversion/letter/README.md) give the pinned two-reference fixture builder and the plain-text / space-prefixed-label flags for both runtime gates. The original fixture is retained unchanged. Re-export IR size/SHA256 differences are recorded because serialization byte identity is not required; metadata comparison separately records the newly required revision/decision fields and ignores only `compilation.date` when comparing the full shipping contract.

## License

The source repository declares **apache-2.0 in its card** and carries **no LICENSE file**. The staged LICENSE is the canonical [Apache License 2.0 text](https://www.apache.org/licenses/LICENSE-2.0.txt), 11,358 bytes, SHA256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`. The author's card separately notes an AG News research/non-commercial distribution condition in its training-data description; this port preserves that declaration without making an additional licensing determination. Source configuration is staged verbatim for provenance.
