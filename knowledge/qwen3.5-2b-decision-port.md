# Qwen3.5-2B-Decision: invert the MLX checkpoint before the Core AI export

Source: [chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16/tree/69a91b6778df7d11b9e13e0f7d4c6a5aa6ad1bc5), revision `69a91b6778df7d11b9e13e0f7d4c6a5aa6ad1bc5`. The model is Qwen/Qwen3.5-2B-Base with a merged LoRA and calibration temperature folded into its final RMSNorm. It is an English typed-decision model with a tied vocabulary head, exported here as a 4,096-token S=1 text-tower LanguageBundle. The author's ECE 0.017 on 1,500 held-out examples and 82.3% accuracy on five tasks are source claims, not measurements made by this port.

## Finding: the MLX layout is more than a prefix

The initial prefix-only premise was incomplete. The checkpoint contains 320 tensors under `language_model.model.*`, with no visual tower, no separate head and no MTP. Its convolution and normalization conventions follow mlx-lm's Qwen3.5 conversion. The inverse conversion is therefore explicit and separate from the frozen loader:

| Tensor group | Published MLX form | Converted HF form |
| --- | --- | --- |
| All keys | `language_model.model.X` | `model.language_model.X` |
| 18 `layers.*.linear_attn.conv1d.weight` tensors | `[6144,4,1]` | `[6144,1,4]`, transpose dimensions 1/2 |
| 48 `layers.*.{input_layernorm,post_attention_layernorm}.weight` tensors | Direct RMSNorm scale | Scale minus 1 |
| 12 full-attention `layers.*.self_attn.{q_norm,k_norm}.weight` tensors | Direct RMSNorm scale | Scale minus 1 |
| Final `norm.weight` | Direct calibrated scale | Scale minus 1 |
| 18 gated `layers.*.linear_attn.norm.weight` tensors | Direct gated scale | Unchanged |
| 18 `layers.*.linear_attn.A_log` tensors | FP32 | FP32 unchanged |

The frozen Torch graph's ordinary RMSNorm uses `1 + weight`; its gated linear-attention norm already takes a direct scale. Applying an offset to that gated group would change the model. The 18 transposes and 61 offsets are computed **and stored in FP32**. BF16 spacing at 1.0 is 2⁻⁷: the source already carries that rounding, and storing the inverse in BF16 could add a second rounding. The remaining 223 tensors retain BF16. The converted file contains 97 FP32 tensors and keeps the tied head absent.

[`mlx_to_hf_qwen3_5.py`](../conversion/mlx_to_hf_qwen3_5.py) exposes `--snapshot --out`, records every key/shape/dtype, and names each transformed tensor. It writes the original `text_config` as a flat `qwen3_5_text` config with `tie_word_embeddings: true`. This avoids the observed transformers 4.57.6 serialization failure on nested `vision_config: null`; the source config and frozen loader are not changed. The source's full config is retained verbatim in Hub staging.

The converted checkpoint is **3,764,783,440 bytes**, SHA256 `09b7e5a8b17abbc66f42333017ee66addfbcc3aff78461d33c05443e914def24`. The flat config SHA256 is `1a318c082474ecad1e796f2b3efea6d79ac6991f1c9d960522cf057b71d5cd90`. The zoo exporter re-ran the converter and reproduced both hashes exactly. Core AI's frozen fork is `397b337e234474a191c0bd96ac9ef71c4f808a3d`; its loader was not patched.

## Numerical proof with two independent references

The [fixture](../models/qwen3.5-2b-decision/fixtures-qwen3.5-2b-decision.json) contains 22 deterministic requests and 58 rows: 34 Choice, 12 Bool and 12 Score, with two Spanish evidence-only rows and two long zoo-only rows of 1,743/1,742 prompt tokens. The source author path is oracle A: unchanged `jev_style_mlx.py`, mlx-lm 0.31.3, MLX 0.32.2, published BF16 checkpoint on Mac GPU. Oracle B is the actual converted artifact loaded strictly into transformers 5.17.0's native text-only Qwen3.5 causal LM, with its head tied and all inference in FP32 on CPU. B applies no further layout or norm transformation.

A versus B agrees on **58/58** option argmaxes, including **57/57** rows with B top-two margin ≥ 0.02. Maximum absolute probability difference is **0.011958122253417969**; mean of row means is **0.0015549686951135543**. Six actual author `decide` / `decide_bool` / `decide_score` calls exactly equal the row readouts. The sole near tie, `package_log__3_score`, has B margin **0.005873829126358032** and also agrees. This establishes the inverse conversion under the measured BF16-GPU/FP32-CPU difference; it does not establish new task accuracy or calibration quality.

[`oracle_decision_function.py`](../conversion/letter/oracle_decision_function.py) embeds the same 22 requests and builds both references. The original accepted fixture is copied unchanged into the family directory. The [letter instructions](../conversion/letter/README.md) document its pinned uv environment and both gate commands.

## Plain-text probability readout

The author renders `You are a decision function. Read the state, then answer the question by choosing exactly one option.`, followed by `[State]`, `[Question]`, `[Options]` with `A. ...` lines, and a final `Answer:`. Use `add_special_tokens=False`, no BOS and no chat template. The label IDs are the single-token encodings of **space plus letter**, ` A`, ` B`, …; the answer slot is `len(ids)-1`.

The author's `embed_tokens.weight[label_ids] @ last_hidden_state` includes the final calibrated norm. Since the head is tied, gathering the same label IDs from final LM logits gives the same readout. Apply softmax **over the listed letters only**, at **T=1**. Choice returns that distribution; Bool uses `["yes", "no"]` and returns p(yes); Score returns the expected zero-based level index. The API does not generate prose.

The generic letter gates add `--label-style space-prefixed --prompt-format decision-function` for this model; their defaults retain APUS behavior. Each row starts with fresh zero states. AOT h16c GPU and `SpecializationOptions.default()` are required on the measured macOS build. S=1 steps produce FP16 `[1,1,248320]` logits. Processes are split at at most 15 evaluations, including resets, to bound the runtime's IOSurface retention. Both within-process and fresh-process first-row repeats are checked for full-vocabulary bit identity.

## Measured Core AI parity

Device: Apple M4 Max, 128 GiB RAM; macOS 27.0 (26A428), Xcode 27.0 (27A266a). Core AI export pins: coreai-core 1.0.0b2, coreai-torch 0.4.1, coreai-opt 0.2.1, torch 2.9.0, transformers 4.57.6. Release Swift tools use tag 0.2.4-zoo (`f7a75ec0f89fab451d277572afe8995b7ef768c1`).

| Bundle | Argmax vs B | At B margin ≥ 0.02 | Max absolute Δp vs B | Mean of row means vs B | Max absolute Δp vs A |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp16 | 58/58 | 57/57 | 0.00578838586807251 | 0.00034364434736209465 | 0.01289278268814087 |
| int8hu block32 symmetric head | 58/58 | 57/57 | 0.0075858235359191895 | 0.0007675302394331634 | 0.013182222843170166 |

Both bundles also agree with A on all 58 option argmaxes. All logits/probabilities are finite and all full-vocabulary argmaxes are listed letters. Each Swift engine's first-token text equals that bundle's Python full-vocabulary argmax text on 58/58 rows: **232/232** comparisons across both bundles and engines. Engines receive raw token IDs, one greedy token, warmup off, and `COREAI_CHUNK_THRESHOLD=1`. See the family directory's separate [fp16 readout](../models/qwen3.5-2b-decision/gate-qwen3.5-2b-decision-readout-fp16.json), [int8hu readout](../models/qwen3.5-2b-decision/gate-qwen3.5-2b-decision-readout-int8hu.json) and engine transcripts.

The self-contained exporter keeps the original Qwen3.5 graph and quantization path. `int8hu --head-sym` clones the tied embedding into a symmetric block-32 int8 head. New metadata carries `source.hf_revision` and `decision: {head: lm, readout: decision-function letters, temperature: 1, labels: space-prefixed A-Z}`. Original accepted bundles remain unchanged; shipping metadata is supplemented on separate copies.

The int8hu re-export produced main.mlirb **3,017,991,723 bytes**, SHA256 `547f2a0d47d3494b42581a755db0963a7fae8e20ee0eb290df0f086c5aebcf29`, compared with the original **3,017,991,690 bytes**, SHA256 `2daf0c1522ab65fc6c124dc6d8ef80cca3c9341ae8a435636b27aeddc8aa4ed3`. The 33-byte serialization difference is recorded; byte identity is not required. Metadata matches the shipping contract except `compilation.date`, and tokenizer files are identical.

## Through the kit

**Measured through coreai-kit**, as supplied by the supervisor on 2026-09-23 06:5x–07:0x JST: `Decision.Format.decisionFunction`, the sequential engine and the kit's rendering of the author's form. This conversion run did not re-measure the kit.

- All 58 fixture rows: tokens, label slots and option argmax each match 58/58 on both bundles. Against B, int8hu max/mean absolute Δp is **0.0079 / 0.0017**; fp16 **0.0058 / 0.0008**. Int8hu median is **986 ms per fixture question**, with long rows up to **13.6 s**.
- SemIf authored144: 144 English three-option rows, SemIf gold labels and unchanged `benchmarks/evaluate.py`; int8hu gives **114/144 raw**, **0.798 mean family balanced accuracy**, **872 ms median per decision**. On the same rows/evaluator, the kit README reports Qwen3.5-4B int8 zero-shot 0.821, MiniCPM5-2B int8 0.681, OpenThai 0.725 and APUS-OpenJev-v1-4B 0.906. These are contextual numbers, not a ranking.
- Author's chipmaker-news example: kit int8hu Business **0.684**, Science-Technology **0.309**, World **0.005**, Sports **0.002**; the author's MLX BF16 card quotes **0.70 / 0.29 / 0.005 / 0.002**.
- Three questions on that sentence (Choice, yes/no, five-level Score): **817 / 633 / 766 ms**, **82 / 73 / 88 tokens**, **0 tokens reused**. The recurrent hybrid re-prefills every question, about **10 ms per token** on the sequential engine.

See [measurements-coreai-kit.json](../models/qwen3.5-2b-decision/measurements-coreai-kit.json) for explicit provenance.

## Throughput limitation and licensing

The int8hu S=1 benchmark uses `-p 128 -g 256 -n 3`, two alternating launches per engine. Median prefill/decode rates are **164.125 / 159.555 tok/s pipelined** and **130.030 / 127.278 tok/s sequential**. **Contended: false** under the prescribed before/after inference-job snapshots and untagged lock; background Swift compiler/build and Storage CPU activity were visible, so this does not claim whole-system idleness. A first attempt, made before all correctness work, began quiet but ended with another Python workload visible; it remains archived as contended. The isolated repeat ran after correctness workers exited. The complete [benchmark record](../models/qwen3.5-2b-decision/llm-benchmark.json) retains trials, load times and process snapshots. S=1 prefill rate is a proxy, not a per-decision latency; the kit's latency measurements are separate.

The source card declares Apache-2.0 and the source repository has no LICENSE file. Hub staging includes the canonical Apache-2.0 text and the original config. No upload, shared catalog edit or phone test is part of this run.
