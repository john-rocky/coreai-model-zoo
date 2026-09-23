License: **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International).

Core AI is Apple's on-device ML runtime in macOS 27 / iOS 27 and the successor to Core ML. Apple's `coreai-torch` exports PyTorch graphs to `.aimodel` bundles. This port is measured on the **Mac GPU only**.

# OpenJev — Core AI

For a routing decision, a yes/no probability or an ordered score, provide a state, a question and the allowed options. **OpenJev**, source [openjev/openjev](https://huggingface.co/openjev/openjev/tree/5ec9e5fd2f80a6fff386779b1e5ac7e389971889), reads the bare letter probabilities at the first output position, then returns a typed decision. There is no generated answer to parse. This is the text-only Core AI port of the 27B model, staged for [mlboydaisuke/OpenJev-CoreAI](https://huggingface.co/mlboydaisuke/OpenJev-CoreAI).

The author's example makes the input and output concrete:

> Customer message: I was charged twice for my order last week and nobody has replied.

Using the author's unchanged helper on its **BF16** weights, “Which team should handle this?” with `billing / shipping / technical` returned **billing 0.9998**, shipping 0.0001 and technical 0.0001. “Is the customer angry?” returned calibrated **noul 0.6183** (before yes/no calibration, p_yes = 0.7073095709). These are the helper's answers on this Mac; the bundle's conversion comparison is below.

OpenJev is a fine-tune of **Qwen/Qwen3.8-27B**, base revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (Apache-2.0), with a `Qwen3_5ForConditionalGeneration` checkpoint and a 64-layer text tower. Its 48 GatedDeltaNet layers interleave with 16 full-attention layers. This export retains the vocabulary LM head, drops the vision tower and MTP weights using the shipped Qwen3.8-27B text recipe, and runs each input token as an S=1 decode step.

The [author's pinned card](https://huggingface.co/openjev/openjev/blob/5ec9e5fd2f80a6fff386779b1e5ac7e389971889/README.md) reports **84.0% accuracy on 10,000 text questions**, **1.4 percentage points behind** the hosted API; 6,922 were fresh for that run and 3,078 had been used during development. It reports **88.0% desktop next-action accuracy** on 2,000 screenshot steps and **87.4% web next-action accuracy** on 975 unseen-website steps, plus **2.3% answer flips** over 2,000 option-shuffle tests. These are the author's results, not task accuracy or calibration re-measured by this port; the screenshot result does not apply to this text-only bundle.

The author reports the following **single-H100, online-FP8** latencies: about **80 ms** for short text decisions, **176 ms median** for a desktop screenshot step, and about **210 ms median** for a web step with roughly 1,460 prompt tokens and 23 options. Its first-read/cached-new-question medians are **227/118, 477/162, 923/213, 1,433/246 and 1,758/257 ms** at about **1.1k, 4.1k, 8.1k, 12.2k and 15.1k prompt tokens**, respectively. These H100 numbers are the author's serving measurements, separate from this Mac conversion.

## Decision contract

Compose exactly this text, with one option line per supplied option in insertion order:

```text
State:
{state}

Question: {instructions}
Options:
[A] {first_key}: {first_description}
[B] {second_key}: {second_description}

Answer with the letter of the best option only.
```

Precisely, concatenate `"State:\n" + state + "\n\nQuestion: " + instructions + "\nOptions:\n"`, the option lines `f"[{L}] {key}: {desc}"` joined by `"\n"`, and `"\n\nAnswer with the letter of the best option only."`. Null descriptions become an empty string, so their option line ends in `": "`. Letters are **A–Z, then a–z**, covering up to 52 options in one question. Send this as **one user message** through the source chat template with `add_generation_prompt=True` and `enable_thinking=False`. The final input position is `slot = len(ids) - 1`.

The helper's TARGETED readout obtains full-vocabulary log-softmax values at this position, then gathers the single-token ids of **bare letters**, without leading spaces. All 52 ids must be unique. Divide the gathered values by **T = 0.85**, then softmax over the listed letters only. Applying the same softmax to the bundle's gathered logits is equivalent because the full-vocabulary normalization is a shared additive constant.

**Choice** returns this distribution in criteria-map order. **Score** appends ` Rate along the ordered levels below (lowest first).` to the instructions, uses numeric keys `0`, `1`, … with the level descriptions, and returns the expected zero-based level **Σ i·p(i)**. **Noul** uses yes first and no second, with explicit `true`/`false` criteria or the defaults `The statement is true.` and `The statement is false.`. Let `p_yes = p[0]`, clamp it to `[1e-4, 1−1e-4]`, then return **sigmoid(logit(p_yes) / 1.829074 + 0)**. The helper's public answers round to four decimals; fixtures retain its unrounded distribution, raw letter log-probabilities, p_yes and calibrated noul, as well as the actual rounded API answers.

The measured serving settings are `READOUT_T=0.85`, `READOUT_NOUL_T=1.829074`, `READOUT_NOUL_BIAS=0`, `READOUT_TARGETED=1`, `READOUT_INSTR_STYLE=pyrepr`, `READOUT_PERMS=1` and `SHIM_STAGGER=1`; `SHIM_PAD`, `SHIM_COMPACT`, `SHIM_LAYOUT` and `SHIM_LOOP_BREAK` are off. `pyrepr` affects object/list instructions; string instructions are unchanged. For a text-only dict state, the helper's `with_image` serializes the whole dict with `json.dumps(..., ensure_ascii=False)`, so `{"text": ...}` retains its `text` key rather than extracting it; a string state stays a string. Screenshot/image extraction in the source helper is outside this text-only port.

## Measured (Apple M4 Max GPU, 128 GiB RAM, macOS 27.0 26A428, 2026-09-23)

The fidelity results are complete; the S=1 prefill-rate proxy was not measured because no quiet GPU window existed on 2026-09-23. The kit's per-question times below are the only speed measurements for this port, and they are contended.

The reference is **oracle A: the author's unchanged helper/shim.py over its BF16 weights**, using the author's `shim_mlx.py` stand-in mechanism with mlx-lm 0.31.3 / MLX 0.32.2. The helper SHA256 is `81a22f1b1b8912a465059207ef9f60b7c6c16b4de6372305d867efbe38a1987a`. No FP16 bundle or second FP32 oracle is used; an FP32 copy of this 27B model would require about 108 GB just for weights.

| Check on all 61 rows | int8hu vs author's BF16 helper |
| --- | ---: |
| Option argmax agreement | 61/61 |
| Agreement at oracle margin ≥ 0.02 | 61/61 |
| Max \|Δp\| | 0.000276405407 |
| Mean of row mean \|Δp\| | 2.05806795118e-05 |
| Max calibrated noul \|Δ\| | 0.000306657991 |
| Full-vocabulary argmax is a listed letter | 61/61 |
| Finite logits and probabilities | yes |
| Fresh-state reset logits, bit-identical | yes |
| Release pipelined first text = Python full-vocabulary argmax text | 61/61 |
| Release sequential first text = Python full-vocabulary argmax text | 61/61 |

The [fixture](fixtures-openjev-27b.json), schema `coreai-letter-fixtures/1`, contains **61 requests / 61 rows, with 51 distinct composed prompts**: 30 Choice rows with 2–26 options, three Choice rows with 27/40/52 options, 14 Noul, 12 Score and two additional long Choice rows marked `zoo_only`. There are six ordinary large-menu rows, six descriptive-choice rows, 23 choice rows with null descriptions, four explicit true/false criteria, two ten-level scores and four JSON/DOM agent-shaped rows. German, French, Japanese and Chinese each contribute one counted row. Kit-sized prompts contain 90–505 tokens; the two `zoo_only` rows contain 1,559 each. Some Noul prompts repeat. All oracle top-two margins are at least **0.988485**; there are no near ties. These synthetic, high-margin fixtures test conversion fidelity and do not establish broad task accuracy or calibration quality.

Readout uses **AOT h16c GPU**, `SpecializationOptions.default()`, fresh zero KV/conv/recurrent states and S=1 steps through every prompt token. Final logits are FP16 `[1,1,248320]`; label gathering and softmax use FP32. Each worker evaluates at most **six prompts including its reset**: normally five distinct rows and a repeated first row, with the long rows isolated. A final fresh process repeats row one. The [readout transcript](gate-openjev-27b-readout-int8hu.json) includes raw-logit scale, finite checks, probability differences, calibration differences and the worst five rows.

The [engine check](gate-openjev-27b-engine-int8hu.json) runs both Release variants with `--raw-tokens`, `--max-tokens 1 --temperature 0.0 --warmup off` and `COREAI_CHUNK_THRESHOLD=1`. Expected text is the **same bundle's Python full-vocabulary argmax** decoded by the tokenizer, not the oracle's selected option. Both engines complete **122/122 exact first-token comparisons** in total. This checks the engine's output token; probabilities are checked by the Python readout above.

The zoo-layout readout and engine scripts were run once on this same bundle: **61/61 option argmaxes** and **122/122 engine texts** agree. Its max |Δp| is **0.000276405406506**, differing from the accepted round-1 maximum by **0** (required ≤1e-6). [Gate summary](gate-openjev-27b.json).


## Through the kit

**Measured through coreai-kit** by the supervisor on **2026-09-23 11:40–12:17 JST**, in worktree `~/code/coreai-kit-models-wt`, branch `decision-models-5`, using **`Decision.Format.letterList`** and **`coreai-sequential`** at about **28 GB resident**. The kit rendered the helper's prompt itself. **All times below are contended:** the GPU was shared with this run's zoo gates. These supplied measurements were not re-derived by the conversion run.

`decide-cli parity` on the canonical `coreai-letter-fixtures/1` fixture used bare labels, the chat template and **T = 0.85**. All **61/61 rows** were **token-, slot- and argmax-identical** to the helper's own readout: **35 Choice** rows with 2–52 options, **14 yes/no** rows and **12 Score** rows. The yes/no values were compared **after the helper's calibration**, against the fixture's `noul`.

| Kit parity check | int8hu |
| --- | ---: |
| Tokens / slots / option argmax identical | 61/61 each |
| Max / mean \|Δp\| | 0.0003 / 0.0001 |
| Median question time, contended | 9.3 s |
| Ordinary row length | 90–505 tokens; median 114 |
| Two long rows | 1,559 tokens each; up to 137 s |

On **SemIf's authored144**, the supervisor used 144 English rows with three options each, SemIf's gold labels and its unchanged `benchmarks/evaluate.py`, rendering every row as a choice in the helper's form. The kit returned **134/144 raw correct** and **0.907 mean family balanced accuracy**: evidence_interpretation **0.944**, rule_application **0.917**, candidate_selection **0.861**. Median decision time was **8.3 s**, contended, for about **102 tokens**.

For scale on the same rows and evaluator, the kit README reports Qwen3.5-4B int8 zero-shot **0.821**, MiniCPM5-2B int8 **0.681**, OpenThai-SystemOne **0.725**, APUS-OpenJev-v1-4B **0.906**, Qwen3.5-2B-Decision **0.798** and the System One scorer 4B **0.844**. These are numbers on that evaluator, not a ranking.

The supervisor also measured a **separate README example request**, using this state:

> Customer message: I was charged twice for my order last week and nobody has replied.

| Question and declared options | Kit answer; helper BF16 comparison when supplied | Tokens | Time, contended |
| --- | --- | ---: | ---: |
| Which team: billing / shipping / technical | billing **1.000**; helper **0.9998** | 70, same row as helper | 5,064 ms |
| Is the customer angry: yes / no | calibrated P(yes), or Noul, **0.630**; helper **0.6183** | 74 | 4,771 ms |
| How urgent: can wait / this week / today / right now | expected level **2.08**; today **0.675** | 95 | 6,198 ms |

These example answers are separate from the 61-row fixture parity and are not covered by its 0.0003 maximum probability error. The request reused **0 tokens**: this decode-only graph reads each complete question row, at about **70 ms per input token** on the sequential engine under these contended conditions.

The kit resolves the format from bundle metadata (`decision.readout == "letters"`, `temperature`, `noul`), renders the helper's text byte for byte under the chat template, lists up to **52 options**, and calibrates yes/no the helper's way. Its catalog entry is **`openjev-27b`**, **macOS only**, with **`license: CC-BY-NC-4.0`**, pinned to the Hub revision. [Supervisor-supplied kit measurements](measurements-coreai-kit.json).

## Bundle

The single **int8hu** LanguageBundle is `gpu-pipelined/openjev_27b_decode_int8hu_block32_sym`. It is **27.8 GiB** (29,803,295,665 bytes); the author's BF16 helper readout is its reference. No FP16 bundle is included.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `.aimodel/main.mlirb` | 29,774,700,211 | `004d52234efd0ac6ef2f7d81c33d72a502ad8a768d5424448049fcf6d3b73cc5` |
| Root `metadata.json` | 887 | `6f15f99cc3e25505d63fb5b086dcd189d9b9684ac13fab6468b513499aa1be1c` |

The 64-layer graph has loop-free single-step enabled on 48 linear-attention layers, vocabulary **248,320**, and compiled context **4,096**. `int8hu --head-sym` uses block-32 int8 linear weights and a symmetric block-32 vocabulary head; embeddings, conv1d and norms remain FP16. The source tokenizer and chat template are preserved. Metadata names `openjev/openjev` and revision `5ec9e5fd2f80a6fff386779b1e5ac7e389971889`, the bare-letter chat readout, T=0.85, and the yes/no calibration. The helper's inherited `compression: null` remains; precision is specified by the recipe.

The **accepted round-1** Python readout process peaked at **111.9 GB RSS** (111,894,233,088 bytes) on this **128 GiB Mac**. This is the measured round-1 worker peak, not a minimum-memory specification. The completed round-2 zoo readout separately peaked at **116.1 GB RSS** (116,084,408,320 bytes) on the same 128 GiB Mac. These are separate observed process peaks for the two gate runs. The one successful 27B export took **234.3 seconds** and peaked at **67.9 GiB RSS** (72,858,877,952 bytes). Oracle, export and readout ran as separate processes to avoid overlapping the BF16 and Torch model allocations. This is a **Mac-only** port; no phone or Neural Engine measurement is claimed.

Use the [extra-states runtime patch](../../apps/coreai-pipelined-extra-states.patch) for KV/conv/recurrent state and set `COREAI_CHUNK_THRESHOLD=1` before engine creation. Release tools come from frozen fork tag **0.2.4-zoo** (`f7a75ec0f89fab451d277572afe8995b7ef768c1`).

## Reproduce

The export was run **once successfully**, in round 1. Round 2 does **not** re-export: the verified source weight blobs were deleted after the first round's gates and shipment verification. An initial round-1 offline cache lookup failed before any checkpoint read; recording the pinned revision in the private cache's `refs/main` resolved that lookup. The exporter diff and emitted metadata contract are the reproduction evidence; no second-export equality is claimed. The completed round-2 zoo readout separately peaked at **116.1 GB RSS** (116,084,408,320 bytes) on the same 128 GiB Mac. These are separate observed process peaks for the two gate runs.

The zoo [exporter](../../conversion/export_qwen3_5_decode_pipelined.py) preserves the shipped Qwen3.8-27B graph and quantization defaults, adding optional `--name`, `--revision` and `--extra-metadata` flags with nil defaults. `--revision` writes `source.hf_revision`; `--extra-metadata` reads a JSON object merged into root metadata through `_bundle.py`'s `extra`. The [OpenJev metadata file](metadata-extra.json) supplies its license and decision blocks; other bundles retain their defaults. [Recipe](recipe.toml) pins the source. With the verified pinned snapshot restored into a private HF cache, the export command is:

```sh
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1
export COREAI_CHUNK_THRESHOLD=1
python conversion/export_qwen3_5_decode_pipelined.py int8hu --head-sym \
  --hf-id openjev/openjev --name openjev_27b_decode_int8hu_block32_sym \
  --revision 5ec9e5fd2f80a6fff386779b1e5ac7e389971889 \
  --extra-metadata models/openjev-27b/metadata-extra.json --out-dir exports
```

Export pins: Python 3.11.13, coreai-core 1.0.0b2, coreai-torch **0.4.1**, coreai-opt 0.2.1, torch 2.9.0, transformers 4.57.6, safetensors 0.7.0, huggingface_hub 0.36.2, numpy 2.3.5, tokenizers 0.22.2 and accelerate 1.15.0; `coreai_models` is frozen fork `397b337e234474a191c0bd96ac9ef71c4f808a3d`. Oracle pins: mlx-lm 0.31.3, MLX 0.32.2, transformers 5.17.0, openai 3.16.2 and httpx 0.28.1. The [fixture builder](../../conversion/letter/oracle_openjev.py) embeds the requests and imports the SHA-checked author helper unchanged. [Letter gate instructions](../../conversion/letter/README.md) specify the bare-label/chat/temperature flags and calibrated Noul comparison. [Port notes](../../knowledge/openjev-27b-port.md) record the dtype-name correction, offline revision pin and memory limits. Defaults for existing APUS and 2B gate invocations are preserved. The generic metadata flags are checked without re-exporting: call `write_bundle_metadata` with those arguments in a temporary directory and compare with the shipped root metadata, allowing only `compilation.date` to differ.

## License

**CC BY-NC 4.0** applies to the converted weights. Source [openjev/openjev](https://huggingface.co/openjev/openjev/tree/5ec9e5fd2f80a6fff386779b1e5ac7e389971889) carries the **19,347-byte `LICENSE`**, staged verbatim. The base **Qwen/Qwen3.8-27B** at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` is **Apache-2.0**, and the source's `helper/` and `serve/` files are **Apache-2.0**. Source `LICENSE-APACHE-2.0`, `NOTICE` and `config.json` are staged verbatim. This port changes the source by exporting its text tower and quantizing the int8hu linear/head weights. The base/helper license does not replace the weights' non-commercial license. Publication remains a separate owner action.
