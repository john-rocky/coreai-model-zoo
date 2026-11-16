# Ornith-1.0-9B (agentic coding, text decoder) — Core AI

The zoo's **first agentic-coding / self-scaffolding model**. Source:
[`deepreinforce-ai/Ornith-1.0-9B`](https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B)
(MIT, released 2026-06-25) — DeepReinforce's Ornith 1.0 family trains the model to
jointly solve coding tasks AND construct the orchestration scaffold that guides the
solution; the 9B dense is its phone-/Mac-class member (the 397B MoE sibling matches
frontier assistants on SWE-Bench Verified). An image-text-to-text checkpoint; this
card is the **text decoder**.

Architecturally it is a **Qwen3.5 hybrid decoder** (`model_type qwen3_5`) — the same
family as the shipped 0.8B / 2B / Qwen3.6-27B — so it rides the
[pipelined engine](../../knowledge/pipelined-engine.md) with **zero new export code**
(the stock script + `--hf-id`):

- 32 layers on the 3:1 interleave of **GatedDeltaNet** linear-attention mixers and
  gated full attention (`full_attention_interval 4` → 8 full / 24 linear);
- full attention: head_dim 256, GQA **16 query / 4 KV heads**, partial mRoPE θ=1e7
  (text-only positions ≡ plain RoPE), sigmoid output gate;
- linear (GatedDeltaNet): **32 value heads over 16 key heads** (the 0.8B/2B/35B GVA
  ratio), key/value head_dim 128, short causal conv (k=4) + delta rule;
- dense `MLP(12288)`, hidden 4096, **untied 248320-vocab `lm_head`** (at the
  checkpoint root — the loader's untied branch, like the 27B);
- the checkpoint carries a vision tower (`model.visual.*`, 333 tensors) and an MTP
  config stub but **no MTP weights**; the text-decoder load skips vision cleanly.

**⬇️ Converted `.aimodel` bundle:** `ornith_1_0_9b_decode_int8hu_block32_sym/`
(9.8 GB, full LanguageBundle incl. tokenizer + chat template; decode-only loop-free,
max_ctx 8192). *HF upload pending.*

## Numbers (verified 2026-07-03)

| check | result |
|---|---|
| oracle (fp32 HF eager, margin-validated prompt) | sweep min top-2 margin **0.205** (most 2–8) |
| fp16 eager teacher-forced (baseline) | **24/24 exact** (16 sweep + 8 decode) |
| **int8hu eager (ship recipe)** | **24/24 exact** — zero flips, even at the 0.205 / 0.058-margin knife edges |
| engine path (release `llm-runner`, greedy, raw prompt ids) | **12/12 token-exact ≡ fp32 oracle** (int8hu AND int4lin) |
| Mac bench int8hu (M4 Max, p128 g256 n3, `COREAI_CHUNK_THRESHOLD=1`) | **decode 48.3 / prefill 48.5 tok/s** (trials ±0.1) |
| Mac bench int4lin (same rig) | **decode 58.9 / prefill 59.0 tok/s** (+22%, 7.5 GB) |

Ship recipe = the family ship shape: linear int8 per-block-32 `symmetric_with_clipping`
body (embed/conv/norms fp16) + **absmax `symmetric` per-block-32 int8** on the untied
big-vocab head (clipping crushes outlier head rows — see the knowledge page). Effective
per-token read ≈ 7.7 GB → the 48 tok/s is bandwidth-bound and healthy.

Chat path: the bundled Qwen chat template works as-is (`llm-runner --prompt` wraps it;
fluent assistant-style code analysis).

## int4 — the family's first clean PTQ pass

`int4lin` (linear per-block-32, head fp16) **gates 24/24 exact** on the same eager
teacher-forced rig — including the 0.205 / 0.058-margin knife edges — and the engine
bundle reproduces the oracle greedy 12/12. That is a first for the Qwen3.5 family:
0.8B failed int4 at every scheme, the 2B failed linear per-block-32, the 27B was
borderline (one confident flip). Quantization sensitivity is a model property; this
RL-trained 9B tolerates 4-bit where its siblings don't. Measured: **7.5 GB bundle,
58.9 tok/s decode on M4 Max (+22% over int8hu)** — this shape is in the
gemma4/LFM "+~24%" class, not the qwen-2B flat class.

**Why int8hu is still the default ship**: every instrument we ran is clean for int4,
but they are all short-context (48 gated tokens). The gemma4-VL port showed int4
clipping error can compound invisibly until real context lengths (272-token gate
caught flips a 27-token gate missed). Until a long-context gate says otherwise,
int8hu carries the quality claim; int4lin is one flag away and fully verified at
short range.

## iPhone status

int8 at 9.8 GB exceeds the entitled jetsam ceiling (~6.4 GB on iPhone 17 Pro) — **Mac
ship for v1**. int4lin (7.5 GB measured) helps but not enough as-is. The arithmetic
route to the phone is int4 body + int8 head (≈ 6.5 GB, at the jetsam edge) or
additionally an in-graph int8 embed table (≈ 5.5 GB, real headroom) — both future
work, the latter needs the gemma4-`--tbl`-style int8 embed gather.

## Convert it yourself

```bash
cd coreai-models && .venv/bin/python \
    ../coreai-models-community/conversion/export_qwen3_5_decode_pipelined.py \
    int8hu --head-sym --hf-id deepreinforce-ai/Ornith-1.0-9B --max-ctx 8192
# gate (eager, fp32-oracle margin rule):
#   _smoke/gen_ornith9b_ref.py (oracle) + _smoke/test_ornith9b_eager_gate.py
COREAI_CHUNK_THRESHOLD=1 llm-benchmark \
    --model exports/ornith_1_0_9b_decode_int8hu_block32_sym -p 128 -g 256 -n 3
```

Requires the qwen3_5 model overlay + the pipelined-engine extra-states patch
(`apps/coreai-pipelined-extra-states.patch`) — identical requirements to the shipped
qwen3.5 family; nothing new.

## License

Model weights MIT (upstream `deepreinforce-ai/Ornith-1.0-9B`). This conversion keeps
the upstream license; see the HF card for the Ornith 1.0 technical report.
