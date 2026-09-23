# laya/ — laya multilingual (convaiinnovations/laya, `multilingual/`) → Core AI

The recipe behind `models/laya-multilingual/` and (once published) `mlboydaisuke/Laya-Multilingual-CoreAI`.
laya is an encoder-type decision model: an mmBERT-base encoder (22 layers, 256k vocabulary) with a typed
decision head. One forward answers one typed question (choice / score / noul) over a state: the
host builds `[CLS] <type> question: … [SEP] [MASK] option … [SEP] state [SEP]`, the graph scores every
position, and the host reads the logits at the option markers.

Five stages, each a script, each gated against the one before; run them in this order from the repo
root (paths resolve through [`../_paths.py`](../_paths.py); the work dir is `<ZOO_WORK_ROOT>/laya-multilingual/`,
bundles go to `<ZOO_EXPORTS>/laya-multilingual/`):

| Stage | Script | Gate |
|---|---|---|
| 0 oracle | `uv run --python 3.12 conversion/laya/oracle_laya.py` | the publisher's `laya.load(...)` model, CPU fp32: reproduces the frozen fixture (ids 402/402, batched answers and logits bit-exact), then every row alone, batch 1, padded to its window, vs the fixture (argmax, \|dp\| ≤ 1e-3) and the LiteRT lane's captures (marker ≤ 1e-3, act relative ≤ 1e-4); every hidden state for 14 rows |
| 1 authoring | `python3 conversion/laya/gate_laya_authoring.py --window 256\|512 --negative-controls` | the re-authored graph vs the oracle: 201 rows (answer + tensor gates), every saved hidden state (1e-4 bar, with the official model's own SDPA-vs-eager distance beside it), pad isolation, five mutations that must be caught |
| 3 export | `python3 conversion/laya/export_laya.py --window 256\|512 --dtype fp32\|wfp16\|fp16 --target macos\|ios` | torch-export + decomposition, the same 201-row gate before conversion, then one multifunction `.aimodel` (`main` + `act`) |
| 4 runtime | `python3 conversion/laya/gate_laya_runtime.py <variant dir> --compute cpu_only\|gpu\|neural_engine` | the bundle on the Core AI runtime: 201 rows, repeat drift, wrong-pairing control, load + warm timings |
| fixture | `python3 conversion/laya/fixtures_laya.py [--check]` | writes `models/laya-multilingual/fixtures-laya-multilingual.json` (schema `coreai-encoder-fixtures/1`: 44 states + the 402 frozen rows) — what coreai-kit's `decide-cli parity` reads |

Stage 2 (the host) is `_laya_host.py`: the NumPy gather / act features / temperature / decode that
every stage above runs, and the algorithm the Swift side ports (HOST_CONTRACT §C–§D of the LiteRT lane).
The sequence builder itself is the publisher's `build_sequence`, reproduced by the oracle on all 402
fixture rows; a Swift port is checked against the same rows (`reference.json` carries ids and markers).

## Graph contract (one question row, batch 1, static window S ∈ {256, 512})

| Function | Tensor | Shape | Dtype |
|---|---|---|---|
| `main` in | `input_ids` | [1,S] | int32, right-padded with PAD 0 |
| `main` in | `attention_mask` | [1,S] | int32, 1 real / 0 pad |
| `main` in | `qtype_onehot` | [1,3] | float32, choice / score / noul |
| `main` out | `token_logits` | [1,S] | float32, read at the marker positions |
| `main` out | `pooled_cls` | [1,768] | float32 |
| `act` in | `pooled_cls`, `feats` | [1,768], [1,4] | float32 |
| `act` out | `act_logits` | [1,2] | float32 (fp32 in every variant) |

`feats` = [top1, top1 − top2, entropy / ln(max(K,2)), max(K,2)/255] of the softmax of the RAW marker
logits. Temperatures are host work: the source config is T = [1,1,1]; `metadata.json` also carries the
fitted calibration of the LiteRT lane (bucket first, then per type).

## Environments

- Stage 0: its own `uv` environment from the script header (Python ≥ 3.12, torch 2.12.1,
  transformers 5.17.0, laya 0.3.4 — transformers 4.x cannot read this encoder's v5 config).
- Stages 1, 3, 4: the zoo venv (torch 2.9.0, coreai-torch 0.4.1, coreai-core 1.0.0b2); the graph
  imports neither transformers nor laya.
- The checkpoint is the pinned HF snapshot `convaiinnovations/laya@1c5edc17…` (sha256-checked on every
  load); the fixtures are the frozen LiteRT-lane files, sha256-pinned in `_common.py`.

## What the port found

- **The layer bar meets a massive activation.** From layer 11 the CLS position carries values up to
  1.4e4 (dims 488/580/530/614/468; one fp32 ulp there is ~1e-3). The official model's own two attention
  paths (SDPA, which `laya.load` uses, and eager) already differ by up to 4.8e-2 on those states; the
  re-authored graph sits in the same band (relative error ≤ 8e-6) while its marker logits match to
  7e-5. The 1e-4 layer bar holds on the embeddings and layers 0–9 and is where it catches
  window63 / all_global / all_local / ignore_padding by 4–6 orders of magnitude; above that the
  authoring record reports it as failed rather than moving it.
- **Pad positions differ, and nothing reads them.** SDPA returns exactly zero for a fully masked
  sliding row (a pad query more than 64 past the last real token); this graph's finite mask
  (−1e4, fp16-safe) averages instead. Every attention masks pad keys, so real positions are
  bit-identical under random pad ids (checked).
- **fp16 compute does not hold the answer bar; fp16 storage does.** The recipe (fp16 everywhere except
  RoPE application and softmax) misses the bar in torch (max \|dp\| 3.4e-3 at S=256, argmax intact) and
  more on the Core AI CPU (\|dp\| 4.4e-2, one argmax flip); keeping the residual stream and LayerNorm in
  fp32 as well does not rescue it (4.9e-3). `wfp16` stores the weights in fp16 — exact, the checkpoint is
  F16 — and casts them to fp32 in the graph: coreai-torch keeps those constants fp16, so the bundle is
  645 MB instead of 1.29 GB and computes bit-identically to fp32 on the CPU.
