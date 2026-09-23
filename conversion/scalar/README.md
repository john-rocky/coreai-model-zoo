# Scalar decision gate — cc-by-nc-4.0

These tools reproduce `pngwn/system-one-qwen3.5-4b-scorer` at revision
`e6464dce15f013c2ef641593a85cc6afcdaea928`, based on
`Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b` (Apache-2.0).
The scorer and its converted bundles carry **cc-by-nc-4.0**.

`oracle_scalar.py` embeds 20 deterministic synthetic English requests: 48
questions and 280 option rows, including 6 long states, 24 ordinary choice
questions, 10 yes/no (`noul`) questions, 10 ordered-score questions and 4 zoo-only
questions with 20–32 options. It imports `system_one.py` unchanged from the pinned
adapter snapshot and checks the script SHA-256. The author's `load_model` builds
the CPU fp32 model, PEFT restores the adapter and scalar head, and an in-memory
fp32 `merge_and_unload()` is checked on three sequences to tolerance 1e-3. The
oracle stays fp32; it never reloads the BF16 storage snapshot. The author's
`encode`, `score_options`, and `softmax` are called directly without modification.

The fixture schema is **`coreai-scalar-fixtures/1`**. Each option is one row with
`ids`, `slot = len(ids) - 1`, fp32 `scalar`, and recorded state truncation. A
question's distribution is `softmax(scalars / 1.75)`; temperature 1.75 is the
source's validation-fitted value. The kit reads this same fixture schema.

Encoding adds no special tokens. The tail is
`\n\nQuestion:\n…\n\nOption:\n…`. When it has fewer than 384 tokens, keep the tail
whole and use the prefix of `State:\n…` that fits the remaining space: the state
is cut from its **end**. When the tail alone has at least 384 tokens, discard the
state and use `tail_ids[-384:]`, left-truncating the tail if it exceeds 384.
Right padding is used for the CPU oracle; pooling selects the last non-pad
position. These synthetic rows contain no pad/EOS tokens, so that position is
exactly `slot`.

`readout_gate_scalar.py` compiles an AOT **h16c GPU** asset and loads it with
`SpecializationOptions.default()`. Four fresh zero states are created for every
row. It calls the S=1 decode graph once per ID, checks output shape `[1,1,1]`, and
reads the final call's `logits[0,0,0]`. It reports finite scalar values and their
scale, max/mean scalar error, per-question argmax agreement and max/mean
probability error. Each worker executes at most 14 rows plus its first row again
(15 executions); another fresh worker repeats global row 1. Reset equality is
checked by dtype and exact bytes. Near ties with oracle top-two probability
margin below 0.02 are listed separately. Correctness execution times may be
contended and are not throughput measurements.

`engine_load_scalar.py` invokes both Release Swift engines on two rows per bundle
with `--raw-tokens`, `--max-tokens 1`, `--temperature 0.0`, `--warmup off`, and
`COREAI_CHUNK_THRESHOLD=1`. Every call is a fresh process, so the checks cover
loads, fresh-process resets and completion. A one-wide head always selects token
ID 0; its generated text is meaningless. The runner logs text and token count,
so the tool checks one output token and its ID-0 decoding. It does not validate
Swift option probabilities; the scalar readout is the numerical gate.

Run from the zoo root, with writable work/cache paths in your own checkout.
Replace the uppercase paths below with local absolute paths. Base and adapter
snapshots must be the pinned, LFS-SHA-256-verified snapshots. `--compare` is
optional; use it to compare to the original unmerged PEFT fixture without
modifying that file. The comparison requires identical IDs/truncation and argmax,
and fp32 scalar differences no greater than the merge tolerance 1e-3. Differences
between merged and unmerged fp32 per-option probabilities are reported as
informational; the separate reproduction gate compares the GPU aggregate maximum
probability error to its previous measurement at tolerance 1e-6.

```sh
HF_HUB_DISABLE_XET=1 HF_HOME=WORK/hf uv run conversion/scalar/oracle_scalar.py \
  --base-snapshot BASE --adapter-snapshot ADAPTER --work-dir WORK/oracle \
  --out WORK/fixtures.json --compare models/system-one-scorer-4b/fixtures-system-one-scorer-4b.json
```

The oracle's PEP 723 header pins the working torch 2.9.0 / transformers 5.17.0 /
PEFT 0.21.0 environment. The Core AI runtime uses a separate Python environment:
coreai-core 1.0.0b2, coreai-torch 0.4.1, coreai-opt 0.2.1, torch 2.9.0,
transformers 4.57.6, safetensors 0.7.0, numpy 2.3.5, tokenizers 0.22.2. Install
`coreai_models` from the frozen fork python tree at
`397b337e234474a191c0bd96ac9ef71c4f808a3d` into this environment. Set
`DEVELOPER_DIR` to the Xcode 27.0 RC developer directory. Its Core AI runtime needs
an open fence on macOS 27.0 build 26A428. Python workers inherit the supplied
`--deadline-epoch`; the portable default is two hours. Pass the same work directory
to both modes to share the repeated-runtime-error stop ledger.

```sh
EXPORT_PYTHON conversion/scalar/readout_gate_scalar.py \
  BUNDLES/system_one_qwen3_5_4b_scorer_decode_fp16 WORK/fixtures.json --mode fp16 \
  --snapshot MERGED --work-dir WORK/runtime --transcript WORK/readout_fp16.json
EXPORT_PYTHON conversion/scalar/readout_gate_scalar.py \
  BUNDLES/system_one_qwen3_5_4b_scorer_decode_int8lin WORK/fixtures.json --mode int8lin \
  --snapshot MERGED --work-dir WORK/runtime --transcript WORK/readout_int8lin.json
EXPORT_PYTHON conversion/scalar/engine_load_scalar.py WORK/fixtures.json \
  --bundle-root BUNDLES --runner RELEASE_LLM_RUNNER --work-dir WORK/runtime \
  --transcript WORK/engine_load.json
```

The Release runner comes from fork tag `0.2.4-zoo`. `MERGED` supplies the text
configuration only for GPU state shapes; GPU execution uses the exported bundle,
not the source checkpoint. No tool imports a conversion-run-specific module or
contains an absolute conversion-run path.
