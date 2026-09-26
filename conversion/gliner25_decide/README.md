# GLiNER2.5-Decide: AOT bundles, the Swift host and the Hub folder

The export and its Python gates are one level up: `../gliner25_decide_oracle.py` writes the gliner2 2.0.0 fp32 oracle
fixtures, `../export_gliner25_decide.py` exports the fp16 graph (S = 256 and S = 512, MMAX = 32) and gates it on the
Mac GPU. This directory takes the exported bundles to the phone and to the Hub. The recipe is
[`../../models/gliner25-decide/recipe.toml`](../../models/gliner25-decide/recipe.toml); the gate records are the
`gate-*.json` files beside it.

| file | what it does |
|---|---|
| `aot_compile.py` | `coreai-build compile` for iOS h18p (iPhone 17 Pro) and h19p (iPhone 18 Pro) and macOS h16c (gpu preferred, fixed shapes: no expect-frequent-reshapes); records wall time, size and ANE regions in `_work/aot/aot_compile_<tag>.json` |
| `swift/` | package `GLiNERDecide`: `DecideGraph` (loads a bundle, one call = input_ids / attention_mask [1, S] + label_idx [1, MMAX] → logits [1, MMAX]), `DecideGateSupport` (a 1:1 port of the export's host side: graph inputs, the decision rule, the comparison), `decide-selftest` (the Mac gate) |
| [`../../apps/DecideGate`](../../apps/DecideGate) | the iPhone gate app: the same package, on the bundle of the phone's own architecture |
| `stage_ship.py` | lays out the Hugging Face repo in `<work>/_gliner25_decide/ship/`, checks every file against the record of the gate that ran it, writes `SHA256SUMS`; `--check <dir>` verifies a download |

This package takes the oracle's token ids from the fixtures. The tokenizer and the schema layout in Swift are
CoreAIKit's `TextClassifier` (kit branch `gliner-decide`), gated on the same fixtures.

## Gates, in order

1. **Oracle** (`../gliner25_decide_oracle.py`, gliner2 2.0.0 in its own environment: `uv run --python 3.12`): the 21
   examples of the model card and rows of `fastino/fast-decisions` become fixtures with token ids, marker positions,
   fp32 logits and decisions. Every case asserts that its decisions equal `classify_text`'s.
2. **GATE-1**, fp32 torch (`../export_gliner25_decide.py`): the re-authored module against the oracle, decisions equal
   on every task and max |Δlogit| ≤ 1e-3.
3. **GATE-2**, the negative control: `label_idx + 1` reads each label's first sub-word instead of its `[L]` marker and
   must change decisions.
4. **GATE-3**, the fp16 bundle on the Mac GPU: decisions equal on every task, every row finite. The same run exports
   the bundle and its side files (`tokenizer/`, `classifier.json`, `reference_s<S>.json`).
5. **decide-selftest** (Swift, below): the same cases through `DecideGraph`; the reference case's inputs bit-equal to
   the export's arrays, the logits bit-equal to GATE-3's.
6. **DecideGate**: the h19p bundles on the iPhone 18 Pro, every case, then a bench.
7. **TextClassifier** (CoreAIKit): token ids from text, equal to the fixtures' on every case; then the decisions.

S = 256 takes the README and the ≤ 256-token fast-decisions fixtures; S = 512 takes all three fixture sets
(`fast_decisions_long.json` adds rows of 257 to 431 tokens).

## Mac self-test

```sh
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
cd swift && swift build -c release
D=~/code/coreai/_gliner25_decide        # fixtures/, exports/, results/ of the two scripts above
.build/release/decide-selftest --bundle $D/exports/gliner25-decide_float16_s256_m32.aimodel \
    --fixtures $D/fixtures/readme21.json $D/fixtures/fast_decisions_s256.json \
    --reference $D/exports/reference_s256.json --pygpu $D/results/gate_s256_gpu.json --bench 100 \
    --json ../_work/selftest_s256.json
```

- `--reference`: the Swift graph inputs of that case against the export's arrays, bit for bit.
- `--pygpu`: the Python Mac GPU logits of the same bundle, case by case; and the decision rule re-run on those
  logits against the Python per-task records (decisions and every metric bit-equal when the port is right).
- `--poison`: label_idx + 1, a negative control that must exit 3.
- Exit 0 = pass, 3 = a gate failed, 4 = an error. An iPhone bundle (`.h18p.`, `.h19p.`) is refused on a Mac (it wedges the GPU stack).

## The Hub folder

```sh
HF_HUB_CACHE=<the cache with fastino/GLiNER2.5-Decide at 7ee5da4> \
    ~/code/coreai/coreai-models/.venv/bin/python stage_ship.py      # stage + check + SHA256SUMS + sizes
~/code/coreai/coreai-models/.venv/bin/python stage_ship.py --check <a download of the repo>
```

`ship/` holds `macos/` (the two `.aimodel` bundles, `tokenizer/`, `classifier.json`, `reference_s256.json` and
`reference_s512.json`), `ios/` (the two h19p `.aimodelc` bundles, the same side files, and a `classifier.json` that
names the compiled bundles), `gate/` (the three fixtures), `source/` (the source `config.json` and
`encoder_config/config.json`), `LICENSE`, `NOTICE`, `config.json` and `SHA256SUMS`. The script writes `NOTICE`,
`config.json` and `ios/classifier.json`; `LICENSE` comes from `$GLINER25_SHIP_LEGAL` (default
`<work>/_gliner25_decide/legal/`) and must have the pinned sha256. Before writing `SHA256SUMS` it checks the staged
copies: the macos bundles against the Mac GPU gate record, the ios bundles and the fixtures against the MD5SUMS the
phone was gated with (`_work/device_stage/DecideAssets/`), each compiled bundle's `sourceHash` against its source's
`main.hash`, and the tokenizer against the source snapshot. `README.md` (the card) is the one file it does not write;
run the script again after the card is in place, so that `SHA256SUMS` covers it.
