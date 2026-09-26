# GLiNER2.5-Decide: AOT bundles and the Swift host

The export and its Python gates are one level up: `../gliner25_decide_oracle.py` writes the gliner2 2.0.0 fp32 oracle
fixtures, `../export_gliner25_decide.py` exports the fp16 graph (S = 256 and S = 512, MMAX = 32) and gates it on the
Mac GPU. This directory takes the exported bundles to the phone.

| file | what it does |
|---|---|
| `aot_compile.py` | `coreai-build compile` for iOS h18p (iPhone 17 Pro) and h19p (iPhone 18 Pro) and macOS h16c (gpu preferred, fixed shapes: no expect-frequent-reshapes); records wall time, size and ANE regions in `_work/aot/aot_compile_<tag>.json` |
| `swift/` | package `GLiNERDecide`: `DecideGraph` (loads a bundle, one call = input_ids / attention_mask [1, S] + label_idx [1, MMAX] → logits [1, MMAX]), `DecideGateSupport` (a 1:1 port of the export's host side: graph inputs, the decision rule, the comparison), `decide-selftest` (the Mac gate) |
| [`../../apps/DecideGate`](../../apps/DecideGate) | the iPhone gate app: the same package, on the bundle of the phone's own architecture |

The Swift host takes the oracle's token ids from the fixtures; tokenization is not ported here.

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

The S = 512 bundle takes all three fixture sets (`fast_decisions_long.json` adds rows of 257 to 431 tokens).
