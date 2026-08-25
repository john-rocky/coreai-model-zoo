# Beta gotcha: a zero-sized dimension aborts the process on GPU and ANE (CPU is fine)

A graph carrying a **zero-sized tensor** converts cleanly, then kills the process at load or run on
the GPU and the Neural Engine. A CPU-only specialization runs the same asset. Two shapes of it, two
different assertions:

| construct | CPU | GPU | ANE | assertion |
|---|---|---|---|---|
| `x.split([n, 0], dim=1)` — a 0-length section, **never used** | runs | abort | abort | `MPSGraphExecutable.mm:4419` `MPSCommonRuntimeCanonicalization` |
| `return x[:, :0]` — a width-0 model output | runs | abort | abort | `MPSNDArray.mm:893` "buffer is not large enough. Must be 128 bytes" |

Neighbouring constructs are **fine** on all three units: `torch.cat` with a width-0 operand,
`new_zeros(n, 0)` fed into a `cat`, and indexing a width-0 tensor. So it is the zero-length `split`
section and the zero-sized *output* specifically, not zero-sized tensors in general.

Measured on M4 Max, macOS 27.0 (26A5416b), `coreai-torch` 0.4.2, `coreai-core` 1.0.0b2, torch 2.13.0.
Filed as [apple/coreai-torch#68](https://github.com/apple/coreai-torch/issues/68).

## Model-free reproducer

```python
class SplitZero(torch.nn.Module):
    def forward(self, x):
        a, b = x.split([x.shape[1], 0], dim=1)  # b is width-0 and never used
        return a + 1.0
```

Convert, `save_asset`, then load with `SpecializationOptions.cpu_only()` (runs) and with
`from_preferred_compute_unit_kind(ComputeUnitKind.gpu())` (aborts). Full script in the issue.

## Why it matters more than it looks

Nobody writes a zero-length split on purpose. It arrives from **generic postprocess code that sizes a
section from a subtraction**. Ultralytics' detection head does exactly this:

```python
extra_shape = pred.shape[-1] - (4 + len(names))     # 0 for plain detect, >0 for segment/pose/obb
boxes, scores, extras = pred.split([4, len(names), extra_shape], dim=2)
```

For a detect model `extra_shape` is 0, and the graph aborts on both accelerators. Dropping the third
section — same tensors, one line — makes the identical graph run on GPU and ANE.

Any "generic over task/variant" tensor-splitting helper is a candidate. Grep for `split(` with a
computed section width before blaming an op.

## The instrumentation trap this cost

**`SpecializationOptions.default()` can silently land on CPU.** A bisection run on `default()` proved
five graph variants "fine" that all abort under an explicit `gpu` / `neural_engine` specialization —
and the wrong conclusion (that the NMS block was the trigger) went out in a public PR comment before
the explicit-unit rerun caught it.

**Pin the compute unit explicitly in any accelerator isolation.** `default()` is a scheduling
decision, not a test target: it is allowed to fall back, and a fallback reads exactly like a pass.

Related: [`coreai-ane-partition-cost.md`](coreai-ane-partition-cost.md) (same YOLO26 work, the fixed
`topk` boundary cost), [`coreai-beta-mpsgraph-kvwrite-bug.md`](coreai-beta-mpsgraph-kvwrite-bug.md)
(the other MPSGraph beta abort, and the same isolate-one-variable shape).
