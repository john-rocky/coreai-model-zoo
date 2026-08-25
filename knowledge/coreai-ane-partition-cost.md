# An op the ANE cannot run charges a fixed cost, not a proportional one

Measured 2026-08-25. iPhone 17 Pro, iOS 27.0 beta 6 (24A5418b), coreai-torch 0.4.2, fp16,
AOT-compiled for h18p, input NDArray allocated once outside the timed loop, three interleaved
blocks of 50 iterations, thermal nominal.

`topk` is the case that surfaced this, because the Core AI dialect has `sort`, `argsort`,
`argmax`, `gather_along_axis` and `gather_nd` but no top-k primitive. `coreai_torch`
(`_aten_to_core.py:replace_topk`) lowers `aten.topk` to a full-axis `sort` plus a full-axis
`argsort`, then slices k off each.

Take a conv stack ending in a reduce, and add one `topk` to it:

| | body only | + one `topk` | delta |
|---|---|---|---|
| ANE-preferred | 0.46 ms | 1.06 ms | **+0.60 ms** |
| GPU-only | 0.96 ms | 1.26 ms | **+0.30 ms** |

The GPU column is the control: on the GPU nothing has to cross, and the op costs half as much.

**The cost does not scale with the work.** Cutting k from 300 to 10 moves the ANE number from
1.057 ms to 1.051 ms. On a real detection graph the same test moved the full model 3%, a second
`topk` over an axis three times longer added 2% on top of the first, and halving the sorting —
one `argsort` plus a `gather_along_axis` is equivalent to the stock `sort`+`argsort` — bought 1.8%.
Four ways of doing less work, and none of them buys anything.

**The magnitude is graph-dependent.** The synthetic stack above pays 0.60 ms. A real detector
paid 1.75 ms, with the GPU control at 0.05 ms rather than 0.30 ms. Treat the existence of a fixed
cost as the general result and the size of it as something to measure per graph.

## How to apply

Count boundary crossings, not ops. Moving one `topk` out of a detection graph and onto the host
took it from 3.06 ms to 1.32 ms, which put it level with Core ML running the same body.

Op counts mislead in the other direction too. Skipping the fuse step on that detector left 96
unfused BatchNorms and grew the exported graph from 367 to 602 ATen ops. On device that is worth
about 1%, because the Core AI compiler folds them itself.

## Unconfirmed

The partition explanation is inferred from the fixed-cost signature and the GPU-only control. No
public API was found that reports which ops ran on which unit — `ComputeUnitKind` is a request,
and `Profiler` / `IntermediateLogger` / `LogEvent` did not expose placement. Filed as
[apple/coreai-torch#66](https://github.com/apple/coreai-torch/issues/66). If Apple names a
different mechanism, this section is what to correct.

## Reproducer

```python
import torch, coreai_torch
from coreai_torch import TorchConverter

STAGE, K = "topk", 300      # STAGE = "body" keeps the graph on the ANE

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        ch = [3, 32, 64, 128, 128]
        self.blocks = torch.nn.Sequential(*[
            torch.nn.Sequential(torch.nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1),
                                torch.nn.SiLU())
            for i in range(4)
        ])
        self.head = torch.nn.Conv2d(128, 80, 1)

    def forward(self, x):
        s = self.head(self.blocks(x)).flatten(2).transpose(1, 2).max(dim=-1)[0]
        return s if STAGE == "body" else s.topk(K, dim=1)[0]

net = Net().eval().half()
ex = torch.rand(1, 3, 640, 640).half()
with torch.no_grad():
    ep = torch.export.export(net, (ex,)).run_decompositions(coreai_torch.get_decomp_table())

c = TorchConverter()
c.add_exported_program(ep, entrypoint_name="main", input_names=["image"], output_names=["out0"])
prog = c.to_coreai(); prog.optimize(); prog.save_asset(f"{STAGE}.aimodel")
```

Compile each stage for both preferences, then time them on device:

```
xcrun coreai-build compile topk.aimodel --platform iOS --preferred-compute neural-engine --architecture h18p --output ane/
xcrun coreai-build compile topk.aimodel --platform iOS --preferred-compute gpu           --architecture h18p --output gpu/
```

Time inside the app, not from the host, and allocate the input `NDArray` once outside the loop.
With marshalling inside the timed loop the same graph reads 8.9 ms instead of 3.0 ms, and the
measurement is of the bridge rather than the model.
