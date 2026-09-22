# Zero ANE regions is a failure mode, not a rounding error

`--preferred-compute neural-engine` is a **preference**. When it cannot be honoured the compile
still succeeds and the bundle still runs — on the GPU. The cheap signal is the region count, and
it is easy to measure wrong in two different ways.

## Count it with the right glob

```bash
find <bundle>.aimodelc -name '*ANE_region_*.mlir.bc' | wc -l    # 0 = silent GPU fallback
```

**Use that pattern, not `*ANE_region*`.** The regions live *inside* the MPSGraph package
(`…/mpsExecutable.mpsgraphpackage/binary_0.llir.bundle/<fn>_ANE_region_0_0.bc/<arch>/…mlir.bc`), so
the loose pattern matches the region **directory** and the file **inside** it and double-counts
every region. On the bundle measured here, `*ANE_region*` reported 14 / 2 / 2 / 2 where the correct
counts are **13 / 1 / 1 / 1**. The ordering survives the error, so it hides behind a correct story —
which is exactly when a count gets reused in four places before anyone checks it.

## Two failure modes, and they are not the same

The repo's fp32 CV exports *do* reach the ANE — CLIP at fp32 lands in a **GPU/ANE tie**
(`apple-models-bench.md`). A text encoder (Granite-Embedding-97M, ModernBERT, S=128) at fp32
instead forms **zero regions**: the compile emits an MPSGraph delegate and nothing else. A tie and
a total fallback look the same in a latency table and are different problems.

| build | regions | warm median | load |
| --- | ---: | ---: | ---: |
| fp32 | **0** | 4.31 ms (GPU) | 797 ms |
| fp16 | 13 | 13.19 ms | — |
| fp16 + two fp32-op removals | **1** | **2.14 ms** | **257 ms** |

The two ops are the ones `compute-units-and-authoring.md` §29 warns about, named for this graph:
`F.softmax(scores, dim=-1, dtype=torch.float32)` in the attention, and `pooled = pooled.float()`
in the pooling head.

**13 regions → 1 region is 13.19 ms → 4.60 ms** for the same graph at the same precision, measured
in the same pass: ~0.72 ms per boundary. That is the compile-side view of the delivery cost
`compute-units-and-authoring.md` reports from Instruments (25 submissions, ~1.1 ms median
inter-submission gap, 28.7% idle) — same phenomenon, different model, different device, and it
supports the same conclusion: **the fix is fewer, larger regions, and the region count is the
metric that tells you whether you got them.**

## Reading the count honestly

It is a **shape** metric, not a quality one: it says how finely the graph is cut, not how much ran
where. Two instruments that pair with it:

- **IOReport, unprivileged.** `AMC Stats → ANE DCS RD/WR` (bytes) and the `ane 0` subgroup's
  interrupt counts: **173 GB / 27,784** under a working ANE run, **0 / 0** under `--compute
  cpuOnly`. They discriminate. `Energy Model → ANE` (mJ) is **frozen** on M4 and must not be read
  as a zero.
- **Not** `xctrace`'s `ane-hw-intervals`: it reports **0** for Core AI graphs, process-scoped *and*
  system-wide, while a Core ML control in the same session reports 1310 intervals. The `Core AI`
  template's own table (`ODIEProfile`, 9,127 rows) carries no compute-unit attribution, and there
  is no ANE-specific template.

## Two more traps in the same family

- **`--compute` cannot override an AOT bundle.** A `.aimodelc` compiled with
  `--preferred-compute neural-engine` uses the ANE under `--compute neuralEngine` *and*
  `--compute gpu` (101.6 GB vs 91.2 GB moved); only `cpuOnly` drops it to zero. Placement is baked
  at compile time, so an AOT bundle is not a general-purpose artifact.
- **`powermetrics`' `ane_power` is not a placement oracle here.** Two runs of the same workload
  reported ~1.9 W and ~0 mW. Whatever it measures, it disagreed with itself by ~200×.

## Compression, for this graph: four negatives

| variant | ANE regions | gate | min cosine | clear pair flips |
| --- | ---: | --- | ---: | ---: |
| w8 + fp16 | 13 | PASS | 0.9993834 | 0 |
| w6 + fp16 | 14 | **FAIL** | 0.9983865 | **1** |
| w4 + fp16 | **0** | **FAIL** | 0.9621623 | **16** |

- **w4 forms zero regions** — it reproduces the repo's *"4-bit g8 does not compile for the ANE"*
  with **group 16**, so the group size is not the explanation — *and* it inverts 16 document pairs.
- **w6** compiles (14 regions) and fails the gate. It shipped for a 2B LLM; it does not hold here.
- **w8** passes the gate and is **4.8× slower** (19.48 vs 4.05 ms): int8-affine weights fold to
  dense fp16 before the data-movement step, so they save storage and not bandwidth. It is a storage
  lever, not a compute one.
- **fp8 is not available on M4.** The type is in the catalog (`float8e4m3fn`, `float8e5m2`,
  `float8e8m0fn`, `float4e2m1fn` in `coreai.runtime._ndarray`) but the datapath is **H18-only**
  (`ane-silicon-reference.md`), and the ANE's documented dtypes are fp16/int8/int16.

## An observation left open

The ANE shows two throughput states on this graph: **burst** ~1.85–2.25 ms (~31 GB/s effective) and
**sustained** ~4.75 ms (~14.5 GB/s). The transition is **one-way** within a run (samples 0–830 fast,
831–20,000 slow in a 20,000-inference run), short 120-inference bursts stay fast, and **4 minutes of
idle does not restore it**. The machine is far too efficient to throttle in 2 s, `pmset -g therm`
raises nothing, the GPU path does not show the same magnitude, and the ANE power rail was not
trustworthy enough to separate a clock drop from a bandwidth ceiling. Reported in case others have
seen it.
