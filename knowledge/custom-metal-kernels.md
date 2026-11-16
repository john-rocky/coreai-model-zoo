# Custom Metal kernels in Core AI (the GPU speed lever)

> Foundation note for the **GPU-now** track. Custom Metal kernels are a first-class Core AI
> authoring feature (WWDC 325) and are **GPU-only** — the ANE cannot run hand-written MSL.
> Sources: `coreai-torch/docs/guides/custom-metal-kernels.ipynb`, `coreai-torch/docs/api/TorchMetalKernel.md`,
> `coreai-torch/coreai_torch/_torch_metal_kernel.py`, `coreai/authoring/metal.py`,
> `ondevice/_wwdc325_transcript.txt` (video `MdlyLT_y3i0`), `skills/.../model-authoring/references/gpu_rules.md`.

## What it is & when to use

A custom Metal kernel lets you write a raw MSL GPU function, wrap it with a PyTorch reference, and have
`torch.export` + `coreai-torch` embed it into the `.aimodel` as a real Core AI op (`coreai.metal4_kernel`).
It is **not a raw-Metal bypass** — the MSL travels inside the single `.aimodel` artifact and runs in the OS
Core AI runtime (WWDC 325, ~line 476–485).

Use it for **op fusion** — collapse several graph ops (and their dispatch + intermediate-memory traffic)
into one kernel dispatch. WWDC 325 (~line 461): *"You can take a group of these ops and fuse them into a
single operation. This replaces several steps with a single kernel dispatch within the graph."* Core AI
already ships fast prepackaged kernels for heavy ops (e.g. SDPA); write custom ones for what it doesn't
cover or to fuse a hot path.

**GPU-only — and that is the structural reason the speed track lives on the GPU.** The ANE runs only fixed
hardware ops (Conv/LayerNorm/…); it cannot execute arbitrary MSL. So "write fused-int8 kernels" is, by
construction, a GPU strategy — independent of any beta bug. (See [`compute-units-and-authoring.md`](compute-units-and-authoring.md).)

## The API

```python
from coreai_torch import TorchMetalKernel, TorchConverter, get_decomp_table
from coreai.authoring import MetalParameter   # re-exported as coreai_torch.MetalParameter

TorchMetalKernel(
    name: str,                       # kernel id; becomes part of the generated kernel name
    input_names: list[str],          # names used in the MSL body; count == torch_defn params
    result_names: list[str],         # output names used in the MSL body (>=1)
    src: str,                        # MSL body ONLY (signature/buffer bindings/#includes auto-generated)
    torch_defn: Callable,            # PyTorch reference — what torch.export sees, used for shape inference
    metal_params: list[MetalParameter] | None = None,   # thread attrs, e.g. thread_position_in_grid
    helper_src: str | None = None,   # extra MSL pasted before the kernel (helpers/typedefs/constants)
    template_dtypes: dict[str,str] | None = None,        # input_name -> placeholder string in `src`
)

MetalParameter(name: str, dtype: str, attr: str)   # e.g. ("gid","uint2","thread_position_in_grid")
```
(`_torch_metal_kernel.py:44-93`; `metal.py:36-52`)

Call it inside an `nn.Module` like a function, passing dispatch + output shapes per call:
```python
out = kernel(*args,
             threads_per_grid=(N,1,1),
             threads_per_thread_group=(T,1,1),
             result_shapes=[list(out_shape)])   # one entry per result; enables dynamic shapes
```
`result_shapes` is how Core AI "bakes in" output-shape-from-input-shape so the kernel works under dynamic
shapes (WWDC 325, ~line 511–518).

### Critical ordering — register kernels BEFORE add_exported_program
```python
converter = TorchConverter()
converter.register_custom_kernels([kernel])        # FIRST
converter.add_exported_program(ep, input_names=[...], output_names=[...])   # THEN
prog = converter.to_coreai(); prog.optimize()
```
WWDC 325 (~line 519): *"I register my custom kernels with the converter, then add the exported program as
before. The metal source gets embedded directly in the asset, a single artifact. The kernel travels with the
model."* This matches `register_custom_kernels` → builds a `register_torch_lowering` handler for
`coreai_metal_kernels::<name>.default` (`converter.py:1003-1033`).

## Minimal end-to-end (from the official guide)
```python
def torch_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:   # reference for shape inference
    return x + y

vadd = TorchMetalKernel(
    "vector_add", input_names=["x","y"], result_names=["output"],
    src="output[id] = x[id] + y[id];",
    torch_defn=torch_add,
    metal_params=[MetalParameter("id","uint","thread_position_in_grid")],
)

class M(nn.Module):
    def forward(self, x, y):
        return vadd(x, y, threads_per_grid=(x.shape[0],1,1),
                    threads_per_thread_group=(1,1,1), result_shapes=[list(x.shape)])

ep = torch.export.export(M().eval(), (x,y)).run_decompositions(get_decomp_table())
conv = TorchConverter(); conv.register_custom_kernels([vadd])
conv.add_exported_program(ep, input_names=["x","y"], output_names=["result"])
conv.to_coreai().optimize()
```

## How it lowers & ships
1. `torch.export` traces the `torch_defn` reference; the op is registered under `coreai_metal_kernels::` (`_torch_metal_kernel.py:265`).
2. `to_coreai()` lowers it to `coreai.metal4_kernel` via `CustomMetalKernel._construct_kernel_op` (`metal.py:423-479`): forces Metal-buffer backing on inputs/results, templates the MSL with the actual dtypes, bakes the result-shape computation.
3. The MSL is embedded in the `.aimodel`; the runtime dispatches it on the GPU at the right graph point.

## Constraints & gotchas
- **Metal-backed buffers** are forced on all I/O (`_ensure_metal_backed`, `metal.py:347-367`) — the GPU can't read host memory mid-kernel.
- **≤31 params** total (inputs + results + metal_params), `metal.py` `PARAMETER_LIMIT=31`.
- **dtypes** must be in the Metal map (`bf16→bfloat, f16→half, f32→float, si8→int8_t, ui8→uint8_t, ui32→uint, si32→int, i1→bool, …`, `metal.py:113-126`).
- **`template_dtypes`**: `{"A":"TYPE"}` replaces the placeholder string `"TYPE"` in `src` with input A's Metal dtype at compile time → one kernel serves multiple dtypes. Keys must be real inputs; placeholders must be unique (`metal.py:152-177`).
- **`torch_defn` validation**: every param annotated `Tensor|int|float|bool`, no `*args/**kwargs`, param count == `len(input_names)`, return `Tensor | list[Tensor] | tuple[Tensor,...]` with concrete length matching `result_names` (`_torch_metal_kernel.py:141-211`).
- Kernels are pure functions — no shared state / no execution-order dependence.
- **Never feed a kernel a transpose-derived tensor** (Apple 178056451, beaten 2026-08-17): the
  runtime binder requires every kernel edge to be an innermost-stride-1 dense MTLTensor, MPSGraph
  values carry no strides, and the layout planner picks the TRANSPOSED VIEW as the edge layout —
  an authoring-side `.contiguous()` cannot survive conversion (hard assert
  `[tensor.strides extentAtDimensionIndex:0] (N) should be 1`, GPUCustomMetalKernelOps.mm). Fix
  pattern: absorb the transpose into kernel INDEXING — feed the projection/conv-native logical
  shape (e.g. `[S, h*dk]` straight off a matmul, or `[C, S]` off a conv) and split (head, dim) with
  baked strides inside the MSL. Safe producer chains: matmul/conv/elementwise/reshape/slice.
  Worked example: `models/macos/qwen3_5_gdn_metal.py` (GDN scan kernel; also moves qk-l2norm
  in-kernel so the graph hands over raw activations).
- **Rank-3 buffer indexing + a DATA-DEPENDENT gather both lower + run on the GPU** (probe
  `ondevice/_moe_kernel_probe.py`). So a kernel can take an index tensor as an INPUT and read only
  the rows it points at — `W[m, n, e]` with `e = uint(IDX[slot])` reads only expert-slab `e` out of
  a `[E, N, M]` tensor (the `gather_qmm` MoE kernel, `models/macos/moe_metal.py` → the deferred
  MoE-over-read fix; LFM2.5-8B-A1B int8 39→141 tok/s). The `torch_defn` must stay fake-traceable:
  express the gather as `torch.index_select(W, 0, idx)` (shape-static), NEVER `int(idx[i])`
  (FakeTensor has no concrete value). Same kernel runs on M4 Max AND the iPhone 17 Pro A19 Pro GPU.
- **Per-slot vs shared activation in MoE.** Gate/up share the token `x` across the routed experts,
  but the **down projection feeds each expert its OWN gated activation** — so the kernel's `A` must
  be `[k, K]` (one row per slot, `A[c, slot]`), with `x` replicated k-wide for gate/up. Treating `A`
  as a single shared `[1, K]` row silently corrupts down (relative error ~1.3 = garbage).

## Performance patterns (WWDC 325 + gpu_rules.md + this project's measured results)
- **The win is killing dispatch overhead via fusion.** Per-op kernelization of small ops does NOT help — measured here: kernelizing attention q/k/v/o was *slower*; any single op-class ≤1.3 ms. The real lever is collapsing ~28 ops/layer into 1–3 mega-kernels (whole-layer fusion). (Detail: `ondevice/_macos_speed_RESULTS.md`.)
- **Custom int8 wins only on BIG memory-bound matmuls** (FFN, the 262144-vocab head): a fused int8 dequant-LUT matvec (reads only uint8 indices + a tiny per-group codebook, LUT gather fused into the matvec) beat both int8-MPSGraph and fp16-MPSGraph at int8 memory. Don't kernelize small projections (k/v).
- **Custom-vs-MPS is 0-for-3 whenever the MPS path is good** (2026-08-17/18): A19 matmul2d prefill (above), int4 M=1 gemv (custom matvec 2.02 vs MPS 1.45 ms on an 850M stack), and the static-M=9 quantized-matmul plateau (`qmm_metal.py`: 9 variants, best 3.19 vs stock 3.05 = tie — both sit at this GPU class's ~5 TFLOP/s practical streaming-gemm ceiling). Custom kernels win only where MPS is BAD at the op itself: int8 gemv, MoE dense-overread, numerically unstable fp16 scans.
- **The Metal compiler's formulation sensitivity is ~2x-level** (measured on `qmm_metal.py`): re-typing the same algorithm with generic index math (`(i & 255, i >> 8)` co-op loads, `m0 + mi` offset guards) cost +65% vs the literal per-loop form (3.19 → 5.33 ms). Micro-benchmark every "cosmetic" rewrite — never trust equivalence by eye.
- **Runtime-indexed private arrays land in stack memory, not registers.** Keep hot-path state in named half4/float4 vector variables; only literal-bound fully-unrollable loops may index an array (fp32 accumulators). fp16 4-term dots + fp32 accumulation are numerically fine for per-block-32 recipes (abs err ≤0.004 @ rms 1.4/layer) — judge them with a two-axis (rel AND abs) error gate, since pure rel-error gates false-fail on near-zero cancellation.
- **Prefer native SDPA on GPU** (`F.scaled_dot_product_attention`) — already fused; don't hand-roll it.
- Compute fp16 for throughput; use fp32 accumulation selectively for numerically sensitive reductions (this is the same fp32-accumulation root cause as the ANE Conv2d-1×1 fix).
- The "Optimize custom ML operations with Metal tensors" talk (WWDC 330) is the hand-tuned-kernel reference for squeezing a single kernel — TensorOps quantized matmul, cooperative tensors, FlashAttention, and the M5/A19 neural accelerator. Extracted: [`tensorops-quantized-kernels.md`](tensorops-quantized-kernels.md).
- **A `metal4_kernel` op is a fusion barrier, and its edges are materialized in the dtype/layout the kernel asks for.** A dtype cast on the boundary (e.g. `state.float()` in, `.half()` out) is a real `coreai.cast` op that blows the tensor up — hand large state/activation tensors across in their native dtype and accumulate fp32 in registers.
- **`blockwise_shift_scale` only fuses into the matmul when its shift is all-zero** — so symmetric int4 dequants cheaper than int8, but asymmetric (zero-point) int4 falls off the fast path (~4.6× slower on device). A fused affine-int4 matvec factors the zero point out (`acc += s*(dot − z*Σx)`) to keep the fast path AND the accuracy; it beats stock-asymmetric 3.3× Mac / ~5× device but only ties/+11% int8 (nibble-unpack ALU caps it). See [`ssm-hybrid-decode-int4-prefill-findings.md`](ssm-hybrid-decode-int4-prefill-findings.md) §3.
- **A decode-step SSM scan kernel is only ~3–8% faster than the plain torch graph** (paired A/B) — not worth the barrier + shape constraints. The SSM kernel win is *prefill* (chunked SSD, 13.7× on Mac), not decode. **Measurement protocol matters more than the kernel:** pair both arms in one process, interleave ≥8 reps, report median+spread; unpaired single-shot on a ±15%-drift machine will confirm anything. See the findings doc §1, §4, §7.
