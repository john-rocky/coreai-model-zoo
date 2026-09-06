# EXC_BREAKPOINT (SIGTRAP) at the first execute — the data-indexed in-graph KV write does not lower on the OS 27 beta MPSGraph (host-cache and input-mask escapes)

On the WWDC26 betas (macOS 27 / iOS 27) the **fixed-shape / ANE decode path** — the one that writes each
new KV column in-graph with `slice_update` at a runtime `in_step` index (Apple's documented
`export/ios.py` + `CoreAIStaticShapeEngine` recipe) — **does not lower on the MPSGraph backend**:

- **Mac GPU**: `EXC_BREAKPOINT` / SIGTRAP at the first execute (process exit 133).
- **iPhone GPU**: SIGSEGV at the first execute (the graph loads + specializes, *then* crashes).
- **iPhone ANE**: `MPSGraphExecutable.mm` → "MLIR pass manager failed" (SIGABRT); corrupts the ANE
  compile cache (next load = ENOENT).

Conversion **succeeds** — it is load + execute that dies.

## The decisive isolation

Same attention block, same `slice_update`, same SDPA, exported three ways that differ in **only the
KV-write column index**:

| write index `begin` | shapes | result |
|---|---|---|
| shape **symint** (`position_ids.shape[-1] − query_len`, the `update_and_fetch` path) | dynamic | **runs** ✅ |
| runtime **tensor** (`in_step` scalar input) | dynamic | **SIGTRAP** ✗ |
| runtime **tensor** (`in_step` scalar input) | static | **SIGTRAP** ✗ |

So it is not the mask, not static-ness, not the model — flipping the begin-index *source* (shape symint →
runtime tensor) alone flips run → crash. Model-agnostic: every model shares the one
`KVCache.update_and_fetch` helper. (Filed: Apple Feedback FB23024751 · issue [apple/coreai-models#5](https://github.com/apple/coreai-models/issues/5) · repro gist [john-rocky](https://gist.github.com/john-rocky/1fd6add76b3d5393ebc44fac52ce6b27).)

The catch: the **dynamic symint path runs but re-specializes per sequence length** (the slow path — a new
`position_ids` length recompiles, ~27 ms → ~1.9 s/step). The fast fixed-shape path is exactly the one that
crashes.

## Workaround — host-cache (no in-graph indexed write)

Express the KV cache as plain model **input/output** instead of a Core AI state, and remove the indexed
write entirely:

- append the new token's K/V in-graph with `torch.cat` (past ++ current),
- attend with a masked SDPA over the concatenated keys (valid past + current marked by an explicit mask),
- the **host** writes the new column back between steps (plain numpy / `[Float16]`).

Only MPSGraph-safe ops (masked SDPA over plain inputs + `cat`) — no state, no `slice_update`. Numerically
identical to the stateful core (8/8 top-1 vs HF). Runs on **Mac GPU, iPhone GPU (full model), and iPhone
ANE (chunked)**. For the ANE, split into ≤~8-layer chunks (the 35-layer monolith OOMs the first-run ANE
compile).

- Win: unblocks on-device decode on the beta *today*, no waiting for the MPSGraph fix.
- Cost: a host round-trip per step + losing Core AI's in-place state. Re-fold to the state path once the
  runtime-tensor `begin` index lowers — or use the input-mask escape below, which restores the state
  path without waiting.

## Update (2026-06-10) — the input-mask escape: stateful KV WITHOUT the fix

Further isolation shows the trigger is **narrower** than "data-tensor write index": what crashes is
deriving the write position **in-graph** from runtime data. Hand the graph the position as a
pre-computed mask **input** and the numerically identical write lowers and runs:

```python
# host builds a one-hot fp16 write_mask[ctx] per step (1.0 at the write column) — 2 KB
sl = cache[slot]                          # state, compile-time slot index
m  = write_mask.reshape(1, 1, ctx, 1)
sl.copy_(sl * (1 - m) + col * m)          # exact one-hot select; NO data-derived index anywhere
```

Five formulations isolated on the beta Mac GPU (each in its own process, multi-step state values
verified exact): constant-mask blend ✅ · **input-mask blend ✅** · shift-append
(`cache ← cat(cache[1:], col)`) ✅ · input-mask blend into one slot of a packed
`[n_slots,…]` state ✅ (both slot-view and whole-state forms) — while the same blend with the
one-hot computed in-graph (`arange == in_step`) crashes exactly like `slice_update`.

Proven at full scale: a 35-layer Gemma 4 E2B static decode core with the blend write (everything
else identical to the official fixed-shape recipe) exports to int8 and runs **8/8 greedy-exact on
the beta macOS GPU** — the first fixed-shape *stateful* core that executes on this beta at all.
You get fixed shapes (no per-step respecialization, flat memory) **and** Core AI states (no host
KV round-trip) at the cost of one tiny mask input per step.

Status: Mac GPU verified; iPhone GPU / ANE re-isolation pending (the crash was platform-agnostic,
the escape should be too — but the ANE's MLIR path is a different lowering, so verify before
betting a port on it).

See [`stateful-kv-cache.md`](stateful-kv-cache.md) for the state-based path this replaces, and
[`conversion-guide.md`](conversion-guide.md) for the `slice_update` / `remove_functionalization` details.


## Strategy note — de-confusing "the ANE wall" (2026-06-10 re-verification, added 2026-07-14)

Three separate facts were historically conflated into "ANE is walled by a SIGSEGV, so we pivoted to GPU":

1. **ANE is correct, not broken**: gemma4 E2B ran **8/8 exact on the device ANE** once fp16 numerics
   were fixed (`[x,-x]` LayerNorm trick + fp32 accumulation for Conv2d-1×1). The earlier "ANE 0/8" read
   was retracted.
2. **ANE is speed-capped, not correctness-capped** (~6 tok/s at the time): a 262k-vocab head plus
   host-cache KV re-feed every step. Lifting it needs stateful KV (this very bug) + a reduced head + AOT.
3. **The sound reason the zoo standardized on GPU** is that custom Metal kernels
   (`coreai_torch.TorchMetalKernel` → `coreai.metal4_kernel`) are **GPU-only by construction** — the ANE
   runs fixed hardware ops, never hand-written MSL. If the speed lever is fused kernels, that is a GPU
   play regardless of this bug.

Caveat kept honest: on-device the GPU lead was small at measurement time (iPhone GPU 7.4 vs ANE 5.9 tok/s);
the big GPU numbers are Mac. Keep the ANE path alive — it is the energy-efficient lane MLX/llama.cpp
cannot touch, and it revives whenever the FB above lifts.
