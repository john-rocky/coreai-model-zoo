"""Graph callables for host_forecast: the re-authored torch core, or a Core AI bundle.

Both take NumPy (patches, cos, sin, attn_mask) and return NumPy float32 quantiles (R, h*32, 21).
A Core AI graph has a fixed row count; a call with a different R is split into calls of the
graph's R (R = 2 inputs on an R = 1 graph -> two calls).
"""
import asyncio
from pathlib import Path

import numpy as np

INPUT_NAMES = ["patches", "cos", "sin", "attn_mask"]
OUTPUT_NAMES = ["quantiles"]


class TorchCore:
    def __init__(self, core, dtype=None):
        import torch
        self.torch, self.core = torch, core
        self.dtype = dtype or torch.float32

    def __call__(self, patches, cos, sin, attn_mask):
        t = self.torch
        with t.inference_mode():
            out = self.core(*[t.tensor(np.ascontiguousarray(v)).to(self.dtype) for v in (patches, cos, sin, attn_mask)])
        return out.float().numpy()


def specialization(unit):
    import coreai.runtime as rt
    if unit == "cpu":
        return rt.SpecializationOptions.cpu_only()
    kind = {"gpu": rt.ComputeUnitKind.gpu, "neural_engine": rt.ComputeUnitKind.neural_engine}[unit]()
    return rt.SpecializationOptions.from_preferred_compute_unit_kind(kind)


class CoreAIGraph:
    """One loaded `.aimodel` (JIT) or `.aimodelc` (AOT). Keeps the AIModel alive (the function alone can be
    collected and then returns garbage)."""

    def __init__(self, path, unit, dtype, rows):
        import coreai.runtime as rt
        self.rt, self.path, self.unit, self.rows = rt, Path(path), unit, rows
        self.np_dtype = np.float16 if dtype == "float16" else np.float32
        self.spec_used, self.load_error = unit, None
        try:
            self.model = asyncio.run(rt.AIModel.load(self.path, specialization(unit)))
        except Exception as e:  # an AOT asset can refuse an explicit unit; say so and try the default options
            if unit == "cpu" or self.path.suffix != ".aimodelc":
                raise
            self.load_error = f"{type(e).__name__}: {e}"[:500]
            self.spec_used = "default (explicit load failed)"
            self.model = asyncio.run(rt.AIModel.load(self.path, rt.SpecializationOptions.default()))
        self.fn = self.model.load_function("main")

    def feeds(self, patches, cos, sin, attn_mask):
        rt, d = self.rt, self.np_dtype
        return {n: rt.NDArray(np.ascontiguousarray(v.astype(d)))
                for n, v in zip(INPUT_NAMES, (patches, cos, sin, attn_mask))}

    async def run(self, feeds):
        out = await self.fn(feeds)
        return out["quantiles"].numpy().astype(np.float32)

    def __call__(self, patches, cos, sin, attn_mask):
        r = patches.shape[0]
        if r == self.rows:
            return asyncio.run(self.run(self.feeds(patches, cos, sin, attn_mask)))
        assert r % self.rows == 0, f"{r} rows on a {self.rows}-row graph"
        parts = [asyncio.run(self.run(self.feeds(patches[i:i + self.rows], cos, sin, attn_mask)))
                 for i in range(0, r, self.rows)]
        return np.concatenate(parts, axis=0)
