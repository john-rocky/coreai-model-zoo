"""Read a ``capture_oracle.py`` dump (``oracle/<tag>/``) for the DiT gates. numpy/torch only.

``t(s)`` is the exact fp32 value the transformer received — ``timesteps.f32[s] / 1000`` in fp32,
as the pipeline computes it — not the 7-decimal ``meta["t"]`` (which is checked against it).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class Oracle:
    def __init__(self, path):
        self.dir = Path(path)
        self.meta = json.load(open(self.dir / "meta.json"))
        m = self.meta
        self.N, self.L, self.steps = int(m["N"]), int(m["L"]), int(m["steps"])
        _, self.H, self.W = (int(v) for v in m["img_shapes"][-1])
        assert self.H * self.W == self.N, (m["img_shapes"], self.N)
        pe = self._raw("prompt_embeds")
        self.ctx = pe.size // self.L
        self.C = self._raw("latent_0").size // self.N
        self.prompt_embeds = torch.from_numpy(pe.reshape(1, self.L, self.ctx))
        self.timesteps = self._raw("timesteps")
        assert self.timesteps.shape[0] == self.steps
        for s in range(self.steps):                    # meta t is rounded to 7 decimals
            assert abs(float(self.t(s)) - m["t"][s]) <= 5.1e-8, (s, float(self.t(s)), m["t"][s])

    def _raw(self, name) -> np.ndarray:
        return np.fromfile(self.dir / f"{name}.f32", "<f4")

    def t(self, s: int) -> torch.Tensor:
        return torch.from_numpy(self.timesteps[s:s + 1].copy()) / 1000          # fp32 [1]

    def latent(self, s: int) -> torch.Tensor:
        return torch.from_numpy(self._raw(f"latent_{s}").reshape(1, self.N, self.C))

    def vel(self, s: int) -> torch.Tensor:
        return torch.from_numpy(self._raw(f"vel_{s}").reshape(1, self.N, self.C))

    def describe(self) -> str:
        m = self.meta
        return (f"{self.dir.name}: size {m['size']} N={self.N} ({self.H}x{self.W}) L={self.L} C={self.C} "
                f"ctx={self.ctx} steps={self.steps} seed {m['seed']} kv_cache={m.get('kv_cache')} "
                f"snapshot {m['snapshot']}")


def parse_steps(spec: str, n: int) -> list[int]:
    if spec == "all":
        return list(range(n))
    return [int(v) if int(v) >= 0 else n + int(v) for v in spec.split(",")]
