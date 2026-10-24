#!/usr/bin/env python3
"""The Swift host's box_pe values against the package's torch CPU fp32 arithmetic and the NumPy copy.

Reads the files `cutout-cli boxpe --out <dir>` wrote (w.f32.bin = every distinct fp32 width a 0-1000 box gives,
ew.f32.bin = Swift's fp32(0.2) * logf(w) + fp32(0.5) for each, values.f32.bin = x1/y1 values then the distinct ew
values, features.f32.bin = Swift's 64 features per value) and compares, bit for bit:
  1  ew: Swift vs torch (`0.2 * w.log() + 0.5`, the package's BoxFourierEncoder line) vs numpy (`box_fourier_np`)
  2  features: Swift vs numpy (float64 sin / cos of the fp32 phase, then fp32); torch's fp32 sin / cos recorded
  3  the full box path on every (a, b) pair: max(fp32(b/1000) - fp32(a/1000), 1e-6) recomputed in numpy, its ew looked
     up in Swift's table == torch's ew of the same width

    python3 -I check_boxpe.py <dir> <out.json>
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

d = Path(sys.argv[1])
w = np.fromfile(d / "w.f32.bin", np.float32)
ew_swift = np.fromfile(d / "ew.f32.bin", np.float32)
values = np.fromfile(d / "values.f32.bin", np.float32)
feats_swift = np.fromfile(d / "features.f32.bin", np.float32).reshape(-1, 64)

# 1 the encoded side
wt = torch.from_numpy(w.copy())
ew_torch = (0.2 * wt.log() + 0.5).numpy()
ew_np = np.float32(0.2) * np.log(w) + np.float32(0.5)
bits = lambda a: a.view(np.uint32)
mis_torch = np.flatnonzero(bits(ew_swift) != bits(ew_torch))
mis_np = np.flatnonzero(bits(ew_swift) != bits(ew_np))
np_vs_torch = np.flatnonzero(bits(ew_np) != bits(ew_torch))

# 2 the features (box_fourier_np's arithmetic) and torch's FourierPE1D
freqs = (2.0 ** np.arange(32)).astype(np.float32)
phase = (np.float32(2.0 * math.pi) * values)[:, None] * freqs
p64 = phase.astype(np.float64)
feats_np = np.concatenate([np.sin(p64), np.cos(p64)], axis=-1).astype(np.float32)
mis_feat = np.argwhere(bits(feats_swift) != bits(feats_np))
vt = torch.from_numpy(values.copy())
ph_t = 2.0 * math.pi * vt[..., None].float() * (2.0 ** torch.arange(32, dtype=torch.float32))
feats_t = torch.cat([ph_t.sin(), ph_t.cos()], dim=-1).numpy()
phase_bits_equal = bool(np.array_equal(bits(ph_t.numpy()), bits(phase)))
feat_t_diff = np.abs(feats_t.astype(np.float64) - feats_swift.astype(np.float64))
ulp = np.abs(bits(feats_t).astype(np.int64) - bits(feats_swift).astype(np.int64))

# 3 every (a, b) pair through the full path
unit = (np.arange(1001, dtype=np.float64) / 1000.0).astype(np.float32)
a_idx, b_idx = np.triu_indices(1001, k=1)
w_pairs = np.maximum(unit[b_idx] - unit[a_idx], np.float32(1e-6))
table = dict(zip(bits(w).tolist(), bits(ew_swift).tolist()))
pairs_missing = int(sum(1 for x in np.unique(bits(w_pairs)).tolist() if x not in table))
ew_pairs_torch = (0.2 * torch.from_numpy(w_pairs).log() + 0.5).numpy()
lookup = np.array([table.get(x, 0) for x in bits(w_pairs).tolist()], np.uint32)
pairs_mismatch = int((lookup != bits(ew_pairs_torch)).sum())

out = {
    "schema": "wevisone-r4-boxpe/1",
    "torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0],
    "widths": int(w.size), "values": int(values.size), "pairs": int(w_pairs.size),
    "ew_swift_vs_torch_mismatch": int(mis_torch.size),
    "ew_swift_vs_numpy_mismatch": int(mis_np.size),
    "ew_numpy_vs_torch_mismatch": int(np_vs_torch.size),
    "ew_mismatch_examples": [{"w": float(w[i]), "swift": float(ew_swift[i]), "torch": float(ew_torch[i]),
                              "numpy": float(ew_np[i])} for i in mis_torch[:20].tolist()],
    "features_swift_vs_numpy_mismatch": int(mis_feat.shape[0]),
    "features_mismatch_examples": [{"value": float(values[i]), "k": int(j), "swift": float(feats_swift[i, j]),
                                    "numpy": float(feats_np[i, j])} for i, j in mis_feat[:20].tolist()],
    "phase_torch_vs_numpy_bit_equal": phase_bits_equal,
    "features_torch_fp32_vs_swift": {"max_abs_d": float(feat_t_diff.max()), "ulp_max": int(ulp.max()),
                                     "ulp_1_count": int((ulp == 1).sum()), "ulp_gt1_count": int((ulp > 1).sum()),
                                     "total": int(ulp.size)},
    "pairs_widths_missing_from_table": pairs_missing,
    "pairs_ew_mismatch_vs_torch": pairs_mismatch,
    # the bar is torch's encoded value on every width and pair, and the NumPy arithmetic for the features; the Swift
    # host carries torch's values for the widths where libm's logf rounds apart, so it differs from NumPy's ew there
    "verdict": "PASS" if (mis_torch.size == 0 and mis_feat.shape[0] == 0
                          and pairs_missing == 0 and pairs_mismatch == 0) else "FAIL",
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: v for k, v in out.items() if not k.endswith("examples")}, indent=1))
