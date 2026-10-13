#!/usr/bin/env python3
"""MiniCPM-V-4.7 host side for the shipped form (one 448x448 thumbnail, no slices), in NumPy + Pillow.

  preprocess(path)  -> pixel_values [1, 3, 448, 448] float32: RGB, Pillow BICUBIC to 448x448 (aspect
                       not kept), (x / 255 - 0.5) / 0.5, CHW. The tower's input (`minicpmv47_vision.py`).
  static_inputs(ids) -> the decoder's ids and its two static scalars: `host_static_inputs` of
                       `minicpmv47_vl_pipelined.py`, called, not re-written.

The HF processor (`MiniCPMV4_6ImageProcessor`, torchvision backend) hands the same pixels packed as
[1, 3, 14, 1024 * 14]; `unpack_processor_pixels` turns that into [1, 3, 448, 448] (a permutation).

Run as a script it is the host gate: `preprocess` on each prompt's fixture (`oracle/fixtures/*.448.png`)
against the oracle's processor output `oracle.npz <P>__pixel_values`, max |d| <= 1e-6 on P1..P4, plus
a red arm (BGR channel order) that must fail on every prompt, and `preprocess` on each fixture's
source image (the resize included) against the fixture -> `oracle/host_preprocess.json`.

    coreai-models/.venv/bin/python -I conversion/minicpmv47/host.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SIDE = 448
PATCH = 14
GRID = SIDE // PATCH                   # 32


def preprocess(path, bgr: bool = False) -> np.ndarray:
    """Image file -> pixel_values [1, 3, 448, 448] float32. `bgr` exists for the gate's red arm."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB").resize((SIDE, SIDE), Image.Resampling.BICUBIC)
        x = np.asarray(im, dtype=np.float32)                          # [448, 448, 3]
    if bgr:
        x = x[..., ::-1]
    x = (x / 255.0 - 0.5) / 0.5
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def unpack_processor_pixels(packed: np.ndarray, grid_h: int = GRID, grid_w: int = GRID) -> np.ndarray:
    """The processor's [1, 3, 14, grid_h * grid_w * 14] (patch k = row-major (i, j)) -> [1, 3, H, W]."""
    b, c = packed.shape[:2]
    x = packed.reshape(b, c, PATCH, grid_h, grid_w, PATCH).transpose(0, 1, 3, 2, 4, 5)
    return np.ascontiguousarray(x.reshape(b, c, grid_h * PATCH, grid_w * PATCH))


def static_inputs(ids):
    """(mapped ids [n] int32, rope_shift_start [1] int32, rope_shift_amount [1] int32) — the decoder's
    host contract, from `minicpmv47_vl_pipelined.host_static_inputs` (torch)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from minicpmv47_vl_pipelined import host_static_inputs

    return host_static_inputs(ids)


# -- gate ---------------------------------------------------------------------------------------

PROMPT_IMAGES = {"P1": "earth", "P2": "saturn_v", "P3": "synthetic", "P4": "earth"}
BAR = 1e-6


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gate(oracle_dir: Path, out: Path) -> dict:
    import PIL

    z = np.load(oracle_dir / "oracle.npz")
    fixtures = oracle_dir / "fixtures"
    fx = json.loads((fixtures / "fixtures.json").read_text())
    rec = {"prompts": {}, "red_bgr": {}, "from_source": {}}
    for P, stem in PROMPT_IMAGES.items():
        want = unpack_processor_pixels(z[f"{P}__pixel_values"])
        got = preprocess(fixtures / f"{stem}.448.png")
        d = float(np.abs(got.astype(np.float64) - want).max())
        ids = z[f"{P}__input_ids"]
        _, start, amount = static_inputs(ids)
        rec["prompts"][P] = {"image": f"{stem}.448.png", "shape": list(got.shape), "dtype": str(got.dtype),
                             "max_abs_diff": d, "bitwise": bool(np.array_equal(got, want)),
                             "gate": "PASS" if d <= BAR else "FAIL",
                             "rope_shift_start": int(start[0]), "rope_shift_amount": int(amount[0])}
        bad = float(np.abs(preprocess(fixtures / f"{stem}.448.png", bgr=True).astype(np.float64) - want).max())
        rec["red_bgr"][P] = {"max_abs_diff": bad, "gate": "PASS" if bad <= BAR else "FAIL"}
        print(f"[{P}] {stem}.448.png max|d| {d:.3e} bitwise {rec['prompts'][P]['bitwise']} -> "
              f"{rec['prompts'][P]['gate']} | start {int(start[0])} amount {int(amount[0])} | red (BGR) max|d| "
              f"{bad:.3f} -> {rec['red_bgr'][P]['gate']}", flush=True)
    for stem, info in fx["images"].items():
        src = fixtures / info["file"]
        a, b = preprocess(src), preprocess(fixtures / info["448"]["file"])
        rec["from_source"][stem] = {"source": info["file"], "source_size": info["448"]["source_size"],
                                    "bitwise_equal_fixture": bool(np.array_equal(a, b)),
                                    "max_abs_diff": float(np.abs(a - b).max())}
        print(f"[source] {info['file']} {info['448']['source_size']} -> 448: equal to the fixture "
              f"{rec['from_source'][stem]['bitwise_equal_fixture']} (max|d| {rec['from_source'][stem]['max_abs_diff']:.3e})",
              flush=True)
    ok = all(r["gate"] == "PASS" for r in rec["prompts"].values())
    red_ok = all(r["gate"] == "FAIL" for r in rec["red_bgr"].values())
    record = {
        "schema": "minicpmv47-host-preprocess/1",
        "purpose": "host preprocess (Pillow BICUBIC 448, (x/255-0.5)/0.5, CHW) vs the oracle's processor pixels",
        "result": "PASS" if ok and red_ok else "FAIL",
        "bar": f"max|d| <= {BAR} on P1..P4; the BGR red arm FAILs on all four",
        "summary": {"pass": f"{sum(r['gate'] == 'PASS' for r in rec['prompts'].values())}/{len(rec['prompts'])}",
                    "max_abs_diff": max(r["max_abs_diff"] for r in rec["prompts"].values()),
                    "red_bgr_fail": f"{sum(r['gate'] == 'FAIL' for r in rec['red_bgr'].values())}/{len(rec['red_bgr'])}",
                    "from_source_bitwise": f"{sum(r['bitwise_equal_fixture'] for r in rec['from_source'].values())}/"
                                           f"{len(rec['from_source'])}"},
        **rec,
        "oracle_npz_sha256": _sha256(oracle_dir / "oracle.npz"),
        "script": {"path": "conversion/minicpmv47/host.py", "sha256": _sha256(Path(__file__))},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pillow": PIL.__version__,
                     "platform": platform.platform()},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"[host] {record['result']} {json.dumps(record['summary'])}\n[host] -> {out}", flush=True)
    return record


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _paths import work_path

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle-dir", default=str(work_path("_minicpmv47", "oracle")))
    ap.add_argument("--out", default=str(work_path("_minicpmv47", "oracle", "host_preprocess.json")))
    args = ap.parse_args()
    t0 = time.perf_counter()
    gate(Path(args.oracle_dir), Path(args.out))
    print(f"[host] wall {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
