#!/usr/bin/env python3
"""HF fp32 vision tower, stage by stage: transformers 5.19.0 `minicpmv4_7` on the CPU, one thread.

The bisect record for the re-authored tower (`minicpmv47_vision.py`): forward hooks on the HF tower
keep every stage of one prompt's image, so a parity miss can be pinned to the first stage that
departs. Input = the oracle's own `<P>__pixel_values` / `<P>__target_sizes` (the processor's packed
[1, 3, 14, 1024 * 14]), through `model.model.get_image_features` exactly as `oracle_minicpmv47.py`
ran it (fp32, eager attention, threads 1 / interop 1, offline).

Stages (rows x 1152 unless noted):
  embeddings       patch Conv2d + position embedding                [1024]
  layer_00..06     encoder layer outputs before the window merger   [1024]
  window_merger    `vit_merger` output (window order)               [256]
  layer_26         last encoder layer                               [256]
  post_layernorm                                                    [256]
  merger           final `merger` output = image_features           [64, 1024]
Constants from transformers' own functions (the module bakes copies of them):
  pos_ids            get_vision_nearest_position_ids([[32, 32]], 70)                 [1024]
  window_index       get_vision_window_index([[1, 32, 32]], 1, 2, 1)[0]              [1024]
  cu_window_seqlens  get_vision_window_index(...)[1]                                  [257]

Writes `<out>/vision_stages.npz` (keys `<P>__<stage>` + the constants) and `<out>/vision_stages.json`
(shape and absmax per stage; whether the merger stage equals the oracle's `<P>__image_features`).

    K/.venv-oracle/bin/python -I conversion/minicpmv47/dump_vision_stages.py [--prompts P1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import hf_snapshot, work_path  # noqa: E402

os.environ["HF_HUB_OFFLINE"] = "1"

HF_ID = "openbmb/MiniCPM-V-4.7-1B"
REVISION = "984a89aafcd5160164f4f0d04b4be0f4eec03cb8"
THREADS = 1
EARLY_LAYERS = range(0, 7)          # 0..insert_layer_id
LAST_LAYER = 26


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompts", default="P1", help="comma list of image prompts (default: P1)")
    ap.add_argument("--oracle-dir", default=str(work_path("_minicpmv47", "oracle")))
    ap.add_argument("--out-dir", default=str(work_path("_minicpmv47", "oracle")))
    args = ap.parse_args()

    import torch

    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(THREADS)
    import transformers
    from transformers import MiniCPMV4_7ForConditionalGeneration
    from transformers.vision_utils import get_vision_nearest_position_ids, get_vision_window_index

    t_all = time.perf_counter()
    snap = Path(hf_snapshot(HF_ID, revision=REVISION))
    oracle_dir, out_dir = Path(args.oracle_dir), Path(args.out_dir)
    z = np.load(oracle_dir / "oracle.npz")

    t0 = time.perf_counter()
    model = MiniCPMV4_7ForConditionalGeneration.from_pretrained(
        str(snap), dtype=torch.float32, attn_implementation="eager")
    model.eval()
    load_secs = time.perf_counter() - t0
    vt = model.model.vision_tower
    assert vt.config.insert_layer_id == 6 and len(vt.encoder.layers) == 27

    seen: dict[str, torch.Tensor] = {}

    def keep(name):
        def hook(mod, inputs, out):
            t = torch.cat(out, 0) if isinstance(out, (list, tuple)) else out
            seen[name] = t.detach().reshape(-1, t.shape[-1]).clone()
        return hook

    hooks = [vt.embeddings.register_forward_hook(keep("embeddings")),
             vt.vit_merger.register_forward_hook(keep("window_merger")),
             vt.encoder.layers[LAST_LAYER].register_forward_hook(keep(f"layer_{LAST_LAYER:02d}")),
             vt.post_layernorm.register_forward_hook(keep("post_layernorm")),
             model.model.merger.register_forward_hook(keep("merger"))]
    hooks += [vt.encoder.layers[i].register_forward_hook(keep(f"layer_{i:02d}")) for i in EARLY_LAYERS]

    arrays, record = {}, {}
    try:
        for P in args.prompts.split(","):
            pv = torch.from_numpy(z[f"{P}__pixel_values"])
            ts = torch.from_numpy(z[f"{P}__target_sizes"])
            assert ts.tolist() == [[32, 32]], ts
            seen.clear()
            t0 = time.perf_counter()
            with torch.inference_mode():
                feats = torch.cat(model.model.get_image_features(pv, ts).pooler_output, 0)
            secs = time.perf_counter() - t0
            assert torch.equal(feats, seen["merger"])
            want = torch.from_numpy(z[f"{P}__image_features"])
            stages = {}
            for name in ["embeddings", *[f"layer_{i:02d}" for i in EARLY_LAYERS], "window_merger",
                         f"layer_{LAST_LAYER:02d}", "post_layernorm", "merger"]:
                a = seen[name].numpy().astype(np.float32)
                arrays[f"{P}__{name}"] = a
                stages[name] = {"shape": list(a.shape), "absmax": float(np.abs(a).max())}
                print(f"[{P}] {name:15s} {str(tuple(a.shape)):13s} absmax {stages[name]['absmax']:.4f}", flush=True)
            record[P] = {"tower_seconds": secs, "stages": stages,
                         "merger_equals_oracle_image_features": bool(torch.equal(feats, want)),
                         "merger_vs_oracle_max_abs": float((feats - want).abs().max())}
            print(f"[{P}] tower {secs:.2f}s, merger == oracle image_features: "
                  f"{record[P]['merger_equals_oracle_image_features']} "
                  f"(max|d| {record[P]['merger_vs_oracle_max_abs']:.3e})", flush=True)
    finally:
        for h in hooks:
            h.remove()

    ts = torch.tensor([[32, 32]])
    pos_ids = get_vision_nearest_position_ids(ts, vt.embeddings.num_patches_per_side)
    window_index, cu_window = get_vision_window_index(
        torch.nn.functional.pad(ts, (1, 0), value=1), spatial_merge_size=1, window_size=2, patch_size=1)
    arrays["pos_ids"] = pos_ids.numpy().astype(np.int64)
    arrays["window_index"] = window_index.numpy().astype(np.int64)
    arrays["cu_window_seqlens"] = cu_window.numpy().astype(np.int32)
    consts = {"pos_ids": {"shape": list(pos_ids.shape), "min": int(pos_ids.min()), "max": int(pos_ids.max()),
                          "first_row": pos_ids[:32].tolist()},
              "window_index": {"shape": list(window_index.shape), "first_8": window_index[:8].tolist()},
              "cu_window_seqlens": {"shape": list(cu_window.shape), "dtype": str(cu_window.dtype),
                                    "first_4": cu_window[:4].tolist(), "last": int(cu_window[-1])}}
    print(f"[const] pos_ids {consts['pos_ids']} | window_index {consts['window_index']} | "
          f"cu_window_seqlens {consts['cu_window_seqlens']}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / "vision_stages.npz"
    np.savez(npz, **arrays)
    meta = {
        "schema": "minicpmv47-vision-stages/1",
        "purpose": "HF fp32 tower stages for bisecting the re-authored tower (minicpmv47_vision.py)",
        "source": {"hf_id": HF_ID, "revision": REVISION, "snapshot": str(snap)},
        "reference": {"class": type(model).__name__, "dtype": str(model.dtype), "device": "cpu",
                      "attn_implementation": model.config._attn_implementation,
                      "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                      "load_seconds": load_secs},
        "input": "oracle.npz <P>__pixel_values (packed [1, 3, 14, 14336]) + <P>__target_sizes [[32, 32]]",
        "prompts": record,
        "constants": consts,
        "npz": {"path": str(npz), "bytes": npz.stat().st_size, "sha256": sha256(npz), "keys": sorted(arrays)},
        "oracle_npz_sha256": sha256(oracle_dir / "oracle.npz"),
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "transformers": transformers.__version__, "numpy": np.__version__,
                     "platform": platform.platform()},
        "script": {"path": "conversion/minicpmv47/dump_vision_stages.py", "sha256": sha256(Path(__file__))},
        "wall_seconds": time.perf_counter() - t_all,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "vision_stages.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[dump] {npz} ({npz.stat().st_size / 1e6:.1f} MB), wall {meta['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
