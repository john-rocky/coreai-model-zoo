#!/usr/bin/env python3
"""fp32 torch parity: the re-authored MiniCPM-V-4.7 vision tower vs the HF fp32 oracle (P1..P4).

`minicpmv47_vision.MiniCPMV47Vision`, loaded from the pinned checkpoint (459 tensors, strict), on the
oracle's own pixels (`oracle.npz <P>__pixel_values`, the processor's packed [1, 3, 14, 14336],
unpacked to [1, 3, 448, 448] — an exact permutation, checked by repacking):

  (a) reference path (run-time position ids / window index / cu_seqlens split, as HF)
  (b) baked path (`bake_constants(32)`: int32 buffers, one batched window attention)

  tower    image_features [64, 1024] vs `<P>__image_features`: per-token cosine (mean / min, float64),
           max |d|, the reference absmax; and max |(a) - (b)|.
  stages   on P1, both paths, every stage `dump_vision_stages.py` kept (embeddings, layer 0..6,
           window merger, layer 26, post_layernorm, merger): per-row cosine min, max |d| — the
           first stage that departs (cos < 0.9999, or max |d| > 1e-3 x max(1, stage absmax)) is
           where a miss starts.
  consts   the baked position ids and window index vs transformers' own functions (the dump).
  Bars (round-2 launch): load 459/459, missing / unexpected / meta 0; (a) and (b) per-token cos
  min >= 0.9999 and max |d| <= 1e-3 on P1..P4; max |(a) - (b)| <= 1e-5.

    QW coreai-models/.venv/bin/python -I conversion/minicpmv47/parity_vision.py [--threads 1]
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _paths import hf_snapshot, work_path  # noqa: E402

HF_ID = "openbmb/MiniCPM-V-4.7-1B"
REVISION = "984a89aafcd5160164f4f0d04b4be0f4eec03cb8"
PROMPTS = ("P1", "P2", "P3", "P4")
STAGE_PROMPT = "P1"
BAR_COS = 0.9999
BAR_MAX_ABS = 1e-3
BAR_A_MINUS_B = 1e-5
STAGES = ("embeddings", *[f"layer_{i:02d}" for i in range(7)], "window_merger", "layer_26",
          "post_layernorm", "merger")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compare(mine, ref) -> dict:
    """Row-wise: rows [n, d] (float64 cosine), max |d|, the reference absmax."""
    m = np.asarray(mine, dtype=np.float64).reshape(-1, mine.shape[-1])
    r = np.asarray(ref, dtype=np.float64).reshape(-1, ref.shape[-1])
    cos = (m * r).sum(-1) / (np.linalg.norm(m, axis=-1) * np.linalg.norm(r, axis=-1))
    d, a = float(np.abs(m - r).max()), float(np.abs(r).max())
    return {"rows": int(m.shape[0]), "cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
            "argmin_row": int(cos.argmin()), "max_abs_diff": d, "ref_absmax": a, "max_abs_diff_rel": d / a}


def departs(row: dict) -> bool:
    """A stage departs when its row cosine drops under the bar or its max |d| exceeds the tower
    bar scaled by the stage's own magnitude (the residual stream reaches absmax ~1.4e3)."""
    return row["cos_min"] < BAR_COS or row["max_abs_diff"] > BAR_MAX_ABS * max(1.0, row["ref_absmax"])


def stage_hooks(model, seen: dict) -> list:
    """Forward hooks on the module's stages; the window merger's output = layer 7's input."""
    vt = model.vision_tower
    layers = vt.encoder.layers

    def keep(name, pre=False):
        def hook(mod, args, out=None):
            t = args[0] if pre else out
            seen[name] = t.detach().reshape(-1, t.shape[-1]).clone()
        return hook

    hs = [vt.embeddings.register_forward_hook(keep("embeddings")),
          layers[7].register_forward_pre_hook(keep("window_merger", pre=True)),
          layers[26].register_forward_hook(keep("layer_26")),
          vt.post_layernorm.register_forward_hook(keep("post_layernorm")),
          model.merger.mlp[0].register_forward_hook(keep("merger"))]
    hs += [layers[i].register_forward_hook(keep(f"layer_{i:02d}")) for i in range(7)]
    return hs


def run(args) -> dict:
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    from minicpmv47_vision import GRID, MiniCPMV47Vision, pack_patches, unpack_patches

    oracle_dir = Path(args.oracle_dir)
    z = np.load(oracle_dir / "oracle.npz")
    st = np.load(oracle_dir / "vision_stages.npz")
    t0 = time.perf_counter()
    model = MiniCPMV47Vision.from_checkpoint(hf_snapshot(HF_ID, revision=REVISION), torch.float32)
    load_seconds = time.perf_counter() - t0
    rep = model.load_report
    print(f"[load] {rep['checkpoint_tower_merger_keys']}/{rep['expected']} tensors, missing "
          f"{len(rep['missing'])} unexpected {len(rep['unexpected'])} meta {len(rep['on_meta'])}, "
          f"{load_seconds:.1f}s", flush=True)

    pixels = {}
    for P in PROMPTS:
        packed = torch.from_numpy(z[f"{P}__pixel_values"])
        assert z[f"{P}__target_sizes"].tolist() == [[GRID, GRID]]
        pv = unpack_patches(packed, GRID, GRID)
        assert tuple(pv.shape) == (1, 3, 448, 448) and torch.equal(pack_patches(pv), packed)
        pixels[P] = pv

    out = {"prompts": {P: {} for P in PROMPTS}, "stages": {}, "paths": list(args.paths)}
    feats = {"a": {}, "b": {}}
    for path in args.paths:
        if path == "b":
            model.bake_constants(GRID)
            out["constants"] = {
                "pos_ids_equal_hf": bool(np.array_equal(model.pos_ids.numpy(), st["pos_ids"])),
                "window_index_equal_hf": bool(np.array_equal(model.window_index.numpy(), st["window_index"])),
                "hf_cu_window_seqlens_is_range_step_4": bool(
                    st["cu_window_seqlens"].tolist() == list(range(0, GRID * GRID + 1, 4))),
                "buffers": {n: [list(b.shape), str(b.dtype)] for n, b in model.named_buffers()},
            }
            print(f"[bake] {out['constants']}", flush=True)
        fwd = model.forward_reference if path == "a" else model.forward_baked
        for P in PROMPTS:
            seen: dict = {}
            hooks = stage_hooks(model, seen) if P == STAGE_PROMPT else []
            try:
                t0 = time.perf_counter()
                with torch.inference_mode():
                    f = fwd(pixels[P])
                secs = time.perf_counter() - t0
            finally:
                for h in hooks:
                    h.remove()
            feats[path][P] = f.numpy().astype(np.float32)
            c = compare(feats[path][P], z[f"{P}__image_features"])
            out["prompts"][P][path] = {**c, "seconds": secs}
            print(f"[{P}] ({path}) cos mean {c['cos_mean']:.9f} min {c['cos_min']:.9f} max|d| "
                  f"{c['max_abs_diff']:.3e} (ref absmax {c['ref_absmax']:.3f}) {secs:.2f}s", flush=True)
            if P == STAGE_PROMPT:
                rows = {}
                for name in STAGES:
                    rows[name] = compare(seen[name].numpy(), st[f"{P}__{name}"])
                    rows[name]["shape"] = list(seen[name].shape)
                out["stages"][path] = rows
                first = next((n for n in STAGES if departs(rows[n])), None)
                out["stages"][f"{path}_first_departing"] = first
                print(f"[{P}] ({path}) stages: " + " | ".join(
                    f"{n} {rows[n]['cos_min']:.7f}/{rows[n]['max_abs_diff']:.1e}" for n in STAGES), flush=True)
                print(f"[{P}] ({path}) first stage outside the bar: {first}", flush=True)
    for P in PROMPTS if args.paths == ("a", "b") else ():
        d = float(np.abs(feats["a"][P].astype(np.float64) - feats["b"][P]).max())
        out["prompts"][P]["a_minus_b_max_abs"] = d
        out["prompts"][P]["a_equals_b_bitwise"] = bool(np.array_equal(feats["a"][P], feats["b"][P]))
        print(f"[{P}] max|(a)-(b)| {d:.3e} bitwise {out['prompts'][P]['a_equals_b_bitwise']}", flush=True)
    out["load_report"] = rep
    out["load_seconds"] = load_seconds
    meta = json.loads((oracle_dir / "oracle.json").read_text())
    out["oracle"] = {"npz_sha256": sha256(oracle_dir / "oracle.npz"), "versions": meta["versions"],
                     "threads": meta["oracle"]["threads"], "revision": meta["source"]["revision"],
                     "vision_stages_npz_sha256": sha256(oracle_dir / "vision_stages.npz")}
    return out


def versions() -> dict:
    import importlib.metadata as md

    import torch
    v = {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
         "platform": platform.platform(), "threads": torch.get_num_threads(),
         "interop_threads": torch.get_num_interop_threads()}
    for pkg in ("transformers", "safetensors", "coreai-torch"):
        try:
            v[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            v[pkg] = None
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--paths", default="a,b", help="a (reference), b (baked) or a,b (default; a - b needs both)")
    ap.add_argument("--oracle-dir", default=str(work_path("_minicpmv47", "oracle")))
    ap.add_argument("--out", default=str(work_path("_minicpmv47", "oracle", "parity_vision.json")))
    args = ap.parse_args()
    args.paths = tuple(args.paths.split(","))
    assert args.paths in (("a",), ("b",), ("a", "b")), args.paths
    t_all = time.perf_counter()
    res = run(args)
    pr = res["prompts"]
    rep = res["load_report"]
    load_ok = (rep["checkpoint_tower_merger_keys"] == rep["expected"] == 459 and not rep["missing"]
               and not rep["unexpected"] and not rep["on_meta"])
    tower_ok = {p: all(pr[P][p]["cos_min"] >= BAR_COS and pr[P][p]["max_abs_diff"] <= BAR_MAX_ABS for P in PROMPTS)
                if p in args.paths else None for p in ("a", "b")}
    ab_ok = all(pr[P]["a_minus_b_max_abs"] <= BAR_A_MINUS_B for P in PROMPTS) if args.paths == ("a", "b") else None
    summary = {
        "load": f"{rep['checkpoint_tower_merger_keys']}/{rep['expected']}, missing {len(rep['missing'])}, "
                f"unexpected {len(rep['unexpected'])}, meta {len(rep['on_meta'])}",
        **{f"path_{p}": {"cos_mean_min": min(pr[P][p]["cos_mean"] for P in PROMPTS),
                         "cos_min": min(pr[P][p]["cos_min"] for P in PROMPTS),
                         "max_abs_diff": max(pr[P][p]["max_abs_diff"] for P in PROMPTS),
                         "ref_absmax": max(pr[P][p]["ref_absmax"] for P in PROMPTS)} for p in args.paths},
        "a_minus_b_max_abs": max(pr[P]["a_minus_b_max_abs"] for P in PROMPTS) if ab_ok is not None else None,
        "baked_constants_equal_hf": (res["constants"]["pos_ids_equal_hf"] and res["constants"]["window_index_equal_hf"])
                                    if "b" in args.paths else None,
    }
    record = {
        "schema": "minicpmv47-parity-vision/1",
        "purpose": "re-authored vision tower (fp32 CPU, reference and baked paths) vs the HF fp32 oracle's "
                   "image_features, P1..P4; P1 stage by stage vs vision_stages.npz",
        "result": ("PASS" if load_ok and all(tower_ok.values()) and ab_ok and summary["baked_constants_equal_hf"]
                   else "FAIL") if args.paths == ("a", "b") else "PARTIAL (one path only)",
        "gates": {"load": load_ok, "tower_a": tower_ok["a"], "tower_b": tower_ok["b"], "a_minus_b": ab_ok},
        "bars": {"load": "459/459, missing / unexpected / meta 0", "cos_min": BAR_COS, "max_abs_diff": BAR_MAX_ABS,
                 "a_minus_b_max_abs": BAR_A_MINUS_B},
        "summary": summary,
        "input": "oracle.npz <P>__pixel_values unpacked to [1, 3, 448, 448] (repack == packed, exact)",
        "module": {"path": "conversion/minicpmv47/minicpmv47_vision.py", "sha256": sha256(HERE / "minicpmv47_vision.py")},
        "script": {"path": "conversion/minicpmv47/parity_vision.py", "sha256": sha256(Path(__file__))},
        **res,
        "versions": versions(),
        "wall_seconds": time.perf_counter() - t_all,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
    print(f"[parity_vision] {record['result']} {json.dumps(summary)}", flush=True)
    print(f"[parity_vision] -> {args.out}")


if __name__ == "__main__":
    main()
