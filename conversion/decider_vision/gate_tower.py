#!/usr/bin/env python3
"""Round-2 gate: the decider-2b-vision tower at its two fixed grids vs the author's fp32 tower.

Mapika/decider-2b-vision is a Qwen3.5-2B VL (tower depth 24 / hidden 1024 / 16 heads / out 2048,
no DeepStack). The Core AI tower is the shipped `qwen3_5_vision.Qwen3_5VisionEncoder` baked at a
FIXED merged grid and exported by `conversion/export_qwen38vl_pipelined.py --skip-decoder`:

    g256 = 256x256 tile -> 16x16 patches -> 8x8 merged   =  64 tokens (game frames, speed)
    g448 = 448x448 tile -> 28x28 patches -> 14x14 merged = 196 tokens (photos)

The target is the round-1 oracle (`oracle_decider_vision.py`): per row and arm, the HF processor's
`pixel_values` [n_patch, 1536] and the HF fp32 tower's merger output `image_embeds` [N, 2048],
both captured inside the checkpoint's own `VisionDecisionModel.prepare()` -> `slot_logits()`.
The tower is per image, so rows are deduplicated by image name; the duplicates are checked to be
bit-identical in the oracle, so the dedup loses nothing.

  G1  host preprocessing vs the processor's pixel_values
        (a) `host.preprocess` = PIL BICUBIC + NumPy rescale/normalize/patchify   bar: max|d| == 0
        (b) all NumPy (`qwen38vl_preprocess.preprocess`, float resize)          recorded, in levels
        (b_u8) the same resize rounded to uint8 (a pixel-buffer host)            recorded, in levels
        (c) all NumPy in Pillow's own order (`host.resize_bicubic`: horizontal pass first, uint8
            between the passes) — the portable form a Swift host copies           recorded, in levels
  G2  the authored tower in fp32, eager CPU, on the oracle pixel_values
        bar: cos >= 0.99999 and min-row >= 0.9999 on every image
  G3  the fp16 AOT h16c `.aimodelc` on the Mac GPU, oracle pixel_values cast to fp16
        bar: cos >= 0.999 and min-row >= 0.999 on every image (the zoo's tower bar)
        + the same comparison for the fp32 export (`--vision-dtype fp32`, rows[].g3_fp32) and
          the fp16-storage / fp32-math export (`--vision-dtype fp16w32`, rows[].g3_fp16w32),
          reported next to it and kept out of the verdict
  G4  (b)'s all-NumPy patches through the same `.aimodelc` (g4; (c)'s = g4c)   recorded
  ms  load seconds + 10 warm encodes per grid — CONTENDED reference values (the GPU is shared
      with other sessions; the timing of record is round 5, under _GPU_LOCK)

Negative control (G2 and G3): the same pixels in raster patch order instead of merge-block-major.
It must miss the stage's bar, or the stage cannot go red. On an image whose patch rows the
permutation maps onto identical rows (a solid colour) it is the identity, so such images are
marked `immune` and left out of the control's verdict instead of being read as a dead gate.

Cosines are float64 (a float32 reduction prints cos > 1 at this size; knowledge/lfm2.5-vl-port.md).
Only the AOT asset is gated: the python-runtime JIT is not evidence for these graphs.

Run from the worktree root (shared venv, offline, pinned snapshot):
    HF_HOME=~/code/coreai/_decider2bv/hf HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 \\
        ../coreai-models/.venv/bin/python conversion/decider_vision/gate_tower.py
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import inspect
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))                    # host.py
sys.path.insert(0, str(HERE.parent))             # conversion/ (_paths)
sys.path.insert(0, str(REPO / "_smoke"))         # qwen38vl_preprocess / lfm25vl_preprocess
import host  # noqa: E402
from _paths import work_path  # noqa: E402
from lfm25vl_preprocess import BICUBIC, resize_antialias  # noqa: E402
from qwen38vl_preprocess import preprocess as numpy_preprocess  # noqa: E402

HF_ID = "Mapika/decider-2b-vision"
REVISION = "863e290863655f1d6b69324d77d09ac972d21609"
GRIDS = {"g256": 8, "g448": 14}                  # merged grid side
G2_BAR = (0.99999, 0.9999)                       # (cos, min-row): fp32 torch vs HF fp32
G3_BAR = (0.999, 0.999)                          # fp16 AOT vs HF fp32
LEVEL = 1.0 / (255.0 * host.IMAGE_STD)           # one 0-255 level after rescale+normalize = 2/255
LANE = work_path("_decider2bv")
OUT_JSON = REPO / "models" / "decider-2b-vision" / "gate-decider-2b-vision-tower.json"


async def maybe(x):
    return await x if inspect.isawaitable(x) else x


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def asset_digest(path: Path) -> dict:
    """sha256 per file of a directory asset (.aimodel / .aimodelc) + one digest over the listing."""
    files = sorted(p for p in path.rglob("*") if p.is_file())
    per = {str(p.relative_to(path)): sha256_file(p) for p in files}
    tree = hashlib.sha256("".join(f"{k}\0{v}\n" for k, v in per.items()).encode()).hexdigest()
    return {"path": str(path), "bytes": sum(p.stat().st_size for p in files), "tree_sha256": tree,
            "files": per}


def cos_stats(got: np.ndarray, want: np.ndarray) -> tuple[dict, np.ndarray]:
    g = np.asarray(got, np.float64)
    w = np.asarray(want, np.float64)
    c = float(g.ravel() @ w.ravel() / (np.linalg.norm(g) * np.linalg.norm(w)))
    rows = (g * w).sum(-1) / (np.linalg.norm(g, axis=-1) * np.linalg.norm(w, axis=-1))
    d = np.abs(g - w)
    i = int(rows.argmin())
    return {"cos": c, "min_row": float(rows[i]), "min_row_index": i,
            "max_abs": float(d.max()), "mean_abs": float(d.mean())}, rows


def passes(s: dict, bar: tuple[float, float]) -> bool:
    return s["cos"] >= bar[0] and s["min_row"] >= bar[1]


def patch_diff(got: np.ndarray, want: np.ndarray) -> dict:
    d = np.abs(got.astype(np.float64) - want.astype(np.float64))
    return {"max_abs": float(d.max()), "max_levels": float(d.max() / LEVEL),
            "n_diff": int((d > 0).sum()), "n_diff_gt_half_level": int((d > 0.5 * LEVEL).sum()),
            "n": int(d.size)}


def numpy_u8_patches(u8: np.ndarray, side: int) -> np.ndarray:
    """All-NumPy resize, rounded to uint8 like a pixel-buffer host, then the same normalize."""
    x = np.rint(resize_antialias(u8, side, side, BICUBIC))
    return host.patchify((x / 255.0 - host.IMAGE_MEAN) / host.IMAGE_STD).astype(np.float32)


def raster_perm(grid: int) -> np.ndarray:
    """raster-ordered patches = block_major_patches[perm] (the negative control)."""
    gp = grid * host.MERGE
    r, c = np.divmod(np.arange(gp * gp), gp)
    m = host.MERGE
    return (((r // m) * grid + c // m) * m + r % m) * m + c % m


def load_oracle(arm: str) -> tuple[dict, dict]:
    """{image: {rows, pixel_values, image_embeds, dup_identical}} for one arm + oracle provenance."""
    fx = json.loads((LANE / "oracle" / "fixture_oracle.json").read_text())
    by_image: dict[str, list[str]] = {}
    for r in fx["rows"]:
        if r["arm"] == arm and r["image"] is not None:
            by_image.setdefault(r["image"], []).append(r["id"])
    grid = GRIDS[arm]
    out = {}
    for name, rows in sorted(by_image.items()):
        first = np.load(LANE / "oracle" / "npz" / f"{rows[0]}__{arm}.npz")
        pv, emb = first["pixel_values"], first["image_embeds"]
        thw = first["image_grid_thw"].tolist()
        assert thw == [[1, 2 * grid, 2 * grid]], (name, arm, thw)
        assert pv.shape == (4 * grid * grid, 1536) and pv.dtype == np.float32, pv.shape
        assert emb.shape == (grid * grid, 2048) and emb.dtype == np.float32, emb.shape
        same = True
        for rid in rows[1:]:
            z = np.load(LANE / "oracle" / "npz" / f"{rid}__{arm}.npz")
            same &= bool(np.array_equal(z["pixel_values"], pv) and np.array_equal(z["image_embeds"], emb))
        out[name] = {"rows": rows, "pixel_values": pv, "image_embeds": emb, "dup_identical": same}
    prov = {"fixture_oracle_sha256": sha256_file(LANE / "oracle" / "fixture_oracle.json"),
            "rows_json_sha256": fx["fixture"]["rows_json_sha256"], "oracle_versions": fx["versions"],
            "oracle_source": fx["source"]["hf_id"] + "@" + fx["source"]["revision"]}
    return out, prov


def isolate_rows(rows_cos: np.ndarray, want: np.ndarray, pv: np.ndarray, bar: float) -> dict:
    """The OvisOCR2 isolation: which rows miss, and do they track the row norm or pixel spread?"""
    ref_norm = np.linalg.norm(want.astype(np.float64), axis=-1)
    px_std = pv.astype(np.float64).reshape(len(want), -1).std(-1)   # 4 patches per merged token
    low = np.flatnonzero(rows_cos < bar)

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None

    return {"rows_below": low.tolist(), "row_cos_below": rows_cos[low].tolist(),
            "ref_norm_below": ref_norm[low].tolist(), "px_std_below": px_std[low].tolist(),
            "ref_norm_median_all": float(np.median(ref_norm)),
            "flat_tokens": int((px_std == 0).sum()), "flat_below": int((px_std[low] == 0).sum()),
            "corr_rowcos_pxstd": corr(rows_cos, px_std), "corr_rowcos_refnorm": corr(rows_cos, ref_norm)}


def pooled_isolation(pool: dict, bar: float) -> dict:
    """isolate_rows over every row of every image of one grid (the OvisOCR2 numbers, pooled)."""
    c, n, p = (np.concatenate(pool[k]) for k in ("cos", "ref_norm", "px_std"))
    low = c < bar

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None

    return {"rows": int(c.size), "rows_below": int(low.sum()),
            "row_cos_min": float(c.min()), "row_cos_p01": float(np.quantile(c, 0.01)),
            "row_cos_median": float(np.median(c)),
            "ref_norm_median_below": float(np.median(n[low])) if low.any() else None,
            "ref_norm_median_all": float(np.median(n)),
            "flat_tokens": int((p == 0).sum()), "flat_below": int((p[low] == 0).sum()),
            "corr_rowcos_pxstd": corr(c, p), "corr_rowcos_refnorm": corr(c, n)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", default="g256,g448")
    ap.add_argument("--stages", default="g1,g2,g3,g4,ms")
    ap.add_argument("--exports", default=str(LANE / "exports"))
    ap.add_argument("--variants", default="fp16,fp32,fp16w32",
                    help="AOT assets to run: fp16 = exports/decider_2b_vision_<arm>_vision_aotc/ (the "
                         "gate); fp32 / fp16w32 = ..._vision_<variant>_aotc/ (reported alongside)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bench", type=int, default=10)
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()
    arms = args.arms.split(",")
    stages = set(args.stages.split(","))
    meta = {im["name"]: im for im in json.loads((LANE / "fixtures" / "meta.json").read_text())["images"]}

    import PIL
    import torch

    record: dict = {
        "schema": "coreai-decider-vision-tower-gate/1",
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": {"hf_id": HF_ID, "revision": REVISION},
        "bars": {"g1a_max_abs": 0.0, "g2": {"cos": G2_BAR[0], "min_row": G2_BAR[1]},
                 "g3": {"cos": G3_BAR[0], "min_row": G3_BAR[1]}},
        "level": LEVEL,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pillow": PIL.__version__,
                     "torch": torch.__version__},
        "platform": {"machine": platform.machine(), "macos": platform.mac_ver()[0],
                     "cpu": os.popen("sysctl -n machdep.cpu.brand_string").read().strip()},
        "grids": {a: {"merged": [GRIDS[a]] * 2, "tile": host.tile_side(GRIDS[a]),
                      "n_patch": 4 * GRIDS[a] ** 2, "tokens": GRIDS[a] ** 2} for a in arms},
        "rows": [], "assets": {}, "timing": {}, "summary": {},
    }
    try:
        import importlib.metadata as md
        for p in ("coreai-core", "coreai-torch", "coreai-models", "transformers", "huggingface_hub"):
            record["versions"][p] = md.version(p)
    except Exception as e:  # noqa: BLE001
        record["versions"]["error"] = repr(e)

    oracle, rows_by = {}, {}
    for arm in arms:
        oracle[arm], record["oracle"] = load_oracle(arm)
        for name, o in oracle[arm].items():
            rows_by[(arm, name)] = {"image": name, "arm": arm, "oracle_rows": o["rows"],
                                    "oracle_dup_identical": o["dup_identical"],
                                    "image_size": meta[name]["size"]}
        print(f"{arm}: {len(oracle[arm])} unique images "
              f"({sum(len(o['rows']) for o in oracle[arm].values())} rows)", flush=True)

    # ---- G1: host preprocessing -------------------------------------------------------------
    if "g1" in stages:
        from PIL import Image
        for arm in arms:
            side = host.tile_side(GRIDS[arm])
            for name, o in oracle[arm].items():
                path = LANE / "fixtures" / meta[name]["path"]
                u8 = np.asarray(Image.open(path).convert("RGB"))
                a = host.preprocess(path, GRIDS[arm])
                b = numpy_preprocess(u8, side)
                b8 = numpy_u8_patches(u8, side)
                c = host.preprocess(path, GRIDS[arm], resize="numpy")
                rec = rows_by[(arm, name)]
                rec["g1a"] = patch_diff(a, o["pixel_values"])
                rec["g1b"] = patch_diff(b, o["pixel_values"])
                rec["g1b_u8"] = patch_diff(b8, o["pixel_values"])
                rec["g1c"] = patch_diff(c, o["pixel_values"])
                o["numpy_patches"] = b.astype(np.float32)
                o["numpy_pil_order_patches"] = c
            worst_b = max(oracle[arm], key=lambda n: rows_by[(arm, n)]["g1b"]["max_abs"])
            worst_c = max(oracle[arm], key=lambda n: rows_by[(arm, n)]["g1c"]["max_abs"])
            print(f"G1 {arm}: (a) max|d| {max(rows_by[(arm, n)]['g1a']['max_abs'] for n in oracle[arm]):.3e}"
                  f"  (b) max {rows_by[(arm, worst_b)]['g1b']['max_levels']:.3f} levels ({worst_b})"
                  f"  (c) max {rows_by[(arm, worst_c)]['g1c']['max_levels']:.3f} levels ({worst_c})", flush=True)

    # ---- G2: fp32 torch ---------------------------------------------------------------------
    if "g2" in stages:
        from huggingface_hub import snapshot_download

        from coreai_models.models.macos.qwen3_5_vision import Qwen3_5VisionEncoder
        snap = snapshot_download(HF_ID, allow_patterns=["*.safetensors", "*.safetensors.index.json",
                                                        "config.json"])
        assert Path(snap).name == REVISION, f"snapshot {snap} is not the pinned {REVISION}"
        record["snapshot"] = snap
        torch.set_num_threads(args.threads)
        for arm in arms:
            grid = GRIDS[arm]
            t0 = time.perf_counter()
            vis = Qwen3_5VisionEncoder.from_hf(HF_ID, target_dtype=torch.float32, grid_h=grid,
                                               grid_w=grid).eval()
            load_s = time.perf_counter() - t0
            perm = raster_perm(grid)
            walls = []
            for name, o in oracle[arm].items():
                pv = o["pixel_values"]
                t0 = time.perf_counter()
                with torch.no_grad():
                    got = vis(torch.from_numpy(pv)).numpy()
                walls.append(time.perf_counter() - t0)
                s, _ = cos_stats(got, o["image_embeds"])
                s["pass"] = passes(s, G2_BAR)
                rec = rows_by[(arm, name)]
                rec["g2"] = s
                if np.array_equal(pv[perm], pv):
                    rec["g2_neg"] = {"immune": True}
                else:
                    with torch.no_grad():
                        ctrl = vis(torch.from_numpy(np.ascontiguousarray(pv[perm]))).numpy()
                    cs, _ = cos_stats(ctrl, o["image_embeds"])
                    rec["g2_neg"] = {"immune": False, "cos": cs["cos"], "min_row": cs["min_row"],
                                     "red": not passes(cs, G2_BAR)}
            record["timing"].setdefault(arm, {})["g2_torch_cpu"] = {
                "load_s": load_s, "encode_s_median": float(np.median(walls)), "threads": args.threads}
            del vis
            worst = min(oracle[arm], key=lambda n: rows_by[(arm, n)]["g2"]["min_row"])
            print(f"G2 {arm}: min cos {min(rows_by[(arm, n)]['g2']['cos'] for n in oracle[arm]):.8f}"
                  f"  min-row {rows_by[(arm, worst)]['g2']['min_row']:.8f} ({worst})"
                  f"  all pass {all(rows_by[(arm, n)]['g2']['pass'] for n in oracle[arm])}", flush=True)

    # ---- G3 / G4 / ms: AOT h16c on the GPU (fp16 = the gate; fp32 = the numerics reference) --
    if stages & {"g3", "g4", "ms"}:
        import coreai.runtime as rt

        def gpu_util() -> int | None:
            txt = os.popen("ioreg -r -d 1 -w 0 -c IOAccelerator").read()
            key = '"Device Utilization %"='
            return int(txt.split(key)[1].split(",")[0].split("}")[0]) if key in txt else None

        async def gpu(variant: str) -> None:
            sfx = "" if variant == "fp16" else f"_{variant}"      # rows[].g3 / rows[].g3_fp32 ...
            in_dtype = np.float16 if variant == "fp16" else np.float32
            for arm in arms:
                grid = GRIDS[arm]
                stem = f"decider_2b_vision_{arm}"
                aimodel = Path(args.exports) / f"{stem}_vision_{variant}" / f"{stem}_vision_{variant}.aimodel"
                aotc_dir = Path(args.exports) / f"{stem}_vision{sfx}_aotc"
                cands = sorted(aotc_dir.glob("*.h16c.aimodelc"))
                assert len(cands) == 1, (aotc_dir, cands)
                aimodelc = cands[0]
                assets = record["assets"].setdefault(arm, {})[variant] = {"aimodelc": asset_digest(aimodelc)}
                if aimodel.exists():
                    assets["aimodel"] = asset_digest(aimodel)
                la0, util0 = os.getloadavg(), gpu_util()
                t0 = time.perf_counter()
                m = await maybe(rt.AIModel.load(str(aimodelc), rt.SpecializationOptions.default()))
                fn = await maybe(m.load_function(m.function_names[0]))
                load_s = time.perf_counter() - t0
                print(f"{arm} {variant}: {aimodelc.name} fn {fn.desc.name} in {list(fn.desc.input_names)} "
                      f"out {list(fn.desc.output_names)} load {load_s:.2f} s", flush=True)

                async def encode(pt: np.ndarray) -> np.ndarray:
                    out = await maybe(fn(inputs={"patches": rt.NDArray(
                        np.ascontiguousarray(pt.astype(in_dtype)))}))
                    return np.asarray(out["image_embeds"].numpy()).astype(np.float32)

                names = list(oracle[arm])
                t0 = time.perf_counter()
                first = await encode(oracle[arm][names[0]]["pixel_values"])
                first_call_s = time.perf_counter() - t0
                again = await encode(oracle[arm][names[0]]["pixel_values"])
                repeat_identical = bool(np.array_equal(first, again))
                perm = raster_perm(grid)
                below_all = []
                pool: dict = {"cos": [], "ref_norm": [], "px_std": []}
                for name in names:
                    o = oracle[arm][name]
                    rec = rows_by[(arm, name)]
                    if "g3" in stages:
                        got = await encode(o["pixel_values"])
                        s, rows_cos = cos_stats(got, o["image_embeds"])
                        s["pass"] = passes(s, G3_BAR)
                        s["finite"] = bool(np.isfinite(got).all())
                        s["rows_below_bar"] = int((rows_cos < G3_BAR[1]).sum())
                        rec[f"g3{sfx}"] = s
                        pool["cos"].append(rows_cos)
                        pool["ref_norm"].append(np.linalg.norm(o["image_embeds"].astype(np.float64), axis=-1))
                        pool["px_std"].append(o["pixel_values"].astype(np.float64).reshape(grid * grid, -1).std(-1))
                        if s["rows_below_bar"]:
                            below_all.append((name, isolate_rows(rows_cos, o["image_embeds"],
                                                                 o["pixel_values"], G3_BAR[1])))
                        pv = o["pixel_values"]
                        if np.array_equal(pv[perm], pv):
                            rec[f"g3{sfx}_neg"] = {"immune": True}
                        else:
                            ctrl = await encode(np.ascontiguousarray(pv[perm]))
                            cs, _ = cos_stats(ctrl, o["image_embeds"])
                            rec[f"g3{sfx}_neg"] = {"immune": False, "cos": cs["cos"],
                                                   "min_row": cs["min_row"], "red": not passes(cs, G3_BAR)}
                    if "g4" in stages and "numpy_patches" in o:
                        for key, pk in (("g4", "numpy_patches"), ("g4c", "numpy_pil_order_patches")):
                            got = await encode(o[pk])
                            s, _ = cos_stats(got, o["image_embeds"])
                            rec[f"{key}{sfx}"] = s
                ms = []
                if "ms" in stages:
                    pt = oracle[arm][names[0]]["pixel_values"]
                    await encode(pt)
                    for _ in range(args.bench):
                        t0 = time.perf_counter()
                        await encode(pt)
                        ms.append((time.perf_counter() - t0) * 1e3)
                record["timing"].setdefault(arm, {})[f"aot_gpu_{variant}"] = {
                    "contended": "shared GPU, no _GPU_LOCK (reference values; round 5 times under the lock)",
                    "load_s": load_s, "first_call_s": first_call_s,
                    "encode_ms": {"n": len(ms), "median": float(np.median(ms)) if ms else None,
                                  "min": min(ms) if ms else None, "max": max(ms) if ms else None,
                                  "all": ms},
                    "loadavg_before": list(la0), "loadavg_after": list(os.getloadavg()),
                    "gpu_util_pct_before_load": util0, "repeat_bit_identical": repeat_identical}
                if below_all:
                    record.setdefault("g3_isolation", {}).setdefault(variant, {})[arm] = dict(below_all)
                if pool["cos"]:
                    record.setdefault("g3_isolation_pooled", {}).setdefault(variant, {})[arm] = \
                        pooled_isolation(pool, G3_BAR[1])
                if "g3" in stages:
                    worst = min(names, key=lambda n: rows_by[(arm, n)][f"g3{sfx}"]["min_row"])
                    print(f"G3 {arm} {variant}: min cos "
                          f"{min(rows_by[(arm, n)][f'g3{sfx}']['cos'] for n in names):.6f}"
                          f"  min-row {rows_by[(arm, worst)][f'g3{sfx}']['min_row']:.6f} ({worst})"
                          f"  all pass {all(rows_by[(arm, n)][f'g3{sfx}']['pass'] for n in names)}", flush=True)
                for key in ("g4", "g4c"):
                    if "g4" in stages and any(f"{key}{sfx}" in rows_by[(arm, n)] for n in names):
                        w4 = min(names, key=lambda n: rows_by[(arm, n)][f"{key}{sfx}"]["cos"])
                        print(f"{key.upper()} {arm} {variant}: min cos "
                              f"{rows_by[(arm, w4)][f'{key}{sfx}']['cos']:.6f} ({w4})  min-row "
                              f"{min(rows_by[(arm, n)][f'{key}{sfx}']['min_row'] for n in names):.6f}",
                              flush=True)
                if ms:
                    print(f"ms {arm} {variant}: load {load_s:.2f} s, first call {first_call_s * 1e3:.1f} ms, "
                          f"encode median {np.median(ms):.1f} (min {min(ms):.1f}, max {max(ms):.1f}) ms "
                          f"[shared GPU, util {util0}% before load, loadavg {la0[0]:.1f}]", flush=True)
                del fn, m

        for variant in args.variants.split(","):
            asyncio.run(gpu(variant))

    # ---- summary ------------------------------------------------------------------------------
    record["rows"] = [rows_by[k] for k in sorted(rows_by, key=lambda k: (k[0], k[1]))]
    summ = {}
    for arm in arms:
        rs = [r for r in record["rows"] if r["arm"] == arm]
        s: dict = {"images": len(rs), "oracle_rows": sum(len(r["oracle_rows"]) for r in rs),
                   "oracle_dups_identical": all(r["oracle_dup_identical"] for r in rs)}

        def worst(key, field, fn=min):
            have = [r for r in rs if key in r and field in r[key]]
            if not have:
                return None
            r = fn(have, key=lambda r: r[key][field])
            return {"value": r[key][field], "image": r["image"]}

        if any("g1a" in r for r in rs):
            s["g1a_max_abs"] = worst("g1a", "max_abs", max)
            s["g1a_pass"] = all(r["g1a"]["max_abs"] == 0.0 for r in rs)
            s["g1b_max_abs"] = worst("g1b", "max_abs", max)
            s["g1b_max_levels"] = worst("g1b", "max_levels", max)
            s["g1b_u8_max_levels"] = worst("g1b_u8", "max_levels", max)
            s["g1c_max_levels"] = worst("g1c", "max_levels", max)
            s["g1c_n_diff"] = sum(r["g1c"]["n_diff"] for r in rs)
        for g in ("g2", "g3", "g3_fp32", "g3_fp16w32"):
            if any(g in r for r in rs):
                s[g] = {"min_cos": worst(g, "cos"), "min_row": worst(g, "min_row"),
                        "max_abs": worst(g, "max_abs", max), "pass": all(r[g]["pass"] for r in rs),
                        "n_pass": sum(r[g]["pass"] for r in rs)}
                neg = [r for r in rs if not r.get(f"{g}_neg", {}).get("immune", True)]
                imm = [r["image"] for r in rs if r.get(f"{g}_neg", {}).get("immune")]
                if neg:
                    hi = max(neg, key=lambda r: r[f"{g}_neg"]["cos"])
                    s[f"{g}_neg"] = {"images": len(neg), "immune": imm,
                                     "max_cos": {"value": hi[f"{g}_neg"]["cos"], "image": hi["image"]},
                                     "min_row_of_max_cos": hi[f"{g}_neg"]["min_row"],
                                     "all_red": all(r[f"{g}_neg"]["red"] for r in neg)}
        for g in ("g4", "g4_fp32", "g4_fp16w32", "g4c", "g4c_fp32", "g4c_fp16w32"):
            if any(g in r for r in rs):
                s[g] = {"min_cos": worst(g, "cos"), "min_row": worst(g, "min_row"),
                        "max_abs": worst(g, "max_abs", max)}
        summ[arm] = s
    record["summary"] = summ
    # The verdict is the round's gate: G1a, G2 and the fp16 G3 with their negative controls.
    # The fp32 asset is the numerics reference reported next to it, not part of the verdict.
    record["verdict"] = {arm: {k: v for k, v in (
        ("g1a", s.get("g1a_pass")), ("g2", s.get("g2", {}).get("pass")),
        ("g2_neg_all_red", s.get("g2_neg", {}).get("all_red")), ("g3", s.get("g3", {}).get("pass")),
        ("g3_neg_all_red", s.get("g3_neg", {}).get("all_red"))) if v is not None} for arm, s in summ.items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {out}")

    for arm, v in record["verdict"].items():
        print(f"verdict {arm}: " + "  ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    ok = all(all(v.values()) for v in record["verdict"].values())
    print("ALL PASS" if ok else "NOT ALL PASS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
