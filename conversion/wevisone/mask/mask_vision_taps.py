#!/usr/bin/env python3
"""The fixed-grid vision tower with the mask decoder's four taps as a third output, and its checks on Core AI.

build:  the overlay's `Qwen3VLVisionEncoder` (fp16, the exporter's externalized composites) with a third output
        `taps [4, 1024, 2g, 2g]` = the block outputs that feed the three deepstack mergers (ViT layers 5 / 11 / 17)
        and the final merger (layer 23), unshuffled to row-major inside the graph (the package's
        `premerge_tokens_to_2d`). Kit and engine read the tower's outputs by name, so `image_embeds` /
        `deepstack_embeds` keep their names and order.
    <coreai venv>/bin/python mask_vision_taps.py build --grid 14 --out-root <exports> --record <json>

run:    Core AI (python runtime, AOT h16c, default specialization, one load each) on every dense row of the grid's
        O2 form, the oracle's own pixel_values (fp16) as input:
          taps vs the oracle's captured `dense_tap_maps`: cos per level and row (bar cos >= 0.999, ROUND2.md)
          alpha: Core AI taps -> torch fp32 package vs oracle taps -> torch fp32 package (bar MAE <= 0.01,
                 IoU(alpha > 0.5) >= 0.95)
          image_embeds / deepstack_embeds vs the two-output tower (`--r1-aot`) on the same patches: bit-equal?
          ms per call, three outputs vs two, alternating in one process (16 each, after 4 warm calls each)
    WATCH -- <coreai venv>/bin/python mask_vision_taps.py run --grid 14 --taps-aot <aimodelc> --r1-aot <aimodelc> \
        --pkg <dir> --sidecar <pt> --answers <dir> --out <json>
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))
sys.path.insert(2, str(HERE.parent.parent))

HF_ID = "tencent/WeVisOne-VL-2B"
REVISION = "3dad9ce8bcb6fbe0aea945bf8860bd2cfc3663bf"
FORM_FOR_GRID = {14: "O2-448", 16: "O2-512"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def tapped_class():
    from coreai_models.models.macos.qwen3_vl_pipelined import Qwen3VLVisionEncoder

    class Tapped(Qwen3VLVisionEncoder):
        """The fixed-grid tower plus `taps [4, 1024, 2g, 2g]` (row-major) for the mask decoder."""

        def forward(self, patches):
            x = self.patch_proj(patches)
            x = x + self.pos_embed_const.to(x.dtype)
            cos = self.cos_const.to(x.dtype)
            sin = self.sin_const.to(x.dtype)
            deep, taps = [], []
            k = 0
            for i, blk in enumerate(self.blocks):
                x = blk(x, cos, sin)
                if i in self.deepstack_visual_indexes:
                    taps.append(x)
                    deep.append(self.deepstack_merger_list[k](x))
                    k += 1
            taps.append(x)
            image_embeds = self.merger(x)
            m = self.vcfg.spatial_merge_size
            h, w = self.grid_h * m, self.grid_w * m
            maps = [t.reshape(h // m, w // m, m, m, -1).permute(0, 2, 1, 3, 4).reshape(h, w, -1).permute(2, 0, 1)
                    for t in taps]
            return image_embeds, torch.cat(deep, dim=0), torch.stack(maps)

    return Tapped


def build(args) -> int:
    import coreai.runtime as rt
    from _paths import hf_cache
    from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai

    repo = hf_cache() / ("models--" + HF_ID.replace("/", "--"))
    ref = (repo / "refs" / "main").read_text().strip()
    if ref != REVISION:
        raise SystemExit(f"refs/main = {ref}, expected {REVISION}")
    g = args.grid
    t0 = time.monotonic()
    Tapped = tapped_class()
    if args.precision == "fp16w32":
        # fp16 weights cast to fp32 in the graph, fp32 compute and I/O (the precision lever for the taps)
        from mask_build import fp16_storage_fp32_compute

        vis = fp16_storage_fp32_compute(Tapped.from_hf(HF_ID, target_dtype=torch.float32, grid_h=g, grid_w=g))
        io = torch.float32
    else:
        vis = Tapped.from_hf(HF_ID, target_dtype=torch.float16, grid_h=g, grid_w=g)
        io = torch.float16
    v = vis.vcfg
    patches = torch.zeros(vis.n_patches, v.in_channels * v.temporal_patch_size * v.patch_size ** 2, dtype=io)
    prog = export_to_coreai(vis, {"patches": patches}, dynamic_shapes={"patches": None}, input_names=("patches",),
                            output_names=("image_embeds", "deepstack_embeds", "taps"), state_names=(),
                            externalize_modules=[s for s in _EXTERNALIZE_SPECS
                                                 if s.composite_op_name != "gated_delta_update"])
    prog.optimize()
    name = f"wevisone_vl_2b_vision_taps_g{g}" + ("_fp16w32" if args.precision == "fp16w32" else "")
    out_dir = Path(args.out_root) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    prog.save_asset(out_dir / f"{name}.aimodel", rt.AIModelAssetMetadata())
    files = sorted(p for p in out_dir.rglob("*") if p.is_file())
    rec = {"schema": "wevisone-r2-vision-taps-build/1", "grid": g, "name": name, "bundle": str(out_dir),
           "bytes": sum(p.stat().st_size for p in files), "wall_seconds": time.monotonic() - t0,
           "outputs": ["image_embeds", "deepstack_embeds", "taps"], "revision": REVISION,
           "script": sha256_file(Path(__file__)),
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    Path(args.record).write_text(json.dumps(rec, indent=1) + "\n")
    print(f"[taps build] {name}: {rec['bytes']:,} B in {rec['wall_seconds']:.0f}s", flush=True)
    return 0


async def run(args) -> int:
    import coreai.runtime as rt

    import gate_common as G
    import mask_graph as MG
    from mask_torch_check import alpha_bar, alpha_of, package_forward, summarize

    torch.set_num_threads(4)
    g = args.grid
    form = FORM_FOR_GRID[g]
    t_all = time.perf_counter()
    out = {"schema": "wevisone-r2-vision-taps-run/1", "grid": g, "form": form, "window": G.window_record(args.label),
           "environment": G.env_record(), "rows": {}}
    fns = {}
    for key, p in (("taps", args.taps_aot), ("r1", args.r1_aot)):
        path = Path(p).resolve()
        t0 = time.perf_counter()
        m = await G.maybe(rt.AIModel.load(str(path), rt.SpecializationOptions.default()))
        fn = await G.maybe(m.load_function("main"))
        fns[key] = (m, fn)
        out[f"{key}_asset"] = {"path": str(path), "digest": G.tree_digest(path), "descriptor": G.fn_desc(fn),
                               "load_seconds": time.perf_counter() - t0}
    dec, _, _ = MG.load_package_decoder(args.pkg, args.sidecar)
    from wevisone_vl import engine as E  # noqa: E402

    in_dtype = {k: np.dtype(out[f"{k}_asset"]["descriptor"]["inputs"]["patches"][1]) for k in fns}

    async def call(key, pv):
        res = await G.maybe(fns[key][1](inputs={"patches": G.nd(rt, np.asarray(pv, in_dtype[key]))}))
        return {k: np.asarray(res[k].numpy()) for k in out[f"{key}_asset"]["descriptor"]["outputs"]}

    cos_min, alpha_insts, bit = [], [], []
    pv_last = None
    for jf in sorted((Path(args.answers) / form).glob("*.json")):
        meta = json.loads(jf.read_text())
        if "dense" not in meta or not meta["dense"].get("count_decoded"):
            continue
        rid = meta["id"]
        z = np.load(jf.with_suffix(".npz"))
        pv = z["pixel_values"].astype(np.float16)
        pv_last = pv
        a = await call("taps", pv)
        b = await call("r1", pv)
        taps = a["taps"].astype(np.float32)
        ref = z["dense_tap_maps"].astype(np.float32)
        lv_cos = [G.cos(taps[i], ref[i]) for i in range(4)]
        cos_min.append(min(lv_cos))
        same = {k: {"bit_equal": bool(np.array_equal(a[k], b[k])),
                    "max_abs_d": float(np.abs(a[k].astype(np.float32) - b[k].astype(np.float32)).max()),
                    "cos": G.cos(a[k], b[k])} for k in ("image_embeds", "deepstack_embeds")}
        bit.append(all(v["bit_equal"] for v in same.values()))
        hw = meta["dense"]["alpha_size_hw"]
        image = torch.from_numpy(z["dense_cnn_image"].astype(np.float32))[None]
        t_seg = torch.from_numpy(z["dense_t_seg"].astype(np.float32))
        boxes = torch.from_numpy(z["dense_boxes01"].astype(np.float32))
        with torch.no_grad():
            ref_out = package_forward(dec, image, [torch.from_numpy(ref[i])[None] for i in range(4)], t_seg, boxes,
                                      2 * g)
            ca_out = package_forward(dec, image, [torch.from_numpy(taps[i])[None] for i in range(4)], t_seg, boxes,
                                     2 * g)
        insts = alpha_bar(alpha_of(E, ca_out["mask_logits"], hw), alpha_of(E, ref_out["mask_logits"], hw))
        for r in insts:
            r["row"] = f"{form}:{rid}"
        alpha_insts.extend(insts)
        out["rows"][rid] = {"taps_cos_per_level": lv_cos, "taps_max_abs_d": float(np.abs(taps - ref).max()),
                            "taps_ref_absmax": float(np.abs(ref).max()), "taps_absmax": float(np.abs(taps).max()),
                            "image_embeds_vs_oracle_cos": G.cos(a["image_embeds"], z["image_embeds"]),
                            "two_vs_three_outputs": same, "alpha_vs_oracle_taps": insts}
        print(f"[taps g{g}] {rid}: taps cos {', '.join(f'{c:.6f}' for c in lv_cos)} | 2 vs 3 outputs bit "
              f"{bit[-1]} | alpha MAE max {max(r['mae'] for r in insts):.2e} IoU min "
              f"{min(r['iou'] for r in insts):.5f}", flush=True)
    # cost of the third output: alternate the two assets on one input, after warm calls
    timing = {"taps": [], "r1": []}
    if pv_last is not None:
        for key in ("taps", "r1"):
            for _ in range(4):
                await call(key, pv_last)
        for _ in range(args.reps):
            for key in ("taps", "r1"):
                t0 = time.perf_counter()
                await call(key, pv_last)
                timing[key].append((time.perf_counter() - t0) * 1e3)
    out["timing_ms"] = {k: {"n": len(v), "median": float(np.median(v)) if v else None,
                            "min": float(np.min(v)) if v else None, "max": float(np.max(v)) if v else None,
                            "all": v} for k, v in timing.items()}
    if timing["r1"]:
        out["timing_ms"]["third_output_cost_pct"] = 100.0 * (out["timing_ms"]["taps"]["median"]
                                                             / out["timing_ms"]["r1"]["median"] - 1.0)
    out["summary"] = {"rows": len(out["rows"]), "taps_cos_min": min(cos_min) if cos_min else None,
                      "taps_cos_bar": 0.999, "taps_pass": bool(cos_min) and min(cos_min) >= 0.999,
                      "two_vs_three_bit_equal_rows": f"{sum(bit)}/{len(bit)}",
                      "alpha": summarize(alpha_insts)}
    out["mpsgraph_scratch_cleanup"] = G.clean_own_mpsgraph_scratch()
    out["wall_seconds"] = time.perf_counter() - t_all
    out["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["script"] = sha256_file(Path(__file__))
    G.write_json(Path(args.out), out)
    print(json.dumps({**out["summary"], "timing": {k: v.get("median") if isinstance(v, dict) else v
                                                   for k, v in out["timing_ms"].items()}}, indent=1), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--grid", type=int, required=True, choices=[14, 16])
    ap.add_argument("--out-root", default="")
    ap.add_argument("--record", default="")
    ap.add_argument("--taps-aot", default="")
    ap.add_argument("--r1-aot", default="")
    ap.add_argument("--pkg", default="")
    ap.add_argument("--sidecar", default="")
    ap.add_argument("--answers", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp16w32"])
    ap.add_argument("--label", default="wv-c-r2")
    args = ap.parse_args()
    if args.cmd == "build":
        return build(args)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
