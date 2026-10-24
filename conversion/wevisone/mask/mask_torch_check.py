#!/usr/bin/env python3
"""Torch-side checks of the static two-graph mask decoder (mask_graph.py) against the package, before any export.

Per dense row (an `<id>.npz` with `dense_*` arrays under `<answers>/<form>/`, from oracle_wevisone.py), with the
oracle's captured decoder inputs (tap maps, `<OBJ>` hidden states, boxes, 1024 x 1024 canvas):
  pkg      the package's `_forward_fp32`, chunks of 16 like `WeVisOneVL.decode_masks` (torch fp32, CPU)
  m        mask_graph.GraphA (n_max 16, padded) + GraphB per instance, gather sampler
  m_gs     the same with the package's F.grid_sample (separates the sampler from the restructure)
  alpha    every output goes through the package's own `WeVisOneVL._masks_from_logits` to the original size
  bar      per instance MAE <= 0.01 and IoU(alpha > 0.5) >= 0.95 (ROUND2.md, fixed before any number)
  floor    pkg (this venv) vs the oracle's captured alpha / logits (same function, other torch, 1 thread)
  padding  p1: same n_max, padding rows zeros vs random -> real rows bit-equal;
           p2: n_max = N vs n_max 16 -> real rows bit-equal or max|d|;
           p3 (red arm): p1 without the key-padding mask -> real rows must move
  red arm  rows with N >= 2: each instance alone (N = 1 calls) vs the package's batch, alpha bar
  absmax   per module output and per self-attention score over all rows (the fp16 decision and the mask fill)

    QW QY <coreai venv>/bin/python mask_torch_check.py --pkg <dir holding wevisone_vl/> --sidecar <mask_decoder.pt> \
        --answers <dir with O2-448/ O2-512/> --out <json> [--forms O2-448,O2-512] [--rows a,b] [--threads 4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mask_graph as MG  # noqa: E402

MAE_BAR, IOU_BAR = 0.01, 0.95
GRID_FOR_FORM = {"O2-448": 28, "O2-512": 32}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def alpha_of(engine_mod, logits: torch.Tensor, hw) -> np.ndarray:
    return np.stack(engine_mod.WeVisOneVL._masks_from_logits(None, logits, int(hw[0]), int(hw[1])))


def alpha_bar(cand: np.ndarray, ref: np.ndarray) -> list[dict]:
    out = []
    for i in range(ref.shape[0]):
        a, b = cand[i].astype(np.float64), ref[i].astype(np.float64)
        ma, mb = a > 0.5, b > 0.5
        union = int(np.logical_or(ma, mb).sum())
        inter = int(np.logical_and(ma, mb).sum())
        iou = 1.0 if union == 0 else inter / union
        mae = float(np.abs(a - b).mean())
        out.append({"instance": i, "mae": mae, "iou": iou, "both_empty": union == 0,
                    "max_abs_d": float(np.abs(a - b).max()), "fg_ref": int(mb.sum()), "fg_cand": int(ma.sum()),
                    "pass": bool(mae <= MAE_BAR and iou >= IOU_BAR)})
    return out


def cmp(a: torch.Tensor, b: torch.Tensor) -> dict:
    a64, b64 = a.detach().double().flatten(), b.detach().double().flatten()
    d = (a64 - b64).abs()
    return {"max_abs_d": float(d.max()), "bit_equal": bool(torch.equal(a.detach(), b.detach())),
            "cos": float(a64 @ b64 / (a64.norm() * b64.norm())), "ref_absmax": float(b64.abs().max())}


def summarize(insts: list[dict]) -> dict:
    if not insts:
        return {"instances": 0}
    worst = max(insts, key=lambda r: (r["mae"] / MAE_BAR, (1 - r["iou"]) / (1 - IOU_BAR)))
    return {"instances": len(insts), "pass": all(r["pass"] for r in insts),
            "mae_max": max(r["mae"] for r in insts), "iou_min": min(r["iou"] for r in insts),
            "worst": {k: worst[k] for k in ("row", "instance", "mae", "iou")},
            "both_empty": [f"{r['row']}#{r['instance']}" for r in insts if r["both_empty"]]}


def package_forward(dec, image, maps, t_seg, boxes, g, chunk=16):
    outs = {k: [] for k in ("mask_logits", "mask_logits_aux", "mask_logits_coarse")}
    for s in range(0, t_seg.shape[0], chunk):
        e = min(t_seg.shape[0], s + chunk)
        o = dec._forward_fp32(image, maps, t_seg[s:e], boxes[s:e], (g, g))
        for k in outs:
            outs[k].append(o[k])
    return {k: torch.cat(v, 0) for k, v in outs.items()}


class AbsMax:
    """Max |output| per named module, over every forward while attached."""

    def __init__(self, root: torch.nn.Module):
        self.max: dict[str, float] = {}
        self.handles = []
        for name, mod in root.named_modules():
            if name:
                self.handles.append(mod.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def hook(_m, _i, out):
            ts = out if isinstance(out, (tuple, list)) else [out]
            for t in ts:
                if isinstance(t, torch.Tensor) and t.is_floating_point():
                    self.max[name] = max(self.max.get(name, 0.0), float(t.detach().abs().max()))
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--forms", default="O2-448,O2-512")
    ap.add_argument("--rows", default="")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-padding", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t_all = time.perf_counter()

    dec, cfg, side = MG.load_package_decoder(args.pkg, args.sidecar)
    from wevisone_vl import engine as E  # noqa: E402  (after import_package put the package on sys.path)

    want = set(filter(None, args.rows.split(",")))
    rec: dict = {"schema": "wevisone-r2-mask-torch-check/1", "torch": torch.__version__, "threads": args.threads,
                 "params": sum(p.numel() for p in dec.parameters()), "cfg": cfg.to_dict(),
                 "sidecar_sha256": sha256_file(Path(args.sidecar)), "bar": {"mae": MAE_BAR, "iou": IOU_BAR},
                 "rows": {}}
    absmax = AbsMax(dec)
    score_absmax: dict[str, float] = {}
    inputs_absmax: dict[str, float] = {}
    lists = {k: [] for k in ("m_vs_pkg", "m_gs_vs_pkg", "pkg_vs_oracle", "m_vs_oracle", "red_n1_vs_pkg",
                             "fourier64_vs_pkg", "m_hf_vs_pkg")}
    pad_rec = []
    for form in args.forms.split(","):
        g = GRID_FOR_FORM[form]
        ga = MG.GraphA(dec, g).eval()
        ga.keep_scores = True
        ga_gs = MG.GraphA(dec, g, sampler=MG.grid_sample_reference).eval()
        ga_hf = MG.GraphAHF(dec, g).eval()
        gb = MG.GraphB(dec).eval()
        for jf in sorted((Path(args.answers) / form).glob("*.json")):
            meta = json.loads(jf.read_text())
            if "dense" not in meta or not meta["dense"].get("count_decoded"):
                continue
            rid = meta["id"]
            if want and rid not in want:
                continue
            key = f"{form}:{rid}"
            t0 = time.perf_counter()
            z = np.load(jf.with_suffix(".npz"))
            hw = meta["dense"]["alpha_size_hw"]
            image = torch.from_numpy(z["dense_cnn_image"].astype(np.float32))[None]
            maps = [torch.from_numpy(z["dense_tap_maps"][i].astype(np.float32))[None] for i in range(4)]
            t_seg = torch.from_numpy(z["dense_t_seg"].astype(np.float32))
            boxes = torch.from_numpy(z["dense_boxes01"].astype(np.float32))
            n = int(t_seg.shape[0])
            for nm, t in (("image", image), ("taps", torch.stack(maps)), ("t_seg", t_seg)):
                inputs_absmax[nm] = max(inputs_absmax.get(nm, 0.0), float(t.abs().max()))
            row: dict = {"form": form, "id": rid, "n": n, "grid": g, "alpha_hw": hw, "boxes01": boxes.tolist()}
            with torch.no_grad():
                pkg = package_forward(dec, image, maps, t_seg, boxes, g)
                m = MG.run_m(ga, gb, image, maps, t_seg, boxes)
                for li, s in enumerate(m["scores"]):
                    score_absmax[f"layer{li % 6}"] = max(score_absmax.get(f"layer{li % 6}", 0.0),
                                                         float(s.abs().max()))
                m_gs = MG.run_m(ga_gs, gb, image, maps, t_seg, boxes)
                m_hf = MG.run_m(ga_hf, gb, image, maps, t_seg, boxes)
                bp = dec.query_builder.box_position
                x1, y1, x2, y2 = boxes.unbind(-1)
                pe_pkg = torch.cat([bp.pe(x1), bp.pe(y1), bp.pe(0.2 * (x2 - x1).clamp_min(1e-6).log() + 0.5),
                                    bp.pe(0.2 * (y2 - y1).clamp_min(1e-6).log() + 0.5)], dim=-1)
            pe_np = torch.from_numpy(MG.box_fourier_np(boxes.numpy()))
            a_pkg = alpha_of(E, pkg["mask_logits"], hw)
            a_m = alpha_of(E, m["mask_logits"], hw)
            a_mgs = alpha_of(E, m_gs["mask_logits"], hw)
            a_mhf = alpha_of(E, m_hf["mask_logits"], hw)
            a_or = z["dense_alpha"].astype(np.float32)
            row["logits"] = {
                "m_vs_pkg": {k: cmp(m[k], pkg[k]) for k in ("mask_logits", "mask_logits_aux", "mask_logits_coarse")},
                "m_gs_vs_pkg": {k: cmp(m_gs[k], pkg[k]) for k in ("mask_logits", "mask_logits_aux",
                                                                   "mask_logits_coarse")},
                "m_vs_m_gs_queries": cmp(m["queries"], m_gs["queries"]),
                "m_hf_vs_pkg": {k: cmp(m_hf[k], pkg[k]) for k in ("mask_logits", "mask_logits_coarse")},
                "box_pe_numpy_vs_package": cmp(pe_np, pe_pkg),
                "pkg_vs_oracle": {k: cmp(pkg[k], torch.from_numpy(z[f"dense_{k}"].astype(np.float32)))
                                  for k in ("mask_logits", "mask_logits_aux", "mask_logits_coarse")}}
            for name, cand, ref in (("m_vs_pkg", a_m, a_pkg), ("m_gs_vs_pkg", a_mgs, a_pkg),
                                    ("pkg_vs_oracle", a_pkg, a_or), ("m_vs_oracle", a_m, a_or),
                                    ("m_hf_vs_pkg", a_mhf, a_pkg)):
                insts = alpha_bar(cand, ref)
                for r in insts:
                    r["row"] = key
                row[f"alpha_{name}"] = insts
                lists[name].extend(insts)
            # the box Fourier features run to 2**31 x 2 pi: their sin / cos depend on the large-argument reduction
            # of the device's sin. Sensitivity: the same package with the phase's sin / cos taken in float64.
            pe_cls = type(dec.query_builder.box_position.pe)
            orig_pe_forward = pe_cls.forward

            def pe64(self, values):
                phase = (2.0 * math.pi * values[..., None].float() * self.frequencies).double()
                return torch.cat([phase.sin(), phase.cos()], dim=-1).to(values.dtype)

            pe_cls.forward = pe64
            try:
                with torch.no_grad():
                    pkg64 = package_forward(dec, image, maps, t_seg, boxes, g)
            finally:
                pe_cls.forward = orig_pe_forward
            insts = alpha_bar(alpha_of(E, pkg64["mask_logits"], hw), a_pkg)
            for r in insts:
                r["row"] = key
            row["alpha_fourier64_vs_pkg"] = insts
            row["logits"]["fourier64_vs_pkg"] = cmp(pkg64["mask_logits"], pkg["mask_logits"])
            lists["fourier64_vs_pkg"].extend(insts)
            if n >= 2:
                # red arm: the package called once per instance (N = 1) instead of one batch
                with torch.no_grad():
                    singles = [package_forward(dec, image, maps, t_seg[i:i + 1], boxes[i:i + 1], g)["mask_logits"]
                               for i in range(n)]
                a_single = alpha_of(E, torch.cat(singles, 0), hw)
                insts = alpha_bar(a_single, a_pkg)
                for r in insts:
                    r["row"] = key
                row["alpha_red_n1_vs_pkg"] = insts
                lists["red_n1_vs_pkg"].extend(insts)
            if not args.skip_padding:
                with torch.no_grad():
                    gen = torch.Generator().manual_seed(1234)
                    rnd_seg = torch.randn(MG.N_MAX, t_seg.shape[1], generator=gen) * t_seg.std()
                    lo = torch.rand(MG.N_MAX, 2, generator=gen) * 0.5
                    rnd_box = torch.cat([lo, lo + 0.1 + torch.rand(MG.N_MAX, 2, generator=gen) * 0.4], 1)
                    p_zero = MG.run_m(ga, gb, image, maps, t_seg, boxes)
                    p_rand = MG.run_m(ga, gb, image, maps, t_seg, boxes, pad_seg=rnd_seg, pad_box=rnd_box)
                    ga_n = MG.GraphA(dec, g, n_max=n).eval()
                    p_exact = MG.run_m(ga_n, gb, image, maps, t_seg, boxes, n_max=n)
                    ga_off = MG.GraphA(dec, g, mask_padding=False).eval()
                    p_off_z = MG.run_m(ga_off, gb, image, maps, t_seg, boxes)
                    p_off_r = MG.run_m(ga_off, gb, image, maps, t_seg, boxes, pad_seg=rnd_seg, pad_box=rnd_box)
                prec = {"row": key, "n": n}
                for name, a, b in (("p1_zero_vs_random", p_rand, p_zero), ("p2_exact_vs_padded", p_exact, p_zero),
                                   ("p3_nomask_zero_vs_random", p_off_r, p_off_z)):
                    prec[name] = {k: cmp(a[k], b[k]) for k in ("queries", "mask_logits_coarse", "mask_logits")}
                prec["p3_alpha"] = alpha_bar(alpha_of(E, p_off_r["mask_logits"], hw),
                                             alpha_of(E, p_off_z["mask_logits"], hw))
                pad_rec.append(prec)
                row["padding"] = prec
            row["seconds"] = time.perf_counter() - t0
            rec["rows"][key] = row
            s = row["alpha_m_vs_pkg"]
            print(f"[torch-check] {key} N={n}: m vs pkg logits max|d| "
                  f"{row['logits']['m_vs_pkg']['mask_logits']['max_abs_d']:.3g}, alpha MAE max "
                  f"{max(r['mae'] for r in s):.2e} IoU min {min(r['iou'] for r in s):.5f}; pkg vs oracle logits max|d| "
                  f"{row['logits']['pkg_vs_oracle']['mask_logits']['max_abs_d']:.3g}; "
                  f"{('p1 bit ' + str(row['padding']['p1_zero_vs_random']['mask_logits']['bit_equal'])) if 'padding' in row else ''} "
                  f"({row['seconds']:.1f}s)", flush=True)
    absmax.close()
    rec["summary"] = {k: summarize(v) for k, v in lists.items()}
    if pad_rec:
        rec["summary"]["padding"] = {
            "rows": len(pad_rec),
            "p1_bit_equal_all": all(all(p["p1_zero_vs_random"][k]["bit_equal"] for k in p["p1_zero_vs_random"])
                                    for p in pad_rec),
            "p1_max_abs_d": max(max(v["max_abs_d"] for v in p["p1_zero_vs_random"].values()) for p in pad_rec),
            "p2_bit_equal_all": all(all(p["p2_exact_vs_padded"][k]["bit_equal"] for k in p["p2_exact_vs_padded"])
                                    for p in pad_rec),
            "p2_max_abs_d": {k: max(p["p2_exact_vs_padded"][k]["max_abs_d"] for p in pad_rec)
                             for k in ("queries", "mask_logits_coarse", "mask_logits")},
            "p3_moved_all": all(not p["p3_nomask_zero_vs_random"]["mask_logits"]["bit_equal"] for p in pad_rec),
            "p3_logits_max_abs_d_min": min(p["p3_nomask_zero_vs_random"]["mask_logits"]["max_abs_d"] for p in pad_rec),
            "p3_alpha": summarize([dict(r, row=p["row"]) for p in pad_rec for r in p["p3_alpha"]])}
    rec["absmax"] = {"inputs": inputs_absmax, "self_attention_scores": score_absmax,
                     "modules_top": dict(sorted(absmax.max.items(), key=lambda kv: -kv[1])[:40]),
                     "overall_module_max": max(absmax.max.values()) if absmax.max else None}
    rec["wall_seconds"] = time.perf_counter() - t_all
    rec["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rec["script"] = {"mask_torch_check.py": sha256_file(Path(__file__)), "mask_graph.py": sha256_file(HERE / "mask_graph.py")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec["summary"], indent=1), flush=True)
    print(f"[torch-check] {len(rec['rows'])} rows in {rec['wall_seconds']:.0f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
