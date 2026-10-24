#!/usr/bin/env python3
"""The Swift host's fixture outputs (Engine/Fixture.swift) against the O2-448 oracle, the Core AI e2e record of
mask/mask_e2e.py (`--r2`, python runtime) and each other. CPU only.

  host inputs   patches.f16.bin vs the oracle's pixel_values (fp16), canvas.u8.bin / 255 vs dense_cnn_image (bit),
                prompt processor_ids vs the oracle's input_ids
  answer        the response vs the e2e record's D1 response (same asset, S=1) and vs the oracle's; boxes vs the oracle's
  divergence    between two forms of the same rows: the first differing generated id, the reference form's softmax
                top-2 gap there (knife edge < 0.1), and the oracle's gap when the prefix is the oracle's
  alpha         alpha_<i>.f32.bin vs the oracle's dense_alpha: MAE, IoU(alpha > 0.5); judged rows = boxes == the oracle's
                (mask_e2e.py's rule); bar MAE <= 0.01, IoU >= 0.95. For the s1-fp32 form also mask_e2e.py's own MAE / IoU
  postprocess   torch F.interpolate(sigmoid(Swift logits), bilinear, align_corners False) vs the Swift alpha (max |d|)
  box_pe        box_pe.f32.bin vs the NumPy copy of the package's arithmetic (with the torch log table) — bit

    python3 -I check_fixture.py --fixture <dir> --forms s1-fp32,pad-fp16 --oracle <oracle dir>/O2-448 --r2 <e2e json>...
        --rows <rows_r4.json> --out <json> [--ref-form s1-fp32] [--other <dir>:<form> (the device's outputs)]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

KNIFE = 0.1
TORCH_EW = {0x3E3645A0: 0x3E1E855E, 0x3E56041A: 0x3E3F66DA, 0x3E5E353F: 0x3E47183C, 0x3EB0A3D6: 0x3E93065A,
            0x3EC41892: 0x3E9DB982, 0x3F1BE76C: 0x3ECD374F, 0x3F39DB24: 0x3EDF3604}


def box_pe_np(boxes01: np.ndarray) -> np.ndarray:
    b = np.asarray(boxes01, np.float32)
    w = np.maximum(b[:, 2] - b[:, 0], np.float32(1e-6))
    h = np.maximum(b[:, 3] - b[:, 1], np.float32(1e-6))

    def enc(v):
        e = np.float32(0.2) * np.log(v) + np.float32(0.5)
        for i, x in enumerate(v.view(np.uint32).tolist()):
            if x in TORCH_EW:
                e[i] = np.array([TORCH_EW[x]], np.uint32).view(np.float32)[0]
        return e

    freqs = (2.0 ** np.arange(32)).astype(np.float32)

    def pe(v):
        ph = (np.float32(2.0 * math.pi) * v.astype(np.float32))[:, None] * freqs
        p = ph.astype(np.float64)
        return np.concatenate([np.sin(p), np.cos(p)], -1).astype(np.float32)

    return np.concatenate([pe(b[:, 0]), pe(b[:, 1]), pe(enc(w)), pe(enc(h))], -1)


def alpha_bar(a: np.ndarray, ref: np.ndarray) -> dict:
    mae = float(np.abs(a.astype(np.float64) - ref.astype(np.float64)).mean())
    p, q = a > 0.5, ref > 0.5
    union = int((p | q).sum())
    iou = 1.0 if union == 0 else float((p & q).sum()) / union
    return {"mae": mae, "iou": iou, "pass": mae <= 0.01 and iou >= 0.95, "both_empty": union == 0}


def first_div(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), None if len(a) == len(b) else n)


def load_row(d: Path):
    r = json.loads((d / "result.json").read_text())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--forms", default="s1-fp32,pad-fp16,engine-fp16")
    ap.add_argument("--ref-form", default="s1-fp32")
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--r2", action="append", default=[])
    ap.add_argument("--rows", required=True)
    ap.add_argument("--other", default="", help="<dir>:<form> of another run (the device) to compare with the same form here")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = Path(args.fixture)
    rows = json.loads(Path(args.rows).read_text())["rows"]
    r2 = {}
    for f in args.r2:
        for k, v in json.loads(Path(f).read_text())["rows"].items():
            r2[k.split(":", 1)[1]] = v["D1"]
    forms = args.forms.split(",")
    out = {"schema": "wevisone-r4-fixture-check/1", "fixture": str(base), "forms": forms, "rows": {}, "summary": {}}
    for form in forms:
        insts, judged_rows, excluded, host = [], [], [], []
        for row in rows:
            rid = row["id"]
            d = base / form / rid
            if not (d / "result.json").exists():
                continue
            r = load_row(d)
            oj = json.loads((Path(args.oracle) / f"{rid}.json").read_text())
            z = np.load(Path(args.oracle) / f"{rid}.npz")
            rec = {"response": r["response"], "boxes": r["boxes"], "stop": r["stop"], "n_hidden": r["n_hidden"],
                   "tokens": len(r["generated_ids"]), "seconds": r["seconds"]}
            # host inputs
            pv = np.fromfile(d / "patches.f16.bin", np.float16).reshape(784, 1536)
            pv_or = z["pixel_values"].astype(np.float16)
            cv = np.fromfile(d / "canvas.u8.bin", np.uint8).reshape(1024, 1024, 3) if (d / "canvas.u8.bin").exists() else None
            hrec = {"patches_bit_equal": bool(np.array_equal(pv.view(np.uint16), pv_or.view(np.uint16))),
                    "patches_values_differ": int((pv.view(np.uint16) != pv_or.view(np.uint16)).sum()),
                    "processor_ids_equal_oracle": r["processor_ids"] == z["input_ids"].tolist()}
            if cv is not None:
                c = (cv.astype(np.float32) / np.float32(255.0)).transpose(2, 0, 1)
                hrec["canvas_bit_equal"] = bool(np.array_equal(c, z["dense_cnn_image"]))
                hrec["canvas_pixels_differ"] = int((np.abs(c - z["dense_cnn_image"]).max(0) > 0).sum())
            rec["host"] = hrec
            host.append(hrec)
            # answer
            rec["response_equal_r2_e2e"] = (r["response"] == r2[rid]["response"]) if rid in r2 else None
            rec["response_equal_oracle"] = r["response"] == oj["response"]
            boxes_or = [[int(round(v * 1000)) for v in b] for b in oj["dense"]["boxes01"]]
            rec["boxes_oracle"] = boxes_or
            rec["boxes_equal_oracle"] = r["boxes"] == boxes_or
            gen_or = [int(t) for t in z["gen_ids"]]
            dv = first_div(r["generated_ids"], gen_or)
            rec["vs_oracle_ids"] = {"first_divergence": dv,
                                    "oracle_gap": float(z["gap_softmax"][dv]) if dv is not None and dv < len(z["gap_softmax"]) else None,
                                    "swift_gap": float(r["gaps"][dv]) if dv is not None and dv < len(r["gaps"]) else None}
            # divergence from the reference form
            if form != args.ref_form and (base / args.ref_form / rid / "result.json").exists():
                ref = load_row(base / args.ref_form / rid)
                dv2 = first_div(r["generated_ids"], ref["generated_ids"])
                rec["vs_ref_form"] = {"form": args.ref_form, "identical": dv2 is None, "first_divergence": dv2,
                                      "ref_gap": float(ref["gaps"][dv2]) if dv2 is not None and dv2 < len(ref["gaps"]) else None,
                                      "this_gap": float(r["gaps"][dv2]) if dv2 is not None and dv2 < len(r["gaps"]) else None,
                                      "prefix_is_oracle": (dv is None or dv >= (dv2 or 0)),
                                      "oracle_gap": float(z["gap_softmax"][dv2]) if dv2 is not None and dv2 < len(z["gap_softmax"]) and (dv is None or dv > dv2) else None}
                if dv2 is not None:
                    g = rec["vs_ref_form"]["ref_gap"]
                    rec["vs_ref_form"]["knife_edge"] = g is not None and g < KNIFE
            # alpha
            a_or = z["dense_alpha"]
            h, w = r["image"]["height"], r["image"]["width"]
            n = len(r["boxes"])
            alist = []
            for i in range(n):
                f = d / f"alpha_{i}.f32.bin"
                if not f.exists() or i >= a_or.shape[0]:
                    continue
                a = np.fromfile(f, np.float32).reshape(h, w)
                m = alpha_bar(a, a_or[i])
                m["instance"] = i
                alist.append(m)
            rec["alpha_vs_oracle"] = alist
            judged = rec["boxes_equal_oracle"] and n == a_or.shape[0]
            rec["judged"] = judged
            if judged:
                judged_rows.append(rid)
                for m in alist:
                    insts.append(dict(m, row=rid, bar_row=row.get("bar", False)))
            else:
                excluded.append({"row": rid, "boxes": r["boxes"], "boxes_oracle": boxes_or})
            if form == args.ref_form and rid in r2:
                r2a = r2[rid].get("alpha") or r2[rid].get("alpha_reference_only") or []
                rec["r2_e2e_alpha"] = [{"mae": x["mae"], "iou": x["iou"]} for x in r2a]
                rec["vs_r2_alpha"] = [{"mae_swift": m["mae"], "mae_r2": x["mae"], "rel": abs(m["mae"] - x["mae"]) / max(x["mae"], 1e-12),
                                       "iou_swift": m["iou"], "iou_r2": x["iou"]} for m, x in zip(alist, r2a)]
            # postprocess (first instance)
            if n and (d / "logits_0.f32.bin").exists() and (d / "alpha_0.f32.bin").exists():
                lg = torch.from_numpy(np.fromfile(d / "logits_0.f32.bin", np.float32).reshape(1, 1, 1024, 1024))
                pt = F.interpolate(lg.sigmoid(), size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
                a0 = np.fromfile(d / "alpha_0.f32.bin", np.float32).reshape(h, w)
                rec["postprocess_max_abs_d"] = float(np.abs(pt - a0).max())
            # box_pe
            pe = np.fromfile(d / "box_pe.f32.bin", np.float32)
            if pe.size and n:
                pe = pe.reshape(-1, 256)
                bx = np.array([[float(np.float32(v / 1000.0)) for v in b] for b in r["boxes"]], np.float32)
                pad = np.tile(np.array([[0.25, 0.25, 0.75, 0.75]], np.float32), (pe.shape[0] - len(bx) % 16 if len(bx) % 16 else 0, 1))
                full = np.concatenate([bx, pad]) if len(pad) else bx
                ref = box_pe_np(full)
                rec["box_pe_bit_equal_numpy_table"] = bool(np.array_equal(pe[: ref.shape[0]].view(np.uint32), ref.view(np.uint32)))
            out["rows"][f"{form}:{rid}"] = rec
        s = {"rows": len(host), "judged_rows": judged_rows, "excluded": excluded, "instances": len(insts)}
        if insts:
            worst = max(insts, key=lambda m: m["mae"])
            s.update({"mae_max": worst["mae"], "mae_max_at": f"{worst['row']}#{worst['instance']}",
                      "iou_min": min(m["iou"] for m in insts), "all_pass": all(m["pass"] for m in insts),
                      "bar_rows_instances": sum(1 for m in insts if m["bar_row"]),
                      "bar_rows_pass": all(m["pass"] for m in insts if m["bar_row"])})
        s["host_patches_bit_equal_rows"] = sum(1 for h in host if h["patches_bit_equal"])
        s["host_canvas_bit_equal_rows"] = sum(1 for h in host if h.get("canvas_bit_equal"))
        s["processor_ids_equal_rows"] = sum(1 for h in host if h["processor_ids_equal_oracle"])
        resp = [out["rows"][f"{form}:{r['id']}"] for r in rows if f"{form}:{r['id']}" in out["rows"]]
        s["response_equal_r2_e2e_rows"] = sum(1 for x in resp if x.get("response_equal_r2_e2e"))
        s["response_equal_oracle_rows"] = sum(1 for x in resp if x.get("response_equal_oracle"))
        if form != args.ref_form:
            s["identical_to_ref_form_rows"] = sum(1 for x in resp if x.get("vs_ref_form", {}).get("identical"))
            s["divergences_vs_ref_form"] = [{"row": rid, **x["vs_ref_form"]} for rid, x in
                                            ((r["id"], out["rows"].get(f"{form}:{r['id']}", {})) for r in rows)
                                            if x and not x.get("vs_ref_form", {}).get("identical", True)]
        s["postprocess_max_abs_d"] = max((x.get("postprocess_max_abs_d", 0) for x in resp), default=None)
        s["box_pe_bit_equal_rows"] = sum(1 for x in resp if x.get("box_pe_bit_equal_numpy_table"))
        out["summary"][form] = s
    if args.other:
        odir, oform = args.other.rsplit(":", 1)
        cmp_rows = []
        for row in rows:
            rid = row["id"]
            a, b = base / oform / rid, Path(odir) / oform / rid
            if not ((a / "result.json").exists() and (b / "result.json").exists()):
                continue
            ra, rb = load_row(a), load_row(b)
            dv = first_div(rb["generated_ids"], ra["generated_ids"])
            ref_s1 = base / args.ref_form / rid
            gap = None
            if dv is not None and (ref_s1 / "result.json").exists():
                s1 = load_row(ref_s1)
                if first_div(s1["generated_ids"], ra["generated_ids"]) is None or (first_div(s1["generated_ids"], ra["generated_ids"]) or 0) > dv:
                    gap = float(s1["gaps"][dv]) if dv < len(s1["gaps"]) else None
            rec = {"row": rid, "identical": dv is None, "first_divergence": dv, "mac_s1_gap": gap,
                   "boxes_equal": ra["boxes"] == rb["boxes"]}
            if ra["boxes"] == rb["boxes"]:
                h, w = ra["image"]["height"], ra["image"]["width"]
                ms = []
                for i in range(len(ra["boxes"])):
                    la, lb = a / f"logits_{i}.f32.bin", b / f"logits_{i}.f32.bin"
                    if la.exists() and lb.exists():
                        ta = torch.from_numpy(np.fromfile(la, np.float32).reshape(1, 1, 1024, 1024))
                        tb = torch.from_numpy(np.fromfile(lb, np.float32).reshape(1, 1, 1024, 1024))
                        pa = F.interpolate(ta.sigmoid(), size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
                        pb = F.interpolate(tb.sigmoid(), size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
                        m = alpha_bar(pb, pa)
                        m["instance"] = i
                        ms.append(m)
                rec["alpha_vs_mac"] = ms
            cmp_rows.append(rec)
        out["other"] = {"dir": odir, "form": oform, "rows": cmp_rows,
                        "identical_rows": sum(1 for r in cmp_rows if r["identical"]),
                        "alpha_mae_max": max((m["mae"] for r in cmp_rows for m in r.get("alpha_vs_mac", [])), default=None),
                        "alpha_iou_min": min((m["iou"] for r in cmp_rows for m in r.get("alpha_vs_mac", [])), default=None)}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps(out["summary"], indent=1, default=str))
    if "other" in out:
        print(json.dumps({k: v for k, v in out["other"].items() if k != "rows"}, indent=1, default=str))


if __name__ == "__main__":
    main()
