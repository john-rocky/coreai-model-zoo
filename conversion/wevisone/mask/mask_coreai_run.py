#!/usr/bin/env python3
"""The two-graph mask decoder on Core AI (python runtime, AOT h16c `.aimodelc`, `SpecializationOptions.default()`,
one load per asset per process) against the torch fp32 package, on every dense row of the given forms.

Per row (the oracle's captured decoder inputs: tap maps, `<OBJ>` hidden states, boxes, 1024 x 1024 canvas):
  ref      the package's `_forward_fp32` (torch fp32, CPU, chunks of 16) -> alpha through the package's own
           `_masks_from_logits` at the original size
  chain    Core AI graph A (padded to its n_max) -> Core AI graph B per instance -> alpha   = the (A) gate
  iso      torch GraphA outputs -> Core AI graph B (graph B's own error)
  table    Core AI vs torch GraphA on the same padded inputs: memory, queries, coarse, semantic_s2 (real rows), and
           graph B's logits_s2 / logits_s1 (iso) — max|d| and cos, to name the stage when a row misses the bar
  p1       Core AI graph A with random padding vs zero padding: real rows must be bit-equal
  repeat   the first row's graph A and B calls run twice: bit-equal
  bar      per instance MAE <= 0.01 and IoU(alpha > 0.5) >= 0.95 (ROUND2.md)

    WATCH -- <coreai venv>/bin/python mask_coreai_run.py --a28 <aimodelc> --a32 <aimodelc> --b <aimodelc> \
        --pkg <dir> --sidecar <pt> --answers <dir> --out <json> [--forms O2-448,O2-512] [--rows a,b] [--label x]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))
import gate_common as G  # noqa: E402
import mask_graph as MG  # noqa: E402
from mask_torch_check import GRID_FOR_FORM, alpha_bar, alpha_of, package_forward, summarize  # noqa: E402

A_OUT = ("queries", "coarse_logits", "semantic_s2", "memory")


def np_cmp(a: np.ndarray, b: np.ndarray) -> dict:
    a64, b64 = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    d = np.abs(a64 - b64)
    return {"max_abs_d": float(d.max()), "mean_abs_d": float(d.mean()), "bit_equal": bool(np.array_equal(a, b)),
            "cos": float(a64 @ b64 / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-300)),
            "ref_absmax": float(np.abs(b64).max()), "finite": bool(np.isfinite(a64).all())}


class Graph:
    def __init__(self, rt, path: Path):
        self.rt, self.path = rt, path

    async def load(self):
        t0 = time.perf_counter()
        self.model = await G.maybe(self.rt.AIModel.load(str(self.path), self.rt.SpecializationOptions.default()))
        t1 = time.perf_counter()
        self.fn = await G.maybe(self.model.load_function("main"))
        self.load_s = {"model": t1 - t0, "function": time.perf_counter() - t1}
        self.desc = G.fn_desc(self.fn)
        self.dtypes = {k: np.dtype(v[1]) for k, v in self.desc["inputs"].items()}
        self.ms: list[float] = []
        return self

    async def __call__(self, **inputs) -> dict:
        feed = {k: G.nd(self.rt, np.asarray(v, self.dtypes[k])) for k, v in inputs.items()}
        t0 = time.perf_counter()
        res = await G.maybe(self.fn(inputs=feed))
        out = {k: np.asarray(res[k].numpy()).astype(np.float32) for k in self.desc["outputs"]}
        self.ms.append((time.perf_counter() - t0) * 1e3)
        return out


async def main_async(args) -> int:
    import coreai.runtime as rt

    torch.set_num_threads(4)
    t_all = time.perf_counter()
    win = G.window_record(args.label)
    dec, cfg, _ = MG.load_package_decoder(args.pkg, args.sidecar)
    from wevisone_vl import engine as E  # noqa: E402

    graphs = {}
    for key, p in (("a28", args.a28), ("a32", args.a32), ("b", args.b)):
        if p:
            graphs[key] = await Graph(rt, Path(p).resolve()).load()
    b = graphs["b"]
    want = set(filter(None, args.rows.split(",")))
    out = {"schema": "wevisone-r2-mask-coreai/1", "label": args.label, "window": win, "environment": G.env_record(),
           "assets": {k: {"path": str(g.path), "digest": G.tree_digest(g.path), "descriptor": g.desc,
                          "load_seconds": g.load_s} for k, g in graphs.items()},
           "bar": {"mae": 0.01, "iou": 0.95}, "rows": {}}
    lists = {"chain_vs_pkg": [], "iso_vs_pkg": []}
    first = True
    gs = G.GpuSampler().__enter__()
    for form in args.forms.split(","):
        g = GRID_FOR_FORM[form]
        a = graphs.get(f"a{g}")
        if a is None:
            continue
        n_max = int(a.desc["inputs"]["t_seg"][0][0])
        hf = "box_pe" in a.desc["inputs"]          # the host-Fourier graph A takes the box features as an input
        ta = (MG.GraphAHF if hf else MG.GraphA)(dec, g, n_max=n_max).eval()
        out.setdefault("host_fourier", {})[form] = hf
        tb = MG.GraphB(dec).eval()
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
            with torch.no_grad():
                pkg = package_forward(dec, image, maps, t_seg, boxes, g)
            a_pkg = alpha_of(E, pkg["mask_logits"], hw)
            row = {"form": form, "id": rid, "n": n, "grid": g, "alpha_hw": hw, "stages": [], "chunks": []}
            chain_logits, iso_logits = [], []
            for s in range(0, n, n_max):
                e = min(n, s + n_max)
                k = e - s
                seg, box, valid = MG.pad_queries(t_seg[s:e], boxes[s:e], n_max)
                a_in = {"image": image.numpy(), "tap5": maps[0].numpy(), "tap11": maps[1].numpy(),
                        "tap17": maps[2].numpy(), "tap23": maps[3].numpy(), "t_seg": seg.numpy(),
                        "boxes01": box.numpy(), "valid": valid.numpy()}
                extra = ()
                if hf:
                    a_in["box_pe"] = MG.box_fourier_np(box.numpy())
                    extra = (torch.from_numpy(a_in["box_pe"]),)
                ca = await a(**a_in)
                with torch.no_grad():
                    tq, tc, tsem, tmem = ta(image, *maps, seg, box, *extra, valid)
                t_a = {"queries": tq.numpy(), "coarse_logits": tc.numpy(), "semantic_s2": tsem.numpy(),
                       "memory": tmem.numpy()}
                stage = {"chunk": [s, e],
                         "a_memory": np_cmp(ca["memory"], t_a["memory"]),
                         "a_queries": np_cmp(ca["queries"][:k], t_a["queries"][:k]),
                         "a_coarse": np_cmp(ca["coarse_logits"][:k], t_a["coarse_logits"][:k]),
                         "a_semantic_s2": np_cmp(ca["semantic_s2"], t_a["semantic_s2"])}
                # p1 on the GPU: random padding contents must leave the real rows bit-equal
                gen = torch.Generator().manual_seed(1234)
                rnd_seg = torch.randn(n_max, t_seg.shape[1], generator=gen) * t_seg.std()
                lo = torch.rand(n_max, 2, generator=gen) * 0.5
                rnd_box = torch.cat([lo, lo + 0.1 + torch.rand(n_max, 2, generator=gen) * 0.4], 1)
                seg_r, box_r, _ = MG.pad_queries(t_seg[s:e], boxes[s:e], n_max, pad_seg=rnd_seg, pad_box=rnd_box)
                a_r = {**a_in, "t_seg": seg_r.numpy(), "boxes01": box_r.numpy()}
                if hf:
                    a_r["box_pe"] = MG.box_fourier_np(box_r.numpy())
                ca_r = await a(**a_r)
                stage["p1_gpu"] = {o: np_cmp(ca_r[o][:k] if o in ("queries", "coarse_logits") else ca_r[o],
                                             ca[o][:k] if o in ("queries", "coarse_logits") else ca[o])
                                   for o in A_OUT}
                for i in range(k):
                    cb = await b(image=image.numpy(), semantic_s2=ca["semantic_s2"], query=ca["queries"][i:i + 1],
                                 coarse_logits=ca["coarse_logits"][i:i + 1])
                    ib = await b(image=image.numpy(), semantic_s2=t_a["semantic_s2"], query=t_a["queries"][i:i + 1],
                                 coarse_logits=t_a["coarse_logits"][i:i + 1])
                    with torch.no_grad():
                        l1, l2 = tb(image, tsem, tq[i:i + 1], tc[i:i + 1])
                    stage.setdefault("b_iso", []).append({"mask_logits": np_cmp(ib["mask_logits"], l1.numpy()),
                                                          "mask_logits_aux": np_cmp(ib["mask_logits_aux"], l2.numpy())})
                    stage.setdefault("b_chain_vs_torch", []).append(
                        {"mask_logits": np_cmp(cb["mask_logits"], l1.numpy()),
                         "mask_logits_aux": np_cmp(cb["mask_logits_aux"], l2.numpy())})
                    chain_logits.append(cb["mask_logits"])
                    iso_logits.append(ib["mask_logits"])
                    if first:
                        cb2 = await b(image=image.numpy(), semantic_s2=ca["semantic_s2"],
                                      query=ca["queries"][i:i + 1], coarse_logits=ca["coarse_logits"][i:i + 1])
                        ca2 = await a(**a_in)
                        out["repeat_bit_equal"] = {"a": {o: bool(np.array_equal(ca2[o], ca[o])) for o in A_OUT},
                                                   "b": bool(np.array_equal(cb2["mask_logits"], cb["mask_logits"]))}
                        first = False
                row["stages"].append(stage)
            a_chain = alpha_of(E, torch.from_numpy(np.concatenate(chain_logits, 0)), hw)
            a_iso = alpha_of(E, torch.from_numpy(np.concatenate(iso_logits, 0)), hw)
            for name, cand in (("chain_vs_pkg", a_chain), ("iso_vs_pkg", a_iso)):
                insts = alpha_bar(cand, a_pkg)
                for r in insts:
                    r["row"] = key
                row[f"alpha_{name}"] = insts
                lists[name].extend(insts)
            row["logits_chain_vs_pkg"] = np_cmp(np.concatenate(chain_logits, 0), pkg["mask_logits"].numpy())
            row["seconds"] = time.perf_counter() - t0
            out["rows"][key] = row
            ac = row["alpha_chain_vs_pkg"]
            st = row["stages"][0]
            print(f"[mask-ca] {key} N={n}: alpha MAE max {max(r['mae'] for r in ac):.2e} IoU min "
                  f"{min(r['iou'] for r in ac):.5f} | queries cos {st['a_queries']['cos']:.7f} max|d| "
                  f"{st['a_queries']['max_abs_d']:.2e} | logits max|d| {row['logits_chain_vs_pkg']['max_abs_d']:.3g} "
                  f"| p1 bit {all(v['bit_equal'] for v in st['p1_gpu'].values() if True)} ({row['seconds']:.1f}s)",
                  flush=True)
    gs.__exit__()
    out["summary"] = {k: summarize(v) for k, v in lists.items()}
    out["summary"]["rows"] = len(out["rows"])
    out["summary"]["n_ge_2_rows"] = sorted(k for k, r in out["rows"].items() if r["n"] >= 2)
    out["summary"]["p1_gpu_bit_equal_all"] = all(
        all(v["bit_equal"] for v in st["p1_gpu"].values()) for r in out["rows"].values() for st in r["stages"])
    out["summary"]["all_finite"] = all(
        st[k]["finite"] for r in out["rows"].values() for st in r["stages"]
        for k in ("a_memory", "a_queries", "a_coarse", "a_semantic_s2"))
    out["ms_per_call"] = {k: {"n": len(g.ms), "median": float(np.median(g.ms)) if g.ms else None,
                              "min": float(np.min(g.ms)) if g.ms else None} for k, g in graphs.items()}
    out["gpu_utilization"] = gs.summary()
    out["mpsgraph_scratch_cleanup"] = G.clean_own_mpsgraph_scratch()
    out["wall_seconds"] = time.perf_counter() - t_all
    out["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["script"] = {"mask_coreai_run.py": G.sha256_file(Path(__file__)),
                     "mask_graph.py": G.sha256_file(HERE / "mask_graph.py")}
    G.write_json(Path(args.out), out)
    print(json.dumps(out["summary"], indent=1), flush=True)
    print(f"[mask-ca] {len(out['rows'])} rows in {out['wall_seconds']:.0f}s -> {args.out}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a28", default="")
    ap.add_argument("--a32", default="")
    ap.add_argument("--b", required=True)
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--forms", default="O2-448,O2-512")
    ap.add_argument("--rows", default="")
    ap.add_argument("--label", default="wv-c-r2")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
