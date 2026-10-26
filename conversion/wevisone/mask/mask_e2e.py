#!/usr/bin/env python3
"""(D) The mask path end to end on Core AI parts only, against the oracle's captured alpha.

image -> vision tower with taps (Core AI) -> decoder with last_hidden (Core AI; prompt + free-run greedy, S=1 "main",
no repetition penalty, <= --max-new) -> the boxes parsed from its answer (the package's `parse_seg_boxes`) and the
hidden state of every fed `<OBJ>` -> graph A once + graph B per instance (Core AI) -> alpha through the package's
`_masks_from_logits` at the original size -> vs the oracle's `dense_alpha` (bar per instance MAE <= 0.01, IoU >= 0.95).
  D1    the oracle's own tensors in: pixel_values (fp16) and the 1024 x 1024 canvas
  D2    the host's tensors: the original photo through mask_host.py (Pillow's BICUBIC written in NumPy) to the
        grid's side -> patches, and to 1024 -> canvas. When both come out bit-equal to D1's inputs the Core AI
        chain is the same computation and D2 = D1 is recorded without re-running it.
  judged rows = the free-run's boxes (integer bins) equal the oracle's; other rows are listed with their numbers
  D-tf  (reference only) the oracle's generated ids teacher-forced through the decoder, the oracle's boxes

    WATCH -- <coreai venv>/bin/python mask_e2e.py --vision <aimodelc> --dec-bundle <dir> --dec <aimodelc> \
        --a <aimodelc> --b <aimodelc> --pkg <dir> --answers <dir> --fixtures <json> [--fixtures <json>] --out <json>
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
import mask_host as HOST  # noqa: E402
from mask_coreai_run import Graph  # noqa: E402
from mask_hidden_run import OBJ, HDecoder  # noqa: E402
from mask_torch_check import alpha_bar, alpha_of, summarize  # noqa: E402

FORM_FOR_GRID = {14: "O2-448", 16: "O2-512"}


async def main_async(args) -> int:
    import coreai.runtime as rt
    import tokenizers
    from PIL import Image

    g = args.grid
    form = FORM_FOR_GRID[g]
    MG.import_package(args.pkg)
    from wevisone_vl import engine as E  # noqa: E402

    t_all = time.perf_counter()
    out = {"schema": "wevisone-r2-e2e/1", "grid": g, "form": form, "window": G.window_record(args.label),
           "environment": G.env_record(), "rows": {}}
    images = {}
    for fx in args.fixtures:
        base = Path(fx).resolve().parent
        for r in json.loads(Path(fx).read_text())["rows"]:
            images[r["id"]] = base / r["image"]
    vision = await Graph(rt, Path(args.vision).resolve()).load()
    ga = await Graph(rt, Path(args.a).resolve()).load()
    gb = await Graph(rt, Path(args.b).resolve()).load()
    bundle = Path(args.dec_bundle).resolve()
    meta = json.loads((bundle / "metadata.json").read_text())
    dec = await HDecoder.load(rt, Path(args.dec).resolve(), int(meta["language"]["max_context_length"]), g * g, 0)
    dec.check_contract()
    tok = tokenizers.Tokenizer.from_file(str(bundle / "tokenizer" / "tokenizer.json"))
    n_max = int(ga.desc["inputs"]["t_seg"][0][0])
    out["assets"] = {k: {"path": str(x.path), "load_seconds": x.load_s} for k, x in
                     (("vision", vision), ("graph_a", ga), ("graph_b", gb))}
    out["assets"]["decoder"] = {"path": args.dec, "bundle": str(bundle), "outputs": list(dec.desc["outputs"])}
    lists = {"D1": [], "D2": [], "Dtf": [], "D1_excluded": []}
    want = set(filter(None, args.rows.split(",")))

    async def tower(pv16):
        t0 = time.perf_counter()
        v = await vision(patches=pv16)
        return v, (time.perf_counter() - t0) * 1e3

    async def decode_free(static, mapped):
        state = dec.fresh_state()
        t0 = time.perf_counter()
        lg = None
        for t, tk in enumerate(mapped):
            lg, _ = await dec.step2(int(tk), t, static, state)
        prompt_ms = (time.perf_counter() - t0) * 1e3
        gen, hid, t = [], [], len(mapped)
        nxt = int(np.argmax(lg.astype(np.float32)))
        t0 = time.perf_counter()
        while True:
            gen.append(nxt)
            if nxt in G.STOP_IDS or len(gen) >= args.max_new:
                break
            lg, hd = await dec.step2(nxt, t, static, state)
            if nxt == OBJ:
                hid.append(hd.astype(np.float32).copy())
            t += 1
            nxt = int(np.argmax(lg.astype(np.float32)))
        return gen, hid, prompt_ms, (time.perf_counter() - t0) * 1e3

    async def decode_tf(static, seq):
        state = dec.fresh_state()
        hid = []
        for t, tk in enumerate(seq):
            _, hd = await dec.step2(int(tk), t, static, state)
            if int(tk) == OBJ:
                hid.append(hd.astype(np.float32).copy())
        return hid

    async def mask(canvas, taps, t_seg, boxes):
        """graph A per chunk of n_max + graph B per instance -> mask_logits [N, 1024, 1024]."""
        logits, ms_a, ms_b = [], [], []
        n = len(boxes)
        for s in range(0, n, n_max):
            e = min(n, s + n_max)
            seg, box, valid = MG.pad_queries(torch.from_numpy(np.asarray(t_seg[s:e], np.float32)),
                                             torch.from_numpy(np.asarray(boxes[s:e], np.float32)), n_max)
            extra = {"box_pe": MG.box_fourier_np(box.numpy())} if "box_pe" in ga.desc["inputs"] else {}
            t0 = time.perf_counter()
            a = await ga(image=canvas[None], tap5=taps[0][None], tap11=taps[1][None], tap17=taps[2][None],
                         tap23=taps[3][None], t_seg=seg.numpy(), boxes01=box.numpy(), valid=valid.numpy(), **extra)
            ms_a.append((time.perf_counter() - t0) * 1e3)
            for i in range(e - s):
                t0 = time.perf_counter()
                b = await gb(image=canvas[None], semantic_s2=a["semantic_s2"], query=a["queries"][i:i + 1],
                             coarse_logits=a["coarse_logits"][i:i + 1])
                ms_b.append((time.perf_counter() - t0) * 1e3)
                logits.append(b["mask_logits"])
        return np.concatenate(logits, 0), ms_a, ms_b

    for jf in sorted((Path(args.answers) / form).glob("*.json")):
        mj = json.loads(jf.read_text())
        if "dense" not in mj or not mj["dense"].get("count_decoded"):
            continue
        rid = mj["id"]
        if want and rid not in want:
            continue
        key = f"{form}:{rid}"
        z = np.load(jf.with_suffix(".npz"))
        hw = mj["dense"]["alpha_size_hw"]
        a_or = z["dense_alpha"].astype(np.float32)
        boxes_or = [list(map(float, b)) for b in mj["dense"]["boxes01"]]
        mapped, start, amount = G.host_inputs(z["input_ids"].tolist(), g)
        row = {"n_obj_oracle": int(a_or.shape[0]), "boxes_oracle": boxes_or}
        # inputs: D1 = the oracle's tensors, D2 = the host's
        pv1 = z["pixel_values"].astype(np.float16)
        cv1 = z["dense_cnn_image"].astype(np.float32)
        rgb = np.asarray(Image.open(images[rid]).convert("RGB"))
        side = 32 * g
        pv2 = HOST.patches(HOST.pil_bicubic(rgb, side, side)).astype(np.float16)
        cv2 = HOST.canvas(rgb, 1024)
        row["host_inputs_vs_oracle"] = {
            "patches_fp16_bit_equal": bool(np.array_equal(pv1, pv2)),
            "patches_max_abs_d": float(np.abs(pv1.astype(np.float32) - pv2.astype(np.float32)).max()),
            "canvas_bit_equal": bool(np.array_equal(cv1, cv2)),
            "canvas_max_abs_d": float(np.abs(cv1 - cv2).max()),
            "canvas_pixels_differ": int((np.abs(cv1 - cv2).max(0) > 0).sum())}
        stages = [("D1", pv1, cv1)]
        same = row["host_inputs_vs_oracle"]["patches_fp16_bit_equal"] and row["host_inputs_vs_oracle"]["canvas_bit_equal"]
        if not same:
            stages.append(("D2", pv2, cv2))
        for name, pv, cv in stages:
            v, vis_ms = await tower(pv)
            taps = v["taps"].astype(np.float32)
            static = dec.static(v["image_embeds"], v["deepstack_embeds"].reshape(3, g * g, -1), start, amount)
            gen, hid, p_ms, d_ms = await decode_free(static, mapped)
            raw = tok.decode(gen, skip_special_tokens=False)
            response = raw.split("<|im_end|>", 1)[0].split("<|endoftext|>", 1)[0].strip()
            boxes = E.parse_seg_boxes(response)
            st = {"response": response, "equal_to_oracle_ids": gen == [int(t) for t in z["gen_ids"]],
                  "boxes": boxes, "boxes_equal_oracle": boxes == boxes_or, "n_hidden": len(hid),
                  "ms": {"vision": vis_ms, "prompt": p_ms, "decode": d_ms, "decode_tokens": len(gen)}}
            n = min(len(boxes), len(hid))
            if n:
                lg, ms_a, ms_b = await mask(cv, taps, np.stack(hid[:n]), boxes[:n])
                st["ms"].update({"graph_a": ms_a, "graph_b": ms_b})
                a_ca = alpha_of(E, torch.from_numpy(lg), hw)
                if st["boxes_equal_oracle"] and n == a_or.shape[0]:
                    insts = alpha_bar(a_ca, a_or)
                    for r in insts:
                        r["row"] = key
                    st["alpha"] = insts
                    lists[name].extend(insts)
                else:
                    st["alpha_reference_only"] = (alpha_bar(a_ca[:min(n, a_or.shape[0])], a_or[:min(n, a_or.shape[0])])
                                                  if n else [])
                    if name == "D1":
                        lists["D1_excluded"].append({"row": key, "boxes": boxes, "boxes_oracle": boxes_or})
            row[name] = st
            if name == "D1":
                v1, taps1 = v, taps
        if same:
            row["D2"] = {"same_as_D1": True}
            if "alpha" in row["D1"]:
                lists["D2"].extend(dict(r, stage="D2=D1") for r in row["D1"]["alpha"])
        # D-tf: the oracle's ids teacher-forced, the oracle's boxes, the D1 vision outputs
        static = dec.static(v1["image_embeds"], v1["deepstack_embeds"].reshape(3, g * g, -1), start, amount)
        seq = np.concatenate([mapped, z["gen_ids"].astype(np.int64)])
        hid_tf = await decode_tf(static, seq)
        n = min(len(boxes_or), len(hid_tf))
        if n:
            lg, _, _ = await mask(cv1, taps1, np.stack(hid_tf[:n]), boxes_or[:n])
            insts = alpha_bar(alpha_of(E, torch.from_numpy(lg), hw), a_or[:n])
            for r in insts:
                r["row"] = key
            row["Dtf"] = {"alpha": insts, "hidden_cos_vs_oracle_tf": [G.cos(h, t) for h, t in zip(hid_tf, z["dense_t_seg"])]}
            lists["Dtf"].extend(insts)
        out["rows"][key] = row
        d1 = row["D1"]
        print(f"[e2e g{g}] {key}: ids==oracle {d1['equal_to_oracle_ids']} boxes==oracle {d1['boxes_equal_oracle']} "
              f"n {len(d1['boxes'])}/{a_or.shape[0]} | D1 alpha "
              f"{'MAE max %.2e IoU min %.5f' % (max(r['mae'] for r in d1['alpha']), min(r['iou'] for r in d1['alpha'])) if 'alpha' in d1 else 'excluded'}"
              f" | host inputs same {same} | Dtf "
              f"{'MAE max %.2e' % max(r['mae'] for r in row['Dtf']['alpha']) if 'Dtf' in row else '-'}", flush=True)
    out["summary"] = {k: (summarize(v) if k != "D1_excluded" else v) for k, v in lists.items()}
    out["summary"]["rows"] = len(out["rows"])
    out["ms_per_call"] = {k: {"median": float(np.median(x.ms)) if x.ms else None, "n": len(x.ms)}
                          for k, x in (("vision", vision), ("graph_a", ga), ("graph_b", gb))}
    out["mpsgraph_scratch_cleanup"] = G.clean_own_mpsgraph_scratch()
    out["wall_seconds"] = time.perf_counter() - t_all
    out["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["script_sha256"] = G.sha256_file(Path(__file__))
    G.write_json(Path(args.out), out)
    print(json.dumps(out["summary"], indent=1, default=str), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vision", required=True)
    ap.add_argument("--dec-bundle", required=True)
    ap.add_argument("--dec", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--fixtures", action="append", required=True)
    ap.add_argument("--grid", type=int, default=14, choices=[14, 16])
    ap.add_argument("--rows", default="")
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="wv-c-r2")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
