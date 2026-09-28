#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.9.0",
#     "transformers==5.17.0",
#     "torchvision",
#     "pillow",
#     "safetensors>=0.8.0",
#     "huggingface_hub>=1.5.0,<2",
#     "numpy",
#     "pyarrow",
# ]
# [tool.uv]
# index-url = "https://pypi.org/simple"
# ///
"""What a fixed image grid costs decider-2b-vision, measured on the author's held-out sets (HF only).

A Core AI tower bakes one grid; the checkpoint's processor picks a grid per image (dynamic
resolution, min 65,536 px). This script answers "how often does a fixed 256x256 (64 tokens) or
448x448 (196 tokens) input change the model's answer, compared with the author's own dynamic
grid?" on the two subsets the author held out of training (`decider/vision/data.py` SUBSETS:
visual7w and vsr are `held_out=True`), before anything is exported.

Rows come from The Cauldron parquet shards (`HuggingFaceM4/the_cauldron`, pinned revision), taken
in file order, filtered and converted exactly as the author's `decider/vision/data.py` does (one
image, first text turn, the same `parse()`, the same context string, images whose longer side
exceeds 768 are `thumbnail((768, 768))`-ed first). Nothing from the dataset is written out except
numbers: no image, question or option text leaves this process.

Arms, all through the checkpoint's own `VisionDecisionModel.prepare()` -> `slot_logits()`:
`native` = the thumbnailed image (the processor chooses the grid), `g256` / `g448` = that image
resized with PIL BICUBIC to 256x256 / 448x448 (grid asserted). One device and dtype for every arm
(MPS fp32 when it reproduces the CPU fp32 oracle on fixture rows, else CPU fp32 with half the
items). Wall seconds are recorded but are contended reference values, not benchmarks.

    uv run conversion/decider_vision/grid_price.py                    # -> $ZOO_WORK_ROOT/_decider2bv/price
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import hf_snapshot, work_path  # noqa: E402

DEFAULT_HF_ID = "Mapika/decider-2b-vision"
DEFAULT_REVISION = "863e290863655f1d6b69324d77d09ac972d21609"
DATASET = "HuggingFaceM4/the_cauldron"
DATASET_REVISION = "847a98a779b1652d65111daf20c972dfcd333605"
SHARDS = {  # the first shard of each held-out subset: rows are taken in file order
    "visual7w": "visual7w/train-00000-of-00009-c6610d0b14d32cb3.parquet",
    "vsr": "vsr/train-00000-of-00001-b56e9224d46b0ed3.parquet",
}
IMAGE_PAD = 248056
GRIDS = {"g256": 256, "g448": 448}
ARMS = ("native", "g256", "g448")

# --- copied from Mapika/decider decider/vision/data.py @ origin/main 23579f7 (Apache-2.0), unchanged ---
LETTER_RX = re.compile(r"^([A-J])\.\s*(.+)$")


def parse(sub, user, assistant):
    """Return (context, question, options, gold) or None."""
    u = user.strip(); a = assistant.strip()
    if sub in ("vsr", "nlvr2", "hateful_memes"):
        q = u.split("\n")[0].strip(); g = 1 if a.lower().startswith("yes") else 0 if a.lower().startswith("no") else None
        if g is None: return None
        return "", q, ["no", "yes"], g
    if sub == "aokvqa":
        m = re.search(r"Options:\s*(.+)$", u, re.S)
        if not m: return None
        opts = [o.strip().rstrip(".") for o in m.group(1).split(",") if o.strip()]
        q = u.split("\n")[0].strip(); ans = a.rstrip(".").strip().lower()
        golds = [i for i, o in enumerate(opts) if o.lower() == ans]
        if len(opts) < 2 or not golds: return None
        return "", q, opts, golds[0]
    if sub == "raven":
        # options are in the image; the answer is a letter A-H
        m = re.match(r"^([A-H])", a)
        if not m: return None
        opts = list("ABCDEFGH"); return "", u.split("\n")[0].strip(), [f"figure {L}" for L in opts], opts.index(m.group(1))
    # letter-choice subsets: "Question: ...\nChoices:\nA. x\nB. y\n...\nAnswer with the letter." / "Answer: B"
    lines = [l.strip() for l in u.split("\n")]
    opts, letters = [], []
    for l in lines:
        m = LETTER_RX.match(l)
        if m: letters.append(m.group(1)); opts.append(m.group(2).strip().rstrip("."))
    m = re.search(r"Answer:\s*([A-J])", a)
    if len(opts) < 2 or not m or m.group(1) not in letters: return None
    ctx_lines = [l for l in lines if not LETTER_RX.match(l) and l not in ("Choices:", "Answer with the letter.") and l]
    q = next((l[len("Question:"):].strip() for l in ctx_lines if l.startswith("Question:")), ctx_lines[-1] if ctx_lines else "Which option is correct?")
    ctx = "\n".join(l for l in ctx_lines if not l.startswith("Question:"))
    return ctx[:2000], q, opts, letters.index(m.group(1))
# --- end of the copied block ---


def load_items(path: Path, sub: str, n: int):
    """The first n rows the author's filter keeps, as (row index, PIL RGB image, context, question, options, gold)."""
    import pyarrow.parquet as pq
    from PIL import Image
    pf = pq.ParquetFile(path)
    out, rejected, idx = [], 0, -1
    for batch in pf.iter_batches(batch_size=64, columns=["images", "texts"]):
        for r in batch.to_pylist():
            idx += 1
            if len(r["images"]) != 1 or not r["texts"]:
                rejected += 1; continue
            t = r["texts"][0]
            p = parse(sub, t["user"], t["assistant"])
            if p is None:
                rejected += 1; continue
            ctx, q, opts, g = p
            im = Image.open(io.BytesIO(r["images"][0]["bytes"]))
            im.load()                                  # what datasets' Image feature does on decode
            size0 = im.size
            if max(im.size) > 768:
                im = im.copy(); im.thumbnail((768, 768))
            im = im.convert("RGB")                     # data.py stores png(im) = convert("RGB"); to_pil() reopens as RGB
            context = "This is a visual question about the image." + ("\n" + ctx if ctx else "")
            out.append(dict(index=idx, image=im, size_in=list(size0), size_thumb=list(im.size),
                            context=context, question=q, options=opts, gold=g))
            if len(out) >= n:
                return out, rejected
    return out, rejected


def fixture_line(oracle):
    """native vs g256 / g448 on the self-made fixture, from the CPU fp32 oracle (image rows only)."""
    nat = [x for x in oracle["rows"] if x["arm"] == "native"]
    fixture = {}
    for arm in ARMS:
        dps, agree, corr, toks, walls, n = [], 0, 0, [], [], 0
        for x in nat:
            y = next(z for z in oracle["rows"] if z["id"] == x["id"] and z["arm"] == arm)
            toks.append(y["n_image_tokens"]); walls.append(y["wall_s"])
            for s in range(len(x["nopts"])):
                dps.append(float(np.abs(np.array(y["probs"][s]) - np.array(x["probs"][s])).max()))
                agree += y["argmax"][s] == x["argmax"][s]; corr += y["gold_correct"][s]; n += 1
        fixture[arm] = dict(n=n, gold_accuracy=corr / n, argmax_agree_with_native=agree / n,
                            mean_max_abs_dp_vs_native=float(np.mean(dps)), max_abs_dp_vs_native=float(np.max(dps)),
                            mean_image_tokens=float(np.mean(toks)), mean_wall_s=float(np.mean(walls)),
                            device="cpu", note="n = slots over image rows; wall per row (all its slots)")
    return fixture


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hf-id", default=DEFAULT_HF_ID)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--parquet-dir", default=None,
                    help="local copies of the shards (named <subset>-train-00000.parquet); default: hf_hub_download")
    ap.add_argument("--out-dir", default=str(work_path("_decider2bv", "price")))
    ap.add_argument("--fixture-oracle", default=str(work_path("_decider2bv", "oracle", "fixture_oracle.json")))
    ap.add_argument("--fixtures", default=str(work_path("_decider2bv", "fixtures")))
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--n-visual7w", type=int, default=150)
    ap.add_argument("--n-vsr", type=int, default=100)
    ap.add_argument("--mps-check-rows", default="r01,r13,r23,t02",
                    help="fixture rows (g256) that MPS fp32 must reproduce against the CPU fp32 oracle")
    ap.add_argument("--mps-tol", type=float, default=2e-3, help="max |dp| allowed for MPS vs the CPU oracle")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--fixture-only", action="store_true",
                    help="no model, no dataset: recompute only the fixture line of an existing grid_price.json "
                         "from --fixture-oracle (after the oracle's rows changed)")
    args = ap.parse_args()

    if args.fixture_only:
        path = Path(args.out_dir).expanduser() / "grid_price.json"
        res = json.loads(path.read_text())
        oracle_path = Path(args.fixture_oracle).expanduser()
        res["fixture"] = fixture_line(json.loads(oracle_path.read_text()))
        res["fixture_oracle_sha256"] = hashlib.sha256(oracle_path.read_bytes()).hexdigest()
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(res, indent=1) + "\n")
        os.replace(tmp, path)
        for arm, v in res["fixture"].items():
            print(f"fixture   {arm:6s} n={v['n']:3d} acc={v['gold_accuracy']:.3f} agree={v['argmax_agree_with_native']:.3f} "
                  f"mean|dp|={v['mean_max_abs_dp_vs_native']:.4f} max|dp|={v['max_abs_dp_vs_native']:.4f}")
        print(f"rewrote the fixture line of {path}")
        return 0

    import torch
    from PIL import Image
    from huggingface_hub import hf_hub_download, snapshot_download

    torch.set_num_threads(args.threads)
    try:
        snapshot = Path(hf_snapshot(args.hf_id, revision=args.revision))
        if not (snapshot / "model.safetensors").exists():
            raise FileNotFoundError(snapshot)
    except FileNotFoundError:
        snapshot = Path(snapshot_download(args.hf_id, revision=args.revision,
                                          allow_patterns=["*.json", "*.jinja", "model.safetensors", "decider/*.py"]))
    sys.path.insert(0, str(snapshot))
    from decider.vision import VisionDecisionModel               # noqa: E402  (author's code, unchanged)
    from decider.infer import Example, Q                         # noqa: E402

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    m = VisionDecisionModel(str(snapshot), dtype=torch.float32, grad_ckpt=False).eval()
    torch.set_grad_enabled(False)

    def readout(img, context, question, options, gold=0):
        ex = Example(context, [Q(question, options, gold)])
        t0 = time.perf_counter()
        inp = m.prepare([(img, ex)])
        lg = m.slot_logits(inp)
        if lg.device.type == "mps":
            torch.mps.synchronize()
        wall = time.perf_counter() - t0
        p = torch.softmax(lg, -1)[0, :len(options)].float().cpu().numpy()
        grid = inp["image_grid_thw"][0].tolist() if "image_grid_thw" in inp else None
        return dict(probs=p.tolist(), argmax=int(p.argmax()), grid_thw=grid,
                    n_image_tokens=int((inp["input_ids"][0] == IMAGE_PAD).sum()), wall_s=wall)

    # Device: MPS fp32 only if it reproduces the CPU fp32 oracle on fixture rows (same pre-resized images).
    device, mps_check = "cpu", None
    oracle = json.loads(Path(args.fixture_oracle).expanduser().read_text()) if Path(args.fixture_oracle).expanduser().exists() else None
    if args.device in ("auto", "mps") and torch.backends.mps.is_available() and oracle is not None:
        fx = Path(args.fixtures).expanduser()
        rows = {r["id"]: r for r in json.loads((fx / "rows.json").read_text())["rows"]}
        meta = {im["name"]: im for im in json.loads((fx / "meta.json").read_text())["images"]}
        try:
            m.to("mps")
            checks = []
            for rid in args.mps_check_rows.split(","):
                r = rows[rid]
                arm = "text" if r["image"] is None else "g256"
                ref = next(x for x in oracle["rows"] if x["id"] == rid and x["arm"] == arm)
                img = None
                if r["image"] is not None:
                    img = Image.open(fx / meta[r["image"]]["path"]).convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
                ex = Example(r["context"], [Q(qq["text"], qq["options"], qq["gold"]) for qq in r["questions"]])
                lg = m.slot_logits(m.prepare([(img, ex)]))
                probs = torch.softmax(lg, -1).float().cpu().numpy()
                for s, n in enumerate(ref["nopts"]):
                    dp = float(np.abs(probs[s, :n] - np.array(ref["probs"][s])).max())
                    checks.append(dict(row=rid, slot=s, max_abs_dp=dp, argmax_equal=int(probs[s, :n].argmax()) == ref["argmax"][s]))
            worst = max(c["max_abs_dp"] for c in checks)
            ok = worst <= args.mps_tol and all(c["argmax_equal"] for c in checks)
            mps_check = dict(rows=args.mps_check_rows, tol=args.mps_tol, worst_max_abs_dp=worst, pass_=ok, slots=checks)
            print(f"MPS fp32 vs CPU fp32 oracle: worst max|dp| {worst:.3e} over {len(checks)} slots -> {'PASS' if ok else 'FAIL'}", flush=True)
            if ok:
                device = "mps"
            else:
                m.to("cpu")
        except Exception as e:                      # noqa: BLE001  (fall back to CPU and say why)
            mps_check = dict(error=repr(e)); m.to("cpu")
            print(f"MPS unusable: {e!r}", flush=True)
    if args.device == "mps" and device != "mps":
        raise SystemExit("MPS requested but it did not reproduce the CPU oracle")
    n_v7w, n_vsr = (args.n_visual7w, args.n_vsr) if device == "mps" else (args.n_visual7w // 2, args.n_vsr // 2)
    print(f"device {device} fp32; visual7w {n_v7w}, vsr {n_vsr}", flush=True)

    shards = {}
    for sub, rel in SHARDS.items():
        local = Path(args.parquet_dir).expanduser() / f"{sub}-train-00000.parquet" if args.parquet_dir else None
        shards[sub] = local if local is not None and local.exists() else Path(
            hf_hub_download(DATASET, rel, repo_type="dataset", revision=DATASET_REVISION))

    items_path = out_dir / "items.jsonl"
    per = {}
    with open(items_path, "w") as fo:
        for sub, n in (("visual7w", n_v7w), ("vsr", n_vsr)):
            items, rejected = load_items(shards[sub], sub, n)
            per[sub] = dict(rejected_before_n=rejected, items=[])
            for k, it in enumerate(items):
                rec = dict(dataset=sub, row_index=it["index"], size_in=it["size_in"], size_thumb=it["size_thumb"],
                           nopts=len(it["options"]), gold=it["gold"], arms={})
                for arm in ARMS:
                    img = it["image"] if arm == "native" else it["image"].resize((GRIDS[arm], GRIDS[arm]), Image.Resampling.BICUBIC)
                    res = readout(img, it["context"], it["question"], it["options"], it["gold"])
                    if arm in GRIDS:
                        g = GRIDS[arm] // 16
                        assert res["grid_thw"] == [1, g, g], (sub, it["index"], arm, res["grid_thw"])
                    res["correct"] = res["argmax"] == it["gold"]
                    rec["arms"][arm] = res
                fo.write(json.dumps(rec) + "\n"); fo.flush()
                per[sub]["items"].append(rec)
                if k % 25 == 0:
                    a = rec["arms"]
                    print(f"{sub} {k:3d} idx {it['index']:4d} {it['size_thumb']} native grid {a['native']['grid_thw']} "
                          f"p {[round(v, 3) for v in a['native']['probs']]} g256 {[round(v, 3) for v in a['g256']['probs']]} "
                          f"{a['native']['wall_s']:.2f}s", flush=True)

    def aggregate(recs):
        out = {}
        for arm in ARMS:
            dps = [float(np.abs(np.array(r["arms"][arm]["probs"]) - np.array(r["arms"]["native"]["probs"])).max()) for r in recs]
            out[arm] = dict(
                n=len(recs), gold_accuracy=float(np.mean([r["arms"][arm]["correct"] for r in recs])),
                argmax_agree_with_native=float(np.mean([r["arms"][arm]["argmax"] == r["arms"]["native"]["argmax"] for r in recs])),
                mean_max_abs_dp_vs_native=float(np.mean(dps)), max_abs_dp_vs_native=float(np.max(dps)),
                mean_image_tokens=float(np.mean([r["arms"][arm]["n_image_tokens"] for r in recs])),
                mean_wall_s=float(np.mean([r["arms"][arm]["wall_s"] for r in recs])))
        return out

    table = {sub: aggregate(per[sub]["items"]) for sub in per}

    # The same comparison on the self-made fixture (CPU fp32 oracle, image rows only).
    fixture = fixture_line(oracle) if oracle is not None else None

    import transformers
    res = dict(
        schema="coreai-decider-vision-grid-price/1",
        source=dict(hf_id=args.hf_id, revision=args.revision),
        dataset=dict(repo=DATASET, revision=DATASET_REVISION, shards=SHARDS,
                     license=("dataset card has no licence field; 'Licensing Information': each sub-dataset is governed "
                              "by its own licence, the prompts (to the extent of HuggingFaceM4's rights) are CC-BY-4.0"),
                     held_out=("visual7w and vsr are held_out=True in decider/vision/data.py SUBSETS"),
                     selection="first rows in file order that pass the author's filter (1 image, parse() not None)",
                     preprocessing="thumbnail((768, 768)) when max side > 768, then RGB; context as in data.py",
                     rejected_before_n={sub: per[sub]["rejected_before_n"] for sub in per}),
        device=device, dtype="float32", torch=torch.__version__, transformers=transformers.__version__,
        mps_check=mps_check, wall_note="wall seconds are contended reference values (other sessions share the Mac)",
        table=table, fixture=fixture,
        fixture_oracle_sha256=(hashlib.sha256(Path(args.fixture_oracle).expanduser().read_bytes()).hexdigest()
                               if oracle is not None else None),
    )
    tmp = out_dir / "grid_price.json.tmp"
    tmp.write_text(json.dumps(res, indent=1) + "\n")
    os.replace(tmp, out_dir / "grid_price.json")
    for sub, rows_ in list(table.items()) + ([("fixture", fixture)] if fixture else []):
        for arm, v in rows_.items():
            print(f"{sub:9s} {arm:6s} n={v['n']:3d} acc={v['gold_accuracy']:.3f} agree={v['argmax_agree_with_native']:.3f} "
                  f"mean|dp|={v['mean_max_abs_dp_vs_native']:.4f} max|dp|={v['max_abs_dp_vs_native']:.4f} "
                  f"tok={v['mean_image_tokens']:.1f} wall={v['mean_wall_s']:.2f}s")
    print(f"wrote {out_dir / 'grid_price.json'} and {items_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
