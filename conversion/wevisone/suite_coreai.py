#!/usr/bin/env python3
"""Core AI answers for the structured-output suite: vision `.aimodelc` -> decoder `.aimodelc`, free-run greedy.

Per fixture row: the oracle's own pixel_values for the same fixed grid (O2-448 for grid 14, O2-512 for grid 16 —
PIL BICUBIC S x S, block-major patches, identical to a NumPy patchify within 6e-8) go through the Core AI vision
tower (fp16 in, fp16 out); its image_embeds and deepstack rows feed the decoder's static inputs; the prompt ids are
the oracle's with the exporter's host rule; the decoder's S=1 "main" function runs the prompt one token per call
and then free-runs greedy (a stop id ends it, at most `--max-new` tokens, 768 = the oracle's cap). No repetition
penalty (the Core AI engine and kit have none). Answers are decoded with the bundle's tokenizer and cut like the
package does (`<|im_end|>` / `<|endoftext|>`, strip). The vision outputs are compared with the oracle's too.

    QH <coreai venv>/bin/python suite_coreai.py <decoder bundle> --aot <decoder .aimodelc> --vision-aot <tower .aimodelc> \
        --oracle <dir> --fixtures <json> --grid 14 --out <json> [--rows a,b] [--max-new 768]
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_common as G  # noqa: E402

FORM_FOR_GRID = {14: "O2-448", 16: "O2-512"}


async def main_async(args) -> int:
    import coreai.runtime as rt
    import tokenizers

    g = args.grid
    form = FORM_FOR_GRID[g]
    bundle = Path(args.bundle).expanduser().resolve()
    meta = json.loads((bundle / "metadata.json").read_text())
    odir = Path(args.oracle) / form
    rows = [r["id"] for r in json.loads(Path(args.fixtures).read_text())["rows"]]
    if args.rows:
        want = args.rows.split(",")
        rows = [r for r in rows if r in want]
    tok = tokenizers.Tokenizer.from_file(str(bundle / "tokenizer" / "tokenizer.json"))
    win = G.window_record(args.label)
    gs = G.GpuSampler().__enter__()
    t_all = time.perf_counter()
    vmodel = await G.maybe(rt.AIModel.load(str(Path(args.vision_aot).resolve()), rt.SpecializationOptions.default()))
    vfn = await G.maybe(vmodel.load_function("main"))
    vdesc = G.fn_desc(vfn)
    dec = await G.Decoder.load(rt, Path(args.aot).resolve(), int(meta["language"]["max_context_length"]), g * g,
                               args.reload_every)
    dec.check_contract()
    out = {"schema": "wevisone-suite-coreai/1", "bundle": str(bundle), "name": meta["name"], "grid": g, "form": form,
           "decoder_aimodelc": G.tree_digest(Path(args.aot).resolve()),
           "vision_aimodelc": G.tree_digest(Path(args.vision_aot).resolve()), "vision_descriptor": vdesc,
           "environment": G.env_record(), "window": win, "max_new_tokens": args.max_new,
           "repetition_penalty": 1.0, "rows": {}}
    for rid in rows:
        npz = odir / f"{rid}.npz"
        if not npz.exists():
            out["rows"][rid] = {"skipped": "no oracle npz"}
            continue
        z = np.load(npz)
        t0 = time.perf_counter()
        pv = z["pixel_values"].astype(np.float16)
        vres = await G.maybe(vfn(inputs={"patches": G.nd(rt, pv)}))
        img = np.asarray(vres["image_embeds"].numpy())
        ds = np.asarray(vres["deepstack_embeds"].numpy()).reshape(3, g * g, -1)
        vis_s = time.perf_counter() - t0
        vrec = {"image_embeds_cos": G.cos(img, z["image_embeds"]),
                "deepstack_cos": [G.cos(ds[k], z["deepstack_embeds"][k]) for k in range(3)],
                "max_abs_d": float(np.abs(img.astype(np.float32) - z["image_embeds"]).max()), "seconds": vis_s}
        mapped, start, amount = G.host_inputs(z["input_ids"].tolist(), g)
        static = dec.static(img, ds, start, amount)
        t1 = time.perf_counter()
        r = await dec.run(mapped, static, free=args.max_new)
        gen = r["gen"]
        raw = tok.decode(gen, skip_special_tokens=False)
        response = raw.split("<|im_end|>", 1)[0].split("<|endoftext|>", 1)[0].strip()
        hit_max = len(gen) >= args.max_new and gen[-1] not in G.STOP_IDS
        oracle = json.loads((odir / f"{rid}.json").read_text())
        out["rows"][rid] = {"task": oracle["task"], "gen_ids": gen, "gen_len": len(gen), "hit_max": hit_max,
                            "raw_text": raw, "response": response, "vision": vrec,
                            "equal_to_oracle_ids": gen == oracle["gen_ids"],
                            "first_difference": next((i for i in range(min(len(gen), len(oracle["gen_ids"])))
                                                      if gen[i] != oracle["gen_ids"][i]), None),
                            "prompt_ms_median": float(np.median(r["step_ms"])),
                            "decode_ms_median": float(np.median(r["gen_ms"])) if r["gen_ms"] else None,
                            "seconds": time.perf_counter() - t1}
        if args.dump_refs and rid in args.dump_ids.split(","):
            # the Swift engine check (Qwen3VLBench --refs): same static inputs, the python S=1 answer as expected
            d = Path(args.dump_refs) / rid
            d.mkdir(parents=True, exist_ok=True)
            np.ascontiguousarray(img.astype(np.float16)).tofile(d / "image_embeds.f16.bin")
            np.ascontiguousarray(ds.reshape(3 * g * g, -1).astype(np.float16)).tofile(d / "deepstack_embeds.f16.bin")
            (d / "vl_ref.json").write_text(json.dumps({
                "prompt_ids": [int(t) for t in mapped], "expected_ids": [int(t) for t in gen],
                "shift_start": int(start), "shift_amount": int(amount), "n_img": g * g, "grid": g,
                "source": f"suite_coreai.py python S=1 free-run on {Path(args.aot).name}"}) + "\n")
        print(f"[suite g{g}] {rid}: gen {len(gen)}{' MAX' if hit_max else ''} ids==oracle "
              f"{out['rows'][rid]['equal_to_oracle_ids']} (first diff {out['rows'][rid]['first_difference']}) "
              f"vision cos {vrec['image_embeds_cos']:.5f} | {response[:90]!r}", flush=True)
    out["wall_seconds"] = time.perf_counter() - t_all
    gs.__exit__()
    out["gpu_utilization"] = gs.summary()
    out["decoder_reloads"] = dec.reloads
    out["mpsgraph_scratch_cleanup"] = G.clean_own_mpsgraph_scratch()
    out["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["script"] = {"path": "conversion/wevisone/suite_coreai.py", "sha256": G.sha256_file(Path(__file__))}
    G.write_json(Path(args.out), out)
    print(f"[suite g{g}] {len(out['rows'])} rows, wall {out['wall_seconds']:.0f}s -> {args.out}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle")
    ap.add_argument("--aot", required=True)
    ap.add_argument("--vision-aot", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--grid", type=int, required=True, choices=[14, 16])
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="")
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--reload-every", type=int, default=0)
    ap.add_argument("--label", default="wv-r1-coreai-suite")
    ap.add_argument("--dump-refs", default="", help="write Qwen3VLBench reference dirs for --dump-ids here")
    ap.add_argument("--dump-ids", default="")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
