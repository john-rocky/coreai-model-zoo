#!/usr/bin/env python3
"""G4: image -> tower -> decoder -> answer, both MiniCPM-V-4.7 assets on the Mac GPU (AOT h16c, python runtime).

One process: the tower form's `.aimodelc` and the decoder's `.aimodelc` (`SpecializationOptions.default()`).
Per image prompt P1..P4: the fixture -> `host.preprocess` -> tower -> `image_features` (cast to fp16) ->
the decoder's `image_embeds`; ids / start / amount = `host_static_inputs(<P>__input_ids)` (the processor's
ids; the bundle tokenizer's own encoding of the expanded prompt is checked against them and recorded);
the prompt one token per call from fresh states, then free-run greedy <= 32 (stop ids end it), judged against
the oracle's `<P>__gen_ids` by coreai_gate.py's rule (first divergence passes only at an oracle softmax top-2
gap < 0.1, `margins.npz`). Recorded too: the tower rows against the oracle's (cos mean / min, max |d|), the last
prompt position's logits against `<P>__logits`, both answers as text. The process ends by re-running P1
end to end: tower output, answer and last-prompt logits must be bit-equal.

Bar (the supervisor's): P1..P4 4/4 with the ship pair.

    QW coreai-models/.venv/bin/python conversion/minicpmv47/gate_e2e.py <exports>/minicpmv47_vision_<mode> \
        <exports>/minicpmv47_vlm_decode_int8hu --out <lane>/r3/gate_e2e_<mode>.json
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
import host as H  # noqa: E402


def rows_cos(got: np.ndarray, want: np.ndarray) -> dict:
    g, w = got.astype(np.float64), want.astype(np.float64)
    r = (g * w).sum(-1) / (np.linalg.norm(g, axis=-1) * np.linalg.norm(w, axis=-1))
    return {"cos_mean": float(r.mean()), "cos_min": float(r.min()), "max_abs_d": float(np.abs(g - w).max())}


async def main_async(args) -> int:
    import coreai.runtime as rt
    import tokenizers
    from minicpmv47_vl_pipelined import host_static_inputs

    tower_b = Path(args.tower).expanduser().resolve()
    dec_b = Path(args.decoder).expanduser().resolve()
    tmeta = json.loads((tower_b / "metadata.json").read_text())
    dmeta = json.loads((dec_b / "metadata.json").read_text())
    t_aot, d_aot = G.aimodelc_for(tower_b), G.aimodelc_for(dec_b)
    z = np.load(G.ORACLE / "oracle.npz")
    mz = np.load(G.ORACLE / "margins.npz")
    ojs = json.loads((G.ORACLE / "oracle.json").read_text())
    tok = tokenizers.Tokenizer.from_file(str(dec_b / "tokenizer" / "tokenizer.json"))
    fixtures = G.ORACLE / "fixtures"
    gs = G.GpuSampler().__enter__()
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    tm = await G.maybe(rt.AIModel.load(str(t_aot), rt.SpecializationOptions.default()))
    names = list(getattr(tm, "function_names", []) or [])
    tfn = await G.maybe(tm.load_function(names[0] if names else "main"))
    tower_load = time.perf_counter() - t0
    tdesc = G.fn_desc(tfn)
    t_in = np.dtype(tdesc["inputs"]["pixel_values"][1])
    dec = await G.Decoder.load(rt, d_aot, int(dmeta["language"]["max_context_length"]))
    dec.check_contract()

    async def encode(px):
        t = time.perf_counter()
        res = await G.maybe(tfn(inputs={"pixel_values": G.nd(rt, px.astype(t_in))}))
        return np.asarray(res["image_features"].numpy()).astype(np.float32), (time.perf_counter() - t) * 1e3

    async def one(P: str) -> dict:
        ids_o = z[f"{P}__input_ids"].astype(np.int64)
        expanded = ojs["prompts"][P]["expanded_prompt"].replace("<|image_pad|>x64", "<|image_pad|>" * 64)
        enc = tok.encode(expanded, add_special_tokens=False).ids
        px = H.preprocess(fixtures / f"{G.PROMPT_IMAGES[P]}.448.png")
        feats, tower_ms = await encode(px)
        mapped, start, amount = host_static_inputs(ids_o)
        mapped = mapped.numpy().astype(np.int64)
        S = len(mapped)
        r = await dec.run(mapped, dec.static(feats.astype(np.float16), int(start[0]), int(amount[0])),
                          keep={S - 1}, free=G.N_FREE)
        og = z[f"{P}__gen_ids"].astype(np.int64).tolist()
        div = G.first_divergence(r["gen"], og, mz[f"{P}__margins"])
        last = r["kept"][S - 1]
        ref = z[f"{P}__logits"]
        a, b = last.astype(np.float64), ref.astype(np.float64)
        rec = {"image": f"{G.PROMPT_IMAGES[P]}.448.png", "prompt_tokens": S,
               "tokenizer_ids_equal_processor_ids": enc == ids_o.tolist(),
               "rope_shift_start": int(start[0]), "rope_shift_amount": int(amount[0]),
               "tower_vs_oracle": rows_cos(feats, z[f"{P}__image_features"]), "tower_ms": tower_ms,
               **div, "engine_gen": r["gen"], "oracle_gen": og,
               "engine_text": tok.decode(r["gen"], skip_special_tokens=True),
               "oracle_text": tok.decode(og, skip_special_tokens=True),
               "last_position_vs_oracle": {"top5_engine": G.topk(last, 5)[0], "top5_oracle": G.topk(ref, 5)[0],
                                           "cos": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
                                           "max_abs_d": float(np.abs(a - b).max())},
               "prompt_step_ms_median": float(np.median(r["step_ms"])),
               "decode_ms_median": float(np.median(r["gen_ms"])) if r["gen_ms"] else None}
        return rec, (feats, r["gen"], last)

    prompts, first = {}, None
    for P in ("P1", "P2", "P3", "P4"):
        rec, arrays = await one(P)
        prompts[P] = rec
        if P == "P1":
            first = arrays
        print(f"[G4 {P}] {rec['verdict']} first divergence {rec['first_divergence']} (margin "
              f"{rec.get('oracle_margin_at_divergence')}) tower cos min {rec['tower_vs_oracle']['cos_min']:.6f} "
              f"ids==processor {rec['tokenizer_ids_equal_processor_ids']} | {rec['engine_text']!r}", flush=True)
    _, again = await one("P1")
    reset = {"tower_bit_equal": bool(np.array_equal(first[0], again[0])), "answer_equal": first[1] == again[1],
             "last_logits_bit_equal": bool(np.array_equal(first[2], again[2]))}
    reset["bit_equal"] = all(reset.values())
    n_pass = sum(v["verdict"] != "FAIL" for v in prompts.values())
    out = {"schema": "minicpmv47-gate-e2e/1",
           "pair": {"tower": tmeta["name"], "decoder": dmeta["name"]},
           "tower": {"bundle": str(tower_b), "aimodelc": G.tree_digest(t_aot), "descriptor": tdesc,
                     "input_fed_as": str(t_in), "load_seconds": tower_load},
           "decoder": {"bundle": str(dec_b), "aimodelc": G.tree_digest(d_aot), "load_seconds": dec.load_seconds},
           "oracle_npz_sha256": G.sha256_file(G.ORACLE / "oracle.npz"),
           "margins_npz_sha256": G.sha256_file(G.ORACLE / "margins.npz"),
           "environment": G.env_record(),
           "bar": "P1..P4 4/4: free-run greedy <= 32 with no divergence from the oracle, or the first one at an "
                  f"oracle softmax top-2 gap < {G.KNIFE_EDGE}",
           "prompts": prompts, "pass": f"{n_pass}/4", "reset_check": reset,
           "result": "PASS" if n_pass == 4 and reset["bit_equal"] else "FAIL",
           "process_prompts": 5, "wall_seconds": time.perf_counter() - t_all,
           "gpu_utilization": (gs.__exit__(), gs.summary())[1],
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "script": {"path": "conversion/minicpmv47/gate_e2e.py", "sha256": G.sha256_file(Path(__file__))}}
    G.write_json(Path(args.out), out)
    print(f"[G4] {tmeta['name']} + {dmeta['name']}: {out['result']} ({out['pass']}), reset {reset} -> {args.out}", flush=True)
    return 0 if out["result"] == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tower")
    ap.add_argument("decoder")
    ap.add_argument("--out", required=True)
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
