#!/usr/bin/env python3
"""Token-exactness of the EXPORTED OvisOCR2 page-parsing chain, end to end.

Runs the ENTIRE shipped path on the python runtime, per the 27B gate's shape:

    uint8 page -> NumPy preprocess (qwen38vl_preprocess, 1280x896 tile)
    -> vision .aimodel -> embed splice + host mRoPE planes (qwen38vl_host)
    -> embeddings decoder .aimodel ("prefill" S=16 chunks + "main" S=1)
    -> greedy tokens, compared against the fp32 HF oracle's

This is the gate that decides shipping, and the one that settles whether the fp16
tower's 4-row cosine tail (test_ovisocr2_tower_gate.py) changes any output token.
Cosine is deliberately NOT the criterion here — it is a single-position summary
that hides argmax flips. Greedy makes every token after a first divergence differ
by construction, so --show-text prints both continuations to judge whether a
divergence matters.

Unlike the 27B gate, the hybrid state shapes are READ FROM THE BUNDLE
(`desc.state_descriptor`) instead of hardcoded, so this runs for any Qwen3.5-family
export without editing constants.

Oracle: _smoke/dump_ovisocr2_oracle.py (system python — the overlay venv's
transformers has no `qwen3_5`).

Run (coreai-models/.venv, GPU, _GPU_LOCK held, python-GPU SOLO):
    ../coreai-models/.venv/bin/python _smoke/test_ovisocr2_suite_gate.py
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import sys
import time
from pathlib import Path

import numpy as np

import coreai.runtime as rt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "conversion"))
from _paths import exports_dir  # noqa: E402

from qwen38vl_host import mrope_positions, splice_embeds  # noqa: E402

DEFAULT_ORACLE = Path(__file__).parent / "ovisocr2_oracle.npz"
KV_SEQ = 2048  # TRACE_KV_CACHE_SEQ_LEN; the bundle leaves this axis dynamic


async def maybe(x):
    return await x if inspect.isawaitable(x) else x


def state_shapes(desc) -> dict[str, tuple[int, ...]]:
    """State buffer shapes straight from the bundle, dynamic axis -> KV_SEQ."""
    out = {}
    for name in desc.state_names:
        shape = [KV_SEQ if d < 0 else d for d in desc.state_descriptor(name).shape]
        out[name] = tuple(shape)
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default=str(DEFAULT_ORACLE))
    ap.add_argument("--vision", default=None)
    ap.add_argument("--decoder", default=None)
    ap.add_argument("--vision-asset", default=None,
                    help="explicit vision asset (e.g. an AOT .aimodelc)")
    ap.add_argument("--decoder-asset", default=None,
                    help="explicit decoder asset (e.g. an AOT .aimodelc); the bundle dir is "
                         "still where embed_tokens/tokenizer are read from")
    ap.add_argument("--show-text", action="store_true")
    ap.add_argument("--gpu-pref", action="store_true",
                    help="load the DECODER with preferred=GPU (ANERegionFormationPass "
                         "asserts on the multifunction prefill graph under default opts)")
    args = ap.parse_args()

    oracle = np.load(args.oracle)
    ids = oracle["input_ids"].astype(np.int64)
    want = oracle["gen_ids"].astype(np.int64)
    patches = oracle["patches"].astype(np.float16)
    grid = tuple(int(v) for v in oracle["grid_thw"])
    print(f"oracle {args.oracle}: prompt {ids.size} tok, {want.size} generated, "
          f"grid {grid}, dtype {oracle['dtype']}")

    root = exports_dir()
    vis_dir = Path(args.vision) if args.vision else root / "ovisocr2_vision_fp16"
    dec_dir = (Path(args.decoder) if args.decoder
               else root / "ovisocr2_vl_decode_int8hu_block32_sym_pf16")
    opts = rt.SpecializationOptions.default()
    vis_asset = Path(args.vision_asset) if args.vision_asset else vis_dir / f"{vis_dir.name}.aimodel"
    vm = await maybe(rt.AIModel.load(str(vis_asset), opts))
    vfn = await maybe(vm.load_function(vm.function_names[0]))
    dec_opts = (rt.SpecializationOptions.from_preferred_compute_unit_kind(
        rt.ComputeUnitKind.gpu()) if args.gpu_pref else opts)
    dec_asset = (Path(args.decoder_asset) if args.decoder_asset
                 else dec_dir / f"{dec_dir.name}.aimodel")
    dm = await maybe(rt.AIModel.load(str(dec_asset), dec_opts))
    dfn = await maybe(dm.load_function("main"))
    # A decode-only bundle has no "prefill": the prompt is ingested at S=1. That is the
    # build that needs no --expect-frequent-reshapes, and so no 4.7x AOT inflation.
    has_pf = "prefill" in list(dm.function_names)
    pfn = await maybe(dm.load_function("prefill")) if has_pf else None
    PF = int(pfn.desc.input_descriptor("inputs_embeds").shape[1]) if has_pf else 1
    shapes = state_shapes(dfn.desc)
    print(f"vision {vis_asset.name} | decoder {dec_asset.name} | "
          f"{'PF=' + str(PF) if has_pf else 'decode-only (S=1 prompt ingest)'}")
    for k, v in shapes.items():
        print(f"    state {k:12s} {v}")

    from safetensors.numpy import load_file
    embed_table = load_file(
        str(dec_dir / "embed_tokens.safetensors"))["embed_tokens.weight"]
    print(f"embed table {embed_table.shape} {embed_table.dtype}")

    decode_txt = None
    if args.show_text:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(dec_dir / "tokenizer" / "tokenizer.json"))
        decode_txt = lambda x: tok.decode([int(i) for i in x], skip_special_tokens=True)  # noqa: E731

    t0 = time.perf_counter()
    out = await maybe(vfn(inputs={"patches": rt.NDArray(np.ascontiguousarray(patches))}))
    embeds_img = np.asarray(out["image_embeds"].numpy()).astype(np.float16)
    t_tower = time.perf_counter() - t0
    print(f"tower {embeds_img.shape} in {1000 * t_tower:.0f} ms")

    pos, delta = mrope_positions(ids, [grid])
    embeds = splice_embeds(ids, embed_table, embeds_img)
    S = ids.size
    state = {k: rt.NDArray(np.zeros(v, np.float16)) for k, v in shapes.items()}

    async def call(fn, x, ramp_len, p3):
        res = await maybe(fn(inputs={
            "inputs_embeds": rt.NDArray(np.ascontiguousarray(x[None])),
            "position_ids": rt.NDArray(np.arange(ramp_len, dtype=np.int32)[None]),
            "pos_t": rt.NDArray(np.ascontiguousarray(p3[0:1].astype(np.int32))),
            "pos_h": rt.NDArray(np.ascontiguousarray(p3[1:2].astype(np.int32))),
            "pos_w": rt.NDArray(np.ascontiguousarray(p3[2:3].astype(np.int32))),
        }, state=state))
        return np.asarray(res["logits"].numpy())[0, -1]

    t0 = time.perf_counter()
    row, o = None, 0
    while has_pf and o + PF <= S:
        row = await call(pfn, embeds[o:o + PF], o + PF, pos[:, o:o + PF])
        o += PF
    while o < S:
        row = await call(dfn, embeds[o:o + 1], o + 1, pos[:, o:o + 1])
        o += 1
    t_pf = time.perf_counter() - t0

    got = [int(row.argmax())]
    t0 = time.perf_counter()
    for k in range(1, want.size):
        p3 = np.full((3, 1), S + k - 1 + delta, dtype=np.int32)
        row = await call(dfn, embed_table[got[-1]][None].copy(), S + k, p3)
        got.append(int(row.argmax()))
    t_dec = time.perf_counter() - t0

    got_arr = np.array(got, dtype=np.int64)
    match = int((got_arr == want).sum())
    first_bad = None if match == want.size else int(np.argmax(got_arr != want))
    ok = match == want.size
    print(f"\n{'PASS' if ok else f'FAIL @{first_bad}'}: "
          f"{match}/{want.size} tokens ({100 * match / want.size:.1f}%)")
    if decode_txt is not None and first_bad is not None:
        print(f"  oracle: {decode_txt(want)!r}")
        print(f"  bundle: {decode_txt(got_arr)!r}")
    print(f"measured: tower {1000 * t_tower:.0f} ms | prefill {S / t_pf:.1f} tok/s "
          f"| decode {(want.size - 1) / t_dec:.1f} tok/s")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
