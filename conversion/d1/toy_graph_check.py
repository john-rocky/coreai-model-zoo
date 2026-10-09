#!/usr/bin/env python3
"""Toy graph check: the d1 decoder class at a tiny config exports to Core AI and runs on the Python runtime (CPU).

`lfm2_d1_decoder.Lfm2D1Decoder` (the overlay's LFM2.5-VL text decoder with an Identity head) is built at a toy
config — hidden 64, 4 query / 2 KV heads of 16, MLP 128, vocabulary 256, layers [conv, full_attention, conv], conv
kernel 3, RoPE theta 1e6 — with seeded random weights (the RMSNorm gains drawn around 1 so a wrong norm shows), and
taken through the round-2 path with no checkpoint:

    build_export_spec(dtype, 4096, TRACE_KV_CACHE_SEQ_LEN, query_len=16)  ->  export_to_coreai (gated_delta_update
    left out of the externalized composites, as export_lfm2_decode_pipelined.py does; the static-S SDPA guard retried
    with torch's bounds, as conversion/kev/export_decoder.py does)  ->  prog.optimize()  ->  save_asset
    ->  coreai.runtime AIModel.load(<asset>, SpecializationOptions.cpu_only())  ->  load_function("main")

The CPU-only runtime loads that graph and refuses its first call (`inferenceFailed(-1)`): the graph's dynamic
dimensions (position_ids [1, seq], the KV sequence axis) are what it refuses — the same module exported with every
shape static runs, with or without the externalized composites (round 1, `results/toy_graph_variants.json`). The
dynamic graph is the one round 2 compiles AOT for the GPU; this check records the refusal (`shipped_spec`) and runs the
same module on the CPU as a chain of static graphs instead (`static_chain`): one graph per call position (position_ids
[1, 16], [1, 32], [1, 48]; KV axis 2048), the three states carried between them as the same NDArrays.

A row of 37 random ids (no extension id, image_embeds zero) runs as 3 calls of S = 16 from zero states, the last
padded with id 255 and its padded rows dropped; the 37 hidden rows are compared with the same module's eager fp32
forward over the whole row in one call from zero states (`forward` -> `forward_stateful_embeds`): the lowest
position cosine (float64) and max |d|. A chunked call order and a one-call forward are the same mathematics, so the
cosine should be at least 0.9999. The row runs twice from fresh states (bit-equal hidden rows = the state reset).

Extension ids: the same row with 4 ids replaced by V + slot and image_embeds[slot] = the embedding row of the id it
replaced (in the graph's dtype) must give the plain row's hidden rows bit for bit (the graph reads image_embeds for
an extension id), and the same extension row with image_embeds zero must not (red arm).

Two dtypes: fp16 (the ship dtype: the four attention projections kept fp32, as the overlay's loader does) and fp32.

    <coreai-models venv>/bin/python toy_graph_check.py --eager-only          # torch only, no export
    ~/code/standup/tools/quiet/quiet_wait.py --max-wait 3600 -- <coreai-models venv>/bin/python toy_graph_check.py

-> $ZOO_WORK_ROOT/_d1_3b/results/toy_graph.json (--eager-only: toy_graph_eager.json); the assets under
$ZOO_WORK_ROOT/_d1_3b/exports/toy/<run>/ (a new directory per run; delete after reading).
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.metadata as md
import inspect
import json
import platform
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _paths import work_path  # noqa: E402
from lfm2_d1_decoder import Lfm2D1Decoder  # noqa: E402

from coreai_models.models.macos.lfm2 import Lfm2Config, build_decode_state  # noqa: E402

LANE = work_path("_d1_3b")
S, T, MAX_CTX, N_IMG, SEED = 16, 37, 4096, 8, 0
PAD = 255
EXT_POSITIONS, EXT_SLOTS = (3, 10, 20, 33), (1, 3, 5, 7)
COS_BAR = 0.9999


def toy_config() -> Lfm2Config:
    return Lfm2Config(hidden_size=64, num_hidden_layers=3, vocab_size=256, intermediate_size=128,
                      block_auto_adjust_ff_dim=False, norm_eps=1e-5, tie_embedding=True, num_attention_heads=4,
                      num_key_value_heads=2, conv_L_cache=3, conv_bias=False, rope_theta=1e6,
                      max_position_embeddings=MAX_CTX, layer_types=["conv", "full_attention", "conv"])


def toy_model() -> Lfm2D1Decoder:
    torch.manual_seed(SEED)
    model = Lfm2D1Decoder(toy_config(), n_image_tokens=N_IMG).float().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
    return model


def row_ids() -> np.ndarray:
    g = torch.Generator().manual_seed(SEED + 1)
    return torch.randint(1, PAD, (T,), generator=g).numpy().astype(np.int32)


def ext_row(ids: np.ndarray, emb: torch.Tensor, dtype: torch.dtype) -> tuple[np.ndarray, torch.Tensor]:
    """ids with EXT_POSITIONS replaced by V + slot, and image_embeds[slot] = the replaced id's embedding row."""
    V = emb.shape[0]
    ext = ids.copy()
    img = torch.zeros(N_IMG, emb.shape[1], dtype=dtype)
    for pos, slot in zip(EXT_POSITIONS, EXT_SLOTS):
        img[slot] = emb[int(ids[pos])].to(dtype)
        ext[pos] = V + slot
    return ext, img


def pos_cos(ref: np.ndarray, got: np.ndarray) -> np.ndarray:
    a, b = ref.astype(np.float64), got.astype(np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def compare(ref: np.ndarray, got: np.ndarray) -> dict:
    c = pos_cos(ref, got)
    d = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    return {"cos_min": float(c.min()), "cos_min_position": int(c.argmin()), "max_abs_diff": float(d.max()),
            "mean_abs_diff": float(d.mean()), "finite": bool(np.isfinite(got).all())}


# --------------------------------------------------------------------------- eager
def eager_states(model: Lfm2D1Decoder, dtype: torch.dtype) -> dict:
    return build_decode_state(model.config, max_seq_len=MAX_CTX, dtype=dtype)


@torch.no_grad()
def eager_oneshot(model: Lfm2D1Decoder, ids: np.ndarray, img: torch.Tensor) -> np.ndarray:
    st = eager_states(model, torch.float32)
    h = model(torch.from_numpy(ids)[None], torch.arange(len(ids), dtype=torch.int32)[None], img,
              st["k_cache"], st["v_cache"], st["conv_state"])
    return h[0].float().numpy()


@torch.no_grad()
def eager_chunked(model: Lfm2D1Decoder, ids: np.ndarray, img: torch.Tensor) -> np.ndarray:
    st = eager_states(model, torch.float32)
    out = []
    for c in range(-(-len(ids) // S)):
        x = np.full(S, PAD, np.int32)
        piece = ids[c * S:(c + 1) * S]
        x[:len(piece)] = piece
        h = model(torch.from_numpy(x)[None], torch.arange(c * S + S, dtype=torch.int32)[None], img,
                  st["k_cache"], st["v_cache"], st["conv_state"])
        out.append(h[0, :len(piece)].float().numpy())
    return np.concatenate(out)


def eager_checks(model: Lfm2D1Decoder, ids: np.ndarray) -> tuple[dict, np.ndarray]:
    zero = torch.zeros(N_IMG, model.config.hidden_size)
    ref = eager_oneshot(model, ids, zero)
    chunked = eager_chunked(model, ids, zero)
    ext, img = ext_row(ids, model.model.embed_tokens.weight.detach(), torch.float32)
    ext_h = eager_oneshot(model, ext, img)
    ext_zero = eager_oneshot(model, ext, zero)
    return {"row_ids": ids.tolist(), "oneshot_vs_chunked_s16": compare(ref, chunked),
            "ext_ids": ext.tolist(), "ext_rows_bit_equal": bool(np.array_equal(ext_h, ref)),
            "ext_zero_embeds_differs": compare(ref, ext_zero), "reference_hidden_absmax": float(np.abs(ref).max())}, ref


# --------------------------------------------------------------------------- export + runtime
def count_ops(prog) -> dict:
    """Op count of the optimized `main` graph, read from its printed form (one `<dialect>.<op>` per op line)."""
    out: dict = {}
    try:
        g = prog.get_graph("main")
        out["graph_type"] = type(g).__name__
        text = str(g)
        names = re.findall(r"^\s*(?:%[^=]+=\s*)?\"?([a-z_][a-z0-9_]*\.[a-z0-9_.]+)\"?[\s({<]", text, flags=re.M)
        hist: dict[str, int] = {}
        for n in names:
            hist[n] = hist.get(n, 0) + 1
        out.update({"ops": len(names), "by_op": dict(sorted(hist.items(), key=lambda kv: -kv[1])),
                    "printed_chars": len(text), "method": "regex over str(prog.get_graph('main'))"})
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:400]
    return out


def tree_bytes(path: Path) -> dict:
    files = sorted(p for p in path.rglob("*") if p.is_file())
    return {str(p.relative_to(path)): p.stat().st_size for p in files}


async def maybe(x):
    return await x if inspect.isawaitable(x) else x


def dsc(d) -> list:
    return [[int(x) for x in d.shape], str(d.dtype).split(".")[-1]]


def fn_desc(fn) -> dict:
    d = fn.desc
    return {"function": getattr(d, "name", None),
            "inputs": {n: dsc(d.input_descriptor(n)) for n in d.input_names},
            "outputs": {n: dsc(d.output_descriptor(n)) for n in d.output_names},
            "states": {n: dsc(d.state_descriptor(n)) for n in d.state_names}}


def typed_module(model: Lfm2D1Decoder, dtype: torch.dtype) -> Lfm2D1Decoder:
    """The module in `dtype`; at fp16 the four attention projections keep their fp32 weights (the overlay loader's
    fp32_attn_proj)."""
    m = copy.deepcopy(model).to(dtype).eval()
    if dtype == torch.float16:
        for src, layer in zip(model.model.layers, m.model.layers):
            if layer.is_full:
                for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    w = getattr(src.self_attn, proj).weight.detach().clone().float()
                    getattr(layer.self_attn, proj).weight = torch.nn.Parameter(w, requires_grad=False)
    return m


def export_one(m: Lfm2D1Decoder, dtype: torch.dtype, aimodel: Path, static_pos: int | None = None) -> dict:
    """Export `m` (the shipped spec, or with `static_pos` every shape static at position_ids [1, static_pos] and the
    KV axis TRACE_KV_CACHE_SEQ_LEN) and save it at `aimodel`."""
    from export_qwen38vl_pipelined import _install_externalize_dim_retry

    from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
    from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai

    spec = m.build_export_spec(dtype, MAX_CTX, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, query_len=S)
    if static_pos is not None:
        st = build_decode_state(m.config, max_seq_len=TRACE_KV_CACHE_SEQ_LEN, dtype=dtype)
        spec["reference_inputs"].update({"position_ids": torch.arange(static_pos, dtype=torch.int32)[None],
                                         "k_cache": st["k_cache"], "v_cache": st["v_cache"],
                                         "conv_state": st["conv_state"]})
        spec["dynamic_shapes"] = {k: None for k in spec["dynamic_shapes"]}
    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    _install_externalize_dim_retry()
    t0 = time.monotonic()
    prog = export_to_coreai(m, spec["reference_inputs"], dynamic_shapes=spec["dynamic_shapes"],
                            input_names=spec["input_names"], output_names=spec["output_names"],
                            state_names=spec["state_names"], externalize_modules=specs)
    t1 = time.monotonic()
    prog.optimize()
    t2 = time.monotonic()
    ops = count_ops(prog)
    import coreai.runtime as rt

    aimodel.parent.mkdir(parents=True, exist_ok=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    t3 = time.monotonic()
    mlirb = aimodel / "main.mlirb"
    return {"aimodel": str(aimodel), "static_pos": static_pos,
            "seconds": {"export": round(t1 - t0, 2), "optimize": round(t2 - t1, 2), "save": round(t3 - t2, 2)},
            "ops": ops, "aimodel_files": tree_bytes(aimodel),
            "main_mlirb_sha256": hashlib.sha256(mlirb.read_bytes()).hexdigest() if mlirb.exists() else None,
            "spec": {"input_names": list(spec["input_names"]), "output_names": list(spec["output_names"]),
                     "state_names": list(spec["state_names"]),
                     "reference_inputs": {k: [list(v.shape), str(v.dtype).replace("torch.", "")]
                                          for k, v in spec["reference_inputs"].items()},
                     "dynamic": {k: (None if v is None else sorted(str(a) for a in v))
                                 for k, v in spec["dynamic_shapes"].items()}}}


async def load(aimodel: Path):
    import coreai.runtime as rt

    t0 = time.monotonic()
    model = await maybe(rt.AIModel.load(aimodel, rt.SpecializationOptions.cpu_only()))
    fn = await maybe(model.load_function("main"))
    return fn, time.monotonic() - t0


def nd(a):
    import coreai.runtime as rt
    return rt.NDArray(np.ascontiguousarray(a))


async def shipped_spec_call(aimodel: Path, ids: np.ndarray, np_dtype) -> dict:
    """The shipped (dynamic) graph on cpu_only: load, describe, one call of 16 ids from zero states."""
    out: dict = {}
    fn, secs = await load(aimodel)
    out["load_seconds"] = round(secs, 3)
    out["descriptor"] = desc = fn_desc(fn)
    state = {n: nd(np.zeros([MAX_CTX if s < 0 else s for s in shape], np.dtype(dt)))
             for n, (shape, dt) in desc["states"].items()}
    try:
        res = await maybe(fn(inputs={"input_ids": nd(ids[:S][None]), "position_ids": nd(np.arange(S, dtype=np.int32)[None]),
                                     "image_embeds": nd(np.zeros((N_IMG, 64), np_dtype))}, state=state))
        out["first_call"] = f"ok {list(np.asarray(res['hidden'].numpy()).shape)}"
    except Exception as e:  # noqa: BLE001
        out["first_call"] = f"{type(e).__name__}: {e}"
    return out


async def static_chain(assets: list[Path], ids: np.ndarray, ext: np.ndarray, img_ext: np.ndarray, np_dtype) -> dict:
    """The row through one static graph per call position, the states carried between them."""
    from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN

    fns, load_s, descs = [], [], []
    for a in assets:
        fn, secs = await load(a)
        fns.append(fn)
        load_s.append(round(secs, 3))
        descs.append(fn_desc(fn))

    def fresh() -> dict:
        return {n: nd(np.zeros([TRACE_KV_CACHE_SEQ_LEN if s < 0 else s for s in shape], np.dtype(dt)))
                for n, (shape, dt) in descs[0]["states"].items()}

    async def row(x_ids: np.ndarray, img: np.ndarray) -> tuple[np.ndarray, list]:
        state, out, ms = fresh(), [], []
        for c in range(-(-len(x_ids) // S)):
            x = np.full(S, PAD, np.int32)
            piece = x_ids[c * S:(c + 1) * S]
            x[:len(piece)] = piece
            t = time.perf_counter()
            res = await maybe(fns[c](inputs={"input_ids": nd(x[None]),
                                             "position_ids": nd(np.arange(c * S + S, dtype=np.int32)[None]),
                                             "image_embeds": nd(img)}, state=state))
            ms.append((time.perf_counter() - t) * 1e3)
            h = np.asarray(res["hidden"].numpy())
            if h.shape != (1, S, 64):
                raise SystemExit(f"hidden {h.shape} != (1, {S}, 64)")
            out.append(h[0, :len(piece)].copy())
        return np.concatenate(out), ms

    zero = np.zeros((N_IMG, 64), np_dtype)
    h1, ms1 = await row(ids, zero)
    h2, _ = await row(ids, zero)
    he, _ = await row(ext, img_ext.astype(np_dtype))
    hz, _ = await row(ext, zero)
    return {"descriptors": descs, "load_seconds": load_s, "call_ms": [round(x, 3) for x in ms1],
            "hidden_dtype": str(h1.dtype), "h": h1, "rerun_bit_equal": bool(np.array_equal(h1, h2)),
            "ext_bit_equal": bool(np.array_equal(he, h1)), "ext_zero": hz}


def versions() -> dict:
    out = {"python": platform.python_version(), "torch": torch.__version__}
    for d in ("coreai-core", "coreai-torch", "coreai-opt", "coreai-models", "numpy"):
        try:
            out[d] = md.version(d)
        except md.PackageNotFoundError:
            out[d] = None
    try:
        out["macos_build"] = __import__("subprocess").run(["sw_vers", "-buildVersion"], capture_output=True,
                                                          text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        out["macos_build"] = None
    return out


def run_dtype(model: Lfm2D1Decoder, ids: np.ndarray, ref: np.ndarray, dtype: torch.dtype, run_dir: Path) -> dict:
    name = str(dtype).replace("torch.", "")
    np_dtype = np.float16 if dtype == torch.float16 else np.float32
    m = typed_module(model, dtype)
    entry: dict = {"dtype": name, "fp32_params": sorted(n for n, p in m.named_parameters() if p.dtype == torch.float32)}
    ext, _ = ext_row(ids, model.model.embed_tokens.weight.detach(), torch.float32)
    _, img = ext_row(ids, m.model.embed_tokens.weight.detach(), dtype)
    try:   # the shipped spec: export, load, one call (the CPU-only runtime refuses the dynamic graph)
        a = run_dir / name / f"d1_toy_{name}_pf{S}.aimodel"
        entry["shipped_spec"] = export_one(m, dtype, a)
        entry["shipped_spec"].update(asyncio.run(shipped_spec_call(a, ids, np_dtype)))
    except Exception as e:  # noqa: BLE001
        entry["shipped_spec"] = {"error": f"{type(e).__name__}: {e}"[:2000],
                                 "traceback_tail": traceback.format_exc().splitlines()[-20:]}
    try:   # the same module as a chain of static graphs, one per call position
        assets, exports = [], []
        for c in range(-(-T // S)):
            a = run_dir / name / f"d1_toy_{name}_pf{S}_pos{c * S + S}.aimodel"
            exports.append(export_one(m, dtype, a, static_pos=c * S + S))
            assets.append(a)
        got = asyncio.run(static_chain(assets, ids, ext, img.float().numpy(), np_dtype))
        h, hz = got.pop("h"), got.pop("ext_zero")
        chain = {"exports": exports, **got, "vs_eager_fp32_oneshot": compare(ref, h),
                 "ext_zero_embeds_differs": compare(h.astype(np.float32), hz.astype(np.float32))}
        chain["pass"] = bool(chain["vs_eager_fp32_oneshot"]["cos_min"] >= COS_BAR and chain["rerun_bit_equal"]
                             and chain["ext_bit_equal"] and chain["ext_zero_embeds_differs"]["max_abs_diff"] > 0
                             and chain["vs_eager_fp32_oneshot"]["finite"])
        entry["static_chain"] = chain
    except Exception as e:  # noqa: BLE001
        entry["static_chain"] = {"error": f"{type(e).__name__}: {e}"[:2000],
                                 "traceback_tail": traceback.format_exc().splitlines()[-20:], "pass": False}
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eager-only", action="store_true", help="torch only: no export, no runtime")
    ap.add_argument("--dtypes", default="float16,float32")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    model = toy_model()
    ids = row_ids()
    eager, ref = eager_checks(model, ids)
    cfg = model.config
    rec: dict = {"what": __doc__.splitlines()[0], "started": started, "versions": versions(),
                 "toy_config": {"hidden_size": cfg.hidden_size, "num_hidden_layers": cfg.num_hidden_layers,
                                "layer_types": cfg.layer_types, "vocab_size": cfg.vocab_size, "ff_dim": cfg.ff_dim,
                                "num_attention_heads": cfg.num_attention_heads, "num_key_value_heads": cfg.num_key_value_heads,
                                "head_dim": cfg.head_dim, "conv_L_cache": cfg.conv_L_cache, "rope_theta": cfg.rope_theta,
                                "n_image_tokens": N_IMG, "seed": SEED},
                 "row": {"T": T, "S": S, "calls": -(-T // S), "pad_id": PAD, "max_ctx": MAX_CTX,
                         "ext_positions": list(EXT_POSITIONS), "ext_slots": list(EXT_SLOTS)},
                 "eager_fp32": eager, "cos_bar": COS_BAR}
    out = Path(args.out) if args.out else LANE / "results" / ("toy_graph_eager.json" if args.eager_only else "toy_graph.json")
    if not args.eager_only:
        run_dir = LANE / "exports" / "toy" / datetime.now().strftime("%Y%m%d_%H%M%S")
        rec["run_dir"] = str(run_dir)
        rec["dtypes"] = {}
        for name in args.dtypes.split(","):
            entry = run_dtype(model, ids, ref, getattr(torch, name), run_dir)
            rec["dtypes"][name] = entry
            ch = entry.get("static_chain", {})
            print(name, json.dumps({"shipped_spec_first_call": entry.get("shipped_spec", {}).get("first_call"),
                                    "shipped_spec_error": entry.get("shipped_spec", {}).get("error"),
                                    "chain": {k: ch.get(k) for k in ("vs_eager_fp32_oneshot", "rerun_bit_equal",
                                                                     "ext_bit_equal", "pass", "error", "call_ms")}}),
                  flush=True)
        rec["pass"] = all(e.get("static_chain", {}).get("pass") for e in rec["dtypes"].values())
    rec["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec["eager_fp32"]["oneshot_vs_chunked_s16"]), "ext bit-equal (eager):", rec["eager_fp32"]["ext_rows_bit_equal"])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
