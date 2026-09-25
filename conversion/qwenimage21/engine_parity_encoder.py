"""Gate: the EXPORTED Core AI text-encoder bundle vs the fp32 oracle, and what its error does downstream.

1. Host path end to end: ``qi21_tokenize.QI21Tokenizer`` encodes the oracle prompt (asserted equal
   to ``enc_input_ids.i32``; drop_idx asserted equal to meta) -> the bundle's ``main`` ->
   ``hidden [1,Lfull,4096]`` vs ``enc_hidden_full.f32``, per token: corr, max|d|, |d|/|ref|.
   Bar: every token corr >= 0.999, NaN 0. The same output is also scored against the CPU torch
   run of the same variant (``_work/torch_bf16*_text/``) when present — context, not the verdict.
2. Pad independence (dynamic L): the ids right-padded to ``--pad-L`` with ``<|endoftext|>``
   (151643). Its first Lfull outputs are scored against the oracle (same bar) and against the
   unpadded run (a different L may take different GPU kernels, so this difference is reported,
   not required to be zero); and a second L=pad-L run whose tail is random ids must give the
   SAME first Lfull outputs bit for bit (same shape, so only a leak through the causal mask
   could change them).
3. ``--dit``: ``hidden[:, drop_idx:]`` -> the fp32 re-authored DiT (real weights) on oracle steps
   (default 0/20/39), teacher-forced, velocity vs the oracle's ``vel_s``. Bar: corr >= 0.999.
   The oracle's own ``prompt_embeds`` go through the same DiT as the reference row.

Engine outputs are saved to ``_work/engine_text/<bundle stem>/hidden_L<L>.f32``.

Run (coreai-models base venv, from conversion/qwenimage21/; the GPU lock, other sessions share it):
  python3 ~/code/coreai-kit/scripts/with-gpu-lock.py -- ~/code/coreai/coreai-models/.venv/bin/python \\
      engine_parity_encoder.py <name>.h16c.aimodelc --dit          # AOT bundle, SpecializationOptions.default()
  ... engine_parity_encoder.py <name>.aimodel --cpu-only           # isolation: the IR on cpu_only()
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import hf_snapshot  # noqa: E402
from qi21_tokenize import QI21Tokenizer  # noqa: E402

MODEL = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
BAR_TOK_CORR, BAR_DIT_CORR = 0.999, 0.999


def per_token(out: np.ndarray, ref: np.ndarray) -> dict:
    a = out.astype(np.float64).reshape(-1, ref.shape[-1])
    b = ref.astype(np.float64).reshape(-1, ref.shape[-1])
    nan = int(np.isnan(a).sum())
    if nan:
        return dict(nan=nan, zero=False)
    d = a - b
    corr = [float(np.corrcoef(a[i], b[i])[0, 1]) for i in range(a.shape[0])]
    return dict(nan=0, zero=bool(np.abs(a).max() == 0), corr=corr,
                maxd=[float(np.abs(d[i]).max()) for i in range(a.shape[0])],
                rel=[float(np.linalg.norm(d[i]) / np.linalg.norm(b[i])) for i in range(a.shape[0])],
                global_corr=float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
                global_maxd=float(np.abs(d).max()), global_rel=float(np.linalg.norm(d) / np.linalg.norm(b)))


def summary(s: dict) -> str:
    if s["nan"]:
        return f"NaN {s['nan']}"
    c = np.array(s["corr"])
    return (f"min tok corr {c.min():.6f} (tok {int(c.argmin())})  max tok max|d| {max(s['maxd']):.3e} "
            f"(tok {int(np.argmax(s['maxd']))})  max tok |d|/|ref| {max(s['rel']):.3e} (tok {int(np.argmax(s['rel']))})  "
            f"global corr {s['global_corr']:.6f}  |d|/|ref| {s['global_rel']:.3e}  n tok < {BAR_TOK_CORR}: "
            f"{int((c < BAR_TOK_CORR).sum())}  NaN 0")


async def run_engine(bundle: Path, cpu_only: bool, feeds: dict[str, np.ndarray]) -> tuple[dict, dict, float]:
    import coreai.runtime as rt
    opts = rt.SpecializationOptions.cpu_only() if cpu_only else rt.SpecializationOptions.default()
    t0 = time.time()
    model = await rt.AIModel.load(bundle, opts)            # keep the AIModel alive while calling
    fn = model.load_function("main")
    t_load = time.time() - t0
    outs, secs = {}, {}
    for key, ids in feeds.items():
        t1 = time.time()
        r = await fn({"input_ids": rt.NDArray(torch.from_numpy(ids).contiguous())})
        outs[key] = np.array(r["hidden"].numpy(), dtype=np.float32)
        secs[key] = time.time() - t1
    del fn, model
    return outs, secs, t_load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", help="<name>.h16c.aimodelc (default options) or <name>.aimodel with --cpu-only")
    ap.add_argument("--oracle", default=str(HERE / "oracle" / "256"))
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--pad-L", type=int, default=64, help="right-padded length for the pad-independence check (0 = skip)")
    ap.add_argument("--dit", action="store_true", help="downstream gate through the fp32 DiT (28 GB RAM)")
    ap.add_argument("--dit-steps", default="0,20,39")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    bundle = Path(args.bundle)
    unit = "cpu_only" if args.cpu_only else ("default(aot)" if bundle.suffix == ".aimodelc" else "default(jit)")
    stem = bundle.name.split(".")[0]

    od = Path(args.oracle)
    meta = json.load(open(od / "meta.json"))
    Lfull, drop = int(meta["Lfull"]), int(meta["drop_idx"])
    ref = np.fromfile(od / "enc_hidden_full.f32", "<f4").reshape(1, Lfull, -1)
    tok = QI21Tokenizer()
    ids = tok.encode_np(meta["prompt"])
    assert ids.tolist() == [np.fromfile(od / "enc_input_ids.i32", "<i4").tolist()], "host ids != oracle ids"
    assert tok.drop_idx == drop, (tok.drop_idx, drop)
    feeds = {f"L{Lfull}": ids}
    if args.pad_L:
        padded = tok.encode_padded(meta["prompt"], args.pad_L)[0]
        alt = padded.copy()                  # same shape, different tail: isolates leakage from shape numerics
        alt[0, Lfull:] = np.random.default_rng(0).integers(0, 151643, args.pad_L - Lfull, dtype=np.int32)
        feeds[f"L{args.pad_L}pad"] = padded
        feeds[f"L{args.pad_L}alt"] = alt
    print(f"[enc-parity] {bundle.name} ({unit}); oracle {od.name}: prompt {meta['prompt']!r} Lfull {Lfull} "
          f"drop_idx {drop} (host == oracle ids, host drop_idx == meta)", flush=True)

    outs, secs, t_load = asyncio.run(run_engine(bundle, args.cpu_only, feeds))
    print(f"[enc-parity] load {t_load:.1f}s; " + "  ".join(f"{k}: fwd {v:.2f}s" for k, v in secs.items()), flush=True)
    sd = HERE / "_work" / "engine_text" / f"{stem}_{unit.split('(')[0]}"
    sd.mkdir(parents=True, exist_ok=True)
    for k, v in outs.items():
        np.ascontiguousarray(v, "<f4").tofile(sd / f"hidden_{k}.f32")

    h = outs[f"L{Lfull}"]
    s = per_token(h, ref)
    ok_enc = s["nan"] == 0 and not s["zero"] and min(s["corr"]) >= BAR_TOK_CORR
    res = dict(bundle=str(bundle), unit=unit, oracle=str(od), Lfull=Lfull, drop_idx=drop, load_s=t_load,
               fwd_s=secs, vs_oracle=s, ok_enc=ok_enc)
    print(f"[enc-parity] engine vs oracle (L={Lfull}): {summary(s)}  {'PASS' if ok_enc else 'FAIL'}", flush=True)
    if s["nan"] == 0:
        print("  tok  id      corr      max|d|     |d|/|ref|", flush=True)
        for i in range(Lfull):
            print(f"  {i:3d} {int(ids[0, i]):6d}  {s['corr'][i]:.6f}  {s['maxd'][i]:.3e}  {s['rel'][i]:.3e}", flush=True)
    variant = ("torch_fp32_text" if "w16a32" in stem else          # w16a32 torch == the fp32 re-author
               "torch_bf16r32_text" if "_r32" in stem else "torch_bf16_text")
    tv = HERE / "_work" / variant / od.name / "hidden.f32"
    if tv.exists() and s["nan"] == 0:
        st = per_token(h, np.fromfile(tv, "<f4").reshape(1, Lfull, -1))
        res["vs_cpu_torch_same_variant"] = st
        print(f"[enc-parity] context — engine vs CPU torch {variant}: {summary(st)}", flush=True)

    if args.pad_L:
        hp = outs[f"L{args.pad_L}pad"]
        assert hp.shape == (1, args.pad_L, ref.shape[-1]), hp.shape
        sp = per_token(hp[:, :Lfull], ref)
        d = np.abs(hp[:, :Lfull].astype(np.float64) - h.astype(np.float64))
        same = bool(np.array_equal(hp[:, :Lfull], h))
        ha = outs[f"L{args.pad_L}alt"]
        tail_blind = bool(np.array_equal(ha[:, :Lfull], hp[:, :Lfull]))
        ok_pad = sp["nan"] == 0 and min(sp["corr"]) >= BAR_TOK_CORR and tail_blind
        res["pad"] = dict(L=args.pad_L, bit_exact_vs_unpadded=same, max_abs_vs_unpadded=float(d.max()),
                          rel_vs_unpadded=float(np.linalg.norm(d) / np.linalg.norm(h.astype(np.float64))),
                          tail_content_blind_bit_exact=tail_blind,
                          alt_tail_max_abs=float(np.abs(ha[:, :Lfull] - hp[:, :Lfull]).max()),
                          vs_oracle=sp, ok=ok_pad)
        print(f"[enc-parity] pad L={args.pad_L} (+{args.pad_L - Lfull} x 151643): first {Lfull} tokens vs unpadded "
              f"run: bit-exact {same}  max|d| {float(d.max()):.3e}  |d|/|h| {res['pad']['rel_vs_unpadded']:.3e}; "
              f"vs oracle: {summary(sp)}", flush=True)
        print(f"[enc-parity] pad L={args.pad_L}, tail = 151643 vs random ids: first {Lfull} tokens bit-exact "
              f"{tail_blind} (max|d| {res['pad']['alt_tail_max_abs']:.3e}) — the valid tokens never read the tail  "
              f"{'PASS' if ok_pad else 'FAIL'}", flush=True)

    ok = ok_enc and res.get("pad", {}).get("ok", True)
    if args.dit and s["nan"] == 0:
        gc.collect()
        from qi21_dit import load_qi21_dit
        from qi21_host import build_inputs
        from qi21_oracle import Oracle, parse_steps
        o = Oracle(od)
        tdir = Path(hf_snapshot(MODEL, revision=REVISION)) / "transformer"
        axes = tuple(json.load(open(tdir / "config.json")).get("axes_dims_rope", (16, 56, 56)))
        t0 = time.time()
        dit = load_qi21_dit(tdir, dtype=torch.float32)
        print(f"[enc-parity] DiT fp32 loaded in {time.time() - t0:.0f}s", flush=True)
        pe_engine = torch.from_numpy(h[:, drop:].copy())
        rows = []
        with torch.no_grad():
            for st in parse_steps(args.dit_steps, o.steps):
                vref = o.vel(st).double().numpy().ravel()
                row = dict(step=st, t=float(o.t(st)))
                for name, pe in (("oracle_prompt_embeds", o.prompt_embeds), ("engine_hidden", pe_engine)):
                    v = dit(**build_inputs(o.latent(st), pe, o.t(st), o.H, o.W, axes_dim=axes)).double().numpy().ravel()
                    row[name] = dict(corr=float(np.corrcoef(v, vref)[0, 1]), maxd=float(np.abs(v - vref).max()),
                                     rel=float(np.linalg.norm(v - vref) / np.linalg.norm(vref)))
                row["ok"] = row["engine_hidden"]["corr"] >= BAR_DIT_CORR
                rows.append(row)
                print(f"  DiT step {st:2d} t={row['t']:.6f}: vel corr vs oracle — oracle prompt_embeds "
                      f"{row['oracle_prompt_embeds']['corr']:.6f} | engine hidden {row['engine_hidden']['corr']:.6f}  "
                      f"max|d| {row['engine_hidden']['maxd']:.3e}  |d|/|ref| {row['engine_hidden']['rel']:.3e}  "
                      f"{'PASS' if row['ok'] else 'FAIL'}", flush=True)
        res["dit"] = rows
        ok = ok and all(r["ok"] for r in rows)
    res["ok"] = ok
    print(f"\n[enc-parity] {'PASS' if ok else 'FAIL'} (bar: every token corr >= {BAR_TOK_CORR}, NaN 0"
          f"{', pad-independent' if args.pad_L else ''}{', DiT vel corr >= %g' % BAR_DIT_CORR if args.dit else ''})",
          flush=True)
    js = Path(args.json or HERE / "_work" / f"engine_parity_encoder_{stem}_{unit.split('(')[0]}.json")
    json.dump(res, open(js, "w"), indent=2)
    print(f"[enc-parity] -> {js}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
