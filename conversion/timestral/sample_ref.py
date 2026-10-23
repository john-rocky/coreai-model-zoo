"""Reference forecasts for the kit sample's Swift host (CoreAIKit `KitTimestralForecaster`).

For every fixture window that fits a set's bucket, runs `host_forecast.forecast` with that set's bundle on Core AI
(`engines.CoreAIGraph`, the engine the e2e gate uses) and saves, in the set's window order:

  forecast.f32.bin   the forecast, (H, 21) float32 per window, concatenated
  inputs.f32.bin     what host_forecast handed the graph, per window: patches (2, L, 64) [series, mirror],
                     cos (S, 64), sin (S, 64), attn_mask (S, S), float32, concatenated
  raw.f32.bin        the graph's output, (2, h*32, 21) float32 per window
  manifest.json      the set (bundle, unit, dtype, rows, bucket) and one entry per window with its offsets (in floats)

and once for all sets, `windows.json` + `contexts.f32.bin`: every fixture window's saved context as the host reads
it (float32, NaN kept), so the Swift side starts from the same floats.

    python sample_ref.py --fixtures <fixtures> --out <dir> \
        --set fp32=<bundle>,gpu,float32,2,64,32 --set ane=<bundle>,neural_engine,float16,1,64,32 ...

A set is `tag=bundle,unit,dtype,rows,L,h[,skip_le]`: unit cpu | gpu | neural_engine, dtype of the graph's inputs,
rows = the graph's R, (L, h) its bucket in patches; skip_le = leave out the windows whose context already fits
skip_le patches (the windows a smaller bucket takes), so a long bucket holds only the long windows.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import host_forecast as H


def load_windows(fixtures):
    man = json.loads((fixtures / "manifest.json").read_text())
    wins, ctx_parts, offset = [], [], 0
    for g in man["groups"]:
        ctxs = np.load(fixtures / f"{g['name']}.context.npy")
        for b in range(ctxs.shape[0]):
            c = np.asarray(ctxs[b], dtype=np.float32).reshape(-1)
            wins.append(dict(id=f"{g['name']}#{b}", group=g["name"], window=b, horizon=int(g["horizon"]),
                             context_length=int(c.size), seasonality=int(g["seasonality"]),
                             context_offset=offset, context_nan=int(np.isnan(c).sum())))
            ctx_parts.append(c)
            offset += c.size
    return wins, np.concatenate(ctx_parts)


class Recorder:
    """The graph callable with every call's inputs and output kept."""

    def __init__(self, core):
        self.core, self.calls = core, []

    def __call__(self, patches, cos, sin, attn_mask):
        out = self.core(patches, cos, sin, attn_mask)
        self.calls.append((patches.copy(), cos.copy(), sin.copy(), attn_mask.copy(), out.copy()))
        return out


def parse_set(text):
    tag, spec = text.split("=", 1)
    parts = spec.split(",")
    bundle, unit, dtype, rows, n_ctx, n_hor = parts[:6]
    skip_le = int(parts[6]) if len(parts) > 6 else 0
    return dict(tag=tag, bundle=bundle, unit=unit, dtype=dtype, rows=int(rows), n_ctx=int(n_ctx), n_hor=int(n_hor),
                skip_le=skip_le)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--set", action="append", default=[], help="tag=bundle,unit,dtype,rows,L,h[,skip_le]")
    a = ap.parse_args()
    import engines

    a.out.mkdir(parents=True, exist_ok=True)
    wins, contexts = load_windows(a.fixtures)
    contexts.astype(np.float32).tofile(a.out / "contexts.f32.bin")
    (a.out / "windows.json").write_text(json.dumps(dict(
        fixtures=str(a.fixtures), quantile_levels=list(H.QUANTILES), count=len(wins),
        contexts_file="contexts.f32.bin", contexts_floats=int(contexts.size), windows=wins), indent=1) + "\n")
    print(f"windows: {len(wins)}, context floats: {contexts.size}", flush=True)

    for spec in map(parse_set, a.set):
        d = a.out / spec["tag"]
        if (d / "manifest.json").exists():
            print(f"set {spec['tag']}: already written, skipped", flush=True)
            continue
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        core = engines.CoreAIGraph(spec["bundle"], spec["unit"], spec["dtype"], spec["rows"])
        load_s = time.time() - t0
        rec = Recorder(core)
        bucket = H.Bucket(spec["n_ctx"], spec["n_hor"])
        entries, errors = [], []
        f_fc, f_in, f_raw = (open(d / n, "wb") for n in ("forecast.f32.bin", "inputs.f32.bin", "raw.f32.bin"))
        o_fc = o_in = o_raw = 0
        for w in wins:
            ll = math.ceil(w["context_length"] / H.PATCH)
            n_pred = math.ceil(w["horizon"] / H.PATCH)
            if not bucket.fits(ll, n_pred) or ll <= spec["skip_le"]:
                continue
            ctx = contexts[w["context_offset"]:w["context_offset"] + w["context_length"]]
            rec.calls.clear()
            t1 = time.perf_counter()
            try:
                q = H.forecast(rec, ctx, w["horizon"], bucket, mode="batched")
            except ValueError as e:
                errors.append(dict(id=w["id"], error=str(e)))
                continue
            ms = (time.perf_counter() - t1) * 1e3
            assert len(rec.calls) == 1, "batched mode makes one graph call per forecast"
            patches, cos, sin, amask, raw = rec.calls[0]
            q = np.asarray(q, np.float32)
            q.tofile(f_fc)
            parts = [np.asarray(v, np.float32).reshape(-1) for v in (patches, cos, sin, amask)]
            for p in parts:
                p.tofile(f_in)
            np.asarray(raw, np.float32).tofile(f_raw)
            n_in = sum(p.size for p in parts)
            entries.append(dict(id=w["id"], group=w["group"], window=w["window"], horizon=w["horizon"],
                                context_length=w["context_length"], forecast_offset=o_fc, forecast_floats=int(q.size),
                                inputs_offset=o_in, inputs_floats=int(n_in), raw_offset=o_raw, raw_floats=int(raw.size),
                                ms=round(ms, 3)))
            o_fc += q.size
            o_in += n_in
            o_raw += raw.size
        for f in (f_fc, f_in, f_raw):
            f.close()
        man = dict(set=spec, engine="coreai.runtime (engines.CoreAIGraph)", spec_used=core.spec_used,
                   load_error=core.load_error, load_s=round(load_s, 3), bucket=[spec["n_ctx"], spec["n_hor"]],
                   inputs_layout="per window: patches (2, L, 64) [series, mirror], cos (S, 64), sin (S, 64), "
                                 "attn_mask (S, S); float32", raw_layout="(2, h*32, 21) float32",
                   forecast_layout="(H, 21) float32", windows=entries, errors=errors,
                   median_ms=float(np.median([e["ms"] for e in entries])) if entries else None)
        (d / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
        print(f"set {spec['tag']}: {len(entries)} windows, {len(errors)} errors, load {load_s:.2f} s, "
              f"median {man['median_ms']} ms/forecast, spec_used {core.spec_used}", flush=True)
        del core, rec
    print("SAMPLE_REF_DONE", flush=True)


if __name__ == "__main__":
    main()
