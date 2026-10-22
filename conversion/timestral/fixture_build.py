"""Build the Timestral parity fixture (public windows + synthetic series).

Reads the raw source files under --raw, one directory per source, and writes one group per (source, context class,
horizon) to the fixture directory:

    <raw>/fev_uci_air_quality_1H/train-00000-of-00001.parquet   autogluon/fev_datasets @ FEV_REV, config uci_air_quality_1H
    <raw>/fev_boomlet_{1062,1631,1855}/train-00000-of-00001.parquet   the same repo, configs boomlet_1062 / 1631 / 1855
    <raw>/gifteval_electricity_{H,D}/data-00000-of-00001.arrow   Salesforce/GiftEval @ GIFT_REV, electricity/H and /D

    <group>.context.npy  float32 [B, L]   saved context, NaN = missing (fed to the oracle as is)
    <group>.target.npy   float32 [B, H]   ground truth after the cutoff
    manifest.json                         every group: source, licence, revision, window ids, L, H, m

Run with an interpreter that has numpy + pyarrow. Paths are passed as arguments. The sources' licences and the
windows' rules are in the fixture's RECIPE.md.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

QUANTILES = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
             0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

FEV_REV = "d3e7e720f4158eab03ebcb021f1cce779cd92acc"
GIFT_REV = "30841734ac5cfddbd0c3bad6d09d2b6b32becbb0"
LIC_UCI = ("UCI ML Repository: \"This dataset is licensed under a Creative Commons Attribution 4.0 "
           "International (CC BY 4.0) license.\"")
LIC_BOOM = "Datadog/BOOM dataset card: \"license: apache-2.0\""


def read_arrow_table(path: Path) -> pa.Table:
    with open(path, "rb") as fh:
        try:
            return pa.ipc.open_stream(fh).read_all()
        except pa.ArrowInvalid:
            fh.seek(0)
            return pa.ipc.open_file(fh).read_all()


def as_f32(seq) -> np.ndarray:
    return np.asarray(seq, dtype=np.float32)


def informative(ctx: np.ndarray, tgt: np.ndarray) -> bool:
    """At least two distinct observed context values and one observed target value."""
    obs = ctx[np.isfinite(ctx)]
    return obs.size >= 2 and np.unique(obs).size >= 2 and np.isfinite(tgt).any()


def cut(series: np.ndarray, cutoff: int, horizon: int, ctx_len: int):
    """`series[max(0, cutoff - ctx_len):cutoff]` and `series[cutoff:cutoff + horizon]`."""
    assert 0 < cutoff and cutoff + horizon <= series.size, (cutoff, horizon, series.size)
    return series[max(0, cutoff - ctx_len):cutoff].copy(), series[cutoff:cutoff + horizon].copy()


class Writer:
    def __init__(self, out: Path):
        self.out = out
        self.groups = []

    def add(self, name, contexts, targets, meta):
        ls = {c.size for c in contexts}
        assert len(ls) == 1, f"{name}: one context length per group, got {sorted(ls)}"
        hs = {t.size for t in targets}
        assert len(hs) == 1, f"{name}: one horizon per group, got {sorted(hs)}"
        c = np.stack(contexts).astype(np.float32)
        t = np.stack(targets).astype(np.float32)
        np.save(self.out / f"{name}.context.npy", c)
        np.save(self.out / f"{name}.target.npy", t)
        meta = dict(name=name, n_windows=int(c.shape[0]), context_length=int(c.shape[1]),
                    horizon=int(t.shape[1]),
                    context_nan=int(np.isnan(c).sum()), target_nan=int(np.isnan(t).sum()), **meta)
        self.groups.append(meta)
        print(f"{name:28s} B={c.shape[0]:3d} L={c.shape[1]:5d} H={t.shape[1]:4d} "
              f"nan(ctx)={meta['context_nan']} nan(tgt)={meta['target_nan']}")


def fev_windows(series, horizon, num_windows, ws, ctx_len):
    """fev windows: cutoff_i = T - horizon * (num_windows - i); `ws` counts back from the last (1 = last)."""
    t = series.size
    out = []
    for w in ws:
        i = num_windows - w
        cutoff = t + (-horizon - (num_windows - 1) * horizon) + i * horizon
        out.append((w, cutoff) + cut(series, cutoff, horizon, ctx_len))
    return out


def build_air_quality(raw: Path, wr: Writer):
    tab = pq.read_table(raw / "fev_uci_air_quality_1H" / "train-00000-of-00001.parquet")
    assert tab.num_rows == 1, tab.num_rows
    targets = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]  # fev-bench task target list
    horizon, nw, m = 168, 20, 24
    item = tab.column("id")[0].as_py()
    for cls, ctx_len, ws in (("c2048", 2048, (1, 2, 3, 4)), ("c8192", 8192, (1,))):
        ctxs, tgts, ids = [], [], []
        for col in targets:
            s = as_f32(tab.column(col)[0].as_py())
            for w, cutoff, c, y in fev_windows(s, horizon, nw, ws, ctx_len):
                ctxs.append(c); tgts.append(y)
                ids.append(dict(item=item, column=col, window_from_end=w, cutoff=int(cutoff), series_length=int(s.size)))
        wr.add(f"airq_h168_{cls}", ctxs, tgts, dict(
            source="autogluon/fev_datasets config uci_air_quality_1H (from UCI ML Repository dataset 360 'Air Quality')",
            revision=FEV_REV, licence=LIC_UCI, freq="1h", seasonality=m,
            setting="fev-bench task uci_air_quality_1H: horizon 168, num_windows 20, seasonality 24; "
                    "each target column forecast as its own univariate series; covariates not used",
            windows=ids))


def build_boom(raw: Path, wr: Writer):
    spec = [  # (config, freq, horizon, num_windows, seasonality, classes)
        ("1062", "5min", 288, 20, 288, (("c2048", 2048, (1, 2, 3)), ("c8192", 8192, (1,)))),
        ("1631", "30min", 96, 20, 48, (("c2048", 2048, (1, 2)),)),
        ("1855", "1h", 24, 20, 24, (("c2048", 2048, (1, 2)),)),
    ]
    for cfg, freq, horizon, nw, m, classes in spec:
        tab = pq.read_table(raw / f"fev_boomlet_{cfg}" / "train-00000-of-00001.parquet")
        assert tab.num_rows == 1, tab.num_rows
        item = tab.column("id")[0].as_py()
        cols = sorted([c for c in tab.column_names if c.startswith("target_")], key=lambda c: int(c.split("_")[1]))
        for cls, ctx_len, ws in classes:
            ctxs, tgts, ids = [], [], []
            for col in cols:  # first four informative variates in column order
                if len({i["column"] for i in ids}) == 4:
                    break
                s = as_f32(tab.column(col)[0].as_py())
                wins = fev_windows(s, horizon, nw, ws, ctx_len)
                if not all(informative(c, y) for _, _, c, y in wins):
                    continue
                for w, cutoff, c, y in wins:
                    ctxs.append(c); tgts.append(y)
                    ids.append(dict(item=item, column=col, window_from_end=w, cutoff=int(cutoff), series_length=int(s.size)))
            wr.add(f"boom{cfg}_h{horizon}_{cls}", ctxs, tgts, dict(
                source=f"autogluon/fev_datasets config boomlet_{cfg} (from Datadog/BOOM)",
                revision=FEV_REV, licence=LIC_BOOM, freq=freq, seasonality=m,
                setting=f"fev-bench task boomlet_{cfg}: horizon {horizon}, num_windows {nw}, seasonality {m}; "
                        "first four informative target variates in column order, each univariate",
                windows=ids))


def gift_windows(min_len, pred_len):
    """GIFT-Eval: windows = min(max(1, ceil(0.1 * min_series_length / prediction_length)), 20)."""
    return min(max(1, int(np.ceil(0.1 * min_len / pred_len))), 20)


def build_electricity(raw: Path, wr: Writer):
    lic = LIC_UCI + "; Salesforce/GiftEval dataset card: \"license: apache-2.0\""
    for freq_dir, freq, m, groups in (
        ("gifteval_electricity_H", "H", 24, (("h48", 48, "short", (("c2048", 2048, 8, 0),)),
                                            ("h720", 720, "long", (("c2048", 2048, 4, 8), ("c8192", 8192, 4, 8))))),
        ("gifteval_electricity_D", "D", 1, (("h30", 30, "short", (("cfull", 1 << 30, 4, 12),)),)),
    ):
        tab = read_arrow_table(raw / freq_dir / "data-00000-of-00001.arrow")
        ids_all = tab.column("item_id").to_pylist()
        col = tab.column("target").combine_chunks()
        offs = col.offsets.to_numpy()
        vals = col.values.to_numpy(zero_copy_only=False)
        series = [vals[offs[i]:offs[i + 1]].astype(np.float32) for i in range(len(col))]
        freqs = set(tab.column("freq").to_pylist())
        min_len = min(s.size for s in series)
        for hname, pred_len, term, classes in groups:
            nw = gift_windows(min_len, pred_len)
            for cls, ctx_len, k, skip in classes:
                if cls == "cfull":  # whole history before the cutoff, cropped to the shortest series
                    ctx_len = min_len - pred_len
                ctxs, tgts, ids = [], [], []
                for row in range(skip, len(series)):
                    if len(ids) == k:
                        break
                    s = series[row]
                    cutoff = s.size - pred_len  # last GIFT-Eval test window
                    c, y = cut(s, cutoff, pred_len, ctx_len)
                    if not informative(c, y):
                        continue
                    ctxs.append(c); tgts.append(y)
                    ids.append(dict(item=ids_all[row], row=row, window_from_end=1, cutoff=int(cutoff), series_length=int(s.size)))
                name = f"elec{'d' if freq == 'D' else ''}_{hname}_{cls}"
                wr.add(name, ctxs, tgts, dict(
                    source=f"Salesforce/GiftEval electricity/{freq} (from UCI ML Repository dataset 321 "
                           "'ElectricityLoadDiagrams20112014')",
                    revision=GIFT_REV, licence=lic, freq=sorted(freqs)[0], seasonality=m,
                    setting=f"GIFT-Eval {term}-term: prediction_length {pred_len}, windows {nw} "
                            f"(min series length {min_len}); last test window; seasonality = gluonts get_seasonality",
                    windows=ids))


def build_synthetic(wr: Writer, seed: int):
    """Each series is drawn as context + horizon in one piece, so the target continues the context."""
    rng = np.random.default_rng(seed)
    lic = "synthetic (this fixture), CC0-1.0"

    def series(n, f):
        return f(np.arange(n, dtype=np.float64))

    specs = []  # (name, context length, horizon, (formula, seasonality), generator); draws in list order
    e = rng.normal(0.0, 0.3, 1024 + 64)
    specs.append(("syn_seasonal_trend", 1024, 64, ("10 + 0.01 t + 3 sin(2 pi t / 24) + N(0, 0.3)", 24),
                  lambda t: 10 + 0.01 * t + 3 * np.sin(2 * np.pi * t / 24) + e[: t.size]))
    w = np.cumsum(rng.normal(0.0, 1.0, 2048 + 128))
    specs.append(("syn_random_walk", 2048, 128, ("cumsum N(0, 1)", 1), lambda t: w[: t.size]))
    st = rng.normal(0.0, 0.5, 1536 + 96)
    specs.append(("syn_level_shift", 1536, 96, ("20, +15 at t >= 1200, + N(0, 0.5)", 1),
                  lambda t: 20 + 15 * (t >= 1200) + st[: t.size]))
    specs.append(("syn_constant", 512, 32, ("constant 5.0", 1), lambda t: np.full(t.size, 5.0)))
    ss = rng.normal(0.0, 0.2, 20 + 32)
    specs.append(("syn_short_20", 20, 32, ("20 points: 3 + sin(2 pi t / 7) + N(0, 0.2)", 7),
                  lambda t: 3 + np.sin(2 * np.pi * t / 7) + ss[: t.size]))
    sb = rng.normal(0.0, 0.3, 100 + 24)
    specs.append(("syn_short_100", 100, 24, ("100 points: 50 + 5 sin(2 pi t / 12) + N(0, 0.3)", 12),
                  lambda t: 50 + 5 * np.sin(2 * np.pi * t / 12) + sb[: t.size]))
    sl = rng.normal(0.0, 0.3, 1024 + 64)
    specs.append(("syn_leading_nan", 1024, 64, ("first 300 NaN, then 7 + 2 sin(2 pi t / 24) + N(0, 0.3)", 24),
                  lambda t: 7 + 2 * np.sin(2 * np.pi * t / 24) + sl[: t.size]))
    sg = rng.normal(0.0, 0.3, 1024 + 64)
    specs.append(("syn_gap_nan", 1024, 64, ("NaN hole t in [400, 520), 12 + 4 sin(2 pi t / 48) + N(0, 0.3)", 48),
                  lambda t: 12 + 4 * np.sin(2 * np.pi * t / 48) + sg[: t.size]))
    sm = rng.normal(0.0, 2.0, 1024 + 64)
    specs.append(("syn_level_1e6", 1024, 64, ("1e6 + 50 sin(2 pi t / 24) + N(0, 2)", 24),
                  lambda t: 1e6 + 50 * np.sin(2 * np.pi * t / 24) + sm[: t.size]))
    sn = rng.normal(0.0, 0.5, 1024 + 64)
    specs.append(("syn_negative", 1024, 64, ("-80 - 0.02 t + 6 sin(2 pi t / 24) + N(0, 0.5)", 24),
                  lambda t: -80 - 0.02 * t + 6 * np.sin(2 * np.pi * t / 24) + sn[: t.size]))
    spikes = (rng.random(1024 + 64) < 0.08) * rng.poisson(4.0, 1024 + 64)
    specs.append(("syn_intermittent", 1024, 64, ("0 with Bernoulli(0.08) x Poisson(4) spikes", 1),
                  lambda t: spikes[: t.size].astype(np.float64)))
    sx = rng.normal(0.0, 0.4, 8192 + 256)
    specs.append(("syn_long_seasonal", 8192, 256, ("8192 points: 30 + 5 sin(2 pi t / 168) + 2 sin(2 pi t / 24) + N(0, 0.4)", 24),
                  lambda t: 30 + 5 * np.sin(2 * np.pi * t / 168) + 2 * np.sin(2 * np.pi * t / 24) + sx[: t.size]))

    for name, n_ctx, horizon, desc, f in specs:
        full = series(n_ctx + horizon, f).astype(np.float64)
        ctx, tgt = full[:n_ctx].copy(), full[n_ctx:].copy()
        if name == "syn_leading_nan":
            ctx[:300] = np.nan
        if name == "syn_gap_nan":
            ctx[400:520] = np.nan
        wr.add(name, [ctx.astype(np.float32)], [tgt.astype(np.float32)], dict(
            source="synthetic", revision=f"numpy default_rng({seed}), draws in this file's order", licence=lic,
            freq="1", seasonality=int(desc[1]), setting=desc[0], windows=[dict(item=name, window_from_end=1)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20261022)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    wr = Writer(a.out)
    build_air_quality(a.raw, wr)
    build_boom(a.raw, wr)
    build_electricity(a.raw, wr)
    build_synthetic(wr, a.seed)
    manifest = dict(quantile_levels=QUANTILES, groups=wr.groups,
                    public_windows=sum(g["n_windows"] for g in wr.groups if g["source"] != "synthetic"),
                    synthetic_windows=sum(g["n_windows"] for g in wr.groups if g["source"] == "synthetic"))
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("public windows", manifest["public_windows"], "synthetic", manifest["synthetic_windows"])


if __name__ == "__main__":
    main()
