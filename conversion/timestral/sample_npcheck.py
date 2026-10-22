"""The NumPy operations host_forecast.py leans on, written down for the Swift port to be checked against:

  trig.f32.bin     the RoPE angles host_forecast feeds np.cos / np.sin (positions -255 ... 287 times the 32 inverse
                   frequencies, float32) followed by NumPy's float32 cos and sin of them
  asinh.f32.bin    float32 inputs over [-40, 40] (and some large ones) with np.arcsinh and np.sinh of them
  sums.json        which order np.sum(axis=-1) uses on rows of 32 float64 values (sequential, pairwise with the
                   identity as the start, or pairwise with the first element as the start), and np.cumsum's

    python sample_npcheck.py --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np

import host_forecast as H


def pairwise(a, start_with_first):
    """NumPy's pairwise_sum for n <= 128 (8 accumulators), started from 0 or from the first element."""
    vals = list(a)
    res0 = 0.0
    if start_with_first:
        res0, vals = vals[0], vals[1:]
    n = len(vals)
    if n < 8:
        r = 0.0
        for v in vals:
            r += v
        return res0 + r
    r = vals[:8]
    i = 8
    while i < n - (n % 8):
        for k in range(8):
            r[k] += vals[i + k]
        i += 8
    res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
    while i < n:
        res += vals[i]
        i += 1
    return res0 + res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    pos = np.arange(-255, 288)
    freqs = (pos.astype(np.float32)[:, None] * H.INV_FREQ[None, :]).reshape(-1)
    emb = np.concatenate([freqs, freqs])  # a contiguous float32 stream, as rope_tables hands np.cos
    c, s = np.cos(emb), np.sin(emb)
    np.concatenate([emb, c, s]).astype(np.float32).tofile(a.out / "trig.f32.bin")

    rng = np.random.default_rng(5)
    x = np.concatenate([rng.uniform(-40, 40, 200_000), rng.uniform(-1e-3, 1e-3, 20_000),
                        np.array([0.0, -0.0, 1e-30, -1e-30, 88.0, 89.5, -89.5, 1e6, -1e6, 3e38])]).astype(np.float32)
    np.concatenate([x, np.arcsinh(x), np.sinh(x)]).astype(np.float32).tofile(a.out / "asinh.f32.bin")

    rows = rng.normal(0, 1, (2000, 32)) * 10 ** rng.uniform(-3, 6, (2000, 1))
    rows = np.ascontiguousarray(rows.reshape(1, 2000, 32))
    got = rows.sum(-1)[0]
    seq = [sum_seq(r) for r in rows[0]]
    pw0 = [pairwise(r, False) for r in rows[0]]
    pw1 = [pairwise(r, True) for r in rows[0]]
    means = rows.mean(-1)[0]
    cum = np.cumsum(rows[0, 0])
    cum_seq = np.array([sum_seq(rows[0, 0][: k + 1]) for k in range(32)])
    res = dict(
        numpy=np.__version__, rows=2000,
        sum_equals_sequential=int(sum(g == v for g, v in zip(got, seq))),
        sum_equals_pairwise_from_zero=int(sum(g == v for g, v in zip(got, pw0))),
        sum_equals_pairwise_from_first=int(sum(g == v for g, v in zip(got, pw1))),
        mean_equals_pairwise_from_zero_over_32=int(sum(m == v / 32.0 for m, v in zip(means, pw0))),
        cumsum_equals_sequential=bool((cum == cum_seq).all()),
        inv_freq_bits=[int(v) for v in H.INV_FREQ.view(np.uint32)],
        trig_count=int(emb.size), asinh_count=int(x.size))
    (a.out / "sums.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "inv_freq_bits"}), flush=True)
    print("SAMPLE_NPCHECK_DONE", flush=True)


def sum_seq(r):
    t = 0.0
    for v in r:
        t += v
    return t


if __name__ == "__main__":
    main()
