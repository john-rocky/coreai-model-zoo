"""Pack the fixture windows' graph inputs for the device harness (apps/TimestralGate): one folder per bucket.

    PY device_pack.py --fixtures <fixtures dir> --refs <dir of ref_L64_h32.npz, ref_L256_h32.npz> --out <inputs dir>

A bucket (L context patches, h horizon patches) holds the windows it serves in the end-to-end check (e2e_gate.py), in
manifest order:
  L64_h32   every group with a context of at most 2,048 steps (the main bucket)
  L256_h32  every group with a longer context (8,192 steps)
  L64_h4    the groups of L64_h32 with a horizon of at most 128 steps (the h = 32 vs h = 4 comparison)
and per window the host's graph inputs (host_forecast.bucket_inputs, positions from the first observed patch) for the
two rows [s, -s] of symmetric averaging: row 0 = the series, row 1 = its mirror. An R = 1 graph takes them in two
calls, as the host's split mode does.
  patches.<dt>.bin  [N, 2, L, 64]          cos.<dt>.bin, sin.<dt>.bin  [N, L+h, 64]
  mask.<dt>.bin     [N, 1, 1, L+h, L+h]    (1 = attend, 0 = blocked)
dt = f32, and f16 = numpy astype (round to nearest even), what the Python engine feeds a float16 bundle.
sanity_ref.f32.bin = the released module's fp32 output for window 0 ([2, h*32, 21]) where gate_core.py --save-ref saved
a reference (ref_<bucket>.npz): the device logs its max |difference| from it as a first look, the judgement is made on
the Mac.
manifest.json: the bucket, the shapes, every window (id = <group>/<index>, context length, horizon, patches to
forecast, p = leading patches with no observation) and every file's md5.
The L64_h32 and L256_h32 packs must equal, bit for bit, the inputs saved next to those references; the script stops
if they do not.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import host_forecast as H

BUCKETS = [
    ("L64_h32", 64, 32, "ref_L64_h32.npz", lambda g: g["context_length"] <= 64 * H.PATCH),
    ("L256_h32", 256, 32, "ref_L256_h32.npz", lambda g: g["context_length"] > 64 * H.PATCH),
    ("L64_h4", 64, 4, None, lambda g: g["context_length"] <= 64 * H.PATCH and g["horizon"] <= 4 * H.PATCH),
]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def first_observed(mask, n_ctx):
    extra = n_ctx * H.PATCH - mask.size
    mb = np.concatenate([np.zeros(extra, bool), mask])
    obs = mb.reshape(n_ctx, H.PATCH).any(-1)
    return int(np.argmax(obs)) if obs.any() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True, type=Path)
    ap.add_argument("--refs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    man = json.loads((a.fixtures / "manifest.json").read_text())
    summary = {}
    for name, n_ctx, n_hor, ref_name, select in BUCKETS:
        rows = {k: [] for k in ("patches", "cos", "sin", "mask")}
        windows = []
        for g in man["groups"]:
            if not select(g):
                continue
            ctxs = np.load(a.fixtures / f"{g['name']}.context.npy")
            n_pred = math.ceil(g["horizon"] / H.PATCH)
            assert n_pred <= n_hor, (g["name"], n_pred, n_hor)
            for b in range(ctxs.shape[0]):
                x, mask, _ = H.prepare_context(ctxs[b])
                assert x.size // H.PATCH <= n_ctx, (g["name"], b, x.size)
                patches, cos, sin, amask = H.bucket_inputs(x, mask, n_ctx, n_hor)
                mirror = patches.copy()
                mirror[..., : H.PATCH] = -mirror[..., : H.PATCH]
                both = np.concatenate([patches, mirror])
                for k, v in zip(rows, (both, cos, sin, amask)):
                    rows[k].append(v)
                windows.append(dict(id=f"{g['name']}/{b}", group=g["name"], index=b, context_length=int(ctxs.shape[1]),
                                    horizon=int(g["horizon"]), n_pred=n_pred, p=first_observed(mask, n_ctx),
                                    rows_md5=hashlib.md5(both.tobytes()).hexdigest()))
        arrays = {k: np.stack(v).astype(np.float32) for k, v in rows.items()}
        s = n_ctx + n_hor
        assert arrays["patches"].shape == (len(windows), 2, n_ctx, 64)
        assert arrays["cos"].shape == (len(windows), s, 64) and arrays["mask"].shape == (len(windows), 1, 1, s, s)
        check = None
        if ref_name:
            z = np.load(a.refs / ref_name)
            same = dict(
                windows=[str(v) for v in z["group"]] == [w["group"] for w in windows]
                and [int(v) for v in z["window"]] == [w["index"] for w in windows],
                patches=bool(np.array_equal(z["patches"], arrays["patches"])),
                cos=bool(np.array_equal(z["cos"], arrays["cos"])),
                sin=bool(np.array_equal(z["sin"], arrays["sin"])),
                mask=bool(np.array_equal(z["attn_mask"], arrays["mask"])),
                bucket=[int(v) for v in z["bucket"]] == [n_ctx, n_hor])
            check = dict(ref=str(a.refs / ref_name), **same)
            if not all(same.values()):
                raise SystemExit(f"{name}: the pack differs from {ref_name}: {same}")
        d = a.out / name
        d.mkdir(parents=True, exist_ok=True)
        files = {}
        for k, v in arrays.items():
            for dt, np_dt in (("f32", np.float32), ("f16", np.float16)):
                p = d / f"{k}.{dt}.bin"
                np.ascontiguousarray(v.astype(np_dt)).tofile(p)
                files[p.name] = dict(md5=md5(p), bytes=p.stat().st_size)
        if ref_name:
            p = d / "sanity_ref.f32.bin"
            np.ascontiguousarray(z["ref32"][0].astype(np.float32)).tofile(p)
            files[p.name] = dict(md5=md5(p), bytes=p.stat().st_size, shape=list(z["ref32"][0].shape),
                                 window=windows[0]["id"])
        doc = dict(schema="timestral-device-inputs/1", bucket=name, n_ctx=n_ctx, n_hor=n_hor, patch=H.PATCH, rows=2,
                   row_order=["series", "mirror"], quantiles=len(H.QUANTILES),
                   shapes=dict(patches=[2, n_ctx, 64], cos=[s, 64], sin=[s, 64], mask=[1, 1, s, s],
                               output=[2, n_hor * H.PATCH, len(H.QUANTILES)]),
                   positions="from the first observed patch (host_forecast.bucket_inputs default)",
                   windows=windows, files=files, ref_check=check,
                   source=dict(fixtures=str(a.fixtures), numpy=np.__version__))
        (d / "manifest.json").write_text(json.dumps(doc, indent=1) + "\n")
        with open(d / "MD5SUMS", "w") as f:
            for fn in sorted(list(files) + ["manifest.json"]):
                f.write(f"{md5(d / fn)}  {fn}\n")
        summary[name] = dict(windows=len(windows), groups=len({w['group'] for w in windows}),
                             bytes=sum(v["bytes"] for v in files.values()), ref_check=check)
        print(name, json.dumps(summary[name]))
    print("PACK_DONE")


if __name__ == "__main__":
    main()
