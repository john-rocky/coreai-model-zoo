#!/usr/bin/env python3
"""Write the host files of the clef-flash head: the lm_head gather table (fp16 raw) and the head's own two files.

The joint schema head reads `lexical = lm_head.weight[option span ids].mean(0)` from the untied
lm_head ([248320, 4096], bf16 in model-00001-of-00004.safetensors). Neither the decoder graph (no
vocabulary head) nor the head graph holds that table, so the host gathers it:

    lm_head_fp16.bin    raw little-endian float16, row-major [248320, 4096], no header
                        (2,034,237,440 B); row t = lm_head.weight[t] rounded bf16 -> fp16
    lm_head_fp16.json   shape, dtype, bytes, sha256, the source (repo, revision, shard and its sha256,
                        the tensor's header entry), the rounding statistics, the read-back check
    joint_head.safetensors, joint_head_config.json
                        verbatim copies of the pinned snapshot's files (sha256 asserted)

bf16 -> fp16: a bf16 value inside fp16's normal range (|x| >= 2^-14) is exact in fp16 (8 significant
bits fit in 11); below it the value becomes an fp16 subnormal or 0. The statistics count both. Round 1
measured the same table at the head (`results/lm_head_table.json`, `lm_head_table_effect.json`: fp16
moves no probability by more than 1.8e-7 = the fp32 re-run floor); this script asserts its counts equal
round 1's.

Read-back check: rows 0..99, the last 100 rows and 1,000 seeded random rows are read again through a
second path (`safetensors.safe_open(framework="pt")`, torch bf16 -> fp16) and must equal the written
file bit for bit.

    cd conversion/clef_flash
    HF_HOME=$ZOO_WORK_ROOT/_clefflash/hf HF_HUB_OFFLINE=1 <venv>/bin/python export_lm_head_table.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _paths import hf_snapshot, record_path, work_path  # noqa: E402

os.environ.setdefault("HF_HOME", str(work_path("_clefflash", "hf")))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

HF_ID = "Cloudflare/clef-flash"
REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"
TENSOR = "lm_head.weight"
SHAPE = (248320, 4096)
FP16_NORMAL_MIN = 2.0 ** -14
HEAD_FILES = {  # sha256 of the pinned revision's files (joint_head.safetensors = its LFS oid)
    "joint_head.safetensors": "19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0",
    "joint_head_config.json": "77efe959a38b5b17b241543e129e695f3c77465ece55a25985279bd8176279a0",
}
SHARD_SHA256 = "8b45a8e968141cdcc58fb71c9adfc258e2c77b5f062bc636c1fd5bc5d916b565"   # model-00001-of-00004 LFS oid
ROUND1 = {"elements_below_fp16_normal_min": 3317193, "elements_flushed_to_zero": 1649}
ROWS_PER_CHUNK = 4096


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def header_entry(path: Path, name: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header[name], 8 + n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(work_path("_clefflash", "exports", "host")))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.monotonic()
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    snap = Path(hf_snapshot(HF_ID, revision=REVISION))
    index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    shard = snap / index[TENSOR]
    info, data_start = header_entry(shard, TENSOR)
    assert info["dtype"] == "BF16" and tuple(info["shape"]) == SHAPE, info
    begin, end = info["data_offsets"]
    V, D = SHAPE
    raw = np.memmap(shard, dtype="<u2", mode="r", offset=data_start + begin, shape=SHAPE)

    # convert + write + hash, in row chunks
    dst = out / "lm_head_fp16.bin"
    tmp = dst.with_name(dst.name + ".tmp")
    h = hashlib.sha256()
    stats = {"nonzero_elements": 0, "elements_below_fp16_normal_min": 0, "elements_flushed_to_zero": 0,
             "elements_fp16_subnormal": 0, "elements_inf": 0, "max_rel_error_nonzero": 0.0,
             "max_rel_error_not_flushed": 0.0, "max_rel_error_fp16_normal_range": 0.0, "max_abs_error": 0.0,
             "absmax": 0.0}
    with open(tmp, "wb") as f:
        for a in range(0, V, ROWS_PER_CHUNK):
            x = (np.asarray(raw[a:a + ROWS_PER_CHUNK]).astype(np.uint32) << 16).view(np.float32)   # exact bf16 values
            y = x.astype(np.float16)                                                             # round to nearest even
            yb = y.astype("<f2").tobytes()
            f.write(yb)
            h.update(yb)
            ax = np.abs(x)
            nz = ax > 0
            yf = y.astype(np.float32)
            err = np.abs(yf - x)
            rel = np.zeros_like(err)
            rel[nz] = err[nz] / ax[nz]
            below = nz & (ax < FP16_NORMAL_MIN)
            flushed = nz & (yf == 0)
            stats["nonzero_elements"] += int(nz.sum())
            stats["elements_below_fp16_normal_min"] += int(below.sum())
            stats["elements_flushed_to_zero"] += int(flushed.sum())
            stats["elements_fp16_subnormal"] += int(((np.abs(yf) > 0) & (np.abs(yf) < FP16_NORMAL_MIN)).sum())
            stats["elements_inf"] += int(np.isinf(yf).sum())
            stats["max_rel_error_nonzero"] = max(stats["max_rel_error_nonzero"], float(rel.max()))
            keep = nz & ~flushed
            if keep.any():
                stats["max_rel_error_not_flushed"] = max(stats["max_rel_error_not_flushed"], float(rel[keep].max()))
            normal = ax >= FP16_NORMAL_MIN
            if normal.any():
                stats["max_rel_error_fp16_normal_range"] = max(stats["max_rel_error_fp16_normal_range"],
                                                               float(rel[normal].max()))
            stats["max_abs_error"] = max(stats["max_abs_error"], float(err.max()))
            stats["absmax"] = max(stats["absmax"], float(ax.max()))
    os.replace(tmp, dst)
    t_write = time.monotonic()
    nbytes = dst.stat().st_size
    assert nbytes == V * D * 2 == 2_034_237_440, nbytes
    digest = h.hexdigest()
    assert sha256_file(dst) == digest, "the file on disk does not hash to the bytes written"

    # read-back through a second path: safetensors (torch) bf16 -> fp16 == the file, bit for bit
    import torch
    from safetensors import safe_open

    rng = np.random.default_rng(args.seed)
    rows = np.unique(np.concatenate([np.arange(100), np.arange(V - 100, V),
                                     rng.choice(np.arange(100, V - 100), size=1000, replace=False)]))
    table = np.memmap(dst, dtype="<f2", mode="r", shape=SHAPE)
    with safe_open(str(shard), framework="pt", device="cpu") as sf:
        sl = sf.get_slice(TENSOR)
        mism = 0
        for r in rows:
            ref = sl[int(r):int(r) + 1].to(torch.float16).numpy().view(np.uint16)
            got = np.asarray(table[int(r)]).view(np.uint16)
            mism += int((ref[0] != got).sum())
    readback = {"rows_checked": int(rows.size), "first_rows": 100, "last_rows": 100, "random_rows": 1000,
                "seed": args.seed, "path": "safetensors.safe_open(framework='pt') get_slice -> torch bf16 -> float16",
                "mismatched_elements": mism, "bit_equal": mism == 0}
    assert mism == 0, readback
    round1 = {k: stats[k] for k in ROUND1}
    assert round1 == ROUND1, (round1, ROUND1)

    # the head's own files, verbatim
    copies = {}
    for name, want in HEAD_FILES.items():
        src = snap / name
        got = sha256_file(src)
        assert got == want, (name, got, want)
        shutil.copyfile(src, out / name)
        assert sha256_file(out / name) == want
        copies[name] = {"bytes": (out / name).stat().st_size, "sha256": want, "source": record_path(src.resolve())}

    shard_sha = sha256_file(shard)
    assert shard_sha == SHARD_SHA256, shard_sha
    rec = {
        "file": dst.name, "shape": list(SHAPE), "dtype": "float16", "byte_order": "little-endian",
        "layout": "row-major, no header: row t at byte offset t * 8192", "bytes": nbytes, "sha256": digest,
        "row_t": "lm_head.weight[t] (the untied output embedding) rounded bf16 -> fp16, round to nearest even",
        "use": "lexical[o] = mean over option o's span ids of the rows, accumulated in f32 or wider "
               "(clef_head.lexical_rows: f64); the head graph's `lexical` input",
        "source": {"hf_id": HF_ID, "revision": REVISION, "shard": index[TENSOR], "shard_sha256": shard_sha,
                   "tensor": TENSOR, "header": info, "data_start": data_start},
        "rounding": {**stats, "fp16_normal_min": FP16_NORMAL_MIN,
                     "note": "bf16 values with |x| >= 2^-14 are exact in fp16; below that they become fp16 "
                             "subnormals (rounded) or 0; max_rel_error_nonzero = 1.0 comes from the flushed ones"},
        "round1_check": {"file": "results/lm_head_table.json", "expected": ROUND1, "got": round1, "equal": True,
                         "effect_at_the_head": "results/lm_head_table_effect.json: fp16 table max |dp| 1.8e-7 "
                                               "(= the fp32 re-run floor), argmax 405/405"},
        "readback": readback,
        "head_files": copies,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "seconds": {"convert_write": t_write - t0, "total": time.monotonic() - t0},
        "written": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (out / "lm_head_fp16.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("bytes", "sha256")}), json.dumps(stats), json.dumps(readback))
    print(f"wrote {dst} and {out / 'lm_head_fp16.json'}; head files {list(copies)} ({rec['seconds']['total']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
