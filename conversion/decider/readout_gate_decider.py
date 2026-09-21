#!/usr/bin/env python3
"""Probability readout gate: a decider Core AI bundle vs the author's fp32 oracle.

Feeds every fixture row's token ids through the bundle one token per step (the decode-only
S=1 graph, fresh zero states per row, full-length position_ids), reads the fp16 logits at the
answer slot, restricts them to the row's option-letter ids, applies
`softmax(logits / 1.03)` and compares with `p_oracle` from `oracle_decider.py`'s file.

    python3 conversion/decider/readout_gate_decider.py exports/decider_0_8b_decode_int8hu_block32_sym \
        models/decider-0.8b/fixtures-decider-0.8b.json --transcript models/decider-0.8b/gate-decider-0.8b-readout.json

Needs the overlay interpreter (`coreai.runtime` + the `coreai_models` zoo overlay) and
`DEVELOPER_DIR` pointing at an Xcode whose Metal toolchain carries `coreai-build`.

The bundle is AOT-compiled first (`xcrun -f coreai-build compile ... --platform macOS
--preferred-compute gpu --architecture h16c --expect-frequent-reshapes`) and the `.aimodelc`
is loaded with `SpecializationOptions.default()`. That is not an optimization: on macOS 27.0
(26A428) the Python runtime's GPU JIT of this graph logged `MTL4CommandQueueErrorDomain
error 1` on every forward and returned all-zero logits, while the AOT asset is correct. The
runtime also leaks one IOSurface per call and dies after roughly 25,000 calls in one process
(`NDArray+SharedStorage.swift:108: Failed to allocate storage`); the 44 rows here are ~5,000
steps, so one process suffices — split longer fixture sets across processes.

PASS = letter argmax equals the oracle on every row (no near-tie exemption; the fixture set has
no oracle margin below 0.5), the full-vocabulary argmax is one of the row's labels on every
row, max |Δp| <= 0.02 and the mean of per-row mean |Δp| <= 0.002 (four times the fp16 floor
measured on the fp16 export of the same graph, 0.0050 / 0.00018), and the first row re-run at
the end reproduces its logits exactly (state reset proof).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

AOT_FLAGS = ["--platform", "macOS", "--preferred-compute", "gpu", "--architecture", "h16c",
             "--expect-frequent-reshapes"]


def aot_compile(aimodel: Path, out_dir: Path) -> tuple[Path, float]:
    """Compile once; a present `.aimodelc` is reused."""
    target = out_dir / f"{aimodel.stem}.h16c.aimodelc"
    if target.exists():
        return target, 0.0
    if not os.environ.get("DEVELOPER_DIR"):
        sys.exit("set DEVELOPER_DIR to the Xcode 27 RC (its Metal toolchain carries coreai-build)")
    cb = subprocess.run(["xcrun", "-f", "coreai-build"], capture_output=True, text=True)
    if cb.returncode != 0 or not cb.stdout.strip():
        sys.exit("xcrun -f coreai-build failed:\n" + cb.stderr)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    subprocess.run([cb.stdout.strip(), "compile", str(aimodel), "--output", str(out_dir), *AOT_FLAGS],
                   check=True)
    if not target.exists():
        sys.exit(f"coreai-build produced no {target}")
    return target, time.monotonic() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle", help="LanguageBundle directory (metadata.json + .aimodel + tokenizer/)")
    ap.add_argument("fixtures", help="oracle_decider.py output")
    ap.add_argument("--transcript", help="where to write the gate JSON")
    ap.add_argument("--aot-dir", help="where the .aimodelc goes (default: <bundle>/../aotc)")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--bar-max", type=float, default=0.02)
    ap.add_argument("--bar-mean", type=float, default=0.002)
    args = ap.parse_args()

    import torch
    import coreai.runtime as rt
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig
    from coreai_models.models.macos import qwen3_5 as q  # registers the qwen3_5_text config shim

    bundle = Path(args.bundle).resolve()
    meta = json.loads((bundle / "metadata.json").read_text())
    aimodel = bundle / meta["assets"]["main"]
    fx = json.loads(Path(args.fixtures).read_text())
    rows, temperature = fx["rows"], float(fx["temperature"])
    hf_id, revision = fx["source"]["hf_id"], fx["source"].get("revision")

    aimodelc, aot_seconds = aot_compile(aimodel, Path(args.aot_dir) if args.aot_dir else bundle.parent / "aotc")
    print(f"asset: {aimodelc} (compile {aot_seconds:.1f} s)", flush=True)

    raw = AutoConfig.from_pretrained(hf_hub_download(hf_id, "config.json", revision=revision).rsplit("/", 1)[0])
    cfg = q.qwen3_5_config_from_hf(getattr(raw, "text_config", raw), args.max_ctx, None)
    state_keys = ["k_cache", "v_cache", "conv_state", "rec_state"]

    def nd(a):
        return rt.NDArray(np.ascontiguousarray(a))

    record = {
        "schema": "coreai-decider-readout-gate/1",
        "bundle": str(bundle), "asset": str(aimodelc), "aot_seconds": aot_seconds,
        "fixtures": str(Path(args.fixtures)), "temperature": temperature, "max_ctx": args.max_ctx,
        "runtime": "coreai python runtime, AOT h16c GPU asset, SpecializationOptions.default()",
        "environment": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "rows": [],
    }

    async def run() -> None:
        model = await rt.AIModel.load(aimodelc, rt.SpecializationOptions.default())
        fn = model.load_function("main")

        async def row_logits(row: dict) -> np.ndarray:
            st = q.build_decode_state(cfg, max_seq_len=args.max_ctx, dtype=torch.float16)
            state = {n: nd(st[k].numpy()) for n, k in zip(q.DECODE_STATE_NAMES, state_keys)}
            out = None
            for t, tok in enumerate(row["ids"]):
                out = await fn(inputs={"input_ids": nd(np.array([[tok]], np.int32)),
                                       "position_ids": nd(np.arange(t + 1, dtype=np.int32)[None])},
                               state=state)
            logits = out["logits"].numpy()
            assert logits.shape[0] == 1 and logits.shape[1] == 1, logits.shape
            return logits[0, -1].astype(np.float32).copy()

        first = None
        for i, row in enumerate(rows):
            t0 = time.monotonic()
            logits = await row_logits(row)
            finite = bool(np.isfinite(logits).all())
            nonconstant = bool(logits.max() > logits.min())
            full_id = int(logits.argmax())
            if i == 0:
                first = logits.copy()
                if not (finite and nonconstant and full_id in row["label_ids"]):
                    record["failure"] = {"row": row["id"], "finite": finite, "nonconstant": nonconstant,
                                         "full_vocab_argmax_id": full_id,
                                         "hint": "all-zero or off-label logits on the first row = the JIT "
                                                 "signature; this gate loads the AOT asset, check the load path"}
                    break
            gathered = logits[row["label_ids"]]
            p = np.exp((gathered - gathered.max()) / temperature)
            p /= p.sum()
            delta = np.abs(p.astype(np.float64) - np.asarray(row["p_oracle"], dtype=np.float64))
            rec = {
                "id": row["id"], "tokens": row["tokens"], "nopts": row["nopts"],
                "letter_argmax": int(p.argmax()), "oracle_argmax": row["argmax"],
                "argmax_agrees": int(p.argmax()) == row["argmax"],
                "full_vocab_argmax_id": full_id, "full_vocab_argmax_is_label": full_id in row["label_ids"],
                "p": p.tolist(), "p_oracle": row["p_oracle"], "oracle_margin": row["top2_margin"],
                "max_abs_delta_p": float(delta.max()), "mean_abs_delta_p": float(delta.mean()),
                "finite": finite, "nonconstant": nonconstant, "seconds": time.monotonic() - t0,
            }
            record["rows"].append(rec)
            print(f"{row['id']:>22} {row['tokens']:>5} tok  argmax {'ok ' if rec['argmax_agrees'] else 'NO '}"
                  f" max|dp| {rec['max_abs_delta_p']:.4f}", flush=True)
        if first is not None and "failure" not in record:
            again = await row_logits(rows[0])
            record["reset_check"] = {"row": rows[0]["id"], "identical": bool(np.array_equal(first, again)),
                                     "max_abs_diff": float(np.max(np.abs(first - again)))}

    asyncio.run(run())

    done = record["rows"]
    s = {
        "rows": len(done), "expected_rows": len(rows),
        "argmax_agreement": sum(r["argmax_agrees"] for r in done),
        "full_vocab_argmax_is_label": sum(r["full_vocab_argmax_is_label"] for r in done),
        "max_abs_delta_p": max((r["max_abs_delta_p"] for r in done), default=None),
        "mean_of_row_mean_abs_delta_p": float(np.mean([r["mean_abs_delta_p"] for r in done])) if done else None,
        "finite_all": all(r["finite"] and r["nonconstant"] for r in done),
        "reset_identical": record.get("reset_check", {}).get("identical", False),
        "bar": {"max_abs_delta_p": args.bar_max, "mean_of_row_mean_abs_delta_p": args.bar_mean},
    }
    passed = ("failure" not in record and s["rows"] == s["expected_rows"]
              and s["argmax_agreement"] == s["rows"] and s["full_vocab_argmax_is_label"] == s["rows"]
              and s["finite_all"] and s["reset_identical"]
              and s["max_abs_delta_p"] <= args.bar_max and s["mean_of_row_mean_abs_delta_p"] <= args.bar_mean)
    record["summary"] = s
    record["result"] = "PASS" if passed else "FAIL"
    record["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line = (f"{record['result']}: argmax {s['argmax_agreement']}/{s['rows']}, "
            f"max|dp| {s['max_abs_delta_p']}, mean-of-row-means {s['mean_of_row_mean_abs_delta_p']}, "
            f"reset {'ok' if s['reset_identical'] else 'FAILED'}")
    print(line)
    if args.transcript:
        Path(args.transcript).parent.mkdir(parents=True, exist_ok=True)
        Path(args.transcript).write_text(json.dumps(record, indent=1) + "\n")
        print(f"  transcript: {args.transcript}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
