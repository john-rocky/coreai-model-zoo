#!/usr/bin/env python3
"""Index the oracle runs (`<oracle>/<form>/<id>.json`) into one `oracle.json`: environment, revision, threads, and
per run the prompt, the answer, its parse (suite.parse), the length, the loop flag and the seconds.

    python3 oracle_index.py --oracle <dir> --fixtures <json> --out <oracle.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from suite import parse  # noqa: E402

REVISION = "3dad9ce8bcb6fbe0aea945bf8860bd2cfc3663bf"
PACKAGE_COMMIT = "40f735b8daed437ece982db9718e9c9856f52a79"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    o = Path(args.oracle)
    fx = Path(args.fixtures)
    rows = [r["id"] for r in json.loads(fx.read_text())["rows"]]
    forms = {}
    envs = {}
    for d in sorted(p for p in o.iterdir() if p.is_dir() and not p.name.startswith("_")):
        runs = {}
        for rid in rows:
            f = d / f"{rid}.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            env = r.get("environment", {})
            envs.setdefault(d.name, {k: env.get(k) for k in ("versions", "python", "platform", "macos_build", "chip",
                                                               "torch_num_threads", "torch_num_interop_threads", "env",
                                                               "model_dir", "fixtures_sha256", "engine_py_sha256",
                                                               "model_dtypes")})
            runs[rid] = {"task": r["task"], "query": r["query"], "prompt": r["prompt"], "chat_text": r["chat_text"],
                         "prompt_len": r["prompt_len"], "image_size_wh": r["image_size_wh"],
                         "image_grid_thw": r["image_grid_thw"], "repetition_penalty": r["repetition_penalty"],
                         "max_new_tokens": r["max_new_tokens"], "dtype": r["dtype"], "gen_len": r["gen_len"],
                         "hit_max": r["hit_max"], "response": r["response"], "raw_text": r["raw_text"],
                         "parse": parse(r["response"]), "seconds": r["seconds"],
                         "script_sha256": env.get("script_sha256"),
                         "min_gap_softmax_first16": r.get("min_gap_softmax_first16"),
                         "dense": ({k: r["dense"][k] for k in ("n_boxes", "retokenize", "obj_hidden_generation_vs_teacher",
                                                              "count_decoded", "grid_hw_premerge", "alpha_size_hw")
                                    if k in r["dense"]} if r.get("dense") else None)}
        forms[d.name] = {"rows": len(runs), "missing": [x for x in rows if x not in runs],
                         "hit_max": [k for k, v in runs.items() if v["hit_max"]],
                         "wall_seconds_sum": sum(v["seconds"].get("total", 0) for v in runs.values()), "runs": runs}
    out = {"schema": "wevisone-oracle-index/1", "hf_id": "tencent/WeVisOne-VL-2B", "revision": REVISION,
           "package": {"repo": "https://github.com/Tencent/WeVisOne-VL", "commit": PACKAGE_COMMIT},
           "fixtures": {"path": str(fx), "sha256": hashlib.sha256(fx.read_bytes()).hexdigest()},
           "environment_per_form": envs,
           "forms": {k: {kk: vv for kk, vv in v.items() if kk != "runs"} for k, v in forms.items()},
           "runs": {k: v["runs"] for k, v in forms.items()}}
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(out["forms"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
