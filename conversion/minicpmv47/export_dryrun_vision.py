#!/usr/bin/env python3
"""torch.export dry run of the baked MiniCPM-V-4.7 vision tower (no Core AI conversion or runtime call).

`VisionExport(MiniCPMV47Vision.bake_constants(32))`: pixel_values [1, 3, 448, 448] -> image_features
[64, 1024], fp32, fixed shape. Exported with the call `coreai_models.export.macos.export_to_coreai`'s
`export_fn` makes (`torch.export.export(module, args=(), kwargs=..., dynamic_shapes=None)` under
no_grad), then decomposed twice for the record: torch's default core-ATen table, and the
`coreai_torch.get_decomp_table()` table that `export_fn` applies before its converter (a torch FX
pass; the converter itself is not run here).

  ops      every call_function target in each graph, counted -> `vision_export_ops.txt`
  banned   sort / argsort / bucketize / nonzero / unique / index_put / while_loop / scan: 0 in every
           graph (asserted; the run fails otherwise)
  run      each exported program's module on the P1 pixels vs the eager baked module (bitwise / max |d|)

    QW coreai-models/.venv/bin/python -I conversion/minicpmv47/export_dryrun_vision.py
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _paths import hf_snapshot, work_path  # noqa: E402

HF_ID = "openbmb/MiniCPM-V-4.7-1B"
REVISION = "984a89aafcd5160164f4f0d04b4be0f4eec03cb8"
BANNED = ("sort", "bucketize", "nonzero", "unique", "index_put", "while_loop", "scan")   # "sort" covers argsort


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def op_name(target) -> str:
    name = getattr(target, "name", None)
    if callable(name):                      # OpOverload: "aten::addmm" + overload
        return f"{target.name()}"
    return getattr(target, "__name__", str(target))


def count_ops(ep) -> Counter:
    return Counter(op_name(n.target) for n in ep.graph.nodes if n.op == "call_function")


def banned_in(ops: Counter) -> dict:
    hits = {}
    for op, n in ops.items():
        base = op.split("::")[-1].split(".")[0].lower()
        if any(b in base for b in BANNED):
            hits[op] = n
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--oracle-dir", default=str(work_path("_minicpmv47", "oracle")))
    ap.add_argument("--out", default=str(work_path("_minicpmv47", "oracle", "vision_export_ops.txt")))
    args = ap.parse_args()

    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    from minicpmv47_vision import GRID, MiniCPMV47Vision, VisionExport, unpack_patches

    t_all = time.perf_counter()
    core = MiniCPMV47Vision.from_checkpoint(hf_snapshot(HF_ID, revision=REVISION), torch.float32)
    core.bake_constants(GRID)
    model = VisionExport(core).eval()
    z = np.load(Path(args.oracle_dir) / "oracle.npz")
    pv = unpack_patches(torch.from_numpy(z["P1__pixel_values"]), GRID, GRID)
    with torch.inference_mode():
        eager = model(pv)

    t0 = time.perf_counter()
    with torch.no_grad():
        ep = torch.export.export(model, args=(), kwargs={"pixel_values": pv}, dynamic_shapes=None)
    export_secs = time.perf_counter() - t0
    graphs = {"export": ep}
    t0 = time.perf_counter()
    graphs["core_aten"] = ep.run_decompositions()
    import coreai_torch

    graphs["coreai_decomp"] = ep.run_decompositions(coreai_torch.get_decomp_table())
    decomp_secs = time.perf_counter() - t0

    sig = ep.graph_signature
    lines_sig = [f"inputs: user {len(sig.user_inputs)}, parameters {len(sig.parameters)}, buffers {len(sig.buffers)} "
                 f"(non-persistent {len(sig.non_persistent_buffers)}), lifted constants "
                 f"{len(getattr(sig, 'lifted_tensor_constants', []))}; outputs {len(sig.user_outputs)}"]
    res, ops, hits = {}, {}, {}
    for name, g in graphs.items():
        ops[name] = count_ops(g)
        hits[name] = banned_in(ops[name])
        with torch.inference_mode():
            out = g.module()(pixel_values=pv)
        res[name] = (tuple(out.shape), bool(torch.equal(out, eager)), float((out - eager).abs().max()))
        print(f"[{name}] {sum(ops[name].values())} call_function nodes, {len(ops[name])} distinct ops, banned "
              f"{hits[name] or 0}, run vs eager: shape {res[name][0]} bitwise {res[name][1]} max|d| {res[name][2]:.3e}",
              flush=True)

    buffers = {n: (tuple(b.shape), str(b.dtype)) for n, b in core.named_buffers()}
    head = [
        f"# MiniCPM-V-4.7 vision tower — torch.export dry run (r2), {datetime.now().strftime('%Y-%m-%d %H:%M')} JST",
        f"# module conversion/minicpmv47/minicpmv47_vision.py sha256 {sha256(HERE / 'minicpmv47_vision.py')}",
        f"# script conversion/minicpmv47/export_dryrun_vision.py sha256 {sha256(Path(__file__))}",
        f"# torch {torch.__version__}, coreai-torch {getattr(coreai_torch, '__version__', '?')}, threads {torch.get_num_threads()}",
        "# wrapper VisionExport(core.bake_constants(32)): pixel_values [1, 3, 448, 448] float32 -> image_features "
        f"{list(eager.shape)} {str(eager.dtype).replace('torch.', '')}",
        "# call: torch.export.export(model, args=(), kwargs={'pixel_values': P1 pixels}, dynamic_shapes=None) under "
        f"no_grad (= export_to_coreai's export_fn); {export_secs:.1f}s; decompositions {decomp_secs:.1f}s",
        "# " + lines_sig[0],
        f"# baked buffers: {buffers}",
        "# graphs: export = torch.export output (what export_fn decomposes); core_aten = ep.run_decompositions() "
        "(torch default); coreai_decomp = ep.run_decompositions(coreai_torch.get_decomp_table()) (what the converter "
        "would receive; the converter was not run)",
        *[f"# run {n}: output {res[n][0]}, bitwise == eager {res[n][1]}, max|d| {res[n][2]:.3e}" for n in graphs],
        f"# banned ops ({' / '.join(BANNED)}; 'sort' covers argsort): "
        + ", ".join(f"{n} {sum(hits[n].values())}" for n in graphs)
        + (" -> PASS" if not any(hits.values()) else f" -> FAIL {hits}"),
    ]
    body = []
    for n in graphs:
        body.append("")
        body.append(f"## {n}: {sum(ops[n].values())} call_function nodes, {len(ops[n])} distinct ops")
        body.append("count  op")
        body += [f"{c:5d}  {op}" for op, c in sorted(ops[n].items(), key=lambda kv: (-kv[1], kv[0]))]
    Path(args.out).write_text("\n".join(head + body) + "\n")
    print("\n".join(head), flush=True)
    print(f"[export_dryrun_vision] -> {args.out} (wall {time.perf_counter() - t_all:.1f}s)", flush=True)
    assert not any(hits.values()), hits
    assert all(r[0] == (64, 1024) for r in res.values())


if __name__ == "__main__":
    main()
