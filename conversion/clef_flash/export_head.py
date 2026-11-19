#!/usr/bin/env python3
"""Export the clef-flash joint schema head (`clef_head.ClefHeadGraph`) to a Core AI asset and AOT-compile it.

The graph is the author's `JointSchemaHead` (joint_head.safetensors loaded strictly) computed for one
record from tensor inputs (see clef_head.py for the inputs and the math). Weight forms:

    fp16w32  every parameter stored fp16 and read through a .float() cast, all math fp32
             (export_qwen38vl_pipelined.fp16_storage_fp32_compute, the tower's ship form); the
             checkpoint's values are bf16, and a bf16 value in fp16's normal range is exact in fp16
    fp32     parameters and math fp32 (twice the bytes; built only if fp16w32 misses its bar)

Shapes:

    dynamic  one function `main`: T in [64, 4096], Q in [q_min, 16], O in [2, 128] (torch.export Dims);
             the host pads T to a multiple of 64 (the decoder's padded rows, masked by key_valid) and
             Q to at least q_min; AOT with --expect-frequent-reshapes
    bucket   one asset, four functions t512 / t1024 / t2048 / t4096 sharing the weights: T fixed per
             function, Q = 16, O = 128 (the host pads every input and masks)

The bundle `<out-dir>/<name>/` holds `<name>.aimodel` and `metadata.json` (kind `decision-head`: the
inputs, the function table, the host padding rule, the source files and their sha256, the aten op set,
the export / AOT record). `--aot` compiles it for the Mac GPU into `<out-dir>/<name>/<name>.h16c.aimodelc`
(coreai-build compile --platform macOS --preferred-compute gpu --architecture h16c, plus
--expect-frequent-reshapes for the dynamic shape). The gates are gate_head.py h1 / h2.

    cd conversion/clef_flash
    HF_HOME=$ZOO_WORK_ROOT/_clefflash/hf HF_HUB_OFFLINE=1 \\
      DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer \\
      <coreai-models venv>/bin/python export_head.py --shape dynamic --weights fp16w32 --aot --record <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _paths import hf_snapshot, record_argv, work_path  # noqa: E402

os.environ.setdefault("HF_HOME", str(work_path("_clefflash", "hf")))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

T_MIN, T_MAX, Q_MAX, O_MIN, O_MAX = 64, 4096, 16, 2, 128
BUCKETS = (512, 1024, 2048, 4096)
EXAMPLE = ("own_t04", "text")          # the trace example: a fixture record of 493 tokens, 4 questions, 11 options
AOT_BASE = ["--platform", "macOS", "--preferred-compute", "gpu", "--architecture", "h16c"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(path: Path) -> dict:
    files = sorted(p for p in path.rglob("*") if p.is_file())
    per = {str(p.relative_to(path)): sha256_file(p) for p in files}
    tree = hashlib.sha256("".join(f"{k}\0{v}\n" for k, v in per.items()).encode()).hexdigest()
    return {"bytes": sum(p.stat().st_size for p in files), "tree_sha256": tree, "files": per}


def example_inputs(t_pad: int, q_pad: int | None, o_pad: int | None):
    """The trace example from the fixture oracle (its fp32 hidden, the fp16 table's lexical)."""
    from clef_head import head_inputs
    lane = work_path("_clefflash")
    rows = json.loads((lane / "oracle" / "records_oracle.json").read_text())["rows"]
    row = next(r for r in rows if (r["id"], r["arm"]) == EXAMPLE)
    z = np.load(lane / "oracle" / "npz" / f"{EXAMPLE[0]}__{EXAMPLE[1]}.npz")
    table = np.memmap(lane / "exports" / "host" / "lm_head_fp16.bin", dtype="<f2", mode="r", shape=(248320, 4096))
    arrays, layout = head_inputs(z["last_hidden"], row["ids"], row["questions"], table, t_pad=t_pad, q_pad=q_pad, o_pad=o_pad)
    return arrays, layout, row


def build_graph(weights: str):
    import torch
    from clef_head import ClefHeadGraph, load_author_head

    head = load_author_head(torch.float32)
    if weights == "fp16w32":
        from export_qwen38vl_pipelined import fp16_storage_fp32_compute
        head = fp16_storage_fp32_compute(head)
    elif weights != "fp32":
        raise ValueError(weights)
    return ClefHeadGraph(head).eval()


def op_census(graph, kwargs: dict, dynamic_shapes) -> dict:
    """aten ops of the traced program, before and after coreai-torch's decomposition table."""
    import coreai_torch
    import torch

    with torch.no_grad():
        ep = torch.export.export(graph, args=(), kwargs=kwargs, dynamic_shapes=dynamic_shapes)
    before = Counter(str(n.target) for n in ep.graph.nodes if n.op == "call_function")
    dec = ep.run_decompositions(coreai_torch.get_decomp_table())
    after = Counter(str(n.target) for n in dec.graph.nodes if n.op == "call_function")
    with torch.no_grad():
        got = ep.module()(**kwargs)
        ref = graph(**kwargs)
    return {"aten_ops": dict(sorted(before.items())), "after_coreai_decomposition": dict(sorted(after.items())),
            "data_dependent_ops": sorted(k for k in before if k in ("aten.item.default", "aten._local_scalar_dense.default",
                                                                    "aten.nonzero.default")),
            "exported_vs_eager_max_abs": float((got - ref).abs().max()),
            "range_constraints": {str(k): str(v) for k, v in ep.range_constraints.items()}}


def export(args, out_dir: Path, name: str) -> dict:
    import torch
    from clef_head import INPUT_NAMES, OUTPUT_NAMES
    from coreai_models.export.macos import export_to_coreai, export_to_coreai_multifunction

    t0 = time.monotonic()
    graph = build_graph(args.weights)
    t_built = time.monotonic()
    rec: dict = {"weights": args.weights, "shape": args.shape}
    if args.shape == "dynamic":
        from torch.export import Dim
        T = Dim("T", min=T_MIN, max=T_MAX)
        Q = Dim("Q", min=args.q_min, max=Q_MAX)
        O = Dim("O", min=O_MIN, max=O_MAX)
        dyn = {"hidden": {0: T}, "key_valid": {0: T}, "q_avg": {0: Q, 1: T}, "o_avg": {0: O, 1: T}, "g_avg": {1: T},
               "lexical": {0: O}, "member": {0: Q, 1: O}, "type_ids": {0: Q}, "q_valid": {0: Q}, "o_valid": {0: O}}
        arrays, layout, row = example_inputs(512, None, None)
        kwargs = {n: torch.from_numpy(arrays[n]) for n in INPUT_NAMES}
        rec["dims"] = {"T": [T_MIN, T_MAX], "Q": [args.q_min, Q_MAX], "O": [O_MIN, O_MAX]}
        rec["example"] = {"run": "/".join(EXAMPLE), "T": 512, "Q": int(arrays["member"].shape[0]),
                          "O": int(arrays["member"].shape[1])}
        rec["op_census"] = op_census(graph, kwargs, dyn)
        attempts = []
        prog = None
        for label, ext in (("default externalize specs", None), ("externalize off", [])):
            t1 = time.monotonic()
            try:
                prog = export_to_coreai(graph, kwargs, dynamic_shapes=dyn, input_names=INPUT_NAMES,
                                        output_names=OUTPUT_NAMES, state_names=(), externalize_modules=ext)
                attempts.append({"externalize": label, "ok": True, "seconds": time.monotonic() - t1})
                break
            except Exception as e:  # noqa: BLE001 — record the first error line, then try the next form
                lines = [x for x in traceback.format_exception_only(type(e), e)[0].splitlines() if x.strip()]
                attempts.append({"externalize": label, "ok": False, "seconds": time.monotonic() - t1,
                                 "error_type": type(e).__name__, "first_error_line": lines[0][:500],
                                 "error_head": "\n".join(lines[:8])[:2500]})
                print(f"export ({label}) failed: {lines[0][:300]}", flush=True)
        rec["export_attempts"] = attempts
        if prog is None:
            raise SystemExit("dynamic export failed in both externalize forms (see the record)")
        functions = ["main"]
    else:
        entries = []
        census = {}
        for tb in args.buckets:
            arrays, layout, row = example_inputs(tb, Q_MAX, O_MAX)
            kwargs = {n: torch.from_numpy(arrays[n]) for n in INPUT_NAMES}
            entries.append((f"t{tb}", {"reference_inputs": kwargs, "dynamic_shapes": None, "input_names": INPUT_NAMES,
                                       "output_names": OUTPUT_NAMES, "state_names": ()}))
            if tb == args.buckets[0]:
                census = op_census(graph, kwargs, None)
        rec["buckets"] = [[f"t{tb}", tb] for tb in args.buckets]
        rec["op_census"] = census
        t1 = time.monotonic()
        attempts = []
        prog = None
        for label, ext in (("default externalize specs", None), ("externalize off", [])):
            try:
                prog = export_to_coreai_multifunction(graph, entries, externalize_modules=ext)
                attempts.append({"externalize": label, "ok": True, "seconds": time.monotonic() - t1})
                break
            except Exception as e:  # noqa: BLE001
                lines = [x for x in traceback.format_exception_only(type(e), e)[0].splitlines() if x.strip()]
                attempts.append({"externalize": label, "ok": False, "error_type": type(e).__name__,
                                 "first_error_line": lines[0][:500], "error_head": "\n".join(lines[:8])[:2500]})
                print(f"export ({label}) failed: {lines[0][:300]}", flush=True)
        rec["export_attempts"] = attempts
        if prog is None:
            raise SystemExit("bucket export failed in both externalize forms (see the record)")
        functions = [f"t{tb}" for tb in args.buckets]
    t_conv = time.monotonic()
    prog.optimize()
    t_opt = time.monotonic()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    import coreai.runtime as rt
    aimodel = out_dir / f"{name}.aimodel"
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    t_saved = time.monotonic()
    files = {str(p.relative_to(aimodel)): p.stat().st_size for p in sorted(aimodel.rglob("*")) if p.is_file()}
    mlirb = aimodel / "main.mlirb"
    rec.update({"functions": functions, "aimodel": str(aimodel), "aimodel_files": files,
                "aimodel_bytes": sum(files.values()),
                "main_mlirb": {"bytes": mlirb.stat().st_size, "sha256": sha256_file(mlirb)} if mlirb.exists() else None,
                "seconds": {"build": t_built - t0, "export": t_conv - t_built, "optimize": t_opt - t_conv,
                            "save": t_saved - t_opt, "total": t_saved - t0}})
    print(f"saved {aimodel} ({rec['aimodel_bytes']:,} B) in {rec['seconds']['total']:.0f}s", flush=True)
    return rec


def aot_compile(aimodel: Path, out_dir: Path, efr: bool) -> dict:
    if not os.environ.get("DEVELOPER_DIR"):
        sys.exit("set DEVELOPER_DIR to the Xcode 27 RC (its Metal toolchain carries coreai-build)")
    cb = subprocess.run(["xcrun", "-f", "coreai-build"], capture_output=True, text=True)
    if cb.returncode != 0 or not cb.stdout.strip():
        sys.exit("xcrun -f coreai-build failed:\n" + cb.stderr)
    target = out_dir / f"{aimodel.stem}.h16c.aimodelc"
    if target.exists():
        shutil.rmtree(target)
    flags = AOT_BASE + (["--expect-frequent-reshapes"] if efr else [])
    cmd = [cb.stdout.strip(), "compile", str(aimodel), "--output", str(out_dir), *flags]
    print(" ".join(cmd), flush=True)
    t0 = time.monotonic()
    subprocess.run(cmd, check=True)
    secs = time.monotonic() - t0
    if not target.exists():
        sys.exit(f"coreai-build produced no {target}")
    return {"aimodelc": str(target), "flags": flags, "seconds": secs, "coreai_build": cb.stdout.strip(),
            "command": cmd, "digest": tree_digest(target)}


def write_metadata(out_dir: Path, name: str, rec: dict, args) -> None:
    from clef_head import INPUT_NAMES, JSM_SHA256, REVISION
    snap = Path(hf_snapshot("Cloudflare/clef-flash", revision=REVISION))
    dyn = rec["shape"] == "dynamic"
    meta = {
        "metadata_version": "0.2", "kind": "decision-head", "name": name, "assets": {"main": f"{name}.aimodel"},
        "head": {
            "graph": "conversion/clef_flash/clef_head.py ClefHeadGraph: the author's JointSchemaHead for one record, "
                     "tensor inputs only",
            "shape": rec["shape"], "functions": rec["functions"], "buckets": rec.get("buckets"),
            "dims": rec.get("dims"),
            "host_pad": "host" if dyn else "bucket",
            "host_pad_rule": (f"T to a multiple of 64 (rows past the real tokens zero, key_valid 0), Q to at least "
                              f"{args.q_min} (padding questions: zero rows, q_valid 0), O as is" if dyn else
                              f"T to the smallest bucket >= T ({', '.join(map(str, args.buckets))}), Q to {Q_MAX}, O to "
                              f"{O_MAX}; padding rows / columns zero, their valid flags 0"),
            "inputs": {"hidden": "[T, 4096] f32", "key_valid": "[T] f32", "q_avg": "[Q, T] f32", "o_avg": "[O, T] f32",
                       "g_avg": "[1, T] f32", "lexical": "[O, 4096] f32", "member": "[Q, O] f32",
                       "type_ids": "[Q] i32", "q_valid": "[Q] f32", "o_valid": "[O] f32"},
            "input_order": list(INPUT_NAMES),
            "outputs": {"logits": "[O] f32; padding options 0"},
            "weights": rec["weights"],
            "softmax": "host: per question, fp32, over that question's options",
            "lexical": "host: lm_head_fp16.bin rows of each option span's ids, averaged (clef_head.lexical_rows)",
            "source": {"hf_id": "Cloudflare/clef-flash", "revision": REVISION,
                       "joint_schema_model_sha256": JSM_SHA256,
                       "joint_head_safetensors_sha256": sha256_file(snap / "joint_head.safetensors"),
                       "joint_head_config_sha256": sha256_file(snap / "joint_head_config.json")},
        },
        "export": {k: rec[k] for k in ("aimodel_bytes", "main_mlirb", "seconds", "export_attempts") if k in rec},
        "aot": rec.get("aot") and {k: rec["aot"][k] for k in ("aimodelc", "flags", "seconds")},
        "compilation": {"date": datetime.now().astimezone().isoformat(timespec="seconds"), "targets": ["h16c"] if rec.get("aot") else []},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shape", choices=["dynamic", "bucket"], default="dynamic")
    ap.add_argument("--weights", choices=["fp16w32", "fp32"], default="fp16w32")
    ap.add_argument("--q-min", type=int, default=1, help="dynamic: the smallest Q of the Dim (the host pads Q to it)")
    ap.add_argument("--buckets", type=lambda x: [int(v) for v in x.split(",")], default=list(BUCKETS),
                    help="bucket: the T of each function, ascending (default %(default)s)")
    ap.add_argument("--out-dir", default=str(work_path("_clefflash", "exports", "head")))
    ap.add_argument("--name", help="default clef_flash_head_<shape>_<weights>")
    ap.add_argument("--aot", action="store_true")
    ap.add_argument("--skip-export", action="store_true", help="reuse the saved .aimodel (AOT only)")
    ap.add_argument("--record", help="write the export / AOT record JSON here")
    args = ap.parse_args()
    name = args.name or f"clef_flash_head_{args.shape}_{args.weights}"
    out_dir = Path(args.out_dir) / name
    record: dict = {"name": name, "argv": record_argv(sys.argv[1:]), "pid": os.getpid(),
                    "started": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "script_sha256": sha256_file(Path(__file__).resolve()),
                    "clef_head_sha256": sha256_file(HERE / "clef_head.py")}

    def save_record() -> None:
        if args.record:
            Path(args.record).parent.mkdir(parents=True, exist_ok=True)
            Path(args.record).write_text(json.dumps(record, indent=1) + "\n")

    if args.skip_export:
        record.update(json.loads((out_dir / "metadata.json").read_text()).get("export_record", {}))
        rec = json.loads(Path(args.record).read_text())["export"] if args.record and Path(args.record).exists() else None
        if rec is None:
            sys.exit("--skip-export needs the earlier --record")
    else:
        rec = export(args, out_dir, name)
    record["export"] = rec
    save_record()
    if args.aot:
        rec["aot"] = aot_compile(out_dir / f"{name}.aimodel", out_dir, efr=args.shape == "dynamic")
        print(f"AOT {rec['aot']['aimodelc']} ({rec['aot']['digest']['bytes']:,} B, {rec['aot']['seconds']:.1f} s)", flush=True)
    write_metadata(out_dir, name, rec, args)
    record["metadata_sha256"] = sha256_file(out_dir / "metadata.json")
    record["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
    save_record()
    print(f"bundle: {out_dir}")


if __name__ == "__main__":
    main()
