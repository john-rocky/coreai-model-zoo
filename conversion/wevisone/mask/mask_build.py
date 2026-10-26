#!/usr/bin/env python3
"""Export the mask decoder's graph A (per response) or graph B (per instance) to a Core AI `.aimodel`.

    <coreai venv>/bin/python mask_build.py a --grid 28 --precision fp32 --out-root <exports dir> --record <json>
    <coreai venv>/bin/python mask_build.py b --precision fp32 ...

`--precision`: fp32 = fp32 weights and compute (the reference form; the package runs the decoder in fp32) /
fp16w32 = weights stored fp16 and cast to fp32 in the graph (compute and I/O fp32) / fp16 = fp16 weights, compute
and I/O. Graph A takes `--grid` (28 for the 448 tower, 32 for 512) and `--n-max` query rows; graph B does not depend
on the grid. The weights are the checkpoint's `mask_decoder.pt` (strict load), the structure is mask_graph.py.
Name = `wv_mask_a_g<G>_n<N>_<precision>` / `wv_mask_b_<precision>`; the bundle dir is `<out-root>/<name>/`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mask_graph as MG  # noqa: E402

FILL = {"fp32": -1e30, "fp16w32": -1e30, "fp16": -65504.0}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fp16_storage_fp32_compute(module):
    """Every parameter stored as fp16, handed to the forward as an fp32 copy (r1's export_wevisone.py, from
    conversion/funasr_nano/export_encoder.py). Buffers stay fp32."""
    import torch.nn.utils.parametrize as P

    class Upcast(torch.nn.Module):
        def forward(self, w):
            return w.float()

    for m in list(module.modules()):
        for name, param in list(m.named_parameters(recurse=False)):
            setattr(m, name, torch.nn.Parameter(param.detach().half(), requires_grad=False))
            P.register_parametrization(m, name, Upcast(), unsafe=True)
    return module


def example_inputs(graph: str, g: int, n_max: int, dtype, host_fourier: bool = False) -> dict:
    gen = torch.Generator().manual_seed(0)
    if graph == "a":
        ins = {"image": torch.rand(1, 3, MG.CANVAS, MG.CANVAS, generator=gen)}
        for k in ("tap5", "tap11", "tap17", "tap23"):
            ins[k] = torch.randn(1, MG.VIT_DIM, g, g, generator=gen)
        ins["t_seg"] = torch.randn(n_max, MG.LLM_DIM, generator=gen)
        ins["boxes01"] = torch.tensor([[0.1, 0.2, 0.6, 0.9]]).expand(n_max, 4).clone()
        if host_fourier:
            ins["box_pe"] = torch.from_numpy(MG.box_fourier_np(ins["boxes01"].numpy()))
        ins["valid"] = torch.zeros(n_max)
        ins["valid"][0] = 1.0
    else:
        ins = {"image": torch.rand(1, 3, MG.CANVAS, MG.CANVAS, generator=gen),
               "semantic_s2": torch.randn(1, 48, MG.CANVAS // 2, MG.CANVAS // 2, generator=gen),
               "query": torch.randn(1, 192, generator=gen),
               "coarse_logits": torch.randn(1, MG.CANVAS // 2, MG.CANVAS // 2, generator=gen)}
    return {k: v.to(dtype) for k, v in ins.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("graph", choices=["a", "b"])
    ap.add_argument("--grid", type=int, default=28)
    ap.add_argument("--n-max", type=int, default=MG.N_MAX)
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16w32", "fp16"])
    ap.add_argument("--host-fourier", action="store_true",
                    help="graph A takes the box Fourier features as an input `box_pe` (name suffix _hf)")
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--record", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    import coreai.runtime as rt
    from coreai_models.export.macos import export_to_coreai

    t0 = time.monotonic()
    dec, cfg, _ = MG.load_package_decoder(args.pkg, args.sidecar)
    io_dtype = torch.float16 if args.precision == "fp16" else torch.float32
    if args.graph == "a":
        cls = MG.GraphAHF if args.host_fourier else MG.GraphA
        mod = cls(dec, args.grid, n_max=args.n_max, fill=FILL[args.precision]).eval()
        name = f"wv_mask_a_g{args.grid}_n{args.n_max}_{args.precision}" + ("_hf" if args.host_fourier else "")
        outs = ("queries", "coarse_logits", "semantic_s2", "memory")
    else:
        mod = MG.GraphB(dec).eval()
        name = f"wv_mask_b_{args.precision}"
        outs = ("mask_logits", "mask_logits_aux")
    if args.precision == "fp16":
        mod = mod.half()
        if args.graph == "a":
            # FourierPE1D's frequencies run to 2**31: fp16 would make them inf (the package casts the phase to fp32)
            pe = mod.query_builder.box_position.pe
            pe.frequencies = pe.frequencies.float()
    elif args.precision == "fp16w32":
        mod = fp16_storage_fp32_compute(mod)
    ins = example_inputs(args.graph, args.grid, args.n_max, io_dtype, host_fourier=args.host_fourier)
    names = tuple(ins)
    with torch.no_grad():
        eager = mod(**ins)
    rec = {"schema": "wevisone-r2-mask-build/1", "graph": args.graph, "name": name, "grid": args.grid,
           "n_max": args.n_max, "precision": args.precision, "fill": FILL[args.precision],
           "inputs": {k: [list(v.shape), str(v.dtype)] for k, v in ins.items()},
           "outputs": {k: [list(v.shape), str(v.dtype)] for k, v in zip(outs, eager)},
           "params": sum(p.numel() for p in mod.parameters()),
           "eager_finite": all(bool(torch.isfinite(v).all()) for v in eager)}
    t1 = time.monotonic()
    prog = export_to_coreai(mod, ins, dynamic_shapes=None, input_names=names, output_names=outs, state_names=(),
                            externalize_modules=[])
    rec["convert_seconds"] = time.monotonic() - t1
    t1 = time.monotonic()
    prog.optimize()
    rec["optimize_seconds"] = time.monotonic() - t1
    out_dir = Path(args.out_root) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    prog.save_asset(out_dir / f"{name}.aimodel", rt.AIModelAssetMetadata())
    files = sorted(p for p in out_dir.rglob("*") if p.is_file())
    rec["bundle"] = str(out_dir)
    rec["bytes"] = sum(p.stat().st_size for p in files)
    rec["files"] = {str(p.relative_to(out_dir)): p.stat().st_size for p in files}
    rec["wall_seconds"] = time.monotonic() - t0
    import coreai_torch

    rec["versions"] = {"torch": torch.__version__, "coreai_torch": getattr(coreai_torch, "__version__", None)}
    rec["script"] = {"mask_build.py": sha256_file(Path(__file__)), "mask_graph.py": sha256_file(HERE / "mask_graph.py")}
    rec["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    Path(args.record).write_text(json.dumps(rec, indent=1) + "\n")
    print(f"[build] {name}: {rec['bytes']:,} B, convert {rec['convert_seconds']:.1f}s optimize "
          f"{rec['optimize_seconds']:.1f}s, eager finite {rec['eager_finite']} -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
