"""Export the Qwen-Image-2.1 DiT with int8 weight-only Linears ("int8lin", round 4).

Same graph as ``export_dit.py`` — bf16 compute, fp32 boundary, RoPE in fp32, text axis
``Dim("ntxt", 8..512)`` and image axis ``Dim("nimg", 64..4096)`` dynamic — with every ``nn.Linear``
weight (232 tensors: img_in, txt_in x2, the timestep MLP x2, the shared modulation, 7 per block x 32,
norm_out.linear, proj_out) stored int8 by ``coreai_models.export.compression.quantize_pytorch_model``
in the zimage ``linear_quant_config`` recipe: eager, ``symmetric_with_clipping`` (-127..127),
``per_block`` 32 along axis 1 (the input features), bf16 scales (the weight dtype), dequantized to
bf16 in the graph. Norms stay bf16 (``qi21_dit.RMSNorm`` / ``ZeroCenterRMSNorm`` excluded;
the LayerNorms have no weight). The census after quantization asserts that exactly the Linear
weights were quantized.

``--torch-ref``: the quantized module, cast to fp32 (int8 values unchanged, scales upcast exactly,
so: int8 weights + fp32 compute), teacher-forced on the oracle at ``--ref-steps`` on the CPU,
velocity vs the oracle's — what the int8 weights cost without the bf16 engine's own error.

Bundle: ``_paths.exports_dir()/qwenimage21/qi21_dit_full_int8lin_dyn_iofp32/``; ``--aot`` =
``xcrun coreai-build compile ... h16c gpu --expect-frequent-reshapes`` -> ``<name>_aot_efr/<name>.h16c.aimodelc``
(``--no-efr``: the same without the flag -> ``<name>_aot/``).

Round-4 result (2026-09-25, 26A428): the efr compile of this graph FAILS — ``coreai-build`` aborts with
``MLIR pass manager failed / Pass failed: MPSMemrefAllocFusion`` and ``operand #0 does not dominate this
use`` located at the first op on a graph input (qi21_dit.py:237 the fp32->bf16 input cast; :239 with
``--bf16-io``), unchanged with img_in / txt_in.in_layer kept bf16 (``--exclude``) and with
``--preferred-compute none``. The plain AOT (``--no-efr``) compiles (int8 kept, 0.57 GiB for 2 layers)
and runs (corr 0.999988), printing an ``ANECCompile() FAILED`` fallback at every new shape. The full
int8lin was not exported (supervisor: the only GPU path of this port is AOT efr).

Run (base venv, from conversion/qwenimage21/):
  python quant_export_dit.py --layers 2 --random-init --aot --check     # probe: reproduces the efr failure
  python quant_export_dit.py --layers 2 --random-init --aot --no-efr --check --check-shapes 40x16x16,40x32x32,8x8x8
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import gc
import json
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import exports_dir, hf_snapshot  # noqa: E402
import export_dit  # noqa: E402
import qi21_dit  # noqa: E402
from export_dit import DEFAULT_CFG, ORDER, example_inputs, randomize, stats  # noqa: E402
from export_encoder import free_gib  # noqa: E402
from qi21_dit import QI21DiT, load_qi21_dit  # noqa: E402

MODEL = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
MIN_FREE_GIB = 30


def fqn(cls) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def quant_config(block: int = 32, exclude: tuple[str, ...] = ()) -> dict:
    ws = {"dtype": "int8", "qscheme": "symmetric_with_clipping",
          "granularity": {"type": "per_block", "block_size": block, "axis": 1}}
    cfg = {
        "execution_mode": "eager",
        "global_config": {"op_state_spec": {"weight": ws}, "op_input_spec": None, "op_output_spec": None},
        "module_type_configs": {
            "torch.nn.modules.sparse.Embedding": None,
            "torch.nn.modules.normalization.LayerNorm": None,
            fqn(qi21_dit.RMSNorm): None,
            fqn(qi21_dit.ZeroCenterRMSNorm): None,
        },
    }
    if exclude:                                  # diagnostic: keep these Linears bf16
        cfg["module_name_configs"] = {n: None for n in exclude}
    return cfg


def census(model, linear_names: list[str]) -> dict:
    import torch.nn.utils.parametrize as P
    quant, qbytes, sbytes, example = [], 0, 0, None
    for name, mod in model.named_modules():
        if not P.is_parametrized(mod):
            continue
        for pname, plist in mod.parametrizations.items():
            for p in plist:
                if hasattr(p, "quantized_data") and hasattr(p, "scale"):
                    quant.append(f"{name}.{pname}")
                    qbytes += p.quantized_data.numel() * p.quantized_data.element_size()
                    sbytes += p.scale.numel() * p.scale.element_size()
                    if example is None:
                        example = dict(param=f"{name}.{pname}", qdtype=str(p.quantized_data.dtype),
                                       qshape=list(p.quantized_data.shape), sdtype=str(p.scale.dtype),
                                       sshape=list(p.scale.shape), qmin=int(p.quantized_data.min()),
                                       qmax=int(p.quantized_data.max()))
    want = sorted(f"{n}.weight" for n in linear_names)
    got = sorted(quant)
    assert got == want, f"quantized set != the {len(want)} Linear weights: extra {sorted(set(got) - set(want))[:5]} " \
                        f"missing {sorted(set(want) - set(got))[:5]}"
    plain = {n: str(p.dtype) for n, p in model.named_parameters() if "parametrizations" not in n}
    return dict(n_quantized=len(got), quantized_data_bytes=qbytes, scale_bytes=sbytes, example=example,
                unquantized_params=len(plain), unquantized_names=sorted(plain)[:8],
                unquantized_dtypes=sorted(set(plain.values())))


def torch_ref(model, steps: str, axes) -> list[dict]:
    """int8 weights + fp32 compute (the quantized module upcast), teacher-forced on oracle/256."""
    from qi21_host import build_inputs
    from qi21_oracle import Oracle, parse_steps
    o = Oracle(HERE / "oracle" / "256")
    m32 = copy.deepcopy(model).to(torch.float32).eval()
    assert m32.img_in.weight.dtype == torch.float32
    rows = []
    with torch.no_grad():
        for s in parse_steps(steps, o.steps):
            t0 = time.time()
            v = m32(**build_inputs(o.latent(s), o.prompt_embeds, o.t(s), o.H, o.W, axes_dim=axes)).numpy()
            maxd, c = stats(v, o.vel(s).numpy())
            r = float(np.linalg.norm(v.astype(np.float64) - o.vel(s).numpy().astype(np.float64))
                      / np.linalg.norm(o.vel(s).numpy().astype(np.float64)))
            rows.append(dict(step=s, corr=c, maxd=maxd, rel=r, nan=int(np.isnan(v).sum()), sec=time.time() - t0))
            print(f"[torch-ref] int8 weights + fp32 compute, oracle/256 step {s:2d}: vel corr {c:.6f}  max|d| "
                  f"{maxd:.3e}  |d|/|ref| {r:.3e}  ({time.time() - t0:.0f}s)", flush=True)
    del m32
    gc.collect()
    return rows


async def engine_check(path: Path, model, shapes, seed: int):
    """Probe check: engine vs the quantized module in fp32 torch (random weights)."""
    import coreai.runtime as rt
    t0 = time.time()
    aim = await rt.AIModel.load(path, rt.SpecializationOptions.default())
    fn = aim.load_function("main")
    print(f"[check] default(aot): load {time.time() - t0:.1f}s", flush=True)
    m32 = copy.deepcopy(model).to(torch.float32)
    res = []
    for (L, H, W) in shapes:
        ins = example_inputs(L, H, W, seed, model.img_in.in_features, model.txt_in.in_layer.in_features)
        t1 = time.time()
        r = await fn({k: rt.NDArray(ins[k].contiguous()) for k in ORDER})
        t_fwd = time.time() - t1
        out = np.array(r["vel"].numpy(), dtype=np.float32)
        with torch.no_grad():
            ref = m32(**ins).numpy()
        nan = int(np.isnan(out).sum())
        maxd, c = stats(out, ref) if nan == 0 else (float("nan"), float("nan"))
        ok = nan == 0 and c >= 0.999
        res.append(dict(L=L, H=H, W=W, nan=nan, corr=c, maxd=maxd, ok=ok, sec=t_fwd))
        print(f"[check] L={L} N={H * W}: fwd {t_fwd:.2f}s NaN {nan} engine vs torch quantized (fp32 compute): "
              f"corr {c:.6f} max|d| {maxd:.3e}  {'PASS' if ok else 'FAIL'}", flush=True)
    del fn, aim
    return res


def aot_compile(bundle: Path, aimodelc: Path, efr: bool) -> float:
    """xcrun coreai-build compile for this Mac's GPU (h16c); ``efr`` = --expect-frequent-reshapes."""
    import subprocess
    out_dir = aimodelc.parent
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    cmd = ["xcrun", "coreai-build", "compile", str(bundle), "--output", str(out_dir), "--platform", "macOS",
           "--architecture", "h16c", "--preferred-compute", "gpu"] + (["--expect-frequent-reshapes"] if efr else [])
    print(f"[aot] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    sec = time.time() - t0
    assert aimodelc.exists(), f"no {aimodelc} after compile: {sorted(p.name for p in out_dir.iterdir())}"
    print(f"[aot] {aimodelc} ({dir_size(aimodelc) / 2**30:.2f} GiB) in {sec:.0f}s; disk free {free_gib():.1f} GiB",
          flush=True)
    return sec


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aot", action="store_true")
    ap.add_argument("--no-efr", action="store_true",
                    help="plain AOT (no --expect-frequent-reshapes) -> <name>_aot/: the efr compile of this graph "
                         "fails in MPSGraph ('operand #0 does not dominate this use', 2026-09-25)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-shapes", default="40x16x16")
    ap.add_argument("--torch-ref", action="store_true")
    ap.add_argument("--ref-steps", default="0,20,39")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--exclude", default="", help="comma list of Linear module names kept bf16 (diagnostic)")
    ap.add_argument("--bf16-io", action="store_true",
                    help="diagnostic probe only: bf16 img_tokens/txt_feats inputs, no fp32->bf16 input cast in the graph")
    ap.add_argument("--name", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    tag = "full" if args.layers is None else f"L{args.layers}"
    exclude = tuple(v for v in args.exclude.split(",") if v)
    assert not args.bf16_io or args.random_init, "--bf16-io is a probe diagnostic (a Swift host cannot fill bf16)"
    io = "bf16io" if args.bf16_io else "iofp32"
    name = args.name or f"qi21_dit_{tag}_int8lin{'_x%d' % len(exclude) if exclude else ''}_dyn_{io}"
    root = exports_dir() / "qwenimage21"
    bundle = root / name / f"{name}.aimodel"
    aimodelc = root / (f"{name}_aot" if args.no_efr else f"{name}_aot_efr") / f"{name}.h16c.aimodelc"
    res = dict(name=name, recipe="int8 per_block 32 axis 1 symmetric_with_clipping, weight-only, bf16 compute", sec={})
    t_start = time.time()
    if args.random_init:
        cfg = dict(DEFAULT_CFG)
        if args.layers is not None:
            cfg["num_layers"] = args.layers
        model = QI21DiT.from_config(cfg, io_fp32=not args.bf16_io, rope_fp32=True)
        randomize(model, args.seed)
        model = model.to(torch.bfloat16).eval()
    else:
        tdir = Path(hf_snapshot(MODEL, revision=REVISION)) / "transformer"
        export_dit.AXES = tuple(json.load(open(tdir / "config.json")).get("axes_dims_rope", export_dit.AXES))
        print(f"[qexport] loading {tdir} (bf16, axes {export_dit.AXES}) ...", flush=True)
        model = load_qi21_dit(tdir, dtype=torch.bfloat16, n_layers=args.layers, io_fp32=True, rope_fp32=True)
    linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n not in exclude]
    res["sec"]["build"] = time.time() - t_start
    print(f"[qexport] {name}: {sum(p.numel() for p in model.parameters()) / 1e9:.3f} B params, "
          f"{len(linear_names)} Linears; built in {res['sec']['build']:.0f}s", flush=True)

    from coreai_models.export.compression import quantize_pytorch_model
    ref = example_inputs(40, 16, 16, args.seed + 1, model.img_in.in_features, model.txt_in.in_layer.in_features)
    if args.bf16_io:
        ref["img_tokens"], ref["txt_feats"] = ref["img_tokens"].bfloat16(), ref["txt_feats"].bfloat16()
    t0 = time.time()
    model = quantize_pytorch_model(model, tuple(ref[k] for k in ORDER), None, quant_config(args.block, exclude))
    res["sec"]["quantize"] = time.time() - t0
    gc.collect()
    res["census"] = c = census(model, linear_names)
    print(f"[qexport] quantized in {res['sec']['quantize']:.0f}s: {c['n_quantized']} Linear weights (quantized_data "
          f"{c['quantized_data_bytes'] / 2**30:.2f} GiB, scales {c['scale_bytes'] / 2**30:.3f} GiB); example "
          f"{c['example']}; unquantized {c['unquantized_params']} {c['unquantized_dtypes']} e.g. {c['unquantized_names']}; "
          f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.1f} GiB", flush=True)

    if args.torch_ref and not args.random_init:
        t0 = time.time()
        res["torch_ref"] = torch_ref(model, args.ref_steps, export_dit.AXES)
        res["sec"]["torch_ref"] = time.time() - t0

    if not args.no_export:
        fg = free_gib()
        print(f"[qexport] disk free {fg:.1f} GiB", flush=True)
        if fg < MIN_FREE_GIB:
            print(f"[qexport] STOP: free {fg:.1f} GiB < {MIN_FREE_GIB} GiB", flush=True)
            return 2
        from torch.export import Dim
        from coreai_models.export.macos import export_to_coreai
        import coreai.runtime as rt
        ntxt = Dim("ntxt", min=8, max=512)
        nimg = Dim("nimg", min=64, max=4096)
        dyn = {"img_tokens": {1: nimg}, "txt_feats": {1: ntxt}, "timestep": None,
               "txt_cos": {1: ntxt}, "txt_sin": {1: ntxt}, "img_cos": {1: nimg}, "img_sin": {1: nimg}}
        t0 = time.time()
        prog = export_to_coreai(model, ref, dynamic_shapes=dyn, input_names=ORDER, output_names=("vel",))
        res["sec"]["convert"] = time.time() - t0
        print(f"[qexport] converted in {res['sec']['convert']:.0f}s", flush=True)
        t0 = time.time()
        prog.optimize()
        res["sec"]["optimize"] = time.time() - t0
        shutil.rmtree(bundle.parent, ignore_errors=True)
        bundle.parent.mkdir(parents=True)
        meta = rt.AIModelAssetMetadata()
        meta.license = "qwen-research"
        meta.model_description = (f"Qwen-Image-2.1 DiT ({tag}{', random weights' if args.random_init else ''}), "
                                  f"int8 per-block-{args.block} Linear weights (symmetric), bf16 compute, fp32 I/O, "
                                  f"dynamic text/image axes. Source: {MODEL}@{REVISION[:7]}.")
        t0 = time.time()
        prog.save_asset(bundle, meta)
        res["sec"]["save"] = time.time() - t0
        res["aimodel_bytes"] = dir_size(bundle)
        print(f"[qexport] saved {bundle} ({res['aimodel_bytes'] / 2**30:.2f} GiB) in {res['sec']['save']:.0f}s; disk "
              f"free {free_gib():.1f} GiB; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.1f} GiB",
              flush=True)
        del prog
        gc.collect()

    if args.aot:
        fg = free_gib()
        if fg < MIN_FREE_GIB:
            print(f"[aot] STOP: free {fg:.1f} GiB < {MIN_FREE_GIB} GiB", flush=True)
            return 2
        res["sec"]["aot"] = aot_compile(bundle, aimodelc, efr=not args.no_efr)
        res["aimodelc_bytes"] = dir_size(aimodelc)

    ok = True
    if args.check:
        shapes = [tuple(int(v) for v in sh.split("x")) for sh in args.check_shapes.split(",")]
        res["check"] = asyncio.run(engine_check(aimodelc, model, shapes, args.seed + 2))
        ok = all(r["ok"] for r in res["check"])
    res["sec"]["total"] = time.time() - t_start
    res["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    js = Path(args.json or HERE / "_work" / f"quant_export_{name}.json")
    json.dump(res, open(js, "w"), indent=2)
    print(f"[qexport] total {res['sec']['total']:.0f}s; -> {js}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
