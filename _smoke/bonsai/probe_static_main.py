"""Does a fully static `main` step faster than the dynamic-position one? (round 3 probe)

The decode contract has `position_ids` [1, ?] (the full position range) and a dynamic KV
sequence dim, so every step is a new input shape; with `--expect-frequent-reshapes` the runtime
re-runs shape inference on the CPU each step (Time Profiler: MPSGraphDelegateKernel
inferValue / mlir inferReturnTypes on the decode thread), without it it recompiles per step
(7 tok/s). This probe exports the truncated model's `main` twice, once as the bundle does it and
once with every input static (position_ids [1, P], KV dim fixed), compiles both, and times
S=1 steps through coreai.runtime with the same states. Timing only: the static graph is fed a
constant position, which is numerically meaningless.

    .venv/bin/python _smoke/bonsai/probe_static_main.py [--num-layers 4] [--steps 60]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "conversion"))


def compile_asset(aimodel: Path) -> Path:
    metal = subprocess.check_output(["xcrun", "-f", "metal"]).decode().strip()
    cb = os.path.join(os.path.dirname(metal), "coreai-build")
    out = aimodel.parent / "aot"
    subprocess.check_call([cb, "compile", str(aimodel), "--platform", "macOS", "--architecture", "h16s",
                           "--preferred-compute", "gpu", "--expect-frequent-reshapes", "--output", str(out)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return next(out.glob("*.aimodelc"))


async def time_steps(asset: Path, cfg, steps: int, static_positions: int | None):
    import coreai.runtime as rt
    from export_bonsai2_27b_decode_pipelined import TRACE_KV_CACHE_SEQ_LEN, DTYPE, build_decode_state
    m = await rt.AIModel.load(str(asset), rt.SpecializationOptions.default())
    fn = m.load_function("main")
    state = build_decode_state(cfg, max_seq_len=TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)
    st = {"keyCache": rt.NDArray(state["k_cache"].numpy()), "valueCache": rt.NDArray(state["v_cache"].numpy()),
          "convState": rt.NDArray(state["conv_state"].numpy()), "recState": rt.NDArray(state["rec_state"].numpy())}
    times = []
    for s in range(steps):
        n = static_positions if static_positions else s + 1
        feed = {"input_ids": rt.NDArray(np.array([[1000 + s]], dtype=np.int32)),
                "position_ids": rt.NDArray(np.arange(n, dtype=np.int32)[None])}
        t0 = time.perf_counter()
        await fn(inputs=feed, state=st)
        times.append(time.perf_counter() - t0)
    warm = times[10:]
    return sum(warm) / len(warm), min(warm)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--positions", type=int, default=65, help="static P for the static variant")
    ap.add_argument("--keep", default=None, help="keep the work dir here")
    args = ap.parse_args()

    import coreai_torch
    from export_bonsai2_27b_decode_pipelined import (
        DECODE_STATE_NAMES, DTYPE, TRACE_KV_CACHE_SEQ_LEN, _EXTERNALIZE_SPECS, build_decode_state,
        decode_spec, gguf_path, remove_functionalization, register_custom_torch_lowering,
        set_next_token_output,
    )
    from coreai_models.models.macos.bonsai2 import load_bonsai2_from_gguf, set_gdn_mode

    path = gguf_path(None)
    model, kern = load_bonsai2_from_gguf(path, num_layers=args.num_layers, dtype=DTYPE, chunk=[64])
    cfg = model.config
    set_gdn_mode(model, "fused")
    set_next_token_output(model, True)
    specs = [s for s in _EXTERNALIZE_SPECS
             if s.composite_op_name not in ("gated_delta_update", "scaled_dot_product_attention")]
    work = Path(args.keep or tempfile.mkdtemp(prefix="bonsai_static_"))
    results = {}
    for variant in ("dynamic", "static"):
        inputs, dyn = decode_spec(cfg, 4096, 1)
        if variant == "static":
            inputs = dict(inputs)
            inputs["position_ids"] = torch.arange(args.positions, dtype=torch.int32).unsqueeze(0)
            dyn = {k: None for k in dyn}                       # every input static
        converter = coreai_torch.TorchConverter()
        converter.register_custom_kernels(kern.all())

        def export_fn(module, _inputs=inputs, _dyn=dyn):
            with torch.no_grad():
                ep = torch.export.export(module, args=(), kwargs=_inputs, dynamic_shapes=_dyn)
            ep = ep.run_decompositions(coreai_torch.get_decomp_table())
            remove_functionalization(ep)
            return ep

        converter.add_pytorch_module(model, export_fn=export_fn, externalize_modules=specs,
                                     input_names=("input_ids", "position_ids"),
                                     output_names=("logits", "next_token"),
                                     state_names=DECODE_STATE_NAMES, entrypoint_name="main")
        register_custom_torch_lowering(converter)
        prog = converter.to_coreai()
        prog.optimize()
        import coreai.runtime as rt
        d = work / variant
        d.mkdir(parents=True, exist_ok=True)
        a = d / f"m_{variant}.aimodel"
        prog.save_asset(a, rt.AIModelAssetMetadata())
        asset = compile_asset(a)
        mean, best = asyncio.run(time_steps(asset, cfg, args.steps,
                                            args.positions if variant == "static" else None))
        results[variant] = (mean, best)
        print(f"[probe] {variant:8s} main: mean {mean * 1e3:.2f} ms/step, best {best * 1e3:.2f} ms  ({asset})", flush=True)
    d, s = results["dynamic"][0], results["static"][0]
    print(f"[probe] static saves {(d - s) * 1e3:.2f} ms/step on {args.num_layers} layers", flush=True)
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
