"""Export weight-quantized variants of the Qwen-Image-2.1 text encoder to Core AI (round 4).

Same graph as ``export_encoder.py --w16a32`` — ``input_ids [1,L] int32 -> hidden [1,L,4096] fp32``,
L dynamic 16..512, ``embed_tokens`` inside the graph, every op computed in fp32 — except that the
seven projection weights of every layer (q/k/v/o, gate/up/down: 252 tensors, 6.95 B of the 7.57 B
parameters) are stored weight-only quantized. The quantizer is
``coreai_models.export.compression.quantize_pytorch_model`` with the zimage ``linear_quant_config``
recipe (eager mode, ``symmetric_with_clipping``, ``per_block`` along axis 1 = the input features;
Embedding / norms excluded). ``embed_tokens`` (151936 x 4096) and every RMSNorm weight stay bf16,
unquantized.

Variants (``--wbits`` / ``--block``)                  bundle name
  8 / 32   int8, one scale per 32 inputs     qi21_encoder_dynL_w8a32_ids_iofp32
  4 / 32   int4 (-7..7), per 32              qi21_encoder_dynL_w4a32_ids_iofp32
  4 / 8    int4, per 8 ("finer groups")      qi21_encoder_dynL_w4g8a32_ids_iofp32

Two ways to make the fp32-compute encoder quantizable (``--struct``):
  w16    the w16a32 module (bf16 weights, fp32 compute). The eager quantizer only quantizes a
         weight that reaches a registered op AS the parameter; ``qi21_text.Linear`` hands
         ``F.linear`` ``self.weight.float()``, so with the plain config every projection is silently
         skipped, and a ``module_state_spec`` cannot catch ``.float()`` either (coreai-opt 0.2.1
         maps the Tensor method to ``torch.ops.aten.float``, which does not exist ->
         AttributeError). ``ToLinear`` (below) is the same module with the upcast spelled
         ``.to(torch.float32)`` (same numbers), which the ``module_state_spec`` on it can catch:
         bf16 scales, dequant -> bf16 -> cast fp32 in the graph.
  lin32  the launch fallback: the projections hold the checkpoint's bf16 values upcast to fp32
         (exact) in plain ``F.linear`` (``compute_fp32`` off, ``residual_fp32`` on), so the
         quantizer sees each weight directly: fp32 scales, dequant -> fp32.
Either way the census after quantization asserts that exactly the 7 x n_layers projection weights
were quantized and nothing else.

``--torch-ref`` runs the quantized module in torch (CPU, the op's eager kernel) on the oracle
prompt and the three sweep prompts, BEFORE the export, and scores it against the fp32 references:
the quantization's own error, separate from the engine's. Saved to ``_work/torch_quant_text/<name>/``.

``--check`` loads the AOT bundle (``SpecializationOptions.default()``), runs ``--check-L`` random ids
and compares with the same quantized module in torch (and, for ``--random-init`` probes, with the
unquantized fp32 module).

Bundle: ``_paths.exports_dir()/qwenimage21/<name>/<name>.aimodel``; AOT (``--aot``) =
``xcrun coreai-build compile ... --architecture h16c --preferred-compute gpu --expect-frequent-reshapes``
-> ``<name>_aot_efr/<name>.h16c.aimodelc`` (``export_encoder.aot_compile``).

Round-4 result (2026-09-25): the ``w16`` probe quantizes, but its engine output is off its own torch
module by rel ~1.2e-3 (``quant_diag_w16_probe.py``); ``lin32`` is exact on the engine (rel ~2e-6), so the
full bundle is ``lin32``. Its AOT efr ``.aimodelc`` is 34.32 GiB against the 8.44 GiB ``.aimodel``: the
compiled resources carry the projections folded to fp32 constants AND the int8 copy.

Run (coreai-models base venv, from conversion/qwenimage21/):
  python quant_export_encoder.py --layers 2 --random-init --struct lin32 --aot --check   # probe (also --struct w16)
  python quant_export_encoder.py --wbits 8 --block 32 --struct lin32 --aot --torch-ref  # full w8a32 (as run)
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import exports_dir, hf_snapshot  # noqa: E402
import qi21_text  # noqa: E402
from qi21_text import QI21TextEncoder, load_qi21_text  # noqa: E402
from export_encoder import DEFAULT_TEXT_CFG, aot_compile, free_gib, randomize  # noqa: E402

MODEL = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
MIN_FREE_GIB = 30
PROJ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def fqn(cls) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


class ToLinear(qi21_text.Linear):
    """``qi21_text.Linear`` with both fp32 upcasts (input and stored weight) written ``.to(torch.float32)``.

    Same numbers as ``.float()``; the difference is that coreai-opt's eager handler can normalise
    ``Tensor.to`` to an aten op, so a ``module_state_spec`` on this class reaches the parameter.
    """

    def forward(self, x):
        if self.compute_fp32:
            return torch.nn.functional.linear(x.to(torch.float32), self.weight.to(torch.float32))
        return super().forward(x)


def variant_tag(wbits: int, block: int) -> str:
    return f"w{wbits}a32" if block == 32 else f"w{wbits}g{block}a32"


def bundle_name(layers, tag: str, struct: str, probe_struct_suffix: bool) -> str:
    lay = "" if layers is None else f"L{layers}_"
    suf = f"_{struct}" if probe_struct_suffix else ""
    return f"qi21_encoder_{lay}dynL_{tag}{suf}_ids_iofp32"


def paths(name: str) -> tuple[Path, Path]:
    root = exports_dir() / "qwenimage21"
    return root / name / f"{name}.aimodel", root / f"{name}_aot_efr" / f"{name}.h16c.aimodelc"


def weight_spec(wbits: int, block: int) -> dict:
    return {"dtype": f"int{wbits}", "qscheme": "symmetric_with_clipping",
            "granularity": {"type": "per_block", "block_size": block, "axis": 1}}


def quant_config(wbits: int, block: int, struct: str) -> dict:
    """zimage ``linear_quant_config`` form: weight-only, activations untouched, norms/Embedding excluded."""
    ws = weight_spec(wbits, block)
    cfg = {
        "execution_mode": "eager",
        "global_config": {"op_state_spec": {"weight": ws}, "op_input_spec": None, "op_output_spec": None},
        "module_type_configs": {
            "torch.nn.modules.sparse.Embedding": None,
            fqn(qi21_text.RMSNorm): None,
        },
    }
    if struct == "w16":
        # F.linear receives self.weight.float(), not the parameter: point the quantizer at the
        # parameter itself through a module-level state spec on the Linear subclass.
        cfg["module_type_configs"][fqn(ToLinear)] = {
            "op_state_spec": {"weight": ws}, "op_input_spec": None, "op_output_spec": None,
            "module_state_spec": {"weight": ws}}
    return cfg


def build(args):
    """The fp32-compute encoder in the requested structure (bf16 embed_tokens in both)."""
    kw = dict(max_len=args.L_max, io_fp32=True)
    if args.struct == "w16":
        kw.update(compute_fp32=True)                 # bf16 weights, fp32 compute (w16a32)
        dtype = torch.bfloat16
    else:
        kw.update(residual_fp32=True)                # fp32 weights in plain F.linear, fp32 residual
        dtype = torch.float32
    if args.random_init:
        cfg = dict(DEFAULT_TEXT_CFG)
        if args.layers is not None:
            cfg["num_hidden_layers"] = args.layers
        m = QI21TextEncoder.from_config(cfg, **kw)
        randomize(m, args.seed)
        m = m.to(torch.bfloat16)                     # the checkpoint's dtype: every weight is a bf16 value
        m = m.to(dtype)
    else:
        tdir = Path(hf_snapshot(MODEL, revision=REVISION)) / "text_encoder"
        print(f"[qexport] loading {tdir} ({dtype}) ...", flush=True)
        m = load_qi21_text(tdir, dtype=dtype, n_layers=args.layers, **kw)
    m.embed_tokens.to(torch.bfloat16)                # the table stays bf16 (unquantized) in both structures
    if args.struct == "w16":
        for mod in m.modules():
            if type(mod) is qi21_text.Linear:
                mod.__class__ = ToLinear
    return m.eval()


def census(model, n_layers: int) -> dict:
    """Which parameters carry a weight-dequant parametrization; assert = exactly the projections."""
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
                                       sshape=list(p.scale.shape), input_dtype=str(getattr(p, "input_dtype", None)),
                                       qmin=int(p.quantized_data.min()), qmax=int(p.quantized_data.max()))
    want = sorted(f"layers.{i}.{p}.weight" for i in range(n_layers) for p in PROJ)
    got = sorted(quant)
    assert got == want, f"quantized set != the {len(want)} projections: extra {sorted(set(got) - set(want))[:5]} " \
                        f"missing {sorted(set(want) - set(got))[:5]}"
    plain = {n: str(p.dtype) for n, p in model.named_parameters() if "parametrizations" not in n}
    return dict(n_quantized=len(got), quantized_data_bytes=qbytes, scale_bytes=sbytes, example=example,
                unquantized_params=len(plain), unquantized_dtypes=sorted(set(plain.values())))


def stats(a: np.ndarray, b: np.ndarray):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(np.abs(a - b).max()), float(np.corrcoef(a, b)[0, 1]), float(np.linalg.norm(a - b) / np.linalg.norm(b))


def tok_corr(out: np.ndarray, ref: np.ndarray) -> np.ndarray:
    a = out.astype(np.float64).reshape(-1, ref.shape[-1])
    b = ref.astype(np.float64).reshape(-1, ref.shape[-1])
    return np.array([np.corrcoef(a[i], b[i])[0, 1] for i in range(a.shape[0])])


def torch_ref(model, name: str) -> dict:
    """The quantized module in torch (CPU) vs the fp32 references: the quantization's own error."""
    od = HERE / "oracle" / "256"
    meta = json.load(open(od / "meta.json"))
    refs = {"oracle_apple": (np.fromfile(od / "enc_input_ids.i32", "<i4")[None],
                             np.fromfile(od / "enc_hidden_full.f32", "<f4").reshape(int(meta["Lfull"]), -1))}
    for d in sorted((HERE / "_work" / "ref_hf_fp32_text").iterdir()):
        if d.is_dir():
            ids = np.fromfile(d / "ids.i32", "<i4")
            refs[f"sweep_{d.name}"] = (ids[None], np.fromfile(d / "hidden.f32", "<f4").reshape(len(ids), -1))
    sd = HERE / "_work" / "torch_quant_text" / name
    sd.mkdir(parents=True, exist_ok=True)
    res = {}
    with torch.no_grad():
        for key, (ids, ref) in refs.items():
            t0 = time.time()
            out = model(torch.from_numpy(ids.astype(np.int32))).float().numpy()[0]
            np.ascontiguousarray(out, "<f4").tofile(sd / f"{key}.f32")
            c = tok_corr(out, ref)
            maxd, gc_, rel = stats(out, ref)
            res[key] = dict(L=int(ids.shape[1]), min_tok_corr=float(c.min()), argmin=int(c.argmin()),
                            tok14=float(c[14]), n_below_0999=int((c < 0.999).sum()), global_corr=gc_,
                            rel=rel, maxd=maxd, nan=int(np.isnan(out).sum()), sec=time.time() - t0)
            print(f"[torch-ref] {key:<14} L {ids.shape[1]:3d}: torch quantized vs fp32 ref: min tok corr "
                  f"{c.min():.6f} (tok {int(c.argmin())})  tok14 {c[14]:.6f}  n<0.999 {int((c < 0.999).sum())}  "
                  f"global corr {gc_:.6f}  |d|/|ref| {rel:.3e}  ({time.time() - t0:.0f}s)", flush=True)
    return res


async def engine_check(path: Path, model, fp32_model, Ls, seed: int):
    import coreai.runtime as rt
    t0 = time.time()
    aim = await rt.AIModel.load(path, rt.SpecializationOptions.default())
    fn = aim.load_function("main")
    t_load = time.time() - t0
    print(f"[check] default(aot): load {t_load:.1f}s ({path.name})", flush=True)
    vocab = model.embed_tokens.num_embeddings
    rows = []
    for L in Ls:
        g = torch.Generator().manual_seed(seed + L)
        ids = torch.randint(0, vocab, (1, L), generator=g, dtype=torch.int32)
        t1 = time.time()
        r = await fn({"input_ids": rt.NDArray(ids.contiguous())})
        t_fwd = time.time() - t1
        out = np.array(r["hidden"].numpy(), dtype=np.float32)
        with torch.no_grad():
            ref_q = model(ids).float().numpy()
        nan = int(np.isnan(out).sum())
        row = dict(L=L, nan=nan, sec=t_fwd)
        if nan == 0:
            row["vs_torch_quant"] = dict(zip(("maxd", "corr", "rel"), stats(out, ref_q)))
            if fp32_model is not None:
                with torch.no_grad():
                    ref32 = fp32_model(ids).float().numpy()
                row["vs_torch_fp32_unquantized"] = dict(zip(("maxd", "corr", "rel"), stats(out, ref32)))
                row["torch_quant_vs_fp32_unquantized"] = dict(zip(("maxd", "corr", "rel"), stats(ref_q, ref32)))
        row["ok"] = nan == 0 and row["vs_torch_quant"]["corr"] >= 0.999
        rows.append(row)
        extra = ""
        if "vs_torch_fp32_unquantized" in row:
            extra = (f" | engine vs fp32 unquantized corr {row['vs_torch_fp32_unquantized']['corr']:.6f} rel "
                     f"{row['vs_torch_fp32_unquantized']['rel']:.3e} | torch quant vs fp32 corr "
                     f"{row['torch_quant_vs_fp32_unquantized']['corr']:.6f} rel "
                     f"{row['torch_quant_vs_fp32_unquantized']['rel']:.3e}")
        print(f"[check] L={L}: fwd {t_fwd:.2f}s NaN {nan}  engine vs torch quantized: corr "
              f"{row.get('vs_torch_quant', {}).get('corr', float('nan')):.6f} rel "
              f"{row.get('vs_torch_quant', {}).get('rel', float('nan')):.3e}{extra}  {'PASS' if row['ok'] else 'FAIL'}",
              flush=True)
    del fn, aim
    return t_load, rows


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wbits", type=int, default=8, choices=[8, 4])
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--struct", default="lin32", choices=["lin32", "w16"],
                    help="lin32 = the gated full bundles (exact on the engine); w16 = the probe-only finding")
    ap.add_argument("--layers", type=int, default=None, help="first n layers only (probe)")
    ap.add_argument("--random-init", action="store_true", help="random weights (probe)")
    ap.add_argument("--name", default=None, help="override the bundle name")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L-min", type=int, default=16)
    ap.add_argument("--L-max", type=int, default=512)
    ap.add_argument("--trace-L", type=int, default=64)
    ap.add_argument("--aot", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-L", default="32,128")
    ap.add_argument("--torch-ref", action="store_true", help="quantized torch vs fp32 refs (oracle + 3 sweep prompts)")
    ap.add_argument("--no-export", action="store_true", help="quantize (+ --torch-ref) only")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    tag = variant_tag(args.wbits, args.block)
    probe = args.layers is not None or args.random_init
    name = args.name or bundle_name(args.layers, tag, args.struct, probe or args.struct != "lin32")
    bundle, aimodelc = paths(name)
    res = dict(name=name, wbits=args.wbits, block=args.block, struct=args.struct, layers=args.layers,
               random_init=args.random_init, embed_tokens="bf16 (unquantized)", sec={})
    t_start = time.time()
    model = build(args)
    n_layers = len(model.layers)
    res["sec"]["build"] = time.time() - t_start
    fp32_model = None
    if args.check and args.random_init:              # the unquantized reference, for probes only
        fp32_model = copy.deepcopy(model).to(torch.float32).eval()
    print(f"[qexport] {name}: {sum(p.numel() for p in model.parameters()) / 1e9:.3f} B params, {n_layers} layers, "
          f"struct {args.struct}, int{args.wbits} per_block {args.block}; built in {res['sec']['build']:.0f}s", flush=True)

    from coreai_models.export.compression import quantize_pytorch_model
    g = torch.Generator().manual_seed(args.seed + 1)
    ref_ids = torch.randint(0, model.embed_tokens.num_embeddings, (1, args.trace_L), generator=g, dtype=torch.int32)
    t0 = time.time()
    model = quantize_pytorch_model(model, (ref_ids,), None, quant_config(args.wbits, args.block, args.struct))
    res["sec"]["quantize"] = time.time() - t0
    gc.collect()
    res["census"] = census(model, n_layers)
    c = res["census"]
    print(f"[qexport] quantized in {res['sec']['quantize']:.0f}s: {c['n_quantized']} projection weights "
          f"(quantized_data {c['quantized_data_bytes'] / 2**30:.2f} GiB in torch, scales "
          f"{c['scale_bytes'] / 2**30:.3f} GiB); example {c['example']}; unquantized params "
          f"{c['unquantized_params']} {c['unquantized_dtypes']}; peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.1f} GiB", flush=True)

    if args.torch_ref:
        t0 = time.time()
        res["torch_ref"] = torch_ref(model, name)
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
        dyn = {"input_ids": {1: Dim("L", min=args.L_min, max=args.L_max)}}
        t0 = time.time()
        prog = export_to_coreai(model, {"input_ids": ref_ids}, dynamic_shapes=dyn, input_names=("input_ids",),
                                output_names=("hidden",))
        res["sec"]["convert"] = time.time() - t0
        print(f"[qexport] converted in {res['sec']['convert']:.0f}s", flush=True)
        t0 = time.time()
        prog.optimize()
        res["sec"]["optimize"] = time.time() - t0
        shutil.rmtree(bundle.parent, ignore_errors=True)            # save_asset does not overwrite
        bundle.parent.mkdir(parents=True)
        meta = rt.AIModelAssetMetadata()
        meta.license = "qwen-research"
        meta.model_description = (
            f"Qwen-Image-2.1 text encoder (Qwen3-VL-8B text stack, "
            f"{'first %d layers' % args.layers if args.layers else 'all 36 layers'}"
            f"{', random weights' if args.random_init else ''}), int{args.wbits} per-block-{args.block} "
            f"projection weights (symmetric), bf16 embed_tokens and norms, fp32 compute, int32 input_ids -> "
            f"fp32 hidden (last layer, before the final norm), dynamic L {args.L_min}..{args.L_max}. "
            f"Source: {MODEL}@{REVISION[:7]}.")
        t0 = time.time()
        prog.save_asset(bundle, meta)
        res["sec"]["save"] = time.time() - t0
        res["aimodel_bytes"] = dir_size(bundle)
        print(f"[qexport] saved {bundle} ({res['aimodel_bytes'] / 2**30:.2f} GiB) in {res['sec']['save']:.0f}s; "
              f"disk free {free_gib():.1f} GiB; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.1f} GiB",
              flush=True)
        del prog
        gc.collect()

    if args.aot:
        fg = free_gib()
        if fg < MIN_FREE_GIB:
            print(f"[aot] STOP: free {fg:.1f} GiB < {MIN_FREE_GIB} GiB", flush=True)
            return 2
        res["sec"]["aot"] = aot_compile(bundle, aimodelc)
        res["aimodelc_bytes"] = dir_size(aimodelc)

    ok = True
    if args.check:
        Ls = [int(v) for v in args.check_L.split(",")]
        res["load_s"], res["check"] = asyncio.run(engine_check(aimodelc, model, fp32_model, Ls, args.seed + 2))
        ok = all(r["ok"] for r in res["check"])
    res["sec"]["total"] = time.time() - t_start
    res["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    js = Path(args.json or HERE / "_work" / f"quant_export_{name}.json")
    json.dump(res, open(js, "w"), indent=2)
    print(f"[qexport] total {res['sec']['total']:.0f}s; -> {js}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
