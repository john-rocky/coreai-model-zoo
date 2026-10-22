"""Export the Timestral graph at one bucket and one form to a Core AI .aimodel.

torch.export (static shapes) -> run_decompositions -> TorchConverter (one function, main) -> optimize
-> strip_debug_info -> save_asset. Inputs / output (dtype = the form's io dtype):

    patches   (R, L, 64)        per context patch: [asinh-normalized values (32), missing indicator (32)]
    cos, sin  (L+h, 64)         RoPE tables for positions arange(L+h) - first observed patch
    attn_mask (1, 1, L+h, L+h)  1 = attend, 0 = masked (the graph selects the scores with it)
    quantiles (R, h*32, 21)     monotone quantiles in the normalized space of the last context patch

Forms (io dtype; what computes in what):
    fp32            float32; everything fp32 (the GPU form, line L2)
    fp16-attn32     float16; fp16 compute, the attention scores and softmax in fp32, RMSNorm as LayerNorm([x, -x])
    fp16w32         float32; fp16 weight storage, fp32 compute
    fp16-layernorm  float16; fp16 everywhere: the scores overflow fp16 on the GPU (the measured counterexample)
    ane             float16; the Neural Engine form (timestral_ane.py: p7 RMSNorm, Q / K / V Linears, sigmoid and
                    SiLU through tanh, 1x1-conv projections). Right on the Neural Engine only; --sigmoid builds its
                    control without the tanh rewrite (misses L3 on the Neural Engine)

    python3 conversion/timestral/export_timestral.py --form ane --bucket 64,32 --rows 1
    python3 conversion/timestral/export_timestral.py --form fp16-attn32 --bucket 256,32 --rows 2 --out-dir <dir>

--st defaults to the pinned revision's model.safetensors in the Hugging Face cache and --out-dir to
<ZOO_EXPORTS>/timestral (conversion/_paths.py). Writes <out-dir>/<name>.aimodel and <out-dir>/<name>.export.json
(sizes, main.hash, op counts, versions); <name> = timestral_209m_L<L>_h<h>_r<R>_<tag>.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import exports_dir, hf_snapshot  # noqa: E402

from engines import INPUT_NAMES, OUTPUT_NAMES  # noqa: E402
from timestral_core import MASK_FILL, load_core, store_fp16_weights  # noqa: E402

SOURCE = "mistralai/Timestral-209M"
REVISION = "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"
AUTHOR = "Mistral AI (Timestral-209M weights); Core AI graph: coreai-model-zoo conversion/timestral"
LICENSE = ("Modified MIT License, attribution notice: 2026 - Mistral AI (the weights' licence, including its "
           "revenue condition; see LICENSE in the bundle's repository)")
# form -> (export dtype, RMSNorm mode, prescale log2, attention scores in fp32, tag in the bundle name)
FORMS = {
    "fp32": ("float32", "fp32", 0, False, "fp32"),
    "fp16-attn32": ("float16", "layernorm", 0, True, "fp16-attn32"),
    "fp16w32": ("float16w32", "fp32", 0, False, "fp16w32"),
    "fp16-layernorm": ("float16", "layernorm", 0, False, "fp16-layernorm"),
    "ane": ("float16", "prescaled", 7, False, "fp16p7-qkv4-tanh-conv"),
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def bundle_name(n_ctx, n_hor, rows, tag):
    return f"timestral_209m_L{n_ctx}_h{n_hor}_r{rows}_{tag}"


def build(form, st, n_hor, sigmoid=False):
    """-> (fp32 / fp16 module, io dtype, tag)."""
    dtype_name, norm, prescale, attn_fp32, tag = FORMS[form]
    io = torch.float16 if dtype_name == "float16" else torch.float32
    if form == "ane":
        from timestral_ane import load_ane_core
        return load_ane_core(st, n_hor, tanh=not sigmoid), io, ("fp16p7-qkv4-conv" if sigmoid else tag)
    assert not sigmoid, "--sigmoid is the ane form's control"
    out_clamp = float(torch.finfo(torch.float16).max) if io == torch.float16 else None
    core = load_core(st, n_hor, norm_mode=norm, prescale_log2=prescale, out_clamp=out_clamp,
                     mask_fill=MASK_FILL[io], attn_fp32=attn_fp32)
    if dtype_name == "float16":
        core = core.half()
    elif dtype_name == "float16w32":
        store_fp16_weights(core)
    return core, io, tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", required=True, choices=sorted(FORMS))
    ap.add_argument("--bucket", required=True, help="L,h in patches: L * 32 context steps, h * 32 forecast steps")
    ap.add_argument("--rows", type=int, default=2, help="2 = the series and its mirror in one call; 1 = two calls")
    ap.add_argument("--sigmoid", action="store_true", help="ane form without the tanh rewrite (the control)")
    ap.add_argument("--st", type=Path, default=None, help=f"model.safetensors (default: {SOURCE} @ {REVISION[:7]} "
                                                          "in the Hugging Face cache)")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <ZOO_EXPORTS>/timestral")
    ap.add_argument("--no-optimize", action="store_true")
    a = ap.parse_args()
    n_ctx, n_hor = (int(v) for v in a.bucket.split(","))
    st = a.st or Path(hf_snapshot(SOURCE, "model.safetensors", revision=REVISION))
    out_dir = a.out_dir or exports_dir() / "timestral"
    core, dtype, tag = build(a.form, st, n_hor, a.sigmoid)
    name = bundle_name(n_ctx, n_hor, a.rows, tag)
    out = out_dir / f"{name}.aimodel"
    s = n_ctx + n_hor
    ex = (torch.zeros(a.rows, n_ctx, 64, dtype=dtype), torch.ones(s, 64, dtype=dtype),
          torch.zeros(s, 64, dtype=dtype), torch.ones(1, 1, s, s, dtype=dtype))
    from coreai.runtime import AIModelAssetMetadata
    from coreai_torch import TorchConverter, get_decomp_table
    from coreai_torch.debugging.debug_info import strip_debug_info

    secs = {}
    t0 = time.perf_counter()
    with torch.no_grad():
        ep = torch.export.export(core, ex)
    ep = ep.run_decompositions(get_decomp_table())
    secs["torch_export"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    prog = (TorchConverter()
            .add_exported_program(exported_program=ep, input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
                                  entrypoint_name="main")
            .to_coreai())
    secs["to_coreai"] = time.perf_counter() - t0
    if not a.no_optimize:
        t0 = time.perf_counter()
        prog.optimize()
        secs["optimize"] = time.perf_counter() - t0
    ops = Counter()

    def walk(op):
        ops[op.name] += 1
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    walk(child.operation)

    walk(prog._mlir_module.operation)
    strip_debug_info(prog)
    meta = AIModelAssetMetadata()  # the constructor ignores keyword arguments: set the fields
    desc = (f"Timestral-209M ({SOURCE} @ {REVISION[:8]}) graph core, form {a.form}{' (sigmoid control)' if a.sigmoid else ''}, "
            f"univariate, bucket L={n_ctx} context patches x h={n_hor} horizon patches ({n_ctx * 32} context steps, "
            f"{n_hor * 32} forecast steps), R={a.rows} rows. The host does the scaling, the refine frame, the cap and "
            "symmetric averaging (conversion/timestral/host_forecast.py). https://huggingface.co/mistralai/Timestral-209M")
    meta.author, meta.license, meta.model_description = AUTHOR, LICENSE, desc
    assert (meta.author, meta.license, meta.model_description) == (AUTHOR, LICENSE, desc)
    shutil.rmtree(out, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    prog.save_asset(out, meta)
    secs["save"] = time.perf_counter() - t0
    files = sorted(p for p in out.rglob("*") if p.is_file())
    home = str(Path.home()).encode()
    mlirb = out / "main.mlirb"
    rec = dict(name=name, form=a.form, sigmoid=a.sigmoid, bucket=[n_ctx, n_hor], rows=a.rows, io_dtype=str(dtype),
               optimized=not a.no_optimize, source=SOURCE, revision=REVISION, safetensors_sha256=sha256_file(st),
               bytes=sum(p.stat().st_size for p in files),
               files={str(p.relative_to(out)): p.stat().st_size for p in files},
               main_hash=(out / "main.hash").read_bytes().hex() if (out / "main.hash").exists() else None,
               main_mlirb_sha256=sha256_file(mlirb) if mlirb.exists() else None,
               home_path_bytes_in_mlirb=mlirb.read_bytes().count(home) if mlirb.exists() else None,
               op_counts=dict(sorted(ops.items())), ops_total=sum(ops.values()), seconds=secs,
               versions={k: version(k) for k in ("torch", "coreai-torch", "coreai-core", "numpy", "safetensors")},
               unused_rows=getattr(core, "unused_rows", None))
    (out_dir / f"{name}.export.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("name", "bytes", "main_hash", "ops_total", "seconds", "home_path_bytes_in_mlirb")}))
    print("EXPORT_DONE", name)


if __name__ == "__main__":
    main()
