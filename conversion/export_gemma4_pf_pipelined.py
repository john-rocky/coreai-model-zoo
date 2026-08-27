"""Gemma 4 E2B QAT int4lin `tbl` decoder + STATIC chunked-prefill function.

Two-entrypoint (`main` S=1 decode + `prefill` S=<pf> chunk) build of the shipped
`gemma4_e2b_qat_decode_int4lin_tbl` graph. A single dynamic S=p prefill allocates
attention scratch ~O(p^2) and jetsams the phone at p>=512; walking the prompt in
fixed pf-token chunks caps the scratch at pf*context, which is what lets the iso
benchmark reach p=1024 at all.

Weights are shared between the two functions (deduped by
`export_to_coreai_multifunction`), so the `prefill` function is nearly free on disk.

Usage (b2 toolchain — call the venv python directly, `uv run` downgrades to b1):
  cd coreai-models
  .venv/bin/python ../coreai-models-community/conversion/export_gemma4_pf_pipelined.py \
      --pf 64 --raw-dir ../ondevice/artifacts/gemma4_qat_ple_raw \
      --hf-id google/gemma-4-E2B-it-qat-q4_0-unquantized --lin-sym --out-dir exports
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from _bundle import write_bundle_metadata

# coreai-torch 0.4.2 moved the private _torch_export_module from converter to
# externalize (same name/signature); the shim below monkeypatches it, so pick
# whichever module carries it.
import coreai_torch.converter as _ct_conv
if not hasattr(_ct_conv, "_torch_export_module"):
    import coreai_torch.externalize as _ct_conv
from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.compression import quantize_pytorch_model
from coreai_models.export.macos import export_to_coreai_multifunction
from coreai_models.models.macos.gemma4_pipelined import Gemma4PipelinedTblForCausalLM
from coreai_models.models.macos.gemma4_text import Gemma4ForCausalLM

DEFAULT_HF_ID = "google/gemma-4-E2B-it-qat-q4_0-unquantized"
DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8",
                        qscheme: str = "symmetric_with_clipping") -> dict:
    """Weight-only linear per-block-32 incl. the head — copied verbatim from
    export_gemma4_decode_pipelined.py so the pf bundle quantizes identically to
    the shipped tbl bundle (qscheme "symmetric" = q4_0 grid, the --lin-sym path)."""
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {
                "weight": {
                    "dtype": dtype,
                    "qscheme": qscheme,
                    "granularity": {"type": "per_block", "block_size": 32, "axis": 1},
                }
            },
            "op_input_spec": None,
            "op_output_spec": None,
        },
        "module_type_configs": {
            "coreai_models.primitives.macos.sdpa.SDPA": None,
            "coreai_models.primitives.macos.rope.RoPE": None,
            "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
            "coreai_models.primitives.macos.rms_norm.RMSNormPlusOne": None,
            "torch.nn.modules.sparse.Embedding": None,
        },
        "module_name_configs": {r".*embed_tokens$": None},
    }


def bundle_basename(hf_id: str) -> str:
    low = hf_id.lower()
    tag = "e4b" if "e4b" in low else "e2b"
    return f"gemma4_{tag}"


def _install_dim_retry_shim() -> None:
    """coreai-torch externalize retry (mirrors export_qwen3_vl_pipelined.py).

    A static S=pf causal SDPA emits a `key_seq >= pf` guard the min=1 fallback
    violates; retry the export with the min/max bounds torch's ConstraintViolation
    message itself suggests.
    """
    orig = _ct_conv._torch_export_module
    if getattr(orig, "_pf_shim", False):
        return

    def _retry(prep):
        for _ in range(3):
            try:
                return orig(prep)
            except Exception as e:  # noqa: BLE001 — retry only on suggested fixes
                fixes = {
                    n: (int(mn), int(mx) if mx else None)
                    for n, mn, mx in re.findall(
                        r"(\w+) = Dim\('\w+', min=(\d+)(?:, max=(\d+))?\)", str(e))
                }
                if not fixes:
                    raise
                print(f"[externalize-retry] {getattr(prep, 'name', '?')}: {fixes}")
                rebuilt: dict[str, object] = {}

                def _remap(dims):
                    if dims is None:
                        return None
                    out = {}
                    for j, d in dims.items():
                        nm = getattr(d, "__name__", None)
                        if nm not in fixes:
                            out[j] = d
                            continue
                        if nm not in rebuilt:
                            mn, mx = fixes[nm]
                            kw = {"min": mn}
                            if mx is not None:
                                kw["max"] = mx
                            rebuilt[nm] = torch.export.Dim(nm, **kw)
                        out[j] = rebuilt[nm]
                    return out

                prep.dynamic_shapes = tuple(_remap(d) for d in prep.dynamic_shapes)
        return orig(prep)

    _retry._pf_shim = True
    _ct_conv._torch_export_module = _retry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pf", type=int, default=64, help="prefill chunk width (S=pf)")
    ap.add_argument("--raw-dir", default="../ondevice/artifacts/gemma4_qat_ple_raw",
                    help="gather-table dump dir (embed_per_layer.i8/.scale.f32 + meta.json)")
    ap.add_argument("--hf-id", default=DEFAULT_HF_ID)
    ap.add_argument("--lin-sym", action="store_true",
                    help="plain absmax symmetric (QAT-grid-aligned q4_0). Match the "
                         "shipped tbl bundle: pass this.")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--local-ckpt", default=None,
                    help="use a local checkpoint dir (safetensors + configs) instead "
                         "of snapshot_download — for when the HF python downloader "
                         "stalls on the single 10 GB file and you curl it yourself")
    args = ap.parse_args()

    name = f"{bundle_basename(args.hf_id)}_qat_decode_int4lin" + \
        ("sym" if args.lin_sym else "") + f"_tbl_pf{args.pf}"

    if args.local_ckpt:
        model_dir = args.local_ckpt
        print(f"using local checkpoint {model_dir}")
    else:
        model_dir = snapshot_download(
            args.hf_id, allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json"])
    print(f"loading {args.hf_id} (fp16) ...")
    causal = Gemma4ForCausalLM.from_local(model_dir, dtype=DTYPE).eval()
    cfg = causal.config
    del causal.model.embed_tokens_per_layer  # keep the 4.7 GB PLE out of RAM

    raw = Path(args.raw_dir)
    meta = json.loads((raw / "meta.json").read_text())
    v, pld = meta["V"], meta["PLD"]
    ple_q = torch.from_numpy(np.array(
        np.memmap(raw / "embed_per_layer.i8", np.int8, "r", shape=(v, pld))))
    ple_s = torch.from_numpy(np.fromfile(raw / "embed_per_layer.scale.f32", np.float32))

    model = Gemma4PipelinedTblForCausalLM(causal).eval()

    # Quantize once against the S=1 decode spec; weights are shared with prefill.
    decode_spec = model.build_export_spec(
        target_dtype=DTYPE, max_context_length=args.max_ctx,
        trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, ple_table=ple_q, ple_scale=ple_s,
        trace_query=1)
    qscheme = "symmetric" if args.lin_sym else "symmetric_with_clipping"
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    print(f"quantizing (linear int4 per-block-32, {qscheme}, incl. untied head) ...")
    model = quantize_pytorch_model(
        model, tuple(decode_spec["reference_inputs"].values()),
        decode_spec["dynamic_shapes"], linear_quant_config("int4", qscheme))

    prefill_spec = model.build_export_spec(
        target_dtype=DTYPE, max_context_length=args.max_ctx,
        trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, ple_table=ple_q, ple_scale=ple_s,
        trace_query=args.pf)

    _install_dim_retry_shim()
    print(f"exporting multifunction (main S=1 + prefill S={args.pf}) ...")
    prog = export_to_coreai_multifunction(
        model,
        [("main", decode_spec), ("prefill", prefill_spec)],
        externalize_modules=[],  # gemma4 opts out (orphan PLE front-end norms)
    )
    print("optimizing ...")
    prog.optimize()

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    import coreai.runtime as rt

    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...")
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())

    write_bundle_metadata(out_dir, name, args.hf_id, cfg.vocab_size, args.max_ctx,
                          functions=("main", "prefill"))
    tok_dir = out_dir / "tokenizer"
    tok_dir.mkdir()
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "chat_template.jinja", "generation_config.json"):
        src = Path(model_dir) / f
        if src.exists():
            shutil.copy(src, tok_dir / f)

    import subprocess
    sz = subprocess.run(["du", "-sh", str(out_dir)], capture_output=True, text=True).stdout.split()[0]
    print(f"bundle ready: {out_dir} ({sz})")


if __name__ == "__main__":
    main()
