#!/usr/bin/env python3
"""Export WeVisOne-VL-2B for Core AI through the Qwen3-VL exporter (`conversion/export_qwen3_vl_pipelined.py`).

WeVisOne-VL-2B is a Qwen3-VL-2B fine-tune (same text decoder and vision tower; vocabulary 152680, untied head, F32
safetensors). The Qwen3-VL exporter takes it as is; what is specific to this checkpoint lives here:

  * the revision: the exporter's loaders take a repo id and read the default branch, so this wrapper asserts that the
    local hub cache's `refs/main` names the pinned commit and runs offline (`HF_HUB_OFFLINE=1`);
  * the tokenizer: the checkpoint's `tokenizer_config.json` is in the transformers-5 layout (`extra_special_tokens`
    is a list), which the export venv's transformers 4.57 cannot load (`AttributeError: 'list' object has no
    attribute 'keys'`) — and the exporter calls `AutoTokenizer.from_pretrained(...).save_pretrained(...)` after
    `save_asset`. The bundle gets the checkpoint's own tokenizer files instead, byte for byte;
  * the provenance in `metadata.json` (`source.hf_revision`, `compression.scheme`);
  * an int8 vision tower (the exporter ships the tower in fp16 only), for the quantization lever.

Products (one per call, the grid picks the vision side: 14 -> 448 x 448 / 196 tokens, 16 -> 512 x 512 / 256):
  dec-int8hu-pf64  int8hu decoder, multifunction `_pf64` = "main" S=1 + "prefill" S=64 (weights shared)
  dec-int8hu-s1    int8hu decoder, static S=1 only
  dec-int8lin-pf64 int8 body, lm_head left fp16 (the exporter's int8lin), multifunction `_pf64`
  dec-fp16w32-s1   fp16 weights cast to fp32 in the graph, fp32 compute and I/O, static S=1 single function
  dec-fp16-s1      fp16 decoder, static S=1 only (an fp16 multifunction would carry one fp16 copy per function)
  vis-fp16         the fixed-grid vision tower, fp16
  vis-int8lin      the tower with every nn.Linear int8 per-block-32 symmetric_with_clipping (LayerNorm fp16)

    QW <coreai venv>/bin/python export_wevisone.py <product> --grid 14 --out-root <exports> --record <json> \
        [--oracle <oracle dir>]   # --oracle: encode the oracle's answers with the bundle tokenizer
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
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

HF_ID = "tencent/WeVisOne-VL-2B"
REVISION = "3dad9ce8bcb6fbe0aea945bf8860bd2cfc3663bf"
SHORT = "wevisone_vl_2b"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
SPECIAL = {"<OBJ>": 152678, "<0>": 151669, "<1000>": 152669, "<box>": 152672, "</box>": 152673,
           "<ref>": 152670, "</ref>": 152671, "<|image_pad|>": 151655, "<|im_end|>": 151645}
PRODUCTS = ("dec-int8hu-pf64", "dec-int8hu-s1", "dec-int8lin-pf64", "dec-fp16-s1", "dec-fp16w32-s1",
            "vis-fp16", "vis-int8lin")


def hub_snapshot() -> Path:
    from _paths import hf_cache

    repo = hf_cache() / ("models--" + HF_ID.replace("/", "--"))
    ref = (repo / "refs" / "main").read_text().strip()
    if ref != REVISION:
        raise SystemExit(f"refs/main = {ref}, expected {REVISION}: the default branch moved; re-pin first")
    snap = repo / "snapshots" / REVISION
    if not (snap / "model.safetensors").exists():
        raise SystemExit(f"no weights in {snap}")
    return snap


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def tree(path: Path) -> dict:
    files = sorted(p for p in path.rglob("*") if p.is_file())
    return {"bytes": sum(p.stat().st_size for p in files),
            "files": {str(p.relative_to(path)): p.stat().st_size for p in files}}


class _RawTokenizer:
    """Stands in for `AutoTokenizer.from_pretrained(hf_id)` inside the exporter: `save_pretrained` copies the
    checkpoint's tokenizer files unchanged."""

    snapshot: Path | None = None
    saved: list[str] = []

    @classmethod
    def from_pretrained(cls, hf_id, *a, **k):
        if hf_id != HF_ID:
            raise SystemExit(f"tokenizer requested for {hf_id}")
        return cls()

    def save_pretrained(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name in TOKENIZER_FILES:
            shutil.copy2(self.snapshot / name, out / name)
        _RawTokenizer.saved.append(str(out))


def count_quantized(model) -> dict:
    import torch
    import torch.nn.utils.parametrize as P

    lins = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    q = [n for n, m in lins if P.is_parametrized(m, "weight")]
    return {"linear_modules": len(lins), "int8_linear_modules": len(q),
            "fp_linear_modules": [n for n, m in lins if not P.is_parametrized(m, "weight")][:8],
            "lm_head_int8": any(n.endswith("lm_head") for n in q)}


def run_exporter(argv: list[str], quant_log: list) -> None:
    import coreai_models.export.compression as C
    import transformers

    import export_qwen3_vl_pipelined as X

    orig = C.quantize_pytorch_model

    def counted(model, *a, **k):
        t0 = time.monotonic()
        out = orig(model, *a, **k)
        quant_log.append({**count_quantized(out), "seconds": time.monotonic() - t0})
        return out

    C.quantize_pytorch_model = counted
    transformers.AutoTokenizer = _RawTokenizer
    old = sys.argv
    sys.argv = ["export_qwen3_vl_pipelined.py", *argv]
    try:
        X.main()
    finally:
        sys.argv = old
        C.quantize_pytorch_model = orig


def fp16_storage_fp32_compute(module):
    """Keep every parameter as fp16 and hand the forward an fp32 copy (copied from
    conversion/export_qwen38vl_pipelined.py, which copied it from conversion/funasr_nano/export_encoder.py).
    Buffers (RoPE caches, M-RoPE masks) stay as they are."""
    import torch
    import torch.nn.utils.parametrize as P

    class Upcast(torch.nn.Module):
        def forward(self, w):
            return w.float()

    for m in list(module.modules()):
        for name, param in list(m.named_parameters(recurse=False)):
            setattr(m, name, torch.nn.Parameter(param.detach().half(), requires_grad=False))
            P.register_parametrization(m, name, Upcast(), unsafe=True)
    return module


def decoder_fp16w32_s1(grid: int, out_dir: Path, max_ctx: int) -> int:
    """fp16 weights, fp32 compute, static S=1 single function; fp32 I/O (image inputs, KV, logits)."""
    import coreai.runtime as rt
    import torch
    from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
    from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai

    from _bundle import write_bundle_metadata
    from export_qwen3_vl_pipelined import PIPELINED_STATE_NAMES, Qwen3VLPipelinedForCausalLM

    model = Qwen3VLPipelinedForCausalLM.from_hf(HF_ID, target_dtype=torch.float32, grid_h=grid, grid_w=grid)
    vocab = model.config.vocab_size
    model = fp16_storage_fp32_compute(model)
    spec = model.build_export_spec(torch.float32, max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, trace_query=1)
    prog = export_to_coreai(model, spec["reference_inputs"], dynamic_shapes=spec["dynamic_shapes"],
                            input_names=spec["input_names"], output_names=spec["output_names"],
                            state_names=PIPELINED_STATE_NAMES,
                            externalize_modules=[s for s in _EXTERNALIZE_SPECS
                                                 if s.composite_op_name != "gated_delta_update"])
    prog.optimize()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    prog.save_asset(out_dir / f"{out_dir.name}.aimodel", rt.AIModelAssetMetadata())
    write_bundle_metadata(out_dir, out_dir.name, HF_ID, vocab, max_ctx)
    _RawTokenizer().save_pretrained(out_dir / "tokenizer")
    return vocab


def vision_int8lin(grid: int, out_dir: Path, quant_log: list) -> None:
    import coreai.runtime as rt
    import torch
    from coreai_models.export.compression import quantize_pytorch_model
    from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai

    from export_qwen3_vl_pipelined import Qwen3VLVisionEncoder, linear_quant_config

    vis = Qwen3VLVisionEncoder.from_hf(HF_ID, target_dtype=torch.float16, grid_h=grid, grid_w=grid)
    v = vis.vcfg
    patches = torch.zeros(vis.n_patches, v.in_channels * v.temporal_patch_size * v.patch_size ** 2, dtype=torch.float16)
    cfg = linear_quant_config("int8")
    cfg["module_type_configs"] = {"torch.nn.modules.normalization.LayerNorm": None,
                                  "coreai_models.primitives.macos.sdpa.SDPA": None}
    cfg["module_name_configs"] = {}
    t0 = time.monotonic()
    vis = quantize_pytorch_model(vis, (patches,), None, cfg)
    quant_log.append({**count_quantized(vis), "seconds": time.monotonic() - t0, "recipe": cfg["global_config"]})
    prog = export_to_coreai(vis, {"patches": patches}, dynamic_shapes={"patches": None}, input_names=("patches",),
                            output_names=("image_embeds", "deepstack_embeds"), state_names=(),
                            externalize_modules=[s for s in _EXTERNALIZE_SPECS
                                                 if s.composite_op_name != "gated_delta_update"])
    prog.optimize()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    prog.save_asset(out_dir / f"{out_dir.name}.aimodel", rt.AIModelAssetMetadata())


def tokenizer_check(bundle: Path, oracle: Path | None, form: str) -> dict:
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(str(bundle / "tokenizer" / "tokenizer.json"))
    ids = {k: tk.token_to_id(k) for k in SPECIAL}
    rec = {"special_ids": ids, "special_ids_equal": ids == SPECIAL,
           "coordinate_tokens_present": all(tk.token_to_id(f"<{i}>") == 151669 + i for i in range(1001)),
           "files_equal_checkpoint": {n: sha256_file(bundle / "tokenizer" / n) == sha256_file(_RawTokenizer.snapshot / n)
                                      for n in TOKENIZER_FILES}}
    if oracle is not None and (oracle / form).is_dir():
        n_eq, diffs = 0, []
        files = sorted((oracle / form).glob("*.json"))
        for f in files:
            o = json.loads(f.read_text())
            enc = tk.encode(o["raw_text"], add_special_tokens=False).ids
            if enc == o["gen_ids"]:
                n_eq += 1
            else:
                d = next((i for i in range(min(len(enc), len(o["gen_ids"]))) if enc[i] != o["gen_ids"][i]),
                         min(len(enc), len(o["gen_ids"])))
                diffs.append({"id": o["id"], "first_difference": d, "encoded": enc[max(0, d - 2):d + 4],
                              "generated": o["gen_ids"][max(0, d - 2):d + 4], "len": [len(enc), len(o["gen_ids"])]})
        rec["oracle_answers"] = {"form": form, "rows": len(files), "encode_equals_generated": n_eq, "differences": diffs}
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("product", choices=PRODUCTS)
    ap.add_argument("--grid", type=int, required=True, choices=[14, 16])
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--oracle", default=None)
    ap.add_argument("--max-ctx", type=int, default=4096)
    args = ap.parse_args()

    snap = hub_snapshot()
    _RawTokenizer.snapshot = snap
    out_root = Path(args.out_root) / f"g{args.grid}"
    out_root.mkdir(parents=True, exist_ok=True)
    quant_log: list = []
    t0 = time.monotonic()
    common = ["--hf-id", HF_ID, "--out-dir", str(out_root), "--grid", str(args.grid), "--max-ctx", str(args.max_ctx)]
    if args.product == "dec-int8hu-pf64":
        run_exporter(["int8hu", *common, "--skip-vision", "--skip-dynamic", "--skip-s1", "--prefill-chunk", "64"],
                     quant_log)
        bundle, mode = out_root / f"{SHORT}_decode_int8hu_pf64", "int8hu"
    elif args.product == "dec-fp16w32-s1":
        bundle, mode = out_root / f"{SHORT}_decode_fp16w32_s1", "fp16w32"
        decoder_fp16w32_s1(args.grid, bundle, args.max_ctx)
    elif args.product == "dec-int8lin-pf64":
        run_exporter(["int8lin", *common, "--skip-vision", "--skip-dynamic", "--skip-s1", "--prefill-chunk", "64"],
                     quant_log)
        bundle, mode = out_root / f"{SHORT}_decode_int8lin_pf64", "int8lin"
    elif args.product == "dec-int8hu-s1":
        run_exporter(["int8hu", *common, "--skip-vision", "--skip-dynamic"], quant_log)
        bundle, mode = out_root / f"{SHORT}_decode_int8hu_s1", "int8hu"
    elif args.product == "dec-fp16-s1":
        run_exporter(["fp16", *common, "--skip-vision", "--skip-dynamic"], quant_log)
        bundle, mode = out_root / f"{SHORT}_decode_fp16_s1", "fp16"
    elif args.product == "vis-fp16":
        run_exporter(["fp16", *common, "--skip-decoder"], quant_log)
        bundle, mode = out_root / f"{SHORT}_vision", "fp16"
    else:
        bundle, mode = out_root / f"{SHORT}_vision_int8lin", "int8lin"
        vision_int8lin(args.grid, bundle, quant_log)
    wall = time.monotonic() - t0

    rec: dict = {"schema": "wevisone-export/1", "product": args.product, "grid": args.grid, "hf_id": HF_ID,
                 "revision": REVISION, "bundle": str(bundle), "mode": mode, "wall_seconds": wall,
                 "quantization": quant_log, "tokenizer_saved": _RawTokenizer.saved}
    meta = bundle / "metadata.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        m["source"]["hf_revision"] = REVISION
        m["compression"] = None if mode == "fp16" else {"scheme": mode}
        meta.write_text(json.dumps(m, indent=2))
        rec["metadata"] = m
        rec["tokenizer"] = tokenizer_check(bundle, Path(args.oracle) if args.oracle else None,
                                           {14: "O2-448", 16: "O2-512"}[args.grid])
    rec["tree"] = tree(bundle)
    rec["aimodel_main_hash_hex"] = next(((p / "main.hash").read_bytes().hex() for p in bundle.glob("*.aimodel")
                                         if (p / "main.hash").exists()), None)
    import coreai
    import coreai_torch
    import torch

    rec["versions"] = {"torch": torch.__version__, "coreai_torch": getattr(coreai_torch, "__version__", None),
                       "coreai": getattr(coreai, "__version__", None)}
    rec["df_g_home"] = subprocess.run(["df", "-g", str(Path.home())], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    rec["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rec["script_sha256"] = sha256_file(Path(__file__))
    rec["exporter_sha256"] = sha256_file(HERE.parent / "export_qwen3_vl_pipelined.py")
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    # quantize_pytorch_model writes torch dtypes back into the config dicts recorded under `quantization`
    Path(args.record).write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(f"[export] {args.product} g{args.grid}: {bundle} {rec['tree']['bytes']:,} B in {wall:.0f}s; "
          f"quant {[(q.get('int8_linear_modules'), q.get('linear_modules')) for q in quant_log]}; "
          f"tokenizer {rec.get('tokenizer', {}).get('special_ids_equal')} "
          f"{(rec.get('tokenizer', {}).get('oracle_answers') or {}).get('encode_equals_generated')}; {rec['df_g_home']}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
