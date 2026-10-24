#!/usr/bin/env python3
"""The WeVisOne-VL-2B text decoder with a second output `last_hidden` for the mask decoder, and a toy of the same
shape for the engine question.

`last_hidden [1, s, 2048]` = the last decoder layer's output before the final norm, the tensor the package's
`LastHiddenCapture` hooks (`wevisone_vl/feature_bridge.py`) and feeds to the mask decoder at every `<OBJ>`. Logits
stay output 0 (the CM pipelined engine binds `outputNames[0]` and looks the prefill function's logits up by that
name); the inputs and states are the one-output decoder's, unchanged.

Products (the wrapper `export_wevisone.py` runs the Qwen3-VL exporter — revision assert, the checkpoint's own
tokenizer files, quantization recipe; this file only swaps the decoder class in):
  toy1 / toy2    a 2-layer, 64-wide decoder of the same class (vocab 152680, the four static inputs, the KV
                 state) with seeded random weights, multifunction `_pf64` (main S=1 + prefill S=64), fp16.
                 toy1 = logits only, toy2 = logits + last_hidden, same weights.
  int8hu-pf64    export_wevisone.py's int8hu `_pf64` g14 decoder + last_hidden
  fp16-s1        export_wevisone.py's fp16 `_s1` g14 decoder + last_hidden

    <coreai venv>/bin/python mask_hidden.py <product> --out-root <dir> --record <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # export_wevisone.py
sys.path.insert(1, str(HERE.parent.parent))    # the exporter, _paths, _bundle

import export_wevisone as R1  # noqa: E402
import export_qwen3_vl_pipelined as X  # noqa: E402

BASE = X.Qwen3VLPipelinedForCausalLM
TOY = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
       "num_key_value_heads": 2, "head_dim": 16, "mrope_section": [4, 2, 2]}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


class HiddenQwen3VL(BASE):
    """The overlay decoder's forward, returning `(logits, last_hidden)`."""

    def forward(self, input_ids, position_ids, image_embeds, deepstack_embeds, rope_shift_start, rope_shift_amount,
                k_cache, v_cache):
        from coreai_models.models.macos.qwen3_vl_pipelined import KVCache

        m = self.model
        V = self.config.vocab_size
        N = self.n_image_tokens
        b, s = input_ids.shape
        seq_len = position_ids.shape[-1]
        torch._check_is_size(s)
        torch._check_is_size(seq_len)
        offset = seq_len - s
        torch._check_is_size(offset)
        p = position_ids.narrow(-1, offset, s)
        ids = input_ids
        is_img = ids >= V
        slot = (ids - V).clamp(0, N - 1)
        flat_slot = slot.reshape(-1)
        e_txt = m.embed_tokens(ids.clamp(0, V - 1))
        e_img = image_embeds.index_select(0, flat_slot).reshape(b, s, -1)
        x = torch.where(is_img.unsqueeze(-1), e_img.to(e_txt.dtype), e_txt)
        shift = torch.where(p >= rope_shift_start, rope_shift_amount, torch.zeros_like(p))
        p_text = p - shift
        s0 = p - slot
        row = torch.div(slot, self.grid_w, rounding_mode="floor")
        col = slot - row * self.grid_w
        pos_t = torch.where(is_img, s0, p_text)
        pos_h = torch.where(is_img, s0 + row, p_text)
        pos_w = torch.where(is_img, s0 + col, p_text)
        img_f = is_img.unsqueeze(-1).to(x.dtype)
        cache = KVCache(k_cache, v_cache)
        for i, layer in enumerate(m.layers):
            x = layer(x, position_ids, pos_t, pos_h, pos_w, cache)
            if i < self.n_deepstack:
                ds = deepstack_embeds.index_select(0, flat_slot + i * N)
                x = x + ds.reshape(b, s, -1).to(x.dtype) * img_f
        return self.lm_head(m.norm(x)), x

    def build_export_spec(self, *a, **k) -> dict:
        spec = super().build_export_spec(*a, **k)
        spec["output_names"] = ("logits", "last_hidden")
        return spec


def toy_from_hf(cls, hf_id, target_dtype=torch.float16, grid_h=14, grid_w=14):
    """Same class and contract as the real decoder, 2 layers x 64 wide, seeded random weights (vocab kept so the
    checkpoint's tokenizer fits)."""
    from transformers import AutoConfig

    snap = R1.hub_snapshot()
    cfg = AutoConfig.from_pretrained(str(snap)).text_config
    for k in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
              "head_dim"):
        setattr(cfg, k, TOY[k])
    for attr in ("rope_parameters", "rope_scaling"):
        rp = getattr(cfg, attr, None)
        if isinstance(rp, dict) and "mrope_section" in rp:
            rp = dict(rp)
            rp["mrope_section"] = list(TOY["mrope_section"])
            setattr(cfg, attr, rp)
    torch.manual_seed(0)
    model = cls(cfg, grid_h=grid_h, grid_w=grid_w)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                p.normal_(0.0, 0.05)
    return model.to(target_dtype).eval()


class Toy1(BASE):
    from_hf = classmethod(toy_from_hf)


class Toy2(HiddenQwen3VL):
    from_hf = classmethod(toy_from_hf)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("product", choices=["toy1", "toy2", "int8hu-pf64", "fp16-s1"])
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--grid", type=int, default=14)
    ap.add_argument("--max-ctx", type=int, default=4096)
    args = ap.parse_args()
    R1._RawTokenizer.snapshot = R1.hub_snapshot()
    out_root = Path(args.out_root) / args.product
    out_root.mkdir(parents=True, exist_ok=True)
    common = ["--hf-id", R1.HF_ID, "--out-dir", str(out_root), "--grid", str(args.grid), "--max-ctx", str(args.max_ctx)]
    pf64 = ["--skip-vision", "--skip-dynamic", "--skip-s1", "--prefill-chunk", "64"]
    cls, argv, bundle_name, mode = {
        "toy1": (Toy1, ["fp16", *common, *pf64], f"{R1.SHORT}_decode_fp16_pf64", "fp16"),
        "toy2": (Toy2, ["fp16", *common, *pf64], f"{R1.SHORT}_decode_fp16_pf64", "fp16"),
        "int8hu-pf64": (HiddenQwen3VL, ["int8hu", *common, *pf64], f"{R1.SHORT}_decode_int8hu_pf64", "int8hu"),
        "fp16-s1": (HiddenQwen3VL, ["fp16", *common, "--skip-vision", "--skip-dynamic"],
                    f"{R1.SHORT}_decode_fp16_s1", "fp16"),
    }[args.product]
    X.Qwen3VLPipelinedForCausalLM = cls
    quant_log: list = []
    t0 = time.monotonic()
    try:
        R1.run_exporter(argv, quant_log)
    finally:
        X.Qwen3VLPipelinedForCausalLM = BASE
    wall = time.monotonic() - t0
    bundle = out_root / bundle_name
    meta_p = bundle / "metadata.json"
    m = json.loads(meta_p.read_text())
    m["source"]["hf_revision"] = R1.REVISION
    m["compression"] = None if mode == "fp16" else {"scheme": mode}
    m.setdefault("language", {})["outputs"] = ["logits", "last_hidden"] if cls is not Toy1 else ["logits"]
    meta_p.write_text(json.dumps(m, indent=2))
    files = sorted(p for p in bundle.rglob("*") if p.is_file())
    rec = {"schema": "wevisone-r2-hidden-export/1", "product": args.product, "class": cls.__name__,
           "argv": argv, "bundle": str(bundle), "wall_seconds": wall, "quantization": quant_log,
           "bytes": sum(p.stat().st_size for p in files),
           "files": {str(p.relative_to(bundle)): p.stat().st_size for p in files}, "metadata": m,
           "toy": TOY if args.product.startswith("toy") else None,
           "script": sha256_file(Path(__file__)), "r1_wrapper": sha256_file(Path(R1.__file__)),
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    Path(args.record).write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(f"[hidden export] {args.product}: {bundle} {rec['bytes']:,} B in {wall:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
