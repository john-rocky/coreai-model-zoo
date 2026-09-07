#!/usr/bin/env python3
"""Export OpenBMB MiniCPM5 (1B / 2B) -> Core AI int8 (FM-format dynamic bundle).

MiniCPM5-1B and MiniCPM5-2B are plain `LlamaForCausalLM` checkpoints (same GQA 16:2,
head_dim 128, RoPE theta 5e6, untied 130560-vocab head, 128K; the 2B is 42 layers x
hidden 2048 instead of 24 x 1536), so unlike the other zoo LLMs (custom
`export_*_decode_pipelined.py` torch exports) they ride the STOCK `coreai.llm.export`
CLI. Two one-time patches to the local `coreai_models` package are required first
(the stock registry has no `llama` family and no metadata for these ids); both live in
`conversion/overlay/`:

  1. coreai_models/models/registry.py  — MODEL_TYPE_REMAPPING:
         "llama": "mistral",
     A plain Llama == the Mistral builder minus the sliding window (GQA + RoPE +
     RMSNorm + SiLU, no qkv bias, no qk-norm, honors config.head_dim).

  2. coreai_models/export/metadata.py  — AIModelMetadataFields:
         "openbmb/MiniCPM5-1B": AIModelMetadataFields(author="OpenBMB", license="Apache-2.0", ...)
         "openbmb/MiniCPM5-2B": AIModelMetadataFields(author="OpenBMB", license="Apache-2.0", ...)

The export itself is symmetric per-channel int8 (absmax, no clipping) via the
quantization_config in `minicpm5_int8sym.yaml` (sibling of this file). On iPhone
int8 is ~2.2x fp16 and lossless (1B: 24/24 token-exact vs HF fp32); on a compute-rich
Mac int8 is SLOWER than fp16 (dequant overhead), so the win is iPhone-side.

Ship the DYNAMIC bundle (this default macOS export) to the iPhone: it routes to the
pipelined engine. A `--platform iOS` static export routes to the staticShape engine
and fails at engine-create. See ../knowledge/minicpm5-1b.md for the full rationale.

    python conversion/export_minicpm5.py [--hf-id openbmb/MiniCPM5-2B] [--qconfig minicpm5_int8sym_b32.yaml] [--output-dir DIR]

`--output-dir` defaults to `<coreai-models>/exports/<model-name-lowercased>` (the 1B's
published bundle came from `exports/minicpm5-1b`). The exporter is run from the
coreai-models checkout so `uv run` resolves its project, and `--output-dir` is passed
explicitly so the bundle lands where this script says, not where the CLI's default
workspace lookup does.

Then verify greedy parity vs HF fp32 (token-exact) before shipping —
`conversion/verify_minicpm5.py --hf-id <id> <bundle>` (the shipped 1B int8 bundle scores
24/24 = lossless).
"""
import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
QCONFIG = HERE / "minicpm5_int8sym.yaml"          # per-channel absmax int8 (ships the 1B)
QCONFIG_B32 = HERE / "minicpm5_int8sym_b32.yaml"  # per-block-32 int8 (the 2B experiment)
COREAI_MODELS = HERE.parent.parent / "coreai-models"
CHAT_EOS = "<|im_end|>"  # id 130073 (same tokenizer on 1B and 2B)
DEFAULT_HF_ID = "openbmb/MiniCPM5-1B"


def export(hf_id: str, out_dir: Path, qconfig: Path = QCONFIG) -> None:
    """Run the stock exporter (assumes the two registry/metadata overlay edits above)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["uv", "run", "coreai.llm.export", hf_id,
         "--experimental", "--compute-precision", "float16",
         "--compression-config", str(qconfig),
         "--output-dir", str(out_dir)],
        cwd=COREAI_MODELS, check=True)


def fix_chat_eos(tok_dir: Path) -> None:
    """Base eos is </s> (raw-text terminator); the chat template ends turns with
    <|im_end|>. The engine stops on the single eosTokenId, so a chat bundle must
    declare <|im_end|> as eos (exactly how Qwen ships) or it never halts."""
    cfg = tok_dir / "tokenizer_config.json"
    d = json.loads(cfg.read_text())
    d["eos_token"] = CHAT_EOS
    cfg.write_text(json.dumps(d, ensure_ascii=False, indent=2))
    stm = tok_dir / "special_tokens_map.json"
    if stm.exists():
        m = json.loads(stm.read_text())
        eos = m.get("eos_token")
        if isinstance(eos, dict):
            eos["content"] = CHAT_EOS
        else:
            m["eos_token"] = CHAT_EOS
        stm.write_text(json.dumps(m, ensure_ascii=False, indent=2))
    print(f"chat eos -> {CHAT_EOS} in {tok_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hf-id", default=DEFAULT_HF_ID,
                    help=f"source checkpoint (default: {DEFAULT_HF_ID}; also openbmb/MiniCPM5-2B)")
    ap.add_argument("--output-dir", default=None,
                    help="bundle parent dir (default: <coreai-models>/exports/<model-name-lowercased>)")
    ap.add_argument("--qconfig", default=str(QCONFIG),
                    help=f"coreai-opt quantization YAML (default: {QCONFIG.name}; the 2B ships {QCONFIG_B32.name})")
    args = ap.parse_args()
    name = args.hf_id.split("/")[-1].lower()
    out = Path(args.output_dir).resolve() if args.output_dir else COREAI_MODELS / "exports" / name
    export(args.hf_id, out, Path(args.qconfig))
    for tok in sorted(out.rglob("tokenizer")):
        fix_chat_eos(tok)
