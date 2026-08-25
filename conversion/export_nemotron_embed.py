# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b2",
#     "coreai-torch>=0.4.2",
#     "coreai-opt",
#     "sentence-transformers>=5.4",
#     "transformers>=5.5",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
# Export nvidia/Nemotron-3-Embed-1B-BF16 (transformer -> mean pooling -> L2 normalize)
# as a single static Core AI graph:
#   (input_ids [1,S] int32, attention_mask [1,S] int32) -> embedding [1, 2048]
#
# Why this model is in the catalog, in one line: it is the only embedder here that
# clearly beats the others on Japanese retrieval, and it is the first bidirectional
# (non-causal) one — every other embedder in this repo is a causal decoder.
#
# SHIP DTYPE IS int8 k-means, and that is a measured decision, not a default.
# On MIRACL-ja (250 queries, 8000 documents, hard negatives kept), paired bootstrap
# against the embeddinggemma-300m already published here:
#
#   fp32   0.8623   +0.0377  [+0.0181, +0.0578]  clear
#   int8   0.8618   +0.0372  [+0.0173, +0.0576]  clear      ~1.14 GB   <- ship
#   int4   0.8452   +0.0206  [-0.0008, +0.0425]  NOT sep.   ~705 MB
#
# int4 is refused on the interval, not the mean: it still looks ahead, but the CI
# crosses zero, so at 705 MB the reason to ship this model at all stops holding.
# top1 drops 0.784 -> 0.740 too, and top-1 is what a RAG pipeline actually hands to
# the generator. Reproduce with _smoke/compare_embedders_retrieval.py --palettize.
#
# Two things about the source worth knowing before reading the graph:
#   * config.json says `is_causal: false` — this is a decoder architecture run
#     bidirectionally, which is the whole risk of the port and the reason for gate 2.
#   * `apply_yarn_scaling: false` in rope_parameters is a key stock transformers
#     IGNORES. YaRN scaling at factor 16 is applied whatever it says. That is the
#     reference every user runs, so it is the behaviour this export matches.
import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table
from sentence_transformers import SentenceTransformer

MODEL_NAME = "nvidia/Nemotron-3-Embed-1B-BF16"

# Reference vectors for the Swift-side parity test. Query/document prompts are the
# model's own, from config_sentence_transformers.json.
REFERENCE_TEXTS = {
    "query_capital": ("query", "what is the capital of Japan"),
    "query_shinkansen": ("query", "東京から大阪までの新幹線の所要時間"),
    "doc_tokyo": ("document", "Tokyo is the capital and largest city of Japan."),
    "doc_shinkansen": ("document", "東海道新幹線のぞみは東京駅から新大阪駅までおよそ2時間30分で結ぶ。"),
}


class Fp16Table(torch.nn.Module):
    """An embedding whose table is stored fp16 and read back as fp32.

    k-means palettization is `F.linear`/`F.conv` only, so nn.Embedding comes through
    it untouched. For this model that is not a detail: the tied 131072x2048 table is
    24% of the parameters, and leaving it fp32 put the first export at 1947 MB —
    over the ~1.5 GB AOT shipping rule, and carrying four
    "Incompatible element type for ANE: expected fp16 ... si8" errors, because an
    fp32 table cannot be placed on the ANE at all.

    fp16 embed + int8 transformer is what this repo already ships for Qwen3.5
    (compression.md), and it is the reason to prefer it over a hand-written int8
    dequant-gather: same shape as a config that has shipped, one dtype, no custom op.
    """

    def __init__(self, emb: torch.nn.Embedding):
        super().__init__()
        self.register_buffer("table", emb.weight.data.to(torch.float16))
        self.padding_idx = emb.padding_idx

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(
            ids, self.table, padding_idx=self.padding_idx).to(torch.float32)


class EmbeddingModule(torch.nn.Module):
    """The SentenceTransformer module chain as one exportable forward."""

    def __init__(self, st: SentenceTransformer):
        super().__init__()
        self.stages = torch.nn.ModuleList(list(st))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        features = {"input_ids": input_ids, "attention_mask": attention_mask}
        for stage in self.stages:
            features = stage(features)
        return features["sentence_embedding"]


def prompted(st: SentenceTransformer, kind: str, text: str) -> str:
    prompts = st.prompts or {}
    return prompts.get(kind, "query: " if kind == "query" else "passage: ") + text


def tokenize(st: SentenceTransformer, text: str, seq_len: int):
    tok = st.tokenizer(text, padding="max_length", truncation=True,
                       max_length=seq_len, return_tensors="pt")
    return tok["input_ids"].to(torch.int32), tok["attention_mask"].to(torch.int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compression", choices=["int8", "none"], default="int8")
    ap.add_argument("--embed-dtype", choices=["fp16", "fp32"], default="fp16",
                    help="fp16 keeps the bundle under the ~1.5 GB AOT shipping rule")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    print(f"[INFO] Sourcing {MODEL_NAME}...")
    # CPU throughout: torch.export traces on CPU, and a mixed CPU/MPS pipeline crashes.
    st = SentenceTransformer(MODEL_NAME, device="cpu").eval()
    cfg = st[0].auto_model.config
    assert getattr(cfg, "is_causal", None) is False, "expected a bidirectional config"
    # The checkpoint is bf16. Export fp32 and let compression do the shrinking —
    # leaving it bf16 makes the graph's output bf16, which numpy's DLPack cannot read
    # and which no gate downstream can then check.
    module = EmbeddingModule(st).to(torch.float32).eval()

    ids, mask = tokenize(st, prompted(st, "query", "what is the capital of Japan"),
                         args.seq_len)

    # Reference vectors come from the UNCOMPRESSED model: the Swift test checks the
    # shipped bundle against the source model, not against itself.
    vectors = {}
    with torch.no_grad():
        for key, (kind, raw) in REFERENCE_TEXTS.items():
            i, m = tokenize(st, prompted(st, kind, raw), args.seq_len)
            vectors[key] = [float(x) for x in module(i, m)[0]]

    if args.compression == "int8":
        from coreai_opt.common import ExportBackend
        from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig

        print("[INFO] Palettizing (int8 k-means) — the gated ship config...")
        torch.manual_seed(0)          # vector k-means is non-deterministic
        palettizer = KMeansPalettizer(module.stages[0].auto_model,
                                      KMeansPalettizerConfig.presets.w8())
        palettizer.prepare((ids.long(), mask.long()), num_workers=1)
        module.stages[0].auto_model = palettizer.finalize(backend=ExportBackend.CoreAI)

    if args.embed_dtype == "fp16":
        inner = module.stages[0].auto_model
        holder = getattr(inner, "model", inner)
        emb = holder.embed_tokens
        holder.embed_tokens = Fp16Table(emb)
        mb = emb.weight.numel() * 2 / 1e6
        print(f"[INFO] Embedding table -> fp16 ({emb.weight.shape[0]}x{emb.weight.shape[1]}, "
              f"{mb:.0f} MB)")

    print("[INFO] Exporting...")
    exported = torch.export.export(
        module, args=(), kwargs={"input_ids": ids.clone(), "attention_mask": mask.clone()})
    exported = exported.run_decompositions(get_decomp_table())

    program = TorchConverter().add_exported_program(
        exported_program=exported,
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
    ).to_coreai()
    program.optimize()

    out_dir = Path(args.output_dir)
    tag = ("int8" if args.compression == "int8" else "float32")
    tag += "-embfp16" if args.embed_dtype == "fp16" else ""
    model_path = out_dir / f"nemotron-3-embed-1b_{tag}_s{args.seq_len}_static.aimodel"
    if model_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{model_path} exists; pass --overwrite")
        shutil.rmtree(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    meta = AIModelAssetMetadata()
    meta.author = "NVIDIA"
    # OpenMDW-1.1, not Apache. Built on Ministral-3-3B-Instruct-2512 (Apache-2.0).
    # Check the redistribution terms before publishing weights anywhere.
    meta.license = "OpenMDW-1.1"
    meta.model_description = (
        "Nemotron-3-Embed-1B bidirectional multilingual text embedding (mean pooling, "
        f"L2-normalized 2048-d), {tag}, sequence grid {args.seq_len}. "
        f"Source: https://huggingface.co/{MODEL_NAME}"
    )
    meta.creation_date = int(time.time())
    program.save_asset(model_path, meta)
    size_mb = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file()) / 1e6
    print(f"[INFO] Saved {model_path} ({size_mb:.0f} MB)")

    (out_dir / "reference.json").write_text(json.dumps({
        "model": MODEL_NAME,
        "seq_len": args.seq_len,
        "compression": args.compression,
        "prompts": st.prompts,
        "texts": {k: {"kind": v[0], "text": v[1]} for k, v in REFERENCE_TEXTS.items()},
        "vectors": vectors,
        "note": "vectors are from the uncompressed fp32 model; the bundle must match "
                "them to cos >= 0.999",
    }, indent=2, ensure_ascii=False))
    tok_dir = out_dir / "tokenizer"
    tok_dir.mkdir(exist_ok=True)
    st.tokenizer.save_pretrained(tok_dir)

    # OpenMDW-1.1 permits redistributing a quantized/converted copy — it is not
    # copyleft and carries no field-of-use restriction — on two conditions: ship a
    # copy of the agreement, and retain every notice of origin. Copying them here
    # rather than at publish time, because publish day is exactly when this is
    # forgotten and the result is a licence violation on a public repo.
    from huggingface_hub import snapshot_download
    src = Path(snapshot_download(MODEL_NAME, allow_patterns=[
        "LICENSE*", "NOTICE*", "THIRD_PARTY_NOTICES*", "README.md"]))
    copied = []
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
        found = list(src.glob(name + "*"))
        if found:
            shutil.copy2(found[0], out_dir / found[0].name)
            copied.append(found[0].name)
    if not copied:
        raise SystemExit("no LICENSE/NOTICE found upstream — do not publish this bundle "
                         "until the notices are located; OpenMDW-1.1 requires them")
    print(f"[INFO] Saved reference.json + tokenizer/ + {', '.join(copied)} to {out_dir}")


if __name__ == "__main__":
    main()
