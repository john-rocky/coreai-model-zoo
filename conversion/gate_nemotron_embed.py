# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b2",
#     "coreai-torch>=0.4.2",
#     "sentence-transformers>=5.4",
#     "transformers>=5.5",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
# START gate for nvidia/Nemotron-3-Embed-1B-BF16 — run this BEFORE writing a recipe.
#
# The candidate's stated kill condition: export the 16 layers with is_causal=false and
# compare against Hugging Face on 20 sentences. Below 0.999 cosine it is a new
# authoring-module job, the estimate is wrong, and the port is dropped.
#
# Three gates, cheapest first, so a failure names its own cause:
#
#   1. WRAPPER   the module chain reproduces SentenceTransformer.encode()   — no export
#   2. CONVERT   torch.export + coreai-torch produce a graph                — no runtime
#   3. NUMERIC   the Core AI graph matches Hugging Face on 20 sentences     — the gate
#   4. BLIND?    a WRONG pairing must fail the same threshold               — the guard
#
# Gate 1 failing means the wrapper is wrong, not the model. Gate 2 failing names the op.
# Only gate 3 is the candidate's kill condition. Running them in this order is what keeps
# a bad wrapper from being reported as an unportable model.
#
# This model is a decoder architecture (Ministral3) run BIDIRECTIONALLY: config.json says
# `is_causal: false`. That is the whole risk. Nothing else here is unusual — mean pooling
# then L2 normalize, the same three-stage chain as embeddinggemma-300m.
import argparse
import asyncio
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from coreai.runtime import AIModel, AIModelAssetMetadata, NDArray
from coreai_torch import TorchConverter, get_decomp_table
from sentence_transformers import SentenceTransformer

MODEL_NAME = "nvidia/Nemotron-3-Embed-1B-BF16"
PASS_COSINE = 0.999

# 20 sentences. Deliberately not all short and not all English: a bidirectional export
# that silently keeps a causal mask still scores well on short text, where the last token
# already sees most of the sentence. The long ones are where a wrong mask shows up.
TEXTS = [
    ("query", "what is the capital of Japan"),
    ("query", "how do I reset a bluetooth keyboard"),
    ("query", "red bicycle parked at the beach"),
    ("query", "difference between mean pooling and CLS pooling"),
    ("query", "東京から大阪までの新幹線の所要時間"),
    ("query", "why does my laptop fan run constantly"),
    ("query", "best way to store fresh basil"),
    ("query", "rope scaling in long context transformers"),
    ("document", "Tokyo is the capital and largest city of Japan, and the seat of its government."),
    ("document", "A crimson bike leaning against a palm tree by the sea."),
    ("document", "To reset a Bluetooth keyboard, hold the power button for ten seconds until the "
                 "indicator flashes, then remove the device from the host's paired list and pair "
                 "it again from scratch. On most models the pairing mode times out after three "
                 "minutes, so start the host-side scan first."),
    ("document", "Mean pooling averages every token's hidden state weighted by the attention "
                 "mask, so padding contributes nothing. CLS pooling reads a single position and "
                 "depends on that position having been trained to summarize the sequence, which "
                 "is true for BERT-style encoders and not for a decoder run bidirectionally."),
    ("document", "東海道新幹線のぞみは東京駅から新大阪駅までおよそ2時間30分で結ぶ。"),
    ("document", "A laptop fan that never stops is usually a background process pinning a core, "
                 "a blocked intake vent, or thermal paste that has dried out after several years "
                 "of use. Check the process list before opening the case."),
    ("document", "Basil wilts in the refrigerator. Trim the stems and stand them in a glass of "
                 "water on the counter, loosely covered, and it keeps for about a week."),
    ("document", "YaRN interpolates rotary position embeddings so that a model trained at one "
                 "context length attends sensibly at a longer one, scaling low and high frequency "
                 "bands by different factors rather than stretching all of them uniformly."),
    ("document", "The Ministral architecture uses grouped-query attention with eight key-value "
                 "heads shared across twenty-four query heads, which cuts the cache without "
                 "measurably moving retrieval quality at this size."),
    ("document", "Embeddings from this model are L2-normalized, so a dot product and a cosine "
                 "similarity give the same ranking."),
    ("document", "Padding tokens are excluded from the pooled vector by the attention mask; a "
                 "batch whose mask is wrong produces embeddings that drift with batch shape."),
    ("document", "静岡県の茶畑は富士山の南麓に広がっており、収穫は八十八夜のころに始まる。"),
]


class EmbeddingModule(torch.nn.Module):
    """The SentenceTransformer module chain, as a single exportable forward."""

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


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.flatten().float(),
                                                       b.flatten().float(), dim=0))


def report(name: str, values: list[float]) -> float:
    lo = min(values)
    print(f"[GATE {name}] n={len(values)}  min={lo:.6f}  mean={sum(values)/len(values):.6f}")
    return lo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    ap.add_argument("--keep", help="directory to keep the exported bundle in")
    args = ap.parse_args()

    print(f"[INFO] {MODEL_NAME}  seq_len={args.seq_len}  dtype={args.dtype}")
    # CPU throughout: torch.export traces on CPU, and a mixed CPU/MPS pipeline crashes.
    st = SentenceTransformer(MODEL_NAME, device="cpu")
    st.eval()
    cfg = st[0].auto_model.config
    print(f"[INFO] {type(st[0].auto_model).__name__}  layers={cfg.num_hidden_layers}  "
          f"is_causal={getattr(cfg, 'is_causal', 'unset')}  prompts={st.prompts}")
    assert getattr(cfg, "is_causal", None) is False, \
        "config is not bidirectional — the candidate's premise is wrong, stop here"

    module = EmbeddingModule(st).eval()
    # The checkpoint is bf16. Leaving it that way makes the graph's output bf16, and
    # numpy's DLPack has no bf16 — the run reaches the last line and dies there. The
    # fp32 export is also what a port wants first: match the oracle, then compress.
    if args.dtype == "float32":
        module = module.to(torch.float32)
    prompts = [prompted(st, kind, text) for kind, text in TEXTS]
    tokens = [tokenize(st, p, args.seq_len) for p in prompts]

    # --- Gate 1: the wrapper is the model -----------------------------------------
    hf = torch.tensor(st.encode(prompts, normalize_embeddings=True))
    with torch.no_grad():
        wrapped = torch.cat([module(i, m) for i, m in tokens], dim=0)
    wrapper_cos = [cos(hf[i], wrapped[i]) for i in range(len(TEXTS))]
    if report("1 WRAPPER", wrapper_cos) < PASS_COSINE:
        raise SystemExit("gate 1 failed: the wrapper does not reproduce encode(). "
                         "That is a bug here, not a verdict on the model.")

    # --- Gate 2: it converts ------------------------------------------------------
    ids, mask = tokens[0]
    example = {"input_ids": ids.clone(), "attention_mask": mask.clone()}
    if args.dtype == "float16":
        # Keep fp32 weights and let autocast put the matmuls in fp16 at trace time; a
        # full .to(float16) overflows activations on some architectures and emits NaN.
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            exported = torch.export.export(module, args=(), kwargs=example)
    else:
        exported = torch.export.export(module, args=(), kwargs=example)
    exported = exported.run_decompositions(get_decomp_table())
    program = TorchConverter().add_exported_program(
        exported_program=exported,
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
    ).to_coreai()
    program.optimize()
    print("[GATE 2 CONVERT] ok")

    # --- Gate 3: the graph matches Hugging Face -----------------------------------
    out_dir = Path(args.keep) if args.keep else Path(tempfile.mkdtemp())
    path = out_dir / f"nemotron-3-embed-1b_{args.dtype}_s{args.seq_len}.aimodel"
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = AIModelAssetMetadata()
    meta.author = "NVIDIA"
    meta.license = "OpenMDW-1.1"
    meta.model_description = (
        "Nemotron-3-Embed-1B bidirectional text embedding (mean pooling, L2-normalized "
        f"2048-d). START gate artifact, seq_len {args.seq_len}."
    )
    program.save_asset(path, meta)
    print(f"[INFO] saved {path}")

    async def run_all():
        # AIModel.load and InferenceFunction.__call__ are coroutines; calling them
        # synchronously returns a coroutine object that never runs, and the failure
        # surfaces as an AttributeError three lines later.
        model = await AIModel.load(str(path))
        fn = model.load_function("main")
        vecs = []
        for i, m in tokens:
            out = await fn({"input_ids": NDArray(i), "attention_mask": NDArray(m)})
            vecs.append(torch.as_tensor(out["embedding"].numpy()))
        return vecs

    vectors = asyncio.run(run_all())
    graph_cos = [cos(hf[idx], vectors[idx]) for idx in range(len(TEXTS))]
    worst = report("3 NUMERIC", graph_cos)

    # --- Gate 4: the gate can fail ------------------------------------------------
    # "worst cosine 1.000000 on all 20" is also what a comparison that pairs a vector
    # with itself prints. Pair every graph vector against the WRONG sentence's
    # reference: if those clear 0.999 too, the threshold is measuring nothing and the
    # PASS above is not evidence.
    n = len(TEXTS)
    off = [cos(hf[j], vectors[i]) for i in range(n) for j in range(n) if i != j]
    print(f"[GATE 4 BLIND?] mismatched pairs n={len(off)}  max={max(off):.6f}  "
          f"mean={sum(off)/len(off):.6f}")
    if max(off) >= PASS_COSINE:
        raise SystemExit("gate 4 failed: a wrong pairing also passes the threshold. "
                         "The numeric gate is blind; fix it before reading gate 3.")

    verdict = "PASS" if worst >= PASS_COSINE else "DROP"
    print(f"\n[VERDICT] {verdict} — worst cosine {worst:.6f} against {PASS_COSINE}")
    if verdict == "DROP":
        order = sorted(range(len(TEXTS)), key=lambda i: graph_cos[i])[:5]
        for i in order:
            print(f"   {graph_cos[i]:.6f}  {TEXTS[i][0]:8} {TEXTS[i][1][:70]}")
    print(json.dumps({"model": MODEL_NAME, "seq_len": args.seq_len, "dtype": args.dtype,
                      "wrapper_min": min(wrapper_cos), "graph_min": worst,
                      "mismatched_max": max(off),
                      "verdict": verdict}, indent=1))
    raise SystemExit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
