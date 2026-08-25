# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "sentence-transformers>=5.4",
#     "transformers>=5.5",
#     "pyarrow",
#     "numpy",
#     "torch",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
# SHIP_RUBRIC 3.2 for embedders: is a candidate measurably better than what this
# repo already publishes?
#
# The rubric's most frequent kill reason is "technically correct but marginal",
# and an embedder is the easy case to get wrong: every one of them passes its
# port gate at cos 0.9999, and none of that says whether it retrieves better
# than the one already shipped.
#
# Matched protocol, which for embedders means each model runs ITS OWN documented
# recipe — prefixes, pooling and dimension are part of the model, not knobs to
# equalize. What is held identical is the corpus, the queries, the qrels, the
# metric and the sequence budget.
#
#   uv run compare_embedders_retrieval.py --datasets NanoSciFact
#
# Quality is measured on the native torch runtime on purpose. It is an attribute
# of the model; the Core AI port's job is to preserve it, and that is what the
# port gate measures separately.
import argparse
import glob
import json
import math
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

# id -> (query prefix, document prefix). Taken from each model's own card /
# config_sentence_transformers.json, not invented here.
MODELS = {
    "nvidia/Nemotron-3-Embed-1B-BF16": ("query: ", "passage: "),
    "Qwen/Qwen3-Embedding-0.6B": (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        "",
    ),
    "google/embeddinggemma-300m": ("task: search result | query: ", "title: none | text: "),
}


# short name -> (hub repo id, config prefix). Every one of these publishes the same
# three-config BEIR layout (queries / corpus / qrels); only the split filename and
# the config prefix differ, so one loader covers them.
DATASETS = {
    "NanoSciFact":  ("zeta-alpha-ai/NanoSciFact", ""),
    "NanoNFCorpus": ("zeta-alpha-ai/NanoNFCorpus", ""),
    "NanoFiQA2018": ("zeta-alpha-ai/NanoFiQA2018", ""),
    "MIRACL-ja":    ("mteb/MIRACLRetrievalHardNegatives", "ja-"),
    "JaQuAD":       ("mteb/JaQuADRetrieval", ""),
}


def load_retrieval(name: str):
    if name not in DATASETS:
        raise SystemExit(f"unknown dataset {name}; known: {', '.join(DATASETS)}")
    repo, prefix = DATASETS[name]
    cache = f"datasets--{repo.replace('/', '--')}"
    dirs = glob.glob(f"{glob.escape(str(HUB))}/{cache}/snapshots/*/")
    if not dirs:
        raise SystemExit(
            f"{name} is not in the hub cache. Fetch it first:\n"
            f"  hf download {repo} --repo-type dataset"
        )
    base = dirs[0]

    def read(part):
        # Split filenames differ across these repos (train / dev / validation), so
        # take whatever parquet the config directory holds rather than guessing.
        files = sorted(glob.glob(f"{glob.escape(base)}{prefix}{part}/*.parquet"))
        if not files:
            raise SystemExit(f"{name}: no parquet under {prefix}{part}/ in {base}")
        tables = [pq.read_table(f).to_pydict() for f in files]
        merged = {k: [v for t in tables for v in t[k]] for k in tables[0]}
        return merged

    q, c, r = read("queries"), read("corpus"), read("qrels")
    idcol = lambda t: "_id" if "_id" in t else "id"   # mteb/ and zeta-alpha-ai/ differ
    text_col = "text" if "text" in c else "body"
    # Some BEIR corpora carry a title column that belongs in the embedded text.
    if "title" in c:
        corpus = {str(i): ((t + " " + b).strip() if t else b)
                  for i, t, b in zip(c[idcol(c)], c["title"], c[text_col])}
    else:
        corpus = dict(zip(map(str, c[idcol(c)]), c[text_col]))
    queries = dict(zip(map(str, q[idcol(q)]), q["text"]))
    qrels: dict[str, set[str]] = {}
    score_col = "score" if "score" in r else None
    for idx, (qid, cid) in enumerate(zip(map(str, r["query-id"]), map(str, r["corpus-id"]))):
        if score_col and r[score_col][idx] <= 0:
            continue                      # explicit non-relevant judgement
        qrels.setdefault(qid, set()).add(cid)
    return queries, corpus, qrels


def subset(queries, corpus, qrels, max_queries, corpus_pool, seed=0):
    """Cut a full-size collection down to something two 1B models can encode twice.

    Keeps every judged document for the sampled queries and fills the rest of the
    pool with random distractors. The resulting score is NOT comparable to a
    published number on the full collection — it is only comparable between the
    models measured here, which is the question this script exists to answer.
    """
    rng = np.random.default_rng(seed)
    judged = [q for q in queries if qrels.get(q)]
    if max_queries and len(judged) > max_queries:
        judged = [judged[i] for i in rng.choice(len(judged), max_queries, replace=False)]
    queries = {q: queries[q] for q in judged}
    qrels = {q: qrels[q] for q in judged}
    # Only ever touch the corpus when a pool size was actually asked for. The
    # earlier version fell through to "keep = judged docs" when corpus_pool was 0
    # and silently shrank NanoSciFact from 2919 documents to 55 — every model then
    # scored above 0.94 with recall@10 of exactly 1.0, because the only documents
    # left to rank were the answers.
    if corpus_pool and len(corpus) > corpus_pool:
        keep = set().union(*qrels.values()) & set(corpus)
        rest = [d for d in corpus if d not in keep]
        extra = rng.choice(len(rest), max(0, corpus_pool - len(keep)), replace=False)
        keep |= {rest[i] for i in extra}
        corpus = {d: corpus[d] for d in keep}
    return queries, corpus, qrels


def assert_retrieval_is_hard(name, corpus, qrels, full_corpus_size):
    """Refuse to score a collection whose distractors THIS SCRIPT removed.

    A retrieval number is only meaningful when the relevant documents are a small
    part of what has to be searched. If the corpus collapses to roughly the judged
    set, every model returns everything and the scores converge near 1.0 — which
    reads exactly like all the models being excellent.

    The distinction that matters is who made it easy. A dataset that ships a small
    corpus is a weak discriminator and is reported as one; a corpus this script
    shrank is a bug, and that is the case worth refusing. Conflating the two would
    reject standard collections (JaQuAD is natively 4x) while still permitting the
    subsetting mistake on a large one.
    """
    judged = set().union(*qrels.values()) if qrels else set()
    ratio = len(corpus) / max(len(judged), 1)
    shrunk = len(corpus) < full_corpus_size
    if ratio < 10 and shrunk:
        raise SystemExit(
            f"{name}: subsetting left {len(corpus)} of {full_corpus_size} documents "
            f"for {len(judged)} judged ones ({ratio:.1f}x). The distractors are gone "
            f"and every model will score near 1.0. Pass --corpus-pool N with "
            f"N >= {10 * len(judged)}, or drop the subsetting flags."
        )
    if ratio < 10:
        print(f"  [weak] {name} ships {len(corpus)} documents for {len(judged)} judged "
              f"ones ({ratio:.1f}x) — a small separation, read the margins with that "
              f"in mind.")
    return ratio


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def evaluate(model_id, prefixes, queries, corpus, qrels, seq_len, device, batch):
    qp, dp = prefixes
    st = SentenceTransformer(model_id, device=device)
    st.max_seq_length = seq_len
    st.eval()

    qids = list(queries)
    cids = list(corpus)
    q_emb = st.encode([qp + queries[i] for i in qids], batch_size=batch,
                      normalize_embeddings=True, show_progress_bar=False)
    d_emb = st.encode([dp + corpus[i] for i in cids], batch_size=batch,
                      normalize_embeddings=True, show_progress_bar=False)
    scores = np.asarray(q_emb) @ np.asarray(d_emb).T          # unit vectors -> cosine

    ndcg10, recall10, top1, evaluated_qids = [], [], [], []
    for row, qid in enumerate(qids):
        rel = qrels.get(qid, set())
        if not rel:
            continue
        order = np.argsort(-scores[row])[:10]
        ranked = [cids[j] for j in order]
        evaluated_qids.append(qid)
        ndcg10.append(ndcg_at_k(ranked, rel, 10))
        recall10.append(len(set(ranked) & rel) / len(rel))
        top1.append(1.0 if ranked[0] in rel else 0.0)
    del st
    return {
        "ndcg@10": float(np.mean(ndcg10)),
        "recall@10": float(np.mean(recall10)),
        "top1": float(np.mean(top1)),
        "n_queries": len(ndcg10),
        "dim": int(np.asarray(q_emb).shape[1]),
        # Kept so the difference between two models can carry an interval. A mean
        # over 50 queries is a direction, not a result, and SHIP_RUBRIC 3.2 asks
        # for "clearly better" — which is a claim about the interval.
        "per_query_ndcg": [float(v) for v in ndcg10],
        "query_ids": evaluated_qids,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["NanoSciFact"])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-queries", type=int, default=0, help="0 = all")
    ap.add_argument("--corpus-pool", type=int, default=0, help="0 = full corpus")
    ap.add_argument("--out")
    args = ap.parse_args()

    results = {}
    for ds in args.datasets:
        queries, corpus, qrels = load_retrieval(ds)
        full_corpus_size = len(corpus)
        if args.max_queries or args.corpus_pool:
            queries, corpus, qrels = subset(queries, corpus, qrels,
                                            args.max_queries, args.corpus_pool)
        ratio = assert_retrieval_is_hard(ds, corpus, qrels, full_corpus_size)
        print(f"\n=== {ds}: {len(queries)} queries, {len(corpus)} docs "
              f"({ratio:.0f}x the judged set), "
              f"{sum(len(v) for v in qrels.values())} qrels "
              f"(seq_len {args.seq_len}, {args.device}) ===")
        print(f"{'model':38} {'dim':>5} {'nDCG@10':>8} {'R@10':>7} {'top1':>7}")
        for mid in args.models:
            if mid not in MODELS:
                raise SystemExit(f"no documented prefixes for {mid}; add them to MODELS")
            r = evaluate(mid, MODELS[mid], queries, corpus, qrels,
                         args.seq_len, args.device, args.batch)
            results.setdefault(ds, {})[mid] = r
            print(f"{mid:38} {r['dim']:5d} {r['ndcg@10']:8.4f} "
                  f"{r['recall@10']:7.4f} {r['top1']:7.4f}")

    # Paired bootstrap on the per-query nDCG differences. Paired because both models
    # answer the same queries, so the query-difficulty variance cancels.
    for ds, per_model in results.items():
        ids = {m: r["query_ids"] for m, r in per_model.items()}
        first = next(iter(ids.values()))
        if not all(v == first for v in ids.values()):
            print(f"[{ds}] query sets differ between models — skipping the interval")
            continue
        best = max(per_model, key=lambda m: per_model[m]["ndcg@10"])
        rng = np.random.default_rng(0)
        n = len(first)
        print(f"\n[{ds}] paired bootstrap vs {best.split('/')[-1]} "
              f"(10k resamples, n={n} queries)")
        for mid, r in per_model.items():
            if mid == best:
                continue
            d = np.asarray(per_model[best]["per_query_ndcg"]) - np.asarray(r["per_query_ndcg"])
            idx = rng.integers(0, n, size=(10000, n))
            boot = d[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            verdict = "clear" if lo > 0 else "NOT separated at 95%"
            print(f"  vs {mid.split('/')[-1]:32} delta nDCG@10 = {d.mean():+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]  {verdict}")
            r["delta_vs_best"] = {"mean": float(d.mean()), "ci95": [float(lo), float(hi)],
                                  "separated": bool(lo > 0)}

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"seq_len": args.seq_len, "device": args.device,
                       "results": results}, f, indent=1)
    print("\nSHIP_RUBRIC 3.2: a candidate that does not clearly beat the shipped rows "
          "above is KEEP-LOCAL, not SHIP.")


HUB = __import__("pathlib").Path.home() / ".cache/huggingface/hub"

if __name__ == "__main__":
    main()
