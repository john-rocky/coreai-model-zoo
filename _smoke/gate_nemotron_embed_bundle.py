# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b2",
#     "sentence-transformers>=5.4",
#     "transformers>=5.5",
#     "numpy",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
# Engine gate for the shipped Nemotron-3-Embed bundle: does the .aimodel that will
# actually be published reproduce the source model?
#
#   uv run gate_nemotron_embed_bundle.py <export-dir>
#
# The reference vectors in reference.json come from the UNCOMPRESSED fp32 model, so
# this compares the shipped bytes against the source rather than against themselves.
# That is the whole point of re-gating after compression: a compressed bundle checked
# against a compressed reference agrees with itself perfectly and says nothing.
#
# Three checks, and the third is the one that matters for an embedder:
#   1. COSINE    each vector vs the fp32 reference, >= 0.999
#   2. SELF      the bundle is deterministic across two runs of the same input
#   3. RANKING   query->document argmax matches the reference's argmax, with the
#                margin reported. An embedding can pass a cosine gate and still
#                reorder results, and the ranking is the product.
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import torch
from coreai.runtime import AIModel, NDArray
from sentence_transformers import SentenceTransformer

PASS_COSINE = 0.999


def cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


async def main() -> int:
    export_dir = Path(sys.argv[1])
    ref = json.loads((export_dir / "reference.json").read_text())
    bundles = sorted(export_dir.glob("*.aimodel"))
    if not bundles:
        raise SystemExit(f"no .aimodel in {export_dir}")
    bundle = bundles[0]
    seq_len = ref["seq_len"]
    print(f"[INFO] {bundle.name}  compression={ref['compression']}  seq_len={seq_len}")

    st = SentenceTransformer(ref["model"], device="cpu").eval()
    model = await AIModel.load(str(bundle))
    fn = model.load_function("main")

    async def embed(text: str):
        tok = st.tokenizer(text, padding="max_length", truncation=True,
                           max_length=seq_len, return_tensors="pt")
        out = await fn({"input_ids": NDArray(tok["input_ids"].to(torch.int32)),
                        "attention_mask": NDArray(tok["attention_mask"].to(torch.int32))})
        return out["embedding"].numpy()

    prompts = ref["prompts"] or {}
    texts = ref["texts"]
    got, failures = {}, []

    # --- 1. cosine against the fp32 reference --------------------------------------
    for key, spec in texts.items():
        prefix = prompts.get(spec["kind"], "")
        got[key] = await embed(prefix + spec["text"])
        c = cos(ref["vectors"][key], got[key])
        flag = "" if c >= PASS_COSINE else "   <-- FAIL"
        print(f"  [1 COSINE ] {key:18} {c:.6f}{flag}")
        if c < PASS_COSINE:
            failures.append(f"{key} cosine {c:.6f}")

    # --- 2. the bundle is deterministic ---------------------------------------------
    key0 = next(iter(texts))
    again = await embed(prompts.get(texts[key0]["kind"], "") + texts[key0]["text"])
    same = float(np.abs(np.asarray(again) - np.asarray(got[key0])).max())
    print(f"  [2 SELF   ] max abs delta across two runs = {same:.3e}")
    if same > 1e-6:
        failures.append(f"non-deterministic ({same:.3e})")

    # --- 3. ranking is preserved -----------------------------------------------------
    queries = [k for k, v in texts.items() if v["kind"] == "query"]
    docs = [k for k, v in texts.items() if v["kind"] == "document"]
    if queries and docs:
        for q in queries:
            mine = [cos(got[q], got[d]) for d in docs]
            theirs = [cos(ref["vectors"][q], ref["vectors"][d]) for d in docs]
            top_mine, top_theirs = docs[int(np.argmax(mine))], docs[int(np.argmax(theirs))]
            order = sorted(mine, reverse=True)
            margin = order[0] - order[1] if len(order) > 1 else float("nan")
            ok = top_mine == top_theirs
            print(f"  [3 RANKING] {q:18} -> {top_mine:16} "
                  f"margin {margin:+.4f} {'ok' if ok else '<-- REORDERED'}")
            if not ok:
                failures.append(f"{q} ranks {top_mine}, reference ranks {top_theirs}")

    print()
    if failures:
        print("BUNDLE GATE: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("BUNDLE GATE: PASS ✅")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
