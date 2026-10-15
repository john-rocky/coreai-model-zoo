#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.9.0",
#     "coreai-torch==0.4.1",
#     "coreai-core==1.0.0b2",
#     "safetensors>=0.7.0",
#     "numpy>=2.2",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
# Community port — NOT an Apple or Google model.
"""Export the EG2 transcribe reader + its word table to a static Core AI ``.aimodel`` (one token window per bundle).

    soft_tokens [1, T, 512] + token_mask [1, T]  ->  top1_index [1, T] int32, top1_logit [1, T], blank_logit [1, T]

T = 750 tokens is a 30 s window of the encoder (3000 frames), 250 a 10 s one. The graph is reader_coreai.ReaderGraph:
the reader of ``reader/train/model.py``, the N-word table as a constant, the argmax over N in the graph; the host
reads labels with ``reader_coreai.host_labels`` and collapses them.

Forms (``--form``):
    fp32        float32 weights, table and arithmetic; float32 inputs and outputs (int32 index)
    fp16        float16 weights, table and arithmetic (every norm divides by its row max |x| up front, LayerNorm
                written out); float16 inputs and outputs (int32 index)
    h-out       the float32 graph without the table: -> h [1, T, 768] (L2-normalized) + blank_logit; the host takes
                the table product and the argmax (reader_coreai.host_argmax)
    h-out-fp16  the same in float16

Steps: the reader (``reader_coreai.load_reader``: the release ``reader.pt`` or ``--checkpoint``) -> ReaderGraph ->
torch.export -> coreai decompositions -> gate the eager and the decomposed graph in torch on a few utterances
against ``ref_T<T>.npz`` (written by ``reader_coreai.py parity`` for the same weights) -> TorchConverter ->
optimize -> save_asset -> optionally ``coreai-build compile`` (macOS, h16c) for the GPU and the Neural Engine,
counting ANE regions by unique name, and ``coreai-build inspect``. Writes ``<out>/<form>/export_<name>.json``,
name = eg2_reader_<form>_T<T> for reader v1 and a --checkpoint, eg2_reader_<version>_<form>_T<T> for a later
release (``reader_config.json`` ``version``, e.g. eg2_reader_v2_fp16_T750). The body's depth (``architecture.layers``)
comes from the same config.

``--random-arch b|c`` exports a reader of that architecture with random weights (seed 0) to see whether the
converter takes its ops; no gate, no AOT; json under ``<out>/random_arch/``.

Run with the shared Core AI venv:
    <venv>/bin/python conversion/eg2_transcribe/export_reader.py --form fp32 --tokens 750 --aot gpu,neural-engine \\
        --checkpoint <run>/best_model.pt
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reader_coreai as RC  # noqa: E402
from _common import write_json  # noqa: E402
from export_encoder import aot_compile, coreai_build, du_bytes, inspect, op_counts  # noqa: E402

FORMS = {"fp32": (torch.float32, "argmax"), "fp16": (torch.float16, "argmax"),
         "h-out": (torch.float32, "h"), "h-out-fp16": (torch.float16, "h")}
LICENSE = "Apache-2.0; word table from google/embeddinggemma-2 text embeddings (Gemma Prohibited Use Policy)"
TRAINED_ON = {"v1": "LibriSpeech train-clean-100", "v2": "LibriSpeech train-clean-100, train-clean-360 and train-other-500"}


def author(version: str | None) -> str:
    data = TRAINED_ON.get(version or "", "LibriSpeech")
    return (f"coreai-model-zoo community port; reader trained with CTC on {data} (CC BY 4.0); "
            "word table: EmbeddingGemma 2 (Google DeepMind) embeddings of the LibriSpeech lexicon")


def example_inputs(tokens: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {"soft_tokens": torch.zeros(1, tokens, RC.D_IN, dtype=dtype), "token_mask": torch.ones(1, tokens, dtype=dtype)}


def output_names(head: str) -> list[str]:
    return ["top1_index", "top1_logit", "blank_logit"] if head == "argmax" else ["h", "blank_logit"]


def gate_torch(module, R: dict, utts: list[dict], ref, tokens: int, dtype: torch.dtype, head: str) -> dict:
    """Labels of the graph (host rule) vs the reference labels of ref_T<T>.npz on the given utterances."""
    N = len(R["vocab"])
    table_t = R["table"].numpy().T.copy() if head == "h" else None
    rows = []
    with torch.no_grad():
        for u in utts:
            key = f"{u['set']}/{u['id']}"
            labels, top_d, blank_d = [], 0.0, 0.0
            for a, b in RC.pieces(u["x"].shape[0], tokens):
                x, m = RC.window(u["x"], a, b, tokens)
                out = module(soft_tokens=torch.from_numpy(x).to(dtype), token_mask=torch.from_numpy(m).to(dtype))
                if head == "h":
                    h, bl = out[0][0, :b - a].float().numpy(), out[1][0, :b - a].float().numpy()
                    idx, best = RC.host_argmax(h, table_t)
                    top = best * R["scale"]
                else:
                    idx, top, bl = (o[0, :b - a].float().numpy() if i else o[0, :b - a].numpy()
                                    for i, o in enumerate(out))
                labels.append(RC.host_labels(idx, top, bl, N))
                top_d = max(top_d, float(np.abs(top - ref[f"{key}/top1_logit"][a:b]).max()))
                blank_d = max(blank_d, float(np.abs(bl - ref[f"{key}/blank_logit"][a:b]).max()))
            lab = np.concatenate(labels)
            rl = ref[f"{key}/labels"].astype(np.int64)
            rows.append({"set": u["set"], "id": u["id"], "tokens": int(lab.size), "tokens_same": int((lab == rl).sum()),
                         "transcript_same": RC.collapse(lab, N) == RC.collapse(rl, N),
                         "max_abs_top1_logit": top_d, "max_abs_blank_logit": blank_d})
    tok = sum(r["tokens"] for r in rows)
    return {"utterances": len(rows), "transcripts_same": sum(r["transcript_same"] for r in rows),
            "token_agreement": sum(r["tokens_same"] for r in rows) / max(1, tok), "tokens": tok,
            "max_abs_top1_logit": max(r["max_abs_top1_logit"] for r in rows),
            "max_abs_blank_logit": max(r["max_abs_blank_logit"] for r in rows), "rows": rows}


def random_reader(arch: str, args) -> dict:
    """A Reader of `arch` with random weights (seed 0) + the real table: does the converter take its ops?"""
    rdir = RC.reader_dir(args.reader_dir)
    M, Dc = RC.repo_modules(rdir)
    with open(rdir / "vocab.txt", encoding="utf-8") as f:
        n_vocab = sum(1 for line in f if line.strip())
    torch.manual_seed(0)
    layers = (args.layers or 4) if arch == "a" else None
    model = M.Reader(arch, n_vocab, k=args.k or 4, layers=layers or 4).eval()
    table = Dc.load_table(str(rdir / "word_table.npy"), "cpu")
    source = {"kind": f"random weights (seed 0), arch {arch}", "arch": arch, "k": args.k or 4 if arch == "b" else None,
              "layers": layers, "params": sum(p.numel() for p in model.parameters()), "n_vocab": n_vocab,
              "model_py_sha256": RC.sha256_of(rdir / "train" / "model.py")}
    return {"model": model, "table": table, "vocab": [None] * n_vocab, "arch": arch, "k": args.k, "layers": layers,
            "scale": float(model.scale.detach()), "source": source}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--form", choices=tuple(FORMS), default="fp32")
    ap.add_argument("--tokens", type=int, choices=RC.TOKEN_BUCKETS, required=True, help="tokens per window")
    ap.add_argument("--aot", default="", help="comma list of gpu,neural-engine: also compile h16c AOT bundles")
    ap.add_argument("--arch-aot", default="h16c", help="AOT architecture (h16c = M4-class Mac)")
    ap.add_argument("--aot-only", action="store_true",
                    help="compile the bundle a previous run saved (exports are not byte-reproducible)")
    ap.add_argument("--random-arch", choices=("a", "b", "c"),
                    help="export a random-weight reader of this arch (converter check only)")
    RC.add_reader_args(ap)
    ap.add_argument("--fixtures")
    ap.add_argument("--cache")
    ap.add_argument("--gate-utterances", type=int, default=0,
                    help="utterances in the torch gate (default: 8 for float32 forms, 2 for float16)")
    ap.add_argument("--out-dir", help="default <work>/exports/reader (reader v1), <work>/exports/reader_<version> "
                                         "(a later release, e.g. reader_v2); the bundle goes to <out>/<form>/")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dtype, head = FORMS[args.form]
    tag = "" if args.random_arch else RC.config_tag(args)
    od = RC.out_dir(args.out_dir, tag)

    if args.random_arch:
        name = f"eg2_reader_random{args.random_arch}_{args.form}_T{args.tokens}"
        form_dir = od / "random_arch"
    else:
        name = RC.bundle_stem(args.form, args.tokens, tag)
        form_dir = od / args.form
    form_dir.mkdir(parents=True, exist_ok=True)
    bundle = form_dir / f"{name}.aimodel"
    rec_path = form_dir / f"export_{name}.json"
    if args.aot_only:
        record = json.loads(rec_path.read_text())
        assert Path(record["bundle"]) == bundle and bundle.exists(), bundle
        t0 = time.time()
        compile_all(args, bundle, name, form_dir, record)
        record["aot_only_seconds"] = round(time.time() - t0, 1)
        write_json(rec_path, record)
        return

    t0 = time.time()
    R = random_reader(args.random_arch, args) if args.random_arch else RC.load_reader(args)
    assert args.random_arch or RC.name_tag(R) == tag, (RC.name_tag(R), tag)
    record: dict = {"form": args.form, "tokens": args.tokens, "window_s": RC.WINDOW_SECONDS[args.tokens],
                    "name": name, "name_tag": tag,
                    "head": head, "io_dtype": str(dtype).replace("torch.", ""), "reader": R["source"],
                    "norms": "max-abs rescaled, LayerNorm written out (float16)" if dtype == torch.float16 else "exact",
                    "software": {"python": platform.python_version(), "torch": torch.__version__},
                    "inputs": {"soft_tokens": [1, args.tokens, RC.D_IN], "token_mask": [1, args.tokens]},
                    "outputs": ({"top1_index": [1, args.tokens], "top1_logit": [1, args.tokens],
                                 "blank_logit": [1, args.tokens]} if head == "argmax" else
                                {"h": [1, args.tokens, RC.D_OUT], "blank_logit": [1, args.tokens]})}
    for pkg in ("coreai-core", "coreai-torch", "coreai-opt"):
        try:
            from importlib.metadata import version
            record["software"][pkg] = version(pkg)
        except Exception:  # noqa: BLE001
            record["software"][pkg] = None
    model = RC.build_graph(R, args.tokens, dtype, head=head)

    gate_utts, ref = [], None
    if not args.random_arch:
        utts = RC.load_utterances(RC.fixtures_dir(args.fixtures), RC.cache_dir(args.cache))
        k = args.gate_utterances or (2 if dtype == torch.float16 else 8)
        gate_utts = [u for u in utts if u["set"] == "fixture"][:(k + 1) // 2] + \
                    [u for u in utts if u["set"] == "test-clean"][:k // 2]
        ref_path = od / f"ref_T{args.tokens}.npz"
        ref = np.load(ref_path)
        if str(ref["meta/weights_sha256"]) != R["source"]["weights_sha256"]:
            raise SystemExit(f"{ref_path} is for other weights: run reader_coreai.py parity for these weights before")
        record["eager_gate"] = gate_torch(model, R, gate_utts, ref, args.tokens, dtype, head)
        print(f"[{name}] eager gate: transcripts {record['eager_gate']['transcripts_same']}/"
              f"{record['eager_gate']['utterances']}, tokens {record['eager_gate']['token_agreement']:.6f} "
              f"({time.time() - t0:.0f} s)", flush=True)

    from coreai.runtime import AIModelAssetMetadata
    from coreai_torch import TorchConverter, get_decomp_table

    with torch.no_grad():
        exported = torch.export.export(model, args=(), kwargs=example_inputs(args.tokens, dtype))
        decomposed = exported.run_decompositions(get_decomp_table())
    if gate_utts:
        record["decomposed_gate"] = gate_torch(decomposed.module(), R, gate_utts, ref, args.tokens, dtype, head)
        print(f"[{name}] decomposed gate: transcripts {record['decomposed_gate']['transcripts_same']}/"
              f"{record['decomposed_gate']['utterances']}, tokens {record['decomposed_gate']['token_agreement']:.6f} "
              f"({time.time() - t0:.0f} s)", flush=True)

    t1 = time.time()
    try:
        program = TorchConverter().add_exported_program(
            exported_program=decomposed, input_names=["soft_tokens", "token_mask"], output_names=output_names(head)
        ).to_coreai()
    except Exception as exc:  # noqa: BLE001 - the failing op is the finding; recorded, then re-raised
        record["convert_error"] = repr(exc)[-2000:]
        write_json(rec_path, record)
        raise
    record["convert_s"] = round(time.time() - t1, 1)
    t1 = time.time()
    program.optimize()
    record["optimize_s"] = round(time.time() - t1, 1)
    record["op_counts"] = op_counts(program)
    meta = AIModelAssetMetadata()
    version = (R["source"].get("config") or {}).get("version")
    meta.author = author(version)
    meta.license = LICENSE
    body = f"arch {R['arch']}" + (f", {R['layers']} layers" if R.get("layers") else "")
    meta.model_description = (f"EG2 transcribe reader {version + ' ' if version else ''}({body}, "
                              f"{R['source']['params']:,} parameters, "
                              f"scale {R['scale']:.4f}) + {len(R['vocab']):,}-word table: soft tokens "
                              f"[1,{args.tokens},512] + token_mask -> "
                              + ("top-1 word index / logit, blank logit" if head == "argmax" else
                                 "L2-normalized vectors [1,{},768], blank logit".format(args.tokens))
                              + f"; {args.form}; weights {R['source'].get('weights_sha256', 'random')[:12]}")
    assert meta.author.startswith("coreai-model-zoo") and meta.license == LICENSE and meta.model_description
    shutil.rmtree(bundle, ignore_errors=True)          # save_asset does not overwrite
    t1 = time.time()
    program.save_asset(bundle, meta)
    record["save_s"] = round(time.time() - t1, 1)
    record["bundle"] = str(bundle)
    record["bytes"] = du_bytes(bundle)
    record["main_hash"] = (bundle / "main.hash").read_bytes().hex() if (bundle / "main.hash").exists() else None
    print(f"[{name}] saved {bundle} {record['bytes'] / 1e6:.1f} MB ({time.time() - t0:.0f} s)", flush=True)
    del program, decomposed, exported

    if not args.random_arch:
        compile_all(args, bundle, name, form_dir, record)
    record["seconds"] = round(time.time() - t0, 1)
    record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(rec_path, record)
    print(json.dumps({k: record[k] for k in ("form", "tokens", "bytes", "seconds")}))


def compile_all(args, bundle: Path, name: str, form_dir: Path, record: dict) -> None:
    cb = coreai_build()
    record["coreai_build"] = subprocess.run([cb, "--version"], capture_output=True, text=True).stdout.strip()
    record["inspect_source"] = inspect(cb, bundle, form_dir / f"inspect_{name}.json")
    record.setdefault("aot", {})
    for compute in [c for c in args.aot.split(",") if c]:
        tag = {"gpu": "gpu", "neural-engine": "ane"}[compute]
        aot = form_dir / f"{name}_{tag}_{args.arch_aot}.aimodelc"
        rec = aot_compile(cb, bundle, aot, compute, args.arch_aot)
        if rec["returncode"] == 0:
            rec["inspect"] = inspect(cb, aot, form_dir / f"inspect_{name}_{tag}_{args.arch_aot}.json")
        record["aot"][tag] = rec
        print(f"[{name}] AOT {compute}: rc {rec['returncode']} {rec['seconds']} s, "
              f"ANE regions {rec.get('ane_regions')}, {rec.get('bytes', 0) / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
