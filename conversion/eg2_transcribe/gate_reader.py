#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.9.0",
#     "coreai-core==1.0.0b2",
#     "safetensors>=0.7.0",
#     "numpy>=2.2",
#     "jiwer==4.0.0",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
# Community port — NOT an Apple or Google model.
"""Gate an exported reader bundle on the Core AI Python runtime against the torch float32 reference.

Backends (``--backend``), each with the bundle it is meant for:
    cpu  the JIT ``.aimodel`` with ``SpecializationOptions.cpu_only()``
    gpu  the h16c GPU AOT ``.aimodelc`` with ``SpecializationOptions.default()``
    ane  the h16c Neural Engine AOT ``.aimodelc`` with the Neural Engine preferred; its ANE regions are counted
         by unique name (0 = the graph silently runs on the GPU)

Inputs (``reader_coreai.load_utterances``): the 20 fixture utterances (float32 reference soft tokens) and the
200 leading test-clean utterances of the soft token cache (float16 -> float32), cut into windows of T tokens
(``reader_coreai.pieces``; at T = 750 every utterance is one window), right-zero-padded, with token_mask.
Reference = ``ref_T<T>.npz`` of ``reader_coreai.py parity``: the reader's own forward + greedy_labels on the same
windows (float32, CPU), for the same weights (sha256 checked against the bundle's export json).

Measured: labels (host rule ``reader_coreai.host_labels``) per token vs the reference labels; transcripts
(collapsed) per utterance; WER of both paths against the LibriSpeech transcripts (word Levenshtein; ``score``
re-counts with jiwer); max |d| of top1_logit / blank_logit; argmax index agreement; for each label mismatch how
close the reference was to a tie (top-1 minus top-2 cosine, blank minus top-1 logit). h-out forms: per-token
cosine of h vs the reference h, the read-out on the host (``reader_coreai.host_argmax``). Every window runs
twice (drift). Two red arms on the 10 shortest fixture utterances, so the gate can fail: token_mask all ones
(attention sees the padding; reader v1 learned to ignore zero padding, so at T = 250 this arm moves no label), and
the second half of each utterance's valid soft tokens set to 0 with token_mask unchanged (red = the labels change on
>= 9 of the 10).

Bars (registered by the lane supervisor before the run):
    float32 forms (fp32, h-out)       fixture: 20/20 transcripts identical to torch; test-clean 200: token
                                       agreement >= 99.9 %
    float16 forms (fp16, h-out-fp16)  test-clean 200: WER <= torch WER + 0.2 pt; fixture: >= 19/20 transcripts
PASS = the bar, label drift 0 and a red arm that fires (token_mask: any label changes; zero-half: >= 9 of 10).

Bundle names: eg2_reader_[<version>_]<form>_T<T>[_<backend>_<arch>].aimodel[c] (reader v1 has no version part,
reader v2 is eg2_reader_v2_fp16_T750...); the export json beside the bundle is export_eg2_reader_[<version>_]<form>_T<T>.json.

Writes ``<bundle dir>/gate_<backend>[_jit]_T<T>.json`` and ``hyp_<backend>[_jit]_T<T>.tsv``. Run ``gate`` inside a
measurement window; ``warm`` (the initial load + one call) before it for a Neural Engine bundle, outside the window;
``score`` needs jiwer (the training venv has it), not Core AI:
    <venv>/bin/python conversion/eg2_transcribe/gate_reader.py gate --bundle <x>_gpu_h16c.aimodelc --backend gpu
    <train venv>/bin/python conversion/eg2_transcribe/gate_reader.py score --gate <gate json>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reader_coreai as RC  # noqa: E402
from _common import write_json  # noqa: E402

FP16_FORMS = ("fp16", "h-out-fp16")
BAR_TOKEN_AGREEMENT = 0.999      # float32 forms, test-clean 200
BAR_WER_DELTA = 0.002            # float16 forms: WER <= torch WER + 0.2 pt
BAR_FIXTURE_FP16 = 19            # float16 forms: fixture transcripts identical
RED_ARM_UTTERANCES = 10
RED_ZERO_HALF_MIN = 9            # the zero-half arm is red when the labels change on >= 9 of the 10 utterances
SPECIALIZATION = {"cpu": "cpu_only()", "gpu": "default()", "ane": "from_preferred_compute_unit_kind(neural_engine)"}
NAME = re.compile(r"eg2_reader_(?:(v\d+)_)?(.+?)_T(\d+)(?:_|\.aimodel)")


def parse_name(bundle: Path) -> tuple[str, int]:
    """-> (form, tokens) of a reader bundle name; the optional version part (v2, ...) is not part of the form."""
    m = NAME.match(bundle.name)
    if not m:
        raise SystemExit(f"cannot read form and tokens from {bundle.name}")
    return m.group(2), int(m.group(3))


def export_record(bundle: Path) -> Path:
    """export_eg2_reader_[<version>_]<form>_T<T>.json beside the bundle (written by export_reader.py)."""
    m = NAME.match(bundle.name)
    if not m:
        raise SystemExit(f"cannot read form and tokens from {bundle.name}")
    return bundle.parent / f"export_{bundle.name[:m.end(3)]}.json"


def zero_half(utts: list[dict]) -> list[dict]:
    """The zero-half red arm's inputs: each utterance's second half of valid soft tokens set to 0 (mask unchanged)."""
    out = []
    for u in utts:
        x = np.array(u["x"], dtype=np.float32, copy=True)
        x[x.shape[0] // 2:] = 0
        out.append({**u, "x": x})
    return out


def load_vocab(rdir: Path) -> list[str]:
    with open(rdir / "vocab.txt", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def load_table_t(path: Path) -> np.ndarray:
    """word_table.npy -> float32, rows L2-normalized (decode.load_table in numpy), transposed [768, N]."""
    t = np.load(path).astype(np.float32)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    return np.ascontiguousarray(t.T)


async def run_windows(fn, rt, utts, T: int, dtype, head: str, ones_mask: bool = False) -> list[list[dict]]:
    """Every window of every utterance through the function. -> per utterance, per window: outputs + call s."""
    out = []
    for u in utts:
        rows = []
        for a, b in RC.pieces(u["x"].shape[0], T):
            x, m = RC.window(u["x"], a, b, T, dtype)
            if ones_mask:
                m[:] = 1
            feed = {"soft_tokens": rt.NDArray(np.ascontiguousarray(x)), "token_mask": rt.NDArray(np.ascontiguousarray(m))}
            t0 = time.perf_counter()
            res = await asyncio.wait_for(fn(feed), timeout=900)
            s = time.perf_counter() - t0
            n = b - a
            if head == "argmax":
                rows.append({"idx": res["top1_index"].numpy()[0, :n].astype(np.int32),
                             "top1": res["top1_logit"].numpy()[0, :n].astype(np.float32),
                             "blank": res["blank_logit"].numpy()[0, :n].astype(np.float32), "s": s})
            else:
                rows.append({"h": res["h"].numpy()[0, :n].astype(np.float32),
                             "blank": res["blank_logit"].numpy()[0, :n].astype(np.float32), "s": s})
        out.append(rows)
    return out


def read_out(rows: list[dict], head: str, table_t, scale: float, n_vocab: int) -> dict:
    """Concatenate an utterance's windows and apply the host rule (h-out: after the host argmax)."""
    blank = np.concatenate([r["blank"] for r in rows])
    if head == "argmax":
        idx = np.concatenate([r["idx"] for r in rows])
        top = np.concatenate([r["top1"] for r in rows])
        h = None
    else:
        h = np.concatenate([r["h"] for r in rows])
        idx, best = RC.host_argmax(h, table_t)
        top = best * np.float32(scale)
    return {"idx": idx, "top1": top, "blank": blank, "h": h, "labels": RC.host_labels(idx, top, blank, n_vocab)}


async def gate(args) -> None:
    import coreai.runtime as rt
    from gate_encoder import ane_regions, options, runtime_info

    bundle = Path(args.bundle).expanduser().resolve()
    form, T = parse_name(bundle)
    head = "h" if form.startswith("h-out") else "argmax"
    dtype = np.float16 if form in FP16_FORMS else np.float32
    if args.backend == "cpu" and bundle.suffix != ".aimodel":
        raise SystemExit("cpu gates the JIT .aimodel (an h16c AOT bundle cannot be specialized for the CPU)")
    if args.backend != "cpu" and bundle.suffix != (".aimodel" if args.jit else ".aimodelc"):
        raise SystemExit("gpu / ane gate the AOT .aimodelc (or the JIT .aimodel with --jit)")
    export_rec = json.loads(export_record(bundle).read_text())
    od = RC.out_dir(args.out_dir, NAME.match(bundle.name).group(1) or "")
    ref = np.load(od / f"ref_T{T}.npz")
    if str(ref["meta/weights_sha256"]) != export_rec["reader"]["weights_sha256"]:
        raise SystemExit("ref npz and bundle come from different reader weights")
    rdir = Path(export_rec["reader"]["reader_dir"]) if not args.reader_dir else RC.reader_dir(args.reader_dir)
    vocab = load_vocab(rdir)
    N = len(vocab)
    scale = float(export_rec["reader"]["scale"])
    table_t = load_table_t(Path(export_rec["reader"]["table"])) if head == "h" else None
    utts = RC.load_utterances(RC.fixtures_dir(args.fixtures), RC.cache_dir(args.cache))
    regions = ane_regions(bundle) if bundle.suffix == ".aimodelc" else []

    t0 = time.perf_counter()
    opts = (rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
            if (args.jit and args.backend == "gpu") else options(args.backend))
    model = await rt.AIModel.load(bundle, opts)
    fn = model.load_function("main")
    load_s = time.perf_counter() - t0
    desc = {"inputs": list(fn.desc.input_names), "outputs": list(fn.desc.output_names)}

    pass1 = await run_windows(fn, rt, utts, T, dtype, head)
    pass2 = await run_windows(fn, rt, utts, T, dtype, head)
    red_ix = sorted((i for i, u in enumerate(utts) if u["set"] == "fixture"),
                    key=lambda i: utts[i]["x"].shape[0])[:RED_ARM_UTTERANCES]
    red_utts = [utts[i] for i in red_ix]
    red = await run_windows(fn, rt, red_utts, T, dtype, head, ones_mask=True)
    red_zh = await run_windows(fn, rt, zero_half(red_utts), T, dtype, head)

    rows, mism, hyp_lines = [], [], []
    d_top = d_blank = drift_logit = 0.0
    drift_labels = idx_same = tok_total = 0
    cos_all = []
    refs_txt = {"fixture": [], "test-clean": []}
    hyps_torch = {"fixture": [], "test-clean": []}
    hyps_core = {"fixture": [], "test-clean": []}
    for u, w1, w2 in zip(utts, pass1, pass2):
        key = f"{u['set']}/{u['id']}"
        o1 = read_out(w1, head, table_t, scale, N)
        o2 = read_out(w2, head, table_t, scale, N)
        rl = ref[f"{key}/labels"].astype(np.int64)
        n = rl.size
        same = o1["labels"] == rl
        tok_total += n
        idx_same += int((o1["idx"] == ref[f"{key}/idx"]).sum())
        dt = np.abs(o1["top1"] - ref[f"{key}/top1_logit"])
        db = np.abs(o1["blank"] - ref[f"{key}/blank_logit"])
        d_top, d_blank = max(d_top, float(dt.max())), max(d_blank, float(db.max()))
        drift_labels += int((o1["labels"] != o2["labels"]).sum())
        drift_logit = max(drift_logit, float(np.abs(o1["top1"] - o2["top1"]).max()),
                          float(np.abs(o1["blank"] - o2["blank"]).max()))
        p = int(dt.argmax())
        row = {"set": u["set"], "id": u["id"], "tokens": n, "tokens_same": int(same.sum()),
               "max_abs_top1_logit": float(dt[p]), "max_abs_top1_pos": p,
               "ref_top1_at_max": float(ref[f"{key}/top1_logit"][p]), "coreai_top1_at_max": float(o1["top1"][p]),
               "tokens_top1_off_by_0.1": int((dt > 0.1).sum()), "max_abs_blank_logit": float(db.max())}
        if head == "h":
            rh = ref[f"{key}/h"].astype(np.float64)
            c = (o1["h"].astype(np.float64) * rh).sum(-1) / (np.linalg.norm(o1["h"], axis=-1) * np.linalg.norm(rh, axis=-1))
            cos_all.append(c)
            row.update({"h_cos_mean": float(c.mean()), "h_cos_min": float(c.min())})
        for p in np.flatnonzero(~same):
            mism.append({"set": u["set"], "id": u["id"], "pos": int(p), "ref": int(rl[p]), "coreai": int(o1["labels"][p]),
                         "ref_gap": float(ref[f"{key}/gap"][p]),
                         "ref_blank_minus_top1": float(ref[f"{key}/blank_logit"][p] - ref[f"{key}/top1_logit"][p]),
                         "coreai_blank_minus_top1": float(o1["blank"][p] - o1["top1"][p])})
        t_words = " ".join(vocab[i] for i in RC.collapse(rl, N))
        c_words = " ".join(vocab[i] for i in RC.collapse(o1["labels"], N))
        row["transcript_same"] = t_words == c_words
        rows.append(row)
        refs_txt[u["set"]].append(u["text"])
        hyps_torch[u["set"]].append(t_words)
        hyps_core[u["set"]].append(c_words)
        hyp_lines.append(f"{u['set']}\t{u['id']}\t{u['text']}\t{t_words}\t{c_words}")

    red_rows = []
    for u, w, w1 in zip(red_utts, red, [pass1[i] for i in red_ix]):
        o = read_out(w, head, table_t, scale, N)
        o1 = read_out(w1, head, table_t, scale, N)
        red_rows.append({"id": u["id"], "tokens": int(o["labels"].size),
                         "labels_changed": int((o["labels"] != o1["labels"]).sum()),
                         "transcript_changed": RC.collapse(o["labels"], N) != RC.collapse(o1["labels"], N)})
    zh_rows = []
    for u, w, w1 in zip(red_utts, red_zh, [pass1[i] for i in red_ix]):
        lb = read_out(w, head, table_t, scale, N)["labels"]
        la = read_out(w1, head, table_t, scale, N)["labels"]
        n, h = int(la.size), int(la.size) // 2
        zh_rows.append({"id": u["id"], "tokens": n, "zeroed_from_token": h, "labels_changed": int((la != lb).sum()),
                        "labels_changed_before_the_zeroed_half": int((la[:h] != lb[:h]).sum()),
                        "transcript_changed": RC.collapse(la, N) != RC.collapse(lb, N)})
    zh_changed = sum(r["labels_changed"] > 0 for r in zh_rows)
    zero_half_arm = {"what": "the second half of the valid soft tokens set to 0, token_mask unchanged",
                     "rule": f"red = labels change on >= {RED_ZERO_HALF_MIN} of {len(zh_rows)} utterances",
                     "utterances_with_labels_changed": zh_changed,
                     "transcripts_changed": sum(r["transcript_changed"] for r in zh_rows),
                     "red": zh_changed >= RED_ZERO_HALF_MIN, "utterances": zh_rows}

    by_set = {}
    for s in ("fixture", "test-clean"):
        rs = [r for r in rows if r["set"] == s]
        tok = sum(r["tokens"] for r in rs)
        by_set[s] = {"utterances": len(rs), "transcripts_same": sum(r["transcript_same"] for r in rs),
                     "tokens": tok, "token_agreement": sum(r["tokens_same"] for r in rs) / max(1, tok),
                     "wer_torch": RC.wer(refs_txt[s], hyps_torch[s]), "wer_coreai": RC.wer(refs_txt[s], hyps_core[s])}
    fx, tc = by_set["fixture"], by_set["test-clean"]
    if form in FP16_FORMS:
        bar_ok = (tc["wer_coreai"]["wer"] <= tc["wer_torch"]["wer"] + BAR_WER_DELTA
                  and fx["transcripts_same"] >= BAR_FIXTURE_FP16)
        bar = {"test_clean_wer_max": round(tc["wer_torch"]["wer"] + BAR_WER_DELTA, 6), "fixture_min": BAR_FIXTURE_FP16}
    else:
        bar_ok = fx["transcripts_same"] == fx["utterances"] and tc["token_agreement"] >= BAR_TOKEN_AGREEMENT
        bar = {"fixture_transcripts": f"{fx['utterances']}/{fx['utterances']}",
               "test_clean_token_agreement_min": BAR_TOKEN_AGREEMENT}
    red_fails = any(r["transcript_changed"] or r["labels_changed"] for r in red_rows)
    secs = [r["s"] for ws in pass2 for r in ws]
    summary = {
        "pass": bool(bar_ok and drift_labels == 0 and (red_fails or zero_half_arm["red"])), "bar_met": bool(bar_ok),
        "fixture": fx, "test_clean": tc, "token_agreement_all": sum(r["tokens_same"] for r in rows) / tok_total,
        "tokens": tok_total, "argmax_index_agreement": idx_same / tok_total,
        "label_mismatches": len(mism), "max_abs_top1_logit": d_top, "max_abs_blank_logit": d_blank,
        "tokens_top1_off_by_0.1": sum(r["tokens_top1_off_by_0.1"] for r in rows),
        "drift_labels": drift_labels, "drift_max_abs_logit": drift_logit,
        "red_arm_changes_labels": red_fails,
        "red_arm_transcripts_changed": sum(r["transcript_changed"] for r in red_rows),
        "red_arm_zero_half_red": zero_half_arm["red"],
        "red_arm_zero_half_utterances_changed": zh_changed,
    }
    if head == "h":
        c = np.concatenate(cos_all)
        summary.update({"h_cos_mean": float(c.mean()), "h_cos_min": float(c.min())})
    result = {
        "bundle": str(bundle), "form": form, "tokens": T, "window_s": RC.WINDOW_SECONDS.get(T), "head": head,
        "backend": args.backend,
        "specialization": ("JIT from_preferred_compute_unit_kind(gpu)" if (args.jit and args.backend == "gpu") else
                           ("JIT " if args.jit else "") + SPECIALIZATION[args.backend]),
        "ane_regions": len(regions), "ane_region_names": regions[:32], "input_dtype": np.dtype(dtype).name,
        "function_descriptor": desc, "reader": export_rec["reader"], "ref_npz": str(od / f"ref_T{T}.npz"),
        "bar": bar, "summary": summary, "mismatches": mism[:100], "red_arm_mask_all_ones": red_rows,
        "red_arm_zero_half": zero_half_arm,
        "windows": sum(len(w) for w in pass1), "load_s": round(load_s, 2), "first_call_s": round(pass1[0][0]["s"], 4),
        "median_call_ms_pass2": round(float(np.median(secs)) * 1000, 3),
        "timing_note": "one pass over the gate windows; the timing gate is bench_pipeline.py",
        "runtime": runtime_info(), "label": args.label, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "utterances_table": rows,
    }
    tag = f"{args.backend}{'_jit' if args.jit else ''}_T{T}"
    out = Path(args.out) if args.out else bundle.parent / f"gate_{tag}.json"
    write_json(out, result)
    (out.parent / f"hyp_{tag}.tsv").write_text("set\tid\treference\ttorch\tcoreai\n" + "\n".join(hyp_lines) + "\n")
    print(f"{form} T{T} {args.backend}: {'PASS' if summary['pass'] else 'FAIL'} fixture {fx['transcripts_same']}/20, "
          f"test-clean tokens {tc['token_agreement']:.6f} transcripts {tc['transcripts_same']}/{tc['utterances']}, "
          f"WER torch {100 * tc['wer_torch']['wer']:.3f} % vs coreai {100 * tc['wer_coreai']['wer']:.3f} %, "
          f"argmax {summary['argmax_index_agreement']:.6f}, max|d| top1 {d_top:.2e} blank {d_blank:.2e}, drift "
          f"{drift_labels}, red arms: mask {summary['red_arm_transcripts_changed']}/{len(red_rows)} transcripts changed, "
          f"zero-half labels changed on {zh_changed}/{len(zh_rows)}, "
          f"ANE regions {len(regions)}, load {load_s:.1f} s, call {result['median_call_ms_pass2']} ms -> {out}")


async def warm(args) -> None:
    """The initial load of a bundle (a Neural Engine AOT specializes for minutes) + one call, outside a measurement window."""
    import coreai.runtime as rt
    from gate_encoder import options

    bundle = Path(args.bundle).expanduser().resolve()
    form, T = parse_name(bundle)
    dtype = np.float16 if form in FP16_FORMS else np.float32
    t0 = time.perf_counter()
    fn = (await rt.AIModel.load(bundle, options(args.backend))).load_function("main")
    t1 = time.perf_counter()
    x, m = RC.window(np.zeros((T, RC.D_IN), np.float32), 0, T, T, dtype)
    await fn({"soft_tokens": rt.NDArray(x), "token_mask": rt.NDArray(m)})
    print(json.dumps({"bundle": str(bundle), "backend": args.backend, "load_s": round(t1 - t0, 2),
                      "first_call_s": round(time.perf_counter() - t1, 3)}))


def score(args) -> None:
    """Re-count the WER of a gate's hyp tsv with jiwer (both paths) and write it into the gate json."""
    import jiwer
    from importlib.metadata import version

    g = Path(args.gate).expanduser()
    res = json.loads(g.read_text())
    tsv = g.with_name(g.name.replace("gate_", "hyp_").replace(".json", ".tsv"))
    lines = [line.split("\t") for line in tsv.read_text().splitlines()[1:] if line]
    block = {"engine": f"jiwer {version('jiwer')}", "hyp_tsv": str(tsv)}
    for s in ("fixture", "test-clean"):
        rs = [r for r in lines if r[0] == s]
        for path, col in (("torch", 3), ("coreai", 4)):
            o = jiwer.process_words([r[2] for r in rs], [r[col] if r[col] else "" for r in rs])
            block[f"{s}/{path}"] = {"wer": o.wer, "substitutions": o.substitutions, "deletions": o.deletions,
                                    "insertions": o.insertions, "ref_words": o.substitutions + o.deletions + o.hits}
    tc = res["summary"]["test_clean"]
    block["matches_numpy"] = all(
        abs(block[f"test-clean/{p}"]["wer"] - tc[f"wer_{p}"]["wer"]) < 1e-12 for p in ("torch", "coreai"))
    res["wer_jiwer"] = block
    write_json(g, res)
    print(json.dumps({k: (v["wer"] if isinstance(v, dict) and "wer" in v else v) for k, v in block.items()}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--bundle", required=True, help=".aimodel (cpu, --jit) or .aimodelc (gpu / ane)")
    g.add_argument("--backend", choices=("cpu", "gpu", "ane"), required=True)
    g.add_argument("--jit", action="store_true", help="gpu / ane on the JIT .aimodel (preferred compute unit)")
    g.add_argument("--reader-dir", help="vocab.txt source (default: the one recorded in the export json)")
    g.add_argument("--fixtures")
    g.add_argument("--cache")
    g.add_argument("--out-dir", help="where ref_T<T>.npz is (default <work>/exports/reader, or reader_<version> "
                                         "for a bundle named eg2_reader_<version>_...)")
    g.add_argument("--out", help="default <bundle dir>/gate_<backend>_T<T>.json")
    g.add_argument("--label", default="", help="free text recorded in the json (e.g. the measurement window)")
    w = sub.add_parser("warm")
    w.add_argument("--bundle", required=True)
    w.add_argument("--backend", choices=("cpu", "gpu", "ane"), required=True)
    s = sub.add_parser("score")
    s.add_argument("--gate", required=True, help="a gate json written by `gate` (its hyp tsv sits beside it)")
    args = ap.parse_args()
    if args.cmd == "gate":
        asyncio.run(gate(args))
    elif args.cmd == "warm":
        asyncio.run(warm(args))
    else:
        score(args)


if __name__ == "__main__":
    main()
