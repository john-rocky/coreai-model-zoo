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
"""Time the reader bundles through the Core AI Python runtime, and run the whole transcriber on a long recording.

    prepare     timing inputs for the reader alone, before any measurement window: 20 windows of 750 and of 250
                soft tokens cut from the long fixture's continuous soft tokens (soft_tokens_continuous.npy),
                token_mask all ones. Writes <out>/bench_inputs_reader_T<T>.npz.
    run         one process, one reader bundle: load (wall clock), 3 warm-up calls, then 20 timed calls cycling
                through the 20 windows; prints one JSON line (ms per call median / p90, process CPU ms per call,
                x real time = window audio / ms, peak footprint). h-out forms also time the host read-out (table
                product + argmax in numpy) of each call. --cold deletes this bundle's Core AI cache entry before the load.
    ab          runs `run` for every --config (form:tokens:backend) in turn, --reps rounds (A/B/A/B ...), so the
                forms compared share one measurement window; writes <out>/<form>/timing_<backend>_T<T>.json and
                <out>/timing_windows/<label>.json.
    long        the transcriber end to end on the long fixture, in one process: 30 s windows (3000 log-mel rows of
                features.npy + half_mask) with 96 tokens of left overlap; encoder bundle -> soft tokens -> reader
                bundle on the same window (token_mask) -> host labels. Each window contributes only its new tokens
                and drops its last token (the lane's window contract; the last window keeps its end); the labels of
                all windows are collapsed once. The log-mel rows are in memory before the clock starts (feature
                extraction is outside the graphs). Records wall clock with and without loading the two bundles,
                encoder / reader / host shares, x real time, the transcript and its WER against the fixture
                transcript; a second pass checks the labels are the same. --overlap adds two passes with two windows
                in flight (the encoder of window i+1 issued together with the reader of window i) and compares their
                labels with the sequential pass. Writes <pl>/pipeline_long_<name>.json, hyp_long_<name>.txt,
                labels_long_<name>.npy (and with --save-windows the encoder outputs).
    torch-long  the same windows through torch float32 on the CPU: the re-authored encoder (eg2_audio_coreai,
                exact) and the reader's own forward + greedy_labels. Writes <pl>/pipeline_long_torch.json,
                hyp_long_torch.txt, labels_long_torch.npy, encoder_windows_torch.npy.
    cross       saved encoder windows of one path through a reader (torch, or a reader bundle): isolates the
                encoder from the reader when the two paths' WER differ. Writes <pl>/pipeline_long_cross_<name>.json.
    score       re-count the WER of pipeline jsons with jiwer (needs jiwer, not Core AI).

x real time = audio seconds / wall seconds. Run run / ab / long / a bundle cross inside a measurement window:
    <venv>/bin/python conversion/eg2_transcribe/bench_pipeline.py prepare
    <venv>/bin/python conversion/eg2_transcribe/bench_pipeline.py ab --label r3b-reader30 \\
        --config fp32:750:gpu --config fp16:750:gpu --reps 8
    <venv>/bin/python conversion/eg2_transcribe/bench_pipeline.py long --reader <reader>/fp16/eg2_reader_fp16_T750_gpu_h16c.aimodelc
    <venv>/bin/python conversion/eg2_transcribe/bench_pipeline.py torch-long --checkpoint <run>/best_model.pt
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reader_coreai as RC  # noqa: E402
from _common import exports_dir, fixtures_dir, write_json  # noqa: E402
from bench_encoder import cache_entry, peak_footprint_bytes  # noqa: E402
from gate_reader import FP16_FORMS, NAME, export_record, load_table_t, load_vocab, parse_name  # noqa: E402

N_WINDOWS, WARMUP, CALLS = 20, 3, 20
AOT_NAME = {"gpu": "gpu_h16c", "ane": "ane_h16c"}
ENC_FRAMES, LEFT_OVERLAP, DROP_LAST = 3000, 96, 1
ENCODER_DEFAULT = Path("fp16") / "eg2_audio_fp16_T3000_gpu_h16c.aimodelc"     # under <work>/exports


def pl_dir(arg: str | None = None) -> Path:
    return Path(arg).expanduser() if arg else exports_dir() / "pipeline"


def bundle_path(out: Path, form: str, tokens: int, backend: str) -> Path:
    """The one bundle of <out>/<form>/ for this form, window and backend: eg2_reader_[<version>_]<form>_T<tokens>
    + .aimodel (JIT, cpu) or _<gpu|ane>_h16c.aimodelc (AOT); a version part (v2, ...) is allowed, two matches are not."""
    suffix = ".aimodel" if (backend.endswith("-jit") or backend == "cpu") else f"_{AOT_NAME[backend]}.aimodelc"
    hits = [p for p in sorted((out / form).glob(f"eg2_reader_*{form}_T{tokens}{suffix}"))
            if (m := NAME.match(p.name)) and m.group(2) == form and p.name == p.name[:m.end(3)] + suffix]
    if len(hits) != 1:
        raise SystemExit(f"{out / form}: {len(hits)} bundles for {form} T{tokens} {backend}: {[p.name for p in hits]}")
    return hits[0]


def options(rt, backend: str):
    if backend == "gpu":
        return rt.SpecializationOptions.default()
    if backend == "gpu-jit":
        return rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    if backend == "cpu":
        return rt.SpecializationOptions.cpu_only()
    return rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.neural_engine())


# ---------------------------------------------------------------- reader alone

def prepare(args) -> None:
    st = np.load(fixtures_dir(args.fixtures) / "long-speckled-band" / "soft_tokens_continuous.npy", mmap_mode="r")
    od = RC.out_dir(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    for T in RC.TOKEN_BUCKETS:
        x = np.stack([np.asarray(st[i * T:(i + 1) * T], dtype=np.float32) for i in range(N_WINDOWS)])
        np.savez(od / f"bench_inputs_reader_T{T}.npz", soft_tokens=x, token_mask=np.ones((N_WINDOWS, T), np.float32))
        print(f"wrote {od / f'bench_inputs_reader_T{T}.npz'} {x.shape}")


async def run_one(args) -> dict:
    import coreai.runtime as rt

    od = RC.out_dir(args.out_dir)
    bundle = bundle_path(od, args.form, args.tokens, args.backend)
    entry = cache_entry(bundle)
    cold = False
    if args.cold and entry is not None and entry.exists():
        shutil.rmtree(entry)
        cold = True
    data = np.load(od / f"bench_inputs_reader_T{args.tokens}.npz")
    dtype = np.float16 if args.form in FP16_FORMS else np.float32
    head = "h" if args.form.startswith("h-out") else "argmax"
    inputs = [{"soft_tokens": rt.NDArray(np.ascontiguousarray(data["soft_tokens"][i][None].astype(dtype))),
               "token_mask": rt.NDArray(np.ascontiguousarray(data["token_mask"][i][None].astype(dtype)))}
              for i in range(N_WINDOWS)]
    fp0 = peak_footprint_bytes()
    cpu0 = time.process_time()
    t0 = time.perf_counter()
    model = await rt.AIModel.load(bundle, options(rt, args.backend))
    fn = model.load_function("main")
    load_s = time.perf_counter() - t0
    cpu_load_s = time.process_time() - cpu0
    fp_load = peak_footprint_bytes()
    warm = []
    for i in range(WARMUP):
        t1 = time.perf_counter()
        await fn(inputs[i % N_WINDOWS])
        warm.append(time.perf_counter() - t1)
    secs, outs = [], []
    cpu1 = time.process_time()
    for i in range(CALLS):
        t1 = time.perf_counter()
        res = await fn(inputs[i % N_WINDOWS])
        secs.append(time.perf_counter() - t1)
        outs.append(res)
    cpu_per_call_ms = (time.process_time() - cpu1) / CALLS * 1000
    host = []
    if head == "h":
        exp = json.loads(export_record(bundle).read_text())
        table_t = load_table_t(Path(exp["reader"]["table"]))
        for res in outs:
            t1 = time.perf_counter()
            h = res["h"].numpy()[0].astype(np.float32)
            RC.host_argmax(h, table_t)
            host.append(time.perf_counter() - t1)
    else:
        _ = outs[-1]["top1_index"].numpy()
    fp_end = peak_footprint_bytes()
    ms = sorted(s * 1000 for s in secs)
    med = statistics.median(ms)
    audio_s = RC.WINDOW_SECONDS[args.tokens]
    rec = {"form": args.form, "tokens": args.tokens, "backend": args.backend, "bundle": str(bundle),
           "cold_requested": args.cold, "cache_entry_deleted": cold, "load_s": round(load_s, 3),
           "load_cpu_s": round(cpu_load_s, 3), "warmup_ms": [round(w * 1000, 2) for w in warm],
           "ms": [round(s * 1000, 3) for s in secs], "ms_median": round(med, 3),
           "ms_p90": round(float(np.percentile(ms, 90)), 3), "cpu_ms_per_call": round(cpu_per_call_ms, 3),
           "x_realtime": round(audio_s / (med / 1000), 1), "audio_s_per_call": audio_s,
           "footprint_before_load": fp0, "footprint_after_load": fp_load, "footprint_end": fp_end,
           "t_start": time.strftime("%H:%M:%S")}
    if host:
        hm = statistics.median(h * 1000 for h in host)
        rec.update({"host_readout_ms": [round(h * 1000, 3) for h in host], "host_readout_ms_median": round(hm, 3),
                    "call_plus_host_ms_median": round(med + hm, 3),
                    "x_realtime_with_host": round(audio_s / ((med + hm) / 1000), 1),
                    "host_readout": "numpy float32: h @ table [768, N] + argmax, 256 rows per chunk"})
    return rec


def ab(args) -> None:
    od = RC.out_dir(args.out_dir)
    configs = [tuple(c.split(":")) for c in args.config]
    runs: dict[str, list[dict]] = {c: [] for c in args.config}
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for rep in range(args.reps):
        for spec, (form, tokens, backend) in zip(args.config, configs):
            cmd = [sys.executable, str(Path(__file__).resolve()), "run", "--form", form, "--tokens", tokens,
                   "--backend", backend] + (["--out-dir", args.out_dir] if args.out_dir else []) + \
                  (["--cold"] if (rep == 0 and args.cold_first) else [])
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            line = [s for s in r.stdout.splitlines() if s.startswith("{")]
            if r.returncode != 0 or not line:
                rec = {"error": (r.stderr or r.stdout).strip().splitlines()[-5:], "rep": rep}
            else:
                rec = json.loads(line[-1])
                rec["rep"] = rep
            runs[spec].append(rec)
            print(f"rep {rep} {spec}: " + (f"load {rec['load_s']} s, median {rec['ms_median']} ms, p90 "
                                           f"{rec['ms_p90']} ms, x{rec['x_realtime']}"
                                           + (f", host {rec['host_readout_ms_median']} ms" if "host_readout_ms_median"
                                              in rec else "") if "ms_median" in rec else f"ERROR {rec['error']}"),
                  flush=True)
    finished = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    window = {"label": args.label, "started": started, "finished": finished, "reps": args.reps,
              "order": args.config, "warmup": WARMUP, "calls_per_rep": CALLS, "configs": {}}
    for spec, (form, tokens, backend) in zip(args.config, configs):
        ok = [r for r in runs[spec] if "ms_median" in r]
        meds = [r["ms_median"] for r in ok]
        allms = sorted(m for r in ok for m in r["ms"])
        loads = [r["load_s"] for r in ok]
        audio_s = RC.WINDOW_SECONDS[int(tokens)]
        summ = {
            "reps_ok": len(ok), "reps_failed": len(runs[spec]) - len(ok), "audio_s_per_call": audio_s,
            "ms_median_of_rep_medians": round(statistics.median(meds), 3) if meds else None,
            "ms_rep_median_range": [min(meds), max(meds)] if meds else None,
            "ms_p90_all_calls": round(float(np.percentile(allms, 90)), 3) if allms else None,
            "x_realtime": round(audio_s / (statistics.median(meds) / 1000), 1) if meds else None,
            "cpu_ms_per_call_median": round(statistics.median([r["cpu_ms_per_call"] for r in ok]), 3) if ok else None,
            "load_s_first_process": loads[0] if loads else None,
            "load_s_first_process_cold": bool(ok and ok[0].get("cache_entry_deleted")),
            "load_s_warm_median": round(statistics.median(loads[1:]), 3) if len(loads) > 1 else None,
            "peak_footprint_gb_max": round(max(r["footprint_end"].get("lifetime_max_phys_footprint", 0)
                                               for r in ok) / 1e9, 3) if ok else None,
        }
        hosts = [r["host_readout_ms_median"] for r in ok if "host_readout_ms_median" in r]
        if hosts:
            summ["host_readout_ms_median_of_reps"] = round(statistics.median(hosts), 3)
            summ["call_plus_host_ms"] = round(summ["ms_median_of_rep_medians"] + summ["host_readout_ms_median_of_reps"], 3)
            summ["x_realtime_with_host"] = round(audio_s / (summ["call_plus_host_ms"] / 1000), 1)
        window["configs"][spec] = summ
        rec = {"form": form, "tokens": int(tokens), "backend": backend, "window_label": args.label,
               "window_started": started, "window_finished": finished, "compared_in_window_with": args.config,
               "summary": summ, "reps": runs[spec]}
        dest = od / form / f"timing_{backend}_T{tokens}.json"
        if dest.exists() and not args.overwrite:
            dest = od / form / f"timing_{backend}_T{tokens}_{args.label}.json"
        write_json(dest, rec)
        print(f"{spec}: median {summ['ms_median_of_rep_medians']} ms (reps {summ['ms_rep_median_range']}), p90 "
              f"{summ['ms_p90_all_calls']} ms, x{summ['x_realtime']}, load at process start {summ['load_s_first_process']} s "
              f"warm {summ['load_s_warm_median']} s, peak footprint {summ['peak_footprint_gb_max']} GB"
              + (f", +host {summ['host_readout_ms_median_of_reps']} ms = x{summ['x_realtime_with_host']}"
                 if hosts else "") + f" -> {dest}", flush=True)
    write_json(od / "timing_windows" / f"{args.label}.json", window)


# ---------------------------------------------------------------- the long recording

def window_plan(n_frames: int, frames: int = ENC_FRAMES, left: int = LEFT_OVERLAP, drop: int = DROP_LAST) -> list[dict]:
    """Windows of `frames` log-mel rows stepping by (T - left - drop) tokens; keep = the window's new tokens.

    Window i starts at token a = i * (T - left - drop) (frame 4a). It keeps its tokens [left, T - drop) (window 0
    from 0); the window that reaches the end of the recording keeps through its last valid token."""
    from eg2_audio_coreai import n_tokens
    T = frames // 4
    total = n_tokens(n_frames)
    step = T - left - drop
    plan, i = [], 0
    while True:
        a = i * step
        nf = min(frames, n_frames - 4 * a)
        m = n_tokens(nf)
        final = a + m >= total
        keep = (0 if i == 0 else left, m if final else T - drop)
        plan.append({"i": i, "token0": a, "frame0": 4 * a, "frames": nf, "valid_tokens": m, "keep": keep})
        if final:
            break
        i += 1
    kept = [(w["token0"] + w["keep"][0], w["token0"] + w["keep"][1]) for w in plan]
    assert kept[0][0] == 0 and kept[-1][1] == total and all(p[1] == q[0] for p, q in zip(kept, kept[1:])), kept[:3]
    return plan


def long_inputs(args) -> tuple[np.ndarray, dict, str]:
    fx = fixtures_dir(args.fixtures)
    feats = np.load(fx / "long-speckled-band" / "features.npy")                 # [n_frames, 128] float32
    manifest = json.loads((fx / "MANIFEST.json").read_text())["long"]
    ref = RC.norm_text((fx / "long-speckled-band" / "transcript.txt").read_text())
    assert feats.shape[0] == manifest["n_frames"]
    return feats, manifest, ref


def encoder_feed(rt, feats: np.ndarray, w: dict, dtype) -> dict:
    x = np.zeros((1, ENC_FRAMES, feats.shape[1]), dtype=dtype)
    x[0, :w["frames"]] = feats[w["frame0"]:w["frame0"] + w["frames"]]
    hm = np.zeros((1, ENC_FRAMES // 2), dtype=dtype)
    hm[0, :(w["frames"] + 1) // 2] = 1
    return {"features": rt.NDArray(x), "half_mask": rt.NDArray(hm)}


def wer_block(ref: str, hyp: str) -> dict:
    e = RC.word_errors(ref.split(), hyp.split())
    n = len(ref.split())
    return {"wer": e / n, "errors": e, "ref_words": n, "hyp_words": len(hyp.split()),
            "engine": "word Levenshtein (numpy); jiwer: bench_pipeline.py score"}


async def long_run(args) -> None:
    import coreai.runtime as rt

    feats, manifest, ref_text = long_inputs(args)
    plan = window_plan(feats.shape[0], left=args.left_overlap, drop=args.drop_last)
    total = plan[-1]["token0"] + plan[-1]["keep"][1]
    enc_b = Path(args.encoder).expanduser().resolve() if args.encoder else exports_dir() / ENCODER_DEFAULT
    rd_b = Path(args.reader).expanduser().resolve()
    form, T = parse_name(rd_b)
    assert T == ENC_FRAMES // 4, f"the reader window must be {ENC_FRAMES // 4} tokens"
    enc_dtype = np.float16 if "_fp16_" in enc_b.name else np.float32
    rd_dtype = np.float16 if form in FP16_FORMS else np.float32
    head = "h" if form.startswith("h-out") else "argmax"
    exp = json.loads(export_record(rd_b).read_text())
    vocab = load_vocab(Path(exp["reader"]["reader_dir"]))
    N, scale = len(vocab), float(exp["reader"]["scale"])
    table_t = load_table_t(Path(exp["reader"]["table"])) if head == "h" else None
    backend = args.backend
    name = args.name or f"{form}_{backend}"

    entries = {str(b): cache_entry(b).exists() for b in (enc_b, rd_b) if b.suffix == ".aimodelc"}
    fp0 = peak_footprint_bytes()
    t0 = time.perf_counter()
    enc = (await rt.AIModel.load(enc_b, rt.SpecializationOptions.default())).load_function("main")
    t1 = time.perf_counter()
    rd = (await rt.AIModel.load(rd_b, options(rt, backend))).load_function("main")
    t2 = time.perf_counter()
    load = {"encoder_s": round(t1 - t0, 3), "reader_s": round(t2 - t1, 3), "total_s": round(t2 - t0, 3),
            "cache_entries_existed": entries}

    def reader_feed(x: np.ndarray, w: dict) -> dict:
        tm = np.zeros((1, T), dtype=rd_dtype)
        tm[0, :w["valid_tokens"]] = 1
        return {"soft_tokens": rt.NDArray(np.ascontiguousarray(x)), "token_mask": rt.NDArray(tm)}

    def read_labels(res, w: dict, labels: np.ndarray) -> None:
        k0, k1 = w["keep"]
        if head == "argmax":
            idx = res["top1_index"].numpy()[0, k0:k1]
            top = res["top1_logit"].numpy()[0, k0:k1]
        else:
            idx, best = RC.host_argmax(res["h"].numpy()[0, k0:k1].astype(np.float32), table_t)
            top = best * np.float32(scale)
        blank = res["blank_logit"].numpy()[0, k0:k1]
        labels[w["token0"] + k0:w["token0"] + k1] = RC.host_labels(idx, top, blank, N)

    passes, windows_out = [], None
    for p in range(2):
        labels = np.full(total, -1, dtype=np.int64)
        enc_ms, rd_ms = [], []
        keep_windows = args.save_windows and p == 0
        if keep_windows:
            windows_out = np.zeros((len(plan), T, RC.D_IN), dtype=np.float16)
        c0 = time.process_time()
        w0 = time.perf_counter()
        for w in plan:
            feed = encoder_feed(rt, feats, w, enc_dtype)
            a = time.perf_counter()
            soft = (await enc(feed))["soft_tokens"]
            b = time.perf_counter()
            x = soft.numpy()
            if x.dtype != rd_dtype:
                x = x.astype(rd_dtype)
            if keep_windows:
                windows_out[w["i"]] = x[0]
            feed = reader_feed(x, w)
            c = time.perf_counter()
            res = await rd(feed)
            d = time.perf_counter()
            read_labels(res, w, labels)
            enc_ms.append((b - a) * 1000)
            rd_ms.append((d - c) * 1000)
        words = RC.collapse(labels, N)
        text = " ".join(vocab[i] for i in words)
        wall = time.perf_counter() - w0
        cpu = time.process_time() - c0
        assert (labels >= 0).all()
        enc_s, rd_s = sum(enc_ms) / 1000, sum(rd_ms) / 1000
        passes.append({"pass": p + 1, "wall_s": round(wall, 4), "encoder_calls_s": round(enc_s, 4),
                       "reader_calls_s": round(rd_s, 4), "host_s": round(wall - enc_s - rd_s, 4),
                       "process_cpu_s": round(cpu, 3),
                       "x_realtime": round(manifest["duration_s"] / wall, 1),
                       "encoder_ms_median": round(statistics.median(enc_ms), 3),
                       "reader_ms_median": round(statistics.median(rd_ms), 3),
                       "encoder_ms": [round(v, 3) for v in enc_ms], "reader_ms": [round(v, 3) for v in rd_ms],
                       "labels": labels, "text": text})
    p1, p2 = passes
    overlap = None
    if args.overlap:
        # two windows in flight: the encoder of window i+1 issued together with the reader of window i
        opasses = []
        for p in range(2):
            labels = np.full(total, -1, dtype=np.int64)
            c0 = time.process_time()
            w0 = time.perf_counter()
            soft = (await enc(encoder_feed(rt, feats, plan[0], enc_dtype)))["soft_tokens"]
            for k, w in enumerate(plan):
                x = soft.numpy().astype(rd_dtype, copy=True)       # a copy: the next encoder call is already issued
                if k + 1 < len(plan):
                    nxt, res = await asyncio.gather(enc(encoder_feed(rt, feats, plan[k + 1], enc_dtype)),
                                                    rd(reader_feed(x, w)))
                    soft = nxt["soft_tokens"]
                else:
                    res = await rd(reader_feed(x, w))
                read_labels(res, w, labels)
            text = " ".join(vocab[i] for i in RC.collapse(labels, N))
            wall = time.perf_counter() - w0
            assert (labels >= 0).all()
            opasses.append({"pass": p + 1, "wall_s": round(wall, 4), "x_realtime": round(manifest["duration_s"] / wall, 1),
                            "process_cpu_s": round(time.process_time() - c0, 3),
                            "labels_differing_from_sequential_pass1": int((labels != p1["labels"]).sum()),
                            "transcript_same_as_sequential": text == p1["text"], "wer": wer_block(ref_text, text)})
            if p == 0 and text != p1["text"]:
                (pl_dir(args.pl_dir) / f"hyp_long_{name}_overlap.txt").write_text(text + "\n")
        overlap = {"what": "encoder of window i+1 issued with the reader of window i (asyncio.gather), soft tokens "
                           "copied to the host between them", "passes": opasses,
                   "wall_s": opasses[0]["wall_s"], "x_realtime": opasses[0]["x_realtime"],
                   "wall_s_with_load": round(load["total_s"] + opasses[0]["wall_s"], 4),
                   "x_realtime_with_load": round(manifest["duration_s"] / (load["total_s"] + opasses[0]["wall_s"]), 1)}
        print(f"overlap: pass 1 {opasses[0]['wall_s']:.3f} s (x{opasses[0]['x_realtime']}), pass 2 "
              f"{opasses[1]['wall_s']:.3f} s; labels differing from the sequential pass "
              f"{[o['labels_differing_from_sequential_pass1'] for o in opasses]}", flush=True)
    drift = int((p1["labels"] != p2["labels"]).sum())
    pd = pl_dir(args.pl_dir)
    pd.mkdir(parents=True, exist_ok=True)
    np.save(pd / f"labels_long_{name}.npy", p1["labels"].astype(np.int32))
    (pd / f"hyp_long_{name}.txt").write_text(p1["text"] + "\n")
    if windows_out is not None:
        np.save(pd / f"encoder_windows_{name}.npy", windows_out)
    torch_json = pd / "pipeline_long_torch.json"
    torch_wer = json.loads(torch_json.read_text())["wer"] if torch_json.exists() else None
    fp_end = peak_footprint_bytes()
    result = {
        "what": "encoder bundle -> reader bundle -> host labels, one process, long-speckled-band",
        "audio_s": manifest["duration_s"], "n_frames": int(feats.shape[0]), "tokens": total, "windows": len(plan),
        "window_contract": {"frames": ENC_FRAMES, "tokens": T, "left_overlap_tokens": args.left_overlap,
                            "dropped_last_tokens": args.drop_last, "step_tokens": T - args.left_overlap - args.drop_last,
                            "first_window_keeps": list(plan[0]["keep"]), "last_window": plan[-1]},
        "encoder": {"bundle": str(enc_b), "io_dtype": np.dtype(enc_dtype).name, "specialization": "default()"},
        "reader": {"bundle": str(rd_b), "form": form, "backend": backend, "io_dtype": np.dtype(rd_dtype).name,
                   "head": head, "weights_sha256": exp["reader"]["weights_sha256"], "arch": exp["reader"]["arch"],
                   "source": exp["reader"]["kind"]},
        "load": load,
        "wall_s_with_load": round(load["total_s"] + p1["wall_s"], 4),
        "x_realtime_with_load": round(manifest["duration_s"] / (load["total_s"] + p1["wall_s"]), 1),
        "wall_s": p1["wall_s"], "x_realtime": p1["x_realtime"],
        "passes": [{k: v for k, v in p.items() if k not in ("labels", "text")} for p in passes],
        "drift_labels_pass1_vs_pass2": drift, "transcript_same_both_passes": p1["text"] == p2["text"],
        "wer": wer_block(ref_text, p1["text"]), "wer_torch_path": torch_wer, "overlap": overlap,
        "reference": "fixtures/long-speckled-band/transcript.txt (Gutenberg #1661 body; the LibriVox intro and outro "
                     "are in the audio, not in the text)",
        "footprint_end": fp_end, "footprint_before_load": fp0, "label": args.label,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(pd / f"pipeline_long_{name}.json", result)
    print(f"long {name}: {len(plan)} windows, wall {p1['wall_s']:.3f} s (x{p1['x_realtime']}), with load "
          f"{result['wall_s_with_load']:.3f} s (x{result['x_realtime_with_load']}); encoder {p1['encoder_calls_s']:.3f} "
          f"+ reader {p1['reader_calls_s']:.3f} + host {p1['host_s']:.3f} s; pass 2 {p2['wall_s']:.3f} s; drift {drift}; "
          f"WER {100 * result['wer']['wer']:.3f} % ({result['wer']['errors']}/{result['wer']['ref_words']})"
          + (f" vs torch {100 * torch_wer['wer']:.3f} %" if torch_wer else "") + f" -> {pd}/pipeline_long_{name}.json")


def torch_long(args) -> None:
    import torch

    import eg2_audio_coreai as E
    from _common import model_dir, verify_checkpoint

    torch.set_num_threads(args.threads)
    feats, manifest, ref_text = long_inputs(args)
    plan = window_plan(feats.shape[0], left=args.left_overlap, drop=args.drop_last)
    total = plan[-1]["token0"] + plan[-1]["keep"][1]
    mdir = model_dir(args.model_dir)
    verify_checkpoint(mdir)
    enc = E.load_encoder(mdir, norm="exact").eval()
    R = RC.load_reader(args)
    M = sys.modules["model"]
    N = len(R["vocab"])
    T = ENC_FRAMES // 4
    labels = np.full(total, -1, dtype=np.int64)
    windows = np.zeros((len(plan), T, RC.D_IN), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for w in plan:
            x = np.zeros((1, ENC_FRAMES, feats.shape[1]), dtype=np.float32)
            x[0, :w["frames"]] = feats[w["frame0"]:w["frame0"] + w["frames"]]
            soft = enc(torch.from_numpy(x), E.half_mask_for(w["frames"], ENC_FRAMES))[0]
            windows[w["i"]] = soft.numpy()
            m = w["valid_tokens"]
            h, blank = R["model"](soft[None], torch.tensor([m]))
            lab = M.greedy_labels(h[0, :m], blank[0, :m], R["table"], R["model"].scale, chunk=256).numpy()
            k0, k1 = w["keep"]
            labels[w["token0"] + k0:w["token0"] + k1] = lab[k0:k1]
            if w["i"] % 25 == 0:
                print(f"window {w['i']}/{len(plan)} ({time.time() - t0:.0f} s)", flush=True)
    assert (labels >= 0).all()
    text = " ".join(R["vocab"][i] for i in RC.collapse(labels, N))
    pd = pl_dir(args.pl_dir)
    pd.mkdir(parents=True, exist_ok=True)
    np.save(pd / "labels_long_torch.npy", labels.astype(np.int32))
    np.save(pd / "encoder_windows_torch.npy", windows)
    (pd / "hyp_long_torch.txt").write_text(text + "\n")
    result = {"what": "torch float32 CPU: re-authored encoder (exact) + the reader's forward + greedy_labels, the same "
                      "windows as `long`", "audio_s": manifest["duration_s"], "tokens": total, "windows": len(plan),
              "window_contract": {"frames": ENC_FRAMES, "left_overlap_tokens": args.left_overlap,
                                  "dropped_last_tokens": args.drop_last},
              "reader": R["source"], "encoder_model_dir": str(mdir), "wer": wer_block(ref_text, text),
              "seconds": round(time.time() - t0, 1), "threads": args.threads,
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_json(pd / "pipeline_long_torch.json", result)
    print(f"torch-long: {len(plan)} windows, WER {100 * result['wer']['wer']:.3f} % "
          f"({result['wer']['errors']}/{result['wer']['ref_words']}) in {result['seconds']} s")


async def cross(args) -> None:
    """Saved encoder windows (one path) -> a reader (torch on the CPU, or a reader bundle) -> WER."""
    feats, manifest, ref_text = long_inputs(args)
    plan = window_plan(feats.shape[0], left=args.left_overlap, drop=args.drop_last)
    total = plan[-1]["token0"] + plan[-1]["keep"][1]
    wins = np.load(Path(args.windows).expanduser())
    assert wins.shape[0] == len(plan)
    labels = np.full(total, -1, dtype=np.int64)
    T = ENC_FRAMES // 4
    if args.reader == "torch":
        import torch
        torch.set_num_threads(args.threads)
        R = RC.load_reader(args)
        M = sys.modules["model"]
        vocab, N = R["vocab"], len(R["vocab"])
        with torch.no_grad():
            for w in plan:
                m = w["valid_tokens"]
                h, blank = R["model"](torch.from_numpy(wins[w["i"]].astype(np.float32))[None], torch.tensor([m]))
                lab = M.greedy_labels(h[0, :m], blank[0, :m], R["table"], R["model"].scale, chunk=256).numpy()
                k0, k1 = w["keep"]
                labels[w["token0"] + k0:w["token0"] + k1] = lab[k0:k1]
        reader_desc = {"reader": "torch float32 CPU", "weights_sha256": R["source"]["weights_sha256"]}
    else:
        import coreai.runtime as rt
        rd_b = Path(args.reader).expanduser().resolve()
        form, Tr = parse_name(rd_b)
        assert Tr == T and not form.startswith("h-out")
        exp = json.loads(export_record(rd_b).read_text())
        vocab = load_vocab(Path(exp["reader"]["reader_dir"]))
        N = len(vocab)
        dt = np.float16 if form in FP16_FORMS else np.float32
        rd = (await rt.AIModel.load(rd_b, options(rt, args.backend))).load_function("main")
        for w in plan:
            tm = np.zeros((1, T), dtype=dt)
            tm[0, :w["valid_tokens"]] = 1
            res = await rd({"soft_tokens": rt.NDArray(np.ascontiguousarray(wins[w["i"]][None].astype(dt))),
                            "token_mask": rt.NDArray(tm)})
            k0, k1 = w["keep"]
            labels[w["token0"] + k0:w["token0"] + k1] = RC.host_labels(
                res["top1_index"].numpy()[0, k0:k1], res["top1_logit"].numpy()[0, k0:k1],
                res["blank_logit"].numpy()[0, k0:k1], N)
        reader_desc = {"reader": str(rd_b), "backend": args.backend, "weights_sha256": exp["reader"]["weights_sha256"]}
    text = " ".join(vocab[i] for i in RC.collapse(labels, N))
    pd = pl_dir(args.pl_dir)
    name = args.name
    np.save(pd / f"labels_long_cross_{name}.npy", labels.astype(np.int32))
    (pd / f"hyp_long_cross_{name}.txt").write_text(text + "\n")
    result = {"what": "saved encoder windows through a reader", "windows_npy": str(args.windows), **reader_desc,
              "wer": wer_block(ref_text, text), "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_json(pd / f"pipeline_long_cross_{name}.json", result)
    print(f"cross {name}: WER {100 * result['wer']['wer']:.3f} % ({result['wer']['errors']}/{result['wer']['ref_words']})")


def score(args) -> None:
    import jiwer
    from importlib.metadata import version

    ref = RC.norm_text((fixtures_dir(args.fixtures) / "long-speckled-band" / "transcript.txt").read_text())
    for j in args.json:
        p = Path(j).expanduser()
        res = json.loads(p.read_text())
        hyp_file = p.with_name(p.name.replace("pipeline_long_", "hyp_long_").replace(".json", ".txt"))
        hyp = RC.norm_text(hyp_file.read_text())
        o = jiwer.process_words(ref, hyp)
        res["wer_jiwer"] = {"engine": f"jiwer {version('jiwer')}", "wer": o.wer, "substitutions": o.substitutions,
                            "deletions": o.deletions, "insertions": o.insertions,
                            "ref_words": o.substitutions + o.deletions + o.hits, "hyp_file": str(hyp_file),
                            "matches_numpy": abs(o.wer - res["wer"]["wer"]) < 1e-12}
        write_json(p, res)
        print(f"{p.name}: jiwer WER {100 * o.wer:.3f} % (S {o.substitutions} D {o.deletions} I {o.insertions}), "
              f"numpy {100 * res['wer']['wer']:.3f} %")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--fixtures")
    p.add_argument("--out-dir")
    r = sub.add_parser("run")
    r.add_argument("--form", required=True)
    r.add_argument("--tokens", type=int, required=True)
    r.add_argument("--backend", choices=("gpu", "ane", "gpu-jit", "ane-jit", "cpu"), required=True,
                   help="gpu / ane = the h16c AOT bundle; *-jit and cpu = the source .aimodel specialized at load")
    r.add_argument("--cold", action="store_true")
    r.add_argument("--out-dir")
    a = sub.add_parser("ab")
    a.add_argument("--config", action="append", required=True, help="form:tokens:backend, e.g. fp16:750:gpu")
    a.add_argument("--reps", type=int, default=8)
    a.add_argument("--label", required=True)
    a.add_argument("--cold-first", action="store_true", help="delete each bundle's cache entry before rep 0")
    a.add_argument("--overwrite", action="store_true")
    a.add_argument("--out-dir")
    lg = sub.add_parser("long")
    lg.add_argument("--reader", required=True, help="reader bundle (T = 750), AOT .aimodelc")
    lg.add_argument("--backend", choices=("gpu", "ane"), default="gpu", help="the reader's backend")
    lg.add_argument("--encoder", help="encoder bundle (default <work>/exports/fp16/eg2_audio_fp16_T3000_gpu_h16c.aimodelc)")
    lg.add_argument("--name", help="output name (default <form>_<backend>)")
    lg.add_argument("--save-windows", action="store_true", help="also save the encoder outputs of every window")
    lg.add_argument("--overlap", action="store_true",
                    help="after the two sequential passes, two passes with the encoder of window i+1 issued together "
                         "with the reader of window i; labels compared with the sequential pass")
    lg.add_argument("--fixtures")
    lg.add_argument("--pl-dir", help="default <work>/exports/pipeline")
    lg.add_argument("--label", default="")
    t = sub.add_parser("torch-long")
    RC.add_reader_args(t)
    t.add_argument("--model-dir", help="google/embeddinggemma-2 snapshot (default: _common.model_dir)")
    t.add_argument("--fixtures")
    t.add_argument("--pl-dir")
    t.add_argument("--threads", type=int, default=8)
    c = sub.add_parser("cross")
    RC.add_reader_args(c)
    c.add_argument("--windows", required=True, help="encoder_windows_<name>.npy of `long` or `torch-long`")
    c.add_argument("--reader", required=True, help="`torch` or a reader bundle (T = 750)")
    c.add_argument("--backend", choices=("gpu", "ane", "cpu"), default="gpu")
    c.add_argument("--name", required=True)
    c.add_argument("--fixtures")
    c.add_argument("--pl-dir")
    c.add_argument("--threads", type=int, default=8)
    for q in (lg, t, c):
        q.add_argument("--left-overlap", type=int, default=LEFT_OVERLAP, help="tokens of left overlap per window")
        q.add_argument("--drop-last", type=int, default=DROP_LAST, help="tokens dropped at the end of each window")
    s = sub.add_parser("score")
    s.add_argument("--json", nargs="+", required=True, help="pipeline_long_*.json (their hyp_long_*.txt beside them)")
    s.add_argument("--fixtures")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "run":
        print(json.dumps(asyncio.run(run_one(args))))
    elif args.cmd == "ab":
        ab(args)
    elif args.cmd == "long":
        asyncio.run(long_run(args))
    elif args.cmd == "torch-long":
        torch_long(args)
    elif args.cmd == "cross":
        asyncio.run(cross(args))
    else:
        score(args)


if __name__ == "__main__":
    main()
