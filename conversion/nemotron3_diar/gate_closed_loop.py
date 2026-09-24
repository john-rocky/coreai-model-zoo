"""Closed-loop gate: host_loop.py (NumPy mel + speaker cache) driving a graph over whole recordings,
against transformers fp32 (make_reference.py: ref_<fixture>_<tag>.npz for the output,
chunk_io_<fixture>_<tag>.npz for every step's packed input and cache state). Export venv:

    ~/code/coreai/coreai-models/.venv/bin/python gate_closed_loop.py --parts a,b,f
    ~/code/coreai/coreai-models/.venv/bin/python gate_closed_loop.py --parts c,d,e

  (a) --engine eager --mel ref, streaming low_latency, both fixtures. PASS: every step's packed rows
      [0, L) within 1e-3 of the captured `chunk_input_embeds`, the (n_cache, n_fifo) before and
      (n_cache, n_fifo, is_compressed) after every step equal to transformers', final logits within
      1e-3 of ref_<fixture>_ll.npz and agreement@0.5 = 100 %.
  (b) --engine eager --mel host: the same checks; max|Δp| and agreement reported (the mel mirror's
      error rides on the chunk rows).
  (c) --engine coreai fp16 --unit gpu --mel host, low_latency, both fixtures. PASS: agreement@0.5
      >= 99.9 %; max|Δp| and the segment differences (extract_speaker_dict @0.5) reported.
  (d) as (c) for very_low_latency and ultra_low_latency (references from make_reference.py --mode);
      the (a) checks with the eager graph are run and reported, not gated (see t).
  (t) teacher-forced compression check: every compression transformers made (all captured modes
      and fixtures) rebuilt from the capture. PASS: the host's boost + select on transformers' own
      float32 scores reproduces the captured cache rows and probs at every one (the loop logic).
      Reported: the same with the host's float64 scores, and the smallest top-k boundary gap of
      transformers' scores in float32 ulp. Where that gap is ~2 ulp, the reference's decision is a
      float32 rounding outcome, and any other arithmetic (the eager graph's 5e-5, fp16, another
      log implementation) may take the other branch; from there the cache follows another trajectory.
  (e) offline profile, T=684: the (a) checks with the eager graph and --mel ref, then (c) with the
      offline fp16 bundle. transformers masks the extractor's extra centered frame as a key but
      does not zero it before the conv, so its garbage reaches the last 8 frames; the host does
      not produce that frame (9,760 frames stacked, no zero row). As transformers' integration test
      does, the last 16 frames (2 encoder rows) are left out of the comparison; the last step's
      packed input then holds one row fewer than transformers' (its masked row), and that step's
      state check compares the rows the host feeds.
  (f) negative controls: (a) with --poison no-compress and --poison no-pop on the 97.6 s fixture;
      each must FAIL (a).

--engine2 / --bundle / --unit override the Core AI run of (c)/(d)/(e) (e.g. the Mac ANE, or an AOT
.aimodelc with --unit default). Per-part numbers go to _work/gate_closed_loop_<part>[_<label>].json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import host_loop as hl  # noqa: E402

WORK = HERE / "_work"
ART = WORK / "artifacts"
FIXTURES = list(hl.FIXTURES)
PACKED_BAR, LOGIT_BAR, AGREE_ENGINE = 1e-3, 1e-3, 0.999
OFFLINE_TAIL = 16


def sig64(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def segments_from_json(s) -> list[dict]:
    return [{**g, "start_frame": int(round(g["Start"] * 100)), "end_frame": int(round(g["End"] * 100))}
            for g in json.loads(str(s))]


def segment_diff(ours: list[dict], ref: list[dict]) -> dict:
    """Pair the segments speaker by speaker by overlap. One-to-one overlaps are 'matched' and give the
    boundary shifts in frames; anything else (a split, a merge, a segment with no counterpart) is
    counted as structural."""
    out = {"n_ref": len(ref), "n_ours": len(ours), "matched": 0, "max_start_shift": 0, "max_end_shift": 0,
           "shifted": 0, "structural": 0, "shift_frames_total": 0, "structural_cases": []}
    for spk in sorted({g["Speaker"] for g in ours + ref}):
        r = [(g["start_frame"], g["end_frame"]) for g in ref if g["Speaker"] == spk]
        o = [(g["start_frame"], g["end_frame"]) for g in ours if g["Speaker"] == spk]

        def overlaps(a, b):
            return min(a[1], b[1]) > max(a[0], b[0])
        used = set()
        for rs in r:
            cand = [j for j, os_ in enumerate(o) if overlaps(rs, os_)]
            if len(cand) == 1 and sum(overlaps(o[cand[0]], x) for x in r) == 1:
                os_ = o[cand[0]]
                used.add(cand[0])
                ds, de = os_[0] - rs[0], os_[1] - rs[1]
                out["matched"] += 1
                out["max_start_shift"] = max(out["max_start_shift"], abs(ds))
                out["max_end_shift"] = max(out["max_end_shift"], abs(de))
                out["shifted"] += int(ds != 0 or de != 0)
                out["shift_frames_total"] += abs(ds) + abs(de)
            else:
                out["structural"] += 1
                used.update(cand)
                out["structural_cases"].append({"speaker": spk, "ref": list(rs), "ours": [list(o[j]) for j in cand]})
        extra = [j for j in range(len(o)) if j not in used]
        out["structural"] += len(extra)
        out["structural_cases"] += [{"speaker": spk, "ref": None, "ours": [list(o[j])]} for j in extra]
    return out


def compare_output(logits: np.ndarray, ref_logits: np.ndarray, tail: int = 0) -> dict:
    n = min(logits.shape[0], ref_logits.shape[0]) - tail
    a, b = logits[:n], ref_logits[:n]
    pa, pb = sig64(a), sig64(b)
    agree = (pa > 0.5) == (pb > 0.5)
    seg_o = hl.speaker_segments(a)
    seg_r = hl.speaker_segments(b)
    return {"frames_ours": int(logits.shape[0]), "frames_ref": int(ref_logits.shape[0]), "compared": int(n),
            "tail_excluded": tail, "max_abs_logit": float(np.abs(a - b).max()), "max_abs_p": float(np.abs(pa - pb).max()),
            "agreement": float(agree.mean()), "disagree": int((~agree).sum()), "elements": int(agree.size),
            "segments": segment_diff(seg_o, seg_r)}


def run_case(engine_fn, fixture: str, profile: str, mode: str, mel: str, poison: str | None = None,
             label: str = "") -> dict:
    assets = hl.load_assets()
    audio = hl.load_audio(hl.FIXTURE_DIR / hl.FIXTURES[fixture])
    tag = hl.ref_tag(profile, mode)
    io = np.load(WORK / f"chunk_io_{fixture}_{tag}.npz")
    ref = np.load(WORK / f"ref_{fixture}_{tag}.npz")
    n_ref_steps = int(io["n_steps"])
    steps, n_frames = hl.build_steps(profile, mode, mel, assets, audio, fixture)
    rows = []
    graph_s = []

    def on_step(i, info):
        graph_s.append(info["graph_s"])
        rec = {"step": i, "L": info["L"], "n_cache": info["n_cache"], "n_fifo": info["n_fifo"],
               "compressed_after": bool(info["compressed_after"])}
        if i >= n_ref_steps:
            rec["state_ok"] = False
            rec["packed_max_abs"] = float("inf")
            rows.append(rec)
            return
        ref_rows = io[f"s{i:03d}_inputs_embeds"]
        L_ref = ref_rows.shape[0]
        mask_key = f"s{i:03d}_step_mask"
        # offline: the reference's last step carries the masked centered-frame row; the host has no such row
        n_masked = int((io[mask_key] == 0).sum()) if mask_key in io.files else 0
        rec["L_ref"] = L_ref
        rec["masked_ref_rows"] = n_masked
        state_ref = (int(io["cache_before"][i]), int(io["fifo_before"][i]), int(io["cache_after"][i]),
                     int(io["fifo_after"][i]), bool(io["compressed_after"][i]))
        state = (info["n_cache"], info["n_fifo"], info["cache_after"], info["fifo_after"],
                 bool(info["compressed_after"]))
        if n_masked:
            # only the offline recording's final step has a masked row (the host's step has that row
            # fewer), so its state after is not comparable and feeds nothing: compare the state before
            state_ref, state = state_ref[:2], state[:2]
        rec["state_ok"] = state == state_ref and info["L"] == L_ref - n_masked
        rec["state"], rec["state_ref"] = list(state), list(state_ref)
        L = info["L"]
        if L == L_ref - n_masked:
            rec["packed_max_abs"] = float(np.abs(info["rows"] - ref_rows[:L]).max())
            # the step's logits vs transformers' chunk logits; with a masked row, the last real row's
            # conv neighbour differs (transformers' garbage vs the host's zero padding): leave 2 rows out
            k = (L - 2) * hl.SUB if n_masked else L * hl.SUB
            rec["step_logit_max_abs"] = float(np.abs(info["logits"][:k] - io[f"s{i:03d}_chunk_logits"][:k]).max())
        else:
            rec["packed_max_abs"] = float("inf")
        cp_ref = io[f"s{i:03d}_cache_probs_after"]
        cp = info["cache_probs_after"]
        if n_masked:
            rec["cache_probs_max_abs"] = 0.0          # not comparable (see above), reported as skipped
            rec["cache_probs_skipped"] = True
        elif cp.shape == cp_ref.shape:
            rec["cache_probs_max_abs"] = float(np.abs(cp - cp_ref).max()) if cp.size else 0.0
        else:
            rec["cache_probs_max_abs"] = float("inf")
        rows.append(rec)

    t0 = time.time()
    logits, cache = hl.run(engine_fn, steps, profile, assets["silence"], n_frames, poison, on_step)
    wall = time.time() - t0
    tail = OFFLINE_TAIL if profile == "offline" else 0
    out = compare_output(logits, ref["logits"], tail)
    # the stored reference segments must be what the NumPy extract_speaker_dict gives on the ref logits
    ref_json = segments_from_json(ref["segments"])
    mine = hl.speaker_segments(ref["logits"][: int(ref["attention_mask"].sum())] if "attention_mask" in ref.files
                               else ref["logits"])
    out["extract_speaker_dict_mirror_ok"] = [(g["Start"], g["End"], g["Speaker"]) for g in mine] == \
        [(g["Start"], g["End"], g["Speaker"]) for g in ref_json]
    out.update({
        "fixture": fixture, "profile": profile, "mode": mode, "mel": mel, "poison": poison, "label": label,
        "steps": len(rows), "steps_ref": n_ref_steps,
        "state_all_ok": all(r["state_ok"] for r in rows) and len(rows) == n_ref_steps,
        "packed_max_abs": max(r["packed_max_abs"] for r in rows),
        "step_logit_max_abs": max((r.get("step_logit_max_abs", 0.0) for r in rows), default=0.0),
        "cache_probs_max_abs": max(r["cache_probs_max_abs"] for r in rows),
        "n_compress": cache.n_compress, "compress_steps_ref": int(np.asarray(io["compressed_after"]).sum()),
        "first_compressed_step": next((r["step"] for r in rows if r["compressed_after"]), None),
        "first_compressed_step_ref": (int(np.nonzero(io["compressed_after"])[0][0])
                                      if np.asarray(io["compressed_after"]).any() else None),
        "ties": cache.ties[:20], "n_ties": len(cache.ties),
        "wall_s": round(wall, 2), "graph_ms_median": round(float(np.median(graph_s)) * 1e3, 2),
        "per_step": rows,
    })
    return out


def torch_scores(probs: np.ndarray) -> np.ndarray:
    """transformers' _get_frame_scores + the latest-frames boost, in torch float32 (the reference's
    own arithmetic, for the teacher-forced check only; the host never uses torch)."""
    import torch

    p = torch.from_numpy(probs)[None]
    lp = torch.log(p.clamp(min=hl.PRED_THRESHOLD))
    lc = torch.log((1.0 - p).clamp(min=hl.PRED_THRESHOLD))
    s = lp - lc + lc.sum(dim=-1, keepdim=True) - np.log(0.5).item()
    speech = p > 0.5
    s = s.masked_fill(~speech, float("-inf"))
    pos = s > 0
    s = s.masked_fill(~pos & speech & (pos.sum(dim=1, keepdim=True) >= 16), float("-inf"))
    s[:, hl.CACHE_LEN:] += hl.LATEST_BOOST
    return s[0].numpy()


def torch_pool(logits: np.ndarray) -> np.ndarray:
    import torch

    t = torch.from_numpy(logits)[None].sigmoid().transpose(1, 2)
    return torch.nn.functional.avg_pool1d(t, hl.SUB, hl.SUB).transpose(1, 2)[0].numpy()


def boundary_gaps(scores: np.ndarray, cache: "hl.SpeakerCache") -> list[dict]:
    """Gap between the k-th and (k+1)-th finite value at every top-k boundary the compression walks
    (strong / weak boost per speaker on the running scores, then the final 264 selection), in float32 ulp."""
    out = []
    s = scores.astype(np.float64).copy()

    def gap(where, values, k, spk=None):
        srt = np.sort(values)[::-1]
        if k < values.size and np.isfinite(srt[k - 1]) and np.isfinite(srt[k]):
            ulp = float(np.spacing(np.float32(abs(srt[k]))))
            out.append({"where": where, "speaker": spk, "gap": float(srt[k - 1] - srt[k]),
                        "gap_f32_ulp": float(srt[k - 1] - srt[k]) / ulp})
    for where, k, b in (("strong", cache.n_strong, hl.STRONG_BOOST), ("weak", cache.n_weak, hl.WEAK_BOOST)):
        for spk in range(hl.N_SPK):
            gap(where, s[:, spk], k, spk)
            idx = hl.topk_desc(s[:, spk], k)
            s[idx, spk] += b
    flat = np.concatenate([s, np.full((1, hl.N_SPK), np.inf)]).T.reshape(-1)
    gap("select", flat, hl.CACHE_LEN)
    return out


def select_rows(scores: np.ndarray, cache: "hl.SpeakerCache") -> np.ndarray:
    """The host's boost + select (SpeakerCache.compress after the scores) on given scores -> frame indices."""
    s = scores.astype(np.float64).copy()
    for k, b in ((cache.n_strong, hl.STRONG_BOOST), (cache.n_weak, hl.WEAK_BOOST)):
        for spk in range(hl.N_SPK):
            idx = hl.topk_desc(s[:, spk], k)
            s[idx, spk] += b
    n = s.shape[0]
    flat = np.concatenate([s, np.full((1, hl.N_SPK), np.inf)]).T.reshape(-1)
    idx = hl.topk_desc(flat, hl.CACHE_LEN)
    sentinel = (n + 1) * hl.N_SPK
    idx = np.sort(np.where(flat[idx] == -np.inf, sentinel, idx))
    return np.where(idx == sentinel, n, np.minimum(idx % (n + 1), n))


def teacher_forced(fixture: str, tag: str) -> list[dict]:
    """Every compression transformers made, rebuilt from the capture (cache rows and stored probs before
    the step, the popped FIFO rows, transformers' pooled probabilities of the step): does the host's
    selection reproduce the captured cache (rows and probs) (i) on transformers' own float32 scores
    (the loop logic alone), (ii) with the host's float64 scores (what the host runs)? Plus the smallest
    top-k boundary gap of transformers' scores."""
    io = np.load(WORK / f"chunk_io_{fixture}_{tag}.npz")
    prof = hl.PROFILES["offline" if tag == "offline" else "streaming"]
    silence = hl.load_assets()["silence"]
    out = []
    n = int(io["n_steps"])
    for i in range(n):
        nc, nf, ncf = int(io["cache_before"][i]), int(io["fifo_before"][i]), int(io["num_chunk_frames"][i])
        mask_key = f"s{i:03d}_step_mask"
        if mask_key in io.files:
            continue          # the offline recording's last step: its masked row has zero probs in transformers
        cache = hl.SpeakerCache(prof["fifo_length"], prof["update_period"])
        pop = cache.num_popped(nf + ncf)
        if not pop or nc + pop <= hl.CACHE_LEN:
            continue
        rows = io[f"s{i:03d}_inputs_embeds"]
        p_t = torch_pool(io[f"s{i:03d}_chunk_logits"])
        stored = io[f"s{i - 1:03d}_cache_probs_after"] if bool(io["compressed_before"][i]) else p_t[:nc]
        embeds = np.concatenate([rows[:nc], rows[nc:nc + pop]])
        probs = np.concatenate([stored, p_t[nc:nc + pop]])
        emb_s = np.concatenate([embeds, silence[None]])
        prb_s = np.concatenate([probs, np.zeros((1, hl.N_SPK), np.float32)])
        ref_rows = io[f"s{i + 1:03d}_inputs_embeds"][:hl.CACHE_LEN] if i + 1 < n else None
        ref_probs = io[f"s{i:03d}_cache_probs_after"]

        def same(frames):
            return bool(np.array_equal(prb_s[frames], ref_probs)
                        and (ref_rows is None or np.array_equal(emb_s[frames], ref_rows)))
        ts = torch_scores(probs)
        hs = cache.frame_scores(probs)
        hs[hl.CACHE_LEN:] += hl.LATEST_BOOST
        gaps = boundary_gaps(ts, cache)
        g = min(gaps, key=lambda x: x["gap_f32_ulp"])
        out.append({"fixture": fixture, "tag": tag, "step": i, "n_frames": int(probs.shape[0]),
                    "logic_on_transformers_scores": same(select_rows(ts, cache)),
                    "host_float64_scores": same(select_rows(hs, cache)),
                    "min_gap": g})
    return out


def verdict_a(r: dict) -> bool:
    return (r["state_all_ok"] and r["packed_max_abs"] <= PACKED_BAR and r["max_abs_logit"] <= LOGIT_BAR
            and r["disagree"] == 0 and r["extract_speaker_dict_mirror_ok"])


def line(r: dict, extra: str = "") -> str:
    s = r["segments"]
    return (f"{r['fixture']:20s} {r['profile']:9s} {r['mode']:17s} mel={r['mel']:4s} {r['label']:22s} "
            f"steps {r['steps']}/{r['steps_ref']} state {'ok' if r['state_all_ok'] else 'DIFF'} "
            f"packed {r['packed_max_abs']:.2e} step-logit {r['step_logit_max_abs']:.2e} "
            f"cache-p {r['cache_probs_max_abs']:.2e} | out {r['compared']}/{r['frames_ref']} fr "
            f"max|Δlogit| {r['max_abs_logit']:.3e} max|Δp| {r['max_abs_p']:.3e} agree {r['agreement'] * 100:.4f} % "
            f"({r['disagree']}/{r['elements']}) | seg ref {s['n_ref']} ours {s['n_ours']} matched {s['matched']} "
            f"shift max {s['max_start_shift']}/{s['max_end_shift']} fr, shifted {s['shifted']}, structural "
            f"{s['structural']}{' ' + str(s['structural_cases'][:3]) if s['structural_cases'] else ''} | "
            f"compress {r['n_compress']} (ref {r['compress_steps_ref']} steps) ties {r['n_ties']} "
            f"| {r['wall_s']} s, graph {r['graph_ms_median']} ms/step {extra}")


def save(part: str, results: list[dict], label: str = ""):
    name = f"gate_closed_loop_{part}{'_' + label if label else ''}.json"
    (WORK / name).write_text(json.dumps(results, indent=1, default=float))
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="a,b,f")
    ap.add_argument("--fixtures", default=",".join(FIXTURES))
    ap.add_argument("--bundle", help="Core AI bundle for (c)/(d) (default: the fp16 streaming .aimodel)")
    ap.add_argument("--offline-bundle", help="Core AI bundle for (e) (default: the fp16 offline .aimodel)")
    ap.add_argument("--unit", default="gpu", choices=["cpu", "gpu", "ane", "default"])
    ap.add_argument("--label", default="", help="suffix of the json name and the table label")
    args = ap.parse_args()
    parts = args.parts.split(",")
    fixtures = args.fixtures.split(",")
    ok_all = True
    print(f"[gate_closed_loop] parts {parts}, fixtures {fixtures}", flush=True)

    if "t" in parts:
        res = []
        for fx in fixtures:
            for tag in ("ll", "vll", "ull", "offline"):
                if not (WORK / f"chunk_io_{fx}_{tag}.npz").exists():
                    continue
                for r in teacher_forced(fx, tag):
                    res.append(r)
                    g = r["min_gap"]
                    print(f"[t] {fx:20s} {tag:7s} compression at step {r['step']:3d} (N {r['n_frames']}): host logic on "
                          f"transformers' scores {'==' if r['logic_on_transformers_scores'] else '!='} capture; host "
                          f"float64 scores {'==' if r['host_float64_scores'] else '!='} capture; smallest boundary gap "
                          f"{g['gap']:.2e} = {g['gap_f32_ulp']:.0f} f32 ulp ({g['where']}, spk {g['speaker']})", flush=True)
        logic_ok = all(r["logic_on_transformers_scores"] for r in res)
        ok_all &= logic_ok
        near = [r for r in res if r["min_gap"]["gap_f32_ulp"] < 4]
        print(f"[t] {len(res)} compressions: logic reproduces transformers at {sum(r['logic_on_transformers_scores'] for r in res)}"
              f"/{len(res)} -> {'PASS' if logic_ok else 'FAIL'}; host float64 scores at "
              f"{sum(r['host_float64_scores'] for r in res)}/{len(res)} (reported); transformers' own scores within "
              f"4 ulp of a boundary at {len(near)}: {[(r['fixture'], r['tag'], r['step']) for r in near]}", flush=True)
        print(f"[t] -> {save('t', res)}", flush=True)

    eager_stream = None
    if any(p in parts for p in ("a", "b", "d", "f")):
        eager_stream = hl.eager_engine(hl.PROFILES["streaming"]["T"])

    if "a" in parts or "b" in parts:
        for part, mel in (("a", "ref"), ("b", "host")):
            if part not in parts:
                continue
            res = []
            for fx in fixtures:
                r = run_case(eager_stream, fx, "streaming", "low_latency", mel, label="eager fp32")
                v = verdict_a(r)
                r["pass"] = v
                ok_all &= v if part == "a" else r["state_all_ok"] and r["packed_max_abs"] <= PACKED_BAR
                print(f"[{part}] " + line(r, "-> " + ("PASS" if v else "FAIL")), flush=True)
                res.append(r)
            print(f"[{part}] -> {save(part, res)}", flush=True)

    if "f" in parts:
        res = []
        for poison in ("no-compress", "no-pop"):
            r = run_case(eager_stream, "diarization_example", "streaming", "low_latency", "ref", poison=poison,
                         label=f"eager fp32 poison {poison}")
            red = not verdict_a(r)
            r["red"] = red
            ok_all &= red
            print(f"[f] " + line(r, "-> " + ("RED (as required)" if red else "GREEN = instrument blind")), flush=True)
            res.append(r)
        print(f"[f] -> {save('f', res)}", flush=True)

    engine_label = args.label or f"coreai fp16 {args.unit}"
    if "c" in parts or "d" in parts:
        bundle = args.bundle or str(ART / "n3d_streaming_float16.aimodel")
        eng = hl.CoreAIEngine(bundle, args.unit)
        print(f"[coreai] loaded {bundle} ({args.unit}) in {eng.load_s:.1f} s", flush=True)
        for part, modes in (("c", ["low_latency"]), ("d", ["very_low_latency", "ultra_low_latency"])):
            if part not in parts:
                continue
            res = []
            for mode in modes:
                for fx in fixtures:
                    if part == "d":
                        # reported, not gated: a compression whose transformers decision sits within float32
                        # rounding (part t) can flip under the eager graph's own 5e-5 logit difference
                        r0 = run_case(eager_stream, fx, "streaming", mode, "ref", label="eager fp32")
                        r0["pass"] = verdict_a(r0)
                        print(f"[d-eager] " + line(r0, "-> (a) checks " + ("hold" if r0["pass"] else "do not hold")
                                                   + ", reported"), flush=True)
                        res.append(r0)
                    r = run_case(eng, fx, "streaming", mode, "host", label=engine_label)
                    r["pass"] = r["agreement"] >= AGREE_ENGINE
                    ok_all &= r["pass"]
                    print(f"[{part}] " + line(r, "-> " + ("PASS" if r["pass"] else "FAIL")), flush=True)
                    res.append(r)
            print(f"[{part}] -> {save(part, res, args.label)}", flush=True)

    if "e" in parts:
        res = []
        print(f"[e] offline: the last {OFFLINE_TAIL} frames (2 encoder rows) are left out of every output comparison: "
              "transformers keeps the feature extractor's extra centered frame as a masked key row whose conv-side "
              "garbage reaches the last 8 frames, and its integration test leaves the last encoder frame out too; "
              "the host stacks only the valid mel frames (no masked row)", flush=True)
        eager_off = hl.eager_engine(hl.PROFILES["offline"]["T"])
        for fx in fixtures:
            r = run_case(eager_off, fx, "offline", "low_latency", "ref", label="eager fp32")
            r["pass"] = verdict_a(r)
            ok_all &= r["pass"]
            print(f"[e-eager] " + line(r, "-> " + ("PASS" if r["pass"] else "FAIL")), flush=True)
            res.append(r)
        bundle = args.offline_bundle or str(ART / "n3d_offline_float16.aimodel")
        eng = hl.CoreAIEngine(bundle, args.unit)
        print(f"[coreai] loaded {bundle} ({args.unit}) in {eng.load_s:.1f} s", flush=True)
        for fx in fixtures:
            r = run_case(eng, fx, "offline", "low_latency", "host", label=engine_label)
            r["pass"] = r["agreement"] >= AGREE_ENGINE
            ok_all &= r["pass"]
            print(f"[e] " + line(r, "-> " + ("PASS" if r["pass"] else "FAIL")), flush=True)
            res.append(r)
        print(f"[e] -> {save('e', res, args.label)}", flush=True)

    print(f"gate_closed_loop {','.join(parts)}: {'PASS' if ok_all else 'FAIL'}", flush=True)
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
