"""Golden files for the Swift host (swift/, n3d-selftest): what the Python side (transformers fp32 and
host_loop.py) produced, as raw little-endian float32 / float64 files the Swift self-test reads without
NumPy. Export venv:

    ~/code/coreai/coreai-models/.venv/bin/python export_golden.py              # everything below
    ~/code/coreai/coreai-models/.venv/bin/python export_golden.py --skip-pygpu # no Core AI GPU runs

Writes _work/golden/ (C order; fixtures diarization_example = 97.6 s, test_multispk = 21.5 s;
tag = ll | vll | ull | offline):

  <fixture>_<tag>_probs.f32le    [n_frames, 8]  transformers fp32 probabilities (ref_<fixture>_<tag>.npz
                                 'probs'): the self-test's --golden. offline keeps transformers' extra
                                 masked frame (9,761 / 2,151 rows); the gate leaves the last 16 frames out.
  <fixture>_<tag>_logits.f32le   [n_frames, 8]  the same run's logits (what gate_closed_loop.py compares;
                                 the self-test reports max|Δlogit| from it). Checked here: probs > 0.5
                                 equals logits > 0 and float64 sigmoid(logits) > 0.5 on every element, so
                                 the agreement is the same on either file.
  diarization_example_ll_packed_step<k>.f32le  [L, 512]  the packed rows [0, L) host_loop.py feeds the
                                 graph at step k with --engine eager --mel host (97.6 s, low_latency),
                                 k = 0, 1, 54, 78, 128 (the gate's five) and the last step 135.
  <fixture>_ll_mel_chunk<k>.f32le  [F, 128]  mel_frontend.log_mel of streaming chunk k (0, 1, last);
  <fixture>_offline_mel.f32le    [F, 128]  the whole-recording mel (center=True) -- mel diagnostics.
  chunks/<fixture>_<tag>_rows.f32le  [sum of rows, 512]  every step's chunk rows (mel + stacking +
                                 projection, incl. look-ahead) in step order; chunks/index.json has each
                                 step's (rows, look-ahead, emitted frames). The self-test's --chunks
                                 feeds these instead of the Swift mel (isolates the mel from the loop).
  cache/<fixture>_<tag>_s<step>_*  teacher-forced speaker-cache units from the eager --mel host runs:
                                 steps 0 and 1 and every step whose update pops FIFO rows (all
                                 compressions). Per unit: rows [L,512], logits [L*8,8], probs_before
                                 [n_cache,8], pooled [L,8] (pool_probs), after-state embeds / probs /
                                 fifo; for a compression also the float64 frame_scores [N,8] (.f64le).
                                 cache/index.json lists the units with their ints.
  pygpu/<fixture>_<tag>_logits.f32le [n_frames, 8]  host_loop.py with the fp16 bundle on the Mac GPU
                                 (--mel host), and pygpu/diarization_example_ll_packed_step<k>.f32le:
                                 the Python run the Swift GPU run should reproduce step for step.
  golden.json                    index: every file with shape, dtype and sha256, and the checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import host_loop as hl  # noqa: E402
import mel_frontend as mf  # noqa: E402

WORK = HERE / "_work"
ART = WORK / "artifacts"
GOLD = WORK / "golden"
TAGS = {"ll": ("streaming", "low_latency"), "vll": ("streaming", "very_low_latency"),
        "ull": ("streaming", "ultra_low_latency"), "offline": ("offline", "low_latency")}
PACKED_STEPS = (0, 1, 54, 78, 128)
FILES: dict[str, dict] = {}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(rel: str, arr: np.ndarray, dtype: str = "<f4") -> None:
    path = GOLD / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.ascontiguousarray(arr, dtype=dtype)
    path.write_bytes(a.tobytes(order="C"))
    FILES[rel] = {"shape": list(a.shape), "dtype": "float64" if dtype == "<f8" else "float32", "sha256": sha256(path)}


def export_reference() -> dict:
    checks = {}
    for fixture in hl.FIXTURES:
        for tag in TAGS:
            d = np.load(WORK / f"ref_{fixture}_{tag}.npz")
            probs, logits = d["probs"], d["logits"]
            assert probs.dtype == np.float32 and logits.dtype == np.float32 and probs.shape == logits.shape
            same = bool(np.array_equal(probs > 0.5, logits > 0)) and \
                bool(np.array_equal(probs > 0.5, 1.0 / (1.0 + np.exp(-logits.astype(np.float64))) > 0.5))
            write(f"{fixture}_{tag}_probs.f32le", probs)
            write(f"{fixture}_{tag}_logits.f32le", logits)
            checks[f"{fixture}_{tag}"] = {"n_frames": int(probs.shape[0]), "threshold_consistent": same,
                                          "min_abs_logit": float(np.abs(logits).min()),
                                          "tail_excluded": 16 if tag == "offline" else 0}
            print(f"[golden] {fixture} {tag}: {probs.shape[0]} frames, p>0.5 == logit>0 == sig64>0.5: {same}",
                  flush=True)
            assert same, (fixture, tag)
    return checks


def export_mels(assets: dict) -> None:
    for fixture, wav in hl.FIXTURES.items():
        audio = hl.load_audio(hl.FIXTURE_DIR / wav)
        chunks = list(mf.stream_chunks(audio.shape[0], *mf.STREAMING_MODES["low_latency"]))
        for k in (0, 1, len(chunks) - 1):
            s, e, first, _last = chunks[k]
            write(f"{fixture}_ll_mel_chunk{k}.f32le", mf.log_mel(audio[s:e], center=first, filters=assets["filters"]))
        write(f"{fixture}_offline_mel.f32le", mf.log_mel(audio, center=True, filters=assets["filters"]))


class Recorder:
    """Wraps SpeakerCache.update / compress to snapshot teacher-forced units (the class is patched only
    inside this process; host_loop.py itself is unchanged)."""

    def __init__(self):
        self.units: list[dict] = []
        self.prefix = ""
        self.step = -1
        self._orig_update = hl.SpeakerCache.update
        self._orig_compress = hl.SpeakerCache.compress
        rec = self

        def update(cache, packed_rows, logits, silence, n_chunk):
            rec.step += 1
            before = (cache.n_cache, cache.n_fifo, cache.probs.copy(), cache.is_compressed, cache.n_compress)
            n_after_fifo = cache.n_fifo + n_chunk
            pops = cache.num_popped(n_after_fifo) > 0
            cache._golden_compress = None
            rec._orig_update(cache, packed_rows, logits, silence, n_chunk)
            if rec.step in (0, 1) or pops:
                rec.save(cache, packed_rows, logits, n_chunk, before, cache._golden_compress)

        def compress(cache, embeds, probs, silence):
            cache._golden_compress = cache.frame_scores(probs)
            return rec._orig_compress(cache, embeds, probs, silence)

        hl.SpeakerCache.update = update
        hl.SpeakerCache.compress = compress

    def start(self, prefix: str):
        self.prefix, self.step = prefix, -1

    def close(self):
        hl.SpeakerCache.update = self._orig_update
        hl.SpeakerCache.compress = self._orig_compress

    def save(self, cache, rows, logits, n_chunk, before, scores):
        n_cache, n_fifo, probs_before, comp_before, n_comp_before = before
        L = rows.shape[0]                                   # hl.run passes [cache | FIFO | chunk] rows
        p = f"cache/{self.prefix}_s{self.step:03d}"
        write(f"{p}_rows.f32le", rows[:L])
        write(f"{p}_logits.f32le", logits)
        write(f"{p}_probs_before.f32le", probs_before.reshape(-1, hl.N_SPK))
        write(f"{p}_pooled.f32le", hl.pool_probs(logits))
        write(f"{p}_embeds_after.f32le", cache.embeds.reshape(-1, hl.HID))
        write(f"{p}_probs_after.f32le", cache.probs.reshape(-1, hl.N_SPK))
        write(f"{p}_fifo_after.f32le", cache.fifo.reshape(-1, hl.HID))
        unit = {"prefix": p, "step": self.step, "L": int(L), "n_cache": n_cache, "n_fifo": n_fifo, "n_chunk": int(n_chunk),
                "compressed_before": bool(comp_before), "compressed_after": bool(cache.is_compressed),
                "compressed_now": cache.n_compress > n_comp_before, "n_cache_after": cache.n_cache,
                "n_fifo_after": cache.n_fifo, "fifo_length": cache.fifo_length, "update_period": cache.update_period}
        if scores is not None:
            write(f"{p}_scores.f64le", scores, "<f8")
            unit["n_scored"] = int(scores.shape[0])
        self.units.append(unit)


def record_steps(steps, sink: list):
    for item in steps:
        sink.append(item)
        yield item


def export_eager(assets: dict, rec: Recorder) -> dict:
    index = {}
    engines = {p: hl.eager_engine(hl.PROFILES[p]["T"]) for p in ("streaming", "offline")}
    for fixture, wav in hl.FIXTURES.items():
        audio = hl.load_audio(hl.FIXTURE_DIR / wav)
        for tag, (profile, mode) in TAGS.items():
            fn = engines[profile]
            steps, n_frames = hl.build_steps(profile, mode, "host", assets, audio, fixture)
            seen: list = []
            packed: dict[int, np.ndarray] = {}

            def on_step(i, info):
                packed[i] = info["rows"]
            rec.start(f"{fixture}_{tag}")
            t0 = time.time()
            logits, cache = hl.run(fn, record_steps(steps, seen), profile, assets["silence"], n_frames,
                                   on_step=on_step)
            rows = np.concatenate([s[0] for s in seen], axis=0)
            write(f"chunks/{fixture}_{tag}_rows.f32le", rows)
            index[f"{fixture}_{tag}"] = {"profile": profile, "mode": mode, "n_frames": int(n_frames),
                                         "steps": [[int(s[0].shape[0]), int(s[1]), int(s[2])] for s in seen],
                                         "rows_file": f"chunks/{fixture}_{tag}_rows.f32le",
                                         "n_compress": cache.n_compress}
            if fixture == "diarization_example" and tag == "ll":
                last = len(seen) - 1
                for k in (*PACKED_STEPS, last):
                    write(f"{fixture}_ll_packed_step{k}.f32le", packed[k])
                index[f"{fixture}_{tag}"]["packed_steps"] = [*PACKED_STEPS, last]
                index[f"{fixture}_{tag}"]["packed_L"] = {str(k): int(packed[k].shape[0]) for k in (*PACKED_STEPS, last)}
            print(f"[eager] {fixture} {tag}: {len(seen)} steps, {logits.shape[0]} frames, {cache.n_compress} "
                  f"compressions, {time.time() - t0:.1f} s", flush=True)
    return index


def export_pygpu(assets: dict) -> dict:
    out = {}
    engines = {p: hl.CoreAIEngine(ART / f"n3d_{p}_float16.aimodel", "gpu") for p in ("streaming", "offline")}
    for fixture, wav in hl.FIXTURES.items():
        audio = hl.load_audio(hl.FIXTURE_DIR / wav)
        for tag, (profile, mode) in TAGS.items():
            steps, n_frames = hl.build_steps(profile, mode, "host", assets, audio, fixture)
            packed: dict[int, np.ndarray] = {}

            def on_step(i, info):
                if fixture == "diarization_example" and tag == "ll":
                    packed[i] = info["rows"]
            logits, cache = hl.run(engines[profile], steps, profile, assets["silence"], n_frames, on_step=on_step)
            write(f"pygpu/{fixture}_{tag}_logits.f32le", logits)
            if packed:
                last = max(packed)
                for k in (*PACKED_STEPS, last):
                    write(f"pygpu/{fixture}_ll_packed_step{k}.f32le", packed[k])
            out[f"{fixture}_{tag}"] = {"n_frames": int(logits.shape[0]), "n_compress": cache.n_compress,
                                       "n_ties": len(cache.ties)}
            print(f"[pygpu] {fixture} {tag}: {logits.shape[0]} frames, {cache.n_compress} compressions, "
                  f"{len(cache.ties)} near-ties", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pygpu", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    GOLD.mkdir(parents=True, exist_ok=True)
    for old in (GOLD / "cache").glob("*.f*le"):
        old.unlink()                # stale units of an earlier run must not outlive its index
    assets = hl.load_assets()
    report = {"reference": export_reference()}
    export_mels(assets)
    rec = Recorder()
    report["chunks"] = export_eager(assets, rec)
    (GOLD / "chunks").mkdir(exist_ok=True)
    (GOLD / "chunks" / "index.json").write_text(json.dumps(report["chunks"], indent=1))
    (GOLD / "cache").mkdir(exist_ok=True)
    (GOLD / "cache" / "index.json").write_text(json.dumps(rec.units, indent=1))
    report["cache_units"] = len(rec.units)
    report["cache_compressions"] = sum(u["compressed_now"] for u in rec.units)
    rec.close()                     # the Python GPU runs below are not recorded
    if not args.skip_pygpu:
        report["pygpu"] = export_pygpu(assets)
    report["files"] = FILES
    report["wall_s"] = round(time.time() - t0, 1)
    (GOLD / "golden.json").write_text(json.dumps(report, indent=1))
    print(f"[golden] {len(FILES)} files, {report['cache_units']} cache units "
          f"({report['cache_compressions']} compressions), {report['wall_s']} s -> _work/golden/", flush=True)


if __name__ == "__main__":
    main()
