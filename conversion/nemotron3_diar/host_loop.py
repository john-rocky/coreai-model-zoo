"""NumPy host loop of the Nemotron-3-Diarization port: audio -> log-mel -> embeds -> per-step packed
input -> graph -> speaker logits, with the speaker cache (Arrival-Order Speaker Cache + FIFO) kept
on the host. This is the algorithm the Swift host implements; gate_closed_loop.py gates it against
transformers. NumPy only, except the `eager` engine (the re-authored fp32 graph, torch) and the
`coreai` engine (coreai.runtime); runs in the export venv.

    ~/code/coreai/coreai-models/.venv/bin/python host_loop.py --fixture diarization_example \
        --engine coreai --bundle _work/artifacts/n3d_streaming_float16.aimodel --unit gpu

The speaker cache mirrors transformers' Nemotron3DiarizationSpeakerCache (modeling
_nemotron3_diarization.py @ 4b28d51, lines 54-255) line by line:

  update(packed rows, logits, n_chunk):
    probs = avg_pool8(sigmoid(logits[:L*8]))                       [L, 8] encoder-rate probabilities
    fifo += the step's n_chunk chunk rows (the look-ahead rows are fed again next step)
    if len(fifo) > fifo_length: pop = min(max(update_period, len(fifo) - fifo_length), len(fifo))
      fifo_probs = probs[n_cache : n_cache + len(fifo)]
      stored = cache probs saved with the rows if the cache is compressed, else probs[:n_cache]
               (an uncompressed cache holds plain chunk rows, whose probabilities each step re-estimates)
      cache += fifo[:pop] with (stored, fifo_probs[:pop]); fifo = fifo[pop:]
      if len(cache) > 264: compress -> 264 rows, is_compressed = True
  compress(embeds [N,512], probs [N,8]):
    scores = log(max(p,.25)) - log(max(1-p,.25)) + sum_k log(max(1-p_k,.25)) - log .5
    non-speech (p <= .5) -> -inf; if a speaker has >= min_positive_scores (16) positive frames,
    its non-positive speech frames -> -inf too
    scores[264:] += 0.05 (the frames just popped from the FIFO)
    per speaker, top 24 frames += 2 log 2, then top 48 += log 2
    one +inf row appended (1 silence slot per speaker: row N = silence_embeds, zero probs)
    top 264 of the speaker-major flat scores [8 * (N+1)] (unsorted), -inf picks -> sentinel,
    ascending sort, frame = N for the sentinel else idx % (N+1); gather embeds / probs

  The scores are computed in float64 from the float32 probabilities (transformers computes them in
  float32). Measured on the transformers capture, teacher-forced (every compression of both fixtures:
  4 low-latency + 5 offline): transformers' own float32 scores come within 2 ulp (1.19e-7) of a top-k
  boundary (97.6 s, step 78, weak boost of speaker 0); NumPy's float32 log/sum rounding breaks that
  near-tie the other way and the cache diverges from then on, while float64 scores pick exactly
  transformers' rows at all 9 compressions (NumPy selection on transformers' own scores does too).
  Top-k ties are broken by the lower index; boundary gaps under 4 float32 ulp are logged
  (SpeakerCache.ties) as decisions that sit inside float32 rounding.

Profiles (graph T fixed per profile):
  streaming  mode low_latency (chunk 9, look-ahead 4) / very_low_latency (6, 2) /
             ultra_low_latency (3, 1); FIFO 264, update period 222; T = 541 for all three modes.
             Each chunk's mel is computed from its own audio slice (mel_frontend.stream_chunks, the
             model card's inputs_generator), the last chunk takes the rest of the audio, has no
             look-ahead and emits its true mel frame count (its last encoder row is zero-padded in
             the log-mel domain, like transformers' embedder).
  offline    whole-recording mel (center=True, one pass), then chunks of 340 encoder rows with up to
             40 look-ahead rows; FIFO 40, update period 300; T = 684; output cut to the recording's
             mel frame count. The feature extractor's extra centered frame (masked in transformers)
             is not produced here: log_mel returns the valid frames only, so no embed row is masked.

Mel source: --mel host (mel_frontend.py + the shipped .f32le constants) or --mel ref (the
transformers input_features captured in _work/chunk_io_<fixture>_<mode>.npz; separates mel error
from loop error).

Poison switches (negative controls for the gate): no-compress (an overflowing cache is truncated to
its first 264 rows instead of compressed, is_compressed still set) and no-pop (frames that leave
the FIFO are dropped instead of moved to the cache).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True  # importing conversion/_paths must not leave a __pycache__ outside this dir
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import mel_frontend as mf  # noqa: E402
from _paths import work_path  # noqa: E402

WORK = HERE / "_work"
ART = WORK / "artifacts"

# config.json (streaming_config) / processor_config.json of nvidia/Nemotron-3-Diarization @ f667ed7
HID, N_SPK, SUB = 512, 8, 8
CACHE_LEN = 264
SIL_PER_SPK = 1
PRED_THRESHOLD = 0.25
LATEST_BOOST = 0.05
MIN_POS_RATE, STRONG_RATE, WEAK_RATE = 0.5, 0.75, 1.5
PROFILES = {
    "streaming": {"fifo_length": 264, "update_period": 222, "T": 541},
    "offline": {"fifo_length": 40, "update_period": 300, "T": 684, "chunk": 340, "lookahead": 40},
}
MODES = mf.STREAMING_MODES
MODE_TAG = {"low_latency": "ll", "very_low_latency": "vll", "ultra_low_latency": "ull"}
FIXTURES = {"diarization_example": "diarization_example_16k.wav", "test_multispk": "test_multispk_16k.wav"}
FIXTURE_DIR = work_path("_n3d", "fixtures")        # <work root>/_n3d/fixtures (ZOO_WORK_ROOT moves it); --fixtures

LOG_HALF = math.log(0.5)
STRONG_BOOST = -2.0 * math.log(0.5)
WEAK_BOOST = -math.log(0.5)
NEAR_TIE_ULP = 4           # boundary gaps below this many float32 ulp are logged

GraphFn = Callable[[np.ndarray, np.ndarray], np.ndarray]   # (packed [1,T,512], valid [1,T]) -> [T*8, 8]


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (np.float32(1.0) / (np.float32(1.0) + np.exp(-x))).astype(np.float32)


def pool_probs(logits: np.ndarray) -> np.ndarray:
    """[L*8, 8] logits -> [L, 8] probabilities at the encoder rate (avg_pool1d(sigmoid, 8, 8))."""
    p = sigmoid(logits)
    return (p.reshape(-1, SUB, N_SPK).sum(axis=1) / np.float32(SUB)).astype(np.float32)


def topk_desc(values: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest values; ties broken by the lower index (stable)."""
    return np.argsort(-values, kind="stable")[:k]


class SpeakerCache:
    def __init__(self, fifo_length: int, update_period: int, poison: str | None = None):
        self.fifo_length = fifo_length
        self.update_period = update_period
        budget = CACHE_LEN // N_SPK - SIL_PER_SPK                      # 32 rows per speaker
        self.min_positive_scores = math.floor(budget * MIN_POS_RATE)    # 16
        self.n_strong = math.floor(budget * STRONG_RATE)                # 24
        self.n_weak = math.floor(budget * WEAK_RATE)                    # 48
        self.embeds = np.zeros((0, HID), np.float32)
        self.probs = np.zeros((0, N_SPK), np.float32)
        self.fifo = np.zeros((0, HID), np.float32)
        self.is_compressed = False
        self.poison = poison
        self.n_compress = 0
        self.ties: list[dict] = []          # exact ties at a top-k boundary (tie-break dependent picks)

    @property
    def n_cache(self) -> int:
        return self.embeds.shape[0]

    @property
    def n_fifo(self) -> int:
        return self.fifo.shape[0]

    def get_embeds(self) -> np.ndarray:
        return np.concatenate([self.embeds, self.fifo], axis=0)

    def num_popped(self, n: int) -> int:
        if n <= self.fifo_length:
            return 0
        return min(max(self.update_period, n - self.fifo_length), n)

    def update(self, packed_rows: np.ndarray, logits: np.ndarray, silence: np.ndarray, n_chunk: int):
        n_cache, n_fifo = self.n_cache, self.n_fifo
        probs = pool_probs(logits)                                      # [L, 8]
        start = n_cache + n_fifo
        fifo = np.concatenate([self.fifo, packed_rows[start:start + n_chunk]], axis=0)
        pop = self.num_popped(fifo.shape[0])
        if pop:
            fifo_probs = probs[n_cache:n_cache + fifo.shape[0]]
            stored = self.probs if self.is_compressed else probs[:n_cache]
            if self.poison == "no-pop":
                cache_e, cache_p = self.embeds, stored
            else:
                cache_e = np.concatenate([self.embeds, fifo[:pop]], axis=0)
                cache_p = np.concatenate([stored, fifo_probs[:pop]], axis=0)
            fifo = fifo[pop:]
            if cache_e.shape[0] > CACHE_LEN:
                if self.poison == "no-compress":
                    cache_e, cache_p = cache_e[:CACHE_LEN], cache_p[:CACHE_LEN]
                else:
                    cache_e, cache_p = self.compress(cache_e, cache_p, silence)
                self.is_compressed = True
                self.n_compress += 1
            self.embeds, self.probs = cache_e.astype(np.float32), cache_p.astype(np.float32)
        self.fifo = fifo

    # -- compression (AOSC) --
    def frame_scores(self, probs: np.ndarray) -> np.ndarray:
        """float64 scores [N, 8] from the float32 probabilities (see the module docstring)."""
        p = probs.astype(np.float64)
        log_p = np.log(np.maximum(p, PRED_THRESHOLD))
        log_c = np.log(np.maximum(1.0 - p, PRED_THRESHOLD))
        scores = log_p - log_c + log_c.sum(axis=1, keepdims=True) - LOG_HALF
        is_speech = probs > 0.5
        scores = np.where(is_speech, scores, -np.inf)
        is_pos = scores > 0
        enough = is_pos.sum(axis=0, keepdims=True) >= self.min_positive_scores
        return np.where(~is_pos & is_speech & enough, -np.inf, scores)

    def _note_tie(self, where: str, values: np.ndarray, k: int, spk: int | None = None):
        if k >= values.size:
            return
        srt = np.sort(values)[::-1]
        kth, nxt = srt[k - 1], srt[k]
        if not (np.isfinite(kth) and np.isfinite(nxt)):
            return
        ulp = float(np.spacing(np.float32(abs(nxt))))
        gap = float(kth - nxt)
        if gap < NEAR_TIE_ULP * ulp:
            self.ties.append({"compress": self.n_compress, "where": where, "speaker": spk, "value": float(kth),
                              "gap": gap, "gap_f32_ulp": gap / ulp, "n_equal": int((values == kth).sum())})

    def boost(self, scores: np.ndarray, k: int, boost: float, where: str) -> np.ndarray:
        out = scores.copy()
        for s in range(N_SPK):
            col = scores[:, s]
            self._note_tie(where, col, k, s)
            idx = topk_desc(col, k)
            out[idx, s] = out[idx, s] + boost
        return out

    def compress(self, embeds: np.ndarray, probs: np.ndarray, silence: np.ndarray):
        n = probs.shape[0]
        scores = self.frame_scores(probs)
        scores[CACHE_LEN:] = scores[CACHE_LEN:] + LATEST_BOOST
        scores = self.boost(scores, self.n_strong, STRONG_BOOST, "strong")
        scores = self.boost(scores, self.n_weak, WEAK_BOOST, "weak")
        scores = np.concatenate([scores, np.full((SIL_PER_SPK, N_SPK), np.inf)], axis=0)
        embeds = np.concatenate([embeds, silence.reshape(1, HID).astype(np.float32)], axis=0)
        probs = np.concatenate([probs, np.zeros((1, N_SPK), np.float32)], axis=0)
        n_scored = n + SIL_PER_SPK
        sentinel = n_scored * N_SPK
        flat = scores.T.reshape(-1)                                     # speaker-major
        self._note_tie("select", flat, CACHE_LEN)
        idx = topk_desc(flat, CACHE_LEN)
        idx = np.where(flat[idx] == -np.inf, sentinel, idx)
        idx = np.sort(idx)
        frames = np.where(idx == sentinel, n, np.minimum(idx % n_scored, n))
        return embeds[frames], probs[frames]


# ----------------------------------------------------------------------------- steps

def stream_steps(audio: np.ndarray | None, mode: str, projection: np.ndarray, filters: np.ndarray | None,
                 ref_mels: list[np.ndarray] | None = None) -> Iterator[tuple[np.ndarray, int, int]]:
    """(chunk rows incl. look-ahead [n, 512], n_lookahead, n_emit mel frames) per streaming chunk.
    Host mel from the audio (mel_frontend), or the captured per-chunk processor output (ref_mels)."""
    c, la = MODES[mode]
    if ref_mels is not None:
        chunks = [(m, i == len(ref_mels) - 1) for i, m in enumerate(ref_mels)]
    else:
        chunks = ((mf.log_mel(audio[s:e], center=first, filters=filters), last)
                  for s, e, first, last in mf.stream_chunks(audio.shape[0], c, la))
    for mel, last in chunks:
        emb = mf.embed(mel, projection)
        if last:
            yield emb, 0, mel.shape[0]
        else:
            assert mel.shape[0] == (c + la) * SUB, (mel.shape, c, la)
            yield emb, la, c * SUB


def offline_steps(mel: np.ndarray, projection: np.ndarray, chunk: int = 340,
                  lookahead: int = 40) -> Iterator[tuple[np.ndarray, int, int]]:
    """Whole-recording embeds cut into chunks of `chunk` rows, each with up to `lookahead` rows of
    the next; the emitted count of the last step is cut to the mel frame count by run()."""
    emb = mf.embed(mel, projection)
    n = emb.shape[0]
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        la = min(lookahead, n - e)
        yield emb[s:e + la], la, (e - s) * SUB


def run(graph_fn: GraphFn, steps: Iterator[tuple[np.ndarray, int, int]], profile: str, silence: np.ndarray,
        n_frames: int | None = None, poison: str | None = None,
        on_step: Callable[[int, dict], None] | None = None) -> tuple[np.ndarray, SpeakerCache]:
    """Drive the loop. Returns the emitted logits [N, 8] (cut to n_frames if given) and the cache."""
    cfg = PROFILES[profile]
    T = cfg["T"]
    cache = SpeakerCache(cfg["fifo_length"], cfg["update_period"], poison)
    out = []
    packed = np.zeros((1, T, HID), np.float32)
    valid = np.zeros((1, T), np.float32)
    for i, (chunk_rows, n_la, n_emit) in enumerate(steps):
        n_cache, n_fifo = cache.n_cache, cache.n_fifo
        rows = np.concatenate([cache.get_embeds(), chunk_rows], axis=0)
        L = rows.shape[0]
        assert L <= T, (i, L, T)
        packed[:] = 0.0
        packed[0, :L] = rows
        valid[:] = 0.0
        valid[0, :L] = 1.0
        t0 = time.perf_counter()
        full = graph_fn(packed, valid)
        dt = time.perf_counter() - t0
        logits = np.asarray(full, np.float32).reshape(T * SUB, N_SPK)[: L * SUB]
        n_chunk = chunk_rows.shape[0] - n_la
        compressed_before = cache.is_compressed
        cache.update(rows, logits, silence, n_chunk)
        s = (n_cache + n_fifo) * SUB
        emitted = logits[s:s + min(n_chunk * SUB, n_emit)]
        out.append(emitted)
        if on_step is not None:
            on_step(i, {"rows": rows, "logits": logits, "L": L, "n_cache": n_cache, "n_fifo": n_fifo,
                        "n_chunk": n_chunk, "n_lookahead": n_la, "compressed_before": compressed_before,
                        "compressed_after": cache.is_compressed, "cache_after": cache.n_cache,
                        "fifo_after": cache.n_fifo, "cache_probs_after": cache.probs, "graph_s": dt,
                        "emitted": emitted})
    logits = np.concatenate(out, axis=0)
    if n_frames is not None:
        logits = logits[:n_frames]
    return logits, cache


# ----------------------------------------------------------------------------- engines

def eager_engine(T: int, **switches) -> GraphFn:
    import torch

    from n3d_model import load_checkpoint

    graph, _ = load_checkpoint(T=T, **switches)

    def fn(packed, valid):
        with torch.inference_mode():
            return graph(torch.from_numpy(packed), torch.from_numpy(valid))[0].numpy()
    return fn


def coreai_options(rt, unit: str):
    if unit == "cpu":
        return rt.SpecializationOptions.cpu_only()
    if unit == "gpu":
        return rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    if unit == "ane":
        return rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.neural_engine())
    if unit == "default":
        return rt.SpecializationOptions.default()
    raise ValueError(unit)


class CoreAIEngine:
    """Sync wrapper around coreai.runtime: load once, call per step. Keeps the model referenced."""

    def __init__(self, bundle: str | Path, unit: str):
        import asyncio

        import coreai.runtime as rt

        self.rt = rt
        self.loop = asyncio.new_event_loop()
        t0 = time.time()
        self.model = self.loop.run_until_complete(rt.AIModel.load(Path(bundle), coreai_options(rt, unit)))
        self.fn = self.model.load_function("main")
        self.load_s = time.time() - t0
        self.bundle, self.unit = str(bundle), unit

    def __call__(self, packed: np.ndarray, valid: np.ndarray) -> np.ndarray:
        res = self.loop.run_until_complete(self.fn({"packed": self.rt.NDArray(packed),
                                                    "valid": self.rt.NDArray(valid)}))
        return res["logits"].numpy()[0]


# ----------------------------------------------------------------------------- helpers

def load_assets() -> dict[str, np.ndarray]:
    return {"projection": mf.load_f32le(ART / "embedder_projection.f32le", (HID, SUB * mf.N_MELS)),
            "silence": mf.load_f32le(ART / "silence_embeds.f32le", (HID,)),
            "filters": mf.mel_filters()}


def load_audio(path: str | Path) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32")
    assert sr == mf.SR and audio.ndim == 1, (sr, audio.shape)
    return audio


def ref_tag(profile: str, mode: str) -> str:
    return "offline" if profile == "offline" else MODE_TAG[mode]


def build_steps(profile: str, mode: str, mel_src: str, assets: dict, audio: np.ndarray | None,
                fixture: str | None) -> tuple[Iterator, int]:
    """(steps iterator, number of mel frames the output is cut to)."""
    if profile == "streaming":
        c, la = MODES[mode]
        if mel_src == "ref":
            d = np.load(WORK / f"chunk_io_{fixture}_{ref_tag(profile, mode)}.npz")
            mels = [d[f"s{i:03d}_input_features"] for i in range(int(d["n_steps"]))]
        else:
            mels = None
        if mels is not None:
            n_frames = c * SUB * (len(mels) - 1) + mels[-1].shape[0]
        else:
            n_frames = sum(min(c * SUB, mf.n_valid_frames(e - s, first)) if not last
                           else mf.n_valid_frames(e - s, first)
                           for s, e, first, last in mf.stream_chunks(audio.shape[0], c, la))
        return stream_steps(audio, mode, assets["projection"], assets["filters"], mels), n_frames
    cfg = PROFILES["offline"]
    if mel_src == "ref":
        d = np.load(WORK / f"chunk_io_{fixture}_offline.npz")
        r = np.load(WORK / f"ref_{fixture}_offline.npz")
        mel = d["input_features"][: int(r["attention_mask"].sum())]   # drop the masked centered frame
    else:
        mel = mf.log_mel(audio, center=True, filters=assets["filters"])
    return offline_steps(mel, assets["projection"], cfg["chunk"], cfg["lookahead"]), mel.shape[0]


def speaker_segments(logits: np.ndarray, threshold: float = 0.5, frame_s: float = 0.01) -> list[dict]:
    """transformers' Nemotron3DiarizationProcessor.extract_speaker_dict for one stream."""
    active = (sigmoid(logits) > threshold).astype(np.int8)
    z = np.zeros((1, active.shape[1]), np.int8)
    changes = np.diff(np.concatenate([z, active, z], axis=0), axis=0)
    segs = []
    for spk in range(active.shape[1]):
        starts = np.nonzero(changes[:, spk] == 1)[0]
        ends = np.nonzero(changes[:, spk] == -1)[0]
        segs.extend({"Start": round(int(a) * frame_s, 2), "End": round(int(b) * frame_s, 2), "Speaker": spk,
                     "start_frame": int(a), "end_frame": int(b)} for a, b in zip(starts, ends))
    segs.sort(key=lambda g: (g["Start"], g["Speaker"]))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", choices=list(FIXTURES))
    ap.add_argument("--fixtures", help=f"directory of the fixture wavs (default {FIXTURE_DIR})")
    ap.add_argument("--wav")
    ap.add_argument("--profile", choices=list(PROFILES), default="streaming")
    ap.add_argument("--mode", choices=list(MODES), default="low_latency")
    ap.add_argument("--mel", choices=["host", "ref"], default="host")
    ap.add_argument("--engine", choices=["eager", "coreai"], default="eager")
    ap.add_argument("--bundle")
    ap.add_argument("--unit", choices=["cpu", "gpu", "ane", "default"], default="gpu")
    ap.add_argument("--poison", choices=["no-compress", "no-pop"])
    ap.add_argument("--out", help="npz with logits / probs / segments")
    args = ap.parse_args()
    if (args.fixture is None) == (args.wav is None):
        ap.error("give exactly one of --fixture / --wav")
    if args.mel == "ref" and args.fixture is None:
        ap.error("--mel ref needs --fixture (the captured input_features)")

    assets = load_assets()
    audio = load_audio(args.wav or Path(args.fixtures or FIXTURE_DIR) / FIXTURES[args.fixture])
    T = PROFILES[args.profile]["T"]
    if args.engine == "eager":
        fn = eager_engine(T)
    else:
        fn = CoreAIEngine(args.bundle, args.unit)
        print(f"[coreai] loaded {args.bundle} ({args.unit}) in {fn.load_s:.1f} s", flush=True)
    steps, n_frames = build_steps(args.profile, args.mode, args.mel, assets, audio, args.fixture)
    t0 = time.time()
    logits, cache = run(fn, steps, args.profile, assets["silence"], n_frames, args.poison)
    wall = time.time() - t0
    segs = speaker_segments(logits)
    print(f"[host_loop] {args.profile}/{args.mode} mel={args.mel} engine={args.engine}: {logits.shape[0]} frames "
          f"({audio.shape[0] / mf.SR:.2f} s audio), {cache.n_compress} compressions, {len(segs)} segments, "
          f"wall {wall:.1f} s, ties at top-k boundaries {len(cache.ties)}")
    if args.out:
        np.savez(args.out, logits=logits, probs=sigmoid(logits),
                 segments=np.array(json.dumps([{k: g[k] for k in ("Start", "End", "Speaker")} for g in segs])))


if __name__ == "__main__":
    main()
