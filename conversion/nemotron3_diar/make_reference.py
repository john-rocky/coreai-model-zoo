"""Oracle for the Nemotron-3-Diarization port: transformers (git main) in fp32, NeMo not involved.
Run in the oracle venv (torch 2.9.0 + transformers from git):

    ~/code/coreai/_n3d/venv-oracle/bin/python make_reference.py [--fixtures diarization_example,test_multispk]
    ~/code/coreai/_n3d/venv-oracle/bin/python make_reference.py --mode very_low_latency,ultra_low_latency

For every fixture and mode (offline = `processor(audio)` -> `model(**inputs)`; ll = low_latency
streaming, chunked exactly like the model card's `inputs_generator`; --mode adds the processor's
other streaming modes, file tags vll = very_low_latency, ull = ultra_low_latency; default
--mode offline,low_latency) it writes to _work/:

  ref_<fixture>_<mode>.npz       logits [N, 8], probs, segments (extract_speaker_dict @0.5, json),
                                 offline also input_features [N, 128] + attention_mask
  chunk_io_<fixture>_<mode>.npz  every encoder step: s###_inputs_embeds [L, 512] (the
                                 `chunk_input_embeds` fed to Nemotron3DiarizationModel),
                                 s###_chunk_logits [L*8, 8] (classifier output), the speaker-cache
                                 state before/after the update, the emitted row range; ll also has
                                 the per-chunk processor output (s###_input_features) and embedder
                                 output (s###_embedder_out)

The capture run (forward hooks + a pass-through wrapper on the speaker-cache update that only reads
its state) is repeated without any instrumentation, and the two logit streams must be bit-identical.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True  # importing conversion/_paths must not leave a __pycache__ outside this dir
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0]))
from _paths import hf_snapshot, work_path  # noqa: E402
from n3d_model import REPO_ID, REVISION, SAFETENSORS_SHA256, sha256_file  # noqa: E402

WORK = HERE / "_work"
FIXTURES = {
    "diarization_example": "diarization_example_16k.wav",  # 97.60 s, the transformers integration-test audio
    "test_multispk": "test_multispk_16k.wav",              # 21.50 s, the 4spk port's fixture
}
EXPECTED_BUCKET = "hf-internal-testing/nemotron3-diarization-integration-test"
MODE_TAGS = {"low_latency": "ll", "very_low_latency": "vll", "ultra_low_latency": "ull"}


def transformers_commit() -> str:
    try:
        info = json.loads(metadata.distribution("transformers").read_text("direct_url.json") or "{}")
        return info.get("vcs_info", {}).get("commit_id", "unknown")
    except Exception:  # noqa: BLE001
        return "unknown"


def load_audio(name: str) -> np.ndarray:
    audio, sr = sf.read(work_path("_n3d", "fixtures", FIXTURES[name]), dtype="float32")
    assert sr == 16000 and audio.ndim == 1, (sr, audio.shape)
    return audio


def inputs_generator(processor, audio, sampling_rate):
    """The model card's streaming chunker, verbatim, plus the sample range of every chunk."""
    end = processor.num_samples_first_audio_chunk
    yield 0, end, processor(audio[:end], sampling_rate=sampling_rate, is_streaming=True,
                            is_first_audio_chunk=True)

    mel_frame_idx = processor.num_mel_frames_per_step
    start_idx = processor.audio_chunk_start(mel_frame_idx)
    while (end_idx := start_idx + processor.num_samples_per_audio_chunk) <= audio.shape[0]:
        yield start_idx, end_idx, processor(audio[start_idx:end_idx], sampling_rate=sampling_rate,
                                            is_streaming=True, is_first_audio_chunk=False)
        mel_frame_idx += processor.num_mel_frames_per_step
        start_idx = processor.audio_chunk_start(mel_frame_idx)

    yield start_idx, audio.shape[0], processor(audio[start_idx:], sampling_rate=sampling_rate,
                                               is_streaming=True, is_first_audio_chunk=False,
                                               is_last_audio_chunk=True)


@contextlib.contextmanager
def capture(model, cache_cls):
    """Records every encoder step. Hooks and the update wrapper read tensors, never change them."""
    rec = {"inputs_embeds": [], "step_mask": [], "chunk_logits": [], "embedder_out": [], "updates": []}

    def model_pre_hook(_mod, _args, kwargs):
        rec["inputs_embeds"].append(kwargs["inputs_embeds"].detach().clone())
        m = kwargs.get("attention_mask")
        rec["step_mask"].append(None if m is None else m.detach().clone())

    def classifier_hook(_mod, _args, out):
        rec["chunk_logits"].append(out.detach().clone())

    def embedder_hook(_mod, _args, out):
        rec["embedder_out"].append(out.detach().clone())

    orig_update = cache_cls.update

    def update(self, chunk_input_embeds, chunk_logits, silence_embeds, num_chunk_frames, mask=None):
        before = (self.num_cache_frames, self.num_fifo_frames, self.is_compressed)
        out = orig_update(self, chunk_input_embeds, chunk_logits, silence_embeds, num_chunk_frames, mask=mask)
        rec["updates"].append({
            "before": before,
            "after": (self.num_cache_frames, self.num_fifo_frames, self.is_compressed),
            "num_chunk_frames": int(num_chunk_frames),
            "cache_probs_after": self.probs[0, : self.num_cache_frames].detach().clone(),
        })
        return out

    handles = [
        model.model.register_forward_pre_hook(model_pre_hook, with_kwargs=True),
        model.classifier.register_forward_hook(classifier_hook),
        model.model.audio_tower.embedder.register_forward_hook(embedder_hook),
    ]
    cache_cls.update = update
    try:
        yield rec
    finally:
        cache_cls.update = orig_update
        for h in handles:
            h.remove()


def run_offline(model, processor, audio):
    inputs = processor(audio, sampling_rate=16000).to("cpu", dtype=torch.float32)
    with torch.inference_mode():
        logits = model(**inputs).logits
    return inputs, logits


def run_streaming(model, processor, audio):
    chunks, logits, cache = [], [], None
    with torch.inference_mode():
        for start, end, inputs in inputs_generator(processor, audio, 16000):
            inputs = inputs.to("cpu", dtype=torch.float32)
            out = model(**inputs, speaker_cache=cache)
            cache = out.speaker_cache
            logits.append(out.logits)
            chunks.append({
                "audio_start": start, "audio_end": end,
                "input_features": inputs["input_features"][0].clone(),
                "attention_mask": inputs["attention_mask"][0].clone(),
                "num_lookahead": int(inputs.get("num_lookahead_frames", 0) or 0),
                "emit_len": out.logits.shape[1],
                "state_after": (cache.num_cache_frames, cache.num_fifo_frames, cache.is_compressed),
            })
    return chunks, torch.cat(logits, dim=1)


def to_np(t):
    return t.detach().float().cpu().numpy()


def meta(fixture, mode, n_samples, model):
    return {
        "transformers_commit": np.array(transformers_commit()),
        "transformers_version": np.array(metadata.version("transformers")),
        "torch_version": np.array(torch.__version__),
        "python": np.array(platform.python_version()),
        "hf_revision": np.array(REVISION),
        "attn_implementation": np.array(str(model.config._attn_implementation)),
        "fixture": np.array(fixture),
        "mode": np.array(mode),
        "n_samples": np.array(n_samples),
    }


def step_arrays(rec, emit_start_fn, allow_mask=False):
    """Per-step arrays shared by both modes. emit_start_fn(i, cached_length) -> row in chunk_logits.

    Streaming steps never carry padding (the processor drops the invalid frames). Offline steps can:
    when the recording's last mel frame is invalid and starts an 8-frame group, its encoder frame is
    masked; that step's mask is kept as s###_step_mask."""
    n = len(rec["inputs_embeds"])
    assert n == len(rec["chunk_logits"]) == len(rec["updates"]), \
        (n, len(rec["chunk_logits"]), len(rec["updates"]))
    out = {}
    L, cb, fb, xb, ca, fa, xa, ncf, es = ([] for _ in range(9))
    for i in range(n):
        emb = rec["inputs_embeds"][i]
        lg = rec["chunk_logits"][i]
        up = rec["updates"][i]
        assert emb.shape[0] == 1 and lg.shape[1] == emb.shape[1] * 8, (emb.shape, lg.shape)
        mask = rec["step_mask"][i]
        if mask is not None and not bool(mask.all()):
            assert allow_mask, f"step {i}: padding in the step mask"
            out[f"s{i:03d}_step_mask"] = mask[0].numpy().astype(np.int8)
        L.append(emb.shape[1])
        (c0, f0, x0), (c1, f1, x1) = up["before"], up["after"]
        cb.append(c0); fb.append(f0); xb.append(x0); ca.append(c1); fa.append(f1); xa.append(x1)
        ncf.append(up["num_chunk_frames"])
        es.append(emit_start_fn(i, c0 + f0))
        out[f"s{i:03d}_inputs_embeds"] = to_np(emb[0])
        out[f"s{i:03d}_chunk_logits"] = to_np(lg[0])
        out[f"s{i:03d}_cache_probs_after"] = to_np(up["cache_probs_after"])
    out.update({
        "n_steps": np.array(n), "L": np.array(L), "num_chunk_frames": np.array(ncf),
        "cache_before": np.array(cb), "fifo_before": np.array(fb), "compressed_before": np.array(xb),
        "cache_after": np.array(ca), "fifo_after": np.array(fa), "compressed_after": np.array(xa),
        "emit_start": np.array(es),
    })
    return out


def do_fixture(name, model, processor, cache_cls, log, modes=("offline", "low_latency")):
    audio = load_audio(name)
    log(f"[{name}] {audio.shape[0]} samples = {audio.shape[0] / 16000:.2f} s")
    if "offline" in modes:
        do_offline(name, audio, model, processor, cache_cls, log)
    for mode in modes:
        if mode != "offline":
            do_streaming(name, audio, mode, model, processor, cache_cls, log)
    return {tag: np.load(WORK / f"ref_{name}_{tag}.npz")["probs"] for tag in ("offline", "ll")
            if (WORK / f"ref_{name}_{tag}.npz").exists()}


def do_offline(name, audio, model, processor, cache_cls, log):
    t0 = time.time()
    inputs, logits = run_offline(model, processor, audio)
    with capture(model, cache_cls) as rec:
        inputs2, logits2 = run_offline(model, processor, audio)
    assert torch.equal(logits, logits2), "offline: instrumented run differs from the plain run"
    feats = inputs["input_features"][0]
    amask = inputs["attention_mask"][0]
    segments = processor.extract_speaker_dict(logits, inputs["attention_mask"])[0]
    np.savez(WORK / f"ref_{name}_offline.npz", logits=to_np(logits[0]), probs=to_np(logits[0].sigmoid()),
             segments=np.array(json.dumps(segments)), input_features=to_np(feats),
             attention_mask=amask.numpy().astype(np.int32), **meta(name, "offline", audio.shape[0], model))

    # offline chunks: the forward walks chunk_length=340 steps with up to 40 look-ahead frames
    assert len(rec["embedder_out"]) == 1
    n_emb = rec["embedder_out"][0].shape[1]
    arr = step_arrays(rec, lambda i, cached: cached * 8, allow_mask=True)
    emit_len, out_start, n_la = [], [], []
    pos = 0
    for i, s in enumerate(range(0, n_emb, 340)):
        ncf = min(s + 340, n_emb) - s
        assert ncf == arr["num_chunk_frames"][i]
        n_la.append(arr["L"][i] - arr["cache_before"][i] - arr["fifo_before"][i] - ncf)
        emit_len.append(ncf * 8)
        out_start.append(pos)
        pos += ncf * 8
    assert len(emit_len) == int(arr["n_steps"]), (len(emit_len), int(arr["n_steps"]))
    # the forward truncates the concatenation to the mel frame count
    rebuilt = torch.cat([torch.from_numpy(arr[f"s{i:03d}_chunk_logits"][arr["emit_start"][i]:arr["emit_start"][i] + emit_len[i]])
                         for i in range(len(emit_len))])[: logits.shape[1]]
    assert torch.equal(rebuilt, logits[0]), "offline: per-step emitted rows do not rebuild the logits"
    np.savez(WORK / f"chunk_io_{name}_offline.npz", **arr, emit_len=np.array(emit_len),
             out_start=np.array(out_start), num_lookahead=np.array(n_la),
             input_features=to_np(feats), embedder_out=to_np(rec["embedder_out"][0][0]),
             **meta(name, "offline", audio.shape[0], model))
    log(f"[{name}] offline: {logits.shape[1]} frames, {int(arr['n_steps'])} steps, "
        f"L max {int(arr['L'].max())}, compressed from step "
        f"{int(np.argmax(arr['compressed_after'])) if arr['compressed_after'].any() else 'never'}, "
        f"{len(segments)} segments, {time.time() - t0:.1f} s")

def do_streaming(name, audio, mode, model, processor, cache_cls, log):
    """One streaming mode of the processor; file tag ll / vll / ull."""
    tag = MODE_TAGS[mode]
    t0 = time.time()
    processor.set_streaming_mode(mode)
    chunks, logits = run_streaming(model, processor, audio)
    with capture(model, cache_cls) as rec:
        chunks2, logits2 = run_streaming(model, processor, audio)
    assert torch.equal(logits, logits2), f"{tag}: instrumented run differs from the plain run"
    segments = processor.extract_speaker_dict(logits)[0]
    np.savez(WORK / f"ref_{name}_{tag}.npz", logits=to_np(logits[0]), probs=to_np(logits[0].sigmoid()),
             segments=np.array(json.dumps(segments)), **meta(name, tag, audio.shape[0], model))

    arr = step_arrays(rec, lambda i, cached: cached * 8)
    n = int(arr["n_steps"])
    assert n == len(chunks), (n, len(chunks))
    out_start, pos = [], 0
    for i, ch in enumerate(chunks):
        assert (int(arr["cache_after"][i]), int(arr["fifo_after"][i]), bool(arr["compressed_after"][i])) \
            == tuple(ch["state_after"]), f"step {i}: wrapper state != returned speaker_cache"
        assert bool(ch["attention_mask"].all()), f"step {i}: padded processor frames"
        es, el = int(arr["emit_start"][i]), ch["emit_len"]
        assert torch.equal(torch.from_numpy(arr[f"s{i:03d}_chunk_logits"][es:es + el]), logits[0, pos:pos + el]), \
            f"step {i}: emitted rows differ from the chunk logits"
        assert rec["embedder_out"][i].shape[1] == arr["L"][i] - arr["cache_before"][i] - arr["fifo_before"][i]
        arr[f"s{i:03d}_input_features"] = to_np(ch["input_features"])
        arr[f"s{i:03d}_embedder_out"] = to_np(rec["embedder_out"][i][0])
        out_start.append(pos)
        pos += el
    assert pos == logits.shape[1]
    np.savez(WORK / f"chunk_io_{name}_{tag}.npz", **arr,
             emit_len=np.array([c["emit_len"] for c in chunks]), out_start=np.array(out_start),
             num_lookahead=np.array([c["num_lookahead"] for c in chunks]),
             audio_start=np.array([c["audio_start"] for c in chunks]),
             audio_end=np.array([c["audio_end"] for c in chunks]),
             mel_frames=np.array([c["input_features"].shape[0] for c in chunks]),
             **meta(name, tag, audio.shape[0], model))
    comp = np.nonzero(arr["compressed_after"])[0]
    log(f"[{name}] {tag}: {logits.shape[1]} frames, {n} steps, L max {int(arr['L'].max())} "
        f"(L=541 at {int((arr['L'] == 541).sum())} steps), first compressed step "
        f"{int(comp[0]) if comp.size else 'never'}, {len(segments)} segments, {time.time() - t0:.1f} s")


def compare_expected(probs_long, log):
    """Optional: the integration test's expected probabilities (NeMo-derived), fetched the way the
    test does (huggingface_hub.download_bucket_files). Reported only; not a gate of this round.

    offline_long: our offline probs on the valid frames vs expected, without the last 8 frames.
    low_latency_long: our streamed probs vs expected, (i) without the last 8 frames, (ii) without
    the final flushed chunk (the test's own exclusion)."""
    import tempfile

    try:
        from huggingface_hub import download_bucket_files
        from safetensors.numpy import load_file
        with tempfile.TemporaryDirectory(dir=WORK) as tmp:
            local = str(Path(tmp) / "expected_probabilities.safetensors")
            download_bucket_files(EXPECTED_BUCKET, files=[("expected_probabilities.safetensors", local)])
            exp = load_file(local)
    except Exception as e:  # noqa: BLE001
        log(f"[expected] not fetched: {type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
        return
    log(f"[expected] fetched {EXPECTED_BUCKET}/expected_probabilities.safetensors: "
        + ", ".join(f"{k} {tuple(v.shape)}" for k, v in sorted(exp.items())))
    off = np.load(WORK / "ref_diarization_example_offline.npz")
    ll = np.load(WORK / "chunk_io_diarization_example_ll.npz")
    if "offline_long" in exp:
        e = exp["offline_long"]
        p = probs_long["offline"][: int(off["attention_mask"].sum())]
        n = min(e.shape[0], p.shape[0]) - 8
        d = float(np.abs(e[:n] - p[:n]).max())
        log(f"[expected] offline_long: exp {e.shape} ours(valid) {p.shape}, max|Δp| without last 8 = {d:.2e} "
            f"-> {'within' if d <= 1e-3 else 'OUTSIDE'} atol 1e-3")
    if "low_latency_long" in exp:
        e, p = exp["low_latency_long"], probs_long["ll"]
        n = min(e.shape[0], p.shape[0]) - 8
        d8 = float(np.abs(e[:n] - p[:n]).max())
        flushed = int(ll["emit_len"][-1])
        m = p.shape[0] - flushed
        dfl = float(np.abs(e[:m] - p[:m]).max())
        log(f"[expected] low_latency_long: exp {e.shape} ours {p.shape}, max|Δp| without last 8 = {d8:.2e} "
            f"({'within' if d8 <= 1e-3 else 'OUTSIDE'} atol 1e-3); without the flushed last chunk "
            f"({flushed} frames, the test's exclusion) = {dfl:.2e} ({'within' if dfl <= 1e-3 else 'OUTSIDE'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default=",".join(FIXTURES))
    ap.add_argument("--skip-expected", action="store_true")
    ap.add_argument("--mode", default="offline,low_latency",
                    help="comma list of offline, low_latency, very_low_latency, ultra_low_latency")
    args = ap.parse_args()
    modes = tuple(args.mode.split(","))
    assert all(m == "offline" or m in MODE_TAGS for m in modes), modes
    WORK.mkdir(exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    from transformers import AutoProcessor, Nemotron3DiarizationForAudioFrameClassification
    from transformers.models.nemotron3_diarization import modeling_nemotron3_diarization as M

    snap = hf_snapshot(REPO_ID, revision=REVISION)
    st = Path(snap) / "model.safetensors"
    got = sha256_file(st)
    assert got == SAFETENSORS_SHA256, f"sha256 {got} != {SAFETENSORS_SHA256}"
    log(f"[weights] {st} sha256 OK")
    log(f"[env] transformers {metadata.version('transformers')} @ {transformers_commit()}, torch {torch.__version__}")

    processor = AutoProcessor.from_pretrained(snap)
    model = Nemotron3DiarizationForAudioFrameClassification.from_pretrained(snap, dtype=torch.float32).eval()
    log(f"[model] attn_implementation={model.config._attn_implementation}, dtype={model.dtype}, "
        f"params={sum(p.numel() for p in model.parameters()):,}")
    log(f"[processor] first chunk {processor.num_samples_first_audio_chunk} samples, later "
        f"{processor.num_samples_per_audio_chunk}, {processor.num_mel_frames_per_step} mel frames per step")

    fe = processor.feature_extractor
    np.savez(WORK / "fe_constants.npz", mel_filters=fe.mel_filters.numpy(),
             hann_window=torch.hann_window(fe.win_length, periodic=False).numpy(),
             librosa_version=np.array(metadata.version("librosa")))
    log(f"[fe] mel_filters {tuple(fe.mel_filters.shape)} {fe.mel_filters.dtype} (librosa "
        f"{metadata.version('librosa')}), preemphasis {fe.preemphasis}, n_fft {fe.n_fft}, "
        f"win {fe.win_length}, hop {fe.hop_length}")

    probs = {}
    for name in args.fixtures.split(","):
        probs[name] = do_fixture(name, model, processor, M.Nemotron3DiarizationSpeakerCache, log, modes)
    if not args.skip_expected and {"offline", "low_latency"} <= set(modes) and "diarization_example" in probs:
        compare_expected(probs["diarization_example"], log)


if __name__ == "__main__":
    main()
