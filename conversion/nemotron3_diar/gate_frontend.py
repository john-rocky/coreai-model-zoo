"""Gate the NumPy host front end (mel_frontend.py) against transformers. Run in the oracle venv:

    ~/code/coreai/_n3d/venv-oracle/bin/python gate_frontend.py --downstream

  A. log-mel: the processor's output for every low-latency chunk of the model card's generator
     (first chunk centered, later chunks not, last chunk flushed) and for the whole recording
     (offline), both fixtures, same frame counts.
     PASS: max|Δ| <= 5e-4 (log domain) and fewer than 0.1 % of the cells above 1e-4.
  B. embedder: mel_frontend.embed (8-frame stacking + embedder_projection.f32le) on the processor's
     per-chunk mel vs the transformers embedder output captured by make_reference.py
     (chunk_io_<fixture>_ll.npz, s###_embedder_out), and on the offline mel vs the offline capture.
     PASS: max|Δ| <= 1e-5 (the stacking + projection alone; measured bit-identical).
     NumPy mel -> NumPy embed end to end vs the same capture: PASS max|Δ| <= 5e-4 (|embed| <= 141).
  C. constants: mel_filters_128x257.f32le and hann_window_400.f32le bit-identical to the feature
     extractor's filterbank and window (torch.hann_window(400, periodic=False)).
  D. (--downstream) effect on the logits: for every low-latency step, the chunk rows of the
     captured packed input are replaced by NumPy mel -> NumPy embed, the eager fp32 graph
     (n3d_model.N3DGraph, T=541) runs, and its logits are compared with transformers' chunk
     logits (the cache/FIFO rows still come from the reference). PASS: max|Δlogit| <= 1e-4.
     The full verdict needs D; without --downstream the verdict is partial.
  Reported with A: (i) FFT attribution -- the same float32 windowed frames (the extractor's own
  preemphasis and torch.hann_window) through torch.stft, NumPy float32 rfft and NumPy float64 rfft;
  each float32 log-mel is compared with the float64 one, which says which side a mel difference
  comes from; the whole mirror (mel_frontend.log_mel) is compared with the same float64 reference,
  and with an all-float64 pipeline (preemphasis, window, FFT, mel, log in float64 from the float32
  audio and filterbank). (ii) chunk boundaries -- every streamed chunk's frames against the same
  global frames of the offline pass, for transformers and for NumPy separately.

Why these bars (revised by the supervisor on 2026-09-24 after round 1; the round-1 bar was a flat
mel max|Δ| <= 1e-4, and embed <= 1e-5 for the stacking alone):
  - the reference is not exact at 1e-4. transformers' own float32 torch.stft is 1.82e-4 (97.6 s) /
    8.26e-5 (21.5 s) away from a float64 FFT of the same float32 frames, and 2.41e-4 / 1.09e-4 away
    from an all-float64 pipeline; a mirror cannot be held closer to it than its own rounding.
  - the round-1 mirror measured 1.81e-4 (97.6 s) / 8.49e-5 (21.5 s) against transformers, 33 of
    1,249,280 cells above 1e-4 (0.0026 %), all near-silent bins (log-mel about -15.4, mel about
    2e-7): the bars keep 2.8x headroom on the max and 38x on the share of cells.
  - what the mel error does downstream is bounded directly: NumPy embed 1.80e-4 (|embed| <= 141),
    and swapped into every step the logits move 4.58e-5 (max|Δp| 9.3e-6), the size of the graph's
    own fp32 difference to transformers (4.96e-5).

`--filters fe` runs A with the feature extractor's own filterbank (before export_n3d.py has written
the .f32le files); the full gate uses the shipped .f32le files.
Numbers go to _work/gate_frontend.json (round 1's copy: _work/gate_frontend_r1.json).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True  # importing conversion/_paths must not leave a __pycache__ outside this dir
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0]))
import mel_frontend as mf  # noqa: E402
from _paths import hf_snapshot, work_path  # noqa: E402
from n3d_model import REPO_ID, REVISION  # noqa: E402

WORK = HERE / "_work"
FIXTURES = {"diarization_example": "diarization_example_16k.wav", "test_multispk": "test_multispk_16k.wav"}
MEL_MAX, MEL_CELL, MEL_CELL_SHARE = 5e-4, 1e-4, 1e-3     # max|Δ|; share of cells above MEL_CELL
EMB_BAR, EMB_E2E_BAR, LOGIT_BAR = 1e-5, 5e-4, 1e-4
MEL_BAR = MEL_CELL                                          # the round-1 bar, still reported


def mel_pass(max_abs: float, n_over: int, elements: int) -> bool:
    return max_abs <= MEL_MAX and n_over < MEL_CELL_SHARE * elements


def all_f64_log_mel(audio: np.ndarray, filters: np.ndarray, n_valid: int) -> np.ndarray:
    """Preemphasis, symmetric Hann(400), FFT, mel and log in float64 (centered, whole recording)."""
    x = audio.astype(np.float64)
    pre = np.concatenate([x[:1], x[1:] - 0.97 * x[:-1]])
    n = np.arange(mf.WIN, dtype=np.float64)
    w = np.zeros(mf.N_FFT)
    w[56:456] = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (mf.WIN - 1))
    y = np.pad(pre, (mf.N_FFT // 2, mf.N_FFT // 2))
    nf = 1 + (y.shape[0] - mf.N_FFT) // mf.HOP
    frames = y[np.arange(mf.N_FFT)[None, :] + mf.HOP * np.arange(nf)[:, None]] * w[None, :]
    return np.log((np.abs(np.fft.rfft(frames, axis=-1)) ** 2) @ filters.astype(np.float64).T + 2.0 ** -24)[:n_valid]


def fft_attribution(audio: np.ndarray, filters: np.ndarray, n_valid: int) -> dict:
    """Same float32 windowed frames through torch.stft (f32), NumPy rfft (f32) and NumPy rfft (f64);
    the whole mirror against the same float64 reference and against an all-float64 pipeline."""
    x = torch.tensor(audio)[None]
    pre = torch.cat([x[:, :1], x[:, 1:] - 0.97 * x[:, :-1]], dim=1)
    win = torch.hann_window(mf.WIN, periodic=False)
    st = torch.stft(pre, mf.N_FFT, hop_length=mf.HOP, win_length=mf.WIN, window=win, return_complex=True,
                    pad_mode="constant", center=True)[0].numpy().T                   # [F, 257] complex64
    w = np.zeros(mf.N_FFT, np.float32)
    w[56:456] = win.numpy()
    y = np.pad(pre[0].numpy(), (mf.N_FFT // 2, mf.N_FFT // 2))
    n = 1 + (y.shape[0] - mf.N_FFT) // mf.HOP
    frames = y[np.arange(mf.N_FFT)[None, :] + mf.HOP * np.arange(n)[:, None]] * w[None, :]

    def lm32(spec):
        re, im = spec.real.astype(np.float32), spec.imag.astype(np.float32)
        return np.log((filters @ (np.sqrt(re * re + im * im) ** 2).T).T + mf.LOG_GUARD)[:n_valid]

    ref64 = np.log((np.abs(np.fft.rfft(frames.astype(np.float64), axis=-1)) ** 2)
                   @ filters.astype(np.float64).T + 2.0 ** -24)[:n_valid]
    mirror = mf.log_mel(audio, center=True, filters=filters).astype(np.float64)
    all64 = all_f64_log_mel(audio, filters, n_valid)
    return {"torch_stft_f32_vs_f64": float(np.abs(lm32(st) - ref64).max()),
            "numpy_rfft_f32_vs_f64": float(np.abs(lm32(np.fft.rfft(frames, axis=-1)) - ref64).max()),
            "mirror_vs_f64_fft_of_extractor_frames": float(np.abs(mirror - ref64).max()),
            "mirror_vs_all_f64": float(np.abs(mirror - all64).max()),
            "torch_stft_f32_vs_all_f64": float(np.abs(lm32(st) - all64).max())}


def downstream(name, projection, filters):
    """Per-step effect of the NumPy front end on the logits (chunk rows swapped, eager graph)."""
    from gate_reauthor import pack
    from n3d_model import SUB, T_STREAM, load_checkpoint

    graph, _ = load_checkpoint(T=T_STREAM)
    io = np.load(WORK / f"chunk_io_{name}_ll.npz")
    audio, sr = sf.read(work_path("_n3d", "fixtures", FIXTURES[name]), dtype="float32")
    worst_l, worst_p, worst_e = 0.0, 0.0, 0.0
    for i, (start, end, first, _last) in enumerate(mf.stream_chunks(audio.shape[0])):
        emb = io[f"s{i:03d}_inputs_embeds"].copy()
        c0 = int(io["cache_before"][i] + io["fifo_before"][i])
        ours = mf.embed(mf.log_mel(audio[start:end], center=first, filters=filters), projection)
        assert ours.shape[0] == emb.shape[0] - c0, (i, ours.shape, emb.shape, c0)
        worst_e = max(worst_e, float(np.abs(ours - emb[c0:]).max()))
        emb[c0:] = ours
        packed, valid = pack(emb, T_STREAM, "zero", i)
        with torch.inference_mode():
            out = graph(packed, valid)[0, : emb.shape[0] * SUB].numpy()
        ref = io[f"s{i:03d}_chunk_logits"]
        worst_l = max(worst_l, float(np.abs(out - ref).max()))
        worst_p = max(worst_p, float(np.abs(1 / (1 + np.exp(-out.astype(np.float64)))
                                            - 1 / (1 + np.exp(-ref.astype(np.float64)))).max()))
    return {"max_abs_embed": worst_e, "max_abs_logit": worst_l, "max_abs_p": worst_p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filters", choices=["f32le", "fe"], default="f32le")
    ap.add_argument("--fixtures", default=",".join(FIXTURES))
    ap.add_argument("--skip-embed", action="store_true")
    ap.add_argument("--downstream", action="store_true")
    args = ap.parse_args()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(hf_snapshot(REPO_ID, revision=REVISION))
    fe = processor.feature_extractor
    fe_filters = fe.mel_filters.numpy()
    report = {"bars": {"mel_max_abs": MEL_MAX, "mel_cell": MEL_CELL, "mel_cell_share": MEL_CELL_SHARE,
                       "embed": EMB_BAR, "embed_e2e": EMB_E2E_BAR, "logit": LOGIT_BAR},
              "filters": args.filters}
    ok = True

    # ---- C. constants ----
    tw = torch.hann_window(fe.win_length, periodic=False).numpy()
    nw = mf.hann_window()
    same_w = bool(np.array_equal(tw.view(np.uint32), nw.view(np.uint32)))
    report["window"] = {"max_abs": float(np.abs(tw - nw).max()), "n_diff": int((tw != nw).sum()),
                        "bit_identical": same_w}
    ok &= same_w
    print(f"[C] hann_window_400.f32le vs torch.hann_window(400, periodic=False): "
          f"{'bit-identical' if same_w else 'DIFFERENT'} ({report['window']['n_diff']} of 400 entries differ)")
    if args.filters == "f32le":
        filters = mf.mel_filters()
        same = np.array_equal(filters.view(np.uint32), fe_filters.view(np.uint32))
        report["mel_filters_bit_identical"] = bool(same)
        ok &= same
        print(f"[C] mel_filters_128x257.f32le vs feature extractor: {'bit-identical' if same else 'DIFFERENT'}")
    else:
        filters = fe_filters.astype(np.float32)
        print("[C] using the feature extractor's filterbank (--filters fe)")

    projection = None
    if not args.skip_embed:
        projection = mf.load_f32le(mf.ARTIFACTS / "embedder_projection.f32le", (512, 1024))

    report["fixtures"] = {}
    for name in args.fixtures.split(","):
        audio, sr = sf.read(work_path("_n3d", "fixtures", FIXTURES[name]), dtype="float32")
        assert sr == mf.SR
        fr = {}

        # ---- A. offline ----
        ref = processor(audio, sampling_rate=sr)
        n_valid = int(ref["attention_mask"].sum())
        ref_mel = ref["input_features"][0, :n_valid].numpy()
        ours = mf.log_mel(audio, center=True, filters=filters)
        assert ours.shape == ref_mel.shape, (ours.shape, ref_mel.shape)
        d_off = np.abs(ours - ref_mel)
        attr = fft_attribution(audio, filters, n_valid)
        worst = np.unravel_index(d_off.argmax(), d_off.shape)
        fr["offline_mel"] = {"frames": n_valid, "max_abs": float(d_off.max()),
                             "n_over_bar": int((d_off > MEL_CELL).sum()), "elements": int(d_off.size),
                             "frames_over_bar": int((d_off.max(1) > MEL_CELL).sum()),
                             "worst_frame": int(worst[0]), "worst_mel": int(worst[1]),
                             "worst_ref_value": float(ref_mel[worst]),
                             **attr}
        o = fr["offline_mel"]
        o["share_over_1e-4"] = o["n_over_bar"] / o["elements"]
        o["below_1e-4"] = o["max_abs"] <= MEL_CELL
        o["pass"] = mel_pass(o["max_abs"], o["n_over_bar"], o["elements"])
        ok &= o["pass"]

        # ---- A. streaming chunks ----
        worst, worst_at, n_chunks, n_frames = 0.0, None, 0, 0
        n_over, n_cells = 0, 0
        chunk_mels = []
        bound_ref, bound_ours = 0.0, 0.0
        full_ref = ref["input_features"][0].numpy()
        processor.set_streaming_mode("low_latency")
        for start, end, first, last in mf.stream_chunks(audio.shape[0]):
            kw = {"is_first_audio_chunk": first}
            if last:
                kw["is_last_audio_chunk"] = True
            r = processor(audio[start:end], sampling_rate=sr, is_streaming=True, **kw)
            rm = r["input_features"][0].numpy()
            om = mf.log_mel(audio[start:end], center=first, filters=filters)
            assert om.shape == rm.shape, (n_chunks, om.shape, rm.shape)
            dd = np.abs(om - rm)
            n_over += int((dd > MEL_CELL).sum())
            n_cells += int(dd.size)
            d = float(dd.max())
            if d > worst:
                worst, worst_at = d, n_chunks
            chunk_mels.append((rm, om))
            g0 = n_chunks * mf.MEL_PER_STEP                        # global index of the chunk's frame 0
            m = min(rm.shape[0], n_valid - g0)
            bound_ref = max(bound_ref, float(np.abs(rm[:m] - full_ref[g0:g0 + m]).max()))
            bound_ours = max(bound_ours, float(np.abs(om[:m] - ours[g0:g0 + m]).max()))
            n_chunks += 1
            n_frames += rm.shape[0]
        fr["stream_mel"] = {"chunks": n_chunks, "frames": n_frames, "max_abs": worst, "worst_chunk": worst_at,
                            "n_over_bar": n_over, "elements": n_cells, "share_over_1e-4": n_over / n_cells,
                            "below_1e-4": worst <= MEL_CELL, "pass": mel_pass(worst, n_over, n_cells),
                            "chunk_vs_offline_transformers": bound_ref, "chunk_vs_offline_numpy": bound_ours}
        ok &= fr["stream_mel"]["pass"]
        s = fr["stream_mel"]
        print(f"[A] {name}: offline {n_valid} frames max|Δ| {o['max_abs']:.2e} ({o['n_over_bar']} of "
              f"{o['elements']} cells over {MEL_CELL:g} = {o['share_over_1e-4'] * 100:.4f} %, in "
              f"{o['frames_over_bar']} frames; worst at frame {o['worst_frame']} mel {o['worst_mel']}, value "
              f"{o['worst_ref_value']:.2f}) -> {'PASS' if o['pass'] else 'FAIL'}, "
              f"{'below' if o['below_1e-4'] else 'above'} 1e-4")
        print(f"[A] {name}: streaming {n_chunks} chunks / {n_frames} frames max|Δ| {worst:.2e} (chunk {worst_at}), "
              f"{n_over} of {n_cells} cells over {MEL_CELL:g} = {s['share_over_1e-4'] * 100:.4f} % -> "
              f"{'PASS' if s['pass'] else 'FAIL'}, {'below' if s['below_1e-4'] else 'above'} 1e-4 "
              f"(bar: max <= {MEL_MAX:g}, share over {MEL_CELL:g} < {MEL_CELL_SHARE * 100:g} %)")
        print(f"[A] {name}: FFT attribution, same f32 frames, log-mel vs a float64 FFT: torch.stft f32 "
              f"{o['torch_stft_f32_vs_f64']:.2e}, NumPy rfft f32 {o['numpy_rfft_f32_vs_f64']:.2e}; whole mirror "
              f"{o['mirror_vs_f64_fft_of_extractor_frames']:.2e}. vs an all-float64 pipeline: mirror "
              f"{o['mirror_vs_all_f64']:.2e}, torch.stft f32 frames {o['torch_stft_f32_vs_all_f64']:.2e}")
        print(f"[A] {name}: chunk boundaries, streamed chunk frames vs the same offline frames: transformers "
              f"{bound_ref:.2e}, NumPy {bound_ours:.2e}")

        # ---- B. embedder ----
        if projection is not None:
            io = np.load(WORK / f"chunk_io_{name}_ll.npz")
            assert int(io["n_steps"]) == n_chunks, (int(io["n_steps"]), n_chunks)
            we, we_e2e = 0.0, 0.0
            for i, (rm, om) in enumerate(chunk_mels):
                cap = io[f"s{i:03d}_embedder_out"]
                assert np.array_equal(io[f"s{i:03d}_input_features"], rm), f"chunk {i}: processor output drifted"
                we = max(we, float(np.abs(mf.embed(rm, projection) - cap).max()))
                we_e2e = max(we_e2e, float(np.abs(mf.embed(om, projection) - cap).max()))
            off = np.load(WORK / f"chunk_io_{name}_offline.npz")
            full_mel = ref["input_features"][0].numpy()          # every frame, as the model receives it
            w_off = float(np.abs(mf.embed(full_mel, projection) - off["embedder_out"]).max())
            fr["embed"] = {"stream_max_abs": we, "offline_max_abs": w_off, "stream_e2e_max_abs": we_e2e,
                           "embed_abs_max": float(max(np.abs(io[f"s{i:03d}_embedder_out"]).max()
                                                      for i in range(n_chunks)))}
            ok &= we <= EMB_BAR and w_off <= EMB_BAR and we_e2e <= EMB_E2E_BAR
            print(f"[B] {name}: embed(processor mel) vs transformers embedder: streaming max|Δ| {we:.2e}, "
                  f"offline {w_off:.2e} (bar {EMB_BAR:g}); NumPy mel -> embed end to end {we_e2e:.2e} "
                  f"(bar {EMB_E2E_BAR:g}; |embeds| up to {fr['embed']['embed_abs_max']:.1f})")
        if args.downstream and projection is not None:
            fr["downstream"] = downstream(name, projection, filters)
            ds = fr["downstream"]
            ok &= ds["max_abs_logit"] <= LOGIT_BAR
            print(f"[D] {name}: NumPy front end swapped into every step's chunk rows: max|Δembed| "
                  f"{ds['max_abs_embed']:.2e}, max|Δlogit| {ds['max_abs_logit']:.2e} (bar {LOGIT_BAR:g}), "
                  f"max|Δp| {ds['max_abs_p']:.2e}")
        report["fixtures"][name] = fr

    full = projection is not None and args.filters == "f32le" and args.downstream
    report["pass"] = bool(ok) and full
    (WORK / "gate_frontend.json").write_text(json.dumps(report, indent=1))
    verdict = "PASS" if report["pass"] else ("PASS (partial: run with --downstream and the .f32le files)"
                                             if ok else "FAIL")
    print(f"gate_frontend: {verdict}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
