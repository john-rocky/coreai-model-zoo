#!/usr/bin/env python3
"""d1-omni-600M's audio front end (mel_host.py) and audio graph (d1_omni_audio.py) against the publisher's audio.py, and
the audio rows end to end through the decision graph.

    python3 conversion/d1_omni/audio_check.py --stage mel                                   # venv-d1, CPU
    python3 conversion/d1_omni/audio_check.py --stage eager                                 # venv-d1, CPU fp32
    python3 conversion/d1_omni/audio_check.py --stage runtime --compute gpu \\
        --audio 5=<work>/compiled/audio-wfp16-5s-h16c --audio 10=... --audio 20=... --audio 30=... \\
        --decide 256=<work>/compiled/wfp16-L256-h16c --decide 512=<work>/compiled/wfp16-L512-h16c-r6  # shared venv

Clips (22): the 16 audio records of the references with their npz (sha256 checked) -- card_audio and aud_01..03
(ref/records_ref.json, round 2: prefix, mel, frames) and aud_04..aud_15 (ref/records_ref_audio.json, round 6: also the
conformer and subsampling outputs) -- and 6 measured-only edge clips cut from them (SYNTHETIC: exactly 0.5 s, the
largest clip of the 5 s bucket and one sample more, a length that is not a multiple of 160, exactly 30 s, 2 s of
digital silence), whose reference is the publisher's Audio run here.

Stage mel (the publisher's waveform() + MelFrontend vs mel_host):
  (a) mel_torch == the publisher's MelFrontend bit for bit, frames equal (and the publisher's mel == the oracle's
      hooked mel bit for bit on the 16 fixture clips); a float-sample input == the int16 input
  (b) mel_numpy (float64) vs (a): max |d| (the launch's 1e-4 recorded per clip); beside it the same function check
      (mel_numpy float64 vs mel_torch run in float64: <= 1e-6) and each clip's fp32 floor (the publisher's fp32 mel vs
      its float64 run); mel_numpy in float32 recorded too
  the filterbank: mel_host.slaney_filterbank == the publisher's bit for bit; mel_filters_128x257_f32.bin bytes + sha256
Stage eager (CPU, the fp32 module of the clip's bucket, venv-d1 = the reference's environment):
  rows  host.audio_request_rows == the references' ids / markers / positions (P = the prefix rows) on the 16 clips
  (i)   the subsampling output (valid rows) vs the publisher's pre_encode: max |d|, cos
  (ii)  the conformer output (valid rows) vs the publisher's encoder: max |d|, cos
  (iii) prefix vs the publisher's Audio (== the oracle npz bit for bit on the fixture clips): max |d|, cos, min row cos.
        Bar: cos >= 0.999999; float64: D1Audio vs the publisher's Audio, both in float64 on the same fp32 mel, max |d|
        <= 1e-6 (the same function); each one's fp32 floor recorded
  (iv)  pad invariance: the clip in every larger bucket vs its own bucket, valid rows: max |d| <= 1e-5
  (v)   aud_14 (0.37 s, padded to 0.5 s) and aud_15 (36 s, cut to 30 s) are among the clips of (iii)
  red arms: the four masks set to 1 everywhere, and every BatchNorm affine removed, must move the prefix out of the bar
  recorded: the fp32 absmax of the subsampling convs, the conformer's residual stream and the output (fp16 headroom)
Stage runtime (Core AI Python runtime, the AOT .aimodelc of each bucket, explicit compute unit, never default()):
  (i)   each clip's prefix from the bundle of its bucket (host.audio_prefix, torch mel) vs the reference prefix: max |d|,
        cos, min row cos; drift = max |d| between --repeats calls; the NumPy mel's prefix beside it
  (ii)  end to end: every audio row (card_audio + aud_01..03: 10, aud_04..aud_15: 36) with this prefix (torch mel and
        NumPy mel) -> host.graph_inputs at the row's bucket -> the wfp16 decision bundle of that bucket -> marker logits
        -> probabilities vs the reference (_metrics.SHIP_BAR, FACTS §7); beside it the reference's own prefix through the
        same bundles, and a control with the next clip's prefix (its rows cut or repeated to the row's P): must FAIL
Cosines are float64.

Output: results/audio_mel.json, results/audio_eager.json, results/audio_<precision>_<sec>s_<compute>.json (one per
bucket) and results/audio_e2e_<precision>.json (an existing file is never replaced: a .runN name is used;
--results-prefix is prepended to the runtime names).
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import hf_snapshot, work_path  # noqa: E402

import host  # noqa: E402
import mel_host  # noqa: E402
from vision_check import compare, environment, sha256_file, write_new  # noqa: E402

WORK = work_path("_d1_omni")
REFERENCE_SHA256 = "e7dba6f44d0452a6aea3e401746d417503056d6f468ff72b56c5511883436c5f"         # round 2
REFERENCE_AUDIO_SHA256 = "c483f8c511cfcdb05a8211e495a7476b11f9318899a1125c387c9e722a6d8fbc"   # round 6
FIXTURES_AUDIO_SHA256 = "18b4ddff73a8044b63f949c69245210e1af8a26c48c5fdb785fd91a7807ead5b"
PREFIX_BAR = {"cos": 0.999999}
FLOAT64_BAR = 1e-6          # D1Audio float64 vs the publisher's Audio float64, max |d| of the prefix
PAD_BAR = 1e-5              # the same clip in a larger bucket, valid prefix rows
MEL_NUMPY_BAR = 1e-4        # the launch's bar for mel_numpy vs mel_torch (recorded per clip)
MEL_SAME_FUNCTION_BAR = 1e-6
FP16_MAX = 65504.0
# measured-only edge clips: (id, what, [(source clip, start, stop) ...] or "silence:<samples>")
SYNTHETIC = [
    ("syn_0p5s", "exactly 0.5 s (8,000 samples: the pad threshold, not padded)", [("aud_02", 0, 8000)]),
    ("syn_5s_edge", "80,159 samples: the largest clip of the 5 s bucket (1 + n // 160 = 501 = F)", [("aud_08", 0, 80159)]),
    ("syn_5s_over", "80,160 samples: one sample more, the smallest clip of the 10 s bucket", [("aud_08", 0, 80160)]),
    ("syn_odd", "123,457 samples (7.716 s): not a multiple of 160", [("aud_10", 0, 123457)]),
    ("syn_30s", "exactly 30 s (480,000 samples: the cut threshold, not cut; the 30 s bucket full)",
     [("aud_12", 0, 447200), ("aud_06", 0, 32800)]),
    ("syn_silence", "2 s of digital silence (every mel row's std is 0: the normalisation divides 0 by 1e-5)",
     "silence:32000"),
]


# --------------------------------------------------------------------------- the clips and their references
def load_clips() -> tuple[list[dict], dict]:
    """[{id, kind, samples, public, record, entry (the reference's audio-mode request), npz, ...}] in reference order
    (card_audio, aud_01..aud_15), then the SYNTHETIC clips; the reference files checked against their pinned sha256."""
    import soundfile as sf

    paths = {"main": WORK / "ref" / "records_ref.json", "audio": WORK / "ref" / "records_ref_audio.json",
             "fixtures": WORK / "fixtures" / "records.json"}
    pinned = {"main": REFERENCE_SHA256, "audio": REFERENCE_AUDIO_SHA256, "fixtures": FIXTURES_AUDIO_SHA256}
    for key, path in paths.items():
        if sha256_file(path) != pinned[key]:
            raise SystemExit(f"{path} is not the pinned file ({pinned[key][:8]}…)")
    refs = {k: json.loads(paths[k].read_text()) for k in ("main", "audio")}
    fixtures = {r["id"]: r for r in json.loads(paths["fixtures"].read_text())["records"]}
    out = []
    for key in ("main", "audio"):
        ref = refs[key]
        for entry in ref["records"]:
            if entry["mode"] != "audio":
                continue
            meta = ref["npz"][entry["id"]]
            npz_path = WORK / "ref" / "npz" / f"{entry['id']}.npz"
            if sha256_file(npz_path) != meta["sha256"]:
                raise SystemExit(f"{npz_path} differs from the reference's npz")
            with np.load(npz_path) as z:
                arrays = {k: z[k] for k in z.files}
            record = fixtures[entry["id"]]
            path = WORK / record["media"]["audio"]
            expected = (record["provenance"].get("clip") or record["provenance"].get("audio") or {}).get("sha256")
            if sha256_file(path) != expected:
                raise SystemExit(f"{path}: sha256 differs from the fixture's")
            samples, rate = sf.read(str(path), dtype="int16")  # as the card and the oracle read it
            assert rate == host.SAMPLE_RATE and samples.ndim == 1, (path, rate, samples.shape)
            out.append({"id": entry["id"], "kind": "fixture", "samples": samples, "public": record["public"],
                        "file": str(path.relative_to(WORK)), "record": record, "entry": entry, "npz": arrays,
                        "npz_file": f"ref/npz/{entry['id']}.npz", "npz_sha256": meta["sha256"], "reference": key})
    by_id = {c["id"]: c for c in out}
    for cid, what, parts in SYNTHETIC:
        if isinstance(parts, str):
            samples = np.zeros(int(parts.split(":")[1]), dtype=np.int16)
            made = parts
        else:
            samples = np.concatenate([by_id[src]["samples"][a:b] for src, a, b in parts])
            made = [f"{src}[{a}:{b}]" for src, a, b in parts]
        out.append({"id": cid, "kind": "synthetic", "samples": samples, "public": False, "what": what, "made": made,
                    "record": None, "entry": None, "npz": None})
    info = {k: {"path": str(paths[k]), "sha256": pinned[k]} for k in paths}
    return out, info


def snapshot_dir() -> Path:
    return Path(hf_snapshot(host.MODEL_ID, revision=host.MODEL_SHA))


def load_publisher_audio(snapshot: Path):
    """The snapshot's audio.py as a module (plain torch + numpy)."""
    spec = importlib.util.spec_from_file_location("d1_publisher_audio", snapshot / "audio.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def publisher_audio_model(snapshot: Path, publisher):
    """The publisher's Audio (frontend + conformer + adapter + residual) with the checkpoint's audio tensors, fp32,
    eval, with hooks keeping the subsampling's and the conformer's outputs (`taps`)."""
    import torch
    from safetensors import safe_open

    config = json.loads((snapshot / "config.json").read_text())
    model = publisher.Audio(config["audio_config"], config["text_config"]["hidden_size"])
    state = {}
    with safe_open(str(snapshot / "model.safetensors"), framework="pt") as f:
        for key in f.keys():
            if key.startswith("audio."):
                state[key[len("audio."):]] = f.get_tensor(key)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    model.taps = {}
    model.encoder.pre_encode.register_forward_hook(lambda m, a, o: model.taps.__setitem__("pre_encode", o))
    model.encoder.register_forward_hook(lambda m, a, o: model.taps.__setitem__("encoder", o))
    return model, config


def as_rows(x, rows: int) -> np.ndarray:
    """A [.., rows', width] tensor or array -> its first `rows` rows as float64 [rows, width]."""
    import torch

    if isinstance(x, torch.Tensor):
        x = x.detach().to(torch.float64).numpy()
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(-1, x.shape[-1])[:rows]


# --------------------------------------------------------------------------- stage mel
def mel_stage(args) -> int:
    import torch

    t_start = time.perf_counter()
    snapshot = snapshot_dir()
    if sha256_file(snapshot / "audio.py") != host.COPIED_FROM["audio.py"]:
        raise SystemExit("audio.py is not the pinned revision")
    publisher = load_publisher_audio(snapshot)
    clips, ref_info = load_clips()
    frontend = publisher.MelFrontend(mel_host.FEATURES)
    fb_mine, fb_theirs = mel_host.slaney_filterbank(), publisher.slaney_filterbank()
    fb_bytes = mel_host.filterbank_bytes()
    import hashlib
    filterbank = {"bit_equal_publisher": bool(np.array_equal(fb_mine, fb_theirs)), "shape": list(fb_mine.shape),
                  "file": mel_host.FILTERBANK_FILE, "bytes": len(fb_bytes),
                  "sha256": hashlib.sha256(fb_bytes).hexdigest(),
                  "layout": "float32 little-endian, row-major [128 mel rows, 257 FFT bins]",
                  "nonzero_bins_per_row": [int(np.count_nonzero(r)) for r in fb_mine[:4]] + ["…"]}
    window_torch = torch.hann_window(mel_host.WINDOW, periodic=False).numpy()
    window_np = mel_host.hann_window_numpy(np.float64)[56:456]
    window = {"numpy_float64_vs_torch_float32_max_abs": float(np.max(np.abs(window_np - window_torch))),
              "numpy_float32_bit_equal_torch": bool(np.array_equal(window_np.astype(np.float32), window_torch))}
    records = []
    for clip in clips:
        t0 = time.perf_counter()
        a = clip["samples"]
        with torch.no_grad():
            mel_p, frames_p = frontend(publisher.waveform(a))
        mel_p, frames_p = mel_p[0].numpy(), int(frames_p[0])
        mel_t, frames_t = mel_host.mel_torch(a)
        mel_t64, _ = mel_host.mel_torch(a, float64=True)
        mel_n, frames_n = mel_host.mel_numpy(a)
        mel_n32, _ = mel_host.mel_numpy(a, float64=False)
        counts = host.audio_frames(len(a))
        d_numpy = np.abs(mel_n.astype(np.float64) - mel_t)
        rec = {"id": clip["id"], "kind": clip["kind"], "samples": int(len(a)), "seconds": len(a) / host.SAMPLE_RATE,
               "after_waveform": counts["samples"], "columns": int(mel_p.shape[1]), "frames": frames_p,
               "bucket_s": host.audio_bucket_for(len(a)),
               "torch_bit_equal_publisher": bool(np.array_equal(mel_t, mel_p)),
               "frames_equal": frames_t == frames_p == frames_n == counts["valid_frames"]
                               and mel_p.shape[1] == counts["stft_frames"],
               "numpy_max_abs": float(d_numpy.max()), "numpy_mean_abs": float(d_numpy.mean()),
               "numpy_elements_over_1e-4": int((d_numpy > MEL_NUMPY_BAR).sum()), "elements": int(d_numpy.size),
               "numpy_worst": [int(v) for v in np.unravel_index(d_numpy.argmax(), d_numpy.shape)],
               "numpy_float32_max_abs": float(np.max(np.abs(mel_n32.astype(np.float64) - mel_t))),
               "same_function_numpy64_vs_torch64_max_abs": float(np.max(np.abs(mel_n.astype(np.float64)
                                                                               - mel_t64.astype(np.float64)))),
               "fp32_floor_publisher_vs_float64_max_abs": float(np.max(np.abs(mel_p.astype(np.float64)
                                                                              - mel_t64.astype(np.float64)))),
               "pad_columns_zero": bool((mel_t[:, frames_t:] == 0).all())}
        if clip["kind"] == "fixture":
            rec["publisher_bit_equal_oracle_npz"] = bool(np.array_equal(mel_p, clip["npz"]["mel"]))
            rec["frames_equal_oracle_npz"] = int(clip["npz"]["frames"][0]) == frames_p
        if a.dtype == np.int16:
            mel_f, _ = mel_host.mel_torch(a.astype(np.float32) / np.float32(32768.0))
            rec["float_input_bit_equal_int16"] = bool(np.array_equal(mel_f, mel_t))
        rec["seconds_run"] = time.perf_counter() - t0
        records.append(rec)
        print(f"{clip['id']}: n {len(a)} F {mel_p.shape[1]} frames {frames_p} bucket {rec['bucket_s']} s; torch "
              f"{'bit' if rec['torch_bit_equal_publisher'] else 'DIFF'}; numpy {rec['numpy_max_abs']:.2e} "
              f"({rec['numpy_elements_over_1e-4']} > 1e-4) float32 {rec['numpy_float32_max_abs']:.2e}; same function "
              f"{rec['same_function_numpy64_vs_torch64_max_abs']:.1e}; fp32 floor "
              f"{rec['fp32_floor_publisher_vs_float64_max_abs']:.2e}", flush=True)
    fixture = [r for r in records if r["kind"] == "fixture"]
    over = [r for r in records if r["numpy_max_abs"] > MEL_NUMPY_BAR]
    summary = {
        "clips": len(records), "fixture_clips": len(fixture), "synthetic_clips": len(records) - len(fixture),
        "torch_bit_equal_publisher": f"{sum(r['torch_bit_equal_publisher'] for r in records)}/{len(records)}",
        "frames_equal": f"{sum(r['frames_equal'] for r in records)}/{len(records)}",
        "publisher_bit_equal_oracle_npz": f"{sum(r['publisher_bit_equal_oracle_npz'] for r in fixture)}/{len(fixture)}",
        "float_input_bit_equal_int16": f"{sum(r.get('float_input_bit_equal_int16', False) for r in records)}/"
                                       f"{sum('float_input_bit_equal_int16' in r for r in records)}",
        "numpy_max_abs": max(r["numpy_max_abs"] for r in records),
        "numpy_worst_clip": max(records, key=lambda r: r["numpy_max_abs"])["id"],
        "numpy_clips_le_1e-4": f"{len(records) - len(over)}/{len(records)}",
        "numpy_elements_over_1e-4": sum(r["numpy_elements_over_1e-4"] for r in records),
        "elements": sum(r["elements"] for r in records),
        "numpy_float32_max_abs": max(r["numpy_float32_max_abs"] for r in records),
        "same_function_max_abs": max(r["same_function_numpy64_vs_torch64_max_abs"] for r in records),
        "fp32_floor_max_abs": max(r["fp32_floor_publisher_vs_float64_max_abs"] for r in records),
        "over_bar_within_own_fp32_floor": all(r["numpy_max_abs"] <= r["fp32_floor_publisher_vs_float64_max_abs"] * 1.001
                                              + 1e-7 for r in over),
    }
    verdict = {
        "torch_bit_equal": summary["torch_bit_equal_publisher"].split("/")[0] == str(len(records)),
        "frames": summary["frames_equal"].split("/")[0] == str(len(records)),
        "oracle_npz": all(r["publisher_bit_equal_oracle_npz"] and r["frames_equal_oracle_npz"] for r in fixture),
        "filterbank": filterbank["bit_equal_publisher"],
        "numpy_same_function": summary["same_function_max_abs"] <= MEL_SAME_FUNCTION_BAR,
        "float_input": all(r.get("float_input_bit_equal_int16", True) for r in records),
    }
    status = "PASS" if all(verdict.values()) else "FAIL"
    numpy_bar = ("PASS" if not over else
                 "RECORDED: over 1e-4 only where the publisher's own fp32 mel is as far from its float64 run"
                 if summary["over_bar_within_own_fp32_floor"] else "FAIL")
    doc = {"status": status, "verdict": verdict, "numpy_bar_1e-4": numpy_bar,
           "stage": "mel (CPU: the publisher's waveform() + MelFrontend vs mel_host.mel_torch / mel_numpy)",
           "bars": {"torch": "bit for bit", "numpy_launch_max_abs": MEL_NUMPY_BAR,
                    "numpy_same_function_max_abs": MEL_SAME_FUNCTION_BAR,
                    "why_same_function": "mel_numpy in float64 against mel_torch run in float64: the steps are the "
                                         "publisher's; the rest of (b) is the fp32 rounding of the publisher's own mel"},
           "reference": ref_info, "filterbank": filterbank, "window": window, "summary": summary,
           "seconds": time.perf_counter() - t_start,
           "code_sha256": {f: sha256_file(HERE / f) for f in ("audio_check.py", "mel_host.py", "host.py")},
           "environment": environment(("torch", "numpy", "soundfile")),
           "synthetic": [{k: c[k] for k in ("id", "what", "made")} for c in clips if c["kind"] == "synthetic"],
           "clips": records}
    out = write_new(WORK / "results" / "audio_mel.json", doc)
    bin_path = WORK / "results" / mel_host.FILTERBANK_FILE
    if not bin_path.exists():
        bin_path.write_bytes(fb_bytes)
    assert sha256_file(bin_path) == filterbank["sha256"], bin_path
    print(json.dumps({"status": status, "verdict": verdict, "numpy_bar_1e-4": numpy_bar, **summary,
                      "filterbank_sha256": filterbank["sha256"]}, indent=1))
    print("->", out, bin_path)
    return 0 if status == "PASS" else 1


# --------------------------------------------------------------------------- stage eager
def eager_stage(args) -> int:
    import torch

    import d1_omni_audio as da

    t_start = time.perf_counter()
    torch.set_grad_enabled(False)
    snapshot = snapshot_dir()
    if sha256_file(snapshot / "audio.py") != host.COPIED_FROM["audio.py"]:
        raise SystemExit("audio.py is not the pinned revision")
    clips, ref_info = load_clips()
    publisher = load_publisher_audio(snapshot)
    pub32, _ = publisher_audio_model(snapshot, publisher)
    pub64, _ = publisher_audio_model(snapshot, publisher)
    pub64 = pub64.double()
    base, load_record = da.load_d1_audio(snapshot, 30, "fp32", verify_sha256=True)
    base64, _ = da.load_d1_audio(snapshot, 30, "fp32")
    base64.as_float64()
    mine = {sec: (base if sec == 30 else base.rebucket(sec)) for sec in da.CLIP_SECONDS}
    mine64 = {sec: (base64 if sec == 30 else base64.rebucket(sec)) for sec in da.CLIP_SECONDS}
    no_bn = {}
    for sec, m in mine.items():
        r = m.rebucket(sec)
        for layer in r.layers:
            layer.conv.bn_scale = torch.ones_like(layer.conv.bn_scale)
            layer.conv.bn_shift = torch.zeros_like(layer.conv.bn_shift)
        no_bn[sec] = r
    tok = host.RawTokenizer(snapshot / "tokenizer.json")
    host.check_token_ids(tok)

    def run(model, inputs: dict, peaks: dict | None = None):
        t = [torch.from_numpy(np.ascontiguousarray(inputs[k])) for k in host.AUDIO_INPUT_NAMES]
        sub = model.subsampling(*t)
        enc = model.encoder(sub, t[4])
        prefix = model.project(enc)
        if peaks is not None:  # the residual stream and the subsampling's convs, over the valid rows (fp32 headroom)
            p = inputs["prefix_rows"]
            peaks.update(fp32_peaks(model, t, p))
        return sub[0], enc[0], prefix[0]

    def run_publisher(model, samples):
        model.taps.clear()
        out = model(samples)
        return model.taps["pre_encode"], model.taps["encoder"], out

    records = []
    for clip in clips:
        t0 = time.perf_counter()
        a = clip["samples"]
        inputs = host.audio_inputs(a)
        sec, p = inputs["sec"], inputs["prefix_rows"]
        (pre_p, pre_len), (enc_p, enc_len), prefix_p = run_publisher(pub32, a)
        assert int(pre_len[0]) == p == int(enc_len[0]) == prefix_p.shape[1], (clip["id"], p, pre_len, enc_len)
        steps_publisher = int(enc_p.shape[1])
        peaks: dict = {}
        sub, enc, prefix = run(mine[sec], inputs, peaks)
        rec = {"id": clip["id"], "kind": clip["kind"], "samples": int(len(a)), "seconds": len(a) / host.SAMPLE_RATE,
               "bucket_s": sec, "frames": inputs["frames"], "prefix_rows": p, "steps_bucket": inputs["steps"],
               "steps_publisher": steps_publisher,
               "host_lengths": {"audio_prefix_length": host.audio_prefix_length(len(a)),
                                "audio_encoder_steps": host.audio_encoder_steps(len(a))},
               "host_lengths_equal": host.audio_prefix_length(len(a)) == p
                                     and host.audio_encoder_steps(len(a)) == steps_publisher}
        if clip["kind"] == "fixture":
            z = clip["npz"]
            rec["publisher_prefix_bit_equal_oracle"] = bool(np.array_equal(prefix_p[0].numpy(), z["prefix"]))
            if "encoder_out" in z:
                rec["publisher_encoder_bit_equal_oracle"] = bool(np.array_equal(enc_p[0].numpy(), z["encoder_out"]))
                rec["publisher_pre_encode_bit_equal_oracle"] = bool(np.array_equal(pre_p[0].numpy(), z["pre_encode_out"]))
            entry, record = clip["entry"], clip["record"]
            rows = host.audio_request_rows(tok, record["request"]["state"], record["request"]["questions"], a)
            same = [r.qid == q["qid"] and r.ids == q["ids"] and r.markers == q["markers"] and r.positions == q["positions"]
                    and r.prefix_len == entry["prefix"] == p for r, q in zip(rows, entry["questions"], strict=True)]
            rec["rows_equal_reference"] = f"{sum(same)}/{len(same)}"
        rec["subsampling"] = compare(as_rows(sub, p), as_rows(pre_p[0], p))
        rec["encoder"] = compare(as_rows(enc, p), as_rows(enc_p[0], p))
        rec["prefix"] = compare(as_rows(prefix, p), as_rows(prefix_p[0], p))
        rec["prefix_pass"] = rec["prefix"]["cos"] >= PREFIX_BAR["cos"]
        # float64: the same function
        prefix64 = run(mine64[sec], inputs)[2]
        with torch.no_grad():
            prefix_p64 = run_publisher(pub64, a)[2]
        rec["float64"] = {"mine64_vs_publisher64": compare(as_rows(prefix64, p), as_rows(prefix_p64[0], p)),
                          "floor_publisher32_vs_publisher64": compare(as_rows(prefix_p[0], p), as_rows(prefix_p64[0], p)),
                          "floor_mine32_vs_mine64": compare(as_rows(prefix, p), as_rows(prefix64, p))}
        rec["float64_pass"] = rec["float64"]["mine64_vs_publisher64"]["max_abs"] <= FLOAT64_BAR
        # (iv) pad invariance: every larger bucket
        pads = []
        for other in [s for s in da.CLIP_SECONDS if s > sec]:
            bigger = host.audio_inputs(a, other)
            out = run(mine[other], bigger)[2]
            d = np.abs(as_rows(out, p) - as_rows(prefix, p))
            pads.append({"bucket_s": other, "max_abs": float(d.max()), "bit_equal": bool((d == 0).all())})
        rec["pad_invariance"] = pads
        rec["pad_pass"] = all(x["max_abs"] <= PAD_BAR for x in pads)
        # red arms
        ones = dict(inputs)
        for k in ("mask_f", "mask_f2", "mask_f4", "mask_t"):
            ones[k] = np.ones_like(inputs[k])
        red_masks = run(mine[sec], ones)[2]
        rec["red_masks_all_ones"] = compare(as_rows(red_masks, p), as_rows(prefix_p[0], p))
        rec["red_masks_caught"] = rec["red_masks_all_ones"]["cos"] < PREFIX_BAR["cos"]
        rec["red_masks_note"] = ("the clip fills its bucket except the last column(s): the masks differ from all ones "
                                 f"on {inputs['mask_t'].size - p} of {inputs['mask_t'].size} steps")
        red_bn = run(no_bn[sec], inputs)[2]
        rec["red_no_batch_norm"] = compare(as_rows(red_bn, p), as_rows(prefix_p[0], p))
        rec["red_bn_caught"] = rec["red_no_batch_norm"]["cos"] < PREFIX_BAR["cos"]
        rec["fp32_absmax"] = peaks
        rec["seconds_run"] = time.perf_counter() - t0
        records.append(rec)
        print(f"{clip['id']}: {rec['seconds']:.2f} s bucket {sec} P {p}/{inputs['steps']}; sub cos "
              f"{rec['subsampling']['cos']:.9f} enc cos {rec['encoder']['cos']:.9f} prefix cos {rec['prefix']['cos']:.10f} "
              f"max {rec['prefix']['max_abs']:.2e} min-row {rec['prefix']['min_row_cos']:.9f}; float64 "
              f"{rec['float64']['mine64_vs_publisher64']['max_abs']:.1e} (floors "
              f"{rec['float64']['floor_publisher32_vs_publisher64']['max_abs']:.1e} / "
              f"{rec['float64']['floor_mine32_vs_mine64']['max_abs']:.1e}); pad "
              f"{max([x['max_abs'] for x in pads], default=0.0):.1e}; red masks cos {rec['red_masks_all_ones']['cos']:.6f} "
              f"bn cos {rec['red_no_batch_norm']['cos']:.6f}; absmax {max(peaks.values()):.0f} "
              f"({rec['seconds_run']:.1f} s)", flush=True)
    fixture = [r for r in records if r["kind"] == "fixture"]
    padded = [r for r in records if r["prefix_rows"] < r["steps_bucket"]]
    peak_keys = sorted({k for r in records for k in r["fp32_absmax"]})
    absmax = {k: max(r["fp32_absmax"][k] for r in records) for k in peak_keys}
    summary = {
        "clips": len(records), "fixture_clips": len(fixture),
        "buckets": {str(s): [r["id"] for r in records if r["bucket_s"] == s] for s in da.CLIP_SECONDS},
        "rows_equal_reference": f"{sum(int(r['rows_equal_reference'].split('/')[0]) for r in fixture)}/"
                                f"{sum(int(r['rows_equal_reference'].split('/')[1]) for r in fixture)}",
        "publisher_prefix_bit_equal_oracle": f"{sum(r['publisher_prefix_bit_equal_oracle'] for r in fixture)}/{len(fixture)}",
        "publisher_encoder_bit_equal_oracle": f"{sum(r.get('publisher_encoder_bit_equal_oracle', False) for r in fixture)}/"
                                              f"{sum('publisher_encoder_bit_equal_oracle' in r for r in fixture)}",
        "host_lengths_equal": f"{sum(r['host_lengths_equal'] for r in records)}/{len(records)}",
        "subsampling_min_cos": min(r["subsampling"]["cos"] for r in records),
        "subsampling_max_abs": max(r["subsampling"]["max_abs"] for r in records),
        "encoder_min_cos": min(r["encoder"]["cos"] for r in records),
        "encoder_max_abs": max(r["encoder"]["max_abs"] for r in records),
        "prefix_min_cos": min(r["prefix"]["cos"] for r in records),
        "prefix_min_row_cos": min(r["prefix"]["min_row_cos"] for r in records),
        "prefix_max_abs": max(r["prefix"]["max_abs"] for r in records),
        "prefix_worst_clip": min(records, key=lambda r: r["prefix"]["cos"])["id"],
        "float64_mine_vs_publisher_max_abs": max(r["float64"]["mine64_vs_publisher64"]["max_abs"] for r in records),
        "fp32_floor_publisher_max_abs": max(r["float64"]["floor_publisher32_vs_publisher64"]["max_abs"] for r in records),
        "fp32_floor_mine_max_abs": max(r["float64"]["floor_mine32_vs_mine64"]["max_abs"] for r in records),
        "pad_invariance_max_abs": max((x["max_abs"] for r in records for x in r["pad_invariance"]), default=0.0),
        "pad_invariance_pairs": sum(len(r["pad_invariance"]) for r in records),
        "pad_cut_clips": {r["id"]: {"seconds": r["seconds"], "prefix_cos": r["prefix"]["cos"],
                                    "prefix_max_abs": r["prefix"]["max_abs"]} for r in records
                          if r["id"] in ("aud_14", "aud_15")},
        "red_masks_caught": f"{sum(r['red_masks_caught'] for r in padded)}/{len(padded)} clips with pad steps "
                            f"({sum(r['red_masks_caught'] for r in records)}/{len(records)} all clips)",
        "red_masks_max_cos_padded": max((r["red_masks_all_ones"]["cos"] for r in padded), default=None),
        "red_bn_caught": f"{sum(r['red_bn_caught'] for r in records)}/{len(records)}",
        "red_bn_max_cos": max(r["red_no_batch_norm"]["cos"] for r in records),
        "fp32_absmax": absmax, "fp32_absmax_overall": max(absmax.values()),
        "fp16_headroom": FP16_MAX / max(absmax.values()),
    }
    verdict = {
        "rows": all(r["rows_equal_reference"].split("/")[0] == r["rows_equal_reference"].split("/")[1] for r in fixture),
        "publisher_equals_oracle": all(r["publisher_prefix_bit_equal_oracle"] for r in fixture)
                                   and all(r.get("publisher_encoder_bit_equal_oracle", True) for r in fixture)
                                   and all(r.get("publisher_pre_encode_bit_equal_oracle", True) for r in fixture),
        "host_lengths": all(r["host_lengths_equal"] for r in records),
        "prefix_cos": all(r["prefix_pass"] for r in records),
        "float64_same_function": all(r["float64_pass"] for r in records),
        "pad_invariance": all(r["pad_pass"] for r in records),
        "red_masks": all(r["red_masks_caught"] for r in padded),
        "red_batch_norm": all(r["red_bn_caught"] for r in records),
    }
    status = "PASS" if all(verdict.values()) else "FAIL"
    doc = {"status": status, "verdict": verdict, "stage": "eager (CPU fp32, D1Audio vs the publisher's Audio)",
           "bars": {"prefix": PREFIX_BAR, "float64_max_abs": FLOAT64_BAR, "pad_invariance_max_abs": PAD_BAR,
                    "red_arms": "cos < 0.999999 (red masks: on the clips with pad steps in their bucket)"},
           "reference": ref_info, "weights": load_record, "summary": summary,
           "seconds": time.perf_counter() - t_start,
           "code_sha256": {f: sha256_file(HERE / f) for f in ("audio_check.py", "d1_omni_audio.py", "host.py",
                                                              "mel_host.py")},
           "environment": environment(("torch", "numpy", "soundfile", "safetensors", "tokenizers")),
           "clips": records}
    out = write_new(WORK / "results" / "audio_eager.json", doc)
    print(json.dumps({"status": status, "verdict": verdict, **summary}, indent=1))
    print("->", out)
    return 0 if status == "PASS" else 1


def fp32_peaks(model, t, p: int) -> dict:
    """The largest |value| over the valid rows at each stage of D1Audio's forward (the same ops, recomputed)."""
    import torch
    from torch.nn import functional as F

    peaks = {}
    conv = model.pre_encode.conv
    m = [w.reshape(1, 1, -1, 1) for w in t[1:]]
    lengths = [int(w.sum()) for w in t[1:]]
    x = t[0].transpose(1, 2).unsqueeze(1) * m[0]
    for name, idx, mask, n in (("sub_conv0", 0, 1, 1), ("sub_dw2", 2, 2, 2), ("sub_pw3", 3, 2, 2),
                               ("sub_dw5", 5, 3, 3), ("sub_pw6", 6, 3, 3)):
        x = model._conv(x, conv[idx]) * m[mask]
        peaks[name] = float(x[:, :, :lengths[n]].abs().max())
        if idx in (0, 3, 6):
            x = torch.relu(x)
    b, c, tt, f = x.shape
    x = model._lin(x.transpose(1, 2).reshape(b, tt, c * f), model.pre_encode.out)
    peaks["pre_encode_out"] = float(x[0, :p].abs().max())
    valid = t[4]
    key_mask = ((1.0 - valid) * -1.0e4).reshape(1, 1, 1, -1)
    query_mask = valid.reshape(1, 1, -1, 1)
    conv_valid = valid.reshape(1, 1, -1)
    pos = model.pos_emb
    residual = 0.0
    for layer in model.layers:
        x = x + model._ff(layer.feed_forward1, model._ln(layer.norm_feed_forward1, x)) * 0.5
        residual = max(residual, float(x[0, :p].abs().max()))
        x = x + model._attention(layer.self_attn, model._ln(layer.norm_self_att, x), pos, key_mask, query_mask)
        residual = max(residual, float(x[0, :p].abs().max()))
        x = x + model._conv_module(layer.conv, model._ln(layer.norm_conv, x), conv_valid)
        residual = max(residual, float(x[0, :p].abs().max()))
        x = x + model._ff(layer.feed_forward2, model._ln(layer.norm_feed_forward2, x)) * 0.5
        residual = max(residual, float(x[0, :p].abs().max()))
        x = model._ln(layer.norm_out, x)
    peaks["conformer_residual"] = residual
    a = model._lin(model._ln(model.adapter.norm, x), model.adapter.linear_1)
    peaks["adapter_hidden"] = float(a[0, :p].abs().max())
    a = model._lin(F.gelu(a), model.adapter.linear_2)
    peaks["adapter_out"] = float(a[0, :p].abs().max())
    return peaks


# --------------------------------------------------------------------------- stage runtime
def runtime_stage(args) -> int:
    import coreai.runtime as rt

    from _metrics import SHIP_BAR, row_record, summarize, wrong_pairing

    t_start = time.perf_counter()
    clips, ref_info = load_clips()
    snapshot = snapshot_dir()
    publisher = load_publisher_audio(snapshot)
    pub32, _ = publisher_audio_model(snapshot, publisher)

    def options(compute: str):
        if compute == "cpu_only":
            return rt.SpecializationOptions.cpu_only()
        kind = rt.ComputeUnitKind.gpu() if compute == "gpu" else rt.ComputeUnitKind.neural_engine()
        return rt.SpecializationOptions.from_preferred_compute_unit_kind(kind)

    def compiled(directory: Path) -> tuple[Path, dict]:
        aot = json.loads((directory / "provenance" / "aot-manifest.json").read_text())
        if aot.get("status") != "COMPILED" or not aot.get("aimodelc_kept", True):
            raise SystemExit(f"{directory}: no compiled bundle")
        aimodelc = directory / aot["aimodelc"]
        for item in aot["files"]:
            if sha256_file(aimodelc / item["path"]) != item["sha256"]:
                raise SystemExit(f"{aimodelc / item['path']} changed since its compile")
        return aimodelc, aot

    audio = {}
    for spec in args.audio:
        sec, directory = spec.split("=", 1)
        audio[int(sec)] = (Path(directory).resolve(), *compiled(Path(directory).resolve()))
    precisions = {v[2]["precision"] for v in audio.values()}
    if len(precisions) != 1:
        raise SystemExit(f"one precision per run: {precisions}")
    precision = precisions.pop()
    decide = {}
    for spec in args.decide:
        length, directory = spec.split("=", 1)
        decide[int(length)] = compiled(Path(directory).resolve())
    # the decision buckets the rows are routed by (round 12: host.ALL_BUCKETS puts an audio row of <= 128 positions
    # on L64 / L128); the default is the shipped set
    buckets = tuple(int(b) for b in args.buckets.split(",")) if args.buckets else host.BUCKETS
    largest = max(audio)

    # the reference prefix of every clip: the oracle npz (fixture clips) or the publisher's Audio here (synthetic)
    import torch
    reference = {}
    with torch.no_grad():
        for clip in clips:
            if clip["kind"] == "fixture":
                reference[clip["id"]] = np.asarray(clip["npz"]["prefix"], dtype=np.float32)
            else:
                reference[clip["id"]] = pub32(clip["samples"])[0].numpy()

    async def main_async():
        opts, dopts = options(args.compute), options(args.decide_compute or args.compute)
        fns, load_s = {}, {}
        for sec, (directory, aimodelc, aot) in sorted(audio.items()):
            t0 = time.perf_counter()
            m = await rt.AIModel.load(aimodelc, opts)
            fns[sec] = (m, m.load_function("main"))
            load_s[f"audio_{sec}s"] = time.perf_counter() - t0
        dfns = {}
        for length, (aimodelc, aot) in sorted(decide.items()):
            t0 = time.perf_counter()
            m = await rt.AIModel.load(aimodelc, dopts)
            dfns[length] = (m, m.load_function("main"), aot)
            load_s[f"decide_L{length}"] = time.perf_counter() - t0

        async def audio_call(inputs: dict) -> np.ndarray:
            out = await fns[inputs["sec"]][1]({k: rt.NDArray(np.ascontiguousarray(inputs[k]))
                                               for k in host.AUDIO_INPUT_NAMES})
            return np.array(out["prefix"].numpy(), copy=True)

        # (i) every clip's prefix, from the bundle of its bucket (the largest bucket here if its own is absent)
        prefixes, prefixes_np, clip_records = {}, {}, []
        for clip in clips:
            a = clip["samples"]
            own = host.audio_bucket_for(len(a))
            sec = own if own in fns else (largest if own < largest else None)
            if sec is None:
                clip_records.append({"id": clip["id"], "kind": clip["kind"], "bucket_s": own, "skipped": True,
                                     "why": f"no bundle for the {own} s bucket in this run"})
                continue
            inputs = host.audio_inputs(a, sec)
            inputs_np = host.audio_inputs(a, sec, numpy_mel=True)
            p = inputs["prefix_rows"]
            mel_equal_ref = (bool(np.array_equal(inputs["mel"][0, :, :clip["npz"]["mel"].shape[1]], clip["npz"]["mel"]))
                             if clip["kind"] == "fixture" else None)
            runs = []
            for _ in range(args.repeats):
                runs.append((await audio_call(inputs)).reshape(inputs["steps"], -1)[:p])
            drift = max(float(np.max(np.abs(r.astype(np.float64) - runs[0]))) for r in runs[1:])
            np_prefix = (await audio_call(inputs_np)).reshape(inputs["steps"], -1)[:p]
            stats = compare(runs[0], reference[clip["id"]])
            stats_np = compare(np_prefix, reference[clip["id"]])
            prefixes[clip["id"]], prefixes_np[clip["id"]] = runs[0], np_prefix
            clip_records.append({"id": clip["id"], "kind": clip["kind"], "public": clip["public"], "bucket_s": sec,
                                 "own_bucket_s": own, "samples": int(len(a)), "prefix_rows": p,
                                 "host_mel_equal_oracle_mel": mel_equal_ref, "prefix": stats,
                                 "prefix_eager_bar_pass": stats["cos"] >= PREFIX_BAR["cos"],
                                 "prefix_numpy_mel": stats_np, "drift": drift,
                                 "finite": bool(np.isfinite(runs[0]).all() and np.isfinite(np_prefix).all())})
            print(f"{clip['id']}: bucket {sec} P {p} cos {stats['cos']:.9f} max {stats['max_abs']:.3e} min-row "
                  f"{stats['min_row_cos']:.9f} drift {drift:.1e}; numpy mel cos {stats_np['cos']:.9f} max "
                  f"{stats_np['max_abs']:.3e}", flush=True)

        # (ii) end to end: every audio row of the fixture clips through the decision bundle of its bucket
        rows, hrows, arms_prefix = [], [], {"torch_mel": [], "numpy_mel": [], "reference": [], "control": []}
        order = [c["id"] for c in clips if c["kind"] == "fixture" and c["id"] in prefixes]
        by_id = {c["id"]: c for c in clips}
        for cid in order:
            clip = by_id[cid]
            entry, record = clip["entry"], clip["record"]
            for q in entry["questions"]:
                question = host.as_question(record["request"]["questions"][q["qid"]])
                hrow = host.Row(question=question, ids=q["ids"], markers=q["markers"], calibrate=q["calibrate"],
                                prefix_len=q["prefix"], mode="audio", max_len=0, qid=q["qid"])
                assert hrow.positions == q["positions"]
                rows.append({"id": cid, "qid": q["qid"], "mode": "audio", "source": entry["source"],
                             "type": q["type"], "K": q["K"], "positions": q["positions"], "near_tie": q["near_tie"],
                             "top2_margin": q["top2_margin"], "argmax_index": q["argmax_index"],
                             "calibrate": q["calibrate"], "bucket": q["bucket"],
                             "question": {"type": question.type, "instructions": question.instructions,
                                          "criteria": question.criteria},
                             "oracle": {"logits_raw": q["logits_raw"], "probs": q["probs"]}})
                hrows.append(hrow)
                donor = order[(order.index(cid) + 1) % len(order)]
                arms_prefix["torch_mel"].append(prefixes[cid])
                arms_prefix["numpy_mel"].append(prefixes_np[cid])
                arms_prefix["reference"].append(reference[cid])
                arms_prefix["control"].append((donor, np.resize(prefixes[donor], (hrow.prefix_len, prefixes[donor].shape[1]))
                                               .astype(np.float32)))

        async def decide_call(hrow, prefix, length):
            inputs, markers = host.graph_inputs(hrow, length, prefix)
            out = await dfns[length][1]({k: rt.NDArray(inputs[k]) for k in ("input_ids", "prefix_embeds", "pad_mask",
                                                                             "prefix_mask", "keep_right", "qtype_onehot")})
            scores = np.array(out["scores"].numpy(), copy=True).reshape(-1)
            return scores[markers], scores[:hrow.positions]

        arms = {"audio_bundle_prefix_torch_mel": [], "audio_bundle_prefix_numpy_mel": [], "reference_prefix": [],
                "control_next_clip_prefix": []}
        drifts = []
        for i, (hrow, row) in enumerate(zip(hrows, rows)):
            length = host.bucket_for(hrow.positions, buckets)
            if length not in dfns:
                raise SystemExit(f"no decision bundle for bucket {length} (row {row['id']}/{row['qid']})")
            outs = [await decide_call(hrow, arms_prefix["torch_mel"][i], length) for _ in range(args.repeats)]
            drift = max(float(np.max(np.abs(o[1].astype(np.float64) - outs[0][1]))) for o in outs[1:])
            drifts.append(drift)
            rec = row_record(row, outs[0][0])
            rec.update(bucket=length, repeat_drift_scores=drift)
            arms["audio_bundle_prefix_torch_mel"].append(rec)
            rec = row_record(row, (await decide_call(hrow, arms_prefix["numpy_mel"][i], length))[0])
            rec.update(bucket=length)
            arms["audio_bundle_prefix_numpy_mel"].append(rec)
            rec = row_record(row, (await decide_call(hrow, arms_prefix["reference"][i], length))[0])
            rec.update(bucket=length)
            arms["reference_prefix"].append(rec)
            donor, wrong = arms_prefix["control"][i]
            rec = row_record(row, (await decide_call(hrow, wrong, length))[0])
            rec.update(bucket=length, donor=donor)
            arms["control_next_clip_prefix"].append(rec)
        assert fns and dfns
        return load_s, clip_records, rows, arms, drifts

    load_s, clip_records, rows, arms, drifts = asyncio.run(main_async())
    summaries = {name: summarize(records) for name, records in arms.items()}
    for name, records in arms.items():
        summaries[name]["by_bucket"] = {str(b): {k: s[k] for k in ("rows", "argmax_equal", "max_abs_dp",
                                                                     "mean_row_max_abs_dp", "max_abs_dlogit")}
                                        for b in sorted({r["bucket"] for r in records})
                                        if (s := summarize([r for r in records if r["bucket"] == b]))}
    shipped = summaries["audio_bundle_prefix_torch_mel"]["ship_bar"]["status"]
    shipped_np = summaries["audio_bundle_prefix_numpy_mel"]["ship_bar"]["status"]
    control = summaries["control_next_clip_prefix"]["ship_bar"]["status"]
    pairing = wrong_pairing(rows, arms["audio_bundle_prefix_torch_mel"])
    common = {"stage": "runtime (Mac, Core AI Python runtime, AOT .aimodelc)", "compute": args.compute,
              "decide_compute": args.decide_compute or args.compute,
              "precision": precision, "load_s": load_s, "reference": ref_info,
              "code_sha256": {f: sha256_file(HERE / f) for f in ("audio_check.py", "host.py", "mel_host.py",
                                                                 "_metrics.py")},
              "environment": environment(("coreai-core", "numpy", "torch", "soundfile"))}
    outs = []
    for sec, (directory, aimodelc, aot) in sorted(audio.items()):
        recs = [r for r in clip_records if not r.get("skipped") and r["bucket_s"] == sec]
        prefix_summary = {
            "clips": len(recs), "clip_ids": [r["id"] for r in recs],
            "min_cos": min(r["prefix"]["cos"] for r in recs), "min_row_cos": min(r["prefix"]["min_row_cos"] for r in recs),
            "max_abs": max(r["prefix"]["max_abs"] for r in recs),
            "worst_clip": min(recs, key=lambda r: r["prefix"]["cos"])["id"],
            "numpy_mel_min_cos": min(r["prefix_numpy_mel"]["cos"] for r in recs),
            "numpy_mel_max_abs": max(r["prefix_numpy_mel"]["max_abs"] for r in recs),
            "drift_max": max(r["drift"] for r in recs), "repeats": args.repeats,
            "host_mel_equal_oracle_mel": f"{sum(bool(r['host_mel_equal_oracle_mel']) for r in recs)}/"
                                         f"{sum(r['host_mel_equal_oracle_mel'] is not None for r in recs)}",
            "bar_eager": PREFIX_BAR, "eager_bar_pass": f"{sum(r['prefix_eager_bar_pass'] for r in recs)}/{len(recs)}"}
        doc = {**common, "status": "PASS" if all(r["finite"] for r in recs) and prefix_summary["drift_max"] == 0.0
               else "RECORDED", "bucket_s": sec,
               "audio": {"folder": str(directory), "aimodelc": aimodelc.name, "bytes": aot["bytes"],
                         "resources_bin_bytes": aot["resources_bin_bytes"], "hashes": aot["hashes"]},
               "summary": prefix_summary, "clips": recs,
               "note": "the bar for the shipped forms is end to end (audio_e2e_<p>.json); the eager prefix bar is shown "
                       "for reference"}
        outs.append(write_new(WORK / "results" / f"{args.results_prefix}audio_{precision}_{sec}s_{args.compute}.json", doc))
    e2e_doc = {**common,
               "status": "PASS" if shipped == "PASS" and shipped_np == "PASS" and control == "FAIL" else "FAIL",
               "bar": SHIP_BAR,
               "audio": {str(k): {"folder": str(v[0]), "aimodelc": v[1].name, "hashes": v[2]["hashes"]}
                         for k, v in sorted(audio.items())},
               "decide": {str(k): {"folder": str(Path(v[0]).parent), "aimodelc": Path(v[0]).name,
                                   "precision": v[1]["precision"], "seq_len": v[1]["seq_len"], "hashes": v[1]["hashes"]}
                          for k, v in sorted(decide.items())},
               "decide_buckets": list(buckets),
               "rows": len(rows), "repeat_drift_max": max(drifts), "summaries": summaries,
               "control_must_fail": control, "wrong_pairing_oracle_swap": pairing,
               "skipped_clips": [r for r in clip_records if r.get("skipped")],
               "arms": arms, "seconds": time.perf_counter() - t_start}
    e_out = write_new(WORK / "results" / f"{args.results_prefix}audio_e2e_{precision}.json", e2e_doc)
    s, n, r = (summaries[k] for k in ("audio_bundle_prefix_torch_mel", "audio_bundle_prefix_numpy_mel",
                                      "reference_prefix"))
    print(json.dumps({"e2e": {
        "status": e2e_doc["status"], "rows": len(rows), "argmax": f"{s['argmax_equal']}/{s['rows']}",
        "non_near_tie": s["non_near_tie"], "max_abs_dp": s["max_abs_dp"], "mean": s["mean_row_max_abs_dp"],
        "max_abs_dlogit": s["max_abs_dlogit"], "numpy_mel_argmax": f"{n['argmax_equal']}/{n['rows']}",
        "numpy_mel_max_abs_dp": n["max_abs_dp"], "numpy_mel_mean": n["mean_row_max_abs_dp"],
        "reference_prefix_max_abs_dp": r["max_abs_dp"], "reference_prefix_mean": r["mean_row_max_abs_dp"],
        "control": control, "control_max_abs_dp": summaries["control_next_clip_prefix"]["max_abs_dp"],
        "wrong_pairing": pairing["status"], "drift": max(drifts), "load_s": load_s}}, indent=1))
    print("->", *outs, e_out)
    return 0 if e2e_doc["status"] == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["mel", "eager", "runtime"], required=True)
    ap.add_argument("--compute", choices=["cpu_only", "gpu", "neural_engine"], default="gpu")
    ap.add_argument("--decide-compute", choices=["cpu_only", "gpu", "neural_engine"], default=None,
                    help="runtime: the decision bundles' compute unit (default: --compute); an ANE probe of the %s "
                         "bundle keeps the decision graph on the GPU" % "audio")
    ap.add_argument("--audio", action="append", default=[], help="runtime: <sec>=<work>/compiled/<dir> (repeat)")
    ap.add_argument("--decide", action="append", default=[], help="runtime: <L>=<work>/compiled/<dir> (repeat)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--results-prefix", default="", help="runtime: prepended to the results files' names "
                    "(round 10: ship_runtime_ for the stripped bundles)")
    ap.add_argument("--buckets", default=None, help="runtime: the decision buckets the rows are routed by, comma list "
                    "(default host.BUCKETS; round 12: 64,128,256,512,1024,2048,4096 with --decide 64=... 128=...)")
    args = ap.parse_args()
    if args.stage == "mel":
        return mel_stage(args)
    if args.stage == "eager":
        return eager_stage(args)
    if not args.audio or not args.decide:
        ap.error("--stage runtime needs --audio and --decide")
    if args.repeats < 2:
        ap.error("at least 2 repeats")
    return runtime_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
