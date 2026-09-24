"""Export the Nemotron-3-Diarization graph (n3d_model.N3DGraph, function `main`) to a Core AI
`.aimodel`, plus the four host-side constants as raw little-endian float32 files.

    ~/code/coreai/coreai-models/.venv/bin/python export_n3d.py --dtype float32   # parity bundle (cpu_only)
    ~/code/coreai/coreai-models/.venv/bin/python export_n3d.py --dtype float16   # ship candidate (Mac GPU)
    ~/code/coreai/coreai-models/.venv/bin/python export_n3d.py --dtype float16 --profile offline --skip-assets
    ~/code/coreai/coreai-models/.venv/bin/python export_n3d.py --assets-only     # the .f32le files only
    ~/code/coreai/coreai-models/.venv/bin/python export_n3d.py --dtype float16 --safe-ln --skip-assets  # ANE variant
    ~/code/coreai/coreai-models/.venv/bin/python export_n3d.py --metadata          # metadata.json (every bundle here)
    ~/code/coreai/coreai-models/.venv/bin/python export_n3d.py --metadata --ship   # metadata.ship.json (the 4 shipped)

Two profiles, one graph each (fixed T):
    streaming  T=541 = speaker cache 264 + FIFO 264 + chunk 9 + look-ahead 4 (low_latency; the
               very_low / ultra_low modes use the same graph with fewer real rows)
    offline    T=684 = speaker cache 264 + FIFO 40 + chunk 340 + look-ahead 40

Graph I/O is float32 for both dtypes (the float16 bundle casts inside the graph), so the host
contract does not change with the bundle:

    packed [1, T, 512]   f32   [cache | FIFO | chunk embeds], real rows at [0, L), padding after
    valid  [1, T]        f32   1.0 for rows < L, 0.0 after
    logits [1, T*8, 8]   f32   speaker logits, rows [0, L*8) real

Before exporting, the eager fp32 graph is re-checked on captured steps of the profile (the L=T step
and shorter steps with noise padding); a miss stops the export.

Writes to _work/artifacts/:
    n3d_<profile>_<dtype>.aimodel
    embedder_projection.f32le   [512, 1024]  model.audio_tower.embedder.projection.weight, C order
    silence_embeds.f32le        [512]        silence_embeds
    mel_filters_128x257.f32le   [128, 257]   librosa.filters.mel(sr=16000, n_fft=512, n_mels=128,
                                             fmin=0, fmax=8000, norm="slaney"), asserted bit-identical
                                             to the transformers feature extractor's filters
                                             (dumped by make_reference.py into _work/fe_constants.npz)
    hann_window_400.f32le       [400]        torch.hann_window(400, periodic=False), asserted
                                             bit-identical to the feature extractor's window (same
                                             file); shipped so the host does not recompute it (a
                                             float64 cos rounds 9 of the 400 entries 1 ulp apart)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gate_reauthor import MAX_ABS, MIN_CORR, corr, pack  # noqa: E402
from n3d_model import HID, REPO_ID, REVISION, SUB, T_OFFLINE, T_STREAM, load_checkpoint  # noqa: E402

WORK = HERE / "_work"
ART = WORK / "artifacts"
PROFILES = {"streaming": T_STREAM, "offline": T_OFFLINE}
PROFILE_TEXT = {"streaming": "speaker cache 264 + FIFO 264 + chunk 9 + look-ahead 4",
                "offline": "speaker cache 264 + FIFO 40 + chunk 340 + look-ahead 40"}


class FP32IO(nn.Module):
    """float32 graph I/O around a lower-precision graph. (The attribute must not be called
    `graph`: torch.export's unlift resolves `graph.*` against GraphModule.graph and fails.)"""

    def __init__(self, inner: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.inner = inner.to(dtype)
        self.dtype = dtype

    def forward(self, packed, valid):
        return self.inner(packed.to(self.dtype), valid.to(self.dtype)).to(torch.float32)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_f32le(path: Path, arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr, dtype="<f4")
    path.write_bytes(a.tobytes(order="C"))
    return sha256(path)


def eager_recheck(graph, profile: str = "streaming") -> None:
    T = PROFILES[profile]
    if profile == "streaming":
        d = np.load(WORK / "chunk_io_diarization_example_ll.npz")
        L = d["L"]
        steps = [int(np.nonzero(L == T)[0][0]), int(np.nonzero(d["compressed_after"])[0][0])]
        short = np.nonzero(L < T)[0]
        steps.append(int(short[-1]))
    else:
        # offline capture: steps 0 (L=380, compressed after it) and 1 (L=684); the last step carries
        # a masked key row (the extractor's extra centered frame) and is gated in gate_closed_loop.py
        d = np.load(WORK / "chunk_io_diarization_example_offline.npz")
        steps = [i for i in range(int(d["n_steps"])) if f"s{i:03d}_step_mask" not in d.files]
    for i in steps:
        emb, ref = d[f"s{i:03d}_inputs_embeds"], d[f"s{i:03d}_chunk_logits"]
        packed, valid = pack(emb, T, "noise", i)
        with torch.inference_mode():
            out = graph(packed, valid)[0, : emb.shape[0] * SUB].numpy()
        mx, c = float(np.abs(out - ref).max()), corr(out, ref)
        ok = mx <= MAX_ABS and c >= MIN_CORR
        print(f"[eager fp32] step {i} L={emb.shape[0]}: max|Δlogit| {mx:.3e} corr {c:.9f} -> "
              f"{'OK' if ok else 'MISS'}", flush=True)
        if not ok:
            raise SystemExit("eager re-check failed: not exporting")


def host_assets(host: dict) -> dict:
    import librosa

    ART.mkdir(parents=True, exist_ok=True)
    proj = host["model.audio_tower.embedder.projection.weight"].numpy()
    sil = host["silence_embeds"].numpy()
    assert proj.shape == (HID, SUB * 128) and sil.shape == (HID,)
    mel = librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0.0, fmax=8000.0, norm="slaney")
    assert mel.shape == (128, 257) and mel.dtype == np.float32, (mel.shape, mel.dtype)
    fe = np.load(WORK / "fe_constants.npz")
    ref = fe["mel_filters"]
    ref = ref if ref.shape == (128, 257) else ref.T
    assert ref.shape == (128, 257) and ref.dtype == np.float32, (ref.shape, ref.dtype)
    if not np.array_equal(mel.view(np.uint32), ref.view(np.uint32)):
        raise SystemExit(f"mel filters differ from the transformers feature extractor: max|Δ| "
                         f"{np.abs(mel - ref).max():.3e}, differing entries {(mel != ref).sum()}")
    print("[assets] librosa 0.11.0 slaney filterbank bit-identical to the transformers feature extractor")
    win = torch.hann_window(400, periodic=False).numpy()
    ref_win = fe["hann_window"]
    assert win.shape == (400,) and win.dtype == np.float32 and ref_win.dtype == np.float32
    if not np.array_equal(win.view(np.uint32), ref_win.view(np.uint32)):
        raise SystemExit(f"Hann window differs from the feature extractor's: {(win != ref_win).sum()} entries")
    print(f"[assets] torch {torch.__version__} hann_window(400, periodic=False) bit-identical to the "
          "transformers feature extractor's window")
    shas = {
        "embedder_projection.f32le": write_f32le(ART / "embedder_projection.f32le", proj),
        "silence_embeds.f32le": write_f32le(ART / "silence_embeds.f32le", sil),
        "mel_filters_128x257.f32le": write_f32le(ART / "mel_filters_128x257.f32le", mel),
        "hann_window_400.f32le": write_f32le(ART / "hann_window_400.f32le", win),
    }
    for k, v in shas.items():
        print(f"[assets] {k} sha256 {v}")
    return shas


def export(graph, dtype_name: str, profile: str = "streaming", tag: str = "") -> Path:
    import coreai.runtime as rt
    from coreai_torch import TorchConverter, get_decomp_table

    T = PROFILES[profile]
    dtype = torch.float16 if dtype_name == "float16" else torch.float32
    model = FP32IO(graph, dtype).eval() if dtype != torch.float32 else graph
    example = (torch.zeros(1, T, HID), torch.ones(1, T))
    t0 = time.time()
    with torch.no_grad():
        ep = torch.export.export(model, example)
    ep = ep.run_decompositions(get_decomp_table())
    t1 = time.time()
    prog = TorchConverter().add_exported_program(
        exported_program=ep, input_names=("packed", "valid"), output_names=("logits",)).to_coreai()
    t2 = time.time()
    prog.optimize()
    t3 = time.time()
    out = ART / f"n3d_{profile}{tag}_{dtype_name}.aimodel"
    shutil.rmtree(out, ignore_errors=True)
    meta = rt.AIModelAssetMetadata()
    meta.license = "openmdw-1.1"
    meta.author = "NVIDIA (Nemotron-3-Diarization); Core AI export: coreai-model-zoo"
    meta.model_description = (
        f"Nemotron-3-Diarization {profile} encoder + head{' (LayerNorm on x/64)' if tag == '_safeln' else ''}, "
        f"{dtype_name}, T={T} "
        f"({PROFILE_TEXT[profile]}). Inputs packed [1,{T},512] "
        f"(embedder output), valid [1,{T}]; output logits [1,{T * SUB},8]. "
        f"Source: {REPO_ID}@{REVISION}")
    meta.creation_date = int(time.time())
    prog.save_asset(out, meta)
    t4 = time.time()
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"[export] {out.name}: {size / 1e6:.1f} MB | torch.export+decomp {t1 - t0:.1f} s, "
          f"convert {t2 - t1:.1f} s, optimize {t3 - t2:.1f} s, save {t4 - t3:.1f} s, total {t4 - t0:.1f} s",
          flush=True)
    stats = {"dtype": dtype_name, "profile": profile, "T": T, "bytes": size, "export_s": round(t1 - t0, 1),
             "convert_s": round(t2 - t1, 1), "optimize_s": round(t3 - t2, 1), "save_s": round(t4 - t3, 1),
             "total_s": round(t4 - t0, 1)}
    name = f"export_{dtype_name}.json" if profile == "streaming" and not tag else \
        f"export_{profile}{tag}_{dtype_name}.json"
    (WORK / name).write_text(json.dumps(stats, indent=1))
    return out


def tree_sha256(root: Path) -> str:
    """sha256 over a bundle directory: the sorted '<relative path>\\0<file sha256>\\n' lines."""
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(f"{f.relative_to(root).as_posix()}\0{sha256(f)}\n".encode())
    return h.hexdigest()


def bundle_contract(path: Path) -> dict:
    """The I/O contract read back from the saved bundle (function `main`), not from the export code."""
    import asyncio

    import coreai.runtime as rt

    opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    model = asyncio.run(rt.AIModel.load(path, opts))
    desc = model.load_function("main").desc

    def nd(d):
        return {"shape": [int(x) for x in d.shape], "dtype": str(d.dtype).split(".")[-1]}
    return {"function": desc.name, "inputs": {n: nd(desc.input_descriptor(n)) for n in desc.input_names},
            "outputs": {n: nd(desc.output_descriptor(n)) for n in desc.output_names},
            "states": list(desc.state_names)}


# The iPhone bundles that ship, as aot_compile.py lays them out: <work>/aot/<tag_>ios_gpu/<name>.h18p.aimodelc.
SHIP_AOT = {"streaming": WORK / "aot" / "ios_gpu", "offline": WORK / "aot" / "offline_ios_gpu"}


def ship_bundles(contracts: dict) -> dict:
    """The four published bundles: the macOS fp16 .aimodel of each profile and its iOS h18p GPU compile.
    An iOS bundle is never loaded on a Mac, so its contract is the one of the .aimodel it was compiled from."""
    import subprocess

    tool = subprocess.run(["xcrun", "coreai-build", "--version"], capture_output=True, text=True).stdout.strip()
    tool = tool.removeprefix("coreai-build").strip()                    # "coreai-build 3600.83.1" -> "3600.83.1"
    out = {}
    for profile in PROFILES:
        src = ART / f"n3d_{profile}_float16.aimodel"
        aot = SHIP_AOT[profile] / f"n3d_{profile}_float16.h18p.aimodelc"
        if not (src.exists() and aot.exists()):
            raise SystemExit(f"--ship needs {src.relative_to(HERE)} and {aot.relative_to(HERE)} (aot_compile.py)")
        if aot.stat().st_mtime < src.stat().st_mtime:
            raise SystemExit(f"{aot.relative_to(HERE)} is older than {src.name}: compile it again (aot_compile.py)")
        rec_file = WORK / "aot" / ("aot_compile.json" if profile == "streaming" else f"aot_compile_{profile}.json")
        rec = json.loads(rec_file.read_text())["targets"][SHIP_AOT[profile].name.replace(f"{profile}_", "")]
        cmd = rec["cmd"].replace(str(src), src.name).replace(rec["out"], "<dir>")
        for p, target in ((src, "macOS 27, Apple silicon GPU (compiled at load)"),
                          (aot, "iOS 27, iPhone 17 Pro GPU (compiled ahead of time, h18p)")):
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            entry = {"profile": profile, "T": PROFILES[profile], "dtype": "float16", "target": target,
                     "compute_preference": "gpu", "bytes": size, "mb": round(size / 1e6, 1), "tree_sha256": tree_sha256(p)}
            if p is src:
                entry["contract"] = contracts[profile]
            else:
                entry["compiled_from"] = src.name
                entry["contract"] = f"that of {src.name}"
                entry["aot"] = {"command": cmd, "coreai_build": tool, "ane_regions": rec["ane_regions"]["n_regions"]}
            out[p.name] = entry
    return out


def write_metadata(ship: bool = False) -> Path:
    """_work/artifacts/metadata.json: what a host needs to drive the exported bundles, and where every
    number comes from (host_loop.py / mel_frontend.py are the source; nothing is typed in twice).
    ship=True writes metadata.ship.json instead, the file the Hugging Face repo carries: the same host
    contract, with `bundles` = the four shipped bundles only (stage_ship.py copies it as metadata.json)."""
    import importlib.metadata as im
    import platform
    import subprocess

    import host_loop as hl
    import mel_frontend as mf
    from n3d_model import SAFETENSORS_SHA256, safetensors_path, sha256_file

    st = safetensors_path()
    st_sha = sha256_file(st)
    assert st_sha == SAFETENSORS_SHA256, (st_sha, SAFETENSORS_SHA256)
    sw = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    cache = hl.SpeakerCache(0, 0)
    assets = {
        "embedder_projection.f32le": {"shape": [HID, SUB * mf.N_MELS], "what": "model.audio_tower.embedder.projection.weight (Linear 1024->512, no bias); embeds = stacked @ W.T"},
        "silence_embeds.f32le": {"shape": [HID], "what": "silence_embeds: the speaker cache's silence row (1 per speaker)"},
        "mel_filters_128x257.f32le": {"shape": [mf.N_MELS, mf.N_FFT // 2 + 1], "what": "librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0, fmax=8000, norm='slaney'); bit-identical to the transformers feature extractor's"},
        "hann_window_400.f32le": {"shape": [mf.WIN], "what": "torch.hann_window(400, periodic=False); bit-identical to the feature extractor's"},
    }
    for name, rec in assets.items():
        rec["sha256"] = sha256(ART / name)
        rec["dtype"] = "float32 little-endian, C order"
    bundles = {}
    if ship:
        bundles = ship_bundles({pr: bundle_contract(ART / f"n3d_{pr}_float16.aimodel") for pr in PROFILES})
    else:
        for profile in PROFILES:
            for dtype in ("float16", "float32"):
                p = ART / f"n3d_{profile}_{dtype}.aimodel"
                if not p.exists():
                    continue
                bundles[p.name] = {"profile": profile, "dtype": dtype, "T": PROFILES[profile],
                                   "bytes": sum(f.stat().st_size for f in p.rglob("*") if f.is_file()),
                                   "tree_sha256": tree_sha256(p), "contract": bundle_contract(p),
                                   "role": "ship candidate (GPU)" if dtype == "float16" else "parity (cpuOnly)"}
    meta = {
        "model": "Nemotron-3-Diarization (streaming Sortformer, 8 speakers)",
        "license": "openmdw-1.1",
        "source": {"repo": REPO_ID, "revision": REVISION, "file": "model.safetensors", "sha256": st_sha,
                   "bytes": Path(st).stat().st_size, "reference": "transformers Nemotron3DiarizationForAudioFrameClassification (fp32)"},
        "toolchain": {"coreai-torch": im.version("coreai-torch"), "coreai-core": im.version("coreai-core"),
                      "torch": torch.__version__, "macos": f"{platform.mac_ver()[0]} ({sw})"},
        "graph": {
            "function": "main",
            "inputs": {"packed": "[1, T, 512] float32: [speaker cache | FIFO | chunk rows (+ look-ahead)] "
                                 "left-packed at rows [0, L); rows [L, T) are ignored (zero them)",
                       "valid": "[1, T] float32: 1.0 for rows < L, 0.0 after"},
            "outputs": {"logits": "[1, T*8, 8] float32 speaker logits at 10 ms; rows [0, L*8) are real"},
            "T": dict(PROFILES),
            "host_side": "sigmoid, the 8x average pool to encoder rate, the speaker cache (AOSC + FIFO), segmentation",
        },
        "host": {
            "frame_s": 0.01, "encoder_frame_s": 0.08, "subsampling": hl.SUB, "num_speakers": hl.N_SPK, "hidden": HID,
            "profiles": {
                "streaming": {"T": PROFILES["streaming"], "fifo_length": hl.PROFILES["streaming"]["fifo_length"],
                              "update_period": hl.PROFILES["streaming"]["update_period"],
                              "modes": {m: {"chunk": c, "lookahead": la} for m, (c, la) in mf.STREAMING_MODES.items()},
                              "chunking": "per chunk mel from its own audio slice: chunk 0 = audio[:((c+la)*8-1)*160+200] "
                                          "center=True; chunk k>=1 starts at 160*k*c*8-256 and holds (c+la)*8*160+400 samples, "
                                          "center=False; the first chunk that would run past the audio takes the rest, is the "
                                          "last (no look-ahead) and emits its true mel frame count"},
                "offline": {"T": PROFILES["offline"], "fifo_length": hl.PROFILES["offline"]["fifo_length"],
                            "update_period": hl.PROFILES["offline"]["update_period"],
                            "chunk": hl.PROFILES["offline"]["chunk"], "lookahead": hl.PROFILES["offline"]["lookahead"],
                            "chunking": "whole-recording mel (center=True) -> embeds -> chunks of 340 rows + up to 40 "
                                        "look-ahead rows; output cut to the mel frame count"},
            },
            "step": "rows = [cache | FIFO | chunk]; logits = graph(rows)[:L*8]; probs = avg_pool8(sigmoid(logits)); "
                    "emit logits[(n_cache+n_fifo)*8 : +min(n_chunk*8, n_emit)]; then the cache update",
            "speaker_cache": {
                "length": hl.CACHE_LEN, "silence_frames_per_speaker": hl.SIL_PER_SPK,
                "prediction_score_threshold": hl.PRED_THRESHOLD, "latest_frames_score_boost": hl.LATEST_BOOST,
                "min_positive_scores": cache.min_positive_scores, "strong_boost": {"frames": cache.n_strong, "add": hl.STRONG_BOOST},
                "weak_boost": {"frames": cache.n_weak, "add": hl.WEAK_BOOST},
                "update": "transformers Nemotron3DiarizationSpeakerCache.update/_compress @ 4b28d51, line by line (host_loop.py)",
                "score_dtype": "float64 from the float32 probabilities (transformers: float32; see host_loop.py)",
                "topk_ties": "lower index first", "sentinel": "the silence row (row N: silence_embeds, zero probs)",
            },
            "segments": "transformers extract_speaker_dict: per speaker, runs of sigmoid(logits) > 0.5; overlaps kept",
        },
        "mel": {"sample_rate": mf.SR, "preemphasis": float(mf.PREEMPH), "preemphasis_first_sample": "kept",
                "n_fft": mf.N_FFT, "win_length": mf.WIN, "hop": mf.HOP, "window": "hann_window_400.f32le, centered in 512 (zeros at [0,56) and [456,512))",
                "power": "|rFFT|^2 (as sqrt(re^2+im^2)^2 in float32)", "n_mels": mf.N_MELS, "fmin": 0.0, "fmax": 8000.0,
                "norm": "slaney", "log": "log(mel + 2^-24)", "log_guard": float(mf.LOG_GUARD), "normalize": None,
                "center_pad": "256 zeros each side when center=True (first chunk / offline), none otherwise",
                "valid_frames": "center: n // 160; not center: (n - 512) // 160 + 1",
                "stacking": "8 frames -> 1024 (zero-pad the frame count to a multiple of 8 in the log-mel domain)"},
        "assets": assets,
        "bundles": bundles,
    }
    if ship:
        meta["notes"] = [
            "compute_preference: load with SpecializationOptions(preferredComputeUnitKind: .gpu); the .h18p.aimodelc "
            "bundles were also compiled with --preferred-compute gpu.",
            "The float32 bundles are for parity only (CPU) and are not shipped: export_n3d.py --dtype float32 "
            "[--profile offline] rebuilds them.",
        ]
    out = ART / ("metadata.ship.json" if ship else "metadata.json")
    out.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[metadata] {out.relative_to(HERE)}: {len(bundles)} bundles, {len(assets)} assets, "
          f"safetensors sha256 {st_sha[:12]}…", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["float32", "float16"])
    ap.add_argument("--profile", choices=list(PROFILES), default="streaming")
    ap.add_argument("--skip-assets", action="store_true")
    ap.add_argument("--assets-only", action="store_true", help="write the .f32le constants and stop")
    ap.add_argument("--safe-ln", action="store_true",
                    help="SafeLayerNorm variant (n3d_model.SafeLayerNorm, for fp16 on the ANE): n3d_<profile>_safeln_<dtype>.aimodel")
    ap.add_argument("--metadata", action="store_true",
                    help="write _work/artifacts/metadata.json (the host contract of the exported bundles) and stop")
    ap.add_argument("--ship", action="store_true",
                    help="with --metadata: write metadata.ship.json, whose bundles are the four shipped ones")
    args = ap.parse_args()
    if args.ship and not args.metadata:
        ap.error("--ship goes with --metadata")
    if args.metadata:
        write_metadata(ship=args.ship)
        return
    if not args.assets_only and args.dtype is None:
        ap.error("--dtype is required unless --assets-only")

    T = PROFILES[args.profile]
    graph, host = load_checkpoint(T=T, safe_ln=args.safe_ln)
    print(f"[weights] strict load OK (sha256 verified), profile {args.profile}, T={T}, safe_ln={args.safe_ln}",
          flush=True)
    if not args.skip_assets:
        shas = host_assets(host)
        (WORK / "assets_sha256.json").write_text(json.dumps(shas, indent=1))
    if args.assets_only:
        return
    eager_recheck(graph, args.profile)
    export(graph, args.dtype, args.profile, "_safeln" if args.safe_ln else "")


if __name__ == "__main__":
    main()
