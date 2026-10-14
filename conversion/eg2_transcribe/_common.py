"""Paths, fixtures and metrics shared by the EmbeddingGemma 2 audio-encoder scripts in this directory.

Every location resolves from an argument, then an environment variable, then a default under
``conversion/_paths.py`` (``ZOO_WORK_ROOT`` / ``ZOO_CODE_ROOT``); nothing here hardcodes a home directory.

    EG2STT_WORK     work directory (default: $ZOO_WORK_ROOT/_eg2stt) — exports, logs
    EG2_MODEL_DIR   google/embeddinggemma-2 snapshot with config.json + model.safetensors
                    (default: <work>/data/hf/embeddinggemma-2, else the HF cache at the pinned revision)
    EG2_FIXTURES    reader fixtures (default: $ZOO_CODE_ROOT/eg2-transcribe/reader/fixtures): test-clean-20/<id>/
                    {features.npy [n_frames, 128], soft_tokens.npy [n_tokens, 512]} from transformers 5.19.0 fp32 sdpa
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import code_path, hf_snapshot, work_path  # noqa: E402

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
MODEL_SAFETENSORS_SHA256 = "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"
BAR_COS_MEAN, BAR_COS_MIN = 0.999, 0.99          # per-token cosine vs the fp32 reference, every clip
BUCKETS = (1000, 3000)                            # frames: 10 s and 30 s windows (2999 valid frames in 30 s)


def fp16_io(form: str) -> bool:
    """Forms whose graph takes and returns float16: fp16 and its variants fp16<x> (not fp16w32), and EverPaper's."""
    return form == "everpaper" or (form.startswith("fp16") and form != "fp16w32")


def _env(name: str) -> Path | None:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else None


def work_dir() -> Path:
    return _env("EG2STT_WORK") or work_path("_eg2stt")


def model_dir(arg: str | None = None) -> Path:
    if arg:
        return Path(arg).expanduser()
    if (p := _env("EG2_MODEL_DIR")) is not None:
        return p
    local = work_dir() / "data" / "hf" / "embeddinggemma-2"
    if (local / "model.safetensors").exists():
        return local
    return Path(hf_snapshot(MODEL_ID, revision=MODEL_REVISION))


def fixtures_dir(arg: str | None = None) -> Path:
    if arg:
        return Path(arg).expanduser()
    return _env("EG2_FIXTURES") or code_path("eg2-transcribe", "reader", "fixtures")


def exports_dir(arg: str | None = None) -> Path:
    if arg:
        return Path(arg).expanduser()
    return work_dir() / "exports"


def sha256_of(path: Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def verify_checkpoint(mdir: Path) -> str:
    digest = sha256_of(mdir / "model.safetensors")
    if digest != MODEL_SAFETENSORS_SHA256:
        raise SystemExit(f"{mdir}/model.safetensors is not {MODEL_ID}@{MODEL_REVISION[:8]} ({digest})")
    return digest


def load_clips(fx: Path) -> list[dict]:
    """The 20 test-clean fixture clips in MANIFEST order: id, features [n_frames, 128], soft_tokens [n, 512]."""
    manifest = json.loads((fx / "MANIFEST.json").read_text())
    clips = []
    for e in manifest["entries"]:
        d = fx / "test-clean-20" / e["id"]
        feats = np.load(d / "features.npy").astype(np.float32)
        ref = np.load(d / "soft_tokens.npy").astype(np.float32)
        assert feats.shape == (e["n_frames"], 128) and ref.shape == (e["n_tokens"], 512), e["id"]
        clips.append({"id": e["id"], "n_frames": int(e["n_frames"]), "n_tokens": int(e["n_tokens"]),
                      "features": feats, "ref": ref})
    return clips


def window_input(feats: np.ndarray, bucket: int) -> tuple[np.ndarray, int]:
    """Right-zero-pad (or cut) a clip's frames to one bucket. Returns ([bucket, 128], valid frames).

    A clip longer than the bucket is cut to its leading `bucket` frames. The encoder is causal apart from the
    one-frame look-ahead of the 3x3 subsampling convs, so the leading bucket/4 tokens of the cut are the clip's
    own leading tokens (token j reads frames <= 4j + 3 < bucket)."""
    n = min(feats.shape[0], bucket)
    x = np.zeros((bucket, feats.shape[1]), dtype=np.float32)
    x[:n] = feats[:n]
    return x, n


def compared_tokens(clip: dict, bucket: int) -> int:
    """How many leading soft tokens of a bucketed window are compared with the clip's reference."""
    from eg2_audio_coreai import n_tokens
    return min(clip["n_tokens"], n_tokens(min(clip["n_frames"], bucket)))


def token_stats(out: np.ndarray, ref: np.ndarray) -> dict:
    """Per-token cosine (float64) and element error of out [n, 512] against ref [n, 512]."""
    a = out.astype(np.float64)
    b = ref.astype(np.float64)
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)
    order = np.argsort(cos)[:5]
    return {"n": int(a.shape[0]), "cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
            "worst5": [{"pos": int(i), "cos": float(cos[i])} for i in order],
            "max_abs": float(np.abs(a - b).max()), "ref_max_abs": float(np.abs(b).max()),
            "nonfinite": int((~np.isfinite(out)).sum())}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, allow_nan=True) + "\n")
