#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""Fun-ASR-Nano round-1 fixtures: 5 model-repo examples + FLEURS test (en_us, cmn_hans_cn, ja_jp).

Writes 16 kHz mono PCM16 wavs under ``<work>/_funasr_nano/fixtures/`` and ``meta.json`` (source,
revision, licence, duration, sha256, reference text for every clip):

- ``examples/{zh,en,ja,ko,yue}.wav`` — ``example/*.mp3`` of FunAudioLLM/Fun-ASR-Nano-2512
  (Apache-2.0 model repo), decoded with ``ffmpeg -ac 1 -ar 16000 -sample_fmt s16``.
- ``fleurs/<config>/<id>.wav`` — google/fleurs (CC BY 4.0) ``parquet-data/<config>/test-*.parquet``:
  rows sorted by (id, path), the first row of each id kept (FLEURS repeats an id once per
  speaker), then ascending id, the first 50 whose audio is <= 30 s.

Run with the oracle venv (pyarrow + soundfile):
    ~/code/coreai/_funasr_nano/venv-oracle/bin/python conversion/funasr_nano/make_fixtures.py
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import work_path  # noqa: E402

WORK = work_path("_funasr_nano")
FIXTURES = WORK / "fixtures"
OFFICIAL_REPO = "FunAudioLLM/Fun-ASR-Nano-2512"
OFFICIAL_REV = "272c57b82523ada6fd87095e955f8e29100979ab"
OFFICIAL_SNAPSHOT = WORK / "hf" / "hub" / "models--FunAudioLLM--Fun-ASR-Nano-2512" / "snapshots" / OFFICIAL_REV
EXAMPLES = ["zh", "en", "ja", "ko", "yue"]
FLEURS_REPO = "google/fleurs"
FLEURS_REV = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_PARQUET = WORK / "hf" / "fleurs" / FLEURS_REV   # <config>/test-00000-of-00001.parquet
FLEURS_CONFIGS = ["en_us", "cmn_hans_cn", "ja_jp"]
PER_CONFIG = 50
MAX_SECONDS = 30.0
SR = 16000
SHARED_META = Path("~/code/standup/handoffs/assets/2026-09-26-funasr-nano/shared/fixtures_meta.json").expanduser()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_pcm16(path: Path, audio: np.ndarray, sr: int) -> None:
    """int16 is written as is; float audio in [-1, 1) is quantized as round(x * 32768), clipped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.dtype != np.int16:
        audio = np.clip(np.round(np.asarray(audio, dtype=np.float64) * 32768.0), -32768, 32767).astype(np.int16)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def entry(name: str, wav: Path, source: dict, licence: str, reference: dict | None) -> dict:
    info = sf.info(str(wav))
    assert info.samplerate == SR and info.channels == 1 and info.subtype == "PCM_16", (wav, info)
    return {
        "name": name,
        "path": str(wav.relative_to(FIXTURES)),
        "source": source,
        "license": licence,
        "duration_s": round(info.frames / SR, 4),
        "num_samples": int(info.frames),
        "sha256": sha256(wav),
        "reference_text": reference,
    }


def make_examples() -> list[dict]:
    out = []
    for lang in EXAMPLES:
        src = OFFICIAL_SNAPSHOT / "example" / f"{lang}.mp3"
        dst = FIXTURES / "examples" / f"{lang}.wav"
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
                        "-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", str(dst)], check=True)
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                                "stream=sample_rate,channels", "-of", "csv=p=0", str(src)],
                               check=True, capture_output=True, text=True).stdout.strip()
        out.append(entry(lang, dst,
                         {"repo": OFFICIAL_REPO, "revision": OFFICIAL_REV, "file": f"example/{lang}.mp3",
                          "file_sha256": sha256(src), "mp3_sample_rate_channels": probe,
                          "decode": "ffmpeg -ac 1 -ar 16000 -sample_fmt s16"},
                         "Apache-2.0 model repo example", None))
    return out


def decode_fleurs_audio(cell) -> tuple[np.ndarray, int, str]:
    """FLEURS parquet audio cell (struct of WAV bytes) -> (mono samples, sample_rate, source subtype).

    int16 stays exact when the source is PCM16; anything else comes back as float64 in [-1, 1).
    """
    if not (isinstance(cell, dict) and cell.get("bytes")):
        raise ValueError(f"unexpected audio cell layout: {type(cell)} {list(cell) if isinstance(cell, dict) else ''}")
    with sf.SoundFile(io.BytesIO(cell["bytes"])) as f:
        sr, subtype, channels = f.samplerate, f.subtype, f.channels
        data = f.read(dtype="int16" if subtype == "PCM_16" else "float64", always_2d=True)
    if channels != 1:
        raise ValueError(f"expected mono FLEURS audio, got {channels} channels")
    return data[:, 0], sr, subtype


def make_fleurs(config: str) -> tuple[list[dict], dict]:
    parquet = FLEURS_PARQUET / config / "test-00000-of-00001.parquet"
    table = pq.read_table(parquet)
    cols = table.column_names
    meta_cols = [c for c in cols if c != "audio"]
    rows = table.select(meta_cols).to_pylist()
    order = sorted(range(len(rows)), key=lambda i: (int(rows[i]["id"]), str(rows[i].get("path", ""))))
    seen, picked, skipped_long = set(), [], []
    for i in order:
        rid = int(rows[i]["id"])
        if rid in seen:
            continue
        seen.add(rid)
        audio, sr, subtype = decode_fleurs_audio(table.column("audio")[i].as_py())
        if sr != SR:
            raise ValueError(f"{config} id {rid}: sample rate {sr} (expected {SR}); resample path not implemented")
        if audio.shape[0] > MAX_SECONDS * SR:
            skipped_long.append(rid)
            continue
        dst = FIXTURES / "fleurs" / config / f"{rid}.wav"
        write_pcm16(dst, audio, sr)
        r = rows[i]
        picked.append(entry(f"{config}_{rid}", dst,
                            {"repo": FLEURS_REPO, "revision": FLEURS_REV, "config": config, "split": "test",
                             "id": rid, "fleurs_file": (r.get("path") or "").rsplit("/", 1)[-1],
                             "gender": r.get("gender"), "source_subtype": subtype,
                             "to_pcm16": "as is" if subtype == "PCM_16" else "round(x * 32768), clipped"},
                            "CC BY 4.0 (FLEURS)",
                            {"transcription": r.get("transcription"), "raw_transcription": r.get("raw_transcription")}))
        if len(picked) == PER_CONFIG:
            break
    stats = {"rows": len(rows), "unique_ids": len({int(r["id"]) for r in rows}), "columns": cols,
             "skipped_over_30s_ids": skipped_long, "picked": len(picked)}
    return picked, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-shared-copy", action="store_true", help="skip the copy into the shared fixtures folder")
    ap.add_argument("--examples-only", action="store_true", help="only the 5 model-repo examples")
    args = ap.parse_args()
    FIXTURES.mkdir(parents=True, exist_ok=True)
    clips = make_examples()
    selection = {}
    for config in [] if args.examples_only else FLEURS_CONFIGS:
        picked, stats = make_fleurs(config)
        clips += picked
        selection[config] = stats
        print(f"[fleurs] {config}: {stats['picked']} clips (rows {stats['rows']}, unique ids {stats['unique_ids']}, "
              f"skipped >30 s: {stats['skipped_over_30s_ids']})", flush=True)
    meta = {
        "fixtures_root": str(FIXTURES),
        "sample_rate": SR,
        "format": "wav PCM_16 mono",
        "selection": {
            "examples": f"{OFFICIAL_REPO}@{OFFICIAL_REV} example/*.mp3 -> ffmpeg 16 kHz mono s16",
            "fleurs": (f"{FLEURS_REPO}@{FLEURS_REV} parquet-data/<config>/test-00000-of-00001.parquet; rows sorted "
                       f"by (id, path), first row per id, ascending id, first {PER_CONFIG} with <= {MAX_SECONDS:g} s"),
            "fleurs_stats": selection,
        },
        "clips": clips,
    }
    (FIXTURES / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"[meta] {len(clips)} clips -> {FIXTURES / 'meta.json'}", flush=True)
    if not args.no_shared_copy and not args.examples_only:
        SHARED_META.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FIXTURES / "meta.json", SHARED_META)
        print(f"[meta] copied -> {SHARED_META}", flush=True)


if __name__ == "__main__":
    main()
