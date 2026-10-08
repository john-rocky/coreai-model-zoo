"""The fixture file a reference was made from, found by its sha256.

fixtures/records.json is rewritten when the fixture changes (round 5: the image records; round 6: twelve more audio
records); the version a reference read stays next to it as records.v<N>.json, byte for byte. A script that checks a
reference against its fixtures asks for the sha256 the reference names (ref["fixtures"]["sha256"]) and gets whichever
file holds it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from _paths import work_path

FIXTURES_DIR = work_path("_d1_omni", "fixtures")
VERSIONS = {  # sha256 -> the file name that keeps that version
    "c0781185cf9b86210024f94fa645f047af843bfaaf7d80c703c4272634030123": "records.v1.json",  # round 1 (round 2's oracle)
    "07ea38a2c0f3d8da4afe4dd4ae79e083c46404cb8ab0ba150b9c0a3fca83ae62": "records.v2.json",  # round 5 (the image rows)
    "18b4ddff73a8044b63f949c69245210e1af8a26c48c5fdb785fd91a7807ead5b": "records.v3.json",  # round 6 (aud_04..aud_15)
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def fixtures_path(sha256: str) -> Path:
    """records.json if it is that version, else the kept copy of it; SystemExit if neither holds it."""
    candidates = [FIXTURES_DIR / "records.json"]
    if sha256 in VERSIONS:
        candidates.append(FIXTURES_DIR / VERSIONS[sha256])
    for path in candidates:
        if path.exists() and _sha256(path) == sha256:
            return path
    raise SystemExit(f"no fixture file with sha256 {sha256} in {FIXTURES_DIR} (looked at {[p.name for p in candidates]})")
