"""Shared paths, source pinning and hashing for the Granite-Embedding-97M port.

Everything the scripts in this directory read or write lives under
`_paths.work_path("granite-embedding-97m")` (checkpoint, fixtures, results) and
`_paths.exports_dir() / "granite-embedding-97m"` (bundles laid out the way the HF repo is).
Set `ZOO_WORK_ROOT` / `ZOO_EXPORTS` to move them; nothing here hardcodes a home directory.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import exports_dir, from_record_path, record_path, work_path  # noqa: E402

MODEL_ID = "ibm-granite/granite-embedding-97m-multilingual-r2"
MODEL_SHA = "835ad14087e140460703cf0fae09f97d469d65c2"
FAMILY = "granite-embedding-97m"
SEQ_LENS = (128, 512)

# Upstream file digests at the pinned revision (LFS sha256 for the weights; verified on
# download by `download_source`). The graph refuses any other checkpoint.
SOURCE_SHA256 = {
    "model.safetensors": "f3ea88b230492811046145513710e76b4cc8c2ad49e8708da0e7247e548903be",
    "tokenizer.json": "4f2842d568e2724370aec203652a42ac783c7937f8347a1a2cc7506d71f1582f",
}


def work_dir() -> Path:
    return work_path(FAMILY)


def source_dir() -> Path:
    return work_dir() / "source"


def fixtures_dir() -> Path:
    return work_dir() / "fixtures"


def results_dir() -> Path:
    return work_dir() / "results"


def export_root() -> Path:
    return exports_dir() / FAMILY


def sha256_of(path: Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def hashes(paths) -> dict[str, str]:
    """sha256 per file, keyed by its `record_path` name (records are published: no absolute path)."""
    return {record_path(p): sha256_of(Path(p)) for p in paths}


def recorded_hash(record: dict[str, str], path: Path) -> str:
    """The digest a `hashes` record holds for `path` (records written before `record_path` key it by
    the absolute path)."""
    for key in (record_path(path), str(path)):
        if key in record:
            return record[key]
    raise KeyError(f"no hash recorded for {record_path(path)}")


def verify_hashes(records: dict[str, str]) -> None:
    for name, digest in records.items():
        actual = sha256_of(from_record_path(name))
        if actual != digest:
            raise ValueError(f"Hash changed since gate: {name}")


def download_source() -> Path:
    """Fetch the pinned checkpoint into `source_dir()` and verify it, or verify what is there."""
    source = source_dir()
    if not (source / "model.safetensors").exists():
        from huggingface_hub import HfApi, snapshot_download

        pinned = HfApi().model_info(MODEL_ID, revision=MODEL_SHA)
        assert pinned.sha == MODEL_SHA, pinned.sha
        snapshot_download(MODEL_ID, revision=MODEL_SHA, local_dir=source, max_workers=1,
                          allow_patterns=["*.json", "README.md", "model.safetensors", "LICENSE*", "NOTICE*"],
                          ignore_patterns=["onnx/*", "openvino/*"])
    return verify_source()


def verify_source() -> Path:
    source = source_dir()
    for name, digest in SOURCE_SHA256.items():
        actual = sha256_of(source / name)
        if actual != digest:
            raise ValueError(f"{name} is not the pinned revision {MODEL_SHA}: {actual}")
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "modernbert":
        raise ValueError("Not a ModernBERT checkpoint")
    return source


def golden(seq_len: int) -> dict:
    path = fixtures_dir() / f"golden_s{seq_len}.json"
    data = json.loads(path.read_text())
    if data["model_sha"] != MODEL_SHA or data["sequence_length"] != seq_len:
        raise ValueError(f"Fixture identity mismatch in {path}")
    return data


def write_json(path: Path, value, fresh: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x" if fresh else "w") as fh:
        fh.write(text)


def file_inventory(root: Path) -> list[dict]:
    return [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha256_of(p)}
            for p in sorted(root.rglob("*")) if p.is_file()]
