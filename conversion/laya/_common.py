"""Shared paths, source pinning, fixtures and records for the laya multilingual port.

Everything the scripts in this directory read or write lives under
`_paths.work_path("laya-multilingual")` (frozen fixtures, the LiteRT lane's captures, oracle dumps,
results) and `_paths.exports_dir() / "laya-multilingual"` (bundles laid out the way the HF repo is).
The checkpoint is the pinned Hugging Face snapshot, resolved through `_paths.hf_snapshot` and
verified by sha256 on every load. Set `ZOO_WORK_ROOT` / `ZOO_EXPORTS` / `HF_HUB_CACHE` to move them;
nothing here hardcodes a home directory.

Work-dir layout:
  fixtures/        the frozen LiteRT-lane fixtures (201 rows per window, official answers at T=1)
  oracle-litert/   the LiteRT lane's per-row captures (s256/NNN.npz, s512/NNN.npz, official model padded to 512)
  oracle/          this port's oracle dumps (oracle_laya.py): outputs_s{S}.npz, hidden/s{S}/NNN.npz
  results/         every gate's JSON record
"""
from __future__ import annotations

import datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import exports_dir, hf_snapshot, work_path  # noqa: E402

MODEL_ID = "convaiinnovations/laya"
MODEL_SHA = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
SUBFOLDER = "multilingual"
FAMILY = "laya-multilingual"
WINDOWS = (256, 512)
ROW_COUNT = 201
HEAD_MAX_LEN = 256          # rl_agent_config.json head_max_len
SOURCE_MAX_LEN = 1024       # rl_agent_config.json max_len (the windows are this port's static cut)
OPTION_TEXT_TOKENS = 48     # laya/common.py build_sequence keeps 48 text tokens per option
PAD_ID, SEP_ID, CLS_ID, UNK_ID, MASK_ID = 0, 1, 2, 3, 4
QTYPES = ("choice", "score", "noul")
HIDDEN = 768

# Upstream file digests at the pinned revision (the HF cache stores these bytes under their
# content-addressed blob names; checked 2026-09-23). The graph refuses any other checkpoint.
SOURCE_SHA256 = {
    "model.safetensors": "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204",
    "tokenizer/tokenizer.json": "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
    "tokenizer/tokenizer_config.json": "6c6b2d8e3c84ce0e671c129cd6b374b235d6f9863042a5836358d00a89bbb5a1",
    "encoder/config.json": "83f6916d13ef0f556ac461f28308dc2bffa7ebeadee8ec9e2db5812020ea5bb4",
    "rl_agent_config.json": "25061739243b617ad88d1219ba6f8a9c86c5881ca28df024fa2d9b3b2fcc30c6",
}

# The frozen fixtures from the LiteRT lane (byte-identical to
# codex-conversions/2026-09-21/laya/fixtures/, checked 2026-09-23). A swapped fixture must stop the
# gates, not silently re-baseline them.
FIXTURE_SHA256 = {
    "ml_rows_s256.json": "98d0e591826c708375191425f4b475d1fa3f997051172ee24b91e9a52f46c83f",
    "ml_rows_s512.json": "dce5e82ea27f86d6027b34c9e76c565d862cb865e27b31f80db9d23e475756e0",
    "ml_fixtures.json": "c5126081671f6e5797b66c08a92b7fd8f4ed1877918c185f620d6221dca951a4",
    "ml_oracle_fp32.json": "1b3025b944f1154c6baa40ef4a4128536e8fa1302c9856419926459a1b892318",
    "ml_agent_config.json": "31dba6ac7bf7cc646237a76a24dd2fb68d8b986bbdce25b95c14ecd4f70d92ef",
    "laya_ml_calibration.json": "80e148c68154b607e7bb66446f064b1fe5da649e1074c917b25e02551a566f2c",
}

# Rows whose every hidden state the oracle saves and the layer gate checks (same indices in both
# windows): EN, JA, mixed, score/noul/choice, K = 2…20, the shortest row, the longest row, and six
# rows that fill the 256 window exactly (state right-truncated, no padding at 256).
SUBSET = (0, 1, 2, 8, 12, 13, 18, 20, 59, 94, 128, 181, 194, 195)


def work_dir() -> Path:
    return work_path(FAMILY)


def fixtures_dir() -> Path:
    return work_dir() / "fixtures"


def captures_dir() -> Path:
    return work_dir() / "oracle-litert"


def oracle_dir() -> Path:
    return work_dir() / "oracle"


def results_dir() -> Path:
    return work_dir() / "results"


def export_root() -> Path:
    return exports_dir() / FAMILY


def snapshot_root() -> Path:
    return Path(hf_snapshot(MODEL_ID, revision=MODEL_SHA))


def source_dir() -> Path:
    return snapshot_root() / SUBFOLDER


def sha256_of(path: Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def hashes(paths) -> dict[str, str]:
    return {str(Path(p)): sha256_of(Path(p)) for p in paths}


def verify_hashes(records: dict[str, str]) -> None:
    for name, digest in records.items():
        if sha256_of(Path(name)) != digest:
            raise ValueError(f"Hash changed since it was gated: {name}")


def verify_source() -> Path:
    """The pinned multilingual checkpoint, or an error naming the file that differs."""
    source = source_dir()
    for name, digest in SOURCE_SHA256.items():
        actual = sha256_of(source / name)
        if actual != digest:
            raise ValueError(f"{name} is not the pinned revision {MODEL_SHA}: {actual}")
    config = json.loads((source / "encoder" / "config.json").read_text())
    if config.get("model_type") != "modernbert":
        raise ValueError("Not a ModernBERT encoder")
    return source


def verify_fixtures() -> Path:
    directory = fixtures_dir()
    for name, digest in FIXTURE_SHA256.items():
        actual = sha256_of(directory / name)
        if actual != digest:
            raise ValueError(f"fixture {name} is not the frozen file: {actual}")
    return directory


def load_rows(window: int) -> list[dict]:
    """The 201 question rows the official builder produced for one window, in fixture order."""
    if window not in WINDOWS:
        raise ValueError(f"window {window} is not one of {WINDOWS}")
    rows = json.loads((verify_fixtures() / f"ml_rows_s{window}.json").read_text())
    ids = [r["row_id"] for r in rows]
    if len(rows) != ROW_COUNT or len(set(ids)) != ROW_COUNT or {r["window"] for r in rows} != {window}:
        raise ValueError(f"fixture identity mismatch for window {window}")
    for r in rows:
        n, k = r["sequence_length"], r["K"]
        if len(r["sequence_ids"]) != n or n > window or len(r["marker_positions"]) != k or k < 2:
            raise ValueError(f"malformed fixture row {r['row_id']}")
        if r["sequence_ids"][0] != CLS_ID or any(r["sequence_ids"][m] != MASK_ID for m in r["marker_positions"]):
            raise ValueError(f"fixture row {r['row_id']} does not start with CLS or a marker is not MASK")
    return rows


def load_fixture_states() -> dict[str, dict]:
    return {f["id"]: f for f in json.loads((verify_fixtures() / "ml_fixtures.json").read_text())}


def load_calibration() -> dict:
    return json.loads((verify_fixtures() / "laya_ml_calibration.json").read_text())


def load_capture(window: int, index: int, row: dict | None = None) -> dict[str, np.ndarray]:
    """One LiteRT-lane capture: the official model on this window's ids padded to 512, batch 1."""
    with np.load(captures_dir() / f"s{window}" / f"{index:03d}.npz") as data:
        capture = {k: data[k] for k in data.files}
    if row is not None:
        n = row["sequence_length"]
        ids = capture["input_ids"][0]
        if ids[:n].tolist() != row["sequence_ids"] or np.any(ids[n:] != PAD_ID):
            raise ValueError(f"capture s{window}/{index:03d} is not row {row['row_id']}")
    return capture


def row_inputs(row: dict, window: int) -> dict[str, np.ndarray]:
    """The graph inputs for one row: ids right-padded with PAD 0, int32 mask, float32 one-hot type."""
    n = row["sequence_length"]
    if n > window:
        raise ValueError(f"{row['row_id']} has {n} tokens, more than the window {window}")
    input_ids = np.full((1, window), PAD_ID, dtype=np.int32)
    input_ids[0, :n] = row["sequence_ids"]
    attention_mask = np.zeros((1, window), dtype=np.int32)
    attention_mask[0, :n] = 1
    qtype_onehot = np.zeros((1, 3), dtype=np.float32)
    qtype_onehot[0, row["qtype"]] = 1.0
    return {"input_ids": input_ids, "attention_mask": attention_mask, "qtype_onehot": qtype_onehot}


def write_json(path: Path, value, fresh: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x" if fresh else "w") as fh:
        fh.write(text)


def file_inventory(root: Path) -> list[dict]:
    return [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha256_of(p)}
            for p in sorted(root.rglob("*")) if p.is_file()]


def _command(argv: list[str]) -> str:
    try:
        return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable ({type(error).__name__})"


def environment(packages=("torch", "coreai-torch", "coreai-core", "numpy", "safetensors")) -> dict:
    """What every gate record carries: machine, OS build, toolchain, package versions, wall clock."""
    from importlib import metadata

    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "date": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds"),
        "machine": _command(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "macos_version": _command(["sw_vers", "-productVersion"]),
        "macos_build": _command(["sw_vers", "-buildVersion"]),
        "coreai_build": _command(["xcrun", "coreai-build", "--version"]),
        "python": platform.python_version(),
        "packages": versions,
        "argv": sys.argv,
    }
