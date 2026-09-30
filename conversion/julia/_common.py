"""Shared paths, source pinning, row sets and records for the Julia-1 port.

Everything the scripts in this directory read or write lives under
`_paths.work_path("julia-1")` (the pinned evaluation data, oracle dumps, results) and
`_paths.exports_dir() / "julia-1"` (bundles laid out the way the HF repo is). The checkpoint is the
pinned Hugging Face snapshot, resolved through `_paths.hf_snapshot` and verified by sha256 on every
load. Set `ZOO_WORK_ROOT` / `ZOO_EXPORTS` / `HF_HOME` to move them; nothing here hardcodes a home
directory.

Work-dir layout:
  data/       typed-decisions test parquet (pinned revision + sha256) and Julia-1-ONNX parity-cases.json
  oracle/     oracle_julia.py's dumps: rows.json (every row: ids, markers, the publisher's raw logits),
              hidden/s{S}/NNN.npz and hidden_eager/s{S}/NNN.npz for the SUBSET rows
  results/    every gate's JSON record
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

MODEL_ID = "SupersonicLabs/Julia-1"
MODEL_SHA = "a85b127321d580d65176c89ced8273f305745d85"
FAMILY = "julia-1"
HF_REPO = "mlboydaisuke/Julia-1-CoreAI"
WINDOWS = (512, 1024)
# The question-and-options budget per window. 1024 / 512 is the publisher's typed-decisions protocol
# (scripts/reproduce_typed.py) and inference-policy.json's head_length. At 512 the publisher's builder
# needs head_length + 4 < max_length, so the 512 window takes load_model's default 256. Under strict
# encoding nothing is ever cut, so the budget only decides which rows are accepted: an accepted row's
# ids are the same at either budget (oracle_julia.py checks this on every row).
HEAD_LENGTH = {512: 256, 1024: 512}
OPTION_TOKENS = 48
PAD_ID, SEP_ID, CLS_ID, MASK_ID = 0, 1, 2, 4
QTYPES = ("choice", "score", "noul")
HIDDEN = 384

# Upstream file digests at the pinned revision (checked 2026-09-30 against the HF lfs oids and the
# publisher's own WEIGHTS_SHA256). The graph refuses any other checkpoint.
SOURCE_SHA256 = {
    "model.safetensors": "df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72",
    "tokenizer/tokenizer.json": "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
    "tokenizer/tokenizer_config.json": "6b069e57db0ce0794c22547725275f22618809c4ef4249307337e651a0dfef8c",
    "encoder/config.json": "c79cda42d42ddf777218254845ce92fb25a10dc35874be40d06abd707894523d",
    "julia_config.json": "b9b881646beeac5414eaa78b39d3fe89ff8a2c062731ad718432620f9aee6bf7",
    "inference-policy.json": "415468ae1c229186e63a277b2fd2e5ee3bbf4c00655125c0e43e01d5a8406e8a",
    "julia/data.py": "e3510fa4152ec11fa193046715991f44d7c2f85fd2488a98ef11c9d3db23da4e",
    "julia/typed.py": "ed89e66a70fcedac1339347bd8fcca69fd2cd1a31535ad0148dae42c610b544d",
    "julia/model.py": "ef2ba82fe20cdf0db7bb887e9ef075476ed08b985ce9a95be0de3e26246ecc81",
    "julia/inference.py": "79b4e716365a6e07da4580ead725d5b76d06ef3a824a3ec23738feba64704b36",
    "julia/router/engine.py": "91bb30987ff8626ed1610955c6fd796939a5d271e2405b73f8674ccb947eb931",
    "julia/router/encoder.py": "df25efee2ed916d52af9c4e9c0d98e80854ca70bbfc8140f1de871f290fd3238",
    "scripts/reproduce_typed.py": "07ed6a7d68d1e367b4417ab61c1f7e7bba4858a4e6ba22b01797f244abb3b160",
}

# The public evaluation data this port gates on (both Apache-2.0).
DATASET_ID = "LocalLLaMA/typed-decisions"
DATASET_REVISION = "c76749ec58bd8c3d2ea706b31c333a9059c38f90"
DATASET_FILE = "test-00000-of-00001.parquet"
DATASET_SHA256 = "4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/all/{DATASET_FILE}"
PARITY_ID = "SupersonicLabs/Julia-1-ONNX"
PARITY_REVISION = "82a2fadf8fccfccdc5fd4e1009ba8f1a265eb7a8"
PARITY_FILE = "parity-cases.json"
PARITY_SHA256 = "534670f25823f3f9abf9e812500918fabc606682dfd4e2456d827889d9bdca20"
PARITY_URL = f"https://huggingface.co/{PARITY_ID}/resolve/{PARITY_REVISION}/{PARITY_FILE}"

# The publisher's CPU FP32 reproduction (metrics/typed-cpu-20260926.json): the oracle must hit these.
PUBLISHED_TYPED_CPU = {"choice": (426, 600), "score": (542, 800), "noul": (483, 600)}


def work_dir() -> Path:
    return work_path(FAMILY)


def data_dir() -> Path:
    return work_dir() / "data"


def oracle_dir() -> Path:
    return work_dir() / "oracle"


def results_dir() -> Path:
    return work_dir() / "results"


def export_root() -> Path:
    return exports_dir() / FAMILY


def source_dir() -> Path:
    return Path(hf_snapshot(MODEL_ID, revision=MODEL_SHA))


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
    """The pinned checkpoint, or an error naming the file that differs."""
    source = source_dir()
    for name, digest in SOURCE_SHA256.items():
        actual = sha256_of(source / name)
        if actual != digest:
            raise ValueError(f"{name} is not the pinned revision {MODEL_SHA}: {actual}")
    return source


def fetch_pinned(url: str, path: Path, digest: str) -> Path:
    """The pinned file, downloaded once over plain HTTP and checked by sha256 every time."""
    if not path.exists():
        import shutil
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=180) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target)
        partial.replace(path)
    actual = sha256_of(path)
    if actual != digest:
        raise ValueError(f"{path} is not the pinned file ({actual} != {digest})")
    return path


def dataset_path() -> Path:
    return fetch_pinned(DATASET_URL, data_dir() / DATASET_FILE, DATASET_SHA256)


def parity_path() -> Path:
    return fetch_pinned(PARITY_URL, data_dir() / PARITY_FILE, PARITY_SHA256)


def typed_requests() -> list[dict]:
    """The 2,000 typed-decisions test questions as the publisher's reproduce_typed.py renders them:
    criteria descriptions become the options (a score rubric in order; noul [false, true], literal
    'false' / 'true' when the question has no criteria), the state stays a JSON object."""
    import pyarrow.parquet as pq

    rows = []
    for case in pq.read_table(dataset_path()).to_pylist():
        state, gold = json.loads(case["state"]), json.loads(case["gold"])
        for question_id, question in json.loads(case["questions"]).items():
            kind, criteria = question["type"], question.get("criteria")
            if criteria is None and kind == "noul":
                criteria = {"false": "false", "true": "true"}
            if isinstance(criteria, list):
                criteria = {str(i): value for i, value in enumerate(criteria)}
            if not isinstance(criteria, dict):
                raise ValueError("Missing option descriptions")
            keys = ["false", "true"] if kind == "noul" else list(criteria)
            rows.append({"row_id": f"typed:{case['id']}:{question_id}", "set": "typed", "type": kind, "keys": keys,
                         "gold": str(gold[question_id]["label"]),
                         "request": {"state": state, "question": question["instructions"], "type": kind,
                                     "options": [criteria[key] for key in keys]},
                         "named": {"id": question_id, "question": question}})
    return rows


def parity_requests() -> list[dict]:
    """The publisher's 100 WebGPU parity requests, each with the publisher's own PyTorch logits."""
    cases = json.loads(parity_path().read_text())
    return [{"row_id": f"parity:{i:03d}", "set": "parity", "type": c["request"].get("type", "choice"),
             "keys": [str(k) for k in range(len(c["request"]["options"]))], "gold": None,
             "request": c["request"], "recorded_pytorch_logits": c["pytorch_logits"]} for i, c in enumerate(cases)]


def load_rows(window: int, sets=("typed", "parity", "fill")) -> list[dict]:
    """The oracle's rows that fit `window`, in oracle order, with the reference for that window."""
    if window not in WINDOWS:
        raise ValueError(f"window {window} is not one of {WINDOWS}")
    document = json.loads((oracle_dir() / "rows.json").read_text())
    rows = []
    for row in document["rows"]:
        ref = row["windows"].get(str(window))
        if ref is None or row["set"] not in sets:
            continue
        rows.append({**{k: v for k, v in row.items() if k != "windows"}, **ref, "window": window})
    return rows


def subset(window: int) -> list[str]:
    document = json.loads((oracle_dir() / "rows.json").read_text())
    return document["subset"][str(window)]


def row_inputs(row: dict, window: int) -> dict[str, np.ndarray]:
    """The graph inputs for one row: ids right-padded with PAD 0, int32 mask, float32 one-hot type."""
    n = len(row["ids"])
    if n > window:
        raise ValueError(f"{row['row_id']} has {n} tokens, more than the window {window}")
    input_ids = np.full((1, window), PAD_ID, dtype=np.int32)
    input_ids[0, :n] = row["ids"]
    attention_mask = np.zeros((1, window), dtype=np.int32)
    attention_mask[0, :n] = 1
    qtype_onehot = np.zeros((1, 3), dtype=np.float32)
    qtype_onehot[0, QTYPES.index(row["type"])] = 1.0
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
