#!/usr/bin/env python3
"""Machine-independent paths for the zoo's conversion, gate, and upload scripts.

Every conversion script needs things that live *outside* this repository: a
Hugging Face snapshot cache, a directory to write `.aimodel` bundles into, and a
scratch area holding downloaded checkpoints and oracle dumps. Hardcoding those to
one machine's home directory makes the script unrunnable anywhere else, so they
all resolve through this module instead.

Four roots, each overridable by an environment variable:

    ZOO_WORK_ROOT   scratch checkouts, downloaded weights, oracle dumps
                    default: the parent directory of this repository
    ZOO_EXPORTS     where exported `.aimodel` bundles are written
                    default: $ZOO_WORK_ROOT/coreai-models/exports
    ZOO_CODE_ROOT   holds sibling checkouts (litertlm-convert, coreai-kit, ...)
                    default: the parent of $ZOO_WORK_ROOT
    HF_HUB_CACHE    Hugging Face snapshot cache (the standard huggingface_hub
                    variable; $HF_HOME/hub and ~/.cache/huggingface/hub are the
                    fallbacks, in that order)

`ZOO_SMOKE` (ad-hoc gates, default `<repo>/_smoke`) and `ZOO_GPU_LOCK` (the
advisory Mac-GPU lock shared by parallel sessions) follow the same rule.

Records (export manifests, gate records, fixtures) are published next to the
bundles they describe, so a path goes into one through `record_path` /
`record_argv`, never as an absolute path: relative to this repository, `~/...`
under the home folder, or the name of the root it is under. `from_record_path` reads
such a name back.

The defaults reproduce the layout the zoo was developed in — a checkout at
`<code>/coreai/coreai-models-community` with bundles in
`<code>/coreai/coreai-models/exports` — so the machine that produced the
published bundles keeps working with no environment set at all.

Usage from a script in `conversion/`:

    from _paths import exports_dir, hf_snapshot, work_path

and from a script in a subdirectory of `conversion/`:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _paths import exports_dir, hf_snapshot, work_path   # noqa: E402
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

__all__ = [
    "repo_root",
    "work_root",
    "work_path",
    "code_root",
    "code_path",
    "exports_dir",
    "smoke_dir",
    "gpu_lock",
    "hf_cache",
    "hf_snapshot",
    "hf_snapshot_files",
    "record_path",
    "record_argv",
    "from_record_path",
]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def repo_root() -> Path:
    """This repository (coreai-models-community)."""
    return Path(__file__).resolve().parents[1]


def work_root() -> Path:
    """Scratch root: checkpoint downloads, oracle dumps, sibling coreai-models."""
    return _env_path("ZOO_WORK_ROOT", repo_root().parent)


def work_path(*parts: str) -> Path:
    return work_root().joinpath(*parts)


def code_root() -> Path:
    """Where sibling projects are checked out (litertlm-convert, coreai-kit, ...)."""
    return _env_path("ZOO_CODE_ROOT", work_root().parent)


def code_path(*parts: str) -> Path:
    return code_root().joinpath(*parts)


def exports_dir() -> Path:
    """Where exported `.aimodel` bundles are written."""
    return _env_path("ZOO_EXPORTS", work_path("coreai-models", "exports"))


def smoke_dir() -> Path:
    """Ad-hoc gates and reference dumps (`_smoke/` at the repo root)."""
    return _env_path("ZOO_SMOKE", repo_root() / "_smoke")


def gpu_lock() -> Path:
    """Advisory lock file serializing Mac-GPU work across parallel sessions."""
    return _env_path("ZOO_GPU_LOCK", work_path("_GPU_LOCK"))


def hf_cache() -> Path:
    """The huggingface_hub snapshot cache, resolved the way the library does."""
    if value := os.environ.get("HF_HUB_CACHE"):
        return Path(value).expanduser()
    if value := os.environ.get("HF_HOME"):
        return Path(value).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _snapshot_root(repo_id: str) -> Path:
    return hf_cache() / ("models--" + repo_id.replace("/", "--")) / "snapshots"


def hf_snapshot_files(repo_id: str, pattern: str) -> list[str]:
    """Every file matching `pattern` inside the local snapshots of `repo_id`.

    Newest snapshot first, so `[0]` is the most recently downloaded revision.
    Returns an empty list when the repo (or the pattern) is not in the cache —
    callers that require a hit should use `hf_snapshot`.
    """
    snaps = sorted(
        (p for p in _snapshot_root(repo_id).glob("*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [hit for snap in snaps for hit in sorted(glob.glob(str(snap / pattern)))]


def hf_snapshot(repo_id: str, pattern: str | None = None, revision: str | None = None) -> str:
    """Local snapshot directory for `repo_id`, or the first file matching `pattern`.

    `revision` pins one snapshot hash — use it where the port was gated against a
    specific revision, so a newer download cannot silently change the weights.

    Raises FileNotFoundError with the command that fixes it — an export that
    silently picks up the wrong checkpoint is worse than one that stops.
    """
    if revision is not None:
        snap = _snapshot_root(repo_id) / revision
        hits = sorted(glob.glob(str(snap / pattern))) if pattern else ([str(snap)] if snap.is_dir() else [])
    elif pattern is None:
        snaps = sorted(
            (p for p in _snapshot_root(repo_id).glob("*") if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        hits = [str(p) for p in snaps]
    else:
        hits = hf_snapshot_files(repo_id, pattern)
    if not hits:
        what = repo_id if pattern is None else f"{repo_id}:{pattern}"
        raise FileNotFoundError(
            f"no local Hugging Face snapshot for {what} under {hf_cache()} — "
            f"run `huggingface-cli download {repo_id}` first, "
            f"or point HF_HUB_CACHE at the cache that has it"
        )
    return hits[0]


def _record_roots() -> list[tuple[str, Path]]:
    """(label, root) pairs, most specific first after the repository and the home folder.

    The `$` labels only show up for a root outside the home folder (set by its variable), so a
    record written there still names no machine and `from_record_path` can still read it back.
    """
    return [("", repo_root()), ("~", Path.home()),
            ("$ZOO_EXPORTS", exports_dir()), ("$ZOO_SMOKE", smoke_dir()), ("$HF_HUB_CACHE", hf_cache()),
            ("$ZOO_WORK_ROOT", work_root()), ("$ZOO_CODE_ROOT", code_root())]


def record_path(path: str | os.PathLike) -> str:
    """How a record names a local file or folder, without this machine's absolute path.

    The first rule that applies:
      inside this repository           relative to it          conversion/laya/aot_laya.py
      under the home folder            `~/` + the rest         ~/.cache/huggingface/hub/models--...
      under a root of this module that is outside the home folder
                                       its variable + the rest $ZOO_WORK_ROOT/laya-multilingual/...
      anywhere else                    the last name only      llm-runner
    Both the path as given and its symlink-resolved form are tried. `from_record_path` reads the
    first three forms back. A value that is not an absolute path (a relative path, a repo id)
    names no machine and comes back unchanged.
    """
    raw = os.fspath(path)
    if not os.path.isabs(os.path.expanduser(raw)):
        return raw
    given = Path(os.path.abspath(os.path.expanduser(raw)))
    forms = list(dict.fromkeys([given, given.resolve()]))
    for label, root in _record_roots():
        for base in dict.fromkeys([Path(os.path.abspath(root)), root.resolve()]):
            for form in forms:
                if form == base:
                    return label or "."
                if form.is_relative_to(base):
                    rest = form.relative_to(base).as_posix()
                    return f"{label}/{rest}" if label else rest
    return given.name or raw


def _is_local_path(arg: str) -> bool:
    """An argument that names a local path: `~...`, or `/...` whose top folder exists (so a
    prompt such as "/no_think ..." stays as typed)."""
    if arg.startswith("~"):
        return True
    return arg.startswith("/") and os.path.exists("/" + arg.split("/")[1])


def record_argv(argv) -> list[str]:
    """`argv` for a record: every argument that is a local path, also as `--flag=<path>`, the way
    `record_path` writes it; every other argument as given."""
    out = []
    for arg in map(str, argv):
        head, sep, tail = arg.partition("=") if arg.startswith("-") else ("", "", arg)
        out.append(head + sep + (record_path(tail) if _is_local_path(tail) else tail))
    return out


def from_record_path(name: str | os.PathLike) -> Path:
    """The local path a `record_path` name stands for; an absolute name (records written before
    record_path existed) is returned as it is."""
    text = os.fspath(name)
    if text.startswith("$"):
        label, _, rest = text.partition("/")
        roots = {lab: root for lab, root in _record_roots() if lab.startswith("$")}
        if label in roots:
            return roots[label] / rest if rest else roots[label]
    p = Path(text).expanduser()
    return p if p.is_absolute() else repo_root() / p


def _prefer_plain_http_transfers() -> None:
    """Make Hugging Face downloads take the plain-HTTP path, unless the caller says otherwise.

    The Xet transfer backend stalls on large shards: the process sits at 0% CPU with a
    `.incomplete` blob and never returns, which is indistinguishable from a slow download until
    you check that the cache has not grown in ten minutes. `hf_transfer` has its own failure
    mode — no reliable mid-file resume — so classic HTTP is what actually finishes.

    Set here rather than at each call site because every export script in this directory
    downloads a checkpoint, and remembering an environment variable per invocation is how this
    keeps recurring. An explicit value in the environment always wins, so a caller who wants
    Xet can still have it.
    """
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


_prefer_plain_http_transfers()


if __name__ == "__main__":  # `python3 _paths.py` prints the resolved layout
    for fn in (repo_root, work_root, code_root, exports_dir, smoke_dir, gpu_lock, hf_cache):
        print(f"{fn.__name__:<12} {fn()}")
