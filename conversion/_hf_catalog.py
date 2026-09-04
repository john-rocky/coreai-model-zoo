#!/usr/bin/env python3
"""Read the zoo's published catalog off Hugging Face without downloading weights.

Shared by `scripts/gen_inventory.py` (what exists) and `conversion/zoo_verify.py`
(whether it is right). Everything here touches only the listing API and small
JSON/text files, so a full pass over the catalog costs seconds and no disk.

The unit is a **bundle**, not a repo: one directory holding a `metadata.json`
beside an `.aimodel` (or an AOT-compiled `.aimodelc`), which is exactly what the
runtime loads. A repo usually publishes several — variants, precisions, an iOS
AOT twin — and which one *shipped* is the question the catalog has to answer.

Two on-disk layouts exist in the wild and both are handled:

    <repo>/gpu-pipelined/<name>/metadata.json          + <name>.aimodel/    (wrapped)
    <repo>/<name>.aimodel/metadata.json                                     (flat)
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://huggingface.co/api/models"
RESOLVE = "https://huggingface.co/{repo}/resolve/{rev}/{path}"
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "hf"
USER_AGENT = "coreai-model-zoo-catalog/1.0"


class Catalog:
    """Listing + small-file reads, memoised on disk so reruns are free."""

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE, offline: bool = False):
        self.cache = Path(cache_dir)
        self.offline = offline
        self.misses: set[str] = set()

    # ---- transport ------------------------------------------------------
    def _cache_path(self, key: str) -> Path:
        return self.cache / urllib.parse.quote(key, safe="")

    def _get(self, url: str, key: str, retries: int = 3) -> str | None:
        """Fetch `url` as text; None when the file does not exist (404/403)."""
        path = self._cache_path(key)
        if path.exists():
            text = path.read_text()
            return None if text == "\0missing" else text
        if self.offline:
            self.misses.add(key)
            return None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=60) as r:
                    text = r.read().decode("utf-8", "replace")
                break
            except urllib.error.HTTPError as e:
                if e.code in (401, 403, 404):
                    text = "\0missing"
                    break
                if attempt == retries - 1:
                    raise
                time.sleep(1.5 * (attempt + 1))
            except (urllib.error.URLError, TimeoutError):
                if attempt == retries - 1:
                    raise
                time.sleep(1.5 * (attempt + 1))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return None if text == "\0missing" else text

    # ---- listing --------------------------------------------------------
    def repos_by_author(self, author: str) -> list[dict]:
        text = self._get(f"{API}?author={author}&limit=1000&full=true", f"author:{author}")
        return json.loads(text) if text else []

    def repo(self, repo_id: str) -> dict | None:
        text = self._get(f"{API}/{repo_id}", f"repo:{repo_id}")
        return json.loads(text) if text else None

    def repo_blobs(self, repo_id: str) -> dict | None:
        """The listing with per-file sizes (`?blobs=true`) — what a download plan needs."""
        text = self._get(f"{API}/{repo_id}?blobs=true", f"repo-blobs:{repo_id}")
        return json.loads(text) if text else None

    # ---- files ----------------------------------------------------------
    def file(self, repo_id: str, path: str, rev: str = "main") -> str | None:
        url = RESOLVE.format(repo=repo_id, rev=rev, path=urllib.parse.quote(path))
        return self._get(url, f"file:{repo_id}@{rev}/{path}")

    def json_file(self, repo_id: str, path: str, rev: str = "main") -> dict | None:
        text = self.file(repo_id, path, rev)
        if text is None:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None


def bundles_of(files: list[str]) -> list[str]:
    """Bundle directories in a repo file list, in publication order."""
    assets = {
        f.rsplit("/", 1)[0]
        for f in files
        if ".aimodel/" in f or ".aimodelc/" in f
    }
    assets = {a for a in assets if a.endswith((".aimodel", ".aimodelc"))}
    has_meta = {f.rsplit("/", 1)[0] for f in files if f.endswith("metadata.json")}
    out = set()
    for asset in assets:
        parent = asset.rsplit("/", 1)[0] if "/" in asset else ""
        out.add(parent if parent in has_meta else asset)
    return sorted(out)


def repo_format(files: list[str]) -> str:
    """coreai / coreml / litert / other, from what the repo actually contains."""
    if any(".aimodel" in f for f in files):
        return "coreai"
    if any(".mlmodelc/" in f or ".mlpackage/" in f for f in files):
        return "coreml"
    if any(f.endswith((".litertlm", ".tflite", ".task")) for f in files):
        return "litert"
    return "other"


def bundle_paths(bundle: str, files: list[str]) -> dict[str, str | None]:
    """Where a bundle keeps the files tier-1 verification reads."""
    inside = [f for f in files if f.startswith(bundle + "/")] if bundle else files

    def find(*names: str) -> str | None:
        for name in names:
            for f in inside:
                if f.endswith("/" + name) or f == name:
                    return f
        return None

    return {
        "metadata": f"{bundle}/metadata.json" if f"{bundle}/metadata.json" in files else find("metadata.json"),
        "tokenizer_config": find("tokenizer/tokenizer_config.json", "tokenizer_config.json"),
        "chat_template": find("tokenizer/chat_template.jinja", "chat_template.jinja"),
        "generation_config": find("tokenizer/generation_config.json", "generation_config.json"),
    }
