#!/usr/bin/env python3
"""Upload a model card to a Hugging Face repo and keep the repo's SHA256SUMS in step, in one commit.

    python3 scripts/hf_card_push.py check <repo>                          # SHA256SUMS vs the repo's files
    python3 scripts/hf_card_push.py push <repo> <card.md> -m "<message>"   # dry run: the check and the plan
    python3 scripts/hf_card_push.py push <repo> <card.md> -m "<message>" --yes [--parent <commit id>]

Some repos carry a SHA256SUMS that lists README.md (the staging scripts of conversion/laya, conversion/julia
and conversion/gliner25_decide write one). A card uploaded alone leaves that line stale, and nothing on the Hub
says so. `push` uploads the card as README.md and, when the repo has a SHA256SUMS that lists README.md, the same
SHA256SUMS with only that line's sha256 replaced, as one commit whose parent is the revision it checked: a commit
pushed meanwhile makes this one fail instead of being overwritten. A repo without SHA256SUMS gets README.md alone.
Pushing the card the repo already has, when its line is stale (a card edited on the Hub, a merged pull request),
commits the corrected SHA256SUMS alone. `--parent` names the revision the card was edited from: when the repo's
head is another commit, the push stops, because the card would undo what that commit changed in README.md.

The check (`check`, and `push` before and after its commit): every SHA256SUMS line names a file of the repo with
the sha256 the line gives (LFS files: the Hub's lfs.sha256; other files: downloaded and hashed), and every file of
the repo is listed except SHA256SUMS itself and .gitattributes. Before the commit, the README.md line is left out
of the comparison (the commit replaces it) and a stale one is reported; any other difference stops the push. After
the commit the whole file is checked again at the new revision. Exit 0 = clean, 1 = a difference or a failed
upload, 2 = usage.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # plain HTTP: Xet stalls on some transfers (conversion/_paths.py)

SUMS = "SHA256SUMS"
CARD = "README.md"
NOT_LISTED = {SUMS, ".gitattributes"}
LINE = re.compile(r"([0-9a-f]{64})  (.+)")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_sums(text: str) -> dict[str, str]:
    """`<sha256>  <path>` per line, the format `sha256sum` writes; anything else is an error."""
    listed = {}
    for number, line in enumerate(text.splitlines(), 1):
        match = LINE.fullmatch(line)
        if not match:
            raise ValueError(f"{SUMS} line {number} is not '<sha256>  <path>': {line[:80]!r}")
        listed[match.group(2)] = match.group(1)
    return listed


def read_repo(api, repo: str, revision: str | None):
    """The revision's commit id, its files (path -> sibling) and the text of SHA256SUMS (None when absent)."""
    from huggingface_hub import hf_hub_download

    info = api.model_info(repo, revision=revision, files_metadata=True)
    files = {s.rfilename: s for s in info.siblings}
    text = None
    if SUMS in files:
        with tempfile.TemporaryDirectory() as cache:
            text = Path(hf_hub_download(repo, SUMS, revision=info.sha, cache_dir=cache)).read_text()
    return info.sha, files, text


def actual_sha256(repo: str, revision: str, sibling, cache: str) -> str:
    from huggingface_hub import hf_hub_download

    if sibling.lfs is not None:
        return sibling.lfs.sha256
    path = hf_hub_download(repo, sibling.rfilename, revision=revision, cache_dir=cache)
    return sha256_bytes(Path(path).read_bytes())


def differences(repo: str, revision: str, files: dict, listed: dict[str, str], skip: set[str] = frozenset()) -> list[str]:
    """Every way SHA256SUMS and the repo's files disagree, leaving out the paths in `skip`."""
    found = []
    with tempfile.TemporaryDirectory() as cache:
        for path, digest in sorted(listed.items()):
            if path in skip:
                continue
            if path not in files:
                found.append(f"listed, not in the repo: {path}")
            elif actual_sha256(repo, revision, files[path], cache) != digest:
                found.append(f"sha256 differs: {path} (listed {digest[:12]}...)")
    for path in sorted(set(files) - set(listed) - NOT_LISTED - set(skip)):
        found.append(f"in the repo, not listed: {path}")
    return found


def check(api, repo: str, revision: str | None = None) -> int:
    sha, files, text = read_repo(api, repo, revision)
    if text is None:
        print(f"{repo}@{sha[:12]}: no {SUMS}")
        return 0
    found = differences(repo, sha, files, parse_sums(text))
    print(f"{repo}@{sha[:12]}: {SUMS} lists {len(parse_sums(text))} files; "
          + ("matches the repo" if not found else f"{len(found)} difference(s)"))
    for line in found:
        print("  " + line)
    return 1 if found else 0


def push(api, repo: str, card: Path, message: str, yes: bool, parent: str | None = None) -> int:
    from huggingface_hub import CommitOperationAdd, hf_hub_download

    new_card = card.read_bytes()
    new_digest = sha256_bytes(new_card)
    sha, files, text = read_repo(api, repo, None)
    if parent is not None and not sha.startswith(parent.lower()):
        print(f"{repo}: the head is {sha[:12]}, not {parent}: the card was written on an older revision; not pushing")
        return 1
    with tempfile.TemporaryDirectory() as cache:
        old_card = Path(hf_hub_download(repo, CARD, revision=sha, cache_dir=cache)).read_bytes() if CARD in files else None
    operations, plan = [], []
    if old_card != new_card:
        operations.append(CommitOperationAdd(path_in_repo=CARD, path_or_fileobj=new_card))
        plan.append(f"{CARD}: {sha256_bytes(old_card)[:12] if old_card else '(new)'}... -> {new_digest[:12]}...")
    if text is not None:
        listed = parse_sums(text)
        found = differences(repo, sha, files, listed, skip={CARD})
        if found:
            print(f"{repo}@{sha[:12]}: {SUMS} and the repo differ apart from {CARD}; not pushing:")
            for line in found:
                print("  " + line)
            return 1
        if CARD not in listed:
            plan.append(f"{SUMS} does not list {CARD}: left as it is")
        elif listed[CARD] != new_digest:
            if old_card is not None and listed[CARD] != sha256_bytes(old_card):
                print(f"note: the {CARD} line of {SUMS} did not match the card on the Hub before this commit")
            lines = text.splitlines(keepends=True)
            hits = [i for i, line in enumerate(lines) if LINE.fullmatch(line.rstrip("\r\n")).group(2) == CARD]
            assert len(hits) == 1, hits
            lines[hits[0]] = f"{new_digest}  {CARD}\n"
            operations.append(CommitOperationAdd(path_in_repo=SUMS, path_or_fileobj="".join(lines).encode()))
            plan.append(f"{SUMS}: the {CARD} line {listed[CARD][:12]}... -> {new_digest[:12]}..., the other "
                        f"{len(listed) - 1} lines unchanged")
    if not operations:
        print(f"{repo}@{sha[:12]}: {CARD} is already this card" + (f" and {SUMS} lists it" if text else "")
              + "; nothing to push")
        return 0
    print(f"{repo}@{sha[:12]}: one commit, parent {sha[:12]}, message {message!r}")
    for line in plan:
        print("  " + line)
    if not yes:
        print("dry run: add --yes to push")
        return 0
    commit = api.create_commit(repo, operations=operations, commit_message=message, parent_commit=sha)
    new_sha = commit.oid
    print(f"pushed: {commit.commit_url}")
    if text is None:
        return 0
    status = check(api, repo, new_sha)
    if status:
        print(f"!! {SUMS} and the repo differ at the new revision {new_sha[:12]}: fix before anything else")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("check", help=f"compare {SUMS} with the repo's files")
    one.add_argument("repo")
    one.add_argument("--revision")
    two = sub.add_parser("push", help=f"upload a card (and the {SUMS} line) in one commit")
    two.add_argument("repo")
    two.add_argument("card", type=Path)
    two.add_argument("-m", "--message", required=True)
    two.add_argument("--yes", action="store_true", help="push; without it, a dry run")
    two.add_argument("--parent", help="the commit id (full or a prefix) the card was written on; stop unless it is "
                                      "the repo's head")
    args = parser.parse_args(argv)
    if args.command == "push" and args.parent is not None and not re.fullmatch(r"[0-9a-fA-F]{7,40}", args.parent):
        parser.error("--parent takes a commit id: 7 to 40 hex digits")

    from huggingface_hub import HfApi

    api = HfApi()
    if args.command == "check":
        return check(api, args.repo, args.revision)
    return push(api, args.repo, args.card, args.message, args.yes, args.parent)


if __name__ == "__main__":
    sys.exit(main())
