#!/usr/bin/env python3
"""Put one definition sentence at the top of every Core AI model card this project owns.

The sentence goes right after the YAML front matter (or at line 1 when a card has none), as
its own paragraph. Nothing else in the card is touched. Default is a dry run that prints a
unified diff per repo and a summary table; `--go` writes, one commit per repo.

    python3 tools/card_first_line.py --list                    # regenerate the target list from the HF API
    python3 tools/card_first_line.py                           # dry run against tools/card_first_line_targets.txt
    python3 tools/card_first_line.py --go                      # write (after the owner's GO)
    python3 tools/card_first_line.py --report path.md          # also save the summary table

Guards, all checked live at run time, never from a cached list:
  * logged-in user must be `mlboydaisuke`;
  * a target must live in one of OWN_NAMESPACES;
  * a target's oldest commit must be authored by `mlboydaisuke` (ownership = created by the
    user, not "has a commit" — see the 2026-09-05 litert-community incident). Anything else is
    listed under EXCLUDED and never written, even with --go.
  * a card whose first body line already equals the sentence is skipped (idempotent); a card
    whose first body line starts with SENTENCE_MARKER but differs is reported as `stale` and
    only rewritten with --replace.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.utils import HfHubHTTPError

HERE = Path(__file__).resolve().parent
LINE_FILE = HERE / "card_first_line.txt"          # the one sentence, one line
TARGETS_FILE = HERE / "card_first_line_targets.txt"  # explicit ids, one per line, `#` comments allowed
OWNER = "mlboydaisuke"
OWN_NAMESPACES = ("mlboydaisuke", "coreai-community")
SEARCH = "coreai"
SENTENCE_MARKER = "Core AI is "                    # how a previous version of the sentence is recognised
COMMIT_MESSAGE = "Card: definition line (Core AI, 2026-09)"


def http_json(url: str, tries: int = 6):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "card-first-line"}), timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(10 * (i + 1))
                continue
            raise
    raise RuntimeError(f"gave up on {url}")


def http_text(url: str, tries: int = 6) -> str:
    """Fetch a text file; on 429 wait and retry like http_json does."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "card-first-line"}), timeout=60) as r:
                return r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(10 * (i + 1))
                continue
            raise
    raise RuntimeError(f"gave up on {url}")


def list_candidates() -> list[str]:
    ids: list[str] = []
    for ns in OWN_NAMESPACES:
        q = urllib.parse.urlencode({"search": SEARCH, "author": ns, "limit": 1000})
        ids += [m["id"] for m in http_json(f"https://huggingface.co/api/models?{q}")]
    return ids


def first_commit_author(api: HfApi, repo: str) -> tuple[str, str]:
    """(author of the oldest commit, its date). Retries on 429."""
    for i in range(6):
        try:
            commits = api.list_repo_commits(repo)
            oldest = commits[-1]
            return (",".join(oldest.authors) if oldest.authors else "?"), str(oldest.created_at)[:10]
        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                time.sleep(10 * (i + 1))
                continue
            raise
    raise RuntimeError(f"gave up on commits of {repo}")


def split_front_matter(text: str) -> int:
    """Index of the first body line (0 when there is no front matter)."""
    lines = text.split("\n")
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                return i + 1
    return 0


def patch(text: str, line: str, replace: bool) -> tuple[str, str]:
    """Return (new_text, status). status in {'insert', 'present', 'stale', 'replace'}."""
    lines = text.split("\n")
    body = split_front_matter(text)
    j = body
    while j < len(lines) and not lines[j].strip():
        j += 1
    first = lines[j] if j < len(lines) else ""
    if first.strip() == line.strip():
        return text, "present"
    if first.startswith(SENTENCE_MARKER):
        if not replace:
            return text, "stale"
        k = j + 1
        while k < len(lines) and not lines[k].strip():
            k += 1
        new = lines[:body] + [""] * (1 if body else 0) + [line, ""] + lines[k:]
        return "\n".join(new), "replace"
    new = lines[:body] + [""] * (1 if body else 0) + [line, ""] + lines[j:]
    return "\n".join(new), "insert"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="write the target list from the HF API and exit")
    ap.add_argument("--go", action="store_true", help="write the commits (default: dry run)")
    ap.add_argument("--replace", action="store_true", help="rewrite a stale definition line")
    ap.add_argument("--report", type=Path, help="save the summary table (markdown) here")
    ap.add_argument("--no-diff", action="store_true", help="dry run without the per-repo diffs")
    ap.add_argument("--line-file", type=Path, default=LINE_FILE)
    ap.add_argument("--targets", type=Path, default=TARGETS_FILE)
    args = ap.parse_args()

    api = HfApi()
    who = api.whoami()["name"]
    if who != OWNER:
        sys.exit(f"logged in as {who}, expected {OWNER}")

    if args.list:
        ids = list_candidates()
        args.targets.write_text(
            "# Targets of tools/card_first_line.py — HF models whose name contains "
            f"'{SEARCH}' in {', '.join(OWN_NAMESPACES)} (HF API, {time.strftime('%Y-%m-%d')}).\n"
            "# Ownership is re-checked live at run time; a repo not created by the owner is excluded there.\n"
            + "\n".join(ids) + "\n")
        print(f"{len(ids)} candidates written to {args.targets}")
        return 0

    line = args.line_file.read_text(encoding="utf-8").strip()
    if not line or "\n" in line:
        sys.exit(f"{args.line_file} must hold exactly one non-empty line")
    targets = [t.strip() for t in args.targets.read_text().splitlines() if t.strip() and not t.startswith("#")]

    rows = []
    for repo in targets:
        ns = repo.split("/")[0]
        try:
            author, created = first_commit_author(api, repo)
        except Exception as e:  # noqa: BLE001
            rows.append((repo, "?", "?", "error", f"commits: {e}", "")); continue
        if ns not in OWN_NAMESPACES or author != OWNER:
            rows.append((repo, author, created, "EXCLUDED", "not created by owner" if ns in OWN_NAMESPACES else "foreign namespace", "")); continue
        try:
            info = api.model_info(repo)
            sha = info.sha
            src = http_text(f"https://huggingface.co/{repo}/raw/{sha}/README.md")
        except Exception as e:  # noqa: BLE001
            rows.append((repo, author, created, "error", f"readme: {e}", "")); continue
        new, status = patch(src, line, args.replace)
        body = split_front_matter(src)
        cur = next((l for l in src.split("\n")[body:] if l.strip()), "")
        if status in ("insert", "replace"):
            if not args.no_diff and not args.go:
                sys.stdout.writelines(difflib.unified_diff(src.splitlines(True), new.splitlines(True),
                                                           fromfile=f"{repo}/README.md@{sha[:8]}", tofile=f"{repo}/README.md (new)", n=1))
                print()
            if args.go:
                for i in range(6):
                    try:
                        api.create_commit(repo_id=repo, operations=[CommitOperationAdd("README.md", new.encode("utf-8"))],
                                          commit_message=COMMIT_MESSAGE, parent_commit=sha)
                        status = f"{status}: committed"
                        break
                    except HfHubHTTPError as e:
                        if e.response is not None and e.response.status_code == 429:
                            time.sleep(15 * (i + 1)); continue
                        status = f"FAILED: {e}"
                        break
                else:
                    status = "FAILED: 429 after retries"
                time.sleep(1.0)
        rows.append((repo, author, created, status, cur[:90], line if status.startswith(("insert", "replace")) else ""))

    table = ["| repo | first commit by | created | status | first body line before |", "|---|---|---|---|---|"]
    table += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4].replace('|', '\\|')} |" for r in rows]
    out = "\n".join(table)
    print(out)
    from collections import Counter
    print("\nsummary:", dict(Counter(r[3].split(":")[0] for r in rows)), "| mode:", "WRITE" if args.go else "dry run")
    if args.report:
        args.report.write_text(f"# card_first_line report — {time.strftime('%Y-%m-%d %H:%M')} ({'write' if args.go else 'dry run'})\n\nSentence:\n\n> {line}\n\n{out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
