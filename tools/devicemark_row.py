#!/usr/bin/env python3
"""Put the DeviceMark row block on every Core AI model card this account owns.

The block is the `gen-cards:devicemark` marker block that scripts/gen-cards/gen_cards.py renders
(one renderer, so the bulk write here and a later gen-cards run agree byte for byte). With a board
row for the repo (cards.json `devicemark.rows`): the row's badge, then the measured decode tok/s
per device from the board's own data files. Without one: a single sentence linking the board, no
numbers. The block goes right after the card's definition sentence (knowledge/card-definition-line.md),
or at the top of the body when a card has none. Nothing else in the card is touched — the front
matter is never re-serialized (text edit, like tools/card_first_line.py).

    python3 tools/devicemark_row.py --list                # regenerate the target list from the HF API
    python3 tools/devicemark_row.py                       # dry run: diff per repo + summary table
    python3 tools/devicemark_row.py --go <repo>           # pilot one repo, verified through the API
    python3 tools/devicemark_row.py --go                  # the rest (repos already right are skipped)
    python3 tools/devicemark_row.py --self-test [--corpus DIR]   # offline: patch logic on synthetic
                                                          # cards (+ every README in DIR)

Guards, checked live at run time, never from a cached list:
  * logged-in user must be `mlboydaisuke`;
  * targets are cut by the API, not by name: author == owner and a `.aimodel` in the file list
    (LiteRT / Core ML / ExecuTorch repos cannot match);
  * a target must live in the owner's namespace AND its oldest commit must be authored by the owner
    (the 2026-09-05 litert-community incident) — anything else is EXCLUDED, even with --go;
  * every write is verified by re-reading the README at the new commit: the block is there once and
    every byte outside it equals the pre-write text. A mismatch is reported as FAILED, loudly.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZOO = HERE.parent
sys.path.insert(0, str(ZOO / "scripts" / "gen-cards"))
import gen_cards  # noqa: E402 — the renderer, the data loader and the marker strings

OWNER = "mlboydaisuke"
TARGETS_FILE = HERE / "devicemark_row_targets.txt"   # explicit ids, one per line, `#` comments allowed
COMMIT_MESSAGE = "Card: DeviceMark row (2026-09)"
DEFINITION_MARKER = "Core AI is "                    # the definition sentence the block follows
OUT = ZOO / "scripts" / "gen-cards" / "out" / "devicemark"   # dry-run diffs (out/ is git-ignored)
DM_BEGIN, DM_END = gen_cards.DM_BEGIN, gen_cards.DM_END


# ---------------------------------------------------------------- text surgery (pure)

def split_front_matter(text: str) -> int:
    """Index of the first body line (0 when there is no front matter)."""
    lines = text.split("\n")
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                return i + 1
    return 0


def first_body_line(text: str) -> str:
    lines = text.split("\n")
    return next((l for l in lines[split_front_matter(text):] if l.strip()), "")


def insert_index(lines: list[str], body: int) -> int:
    """Where the block goes: after the definition paragraph and the blank line(s) that follow
    it, so the definition line stays the first body line (tools/card_first_line.py stays
    idempotent); otherwise before the first body line."""
    j = body
    while j < len(lines) and not lines[j].strip():
        j += 1
    if j < len(lines) and lines[j].startswith(DEFINITION_MARKER):
        k = j
        while k < len(lines) and lines[k].strip():
            k += 1
        while k < len(lines) and not lines[k].strip():
            k += 1
        return k
    return j


def patch(text: str, block: str) -> tuple[str, str]:
    """Return (new_text, status); status in {'insert', 'replace', 'present', 'error: …'}."""
    lines = text.split("\n")
    if DM_BEGIN in lines:
        b = lines.index(DM_BEGIN)
        try:
            e = lines.index(DM_END, b)
        except ValueError:
            return text, "error: begin marker without end marker"
        if "\n".join(lines[b:e + 1]) == block:
            return text, "present"
        return "\n".join(lines[:b] + block.split("\n") + lines[e + 1:]), "replace"
    if "gen-cards:devicemark" in text:
        return text, "error: a devicemark marker variant this tool does not know — fix by hand"
    k = insert_index(lines, split_front_matter(text))
    return "\n".join(lines[:k] + block.split("\n") + [""] + lines[k:]), "insert"


def strip_block(text: str) -> str:
    """The card without its block (and without the blank line the insert adds after it) —
    what must be byte-identical before and after a write."""
    lines = text.split("\n")
    if DM_BEGIN not in lines:
        return text
    b = lines.index(DM_BEGIN)
    e = lines.index(DM_END, b)
    rest = lines[:b] + lines[e + 1:]
    if b < len(rest) and not rest[b].strip():
        del rest[b]
    return "\n".join(rest)


# ---------------------------------------------------------------- hub access (429-aware)

def retry(fn, tries: int = 6):
    from huggingface_hub.utils import HfHubHTTPError
    for i in range(tries):
        try:
            return fn()
        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                time.sleep(10 * (i + 1))
                continue
            raise
    raise RuntimeError("429 after retries")


def read_readme(repo: str, revision: str) -> str:
    from huggingface_hub import hf_hub_download
    p = retry(lambda: hf_hub_download(repo, "README.md", revision=revision, force_download=True))
    return Path(p).read_text(encoding="utf-8")


def first_commit_author(api, repo: str) -> tuple[str, str]:
    commits = retry(lambda: api.list_repo_commits(repo))
    oldest = commits[-1]
    return (",".join(oldest.authors) if oldest.authors else "?"), str(oldest.created_at)[:10]


def list_targets(api, me: str) -> list[str]:
    ids = []
    for m in api.list_models(author=me, limit=1000):
        info = retry(lambda: api.model_info(m.id))
        if any(".aimodel" in s.rfilename for s in info.siblings or []):
            assert m.id.split("/")[0] == me, m.id
            ids.append(m.id)
    return sorted(ids)


# ---------------------------------------------------------------- self-test (offline)

def self_test(corpus: Path | None) -> int:
    dm = {"board": "https://devicemark.github.io/",
          "dataset": "https://huggingface.co/datasets/devicemark/results"}
    row = {"slug": "qwen3.5-2b", "quant": "int8hu", "model": "Qwen3.5-2B",
           "devices": {"iPhone 17 Pro": 29, "M4 Max": 165}}
    blocks = [gen_cards.render_devicemark_block(row, dm), gen_cards.render_devicemark_block(None, dm)]
    assert "29 tok/s" in blocks[0] and "165 tok/s" in blocks[0] and "badge/qwen3.5-2b.svg" in blocks[0]
    assert "tok/s" not in blocks[1] and dm["board"] in blocks[1]
    one_dev = dict(row, devices={"iPhone 17 Pro": 23.3})
    assert "Mac" not in gen_cards.devicemark_line(one_dev, dm)      # unmeasured device: not mentioned
    docs = {
        "fm+def": "---\na: 1\n---\n\nCore AI is X.\nmore.\n\n# T\n\nbody\n",
        "fm+def-tight": "---\na: 1\n---\nCore AI is X.\n\n# T\n",
        "fm-nodef": "---\na: 1\n---\n\n# T\n\nbody\n",
        "nofm": "# T\n\nbody\n",
        "def-only": "Core AI is X.\n",
        "def-eof": "---\na: 1\n---\n\nCore AI is X.",
    }
    if corpus:
        for p in sorted(corpus.glob("*.md")):
            docs[p.name] = p.read_text(encoding="utf-8")
    n = 0
    for name, doc in docs.items():
        for block in blocks:
            other = blocks[1] if block is blocks[0] else blocks[0]
            new, st = patch(doc, block)
            assert st == "insert", (name, st)
            assert new.count(DM_BEGIN) == 1 and new.count(DM_END) == 1, name
            assert strip_block(new) == doc, name                       # outside bytes untouched
            assert first_body_line(new) == first_body_line(doc) or not first_body_line(doc).startswith(
                DEFINITION_MARKER), name                                # definition line stays first
            assert block in new, name
            again, st2 = patch(new, block)
            assert st2 == "present" and again == new, (name, st2)     # idempotent
            rep, st3 = patch(new, other)
            assert st3 == "replace" and strip_block(rep) == doc and other in rep, (name, st3)
            n += 1
    broken = docs["fm+def"].replace("# T", DM_BEGIN + "\n# T")          # begin without end
    assert patch(broken, blocks[0])[1].startswith("error"), "unterminated block must not be rewritten"
    print(f"self-test OK: {n} patch cases over {len(docs)} cards")
    return 0


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="write the target list from the HF API and exit")
    ap.add_argument("--go", action="store_true", help="write the commits (default: dry run)")
    ap.add_argument("--no-diff", action="store_true", help="dry run without printing the per-repo diffs")
    ap.add_argument("--targets", type=Path, default=TARGETS_FILE)
    ap.add_argument("--devicemark", default=gen_cards.DEVICEMARK_DEFAULT,
                    help="DeviceMark data dir (board.json + measurements.jsonl)")
    ap.add_argument("--message", default=COMMIT_MESSAGE, help="commit message")
    ap.add_argument("--self-test", action="store_true", help="offline check of the patch logic; no hub access")
    ap.add_argument("--corpus", type=Path, help="--self-test: also run every *.md in this directory")
    ap.add_argument("repos", nargs="*", help="restrict to these repo ids (default: the whole target list)")
    args = ap.parse_args()

    if args.self_test:
        return self_test(args.corpus)

    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()
    who = api.whoami()["name"]
    if who != OWNER:
        sys.exit(f"logged in as {who}, expected {OWNER}")

    if args.list:
        ids = list_targets(api, who)
        args.targets.write_text(
            f"# Targets of tools/devicemark_row.py — HF models by {who} with a .aimodel in the file list "
            f"(HF API, {time.strftime('%Y-%m-%d')}).\n"
            "# Ownership is re-checked live at run time; a repo not created by the owner is excluded there.\n"
            + "\n".join(ids) + "\n")
        print(f"{len(ids)} targets written to {args.targets}")
        return 0

    dm_dir = Path(args.devicemark).expanduser()
    top = json.loads((ZOO / "scripts" / "gen-cards" / "cards.json").read_text())
    dm_cfg = top["devicemark"]
    rows, board = gen_cards.load_devicemark(dm_dir)
    gen_cards.check_devicemark_map(dm_cfg, rows, board)
    if gen_cards.failures:
        sys.exit("DeviceMark data or cards.json mapping failed its checks — nothing written")

    targets = [t.strip() for t in args.targets.read_text().splitlines() if t.strip() and not t.startswith("#")]
    if args.repos:
        unknown = [r for r in args.repos if r not in targets]
        if unknown:
            sys.exit(f"not in the target list (run --list first): {unknown}")
        targets = [t for t in targets if t in args.repos]
    OUT.mkdir(parents=True, exist_ok=True)

    table = []
    for repo in targets:
        ns = repo.split("/")[0]
        try:
            author, created = first_commit_author(api, repo)
        except Exception as e:  # noqa: BLE001
            table.append((repo, "?", "?", "error", f"commits: {e}")); continue
        if ns != who or author != OWNER:
            table.append((repo, author, created, "EXCLUDED",
                          "not created by owner" if ns == who else "foreign namespace")); continue
        try:
            sha = retry(lambda: api.model_info(repo)).sha
            src = read_readme(repo, sha)
        except Exception as e:  # noqa: BLE001
            table.append((repo, author, created, "error", f"readme: {e}")); continue
        row = rows.get(dm_cfg["rows"].get(repo))
        block = gen_cards.render_devicemark_block(row, dm_cfg)
        new, status = patch(src, block)
        note = f"row `{row['slug']}`" if row else "no row (board link only)"
        if status in ("insert", "replace"):
            diff = "".join(difflib.unified_diff(src.splitlines(True), new.splitlines(True),
                                                f"{repo}/README.md@{sha[:8]}", f"{repo}/README.md (new)", n=1))
            (OUT / (repo.split("/")[1] + ".diff")).write_text(diff)
            if not args.go and not args.no_diff:
                print(diff)
            if args.go:
                try:
                    c = retry(lambda: api.create_commit(
                        repo_id=repo, operations=[CommitOperationAdd("README.md", new.encode("utf-8"))],
                        commit_message=args.message, parent_commit=sha))
                    got = read_readme(repo, c.oid)
                    if got == new and strip_block(got) == strip_block(src) and got.count(DM_BEGIN) == 1:
                        status = f"{status}: committed {c.oid[:8]} verified"
                    else:
                        status = f"FAILED verify after {c.oid[:8]}: re-read differs from the intended text"
                except Exception as e:  # noqa: BLE001
                    status = f"FAILED: {e}"
                time.sleep(1.0)
        table.append((repo, author, created, status, note))

    out = ["| repo | first commit by | created | status | block |", "|---|---|---|---|---|"]
    out += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4].replace('|', '\\|')} |" for r in table]
    print("\n".join(out))
    print("\nsummary:", dict(Counter(r[3].split(":")[0] for r in table)),
          "| mode:", "WRITE" if args.go else "dry run", f"| diffs: {OUT}")
    return 1 if any(r[3].startswith(("FAILED", "error")) for r in table) else 0


if __name__ == "__main__":
    sys.exit(main())
