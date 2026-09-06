#!/usr/bin/env python3
"""Generate llms.txt — the one fetch that tells a model what this repository knows.

Why it matters here specifically: Apple's documentation answers "what is the API." Its plain
URLs return no body to a fetcher — the content is reachable, but only through the
`developer.apple.com/tutorials/data/documentation/<path>.json` backing endpoint, which a
reader has to already know about. What no source answers is "what does it do when you run
it": thresholds, failure modes, measured numbers. The notes in `knowledge/` are that, and
they are useless if nothing announces them.

Follows the llms.txt convention: an H1, a blockquote summary, then link sections where every
entry is `[title](url): description`. Descriptions are lifted from `knowledge/README.md`, so
the index and the notes cannot disagree — this file is generated, never hand-edited.

    python3 scripts/gen_llms_txt.py           # regenerate
    python3 scripts/gen_llms_txt.py --check   # CI: fail if stale
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KNOWLEDGE = REPO / "knowledge" / "README.md"
OUT = REPO / "llms.txt"
SITE = "https://john-rocky.github.io/coreai-model-zoo"
REPO_URL = "https://github.com/john-rocky/coreai-model-zoo"

# `- [`file.md`](file.md) — description, possibly wrapped over the following indented lines.`
ENTRY = re.compile(r"^- \[`([^`]+)`\]\(([^)]+)\)\s*[—-]\s*(.*)$")
SECTION = re.compile(r"^## +(.*)$")

PREAMBLE = f"""\
# Core AI model zoo

> Community ports of open models to Apple's Core AI runtime (`.aimodel`, iOS/macOS 27), each
> with the recipe that produced it, plus a knowledge base of verified findings about the
> runtime itself. Apple documents the API surface; these notes cover what it does when you run
> it — thresholds, failure modes, and measured numbers — and are specific about what was
> verified, on what hardware, and how.

## Start here

- [AGENTS.md]({SITE}/AGENTS.html): the porting contract in one file — why conversion is not format
  conversion, the two gates every port gets, the traps agents specifically hit, and which
  actions stay a human's call.
- [README.md]({SITE}/): the catalog itself — every model, its card, its Hugging Face
  repo, and the one-line Swift call that runs it.
- [models/index.json]({SITE}/models/index.json): the same catalog machine-readable. Per recipe:
  `status` (is the configuration recorded), `source_model` (what it was converted from), and
  `gate_transcript` (is the numerical check against the original published, and where).
- [PORTING.md]({SITE}/PORTING.html): the full walk from a Hugging Face checkpoint to a verified
  bundle on an iPhone, with two worked examples.
- [knowledge/coreai-error-index.md]({SITE}/knowledge/coreai-error-index.html): hit an error
  string? Every exact Core AI / coreai-torch / coreai-build / Swift-engine error this project has
  observed, verbatim as a heading, with when it appears, the verified cause (or "Not isolated"),
  the fix, the log or Apple issue behind it, and the OS / toolchain. Search the string; land on
  the section.
- [SECURITY.md]({SITE}/SECURITY.html): what the integrity story is, including the parts that are
  absent — pinned revisions rather than signatures, and no checksum manifest.
- [Source repository]({REPO_URL}): the conversion scripts, the gates, and the recipes behind
  every page here. The site renders the same files; nothing is written for it separately.
- [The Art of Core AI](https://john-rocky.github.io/the-art-of-core-ai/): the long-form version
  — a free book built from these same measurements, for reading start to finish rather than
  looking one thing up.
"""

FOOTER = f"""
## Optional

- [CONTRIBUTING.md]({SITE}/CONTRIBUTING.html): what an accepted port must clear, and the device
  gate — the one step a contributor without an iOS 27 device can hand back.
- [BENCHMARKS.md]({SITE}/BENCHMARKS.html): community-submitted device measurements, explicitly not
  a controlled-environment benchmark.
"""


def parse_sections() -> list[tuple[str, list[tuple[str, str, str]]]]:
    """`knowledge/README.md` -> [(section title, [(file, url, description)])]."""
    sections: list[tuple[str, list[tuple[str, str, str]]]] = []
    current: list[tuple[str, str, str]] = []
    title = "Knowledge base"
    pending: list[str] | None = None
    entry: tuple[str, str] | None = None

    def flush() -> None:
        nonlocal pending, entry
        if entry and pending is not None:
            desc = " ".join(" ".join(pending).split())
            current.append((entry[0], entry[1], desc.rstrip(".")))
        pending, entry = None, None

    for line in KNOWLEDGE.read_text().splitlines():
        if m := SECTION.match(line):
            flush()
            if current:
                sections.append((title, current.copy()))
                current.clear()
            title = m.group(1).strip()
        elif m := ENTRY.match(line):
            flush()
            entry, pending = (m.group(1), m.group(2)), [m.group(3)]
        elif pending is not None and line.startswith("  ") and line.strip():
            pending.append(line.strip())
        elif not line.strip():
            flush()
    flush()
    if current:
        sections.append((title, current))
    return sections


def heading(path: Path) -> str:
    """A note's own H1, used when the curated index does not describe it."""
    for line in path.read_text().splitlines():
        if line.startswith("# "):
            return re.sub(r"\*\*|`", "", line[2:].strip())
    return path.stem.replace("-", " ")


def uncurated(covered: set[str]) -> list[tuple[str, str]]:
    """Notes on disk that `knowledge/README.md` does not list, newest content last.

    They are published either way — they live in `knowledge/`, so the site renders them — and
    being absent from the index only made them unfindable. Including them automatically means
    adding a note costs nothing: write it, regenerate, and it is announced. The curated
    sections still win where a description was written by hand.
    """
    out = []
    for path in sorted((REPO / "knowledge").glob("*.md")):
        if path.name == "README.md" or path.name in covered:
            continue
        out.append((path.name, heading(path)))
    return out


def render() -> str:
    out = [PREAMBLE]
    covered: set[str] = set()
    for title, entries in parse_sections():
        if not entries:
            continue
        out.append(f"\n## {title}\n")
        for name, url, desc in entries:
            covered.add(url)
            # Strip the emphasis markers that read as noise once flattened to one line.
            clean = re.sub(r"\*\*|`", "", desc)
            page = url.removesuffix(".md") + ".html"
            out.append(f"- [{name}]({SITE}/knowledge/{page}): {clean}")
        out.append("")

    if rest := uncurated(covered):
        out.append("\n## Everything else in the knowledge base\n")
        for name, title in rest:
            page = name.removesuffix(".md") + ".html"
            out.append(f"- [{name}]({SITE}/knowledge/{page}): {title}")
        out.append("")

    out.append(FOOTER)
    return "\n".join(out).replace("\n\n\n", "\n\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit non-zero if llms.txt is stale")
    args = ap.parse_args()

    want = render()
    if args.check:
        have = OUT.read_text() if OUT.exists() else ""
        if have != want:
            sys.exit("llms.txt is stale — a note was added, or knowledge/README.md changed,\n"
                     "  without regenerating the index. An unannounced note is one no agent\n"
                     "  will find. Fix with: python3 scripts/gen_llms_txt.py")
        print(f"OK: llms.txt matches knowledge/README.md ({want.count('](') } links)")
        return
    OUT.write_text(want)
    print(f"wrote llms.txt ({want.count('](')} links, {len(want)} bytes)")


if __name__ == "__main__":
    main()
