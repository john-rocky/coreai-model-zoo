#!/usr/bin/env python3
"""Aggregate community bench blobs into BENCHMARKS.md (device x model median table).

Sources
  * GitHub issues labeled `bench-result` on john-rocky/coreai-model-zoo (default) —
    the public audit log. Each issue body carries one JSON blob produced by the
    Bench tab of the CoreAI Zoo app (App Store, 2.0+; protocol zoo-chat-v1) or of
    the July 2026 CoreAIChat TestFlight build (protocol pb-random-v1). See
    .github/ISSUE_TEMPLATE/bench-result.yml.
  * --local DIR: *.json blob files (the on-device DoD loop / offline testing).

Trust model (mirrors the app side): the harness measured, the submitter only pasted.
Blobs are validated hard — schema_version, kind, and an EXACT protocol match. The
blob's protocol.name selects one of PROTOCOLS; every other key of that protocol
block must match verbatim (for zoo-chat-v1 that includes the prompt string, copied
from coreai-zoo-app Sources/Bench/BenchRunner.swift). Anything else is rejected
loudly, never silently dropped.

The two protocols are NOT byte-comparable (zoo-chat-v1 prefills a fixed text prompt
through the kit ChatSession; pb-random-v1 fed 128 random token ids to a forked
engine), so BENCHMARKS.md carries one table per protocol.

results.load_s is never published: in the zoo app it wraps the catalog download
too, so a first run reports the download time, not the load.

Field-data policy: sloppy environments show up as outliers, so we post-filter on
environment metadata instead of trusting the numbers: Low Power Mode ON or a
serious/critical thermal state BEFORE the run excludes a blob from the medians
(still counted + listed as excluded). Cells are the median across submissions of
each submission's own median warm decode tok/s; n >= 3 shows plain, n < 3 is
marked provisional.

Usage
  python3 scripts/aggregate_bench.py                 # fetch issues, write BENCHMARKS.md
  python3 scripts/aggregate_bench.py --local DIR     # also read local blobs
  python3 scripts/aggregate_bench.py --local-only DIR
  GH_TOKEN=... raises the API rate limit (optional).
"""

import argparse
import json
import os
import re
import statistics
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_DEFAULT = "john-rocky/coreai-model-zoo"
SCHEMA_VERSION = 1
BLOB_KIND = "coreai-community-bench"
# Exact protocol blocks, keyed by protocol.name. Order = table order in BENCHMARKS.md
# (first = the live one). A blob whose block differs in any key is rejected.
ZOO_CHAT_V1_PROMPT = (
    "Explain, in detail, how a transformer language model generates text one token at a "
    "time. Cover attention, the KV cache, and why the decode phase is bound by memory "
    "bandwidth rather than compute. Write several paragraphs."
)
PROTOCOLS = {
    # CoreAI Zoo app 2.0+ (App Store). Sources/Bench/BenchRunner.swift in coreai-zoo-app.
    "zoo-chat-v1": {
        "name": "zoo-chat-v1",
        "prompt": ZOO_CHAT_V1_PROMPT,
        "max_tokens": 256,
        "temperature": 0,
        "chunk_threshold": 1,
        "cold_runs": 1,
        "warm_runs": 3,
        "surface": "kit-chatsession",
    },
    # CoreAIChat TestFlight build, July 2026 (apps/CoreAIChat). No longer produced.
    "pb-random-v1": {
        "name": "pb-random-v1",
        "prompt_tokens": 128,
        "max_tokens": 256,
        "prompt_seed": 0,
        "temperature": 0,
        "chunk_threshold": 1,
        "cold_runs": 1,
        "warm_runs": 3,
    },
}
PROTOCOL_NOTES = {
    "zoo-chat-v1": (
        "**Protocol `zoo-chat-v1`** (live — [CoreAI Zoo](https://apps.apple.com/us/app/coreai-zoo/id6780135339)"
        " 2.0+, Bench tab): a fixed text prompt (\"Explain, in detail, how a transformer"
        " language model generates text one token at a time…\", ~52–56 tokens depending on"
        " the tokenizer) → 256 greedy decode tokens, S=1 prefill (`COREAI_CHUNK_THRESHOLD=1`),"
        " 1 cold + 3 warm runs, session reset before every run so each run is a full prefill,"
        " measured through the kit `ChatSession` — the app's real chat path. decode tok/s is"
        " the 32-token rolling-window steady-state rate; prefill tok/s = prompt tokens / TTFT."
    ),
    "pb-random-v1": (
        "**Protocol `pb-random-v1`** (July 2026 CoreAIChat TestFlight build, no longer"
        " produced — kept as the historical table): fixed 128-token random prompt (seed 0)"
        " → 256 greedy decode tokens, S=1 prefill (`COREAI_CHUNK_THRESHOLD=1`), 1 cold + 3 warm"
        " runs on a freshly created engine. Raw token ids fed to a forked engine, so its"
        " numbers are not byte-comparable with `zoo-chat-v1`."
    ),
}
# Canonical column order = the CoreAI Zoo chat catalog (coreai-kit ModelCatalog.swift,
# kind .chat, catalog order); unknown model ids append after these. A protocol's table
# shows only the models that have at least one accepted submission under it.
MODEL_ORDER = [
    "qwen3-0.6b", "qwen3-4b", "mistral-7b-v0.3", "gemma-3-4b-it",
    "qwen3.5-0.8b", "qwen3.5-2b", "youtu-llm-2b", "lfm2.5-1.2b", "lfm2.5-2.6b",
    "granite-4.0-h-1b", "nemotron-3-nano-4b", "minicpm5-1b", "minicpm5-2b",
    "nanbeige4.1-3b", "nanbeige4.2-3b", "qwen3-8b", "gemma-3-12b-it",
    "qwen3.6-35b-a3b", "qwen3.6-27b", "qwen3.8-27b", "glm-4.7-flash", "lfm2.5-8b-a1b",
    "gemma-4-12b", "gemma-4-31b", "gemma-4-e2b", "gemma-4-e4b",
]

# utsname.machine -> (marketing name, chip). Unknown ids show raw — extend as needed.
DEVICE_NAMES = {
    "iPhone13,1": ("iPhone 12 mini", "A14"),
    "iPhone13,2": ("iPhone 12", "A14"),
    "iPhone13,3": ("iPhone 12 Pro", "A14"),
    "iPhone13,4": ("iPhone 12 Pro Max", "A14"),
    "iPhone14,4": ("iPhone 13 mini", "A15"),
    "iPhone14,5": ("iPhone 13", "A15"),
    "iPhone14,2": ("iPhone 13 Pro", "A15"),
    "iPhone14,3": ("iPhone 13 Pro Max", "A15"),
    "iPhone14,6": ("iPhone SE (3rd gen)", "A15"),
    "iPhone14,7": ("iPhone 14", "A15"),
    "iPhone14,8": ("iPhone 14 Plus", "A15"),
    "iPhone15,2": ("iPhone 14 Pro", "A16"),
    "iPhone15,3": ("iPhone 14 Pro Max", "A16"),
    "iPhone15,4": ("iPhone 15", "A16"),
    "iPhone15,5": ("iPhone 15 Plus", "A16"),
    "iPhone16,1": ("iPhone 15 Pro", "A17 Pro"),
    "iPhone16,2": ("iPhone 15 Pro Max", "A17 Pro"),
    "iPhone17,1": ("iPhone 16 Pro", "A18 Pro"),
    "iPhone17,2": ("iPhone 16 Pro Max", "A18 Pro"),
    "iPhone17,3": ("iPhone 16", "A18"),
    "iPhone17,4": ("iPhone 16 Plus", "A18"),
    "iPhone17,5": ("iPhone 16e", "A18"),
    "iPhone18,1": ("iPhone 17 Pro", "A19 Pro"),
    "iPhone18,2": ("iPhone 17 Pro Max", "A19 Pro"),
    "iPhone18,3": ("iPhone 17", "A19"),
    "iPhone18,4": ("iPhone Air", "A19 Pro"),
    "iPhone19,7": ("iPhone 18 Pro Max", "A20 Pro"),
    "Mac16,7": ("MacBook Pro 16-inch 2024", "M4 Pro"),
}


def log(msg):
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------- blob intake

def extract_blob_text(issue_body):
    """The issue form (render: json) fences the paste; accept a bare object too."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", issue_body, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"\{.*\}", issue_body, re.DOTALL)
    return m.group(0) if m else None


def validate(blob):
    """Return a rejection reason, or None when the blob is acceptable."""
    if not isinstance(blob, dict):
        return "not a JSON object"
    if blob.get("schema_version") != SCHEMA_VERSION:
        return f"schema_version {blob.get('schema_version')!r} != {SCHEMA_VERSION}"
    if blob.get("kind") != BLOB_KIND:
        return f"kind {blob.get('kind')!r}"
    proto = blob.get("protocol") or {}
    spec = PROTOCOLS.get(proto.get("name"))
    if spec is None:
        return f"protocol.name {proto.get('name')!r} not in {sorted(PROTOCOLS)}"
    for key, want in spec.items():
        if proto.get(key) != want:
            return f"protocol.{key} {proto.get(key)!r} != {want!r} (protocol is fixed)"
    for path in (("device", "model_identifier"), ("model", "id"),
                 ("environment", "low_power_mode"), ("results", "runs")):
        node = blob
        for part in path:
            node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            return f"missing {'.'.join(path)}"
    warm = [r for r in blob["results"]["runs"]
            if r.get("kind") == "warm" and isinstance(r.get("decode_tok_s"), (int, float))]
    if len(warm) != spec["warm_runs"]:
        return f"expected {spec['warm_runs']} warm runs, got {len(warm)}"
    if any(r["decode_tok_s"] <= 0 for r in warm):
        return "non-positive warm decode_tok_s"
    return None


def excluded_reason(blob):
    """Environment post-filter (field-data policy). None = counts toward medians."""
    env = blob["environment"]
    if env.get("low_power_mode"):
        return "Low Power Mode on"
    if env.get("thermal_state_before") in ("serious", "critical"):
        return f"thermal {env['thermal_state_before']} before run"
    return None


class Submission:
    def __init__(self, blob, source, submitter):
        self.blob = blob
        self.source = source          # issue URL or file path
        self.submitter = submitter    # GitHub login or "local"
        self.protocol = blob["protocol"]["name"]
        self.device = blob["device"]["model_identifier"]
        self.model = blob["model"]["id"]
        warm = [r["decode_tok_s"] for r in blob["results"]["runs"] if r["kind"] == "warm"]
        self.warm_decode_median = statistics.median(warm)
        prefill = [r["prefill_tok_s"] for r in blob["results"]["runs"]
                   if r["kind"] == "warm" and isinstance(r.get("prefill_tok_s"), (int, float))]
        self.warm_prefill_median = statistics.median(prefill) if prefill else None
        self.excluded = excluded_reason(blob)


# ---------------------------------------------------------------- sources

def fetch_issues(repo):
    """All open+closed issues labeled bench-result (closing an issue does NOT
    remove the row — the issue stream is an append-only audit log)."""
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    issues, page = [], 1
    while True:
        url = (f"https://api.github.com/repos/{repo}/issues"
               f"?labels=bench-result&state=all&per_page=100&page={page}")
        req = urllib.request.Request(url, headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "aggregate-bench",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            batch = json.load(resp)
        issues += [i for i in batch if "pull_request" not in i]
        if len(batch) < 100:
            return issues
        page += 1


def load_submissions(repo, local_dirs, use_github):
    subs, rejected = [], []
    if use_github:
        try:
            issues = fetch_issues(repo)
            log(f"fetched {len(issues)} bench-result issue(s) from {repo}")
        except Exception as e:  # noqa: BLE001 — network failure is a normal offline case
            log(f"WARNING: could not fetch issues from {repo}: {e}")
            issues = []
        for issue in issues:
            ref = f"#{issue['number']}"
            text = extract_blob_text(issue.get("body") or "")
            if not text:
                rejected.append((ref, "no JSON blob found in body"))
                continue
            try:
                blob = json.loads(text)
            except json.JSONDecodeError as e:
                rejected.append((ref, f"invalid JSON: {e}"))
                continue
            reason = validate(blob)
            if reason:
                rejected.append((ref, reason))
                continue
            subs.append(Submission(blob, issue["html_url"],
                                   (issue.get("user") or {}).get("login", "unknown")))
    for d in local_dirs:
        for path in sorted(Path(d).glob("*.json")):
            try:
                blob = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as e:
                rejected.append((str(path), f"unreadable: {e}"))
                continue
            reason = validate(blob)
            if reason:
                rejected.append((str(path), reason))
                continue
            subs.append(Submission(blob, str(path), "local"))
    return subs, rejected


# ---------------------------------------------------------------- table

def device_label(identifier):
    if identifier in DEVICE_NAMES:
        name, chip = DEVICE_NAMES[identifier]
        return f"{name} ({chip}, `{identifier}`)"
    return f"`{identifier}`"


DEVICE_FAMILY_ORDER = ["iPhone", "iPad", "Mac"]


def device_sort_key(identifier):
    """iPhone rows first, newest first; then iPad, Mac, then anything else."""
    m = re.match(r"([A-Za-z]+)(\d+),(\d+)", identifier)
    if m:
        fam = m.group(1)
        rank = DEVICE_FAMILY_ORDER.index(fam) if fam in DEVICE_FAMILY_ORDER else len(DEVICE_FAMILY_ORDER)
        return (rank, fam, -int(m.group(2)), -int(m.group(3)))
    return (len(DEVICE_FAMILY_ORDER) + 1, identifier, 0, 0)


def protocol_table(subs, protocol):
    """Markdown lines for one protocol's device x model table (accepted subs only)."""
    included = [s for s in subs if s.protocol == protocol and not s.excluded]
    present = {s.model for s in included}
    models = [m for m in MODEL_ORDER if m in present] + sorted(present - set(MODEL_ORDER))
    devices = sorted({s.device for s in included}, key=device_sort_key)

    cells = {}  # (device, model) -> [median warm decode per submission]
    for s in included:
        cells.setdefault((s.device, s.model), []).append(s.warm_decode_median)

    lines = [f"## `{protocol}` — decode tok/s (median warm)", "", PROTOCOL_NOTES[protocol], ""]
    if not devices:
        lines += ["_no accepted submissions yet_", ""]
        return lines
    header = "| Device | " + " | ".join(f"`{m}`" for m in models) + " |"
    sep = "|---" + "|---:" * len(models) + "|"
    lines += [header, sep]
    for dev in devices:
        row = [device_label(dev)]
        for model in models:
            vals = cells.get((dev, model))
            if not vals:
                row.append("—")
            else:
                med = statistics.median(vals)
                mark = "" if len(vals) >= 3 else "\\*"
                row.append(f"{med:.1f}{mark} (n={len(vals)})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


THANKS_HEADING = "## Thank you to the contributors"
THANKS_LEAD = ("Every row below exists because someone ran the bench on their own device"
               " and sent the result in.")


def thanks_section(subs, repo, heading=THANKS_HEADING, lead=THANKS_LEAD):
    """Credit block shown right under the title: one line per contributor with
    what they measured, each model linked to its issue. Excluded submissions are
    credited too — the run was still done. The maintainer's own rows are not."""
    owner = repo.split("/")[0].lower()
    by_user = {}
    for s in subs:
        if s.submitter in ("local", "unknown") or s.submitter.lower() == owner:
            continue
        by_user.setdefault(s.submitter, {}).setdefault(s.device, []).append(s)
    if not by_user:
        return []
    lines = [heading, "", lead, ""]
    for user in sorted(by_user, key=str.lower):
        parts = []
        for dev in sorted(by_user[user], key=device_sort_key):
            runs = sorted(by_user[user][dev],
                          key=lambda s: MODEL_ORDER.index(s.model) if s.model in MODEL_ORDER
                          else len(MODEL_ORDER))
            models = ", ".join(f"[`{s.model}`]({s.source})" for s in runs)
            parts.append(f"{device_label(dev)}: {models}")
        lines.append(f"- **[@{user}](https://github.com/{user})** — " + " · ".join(parts))
    lines.append("")
    return lines


def build_markdown(subs, rejected, repo):
    included = [s for s in subs if not s.excluded]
    excluded = [s for s in subs if s.excluded]

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Community Benchmarks",
        "",
    ]
    lines += thanks_section(subs, repo)
    lines += [
        "**Community field data** — measured by the Bench tab of the",
        "[CoreAI Zoo](https://apps.apple.com/us/app/coreai-zoo/id6780135339) app",
        "on contributors' own devices and submitted as",
        "[bench-result issues](https://github.com/" + repo + "/issues?q=label%3Abench-result)",
        "(the public audit log — closing an issue does not remove its row). The app measures",
        "and builds the result blob; no number in this table was typed by a human. This is NOT",
        "a controlled-environment benchmark — background load and heat show up here as",
        "real-world variance. For the maintainer's controlled numbers see each model card.",
        "",
        "One table per protocol. Cell = median across submissions of each submission's",
        "median **warm decode tok/s**; `n` = accepted submissions. Cells with n < 3 are",
        "**provisional** (marked \\*). Blobs with Low Power Mode on or a serious/critical",
        "thermal state before the run are excluded from medians (counted below). Model load",
        "time is not published (the app's `load_s` includes a first download). Devices are",
        "the `utsname.machine` id; ids without a marketing name yet show raw.",
        "",
        "**Add your device**: install [CoreAI Zoo](https://apps.apple.com/us/app/coreai-zoo/id6780135339)",
        "from the App Store → Bench tab → pick a chat model → Run → *Submit on GitHub*. Your",
        "device becomes a row here on the next aggregation",
        "(`python3 scripts/aggregate_bench.py`).",
        "",
        f"_Generated by `scripts/aggregate_bench.py` — do not edit by hand. Last run: {now}._",
        "",
    ]
    for protocol in PROTOCOLS:
        lines += protocol_table(subs, protocol)

    lines += [
        f"Accepted submissions: **{len(included)}** · excluded by environment filter:"
        f" **{len(excluded)}** · rejected (schema/protocol): **{len(rejected)}**",
        "",
    ]

    if excluded:
        lines += ["<details><summary>Excluded submissions (environment filter)</summary>", ""]
        for s in excluded:
            lines.append(f"- {s.device} · {s.model} · {s.protocol} — {s.excluded} ({s.source})")
        lines += ["", "</details>", ""]
    if rejected:
        lines += ["<details><summary>Rejected submissions (schema/protocol)</summary>", ""]
        for ref, reason in rejected:
            lines.append(f"- {ref} — {reason}")
        lines += ["", "</details>", ""]

    return "\n".join(lines) + "\n"


DEFAULT_OUT = Path(__file__).resolve().parent.parent / "BENCHMARKS.md"
README_PATH = Path(__file__).resolve().parent.parent / "README.md"
README_BEGIN = "<!-- bench-thanks begin (managed by scripts/aggregate_bench.py — do not edit by hand) -->"
README_END = "<!-- bench-thanks end -->"


def update_readme_thanks(subs, repo):
    """Rewrite the credit block near the top of README.md between its two markers."""
    text = README_PATH.read_text()
    if README_BEGIN not in text or README_END not in text:
        log(f"WARNING: bench-thanks markers not found in {README_PATH}; README not updated")
        return
    body = thanks_section(
        subs, repo, heading="### Community benchmarks",
        lead="People who ran the Bench tab of the CoreAI Zoo app on their own devices:")
    if body:
        body = body[:-1] + ["", "The results are in [`BENCHMARKS.md`](BENCHMARKS.md). To add your device, use the"
                            " Bench tab of [CoreAI Zoo](https://apps.apple.com/us/app/coreai-zoo/id6780135339).", ""]
    head, rest = text.split(README_BEGIN, 1)
    _, tail = rest.split(README_END, 1)
    new = head + README_BEGIN + "\n" + "\n".join(body) + README_END + tail
    if new != text:
        README_PATH.write_text(new)
        log(f"updated {README_PATH}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=REPO_DEFAULT, help=f"GitHub repo (default {REPO_DEFAULT})")
    ap.add_argument("--local", action="append", default=[],
                    help="directory of local *.json blobs (repeatable), in addition to issues")
    ap.add_argument("--local-only", action="append", default=[],
                    help="directory of local *.json blobs; skip the GitHub fetch")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output path (default: repo-root BENCHMARKS.md)")
    args = ap.parse_args()

    local_dirs = args.local + args.local_only
    subs, rejected = load_submissions(args.repo, local_dirs, use_github=not args.local_only)
    log(f"accepted {len([s for s in subs if not s.excluded])}, "
        f"excluded {len([s for s in subs if s.excluded])}, rejected {len(rejected)}")
    for ref, reason in rejected:
        log(f"  rejected {ref}: {reason}")

    md = build_markdown(subs, rejected, args.repo)
    Path(args.out).write_text(md)
    log(f"wrote {args.out}")

    # The README credit block follows the published table only — not a --local-only
    # dry run, not a run written somewhere else.
    if Path(args.out).resolve() == DEFAULT_OUT and not args.local_only:
        update_readme_thanks(subs, args.repo)


if __name__ == "__main__":
    main()
