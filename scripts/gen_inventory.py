#!/usr/bin/env python3
"""Generate models/_INVENTORY.md and models/index.json — the catalog's worklist and its
machine-readable index.

For every repo the zoo publishes, records what the repo actually contains (bundle count,
format), which `models/<family>/` documents it, whether that family carries a recipe, the
tier-1 verification verdict, and how often it is downloaded — so work can be ordered by
reach instead of by guess.

    python3 scripts/gen_inventory.py            # refresh both files
    python3 scripts/gen_inventory.py --print    # dry run, print the inventory to stdout
    python3 scripts/gen_inventory.py --offline  # use only what is already cached

Reads the Hugging Face listing API and nothing else — no weights, no per-file fetches.
`models/index.json` is what an agent should read: one entry per model family, with the
recipe name to run and the bundle it produces.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from collections import defaultdict
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
sys.path.insert(0, str(REPO / "conversion"))
from _hf_catalog import Catalog, bundles_of, repo_format  # noqa: E402
from _recipe import source_model  # noqa: E402

AUTHORS = ["mlboydaisuke"]
# Escape hatch: a published repo the zoo links but no recipe names. Contributor-hosted ports
# are picked up from their recipes instead (see contributor_repos), because a hand-kept list
# is a list someone outside this repo cannot know to append to — and a port whose bundles
# live on the author's own account is the normal case, not the exception.
EXTRA_REPOS: list[str] = []

HF_LINK = re.compile(r"https://huggingface\.co/([\w.-]+/[\w.-]+)")
CARD_LINK = re.compile(r"\([\w./-]*models/([\w.-]+)/README\.md\)")


def families() -> list[str]:
    return sorted(p.name for p in MODELS.iterdir() if (p / "README.md").exists())


def recipes() -> dict[str, dict]:
    """Every models/<family>/recipe.toml entry, tagged with its family."""
    out: dict[str, dict] = {}
    for path in sorted(MODELS.glob("*/recipe.toml")):
        with open(path, "rb") as fh:
            for name, recipe in tomllib.load(fh).items():
                recipe["family"] = path.parent.name
                out[name] = recipe
    return out


def repo_to_family(all_recipes: dict[str, dict]) -> dict[str, set[str]]:
    """Which `models/<family>/` documents a published repo.

    In precedence order, because the last one is a guess: the recipe's own `hf_repo`, the
    kit catalog joined through the gen-cards sidecar, a README table row that carries both
    a repo link and a card link, and only then a bare link inside some card's body — which
    is why a card that mentions a sibling model used to claim it.
    """
    out: dict[str, set[str]] = defaultdict(set)
    for recipe in all_recipes.values():
        if repo := recipe.get("hf_repo"):
            out[repo].add(recipe["family"])

    sidecar = REPO / "scripts" / "gen-cards" / "cards.json"
    if sidecar.exists():
        by_slug = json.loads(sidecar.read_text()).get("models", {})
        for repo, slug in kit_slugs().items():
            card = (by_slug.get(slug) or {}).get("zooCard")
            if repo and card:
                out[repo].add(Path(card).parts[-2])

    for line in (REPO / "README.md").read_text().splitlines():
        if not line.startswith("|"):
            continue
        repos = {r for r in HF_LINK.findall(line) if not r.startswith("john-rocky/")}
        cards = set(CARD_LINK.findall(line))
        for r in repos:
            out[r] |= cards

    for family in families():
        text = (MODELS / family / "README.md").read_text(errors="ignore")
        for rid in set(HF_LINK.findall(text)):
            if not rid.startswith("john-rocky/") and not out.get(rid):
                out[rid].add(family)
    return out


def kit_slugs() -> dict[str, str]:
    """HF repo id -> CoreAIKit catalog slug.

    Read from the kit's own catalog when it is checked out beside this repo, because that
    is the list of models the kit actually ships; the gen-cards sidecar only knows the
    ones whose card it manages, which is how "shipped through the kit but undocumented
    here" stayed invisible.
    """
    for base in (REPO.parent / "coreai-kit", REPO.parent.parent / "coreai-kit"):
        path = base / "catalog.json"
        if path.exists():
            data = json.loads(path.read_text())
            models = data.get("models", data)
            items = models.items() if isinstance(models, dict) else (
                (m.get("id"), m) for m in models)
            return {m.get("repo") or m.get("hf"): slug for slug, m in items
                    if (m.get("repo") or m.get("hf"))}
    return {}


def verify_results() -> dict[str, list[dict]]:
    """HF repo id -> per-bundle tier-1 verdicts, from `conversion/zoo_verify.py --json`."""
    path = MODELS / "_VERIFY.json"
    if not path.exists():
        return {}
    out: dict[str, list[dict]] = defaultdict(list)
    for b in json.loads(path.read_text())["bundles"]:
        out[b["repo"]].append(b)
    return out


def smoke_results() -> dict[str, list[dict]]:
    """HF repo id -> per-bundle tier-0 records, from `conversion/zoo_smoke.py`."""
    path = MODELS / "_SMOKE.json"
    if not path.exists():
        return {}
    out: dict[str, list[dict]] = defaultdict(list)
    for b in json.loads(path.read_text())["bundles"]:
        out[b["repo"]].append(b)
    return out


def contributor_repos(all_recipes: dict[str, dict]) -> list[str]:
    """Published repos a recipe names that we do not own — a contributor's own account.

    The recipe already carries `hf_repo`, so the inventory learns about a contributed port
    from the PR that adds it rather than from a maintainer remembering to register it.
    """
    ours = {a.lower() for a in AUTHORS}
    named = {r["hf_repo"] for r in all_recipes.values() if r.get("hf_repo")}
    return sorted(r for r in named if r.split("/")[0].lower() not in ours)


def collect(cat: Catalog) -> list[dict]:
    all_recipes = recipes()
    published = [m for a in AUTHORS for m in cat.repos_by_author(a)]
    published += [m for m in (cat.repo(r) for r in
                              dict.fromkeys(contributor_repos(all_recipes) + EXTRA_REPOS)) if m]

    by_repo = repo_to_family(all_recipes)
    by_family_recipes: dict[str, list[str]] = defaultdict(list)
    for name, r in all_recipes.items():
        by_family_recipes[r["family"]].append(name)
    slugs, verified, smoked = kit_slugs(), verify_results(), smoke_results()

    rows = []
    for m in published:
        rid = m["id"]
        files = [s["rfilename"] for s in m.get("siblings", [])]
        fams = sorted(by_repo.get(rid, set()))
        rows.append({
            "id": rid,
            "dl30": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "updated": (m.get("lastModified") or "")[:10],
            "format": repo_format(files),
            # Apple's own export recipes re-run for the bench matrix (ZOO_BLUEPRINT P2),
            # not zoo ports — they answer to Apple's repo, not to a zoo card.
            "role": "official" if rid.endswith("-CoreAI-official") else "port",
            "bundles": bundles_of(files),
            "families": fams,
            "recipes": sorted({n for f in fams for n in by_family_recipes.get(f, [])}),
            "kit": [slugs[rid]] if rid in slugs else [],
            "tier1": verified.get(rid, []),
            "smoke": smoked.get(rid, []),
        })
    rows.sort(key=lambda r: (-r["dl30"], r["id"].lower()))
    return rows


def cell(text: str) -> str:
    """Special tokens such as `<|im_end|>` contain the column separator."""
    return text.replace("|", "\\|")


def tier1_cell(row: dict) -> str:
    """Compact per-repo tier-1 result: failures first, because that is the point."""
    if not row["tier1"]:
        return "—"
    tally: dict[str, int] = defaultdict(int)
    for b in row["tier1"]:
        tally[b["verdict"]] += 1
    return " ".join(f"**{tally[v]} {v}**" if v in ("FAIL", "DIFF") else f"{tally[v]} {v.lower()}"
                    for v in ("FAIL", "DIFF", "PASS", "SKIPPED") if tally.get(v))


def smoke_cell(row: dict) -> str:
    """Per-repo tier-0 result: which OS build loaded it, failures first."""
    if not row["smoke"]:
        return "—"
    tally: dict[str, int] = defaultdict(int)
    builds: set[str] = set()
    for b in row["smoke"]:
        status = (b.get("load") or {}).get("status", "stamp")
        tally["ok" if status == "ok" else "device-only" if status == "skipped"
              else "deferred" if status in ("deferred", "stamp") else "FAIL"] += 1
        if status == "ok":
            builds.add((b.get("host") or {}).get("os_build") or "?")
    parts = [f"**{tally['FAIL']} FAIL**"] if tally.get("FAIL") else []
    if tally.get("ok"):
        parts.append(f"{tally['ok']} load ({', '.join(sorted(builds))})")
    parts += [f"{tally[k]} {k}" for k in ("device-only", "deferred") if tally.get(k)]
    return " ".join(parts)


def render(rows: list[dict]) -> str:
    all_recipes = recipes()
    coreai = [r for r in rows if r["format"] == "coreai"]
    carded = [r for r in coreai if r["families"]]
    recipe_rows = [r for r in rows if r["recipes"]]
    no_card = [r for r in coreai if not r["families"] and r["role"] == "port"]
    official_no_card = [r for r in coreai if not r["families"] and r["role"] == "official"]
    ambiguous = [r for r in carded if len(r["bundles"]) > 1 and not r["recipes"]]
    single = [r for r in carded if len(r["bundles"]) == 1 and not r["recipes"]]

    L = [
        "# Published model inventory",
        "",
        f"Generated by `scripts/gen_inventory.py` on {date.today().isoformat()} from the",
        "Hugging Face listing API. **Do not hand-edit** — rerun the script.",
        "",
        "One row per published repo. `bundles` counts the directories holding a",
        "`metadata.json` beside an `.aimodel` — what the runtime loads, and what",
        "verification runs against. More than one bundle means the repo ships variants,",
        "so the card and the recipe have to say which one is *the* published",
        "configuration; a single bundle answers that question by itself.",
        "",
        "`fmt` is derived from the files, not the name: `coreai` (`.aimodel`),",
        "`coreml` (pre-Core-AI ports, kept for their download history), `litert`",
        "(the Google LiteRT collaboration), `other`.",
        "",
        "| metric | count |",
        "| --- | --- |",
        f"| published repos | {len(rows)} |",
        f"| Core AI repos | {len(coreai)} |",
        f"| Core AI bundles inside them | {sum(len(r['bundles']) for r in coreai)} |",
        f"| Core AI repos with a `models/<family>/` card | {len(carded)} |",
        f"| repos covered by a recipe | {len(recipe_rows)} |",
        f"| Core AI repos with 0 downloads in the last 30 days | {sum(1 for r in coreai if not r['dl30'])} |",
        "",
        "## All repos, by 30-day downloads",
        "",
        "| repo | 30d DL | ♥ | fmt | role | bundles | tier-1 | tier-0 load | model | recipe | kit |",
        "| --- | ---: | ---: | --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        model = ", ".join(f"[{f}]({f}/README.md)" for f in r["families"]) or "—"
        L.append(
            f"| [{r['id']}](https://huggingface.co/{r['id']}) | {r['dl30']} | {r['likes']} | "
            f"{r['format']} | {r['role']} | {len(r['bundles'])} | {tier1_cell(r)} | "
            f"{smoke_cell(r)} | {model} | "
            f"{', '.join(f'`{x}`' for x in r['recipes']) or '—'} | "
            f"{', '.join(f'`{x}`' for x in r['kit']) or '—'} |"
        )

    defects = [(r, b) for r in rows for b in r["tier1"] if b["verdict"] in ("FAIL", "DIFF")]
    checked = sum(len(r["tier1"]) for r in rows)
    L += [
        "",
        "## Tier-1 defects",
        "",
        f"From `conversion/zoo_verify.py --all` over {checked} published bundles: the",
        "bundle's own tokenizer, chat template, context length and declared precision",
        "compared against the source repository it names in its `metadata.json`. No",
        "oracle, no device, no weights.",
        "",
        "**FAIL** = wrong on its own terms. **DIFF** = deviates from the source with no",
        "recorded reason; record the expectation in `models/<family>/verify.toml` and it",
        "becomes the bar instead of the deviation.",
        "",
    ]
    if defects:
        L += ["| repo | bundle | verdict | what |", "| --- | --- | --- | --- |"]
        for r, b in sorted(defects, key=lambda x: (x[1]["verdict"], -x[0]["dl30"])):
            for c in b["checks"]:
                if c["status"] in ("FAIL", "DIFF"):
                    L.append(f"| {r['id'].split('/')[-1]} | `{cell(b['bundle'])}` | "
                             f"{b['verdict']} | {c['check']}: {cell(c['detail'])} |")
    elif checked:
        L.append(f"**None.** All {checked} bundles either agree with their source or carry a")
        L.append("declared expectation in `models/<family>/verify.toml`.")
    else:
        L.append("- (not run — `conversion/zoo_verify.py --all --json models/_VERIFY.json`)")

    smoked = [(r, b) for r in rows for b in r["smoke"]]
    loaded = [(r, b) for r, b in smoked if (b.get("load") or {}).get("status") == "ok"]
    failed = [(r, b) for r, b in smoked
              if (b.get("load") or {}).get("status") not in (None, "ok", "skipped", "deferred")]
    stale_ir = [(r, b) for r, b in smoked if str(b.get("ir", "")).startswith("0.4.0")]
    L += [
        "",
        "## Tier-0 load check",
        "",
        "From `conversion/zoo_smoke.py`: the published bundle downloaded and loaded through the",
        "Core AI runtime on a named OS build, plus its asset producer stamp read off the Hub.",
        "Staged on purpose — the most-downloaded bundles first, the rest afterwards — so the",
        "table says which build checked which bundle rather than implying a sweep that did not",
        "run. `device-only` = compiled or iOS-path bundles, checked on a device, never on a Mac.",
        "",
        f"- bundles with a load record: {len(loaded)} loaded, {len(failed)} failing, "
        f"{sum(1 for _, b in smoked if (b.get('load') or {}).get('status') == 'skipped')} device-only, "
        f"{sum(1 for _, b in smoked if (b.get('load') or {}).get('status') in ('deferred', None))} not yet run",
        f"- bundles still carrying 0.4.0-era IR (no producer stamp): {len(stale_ir)} of those "
        "with a record — loads on OS 27 beta 1 only; refused at load on every build measured "
        "since, through 26A5416b on 2026-09-04 (`cli/DOCTOR_RULES.md` IR-040)",
    ]
    if failed:
        L += ["", "| repo | bundle | build | what |", "| --- | --- | --- | --- |"]
        for r, b in sorted(failed, key=lambda x: -x[0]["dl30"]):
            load = b["load"]
            detail = load.get("detail") or "; ".join(
                f"{a.split('/')[-1]}: {v.get('detail', '')}"
                for a, v in (load.get("assets") or {}).items() if v.get("status") != "ok")
            L.append(f"| {r['id'].split('/')[-1]} | `{cell(b['bundle'])}` | "
                     f"{(b.get('host') or {}).get('os_build', '?')} | "
                     f"{load['status']}: {cell(' '.join(detail.split())[:160])} |")
    if not smoked:
        L.append("- (not run — `python3 conversion/zoo_smoke.py --top 20 --badge <path>`)")

    L += [
        "",
        "## Needs owner input",
        "",
        "### 1. Zoo ports with no card",
        "",
        "Published and downloadable, undocumented here. Each is either a card to write",
        "or a repo to unpublish. (Bench exports of Apple's own recipes are listed",
        "separately below — those answer to Apple's repo, not to a zoo card.)",
        "",
    ]
    L += [f"- [{r['id']}](https://huggingface.co/{r['id']}) — {r['dl30']} DL/30d, "
          f"{len(r['bundles'])} bundle(s), updated {r['updated']}"
          for r in no_card] or ["- (none)"]

    L += [
        "",
        f"Bench exports of Apple's own recipes, no card expected ({len(official_no_card)}): "
        + ", ".join(f"`{r['id'].split('/')[-1]}`" for r in official_no_card),
        "",
        "### 2. Carded, several bundles, no recipe — which one shipped?",
        "",
        'These are the `status = "unverified"` candidates: the card documents the model,',
        "the repo publishes more than one bundle, and nothing records which is the",
        "published configuration. **Do not guess their `args`.**",
        "",
    ]
    for r in ambiguous:
        L.append(f"- **{r['id']}** ({r['dl30']} DL/30d) — {len(r['bundles'])} bundles:")
        L += [f"  - `{b}`" for b in r["bundles"]]
    if not ambiguous:
        L.append("- (none)")

    unverified = {n: r for n, r in all_recipes.items() if r.get("status") == "unverified"}
    L += [
        "",
        "### 3. Recipes recorded, shipped configuration unknown",
        "",
        f'{len(unverified)} of the {len(all_recipes)} recipes carry `status = "unverified"`:',
        "the script is known, the arguments that produced the published bundle are not,",
        "and nothing in the repo records them. `zoo_convert.py` refuses to run these",
        "without `--force`. Each needs one answer from the owner.",
        "",
    ]
    for name, r in unverified.items():
        # open_questions is stored line-wrapped in the TOML; one question per entry.
        question = " ".join(r.get("open_questions", [])) or "not stated"
        L.append(f"- **`{name}`** ({r.get('hf_repo', '?')}) — {question}")

    L += [
        "",
        "### 4. Carded, exactly one bundle, no recipe",
        "",
        "Unambiguous by construction — the single published bundle *is* the shipped",
        "configuration. These can get a recipe without asking anyone, provided the",
        "export flags are recoverable from the card or the conversion script.",
        "",
    ]
    L += [f"- {r['id']} — `{r['bundles'][0]}`" for r in single] or ["- (none)"]
    L.append("")
    return "\n".join(L)


def render_index(rows: list[dict]) -> dict:
    """models/index.json — what an agent reads to find and run a model."""
    all_recipes = recipes()
    by_family: dict[str, dict] = {
        f: {"family": f, "card": f"models/{f}/README.md", "recipes": [], "repos": []}
        for f in families()
    }
    for name, r in all_recipes.items():
        entry = by_family.setdefault(
            r["family"], {"family": r["family"], "card": f"models/{r['family']}/README.md",
                          "recipes": [], "repos": []})
        # Whether a gate transcript is published for this bundle, and where. `status` says
        # the recipe reproduces the artifact; this says the artifact was checked against the
        # original *and the evidence is readable* — a distinction a reader deciding whether
        # to depend on a model has to be able to make without reading prose.
        transcript = Path(f"models/{r['family']}/gate-{name}.json")
        # What this was converted from. Resolved, never copied into recipe.toml: most
        # recipes name no checkpoint because they use their exporter's default, and a
        # duplicated default is a second source of truth that drifts.
        checkpoint, checkpoint_from = source_model(r)
        entry["recipes"].append({
            "name": name,
            "status": r.get("status", "unknown"),
            "source_model": checkpoint,
            "source_model_from": checkpoint_from,
            "hf_repo": r.get("hf_repo"),
            "bundle": r.get("bundle"),
            "steps": len(r.get("steps", [])) or 1,
            "run": f"python3 conversion/zoo_convert.py run {name}",
            "gate_transcript": str(transcript) if (REPO / transcript).is_file() else None,
            "open_questions": r.get("open_questions", []),
        })
    for row in rows:
        for family in row["families"]:
            entry = by_family.setdefault(
                family, {"family": family, "card": f"models/{family}/README.md",
                         "recipes": [], "repos": []})
            entry["repos"].append({
                "id": row["id"],
                "downloads_30d": row["dl30"],
                "bundles": row["bundles"],
                "tier1": {v: sum(1 for b in row["tier1"] if b["verdict"] == v)
                          for v in ("PASS", "DIFF", "FAIL", "SKIPPED")
                          if any(b["verdict"] == v for b in row["tier1"])},
                # Tier-0: per bundle, whether it loaded and on which OS build (zoo_smoke.py).
                "smoke": [{"bundle": b["bundle"], "ir": b.get("ir"),
                           "load": (b.get("load") or {}).get("status"),
                           "os_build": (b.get("host") or {}).get("os_build"),
                           "date": b.get("date")} for b in row["smoke"]],
            })
    return {
        "generated": date.today().isoformat(),
        "how_to_use": {
            "list": "python3 conversion/zoo_convert.py list",
            "show": "python3 conversion/zoo_convert.py show <recipe>",
            "reproduce": "python3 conversion/zoo_convert.py run <recipe>",
            "verify": "python3 conversion/zoo_verify.py <hf_repo>",
            "skill": "skills/skills/reproduce-a-zoo-model/SKILL.md",
        },
        "models": [by_family[f] for f in sorted(by_family)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", help="print instead of writing the files")
    ap.add_argument("--offline", action="store_true", help="use only the local cache")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    cat = Catalog(**({"cache_dir": args.cache_dir} if args.cache_dir else {}), offline=args.offline)
    rows = collect(cat)
    out = render(rows)
    if args.print:
        print(out)
        return 0
    (MODELS / "_INVENTORY.md").write_text(out)
    (MODELS / "index.json").write_text(json.dumps(render_index(rows), indent=1) + "\n")
    coreai = [r for r in rows if r["format"] == "coreai"]
    print(f"wrote models/_INVENTORY.md and models/index.json — {len(rows)} repos "
          f"({len(coreai)} Core AI, {sum(len(r['bundles']) for r in coreai)} bundles), "
          f"{sum(1 for r in coreai if r['families'])} documented, "
          f"{sum(1 for r in rows if r['recipes'])} with a recipe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
