#!/usr/bin/env python3
"""gen-cards — render each model's "Use it" block from catalog.json + cards.json and keep
both card surfaces (zoo/<model>.md and the HF README) byte-identical.

The generator is the "cards never lie" enforcer (design: _ZOO_DX_DESIGN.md §6.3):
  - a ▶️ door is emitted only if the runner smoke-builds in this run (swift build for the
    CLI, xcodebuild for the app) AND the committed xcodeproj matches a fresh `xcodegen
    generate` (lockfile regen-diff guard). A broken runner drops the door and FAILS the run.
  - a 💻 snippet is extracted from the runner's QuickStart.swift CARD-SNIPPET markers (not
    hand-written), then compiled standalone in a scratch package against kit as a url-dep.
  - a 🟢 door is emitted only if a distribution link is configured (none today).
  - the hero image is emitted only if demo.gif exists in the model's HF repo (HEAD-check);
    missing media is info, not failure.
  - HF pushes are dry-run by default; --push uploads after showing the diff.

Usage:
  gen_cards.py --kit ~/code/coreai-kit               # dry-run: verify, diff, exit 1 on drift
  gen_cards.py --kit ~/code/coreai-kit --write        # apply the block to zoo/<model>.md
  gen_cards.py --kit ~/code/coreai-kit --write --push # …and upload the HF README
  gen_cards.py --skip-builds                          # template iteration only; refuses
                                                      # --write/--push (doors unverified)
"""

import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ZOO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

BEGIN_FMT = ("<!-- gen-cards:use-it begin id={id} "
             "(managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->")
END_MARK = "<!-- gen-cards:use-it end -->"

failures: list[str] = []


def log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}")


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"\n!!! [FAIL] {msg}\n", file=sys.stderr)


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


# ---------------------------------------------------------------- enforcement gates

def gate_regen_diff(kit: Path, runner_dir: Path, xcodeproj: str) -> bool:
    """Lockfile check: committed xcodeproj must equal a fresh `xcodegen generate`.

    Generates in place (xcodegen rewrites relative paths when generating elsewhere, so
    out-of-place output is not byte-comparable) and uses git to compare and restore."""
    proj_rel = str((runner_dir / xcodeproj).relative_to(kit))
    if run(["git", "status", "--porcelain", "--", proj_rel], cwd=kit).stdout.strip():
        fail(f"{proj_rel} has uncommitted changes — commit it (with its project.yml) first")
        return False
    r = run(["xcodegen", "generate"], cwd=runner_dir)
    if r.returncode != 0:
        fail(f"xcodegen generate failed for {runner_dir.name}: {r.stderr.strip()[:500]}")
        return False
    dirty = run(["git", "status", "--porcelain", "--", proj_rel], cwd=kit).stdout.strip()
    if dirty:
        diff = run(["git", "diff", "--", proj_rel], cwd=kit).stdout
        run(["git", "checkout", "--", proj_rel], cwd=kit)
        fail(f"{proj_rel} differs from fresh xcodegen output — project.yml edits must land "
             f"with their regenerated project:\n{diff[:800]}")
        return False
    log("gate", f"regen-diff clean: {xcodeproj}")
    return True


def gate_cli_build(runner_dir: Path) -> bool:
    r = run(["swift", "build", "--package-path", str(runner_dir)])
    if r.returncode != 0:
        fail(f"CLI smoke build failed in {runner_dir}:\n{r.stderr[-1500:]}")
        return False
    log("gate", f"swift build OK: {runner_dir.name}")
    return True


def gate_app_build(runner_dir: Path, xcodeproj: str, scheme: str) -> bool:
    r = run(["xcodebuild", "build", "-project", str(runner_dir / xcodeproj),
             "-scheme", scheme, "-destination", "platform=macOS",
             "CODE_SIGNING_ALLOWED=NO", "-quiet"])
    if r.returncode != 0:
        fail(f"app smoke build failed ({scheme}):\n{(r.stderr or r.stdout)[-1500:]}")
        return False
    log("gate", f"xcodebuild OK: {scheme} (macOS)")
    return True


def extract_snippet(kit: Path, cfg: dict, model_id: str) -> str | None:
    """Lift the CARD-SNIPPET block from QuickStart.swift and specialize it for the card.

    Transforms (defined, minimal — the scratch compile below guards them):
      1. dedent
      2. `catalog: id` -> `catalog: "<model-id>"`
      3. strip comma-prefixed forwarded optional args (`, x: x`)
      4. leading `return ` on the last line -> `let result = `
    """
    qs = kit / cfg["quickstart"]
    text = qs.read_text()
    m = re.search(r"// CARD-SNIPPET-BEGIN\n(.*?)[ \t]*// CARD-SNIPPET-END", text, re.S)
    if not m:
        fail(f"no CARD-SNIPPET markers in {qs}")
        return None
    lines = m.group(1).rstrip("\n").split("\n")
    indent = min(len(l) - len(l.lstrip()) for l in lines if l.strip())
    lines = [l[indent:] if l.strip() else "" for l in lines]
    lines = [re.sub(r"catalog: id\b", f'catalog: "{model_id}"', l) for l in lines]
    lines = [re.sub(r",\s*(\w+): \1\b", "", l) for l in lines]
    if lines and lines[-1].startswith("return "):
        lines[-1] = "let result = " + lines[-1][len("return "):]
    body = "\n".join(lines)
    imports = cfg["snippetImport"]
    if isinstance(imports, str):
        imports = [imports]
    header = "\n".join(f"import {m}" for m in imports)
    return f"{header}\n\n{body}\n{cfg['snippetResultComment']}"


def gate_snippet_compiles(snippet: str, kit_url: str, local_kit: Path | None,
                          free_vars: dict, product: str = "CoreAIKit") -> bool:
    """Compile the extracted snippet in a scratch package against kit (url-dep by default) —
    catches the "compiles in the runner, not in the reader's app" class. `free_vars` names the
    reader-supplied inputs the snippet references (e.g. {"url": "URL"} / {"prompt": "String"});
    they become parameters of the wrapper function."""
    scratch = HERE / ".scratch" / "SnippetCheck"
    src = scratch / "Sources" / "SnippetCheck"
    src.mkdir(parents=True, exist_ok=True)
    # A branch dep pins to a revision in Package.resolved and never auto-updates — a
    # stale lockfile from an earlier run silently compiles against old kit main (or
    # fails on products that didn't exist yet). Force re-resolution every gate.
    (scratch / "Package.resolved").unlink(missing_ok=True)
    dep = (f'.package(path: "{local_kit}")' if local_kit
           else f'.package(url: "{kit_url}", branch: "main")')
    (scratch / "Package.swift").write_text(f"""// swift-tools-version: 6.0
import PackageDescription
let package = Package(
    name: "SnippetCheck",
    platforms: [.macOS("27.0")],
    dependencies: [{dep}],
    targets: [.executableTarget(name: "SnippetCheck",
        dependencies: [.product(name: "{product}", package: "coreai-kit")])]
)
""")
    imports = [l for l in snippet.split("\n") if l.startswith("import ")]
    body = [l for l in snippet.split("\n") if not l.startswith("import ")]
    if "import Foundation" not in imports:
        imports.append("import Foundation")
    params = ", ".join(f"{n}: {t}" for n, t in free_vars.items())
    lets = re.findall(r"^let (\w+)", "\n".join(body), re.M)
    tail = f"    _ = {lets[-1]}\n" if lets else ""
    wrapped = "\n".join(imports) + f"\n\nfunc __snippet({params}) async throws {{\n" + \
        "\n".join(("    " + l).rstrip() for l in body) + f"\n{tail}}}\n"
    (src / "main.swift").write_text(wrapped)
    r = run(["swift", "build", "--package-path", str(scratch)])
    if r.returncode != 0:
        fail(f"standalone snippet compile failed:\n{r.stderr[-1500:]}")
        return False
    log("gate", "standalone snippet compile OK (scratch package, "
        + ("path-dep" if local_kit else "url-dep") + ")")
    return True


def hero_media(repo: str) -> str | None:
    """The hero file hosted next to the weights: demo.gif preferred, demo.png accepted
    (a still capture is honest too — principle 9 names both)."""
    for name in ("demo.gif", "demo.png"):
        req = urllib.request.Request(
            f"https://huggingface.co/{repo}/resolve/main/{name}", method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=15):
                return name
        except Exception:
            continue
    return None


# ---------------------------------------------------------------- rendering

def render_block(model_id: str, entry: dict, cfg: dict, top: dict,
                 snippet: str, doors: dict) -> str:
    kit_url = top["kitURL"]
    clone = top["cloneDirName"]
    parts: list[str] = []

    if doors["hero"]:
        cap = cfg.get("mediaCaption")
        if not cap:
            fail(f"{model_id}: {doors['hero']} exists on HF but cards.json has no mediaCaption "
                 f"(device/OS/config line) — refusing to emit an uncaptioned capture")
        else:
            parts.append(f"![{entry['name']} demo]"
                         f"(https://huggingface.co/{entry['repo']}/resolve/main/{doors['hero']})\n"
                         f"*{cap}*\n")

    parts.append("## Use it\n")

    if doors["green"]:
        parts.append(doors["green"] + "\n")

    # ⚡ door: the kit's task layer — one line, model resolved behind the op (cards.json
    # `op`; gated by the same scratch compile as 💻, against product CoreAIOps).
    if doors.get("op"):
        o = cfg["op"]
        lead = ("this model is the default behind the kit's task op"
                if o.get("isDefault") else "run the kit's task op on this model")
        parts.append(
            f"⚡ **One line** — {lead}\n"
            "(`import CoreAIOps`; no session, no model plumbing, downloads on first use):\n"
            "\n"
            "```swift\n"
            f"let {o['result']} = try await {o['call']}\n"
            "```\n"
            "\n"
            # No count here on purpose: this line lands in ~40 external READMEs at once,
            # so a numeral goes stale in 40 places and costs 40 pushes to correct — which
            # is how it came to say "Twenty" while the enum held 23.
            f"Every op, one shape — [Cookbook]({kit_url}/blob/main/docs/COOKBOOK.md).\n")

    # Engine-showcase models (custom backends the generic kit path can't drive yet):
    # the ▶️ door points at their zoo app; 💻 is omitted until the backend is ported
    # into kit (design §4).
    if doors.get("appDoor"):
        a = cfg["appDoor"]
        parts.append(
            f"▶️ **Run it (source)** — [`{a['path']}`]({top['zooURL']}/tree/main/{a['path']}),\n"
            f"the zoo app that ships this model ({a['blurb']}; build & run steps in its README).\n")

    if doors["runner"]:
        r = cfg["runner"]
        runner_name = Path(r["dir"]).name
        # The GUI hint after "Run": defaults to the catalog picker convention; a runner
        # whose picker labels differ (or that auto-loads its single model) overrides it
        # with `runner.pickNote` — the line must match what the app actually shows.
        pick = r.get("pickNote", f"Run, then pick \"{entry['name']}\" in the model picker")
        parts.append(
            f"▶️ **Run it (source)** — the [{runner_name} runner]({kit_url}/tree/main/{r['dir']})\n"
            f"({r['blurb']}):\n"
            "\n"
            "```bash\n"
            f"git clone {kit_url}\n"
            f"open {clone}/{r['dir']}/{r['xcodeproj']}\n"
            f"# → {pick}\n"
            "\n"
            "# agents / headless (macOS):\n"
            f"cd {clone}/{r['dir']}\n"
            f"swift run {r['cliTarget']} --model {model_id} {r['cliArgs']}\n"
            "```\n")

    if doors["snippet"]:
        sizes = " / ".join(
            f"{v['sizeMB'] / 1000:.1f} GB ({top['variantLabels'][k]})"
            for k, v in entry["variants"].items())
        # FM-provider pitch (chat-family cards): cards.json `fmProvider` names the
        # provider type (KitLanguageModel / KitGemmaModel); wording is fixed here.
        fm_note = ""
        if cfg.get("fmProvider"):
            fm_note = (
                "**When Apple's FoundationModels built-in model isn't enough, keep your "
                "session code and swap the model — one line.** CoreAIKit's "
                f"[`{cfg['fmProvider']}`]({kit_url}#when-foundationmodels-isnt-enough) "
                "plugs this bundle into the same system `LanguageModelSession`; your "
                "`Tool`s, `@Generable` types and transcripts work unchanged, and "
                "capabilities (tool calling, guided generation) auto-detect per model.\n"
                "\n")
        parts.append(
            "💻 **Build with it** — complete; the glue is kit API, copy-paste runs:\n"
            "\n"
            "```swift\n"
            f"{snippet}\n"
            "```\n"
            "\n"
            + fm_note +
            f"The take-home is [`{cfg['quickstart']}`]({kit_url}/blob/main/{cfg['quickstart']})\n"
            + cfg.get("takeHomeTagline",
                      "— this exact code as one typed function, no UI; both the runner's GUI "
                      "and its CLI call it.") + "\n"
            f"{cfg['takeHomeNote']}\n"
            "\n"
            "**Integration checklist**\n"
            "\n"
            f"- SPM: `{kit_url}` → product **{cfg['product']}**\n"
            f"- {cfg['checklistInfoPlist']}\n"
            f"- {cfg['checklistEntitlements']}\n"
            f"- First run downloads the model — {sizes} — then it loads from the\n"
            "  local cache (Application Support; progress via the `downloadProgress` callback)\n"
            "- Measure in Release — Debug is ~3× slower on per-token host work")

    return "\n".join(parts)


def replace_region(text: str, model_id: str, block: str, where: str) -> str | None:
    begin = BEGIN_FMT.format(id=model_id)
    lines = text.split("\n")
    try:
        b = lines.index(begin)
        e = lines.index(END_MARK, b)
    except ValueError:
        fail(f"{where}: gen-cards markers for {model_id} not found — enrollment adds them "
             f"by hand once (begin marker + end marker around the Use-it block)")
        return None
    return "\n".join(lines[: b + 1] + block.split("\n") + lines[e:])


def show_diff(old: str, new: str, label: str) -> bool:
    if old == new:
        log("ok", f"{label}: clean (byte-identical)")
        return False
    print(f"\n--- {label}: DRIFT ---")
    for l in difflib.unified_diff(old.split("\n"), new.split("\n"),
                                  fromfile=f"{label} (current)", tofile=f"{label} (generated)",
                                  lineterm=""):
        print(l)
    print()
    return True


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default=str(ZOO.parent / "coreai-kit"),
                    help="path to the coreai-kit checkout (default: sibling ../coreai-kit)")
    ap.add_argument("--write", action="store_true", help="apply changes to zoo card files")
    ap.add_argument("--push", action="store_true", help="upload updated HF READMEs (needs hf auth)")
    ap.add_argument("--skip-builds", action="store_true",
                    help="skip build gates (template iteration only; no --write/--push)")
    ap.add_argument("--local-kit-dep", action="store_true",
                    help="scratch snippet compile uses a path dep instead of the url dep")
    ap.add_argument("ids", nargs="*", help="model ids to process (default: all enrolled)")
    args = ap.parse_args()

    if args.skip_builds and (args.write or args.push):
        print("refusing --write/--push with --skip-builds: doors would be unverified",
              file=sys.stderr)
        return 2

    kit = Path(args.kit).expanduser().resolve()
    top = json.loads((HERE / "cards.json").read_text())
    catalog = {e["id"]: e for e in json.loads((kit / "catalog.json").read_text())["models"]}
    ids = args.ids or list(top["models"].keys())
    OUT.mkdir(exist_ok=True)
    drift = False

    for model_id in ids:
        cfg = top["models"][model_id]
        if model_id in catalog:
            entry = catalog[model_id]
        else:
            # Engine-showcase models live outside the kit catalog (no generic runner
            # drives them); the sidecar names the repo directly.
            entry = {"id": model_id, "name": cfg["name"], "repo": cfg["repo"], "variants": {}}
        log("model", f"{model_id} ({entry['name']})")

        doors: dict = {"green": None, "runner": False, "snippet": False, "hero": None,
                       "appDoor": False, "op": False}
        snippet = None

        if "appDoor" in cfg:
            app_path = ZOO / cfg["appDoor"]["path"]
            if app_path.is_dir():
                doors["appDoor"] = True
                log("gate", f"zoo app exists: {cfg['appDoor']['path']}")
            else:
                fail(f"{model_id}: appDoor path {cfg['appDoor']['path']} not found in the zoo")
        elif args.skip_builds:
            log("warn", "--skip-builds: emitting doors WITHOUT verification (dry-run only)")
            doors["runner"] = True
            snippet = extract_snippet(kit, cfg, model_id)
            doors["snippet"] = snippet is not None
            doors["op"] = bool(cfg.get("op"))
        else:
            runner_dir = kit / cfg["runner"]["dir"]
            ok = gate_regen_diff(kit, runner_dir, cfg["runner"]["xcodeproj"])
            ok = gate_cli_build(runner_dir) and ok
            ok = gate_app_build(runner_dir, cfg["runner"]["xcodeproj"],
                                cfg["runner"]["scheme"]) and ok
            doors["runner"] = ok
            snippet = extract_snippet(kit, cfg, model_id)
            doors["snippet"] = snippet is not None and gate_snippet_compiles(
                snippet, top["kitURL"], kit if args.local_kit_dep else None,
                cfg.get("snippetFreeVars", {"url": "URL"}),
                product=cfg.get("product", "CoreAIKit"))
            if cfg.get("op"):
                o = cfg["op"]
                # CoreGraphics: CGImage free vars (upscale / estimateDepth) need the type.
                op_line = ("import CoreAIOps\nimport CoreGraphics\n\n"
                           f"let {o['result']} = try await {o['call']}")
                doors["op"] = gate_snippet_compiles(
                    op_line, top["kitURL"], kit if args.local_kit_dep else None,
                    o.get("freeVars", {}), product="CoreAIOps")

        if cfg.get("testflightURL") or cfg.get("dmgURL"):
            fail("green door configured but its template is not implemented yet")
        else:
            log("info", "🟢 door: no TestFlight/dmg link configured — omitted")

        doors["hero"] = hero_media(entry["repo"])
        if not doors["hero"]:
            log("info", "hero: no demo.gif/png in the HF repo — omitted (capture is demand-driven)")

        block = render_block(model_id, entry, cfg, top, snippet or "", doors)

        # zoo surface (optional — official models have no zoo/<id>.md, HF surface only)
        if cfg.get("zooCard") is None:
            log("info", "no zooCard configured — zoo surface skipped (HF-only enrollment)")
        else:
            zoo_card = ZOO / cfg["zooCard"]
            old = zoo_card.read_text()
            new = replace_region(old, model_id, block, str(cfg["zooCard"]))
            if new is not None:
                if show_diff(old, new, f"zoo:{cfg['zooCard']}"):
                    drift = True
                    if args.write:
                        zoo_card.write_text(new)
                        log("write", f"updated {cfg['zooCard']}")

        # HF surface
        try:
            hf_old = urllib.request.urlopen(
                f"https://huggingface.co/{entry['repo']}/raw/main/README.md",
                timeout=30).read().decode()
        except Exception as e:
            fail(f"could not fetch HF README for {entry['repo']}: {e}")
            continue
        hf_new = replace_region(hf_old, model_id, block, f"HF:{entry['repo']}")
        if hf_new is None:
            continue
        out_path = OUT / f"{model_id}_hf_README.md"
        out_path.write_text(hf_new)
        if show_diff(hf_old, hf_new, f"HF:{entry['repo']}"):
            drift = True
            if args.push:
                r = run(["hf", "upload", entry["repo"], str(out_path), "README.md",
                         "--commit-message", "gen-cards: regenerate Use-it block"])
                if r.returncode != 0:
                    fail(f"hf upload failed: {r.stderr[-500:]}")
                else:
                    log("push", f"HF README updated: {entry['repo']}")

    log("info", f"zoo README model-table regen: not implemented ({len(ids)} enrolled) — "
        "Run-in-app cells are still hand-maintained")

    if failures:
        print(f"\n{'!' * 70}\n{len(failures)} gate failure(s) — doors were dropped or output "
              f"is unverified; DO NOT ship this state silently:\n" +
              "\n".join(f"  - {f[:200]}" for f in failures) + f"\n{'!' * 70}")
        return 1
    if drift and not args.write:
        print("\ndrift detected (dry-run) — rerun with --write to apply")
        return 1
    print("\nall cards clean" + (" / applied" if args.write else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
