#!/usr/bin/env python3
"""Tier-0 verification: does a published bundle still load on THIS OS and toolchain?

`zoo_verify.py` (tier 1) asks whether a bundle agrees with the model it came from, and
needs no weights. This asks the question an OS update makes urgent: the catalog was
converted and gated on one generation of the runtime, and the loader is versioned. The
coreai-torch 0.4.0 incident (2026-07-18) broke every published bundle at `AIModel.load`
on OS 27 beta 2, and nothing in this repository could see it until someone loaded one.
So this loads them, and writes down which OS build did.

Three stages per bundle, each recorded with the host that ran it:

    stamp    read the asset's own metadata.json off the Hub (no download). A 0.4.0-era
             asset carries no `producer`; a 0.4.1+ one says `coreai-core 1.0.0b2`.
    load     download the bundle and load every .aimodel in it in a child interpreter
             that has the coreai runtime. cpu_only by default — the parity option, and it
             does not contend for the GPU. A broken asset ABORTS the child (LLVM ERROR),
             it does not raise, which is why this is a subprocess.
    verify   the tier-1 checks from zoo_verify.py, so one command answers both questions.

    python3 conversion/zoo_smoke.py --top 20              # the sweep a release day needs
    python3 conversion/zoo_smoke.py --top 20 --dry-run    # the plan: bundles, sizes, skips
    python3 conversion/zoo_smoke.py mlboydaisuke/LFM2.5-2.6B-CoreAI
    python3 conversion/zoo_smoke.py --all                 # the whole catalog: days, not hours
    python3 conversion/zoo_smoke.py --all --stamp-only    # producer audit only: minutes
    python3 conversion/zoo_smoke.py --badge out.json      # shields endpoint for the README

Results merge into models/_SMOKE.json keyed by (repo, bundle). A bundle not run this time
keeps its previous entry, so the sweep can be STAGED — the most-downloaded bundles the day
the release lands, the rest over the following week — and the record says which build
checked which bundle. The badge counts only bundles checked on the current host build, so
a new OS build starts the count from zero rather than inheriting last month's green.

Two things it refuses to do on a Mac, on purpose: load a compiled `.aimodelc` (device- and
architecture-specific; the AOT twin of a bundle that loads is checked on the device) and
load anything published under an `ios` path — running an iOS bundle on a Mac wedges the GPU
stack and costs a reboot (AGENTS.md). Both are recorded as `device-only`, not as passes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path

# huggingface_hub reads these at import time, so they go before any import of it.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")            # plain HTTP is faster here
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # a log, not a terminal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hf_catalog import Catalog, bundles_of, repo_format  # noqa: E402
from _paths import gpu_lock, work_path  # noqa: E402
import zoo_verify  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
AUTHORS = zoo_verify.AUTHORS
SMOKE_JSON = MODELS / "_SMOKE.json"
IOS_TOKENS = ("ios", "iphone", "h18p", "h17p", "h16p")
GB = 1e9

# The loader-side failure signatures this exists to catch. A match names the incident;
# anything else is reported verbatim.
SIGNATURES = (
    ("ir-incompatible", re.compile(r"versioned IR|AICode versioned location|odiec_module_t")),
    ("specialization", re.compile(r"failedToSpecialize|[Ss]pecializ")),
    ("memory", re.compile(r"bad_alloc|mmap|No space left")),
)

LOAD_CHILD = r'''
import asyncio, json, sys, time
from pathlib import Path
import coreai.runtime as rt

asset, compute = sys.argv[1], sys.argv[2]

async def main():
    t = time.time()
    if compute == "cpu":
        opts = rt.SpecializationOptions.cpu_only()
    else:
        opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    m = await rt.AIModel.load(Path(asset), opts)
    print("<<<JSON>>>" + json.dumps({"functions": list(m.function_names),
                                     "seconds": round(time.time() - t, 1)}))

asyncio.run(main())
'''


# --------------------------------------------------------------------------
# host


def sh(args: list[str], timeout: int = 60) -> str:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def resolve_python(flag: str | None) -> str:
    """Same order as zoo_convert.py / cli: flag > $ZOO_CONVERT_PYTHON > sibling .venv > PATH."""
    if flag:
        return flag
    if env := os.environ.get("ZOO_CONVERT_PYTHON"):
        return env
    sibling = REPO.parent / "coreai-models" / ".venv" / "bin" / "python"
    if sibling.exists():
        return str(sibling)
    return shutil.which("python3") or "python3"


def host_facts(python: str) -> dict:
    wheels = sh([python, "-c",
                 "import importlib.metadata as m; "
                 "print(' '.join(f'{p} {m.version(p)}' for p in ('coreai-core', 'coreai-torch')))"])
    return {
        "os": sh(["sw_vers", "-productName"]) + " " + sh(["sw_vers", "-productVersion"]),
        "os_build": sh(["sw_vers", "-buildVersion"]) or None,
        "coreai_build": (re.search(r"coreai-build\s+([\d.]+)", sh(["xcrun", "coreai-build", "--version"]))
                         or [None, None])[1],
        "wheels": wheels or None,
        "python": python,
    }


def is_seed(build: str | None) -> bool | None:
    """Apple seeds carry a 4-digit build number that starts with 5 (26A5406e); release
    builds do not (26A353). None when the build cannot be read."""
    m = re.match(r"^\d+[A-Z](\d+)[a-z]?$", build or "")
    if not m:
        return None
    return len(m.group(1)) >= 4 and m.group(1).startswith("5")


# --------------------------------------------------------------------------
# selection


def index_recipes() -> tuple[dict[str, list[str]], str | None]:
    """repo id -> the bundle paths its recipes name as *the* published configuration."""
    path = MODELS / "index.json"
    if not path.exists():
        return {}, None
    data = json.loads(path.read_text())
    out: dict[str, list[str]] = {}
    for family in data.get("models", []):
        for r in family.get("recipes", []):
            if r.get("hf_repo") and r.get("bundle"):
                out.setdefault(r["hf_repo"], []).append(r["bundle"])
    return out, data.get("generated")


def kit_paths() -> dict[str, set[str]]:
    """repo id -> bundle paths the CoreAIKit catalog serves, when the kit is checked out."""
    out: dict[str, set[str]] = {}
    for base in (REPO.parent / "coreai-kit", REPO.parent.parent / "coreai-kit"):
        path = base / "catalog.json"
        if not path.exists():
            continue
        models = json.loads(path.read_text()).get("models", [])
        for m in (models.values() if isinstance(models, dict) else models):
            repo = m.get("repo") or m.get("hf")
            for v in (m.get("variants") or {}).values():
                if repo and v.get("path"):
                    out.setdefault(repo, set()).add(v["path"].rstrip("/"))
        break
    return out


def device_only(bundle: str) -> str | None:
    """Why a bundle must not be loaded on a Mac, or None."""
    if bundle.endswith(".aimodelc"):
        return "compiled (.aimodelc) — device/arch-specific"
    parts = re.split(r"[/_.-]", bundle.lower())
    if any(p in IOS_TOKENS for p in parts) or any(p.endswith("ios") for p in parts):
        return "published under an iOS path — never load an iOS bundle on a Mac"
    return None


def pick_bundles(repo: str, published: list[str], recipe: dict[str, list[str]],
                 kit: dict[str, set[str]], everything: bool) -> list[str]:
    """Which of a repo's bundles the sweep loads: the ones a recipe or the kit names as
    *the* shipped configuration; every bundle when nothing names one, or with --all."""
    if everything:
        return published
    named: list[str] = []
    for want in recipe.get(repo, []) + sorted(kit.get(repo, set())):
        for b in published:
            if b == want or b.startswith(want + "/") or want.startswith(b + "/"):
                if b not in named:
                    named.append(b)
    return named or published


def asset_dirs(bundle: str, files: list[str]) -> list[str]:
    """The .aimodel / .aimodelc directories inside a bundle, from the repo file list."""
    if bundle.endswith((".aimodel", ".aimodelc")):
        return [bundle]
    out = set()
    for f in files:
        if not f.startswith(bundle + "/"):
            continue
        for ext in (".aimodel/", ".aimodelc/"):
            if ext in f:
                out.add(f[: f.index(ext) + len(ext) - 1])
    return sorted(out)


def bundle_size(bundle: str, blobs: dict | None) -> int | None:
    if not blobs:
        return None
    total = 0
    for s in blobs.get("siblings", []):
        f = s.get("rfilename", "")
        if f == bundle or f.startswith(bundle + "/"):
            total += s.get("size") or 0
    return total


# --------------------------------------------------------------------------
# stages


def stamp(cat: Catalog, repo: str, rev: str, bundle: str, files: list[str]) -> dict:
    """Producer per asset, read off the Hub. `ir` summarises the bundle: the worst asset wins."""
    assets = {}
    for a in asset_dirs(bundle, files):
        meta = cat.json_file(repo, a + "/metadata.json", rev) or {}
        assets[a] = meta.get("producer")
    if not assets:
        ir = "unknown"
    elif any(p is None for p in assets.values()):
        ir = "0.4.0-era (no producer stamp)"
    elif all(str(p).startswith("coreai-build") for p in assets.values()):
        ir = "compiled"
    else:
        ir = sorted({str(p) for p in assets.values() if p})[0]
    return {"assets": assets, "ir": ir}


def download(repo: str, rev: str, bundle: str, root: Path) -> Path:
    from huggingface_hub import snapshot_download  # lazy: the stamp-only path needs no hf_hub

    local = root / repo.replace("/", "__")
    snapshot_download(repo, revision=rev, allow_patterns=[f"{bundle}/*"], local_dir=str(local))
    return local / bundle


def load_asset(python: str, asset: Path, compute: str, timeout: int) -> dict:
    with tempfile.NamedTemporaryFile("w", prefix="zoo_smoke_", suffix=".py", delete=False) as f:
        f.write(LOAD_CHILD)
        script = f.name
    t = time.time()
    try:
        res = subprocess.run([python, script, str(asset), compute],
                             capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "seconds": round(time.time() - t), "detail": f"> {timeout}s"}
    finally:
        os.unlink(script)
    for line in res.stdout.splitlines():
        if line.startswith("<<<JSON>>>"):
            info = json.loads(line[len("<<<JSON>>>"):])
            return {"status": "ok", "seconds": info["seconds"], "functions": info["functions"]}
    tail = "\n".join((res.stderr.strip() or res.stdout.strip()).splitlines()[-4:])
    kind = next((k for k, rx in SIGNATURES if rx.search(tail)), "error")
    how = f"signal {-res.returncode}" if res.returncode < 0 else f"exit {res.returncode}"
    return {"status": kind, "seconds": round(time.time() - t), "detail": f"{how}: {tail}"}


# --------------------------------------------------------------------------
# records


def load_smoke(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"generated": None, "bundles": []}


def save_smoke(path: Path, data: dict, entries: list[dict]) -> None:
    keyed = {(b["repo"], b["bundle"]): b for b in data["bundles"]}
    for e in entries:
        keyed[(e["repo"], e["bundle"])] = e
    data["generated"] = date.today().isoformat()
    data["bundles"] = [keyed[k] for k in sorted(keyed)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")


def badge(data: dict, host: dict, total: int) -> dict:
    """shields.io endpoint JSON. Counts only what THIS build checked."""
    build = host.get("os_build") or "unknown build"
    on_build = [b for b in data["bundles"]
                if (b.get("host") or {}).get("os_build") == build and b.get("load")]
    ok = sum(1 for b in on_build if b["load"]["status"] == "ok")
    bad = sum(1 for b in on_build if b["load"]["status"] not in ("ok", "skipped", "deferred"))
    os_name = host.get("os", "macOS").strip()
    where = f"{os_name} ({build})" if build else os_name
    if is_seed(build):
        where = f"{where} beta"
    if bad:
        return {"schemaVersion": 1, "label": "GA validation", "color": "red",
                "message": f"{bad} failing · {ok}/{total} bundles load on {where}"}
    if ok < total:
        return {"schemaVersion": 1, "label": "GA validation", "color": "yellow",
                "message": f"in progress · {ok}/{total} bundles load on {where}"}
    return {"schemaVersion": 1, "label": "GA validation", "color": "brightgreen",
            "message": f"{ok}/{total} bundles load on {where}"}


def gb(n: int | None) -> str:
    return "?" if n is None else f"{n / GB:5.2f} GB"


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repos", nargs="*", help="Hugging Face repo ids")
    ap.add_argument("--top", type=int, metavar="N",
                    help="the N most-downloaded Core AI repos (30-day), shipped bundles only")
    ap.add_argument("--all", action="store_true", help="every bundle of every Core AI repo")
    ap.add_argument("--bundle", help="restrict to bundles whose path contains this substring")
    ap.add_argument("--stamp-only", action="store_true",
                    help="producer audit off the Hub; no download, no load, no verify")
    ap.add_argument("--no-verify", action="store_true", help="skip the tier-1 checks")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    ap.add_argument("--compute", choices=["cpu", "gpu"], default="cpu",
                    help="specialization for the load (gpu takes the machine-wide GPU lock)")
    ap.add_argument("--ignore-gpu-lock", action="store_true")
    ap.add_argument("--max-bundle-gb", type=float, default=12.0,
                    help="defer bundles larger than this (default 12; 0 = no limit)")
    ap.add_argument("--budget-gb", type=float, default=60.0,
                    help="defer once this much would have been downloaded (default 60; 0 = no limit)")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per asset load")
    ap.add_argument("--keep", action="store_true",
                    help="keep downloaded bundles after a passing load (failures are always kept)")
    ap.add_argument("--python", help="interpreter with the coreai runtime")
    ap.add_argument("--json", default=str(SMOKE_JSON), help="where results merge")
    ap.add_argument("--badge", help="write the shields endpoint JSON here")
    ap.add_argument("--offline", action="store_true", help="use only the local listing cache")
    args = ap.parse_args()

    if not (args.repos or args.top or args.all):
        ap.error("pass repo ids, --top N, or --all")
    out_json = Path(args.json)

    # A fresh listing every day: the permanent cache gen_inventory relies on would happily
    # report last month's revision, and a sweep exists to check what is published NOW.
    cat = Catalog(cache_dir=REPO / ".cache" / "hf-smoke" / date.today().isoformat(),
                  offline=args.offline)
    python = resolve_python(args.python)
    host = host_facts(python)
    recipe, generated = index_recipes()
    if generated and date.fromisoformat(generated) < date.today() - timedelta(days=14):
        print(f"note: models/index.json is from {generated} — the download ranking may be stale "
              f"(python3 scripts/gen_inventory.py refreshes it)")
    kit = kit_paths()

    # ---- which repos ------------------------------------------------------
    if args.repos:
        listing = [m for m in (cat.repo(r) for r in args.repos) if m]
    else:
        listing = [m for a in AUTHORS for m in cat.repos_by_author(a)]
        listing += [m for m in (cat.repo(r) for r in zoo_verify.contributor_repos()) if m]
    coreai = []
    for m in listing:
        files = [s["rfilename"] for s in m.get("siblings", [])]
        if repo_format(files) == "coreai" and bundles_of(files):
            coreai.append((m, files))
    coreai.sort(key=lambda x: (-(x[0].get("downloads") or 0), x[0]["id"].lower()))
    total_bundles = sum(len(bundles_of(f)) for _, f in coreai)
    if args.top:
        coreai = coreai[: args.top]

    # ---- the plan -----------------------------------------------------------
    plan: list[dict] = []
    spent = 0
    free = shutil.disk_usage(work_path()).free
    for m, files in coreai:
        rid, rev = m["id"], m.get("sha") or "main"
        blobs = None if args.stamp_only else cat.repo_blobs(rid)
        for b in pick_bundles(rid, bundles_of(files), recipe, kit, args.all):
            if args.bundle and args.bundle not in b:
                continue
            size = bundle_size(b, blobs)
            row = {"repo": rid, "revision": rev, "bundle": b, "dl30": m.get("downloads") or 0,
                   "size": size, "files": files, "action": "load"}
            if args.stamp_only:
                row["action"] = "stamp"
            elif reason := device_only(b):
                row["action"], row["why"] = "device-only", reason
            elif size and args.max_bundle_gb and size > args.max_bundle_gb * GB:
                row["action"], row["why"] = "deferred", f"> --max-bundle-gb {args.max_bundle_gb:g}"
            elif size and args.budget_gb and spent + size > args.budget_gb * GB:
                row["action"], row["why"] = "deferred", f"over --budget-gb {args.budget_gb:g}"
            elif size and size + 10 * GB > free - spent:
                row["action"], row["why"] = "deferred", "not enough free disk"
            else:
                spent += size or 0
            plan.append(row)

    print(f"host: {host['os']} {host['os_build']}  coreai-build {host['coreai_build']}  "
          f"{host['wheels']}\n")
    print(f"{'repo/bundle':78} {'30d DL':>6} {'size':>9}  action")
    for r in plan:
        name = f"{r['repo'].split('/')[-1]}/{r['bundle']}"
        why = f"  ({r['why']})" if r.get("why") else ""
        print(f"{name[:78]:78} {r['dl30']:6d} {gb(r['size']):>9}  {r['action']}{why}")
    n_load = sum(1 for r in plan if r["action"] == "load")
    print(f"\n{len(plan)} bundles from {len(coreai)} repos: {n_load} to load "
          f"({spent / GB:.1f} GB to download), "
          f"{sum(1 for r in plan if r['action'] == 'device-only')} device-only, "
          f"{sum(1 for r in plan if r['action'] == 'deferred')} deferred; "
          f"catalog total {total_bundles} bundles")
    if args.dry_run:
        return 0

    if args.compute == "gpu" and n_load:
        lock = gpu_lock()
        if lock.exists() and not args.ignore_gpu_lock:
            tag = lock.read_text().strip() or "(untagged — a 0-byte file left by a finished "
            "with-gpu-lock.py run reads as held; check for GPU jobs before ignoring it)"
            print(f"\nREFUSED: --compute gpu and {lock} is held: {tag}. "
                  f"The beta driver kernel-panics under parallel GPU load. Re-run when it clears, "
                  f"or pass --ignore-gpu-lock if you know it is stale.")
            return 2
        lock.write_text(f"zoo_smoke pid {os.getpid()} {datetime.now().isoformat(timespec='seconds')}\n")

    # ---- run ------------------------------------------------------------------
    root = work_path("_smoke_bundles")
    root.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    failures: list[str] = []
    try:
        for r in plan:
            rid, rev, b, files = r["repo"], r["revision"], r["bundle"], r["files"]
            print(f"\n== {rid} / {b}", flush=True)
            entry = {"repo": rid, "bundle": b, "revision": rev, "size": r["size"],
                     "date": date.today().isoformat(), "host": host}
            st = stamp(cat, rid, rev, b, files)
            entry["producer"] = st["assets"]
            entry["ir"] = st["ir"]
            print(f"   stamp   {st['ir']}")
            if st["ir"].startswith("0.4.0"):
                failures.append(f"{rid}/{b}: 0.4.0-era IR (no producer stamp)")

            if r["action"] in ("device-only", "deferred"):
                entry["load"] = {"status": "skipped" if r["action"] == "device-only" else "deferred",
                                 "detail": r["why"]}
                print(f"   load    {entry['load']['status']} — {r['why']}")
            elif r["action"] == "load":
                try:
                    local = download(rid, rev, b, root)
                except Exception as e:  # network, auth, disk — report, keep going
                    entry["load"] = {"status": "download-failed", "detail": str(e)[:300]}
                    failures.append(f"{rid}/{b}: download failed: {str(e)[:120]}")
                    print(f"   load    download failed: {str(e)[:200]}")
                    entries.append(entry)
                    continue
                per_asset = {}
                worst = "ok"
                for a in asset_dirs(b, files):
                    if a.endswith(".aimodelc"):
                        per_asset[a] = {"status": "skipped", "detail": "compiled twin — device"}
                        continue
                    res = load_asset(python, local / a[len(b) + 1:] if a != b else local,
                                     args.compute, args.timeout)
                    per_asset[a] = res
                    print(f"   load    {a.split('/')[-1]}: {res['status']}"
                          f"{' ' + str(res.get('seconds')) + 's' if res.get('seconds') else ''}"
                          f"{'  ' + res['detail'] if res.get('detail') else ''}")
                    if res["status"] != "ok":
                        worst = res["status"]
                entry["load"] = {"status": worst, "compute": args.compute, "assets": per_asset}
                if worst != "ok":
                    failures.append(f"{rid}/{b}: load {worst}")
                elif not args.keep:
                    shutil.rmtree(local, ignore_errors=True)

            if not args.no_verify and not args.stamp_only:
                rows = zoo_verify.verify_bundle(cat, rid, b, files, zoo_verify.load_expected(rid))
                v = zoo_verify.verdict(rows)
                entry["verify"] = v
                entry["verify_notes"] = [f"{c['check']}: {c['status']} {c['detail']}"
                                         for c in rows if c["status"] not in (zoo_verify.OK,)]
                print(f"   verify  {v}")
                if v == "FAIL":
                    failures.append(f"{rid}/{b}: tier-1 FAIL")
            entries.append(entry)
            # Save as we go: a sweep killed at bundle 12 of 20 keeps the 11 it measured.
            save_smoke(out_json, load_smoke(out_json), [entry])
            sys.stdout.flush()
    finally:
        if args.compute == "gpu" and n_load:
            gpu_lock().unlink(missing_ok=True)

    data = load_smoke(out_json)
    print(f"\nwrote {out_json} ({len(entries)} updated, {len(data['bundles'])} recorded)")
    if args.badge:
        payload = badge(data, host, total_bundles)
        Path(args.badge).write_text(json.dumps(payload) + "\n")
        print(f"badge: {payload['message']}  -> {args.badge}")
    if failures:
        print("\nFAILURES")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nno failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
