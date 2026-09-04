#!/usr/bin/env python3
"""Cut a coreai-cli release in one command — the day the release OS lands.

    python3 cli/release.py 0.2.0 --ios-build 24A353            # the real thing
    python3 cli/release.py 0.2.0 --ios-build 24A353 --dry-run  # show every step, change nothing
    python3 cli/release.py 0.2.0 --allow-beta                  # stamp a seed build anyway

In order, stopping at the first failure:

  1. regenerate coreai_zoo_routes.py from models/*/recipe.toml (make_zoo_routes.py)
  2. python3 selftest.py must pass
  3. read the host: sw_vers, Xcode, coreai-build, the runtime wheels — and refuse to stamp
     an Apple seed build (26A5xxx) as "validated" unless --allow-beta says so
  4. CHANGELOG.md: retitle "## [Unreleased]" to the version and date, fill "Validated on:"
  5. README.md: the same validated-on line, between the <!-- validated-on --> markers
  6. pyproject.toml: bump the version
  7. python3 -m build, then twine check
  8. print the commit and upload lines — nothing is committed, tagged or uploaded here

The upload is a public act and stays a human's: the last line printed is the exact
`twine upload` to run. `--upload` runs it for you, interactively, once you have read the
summary above it.
"""

from __future__ import annotations

import argparse
import datetime
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

BETA_BUILD = re.compile(r"^\d+[A-Z]5\d{3}[a-z]?$")   # Apple seeds: 26A5406e; release: 26A353


def sh(args: list[str], cwd: Path | None = None, check: bool = True) -> str:
    res = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if check and res.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(args)}\n{res.stdout}{res.stderr}")
    return res.stdout.strip()


def host_line(ios_build: str | None, python: str) -> tuple[str, bool]:
    """'macOS 27.0 (26A353) · iOS 27.0 (24A353) · Xcode 27.0 (27A...) · coreai-build ...'
    plus whether the macOS build is a seed."""
    mac_ver = sh(["sw_vers", "-productVersion"])
    mac_build = sh(["sw_vers", "-buildVersion"])
    xcode = " ".join(sh(["xcodebuild", "-version"], check=False).split("\n")[:2])
    xcode = re.sub(r"Xcode (\S+) Build version (\S+)", r"Xcode \1 (\2)", xcode) or "Xcode ?"
    cb = re.search(r"coreai-build\s+([\d.]+)",
                   sh(["xcrun", "coreai-build", "--version"], check=False) or "")
    wheels = sh([python, "-c",
                 "import importlib.metadata as m; "
                 "print(' / '.join(f'{p} {m.version(p)}' for p in ('coreai-core', 'coreai-torch')))"],
                check=False) or "runtime wheels not found"
    parts = [f"macOS {mac_ver} ({mac_build})"]
    if ios_build:
        parts.append(f"iOS 27 ({ios_build})")
    parts += [xcode, f"coreai-build {cb.group(1) if cb else '?'}", wheels]
    return " · ".join(parts), bool(BETA_BUILD.match(mac_build)) or bool(
        ios_build and BETA_BUILD.match(ios_build))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version", help="the version to release, e.g. 0.2.0")
    ap.add_argument("--ios-build", help="the iOS build the phone gates ran on, e.g. 24A353")
    ap.add_argument("--allow-beta", action="store_true",
                    help="stamp an Apple seed build as validated (the default refuses)")
    ap.add_argument("--dry-run", action="store_true", help="print every step, change nothing")
    ap.add_argument("--upload", action="store_true", help="run twine upload at the end")
    ap.add_argument("--python", help="interpreter with the coreai runtime (for the wheel versions)")
    args = ap.parse_args()

    if not re.match(r"^\d+\.\d+\.\d+$", args.version):
        ap.error("version must look like 0.2.0")
    from coreai_export import resolve_python  # the same venv resolution export uses
    python = resolve_python(args.python)
    today = datetime.date.today().isoformat()
    changelog, readme, pyproject = HERE / "CHANGELOG.md", HERE / "README.md", HERE / "pyproject.toml"

    # 0. state of the tree — a release from a dirty cli/ ships something the log does not name
    dirty = sh(["git", "status", "--porcelain", "--", str(HERE)], cwd=HERE, check=False)
    if dirty:
        print("note: uncommitted changes under cli/ — the wheel is built from the working tree:\n"
              + "\n".join("    " + line for line in dirty.splitlines()))

    # 1. routes snapshot
    print("1. regenerating coreai_zoo_routes.py")
    if not args.dry_run:
        import make_zoo_routes
        make_zoo_routes.main()
    else:
        print("   (dry run) python3 cli/make_zoo_routes.py")

    # 2. self-test
    print("2. selftest.py")
    out = sh([sys.executable, str(HERE / "selftest.py")], cwd=HERE, check=False)
    print("   " + out.splitlines()[-1])
    if "all pass" not in out:
        raise SystemExit("selftest failed — no release")

    # 3. the host
    line, seed = host_line(args.ios_build, python)
    print(f"3. validated on: {line}")
    if seed and not args.allow_beta:
        raise SystemExit("   that is an Apple SEED build, not a release build. A release that says "
                         "'validated on macOS 27' must be validated on the release OS. Install it, "
                         "or pass --allow-beta to stamp a beta on purpose.")
    if not args.ios_build:
        print("   no --ios-build: the line names macOS only. Pass the phone's build if the device "
              "gates ran on it.")

    # 4. CHANGELOG
    text = changelog.read_text()
    if "## [Unreleased]" not in text:
        raise SystemExit("CHANGELOG.md has no '## [Unreleased]' section to release")
    new = text.replace("## [Unreleased]", f"## [{args.version}] — {today}", 1)
    new = re.sub(r"Validated on: _stamped by `cli/release.py` at release time_",
                 f"Validated on: {line}", new, count=1)
    if new == text.replace("## [Unreleased]", f"## [{args.version}] — {today}", 1):
        raise SystemExit("CHANGELOG.md: the Unreleased section has no validated-on placeholder")
    print(f"4. CHANGELOG.md: [Unreleased] -> [{args.version}] — {today}")
    if not args.dry_run:
        changelog.write_text(new)

    # 5. README
    text = readme.read_text()
    marker = re.compile(r"(<!-- validated-on -->).*?(<!-- /validated-on -->)", re.S)
    if not marker.search(text):
        raise SystemExit("README.md has no <!-- validated-on --> markers")
    new = marker.sub(rf"\1Validated on {line}.\2", text, count=1)
    print("5. README.md validated-on line updated")
    if not args.dry_run:
        readme.write_text(new)

    # 6. version
    text = pyproject.read_text()
    old = re.search(r'^version = "([^"]+)"', text, re.M)
    if not old:
        raise SystemExit("pyproject.toml has no version line")
    print(f"6. pyproject.toml: version {old.group(1)} -> {args.version}")
    if not args.dry_run:
        pyproject.write_text(text.replace(old.group(0), f'version = "{args.version}"', 1))

    # 7. build + check
    print("7. python3 -m build && twine check")
    if not args.dry_run:
        shutil.rmtree(HERE / "dist", ignore_errors=True)
        sh([sys.executable, "-m", "build", "--outdir", str(HERE / "dist"), str(HERE)], cwd=HERE)
        print("   " + sh([sys.executable, "-m", "twine", "check", *map(str, (HERE / "dist").glob("*"))],
                         cwd=HERE).replace("\n", "\n   "))
        built = sorted(p.name for p in (HERE / "dist").glob("*"))
        print("   built: " + ", ".join(built))

    # 8. the human's part
    files = "cli/CHANGELOG.md cli/README.md cli/pyproject.toml cli/coreai_zoo_routes.py"
    print(f"""
8. next — these are yours:
   git add {files} && git commit -m "coreai-cli {args.version}: validated on the release OS"
   git tag cli-v{args.version}
   python3 -m twine upload cli/dist/coreai_cli-{args.version}*
   then: pip install -U coreai-cli && coreai --version   # must print coreai-cli {args.version}""")
    if args.upload and not args.dry_run:
        subprocess.run([sys.executable, "-m", "twine", "upload",
                        *map(str, (HERE / "dist").glob(f"coreai_cli-{args.version}*"))], cwd=HERE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
