# Changelog — coreai-cli

The PyPI package `coreai-cli`: `coreai export | doctor | verify | eval`. Versions follow
the wheel, not the zoo. Each release names the OS and toolchain it was validated on,
because the runtime is versioned and a claim without a build number is not a claim.

## [Unreleased]

Validated on: _stamped by `cli/release.py` at release time_

### doctor

- `IR-040-DEBUG-LOC` (0.4.0-era IR, no producer stamp) now reports by the **host OS
  build**: `info` on OS 27 beta 1, which loads it, `fatal` on every build from beta 2 on.
  Apple's beta 5 note lists the incident (177008303) as fixed; a load on 26A5416b and a
  `coreai-build compile` on the same asset say otherwise, so the rule follows the
  measurement — `IR040_MEASURED_OK_FROM` flips it for builds a sweep measures, and a
  release build inherits nothing until then (see `DEVELOPMENT.md`).
- `AOTC-STALE-TOOLCHAIN` is `fatal` only for an artifact compiled by a beta-2-or-earlier
  `coreai-build` (below `3600.75.3`, the 181264112 class) and `info` for one merely older
  than the installed toolchain — it flagged every published h18p artifact as fatal before.
- `--host-build 26A5353q` judges the build-conditional rule against another build. The
  report prints the host build it judged against; `--json` carries `host_build` and both
  the effective and the rule severity per finding.
- Self-test covers the build comparator (seed vs release builds, trains, majors), the
  severity of both rules, and the flip once a build is measured.

### export

- Zoo routes snapshot regenerated from `models/*/recipe.toml` — the date at the top of
  `coreai_zoo_routes.py` is the snapshot; a live checkout always wins over it.

### packaging

- `cli/release.py` — the release in one command: regenerate the routes snapshot, run the
  self-test, stamp the validated-on line from the host (`sw_vers`, Xcode, `coreai-build`,
  the runtime wheels, `--ios-build` for the phone), bump the version, build and check the
  wheel, print the upload line. It refuses to stamp a beta build without `--allow-beta`.
- The PyPI page (`README.md`) is now the user's document only: install, the four commands,
  what each answers and refuses. The engineering record — validation logs, the design
  rationale, the incidents each command was built from — moved to `DEVELOPMENT.md`.
- `Changelog` project URL on PyPI.

## [0.1.0] — 2026-08-15

Validated on: macOS 27.0 beta (26A5378j) · Xcode 27.0 beta 3 · coreai-build 3600.75.3 ·
coreai-core 1.0.0b2 / coreai-torch 0.4.1

- First release. `coreai export` (router: Apple presets, generic `model_type` routing, the
  zoo's recorded recipes, and the iOS cliff stated up front), `coreai doctor` (45 of 64
  documented failure patterns implemented), `coreai verify` (token-exact gate with the
  prompt-margin rule), `coreai eval` (task scoring that refuses to compare arms run under
  different protocols).
- The wheel carries a dated snapshot of the zoo's routes so `export` answers without a
  checkout; converting needs Apple's `coreai_models`, running a zoo recipe needs the zoo.
