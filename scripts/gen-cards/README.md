# gen-cards — the "Use it" block generator (and honesty enforcer)

Renders each enrolled model's **Use it** block from `../coreai-kit/catalog.json` +
`cards.json` (sidecar), and keeps both card surfaces **byte-identical**:

- `zoo/<model>.md` — between `<!-- gen-cards:use-it begin/end -->` markers
- the HF README of the model's repo — same markers, fetched live, diffed, `--push` to upload

Design: [`_ZOO_DX_DESIGN.md` §6.2–6.3] — "cards never lie", enforced:

| Emitted | Only if |
|---|---|
| ▶️ runner door | the runner smoke-builds **in this run** (`swift build` + `xcodebuild`) and the committed xcodeproj matches fresh `xcodegen generate` output (lockfile guard) |
| 💻 snippet | extracted from the runner's `QuickStart.swift` `CARD-SNIPPET` markers (never hand-written), then compiled standalone in a scratch package against kit as a **url-dep** (catches non-public API / missing imports) |
| 🟢 app door | a TestFlight/dmg link is configured (none yet) |
| hero `demo.gif` | the file exists in the model's HF repo (HEAD-check); missing = info, not failure |
| **Measured decode** line | the model's HF repo maps to a [DeviceMark](https://devicemark.github.io/) row (`cards.json` → `devicemark.rows`) and the board's data files carry a decode measurement for it; a device without one is not mentioned |

A failed gate **drops the door and fails the run loudly** (nonzero exit) — a kit regression
cannot silently strip doors across the cards.

## DeviceMark rows — the measured numbers on the cards

Two surfaces carry the board's numbers, both rendered by this generator from the board's own
data files (`board.json` + `measurements.jsonl`, the rows the site renders and the HF dataset
`devicemark/results` publishes; default location `~/code/devicemark/data/leaderboard`,
`--devicemark DIR` otherwise; a run without the files refuses to render):

1. **The measured line inside the Use-it block** (zoo card and HF README, byte-identical):
   `**Measured decode** — iPhone 17 Pro: 29 tok/s · Mac (M4 Max): 165 tok/s (DeviceMark row …)`.
   Only for a model whose HF repo is in `devicemark.rows`; the numbers are the `coreai`
   measurements for that artifact, cross-checked against the board's iPhone/Mac columns (a
   disagreement between the two files is a gate failure).
2. **The `gen-cards:devicemark` block on every own HF card** — the board's badge markup
   (`[![DeviceMark](https://devicemark.github.io/badge/<slug>.svg)](https://devicemark.github.io/)`,
   what the site's footer asks vendors to embed) plus the same measured line; a repo with no
   row gets one sentence linking the board and no numbers. `tools/devicemark_row.py` puts the
   block on all own `.aimodel` repos (enrolled here or not), right after the card's definition
   sentence; when an HF README already has the block, this generator regenerates it too, so
   the two writers cannot disagree.

`devicemark.rows` is an explicit map, HF repo → board `artifact_id`, because the board does not
carry repo ids and the model names differ (`nemotron-4b` is `Nemotron-3-Nano-4B-CoreAI`). A
mapped artifact must be an `aimodel` row on the `coreai` runtime with a measurement; a board
row that no repo maps to is a warning. The Gemma 4 E2B row is LiteRT-LM, not the Core AI
bundle, so `gemma-4-E2B-CoreAI` is deliberately unmapped.

Refreshing after a new measurement drop (GM day): sync the board's data files, then

```bash
DEVELOPMENT_TEAM=<team id> python3 scripts/gen-cards/gen_cards.py --kit ~/code/coreai-kit --write --push \
  && python3 tools/devicemark_row.py --go
```

(`gen_cards` rewrites both blocks on the enrolled cards; the tool covers the rest of the own
repos and skips cards that are already right.) Run them one after the other, never at the
same time: `gen_cards --push` uploads the whole README it fetched at the start of that model's
turn, so a tool commit landing in between would be overwritten.

## Usage

**Export `DEVELOPMENT_TEAM` first.** The lockfile guard regenerates each Example's project with
`xcodegen` and diffs it against the committed one. The committed projects were generated with a
team id in the environment, so a run without it regenerates `DEVELOPMENT_TEAM = "${DEVELOPMENT_TEAM}"`
instead of the literal id and every project "differs" — 44 gate failures, all false, all six lines
of signing. Measured 2026-07-25: same run with `DEVELOPMENT_TEAM=<team id>` exported = 1 failure
(an external repo's missing markers) and 79 clean surfaces.

```bash
DEVELOPMENT_TEAM=<team id> python3 scripts/gen-cards/gen_cards.py --kit ~/code/coreai-kit  # verify (dry-run)
python3 scripts/gen-cards/gen_cards.py --kit ~/code/coreai-kit --write        # apply zoo cards
python3 scripts/gen-cards/gen_cards.py --kit ~/code/coreai-kit --write --push # + HF upload
```

Exit 0 = all cards clean. Exit 1 = drift (rerun with `--write`) or a gate failure.

**Drift here is not only template drift.** The block quotes `catalog.json` — sizes above
all — so a card goes stale the moment the kit corrects one and nobody regenerates. Run
2026-08-25 (a one-word template change) turned up five cards advertising a first-run
download the catalog no longer agreed with, one of them by almost 2x. Worth a periodic
full run even when no template changed.

`--skip-builds` iterates templates without gates and refuses `--write`/`--push`.

## Enrolling a model

1. Its runner needs `Sources/QuickStart.swift` with the `CARD-SNIPPET-BEGIN/END` block
   (see Examples/Transcribe — the convention: one typed function, no UI).
2. Add the model to `cards.json` (runner, product, checklist lines, take-home note).
3. Add the begin/end markers around the card's Use-it section by hand once, both surfaces.
   Models without a `zoo/<id>.md` (official ports): omit `zooCard` — the generator skips
   the zoo surface and manages the HF README only.
4. Run the generator; it owns the block from then on.

Snippet specialization is deliberately minimal and compile-guarded: dedent, `catalog: id` →
the literal id, strip comma-prefixed forwarded args (`, x: x`), `return` → `let result =`.
