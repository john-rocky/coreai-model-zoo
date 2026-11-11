# Contributing a model port

External ports are welcome — the zoo's job is the **standard and the gate**, not a single
author's queue. A model you port ships to every CoreAIKit app as one line of Swift
(`ChatSession(catalog: "your-model")`), revision-pinned and nightly-verified, **with your
name on the card, the README row, and the release notes**.

## What gets accepted

Any model, any modality, if it clears three bars:

1. **License** — the upstream license must permit redistributing converted weights (Apache-2.0,
   MIT, BSD, most Gemma/LFM-style community licenses are fine; note the license in the card and
   ship any required LICENSE file inside the HF repo).
2. **Parity** — the ship gate the zoo applies to its own ports: teacher-forced / oracle top-1
   parity against the fp32 reference implementation (HF `transformers` or the official repo),
   plus a greedy-rollout sanity check. See [`PORTING.md`](PORTING.md) §gates and
   [`knowledge/evaluations-framework.md`](knowledge/evaluations-framework.md).
3. **Real hardware** — measured on an Apple silicon Mac at minimum (tok/s for LLMs, RTF for
   audio); iPhone numbers if you publish an iOS variant. Debug builds don't count — measure
   Release. No iOS 27 device? That is the one step you can hand back — see
   [Device gate](#device-gate-the-step-you-dont-have-to-own) below.

## Before you commit

Run `scripts/install-hooks.sh` once per clone. It installs a pre-commit hook that runs the
Catalog workflow's offline checks — catalog/recipe consistency, `llms.txt` freshness, and
Python syntax — so a stale generated file costs you two seconds instead of a red CI run. The
one that bites most often is `llms.txt`: it is generated from `knowledge/README.md`, so adding
a note means re-running `python3 scripts/gen_llms_txt.py` in the same commit.

## Toolchain requirement

Export with **coreai-core ≥ 1.0.0b2**. Bundles exported with earlier wheels are rejected by the
Xcode 27 beta 3+ SDK loader (`Failed to convert to versioned IR` — tracked as FB23666783); the
zoo's own pre-b2 artifacts are being migrated for the same reason.

## What blocks a merge, and what does not

Three things block, and they are the three bars above: the licence permits it, the model
demonstrably works, and the bundle is published somewhere with a revision to pin.

Nothing else does. **A port that works and is legal gets merged, and the rest is ours** —

- **the card's shape.** Send what you measured, in prose if that is easier. Matching the house
  structure is an edit, and edits are cheaper for the person who wrote the other sixty cards.
- **`recipe.toml` exactness.** If `zoo_convert.py show <name>` prints something a person could
  run, that is enough to merge on.
- **the indexes** — `models/index.json`, `models/_INVENTORY.md`, `MODELS.md`, the README tables, the
  cross-links from neighbouring cards. All generated or maintainer-owned.
- **CoreAIKit enrollment** and the card's generated "Use it" block.
- **`knowledge/` notes.** If your port taught you something, say it in the PR in whatever form
  it comes out. Turning that into a note is a maintainer job.
- **CI red that comes from generated files, or from a fork's first workflow run waiting on
  approval.** Both are ours; see step 4.

If a reviewer asks you for something before merge that is on this list, that is the reviewer
failing to absorb it. Say so, and it gets absorbed.

## Process

1. **Open (or claim) a [model request](../../issues/new?template=model-request.yml)** so work
   isn't duplicated — say you're porting it yourself. Maintainer support happens in the thread.
2. **Port it** — [`PORTING.md`](PORTING.md) is the runbook, [`knowledge/`](knowledge/) has
   per-architecture recipes (start with
   [`knowledge/README.md`](knowledge/README.md)). Ask early when something looks
   architecture-specific; it usually is.
3. **Host the bundle on your own Hugging Face account** — that's fine and encouraged (your
   models, your name). The CoreAIKit catalog pins a specific revision hash, so apps get the
   exact verified bytes regardless of where the repo lives.
4. **Open a PR here** with: `models/<model-id>/README.md` (the card — copy an existing one's
   structure), `models/<model-id>/recipe.toml` (the exact configuration that produced the bundle
   you published — `python3 conversion/zoo_convert.py show <name>` must print a complete
   command), the conversion script under `conversion/`, and the gate outputs (parity numbers +
   measured speed, environment noted). `python3 conversion/zoo_verify.py <your-hf-repo>` should
   report no FAIL.

   Two things about that PR's CI are ours, not yours. A first PR from a fork waits for a
   maintainer to approve the workflow run, so "no checks reported" means we haven't pressed the
   button yet. And `models/index.json` and `models/_INVENTORY.md` are **generated**
   (`scripts/gen_inventory.py`) — a maintainer regenerates them when your PR lands, so the
   catalog check failing with *model directories missing from the index* is expected and not
   yours to fix. Everything else that check reports is.
5. **Review + enrollment** — review is a read, not a checklist you have to pass. If the port
   works, it merges, and anything cosmetic gets fixed on `main` afterwards rather than bounced
   back to you. Enrollment in the [coreai-kit](https://github.com/john-rocky/coreai-kit) catalog
   with its revision pin (plus engine/runtime glue if it's a new capability kind) and the card's
   generated "Use it" block happen after the merge, on our side.

   You are welcome to write the kit pipeline for your own model — one contributor has, and the
   review that comes back is about kit conventions you cannot see from outside. It is an
   invitation, never a condition, and you can hand it back at any point without explaining why.

## Device gate — the step you don't have to own

Everything in a port is reproducible on any Apple silicon Mac except one thing: what the model
does on a phone. AOT load, thermals, sustained tok/s under DVFS, the memory ceiling — those need
an iOS 27 device, and the first community port hit exactly that wall ("device acceptance remains
pending on matching iOS 27 hardware", in the contributor's own recipe).

So don't buy hardware to finish a port. Clear the Mac-side gates, then open a
[device gate request](../../issues/new?template=device-gate-request.yml) with your HF repo,
revision, and headless entrypoint. A maintainer runs it on an **iPhone 17 Pro (iOS 27)**,
and posts back load time, cold + settled runs, parity against your Mac reference, and any
thermal behavior — for your card, under your name. Best-effort and queued; a gate can also come
back no-go, which is still a result worth publishing.

## Also welcome without a full port

- **Benchmark rows** from your device — the
  [bench-result issue template](../../issues/new?template=bench-result.yml) (the app measures,
  you paste).
- **Knowledge fixes** — corrections or additions to `knowledge/` notes, especially where an OS build
  changed behavior.
- **Bug reports** with a catalog id and device/OS —
  [bug template](../../issues/new?template=bug-report.yml).
