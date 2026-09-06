# AGENTS.md — for coding agents porting a model to Core AI

You are probably here because someone asked you to "convert *some model* to Core AI" or "run
*some model* on iPhone." This repo is the reference for how that is done and what counts as done.
Read this file first; it routes you to the rest. It applies whether you are working inside this
repo or borrowing its method for your own.

**Core AI** is Apple's on-device inference runtime (iOS 27 / macOS 27, `.aimodel` bundles). This
repo is a community catalog of models ported to it, each with the recipe that produced it.

## The one thing to understand before you start

**Porting is not format conversion.** There is no `convert(model)` that works. A port is:
re-author the network in plain PyTorch from the raw `model.safetensors`, export that, and prove
numerically at every stage that the export matches the original. An agent that reaches for a
one-shot converter produces a bundle that loads, runs, and emits plausible garbage — the most
expensive failure mode here, because it looks like success.

The rule that prevents it: **the oracle comes first, and every stage gates against it.** Run the
original model on fixed inputs, save every intermediate tensor, and compare. A port without gates
is a guess with extra steps.

## Route

| Your task | Go to |
| --- | --- |
| Port a **new** model, end to end | [`PORTING.md`](PORTING.md) — the full walk, two worked examples |
| Rebuild a model this repo already publishes | `python3 conversion/zoo_convert.py run <name>` · [`models/<model>/recipe.toml`](models/) |
| Find out whether a model is already ported | [`models/index.json`](models/index.json) — machine-readable catalog; start here |
| Export API details and gotchas | [`knowledge/conversion-guide.md`](knowledge/conversion-guide.md) |
| An error string you just hit (`LLVM ERROR: …`, `failed assertion …`, `failedToSpecialize`, `NSPOSIXErrorDomain Code=2`) | [`knowledge/coreai-error-index.md`](knowledge/coreai-error-index.md) — every exact string this project has observed, verbatim as a heading, with the verified cause and the fix. Search the string; `coreai doctor` links here too |
| Quantization / compression | [`knowledge/compression.md`](knowledge/compression.md) |
| Make it fast (custom Metal kernels, the bandwidth floor) | [`knowledge/custom-metal-kernels.md`](knowledge/custom-metal-kernels.md) · [`knowledge/performance-ceiling.md`](knowledge/performance-ceiling.md) |
| Use an already-ported model in a Swift app | [CoreAIKit ↗](https://github.com/john-rocky/coreai-kit) — `ChatSession(catalog: "<id>")` |
| Anything architecture-specific | [`knowledge/README.md`](knowledge/README.md) — per-architecture notes, written from ports that shipped |
| A list of everything here, in one fetch | [`llms.txt`](llms.txt) — every note with a one-line description and a raw URL |

Worth knowing why that last row exists: Apple's own Core AI documentation is a JavaScript
application, and fetching it returns a page whose entire body is "This page requires
JavaScript." If you need a runtime fact and cannot find it in Apple's docs, that is why — and
the notes below were written to be the fetchable answer.

The closest existing exporter is a better starting point than a blank file:
`conversion/export_da3.py` is the stateless archetype, `conversion/export_qwen3_5_decode_pipelined.py`
the stateful-LLM archetype.

## Decide whether to port at all

Two hard gates, from `PORTING.md` §2. Apply them **before** writing code, and report the answer
even when it is no:

- **GAP** — Apple's stock stack does not already ship this capability. If it does, stop.
- **EDGE** — the Core AI port will be at least as good as the realistic alternative, especially
  MLX. A Mac-only port of something MLX already runs fast is a "worse-MLX"; this repo has shipped
  and then pulled two of those.

State in one sentence what the port does that stock Apple + MLX do not. If you cannot write that
sentence, say so instead of porting. "The user asked for it" is not an answer to EDGE.

## The two gates a port must pass

1. **Graph parity vs the oracle** — `cos ≥ 0.999`, and **token-exact** for LLMs, per token, not
   just the first one. `python3 conversion/coreai_gate.py` runs this: it detects the architecture,
   rebuilds the fp32 reference in a child interpreter, and steps it against the engine.
2. **Host processing gated in NumPy first** — resize, mel, tokenize, detokenize. Write it in
   NumPy and gate it *before* porting it to Swift. Host bugs are unfindable inside an app.

Then re-run gate 1 **after** compression — compression is where ports die — and check the
published bundle with `python3 conversion/zoo_verify.py <hf-repo>` (catches a missing chat
template, a tokenizer that silently disagrees with the source).

## Traps that specifically catch agents

- **Trusting notes over the oracle.** A handoff note said "no input normalization"; the oracle
  showed the feature extractor always normalizes. Gate, don't read.
- **Re-authoring from the HF `modeling_*.py` instead of the weights.** The modeling file has
  branches that never run for this checkpoint, and hides ones that do.
- **Believing int4 because the loss looks fine.** int4 is a cliff, not a slope. Read generations.
- **Timing with `cpu_only()`.** That is the parity option, not a performance option.
- **Benchmarking through a chat UI.** Headless self-test entrypoint, or it did not happen.
- **JIT-ing a ≥1 GB graph on device.** AOT-compile it (`--architecture h18p` for iPhone 17 Pro).
- **Running an iOS bundle on a Mac.** Wedges the GPU stack; costs a reboot.
- **Naked `exp()` in a hand-written kernel.** Three separate sessions lost to this; subtract the
  max first.
- **Comparing quality across runtimes without matching the generation budget.** A 12-point
  "quality gap" in this repo's history turned out to be a 600-vs-2048 token cap difference.

## What a finished port ships

Not "the bundle exists." Four things, together:

1. **Weights on Hugging Face** under your own account, tokenizer assets and a card included.
2. **Export + gate scripts** in `conversion/`, self-contained, paths through `conversion/_paths.py`.
3. **A card** at `models/<model>/README.md` with measured, device-attributed numbers.
4. **A recipe** at `models/<model>/recipe.toml` — `script` + `args` + `hf_repo` + `bundle`, so
   anyone can rerun it. If a flag changed the artifact and is not recoverable, say so in
   `open_questions` and mark `status = "unverified"` rather than guessing.

Then open a PR. [`CONTRIBUTING.md`](CONTRIBUTING.md) has the acceptance bars;
[PR #6](https://github.com/john-rocky/coreai-model-zoo/pull/6) is the worked reference for what a
contributed port looks like.

**Don't hold a working port back to polish it.** Only three things block a merge — the licence,
that it works, and a published bundle with a revision to pin. Card structure, `recipe.toml`
exactness, indexes, kit enrollment and `knowledge/` notes are maintainer work
([what blocks a merge](CONTRIBUTING.md#what-blocks-a-merge-and-what-does-not)). Open the PR when
the model runs and the numbers are real.

## The step you cannot do, and don't have to

Everything above runs on any Apple silicon Mac. What needs an iOS 27 device — AOT load, thermals,
sustained tok/s under DVFS, the memory ceiling — you can hand back: open a
[device gate request](../../issues/new?template=device-gate-request.yml) and a maintainer runs it
on real hardware and posts the numbers into the thread, for your card, under your name. Do not
report iPhone numbers you did not measure, and do not let an unmeasured device claim reach a card.

## Not your call

Ask the human, every time:

- Publishing weights to Hugging Face, or any push to a namespace you were not handed.
- Posting publicly (X, HN, Reddit) about a port.
- Opening issues or PRs against `apple/*` repositories.
- Marking a port `status = "verified"` on numbers you did not produce.

## Installing these as skills

If your harness supports skills, this repo ships two — `port-a-model-to-the-zoo` and
`reproduce-a-zoo-model` — with install instructions for Claude Code, Codex CLI, and Gemini CLI in
the [README](README.md#agent-skills). Apple's own
[`coreai-skills`](https://github.com/apple/coreai-models) covers the toolchain itself; install
both. Without a skills mechanism, this file plus `PORTING.md` is the whole contract.
