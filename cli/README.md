# coreai-cli — export / doctor / verify / eval

Four commands for the part of a Core AI port that is knowledge rather than code: which
route a model has, which known trap an artifact is standing on, whether the bundle still
speaks, and whether it still does the job. Community tool; not an Apple product.

<!-- validated-on -->Validated on macOS 27.0 beta (26A5416b) · Xcode 27.0 beta 5 · coreai-build 3600.82.1 · coreai-core 1.0.0b2 / coreai-torch 0.4.2 — the release-OS stamp lands with 0.2.0.<!-- /validated-on -->
What "validated" means, and the logs behind it, are in [`DEVELOPMENT.md`](DEVELOPMENT.md);
what changed per version is in [`CHANGELOG.md`](CHANGELOG.md).

## Install

```
pip install coreai-cli
```

```
coreai export Qwen/Qwen3-0.6B --device iphone     # which route, and what blocks it
coreai doctor <bundle>                            # which known trap it stands on
coreai verify <bundle> --plan                     # does it compute what the reference computes
coreai eval --tasks                               # does it still do the job
```

The wheel carries the router, the lint, both gates, and a dated snapshot of the zoo's
recorded routes, so `export` answers the routing question without a checkout. Two things
still need more than the wheel: converting a checkpoint needs Apple's `coreai_models`
toolchain, and running a zoo recipe needs the
[zoo checkout](https://github.com/john-rocky/coreai-model-zoo) — `export` prints the clone
line when it routes to one. `pip install 'coreai-cli[hf]'` adds `huggingface_hub` for
`org/name` targets (config files and safetensors *headers* only; no weights are downloaded).
`xcrun coreai-build inspect` is used for the graph-level rules when the Xcode 27 toolchain
is present, and skipped with a note when it is not.

From a zoo checkout the same four run as `python3 cli/coreai_<command>.py …`; the commands
compose in the obvious order — `verify` gets the graph facts it routes on from `doctor`, and
`export` runs `doctor`'s checkpoint rules as a pre-flight and refuses `--run` on a fatal or
silent finding.

---

## `export` — the router

**It does not convert anything.** It answers, before you spend an hour finding out the hard
way: does this model have a route, through which backend, does that route have an iOS path,
and what exactly is unvalidated about it.

| backend | meaning |
|---|---|
| `preset` | Apple's stock exporter with a named preset for this exact checkpoint. Precision, compression and context length are all resolved, and Apple has run the combination. |
| `generic` | Apple's stock exporter routing by HF `model_type` only. It runs. Nothing about the recipe is validated for *these* weights. |
| `zoo` | A recorded community recipe. Reproduces a bundle that shipped and gated. |
| `none` | The `model_type` does not route. Not a CLI problem — a new architecture needs a re-authored model class. Saying so plainly is the output. |

Default is print-and-stop; `--run` executes, and refuses if the route is blocked or if
doctor's checkpoint pre-flight found a fatal or silent-corruption pattern.
`python3 cli/coreai_export.py --list` prints the whole support matrix.

| | |
|---|---:|
| Apple named presets (validated combinations) | 12 checkpoints |
| Apple `model_type` — generic, unvalidated | 19 values |
| **zoo recorded recipes** | **55 source checkpoints** |
| zoo ports whose upstream is not on the Hub (RF-DETR, YOLOX, AdcSR, …) | 5 |

**The iOS cliff, stated up front.** Apple's stock exporter accepts 14 `model_type`s and only
6 have an iOS path (mistral, olmo2, phi3, qwen2, qwen3, smollm3 — plus `llama`→`mistral` and
`qwen2_5`→`qwen2`, which is why most plain Llama checkpoints route). Gemma-3, Gemma-4,
gpt-oss, Mixtral, Qwen3-MoE, Qwen3-VL and Qwen3.5 are macOS-only. The exporter's own
`--dry-run` resolves `gemma-3-4b-it --platform iOS` without a murmur and fails only after
`AutoConfig` has read the checkpoint; `export` turns that into a `BLOCKED` line before
anything downloads.

`export` reads Apple's tables out of the installed `coreai_models` and falls back to a
dated snapshot, loudly, if it cannot; `--verify-tables` diffs the snapshot against the
installed package and exits non-zero on drift.

---

## `verify` — the gate

**A bundle that loads is not a port.** This drives the bundle and the reference over the
same ids and compares them, with two rules:

- **Validate the prompt before the bundle.** Every oracle position must clear a top-2
  margin floor (0.1) in fp32. A near-tie is a coin flip that healthy int8 noise flips and
  fp16 passes by luck, so a prompt with one is *refused* (exit 3) rather than scored. This
  is computable from the oracle alone, before a bundle exists.
- **Judge a divergence by the margin, not by the divergence.** A first mismatch below the
  floor is an fp16 knife-edge tie; above it, a real disagreement.

Two backends, chosen automatically: `zoo` when the family has a hand-transcribed fp32 oracle
in the zoo's `conversion/coreai_gate.py` (that is the authority for those models, so
`verify` prints the delegated command instead of keeping a second copy to drift), `stock`
for everything else, whose reference is plain `transformers`.

**Use a prompt that stays deterministic for the whole continuation.** The default,
`"The alphabet begins A, B, C, D, E, F,"`, was chosen by measurement across three families
at n=16, fp32 — the two obvious alternatives each fail somewhere:

| prompt | Qwen3-0.6B | SmolLM2-360M | gemma-3-1b-it |
| --- | --- | --- | --- |
| `"The capital of France is"` | ✗ min 0.0041 | ✗ min 0.0172 | ✓ 0.3231 |
| `"Counting up: 1, 2, 3, 4, 5, 6,"` | ✓ 0.6500 | ✗ min 0.0289 | ✗ min 0.0465 |
| `"The alphabet begins A, B, C, D, E, F,"` | ✓ **0.9585** | ✓ **0.9351** | ✓ **0.8020** |

Which driver can run the bundle is a property of the **graph**: a dynamic-shaped logits
output cannot be executed by the Python runtime, so it must go through `llm-runner` — which
means the GPU, and therefore the machine-wide exclusive-GPU convention (`_GPU_LOCK`). Both
are checked before anything long-running starts; `--plan` prints what would run and what
blocks it.

```
coreai verify <bundle-dir> [-n 16] [--prompt "..."] [--transcript out.json]
coreai verify <bundle-dir> --plan
```

---

## `doctor` — the lint

Reads an artifact, reports the known failure patterns it matches, and cites where each one
is written down. A conversion that errors is cheap; the expensive class is the one where
`torch.export` succeeds, the bundle loads, the model generates fluent text — and the numbers
are wrong, or the app never stops generating, or it works on your Mac and produces garbage
on the phone. [`DOCTOR_RULES.md`](DOCTOR_RULES.md) is the table: 64 patterns, each with the
symptom as you experience it, how to detect it mechanically, and a citation; 45 run.

| scope | target | catches |
|---|---|---|
| asset | `.aimodel` / `.aimodelc` directory | IR provenance, AOT staleness, symlink traps |
| graph | via `coreai-build inspect --ops --json` | state count, IO shapes, op distribution, vocab agreement |
| bundle | LanguageBundle directory | runtime contract, tokenizer class, chat surface, eos |
| checkpoint | HF checkpoint directory or repo id | quant recipe, activation scales, eos, block divisibility |
| source | PyTorch modelling code | the converter and delegate op traps |
| env | the working directory | the one env defect whose output is a bad asset |

Findings split into **DEFECTS** (something is wrong with the artifact) and **NOTES AND SHIP
REQUIREMENTS** (the artifact is fine and its host must do something specific, or it breaks).
Only defects affect the exit status: `2` for fatal/silent, `1` for runaway/perf, `0`
otherwise. A healthy, device-gated 4.6 GB bundle legitimately comes back with four
requirements and zero defects.

**One rule is true only on some OS 27 builds, and says so.** A 0.4.0-era asset (no producer
stamp) loads on OS 27 beta 1 and is refused at load by every build measured since — through
macOS 26A5416b on 2026-09-04, Apple's beta 5 release note notwithstanding. Doctor reads the
host build from `sw_vers` and reports `IR-040-DEBUG-LOC` as `fatal` on a build that refuses
the artifact and `info` on one measured to load it; `--host-build 26A5353q` judges against
another build. A release build inherits nothing: the rule flips there only after the zoo's
load sweep measures it. `AOTC-STALE-TOOLCHAIN` is fatal for an artifact compiled by a
beta-2-or-earlier `coreai-build` (181264112) and `info` for one merely older than the
installed toolchain — it used to call every published h18p artifact fatal.

```
coreai doctor <bundle-dir | *.aimodel | *.aimodelc | checkpoint-dir | hf-id | *.py>
coreai doctor <target> --json          # machine-readable, with the effective severity per finding
coreai doctor --rules                  # every rule
coreai doctor --env                    # the conversion environment (venv + zoo overlay)
```

---

## `eval` — the other question

`verify` asks whether the bundle computes what the reference computes. Its blind spot is
stated plainly: **an equivalence gate cannot detect a defect its reference shares.** The
case that produced this command: identical weights, int8 activations scoring 85/100 on
GSM8K and fp16 activations scoring 48/100. Token-exact against an fp16 oracle passes all
day. So `verify` gates the export and `eval` gates the product.

**Most of it is about comparing, not scoring.** A task number means almost nothing next to
a number produced under a different protocol, so an arm records its configuration, and
`--compare` **refuses to print a delta** until the arms agree on the fields that decide the
answer:

| protocol field | why it is on the list |
|---|---|
| `task`, `n`, `data_digest` | the same questions, or it is not the same test |
| `instruction_digest` | the prompt suffix changes the format the answer arrives in |
| `template_digest` | whether a thinking model thinks is a property of the *renderer*, not the weights |
| `max_new_tokens` | the field behind a published 12-point "quality gap" that was a 600-vs-2048 budget |
| `temperature`, `stop` | greedy vs sampled, and where generation was cut |

Everything else — bundle, driver, device, precision — is free, because that is what a
comparison is *for*. **Unrecorded is not the same as equal**: two runs that both omit a
field are refused rather than compared.

**Truncation is reported whether or not you asked.** An arm that hits the cap before the
answer marker is being scored on a different task, so the unanswered rate sits next to every
score, split into *ran out of budget* and *finished without the marker* — they look the same
in the score and have opposite fixes.

**Driver-agnostic on purpose.** `--score` takes generations from anything that writes JSON
(`llm-runner`, a device batch run, `transformers`, an ad-hoc script) — a list, an object
keyed by index, or either carrying `{"id": …, "text": …}`. It will not decode token ids: the
moment scoring owns a tokenizer, the two arms are no longer scored by identical code. Decode
in the driver and emit `text`. Bring your own task with `--task path/to/task.json`.

**`--run` records the protocol instead of asking for it.** It drives the bundle through
`verify`'s drivers and captures every field `--compare` checks from what actually happened.
The template digest is taken from the *rendered* prefix — on Qwen3-0.6B, `--thinking off`
and the default render different assistant prefixes, so two people evaluating "the same
model" with different flags get different digests and a refusal, which is the point.

```
coreai eval --run <bundle-dir> --task gsm8k -n 100 --max-new-tokens 2048
coreai eval --score gen.json --task gsm8k --arm "iphone int8" --max-new-tokens 2048
coreai eval --compare a.json b.json      # refuses on a protocol mismatch
coreai eval --tasks
```

---

## The honest boundary

`export` routes over the set Apple already supports plus the zoo's recorded recipes. It does
**not** widen that set — the other half of "model-by-model support does not scale" is how
to make a new architecture's re-authoring cheap, and that is not a CLI feature. `doctor` is
a static lint and cannot check numerics; `verify` cannot see a defect its reference shares;
`eval` scores what you give it and refuses what it cannot compare. Each says so in its
output rather than letting the command's name imply otherwise.

License: BSD-3-Clause. Source, issues and the engineering record:
[github.com/john-rocky/coreai-model-zoo/tree/main/cli](https://github.com/john-rocky/coreai-model-zoo/tree/main/cli).
