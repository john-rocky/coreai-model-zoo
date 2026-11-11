# Rebuild a bundle — the conversion recipes

Every published bundle carries the configuration that produced it in
`models/<model>/recipe.toml`, and one command runs it:

```bash
python3 conversion/zoo_convert.py list                    # what can be rebuilt
python3 conversion/zoo_convert.py show  qwen3.5-0.8b      # the command + everything it needs
python3 conversion/zoo_convert.py run   qwen3.5-0.8b --dry-run
python3 conversion/zoo_convert.py run   qwen3.5-0.8b      # do it
```

`show` prints four kinds of prerequisite, and the run-time ones matter as much as the rest — an
export that skips them still succeeds, and the bundle then misbehaves inside the app:

| line | means |
|---|---|
| `overlay` | the interpreter needs `coreai_models` with [`conversion/overlay/`](../conversion/overlay/) applied — `zoo_convert.py doctor` checks it |
| `needs` | something the export cannot run without: a checkpoint download, a gather-table dump, a package patch |
| `runtime` | what the **app** needs to run the result: an engine patch from [`apps/`](../apps/), an environment variable such as `COREAI_CHUNK_THRESHOLD=1` |
| `device` | the AOT compile step for the iPhone bundle |
| `uv` | this script declares its own dependencies — no venv, no overlay, nothing to install |

Ports whose exporter is self-contained need no setup at all:

```bash
uv run conversion/export_da3.py --variant small --dtype float16 --res 504
```

Paths never assume a machine: `python3 conversion/_paths.py` prints where downloads, exports and
the Hugging Face cache resolve, and `ZOO_WORK_ROOT` / `ZOO_EXPORTS` / `ZOO_CODE_ROOT` /
`HF_HUB_CACHE` move them.

**A recipe marked `unverified` refuses to run** without `--force`. It means the repository does
not record which configuration produced the published bundle, and it prints the exact question
it cannot answer. Running it anyway yields *a* bundle, not *the* bundle.

**Do not expect a checksum match.** Conversion is not byte-deterministic here: the same recipe
run twice on the same machine produces bundles that differ from each other (measured:
`main.mlirb` by 7 bytes, `main.hash` entirely). A rebuild is judged by the gates the export
runs — and by checking the published artifact itself:

```bash
python3 conversion/zoo_verify.py mlboydaisuke/Gemma-4-12B-CoreAI     # one repo
python3 conversion/zoo_verify.py --all --json models/_VERIFY.json    # the whole catalog, minutes
```

That compares a bundle's tokenizer, chat template, context length and declared precision against
the source model it names in its own `metadata.json` — no oracle, no device, no weights.

**The numerical check is a different command.** `zoo_verify.py` checks that a bundle is described
correctly; it does not check that the bundle still computes the right thing. That is
`conversion/coreai_gate.py`, which rebuilds the reference model in fp32 and compares a greedy
decode token for token:

```bash
python3 conversion/coreai_gate.py <bundle-dir> Qwen/Qwen3.5-2B --revision <sha> \
    --transcript models/qwen3.5-2b/gate.json
```

It runs outside this working tree — point `--runner` / `ZOO_LLM_RUNNER` at your `llm-runner` and
`--python` / `ZOO_CONVERT_PYTHON` at an interpreter that has the export overlay, and it tells you
which one is missing rather than failing obscurely. It currently covers the decode
architectures in `--arch` (the oracle is a per-architecture transcription of that model's export,
so it grows one port at a time, not all at once).

`--transcript` is the part worth publishing: the pinned revision, the exact `input_ids`, both
sides' generated tokens, the tie margins, and the verdict. **Rebuilding the oracle is expensive;
re-running the engine side against a published transcript is not** — it needs the bundle,
`llm-runner`, and the recorded `input_ids`, and the output must match `engine.gen_text`.

New ports publish one. For models ported before the flag existed,
`conversion/backfill_gate_transcripts.py` prints what can still be gated and, with `--run`, gates
it — preferring the **published** bundle out of the CoreAIKit cache over a local export, so the
transcript describes the bytes apps download at the revision the catalog pins. Nothing
reconstructs a transcript from a card's prose after the fact: a model that cannot be re-gated
keeps a card that says what was gated, without a file claiming to prove it.

Results land in [`models/_INVENTORY.md`](../models/_INVENTORY.md); [`models/index.json`](../models/index.json)
is the same catalog machine-readable, which is where an agent should start. Each recipe entry
there carries `status` (does the repository record what produced this bundle) and
`gate_transcript` (is the numerical check against the original published, and where) — two
different questions, kept as two fields so neither has to be inferred from prose.
