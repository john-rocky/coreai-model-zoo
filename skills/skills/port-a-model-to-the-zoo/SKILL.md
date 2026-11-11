---
name: port-a-model-to-the-zoo
description: Use this skill when porting a NEW model to Core AI for the zoo — "convert <some HF model> to .aimodel", "add <model> to the zoo", "why does my exported bundle produce garbage", "how do I gate a port", "what belongs in a zoo PR". Covers the oracle-first method, the two gates, compression choices, the device tier, and what a finished port must ship. For rebuilding a model the zoo already publishes, use `reproduce-a-zoo-model` instead.
---

# Port a model to the zoo

Porting is **not** format conversion. It is: re-author the network in plain PyTorch, export it,
and prove numerically that the export matches the original at every stage.

One rule governs the whole walk: **the oracle comes first, and every stage gates against it.**
A port without gates is a guess with extra steps.

**Read [`PORTING.md`](../../../PORTING.md) — it is the full walk with two worked examples.**
This skill is the map of it, plus the parts an agent gets wrong.

**Related**: `Skill("coreai-skills:working-with-coreai")` and
`Skill("coreai-skills:model-authoring")` (Apple's own skills — the toolchain and the authoring
patterns) | `reproduce-a-zoo-model` (rebuild something already published).

---

## Before writing any code: the gate on the port itself

The most expensive porting mistake is made before the first line of code. From
`PORTING.md` §2:

- **GAP** (hard): Apple's stock stack does not already ship this capability. If it does, stop.
- **EDGE** (hard): the Core AI port will be at least as good as the realistic alternative,
  especially MLX. A Mac-only port of something MLX already runs fast is a "worse-MLX" — the zoo
  has shipped and then pulled two of those.
- **FIRST / DEVICE / QUALITY / license** — see the card.

Say in one sentence what the port does that stock Apple + MLX do not. If you cannot, say so
instead of porting.

---

## The walk

| Stage | What | Where it is written down |
| --- | --- | --- |
| 1 | Oracle: run the **original** HF model on fixed inputs, save every tensor | `PORTING.md` §3 |
| 2 | Re-author in plain PyTorch from `model.safetensors` (not the HF modeling file) | §4 |
| 3 | Export → `.aimodel` (static shapes; state via `slice_update` + `remove_functionalization`) | §4, `knowledge/conversion-guide.md` |
| 4 | **Gate A** — graph parity vs the oracle (`cos ≥ 0.999`; LLMs also token-exact) | §5 |
| 5 | **Gate B** — host processing (resize, mel, detokenize) in NumPy first, gated, *then* Swift | §5 |
| 6 | Compress, then re-run Gate A on the compressed bundle | §6, `knowledge/compression.md` |
| 7 | Mac run, timed with `SpecializationOptions.default()` | §7 |
| 8 | Device: AOT-compile ≥1 GB graphs, sideload, measure with a self-test entrypoint | §8 |
| 9 | Publish: HF weights + export/gate scripts + a card + a `MODELS.md` row | §9 |

Existing exporters are the templates — read the closest one before writing a new one.
`conversion/export_da3.py` is the stateless archetype; `conversion/export_qwen3_5_decode_pipelined.py`
is the stateful-LLM archetype.

---

## The traps that cost real sessions

- **Trusting notes over the oracle.** A port's handoff notes said "no input normalization"; the
  oracle showed the feature extractor always normalizes. Gate before you port.
- **Writing Swift before NumPy.** Host bugs become unfindable inside an app.
- **Skipping the per-token gate on LLMs.** Step 1 looking fine is not a gate; AR drift shows up
  late.
- **Believing int4 without reading generations.** int4 is a cliff, not a slope.
- **Timing with `cpu_only()`.** That is the parity option, not a performance option.
- **Running an iOS bundle on a Mac.** Wedges the GPU stack; costs a reboot.
- **JIT-ing a 1 GB+ graph on device.** AOT it (`--architecture h18p` for iPhone 17 Pro).
- **Benchmarking through a chat UI.** Self-test entrypoint or it did not happen.

---

## What a finished port ships

A zoo port is not done when the bundle exists. It ships four things together:

1. **Weights on Hugging Face** under your own account, with tokenizer assets and a card.
2. **The export + gate scripts** in `conversion/` — self-contained, no hardcoded home
   directories (resolve paths through `conversion/_paths.py`).
3. **A card** at `models/<model>/README.md`, structured like an existing one, with measured,
   device-attributed numbers and a "lessons" section.
4. **A recipe** at `models/<model>/recipe.toml` recording the configuration that produced the
   published bundle — `script` + `args` + `hf_repo` + `bundle`, plus `overlay` / `needs` /
   `runtime_patches` / `runtime_env` prerequisites. If a flag changed the artifact but is not
   recoverable later, say so in `open_questions` and mark `status = "unverified"` rather than
   guessing.

Then check your own work the way the catalog will:
`python3 conversion/zoo_verify.py <your-hf-repo>` must not report FAIL — most commonly it
catches a bundle shipped without its chat template, or a tokenizer that silently disagrees with
the source model.

Publishing to Hugging Face, posting, and PRs against `apple/*` are the owner's calls, not
yours.
