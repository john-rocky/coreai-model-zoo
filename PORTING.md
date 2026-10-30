# Porting a model to Core AI — the complete walk

This document walks the whole distance once: from a Hugging Face checkpoint to a verified
`.aimodel` running on a Mac and an iPhone, published with numbers you can defend. It is written
for someone who has trained or fine-tuned models in PyTorch but has never touched Core AI.

Two worked examples run through every stage, chosen because they are the two archetypes almost
every port reduces to:

| Track | Model | Archetype | Why it teaches |
|---|---|---|---|
| **V** (start here) | [Depth Anything 3](models/depth-anything-3/README.md) (`conversion/export_da3.py`) | **Stateless single graph** — image in, depth out. No tokenizer, no state, no host loop. | The full pipeline with the fewest moving parts. A first port should look like this. |
| **L** (full course) | [Qwen3.5](models/qwen3.5/README.md) (`conversion/export_qwen3_5_decode_pipelined.py`) | **Stateful autoregressive LLM** — KV cache, prefill/decode, tokenizer, host sampling loop. | Everything Track V has, plus state and a host loop. Most of the zoo is this shape. |

The per-topic deep dives live in [`knowledge/`](knowledge/) — this page is the *path*; those are
the *depth*. When a stage says "see X", the path still makes sense without opening X; open it when
you hit that stage for real.

---

## 0. What a port actually is

Core AI (iOS 27 / macOS 27) runs models as **`.aimodel` bundles**: one or more compiled static
graphs plus assets (tokenizer files, filterbanks, metadata). The runtime loads a graph
(`GraphModel` in Swift, `coreai.runtime` in Python) and executes it on GPU/ANE/CPU.

Porting is **not** format conversion. The path is:

1. **Re-author** the network in plain PyTorch from `model.safetensors` — a clean `nn.Module` that
   `torch.export` can trace. You do this instead of exporting the Hugging Face modeling file
   because HF code carries training-time baggage (dynamic control flow, complex-number RoPE,
   optional branches) that either fails to trace or lowers badly. Re-authoring sounds heavier than
   it is: for a ViT it is an afternoon; the zoo's export scripts are the templates.
2. **Export** the re-authored module → `.aimodel`.
3. **Verify** every step numerically against the original HF model (the *oracle*). Nothing is
   trusted because it "looks right"; it is trusted because the numbers match.

One rule governs the whole walk: **the oracle comes first, and every stage gates against it.**
A port without gates is a guess with extra steps.

---

## 1. Setup

You need:

- An Apple silicon Mac on **macOS 27** (the runtime is OS-bound; betas count).
- **Xcode 27** (for the Swift side and device deploys) + command-line tools.
- Python 3.11+ in a fresh venv.
- This repo, plus Apple's [`coreai-models`](https://github.com/apple/coreai-models) checkout
  (`coreai_torch`, `coreai.runtime`, and the `coreai_models` authoring primitives come from there).
- An iPhone on iOS 27 if you want the device tier (optional until stage 8).

Practical notes that save an afternoon:

- Keep **two venvs** if your target model needs a newer `transformers` than the export stack
  likes: one env to run the HF oracle, one to export. Don't cross-contaminate.
- GPU work on the OS 27 beta driver was happiest **serialized** (not re-tested on the release OS) — run one export/verify at a time.

> **Checkpoint 1** — in your export venv, this runs clean:
> `python -c "import torch, coreai_torch, coreai.runtime"`.

---

## 2. Choosing a model (what makes a good zoo port)

The zoo applies a gate before any port starts. Use it too — the most expensive porting mistake is
made before the first line of code:

- **GAP** (hard): Apple's stock stack does *not* already ship the capability. If it does, don't port.
- **EDGE** (hard): the Core AI port will be **at least as good** as the realistic alternative,
  especially MLX. A Mac-only port of something MLX already runs fast is a "worse-MLX" — skip it.
  (The zoo has shipped and then pulled two of those; the gate exists because of them.)
- **FIRST**: no Core AI / Apple-silicon path exists yet. First-of-kind is worth more than niche-size suggests.
- **DEVICE**: it fits an iPhone (~6 GB practical ceiling) = top tier. Mac-only = tier 2.
- **QUALITY**: a shippable quantization plausibly survives (see stage 6).
- **License**: redistribute-able weights (check the HF card *and* the upstream license file).

For a **first** port, also add: stateless, single graph, < 1 GB fp16. That is Track V territory —
depth, detection, segmentation, embeddings, encoders.

> **Checkpoint 2** — you can say in one sentence what your port does that stock Apple + MLX don't.

---

## 3. The oracle comes first

Before re-authoring anything, run the **original** HF model on fixed inputs and save everything:

```python
# oracle.py (pattern; see conversion/export_da3.py, conversion/export_qwen3_5_verify_pipelined.py)
out = hf_model(**inputs)
np.savez("oracle.npz", **{k: v.float().cpu().numpy() for k, v in tensors.items()})
```

- **Track V**: one image (plus a couple of edge cases: non-square, high-contrast) → the output
  map(s). Save the *preprocessed tensor* too, not just the raw image — you will gate host
  preprocessing against it in stage 5.
- **Track L**: a fixed prompt → per-step logits (or at minimum per-step argmax token ids) for a
  few dozen steps of greedy decode. Per-step matters: an AR loop can look fine at step 1 and
  drift by step 30.

Do this **even if you have someone's notes claiming how the model behaves**. A real example from
the zoo: a port's handoff notes said "no input normalization"; the oracle showed the feature
extractor *always* normalizes (the flag that appeared to control it was dead code). The oracle
caught it; the notes would have shipped a broken model. Gate before you port.

> **Checkpoint 3** — `oracle.npz` exists, regenerates deterministically, and you know which
> tensors are inputs, which are outputs, and their dtypes/shapes.

---

## 4. Re-author and export

### The canonical export (both tracks)

Every export in this repo reduces to this skeleton (full API notes + a dozen hard-won gotchas:
[`knowledge/conversion-guide.md`](knowledge/conversion-guide.md) — read its gotcha list *before*
your first export, not after your first crash):

```python
import torch, shutil
from pathlib import Path
from coreai_torch import TorchConverter, get_decomp_table
import coreai.runtime as rt

ep = torch.export.export(m.eval(), args=(), kwargs=inputs).run_decompositions(get_decomp_table())
prog = (TorchConverter()
        .add_exported_program(exported_program=ep, input_names=[...], output_names=[...])
        .to_coreai())
prog.optimize()
shutil.rmtree(out, ignore_errors=True)          # save_asset will NOT overwrite
prog.save_asset(Path(out), rt.AIModelAssetMetadata())
```

Core AI graphs are **static-shape**. You don't fight this; you design around it: fix the shapes
that can be fixed, enumerate the ones that can't (a bundle may hold multiple graphs), and push
variable-length logic to the host.

### Track V — Depth Anything 3, the stateless pattern

`conversion/export_da3.py` is the template to imitate. The moves, in order:

1. **Re-author the backbone + head in plain torch**, loading weights straight from
   `model.safetensors`. Keep it boring: explicit cos/sin RoPE (no complex ops), explicit RMS/L2
   norms with the eps inside (`F.normalize` silently drops it), no data-dependent branches.
2. **Fix the input contract.** DA3 exports as one square `R×R` graph (`R=504`); the host resizes
   any image to `R×R` and resizes the depth map back. Aspect distortion cancels for relative
   depth — *measured* (mean r ≈ 0.98 vs the official viewer across aspect ratios), not assumed.
   Your model will have its own contract; the point is you choose it, verify it, and write it down.
3. **Fold what you can into the graph.** DA3 bakes ImageNet mean/std normalization in-graph, so
   the host feeds raw `[0,1]` RGB and one class of host bugs disappears.
4. **Let dead code die.** Export only the outputs you need — `optimize()` dead-code-eliminates
   the branches that don't feed them (DA3 drops its camera/ray heads this way, no surgery needed).

### Track L — the stateful LLM pattern

An LLM port is Track V plus three systems. Read
`conversion/export_qwen3_5_decode_pipelined.py` as the reference for what the result looks like:

1. **KV cache lives in the graph as mutable state** — in-place writes via `slice_update`, which
   requires `remove_functionalization(ep)` after `run_decompositions` or the mutation is silently
   dropped. Concepts and contracts: [`knowledge/stateful-kv-cache.md`](knowledge/stateful-kv-cache.md);
   there is a known pitfall, still present on 26A428, with in-graph KV writes and a workaround —
   [`knowledge/coreai-beta-mpsgraph-kvwrite-bug.md`](knowledge/coreai-beta-mpsgraph-kvwrite-bug.md).
2. **Prefill and decode are different shapes** of the same weights (chunked prompt ingestion vs
   one-token steps). The zoo's engine runs them as a pipelined pair:
   [`knowledge/pipelined-engine.md`](knowledge/pipelined-engine.md).
3. **The tokenizer and the sampling loop live on the host** (stage 5).

> ⚠️ **Honest caveat (2026-07):** the zoo's *registered* LLM exports (`qwen3_5`, `gemma4_text`, …)
> currently depend on an overlay of the `coreai-models` package that lives as working-tree edits —
> see the note in [`conversion/README.md`](conversion/README.md). Until that overlay is packaged,
> the reproducible-from-this-repo-alone path for a **new** LLM is the self-contained pattern the
> bespoke ports use (`conversion/bitcpm/`, `conversion/dllm/`, `conversion/rwkv7/`): re-author in
> plain torch inside your own script, exactly like Track V, plus the KV-cache moves above.

> **Checkpoint 4** — your `.aimodel` saves, loads, and runs under `coreai.runtime`
> (Track V: once; Track L: prefill once + N decode steps with state carried across calls).

---

## 5. Verify — the gates

This is the stage that makes a zoo port a zoo port. Two gates, in this order:

**Gate A — graph parity.** Load the bundle and compare against `oracle.npz`:

```python
model = await rt.AIModel.load(Path(out), rt.SpecializationOptions.cpu_only())  # cpu_only for parity
fn = model.load_function("main")
res = await fn({"image": rt.NDArray(x)})
```

- Use `cpu_only()` for the *parity* run (fp16 GPU/ANE adds harmless but distracting noise);
  anything you *time* must use `SpecializationOptions.default()` instead — it is ~an order of
  magnitude faster and that is what ships.
- Pass bar, Track V: **cos ≥ 0.999** on every output tensor (you will usually see 1.000000).
- Pass bar, Track L: **per-token cosine ≥ 0.999 on logits, and greedy argmax token-exact** over
  the oracle's decode steps. Token-exact is the headline; per-token cosine tells you *where* it
  broke when it isn't.

**Gate B — host processing parity.** Everything the *app* will compute — image resize/normalize,
mel spectrograms, detokenization, samplers — gets implemented **in NumPy first**, as the exact
algorithm the Swift will use, and gated end-to-end against the oracle's preprocessed tensors.
Only then is it translated to Swift. This ordering exists because host-side mismatches are the #1
source of "the graph is perfect but the output is garbage", and they are unfindable once the only
implementation is inside an app.

> **Checkpoint 5** — a `gate_*.py` script prints PASS with the numbers above, from a clean run,
> with no manual steps in between. This script goes in your PR; it is the reviewable artifact.

---

## 6. Compress

Rules of thumb the zoo has paid to learn (details and per-scheme mechanics:
[`knowledge/compression.md`](knowledge/compression.md)):

- **Track V / small models (< ~1.5 GB fp16): ship fp16.** Quantization risk buys you little there.
- **Track L default: int8 linear.** It is the reliably-safe LLM scheme on this stack.
- **int4 is a cliff, not a slope.** Some models survive int4 (with k-means / outlier-robust
  variants), some visibly break; the failure is capacity, so no clever rounding rescues it.
  Gate int4 with fresh eyes (Gate A per-token + actually reading generations) before believing it.
- **The ANE path has its own rule**: statically-compiled ANE execution requires palettized (LUT)
  weights — blockwise-linear int4 is a GPU-only format there. If you aren't explicitly targeting
  ANE, target GPU and move on.

Whatever you ship, re-run **Gate A on the compressed bundle** and record its numbers: compression is
part of the model, and Gate A shows what it changed. The compressed bundle passes on the task bar
(AGENTS.md, "The two gates a port must pass"), not on Gate A's per-token bars.

> **Checkpoint 6** — the shipping-precision bundle passes the task bar, its Gate A numbers are
> recorded, and size and precision are recorded for the model card.

---

## 7. Run it on a Mac

Wire the bundle into a runner and produce your first real (timed) runs:

- **Track V**: a ~50-line Python or Swift harness — load, preprocess (the NumPy-gated code),
  run, postprocess. [`swift/`](swift/) (`CoreAIRunner`) shows the Swift loading pattern.
- **Track L**: run it under the zoo's Swift engine (`swift/` for the runner package,
  [`knowledge/swift-runtime.md`](knowledge/swift-runtime.md) for the engine anatomy —
  `GraphModel(contentsOf:computeUnits:)`, tokenizer via `AutoTokenizer.from(modelFolder:)`,
  host greedy/sampling loop).
- Time with `SpecializationOptions.default()` / GPU compute units. Report load time and
  steady-state throughput separately (first call includes JIT specialization).

> ⚠️ **Never execute an iOS-compiled bundle on a Mac.** It can wedge the GPU/ANE stack and take
> the whole machine down (watchdog reboot). Mac bundles on Mac, iOS bundles on device. This is
> the one mistake in this document that costs a reboot instead of an afternoon.

> **Checkpoint 7** — same outputs as Gate A from the runner path, plus a first tok/s or ms/run
> number on your Mac.

---

## 8. Run it on an iPhone

The device tier is what separates "converted" from "shipped" — Mac-verified is tier 2;
device-verified is the gold standard, and the zoo README numbers are device numbers.

1. **AOT-compile large graphs.** On-device JIT specialization of a big static graph stalls or
   gets killed; roughly **≥ 1 GB means AOT**, ≤ ~50 MB JITs fine, in between try it:

   ```
   xcrun coreai-build compile model.aimodel --output out/ \
        --platform iOS --architecture h18p \
        --preferred-compute gpu --min-deployment-version 27.0
   ```

   The architecture name tracks the **device identifier**, not the marketing name
   (iPhone 17 Pro = `iPhone18,1` → `h18p`; M-series Macs → `h16c`). The result
   (`*.h18p.aimodelc`, ~2× the `.aimodel` size) embeds the precompiled graph.
2. **Get it onto the device.** Simplest reliable path: a debug app whose `Documents/Models/<X>/`
   is filled via Finder/`devicectl` file copy, with the app preferring a sideloaded bundle over a
   download. Push many files individually rather than one giant transfer, and verify a copy by
   reading it back — wired-tunnel transfers can report success falsely.
3. **Measure with a self-test, not a UI.** The zoo pattern: an env-gated headless entrypoint in
   the app (`<X>_SELFTEST=1`) that loads the bundle, runs 1 cold + N warm passes, computes the
   metric (tok/s, RTF, ms/frame), and writes a result file you copy back. Numbers measured
   through a chat UI are not comparable to anything.

> **Checkpoint 8** — a result file from a real device: model, device, OS, load time, cold run,
> warm runs. These are the numbers your card publishes.

---

## 9. Publish

A zoo submission is a PR with four things:

1. **Weights on Hugging Face — under your own account.** Your port, your repo, your download
   counts; the zoo links to it and credits you in the row. Include: the bundle(s), tokenizer
   assets, a card with sizes/precision/measured speeds and a "convert it yourself" pointer to
   your export script. One known trap: `swift-transformers` rejects unregistered
   `tokenizer_class` values — retag the bundle's `tokenizer_config.json` to a registered class
   (e.g. `PreTrainedTokenizer` → `BPETokenizer`) in your upload script; decode stays exact because
   it is driven by `tokenizer.json`.
2. **The export + gate scripts** in `conversion/` (self-contained: plain-torch re-author, oracle,
   Gate A/B). This is the reproducibility contract — anyone can re-derive your bundle.
3. **A model card** in `models/<model>/README.md` — copy the structure of
   [`models/depth-anything-3/README.md`](models/depth-anything-3/README.md) (Track V) or
   [`models/qwen3.5/README.md`](models/qwen3.5/README.md) (Track L): pipeline, graph contracts, measured speeds,
   lessons learned. The "lessons" section is not optional decoration; it is where the next
   porter's afternoon gets saved.
4. **A recipe** in `models/<model>/recipe.toml` — the script, the exact arguments that
   produced the bundle you published, the prerequisites (`overlay`, `needs`, `runtime_patches`,
   `runtime_env`), and `status = "verified"`. If a flag changed the artifact but cannot be
   recovered from the bundle later, record the question in `open_questions` and mark it
   `unverified` instead of guessing.
5. **A row in the README model table** linking your HF repo and card.

Then run `python3 conversion/zoo_verify.py <your-hf-repo>` — it compares what you published
against the source model and catches the two most common publishing mistakes (a missing chat
template, a tokenizer that disagrees with the source).

**Review bar** (what gets checked, so you can pre-check it): Gate A numbers as claimed and
re-runnable · host processing NumPy-gated · license clean · card complete with measured,
device-attributed numbers · no weights/binaries in the git PR itself.

> **Checkpoint 9** — someone who has never spoken to you could: download your HF repo, run your
> gate script, see PASS, and run the model in the app. That's a finished port.

---

## 10. The traps, in one place

Each of these cost a real porting session real time. The full technical list lives in
[`knowledge/conversion-guide.md`](knowledge/conversion-guide.md) (export/runtime gotchas) — these
are the *process* traps this walk is designed around:

- **Trusting notes over the oracle** (stage 3). Gate before you port.
- **Writing Swift before NumPy** (stage 5). Host bugs become unfindable inside an app.
- **Skipping the per-token gate on LLMs** — step-1-looks-fine is not a gate; AR drift shows up late.
- **Believing int4 without reading generations** (stage 6). The cliff is real and model-specific.
- **Timing with `cpu_only()`** (stages 5/7) — parity option, not a performance option; real runs
  use `default()`.
- **Running an iOS bundle on a Mac** (stage 7). Reboot.
- **JIT-ing a 1 GB+ graph on device** (stage 8). AOT it.
- **Benchmarking through a chat UI** (stage 8). Self-test entrypoint or it didn't happen.

---

## Getting help

Open an issue with: the model, which **checkpoint** you are stuck at, and your gate output (the
`oracle.npz` shapes and the cos numbers). Stuck-at-a-checkpoint issues get real review — the
gates make your problem diagnosable from the numbers alone, which is exactly why they exist.
