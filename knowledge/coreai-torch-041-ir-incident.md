# LLVM ERROR: cannot unwrap empty odiec_module_t — every coreai-torch 0.4.0 asset stops loading on OS 27 beta 2+, and how to recover / re-verify

**What happened (2026-07-18).** Every `.aimodel` converted with `coreai-torch` **0.4.0**
stops loading on **iOS/macOS 27 beta 2 and later**. It runs on beta 1. On beta 2+ both
`AIModel.load` and `coreai-build compile` abort with:

```
error: expected AICode versioned location, got: loc(fused<...>)
error: Failed to convert to versioned IR
LLVM ERROR: cannot unwrap empty `odiec_module_t`
```

Root cause (Apple, [apple/coreai-torch#37](https://github.com/apple/coreai-torch/issues/37),
[v0.4.1 release notes](https://github.com/apple/coreai-torch/releases/tag/v0.4.1)): 0.4.0
baked PyTorch stack traces into the IR as MLIR `fused` locations; the beta-2 compiler no
longer parses that nested form. It fires on deep module hierarchies. Apple's original guidance was **re-convert with coreai-torch 0.4.1+** — **UPDATE 2026-07-21: a cheaper in-place fix exists, see the strip_debug_info section at the end of this doc.** Things that do NOT work (all verified):

- `coreai-build package` — re-emits the asset (producer bumps) but leaves IR locations
  untouched; compile fails identically.
- Pinning `coreai-core` back to `1.0.0b1` — the gate is OS-side, not in the wheel.
- Re-AOT with the beta-3 toolchain — dies at the same op.
- `coreai-build inspect` still reads the asset fine, which makes it look recoverable. It
  isn't; only re-conversion from PyTorch is.

## Telling a broken asset from a fixed one (the producer fingerprint)

A 0.4.1-converted `.aimodel/metadata.json` carries a `producer` field; a 0.4.0 one does not:

```
0.4.1 (good):  {"producer": "coreai-core 1.0.0b2", "assetVersion": "2.0", "creationDate": ...}
0.4.0 (dead):  {"assetVersion": "2.0"}
```

Audit any tree by that field alone — no dates, no guessing. `.aimodelc` bundles always carry
a `producer` (the `coreai-build-<ver>` string), so for those use the source `.aimodel`'s
producer, not the compiled one.

## Environment the fix needs

- `coreai-torch` **0.4.1+**, `coreai-core` **1.0.0b2**, `coreai-opt` **0.2.1**, on the pinned
  `torch==2.9.0` (do NOT let `uv` bump torch to 2.11 — it breaks torchvision with a circular
  import and every export dies at load).
- Xcode 27 **Beta 3** (`27A5218g`) for AOT — `xcrun coreai-build` → `3600.75.3`. (Beta 2 or
  earlier `.aimodelc` also need a beta-3 recompile per Apple 181264112, but that is a
  *separate* issue from the 0.4.0 conversion break — do not conflate them.)
- **⚠️ Never run python with the coreai-torch clone as cwd**: its `coreai_torch.egg-info`
  (0.4.0) shadows the installed 0.4.1 via `sys.path[0]`, so exports silently use 0.4.0.

## Re-verifying a recovered bundle: `conversion/coreai_gate.py`

Loading is necessary but not sufficient — a bundle must still *speak*. The gate drives the
exported bundle through the Core AI engine and compares its greedy decode, token-for-token,
against the overlay model run in fp32 (the "16/16 oracle" the cards publish). It checks the
**conversion**, which is what this incident broke — unlike the eager numerics gates, which
check the quant recipe and pass regardless of converter version.

```
python3 conversion/coreai_gate.py <bundle-dir> <hf-id> [--arch KEY] [-n 16]
```

PASS = token-for-token match, or a first divergence only at a top-2 margin < 0.1 (a
knife-edge tie, fp16 class). Use a prompt that stays **deterministic for the whole
continuation** — the default, `"The alphabet begins A, B, C, D, E, F,"`, holds at min 0.80+
for 16 tokens on all three models measured. This note used to recommend `"The capital of
France is"`, which is unreliable past the first token: it answers "Paris." and then free-runs
into a list of countries where the next one is a coin flip, failing the 0.1 margin rule on
Qwen3-0.6B (0.0041) and SmolLM2-360M (0.0172) while clearing on gemma-3-1b-it (0.3231). That
model-dependence is why it lasted — it gates on some checkpoints and refuses on others.
Open-ended prompts hit ties everywhere and aren't gate material either.

- **Large models: `--oracle-dtype fp16`.** The fp32 oracle materialises all weights in
  fp32 — a 35B is ~140 GB and won't fit (27B at ~108 GB was the largest that fit 137 GB RAM).
  fp16 is the export's own trace dtype, so an fp16 oracle is still a valid conversion check.
- **MoE + custom-metal-kernel bundles gate fine.** The `sym8_gather` MoE bundles
  (`export_*_moe_metal_decode_pipelined.py`, custom `gather_qmm` Metal kernel) load and run
  through the engine on OS 27 beta 3 — the "custom Metal kernels fail to load" known issue
  (178056451) did NOT fire for them. Gate arch: `qwen3_6_moe` / `lfm2_moe`.

Non-obvious things the gate encodes (documented nowhere else):

- Engine launch needs `COREAI_CHUNK_THRESHOLD=1` + `--inference-engine-variant coreai-pipelined`
  + `--warmup off`. The default warmup does a synthetic 256-token prefill that a static-S=1
  decode graph can't serve (`Shape at dimension 1 of 256 is not a valid substitution for
  source shape 1`).
- `llm-runner --inference-engine-variant` help text is **stale**; the real values are
  `auto / coreai-sequential / coreai-pipelined / static-shape`.
- The fp32 oracle steps S=1 but `position_ids` carries the **full 0..t** range each step
  (dynamic full-length positions); a single position yields plausible-looking garbage.
- The oracle must stop at EOS and step only after the prompt is consumed
  (`t >= len(prompt)-1`), else it emits prompt-position predictions.

Each overlay builds its fp32 model differently (transcribed in `coreai_gate.py` `ARCH`):
`qwen3.5` = `Qwen3_5StatefulForCausalLM.from_hf_memory_efficient(hf_config_attr="text_config")`
with a pure-text fallback (Ornith); `lfm2_5` = `lfm2_from_hf(stateful=True)`;
`granite` = `Granite4HForCausalLMStateful.from_hf`; `youtu` = `YoutuAbsorbedStatefulForCausalLM
.from_causal_lm(youtu_absorbed_from_hf(...))` (states `kv_a`/`kv_b`); `nanbeige` = plain-Llama
`LlamaForCausalLM` (KV cache only). Models that reuse a family map through `ALIASES`
(`ornith`, `qwen3_6` → `qwen3.5`).

## Recovery loop (per model)

1. Export with 0.4.1 using the model's recorded ship command (`models/<model>/recipe.toml`, or the zoo
   card / knowledge note).
2. Gate: `coreai_gate.py <bundle> <hf-id>` → PASS.
3. Upload to HF (bundle path unchanged where possible; `perchan_sym`→`block32_sym` and
   `absorbed_msdpa`→`absorbed_int8_msdpa` were renames). Big files (10–40 GB) need
   `HF_HUB_DISABLE_PROGRESS_BARS=1`, retries, and background — a flaky link kills one-shot
   uploads mid-file.
4. Verify the uploaded `producer` is `coreai-core 1.0.0b2`.
5. For catalog-served models, re-pin `catalog.json` `revision` (+ path if renamed) in
   coreai-kit — the catalog is fetched remotely and revision-pinned, so upload alone does
   not reach users; the re-pin commit does. HF-direct models (e.g. Ornith) need no catalog
   change.
6. Free disk: delete the local export and the source weights once the upload is verified.

## UPDATE 2026-07-21 — the in-place fix: `strip_debug_info` (no re-conversion needed)

Apple later shared a workaround ([coreai-torch#44](https://github.com/apple/coreai-torch/issues/44)):
the broken assets are healthy except the debug locations, and those can simply be stripped:

```python
from coreai_torch.debugging.debug_info import strip_debug_info
from coreai.authoring import AIModelAsset
asset = AIModelAsset.load(path)          # <-- fails on beta 2+ with b2 wheels, see below
strip_debug_info(asset.program)
asset.program.save_asset(out_path)
```

Verified here on 40 zoo bundles: weights byte-identical, minutes per model, stripped assets
load clean on beta 3.

**Caveat — on a beta 2+ machine the snippet above cannot even load the asset** (the authoring
bytecode reader in coreai-core 1.0.0b2 wheels runs the same versioned-IR conversion and aborts).
The working recipe on this machine:

1. Isolated venv with **coreai-torch 0.4.0 + coreai-core 1.0.0b1** (`_recovery_venvs/strip`) —
   the b1 wheel's bundled MLIR parses the old fused locations fine. (The earlier "pinning
   coreai-core back to 1.0.0b1 does not help, the gate is OS-side" finding in this doc was
   about the RUNTIME load path; for the AUTHORING parse the gate is in the wheel, not the OS.)
2. `AIProgram._load_bytecode(bundle/main.mlirb)` -> vendored 0.4.1 `strip_debug_info`
   (0.4.0 lacks it; two helper signatures need adapting) -> `save_asset`.
3. Re-load + re-save with the b2 wheel (now parses fine) to get a proper
   `producer: coreai-core 1.0.0b2` fingerprint, then probe + publish.

Driver scripts: recovery session scratchpad `strip_b1.py` / `strip_sweep.py`.
`.aimodelc` (compiled) artifacts cannot be stripped — those need re-export + AOT recompile.
