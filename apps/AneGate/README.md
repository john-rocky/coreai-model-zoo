# AneGate — the zoo's Neural Engine gate and bench (iPhone apps + Mac scripts)

This directory is the checked-in copy of the tooling that gated every `ios-ane-h18p/` bundle in this repo
(`ondevice/_ane_gate/` on the maintainer's Mac, copied 2026-09-17; logs, derived data and the vendored Apple
package are not committed). Two iPhone apps on the **unmodified** Apple main package and a set of Mac-side scripts:

| piece | what it answers |
|---|---|
| `Sources/AneGateRunnerApp.swift` + `make_fixture.py` + `fixtures/` | **Token gate** — does the static (ANE) bundle reproduce the fp32 HF oracle token for token (teacher-forced sweep + free-run incl. the stop), on 3 short prompts + a 109-token free-form answer? Must go RED on the poisoned fixture first. |
| `simulate_recipe.py`, `scan_layer_sensitivity.py`, `bench/ref_gsm8k_hf.py --recipe`, `bench/score_gsm8k.py` | **Mac simulation of a recipe** — coreai-opt's own k-means (or the int8 linear recipe) applied to the HF fp32 weights, then the same token gate and the same 200 GSM8K questions. Predicted the phone within 0.5 pt (MiniCPM5-2B, 2026-09-17) — run this before any export. |
| `bench/Sources/TaskEval.swift`, `bench/_task_f.sh`, `bench/tasks/gsm8k_200.jsonl` | **Task gate on the phone** — the same 200 GSM8K questions through the bundle (any engine), answers pulled as JSONL, scored by `score_gsm8k.py` with the litertlm-convert extraction. |
| `bench/Sources/AppleBenchRunnerApp.swift`, `_abn_f.sh`, `_sustain_f.sh`, `Sustain.swift`, `Demo.swift` | Speed: Apple's llm-benchmark method, same-day interleaved A/B, thermal-recovered sustained runs, a recordable demo. |
| `_ane_export_s1.py`, `*.yaml` | Export → AOT h18p → ANE region count → device bundle; the palettization recipes tried (6-bit g8 shipped the 2B; 4-bit g8 does not compile for the ANE). |
| `bench/_cold_cache_s5*.sh` | First-load experiments: cold ANE program build, the two-level specialization cache, pre-seeding a container. |

Rules and results: [`../../knowledge/ane-quality-gate.md`](../../knowledge/ane-quality-gate.md) (the gate) and
[`../../knowledge/ane-vs-gpu-iphone-2026-09.md`](../../knowledge/ane-vs-gpu-iphone-2026-09.md) §9 (the measurements).

**Recreate the vendored Apple package** before building either app (not committed, 7359dbc = the Apple main this
tooling was validated on): `mkdir -p vendor/coreai-models-main-7359dbc && git -C <apple/coreai-models checkout> archive 7359dbc | tar -x -C vendor/coreai-models-main-7359dbc`.
Both apps borrow the `com.coreai.pipelinedbench` bundle id for its provisioning profile; `APP_ID=` builds under another id.
The original tool README follows.

---

# `_ane_gate` — on-device token-match gate for Apple's stock static (ANE) iOS bundles

Judges a `coreai.llm.export --platform iOS` bundle (chunked-static `extend_*` graphs, AOT
`--preferred-compute neural-engine --architecture h18p`) on the iPhone against an fp32 HF
oracle, with the zoo's existing rules (`cli/coreai_verify.py`, `knowledge/pipelined-engine.md`).
The bench app (`../_abr_ga_qwen06`) only measures speed on a synthetic prompt; this is the
correctness half.

## Pieces

| file | role |
| --- | --- |
| `make_fixture.py` | fp32 `transformers` greedy oracle → `fixtures/<model>/fixture.json` (3 prompts: `natural` raw list, `chat` no-think turn that ends on EOS, `long` >256-token raw prompt so the engine walks `prompt_opt_256_64` into the 512-context graphs). Records per-step top-2 softmax gap (`margins`, the coreai_verify floor) and raw-logit gap. Refuses a prompt with a sub-floor step. `--poison NAME:K` writes the RED fixture. |
| `Sources/AneGateRunnerApp.swift` | iPhone app on the **unmodified** Apple main package (`vendor/`, git archive of `7359dbc`). `EngineFactory` → `StaticShapeEngine`. Per prompt: **TF** teacher-forced sweep (`forcedContinuation` + `includeLogits`, argmax of the engine's own logits at every step vs the oracle token; mismatch on a margin-clear step = FAIL, on a sub-floor step = knife-edge, excluded) and **FR** free-running greedy rollout judged at the first divergence by the oracle's margin there, stop included. PASS needs both. |
| `project.yml` | xcodegen; the bundle under test is the `type: folder` resource (`devbundles/<name>/` = metadata.json with `assets.main` pointing at the `.h18p.aimodelc` + `tokenizer/`). Borrows the `com.coreai.pipelinedbench` id for its profile. |
| `_build.sh` | `BUNDLE=<devbundle dir> ./_build.sh`: writes `project.yml` from `project.yml.in`, `xcodegen generate` + `xcodebuild` (RC toolchain) → `_app_path.txt`. |
| `explore_chat.py` | tries candidate no-think chat prompts on a checkpoint and prints rollout length / EOS margin, to pick the `chat` prompt per model (`--chat-msg`). |
| `_run.sh <tag> <fixture-stem>` | launch with console (stem relative to `fixtures/`, e.g. `minicpm5_1b/fixture`), poll for `GATE_SUMMARY`, log to `_device_<tag>.log`. |

## Protocol

1. `explore_chat.py --hf-id <id>` to pick a chat prompt that stops on a margin-clear step, then
   `make_fixture.py --hf-id <id> --chat-msg "..." --out fixtures/<model>/fixture.json` (HF env
   `~/.venv-coreml-llm-py312`), and the same with `--poison natural:5 --out fixtures/<model>/fixture_red.json`.
2. `BUNDLE=<devbundle dir> ./_build.sh`; install `xcrun devicectl device install app --device <udid> $(cat _app_path.txt)`.
3. `./_run.sh red <model>/fixture_red` → must print `VERDICT=FAIL` (the gate can go red).
4. `./_run.sh clean <model>/fixture` → `VERDICT=PASS` is the bundle's correctness claim.
5. Reinstall `../PipelinedBench/build/Build/Products/Release-iphoneos/PipelinedBench.app` (same id) and
   confirm its `Documents/models` container still lists 135 files.

Rules that apply on the device: a running `devicectl … launch` means wait; never
run an iOS bundle in the Mac llm-runner; never `copy … --remove-existing-content`.

## Output grammar

```
TF <prompt> k=<step> exp=<id> got=<id> second=<id> oracle_margin=<p> dev_gap=<fp16 logit gap> ok|KNIFE|FAIL
FR <prompt> got(<n>)=[ids]
FR <prompt> PASS|FAIL <reason>
GATE prompt=<name> tf=<ok>/<n> knife=<k> fail=<f> fr=PASS|FAIL tf_s=… fr_s=… verdict=PASS|FAIL
GATE_SUMMARY fixture=<stem> model=<name> prompts=<n> pass=<p> fail=<f> load_s=… footprint_gb=… VERDICT=PASS|FAIL
```

## S1 additions (2026-09-15, speed/sustained/battery — `bench/`)

| file | role |
| --- | --- |
| `bench/_abn.sh <tag> <A,B[,C]> [rounds] [P] [G] [N] [gap]` | N-arm same-day interleaved A/B (A-B-C-A-B-C), one embedded app; `_build.sh` takes `BUNDLE3=` too. |
| `bench/_sustain.sh <tag> <A[,B…]> [duration=600] [idle=180] [P=128] [G=256]` | one launch runs the arms in sequence: idle (≥ idle s **and until thermal ≤ fair**, cap `AB_IDLE_MAX_S`=600), load, warmup, trials until `duration`; per-trial `battery bstate thermal` + tokens, also appended to `Documents/sustain/<tag>.log` in the shared container (survives a USB unplug — the battery-% protocol needs the cable out). Grammar in `bench/Sources/Sustain.swift`. |
| `bench/Sources/Sustain.swift` | the mode above; `AppleBenchRunnerApp.swift` only branches into it when `AB_DURATION_S>0`. The `model dir:` line now carries `free_gb=`. |

Pollers (`_run.sh`, `bench/_ab.sh`, `_abn.sh`, `_sustain.sh`, `_probe.sh`) stop on `ERROR:` as well — a launch that dies
with `CoreDeviceError 10002` right after an install (post-install race) no longer burns the 900 s cap; sleep ~10 s after
`_install.sh` and retry. Never run two `_build.sh` in the same dir at once (shared `project.yml`), and never build into
another build's scratch directory (derived data now defaults to `/private/tmp/ane_gate_dd/<name>`).

## S5 additions (2026-09-16, the 2B on the ANE with Apple's other recipes)

| file | role |
| --- | --- |
| `make_fixture.py --chat-n N` | rollout cap for the chat-templated prompts (was a fixed 16); the fixture now carries a 4th prompt `sky` (109-token free-form answer incl. the stop; its sub-floor steps are `knife_ok` = excluded on device, not a reason to refuse the fixture). `fixtures/minicpm5_2b/` rebuilt with it (natural/chat/long byte-identical to 2026-09-15). |
| `simulate_recipe.py` | **fp32 simulation of a palettization yaml/preset with coreai-opt's own k-means on the HF weights, judged by the fixture's TF rule** — run in the `coreai-models-rebase` venv; it reproduced the device verdicts (4-bit g32 / mixed 4-8 / 6-bit) and says which recipes can reach the floor at all (only 8-bit k-means g32 and int8 per-block-32 linear at 2B). Run it before exporting. `--preset int8_block32_linear` simulates the shipped GPU recipe. |
| `scan_layer_sensitivity.py [--engine coreai-opt] --n-bits N --group G` | per-layer sensitivity (KL + margin-clear flips on `sky`, one layer palettized at a time) to pick the 8-bit layers of a mixed recipe. |
| `minicpm5_pal6_g8.yaml`, `minicpm5_mixed_4bit_8bit{,_g32}.yaml`, `minicpm5_mixed_6bit_8bit_*.yaml` | the S5 arms (the `_g8`-based 4-bit shapes do not compile for the ANE: `ANECCompileOffline … CompilationFailure`, 0 regions). |
| `_build.sh` / `project.yml.in` (gate) | `BUNDLE2=`/`BUNDLE3=` embed up to three arms, `AG_MODEL=<dir>` picks one per launch (`_run_f.sh <tag> <stem> <dir>`, poll cap 30 min); `APP_ID=` builds under another bundle id = another app container; destination `generic/platform=iOS`. Same `APP_ID=` in `bench/_build.sh`, `bench/_launch_f.sh`, `_install.sh`. |
| `_queue_s5_gate.sh` | install + red + clean runs over the embedded arms, boundary arm last, stops on a poll cap. |
| `bench/_cold_cache_s5.sh`, `bench/_cold_cache_s5b.sh` | first-load experiments in a fresh container (a provisioned, **really absent** id — `uninstall app` wipes the container): 4096 vs 1024 graphs cold, pull/push of `Library/Caches/coreai-cache`, second fresh container. Results in the zoo knowledge note §9b. |

Rules learned: count `*ANE_region*` after every AOT (0 = the ANE compiler rejected a graph and `coreai-build` fell back to the GPU with exit 0); the plain A/B mode has no thermal readout — after a long ANE program build both arms read half, take nominal numbers with `_sustain_f.sh <tag> <A,B> 60 60`; a `tail -F` monitor on a log that `_launch_f.sh` re-pulls every 10 s re-emits the whole file.

## S6 additions (2026-09-17/18, LFM2.5-1.2B on the ANE — a conv hybrid on the static path)

| file | role |
| --- | --- |
| `s6_lfm2_check.py` | Mac twin of a static bundle driven exactly as the engine drives it (aligned batches, q ladder, mask/in_step fill). `--fixture` teacher-forces the gate fixture; `--device-log` compares the phone's per-step choice and top-2 gap with the twin's; `--rollout-fixture` writes a fixture whose oracle is the twin; `--palettize/--load-palettized` apply coreai-opt's k-means to the twin; `--ane-emu a8:tensor|fp16acc:N|bf16`, `--eps-zero`, `--ftz`, `ANE_EMU_FP16="mlp:1,7;attn:*"` are noise models of the accelerator. |
| `_ane_export_s6.py` | export with the overlay builder registered at runtime → AOT h18p → region / ANEC diagnostics → devbundle; `--num-layers` for 25-second probes; `LFM2_IOS_DIAG=` bisect knobs. |
| `_s6_bisect.sh`, `_s6_a8_layer_scan.sh`, `_s6_norm_scan.py`, `_s6_subnormal_scan.py` | the ANE-lowering bisect, the per-layer int8-activation sensitivity scan, the RMSNorm fp16-range scan, the subnormal-exposure scan. |
| `_s6_device_queue.sh`, `_s6_k8_queue.sh` | unattended device runs (wait for the phone by a live container listing, gate red/clean, GSM8K task with `AB_PROMPT_FORMAT`, trials, sustain, PipelinedBench restore, hold release). |
| `bench/ref_gsm8k_hf.py --embed-int8-per-tensor` | the iOS embedding path in the Mac GSM8K simulation (tied head reuses the int8 table). |
| `fixtures/lfm25_1_2b/`, `records/lfm25_s6/` | the 4-prompt fixture (+ red, twins), the recipe simulation, the device gate logs, the per-step comparisons, the bisect summaries. |

Result: the bundle runs on the ANE (31/31 regions, 38 tok/s, GSM8K at the checkpoint's level) but is not fp32-faithful on the phone
while its Mac fp16 twin is — knowledge/ane-vs-gpu-iphone-2026-09.md §10. Not shipped; mixed-fp16 candidate bundles are prepared.

