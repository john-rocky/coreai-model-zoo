# recovery — the in-place repair for coreai-torch 0.4.0-era bundles

Background and evidence: [`knowledge/coreai-torch-041-ir-incident.md`](../../knowledge/coreai-torch-041-ir-incident.md).
A bundle converted with coreai-torch 0.4.0 (no `producer` in `metadata.json`) is refused at
`AIModel.load` on every OS 27 build measured since beta 2 (`LLVM ERROR: cannot unwrap empty
odiec_module_t`). Stripping the debug locations repairs it in place; the weights are untouched.

Two steps, two interpreters, because the b2 wheel's bytecode reader cannot parse the old
locations and the 0.4.0 wheel lacks `strip_debug_info`:

```bash
# 1. isolated venv: coreai-torch 0.4.0 + coreai-core 1.0.0b1 (once)
uv venv -p 3.11 ~/code/coreai/_recovery_venvs/strip
uv pip install --python ~/code/coreai/_recovery_venvs/strip/bin/python "coreai-torch==0.4.0" "coreai-core==1.0.0b1"

# 2. strip in the b1 venv, then re-save with the b2 wheel (stamps producer) and load it
~/code/coreai/_recovery_venvs/strip/bin/python conversion/recovery/strip_b1.py <in.aimodel> /tmp/stage1.aimodel
../coreai-models/.venv/bin/python conversion/recovery/resave_b2.py /tmp/stage1.aimodel <out.aimodel>
```

Re-verified 2026-09-09 on macOS 27 26A5416b (coreai-core 1.0.0b2 for step 2):
`mlboydaisuke/qwen3.5-0.8B-CoreAI` `gpu-pipelined/qwen3_5_0_8b_decode_int8hu_perchan_sym`
(1.2 GB, 3253 ops) — strip 2.2 s, re-save 0.9 s, `AIModel.load(cpu_only)` OK where the
original aborts. `.aimodelc` (compiled) artifacts cannot be stripped; re-export and re-AOT those.

After the repair: `python3 conversion/zoo_smoke.py <repo> --all` records the load with the host
build, `conversion/zoo_verify.py` runs tier-1, and the upload follows `_publish_tier1_fixes.py`.
