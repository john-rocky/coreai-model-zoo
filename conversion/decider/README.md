# decider — gates for a System One decision model (Mapika/decider-0.8b)

The bundle comes from the Qwen3.5 exporter (`../export_qwen3_5_decode_pipelined.py int8hu
--head-sym --hf-id Mapika/decider-0.8b`, see `models/decider-0.8b/recipe.toml`). These three
scripts gate the thing the model is for — calibrated option-letter probabilities — instead of a
greedy transcript:

1. `oracle_decider.py` (uv-managed, CPU, ~5 min): downloads the checkpoint's own `decider/`
   package, builds 13 typed requests into 44 rows exactly as `Decider.system_one` plans them,
   records every row's ids / slot / label ids / fp32 slot logits / probabilities and the API
   assembly → `models/decider-0.8b/fixtures-decider-0.8b.json`.
2. `readout_gate_decider.py` (overlay interpreter + `DEVELOPER_DIR` = Xcode 27 RC): AOT-compiles
   the bundle (h16c GPU), steps each row S=1 through the Python runtime with fresh states, reads
   the slot logits, `softmax(logits[labels] / 1.03)`, compares with the oracle. PASS = argmax
   44/44, max |Δp| ≤ 0.02, mean of row means ≤ 0.002, reset proof.
3. `engine_argmax_decider.py` (Release `llm-runner` from the patched fork): the Swift pipelined
   engine's first greedy token on the same ids must be the oracle's label string, 44/44.

Why AOT and why two paths: the Python runtime's GPU JIT of this graph returns zero logits on
macOS 27.0 (26A428), and the pipelined engine exposes no logits at all — see
`knowledge/decider-0.8b-port.md`.
