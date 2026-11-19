#!/bin/bash
# Three-way: Core AI vs Meta's ExecuTorch-Metal build vs raw MLX, same machine.
#
# MLX is the arm that matters for interpreting the ExecuTorch result: their
# `metal` backend is MLX-native (their README), so beating ExecuTorch could be
# beating MLX or could be beating the ExecuTorch wrapper around it. Only raw
# MLX separates the two.
#
# Interleaved CA/ET/MLX per prompt with a cooldown between every run — a 30B
# heats the GPU in seconds and absolute tok/s on this machine swings ~30% with
# thermal state, so block ordering measures the order, not the engines.
set -u

ET_ROOT="$HOME/code/et-pr/executorch"
SOLO=$ET_ROOT/cmake-out/examples/models/muse-glimmer/solo_runner
PTE_DIR=$(ls -d ~/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B-ExecuTorch-PTE/snapshots/*)
VARIANT=muse-glimmer-k-quant-17G-128K-text-solo-metal
PTE=$PTE_DIR/$VARIANT/$VARIANT.pte
TOK=$(ls -d ~/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B/snapshots/*)/tokenizer.json

CA_ROOT="$HOME/code/coreai/coreai-models"
RUNNER=$CA_ROOT/.build/release/llm-runner
BUNDLE=$CA_ROOT/exports/muse_glimmer_30b_decode_int4hu_block32_sym

SCRATCH="${SCRATCH:-${TMPDIR:-/tmp}/muse_scratch}"
MLXPY=$SCRATCH/mlx_venv/bin/python
MLX_MODEL=mlx-community/Muse-Glimmer-30B-4bit

N=${N:-192}
COOL=${COOL:-45}

P1="Write a Python function that merges two sorted lists into one sorted list, then explain its time complexity."
P2="A farmer has 17 sheep. All but 9 run away. How many are left? Think step by step."

run_ca() {
  "$RUNNER" --model "$BUNDLE" --prompt "$1" --max-tokens $N --temperature 0.0 \
            --inference-engine-variant coreai-pipelined --warmup off 2>&1 \
    | grep -E "^Generation:" | sed 's/^/    CA   /'
}

run_et() {
  "$SOLO" --model_path "$PTE" --tokenizer_path "$TOK" \
          --prompt "$1" --max_new_tokens $N --temperature 0 --ignore_eos=true 2>&1 \
    | grep -E "^Decode:" | sed 's/^/    ET   /'
}

run_mlx() {
  "$MLXPY" -m mlx_vlm.generate --model "$MLX_MODEL" --prompt "$1" \
           --max-tokens $N --temperature 0 --verbose 2>&1 \
    | grep -iE "generation:|tokens-per-sec|tokens per second" | sed 's/^/    MLX  /'
}

echo "protocol: $N new tokens, greedy, batch 1, interleaved CA/ET/MLX, ${COOL}s cooldown between runs"
sleep "$COOL"

for i in 1 2; do
  eval "P=\$P$i"
  echo "=== prompt $i"
  for round in 1 2; do
    echo "  round $round"
    run_ca  "$P"; sleep "$COOL"
    run_et  "$P"; sleep "$COOL"
    run_mlx "$P"; sleep "$COOL"
  done
done
