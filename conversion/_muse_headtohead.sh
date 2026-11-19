#!/bin/bash
# Head-to-head: Core AI vs Meta's ExecuTorch-Metal build of the same model.
#
# INTERLEAVED, not block-ordered. The first attempt ran all three ExecuTorch
# prompts and then all three Core AI prompts, and both sides decayed inside
# their own block (ExecuTorch 23.5 -> 17.4 tok/s across two prompts) — running
# a 30B for minutes heats the GPU, so whoever goes second is measured on a
# hotter machine. Absolute tok/s on this machine swings ~30% with thermal
# state; only an A/B/A/B ratio taken close in time is meaningful.
#
# Per prompt: ExecuTorch, Core AI, ExecuTorch, Core AI. Report both rounds.
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

N=${N:-192}
COOL=${COOL:-45}

P1="Write a Python function that merges two sorted lists into one sorted list, then explain its time complexity."
P2="A farmer has 17 sheep. All but 9 run away. How many are left? Think step by step."
P3="Summarize the tradeoff between sliding-window attention and full attention for long-context inference."

run_et() {
  "$SOLO" --model_path "$PTE" --tokenizer_path "$TOK" \
          --prompt "$1" --max_new_tokens $N --temperature 0 --ignore_eos=true 2>&1 \
    | grep -E "^Decode:" | sed 's/^/    ET  /'
}

run_ca() {
  "$RUNNER" --model "$BUNDLE" --prompt "$1" --max-tokens $N --temperature 0.0 \
            --inference-engine-variant coreai-pipelined --warmup off 2>&1 \
    | grep -E "^Generation:" | sed 's/^/    CA  /'
}

echo "protocol: $N new tokens, greedy, batch 1, interleaved A/B/A/B, ${COOL}s cooldown between runs"
sleep "$COOL"

for i in 1 2 3; do
  eval "P=\$P$i"
  echo "=== prompt $i"
  for round in 1 2; do
    echo "  round $round"
    run_et "$P"; sleep "$COOL"
    run_ca "$P"; sleep "$COOL"
  done
done
