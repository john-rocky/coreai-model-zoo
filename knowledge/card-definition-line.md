# Card definition line

Every model card in this zoo's own namespaces (`mlboydaisuke`, `coreai-community`) opens with the
same one-sentence definition of Core AI, placed right after the YAML front matter. The related
repos (`coreai-model-zoo`, `coreai-kit`, `apple-silicon-llm-bench`, `awesome-core-ai`,
`coreai-assets`) open with the same sentence. Adopted 2026-09-05.

## The sentence

> Core AI is Apple's on-device ML runtime in iOS 27 / macOS 27 and the successor to Core ML: PyTorch models are exported with Apple's `coreai-torch` (LLMs: `coreai.llm.export`) into `.aimodel` bundles that run on the GPU or the Neural Engine, e.g. Qwen3-8B 4-bit decodes at 94 tok/s on an M4 Max GPU, MLX 90 under the same protocol ([apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench), macOS 27 beta, 2026-06).

## Why one line, and why this one

- A reader who lands on any single card (a person, a search engine, a coding agent reading the
  raw README) gets the same definition: what Core AI is, which Apple tools export to it, one
  measured number with its device, OS and date, and where the number comes from.
- One sentence, nothing repeated elsewhere on the card. The card's own content is untouched below it.
- The number is one row of the bench repo's README: Qwen3-8B 4-bit, Apple `llm-benchmark` vs
  `mlx_lm benchmark` (mlx-lm 0.31.3), 512-token prompt / 1024 generated / mean of 5, macOS 27.0
  beta 26A5353q, 2026-06-11, raw logs under `results/raw/2026-06-11-m4max-coreai-matrix/`. It
  was chosen because it is the row whose headline, detail table and raw logs agree, on
  current-generation (27β) artifacts. The Qwen3-0.6B iPhone numbers depend on the export
  generation and were not used.
- No superlatives. The comparator (MLX 90) is stated so the sentence reads as a measurement,
  not a claim.

## Updating it

- Edit `tools/card_first_line.txt` (one line). `python3 tools/card_first_line.py` dry-runs and
  prints the diff per card; `--replace` rewrites a card that carries an older variant (recognised
  by the leading "Core AI is "); `--go` writes one commit per card
  (`Card: definition line (Core AI, YYYY-MM)`). Update the repo READMEs by hand.
- Ownership is checked live: a card whose oldest commit is not the owner's is excluded and gets a
  pull request instead. On 2026-09-05 that was `Real-ESRGAN-CoreAI` and `Moebius-CoreAI` in
  `coreai-community` (created by a contributor).
- Mirrors in `coreai-community` carry the sentence too, above their "Mirror of" line.
- The LiteRT cards the owner created in `litert-community` carry a sibling sentence (LiteRT / `litert-torch`, one
  parity number) with the same tool, its targets and sentence kept outside this repo. When the wording or the
  freshness rule changes, update both sets.
- When the number goes stale (new OS, new export generation), change the sentence and re-run with
  `--replace`.
