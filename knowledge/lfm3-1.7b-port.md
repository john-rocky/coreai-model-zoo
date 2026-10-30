# LFM3-1.7B — port knowledge

`LiquidAI/LFM3-1.7B` (LFM Open License v1.0, revision `ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7`) is the first
LFM3 checkpoint in the zoo. It keeps LFM2.5's idea — short-conv mixers interleaved with attention, no recurrent
scan — and changes almost everything around it: the conv window, a sliding window on five of the eleven attention
layers, the rotary, an output gate, the norms, and the prompt surface (thinking on by default, XML tool calls).
The decoder module written for it is shared by the three LFM3 multimodal ports in progress (VL, Omni, Audio;
separate pull requests), so this note is also where their language model comes from. Card:
[`models/lfm3-1.7b`](../models/lfm3-1.7b/README.md).

The engine here is the zoo's fork of apple/coreai-models
([john-rocky/coreai-models](https://github.com/john-rocky/coreai-models)): `llm-runner` and the zoo's provider
were built from its branch `zoo-0.2` at 661a5d4 (the checkout that ran the gates carried one more local commit, an
unrelated Python module, so its Swift engine is 661a5d4's), and CoreAIKit pins its tag `0.2.8-zoo`. Below, "the
fork" is that branch at 661a5d4 and "the kit's pin" is the tag.

## Six differences from LFM2 / LFM2.5

| | LFM2.5-1.2B / 2.6B | LFM3-1.7B |
|---|---|---|
| 1. conv mixer | depthwise causal conv, kernel 3; state `[n, 1, 2048, 2]` | kernel **7**; state `[13, 1, 2048, 6]` (13 conv layers, the last 6 columns of B·x) |
| 2. attention layers | full attention only (6 / 8 layers), 32 q / 8 kv heads, head_dim 64 | **11 = 6 full + 5 sliding (window 256)**, interleaved with the conv layers; 16 q / 4 kv heads, head_dim 128 |
| 3. rotary | full head_dim, one θ (1e6 / 1e7) | full layers: θ 1e6 on **half** the head_dim (`partial_rotary_factor` 0.5, zero-tail, below); sliding layers: θ **1e4** on the full head_dim — two tables per step |
| 4. attention output | o_proj of the heads | **output gate**: `sigmoid(gate_proj(x))` (2048 → 16, one scalar per head, x = the mixer's normed input) times each head before o_proj |
| 5. norms | pre-norm per block (operator_norm → mixer → + residual → ffn_norm → MLP → + residual), final `embedding_norm` | **sandwich norm**: operator_norm → mixer → post_operator_norm → + residual → ffn_norm → SwiGLU → post_ffn_norm → + residual; the final norm keeps LFM2's name `embedding_norm` |
| 6. prompt surface | pythonic calls between `<\|tool_call_start\|>` / `<\|tool_call_end\|>`; no reasoning block (1.2B) or `<think>` opened by the template (2.6B) | **XML** calls (`<tool_call>` / `<function=NAME>` / `<parameter=KEY>`); the template opens `<think>` by default; the vocab still carries LFM2.5's `<\|tool_call_start\|>` pair |

Unchanged: hidden 2048, vocab 128 000 with a tied head, a SwiGLU MLP (7680 here), RMSNorm with eps 1e-5,
per-head q/k RMSNorm. The config is transformers-v5 shaped: `tokenizer_class: TokenizersBackend` (the
tokenizer trap of [`lfm2.5-2.6b-port.md`](lfm2.5-2.6b-port.md) applies as it is), and `rope_parameters` one
level deeper than the 2.6B's — a dict per layer type (`full_attention`, `sliding_attention`).

The module is new (`coreai_models/models/macos/lfm3.py` in the authoring overlay), not a flag on `lfm2.py`: it
imports LFM2's conv mixer and `_rotate_half`, uses the shared primitives (`MLP`, `RMSNorm`, `SDPA`, `KVCache`),
and writes the attention, the decoder layer and the stack. It was gated against transformers 5.20.2's fp32 eager
on 17 sequences (8 short questions with thinking off and on, and one 1,648-token document) along three paths — one
call over the whole sequence, S=1 with a 4,096-slot KV, and the engine's own walk (S=1, KV grown from 256 by
doubling): worst max|Δlogit| 1.22e-3 / 2.61e-3 / 2.61e-3, top-1 2682/2682 on every path (one position whose HF
fp32 top-2 gap is under 1e-3 left out). Three negative controls turn it red: dropping the output gate (23.4 on the
long sequence, top-1 889/1647), rotating every dimension (16.8, 1450/1647), and removing the window (24.3,
1081/1647; q1_off, at 31 tokens, stays green: under 256 tokens the window never bites).

## The sliding window is the SDPA composite's `window_size`, with no mask input

The decode graph keeps the whole KV for all 11 attention layers in one KV pair (`[11, 1, 4, L, 128]`, L
dynamic) and lets the composite apply the window: `Lfm3Attention` passes `window_size=256` to the SDPA
composite, the same path the stock `gemma3_text.py` takes. The composite's window branch is the lower-right
causal mask XOR `q_idx >= k_idx − (k_len − q_len − W)`, i.e. "absolute query position − key position < 256",
which is transformers' `kv_idx > q_idx − sliding_window`. The host passes `position_ids` (the full length) and
the three states; there is no mask input, and the engine's contract (one KV pair + one fixed extra state)
holds.

Two things were checked before trusting it:

- **The composite's mask is additive and finite** (`−1e4` where masked). The attention scores' absmax over
  the 17 sequences is 22.8, so nothing leaks through it. A model whose scores reach the thousands would need
  a look at this before reusing the path.
- **The GPU delegate applies the window.** A tiny random 6-layer model with window 8 and 45 tokens, exported
  in fp32 and compiled AOT for the Mac GPU, matched transformers to 6.2e-5, with the top-1 right on all 37
  positions past the window. On the real model the fp32 graph on the GPU is at 2.20e-3 over the 1,648-token
  sequence.

A ring-buffer KV for the five sliding layers (256 slots) was not built: the decode over a ~3.7–4k KV is only 1.9 %
slower than over ~1–1.3k for all 11 attention layers, so a ring on 5 of them is worth at most about 0.9 % at a
4,096-token context.

## The partial rotary is a zero tail, and there are two θ per step

`partial_rotary_factor: 0.5` on the full layers does not mean "rotate the first 64 dimensions". transformers
builds `inv_freq` over the whole head_dim — its first 32 entries are θ^(−2i/128) and the last 32 are **0** — and
then concatenates the frequencies with themselves and applies `rotate_half`. The rotated pairs are (i, i+64) for
i < 32; dimensions 32–63 and 96–127 get angle 0 (cos 1, sin 0) and pass through. A port that rotates the first 64
dimensions instead pairs other dimensions at other frequencies; that form was not run here. The control that was
run — the full layers rotating every dimension, with no zero tail — moves the long sequence's logits by up to 16.8
(top-1 1450/1647). The sliding layers rotate every dimension with θ 1e4.

The module builds both tables once per step from `position_ids` in fp32 and shares each between the layers of its
type. `inv_freq` is a plain fp32 attribute, not a registered buffer, so a `.to(torch.float16)` of the model does
not round it; the sliding θ stays the integer 10000 from the config (the tiny model's prefill logits match
transformers at max|Δ| 0.0).

## What stays fp32 in an fp16 or int8 export

The same two GPU-delegate traps as LFM2.5 (`models/lfm2.5`, its "Two GPU-delegate findings"), and the
same fixes: the conv layers return their new columns and the stack writes all 13 slots of the conv state with
one `slice_update` per step, and the attention projections keep fp32 weights. For LFM3 the fp32 path is wider:

- q / k / v / o_proj and **gate_proj** weights, and the q / k norm weights;
- q and k stay fp32 from the projection through the per-head norm and the rotary; they become fp16 only when
  written to the KV and handed to SDPA;
- the gate's sigmoid is fp32 (an fp16 sigmoid differs by up to 3.45e-4 on the real activations, for a cost of
  `[1, 1, 16]` elementwise work), and the SDPA output is multiplied by it in fp32 before o_proj.

The activations themselves fit fp16 with room: over the 17 sequences the largest values were 4.53 (mixer
output before its post-norm), 30.2 (MLP output before its post-norm), 55.6 (SwiGLU intermediate), 25.7
(residual stream) — far from 65 504, so the sandwich norm needs no residual scaling.

## fp16 activations have a floor on this model

The fp16 bundle fails the strict parity bar on the long sequence: teacher-forced on the GPU against transformers'
fp32 over the same 17 sequences, max|Δlogit| is 0.353 (p50 0.026), and 6 of the 1,647 counted positions of the
1,648-token sequence flip their top-1, at positions where transformers' own top-2 gap is 0.003–0.061. It is not
the conversion: the same module run in fp16 on the CPU in PyTorch shows the same width (0.49, 4 flips), the fp16
logits' rounding alone accounts for 0.008–0.011, and the same graph exported in fp32 and run on the GPU passes at
2.20e-3 with every top-1 right. Read fp16 numbers for this model, and for the VL / Omni / Audio language models
built on the same module, against that floor. The shipped bundle's int8 weights (`int8hu`: the MLP and conv-mixer
linears int8 per-block-32, the head untied and int8, the attention fp32), teacher-forced S=1 through `main`, keep
2637/2682 top-1 = 98.32 % (1027/1035 on the short sequences, 1610/1647 on the long one); the same quantized
weights computed in fp32 in PyTorch keep 2636/2682, so the fp16 activations add almost nothing on top of the
quantization.

## The chat template: eight rewrites that change no rendered string

The vendor `chat_template.jinja` (sha256 `f45005c5…`) uses Python-jinja constructs the Swift renderer does
not handle in every version that ships:

| swift-jinja | vendor template | the bundle's template |
|---|---|---|
| 2.3.2 (the pin of `apple/coreai-models`) | does not parse: `Expected '%}' after for loop.. Got or instead` (`{% for tool in tools or [] %}`) | 30/30 |
| 2.3.6 (CoreAIKit's pin) | parses; the 10 conversations with tools differ (`tojson` is a JSONEncoder: sorted keys, no spaces, `\uXXXX`, `\/`) | 30/30 |
| 2.5.2 | 30/30 | 30/30 |

30 conversations = 15 scenes with thinking off and on — the 8 questions, a three-turn chat with reasoning in the
history, a turn with a system prompt and no tools, and five tool scenes (a round trip with and without a system
prompt, two calls in one turn, a finished round trip in the history, a tool description full of characters that
JSON escapes) — compared string for string and id for id with transformers 5.20.2's `apply_chat_template`. The
bundle carries the vendor template with eight rewrites: parentheses around `tools or []`, and a `_json(v)` macro
in place of the template's `tojson` calls — it uses `tojson` when that already prints `json.dumps`' format and
writes the format out otherwise. transformers' jinja2 renders the rewritten template to the same string as the
vendor's on all 30. The other tokenizer files are the source's, verbatim.

Watch for one quiet fallback: when the template does not parse, Apple's `CoreAILanguageModel` (and the fork's)
catches the error, logs one `CLILogger` line (off by default) and encodes the messages' text joined by newlines
without a chat template; the caller is not told (read in the code, not run). That is why the rewrite is shipped
instead of relying on a newer swift-jinja being resolved.

## Two prompt-surface traps

- **The tool-dialect probe has to look for LFM3 first.** LFM3's vocab keeps LFM2.5's `<|tool_call_start|>` /
  `<|tool_call_end|>` tokens, so a probe that recognises LFM2.5 by them picks the pythonic dialect for a model
  that calls tools in XML. The zoo's `defaultDialect(probing:)` and CoreAIKit's `KitLanguageModel` check
  for the single tokens `<function=` and `<parameter=` first (`LFM3Dialect` / `LFM3ToolDialect`).
- **The answer starts inside a thinking block.** With thinking on, the generation prompt ends with `<think>`, so
  the model's first token is already reasoning. Stream parsers that start "outside" — Apple's `ThinkTagParser`,
  and the zoo's and the kit's parsers before this port — would send the reasoning into the answer text (checked on
  the parsers with synthetic streams). The rule both repos now apply: when the rendered prompt ends with an open
  thinking tag (and an optional newline), the parser starts inside it. Apple's parser is unchanged.

## On the real engine, with the shipped int8 weights

- **Thinking on** (greedy, a 3,584-token budget, the 8 questions in 3 runs of fresh processes): every reply closed
  `</think>` after 40–237 tokens (median 56) and answered right; the three runs gave the same text. In all 24 the
  answer starts right after `</think>`, with no whitespace in between. (Run on the `main`-only build of the same
  weights: these 20–29-token prompts never reach `prefill`, and the shipped bundle's `main` gives that build's
  values at all 2,683 positions.)
- **Whitespace after `</think>` would shift the history.** The zoo executor keeps whatever follows `</think>`
  as the answer, where LFM3's tokenizer_config `response_template` consumes whitespace there
  (`</think>\s*`), so a reply that began with a newline would re-render a token off transformers' history. Not
  seen in these runs: the prompts of a three-turn thinking-on chat, re-rendered by the zoo provider and by
  CoreAIKit's `ChatSession`, equal transformers' render id for id on every turn.
- **A tool round trip** through Apple's `LanguageModelSession`, with one tool that returns a fresh random value
  each run: 3 / 3 through CoreAIKit's `KitLanguageModel` and 3 / 3 through the zoo's `ZooLanguageModel`. The model
  reasons, calls `lookup_secret` in XML with `key: alpha`, reasons over the result and answers with the value
  verbatim; the reasoning lands in the transcript's reasoning entries, in none of the six runs in the answer.
- **Neither provider reused the KV across LFM3 turns.** The kit's pin refuses a partial rewind over the conv
  state, so CoreAIKit falls back to a full reset, and the zoo executor resets when its cached tokens diverge from
  the next prompt; both re-prefill the next prompt in full, which costs time, not correctness (the re-rendered
  history drops the earlier turns' reasoning, as the chat template does). A separate difference: the fork's engine
  keeps generating to `maxTokens` after the consumer stops at the stop token (the zoo executor drains and records
  it, as its type comment explains), so each zoo turn here ran to `maximumResponseTokens` (2,048), while the kit's
  pin stops at the stop token (50 tokens for the call turn). Left open: in the tool loop, the call turn
  re-rendered from its parsed reasoning and call has the generated turn's token count (50, with no whitespace
  after `</think>`) but not all of its tokens; where they differ was not isolated.

## Chunked prefill: the bundle's second function

- One asset, two functions: `main` is the S=1 decode graph (the composites' own causal and window masks) and
  `prefill` is a static 64-token query over the same positions, KV and conv state, with the masks built from the
  positions (`Lfm3Model.set_explicit_mask()`). Exporter flag `--prefill-chunk 64` (bundle suffix `_pf64`, for
  "prefill chunk 64"). The weights are shared in the `.aimodel` (+174 KB for the second graph); the Mac AOT
  (`--architecture h16c --expect-frequent-reshapes`; the architecture codes this port compiled for are h16c on the
  M4 Max, h18p on the iPhone 17 Pro and h19p on the iPhone 18 Pro) holds an fp16 copy of the int8 Linears per
  function (resources.bin 6.39 → 10.08 GB).
- The pipelined engine finds `prefill` by name — the fork and the kit's pin alike: a prompt of N tokens runs
  floor((N − 1) / 64) chunks through `prefill` and the rest one token at a time through `main`, the last one as
  the sampled step (a 1,024-token prompt: 15 chunks, then 64 single steps). Its log line (`CLILogger` level 1, off
  by default in an app): `Pipelined engine loaded 'prefill' function: static chunk S=64`.
- Fed that way and teacher-forced on the 17 sequences: top-1 2639 / 2682 = 98.40 % (`main` alone: 2637); the
  logits at the chunk boundaries equal the `main`-only path's at cos ≥ 0.9999922; `main` gives the `main`-only
  bundle's values at all 2,683 positions. Greedy, a 215-token prompt that rides 3 chunks gives the fp32 oracle's
  16 tokens (the token gate: `conversion/coreai_gate.py`, 16 greedy tokens against transformers fp32); the
  stripped `.aimodel` gives the export's values at every position, chunked and not.
- Through the kit's pin (the `.aimodel` specialized on first load on the Mac, a 1,233-token prompt, timed while
  other jobs ran on the Mac): the first token after 0.68 s with `prefill`, 7.4–7.8 s with `main` only. A tool
  round trip's prompts (127 and 197 tokens) ride it too.
- The last 1–64 prompt tokens go S=1 through `main`; at a 1,024-token prompt those 64 steps are about half of the
  first-token time.

### A padded last chunk (`--pad-tail`, not in an engine yet)

- What it buys: the prompt's tail goes through `prefill` as one padded call instead of one token at a time — at a
  1,000-token prompt the first token comes after 0.444 s instead of 0.767 s, at 200 tokens after 0.122 s instead
  of 0.157 s, with the same decode (M4 Max, 8 runs of each, alternated). It needs an engine that pads the tail:
  the one measured is a patched copy of the fork's engine; neither the fork nor the kit's pin pads.
- The exporter's `--pad-tail` gives `prefill` a chunk padded at its end: the padding engine writes token 0 and
  position id -1 into the padded rows, and the graph reads the new conv state after the chunk's real tokens —
  columns [n, n + 6) of [state | Bx], n = the number of query position ids ≥ 0 — instead of the last 6. Nothing
  else changes: the causal mask comes from indices, so no real query sees a pad; the KV rows the pads write lie
  past the cursor and are written again before anything reads them. A full chunk (n = 64) reads the last 6
  columns, so an engine that never pads computes exactly what `_pf64` computes (all 2,683 positions equal, chunked
  and through `main`), and the AOT keeps `_pf64`'s bytes (resources.bin 10,078,970,328 B both).
- With the prompt's tail as one padded call (a pipelined-engine copy that pads tails of 4 or more tokens): top-1
  2639 / 2682 (every sequence through one padded call), the call boundaries' logits vs `main` alone cos ≥
  0.9999922, the token gate 16 / 16 on the alphabet (14 tokens + 50 pads) and on the 215-token prompt (22 + 42),
  the stop-token check passes.
- The control that has to go red: `_pf64`'s graph walked the same way keeps the pads' columns as its conv state —
  the boundary cos falls to 0.8645 while top-1 stays at 95.79 %, above the 93 % bar. Top-1 alone does not catch
  a wrong conv state; the boundary cos does.

### Turn 2 from a conv-state checkpoint (an engine patch, not in the kit's pin yet)

- What it buys: turn 2 of a long conversation (1,672 tokens) gets its first token after 255.3 ms from the
  checkpoint (89 tokens to prefill), or 228.1 ms the kit's way (its rewind to the longest shared prefix keeps the
  reply: 25 tokens), instead of 774.5 ms for the full re-prefill; on two short chats 469.1 → 315.4 ms (thinking
  on) and 486.0 → 324.7 ms (off) (M4 Max, the engine driven directly, the rewind included, 8 rounds each).
- Today the kit and the zoo re-prefill the whole history every LFM3 turn: the conv state is a fixed-shape state,
  and the engine refuses to rewind into the middle of it.
- A checkpoint of the conv state (13 × 2048 × 6 fp16 = 320 KB) at the end of turn 1's prompt, written back when
  turn 2 rewinds to it, with the KV cursor moved back (the KV rows need no copy): the kit's pin's pipelined engine
  with a patch that implements `checkpoint()` for it (the pin's `InferenceEngine` declares it; only its sequential
  engine had it), driven the kit's way (rewind to the shared prefix, then generate), gives turn 2 the full
  re-prefill's 16 greedy tokens on 3 conversations (thinking on, thinking off, and a long one that opens with the
  document's 1,584-token prompt); on the two short ones the first token's logits are bit-equal. With the conv
  state not written back the logits move by up to 9.2 — the check goes red.
- What the kit would have to call: `engine.checkpoint()` after a turn's prompt; its existing rewind to the
  longest shared prefix then lands on the checkpoint.

## Shipping

- The exported `main.mlirb` carries the export machine's absolute paths (every op's source location). The
  shipped `.aimodel` is re-saved after coreai-torch's `strip_debug_info`; that is a new asset (a new
  `main.hash`, a new compile-cache entry), so its gates are run again on the stripped bytes.
- The bundle's `metadata.json` points `assets.main` at the `.aimodel` (functions `main` and `prefill`); on the
  Mac the engine specializes it on first load.
- Thinking off under sampling (temperature 0.7, the card's default configuration) put a stray `</think>` into one
  answer in 21 runs (0 in 21 on the `main`-only build; the prompt never reaches `prefill`): CoreAIKit's
  `ChatSession` passes it into the reply.

## Toolchain notes

- The fork at 661a5d4 does not compile with Xcode 27.0 RC's SDK: `extraneous argument label 'capabilities:' in
  call` (`CoreAILanguageModel.swift:51`; apple/coreai-models' main of 2026-10-28 makes that call without the
  label). Dropping the label builds it; the engine is unchanged.
- The Python runtime's `cpu_only()` rejected the tiny model's stateful dynamic graph on every call with
  `inferenceFailed(-1)`, in one run; the 1.7B graph was not tried on the CPU. The CPU reference is the PyTorch
  module; the graph's own reference is its fp32 export compiled for the GPU.
- A static S>1 graph (chunked prefill) with the composite's window fails `torch.export` with
  `Constraints violated (d_32)`: the externalized SDPA has no lower bound on the KV length and meets the
  composite's `_check_is_size(k_len − q_len)`. `Lfm3Model.set_explicit_mask()` switches the layers to a bool
  mask built from the positions (the composite then runs without its window); a tiny static S=16 graph passes
  on the GPU with it. The 1.7B's S=64 graph: "Chunked prefill" above.
- mlx-lm 0.32.2 (the newest release on 2026-10-29) and its main branch that day have no `lfm3` model type: loading
  LiquidAI's own `LFM3-1.7B-MLX-8bit` / `-4bit` stops at `ValueError: Model type lfm3 not supported.`, so there is
  no MLX number to compare with yet.

## iPhone

This release has no iOS bundle. What ran on iPhones (iOS 27.2), with a 768-token prompt and 256 generated tokens —
prompt + answer = 1,024, the cap the fork's pipelined engine puts on the dynamic KV on iOS:

| form, on an iPhone 17 Pro | prefill / decode tok/s | a 768 + 256 turn | thermal | answers |
|---|---|---|---|---|
| A. `int8hu`, `main` only, compiled ahead of time for the GPU, on the pipelined engine | 10.0 [9.2–11.0] / 9.25 [8.0–10.7], the 4 warm trials (the cold trial: 12.1 / 10.5) | 104.37 s | nominal → fair in the first run, serious from the second (all 4 warm trials at serious) | the 8 questions' greedy tokens = the Mac's on 8 / 8; the token gate 16 / 16 in all 3 runs; stops where the Mac stops |
| B. the static Neural Engine form, on Apple's `StaticShapeEngine` | 339.6 [336.1–340.6] / 32.86 [32.36–33.17], n = 6 | 10.02 s | fair at the start and end of both speed runs (serious in the three correctness runs) | top-1 692 / 721 = 95.98 % teacher-forced vs transformers fp32 (bar 93 %); the 8 questions' verdicts of transformers bf16, 8 / 8; stops on `<\|im_end\|>` |

- A is the shipped weights in a `main`-only build, compiled for h18p with `--expect-frequent-reshapes` (6.39 GB:
  the int8 Linears' fp16 copies), on the pipelined engine with an S=1 prefill. B is
  `conversion/export_lfm3_ane_static.py` over `models/ios/lfm3.py`: 8-bit palettized weights, the tied head as an
  int8 table, contexts up to 4,096; `.aimodel` 1.76 GB, compiled for h18p 2.15 GB with each of the 30 transformer
  functions in a Neural Engine region.
- The turn column is computed from the medians (768 / prefill + 255 / decode), not timed. B's top-1 counts the 721
  reply steps of the 17 sequences (up to 64 a sequence, after its prompt); the Mac checks above count all 2,682
  positions of the same sequences, prompts included.
- Per turn the static form came out 10.4× faster (prefill 34×, decode 3.6×); the two forms were measured in
  separate runs 40 minutes apart, not alternated, and A's warm trials ran with the phone at thermal serious. A was
  ready 13.2 s after a cold launch and 2.9 s after a warm one, at a peak footprint of 0.205 GB; B loaded in 51.8 s
  the first time, in 0.2 s for the two correctness runs after it, and in 6.8 and 0.07 s for the two speed runs
  after a reinstall, at a peak footprint of 2.79 GB.
- The static form has not been checked for thinking on, tool calls or the CoreAIKit path. A static iOS bundle is
  checked on the phone only, so the Mac's checks above do not cover it.
- The `.aimodel` specialized on the phone (device JIT) was not tried on the 17 Pro: the device JIT of two other
  LFM3 decoders on that phone had ended without an engine, with jetsam events. On an iPhone 18 Pro the int8hu
  `.aimodel` (`main` only) had not built its engine 300 s after launch (the cause is not isolated); the shipped
  `_pf64` `.aimodel`, the h19p AOT and the h19p static form were not run there.

Mac speed is on the card.
