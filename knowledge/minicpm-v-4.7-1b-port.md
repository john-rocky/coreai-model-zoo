# MiniCPM-V-4.7-1B: canvas M-RoPE on the Qwen3.5 hybrid, on Core AI

> 2026-10-13. openbmb/MiniCPM-V-4.7-1B (revision `984a89a`, Apache-2.0) is MiniCPM-V-4.6's vision code in front of a
> Qwen3.5 hybrid decoder with a new position scheme, canvas M-RoPE, and an untied head. The port is one re-authored
> tower, one new decoder module on the decider-2b-vision contract, and the engine's existing static-input and
> extra-state hooks. Card: [`models/minicpm-v-4.7-1b/README.md`](../models/minicpm-v-4.7-1b/README.md). Transcripts
> are in `models/minicpm-v-4.7-1b/`; records outside the repo are marked "lane" and live under
> `$ZOO_WORK_ROOT/_minicpmv47/`.

## What was reused and what is new

| part | source | change |
|---|---|---|
| text decoder | overlay `models/macos/qwen3_5.py`: `Qwen3_5StatefulForCausalLM`, `_mrope_freq_masks`, `forward_stateful_core(cos_sin=…)`, the loop-free Gated DeltaNet step | none in the overlay; the config is written by hand (the overlay's transformers 4.57.6 has no `minicpmv4_7`), the untied head is loaded from `lm_head.weight` |
| VL decoder module | decider-2b-vision's contract (`conversion/decider_vision/qwen3_5_vl_pipelined.py`: ids in, image rows as a static input, M-RoPE derived in the graph) | new `conversion/minicpmv47/minicpmv47_vl_pipelined.py`: the canvas rule instead of the Qwen rule, the 8×8 grid baked as a constant [64, 2] int32 table, no `image_rc` input |
| vision tower | transformers 5.19.0 `modeling_minicpmv4_7.py` (4.6's classes renamed); the 4.6 port's own tower module is not in the repository | new `conversion/minicpmv47/minicpmv47_vision.py`: a reference path in Hugging Face's order and a baked path (`bake_constants(32)`, one batched window attention) for export |
| host | none to reuse | `host.py`: Pillow BICUBIC 448 → (x/255 − 0.5)/0.5; `host_static_inputs` for ids, start and amount |
| exporters, AOT, gates | the decider and 4.6 export pattern; `conversion/coreai_gate.py`'s knife-edge rule | new scripts in `conversion/minicpmv47/` |
| device harness | PipelinedBench's GLM-OCR form (binds `image_embeds` + `rope_shift_start` / `rope_shift_amount` by name) | its reference directory became a spec field (`vlRefDir`; it was the fixed name `glm_ocr_ref`) + a 4.7 spec |

The 4.7 vision classes are 4.6's after renaming: `modeling_minicpmv4_7.py` lines 55–604 equal `modeling_minicpmv4_6.py`
lines 50–599 with `4_7` → `4_6` (lane `FEASIBILITY.md` §r1, `cmp`).

## The canvas rule and why the gate is on the planes

- One 448×448 image after plain text, O = index of `<image>`: text before (p, p, p); `<image>` (O, O−1, O−1); row k
  (O, O + k//8, O + k%8); `</image>` (O, O+8, O+8); the `"\n"` after it (O, O, O); everything from `start` on
  (p − amount) × 3. amount = start − (O + 9). The card has the table.
- With this chat template the frame is followed by one `"\n"` (198), so start = i0 + 67 and amount = 58 on every
  image prompt measured (O = 6, 6, 6, 17).
- In fp32 torch the rule equals the planes Hugging Face's rotary received: 433/433 prompt positions, 458/458 generate
  positions, every generated token at p − amount 92/92 (lane `oracle/parity_vl_torch.json`).
- Red arms in torch: amount + 1 fails 4/4 (only the text before the frame still matches), 1-D planes fail 4/4.
- **A last-position logits bar passes the wrong scheme**: with 1-D planes (4.6's) the last-position top-5 stays equal
  on all four image prompts, cos 0.9981–0.9996 against the oracle's 0.99999999999. Gate the planes themselves.
- On the engine (Mac AOT, values only): the 1-D-planes arm moves the logits at the oracle's top-5 by 11.98 against
  the green run's 0.372 (red); amount + 1 is red on the fp16 decoder (0.640 vs 0.140) and not on int8hu (0.446 vs
  0.372), so the engine-side plane arm is `planes_1d` (`gate-minicpm-v-4.7-1b.json` `red_arms`).
- The oracle records planes with a pre-hook on `language_model.rotary_emb`: the plain forward hands [3, 1, S],
  `generate` hands [4, 1, S] then [4, 1, 1], and a text-only prompt hands None (the text model counts positions).
- `MiniCPMV4_7Model.compute_3d_position_ids` adds the stored `rope_deltas` whenever the cache is non-empty and no
  `target_sizes` are passed: a manual cached loop after an image prompt inherits that image's shift. The oracle resets
  `model.model.rope_deltas = None` per prompt (the stale case was read in code, not run).
- Core AI lowers the constant [64, 2] int32 table and its `index_select` correctly (G2 4/4, the 1-D-planes arm red).
- Text-only: start = 2³⁰, amount = 0; in fp32 torch the module then equals the overlay's plain text decoder bit for
  bit.

## The head is untied but 4.9e-4 from the embedding

- `lm_head.weight` vs `embed_tokens.weight`: max |Δ| 4.9e-4, relative Frobenius 1.35e-3, row cosine ≥ 0.99998, no
  identical row (lane `oracle/parity_text.json` `head_vs_embed_weights`).
- The embedding as the head (red arm, P0's last 48 positions): argmax 48/48 and cosine 0.999998 survive; the ordered
  top-5 drops to 46/48 and max |Δ| rises from 2.4e-4 to 0.196. Keep the ordered top-5 in a head bar.
- int8hu has nothing to untie: 187 linears int8 per block of 32 (186 body `symmetric_with_clipping`, the head
  `symmetric`), the embedding fp16; export 41 s, quantize 10 s.
- The quantizer prints "Tensor size 1 along axis 1 is not divisible by block size 32. Skipping quantization" 18 times;
  18 is the Gated DeltaNet layer count and the depthwise Conv1d weight [6144, 1, 4] is the one weight with size 1 on
  axis 1, and the record shows Conv1d not quantized (which modules printed it was not traced).

## fp16: the tower's norm inputs overflow fp16 when squared, and the GPU's fp16 tower passes

- fp32 activations (P1–P4): largest 1,498.8 (tower layer 26 residual, P2), text ≤ 76.4, none above 30,000 (lane
  `oracle/fp16_precheck.json`).
- One residual channel is written by layer 9's MLP (`fc2` absmax 1,299) and carried to `post_layernorm`: 35 LayerNorm
  inputs (layers 10–26, `post_layernorm`) reach a row sum of squares of 2.39e6 (> 65,504); row mean(x²) ≤ 2,077.
- CPU torch fp16 (weights and math) against the fp32 tower: cos min 0.99983, rows below 0.999: 0, max |Δ| 1.8e-2;
  fp16w32 (fp16 weights, fp32 math): max |Δ| 2.2e-5.
- The Mac GPU (AOT h16c) fp16 tower against the oracle: cos min 0.999957, max |Δ| 0.0093, which meets the fp16w32 bar
  (mean ≥ 0.9999, min ≥ 0.999); no overflow showed (`gate-minicpm-v-4.7-1b-tower-fp16.json`).
- fp16w32 on the GPU: cos min 0.9999999986, and 25.5 % slower (55.03 vs 43.85 ms); it does not ship.
- The tower on the Mac Neural Engine (`--preferred-compute neural-engine`, fp16): 31 ANE regions, 177.3 ms warm
  against 43.85 ms on the GPU, cos min 0.99627 (below the fp16w32 bar) → the GPU stays (lane `r4/ane_tower.json`).
  The int8lin tower compiled to 24 regions and was not timed.

## int8lin is stored as fp16 on the Mac AOT and on the phone

- Mac h16c AOT: the int8lin tower's `resources.bin` is 1,087,609,504 B, the fp16 tower's size, with different
  contents (sha256 `261c57c2…` vs `152bc955…`); its `stats.json` still counts Int8 storage (541,578,240 elements).
  Compile 24.3 s vs 5.9 s.
- iPhone 18 Pro: the specialization cache holds a 1,087,609,504-B `resources.bin` for both towers, and
  1,308,616,980 B for the int8hu decoder (its `.aimodel` main.mlirb is 1,308,566,922 B: the decoder stays int8).
- Speed: Mac 43.81 (int8lin) vs 43.85 ms (fp16), iPhone 67.8 / 68.5 vs 66.5 / 68.6 ms (order reversed between pairs).
- What int8lin changes: download 614 MB vs 1.09 GB; per-token cos min 0.99411 vs 0.999957; answers identical (G4).
- The decoder on the Mac Python runtime: int8hu 7.44 vs fp16 7.42 ms per S=1 call interleaved in one process.
- An AOT compile with `--expect-frequent-reshapes` is larger than the `.aimodel`: decoder int8hu 1.31 → 2.81 GB, fp16
  2.01 → 3.52 GB. The towers (compiled without the flag) land at the fp16 size: int8lin 0.61 → 1.09 GB, fp16 1.09 →
  1.09 GB.

## The engine on the iPhone 18 Pro

- PipelinedBench (Swift, Release, Xcode 27.0 RC, `increased-memory-limit`), iOS 27.2 24B5099f, the `.aimodel` files
  specialized on the phone (JIT). Engine ready 6.4 s cold, 0.2 s warm.
- Decode 98.0 tok/s on a warm launch, 96.0 cold, 97.1 with the HF tower's rows; prefill 105.7 tok/s at S=1
  (`COREAI_CHUNK_THRESHOLD=1`), p = 128, g = 256, text-only trial (`gate-minicpm-v-4.7-1b-iphone.json`).
- g = 1024 asked: 96.9 tok/s, completed without an error; the engine's iOS dynamic-KV guard caps prompt + generated at
  1,024 positions (896 decoded by the code path; the app prints no count).
- Numerics on every launch: nat (P0) 23/23 and oracle (P1) 32/32 = the fp32 oracle token for token, with the Mac
  tower's rows and with the HF rows; P0's knife edge at step 8 (gap 0.0553) went the oracle's way.
- The GLM-OCR form binds exactly this decoder's three static inputs by name; the only harness change was the
  reference directory as a spec field.
- PipelinedBench strong-links FoundationModels; the iOS 27.0 RC SDK build launched on iOS 27.2 24B5099f (a probe
  launch right after install checks dyld, which the Mac cannot).
- The cold specialization took 1.4 GB of device space (32.2 → 30.8 GB free); the container's caches ended at 3.50 GB.
- The container's opening tower call took 2,214 ms, later launches' cold calls 82–91 ms with either form (Metal
  pipeline compile vs graph specialization not separated).
- Towers on the phone = the Mac's same form at cos 0.999997 (max |d| 0.007 int8lin, 0.013 fp16).
- The same bundles compiled on the Mac for the phone (`--platform iOS --architecture h19p`, the decoder with
  `--expect-frequent-reshapes`, 2.81 GB) stay exact (23/23, 32/32 on three launches) and load in 1.3 s at the initial
  load (JIT 6.4 s), but the AOT decoder decodes at 51.4 tok/s against the JIT's 98.0 / 98.3 in the same session (1.9×
  slower; the asset holds an fp16 copy of the Linears beside the int8 weights, 2.15× the bytes of the JIT cache entry);
  the AOT tower runs at the JIT's speed (66.4 / 69.0 vs 66.3 ms). The JIT `.aimodel` ships (lane `r5/DEVICE.md`).

## Prefill in chunks and a second turn (Mac, not uploaded)

- `export_decoder.py int8hu --prefill-chunk 16` (S=1 `main` + S=16 `prefill`; `export_qwen38vl_pipelined`'s
  externalize retry fixed the static-S SDPA guard of the 6 full-attention layers on the initial export) answers P1–P4
  token for token as the S=1 order and the oracle, and cuts P1's 88-token TTFT from 673.3 to 162.7 ms (4.14×); the
  last prompt position is not bit-equal to the S=1 order (cos ≥ 0.999992, max |Δ| ≤ 0.0625), and S=32 would not help
  P1 (88 = 2 × 32 + 24, and those 24 S=1 calls alone take 178 ms) (lane `r5/chunk.json`).
- A second question on the same image: the template renders the conversation as turn 1's ids + its answer + 248046,
  198 + the new turn, so the state after turn 1 is reusable — 20 calls (174.3 ms) instead of 140 (1,318.2 ms), the
  answer = the fp32 oracle's 32 tokens, the carried and re-prefilled last-prompt logits bit-equal (lane `r5/state.json`).

## CoreAIKit (VLChat)

- The kit's MiniCPM-V 4.7 entry renders the processor's frame (`<image_id>0</image_id><image>` + 64 pads + `</image>`
  + `"\n"`), shifts the canvas from the index where the pads begin + 66 by 58 (Qwen-VL's rule would give + 64 and 56:
  4.7 counts `</image>` and the `"\n"` inside the frame), and puts `<think>\n\n</think>\n\n` after every assistant role
  (thinking off); the rendered ids equal the processor's for P0–P4 and a two-turn prompt (140 ids).
- Prompt order: the source processor and card put the image before the question; the kit renders a prompt's segments
  in order and 4.7's template joins a turn's parts with `"\n"`, so the question placed before the image is a different
  id sequence from anything the oracle ran — the kit's VLM examples were changed to put the image before the question.
- On the iPhone 18 Pro the VLChat app needs `com.apple.developer.kernel.increased-memory-limit`: without it the tower's
  MPSGraph specialization threw `std::bad_alloc` (`foldTransposeOp`) once the decoder's engine was up; signed with it,
  the app's answers on P1–P4 equal the Mac kit's byte for byte, from the uploaded bundles.

## The bytes that ship

- coreai-torch 0.4.1 writes each op's source location into `main.mlirb`: the three exports held 19 / 12 / 7
  `/Users/` strings (with the user name: 57 / 36 / 21 needle hits), and their AOT `.aimodelc` carried them too
  (`main-h16c.mlirb`, the MPSGraph `*.mpsgraph` files). A text-only `grep -rIl` skips both.
- `strip_debug_info` (via `AIModelAsset.load` → `program.save_asset` with the source's metadata) removes all of them:
  op counts and `coreai-build inspect` unchanged, main.mlirb 457 KB / 257 KB / 196 KB smaller, the AOT h16c
  `resources.bin` byte-equal to the export's, the MPSGraph packages smaller (decoder `specialized_model_1.mpsgraph`
  1,050,915 → 622,419 B) (lane `r6/strip_*.json`, `r6/regate_stripped.json`).
- The stripped AOT gets a new `main.hash`, so the Python runtime specializes it into a new cache entry instead of
  reusing the export's (`~/Library/Caches/coreai-cache/<os build>/python/<main.hash>`).
- The stripped int8lin tower + int8hu decoder re-ran G4: greedy ids, tower statistics and last-position logits
  statistics equal the export's on P1–P4.

## Traps

- `devicectl device copy to` skips a destination file of the same size whose mtime is newer than the source; the
  bytes check passes, a pulled md5 does not. `touch` the source before re-pushing.
- `push … && run … | cut`: the pipe's status is `cut`'s, so a failed push did not stop the run; keep guards out of
  pipes or read `pipestatus`.
- Separate processes timed the same decoder up to 20 % apart (the second process of a window was slower both times);
  compare forms interleaved in one process, rotated per round.
- While another job holds the machine's measurement window, this workspace's shell guard refuses a
  `python3 - <<'EOF'` heredoc and even `xcodebuild -version`; edit files with the editor, read JSON with `jq`, read
  Xcode's version from its `Info.plist`.
- `quiet_wait` looks once at the start; a long job launched through it alone keeps running through windows that open
  later. Wrap it in `quiet_yield --poll 1`.
- `python -I` ignores `PYTHONDONTWRITEBYTECODE`; add `-B` or the worktree collects `__pycache__`.
- zsh: an unquoted `echo =====` is `=cmd` expansion and exits 1; quote globs such as `--include=*.py`.
- `AIModelAssetMetadata(author=…)`: keyword arguments are dropped silently; assign the attributes.
- Batched window attention on views of the projection (strides 4608, 72, 1152, 1) rounds differently from the
  per-window calls by an ulp, which the layer-9 channel carries to 2.3e-5 at the output; `.contiguous()` on q/k/v
  makes the batched path bit-equal.
- A per-stage bar of 1e-3 absolute calls layer 26 (absmax 1,425, max |Δ| 2.0e-3 = 1.4e-6 relative) a departure;
  scale it by the stage's absmax.
- A hook closure that captures a dict rebuilt in a loop writes every layer into the last dict; bind it as a default
  argument and assert that every stage fired on every prompt.
- `tee` to the same log path loses the previous run's log; name logs per attempt.

## Numbers of record

- Mac M4 Max, Python runtime, AOT h16c, interleaved: decoder int8hu 7.44 ms per S=1 call (134.4 tok/s), fp16 7.42
  (134.9); towers int8lin 43.81, fp16 43.85, fp16w32 55.03 ms (lane `r3/timing_run3.json`).
- iPhone 18 Pro, PipelinedBench, JIT: decode 98.0 tok/s warm, prefill 105.7 tok/s, engine ready 6.4 / 0.2 s; towers
  67.8 / 68.5 ms (int8lin), 66.5 / 68.6 ms (fp16).
- Export walls: decoder int8hu 41 s, fp16 32 s; towers fp16w32 5 s, int8lin 17 s, fp16 4 s. AOT h16c: 37.0 / 10.9 /
  2.6 / 24.3 / 5.9 s; ANE regions 0 on all five GPU compiles.
- The decoder's static iOS Neural Engine path is a no-go for this hybrid
  ([`qwen3.5-static-ane.md`](qwen3.5-static-ane.md)); the tower's Mac ANE arm is above.
