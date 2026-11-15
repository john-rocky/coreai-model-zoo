# Kev: a LoRA + pointer-head decision model on Core AI, at 0.8B and 4B

> 2026-10-04. jaredpalmer/kev-0.8b and kev-4b (tag `v1.0`, Apache-2.0) answer typed questions about a state with a
> probability for every option: a rank-16 LoRA on Qwen3.5-0.8B-Base / 4B-Base and a pointer head that scores each
> option's closing token against the question's last token. The port is the author's merge, the overlay's text
> decoder returning hidden rows, and the head on the host. Both published graphs take 128 tokens a call and run each
> Gated DeltaNet recurrence in the overlay's fp32 Metal kernel; the earlier graphs took 16 tokens a call, unrolled.
> Cards:
> [`models/kev-0.8b/README.md`](../models/kev-0.8b/README.md), [`models/kev-4b/README.md`](../models/kev-4b/README.md).
> Transcripts below are in `models/kev-<size>/`; records outside the repo are marked "local" and live under
> `$ZOO_WORK_ROOT/_kev/`. Mac M4 Max, macOS 27.0 26A428, Xcode 27.0 RC; iPhone 18 Pro, iOS 27.0 24A437.

## What was reused and what is new

| part | source | change |
|---|---|---|
| merged weights | the author's `scripts/merge_lora_checkpoint.py` (tag `kev-1.0`) | none: run as is, its output read through a local HF-cache id |
| decoder | the overlay's `Qwen3_5StatefulForCausalLM` (`forward_stateful`; the Gated DeltaNet recurrence unrolled, chunked in the graph, or in the overlay's fp32 Metal kernel) | subclass `conversion/kev/qwen3_5_kev_decoder.py`: `lm_head` = identity, the final-norm hidden state at every position out, one static-S function; no code change between 0.8B and 4B |
| oracle | the author's package (`Checkpoint.load`, `kev.model.admit`, row-form `forward`) | `oracle_kev.py` runs it unchanged and asserts the readout on every record |
| head | the author's `PointerHead` (`head.pt`) | `export_head.py` writes `head.safetensors` + `kev_head.json`; the hosts run it in float64 |
| host | clef-flash's `host.py` / `decide.py` form | new `conversion/kev/host.py` (the author's `to_record`, `render`, `user_tokens`, `encode`, `to_answers` without the package) and `decide.py` |
| gates | clef-flash's readout gate, bisect and Swift scorer | new `readout_gate.py`, `parity_*.py`, `int8_bisect_torch.py`, `gate_swift.py` |
| Swift host | ClefFlash / DeciderVision's low-level runtime pattern | new `apps/Kev` (library + CLI), `apps/KevGate` (device gate) |

## The readout contract, and the token ids not to trust

One causal row per question: `[<|fim_prefix|>] + state + [<|fim_middle|>] + instructions + for each option
([<|box_start|>] + option + [<|box_end|>]) + [<|fim_suffix|>]`, positions 0..L − 1, fresh states. The head reads
the hidden state at `<|fim_suffix|>` and at every `<|box_end|>`: `z = (k(h_opt) · q(h_decide)) / 16`, `p =
softmax(z / T)` within the question (T 2.3510958125672174 / 2.406050072164233).

- **The delimiter ids come from the base tokenizer**: 248060 / 248061 / 248049 / 248050 / 248062, pad = eos 248044,
  no bos (`results/tokenizer_check.json`, local). The `kev-4b` repository still carries Qwen3-era `vocab.json`,
  `merges.txt`, `added_tokens.json` and `special_tokens_map.json`, whose delimiter ids (151659…) belong to the older
  tokenizer. The author's loader reads the tokenizer from the base (`load_tokenizer(meta.base, …)`); so do the bundles
  (`tokenizer/` = the base's `tokenizer.json` + `tokenizer_config.json`, verbatim). The two bases' `tokenizer.json`
  are byte-identical; their `tokenizer_config.json` differ only in the chat template's thinking default.
- **User text cannot produce a delimiter**: `<|name|>` is rewritten to `<¦name¦>` before tokenizing without special
  tokens (`user_tokens`).
- **The response is the author's**, including its quirks: `output_tokens` counts the tokens of Python's
  `json.dumps(answers)` (so a printed `0.25` and `0.2501` differ), the body's `model` echoes the request's (the
  fixture's requests say `kev-0.8b`, so the 4B oracle's bodies do too; the hosts write the bundle name), and a
  `score` confidence jumps when two levels of a near-tie swap the mode (a `score` question of `own_t02` on 4B, oracle
  margin 0.0024: 0.3058 on the graph against 0.1813) — compare probabilities, not confidences (`gate-kev-4b-host.json`).
- The Hub tag `v1.0` (`788ddbdd` for kev-0.8b, `591dcb5b` for kev-4b) is a tag object. For kev-0.8b the Hub resolves
  it to commit `bf75a6a8`, whose LFS objects equal the card's weights revision (local `results/source_sha256.json`);
  kev-4b resolves the same way (local `ROUND4.md`). huggingface_hub 1.32 (the author's
  lock) resolves a 40-hex revision offline only after one online `snapshot_download` with that version has written
  its tree listing.

## The merge is the author's script, and the proof is bit equality

`scripts/merge_lora_checkpoint.py` folds `W + (B @ A) · α / r` (α / r = 2) in fp32 and writes a full-weight checkpoint
the author's loader reads (`head.pt` weights "full"). Through that loader the merged checkpoint gives the adapter
checkpoint's logits bit for bit on 434/434 questions for both sizes, and three tensors per size equal the formula bit
for bit (the adapter moves them by 2.8–3.6 % at 0.8B, 1.4–2.0 % at 4B). The red arms show the check can fail: the base
alone moves p by up to 0.789 (25/53 argmax), the base's Gated DeltaNet projections put back by up to 0.282
(`gate-kev-0.8b-merge.json`, `gate-kev-4b-merge.json`).

The merge is reproducible. Kev-0.8B's merged checkpoint was deleted after round 15 and written again by the same script
in round 16: `model.safetensors` (`b5ebf92a…`) and `head.pt` (`c33e2288…`) came out byte-identical to round 1's. Only
`merge.json` differs, by its phase times, so its sha256 is not the one `gate-kev-0.8b-merge.json` lists.

## The oracle runs on one thread

transformers' reference causal conv (no `causal_conv1d` package) is a depthwise `F.conv1d` with 6,144 groups
(8,192 at 4B), which the CPU runs as many tiny calls; their thread start-up dominates. The decoder module's probe on
the same conv: `F.conv1d` at 1 / 4 / 12 threads = 14.8 / 86.9 / 262.7 ms per token at 0.8B with bit-identical hidden
rows; the same conv as one `bmm` over the windows = 7.5 ms at 1 thread (`gate-kev-0.8b-torch-parity.json` `probe`).
At 4B, 12 threads also change the bits (`gate-kev-4b-torch-parity.json` `probe`). The oracle runs with one thread
(`oracle_summary.json` `threads: 1`); the parity harness uses the `bmm` form and checks it against `F.conv1d` at the
last chunk of every row (7,812 checks at 0.8B, 10,416 at 4B, all bit-identical).

## The decoder is the overlay's text graph without a head

`Qwen3_5KevDecoder` is the overlay's stateful text decoder with `lm_head` replaced by the identity: `load_report`
reads 320 / 426 keys and leaves none, and the checkpoint has no `lm_head` key (tied). Making the head an identity
also keeps the tied embedding out of any int8 pass that targets linears. In fp32 torch, driven like the graph, it
gives the oracle's p within 3.5e-6 (0.8B) / 6.6e-6 (4B) on all 434 questions, and the overlay's plain
`forward_stateful` bit for bit (3 rows, every chunk and layer).

- **One function at a static S, no S = 1.** Nothing is generated. A row of T ids runs ⌈T / S⌉ calls from zeroed
  states (S = 128 in both published graphs, 16 in the earlier ones); the last call is padded with 248044
  and its rows dropped (the graph is causal).
- **The row limit is 3,968 tokens at S = 128 and 4,080 at S = 16, not 4,096.** The position input's upper bound is
  `max_context_length − 1` = 4,095, so the last padded call must end at or below it. The author's server accepts
  states of 65,536 tokens and rows of 73,728; the hosts refuse a longer row before any call
  (`host.graph_context_check`, `KevError.graphLimit`).
- **The longest rows read correctly on the S = 16 graph.** No diagnosis of this kind was run on the published K128
  graph. A diagnosis on rows of 4,072 / 4,030 / 3,039 / 2,997 tokens built from the
  fixture's long states: every argmax equal, max |Δp| 0.0023 (0.8B) / 0.00046 (4B); hidden cosine to the oracle at
  least 0.99994 everywhere at 0.8B and 0.99902 at and after position 4,032 at 4B. At 4B eight positions earlier in
  the state's text read a cosine below 0.99 (lowest 0.46 at position 673), while every readout position stays above
  0.99999 (local `results/longrow_kev-<size>.json`). Why those positions drift is not isolated.

## The scan form sets the cost of a call

Each Gated DeltaNet layer runs its recurrence inside the call. The overlay has three forms of it, chosen with
`--gdn-scan` in `conversion/kev/export_decoder.py`. The graph's inputs, outputs and states are the same in all three.

| form | `--gdn-scan` | what runs inside a call |
|---|---|---|
| unrolled, U*S* | `unroll` | the S steps, one after another, fp32 |
| in-graph chunk, C*S* | `chunk` | the overlay's `_gated_delta_chunk`: the S tokens at once, the triangular inverse as ⌈log2 S⌉ doublings, fp32 |
| Metal kernel, K*S* | `metal` | the overlay's fp32 Metal chunk kernel (`qwen3_5_gdn_metal`), one dispatch per layer for the whole call |

A call's time fits a + b·S, with S the tokens in the call. The table gives a and b on the M4 Max (macOS 27.0,
2026-10-04; `gate-kev-<size>-forms.json` `call_cost`), for Kev-0.8B unless a row names Kev-4B. A window marked shared
ran beside other GPU work.

| form | window | a, ms | b, ms per token |
|---|---|---:|---:|
| unrolled | round 11 screen, shared | 1.7 | 1.04 |
| in-graph chunk | round 11 screen, shared, two points | 9.2 | 0.10 |
| Metal kernel | round 11 screen, shared | 7.9 | 0.17 |
| unrolled | round 11 lock window, two points | 0.34 | 1.387 |
| Metal kernel | round 11 lock window | 8.84 | 0.166 |
| Metal kernel, static S | round 14 | 8.52 | 0.169 |
| Metal kernel, dynamic length, cap 128 / 256 / 512 | round 14 | 7.99 / 7.34 / 7.44 | 0.184 / 0.181 / 0.182 |
| Metal kernel, iPhone 18 Pro | round 13 bench, S = 16 to 128 | 12.81 | 0.181 |
| unrolled, Kev-4B | round 12 screen, shared | 16.24 | 2.509 |
| Metal kernel, Kev-4B | round 12 screen, shared | 22.84 | 0.700 |

- **An unrolled call is almost all per-token cost.** A wider S saves calls, not time. Round 2 timed S = 16, 32, 64
  and 128 in their own windows: 18.8, 40.2, 79.9 and 167.0 ms per call, 1.17–1.30 ms per padded token. In one
  interleaved run over the fixture's 423 warm rows, S = 16 took 72.8 s of graph calls and S = 32 74.4 s. S = 32 pads
  10.4 % of the fixture, S = 16 5.1 %.
- **A kernel call is mostly its fixed part a.** What a is made of was not isolated. A graph with a dynamic query
  length costs what the static kernel of the same length costs (round 14's rows above, one window).
- **The in-graph chunk breaks between S = 16 and S = 32, already in fp32 torch.** On the 21 rows the oracle keeps
  hidden states for, C32 is off the unrolled form by up to 6.4 in the hidden state and 0.12 in p; the lowest
  position cosine to the oracle is 0.845 and 20 of 21 argmaxes hold (local `results/r11_torch_chunk.json`). C16 is off
  by 4.4e-4. On the GPU, C32 fails the screen: non-near-tie argmax 112/127, max |Δp| 0.125, non-finite rows
  (`gate-kev-0.8b-forms.json`). So the GPU failure is the algorithm, not fp16.
- **The kernel is not bit-equal to the unrolled form, and both pass.** The kernel keeps the recurrent state in fp32
  for the whole call and computes the q / k l2-norm in fp32; the unrolled form computes the l2-norm in fp16 (the
  overlay's source). On the 130-row screen, the largest difference in p between U16 and each of K16 to K128 is 0.0015
  to 0.0021.
- **The kernel has no torch values.** Its `torch_defn` returns zeros of the right shapes, so torch cannot check
  it. The readout gate against the author's fp32 oracle is its only parity check.

## A dynamic query length: measured, not shipped

`--dynamic-query` exports the Metal-kernel form with a query length of 2..cap (`input_ids [1, -1]`); the kernel is
built for the cap. Round 14 measured caps 128, 256 and 512 on the M4 Max; round 15 gated and timed the cap-512 graph.

- **A call costs what the static kernel of its length costs** (the table above). A row pays for its real tokens,
  not for whole calls.
- **The host fixes the call lengths.** It reads L (the longest call) and q (every call length a multiple of q) from
  `metadata.json`, cuts a row into calls of L ids and pads the last call up to a multiple of q (`host.plan`). A static
  graph is the case L = q = S, so the hosts read both kinds of bundle.
- **Shared and direct runs stop being bit-equal.** The shared prefix ends at a multiple of q, so its calls are cut
  at other places than a direct run's. On the cap-512 graph both pass the bar: the shared fixture pass has max |Δp|
  0.0128, at most 0.0022 from the direct p (local `results/r15_gate_0_8b.json` `shared_vs_direct`). A prepared state
  equals the shared prefix bit for bit.
- **It does not ship.** In round 15's lock window it decided the 94-token question in 25.2 ms (the `.aimodel`
  specialized by Swift), against 29.9 ms for the static K128 graph and 100.1 ms for U16. Its footprint grows while
  the call length keeps changing (next section), on the Mac and on the iPhone.
- **Kev-4B: the same picture.** In round 12's lock window the cap-512 graph decided the 94-token question in 83.3 ms,
  against 100.2 ms for K128. Its two Swift timing processes ended at 6.31 and 6.32 GB of footprint, against 0.89 and
  0.95 GB for K128 (`gate-kev-4b-forms.json`).

## int8: measured, not shipped

**Kev-0.8B** (`gate-kev-0.8b-int8.json`): int8 per block of 32 over all 186 linears (`symmetric_with_clipping`) fails
(0.0431 / mean 0.00326; held out 0.0273). The fp32 bisect with the exporter's own int8 weights reproduces it (0.0420,
per-row correlation 0.972). Kept fp16 alone: one layer at best 0.0360 (layer 3); all MLP linears (53 % of the
weights) 0.0308; all Gated DeltaNet projections (38 %) 0.0359; all attention (9 %) 0.0390. The rule (worst < 0.010
with at most 50 % fp16, by single-layer then single-kind rank, then unions) found no set (best 0.0141 at 45.5 %).
Layers 0–11 fp16 (50 %, outside the rule) give 0.0046 on the bisect rows, 0.0089 on the fixture and 0.0043 on the
held-out set, at 85 % of fp16's bytes.

- **Rank layers by the mean, not the worst row.** Alone, layers 0, 7, 10 and 11 leave the worst row where all-int8
  put it (7 and 11 make it worse), so a worst-row ranking puts layers 12–23, which change nothing, ahead of them. By
  the bisect rows' mean every layer of 0–11 ranks above every layer of 12–23: the top 12 by mean is the set that works
  (local `ROUND3.md`).

**Kev-4B** (`gate-kev-4b-int8.json`): int8lin fails (0.0601; held out 0.0285). The instrument (40 rows, a rule written
first, at most 25 % fp16, target worst ≤ 0.006 and mean ≤ 0.0012):

| int8 variant, fp16 kept | share | worst | mean |
|---|---:|---:|---:|
| all, block 32, `symmetric_with_clipping` | 0 % | 0.0710 | 0.00689 |
| all, block 16 | 0 % | 0.0470 | 0.00453 |
| the embedding table only (body fp32) | 100 % | 0.0052 | 0.00056 |
| asymmetric block 32, `in_proj_qkv` + `out_proj` | 21.2 % | 0.0163 | 0.00208 |
| asymmetric block 32, `in_proj_qkv` + `out_proj` + `v_proj` | 21.7 % | 0.0164 | 0.00191 |

No set met the target; the rule took the two best by mean. On the Mac GPU both pass the 434 rows (0.0174 / 0.0175);
the one run on the held-out set fails on one `score` row (0.0277, `tv4sh_11`, fp16's held-out worst too).

- **A bisect's rows do not transfer between int8 variants.** The 40 rows came from the symmetric int8 gate's worst
  rows (per-row correlation with that gate 0.973); the worst 434-row rows of both asymmetric candidates
  (`semif_fda0ca94…`, `tv4x_composition_holdout_13`) are not among them. Re-pick the rows when the variant changes.
- The quantizer's `qscheme` accepts `symmetric`, `asymmetric` and `symmetric_with_clipping` (`affine` is refused by
  name); block 16 and an int8 embedding table quantize, export and compile (`quant_api`).
- `torch.nn.utils.parametrize` renames the class (`Embedding` → `ParametrizedEmbedding`): check quantized types with
  `isinstance`.

## What the compiled asset holds

- **`--expect-frequent-reshapes` adds an fp16 copy of every linear.** The MPSGraph package then holds
  `original_model_0` and `specialized_model_1`. iOS h19p: 0.8B fp16 2,505,674,625 B with the flag, 1,506,400,620
  without; 4B fp16 15,557,759,365 / 8,413,701,860; the 4B int8 candidates 13.03 / 13.01 GB with it. The difference is
  the fp16 bytes of the 186 / 248 linears, whatever the bundle's linears are (`ios_aot_bytes`).
- **A compiled graph holds int8 linears as fp16.** A toy (an embedding and two linears) compiled with a fixed length
  holds `resources.bin` 12,582,936 B on macOS and iOS with and without the flag — the fp16 toy's bytes — against an
  int8 `main.mlirb` of 10,623,080 B (`static_toy`, `quant_api.fp16_reference`). Only an unspecialized original keeps
  int8 bytes.
- **Without the flag, a dynamic-length graph specializes again for every new position length.** 4B fp16 on 33 rows:
  9.6–14.6 s for each new length, 46.4 ms per call at a length already seen, probabilities still within 0.0055 of the
  oracle. An int8 asset without the flag folds the dequantization into fp16 constants during that specialization
  (sampled: `specializeWithDevice` → `LowerDequantizeND` → `createOrFoldConstant` → `foldCastAttribute`) and had not
  finished its first length after 12 minutes (`efr_abba`, `resident_memory`; local
  `logs/r6_sample_int8_noefr_worker_11034.txt`). The runtime leaves the MPSGraph scratch of those specializations
  under `$TMPDIR/com.apple.MetalPerformanceShadersGraph` after the process exits.
- **Resident on the Mac.** A gate worker on the 4B efr asset: rss up to 16.6 GB, up to 16.4 GB of it clean mapped
  file, phys_footprint at most 1.15 GB. What an iPhone counts against an app for such an asset was not measured; the
  4B asset crashed before it loaded (below).
- So an int8 `.aimodel` downloads smaller, and its compiled asset is not smaller by the same amount; for Kev-4B none of
  the int8 sets reached an iPhone-sized asset with the flag.

## First calls and caches

- **A process's first call is slower.** With the static K128 graph the opening 128-token call of each Python gate
  process took 211–1,188 ms on the Mac, against a median of 29.9 ms for its later calls. On the iPhone the opening
  decision took 66.9 ms, against 37.8 ms in the bench of the same build (`gate-kev-0.8b-iphone.json`).
- **A dynamic-length graph pays a first call per new length.** On the Mac in round 14 (cap 128), a new length's first
  call took up to 2,458 ms in the process that had just made the runtime's cache entry. Later processes ran every
  length at its steady cost from the first call. Where that state is kept on the Mac was not found. On the iPhone
  (round 13) the cost came only at the first launch after install. During that launch the app's Metal shader cache
  grew and the Core AI cache did not change (local `ROUND13.md`, `device_r13/runs/listings/probe_d128_*/`). Whether
  removing that cache brings the cost back was not tested.

### An AOT cache entry is named by the function's type

- The Python runtime keeps every AOT asset it loads under `~/Library/Caches/coreai-cache/<build>/python/<sha256 of
  main-h16c.mlirb>/`. For the dynamic-length graphs that stub holds the function's type and the source file's path and
  hash. It does not hold the cap.
- So the cap-128 and cap-256 graphs, exported from the same version of `qwen3_5_kev_decoder.py`, share one stub hash
  and one entry name (`3fdbe104…`). Loading the cap-256 asset while the entry held the cap-128 graph ran the cap-128
  graph, with no error. At a call of 200 tokens its hidden rows were off the cap-256 graph's own by up to 34.3, all
  finite. At 64 tokens they were bit-equal (local `results/r14_cache_collision_D256.json`, `results/r14_cache_stale.json`).
- An edit of the source changes the stub: the cap-512 graph, exported after one, got its own entry.
- The `.aimodel` specialized by the Swift runtime keeps its entry under the `.aimodel`'s `main.mlirb` sha256
  (`ab60bacd…` for the cap-512 graph), so two different graphs do not share one.
- Before a gate on an AOT asset, check what the entry holds (the `manifest.plist` hash inside it), or move the entry
  aside when the cap changes.

### A dynamic-length graph's footprint grows while the call length keeps changing

Round 15 probed the cap-512 dynamic graph through the `.aimodel` specialized by the Swift runtime, on the M4 Max (local
`results/r15_leak_0_8b.json`, phys_footprint of one process):

- **Same length: the first call adds the memory, later calls almost none.** 200 calls of the same 96 tokens from zero
  states added 269.1 MB, 267.5 MB of it at the first call. 126 calls that kept the length and moved the position added 291.3 MB in the first
  lap of 42 calls, then 0.3 and 0.4 MB.
- **A new length each call: growth every call.** Three laps of one call at each length 16, 32, …, 512 added 1,433.7,
  1,125.2 and 1,122.4 MB, from 77.4 to 3,758.7 MB: 35.1 MB per call in the third lap.
- **The static S = 128 graph does not grow.** Its 93 calls added 282.3 MB in the first lap, then 0.1 and -0.1 MB.
- **What gives memory back.** After the same-length loops, a 2-second sleep lowered the footprint from 392.0 to 169.7 MB
  and from 368.5 to 145.2 MB. A minute more changed nothing (145.1 MB). Loading the function again changed nothing
  (145.1 → 145.2 MB). Creating the `AIModel` again lowered it from 388.2 to 133.0 MB, in 2.1 s.
- **Four lengths in turn keep growing, in Swift and in Python.** Cycling 128, 256, 384 and 512 tokens three times
  added 467.1, 166.9 and 168.5 MB per lap in Swift and 460.7, 169.5 and 169.6 MB in the Python runtime (`leak3`).
  Every set of 4 to 32 lengths probed grew on every lap.
- **Two lengths in turn stop growing.** Alternating 96 and 112 tokens grew only at the opening calls. Bringing in 128
  added 19.1 MB, and going back to 96 / 112 added 1.3 MB (`leak2`).
- **On the iPhone 18 Pro the same happens** (round 13, local `results/r13_c_memory.json`). A gate run of the cap-512
  graph grew from 199 MB after the load to 1,381 MB after warm_up (32 calls, 36.4 MB per call). It reached 3,497 MB
  after 370 records and 439 calls, with 43 MB available, and the app wrote nothing after that sample. At the last
  decision of the bench launches the footprint was 3,089, 2,092 and 3,069 MB for host calls of at most 128, 256 and 512
  ids, against 332 MB for the unrolled S = 16 graph and 309 MB for the static S = 128 graph.
- Why a new length keeps memory is not isolated.

## Host: float64 head, Python's sum, Python's text

- **The head in float64.** NumPy's fp32 head does not reproduce PyTorch's bits: on the same fp16 hidden rows the batched
  `k` projection (`addmm` with the bias) and the softmax do not give PyTorch's bits, while the one-row `q` projection
  does. The hosts compute the head and the softmax in float64 and round p to fp32 once; Swift's
  `vDSP_dotprD` and NumPy then differ by at most 8.9e-15 in z and give the same fp32 p on 1,128/1,128 rows. Against the
  gate's fp32 torch head the host's p differ by at most 5.1e-7 (`gate-kev-<size>-host.json`).
- **Python 3.12's `sum()`.** The author's confidences sum floats with Python 3.12's `sum()`, which uses Neumaier
  compensation; 3.11's is plain. The hosts implement the 3.12 algorithm; on these fixtures the plain sum changes none of
  the 1,028 answer sets (`host_test` `py311_plain_sum_changes_answers`).
- **The text the model reads is Python's `str()`.** `render()` prints values with `str()`: `1250` and `1250.0` differ,
  `2 ** 70` keeps every digit, `-0` prints `0`, `1e400` prints `inf`, `True` is capitalized. The Swift host keeps each number literal as
  written and copies `str` / `repr` / `round` / `json.dumps`: equal to CPython on 2,787 rendered values, 599 requests and
  203,635 doubles (`gate-kev-0.8b-swift.json` `render_test`). `str.lstrip` strips Python's whitespace set, not Swift's.

## The shared prefix is exact on a static graph

Every row of a request starts with the state's Ls tokens, so with a static call of S tokens and k = ⌊Ls / S⌋ the
first k calls are identical in every row. Run them once, copy the four states after them (the KV axis up to S·k, the
conv and recurrent states as they are), and run each row's remaining tokens from position S·k in the same call grid.
At S = 16 the hidden rows equal the direct run's bit for bit on 564/564 rows of each size, and the fixture's 20
multi-question records take 662 calls instead of 1,830 (`gate-kev-<size>-host.json` `shared_prefix`). With the
published K128 graph the Swift host's shared rows equal its direct rows on 434/434 and 130/130 rows
(`gate-kev-0.8b-swift.json`). A graph with a dynamic query length cuts the calls elsewhere (above). `coreai.runtime.NDArray(array)` copies its array, and
`state[name].numpy()` reads what the calls wrote: read the states back once, then build fresh arrays per question.

## Swift host notes

- `Sources/Kev` (the library) and `Sources/kev` (a CLI) are one directory on a case-insensitive volume, the Mac's
  default APFS: the CLI lives in `Sources/kev-cli` (target `KevCLI`, product `kev`). Xcode lists schemes `Kev` and
  `kev`; `-scheme Kev` builds the library.
- **JIT = AOT at both sizes.** The `.aimodel` specialized by Swift (GPU preferred, `expectFrequentReshapes`) equals the
  h16c AOT asset bit for bit on 25 rows of 10 records: 0.8B specialized in 5.4 s and added 2,510,934,016 B to the
  runtime's cache; 4B in 9.4 s, +15,574,564,864 B, no MPSGraph scratch (local `results/swift_jit_kev-<size>.json`).
  Those were the S = 16 graphs. The published K128 graphs also equal their AOT assets on 25 rows; they specialized in
  3.34 s (0.8B) and 16.6 s (4B, a 15.55 GB cache entry) (`gate-kev-<size>-swift.json`). The
  cache entry is named by the `.aimodel`'s `main.mlirb` sha256, on the Mac and on the phone.
- One process answers all 384 fixture records; on the S = 16 graphs its footprint stayed within 556–621 MB (0.8B) and
  844–976 MB (4B) over the pass (local `results/swift_gate_kev-<size>.json` `sets.fixture.footprint`).
- **The GPU lock has two conventions.** This port's scripts write a tag into `_GPU_LOCK`; the kit's `scripts/with-gpu-lock.py`
  takes a `flock(2)` and opens the file with mode `"w"`, which empties it even while it waits. Check the file's text
  and `lsof` before GPU work; keep the lock's file object open under its own name for the whole window (a reused
  variable name closed it once and let another job in) (local `ROUND7.md`).

## Timing on a shared machine

Other jobs shared this Mac's GPU while the forms were timed. A time in the cards counts only if it was measured under
this rule, fixed before the timing runs:

- A process counts when no other GPU job ran during it and the one-minute load average was at most 12 at its start and
  at its end. A form needs two counted processes, and a window runs up to four rounds to get them. Medians are over the
  counted processes only; the others stay in the record (local `swift/timing/<run>/clean.json`).
- **The unrolled graph reacts to load.** Round 11's four U16 processes decided the same 94-token question in 177.3,
  106.5, 493.0 and 115.2 ms; one counted. The first ran at a load average of 23; in the same contended round K128 took
  31.0 ms, against 29.7 and 30.1 ms in its counted processes. The third U16 process overlapped other GPU jobs
  that no process-list pattern matched; their owner's log showed them afterwards. Round 7's two U16 processes gave 152.9
  and 104.1 ms.
- Inside one screen window the same U16 graph took 27.03 ms per call at the start, while this port's exports ran, and
  18.26 ms at the quiet end.
- **The lock does not make every GPU job visible.** A job that does not take it, or that runs from a command line the
  detector does not know, shows up only in its owner's log. Keep a `ps` sample per process, and read the other jobs'
  logs before calling a window clean.

## iPhone 18 Pro

Kev-0.8B through `apps/KevGate` (Release, no increased-memory-limit entitlement, 3,529 MB available at launch)
(`gate-kev-0.8b-iphone.json`). Round 8 gated the earlier upload's graph (U16), round 13 the Metal-kernel forms and the
dynamic-length graph, round 15 the published K128 graph:

- **The published graph passes on the phone.** Run `20261004-140048`, the `.aimodel` specialized on the device:
  fixture 420/420, near-ties 13/14, max |Δp| 0.0110; held out 125/125, 0.0059.
- **Time.** Round 13 benched the forms in one session per stage, with 60 s of rest before each item. The 94-token
  question took 36.5 ms with K128 and 157.1 ms with U16 (stage A), and 37.8 and 156.7 ms (stage C). A kernel call
  fits 12.81 + 0.181·S ms on the phone (the scan-form table above).

The rest of this section is round 8's, on the U16 graph:

- **Both load paths work at the default memory limit.** The `.aimodel` specialized on the device: 6.65 s cold (5.87 s
  in `AIModel(contentsOf:)`), +2,505,661,047 B of cache, peak footprint 316 MB; 0.61 s warm. The h19p AOT asset with
  `--expect-frequent-reshapes` (2,505,674,625 B): 4.64 s cold with `.default`, peak 223 MB.
  The device's free space did not drop for the AOT load's cache entry, as if it shared the asset's blocks (inferred
  from free space only).
- **The gate passes.** Device JIT 434 rows: 420/420, near-ties 13/14, max |Δp| 0.0102; held out 125/125, 0.0049; AOT
  90 rows 80/80, 0.0071. AOT and JIT agree bit for bit on the phone (90/90); no hidden row equals the Mac's (max |Δp|
  between them 0.0027).
- **Time.** A 16-token call takes 25.9–26.6 ms (bench medians) at any row length: 156.6 ms for a 94-token question,
  507.8 ms for five questions on a 137-token state with the shared prefix, 2,761.9 ms for four on a 1,477-token state.
  The first decision after a cold specialization took 3.58 s. In the 3-minute e2e passes, without rest, the call
  median was 30.4 ms (fixture) and 36.2 ms (the held-out set right after), at thermal state nominal: time with rest
  between items.
- **Kev-4B's h19p AOT asset (15,557,759,365 B) crashed at load**, once, inside `AIModel(contentsOf:)`, after the memory sample at 0.32 s:
  `EXC_BAD_ACCESS (SIGSEGV) KERN_INVALID_ADDRESS at 0x2c`, faulting thread `llvm-worker-0` in
  `mlir::ForwardIterator::makeIterable` ← `-[MPSGraphAICodeCompilerDelegate
  getInitializedAICodeBytecodeWithPayloadPrefix:delegateId:]` ← `Compiler_coreAI.compile` ← libODIECompiler
  `CompileForDelegates`. That sample read a 175.9 MB footprint with 3,364.1 MB available. The same frames appear in
  [`aot-and-specialization.md`](aot-and-specialization.md) for a fixed-shape graph loaded with
  `expectFrequentReshapes`; here the options were `.default` and the graph is dynamic. Not isolated, not retried.
- `devicectl device info files` prints rounded sizes; `--json-output` gives bytes. The phone's
  `volumeAvailableCapacityForImportantUsage` does not move inside the launch that writes or deletes gigabytes: read it at
  the start of a launch.

## The fixture's invented names

The 20 records written for the port use ten invented stems that had no DNS A record (`.com/.net/.io/.co.uk/.app`), no App
Store result and no Companies House company. An exact-word web search on 2026-10-04 found four of them in use: a Kindle
book series, a retail product, places in two games, a fantasy wiki and a fan-wiki character. The nine records using
those four stems are withheld from the published fixture (their ids, `request_sha256` and numbers stay); the oracle
was not re-run with new names. A screen of invented names needs the exact-word web search as well as the registries.

## Agreement with the fixture's gold labels (our subset)

The author's fp32 oracle against the fixture's labels: our subset of 434 questions, not the author's evaluation suites
or tables (`oracle_summary.json`, `oracle_summary_4b.json`, local).

| group | questions | Kev-0.8B | Kev-4B |
|---|---:|---:|---:|
| transfer-v4 development, first 60 (MMLU) | 60 | 25 | 45 |
| transfer-v4, 20 of each other source | 140 | 101 | 119 |
| transfer-v4 `score`, first 20 | 20 | 8 | 14 |
| SemIf authored144 | 144 | 104 | 129 |
| written for the port | 70 | 52 | 66 |

## Numbers of record

- Gates on the Mac GPU (AOT h16c, fp16) against the author's fp32 oracle. 0.8B, K128: fixture 420/420 + near-ties
  13/14, max |Δp| 0.0124, mean 0.00096; held out 125/125 + 5/5, 0.0058. 4B, K128: 422/422 + 10/12, 0.0153, 0.00077;
  held out 126/126 + 3/4, 0.0096. Swift = Python bit for bit on 434 + 130 rows (0.8B); the 4B Swift host's hidden rows
  equal the Python gate's on 434 + 130 rows.
- A decision in Swift on the M4 Max, round 15's lock window (two clean processes, median of 20), 0.8B: K128 29.9 ms for
  94 tokens, 357.0 ms for 1,518, 186.4 ms for five questions shared; U16 100.1, 1,600.1 and 327.7 ms. 4B, round 12's
  lock window (two clean processes, median of 12): K128 100.2 ms, 1,189.0 ms and 615.6 ms; U16 269.5, 4,341.9 and
  894.7 ms.
- iPhone 18 Pro, 0.8B: K128 37.8 ms for 94 tokens, U16 156.7 ms in the same session (round 13, stage C).
- Bundles: 0.8B K128 `main.mlirb` 1,505,385,733 B (`19d5a480…`), export 31.1 s, AOT 4.9 s; the earlier upload's U16
  1,506,481,909 B (`4131d028…`). 4B K128 8,412,500,461 B (`da529dd4…`), export 35.9 s, AOT 42.3 s; the earlier 4B U16
  8,414,007,636 B (`e38dc3ca…`).
