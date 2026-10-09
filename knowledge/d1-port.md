# d1-3B: an option-token decision model with pictures on Core AI

> 2026-10-08 – 10-09. LiquidAI/d1-3B (revision `da1fe36a`, LFM Open License v1.0) answers typed questions about a state
> (text, JSON, pictures) with a probability for every option, read from the logits of a few option tokens at the prompt's
> last position. The port is the overlay's LFM2.5-VL decoder returning hidden rows, the tied embedding rows of the
> option tokens on the host, and the SigLIP2 tower as a second graph. Card: [`models/d1-3b/README.md`](../models/d1-3b/README.md).
> Transcripts below are in `models/d1-3b/`; records outside the repo are marked "lane" and live under
> `$ZOO_WORK_ROOT/_d1_3b/`. Mac M4 Max, macOS 27.0 26A428, Xcode 27.0 RC; iPhone 18 Pro, iOS 27.2 24B5099f.

## What was reused and what is new

| part | source | change |
|---|---|---|
| decoder | the overlay's `Lfm2VlPipelinedForCausalLM` (LFM2.5-VL text decoder, attention projections fp32) | subclass `conversion/d1/lfm2_d1_decoder.py`: `lm_head` = identity, the final-norm hidden state at every position out, one static-S function, `image_embeds [2816, 2048]` as a static input; `lfm2_d1_static.py` for the static form |
| oracle | the provider's code in the checkpoint (`modeling_d1.py`, `runner.py`, `prompt.py`, `api.py`) | `oracle_d1.py` runs it unchanged (transformers ≥ 5.14, fp32, CPU) and asserts the readout on every question |
| readout | the provider's option-token logits | `export_option_rows.py`: the tied embedding rows of 2,134 candidate ids; `host.py` computes the logits from the slot's hidden row |
| vision | the overlay's `Lfm2VlVisionEncoder` | `lfm2_vl_tower.py`: SigLIP2 + projector with the crop's grid as inputs (patches, the resized position table, a key bias, the unshuffle index), every shape static |
| hosts | Kev's `host.py` / `decide.py` form, ClefFlash's runtime pattern | `host.py`, `vision_host.py`, `decide.py`; Swift `apps/D1` (library + CLI), `apps/D1Gate` (device gate) |
| gates | Kev / clef-flash's readout gate, bisect and Swift scorer | `readout_gate.py`, `readout_gate_vision.py`, `gate_tower.py`, `parity_decoder_torch.py`, `int8_bisect_torch.py`, `gate_swift.py`, `gate_swift_pixels.py`, `compute_unit_probe.py` |

## The readout: option rows, not a vocabulary head

The provider computes a log-softmax over the whole vocabulary at the last position and reads each option's best form.
That constant cancels in a softmax over the options, so the graph needs no head: the host takes `z = h · E[id]` over the
question's option groups in float64, a max per group and a softmax. On random full-vocabulary logits the two agree
within 1.75e-7 (lane `results/test_host.json` `readout_equivalence`).

The option table is every id the host's code rules can read: A..Z, a..z, 0..9, 00..99, 100..999, AA..ZZ (587 are one
token), the space-prefixed forms, and yes / Yes / YES / no / No / NO: 2,134 rows, 17.5 MB in fp32. A request that would
read an id outside it (a one-letter label in another script) is refused whole; the provider's code reads any id
(lane `results/r2a_option_rows_check_v2.json`).

A question without `instructions` makes the provider's `as_question` raise `KeyError`, which refuses the whole request.
The Kev-derived fixture has one (`own_email_03` / `next_step`); the host refuses it the same way, so the fixture has
393 answerable questions of 394 (lane `results/test_host.json` `provider_dry_run.refused`).

Several questions in one request go through the provider's Tree (the prefix and each suffix encoded apart). The gates
run one row per question; on the fixture's 45 multi-question rows the trunk plus the branch equals the row's ids, and
Tree against row moves p by at most 7.6e-6 in the oracle (lane `ROUND1.md`, `ROUND4.md`).

## The tokenizer: `<|startoftext|>` is text, and swift-transformers needs three steps

`encode("<|startoftext|>")` gives `[124894]`, and `add_special_tokens=True` adds no BOS: the prompt writes it as text
(lane `results/test_host.json`). The checkpoint's class name `TokenizersBackend` needs transformers 5; the oracle's venv
is separate from the overlay's.

swift-transformers 1.3.4 reads the BPE model but ignores `ignore_merges: true` (upstream PR #397) and splits the regex on
grapheme clusters (PR #398): its own ids match on 74 of 393 rows and 1,744 of 2,000 stress texts. `apps/D1` cuts the
added tokens out before anything else, splits with `NSRegularExpression` on UTF-16, looks each piece up in the vocabulary, and only then
runs a ByteLevel-only tokenizer's BPE: 393/393 and 2,000/2,000 (`gate-d1-3b-swift.json`, lane `results/r3a_swift_text.json`).

Two Foundation traps in the Swift host: `String(bytes:encoding:)` drops a U+FEFF at the start of a run as a byte order
mark (use `String(validating:as:)`), and Accelerate's `cblas_ddot` sums in another order when its operands start below a
256-byte boundary (copy them to page-aligned buffers, as NumPy's arrays of these sizes are) (lane `ROUND3a.md`).

## The picture path: torch's resize, not Pillow's

transformers 5.19's `Lfm2VlImageProcessor` is a torchvision backend class: its resize is torch's uint8 bicubic antialias
kernel (int16 fixed-point weights). Pillow's differs by up to 1 level on the fixture and 2 on random pictures, the float
path by up to 23. `vision_host.py` and `apps/D1`'s `ImagePixels.swift` copy torch's kernel; only the one-megapixel cap
uses Pillow's BICUBIC. The zoo note `lfm2.5-vl-port.md`'s "Pillow BICUBIC" is the transformers 4 processor (lane
`results/vision_rules.md`, `ROUND2b.md`).

The position table's resize matches torch only when its products are summed with fused multiply-adds (or in float64 and
rounded once): torch on aarch64 contracts `output += t * wts`. Separate roundings move it by up to 4.8e-7 (lane
`results/vision_host.json`).

ImageIO does not apply the EXIF orientation to the pixels; the host applies Pillow's `exif_transpose` table. ImageIO and
libjpeg decode a 4:2:0 JPEG up to 57 levels apart (4:4:4 and 4:2:2: up to 4); PNG decodes bit for bit (lane
`results/r3d_swift_pixels.json`).

A picture needs at most 2,810 image tokens (10 tiles and a thumbnail); the decoder's `image_embeds` holds 2,816 rows. On
a toy with the model's width, 256 / 2,816 / 4,096 rows cost the same per call (1.746 / 1.721 / 1.732 ms) (lane
`results/r3b_image_rows_cost.json`).

## The tower: fp16w32, and a toy that misleads about its size

The tower in fp16 math misses a row (lowest row cosine 0.9807 on img08's thumbnail); fp16 storage with fp32 math holds
every row at cosine 0.99999998 or more against transformers' fp32 rows (`gate-d1-3b-images.json` `tower_gate`, lane
`results/r5b_tower_fp16.json`).

On a toy, `optimize` folded the fp16w32 casts into fp32 constants and the asset came out at fp32's size. On the model
only the biases and LayerNorms fold: `resources.bin` 852,790,576 B against 851,982,536 B for fp16 and 1,703,963,308 B
for fp32. Measure a dtype's bytes on the model (lane `results/r5b_export_vision_*.json`).

## int8: the MLP linears pass, the rest do not

| decoder (S = 16) | fixture max \|Δp\| | held-out max \|Δp\| | bar |
|---|---:|---:|---|
| fp16 | 0.0046 | 0.0021 | PASS |
| int8lin: every MLP and conv-projection linear | 0.0227 | — | FAIL |
| int8lin per block of 16 | 0.0205 | — | FAIL |
| int8mix: int8lin with layers 2, 3, 6, 8, 9, 10, 12 fp16 | 0.0203 | — | FAIL |
| int8conv (a map: the conv projections int8 alone) | 0.0169 | — | not a candidate |
| **int8mlp: the MLP linears int8** | **0.0187** | **0.0137** | PASS |
| int8mlp per block of 16 | 0.0156 | 0.0081 | PASS |
| int4lin | 0.668 | — | FAIL (a cliff: 19 argmaxes move) |

An fp32 torch instrument with the exporter's own int8 weights reproduces int8lin's error (0.0235 against the graph's
0.0227), so it comes from the int8 weights, not from fp16 activations. A set chosen on six bisect rows (int8mix)
improved them and broke another row (tv4_031: 0.0052 under int8lin, 0.0203 under int8mix). Block 16 lowers the conv
projections' error but not the MLP's (lane `results/r5a_modes.json`, `results/r5c_modes.json`).

The shipped int8mlp at S = 64 is 0.0197 on the fixture and 0.0140 on the held-out set. Its worst question
(`semif_33d6e5da58f0fee2490d`, `answer`) lands between 0.0180 and 0.0223 across S and the two machines; the phone's
S = 32 puts it over the bar. When a new form goes to the phone, run that record before the rest (`gate-d1-3b-forms.json`).

The quantizer prints `Tensor size 1 along axis 1 is not divisible by block size 32. Skipping quantization.` (22 times for
int8lin) and `dynamic_shapes is only supported in graph mode and will be ignored.` (coreai-opt's eager mode); any other
warning, or a quantized set other than the mode's, stops the export (lane `ROUND5a.md`).

## What the compiled asset holds

- `--expect-frequent-reshapes` adds an fp16 copy of every linear: int8mlp's `.aimodel` is 3,704,646,757 B and its h16c
  asset 8,742,889,020 B; int4lin gets no copy (2.18 → 2.52 GB) (lane `results/r6a_forms.json`, `r5a_modes.json`).
- A compile that specializes once (the static form, no flag) folds the int8 weights into fp16 constants: 5,562,787,469 B
  for int8mlp at S = 64, the fp16 decoder's size. `stats.json` still counts Int8 elements; measure `resources.bin`
  (lane `results/r9b_export_*_static.json`).
- The phone's JIT of the int8mlp `.aimodel` keeps a 4,779,330,441 B cache entry, and the static `.aimodel`'s JIT
  3,704,923,670 B where the static AOT asset is 5,562,788,695 B (`gate-d1-3b-iphone.json`, lane
  `results/r10c_device_d1-3b.json`). Whether the phone's JIT keeps the int8 weights as int8 was not isolated.

## On the Mac, an int8 `.aimodel`'s JIT is not its AOT asset

The zoo ships `.aimodel`s, so on the Mac the shipped path is the Swift runtime's own specialization (GPU preferred,
`expectFrequentReshapes`), not the h16c AOT asset the Python gates load. For the fp16 decoder the two are the same: its
JIT gives the AOT asset's rows bit for bit at +0.2 % (`gate-d1-3b-timing-mac.json` `round7`). For the int8mlp decoder they
are not: in one lock window its JIT decided one question in 42.8 ms against 34.0 for the AOT asset (+25.7 %, the
3,470-token state +25.7 %, a picture +12.9 %), and its probabilities differ from the AOT asset's on every timed question
(max |Δp| 0.00046; against the oracle 0.00032, the same bits in both processes). Its cache entry is 4,667,336 KiB,
the phone's JIT size, against 8,742,889,020 B for the AOT asset with its fp16 copy of the linears: the JIT keeps the int8
weights another way. The cause of the slowdown was not isolated; on the phone the same JIT is faster than the efr AOT
asset (the iPhone section below). The fp16 S = 64 `.aimodel` behaves like the S = 16 one: in the next window its JIT
gave the AOT asset's p and hidden rows bit for bit on all 9 timed questions at +0.06 to +0.55 %, and so did the shipped,
stripped copy (below) at +0.07 to +0.74 %. So the Mac ships the fp16 decoder and the iPhone the int8mlp one. Gate and time
the form you ship, on the path it ships by (`gate-d1-3b-timing-mac.json` `round11_*`, lane `results/r11_jit_vs_aot_p.json`,
`results/r11s_jit_vs_aot_p_fp16_s.json`).

## The export writes local paths into `main.mlirb`

coreai-torch records every op's Python source location in the bytecode (a `sources` / `call_stack` string table at its
end), so an exported `main.mlirb` names the export machine's files: 14 absolute paths in the fp16 decoder, 19 in the
int8mlp decoder, 11 in the tower (this port's source tree, the overlay's `coreai_models`, the venv's torch modules).
`conversion/d1/strip_bundle.py` copies a bundle to a new name (`_s`) and re-saves its `.aimodel` after `strip_debug_info`;
the `main.mlirb` then holds no path and is 245,767 / 276,012 / 246,257 B smaller. Keep one program object:
`AIModelAsset.program` builds a new object on every access, and stripping one while saving another writes the source
bytes unchanged (`coreai-torch-041-ir-incident.md`).

A stripped `.aimodel` is a new asset: a new `main.hash` (a new runtime cache entry), and in its h16c AOT asset
`specialized_model_*.mpsgraph`, `main-h16c.mlirb` and `main.hash` differ while `resources.bin` and `manifest.plist` are
byte-identical. On the Mac the stripped bundles gave every gate row of the exports bit for bit (fixture and red arms 406
rows for each decoder, held out 120, pictures 24 rows and 40 tower crops for each decoder, the tower gate's 64 crops;
`gate-d1-3b-strip.json`). Grep the staged files for `/Users/` before an upload: only the three `main.mlirb` had it.

An export does not reproduce its `main.mlirb` byte for byte, even with unchanged scripts: it names every weight resource
`resource_<random 64-bit number>` (440 in the tower, 340 in the int8mlp decoder) and a few ops with a random suffix, in
the string table at the end; the weights' bytes and every other file match (lane `results/r11_reexport.json`). Check a
rebuild with the gates, not with sha256. `models/index.json` shows `source_model: null` for d1: the generator
(`conversion/_recipe.py`) reads a checkpoint only from a recipe's top-level `script` / `args`, and `export_decoder.py`
keeps it in a `MODEL` dict.

## An AOT cache entry is named by the kind of linear and S, not the bit width

The AOT asset's `main-h16c.mlirb` holds the function's type and the source paths, not the weights: int8lin at block 32
and 16 and int4lin compile to one `main.hash`, int8mlp at block 32 and 16 to another. The Python runtime names its cache
entry `python/<main.hash>`, so loading one of them while the other's entry is there runs the other's graph; a red-arm
control measured it. Move the entry aside before gating or timing another mode, and check its `manifest.plist` sha256
afterwards. Each S compiles to its own name (lane `ROUND5a.md`, `ROUND5c.md`, `ROUND6a.md`).

The cache directory is named after the executable file: `d1/` for the Swift CLI, `python/` for `python`, `python3-11/`
for `python3.11`. Starting the interpreter under another file name gives a second entry for the same asset (lane
`ROUND7.md`).

Loading an AOT asset makes an entry that shares blocks with the asset (removing it frees nothing); a JIT entry is new
data (the fp16 S = 16 decoder: +9.87 GiB on the Mac). Loading a shipped asset with other specialization options adds a
second copy inside its entry (lane `ROUND6a.md`, `ROUND6b.md`, `ROUND7.md`).

## Dynamic positions without `--expect-frequent-reshapes`

The decoder compiled without the flag runs, but the runtime specializes it again at every new position length: 5.9–12.5 s
for each new length on the Mac, in every process, with 4.7 GB of MPSGraph scratch per length left in `$TMPDIR` after the
process exits. On the phone each new length took 11.2–59.8 s, a call right after such a specialization once returned a
wrong row (max |Δp| 0.376, the same row correct on the second pass), and a later call ended in `SIGTRAP` inside
CoreAIRuntime. The cause was not isolated (lane `results/r6a_probe_noefr.json`, `results/r9a_device_d1-3b.json`).

## The static form: specialized once, slower per call

`lfm2_d1_static.py` sends each call its own S positions and keeps the caches at 4,096 slots with the mask built in the
graph from the positions; the new keys and values go in with `mutable_slice_update` at an index taken from the data.
That write did not trap on the Mac GPU on this export path, unlike the minimal export in
[`coreai-beta-mpsgraph-kvwrite-bug.md`](coreai-beta-mpsgraph-kvwrite-bug.md). In fp32 torch the static form equals the
dynamic one bit for bit (lane `results/r9b_fp32_static_vs_dynamic.json`).

Every call attends all 4,096 slots. The plain attention chain is wrong from 4,032 keys
([`clef-flash-port.md`](clef-flash-port.md)), so the static form uses the SDPA composite: a 4,050-token row keeps cosine
0.99996 at every position against the dynamic form, 0.9999965 from position 4,032 on (lane `results/r9b_long_tail.json`).

Compiled without the flag, the static form specializes once and never again: a process's opening call 0.10–0.14 s against
0.9–8.1 s for the dynamic efr form. Each call costs more: a 64-token call 39.9 ms against 34.1 on the Mac, 10–19 % per
decision; on the phone 118.3 (JIT) and 150.6 ms (AOT) against 87.8 in the same thermal state. Whether the cost is the
4,096 keys every call reads was not isolated (`gate-d1-3b-timing-mac.json` `round10b`, `gate-d1-3b-forms.json`).

## The chunk width

The cost of a call grows with S, but the calls shrink: on the Mac a call takes 20.5 / 25.0 / 34.1 / 56.2 ms at
S = 16 / 32 / 64 / 128 (fp16 at 16–64, int8mlp at 128). One question of 39 tokens is one call at S = 64; S = 128 takes it
in 55.5 ms against 34.6, and decides a 3,470-token state 16.2 % faster. On the phone S = 64 decides 2.6–3.3 times faster
than S = 16 (one call 49.4 against 41.1 ms, a quarter of the calls) (`gate-d1-3b-timing-mac.json`, `gate-d1-3b-iphone.json`).

The fp16 graph rounds in another order at each S: p moves by 0.0037–0.0076 against S = 16 with every argmax kept, while
fp32 torch gives the same hidden rows at every width (lane `results/r6a_forms.json`, `gate-d1-3b-torch-parity.json`).

## The Neural Engine: three walls

The dynamic decoder compiled with `--preferred-compute neural-engine` gets no Neural Engine region and runs bit-equal
to the GPU. Specialized at a fixed position length, the runtime builds one region whose IR (4,702,647,541 B) is the
size of the 134 MLP and conv-projection linears in fp16 (4,701,814,784 B) plus 0.8 MB, and
`MLIR MPS to ANEC conversion failed (default/nonbonded phase)` every time; the call falls back to the GPU. The fp16w32
tower computes in fp32 (`Incompatible element type for ANE: expected fp16`). fp16 weights do reach a region: the
fp16-math tower ran on the Neural Engine, at lowest row cosine 0.9755 (lane `results/r6a_probe_ane.json`).

`coreai-build`'s `--preferred-compute` takes `gpu`, `neural-engine` or `none`; `ComputeUnitKind.neural_engine` is a
method to call. The runtime's `Profiler` does not help on 1.0.0b2: with callbacks it stops the process (`SIGTRAP`),
without them it records nothing (lane `ROUND6a.md`).

## iPhone 18 Pro: the memory limit decides

At the default memory limit (3,530 MB available at launch) neither decoder form loads. The `.aimodel`'s on-device
specialization dies with `std::bad_alloc` while it folds a weight transpose
(`CanonicalizeMatMulNNToNT → foldTransposeOp → BumpMmapResourceAllocator::allocateResource`); the efr AOT asset
(8.74 GB) dies with `SIGSEGV` inside the delegate compile (`MPSGraphAICodeCompilerDelegate … CompileForDelegates`).
With `com.apple.developer.kernel.increased-memory-limit` (6,432 MB) both load (`gate-d1-3b-iphone.json`
`without_entitlement`, lane `results/r9a_device_d1-3b.json`).

The team's wildcard profile cannot carry the key: the entitled build fails with `Provisioning profile "iOS Team
Provisioning Profile: *" doesn't include the Increased Memory Limit capability.` `xcodebuild -allowProvisioningUpdates`
let Xcode's account add the capability to the explicit App ID and fetch its profile; no Xcode GUI step was needed
(`gate-d1-3b-iphone.json` `signing`).

The phone's JIT of the `.aimodel` is faster per call than the Mac-compiled efr AOT asset: 40.5 against 55.9 ms at
S = 16, and the two give different rows (max |Δp| 0.0013). The efr asset's opening call after its load took 9.7 s (lane
`results/r9a_device_d1-3b.json`).

A shared phone handed back at thermal state serious costs the next lane a cooldown: let it return to nominal before
releasing the hold. The hold file is one JSON line (`{"pid", "script", "device", "session", "started"}`):
`queue_cli.py wait` treats a text hold as legacy and overwrites it (lane `ROUND9a.md`).

## The gate's instruments

- **The oracle runs on one thread.** On the CPU, one and four threads give the same bits, twelve move a long row's
  hidden rows; one thread also took the least loop time on three records, 21.0 s against 24.1 and 37.8
  (lane `results/r4_oracle_threads.json`).
- **A red arm must move the oracle before it tests the graph.** The word arm on tv4_000 moved the oracle by 0.0069 only, so it was
  replaced by the same change on tv4_001 (0.976) before any graph ran (`gate-d1-3b-red.json` `precheck`).
- **A held-out row can equal a fixture row.** tv4h_28 renders to tv4_002's row (a permuted MMLU row whose order came out
  unchanged); the verdict was read with and without it. Five pairs of fixture records are identical, so the gate's mean
  counts them twice (lane `fixtures/heldout.json`, `ROUND5a.md`).
- **The CPU-only runtime does not run dynamic dimensions** (`inferenceFailed(-1)` on the opening call); every Mac gate ran
  the AOT asset on the GPU (lane `results/toy_graph.json`).
- **Timing on a shared Mac.** A process counts when no other GPU job ran during it and the one-minute load average
  stayed at or below 12. The driver writes `<tag> timing pid <n> since <time>` into `_GPU_LOCK`: without the word
  `timing` the other lanes' guards do not see the window. `quiet_wait.py` looks only at the start, so a long GPU job
  needs a guard that stops it (`SIGSTOP`) while another lane's window is open (lane `ROUND5c.md`, `ROUND7.md`).

## The fixture's text

The fixture's transfer-v4 rows (MMLU, emotion, QNLI, PAWS, SciQ and the Kev suite's generated sources) are published as
references with their hashes; SciQ is CC BY-NC 3.0 and emotion / GLUE / PAWS say "other". The 20 tweet_offensive
records were left out of the fixture (licence unknown, real public figures' names, profanity). The LiteRT Kev port's
own records name invented people and companies that were never screened, so only their numbers are published
(`fixtures-d1-3b.json` `sources`, lane `results/r11_fixture_publication.json`).

## Numbers of record

| what | value | where |
|---|---|---|
| fixture, fp16 S = 64 (the Mac's decoder), Mac | 392/392 + 1/1, max \|Δp\| 0.0039, mean 0.00018 | `gate-d1-3b-readout.json` |
| pictures, fp16 S = 64 + tower fp16w32, Mac | 24/24, max \|Δp\| 0.00049 | `gate-d1-3b-images.json` |
| fixture, int8mlp S = 64 (the iPhone's decoder), Mac | 392/392 + 1/1, max \|Δp\| 0.0197, mean 0.00126 | `gate-d1-3b-readout.json` |
| held out, int8mlp S = 64, Mac | 118/118 + 2/2, max \|Δp\| 0.0140, mean 0.00101 | `gate-d1-3b-heldout.json` |
| pictures, int8mlp S = 64 + tower fp16w32, Mac | 24/24, max \|Δp\| 0.0024 | `gate-d1-3b-images.json` |
| the stripped bundles against the exports, Mac | every gate row bit for bit | `gate-d1-3b-strip.json` |
| iPhone 18 Pro, the phone's JIT (int8mlp as exported) | 416/416 + 1/1, max \|Δp\| 0.0186 | `gate-d1-3b-iphone.json` |
| one question, Mac (fp16 JIT, stripped) / iPhone | 34.3 / 48.0 ms | `gate-d1-3b-timing-mac.json`, `gate-d1-3b-iphone.json` |
| 3,470-token state, Mac / iPhone | 1,888.0 / 2,794.4 ms | the same |
