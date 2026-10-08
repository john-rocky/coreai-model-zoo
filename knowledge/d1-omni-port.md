# d1-omni-600M: an encoder decision model with an image and an audio tower, on Core AI

> 2026-10-08. [LiquidAI/d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M) (revision `414f8d64…`, LFM Open License v1.0) answers typed questions (`noul`, `choice`, `score`) about a state (text or JSON, with images or one voice clip) with a probability for every option, in one forward pass and without generating text. The port re-authors its three networks in plain PyTorch, exports them in fp16, removes their debug locations, compiles them ahead of time, and gates every stage against the publisher's own model in fp32 on the CPU. Card: [`models/d1-omni-600m/`](../models/d1-omni-600m/). Scripts: [`conversion/d1_omni/`](../conversion/d1_omni/README.md). Swift host: [`apps/D1Omni/`](../apps/D1Omni/README.md). Transcripts below are `models/d1-omni-600m/gate-d1-omni-600m-*.json`; records outside the repo are marked "lane" and live under `$ZOO_WORK_ROOT/_d1_omni/`. One fact per line, with the round it came from. Mac = Apple M4 Max, macOS 27.0 (26A428), Xcode 27.0 RC, coreai-build 3600.83.1, coreai-torch 0.4.1, coreai-core 1.0.0b2.

## What the checkpoint is

- One `model.safetensors` of 2,348,774,500 B, sha256 `0713bb05…`, 1,084 tensors: F32 except 17 int64 BatchNorm counters (round 2; lane `results/weights_check.json`).
- 587.2 M parameters: a 354.5 M bidirectional trunk, a 26.2 M decision head, a 94.2 M vision tower and a 112.2 M audio tower; every modality runs the same trunk weights (round 1; lane `results/source_check.json`).
- The trunk is LFM2.5-Encoder-350M's: 16 layers (10 short convolutions, 6 GQA attentions at layers 2, 5, 8, 10, 12, 14), hidden 1,024, MLP 4,608, 16 query / 8 key-value heads of 64, RoPE θ 1e6, vocabulary 65,536 (round 1; `conversion/d1_omni/d1_omni_model.py`).
- Its short convolution is a centred 3-tap filter, so a text position reads its right neighbour; `keep_right` zeroes that read at the last media position, and media queries do not attend to text keys (round 1; `d1_omni_model.py`).
- The head adds a type embedding (choice / score / noul), runs two norm-first `nn.TransformerEncoderLayer`s (16 heads, ReLU, 4,096) with a key-padding mask, and a scorer (LayerNorm → Linear → GELU → Linear(1,024, 1)) at every position (round 1; `d1_omni_model.py`).
- A row is `[1, 17] + state + [18] + instructions + Σ([19, 16] + option + [20]) + [21]`; the options are read at the `<|mask|>` (16) positions (round 1; `conversion/d1_omni/host.py`).
- Text answers divide the marker logits by a temperature chosen by type and option count (`choice:3-5` 1.3999, `noul:2` 1.6663, …, type default 1.0); image and audio answers are the raw softmax (round 1; `host.py` CONFIG).
- A `noul` question is scored in the order [false, true] and reported as [yes, no] (round 1; `host.py` `probabilities_from_logits`).
- An audio request writes choice options as `option_000: <description>` and a yes / no question as `false: no` / `true: yes`, as the audio questions were trained (round 1; `host.py`).
- The vision tower is LFM2.5-VL-450M's SigLIP2 NaFlex (12 layers, hidden 768); a crop of up to 1,024 16-px patches becomes up to 256 prefix rows through a 2 × 2 unshuffle and the projector (round 5; `conversion/d1_omni/d1_omni_vision.py`).
- An image whose rounded area exceeds 2 × 512² is cut into 512-px tiles of the closest grid (2 to 10 tiles) plus a thumbnail; a 384 × 384 image is one crop of 144 rows, a 768 × 1024 image 7 crops and 1,770 rows (round 1, 5; lane `results/media_lengths.json`).
- Only a square image yields 256 rows at 512 px; 512 × 384 yields 192 (round 5; lane `images/manifest.json`).
- The audio tower is 8× subsampling, 17 FastConformer layers (d 512, 8 heads, conv kernel 9), an adapter and a residual block; 5 / 10 / 20 / 30 s of 16 kHz audio become 63 / 125 / 250 / 375 prefix rows (round 1, 6; `conversion/d1_omni/d1_omni_audio.py`).
- With an image the state and question text is cut to 896 tokens, with audio to 15,360; all positions together at most 16,384 (round 1; `host.py`).

## The oracle and the bar

- The oracle is the publisher's model (`AutoModel.from_pretrained(..., trust_remote_code=True)`, `probabilities()`), fp32 on the CPU, transformers 5.19.0 in its own venv: the checkpoint's code needs transformers ≥ 5.15, which the zoo venv does not have (round 2; lane `ref/records_ref.json`).
- transformers 5.19 loads the checkpoint's `vision.tower.vision_model.*` (197 tensors) as `vision.tower.*`; the loaded state equals the file tensor for tensor (round 2; `conversion/d1_omni/reference_d1_omni.py`).
- huggingface_hub 1.33 resolves a 40-hex revision offline only after one online `snapshot_download` has written the tree listing (round 2; lane `results/weights_check.json`).
- The fp32 module bar is 2e-5 in p, not 1e-5: the fp32 oracle is itself 9.3e-6 from the publisher's code run in float64 and the re-authored module 7.9e-6 (round 2; `gate-d1-omni-600m-eager.json` `bar_basis`).
- With every fp32 step in float64 and the RoPE table built for the same length, the module and the publisher's code agree to 1.2e-13 in the marker logits: the same function (round 2; `gate-d1-omni-600m-eager.json`).
- The module in fp32 meets that bar on all 470 round-2 rows: argmax 470/470, max |Δp| 1.45e-5, marker logits 8.6e-5 (round 2; `gate-d1-omni-600m-eager.json`).
- The publisher's RoPE table changes in its last bit with the length it is built for, at 12 threads (n = 191: 18 elements; 1 and 4 threads: none) (round 2; lane `results/rope_length.json`).
- The publisher pads a request's questions to its longest row; one question alone and in its request differ by up to 4.9e-6 in p (round 2; lane `ref/records_ref.json` `checks.batch_invariance`).
- A wrong mask on a media row can leave p nearly unchanged when one option is saturated (keep_right all ones: |Δp| 0.0084) while the marker logits move by 0.04 to 0.74: gate the media rows on logits too (round 2; `gate-d1-omni-600m-eager.json` `red_arms`).
- The largest fp32 activation is 2,069, the residual of one text position in layers 10–14 (a massive activation written by layer 9's MLP): 31.7× below fp16's 65,504; nothing is non-finite (round 2; `gate-d1-omni-600m-eager.json` `activation_range`).
- The ship bar for every reduced-precision form is FACTS §7: argmax on every row whose oracle top-2 margin is above 0.02 (near ties reported apart), max |Δp| ≤ 0.02, mean of the rows' max |Δp| ≤ 0.002 (round 3; `gate-d1-omni-600m-runtime-decide.json` `bar`).
- Every runtime gate carries a wrong-pairing control (each row judged against another row's oracle) that must fail, and does on every run (round 3–12; `gate-d1-omni-600m-runtime-decide.json`).

## The graph's form

- One function `main` per static length L takes `input_ids`, `prefix_embeds`, `pad_mask`, `prefix_mask`, `keep_right` and `qtype_onehot` and returns a score at every position; the masks are float inputs, so the trace specializes nothing (round 3; `conversion/d1_omni/export_decide.py`).
- The GQA repeat form lowers as written; the broadcast form is not needed (round 3; lane `ROUND3.md`).
- `AIModelAssetMetadata(author=…, license=…, model_description=…)` drops its keyword arguments silently; set the attributes and assert them (round 3; `export_decide.py`).
- L = 4096 computes attention in two blocks of 2,048 keys (`KEY_BLOCK`): the zoo's GPU softmax over more than 4,032 keys is not trusted; the blocked form equals the plain one to 6.7e-14 in float64 (round 4; `gate-d1-omni-600m-eager.json` `decision_graph_L4096_blocks`).
- In fp32 the blocked and plain forms differ by 3.3e-5, which is this graph's own fp32 floor (the plain form is 3.0e-5 from its float64 run): bar the block form in float64 and against the oracle, not against the plain form in fp32 (round 4; lane `results/block_softmax_diag.json`).
- The fp16 graph has 1,315 operations at L = 64 … 2,048 and 1,513 at 4,096 (no softmax op: reduce_max / exp / reduce_sum / divide per block) (round 4, 12; `gate-d1-omni-600m-strip.json` `bundles`).
- The vision unshuffle (`F.embedding` on an index input) lowers to one `coreai.gather_nd` (round 5; lane `ROUND5.md`).
- The audio tower's relative shift (pad → reshape → slice), its depthwise Conv1d and the stride-2 depthwise Conv2d lower as written; the 17 BatchNorms are folded into a scale and a shift (round 6; `d1_omni_audio.py`).
- An audio clip's bucket is chosen by mel columns (1 + n // 160 ≤ F), which differs from "the clip's seconds" by up to 159 samples (round 6; `host.py` `audio_bucket_for`).
- An h16c AOT asset does not load with `cpu_only()` (`failedToSpecialize`), and `--preferred-compute none` compiles the same bytes as `gpu`: the Mac's runtime parity reference is the fp32 graph on the GPU (round 3; lane `results/aot_fp32_L256_none.json`).

## Precision and compute unit

- fp16 (weights and compute, norms / softmax / scorer in fp32) passes the ship bar at every length on the Mac GPU: L256 max |Δp| 0.0067, mean 0.00098, argmax 436/436 (round 3–12; `gate-d1-omni-600m-runtime-decide.json`).
- On the GPU the fp16 graph is closer to the oracle than torch's fp16 on the CPU (L256: 0.0067 against 0.012); why is not isolated (round 3; `gate-d1-omni-600m-forms.json`).
- wfp16 (fp16 weights, fp32 compute) keeps its fp16 bytes through AOT and passes (L256 max |Δp| 0.0020), but it is the slowest form of every graph on the Mac GPU (decision L256 23.7 ms against fp16 16.8 and fp32 17.6) (round 3, 7; `gate-d1-omni-600m-forms.json`).
- int8 per block of 32 on every decision linear fails the bar (L256: argmax 429/436, max |Δp| 0.097, mean 0.0102) on the GPU as in torch, so the loss is the quantization, not the runtime (round 4; `gate-d1-omni-600m-forms.json`).
- The int8 graph's compiled `resources.bin` is 761,554,008 B, the size of the fp16 graph's (with another sha256): only the download is smaller (469 MB against 762 MB), and it runs at fp16's speed (16.70 against 16.81 ms at L256) (round 4, 7; `gate-d1-omni-600m-forms.json`, lane `results/aot_int8_L256_h16c_r7.json`).
- coreai-opt's eager quantizer intercepts `F.linear` inside a module's forward; a `module_name_configs` entry for a parent needs `"weight"` in its spec or the child `out_proj` stays unquantized (98 of 100 weights) (round 4; `export_decide.py` `set_quant_mode`).
- The Neural Engine compile of the fp16 decision graph has 52 regions and misses the bar (argmax 409/436, max |Δp| 0.95); the wfp16 compile has one region and returns values that change between calls (drift 13.8) (round 3; `gate-d1-omni-600m-forms.json`).
- Loading the fp16 decision graph's Neural Engine compile logs `Incompatible element type for ANE: expected fp16, f8E4M3, si8, ui8, si16, or ui16 … ANE regionCall op not found` at the short convolution's weight cast, and runs on (round 7; lane `ROUND7.md`).
- The vision graph's Neural Engine compile has 40 regions and misses the bar end to end (mean 0.0031); the audio graph's (10 s) has 64 and passes (24 rows), but each media call is slower than on the GPU (79–84 / 28 ms against 22 / 8.6 ms) (round 7; `gate-d1-omni-600m-forms.json` `neural_engine_compiles`).
- A call of the wfp16 Neural Engine compile aborted the process twice with `MPSCommandBufferImageCache.mm:1220: failed assertion 'Internal error: Released a texture not in current cache frame.'`, once with no other GPU job running; the condition is not isolated (round 3, 7; lane `ROUND7.md`).
- From Python, `SpecializationOptions.from_preferred_compute_unit_kind(gpu)` allows CPU, GPU and Neural Engine and prefers the GPU: Python sets a preference, not a placement (round 7; lane `ROUND7.md`).

## The image and audio host

- The vision tower's 16 × 16 position table is resized per crop by torch's float32 antialias kernel (weights in fp32, a fused multiply-add per tap, width before height); a NumPy copy of that order is bit-equal on every crop, a float64 filter is 2e-5 off (round 5; `gate-d1-omni-600m-eager.json` `vision_tower`).
- torch clamps the antialias tap count to ceil(support)·2 + 1; the NumPy copy needs the same clamp (round 9; `host.py` `_aa_weights_f32`).
- On a Mac CPU without AVX2, torchvision 0.24 resizes uint8 images through float32 (`interpolate`, antialias) and rounds half to even; the NumPy copy equals it on 62/62 crops, and rounding half up changes 313,774 values (round 9; `gate-d1-omni-600m-eager.json` `host_numpy`).
- The tower's own fp32 floor is large (the publisher's fp32 prefix is 0.117 from its float64 run), so the vision bar is cos ≥ 0.999999 in fp32 plus float64 equality (2.7e-10), not max |Δ| ≤ 1e-3 (round 5; `gate-d1-omni-600m-eager.json`).
- ImageIO decodes a 4:4:4 baseline JPEG up to 3 levels away from PIL (libjpeg-turbo) on 19 % of the bytes; a decoder that copies libjpeg-turbo's integer IDCT (`jpeg_idct_islow`) and colour tables is bit-equal (round 9; `apps/D1Omni/Sources/D1Omni/JPEGBaseline.swift`).
- That decoder reads only baseline JPEGs whose components share one sampling factor; 4:2:0, 4:2:2 and progressive JPEGs fall back to ImageIO, whose effect on answers was not measured (round 9; `apps/D1Omni/README.md`).
- ImageIO's PNG pixels equal PIL's on every fixture PNG, including ones with an ICC profile or EXIF (round 9; lane `results/swift_media_image.json`).
- The mel front end runs on the host: the publisher's torch code is copied bit for bit, and a NumPy float64 form (the Swift host's specification) equals it run in float64 to 4.5e-13 (round 6; `gate-d1-omni-600m-eager.json` `audio_tower.mel`).
- Against the publisher's fp32 mel the float64 form is within 1e-4 on 14 of 22 clips; the rest is the publisher's own fp32 rounding (up to 3.1e-4 on speech, 0.16 on digital silence, where the per-row std is 0) (round 6; lane `results/audio_mel.json`).
- NumPy's axis-1 mean adds 0 and then pairs all n values; a Swift copy that starts from the first value matches 3,125 of 5,120 rows; float64 `log` / `cos` are libm and matmul is Accelerate's dgemm, so the same order in Swift gives the same mel except one value in 3,522,048 (1 ulp, the FFT) (round 9; lane `results/swift_media_audio.json`).
- The publisher's relative-position table changes in its last bit with the length it is built for, so float64 comparisons at 20 and 30 s read 1e-9 instead of 1e-14 (round 6; lane `ROUND6.md`).

## Strip

- The exported graphs carry the exporting machine's absolute path in their MLIR debug locations (`main.mlirb`, 2 strings per graph), which `grep -rIl` does not see because the file is binary (round 8, 10; `gate-d1-omni-600m-strip.json`).
- `coreai.authoring.AIModelAsset.load` → `strip_debug_info(asset.program)` → `save_asset` removes them; the op counts and `coreai-build inspect` summaries are unchanged, bytes drop by 73 to 267 KB, `main.hash` changes (round 10; `gate-d1-omni-600m-strip.json` `bundles`).
- The compiled `resources.bin` of a stripped bundle has the same sha256 as the unstripped one's; `specialized_model_0.mpsgraph` shrinks (L256: 368,766 → 210,881 B), so the location strings were in it too (round 10; lane `ROUND10.md`).
- Every gate on the stripped bundles reproduces the unstripped numbers bit for bit (Python runtime: 12 comparisons SAME; Swift: 562 rows) (round 10, 12; `gate-d1-omni-600m-runtime-decide.json` `stripped_vs_unstripped`).
- `save_asset` removes an existing output path before it writes: write to a new folder (round 10; `conversion/d1_omni/strip_ship.py`).
- Writing the scan's needle (the users-folder prefix) into a JSON key or a note makes the text scan find it; name it in words (round 10; `conversion/d1_omni/stage_hf.py`).

## The Swift host

- The Swift host's token rows equal the Python host's on every fixture row (v1 479, v2 551, v3 623), with a control that flags the one changed row (round 8; `gate-d1-omni-600m-swift.json` `token_rows`).
- On the Mac the `.aimodel` specialized by the Swift runtime and the h16c AOT asset give the same marker logits bit for bit (470 text rows, 92 media rows, every timed call) and the same time; the Mac ships the `.aimodel` (round 8, 9; `gate-d1-omni-600m-swift.json`).
- The Swift host's marker logits equal the Python runtime's on the same AOT bit for bit: 562 rows at L256–4096 and 342 text rows at L64 / L128 (round 10, 12; `gate-d1-omni-600m-swift.json`).
- swift-transformers 1.3.3 tokenizes a 3.4k-token state in about 0.40 s (Python's tokenizers: 2.0 ms), longer than the decision itself (0.30 s); a one-question state takes 3.4 ms (round 8; `gate-d1-omni-600m-timing-mac.json`).
- `D1Omni.folders(macos:)` takes every `fp16-L<L>` / `decide-fp16-L<L>` folder of a directory as a bucket: keep a trial bucket out of the shipped folder or it reroutes the short rows (round 12; `apps/D1Omni/Sources/D1Omni/D1Omni.swift`).
- The 9 audio rows of 128 positions or fewer were run at L64 / L128 by the Python host only; the Swift host's small-bucket parity covers the 342 text rows (round 12; `gate-d1-omni-600m-swift.json` `small_buckets`).
- A Swift cold load of the L256 graph took 1.93 s as `.aimodel` and 2.00 s as AOT; a later launch 1.04–1.64 s; the runtime caches the `.aimodel` under its `main.mlirb` sha256 (round 8; `gate-d1-omni-600m-timing-mac.json` `swift_host.loads_round8`).

## The small buckets (L64 / L128)

- The decision graph also ships at L = 64 and 128, so a short row pays for 64 or 128 positions instead of 256: L64 gates 64 rows (argmax 64/64, max |Δp| 0.0058) and L128 298 rows, 278 of its own and 20 shorter ones padded in (argmax 290/290 and the 8 near ties, max |Δp| 0.0117) (round 12; `gate-d1-omni-600m-runtime-decide.json`).
- `host.BUCKETS` stays (256 … 4,096) as `bucket_for`'s default because a dozen scripts compare against buckets recorded with it; the small buckets go through `host.ALL_BUCKETS` explicitly (round 12; `conversion/d1_omni/host.py`).
- L64 has no padded rows (no smaller bucket), and no media row of the oracle has 128 positions or fewer: the 9 audio rows that land at L64 / L128 were gated end to end with `audio_check.py --buckets` (round 12; `gate-d1-omni-600m-runtime-audio.json`).
- The small graphs have L256's 1,315 operations, and their compiled `resources.bin` has one sha256 across the unstripped h16c, the stripped h16c and the h19p compile (761,504,856 B at L64, 761,521,240 B at L128) (round 12; `gate-d1-omni-600m-strip.json`, lane `ROUND12.md`).
- The Swift host takes a bucket from the folder name and `metadata.json`'s `seq_len`, so L64 / L128 needed no Swift change (round 12; `apps/D1Omni/Sources/D1Omni/D1Omni.swift`).
- L64 costs 45 % of L256's decision on the Mac (7.35 against 16.20 ms) and 54 % on the iPhone 18 Pro (10.79 against 20.12 ms, `.aimodel`) for the same 47-position question (round 11, 12; `gate-d1-omni-600m-timing-mac.json`, `gate-d1-omni-600m-iphone.json`).

## Speed on the Mac

- One decision on the Mac GPU (Python runtime, AOT h16c, stripped fp16, window `d1d-r12`): 7.35 ms for a 47-position question at L64, 21.97 ms for three questions (three calls) at L64 (round 12; `gate-d1-omni-600m-timing-mac.json`).
- The 47-position question costs 10.27 ms at L128 and 16.20 ms at L256 in the same window: the call grows with L, so the smallest bucket that holds the row is the fast form (round 12; `gate-d1-omni-600m-timing-mac.json`).
- Above L256 the call grows roughly with L (fp16: 28.7 / 55.2 / 112.4 / 299.9 ms at 512 / 1,024 / 2,048 / 4,096, two windows); the 3.4k-token state takes 299.2 ms at L4096 (round 7; `gate-d1-omni-600m-timing-mac.json`).
- A 384-px image takes 38.3 ms (vision 21.6–21.8 ms + decision 16.6 ms) and a 9.6 s clip 24.94 ms (audio 8.7 ms + decision L256), a 2.8 s clip 17.72 ms at L128 (round 7, 12; `gate-d1-omni-600m-timing-mac.json`).
- The Swift host's decision time equals the Python runtime's within 3 % (W1 16.14 ms JIT / 16.16 AOT against 16.67 in the same windows) (round 8; `gate-d1-omni-600m-timing-mac.json`).
- On a Mac shared with other measurement windows, a job that only checks the lock at its start runs into the next window; a wrapper that stops the job while another lane's window is open (SIGSTOP / SIGCONT) kept the timings clean, and a stopped process (`ps` state T) must not be waited on (round 7, 8; lane `ROUND7.md`, `ROUND8.md`).

## iPhone

- The gate app [`apps/D1OmniGate`](../apps/D1OmniGate/README.md) runs the Swift host `apps/D1Omni` on an iPhone 18 Pro (iPhone19,2, iOS 27.2 24B5099f, h19p) on USB power: 78 fixture rows (60 text, 9 image, 9 audio) and 18 of them again at L64, with each `.aimodel` specialized on the phone (JIT) and with the h19p compiles (AOT) (round 11; `gate-d1-omni-600m-iphone.json`).
- Every row passes the ship bar with the `.aimodel` (max |Δp| 0.0059, mean 0.00090) and with the h19p compiles on the 74 rows they ran, and the two give the same bits on every row (92/92 with the L64 rows) (round 11; `gate-d1-omni-600m-iphone.json` `parity`).
- The `.aimodel` and the h19p compile time the same on the phone, within 1.4 %, and neither is faster on every workload: one question at L64 10.79 / 10.70 ms, three questions 32.52 / 32.07, an image 57.67 / 57.72, 9.6 s of audio 27.02 / 27.30 (round 11; `gate-d1-omni-600m-iphone.json` `bench`).
- The phone's GPU does not give the Mac's bits: 0 of 78 rows are bit-equal (max |Δp| 0.0040, max |Δlogit| 0.047), while every host array before a graph (RGB, crops, vision inputs, samples, mel, masks) is, so the difference is the graphs' arithmetic on the phone (round 11; `gate-d1-omni-600m-iphone.json` `host_arrays_equal_mac`).
- The h19p compile of the L4096 graph loads (1.16 s) but its first call is killed: the app's footprint goes from 1,535 to 3,306 MB with 234 MB left, JetsamEvent reason `per-process-limit`, while the `.aimodel` at L4096 peaks at 2,778 MB and passes; so `ios-h19p/` ships without L4096 and a phone runs `ios/decide-fp16-L4096` (round 11, 14; `gate-d1-omni-600m-iphone.json` `l4096_h19p`).
- The app had the phone's default memory limit (3,529 MB available at launch): a wildcard provisioning profile cannot carry the increased-memory-limit entitlement, so the L4096 compile's time was not measured (round 11; lane `ROUND11.md`).
- A phone app with a heavy graph should load it last in its own stage and relaunch for the stages left: a jetsam kills the process and every stage after it (round 11; `apps/D1OmniGate/_gate.sh`).
- The h19p compiles made with `--preferred-compute neural-engine` (audio 10 s, decision L256) fail to load on the phone with `CoreAIDelegates.AIModelError.failedToSpecialize`; the same `.aimodel` loaded with the Neural Engine preferred loads (audio 4.3 s, decision 11.6 s) (round 11; `gate-d1-omni-600m-iphone.json` `neural_engine`).
- With the Neural Engine preferred the audio rows pass (9 rows, max |Δp| 0.0064) but the 9.6 s request takes 59.98 ms against the GPU's 27.19 ms in the same series, and the decision graph fails (argmax 33/55); where it ran is not confirmed (round 11; `gate-d1-omni-600m-iphone.json` `neural_engine`).
- Counting `_ANE_region_` files in the app's Core AI cache does not show Neural Engine placement on the phone: none appeared after loads that added 225 and 762 MB (round 11; `gate-d1-omni-600m-iphone.json` `neural_engine`).
- Alternating the Neural Engine and the GPU in one series moves the GPU's time too (one question at L256: 20.12 ms in the bench, 23.85 ms alternating): compare within one series (round 11; `gate-d1-omni-600m-iphone.json`).
- First loads (into the app's Core AI cache) take 0.40–1.79 s for the `.aimodel` and 0.29–1.65 s for the h19p compile; later loads 0.03–0.84 s (round 11; `gate-d1-omni-600m-iphone.json` `load_s`).
- Pushing the 8.05 GB of graphs to the phone took 254 s (31.7 MB/s, 215 files, sizes checked, none pushed again) (round 11; `gate-d1-omni-600m-iphone.json` `transfer`).
- The phone was held 491 s in two holds, 13.5 min from the first take to the last release (round 11; `gate-d1-omni-600m-iphone.json` `hold`).
