# Gemma-4 E2B mobile mixed-bit QAT weights — extraction + Core AI transplant (P4+P5c device, P5d Mac DONE)

## P6 DONE (2026-07-17) — OFFICIAL checkpoint replaces the .litertlm reverse-engineering (bit-exact)

Google published the mobile mixed-bit QAT weights as a plain HF transformers checkpoint,
`google/gemma-4-E2B-it-qat-mobile-transformers` (Apache-2.0, `quant_method: "gemma"`, wNa8o8),
so the whole `litertlm_peek` reverse-engineering below is no longer needed as the SOURCE.
**Proven bit-exact equivalent to our .litertlm extract**, so the swap is safe and the published
conversion recipe stops reverse-engineering Google's binary.

- **Equivalence test** (`litertlm-convert/scripts/gemma4_official_vs_litertlm_equiv.py`): all 13
  weight families dequantize to `max|Δ|=0.0`, `code-exact=1.0`, `cos=+1.0` vs the validated
  extract. Official packing = UNSIGNED code minus midpoint (int4 −8, int2 −2); our extract stores
  two's-complement — different BYTES, identical VALUES. int8 (PLE gate/proj) genuinely signed.
  Per-row `weight_scale` bit-identical. `embed_tokens_per_layer.embedding_quantized [V,4480]` =
  35 layer-major 128-byte int4 blocks + `embedding_scale [V,35]`.
- **Converter** (`scripts/gemma4_official_to_mixedbit_extract.py`): official checkpoint → the SAME
  four artifacts the export consumes (`gemma4e2b_mixedbit_weights.safetensors` + `.scale`,
  manifest, `gemma4e2b_fp32_norms.safetensors`, `final_norm.f32.npy`) by unpack→repack into our
  kernel layout. Output vs the litertlm extract: **312/313 packed tensors byte-identical**; the
  only differing tensor is `decode.ple.model_proj` (official BF16 vs our int8, rel 0.4%, and the
  export requantizes it to int8 per-block-32 anyway). All 261 norm keys present, `skip_scale` =
  official `layer_scalar`. **Norms use official bf16, shift=0** (direct assignment; test:
  `|Δdirect| << |Δ(off+1)|`) — this ELIMINATES the fragile fp32-norm graph-attribution extract.
- **Norm gate** (`scripts/gemma4_official_norm_gate.sh`): fp16 MLX oracle, official bf16 norms vs
  litertlm fp32 norms, 3 canonical prompts → **FULL greedy match 40/40, 8/8, 35/35** (not even a
  near-tie fork). The 0.4% bf16 norm rounding is greedy-lossless. The official oracle's sky
  continuation reproduces the shipped `oracle_refs.json` fp16 EXACTLY.
- **Export + engine A/B**: `exports_official/gemma4_e2b_mixedbit_decode_ffnfused` (1.9 GB, from the
  official extract, `--fused-ffn`) exports clean; llm-benchmark p128 g128 = decode **79.4** /
  prefill 80.7 vs the shipped litertlm bundle's 79.8 / 81.0 (parity, same graph/kernels).
  ⚠️ llm-runner `--raw-tokens` greedy crashes `NDArrayDescriptor.swift:139 Shape … 256 … source
  shape 1` on BOTH bundles identically — the known Mac dynamic-query JIT limit (`coreai-env`;
  static-S=1 llm-benchmark path is fine), NOT an official-swap regression.
- **Repro**: `HF_HUB_DISABLE_XET=1` snapshot (or curl-resume the resolve URL — the metadata API
  504s intermittently and `snapshot_download` then falsely returns the local dir) →
  `scripts/gemma4_official_to_mixedbit_extract.py --official <dir> --out out/gemma4e2b_extract_official`
  → export `--extract-dir out/gemma4e2b_extract_official --out-dir exports_official`.
- **Not adopted/pushed yet** (user-gated external): switching the shipped export default source +
  publishing the official-sourced recipe. E4B (`gemma-4-E4B-it-qat-mobile-transformers`) is a
  NEW config (all-layer int4 MLP, int2 PLE) but 2258 MB/token ⇒ ~13-19 tok/s on A19 (byte floor) —
  too heavy to ship. MTP drafters now public too: `gemma-4-*-it-assistant` (the Section-11 drafter,
  non-quantized; `pre_projection [256,3072]` confirms our seam), incl. 12B/26B-A4B/31B for Stream C.

> **META (2026-07-03, post-hoc):** the whole "why is Core AI slower than LiteRT/MLX, can fusion/MTP
> fix it" investigation below re-derived what `coreai-vs-mlx-speed.md` already maps: dense CA≥MLX,
> MoE at MLX parity via the `gather_qmm` kernel, ceiling = MLX parity = byte floor (no beat). gemma4
> is the op-heavy-dense outlier LiteRT hand-fused; kernel tuning (already BW-bound 43.5 GB/s) and MTP
> (ALU-bound A19 ⇒ wash) both have no upside. **Read `coreai-vs-mlx-speed.md` before any future
> Core-AI-speed investigation.** The shippable win here is the +20% decode; 57 needs a gemma-specific
> fused loop that only ties LiteRT.

## P5d DONE (2026-07-15) — Mac MTP measured: ENGINE PARITY, not 150-185; the ceiling is the host encode tax

The P5c loop rehosted on the Mac (`coreai/_mtp_mac/`, standalone SwiftPM CLI on the bare CoreAI
framework; C1 discipline: owned MTLBuffers + `AsyncValue` zero-copy binds + `fn.encode(to:
ComputeStream)` + fp16 row-argmax straight off the logits buffer — no [1,S,vocab] flatten).
M4 Max, GEN=256, greedy, full-length BASE warm pass so per-position-length specialization is
outside the timed runs. **LOSSLESS 256/256 vs the verify-graph baseline on all three prompts.**

| prompt | pipelined engine (ffnfused) | BASE (verify-as-decoder) | MTP | tokens/round | vs engine |
|---|---|---|---|---|---|
| sky (explanatory) | **82.9** | 52.2 | **84.0** | 2.19 | **1.01× — parity** |
| sky rerun | | 52.1 | 83.6 | 2.19 | 1.01× |
| photosynthesis | | 50.7 | 66.0 | 1.75 | 0.80× |
| France (degenerates to EOS padding at forced 256) | | 52.4 | 59.1 | 1.52 | 0.71× |

- **The drafter transplant is fully healthy on the Mac**: tokens/round 2.19 == the α-oracle's
  2.22 for sky. (France's 1.52 is an artifact of forcing 256 tokens past a short answer — the
  model pads with turn-end tokens the drafter can't predict; real chat usage sits at 2.2-2.5.)
- **"verify ≈ free" IS true on the Mac GPU** — the whole S=4 verify executes in **6.2 ms of GPU
  time**. What kills the projection is the HOST side: `fn.encode` costs **12.7 ms of CPU per
  forward** (measured enc/drain split), + 6.3 ms for the 3 chained draft steps ⇒ ~25 ms/round.
  2.19 tokens / 25 ms = 84 tok/s = exactly engine parity, because the pipelined engine's own
  12.1 ms/token is the SAME encode tax hidden by overlapping encode(N+1) with GPU(N).
- **Conclusion: the 84→150-185 envelope is refuted for a host-driven loop** — it assumed verify
  rides the engine's per-token cost. See P5e below: the engine-integration ceiling itself is
  lower than that projection, and we reached it WITHOUT engine surgery.

## P5e DONE (2026-07-15, same session) — GPU-CHAINED round: 96.3 tok/s (1.16× engine), and the S=4 ceiling is now fully mapped

The encode tax can't be removed, but everything else can be folded under ONE drain per round.
Mechanism (all in `coreai/_mtp_mac/`, no engine patch):
- **drafter re-export adds an in-graph `amax` output** (`torch.argmax(logits).int()` — export
  script + `gemma4_mtp_drafter.py`, 2026-07-15). Draft tokens never round-trip to the host.
- **buffer aliasing via `AsyncValue(unsafeBuffer:byteOffset:)`**: draft i's `amax` output binds
  INTO the verify `input_ids` buffer at byteOffset 4·(i+1), and draft i+1's `input_ids` reads
  the same slot; draft i+1's `hidden` binds draft i's `proj_hidden` buffer directly.
  byteOffset=4 (int32 element) binds are ACCEPTED by the runtime, and in-flight encodes on one
  ComputeStream execute in encode order with inputs referenced at EXECUTION time (probe:
  PROBE=1, argmax + proj cos 1.000000 vs the CPU-synced reference).
- Round = encode d1,d2,d3,verify back-to-back (CPU), one drain, then CPU acceptance/argmaxes.

| prompt (GEN=256, greedy) | engine | classic MTP | **chained** | tokens/round | vs engine |
|---|---|---|---|---|---|
| sky | 82.9 | 79.5 | **96.3** | 2.19 | **1.16×** |
| photosynthesis | 82.9 | 67.0 | 77.4 | 1.75 | 0.93× |

Byte-lossless vs BASE 256/256 on both. Round = 22.7 ms = drafts-encode ~3 + verify-encode 12.7
+ verify GPU tail 6.2 + host ~0.8.

**Why this is the S=4 ceiling, engine-internal or not:**
1. verify encode 12.7 ms ≈ the pipelined engine's own 12.1 ms/token — the encode IS the
   runtime's per-forward floor for this graph (already gateup-FUSED; op count is the cost).
2. the 6.2 ms GPU tail can't overlap the next round: acceptance (CPU) decides the next
   position length and ids. An engine-internal implementation faces the same dependency ⇒
   same ~22.7 ms round. **The earlier "engine integration ⇒ 130-150" projection is refuted;
   the true S=4 engine-integration ceiling is ~96-100 sky-class, and the chained harness
   already delivers it.**
3. **S=8 is refuted by the α-oracle at depth 7** (`gemma4_mtp_alpha_oracle.py`, now
   `G_STEPS=7` env-selectable): aggregate tokens/step 3.31, but the gain is all in
   structured tasks (primes 5.29, tea 4.11); chat-class barely moves (sky 2.56, photo 2.14,
   creative 1.58). With ~+1 ms encode per extra draft and ~28 ms rounds, sky-class S=8 ≈ 91
   tok/s < S=4's 96.3. S=8 would pay ONLY on structured/extractive workloads (tea ~147) —
   a task-adaptive S is the only way it earns its keep.

Ladder after this session (Mac, same arch): stock engine 82.9 < **chained MTP 96.3** <
LiteRT-LM 113 < MLX 151. Remaining honest levers: task-adaptive S (structured workloads),
or nothing — the runtime encode floor bounds everything else.
⚠️ note for the device loop: the re-exported drafter now has THREE outputs (logits,
proj_hidden, amax); PipelinedBench's MtpDecoder binds two — bind amax to a scratch buffer
(or ignore-if-supported) before any device rerun.
- ⚠️ Session prerequisite: ALL beta1-era `.aimodel` exports abort on the beta3 runtime
  (26A5378j) with `LLVM ERROR: cannot unwrap empty odiec_module_t` — the known leaderboard-P0
  churn. The mixedbit family (decode_ffnfused / verify_s4 / mtp_drafter) was re-exported with
  the b2 toolchain (same scripts, no recipe change): loads clean, speed parity (82.9 vs 84.2),
  quality lossless. beta1 originals parked as `exports/<name>.beta1`.

## P5c DONE (2026-07-03) — MTP spec-decode runs on device; refutes "verify is free" on ALU-bound A19

Full MTP loop shipped and measured on the iPhone (PB_MTP=1, `ondevice/PipelinedBench/Sources/
MtpDecoder.swift`; two host-driven AOT/JIT bundles). **The drafter transplant WORKS: 2.40
tokens/verify-step on device, matching the α-oracle's 2.5** (the drafter drafts as Google trained
it). Warm speedup over the verify-graph-as-decoder baseline = **1.79×** (14.3 → 25.7 tok/s, draft
19 ms/round warm).

**But MTP does NOT beat the plain mixed-bit decode bundle (P4b fresh 36.5 tok/s > MTP 25.7), and
that is the real finding:** the envelope in the "MTP drafter — RESOLVED" section assumed a 4-token
verify ≈ one decode step (true only when decode is BANDWIDTH-bound). P4b already measured this
graph is **ALU/dispatch-bound on the A19** (prefill ≈ decode, 43 GB/s ⇒ compute-bound not
byte-bound). So the S=4 verify costs ~2.6× a decode step (70 ms/round vs the decode bundle's 27
ms/token — the 262144-row head + FFN run 4× the FLOPs), and at 2.4 accepted tokens/round the
amortization can't cover it: break-even needs ~3.3 tokens/round. **MTP is a bandwidth-bound-runtime
lever; on the compute-bound A19 mixed-bit graph it's a wash-to-loss.** It WOULD win (a) on a
bandwidth-bound device/runtime (the Mac dispatch-bound path, or a future weight-bound A-series
kernel), or (b) if the verify head is shrunk so S=4 isn't 4× the head cost (vocab-prune / two-stage
head), or (c) paired with the decode graph as the base instead of a full S=4 verify (draft-cheap,
verify-cheap asymmetry). Numbers/tokens: `ondevice/_pipelined_dev_mtp_r{3,4}.log`.

Losslessness: MTP greedy matches the baseline for a 4-token prefix then forks and RE-CONVERGES
(e.g. `...496, [1822 vs 14853], 2934, 236888, ...`) — the documented fp16 near-tie fork class
(batched S=4 attention reduces in a different order than S=1; argmax flips at ties, greedy heals),
NOT a logic bug. A bitwise-lossless claim needs a weight-bound verify == decode numerics path.

Artifacts (all uncommitted): `models/macos/gemma4_mtp_drafter.py` (Section-11 transplant torch
module + static int8 activation fake-quant at the 18 QAT boundaries — fp16-only drafting drops α),
`gemma4_mixedbit_verify.py` (S=4 companion, extra `activations` + slot-11/14 kv-row outputs),
`gemma4_metal_mlp_m4.py` (M=4 verify kernels: weights read once for 4 rows, float4 across M),
exports `export_gemma4_mtp_drafter.py` / `export_gemma4_mixedbit_verify_pipelined.py`, gates
`_smoke/check_gemma4_mtp_drafter_parity_real.py` (torch 2.043 tok/step vs tflite 2.000) +
`gate_gemma4_mixedbit_verify_s4.py` (France 8/8 exact drafts) + `gate_gemma4_mtp_drafter_bundle.py`
(GPU-runtime argmax 5/5). Drafter AOT h18p FAILS (ANE f32/f16 verify in the post_proj RMSNorm) →
loads as JIT `.aimodel` (small static graph, specializes once); verify uses AOT `.aimodelc`.
**Extraction extras** (litertlm-convert): `scripts/extract_gemma4_drafter_extras.py` (norms /
q-norms / skip-scales / rope freqs / post_proj int8 / softcap 30) + `gemma4_drafter_act_quant.json`
(18 activation scales) + `gemma4_drafter_real_cases.py` (MLX→tflite parity cases).

## P4 DONE (2026-07-03) — iPhone 17 Pro: fresh decode 36.5 / prefill 51.7 (+20%/+28% over ship)

PipelinedBench (stock engine, AOT h18p `.aimodelc`, `PB_MODEL=gemma4_e2b_mixedbit_device`,
p128 g256 ×2, logs `ondevice/_pipelined_dev_mixedbit_r{1,2,3}.log`):

| run | trial1 decode | trial2 decode | prefill | state |
|---|---|---|---|---|
| r1 | 13.5 | 14.0 | 16.4 | just-installed — INVALID (see traps) |
| r2 | **35.9** | 30.2 | 39.4 | settled |
| r3 | **36.5** | 25.9 | 51.7 (t1) | 30 s after r2 → bigger thermal droop |

- **Fresh = 36.5 decode / 51.7 prefill** (trial1 of settled runs; 35.9/36.5 agree).
  vs shipped int4lin tbl (30.3/40.5): **+20% decode, +28% prefill**. Sustained trend ~26-30.
- **Gate: ≥45 MISSED (81%), ≥57 missed.** Effective BW 0.783×36.5 = **28.6 GB/s** vs int4lin's
  ~35 — the custom unpack kernels (int2sym/int4aff/gateup, M4-tuned R=4/SGY=8) convert only
  ~82% of the native path's bandwidth on A19; LiteRT hits ~44 GB/s on the same silicon+weights,
  so the remaining 36→57 is a KERNEL-ENGINEERING gap, not a byte-thesis failure. r1's shape was
  diagnostic: prefill≈decode ⇒ graph-bound (unpack ALU/occupancy), not sampler-roundtrip-bound.
- Quality: device greedy = the SAME engine near-tie forks as the Mac Swift engine (sky forks at
  token 2 "appears"→"is", France "**Paris**."→" Paris."), coherent correct answers — the
  documented engine-vs-delegate fp16 class; quality anchor stays the python-delegate 3/3 id-exact.
- Loads: cold 6.6 s / warm 2.5-2.8 s (AOT, no on-device specialization).
- **Paths to 57**: (a) A19 kernel tuning session (vectorized unpack, fp16 accumulate, R/SGY sweep,
  or TensorOps int8 quantized-matmul path per WWDC330); (b) **P5c MTP** — measured tokens/step
  2.506 ⇒ 36.5 × ~2.2 ≈ **~80 fresh** (verify S=4 amortizes the unpack, so the multiplier holds
  on an ALU-bound graph too). MTP alone clears 57 even at today's kernels.

### Kernel-probe CORRECTION (2026-07-03, same day) — the kernels were never the problem

A19 kernel sweep (`conversion/export_gemma4_kprobe_int2.py` → 4 tiny bundles, 6×int2-FFN +
262144-row head = 185 MB/run, PB_MM harness, logs `ondevice/_pipelined_dev_kp_*.log`):

| variant | warm_med ms | warm_min |
|---|---|---|
| base (shipped scalar, R4/SGY8) | **4.273** | 4.101 |
| r8sgy4 (tiling) | 4.949 | 4.155 |
| lutf4 (threadgroup byte-LUT + float4) | 5.006 | 4.114 |
| lutr2 | 6.131 | 4.477 |

**The shipped scalar kernel already runs BANDWIDTH-BOUND at ~43.5 GB/s on A19** (LiteRT-class;
the unpack ALU hides entirely under memory latency — all variants' warm_min within noise, LUT
no better). Revised token accounting: prefill 51.7 tok/s = 19.3 ms ≈ pure BW at ~43 GB/s ⇒ the
GRAPH is at the BW limit already; decode 27.4 ms = prefill + **~8 ms/token serialized sampler
roundtrip** (same class as the int4lin tbl bundle's measured +13 ms sampler tax; the earlier
"custom kernels underperform on A19" diagnosis was WRONG — an artifact of r1's invalid
just-installed numbers). ⇒ Kernel tuning is closed (no re-export); the decode levers are
**(1) MTP verify amortization (1 sampler roundtrip per ~2.5 accepted tokens → ~90 fresh
projected) and (2) engine sampler/command-buffer fusion** (the G-TBL lever, engine surgery).
- Device traps (new): **just-installed runs measure ~60% low** after a 2 GB push (r1 13.8 vs 36.5
  settled — far beyond the −15% precedent; wait minutes + rerun); back-to-back runs droop trial2
  by 16-29% (thermal) — fresh = trial1 of a settled run.

**Goal:** match/beat LiteRT-LM's Gemma-4 iPhone decode (30.8 tok/s sustained, 0.146 J/tok,
4,074 tok/1% — the ONLY model where LiteRT beats Core AI on iPhone; see
`litertlm-convert/reports/litert-community-vs-mlx-coreai.md`) by transplanting Google's mobile
QAT mixed 2/4/8-bit weights into the Core AI gemma4 runtime. Core AI today: E2B VL 25.5 tok/s
(int4linsym, 2.0 GB). The gap driver is bytes/token, not kernels — LiteRT loses on every generic
model on Apple GPU (WebGPU delegate), it wins Gemma only via the recipe below.

## Phase 0 result (2026-07-02): recipe fully decoded, weights extractable

**Artifact:** `litert-community/gemma-4-E2B-it-litert-lm` / `gemma-4-E2B-it.litertlm` (2.59 GB)
→ `litertlm-convert/src_models/gemma-4-E2B-it-litert-lm/`.
**Extraction:** `litert_lm_builder.litertlm_peek.peek_litertlm_file(path, dump_dir, stream)`
(in `litertlm-convert/.venv`) dumps all 12 sections → `litertlm-convert/out/gemma4e2b_extract/`.
Per-tensor map (name/type/shape/bytes/nscale/nzp): `out/gemma4e2b_extract/decode_weight_map.json`
(parse = `ai_edge_litert.schema_py_generated`, mmap the .tflite, walk Model→Subgraph(0)→Tensors,
buffer size via `Buffers(t.Buffer()).DataLength()`).

### Sections (12)
| # | model_type | size | content |
|---|---|---|---|
| 0 | LlmMetadataProto | 12 KB | tokens, jinja template, gemma4 model_type |
| 1 | SP_Tokenizer | 4.7 MB | sentencepiece |
| 2 | tf_lite_embedder | 104 MB | input embed [262144,1536] **INT2** |
| 3 | tf_lite_per_layer_embedder | 1.28 GB | PLE table, **INT4** (1174 MB) — mmap'd, gather-only (~4.5 KB/tok read) |
| 4–6 | audio encoder/adapter/eoa | 104 MB | cpu-constrained |
| 7–9 | vision encoder/adapter/eov | 229 MB | fp16-activation |
| 10 | tf_lite_prefill_decode | 818 MB | **the transformer — mixed-bit map below** |
| 11 | **tf_lite_mtp_drafter** | 44 MB | INT4 36.7 + INT8 3.8 — **bundled spec-decode draft model** |

### The decode recipe (Section 10, 35 layers — per-channel SYMMETRIC linear: all zero_points are 0)
| weights | layers | bits | shape | notes |
|---|---|---|---|---|
| FFN gating1/gating2/down | L0–L14 | **INT4** | [6144,1536]×2 + [1536,6144] | these 15 layers also have own k/v (INT4, [256,1536] = 1 KV head × 256) |
| FFN gating1/gating2/down | L15–L34 | **INT2** | **[12288,1536]×2 + [1536,12288]** | **2× wider FFN at 2-bit = SAME bytes as 4bit@6144** (elastic capacity trade, bandwidth-neutral); KV shared from earlier layers |
| attn q / o | all 35 | **INT4** | [2048,1536] / [1536,2048] | 8 heads × head_dim 256 |
| PLE gate/proj | all 35 | **INT8** | [256,1536] / [1536,256] | + per_layer_model_projection [8960,1536] INT8 |
| embed ≡ lm_head (tied) | — | **INT2** | [262144,1536] | per-row scale, zp=0 |
| norms etc. | — | FLOAT32 | | 1.1 MB total |

- Quantization: `nscale = nzp = out_channels`, `qdim=0`, **zero_points stored but ALL ZERO ⇒
  dequant = code × scale[row]** (simpler than int4km — no LUT, no zp term).
  `bytes == nel*bits/8` exactly (dense packing, TensorType INT2/INT4 — the Python Interpreter
  API masks these as int8; parse the flatbuffer, not `get_tensor_details()`).
- **Packing convention VERIFIED** (vs MLX-4bit dequant oracle, embed rows mean-cos **0.8904**;
  wrong conventions give −0.34/≈0): INT2 = 4 codes/byte, first code in bits[1:0], signed two's
  complement (-2..1). INT4 assumed analogous (low nibble first, signed).
- Totals: INT2 383.8 MB / INT4 351.5 MB / INT8 41.3 MB ⇒ **active decode footprint ~777 MB**
  vs Core AI int4linsym E2B **2.0 GB** = **~2.6× fewer bytes/token** (decode is BW-bound; this
  is LiteRT's Gemma win, quantified).
- PLE tables: 35 × [262144,256] INT4 (1.17 GB, mmap/gather-only). Drafter: 22 tensors INT4/INT8.

### P1 DONE — portable extraction
`litertlm-convert/scripts/extract_gemma4_mixedbit.py` → `out/gemma4e2b_extract/
gemma4e2b_mixedbit_weights.safetensors` (335 tensors, 2.09 GB packed: embed 1 / ple_table 35 /
decode 277 / drafter 22) + `gemma4e2b_mixedbit_manifest.json` (per-tensor shape/bits/qdim/zp0).
Keys: `decode.layer_NN.{attn.{q,k,v,o}, mlp.{gating1,gating2,down}, ple.{gate,proj}}`,
`decode.lm_head`, `embed.composite`, `ple_table.compositeN`.

### MTP drafter — RESOLVED (2026-07-03): OFF in every LiteRT number; full wiring decoded
**`enable_speculative_decoding = false` is the litert-lm engine default** (`llm_executor_settings.h:258`,
`litert_lm_lib.h:127`; our bench never set it) ⇒ **LiteRT's Gemma4 iPhone numbers (57 fresh /
30.8 sustained) are drafter-OFF** — pure bandwidth+kernels. The bundled drafter is an UNUSED
multiplier; transplanting it is a lever LiteRT itself doesn't ship enabled.

**Drafter architecture (Section 11 graph + `llm_litert_mtp_drafter.{h,cc}`):** EAGLE-class trained
draft head —
- input `activations [1,1,3072]` = concat(main final hidden 1536, token embedding 1536) →
  `dot_general [256,3072]` int8 down-projection to drafter dim **256**;
- 4 tiny layers (FFN 2048 int4, attn q/o int8, 4 heads, partial RoPE 128) with **NO own K/V:
  cross-attention into the MAIN model's producer KV caches** — layers 0-2 read L13 (sliding,
  hd 256), layer 3 reads L14 (full, hd 512). Our unified pipelined KV pair already holds both slots;
- own narrow head `lm_head [262144, 256]` int4 (33 MB = most of the 44 MB) + second output
  `projected hidden [1,1,1536]` for CHAINED drafting (no main-model re-entry per draft);
- draft cost ≈ 35 MB read/draft ≈ 4.5% of a main step.

**Verify:** the base tflite has a `verify` signature `[1, 4]` (⇒ **G = 3 draft steps**), decode
outputs `activations [1,1,1536]` (the drafter kickoff tap); verifier = one batched S=4 main
forward. KEY: verify reads the weights ONCE for 4 tokens — on the dispatch-bound Mac AND the
BW-bound iPhone a 4-token verify costs ≈ one decode step, so expected tokens/step at acceptance α
is ~1+α+α²+α³ (α .7 → ~2.2×, α .8 → ~2.5×). Envelope: Mac 84 → ~150-185 (would pass LiteRT 113
and MLX 151); iPhone 45 → ~80-100 fresh vs LiteRT's 57.

**P5 plan:** (b) MLX α-oracle — implement the drafter in mlx cross-attending the main cache,
measure α on real prompts = the go/no-go gate; (c) Core AI integration — drafter as a tiny extra
graph (packed-embed gather + int8/int4 transplant machinery all exists), verify companion graph
S=4 (matvec kernels need M>1 or verify rides MPSGraph int4lin), Stream C spec-decode loop (C2
draft-model slot).

### P5b DONE (2026-07-03) — α measured with the REAL drafter tflite: tokens/step 2.51

**Method (better than the planned MLX reimplementation):** Section 11 runs directly on the
ai_edge_litert CPU interpreter (XNNPACK executes the composite decompositions bundled as
subgraphs 1-29; **0.02 s/invoke**) — Google's quantized drafter bit-exact, zero transplant risk.
The MLX fp16 main model (the delegate-id-exact oracle host) supplies greedy targets + the
cross-runtime seam buffers. `litertlm-convert/scripts/gemma4_mtp_alpha_oracle.py`
(results `_gemma4mb_mtp_alpha.{log,json}` in coreai/).

**Seam contract (all verified against `llm_litert_mtp_drafter.cc` + graph dumps
`_gemma4mb_drafter_dump.txt`):**
- `Draft(position=p, token_id=seq[p])`: drafter `input_pos = p-1` (the hidden's position),
  `param_tensor[1,1,1,7] = [p-1, p, p, 0,0,0,0]` (int32; = LlmRuntimeParams: cache write window
  + bmm end-channel ⇒ in an oracle, just the valid-cache length), `mask` = **bool** causal
  true[0..p-1] — all three FIXED across the G=3 chained steps (litert sets them once per Draft).
- activations = concat(**embedding first**, hidden second) — `emb_norm(seq[p]) ⊕ h̃[p-1]`;
  emb includes the ×√1536 normalizer (baked into the Section-2 embedder as a trailing MUL,
  39.191837 == mlx `embed_scale`); h̃ = **post-final-norm** hidden (the decode `activations`
  output taps final_norm(h) — traced in the Section-10 graph). Chained steps:
  `emb_norm(d_i) ⊕ projected_activations` (drafter's 2nd output = post_proj(final_norm(h_d))).
- Drafter KV inputs = the main L13/L14 caches **int8 per-tensor** (zp=0, scales from the tflite:
  k13 0.00595525 / v13 0.04724411 / k14 0.00109123 / v14 0.01785714), K [1,1,32003,hd]
  position-major, V transposed [1,1,hd,32003]; contents = post-q/k/v-norm post-RoPE (mlx cache
  quantized with those scales reproduces it). Layers 0-2 build their own sliding-window mask
  in-graph from input_pos (window constants baked); only layer 3 (L14, full attn) reads the
  external mask.
- Layer-3 q-norm has its own [512] weight; layers 0-2 share one [256] (`arith.constant3`);
  head has the tanh softcap (MUL-TANH-MUL, logits max ≈22 < 30 observed ✓).

**Results (409 draftable positions, 6 prompts, greedy, fp16 main):**
| prompt class | α1 | tokens/step |
|---|---|---|
| extractive (France) | 1.000 | 4.00 |
| structured list (primes) | 0.903 | 3.43 |
| procedural (tea steps) | 0.806 | 2.92 |
| explanatory (sky) | 0.645 | 2.22 |
| explanatory short (photosynthesis) | 0.531 | 1.97 |
| creative story (robot) | 0.366 | 1.56 |
| **aggregate** | **0.672** | **2.506** |

α2|1 = 0.702, α3|12 = 0.767 — conditional acceptance RISES down the chain (EAGLE signature; the
projected-activations chaining works). Accept hist (0..3): [134, 82, 45, 148] — 36% of steps
accept the full 3 drafts.

**Gate verdict: α1 0.672 = gray zone by the letter (0.5–0.7), but GO by the metric that sets
speed:** measured 2.506 tokens/verify-step ≥ the ~2.2 the "α=0.7 GO" envelope promised (the α1
gate under-counts because chain conditionals exceed α1). Net of ~13.5% draft cost (3 × 4.5%):
**~2.2× expected — Mac 84 → ~185, iPhone 45-proj → ~99 fresh; worst case (creative) 1.37× →
iPhone ~62, still > LiteRT 57.** Caveats: α measured vs the fp16 oracle as target (device engine
near-tie forks may shave a few points); chat-style usage skews to the explanatory/structured
rows. P5c decision = user call; recommendation GO.

## Why transplant works (vs re-QAT)
These are Google's QAT weights FOR this exact config — transplanting the quantized values bit-exact
and dequantizing identically (per-channel affine) reproduces LiteRT's numerics by construction.
No training, no quality risk beyond runtime numerics. (The `q4_0-unquantized` HF release is a
DIFFERENT QAT run for uniform 4-bit — do not mix.)

## P2 progress (2026-07-02, same session)
- **All weight families verified vs MLX oracle** (`mlx-community/gemma-4-e2b-it-4bit` dequant),
  layer-0 cos: q .910 / k .944 / v .948 / o .915 / gate .926 / up .931 / down .943 / ple.gate .952
  / ple.proj .963 — cross-terms ≈0 ⇒ **gating1=gate_proj, gating2=up_proj** resolved.
- **Architecture = the public E2B arch exactly** (my initial "mobile differs" read was wrong):
  L4/9/14/19/24/29/34 = global-attn, head_dim **512** (q [4096,1536], kv [512,1536]); sliding
  layers head_dim 256 (q [2048,1536], kv [256,1536]); L15–34 FFN 12288 = `use_double_wide_mlp`.
  All of this is config-driven in `mlx_lm/models/gemma4_text.py` (global_head_dim / 
  num_kv_shared_layers / use_double_wide_mlp) → perfect reference implementation + oracle host.
- **fp32 norms extracted** → `gemma4e2b_fp32_norms.safetensors` (261 tensors,
  `scripts/extract_gemma4_norms.py`, consumer-scope attribution buffer-level across all 976
  subgraphs — names are anonymized `jax2tf_arg_N`/deduped `arith.constantN`):
  - 5 residual norms × 35 layers = real learned vectors (values ~0.05–75 ⇒ stored EFFECTIVE
    scale; HF/(1+w) convention ⇒ subtract 1 when mapping — verify in oracle).
  - **q_norm/k_norm = per-layer SCALARS** (nuniq=1, e.g. L00 q=0.9846 k=0.1269; L04 k=0.0648)
    = attention temperature per layer; k_norm absent L15+ by construction (no K computed).
  - **skip_scale = per-layer learned scalar** (`_maybe_apply_skip_scale`, e.g. L10=0.4438) —
    find its application point + MLX equivalent before the oracle run.
  - TODO: final_norm regex missed (the 1×(1536,) consumed by `StatefulPartitionedCall:N`) — grab it.
  - ple_projection_norm = real vector (256,).

## P2 DONE — transplant verified end-to-end (2026-07-02)
`scripts/gemma4_mixedbit_oracle.py` dequantizes everything and injects into
`mlx_lm.models.gemma4_text.Model` (540 params, `load_weights(strict=True)`), greedy-decodes.
Conventions settled by direct comparison with the MLX checkpoint (all cos ≥0.9987):
- norms = direct assignment (MLX `nn.RMSNorm` consumes effective scales — NO (1+w) shift);
- **`layer_scalar` exists in the public arch** (`h = h * layer_scalar` at layer end) and our
  extracted skip_scale matches it (L10: 0.4438 vs 0.443) — no module patch needed;
- q/k norms are constant vectors in the public checkpoint too (per-layer attention temperature);
- embed ≡ lm_head bit-identical (tied) — `tie_word_embeddings=True` correct;
- PLE = 35 × [262144,256] tables concat axis=1 → embed_tokens_per_layer [262144,8960].

**Greedy match vs `litert-mac-verify --greedy` (same prompt, "Why is the sky blue?"):**
14-token EXACT prefix ("The sky appears blue due to a phenomenon called **Rayleigh
scattering**. Here"), then a synonym fork ("Here's a breakdown" vs "Here is a detailed
breakdown") that MOVES with oracle dtype (bf16 "of the process" → fp16 "of why" = LiteRT's
wording) ⇒ near-tie fusion-numerics class, same PASS standard as prior device ports
(margin-fork precedent in [[project_gemma4_vl_port]]). **Transplant is numerically faithful.**

Bonus datapoint: LiteRT-LM Gemma4-E2B on Mac M4 Max (WebGPU): **decode 113.2 / prefill 52.2
tok/s** — their Mac decode BEATS our current E2B int4linsym engine (82.4 tbl) purely on bytes
(777MB vs 2.0GB), while their prefill is 20–80× slower than ours. Byte-parity thesis reconfirmed.

## Plan
1. ~~P1 extract~~ ✅  2. ~~P2 oracle~~ ✅  3. ~~P3 Core AI receiving end (Mac)~~ ✅ (below)
4. **P4 — device**: iPhone PipelinedBench (A19), AOT h18p required (large graph). **Goalposts
   REVISED (2026-07-03, user-confirmed): LiteRT Gemma4 iPhone = 57 tok/s FRESH** (the 30.8 in
   `litert-community-vs-mlx-coreai.md` is the SUSTAINED/energy-protocol number — do not mix
   conditions). Fresh-vs-fresh today: our int4lin tbl 30.3 (~1.15 GB/tok → ~35 GB/s effective)
   vs LiteRT 57 (777 MB → ~44 GB/s effective) — their lead = 1.48× bytes × ~1.27× kernel
   efficiency. Mixed-bit at our current effective BW projects **~45 tok/s fresh** (35/0.783);
   beating 57 fresh needs (a) LiteRT-class effective BW (44 GB/s) and/or (b) the extracted MTP
   drafter as spec-decode (~1.3–2× on greedy; 22 tensors already in the extraction — natural P5,
   rides Stream C machinery). Sustained target stays ≥30.8 at byte parity (similar DRAM energy).
   Success gates: fresh ≥45 confirms the byte thesis; fresh ≥57 wins outright; quality =
   greedy-match vs the python-delegate ids; prefill lead intact.

## P3 DONE (2026-07-02) — the transplant runs on Core AI, quality gate PASS

**Bundle:** `coreai-models/exports/gemma4_e2b_mixedbit_decode/` (1.9 GB, `--max-ctx 4096`) via
`conversion/export_gemma4_mixedbit_decode_pipelined.py`. Per-token read audit: INT2 383.8 +
INT4 328.5 + INT8 39.4 + fp16 k/v 28.3 + scales ≈ **783 MB/token** (shipped int4lin ≈ 2.0 GB).

### Receiving-end design (what got built)
- **ONE new kernel** — `gemma4_metal_mlp_int2.py::build_fused_int2sym_kernel`: INT2 pure-symmetric
  matvec, 16 codes/uint32, branchless sign decode `(q^2)-2` (no LUT, no tg staging needed);
  per-ROW scale ⇒ the hot loop accumulates the raw integer-weighted sum and multiplies by
  `scale[n]` ONCE after `simd_sum` — cheaper inner loop than int4km. The raw extract bytes ARE
  the kernel layout (`packed_u8.view(torch.uint32)`, zero repacking). Serves FFN L15-34 + the
  262144-row lm_head.
- **INT4 = existing affine kernel, exact mapping** (`int4sym_to_affine`): q_u = q_s+8,
  sc = scale[n] broadcast along K-groups, bi = −8·scale[n] (exact in fp16). Serves FFN L0-14 +
  attn q/o. attn k/v (small-N) ride dequantized fp16; PLE gate/proj + model_proj = shipped int8
  per-block-32 requant (near-lossless re-grid of per-channel int8).
- **Tables in-graph, NO static inputs** (`gemma4_mixedbit_pipelined.py`): embed INT2
  ([V,384] u8 + [V] f32 scale) and PLE INT4 ([V,4480] u8 + [V,35] f32 scales) ride as in-graph
  uint8 constants; bit-unpack = **byte-LUT `index_select`** (lut2 [256,4] / lut4 [256,2] fp16) —
  the exact op class the shipped tbl variant proved, no bitwise ops in-graph. The bundle runs on
  stock llm-benchmark / PipelinedBench with no `staticInputBuffers` binding and no engine patch.
- Kernel GPU parity on real extract tensors (int2 FFN L20 / int4 FFN L00 / attn.q / full lm_head):
  rel_l2 3–7e-4 (fp16 I/O class), cos 1.000000, head argmax exact.

### Quality gate (python GPU delegate, id-exact vs the P2 MLX oracle) — 3/3 PASS
Same raw ids (BOS + spm template), S=1 steps, host argmax:
- "Why is the sky blue?" — **== fp16 oracle EXACT 32/32** (forks from the bf16 oracle exactly at
  the known dtype-fork position 20);
- "What is the capital of France?" — **8/8 EXACT** (both dtypes, incl. the `**Paris**.` tokens);
- "Explain photosynthesis in one sentence." — **32/32 EXACT** (both dtypes).
Stronger than the P2 pass standard (14-token prefix + margin fork); exercises 35 layers, both
attn types, KV sharing, PLE, packed-table gathers, both kernels, softcap head over ~120 engine
steps. Oracle refs live in the bundle dir (`oracle_refs.json`).
- ⚠️ python-runtime harness notes: every new position_ids length triggers a delegate
  re-specialization (~10 s/step incl. a benign per-step ANECCompile-fail→GPU fallback log), and a
  long run eventually wedged the MTL4 command queue (`MTL4CommandQueueErrorDomain error 1`,
  process-local, killed cleanly) — keep python gates short; the Swift engine specializes once and
  has neither issue.
- **Swift-engine sampler forks near-ties early** (llm-runner `--raw-tokens`, greedy): sky forks
  at token 2 ("appears"→"is"), France at 6 ("**Paris**."→"Paris."), answers correct and fluent;
  NOT warmup/KV pollution (same with `--warmup off`). Same engine-vs-delegate fp16-numerics class
  as the shipped bundles' "Mac-engine determinism fork" (granite precedent: judge by the gate).
  Device gate in P4 should anchor on the python-delegate ids + in-app behavior.

### Speed (M4 Max, llm-benchmark Release, chunk=1, p128 g256, n=3)
Base bundle **decode 78.1 / prefill 80.1 tok/s** — parity with the shipped int4lin `--tbl`
(77.0–78.9) despite 2.6× fewer bytes/token: at 783 MB/token this is only ~61 GB/s effective —
**the Mac is dispatch/ALU-bound on this graph, not bandwidth-bound**. Runtime ladder measured on
this Mac, same arch: Core AI pipelined 78 / LiteRT-LM 113.2 / **MLX 151**
(`litertlm-convert/scripts/gemma4_mlx_decode_bench.py`, ~166 GB/s effective) — the gap to
113–151 is RUNTIME-class (fused loop vs per-op delegate), not arch- or byte-bound.

**`--fused-ffn` (+8%): decode 84.2 / prefill 86.1.** The one dispatch lever NOT killed by the
2026-06-10 Lever-B closure (glue fusion = wash because MPSGraph already fuses elementwise; SDPA
fold = occupancy regression): merge the two REAL gate/up matvecs + gelu + mul into ONE kernel
dispatch (`build_gateup_int2sym_kernel` / `build_gateup_int4aff_kernel`, fp32 precise-tanh gelu
in the epilogue; France gate stays 8/8 id-exact). FFN goes ~5 dispatches → 2 per layer. This is
the P4 default too (fewer launches + activation read once also help the device). Remaining Mac
ideas are each ≤~2-3%: pre-FFN-norm folded into the gateup prologue (rides the same dispatch),
PLE-block fusion (6 tiny int8 ops → 1-2 kernels); MLX-class 151 is unreachable inside the
per-op delegate (measured precedent), so Mac stops here unless the engine itself changes.
The byte win is aimed at the **bandwidth-bound A19** (P4).

## Repro commands
```bash
cd ~/code/litertlm-convert
HF_HUB_DISABLE_XET=1 .venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('litert-community/gemma-4-E2B-it-litert-lm', local_dir='src_models/gemma-4-E2B-it-litert-lm')"
.venv/bin/python -c "
import sys
from litert_lm_builder import litertlm_peek as peek
peek.peek_litertlm_file('src_models/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm','out/gemma4e2b_extract',sys.stdout)"
```
