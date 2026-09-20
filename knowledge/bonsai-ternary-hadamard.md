# Bonsai 2 27B — ternary g128 + blockwise Hadamard: the packing contract for a Core AI port

Read-through of PrismML's kernel sources and the shipped packs, written before any kernel work so
the port is built on the file formats and the math as they actually are, not as the model cards
summarise them. Everything here is cited to a file and line; the few claims that are not are
marked as **verified** (decoded from real bytes) or **inferred**.

**Sources (pinned):**

- `PrismML-Eng/llama.cpp`, branch `prism`, commit `f0a2b5dc9ea066780b9b410d2c0f95d675859bf7`
  (2026-09-17). Cited below as `lc:<path>:<line>`.
- `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` (HF), the bundled loader `runtime/*.py` — the
  **reference runtime for the token gate** (sha256 of the four files matches the pin in
  `PrismML-Eng/Bonsai-demo/scripts/bonsai2-runtime.sha256`). Cited as `mlx:<file>:<line>`.
- `prism-ml/Ternary-Bonsai-2-27B-gguf`: the first 48 MB of `Ternary-Bonsai-2-27B-PQ2_0.gguf`
  (header + start of the data section), parsed by hand. No full model was downloaded.
- Whitepaper `bonsai-2-27b-whitepaper.pdf` (Bonsai-demo, commit `c398c6e`), §2.1–2.4, A.2.
- The zoo side: `coreai-models` `models/macos/qwen3_5.py` (commit `b1cb71b`, referred to as
  `zoo:qwen3_5.py`), `bitcpm_ternary_metal.py`, and the overlay's `bitcpm_ternary_gemm.py`.

## 0. TL;DR

- **The weights are ternary at group 128 with one fp16 scale per group**, stored in a rotated
  basis. Two GGUF packings (PQ2_0 = one 2-bit slot per trit, PTQ1_0 = five trits per byte) and
  one MLX packing (affine 2-bit, `bias = -scale`) all decode to **the same ternary values and
  the same scales**. PQ2_0's 2-bit words are **bit-identical to the zoo's BitCPM `[N, K/16]
  uint32` packing** (code `j` of a 16-group at bits `2j`, `value = code - 1`). Verified on real
  blocks. The only kernel change is the scale granularity: **256 → 128**.
- **The rotation is `R = (1/√1024) · H₁₀₂₄ · S`**, applied blockwise along the input (K) axis:
  multiply the activation by a fixed ±1 sign vector (one per K-width: 5120, 6144, 17408), then
  a 1024-point Walsh–Hadamard transform on each of the K/1024 consecutive blocks, scaled by
  1/32. It sits **immediately before every folded matmul**, once per activation site (q/k/v
  share one transform, gate/up share one). K = 17408 is just 17 independent 1024-blocks — no
  factorisation.
- **Everything with a K axis is ternary**: all 4 attention linears, all 3 GDN linears
  (qkv, z-gate, out), all 3 MLP linears, the **lm_head** and the **embedding table** (stored
  rotated; the lookup result gets the inverse transform). 401 folded matrices + 1 inverse table.
- **Non-ternary (26.2 M params, ~52 MB):** the GDN's `in_proj_a` / `in_proj_b` (bf16 in GGUF),
  `conv1d`, `A_log`, `dt_bias`, the gated `norm`, and every RMSNorm gain.
- **Architecture = Qwen3.6-27B's shapes exactly** (the zoo already ports that in `qwen3_5.py`):
  64 layers, 3:1 GDN/full, 5120 hidden, 17408 FFN, 24q/4kv × 256, GDN 48v/16k × 128, vocab
  248320, untied head. Bonsai drops the MTP head. The zoo's model file needs a linear
  replacement and three loader conventions — no new architecture code.

## 1. The model behind the pack

`general.architecture = qwen35` in the GGUF; the MLX pack's `text_config.model_type` is
`qwen3_5_text` with `model_type: prism_hadamard_qwen35` only at the top level (the marker that
tells stock loaders to refuse). Hyperparameters from the GGUF header, all matching
`Qwen/Qwen3.8-27B`'s `config.json`:

| key | value |
|---|---|
| block_count / embedding_length / feed_forward_length | 64 / 5120 / 17408 |
| attention head_count / head_count_kv / key_length | 24 / 4 / 256 |
| rope: freq_base, dimension_count, dimension_sections | 1e7, 64 (=0.25 × 256), [11, 11, 10, 0] |
| ssm: state_size (dk), inner_size, time_step_rank (n_v), group_count (n_k), conv_kernel | 128, 6144, 48, 16, 4 |
| full_attention_interval | 4 (layers 3, 7, …, 63 are full attention: 16 of 64) |
| vocab | 248320, untied (`output.weight` and `token_embd.weight` are separate tensors) |
| rms eps | 1e-6 |

This is the same decoder the zoo's `models/qwen3.6-27b` card already runs at int8 (15.9 tok/s
on M4 Max, `~28 GB/token`). `Qwen3.8-27B` differs from `Qwen3.6-27B` only in weights, MTP config
(`mtp_num_hidden_layers` 1 in the base, 0 in the Bonsai pack) and training; the zoo never
exports MTP anyway.

## 2. Tensor inventory (from the PQ2_0 header, 851 tensors)

GGUF shapes are `ne` order (K first). `N` = output rows, `K` = reduction.

| GGUF name (per layer unless noted) | type | K × N | count | zoo module |
|---|---|---|---|---|
| `token_embd.weight` | **PQ2_0** | 5120 × 248320 | 1 | `embed_tokens` — **inverse-transformed after lookup** |
| `output.weight` | **PQ2_0** | 5120 × 248320 | 1 | `lm_head` — folded |
| `output_norm.weight` | F32 | 5120 | 1 | `norm` |
| `attn_norm.weight`, `post_attention_norm.weight` | F32 | 5120 | 64 each | `input_layernorm`, `post_attention_layernorm` |
| **GDN layers (48)** | | | | |
| `attn_qkv.weight` | **PQ2_0** | 5120 × 10240 | 48 | `linear_attn.in_proj_qkv` (q 2048 \| k 2048 \| v 6144) |
| `attn_gate.weight` | **PQ2_0** | 5120 × 6144 | 48 | `linear_attn.in_proj_z` |
| `ssm_out.weight` | **PQ2_0** | 6144 × 5120 | 48 | `linear_attn.out_proj` |
| `ssm_alpha.weight`, `ssm_beta.weight` | **BF16** | 5120 × 48 | 48 each | `in_proj_a`, `in_proj_b` — **not ternary, not rotated** |
| `ssm_conv1d.weight` | F32 | 4 × 10240 | 48 | `conv1d` (depthwise) |
| `ssm_a`, `ssm_dt.bias`, `ssm_norm.weight` | F32 | 48, 48, 128 | 48 each | `A_log`, `dt_bias`, `norm` |
| **Full-attention layers (16)** | | | | |
| `attn_q.weight` | **PQ2_0** | 5120 × 12288 | 16 | `self_attn.q_proj` (query \| gate, interleaved per head) |
| `attn_k.weight`, `attn_v.weight` | **PQ2_0** | 5120 × 1024 | 16 each | `k_proj`, `v_proj` |
| `attn_output.weight` | **PQ2_0** | 6144 × 5120 | 16 | `o_proj` |
| `attn_q_norm.weight`, `attn_k_norm.weight` | F32 | 256 | 16 each | `q_norm`, `k_norm` |

Byte totals: PQ2_0 **7.137 GB**, BF16 0.047 GB, F32 0.011 GB. The two vocab tables are 0.34 GB
each packed (2.54 GB each at fp16 — the reason Bonsai ternarises them). Per-token weight
traffic at decode is everything except the embedding table, **~6.8 GB**; at the M4 Pro's
~273 GB/s that is the ~40 tok/s ceiling.

Every PQ2_0 tensor is in `prism.hadamard.weight_names` (401 entries) except `token_embd.weight`,
which is the sole entry of `prism.hadamard.inverse_weight_names`. There is **no ternary tensor
that is unrotated**, and no rotated tensor that is not ternary (the MLX loader refuses a float
folded matrix outright, `mlx:runtime.py:254-255`; the GGUF inventory has none).

The whitepaper's Table 2 (§2.2) lists exactly the non-ternary set above: 26,238,464 params,
0.0976 % of the language model.

## 3. The bit layouts

### 3.1 PQ2_0 — one fp16 scale + 32 bytes per 128 weights (34 B, 2.125 bpw)

```c
// lc:ggml/src/ggml-common.h:202-206
#define QK_PQ2_0 128
typedef struct {
    ggml_half d;                   // delta (scale)
    uint8_t qs[QK_PQ2_0 / 4];      // 2 bits per element
} block_pq2_0;
```

Type id **142** (`lc:ggml/include/ggml.h:435`, `gguf-py/gguf/constants.py:5409`); file type
141 (`general.file_type`). Element `j` of the block lives in byte `j/4` at bit offset
`2·(j%4)`; code `q ∈ {0,1,2}` → value `(q − 1)·d` (`lc:ggml/src/ggml-quants.c:494-511`; the
Metal dequant `lc:ggml/src/ggml-metal/kernels/dequantize.h:132-148` reads bits 0/2/4/6 of each
byte in order). The quantiser is `d = max|w|` over the group and `q = round(w/d) + 1`
(`lc:ggml-quants.c:113-146`) — so **`d` is the ternary magnitude itself**, and code 3 never
occurs in a ternary checkpoint.

**Verified on 800,000 real blocks** (the first 20,000 rows of `output.weight`): code histogram
{0: 34.40 M, 1: 33.56 M, 2: 34.44 M}, **no code 3**; scales finite, 0.0043–0.0264 (median 0.015),
none zero; **32.8 % of the head's weights are exact zeros**.

**Word view.** Bytes `qs[4i..4i+3]` read as a little-endian `uint32` put element `16i + j` at bits
`2j` — the same layout the zoo's `pack_tern_u32` produces
(`bitcpm_ternary_metal.py:83-95`: `qp |= c[..., j] << (j*2)`) and its kernels consume
(`bitcpm_ternary_gemm.py:60`: `(packed >> (j*2)) & 3`). Verified: the first word of the first
block decodes identically both ways. So **PQ2_0 → BitCPM `QP[N, K/16]` is a reinterpret of the
`qs` bytes, row by row**; only the scales move out to a separate `D[N, K/128]` fp16 array.
The MLX pack's `codec.py:29` does exactly this reinterpret (`data[:, 2:].view("<u4")`).

### 3.2 PTQ1_0 — 24 + 2 bytes of base-3 trits + fp16 scale (28 B, 1.75 bpw)

```c
// lc:ggml/src/ggml-common.h:214-219
#define QK_PTQ1_0 128
typedef struct {
    uint8_t qs[(QK_PTQ1_0 - 4*QK_PTQ1_0/64)/5]; // 24 B, 5 trits per byte -> 120 values
    uint8_t qh[QK_PTQ1_0/64];                   //  2 B, 4 trits per byte ->   8 values
    ggml_half d;                                // scale (LAST in the block, unlike PQ2_0)
} block_ptq1_0;
```

Type id **143**. Same scale semantics and same trit values as PQ2_0 — the two packs are lossless
transcodes of each other (`mlx:codec.py:1-49` transcodes either to the same words). It is
upstream TQ1_0's trit codec at group 128 instead of 256 (`lc:ggml-common.h:209-213` explains
why: a 256-wide scale would have to drop one of the two g128 scales it straddles).

Element order is **not positional** (`lc:ggml-quants.c:2203-2253`, Metal map
`lc:dequantize.h:191-204`):

| elements | bytes | rule |
|---|---|---|
| 0–79 | `qs[0..15]` | element `16·n + m` = trit `n` of byte `m` (n = 0..4) |
| 80–119 | `qs[16..23]` | element `80 + 8·n + m` = trit `n` of byte `16+m` |
| 120–127 | `qh[0..1]` | element `120 + 2·n + h` = trit `n` of byte `qh[h]` (n = 0..3) |

A byte holds `q = ceil(256 · (t₀·81 + t₁·27 + t₂·9 + t₃·3 + t₄) / 243)` (the *first* trit is
most significant); trit `n` is recovered as `((q · 3ⁿ) & 0xFF) · 3 >> 8`. The Metal side either
walks a 256-entry LUT (`dequantize.h:169-186`) or does it in the float pipe with `floor` chains.

**For the port, PTQ1_0 is a download-size option only**: unpack to the PQ2_0 word layout at load
time (or ship PQ2_0). The whitepaper's own Apple measurements are on PQ2_0; the trit unpack is
arithmetic the bandwidth-bound decode would pay for nothing on a GPU.

### 3.3 MLX affine 2-bit — the reference runtime's container (2.25 bpw)

`mx.quantize(bits=2, group_size=128)` words: `uint32 [N, K/32·… ]` = `[N, K/16]` with element `j`
at bits `2j` (`mlx:codec.py:40-44`, `unpack` at `:52-62`), plus `scales[N, K/128]` and
`biases[N, K/128]` fp16 with **`biases = −scales`** (`codec.py:30, 47-48`), so
`code·s − s = (code − 1)·s`. Same words, same scales; the bias is dead weight
(README: 7.67 GB vs PQ2_0's 7.21). The pack's `config.json` records
`quantization: {bits: 2, group_size: 128, mode: affine}` and a `modules` list of the 402 packed
modules with `block: 1024`, `embedding: true` only for `model.embed_tokens`.

**Loader choice for the zoo:** the MLX `model.safetensors` is the easiest source — HF tensor
names and HF (grouped) head order, words already in kernel layout — at the cost of 8.6 GB
download including a 0.92 GB fp16 vision tower we do not use. The GGUF PQ2_0 (7.21 GB) needs the
tiled→grouped V-head reorder of §6 but is smaller. Either way, the words and scales are used
as-is; nothing is re-quantised.

## 4. The Hadamard rotation

### 4.1 Definition (whitepaper §2.4, `lc:src/llama-model.cpp:1194-1226`)

```
R = (1/√n) · Hₙ · S,   n = 1024
```

`Hₙ` is the Sylvester (natural-order) Walsh–Hadamard matrix — llama.cpp materialises it as
`H[row][col] = (−1)^popcount(row & col) / √n` (`lc:llama-model.cpp:2000-2024`) — and `S` is a
diagonal of ±1 signs. Metadata contract (all in the GGUF `prism.hadamard.*` keys and, verbatim,
in the MLX pack's `hadamard.json`):

| key | value in this model |
|---|---|
| `version` | 1 |
| `block_size` | **1024** |
| `transform` | `normalized-sylvester-walsh-hadamard` (the only accepted value) |
| `axis` | `input-last-dimension` (rotation acts on K, never on N) |
| `sign_mode` | `explicit`; `sign_widths = [5120, 6144, 17408]`, `sign_values` = 28672 ±1 entries, concatenated in that order |
| `weight_names` | 401 folded matrices; `inverse_weight_names = [token_embd.weight]` |
| `gdn_v_grouped` | `true` (see §6) |

**One sign vector per K-width, shared by every layer** — the loader keys signs by
`weight->ne[0]` (`lc:llama-model.cpp:2033-2036`). Width 5120 covers everything reading the
residual stream plus the embedding inverse; 6144 covers `o_proj` and GDN `out_proj`; 17408 covers
`down_proj`. Roughly half the signs are −1 (2639/5120, 3112/6144, 8753/17408).

**Blockwise means block-diagonal:** a K-wide activation is reshaped to `[K/1024, 1024]` and each
row gets its own H₁₀₂₄ (`mlx:runtime.py:23`: `x.reshape(-1, block)`; llama.cpp's
`llama_mul_mat_hadamard` reshapes to `[n, nelements/n]` and matmuls with the `n×n` rotation,
`lc:src/llama-impl.h:57-75`). K = 5120 → 5 blocks, 6144 → 6, **17408 → 17 blocks**. There is no
17408-point transform and nothing to factor; the sign vector is the only thing that is K-wide.

### 4.2 Where it sits: the activation side of every folded matmul

Stored weight: `W' = W · Rᵀ` (folded offline; ternary quantisation happens *after* the fold).
At runtime, for each folded linear: `y = W' · (R x)`, i.e.

```
x' = x ⊙ s_K                       # sign flip, elementwise over the full K
x' = FWHT₁₀₂₄(x' per 1024-block) / 32
y  = ternary_matmul(W', x')
```

`lc:src/llama-graph.cpp:1506-1536` (`build_lora_mm`: permute-if-GDN → `ggml_mul` by signs →
`llama_mul_mat_hadamard` → `ggml_mul_mat`); `mlx:runtime.py:16-28, 60-70` (`fwht`: cast to
fp32, `x * signs`, `mx.hadamard_transform(scale=1/√block)`, cast back; then
`quantized_matmul`). Sign first, then transform, in both runtimes.

**Memoised per activation site** (`lc:llama-graph.cpp:1516-1519`, keyed on `(input tensor,
rotation)`): q/k/v read the same transformed `x'`; gate/up likewise; GDN's qkv and z-gate
likewise. Per layer that is **4 transform sites** (normed residual → {q,k,v} or {qkv,z};
attention/GDN output → o/out_proj; normed residual → {gate,up}; act → down), plus the head and the
embedding inverse: **258 FWHT dispatches per token** at 64 layers. The whitepaper (A.2, "Rotation
overhead") names this the largest remaining non-matmul cost of a decode step on Metal. Note that
`in_proj_a`/`in_proj_b` are *not* folded, so they read the **untransformed** normed `x` — the
graph must keep both `x` and `x'` alive at that site.

**Embedding inverse** (`lc:llama-graph.cpp:2383-2395`, `mlx:runtime.py:43-59`): the table stores
`R·e` per row; after the gather, `e = Rᵀ z = S · (H z / 32)` — **transform first, then sign
flip** (the reverse order; both factors are their own inverse).

### 4.3 The FWHT kernel as PrismML runs it on Metal

- `lc:ggml/src/ggml-metal/kernels/misc.metal:377-436` (`kernel_fwht<N>`, one row per
  simdgroup, N ≤ 256) and `:441-513` (`kernel_fwht_tg<N, NT=256>`, one row per **256-thread
  threadgroup**, used for N ≥ 512 — `ggml-metal-impl.h:1220-1223`). **N = 1024 → the
  threadgroup variant**: each thread holds 4 values; butterflies of stride < 32 use
  `simd_shuffle_xor`, strides 32..128 go through `threadgroup float shmem[1024]`, strides ≥ 256
  stay in registers.
- Scale `1/√N` is applied **on load**, together with the sign (`:399-411`, `:464-474`:
  `reg[i] = float(src[i]) * s * scale`). Accumulation is fp32; input f32 or f16; **output always
  f32** (`ggml-metal-ops.cpp:2467`).
- The sign multiply is fused into the transform's load by pattern-matching `MUL → RESHAPE →
  MUL_MAT(hint=SRC0_IS_HADAMARD)` (`ggml-metal-ops.cpp:2503-2556`); `n_blk = K/N` selects which
  1024-slice of the K-wide sign vector a row uses (`:2463`, `misc.metal:400`).
- Supported N: 64…8192 (`ggml-metal-device.h:152-154`). On the CPU the same op is a plain
  in-place butterfly (`lc:ggml/src/ggml-cpu/ops.cpp:11890-11964`), so the graph is
  backend-portable: a `MUL_MAT` with an explicit 1024×1024 `H` tensor whose Metal lowering is the
  FWHT.

### 4.4 Numerics to match for a token-identical gate

The gate target is the MLX bundled runtime (same words, same scales, same signs). Its rounding
points, per folded linear: activation fp16 → **fp32** sign+FWHT → **fp16** (`runtime.py:28`)
→ `quantized_matmul` (fp32 accumulate, fp16 out). The zoo's kernels already take a fp16 `x`,
accumulate in fp32, and emit fp16 — so a fp32 FWHT kernel that writes fp16 reproduces the MLX
chain at every cast. llama.cpp keeps the transformed activation in **f32** into the matvec,
which is a different rounding — another reason the MLX runtime, not llama.cpp, is the yardstick.

## 5. The ternary matvec / GEMM on the fork (for calibration, not for copying)

- Decode matvec `kernel_mul_mv_pq2_0_f32` (`lc:mul_mv.metal:1063-1136`): a simdgroup owns
  `N_R0_PQ2_0 = 8` output rows; 32 lanes split a 128-block into 8 sub-blocks of 16 codes;
  the 16 activations are pre-combined into base-4 "collapse coefficients" so each packed byte
  costs three `floor`s and four FMAs, no integer ops (`:1034-1058`). Per-block scale applied after
  the 16-wide partial, `−sumy` handles the `−1` offset. Same structure as the zoo's
  `bitcpm_ternary_metal.py` matvec (lane = 16 codes, scale per lane, `simd_sum`), which is why the
  zoo kernel is the right starting point.
- Prefill GEMM is the stock `kernel_mul_mm` template with a `dequantize_pq2_0` hook
  (`lc:mul_mm.metal:746`), i.e. dequant-to-half tiles + simdgroup matmul — the same design as
  the overlay's `bitcpm_ternary_gemm.py`.
- Reported Apple numbers (README / whitepaper Table 5, PQ2_0, `llama-bench`): **M4 Pro 18.0
  tg128 / 125 pp512** (pre-rotation build, same weights), M5 Pro 28.1 / 387, M5 Max 47 / 765;
  the M5 Pro decode streams ~204 GB/s of weights.

## 6. Layout conventions the loader must honour

These are the three places a naive "map GGUF names to `qwen3_5.py`" port silently produces
wrong logits.

1. **RMSNorm gain offset.** The GGUF converter stores `w + 1` for every `*norm.weight` except
   `linear_attn.norm.weight` (`lc:conversion/qwen.py:394-395`), and the MLX pack inherits that.
   `zoo:qwen3_5.py:125, 418` use `RMSNormPlusOne` (adds 1 at runtime). **Subtract 1** from
   `attn_norm`, `post_attention_norm`, `output_norm`, `attn_q_norm`, `attn_k_norm` when loading
   from either pack — or swap those five to a plain `RMSNorm`. `ssm_norm` (`RMSNormGated`,
   `:315`) is stored raw.
2. **GDN V-head order (`gdn_v_grouped = true`).** HF/zoo order is *grouped by K head*
   (`[k0: v0 v1 v2][k1: v0 v1 v2]…`, `zoo:qwen3_5.py:380-384` expands q/k with
   `expand+reshape` over that order). llama.cpp's converter permutes V to *tiled* order
   (`[all k's v0][all k's v1][all k's v2]`, `lc:conversion/qwen.py:448-470, 572-616`) for the V
   rows of `attn_qkv`, all rows of `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_a`, `ssm_dt.bias`,
   and the V channels of `ssm_conv1d`. **But `ssm_out.weight`'s input columns are left in
   grouped order** (`:617-630`) because a column permutation cannot be pushed through the fold;
   at runtime llama.cpp permutes the *activation* tiled→grouped before the transform
   (`lc:llama-graph.cpp:1521-1528`, `llama-model.cpp:2080-2086`). Consequences:
   - loading from **GGUF**: apply the inverse (tiled→grouped) row permutation to the seven
     tensors above — the MLX loader's `vperm`/`reorder` (`mlx:runtime.py:171-192`) is the exact
     recipe — and leave `ssm_out` alone;
   - loading from the **MLX pack**: nothing to do (`PACK-RUNTIME.md`: "GDN activations are
     already grouped … do not permute them again");
   - in the zoo graph: **no activation permutation** — the torch order is already grouped, so
     the `out_proj` transform is a plain sign+FWHT on the `[b, s, 6144]` output.
3. **Query|gate interleave.** `attn_q` is `[24 heads × (256 query | 256 gate)]` per head
   (`lc:src/models/qwen35.cpp:298-322` views with stride `2·head_dim`), identical to
   `zoo:qwen3_5.py:121, 148-149` (`view(b,s,H,2D).chunk(2)`). No permutation.

Smaller ones: `ssm_a` is stored as `−exp(A_log)` (`lc:conversion/qwen.py:388-389`; the zoo keeps
`A_log`, so `A_log = log(−ssm_a)`, `mlx:runtime.py:248-251`); `ssm_dt.bias` ↔ `dt_bias`;
`ssm_conv1d` is squeezed to `[10240, 4]` (zoo wants `[10240, 1, 4]`). `ssm_alpha`/`ssm_beta` are
BF16 in the GGUF → cast to fp16 for the zoo's `in_proj_a`/`in_proj_b` (`zoo:qwen3_5.py:308-309`).

## 7. Diff against the zoo's `qwen3_5.py`

What stays: the whole hybrid graph. Config fields, layer interleave, the GDN loop-free step /
chunk / Metal scan, partial mRoPE (text = plain RoPE on 64 dims), the `q|gate` split, the GVA
expand, the stateful export spec and the pipelined-engine states — all unchanged, and the
27B shapes are already exercised by the `qwen3.6-27b` export (`export_qwen3_5_decode_pipelined.py
--hf-id Qwen/Qwen3.6-27B`).

What changes, module by module:

| zoo module | today | Bonsai |
|---|---|---|
| `q_proj, k_proj, v_proj, o_proj` (×16) | `nn.Linear` fp16/int8 | `TernaryHadamardLinear`: signs(K) → FWHT₁₀₂₄ → PQ2_0 matvec/GEMM, scale/128 |
| `in_proj_qkv, in_proj_z, out_proj` (×48) | `nn.Linear` | same; `out_proj` uses the 6144-wide signs |
| `mlp.gate_proj, up_proj, down_proj` (×64) | `nn.Linear` | same; `down_proj` uses the 17408-wide signs (17 blocks) |
| `in_proj_a, in_proj_b` (×48) | `nn.Linear` | **unchanged** fp16 `nn.Linear` reading the *untransformed* normed x |
| `lm_head` | fp16 / int8 `nn.Linear` (`:564`) | ternary + Hadamard, N = 248320 (= 3880 × 64) |
| `embed_tokens` | `nn.Embedding` (`:468`) | packed rows; gather → dequant → FWHT → signs |
| 5 RMSNorm kinds | `RMSNormPlusOne` | gains loaded as `w − 1` (or plain RMSNorm) |
| `conv1d, A_log, dt_bias, norm` | as is | as is (fp32/fp16) |
| config shim | `Qwen3_5VLConfig` from HF `config.json` | build `Qwen3_5Config` from the pack's `text_config` (MLX) or the `qwen35.*` GGUF keys (`mlx:runtime.py:116-140` is the mapping) |

Shared transforms: implement the transform as a module on the *activation* (one per site) and
let the linears take `x'` — that is how both reference runtimes get 4 FWHTs per layer instead
of 10. The `in_proj_a/b` split means the GDN site keeps `x` too.

Kernel constraints, checked against every Bonsai shape:

| linear | K | N | K % 512 | N % 64 |
|---|---:|---:|---|---|
| q / k / v / qkv / z / gate / up / head / embed | 5120 | 12288 / 1024 / 10240 / 6144 / 17408 / 248320 | ✓ (10) | ✓ |
| o_proj / out_proj | 6144 | 5120 | ✓ (12) | ✓ |
| down_proj | 17408 | 5120 | ✓ (34) | ✓ |

## 8. What this means for steps 2–3 (kernels and export)

- **Scale block 256 → 128 in both zoo kernels.** `bitcpm_ternary_metal.py:31` (`_SCALE_BLK`),
  `:48` (`g = k0 >> 8`), and `bitcpm_ternary_gemm.py:58` (`D[k0 >> 8, …]`) become `>> 7` with
  `D[N, K/128]`. The lane invariant (16 codes inside one scale group) and the GEMM invariant
  (`BK = 64` never straddles a group) both still hold at 128. `ternary_from_dequant` is not needed:
  words and scales come straight from the pack (§3.1). The PQ2_0 file's `d` is the exact ternary
  magnitude, so the `torch_defn` reference is `(code − 1) · d` with no rounding ambiguity.
- **Standalone Hadamard kernel** (the plan's step 2): input `[S, K]` fp16, sign vector `[K]`
  fp16/fp32, output `[S, K]` fp16; grid = `S × K/1024` rows of 1024, one 256-thread threadgroup
  per row as in `kernel_fwht_tg<1024,256>` (4 values/thread, shuffles below 32, 7 `threadgroup`
  passes up to 256, 2 register passes above). Fold `× s × (1/32)` into the load, accumulate fp32,
  write half. `torch_defn`: `((x.float() * s).view(-1, 1024) @ H₁₀₂₄ / 32).view_as(x).half()`,
  with `H` from `hadamard(1024)` in natural (Sylvester) order — the `(−1)^popcount(i&j)` matrix,
  not the sequency-ordered one. Gate it alone against the MLX `fwht` with the pack's real sign
  vectors. Rows are independent, so the same kernel serves S = 1 and the chunked prefill.
- **The embedding** is 0.34 GB packed but 2.5 GB if dequantised to fp16 for an in-graph
  `nn.Embedding`. The pack's answer is dequant-after-gather (`mlx:runtime.py:43-59`), which is
  one 5120-wide row per token: cheap on the CPU front-end in Swift (dequant + FWHT + signs on
  one row), or a tiny gather kernel. Decide at step 3; on the phone the fp16 table is not an
  option.
- **Bytes per token** for the bandwidth model: 63 × (per-layer PQ2_0) + head ≈ 6.8 GB; the
  fp16 in_proj_a/b, conv, norms add ~0.05 GB. The whitepaper's M4 Pro 18 tok/s (llama.cpp,
  pre-rotation) is the baseline to beat; ~40 tok/s is the ceiling.

### 8.1 Status: the Hadamard kernel exists and is gated (2026-09-17)

`bonsai_hadamard_metal.py` (overlay, next to `bitcpm_ternary_gemm.py`) implements the design
above as one `TorchMetalKernel` (`bonsai_fwht1024`): 256 threads per 1024-row, sign and 1/32 on
load, fp32 butterflies (5 simd-shuffle stages, 3 threadgroup-memory stages, 2 register stages),
one fp16 rounding at the store. `BonsaiHadamard(signs, kernel)` wraps it per K width.
`_smoke/bonsai/gate_hadamard_kernel.py` exports a one-op graph per (K, S), runs it on the GPU
through `coreai.runtime`, and checks it against the exact fp64 transform with the model's real
sign vectors (`_smoke/bonsai/hadamard_signs.json`, lifted from the GGUF header):

| K | S | rows | max err | fp16 flips vs exact | round-trip | ms/call (M4 Pro) |
|---:|---:|---:|---:|---:|---:|---:|
| 5120 / 6144 / 17408 | 1 | 5 / 6 / 17 | 1.95e-3 | 0 / 1 / 3 | ≤ 0.37 ulp | 0.29–0.37 |
| 5120 / 6144 / 17408 | 64 | 320 / 384 / 1088 | 1.95e-3 | 83 / 83 / 232 (0.02 %) | ≤ 0.41 ulp | 0.38–0.46 |
| 5120 / 6144 / 17408 | 256 | 1280 / 1536 / 4352 | 1.95e-3 | 280 / 376 / 959 (0.02 %) | ≤ 0.44 ulp | 0.50–0.69 |

Reading the numbers: the max error is exactly half an fp16 ulp at the output magnitude, i.e. the
single output rounding, on every configuration and run-to-run bit-identical. The "flips" are
elements whose fp32 result lands on the other side of a rounding boundary from the fp64 value —
the same 0.02 % any fp32 implementation (MLX's included) produces, and the kernel's own fp32
matmul reference disagrees with it at the same rate. Near zero the residual is ~1e-7 absolute
(fp32 cancellation noise), far below anything a ternary matmul will see. The ms/call figures are
dominated by the runtime's per-invocation floor (~0.3 ms), not the kernel; the per-token cost in
the real graph is a per-node dispatch question for step 3.

**iPhone AOT: passes.** `coreai-build compile <bundle> --platform iOS --preferred-compute gpu
--architecture h18p --min-deployment-version 27.0` → EXIT 0, a 56 KB `fwht_k5120_s1.h18p.aimodelc`.
The trap on the way there is worth its own line: `aimodelc`, the binary in Xcode 27 beta's
`usr/bin`, refuses with "Core AI requires the Metal Toolchain" even with the Metal Toolchain
component (27A5237l) installed. It is only a proxy (`IDEMLCompilerCore.MLAssetCompilerProxy`)
that resolves the real compiler through the DVT toolchain registry's *default* toolchain, which
from a CLI process does not see the cryptex-mounted Metal toolchain. Do not use it. The real
compiler is `Metal.xctoolchain/usr/bin/coreai-build`; once the component is installed
`xcrun coreai-build` resolves it (before the install `xcrun` reports "unable to find utility"),
and the explicit path works regardless:

```bash
"$(dirname "$(xcrun -f metal)")/coreai-build" compile <bundle.aimodel> --platform iOS \
    --preferred-compute gpu --architecture h18p --min-deployment-version 27.0 --output <dir>
```

### 8.2 Status: the decoder runs on Core AI (2026-09-17, step 3)

`bonsai2.py` (overlay) builds the zoo's `Qwen3_5ForCausalLMStateful` from the PQ2_0 GGUF with
the loader conventions of §6, swaps the 401 folded linears for `TernaryLinear128` (words used
as-is, scale group 128), one `HadamardSite` per activation site (memoised on tensor identity,
so q/k/v share one transform), a `PackedEmbedding` whose kernel fuses gather + dequant + the
inverse transform, and the untied ternary head. `conversion/export_bonsai2_27b_decode_pipelined.py`
exports the S=1 static-ids decode graph with the four kernels registered (same state contract
as the qwen3.5 decode bundles: keyCache/valueCache + convState/recState).

| stage | result |
|---|---|
| kernels on real layer-0 weights (`_smoke/bonsai/gate_kernels_real.py`) | embed bit-exact; matvec K=5120/17408 and GEMM S=64 within fp16 rounding (rel 2–4e-4) |
| 4-layer bundle, GPU vs torch reference, 39 teacher-forced steps | argmax 39/39, max\|Δlogit\| 0.016 on logits ≈ 15 |
| full 64-layer export | 159 s trace+convert, 6.7 GB `.aimodel`, 7.5 GB packed buffers in RAM |
| AOT h16s (M4 Pro), `coreai-build --platform macOS --preferred-compute gpu --expect-frequent-reshapes` | EXIT 0 |
| full model, GPU, "The capital of France is" (chat template, thinking on) | `User asks: "…". Need answer concise. Final: Paris.</think>\n\nParis<\|im_end\|>` |
| decode speed, Python S=1 loop through `coreai.runtime`, M4 Pro 48 GB | **15.9 tok/s** (prefill as S=1 walk 14.0) |

Two Mac-side traps, both already in the zoo's notes and both confirmed here: the JIT
`.aimodel` load with GPU preferred still tries ANE placement, `ANECCompile() FAILED`, and the
command buffer then dies (`MTL4CommandQueueErrorDomain error 1`, every token 0) — route through
the AOT compile and load the `.aimodelc` with default options; and a truncated build must
include a full-attention layer (index 3, 7, …) or the KV caches are never written and the
converter refuses the four-state contract.

PrismML's bundled MLX loader refuses this GGUF as shipped (`Unsupported auxiliary type BF16` —
the a/b projections are BF16); the token gate runs a scratch copy that dequantizes BF16 to
float32, a whitelist change only.

### 8.3 Chunked prefill: the two-entrypoint bundle

`--chunk 64` adds a `prefill` function traced at static S=64 next to `main` (S=1), one
`AIProgram`, weights shared, following `ternary-chunked-prefill.md`. Per entrypoint the model's
trace-time switches decide the path: `TernaryLinear128` picks the tiled GEMM when the traced
query length is not 1; `HadamardSite` and `PackedEmbedding` are row-parallel already; the GDN
layers flip from the loop-free single step to the fp32 Metal chunk scan (`set_gdn_mode`). Both
entrypoints drop the SDPA composite (static S>1 trips its auto-Dim min=2, the verify-export
precedent) so the shared model carries one externalization.

**Trap found here (worth a line in the GDN kernel's own notes):** the zoo's `MetalGDNChunk`
transposes `g`/`beta` to `[heads, S]` and calls `.contiguous()`, but Core AI's optimizer elides
that copy and the custom-kernel op then asserts at load
(`GPUCustomMetalKernelOps.mm: [tensor.strides extentAtDimensionIndex:0] (48) should be 1`).
`bonsai2.BonsaiGDNChunk` builds the same MSL with `G[hh, t]` / `BETA[hh, t]` and hands the
tensors over in the `[S, heads]` layout the projections produce, so nothing needs a copy.
Kernel inputs must be contiguous *as produced*; do not rely on `.contiguous()` surviving.

Gate on the 4-layer bundle (`--check-only --chunk 64 --self-check`): the prefill chunk vs the
same bundle's S=1 walk agrees on the argmax at **125/125** prompt positions (64 in the chunk,
61 S=1 steps on the chunk-primed state), worst max|Δlogit| 0.016; chunk 554 tok/s vs 119 tok/s
for the walk at 4 layers.

**Full model, two entrypoints** (`exports/bonsai2_27b_decode_pq2_0_pf64`, 6.7 GB `.aimodel`,
420 s export, h16s AOT EXIT 0), 125-token chat prompt, 40 new tokens, M4 Pro:

| | result |
|---|---|
| prefill, one S=64 chunk | **61.0 tok/s** (S=1 walk of the same bundle: 15.0) |
| decode after the chunk | 14.6 tok/s |
| chunk vs the bundle's own S=1 walk, 125 prompt positions | argmax 125/125, worst max\|Δlogit\| 0.114 |
| vs PrismML's MLX runtime, teacher-forced | 161/164 positions; **all 40 generated tokens identical** (3 misses at system-prompt ties) |

Token identity against the reference runtime now holds on three prompts (24 + 48 + 40 new
tokens). The chunk is 4.1× the walk at C=64 (BitCPM's was 5.9×): with 64 layers of four FWHT
sites each plus the GDN scan, per-node dispatch is the suspect, which is step 4's perf round.

### 8.4 Status: the Swift host runs it (2026-09-18, step 4)

`bonsai-swift` (RahulRachuri/bonsai-swift, private for now) loads the two-entrypoint bundle on the
low-level `CoreAI` framework: the AOT `aot_mac/<name>.<arch>.aimodelc` with
`SpecializationOptions.default` (arch from `AIModel.deviceArchitectureName`), `main` and `prefill`
from one `AIModel`, the four fp16 states allocated once (KV sequence axis at 2048, the export's
traced minimum) and passed as `states:` on every run, `position_ids` = `[0, total)` each call.
Whole 64-token chunks go through `prefill`, the remainder and always the last prompt token through
`main`. The tokenizer is swift-transformers over the bundle's `tokenizer/`; its chat template
renders the HF reference ids exactly on all three prompts. First load 28 s (runtime cache staging),
then 1.3 s.

Not Apple's pipelined engine, deliberately: the gate needs per-position logits, and at 27B the
host overhead the engine hides is a few percent of a 65 ms token.

| gate (teacher-forced vs `mlx_reference.py` fixtures) | positions | generated |
|---|---:|---|
| France, 24 new, prompt walked | 77/80 | 24/24 identical |
| train time, 48 new, one chunk + walk | 127/130 | 48/48 identical |
| lighthouse 125 tok, 40 new, one chunk + walk | 161/164 | 40/40 identical |
| same, prompt walked | 161/164 | 40/40 identical |

The misses are steps 2, 21 and 28 in every run, all in the chat template's system prompt, reference
margins 0.002 / 0.024 / 0.005. (The Python gate of the decode-only bundle reported 78/80 on the
France prompt; the two-entrypoint compile lands step 2's 0.002-margin tie on the other side. Tie
band, not a regression.)

Speed, M4 Pro, release build: decode **15.2 tok/s**, chunk **59 tok/s**, walk 15.2 — the same as
the Python loop on this bundle, so the loop is not where the time goes. Naive ceiling 7.2 GB/token
at ~273 GB/s ≈ 38 tok/s; decode is at 40% of it. Perf rounds start from here.

**Swift 6.4 trap (Xcode 27 beta 27A5237l):** `InferenceFunction.MutableViews.insert(&self.keyCache, …)`
on a class stored property fails to compile — "lifetime-dependent variable 'states' escapes its
scope … depends on this scoped access to variable 'keyCache'" — although Apple's
`CoreAISequentialEngine` is written exactly that way. The scoped access to a class property ends at
the statement; a view from a local `var` lives to the end of the scope (parakeet-swift's decode loop
found the same). Fix in `BonsaiEngine.run`: `swap` each state/output NDArray out of its slot into a
local, insert the locals, run, `defer` the swap back. Handles move, bytes do not.

One more contract note for iOS: `main`'s `position_ids` Dim has `min=2` (§8.3) but step 0 of a walk
feeds length 1. The Mac accepts it; the device may not (`ternary-chunked-prefill.md` §6), so a
device host should prime position 0 through a chunk or pad — untested.

### 8.5 Perf round 1: where a token goes, and the matvec wall (2026-09-18)

Metal System Trace of `bonsai-swift chat` on the M4 Pro (release build, AOT h16s bundle), decode
steady state, **66 ms per token**:

| where | ms/token | how measured |
|---|---:|---|
| host round trip (await run → logits readback → argmax → encode next) | ~4.3 | the one gap per token between command buffers |
| GPU, ~30 command buffers of ~2 ms | ~60 | `metal-gpu-intervals`, 93% busy |
| of which `bonsai_tern_mv128` (401 dispatches) | ~47 | shader profiler durations × dispatch counts |
| of which `bonsai_gdn_chunk_sh` (48 × 45 µs) | ~2.2 | |
| of which ~600 small MPSGraph dispatches (identity copies, reduces, fp16 a/b matvecs, gathers, conv) | ~5 | |

The matvec ran every shape at **130–140 GB/s**: 8.4 MB linears ~55 µs, 13.9 MB ~100, 16.7 MB ~120,
23.7 MB FFN ~165, the 338 MB head 2.4 ms. This chip's practical streaming-read ceiling is **~245 GB/s**
(a plain xor-reduce over 1–4 GB buffers; nominal 273), so the kernel sat at ~54% of it.

A standalone Metal microbench of the exact loop (same layouts, real shapes, 20 dispatches per
command buffer after a warm-up, 8 rotating weight copies so nothing hits the system cache) found
the wall by elimination: a loads-only twin of the same access pattern streams at **248 GB/s**, and
fewer integer ops per code, `extract_bits`, folding the −1 into a per-lane activation sum, more rows
per simdgroup (8, 16), `half4` activation loads, explicit prefetch and `uint2` weight loads all stayed
at **132 ± 3**. The one instruction every compute variant kept, once per code, was the
**`int → float` conversion** — a reduced-rate op on this GPU, and at 16 conversions per 4-byte word
it caps the kernel at exactly the observed rate. Vector types (`uint4`, `half4`) do not help because
Apple GPU lanes are scalar.

**Fix (landed in `_MV_SRC`, same launch geometry R=4/SGY=8/16 codes per lane):** convert four codes
per instruction through the unorm unpack unit —
`unpack_unorm4x8_to_float((w >> 2j) & 0x03030303)` reads bytes holding codes j, j+4, j+8, j+12 as
c/255 — and fold the ternary map's −1 into one subtraction, `sum(x·(c−1)) = 255·sum(x·c/255) − sum(x)`.
fp32 accumulation unchanged.

| shape | before | after |
|---|---:|---:|
| K=5120 N=17408 (FFN gate/up) | 132 GB/s | 242 |
| K=17408 N=5120 (FFN down) | 132 | 246 |
| K=5120 N=10240 (GDN qkv) | 133 | 232 |
| K=5120 N=6144 (GDN gate / attn o) | 124 | 231 |

rel err vs the fp32 reference unchanged (4e-4, the fp16 output rounding); `gate_kernels_real.py`
ALL OK on real layer-0 weights (matvec rel 3.9e-4 / 1.8e-4). A mantissa bit-trick
(`as_type<float>(0x3F800000 | c<<22)` = 1 + c/2) reaches the same speed and is the fallback if a
runtime ever lacks the unpack intrinsic. The GEMM converts each code once per K-step per threadgroup
and reuses it over 64 rows, so conversion is not its bottleneck; unchanged.

**Full model, re-exported with the fix** (`exports/bonsai2_27b_decode_pq2_0_pf64`, 423 s export, h16s
compile 15 s), `bonsai-swift` on the M4 Pro: decode **22.5 tok/s** (was 15.2, +48%), S=1 walk 22.8,
S=64 chunk 59.4 (unchanged, as predicted). All three MLX fixtures still token-identical (77/80, 127/130,
161/164 with the same three system-prompt ties). Next levers after
that, in order of size: the ~600 small dispatches and their gaps (~5–8 ms), the 4.3 ms host gap
(GPU-side argmax, or encode-ahead on a compute stream), the GDN scan.

Two microbench traps: a short kernel timed one dispatch per command buffer runs on a cold GPU clock
(8 MB shapes read 5× slow, the 338 MB head looked fine); and a 23.7 MB weight buffer re-read every
rep sits in the system cache and reports "270 GB/s" — rotate copies.

### 8.6 Perf round 2: the host gap and the prompt tail (2026-09-18)

After round 1 the token is 46 ms: 42 ms of GPU across ~29 command buffers and **3.9 ms inside the
runtime's `run()` before the GPU starts** (host-side encode + completion; the host's own argmax over
248k fp16 is 0.3 ms). The GPU idles for that gap every token, and a synchronous loop cannot hide it:
step N+1's `input_ids` is step N's argmax.

**Export change:** `main` now emits its greedy choice as a second output, `next_token` (int32 `[1,1]`,
`logits.argmax(-1)`, `coreai.argmax` lowers fine), behind a trace-time switch like the GDN mode. The
switch must be flipped *inside the export closure*: `TorchConverter` traces lazily at `to_coreai()`,
after the entry loop, so anything set in the loop body is the last entry's state. That is also why
the gated bundles' `main` was traced with the fp32 chunk-scan kernel at S=1 (the loop left "chunk"
mode on); it is kept that way on purpose — the proven path — and the loop-free torch step remains the
untested alternative. `--chunks 16` adds `prefill16`: one more tiled GEMM instance (BM is baked into
the MSL; `TernaryLinear128` now holds a dict keyed by query length), the GDN scan kernel already
takes any S ≤ its `chunk_max` (output sliced `[:, :S]`). Metadata: `prefill_chunks: [64, 16]`,
`next_token_output`.

**Host change (`bonsai-swift`):** `generateStreamed` encodes `main` onto a `ComputeStream` with
`encode(inputs:states:to:)`, the four states as `AsyncMutableValue`s, and hands step N's `next_token`
`AsyncValue` straight to step N+1's `input_ids`; the host reads each token one step late for
streaming and the stop check, so two steps stay in flight and the encode of N+1 overlaps the GPU on
N. On a stop one extra step is already encoded (its position is discarded). Prefill feeds the prompt
greedily, largest chunk that leaves the last token for `main`: 125 tokens = 64 + 16 + 16 + 16 + 13
walked instead of 64 + 61 walked.

4-layer dev bundle: prefill vs walk **125/125** (worst |Δ| 0.016), streamed tokens **identical** to
the sync loop, streamed decode 190 vs 155 tok/s (the host gap is a larger share of a 4-layer token).

**Full model** (re-exported, `exports/bonsai2_27b_decode_pq2_0_pf64`, 14 GB with the h16s compile):
all three MLX fixtures token-identical as before; prefill16 vs walk **125/125** on the 125-token
prompt (worst |Δ| 0.055); 114-token bench prompt **48 tok/s end to end (was 26)**, i.e. ~2 s less
to the first token; decode unchanged at 22.5.

**Negative result — the streamed loop gains nothing on the full model.** `encode(to:)` returns only
when the previous step's GPU work is done: 46 ms per call, the token await then 0.02 ms. Tried,
all identical: tracked `AsyncMutableValue` states vs untracked `unsafeBuffer` Metal buffers, one vs
two loaded instances of `main`, one vs two `ComputeStream`s, chained `next_token` input vs a fresh
host NDArray. So the runtime paces its encode by the GPU for a graph this size (scratch recycling
per command buffer is the likely mechanism: within a token the ~29 command buffers run gap-free,
the 3.9 ms sits before the first one), and the per-step setup cannot be hidden from the host.
The `next_token` output stays (harmless, and it is the right contract for a future runtime);
`chat --stream` stays as a validated option; the sync loop stays the default.

The per-step 3.9 ms is most likely the dynamic `position_ids` length: every step is a new shape
(`--expect-frequent-reshapes`), and the runtime re-resolves the graph for it. A static-shape
decode contract (fixed-length position input plus an in-graph length mask, the zoo's ANE-style
"chunked static" pattern) would remove it, at the cost of an export redesign — a round of its own.

Remaining budget of a 46 ms token: ~28 ms matvec (near bandwidth), ~9 ms inter-dispatch gaps
across ~900 small dispatches (identity copies 300/step, reduces 120, fp16 a/b matvecs 90, the four
FWHT sites × 64 layers, the scan), ~4 ms runtime setup, ~2 ms GDN scan. Next lever: fewer
dispatches — fold the FWHT into the matvec kernel (256 dispatches + intermediates per token), find
what materialises the 300 identity copies.

### 8.7 Perf round 3: fewer kernel calls, and what the runtime really does per step (2026-09-18)

**Where the earlier picture was wrong.** Three probes, each a few minutes on the 4-layer dev
bundle, killed the two levers round 2 had queued:

- *The dynamic position input costs nothing.* `_smoke/bonsai/probe_static_main.py` exports `main`
  twice, as the bundle does it and with every input static (position_ids `[1, 65]`, KV dim fixed),
  and times S=1 steps through `coreai.runtime`: 4.89 vs 4.88 ms. Without `--expect-frequent-reshapes`
  the same bundle decodes at 7 tok/s (a recompile per new shape); with it the runtime re-runs shape
  inference per step, which is what the Time Profiler shows, but that work is not on the critical path.
  The static-shape decode contract is dead.
- *Per-op cost is small on the GPU side.* `probe_op_overhead.py` times chains of N ops and takes the
  slope: a custom kernel call costs ~10 µs all-in (dispatch + the converter's copy-in/copy-out), a
  small graph reduce+multiply ~29 µs, an elementwise op ~9 µs. Fusing the norm/residual/SiLU chains
  into the transform kernels (below) removed ~17% of the graph's ops and moved the full model by 1 ms.
- *Copies between chained kernels are not elided.* A two-kernel chain keeps all of them
  (`copy_with_constraints` on every input, constants included; `copy_discarding_constraints` on every
  result), so every kernel boundary is a graph island.

**What the step really is.** One Metal System Trace with the Time Profiler attached: the GPU is busy
88% of a 43 ms step across ~15 command buffers of ~2.5 ms each, back to back, then a **3.7 ms gap**
before the last one (the head matvec, 1.3 ms). In that gap the decode thread is inside the MPS
runtime — `MPSGraphDelegateKernel.inferValue → MPSRuntime::evaluateOps → GPURegionCallOpHandler::
encodeOp` — i.e. the runtime is still *encoding the step* (one MPSGraph region executable per
graph island, one custom-kernel encode per call) while the GPU has run out of work. The CPU encode
of a step is nearly as long as the GPU work, so the decode is now co-limited by it, and every GPU
saving beyond that tail is hidden until the encode shrinks. The currency of this round is therefore
**kernel calls and graph islands per step**, not GPU microseconds.

**Fusions (all S=1, trace-time mode `fused`, `--gdn-main fused` is the default; S>1 keeps the round-2
chunk path):**

1. `bonsai_gdn_step` — the whole GDN block between the projections in one dispatch: conv step +
   SiLU, the GVA head repeat, both l2-norms and the q scale, softplus/sigmoid for decay and beta,
   the fp32 delta step on the fp16 state, the gated RMSNorm, the two new states; plus the dense a/b
   projections (`in_proj_a/b`, 2 × [5120 → 48]) as per-head dot products inside it, reading the
   un-rotated normalized input. Numerics mirror the graph's fp16 materialisation points; gate
   `_smoke/bonsai/gate_gdn_step.py` (torch reference vs the zoo's loop-free step, and MSL vs
   reference on the GPU). Second version: one threadgroup per 8 heads (1024 threads, the state
   re-read in two passes instead of 128 registers per thread) so the group's outputs are one
   Hadamard block and the out_proj site's signed FWHT/32 is the kernel's epilogue; in situ 44–51 µs
   vs 33 for the one-head version, in exchange for one fewer kernel call per linear layer.
2. `bonsai_add_norm_fwht` / `bonsai_norm_fwht` — residual add, RMSNormPlusOne and the site transform
   in one dispatch, returning the new residual, the un-rotated normalized vector (for the a/b
   projections) and the rotated one; the layer passes the residual stream *unadded* as (h, r) so the
   add lands in the next site's kernel, and the final norm + head site is the same kernel.
   `bonsai_swiglu_fwht` does up ⊙ silu(gate) + the h_down transform. The FWHT butterfly is the
   round-1 kernel's; the norm's mean of squares is reduced redundantly per threadgroup (K/256 loads
   per thread), cheaper than a second dispatch. Gate: `gate_fused_sites.py`.
3. `bonsai_tern_mv128_x2` / `_x3` — the M=1 matvec over 2 or 3 weight sets that share an activation
   (gate+up, qkv+z, q+k+v): the grid spans all rows, each simdgroup picks its set from its row index,
   per-row code identical to the single kernel, results bit-identical (gated). One call instead of
   two or three; 144 fewer calls per step.
4. The site memo trick: the fused kernels return the activation already rotated for a
   `HadamardSite`, and `seed_site(site, y)` sets the site's memo to `(y, y)` so the ternary linears'
   `site.transform(y)` returns `y` untouched. Class swaps in the loader (`BonsaiGDN`,
   `BonsaiAttention`, `BonsaiMLP`, `BonsaiDecoderLayer`, `BonsaiModel`) add the fused forwards;
   nothing about the zoo classes changes and the prefill traces still run their original code.

**Results (M4 Pro, 64 layers, `bench` 114-token prompt, 48 new):**

| bundle | decode ms/step (interleaved A/B, same minute) | tok/s |
|---|---|---|
| round 2 (chunk-scan GDN, unfused sites) | 44.4 | 22.5 |
| fused GDN step + fused sites | 43.0 | 23.3 |
| + paired matvecs (`exports/bonsai2_27b_decode_pq2_0_pf64`, canonical) | 42.4 | 23.6 |

All three MLX fixtures token-identical (78/80, 127/130, 161/164 teacher-forced positions, same
three system-prompt ties as before); streamed loop identical to the sync loop; prefill16 vs walk
124/125 with the one miss a near-tie (position 21: margins 0.016 vs 0.031) and worst |Δ| 0.32 at
position 82 — larger than round 2's 0.055 because `main` no longer shares the chunk kernel's
rounding chain with `prefill`; against the exact torch model the fused `main` is *closer* than the
chunk path (4-layer bundle, 62 steps: worst |Δ| 0.0078 vs 0.0156, argmax 62/62 both).

The 4-layer dev bundle is a poor proxy for these rounds: it stepped 5.52 → 4.90 ms with the GDN
fusion and did not move for any later change, while the full model moved 1.4 ms in total; small
graphs never expose the encode tail. Measure kernel-call changes on the full model only.

Two more probes on the tail (4-layer bundle, interleaved): moving the greedy argmax back to the
host (`--no-next-token`) changes nothing (5.02 vs 5.04 ms), and a head truncated to 8192 rows
(`--head-rows`, debug) saves only the head matvec's own ~1.3 ms of GPU time. So the tail is neither
the `reduce_index` op nor per-step handling of the 318 MB head constant; it is the runtime's
generic per-step encode and stays out of reach from the graph.

The remaining budget of a token is the matvec near bandwidth (~30 ms: 22 MB shapes at ~223 GB/s,
the 318 MB head at 240), ~2 ms in the fused kernels, and whatever the runtime's encode of ~500
kernel calls still fails to overlap. Beyond this the levers are structural: fold the remaining
per-site kernels into the matvec prologue (transform per threadgroup; untested cost), or a runtime
that encodes once.

### 8.8 Review round: what an adversarial read found, and what the fixes changed (2026-09-18)

Two agents after round 3: one searched for decode-speed techniques, one read every kernel, the
loader, the export and the Swift host adversarially. Full reports in
`knowledge/bonsai-review-2026-09-18.md`.

**Speed research, in one line:** nothing online beats the one lever already on the table. The
Hadamard-in-matvec prologue (ceiling ~2 ms, 4.7%) is the only safe experiment left; speculative
decoding is a decode-time algorithm any model can run, but PrismML's own drafter measures ~1.2x on
code and *slower* on chat on Apple Silicon (`Bonsai-demo/SPECULATIVE.md`), and our verify pass would
run on the scalar-FMA GEMM; PTQ1_0 (1.75 bpw) saves 14% of bytes but base-3 trits have no hardware
unpack, so it risks the round-1 conversion wall. Indirect command buffers and residency sets cannot
reach the encode tail: the runtime owns the encoder.

**Review: the kernel math held.** Checked by numpy snippets: `255 * unpack_unorm4x8` is bit-exact for
codes 0..3; the folded `255*sum(x*c/255) - sum(x)` costs nothing (rel err 1.6e-5 vs 1.5e-5 for the
plain fp32 dequant, 0 of 64 outputs flip at the fp16 store); every Bonsai K is a multiple of 512 and
every N of 64 (the 248320-row head has no tail); the Hadamard butterfly index algebra and the natural
Sylvester order; the GDN delta rule against `_gated_delta_step` (state layout, GVA mapping, conv taps,
softplus threshold, barrier discipline); the fused norm/swiglu rounding chains against the graph ops
they replace; the host's position math (no off-by-one in decode or prefill).

**What was wrong, and the fixes (all landed):**

1. *The parity gate never ran the shipping chunk plan.* `runGate` used 64-token chunks only, so the
   16-row GEMM instance (`bonsai_tern_gemm128_m16`) and the GDN chunk kernel at S=16 were covered
   by the self-check alone (124/125). Now the gate runs the engine's greedy plan (64+16+16+16 on the
   125-token prompt) and all three fixtures still pass, same three system-prompt ties.
2. *The streamed loop's error path left one-scalar placeholders as the states* and advanced the
   position counter; `reset` zeroed them happily. Now a state the stream cannot hand back is
   re-allocated, the sequence is reset, and the call throws.
3. *Outputs were trusted by position* (`outputNames[1]` as `next_token`) while states were checked
   by name; the metadata's `next_token_output` was parsed and unused. Every tensor is now looked up
   by name, the announced next-token output must match.
4. *Two argmax tie-breakers, one gated:* the host scan (lowest index wins) in the gate, Core AI's
   `reduce_index` in generation, on fp16 logits where ties near a 0.01 margin are routine. The gate
   now compares both at every walked step (0 mismatches over 134 steps on the three fixtures) and
   `BONSAI_CHECK_ARGMAX=1` does the same on a chat run.
5. *`--kv` was unvalidated:* below the traced KV minimum of 2048 or at exactly `max_ctx` (where
   `position_ids` of length 4096 exceeds its traced max of 4095). The host now refuses capacities
   outside [2048, max_ctx] and caps positions at `min(kv, max_ctx - 1)`. Still open: the first
   walked step passes `position_ids` of length 1 against `Dim(min=2)`; the Mac accepts it, a device
   may not (untested, no device yet).
6. *`A_log` and `dt_bias` were cast to fp16* by the loader's `param`, contradicting the file's own
   docstring; `exp(fp16(log A))` is off by up to 1e-3 relative, 2.2x the error of keeping fp32, on
   the factor that multiplies the recurrent state every token. Both are fp32 now (the zoo's forward
   reads them through `.float()`, the fused step kernel binds the tensor's own dtype). Kernel gate
   ALL OK; full re-export (`exports/bonsai2_27b_decode_pq2_0_pf64`, h16s compile 15 s): all three
   fixtures token-identical, self-check 124/125 (unchanged; the miss is the same fp16 tie at
   position 21), decode 24.3 tok/s (noise-level up from 23.6).

**Broad host battery (2026-09-19):** ten isolated prompts from 53 to 848 tokens covered prose,
code, numbers, repetition, newlines, CJK, mixed scripts and emoji. Chunked prefill vs the S=1
fused walk agreed on 4,347/4,405 teacher-forced prompt argmaxes and **80/80 continuation tokens**.
The intermediate drift is largest on repeated newline/high-ID-token inputs. Follow-up controls on
the 419-position `mixed_scripts` case identified one cause directly: the chunk kernel retains the
GDN recurrent state in fp32 through the whole chunk, whereas an S=1 walk writes it to fp16 after
every token. A 4-layer, same-GDN-path S=64 control moved from 418/419 to 419/419 when the chunk
kernel deliberately reproduced the per-token fp16 writeback. At 64 layers, per-token state
rounding moved the shipping fused-main comparison from 399/419 to 407/419; also using the same GDN
path reached 413/419 at S=64 and 410/419 at S=16. All controls retained 8/8 continuation agreement.
The residual is therefore shape- and path-dependent floating-point reduction order elsewhere in
the chunk graph, not a monotonic chunk-length error; the q/k L2 reduction remains a known candidate
but was not isolated as the sole cause. This is measured internal drift, not a demonstrated
generation failure, and the three MLX-reference generations also remain identical. Per-token state
rounding stays diagnostic because it throws away the chunk recurrence's extra precision and adds
work without making the full model exact.

**Other latent traps:** `TernaryLinear128` under a dynamic query length would compute row 0 only
(the `else` branch of the int check); `last_token_only=True` would double-apply the head transform
(the memo is identity-keyed, the slice is a new tensor); no bounds check on token ids into the
embedding (reachable through `--head-rows`); `BonsaiEngine` is `@unchecked Sendable` with no lock.
Cosmetic: the streamed printer decodes one token at a time and mangles multi-byte UTF-8; EOS is
printed; context overflow exits the process.

## 9. Things not yet verified (carry into the perf rounds)

- **Sign vector semantics are verified by construction, not by numbers**: I have not run the MLX
  runtime (needs the 8.6 GB pack). The first token gate will be the proof that "signs then
  H, scale 1/32 on load, fp32 → fp16" is the whole story.
- The GGUF header's `sign_values` and the MLX pack's `hadamard.json` were compared on width
  list and lengths, not element by element (the pack is the same fold; the MLX loader reads the
  GGUF keys verbatim, `mlx:runtime.py:87-101`).
- Whether Core AI's graph lowering keeps a `[S, K] → [S·K/1024, 1024]` reshape around a custom
  kernel static under a dynamic `S` is the same S=1 question `bitcpm-ternary-1.58bit.md` §3
  answered for the matvec; the two-entrypoint pattern from `ternary-chunked-prefill.md` applies
  unchanged.
