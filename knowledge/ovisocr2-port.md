# OvisOCR2 (0.8B page parser) — Core AI port

Status **in progress**. Numerics are done: **128/128 tokens exact** end to end vs the fp32 HF
oracle, on Mac, through both the JIT and the AOT assets. Mac is shippable at ~6.3 s/page. The phone
**loads** the 2.15 GiB bundle but cannot yet run it — that needs one authoring variant, which is the
whole of the handoff below. Nothing is published, so there is deliberately **no `models/ovisocr2/`**
— that directory is the catalog's published surface (every entry needs a real `hf_repo` and a card
whose numbers are measured). The recipe below moves there when it ships.

```toml
["ovisocr2-vision"]
script = "export_qwen38vl_pipelined.py"
args = ["fp16", "--hf-id", "ATH-MaaS/OvisOCR2", "--name", "ovisocr2",
        "--grid-h", "40", "--grid-w", "28", "--skip-decoder"]
card = "README.md"
source_hf_id = "ATH-MaaS/OvisOCR2"
bundle = "gpu-pipelined/ovisocr2_vision_fp16"
status = "unverified"
overlay = true
notes = [
  "The ~98M Qwen3.5 tower at a PORTRAIT grid: patches [4480,1536] -> image_embeds",
  "[1120,1024], 80x56 patches / 40x28 merged = a 1280x896 page. 196 MB fp16.",
  "First NON-SQUARE grid through qwen3_5_vision.py — the 27B only ever baked 32x32,",
  "so the row/col paths in _init_positional_constants were untested. Gate:",
  "_smoke/test_ovisocr2_tower_gate.py (fp32 fixture captured under the SYSTEM python;",
  "the overlay venv's transformers has no qwen3_5 classes).",
  "Measured: torch fp32 cos 1.000000 / min-row 1.000000 vs HF-fp32; negative control",
  "(raster instead of merge-block-major patch order) collapses to cos 0.389 = the gate",
  "can go red. fp16 .aimodel on the M4 Max GPU: cos 0.999985 but min-row 0.997018 —",
  "4 of 1120 rows land in the 0.997-0.999 band, so it MISSES the repo bar (both cos and",
  "min-row >= 0.999, the same condition the 27B gate uses). Isolated, not explained away:",
  "the low rows are NOT the blank margin (their |ref| median 3.33 vs 3.17 over all rows;",
  "corr(row-cos, pixel-std) = 0.043; 377 tokens are perfectly flat and only 2 of the 4 are).",
  "fp32 authoring is exact, so this is fp16 execution noise — whether it MATTERS is a",
  "question for the end-to-end token gate, which is not done.",
  "Encode 119.7 ms median (10x, M4 Max GPU).",
  "Host preprocessing: _smoke/qwen38vl_preprocess.py preprocess(u8, 1280, 896) — gated",
  "byte-equal (max|d| 0.000e+00) against the HF processor at this grid.",
]

["ovisocr2"]
script = "export_qwen38vl_pipelined.py"
args = ["int8hu", "--hf-id", "ATH-MaaS/OvisOCR2", "--name", "ovisocr2",
        "--grid-h", "40", "--grid-w", "28", "--max-ctx", "8192", "--skip-vision"]
card = "README.md"
source_hf_id = "ATH-MaaS/OvisOCR2"
bundle = "gpu-pipelined/ovisocr2_vl_decode_int8hu_block32_sym_pf16"
status = "unverified"
overlay = true
notes = [
  "OvisOCR2 is a pure fine-tune of Qwen/Qwen3.5-0.8B: text_config AND vision_config are",
  "field-for-field identical to the base, so this rides the shipped qwen3.5-0.8b ship",
  "recipe (int8hu per-block-32 body + absmax-sym int8 head) with new weights.",
  "EMBEDDINGS-INPUT multifunction bundle ('main' S=1 decode + 'prefill' S=16 chunk),",
  "764 MB, + embed_tokens.safetensors 508 MB fp16 (vocab 248320 x 1024 — a third of the",
  "shipped bytes for a 0.8B model). Whole page: 196 + 764 + 508 = ~1.4 GB, inside the",
  "~1.5 GB iOS ship rule.",
  "PF stays 16, not 32: same family, same fp16 doubling-inverse bound as the 27B.",
  "Gated end-to-end: 128/128 tokens exact vs the fp32 HF oracle at this grid",
  "(_smoke/test_ovisocr2_suite_gate.py). No tok/s and no device run yet - timing needs the AOT",
  "compile, which is blocked on this machine's Metal Toolchain, not on the port.",
]
open_questions = [
  "STOP TOKEN: config.json says eos_token_id 248044 (<|endoftext|>) but the tokenizer's",
  "eos_token is <|im_end|> = 248046, and 248044 is what a chat-tuned model never emits.",
  "Driven on the config value the model does not terminate — measured: it ran the full",
  "3000-token cap and restarted the document from the top; on 248046 the SAME page ends",
  "cleanly at 640 tokens. Any host must stop on 248046. Same class as the muse-glimmer",
  "<|eot|> miss. The bundle's own tokenizer_config.json does carry <|im_end|>, so only a",
  "host that reads config.json is exposed.",
  "Grid 40x28 = 1120 tokens is a first pick, not a measured optimum: the HF processor at",
  "max_pixels 1.4 MP chose 43x31 = 1333 for the same page. Ship-grid choice should follow",
  "an accuracy-vs-prefill measurement that has not been run.",
  "The fp16 tower's 4-row cosine tail is CLOSED by the end-to-end gate (128/128 exact) - the",
  "tower recipe entry still reports a red cosine gate, and that is the honest record: the",
  "shipping criterion for this family is tokens, not cosine.",
]
```

## Start here (the port is unfinished — this is the handoff)

**One thing is left: a static-input `image_embeds` VL variant for the qwen3.5 family.** Everything
else is measured and recorded below. Why that one thing, and why it is not blocked, is in
*The iPhone gap is authoring, not the kit*.

**Write it here** — the live checkout, not this repo:

    ~/code/coreai/coreai-models/python/src/coreai_models/models/macos/qwen3_5.py

Model it on `qwen3_vl_pipelined.py` in the same directory (`Qwen3VLPipelinedForCausalLM`). `qwen3_5.py`
is **tracked** upstream, so the new class rides `overlay/patches/python-overlay.patch`, not
`overlay/files/`.

⚠️ **`overlay/regen.sh` sweeps the whole checkout.** It runs
`git diff <BASE> -- python/src/coreai_models` and copies every untracked package file into `files/`.
As of this writing the checkout also holds another session's in-flight work — `bailing_moe_v3.py`
(Ling-3.0), `qwen3_5_mtp_ios.py`, and edits to `lfm2.py` / `model_registry.py` / `models/base.py` /
`models/registry.py`. Running regen as-is publishes all of it. Check
`git -C ~/code/coreai/coreai-models status --short -- python/src/coreai_models` first.

**What already exists** (`~/code/coreai/coreai-models/exports/`, ~16 GB, keep — re-exporting costs an hour):

| dir | what |
|---|---|
| `ovisocr2_vision_fp16` · `ovisocr2_vision_aotc` · `ovisocr2_vision_ios_aotc` | the 196 MB tower, source + macOS h16c + iOS h18p |
| `ovisocr2_vl_decode_int8hu_block32_sym_pf16` | multifunction decoder (`main` S=1 + `prefill` S=16), 764 MB + 508 MB embed table |
| `ovisocr2_vl_decode_int8hu_block32_sym_s1` | decode-only source |
| `ovisocr2_vl_decode_aotc` · `_ios_aotc` | multifunction AOT, **3.6 GB** — Mac only |
| `ovisocr2_vl_decode_s1_efr_aotc` · `_s1_efr_ios_aotc` | decode-only AOT, **2.1 GB** — the one that loads on the phone |

**Already on the iPhone 17 Pro**, app container `com.coreai.pipelinedbench`:
`Documents/models/ovisocr2_s1b/ovisocr2_vl_decode_int8hu_block32_sym_s1.h18p.aimodelc` (loads; see the
device section). `Documents/models/ovisocr2_s1` is the **flattened, broken** copy from the first
attempt — `devicectl` has no delete, so it is dead weight until someone wipes the container.

**Re-run the gates** (from this repo's root; the GPU ones want `_GPU_LOCK` held and the GPU to themselves):

    # fixture (system python — the overlay venv has no qwen3_5 classes)
    python3 _smoke/test_ovisocr2_tower_gate.py <page.png> --capture-ref
    python3 _smoke/dump_ovisocr2_oracle.py <page.png> --max-new 128

    # gates (overlay venv)
    V=~/code/coreai/coreai-models/.venv/bin/python; E=~/code/coreai/coreai-models/exports
    $V _smoke/test_ovisocr2_tower_gate.py <page.png>                  # fp32 authoring
    $V _smoke/test_ovisocr2_tower_gate.py <page.png> --stage aimodel   # fp16 .aimodel
    $V _smoke/test_ovisocr2_suite_gate.py \
        --vision-asset  $E/ovisocr2_vision_aotc/ovisocr2_vision_fp16.h16c.aimodelc \
        --decoder-asset $E/ovisocr2_vl_decode_aotc/ovisocr2_vl_decode_int8hu_block32_sym_pf16.h16c.aimodelc

The page itself is `_smoke/ovisocr2_jp_page.html` (render command in the tower gate's docstring); the
`.npz` fixtures are gitignored by repo convention and regenerate from it.

**Re-export** (from `~/code/coreai/coreai-models`, not from `conversion/` — the drivers' `--out-dir`
default is relative and the repo convention is to run them there):

    python conversion/export_qwen38vl_pipelined.py int8hu --hf-id ATH-MaaS/OvisOCR2 \
      --name ovisocr2 --grid-h 40 --grid-w 28 --max-ctx 8192 --skip-vision

**Then the ship tail**, none of which is started: `models/ovisocr2/{README.md,recipe.toml}` with a real
`hf_repo` (the recipe is drafted below), the HF upload, the four indexes, `gen_inventory.py` →
`gen_llms_txt.py` → `validate_catalog.py`, and kit enrollment (`catalog.json` + `ModelCatalog.swift`)
— OvisOCR2 becomes the fourth reader behind `CoreAI.read(documentAt:)`.

## Why this one

[`ATH-MaaS/OvisOCR2`](https://huggingface.co/ATH-MaaS/OvisOCR2) (Apache-2.0) scores **96.58 on
OmniDocBench v1.6** — above the zoo's own best page parser, MinerU2.5-Pro at **95.69** — and is the
first end-to-end model to top a leaderboard that pipeline methods had held. One model does the whole
page: image → Markdown, tables as HTML, formulas as LaTeX, visual regions as `<img src="images/
bbox_{l}_{t}_{r}_{b}.jpg" />` in reading order. No separate layout pass (MinerU runs two).

## The port is a weights swap, and that is a measured claim

`ATH-MaaS/OvisOCR2` and `Qwen/Qwen3.5-0.8B` have **field-for-field identical** `text_config` *and*
`vision_config` — hidden 1024 / 24 L / head_dim 256 / GQA 8-2 / vocab 248320 / `full_attention_interval`
4 / mrope `[11,11,10]` `partial_rotary_factor` 0.25, tower depth 12 / 768 / patch 16 / merge 2 /
`out_hidden_size` 1024 / `deepstack_visual_indexes: []`. It is a pure fine-tune. So it rides:

* the shipped **qwen3.5-0.8b** decode recipe (`int8hu` per-block-32 body + absmax-sym int8 head), and
* the shipped **`qwen3_5_vision.py`** tower authoring, and
* the shipped **`qwen38vl_host.py`** contract verbatim — `image_token_id` is 248056 in both.

No new authoring module. `export_qwen38vl_pipelined.py` gained `--name` / `--grid-h` / `--grid-w`
instead of being copied, the same way `qwen3.5-2b` reuses the 0.8B script through `--hf-id`.

## The one genuinely new thing: a portrait grid

The 27B only ever baked a **square** 32×32 tile, so `_init_positional_constants` (bilinear pos-embed
interpolation, 2D-rope coordinate build) had never run with h ≠ w. A row/col transposition there is
silent everywhere else. OvisOCR2 bakes **80×56 patches → 40×28 = 1120 merged tokens = a 1280×896
page**, which keeps A4's aspect instead of squashing it into a square (the 27B recipe files that
squash as an open question).

Gated in [`_smoke/test_ovisocr2_tower_gate.py`](../_smoke/test_ovisocr2_tower_gate.py):

| stage | result |
|---|---|
| host preprocess vs HF processor at 1280×896 | **byte-equal**, max\|d\| 0.000e+00 |
| authored fp32 tower vs HF-fp32 | cos **1.000000**, min-row **1.000000** |
| negative control (raster instead of merge-block-major patch order) | cos **0.389** — the gate can go red |
| exported fp16 `.aimodel`, M4 Max GPU | cos 0.999985, **min-row 0.997018 → MISSES the bar** |
| encode | **119.7 ms** median (10×, M4 Max GPU) |

The bar is `cos >= 0.999 AND min-row >= 0.999` — the repo's own, the identical condition the 27B
tower gate uses.

**The fp16 miss, isolated rather than explained away.** 4 of 1120 rows land in 0.997–0.999. The
obvious story — "it's the blank page margin, low-norm rows, cosine noise" — is **wrong**: those rows'
reference norms (median 3.33) are *not* below the all-row median (3.17), `corr(row-cos, pixel-std)`
is **0.043**, and of the 377 perfectly-flat (blank) tokens only 2 are among the 4. Since the fp32
stage is exact, this is fp16 execution noise, not wiring.

**Resolved by the end-to-end gate: it changes nothing.** The full chain is **128/128 tokens exact**
against the fp32 HF oracle (below), so the 4-row tail does not flip a single argmax. Recorded
because the *cosine* gate stays red and the next person will see it: the shipping criterion for
this family is tokens, not cosine — a single-position summary that hides argmax flips, and here
over-reports risk in the other direction.

## Stop token: `<|im_end|>` (248046), not the config's 248044

`config.json` says `eos_token_id: 248044` (`<|endoftext|>`); `tokenizer_config.json` says
`eos_token: <|im_end|>` = **248046**. A chat-tuned model never emits 248044, so driven on the config
value **generation does not stop** — measured on the same page: the full 3000-token cap, ending with
the model re-emitting `<think></think>` and restarting the document from the title. On 248046 the
same page ends cleanly at **640 tokens**. Second sighting of this class (Muse-Glimmer needed `<|eot|>`
declared) — now recorded in [`conversion-guide.md`](conversion-guide.md). The exported bundle's own
`tokenizer/tokenizer_config.json` carries `<|im_end|>`, so only a host that reads `config.json` is
exposed.

## Japanese: reads a real page

The card documents no languages and OmniDocBench is CN+EN, so this was a ship gate. Rendered an A4
technical page (headings, justified body, a 4-row bordered table, a display formula with a caption,
a numbered list) and ran the HF checkpoint on it: headings, both body paragraphs, **all 16 table
cells** as `<table>`, the formula as `$$ T = n \cdot c + \alpha \log_{2}(n + 1) $$`, and the list —
all correct. Errors in ~640 tokens: one character-level miss (ルック＆フィール → ルック＆ファイル) and
one heading that lost its `###`. **Good enough to ship on**; Japanese is not a blocker.

## The AOT size lever: it is the function count, not the flag

`--expect-frequent-reshapes` is **not optional** here, and the first guess about why was wrong.
The reshapes do not come from the `main` S=1 / `prefill` S=16 split — they come from
**`position_ids`, which grows by one every call** (`shape=[1, -1]` on both functions). Any build
therefore reshapes constantly, so dropping efr just moves the cost to the runtime:

| decoder build | iOS AOT (h18p, gpu) | ANE re-specialisations during the gate | prefill | decode | tokens |
|---|---|---|---|---|---|
| multifunction (`main` S=1 + `prefill` S=16) + efr | **3.6 GB** | **0** | **892.0 tok/s** | 134.1 | 128/128 |
| decode-only (`main` S=1) + efr | **2.1 GB** | **0** | 141.4 tok/s | 136.1 | 128/128 |
| decode-only, no efr | 763 MB | **428** — compile-bound, unusable | — | — | — |
| multifunction, no efr | 764 MB | 392+ — compile-bound, unusable | — | — | — |

Without efr the AOT is byte-for-byte the source `.aimodel` and buys nothing. **What efr costs is
~1.5 GB per exported function**: dropping `prefill` takes 3.6 GB → 2.1 GB. That is the lever —
export fewer functions, not fewer flags.

Page latency on an M4 Max through the python driver (1247-token prompt, 640-token page):

* multifunction — 0.14 s tower + 1.4 s prefill + 4.8 s decode = **~6.3 s**
* decode-only — 0.14 s tower + 8.8 s prefill + 4.7 s decode = **~13.6 s**

So the phone build costs ~2.2× the page time to save 1.5 GB — the chunked prefill is worth
**6.3×** on the prompt (892.0 vs 141.4 tok/s), and a page prompt is 1247 tokens.

⚠️ These numbers were re-taken **with the GPU to themselves**. A first pass measured prefill at
472.7 tok/s because an unrelated MPS job was running — nearly a 2× understatement from contention
alone. Anything timed here has to hold `_GPU_LOCK` and run solo, exactly as the repo's rule says. Note the opposite trap still holds:
efr on a genuinely **fixed-shape** graph SIGSEGVs on iOS. Decide it per graph.

## Mac ships. The iPhone load question is answered: it loads.

**Mac: green.** Multifunction + efr, 128/128 exact, ~6.3 s/page.

**iPhone 17 Pro: the 2.15 GiB decode-only bundle loads.** Measured on device
(`PipelinedBench`, `PB_TERNPF=ovisocr2_s1b`, iOS 27 beta):

```
MEM start          footprint=0.012 GB  headroom=6.430 GB
OPTS gpu-reshape → LOAD 1.68 s   |  OPTS default → LOAD 3.51 s
MEM model loaded   footprint=0.048 GB  headroom=6.395 GB
TERNPF ERROR main/prefill missing (functions: ["main"])   <- expected: this bundle has only "main"
```

No `NSPOSIXErrorDomain 2`. The feared multi-IO-above-2 GB wall did not fire for this graph
(5 inputs / 1 output / 4 states, `resources.bin` 2,304,556,708 B = 2.15 GiB), and the post-load
footprint is **0.048 GB** — the weights are mapped, not resident. For calibration, this same phone
already carries Shieldstral's 2.34 GB `resources.bin`, so 2.15 GiB was never the outlier; the open
question was the IO shape, and it is now closed for this bundle. The 3.6 GB multifunction build is
still untested on device and stays Mac-only.

### The trap that cost two transfers: `devicectl copy to` flattens a `.aimodelc`

Pointing `--source` at the `.aimodelc` directory copies its **contents** into the destination, so
the `.aimodelc` level disappears. The runtime then cannot select the AOT specialization and fails
with **`failedToSpecialize` (`CoreAIDelegates.AIModelError` code 1)** — which reads like a bad
compile, not a bad path. All three `SpecializationOptions` modes fail identically, so option
A/B tells you nothing. Push so the `.aimodelc` name survives:

    --destination Documents/models/<dir>/<name>.h18p.aimodelc

Every working sideload on the device has that level (`bitcpm_pf64/bitcpm_8b_ternary_pf64.h18p.aimodelc/…`).
`TernaryPrefill.swift`'s comment says it handles the flattened layout by using `<dir>` itself as the
asset — that works for a `.aimodel`, **not** for an AOT `.aimodelc`.

Also: `--destination` is relative to the **appDataContainer root**, not `Documents`. A first push to
`models/…` reported `EXIT=0` and printed a plausible path, and nothing landed — the container held
only `Documents`/`Library`/`tmp` afterwards. Verify by listing, never by exit code. A genuine 2.15 GiB
transfer took 77 s flat and 311 s nested; anything much faster did not happen.

## Grid: 1120 is not costing anything measurable

Ran the HF checkpoint on the same page at the ship grid and at the grid the processor picks itself:

| grid | tokens in | tokens out | errors |
|---|---|---|---|
| 40×28 = **1120** (ship) | 1120 | 639 | 2 — `ルック＆フィルール`, table cell `導入パターン` |
| 43×31 = 1333 (processor default at `max_pixels` 1.4 MP) | 1333 | 641 | 2 — `ルック＆ファイル`, one heading loses its `###` |

**Two errors each, in different places.** 19% fewer image tokens for no measurable loss, so 1120
stays. One page is thin evidence and this is not an OmniDocBench run — but it is evidence, and it
points the cheap way.

## Sizes — and where a 0.8B model's bytes actually go

| | |
|---|---|
| `ovisocr2_vision_fp16.aimodel` | 196 MB |
| `ovisocr2_vl_decode_int8hu_block32_sym_pf16.aimodel` | 764 MB |
| `embed_tokens.safetensors` (fp16, host-side gather table) | **508 MB** |
| total, uncompiled | **~1.4 GB** |
| total, AOT multifunction + efr | ~4.3 GB — Mac only |
| total, AOT decode-only + efr | **~2.8 GB** — the iPhone candidate |

The embed table is a third of the shipped bytes because vocab is 248320 × 1024. For this family the
vocabulary, not the body, is what to attack if the budget ever gets tight.

## Tooling written for this port

| file | what |
|---|---|
| `_smoke/ovisocr2_jp_page.html` | the fixture's source page (render → capture → gate; `.npz` is gitignored by convention) |
| `_smoke/test_ovisocr2_tower_gate.py` | tower gate + `--capture-ref`, with a raster-order negative control |
| `_smoke/dump_ovisocr2_oracle.py` | fp32 greedy oracle **at the shipped grid** (asserts the processor did not pick its own) |
| `_smoke/test_ovisocr2_suite_gate.py` | end-to-end token gate; reads hybrid state shapes from the bundle's own `desc.state_descriptor` instead of hardcoding them, so it runs for any Qwen3.5-family export |

Three existing files were **generalized rather than copied**, the way `qwen3.5-2b` reuses the 0.8B
script: `export_qwen38vl_pipelined.py` gained `--name`/`--grid-h`/`--grid-w`, and
`qwen38vl_preprocess.py`'s `preprocess()` gained a rectangular tile (its own 6-case gate still
passes byte-equal). `qwen38vl_host.py` needed nothing.

## End to end: 128/128 tokens exact

`_smoke/test_ovisocr2_suite_gate.py` runs the whole shipped chain — NumPy preprocess → fp16 tower
`.aimodel` → embed splice + host mRoPE → int8hu embeddings-decoder (`prefill` S=16 chunks, then
`main` S=1) → greedy — against the fp32 HF oracle captured at the same grid:

```
oracle: prompt 1247 tok, 128 generated, grid (1, 80, 56), dtype float32
state   keyCache (6,1,2,2048,256) · convState (18,1,6144,3) · recState (18,1,16,128,128)
tower   (1120, 1024) in 149 ms
PASS: 128/128 tokens (100.0%)
```

So **quantization is free here**: int8hu body + absmax-sym int8 head + an fp16 tower reproduce fp32
greedy exactly over 128 tokens. The zoo's qwen3.5-0.8b ship recipe transfers to this fine-tune
without a numerics concession.

## Measured (M4 Max, macOS GPU, AOT h16c, python driver)

Same 128/128 through the AOT assets, with usable timings:

| | |
|---|---|
| tower (1120 tokens, one shot) | **135 ms** (10× median on the tower gate: 125.9 ms) |
| prefill (chunked, S=16) | **892.0 tok/s** |
| decode | **134.1 tok/s** |

A full page at the measured 640-token output is therefore roughly **6.3 s** on an M4 Max through
the python driver. The pipelined engine would do better — the shipped qwen3.5-0.8b text bundle is
210 tok/s on this machine against 134.1 here, the same python-driver overhead the 27B recorded.

The AOT tower is **bit-identical to the JIT one** (cos 0.999985, min-row 0.997018 both ways), which
closes the last alternative explanation for the fp16 tail: it is the graph's own fp16 numerics, not
a compile artifact.

## The iPhone gap is authoring, not the kit — and this port exported the wrong variant

Correcting an earlier reading of this. The kit is **not** missing a driver. `CoreAIKit/Vision`
(`VLArchitecture` + `VLRuntime`) is a generic VLM driver with presets for Qwen3-VL 2B/4B/8B,
LFM2.5-VL, North-Micro-Vision and GLM-OCR; `CoreAIKit/OCR` has three shipped readers
(`KitDocReader`, `KitMineruReader`, `KitGlmOcrReader`) behind `CoreAI.read(documentAt:)`, running
on iPhone today — GLM-OCR even on a **portrait non-square** 32×24 grid, so the grid shape was never
the obstacle either.

What the kit drives is one specific contract, the one `qwen3_vl_pipelined.py` exports:

    input_ids [1,s]          image tokens encoded as V + slot
    position_ids [1,total]
    image_embeds [N,h]       static input
    rope_shift_start/amount  static inputs
    embedding = ids < V ? embed_tokens[ids] : image_embeds[ids - V]     -- INSIDE the graph
    position  = p >= rope_shift_start ? p - rope_shift_amount : p

The embed table stays in the graph, so the engine drives it and `VLRuntime` binds the image rows as
a static buffer. **This port exported the other variant** — `Qwen3_5VLStatefulEmbeds`: `inputs_embeds`
plus three interleaved-mRoPE planes, with the host doing the gather and a 508 MB
`embed_tokens.safetensors` shipped alongside. That is correct for the 27B, which is Mac-only and
driven from python, and it is what makes the bundle *not* engine-drivable — so neither the stock
engine nor the kit can run it, whatever the kit contains.

The actual gap is upstream of both: **`coreai_models/models/macos/qwen3_5.py` has exactly one VL
class, `Qwen3_5VLStatefulEmbeds`.** The qwen3.5 family has no static-input `image_embeds` variant at
all. Writing one — modelled on `Qwen3VLPipelinedForCausalLM` — is a **new authoring module**, which
is this project's session boundary.

Two things it would fix for free: no 508 MB host-side embed table (it moves back into the graph,
cutting the phone footprint), and chunked prefill through the engine instead of the S=1 ingest that
costs 6.3× on the prompt.

**The question I flagged as open is closed, and its premise was wrong.** I wrote that the
rope-shift hook "collapses image positions with a single scalar subtraction, which suits Qwen3-VL's
*sectioned* M-RoPE" and that Qwen3.5's interleaved form might not fit. Both halves are wrong.
`qwen3_vl_pipelined.py` says it plainly: Qwen3-VL is **already interleaved**, and the 3D position is
computed **in-graph from `(ids, position)` alone** —

    image tokens self-locate: slot = ids - V, s0 = pos - slot,
                              t = s0, h = s0 + slot//W, w = s0 + slot%W
    the interleave is three constant 0/1 masks over head_dim
      (freq j: j%3==1 and j<3*sec_h -> h;  j%3==2 and j<3*sec_w -> w;  else t)

The shift does none of the 3D work — it only fixes *post-image text* positions, because an image
consumes `max(H,W)` rope positions rather than `N`. So there is no scalar-vs-3D problem to solve.
Qwen3.5's `mrope_section [11,11,10]` sums to 32 pairs, exactly the 64 rope dims that head_dim 256 ×
`partial_rotary_factor` 0.25 gives, and drops straight into the mask scheme.

What genuinely differs from Qwen3-VL, and is already solved elsewhere: Qwen3-VL is pure attention
("NO extra states"), while Qwen3.5 is a GDN/full-attention hybrid with `convState`/`recState`. The
shipped `qwen3.5-0.8b` decode bundle already rides the pipelined engine with those four states via
`apps/coreai-pipelined-extra-states.patch`, so that piece exists too.

**Net: the variant is tractable and closely modelled on an existing file.** It is still a new
authoring module, which is this project's session boundary — but it is not blocked on an unknown.

## Not done

* **Everything above.** The phone can load the bundle; it cannot run it, and the route to running it
  is the authoring variant, not app code.
* **The 3.6 GB multifunction build on device** — untested; Mac-only for now.
* **No publish.** Nothing is on the Hub; there is no `models/ovisocr2/` yet by design.
* The grid evidence is one page, not a benchmark. If a real OmniDocBench-style sweep ever runs,
  1120 vs 1333 is the first thing to re-check.
* The tower's fp16 4-row cosine tail is closed for *tokens* (128/128, twice, JIT and AOT). It has
  not been checked on a page whose content differs sharply from this one.
