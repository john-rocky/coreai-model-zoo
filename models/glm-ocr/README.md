# GLM-OCR (0.9B) — Core AI document OCR

**On-device document recognition on Core AI.** A port of
[`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR) (**MIT**, 0.9B) — a small, SOTA-quality
document recognizer (OmniDocBench v1.5 **94.62**, #1 with its layout pipeline). Prompt it with
`Text Recognition:` / `Table Recognition:` / `Formula Recognition:` and get plain text (reading
order), HTML tables (`<table>…`), or LaTeX. zh / en / fr / es / ru / de / **ja** / ko. The zoo's
second doc-OCR, after [Unlimited-OCR](../unlimited-ocr/README.md) — a higher-quality, simpler architecture.

GLM-OCR is a small OCR variant of **GLM-4.V** (`Glm4v`): a CogViT vision tower + a 16-layer GLM text
decoder with **sectioned 3D M-RoPE**. This port is the shipped [Qwen3-VL](../qwen3-vl/README.md) vision idiom
(the `image_embeds` + rope-shift static-input hook) with a GLM text decode — **no R-SWA, no MoE, no
MLA**. The vision tower runs once; its `image_embeds` are injected at the image placeholder positions
(`V + slot`, row-major over the merged grid) and the text decodes on top.

**⬇️ Converted `.aimodel` bundles:
[mlboydaisuke/GLM-OCR-CoreAI](https://huggingface.co/mlboydaisuke/GLM-OCR-CoreAI)** —
`vision/` (CogViT, fp16, 829 MB) + `decoder/` (GLM decoder, S=1 pipelined, int8hu, 764 MB) +
`tokenizer/`. MIT.

## Architecture

- **Vision (CogViT ≈ Glm4v ViT)**: hidden 1024 / 24 L / patch 14 / 336² / spatial-merge 2, out 1536.
  **RMSNorm** blocks (not LayerNorm) + per-head q/k-norm + gated-SiLU MLP (bias) + 2D rotary + a
  `downsample` conv (2×2) + a GLM patch merger. **No deepstack, no learned pos-embed** — just baked
  2D-rope constants. Exported as one fp16 `.aimodel`; `N` (visual tokens) is fixed by the export grid.
- **Decoder (GLM text)**: hidden 1536 / 16 L / GQA 16-8 / head_dim 128 / vocab 59392. **Sandwich
  norm** (4 RMSNorm/layer), fused `gate_up_proj`, **sectioned M-RoPE `[16,24,24]`** with GLM
  interleaved rotate. Driven on the pipelined-engine S=1 contract: `input_ids [1,1]`,
  `position_ids`, static `image_embeds [N,1536]` + `rope_shift_start` + `rope_shift_amount`. Zero
  embeds + `shift_start = 1<<30` → a plain GLM text decoder.

## Verified (M4 Max, GPU, Core AI pipelined engine)

- **2026-11-07 (JST): decoder re-exported with its RoPE frequencies kept in fp32.** The previous
  decoder stored its 64 RoPE frequencies rounded to fp16 precision (relative error up to 4.1e-4); the
  new one stores them bit-equal to fp32 (64/64), and that is the only constant that changed. Only
  `decoder/` is replaced; `vision/` and the tokenizer files are unchanged. On everything measured the
  output did not change (M4 Max, macOS 27.0, pipelined engine, greedy): on the synthetic document both
  decoders give 40/40, and all 93 tokens up to the end-of-turn token, identical to the fp32 HF
  reference; on three text-only prompts of 2100, 2800 and 3400 tokens (64 generated each, reaching
  positions 2163, 2863 and 3463) they agree token for token — on the 2800-token prompt both leave the
  fp32 reference at step 5, where its top two logits differ by 0.036. One near-tie does flip: on the
  bare prompt `Text Recognition:` with no image, the fp32 reference stops at once with the end-of-turn
  token, 0.056 logits ahead of the backtick token; the previous decoder also stopped at once, the new
  one starts with a backtick instead. Decode speed is unchanged (after / before = 1.004, measured back
  to back on the same Mac). On iPhone 18 Pro (iOS 27.2) the new decoder gives the same 93 tokens on
  the synthetic document and the same backtick on the bare prompt as the Mac.
- **End-to-end real generation on the engine: 40/40 tokens identical to the fp32-HF reference** — a
  synthetic document read verbatim (*"Quarterly Report / On-device inference shipped across all
  product lines this quarter…"*), **~375 tok/s** decode (int8hu S=1), driving the S=1 bundle on the
  pipelined engine with `COREAI_CHUNK_THRESHOLD=1`.
- Torch ladder vs HF: decoder logits cos **1.000020**, vision cos **1.000061**, full-VLM argmax
  **694/694**.
- Engine gate: vision `image_embeds` cos **0.9998**; decoder argmax exact over the sampled positions.
- **int8hu vs fp16: 7 / 694 argmax flips**, all at visual-token positions (0 in the text region), the
  generation-driving position exact — the OCR text is preserved. (int8lin: 9/694.)

## Pipeline (host side)

```
page image → resize to the export grid (here 22×31 merged = 682 visual tokens)
           → vision .aimodel                                   → image_embeds [682,1536]
           → prompt: [ …, <image>×682, "Text Recognition:" ]   (image ids → V+slot, row-major)
           → decoder S=1 pipelined decode (image_embeds injected, shift_start=img_start+N,
                                           shift_amount=N−max(gh,gw))                 → tokens
           → detokenize                                        → text / <table>HTML / LaTeX
```

## Use / reproduce

- **Convert**: [`conversion/export_glm_ocr_pipelined.py`](../../conversion/export_glm_ocr_pipelined.py)
  (`fp16` / `int8lin` / `int8hu`; vision stays fp16). Bundles + HF upload:
  [`conversion/_glmocr_hf_upload.py`](../../conversion/_glmocr_hf_upload.py).
- **Run (Mac)**: drive the S=1 decoder bundle on the pipelined engine with three static inputs
  (`image_embeds` + `rope_shift_start` + `rope_shift_amount`) and `COREAI_CHUNK_THRESHOLD=1`; feed the
  prompt with the image placeholders rewritten to `V+slot`. The host contract + the exact static-input
  values are in the knowledge doc below.
- **Knowledge**: [`knowledge/glm-ocr-port.md`](../../knowledge/glm-ocr-port.md).

## Notes

- **Recognition model**: per-prompt text / table / formula. Whole-page auto-structuring (the 94.62
  full-pipeline number) additionally needs a layout detector (PP-DocLayoutV3), not part of this port.
- **Appropriate input**: clean single-page documents, resized to the export grid (dense small text is
  resolution-dependent — pick a larger grid at export).
- int4 not shipped (weight-only int4 without QAT risks a quality cliff on 0.9B).
- License: **MIT**. *Community port — not affiliated with Apple or Z.ai.*
