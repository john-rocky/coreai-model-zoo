# GLiNER2-PII — Core AI

The zoo's **first NER / schema-driven information-extraction** model, and its first **DeBERTa-v3**
(disentangled-attention) port. Zero-shot entity extraction — pass any label set at call time and the
model finds those entities in the text. The flagship use is **on-device PII redaction**.
[`fastino/gliner2-privacy-filter-PII-multi`](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi)
(Apache-2.0) on a multilingual [mDeBERTa-v3](https://huggingface.co/microsoft/mdeberta-v3-base) base
(278M), fused into **one static Core AI graph**; the tokenizer, schema linearization, and span decode
run in the Swift host.

> **Uncontested on iPhone.** An on-device GLiNER2 already exists (GLiNER2Swift) but it is
> macOS-CPU / MLX only. This runs the GPU on iPhone (and, AOT-compiled, the ANE) — the first GLiNER
> on Apple Silicon's accelerators.

## Use it

Three lines with [**CoreAIKit**](https://github.com/john-rocky/coreai-kit) — `InformationExtractor`
downloads the bundle once, then runs fully offline:

```swift
import CoreAIKitEmbeddings

let extractor = try await InformationExtractor(model: .gliner2PII)

// zero-shot: any labels, decided at call time
let entities = try await extractor.extract(
    from: "Contact Dr. Sarah Johnson at sarah.j@acme.com or +1-415-555-0142.",
    entities: ["person", "email", "phone number"])
// ["person": ["Sarah Johnson"], "email": ["sarah.j@acme.com"], "phone number": ["+1-415-555-0142"]]

// or redact in place (default replacement is [LABEL]; pass your own for ██ blocks)
let clean = try await extractor.redact(
    "SSN 123-45-6789, card 4111 1111 1111 1111.",
    entities: ["social security number", "credit card number"])
// "SSN [SOCIAL SECURITY NUMBER], card [CREDIT CARD NUMBER]."
```

Runnable demo: **[Examples/InfoExtract ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/InfoExtract)**
— a paste-text → detect-and-redact PII app (iOS), plus an `infoextract-cli` (macOS,
`--redact` / `--gate`).

## How it works

One fused static graph runs the whole model; the host handles the text↔schema plumbing that keeps it
schema-agnostic.

- **Fused graph** — `forward(input_ids[1,256], attention_mask[1,256], text_word_idx[1,96],
  schema_idx[1,17]) → span_scores[1,16,96,8]`. Inside: mDeBERTa-v3 (disentangled attention, exported
  at a fixed shape so the relative-position buckets gather cleanly — no hand-written attention
  rewrite) → "first" sub-word pooling → SpanMarker → CountLSTM → einsum → sigmoid. `MMAX=16` labels,
  `T=96` words, span width `K=8`.
- **Schema-agnostic** — the label set is **not** baked in. The host linearizes the user's labels into
  `input_ids` (`( [P] entities ( [E] l0 [E] l1 … ) ) [SEP_TEXT] …`) and supplies the gather indices,
  so one converted bundle answers any schema up to `MMAX`.
- **Host collator** — mDeBERTa SentencePiece/Unigram tokenization + GLiNER word-split + schema
  linearization + first-sub-word / schema-marker gather positions. Byte-identical to GLiNER2's Python
  `collate_fn_inference`.
- **Host decode** — per-label threshold + confidence-descending greedy NMS over character spans,
  byte-identical to GLiNER2 `_format_spans`.

## Verification

Byte-gated against reference GLiNER2 `ext.extract` at every tier — the Swift collator matches Python
`collate_fn_inference` (input ids + gather indices), the fp16 graph matches the fp32 reference
(span-scores cos **0.999993**), and decoded entities match exactly:

- **Python** — fp32 span-scores cos 1.0, decoded entities == `ext.extract`.
- **Swift on Mac GPU** — the demo PII suite decodes byte-identically; arbitrary runtime schemas
  (credentials, org/money/date/location) also match `ext.extract` exactly.
- **iPhone 17 Pro** (A19 Pro, AOT h18p) — same suite, `GATE_RESULT: PASS`. Load ~1.8 s; extraction
  ~22–32 ms per text (warm).
- **iPhone 18 Pro** (A20 Pro, h19p) — the `ios/` JIT `.aimodel` loads in 0.75 s on the first launch (the
  phone specializes it itself) and 0.09 s after; first call 1.4 s, then warm. Load-only measurement
  (2026-09-26, [`knowledge/jit-distribution.md`](../../knowledge/jit-distribution.md)); the extraction suite
  was not re-run on this phone.

Reproduce (Mac):

```bash
python conversion/export_gliner2_pii.py --output-dir /tmp/gliner2_out --dtype float16   # all gates
# then, in coreai-kit:
swift run infoextract-cli --bundle /tmp/gliner2_out --gate                              # kit E2E, ALL PASS
```

Conversion + the fused-graph export: [`conversion/export_gliner2_pii.py`](../../conversion/export_gliner2_pii.py).
Porting lessons (disentangled-attention export, the swift-transformers tokenizer routing, the
schema-agnostic collator): [`knowledge/gliner2-pii.md`](../../knowledge/gliner2-pii.md).

- 🤗 [GLiNER2-PII-CoreAI](https://huggingface.co/mlboydaisuke/GLiNER2-PII-CoreAI)
  — `macos/` and `ios/` each hold the JIT `.aimodel` (fp16, 611 MB; every iPhone generation specializes it on
  its first load) + `tokenizer/` + `extractor.json`; `ios-h18p/` keeps the AOT-compiled h18p bundle for the
  iPhone 17 Pro (revision `887627e`, 2026-09-26; before it `ios/` carried the h18p bundle, which the iPhone 18 Pro
  refuses). The h18p package dates from the 2026-07-07 export and was not recompiled after the 2026-07-20
  coreai-torch 0.4.1 re-export that the JIT `.aimodel` carries.
- Base model: [fastino/gliner2-privacy-filter-PII-multi](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi) (Apache-2.0).
