# Streaming Sortformer v2 (speaker diarization) — STATE (updated 2026-07-05)

Pillar ② of the "audio suite completion" campaign (話者分離).
Rubric = `strategy/SHIP_RUBRIC.md`.

**target**: HF `nvidia/diar_streaming_sortformer_4spk-v2`, **license cc-by-4.0** (commercial+redistribute OK).
⛔ offline `diar_sortformer_4spk-v1` is CC-BY-NC = never use. Single `.nemo` (471MB fp32 / 117M params).

---

## STATUS: SWIFT PORT + iPhone DEVICE GATE DONE (2026-07-06). Remaining = ship only.

The neural core is an exported, byte-faithful Core AI static graph, and the full streaming diarization
algorithm (incl. AOSC speaker-cache compression) is validated end-to-end at **100% speaker-activity
agreement** vs NeMo — in Python, in **Swift on Mac GPU**, AND on **iPhone 17 Pro (A19 Pro, AOT h18p)**,
all driving the exported fp16 graph. The Transcribe tab is wired for a diarized transcript. Only the
outward ship (HF/zoo) remains.

### ✅ iPhone device gate PASS (2026-07-06, A19 Pro, AOT h18p GPU)
`coreai-build compile ... --platform iOS --architecture h18p --preferred-compute gpu` → 450 MB
`sortformer_float16.h18p.aimodelc` (fixed shapes, no `--expect-frequent-reshapes`). Installed app +
sideloaded `Library/Application Support/DiarizeAssets` (`sideload_ios.sh`), ran `DIARIZE_SELFTEST=1`:
**byte-identical to Mac** — demo(270) + long(808) loop(golden mel) **activity-agree 100.00%**, full
e2e (Swift mel→loop) **100.00%**, same 7 / 21 speaker turns. RTF 10.3× (first-call warmup) → 420.6×
(long, warm). ship_ios/ staged; `sideload_ios.sh <udid>` reproduces.

### ✅ DONE 2026-07-05 (session 2): Swift host port + Transcribe-tab integration + a long-clip gate
- **host_loop.py compress bug FIXED**: `get_topk_indices` derived `n_frames` from the *pre-pad* count
  (it should be `scores.shape[1]` = padded count, per NeMo `_get_topk_indices`). Harmless on the 2-chunk
  demo (compress never affects its output) but would shift real-frame indices by SIL/spk on long clips.
- **long-clip golden + gate**: `capture_stream_ref_long.py` (demo wav ×3 = 64.5 s → mel[1,128,6464],
  808 out frames, compress exercised ~4×) + `gate_long.py`: eager AND shipped `.aimodel` both
  **activity-agree 100.00%** (cos 0.99997). This is the first output-level validation of AOSC compress.
- **Swift port** (`apps/coreai-audio/Sources/SortformerDiarizer.swift`): `SortformerMel` (NeMo 128-mel,
  normalize=NA, reflect center-pad — STFT+mel bit-identical to NeMo, cos 1.0 pre-log; the log-mel
  "cos 0.995 vs golden" is only the log-amplified silence floor + NeMo dither, irrelevant at the 0.5
  decision) + the full streaming loop + NeMo-faithful AOSC + fixed-buffer graph driving + speaker-turn
  segmentation. Mirrors `gate_e2e_engine.py`/`host_loop.py`/`sortformer_modules.py` 1:1.
- **Transcribe-tab wiring**: `TranscribeModel`/`TranscribeView` get a "Diarize — who said what" toggle;
  diarize → speaker turns → per-turn ASR (any of the 4 engines) → "Speaker N [t0–t1]: text". No ASR
  word timestamps needed (Kit `Transcription` has none) — the diarizer supplies turn boundaries.
- **DiarizeSelfTest.swift** (`DIARIZE_SELFTEST=1`): Mac-GPU gate mirroring gate_e2e/gate_long. **PASS**:
  demo(270) + long(808) loop(golden mel) activity-agree **100.00%**, and full e2e (Swift mel→loop)
  **100.00%** on both. Run: build macOS Release, then
  `DYLD_FRAMEWORK_PATH=<app>/Contents/Frameworks DIARIZE_SELFTEST=1 DIAR_RESULT=/tmp/d.txt <app>/Contents/MacOS/coreai-audio`.
- **ship_macos/** staged (`stage_ship.py`): fp16 `.aimodel` + mel filterbank + golden mel/preds (demo +
  long) + demo wav + metadata; dev symlink `Sources/DiarizeAssets` → it (excluded in project.yml).

### ✅ DONE this session (all gated)
1. **Capture** (`capture_chunk_io.py` → `chunk_io.npz`): real per-stage streaming I/O for both chunks
   (mel_chunk, spkcache, chunk_pe, concat, fc, preds). `forward_for_export` is NOT hit by diarize()
   (goes via forward_streaming_step) — hooked that instead.
2. **Plain-torch re-authoring** (`sortformer_model.py`): `forward_for_export` core re-authored, module
   names match the NeMo ckpt 1:1 (strict load). Gate (`gate_reauthor.py`) = **cos 1.000000 every stage,
   both chunks**. Base = parakeet FastConformer idiom; key deltas baked in: all-bias, `xscale=√512`
   (xscaling=True), non-causal full attention, batch_norm conv, `pre_encode` dw_striding (SUB_CH=256,
   out 4096→512), 18L post-LN transformer (no final LN), head = first_hidden_to_hidden→single_hidden_to_spks.
3. **Core AI export** (`export_sortformer.py` → `artifacts/sortformer_float16.aimodel`, 236.7 MB):
   static graph, masks derived in-graph from one `valid [1,378]` vector (additive attn bias + multiply
   conv mask — equiv to NeMo masked_fill for valid rows since conv re-zeroes padded frames each layer).
   **Mac GPU gate: c0 cos 1.000000 |Δ|0.0049, c1 cos 1.000000 |Δ|0.0072. CPU also PASS.**
4. **Host loop + AOSC** (`host_loop.py`): streaming_feat_loader + sync streaming_update + `compress_spkcache`
   (inference path) re-implemented in torch, drives the eager graph over the full clip, gated vs NeMo
   forward_streaming (`stream_ref.npz`): **cos 0.999996, activity-agree 100.00%, shapes match (1,270,4)**.
5. **End-to-end via the SHIPPED artifact** (`gate_e2e_engine.py`): the exact fp16 `.aimodel` (Mac GPU)
   drives the host loop over the full clip. Graph outputs BOTH `preds` and `chunk_pe` (host needs chunk_pe
   to update spkcache — same dual output as NeMo forward_for_export). **cos 0.999996, activity-agree
   100.00%, shape (1,270,4)**. This is the definitive "Sortformer diarizes on Core AI" proof.

### Fixed-buffer graph contract (what the exported .aimodel expects)
```
inputs:  chunk_mel [1,1520,128]  host zero-pads each mel chunk to 1520  (pre_encode 8x -> ≤190 embs)
         spkcache  [1,188,512]   host-maintained cache, zero-padded to 188
         valid     [1,378]       1 for real frames / 0 for pad. Two blocks:
                                   spkcache region: [0 : spkcache_len]
                                   chunk    region: [188 : 188+chunk_pe_len]   (fixed 188 offset)
outputs: preds     [1,378,4]     sigmoid speaker activity.
                                 host reads spkcache-preds [0:spkcache_len], chunk-preds [188:188+chunk_pe_len]
         chunk_pe  [1,190,512]   pre-encode embeddings (pre-xscale); host takes [0:chunk_pe_len] to
                                 append to the speaker cache during streaming_update
pos_emb baked as a constant buffer (T=378). Layout is the async/fixed-188 convention; equal to the sync
concat_embs path for the valid region (rel-pos shift-invariance + local conv + masking — verified).
```

### Streaming params (from `_nemo/model_config.yaml`, async_streaming=False so SYNC update path)
spkcache_len=188, fifo_len=0, chunk_len=188, chunk_left/right_context=1, subsampling_factor=8,
spkcache_update_period=188, spkcache_sil_frames_per_spk=3, n_spk=4, scores_boost_latest=0.05,
sil_threshold=0.2, pred_score_threshold=0.25, strong/weak/min_pos rate=0.75/1.5/0.5, max_index=99999.
Chunks for a 2160-frame clip: chunk0 mel 1512→pe189 (spkcache empty), chunk1 mel 664→pe83 (spkcache=188).

---

## REMAINING — iPhone + ship

1. ✅ **Swift host loop** — DONE (`SortformerDiarizer.swift`, Mac-GPU-gated 100% activity-agree).
2. ✅ **Transcribe tab** — DONE (Diarize toggle → per-turn ASR → "Speaker N: text"). NOTE: went with
   diarize-then-transcribe-each-turn (Kit `Transcription` exposes no word timestamps), NOT word-timing
   alignment — cleaner and robust; the diarizer supplies turn boundaries.
3. ✅ **iPhone** — DONE (A19 Pro, AOT h18p GPU, DIARIZE_SELFTEST 100% activity-agree vs Mac; see above).
4. ship: HF `mlboydaisuke/Streaming-Sortformer-Diar-CoreAI` (bundle `sortformer_float16.aimodel`
   macOS + `.h18p.aimodelc` iOS + mel filterbank + CC-BY-4.0 text) + zoo README/conversion.

### ⚠️ strategy (unchanged): parity-not-novelty
Sortformer already ships on-device via **FluidAudio (CoreML/ANE)** and **soniqo/speech-swift**. Core AI
here is speed parity, not a novelty flag. Ship as "zoo audio set complete + diarized transcript with our
own ASR", not as a diarization first. START is long-passed; only the Swift/app/device port remains.

### files (conversion/sortformer_diar/)
`sortformer_model.py` (re-authoring) · `gate_reauthor.py` (stage gate) · `export_sortformer.py` (Core AI
export+GPU gate) · `host_loop.py` (AOSC reference, PASS) · `gate_e2e_engine.py` (shipped .aimodel gate) ·
`capture_stream_ref{,_long}.py` + `capture_chunk_io.py` (oracles) · `gate_long.py` (long-clip compress gate) ·
`extract_mel_frontend.py` (mel fb/window from NeMo) · `stage_ship.py` (assembles `ship_macos/`) ·
`chunk_io.npz`/`stream_ref{,_long}.npz`/`oracle_frames.npy` · `artifacts/` · `ship_macos/` (app bundle).
Swift: `apps/coreai-audio/Sources/{SortformerDiarizer,DiarizeSelfTest}.swift` + `TranscribeModel/View` diarize wiring.
envs: NeMo oracle = `_sortformer_oracle/.venv` (torch2.10); re-author/export = `coreai-models/.venv` (torch2.9).

### Reproduce
```bash
# Python — all PASS (activity-agree 100%):
coreai-models/.venv/bin/python gate_reauthor.py      # stage cos 1.0
coreai-models/.venv/bin/python export_sortformer.py  # Mac GPU cos 1.0 (rebuilds .aimodel)
coreai-models/.venv/bin/python host_loop.py          # AOSC eager, demo, 100%
coreai-models/.venv/bin/python gate_e2e_engine.py    # shipped .aimodel, demo, 100%
coreai-models/.venv/bin/python gate_long.py          # shipped .aimodel, 64.5s (compress ~4x), 100%
# Swift self-test on Mac GPU (mirror of gate_e2e+gate_long) — PASS:
cd apps/coreai-audio && xcodegen generate && \
  xcodebuild -scheme coreai-audio -configuration Release -destination 'platform=macOS,arch=arm64' -derivedDataPath build build
APP=build/Build/Products/Release/coreai-audio.app; \
  DYLD_FRAMEWORK_PATH=$PWD/$APP/Contents/Frameworks DIARIZE_SELFTEST=1 DIAR_RESULT=/tmp/d.txt $APP/Contents/MacOS/coreai-audio; cat /tmp/d.txt
# then: iPhone AOT (h18p) + device gate, ship
```

### TODO (2026-07-07) — zoo card: add the kit door
Kit enroll SHIPPED (coreai-kit `6a3738d`, pushed; live catalog has `sortformer-diar-v2`, kind
`diarization`): `KitDiarizer` + `MeetingTranscriber` (diarize-then-transcribe-each-turn as a kit
API) + `Examples/Meeting` CLI. Both golden gates 100% on Mac GPU; ModelStore seeded from
`ship_macos/`. **Remaining here**: update `models/sortformer-diar/README.md` "Use it" (via `cards.json` +
`scripts/gen-cards/gen_cards.py sortformer-diar --write`) to add the kit snippet —
```swift
let meeting = try await MeetingTranscriber(asr: "whisper-large-v3-turbo")
print(try await meeting.transcribe(samples: AudioFile.pcm16kMono(url)).text)
```
— and regenerate the HF README (byte-identical). Also pending: run
`Examples/Meeting` once on the iPhone (graph already device-gated via the app self-test).
