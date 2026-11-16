# Chatterbox TTS — zero-shot voice cloning, text→speech on iPhone (Core AI port notes)

Verified engineering notes from porting [`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox)
to Core AI. This is the zoo's **first zero-shot voice-cloning TTS** and its **first multi-network
generative-audio pipeline** proven end-to-end on iPhone: text goes in, a language model + a
flow-matching decoder + a neural vocoder all run on the A19 GPU, ~2 s of compute yields 2.16 s of
audio, no server.

## The model is four networks + host DSP

| net | role | Core AI form |
|---|---|---|
| **T3** | AR speech-token LM (Llama_520M, embeds-in) | int8 stateful graph, KV cache |
| **S3Gen encoder** | speech tokens → mel-conditioning `mu` (UpsampleConformer + proj) | static graph (bucketed) |
| **S3Gen estimator** | CFM flow-matching velocity UNet1D (CFG batch-2) | **fp16** static graph (bucketed) |
| **HiFT** | vocoder: f0-predictor + HnNSF source + conv trunk | f0-predictor graph + trunk graph + **host** STFT/iSTFT/source |

The host (Swift) assembles all embeddings, runs the CFM Euler loop, and does the vocoder source +
STFT/iSTFT (the FFT/rand ops don't lower) — the Kokoro `compute_har` idiom, generalized.

## Gates (all PASS, A19 GPU vs Mac/HF)

- **Every graph verified numerically on iPhone 17 Pro:** T3 prefill logits cos **0.99994** / argmax
  exact; T3 decode (KV-grow) cos **0.99998** / argmax exact / **29 ms/token**; S3Gen encoder cos
  **1.00000**; CFM estimator cos **1.00000** (fp16, **150 ms/step** vs 3.3 s fp32 = 22×); vocoder
  wav cos **0.9998**.
- **Full audio pipeline (speech tokens → wav) on device:** encoder + CFM Euler-10 + vocoder →
  wav cos **0.9998** vs Mac, ~6 s.
- **Full text→wav on device:** text → T3 AR (CFG + sampling, natural stop) → bucketed S3Gen →
  vocoder → 2.16 s wav, ~2 s total.

## Findings worth keeping

- **fp16 is the estimator's ship lever, not int8.** The CFM UNet1D is the 10×-called bottleneck.
  fp32 = 3.3 s/call → too slow (~40 s/utterance). `export_to_coreai(model.half(), half_inputs)`
  works and lands **150 ms/call at cos 1.000000** — the earlier "fp16/int8 export fails" was a bug
  in the *reference* computation (`.float()` on fp16 inputs), not the export. Step-count reduction
  is a false economy: wav-vs-10-step cos falls 8→0.958 / 6→0.814 / 4→0.443; with fp16 speed you
  keep all 10 steps.
- **The T3 needs CFG + sampling; greedy degenerates.** Greedy argmax (even with CFG) locks into a
  repeated token and never emits stop. The shipped recipe is CFG `cfg_weight 0.5` (**two KV caches**:
  cond = cond-prefix+text+speech, uncond = the same with the *text-token* embeddings zeroed but
  their positional embeddings kept; `logits = cond + 0.5·(cond − uncond)`) + sampling
  (temperature 0.8, repetition-penalty 1.2, top-p 0.95). With it, the AR loop emits diverse tokens
  and stops on its own.
- **The graph position-id convention is full-range.** A decode step passes `position_ids = [0…P]`
  (length processed+q), **not** `[P]`; `offset = len(position_ids) − q` selects the new token.
  Getting this wrong on the *reference* (overlay `KVCache` also derives offset from the position-id
  length) produced a phantom "KV-grow mismatch" until both sides used the same convention.
- **Static shapes + bucketing beat dynamic export.** Neither the Conformer encoder (`(2N−1)%N`
  guard from the 2× upsample) nor the HiFT trunk (`s_stft = 120·T+1` coupled to mel `T` through the
  conv strides) exports cleanly with `torch.export.Dim`. Bucketing is the answer: export encoder@256
  tokens and estimator@512 mel; **pad the tokens and pass the real `xs_lens`** (encoder masks the
  padding → real-region `mu` cos 0.9996); **pad `mu`/`z`/`cond` to the bucket with `mask=1` only on
  the real region** (CFM real-region mel cos 1.0). The vocoder buckets to mel=256 the same way and
  trims the wav to `real_T·480`.
- **`reflection_pad` is load-bearing in the HiFT trunk.** `HiFTGenerator.decode` applies
  `ReflectionPad1d((1,0))` on the *last* upsample stage before the source fusion. Omit it and the
  source/`x` lengths align only at the traced length (fine at T=112, cos 0.9998) but broadcast-fail
  at any other bucket — it's the difference between 0.9998 and exact, and the reason a second bucket
  wouldn't export until added.
- **ELU doesn't lower** — the f0-predictor's `nn.ELU` must be swapped for the identity
  `where(x>0, x, expm1(x))` before export (cos 1.0 after).
- **`~/Library/Caches/coreai-cache` serves stale compiles at the same asset path** — always clear it
  after re-exporting to the same directory, or a fixed graph reports its predecessor's numerics.
- **Device gotchas:** the estimator/T3 `inputs_embeds` and KV states are **Float16**; KV must be
  zeroed (prefill computes fresh at offset 0 so it passes regardless — the decode is what reads the
  grown cache); `devicectl copy to` into an existing directory does **not** reliably add new files,
  copy new `.bin` individually.

## Reproduce

```bash
cd coreai-models   # chatterbox installed in the venv (uv pip install chatterbox-tts)
V=.venv/bin/python; C=../coreai-models-community/conversion/chatterbox
$V $C/oracle_chatterbox.py                    # text+ref → wav + intermediates (reference)
$V $C/export_chatterbox_t3.py                 # T3 int8 embeds-in stateful graph
$V $C/export_chatterbox_s3gen.py              # encoder + CFM estimator + HiFT trunk
$V $C/export_chatterbox_hostdata.py           # embed tables, cond prefix, default-voice conditioning
# fp16 estimator + bucketed encoder@256 / estimator@512 / f0@256 / hift_trunk@256 + f0 ELU-swap:
#   all are export_to_coreai(...) with a padded static shape.
```

On-device: `PipelinedBench` env flags drive the headless gates — `PB_CHATTERBOX` (estimator),
`PB_CHATTERBOX_VOC` (mel→wav), `PB_CHATTERBOX_FULL` (tokens→wav), `PB_CHATTERBOX_T3` (T3
prefill+decode), `PB_CHATTERBOX_SPEAK` (**text→wav end-to-end**). Sideload the graphs + host-data
(`.bin`) to `Documents/models/`; the run writes `Documents/cbx_speak.f32` to pull back and play.

The `nn.ELU`→manual and `reflection_pad` fixes live in the export reconstructions; the CFG + sampling
+ bucketing live in `chatterboxSpeakBench`. Overlay: `models/macos/chatterbox_t3.py`.
