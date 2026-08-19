# Parakeet-TDT-0.6B-v2 — Core AI

The **English-specialist sibling** of the zoo's [`parakeet`](../parakeet/README.md) entry.
[`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
(cc-by-4.0, 600M) is the same token-and-duration transducer architecture as the enrolled
v3 port — FastConformer encoder, LSTM predictor, joint network with a duration head,
three stateless `.aimodel` graphs plus a host-driven greedy loop — trained for English
alone rather than 25 European languages.

The two differ where the vocabulary does: v2 has **1025 entries with blank at 1024**,
against v3's 8193 with blank at 8192. Everything else in the pipeline is shared, which is
why both checkpoints export from one pair of scripts here rather than a fork.

## Why v2 alongside v3

v3 buys 25 languages; v2 spends the same 600M parameters on English only. For
English-only long-form transcription that is a trade worth having both sides of, and this
port exists because its author runs v2 in production for exactly that reason. It is also
the ASR that gates the pocket-tts port's acceptance ladder, so it already does work
inside this catalog.

Related: [apple/coreai-models#163](https://github.com/apple/coreai-models/issues/163) asks
for v2 support alongside the v3 support that landed in that repo's #136.

## Pipeline

```
audio 16 kHz ──(host: librosa-slaney mel, 128 × 2885)──▶
  1. parakeet_encoder_float16_L2885.aimodel   mel[1,128,2885] -> enc_proj[1,361,640]
  2. parakeet_predict_float32.aimodel         token[1,1], h,c[2,1,640] -> dec_out[1,640], h', c'
  3. parakeet_joint_float32.aimodel           relu(enc_frame + dec_out) -> token_logits[1,1025],
                                                                          dur_logits[1,5]
host: greedy TDT loop — blanks advance time, non-blank tokens advance the predictor,
      the duration head decides how many frames to skip. durations [0,1,2,3,4].
```

L2885 is 30 s of mel frames baked into the graph, giving T = 361 encoder frames.

## Gates

Full transcript ships beside the bundles as `gates_v2.txt`. Reference is HF
`ParakeetForTDT` on the converted checkpoint; the clip is 14.84 s of librispeech speech
plus trailing silence into the L2885 bucket, 82 gold tokens.

| gate | result |
|---|---|
| encoder, eager fp32 | global cos 1.000006, per-token mean 1.000000, max abs 0.0 |
| encoder, `cpu_only` | per-token cos mean 0.999911, min 0.998284, max abs 0.430 — PASS |
| encoder, `gpu` | per-token cos mean 0.999998, min 0.999970, max abs 0.055 — PASS |
| decoder, eager | 82/82 tokens exact, step-logit cos 1.000000 |
| decoder, engine `gpu` | 82/82 tokens, exact — PASS |
| end to end, `gpu` | 82/82 emitted, token-agree 82/82, exact — PASS |

The mel gate is worth reading in the transcript rather than the table. Feeding per-clip
mel with zero padding emits 85 tokens against 82 gold and agrees on only 22, while the
oracle-style path and a manual-DFT Swift simulation are both 82/82 exact. The padding
convention is load-bearing, and a host that gets it wrong fails quietly with plausible
text.

## Speed

M4 Pro, over a 340-minute long-form corpus at the published precision: **291× realtime at
peak**, about 284× typical, and roughly 260× with the machine under normal desktop load.

iPhone 17 Pro Max (`iPhone18,2`), iOS 27.0 (build 24A5408d), Release, charging, encoder
on gpu with predictor and joint on cpu, stream depth 2, one decode worker. Corpus is 152
chunks over 3989.9 s (66 minutes) of audio.

| metric | value |
|---|---:|
| RTF excluding load | **175.3× realtime** |
| RTF including load | 173.3× |
| load | 0.27 s |
| peak RSS | 1458 MB |
| accuracy over the run | 152/152 chunks token-exact, 20376/20376 tokens (100.0000 %) |

Thermal state went nominal to fair across the 66-minute run, with per-chunk time drifting
from about 0.50 s to about 0.78 s as it did. Of the compute, the encoder is 73.78 s and
the TDT loop 11.40 s, and 90 % of that loop is predictor dispatch at 500 µs across 20528
calls — the decode side is dispatch-bound, not compute-bound.

Through CoreAIKit rather than the author's host: `KitParakeetModel` loads all three graphs on
one compute unit and runs the TDT loop synchronously, one `joint` call per encoder frame. On an
iPhone 17 Pro with everything on `gpu`, one full 28.75 s window measures 0.42 s — 68× realtime,
warmup discarded, 256 MB peak. That is the same dispatch-bound decode the table above describes,
without the two levers it uses (predictor and joint on `cpu`, stream depth 2), so read it as a
floor for the kit path rather than a comparison with the numbers above.

## Bundles

[`rahulrachuri/parakeet-tdt-0.6b-v2-coreai`](https://huggingface.co/rahulrachuri/parakeet-tdt-0.6b-v2-coreai),
cc-by-4.0 inherited from the model.

| bundle | size | role |
|---|---:|---|
| `parakeet_encoder_float16_L2885.aimodel` | 1165 MB | FastConformer encoder + projector |
| `parakeet_predict_float32.aimodel` | 30 MB | embedding, 2-layer LSTM, projector |
| `parakeet_joint_float32.aimodel` | 3 MB | joint network, token + duration heads |

The repo also carries the tokenizer assets, the librosa-slaney mel filterbank, the gate
transcript, and the converter patch described below.

## Converting the checkpoint

NVIDIA publishes v2 as a `.nemo` only, so the HF-layout checkpoint the exporters read is
itself a conversion, published as
[`rahulrachuri/parakeet-tdt-0.6b-v2`](https://huggingface.co/rahulrachuri/parakeet-tdt-0.6b-v2).
It needed two fixes to transformers' `convert_nemo_to_hf.py`, both because v2's NeMo vocab
has 1024 labels and, unlike v3's, carries no `<pad>`:

- `write_processor` must take the RNN-T ordering when a TDT vocab has no `<pad>`, so
  `<blank>` lands on id 1024, the model's own `blank_token_id`. Stock adds `<pad>` at 1024
  and `<blank>` at 1025, which is off by one.
- `convert_tdt_config` must fall back to `blank_token_id` when `<pad>` is absent. Stock
  raises `ValueError: '<pad>' is not in list`.

The patch ships beside the bundles as `convert_nemo_to_hf_v2.patch`.

## Licence

Model weights are cc-by-4.0 from NVIDIA and carry that licence into the converted
bundles. The conversion scripts here follow this repository's licence, and the Swift host
[parakeet-swift](https://github.com/RahulRachuri/parakeet-swift) is separate and
distributes no weights.
