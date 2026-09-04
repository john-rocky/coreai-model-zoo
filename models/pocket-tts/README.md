# pocket-tts (Kyutai) on Core AI

Text to speech from [`kyutai/pocket-tts-without-voice-cloning`](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning)
(CC-BY-4.0), running end to end on Core AI with a native Swift host:
[pocket-tts-swift](https://github.com/RahulRachuri/pocket-tts-swift). The model is an
autoregressive flow-matching LM over Mimi audio latents at 12.5 Hz, a one-step flow
decoder, and a streaming Mimi decoder to 24 kHz PCM. Eight shipped voices via
conditioning embeddings from [kyutai/tts-voices](https://huggingface.co/kyutai/tts-voices).

The GAP/EDGE sentence: Apple's stock stack has no natural multi-voice neural TTS of this
class, MLX runs this model faster on a Mac (10.5x against this port's 8.5x) but has no
iPhone path, so this port's case is the phone, where the fully gated pipeline holds
7.8x real time in a 169 MB peak footprint from a native Swift host.

## Pipeline

```
text ──(host: native sentencepiece + chunker, hard cap voice_pos+n_text+max_gen_len<=512)──▶ token ids
  1. flowlm_*.aimodel      two functions over one shared rank-5 KV state (S_MAX=512, derived):
       prefill(text_emb[1,16,1024], pos[1] i32) -> cond[1,1024]        windowed, pos carried
       step(latent_in[1,1,32], is_bos[1], pos[1] i32) -> cond, eos_logit
  2. flow_decoder_*.aimodel  flow(cond[1,1024], noise[1,32]) -> latent[1,32]   one-step (lsd1), stateless
     host: EOS compare (logit > -4.0), seeded torch-exact noise, AR loop
  3. mimi_decoder_*_q_gs.aimodel  latent[1,32] -> pcm[1,1,1920]
     rescale + k=1 quantizer folded in-graph; 12 streaming-state tensors held as
     in-graph Core AI state, never reset across chunks (structural, not conventional)
audio: chunks concatenated directly, no crossfade (gated: max join step 0.0246 vs in-audio p99.9 0.1878)
```

The KV cache has no context window, so capacity is derived from the model's own bound:
worst legal chunk is 162 voice + 50 text + 234 generated frames = 446, so S_MAX = 512.
Upstream's chunker overshoots its own 50-token cap (observed max 121 tokens), so the
Swift host enforces the cap by whitespace-splitting the remainder, plus a runtime
precondition in the AR loop. This checkpoint generates 2.47 frames per text token on
average, which is why a fixture-sized S_MAX = 256 silently truncated real sentences and
only an end-to-end sweep could see it.

## Gates

Per-graph, teacher-forced on the oracle's own latents (fp32, S=512 assets):

| gate | cpu_only | gpu |
|---|---|---|
| flow-LM step + prefill | cos 1.000000, max abs 3.0e-6, 0 EOS flips | cos 1.000000, 3.9e-6, 0 flips |
| flow decoder | cos 1.000000, max abs 3.34e-6 | cos 1.000000, 1.43e-6 |
| Mimi decoder (`_q_gs`) | cos 1.000000, max abs 0.0 (bit-identical) | cos 1.000000, 0.0 |

End to end, free-running, oracle seed and noise protocol: framing identical to the
PyTorch oracle (14 tokens to 32 steps, EOS at step 28, 31 frames), wav cos 1.000000,
max abs 1e-4. ASR round trip through the sibling parakeet-swift port: 0.00 % WER on the
oracle prompt, 1.38 % (2/145) on a 148-word paragraph.

Corpus sweep, 302 sentences across all eight voices: 4.27 % WER, 0 capped chunks, 0
missing EOS, 0 clipped clips. The short-utterance tail (1 to 6 words, about 19 % WER) is
upstream's, reproduced at the same rate by pure PyTorch (19.0 % port against 20.0 %
upstream, seed-matched); on matched rows longer than 6 words the two are 2.22 % against
1.89 %. Ten-minute long-form run (2183 words, 97 chunks): 2.53 % WER, drift flat
(+0.09 pp per decile against 4.7 pp decile spread), all 96 stitch points below the
in-audio p99.9 step, no crossfade needed.

Device transfer (iPhone 17 Pro Max, A19 Pro, iOS 27 beta 5 (24A5408d), Release): under
`cpuOnly` all 8 per-graph probe dumps are bit-identical to the M4 Pro, including post-run KV and Mimi
state. On gpu the oracle-prompt wav scores cos 1.000000, max abs 4.3e-5, and the ASR
numbers reproduce the Mac's exactly (0.00 % and 1.38 %). fp16 does not transfer bit-wise
across chips (A19 Pro against M4 Pro), and gpu fp32 is not bit-transferable across GPU
architectures either; the strict transfer property belongs to `cpuOnly`, and gpu
correctness is carried by the oracle-prompt cosine, the framing, and the ASR gate.

## Speed

Mac, M4 Pro, macOS 27 Golden Gate beta 5 (26A5406e), 148-word paragraph, voice alba, load
excluded, 1 warmup + 3 timed, medians.
The quiet-machine ladder (M1-era assets, fresh subprocess per run):

| stack | RTF | x realtime |
|---|---:|---:|
| upstream PyTorch (mps) | 0.2027 | 4.9x |
| this port, fp32 (gpu) | 0.1663 | 6.0x |
| this port, fp16 (gpu) | 0.1562 | 6.4x |
| pocket-tts-mlx 0.2.1 (metal) | 0.0953 | 10.5x |

The published M2 assets (in-graph Mimi state) on the same protocol, ambient machine:
fp32 0.1281 (7.8x), fp16 0.1171 (8.5x). The fp16 gap to MLX is 1.23x, and 58 % of the
remaining wall is the flow-LM step graph itself. Honest framing: faster than upstream
PyTorch on the same Mac, behind Metal-native MLX; the case for this port is the phone
and the Swift host, not Mac headline speed.

Device, iPhone 17 Pro Max (A19 Pro), iOS 27 beta 5 (24A5408d), Release build, JIT
`.aimodel`, charging, thermal nominal before and after every run, same paragraph and protocol:

| config | RTF median | x realtime | peak RSS | load |
|---|---:|---:|---:|---:|
| fp32 gpu | 0.1646 | 6.1x | 202 MB | 0.7 s |
| fp16 gpu | 0.1281 | 7.8x | 169 MB | 0.4 s |
| fp32 cpuOnly (spot check) | 0.5324 | 1.9x | 225 MB | 0.02 s |

Those two beta 5 builds are load-bearing, and more than they should be. On an iPhone 17
Pro running an earlier iOS 27 build (24A5380h), these bundles do not load at all: SIGSEGV
inside `MPSGraph GPU::AssignVariableOpHandler` during `GPURegionRuntime::initializeOps()`
under `.gpu`, `failedToSpecialize` under `.cpuOnly`, fp32 and fp16 alike, and the same
after AOT-compiling them for h18p locally. Re-exporting this recipe with a different
toolchain reproduces the other half of it: the two graphs that carry in-graph state
(`flowlm`, `mimi`) then fail to load even on a Mac with `AIModelError error 1`, while the
stateless `flow_decoder` exports and gates clean on both cpu and gpu. So loadability of a
stateful bundle appears coupled to the OS and SDK build pair rather than to anything in
the recipe, and the builds above are a floor rather than a note on provenance. Both halves
of that were reported by @john-rocky.

gpu is 3.2x faster than cpuOnly on device, measured rather than assumed from the Mac.
Warmup (first-call specialization) is about 0.3 s on device against about 4.5 s on the
Mac. fp16 keeps the fp32 Mimi decoder throughout; the flow-LM and flow decoder carry the
precision split.

For a same-device comparison, FluidInference's `pocket-tts-coreml` was run through their
own FluidAudio SDK (0.15.5) on the same phone with default settings (their real shipping
configuration: fp16, gpu placement, Mimi pinned to cpuOnly by their loader) on a matched
~150-word passage through their whole-utterance API. Their route measures RTF 0.399
(2.51x realtime, median of 3); this port's fp16 path is about 3.1x faster on the same
hardware. Two protocol notes for fairness: the passage is matched in length but not
byte-identical to the paragraph above, and their internal chunking runs inside the single
timed call. MLX is not a comparator on iOS: no MLX-Swift port of pocket-tts exists, so
Core ML and Core AI are the only phone routes for this model today.

## The port in five Core AI bugs

All five are verified with minimal repros, and all five are filed with Apple: FB24322424 (1),
FB24322437 (2), FB24322585 (3), FB24322596 (4), FB24322605 (5). Repros are available on request.
**FB24322585 (3) is fixed in coreai-torch 0.4.2.** The other four still reproduce there.

1. **ConvTranspose1d is numerically wrong on the `cpu_only` delegate** (the same asset
   is correct on gpu). **The trigger is wider than this card first claimed.** It was
   written up as stride >= 8 *and* kernel >= 16; a kernel x stride grid on 0.4.2 puts it
   at kernel >= 8 at any stride, stride 1 included, or stride >= 16 at any kernel. So
   k=8/s=1 fails and k=2/s=16 fails, while k=4/s=8 is clean. A nonzero `output_padding`
   escapes it. Mimi's k=32/s=16/groups=512
   upsample hit it at cos -0.01. Workaround: at T=1 with groups == channels the op
   degenerates to a per-channel outer product (`out[c,:] = x[c,0] * W[c,0,:]`), and
   re-authoring it that way takes `cpu_only` to bit-exact. T=1 is what makes that escape
   available, not a condition of the defect: T=1, T=4 and T=64 all fail.
2. **The Python `coreai.runtime` bindings leak one IOSurface per call.** With this
   pipeline's state sizes the process dies at about 2,250 calls, climbing 1.9 MB per
   call to about 3 GB. Every Python harness runs generation in subprocess workers on a
   1,500-call budget. The Swift `CoreAI` framework does not leak: one process ran 4,773
   calls flat at 300 MB.
3. **coreai-torch lowered ConvTranspose `output_padding` by padding the input**, which
   silently produced the wrong output length. **Fixed in coreai-torch 0.4.2**: a 66-case
   sweep gives 41 wrong lengths on 0.4.1 and 0 on 0.4.2. The streaming rewrite here never
   uses `output_padding`, so the port was unaffected either way.
4. **fp32 graphs with in-graph state abort the process under the default
   `SpecializationOptions`** (SIGABRT in ANERegionFormationPass) instead of falling back.
   Since both main graphs carry in-graph state, every load states an explicit `gpu` or
   `cpuOnly` preference, and the host refuses the aborting pairs in code rather than by
   convention. ANE work is parked behind this bug.
5. **`preferredComputeUnitKind: .cpu` silently returns wrong numerics**; the same asset
   is correct with `.cpuOnly` and with `.gpu`. Parity work must use `.cpuOnly`, and
   `.cpu` should not appear in a host at all. **The trigger is a preference expressed over
   a heterogeneous allowed set, not an op.** `.gpu` and `.cpu` report identical
   `allowedComputeUnitKinds` of `[cpu, gpu, neuralEngine]` and differ only in which unit is
   preferred; `.cpuOnly` collapses that set to one and is exact. It scales with how much
   graph there is to partition — a 6-layer prefill over 16 positions and a multi-stage
   convolutional decoder both fail, the same transformer at 1 position and a small stateless
   function are correct. What settles it as placement rather than arithmetic: under `.cpu`
   the first four samples of a 1920-sample frame are bit-identical to `.gpu` and `.cpuOnly`,
   and the divergence starts later in that same frame — precision loss does not leave a
   bit-identical prefix. Blast radius here is cos mean 0.501 / min 0.222 with 27 of 32 EOS
   decisions flipped, on fp32 assets. Bug 4 may be the same mechanism: it fires under the
   same condition and also disappears under `.cpuOnly`.

## Bundle

Five `.aimodel` bundles, hosted on the contributor's Hugging Face
([rahulrachuri/pocket-tts-coreai](https://huggingface.co/rahulrachuri/pocket-tts-coreai)),
CC-BY-4.0 inherited from the model:

| bundle | size | role |
|---|---:|---|
| `flowlm_float32_s512.aimodel` | 302.4 MB | flow-LM prefill + step, parity config |
| `flowlm_float16_s512.aimodel` | 151.3 MB | flow-LM, device config |
| `flow_decoder_float32_lsd1.aimodel` | 39.1 MB | one-step flow decoder |
| `flow_decoder_float16_lsd1.aimodel` | 19.6 MB | flow decoder, device config |
| `mimi_decoder_float32_ring272_outer_q_gs.aimodel` | 41.3 MB | Mimi decoder, fp32 always |

The host additionally needs Kyutai's own files, fetched from Kyutai's repos and not
redistributed here: `model.safetensors` (embedding LUT, rescale constants, quantizer),
`tokenizer.model`, and the per-voice embeddings from
[kyutai/tts-voices](https://huggingface.co/kyutai/tts-voices).

Convert yourself with [`conversion/pocket-tts/`](../../conversion/pocket-tts/): generate
the oracle capture first (`gen_oracle.py --tag orc_a`), then run the three exporters per
`recipe.toml`. Each exporter gates its graph against the oracle (eager, `cpu_only`, and
`gpu`) before writing the bundle. The full gate ladder, the corpus sweep harness, and
the device probes live in the host repo.
