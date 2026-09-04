# Kokoro-82M (StyleTTS2) on Core AI: variable length on a fixed-shape engine, and why it runs on the CPU

Kokoro-82M (StyleTTS2 + iSTFTNet, 82M, 24 kHz) ported to Core AI as the zoo's first **text-to-speech**
model — type English text on an iPhone, hear it ~1–2 s later, fully offline, bit-identical to the Mac.
Kokoro is **non-autoregressive**: phonemes + a voice vector (`ref_s`) go in, a whole waveform comes
out in one pass (no token-by-token decode). The catch is that it's a tangle of LSTMs, InstanceNorms,
a baked-in complex-STFT vocoder, and one data-dependent length — none of which loves a fixed-shape
engine. It runs on the **CPU** (sibling: [compute-units-and-authoring]; companion audio
*understanding* port: [qwen2.5-omni-audio-understanding], same `coreai-audio` app, "Understand" tab).

## The split: 3 bundles around the one data-dependent length

There is exactly one data-dependent length — the **duration → alignment expansion** (`L = Σ pred_dur`
upsamples phonemes to frames). Cut the graph there, into 3 `.aimodel` bundles + host DSP:

```
text ──(misaki G2P, host)──▶ phoneme ids
  1. predictor.aimodel : ids, ref_s, attn_mask → duration, d, t_en
  host: pred_dur = round(duration);  alignment aln[1,T,L] + frame_mask
  2. prosody.aimodel   : d, t_en, aln, ref_s, frame_mask → asr, F0, N
  host: har = STFT(SineGen(f0_upsamp(F0)))        ← hn-nsf excitation (windowed FFT)
  3. vocoder.aimodel   : asr, F0, N, har, ref_s, frame_mask → waveform
  host: trim to L·600
```

Bundles are **voice-independent** (the voice is the `ref_s` input, `pack[len(ids)-1]`). Token length
T and frame length L are **fixed buckets** (default 128 / 512); the host pads to the bucket and trims
the output. Long text is split per sentence (like the upstream `KPipeline`). fp32, ~335 MB total.

## weight_norm: non-determinism → suspect weight loading first (the day-eater)

Every `build_model()` produced a **non-deterministically exploding** output (range 0.3, then 3.4M,
then 0.8…). Cause: kokoro uses the **old hook-based `torch.nn.utils.weight_norm`**, where
`module.weight` stays at its **random init value until a forward fires the hook**. The custom conv
kernels (below) read `.weight` *directly* → they were reading random init, not the loaded weights.

```python
from torch.nn.utils import remove_weight_norm
for m in model.modules():
    try: remove_weight_norm(m)   # fold weight_g/v into a plain .weight
    except (ValueError, RuntimeError): pass
```

This one line made every other symptom disappear and snapped fidelity to magspec-corr 0.999. Hours
were lost chasing an unrelated "STFT phase" red herring first. **Lesson: non-determinism → suspect
weight loading before anything else.**

## 6 bi-LSTMs → masked unrolled bi-LSTM

StyleTTS2 has **6 bidirectional LSTMs**. `torch.export` specializes `nn.LSTM`'s sequence length to a
constant (dynamic-length export fails) → fixed buckets + host padding. But a right-pad into a
fixed-length LSTM **poisons the backward pass** — the pad leaks into the reverse direction and
corrupts every real token (audio corr collapses to 0.02).

Fix = a **masked unrolled bi-LSTM**: at pad positions, carry the state instead of updating it, so the
backward pass behaves as if it started fresh at the last real token.

```python
h = m*h2 + (1-m)*h   # m: 1=real, 0=pad → pad carries state (skipped)
c = m*c2 + (1-m)*c
```

Bit-identical to `nn.LSTM` at full length; exact on the real-token span when padded (maxd 0).

## 58 AdaIN InstanceNorms → frame-masked

The decoder's `AdaIN1d` computes **InstanceNorm over the L (time) axis**. Padding to the bucket lets
**pad frames poison the mean/variance** → real-frame normalization is wrong (synthesis corr 0.13).
Fix = a **frame-masked InstanceNorm**: take mean/var over **real frames only**, zero the pad frames in
the output (so the next conv sees the engine's own zero-pad). Resize the mask with
**`arange < real·N/Lb`**, *not* `F.interpolate(nearest)` — the engine's nearest rounding differs from
torch's and shifts the mask. 0.13 → 0.98.

## The hn-nsf STFT phase + Core AI op traps

- **STFT phase (atan2) flips 2π** near the `F0 → 0` pad boundary on the fp32 engine. This is the
  classic on-device-TTS trap; existing CoreML ports also compute this **one windowed FFT on the
  host**. So `compute_har` (f0_upsamp + SineGen + STFT) runs host-side (torch/Swift, fp32-stable);
  engine output then matches torch (magspec 0.9994).
- **`ConvTranspose1d`** → both uses are rewritten as **zero-insertion + `conv1d`** (insert stride-1
  zeros + plain conv with the flipped kernel) → bit-exact and correct on the engine. **The two
  symptoms this was written for are gone; the rewrite still has to stay.** As of coreai-torch
  0.4.1/0.4.2 the `output_padding` case (k3 s2 p1 op1) converts and runs clean on gpu and cpu_only
  to 2.4e-07 through a downstream concat, and the iSTFT case (stride 5, 11→1, k=20) returns real
  audio rather than zeros. The iSTFT case still comes out wrong on **cpu_only**
  (max|Δ| 4.86 against torch, correct on gpu at 4.8e-07): this port ships
  `GraphModel(computeUnits: .cpu)`, so `_manual_convT_general` is holding that. The cause is
  **FB24322424**, still open: ConvTranspose on `cpu_only` is wrong at kernel ≥ 8 at any stride, or
  stride ≥ 16 at any kernel.
- `aten.atan2` unsupported → half-angle `2·atan(y / (√(x²+y²) + x))`. `aten.remainder` (`%1`) is
  identity in the audio range → removed. `input_ids` must be **int32**.

## Why the CPU (the counter-intuitive bit)

| | GPU | CPU |
|---|---|---|
| masked bi-LSTM (len 512) | 42 ms | **8 ms** |
| full synthesis (1 utterance) | ~5 s | **~0.7 s (Mac) / 1–2 s (iPhone)** |

Because the LSTMs are **unrolled**, inference is "hundreds of tiny sequential ops," which is
**dispatch-bound** on the GPU. The CPU has no per-op dispatch overhead and wins outright. **Small
model + lots of sequential ops ⇒ CPU.** "GPU = fast" is not a law. Drive it with
`GraphModel(computeUnits: .cpu)`.

## On-device G2P (free text)

Free-text reading needs grapheme→phoneme on device: **MisakiSwift** (mlalma; dictionary + neural
fallback, English).

```swift
let (phonemes, _) = EnglishG2P(british: false).phonemize(text: text)
let ids = [0] + phonemes.compactMap { vocab[String($0)] } + [0]   // split per sentence, synth, concat
```

- **iOS embed trap:** MisakiSwift is a `.dynamic` SPM product (MLX dependency) → iOS **dyld-crashes**
  (`MisakiSwift.framework: no such file`) unless explicitly embedded. Mac's `swiftStdLibTool` embeds
  it automatically; iOS does not. In xcodegen, add `embed: true` + `link: true` to create the "Embed
  Frameworks (CodeSignOnCopy)" phase (MLX itself links static).
- **OOV → silence:** upstream G2P is misaki + an **espeak fallback**. With the fallback off, an
  out-of-dictionary word ("Kokoro") became `❓` → dropped token → silence. The model was faithful;
  only the G2P config was wrong. Read it as `kəkˈɔɹO` (espeak-compatible) to fix.

## Memory / device + Net

iPhone 17 Pro (A19 Pro), CPU: the app auto-downloads the 3 bundles (321 MB) from HF on first run.
Demo clip: 4.5 s of audio in ~1.8 s. **Free text** "The weather is lovely today, so let's go for a
walk in the park." → 68 phonemes on-device → **4.6 s of speech in 1202 ms**, **bit-identical to the
Mac (wav-corr 1.0000)**, magspec-corr 0.999 vs the reference. A deterministic CPU pipeline ⇒
device-independent output.

The takeaways that generalize: **non-determinism → suspect weight loading (weight_norm fold) first**;
**sequential-op-heavy small models run fastest on the CPU**; **variable length rides a fixed-shape
engine via masked-unrolled LSTMs + frame-masked InstanceNorm + a host-side STFT**; and **iOS
`.dynamic` SPM frameworks must be explicitly embedded.**
