"""Chatterbox TTS fp32 oracle — capture configs + intermediates for the per-net Core AI gates.
Run in .venv-chatterbox. Default voice (conds.pt), fixed text, seeded. Saves:
  text_tokens, t3_cond (speaker_emb / cond_prompt_speech_tokens / emotion_adv),
  speech_tokens (T3 AR output), and the reference 24 kHz wav.
Also dumps the T3 hp config + component module trees for the export."""
import inspect
import os
import tempfile
import numpy as np
import torch
import torchaudio

from chatterbox.tts import ChatterboxTTS

# perth (inaudible output watermark) is a post-process irrelevant to the port; its
# implicit watermarker fails to load on this setup -> shim it to a pass-through.
import perth
class _NoWatermark:
    def apply_watermark(self, wav, sample_rate=None, **k):
        return wav
perth.PerthImplicitWatermarker = _NoWatermark

OUT = os.path.join(os.environ.get("SCRATCH") or os.path.join(tempfile.gettempdir(), "chatterbox_scratch"), "chatterbox_oracle")
TEXT = "The capital of France is Paris."
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

model = ChatterboxTTS.from_pretrained(device="cpu")
print("loaded ChatterboxTTS | sr:", model.sr)

# --- configs for the export ---
hp = model.t3.hp
print("=== T3 hp ===")
for k in ("start_text_token", "stop_text_token", "start_speech_token", "stop_speech_token",
          "n_channels", "text_tokens_dict_size", "speech_tokens_dict_size", "max_text_tokens",
          "max_speech_tokens", "llama_config_name", "input_pos_emb", "speech_cond_prompt_len"):
    print(f"  {k} = {getattr(hp, k, '<none>')}")
print("=== t3.inference signature ===", inspect.signature(model.t3.inference))
print("=== s3gen.inference signature ===", inspect.signature(model.s3gen.inference))
print("=== T3 tfmr type ===", type(model.t3.tfmr).__name__)

# --- default conditioning (the packaged default voice) ---
cond = model.conds.t3
print("=== T3Cond shapes ===",
      "speaker_emb", tuple(cond.speaker_emb.shape),
      "| cond_prompt_speech_tokens",
      None if cond.cond_prompt_speech_tokens is None else tuple(cond.cond_prompt_speech_tokens.shape),
      "| emotion_adv", tuple(cond.emotion_adv.shape))

# --- text tokens (mirror generate) ---
from chatterbox.tts import punc_norm
import torch.nn.functional as F
text = punc_norm(TEXT)
text_tokens = model.tokenizer.text_to_tokens(text)
sot, eot = hp.start_text_token, hp.stop_text_token
tt = F.pad(text_tokens, (1, 0), value=sot)
tt = F.pad(tt, (0, 1), value=eot)
print("=== text_tokens ===", tt.tolist())

# --- run generate (seeded) end-to-end; capture speech_tokens via a hook on s3gen.inference ---
captured = {}
orig_s3 = model.s3gen.inference
def spy_s3(*a, **k):
    st = k.get("speech_tokens", a[0] if a else None)
    captured["speech_tokens"] = st.detach().cpu().clone()
    out = orig_s3(*a, **k)
    return out
model.s3gen.inference = spy_s3

torch.manual_seed(SEED)
wav = model.generate(TEXT, temperature=0.8, cfg_weight=0.5, exaggeration=0.5)
wav = wav.detach().cpu()
print("=== wav ===", tuple(wav.shape), "sr", model.sr,
      "| speech_tokens", tuple(captured["speech_tokens"].shape))

torchaudio.save(OUT + ".wav", wav, model.sr)
np.savez(
    OUT + ".npz",
    text_tokens=tt.cpu().numpy().astype(np.int64),
    speaker_emb=cond.speaker_emb.detach().cpu().numpy().astype(np.float32),
    cond_prompt_speech_tokens=(cond.cond_prompt_speech_tokens.detach().cpu().numpy().astype(np.int64)
                               if cond.cond_prompt_speech_tokens is not None else np.array([], np.int64)),
    emotion_adv=cond.emotion_adv.detach().cpu().numpy().astype(np.float32),
    speech_tokens=captured["speech_tokens"].numpy().astype(np.int64),
    wav=wav.numpy().astype(np.float32),
    sr=np.array(model.sr),
)
print("saved:", OUT + ".{wav,npz}")
