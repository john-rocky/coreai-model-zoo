"""Qwen-Image-2.1 VAE decoder graph for the Core AI export — a wrapper over the diffusers-main class.

``AutoencoderKLQwenImage21`` (diffusers main @ 4295ee3) is used as-is (``.venv-qi21``); the wrapper
only fixes what the pipeline does around ``vae.decode`` for text-to-image, batch 1, one frame:

  latents_packed [1,N,64]   the sampler's output (normalised space, the DiT token layout)
  -> unpack                 ``transpose(1,2).reshape(1,64,1,h,w)``  (``QwenImage21Pipeline._unpack_latents``)
  -> ``* std + mean``       the config's 64 ``latents_std`` / ``latents_mean``, baked as constants
  -> ``post_quant_conv`` -> ``decoder(first_chunk=True)``  one frame, no feature cache
  -> ``clamp(-1, 1)`` -> ``[:, :, 0]``
  -> image [1,4,H,W] fp32, RGBA in [-1,1]  (H = 16h)

Feed it the raw sampler latent: the un-normalisation is inside the graph (Z-Image's Swift port
lost 18 dB by doing it twice).

Why no feature cache: for the first (only) frame every ``QwenImage21CausalConv3d`` gets
``cache_x=None`` either way, and ``QwenImage21Resample('upsample3d')`` skips ``time_conv`` both when
the cache is ``None`` and on the first-chunk ``"Rep"`` path. ``first_chunk=True`` is NOT optional:
``QwenImage21DupUp3D`` doubles the time axis in the three temporal up-blocks and only the
first-chunk slice brings it back to one frame (``parity_vae_torch.py`` checks both, the second as
a red control).

``patch_nearest_upsample`` swaps the decoder's ``nearest-exact`` 2x upsample for
``repeat_interleave`` (identical for an integer factor of 2: source index ``floor((i + 0.5) / 2)
= i // 2``), the Z-Image ``_patch_nearest_upsample`` move for this class.
"""
from __future__ import annotations

import torch
import torch.nn as nn

MODEL = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"


def load_vae(snapshot=None, dtype=torch.float32):
    from pathlib import Path

    from diffusers import AutoencoderKLQwenImage21
    if snapshot is None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from _paths import hf_snapshot
        snapshot = hf_snapshot(MODEL, revision=REVISION)
    return AutoencoderKLQwenImage21.from_pretrained(str(snapshot), subfolder="vae", torch_dtype=dtype).eval()


def patch_nearest_upsample(module: nn.Module) -> int:
    """Replace every ``QwenImage21Upsample`` (nearest-exact, x2) forward with ``repeat_interleave``."""
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import QwenImage21Upsample
    n = 0
    for mod in module.modules():
        if isinstance(mod, QwenImage21Upsample):
            assert mod.mode == "nearest-exact" and tuple(mod.scale_factor) == (2.0, 2.0), (mod.mode, mod.scale_factor)
            mod.forward = lambda x: x.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
            n += 1
    return n


class QI21VAEDecode(nn.Module):
    """``latents_packed [1,h*w,64]`` -> ``image [1,4,16h,16w]`` (fp32, [-1,1]) for one fixed latent size."""

    def __init__(self, vae, lat_h: int, lat_w: int):
        super().__init__()
        self.vae = vae
        self.h, self.w = int(lat_h), int(lat_w)
        z = int(vae.config.z_dim)
        # Same construction as the pipeline: torch.tensor(config list) -> view -> vae dtype.
        dt = next(vae.parameters()).dtype
        self.register_buffer("mean", torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(dt),
                             persistent=False)
        self.register_buffer("std", torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(dt),
                             persistent=False)

    def unnormalize(self, latents_packed: torch.Tensor) -> torch.Tensor:
        b, n, c = latents_packed.shape
        z = latents_packed.transpose(1, 2).reshape(b, c, 1, self.h, self.w)
        return z * self.std + self.mean

    def forward(self, latents_packed: torch.Tensor) -> torch.Tensor:
        z = self.unnormalize(latents_packed)
        x = self.vae.decoder(self.vae.post_quant_conv(z), first_chunk=True)
        return torch.clamp(x, min=-1.0, max=1.0)[:, :, 0]
