"""Sensitivity of the fp32 REFERENCE pipeline itself: perturb prompt_embeds by a relative
Gaussian factor and re-run the same seed/noise; PSNR vs the unperturbed oracle image.
If the fp32 reference splits by ~10 dB under a 1e-5 perturbation, the trajectory is
intrinsically sensitive at that size/seed and a port cannot be held to a tighter bar."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--oracle", required=True)
ap.add_argument("--rel", type=float, default=1e-5)
ap.add_argument("--pseed", type=int, default=7)
ap.add_argument("--out", required=True)
args = ap.parse_args()
o = Path(args.oracle); meta = json.load(open(o / "meta.json"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from _paths import hf_snapshot  # noqa: E402
snap = hf_snapshot("Qwen/Qwen-Image-2.1", revision=meta["snapshot"])
from diffusers import QwenImage21Pipeline
t0 = time.time()
pipe = QwenImage21Pipeline.from_pretrained(snap, torch_dtype=torch.float32)
pipe.set_progress_bar_config(disable=True)
print(f"loaded {time.time()-t0:.0f}s", flush=True)
L, N, size = meta["L"], meta["N"], meta["size"]
pe = torch.from_numpy(np.fromfile(o / "prompt_embeds.f32", "<f4")).reshape(1, L, 4096)
g = torch.Generator("cpu").manual_seed(args.pseed)
pe_p = pe * (1 + args.rel * torch.randn(pe.shape, generator=g))
print(f"perturbation rel {args.rel}: |d|/|ref| {float((pe_p-pe).norm()/pe.norm()):.3e}", flush=True)
noise0 = torch.from_numpy(np.fromfile(o / "noise0.f32", "<f4")).reshape(1, N, 64)
traj = []
real_step = pipe.scheduler.step
def step_hook(mo, t, s, *a, **k):
    r = real_step(mo, t, s, *a, **k); traj.append(r[0].detach().clone()); return r
pipe.scheduler.step = step_hook
t0 = time.time()
with torch.no_grad():
    img = pipe(prompt_embeds=pe_p, height=size, width=size, num_inference_steps=meta["steps"],
               latents=noise0, generator=torch.Generator("cpu").manual_seed(meta["seed"]),
               true_cfg_scale=1.0).images[0]
print(f"gen {time.time()-t0:.0f}s", flush=True)
Path(args.out).mkdir(parents=True, exist_ok=True)
img.save(Path(args.out) / "image.png")
white = Image.new("RGB", img.size, (255, 255, 255)); white.paste(img, mask=img.getchannel("A")); white.save(Path(args.out) / "image_rgb.png")
ref = np.asarray(Image.open(o / "image_ref_rgb.png").convert("RGB")).astype(np.float32)
im = np.asarray(white).astype(np.float32)
mse = float(((im - ref) ** 2).mean()); psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else 99.0
fl = np.fromfile(o / "final_latents.f32", "<f4").reshape(1, N, 64)
corr = float(np.corrcoef(traj[-1].numpy().ravel(), fl.ravel())[0, 1])
rels = []
for s in range(len(traj)):
    if s + 1 < meta["steps"]:
        ref_l = np.fromfile(o / f"latent_{s+1}.f32", "<f4").reshape(1, N, 64)
        rels.append(float(np.linalg.norm(traj[s].numpy() - ref_l) / np.linalg.norm(ref_l)))
print(f"RESULT size={size} rel={args.rel} PSNR_rgb={psnr:.2f} dB final_latent_corr={corr:.6f} "
      f"latent_rel_err step5={rels[4]:.2e} step10={rels[9]:.2e} step20={rels[19]:.2e} step38={rels[-1]:.2e}", flush=True)
