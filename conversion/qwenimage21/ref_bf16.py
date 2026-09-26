"""The bf16 diffusers pipeline (the way the model is normally run) vs the fp32 oracle:
same prompt / seed / noise0; PSNR of the white-composited RGB and final-latent corr.
This is the fidelity band the bf16 reference itself lives in — a port cannot be held tighter."""
import argparse, sys, json, time
from pathlib import Path
import numpy as np, torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--oracle", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cpu")
args = ap.parse_args()
o = Path(args.oracle); meta = json.load(open(o / "meta.json"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from _paths import hf_snapshot  # noqa: E402
snap = hf_snapshot("Qwen/Qwen-Image-2.1", revision=meta["snapshot"])
from diffusers import QwenImage21Pipeline
t0 = time.time()
pipe = QwenImage21Pipeline.from_pretrained(snap, torch_dtype=torch.bfloat16).to(args.device)
pipe.set_progress_bar_config(disable=True)
print(f"loaded bf16 on {args.device} {time.time()-t0:.0f}s", flush=True)
N, size = meta["N"], meta["size"]
noise0 = torch.from_numpy(np.fromfile(o / "noise0.f32", "<f4")).reshape(1, N, 64)
traj = []
real_step = pipe.scheduler.step
def step_hook(mo, t, s, *a, **k):
    r = real_step(mo, t, s, *a, **k); traj.append(r[0].detach().float().cpu().clone()); return r
pipe.scheduler.step = step_hook
t0 = time.time()
with torch.no_grad():
    img = pipe(prompt=meta["prompt"], height=size, width=size, num_inference_steps=meta["steps"],
               latents=noise0.to(args.device), generator=torch.Generator("cpu").manual_seed(meta["seed"]),
               true_cfg_scale=1.0).images[0]
print(f"gen {time.time()-t0:.0f}s ({(time.time()-t0)/meta['steps']:.2f} s/step incl. overhead)", flush=True)
Path(args.out).mkdir(parents=True, exist_ok=True)
img.save(Path(args.out) / "image.png")
white = Image.new("RGB", img.size, (255, 255, 255)); white.paste(img, mask=img.getchannel("A")); white.save(Path(args.out) / "image_rgb.png")
ref = np.asarray(Image.open(o / "image_ref_rgb.png").convert("RGB")).astype(np.float32)
im = np.asarray(white).astype(np.float32)
mse = float(((im - ref) ** 2).mean()); psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else 99.0
fl = np.fromfile(o / "final_latents.f32", "<f4").reshape(1, N, 64)
corr = float(np.corrcoef(traj[-1].numpy().ravel(), fl.ravel())[0, 1])
rels = []
for s in range(len(traj) - 1):
    ref_l = np.fromfile(o / f"latent_{s+1}.f32", "<f4").reshape(1, N, 64)
    rels.append(float(np.linalg.norm(traj[s].numpy() - ref_l) / np.linalg.norm(ref_l)))
print(f"RESULT bf16-{args.device} size={size} PSNR_rgb={psnr:.2f} dB final_latent_corr={corr:.6f} "
      f"latent_rel_err step5={rels[4]:.2e} step10={rels[9]:.2e} step20={rels[19]:.2e} step38={rels[-1]:.2e}", flush=True)
