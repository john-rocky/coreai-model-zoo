"""Host sampler for Qwen-Image-2.1 — ``FlowMatchEulerDiscreteScheduler`` as the pipeline drives it, numpy only.

What ``QwenImage21Pipeline.__call__`` + ``scheduler.set_timesteps(sigmas=..., mu=...)`` + ``step`` compute
for text-to-image (no CFG), with the checkpoint's ``scheduler_config.json`` (rev 790c926):

  sigmas   = linspace(1, 1/steps, steps)               (float64, then cast to fp32 as set_timesteps does)
  mu       = calculate_shift(N, 256, 8192, 0.5, 0.9)   (N = image tokens = (size/16)^2)
  sigmas   = exp(mu) / (exp(mu) + (1/sigmas - 1))      (time_shift_type "exponential", sigma 1.0)
  sigmas   = 1 - (1 - sigmas) / ((1 - sigmas[-1]) / (1 - 0.02))   (stretch_shift_to_terminal, 0.02)
  timesteps = sigmas * 1000;  sigmas += [0]
  the transformer sees  t = timesteps[i] / 1000        (fp32, as the pipeline divides)
  step:  x_{i+1} = x_i + (sigmas[i+1] - sigmas[i]) * v_i   (fp32)

Every operation is fp32 with Python scalars kept weak, which is what numpy >= 2 (NEP 50) and the
reference both do, so the arrays match the reference bit for bit (gated below).

Gate (``python qi21_sched.py``, any venv with numpy): against ``oracle/<size>/`` — ``sigmas.f32`` and
``timesteps.f32`` (bar max|d| <= 1e-6), the transformer ``t`` in ``meta.json``, and the Euler step
teacher-forced on every recorded step (``latent_s`` + ``vel_s`` -> ``latent_{s+1}``, the last one ->
``final_latents``). Also checks the constants below against the snapshot's ``scheduler_config.json``
when the snapshot is present.
"""
from __future__ import annotations

import math

import numpy as np

# scheduler/scheduler_config.json of Qwen/Qwen-Image-2.1 @ 790c926
SCHED_CFG = dict(base_image_seq_len=256, max_image_seq_len=8192, base_shift=0.5, max_shift=0.9,
                 num_train_timesteps=1000, shift_terminal=0.02, time_shift_type="exponential",
                 use_dynamic_shifting=True, invert_sigmas=False, stochastic_sampling=False, shift=1.0)


def calculate_shift(n_tokens: int, base_seq_len: int = 256, max_seq_len: int = 8192,
                    base_shift: float = 0.5, max_shift: float = 0.9) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return n_tokens * m + b


def schedule(steps: int, n_tokens: int, cfg: dict = SCHED_CFG) -> tuple[np.ndarray, np.ndarray, float]:
    """-> (sigmas [steps+1] fp32 ending in 0, timesteps [steps] fp32, mu)."""
    mu = calculate_shift(n_tokens, cfg["base_image_seq_len"], cfg["max_image_seq_len"], cfg["base_shift"],
                         cfg["max_shift"])
    s = np.linspace(1.0, 1 / steps, steps).astype(np.float32)
    e = math.exp(mu)
    s = e / (e + (1 / s - 1) ** 1.0)
    one_minus = 1 - s
    scale = one_minus[-1] / (1 - cfg["shift_terminal"])
    s = 1 - (one_minus / scale)
    s = s.astype(np.float32)
    timesteps = s * np.float32(cfg["num_train_timesteps"])
    sigmas = np.concatenate([s, np.zeros(1, np.float32)])
    return sigmas, timesteps.astype(np.float32), mu


def model_t(timesteps: np.ndarray, i: int) -> np.ndarray:
    """The ``timestep`` the transformer receives at step ``i``: ``[1]`` fp32 = timesteps[i] / 1000."""
    return (timesteps[i:i + 1].astype(np.float32) / np.float32(1000)).astype(np.float32)


def step(x: np.ndarray, v: np.ndarray, sigmas: np.ndarray, i: int) -> np.ndarray:
    """One FlowMatch Euler step in fp32: ``x + (sigmas[i+1] - sigmas[i]) * v``."""
    dt = np.float32(sigmas[i + 1]) - np.float32(sigmas[i])
    return (x.astype(np.float32) + dt * v.astype(np.float32)).astype(np.float32)


def _gate() -> int:
    import argparse
    import json
    import sys
    from pathlib import Path

    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", nargs="+", default=[str(here / "oracle" / "256"), str(here / "oracle" / "512")])
    ap.add_argument("--json", default=str(here / "_work" / "gate_sched.json"))
    args = ap.parse_args()
    bar = 1e-6
    res, ok = {}, True
    try:
        sys.path.insert(0, str(here.parent))
        from _paths import hf_snapshot
        snap = Path(hf_snapshot("Qwen/Qwen-Image-2.1", revision="790c92633540aa0cb11d9abf19eb46d861714758"))
        cfg = json.load(open(snap / "scheduler" / "scheduler_config.json"))
        diff = {k: (v, cfg.get(k)) for k, v in SCHED_CFG.items() if cfg.get(k) != v}
        res["config_matches_snapshot"] = not diff
        ok &= not diff
        print(f"[sched] constants vs snapshot scheduler_config.json: {'identical' if not diff else diff}", flush=True)
    except Exception as e:  # snapshot absent: the oracle arrays still gate the constants
        print(f"[sched] snapshot config not checked ({type(e).__name__})", flush=True)
    for o in args.oracle:
        od = Path(o)
        meta = json.load(open(od / "meta.json"))
        steps, N = int(meta["steps"]), int(meta["N"])
        ref_sig = np.fromfile(od / "sigmas.f32", "<f4")
        ref_ts = np.fromfile(od / "timesteps.f32", "<f4")
        sig, ts, mu = schedule(steps, N)
        r = dict(N=N, steps=steps, mu=mu, mu_meta=meta.get("mu"),
                 sigmas_maxd=float(np.abs(sig.astype(np.float64) - ref_sig).max()),
                 sigmas_exact=bool(np.array_equal(sig, ref_sig)),
                 timesteps_maxd=float(np.abs(ts.astype(np.float64) - ref_ts).max()),
                 timesteps_exact=bool(np.array_equal(ts, ref_ts)))
        r["t_maxd_vs_meta"] = max(abs(float(model_t(ts, i)[0]) - meta["t"][i]) for i in range(steps))
        # Euler step, teacher-forced on every recorded step
        C = 64
        worst, n_exact = 0.0, 0
        for i in range(steps):
            x = np.fromfile(od / f"latent_{i}.f32", "<f4")
            v = np.fromfile(od / f"vel_{i}.f32", "<f4")
            nxt = od / (f"latent_{i + 1}.f32" if i + 1 < steps else "final_latents.f32")
            ref = np.fromfile(nxt, "<f4")
            y = step(x.reshape(1, N, C), v.reshape(1, N, C), sig, i).ravel()
            worst = max(worst, float(np.abs(y.astype(np.float64) - ref).max()))
            n_exact += bool(np.array_equal(y, ref))
        r["step_maxd"], r["step_bit_exact"] = worst, f"{n_exact}/{steps}"
        r["ok"] = (r["sigmas_maxd"] <= bar and r["timesteps_maxd"] <= bar and r["t_maxd_vs_meta"] <= 5.1e-8
                   and worst <= bar)
        ok &= r["ok"]
        res[od.name] = r
        print(f"[sched] {od.name}: N {N} steps {steps} mu {mu:.10f} (meta {meta.get('mu')})\n"
              f"  sigmas    max|d| {r['sigmas_maxd']:.3e} bit-exact {r['sigmas_exact']}\n"
              f"  timesteps max|d| {r['timesteps_maxd']:.3e} bit-exact {r['timesteps_exact']}\n"
              f"  model t (timesteps/1000) vs meta t (7 dp): max|d| {r['t_maxd_vs_meta']:.2e}\n"
              f"  Euler step teacher-forced, {steps} steps: max|d| {worst:.3e}, bit-exact {n_exact}/{steps}  "
              f"{'PASS' if r['ok'] else 'FAIL'}", flush=True)
    res["ok"] = bool(ok)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.json, "w"), indent=2)
    print(f"\n[sched] {'PASS' if ok else 'FAIL'} (bar: sigmas and timesteps max|d| <= {bar}, step <= {bar}) "
          f"-> {args.json}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_gate())
