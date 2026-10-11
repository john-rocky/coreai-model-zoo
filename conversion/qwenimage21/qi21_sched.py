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

Fixed grid (``schedule_fixed``, Qwen-Image-2.1-Turbo @ d65dbc9, diffusers >= 85c9fa4): the checkpoint
carries its own sigmas in ``model_index.json`` (``sample_sigmas``, 8 values) and the pipeline passes them
to ``set_timesteps`` instead of the linspace; ``num_inference_steps`` is ignored. Its
``scheduler_config.json`` has ``use_dynamic_shifting`` false and ``shift_terminal`` null, so ``mu`` is
computed by the pipeline but never read:

  sigmas   = fp32(sample_sigmas)                      (np.array(list).astype(float32), as set_timesteps does)
  sigmas   = shift * sigmas / (1 + (shift - 1) * sigmas)   (shift 1.0: the identity, bit for bit)
  sigmas   = stretch_shift_to_terminal(sigmas)        (only if shift_terminal is truthy; Turbo: null)
  timesteps = sigmas * 1000;  sigmas += [0]           (t and the Euler step as above)

Gate (``python qi21_sched.py``, any venv with numpy): against ``oracle/<size>/`` — ``sigmas.f32`` and
``timesteps.f32`` (bar max|d| <= 1e-6), the transformer ``t`` in ``meta.json``, and the Euler step
teacher-forced on every recorded step (``latent_s`` + ``vel_s`` -> ``latent_{s+1}``, the last one ->
``final_latents``). Also checks the constants below against the snapshot's ``scheduler_config.json``
when the snapshot is present. ``--turbo`` (alias ``--fixed``) gates ``schedule_fixed`` instead, plus the
oracle meta's ``sample_sigmas`` / ``scheduler_config`` against the constants; ``--snapshot <dir>``
checks them against that snapshot's ``scheduler_config.json`` and ``model_index.json``:

  python qi21_sched.py --oracle <base wt>/oracle/256 <base wt>/oracle/512
  python qi21_sched.py --turbo --oracle oracle/turbo256 oracle/turbo512 oracle/turbo1024 --snapshot <Turbo snapshot>
"""
from __future__ import annotations

import math

import numpy as np

# scheduler/scheduler_config.json of Qwen/Qwen-Image-2.1 @ 790c926
SCHED_CFG = dict(base_image_seq_len=256, max_image_seq_len=8192, base_shift=0.5, max_shift=0.9,
                 num_train_timesteps=1000, shift_terminal=0.02, time_shift_type="exponential",
                 use_dynamic_shifting=True, invert_sigmas=False, stochastic_sampling=False, shift=1.0)

# scheduler/scheduler_config.json of Qwen/Qwen-Image-2.1-Turbo @ d65dbc9 (the base_* / max_* keys feed only
# the unused mu)
SCHED_CFG_TURBO = dict(base_image_seq_len=256, max_image_seq_len=8192, base_shift=0.5, max_shift=0.9,
                       num_train_timesteps=1000, shift_terminal=None, time_shift_type="exponential",
                       use_dynamic_shifting=False, invert_sigmas=False, stochastic_sampling=False, shift=1.0)
# model_index.json `sample_sigmas` of Qwen/Qwen-Image-2.1-Turbo @ d65dbc9 — the 8-step grid
SAMPLE_SIGMAS_TURBO = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)


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


def schedule_fixed(sample_sigmas, cfg: dict = SCHED_CFG_TURBO) -> tuple[np.ndarray, np.ndarray]:
    """A checkpoint's own sigma grid -> (sigmas [n+1] fp32 ending in 0, timesteps [n] fp32); n = len(grid)."""
    assert not cfg["use_dynamic_shifting"], "dynamic shifting needs mu and a grid-free schedule: use schedule()"
    s = np.asarray(sample_sigmas, dtype=np.float64).astype(np.float32)
    shift = cfg["shift"]
    s = shift * s / (1 + (shift - 1) * s)
    if cfg["shift_terminal"]:
        one_minus = 1 - s
        s = 1 - one_minus / (one_minus[-1] / (1 - cfg["shift_terminal"]))
    s = s.astype(np.float32)
    timesteps = s * np.float32(cfg["num_train_timesteps"])
    sigmas = np.concatenate([s, np.zeros(1, np.float32)])
    return sigmas, timesteps.astype(np.float32)


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
    ap.add_argument("--oracle", nargs="+", default=None,
                    help="oracle dirs (default oracle/256 oracle/512; with --turbo oracle/turbo{256,512,1024})")
    ap.add_argument("--turbo", "--fixed", dest="fixed", action="store_true",
                    help="gate schedule_fixed(SAMPLE_SIGMAS_TURBO, SCHED_CFG_TURBO) (Qwen-Image-2.1-Turbo)")
    ap.add_argument("--snapshot", default=None,
                    help="snapshot dir whose scheduler_config.json (and with --turbo model_index.json) the constants "
                         "must equal; default: the pinned base snapshot in base mode, not checked with --turbo")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.oracle is None:
        args.oracle = [str(here / "oracle" / s) for s in
                       (("turbo256", "turbo512", "turbo1024") if args.fixed else ("256", "512"))]
    args.json = args.json or str(here / "_work" / ("gate_sched_turbo.json" if args.fixed else "gate_sched.json"))
    cfg_ref = SCHED_CFG_TURBO if args.fixed else SCHED_CFG
    bar = 1e-6
    res, ok = dict(mode="fixed grid (Turbo)" if args.fixed else "base (linspace + exponential shift + terminal)"), True
    snap = None
    if args.snapshot:
        snap = Path(args.snapshot)
    elif not args.fixed:
        try:
            sys.path.insert(0, str(here.parent))
            from _paths import hf_snapshot
            snap = Path(hf_snapshot("Qwen/Qwen-Image-2.1", revision="790c92633540aa0cb11d9abf19eb46d861714758"))
        except Exception as e:  # snapshot absent: the oracle arrays still gate the constants
            print(f"[sched] snapshot config not checked ({type(e).__name__})", flush=True)
    if snap is not None:
        cfg = json.load(open(snap / "scheduler" / "scheduler_config.json"))
        diff = {k: (v, cfg.get(k)) for k, v in cfg_ref.items() if cfg.get(k) != v}
        res["snapshot"] = str(snap)
        res["config_matches_snapshot"] = not diff
        ok &= not diff
        print(f"[sched] constants vs {snap.name[:7]} scheduler_config.json: {'identical' if not diff else diff}",
              flush=True)
        if args.fixed:
            grid = json.load(open(snap / "model_index.json")).get("sample_sigmas")
            same = grid == list(SAMPLE_SIGMAS_TURBO)
            res["sample_sigmas_match_model_index"] = same
            ok &= same
            print(f"[sched] SAMPLE_SIGMAS_TURBO vs {snap.name[:7]} model_index.json sample_sigmas: "
                  f"{'identical' if same else grid}", flush=True)
    for o in args.oracle:
        od = Path(o)
        meta = json.load(open(od / "meta.json"))
        steps, N = int(meta["steps"]), int(meta["N"])
        ref_sig = np.fromfile(od / "sigmas.f32", "<f4")
        ref_ts = np.fromfile(od / "timesteps.f32", "<f4")
        meta_ok = True
        if args.fixed:
            assert steps == len(SAMPLE_SIGMAS_TURBO), (steps, len(SAMPLE_SIGMAS_TURBO))
            sig, ts = schedule_fixed(SAMPLE_SIGMAS_TURBO, SCHED_CFG_TURBO)
            mu = None
            mcfg = meta.get("scheduler_config") or {}
            meta_diff = {k: (v, cfg_ref.get(k)) for k, v in mcfg.items() if cfg_ref.get(k) != v}
            meta_ok = meta.get("sample_sigmas") == list(SAMPLE_SIGMAS_TURBO) and bool(mcfg) and not meta_diff
        else:
            sig, ts, mu = schedule(steps, N)
        r = dict(N=N, steps=steps, mu=mu, mu_meta=meta.get("mu"),
                 sigmas_maxd=float(np.abs(sig.astype(np.float64) - ref_sig).max()),
                 sigmas_exact=bool(np.array_equal(sig, ref_sig)),
                 timesteps_maxd=float(np.abs(ts.astype(np.float64) - ref_ts).max()),
                 timesteps_exact=bool(np.array_equal(ts, ref_ts)))
        if args.fixed:
            r["meta_grid_and_config_match"] = meta_ok
        r["t_maxd_vs_meta"] = max(abs(float(model_t(ts, i)[0]) - meta["t"][i]) for i in range(steps))
        r["t_equals_sigma_bit_exact"] = f"{sum(bool(model_t(ts, i)[0] == sig[i]) for i in range(steps))}/{steps}"
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
                   and worst <= bar and meta_ok)
        ok &= r["ok"]
        res[od.name] = r
        mu_s = f"mu {mu:.10f}" if mu is not None else "mu not used (use_dynamic_shifting false)"
        print(f"[sched] {od.name}: N {N} steps {steps} {mu_s} (meta {meta.get('mu')})\n"
              + (f"  meta sample_sigmas + scheduler_config == constants: {meta_ok}\n" if args.fixed else "")
              + f"  sigmas    max|d| {r['sigmas_maxd']:.3e} bit-exact {r['sigmas_exact']}\n"
              f"  timesteps max|d| {r['timesteps_maxd']:.3e} bit-exact {r['timesteps_exact']}\n"
              f"  model t (timesteps/1000) vs meta t (7 dp): max|d| {r['t_maxd_vs_meta']:.2e}; "
              f"t == sigma bit-exact {r['t_equals_sigma_bit_exact']}\n"
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
