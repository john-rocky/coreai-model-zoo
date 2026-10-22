"""Ladder 1 (L1): the re-authored graph vs the released module on the same normalized input.

For every fixture window that fits the bucket, the released code (github.com/mistralai/timestral,
on PYTHONPATH) runs its own `_predict_context_only` steps up to the head (scale -> representation
-> mask and positions -> _run_two_stream -> head, before denormalization) on the two rows
[s, -s] left-padded to the bucket; the re-authored TimestralCore runs on the host's inputs
(host_forecast.bucket_inputs) for the same rows. Measured, in this order:

  floor   released fp32 vs released `.double()`, and vs a true float64 run (RMSNorm and RoPE
          patched to stay in float64), and released fp32 twice; bar = 2 x (fp32 vs .double())
  L1      re-authored fp32 vs released fp32: max|d|, median-channel max|d| and corr, monotone
  red     the same with wQKV read as (qkv, heads, head_dim) (must fail)
  fp16    per-layer absmax of every Linear output, of the residual after every block, and of
          RMSNorm's sum of squares over the 768 dims, over all rows and tokens (pads included)

Writes one JSON with per-window rows and the aggregates.
"""
import argparse
import contextlib
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

import host_forecast as H
from timestral_core import RMSNorm as CoreRMSNorm
from timestral_core import TimeAttention, load_core


def released_normalized(model, inputs, observed, group_ids, h):
    """Lines 213-226 of the released model.py (_predict_context_only), returning the head output."""
    from timestral import masks as rmask
    c = model.config
    ps = c.patch_size
    rows, t = inputs.shape
    ll = t // ps
    scaled, _, _ = model.scale(inputs, observed)
    x_enc = model.representation(scaled, observed)
    x_dec = model.representation.forecast_tokens(rows, h)
    obs_per_patch = observed.unflatten(-1, (ll, ps)).any(dim=-1)
    key_valid = rmask.compute_enc_key_validity(obs_per_patch, c.attend_missing_patches)
    time_mask = rmask.build_forecast_time_mask(ll, h, key_valid, inputs.device)
    variate_mask = rmask.build_two_stream_variate_mask(group_ids, key_valid, h)
    unique_ids, inverse = torch.unique(group_ids, return_inverse=True)
    first_obs = rmask.group_first_observed_patch(obs_per_patch, inverse, unique_ids.numel())[inverse]
    positions = rmask.build_forecast_positions(ll, h, first_obs)
    return model._run_two_stream(x_enc, x_dec, time_mask, variate_mask, positions)


@contextlib.contextmanager
def true_float64():
    """Patch the released RMSNorm (x.float()) and RoPE (float32 tables) to compute in the input dtype."""
    from timestral import layers as rl
    old_norm, old_rope = rl.RMSNorm.forward, rl.SinCosRoPE.forward

    def norm(self, x):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * self.scale

    def rope(self, x, position_ids):
        inv = torch.zeros(self.dim // 2, dtype=torch.float64)
        idx = torch.arange(0, self.rotary_dim, 2, dtype=torch.float64)
        inv[: self.rotary_dim // 2] = 1.0 / (self.base ** (idx / self.rotary_dim))
        freqs = position_ids.to(torch.float64)[..., None] * inv
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

    rl.RMSNorm.forward, rl.SinCosRoPE.forward = norm, rope
    try:
        yield
    finally:
        rl.RMSNorm.forward, rl.SinCosRoPE.forward = old_norm, old_rope


class Absmax:
    """Forward hooks on the re-authored core: per-module absmax over everything that passes."""

    def __init__(self, core):
        self.stats, self.hooks = {}, []
        for name, mod in core.named_modules():
            if isinstance(mod, torch.nn.Linear):
                self.hooks.append(mod.register_forward_hook(self._out(name)))
            elif isinstance(mod, CoreRMSNorm):
                self.hooks.append(mod.register_forward_hook(self._norm(name)))
            elif name.startswith("blocks.") and name.count(".") == 1:
                self.hooks.append(mod.register_forward_hook(self._out(name + ".residual")))
            if isinstance(mod, TimeAttention):
                mod.stats = self.stats
                mod.stats_name = name

    def _put(self, key, v):
        self.stats[key] = max(self.stats.get(key, 0.0), float(v))

    def _out(self, name):
        return lambda m, i, o: self._put(name, o.detach().abs().max())

    def _norm(self, name):
        def f(m, i, o):
            x = i[0].detach().double()
            self._put(name + ".sumsq", x.pow(2).sum(-1).max())
            self._put(name + ".in", x.abs().max())
        return f

    def remove(self):
        for h in self.hooks:
            h.remove()


def to_np(t):
    return t.detach().double().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path, help="dir with config.json + model.safetensors")
    ap.add_argument("--fixtures", required=True, type=Path)
    ap.add_argument("--bucket", required=True, help="L,h in patches, e.g. 64,32")
    ap.add_argument("--max-context-steps", type=int, default=None, help="only windows whose context fits")
    ap.add_argument("--min-context-steps", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--save-ref", type=Path, default=None,
                    help="npz with the graph inputs and the released fp32 output per window (for engine_l1.py)")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    n_ctx, n_hor = (int(v) for v in a.bucket.split(","))
    from timestral.loading import load_model

    torch.manual_seed(0)
    model32 = load_model(a.weights)
    model64 = copy.deepcopy(model32).double()
    core = load_core(a.weights / "model.safetensors", n_hor)
    red = load_core(a.weights / "model.safetensors", n_hor, qkv_layout="naive")
    tf = torch.tensor(H.INV_FREQ)
    rope_buf = model32.transformer.rope.inv_freq
    inv_freq_equal = bool(torch.equal(tf, rope_buf.float()))
    probe = Absmax(core)

    man = json.loads((a.fixtures / "manifest.json").read_text())
    rows_out, skipped = [], []
    saved = {k: [] for k in ("patches", "cos", "sin", "attn_mask", "ref32", "ref64t")}
    t0 = time.time()
    for g in man["groups"]:
        ctxs = np.load(a.fixtures / f"{g['name']}.context.npy")
        for b in range(ctxs.shape[0]):
            x, mask, _ = H.prepare_context(ctxs[b])
            steps = ctxs.shape[1]
            if x.size // H.PATCH > n_ctx or (a.max_context_steps and steps > a.max_context_steps) \
                    or steps < a.min_context_steps:
                skipped.append(dict(group=g["name"], window=b, context_steps=int(steps)))
                continue
            extra = n_ctx * H.PATCH - x.size
            xb = np.concatenate([np.zeros(extra, H.F32), x])
            mb = np.concatenate([np.zeros(extra, bool), mask])
            inputs = torch.tensor(np.stack([xb, -xb]))
            observed = torch.tensor(np.stack([mb, mb]))
            gids = torch.tensor([0, 1])
            with torch.inference_mode():
                r32 = released_normalized(model32, inputs, observed, gids, n_hor)
                r32b = released_normalized(model32, inputs, observed, gids, n_hor)
                r64 = released_normalized(model64, inputs.double(), observed, gids, n_hor)
                with true_float64():
                    r64t = released_normalized(model64, inputs.double(), observed, gids, n_hor)
                patches, cos, sin, bias = H.bucket_inputs(x, mask, n_ctx, n_hor)
                mirror = patches.copy()
                mirror[..., : H.PATCH] = -mirror[..., : H.PATCH]
                p2 = H.bucket_inputs(-x, mask, n_ctx, n_hor)[0]
                args = (torch.tensor(np.concatenate([patches, mirror])), torch.tensor(cos), torch.tensor(sin),
                        torch.tensor(bias))
                mine = core(*args)
                bad = red(*args)
            ref = to_np(r32)
            got = to_np(mine)
            if a.save_ref:
                for k, v in zip(saved, (args[0].numpy(), cos, sin, bias, r32.numpy(), r64t.numpy())):
                    saved[k].append(v)
            med_r, med_g = ref[..., H.MEDIAN].ravel(), got[..., H.MEDIAN].ravel()
            rows_out.append(dict(
                group=g["name"], window=b, context_steps=int(steps),
                floor_double=float(np.abs(ref - to_np(r64)).max()),
                floor_true64=float(np.abs(ref - to_np(r64t)).max()),
                rerun=float(np.abs(ref - to_np(r32b)).max()),
                l1_max=float(np.abs(got - ref).max()),
                l1_vs_true64=float(np.abs(got - to_np(r64t)).max()),
                median_max=float(np.abs(med_g - med_r).max()),
                median_corr=float(np.corrcoef(med_g, med_r)[0, 1]) if med_r.std() > 0 else None,
                monotone=bool((np.diff(got, axis=-1) >= 0).all()),
                ref_monotone=bool((np.diff(ref, axis=-1) >= 0).all()),
                red_max=float(np.abs(to_np(bad) - ref).max()),
                red_median_corr=float(np.corrcoef(to_np(bad)[..., H.MEDIAN].ravel(), med_r)[0, 1]),
                mirror_inputs_equal=bool(np.array_equal(p2, mirror)),
                ref_absmax=float(np.abs(ref).max()),
            ))
    probe.remove()
    keys = ["floor_double", "floor_true64", "rerun", "l1_max", "l1_vs_true64", "median_max", "red_max"]
    agg = {k: max(r[k] for r in rows_out) for k in keys}
    agg["median_corr_min"] = min(r["median_corr"] for r in rows_out if r["median_corr"] is not None)
    agg["red_median_corr_min"] = min(r["red_median_corr"] for r in rows_out)
    agg["monotone_all"] = all(r["monotone"] for r in rows_out)
    agg["ref_monotone_all"] = all(r["ref_monotone"] for r in rows_out)
    agg["mirror_inputs_equal_all"] = all(r["mirror_inputs_equal"] for r in rows_out)
    agg["bar"] = 2.0 * agg["floor_double"]
    agg["l1_pass"] = bool(agg["l1_max"] <= agg["bar"] and agg["monotone_all"])
    agg["red_fails"] = bool(agg["red_max"] > agg["bar"])
    agg["windows"] = len(rows_out)
    agg["skipped"] = len(skipped)
    top = sorted(probe.stats.items(), key=lambda kv: -kv[1])
    res = dict(bucket=[n_ctx, n_hor], torch=torch.__version__, threads=torch.get_num_threads(),
               inv_freq_equal_to_released=inv_freq_equal, seconds=round(time.time() - t0, 1),
               aggregate=agg, absmax_top=top[:40], absmax=probe.stats, rows=rows_out, skipped=skipped,
               unused_rows=core.unused_rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    if a.save_ref:
        np.savez(a.save_ref, **{k: np.stack(v) for k, v in saved.items()},
                 group=np.array([r["group"] for r in rows_out]), window=np.array([r["window"] for r in rows_out]),
                 bar=np.float64(agg["bar"]), bucket=np.array([n_ctx, n_hor]))
    print(json.dumps(dict(bucket=res["bucket"], inv_freq_equal=inv_freq_equal, **agg), indent=1))
    print("absmax top 8:", [(k, round(v, 3)) for k, v in top[:8]])
    print("L1_DONE")


if __name__ == "__main__":
    main()
