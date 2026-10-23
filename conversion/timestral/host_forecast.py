"""Host side of the Timestral port: everything outside the graph, in NumPy.

Mirrors `TimestralForecaster.forecast()` (github.com/mistralai/timestral @ a534bdb) for one
univariate series without covariates and a horizon of at most 1,024 steps (one decode block),
with the network replaced by a graph that runs at a fixed bucket: `L` context patches and `h`
horizon patches. float64 where the released code uses float64 (the causal Welford statistics, the
advancing refine frame), float32 everywhere else, in the released order of operations.

    core(patches (R, L, 64), cos (L+h, 64), sin (L+h, 64), attn_mask (1, 1, L+h, L+h), 1 = attend / 0)
        -> normalized quantiles (R, h * 32, 21)

A shorter context is left-padded with unobserved patches up to the bucket: the padding is never a
valid key, and the positions count from the first observed patch, so a padded call computes the
same function as an unpadded one. The first `ceil(H / 32)` of the bucket's `h` horizon patches are
the forecast (a decoder token reads the context and the decoder tokens before it, nothing after).
"""
import math

import numpy as np

PATCH = 32
QUANTILES = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
             0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)
MEDIAN = QUANTILES.index(0.5)
MIN_SCALE = 1e-6
CAP_K = 1e4
MAX_CONTEXT = 32768
BLOCK = 32  # patches per decode block (per_shot_horizon_patches)
HEAD_DIM, ROPE_DIM, THETA = 64, 32, 10000.0
F32 = np.float32


def inv_freq():
    """(32,) float32: rotary frequencies for the first ROPE_DIM / 2 pairs, 0 (identity) for the rest."""
    out = np.zeros(HEAD_DIM // 2, dtype=F32)
    idx = np.arange(0, ROPE_DIM, 2, dtype=F32)
    out[: ROPE_DIM // 2] = F32(1.0) / (F32(THETA) ** (idx / F32(ROPE_DIM)))
    return out


INV_FREQ = inv_freq()


# ---- released scaler.py, float64 statistics ------------------------------------------------------

def causal_statistics(data, mask):
    """compute_causal_statistics: causal running mean / std (Bessel) in float64, returned as float32."""
    x = data.astype(np.float64)
    w = mask.astype(F32).astype(np.float64)
    weighted = w * x
    cum_w = np.cumsum(w, axis=-1)
    cum_x = np.cumsum(weighted, axis=-1)
    den = np.maximum(cum_w, 1.0)
    means = cum_x / den
    shifted = np.zeros_like(means)
    shifted[..., 1:] = means[..., :-1]
    m2 = np.cumsum((x - shifted) * (x - means) * w, axis=-1)
    variance = m2 / np.maximum(den - 1.0, 1.0)
    scales = np.maximum(np.sqrt(variance), MIN_SCALE)
    return means.astype(F32), scales.astype(F32)


def patch_loc_scale(data, mask):
    """causal_patch_loc_scale: the statistics at each patch's last step, broadcast over the patch."""
    means, scales = causal_statistics(data, mask)
    pm = means[..., PATCH - 1::PATCH]
    ps = scales[..., PATCH - 1::PATCH]
    return np.repeat(pm, PATCH, axis=-1), np.repeat(ps, PATCH, axis=-1)


def affine_asinh(data, loc, scale, mask):
    """apply_affine_asinh_masked: asinh((x - loc) / scale) on observed steps, 0 elsewhere (float32)."""
    scaled = (data.astype(F32) - loc) / np.maximum(scale, np.finfo(F32).eps)
    scaled = np.where(mask, scaled, F32(0.0))
    return np.where(mask, np.arcsinh(scaled), scaled).astype(F32)


def clamp_nonfinite(v):
    """+/-inf -> the nearest finite value along the last axis; a row with no finite value -> 0; NaN kept."""
    fin = np.isfinite(v)
    hi = np.where(fin, v, -np.inf).max(axis=-1, keepdims=True)
    lo = np.where(fin, v, np.inf).min(axis=-1, keepdims=True)
    c = np.where(v == np.inf, hi, np.where(v == -np.inf, lo, v))
    return np.where(np.isinf(c), F32(0.0), c).astype(v.dtype)


def welford_combine(w_a, mean_a, m2_a, w_b, mean_b, m2_b):
    w = w_a + w_b
    w_safe = np.maximum(w, np.finfo(w.dtype).tiny)
    delta = mean_b - mean_a
    mean = mean_a + delta * w_b / w_safe
    m2 = m2_a + m2_b + np.square(delta) * w_a * w_b / w_safe
    zero = w == 0
    return w, np.where(zero, 0.0, mean), np.where(zero, 0.0, m2)


def cumulative_welford(w, mean, m2):
    """Inclusive prefix scan along axis 1 (Hillis-Steele), as released."""
    n = w.shape[1]
    d = 1
    while d < n:
        def shift(a):
            out = np.zeros_like(a)
            out[:, d:] = a[:, : n - d]
            return out
        w, mean, m2 = welford_combine(shift(w), shift(mean), shift(m2), w, mean, m2)
        d *= 2
    return w, mean, m2


def advancing_block_frames(context, mask, sinh_median):
    """(R, H) float64 loc / scale of each horizon patch in the advancing refine frame."""
    r, n_pred, ps = sinh_median.shape
    blank = np.zeros((r, n_pred, ps), dtype=np.float64)
    values = np.concatenate([context.reshape(r, -1, ps).astype(np.float64), blank], axis=1)
    weights = np.concatenate([mask.reshape(r, -1, ps).astype(np.float64), blank], axis=1)
    cnt = weights.sum(-1)
    mean = (values * weights).sum(-1) / np.maximum(cnt, 1.0)
    m2 = (weights * np.square(values - mean[..., None])).sum(-1)
    zero = cnt == 0
    mean = np.where(zero, 0.0, mean)
    m2 = np.where(zero, 0.0, m2)
    cnt, mean, m2 = (s[:, -n_pred - 1] for s in cumulative_welford(cnt, mean, m2))
    offset = mean
    clip = float(np.finfo(F32).max) ** 0.5 / 1e3
    locs, scales = [], []
    for h in range(n_pred):
        var = np.maximum(m2 / np.maximum(cnt - 1.0, 1.0), 0.0)
        scale = np.maximum(np.sqrt(var), MIN_SCALE)
        locs.append(mean)
        scales.append(scale)
        folded = np.clip(sinh_median[:, h] * scale[:, None] + mean[:, None] - offset[:, None], -clip, clip)
        folded = folded + offset[:, None]
        b_mean = folded.mean(-1)
        b_m2 = np.square(folded - b_mean[:, None]).sum(-1)
        cnt, mean, m2 = welford_combine(cnt, mean, m2, np.full_like(cnt, ps), b_mean, b_m2)
    return np.stack(locs, axis=1), np.stack(scales, axis=1)


# ---- the graph's inputs ---------------------------------------------------------------------------

def prepare_context(context):
    """forecast()'s _as_context + _assemble_rows for one univariate series: (T',) float32 + mask, T' % 32 == 0.

    NaN / inf are missing (value 0, mask False). The released code also forward-fills missing
    steps; every consumer weights them by the mask, so the fill changes no number and is skipped.
    """
    x = np.asarray(context, dtype=F32).reshape(-1)
    mask = np.isfinite(x)
    x = np.where(mask, x, F32(0.0)).astype(F32)
    nonneg = bool(mask.any() and ((x >= 0) | ~mask).all())
    if x.size > MAX_CONTEXT:
        x, mask = x[-MAX_CONTEXT:], mask[-MAX_CONTEXT:]
    pad = (-x.size) % PATCH
    if pad:
        x = np.concatenate([np.zeros(pad, F32), x])
        mask = np.concatenate([np.zeros(pad, bool), mask])
    return x, mask, nonneg


def rope_tables(positions):
    """(S, 64) cos / sin in float32: concatenated halves, identity on the unrotated pairs."""
    freqs = positions.astype(F32)[:, None] * INV_FREQ[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(F32), np.sin(emb).astype(F32)


def attention_mask(key_valid, n_ctx, n_hor):
    """(1, 1, S, S) float mask, 1 = attend: the encoder is causal over valid keys (self always open),
    each decoder token reads every valid context patch and the decoder tokens up to itself."""
    s = n_ctx + n_hor
    allowed = np.zeros((s, s), dtype=bool)
    i = np.arange(n_ctx)[:, None]
    j = np.arange(n_ctx)[None, :]
    allowed[:n_ctx, :n_ctx] = ((j <= i) & key_valid[None, :]) | (i == j)
    allowed[n_ctx:, :n_ctx] = key_valid[None, :]
    allowed[n_ctx:, n_ctx:] = np.tril(np.ones((n_hor, n_hor), dtype=bool))
    return allowed.astype(F32)[None, None]


def bucket_inputs(x, mask, n_ctx, n_hor, positions_from_first_observed=True):
    """Graph inputs for one prepared series (x, mask) left-padded to `n_ctx` patches.

    Returns (patches (1, n_ctx, 64), cos, sin, attn_mask). The statistics of the padded row equal
    the unpadded row's at every real step (the padding has zero weight)."""
    ll = x.size // PATCH
    assert ll <= n_ctx, f"context of {ll} patches does not fit a {n_ctx}-patch bucket"
    extra = (n_ctx - ll) * PATCH
    xb = np.concatenate([np.zeros(extra, F32), x])
    mb = np.concatenate([np.zeros(extra, bool), mask])
    loc, scale = patch_loc_scale(xb, mb)
    scaled = affine_asinh(xb, loc, scale, mb)
    patches = np.concatenate([scaled.reshape(n_ctx, PATCH), (~mb).reshape(n_ctx, PATCH).astype(F32)], axis=-1)
    obs = mb.reshape(n_ctx, PATCH).any(-1)
    key_valid = obs | np.maximum.accumulate(obs)
    first = int(np.argmax(obs)) if obs.any() else 0
    pos = np.arange(n_ctx + n_hor) - (first if positions_from_first_observed else 0)
    cos, sin = rope_tables(pos)
    return patches[None], cos, sin, attention_mask(key_valid, n_ctx, n_hor)


# ---- one pass and the full forecast ---------------------------------------------------------------

def refine_block(q_raw, x, mask):
    """_refine_block: horizon patch j >= 1 re-denormalized in the advancing refine frame."""
    r, hl, nq = q_raw.shape
    n_pred = hl // PATCH
    if n_pred <= 1:
        return q_raw
    loc, scale = patch_loc_scale(x[None], mask[None])
    loc0, scale0 = loc[:, -1], scale[:, -1]  # float32 (R,)
    median = q_raw[..., MEDIAN].reshape(r, n_pred, PATCH).astype(np.float64)
    sinh_median = (median - loc0[:, None, None].astype(np.float64)) / scale0[:, None, None].astype(np.float64)
    loc_j, scale_j = advancing_block_frames(x[None], mask[None], sinh_median)
    first = np.zeros(loc_j.shape, dtype=bool)
    first[:, 0] = True
    loc_j = np.repeat(np.where(first, loc0[:, None], loc_j.astype(F32)), PATCH, axis=-1)
    scale_j = np.repeat(np.where(first, scale0[:, None], scale_j.astype(F32)), PATCH, axis=-1)
    sinh_norm = (q_raw - loc0[:, None, None]) / scale0[:, None, None]
    return clamp_nonfinite((sinh_norm * scale_j[..., None] + loc_j[..., None]).astype(F32))


def cap_quantiles(q, x, mask):
    """_cap_quantiles with k = 1e4: clamp to [min - k s, max + k s] of the observed context."""
    ctx_max = F32(x[mask].max()) if mask.any() else F32(0.0)
    ctx_min = F32(x[mask].min()) if mask.any() else F32(0.0)
    _, scales = causal_statistics(x, mask)
    a = scales[-1] * F32(CAP_K)
    return np.clip(q, F32(ctx_min - a), F32(ctx_max + a)).astype(F32)


def finish_pass(q_norm, x, mask, prediction_length):
    """One _forecast_raw pass after the graph: (h*32, 21) normalized -> (H, 21) raw, capped."""
    n_pred = math.ceil(prediction_length / PATCH)
    q_norm = q_norm[: n_pred * PATCH].astype(F32)
    loc, scale = patch_loc_scale(x, mask)
    loc0, scale0 = loc[-1], scale[-1]
    q_raw = clamp_nonfinite((np.sinh(q_norm) * scale0 + loc0).astype(F32))
    q_raw = refine_block(q_raw[None], x, mask)[0][:prediction_length]
    q_raw = cap_quantiles(q_raw, x, mask)
    if np.isnan(q_raw).any():
        raise ValueError(f"forecast produced {int(np.isnan(q_raw).sum())} NaN values; the input is degenerate")
    return q_raw


class Bucket:
    def __init__(self, n_ctx, n_hor):
        self.n_ctx, self.n_hor = n_ctx, n_hor

    def fits(self, ll, n_pred):
        return ll <= self.n_ctx and n_pred <= self.n_hor


def forecast(core, context, prediction_length, bucket, mode="batched", positions_from_first_observed=True):
    """TimestralForecaster.forecast() defaults (symmetric averaging, non-negative clip), one series.

    mode "batched": one graph call on R = 2 rows [s, -s]; "split": two calls with R = 1.
    Returns (H, 21) float32.
    """
    assert 1 <= prediction_length <= BLOCK * PATCH, "v1 forecasts one decode block (<= 1,024 steps)"
    x, mask, nonneg = prepare_context(context)
    n_pred = math.ceil(prediction_length / PATCH)
    assert bucket.fits(x.size // PATCH, n_pred), "window does not fit the bucket"
    patches, cos, sin, amask = bucket_inputs(x, mask, bucket.n_ctx, bucket.n_hor, positions_from_first_observed)
    mirror = patches.copy()
    mirror[..., :PATCH] = -mirror[..., :PATCH]  # stats of -x: loc negates, scale is unchanged
    if mode == "batched":
        q = core(np.concatenate([patches, mirror], axis=0), cos, sin, amask)
        q_s, q_m = q[0], q[1]
    else:
        q_s = core(patches, cos, sin, amask)[0]
        q_m = core(mirror, cos, sin, amask)[0]
    a = finish_pass(q_s, x, mask, prediction_length)
    b = finish_pass(q_m, -x, mask, prediction_length)
    out = (F32(0.5) * (a + (-b)[..., ::-1])).astype(F32)
    if nonneg:
        out = np.maximum(out, F32(0.0))
    return out
