#!/usr/bin/env python3
"""test_lfm3_tiny_parity.py OUT_DIR — step 2 of the LFM3 tiny parity smoke: lfm3.py (export venv, fp32, CPU)
against the HF tiny reference that test_lfm3_tiny_hf_ref.py wrote to OUT_DIR. Seconds to run; exits 1 on a miss.

Paths (each compared with the HF fp32 eager logits, all positions):
  prefill   Lfm3ForCausalLM, one stateless call
  i         Lfm3ForCausalLMStateful, one call S=N on a fresh state
  ii        S=1 steps, fixed KV capacity
  iii       S=1 steps, KV capacity grown like the engine's GrowingKVCache (2x, old rows copied)
  chunk     S=16 chunks (S>1 with an offset: a prefill continued from a cache)
  iv        Lfm3EmbedsForCausalLMStateful + emit_hidden: 20-token prompt in one call, then S=1;
            logits and hidden (vs HF last_hidden_state)
  audio     the Audio-2B layout (`lfm.model.` prefix, config under `lfm`, decoy towers) == plain
Negative controls (must be RED): gate removed (gate = 1), zero-tail rotated over all dims, sliding
window removed.
"""
import sys

import numpy as np
import torch

from coreai_models.models.macos import lfm3 as L

TOL = 1e-4


def main(out: str) -> int:
    torch.set_grad_enabled(False)
    ref = np.load(f"{out}/ref.npz")
    ids = torch.tensor(ref["ids"], dtype=torch.int32)[None]
    n = ids.shape[1]
    want, want_h = ref["logits"], ref["last_hidden"]
    fails = []

    def report(name, got, ref_arr=want, red=False):
        d = float(np.abs(got - ref_arr).max())
        top1 = bool((got.argmax(-1) == ref_arr.argmax(-1)).all())
        ok = d <= TOL and top1
        if red:
            status = "RED (expected)" if not ok else "GREEN (control did not bite!)"
            if ok:
                fails.append(name)
        else:
            status = "PASS" if ok else "FAIL"
            if not ok:
                fails.append(name)
        print(f"  {name:28s} max|d| {d:.3e}  top1 {'all' if top1 else 'MISS'}  {status}")
        return got

    plain = f"{out}/plain"
    m0 = L.lfm3_from_hf(plain, target_dtype=torch.float32, stateful=False)
    cfg = m0.config
    print(f"tiny: {cfg.num_hidden_layers} layers {cfg.layer_types}, window {cfg.sliding_window}, "
          f"attn slots {cfg.attention_layer_ids}, conv slots {cfg.conv_layer_ids}")
    pos = torch.arange(n, dtype=torch.int32)[None]
    report("prefill (stateless)", m0(ids, pos)[0].numpy())

    m = L.lfm3_from_hf(plain, target_dtype=torch.float32)
    st = L.build_decode_state(cfg, 64, torch.float32)
    report("i  stateful S=N", m(ids, pos, st["k_cache"], st["v_cache"], st["conv_state"])[0].numpy())

    st = L.build_decode_state(cfg, 64, torch.float32)
    rows = [m(ids[:, t:t + 1], pos[:, :t + 1], st["k_cache"], st["v_cache"], st["conv_state"])[0, 0]
            for t in range(n)]
    report("ii S=1 fixed KV", torch.stack(rows).numpy())

    cap = 4
    st = L.build_decode_state(cfg, cap, torch.float32)
    k, v, c = st["k_cache"], st["v_cache"], st["conv_state"]
    rows = []
    for t in range(n):
        if t + 1 > cap:
            new = cap
            while new < t + 1:
                new *= 2
            k2 = torch.zeros(*k.shape[:3], new, k.shape[4])
            v2 = torch.zeros_like(k2)
            k2[..., :cap, :] = k
            v2[..., :cap, :] = v
            k, v, cap = k2, v2, new
        rows.append(m(ids[:, t:t + 1], pos[:, :t + 1], k, v, c)[0, 0])
    report(f"iii S=1 grown KV (->{cap})", torch.stack(rows).numpy())

    st = L.build_decode_state(cfg, 64, torch.float32)
    rows = []
    for s0 in range(0, n, 16):
        s1 = min(n, s0 + 16)
        rows.append(m(ids[:, s0:s1], pos[:, :s1], st["k_cache"], st["v_cache"], st["conv_state"])[0])
    report("chunk S=16", torch.cat(rows).numpy())

    me = L.lfm3_from_hf(plain, target_dtype=torch.float32, embeds=True, emit_hidden=True)
    emb = me.model.embed_tokens(ids)
    st = L.build_decode_state(cfg, 64, torch.float32)
    lg, hd = me(emb[:, :20], pos[:, :20], st["k_cache"], st["v_cache"], st["conv_state"])
    lrows, hrows = [lg[0]], [hd[0]]
    for t in range(20, n):
        lg, hd = me(emb[:, t:t + 1], pos[:, :t + 1], st["k_cache"], st["v_cache"], st["conv_state"])
        lrows.append(lg[0])
        hrows.append(hd[0])
    report("iv embeds+hidden: logits", torch.cat(lrows).numpy())
    report("iv embeds+hidden: hidden", torch.cat(hrows).numpy(), want_h)

    mx = L.lfm3_from_hf(plain, target_dtype=torch.float32)
    mx.model.set_explicit_mask()
    st = L.build_decode_state(cfg, 64, torch.float32)
    report("i  S=N, explicit mask", mx(ids, pos, st["k_cache"], st["v_cache"], st["conv_state"])[0].numpy())
    st = L.build_decode_state(cfg, 64, torch.float32)
    rows = []
    for s0 in range(0, n, 16):
        s1 = min(n, s0 + 16)
        rows.append(mx(ids[:, s0:s1], pos[:, :s1], st["k_cache"], st["v_cache"], st["conv_state"])[0])
    report("chunk S=16, explicit mask", torch.cat(rows).numpy())
    st = L.build_decode_state(cfg, 64, torch.float32)
    rows = [mx(ids[:, t:t + 1], pos[:, :t + 1], st["k_cache"], st["v_cache"], st["conv_state"])[0, 0]
            for t in range(n)]
    report("ii S=1, explicit mask", torch.stack(rows).numpy())

    ma = L.lfm3_from_hf(f"{out}/audio", key_prefix="lfm.model.", config_key="lfm",
                        target_dtype=torch.float32, stateful=False)
    report("audio layout (prefix lfm.model.)", ma(ids, pos)[0].numpy())

    print("negative controls:")
    mg = L.lfm3_from_hf(plain, target_dtype=torch.float32, stateful=False)
    for layer in mg.model.layers:
        if layer.is_attention:
            a = layer.self_attn
            a.gate = lambda xp, a=a: torch.ones(*xp.shape[:-1], a.n_heads, dtype=xp.dtype)
    report("gate removed", mg(ids, pos)[0].numpy(), red=True)

    mz = L.lfm3_from_hf(plain, target_dtype=torch.float32, stateful=False)
    full_cfg = L.Lfm3Config(**{**cfg.__dict__, "rope_parameters": {
        **cfg.rope_parameters,
        L.FULL: {**cfg.rope_parameters[L.FULL], "partial_rotary_factor": 1.0}}})
    mz.model.inv_freq[L.FULL] = L.rope_inv_freq(full_cfg, L.FULL)
    report("zero-tail rotated (all dims)", mz(ids, pos)[0].numpy(), red=True)

    mw = L.lfm3_from_hf(plain, target_dtype=torch.float32, stateful=False)
    for layer in mw.model.layers:
        if layer.layer_type == L.SLIDING:
            layer.self_attn.sdpa.window_size = 0
    report("sliding window removed", mw(ids, pos)[0].numpy(), red=True)

    print("ALL PASS" if not fails else f"FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
