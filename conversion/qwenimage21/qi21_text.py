"""Qwen-Image-2.1 text encoder (the Qwen3-VL-8B text stack), re-authored in plain torch.

Built from ``text_encoder/config.json`` (``text_config``) and the checkpoint, not from
transformers: the parameter names are the checkpoint's ``model.language_model.*`` names with
that prefix stripped, so the 36-layer text stack loads with ``strict`` checks from the four
``text_encoder/model-0000k-of-00004.safetensors`` shards. What the pipeline reads is the
residual stream after the LAST decoder layer, BEFORE the final RMSNorm (``capture_oracle.py``
neutralises the norm with a forward hook), so ``model.language_model.norm``, ``lm_head`` and
the whole ``model.visual.*`` tower are never loaded.

Text-only inputs make the interleaved 3-axis M-RoPE collapse to plain 1D RoPE: all three axes
carry the same position ``i``, so the T/H/W frequency interleave picks the same number from
every axis (``parity_text_oracle.py`` gates this against the transformers rotary module).

Numerics follow the reference module by module:

* RMSNorm: fp32 statistics, cast back to the input dtype, then ``weight * x`` (Qwen3 order).
  ``q_norm`` / ``k_norm`` are the same RMSNorm over ``head_dim`` (128), with weights.
* RoPE: rotate-half form, ``inv_freq = 1 / theta ** (arange(0, hd, 2) / hd)``, positions
  ``0 .. L-1``. The cos/sin tables are fp32 constants for positions ``0 .. max_len-1`` (plain
  attributes, NOT buffers, so ``model.to(torch.bfloat16)`` cannot round them) and q/k are
  rotated in fp32, then cast back to the activation dtype. transformers rounds cos/sin to
  bf16 in a bf16 model; the target here is the fp32 oracle, and this is strictly closer to it.
* GQA: k/v repeated along the head axis (32 query heads over 8 kv heads, head ``h`` reads kv
  head ``h // 4``), then ``F.scaled_dot_product_attention(..., is_causal=True)``.
* MLP: ``down(silu(gate(x)) * up(x))``.

``residual_fp32`` (the "r32" variant): the residual stream stays fp32 — the norms read it with
fp32 statistics and hand bf16 to the matmuls, the sublayer outputs are added in fp32 — while
every matmul keeps bf16 weights and bf16 inputs. Plain bf16 cannot hold this model's residual:
tokens 0 and 14 (the two ``<|im_start|>`` before the prompt) grow to |h| ~13,000 / ~9,100 in
the middle layers and the last two layers cancel them to ~4,600 / ~100, so the bf16 rounding of
the big values survives the cancellation (token 14 corr 0.968 vs fp32; 0.9996 with the fp32
residual; ``diag_text_bf16.py``, 2026-09-25).

``compute_fp32`` (the "w16a32" variant, the one that ships): the weights stay bf16 in the
checkpoint and the bundle, and every Linear / RMSNorm upcasts them in the graph, so the whole
stack computes in fp32 — the same numbers as the fp32 reference. r32 is not enough: on the
Core AI GPU its token-14 corr moves with the sequence length (0.99987 at L <= 32, 0.9974 above),
and the empty prompt's token 20 (the assistant ``<|im_start|>``, no massive activation) lands
anywhere from 0.994 to 0.66; no partial fp32 island fixed all prompts on CPU, the full fp32
stack did (``sweep_encoder_L.py`` / ``diag_text_islands.py``). The converter keeps the bf16
constants and casts at run time, so the bundle stays bf16-sized (``probe_fp32_cost.py``).

Graph contract (``forward``):
    input_ids [1,L] int32   ->   hidden [1,L,4096]   (fp32 when ``io_fp32``)
``embed_tokens`` (151936 x 4096) is inside the graph. Causal attention makes the output at
token ``i`` independent of everything after it, so right-padding never changes valid tokens.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PREFIX = "model.language_model."


class RMSNorm(nn.Module):
    """Qwen3 RMSNorm: fp32 variance, normalise, cast to the weight dtype, then ``weight *``.

    The reference casts to the INPUT dtype; that is the weight dtype everywhere except in the
    fp32-residual variant, where an fp32 stream is normalised into the bf16 matmuls.
    ``compute_fp32``: the stored weight is upcast and the output stays fp32 (w16a32).
    """
    compute_fp32 = False

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        w = self.weight.float() if self.compute_fp32 else self.weight
        return w * xf.to(w.dtype)


class Linear(nn.Linear):
    """``nn.Linear`` (same parameter names). ``compute_fp32``: input and stored weight upcast, fp32 out."""
    compute_fp32 = False

    def forward(self, x):
        if self.compute_fp32:
            return F.linear(x.float(), self.weight.float())
        return super().forward(x)


def rope_inv_freq(head_dim: int, theta: float) -> torch.Tensor:
    # Same expression as Qwen3VLTextRotaryEmbedding.compute_default_rope_parameters (fp32).
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float, device="cpu") / head_dim))


def rope_tables(n: int, head_dim: int, theta: float):
    """fp32 cos/sin ``[n, head_dim]`` for positions ``0 .. n-1`` (the two halves repeat)."""
    pos = torch.arange(n, dtype=torch.float, device="cpu")
    freqs = torch.outer(pos, rope_inv_freq(head_dim, theta))          # [n, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().contiguous(), emb.sin().contiguous()


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(x, cos, sin):
    """x [B,H,L,hd] (any float dtype); cos/sin [L,hd] fp32. Rotated in fp32, cast back."""
    dt = x.dtype
    xf = x.float()
    return (xf * cos + rotate_half(xf) * sin).to(dt)


class SelfAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int, eps: float):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.q_proj = Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = Linear(heads * head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)

    def forward(self, x, cos, sin):
        B, L, _ = x.shape
        hd = self.head_dim
        q = self.q_norm(self.q_proj(x).view(B, L, self.heads, hd)).transpose(1, 2)      # [B,H,L,hd]
        k = self.k_norm(self.k_proj(x).view(B, L, self.kv_heads, hd)).transpose(1, 2)   # [B,KV,L,hd]
        v = self.v_proj(x).view(B, L, self.kv_heads, hd).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        rep = self.heads // self.kv_heads
        k = k[:, :, None].expand(B, self.kv_heads, rep, L, hd).reshape(B, self.heads, L, hd)
        v = v[:, :, None].expand(B, self.kv_heads, rep, L, hd).reshape(B, self.heads, L, hd)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(o.transpose(1, 2).reshape(B, L, self.heads * hd))


class MLP(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.gate_proj = Linear(hidden, inter, bias=False)
        self.up_proj = Linear(hidden, inter, bias=False)
        self.down_proj = Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, hidden, heads, kv_heads, head_dim, inter, eps):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden, eps)
        self.self_attn = SelfAttention(hidden, heads, kv_heads, head_dim, eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.mlp = MLP(hidden, inter)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin).to(x.dtype)
        return x + self.mlp(self.post_attention_layernorm(x)).to(x.dtype)


class QI21TextEncoder(nn.Module):
    def __init__(self, vocab_size=151936, hidden_size=4096, num_hidden_layers=36, num_attention_heads=32,
                 num_key_value_heads=8, head_dim=128, intermediate_size=12288, rms_norm_eps=1e-6,
                 rope_theta=5000000.0, max_len=512, io_fp32=False, residual_fp32=False, compute_fp32=False):
        super().__init__()
        self.hidden_size, self.head_dim, self.io_fp32 = hidden_size, head_dim, io_fp32
        self.residual_fp32 = residual_fp32 or compute_fp32
        self.compute_fp32 = compute_fp32
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(hidden_size, num_attention_heads, num_key_value_heads, head_dim, intermediate_size,
                          rms_norm_eps) for _ in range(num_hidden_layers)])
        if compute_fp32:
            for m in self.modules():
                if isinstance(m, (Linear, RMSNorm)):
                    m.compute_fp32 = True
        self.set_rope(rope_theta, max_len)

    def set_rope(self, theta: float, max_len: int):
        """(Re)build the fp32 RoPE tables. Plain attributes on the CPU, never meta, never bf16."""
        self.rope_theta, self.max_len = theta, max_len
        self.rope_cos, self.rope_sin = rope_tables(max_len, self.head_dim, theta)

    @classmethod
    def from_config(cls, text_cfg: dict, **kw):
        rope = text_cfg.get("rope_parameters") or text_cfg.get("rope_scaling") or {}
        theta = rope.get("rope_theta", text_cfg.get("rope_theta"))
        assert text_cfg.get("hidden_act", "silu") == "silu" and not text_cfg.get("attention_bias", False)
        return cls(vocab_size=text_cfg["vocab_size"], hidden_size=text_cfg["hidden_size"],
                   num_hidden_layers=text_cfg["num_hidden_layers"],
                   num_attention_heads=text_cfg["num_attention_heads"],
                   num_key_value_heads=text_cfg["num_key_value_heads"], head_dim=text_cfg["head_dim"],
                   intermediate_size=text_cfg["intermediate_size"], rms_norm_eps=text_cfg["rms_norm_eps"],
                   rope_theta=float(theta), **kw)

    def forward(self, input_ids):
        L = input_ids.shape[1]
        x = self.embed_tokens(input_ids)
        if self.residual_fp32:
            x = x.float()
        cos, sin = self.rope_cos[:L], self.rope_sin[:L]
        for layer in self.layers:
            x = layer(x, cos, sin)
        return x.float() if self.io_fp32 else x


def text_config(text_encoder_dir: str | Path) -> dict:
    cfg = json.load(open(Path(text_encoder_dir) / "config.json"))
    return cfg.get("text_config", cfg)


def load_qi21_text(text_encoder_dir: str | Path, dtype=torch.bfloat16, n_layers: int | None = None,
                   **kw) -> QI21TextEncoder:
    """Build from ``text_encoder/config.json`` and load the text stack from the sharded checkpoint.

    Only the ``model.language_model.*`` keys the module needs are read (embed_tokens + layers);
    the final norm, ``lm_head`` and the vision tower are skipped. Built on the meta device and
    the checkpoint tensors assigned, so peak memory is one copy of the weights in ``dtype``.
    """
    from safetensors import safe_open
    d = Path(text_encoder_dir)
    cfg = text_config(d)
    if n_layers is not None:
        cfg = dict(cfg, num_hidden_layers=n_layers)
    with torch.device("meta"):
        m = QI21TextEncoder.from_config(cfg, **kw)
    want = set(m.state_dict().keys())
    idx = json.load(open(d / "model.safetensors.index.json"))["weight_map"]
    by_file: dict[str, list[str]] = {}
    for k in sorted(want):
        f = idx.get(PREFIX + k)
        assert f is not None, f"checkpoint has no {PREFIX + k}"
        by_file.setdefault(f, []).append(k)
    sd = {}
    for f, keys in by_file.items():
        with safe_open(d / f, framework="pt") as fh:
            for k in keys:
                sd[k] = fh.get_tensor(PREFIX + k).to(dtype)
    missing, unexpected = m.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected, unexpected
    assert not missing, missing
    assert not any(p.is_meta for p in m.parameters())
    assert m.rope_cos.dtype == torch.float32 and not m.rope_cos.is_meta and not m.rope_sin.is_meta
    return m.to(dtype).eval()


def load_final_norm_weight(text_encoder_dir: str | Path) -> torch.Tensor:
    """``model.language_model.norm.weight`` — for the negative control only (the graph never applies it)."""
    from safetensors import safe_open
    d = Path(text_encoder_dir)
    idx = json.load(open(d / "model.safetensors.index.json"))["weight_map"]
    with safe_open(d / idx[PREFIX + "norm.weight"], framework="pt") as fh:
        return fh.get_tensor(PREFIX + "norm.weight")
