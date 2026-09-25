"""Bonsai 2 27B (prism-ml/Ternary-Bonsai-2-27B, PQ2_0 GGUF) on the zoo's Qwen3.5 hybrid decoder.

Community port — NOT an Apple model. Builds `Qwen3_5ForCausalLMStateful` from the GGUF header,
then replaces every folded linear with a `TernaryLinear128` (PQ2_0 words used as-is, no
requantisation) fed by a per-site `HadamardSite`, the embedding with the packed
gather+inverse-transform kernel, and the untied head with a ternary linear. The GDN's a/b
projections, conv, A_log, dt_bias and every norm stay float, as in the pack.

Loader conventions (all from knowledge/bonsai-ternary-hadamard.md §6):
  * RMSNorm gains are stored as `w + 1` in the GGUF for every norm except the GDN's gated norm;
    the zoo's RMSNormPlusOne adds 1 at runtime, so subtract 1 on load.
  * GDN value heads are in llama.cpp's *tiled* order for the V rows of attn_qkv, attn_gate,
    ssm_alpha/beta, ssm_a, ssm_dt.bias and the V channels of ssm_conv1d; permute to HF grouped
    order (`vperm`). ssm_out's input columns are already grouped (`gdn_v_grouped`) — untouched.
  * ssm_a is stored as -exp(A_log); attn_q is [query | gate] per head like HF (no permutation).
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn

from coreai_models.models.macos.bonsai_fused_sites_metal import (
    add_norm_fwht, build_add_norm_fwht_kernel, build_norm_fwht_kernel, build_swiglu_fwht_kernel,
    norm_fwht, seed_site, swiglu_fwht,
)
from coreai_models.models.macos.bonsai_gdn_step_metal import build_gdn_step_kernel, fused_gdn_step
from coreai_models.models.macos.bonsai_hadamard_metal import (
    BLOCK, HadamardSite, PackedEmbedding, build_embed_kernel, build_fwht_kernel,
)
from coreai_models.models.macos.bonsai_ternary_metal import (
    TernaryLinear128, build_gemm_kernel, build_mv_kernel, build_mv_multi_kernel, pq2_words_and_scales,
    ternary_multi,
)
from coreai_models.models.macos.qwen3_5 import (
    Qwen3_5Config, Qwen3_5DecoderLayer, Qwen3_5ForCausalLMStateful, Qwen3_5FullAttention,
    Qwen3_5GatedDeltaNet, Qwen3_5Model, apply_rope,
)
from coreai_models.primitives.macos.mlp import MLP

HF_ID = "Qwen/Qwen3.8-27B"
PQ2_0 = 142


class BonsaiKernels:
    """The kernels one export registers: matvec, FWHT, embed, the fused S=1 GDN step, and with a
    prefill chunk also the tiled GEMM and the zoo's fp32 GDN chunk-scan (`qwen3_5_gdn_metal`)."""

    def __init__(self, chunk: int | list[int] | None = None, rms_eps: float = 1e-6,
                 round_chunk_state_each_token: bool = False) -> None:
        chunks = sorted({int(c) for c in ([chunk] if isinstance(chunk, int) else (chunk or [])) if c > 1}, reverse=True)
        self.chunks = chunks
        self.chunk = chunks[0] if chunks else None       # the largest: sizes the GDN scan's buffers
        self.mv = build_mv_kernel()
        self.mv2 = build_mv_multi_kernel(2)      # gate+up, qkv+z: one dispatch per pair (round 3)
        self.mv3 = build_mv_multi_kernel(3)      # q+k+v of the full-attention layers
        self.gemms = {c: build_gemm_kernel(c) for c in chunks}   # one tiled GEMM per static query length
        self.gemm = self.gemms.get(self.chunk)
        self.fwht = build_fwht_kernel()
        self.embed = build_embed_kernel()
        self.gdn_chunk = build_gdn_chunk_kernel_sh(
            chunk_max=self.chunk, round_state_each_token=round_chunk_state_each_token
        ) if self.chunk else None
        self.gdn_step = build_gdn_step_kernel(rms_eps)
        # S=1 site fusions (round 3): residual add + pre-norm + transform, and swiglu + transform
        self.add_norm_fwht = build_add_norm_fwht_kernel(rms_eps)
        self.norm_fwht = build_norm_fwht_kernel(rms_eps)
        self.swiglu_fwht = build_swiglu_fwht_kernel()

    def all(self) -> list:
        return [k for k in (self.mv, self.mv2, self.mv3, *self.gemms.values(), self.fwht, self.embed,
                            self.gdn_chunk, self.gdn_step, self.add_norm_fwht, self.norm_fwht,
                            self.swiglu_fwht) if k is not None]


def build_gdn_chunk_kernel_sh(name: str = "bonsai_gdn_chunk_sh", max_dk: int = 128,
                              chunk_max: int = 64, round_state_each_token: bool = False):
    """The zoo's fp32 GDN chunk-scan kernel with G/BETA read from the [S, h] layout.

    `qwen3_5_gdn_metal` transposes g/beta to [h, S] and `.contiguous()`s them; the Core AI graph
    optimizer elides that copy and the custom-kernel op then asserts on the strided view
    (`extentAtDimensionIndex:0 (48) should be 1`). The projections produce [S, h] contiguous
    already, so index it as G[hh, t] (DSL reversed) and pass it untransposed.
    """
    from coreai_torch import MetalParameter, TorchMetalKernel
    from coreai_models.models.macos.qwen3_5_gdn_metal import _GDN_CHUNK_SRC

    src = _GDN_CHUNK_SRC.replace("G[t, hh]", "G[hh, t]").replace("BETA[t, hh]", "BETA[hh, t]")
    assert "G[hh, t]" in src and "BETA[hh, t]" in src
    if round_state_each_token:
        marker = "        OUT[c, t, hh] = TYPE(oc);\n"
        rounded = marker + "        for (uint d = 0; d < dk; ++d) st[d] = float(half(st[d]));\n"
        assert src.count(marker) == 1
        src = src.replace(marker, rounded)
        name += "_roundstate"

    def _torch_defn(QN: torch.Tensor, KN: torch.Tensor, V: torch.Tensor, G: torch.Tensor,
                    BETA: torch.Tensor, S0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = KN.shape[0]
        dv = V.shape[-1]
        return QN.new_zeros(h, chunk_max, dv), S0.clone()

    return TorchMetalKernel(name, input_names=["QN", "KN", "V", "G", "BETA", "S0"],
                            result_names=["OUT", "SNEW"], src=src.replace("__MAXDK__", str(max_dk)),
                            torch_defn=_torch_defn,
                            metal_params=[MetalParameter("gid", "uint2", "thread_position_in_grid")],
                            template_dtypes={"QN": "TYPE"})


class BonsaiGDNChunk(nn.Module):
    """`MetalGDNChunk` with g/beta handed over in their native [S, h] layout (no transpose copy)."""

    coreai_externalize_specs: tuple = ()

    def __init__(self, kernel, use_qk_l2_norm: bool = True, chunk_max: int = 64) -> None:
        super().__init__()
        self.kernel, self.use_qk_l2_norm, self.chunk_max = kernel, use_qk_l2_norm, chunk_max

    def forward(self, q, k, v, g, beta, S0):
        def l2norm(x):
            return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)
        b, h, S, dk = k.shape
        dv = v.shape[-1]
        if self.use_qk_l2_norm:
            q = l2norm(q)
            k = l2norm(k)
        q = q * (dk ** -0.5)
        qn = q[0].contiguous(); kn = k[0].contiguous(); vv = v[0].contiguous()
        gg = g[0].transpose(0, 1)          # [S, h]: the layout the projection produced (contiguous)
        bb = beta[0].transpose(0, 1)
        s0 = S0[0].contiguous()
        out, snew = self.kernel(qn, kn, vv, gg, bb, s0,
                                threads_per_grid=(dv, h, 1), threads_per_thread_group=(dv, 1, 1),
                                result_shapes=[[h, self.chunk_max, dv], [h, dk, dv]])
        out = out[:, :S, :].unsqueeze(0).transpose(1, 2)     # [1, S, h, dv]
        return out, snew.unsqueeze(0)


class BonsaiGDN(Qwen3_5GatedDeltaNet):
    """The zoo's GDN layer with a third trace-time mode: `use_fused_step` routes the S=1 stateful
    forward through the fused `bonsai_gdn_step` kernel (one dispatch between the projections,
    a/b projections included). Instances are re-classed in the loader; nothing else changes."""

    use_fused_step: bool = False
    step_kernel = None

    def forward(self, x, conv_in=None, rec_in=None, nrm=None):
        if self.use_fused_step and conv_in is not None and x.shape[1] == 1:
            kern = getattr(self, "_kern", None)          # absent on a bare layer (the kernel gate)
            return fused_gdn_step(self, self.step_kernel, x, conv_in, rec_in, nrm,
                                  mv2=kern.mv2 if kern is not None else None)
        return super().forward(x, conv_in, rec_in)


class BonsaiMLP(MLP):
    """The zoo's MLP with the S=1 fused down_proj input: up ⊙ silu(gate) and the h_down transform
    in one kernel. `x` arrives already transformed for the h_mlp site (seeded memo)."""

    use_fused_sites: bool = False

    def forward(self, x):
        if self.use_fused_sites and x.shape[1] == 1:
            up, gate = ternary_multi(self._kern.mv2, x, [self.up_proj, self.gate_proj])
            y = seed_site(self._h_down, swiglu_fwht(self._kern.swiglu_fwht, up, gate, self._h_down.signs))
            return self.down_proj(y)
        return super().forward(x)


class BonsaiAttention(Qwen3_5FullAttention):
    """The zoo's full attention with the S=1 q/k/v projections in one triple matvec; the rest of
    the block is the zoo's forward verbatim. `x` arrives transformed for the h_in site."""

    use_fused_sites: bool = False

    def forward(self, x, cos, sin, kv_cache=None, offset=0, seq_len=None, full_idx=0):
        if not (self.use_fused_sites and kv_cache is not None and x.shape[1] == 1):
            return super().forward(x, cos, sin, kv_cache, offset, seq_len, full_idx)
        b, s, _ = x.shape
        H, HKV, D = self.n_heads, self.n_kv_heads, self.head_dim
        qg, kk, vv = ternary_multi(self._kern.mv3, x, [self.q_proj, self.k_proj, self.v_proj])
        qg = qg.view(b, s, H, D * 2)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(b, s, H * D)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(kk.view(b, s, HKV, D)).transpose(1, 2)
        v = vv.view(b, s, HKV, D).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        k, v = kv_cache.update_and_fetch(full_idx, offset, k, v, seq_len=seq_len, query_len=s)
        out = self.sdpa(q, k, v)
        out = out.transpose(1, 2).reshape(b, s, H * D)
        out = out * torch.sigmoid(gate)
        return self.o_proj(out)


class BonsaiDecoderLayer(Qwen3_5DecoderLayer):
    """Decoder layer with the S=1 fused sites: the residual add, the pre-norm and the Hadamard
    transform of each site's input are one kernel; the layer takes and returns the residual
    stream *unadded* as (h, r) so the add lands in the next site's kernel."""

    use_fused_sites: bool = False

    def forward_fused(self, x, r_in, cos, sin, kv_cache, conv_cache, rec_cache, offset, seq_len,
                      full_idx, lin_idx):
        k = self._kern
        if r_in is None:                        # first layer: the embedding is the residual
            h = x
            nrm, y = norm_fwht(k.norm_fwht, x, self.input_layernorm.weight, self.h_in.signs)
        else:
            h, nrm, y = add_norm_fwht(k.add_norm_fwht, x, r_in, self.input_layernorm.weight, self.h_in.signs)
        seed_site(self.h_in, y)
        if self.is_full:
            r = self.self_attn(y, cos, sin, kv_cache, offset, seq_len, full_idx)
        else:
            conv_in = conv_cache.states.narrow(0, lin_idx, 1).squeeze(0)
            rec_in = rec_cache.states.narrow(0, lin_idx, 1).squeeze(0)
            r, new_conv, new_rec = self.linear_attn(y, conv_in, rec_in, nrm=nrm)
            conv_cache.update_states(lin_idx, new_conv)
            rec_cache.update_states(lin_idx, new_rec)
        h2, _, y2 = add_norm_fwht(k.add_norm_fwht, h, r, self.post_attention_layernorm.weight, self.h_mlp.signs)
        seed_site(self.h_mlp, y2)
        return h2, self.mlp(y2)


class BonsaiModel(Qwen3_5Model):
    """The decoder stack with the S=1 fused-site walk (`use_fused_sites`): layers pass the
    residual stream unadded; the final add + norm + head transform is the last fused kernel and
    the head's site is seeded with its result (the caller's lm_head reads it from the memo)."""

    use_fused_sites: bool = False

    def forward_stateful_core(self, inputs_embeds, position_ids, kv_cache, conv_cache, rec_cache):
        if not (self.use_fused_sites and inputs_embeds.shape[1] == 1):
            return super().forward_stateful_core(inputs_embeds, position_ids, kv_cache, conv_cache, rec_cache)
        seq_len = position_ids.shape[1]
        offset = seq_len - 1
        cos, sin = self.rope_cos_sin(position_ids.narrow(1, offset, 1))
        h, r = inputs_embeds, None
        full_idx = lin_idx = 0
        for layer in self.layers:
            h, r = layer.forward_fused(h, r, cos, sin, kv_cache, conv_cache, rec_cache, offset, seq_len,
                                       full_idx, lin_idx)
            if layer.is_full:
                full_idx += 1
            else:
                lin_idx += 1
        _, _, y = add_norm_fwht(self._kern.add_norm_fwht, h, r, self.norm.weight, self._head_site.signs)
        return seed_site(self._head_site, y)


def set_gdn_mode(model: nn.Module, mode: str) -> None:
    """'step': loop-free single-step recurrence in graph ops (the S=1 decode trace);
    'chunk': the fp32 Metal chunk-scan kernel (the S=C prefill trace; also runs at S=1);
    'fused': the fused S=1 kernels: GDN step + a/b, add+norm+transform per site, swiglu+transform
    (decode; S>1 falls back to the chunk path). Flip between traces."""
    model.model.use_fused_sites = mode == "fused"
    for layer in model.model.layers:
        layer.use_fused_sites = mode == "fused"
        layer.mlp.use_fused_sites = mode == "fused"
        if layer.is_full:
            layer.self_attn.use_fused_sites = mode == "fused"
            continue
        la = layer.linear_attn
        la.use_fused_step = False
        if mode == "step":
            la.use_metal_chunk, la.use_loopfree_step = False, True
        elif mode in ("chunk", "fused"):
            if la.metal_chunk is None:
                raise RuntimeError("model was loaded without a prefill chunk")
            la.use_metal_chunk, la.use_loopfree_step = True, False
            if mode == "fused":
                if la.step_kernel is None:
                    raise RuntimeError("model was loaded without the fused GDN step kernel")
                la.use_fused_step = True
        else:
            raise ValueError(mode)


def config_from_gguf(reader, num_layers: int | None = None) -> Qwen3_5Config:
    f = {k: v.contents() for k, v in reader.fields.items() if not k.startswith("tokenizer.")}
    if f.get("general.architecture") != "qwen35":
        raise ValueError(f"expected a qwen35 GGUF, got {f.get('general.architecture')!r}")
    g = lambda k: f["qwen35." + k]
    nv, nk = int(g("ssm.time_step_rank")), int(g("ssm.group_count"))
    n_layers = int(g("block_count"))
    interval = int(g("full_attention_interval"))
    layer_types = ["full_attention" if (i % interval == interval - 1) else "linear_attention"
                   for i in range(n_layers)]
    if num_layers is not None:
        n_layers, layer_types = num_layers, layer_types[:num_layers]
    vocab = int(reader.tensors[[t.name for t in reader.tensors].index("token_embd.weight")].shape[1])
    return Qwen3_5Config(
        hidden_size=int(g("embedding_length")), num_hidden_layers=n_layers, vocab_size=vocab,
        intermediate_size=int(g("feed_forward_length")),
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")), tie_word_embeddings=False,
        head_dim=int(g("attention.key_length")), num_attention_heads=int(g("attention.head_count")),
        num_key_value_heads=int(g("attention.head_count_kv")),
        partial_rotary_factor=int(g("rope.dimension_count")) / int(g("attention.key_length")),
        rope_theta=float(g("rope.freq_base")), linear_num_key_heads=nk, linear_num_value_heads=nv,
        linear_key_head_dim=int(g("ssm.state_size")), linear_value_head_dim=int(g("ssm.inner_size")) // nv,
        linear_conv_kernel_dim=int(g("ssm.conv_kernel")), full_attention_interval=interval,
        layer_types=layer_types,
    )


def hadamard_signs_from_gguf(reader) -> tuple[int, dict[int, torch.Tensor]]:
    f = {k: v.contents() for k, v in reader.fields.items() if k.startswith("prism.hadamard.")}
    if f.get("prism.hadamard.version") != 1 or f.get("prism.hadamard.block_size") != BLOCK:
        raise ValueError(f"unsupported Hadamard contract: {f}")
    if f.get("prism.hadamard.transform") != "normalized-sylvester-walsh-hadamard" \
            or f.get("prism.hadamard.axis") != "input-last-dimension" \
            or f.get("prism.hadamard.sign_mode") != "explicit":
        raise ValueError(f"unsupported Hadamard contract: {f}")
    if not f.get("prism.hadamard.gdn_v_grouped", False):
        raise ValueError("this loader assumes gdn_v_grouped (ssm_out columns in grouped order)")
    signs, off = {}, 0
    values = f["prism.hadamard.sign_values"]
    for w in f["prism.hadamard.sign_widths"]:
        signs[int(w)] = torch.tensor(values[off:off + w], dtype=torch.float32)
        off += w
    if off != len(values):
        raise ValueError("trailing sign values")
    folded = set(f["prism.hadamard.weight_names"])
    inverse = set(f.get("prism.hadamard.inverse_weight_names", []))
    if inverse != {"token_embd.weight"}:
        raise ValueError(f"unexpected inverse manifest {inverse}")
    return BLOCK, signs, folded


class _Src:
    """Tensor access over the GGUF: packed words/scales for PQ2_0, float32 for the rest."""

    def __init__(self, reader) -> None:
        self.r = reader
        self.t = {t.name: t for t in reader.tensors}

    def shape(self, name):                      # (N rows, K cols)
        s = self.t[name].shape
        return (int(s[1]), int(s[0])) if len(s) == 2 else (int(s[0]),)

    def packed(self, name: str, row_perm: np.ndarray | None = None):
        t = self.t[name]
        if int(t.tensor_type) != PQ2_0:
            raise ValueError(f"{name}: expected PQ2_0, got {t.tensor_type}")
        n, k = self.shape(name)
        qp, d = pq2_words_and_scales(np.asarray(t.data), n, k)
        if row_perm is not None:
            idx = torch.from_numpy(row_perm)
            qp, d = qp[idx].contiguous(), d[idx].contiguous()
        return qp, d

    def float(self, name: str) -> torch.Tensor:
        from gguf.quants import dequantize
        t = self.t[name]
        if int(t.tensor_type) == PQ2_0:
            raise ValueError(f"{name} is PQ2_0; use packed()")
        a = np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(a).copy())


def _vperm(nv: int, nk: int, unit: int) -> np.ndarray:
    """llama.cpp tiled V order -> HF grouped order (a_hf = a_tiled[vperm]); mlx runtime.py:171."""
    return np.arange(nv * unit).reshape(nv // nk, nk, unit).transpose(1, 0, 2).reshape(-1)


def load_bonsai2_from_gguf(gguf_path: str, *, num_layers: int | None = None,
                           chunk: int | list[int] | None = None, dtype: torch.dtype = torch.float16,
                           head_rows: int | None = None,
                           round_chunk_state_each_token: bool = False):
    """Build the stateful decoder from the PQ2_0 GGUF. Returns (model, kernels)."""
    import gguf

    reader = gguf.GGUFReader(gguf_path)
    cfg = config_from_gguf(reader, num_layers)
    _, signs, folded = hadamard_signs_from_gguf(reader)
    src = _Src(reader)
    kern = BonsaiKernels(chunk, rms_eps=cfg.rms_norm_eps,
                         round_chunk_state_each_token=round_chunk_state_each_token)
    d_model = cfg.hidden_size

    with torch.device("meta"):
        model = Qwen3_5ForCausalLMStateful(cfg)

    def site(width: int) -> HadamardSite:
        return HadamardSite(signs[width], kern.fwht)

    def tern(name: str, s: HadamardSite | None, row_perm=None) -> TernaryLinear128:
        if name not in folded:
            raise ValueError(f"{name} is not in the Hadamard fold manifest")
        qp, d = src.packed(name, row_perm)
        return TernaryLinear128(qp, d, kern.mv, kern.gemms, site=s)

    def norm_gain(name: str) -> nn.Parameter:           # stored as w+1 -> RMSNormPlusOne wants w
        return nn.Parameter((src.float(name) - 1.0).to(dtype), requires_grad=False)

    def param(t: torch.Tensor) -> nn.Parameter:
        return nn.Parameter(t.to(dtype), requires_grad=False)

    nv, nk = cfg.linear_num_value_heads, cfg.linear_num_key_heads
    hd, hk = cfg.linear_value_head_dim, cfg.linear_key_head_dim
    qk_rows = 2 * nk * hk
    perm_v = _vperm(nv, nk, hd)
    perm_h = _vperm(nv, nk, 1)
    conv_perm = np.concatenate([np.arange(qk_rows), qk_rows + perm_v])

    with torch.no_grad():
        # embedding (rotated rows, inverse after lookup) and untied head
        qp, d = src.packed("token_embd.weight")
        model.model.embed_tokens = PackedEmbedding(qp, d, signs[d_model], kern.embed)
        model.h_head = site(d_model)
        model.lm_head = tern("output.weight", model.h_head)
        if head_rows:                                   # debug: a truncated head (timing probes only)
            lh = model.lm_head
            lh.qp = lh.qp[:head_rows].contiguous(); lh.d = lh.d[:head_rows].contiguous(); lh.N = head_rows
            model.config = replace(model.config, vocab_size=head_rows)
        model.model.norm.weight = norm_gain("output_norm.weight")

        model.model.__class__ = BonsaiModel
        object.__setattr__(model.model, "_kern", kern)
        object.__setattr__(model.model, "_head_site", model.h_head)     # not a child: h_head owns it

        for i, layer in enumerate(model.model.layers):
            p = f"blk.{i}."
            layer.__class__ = BonsaiDecoderLayer
            object.__setattr__(layer, "_kern", kern)
            layer.input_layernorm.weight = norm_gain(p + "attn_norm.weight")
            layer.post_attention_layernorm.weight = norm_gain(p + "post_attention_norm.weight")
            layer.h_in = site(d_model)
            layer.h_mlp = site(d_model)
            layer.h_down = site(cfg.intermediate_size)
            if layer.is_full:
                a = layer.self_attn
                a.__class__ = BonsaiAttention
                object.__setattr__(a, "_kern", kern)
                layer.h_out = site(cfg.num_attention_heads * cfg.head_dim)
                a.q_proj = tern(p + "attn_q.weight", layer.h_in)
                a.k_proj = tern(p + "attn_k.weight", layer.h_in)
                a.v_proj = tern(p + "attn_v.weight", layer.h_in)
                a.o_proj = tern(p + "attn_output.weight", layer.h_out)
                a.q_norm.weight = norm_gain(p + "attn_q_norm.weight")
                a.k_norm.weight = norm_gain(p + "attn_k_norm.weight")
            else:
                g = layer.linear_attn
                g.__class__ = BonsaiGDN                 # adds the fused S=1 mode (set_gdn_mode)
                g.step_kernel = kern.gdn_step
                object.__setattr__(g, "_kern", kern)
                layer.h_out = site(nv * hd)
                g.in_proj_qkv = tern(p + "attn_qkv.weight", layer.h_in, row_perm=conv_perm)
                g.in_proj_z = tern(p + "attn_gate.weight", layer.h_in, row_perm=perm_v)
                g.out_proj = tern(p + "ssm_out.weight", layer.h_out)
                g.in_proj_a.weight = param(src.float(p + "ssm_alpha.weight")[torch.from_numpy(perm_h)])
                g.in_proj_b.weight = param(src.float(p + "ssm_beta.weight")[torch.from_numpy(perm_h)])
                conv = src.float(p + "ssm_conv1d.weight")               # [conv_dim, kernel]
                g.conv1d.weight = param(conv[torch.from_numpy(conv_perm)].unsqueeze(1))
                ssm_a = src.float(p + "ssm_a")
                if not bool((ssm_a < 0).all()):
                    raise ValueError(f"{p}ssm_a must be negative (stored as -exp(A_log))")
                # A_log and dt_bias stay fp32 (the pack's precision; the zoo's forward reads them
                # through .float() and the fused step kernel binds the tensor's own dtype). Cast
                # to fp16 they cost 2x the decay error: exp(fp16(log A)) is off by up to 1e-3
                # relative, and that factor multiplies the recurrent state on every token.
                g.A_log = nn.Parameter(torch.log(-ssm_a)[torch.from_numpy(perm_h)].float(), requires_grad=False)
                g.dt_bias = nn.Parameter(src.float(p + "ssm_dt.bias")[torch.from_numpy(perm_h)].float(),
                                         requires_grad=False)
                g.norm.weight = param(src.float(p + "ssm_norm.weight"))
                g.use_loopfree_step = True
                if kern.gdn_chunk is not None:
                    g.metal_chunk = BonsaiGDNChunk(kern.gdn_chunk, use_qk_l2_norm=g.gdu.use_qk_l2_norm,
                                                   chunk_max=kern.chunk)
            m = layer.mlp
            m.__class__ = BonsaiMLP
            object.__setattr__(m, "_kern", kern)
            object.__setattr__(m, "_h_down", layer.h_down)                # shared with the layer
            m.gate_proj = tern(p + "ffn_gate.weight", layer.h_mlp)
            m.up_proj = tern(p + "ffn_up.weight", layer.h_mlp)
            m.down_proj = tern(p + "ffn_down.weight", layer.h_down)

    model.model.reset_buffers()
    meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if meta:
        raise RuntimeError(f"tensors still on meta: {meta[:8]}")
    model.eval()
    return model, kern
