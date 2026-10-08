# d1 decoder, static form: the same network as lfm2_d1_decoder.Lfm2D1Decoder with no dynamic dimension in any
# input or state, so an AOT compile without --expect-frequent-reshapes specializes the graph once.
#
# Community port — NOT an Apple model.
#
# The dynamic form (lfm2_d1_decoder.py) takes every position seen so far (position_ids [1, p+S], 0..p+S-1) and keeps
# the KV sequence axis dynamic: the runtime specializes the graph again for every new position length unless the
# asset is compiled and loaded with expect-frequent-reshapes, which on the Mac adds an fp16 copy of every linear
# (round 6a) and on the iPhone 18 Pro did not load (round 8). This form changes only the full-attention layers:
#
#   inputs  input_ids     [1, S]        int32  as in the dynamic form
#           position_ids  [1, S]        int32  the S NEW positions p .. p+S-1 only
#           image_embeds  [2816, 2048]  fp16   as in the dynamic form
#   states  keyCache / valueCache [8, 1, 8, C, 64] (C static, the context; 4,096 by default), convState [22, 1, 2048, 2]
#   output  hidden        [1, S, 2048]  as in the dynamic form
#
# Attention reads all C slots of its layer's cache through the coreai SDPA composite with a bool mask built inside
# the graph from position_ids (no extra input): key slot j at absolute position kpos_j is visible to query i when
# kpos_j <= position_ids[i] (and, for `roll`, kpos_j >= 0). The composite takes the mask as bool (the tower's sdpa
# form, lfm2_vl_tower.py); the composite, not the bare softmax(q k^T) v chain, because the bare chain compiled for the
# GPU is wrong from 4,032 keys on (zoo knowledge/clef-flash-port.md "Attention over 4,032 keys") and here every call
# attends C keys.
#
# The new keys and values (after the per-head norms and RoPE at position_ids) enter the cache one of two ways
# (`kv_write`, recorded in metadata.json):
#
#   slice  written in place at slots p .. p+S-1 (`mutable_slice_update`, begin = position_ids[0, 0]): slot j holds
#          position j. The write index comes from runtime data, the form zoo
#          knowledge/coreai-beta-mpsgraph-kvwrite-bug.md records as trapping on the Mac GPU of the macOS export path.
#   roll   every layer's cache shifted left by S slots with the new S appended (slot j holds position p+S-C+j; slots
#          of negative positions are masked), the eight layers written back as one whole-state write per state at the
#          end of the call (begin / end constants, the conv state's form): no data-derived write index; a call reads
#          and writes the whole cache (2 x 8 x 8 x C x 64 fp16 = 67.1 MB at C = 4,096).
#
# Everything else is the parent's code path: the image-row gather, RoPE (`rope_cos_sin(position_ids)` = the dynamic
# form's q_pos), the conv mixer with its state and its one fused write, the MLP, the norms, the fp32 attention
# projections on an fp16 export, the Identity head. A row of T ids runs ceil(T / S) calls from zero states, call c
# with position_ids cS .. cS+S-1; it fits when ceil(T / S) * S <= C.
from __future__ import annotations

import torch

from coreai_models.models.macos.lfm2 import DECODE_STATE_NAMES, apply_rope, build_decode_state
from coreai_models.primitives._ops import mutable_slice_update
from coreai_models.primitives.macos.sdpa import SDPA
from lfm2_d1_decoder import INPUT_NAMES, N_IMAGE_TOKENS, OUTPUT_NAMES, Lfm2D1Decoder

CONTEXT = 4096
KV_WRITES = ("slice", "roll")

__all__ = ["CONTEXT", "KV_WRITES", "Lfm2D1StaticDecoder", "make_static", "row_fits"]


def row_fits(T: int, S: int, context: int) -> bool:
    """A row of T ids fits the static graph: its last call ends at or below slot C - 1."""
    return -(-T // S) * S <= context


class Lfm2D1StaticDecoder(Lfm2D1Decoder):
    """Lfm2D1Decoder with position_ids [1, S] and a static KV cache of `context` slots; contract in the header."""

    def __init__(self, config, n_image_tokens: int = N_IMAGE_TOKENS, context: int = CONTEXT, kv_write: str = "slice"):
        super().__init__(config, n_image_tokens=n_image_tokens)
        self._init_static(context, kv_write)

    def _init_static(self, context: int, kv_write: str) -> None:
        if kv_write not in KV_WRITES:
            raise ValueError(f"kv_write {kv_write!r} not in {KV_WRITES}")
        if context < 2:
            raise ValueError("context must be >= 2")
        self.context = int(context)
        self.kv_write = kv_write
        for layer in self.model.layers:
            if layer.is_full:
                a = layer.self_attn
                a.static_sdpa = SDPA(scale=a.head_dim ** -0.5, is_causal=False)

    @classmethod
    def from_hf(cls, hf_id_or_dir: str, target_dtype: torch.dtype = torch.float16, n_image_tokens: int = N_IMAGE_TOKENS,
                fp32_attn_proj: bool = True, context: int = CONTEXT, kv_write: str = "slice") -> "Lfm2D1StaticDecoder":
        """The dynamic form's loader (load_report included), then this form's attention."""
        return make_static(Lfm2D1Decoder.from_hf(hf_id_or_dir, target_dtype=target_dtype, n_image_tokens=n_image_tokens,
                                                 fp32_attn_proj=fp32_attn_proj), context, kv_write)

    # -- the graph ----------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,     # [1, S] int32
        position_ids: torch.Tensor,  # [1, S] int32: the S new positions
        image_embeds: torch.Tensor,  # [N, hidden]
        k_cache: torch.Tensor,       # [n_full, 1, n_kv, C, head_dim]
        v_cache: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> torch.Tensor:
        """-> hidden [1, S, hidden]: the final-norm hidden state at every query position."""
        if self.last_token_only:
            raise ValueError("the d1 decoder returns the final-norm hidden at every position only")
        V = self.config.vocab_size
        N = self.n_image_tokens
        b, s = input_ids.shape
        is_img = input_ids >= V
        slot = (input_ids - V).clamp(0, N - 1).reshape(-1)
        e_txt = self.model.embed_tokens(input_ids.clamp(0, V - 1))
        e_img = image_embeds.index_select(0, slot).reshape(b, s, -1)
        embeds = torch.where(is_img.unsqueeze(-1), e_img.to(e_txt.dtype), e_txt)
        return self.lm_head(self._stack(embeds, position_ids, k_cache, v_cache, conv_state))

    def key_positions(self, position_ids: torch.Tensor, C: int, S: int) -> torch.Tensor:
        """[1, C] int32: the absolute position each cache slot holds after this call's write."""
        j = torch.arange(C, dtype=torch.int32).reshape(1, C)
        if self.kv_write == "slice":
            return j
        return j + (position_ids.narrow(1, 0, 1).reshape(1, 1) + (S - C))

    def attention_mask(self, position_ids: torch.Tensor, C: int, S: int) -> torch.Tensor:
        """[1, 1, S, C] bool: key slot j visible to query i (causal over absolute positions, written slots only)."""
        kpos = self.key_positions(position_ids, C, S)
        visible = kpos <= position_ids.reshape(S, 1)
        if self.kv_write == "roll":
            visible = torch.logical_and(visible, kpos >= 0)
        return visible.reshape(1, 1, S, C)

    def _attention(self, a, x, cos, sin, position_ids, k_cache, v_cache, full_idx: int, mask):
        """One full-attention layer: projections, per-head norms, RoPE, the cache write, SDPA over the C slots."""
        b, s, _ = x.shape
        H, HKV, D = a.n_heads, a.n_kv_heads, a.head_dim
        C = k_cache.shape[3]
        xp = x.to(a.q_proj.weight.dtype)
        q = a.q_layernorm(a.q_proj(xp).to(x.dtype).view(b, s, H, D)).transpose(1, 2)
        k = a.k_layernorm(a.k_proj(xp).to(x.dtype).view(b, s, HKV, D)).transpose(1, 2)
        v = a.v_proj(xp).to(x.dtype).view(b, s, HKV, D).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        if self.kv_write == "slice":
            p = position_ids.narrow(1, 0, 1).reshape(1)
            zero = torch.tensor((0,), dtype=torch.int32)
            lo = torch.tensor((full_idx,), dtype=torch.int32)
            begin = torch.cat([lo, zero, zero, p, zero])
            end = torch.cat([lo + 1, torch.tensor((1,), dtype=torch.int32), torch.tensor((HKV,), dtype=torch.int32),
                             p + s, torch.tensor((D,), dtype=torch.int32)])
            mutable_slice_update(x=k_cache, update=k.unsqueeze(0), begin=begin, end=end)
            mutable_slice_update(x=v_cache, update=v.unsqueeze(0), begin=begin, end=end)
            kk = k_cache.narrow(0, full_idx, 1).squeeze(0)
            vv = v_cache.narrow(0, full_idx, 1).squeeze(0)
        else:
            kk = torch.cat([k_cache.narrow(0, full_idx, 1).squeeze(0).narrow(2, s, C - s), k], dim=2)
            vv = torch.cat([v_cache.narrow(0, full_idx, 1).squeeze(0).narrow(2, s, C - s), v], dim=2)
        out = a.static_sdpa(q, kk, vv, attn_mask=mask)
        out = out.transpose(1, 2).reshape(b, s, H * D)
        return a.out_proj(out.to(a.out_proj.weight.dtype)).to(x.dtype), kk, vv

    def _stack(self, h, position_ids, k_cache, v_cache, conv_state):
        m = self.model
        S = h.shape[1]
        C = k_cache.shape[3]
        cos, sin = m.rope_cos_sin(position_ids)
        mask = self.attention_mask(position_ids, C, S)
        new_convs, new_k, new_v = [], [], []
        full_idx = conv_idx = 0
        for layer in m.layers:
            normed = layer.operator_norm(h)
            if layer.is_full:
                r, kk, vv = self._attention(layer.self_attn, normed, cos, sin, position_ids, k_cache, v_cache, full_idx,
                                            mask)
                new_k.append(kk)
                new_v.append(vv)
                full_idx += 1
            else:
                r, new_conv = layer.conv(normed, conv_state.narrow(0, conv_idx, 1).squeeze(0))
                new_convs.append(new_conv)
                conv_idx += 1
            h = h + r
            h = h + layer.feed_forward(layer.ffn_norm(h))
        zero = torch.tensor((0,), dtype=torch.int32)
        mutable_slice_update(x=conv_state, update=torch.cat([n.unsqueeze(0) for n in new_convs], dim=0),
                             begin=torch.cat([zero, zero, zero, zero]),
                             end=torch.tensor(tuple(conv_state.shape), dtype=torch.int32))
        if self.kv_write == "roll":
            for cache, rows in ((k_cache, new_k), (v_cache, new_v)):
                mutable_slice_update(x=cache, update=torch.cat([r.unsqueeze(0) for r in rows], dim=0),
                                     begin=torch.cat([zero, zero, zero, zero, zero]),
                                     end=torch.tensor(tuple(cache.shape), dtype=torch.int32))
        return m.embedding_norm(h)

    # -- export -------------------------------------------------------------

    def build_static_export_spec(self, target_dtype: torch.dtype, query_len: int, trace_offset: int = 64) -> dict:
        """Every input and state at a fixed shape: position_ids [1, S] (traced at trace_offset..), the KV caches at
        `context` slots, no dynamic dimension."""
        if query_len < 2:
            raise ValueError("this graph has no S=1 function: query_len must be >= 2")
        if trace_offset + query_len > self.context:
            raise ValueError(f"trace_offset + S = {trace_offset + query_len} exceeds the context {self.context}")
        cfg = self.config
        state = build_decode_state(cfg, max_seq_len=self.context, dtype=target_dtype)
        reference_inputs = {
            "input_ids": torch.randint(1, cfg.vocab_size, (1, query_len), dtype=torch.int32),
            "position_ids": torch.arange(trace_offset, trace_offset + query_len, dtype=torch.int32).unsqueeze(0),
            "image_embeds": torch.zeros(self.n_image_tokens, cfg.hidden_size, dtype=target_dtype),
            "k_cache": state["k_cache"], "v_cache": state["v_cache"], "conv_state": state["conv_state"],
        }
        return {"reference_inputs": reference_inputs, "dynamic_shapes": {k: None for k in reference_inputs},
                "input_names": INPUT_NAMES, "output_names": OUTPUT_NAMES, "state_names": DECODE_STATE_NAMES}


def make_static(model: Lfm2D1Decoder, context: int = CONTEXT, kv_write: str = "slice") -> Lfm2D1StaticDecoder:
    """A loaded Lfm2D1Decoder (its weights, load_report and module names kept) turned into the static form in place."""
    if not isinstance(model, Lfm2D1Decoder):
        raise TypeError(f"expected an Lfm2D1Decoder, got {type(model).__name__}")
    model.__class__ = Lfm2D1StaticDecoder
    model._init_static(context, kv_write)
    return model
