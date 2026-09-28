# Qwen3.5-family VL decoder on the ids-input + static `image_embeds` contract.
#
# Community port — NOT an Apple model.
#
# The overlay's only Qwen3.5 VL decoder, `Qwen3_5VLStatefulEmbeds`, takes host-built
# `inputs_embeds` plus three host-fed M-RoPE planes, so the embed table ships next to the
# graph and no engine can drive it. This module is the other variant: the Qwen3-VL engine
# contract (`qwen3_vl_pipelined.Qwen3VLPipelinedForCausalLM`) on the Qwen3.5 hybrid
# (GatedDeltaNet + gated full attention) stack. The embed table stays in the graph, image
# rows ride a static input, and the interleaved M-RoPE is derived in-graph from (ids, position).
# Nothing here is specific to one checkpoint: the config and the image grid decide everything.
#
# Graph (S = 1, the decode-only loop-free export; states mutate in place):
#
#   inputs  input_ids          [1, 1]       int32  static; image token k (row-major) = V + k
#           position_ids       [1, seq]     int32  dynamic; the cache ramp 0..seq-1 (offset = seq-1)
#           image_embeds       [NMAX, h]    float  static; tower rows 0..N-1, rows N.. zero
#           image_rc           [NMAX, 2]    int32  static; (row, col) of slot k on the merged grid
#           rope_shift_start   [1]          int32  static; index of <|vision_end|>
#           rope_shift_amount  [1]          int32  static; N - max(H, W)
#   states  keyCache/valueCache [n_full, 1, n_kv, ctx, head_dim]   (ctx dynamic)
#           convState [n_lin, 1, conv_dim, kernel-1], recState [n_lin, 1, n_v, dk, dv]
#   output  logits             [1, 1, V]
#
#   per token:  is_img = ids >= V;  slot = clamp(ids - V, 0, NMAX-1)
#               embed  = is_img ? image_embeds[slot] : embed_tokens[clamp(ids, 0, V-1)]
#               p      = position_ids[offset];  p_text = p - (p >= start ? amount : 0)
#               s0     = p - slot;  (row, col) = image_rc[slot]
#               (t, h, w) = is_img ? (s0, s0 + row, s0 + col) : (p_text, p_text, p_text)
#               freq j rotates by h if j%3==1 and j<3*sec[1], by w if j%3==2 and j<3*sec[2],
#               else by t (HF apply_interleaved_mrope) — three 0/1 masks, `_mrope_freq_masks`
#
# Host values for one image of merged grid H x W (N = H*W <= NMAX) whose <|vision_start|> sits
# at index i0 (`host_static_inputs`): the N <|image_pad|> ids become V + k (k = 0..N-1,
# row-major), image_rc[k] = (k // W, k % W), start = i0 + 1 + N, amount = N - max(H, W). This
# reproduces HF get_rope_index: the image occupies rope positions i0+1 .. i0+max(H,W), and text
# after it resumes at i0 + 1 + max(H, W). The row/col table (instead of a baked grid width)
# lets one decoder take any grid up to NMAX tokens: square 8x8 / 14x14 and native rectangles.
#
# Text-only row: image_embeds 0, image_rc 0, start = 1 << 30, amount = 0. Then t == h == w,
# the three masked rotations collapse to the text decoder's plain partial RoPE, and the graph
# is the overlay's text decode graph (`Qwen3_5StatefulForCausalLM`) bit for bit.
#
# Export: set `use_loopfree_step = True` on every linear-attention layer before tracing or
# quantizing (the GatedDeltaUpdate while_loop does not lower on the device delegates), and
# drop gated_delta_update from the externalize specs. No DeepStack (Qwen3.5 vision has none).
#
# Prefill chunk (`build_export_spec(query_len=S)`, the "prefill" function of a multifunction
# asset next to the S=1 "main"): input_ids [1, S], the same static image inputs, logits of the
# LAST position only ([1, 1, V], `last_token_only`), GDN on `use_loopfree_unroll` (S single
# steps unrolled in-graph, fp32 inside the chunk, no doubling inverse). Everything above is
# per token, so S tokens go through the same formulas; a chunk may end mid-image or mid-text.
from __future__ import annotations

import torch

from coreai_models.models.macos.qwen3_5 import (
    DECODE_STATE_NAMES,
    QWEN3_5_TEXT_PREFIX,
    Qwen3_5StatefulForCausalLM,
    _mrope_freq_masks,
    build_decode_state,
)
from coreai_models.primitives.macos.cache import KVCache, SSMState

N_IMAGE_MAX = 256
TEXT_ONLY_SHIFT_START = 1 << 30
INPUT_NAMES = (
    "input_ids", "position_ids", "image_embeds", "image_rc",
    "rope_shift_start", "rope_shift_amount",
)


class Qwen3_5VLPipelinedForCausalLM(Qwen3_5StatefulForCausalLM):
    """Qwen3.5 hybrid VL decoder, ids-input + static image inputs; contract in the header."""

    def _init_model(self, config) -> None:
        super()._init_model(config)
        # Rows of the static image buffer. The module is shape-generic; the export spec
        # and the slot clamp read this, so override it before exporting a different size.
        self.n_image_max = N_IMAGE_MAX

    def forward(
        self,
        input_ids: torch.Tensor,          # [1, s] int32; image tokens = V + slot
        position_ids: torch.Tensor,       # [1, seq] int32 cache ramp
        image_embeds: torch.Tensor,       # [NMAX, h]
        image_rc: torch.Tensor,           # [NMAX, 2] int32 (row, col) per slot
        rope_shift_start: torch.Tensor,   # [1] int32
        rope_shift_amount: torch.Tensor,  # [1] int32
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        conv_state: torch.Tensor,
        rec_state: torch.Tensor,
    ) -> torch.Tensor:
        m = self.model
        V = self.config.vocab_size
        n_max = self.n_image_max
        b, s = input_ids.shape

        seq_len = position_ids.shape[-1]
        torch._check_is_size(s)
        torch._check_is_size(seq_len)
        offset = seq_len - s
        torch._check_is_size(offset)
        p = position_ids.narrow(-1, offset, s)  # [1, s] int32

        ids = input_ids
        is_img = ids >= V
        slot = (ids - V).clamp(0, n_max - 1)
        flat_slot = slot.reshape(-1)

        # token embedding: text table row or image slot row
        e_txt = m.embed_tokens(ids.clamp(0, V - 1))
        e_img = image_embeds.index_select(0, flat_slot).reshape(b, s, -1)
        x = torch.where(is_img.unsqueeze(-1), e_img.to(e_txt.dtype), e_txt)

        pos_t, pos_h, pos_w = self._rope_planes(
            is_img, slot, p, image_rc, rope_shift_start, rope_shift_amount)

        # interleaved M-RoPE cos/sin, assembled as Qwen3_5VLStatefulEmbeds does
        inv = m.inv_freq  # [rotary_dim/2] fp32
        masks = _mrope_freq_masks(inv.shape[0], self.config.mrope_section).to(inv.dtype)
        freqs = (
            pos_t[..., None].float() * inv * masks[0]
            + pos_h[..., None].float() * inv * masks[1]
            + pos_w[..., None].float() * inv * masks[2]
        )  # [b, s, rotary_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)

        kv = KVCache(k_cache, v_cache)
        conv = SSMState(conv_state)
        rec = SSMState(rec_state)
        h = m.forward_stateful_core(x, position_ids, kv, conv, rec, cos_sin=(emb.cos(), emb.sin()))
        if self.last_token_only:
            h = h[:, -1:, :]
        return self.lm_head(h)

    def _rope_planes(self, is_img, slot, p, image_rc, rope_shift_start, rope_shift_amount):
        """(t, h, w) rope positions per token, [b, s] int32 each, from (ids, position) alone:
        image tokens self-locate from their slot and the static (row, col) table; text tokens
        take the cache position minus the post-image shift."""
        b, s = slot.shape
        rc = image_rc.index_select(0, slot.reshape(-1)).reshape(b, s, 2)
        row, col = rc[..., 0], rc[..., 1]
        shift = torch.where(p >= rope_shift_start, rope_shift_amount, torch.zeros_like(p))
        p_text = p - shift
        s0 = p - slot  # image start position (valid where is_img)
        pos_t = torch.where(is_img, s0, p_text)
        pos_h = torch.where(is_img, s0 + row, p_text)
        pos_w = torch.where(is_img, s0 + col, p_text)
        return pos_t, pos_h, pos_w

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_hf(
        cls,
        hf_id: str,
        target_dtype: torch.dtype = torch.float16,
        max_context_length: int | None = 4096,
        n_image_max: int = N_IMAGE_MAX,
    ) -> "Qwen3_5VLPipelinedForCausalLM":
        """Load the text decoder of a Qwen3.5 VL checkpoint (``model.language_model.*``; the
        tower and MTP weights are skipped). ``load_report`` records every checkpoint text key
        against the module's tensors — the overlay loader itself is strict=False."""
        import glob
        import os

        from huggingface_hub import snapshot_download
        from safetensors import safe_open

        model = cls.from_hf_memory_efficient(
            hf_id, max_context_length=max_context_length, target_dtype=target_dtype,
            hf_config_attr="text_config")
        model.n_image_max = n_image_max
        model.eval()

        model_dir = snapshot_download(
            hf_id, allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"])
        ckpt, other = set(), 0
        for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    if key.startswith(QWEN3_5_TEXT_PREFIX):
                        ckpt.add("model." + key[len(QWEN3_5_TEXT_PREFIX):])
                    elif key == "lm_head.weight" and not model.config.tie_word_embeddings:
                        ckpt.add(key)
                    else:
                        other += 1
        own = set(model.state_dict(keep_vars=True).keys())
        tied = {"lm_head.weight"} if model.config.tie_word_embeddings else set()
        model.load_report = {
            "checkpoint_text_keys": len(ckpt),
            "checkpoint_other_keys_skipped": other,
            "module_tensors": len(own),
            "unread_checkpoint_keys": sorted(ckpt - own),
            "module_tensors_not_in_checkpoint": sorted(own - ckpt - tied),
            "tied_lm_head": bool(model.config.tie_word_embeddings
                                 and model.lm_head.weight is model.model.embed_tokens.weight),
        }
        return model

    # -- export -------------------------------------------------------------

    def build_export_spec(
        self,
        target_dtype: torch.dtype,
        max_context_length: int,
        trace_kv_len: int,
        trace_past: int = 64,
        query_len: int = 1,
    ) -> dict:
        """Static-S entrypoint: static ``input_ids`` [1, S] and static image inputs;
        ``position_ids`` and the KV sequence dim stay dynamic (the decode-only Qwen3.5
        export's dims), the conv/rec states are fixed-shape. S = 1 is the decode graph
        ("main"); S > 1 is a prefill chunk ("prefill" next to "main" in one multifunction
        asset) — trace it with ``last_token_only`` set for a [1, 1, V] output and every GDN
        layer on ``use_loopfree_unroll`` (the S single steps unrolled in-graph)."""
        cfg = self.config
        n_max, hidden = self.n_image_max, cfg.hidden_size
        S = query_len
        state = build_decode_state(cfg, max_seq_len=trace_kv_len, dtype=target_dtype)
        reference_inputs = {
            "input_ids": torch.randint(1, cfg.vocab_size, (1, S), dtype=torch.int32),
            "position_ids": torch.arange(trace_past + S, dtype=torch.int32).unsqueeze(0),
            "image_embeds": torch.zeros(n_max, hidden, dtype=target_dtype),
            "image_rc": torch.zeros(n_max, 2, dtype=torch.int32),
            "rope_shift_start": torch.tensor([TEXT_ONLY_SHIFT_START], dtype=torch.int32),
            "rope_shift_amount": torch.tensor([0], dtype=torch.int32),
            "k_cache": state["k_cache"],
            "v_cache": state["v_cache"],
            "conv_state": state["conv_state"],
            "rec_state": state["rec_state"],
        }
        # The first chunk of a row runs at seq_len == S (offset 0), so the dim admits it.
        seq_pos = torch.export.Dim("seq_pos", min=max(2, S), max=max_context_length - 1)
        k_seq = torch.export.Dim("k_seq", min=trace_kv_len, max=max_context_length)
        v_seq = torch.export.Dim("v_seq", min=trace_kv_len, max=max_context_length)
        dynamic_shapes = {
            "input_ids": None,  # static [1, S]: no scan, no while_loop
            "position_ids": {1: seq_pos},
            "image_embeds": None,
            "image_rc": None,
            "rope_shift_start": None,
            "rope_shift_amount": None,
            "k_cache": {KVCache.seq_len_dim(): k_seq},
            "v_cache": {KVCache.seq_len_dim(): v_seq},
            "conv_state": None,
            "rec_state": None,
        }
        return {
            "reference_inputs": reference_inputs,
            "dynamic_shapes": dynamic_shapes,
            "input_names": INPUT_NAMES,
            "output_names": ("logits",),
            "state_names": DECODE_STATE_NAMES,
        }


def host_static_inputs(
    ids,
    grid_hw: tuple[int, int] | None,
    vocab_size: int,
    image_pad_id: int,
    vision_start_id: int,
    n_image_max: int = N_IMAGE_MAX,
):
    """Host side of the contract for one prompt with at most one image.

    ``ids``: the processor's token ids (``<|image_pad|>`` repeated N times); ``grid_hw``: the
    MERGED grid (H, W) = (grid_thw[1] // merge, grid_thw[2] // merge), or None for text.
    Returns (mapped ids, image_rc [NMAX, 2] int32, start [1] int32, amount [1] int32) as
    int32 torch tensors; the caller places the tower rows at image_embeds[0:N].
    """
    ids = torch.as_tensor(ids, dtype=torch.int64).reshape(-1)
    image_rc = torch.zeros(n_image_max, 2, dtype=torch.int32)
    pads = (ids == image_pad_id).nonzero().reshape(-1)
    if grid_hw is None:
        if pads.numel():
            raise ValueError("image tokens in a text-only prompt")
        return (ids.to(torch.int32), image_rc,
                torch.tensor([TEXT_ONLY_SHIFT_START], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32))
    H, W = (int(v) for v in grid_hw)
    N = H * W
    if N > n_image_max:
        raise ValueError(f"{H}x{W} = {N} image tokens > n_image_max {n_image_max}")
    starts = (ids == vision_start_id).nonzero().reshape(-1)
    if starts.numel() != 1 or pads.numel() != N:
        raise ValueError(f"expected one image block of {N} tokens, got {starts.numel()} "
                         f"<|vision_start|> and {pads.numel()} <|image_pad|>")
    i0 = int(starts[0])
    if not torch.equal(pads, torch.arange(i0 + 1, i0 + 1 + N)):
        raise ValueError("image tokens are not one contiguous block after <|vision_start|>")
    k = torch.arange(N)
    mapped = ids.clone()
    mapped[i0 + 1:i0 + 1 + N] = vocab_size + k
    image_rc[:N, 0] = (k // W).to(torch.int32)
    image_rc[:N, 1] = (k % W).to(torch.int32)
    return (mapped.to(torch.int32), image_rc,
            torch.tensor([i0 + 1 + N], dtype=torch.int32),
            torch.tensor([N - max(H, W)], dtype=torch.int32))
