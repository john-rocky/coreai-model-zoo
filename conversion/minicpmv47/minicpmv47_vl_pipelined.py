# MiniCPM-V-4.7-1B VL decoder on the ids-input + static `image_embeds` contract, canvas M-RoPE.
#
# Community port — NOT an Apple model.
#
# The text backbone is the overlay's Qwen3.5 hybrid (`Qwen3_5StatefulForCausalLM`: GatedDeltaNet
# + gated full attention, 24 layers [lin, lin, lin, full] x6). What 4.7 adds over 4.6 is the
# position scheme: transformers' `MiniCPMV4_7Model.get_rope_index` ("canvas" M-RoPE) gives every
# token three rope positions (T, H, W) that the interleaved M-RoPE [11, 11, 10] rotates by. This
# module derives those planes in-graph from (ids, cache position) and two static scalars, so the
# graph takes the same inputs as the zoo's other VL decoders (`decider_vision/qwen3_5_vl_pipelined.py`
# is the closest) and the kit / PipelinedBench bind it by name. The head is the checkpoint's own
# `lm_head.weight` (4.7 is untied).
#
# Shipped image form: thumbnail only, one 448x448 image -> 32x32 patches -> 8x8 = 64 LLM tokens.
# The grid is baked (GRID = 8): no row/col input, the slot -> (row, col) table is a constant.
#
# Graph (S = 1 decode; S > 1 a prefill chunk with `last_token_only`; states mutate in place):
#
#   inputs  input_ids          [1, S]       int32  static; image token k (row-major) = V + k
#           position_ids       [1, seq]     int32  dynamic; the cache ramp 0..seq-1 (offset = seq - S)
#           image_embeds       [64, h]      float  static; tower rows 0..63 (zero for a text-only row)
#           rope_shift_start   [1]          int32  static; index of the first token after the frame
#           rope_shift_amount  [1]          int32  static; start - (O + GRID + 1)
#   states  keyCache/valueCache [6, 1, 2, ctx, 256] (ctx dynamic), convState [18, 1, 6144, 3],
#           recState [18, 1, 16, 128, 128]
#   output  logits             [1, S or 1, V]
#
#   per token (p = position_ids[offset + i], V = 248094, G = GRID = 8):
#     embed  = ids >= V ? image_embeds[ids - V] : embed_tokens[ids]
#     O      = start - amount - (G + 1)                      canvas origin = index of <image>
#     frame  = O <= p < start                                <image>, 64 rows, </image>, trailing "\n"
#     p_text = p - (p >= start ? amount : 0)
#     (t, h, w) = frame ? ( O, hw, hw' ) : (p_text, p_text, p_text), with in the frame
#         <image>  (id 248078)  h = w = O - 1                halo outside the top-left corner
#         row k    (id V + k)   h = O + k // G, w = O + k % G
#         </image> (id 248079)  h = w = O + G                far corner
#         anything else         h = w = O                    base coat ("\n" after </image>)
#   then freq j rotates by h if j%3==1 and j<3*sec[1], by w if j%3==2 and j<3*sec[2], else by t
#   (HF interleaved M-RoPE; `_mrope_freq_masks`).
#
# This is transformers 5.19.0 `get_rope_index` for one thumbnail-only image placed after plain
# text (canvas = the 8x8 thumbnail grid, linspace degenerates to arange, text after the frame
# resumes at O + max(H, W) + 1); round 1 checked it against the planes the HF rotary received on
# every prompt and generate step (`parity_vl_torch.py`). Generation continues at p - amount on all
# three planes, which is HF's `rope_deltas = -amount`.
#
# Host values (`host_static_inputs`): the processor's ids with the 64 <|image_pad|> rewritten to
# V + k, start = the first index after </image> and the marker tokens that follow it (newline 198,
# <image>, </image>, <slice>, </slice> — HF's frame end), amount = start - (i0 + 9) with i0 the
# index of <image>. Text-only row: start = 1 << 30, amount = 0 -> (p, p, p) everywhere, the
# three masked rotations collapse to the plain partial RoPE and the graph is the text decoder.
#
# Export (round 3): `use_loopfree_step = True` on every linear-attention layer for S = 1 (the
# GatedDeltaUpdate while_loop does not lower on the device delegates), `use_loopfree_unroll` for
# a static-S prefill chunk, gated_delta_update dropped from the externalize specs.
from __future__ import annotations

import torch

from coreai_models.models.macos.qwen3_5 import (
    DECODE_STATE_NAMES,
    QWEN3_5_TEXT_PREFIX,
    Qwen3_5Config,
    Qwen3_5StatefulForCausalLM,
    _mrope_freq_masks,
    build_decode_state,
)
from coreai_models.primitives.macos.cache import KVCache, SSMState

VOCAB = 248094
GRID = 8                                  # merged thumbnail grid side (448 px / 14 / 4)
N_IMAGE = GRID * GRID                     # 64 image rows
IMAGE_PAD, IMAGE_START, IMAGE_END = 248056, 248078, 248079
SLICE_START, SLICE_END, NEWLINE = 248088, 248089, 198
FRAME_MARKERS = (SLICE_START, SLICE_END, IMAGE_START, IMAGE_END, NEWLINE)   # HF `_frame_end_idx`
TEXT_ONLY_SHIFT_START = 1 << 30
INPUT_NAMES = ("input_ids", "position_ids", "image_embeds", "rope_shift_start", "rope_shift_amount")


def minicpmv47_text_config() -> Qwen3_5Config:
    """MiniCPM-V-4.7-1B `text_config` (config.json), written out: the overlay venv's transformers
    has no `minicpmv4_7`, so AutoConfig cannot build it."""
    lt = (["linear_attention"] * 3 + ["full_attention"]) * 6
    return Qwen3_5Config(
        hidden_size=1024, num_hidden_layers=24, vocab_size=VOCAB, intermediate_size=3584,
        rms_norm_eps=1e-6, tie_word_embeddings=False, head_dim=256,
        num_attention_heads=8, num_key_value_heads=2, attn_output_gate=True,
        partial_rotary_factor=0.25, rope_theta=1e7,
        linear_num_key_heads=16, linear_num_value_heads=16,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
        full_attention_interval=4, layer_types=lt, mrope_section=(11, 11, 10),
    )


def _slot_rc_table() -> torch.Tensor:
    k = torch.arange(N_IMAGE, dtype=torch.int32)
    return torch.stack([k // GRID, k % GRID], dim=-1)          # [64, 2] (row, col)


class MiniCPMV47VLPipelinedForCausalLM(Qwen3_5StatefulForCausalLM):
    """MiniCPM-V-4.7 VL decoder, ids-input + 3 static inputs, canvas M-RoPE; contract in the header."""

    def _init_model(self, config) -> None:
        super()._init_model(config)
        self.n_image = N_IMAGE
        self.grid = GRID
        self.register_buffer("slot_rc", _slot_rc_table(), persistent=False)

    def reset_buffers(self) -> None:
        """Recreate the non-persistent buffers after meta-device construction + assign-load."""
        self.model.reset_buffers()
        self.slot_rc = _slot_rc_table()

    def forward(
        self,
        input_ids: torch.Tensor,          # [1, s] int32; image rows = V + slot
        position_ids: torch.Tensor,       # [1, seq] int32 cache ramp
        image_embeds: torch.Tensor,       # [64, h]
        rope_shift_start: torch.Tensor,   # [1] int32
        rope_shift_amount: torch.Tensor,  # [1] int32
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        conv_state: torch.Tensor,
        rec_state: torch.Tensor,
    ) -> torch.Tensor:
        m = self.model
        V = self.config.vocab_size
        b, s = input_ids.shape

        seq_len = position_ids.shape[-1]
        torch._check_is_size(s)
        torch._check_is_size(seq_len)
        offset = seq_len - s
        torch._check_is_size(offset)
        p = position_ids.narrow(-1, offset, s)  # [1, s] int32

        ids = input_ids
        is_img = ids >= V
        slot = (ids - V).clamp(0, self.n_image - 1)

        e_txt = m.embed_tokens(ids.clamp(0, V - 1))
        e_img = image_embeds.index_select(0, slot.reshape(-1)).reshape(b, s, -1)
        x = torch.where(is_img.unsqueeze(-1), e_img.to(e_txt.dtype), e_txt)

        pos_t, pos_h, pos_w = self._rope_planes(ids, is_img, slot, p, rope_shift_start, rope_shift_amount)

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

    def _rope_planes(self, ids, is_img, slot, p, rope_shift_start, rope_shift_amount):
        """(t, h, w) rope positions per token, [b, s] int32 each, from (ids, position, start, amount):
        the canvas rule of the header. Text-only rows (start = 1 << 30) never enter the frame."""
        b, s = slot.shape
        G = self.grid
        rc = self.slot_rc.index_select(0, slot.reshape(-1)).reshape(b, s, 2)
        row, col = rc[..., 0], rc[..., 1]
        origin = rope_shift_start - rope_shift_amount - (G + 1)          # [1]
        shift = torch.where(p >= rope_shift_start, rope_shift_amount, torch.zeros_like(p))
        p_text = p - shift
        in_frame = (p >= origin) & (p < rope_shift_start)
        is_open = ids == IMAGE_START
        is_close = ids == IMAGE_END
        o = origin.expand_as(p)
        corner = torch.where(is_open, o - 1, torch.where(is_close, o + G, o))
        f_h = torch.where(is_img, o + row, corner)
        f_w = torch.where(is_img, o + col, corner)
        pos_t = torch.where(in_frame, o, p_text)
        pos_h = torch.where(in_frame, f_h, p_text)
        pos_w = torch.where(in_frame, f_w, p_text)
        return pos_t, pos_h, pos_w

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, snapshot_dir, target_dtype: torch.dtype = torch.float32):
        """Text decoder + untied head from the pinned MiniCPM-V-4.7 snapshot: ``model.language_model.*``
        -> ``model.*``, ``lm_head.weight`` as is; the tower and merger are skipped. ``load_report``
        records every checkpoint text key against the module's tensors."""
        from pathlib import Path

        from safetensors import safe_open

        model = cls(minicpmv47_text_config(), model_device="meta")
        sd, other = {}, 0
        with safe_open(str(Path(snapshot_dir) / "model.safetensors"), framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                if key.startswith(QWEN3_5_TEXT_PREFIX):
                    sd["model." + key[len(QWEN3_5_TEXT_PREFIX):]] = f.get_tensor(key).to(target_dtype)
                elif key == "lm_head.weight":
                    sd[key] = f.get_tensor(key).to(target_dtype)
                else:
                    other += 1
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        model.reset_buffers()
        model.eval()
        on_meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
        model.load_report = {
            "checkpoint_text_keys": len(sd), "checkpoint_other_keys_skipped": other,
            "module_parameters": len(dict(model.named_parameters())),
            "missing": list(missing), "unexpected": list(unexpected), "on_meta": on_meta,
            "lm_head_is_embed": bool(model.lm_head.weight is model.model.embed_tokens.weight),
        }
        if missing or unexpected or on_meta:
            raise RuntimeError(f"load: {model.load_report}")
        return model

    # -- export (round 3) -----------------------------------------------------

    def build_export_spec(
        self,
        target_dtype: torch.dtype,
        max_context_length: int,
        trace_kv_len: int,
        trace_past: int = 64,
        query_len: int = 1,
    ) -> dict:
        """Static-S entrypoint, as `qwen3_5_vl_pipelined.build_export_spec` minus `image_rc`:
        static ``input_ids`` [1, S] and static image inputs; ``position_ids`` and the KV
        sequence dim dynamic, the conv/rec states fixed-shape."""
        cfg = self.config
        S = query_len
        state = build_decode_state(cfg, max_seq_len=trace_kv_len, dtype=target_dtype)
        reference_inputs = {
            "input_ids": torch.randint(1, cfg.vocab_size, (1, S), dtype=torch.int32),
            "position_ids": torch.arange(trace_past + S, dtype=torch.int32).unsqueeze(0),
            "image_embeds": torch.zeros(self.n_image, cfg.hidden_size, dtype=target_dtype),
            "rope_shift_start": torch.tensor([TEXT_ONLY_SHIFT_START], dtype=torch.int32),
            "rope_shift_amount": torch.tensor([0], dtype=torch.int32),
            "k_cache": state["k_cache"],
            "v_cache": state["v_cache"],
            "conv_state": state["conv_state"],
            "rec_state": state["rec_state"],
        }
        seq_pos = torch.export.Dim("seq_pos", min=max(2, S), max=max_context_length - 1)
        k_seq = torch.export.Dim("k_seq", min=trace_kv_len, max=max_context_length)
        v_seq = torch.export.Dim("v_seq", min=trace_kv_len, max=max_context_length)
        dynamic_shapes = {
            "input_ids": None,
            "position_ids": {1: seq_pos},
            "image_embeds": None,
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


def host_static_inputs(ids, vocab_size: int = VOCAB):
    """Host side of the contract for one prompt with at most one thumbnail-only image.

    ``ids``: the processor's token ids (``<|image_pad|>`` x64 inside ``<image>…</image>``).
    Returns (mapped ids [n] int32, rope_shift_start [1] int32, rope_shift_amount [1] int32);
    the caller puts the tower's 64 rows in ``image_embeds``.
    """
    ids = torch.as_tensor(ids, dtype=torch.int64).reshape(-1)
    opens = (ids == IMAGE_START).nonzero().reshape(-1)
    pads = (ids == IMAGE_PAD).nonzero().reshape(-1)
    if opens.numel() == 0 and pads.numel() == 0:
        return (ids.to(torch.int32), torch.tensor([TEXT_ONLY_SHIFT_START], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32))
    if opens.numel() != 1 or pads.numel() != N_IMAGE:
        raise ValueError(f"expected one thumbnail of {N_IMAGE} tokens, got {opens.numel()} <image> "
                         f"and {pads.numel()} <|image_pad|>")
    if ((ids == SLICE_START) | (ids == SLICE_END)).any():
        raise ValueError("slices are not in this contract (thumbnail only, max_slice_nums=1)")
    i0 = int(opens[0])
    if not torch.equal(pads, torch.arange(i0 + 1, i0 + 1 + N_IMAGE)) or int(ids[i0 + 1 + N_IMAGE]) != IMAGE_END:
        raise ValueError("the 64 image tokens must sit between <image> and </image>")
    end = i0 + 2 + N_IMAGE
    while end < ids.numel() and int(ids[end]) in FRAME_MARKERS:
        end += 1
    mapped = ids.clone()
    mapped[i0 + 1:i0 + 1 + N_IMAGE] = vocab_size + torch.arange(N_IMAGE)
    return (mapped.to(torch.int32), torch.tensor([end], dtype=torch.int32),
            torch.tensor([end - (i0 + GRID + 1)], dtype=torch.int32))
