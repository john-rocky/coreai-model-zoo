"""Static Core AI form of the WeVisOne-VL-2B mask decoder: two graphs that reproduce the package's
`WeVisOneMaskDecoder._forward_fp32` (Tencent/WeVisOne-VL 40f735b8, `wevisone_vl/mask_decoder.py`, `msda.py`).

graph A — once per response, `n_max` query rows (the response's `<OBJ>` rows first, then padding):
    image [1,3,1024,1024], tap5 / tap11 / tap17 / tap23 [1,1024,G,G] (row-major pre-merge ViT maps),
    t_seg [n_max,2048] (last decoder layer output at each `<OBJ>`), boxes01 [n_max,4] (xyxy in 0..1),
    valid [n_max] (1 = a real `<OBJ>`, 0 = padding)
 -> queries [n_max,192], coarse_logits [n_max,512,512], semantic_s2 [1,48,512,512], memory [1,4G^2,192]
graph B — once per instance:
    image [1,3,1024,1024], semantic_s2 [1,48,512,512], query [1,192], coarse_logits [1,512,512]
 -> mask_logits [1,1024,1024], mask_logits_aux [1,512,512]
The host turns mask_logits into the alpha exactly as the package does (sigmoid, bilinear align_corners=False to the
original size).

The weights are the package's own modules, loaded from the checkpoint's `mask_decoder.pt` (strict). What differs
from the package's forward, and why:
  * the levels' (h, w) are Python ints: the package builds them as a tensor and reads them back with `.tolist()` /
    `.item()`, which torch.export turns into data-dependent symbols (PROBE_MASK.md (i) pass 1);
  * `F.grid_sample` -> the gather-based bilinear sampler of `conversion/export_rf_detr.py` (`aten.grid_sampler_2d`
    has no Core AI lowering, pass 2). The four levels share one G x G grid, so one sampler call per decoder layer
    takes (heads x levels) as its batch instead of four calls;
  * the decoder self-attention couples the queries of one response (`nn.MultiheadAttention` over the `<OBJ>` rows),
    so padding is removed from its keys with `where(keep, scores, fill)` and from its values with
    `where(keep, v, 0)`; every other stage is per query;
  * the sine position embedding plus the level embedding is a constant buffer (it depends on G only);
  * the encoder's depthwise / pointwise convolutions run the four levels as one batch of four (same weights);
  * the alpha cascade (`FullCanvasAlphaCascadeHead`, independent per query) is cut after
    `semantic_s2 = self.semantic(pixel_feature)`: graph A computes it once, graph B runs the S/4 -> S/2 -> S/1
    stages for one query.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LLM_DIM = 2048
VIT_DIM = 1024
N_MAX = 16          # the package's decoder_batch_size: queries in different chunks of 16 never attend to each other
CANVAS = 1024


def import_package(pkg_dir: str | Path):
    """The reference package by path, appended to sys.path (never in front of it)."""
    p = str(Path(pkg_dir).expanduser().resolve())
    if p not in sys.path:
        sys.path.append(p)
    from wevisone_vl import mask_decoder as MD
    from wevisone_vl import msda as MSDA
    from wevisone_vl.config import Qwen3VLSegConfig

    return MD, MSDA, Qwen3VLSegConfig


def load_package_decoder(pkg_dir: str | Path, sidecar: str | Path):
    MD, _, Cfg = import_package(pkg_dir)
    side = torch.load(str(sidecar), map_location="cpu", weights_only=True)
    cfg = Cfg.from_dict(side.get("cfg"))
    dec = MD.WeVisOneMaskDecoder(visual_in_dims=[VIT_DIM] * len(cfg.vision_feature_layers), llm_dim=LLM_DIM, cfg=cfg)
    res = dec.load_state_dict(side["state_dict"], strict=True)
    if res.missing_keys or res.unexpected_keys:
        raise RuntimeError(f"mask decoder state mismatch: {res}")
    return dec.float().eval(), cfg, side


def gather_bilinear(input: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """F.grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False) without
    `aten.grid_sampler_2d`: bool-free, floor-safe on every compute unit, gather-based (from
    conversion/export_rf_detr.py `manual_bilinear_grid_sample`)."""
    batch_size, channels, height, width = input.shape
    grid_height, grid_width = grid.shape[1], grid.shape[2]
    ix = (grid[..., 0] + 1) * width / 2 - 0.5
    iy = (grid[..., 1] + 1) * height / 2 - 0.5

    def _floor(t):
        # aten.floor lowers to identity on the GPU delegate; floor-division by 2 of 2t survives every unit
        return torch.div(t * 2.0, 2.0, rounding_mode="floor")

    ix0f, iy0f = _floor(ix), _floor(iy)
    ix1f, iy1f = ix0f + 1.0, iy0f + 1.0
    wx1 = (ix - ix0f).to(input.dtype).unsqueeze(1)
    wy1 = (iy - iy0f).to(input.dtype).unsqueeze(1)
    wx0, wy0 = 1.0 - wx1, 1.0 - wy1
    ix0c, iy0c = ix0f.clamp(0, width - 1), iy0f.clamp(0, height - 1)
    ix1c, iy1c = ix1f.clamp(0, width - 1), iy1f.clamp(0, height - 1)
    # integer-valued floats: exact 0 / 1 masks for the zeros padding
    in_x0 = 1.0 - (ix0f - ix0c).abs().clamp(max=1.0)
    in_x1 = 1.0 - (ix1f - ix1c).abs().clamp(max=1.0)
    in_y0 = 1.0 - (iy0f - iy0c).abs().clamp(max=1.0)
    in_y1 = 1.0 - (iy1f - iy1c).abs().clamp(max=1.0)
    flat = input.flatten(2)

    def _gather(iyc, ixc):
        # index math in fp32 (fp16 integers stop being exact at 2048)
        idx = (iyc.float() * width + ixc.float()).long()
        idx = idx.flatten(1).unsqueeze(1).expand(batch_size, channels, -1)
        return flat.gather(2, idx).view(batch_size, channels, grid_height, grid_width)

    v00 = _gather(iy0c, ix0c) * (in_x0 * in_y0).to(input.dtype).unsqueeze(1)
    v10 = _gather(iy0c, ix1c) * (in_x1 * in_y0).to(input.dtype).unsqueeze(1)
    v01 = _gather(iy1c, ix0c) * (in_x0 * in_y1).to(input.dtype).unsqueeze(1)
    v11 = _gather(iy1c, ix1c) * (in_x1 * in_y1).to(input.dtype).unsqueeze(1)
    return wx0 * wy0 * v00 + wx1 * wy0 * v10 + wx0 * wy1 * v01 + wx1 * wy1 * v11


def grid_sample_reference(input: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """The package's own sampler call (for the torch-side A/B of the substitute; not exportable)."""
    return F.grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def position_constant(encoder, g: int) -> torch.Tensor:
    """`pos` of `SameScaleMSDAEncoder.forward` for a G x G grid: the sine embedding of each level plus its level
    embedding, flattened and concatenated in level order -> [1, levels*G*G, hidden]."""
    d = encoder.level_embed.shape[-1]
    feature = torch.zeros(1, d, g, g)
    parts = []
    with torch.no_grad():
        for level in range(encoder.level_embed.shape[0]):
            pos = encoder.position(feature) + encoder.level_embed[level][None, :, None, None]
            parts.append(pos.flatten(2).transpose(1, 2))
    return torch.cat(parts, dim=1).float().contiguous()


def boxes_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    boxes = boxes.clamp(0.0, 1.0)
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1).clamp_min(1e-4), (y2 - y1).clamp_min(1e-4)],
                       dim=-1)


class GraphA(nn.Module):
    """Visual encoder, query builder, 6-layer DINO decoder (key-padded self-attention + static MSDA), pixel fusion,
    dynamic mask predictor and the cascade's shared semantic projection."""

    def __init__(self, dec, g: int, n_max: int = N_MAX, sampler=gather_bilinear, fill: float = -1e30,
                 mask_padding: bool = True):
        super().__init__()
        enc = dec.visual_encoder
        self.input_proj = enc.input_proj
        self.enc_layers = enc.encoder.layers
        self.query_builder = dec.query_builder
        self.dec_layers = dec.transformer.layers
        self.pixel_fusion = dec.pixel_fusion
        self.mask_predictor = dec.mask_predictor
        self.semantic = dec.highres_refine.semantic
        self.register_buffer("pos_const", position_constant(enc, g), persistent=False)
        self.g, self.n_max, self.levels = int(g), int(n_max), int(enc.level_embed.shape[0])
        self.sampler, self.fill, self.mask_padding = sampler, float(fill), bool(mask_padding)
        self.keep_scores = False    # torch-side checks only (the score absmax); off for export
        self.last_scores: list = []

    def encode(self, feats):
        g, lv = self.g, self.levels
        projected = [proj(f) for proj, f in zip(self.input_proj, feats)]
        d = projected[0].shape[1]
        src = torch.cat([p.flatten(2).transpose(1, 2) for p in projected], dim=1)
        pos = self.pos_const.to(src.dtype)
        for layer in self.enc_layers:
            x = (src + pos).reshape(lv, g * g, d).transpose(1, 2).reshape(lv, d, g, g)
            x = layer.pointwise(F.gelu(layer.depthwise(x)))
            mixed = x.reshape(lv, d, g * g).transpose(1, 2).reshape(1, lv * g * g, d)
            src = layer.norm1(src + layer.dropout(mixed))
            ffn = layer.linear2(F.gelu(layer.linear1(src)))
            src = layer.norm2(src + layer.dropout(ffn))
        maps = [src[:, i * g * g:(i + 1) * g * g].transpose(1, 2).reshape(1, d, g, g) for i in range(lv)]
        return src, maps

    def self_attention(self, mha, qk, value, keep_k, keep_v):
        d, h = mha.embed_dim, mha.num_heads
        hd = d // h
        n = qk.shape[1]
        w_q, w_k, w_v = mha.in_proj_weight.chunk(3)
        b_q, b_k, b_v = mha.in_proj_bias.chunk(3)
        q = F.linear(qk, w_q, b_q).reshape(1, n, h, hd).transpose(1, 2)
        k = F.linear(qk, w_k, b_k).reshape(1, n, h, hd).transpose(1, 2)
        v = F.linear(value, w_v, b_v).reshape(1, n, h, hd).transpose(1, 2)
        s = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(hd))
        if self.keep_scores:
            # the scores the fill must stay below: real query x real key only
            real = keep_v & keep_k      # [1,1,n,1] rows (queries) x [1,1,1,n] columns (keys)
            self.last_scores.append(torch.where(real, s, torch.zeros_like(s)))
        if self.mask_padding:
            v = torch.where(keep_v, v, torch.zeros_like(v))
            s = torch.where(keep_k, s, torch.full_like(s, self.fill))
        p = torch.softmax(s, dim=-1)
        o = torch.matmul(p, v).transpose(1, 2).reshape(1, n, d)
        return mha.out_proj(o)

    def msda(self, m, query, ref, memory):
        g, n = self.g, query.shape[1]
        H, L, P = m.num_heads, m.num_levels, m.num_points
        hd = m.embed_dim // H
        value = m.value_proj(memory)
        off = m.sampling_offsets(query).reshape(1, n, H, L, P, 2)
        w = m.attention_weights(query).reshape(1, n, H, L * P).softmax(-1).reshape(1, n, H, L, P)
        loc = ref[:, :, None, :, None, :2] + off / float(P) * ref[:, :, None, :, None, 2:] * 0.5
        v = value.reshape(1, L, g * g, H, hd).permute(0, 3, 1, 4, 2).reshape(H * L, hd, g, g)
        grid = (loc * 2.0 - 1.0).permute(0, 2, 3, 1, 4, 5).reshape(H * L, n, P, 2)
        s = self.sampler(v, grid)
        s = s.reshape(H, L, hd, n, P).permute(0, 2, 3, 1, 4).reshape(H, hd, n, L * P)
        wt = w.permute(0, 2, 1, 3, 4).reshape(H, 1, n, L * P)
        out = (s * wt).sum(-1).reshape(1, H * hd, n).transpose(1, 2)
        return m.output_proj(out)

    def forward(self, image, tap5, tap11, tap17, tap23, t_seg, boxes01, valid):
        return self._forward(image, tap5, tap11, tap17, tap23, t_seg, boxes01, valid, None)

    def _forward(self, image, tap5, tap11, tap17, tap23, t_seg, boxes01, valid, box_pe):
        if self.keep_scores:
            self.last_scores = []
        memory, maps = self.encode([tap5, tap11, tap17, tap23])
        if box_pe is None:
            content, position = self.query_builder(t_seg, boxes01)
        else:
            qb = self.query_builder
            content = qb.content_norm(qb.content_proj(t_seg))
            position = qb.position_norm(qb.box_position.mlp(box_pe))
        target, query_pos = content.unsqueeze(0), position.unsqueeze(0)
        keep = valid > 0.5
        keep_k = keep.reshape(1, 1, 1, -1)
        keep_v = keep.reshape(1, 1, -1, 1)
        ref = boxes_xyxy_to_cxcywh(boxes01)[None, :, None].expand(1, -1, self.levels, -1)
        for layer in self.dec_layers:
            qk = target + query_pos
            target = layer.norm1(target + layer.dropout1(self.self_attention(layer.self_attn, qk, target, keep_k,
                                                                             keep_v)))
            cross = self.msda(layer.cross_attn, target + query_pos, ref, memory)
            target = layer.norm2(target + layer.dropout2(cross))
            ffn = layer.linear2(F.gelu(layer.linear1(target)))
            target = layer.norm3(target + layer.dropout3(ffn))
        queries = target.squeeze(0)
        pixel_feature = self.pixel_fusion(image, maps)
        coarse = self.mask_predictor(queries, pixel_feature)
        semantic_s2 = self.semantic(pixel_feature)
        return queries, coarse, semantic_s2, memory


class GraphAHF(GraphA):
    """GraphA with the box Fourier features as an input: `box_pe [n_max, 256]` = the four `FourierPE1D` blocks of
    `BoxFourierEncoder` (x1, y1, 0.2 log w + 0.5, 0.2 log h + 0.5; sin / cos of 2 pi v 2^k, k = 0..31) made on the host
    (`box_fourier_np`). The phases run to about 1e10 rad (1.12e10 at k = 31 over the fixture's boxes), so one fp32 ulp of difference in an input moves the
    high-k features to unrelated values: inside a Core AI GPU graph (GraphA) 0.2 log w + 0.5 came out one ulp off the
    CPU's on some boxes, while the GPU's sin / cos of a given fp32 phase stayed within 1.2e-7 of float64."""

    def forward(self, image, tap5, tap11, tap17, tap23, t_seg, boxes01, box_pe, valid):
        return self._forward(image, tap5, tap11, tap17, tap23, t_seg, boxes01, valid, box_pe)


def box_fourier_np(boxes01: np.ndarray, num_freqs: int = 32) -> np.ndarray:
    """The host's `BoxFourierEncoder` features before its MLP: fp32 inputs and phase as the package computes them on
    the CPU (phase = fl32(fl32(2 pi) * v) * 2^k, exact), sin / cos in float64 of that fp32 phase, then fp32.
    `np.log` is not torch's CPU `log` everywhere: of the 3,494 fp32 widths a box with integer per-mille corners can
    have, 7 round one or two ulp apart (926 of the 500,500 corner pairs; torch 2.9 and 2.11), and with NumPy's value
    the matte of such a box fell under the IoU bar for 5 of the 7 widths. CoreAIKit's Examples/Cutout host carries
    torch's values for those 7 widths (`Sources/Engine/BoxFourier.swift`); this copy does not."""
    b = np.asarray(boxes01, np.float32)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    w = np.maximum(x2 - x1, np.float32(1e-6))
    h = np.maximum(y2 - y1, np.float32(1e-6))
    ew = np.float32(0.2) * np.log(w) + np.float32(0.5)
    eh = np.float32(0.2) * np.log(h) + np.float32(0.5)
    freqs = (2.0 ** np.arange(num_freqs)).astype(np.float32)

    def pe(v):
        phase = (np.float32(2.0 * math.pi) * v.astype(np.float32))[:, None] * freqs
        p64 = phase.astype(np.float64)
        return np.concatenate([np.sin(p64), np.cos(p64)], axis=-1).astype(np.float32)

    return np.concatenate([pe(x1), pe(y1), pe(ew), pe(eh)], axis=-1)


class GraphB(nn.Module):
    """`FullCanvasAlphaCascadeHead.forward` for one query, from the shared `semantic_s2` on."""

    def __init__(self, dec):
        super().__init__()
        h = dec.highres_refine
        for name in ("rgb_s4", "rgb_s2", "rgb_s1", "coarse_s4", "coarse_s2", "previous_s2", "previous_s1",
                     "fuse_s4", "blocks_s4", "up_s4_to_s2", "fuse_s2", "blocks_s2", "up_s2_to_s1", "fuse_s1",
                     "blocks_s1", "query_s4", "query_s2", "query_s1", "output_s4", "output_s2", "output_s1"):
            setattr(self, name, getattr(h, name))
        self._resize = h._resize
        self._condition = h._condition

    def forward(self, image, semantic_s2, query, coarse_logits):
        s2_hw = (int(coarse_logits.shape[-2]), int(coarse_logits.shape[-1]))
        s1_hw = (int(image.shape[-2]), int(image.shape[-1]))
        s4_hw = (max(1, s2_hw[0] // 2), max(1, s2_hw[1] // 2))
        semantic_s4 = self._resize(semantic_s2, s4_hw)
        rgb_s4 = self.rgb_s4(self._resize(image, s4_hw))
        coarse_s4_logits = self._resize(coarse_logits[:, None], s4_hw)[:, 0]
        feature_s4 = self.fuse_s4(torch.cat([semantic_s4, rgb_s4,
                                             self.coarse_s4(coarse_s4_logits[:, None].to(semantic_s4.dtype))], dim=1))
        feature_s4 = self._condition(feature_s4, query, self.query_s4)
        feature_s4 = self.blocks_s4(feature_s4)
        logits_s4 = coarse_s4_logits.float() + self.output_s4(feature_s4)[:, 0].float()

        rgb_s2 = self.rgb_s2(self._resize(image, s2_hw))
        previous_s2 = self._resize(logits_s4[:, None], s2_hw)
        feature_s2 = self.fuse_s2(torch.cat([
            self.up_s4_to_s2(feature_s4), semantic_s2, rgb_s2,
            self.coarse_s2(coarse_logits[:, None].to(semantic_s2.dtype)),
            self.previous_s2(previous_s2.sigmoid().to(semantic_s2.dtype))], dim=1))
        feature_s2 = self._condition(feature_s2, query, self.query_s2)
        feature_s2 = self.blocks_s2(feature_s2)
        logits_s2 = coarse_logits.float() + self.output_s2(feature_s2)[:, 0].float()

        semantic_s1 = self._resize(semantic_s2, s1_hw)
        rgb_s1 = self.rgb_s1(image)
        upsampled_s2 = self.up_s2_to_s1(feature_s2)
        previous_s1 = self._resize(logits_s2[:, None], s1_hw)
        feature_s1 = self.fuse_s1(torch.cat([
            upsampled_s2, semantic_s1, rgb_s1,
            self.previous_s1(previous_s1.sigmoid().to(semantic_s1.dtype))], dim=1))
        feature_s1 = self._condition(feature_s1, query, self.query_s1)
        feature_s1 = self.blocks_s1(feature_s1)
        logits_s1 = previous_s1[:, 0].float() + self.output_s1(feature_s1)[:, 0].float()
        return logits_s1, logits_s2


def pad_queries(t_seg: torch.Tensor, boxes01: torch.Tensor, n_max: int = N_MAX, pad_seg=None, pad_box=None):
    """The host's padding contract: real rows first; padding rows = zeros (t_seg) and a fixed in-range box, valid 0."""
    n = int(t_seg.shape[0])
    if n > n_max:
        raise ValueError(f"{n} queries > n_max {n_max}: the host splits them into chunks of {n_max}")
    seg = torch.zeros(n_max, t_seg.shape[1], dtype=t_seg.dtype) if pad_seg is None else pad_seg.clone()
    box = torch.tensor([[0.25, 0.25, 0.75, 0.75]], dtype=boxes01.dtype).expand(n_max, 4).clone() \
        if pad_box is None else pad_box.clone()
    seg[:n], box[:n] = t_seg, boxes01
    valid = torch.zeros(n_max, dtype=t_seg.dtype)
    valid[:n] = 1.0
    return seg, box, valid


def run_m(graph_a: GraphA, graph_b: GraphB, image, maps, t_seg, boxes01, n_max: int = N_MAX, **pad):
    """Host order: graph A once per chunk of n_max, graph B once per real query. Returns the package's dict keys."""
    n = int(t_seg.shape[0])
    outs = {"mask_logits": [], "mask_logits_aux": [], "mask_logits_coarse": [], "queries": [], "memory": None,
            "semantic_s2": None, "scores": []}
    for start in range(0, n, n_max):
        end = min(n, start + n_max)
        seg, box, valid = pad_queries(t_seg[start:end], boxes01[start:end], n_max, **pad)
        if isinstance(graph_a, GraphAHF):
            q, coarse, sem, memory = graph_a(image, *maps, seg, box, torch.from_numpy(box_fourier_np(box.numpy())),
                                             valid)
        else:
            q, coarse, sem, memory = graph_a(image, *maps, seg, box, valid)
        outs["memory"], outs["semantic_s2"] = memory, sem
        outs["scores"].extend(graph_a.last_scores)
        for i in range(end - start):
            l1, l2 = graph_b(image, sem, q[i:i + 1], coarse[i:i + 1])
            outs["mask_logits"].append(l1)
            outs["mask_logits_aux"].append(l2)
        outs["mask_logits_coarse"].append(coarse[:end - start])
        outs["queries"].append(q[:end - start])
    for k in ("mask_logits", "mask_logits_aux", "mask_logits_coarse", "queries"):
        outs[k] = torch.cat(outs[k], dim=0)
    return outs
