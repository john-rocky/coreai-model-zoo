# MiniCPM-V-4.7 vision tower in plain torch: SigLIP 27 layers + window-attention merger after
# layer 6 + DownsampleMLP merger, for one 448x448 thumbnail (32x32 patches -> 8x8 = 64 LLM rows).
#
# Community port — NOT an Apple model.
#
# Re-authored from transformers 5.19.0 `models/minicpmv4_7/modeling_minicpmv4_7.py` L55-604
# (4.6's code, renamed; main 22cfdfaa is identical) and `vision_utils.py`
# (`get_vision_nearest_position_ids` L326, `get_vision_window_index` L130). Weights: the
# checkpoint's `model.vision_tower.*` + `model.merger.*`, 459 tensors, loaded strictly.
#
# Shapes (hidden 1152, 16 heads x 72, MLP 4304, LayerNorm eps 1e-6 everywhere):
#   pixel_values [1, 3, 448, 448]
#   embeddings      patch Conv2d 3->1152 (k = s = 14, bias) + position_embedding [4900, 1152]
#                   (70 x 70 table) at nearest-bucket ids                          -> [1, 1024, 1152]
#   layers 0..6     pre-LN encoder layers on 1024 tokens, plain bidirectional attention
#   vit_merger      layer_norm1 -> windowed attention (2x2 windows, 4 tokens) + residual
#                   -> 2x2 concat [256, 4608] -> pre_norm -> linear_1 4608->17216 -> GELU
#                   -> linear_2 17216->1152, + the window mean                      -> [1, 256, 1152]
#   layers 7..26    the same encoder layers on 256 tokens
#   post_layernorm
#   merger          2x2 concat [64, 4608] -> pre_norm -> linear_1 4608->4608 -> GELU
#                   -> linear_2 4608->1024                                           -> [64, 1024]
#
# Activations, read in the code (not carried over from the 4.6 notes):
#   encoder MLP        ACT2FN["gelu_pytorch_tanh"] = transformers GELUTanh = F.gelu(approximate="tanh")
#   vit_merger MLP     `gelu_pytorch_tanh` = the same GELUTanh                  (tanh)
#   merger (Downsample) nn.GELU()                                                (erf)
# Every Linear and the patch Conv2d carry a bias. Attention: scale 72**-0.5 applied after q.k^T,
# softmax in fp32 (HF `eager_attention_forward`).
#
# Two paths on the same weights:
#   (a) reference (default) — HF's order of computation: the position ids and the window index
#       computed at run time by copies of the two vision_utils functions; attention split by
#       cu_seqlens as HF's non-flash branch does (one segment of 1024 / 256 tokens in the encoder
#       = plain bidirectional attention; 256 segments of 4 in the window merger, one call each).
#   (b) baked — `bake_constants(grid)` stores the position ids, the window index and the final
#       2x2 merge order as int32 buffers; forward is then gathers, reshapes and matmuls, and the
#       window merger's attention is ONE batched attention over [256 windows, 16 heads, 4, 72].
#       No bucketize / sort / nonzero / unique / data-dependent split reaches torch.export.
# The window merger leaves its output in window order (the HF code has no inverse permutation):
# window (wh, ww) row-major is the 16x16 merged grid row-major, which the final merge reads.
#
# Pixels: (x / 255 - 0.5) / 0.5, CHW. The HF processor hands the same pixels packed as
# [1, 3, 14, 1024 * 14] (patch k = row-major (i, j) in columns 14k .. 14k + 13); the patch Conv2d
# sees the same 1024 patches in the same order either way. `pack_patches` / `unpack_patches`
# convert exactly (a permutation).
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HIDDEN = 1152
LAYERS = 27
HEADS = 16
HEAD_DIM = HIDDEN // HEADS                # 72
MLP_DIM = 4304
EPS = 1e-6
PATCH = 14
POS_SIDE = 980 // PATCH                   # 70: position_embedding [4900, 1152]
INSERT_LAYER_ID = 6
WINDOW = 2                                # window_kernel_size (2, 2)
MERGE = 2                                 # merge_kernel_size (2, 2), merger_times 1
LLM_DIM = 1024
GRID = 448 // PATCH                       # 32 patches per side for the shipped thumbnail
N_ROWS = (GRID // WINDOW // MERGE) ** 2   # 64 image rows
CKPT_PREFIXES = ("model.vision_tower.", "model.merger.")
N_CKPT_TENSORS = 459


# -- copies of transformers 5.19.0 vision_utils (kwargs pop dropped) --------------------------

def nearest_position_ids(target_sizes: torch.Tensor, num_patches_per_side: int) -> torch.Tensor:
    """`get_vision_nearest_position_ids` (vision_utils.py L326-356): per image (h, w), bucket
    i / h on a `num_patches_per_side` grid (`bucketize`, right=True) -> flat ids into the table."""
    device = target_sizes.device
    boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side, device=device)
    pos_ids_list = []
    for height, width in target_sizes.tolist():
        height, width = int(height), int(width)
        h_coords = torch.arange(height, device=device) / height
        w_coords = torch.arange(width, device=device) / width
        bucket_h = torch.bucketize(h_coords, boundaries, right=True)
        bucket_w = torch.bucketize(w_coords, boundaries, right=True)
        pos_ids_list.append((bucket_h[:, None] * num_patches_per_side + bucket_w).flatten())
    return torch.cat(pos_ids_list)


def window_index(grid_thw: torch.Tensor, spatial_merge_size: int, window_size: int,
                 patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`get_vision_window_index` (vision_utils.py L130-188). For [[1, 32, 32]], merge 1, window 2,
    patch 1: the grid is padded by a whole window (pad = 2 - 32 % 2 = 2, all -100, dropped), the
    1024 tokens come out window-major ((wh, ww, dh, dw) -> (2wh + dh) * 32 + 2ww + dw) and
    cu_window_seqlens = [0, 4, ..., 1024]."""
    window_index: list = []
    cu_window_seqlens: list = [0]
    window_index_id = 0
    vit_merger_window_size = window_size // spatial_merge_size // patch_size
    spatial_merge_unit = spatial_merge_size**2

    for grid_t, grid_h, grid_w in grid_thw.tolist():
        grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
        llm_grid_h = grid_h // spatial_merge_size
        llm_grid_w = grid_w // spatial_merge_size
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = index_padded.reshape(
            grid_t, num_windows_h, vit_merger_window_size, num_windows_w, vit_merger_window_size
        )
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t, num_windows_h * num_windows_w, vit_merger_window_size, vit_merger_window_size
        )
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        window_index.append(index_new + window_index_id)
        cu_seqlens_tmp = seqlens.cumsum(0) * spatial_merge_unit + cu_window_seqlens[-1]
        cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
        window_index_id += grid_t * llm_grid_h * llm_grid_w

    window_index = torch.cat(window_index, dim=0)
    cu_window_seqlens = torch.tensor(cu_window_seqlens, device=grid_thw.device, dtype=torch.int32)
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
    return window_index, cu_window_seqlens


def segment_bounds(target_sizes: torch.Tensor) -> torch.Tensor:
    """cu_seqlens of the encoder attention: one segment per image (h * w tokens)."""
    return F.pad(torch.cumsum(target_sizes[:, 0] * target_sizes[:, 1], dim=0, dtype=torch.int32), (1, 0))


# -- pixel layout -------------------------------------------------------------------------------

def pack_patches(pixel_values: torch.Tensor, patch: int = PATCH) -> torch.Tensor:
    """[B, 3, H, W] -> the processor's packing [B, 3, patch, (H/patch)(W/patch) patch]."""
    b, c, h, w = pixel_values.shape
    gh, gw = h // patch, w // patch
    x = pixel_values.reshape(b, c, gh, patch, gw, patch).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(b, c, patch, gh * gw * patch)


def unpack_patches(packed: torch.Tensor, grid_h: int, grid_w: int, patch: int = PATCH) -> torch.Tensor:
    """The processor's packing [B, 3, patch, grid_h * grid_w * patch] -> [B, 3, H, W]."""
    b, c = packed.shape[:2]
    x = packed.reshape(b, c, patch, grid_h, grid_w, patch).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(b, c, grid_h * patch, grid_w * patch)


# -- modules (names = checkpoint keys minus "model.") -------------------------------------------

def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


class VisionEmbeddings(nn.Module):
    """`MiniCPMV4_7VisionEmbeddings` (L55)."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv2d(3, HIDDEN, kernel_size=PATCH, stride=PATCH, padding="valid")
        self.position_embedding = nn.Embedding(POS_SIDE * POS_SIDE, HIDDEN)

    def forward(self, pixel_values: torch.Tensor, pos_ids: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)    # [1, N, D]
        return x + self.position_embedding(pos_ids).unsqueeze(0)


class VisionAttention(nn.Module):
    """`MiniCPMV4_7VisionAttention` (L153), eager."""

    def __init__(self) -> None:
        super().__init__()
        self.num_heads = HEADS
        self.head_dim = HEAD_DIM
        self.scaling = HEAD_DIM**-0.5
        self.k_proj = nn.Linear(HIDDEN, HIDDEN)
        self.v_proj = nn.Linear(HIDDEN, HIDDEN)
        self.q_proj = nn.Linear(HIDDEN, HIDDEN)
        self.out_proj = nn.Linear(HIDDEN, HIDDEN)

    def _attend(self, q, k, v):
        """`eager_attention_forward` (L128-150): [.., H, T, hd] -> [.., T, H, hd]."""
        w = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(w, v).transpose(-3, -2).contiguous()

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor | None = None) -> torch.Tensor:
        """x [1, N, D]. `cu_seqlens` (reference path): one eager call per segment, as HF's non-flash
        branch (L214-234); None = the whole row is one segment (the baked path)."""
        b, n, _ = x.shape
        shape = (b, n, -1, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        if cu_seqlens is None:
            o = self._attend(q, k, v)
        else:
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            splits = [torch.split(t, lengths, dim=2) for t in (q, k, v)]
            o = torch.cat([self._attend(qs, ks, vs) for qs, ks, vs in zip(*splits)], dim=1)
        return self.out_proj(o.reshape(b, n, -1).contiguous())

    def forward_windows(self, x: torch.Tensor) -> torch.Tensor:
        """x [W, T, D] = W independent windows of T tokens -> [W, T, D], one batched attention.

        q / k / v are made contiguous in [W, H, T, hd]: torch.matmul then folds (W, H) into one
        batch as a view and hands k^T to BLAS as a transposed operand, the same call the
        reference path's per-window views make, so the two paths round alike on the CPU. Without
        it the fold copies k^T into a plain matrix and the scores differ by an ulp, which grows to
        ~2e-5 at the tower output through the residual stream's large channels (absmax ~1.3e3)."""
        w, t, _ = x.shape
        shape = (w, t, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2).contiguous()         # [W, H, T, hd]
        k = self.k_proj(x).view(shape).transpose(1, 2).contiguous()
        v = self.v_proj(x).view(shape).transpose(1, 2).contiguous()
        o = self._attend(q, k, v)                                           # [W, T, H, hd]
        return self.out_proj(o.reshape(w, t, -1))


class VisionMLP(nn.Module):
    """`MiniCPMV4_7VisionMLP` (L101): fc1 -> gelu tanh -> fc2."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(HIDDEN, MLP_DIM)
        self.fc2 = nn.Linear(MLP_DIM, HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(_gelu_tanh(self.fc1(x)))


class EncoderLayer(nn.Module):
    """`MiniCPMV4_7VisionEncoderLayer` (L242): pre-LN attention and MLP, each with a residual."""

    def __init__(self) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(HIDDEN, eps=EPS)
        self.self_attn = VisionAttention()
        self.layer_norm2 = nn.LayerNorm(HIDDEN, eps=EPS)
        self.mlp = VisionMLP()

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x), cu_seqlens)
        return x + self.mlp(self.layer_norm2(x))


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(LAYERS)])


class WindowMerger(nn.Module):
    """`MiniCPMV4_7ViTWindowAttentionMerger` (L304): 2x2 windows -> one token each."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = VisionAttention()
        self.layer_norm1 = nn.LayerNorm(HIDDEN, eps=EPS)
        self.pre_norm = nn.LayerNorm(HIDDEN * WINDOW * WINDOW, eps=EPS)
        self.linear_1 = nn.Linear(HIDDEN * WINDOW * WINDOW, MLP_DIM * WINDOW * WINDOW)
        self.linear_2 = nn.Linear(MLP_DIM * WINDOW * WINDOW, HIDDEN)

    def _merge(self, patch: torch.Tensor) -> torch.Tensor:
        """patch [W, 4, D] (attention + residual, window order) -> [1, W, D] (L388-396)."""
        flat = patch.flatten(1)
        patch_residual = patch.mean(dim=1)
        h = self.linear_2(_gelu_tanh(self.linear_1(self.pre_norm(flat))))
        return (h + patch_residual).unsqueeze(0)

    def forward_reference(self, x: torch.Tensor, target_sizes: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.layer_norm1(x)
        idx, cu = window_index(F.pad(target_sizes, (1, 0), value=1), spatial_merge_size=1,
                               window_size=WINDOW, patch_size=1)
        h = h[:, idx, :]
        h = self.self_attn(h, cu)
        h = residual[:, idx, :] + h
        return self._merge(h.reshape(-1, WINDOW * WINDOW, h.shape[-1]))

    def forward_baked(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.layer_norm1(x).index_select(1, idx)                        # [1, N, D] window order
        n_win = h.shape[1] // (WINDOW * WINDOW)
        h = self.self_attn.forward_windows(h.reshape(n_win, WINDOW * WINDOW, HIDDEN))
        h = residual.index_select(1, idx).reshape(n_win, WINDOW * WINDOW, HIDDEN) + h
        return self._merge(h)


class VisionTower(nn.Module):
    """`MiniCPMV4_7VisionModel` (L415)."""

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = VisionEmbeddings()
        self.encoder = Encoder()
        self.post_layernorm = nn.LayerNorm(HIDDEN, eps=EPS)
        self.vit_merger = WindowMerger()


class DownsampleMLP(nn.Module):
    """`MiniCPMV4_7DownsampleMLP` (L499): LayerNorm(4608) -> Linear -> GELU (erf) -> Linear."""

    def __init__(self, hidden_size: int, llm_embed_dim: int) -> None:
        super().__init__()
        merged = hidden_size * MERGE * MERGE
        self.pre_norm = nn.LayerNorm(merged, eps=1e-6)
        self.linear_1 = nn.Linear(merged, merged)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(merged, llm_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x).view(-1, self.linear_1.in_features)
        return self.linear_2(self.act(self.linear_1(x)))


class Merger(nn.Module):
    """`MiniCPMV4_7Merger` (L518), merger_times 1: one 2x2 merge + DownsampleMLP into the LLM."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.ModuleList([DownsampleMLP(HIDDEN, LLM_DIM)])

    def forward_reference(self, x: torch.Tensor, target_sizes: torch.Tensor) -> torch.Tensor:
        height, width = (int(v) for v in target_sizes[0].tolist())
        merged_h, merged_w = height // MERGE, width // MERGE
        h = (x[0, : height * width, :]
             .view(merged_h, MERGE, merged_w, MERGE, x.shape[-1])
             .permute(0, 2, 1, 3, 4)
             .reshape(merged_h * merged_w, MERGE * MERGE * x.shape[-1]))
        return self.mlp[0](h)


class MiniCPMV47Vision(nn.Module):
    """pixel_values [1, 3, 448, 448] -> image_features [64, 1024]; contract in the header."""

    def __init__(self) -> None:
        super().__init__()
        self.vision_tower = VisionTower()
        self.merger = Merger()
        self.baked_grid: int | None = None

    # -- (a) reference --------------------------------------------------------------------------

    def forward_reference(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vt = self.vision_tower
        ts = torch.tensor([[pixel_values.shape[-2] // PATCH, pixel_values.shape[-1] // PATCH]])
        x = vt.embeddings(pixel_values, nearest_position_ids(ts, POS_SIDE))
        cu = segment_bounds(ts)
        for i, layer in enumerate(vt.encoder.layers):
            x = layer(x, cu)
            if i == INSERT_LAYER_ID:
                x = vt.vit_merger.forward_reference(x, ts)
                ts = ts // 2                                   # `get_downsampled_inputs` (L426)
                cu = segment_bounds(ts)
        x = vt.post_layernorm(x)
        return self.merger.forward_reference(x, ts)

    # -- (b) baked ------------------------------------------------------------------------------

    def bake_constants(self, grid: int = GRID) -> "MiniCPMV47Vision":
        """Store the grid's position ids, window index and 2x2 merge order as int32 buffers; the
        forward then has no data-dependent op. Checks the window index against its closed form."""
        ts = torch.tensor([[grid, grid]])
        pos = nearest_position_ids(ts, POS_SIDE)
        idx, cu = window_index(F.pad(ts, (1, 0), value=1), spatial_merge_size=1, window_size=WINDOW,
                               patch_size=1)
        side = grid // WINDOW                                            # 16 after the window merger
        if cu.tolist() != list(range(0, grid * grid + 1, WINDOW * WINDOW)):
            raise ValueError(f"grid {grid}: windows are not all {WINDOW * WINDOW} tokens")
        wh, ww, dh, dw = torch.meshgrid(*(torch.arange(n) for n in (side, side, WINDOW, WINDOW)), indexing="ij")
        if not torch.equal(idx, ((WINDOW * wh + dh) * grid + WINDOW * ww + dw).reshape(-1)):
            raise ValueError("window index differs from the window-major closed form")
        m = side // MERGE
        merge = torch.arange(side * side).view(m, MERGE, m, MERGE).permute(0, 2, 1, 3).reshape(-1)
        self.register_buffer("pos_ids", pos.to(torch.int32), persistent=False)
        self.register_buffer("window_index", idx.to(torch.int32), persistent=False)
        self.register_buffer("merge_index", merge.to(torch.int32), persistent=False)
        self.baked_grid = grid
        return self

    def forward_baked(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vt = self.vision_tower
        layers = vt.encoder.layers
        x = vt.embeddings(pixel_values, self.pos_ids)
        for layer in layers[: INSERT_LAYER_ID + 1]:
            x = layer(x)
        x = vt.vit_merger.forward_baked(x, self.window_index)
        for layer in layers[INSERT_LAYER_ID + 1:]:
            x = layer(x)
        x = vt.post_layernorm(x)
        n = self.merge_index.shape[0] // (MERGE * MERGE)
        x = x[0].index_select(0, self.merge_index).reshape(n, MERGE * MERGE * HIDDEN)
        return self.merger.mlp[0](x)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.baked_grid is None:
            return self.forward_reference(pixel_values)
        return self.forward_baked(pixel_values)

    # -- loading --------------------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, snapshot_dir, dtype: torch.dtype = torch.float32) -> "MiniCPMV47Vision":
        """Built on the meta device, then every tensor assigned from the checkpoint."""
        with torch.device("meta"):
            model = cls()
        model.load_from_safetensors(snapshot_dir, dtype)
        return model.eval()

    def load_from_safetensors(self, snapshot_dir, dtype: torch.dtype = torch.float32) -> dict:
        """`model.vision_tower.*` + `model.merger.*` from the snapshot's model.safetensors,
        strict: every module tensor filled, every tower/merger key used. `self.load_report`."""
        from safetensors import safe_open

        sd, other = {}, 0
        with safe_open(str(Path(snapshot_dir) / "model.safetensors"), framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                if key.startswith(CKPT_PREFIXES):
                    sd[key[len("model."):]] = f.get_tensor(key).to(dtype)
                else:
                    other += 1
        missing, unexpected = self.load_state_dict(sd, strict=False, assign=True)
        on_meta = [n for n, t in list(self.named_parameters()) + list(self.named_buffers()) if t.is_meta]
        self.load_report = {
            "checkpoint_tower_merger_keys": len(sd), "expected": N_CKPT_TENSORS,
            "checkpoint_other_keys_skipped": other,
            "module_tensors": len(self.state_dict()),
            "missing": list(missing), "unexpected": list(unexpected), "on_meta": on_meta,
            "dtype": str(dtype),
        }
        if missing or unexpected or on_meta or len(sd) != N_CKPT_TENSORS:
            raise RuntimeError(f"load: {self.load_report}")
        return self.load_report


class VisionExport(nn.Module):
    """Export wrapper: pixel_values [1, 3, 448, 448] -> image_features [64, 1024], grid baked."""

    def __init__(self, core: MiniCPMV47Vision) -> None:
        super().__init__()
        if core.baked_grid is None:
            raise ValueError("call core.bake_constants(grid) first")
        self.core = core

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.core.forward_baked(pixel_values)
