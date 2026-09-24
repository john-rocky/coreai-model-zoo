"""Plain-PyTorch re-authoring of nvidia/Nemotron-3-Diarization's per-chunk network, loaded straight
from `model.safetensors` (no transformers import). Runs in the export venv (torch 2.9).

One static graph, fixed T (541 = speaker cache 264 + FIFO 264 + chunk 9 + look-ahead 4 for the
low-latency streaming mode):

    packed [1, T, 512]   host-built [cache | FIFO | chunk embeds]: the embedder output
                         (8-frame stacking + Linear 1024->512, computed on the host), i.e. the
                         `chunk_input_embeds` transformers feeds to `Nemotron3DiarizationModel`.
                         The L real rows sit at [0, L); rows [L, T) are padding (any value).
    valid  [1, T]        1.0 for rows < L, 0.0 after.
    -> logits [1, T*8, 8]  speaker logits at the 10 ms rate; rows [0, L*8) are the real ones.

In graph: input LayerNorm -> 31 pre-LN layers (RoPE on all 64 head dims, theta 1e4, positions
0..T-1 baked as constant cos/sin buffers; plain matmul/softmax attention with the additive key mask
(1 - valid) * -1e4; erf GELU) -> final LayerNorm -> proj 512->192 -> x valid (zeroes the padding
rows so the next Conv1d sees the same implicit zero padding as the unpadded sequence) -> sub-pixel
Conv1d 192->1536 (k=3, p=1) -> reshape [1, T*8, 192] -> classifier (relu, dense, relu, out_proj).
Sigmoid, the 8x average pool and the speaker cache (AOSC + FIFO) stay on the host.

Rows are left-aligned, so every real row gets the RoPE position transformers gives it (positions
restart at 0 every chunk there too).

Checkpoint key -> where it goes (417 tensors, all F32, 99,226,504 parameters):

    model.audio_tower.embedder.projection.weight [512, 1024]   host (HOST_KEYS)
    silence_embeds [512]                                       host (HOST_KEYS)
    model.audio_tower.input_layer_norm.{weight,bias}           graph, same name
    model.audio_tower.layers.{0..30}.layer_norm1.{weight,bias} graph, same name
    model.audio_tower.layers.{i}.self_attn.{q,k,v}_proj.weight graph, same name (no bias)
    model.audio_tower.layers.{i}.self_attn.o_proj.{weight,bias} graph, same name
    model.audio_tower.layers.{i}.layer_norm2.{weight,bias}     graph, same name
    model.audio_tower.layers.{i}.mlp.fc{1,2}.{weight,bias}     graph, same name
    model.audio_tower.layer_norm.{weight,bias}                 graph, same name
    model.proj.{weight [192, 512], bias}                       graph, same name
    model.upsampler.conv.{weight [1536, 192, 3], bias}         graph, same name
    classifier.dense.{weight, bias}                            graph, same name
    classifier.out_proj.{weight [8, 192], bias}                graph, same name
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.dont_write_bytecode = True  # importing conversion/_paths must not leave a __pycache__ outside this dir
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import hf_snapshot  # noqa: E402

REPO_ID = "nvidia/Nemotron-3-Diarization"
REVISION = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
SAFETENSORS_SHA256 = "c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6"
N_PARAMS = 99_226_504
N_TENSORS = 417

HID, HEADS, HDIM, FF, NLAYERS = 512, 8, 64, 2048, 31
HEAD_HID, N_SPK, SUB, MEL = 192, 8, 8, 128
ROPE_THETA = 10000.0
LN_EPS = 1e-5
NEG = -1e4

T_STREAM = 541   # low_latency: 264 cache + 264 FIFO + 9 chunk + 4 look-ahead
T_OFFLINE = 684  # offline: 264 cache + 40 FIFO + 340 chunk + 40 look-ahead (round 2)

HOST_KEYS = ("model.audio_tower.embedder.projection.weight", "silence_embeds")


def safetensors_path() -> str:
    return hf_snapshot(REPO_ID, "model.safetensors", revision=REVISION)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rope_tables(T: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [T, 64] for positions 0..T-1, computed with the same fp32 ops as transformers'
    default RoPE (inv_freq @ positions, cat(freqs, freqs), cos/sin), so the baked constants are
    bit-identical to what the reference computes per call."""
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HDIM, 2, dtype=torch.float) / HDIM))
    position_ids = torch.arange(T)[None, :]
    inv_freq_expanded = inv_freq[None, :, None].float().expand(1, -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[0], emb.sin()[0]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : HDIM // 2]
    x2 = x[..., HDIM // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HID, HID, bias=False)
        self.k_proj = nn.Linear(HID, HID, bias=False)
        self.v_proj = nn.Linear(HID, HID, bias=False)
        self.o_proj = nn.Linear(HID, HID, bias=True)

    def forward(self, x, cos, sin, key_bias):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, HEADS, HDIM).transpose(1, 2)   # [B, H, T, 64]
        k = self.k_proj(x).view(B, T, HEADS, HDIM).transpose(1, 2)
        v = self.v_proj(x).view(B, T, HEADS, HDIM).transpose(1, 2)
        if cos is not None:
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
        scores = torch.matmul(q, k.transpose(2, 3)) * (HDIM ** -0.5)
        if key_bias is not None:
            scores = scores + key_bias                                 # [B, 1, 1, T] broadcast
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).reshape(B, T, HID)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(HID, FF)
        self.fc2 = nn.Linear(FF, HID)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))                          # erf GELU (approximate="none")


class SafeLayerNorm(nn.LayerNorm):
    """nn.LayerNorm computed on x / s (s = 64): xs = x * (1/s), ys = (xs - mean) / sqrt(var(xs) + eps/s^2),
    y = ys * w + b. Equal to LayerNorm(x) in exact arithmetic; in fp16 the squares stay in range. Measured
    on the fp32 graph (captured steps of both fixtures): the in-layer and final LN inputs reach |x| 1,216
    and a row variance of 24,968, so (x - mean)^2 alone passes fp16's 65,504; after /64 the largest
    square is ~361 and the largest row sum of squares 3,121. Same parameter names as nn.LayerNorm.
    Not used for input_layer_norm: its input is the packed embeds (|x| <= 141), and its zero padding
    rows would divide 0 by 0 once eps/s^2 (2.4e-9) underflows to 0 in fp16."""

    def __init__(self, dim: int, eps: float, scale: float = 64.0):
        super().__init__(dim, eps=eps)
        self.inv_scale = 1.0 / scale
        self.eps_scaled = eps / (scale * scale)

    def forward(self, x):
        xs = x * self.inv_scale
        c = xs - xs.mean(-1, keepdim=True)
        var = (c * c).mean(-1, keepdim=True)
        return c / torch.sqrt(var + self.eps_scaled) * self.weight + self.bias


def layer_norm(safe: bool) -> nn.LayerNorm:
    return SafeLayerNorm(HID, LN_EPS) if safe else nn.LayerNorm(HID, eps=LN_EPS)


class Layer(nn.Module):
    def __init__(self, safe_ln: bool = False):
        super().__init__()
        self.layer_norm1 = layer_norm(safe_ln)
        self.self_attn = Attention()
        self.layer_norm2 = layer_norm(safe_ln)
        self.mlp = MLP()

    def forward(self, x, cos, sin, key_bias):
        x = x + self.self_attn(self.layer_norm1(x), cos, sin, key_bias)
        return x + self.mlp(self.layer_norm2(x))


class AudioTower(nn.Module):
    """transformers' Nemotron3DiarizationAudioModel minus the embedder (which runs on the host)."""

    def __init__(self, safe_ln: bool = False):
        super().__init__()
        self.input_layer_norm = nn.LayerNorm(HID, eps=LN_EPS)
        self.layers = nn.ModuleList([Layer(safe_ln) for _ in range(NLAYERS)])
        self.layer_norm = layer_norm(safe_ln)


class Upsampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(HEAD_HID, HEAD_HID * SUB, kernel_size=3, padding=1)


class Core(nn.Module):
    def __init__(self, safe_ln: bool = False):
        super().__init__()
        self.audio_tower = AudioTower(safe_ln)
        self.proj = nn.Linear(HID, HEAD_HID)
        self.upsampler = Upsampler()


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(HEAD_HID, HEAD_HID)
        self.out_proj = nn.Linear(HEAD_HID, N_SPK)


class N3DGraph(nn.Module):
    """The exported graph. Parameter names equal the checkpoint keys (minus HOST_KEYS).

    The first three switches exist only for the gate's negative controls; the export uses the
    defaults. safe_ln=True swaps the in-layer and final LayerNorms for SafeLayerNorm (the fp16 / ANE
    variant; same parameters, same function in exact arithmetic).
    """

    def __init__(self, T: int = T_STREAM, use_rope: bool = True, use_key_mask: bool = True,
                 zero_pad_rows: bool = True, safe_ln: bool = False):
        super().__init__()
        self.T = T
        self.use_rope = use_rope
        self.use_key_mask = use_key_mask
        self.zero_pad_rows = zero_pad_rows
        self.safe_ln = safe_ln
        self.model = Core(safe_ln)
        self.classifier = Head()
        cos, sin = rope_tables(T)
        self.register_buffer("rope_cos", cos, persistent=False)      # [T, 64]
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, packed: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        T = self.T
        tower = self.model.audio_tower
        cos = self.rope_cos.view(1, 1, T, HDIM) if self.use_rope else None
        sin = self.rope_sin.view(1, 1, T, HDIM) if self.use_rope else None
        key_bias = ((1.0 - valid) * NEG).view(1, 1, 1, T) if self.use_key_mask else None

        x = tower.input_layer_norm(packed)
        for layer in tower.layers:
            x = layer(x, cos, sin, key_bias)
        x = tower.layer_norm(x)

        h = self.model.proj(x)                                          # [1, T, 192]
        if self.zero_pad_rows:
            h = h * valid.view(1, T, 1)
        h = self.model.upsampler.conv(h.transpose(1, 2)).transpose(1, 2)   # [1, T, 1536]
        h = h.reshape(1, T * SUB, HEAD_HID)                             # [1, T*8, 192]
        h = self.classifier.dense(F.relu(h))
        return self.classifier.out_proj(F.relu(h))                      # [1, T*8, 8]


def load_checkpoint(path: str | None = None, T: int = T_STREAM, verify_sha: bool = True,
                    **switches) -> tuple[N3DGraph, dict[str, torch.Tensor]]:
    """Strict load: every checkpoint tensor lands in exactly one place (graph param or host),
    nothing is missing, nothing is left over, and the parameter count is the published one."""
    from safetensors.torch import load_file

    path = path or safetensors_path()
    if verify_sha:
        got = sha256_file(path)
        assert got == SAFETENSORS_SHA256, f"model.safetensors sha256 {got} != {SAFETENSORS_SHA256}"
    sd = load_file(path)
    assert len(sd) == N_TENSORS, f"{len(sd)} tensors, expected {N_TENSORS}"
    assert all(v.dtype == torch.float32 for v in sd.values()), "non-F32 tensor in the checkpoint"
    n = sum(v.numel() for v in sd.values())
    assert n == N_PARAMS, f"{n} params, expected {N_PARAMS}"

    host = {k: sd.pop(k) for k in HOST_KEYS}
    graph = N3DGraph(T, **switches)
    graph.load_state_dict(sd, strict=True)
    n_graph = sum(p.numel() for p in graph.parameters())
    assert n_graph + sum(v.numel() for v in host.values()) == N_PARAMS
    return graph.eval(), host


def embed_chunk(mel: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    """transformers' feature stacking (torch reference for the host path): mel [N, 128] ->
    zero-pad N to a multiple of 8 -> [N/8, 1024] -> @ projection.T -> [N/8, 512]."""
    pad = -mel.shape[0] % SUB
    mel = F.pad(mel, (0, 0, 0, pad))
    return mel.reshape(-1, SUB * MEL) @ projection.T


if __name__ == "__main__":
    g, host = load_checkpoint()
    n_graph = sum(p.numel() for p in g.parameters())
    print(f"strict load OK: graph {n_graph:,} + host {sum(v.numel() for v in host.values()):,} "
          f"= {N_PARAMS:,} params")
