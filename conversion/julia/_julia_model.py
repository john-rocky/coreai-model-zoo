"""Static, plain-PyTorch authoring graph for Julia-1 (mmBERT-small encoder + typed decision head).

Re-authored from the raw checkpoint (model.safetensors, encoder/config.json, julia_config.json) and
gated against the publisher's own model (oracle_julia.py). It neither wraps nor imports transformers
or the publisher's `julia` package. One call is one question row, batch 1, static sequence length S:

  main  input_ids [1,S] int32 (right-padded with PAD 0), attention_mask [1,S] int32 (1 real / 0 pad),
        qtype_onehot [1,3] float32 (choice / score / noul)
        -> token_logits [1,S] float32 (the scorer at every position)

Marker gathering, softmax (T = 1) and the typed answers are host work (_julia_host.py).

What the checkpoint runs (julia/model.py JuliaDecisionModel over transformers 5.0 ModernBERT; the
same forward as laya's DecisionModel, at mmBERT-small's width):
  - encoder: token embedding + biasless LayerNorm; 22 pre-norm layers (layer 0 has no attention norm),
    GLU MLP `Wo(gelu(x) * gate)` with exact GELU (intermediate 1152); 6 heads x 64; global attention on
    layers 0, 3, …, 21, sliding attention elsewhere with the inclusive radius |i - j| <= 64
    (local_attention 128 / 2); RoPE theta 160000 for both layer kinds, half-split rotation, positions
    arange(S), trig in fp32; final_norm
  - `h + type_emb[qtype]`, then two post-type `nn.TransformerEncoderLayer(384, 6, 1536, relu,
    norm_first=True)` with the key padding mask, re-implemented explicitly here (PyTorch's fused eval
    fast path is not a graph)
  - scorer LayerNorm -> Linear -> GELU -> Linear(384, 1) at every position
  - the checkpoint also carries `act_head` (Linear(388, 256) -> GELU -> Linear(256, 2)) and a
    `temperature` buffer of [1, 1, 1]; the publisher's inference API reads neither (forward with
    return_actions=False; no temperature anywhere), so neither is exported. They are loaded (strict) and
    left out of the graph.

Masks are additive, float arithmetic only, with MASK_VALUE = -1e4: masked keys get exactly zero weight
in fp32 and fp16 (exp(-1e4) underflows), and a fully masked query row (a pad position more than 64 past
the last real token in a sliding layer) is finite instead of NaN. The publisher's SDPA path returns zero
for those rows instead; pad positions are never read by any output (every attention masks pad keys;
markers are real tokens), so the two agree at every real position.

Precisions (`set_precision`): fp32; wfp16 = fp16 weight storage with fp32 compute. The checkpoint is
F32, so wfp16 ROUNDS every stored weight (laya's F16 checkpoint made the same storage exact): it is a
different model by the rounding, gated against the fp32 oracle like any other variant.

Mutations change one behaviour at a time and are negative controls, never export variants.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

MODEL_ID = "SupersonicLabs/Julia-1"
MODEL_SHA = "a85b127321d580d65176c89ced8273f305745d85"
PARAMETER_COUNT = 144_292_870
WIDTH, HEADS, HEAD_DIM, INTERMEDIATE, LAYERS, VOCAB = 384, 6, 64, 1152, 22, 256000
HEAD_FFN = 1536
MUTATIONS = ("none", "all_global", "all_local", "window63", "ignore_padding", "no_type_emb")
MASK_VALUE = -1.0e4
PRECISIONS = ("fp32", "wfp16")


def validate_config(encoder: dict[str, Any], julia: dict[str, Any]) -> None:
    """Reject any checkpoint outside this deliberately narrow recipe."""
    required = {
        "model_type": "modernbert", "hidden_size": WIDTH, "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS, "num_hidden_layers": LAYERS, "vocab_size": VOCAB, "pad_token_id": 0,
        "hidden_activation": "gelu", "attention_bias": False, "mlp_bias": False, "norm_bias": False,
        "attention_dropout": 0.0, "embedding_dropout": 0.0, "mlp_dropout": 0.0, "global_attn_every_n_layers": 3,
        "local_attention": 128, "norm_eps": 1e-5, "position_embedding_type": "sans_pos",
    }
    mismatches = {k: (encoder.get(k), v) for k, v in required.items() if encoder.get(k) != v}
    expected_types = ["full_attention" if i % 3 == 0 else "sliding_attention" for i in range(LAYERS)]
    if encoder.get("layer_types") != expected_types:
        mismatches["layer_types"] = (encoder.get("layer_types"), expected_types)
    rope = {kind: (p.get("rope_theta"), p.get("rope_type")) for kind, p in (encoder.get("rope_parameters") or {}).items()}
    if rope != {"full_attention": (160000, "default"), "sliding_attention": (160000, "default")}:
        mismatches["rope_parameters"] = (rope, "theta 160000, default, both kinds")
    julia_required = {"format_version": 1, "architecture": "JuliaDecisionModel", "head_layers": 2, "n_act": 2,
                      "weight_dtype": "float32"}
    mismatches.update({f"julia.{k}": (julia.get(k), v) for k, v in julia_required.items() if julia.get(k) != v})
    if mismatches:
        raise ValueError(f"Unsupported Julia configuration (actual, expected): {mismatches}")


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class _Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(VOCAB, WIDTH, padding_idx=0)
        self.norm = nn.LayerNorm(WIDTH, eps=1e-5, bias=False)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wqkv = nn.Linear(WIDTH, 3 * WIDTH, bias=False)
        self.Wo = nn.Linear(WIDTH, WIDTH, bias=False)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wi = nn.Linear(WIDTH, 2 * INTERMEDIATE, bias=False)
        self.Wo = nn.Linear(INTERMEDIATE, WIDTH, bias=False)


class _EncoderLayer(nn.Module):
    def __init__(self, layer_id: int):
        super().__init__()
        # The checkpoint has no attention norm on layer 0 (the embedding LayerNorm serves).
        self.attn_norm = nn.Identity() if layer_id == 0 else nn.LayerNorm(WIDTH, eps=1e-5, bias=False)
        self.attn = _Attention()
        self.mlp_norm = nn.LayerNorm(WIDTH, eps=1e-5, bias=False)
        self.mlp = _MLP()


class _Encoder(nn.Module):
    def __init__(self, seq_len: int, mutation: str):
        super().__init__()
        self.embeddings = _Embeddings()
        self.layers = nn.ModuleList([_EncoderLayer(i) for i in range(LAYERS)])
        self.final_norm = nn.LayerNorm(WIDTH, eps=1e-5, bias=False)
        # RoPE exactly as transformers 5.x computes it (fp32 inv_freq, fp32 outer product, cat, cos/sin),
        # precomputed so no trig of large constant arguments is left for the converter to fold.
        inv_freq = 1.0 / (160000 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float) / HEAD_DIM))
        positions = torch.arange(seq_len).unsqueeze(0)
        freqs = (inv_freq[None, :, None].float() @ positions[:, None, :].float()).transpose(1, 2)
        angles = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("rope_cos", angles.cos()[:, None].contiguous(), persistent=False)  # [1,1,S,64]
        self.register_buffer("rope_sin", angles.sin()[:, None].contiguous(), persistent=False)
        offsets = torch.arange(seq_len)
        radius = 63 if mutation == "window63" else 64
        band = (offsets[:, None] - offsets[None, :]).abs() <= radius
        self.register_buffer("band_mask", torch.where(band, 0.0, MASK_VALUE)[None, None].float().contiguous(),
                             persistent=False)  # [1,1,S,S]
        self.local = [(i % 3 != 0) if mutation not in ("all_global", "all_local") else mutation == "all_local"
                      for i in range(LAYERS)]


class _HeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_weight = nn.Parameter(torch.empty(3 * WIDTH, WIDTH))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * WIDTH))
        self.out_proj = nn.Linear(WIDTH, WIDTH)


class _HeadLayer(nn.Module):
    """nn.TransformerEncoderLayer(384, 6, 1536, relu, norm_first=True, batch_first=True), eval, explicit."""

    def __init__(self):
        super().__init__()
        self.self_attn = _HeadAttention()
        self.linear1 = nn.Linear(WIDTH, HEAD_FFN)
        self.linear2 = nn.Linear(HEAD_FFN, WIDTH)
        self.norm1 = nn.LayerNorm(WIDTH, eps=1e-5)
        self.norm2 = nn.LayerNorm(WIDTH, eps=1e-5)


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_HeadLayer() for _ in range(2)])


class _MainBody(nn.Module):
    """The `main` graph: encoder, type embedding, head, scorer. `shared` reuses another body's modules."""

    def __init__(self, seq_len: int, mutation: str = "none", shared: "_MainBody | None" = None):
        super().__init__()
        if mutation not in MUTATIONS:
            raise ValueError(f"Unknown mutation {mutation!r}; expected one of {MUTATIONS}")
        self.seq_len, self.mutation = int(seq_len), mutation
        if shared is None:
            self.encoder = _Encoder(self.seq_len, mutation)
            self.type_emb = nn.Embedding(3, WIDTH)
            self.head = _Head()
            self.scorer = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, WIDTH), nn.GELU(), nn.Linear(WIDTH, 1))
        else:
            self.encoder, self.type_emb, self.head, self.scorer = shared.encoder, shared.type_emb, shared.head, shared.scorer

    @staticmethod
    def _linear(module: nn.Module | None, x: torch.Tensor, weight=None, bias=None) -> torch.Tensor:
        """F.linear in fp32. A stored fp16 weight (wfp16) becomes a cast in the graph; coreai-torch 0.4.1
        keeps the constant fp16 (laya: bundle size = fp16 bytes)."""
        if module is not None:
            weight, bias = module.weight, module.bias
        return F.linear(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))

    @staticmethod
    def _layer_norm(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if isinstance(norm, nn.Identity):
            return x
        bias = norm.bias.float() if norm.bias is not None else None
        return F.layer_norm(x, norm.normalized_shape, norm.weight.float(), bias, norm.eps)

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """q, k, v [1,6,S,64]; mask [1,1,*,S] fp32 additive. The scale 64**-0.5 = 0.125 is exact."""
        scores = torch.matmul(q, k.transpose(2, 3)) * 0.125 + mask
        probabilities = F.softmax(scores, dim=-1)
        return torch.matmul(probabilities, v).transpose(1, 2).reshape(1, self.seq_len, WIDTH)

    def _masks(self, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(attention_mask.shape) != (1, self.seq_len):
            raise ValueError(f"Expected attention_mask [1,{self.seq_len}], got {tuple(attention_mask.shape)}")
        keep = attention_mask.to(torch.float32)
        if self.mutation == "ignore_padding":
            keep = torch.ones_like(keep)
        key_mask = ((1.0 - keep) * MASK_VALUE).reshape(1, 1, 1, self.seq_len)
        return key_mask, key_mask + self.encoder.band_mask

    def _encoder_layer(self, index: int, h: torch.Tensor, key_mask, local_mask) -> torch.Tensor:
        layer = self.encoder.layers[index]
        x = self._layer_norm(layer.attn_norm, h)
        q, k, v = self._linear(layer.attn.Wqkv, x).reshape(1, self.seq_len, 3, HEADS, HEAD_DIM).unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        cos, sin = self.encoder.rope_cos, self.encoder.rope_sin
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        attended = self._attend(q, k, v, local_mask if self.encoder.local[index] else key_mask)
        h = h + self._linear(layer.attn.Wo, attended)
        inputs, gate = self._linear(layer.mlp.Wi, self._layer_norm(layer.mlp_norm, h)).chunk(2, dim=-1)
        return h + self._linear(layer.mlp.Wo, F.gelu(inputs) * gate)

    def _head_layer(self, index: int, h: torch.Tensor, key_mask) -> torch.Tensor:
        layer = self.head.layers[index]
        x = self._layer_norm(layer.norm1, h)
        qkv = self._linear(None, x, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
        q, k, v = qkv.reshape(1, self.seq_len, 3, HEADS, HEAD_DIM).unbind(dim=2)
        attended = self._attend(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), key_mask)
        h = h + self._linear(layer.self_attn.out_proj, attended)
        x = F.relu(self._linear(layer.linear1, self._layer_norm(layer.norm2, h)))
        return h + self._linear(layer.linear2, x)

    def _scorer(self, h: torch.Tensor) -> torch.Tensor:
        norm, first, _, last = self.scorer
        x = F.gelu(self._linear(first, self._layer_norm(norm, h)))
        return self._linear(last, x).reshape(1, self.seq_len)

    def _run(self, input_ids, attention_mask, qtype_onehot, keep: dict | None):
        if tuple(input_ids.shape) != (1, self.seq_len):
            raise ValueError(f"Expected input_ids [1,{self.seq_len}], got {tuple(input_ids.shape)}")
        key_mask, local_mask = self._masks(attention_mask)
        embeddings = self.encoder.embeddings
        rows = F.embedding(input_ids, embeddings.tok_embeddings.weight).to(torch.float32)
        h = self._layer_norm(embeddings.norm, rows)
        if keep is not None:
            keep["embeddings"] = h
        for index in range(LAYERS):
            h = self._encoder_layer(index, h, key_mask, local_mask)
            if keep is not None:
                keep[f"layer_{index:02d}"] = h
        h = self._layer_norm(self.encoder.final_norm, h)
        if keep is not None:
            keep["final_norm"] = h
        if self.mutation != "no_type_emb":
            table = self.type_emb.weight.to(torch.float32)
            h = h + torch.matmul(qtype_onehot.to(torch.float32), table).reshape(1, 1, WIDTH)
        if keep is not None:
            keep["head_input"] = h
        for index in range(2):
            h = self._head_layer(index, h, key_mask)
            if keep is not None:
                keep[f"head_{index}"] = h
        return self._scorer(h)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, qtype_onehot: torch.Tensor):
        return self._run(input_ids, attention_mask, qtype_onehot, None)

    def forward_intermediates(self, input_ids, attention_mask, qtype_onehot) -> dict[str, torch.Tensor]:
        """Every hidden state the oracle saves, under the oracle's names (diagnostic eager path)."""
        keep: dict[str, torch.Tensor] = {}
        keep["token_logits"] = self._run(input_ids, attention_mask, qtype_onehot, keep)
        return keep


class JuliaDecision(_MainBody):
    """The whole checkpoint. Parameter names match model.safetensors exactly (strict load); `act_head`
    and `temperature` are loaded and never exported (the publisher's API reads neither)."""

    def __init__(self, seq_len: int, mutation: str = "none"):
        super().__init__(seq_len, mutation)
        self.act_head = nn.Sequential(nn.Linear(WIDTH + 4, 256), nn.GELU(), nn.Linear(256, 2))
        self.register_buffer("temperature", torch.ones(3))
        self.precision = "fp32"

    def set_precision(self, precision: str) -> "JuliaDecision":
        """fp32, or wfp16: every weight except the LayerNorm parameters stored in fp16 (rounded from the
        F32 checkpoint) and cast to fp32 right before its use."""
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        self.precision = precision
        storage = torch.float32 if precision == "fp32" else torch.float16
        for module in (self.encoder, self.type_emb, self.head, self.scorer):
            for sub in module.modules():
                if isinstance(sub, nn.LayerNorm):
                    continue
                for param in sub.parameters(recurse=False):
                    param.data = param.data.to(storage)
        return self


class MainGraph(_MainBody):
    """The exported `main` function: shares the modules it reads and nothing else."""

    def __init__(self, model: JuliaDecision):
        super().__init__(model.seq_len, model.mutation, shared=model)


def load_julia(source_dir: str | Path, seq_len: int, precision: str = "fp32", mutation: str = "none") -> JuliaDecision:
    """Strict load of every tensor in the checkpoint; no network, no HF model object."""
    from safetensors.torch import load_file

    source = Path(source_dir)
    validate_config(json.loads((source / "encoder" / "config.json").read_text()),
                    json.loads((source / "julia_config.json").read_text()))
    if seq_len not in (512, 1024):
        raise ValueError("windows 512 and 1024 only (the oracle's windows)")
    model = JuliaDecision(seq_len, mutation=mutation)
    weights = load_file(str(source / "model.safetensors"), device="cpu")
    count = sum(value.numel() for value in weights.values())
    if count != PARAMETER_COUNT:
        raise ValueError(f"Checkpoint parameter count {count} != {PARAMETER_COUNT}")
    if weights["temperature"].tolist() != [1.0, 1.0, 1.0]:
        raise ValueError(f"temperature buffer {weights['temperature'].tolist()} is not [1, 1, 1]")
    model.load_state_dict({k: v.float() for k, v in weights.items()}, strict=True)
    model.eval().requires_grad_(False)
    return model.set_precision(precision)
