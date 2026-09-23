"""Static, plain-PyTorch authoring graph for laya multilingual (mmBERT-base encoder + typed decision head).

Re-authored from the raw checkpoint (model.safetensors, encoder/config.json, rl_agent_config.json) and
gated against the publisher's own model (oracle_laya.py). It neither wraps nor imports transformers
or laya. One call is one question row, batch 1, static sequence length S:

  main  input_ids [1,S] int32 (right-padded with PAD 0), attention_mask [1,S] int32 (1 real / 0 pad),
        qtype_onehot [1,3] float32 (choice / score / noul)
        -> token_logits [1,S] float32 (the scorer at every position), pooled_cls [1,768] float32
  act   pooled_cls [1,768] float32, feats [1,4] float32 -> act_logits [1,2] float32

Marker gathering, the act features, temperature and softmax are host work (_laya_host.py).

What the checkpoint runs (laya 0.3.4 common.py DecisionModel over transformers 5.x ModernBERT):
  - encoder: token embedding + biasless LayerNorm; 22 pre-norm layers (layer 0 has no attention norm),
    GLU MLP `Wo(gelu(x) * gate)` with exact GELU; global attention on layers 0, 3, …, 21, sliding
    attention elsewhere with the inclusive radius |i - j| <= 64 (local_attention 128 / 2); RoPE theta
    160000 for both layer kinds, half-split rotation, positions arange(S), trig in fp32; final_norm
  - `h + type_emb[qtype]`, then two post-type `nn.TransformerEncoderLayer(768, 12, 3072, relu,
    norm_first=True)` with the key padding mask, re-implemented explicitly here (PyTorch's fused
    eval fast path is not a graph)
  - scorer LayerNorm -> Linear -> GELU -> Linear(768, 1) at every position; pooled_cls = h[:, 0]
  - act head Linear(772, 256) -> GELU -> Linear(256, 2) on cat(pooled_cls, feats)

Masks are additive, float arithmetic only (no integer or boolean comparison chains in the graph), with
MASK_VALUE = -1e4: masked keys get exactly zero weight in fp32 and fp16 (exp(-1e4) underflows), and a
key-padding mask plus the band mask (-2e4) stays finite in fp16, so a fully masked query row (a pad
position more than 64 past the last real token) is finite instead of NaN. The official SDPA path
returns zero for those rows instead; pad positions are never read by any output (every attention masks
pad keys; markers and CLS are real tokens), so the two agree at every real position.

Precisions (`set_precision`): fp32; fp16 = weights and activations in fp16 except named fp32 islands
(the recipe keeps RoPE application and softmax in fp32); wfp16 = fp16 weight storage with fp32 compute.
The act head is fp32 in all three.

Mutations change one behaviour at a time and are negative controls, never export variants.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

MODEL_ID = "convaiinnovations/laya"
MODEL_SHA = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
PARAMETER_COUNT = 321_908_998
MUTATIONS = ("none", "all_global", "all_local", "window63", "ignore_padding", "no_type_emb")
MASK_VALUE = -1.0e4
FP32_ISLANDS = ("rope", "softmax", "layernorm", "residual", "scorer")
# The recipe for --dtype fp16 (supervisor 2026-09-23, §0.2 item 12): model.half() with fp32 graph I/O,
# RoPE applied in fp32, attention softmax in fp32. The act head is a separate function and stays fp32.
FP16_RECIPE = ("rope", "softmax")
# fp32: fp32 weights and compute. fp16: fp16 weights and compute except the fp32 islands.
# wfp16: fp16 weight storage (linear weights, biases, embedding tables), fp32 compute — every weight is
# cast to fp32 in the graph right before its use (the checkpoint is F16, so the stored values are exact).
PRECISIONS = ("fp32", "fp16", "wfp16")


def validate_config(encoder: dict[str, Any], agent: dict[str, Any]) -> None:
    """Reject any checkpoint outside this deliberately narrow recipe."""
    required = {
        "model_type": "modernbert", "hidden_size": 768, "intermediate_size": 1152, "num_attention_heads": 12,
        "num_hidden_layers": 22, "vocab_size": 256000, "pad_token_id": 0, "hidden_activation": "gelu",
        "attention_bias": False, "mlp_bias": False, "norm_bias": False, "attention_dropout": 0.0,
        "embedding_dropout": 0.0, "mlp_dropout": 0.0, "global_attn_every_n_layers": 3, "local_attention": 128,
        "norm_eps": 1e-5, "position_embedding_type": "sans_pos",
    }
    mismatches = {k: (encoder.get(k), v) for k, v in required.items() if encoder.get(k) != v}
    expected_types = ["full_attention" if i % 3 == 0 else "sliding_attention" for i in range(22)]
    if encoder.get("layer_types") != expected_types:
        mismatches["layer_types"] = (encoder.get("layer_types"), expected_types)
    rope = {kind: (p.get("rope_theta"), p.get("rope_type")) for kind, p in (encoder.get("rope_parameters") or {}).items()}
    if rope != {"full_attention": (160000, "default"), "sliding_attention": (160000, "default")}:
        mismatches["rope_parameters"] = (rope, "theta 160000, default, both kinds")
    agent_required = {"head_layers": 2, "act_costs": {"escalate": 0.5}, "max_len": 1024, "head_max_len": 256}
    mismatches.update({f"agent.{k}": (agent.get(k), v) for k, v in agent_required.items() if agent.get(k) != v})
    if mismatches:
        raise ValueError(f"Unsupported laya configuration (actual, expected): {mismatches}")


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class _Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(256000, 768, padding_idx=0)
        self.norm = nn.LayerNorm(768, eps=1e-5, bias=False)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wqkv = nn.Linear(768, 3 * 768, bias=False)
        self.Wo = nn.Linear(768, 768, bias=False)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wi = nn.Linear(768, 2 * 1152, bias=False)
        self.Wo = nn.Linear(1152, 768, bias=False)


class _EncoderLayer(nn.Module):
    def __init__(self, layer_id: int):
        super().__init__()
        # The checkpoint has no attention norm on layer 0 (the embedding LayerNorm serves).
        self.attn_norm = nn.Identity() if layer_id == 0 else nn.LayerNorm(768, eps=1e-5, bias=False)
        self.attn = _Attention()
        self.mlp_norm = nn.LayerNorm(768, eps=1e-5, bias=False)
        self.mlp = _MLP()


class _Encoder(nn.Module):
    def __init__(self, seq_len: int, mutation: str):
        super().__init__()
        self.embeddings = _Embeddings()
        self.layers = nn.ModuleList([_EncoderLayer(i) for i in range(22)])
        self.final_norm = nn.LayerNorm(768, eps=1e-5, bias=False)
        # RoPE exactly as transformers 5.x computes it (fp32 inv_freq, fp32 outer product, cat, cos/sin),
        # precomputed so no trig of large constant arguments is left for the converter to fold.
        inv_freq = 1.0 / (160000 ** (torch.arange(0, 64, 2, dtype=torch.float) / 64))
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
                      for i in range(22)]


class _HeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_weight = nn.Parameter(torch.empty(3 * 768, 768))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * 768))
        self.out_proj = nn.Linear(768, 768)


class _HeadLayer(nn.Module):
    """nn.TransformerEncoderLayer(768, 12, 3072, relu, norm_first=True, batch_first=True), eval, explicit."""

    def __init__(self):
        super().__init__()
        self.self_attn = _HeadAttention()
        self.linear1 = nn.Linear(768, 3072)
        self.linear2 = nn.Linear(3072, 768)
        self.norm1 = nn.LayerNorm(768, eps=1e-5)
        self.norm2 = nn.LayerNorm(768, eps=1e-5)


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
            self.type_emb = nn.Embedding(3, 768)
            self.head = _Head()
            self.scorer = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 768), nn.GELU(), nn.Linear(768, 1))
            self.dtype, self.fp32_islands = torch.float32, frozenset(FP32_ISLANDS)
        else:
            self.encoder, self.type_emb, self.head, self.scorer = shared.encoder, shared.type_emb, shared.head, shared.scorer
            self.dtype, self.fp32_islands = shared.dtype, shared.fp32_islands

    @staticmethod
    def _linear(module: nn.Module | None, x: torch.Tensor, weight=None, bias=None) -> torch.Tensor:
        """F.linear in the input's dtype. A stored fp16 weight under fp32 compute (wfp16) becomes a cast
        in the graph; coreai-torch 0.4.1 keeps the constant fp16 (checked: bundle size = fp16 bytes)."""
        if module is not None:
            weight, bias = module.weight, module.bias
        return F.linear(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))

    def _island(self, name: str) -> torch.dtype:
        return torch.float32 if name in self.fp32_islands else self.dtype

    def _layer_norm(self, norm: nn.Module, x: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        out_dtype = out_dtype or self.dtype
        if isinstance(norm, nn.Identity):
            return x.to(out_dtype)
        dtype = torch.float32 if ("layernorm" in self.fp32_islands or out_dtype == torch.float32) else self.dtype
        bias = norm.bias.to(dtype) if norm.bias is not None else None
        return F.layer_norm(x.to(dtype), norm.normalized_shape, norm.weight.to(dtype), bias, norm.eps).to(out_dtype)

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """q, k, v [1,12,S,64]; mask [1,1,*,S] fp32 additive. The scale 64**-0.5 = 0.125 is exact."""
        softmax_dtype = self._island("softmax")
        scores = torch.matmul(q, k.transpose(2, 3)).to(softmax_dtype) * 0.125 + mask.to(softmax_dtype)
        probabilities = F.softmax(scores, dim=-1).to(v.dtype)
        return torch.matmul(probabilities, v).transpose(1, 2).reshape(1, self.seq_len, 768)

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
        q, k, v = self._linear(layer.attn.Wqkv, x).reshape(1, self.seq_len, 3, 12, 64).unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        rope = self._island("rope")
        cos, sin = self.encoder.rope_cos.to(rope), self.encoder.rope_sin.to(rope)
        qr, kr = q.to(rope), k.to(rope)
        q = (qr * cos + _rotate_half(qr) * sin).to(self.dtype)
        k = (kr * cos + _rotate_half(kr) * sin).to(self.dtype)
        attended = self._attend(q, k, v, local_mask if self.encoder.local[index] else key_mask)
        h = h + self._linear(layer.attn.Wo, attended).to(h.dtype)
        inputs, gate = self._linear(layer.mlp.Wi, self._layer_norm(layer.mlp_norm, h)).chunk(2, dim=-1)
        return h + self._linear(layer.mlp.Wo, F.gelu(inputs) * gate).to(h.dtype)

    def _head_layer(self, index: int, h: torch.Tensor, key_mask) -> torch.Tensor:
        layer = self.head.layers[index]
        x = self._layer_norm(layer.norm1, h)
        qkv = self._linear(None, x, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
        q, k, v = qkv.reshape(1, self.seq_len, 3, 12, 64).unbind(dim=2)
        attended = self._attend(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), key_mask)
        h = h + self._linear(layer.self_attn.out_proj, attended).to(h.dtype)
        x = F.relu(self._linear(layer.linear1, self._layer_norm(layer.norm2, h)))
        return h + self._linear(layer.linear2, x).to(h.dtype)

    def _scorer(self, h: torch.Tensor) -> torch.Tensor:
        dtype = self._island("scorer")
        norm, first, _, last = self.scorer
        x = F.gelu(self._linear(first, self._layer_norm(norm, h, out_dtype=dtype)))
        return self._linear(last, x).reshape(1, self.seq_len).to(torch.float32)

    def _run(self, input_ids, attention_mask, qtype_onehot, keep: dict | None):
        if tuple(input_ids.shape) != (1, self.seq_len):
            raise ValueError(f"Expected input_ids [1,{self.seq_len}], got {tuple(input_ids.shape)}")
        key_mask, local_mask = self._masks(attention_mask)
        residual = self._island("residual")
        embeddings = self.encoder.embeddings
        rows = F.embedding(input_ids, embeddings.tok_embeddings.weight).to(self.dtype)
        h = self._layer_norm(embeddings.norm, rows, out_dtype=residual)
        if keep is not None:
            keep["embeddings"] = h
        for index in range(22):
            h = self._encoder_layer(index, h, key_mask, local_mask)
            if keep is not None:
                keep[f"layer_{index:02d}"] = h
        h = self._layer_norm(self.encoder.final_norm, h, out_dtype=residual)
        if keep is not None:
            keep["final_norm"] = h
        if self.mutation != "no_type_emb":
            table = self.type_emb.weight
            h = h + torch.matmul(qtype_onehot.to(table.dtype), table).reshape(1, 1, 768).to(residual)
        if keep is not None:
            keep["head_input"] = h
        for index in range(2):
            h = self._head_layer(index, h, key_mask)
            if keep is not None:
                keep[f"head_{index}"] = h
        return self._scorer(h), h[:, 0].to(torch.float32)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, qtype_onehot: torch.Tensor):
        return self._run(input_ids, attention_mask, qtype_onehot, None)

    def forward_intermediates(self, input_ids, attention_mask, qtype_onehot) -> dict[str, torch.Tensor]:
        """Every hidden state the oracle saves, under the oracle's names (diagnostic eager path)."""
        keep: dict[str, torch.Tensor] = {}
        token_logits, pooled_cls = self._run(input_ids, attention_mask, qtype_onehot, keep)
        keep.update(token_logits=token_logits, pooled_cls=pooled_cls)
        return keep


class LayaDecision(_MainBody):
    """The whole checkpoint. Parameter names match model.safetensors exactly (strict load);
    `temperature` is the buffer the checkpoint ships and the API never reads."""

    def __init__(self, seq_len: int, mutation: str = "none"):
        super().__init__(seq_len, mutation)
        self.act_head = nn.Sequential(nn.Linear(772, 256), nn.GELU(), nn.Linear(256, 2))
        self.register_buffer("temperature", torch.ones(3))
        self.precision = "fp32"

    def set_precision(self, precision: str, fp32_islands=FP16_RECIPE) -> "LayaDecision":
        """See PRECISIONS. LayerNorm parameters stay fp32 tensors in every mode (cast in the graph when
        LayerNorm is not an fp32 island); the act head always stays fp32. The fp16 weights are exact: the
        checkpoint stores every tensor in F16."""
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        unknown = set(fp32_islands) - set(FP32_ISLANDS)
        if unknown:
            raise ValueError(f"unknown fp32 islands {sorted(unknown)}")
        self.precision = precision
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self.fp32_islands = frozenset(fp32_islands) if precision == "fp16" else frozenset(FP32_ISLANDS)
        storage = torch.float32 if precision == "fp32" else torch.float16
        for module in (self.encoder, self.type_emb, self.head, self.scorer):
            keep_fp32 = precision == "fp16" and module is self.scorer and "scorer" in self.fp32_islands
            for sub in module.modules():
                if isinstance(sub, nn.LayerNorm):
                    continue
                for param in sub.parameters(recurse=False):
                    param.data = param.data.to(torch.float32 if keep_fp32 else storage)
        return self

    def act(self, pooled_cls: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        return self.act_head(torch.cat((pooled_cls.float(), feats.float()), dim=-1))


class MainGraph(_MainBody):
    """The exported `main` function: shares the modules it reads and nothing else."""

    def __init__(self, model: LayaDecision):
        super().__init__(model.seq_len, model.mutation, shared=model)


class ActGraph(nn.Module):
    """The exported `act` function: the fp32 act head alone."""

    def __init__(self, model: LayaDecision):
        super().__init__()
        self.act_head = model.act_head

    def forward(self, pooled_cls, feats):
        return self.act_head(torch.cat((pooled_cls, feats), dim=-1))


def load_laya(source_dir: str | Path, seq_len: int, precision: str = "fp32", mutation: str = "none",
              fp32_islands=FP16_RECIPE) -> LayaDecision:
    """Strict load of every tensor in the multilingual checkpoint; no network, no HF model object."""
    from safetensors.torch import load_file

    source = Path(source_dir)
    validate_config(json.loads((source / "encoder" / "config.json").read_text()),
                    json.loads((source / "rl_agent_config.json").read_text()))
    if seq_len not in (256, 512):
        raise ValueError("windows 256 and 512 only (the fixtures' windows)")
    model = LayaDecision(seq_len, mutation=mutation)
    weights = load_file(str(source / "model.safetensors"), device="cpu")
    count = sum(value.numel() for value in weights.values())
    if count != PARAMETER_COUNT:
        raise ValueError(f"Checkpoint parameter count {count} != {PARAMETER_COUNT}")
    model.load_state_dict({k: v.float() for k, v in weights.items()}, strict=True)
    model.eval().requires_grad_(False)
    return model.set_precision(precision, fp32_islands)
