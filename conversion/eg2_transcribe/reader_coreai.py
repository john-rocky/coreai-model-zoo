#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.9.0",
#     "numpy>=2.2",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
# Community port — NOT an Apple or Google model.
"""The EG2 transcribe reader + its word table as one static Core AI graph, and the host side of the read-out.

The reader (``reader/train/model.py`` of the eg2-transcribe repo, ``Reader(arch, n_vocab, k, layers)``) maps the
audio soft tokens of EmbeddingGemma 2 to one unit vector per 40 ms token plus a blank logit; each token reads the
word whose table row is closest (scale * cosine) or blank, and greedy CTC (merge repeats, drop blanks) gives the
words. The body's depth comes from ``reader_config.json`` (``architecture.layers``: reader v1 has 4, v2 has 6; a
config without the field means 4). ``ReaderGraph`` wraps a trained reader for export, one static graph per window of
T tokens:

    soft_tokens [1, T, 512] + token_mask [1, T]  ->  top1_index  [1, T] int32   argmax over the N words
                                                      top1_logit  [1, T]         scale * the largest cosine
                                                      blank_logit [1, T]
    head "h" (the fallback form h-out)            ->  h [1, T, 768] (L2-normalized), blank_logit [1, T]

token_mask is 1 on the valid tokens and 0 on padding (the meaning of ``model.key_mask(lengths, T)``); padded keys
get -1e4 before the softmax. The table [N, 768] (``word_table.npy``, float16, re-normalized in float32 the way
``decode.load_table`` does) is a constant of the graph; the scale is the trained ``exp(log_scale)``, applied
after the max as ``greedy_labels`` does. The rotary cos / sin of positions 0..T-1 are constants. The host turns
the outputs into labels with ``host_labels``: blank where blank_logit > top1_logit, else top1_index (the strict
comparison of ``model.greedy_labels``: a tie reads the word); ``model.collapse`` gives the words.

Precision: float32, or float16 throughout (inputs, constants, arithmetic, outputs; ``half()``). In float16
every LayerNorm / RMSNorm and the final L2 normalization divide the row by its max |x| up front (exact in real
arithmetic; no square can overflow), and the LayerNorm is written out (the converter's layer_norm takes the
statistics of a float16 input in float32, which would put float32 ops into the float16 graph).

Utterances longer than T tokens are cut into consecutive pieces of at most T tokens (``pieces``); each piece is
its own window (the reader attends within the window only), the labels of the pieces are concatenated and then
collapsed. With T = 750 every gate utterance fits in one piece.

Locations (argument, then environment, then default):
    EG2_READER_DIR  the reader directory of eg2-transcribe (default $ZOO_CODE_ROOT/eg2-transcribe/reader):
                    vocab.txt, word_table.npy, train/model.py + decode.py, and after the release reader.pt +
                    reader_config.json. A checkpoint that is not the release one goes in with --checkpoint
                    (its arch from --arch, else from the metrics.json beside it).
    EG2STT_CACHE    soft token cache (default <work>/cache): test-clean/index.tsv + <id>.npy (float16)

CLI (CPU torch, no Core AI runtime):
    parity   ReaderGraph in float32 eager vs the reader's own forward + greedy_labels + collapse on the same padded
             inputs (fixture 20 + the 200 leading test-clean utterances): label agreement per token and per
             utterance -> <out>/reader_wrapper_parity.json; also writes the reference outputs ref_T<T>.npz that
             gate_reader.py compares the bundles with
    absmax   float16 headroom: the largest magnitudes inside the reader on the same inputs -> <out>/absmax_reader.json
    info     arch, parameters, scale, table, contract

    <venv>/bin/python conversion/eg2_transcribe/reader_coreai.py parity --checkpoint <run>/best_model.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import exports_dir, fixtures_dir, sha256_of, work_dir, write_json  # noqa: E402
from _paths import code_path  # noqa: E402

D_IN, D_OUT = 512, 768
MASK_BIAS = 1.0e4         # added as -1e4 on padded keys: finite in float16, exp() of it is 0 in both precisions
NORM_FLOOR = 1.0e-3       # max-abs floor of the float16 norms (rows of zeros stay zero)
TOKEN_BUCKETS = (750, 250)
WINDOW_SECONDS = {750: 30.0, 250: 10.0}   # 25 tokens per second: the encoder's 3000 / 1000 frame windows
N_TEST = 200


# ---------------------------------------------------------------- locations and the reader repo

def reader_dir(arg: str | None = None) -> Path:
    if arg:
        return Path(arg).expanduser()
    v = os.environ.get("EG2_READER_DIR")
    return Path(v).expanduser() if v else code_path("eg2-transcribe", "reader")


def cache_dir(arg: str | None = None) -> Path:
    if arg:
        return Path(arg).expanduser()
    v = os.environ.get("EG2STT_CACHE")
    return Path(v).expanduser() if v else work_dir() / "cache"


def out_dir(arg: str | None = None, tag: str = "") -> Path:
    """Unless given: <work>/exports/reader for reader v1 and a --checkpoint, <work>/exports/reader_<version> for a later
    release (reader_v2: the folder strip_ship.py reads reader v2 from)."""
    if arg:
        return Path(arg).expanduser()
    return exports_dir() / ("reader" + (f"_{tag}" if tag else ""))


def repo_modules(rdir: Path):
    """model.py and decode.py of the reader repo, imported without writing bytecode next to them."""
    sys.dont_write_bytecode = True
    train = str(rdir / "train")
    if train not in sys.path:
        sys.path.insert(0, train)
    import decode  # noqa: E402
    import model  # noqa: E402
    return model, decode


def add_reader_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--reader-dir", help="default: $EG2_READER_DIR or <code>/eg2-transcribe/reader")
    ap.add_argument("--checkpoint", help="a state_dict to use instead of <reader-dir>/reader.pt (e.g. a run's "
                                         "best_model.pt); vocab and table still come from the reader dir")
    ap.add_argument("--arch", choices=("a", "b", "c"), help="with --checkpoint: default from its metrics.json")
    ap.add_argument("--k", type=int, help="arch b: text layers kept (default from metrics.json, else 4)")
    ap.add_argument("--layers", type=int, help="arch a with --checkpoint: transformer layers (default from its "
                                               "metrics.json, else 4)")


def load_reader(args) -> dict:
    """-> model (eval, float32, CPU), table [N, 768] float32 unit rows, vocab, arch, k, layers, scale, source record.

    The architecture (name, k, layers) comes from reader_config.json for the release reader, from --arch / --k /
    --layers or the metrics.json beside a --checkpoint otherwise; a missing ``layers`` means 4 (reader v1)."""
    rdir = reader_dir(args.reader_dir)
    M, Dc = repo_modules(rdir)
    with open(rdir / "vocab.txt", encoding="utf-8") as f:
        vocab = [line.rstrip("\n") for line in f if line.strip()]
    if args.checkpoint:
        ck = Path(args.checkpoint).expanduser().resolve()
        arch, k, layers = args.arch, args.k, getattr(args, "layers", None)
        metrics = ck.parent / "metrics.json"
        if arch is None:
            met = json.loads(metrics.read_text())
            arch, k = met["arch"]["name"], k or met["arch"].get("k")
            layers = layers or met["arch"].get("layers")
        kind, table_path, cfg = "checkpoint", rdir / "word_table.npy", None
    else:
        cfg = json.loads((rdir / "reader_config.json").read_text())
        arch, k = cfg["architecture"]["name"], args.k or cfg["architecture"].get("k")
        layers = cfg["architecture"].get("layers")
        kind, ck, table_path = "release", rdir / cfg["weights"]["file"], rdir / cfg["table"]["file"]
    if arch == "a":
        layers = int(layers or 4)
    else:
        layers = None
    sd = torch.load(ck, map_location="cpu", weights_only=True)
    if isinstance(sd.get("model"), dict):          # a ckpt_epoch*.pt holds the state_dict under "model"
        sd = sd["model"]
    model = M.Reader(arch, len(vocab), k=k or 4, layers=layers or 4)
    model.load_state_dict(sd)
    model.eval()
    table = Dc.load_table(str(table_path), "cpu")
    if tuple(table.shape) != (len(vocab), D_OUT):
        raise SystemExit(f"table {tuple(table.shape)} does not match {len(vocab)} words")
    source = {"kind": kind, "weights": str(ck), "weights_sha256": sha256_of(ck), "arch": arch, "k": k,
              "layers": layers, "params": sum(p.numel() for p in model.parameters()),
              "scale": float(model.scale.detach()), "n_vocab": len(vocab), "table": str(table_path),
              "reader_dir": str(rdir), "model_py_sha256": sha256_of(rdir / "train" / "model.py"),
              "decode_py_sha256": sha256_of(rdir / "train" / "decode.py")}
    if cfg is not None:
        source["config"] = {key: cfg.get(key) for key in ("version", "window", "decode", "scale", "wer")}
    return {"model": model, "table": table, "vocab": vocab, "arch": arch, "k": k, "layers": layers,
            "scale": float(model.scale.detach()), "source": source}


def name_tag(R: dict) -> str:
    """The version part of a bundle name: "" for reader v1 and for a --checkpoint (``eg2_reader_fp16_T750``), else
    the release's ``version`` (``eg2_reader_v2_fp16_T750``), so the bundles of two readers never share a name."""
    version = str((R["source"].get("config") or {}).get("version") or "")
    return version if version and version != "v1" else ""


def config_tag(args) -> str:
    """name_tag() from reader_config.json alone (no weights loaded): what a bundle of these arguments is named."""
    if getattr(args, "checkpoint", None):
        return ""
    cfg = json.loads((reader_dir(args.reader_dir) / "reader_config.json").read_text())
    return name_tag({"source": {"config": {"version": cfg.get("version")}}})


def bundle_stem(form: str, tokens: int, tag: str = "") -> str:
    """eg2_reader_[<tag>_]<form>_T<tokens>: the .aimodel's name without its extension."""
    return f"eg2_reader_{tag + '_' if tag else ''}{form}_T{tokens}"


# ---------------------------------------------------------------- the export graph

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


class ReaderGraph(nn.Module):
    """A trained ``Reader`` + the word table for one window of ``tokens`` tokens (see the module docstring).

    The parameters are the reader's own modules; only the forward is rewritten: named mask input, rotary
    constants, attention through scaled_dot_product_attention with an additive mask, written-out norms in float16
    (``maxabs=True``), and the table read-out. ``probe`` (a dict) records the largest magnitudes (absmax)."""

    def __init__(self, reader: nn.Module, table: torch.Tensor, tokens: int, head: str = "argmax",
                 maxabs: bool = False):
        super().__init__()
        M = sys.modules["model"]
        self.arch, self.body, self.blank = reader.arch, reader.body, reader.blank
        self.head, self.maxabs, self.tokens = head, maxabs, tokens
        self.register_buffer("scale", reader.scale.detach().clone().reshape(()))
        if self.arch in ("a", "b"):
            dim = (self.body.inp.out_features // self.body.heads) if self.arch == "a" else self.body.head_dim
            cos, sin = M.rope(tokens, dim, self.body.rope_base, torch.device("cpu"))
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)
        if head == "argmax":
            self.register_buffer("table_t", table.t().contiguous())   # [768, N]
        self.probe: dict | None = None

    def _p(self, name: str, x: torch.Tensor) -> None:
        if self.probe is not None:
            v = float(x.detach().abs().amax())
            self.probe[name] = max(self.probe.get(name, 0.0), v)

    # norms ------------------------------------------------------------------------------------------------
    def _inv_max(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.rsqrt(torch.clamp(x.abs().amax(-1, keepdim=True), min=NORM_FLOOR))
        return r * r                                       # 1 / max|x| without a divide op

    def _ln(self, mod: nn.LayerNorm, x: torch.Tensor, tag: str) -> torch.Tensor:
        self._p(f"{tag}.in", x)
        if not self.maxabs:
            return F.layer_norm(x, mod.normalized_shape, mod.weight, mod.bias, mod.eps)
        c = x - x.mean(-1, keepdim=True)
        self._p(f"{tag}.centered", c)
        inv_m = self._inv_max(c)
        u = c * inv_m
        e = inv_m * math.sqrt(mod.eps)                     # its square is eps / max^2
        return u * torch.rsqrt((u * u).mean(-1, keepdim=True) + e * e) * mod.weight + mod.bias

    def _rms(self, mod: nn.Module, x: torch.Tensor, tag: str) -> torch.Tensor:
        self._p(f"{tag}.in", x)
        if not self.maxabs:
            return mod(x)                                  # model.py RMSNorm: float32 statistics
        inv_m = self._inv_max(x)
        u = x * inv_m
        e = inv_m * math.sqrt(mod.eps)
        y = u * torch.rsqrt((u * u).mean(-1, keepdim=True) + e * e)
        return y * mod.weight if mod.weight is not None else y

    def _l2(self, v: torch.Tensor) -> torch.Tensor:
        if not self.maxabs:
            return F.normalize(v, dim=-1)
        u = v * self._inv_max(v)
        return u * torch.rsqrt((u * u).sum(-1, keepdim=True))

    def _rotary(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.cos + rotate_half(x) * self.sin

    # bodies -----------------------------------------------------------------------------------------------
    def _body_a(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        b = self.body
        h = b.inp(self._ln(b.ln_in, x, "ln_in"))
        for i, blk in enumerate(b.blocks):
            B, T, D = h.shape
            H = blk.heads
            qkv = blk.qkv(self._ln(blk.ln1, h, f"b{i}.ln1"))
            self._p(f"b{i}.qkv", qkv)
            q = qkv[..., :D].reshape(B, T, H, D // H).transpose(1, 2)
            k = qkv[..., D:2 * D].reshape(B, T, H, D // H).transpose(1, 2)
            v = qkv[..., 2 * D:].reshape(B, T, H, D // H).transpose(1, 2)
            q, k = self._rotary(q), self._rotary(k)
            if self.probe is not None:
                self._p(f"b{i}.scores", (q @ k.transpose(-1, -2)) / math.sqrt(D // H))
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            h = h + blk.proj(a.transpose(1, 2).reshape(B, T, D))
            f = blk.ff1(self._ln(blk.ln2, h, f"b{i}.ln2"))
            self._p(f"b{i}.ff1", f)
            h = h + blk.ff2(F.gelu(f))
            self._p(f"b{i}.residual", h)
        return b.out(self._ln(b.ln_out, h, "ln_out"))

    def _body_b(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        M = sys.modules["model"]
        b = self.body
        B, T, _ = x.shape
        ple = b.per_layer_model_projection(x) * (b.d ** -0.5)
        ple = self._rms(b.per_layer_projection_norm, ple.reshape(B, T, b.k, b.d_ple), "ple_norm")
        h = x
        for i, L in enumerate(b.layers):
            g = self._rms(L.input_layernorm, h, f"l{i}.input")
            q = self._rms(L.q_norm, L.q_proj(g).reshape(B, T, L.heads, L.head_dim), f"l{i}.q").transpose(1, 2)
            k = self._rms(L.k_norm, L.k_proj(g).reshape(B, T, L.kv_heads, L.head_dim), f"l{i}.k").transpose(1, 2)
            v = self._rms(L.v_norm, L.v_proj(g).reshape(B, T, L.kv_heads, L.head_dim), f"l{i}.v").transpose(1, 2)
            q, k = self._rotary(q), self._rotary(k)
            rep = L.heads // L.kv_heads
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=1.0)
            a = L.o_proj(a.transpose(1, 2).reshape(B, T, -1))
            h = h + self._rms(L.post_attention_layernorm, a, f"l{i}.post_attn")
            g = self._rms(L.pre_feedforward_layernorm, h, f"l{i}.pre_ffn")
            g = L.down_proj(M.gelu_tanh(L.gate_proj(g)) * L.up_proj(g))
            h = h + self._rms(L.post_feedforward_layernorm, g, f"l{i}.post_ffn")
            g = L.per_layer_projection(M.gelu_tanh(L.per_layer_input_gate(h)) * ple[:, :, i])
            h = (h + self._rms(L.post_per_layer_input_norm, g, f"l{i}.post_ple")) * L.layer_scalar
            self._p(f"l{i}.residual", h)
        return b.embedding_projection(self._rms(b.norm, h, "norm"))

    def _body_c(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        ln, fc1, _, fc2 = self.body.net
        f = fc1(self._ln(ln, x, "ln"))
        self._p("fc1", f)
        return fc2(F.gelu(f))

    def forward(self, soft_tokens: torch.Tensor, token_mask: torch.Tensor):
        self._p("soft_tokens", soft_tokens)
        bias = ((token_mask - 1.0) * MASK_BIAS)[:, None, None, :]           # 0 = attend, -1e4 = padding
        v = {"a": self._body_a, "b": self._body_b, "c": self._body_c}[self.arch](soft_tokens, bias)
        self._p("v", v)
        blank = self.blank(v).squeeze(-1)
        h = self._l2(v)
        if self.head == "h":
            return h, blank
        best, idx = torch.max(h @ self.table_t, dim=-1)
        return idx.to(torch.int32), best * self.scale, blank


def build_graph(R: dict, tokens: int, dtype: torch.dtype, head: str = "argmax") -> ReaderGraph:
    """float32: exact norms; float16: max-abs norms, then every parameter and constant in float16 (on a copy of
    the reader: the graph shares its modules, and half() would otherwise change R["model"] in place)."""
    if dtype != torch.float16:
        return ReaderGraph(R["model"], R["table"], tokens, head=head).eval()
    return ReaderGraph(copy.deepcopy(R["model"]), R["table"], tokens, head=head, maxabs=True).eval().half()


# ---------------------------------------------------------------- host side

def host_labels(top1_index: np.ndarray, top1_logit: np.ndarray, blank_logit: np.ndarray, n_vocab: int) -> np.ndarray:
    """Per token: blank (= n_vocab) where blank_logit > top1_logit, else top1_index (greedy_labels' rule)."""
    lab = np.asarray(top1_index).astype(np.int64)
    lab[np.asarray(blank_logit) > np.asarray(top1_logit)] = n_vocab
    return lab


def host_argmax(h: np.ndarray, table_t: np.ndarray, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """The h-out form's read-out on the host: argmax and max cosine of h [n, 768] against table_t [768, N]."""
    n = h.shape[0]
    idx, best = np.zeros(n, np.int32), np.zeros(n, np.float32)
    for a in range(0, n, chunk):
        c = h[a:a + chunk].astype(np.float32) @ table_t
        i = c.argmax(1)
        idx[a:a + chunk], best[a:a + chunk] = i, c[np.arange(len(i)), i]
    return idx, best


def collapse(labels, blank_id: int) -> list[int]:
    """model.collapse (merge repeats, drop blanks), kept here so the host side runs without the reader repo."""
    out, prev = [], None
    for c in labels:
        if c != blank_id and c != prev:
            out.append(int(c))
        prev = c
    return out


def norm_text(text: str) -> str:
    """eval_wer.norm: lower case, runs of whitespace to one space."""
    return " ".join(text.lower().split())


def word_errors(ref: list[str], hyp: list[str]) -> int:
    """Word-level Levenshtein distance (substitutions + deletions + insertions), one numpy row at a time."""
    ids: dict[str, int] = {}
    r = np.array([ids.setdefault(w, len(ids)) for w in ref], dtype=np.int64)
    h = np.array([ids.setdefault(w, len(ids)) for w in hyp], dtype=np.int64)
    if len(r) == 0 or len(h) == 0:
        return int(max(len(r), len(h)))
    col = np.arange(len(h) + 1, dtype=np.int64)
    prev = col.copy()
    for i in range(1, len(r) + 1):
        cur = np.empty_like(prev)
        cur[0] = i
        cur[1:] = np.minimum(prev[:-1] + (h != r[i - 1]), prev[1:] + 1)
        prev = np.minimum.accumulate(cur - col) + col      # insertions: cur[j] = min_k (cur[k] + j - k)
    return int(prev[-1])


def wer(refs: list[str], hyps: list[str]) -> dict:
    errs = sum(word_errors(r.split(), h.split()) for r, h in zip(refs, hyps))
    n = sum(len(r.split()) for r in refs)
    return {"wer": errs / max(1, n), "errors": errs, "ref_words": n, "utterances": len(refs),
            "engine": "word Levenshtein (numpy); jiwer: gate_reader.py score"}


# ---------------------------------------------------------------- inputs

def load_utterances(fx: Path, cache: Path, n_test: int = N_TEST) -> list[dict]:
    """Fixture 20 (float32 reference soft tokens) then the leading n_test test-clean utterances of the cache
    (index order, float16 -> float32). Each: set, id, x [n, 512] float32, text (normalized)."""
    utts = []
    manifest = json.loads((fx / "MANIFEST.json").read_text())
    for e in manifest["entries"]:
        d = fx / "test-clean-20" / e["id"]
        x = np.load(d / "soft_tokens.npy").astype(np.float32)
        utts.append({"set": "fixture", "id": e["id"], "x": x, "text": norm_text((d / "transcript.txt").read_text())})
    with open(cache / "test-clean" / "index.tsv", encoding="utf-8") as f:
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()][:n_test]
    for uid, _, _, n_tok, text in rows:
        x = np.load(cache / "test-clean" / f"{uid}.npy").astype(np.float32)
        assert x.shape == (int(n_tok), D_IN), (uid, x.shape)
        utts.append({"set": "test-clean", "id": uid, "x": x, "text": norm_text(text)})
    return utts


def pieces(n: int, tokens: int) -> list[tuple[int, int]]:
    """Consecutive cuts [a, b) of an utterance of n tokens into windows of at most `tokens` tokens."""
    return [(a, min(n, a + tokens)) for a in range(0, n, tokens)]


def window(x: np.ndarray, a: int, b: int, tokens: int, dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
    """Rows a..b-1 of x, right-zero-padded to [1, tokens, 512], and token_mask [1, tokens]."""
    xs = np.zeros((1, tokens, x.shape[1]), dtype=dtype)
    xs[0, :b - a] = x[a:b]
    m = np.zeros((1, tokens), dtype=dtype)
    m[0, :b - a] = 1
    return xs, m


# ---------------------------------------------------------------- CLI: parity, absmax, info

@torch.no_grad()
def reference_piece(R: dict, x: np.ndarray, n: int, tokens: int) -> dict:
    """The reader's own path on one padded window: forward(x, lengths) + greedy_labels; plus the top-2 cosines."""
    M = sys.modules["model"]
    model, table = R["model"], R["table"]
    xt = torch.from_numpy(x)
    h, blank = model(xt, torch.tensor([n]))
    h, blank = h[0, :n], blank[0, :n]
    labels = M.greedy_labels(h, blank, table, model.scale, chunk=256)
    top = torch.topk(h @ table.T, 2, dim=1)
    return {"labels": labels.numpy().astype(np.int32), "idx": top.indices[:, 0].numpy().astype(np.int32),
            "top1_logit": (top.values[:, 0] * model.scale).numpy().astype(np.float32),
            "blank_logit": blank.numpy().astype(np.float32),
            "gap": (top.values[:, 0] - top.values[:, 1]).numpy().astype(np.float32),
            "h": h.numpy().astype(np.float32)}


def margin_summary(gap: np.ndarray, dlogit: np.ndarray) -> dict:
    """How close the reference decisions are to a tie: top-1 minus top-2 cosine, |blank - top-1 logit|."""
    q = [0.0, 0.001, 0.01, 0.5]
    return {"top1_minus_top2_cos": {"quantiles": dict(zip(map(str, q), np.quantile(gap, q).round(7).tolist())),
                                    "below_1e-6": int((gap < 1e-6).sum()), "below_2.4e-4": int((gap < 2.4e-4).sum()),
                                    "below_4.9e-4": int((gap < 4.9e-4).sum())},
            "abs_blank_minus_top1_logit": {"quantiles": dict(zip(map(str, q), np.quantile(dlogit, q).round(6).tolist())),
                                           "below_1e-4": int((dlogit < 1e-4).sum()),
                                           "below_0.0156": int((dlogit < 0.0156).sum()),
                                           "below_0.0313": int((dlogit < 0.0313).sum())},
            "tokens": int(gap.size),
            "note": "float16 spacing: cosines in [0.25, 0.5) 2.4e-4, [0.5, 1) 4.9e-4; logits in [16, 32) 0.0156, "
                    "[32, 64) 0.0313"}


@torch.no_grad()
def cmd_parity(args) -> None:
    torch.set_num_threads(args.threads)
    R = load_reader(args)
    utts = load_utterances(fixtures_dir(args.fixtures), cache_dir(args.cache), args.n_test)
    N = len(R["vocab"])
    od = out_dir(args.out_dir, name_tag(R))
    od.mkdir(parents=True, exist_ok=True)
    result = {"reader": R["source"], "inputs": {"fixture": sum(u["set"] == "fixture" for u in utts),
                                                 "test-clean": sum(u["set"] == "test-clean" for u in utts)},
              "rule": "host: blank where blank_logit > top1_logit, else top1_index; collapse = model.collapse",
              "reference": "model.Reader.forward(padded window, lengths) + model.greedy_labels + model.collapse "
                           "(float32, CPU)", "wrapper": "ReaderGraph float32 eager (the export graph)",
              "threads": args.threads}
    tb = R["table"].numpy()
    rows16 = tb.astype(np.float16)
    uniq = np.unique(rows16.view(np.dtype((np.void, rows16.dtype.itemsize * rows16.shape[1]))))
    result["table_duplicate_rows_fp16"] = int(rows16.shape[0] - uniq.shape[0])
    for T in args.tokens:
        t0 = time.time()
        graph = build_graph(R, T, torch.float32)
        ref_store: dict[str, np.ndarray] = {}
        per_utt, mism, d_top, d_blank, gaps, dls, idx_same = [], [], 0.0, 0.0, [], [], 0
        tok_total = tok_same = 0
        for u in utts:
            n = u["x"].shape[0]
            parts = {key: [] for key in ("labels", "idx", "top1_logit", "blank_logit", "gap", "h")}
            w_labels = []
            for a, b in pieces(n, T):
                x, m = window(u["x"], a, b, T)
                ref = reference_piece(R, x, b - a, T)
                for key in parts:
                    parts[key].append(ref[key])
                i, top, bl = graph(torch.from_numpy(x), torch.from_numpy(m))
                i, top, bl = i[0, :b - a].numpy(), top[0, :b - a].numpy(), bl[0, :b - a].numpy()
                w_labels.append(host_labels(i, top, bl, N))
                d_top = max(d_top, float(np.abs(top - ref["top1_logit"]).max()))
                d_blank = max(d_blank, float(np.abs(bl - ref["blank_logit"]).max()))
                idx_same += int((i == ref["idx"]).sum())
            ref_cat = {key: np.concatenate(v) for key, v in parts.items()}
            wl = np.concatenate(w_labels)
            rl = ref_cat["labels"].astype(np.int64)
            same = wl == rl
            tok_total += n
            tok_same += int(same.sum())
            gaps.append(ref_cat["gap"])
            dls.append(np.abs(ref_cat["blank_logit"] - ref_cat["top1_logit"]))
            for p in np.flatnonzero(~same)[:20]:
                mism.append({"set": u["set"], "id": u["id"], "pos": int(p), "ref": int(rl[p]), "wrapper": int(wl[p]),
                             "ref_gap": float(ref_cat["gap"][p]),
                             "ref_blank_minus_top1": float(ref_cat["blank_logit"][p] - ref_cat["top1_logit"][p])})
            r_words, w_words = collapse(rl, N), collapse(wl, N)
            per_utt.append({"set": u["set"], "id": u["id"], "tokens": n, "pieces": len(pieces(n, T)),
                            "tokens_same": int(same.sum()), "transcript_same": r_words == w_words})
            key = f"{u['set']}/{u['id']}"
            for name, arr in ref_cat.items():
                ref_store[f"{key}/{name}"] = arr
        ref_store["meta/weights_sha256"] = np.array(R["source"]["weights_sha256"])
        ref_store["meta/tokens"] = np.array(T)
        np.savez(od / f"ref_T{T}.npz", **ref_store)
        by_set = {}
        for s in ("fixture", "test-clean"):
            rows = [r for r in per_utt if r["set"] == s]
            by_set[s] = {"utterances": len(rows), "transcripts_same": sum(r["transcript_same"] for r in rows),
                         "tokens": sum(r["tokens"] for r in rows), "tokens_same": sum(r["tokens_same"] for r in rows),
                         "token_agreement": sum(r["tokens_same"] for r in rows) / max(1, sum(r["tokens"] for r in rows))}
        result[f"T{T}"] = {
            "tokens_per_window": T, "window_s": WINDOW_SECONDS.get(T), "pieces_total": sum(r["pieces"] for r in per_utt),
            "by_set": by_set, "token_agreement": tok_same / tok_total, "tokens": tok_total,
            "argmax_index_agreement": idx_same / tok_total,
            "max_abs_top1_logit": d_top, "max_abs_blank_logit": d_blank, "mismatches": mism[:50],
            "pass": (by_set["fixture"]["transcripts_same"] == by_set["fixture"]["utterances"]
                     and tok_same == tok_total),
            "reference_margins": margin_summary(np.concatenate(gaps), np.concatenate(dls)),
            "ref_npz": str(od / f"ref_T{T}.npz"), "seconds": round(time.time() - t0, 1),
            "utterances_table": per_utt,
        }
        print(f"T{T}: tokens {tok_same}/{tok_total} same, fixture transcripts "
              f"{by_set['fixture']['transcripts_same']}/{by_set['fixture']['utterances']}, test-clean "
              f"{by_set['test-clean']['transcripts_same']}/{by_set['test-clean']['utterances']}, max|d| top1 "
              f"{d_top:.2e} blank {d_blank:.2e} ({time.time() - t0:.0f} s)", flush=True)
    result["pass"] = all(result[f"T{T}"]["pass"] for T in args.tokens)
    result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    dest = od / "reader_wrapper_parity.json"
    write_json(dest, result)
    print(f"{'PASS' if result['pass'] else 'FAIL'} -> {dest}")


@torch.no_grad()
def cmd_absmax(args) -> None:
    torch.set_num_threads(args.threads)
    R = load_reader(args)
    utts = load_utterances(fixtures_dir(args.fixtures), cache_dir(args.cache), args.n_test)
    T = max(args.tokens)
    graph = build_graph(R, T, torch.float32)
    graph.probe = {}
    vmin_norm, blank_rng, top_rng = float("inf"), [float("inf"), -float("inf")], [float("inf"), -float("inf")]
    for u in utts:
        n = u["x"].shape[0]
        for a, b in pieces(n, T):
            x, m = window(u["x"], a, b, T)
            i, top, bl = graph(torch.from_numpy(x), torch.from_numpy(m))
            top, bl = top[0, :b - a], bl[0, :b - a]
            blank_rng = [min(blank_rng[0], float(bl.min())), max(blank_rng[1], float(bl.max()))]
            top_rng = [min(top_rng[0], float(top.min())), max(top_rng[1], float(top.max()))]
    probe = dict(sorted(graph.probe.items()))
    graph.probe = None
    fp16_max = 65504.0
    sq = {k: v * v for k, v in probe.items() if k.endswith((".in", ".centered"))}
    result = {"reader": R["source"], "tokens": T, "utterances": len(utts), "absmax": probe,
              "blank_logit_range": blank_rng, "top1_logit_range": top_rng,
              "norm_inputs_square_max": max(sq.values()) if sq else None,
              "headroom_x_fp16": {k: round(fp16_max / v, 2) for k, v in probe.items() if v > 0},
              "note": "max |x| over every valid and padded position of the gate windows (float32 eager); a norm "
                      "input's square is what an unscaled float16 norm would form"}
    dest = out_dir(args.out_dir, name_tag(R)) / "absmax_reader.json"
    write_json(dest, result)
    worst = sorted(result["headroom_x_fp16"].items(), key=lambda kv: kv[1])[:6]
    print(f"absmax: largest {max(probe.items(), key=lambda kv: kv[1])}, blank {blank_rng}, top1 {top_rng}, "
          f"lowest fp16 headroom {worst} -> {dest}")


def cmd_info(args) -> None:
    R = load_reader(args)
    for T in args.tokens:
        g = build_graph(R, T, torch.float32)
        x, m = window(np.zeros((T, D_IN), np.float32), 0, T, T)
        with torch.no_grad():
            i, top, bl = g(torch.from_numpy(x), torch.from_numpy(m))
        print(f"T{T}: soft_tokens {list(x.shape)} + token_mask {list(m.shape)} -> top1_index {list(i.shape)} "
              f"{i.dtype}, top1_logit {list(top.shape)}, blank_logit {list(bl.shape)}")
    print(json.dumps(R["source"], indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("cmd", choices=("parity", "absmax", "info"))
    add_reader_args(ap)
    ap.add_argument("--tokens", type=int, nargs="+", default=list(TOKEN_BUCKETS), help="window sizes (tokens)")
    ap.add_argument("--fixtures")
    ap.add_argument("--cache", help="soft token cache root (default $EG2STT_CACHE or <work>/cache)")
    ap.add_argument("--n-test", type=int, default=N_TEST, help="the leading N test-clean utterances of the cache index")
    ap.add_argument("--out-dir", help="default <work>/exports/reader (reader v1), <work>/exports/reader_<version> (later)")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    {"parity": cmd_parity, "absmax": cmd_absmax, "info": cmd_info}[args.cmd](args)


if __name__ == "__main__":
    main()
