"""The host half of the graph contract, in NumPy (HOST_CONTRACT §C–§D of the LiteRT lane).

The graph returns `token_logits` [1,S] and `pooled_cls` [1,768]; everything below is host work,
written as the exact algorithm the Swift side ports: gather the marker positions, compute the four
act features from the RAW (untempered) marker softmax, run the act function, then apply the
question's temperature and decode the answer. The arithmetic and rounding follow the LiteRT lane's
host_decode.py, which reproduced every official `Agent.predict` answer exactly
(laya 0.3.4, laya/agent.py and laya/common.py; Apache-2.0).
"""
from __future__ import annotations

import math

import numpy as np

QTYPES = {"choice": 0, "score": 1, "noul": 2}


def softmax(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    exp = np.exp(values - values.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def gather_markers(token_logits, markers) -> np.ndarray:
    return np.asarray(token_logits, dtype=np.float32).reshape(-1)[list(markers)]


def act_features(raw_logits) -> np.ndarray:
    """[top1, top1 - top2, entropy / ln(max(K,2)), max(K,2) / 255] from the raw marker softmax."""
    p = softmax(raw_logits).reshape(-1)
    k = max(len(p), 2)
    top = np.sort(p)[::-1]
    top1 = top[0]
    top2 = top[1] if len(top) > 1 else np.float32(0)
    entropy = -(p * np.log(np.maximum(p, np.float32(1e-9)))).sum() / np.log(np.float32(k))
    return np.array([[top1, top1 - top2, entropy, np.float32(k) / np.float32(255)]], dtype=np.float32)


def temperature_bucket(qtype, k: int) -> str:
    name = ("choice", "score", "noul")[qtype] if isinstance(qtype, int) else qtype
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return name + ":" + size


def temperature(qtype, k: int, config: dict) -> float:
    """Bucket first (`temperature_by_options`), then the per-type default — the order agent.py uses."""
    index = qtype if isinstance(qtype, int) else QTYPES[qtype]
    by_options = config.get("temperature_by_options", {})
    return float(by_options.get(temperature_bucket(index, k), config.get("temperature", [1.0, 1.0, 1.0])[index]))


def probabilities(raw_logits, qtype, config: dict) -> np.ndarray:
    logits = np.asarray(raw_logits, dtype=np.float32).reshape(-1)
    z = logits / max(1e-3, temperature(qtype, len(logits), config))
    p = np.exp(z - z.max())
    return p / p.sum()


def confidence(p) -> float:
    k = len(p)
    if k < 2:
        return 1.0
    entropy = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - entropy / math.log(k), 0.0, 1.0))


def decode(raw_logits, act_logits, question: dict, config: dict) -> dict:
    """Marker logits + act logits -> the upstream answer dictionary, with its rounding."""
    t = question["type"]
    p = probabilities(raw_logits, t, config)
    action = {"act_probability": round(float(softmax(act_logits).reshape(-1)[0]), 4)}
    if t == "choice":
        criteria = question["criteria"]
        keys = list(criteria) if not isinstance(criteria, list) else list(dict.fromkeys(criteria))
        return dict(type=t, choice=keys[int(p.argmax())],
                    probabilities={key: round(float(v), 4) for key, v in zip(keys, p)},
                    confidence=round(confidence(p), 4), action=action)
    if t == "score":
        return dict(type=t, score=round(float((np.arange(len(p)) * p).sum()), 4),
                    legend={str(i): c for i, c in enumerate(question["criteria"])},
                    probabilities={str(i): round(float(v), 4) for i, v in enumerate(p)},
                    confidence=round(confidence(p), 4), action=action)
    return dict(type="noul", noul=round(float(p[1]), 4),
                confidence=round(max(float(p[1]), 1.0 - float(p[1])), 4), action=action)
