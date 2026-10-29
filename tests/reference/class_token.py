"""NumPy mirror of shared-key attention with a separately updated class row.

The arithmetic core follows the existing round_each_op policy. The mirror is
bounded: double accumulation order and exp are not certified against EE.
"""
from __future__ import annotations

import numpy as np


def concat_rows(parts):
    """Exact row move; signed zero and float32 payload bits are preserved."""
    return np.concatenate(parts, axis=0)


def expand_qkv(base, heads: int, head_dim: int):
    """Compact random fixture coefficients expanded to distinct head columns.

    Every head reads the same random values under a different channel rotation.
    This avoids embedding a separate high-width input in every live request.
    """
    return np.concatenate([
        np.concatenate([np.roll(base[:, s * head_dim:(s + 1) * head_dim], -h, axis=1)
                        for h in range(heads)], axis=1)
        for s in range(3)], axis=1)


def _attention(q, k, v, scale):
    scores = (q.astype(np.float64) @ k.astype(np.float64).T).astype(np.float32)
    scaled = (scores.astype(np.float64) * np.float32(scale)).astype(np.float32)
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(shifted.astype(np.float64)).astype(np.float32)
    total = np.cumsum(exp.astype(np.float64), axis=1)[:, -1:].astype(np.float32)
    probability = (exp.astype(np.float64) / total).astype(np.float32)
    return (probability.astype(np.float64) @ v.astype(np.float64)).astype(np.float32)


def class_token_attention(patch_qkv, cls_qkv, *, dim: int, heads: int):
    """Both query partitions attend to `[class; spatial]` keys and values."""
    d = dim // heads
    patch, cls = [], []
    for h in range(heads):
        p = [patch_qkv[:, s * dim + h * d:s * dim + (h + 1) * d] for s in range(3)]
        c = [cls_qkv[:, s * dim + h * d:s * dim + (h + 1) * d] for s in range(3)]
        k, v = [concat_rows([c[s], p[s]]) for s in (1, 2)]
        patch.append(_attention(p[0], k, v, d ** -0.5))
        cls.append(_attention(c[0], k, v, d ** -0.5))
    return {"patch": np.concatenate(patch, axis=1), "cls": np.concatenate(cls, axis=1)}
