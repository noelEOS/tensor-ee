"""Deterministic hash-generated test tensors (moved from ``ee_attention/earth_engine/common.py``).

value(a, b, c; seed) is an integer hash mapped to a dyadic rational in [-1, 1) with 1/2048
resolution. Every intermediate is a nonnegative integer below 2**53 (for nonnegative a, b, c),
so EE doubles and NumPy int64 agree exactly, and the final value is exactly representable in
float32. Raster path: a = row y, b = column x; channel c selects the embedding coordinate.
The research code also built the same values on the Earth Engine server; that half is archived
at the tag ``research-archive-2026-10-12``.
"""

from __future__ import annotations

import numpy as np

_HASH = dict(a=7919, b=6007, c=104729, seed=1299709, offset=12345)


def hash_numpy(a, b, c, seed: int) -> np.ndarray:
    a, b, c = (np.asarray(v, dtype=np.int64) for v in (a, b, c))
    if (a < 0).any() or (b < 0).any() or (c < 0).any():
        raise ValueError("hash inputs must be nonnegative")
    h = (a * _HASH["a"] + b * _HASH["b"] + c * _HASH["c"]
         + seed * _HASH["seed"] + _HASH["offset"]) % 65536
    h = (h * 48271) % 65537
    h = (h * h + 1) % 65521
    return ((h % 4096).astype(np.float64) / 2048.0 - 1.0).astype(np.float32)


def generated_raster_numpy(height: int, width: int, channels: int, seed: int,
                           scale: float = 1.0, col_start: int = 0) -> np.ndarray:
    """[H, W, C] float32, value at (y, x, c) = hash(y, x, col_start + c)."""
    y = np.arange(height)[:, None, None]
    x = np.arange(width)[None, :, None]
    c = np.arange(col_start, col_start + channels)[None, None, :]
    values = hash_numpy(y, x, c, seed)
    return (values * np.float32(scale)).astype(np.float32)
