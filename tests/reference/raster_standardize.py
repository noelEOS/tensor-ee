"""NumPy mirror of raster ``standardize``: every operation in IEEE float32.

``fl32(fl32(fl32(x * float32(scale)) - mean) / std)``: NumPy's float32 arithmetic rounds each
operation, which is what the block's double-then-``toFloat`` steps give for float32 operands.
Values only; nothing reads the library.
"""
from __future__ import annotations

import numpy as np


def standardize(x, mean, std, scale=None) -> np.ndarray:
    """[H, W, C] -> [H, W, C] float32."""
    x = np.asarray(x, np.float32)
    if scale is not None:
        x = x * np.float32(scale)
    return (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
