"""NumPy mirror of ``conv3x3_window`` with ``crop``: a grid of independent crops.

Each crop of ``side`` x ``side`` tokens is convolved alone, with zero padding at its own
edges, by the window-im2col mirror of the unsplit block (one sequential dot over
(ky, kx, c) in double, the bias added in double, one float32 rounding), and the crops are
put back in place. Values only; nothing reads the library.
"""
from __future__ import annotations

import numpy as np

from .upernet import ee_mirror_decoder as M


def conv3x3_crops(x, weight, bias, side: int) -> np.ndarray:
    """[H, W, Cin] -> [H, W, Cout]; H and W are multiples of ``side``."""
    x = np.asarray(x, np.float32)
    h, w, _ = x.shape
    if h % side or w % side:
        raise ValueError(f"a {h} x {w} input is not a grid of {side}-token crops")
    out = np.empty((h, w, np.shape(weight)[0]), np.float32)
    for y in range(0, h, side):
        for x0 in range(0, w, side):
            block = M.conv3x3(x[y:y + side, x0:x0 + side], weight, bias, method="window_im2col")
            out[y:y + side, x0:x0 + side] = np.asarray(block, np.float32)
    return out
