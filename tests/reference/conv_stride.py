"""NumPy mirror of strided window convolution, independent of the library.

One left-to-right double dot in (ky,kx,c) order, bias in double, rounded once.
Only requested output centers are computed; each crop has its own padding.
"""
from __future__ import annotations

import numpy as np


def conv3x3_stride(x, weight, bias=None, *, stride=2, crop_side=None):
    """HWC input -> H/stride, W/stride, Cout; zero padding 1, dilation/groups 1."""
    x, weight = np.asarray(x, np.float32), np.asarray(weight, np.float32)
    h, w, cin = x.shape
    if stride not in (1, 2) or h % stride or w % stride:
        raise ValueError("stride must be 1 or 2 and divide both input extents")
    if weight.shape[1:] != (cin, 3, 3):
        raise ValueError("weight must be [Cout,Cin,3,3]")
    if crop_side is not None and (crop_side < 1 or h % crop_side or w % crop_side or crop_side % stride):
        raise ValueError("crop side must divide both extents and be divisible by stride")
    result = np.empty((h // stride, w // stride, weight.shape[0]), np.float32)
    side_y, side_x = (h, w) if crop_side is None else (crop_side, crop_side)
    for top in range(0, h, side_y):
        for left in range(0, w, side_x):
            padded = np.pad(x[top:top + side_y, left:left + side_x], ((1, 1), (1, 1), (0, 0)))
            total = np.zeros((side_y // stride, side_x // stride, weight.shape[0]), np.float64)
            for ky in range(3):
                for kx in range(3):
                    values = padded[ky:ky + side_y:stride, kx:kx + side_x:stride].astype(np.float64)
                    for channel in range(cin):
                        total += values[..., channel, None] * weight[:, channel, ky, kx].astype(np.float64)
            if bias is not None:
                total += np.asarray(bias, np.float64)
            result[top // stride:(top + side_y) // stride,
                   left // stride:(left + side_x) // stride] = total.astype(np.float32)
    return result
