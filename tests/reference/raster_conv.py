"""Grouped cross-correlation mirror: zero padding, double work, one float32 rounding.

Inputs/outputs are HWC; weights keep PyTorch's [out, in_per_group, kh, kw] order.
Reference-only loops apply each group's actual channels without zero-filled
block-diagonal weights. Tap accumulation follows channel, row, column order.
"""

import numpy as np


def grouped_conv2d(x, weight, bias=None, *, groups, stride=1):
    x = np.asarray(x, np.float32).astype(np.float64)
    weight = np.asarray(weight, np.float32).astype(np.float64)
    out, inputs, kh, kw = weight.shape
    h, w, _ = x.shape
    padded = np.pad(x, ((kh // 2, kh // 2), (kw // 2, kw // 2), (0, 0)))
    result = np.zeros(((h + stride - 1) // stride, (w + stride - 1) // stride, out), np.float64)
    for group in range(groups):
        outputs = slice(group * (out // groups), (group + 1) * (out // groups))
        for c in range(inputs):
            for ky in range(kh):
                for kx in range(kw):
                    values = padded[ky:ky + h:stride, kx:kx + w:stride, group * inputs + c]
                    result[..., outputs] += values[..., None] * weight[outputs, c, ky, kx]
    if bias is not None:
        result += np.asarray(bias, np.float32).astype(np.float64)
    return result.astype(np.float32)
