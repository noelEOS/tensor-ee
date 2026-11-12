"""Grouped cross-correlation mirror: zero padding, double work, one float32 rounding.

Inputs/outputs are HWC; weights keep PyTorch's [out, in_per_group, kh, kw] order.
Reference-only loops apply each group's actual channels without zero-filled
block-diagonal weights. Tap accumulation follows channel, row, column order.

``padding`` is PyTorch's zero padding ``(left, right, top, bottom)``; the default is the
symmetric ``kernel // 2`` on each axis. The output has PyTorch's count on each axis,
``(n + before + after - k) // stride + 1``, which is ``ceil(n / stride)`` at the default.
"""

import numpy as np


def grouped_conv2d(x, weight, bias=None, *, groups, stride=1, padding=None):
    x = np.asarray(x, np.float32).astype(np.float64)
    weight = np.asarray(weight, np.float32).astype(np.float64)
    out, inputs, kh, kw = weight.shape
    h, w, _ = x.shape
    left, right, top, bottom = (kw // 2, kw // 2, kh // 2, kh // 2) if padding is None else padding
    padded = np.pad(x, ((top, bottom), (left, right), (0, 0)))
    rows = (h + top + bottom - kh) // stride + 1
    columns = (w + left + right - kw) // stride + 1
    result = np.zeros((rows, columns, out), np.float64)
    for group in range(groups):
        outputs = slice(group * (out // groups), (group + 1) * (out // groups))
        for c in range(inputs):
            for ky in range(kh):
                for kx in range(kw):
                    values = padded[ky:ky + stride * (rows - 1) + 1:stride,
                                    kx:kx + stride * (columns - 1) + 1:stride, group * inputs + c]
                    result[..., outputs] += values[..., None] * weight[outputs, c, ky, kx]
    if bias is not None:
        result += np.asarray(bias, np.float32).astype(np.float64)
    return result.astype(np.float32)
