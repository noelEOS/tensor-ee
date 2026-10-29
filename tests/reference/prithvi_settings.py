"""NumPy mirrors of the window blocks at the Prithvi channel settings.

Products accumulate in double; each operation rounds once to float32. Structured
random tiles let the live fixtures exercise all channels inside the request cap.
"""
from __future__ import annotations

import numpy as np


def linear(x, weight, bias=None):
    product = np.asarray(x, np.float64) @ np.asarray(weight, np.float64).T
    return np.asarray(product if bias is None else product + bias, np.float32)


def norm(x, gamma, beta, eps=1e-6):
    x = np.asarray(x, np.float64)
    mean = np.cumsum(x, axis=-1)[..., -1:] / x.shape[-1]
    centered = x - mean
    var = np.cumsum(centered * centered, axis=-1)[..., -1:] / x.shape[-1]
    return np.asarray(centered / np.sqrt(var + eps) * gamma + beta, np.float32)


def patch(x, weight, bias, size=16):
    h, w, channels = x.shape
    rows = x.reshape(h // size, size, w // size, size, channels)
    columns = rows.transpose(0, 2, 4, 1, 3).reshape(h // size, w // size, channels * size * size)
    return linear(columns, weight, bias)


def conv3x3(x, weight):
    h, w, channels = x.shape
    padded = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    taps = np.stack([padded[dy:dy+h, dx:dx+w] for dy in range(3) for dx in range(3)], axis=2)
    matrix = weight.transpose(0, 2, 3, 1).reshape(weight.shape[0], -1)
    return linear(taps.reshape(h, w, 9 * channels), matrix)


def batch_norm(x, gamma, beta, mean, var, eps=1e-5):
    return np.asarray((np.asarray(x, np.float64) - mean) / np.sqrt(np.asarray(var, np.float64) + eps)
                      * gamma + beta, np.float32)
