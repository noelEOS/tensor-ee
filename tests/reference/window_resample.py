"""Separable window-resize mirror, independent of library coefficients and model extras.

Float32 PyTorch one-hot coefficients; double products and sums, y before x,
then one float32 rounding. Each axis has at most two nonzero taps. The optional
FMA probe computes the x-stage fused variant with exact rational arithmetic.
Extracted from reference.upernet.ee_mirror_head; its historical API delegates here.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np


def resize_weights(n_in: int, n_out: int, align_corners: bool) -> np.ndarray:
    """Read one axis's float32 coefficients from PyTorch, without library code."""
    import torch
    import torch.nn.functional as F

    eye = torch.eye(n_in, dtype=torch.float32)[:, None, None, :]
    with torch.no_grad():
        out = F.interpolate(eye, size=(1, n_out), mode="bilinear", align_corners=align_corners)
    return out[:, 0, 0, :].T.double().numpy()


def _pairs(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per output row the (first, second) non-zero columns and weights of a 1-D weight
    matrix (second weight 0 and column = first when the row has one non-zero entry)."""
    n_out = matrix.shape[0]
    i0 = np.zeros(n_out, int)
    i1 = np.zeros(n_out, int)
    w0 = np.zeros(n_out)
    w1 = np.zeros(n_out)
    for o in range(n_out):
        nz = np.flatnonzero(matrix[o])
        if not 1 <= len(nz) <= 2:
            raise AssertionError(f"row {o} has {len(nz)} non-zero weights")
        i0[o], w0[o] = nz[0], matrix[o, nz[0]]
        i1[o], w1[o] = nz[-1], (matrix[o, nz[-1]] if len(nz) == 2 else 0.0)
    return i0, i1, w0, w1


def _exact_fma(w: np.ndarray, a: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """round_double(w * a + acc) with one rounding (an FMA), elementwise, in rationals."""
    flat = [float(Fraction(float(wi)) * Fraction(float(ai)) + Fraction(float(ci)))
            for wi, ai, ci in zip(np.ravel(w), np.ravel(a), np.ravel(acc))]
    return np.asarray(flat).reshape(np.shape(acc))


def upsample(x, size=None, factor: int = 4, align_corners: bool = True, exact: bool = False,
             fma: bool = False, weights=None) -> np.ndarray:
    """EE's window_resize(align_corners) on x [H, W, C] (float32 values): y then x.
    weights: optional (My, Mx) to use instead of PyTorch's float32 ones."""
    x64 = np.asarray(x, np.float64)
    h, w, c = x64.shape
    size = (h * factor, w * factor) if size is None else tuple(size)
    my, mx = weights if weights is not None else (
        resize_weights(h, size[0], align_corners),
        resize_weights(w, size[1], align_corners))
    # stage 1 (y): every product exact in double, <= 2 non-zero terms: any order and FMA agree
    j0, j1, v0, v1 = _pairs(my)
    a = v0[:, None, None] * x64[j0] + v1[:, None, None] * x64[j1]
    # stage 2 (x): float32 coefficient x double: two rounded products, one rounded sum
    i0, i1, w0, w1 = _pairs(mx)
    p0 = a[:, i0, :] * w0[None, :, None]
    if fma:
        b = _exact_fma(np.broadcast_to(w1[None, :, None], a[:, i1, :].shape), a[:, i1, :], p0)
    else:
        p1 = a[:, i1, :] * w1[None, :, None]
        b = p0 + p1
    return b if exact else b.astype(np.float32).astype(np.float64)

