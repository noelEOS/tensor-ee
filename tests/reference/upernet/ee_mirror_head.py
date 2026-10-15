"""Milestone 11, Phase 6: the float64 mirror of the EE segmentation head
(upernet/earth_engine/segmentation_head.py), rounding to float32 exactly where EE rounds.

    head_conv   Milestone 2's linear per token: [.., 256] x W^T [256, 2] as sequential double
                dots (float32 x float32 products are exact in double), + bias in double, one
                float32 rounding
    upsample    resize.window_resize with align_corners=True: the float32 model's 1-D
                coefficients (PyTorch's own, read off its float32 kernel), the y axis first
                (one matrixMultiply per window: products exact in double, the <= 2 non-zero
                products of each output summed in double), then x (float32 coefficient x
                double: each product rounded in double, the two summed in double), one
                float32 rounding at the end

The second stage is the only place where the head's double arithmetic is inexact, so it is
written out (two explicit products, one addition) instead of a library matmul, whose
kernels may fuse the multiply and the add (FMA, a different rounding).  `fma=True` computes
the fused variant exactly (rational arithmetic, small fixtures only), so a live probe can
tell which one EE performs.

Negative controls (Phase 4 / 5, numerical characterization only):
    upsample_per_window    the same align_corners=True interpolation applied to each 7 x 7
                           window independently (7 -> 28 with its own endpoints): what
                           "repeating a local template" would compute
    upsample_exact_coefficients   the textbook rational coefficients instead of the float32
                           model's (float64 / exact arithmetic)
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import torch

from tensor_ee.blocks.window import resample as RS
from reference.upernet import ee_mirror_decoder as M

HEAD = "segmentation_head.0"
WEIGHT = f"{HEAD}.weight"
BIAS = f"{HEAD}.bias"
FACTOR = 4


def head_conv(x, weight, bias, exact: bool = False) -> np.ndarray:
    """Conv2d 256 -> 2, 1 x 1, with bias: x [.., 256] float32 values, weight [2, 256] (or
    [2, 256, 1, 1]), bias [2]."""
    w = np.asarray(weight, np.float32).reshape(np.shape(weight)[0], -1)
    return M.linear(x, w, bias, exact)


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


def upsample(x, size=None, factor: int = FACTOR, align_corners: bool = True, exact: bool = False,
             fma: bool = False, weights=None) -> np.ndarray:
    """EE's window_resize(align_corners) on x [H, W, C] (float32 values): y then x.
    weights: optional (My, Mx) to use instead of PyTorch's float32 ones."""
    x64 = np.asarray(x, np.float64)
    h, w, c = x64.shape
    size = (h * factor, w * factor) if size is None else tuple(size)
    my, mx = weights if weights is not None else (
        RS.model_bilinear_weights(h, size[0], align_corners=align_corners),
        RS.model_bilinear_weights(w, size[1], align_corners=align_corners))
    # stage 1 (y): every product exact in double, <= 2 non-zero terms: any order and FMA agree
    a = (torch.from_numpy(np.ascontiguousarray(my)) @ torch.from_numpy(x64.reshape(h, w * c))).numpy()
    a = a.reshape(size[0], w, c)
    # stage 2 (x): float32 coefficient x double: two rounded products, one rounded sum
    i0, i1, w0, w1 = _pairs(mx)
    p0 = a[:, i0, :] * w0[None, :, None]
    if fma:
        b = _exact_fma(np.broadcast_to(w1[None, :, None], a[:, i1, :].shape), a[:, i1, :], p0)
    else:
        p1 = a[:, i1, :] * w1[None, :, None]
        b = p0 + p1
    return b if exact else b.astype(np.float32).astype(np.float64)


def matrix_upsample(x, size, align_corners: bool = True, exact: bool = False) -> np.ndarray:
    """resize.matrix_resize (any sizes): per output pixel one sequential double dot of the 2-D
    coefficients (products of the float32 1-D ones, exact in double) with the row-major
    source pixels, each product rounded in double (no FMA), one float32 rounding."""
    h, w, c = np.shape(x)
    weights = RS.output_weights((h, w), tuple(size), align_corners)          # [Ho Wo, h w]
    src = np.asarray(x, np.float64).reshape(h * w, c)
    acc = np.zeros((weights.shape[0], c))
    for k in range(h * w):                                                     # sequential
        acc = acc + weights[:, k:k + 1] * src[k][None, :]
    y = acc.reshape(size[0], size[1], c)
    return y if exact else y.astype(np.float32).astype(np.float64)


def upsample_per_window(x, factor: int = FACTOR, window: int = 7) -> np.ndarray:
    """Negative control: align_corners=True applied to every window x window block on its
    own (window -> factor * window, endpoints of each block aligned), the mirror's rounding."""
    x64 = np.asarray(x, np.float64)
    h, w, c = x64.shape
    out = np.zeros((h * factor, w * factor, c))
    n = window * factor
    for i in range(0, h, window):
        for j in range(0, w, window):
            out[i * factor:i * factor + n, j * factor:j * factor + n] = upsample(
                x64[i:i + window, j:j + window], factor=factor)
    return out


def membership(matrix: np.ndarray) -> list[tuple[int, ...]]:
    """The source indices with a non-zero weight, per output index."""
    return [tuple(np.flatnonzero(row)) for row in matrix]
