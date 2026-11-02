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

import numpy as np

from tensor_ee.blocks.window import resample as RS
from reference.upernet import ee_mirror_decoder as M
from reference.window_resample import _pairs, _exact_fma  # historical helper aliases

HEAD = "segmentation_head.0"
WEIGHT = f"{HEAD}.weight"
BIAS = f"{HEAD}.bias"
FACTOR = 4


def head_conv(x, weight, bias, exact: bool = False) -> np.ndarray:
    """Conv2d 256 -> 2, 1 x 1, with bias: x [.., 256] float32 values, weight [2, 256] (or
    [2, 256, 1, 1]), bias [2]."""
    w = np.asarray(weight, np.float32).reshape(np.shape(weight)[0], -1)
    return M.linear(x, w, bias, exact)


def upsample(x, size=None, factor: int = FACTOR, align_corners: bool = True, exact: bool = False,
             fma: bool = False, weights=None) -> np.ndarray:
    """Historical head mirror API; the separable arithmetic lives in window_resample."""
    from reference.window_resample import upsample as resize

    h, w, _ = np.shape(x)
    size = (h * factor, w * factor) if size is None else tuple(size)
    if weights is None:
        weights = (RS.model_bilinear_weights(h, size[0], align_corners=align_corners),
                   RS.model_bilinear_weights(w, size[1], align_corners=align_corners))
    return resize(x, size, factor, align_corners, exact, fma, weights)


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
