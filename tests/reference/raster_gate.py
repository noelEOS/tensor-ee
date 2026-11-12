"""NumPy mirrors of the raster lane's global average pooling and elementwise multiply.

``global_avg_pool2d`` (``tensor_ee/blocks/raster/pooling.py``): each channel's float32 values
summed in double in row-major order, divided once by the exact count H W, rounded once to
float32 (the node boundary of policy ``float32_operands``). Earth Engine's accumulation order
and type for ``reduceResolution`` are not established; with inputs whose double sums are
exact (dyadic values of bounded magnitude), every order in double gives this result.

``multiply`` (``tensor_ee/blocks/raster/merge.py``): the product of two float32 values in
double, rounded once to float32: the correctly rounded float32 product, in either operand
order. A ``[1, 1, C]`` gate broadcasts over the map's pixels.

Arrays are [H, W, C]. Values only; nothing reads the library.
"""
from __future__ import annotations

import numpy as np


def global_avg_pool2d(x) -> np.ndarray:
    """[H, W, C] -> [1, 1, C] float32."""
    x = np.asarray(x, np.float32)
    height, width, channels = x.shape
    total = np.zeros(channels, np.float64)
    for i in range(height):
        for j in range(width):
            total = total + x[i, j].astype(np.float64)
    return (total / float(height * width)).astype(np.float32).reshape(1, 1, channels)


def multiply(x, other) -> np.ndarray:
    """[H, W, C] times [H, W, C] or [1, 1, C] -> [H, W, C] float32."""
    x = np.asarray(x, np.float32)
    other = np.asarray(other, np.float32)
    if other.shape != x.shape and other.shape != (1, 1, x.shape[2]):
        raise ValueError("multiply takes a same-shape operand or a [1, 1, C] gate")
    return (x.astype(np.float64) * other.astype(np.float64)).astype(np.float32)
