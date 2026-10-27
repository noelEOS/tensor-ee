"""NumPy mirrors of the bounded activations the raster lane emits.

``relu6``, ``hardsigmoid`` and ``hardswish`` (``tensor_ee/blocks/raster/activations.py``) are
pointwise expressions over float32 operands, materialized with ``.toFloat()`` at the node
boundary (policy ``float32_operands``). Each mirror evaluates the block's expression in
float64, rounds the result to float32 once, and adds no other rounding: the rounding the
policy places at the node boundary, over the same double work.

``relu6`` is a selection between two representable bounds, so its mirror is exact: a float32
input comes back unchanged, ``0`` or ``6``.

``hardsigmoid`` and ``hardswish`` are not exact. PyTorch's float32 kernels round ``x + 3``,
the clamp and the sixth one at a time in float32; a mirror that rounds once differs by at
most one ulp of the result: 5.96e-8 for a hardsigmoid in [0, 1], and 4.77e-7 for a hardswish
of an input in [-8, 8] (one ulp of 8). Above the clamp PyTorch still rounds ``x * 6 * (1/6)``
in float32 and this mirror returns ``x`` exactly, so the two diverge with the input's
magnitude: it can differ from it by one ulp, which exceeds the 1e-4 absolute target from
``|x|`` of about 1e6 (measured: the first crossing at ``|x| = 1.014e6``). The fixtures keep
``|x| <= 8`` for that reason.

The blocks divide by 6 instead of multiplying by binary64 ``1/6``; the two give the same
float32 bits over that range (the blocks' docstrings record the measurement), and the
mirrors divide too.
"""

from __future__ import annotations

import numpy as np


def _clamp_shifted(values: np.ndarray) -> np.ndarray:
    """``min(max(x + 3, 0), 6)`` in float64."""
    return np.minimum(np.maximum(np.asarray(values, np.float64) + 3.0, 0.0), 6.0)


def relu6(values: np.ndarray) -> np.ndarray:
    """``min(max(x, 0), 6)``, exact for float32 inputs."""
    return np.minimum(np.maximum(np.asarray(values, np.float32), np.float32(0)), np.float32(6))


def hardsigmoid(values: np.ndarray) -> np.ndarray:
    """``min(max(x + 3, 0), 6) / 6``, rounded to float32 once."""
    return (_clamp_shifted(values) / 6.0).astype(np.float32)


def hardswish(values: np.ndarray) -> np.ndarray:
    """``x * min(max(x + 3, 0), 6) / 6``, the product first, rounded to float32 once."""
    return (np.asarray(values, np.float64) * _clamp_shifted(values) / 6.0).astype(np.float32)
