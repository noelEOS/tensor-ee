"""R inference BatchNorm: ``(x - mean) / sqrt(var + eps) * gamma + beta``.

The formula, operation order and constant handling follow the window block's
explicit BatchNorm exactly, without importing that lane. Mean and variance start
as float32 constants, widened to double; ``sqrt(var + eps)`` is evaluated once
on the constant array, never per pixel. Gamma and beta retain their exact float32
values. ``affine=False`` is represented by gamma=None and beta=None, expanded
ahead to float32 ones and zeros (also independently for functional BatchNorm).

Each constant is flattened to scalar image bands in the supplied ``bands`` order.
EE pairs image operands by position: the input is explicitly selected in its
Raster band order and the constants keep their positional order even when those
names differ. Channel counts must agree. Coefficients must be finite float32
vectors, eps a finite static real, and var + eps positive.

Policy ``float32_operands``: the work is double and the caller materializes the
output in float32, as for R activations; this block only calls ``_finish``.
Offline evidence covers three distinct channels, affine True/False, eps 1e-5,
and clip_footprints=False. It has not been validated live.
"""

from __future__ import annotations

import math
from numbers import Real

import ee
import numpy as np

from ...core.errors import UnsupportedOperation
from ...core.metadata import NOT_LIVE, BlockMetadata, Locality, validate_metadata
from ...core.precision import FLOAT32_OPERANDS, exact_scalar, validate_policy
from ...core.tensor import Raster
from ...layouts.raster import finish as _finish
from ...weights.constants import constant_value


class BatchNormConstants:
    """Inline per-channel images in ``bands`` order; no asset-backed constants.

    ``gamma``/``beta`` may be None for identity affine coefficients. ``mean`` and
    ``var`` are required float32 vectors. eps defaults to 1e-5.
    """

    def __init__(self, gamma, beta, mean, var, *, bands, eps: float = 1e-5):
        self.bands = tuple(bands)
        self.channels = len(self.bands)
        if (not self.bands or any(not isinstance(b, str) or not b for b in self.bands)
                or len(set(self.bands)) != self.channels):
            raise UnsupportedOperation("BatchNorm requires nonempty unique channel names")
        if isinstance(eps, bool) or not isinstance(eps, Real) or not math.isfinite(eps):
            raise UnsupportedOperation("BatchNorm eps must be a finite static real")

        def vector(value, name):
            value = np.asarray(value)
            if value.dtype != np.float32 or value.shape != (self.channels,):
                raise UnsupportedOperation(f"BatchNorm {name} must be a float32 vector of {self.channels} channels")
            if not np.isfinite(value).all():
                raise UnsupportedOperation(f"BatchNorm {name} must be finite")
            return value

        mean, var = vector(mean, "mean"), vector(var, "var")
        gamma = vector(np.ones(self.channels, np.float32) if gamma is None else gamma, "gamma")
        beta = vector(np.zeros(self.channels, np.float32) if beta is None else beta, "beta")
        if not np.all(var.astype(np.float64) + eps > 0):
            raise UnsupportedOperation("BatchNorm running variance plus eps must be positive")

        def image(value):
            return ee.Image(value).arrayFlatten([list(self.bands)])

        self.mean = image(constant_value(ee, mean).double())
        self.std = image(constant_value(ee, var).double().add(exact_scalar(float(eps))).sqrt())
        self.gamma = image(constant_value(ee, gamma))
        self.beta = image(constant_value(ee, beta))

    @property
    def embedded_values(self) -> int:
        return batch_norm_values(self.channels)


def batch_norm(raster: Raster, constants: BatchNormConstants, *, clip_footprints: bool = False,
               policy: str = FLOAT32_OPERANDS) -> Raster:
    """Per-band inference BatchNorm, preserving the Raster's positional channel order."""
    validate_policy(policy, allowed=FLOAT32_OPERANDS)
    if len(raster.bands) != constants.channels:
        raise UnsupportedOperation("BatchNorm input channel count does not match constants")
    x = raster.image.select(list(raster.bands))
    image = x.subtract(constants.mean).divide(constants.std).multiply(constants.gamma).add(constants.beta)
    return _finish(image, raster.bands, raster.grid, clip_footprints)


def batch_norm_values(channels: int) -> int:
    """Four inline vectors: mean, std, gamma and beta (including identity affine)."""
    return 4 * channels


def batch_norm_arrays() -> int:
    """No asset arrays: all four vectors stay inline."""
    return 0


METADATA = validate_metadata({
    "batch_norm": BlockMetadata(
        name="batch_norm", layout_in="R", layout_out="R", locality=Locality.POINTWISE,
        policy=FLOAT32_OPERANDS, embedded_constants=batch_norm_values, asset_arrays=batch_norm_arrays,
        status="Fixed — offline only: 3 channels, affine True/False, eps 1e-5, clip_footprints=False",
        evidence=("docs/evidence/raster-batch-norm-offline.json",), validated=NOT_LIVE),
})
