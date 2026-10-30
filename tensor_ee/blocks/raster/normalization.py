"""R per-band normalization: inference BatchNorm and input standardization.

**BatchNorm**: ``(x - mean) / sqrt(var + eps) * gamma + beta``.

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
Validated live at eps 1e-5 and clip_footprints=False, with affine True and False and three
channels of distinct statistics (Earth Engine equal to the NumPy mirror in every value).

**Standardize**: ``(x * scale - mean) / std`` per band, the input preparation of models
trained on scaled and normalized imagery (``scale`` optional: ``(x - mean) / std``).

Policy ``round_each_op``: the input is read as float32, and every operation is rounded to
float32 at once: ``fl32(fl32(fl32(x * scale) - mean) / std)``. Mean, std and scale are
float32 values, held exactly as doubles; Earth Engine computes each product, difference and
quotient of those operands in double, and rounding that double result to float32 gives the
IEEE float32 result (double rounding is harmless for +, -, *, / since 53 >= 2 * 24 + 2). So
the block reproduces a float32 pipeline such as ``x * 1e-4`` on a float32 array followed by
``torchvision.transforms.functional.normalize`` bit for bit. Missing pixels are the caller's:
``from_image`` zero-fills them before the block, as the source's nodata replacement does.

Each constant is a scalar band in the supplied ``bands`` order (positional pairing, as for
BatchNorm); the scale is one scalar for every band, multiplied as ``float32(scale)``, as a
float32 array multiplied by a Python float is. Mean and std must be finite float32 vectors,
std positive; the scale finite and nonzero in float32. Not validated live: the status names the settings its offline
fixtures cover.
"""

from __future__ import annotations

import math
from numbers import Real

import ee
import numpy as np

from ...core.errors import UnsupportedOperation, reject
from ...core.metadata import NOT_LIVE, BlockMetadata, Locality, Validated, validate_metadata
from ...core.precision import FLOAT32_OPERANDS, ROUND_EACH_OP, exact_scalar, validate_policy
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


class StandardizeConstants:
    """Inline per-band float32 ``mean`` and ``std`` in ``bands`` order, and an optional
    float32 ``scale`` for every band; no asset-backed constants. Each value is held
    exactly, as the double equal to its float32 value."""

    def __init__(self, mean, std, *, bands, scale=None):
        self.bands = tuple(bands)
        self.channels = len(self.bands)
        if (not self.bands or any(not isinstance(b, str) or not b for b in self.bands)
                or len(set(self.bands)) != self.channels):
            reject("standardize requires nonempty unique channel names")

        def vector(value, name):
            value = np.asarray(value)
            if value.dtype != np.float32 or value.shape != (self.channels,):
                reject(f"standardize {name} must be a float32 vector of {self.channels} channels")
            if not np.isfinite(value).all():
                reject(f"standardize {name} must be finite")
            return value

        mean, std = vector(mean, "mean"), vector(std, "std")
        if not np.all(std > 0):
            reject("standardize std must be positive")
        if scale is not None:
            if isinstance(scale, bool) or not isinstance(scale, Real) or not math.isfinite(scale):
                reject("standardize scale must be None or a finite real")
            with np.errstate(over="ignore"):
                single = np.float32(scale)
            if not np.isfinite(single) or single == 0:
                reject("standardize scale must be nonzero and within float32 range")

        def image(value):
            return ee.Image(constant_value(ee, value).double()).arrayFlatten([list(self.bands)])

        self.mean = image(mean)
        self.std = image(std)
        # A float32 pipeline multiplies by float32(scale) (1e-4 is not a float32 value):
        # the constant is that float32 value, exactly.
        self.scale = None if scale is None else ee.Image.constant(exact_scalar(float(np.float32(scale))))

    @property
    def embedded_values(self) -> int:
        return standardize_values(self.channels, scale=self.scale is not None)


def standardize(raster: Raster, constants: StandardizeConstants, *, clip_footprints: bool = False,
                policy: str = ROUND_EACH_OP) -> Raster:
    """Per-band ``(x * scale - mean) / std``, each operation rounded to float32 (the input
    read as float32), preserving the Raster's positional channel order. Not validated live."""
    validate_policy(policy, allowed=ROUND_EACH_OP)
    if len(raster.bands) != constants.channels:
        reject("standardize input channel count does not match constants")
    x = raster.image.select(list(raster.bands)).toFloat()
    if constants.scale is not None:
        x = x.multiply(constants.scale).toFloat()
    image = x.subtract(constants.mean).toFloat().divide(constants.std).toFloat()
    return _finish(image, raster.bands, raster.grid, clip_footprints)


def standardize_values(channels: int, *, scale: bool = False) -> int:
    """Inline values: the mean and std vectors, and the scale scalar if any."""
    return 2 * channels + int(bool(scale))


def standardize_arrays() -> int:
    """No asset arrays: the constants stay inline."""
    return 0


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
        status="Ready",
        evidence=("docs/evidence/raster-batch-norm-offline.json", "docs/evidence/live-fixtures/20261029T070703Z-batch_norm.json"),
        validated=Validated(({"eps": (1e-5,), "clip_footprints": (False,)},), sources=(BatchNormConstants,))),
    "standardize": BlockMetadata(
        name="standardize", layout_in="R", layout_out="R", locality=Locality.POINTWISE,
        policy=ROUND_EACH_OP, embedded_constants=standardize_values, asset_arrays=standardize_arrays,
        status="Fixed — offline only: 6 bands with scale 1e-4; 3 bands without scale; clip_footprints=False",
        evidence=("docs/evidence/raster-standardize-offline.json",),
        validated=NOT_LIVE),
})
