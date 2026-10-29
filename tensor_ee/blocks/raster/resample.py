"""R resampling: nearest and separable bilinear upsampling by an integer factor.

Moved whole from ``tensor_ee.compiler`` (``nearest_upsample``) and
``tensor_ee.semantics`` (``nearest_factor``). The old names are re-exports of
these objects.

Supported settings (contract section 8): one equal positive integer factor on
both axes. Factors 2, 3 and 4 are the validated settings: raster-setting validation's fixtures
``raster/nearest_upsample/f2``, ``f3`` and ``f4`` (``tools/fixtures/raster.py``)
compare them with PyTorch bit for bit. Other factors are accepted, as at the
release, but not validated.

``bilinear_upsample`` takes PyTorch's own 1-D coefficient matrices through
``BilinearConstants``; this lane never derives interpolation coefficients. The
compiler and fixtures obtain them from ``model_bilinear_weights`` in the window
lane with explicit float32 model dtype under both alignment rules. Every matrix
row must read only floor(output / factor) and its immediate
neighbours. Out-of-raster taps have zero weight; the matrices already include
PyTorch's edge clamp. Coefficients must be finite, exact float32 values.

The constants store three vectors per axis, one weight per output row/column.
Fine-grid pixel coordinates select each weight from its constant array. Nine
integer relabelings of the zero-extended source sample the parent's 3 x 3 ring,
nearest, on that grid. The y taps are summed first, then the x taps, as in the
window block: out = sum_x Wx * (sum_y Wy * input[parent + (y, x)]).
Constant storage and graph size grow linearly with H_out + W_out, never with
their product. Exact float32 coefficients are widened through the existing
constant helper for double work; policy ``float32_operands`` leaves the final
float32 materialization to the caller and returns through ``_finish``.

Bilinear evidence is offline only: factors 2 and 4, align_corners False/True,
and factor 3, align_corners=False,
5 x 7 input, two channels, clip_footprints=False. No live validation is claimed.
The ring radius is one input pixel. With align_corners=True coefficients depend
on the full source extent, so halos cannot reproduce a resize on another chip.
"""

from __future__ import annotations

from typing import Any
import math
from numbers import Real

import ee
import numpy as np

from ...core.metadata import NOT_LIVE, BlockMetadata, Locality, Validated, validate_metadata
from ...core.precision import FLOAT32_OPERANDS, validate_policy
from ...core.tensor import Raster
from ...layouts.raster import finish as _finish, project as _project, zero_extended as _zero_extended
from ...core.errors import UnsupportedOperation, reject as _reject
from ...weights.constants import constant_value
from .conv import _no_coefficients, _pair


def nearest_factor(value: Any) -> int:
    factors = _pair(value)
    if (any(isinstance(x, bool) or not isinstance(x, Real) for x in factors)
            or factors[0] != factors[1]):
        _reject("Nearest upsampling requires one equal integer scale factor")
    factor = factors[0]
    if not math.isfinite(factor) or factor < 1 or int(factor) != factor:
        _reject("Nearest upsampling requires a positive integer scale factor")
    return int(factor)


def nearest_upsample(raster: Raster, scale_factor: Any,
                     *, clip_footprints: bool = False, policy: str = FLOAT32_OPERANDS) -> Raster:
    validate_policy(policy, allowed=FLOAT32_OPERANDS)
    factor = nearest_factor(scale_factor)
    grid = raster.grid.resized(1 / factor, raster.grid.width * factor, raster.grid.height * factor)
    # EE's default reprojection uses nearest neighbor. Aligned corner transforms
    # produce exactly PyTorch's repeated-pixel integer upsampling.
    return _finish(_project(raster.image, grid, clip_footprints), raster.bands, grid, clip_footprints)


class BilinearConstants:
    """Three exact float32 tap weights per output position, stored as double arrays.

    ``weights_y``/``weights_x`` are [output, input] matrices, supplied by the caller.
    ``factor`` is one positive integer for both axes. Shapes, finite exact float32
    representability and every row's one-parent-ring support are checked ahead.
    """

    def __init__(self, weights_y, weights_x, factor: int):
        if isinstance(factor, bool) or not isinstance(factor, (int, np.integer)) or factor < 1:
            raise UnsupportedOperation("Bilinear constants require a positive integer factor")
        self.factor = int(factor)

        def taps(matrix):
            matrix = np.asarray(matrix)
            if (matrix.ndim != 2 or min(matrix.shape) < 1
                    or matrix.shape[0] != self.factor * matrix.shape[1]):
                raise UnsupportedOperation("Bilinear matrix must have shape [factor * input, input]")
            if matrix.dtype.kind not in "fiu" or not np.isfinite(matrix).all():
                raise UnsupportedOperation("Bilinear weights must be finite exact float32 values")
            with np.errstate(over="ignore", invalid="ignore"):
                rounded = matrix.astype(np.float32)
            if not np.array_equal(rounded.astype(np.float64), matrix):
                raise UnsupportedOperation("Bilinear weights must be finite exact float32 values")
            local = np.zeros((matrix.shape[0], 3), np.float32)
            for out, row in enumerate(rounded):
                parent = out // self.factor
                for tap, source in enumerate(range(parent - 1, parent + 2)):
                    if 0 <= source < matrix.shape[1]:
                        local[out, tap] = row[source]
                if (np.any(row[:max(0, parent - 1)] != 0)
                        or np.any(row[parent + 2:] != 0)):
                    raise UnsupportedOperation("Bilinear output reads beyond the parent's one-pixel ring")
            return local, matrix.shape[1]

        self.taps_y, self.height = taps(weights_y)
        self.taps_x, self.width = taps(weights_x)
        self.weights_y = tuple(ee.Image(constant_value(ee, self.taps_y[:, i]).double()) for i in range(3))
        self.weights_x = tuple(ee.Image(constant_value(ee, self.taps_x[:, i]).double()) for i in range(3))

    @property
    def embedded_values(self) -> int:
        return bilinear_upsample_values(self.height, self.width, self.factor)


def bilinear_upsample(raster: Raster, constants: BilinearConstants, *, align_corners: bool = False,
                      clip_footprints: bool = False, policy: str = FLOAT32_OPERANDS) -> Raster:
    """Apply caller-supplied separable coefficients, y first, on an integer-refined grid.

    ``align_corners`` describes the supplied matrices for locality metadata; the
    caller must obtain matrices at that setting. It is a static bool, never a
    second coefficient implementation. Input dimensions must match the constants.
    """
    validate_policy(policy, allowed=FLOAT32_OPERANDS)
    bilinear_upsample_edge(align_corners=align_corners)
    if (raster.grid.height, raster.grid.width) != (constants.height, constants.width):
        raise UnsupportedOperation("Bilinear input dimensions do not match constants")
    factor = constants.factor
    grid = raster.grid.resized(1 / factor, raster.grid.width * factor, raster.grid.height * factor)
    coords = ee.Image.pixelCoordinates(grid.projection()).floor()
    rows = coords.select("y").clamp(0, grid.height - 1).toInt()
    cols = coords.select("x").clamp(0, grid.width - 1).toInt()
    wy = tuple(weight.arrayGet(rows) for weight in constants.weights_y)
    wx = tuple(weight.arrayGet(cols) for weight in constants.weights_x)
    base = _zero_extended(raster).select(list(raster.bands))
    a, b, c, d, e, f = raster.grid.transform
    result = None
    for ix, dx in enumerate((-1, 0, 1)):
        vertical = None
        for iy, dy in enumerate((-1, 0, 1)):
            shifted = ee.Projection(raster.grid.crs, [a, b, c - a * dx - b * dy,
                                                     d, e, f - d * dx - e * dy])
            sample = base.changeProj(raster.grid.projection(), shifted).reproject(grid.projection())
            term = sample.multiply(wy[iy])
            vertical = term if vertical is None else vertical.add(term)
        term = vertical.multiply(wx[ix])
        result = term if result is None else result.add(term)
    return _finish(result, raster.bands, grid, clip_footprints)


def bilinear_upsample_values(height: int, width: int, factor: int) -> int:
    """Three inline tap vectors per axis, on the output grid."""
    return 3 * factor * (height + width)


def bilinear_upsample_arrays() -> int:
    """All coefficients stay inline; no asset arrays."""
    return 0


def bilinear_upsample_edge(*, align_corners: bool = False) -> dict:
    """Only align_corners=True requires the full chip's interpolation extent."""
    if not isinstance(align_corners, bool):
        raise UnsupportedOperation("Bilinear align_corners must be a static bool")
    return {"chip_edge_dependent": align_corners}


# ---------------------------------------------------------------------------
# Locality and cost metadata (contract section 9)
# ---------------------------------------------------------------------------
METADATA = validate_metadata({
    "bilinear_upsample": BlockMetadata(
        name="bilinear_upsample", layout_in="R", layout_out="R", locality=Locality.KERNEL, radius=1,
        policy=FLOAT32_OPERANDS, embedded_constants=bilinear_upsample_values,
        asset_arrays=bilinear_upsample_arrays, chip_edge_dependent=True, per_setting=bilinear_upsample_edge,
        status="Fixed — offline only: factors 2, 4 (align_corners False/True), 3 (align_corners=False); 5 x 7 input; 2 channels; clip_footprints=False",
        evidence=("docs/evidence/raster-bilinear-offline.json",), validated=NOT_LIVE),
    # receptive_field._upsample_axis: output index i reads input floor(i / factor),
    # a single input pixel.
    "nearest_upsample": BlockMetadata(
        name="nearest_upsample", layout_in="R", layout_out="R", locality=Locality.POINTWISE,
        policy=FLOAT32_OPERANDS, embedded_constants=_no_coefficients,
        asset_arrays=_no_coefficients, status="Ready — factors 2, 3, 4",
        evidence=("docs/evidence/live-fixtures/20261013T065257Z-p02.json", "docs/evidence/p02-raster-offline.json",
                  "docs/history/cnn/results/parity.json", "docs/history/cnn/results/resnet18_full64_float32_policy.json",
                  "docs/history/cnn/results/segwater_js_parity_1x1.json"),
        # scale_factor as one integer (``nearest_factor`` normalizes an equal pair).
        validated=Validated(({"scale_factor": (2, 3, 4)},))),
})
