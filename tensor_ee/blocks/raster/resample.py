"""R resampling: PyTorch nearest upsampling by an integer factor.

Moved whole from ``tensor_ee.compiler`` (``nearest_upsample``) and
``tensor_ee.semantics`` (``nearest_factor``). The old names are re-exports of
these objects.

Supported settings (contract section 8): one equal positive integer factor on
both axes. Factors 2, 3 and 4 are the validated settings: task P02's fixtures
``raster/nearest_upsample/f2``, ``f3`` and ``f4`` (``tools/fixtures/raster.py``)
compare them with PyTorch bit for bit. Other factors are accepted, as at the
release, but not validated.
"""

from __future__ import annotations

from typing import Any
import math
from numbers import Real

from ...core.metadata import BlockMetadata, Locality, Validated, validate_metadata
from ...core.precision import FLOAT32_OPERANDS, validate_policy
from ...core.tensor import Raster
from ...layouts.raster import finish as _finish, project as _project
from ...core.errors import reject as _reject
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


# ---------------------------------------------------------------------------
# Locality and cost metadata (contract section 9)
# ---------------------------------------------------------------------------
METADATA = validate_metadata({
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
