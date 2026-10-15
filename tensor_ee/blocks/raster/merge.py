"""R merges: channel concatenation (``torch.cat`` on dim 1) and residual addition.

Moved whole from ``tensor_ee.compiler`` (``concatenate``, ``residual_add``).
The old names are re-exports of these objects. Both require identical grids;
there is no broadcasting.
"""

from __future__ import annotations

from typing import Sequence

from ...core.errors import UnsupportedOperation
from ...core.metadata import ANY_SETTINGS, BlockMetadata, Locality, Validated, validate_metadata
from ...core.precision import FLOAT32_OPERANDS, validate_policy
from ...core.tensor import Raster
from ...layouts.raster import finish as _finish
from .conv import _names, _no_coefficients


def concatenate(rasters: Sequence[Raster], dim: int = 1,
                 *, clip_footprints: bool = False, policy: str = FLOAT32_OPERANDS) -> Raster:
    validate_policy(policy, allowed=FLOAT32_OPERANDS)
    if dim not in (1, -3) or not rasters:
        raise UnsupportedOperation("Concatenation requires a nonempty sequence and NCHW channel dim=1")
    grid = rasters[0].grid
    if any(raster.grid != grid for raster in rasters):
        raise UnsupportedOperation("Skip connection tensors must have identical spatial grids")
    parts = [raster.image.rename([f"part{i}_{band}" for band in raster.bands])
             for i, raster in enumerate(rasters)]
    result = parts[0]
    for part in parts[1:]:
        result = result.addBands(part)
    return _finish(result, _names(sum(len(raster.bands) for raster in rasters)), grid, clip_footprints)


def residual_add(left: Raster, right: Raster, *, clip_footprints: bool = False,
                 policy: str = FLOAT32_OPERANDS) -> Raster:
    """Elementwise addition of equally shaped tensors; broadcasting is rejected."""
    validate_policy(policy, allowed=FLOAT32_OPERANDS)
    if not isinstance(left, Raster) or not isinstance(right, Raster):
        raise UnsupportedOperation("Residual addition requires two raster tensors")
    if left.grid != right.grid or len(left.bands) != len(right.bands):
        raise UnsupportedOperation("Residual addition requires identical spatial grids and channel counts")
    return _finish(left.image.add(right.image.rename(list(left.bands))), left.bands, left.grid, clip_footprints)


# ---------------------------------------------------------------------------
# Locality and cost metadata (contract section 9)
# ---------------------------------------------------------------------------
METADATA = validate_metadata({
    # The U-Net's skip concatenations are torch.cat(dim=1); dim -3, the same axis, never ran.
    "concatenate": BlockMetadata(
        name="concatenate", layout_in="R", layout_out="R", locality=Locality.POINTWISE,
        policy=FLOAT32_OPERANDS, embedded_constants=_no_coefficients,
        asset_arrays=_no_coefficients, status="Ready",
        evidence=("docs/history/cnn/results/parity.json", "docs/history/cnn/results/resnet18_full64_float32_policy.json",
                  "docs/history/cnn/results/segwater_js_parity_1x1.json"),
        validated=Validated(({"dim": (1,)},))),
    "residual_add": BlockMetadata(
        name="residual_add", layout_in="R", layout_out="R", locality=Locality.POINTWISE,
        policy=FLOAT32_OPERANDS, embedded_constants=_no_coefficients,
        asset_arrays=_no_coefficients, status="Ready",
        evidence=("docs/history/cnn/results/resnet_primitives.json", "docs/history/cnn/results/resnet18_full64_float32_policy.json",
                  "docs/history/cnn/results/segwater_js_parity_1x1.json"),
        validated=ANY_SETTINGS),
})
