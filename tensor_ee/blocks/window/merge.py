"""W merges: the FPN's addition and the channel concatenation.

Moved from ``upernet/earth_engine/feature_ops.py`` (``add``, ``concat``), research code
archived at the tag ``research-archive-2026-10-12`` (docs/history/README.md). The
operands are W tensors (the decoder's ``Feature``)
carried with their grid and layout, so operands that do not share a grid, layout or
window are refused client-side (``same_grid``). Earth Engine itself would not refuse
them: binary operations on images of different grids are evaluated in the request's
projection with each input resampled (nearest), which silently replicates coarse values.

- ``add``: float32 + float32 -> float32; EE rounds the exact sum once, as IEEE / PyTorch
  do (policy ``double_round_once``: one rounding at the operation boundary).
- ``concat``: torch.cat(dim=1) order (first operand's channels first); window layout
  ``arrayCat`` on axis 1, dense bands renamed and stacked. Values are moved, never
  rounded: no arithmetic, no cast, no policy.
"""

from __future__ import annotations

from ...core.errors import GridMismatch
from ...core.metadata import ANY_SETTINGS, NO_ARITHMETIC, BlockMetadata, Locality, validate_metadata
from ...core.precision import DOUBLE_ROUND_ONCE, validate_policy
from ...core.tensor import _layout, band_names, same_grid
from .linear import no_constants


def add(a, b, *, policy: str = DOUBLE_ROUND_ONCE):
    """a + b of two W tensors on the same grid, layout and window, with equal channels."""
    validate_policy(policy, allowed=DOUBLE_ROUND_ONCE)
    same_grid(a, b)
    if a.channels != b.channels:
        raise GridMismatch(f"{a.channels} vs {b.channels} channels")
    return a.like(a.image.add(b.image).toFloat())


def concat(parts: list):
    """torch.cat(parts, dim=1) of W tensors: the decoder's ``Feature`` or a bare
    ``WindowTensor`` (window layout)."""
    same_grid(*parts)
    total = sum(p.channels for p in parts)
    if _layout(parts[0]) == "dense_bands":
        renamed = [p.image.select(band_names(p.channels), band_names(p.channels, f"p{i}_"))
                   for i, p in enumerate(parts)]
        image = renamed[0]
        for part in renamed[1:]:
            image = image.addBands(part)
        return parts[0].like(image.rename(band_names(total)), total)
    image = parts[0].image
    for p in parts[1:]:
        image = image.arrayCat(p.image, 1)
    return parts[0].like(image, total)


METADATA = validate_metadata({
    "add": BlockMetadata(
        name="add", layout_in="W", layout_out="W", locality=Locality.POINTWISE,
        policy=DOUBLE_ROUND_ONCE, embedded_constants=no_constants, asset_arrays=no_constants,
        status="Ready",
        evidence=("docs/history/building-blocks-2026-10-11.md#35-activations-merges-and-outputs", "docs/history/upernet/notes/milestone9_findings.md"),
        validated=ANY_SETTINGS),
    "concat": BlockMetadata(
        name="concat", layout_in="W", layout_out="W", locality=Locality.POINTWISE,
        policy=NO_ARITHMETIC, embedded_constants=no_constants, asset_arrays=no_constants,
        status="Ready",
        evidence=("docs/history/building-blocks-2026-10-11.md#35-activations-merges-and-outputs", "docs/history/upernet/notes/milestone8_findings.md"),
        validated=ANY_SETTINGS),
})
