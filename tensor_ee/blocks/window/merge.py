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

Row concatenation moves raw operands [R1,C], [R2,C], ... to
[sum(R),C] in operand order, including a one-row class token beside a
196-row spatial window. It has no arithmetic, casts or coefficient constants.
Validated live with one class row before 196 and before 9 spatial rows (Earth Engine
equal to the NumPy mirror in every value).
"""

from __future__ import annotations

import numpy as np

from ...core.errors import GridMismatch, reject
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


def concat_rows(ops, parts, *, rows, channels: int):
    """Concatenate raw W/V matrices along rows without arithmetic or casts.

    ``parts`` share one physical grid on the W path. ``rows`` gives their
    positive row counts; each has ``channels`` columns. Unlike channel concat,
    the result can have a non-square row count, such as one class row followed
    by 196 spatial rows. Such an intermediate is an ops operand, not a spatial
    ``WindowTensor``. The caller supplies aligned raw operands: no implicit
    resampling or square-window reinterpretation is performed here.

    There are no coefficient constants or constants builder. Row ordering and
    the class-plus-spatial settings have offline evidence only; no live claim.
    """
    parts, rows = tuple(parts), tuple(rows)
    if not parts or len(parts) != len(rows):
        reject("parts and rows must be non-empty sequences of equal length")
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1
           for n in (*rows, channels)):
        reject("row counts and channels must be positive integers")
    return ops.cat_rows(parts)


def concat_rows_values(*, rows, channels: int) -> int:
    """No embedded coefficient values: concatenation only moves rows."""
    return 0


def concat_rows_arrays() -> int:
    """No coefficient arrays are read."""
    return 0


METADATA = validate_metadata({
    "concat_rows": BlockMetadata(
        name="concat_rows", layout_in="W", layout_out="W", locality=Locality.POINTWISE,
        policy=NO_ARITHMETIC, embedded_constants=concat_rows_values, asset_arrays=concat_rows_arrays,
        status="Ready",
        evidence=("docs/evidence/class-token-offline.json", "docs/evidence/live-fixtures/20261029T174042Z-class_token.json"),
        validated=ANY_SETTINGS),
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
