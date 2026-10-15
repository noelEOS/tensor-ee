"""Coefficient transport: how weights and per-channel constants enter a graph.

- ``ServerArray`` and ``constant_value`` (formerly ``swin_block.earth_engine.ops``):
  the two sources of float32 constants, server-side arrays (asset-backed) and
  NumPy values embedded with ``encode_float32``. ``ServerArray`` is the V
  carrier, defined in ``tensor_ee.core.tensor``; ``constant_value`` is defined
  in ``tensor_ee.core.precision``, below the ops interface whose ``constant``
  method uses it. Both are re-exported here as the same objects: this is the
  public coefficient-transport namespace.
- ``inline_float32_array`` (formerly ``tensor_ee.compiler._inline_float32_array``):
  the R convolution's inline coefficient matrix.
- **Layout-derived row repetition** (docs/library-contract.md, section 6).
  EE arrays do not broadcast, so a per-channel constant is pre-repeated to
  ``[rows, C]`` for the layout it serves: W with window ``w`` -> ``w**2`` rows,
  W with window 1 -> 1 row, V -> the value's row count. ``constant_rows``
  derives the count from a layout or tensor; ``row_constant`` builds a
  constant that records it; ``check_rows`` refuses constants built for another
  layout. The repeat expression is ``constant_value``'s unchanged
  ``reshape([1, C]).repeat(0, rows)``, so window 7 serializes exactly as
  ``repeat_rows=49`` does today, and the check is client-side only.

L02 provides the mechanism. Since L04 and L05 the constant builders
(``LinearConstants``, ``Conv3x3Constants``, ``BatchNormConstants``, the
reference model's ``block_constants`` and decoder ``*_constants``) take a
layout and record its rows; the L05 composites pass layouts. A row count is
still accepted (the builders' original call form). Windows 1 (dense rows), 5, 7,
8 and 12 are validated (5, 8 and 12 by task P01); other row counts are not.
"""

from __future__ import annotations

from dataclasses import dataclass

import ee
import numpy as np

from ..core.errors import GridMismatch
from ..core.precision import constant_value, encode_float32, exact_scalar  # noqa: F401
from ..core.tensor import WINDOW, ServerArray, WindowTensor


def inline_float32_array(values: list[list[float]]) -> ee.Array:
    """Preserve float32 coefficients outside EE's integral literal range.

    Ordinary coefficients retain their literal graph. The backend can decode a
    huge integral-valued JSON double through int64 and clamp it; parsed numbers
    avoid that path without changing float32 matrix arithmetic or asset arrays.
    """
    return ee.Array([[(exact_scalar(value) if value >= 2**63 or value < -(2**63) else value)
                      for value in row] for row in values])


# ---------------------------------------------------------------------------
# Layout-derived row repetition
# ---------------------------------------------------------------------------
class RowMismatch(GridMismatch):
    """Constants built for one layout's row count, used with another layout."""


def _positive(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer; received {value!r}")
    return int(value)


@dataclass(frozen=True)
class WindowLayout:
    """W layout with window ``w``: one ``[w**2, C]`` array per pixel. ``WindowLayout(1)``
    is the one-row dense layout. Windows 1, 5, 7, 8 and 12 are validated (P01)."""

    window: int = WINDOW

    def __post_init__(self):
        object.__setattr__(self, "window", _positive(self.window, "window"))

    @property
    def rows(self) -> int:
        return self.window * self.window


@dataclass(frozen=True)
class ValueLayout:
    """V layout: one ``[rows, C]`` ee.Array value (not spatial)."""

    rows: int

    def __post_init__(self):
        object.__setattr__(self, "rows", _positive(self.rows, "rows"))


def layout_of(layout) -> WindowLayout | ValueLayout:
    """The layout a constant must serve, from a layout or a tensor.

    - ``WindowLayout`` / ``ValueLayout``: itself;
    - a W tensor (``WindowTensor``, including the decoder's ``Feature``):
      ``WindowLayout(tensor.window)``;
    - a V value with a client-known 2-D shape (``ServerArray``, NumPy array):
      ``ValueLayout(shape[0])``.

    Anything else (an ``int``, a bare ``ee`` object, a vector) is refused: the
    row count must come from the layout, not from a literal.
    """
    if isinstance(layout, (WindowLayout, ValueLayout)):
        return layout
    if isinstance(layout, WindowTensor):
        return WindowLayout(layout.window)
    if isinstance(layout, (ServerArray, np.ndarray)):
        shape = tuple(layout.shape)
        if len(shape) != 2:
            raise ValueError(f"a V value has a 2-D [rows, C] shape; received {shape}")
        return ValueLayout(shape[0])
    raise TypeError(f"cannot derive a row count from {type(layout).__name__}; pass a layout or a tensor")


def constant_rows(layout) -> int:
    """Rows of a per-channel constant for ``layout`` (W window w: w**2; V: its rows).

    Layouts and W tensors carry ``rows`` (``WindowTensor.rows``); a V value's
    count comes from its client-known shape."""
    if isinstance(layout, (WindowLayout, ValueLayout, WindowTensor)):
        return layout.rows
    return layout_of(layout).rows


@dataclass(frozen=True)
class RowConstant:
    """A per-channel constant pre-repeated to ``[rows, C]``, with its row count.

    ``value`` is what ``ops.constant`` returns (an ee.Array for V, an ee.Image
    for W) and is used in the graph as is."""

    value: object
    rows: int

    def check(self, layout):
        """``value``, after refusing a layout with another row count."""
        check_rows(self, layout)
        return self.value


def row_constant(ops, values, layout) -> RowConstant:
    """``ops.constant(values, repeat_rows=constant_rows(layout))``, recording the rows.

    The emitted expression is exactly ``ops.constant(values, repeat_rows=rows)``."""
    rows = constant_rows(layout)
    return RowConstant(ops.constant(values, repeat_rows=rows), rows)


def check_rows(constants, layout) -> None:
    """Refuse ``constants`` (anything with a ``rows`` attribute: a ``RowConstant``
    or a builder that records its rows) built for another layout. Client-side
    only: nothing is added to the graph."""
    expected = constant_rows(layout)
    rows = getattr(constants, "rows", None)
    if rows != expected:
        raise RowMismatch(f"constants built for {rows} rows; this layout needs {expected}")
