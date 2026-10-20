"""Tensor carriers: R (``Raster``) and W (``WindowTensor``), plus ``Feature``.

- **R** ``Raster(image, bands, grid)``: one band per channel on a ``Grid``
  (formerly ``tensor_ee.compiler.Raster``; fields, constructor and equality
  unchanged). ``tensor_ee.compiler.CompiledModel`` subclasses it.
- **W** ``WindowTensor(image, grid, channels, window=7)``: one
  ``[window**2, channels]`` float32 array per pixel of ``grid.coarse(window)``.
  Token ``t = window * a + b`` of window ``(I, J)`` is pixel
  ``(window * I + a, window * J + b)`` of ``grid``. A one-row array per pixel
  of ``grid`` (Feature's ``dense_row``) is W with ``window=1``.
- ``Feature(image, grid, channels, layout="window")``: the decoder's carrier
  (formerly ``upernet.earth_engine.feature_ops.Feature``), a ``WindowTensor``
  whose window follows from its layout.
- **V** ``ServerArray(array, shape)``: a float32 parameter already on the
  server, with its shape known to the client (formerly
  ``swin_block.earth_engine.ops.ServerArray``; moved in core ops and weight transport and re-exported
  from ``tensor_ee.weights.constants``). Not spatial; no grid.

Windows 5, 7, 8 and 12 (and 1 for ``dense_row``) are validated live (including the window-size checks);
a window field is not evidence that other windows work (contract section 8).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import ee

from .errors import GridMismatch
from .grid import Grid

WINDOW = 7
"""Default window: today's value, the only one validated live."""

LAYOUTS = ("window", "dense_bands", "dense_row")
_LAYOUT_WINDOWS = {"window": WINDOW, "dense_bands": 1, "dense_row": 1}


def band_names(channels: int, prefix: str = "c") -> list[str]:
    """Scalar band names ``c0 .. c{channels-1}`` (or another prefix)."""
    return [f"{prefix}{i}" for i in range(channels)]


@dataclass(frozen=True)
class ServerArray:
    """A float32 parameter that already exists on the server (an ee.Array,
    e.g. read from a table asset), with its shape known to the client.

    Accepted wherever a NumPy parameter is: constant_value uses the array as
    is instead of embedding the values in the request.
    """

    array: object
    shape: tuple

    @property
    def T(self) -> "ServerArray":  # noqa: N802 (NumPy's name)
        if len(self.shape) != 2:
            raise ValueError("transpose takes a matrix")
        return ServerArray(self.array.transpose(), (self.shape[1], self.shape[0]))


@dataclass(frozen=True)
class Raster:
    image: ee.Image
    bands: tuple[str, ...]
    grid: Grid


@dataclass(frozen=True)
class WindowTensor:
    """W tensor: ``image`` holds one ``[window**2, channels]`` array per pixel of
    ``grid.coarse(window)``; ``grid`` is the token grid the values belong to
    (a ``TokenGrid`` or ``FeatureGrid``). Only window 7 is validated."""

    image: object
    grid: object
    channels: int
    window: int = WINDOW

    def __post_init__(self):
        if isinstance(self.window, bool) or not isinstance(self.window, int) or self.window < 1:
            raise ValueError(f"window must be a positive integer; received {self.window!r}")

    def like(self, image, channels: int | None = None):
        """The same grid, layout and window with a new image (and channel count)."""
        return replace(self, image=image, channels=self.channels if channels is None else channels)

    @property
    def rows(self) -> int:
        """Rows of each per-pixel array, ``window ** 2`` (contract section 6's
        ``layout.rows``). A read-only property, not a field: ``repr``, the
        dataclass fields and equality are unchanged."""
        return self.window ** 2


@dataclass(frozen=True)
class Feature(WindowTensor):
    """image: window layout [49, C] on grid.coarse(7) ("window"), C scalar bands c0.. on
    grid ("dense_bands") or one [1, C] array band on grid ("dense_row").  grid is the
    token grid (sa.TokenGrid or FeatureGrid) the values belong to.

    The window follows from the layout (7, or 1 for the two dense layouts); it is
    not a constructor argument, so ``Feature(image, grid, channels, layout)`` is
    unchanged."""

    window: int = field(default=WINDOW, init=False)
    layout: str = "window"

    def __post_init__(self):
        if self.layout not in LAYOUTS:
            raise ValueError(self.layout)
        object.__setattr__(self, "window", _LAYOUT_WINDOWS[self.layout])
        super().__post_init__()


def _layout(tensor) -> str:
    return getattr(tensor, "layout", "window")


def same_grid(*features: WindowTensor):
    """Refuse operands on different grids, layouts or windows (client-side only:
    EE would resample one of them implicitly)."""
    first = features[0]
    for f in features[1:]:
        if f.grid != first.grid or _layout(f) != _layout(first):
            raise GridMismatch(f"{_layout(f)} on {f.grid} vs {_layout(first)} on {first.grid}")
        if f.window != first.window:
            raise GridMismatch(f"window {f.window} vs window {first.window} on {first.grid}")
