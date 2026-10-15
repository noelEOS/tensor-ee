"""The three grid kinds behind one protocol, with explicit conversions.

There are three grid kinds, and they stay three classes:

- ``Grid(crs, transform, width, height)``: the converter's georeferenced
  north-up grid (formerly ``tensor_ee.compiler.Grid``);
- ``TokenGrid(height, width, scale=10.0, crs="EPSG:3857")``: the abstract
  Swin grid with origin (0, 0) (formerly
  ``ee_attention.earth_engine.spatial_attention.TokenGrid``);
- ``FeatureGrid(crs, x0, y0, scale, height, width)``: the decoder's
  georeferenced grid (formerly ``upernet.earth_engine.feature_interface.FeatureGrid``).

They describe the same thing but do not serialize the same way: int and float
zeros in the affine reach the graph, ``-s * 0`` gives ``-0.0``, and the
``TokenGrid``/``FeatureGrid`` domains floor the pixel coordinates while
``Grid.mask()`` does not (docs/evidence/l00-consolidation-probes.json). Each
kind therefore keeps its own constructor, equality, defaults and every
serialized projection and mask expression byte for byte. Merging them into
one class would change every Swin graph (a G2 item, deferred; contract D4).

Every kind implements ``GridProtocol``. Two members differ in call form, and
that difference is kept rather than papered over:

- ``projection``: ``Grid.projection()`` takes no argument;
  ``TokenGrid``/``FeatureGrid.projection(ee, dx=0, dy=0)`` take the ``ee``
  module and an optional integer translation.
- domain mask: ``Grid.mask()`` (unfloored coordinates; ``Grid.domain(ee)`` is
  the same expression); ``TokenGrid``/``FeatureGrid.domain(ee)`` (floored).

Conversions between kinds are explicit (``to_grid``, ``to_token_grid``,
``to_feature_grid``). A converted grid compares equal to its source after a
round trip, but its literals are those of its own kind.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol, Sequence, runtime_checkable

import ee


@runtime_checkable
class GridProtocol(Protocol):
    """Members every grid kind provides (see the module docstring for the two
    members whose call forms differ between kinds)."""

    crs: str
    height: int
    width: int

    @property
    def transform(self) -> Sequence[float]: ...

    @property
    def scale(self) -> float: ...

    def projection(self, *args: Any, **kwargs: Any) -> Any: ...

    def domain(self, ee: Any) -> Any: ...

    def coarse(self, factor: int) -> Any: ...

    def fine(self, factor: int) -> Any: ...

    def resized(self, factor: float, width: int, height: int) -> Any: ...


@dataclass(frozen=True)
class Grid:
    """North-up EE affine grid and finite tensor width/height, in pixels."""

    crs: str
    transform: tuple[float, float, float, float, float, float]
    width: int
    height: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "transform", tuple(self.transform))
        if len(self.transform) != 6 or not all(math.isfinite(x) for x in self.transform):
            raise ValueError("Grid transform must contain six finite numbers")
        a, b, _, d, e, _ = self.transform
        if b != 0 or d != 0 or a <= 0 or e >= 0:
            raise ValueError("This compiler requires a north-up grid: +x, -y, no shear")
        if not isinstance(self.width, int) or not isinstance(self.height, int):
            raise ValueError("Grid width and height must be integers")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Grid width and height must be positive")

    def projection(self) -> ee.Projection:
        return ee.Projection(self.crs, list(self.transform))

    def geometry(self) -> ee.Geometry:
        """Finite tensor extent, including its pixel corners, in the grid CRS."""
        a, _, c, _, e, f = self.transform
        return ee.Geometry.Rectangle([c, f + e * self.height, c + a * self.width, f],
                                     proj=self.crs, geodesic=False)

    def mask(self) -> ee.Image:
        coordinates = ee.Image.pixelCoordinates(self.projection())
        x, y = coordinates.select("x"), coordinates.select("y")
        return x.gte(0).And(x.lt(self.width)).And(y.gte(0)).And(y.lt(self.height))

    def resized(self, factor: float, width: int, height: int) -> Grid:
        """Multiply pixel spacing by factor, retaining the upper-left corner."""
        a, b, c, d, e, f = self.transform
        return Grid(self.crs, (a * factor, b * factor, c, d * factor, e * factor, f), width, height)

    # Protocol members (additive; nothing in the converter calls them).
    @property
    def scale(self) -> float:
        """Pixel spacing of a square-pixel grid; ValueError for non-square pixels."""
        a, _, _, _, e, _ = self.transform
        if a != -e:
            raise ValueError("Grid.scale requires square pixels")
        return a

    def domain(self, ee: Any = None) -> ee.Image:
        """``mask()``: the same unfloored expression. ``ee`` is accepted only for
        protocol symmetry and is not used."""
        return self.mask()

    def coarse(self, factor: int) -> Grid:
        """``resized(factor, width // factor, height // factor)``; extents must divide."""
        if self.height % factor or self.width % factor:
            raise ValueError("extents not divisible by the factor")
        return self.resized(factor, self.width // factor, self.height // factor)

    def fine(self, factor: int) -> Grid:
        """``resized(1 / factor, width * factor, height * factor)``, the form the
        converter's nearest upsampling uses."""
        return self.resized(1 / factor, self.width * factor, self.height * factor)


@dataclass(frozen=True)
class TokenGrid:
    """North-up grid with origin (0, 0), `scale` metres per token, rows downward."""

    height: int
    width: int
    scale: float = 10.0
    crs: str = "EPSG:3857"

    @property
    def transform(self) -> list[float]:
        return [self.scale, 0, 0, 0, -self.scale, 0]

    def projection(self, ee, dx: float = 0, dy: float = 0):
        """With dx/dy, the changeProj target of an exact integer shift:
        image.changeProj(P, P(dx, dy)) gives new(y, x) = old(y + dy, x + dx)."""
        s = self.scale
        return ee.Projection(self.crs, [s, 0, -s * dx, 0, -s, s * dy])

    def region(self, ee):
        return ee.Geometry.Rectangle([0, -self.scale * self.height,
                                      self.scale * self.width, 0], self.crs, False)

    def domain(self, ee):
        coords = ee.Image.pixelCoordinates(self.projection(ee)).floor()
        x, y = coords.select("x"), coords.select("y")
        return x.gte(0).And(x.lt(self.width)).And(y.gte(0)).And(y.lt(self.height))

    def coarse(self, window: int) -> "TokenGrid":
        if self.height % window or self.width % window:
            raise ValueError("window partition requires extents divisible by the window")
        return TokenGrid(self.height // window, self.width // window,
                         self.scale * window, self.crs)

    def pixel_grid(self) -> dict:
        """computePixels grid specification."""
        return {"dimensions": {"width": self.width, "height": self.height},
                "affineTransform": {"scaleX": self.scale, "shearX": 0, "translateX": 0,
                                    "shearY": 0, "scaleY": -self.scale, "translateY": 0},
                "crsCode": self.crs}

    # Protocol members (additive). ``fine`` builds exactly the grid that the research
    # resize (upernet/earth_engine/resize.py, archived at the tag
    # research-archive-2026-10-12) built for a grid without ``fine``.
    def fine(self, factor: int) -> "TokenGrid":
        return TokenGrid(self.height * factor, self.width * factor,
                         self.scale / factor, self.crs)

    def resized(self, factor: float, width: int, height: int) -> "TokenGrid":
        """Multiply the spacing by factor (origin (0, 0) is fixed)."""
        return TokenGrid(height, width, self.scale * factor, self.crs)


@dataclass(frozen=True)
class FeatureGrid:
    """A north-up grid: top-left corner (x0, y0), `scale` CRS units per pixel."""

    crs: str
    x0: float
    y0: float
    scale: float
    height: int
    width: int

    @property
    def transform(self) -> list[float]:
        return [self.scale, 0.0, self.x0, 0.0, -self.scale, self.y0]

    def projection(self, ee, dx: float = 0, dy: float = 0):
        """With dx/dy, the changeProj target of an exact integer translation:
        image.changeProj(P, P(dx, dy)) gives new(row, col) = old(row + dy, col + dx)."""
        s = self.scale
        return ee.Projection(self.crs, [s, 0, self.x0 - s * dx, 0, -s, self.y0 + s * dy])

    def coarse(self, factor: int) -> "FeatureGrid":
        if self.height % factor or self.width % factor:
            raise ValueError("extents not divisible by the factor")
        return FeatureGrid(self.crs, self.x0, self.y0, self.scale * factor,
                           self.height // factor, self.width // factor)

    def fine(self, factor: int) -> "FeatureGrid":
        return FeatureGrid(self.crs, self.x0, self.y0, self.scale / factor,
                           self.height * factor, self.width * factor)

    @property
    def bounds(self) -> list[float]:
        return [self.x0, self.y0 - self.scale * self.height,
                self.x0 + self.scale * self.width, self.y0]

    def region(self, ee):
        return ee.Geometry.Rectangle(self.bounds, self.crs, False)

    def domain(self, ee):
        coords = ee.Image.pixelCoordinates(self.projection(ee)).floor()
        x, y = coords.select("x"), coords.select("y")
        return x.gte(0).And(x.lt(self.width)).And(y.gte(0)).And(y.lt(self.height))

    def pixel_grid(self, row_start: int = 0, rows: int | None = None) -> dict:
        rows = self.height - row_start if rows is None else rows
        return {"dimensions": {"width": self.width, "height": rows},
                "affineTransform": {"scaleX": self.scale, "shearX": 0, "translateX": self.x0,
                                    "shearY": 0, "scaleY": -self.scale,
                                    "translateY": self.y0 - self.scale * row_start},
                "crsCode": self.crs}

    def record(self) -> dict:
        return {"crs": self.crs, "crs_transform": self.transform,
                "dimensions": [self.width, self.height]}

    # Protocol member (additive).
    def resized(self, factor: float, width: int, height: int) -> "FeatureGrid":
        """Multiply the spacing by factor, retaining the top-left corner."""
        return FeatureGrid(self.crs, self.x0, self.y0, self.scale * factor, height, width)


# ---------------------------------------------------------------------------
# Explicit conversions
# ---------------------------------------------------------------------------
def _square_origin(grid: Grid) -> tuple[float, float, float]:
    a, _, x0, _, e, y0 = grid.transform
    if a != -e:
        raise ValueError("conversion from Grid requires square pixels")
    return a, x0, y0


def to_grid(grid: Grid | TokenGrid | FeatureGrid) -> Grid:
    """The same pixels as a converter ``Grid`` (transform literals of the source kind)."""
    if isinstance(grid, Grid):
        return grid
    if isinstance(grid, (TokenGrid, FeatureGrid)):
        return Grid(grid.crs, tuple(grid.transform), grid.width, grid.height)
    raise TypeError(f"not a grid: {grid!r}")


def to_feature_grid(grid: Grid | TokenGrid | FeatureGrid) -> FeatureGrid:
    """The same pixels as a ``FeatureGrid``.

    From a ``Grid`` this is the adapter's ``Grid`` -> ``FeatureGrid`` view
    (``tensor_ee/swin.py::_feature_grid``), with a square-pixel check; from a
    ``TokenGrid`` it is ``feature_interface.abstract_grid`` (origin 0.0, 0.0).
    """
    if isinstance(grid, FeatureGrid):
        return grid
    if isinstance(grid, Grid):
        a, x0, y0 = _square_origin(grid)
        return FeatureGrid(grid.crs, x0, y0, a, grid.height, grid.width)
    if isinstance(grid, TokenGrid):
        return FeatureGrid(grid.crs, 0.0, 0.0, grid.scale, grid.height, grid.width)
    raise TypeError(f"not a grid: {grid!r}")


def to_token_grid(grid: Grid | TokenGrid | FeatureGrid) -> TokenGrid:
    """The same pixels as an abstract ``TokenGrid``.

    A ``TokenGrid`` has no origin field: its origin is always (0, 0). A source
    with any other origin is refused instead of silently relocated (a
    client-side check; nothing reaches the graph).
    """
    if isinstance(grid, TokenGrid):
        return grid
    if isinstance(grid, Grid):
        scale, x0, y0 = _square_origin(grid)
    elif isinstance(grid, FeatureGrid):
        scale, x0, y0 = grid.scale, grid.x0, grid.y0
    else:
        raise TypeError(f"not a grid: {grid!r}")
    if x0 != 0 or y0 != 0:
        raise ValueError("a TokenGrid has origin (0, 0); relabel instead of converting")
    return TokenGrid(grid.height, grid.width, scale, grid.crs)


def check_model_grid(grid: TokenGrid, patch: int = 4) -> TokenGrid:
    """Assert the coupling an abstract model input grid relies on; return the grid.

    The Swin-T224 adapter's model input grid is ``TokenGrid(224, 224, 2.5)``.
    Patch embedding coarsens it by ``patch``, and the encoder builds its stage-1
    token grid as a default ``TokenGrid(rows * 7, cols * 7)`` (scale 10,
    EPSG:3857, origin (0, 0)). The two agree only because 2.5 * 4 equals
    TokenGrid's default scale. This is a client-side check: it adds nothing to
    any graph. (The adapter's own call to it belongs to the model layer.)
    """
    if not isinstance(grid, TokenGrid):
        raise TypeError("the model grid must be a TokenGrid")
    tokens = grid.coarse(patch)
    if tokens != TokenGrid(tokens.height, tokens.width):
        raise ValueError(f"{grid} coarsened by {patch} is {tokens}, not the default-scale "
                         "EPSG:3857 TokenGrid the encoder builds")
    return grid
