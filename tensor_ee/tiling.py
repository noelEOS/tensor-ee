"""TensorEE inference on aligned context windows with disjoint output cores.

The source is one *finite* global tensor. Each tile is compiled on its larger
context window, with the window affine translated from the same global grid.
Context is clamped at the true source boundary, so every intermediate tensor
has the same finite-padding boundary there as a full-image forward pass. Only
artificial interior context edges are protected by the caller-supplied halo.

The halo must cover the model's exact receptive support. Alignment must cover
the deepest striding/pooling lattice. These requirements cannot be inferred
from a random fixture's numerically small distant effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import ee
from torch import nn

from tensor_ee.compiler import (CompiledModel, Grid, Raster, UnsupportedOperation,
                                WeightProvider, compile_model)
from tensor_ee.receptive_field import ReceptiveFieldReport, analyze_receptive_field


@dataclass(frozen=True)
class PixelWindow:
    """A half-open rectangle in integer coordinates of the global input grid."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in (self.x, self.y, self.width, self.height)):
            raise ValueError("PixelWindow coordinates and dimensions must be integers")
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("PixelWindow requires nonnegative origins and positive dimensions")

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def grid(self, global_grid: Grid) -> Grid:
        if self.right > global_grid.width or self.bottom > global_grid.height:
            raise ValueError("PixelWindow is outside the global finite input")
        a, b, c, d, e, f = global_grid.transform
        return Grid(global_grid.crs,
                    (a, b, c + a * self.x, d, e, f + e * self.y),
                    self.width, self.height)


@dataclass(frozen=True)
class Tile:
    """One output core and its larger finite context, in global pixel indices."""

    index: int
    row: int
    column: int
    core: PixelWindow
    context: PixelWindow


@dataclass(frozen=True)
class TilePlan:
    global_grid: Grid
    roi: PixelWindow
    core_size: tuple[int, int]
    halo: tuple[int, int]
    alignment: int
    tiles: tuple[Tile, ...]

    @property
    def roi_grid(self) -> Grid:
        return self.roi.grid(self.global_grid)

    @property
    def vertical_seams(self) -> tuple[int, ...]:
        return tuple(sorted({tile.core.right for tile in self.tiles
                             if tile.core.right < self.roi.right}))

    @property
    def horizontal_seams(self) -> tuple[int, ...]:
        return tuple(sorted({tile.core.bottom for tile in self.tiles
                             if tile.core.bottom < self.roi.bottom}))

    def seam_pixel_count(self, width: int = 1) -> int:
        """Count pixels within ``width`` pixels on each side of internal seams."""
        _positive_integer(width, "seam width")

        def stripe_pixels(seams: tuple[int, ...], start: int, end: int) -> int:
            intervals = sorted((max(start, seam - width), min(end, seam + width))
                               for seam in seams)
            total, previous_end = 0, start
            for left, right in intervals:
                total += max(0, right - max(left, previous_end))
                previous_end = max(previous_end, right)
            return total

        x_pixels = stripe_pixels(self.vertical_seams, self.roi.x, self.roi.right)
        y_pixels = stripe_pixels(self.horizontal_seams, self.roi.y, self.roi.bottom)
        return (x_pixels * self.roi.height + y_pixels * self.roi.width
                - x_pixels * y_pixels)

    def metadata(self) -> dict:
        def rectangle(window: PixelWindow) -> dict:
            return {"x": window.x, "y": window.y,
                    "width": window.width, "height": window.height}

        return {
            "global_dimensions": [self.global_grid.width, self.global_grid.height],
            "global_transform": list(self.global_grid.transform),
            "roi": rectangle(self.roi), "core_size": list(self.core_size),
            "halo": list(self.halo), "alignment": self.alignment,
            "tile_count": len(self.tiles), "core_pixel_count": self.roi.pixels,
            "context_pixel_count_sum": sum(tile.context.pixels for tile in self.tiles),
            "vertical_seams": list(self.vertical_seams),
            "horizontal_seams": list(self.horizontal_seams),
            "tiles": [{"index": tile.index, "row": tile.row, "column": tile.column,
                       "core": rectangle(tile.core), "context": rectangle(tile.context),
                       "context_transform": list(tile.context.grid(self.global_grid).transform)}
                      for tile in self.tiles],
        }


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _pair(value: int | tuple[int, int], name: str, *, allow_zero: bool = False) -> tuple[int, int]:
    try:
        values = (value, value) if type(value) is int else tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer or an integer pair") from error
    if len(values) != 2 or any(type(item) is not int or item < (0 if allow_zero else 1)
                              for item in values):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must contain two {qualifier} integers")
    return values


def plan_tiles(global_grid: Grid, core_size: int | tuple[int, int],
               halo: int | tuple[int, int], *, alignment: int,
               roi: PixelWindow | None = None) -> TilePlan:
    """Partition a finite aligned ROI; clamp contexts to real global boundaries.

    Core sizes are (width, height), halo is (horizontal, vertical). All origins,
    dimensions, and halo widths must be multiples of ``alignment``. A final
    core may be shorter than ``core_size`` while remaining aligned. The caller
    must supply a halo at least as large as the model's exact required support.
    """
    _positive_integer(alignment, "alignment")
    core = _pair(core_size, "core_size")
    margin = _pair(halo, "halo", allow_zero=True)
    region = roi or PixelWindow(0, 0, global_grid.width, global_grid.height)
    region.grid(global_grid)  # Validate actual finite domain before planning.
    aligned = {"global width": global_grid.width, "global height": global_grid.height,
               "ROI x": region.x, "ROI y": region.y, "ROI width": region.width,
               "ROI height": region.height, "core width": core[0], "core height": core[1],
               "horizontal halo": margin[0], "vertical halo": margin[1]}
    for name, value in aligned.items():
        if value % alignment:
            raise ValueError(f"{name} must be a multiple of alignment {alignment}")
    tiles = []
    for row, y in enumerate(range(region.y, region.bottom, core[1])):
        for column, x in enumerate(range(region.x, region.right, core[0])):
            output = PixelWindow(x, y, min(core[0], region.right - x),
                                 min(core[1], region.bottom - y))
            left, top = max(0, x - margin[0]), max(0, y - margin[1])
            right = min(global_grid.width, output.right + margin[0])
            bottom = min(global_grid.height, output.bottom + margin[1])
            context = PixelWindow(left, top, right - left, bottom - top)
            tiles.append(Tile(len(tiles), row, column, output, context))
    return TilePlan(global_grid, region, core, margin, alignment, tuple(tiles))


def seam_mask(plan: TilePlan, width: int = 1) -> ee.Image:
    """Mask an exact union of bands around internal core boundaries."""
    _positive_integer(width, "seam width")
    coordinates = ee.Image.pixelCoordinates(plan.global_grid.projection())
    x, y = coordinates.select("x"), coordinates.select("y")
    mask = ee.Image.constant(0).reproject(plan.global_grid.projection())
    for seam in plan.vertical_seams:
        mask = mask.Or(x.gte(seam - width).And(x.lt(seam + width)))
    for seam in plan.horizontal_seams:
        mask = mask.Or(y.gte(seam - width).And(y.lt(seam + width)))
    return mask.And(plan.roi_grid.mask()).rename("seam")


@dataclass(frozen=True)
class CompiledTiledModel(Raster):
    plan: TilePlan | None = None
    tiles: tuple[CompiledModel, ...] = ()
    nodes: dict[str, Raster] = field(default_factory=dict)
    coverage: ee.Image | None = None
    planned_coverage: ee.Image | None = None
    receptive_field: ReceptiveFieldReport | None = None


def compile_tiled_model(model: nn.Module, image: ee.Image,
                        input_bands: Sequence[str], plan: TilePlan, *,
                        weight_provider: WeightProvider | None = None,
                        output_nodes: Sequence[str] = ()) -> CompiledTiledModel:
    """Compile every context in EE, retain disjoint cores, and mosaic in EE.

    One shared weight provider serves all tiles, so an asset provider can reuse
    its arrays. Selected diagnostic output nodes must have input resolution.
    The returned grid is the *global* grid; its image is masked to the ROI.
    Every compiled intermediate uses float32 and clipped finite footprints.
    """
    if not plan.tiles:
        raise ValueError("TilePlan must contain at least one core")
    support = analyze_receptive_field(model)
    if plan.alignment % support.alignment:
        raise ValueError(f"Tile alignment must be a multiple of model alignment {support.alignment}")
    for tile in plan.tiles:
        context, core = tile.context, tile.core
        for value in (core.x, core.y, core.width, core.height,
                      context.x, context.y, context.width, context.height):
            if value % support.alignment:
                raise ValueError(f"Tile origins and dimensions require model alignment {support.alignment}")
        artificial_margins = []
        if context.x > 0:
            artificial_margins.append(core.x - context.x)
        if context.y > 0:
            artificial_margins.append(core.y - context.y)
        if context.right < plan.global_grid.width:
            artificial_margins.append(context.right - core.right)
        if context.bottom < plan.global_grid.height:
            artificial_margins.append(context.bottom - core.bottom)
        if any(margin < support.safe_halo for margin in artificial_margins):
            raise ValueError(f"Artificial context boundaries require halo at least {support.safe_halo}")
    compiled_tiles = tuple(compile_model(
        model, image, input_bands, tile.context.grid(plan.global_grid),
        weight_provider=weight_provider, include_unfused_nodes=False,
        execution_dtype="float32", clip_footprints=True)
        for tile in plan.tiles)

    def core_images(rasters: Sequence[Raster]) -> list[ee.Image]:
        bands = rasters[0].bands
        pieces = []
        for tile, raster in zip(plan.tiles, rasters, strict=True):
            if raster.grid != tile.context.grid(plan.global_grid):
                raise UnsupportedOperation("Tiled output must preserve the input spatial grid")
            if raster.bands != bands:
                raise UnsupportedOperation("All tiled outputs must have identical bands")
            core_grid = tile.core.grid(plan.global_grid)
            pieces.append(raster.image.select(list(bands)).clip(core_grid.geometry())
                          .updateMask(core_grid.mask()))
        return pieces

    def mosaic(rasters: Sequence[Raster]) -> Raster:
        result = (ee.ImageCollection.fromImages(core_images(rasters)).mosaic().rename(list(rasters[0].bands))
                  .setDefaultProjection(plan.global_grid.projection())
                  .updateMask(plan.roi_grid.mask()).toFloat())
        return Raster(result, rasters[0].bands, plan.global_grid)

    output = mosaic(compiled_tiles)
    nodes = {}
    for name in output_nodes:
        if any(name not in tile.nodes for tile in compiled_tiles):
            raise ValueError(f"Requested tiled output node {name!r} is absent")
        nodes[name] = mosaic([tile.nodes[name] for tile in compiled_tiles])
    masks = [ee.Image.constant(1).rename("tile_coverage")
             .reproject(plan.global_grid.projection())
             .updateMask(tile.core.grid(plan.global_grid).mask()) for tile in plan.tiles]
    planned_coverage = (ee.ImageCollection.fromImages(masks).sum().rename("planned_tile_coverage")
                        .setDefaultProjection(plan.global_grid.projection())
                        .updateMask(plan.roi_grid.mask()))
    actual_masks = [piece.mask().reduce(ee.Reducer.min()).unmask(0, sameFootprint=False)
                    .rename("tile_coverage").reproject(plan.global_grid.projection())
                    for piece in core_images(compiled_tiles)]
    coverage = (ee.ImageCollection.fromImages(actual_masks).sum().rename("tile_coverage")
                .setDefaultProjection(plan.global_grid.projection()).updateMask(plan.roi_grid.mask()))
    return CompiledTiledModel(output.image, output.bands, output.grid,
                              plan, compiled_tiles, nodes, coverage, planned_coverage, support)
