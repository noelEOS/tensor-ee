"""Stage boundaries: how saved images enter a stage's graph and how its outputs leave it.

A staged model runs as stages whose inputs and outputs are saved boundaries, images that one
stage exports and the next reads . These are the
conventions of the fixed SegFormer-B0 and of its sizing tool, moved here so that the
block-graph builder (``tensor_ee/converter/build.py``) emits the same boundary expressions. They
are layout transforms, below the blocks: the model imports them without loading the converter.

- ``grid_view``: the grid a boundary or a request is pinned on; a converter ``Grid`` is viewed
  as the ``FeatureGrid`` of the same pixels, other kinds are themselves (from
  ``tools/segformer_b0_request_plan.py``, ``_grid_view``);
- ``pin_boundary``: a boundary image reprojected to its grid and masked to the grid's domain
  (that tool's ``_boundary_image``, for literal and asset sources alike);
- ``saved_image``: a saved boundary read from an asset, its bands selected, pinned (that
  tool's ``_boundary_image`` with ``asset-reference`` sources);
- ``boundary_windows``: a dense boundary (bands ``c0`` ..) as a W tensor (``_boundary_window``);
- ``boundary_raster``: a W tensor as dense bands ``c0`` .. on its own token grid, the form a
  stage exports (the stage outputs of that tool's ``build_cut``);
- ``relabel_to_windows``: a north-up raster's pixels relabeled onto a model's abstract token
  grid, then windowed (the model input interface,
  ``tensor_ee.models.segformer_b0.decoder.input_windows``);
- ``relabel_from_windows``: the inverse, a W output relabeled back onto a source grid (the end
  of ``segformer_b0()`` and of ``build_cut``'s decoder cut).

The fixed model keeps its checks (bands ``b0``, ``b1``; a 224 x 224 chip) and calls these;
there is one implementation of each.

These move values only: select, rename, reproject, ``changeProj`` and masks, with no
arithmetic and no cast of their own (``from_dense`` and ``to_dense`` are the layouts'
converters; ``to_dense`` keeps its float32 bands). A relabel moves pixels one to one, so the
two grids must have the same height and width; ``changeProj`` relabels values where
``reproject`` alone would resample them. Graph construction only: nothing is evaluated.

**Validated.** Live only as the SegFormer-B0's input interface, inverse relabel and saved
boundaries (W7, a 224 x 224 chip at 2.5 m; docs/evidence/segformer_b0-fullmodel-live.json).
"""

from __future__ import annotations

from ..core.errors import reject
from ..core.grid import Grid, TokenGrid, to_feature_grid, to_grid
from ..core.tensor import WINDOW, Raster, WindowTensor, band_names
from .window import from_dense, to_dense


def grid_view(grid):
    """Expose request methods without relabeling or moving any source pixels."""
    return to_feature_grid(grid) if isinstance(grid, Grid) else grid


def pin_boundary(ee, image, grid):
    """``image`` reprojected to ``grid`` and masked to its domain (seen through ``grid_view``)."""
    view = grid_view(grid)
    return image.reproject(view.projection(ee)).updateMask(view.domain(ee))


def saved_image(ee, asset_id: str, bands, grid):
    """The saved boundary ``asset_id``: its ``bands``, in order, pinned to ``grid``."""
    return pin_boundary(ee, ee.Image(asset_id).select(list(bands)), grid)


def boundary_windows(ee, image, grid, channels: int, *, window: int = WINDOW) -> WindowTensor:
    """A dense boundary image (bands ``c0`` ..) on the token grid ``grid`` as a W tensor."""
    return WindowTensor(from_dense(ee, grid, image, channels, window=window), grid, channels, window)


def boundary_raster(ee, x: WindowTensor) -> Raster:
    """``x`` as float32 bands ``c0`` .. on its own token grid: the image a stage exports."""
    return Raster(to_dense(ee, x.grid, x.image, x.channels, window=x.window), tuple(band_names(x.channels)), x.grid)


def relabel_to_windows(ee, raster: Raster, grid: TokenGrid, *, window: int = WINDOW) -> WindowTensor:
    """Relabel a north-up ``raster`` onto the abstract token grid ``grid`` and window it.

    Its bands, in order, become ``c0`` ..; the raster's own grid may be georeferenced anywhere.
    Masks are retained (no implicit zero fill); fully valid finite float32 values are the
    caller's precondition, which cannot be checked without evaluation.
    """
    if not isinstance(grid, TokenGrid):
        reject("a raster is relabeled onto a model's TokenGrid")
    source = to_grid(raster.grid)
    if (source.height, source.width) != (grid.height, grid.width):
        reject(f"a {source.height} x {source.width} raster cannot be relabeled onto a "
               f"{grid.height} x {grid.width} grid")
    channels = len(raster.bands)
    # changeProj relabels the source's existing pixel values; reproject alone would resample them.
    image = raster.image.select(list(raster.bands)).rename(band_names(channels)).reproject(source.projection())
    image = image.changeProj(source.projection(), grid.projection(ee)).updateMask(grid.domain(ee))
    return WindowTensor(from_dense(ee, grid, image, channels, window=window), grid, channels, window)


def relabel_from_windows(ee, x: WindowTensor, grid, bands) -> Raster:
    """Relabel the W tensor ``x`` back onto the source ``grid`` as the named ``bands``."""
    target, bands = to_grid(grid), tuple(bands)
    if len(bands) != x.channels:
        reject(f"{len(bands)} band names for {x.channels} channels")
    if (target.height, target.width) != (x.grid.height, x.grid.width):
        reject(f"a {x.grid.height} x {x.grid.width} map cannot be relabeled onto a "
               f"{target.height} x {target.width} grid")
    dense = to_dense(ee, x.grid, x.image, x.channels, window=x.window)
    image = dense.changeProj(x.grid.projection(ee), target.projection()).reproject(target.projection())
    image = image.rename(list(bands)).updateMask(target.domain(ee))
    return Raster(image, bands, target)
