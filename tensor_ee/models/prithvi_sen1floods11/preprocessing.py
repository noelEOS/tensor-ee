"""The source's input preparation for Sentinel-2 L1C, composed from library blocks.

Prithvi-EO-1.0 Sen1Floods11 reads six bands of the 13-band L1C stack (indices 1, 2, 3, 8, 11
and 12: B2, B3, B4, B8A, B11, B12), replaces nodata with 0, multiplies by 1e-4 and
normalizes each band with the statistics of its configuration, all in float32. Here:

- ``from_image``: the six bands in that order, missing pixels zero-filled (``unmask(0)``),
  pinned to the crop's grid. Its ``reproject`` resamples the 20 m bands (B8A, B11, B12) onto a
  10 m grid by nearest neighbour, Earth Engine's default; the source's resampling is
  undocumented, which matters for accuracy only.
- ``standardize``: ``(x * 1e-4 - mean) / std`` per band, each operation rounded to float32,
  which reproduces the source's float32 recipe bit for bit (validated live at this setting).
- ``to_input_grid``: the prepared crop relabeled onto the model's abstract ``INPUT_GRID``
  (``changeProj``: values moved, never resampled), as ``LibraryBackend.patch`` relabels any
  georeferenced crop, so that a staged run reads it as its saved input.

No arithmetic is added here: the blocks do it. Nothing is evaluated.
"""
from __future__ import annotations

import numpy as np

from ...blocks.raster.normalization import StandardizeConstants, standardize
from ...core.grid import to_grid
from ...core.tensor import Raster
from ...layouts.raster import from_image
from .model import BANDS, crops_of, region_grid

S2_BANDS = ("B2", "B3", "B4", "B8A", "B11", "B12")
"""The source's six bands, in its order (indices 1, 2, 3, 8, 11, 12 of the L1C stack)."""

SCALE = 1e-4
MEANS = (0.14245495, 0.13921481, 0.12434631, 0.31420089, 0.20743526, 0.12046503)
STDS = (0.04036231, 0.04186983, 0.05267646, 0.0822221, 0.06834774, 0.05294205)
"""The configuration's per-band statistics, as it states them; used as float32."""


def raw_bands(image, grid) -> Raster:
    """The six source bands of an L1C image on ``grid``, zero-filled, float32 (``c0``..)."""
    return from_image(image, S2_BANDS, grid)


def input_constants(bands) -> StandardizeConstants:
    return StandardizeConstants(np.array(MEANS, np.float32), np.array(STDS, np.float32),
                                bands=bands, scale=SCALE)


def prepare(image, grid) -> Raster:
    """An L1C ``ee.Image`` as the model's input on ``grid`` (whole 224 x 224 crops)."""
    crops_of(grid)
    raw = raw_bands(image, grid)
    return standardize(raw, input_constants(raw.bands))


def to_input_grid(ee, raster: Raster) -> Raster:
    """``raster`` relabeled onto ``region_grid`` of its crops, as bands ``b0``..``b5``."""
    grid = region_grid(*crops_of(raster.grid))
    source = to_grid(raster.grid)
    image = raster.image.select(list(raster.bands)).rename(list(BANDS[:len(raster.bands)]))
    image = image.reproject(source.projection()).changeProj(source.projection(), grid.projection(ee))
    return Raster(image.updateMask(grid.domain(ee)), tuple(BANDS[:len(raster.bands)]), grid)
