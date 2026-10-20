"""R-layout converters: pinning, zero extension, stride sampling, embedded fixtures.

Moved from ``tensor_ee.compiler`` (``_project``, ``_finish``,
``_zero_extended``, ``_stride_sample``), ``tensor_ee.primitives``
(``from_image``, ``_float32``), ``upernet.earth_engine.feature_transport``
(``stride_sample``, here ``stride_sample_phase``) and
``upernet.earth_engine.feature_interface`` (``embedded_raster``). The old
``tensor_ee`` names are re-exports of these objects; the research modules are
archived at the tag ``research-archive-2026-10-12`` (docs/history/README.md).

The two stride samplers emit different expressions (``stride_sample_2`` probe)
and stay two named transforms; merging them would change graphs (G2, deferred).
"""

from __future__ import annotations

from typing import Sequence

import ee
import numpy as np

from ..core.grid import FeatureGrid, Grid
from ..core.tensor import Raster, band_names


def project(image: ee.Image, grid: Grid, clip_footprints: bool = False) -> ee.Image:
    if clip_footprints:
        # Clip before forced reprojection, rather than only masking its result:
        # this allows EE to avoid requesting the full upstream tile footprint.
        image = image.clip(grid.geometry())
    return image.reproject(grid.projection())


def finish(image: ee.Image, bands: Sequence[str], grid: Grid,
           clip_footprints: bool = False) -> Raster:
    # Explicit reprojection is essential: EE otherwise requests upstream
    # neighborhoods at the caller's output scale instead of this layer's scale.
    image = project(image.rename(list(bands)), grid, clip_footprints).updateMask(grid.mask())
    return Raster(image, tuple(bands), grid)


def zero_extended(raster: Raster) -> ee.Image:
    # False expands even a finite asset footprint to the global zero image.
    return raster.image.unmask(0, sameFootprint=False).reproject(raster.grid.projection())


def stride_sample(image: ee.Image, bands: Sequence[str], grid: Grid,
                  stride: int, clip_footprints: bool = False) -> Raster:
    """Sample indices 0,s,2s,... and retain a canonical tensor grid.

    EE's same-corner coarse grid would sample centers at input indices 1,3,...
    for stride two. Move its corner by half an input pixel to sample 0,2,...,
    then ``changeProj`` moves those *values* onto canonical tensor coordinates.
    A plain reproject at the second step would resample and lose the alignment.

    The stride is a parameter, but only strides 1 and 2 are evaluated.
    """
    if stride == 1:
        return finish(image, bands, grid, clip_footprints)
    width = (grid.width + stride - 1) // stride
    height = (grid.height + stride - 1) // stride
    canonical = grid.resized(stride, width, height)
    a, b, c, d, e, f = canonical.transform
    sample = Grid(grid.crs,
                  (a, b, c - (stride - 1) * grid.transform[0] / 2,
                   d, e, f - (stride - 1) * grid.transform[4] / 2),
                  width, height)
    sampled = finish(image, bands, sample, clip_footprints)
    relabeled = sampled.image.changeProj(sample.projection(), canonical.projection())
    return finish(relabeled, bands, canonical, clip_footprints)


def _float32(raster: Raster) -> Raster:
    return Raster(raster.image.toFloat(), raster.bands, raster.grid)


def from_image(image: ee.Image, bands: Sequence[str], grid: Grid, *,
               clip_footprints: bool = False) -> Raster:
    """Select ordered channels, zero-fill missing values, and fix the grid.

    Output channels are named c0, c1, ... . Outside this finite rectangle the
    tensor is masked; convolutions supply zero padding at each layer boundary.
    """
    bands = tuple(bands)
    if not bands or any(not isinstance(band, str) or not band for band in bands):
        raise ValueError("bands must contain nonempty names")
    if len(set(bands)) != len(bands):
        raise ValueError("bands must be unique")
    image = image.select(list(bands)).unmask(0, sameFootprint=False)
    return _float32(finish(
        image, tuple(f"c{i}" for i in range(len(bands))), grid, clip_footprints))


def stride_sample_phase(ee, image, fine: FeatureGrid, factor: int, a: int = 0, b: int = 0):
    """Image on fine.coarse(factor) whose pixel (I, J) is fine pixel (factor I + a,
    factor J + b): sampled exactly at that fine pixel's centre, then relabelled
    (the original attention implementation's stride sample on a georeferenced grid)."""
    s, w = fine.scale, factor
    sample = ee.Projection(fine.crs, [s * w, 0, fine.x0 + s * (b + 0.5 - w / 2), 0, -s * w,
                                      fine.y0 - s * (a + 0.5 - w / 2)])
    coarse = fine.coarse(factor)
    return image.reproject(fine.projection(ee)).reproject(sample).changeProj(
        sample, coarse.projection(ee))


def embedded_raster(ee, grid: FeatureGrid, values: np.ndarray):
    """Embed an exact float32 [H, W, C] raster as constants on the grid, masked outside
    (fixtures only: every value is in the request)."""
    coords = ee.Image.pixelCoordinates(grid.projection(ee)).floor()
    index = ee.Image.cat([coords.select("y").clamp(0, grid.height - 1),
                          coords.select("x").clamp(0, grid.width - 1)]).toInt()
    values = np.asarray(values, dtype=np.float32)
    # The float32 encoder lives in core, which layouts may import.
    from ..core.precision import encode_float32
    bands = [ee.Image(ee.Array(encode_float32(values[:, :, c])).float()).arrayGet(index).toFloat()
             for c in range(values.shape[2])]
    return ee.Image.cat(bands).rename(band_names(values.shape[2])).updateMask(grid.domain(ee))
