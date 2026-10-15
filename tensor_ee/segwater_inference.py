"""Fixed224 SegWater inference on a rectangular Sentinel-1 Earth Engine grid.

Every chip is an independent 224 x 224 model input. The compiler zero-pads
outside *each intermediate tensor*, including after biased layers. Adjacent
chips have no overlap or halo, so this is not a seamless full-scene forward
pass. Their origins are 224 pixels apart and retain the encoder's 32-pixel
sampling phase relative to the supplied grid's upper-left corner.

At the right and bottom AOI edges, a full 224-pixel chip is compiled using the
raw scene beyond the AOI before its output is cropped. The AOI is therefore an
output boundary, not an input-padding boundary. Callers should supply the raw
scene rather than clip it to the AOI first. Missing source pixels in a chip
are filled with exact zero *after* normalization, and predictions at those
pixel centers remain invalid.

This module constructs native EE graphs only. It does not initialize EE,
download pixels, submit requests, or call an external inference endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Sequence

import ee
from torch import nn

# compile_model is bound here: predict_segwater looks it up in this module (contract 2.3).
from tensor_ee.compiler import WeightProvider, compile_model
from tensor_ee.core.errors import UnsupportedOperation
from tensor_ee.core.grid import Grid
from tensor_ee.segwater_preprocessing import BANDS, preprocess_db_ee
from tensor_ee.segwater_pytorch import INPUT_SHAPE


CHIP_SIZE = INPUT_SHAPE[-1]
MODEL_ALIGNMENT = 32
OUTPUT_BANDS = ("logit_0", "logit_1", "water_probability", "water_mask", "coverage")
FLOAT32_MAX = 3.4028234663852886e38


def _validate_grid(grid: Grid) -> None:
    if not isinstance(grid, Grid):
        raise ValueError("SegWater AOI must be a Grid")
    if type(grid.width) is not int or type(grid.height) is not int:
        raise ValueError("SegWater grid dimensions must be positive integers")
    match = re.fullmatch(r"EPSG:(326|327)(\d{2})", grid.crs)
    if match is None or not 1 <= int(match[2]) <= 60:
        raise ValueError("SegWater requires a WGS84 UTM CRS (EPSG:32601..32660 or 32701..32760)")
    a, b, _, d, e, _ = grid.transform
    if (a, b, d, e) != (10, 0, 0, -10):
        raise ValueError("SegWater requires a north-up 10 m grid")


def _grid_metadata(grid: Grid) -> dict:
    return {"crs": grid.crs, "transform": list(grid.transform),
            "width": grid.width, "height": grid.height}


@dataclass(frozen=True)
class SegWaterTile:
    """One disjoint output crop and its full224 inference grid.

    ``x`` and ``y`` are pixel offsets in the AOI grid. ``width`` and ``height``
    describe the retained output; ``grid`` always describes all 224 x 224
    input pixels, even when that context extends beyond the AOI.
    """

    index: int
    row: int
    column: int
    x: int
    y: int
    width: int
    height: int
    grid: Grid

    @property
    def output_grid(self) -> Grid:
        return Grid(self.grid.crs, self.grid.transform, self.width, self.height)

    def metadata(self) -> dict:
        return {"index": self.index, "row": self.row, "column": self.column,
                "x": self.x, "y": self.y, "width": self.width, "height": self.height,
                "chip_grid": _grid_metadata(self.grid)}


def plan_segwater_tiles(grid: Grid) -> tuple[SegWaterTile, ...]:
    """Cover an arbitrary positive rectangular AOI with disjoint fixed chips.

    AOI dimensions need not be divisible by 224 or 32. A final output crop can
    have any positive size; its model input remains exactly 224 x 224.
    """
    _validate_grid(grid)
    a, b, c, d, e, f = grid.transform
    tiles = []
    for row, y in enumerate(range(0, grid.height, CHIP_SIZE)):
        for column, x in enumerate(range(0, grid.width, CHIP_SIZE)):
            chip_grid = Grid(grid.crs, (a, b, c + a * x, d, e, f + e * y),
                             CHIP_SIZE, CHIP_SIZE)
            tiles.append(SegWaterTile(len(tiles), row, column, x, y,
                                     min(CHIP_SIZE, grid.width - x),
                                     min(CHIP_SIZE, grid.height - y), chip_grid))
    return tuple(tiles)


@dataclass(frozen=True)
class SegWaterPrediction:
    """An EE mosaic on the supplied AOI grid, with reproducible chip metadata.

    All five bands use float32. ``coverage`` is an unmasked 0/1 image inside
    the AOI: one requires a valid source center in *both* raw input bands and
    valid, finite values in *both* logits. The other bands are masked when
    coverage is zero. Everything is masked outside the rectangular AOI.

    With ``retain_inputs=True``, ``raw_input`` and ``normalized_input`` retain
    the images used to construct chip inputs on the common 10 m grid. They
    remain available beyond the AOI for inspecting partial-chip context;
    ``normalized_input`` has global zero fill at missing source pixels.
    """

    image: ee.Image
    grid: Grid
    tiles: tuple[SegWaterTile, ...]
    threshold: float = 0.5
    raw_input: ee.Image | None = None
    normalized_input: ee.Image | None = None

    @property
    def bands(self) -> tuple[str, ...]:
        return OUTPUT_BANDS

    def metadata(self) -> dict:
        return {
            "grid": _grid_metadata(self.grid), "bands": list(OUTPUT_BANDS),
            "dtype": "float32", "chip_size": [CHIP_SIZE, CHIP_SIZE],
            "alignment": MODEL_ALIGNMENT, "tile_count": len(self.tiles),
            "halo": 0, "overlap": 0, "independent_chip_inference": True,
            "padding": "zero outside each finite model-layer tensor",
            "missing_input": "shared raw VV/VH mask; exact zero in normalized space",
            "partial_chip_context": "full224 raw-scene context beyond AOI; crop output to AOI",
            "coverage": "valid raw VV/VH center and two valid finite logits",
            "water_class": 1, "water_threshold": self.threshold,
            "tiles": [tile.metadata() for tile in self.tiles],
        }


def _valid_logits(logits: ee.Image) -> ee.Image:
    masks = logits.mask().reduce(ee.Reducer.min()).gt(0)
    finite = logits.eq(logits).And(logits.abs().lte(FLOAT32_MAX)).reduce(ee.Reducer.min())
    return masks.And(finite).unmask(0, sameFootprint=False)


def _water_probability(logits: ee.Image) -> ee.Image:
    """Stable two-class softmax for class 1, with float32 arithmetic."""
    delta = (logits.select("logit_1").toFloat()
             .subtract(logits.select("logit_0").toFloat()).toFloat())
    exponential = delta.abs().multiply(-1).toFloat().exp().toFloat()
    one = ee.Image.constant(1).toFloat()
    denominator = exponential.add(one).toFloat()
    positive = one.divide(denominator).toFloat()
    negative = exponential.divide(denominator).toFloat()
    return positive.where(delta.lt(0), negative).toFloat().rename("water_probability")


def predict_segwater(scene: ee.Image, grid: Grid, model: nn.Module,
                     weight_provider: WeightProvider | None = None, *,
                     input_bands: Sequence[str] = BANDS, threshold: float = 0.5,
                     retain_inputs: bool = False) -> SegWaterPrediction:
    """Normalize raw S1 dB, compile fixed224 chips, and mosaic outputs in EE.

    Pass ``load_segwater_checkpoint(...).compilable_model`` and, for uploaded
    weights, a ``MatchedWeightProvider`` that verifies its effective folded
    coefficients. The model must retain its ``(2,224,224)`` input contract and
    return two full-resolution logits. Omitting the provider embeds the model's
    float32 coefficients using the existing compiler.

    ``grid`` alone defines the rectangular AOI. Input names identify VV then
    VH. Masks are intersected before normalized missing values are filled, so
    a pixel missing either input receives zeros in both normalized channels.
    Numeric zero or -55 dB is never interpreted as a missing-value sentinel.
    The default water rule is class-1 probability >= 0.5, including exact ties.

    Partial chips use raw scene context beyond the AOI. Interior chip borders
    use independent finite-model padding and can create seams; this API makes
    no overlap, halo, or larger-spatial-adapter parity claim.
    """
    tiles = plan_segwater_tiles(grid)
    fixed_shape = getattr(model, "tensor_ee_input_shape", None)
    if fixed_shape is None or tuple(fixed_shape) != INPUT_SHAPE[1:]:
        raise UnsupportedOperation("SegWater requires the fixed CHW input contract (2, 224, 224)")
    input_bands = tuple(input_bands)
    if (len(input_bands) != 2 or any(not isinstance(band, str) or not band for band in input_bands)
            or len(set(input_bands)) != 2):
        raise ValueError("input_bands must contain distinct VV and VH names in that order")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be a finite number in [0, 1]")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite number in [0, 1]")
    if type(retain_inputs) is not bool:
        raise ValueError("retain_inputs must be a boolean")

    projection = grid.projection()
    raw = scene.select(list(input_bands)).toFloat().reproject(projection)
    source_valid = (raw.mask().reduce(ee.Reducer.min()).gt(0)
                    .unmask(0, sameFootprint=False).reproject(projection))
    normalized = (preprocess_db_ee(raw, input_bands, masked_policy="preserve")
                  .updateMask(source_valid).unmask(0, sameFootprint=False)
                  .toFloat().reproject(projection))
    pieces = []
    for tile in tiles:
        compiled = compile_model(model, normalized, input_bands, tile.grid,
                                 weight_provider=weight_provider,
                                 include_unfused_nodes=False,
                                 execution_dtype="float32", clip_footprints=True)
        if compiled.grid != tile.grid or len(compiled.bands) != 2:
            raise UnsupportedOperation("SegWater must return two logits on the full224 input grid")
        logits = compiled.image.select(list(compiled.bands)).rename(list(OUTPUT_BANDS[:2])).toFloat()
        valid = source_valid.And(_valid_logits(logits)).unmask(0, sameFootprint=False)
        logits = logits.updateMask(valid).toFloat()
        probability = _water_probability(logits).updateMask(valid).toFloat()
        water = probability.gte(float(threshold)).updateMask(valid).rename("water_mask").toFloat()
        coverage = valid.rename("coverage").toFloat()
        output_grid = tile.output_grid
        pieces.append(ee.Image.cat([logits, probability, water, coverage])
                      .clip(output_grid.geometry()).updateMask(output_grid.mask()).toFloat())

    mosaic = (ee.ImageCollection.fromImages(pieces).mosaic().select(list(OUTPUT_BANDS))
              .setDefaultProjection(projection).clip(grid.geometry())
              .updateMask(grid.mask()).toFloat())
    return SegWaterPrediction(mosaic, grid, tiles, float(threshold),
                              raw if retain_inputs else None,
                              normalized if retain_inputs else None)
