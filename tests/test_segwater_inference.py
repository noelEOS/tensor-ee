"""Fixed-chip planning, offline EE graphs, and scalar mask-policy checks."""

import json
from unittest.mock import patch

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest
from torch import nn

from tensor_ee.compiler import CompiledModel, Grid, UnsupportedOperation, compile_model
from tensor_ee.segwater_inference import (
    CHIP_SIZE, OUTPUT_BANDS, plan_segwater_tiles, predict_segwater,
)
from tensor_ee.segwater_preprocessing import preprocess_db_numpy


def grid(width=224, height=224, *, crs="EPSG:32651", spacing=10):
    return Grid(crs, (spacing, 0, 500000, 0, -spacing, 2600000), width, height)


def fixed_model(out_channels=2):
    model = nn.Sequential(nn.Conv2d(2, 3, 3, padding=1, bias=False),
                          nn.BatchNorm2d(3), nn.ReLU(), nn.Conv2d(3, out_channels, 1)).eval()
    model.tensor_ee_input_shape = (2, 224, 224)
    return model


@pytest.mark.parametrize("width,height", [(1, 1), (223, 32), (224, 224), (225, 225),
                                         (448, 448), (449, 300)])
def test_disjoint_cores_cover_every_aoi_pixel_once_with_full224_contexts(width, height):
    target = grid(width, height)
    tiles = plan_segwater_tiles(target)
    coverage = np.zeros((height, width), dtype=np.int8)
    for index, tile in enumerate(tiles):
        coverage[tile.y:tile.y + tile.height, tile.x:tile.x + tile.width] += 1
        assert tile.index == index
        assert tile.x == tile.column * CHIP_SIZE and tile.y == tile.row * CHIP_SIZE
        assert tile.x % 32 == tile.y % 32 == 0
        assert (tile.grid.width, tile.grid.height) == (224, 224)
        assert tile.grid.transform == (10, 0, 500000 + 10 * tile.x,
                                       0, -10, 2600000 - 10 * tile.y)
        assert (tile.output_grid.width, tile.output_grid.height) == (tile.width, tile.height)
    assert np.all(coverage == 1)
    assert sum(tile.width * tile.height for tile in tiles) == width * height


def test_last_chip_context_extends_beyond_aoi_before_its_output_crop():
    tile = plan_segwater_tiles(grid(225, 230))[-1]
    assert (tile.x, tile.y, tile.width, tile.height) == (224, 224, 1, 6)
    assert tile.grid.transform == tile.output_grid.transform
    metadata = tile.metadata()
    assert metadata["chip_grid"]["width"] == metadata["chip_grid"]["height"] == 224
    assert metadata["width"] == 1 and metadata["height"] == 6


@pytest.mark.parametrize("crs,spacing", [("EPSG:3857", 10), ("EPSG:4326", 10),
                                        ("EPSG:32600", 10), ("EPSG:32661", 10),
                                        ("EPSG:32761", 10), ("EPSG:32651", 20)])
def test_inference_grid_requires_wgs84_utm_and_ten_meters(crs, spacing):
    with pytest.raises(ValueError):
        plan_segwater_tiles(grid(crs=crs, spacing=spacing))


def test_southern_utm_grid_is_supported_and_bool_dimensions_are_rejected():
    assert plan_segwater_tiles(grid(crs="EPSG:32760"))[0].grid.crs == "EPSG:32760"
    with pytest.raises(ValueError, match="integers"):
        plan_segwater_tiles(grid(True, 224))


@pytest.mark.parametrize("shape", [None, (2, 256, 256), (3, 224, 224)])
def test_missing_or_generalized_model_contract_is_rejected_before_ee_graphs(shape):
    model = fixed_model()
    model.tensor_ee_input_shape = shape
    with pytest.raises(UnsupportedOperation, match="fixed CHW"):
        predict_segwater(None, grid(), model)


@pytest.mark.parametrize("options", [
    {"input_bands": ("VV",)}, {"input_bands": ("VV", "VV")},
    {"input_bands": ("", "VH")}, {"input_bands": ("VV", None)},
    {"threshold": -0.1}, {"threshold": 1.1}, {"threshold": float("nan")},
    {"threshold": float("inf")}, {"threshold": True}, {"threshold": "0.5"},
    {"retain_inputs": 1},
])
def test_invalid_options_are_rejected_before_ee_graphs(options):
    with pytest.raises(ValueError):
        predict_segwater(None, grid(), fixed_model(), **options)


def algorithm_names(value):
    if isinstance(value, dict):
        result = [value["functionName"]] if "functionName" in value else []
        return result + [name for item in value.values() for name in algorithm_names(item)]
    if isinstance(value, list):
        return [name for item in value for name in algorithm_names(item)]
    return []


class TestSegWaterInferenceGraph(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.scene = ee.Image.constant([-20, -30, 99]).rename(["VV", "VH", "other"])
        self.model = fixed_model()

    def test_graph_uses_native_inference_mosaic_and_stable_float32_probability(self):
        result = predict_segwater(self.scene, grid(225, 225), self.model, retain_inputs=True)
        names = algorithm_names(json.loads(result.image.serialize()))
        for name in ("ImageCollection.mosaic", "Image.matrixMultiply", "Image.clip",
                     "Image.setDefaultProjection", "Image.mask", "Image.toFloat",
                     "Image.mod", "Image.exp", "Image.where"):
            self.assertIn(name, names)
        self.assertFalse(any("predict" in name.lower() for name in names))
        self.assertEqual(result.grid, grid(225, 225))
        self.assertEqual(result.bands, OUTPUT_BANDS)
        self.assertEqual(len(result.tiles), 4)
        self.assertIsNotNone(result.raw_input)
        self.assertIsNotNone(result.normalized_input)
        # The scene context must survive until the compiler clips each full
        # chip; a clip here would remove raw context of partial final chips.
        self.assertNotIn("Image.clip", result.normalized_input.serialize())
        metadata = result.metadata()
        self.assertEqual(metadata["halo"], 0)
        self.assertEqual(metadata["overlap"], 0)
        self.assertTrue(metadata["independent_chip_inference"])
        self.assertEqual(metadata["dtype"], "float32")
        json.dumps(metadata)

    def test_every_chip_uses_fixed224_float32_and_shared_effective_weight_provider(self):
        requested, captured = [], []

        def provider(key, effective):
            requested.append(key)
            width = effective.weight.numel() // effective.out_channels
            weights = ee.Array([[0.0]]).float().repeat(0, effective.out_channels).repeat(1, width)
            bias = ee.Array([[0.0]]).float().repeat(0, effective.out_channels)
            return weights, bias

        def compile_capture(model, image, bands, chip_grid, **options):
            captured.append((image, tuple(bands), chip_grid, options))
            return compile_model(model, image, bands, chip_grid, **options)

        with patch("tensor_ee.segwater_inference.compile_model", side_effect=compile_capture):
            result = predict_segwater(self.scene, grid(225, 225), self.model, provider)
        self.assertEqual(len(captured), 4)
        self.assertEqual(len(requested), 8)  # one folded Conv/BN and one logits head per chip
        self.assertEqual(len(set(requested)), 2)
        for index, (image, bands, chip_grid, options) in enumerate(captured):
            self.assertIs(image, captured[0][0])
            self.assertEqual(bands, ("VV", "VH"))
            self.assertEqual(chip_grid, result.tiles[index].grid)
            self.assertIs(options["weight_provider"], provider)
            self.assertEqual(options["execution_dtype"], "float32")
            self.assertFalse(options["include_unfused_nodes"])
            self.assertTrue(options["clip_footprints"])
        self.assertIsNone(result.raw_input)
        self.assertIsNone(result.normalized_input)

    def test_two_full_resolution_logits_are_required(self):
        with self.assertRaisesRegex(UnsupportedOperation, "two logits"):
            predict_segwater(self.scene, grid(), fixed_model(out_channels=1))
        downsampled = nn.Sequential(nn.Conv2d(2, 2, 3, padding=1, stride=2)).eval()
        downsampled.tensor_ee_input_shape = (2, 224, 224)
        with self.assertRaisesRegex(UnsupportedOperation, "full224 input grid"):
            predict_segwater(self.scene, grid(), downsampled)


class ScalarImage:
    """Pointwise EE math/mask fixture; real graph tests cover projections/crops.

    One value per band is sufficient to exercise the source/logit mask policy.
    All geometry operations are identities for the first pixel of one chip.
    """

    def __init__(self, values, bands=None, mask=False):
        self.values = np.ma.array(np.atleast_1d(values), mask=mask)
        self.bands = tuple(bands or [f"constant_{index}" for index in range(len(self.values))])

    def _new(self, values):
        return ScalarImage(values, self.bands)

    def _binary(self, other, operation):
        return self._new(operation(self.values, other.values if isinstance(other, ScalarImage) else other))

    def select(self, bands):
        indices = [self.bands.index(band) for band in ([bands] if isinstance(bands, str) else bands)]
        return ScalarImage(self.values[indices], [self.bands[index] for index in indices])

    def rename(self, bands):
        return ScalarImage(self.values, [bands] if isinstance(bands, str) else bands)

    def toFloat(self):
        return self._new(self.values.astype(np.float32))

    def reproject(self, projection):
        return self

    def clip(self, geometry):
        return self

    def setDefaultProjection(self, projection):
        return self

    def mask(self):
        return self._new((~np.ma.getmaskarray(self.values)).astype(np.float32))

    def updateMask(self, mask):
        invalid = np.ma.getmaskarray(self.values) | np.ma.getmaskarray(mask.values) | (mask.values.data <= 0)
        return self._new(np.ma.array(self.values.data, mask=invalid))

    def unmask(self, value, sameFootprint):
        replacement = value.values if isinstance(value, ScalarImage) else value
        return self._new(np.ma.where(np.ma.getmaskarray(self.values), replacement, self.values.data))

    def reduce(self, reducer):
        return ScalarImage([self.values.min()], ["min"])

    def add(self, other): return self._binary(other, np.ma.add)
    def subtract(self, other): return self._binary(other, np.ma.subtract)
    def multiply(self, other): return self._binary(other, np.ma.multiply)
    def divide(self, other): return self._binary(other, np.ma.divide)
    def mod(self, other): return self._binary(other, np.ma.remainder)
    def eq(self, other): return self._binary(other, np.ma.equal)
    def neq(self, other): return self._binary(other, np.ma.not_equal)
    def gt(self, other): return self._binary(other, np.ma.greater)
    def gte(self, other): return self._binary(other, np.ma.greater_equal)
    def lt(self, other): return self._binary(other, np.ma.less)
    def lte(self, other): return self._binary(other, np.ma.less_equal)
    def And(self, other): return self._binary(other, np.ma.logical_and)
    def Or(self, other): return self._binary(other, np.ma.logical_or)
    def floor(self): return self._new(np.ma.floor(self.values))
    def abs(self): return self._new(np.ma.abs(self.values))
    def exp(self): return self._new(np.ma.exp(self.values))
    def clamp(self, lower, upper): return self._new(np.ma.clip(self.values, lower, upper))

    def where(self, condition, value):
        return self._new(np.ma.where(condition.values, value.values, self.values))

    @staticmethod
    def cat(images):
        return ScalarImage(np.ma.concatenate([image.values for image in images]),
                           [band for image in images for band in image.bands])


class TestSegWaterValidityPolicy(ApiTestCase):
    """Numerical source and output mask checks, without any EE evaluation."""

    def run_prediction(self, raw_values=(-20, -30), source_masks=(1, 1),
                       logits=(-1, 1), logit_masks=(1, 1), threshold=0.5):
        source = ScalarImage(raw_values, ["VV", "VH"], mask=np.logical_not(source_masks))

        def compile_stub(model, image, bands, chip_grid, **options):
            result = ScalarImage(logits, ["c0", "c1"], mask=np.logical_not(logit_masks))
            return CompiledModel(result, ("c0", "c1"), chip_grid)

        class ScalarCollection:
            def __init__(self, images): self.images = images
            def mosaic(self): return self.images[0]

        with (patch.object(ee.Image, "constant", side_effect=ScalarImage),
              patch.object(ee.Image, "pixelCoordinates", return_value=ScalarImage([0.5, 0.5], ["x", "y"])),
              patch.object(ee.Image, "cat", side_effect=ScalarImage.cat),
              patch.object(ee.ImageCollection, "fromImages", side_effect=ScalarCollection),
              patch("tensor_ee.segwater_inference.compile_model", side_effect=compile_stub)):
            return predict_segwater(source, grid(), fixed_model(), retain_inputs=True, threshold=threshold)

    def test_either_missing_source_channel_zero_fills_both_normalized_inputs(self):
        for masks in ((0, 1), (1, 0), (0, 0)):
            with self.subTest(source_masks=masks):
                result = self.run_prediction(source_masks=masks)
                np.testing.assert_array_equal(result.normalized_input.values, [0, 0])
                np.testing.assert_array_equal(np.ma.getmaskarray(result.normalized_input.values), [False, False])
                self.assertEqual(result.image.values.data[-1], 0)
                np.testing.assert_array_equal(np.ma.getmaskarray(result.image.values), [True] * 4 + [False])

    def test_valid_numeric_zero_and_minus55_remain_observations(self):
        result = self.run_prediction(raw_values=(0, -55))
        expected = preprocess_db_numpy(np.array([0, -55], dtype=np.float32))
        np.testing.assert_array_equal(result.normalized_input.values, expected)
        self.assertEqual(result.image.values[-1], 1)
        self.assertEqual(result.normalized_input.values.dtype, np.float32)
        self.assertEqual(result.image.values.dtype, np.float32)

    def test_either_masked_or_nonfinite_logit_invalidates_all_predictions_but_coverage_is_zero(self):
        cases = [((-1, 1), (0, 1)), ((-1, 1), (1, 0)),
                 ((np.nan, 1), (1, 1)), ((0, np.inf), (1, 1)), ((-np.inf, 0), (1, 1))]
        for logits, masks in cases:
            with self.subTest(logits=logits, masks=masks):
                result = self.run_prediction(logits=logits, logit_masks=masks)
                self.assertEqual(result.image.values.data[-1], 0)
                np.testing.assert_array_equal(np.ma.getmaskarray(result.image.values), [True] * 4 + [False])

    def test_probability_handles_extreme_logits_and_includes_threshold_ties(self):
        for logits, expected_probability, expected_water in (
            ((-1000, 1000), 1, 1), ((1000, -1000), 0, 0), ((7, 7), 0.5, 1),
        ):
            with self.subTest(logits=logits):
                result = self.run_prediction(logits=logits)
                self.assertEqual(result.image.values[2], expected_probability)
                self.assertEqual(result.image.values[3], expected_water)
                self.assertEqual(result.image.values[4], 1)
                self.assertFalse(np.ma.getmaskarray(result.image.values).any())
        result = self.run_prediction(logits=(7, 7), threshold=0.75)
        self.assertEqual(result.image.values[3], 0)
        self.assertEqual(result.metadata()["water_threshold"], 0.75)
