"""Public composition must preserve the compiler graph and fixed-chip bounds."""

from unittest.mock import patch

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest
import torch
from torch import nn

from tensor_ee import (Grid, compile_model, from_image, conv2d, relu, sigmoid,
                       pool2d, nearest_upsample, concatenate, residual_add)
from examples.building_blocks import build_manually, model_and_grid
from examples.water_chip import GRID, build_prediction, export_task
from examples.verify_water_chip import BANDS, decode_pixels


class TestPublicComposition(ApiTestCase):
    def test_manual_example_matches_compiler_expression_exactly(self):
        model, grid = model_and_grid()
        image = ee.Image.constant([-0.5, 0.25]).rename(["VV", "VH"])
        manual = build_manually(model, image, grid)
        compiled = compile_model(model, image, ["VV", "VH"], grid)
        self.assertEqual(manual.image.serialize(), compiled.image.serialize())

    def test_skip_pool_and_residual_composition_matches_compiler(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(2, 2, 3, padding=1)
                self.pool = nn.MaxPool2d(2)
                self.up = nn.Upsample(scale_factor=2, mode="nearest")

            def forward(self, x):
                y = torch.relu(self.conv(x))
                return torch.sigmoid(torch.cat([y + x, self.up(self.pool(y))], dim=1))

        model = Model().eval()
        grid = Grid("EPSG:32651", (10, 0, 0, 0, -10, 1000), 16, 16)
        image = ee.Image.constant([-1, 2]).rename(["VV", "VH"])
        for clip in (False, True):
            options = {"clip_footprints": clip}
            x = from_image(image, ["VV", "VH"], grid, **options)
            y = relu(conv2d(x, model.conv, **options), **options)
            pooled = nearest_upsample(pool2d(y, model.pool, **options), 2, **options)
            manual = sigmoid(concatenate([residual_add(y, x, **options), pooled], **options), **options)
            compiled = compile_model(model, image, ["VV", "VH"], grid, **options)
            self.assertEqual(manual.image.serialize(), compiled.image.serialize())

    def test_single_chip_example_exports_exact_extent_and_reference_inputs(self):
        model = nn.Sequential(nn.Conv2d(2, 2, 1)).eval()
        model.tensor_ee_input_shape = (2, 224, 224)
        prediction = build_prediction(model, None)
        self.assertEqual(len(prediction.tiles), 1)
        self.assertEqual(prediction.grid, GRID)
        with patch.object(ee.batch.Export.image, "toAsset") as export:
            export_task(prediction, "projects/example/assets/chip", "hash")
        options = export.call_args.kwargs
        self.assertEqual(options["crsTransform"], list(GRID.transform))
        self.assertEqual(options["crs"], GRID.crs)
        self.assertEqual(options["maxPixels"], 50176)
        self.assertFalse(options["overwrite"])
        self.assertNotIn("scale", options)
        self.assertIn("z_VV", options["image"].serialize())
        self.assertIn("pixel_x", options["image"].serialize())


def test_single_chip_example_rejects_expansion_before_graph_construction():
    with pytest.raises(ValueError, match="exactly one"):
        build_prediction(None, None, grid=Grid(GRID.crs, GRID.transform, 448, 224))


def pixels():
    values = np.zeros((224, 224), dtype=[(band, np.float32) for band in BANDS])
    values["pixel_y"], values["pixel_x"] = np.indices((224, 224), dtype=np.float32)
    values["coverage"] = 1
    return values


def test_validation_rejects_wrong_orientation_and_missing_pixels():
    values = pixels()
    assert decode_pixels(values)["ee_logits"].shape == (2, 224, 224)
    with pytest.raises(ValueError, match="coordinates"):
        decode_pixels(values[::-1])
    values["coverage"][0, 0] = 0
    with pytest.raises(ValueError, match="valid coverage"):
        decode_pixels(values)
