"""The staged run's wiring: each stage reads its predecessor's saved assets; no cloud calls."""
import json
import math
import re
import sys

import ee
import numpy as np
import pytest

from tensor_ee.models.prithvi_sen1floods11.configuration import parameter_shapes, stage_names
from tensor_ee.models.prithvi_sen1floods11.model import constants
from tensor_ee.models.prithvi_sen1floods11.planning import stage_rasters
from tools import prithvi_trained_run as RUN
from tools.live_fixtures import offline_earth_engine
from test_prithvi_sen1floods11 import REDUCED, _server_state

NAMESPACE = "projects/example/assets/prithvi"


def _graph(raster):
    from tensor_ee.converter.request import RECURSION_LIMIT

    sys.setrecursionlimit(max(sys.getrecursionlimit(), RECURSION_LIMIT))
    return json.dumps(ee.serializer.encode(raster.image, for_cloud_api=True))


def test_each_stage_reads_only_its_predecessors_saved_boundary():
    with offline_earth_engine() as attempts:
        stages = list(stage_rasters(ee, constants(ee, _server_state(REDUCED), config=REDUCED),
                                    config=REDUCED, namespace=NAMESPACE))
        assert [stage for stage, _ in stages] == list(stage_names(REDUCED)[1:])
        previous = ["input"]
        for stage, images in stages:
            paired = stage == "patch" or stage.startswith("block") or stage == "norm"
            assert [name for name, _ in images] == ([stage + "_spatial", stage + "_class"] if paired else [stage])
            for index, (name, raster) in enumerate(images):
                read = set(re.findall(rf'"{re.escape(NAMESPACE)}_([a-z0-9_]+)"', _graph(raster)))
                if name == "patch_class":       # cls_token + pos_embed[0]: reads no input
                    expected = set()
                elif stage == "norm":           # row-wise: each part reads its own part
                    expected = {previous[index]}
                elif stage == "upsample28":     # the neck drops the class row
                    expected = {previous[0]}
                else:                           # attention mixes both parts in each block
                    expected = set(previous)
                assert read == expected, (name, read, expected)
            previous = [name for name, _ in images]
        assert stages[-1][1][0][1].bands == ("c0", "c1")
    assert not attempts


def test_the_input_crop_is_deterministic_dyadic_float32():
    crop = RUN.input_crop()
    assert crop.shape == (224, 224, 6) and crop.dtype == np.float32
    assert np.array_equal(crop.view(np.uint32), RUN.input_crop().view(np.uint32))
    assert np.all(np.abs(crop) <= 2) and np.all(crop * 2**11 == np.round(crop * 2**11))


def _manifest(shapes):
    return {"format": "native_ee_featurecollection_weights_v2", "status": "ready", "shards": [],
            "layers": {name: {"shape": shape, "dtype": "float32", "bias": False, "source": name,
                              "sha256": "0", "assets": [f"projects/example/assets/w_part_{index:04d}"]}
                       for index, (name, shape) in enumerate(shapes.items())}}


def test_manifest_shapes_are_checked_before_any_graph():
    shapes = {name: [1, shape[0]] if len(shape) == 1 else [shape[0], math.prod(shape[1:])]
              for name, shape in parameter_shapes().items()}
    with offline_earth_engine() as attempts:
        parameters = RUN._parameters(ee, _manifest(shapes))
        assert parameters["backbone.norm.weight"].shape == (768,)
        assert parameters["backbone.patch_embed.proj.weight"].shape == (768, 1536)
        shapes["backbone.patch_embed.proj.weight"] = [768, 1535]
        with pytest.raises(ValueError, match="patch_embed.proj.weight"):
            RUN._parameters(ee, _manifest(shapes))
    assert not attempts
