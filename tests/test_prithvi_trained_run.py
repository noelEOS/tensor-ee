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
from tensor_ee.models.prithvi_sen1floods11.planning import stage_cuts, stage_rasters
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


MERGED = ("patch", "block3", "block7", "block11", "upsample56", "logits")


def test_the_default_cuts_are_every_boundary_with_identical_graphs():
    with offline_earth_engine() as attempts:
        coefficients = constants(ee, _server_state(REDUCED), config=REDUCED)
        default = list(stage_rasters(ee, coefficients, config=REDUCED, namespace=NAMESPACE))
        explicit = list(stage_rasters(ee, coefficients, config=REDUCED, namespace=NAMESPACE,
                                      cuts=stage_names(REDUCED)[1:], start="input"))
        assert [stage for stage, _ in explicit] == [stage for stage, _ in default]
        for (_, ours), (_, theirs) in zip(explicit, default):
            assert [(name, _graph(raster)) for name, raster in ours] == \
                [(name, _graph(raster)) for name, raster in theirs]
    assert not attempts
    assert stage_cuts(config=REDUCED) == stage_names(REDUCED)[1:]
    assert stage_cuts(config=REDUCED, start="block11")[0] == "norm"


def test_a_merged_stage_reads_only_the_previous_cut_and_the_first_its_source():
    source = "projects/example/assets/earlier"
    with offline_earth_engine() as attempts:
        stages = list(stage_rasters(ee, constants(ee, _server_state(REDUCED), config=REDUCED),
                                    config=REDUCED, namespace=NAMESPACE, cuts=MERGED,
                                    source_namespace=source))
        assert [stage for stage, _ in stages] == list(MERGED)
        previous, prefix = ["input"], source
        for stage, images in stages:
            for index, (name, raster) in enumerate(images):
                graph = _graph(raster)
                read = {(namespace, part) for namespace in (source, NAMESPACE)
                        for part in re.findall(rf'"{re.escape(namespace)}_([a-z0-9_]+)"', graph)}
                if name == "patch_class":
                    expected = set()
                elif stage == "upsample56":     # norm is row-wise; the neck drops the class row
                    expected = {(prefix, previous[0])}
                else:
                    expected = {(prefix, part) for part in previous}
                assert read == expected, (name, read, expected)
            previous, prefix = [name for name, _ in images], NAMESPACE
        assert [name for name, _ in stages[1][1]] == ["block3_spatial", "block3_class"]
        assert stages[-1][1][0][1].bands == ("c0", "c1")
    assert not attempts


@pytest.mark.parametrize("cuts, start", [
    (("patch", "bogus"), "input"), (("block3", "block3"), "input"), (("block7", "block3"), "input"),
    (("block3",), "block3"), (("block3",), "block5"), ((), "input"), (None, "logits"), (None, "nope"),
])
def test_cut_lists_out_of_model_order_are_refused(cuts, start):
    with pytest.raises(ValueError):
        stage_cuts(cuts, config=REDUCED, start=start)


def test_layouts_decide_the_input_export_and_resumes_must_match(tmp_path):
    assert RUN.layout()["cuts"] == list(stage_names()[1:])
    assert RUN.exports_input(RUN.layout(MERGED))
    assert not RUN.exports_input(RUN.layout(MERGED, source_namespace="projects/x/assets/run1"))
    assert not RUN.exports_input(RUN.layout(("block3",), "patch"))
    merged = RUN.layout(MERGED)
    path = tmp_path / "state.json"
    state = RUN._open_state(path, NAMESPACE, merged)
    assert state["cuts"] == list(MERGED) and state["start"] == "input"
    RUN._save(path, state)
    assert RUN._open_state(path, NAMESPACE, merged) == state
    for other in (RUN.layout(), RUN.layout(MERGED, source_namespace="projects/x/assets/run1")):
        with pytest.raises(ValueError, match="another layout"):
            RUN._open_state(path, NAMESPACE, other)
    # a state recorded before cuts existed resumes only as every boundary from the input
    RUN._save(path, {"namespace": NAMESPACE, "input_seed": RUN.INPUT_SEED, "stages": {}})
    assert RUN._recorded_layout(RUN._open_state(path, NAMESPACE, RUN.layout())) == RUN.layout()
    with pytest.raises(ValueError, match="another layout"):
        RUN._open_state(path, NAMESPACE, merged)


def test_run_refuses_an_until_outside_its_cuts_before_any_call(tmp_path):
    with pytest.raises(ValueError, match="--until"):
        RUN.run({}, tmp_path / "run", "project", NAMESPACE, "block5", RUN.layout(MERGED))
    with pytest.raises(ValueError, match="input"):
        RUN.run({}, tmp_path / "run", "project", NAMESPACE, "input",
                RUN.layout(("block3",), "patch"))
    assert not (tmp_path / "run").exists()


def test_identity_compares_same_named_images_on_their_own_grids():
    from tensor_ee.models.prithvi_sen1floods11.model import INPUT_GRID, TOKEN_GRID

    assert RUN.boundary("block3_spatial") == (tuple(f"c{i}" for i in range(768)), TOKEN_GRID)
    assert RUN.boundary("block3_class")[1] == TOKEN_GRID.coarse(14)
    assert RUN.boundary("upsample56")[1] == INPUT_GRID.coarse(4)
    assert RUN.boundary("logits") == (("c0", "c1"), INPUT_GRID)
    with pytest.raises(ValueError):
        RUN.boundary("norm")
    with offline_earth_engine() as attempts:
        bands, grid = RUN.boundary("upsample56")
        expression = RUN.difference(ee, NAMESPACE + "_upsample56", "projects/x/assets/run1_upsample56",
                                    bands, grid)
        graph = json.dumps(ee.serializer.encode(expression, for_cloud_api=True))
        assert NAMESPACE + "_upsample56" in graph and "projects/x/assets/run1_upsample56" in graph
    assert not attempts
