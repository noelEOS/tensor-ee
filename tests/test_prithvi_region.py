"""A region of whole Prithvi crops: one graph, each crop as if it ran alone; no cloud calls.

- One crop builds today's graphs and export requests byte for byte (SHA-256 pinned from
  ``origin/main`` at 67fbf20, before regions existed).
- A region's stages carry the region grid; only the logits stage gains the per-crop
  padding masks; weight references do not grow with the crops.
- The reduced model on a 2 x 2 region equals PyTorch on each crop alone (eager mirror).
- The run tool: layouts, the input assembly, per-export shard sizes, the per-crop
  comparison with its seam ring, and ``identity --crop``.
"""
import hashlib
import json
import math
import re
import sys

import ee
import numpy as np
import pytest

from tensor_ee.core.grid import Grid
from tensor_ee.core.tensor import Raster
from tensor_ee.models.prithvi_sen1floods11.configuration import parameter_shapes
from tensor_ee.models.prithvi_sen1floods11.model import (
    INPUT_GRID, constants, crops_of, forward, region_grid)
from tensor_ee.models.prithvi_sen1floods11.planning import stage_rasters
from tools import prithvi_trained_run as RUN
from tools.live_fixtures import offline_earth_engine
from test_prithvi_sen1floods11 import REDUCED, _server_state

NAMESPACE = "projects/example/assets/prithvi"
NINE = ("patch", "block1", "block3", "block5", "block7", "block9", "block11", "upsample56", "logits")

# SHA-256 of the single-crop graphs and export requests at origin/main 67fbf20, built as below.
SINGLE_CROP = {
    "forward_logits": "d6650cd6ca1bec4c8cffecff159af2f9336b8de17a42136270f075bbc7d70290",
    "stages_default": "3773fe2d2617da740e806d02e4b94a080dd521ca94d1adaf2d6b6bcb663d290b",
    "stages_nine": "fb39d7ba257b7bde10829624fd8ba7eb60526c9c1345a2fa28b7bda68a262a85",
    "requests_default": "3772bcfaa7174803b6e1d254f6c031a34bf78fae4a8be2d756e0e5f291bbfc97",
    "requests_nine": "8d8bcae8915dc29d469d2866a9c595132e61964cd911d14d8c8d0ff27afb0031",
}


@pytest.fixture(autouse=True)
def _recursion():
    from tensor_ee.converter.request import RECURSION_LIMIT

    sys.setrecursionlimit(max(sys.getrecursionlimit(), RECURSION_LIMIT))


def _encoded(obj) -> str:
    return json.dumps(ee.serializer.encode(obj, for_cloud_api=True), sort_keys=True)


def _sha(obj) -> str:
    return hashlib.sha256(_encoded(obj).encode()).hexdigest()


def _task_json(task) -> str:
    config = dict(task.config, expression=ee.serializer.encode(task.config["expression"], for_cloud_api=True))
    return json.dumps(config, sort_keys=True)


def _manifest():
    shapes = {n: [1, s[0]] if len(s) == 1 else [s[0], math.prod(s[1:])] for n, s in parameter_shapes().items()}
    return {"format": "native_ee_featurecollection_weights_v2", "status": "ready", "shards": [],
            "layers": {n: {"shape": s, "dtype": "float32", "bias": False, "source": n, "sha256": "0",
                           "assets": [f"projects/example/assets/w_part_{i:04d}"]}
                       for i, (n, s) in enumerate(shapes.items())}}


def _raster(width=224, height=224):
    grid = Grid("EPSG:3857", (10, 0, 100, 0, -10, 200), width, height)
    image = ee.Image.constant([0] * 6).toFloat().rename([f"b{i}" for i in range(6)])
    return Raster(image, tuple(f"b{i}" for i in range(6)), grid)


# ---------------------------------------------------------------------------
# One crop: today's graphs, byte for byte
# ---------------------------------------------------------------------------
def test_one_crop_builds_todays_graphs_and_requests_byte_for_byte():
    found = {}
    with offline_earth_engine() as attempts:
        c = constants(ee, _server_state(REDUCED), config=REDUCED)
        found["forward_logits"] = _sha(forward(ee, _raster(), c)["logits"].image)
        for label, cuts in (("stages_default", None), ("stages_nine", NINE)):
            digest = hashlib.sha256()
            for _, images in stage_rasters(ee, c, config=REDUCED, namespace=NAMESPACE, cuts=cuts, crops=(1, 1)):
                for name, raster in images:
                    digest.update(name.encode() + _sha(raster.image).encode())
            found[label] = digest.hexdigest()
        full = constants(ee, RUN._parameters(ee, _manifest()))
        for label, cuts in (("requests_default", None), ("requests_nine", NINE)):
            run_layout = RUN.layout(cuts)
            assert not RUN.is_region(run_layout)
            digest = hashlib.sha256()
            for _, images in RUN.input_stages(ee, run_layout, NAMESPACE):
                for name, raster, shard in images:
                    assert shard is None
                    digest.update(_task_json(RUN._export(ee, raster, NAMESPACE + "_" + name,
                                                         "prithvi_" + name, shard)).encode())
            for _, images in stage_rasters(ee, full, namespace=NAMESPACE, cuts=cuts):
                for name, raster in images:
                    shard = RUN.shard_size(raster.grid, run_layout)
                    assert shard is None
                    digest.update(_task_json(RUN._export(ee, raster, NAMESPACE + "_" + name,
                                                         "prithvi_" + name, shard)).encode())
            found[label] = digest.hexdigest()
    assert not attempts
    assert found == SINGLE_CROP


def test_region_grids_are_whole_crops():
    assert region_grid() == INPUT_GRID and region_grid(2, 3).height == 448 and region_grid(2, 3).width == 672
    assert crops_of(region_grid(3, 1)) == (3, 1)
    for rows, cols in ((0, 1), (1, -1), (True, 1), (1.5, 1)):
        with pytest.raises(ValueError):
            region_grid(rows, cols)
    with pytest.raises(ValueError, match="complete 224"):
        crops_of(Grid("EPSG:3857", (10, 0, 0, 0, -10, 0), 448, 100))


# ---------------------------------------------------------------------------
# A region's graphs
# ---------------------------------------------------------------------------
def test_a_region_stage_graph_carries_the_region_grid_and_the_crop_padding_only_at_the_head():
    with offline_earth_engine() as attempts:
        c = constants(ee, _server_state(REDUCED), config=REDUCED)
        single = dict((s, images) for s, images in stage_rasters(ee, c, config=REDUCED, namespace=NAMESPACE, cuts=NINE))
        region = dict((s, images) for s, images in stage_rasters(ee, c, config=REDUCED, namespace=NAMESPACE,
                                                                cuts=NINE, crops=(2, 2)))
        grid = region_grid(2, 2)
        for stage, images in region.items():
            for (name, raster), (_, alone) in zip(images, single[stage]):
                assert raster.grid.height == 2 * alone.grid.height and raster.grid.width == 2 * alone.grid.width
                graph, one = _encoded(raster.image), _encoded(alone.image)
                # the same weights and saved boundaries are read, whatever the number of crops
                assert sorted(re.findall(r'"projects/[^"]+"', graph)) == sorted(re.findall(r'"projects/[^"]+"', one))
                masks = graph.count('"Image.mod"') - one.count('"Image.mod"')
                assert (masks > 0) == (stage == "logits"), (stage, masks)
        assert region["logits"][0][1].grid == grid
    assert not attempts


def test_the_eager_mirror_of_a_region_is_each_crop_alone():
    """The reduced model on a 2 x 2 region equals PyTorch on each crop alone, and crop (0, 0)
    equals the single-crop mirror bit for bit."""
    import torch
    from reference.prithvi_sen1floods11.execution import execute, read
    from reference.prithvi_sen1floods11.reference import random_state
    from tools.fixtures.prithvi_model import region_reference

    torch.set_num_threads(2)
    state = random_state(REDUCED.width, REDUCED.depth, REDUCED.decoder)
    image = np.random.default_rng(5).normal(0, .5, (448, 448, 6)).astype(np.float32)
    got = read(execute(image, state, REDUCED)["logits"])
    error = np.abs(got.astype(np.float64) - region_reference(image, state))
    assert error.max() <= 1e-4
    alone = read(execute(image[:224, :224], state, REDUCED)["logits"])
    assert np.array_equal(alone.view(np.uint32), got[:224, :224].view(np.uint32))
    # Without per-crop padding the seams would read the neighbouring crops.
    from reference.prithvi_sen1floods11 import execution as X

    halo = X.halo
    try:
        X.halo = lambda ee, x, coarse, channels, *, window, crop=None: X.Halo(x, coarse, channels, window)
        joined = read(execute(image, state, REDUCED)["logits"])
    finally:
        X.halo = halo
    ring = RUN.seam_ring(448, 448)
    assert np.abs(joined.astype(np.float64) - got)[ring].max() > 1e-3


# ---------------------------------------------------------------------------
# The run tool
# ---------------------------------------------------------------------------
def test_region_layouts_seeds_and_resumes(tmp_path):
    region = RUN.layout(NINE, crops=(2, 2))
    assert region["crops"] == [2, 2] and region["input_seeds"] == [2024, 2025, 2026, 2027]
    assert region["shard_crops"] == 1 and RUN.is_region(region)
    assert RUN.layout(NINE)["crops"] == [1, 1] and RUN.layout(NINE)["input_seeds"] is None
    for options in ({"input_seeds": [1, 2, 3]}, {"input_seeds": [1, 1, 2, 3]}, {"shard_crops": 0},
                    {"input_seeds": [1, 2, 3, 4], "reuse_input": "projects/x/assets/run1_input"}):
        with pytest.raises(ValueError):
            RUN.layout(NINE, crops=(2, 2), **options)
    for options in ({"input_seeds": [2024]}, {"reuse_input": "projects/x/assets/run1_input"}, {"shard_crops": 2}):
        with pytest.raises(ValueError, match="region"):
            RUN.layout(NINE, **options)
    path = tmp_path / "state.json"
    state = RUN._open_state(path, NAMESPACE, region)
    RUN._save(path, state)
    assert RUN._open_state(path, NAMESPACE, region) == state
    with pytest.raises(ValueError, match="another layout"):
        RUN._open_state(path, NAMESPACE, RUN.layout(NINE, crops=(2, 2), shard_crops=2))
    # a state recorded before regions existed resumes only as one crop
    RUN._save(path, {"namespace": NAMESPACE, "input_seed": RUN.INPUT_SEED, "cuts": list(NINE),
                     "start": "input", "source_namespace": None, "stages": {}})
    assert RUN._recorded_layout(RUN._open_state(path, NAMESPACE, RUN.layout(NINE))) == RUN.layout(NINE)


def test_the_region_input_is_assembled_from_its_crops():
    reuse = "projects/x/assets/run1_input"
    run_layout = RUN.layout(NINE, crops=(2, 2), reuse_input=reuse)
    with offline_earth_engine() as attempts:
        stages = RUN.input_stages(ee, run_layout, NAMESPACE)
        assert [stage for stage, _ in stages] == ["input_crops", "input"]
        assert [name for name, _, _ in stages[0][1]] == ["input_r0c1", "input_r1c0", "input_r1c1"]
        assert all(shard is None and raster.grid == INPUT_GRID for _, raster, shard in stages[0][1])
        (name, region, shard), = stages[1][1]
        assert name == "input" and region.grid == region_grid(2, 2) and shard == 448 // 2
        graph = _encoded(region.image)
        assert set(re.findall(r'"(projects/[^"]+)"', graph)) == {
            reuse, *(NAMESPACE + "_input_r" + rc for rc in ("0c1", "1c0", "1c1"))}
        task = RUN._export(ee, region, NAMESPACE + "_input", "prithvi_input", shard)
        assert task.config["assetExportOptions"]["tileSize"] == {"value": 224}     # shardSize, as sent
        single = RUN._export(ee, region, NAMESPACE + "_input", "prithvi_input")
        assert "tileSize" not in single.config["assetExportOptions"]
        sizes = [RUN._request_bytes(ee, RUN._export(ee, r, NAMESPACE + "_" + n, "prithvi_" + n, s))
                 for _, images in stages for n, r, s in images]
        assert max(sizes) <= RUN.STAGE_LIMIT_BYTES
    assert not attempts
    seeds = RUN.layout(NINE, crops=(2, 2), input_seeds=[7, 8, 9, 10])
    with offline_earth_engine():
        assert len(RUN.input_stages(ee, seeds, NAMESPACE)[0][1]) == 4


@pytest.mark.parametrize("shard_crops", [1, 2])
def test_region_exports_shard_by_crops(shard_crops):
    run_layout = RUN.layout(NINE, crops=(4, 4), shard_crops=shard_crops)
    grid = region_grid(4, 4)
    expected = {"patch_spatial": 14, "patch_class": 1, "upsample56": 56, "logits": 224, "input": 224}
    for name, extent in expected.items():
        _, boundary_grid = RUN.boundary(name, crops=(4, 4))
        assert RUN.shard_size(boundary_grid, run_layout) == extent * shard_crops
    assert RUN.shard_size(grid, RUN.layout(NINE)) is None


def test_the_crop_comparison_reports_each_crop_and_the_seam_ring():
    seeds = [2024, 2025, 2026, 2027]
    image = np.concatenate([np.concatenate([RUN.input_crop(seeds[2 * r + c]) for c in range(2)], axis=1)
                            for r in range(2)], axis=0)
    reference = lambda crop: crop[..., :2].astype(np.float64) * 0.5            # noqa: E731
    logits = np.concatenate([np.concatenate([reference(image[224 * r:224 * r + 224, 224 * c:224 * c + 224])
                                             for c in range(2)], axis=1) for r in range(2)], axis=0).astype(np.float32)
    record = RUN.crop_comparison(image, logits, reference, seeds, (2, 2))
    assert record["input_equals_local"] and record["max_abs"] == 0 and record["seam_max_abs"] == 0
    assert set(record["crops"]) == {"0,0", "0,1", "1,0", "1,1"}
    assert record["seam_pixels"] == int(RUN.seam_ring(448, 448).sum()) == 4 * 448 - 4
    logits[224, 100] += 0.01                                       # a seam pixel of crop (1, 0)
    record = RUN.crop_comparison(image, logits, reference, seeds, (2, 2))
    assert record["crops"]["1,0"]["max_abs"] > 0.009 and record["crops"]["0,0"]["max_abs"] == 0
    assert record["seam_max_abs"] == pytest.approx(record["max_abs"])
    single = RUN.crop_comparison(image[:224, :224], logits[:224, :224], reference, [2024], (1, 1))
    assert "crops" not in single and single["max_abs"] == 0
    ring = RUN.seam_ring(448, 448)
    assert ring[223, 5] and ring[224, 5] and ring[5, 223] and ring[0, 223] and not ring[0, 5] and not ring[222, 5]


def test_identity_of_a_crop_relabels_the_single_crop_reference():
    state = {"namespace": NAMESPACE, **RUN.layout(NINE, crops=(2, 2)), "stages": {
        "input": [{"name": "input", "status": "COMPLETED"}],
        "upsample56": [{"name": "upsample56", "asset_id": NAMESPACE + "_upsample56", "status": "COMPLETED"}],
        "block1": [{"name": "block1_class", "asset_id": NAMESPACE + "_block1_class", "status": "COMPLETED"}]}}
    reference = "projects/x/assets/run2"
    with offline_earth_engine() as attempts:
        plain = {name: (graph, expected) for name, _, _, graph, expected in
                 RUN.identity_requests(ee, state, reference, (0, 0))}
        shifted = {name: (graph, expected) for name, _, _, graph, expected in
                   RUN.identity_requests(ee, state, reference, (1, 1))}
        whole = {name: expected for name, *_, expected in RUN.identity_requests(ee, state, reference)}
        assert plain["upsample56"][1] == 56 * 56 and plain["block1_class"][1] == 1
        assert whole["upsample56"] == 112 * 112
        assert "changeProj" not in _encoded(plain["upsample56"][0])
        assert "changeProj" in _encoded(shifted["upsample56"][0])
        assert reference + "_upsample56" in _encoded(shifted["upsample56"][0])
        with pytest.raises(ValueError, match="outside"):
            RUN.identity_requests(ee, state, reference, (2, 0))
    assert not attempts


def test_identity_does_not_request_an_image_already_found_identical(tmp_path):
    reference = "projects/x/assets/run2"
    good = {"max_abs": 0, "pixels": 196, "expected_pixels": 196}
    RUN._save(tmp_path / "identity.json", {"reference_namespace": reference, "crop": [0, 0], "images": {
        "patch_spatial": good, "patch_class": {"max_abs": 0, "pixels": 0, "expected_pixels": 1},
        "block1_spatial": {"max_abs": 1e-7, "pixels": 196, "expected_pixels": 196}}})
    assert set(RUN.checked_images(tmp_path, reference, (0, 0))) == {"patch_spatial"}
    assert RUN.checked_images(tmp_path, reference, (1, 1)) == {}
    assert RUN.checked_images(tmp_path, "projects/x/assets/run1", (0, 0)) == {}
    assert RUN.checked_images(tmp_path / "none", reference, (0, 0)) == {}


def test_the_twin_fixture_reads_both_sides_of_every_seam():
    from tools.fixtures import registry
    from tools.fixtures.prithvi_model import REGION_LATTICES

    fixture = registry()["window/concat_rows/prithvi-region-2x2"]
    assert fixture.shapes["crops"] == (2, 2) and fixture.requests == 3
    lattices = {name: np.arange(first, 448, 8) for name, first in REGION_LATTICES.items()}
    assert 223 in lattices["logits_last"] and 224 in lattices["logits_first"] and 447 in lattices["logits_last"]
    assert 0 in lattices["logits_first"]
