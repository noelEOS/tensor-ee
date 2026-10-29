"""Offline integer raster bilinear graphs, numerical interpretation and compiler contracts."""
from dataclasses import dataclass
import json
from types import SimpleNamespace
from unittest.mock import patch

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.fx import Graph, GraphModule

from tensor_ee import Grid, Raster, UnsupportedOperation, check_compatibility, compile_model
from tensor_ee.blocks.raster import resample as R
from tensor_ee.blocks.window.resample import model_bilinear_weights
from tensor_ee.core.metadata import metadata_at
from tensor_ee.core.precision import DOUBLE_ROUND_ONCE
from tensor_ee.layouts.raster import _float32
from tensor_ee.receptive_field import UnsupportedReceptiveField, analyze_receptive_field
from tensor_ee.tiling import compile_tiled_model, plan_tiles
from tensor_ee.weight_assets import effective_convolutions
from reference.raster_resample import bilinear_upsample as mirror


@pytest.fixture
def offline_ee():
    case = ApiTestCase()
    case.setUp()
    try:
        yield
    finally:
        case.tearDown()


def grid():
    return Grid("EPSG:3857", (20, 0, 1000, 0, -20, 2000), 7, 5)


def matrices(factor, align, height=5, width=7):
    return (model_bilinear_weights(height, height * factor, dtype=torch.float32, align_corners=align),
            model_bilinear_weights(width, width * factor, dtype=torch.float32, align_corners=align))


def compile_two(model, **options):
    return compile_model(model.eval(), ee.Image.constant([1, -2]).rename(["z", "a"]), ("z", "a"), grid(), **options)


def functional(**options):
    graph = Graph()
    x = graph.placeholder("x")
    graph.output(graph.call_function(F.interpolate, (x,), options))
    return GraphModule(nn.Module(), graph).eval()


@pytest.mark.parametrize("factor", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("align", [False, True])
def test_function_module_size_and_manual_graphs_match(offline_ee, factor, align):
    model = nn.Upsample(scale_factor=factor, mode="bilinear", align_corners=align)
    compiled = compile_two(model)
    variants = [functional(scale_factor=float(factor), mode="bilinear", align_corners=align),
                functional(size=(5 * factor, 7 * factor), mode="bilinear", align_corners=align),
                nn.Upsample(size=(5 * factor, 7 * factor), mode="bilinear", align_corners=align)]
    if align:
        variants.append(nn.UpsamplingBilinear2d(scale_factor=factor))
    for variant in variants:
        assert compile_two(variant).image.serialize() == compiled.image.serialize()
    source = next(iter(compiled.nodes.values()))
    c = R.BilinearConstants(*matrices(factor, align), factor)
    manual = _float32(R.bilinear_upsample(source, c, align_corners=align))
    assert manual.image.serialize() == compiled.image.serialize()
    assert manual.bands == source.bands
    assert manual.grid == grid().resized(1 / factor, 7 * factor, 5 * factor)
    assert c.embedded_values == 3 * factor * (5 + 7)


@pytest.mark.parametrize("recompute", [None, False, True])
def test_integer_scale_recomputation_preserves_graphs(offline_ee, recompute):
    a = compile_two(functional(scale_factor=2, mode="bilinear", align_corners=False,
                               recompute_scale_factor=recompute))
    b = compile_two(functional(scale_factor=2, mode="bilinear", align_corners=False))
    assert a.image.serialize() == b.image.serialize()


def test_shape_specialization_reuses_the_integer_mapping(offline_ee):
    class Resize(nn.Module):
        def forward(self, x):
            h, w = x.shape[-2:]
            return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

    assert compile_two(Resize()).image.serialize() == compile_two(
        nn.Upsample(scale_factor=1, mode="bilinear", align_corners=False)).image.serialize()


@pytest.mark.parametrize("options,reason", [
    ({"scale_factor": 0.5}, "downsampling"),
    ({"scale_factor": 1.5}, "integer scale"),
    ({"scale_factor": (2, 4)}, "equal scale"),
    ({"scale_factor": True}, "static finite"),
    ({"scale_factor": float("nan")}, "static finite"),
    ({"scale_factor": 2, "align_corners": None}, "static bool"),
    ({"scale_factor": 2, "align_corners": 0}, "static bool"),
    ({"scale_factor": 2, "antialias": True}, "antialias"),
    ({"scale_factor": 2, "mode": "bicubic"}, "Unsupported interpolation mode"),
    ({"scale_factor": 2, "mode": "area"}, "Unsupported interpolation mode"),
    ({"scale_factor": 2, "mode": "trilinear"}, "Unsupported interpolation mode"),
    ({"size": (4, 6)}, "downsampling"),
    ({"size": (10, 15)}, "equal scale"),
    ({"size": (6, 8)}, "equal scale"),
    ({"size": 14}, "static pair"),
    ({"size": (True, 14)}, "static pair"),
    ({"size": (10, 14), "scale_factor": 2}, "exactly one"),
    ({"size": (10, 14), "recompute_scale_factor": True}, "does not accept"),
    ({"scale_factor": 2, "recompute_scale_factor": 1}, "static bool"),
])
def test_refusals_agree_between_compiler_and_compatibility(offline_ee, options, reason):
    options = dict(mode="bilinear", align_corners=False) | options
    model = functional(**options)
    report = check_compatibility(model, (1, 2, 5, 7)).to_dict()
    assert report["status"] == "unsupported"
    with pytest.raises(UnsupportedOperation, match=reason) as error:
        compile_two(model)
    assert any(entry["message"] in str(error.value) for entry in report["blockers"])


def test_constants_refuse_nonlocal_or_inexact_weights(offline_ee):
    my, mx = matrices(2, False)
    bad = my.copy()
    bad[0, 0], bad[0, 3] = 0, 1  # Same row sum, outside the parent ring.
    with pytest.raises(UnsupportedOperation, match="one-pixel ring"):
        R.BilinearConstants(bad, mx, 2)
    for value in (1 / 3, float("inf"), float("nan"), 1e40):
        bad = my.copy()
        bad[0, 0] = value
        with pytest.raises(UnsupportedOperation, match="exact float32"):
            R.BilinearConstants(bad, mx, 2)
    with pytest.raises(UnsupportedOperation, match="shape"):
        R.BilinearConstants(my[:-1], mx, 2)
    for factor in (True, 0, 2.5):
        with pytest.raises(UnsupportedOperation, match="positive integer"):
            R.BilinearConstants(my, mx, factor)


def test_block_refuses_policy_size_and_dynamic_alignment(offline_ee):
    raster = Raster(ee.Image.constant([1, 2]).rename(["z", "a"]), ("z", "a"), grid())
    c = R.BilinearConstants(*matrices(2, False), 2)
    with pytest.raises(UnsupportedOperation, match="not implemented"):
        R.bilinear_upsample(raster, c, policy=DOUBLE_ROUND_ONCE)
    with pytest.raises(UnsupportedOperation, match="dimensions"):
        R.bilinear_upsample(Raster(raster.image, raster.bands, grid().resized(2, 4, 3)), c)
    with pytest.raises(UnsupportedOperation, match="static bool"):
        R.bilinear_upsample(raster, c, align_corners=None)
    assert 'Image.clip' in R.bilinear_upsample(raster, c, clip_footprints=True).image.serialize()


@pytest.mark.parametrize("align", [False, True])
def test_metadata_and_receptive_bounds(align):
    entry = R.METADATA["bilinear_upsample"]
    assert metadata_at(entry, {"align_corners": align}).chip_edge_dependent == align
    support = analyze_receptive_field(nn.Upsample(scale_factor=4, mode="bilinear", align_corners=align))
    for x, y in ((0, 0), (3, 9), (21, 15)):
        assert support.input_bounds(x, y) == (x // 4 - 1, y // 4 - 1, x // 4 + 1, y // 4 + 1)
    assert support.chip_edge_dependent == align


def test_extent_dependent_resize_refuses_a_tiled_halo_guarantee(offline_ee):
    model = nn.Sequential(nn.AvgPool2d(2), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)).eval()
    support = analyze_receptive_field(model)
    assert support.as_dict()["safe_halo"] is None
    with pytest.raises(UnsupportedReceptiveField, match="full source extent"):
        _ = support.safe_halo
    large = Grid("EPSG:3857", (20, 0, 1000, 0, -20, 2000), 32, 24)
    plan = plan_tiles(large, core_size=8, halo=4, alignment=2)
    with pytest.raises(UnsupportedReceptiveField, match="full source extent"):
        compile_tiled_model(model, ee.Image.constant([1, 2]), ("constant_0", "constant_1"), plan)


@pytest.mark.parametrize("factor,align", [(3, False), (4, True)])
def test_compatibility_counts_inline_taps_and_assets_collect_only_convolutions(offline_ee, factor, align):
    model = nn.Sequential(nn.Conv2d(2, 3, 1), nn.Upsample(
        scale_factor=factor, mode="bilinear", align_corners=align)).eval()
    with patch.object(ee, "Image", side_effect=AssertionError("local report built EE graph")):
        report = check_compatibility(model, (1, 2, 5, 7)).to_dict()
    assert report["status"] == "supported" and report["output_shape"] == [1, 3, 5 * factor, 7 * factor]
    assert report["weight_storage"]["coefficient_count"] == 9 + 3 * factor * (5 + 7)
    assert "bilinear" in report["weight_storage"]["basis"]
    weights = effective_convolutions(model)
    assert [w.key for w in weights] == ["_0"]
    provider = lambda key, conv: (ee.Array(conv.weight.detach().reshape(3, 2).tolist()),
                                  ee.Array(conv.bias.detach().reshape(3, 1).tolist()))
    with patch("tensor_ee.blocks.window.resample.model_bilinear_weights", wraps=model_bilinear_weights) as reader:
        compile_two(model, weight_provider=provider)
    assert reader.call_count == 2
    assert all(call.kwargs["dtype"] is torch.float32 for call in reader.call_args_list)


def test_nearest_graph_is_unchanged(offline_ee):
    model = nn.Upsample(scale_factor=2, mode="nearest")
    compiled = compile_two(model)
    source = next(iter(compiled.nodes.values()))
    fine = source.grid.resized(0.5, 14, 10)
    expected = _float32(R._finish(R._project(source.image, fine, False), source.bands, fine, False))
    assert compiled.image.serialize() == expected.image.serialize()


def _functions(value):
    if isinstance(value, dict):
        return ([value["functionName"]] if "functionName" in value else []) + [
            name for child in value.values() for name in _functions(child)]
    if isinstance(value, list):
        return [name for child in value for name in _functions(child)]
    return []


def test_graph_has_a_fixed_ring_and_linear_constant_storage(offline_ee):
    for h, w in ((5, 7), (25, 35)):
        g = Grid("EPSG:3857", (20, 0, 1000, 0, -20, 2000), w, h)
        x = Raster(ee.Image.constant([1, 2]).rename(["z", "a"]), ("z", "a"), g)
        c = R.BilinearConstants(*matrices(4, True, h, w), 4)
        out = R.bilinear_upsample(x, c, align_corners=True)
        names = _functions(json.loads(out.image.serialize()))
        assert names.count("Image.arrayGet") == 6
        assert names.count("Image.changeProj") == 9
        assert names.count("Image.multiply") == 12
        assert c.embedded_values == 12 * (h + w)
        assert "Image.toFloat" not in names


@dataclass
class Projection:
    crs: str
    transform: tuple
    height: int = 0
    width: int = 0


@dataclass
class NumericGrid:
    crs: str
    transform: tuple
    width: int
    height: int

    def projection(self): return Projection(self.crs, self.transform, self.height, self.width)
    def resized(self, scale, width, height):
        a, b, c, d, e, f = self.transform
        return NumericGrid(self.crs, (a * scale, b, c, d, e * scale, f), width, height)


class Array:
    def __init__(self, values): self.values = np.asarray(values, np.float64)
    def float(self): return Array(self.values.astype(np.float32).astype(np.float64))
    def double(self): return self


class Image:
    """NumPy implementation of the EE operations used by this block, including relabeling.

    This interprets the block's taps and offsets, independently of the full-matrix
    mirror. It has no interpolation coefficient formula.
    """
    def __init__(self, values, bands=None, projection=None, offset=(0, 0)):
        self.values = values.values if isinstance(values, Array) else np.asarray(values, np.float64)
        self.bands, self.projection, self.offset = bands, projection, offset

    @staticmethod
    def pixelCoordinates(projection):
        y, x = np.indices((projection.height, projection.width))
        return Image(np.stack((x, y), axis=-1), ("x", "y"), projection)

    def floor(self): return Image(np.floor(self.values), self.bands, self.projection)
    def toInt(self): return Image(self.values.astype(int), self.bands, self.projection)
    def clamp(self, low, high): return Image(np.clip(self.values, low, high), self.bands, self.projection)
    def select(self, bands):
        bands = (bands,) if isinstance(bands, str) else tuple(bands)
        return Image(self.values[..., [self.bands.index(b) for b in bands]], bands, self.projection, self.offset)
    def arrayGet(self, position):
        return Image(self.values[position.values.astype(int)[..., 0]][..., None], projection=position.projection)
    def changeProj(self, source, target):
        dx = round((source.transform[2] - target.transform[2]) / source.transform[0])
        dy = round((source.transform[5] - target.transform[5]) / source.transform[4])
        return Image(self.values, self.bands, source, (dy, dx))
    def reproject(self, target):
        factor = round(self.projection.transform[0] / target.transform[0])
        y, x = np.indices((target.height, target.width))
        y, x = y // factor + self.offset[0], x // factor + self.offset[1]
        valid = (y >= 0) & (y < self.values.shape[0]) & (x >= 0) & (x < self.values.shape[1])
        out = np.zeros((target.height, target.width, self.values.shape[2]), np.float64)
        out[valid] = self.values[y[valid], x[valid]]
        return Image(out, self.bands, target)
    def multiply(self, other): return Image(self.values * other.values, self.bands, self.projection)
    def add(self, other): return Image(self.values + other.values, self.bands, self.projection)


@pytest.mark.parametrize("factor,align", [(2, False), (2, True), (3, False), (4, False), (4, True)])
def test_block_values_band_order_and_edge_clamp_match_full_matrix_mirror_and_pytorch(monkeypatch, factor, align):
    monkeypatch.setattr(R, "ee", SimpleNamespace(Array=Array, Image=Image, Projection=Projection))
    monkeypatch.setattr(R, "_zero_extended", lambda raster: raster.image)
    monkeypatch.setattr(R, "_finish", lambda image, bands, grid, clip: Raster(image, bands, grid))
    g = NumericGrid("EPSG:3857", (20, 0, 1000, 0, -20, 2000), 7, 5)
    x = np.random.default_rng(15).uniform(-1, 1, (5, 7, 2)).astype(np.float32)
    raster = Raster(Image(x[..., ::-1], ("a", "z"), g.projection()), ("z", "a"), g)
    my, mx = matrices(factor, align)
    out = R.bilinear_upsample(raster, R.BilinearConstants(my, mx, factor), align_corners=align)
    actual = out.image.values.astype(np.float32)
    np.testing.assert_array_equal(actual, mirror(x, my, mx))
    expected = F.interpolate(torch.from_numpy(x.transpose(2, 0, 1)[None]), scale_factor=factor,
                             mode="bilinear", align_corners=align).numpy()[0].transpose(1, 2, 0)
    assert np.max(np.abs(actual - expected)) <= 1e-4
    np.testing.assert_array_equal(actual[[0, -1]][:, [0, -1]], x[[0, -1]][:, [0, -1]])
