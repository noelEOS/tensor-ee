"""Static FX plumbing: exact graph identities and shared refusal reasons."""
import builtins
from copy import deepcopy
import operator

import ee
from ee.apitestcase import ApiTestCase
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.fx import Graph, GraphModule, symbolic_trace

from tensor_ee import Grid, UnsupportedOperation, check_compatibility, compile_model
from tensor_ee.blocks.raster.resample import nearest_factor, METADATA
from tensor_ee.semantics import specialize_plumbing


def indexed_model(container, index):
    graph = Graph()
    x = graph.placeholder("x")
    source = container(graph, x)
    selected = graph.call_function(operator.getitem, (source, index))
    graph.output(selected)
    return GraphModule(nn.Module(), graph).eval()


class ShapeResize(nn.Module):
    def __init__(self, size_form=True):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1)
        self.pool = nn.AvgPool2d(2)
        self.size_form = size_form

    def forward(self, x):
        y = self.pool(x)
        if self.size_form:
            h, w = x.shape[2:]
            y = F.interpolate(y, size=(h, w), mode="nearest")
        else:
            y = F.interpolate(y, scale_factor=2, mode="nearest")
        return self.conv(y)


class TestStaticPlumbing(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.grid = Grid("EPSG:3857", (10, 0, 0, 0, -10, 0), 8, 8)
        self.image = ee.Image.constant([1, -2])

    def compile(self, model, **kwargs):
        return compile_model(model, self.image, ("constant_0", "constant_1"), self.grid, **kwargs)

    def test_shape_resize_graph_identity_and_nonmutation(self):
        size = ShapeResize().eval()
        scale = deepcopy(size)
        scale.size_form = False
        original = symbolic_trace(size)
        code = original.code
        assert check_compatibility(original, (1, 2, 8, 8)).status == "supported"
        assert original.code == code
        for clip in (False, True):
            a = self.compile(size, clip_footprints=clip)
            b = self.compile(scale, clip_footprints=clip)
            assert a.image.serialize() == b.image.serialize()
            assert tuple(a.nodes) == tuple(b.nodes)
        x = torch.arange(128, dtype=torch.float32).reshape(1, 2, 8, 8)
        lowered = specialize_plumbing(original, (1, 2, 8, 8)).graph
        assert torch.equal(size(x), lowered(x))

    def test_literal_feature_selection_emits_no_tensor_cast(self):
        class Passthrough(nn.Module):
            def forward(self, x):
                return x

        for container in (lambda g, x: [x, x], lambda g, x: (x, x)):
            for index in (0, -1):
                selected = indexed_model(container, index)
                assert check_compatibility(selected, (1, 2, 8, 8)).status == "supported"
                assert self.compile(selected).image.serialize() == self.compile(Passthrough().eval()).image.serialize()
        graph = Graph()
        x = graph.placeholder("x")
        subsequence = graph.call_function(operator.getitem, ([x, x, x], slice(None, None, -1)))
        selected = graph.call_function(operator.getitem, (subsequence, -1))
        graph.output(selected)
        assert self.compile(GraphModule(nn.Module(), graph).eval()).image.serialize() == self.compile(Passthrough().eval()).image.serialize()

    def test_nn_upsample_size_uses_the_same_lowering(self):
        for factor in (1, 2, 3, 4, 5):
            size = nn.Upsample(size=(8 * factor, 8 * factor), mode="nearest").eval()
            scale = nn.Upsample(scale_factor=factor, mode="nearest").eval()
            assert check_compatibility(size, (1, 2, 8, 8)).status == "supported"
            assert self.compile(size).image.serialize() == self.compile(scale).image.serialize()

    def test_refusals_have_matching_report_and_compiler_reasons(self):
        for options in (
                dict(size=(12, 12)), dict(size=(16, 24)), dict(size=(4, 4)),
                dict(size=(0, 16)), dict(size=(True, 16)), dict(size=16),
                dict(size=(16, 16), scale_factor=2),
                dict(size=(16, 16), recompute_scale_factor=True),
                dict(size=(16, 16), mode="bilinear"),
                dict(size=(16, 16), align_corners=False),
                dict(size=(16, 16), antialias=True)):
            graph = Graph()
            x = graph.placeholder("x")
            graph.output(graph.call_function(F.interpolate, (x,), options))
            model = GraphModule(nn.Module(), graph).eval()
            report = check_compatibility(model, (1, 2, 8, 8)).to_dict()
            assert report["status"] == "unsupported", options
            reason = report["blockers"][0]
            with self.assertRaises(UnsupportedOperation) as raised:
                self.compile(model)
            assert reason["node"] in str(raised.exception)
            assert reason["message"] in str(raised.exception)

    def test_invalid_and_tensor_value_indexing_remain_refused(self):
        for container, index in ((lambda g, x: [x], 2), (lambda g, x: [x], True),
                                 (lambda g, x: [x], slice(None, None, 0)),
                                 (lambda g, x: x, 0)):
            model = indexed_model(container, index)
            assert check_compatibility(model, (1, 2, 8, 8)).status == "unsupported"
            with self.assertRaises(UnsupportedOperation):
                self.compile(model)


@pytest.mark.parametrize("factor", range(1, 17))
@pytest.mark.parametrize("spatial", [(1, 1), (2, 3), (7, 9), (32, 32)])
def test_size_nearest_indices_are_floor_i_over_k(factor, spatial):
    # The accepted integer domain is unbounded. Exercise every factor 1..16
    # (including all validated settings) at each tested spatial size.
    assert nearest_factor(factor) == factor
    h, w = spatial
    indices = torch.arange(h * w, dtype=torch.float32).reshape(1, 1, h, w)
    actual = F.interpolate(indices, size=(h * factor, w * factor), mode="nearest")
    rows = torch.arange(h * factor) // factor
    cols = torch.arange(w * factor) // factor
    expected = indices[:, :, rows[:, None], cols]
    assert torch.equal(actual, expected)
    assert torch.equal(actual, F.interpolate(indices, scale_factor=factor, mode="nearest"))


def test_nearest_validated_record_is_unchanged():
    assert tuple(dict(x) for x in METADATA["nearest_upsample"].validated.configurations) == ({"scale_factor": (2, 3, 4)},)


@pytest.mark.parametrize("index", [4, -5, True, slice(None, None, 0)])
def test_invalid_shape_indices(index):
    model = indexed_model(lambda g, x: g.call_function(builtins.getattr, (x, "shape")), index)
    result = specialize_plumbing(model, (1, 2, 8, 8))
    assert result.errors


def test_dynamic_shape_index_and_arbitrary_attribute():
    for dynamic in (False, True):
        graph = Graph()
        x = graph.placeholder("x")
        shape = graph.call_function(builtins.getattr, (x, "shape" if dynamic else "dtype"))
        output = graph.call_function(operator.getitem, (shape, x)) if dynamic else shape
        graph.output(output)
        report = check_compatibility(GraphModule(nn.Module(), graph).eval(), (1, 2, 8, 8)).to_dict()
        assert report["status"] == "unsupported"
        assert any("literal" in b["message"] or "getattr" in b["message"] for b in report["blockers"])


def test_unavailable_shape_is_unknown_without_executing_activations():
    class UnknownShape(nn.Module):
        def forward(self, x):
            y = torch.sin(x)
            return F.interpolate(x, size=y.shape[2:], mode="nearest")

    report = check_compatibility(UnknownShape().eval(), (1, 2, 8, 8)).to_dict()
    assert report["status"] == "unsupported"
    assert any("sin" in b["message"] for b in report["blockers"])
    assert any("shape" in u["message"].lower() for u in report["unknowns"])
