"""``compile_model(..., saved_nodes=...)`` offline: graph identity of the whole model with a saved
node against the model's tail compiled alone from the same image, clip on and off; what a request
reads when a skip crosses the cut; the refusals. Graph construction only: no request is sent."""
from __future__ import annotations

import json

import ee
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tensor_ee import Grid, compile_model
from tensor_ee.converter.request import SIZING_REQUEST_ID, deep_recursion, export_request, request_reads
from tensor_ee.core.errors import UnsupportedOperation
from tools.fixtures.core import dyadic
from tools.live_fixtures import offline_earth_engine

GRID = Grid("EPSG:3857", (10, 0, 1000, 0, -10, 2000), 16, 16)
BANDS = ("b0", "b1")
INPUT = "projects/example/assets/saved/input"


def _integer_weights(module: nn.Module, seed: int) -> nn.Module:
    """Every parameter from ``dyadic`` (integer arithmetic), BatchNorm statistics fixed."""
    with torch.no_grad():
        for name, value in module.named_parameters():
            value.copy_(torch.from_numpy(dyadic(seed, name, tuple(value.shape), scale=0.25)))
        for name, value in module.named_buffers():
            if name.endswith("running_mean"):
                value.copy_(torch.from_numpy(dyadic(seed, name, tuple(value.shape), scale=0.125)))
            elif name.endswith("running_var"):
                value.copy_(torch.from_numpy(dyadic(seed, name, tuple(value.shape), scale=0.25, offset=1.0, bits=16)))
    return module.eval()


class Gate(nn.Module):
    """A squeeze-excitation gate after the cut, then a 1 x 1 convolution."""

    def __init__(self, channels=4, squeezed=2):
        super().__init__()
        self.reduce, self.expand = nn.Conv2d(channels, squeezed, 1), nn.Conv2d(squeezed, channels, 1)
        self.project = nn.Conv2d(channels, 3, 1)

    def forward(self, y):
        gate = torch.sigmoid(self.expand(F.silu(self.reduce(F.adaptive_avg_pool2d(y, 1)))))
        return self.project(gate * y)


def _chain():
    head = nn.Sequential(nn.Conv2d(2, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
                         nn.Conv2d(4, 4, 3, stride=2, padding=1), nn.ReLU())
    tail = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 3, 1))
    return _integer_weights(nn.Sequential(head, tail), 301)


def _gated():
    head = nn.Sequential(nn.Conv2d(2, 4, 3, padding=1), nn.SiLU())
    return _integer_weights(nn.Sequential(head, Gate()), 302)


class Skip(nn.Module):
    """A U-Net step: a skip at full resolution crosses the cut beside the downsampled path."""

    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Conv2d(2, 4, 3, padding=1), nn.Conv2d(4, 4, 3, stride=2, padding=1)
        self.c, self.out = nn.Conv2d(4, 4, 3, padding=1), nn.Conv2d(8, 3, 1)

    def forward(self, x):
        a = F.relu(self.a(x))
        b = F.relu(self.b(a))
        up = F.interpolate(self.c(b), scale_factor=2, mode="nearest")
        return self.out(torch.cat([up, a], dim=1))


def _serialized(image) -> str:
    with deep_recursion():
        return ee.serializer.toJSON(image)


def _node_for(compiled, target: str) -> str:
    """The FX node name that calls submodule ``target``."""
    graph = compiled.graph_module.graph
    return next(node.name for node in graph.nodes if node.op == "call_module" and str(node.target) == target)


def _input():
    return ee.Image(INPUT).select(list(BANDS))


@pytest.mark.parametrize("clip", [False, True])
@pytest.mark.parametrize("make, cut", [(_chain, "0.4"), (_gated, "0.1")])
def test_the_whole_model_with_a_saved_node_is_its_tail_compiled_alone(make, cut, clip):
    model = make()
    with offline_earth_engine():
        whole = compile_model(model, _input(), BANDS, GRID, clip_footprints=clip)
        name = _node_for(whole, cut)
        node = whole.nodes[name]
        saved = ee.Image("projects/example/assets/saved/node")
        staged = compile_model(model, _input(), BANDS, GRID, clip_footprints=clip, saved_nodes={name: saved})
        tail = compile_model(model[1], saved, node.bands, node.grid, clip_footprints=clip)
        assert staged.bands == tail.bands and staged.grid == tail.grid
        assert _serialized(staged.image) == _serialized(tail.image)
        assert _serialized(staged.image) != _serialized(whole.image)
        # None is today's graph
        assert _serialized(compile_model(model, _input(), BANDS, GRID, clip_footprints=clip,
                                         saved_nodes=None).image) == _serialized(whole.image)


def _reads(compiled) -> dict:
    with deep_recursion():
        request = export_request(compiled.image, compiled.grid, asset_id="projects/example/assets/saved/out",
                                 description="out", request_id=SIZING_REQUEST_ID)
    return request_reads(json.loads(json.dumps(request)))


@pytest.mark.parametrize("clip", [False, True])
def test_a_skip_across_the_cut_reads_exactly_the_saved_images(clip):
    """A two-input tail cannot be compiled alone (``compile_model`` takes one input), so the skip
    case is checked on what the request reads: with both crossing tensors saved, exactly their
    two assets and not the input; with only the downsampled one saved, the input too."""
    model = _integer_weights(Skip(), 303)
    with offline_earth_engine():
        whole = compile_model(model, ee.Image(INPUT), BANDS, GRID, clip_footprints=clip)
        a, b = _node_for(whole, "a"), _node_for(whole, "b")
        relu_a = next(user.name for user in whole.graph_module.graph.nodes if user.name != a
                      and any(getattr(arg, "name", None) == a for arg in user.args))
        relu_b = next(user.name for user in whole.graph_module.graph.nodes if user.name != b
                      and any(getattr(arg, "name", None) == b for arg in user.args))
        assert whole.nodes[relu_a].grid != whole.nodes[relu_b].grid
        both = compile_model(model, ee.Image(INPUT), BANDS, GRID, clip_footprints=clip,
                             saved_nodes={relu_a: ee.Image("projects/example/assets/saved/a"),
                                          relu_b: ee.Image("projects/example/assets/saved/b")})
        assert _reads(both)["images"] == ["projects/example/assets/saved/a", "projects/example/assets/saved/b"]
        one = compile_model(model, ee.Image(INPUT), BANDS, GRID, clip_footprints=clip,
                            saved_nodes={relu_b: ee.Image("projects/example/assets/saved/b")})
        assert _reads(one)["images"] == sorted([INPUT, "projects/example/assets/saved/b"])
        assert _reads(whole)["images"] == [INPUT]


def test_saved_nodes_are_refused_where_they_cannot_be_read_exactly():
    gated = _gated()
    chain = _chain()
    with offline_earth_engine():
        whole = compile_model(gated, _input(), BANDS, GRID)
        image = ee.Image("projects/example/assets/saved/node")
        for name, match in (("nope", "not a tensor operation"), ("input_1", "not a tensor operation"),
                            ("output", "not a tensor operation")):
            with pytest.raises(UnsupportedOperation, match=match):
                compile_model(gated, _input(), BANDS, GRID, saved_nodes={name: image})
        pool = next(node.name for node in whole.graph_module.graph.nodes if node.target is F.adaptive_avg_pool2d)
        with pytest.raises(UnsupportedOperation, match="one-pixel tensor"):
            compile_model(gated, _input(), BANDS, GRID, saved_nodes={pool: image})
        with pytest.raises(TypeError, match="ee.Image"):
            compile_model(gated, _input(), BANDS, GRID, saved_nodes={pool: "projects/example/assets/saved/node"})
        # the convolution BatchNorm folds away, not built with include_unfused_nodes=False
        conv = _node_for(compile_model(chain, _input(), BANDS, GRID), "0.0")
        with pytest.raises(UnsupportedOperation, match="compute no raster tensor"):
            compile_model(chain, _input(), BANDS, GRID, include_unfused_nodes=False, saved_nodes={conv: image})
