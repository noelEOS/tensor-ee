"""Raster global average pooling and the elementwise multiply, offline.

- ``global_avg_pool2d`` shares ``pool2d``'s aligned expression (``_aligned_reduce``): on a
  k x k map it is byte-identical to ``pool2d(AvgPool2d(k))``; on any map its output is on
  ``one_pixel(grid)``, the one pixel covering exactly the map.
- ``multiply`` accepts a same-grid operand or a one-pixel gate on the map's own one-pixel
  grid, with the map's channel count, and refuses everything else.
- ``compile_model`` recognizes ``AdaptiveAvgPool2d(1)``, ``F.adaptive_avg_pool2d(x, 1)``,
  ``torch.mean``/``Tensor.mean`` over (2, 3) with ``keepdim=True``, and ``*``, ``torch.mul``
  and ``Tensor.mul``, in either operand order; the gate's convolutions and activations run
  on the one-pixel grid.
- ``analyze_receptive_field`` lists a global mean in ``extent_nodes`` and refuses every
  halo, so tiled compilation refuses the model; ``check_compatibility`` reports the stand-ins
  supported and the EfficientNet-B3 U-Net's pools and products no longer blocked.

Graphs are built against ``ee.apitestcase.ApiTestCase``; no Earth Engine request.
"""
from __future__ import annotations

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tensor_ee import Grid, Raster, UnsupportedOperation, check_compatibility, compile_model, from_image
from tensor_ee.blocks.raster import activations as A, merge as MG, pooling as PL
from tensor_ee.core.grid import one_pixel
from tensor_ee.core.metadata import NOT_LIVE, Locality
from tensor_ee.receptive_field import UnsupportedReceptiveField, analyze_receptive_field
from tensor_ee.tiling import compile_tiled_model, plan_tiles
from reference.raster_gate import global_avg_pool2d as pool_mirror, multiply as multiply_mirror


@pytest.fixture
def offline_ee():
    case = ApiTestCase()
    case.setUp()
    try:
        yield
    finally:
        case.tearDown()


def grid(width=7, height=5, scale=10):
    return Grid("EPSG:3857", (scale, 0, 1000, 0, -scale, 2000), width, height)


def raster(g, channels=2):
    names = [f"b{i}" for i in range(channels)]
    return from_image(ee.Image.constant([0.0] * channels).rename(names), names, g)


# ---------------------------------------------------------------------------
# one_pixel and global_avg_pool2d
# ---------------------------------------------------------------------------
def test_one_pixel_covers_exactly_the_map():
    g = grid(7, 5)
    one = one_pixel(g)
    assert (one.width, one.height, one.crs) == (1, 1, g.crs)
    assert one.transform == (70, 0, 1000, 0, -50, 2000)   # a non-square pixel for a non-square map
    for k in (2, 3, 4, 128):
        square = grid(k, k)
        assert one_pixel(square) == square.resized(k, 1, 1)
        assert one_pixel(square).transform == square.resized(k, 1, 1).transform
    assert one_pixel(one) == one


@pytest.mark.parametrize("k", [2, 3, 4])
def test_global_pool_of_a_k_by_k_map_is_pool2d_avg_k(offline_ee, k):
    """One implementation: the aligned k x k average and the global mean of a k x k map are
    the same expression, byte for byte."""
    x = raster(grid(k, k))
    pooled, aligned = PL.global_avg_pool2d(x), PL.pool2d(x, nn.AvgPool2d(k))
    assert pooled.image.serialize() == aligned.image.serialize()
    assert (pooled.bands, pooled.grid) == (aligned.bands, aligned.grid)


def test_global_pool_reduces_onto_the_one_pixel_grid(offline_ee):
    x = raster(grid(20, 12), 3)
    pooled = PL.global_avg_pool2d(x)
    assert pooled.grid == one_pixel(x.grid) and pooled.bands == x.bands
    text = pooled.image.serialize()
    assert text.count('"Image.reduceResolution"') == 1 and '"Reducer.mean"' in text
    assert "308" in text                                  # maxPixels (12 + 2)(20 + 2)
    with pytest.raises(UnsupportedOperation, match="raster tensor"):
        PL.global_avg_pool2d(ee.Image.constant(0))


def test_metadata_is_offline_only_and_extent():
    pool, product = PL.METADATA["global_avg_pool2d"], MG.METADATA["multiply"]
    assert pool.locality is Locality.EXTENT and product.locality is Locality.POINTWISE
    assert pool.validated == product.validated == NOT_LIVE
    assert not pool.chip_edge_dependent and not product.chip_edge_dependent
    for entry in (pool, product):
        assert entry.status.startswith("Fixed — offline only: ") and "Ready" not in entry.status
        assert entry.embedded_constants() == entry.asset_arrays() == 0


# ---------------------------------------------------------------------------
# multiply: the two forms and the refusals
# ---------------------------------------------------------------------------
def test_multiply_same_grid_and_one_pixel_gate(offline_ee):
    x = raster(grid())
    same = MG.multiply(x, x)
    assert (same.bands, same.grid) == (x.bands, x.grid)
    gate = PL.global_avg_pool2d(x)
    gated = MG.multiply(x, gate)
    assert (gated.bands, gated.grid) == (x.bands, x.grid)
    text = gated.image.serialize()
    assert text.count('"Image.multiply"') == 1 and '"Image.reproject"' in text


def test_multiply_refuses_every_other_form(offline_ee):
    x = raster(grid())
    gate = PL.global_avg_pool2d(x)
    other_map = PL.global_avg_pool2d(raster(grid(8, 5)))           # a gate pooled from another map
    with pytest.raises(UnsupportedOperation, match="scalar"):
        MG.multiply(x, 2.0)
    with pytest.raises(UnsupportedOperation, match="no channel broadcasting"):
        MG.multiply(x, raster(grid(), 1))                            # a [N, 1, H, W] spatial gate
    with pytest.raises(UnsupportedOperation, match="no channel broadcasting"):
        MG.multiply(x, PL.global_avg_pool2d(raster(grid(), 3)))
    with pytest.raises(UnsupportedOperation, match="neither"):
        MG.multiply(x, other_map)
    with pytest.raises(UnsupportedOperation, match="neither"):
        MG.multiply(x, raster(grid(scale=20)))
    with pytest.raises(UnsupportedOperation, match="neither"):
        MG.multiply(gate, x)                                         # the block takes the map first


def test_relu_runs_on_the_one_pixel_grid(offline_ee):
    pooled = PL.global_avg_pool2d(raster(grid(20, 12)))
    out = A.relu(pooled)
    assert out.grid == one_pixel(grid(20, 12)) and out.bands == pooled.bands


# ---------------------------------------------------------------------------
# Mirrors (NumPy) against PyTorch, locally
# ---------------------------------------------------------------------------
def test_mirrors_agree_with_pytorch():
    from tools.fixtures.core import dyadic, hwc, nchw

    x = dyadic(3, "x", (12, 20, 3), scale=4.0, offset=2.0)
    g = dyadic(3, "g", (1, 1, 3), scale=0.25, offset=0.5, bits=23)
    with torch.no_grad():
        pooled = hwc(F.adaptive_avg_pool2d(nchw(x), 1))
        product = hwc(nchw(g) * nchw(x))
    assert np.abs(pool_mirror(x).astype(np.float64) - pooled).max() <= 2.4e-7
    exact = np.add.reduce(x.astype(np.float64), axis=(0, 1)) / (12 * 20)
    assert np.array_equal(pool_mirror(x)[0, 0], exact.astype(np.float32))
    assert np.array_equal(multiply_mirror(x, g), product)                 # the same correctly rounded product
    assert np.array_equal(multiply_mirror(x, g), multiply_mirror(x, np.broadcast_to(g, x.shape)))
    with pytest.raises(ValueError):
        multiply_mirror(x, g[:, :, :1])


# ---------------------------------------------------------------------------
# compile_model: recognition, operand order, refusals
# ---------------------------------------------------------------------------
class Functional(nn.Module):
    def __init__(self, function, *modules):
        super().__init__()
        self.layers = nn.ModuleList(modules)
        self.function = function

    def forward(self, x):
        return self.function(self, x)


def squeeze_excitation(left=True, activation=None):
    """A squeeze-excitation stand-in with nn.Conv2d 1 x 1 (SMP MBConv's order: sigmoid(g) * x)."""
    class SqueezeExcite(nn.Module):
        def __init__(self):
            super().__init__()
            self.reduce = nn.Conv2d(4, 2, 1)
            self.act = activation or nn.SiLU()
            self.expand = nn.Conv2d(2, 4, 1)

        def forward(self, x):
            gate = torch.sigmoid(self.expand(self.act(self.reduce(F.adaptive_avg_pool2d(x, 1)))))
            return gate * x if left else x * gate

    model = SqueezeExcite()
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.copy_(torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape) / 16
                            - index / 8)
    return model.eval()


def compile_on(model, g, channels=4):
    names = [f"b{i}" for i in range(channels)]
    return compile_model(model, ee.Image.constant([0.0] * channels).rename(names), names, g)


def test_every_global_mean_form_compiles_to_the_same_graph(offline_ee):
    g = grid(20, 12)
    forms = [nn.AdaptiveAvgPool2d(1), nn.AdaptiveAvgPool2d((1, 1)),
             Functional(lambda self, x: F.adaptive_avg_pool2d(x, 1)),
             Functional(lambda self, x: F.adaptive_avg_pool2d(x, [1, 1])),
             Functional(lambda self, x: torch.mean(x, (2, 3), keepdim=True)),
             Functional(lambda self, x: torch.mean(x, dim=[-1, -2], keepdim=True)),
             Functional(lambda self, x: x.mean((3, 2), keepdim=True)),
             Functional(lambda self, x: x.mean(dim=(-2, -1), keepdim=True))]
    graphs = set()
    for model in forms:
        compiled = compile_on(model.eval(), g)
        assert compiled.grid == one_pixel(g) and compiled.bands == ("c0", "c1", "c2", "c3")
        graphs.add(compiled.image.serialize())
    assert len(graphs) == 1
    nodes = compile_on(nn.Identity().eval(), g).nodes
    placeholder = nodes[next(iter(nodes))]                              # the float32 input node
    assert graphs == {PL.global_avg_pool2d(placeholder).image.toFloat().serialize()}


def test_every_multiply_form_and_order_compiles_to_the_same_graph(offline_ee):
    g = grid(20, 12)
    def graphs(*functions):
        return {compile_on(Functional(f).eval(), g).image.serialize() for f in functions}

    # Same grid: the call forms agree for one operand order (the order is kept: either
    # graph gives the same correctly rounded products).
    assert len(graphs(lambda self, x: x * torch.sigmoid(x), lambda self, x: torch.mul(x, torch.sigmoid(x)),
                      lambda self, x: x.mul(torch.sigmoid(x)), lambda self, x: x.mul(other=torch.sigmoid(x)))) == 1
    assert len(graphs(lambda self, x: torch.sigmoid(x) * x, lambda self, x: torch.mul(torch.sigmoid(x), x))) == 1
    # A one-pixel gate: the map is x in either operand order, so the graphs are identical.
    left, right = squeeze_excitation(True), squeeze_excitation(False)
    a, b = compile_on(left, g), compile_on(right, g)
    assert a.image.serialize() == b.image.serialize()
    assert (a.grid, a.bands) == (g, ("c0", "c1", "c2", "c3"))
    gate = a.nodes["sigmoid"]
    assert gate.grid == one_pixel(g) and a.nodes["reduce"].grid == one_pixel(g)
    assert a.nodes["adaptive_avg_pool2d"].grid == one_pixel(g)


def test_squeeze_excitation_with_relu_compiles_on_the_one_pixel_grid(offline_ee):
    compiled = compile_on(squeeze_excitation(activation=nn.ReLU()), grid(16, 16))
    assert compiled.nodes["act"].grid == one_pixel(grid(16, 16))
    assert compiled.grid == grid(16, 16)


@pytest.mark.parametrize("function, message", [
    (lambda self, x: F.adaptive_avg_pool2d(x, 2), "only to one pixel"),
    (lambda self, x: F.adaptive_avg_pool2d(x, (1, None)), "only to one pixel"),
    (lambda self, x: F.adaptive_avg_pool2d(x, True), "only to one pixel"),
    (lambda self, x: F.adaptive_max_pool2d(x, 1), "Unsupported function"),
    (lambda self, x: torch.amax(x, (2, 3), keepdim=True), "Unsupported function"),
    (lambda self, x: x.mean((2, 3)), "keepdim=True"),
    (lambda self, x: torch.mean(x, (2, 3), keepdim=False), "keepdim=True"),
    (lambda self, x: x.mean((1, 2), keepdim=True), "both spatial axes"),
    (lambda self, x: x.mean(3, keepdim=True), "both spatial axes"),
    (lambda self, x: x.mean((True, 3), keepdim=True), "both spatial axes"),
    (lambda self, x: x.mean((2, 3), keepdim=1), "keepdim=True"),
    (lambda self, x: x.mean((2, 3), keepdim=True, dtype=torch.float32), "dtype=None"),
    (lambda self, x: x * 2, "tensor inputs"),
    (lambda self, x: torch.mul(x, 0.5), "tensor inputs"),
    (lambda self, x: x * x[:, :1], "getitem"),
])
def test_unsupported_pooling_and_products_are_refused(offline_ee, function, message):
    with pytest.raises(UnsupportedOperation, match=message):
        compile_on(Functional(function).eval(), grid(8, 8))


def test_broadcasts_and_foreign_gates_are_refused(offline_ee):
    channel_gate = Functional(lambda self, x: x * torch.sigmoid(self.layers[0](x)), nn.Conv2d(4, 1, 1)).eval()
    with pytest.raises(UnsupportedOperation, match="Multiplication requires identical shapes"):
        compile_on(channel_gate, grid(8, 8))
    # A gate pooled from the stride-2 map, multiplying the full map: the scales differ.
    strided = Functional(lambda self, x: x * F.adaptive_avg_pool2d(self.layers[0](x), 1),
                         nn.Conv2d(4, 4, 3, stride=2, padding=1)).eval()
    with pytest.raises(UnsupportedOperation, match="Multiplication requires identical shapes"):
        compile_on(strided, grid(8, 8))
    # Same scale and shape relation, another map: 15 -> 8 (stride 2) -> 16 (x 2), pooled, times
    # the 15 x 15 input. The shapes pass; the block's grid check refuses the foreign gate.
    foreign = Functional(lambda self, x: x * F.adaptive_avg_pool2d(
        F.interpolate(self.layers[0](x), scale_factor=2, mode="nearest"), 1),
        nn.Conv2d(4, 4, 3, stride=2, padding=1)).eval()
    with pytest.raises(UnsupportedOperation, match="neither"):
        compile_on(foreign, grid(15, 15))
    with pytest.raises(UnsupportedOperation, match="Unsupported module AdaptiveMaxPool2d"):
        compile_on(nn.Sequential(nn.AdaptiveMaxPool2d(1)).eval(), grid(8, 8))


# ---------------------------------------------------------------------------
# Receptive field and tiling: a global mean needs the whole chip
# ---------------------------------------------------------------------------
def test_receptive_field_lists_the_global_mean_and_refuses_halos():
    report = analyze_receptive_field(squeeze_excitation())
    assert report.extent_nodes == ("adaptive_avg_pool2d",)
    for halo in ("safe_halo", "halo_for_any_phase"):
        with pytest.raises(UnsupportedReceptiveField, match="whole chip"):
            getattr(report, halo)
    metadata = report.as_dict()
    assert metadata["safe_halo"] is metadata["halo_for_any_phase"] is None
    assert metadata["extent_nodes"] == ["adaptive_avg_pool2d"]
    assert metadata["context_requirement"] == "full source extent required"
    root = analyze_receptive_field(nn.AdaptiveAvgPool2d((1, 1)))
    assert root.extent_nodes == ("_0",)
    # Without a global mean nothing changes: no extent_nodes key.
    assert "extent_nodes" not in analyze_receptive_field(nn.Conv2d(2, 2, 3, padding=1)).as_dict()


def test_tiled_compilation_refuses_a_model_with_a_global_mean(offline_ee):
    plan = plan_tiles(grid(16, 16), 8, 4, alignment=1)
    assert len(plan.tiles) == 4
    with pytest.raises(UnsupportedReceptiveField, match="whole chip"):
        compile_tiled_model(squeeze_excitation(), ee.Image.constant([0.0] * 4).rename(["a", "b", "c", "d"]),
                            ["a", "b", "c", "d"], plan)


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------
class MBConv(nn.Module):
    """An EfficientNet MBConv stand-in with module convolutions: expand 1 x 1, depthwise 3 x 3,
    squeeze-excitation (SiLU), project 1 x 1, BatchNorm after each convolution, identity skip."""

    def __init__(self, channels=4, expanded=8, squeezed=2):
        super().__init__()
        self.expand = nn.Sequential(nn.Conv2d(channels, expanded, 1, bias=False), nn.BatchNorm2d(expanded), nn.SiLU())
        self.depthwise = nn.Sequential(nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded, bias=False),
                                       nn.BatchNorm2d(expanded), nn.SiLU())
        self.se_reduce, self.se_act = nn.Conv2d(expanded, squeezed, 1), nn.SiLU()
        self.se_expand = nn.Conv2d(squeezed, expanded, 1)
        self.project = nn.Sequential(nn.Conv2d(expanded, channels, 1, bias=False), nn.BatchNorm2d(channels))

    def forward(self, x):
        y = self.depthwise(self.expand(x))
        gate = self.se_expand(self.se_act(self.se_reduce(F.adaptive_avg_pool2d(y, 1))))
        return self.project(torch.sigmoid(gate) * y) + x


def test_compatibility_reports_the_stand_ins_supported(offline_ee):
    for model, shape, pooled in ((squeeze_excitation(), (1, 4, 12, 20), 4), (MBConv().eval(), (1, 4, 16, 16), 8)):
        report = check_compatibility(model, shape).to_dict()
        assert report["status"] == "supported", report["blockers"]
        assert report["output_shape"] == list(shape)
        assert any("global pooling" in restriction for restriction in report["restrictions"])
        nodes = {node["name"]: node for node in report["nodes"]}
        assert nodes["adaptive_avg_pool2d"]["output_shape"] == [1, pooled, 1, 1]
    compiled = compile_on(MBConv().eval(), grid(16, 16))
    assert compiled.grid == grid(16, 16)
    assert "global pooling" not in " ".join(check_compatibility(nn.Conv2d(2, 2, 1).eval(), (1, 2, 8, 8))
                                            .to_dict()["restrictions"])


def test_efficientnet_b3_unet_pools_and_products_are_no_longer_blocked():
    """Trace only, no forward: SMP 0.5.0's EfficientNet-B3 U-Net at 256 px keeps its other
    blockers (functional conv2d with get_attr weights, ZeroPad2d, in-place ReLU, ...), but its 26
    squeeze-excitation pools and products are no longer among them."""
    smp = pytest.importorskip("segmentation_models_pytorch")
    model = smp.Unet("efficientnet-b3", encoder_weights=None, in_channels=8, classes=3).eval()
    report = check_compatibility(model, (1, 8, 256, 256)).to_dict()
    nodes = {node["name"]: node for node in report["nodes"]}
    pools = [n for n in nodes.values() if n["target"].endswith("functional.adaptive_avg_pool2d")]
    products = [n for n in nodes.values() if n["target"] == "_operator.mul"]
    assert len(pools) == len(products) == 26
    blocked = {b["node"] for b in report["blockers"]}
    assert not blocked & {n["name"] for n in pools + products}
    assert all(n["status"] != "unsupported" for n in pools + products)
    assert not any("adaptive_avg_pool2d" in b["message"] or "mul" in b["message"].split() for b in report["blockers"])
    assert report["status"] == "unsupported"          # the functional convolutions remain
