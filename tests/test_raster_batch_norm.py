"""Standalone raster inference BatchNorm: arithmetic, FX routing and strict constants.

All graphs use offline API signatures; the numerical mirror is the decoder's
existing channel-last BatchNorm, also reused by the two raster fixtures.
"""
from types import SimpleNamespace
from unittest.mock import patch

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.fx import symbolic_trace

from tensor_ee import Grid, Raster, UnsupportedOperation, check_compatibility, compile_model
from tensor_ee.blocks.raster import normalization as N
from tensor_ee.core.precision import DOUBLE_ROUND_ONCE
from tensor_ee.layouts.raster import _float32
from tensor_ee.receptive_field import analyze_receptive_field
from tensor_ee.weight_assets import effective_convolutions
from tools.fixtures.raster import _batch_norm_data, _batch_norm_module, _batch_norm_mirror


@pytest.fixture
def offline_ee():
    case = ApiTestCase()
    case.setUp()
    try:
        yield
    finally:
        case.tearDown()


def grid():
    return Grid("EPSG:3857", (20, 0, 1000, 0, -20, 2000), 8, 8)


def compile_three(model, **kwargs):
    return compile_model(model, ee.Image.constant([1, 2, 3]).rename(["z", "a", "m"]),
                         ("z", "a", "m"), grid(), **kwargs)


class FunctionalBatchNorm(nn.Module):
    """The F.batch_norm call and get_attr parameters used by BatchNormAct2d."""
    def __init__(self, affine=True, training=False, momentum=0.1, eps=1e-5):
        super().__init__()
        source = _batch_norm_module(_batch_norm_data(41), affine)
        self.register_buffer("running_mean", source.running_mean)
        self.register_buffer("running_var", source.running_var)
        self.weight = source.weight
        self.bias = source.bias
        self.bn_training = training
        self.momentum = momentum
        self.eps = eps

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.bn_training, self.momentum, self.eps)


def constants(bands=("z", "a", "m"), affine=True):
    w = _batch_norm_data(41).weights
    return N.BatchNormConstants(w["gamma"] if affine else None, w["beta"] if affine else None,
                                w["mean"], w["var"], bands=bands)


@pytest.mark.parametrize("affine", [True, False])
def test_constant_graph_order_and_materialization(offline_ee, affine):
    c = constants(affine=affine)
    raster = Raster(ee.Image.constant([3, 1, 2]).rename(["m", "z", "a"]), ("z", "a", "m"), grid())
    out = N.batch_norm(raster, c)
    assert out.bands == raster.bands and out.grid == raster.grid
    assert c.bands == raster.bands and c.embedded_values == 12
    encoded = out.image.serialize()
    assert 'Array.sqrt' in encoded and 'Image.sqrt' not in encoded
    assert 'Image.toFloat' not in encoded  # Only array coefficients have their float cast.
    assert 'Image.toFloat' in _float32(out).image.serialize()
    assert 'Image.clip' in N.batch_norm(raster, c, clip_footprints=True).image.serialize()
    with pytest.raises(UnsupportedOperation, match="channel count"):
        N.batch_norm(Raster(raster.image, ("z", "a"), grid()), c)
    with pytest.raises(UnsupportedOperation, match="not implemented"):
        N.batch_norm(raster, c, policy=DOUBLE_ROUND_ONCE)


@pytest.mark.parametrize("field,value", [
    ("mean", np.zeros(2, np.float32)), ("var", np.ones(3, np.float64)),
    ("gamma", np.array([1, np.inf, 1], np.float32)),
    ("beta", np.array([1, np.nan, 1], np.float32)),
    ("var", np.array([1, -1, 1], np.float32)),
    ("eps", True), ("eps", float("nan")), ("eps", float("inf")),
    ("bands", ("z", "z", "m")),
])
def test_constants_refuse_invalid_settings_before_building_images(offline_ee, field, value):
    parameters = dict(_batch_norm_data(41).weights, bands=("z", "a", "m"))
    parameters[field] = value
    with pytest.raises(UnsupportedOperation):
        N.BatchNormConstants(**parameters)


class Numeric:
    """Small NumPy interpreter of the scalar-band/constant-array ops used here."""
    def __init__(self, values, bands=None):
        if isinstance(values, Numeric):
            self.values, self.bands = values.values, values.bands
        else:
            self.values, self.bands = np.asarray(values, np.float64), bands

    def float(self): return Numeric(self.values.astype(np.float32).astype(np.float64), self.bands)
    def double(self): return Numeric(self.values.astype(np.float64), self.bands)
    def sqrt(self): return Numeric(np.sqrt(self.values), self.bands)
    def arrayFlatten(self, names): return Numeric(self.values, tuple(names[0]))
    def select(self, bands):
        return Numeric(self.values[..., [self.bands.index(b) for b in bands]], tuple(bands))
    def _binary(self, other, op):
        return Numeric(op(self.values, other.values if isinstance(other, Numeric) else other), self.bands)
    def add(self, other): return self._binary(other, np.add)
    def subtract(self, other): return self._binary(other, np.subtract)
    def divide(self, other): return self._binary(other, np.divide)
    def multiply(self, other): return self._binary(other, np.multiply)


@pytest.mark.parametrize("affine", [True, False])
def test_values_and_positional_bands_match_the_mirror_and_pytorch(monkeypatch, affine):
    # Exercise the builder and block against NumPy ops as well as building real EE graphs.
    monkeypatch.setattr(N, "ee", SimpleNamespace(Array=Numeric, Image=Numeric))
    monkeypatch.setattr(N, "exact_scalar", float)
    monkeypatch.setattr(N, "_finish", lambda image, bands, grid, clip: Raster(image, bands, grid))
    data = _batch_norm_data(41)
    x = data.inputs["x"]
    # Physical storage m,z,a differs from Raster order z,a,m. Constant names
    # differ too: image arithmetic must pair their values by position.
    raster = Raster(Numeric(x[..., [2, 0, 1]], ("m", "z", "a")), ("z", "a", "m"), grid())
    c = constants(bands=("third", "first", "second"), affine=affine)
    result = N.batch_norm(raster, c).image.values.astype(np.float32)
    mirror = _batch_norm_mirror(affine)(data)["out"].astype(np.float32)
    np.testing.assert_array_equal(result, mirror)
    with torch.no_grad():
        expected = _batch_norm_module(data, affine)(torch.from_numpy(x.transpose(2, 0, 1)[None]))
    expected = expected.numpy()[0].transpose(1, 2, 0)
    assert np.max(np.abs(result - expected)) <= 1e-4
    assert c.mean.values.dtype == c.std.values.dtype == np.float64
    np.testing.assert_array_equal(c.mean.values, data.weights["mean"].astype(np.float64))
    np.testing.assert_array_equal(c.std.values, np.sqrt(data.weights["var"].astype(np.float64) + 1e-5))
    np.testing.assert_array_equal(c.gamma.values, data.weights["gamma"] if affine else np.ones(3, np.float32))
    np.testing.assert_array_equal(c.beta.values, data.weights["beta"] if affine else np.zeros(3, np.float32))


@pytest.mark.parametrize("affine", [True, False])
def test_module_and_functional_graphs_are_the_same_standalone_block(offline_ee, affine):
    module = _batch_norm_module(_batch_norm_data(41), affine)
    functional = FunctionalBatchNorm(affine).eval()
    with patch("torch.nn.utils.fuse_conv_bn_eval", side_effect=AssertionError("unexpected fold")):
        a, b = compile_three(module), compile_three(functional)
    assert a.image.serialize() == b.image.serialize()
    assert a.folded_batchnorms == b.folded_batchnorms == ()
    assert len(a.nodes) == len(b.nodes) == 2  # get_attr coefficients are not raster activations.
    source = next(iter(a.nodes.values()))
    manual = _float32(N.batch_norm(source, constants(bands=source.bands, affine=affine)))
    assert a.image.serialize() == manual.image.serialize()
    for model in (module, functional):
        report = check_compatibility(model, (1, 3, 8, 8)).to_dict()
        assert report["status"] == "supported"
        assert report["output_shape"] == [1, 3, 8, 8]
        assert report["weight_storage"]["coefficient_count"] == 12
        support = analyze_receptive_field(model)
        assert support.input_bounds(4, 3) == (4, 3, 4, 3)


class AfterCat(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(6)
        self.conv = nn.Conv2d(6, 2, 1)

    def forward(self, x):
        return self.conv(self.bn(torch.cat([x, x], dim=1)))


class Shared(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.bn = nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x) + x


@pytest.mark.parametrize("model", [
    nn.Sequential(nn.BatchNorm2d(3), nn.ReLU(), nn.Conv2d(3, 2, 1)),
    AfterCat(), Shared(), nn.Sequential(FunctionalBatchNorm(), nn.ReLU(), nn.Conv2d(3, 2, 1))])
def test_standalone_routing_keeps_conv_assets_and_pointwise_support(offline_ee, model):
    model = model.eval()
    compiled = compile_three(model, include_unfused_nodes=False)
    assert compiled.folded_batchnorms == ()
    assert 'Image.divide' in compiled.image.serialize()
    weights = effective_convolutions(model)
    assert len(weights) == 1
    assert weights[0].weights.size in (6, 9, 12)
    assert analyze_receptive_field(model).input_bounds(4, 3) == (4, 3, 4, 3)
    report = check_compatibility(model, (1, 3, 8, 8)).to_dict()
    assert report["status"] == "supported"
    conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    bn_channels = 6 if isinstance(model, AfterCat) else 3
    assert report["weight_storage"]["coefficient_count"] == conv.weight.numel() + conv.bias.numel() + 4 * bn_channels


@pytest.mark.parametrize("training", [True, 0, 1, None])
def test_functional_training_requires_literal_false(offline_ee, training):
    model = FunctionalBatchNorm(training=training).eval()
    with pytest.raises(UnsupportedOperation, match="literal False"):
        compile_three(model)
    assert check_compatibility(model, (1, 3, 8, 8)).to_dict()["status"] == "unsupported"


@pytest.mark.parametrize("momentum", [None, 0, 0.1, 0.9, "ignored"])
def test_functional_momentum_is_irrelevant_in_eval(offline_ee, momentum):
    assert compile_three(FunctionalBatchNorm(momentum=momentum).eval()).folded_batchnorms == ()


@pytest.mark.parametrize("field", ["running_mean", "running_var", "weight", "bias"])
@pytest.mark.parametrize("invalid", ["shape", "nonfinite"])
def test_functional_coefficients_receive_the_same_static_checks(offline_ee, field, invalid):
    model = FunctionalBatchNorm().eval()
    tensor = getattr(model, field)
    with torch.no_grad():
        if invalid == "nonfinite":
            tensor[1] = torch.inf
        elif isinstance(tensor, nn.Parameter):
            setattr(model, field, nn.Parameter(torch.zeros(2)))
        else:
            setattr(model, field, torch.zeros(2))
    with pytest.raises(UnsupportedOperation):
        compile_three(model)
    assert check_compatibility(model, (1, 3, 8, 8)).to_dict()["status"] == "unsupported"


def test_functional_accepts_nested_attributes_keywords_and_independent_affine_defaults(offline_ee):
    model = FunctionalBatchNorm().eval()
    for name in ("weight", "bias"):
        model.register_parameter(name, None)
        compiled = compile_three(nn.Sequential(model).eval())
        assert compiled.folded_batchnorms == ()
    traced = symbolic_trace(nn.Sequential(FunctionalBatchNorm()).eval())
    bn = next(n for n in traced.graph.nodes if n.op == "call_function")
    bn.kwargs = dict(bn.kwargs, input=bn.args[0], running_mean=bn.args[1], running_var=bn.args[2])
    bn.args = ()
    traced.recompile()
    traced.eval()  # FX creates an attribute-only container for the nested parameters.
    assert compile_three(traced).image.serialize() == compile_three(FunctionalBatchNorm().eval()).image.serialize()


@pytest.mark.parametrize("eps", [True, float("inf"), float("nan")])
def test_functional_eps_is_static_and_finite(offline_ee, eps):
    with pytest.raises(UnsupportedOperation, match="eps"):
        compile_three(FunctionalBatchNorm(eps=eps).eval())


def test_functional_constants_survive_shape_specialization(offline_ee):
    class Resize(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = FunctionalBatchNorm()

        def forward(self, x):
            return F.interpolate(self.bn(x), size=x.shape[-2:], mode="nearest")

    model = Resize().eval()
    assert compile_three(model).grid == grid()
    assert check_compatibility(model, (1, 3, 8, 8)).to_dict()["status"] == "supported"


@pytest.mark.parametrize("folded", [True, False])
@pytest.mark.parametrize("bad", ["training", "untracked", "shape", "nonfinite", "eps", "variance"])
def test_static_refusals_apply_to_both_module_paths(offline_ee, folded, bad):
    bn = nn.BatchNorm2d(3).eval()
    if bad == "training": bn.train()
    if bad == "untracked": bn.track_running_stats = False
    if bad == "shape": bn.running_mean = torch.zeros(2)
    if bad == "nonfinite": bn.running_var[1] = torch.nan
    if bad == "eps": bn.eps = float("inf")
    if bad == "variance": bn.running_var[1] = -1
    model = nn.Sequential(nn.Conv2d(3, 3, 1).eval(), bn) if folded else bn
    model.eval()
    if bad == "training": bn.train()
    with pytest.raises(UnsupportedOperation):
        compile_three(model)


def test_channel_mismatch_and_nonconstant_statistics_are_refused(offline_ee):
    with pytest.raises(UnsupportedOperation, match="channel count"):
        compile_three(nn.BatchNorm2d(2).eval())
    model = FunctionalBatchNorm().eval()
    traced = symbolic_trace(model)
    bn = next(n for n in traced.graph.nodes if n.op == "call_function")
    x = next(n for n in traced.graph.nodes if n.op == "placeholder")
    args = list(bn.args)
    args[1] = x
    bn.args = tuple(args)
    for node in list(traced.graph.nodes):
        if node.op == "get_attr" and not node.users:
            traced.graph.erase_node(node)
    traced.recompile()
    with pytest.raises(UnsupportedOperation, match="get_attr buffers or parameters"):
        compile_three(traced)


@pytest.mark.parametrize("outside", ["add", "input", "momentum"])
def test_get_attr_remains_refused_outside_the_four_coefficient_inputs(offline_ee, outside):
    traced = symbolic_trace(FunctionalBatchNorm().eval())
    mean = next(n for n in traced.graph.nodes if n.op == "get_attr" and n.target == "running_mean")
    bn = next(n for n in traced.graph.nodes if n.op == "call_function")
    if outside == "add":
        with traced.graph.inserting_before(bn):
            extra = traced.graph.call_function(torch.add, (bn.args[0], mean))
        args = list(bn.args)
        args[0] = extra
    elif outside == "input":
        args = list(bn.args)
        args[0] = mean
    else:
        args = list(bn.args)
        bn.kwargs = dict(bn.kwargs, momentum=mean)
    bn.args = tuple(args)
    traced.recompile()
    with pytest.raises(UnsupportedOperation, match="Unsupported FX operation get_attr"):
        compile_three(traced)
    assert check_compatibility(traced, (1, 3, 8, 8)).to_dict()["status"] == "unsupported"


def test_conv_batchnorm_still_folds_and_uses_folded_assets(offline_ee):
    model = nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3)).eval()
    fused = torch.nn.utils.fuse_conv_bn_eval(model[0], model[1])
    compiled = compile_three(model, include_unfused_nodes=False)
    assert compiled.folded_batchnorms == ("_1",)
    assert compiled.image.serialize() == compile_three(nn.Sequential(fused).eval()).image.serialize()
    weights = effective_convolutions(model)
    assert [w.key for w in weights] == ["_1"]
    np.testing.assert_array_equal(weights[0].weights, fused.weight.detach().numpy().reshape(3, -1))
