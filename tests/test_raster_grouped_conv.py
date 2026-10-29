"""Offline grouped/depthwise convolution: array arithmetic, FX folding and weight assets."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import patch

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest
import torch
from torch import nn

from tensor_ee import Grid, Raster, UnsupportedOperation, check_compatibility, compile_model
from tensor_ee.blocks.raster import conv as C
from tensor_ee.core.precision import DOUBLE_ROUND_ONCE
from tensor_ee.layouts.raster import _float32
from tensor_ee.receptive_field import analyze_receptive_field
from tensor_ee.weights import constants as K
from tensor_ee.weight_assets import effective_convolutions
from reference.raster_conv import grouped_conv2d as mirror
from tools.fixtures.raster import _GROUPED, _Grouped, _grouped_data, _grouped_module


@pytest.fixture(autouse=True)
def random_state():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        yield


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


def compile_channels(model, channels, **kwargs):
    bands = tuple(f"b{i}" for i in range(channels))
    image = ee.Image.constant(list(range(1, channels + 1))).rename(list(bands))
    return compile_model(model.eval(), image, bands, grid(), **kwargs)


def _functions(value):
    if isinstance(value, dict):
        return ([value["functionName"]] if "functionName" in value else []) + [
            name for child in value.values() for name in _functions(child)]
    if isinstance(value, list):
        return [name for child in value for name in _functions(child)]
    return []


@pytest.mark.parametrize("groups", [2, 4, 32])
def test_one_product_and_reduction_for_all_groups_and_only_actual_weights_embedded(offline_ee, groups):
    module = nn.Conv2d(128, 128, 3, padding=1, groups=groups).eval()
    bands = tuple(f"x{i}" for i in range(128))
    raster = Raster(ee.Image.constant(list(range(128))).rename(list(bands)), bands, grid())
    values = []
    original = C._inline_float32_array
    with patch.object(C, "_inline_float32_array", side_effect=lambda rows: values.extend(
            v for row in rows for v in row) or original(rows)):
        out = C.grouped_conv2d(raster, module)
    names = _functions(json.loads(out.image.serialize()))
    assert names.count("Image.multiply") == names.count("Image.arrayReduce") == 1
    assert names.count("Image.arrayRepeat") == names.count("Image.neighborhoodToBands") == 1
    assert "Image.matrixMultiply" not in names and "Image.toFloat" not in names
    assert len(values) == module.weight.numel() + module.bias.numel() == C.grouped_conv2d_values(module)
    assert C.grouped_conv2d_arrays(module) == 2
    assert out.bands == C._names(128)


@pytest.mark.parametrize("spec", _GROUPED, ids=lambda s: s.name)
def test_inline_and_parameters_paths_build_the_same_graph(offline_ee, spec):
    data = _grouped_data(spec)(61)
    module = _grouped_module(spec, data)
    bands = tuple(f"z{i}" for i in range(spec.inputs))
    raster = Raster(ee.Image.constant(list(range(spec.inputs))).rename(list(bands)), bands, grid())
    parameters = (K.inline_float32_array(module.weight.detach().reshape(spec.outputs, -1).tolist()),
                  None if module.bias is None else K.inline_float32_array(module.bias.detach().reshape(-1, 1).tolist()))
    a = C.grouped_conv2d(raster, module)
    b = C.grouped_conv2d(raster, module, parameters)
    assert a.image.serialize() == b.image.serialize()
    assert a.grid == grid().resized(spec.stride, (7 + spec.stride - 1) // spec.stride,
                                    (5 + spec.stride - 1) // spec.stride)
    assert C.grouped_conv2d_arrays(module) == (2 if spec.bias else 1)
    assert 'Image.clip' in C.grouped_conv2d(raster, module, clip_footprints=True).image.serialize()
    with pytest.raises(UnsupportedOperation, match="not implemented"):
        C.grouped_conv2d(raster, module, policy=DOUBLE_ROUND_ONCE)


@pytest.mark.parametrize("bad,reason", [
    ("groups1", "groups > 1"), ("groups3", "dividing"), ("groups_float", "groups > 1"),
    ("stride", "stride=1 or 2"), ("dilation", "dilation=1"), ("padding", "same padding"),
    ("padding_mode", "padding_mode"), ("even", "odd height"), ("same_stride2", "stride-2 'same'"),
    ("dtype", "float32"), ("shape", "shape must"), ("nonfinite", "finite"),
])
def test_refusals_share_the_existing_spatial_and_coefficient_checks(offline_ee, bad, reason):
    module = nn.Conv2d(4, 4, 3, padding=1, groups=4).eval()
    if bad == "groups1": module.groups = 1
    if bad == "groups3": module.groups = 3
    if bad == "groups_float": module.groups = 4.0
    if bad == "stride": module.stride = (3, 3)
    if bad == "dilation": module.dilation = (2, 2)
    if bad == "padding": module.padding = (0, 1)
    if bad == "padding_mode": module.padding_mode = "reflect"
    if bad == "even": module.kernel_size = (2, 2)
    if bad == "same_stride2": module.padding, module.stride = "same", (2, 2)
    if bad == "dtype": module.double()
    if bad == "shape": module.weight = nn.Parameter(torch.zeros(4, 2, 3, 3))
    if bad == "nonfinite":
        with torch.no_grad(): module.weight[0, 0, 0, 0] = torch.nan
    raster = Raster(ee.Image.constant([1, 2, 3, 4]), ("constant_0", "constant_1", "constant_2", "constant_3"), grid())
    with pytest.raises(UnsupportedOperation, match=reason):
        C.grouped_conv2d(raster, module)
    if bad != "groups1":  # groups=1 is intentionally handled by the other block.
        assert check_compatibility(module, (1, 4, 5, 7)).status == "unsupported"


def test_grouped_block_and_legacy_block_keep_separate_acceptance_and_status(offline_ee):
    module = nn.Conv2d(4, 4, 3, padding=1, groups=4).eval()
    raster = Raster(ee.Image.constant([1, 2, 3]), ("constant_0", "constant_1", "constant_2"), grid())
    with pytest.raises(UnsupportedOperation, match="channel count"):
        C.grouped_conv2d(raster, module)
    with pytest.raises(UnsupportedOperation, match="groups=1"):
        C.conv_options(module)
    assert C.METADATA["conv2d"].validated.configurations == ((),)
    assert C.METADATA["grouped_conv2d"].status.startswith("Ready — depthwise")
    assert C.METADATA["grouped_conv2d"].radius == 2


def inverted_residual():
    return nn.Sequential(nn.Conv2d(4, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU6(),
                         nn.Conv2d(8, 8, 3, padding=1, stride=2, groups=8, bias=False),
                         nn.BatchNorm2d(8), nn.ReLU6(), nn.Conv2d(8, 4, 1, bias=False), nn.BatchNorm2d(4)).eval()


class GroupedResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(4, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
                                  nn.Conv2d(128, 128, 3, padding=1, groups=32, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 4, 1, bias=False), nn.BatchNorm2d(4))

    def forward(self, x):
        return self.body(x) + x


@pytest.mark.parametrize("factory,groups,spatial", [(inverted_residual, 8, (3, 4)), (GroupedResidual, 32, (5, 7))])
def test_grouped_architecture_folding_reports_and_assets_agree(offline_ee, factory, groups, spatial):
    model = factory().eval()
    before = deepcopy(model.state_dict())
    effective = effective_convolutions(model)
    seen = []
    def provider(key, module):
        seen.append((key, module.groups, tuple(module.weight.shape)))
        return (ee.Array(module.weight.detach().reshape(module.out_channels, -1).tolist()),
                None if module.bias is None else ee.Array(module.bias.detach().reshape(-1, 1).tolist()))
    compiled = compile_channels(model, 4, include_unfused_nodes=False, weight_provider=provider)
    assert len(compiled.folded_batchnorms) == len(effective) == len(seen) == 3
    assert [w.key for w in effective] == [key for key, _, _ in seen]
    assert [g for _, g, _ in seen] == [1, groups, 1]
    for asset, (_, g, shape) in zip(effective, seen):
        assert asset.weights.shape == (shape[0], int(np.prod(shape[1:])))
        if g > 1:
            assert asset.weights.shape[1] == (8 if groups == 8 else 128) // groups * 9
    report = check_compatibility(model, (1, 4, 5, 7)).to_dict()
    assert report["status"] == "supported" and report["output_shape"] == [1, 4, *spatial]
    assert report["weight_storage"]["coefficient_count"] == sum(w.weights.size + w.bias.size for w in effective)
    support = analyze_receptive_field(model)
    assert support.input_bounds(2, 1) == ((3, 1, 5, 3) if groups == 8 else (1, 0, 3, 2))
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_grouped_fold_is_the_same_fused_module_and_diagnostics_can_be_skipped(offline_ee):
    conv = nn.Conv2d(4, 8, 3, stride=2, padding=1, groups=4)
    bn = nn.BatchNorm2d(8)
    with torch.no_grad():
        bn.running_mean.copy_(torch.linspace(-0.5, 0.5, 8))
        bn.running_var.copy_(torch.linspace(0.1, 2, 8))
        bn.weight.copy_(torch.linspace(-1, 1, 8))
        bn.bias.copy_(torch.linspace(0.25, -0.25, 8))
    model = nn.Sequential(conv, bn).eval()
    fused = torch.nn.utils.fuse_conv_bn_eval(conv, bn)
    a = compile_channels(model, 4, include_unfused_nodes=False)
    b = compile_channels(nn.Sequential(fused), 4)
    assert a.image.serialize() == b.image.serialize() and a.folded_batchnorms == ("_1",)
    assert "_0" not in a.nodes
    assert "_0" in compile_channels(model, 4).nodes
    weights = effective_convolutions(model)
    assert [w.key for w in weights] == ["_1"]
    np.testing.assert_array_equal(weights[0].weights, fused.weight.detach().numpy().reshape(8, 9))
    np.testing.assert_array_equal(weights[0].bias, fused.bias.detach().numpy())


def test_shared_grouped_convolution_keeps_standalone_batchnorm(offline_ee):
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv, self.bn = nn.Conv2d(4, 4, 3, padding=1, groups=4), nn.BatchNorm2d(4)
        def forward(self, x):
            x = self.conv(x)
            return self.bn(x) + x
    model = Shared().eval()
    out = compile_channels(model, 4, include_unfused_nodes=False)
    assert out.folded_batchnorms == ()
    assert [w.key for w in effective_convolutions(model)] == ["conv"]
    assert check_compatibility(model, (1, 4, 5, 7)).to_dict()["weight_storage"]["coefficient_count"] == 56


class Array:
    def __init__(self, values): self.values = np.asarray(values, np.float64)
    def float(self): return Array(self.values.astype(np.float32))
    def double(self): return self
    def reshape(self, shape): return Array(self.values.reshape(shape))


class Image:
    """NumPy interpreter for the block's actual array graph and labelled neighbourhoods."""
    def __init__(self, value, bands=None):
        self.constant = isinstance(value, Array)
        self.values = value.values if self.constant else np.asarray(value, np.float64)
        self.bands = bands
    def neighborhoodToBands(self, kernel):
        kh, kw = kernel
        h, w, channels = self.values.shape
        padded = np.pad(self.values, ((kh // 2, kh // 2), (kw // 2, kw // 2), (0, 0)))
        # Deliberately use tap-major storage; the block must select channel-major labels.
        arrays, bands = [], []
        for ky in range(kh):
            for kx in range(kw):
                for c in range(channels):
                    arrays.append(padded[ky:ky + h, kx:kx + w, c])
                    bands.append(f"{self.bands[c]}_{kx - kw // 2}_{ky - kh // 2}")
        return Image(np.stack(arrays, -1), bands)
    def select(self, bands): return Image(self.values[..., [self.bands.index(b) for b in bands]], list(bands))
    def toArray(self): return self
    def arrayReshape(self, shape, dimensions):
        target = tuple(int(n) for n in shape.values)
        assert len(target) == dimensions
        return Image(self.values.reshape(*self.values.shape[:2], *target))
    def arrayRepeat(self, axis, copies): return Image(np.repeat(self.values, copies, axis=axis + 2))
    def _operand(self, other):
        assert self.values.shape[2:] == (other.values.shape if other.constant else other.values.shape[2:])
        return other.values
    def multiply(self, other): return Image(self.values * self._operand(other))
    def add(self, other): return Image(self.values + self._operand(other))
    def arrayReduce(self, reducer, axes): return Image(self.values.sum(tuple(a + 2 for a in axes), keepdims=True))
    def arrayProject(self, axes):
        value = self.values
        for axis in reversed(range(value.ndim - 2)):
            if axis not in axes:
                assert value.shape[axis + 2] == 1
                value = np.squeeze(value, axis + 2)
        return Image(value)
    def arrayFlatten(self, names): return Image(self.values, names[0])


@pytest.mark.parametrize("spec", (*_GROUPED, _Grouped("groups32-stride2", inputs=128, outputs=128, groups=32, stride=2)),
                         ids=lambda s: s.name)
def test_values_group_order_stride_and_parameters_match_mirror_and_pytorch(monkeypatch, spec):
    fake = SimpleNamespace(Array=Array, Image=Image,
                           Kernel=SimpleNamespace(fixed=lambda kw, kh, *args: (kh, kw)),
                           Reducer=SimpleNamespace(sum=lambda: "sum"))
    monkeypatch.setattr(C, "ee", fake)
    monkeypatch.setattr(K, "ee", fake)
    monkeypatch.setattr(C, "_zero_extended", lambda raster: raster.image)
    monkeypatch.setattr(C, "_project", lambda image, grid, clip: image)
    def sample(image, bands, g, stride, clip):
        return Raster(Image(image.values[::stride, ::stride], bands), bands,
                      g.resized(stride, (g.width + stride - 1) // stride, (g.height + stride - 1) // stride))
    monkeypatch.setattr(C, "_stride_sample", sample)
    data = _grouped_data(spec)(61)
    x = ((np.arange(5 * 7 * spec.inputs).reshape(5, 7, spec.inputs) % 97 - 48) / 64).astype(np.float32)
    module = _grouped_module(spec, data)
    bands = tuple(f"z{i}" for i in range(spec.inputs))
    raster = Raster(Image(x, bands), bands, grid())
    parameters = (Array(data.weights["weight"].reshape(spec.outputs, -1)),
                  None if not spec.bias else Array(data.weights["bias"].reshape(-1, 1)))
    a = C.grouped_conv2d(raster, module).image.values.astype(np.float32)
    b = C.grouped_conv2d(raster, module, parameters).image.values.astype(np.float32)
    expected = mirror(x, data.weights["weight"], data.weights.get("bias"), groups=spec.groups, stride=spec.stride)
    np.testing.assert_array_equal(a, expected)
    np.testing.assert_array_equal(a, b)
    with torch.no_grad():
        torch_out = module(torch.from_numpy(x.transpose(2, 0, 1)[None])).numpy()[0].transpose(1, 2, 0)
    assert np.max(np.abs(a - torch_out)) <= 1e-4
