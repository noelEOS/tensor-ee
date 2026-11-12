"""Zero padding ahead of a convolution: the sampling phase of ``conv2d`` and
``grouped_conv2d``, and the compiler's recognition of the padded forms. Offline only."""
from __future__ import annotations

from dataclasses import replace
import json
import re

import ee
from ee.apitestcase import ApiTestCase
import pytest
import torch
from torch import nn
from torch.fx import symbolic_trace
from torch.nn import functional as F

from tensor_ee import Grid, Raster, UnsupportedOperation, check_compatibility, compile_model
from tensor_ee.blocks.raster import conv as C
from tensor_ee.layouts import raster as L
from tensor_ee.receptive_field import UnsupportedReceptiveField, analyze_receptive_field
from tensor_ee.semantics import batchnorm_predecessor, convolution, semantic_call
from tensor_ee.weight_assets import effective_convolutions
from tensor_ee.weight_mapping import MatchedWeightProvider


@pytest.fixture
def offline_ee():
    case = ApiTestCase()
    case.setUp()
    try:
        yield
    finally:
        case.tearDown()


def _grid(width, height, scale=20):
    return Grid("EPSG:3857", (scale, 0, 1000, 0, -scale, 2000), width, height)


def _base_stride_sample(image, bands, grid, stride, clip_footprints=False):
    """``stride_sample`` as it stood before the phase setting, verbatim."""
    if stride == 1:
        return L.finish(image, bands, grid, clip_footprints)
    width = (grid.width + stride - 1) // stride
    height = (grid.height + stride - 1) // stride
    canonical = grid.resized(stride, width, height)
    a, b, c, d, e, f = canonical.transform
    sample = Grid(grid.crs,
                  (a, b, c - (stride - 1) * grid.transform[0] / 2,
                   d, e, f - (stride - 1) * grid.transform[4] / 2),
                  width, height)
    sampled = L.finish(image, bands, sample, clip_footprints)
    relabeled = sampled.image.changeProj(sample.projection(), canonical.projection())
    return L.finish(relabeled, bands, canonical, clip_footprints)


# ---------------------------------------------------------------------------
# The sampler: phase 0 is today's graph; a phase moves the sample grid exactly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("size", [(8, 8), (10, 8), (7, 5), (9, 9)])
@pytest.mark.parametrize("clip", [False, True])
@pytest.mark.parametrize("phase", [0, (0, 0), [0, 0]])
def test_the_sampler_at_phase_zero_is_todays_graph(offline_ee, stride, size, clip, phase):
    grid = _grid(*size)
    image = ee.Image.constant([0.5, -1.0]).rename(["c0", "c1"])
    base = _base_stride_sample(image, ("c0", "c1"), grid, stride, clip)
    default = L.stride_sample(image, ("c0", "c1"), grid, stride, clip)
    stated = L.stride_sample(image, ("c0", "c1"), grid, stride, clip, phase=phase)
    assert base.image.serialize() == default.image.serialize() == stated.image.serialize()
    assert base.grid == default.grid == stated.grid and base.bands == stated.bands


@pytest.mark.parametrize("size", [(8, 10), (7, 9), (10, 8), (5, 7)])
@pytest.mark.parametrize("phase", [(0, 0), (1, 1), (0, 1), (1, 0)])
def test_each_sample_pixel_is_centred_on_its_input_pixel(size, phase):
    """Pixel (i, j) of the sample grid is centred on input pixel (2 i + phase_y,
    2 j + phase_x), on a non-square grid with the y spacing negative; the canonical grid
    keeps the input's corner and has ceil((n - phase) / 2) pixels on each axis."""
    width, height = size
    grid = Grid("EPSG:3857", (20, 0, 1000, 0, -30, 2000), width, height)
    sample, canonical = L.sample_grids(grid, 2, phase)
    py, px = phase
    a, _, c, _, e, f = grid.transform
    sa, _, sc, _, se, sf = sample.transform
    assert (sa, se) == (2 * a, 2 * e)
    assert (canonical.width, canonical.height) == (-(-(width - px) // 2), -(-(height - py) // 2))
    assert (sample.width, sample.height) == (canonical.width, canonical.height)
    assert canonical.transform == (2 * a, 0, c, 0, 2 * e, f)
    for i in range(sample.height):
        for j in range(sample.width):
            centre = (sc + (j + 0.5) * sa, sf + (i + 0.5) * se)
            assert centre == (c + (2 * j + px + 0.5) * a, f + (2 * i + py + 0.5) * e)


@pytest.mark.parametrize("phase", [1, (1, 0), (0, 1), (1, 1), [1, 1]])
def test_a_phase_changes_only_the_sample_grid(offline_ee, phase):
    grid = _grid(10, 8)
    image = ee.Image.constant([0.5]).rename(["c0"])
    shifted = L.stride_sample(image, ("c0",), grid, 2, phase=phase)
    assert shifted.image.serialize() != L.stride_sample(image, ("c0",), grid, 2).image.serialize()
    py, px = L.phase_pair(phase, 2)
    assert shifted.grid == grid.resized(2, (10 - px + 1) // 2, (8 - py + 1) // 2)


@pytest.mark.parametrize("bad", [True, 1.0, 2, -1, (0, 2), (0,), (0, 0, 0), "1", None, (True, 0)])
def test_invalid_phases_are_refused(bad):
    with pytest.raises(UnsupportedOperation, match="phase must be an integer or a pair"):
        L.phase_pair(bad, 2)


def test_stride_one_accepts_phase_zero_only():
    assert L.phase_pair(0, 1) == (0, 0)
    with pytest.raises(UnsupportedOperation, match="0 <= phase < stride 1"):
        L.phase_pair(1, 1)


def test_a_phase_with_no_output_pixels_is_refused():
    with pytest.raises(UnsupportedOperation, match="no output pixels"):
        L.sample_grids(_grid(1, 4), 2, (0, 1))


def test_the_canonical_phase():
    assert [L.canonical_phase(p, 2) for p in (0, (0, 0), [0, 0], 1, (1, 1), (0, 1), [1, 0])] == [
        0, 0, 0, 1, 1, (0, 1), (1, 0)]


# ---------------------------------------------------------------------------
# The blocks: the setting, its default and its refusals
# ---------------------------------------------------------------------------
def _dense(stride=2, kernel=3):
    module = nn.Conv2d(2, 3, kernel, stride=stride, padding=kernel // 2).eval()
    with torch.no_grad():
        module.weight.copy_(torch.arange(module.weight.numel(), dtype=torch.float32)
                            .reshape(module.weight.shape) / 64)
        module.bias.copy_(torch.tensor([0.5, -0.25, 0.125]))
    return module


def _depthwise(stride=2, kernel=3):
    module = nn.Conv2d(4, 4, kernel, stride=stride, padding=kernel // 2, groups=4).eval()
    with torch.no_grad():
        module.weight.copy_(torch.arange(module.weight.numel(), dtype=torch.float32)
                            .reshape(module.weight.shape) / 64)
        module.bias.copy_(torch.tensor([0.5, -0.25, 0.125, 1.0]))
    return module


def _raster(channels, grid):
    bands = tuple(f"c{i}" for i in range(channels))
    return Raster(ee.Image.constant(list(range(1, channels + 1))).rename(list(bands)), bands, grid)


@pytest.mark.parametrize("block,module,channels", [
    (C.conv2d, _dense, 2), (C.grouped_conv2d, _depthwise, 4)])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("clip", [False, True])
def test_the_blocks_at_phase_zero_are_todays_graph(offline_ee, block, module, channels, stride, clip):
    raster = _raster(channels, _grid(10, 8))
    conv = module(stride)
    today = block(raster, conv, clip_footprints=clip)
    for phase in (0, (0, 0), [0, 0]):
        stated = block(raster, conv, clip_footprints=clip, phase=phase)
        assert stated.image.serialize() == today.image.serialize() and stated.grid == today.grid


@pytest.mark.parametrize("block,module,channels", [
    (C.conv2d, _dense, 2), (C.grouped_conv2d, _depthwise, 4)])
@pytest.mark.parametrize("phase", [1, (0, 1), (1, 0)])
def test_the_blocks_differ_from_today_only_in_the_sampler(offline_ee, block, module, channels, phase):
    """A phase reuses the block's convolution expression: only the sample grid changes."""
    raster = _raster(channels, _grid(10, 8))
    calls = []
    original = C._stride_sample

    def record(image, bands, grid, stride, clip_footprints=False, *, phase=0):
        calls.append((image.serialize(), bands, grid, stride, clip_footprints, phase))
        return original(image, bands, grid, stride, clip_footprints, phase=phase)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(C, "_stride_sample", record)
        today = block(raster, module(2))
        shifted = block(raster, module(2), phase=phase)
    (image0, *rest0, phase0), (image1, *rest1, phase1) = calls
    assert image0 == image1 and rest0 == rest1
    assert (phase0, phase1) == ((0, 0), L.phase_pair(phase, 2))
    py, px = L.phase_pair(phase, 2)
    assert shifted.grid == _grid(10, 8).resized(2, (10 - px + 1) // 2, (8 - py + 1) // 2)
    assert shifted.image.serialize() == L.stride_sample(
        ee.deserializer.fromJSON(image1), rest1[0], rest1[1], 2, phase=phase).image.serialize()
    assert today.image.serialize() != shifted.image.serialize()


@pytest.mark.parametrize("block,module,channels", [
    (C.conv2d, _dense, 2), (C.grouped_conv2d, _depthwise, 4)])
def test_a_stride_one_phase_and_bad_values_are_refused(offline_ee, block, module, channels):
    raster = _raster(channels, _grid(10, 8))
    for bad in (1, (0, 1), True, 0.0):
        with pytest.raises(UnsupportedOperation, match="phase must be"):
            block(raster, module(1), phase=bad)
    for bad in (2, (0, 2), -1):
        with pytest.raises(UnsupportedOperation, match="phase must be"):
            block(raster, module(2), phase=bad)


# ---------------------------------------------------------------------------
# The records: phase at its default only, in one canonical form
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["conv2d", "grouped_conv2d"])
def test_every_nonzero_phase_is_outside_the_records_in_either_spelling(name):
    from tensor_ee.core.metadata import check_settings, describe_validated

    entry = C.METADATA[name]
    base = {} if name == "conv2d" else {"clip_footprints": False}
    assert describe_validated(entry.validated) == ("{phase 0}" if name == "conv2d"
                                                   else "{clip_footprints False; phase 0}")
    for spelling in (0, (0, 0), [0, 0]):
        assert check_settings(entry, {**base, "phase": L.canonical_phase(spelling, 2)}).inside
    for spelling in (1, (1, 1), [1, 1], (0, 1), (1, 0), [1, 0]):
        result = check_settings(entry, {**base, "phase": L.canonical_phase(spelling, 2)})
        assert not result.inside and result.outside == ("phase",), (spelling, result)
    # The canonical form matters: a pair spelling of phase 0 is not the recorded value.
    assert check_settings(entry, {**base, "phase": (0, 0)}).outside == ("phase",)
    # Calls that predate the setting leave it unstated, never assumed.
    assert check_settings(entry, base).unstated == ("phase",)
    assert entry.radius == (3 if name == "conv2d" else 2)


# ---------------------------------------------------------------------------
# Recognition: zero padding in its spellings, directly ahead of a convolution in either spelling
# ---------------------------------------------------------------------------
def _values(shape, salt=0):
    """Integer-generated dyadic values (no RNG)."""
    count = 1
    for n in shape:
        count *= n
    return ((torch.arange(count, dtype=torch.int64) * 7 + salt) % 23 - 11).to(torch.float32).reshape(shape) / 32


def _batch_norm(channels):
    bn = nn.BatchNorm2d(channels)
    with torch.no_grad():
        bn.running_mean.copy_(_values((channels,), 3))
        bn.running_var.copy_(_values((channels,), 5).abs() + 0.5)
        bn.weight.copy_(_values((channels,), 7) + 1)
        bn.bias.copy_(_values((channels,), 9))
    return bn.eval()


class PaddedConv(nn.Module):
    """A padding node in one spelling, then a convolution in one spelling (the stand-in for
    a convolution with static "same" padding written as a padding module and ``F.conv2d``
    with the module's own weight and bias), optionally followed by a BatchNorm."""

    def __init__(self, cin, cout, kernel, *, stride=1, groups=1, bias=True, pad=(1, 1, 1, 1),
                 spelling="ZeroPad2d", call="F.conv2d", own=0, bn=False):
        super().__init__()
        self.spelling, self.call, self.stride, self.groups, self.own = spelling, call, stride, groups, own
        self.pad_values = tuple(pad) if pad is not None else None
        if spelling == "ZeroPad2d":
            self.pad = nn.ZeroPad2d(pad)
        elif spelling == "ConstantPad2d":
            self.pad = nn.ConstantPad2d(pad, 0.0)
        elif spelling == "Identity":
            self.pad = nn.Identity()
        weight = _values((cout, cin // groups, kernel, kernel), 1)
        bias_value = _values((cout,), 2) if bias else None
        if call == "nn.Conv2d":
            self.conv = nn.Conv2d(cin, cout, kernel, stride, own, groups=groups, bias=bias)
            with torch.no_grad():
                self.conv.weight.copy_(weight)
                if bias:
                    self.conv.bias.copy_(bias_value)
        else:
            self.weight = nn.Parameter(weight)
            self.bias = nn.Parameter(bias_value) if bias else None
        self.bn = _batch_norm(cout) if bn else None

    def forward(self, x):
        if self.spelling == "F.pad":
            x = F.pad(x, self.pad_values)
        elif self.spelling == "F.pad-value":
            x = F.pad(x, self.pad_values, "constant", 0.0)
        elif self.spelling in ("ZeroPad2d", "ConstantPad2d", "Identity"):
            x = self.pad(x)
        if self.call == "nn.Conv2d":
            x = self.conv(x)
        elif self.call == "F.conv2d":
            x = F.conv2d(x, self.weight, self.bias, self.stride, self.own, 1, self.groups)
        else:
            x = torch.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.own, groups=self.groups)
        return x if self.bn is None else self.bn(x)


def _reference(cin, cout, kernel, *, stride=1, groups=1, bias=True, bn=False):
    """The same convolution as an ``nn.Conv2d`` with padding kernel // 2."""
    conv = nn.Conv2d(cin, cout, kernel, stride, kernel // 2, groups=groups, bias=bias)
    with torch.no_grad():
        conv.weight.copy_(_values((cout, cin // groups, kernel, kernel), 1))
        if bias:
            conv.bias.copy_(_values((cout,), 2))
    return nn.Sequential(conv, *([_batch_norm(cout)] if bn else [])).eval()


def _compile(model, channels, width=10, height=8, **kwargs):
    bands = tuple(f"b{i}" for i in range(channels))
    image = ee.Image.constant(list(range(1, channels + 1))).rename(list(bands))
    return compile_model(model.eval(), image, bands, _grid(width, height), **kwargs)


SYMMETRIC_SPELLINGS = (  # (padding spelling, convolution spelling, own padding): total kernel // 2
    ("ZeroPad2d", "F.conv2d", 0), ("ConstantPad2d", "F.conv2d", 0), ("F.pad", "torch.conv2d", 0),
    ("F.pad-value", "F.conv2d", 0), ("ZeroPad2d", "nn.Conv2d", 0), ("Identity", "F.conv2d", None),
    ("none", "F.conv2d", None), ("none", "torch.conv2d", None))


@pytest.mark.parametrize("groups", [1, 4], ids=["dense", "depthwise"])
@pytest.mark.parametrize("kernel", [1, 3, 5])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("bn", [False, True], ids=["conv", "conv-bn"])
@pytest.mark.parametrize("spelling,call,own", SYMMETRIC_SPELLINGS, ids=lambda v: str(v))
def test_symmetric_padding_compiles_to_the_nn_conv2d_graph(offline_ee, groups, kernel, stride, bn, spelling, call, own):
    """Zero padding of kernel // 2 ahead of a convolution, in any spelling, with the
    convolution as F.conv2d (get_attr weights) or as an nn.Conv2d behind the pad, compiles
    byte for byte to today's graph of nn.Conv2d(padding=kernel // 2), with and without a
    folded BatchNorm."""
    p = kernel // 2
    model = PaddedConv(4, 4, kernel, stride=stride, groups=groups, bias=not bn, pad=(p, p, p, p), spelling=spelling,
                       call=call, own=p if own is None else own, bn=bn)
    reference = _reference(4, 4, kernel, stride=stride, groups=groups, bias=not bn, bn=bn)
    for unfused in (True, False):
        a = _compile(model, 4, include_unfused_nodes=unfused)
        b = _compile(reference, 4, include_unfused_nodes=unfused)
        assert a.image.serialize() == b.image.serialize() and a.grid == b.grid
        assert len(a.folded_batchnorms) == len(b.folded_batchnorms) == int(bn)


def _block_graph(module, phase, channels=4, width=10, height=8):
    """The compiled graph of ``module``'s block at ``phase``, as compile_model emits it: the
    input materialized once, the block, and its float32 materialization."""
    from tensor_ee.compiler import _finish

    bands = tuple(f"b{i}" for i in range(channels))
    image = ee.Image.constant(list(range(1, channels + 1))).rename(list(bands))
    grid = _grid(width, height)
    raster = _finish(image.select(list(bands)).unmask(0, sameFootprint=False), C._names(channels), grid)
    raster = Raster(raster.image.toFloat(), raster.bands, raster.grid)
    block = C.conv2d if module.groups == 1 else C.grouped_conv2d
    out = block(raster, module, phase=phase)
    return out.image.toFloat().serialize(), out.grid


PHASED_SPELLINGS = (  # (phase, kernel, [(padding spelling, pad, convolution spelling, own padding), ...])
    (1, 3, [("ZeroPad2d", (0, 1, 0, 1), "F.conv2d", 0), ("F.pad", (0, 1, 0, 1), "nn.Conv2d", 0),
            ("ConstantPad2d", (0, 1, 0, 1), "torch.conv2d", 0), ("F.pad", (0, 2, 0, 2), "F.conv2d", 0)]),
    (1, 5, [("ZeroPad2d", (1, 2, 1, 2), "F.conv2d", 0), ("F.pad", (0, 1, 0, 1), "nn.Conv2d", 1)]),
    ((1, 0), 3, [("F.pad", (1, 1, 0, 1), "F.conv2d", 0), ("ZeroPad2d", (0, 0, 0, 1), "F.conv2d", (0, 1)),
                 ("ZeroPad2d", (1, 1, 0, 1), "nn.Conv2d", 0)]),
    ((0, 1), 3, [("F.pad", (0, 1, 1, 1), "torch.conv2d", 0), ("ZeroPad2d", (0, 1, 0, 0), "nn.Conv2d", (1, 0))]),
)


@pytest.mark.parametrize("groups", [1, 4], ids=["dense", "depthwise"])
@pytest.mark.parametrize("phase,kernel,spellings", PHASED_SPELLINGS, ids=["p1-k3", "p1-k5", "p10-k3", "p01-k3"])
@pytest.mark.parametrize("size", [(10, 8), (9, 7)], ids=["even", "odd"])
def test_each_padded_spelling_is_the_block_at_its_phase(offline_ee, groups, phase, kernel, spellings, size):
    """Spellings of one padded convolution compile to one graph: the block at the phase the
    padding gives (left and top pads kernel // 2 - phase), on even and odd inputs. A right or
    bottom pad that gives PyTorch another output count at the size is refused instead."""
    width, height = size
    expected, grid = _block_graph(_reference(4, 4, kernel, stride=2, groups=groups)[0], phase,
                                  width=width, height=height)
    compiled = 0
    for spelling, pad, call, own in spellings:
        model = PaddedConv(4, 4, kernel, stride=2, groups=groups, pad=pad, spelling=spelling, call=call, own=own)
        own_y, own_x = own if isinstance(own, tuple) else (own, own)
        counts = ((height + pad[2] + pad[3] + 2 * own_y - kernel) // 2 + 1,
                  (width + pad[0] + pad[1] + 2 * own_x - kernel) // 2 + 1)
        if counts != (grid.height, grid.width):
            with pytest.raises(UnsupportedOperation, match="gives PyTorch"):
                _compile(model, 4, width, height)
            continue
        out = _compile(model, 4, width, height)
        assert out.image.serialize() == expected and out.grid == grid, (spelling, pad, call, own)
        compiled += 1
    assert compiled >= 2


def test_a_bare_strided_convolution_without_padding_follows_the_same_rule(offline_ee):
    """``nn.Conv2d(3, stride=2, padding=0)`` reads input pixels 2i .. 2i + 2: phase 1. On an
    odd input PyTorch's count is ceil((n - 1) / 2), so it compiles (the valid stride-2
    convolution of an odd map); on an even input it gives one output fewer and is refused."""
    model = PaddedConv(4, 4, 3, stride=2, pad=None, spelling="none", call="nn.Conv2d", own=0)
    expected, grid = _block_graph(_reference(4, 4, 3, stride=2)[0], 1, width=9, height=7)
    out = _compile(model, 4, 9, 7)
    assert out.image.serialize() == expected and out.grid == grid == _grid(9, 7).resized(2, 4, 3)
    with pytest.raises(UnsupportedOperation, match=r"gives PyTorch 3 x 4 outputs.*4 x 5"):
        _compile(model, 4, 10, 8)
    # A bare unpadded convolution at stride 1 shrinks the map: refused with today's words.
    with pytest.raises(UnsupportedOperation, match=r"same padding \(kernel_size // 2\)"):
        _compile(PaddedConv(4, 4, 3, pad=None, spelling="none", call="nn.Conv2d", own=0), 4)


def test_folding_reaches_through_a_recognised_convolution_with_its_phase(offline_ee):
    model = PaddedConv(4, 6, 3, stride=2, bias=False, pad=(0, 1, 0, 1), bn=True)
    reference = _reference(4, 6, 3, stride=2, bias=False, bn=True)
    fused = torch.nn.utils.fuse_conv_bn_eval(reference[0], reference[1])
    expected, grid = _block_graph(fused, 1)
    for unfused in (True, False):
        out = _compile(model, 4, include_unfused_nodes=unfused)
        assert out.image.serialize() == expected and out.grid == grid
        assert out.folded_batchnorms == ("bn",)
        assert ("conv2d" in out.nodes) is unfused
    traced = symbolic_trace(model)
    call = semantic_call(next(n for n in traced.graph.nodes if n.name == "bn"), traced)
    assert call.kind == "batchnorm" and call.options["phase"] == 1 and call.options["padding"] == (0, 1, 0, 1)
    assert call.options["source"].name == "x"
    predecessor, source, _ = batchnorm_predecessor(next(n for n in traced.graph.nodes if n.name == "bn"), traced)
    assert (predecessor.name, source.name) == ("conv2d", "x")


def test_an_identity_is_absorbed_only_ahead_of_a_functional_convolution(offline_ee):
    functional = _compile(PaddedConv(4, 4, 3, pad=None, spelling="Identity", call="F.conv2d", own=1), 4)
    module = _compile(PaddedConv(4, 4, 3, pad=None, spelling="Identity", call="nn.Conv2d", own=1), 4)
    assert "pad" not in functional.nodes and "pad" in module.nodes    # the Identity keeps its node
    assert functional.image.serialize() != module.image.serialize()  # today's extra materialization


def test_recognition_draws_no_random_numbers():
    model = PaddedConv(4, 4, 3, stride=2, pad=(0, 1, 0, 1)).eval()
    traced = symbolic_trace(model)
    state = torch.random.get_rng_state()
    node = next(n for n in traced.graph.nodes if n.op == "call_function" and n.target is torch.conv2d)
    recognised = convolution(node, traced)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert recognised.module.weight is model.weight and recognised.module.bias is model.bias
    assert recognised.module.padding == (1, 1) and recognised.phase == 1 and recognised.path == "weight"
    buffered = PaddedConv(4, 4, 3, pad=(1, 1, 1, 1)).eval()
    weight = buffered.weight.detach().clone()
    del buffered.weight
    buffered.register_buffer("weight", weight)
    traced = symbolic_trace(buffered)
    node = next(n for n in traced.graph.nodes if n.op == "call_function" and n.target is torch.conv2d)
    assert torch.equal(convolution(node, traced).module.weight, weight)


# ---------------------------------------------------------------------------
# Refusals, each with its message
# ---------------------------------------------------------------------------
class ShapePad(nn.Module):
    """Padding computed from the input's shape, as dynamic "same" padding does."""

    def __init__(self, form):
        super().__init__()
        self.form = form
        self.weight = nn.Parameter(_values((4, 4, 3, 3), 1))

    def forward(self, x):
        if self.form == "arithmetic":
            pad = (0, x.shape[-1] % 2 + 1, 0, x.shape[-2] % 2 + 1)
        else:                                              # a shape element itself (batch 1)
            pad = (0, x.shape[0], 0, x.shape[0])
        return F.conv2d(F.pad(x, pad), self.weight, None, 2)


class Reused(nn.Module):
    def __init__(self, use):
        super().__init__()
        self.use = use
        self.pad = nn.ZeroPad2d(1)
        self.weight = nn.Parameter(_values((4, 4, 3, 3)))
        self.pool = nn.MaxPool2d(3, 1)

    def forward(self, x):
        padded = self.pad(x)
        if self.use == "pool":
            return self.pool(padded)
        return F.conv2d(padded, self.weight) + padded[:, :, 1:-1, 1:-1]


class Functional(nn.Module):
    def __init__(self, form):
        super().__init__()
        self.form = form
        self.weight = nn.Parameter(_values((4, 4, 3, 3)))

    def forward(self, x):
        if self.form == "computed":
            return F.conv2d(x, self.weight * 2, None, 1, 1)
        if self.form == "shared":
            return F.conv2d(x, self.weight, None, 1, 1) + F.conv2d(x, self.weight, None, 1, 1)
        if self.form == "dilation":
            return F.conv2d(x, self.weight, None, 1, 2, 2)
        return F.conv2d(x, self.weight, None, 3, 1)


class Explicit(nn.Module):
    def __init__(self, form):
        super().__init__()
        self.form = form
        self.weight = nn.Parameter(_values((4, 4, 3, 3)))
        pads = {"nonzero value": nn.ConstantPad2d(1, 0.5), "reflection": nn.ReflectionPad2d(1),
                "replication": nn.ReplicationPad2d(1)}
        self.pad = pads.get(form, nn.Identity())

    def forward(self, x):
        if self.form == "F.pad nonzero value":
            x = F.pad(x, (1, 1, 1, 1), value=1.0)
        elif self.form == "F.pad reflect":
            x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        elif self.form == "F.pad replicate":
            x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        else:
            x = self.pad(x)
        return F.conv2d(x, self.weight)


REFUSALS = [
    ("pad from shape arithmetic", lambda: ShapePad("arithmetic"), "padding computed from a tensor's shape"),
    ("negative pad", lambda: PaddedConv(4, 4, 3, pad=(-1, 1, 1, 1)), r"negative padding \(cropping\)"),
    ("F.pad of two", lambda: PaddedConv(4, 4, 3, pad=(1, 1), spelling="F.pad"), "four integers"),
    ("F.pad of six", lambda: PaddedConv(4, 4, 3, pad=(1, 1, 1, 1, 0, 0), spelling="F.pad"), "four integers"),
    ("nonzero value", lambda: Explicit("nonzero value"), r"ConstantPad2d value must be 0"),
    ("F.pad nonzero value", lambda: Explicit("F.pad nonzero value"), r"F.pad value must be 0"),
    ("reflection", lambda: Explicit("reflection"), "ReflectionPad2d is unsupported"),
    ("replication", lambda: Explicit("replication"), "ReplicationPad2d is unsupported"),
    ("F.pad reflect", lambda: Explicit("F.pad reflect"), "F.pad mode 'reflect' is unsupported"),
    ("F.pad replicate", lambda: Explicit("F.pad replicate"), "F.pad mode 'replicate' is unsupported"),
    ("pad before a pool", lambda: Reused("pool"), "ZeroPad2d has another use"),
    ("pad read twice", lambda: Reused("twice"), "ZeroPad2d has another use"),
    ("phase beyond the stride", lambda: PaddedConv(4, 4, 3, pad=(0, 0, 0, 0)),
     r"gives phase \(1, 1\) at kernel \(3, 3\) and stride 1"),
    ("padding that enlarges", lambda: PaddedConv(4, 4, 3, pad=(2, 2, 2, 2)), r"gives phase \(-1, -1\)"),
    ("extra output", lambda: PaddedConv(4, 4, 3, stride=2, pad=(0, 3, 0, 3)), r"gives PyTorch 5 x 6 outputs"),
    ("weight computed", lambda: Functional("computed"), "get_attr parameters or buffers used only as this call's operands"),
    ("weight shared", lambda: Functional("shared"), "get_attr parameters or buffers used only as this call's operands"),
    ("dilation", lambda: Functional("dilation"), "dilation=1"),
    ("stride 3", lambda: Functional("stride3"), "stride=1 or 2"),
]


ARITHMETIC_FIRST = {"pad from shape arithmetic": "Unsupported function <built-in function mod>",
                    "weight computed": "Unsupported FX operation get_attr"}
"""compile_model stops at the first refused node: here the arithmetic ahead of the padding or
the convolution, itself refused. check_compatibility reports every node."""


@pytest.mark.parametrize("name,model,message", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_each_refusal_fires_with_its_message(offline_ee, name, model, message):
    model = model().eval()
    with pytest.raises(UnsupportedOperation, match=re.escape(ARITHMETIC_FIRST.get(name, "")) or message):
        _compile(model, 4)
    report = check_compatibility(model, (1, 4, 8, 10)).to_dict()
    assert report["status"] == "unsupported"
    assert any(re.search(message, blocker["message"]) for blocker in report["blockers"]), report["blockers"]


def test_a_pad_that_is_a_shape_element_is_a_literal_at_the_proven_size(offline_ee):
    """``specialize_plumbing`` resolves a shape element into a literal at the compiled input
    size, as it does for an interpolation size; the raw trace (receptive field) refuses the
    shape read itself."""
    model = ShapePad("element").eval()
    expected, grid = _block_graph(_reference(4, 4, 3, stride=2, bias=False)[0], 1)
    out = _compile(model, 4)
    assert out.image.serialize() == expected and out.grid == grid
    assert check_compatibility(model, (1, 4, 8, 10)).status == "supported"
    for form in ("element", "arithmetic"):
        with pytest.raises(UnsupportedReceptiveField, match="getattr"):
            analyze_receptive_field(ShapePad(form).eval())


# ---------------------------------------------------------------------------
# Receptive field, weight collection and matching
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pad,kernel,bounds", [
    ((0, 1, 0, 1), 3, ((0, 2), (2, 4))), ((1, 1, 1, 1), 3, ((-1, 1), (1, 3))),
    ((1, 2, 1, 2), 5, ((-1, 3), (1, 5))), ((2, 2, 2, 2), 5, ((-2, 2), (0, 4)))])
def test_the_receptive_field_uses_the_left_and_top_pads(pad, kernel, bounds):
    report = analyze_receptive_field(PaddedConv(4, 4, kernel, stride=2, pad=pad).eval())
    assert report.x.phase_bounds == (bounds[0],) and report.x.bounds(1) == bounds[1]
    assert report.y.phase_bounds == (bounds[0],)
    mixed = analyze_receptive_field(PaddedConv(4, 4, 3, stride=2, pad=(1, 1, 0, 1)).eval())
    assert mixed.y.bounds(0) == (0, 2) and mixed.x.bounds(0) == (-1, 1)


class Stem(nn.Module):
    """A functional stem with a folded BatchNorm, then a biased functional 1 x 1 convolution
    behind an Identity: weight paths ``stem`` and ``head``."""

    def __init__(self):
        super().__init__()
        self.stem = PaddedConv(4, 6, 3, stride=2, bias=False, pad=(0, 1, 0, 1), bn=True)
        self.act = nn.SiLU()
        self.head = PaddedConv(6, 2, 1, pad=None, spelling="Identity")

    def forward(self, x):
        return self.head(self.act(self.stem(x)))


def test_weights_are_collected_and_matched_by_their_state_dict_paths(offline_ee):
    model = Stem().eval()
    parameters = effective_convolutions(model)
    assert [(p.key, p.source) for p in parameters] == [("stem_bn", "stem+stem.bn"), ("conv2d_1", "head")]
    fused = torch.nn.utils.fuse_conv_bn_eval(_reference(4, 6, 3, stride=2, bias=False)[0], model.stem.bn)
    assert torch.equal(torch.from_numpy(parameters[0].weights), fused.weight.detach().reshape(6, -1))
    assert {"stem.weight", "head.weight", "head.bias"} <= set(model.state_dict())
    layers = {}
    for index, parameter in enumerate(parameters):
        saved = replace(parameter, key=f"saved_{index}")
        layers[saved.key] = {"shape": list(saved.shape), "dtype": "float32", "bias": saved.bias is not None,
                             "sha256": saved.sha256, "assets": ["projects/example/assets/weights"]}
    manifest = {"format": "native_ee_featurecollection_weights_v2", "status": "ready", "layers": layers}
    provider = MatchedWeightProvider(model, manifest)
    assert provider.mapping == {"stem_bn": "saved_0", "conv2d_1": "saved_1"}
    out = _compile(model, 4, weight_provider=provider, include_unfused_nodes=False)
    assert provider.used_asset_keys == {"saved_0", "saved_1"} and not provider.fallback_keys
    assert out.grid == _grid(10, 8).resized(2, 5, 4) and out.folded_batchnorms == ("stem_bn",)


# ---------------------------------------------------------------------------
# SMP's EfficientNet encoders (trace and the adapter's one PyTorch probe; no evaluation)
# ---------------------------------------------------------------------------
STRIDED = {   # encoder -> {weight path: (kernel, groups > 1, total padding (l, r, t, b), phase)} at 256 px
    "efficientnet-b0": {
        "encoder._conv_stem": (3, False, (0, 1, 0, 1), 1),
        "encoder._blocks.1._depthwise_conv": (3, True, (0, 1, 0, 1), 1),
        "encoder._blocks.3._depthwise_conv": (5, True, (1, 2, 1, 2), 1),
        "encoder._blocks.5._depthwise_conv": (3, True, (0, 1, 0, 1), 1),
        "encoder._blocks.11._depthwise_conv": (5, True, (1, 2, 1, 2), 1)},
    "efficientnet-b3": {
        "encoder._conv_stem": (3, False, (0, 1, 0, 1), 1),
        "encoder._blocks.2._depthwise_conv": (3, True, (0, 1, 0, 1), 1),
        "encoder._blocks.5._depthwise_conv": (5, True, (2, 2, 2, 2), 0),
        "encoder._blocks.8._depthwise_conv": (3, True, (0, 1, 0, 1), 1),
        "encoder._blocks.18._depthwise_conv": (5, True, (2, 2, 2, 2), 0)},
}


@pytest.mark.parametrize("encoder,channels,classes", [("efficientnet-b0", 3, 2), ("efficientnet-b3", 8, 3)])
def test_smp_efficientnet_unets_are_supported_through_the_adapter(encoder, channels, classes):
    """Each stride-2 convolution carries the phase its padding gives. ``efficientnet-b0``
    pads its 5 x 5 stride-2 layers (1, 2, 1, 2), phase 1; ``efficientnet-b3`` pads its
    (2, 2, 2, 2), phase 0 (SMP fixes the padding for the encoder's configured image size)."""
    smp = pytest.importorskip("segmentation_models_pytorch")
    from tensor_ee.smp import adapt_smp_model

    shape = (1, channels, 256, 256)
    model = smp.Unet(encoder, encoder_weights=None, in_channels=channels, classes=classes).eval()
    adapted = adapt_smp_model(model, input_shape=shape)
    report = check_compatibility(adapted, shape).to_dict()
    assert report["status"] == "supported", report["blockers"] + report["unknowns"]
    assert report["output_shape"] == [1, classes, 256, 256]
    strided = {}
    for node in adapted.graph.nodes:
        if node.op in ("call_module", "call_function"):
            recognised = convolution(node, adapted)
            if recognised is not None and tuple(recognised.module.stride) == (2, 2):
                strided[recognised.path] = (recognised.module.kernel_size[0], recognised.module.groups > 1,
                                            recognised.padding, recognised.phase)
    assert strided == STRIDED[encoder]
