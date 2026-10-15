"""P02: pooling configurations and nearest factors 3 and 4, offline.

No network and no Earth Engine call: graphs are built against the offline signatures of
``ee.apitestcase.ApiTestCase``. References are ``torch.nn`` and ``torch.nn.functional``.

- The block: every supported configuration, every rejection, the graphs it emits, and
  today's configurations serialized exactly as before P02.
- Edge semantics: a NumPy model of the graph's masked reductions against PyTorch, at every
  border pixel, and the wrong alternatives (a zero pad, the other divisor) against it.
- The converter: ``check_compatibility`` and ``compile_model`` shapes, the receptive field,
  the metadata radius, and tiling with the computed halo against the untiled forward.
- The ``p02`` fixtures: selection, plan within the budget, determinism, injected errors
  through a mocked ``computePixels``, and the live plan document.
"""
from __future__ import annotations

import contextlib
import hashlib
from importlib.metadata import version as package_version
import importlib.util
import json
from pathlib import Path
import re
import socket
import sys

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

import tensor_ee
from tensor_ee.blocks.raster import pooling as PL
from tensor_ee.compatibility import check_compatibility
from tensor_ee.compiler import UnsupportedOperation, compile_model
from tensor_ee.core.grid import Grid
from tensor_ee.layouts import raster as LR
from tensor_ee.models import initialize_fixture
from tensor_ee.receptive_field import UnsupportedReceptiveField, analyze_receptive_field
from tensor_ee.tiling import PixelWindow, compile_tiled_model, plan_tiles

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("p02_live_fixtures", ROOT / "tools" / "live_fixtures.py")
LF = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = LF                     # dataclasses resolve annotations through sys.modules
_SPEC.loader.exec_module(LF)

from tools.fixtures import core, raster as RF, registry, select, selections  # noqa: E402

# (module, PoolConfig) for every supported configuration, in the brief's order.
SUPPORTED = [
    *[(cls(k), PL.PoolConfig(reducer, k, k, 0, include))
      for k in (2, 3, 4)
      for cls, reducer, include in ((nn.AvgPool2d, "mean", True), (nn.MaxPool2d, "max", None))],
    (nn.AvgPool2d(3, count_include_pad=False), PL.PoolConfig("mean", 3, 3, 0, False)),
    (nn.MaxPool2d((4, 4), (4, 4), (0, 0)), PL.PoolConfig("max", 4, 4, 0, None)),
    (nn.MaxPool2d(3, 1, 1), PL.PoolConfig("max", 3, 1, 1, None)),
    (nn.AvgPool2d(3, 1, 1, count_include_pad=True), PL.PoolConfig("mean", 3, 1, 1, True)),
    (nn.AvgPool2d(3, 1, 1, count_include_pad=False), PL.PoolConfig("mean", 3, 1, 1, False)),
    (nn.MaxPool2d(3, 2, 1), PL.PoolConfig("max", 3, 2, 1, None)),
]
NEW = [module for module, config in SUPPORTED
       if (config.kernel, config.stride) != (2, 2) and (config.stride, config.padding) != (2, 1)]

UNSUPPORTED = [   # (module, message fragment)
    (nn.AvgPool2d(3, 2, 1), "Pooling supports"),             # average stride 2 padded: not in the table
    (nn.AvgPool2d(5), "Pooling supports"),
    (nn.MaxPool2d(5, 1, 2), "Pooling supports"),             # SPPF's 5 x 5 "same" max pool
    (nn.MaxPool2d(3, 1), "Pooling supports"),                # stride 1 without padding
    (nn.MaxPool2d(2, 1), "Pooling supports"),
    (nn.AvgPool2d(4, 2), "Pooling supports"),
    (nn.MaxPool2d(3, 3, 1), "Pooling supports"),
    (nn.MaxPool2d((3, 2)), "Pooling supports"),
    (nn.AvgPool2d(3, 1, 1, count_include_pad=False, divisor_override=9), "divisor_override"),
    (nn.AvgPool2d(3, divisor_override=9), "divisor_override"),
    (nn.MaxPool2d(3, 1, 1, dilation=2), "dilation=1"),
    (nn.MaxPool2d(3, 1, 1, return_indices=True), "return_indices"),
    (nn.MaxPool2d(3, 1, 1, ceil_mode=True), "ceil_mode"),
    (nn.AvgPool2d(3, ceil_mode=True), "ceil_mode"),
    (nn.MaxPool2d(4, ceil_mode=True), "ceil_mode"),
]

GRID = Grid("EPSG:3857", (10, 0, 1000, 0, -10, 2000), 24, 36)   # 36 rows, 24 columns


@pytest.fixture
def offline_ee():
    case = ApiTestCase()
    case.setUp()
    try:
        yield ee
    finally:
        case.tearDown()


@pytest.fixture
def no_network(monkeypatch):
    attempts = []

    def refuse(*args, **kwargs):
        attempts.append(args)
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    return attempts


def _raster(grid: Grid = GRID, channels: int = 2) -> tensor_ee.Raster:
    bands = [f"b{i}" for i in range(channels)]
    return tensor_ee.from_image(ee.Image.constant(list(range(1, channels + 1))).rename(bands), bands, grid)


def _tree(image) -> dict:
    """A serialized graph as one expression tree, with every value reference expanded."""
    graph = json.loads(image.serialize())

    def expand(node):
        if isinstance(node, dict):
            if "valueReference" in node:
                return expand(graph["values"][node["valueReference"]])
            return {key: expand(value) for key, value in node.items()}
        if isinstance(node, list):
            return [expand(value) for value in node]
        return node
    return expand(graph["values"][graph["result"]])


def _calls(image) -> list[tuple[str, dict]]:
    """Every (functionName, arguments) in a graph, outermost first."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            if "functionName" in node:
                found.append((node["functionName"], node["arguments"]))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(_tree(image))
    return found


def _functions(image) -> list[str]:
    return [name for name, _ in _calls(image)]


def _invocation(image, name: str) -> dict:
    """The arguments of the first ``name`` call in a graph."""
    return next(arguments for function, arguments in _calls(image) if function == name)


def _constant(argument: dict) -> object:
    """The constant an (expanded) argument holds, directly or through ``Image.constant``."""
    if "functionInvocationValue" in argument:
        invocation = argument["functionInvocationValue"]
        assert invocation["functionName"] == "Image.constant"
        return _constant(invocation["arguments"]["value"])
    return argument["constantValue"]


# ---------------------------------------------------------------------------
# The block: supported configurations and rejections
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("module,expected", SUPPORTED, ids=repr)
def test_pool_config_normalizes_every_supported_configuration(module, expected):
    assert PL.pool_config(module) == expected
    assert PL.pool_options(module) is (expected.padding == 1)
    # PyTorch's output size on an odd and an even axis, with ceil_mode=False.
    for size in (12, 13, 24):
        if expected.aligned and size % expected.kernel:
            continue
        assert expected.output_size(size) == module(torch.zeros(1, 1, size, size)).shape[-1]


@pytest.mark.parametrize("module,fragment", UNSUPPORTED, ids=repr)
def test_unsupported_configurations_are_rejected_everywhere(module, fragment, offline_ee):
    with pytest.raises(UnsupportedOperation, match=re.escape(fragment)):
        PL.pool_config(module)
    with pytest.raises(UnsupportedOperation, match=re.escape(fragment)):
        tensor_ee.pool2d(_raster(), module)
    report = check_compatibility(nn.Sequential(module).eval(), (1, 2, 36, 24)).to_dict()
    assert report["status"] == "unsupported"
    assert [b["code"] for b in report["blockers"]] == ["unsupported_option"]
    assert fragment in report["blockers"][0]["message"]
    with pytest.raises(UnsupportedOperation, match=re.escape(fragment)):
        compile_model(nn.Sequential(module).eval(), ee.Image.constant([1, 2]).rename(["a", "b"]), ["a", "b"], GRID)
    with pytest.raises(UnsupportedReceptiveField):
        analyze_receptive_field(nn.Sequential(module).eval())


def test_the_rejection_names_the_supported_configurations():
    with pytest.raises(UnsupportedOperation) as error:
        PL.pool_config(nn.MaxPool2d(5, 1, 2))
    text = str(error.value)
    assert "k in (2, 3, 4)" in text and "max 3x3 padding 1 with stride 1 or 2" in text
    assert "avg 3x3 stride 1 padding 1" in text and "MaxPool2d(kernel_size=5, stride=1, padding=2)" in text


@pytest.mark.parametrize("module,height,width,fragment", [
    (nn.AvgPool2d(3), 36, 25, "divisible by 3"), (nn.MaxPool2d(3), 35, 24, "divisible by 3"),
    (nn.AvgPool2d(4), 36, 30, "divisible by 4"), (nn.MaxPool2d(4), 34, 24, "divisible by 4"),
    (nn.MaxPool2d(2), 36, 23, "even spatial"),
])
def test_aligned_pools_reject_sizes_their_kernel_does_not_divide(module, height, width, fragment, offline_ee):
    grid = Grid(GRID.crs, GRID.transform, width, height)
    with pytest.raises(UnsupportedOperation, match=fragment):
        tensor_ee.pool2d(_raster(grid), module)
    report = check_compatibility(nn.Sequential(module).eval(), (1, 2, height, width)).to_dict()
    assert report["status"] == "unsupported" and report["blockers"][0]["code"] == "input_shape"
    assert fragment in report["blockers"][0]["message"]


# ---------------------------------------------------------------------------
# The block: the graphs
# ---------------------------------------------------------------------------
def _pool2d_before_p02(raster, module, clip_footprints=False):
    """``pool2d`` at the base revision (aad5292) for today's two configurations, frozen."""
    def pair(value):
        return tuple(value) if isinstance(value, (tuple, list)) else (value, value)

    padded = (isinstance(module, nn.MaxPool2d) and pair(module.kernel_size) == (3, 3)
              and pair(module.stride) == (2, 2) and pair(module.padding) == (1, 1))
    if padded:
        kernel = ee.Kernel.square(1, "pixels", False)
        maximum = raster.image.reduceNeighborhood(ee.Reducer.max(), kernel, skipMasked=False)
        maximum = LR.project(maximum, raster.grid, clip_footprints)
        return LR.stride_sample(maximum, raster.bands, raster.grid, 2, clip_footprints)
    reducer = ee.Reducer.max() if isinstance(module, nn.MaxPool2d) else ee.Reducer.mean()
    grid = raster.grid.resized(2, raster.grid.width // 2, raster.grid.height // 2)
    result = raster.image.reduceResolution(reducer, bestEffort=False, maxPixels=16)
    result = LR.project(result, grid, clip_footprints)
    return LR.finish(result, raster.bands, grid, clip_footprints)


@pytest.mark.parametrize("module", [nn.AvgPool2d(2), nn.MaxPool2d(2), nn.AvgPool2d(2, 2, 0, count_include_pad=False),
                                    nn.MaxPool2d((2, 2), (2, 2)), nn.MaxPool2d(3, 2, 1),
                                    nn.MaxPool2d((3, 3), (2, 2), (1, 1))], ids=repr)
@pytest.mark.parametrize("clip", [False, True])
def test_todays_configurations_serialize_as_before_p02(module, clip, offline_ee):
    raster = _raster()
    new, old = PL.pool2d(raster, module, clip_footprints=clip), _pool2d_before_p02(raster, module, clip)
    assert (new.image.serialize(), new.bands, new.grid) == (old.image.serialize(), old.bands, old.grid)


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("cls,reducer", [(nn.AvgPool2d, "Reducer.mean"), (nn.MaxPool2d, "Reducer.max")])
def test_aligned_pools_reduce_resolution_onto_the_k_times_coarser_grid(k, cls, reducer, offline_ee):
    raster = _raster()
    out = PL.pool2d(raster, cls(k))
    assert (out.grid.width, out.grid.height) == (24 // k, 36 // k)
    assert out.grid.transform == (10 * k, 0, 1000, 0, -10 * k, 2000)
    functions = _functions(out.image)
    assert reducer in functions and "Image.reduceNeighborhood" not in functions
    arguments = _invocation(out.image, "Image.reduceResolution")
    assert arguments["image"] == _tree(raster.image)                 # the pinned input itself
    assert _constant(arguments["maxPixels"]) == (k + 2) ** 2
    assert _constant(arguments["bestEffort"]) is False


@pytest.mark.parametrize("module,reducer,divide", [
    (nn.MaxPool2d(3, 1, 1), "Reducer.max", False),
    (nn.AvgPool2d(3, 1, 1, count_include_pad=True), "Reducer.sum", True),
    (nn.AvgPool2d(3, 1, 1, count_include_pad=False), "Reducer.mean", False),
], ids=repr)
def test_stride_one_padded_pools_reduce_the_3x3_neighbourhood_on_the_input_grid(module, reducer, divide, offline_ee):
    raster = _raster()
    out = PL.pool2d(raster, module)
    assert out.grid == raster.grid and out.bands == raster.bands
    functions = _functions(out.image)
    assert reducer in functions and "Image.reduceNeighborhood" in functions
    assert {"Reducer.max", "Reducer.sum", "Reducer.mean"} & set(functions) == {reducer}
    assert "Image.changeProj" not in functions and "Image.reduceResolution" not in functions
    kernel = _invocation(out.image, "Kernel.square")
    assert _constant(kernel["radius"]) == 1 and _constant(kernel["normalize"]) is False
    assert _constant(kernel["units"]) == "pixels"
    neighbourhood = _invocation(out.image, "Image.reduceNeighborhood")
    assert _constant(neighbourhood["skipMasked"]) is False
    # The reduction reads the input as it is, masked outside its grid (never zero-filled),
    # so the padding takes no part in the reducer.
    assert neighbourhood["image"] == _tree(raster.image)
    if divide:          # count_include_pad=True: the in-bounds sum over the full window of 9
        assert _constant(_invocation(out.image, "Image.divide")["image2"]) == 9
    else:
        assert "Image.divide" not in functions
    # The stride-2 padded max pool keeps its stride sampling.
    assert "Image.changeProj" in _functions(PL.pool2d(raster, nn.MaxPool2d(3, 2, 1)).image)


# ---------------------------------------------------------------------------
# Edge semantics: the graph's masked reductions against PyTorch, at every border pixel
# ---------------------------------------------------------------------------
def _neighbourhood_model(x: np.ndarray, reducer: str, *, include_pad: bool = False,
                         pad_value: float | None = None) -> np.ndarray:
    """The 3x3 stride-1 reduction the graph emits, in NumPy: pixels outside the grid are
    masked and Earth Engine reducers skip them, so only in-bounds pixels enter (max, sum / 9,
    or the in-bounds mean). ``pad_value`` instead pads with a value (a wrong padding)."""
    height, width = x.shape[:2]
    out = np.empty(x.shape, np.float64)
    padded = None if pad_value is None else np.pad(x.astype(np.float64), ((1, 1), (1, 1), (0, 0)),
                                                   constant_values=pad_value)
    for i in range(height):
        for j in range(width):
            if padded is not None:
                window = padded[i:i + 3, j:j + 3]
            else:
                window = x[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2].astype(np.float64)
            flat = window.reshape(-1, x.shape[2])
            if reducer == "max":
                out[i, j] = flat.max(0)
            elif include_pad:
                out[i, j] = flat.sum(0) / 9
            else:
                out[i, j] = flat.mean(0)
    return out.astype(np.float32)


AVERAGE_ROUNDING = 1e-5
"""How far the double-precision model may sit from PyTorch's float32 average pooling: a few
float32 ulps of values below 1.5 (2.4e-7 on macOS arm64), with room for another platform's
summation order. The wrong divisor rule is off by more than 0.16 at every border pixel."""


def _border(shape) -> np.ndarray:
    mask = np.zeros(shape[:2], bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
    return mask


def _torch(module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return core.hwc(module(core.nchw(x)))


def test_max_padding_never_wins_at_any_border_pixel():
    fixture = registry()["raster/pool2d/max-k3-s1-p1"]
    x = fixture.data(fixture.seed).inputs["x"]
    assert x.max() < 0                                                 # all negative, never 0
    reference = _torch(nn.MaxPool2d(3, 1, 1), x)
    assert np.array_equal(_neighbourhood_model(x, "max").view(np.uint32), reference.view(np.uint32))
    zero_pad = _neighbourhood_model(x, "max", pad_value=0.0)
    differs = (zero_pad != reference).any(axis=2)
    assert np.array_equal(differs, _border(x.shape))                   # every border pixel, nothing inside
    # The same model also holds on signed data (where a zero pad can tie or lose).
    signed = core.dyadic(5, "signed", (7, 9, 2))
    assert np.array_equal(_neighbourhood_model(signed, "max"), _torch(nn.MaxPool2d(3, 1, 1), signed))


@pytest.mark.parametrize("include", [True, False])
def test_average_divisor_rules_match_pytorch_at_every_border_pixel(include):
    name = "incl" if include else "excl"
    fixture = registry()[f"raster/pool2d/avg-k3-s1-p1-{name}"]
    x = fixture.data(fixture.seed).inputs["x"]
    assert x.min() >= 0.5                                              # every border sum >= 2
    reference = _torch(nn.AvgPool2d(3, 1, 1, count_include_pad=include), x)
    model = _neighbourhood_model(x, "mean", include_pad=include)
    assert np.abs(model - reference).max() <= AVERAGE_ROUNDING
    other = _neighbourhood_model(x, "mean", include_pad=not include)
    wrong = (np.abs(other - reference) > 1e-4).any(axis=2)
    assert np.array_equal(wrong, _border(x.shape))                     # every border pixel, nothing inside
    assert np.abs(other - reference)[_border(x.shape)].min() > 0.16
    # count_include_pad=True is the zero-padded sum over 9; False ignores the padding entirely.
    zero_padded = _neighbourhood_model(x, "mean", pad_value=0.0)
    assert bool(np.abs(zero_padded - reference).max() <= AVERAGE_ROUNDING) is include


@pytest.mark.parametrize("cls,k", [(nn.AvgPool2d, 3), (nn.MaxPool2d, 3), (nn.AvgPool2d, 4), (nn.MaxPool2d, 4),
                                   (nn.MaxPool2d, 2)])
def test_aligned_block_reduction_matches_pytorch(cls, k):
    x = core.dyadic(9, f"block{k}", (4 * k, 3 * k, 2))
    blocks = x.reshape(4, k, 3, k, 2).astype(np.float64)
    model = (blocks.max((1, 3)) if cls is nn.MaxPool2d else blocks.mean((1, 3))).astype(np.float32)
    reference = _torch(cls(k), x)
    if cls is nn.MaxPool2d:
        assert np.array_equal(model, reference)
    else:
        assert np.abs(model - reference).max() <= AVERAGE_ROUNDING


@pytest.mark.parametrize("factor", [3, 4])
def test_nearest_factors_repeat_each_pixel(factor):
    x = core.dyadic(3, "nearest", (5, 7, 2))
    with torch.no_grad():
        reference = core.hwc(F.interpolate(core.nchw(x), scale_factor=factor, mode="nearest"))
    assert np.array_equal(np.repeat(np.repeat(x, factor, 0), factor, 1), reference)


# ---------------------------------------------------------------------------
# The converter: support, shapes, receptive field and tiling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("module", [m for m, _ in SUPPORTED], ids=repr)
def test_check_compatibility_and_compile_model_agree_with_pytorch_shapes(module, offline_ee):
    config = PL.pool_config(module)
    for height, width in ((36, 24), (13, 17)):
        if config.aligned and (height % config.kernel or width % config.kernel):
            continue
        expected = list(module(torch.zeros(1, 2, height, width)).shape)
        report = check_compatibility(nn.Sequential(module).eval(), (1, 2, height, width)).to_dict()
        assert report["status"] == "supported", report["blockers"]
        assert report["output_shape"] == expected
        grid = Grid(GRID.crs, GRID.transform, width, height)
        compiled = compile_model(nn.Sequential(module).eval(), ee.Image.constant([1, 2]).rename(["a", "b"]),
                                 ["a", "b"], grid)
        assert (compiled.grid.height, compiled.grid.width) == tuple(expected[2:])
        assert compiled.grid.transform[0] == 10 * config.stride
        assert compiled.bands == ("c0", "c1")


@pytest.mark.parametrize("factor", [3, 4])
def test_nearest_factors_through_the_converter(factor, offline_ee):
    for model in (nn.Upsample(scale_factor=factor, mode="nearest"),
                  nn.Upsample(scale_factor=float(factor), mode="nearest")):
        report = check_compatibility(nn.Sequential(model).eval(), (1, 2, 8, 6)).to_dict()
        assert report["status"] == "supported" and report["output_shape"] == [1, 2, 8 * factor, 6 * factor]
        grid = Grid(GRID.crs, (30, 0, 1000, 0, -30, 2000), 6, 8)
        compiled = compile_model(nn.Sequential(model).eval(), ee.Image.constant([1, 2]).rename(["a", "b"]),
                                 ["a", "b"], grid)
        assert (compiled.grid.height, compiled.grid.width) == (8 * factor, 6 * factor)
        assert compiled.grid.transform[0] == pytest.approx(30 / factor)
        support = analyze_receptive_field(nn.Sequential(model).eval())
        assert support.x.phase_bounds == tuple((p // factor, p // factor) for p in range(factor))


@pytest.mark.parametrize("modules,period,shift,bounds,alignment,halos", [
    # Stride-1 padded pools: one-pixel radius on the full-resolution lattice.
    ((nn.MaxPool2d(3, 1, 1),), 1, 1, ((-1, 1),), 1, (1, 1)),
    ((nn.AvgPool2d(3, 1, 1, count_include_pad=True),), 1, 1, ((-1, 1),), 1, (1, 1)),
    ((nn.AvgPool2d(3, 1, 1, count_include_pad=False),), 1, 1, ((-1, 1),), 1, (1, 1)),
    # Aligned k x k: output i reads [k i, k i + k - 1]; the lattice spacing is k.
    ((nn.AvgPool2d(3),), 1, 3, ((0, 2),), 3, None),
    ((nn.MaxPool2d(4),), 1, 4, ((0, 3),), 4, None),
    ((nn.MaxPool2d(3, 2, 1),), 1, 2, ((-1, 1),), 2, None),
    # Down by 3, a padded pool at /3 (coarse j reads [j - 1, j + 1]), up by 3: output i reads
    # [3 (i // 3) - 3, 3 (i // 3) + 5]. An aligned core (start and end at multiples of 3)
    # needs 3 on each side; a core of any phase needs 5 (phase 2 reads 5 to the left),
    # rounded up to the lattice: 6.
    ((nn.AvgPool2d(3), nn.AvgPool2d(3, 1, 1), nn.Upsample(scale_factor=3)), 3, 3, ((-3, 5),) * 3, 3, (3, 6)),
    # The same at k = 4: [4 (i // 4) - 4, 4 (i // 4) + 7]; aligned max(4, 7 - 3) = 4; any
    # phase 7 (phase 3 reads 7 to the left), rounded up to 8.
    ((nn.MaxPool2d(4), nn.MaxPool2d(3, 1, 1), nn.Upsample(scale_factor=4)), 4, 4, ((-4, 7),) * 4, 4, (4, 8)),
])
def test_receptive_field_of_each_configuration(modules, period, shift, bounds, alignment, halos):
    report = analyze_receptive_field(nn.Sequential(*modules).eval())
    for axis in (report.x, report.y):
        assert (axis.period, axis.input_shift, axis.phase_bounds) == (period, shift, bounds)
    assert report.alignment == alignment
    if halos is None:
        with pytest.raises(UnsupportedReceptiveField, match="full-resolution"):
            report.safe_halo
    else:
        assert (report.safe_halo, report.halo_for_any_phase) == halos


@pytest.mark.parametrize("module", [m for m, _ in SUPPORTED], ids=repr)
def test_metadata_radius_bounds_every_supported_configuration(module):
    radius = PL.METADATA["pool2d"].radius
    config = PL.pool_config(module)
    axis = analyze_receptive_field(nn.Sequential(module).eval()).x
    for index in range(3):                      # offsets from the input pixel stride * index
        lower, upper = axis.bounds(index)
        assert config.stride * index - lower <= radius and upper - config.stride * index <= radius
    assert axis.maximum_support_width <= 2 * radius + 1
    assert radius == max(k - 1 for k in PL.ALIGNED_KERNELS) == 3


class Mixed(nn.Module):
    """Every new pooling configuration and both new nearest factors in one model:
    full resolution -> /3 -> /12 -> /3 -> full resolution."""

    def __init__(self, aligned3: type = nn.AvgPool2d, aligned4: type = nn.MaxPool2d):
        super().__init__()
        self.stem = nn.Conv2d(2, 3, 3, padding=1)
        self.max_same = nn.MaxPool2d(3, 1, 1)
        self.avg_incl = nn.AvgPool2d(3, 1, 1, count_include_pad=True)
        self.down3 = aligned3(3)
        self.avg_excl = nn.AvgPool2d(3, 1, 1, count_include_pad=False)
        self.down4 = aligned4(4)
        self.bottom = nn.MaxPool2d(3, 1, 1)
        self.up4 = nn.Upsample(scale_factor=4, mode="nearest")
        self.mid = nn.AvgPool2d(3, 1, 1, count_include_pad=False)
        self.up3 = nn.Upsample(scale_factor=3, mode="nearest")
        self.head = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        x = self.avg_incl(self.max_same(self.stem(x)))
        x = self.avg_excl(self.down3(x))
        x = self.bottom(self.down4(x))
        x = self.up3(self.mid(self.up4(x)))
        return self.head(x)


MIXED = [(nn.AvgPool2d, nn.MaxPool2d), (nn.MaxPool2d, nn.AvgPool2d)]


def _tiled_forward(model, tensor, plan) -> torch.Tensor:
    """The tiling semantics in PyTorch: each context alone (its own finite boundary, clamped
    to the true one at the scene edge), then its core."""
    out = torch.empty(1, 1, plan.roi.height, plan.roi.width)
    with torch.no_grad():
        for tile in plan.tiles:
            c, k = tile.context, tile.core
            result = model(tensor[..., c.y:c.bottom, c.x:c.right])
            out[..., k.y - plan.roi.y:k.bottom - plan.roi.y, k.x - plan.roi.x:k.right - plan.roi.x] = (
                result[..., k.y - c.y:k.bottom - c.y, k.x - c.x:k.right - c.x])
    return out


@pytest.mark.parametrize("aligned3,aligned4", MIXED)
def test_tiled_mosaic_equals_the_untiled_forward_at_the_computed_halo(aligned3, aligned4):
    model = initialize_fixture(Mixed(aligned3, aligned4), 2026).eval()
    report = analyze_receptive_field(model)
    # Phase 0: up3 -> j = 0; mid -> [-1, 1]; up4 -> [-1, 0]; bottom -> [-2, 1]; down4 ->
    # [-8, 7]; avg_excl -> [-9, 8]; down3 -> [-27, 26]; avg_incl, max_same, stem -> [-30, 29].
    # Phase 11: [-18, 41]. The halo is max(30, 41 - 11) = 30, rounded up to the lattice 12.
    assert report.alignment == 12 and report.period == 12
    assert report.x.bounds(0) == (-30, 29) and report.x.bounds(11) == (-18, 41)
    assert report.safe_halo == 36
    grid = Grid("EPSG:32651", (10, 0, 500000, 0, -10, 2600000), 144, 120)
    tensor = core.nchw(core.dyadic(41, "mixed", (120, 144, 2)))
    with torch.no_grad():
        full = model(tensor)
    plan = plan_tiles(grid, 48, report.safe_halo, alignment=12)
    assert len(plan.tiles) == 9
    # The pools and copies are exact on any input size, but PyTorch's float32 convolution
    # may round a cropped context differently from the full scene (about 1 ulp of the
    # stem: 2.4e-7 on macOS arm64, 1.2e-6 on Linux CI). 1e-5 allows that and stays far
    # below the halo errors this test must catch (0.048 and more, below).
    assert (_tiled_forward(model, tensor, plan) - full).abs().max() <= 1e-5
    # One lattice step less halo changes the cores next to the internal seams.
    short = plan_tiles(grid, 48, report.safe_halo - 12, alignment=12)
    assert (_tiled_forward(model, tensor, short) - full).abs().max() > 1e-2


def test_the_tiled_graph_builds_at_the_computed_halo_and_refuses_less(offline_ee):
    model = initialize_fixture(Mixed(), 2026).eval()
    grid = Grid("EPSG:32651", (10, 0, 500000, 0, -10, 2600000), 144, 120)
    image = ee.Image.constant([1, 2]).rename(["first", "second"])
    compiled = compile_tiled_model(model, image, ["first", "second"], plan_tiles(grid, 48, 36, alignment=12))
    assert compiled.receptive_field.safe_halo == 36 and len(compiled.tiles) == 9
    assert compiled.grid == grid and compiled.bands == ("c0",)
    for tile, part in zip(compiled.plan.tiles, compiled.tiles):
        assert part.grid == tile.context.grid(grid)              # each tile at its context, full resolution
    with pytest.raises(ValueError, match="halo at least 36"):
        compile_tiled_model(model, image, ["first", "second"], plan_tiles(grid, 48, 24, alignment=12))
    with pytest.raises(ValueError, match="model alignment 12"):
        compile_tiled_model(model, image, ["first", "second"], plan_tiles(grid, 48, 36, alignment=6))
    # A single tile that is the whole scene is the untiled graph.
    whole = compile_tiled_model(model, image, ["first", "second"], plan_tiles(grid, (144, 120), 0, alignment=12))
    untiled = compile_model(model, image, ["first", "second"], grid, include_unfused_nodes=False,
                            clip_footprints=True)
    assert whole.tiles[0].image.serialize() == untiled.image.serialize()


# ---------------------------------------------------------------------------
# The p02 fixtures
# ---------------------------------------------------------------------------
P02 = ("raster/pool2d/avg-k2-s2", "raster/pool2d/max-k3-s2-p1", "raster/nearest_upsample/f2",
       "raster/pool2d/max-k2-s2", "raster/pool2d/avg-k3-s3", "raster/pool2d/max-k3-s3",
       "raster/pool2d/avg-k4-s4", "raster/pool2d/max-k4-s4", "raster/pool2d/max-k3-s1-p1",
       "raster/pool2d/avg-k3-s1-p1-incl", "raster/pool2d/avg-k3-s1-p1-excl",
       "raster/pool2d/max-k3-s2-p1-odd", "raster/nearest_upsample/f3", "raster/nearest_upsample/f4")

# SHA-256 of each new fixture's inputs (no weights): the same seed gives the same bytes.
INPUTS_SHA256 = {
    "raster/pool2d/max-k2-s2": "da494e0455268dae920ce0a5065ab06bcf20f81087b19a0315c72be5ea95f82c",
    "raster/pool2d/avg-k3-s3": "848c63787176c5ea38a7476e717d7c756c695bc2bc307a3fb4295d1bbaa3d953",
    "raster/pool2d/max-k3-s3": "45b17071a93c5cc6a7f9fcc1ebce6e38703715ca35ef214eeba5df6d471dab72",
    "raster/pool2d/avg-k4-s4": "373383272b9e9b4542ee455b8156f11fb0a7c2aec0a2ee7cd766186078ab7525",
    "raster/pool2d/max-k4-s4": "ee43356f4c538396841f70f0f7059cba269f7ae12d583b7fb389acf243835e15",
    "raster/pool2d/max-k3-s1-p1": "203c274b7e54cf74002e800aa8f95bdcb5f4f163b1e2ae34c16f152ed1ad2ed8",
    "raster/pool2d/avg-k3-s1-p1-incl": "573d8f87060e963fbfb0efe93fe95ef24ceb26b56128c95df693faa4ea18f641",
    "raster/pool2d/avg-k3-s1-p1-excl": "bbeea2d896da5d03d9b6af828b01f7b12d78c2d8dab3d37d4545f7a930be04bb",
    "raster/nearest_upsample/f3": "1998bb1cf6a0747d3a3a3d0ea79d5ad39be9a267a4987ec2a2c22ae7918a46c9",
    "raster/nearest_upsample/f4": "1667d68069cc695112027a4dea89e8e332b757af2da7bdd1e06fd20389e3c533",
    "raster/pool2d/max-k3-s2-p1-odd": "059cdc72cc6e75787eb7830c2499c0c671b45b00df4a7ef958ecf7410f15918c",
}
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


@pytest.fixture(scope="module")
def plan():
    return LF.make_plan(select(["p02"]))


@pytest.fixture(scope="module")
def local(plan):
    return {f.id: LF.evaluate_local(f, plan.data[f.id]) for f in plan.fixtures}


@contextlib.contextmanager
def offline_session(project, credentials):
    """Stands in for ``live_session``: the offline signatures, no initialization."""
    case = ApiTestCase()
    case.setUp()
    try:
        yield
    finally:
        case.tearDown()


def fake_compute_pixels(responses: dict, calls: list):
    def computePixels(params):
        assert params["fileFormat"] == "NUMPY_NDARRAY"
        sha = hashlib.sha256(params["expression"].serialize().encode()).hexdigest()
        calls.append(sha)
        return responses[sha].copy()
    return computePixels


def simulated(plan, local) -> dict:
    """{graph SHA-256: the response Earth Engine would give if it equalled PyTorch}."""
    responses = {}
    for f in plan.fixtures:
        data, values = plan.data[f.id], local[f.id]
        for request in plan.requests[f.id]:
            expected = {r.name: data.inputs[r.source] if r.kind == "input" else values.reference[r.name]
                        for r in request.readouts}
            responses[request.graph_sha256] = LF.simulate_response(request, expected)
    return responses


def _request_of(plan, fixture_id: str, readout: str):
    return next(r for r in plan.requests[fixture_id] if any(x.name == readout for x in r.readouts))


def test_the_p02_selection_and_its_plan(plan, no_network, capsys):
    assert tuple(f.id for f in plan.fixtures) == P02
    assert selections()["p02"] == P02
    assert not plan.violations
    totals = plan.totals()
    assert totals["requests"] == sum(f.requests for f in plan.fixtures) == 25 <= LF.DEFAULT_BUDGET.requests
    assert totals["max_values"] <= LF.DEFAULT_BUDGET.values
    assert totals["max_request_bytes"] <= LF.DEFAULT_BUDGET.request_bytes
    assert totals["assets"] == totals["tasks"] == 0 and totals["embedded_constants"] == 0
    assert max(max(r.width, r.height) for r in plan.all_requests) <= LF.DEFAULT_BUDGET.raster_side
    assert LF.main(["--plan", "p02"]) == 0 and "Within budget." in capsys.readouterr().out
    assert no_network == []


def test_selection_names_are_checked(monkeypatch):
    # A selection is a sequence of ids or a zero-argument callable returning one (P01's form).
    monkeypatch.setattr(RF, "SELECTIONS", {"p02": lambda: list(P02)})
    assert selections()["p02"] == P02 and [f.id for f in select(["p02"])] == list(P02)
    monkeypatch.setattr(RF, "SELECTIONS", {"p02": ("raster/pool2d/no-such-fixture",)})
    with pytest.raises((KeyError, ValueError)):
        selections()
    for name in ("all", "calibration", "raster/conv2d/k3"):
        monkeypatch.setattr(RF, "SELECTIONS", {name: ("raster/conv2d/k3",)})
        with pytest.raises((KeyError, ValueError)):
            selections()


def test_p02_fixtures_are_deterministic_and_their_data_exercises_the_edges(plan):
    new = {f.id: f for f in plan.fixtures if not f.calibration}
    assert set(new) == set(INPUTS_SHA256)
    for fid, f in new.items():
        first, second = f.data(f.seed), f.data(f.seed)
        assert first.inputs_sha256 == second.inputs_sha256 == INPUTS_SHA256[fid], fid
        assert first.weights_sha256 == EMPTY_SHA256 and f.mirror is None and not f.calibration
        x = first.inputs["x"]
        assert x.dtype == np.float32 and x.shape[0] != x.shape[1], fid             # not square
        assert list(x.shape) == f.shapes["input"]
        assert f.tolerance == (1e-4 if "/avg-" in fid else 0.0), fid
        if fid.endswith(("max-k3-s1-p1", "max-k3-s2-p1-odd")):
            assert x.max() < 0
        if fid.endswith("-odd"):
            assert x.shape[0] % 2 and x.shape[1] % 2                              # odd: bottom and right padding
        if "/avg-k3-s1-p1" in fid:
            assert x.min() >= 0.5


def test_offline_p02_passes_with_zero_cloud_calls(plan, local, no_network, tmp_path):
    report = LF.offline(select(["p02"]))
    assert report["status"] == "passed" and report["cloud_calls"] == 0
    for f in plan.fixtures:
        assert list(local[f.id].reference["out"].shape) == f.shapes["output"], f.id
        assert local[f.id].reference["out"].dtype == np.float32
    assert LF.main(["--offline", "p02", "--output", str(tmp_path / "offline.json")]) == 0
    assert no_network == []


def test_a_p02_run_answered_by_pytorch_passes(plan, local, monkeypatch, no_network):
    calls = []
    monkeypatch.setattr(ee.data, "computePixels", fake_compute_pixels(simulated(plan, local), calls))
    record = LF.run(plan.fixtures, project="test-project", session=offline_session, command="test")
    assert record["status"] == "passed", [e["failures"] for e in record["fixtures"]]
    assert record["cloud_calls"]["requests"] == len(calls) == 25
    assert no_network == []


def _run(plan, local, monkeypatch, change) -> dict:
    responses = simulated(plan, local)
    change(responses)
    monkeypatch.setattr(ee.data, "computePixels", fake_compute_pixels(responses, []))
    record = LF.run(plan.fixtures, project="p", session=offline_session)
    return {e["spec"]["id"]: e for e in record["fixtures"]}


@pytest.mark.parametrize("fid,stride,count", [("raster/pool2d/max-k3-s1-p1", 1, 100),
                                              ("raster/pool2d/max-k3-s2-p1-odd", 2, 52)])
def test_a_padding_that_wins_fails_every_border_pixel(plan, local, monkeypatch, fid, stride, count):
    x = plan.data[fid].inputs["x"]
    # Zero padding: every output window that reaches past the grid sees a 0 that wins over
    # the all-negative inputs. At stride 2 on 13 x 15 that is every border pixel of the 7 x 8
    # output: the top and left padding, and the bottom and right padding the odd sizes reach.
    zero_pad = _neighbourhood_model(x, "max", pad_value=0.0)[::stride, ::stride]
    reference = local[fid].reference["out"]
    assert np.array_equal(_neighbourhood_model(x, "max")[::stride, ::stride], reference)   # masked: exact
    assert np.array_equal((zero_pad != reference).any(axis=2), _border(reference.shape))
    request = _request_of(plan, fid, "out")

    def change(responses):
        for c in range(2):
            responses[request.graph_sha256][f"out_{c}"] = zero_pad[..., c]
    by_id = _run(plan, local, monkeypatch, change)
    failed = {k for k, e in by_id.items() if e["status"] == "failed"}
    assert failed == {fid}
    assert by_id[fid]["outputs"]["out"]["pytorch"]["over_tolerance"] == 2 * int(_border(reference.shape).sum()) == count


@pytest.mark.parametrize("include", [True, False])
def test_the_wrong_divisor_fails_every_border_pixel(plan, local, monkeypatch, include):
    fid = f"raster/pool2d/avg-k3-s1-p1-{'incl' if include else 'excl'}"
    x = plan.data[fid].inputs["x"]
    other = _neighbourhood_model(x, "mean", include_pad=not include)
    request = _request_of(plan, fid, "out")

    def change(responses):
        for c in range(2):
            responses[request.graph_sha256][f"out_{c}"] = other[..., c]
    by_id = _run(plan, local, monkeypatch, change)
    assert {k for k, e in by_id.items() if e["status"] == "failed"} == {fid}
    assert by_id[fid]["outputs"]["out"]["pytorch"]["over_tolerance"] == 100


EXACT = ("raster/pool2d/max-k2-s2", "raster/pool2d/max-k3-s3", "raster/pool2d/max-k4-s4",
         "raster/pool2d/max-k3-s1-p1", "raster/pool2d/max-k3-s2-p1-odd", "raster/nearest_upsample/f3",
         "raster/nearest_upsample/f4")


def test_one_ulp_fails_every_selection_and_copy_but_not_an_average(plan, local, monkeypatch):
    changed = {}

    def change(responses):
        for fid in (*EXACT, "raster/pool2d/avg-k3-s3", "raster/pool2d/avg-k3-s1-p1-excl"):
            band = responses[_request_of(plan, fid, "out").graph_sha256]["out_1"]
            row, col = band.shape[0] - 1, band.shape[1] // 2          # a border pixel
            band[row, col] = np.nextafter(band[row, col], np.float32(np.inf))
            changed[fid] = (row, col)
    by_id = _run(plan, local, monkeypatch, change)
    assert {k for k, e in by_id.items() if e["status"] == "failed"} == set(EXACT)
    for fid in EXACT:
        result = by_id[fid]["outputs"]["out"]["pytorch"]
        row, col = changed[fid]
        assert result["over_tolerance"] == 1 and (result["worst"]["row"], result["worst"]["col"]) == (row, col)
        assert any("differ bit for bit" in f and f"row {row}, col {col}, channel 1" in f
                   for f in by_id[fid]["failures"])
    for fid in ("raster/pool2d/avg-k3-s3", "raster/pool2d/avg-k3-s1-p1-excl"):   # within 1e-4
        assert by_id[fid]["status"] == "passed" and by_id[fid]["outputs"]["out"]["pytorch"]["max_abs"] > 0


# ---------------------------------------------------------------------------
# R5: the live plan's numbers are --plan's
# ---------------------------------------------------------------------------
PLAN_DOC = ROOT / "docs" / "evidence" / "p02-live-plan.md"
ROW = re.compile(r"^\| `(?P<id>raster/[^`]+)` \| (?P<index>\d+) \| (?P<rows>\d+) × (?P<cols>\d+) \| "
                 r"(?P<bands>[\d,]+) \| (?P<values>[\d,]+) \| (?P<payload>[\d,]+) \| (?P<request>[\d,]+) \| "
                 r"(?P<inline>[\d,]+) \| [^|]+ \| `(?P<graph>[0-9a-f]{12})` \|$")


def _number(text: str) -> int:
    return int(text.replace(",", ""))


def test_the_live_plan_document_matches_the_plan(plan):
    text = PLAN_DOC.read_text()
    recorded = re.search(r"earthengine-api\s+(\d+\.\d+\.\d+)", text).group(1)
    rows = {(m["id"], int(m["index"])): m for m in map(ROW.match, text.splitlines()) if m}
    requests = {(r.fixture, r.index): r for r in plan.all_requests}
    assert set(rows) == set(requests)
    same_client = package_version("earthengine-api") == recorded
    for key, r in requests.items():
        m = rows[key]
        assert (_number(m["rows"]), _number(m["cols"])) == (r.height, r.width), key
        assert (_number(m["bands"]), _number(m["values"]), _number(m["payload"])) == (
            len(r.bands), r.values, r.payload_bytes), key
        assert _number(m["inline"]) == r.inline_values, key
        if same_client:
            assert _number(m["request"]) == r.request_bytes, key
            assert m["graph"] == r.graph_sha256[:12], key
    totals = plan.totals()
    for label, value in (("requests", totals["requests"]), ("values", totals["values"]),
                         ("payload", totals["payload_bytes"]), ("inline", totals["inline_values"])):
        assert f"{value:,}" in text, label
    if same_client:
        assert f"{totals['request_bytes']:,}" in text
    assert "tools/live_fixtures.py --run p02 --project ORIGINAL_PROJECT" in text
