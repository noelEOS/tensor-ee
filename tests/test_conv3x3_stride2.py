"""Stride-2 window convolution: phase, padding, bias, grids and refusals."""
from __future__ import annotations

import ee
import numpy as np
import pytest

from tensor_ee.blocks.window.conv import Conv3x3Constants, METADATA, conv3x3_window
from tensor_ee.converter.request import deep_recursion
from tensor_ee.core.errors import UnsupportedOperation
from tensor_ee.core.grid import TokenGrid
from tensor_ee.core.metadata import Locality, check_settings
from tensor_ee.core.ops import WindowOps
from tensor_ee.weights.constants import WindowLayout
from tools.fixtures import select
from tools.fixtures import conv3x3_stride2 as F
from tools.live_fixtures import make_plan, offline_earth_engine


def test_stride2_selection_plan_and_deterministic_data():
    fixtures = select(["conv3x3_stride2"])
    assert tuple(f.id for f in fixtures) == F.SELECTIONS["conv3x3_stride2"]
    for fixture in fixtures:
        first, second = fixture.data(fixture.seed), fixture.data(fixture.seed)
        assert (first.inputs_sha256, first.weights_sha256) == (second.inputs_sha256, second.weights_sha256)
        other = fixture.data(fixture.seed + 1)
        assert first.inputs_sha256 != other.inputs_sha256 and first.weights_sha256 != other.weights_sha256
    with deep_recursion():
        plan = make_plan(fixtures)
    assert len(plan.all_requests) == 4


@pytest.mark.parametrize("fixture", F.fixtures(), ids=lambda f: f.id)
def test_graph_equals_mirror_bitwise_and_torch_within_target(fixture):
    import test_p01_window_settings as P01

    data = fixture.data(fixture.seed)
    reference, mirror = fixture.reference(data), fixture.mirror.fn(data)
    with offline_earth_engine(), deep_recursion():
        readouts = fixture.build(ee, data)
        evaluate = P01.GraphEvaluator()
        for readout in readouts:
            values, valid = P01.read_grid(evaluate(readout.image), readout.grid)
            got = np.stack(values, axis=-1)
            assert valid.all() and got.dtype == np.float32 and got.shape == readout.shape
            expected = data.inputs[readout.source] if readout.kind == "input" else mirror[readout.name]
            np.testing.assert_array_equal(got.view(np.uint32), expected.view(np.uint32))
            if readout.kind != "input":
                assert np.abs(got.astype(np.float64) - reference[readout.name]).max() <= 1e-4
    if fixture.settings["crop"] is not None:
        whole = F._reference(data, window=fixture.window, crop=None)["out"]
        assert np.abs(whole - reference["out"]).max() > 1e-3
    from reference.conv_stride import conv3x3_stride
    unbiased = conv3x3_stride(data.inputs["x"], data.weights["weight"],
                             crop_side=None if fixture.settings["crop"] is None else fixture.window * fixture.settings["crop"])
    assert np.abs(unbiased - mirror["out"]).max() > 1e-3


@pytest.mark.parametrize("stride", [0, -1, 3, True, 2.0, "2", None])
def test_unsupported_stride_is_refused(stride):
    with offline_earth_engine():
        ops = WindowOps(ee)
        constants = Conv3x3Constants(ops, np.ones((1, 1, 3, 3), np.float32), rows=49)
        with pytest.raises(UnsupportedOperation, match="stride"):
            conv3x3_window(ops, ee.Image("stride/x"), constants, TokenGrid(14, 14).coarse(14),
                           window=14, stride=stride)


@pytest.mark.parametrize("window", [7, 3, 0, -2, True, 14.0])
def test_stride_must_divide_positive_integer_window(window):
    with offline_earth_engine():
        ops = WindowOps(ee)
        constants = Conv3x3Constants(ops, np.ones((1, 1, 3, 3), np.float32))
        with pytest.raises(UnsupportedOperation, match="window"):
            conv3x3_window(ops, ee.Image("stride/x"), constants, TokenGrid(14, 14).coarse(14),
                           window=window, stride=2)


def test_constants_serve_output_rows_and_default_graph_is_unchanged():
    from tensor_ee.blocks.window.conv import _finish
    from tensor_ee.layouts.window import halo, im2col
    from tensor_ee.weights.constants import RowMismatch

    with offline_earth_engine(), deep_recursion():
        ops = WindowOps(ee)
        coarse = TokenGrid(28, 28).coarse(14)
        x = ee.Image("stride/x")
        weight, bias = np.ones((2, 1, 3, 3), np.float32), np.ones(2, np.float32)
        output_constants = Conv3x3Constants(ops, weight, bias, rows=WindowLayout(7))
        input_constants = Conv3x3Constants(ops, weight, bias, rows=WindowLayout(14))
        with pytest.raises(RowMismatch):
            conv3x3_window(ops, x, input_constants, coarse, window=14, stride=2)
        got = conv3x3_window(ops, x, output_constants, coarse, window=14, stride=2)
        assert got.serialize().count('Image.matrixMultiply') == 1
        for bias_value in (None, bias):
            c = Conv3x3Constants(ops, weight, bias_value, rows=WindowLayout(14))
            old = _finish(ops, ops.matmul_double(im2col(ee, halo(ee, x, coarse, c.cin, window=14),
                                                       c.cin, c.k, window=14), c.weight_t), c)
            assert conv3x3_window(ops, x, c, coarse, window=14).serialize() == old.serialize()
            assert conv3x3_window(ops, x, c, coarse, window=14, stride=1).serialize() == old.serialize()
        with pytest.raises(UnsupportedOperation, match="crop"):
            conv3x3_window(ops, x, output_constants, coarse, window=14, stride=2, crop=3)


def test_stride2_is_outside_record_and_existing_coverage_is_preserved():
    entry = METADATA["conv3x3_window"]
    assert (entry.layout_in, entry.layout_out, entry.locality, entry.radius, entry.chip_edge_dependent) == (
        "W", "W", Locality.WINDOW_RING, 1, False)
    for window, crop in [(w, None) for w in (5, 7, 8, 12)] + [(7, 2), (7, 32)]:
        settings = {"k": 3, "window": window, "crop": crop, "stride": 1}
        assert check_settings(entry, settings).inside
        settings["stride"] = 2
        check = check_settings(entry, settings)
        assert not check.inside and check.outside == ("stride",) and check.unstated == ()


def test_full_dpt_width_1024_offline():
    from reference.conv_stride import conv3x3_stride

    data = F._data(9514, window=14, windows=1, cin=1024, cout=1024)
    expected = F._reference(data, window=14, crop=None)["out"]
    got = conv3x3_stride(data.inputs["x"], data.weights["weight"], data.weights["bias"])
    error = np.abs(got.astype(np.float64) - expected).max()
    print(f"DPT 1024 channels mirror-vs-PyTorch max_abs={error:.17g}")
    assert got.shape == (7, 7, 1024) and error <= 1e-4
