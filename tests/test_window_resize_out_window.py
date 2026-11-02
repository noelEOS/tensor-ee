"""Output windows affect coefficient slices, never interpolation arithmetic or casts."""
import ee
import numpy as np
import pytest

from tensor_ee.blocks.window import resample as R
from tensor_ee.converter.request import deep_recursion
from tensor_ee.core.errors import UnsupportedOperation
from tensor_ee.core.grid import TokenGrid
from tensor_ee.core.metadata import check_settings
from tools.fixtures import resize_x2_out_window as F
from tools.fixtures import select
from tools.live_fixtures import make_plan, offline_earth_engine
from reference.window_resample import resize_weights, upsample


@pytest.mark.parametrize("extent,window,out_window", F.CASES)
def test_new_fixtures_cover_the_dpt_fusion_layouts_and_full_outputs(extent, window, out_window):
    fixture = next(f for f in F.fixtures() if f.settings["window"] == window)
    data = fixture.data(fixture.seed)
    assert fixture.settings == {"factor": 2, "align_corners": True, "window": window, "out_window": out_window}
    assert fixture.shapes["tokens"] == [extent, extent]
    assert fixture.reference(data)["out_0"].shape == (2 * extent, 2 * extent, 2)
    np.testing.assert_array_equal(fixture.mirror.fn(data)["out_0"],
                                  upsample(F.R.full_input(data), factor=2).astype(np.float32))


@pytest.mark.parametrize("extent,window,out_window", [
    (7, 7, 14), (14, 7, 14), (28, 7, 14), (14, 14, 7), (28, 14, 7),
])
def test_every_coefficient_row_is_global_and_reads_only_the_parent_halo(extent, window, out_window):
    matrix = resize_weights(extent, 2 * extent, True)
    stack, index = R.local_matrices(extent, 2, True, window=window, out_window=out_window)
    reconstructed = np.zeros_like(matrix)
    for child, k in enumerate(index):
        parent = child * out_window // (2 * window)
        for h in range(window + 2):
            source = window * parent - 1 + h
            if 0 <= source < extent:
                reconstructed[out_window*child:out_window*(child+1), source] = stack[
                    out_window*k:out_window*(k+1), h]
    np.testing.assert_array_equal(reconstructed, matrix)
    assert R.window_resize_values(extent, extent, 2, align_corners=True, window=window,
                                   out_window=out_window) == 2 * (stack.size + len(index))


@pytest.mark.parametrize("window,out_window,height,width", [
    (7, 14, 7, 7), (14, 7, 14, 14), (7, 14, 14, 28), (14, 7, 28, 14),
])
def test_graph_matches_mirror_bitwise_and_pytorch_within_target(window, out_window, height, width):
    import test_p01_window_settings as P01
    import torch
    import torch.nn.functional as functional
    from tools.fixtures.core import FixtureData, dyadic, hwc, nchw

    x = dyadic(9800 + window, "rectangular", (height, width, 2))
    data = FixtureData(inputs={"x_0_0": x}, weights={})
    expected = upsample(x, factor=2).astype(np.float32)
    with torch.no_grad():
        reference = hwc(functional.interpolate(nchw(x), scale_factor=2, mode="bilinear", align_corners=True))
    assert np.abs(expected.astype(np.float64) - reference).max() <= 1e-4
    with offline_earth_engine(), deep_recursion():
        for readout in F._build(ee, data, window=window, out_window=out_window):
            values, valid = P01.read_grid(P01.GraphEvaluator()(readout.image), readout.grid)
            got = np.stack(values, axis=-1)
            assert valid.all() and got.dtype == np.float32 and got.shape == readout.shape
            target = x if readout.kind == "input" else expected
            np.testing.assert_array_equal(got.view(np.uint32), target.view(np.uint32))


@pytest.mark.parametrize("window", [7, 8])
@pytest.mark.parametrize("factor,corners", [(2, False), (4, False), (8, False), (4, True)])
def test_default_and_explicit_same_window_graphs_and_costs_stay_identical(window, factor, corners):
    with offline_earth_engine(), deep_recursion():
        tokens, x = TokenGrid(2 * window, 2 * window), ee.Image("resize/x")
        default, fine = R.window_resize(ee, x, tokens, 2, factor, corners, window=window)
        for output in (None, window):
            got, got_grid = R.window_resize(ee, x, tokens, 2, factor, corners, window=window, out_window=output)
            assert got_grid == fine and got.serialize() == default.serialize()
            assert R.window_resize_values(tokens.height, tokens.width, factor, align_corners=corners,
                                          window=window, out_window=output) == R.window_resize_values(
                                              tokens.height, tokens.width, factor, align_corners=corners, window=window)


@pytest.mark.parametrize("out_window", [0, -1, True, 7.0, "7", 3, 28])
def test_unsupported_output_windows_are_refused_before_a_graph_is_built(out_window):
    with pytest.raises(UnsupportedOperation, match="out_window"):
        R.local_matrices(7, 2, True, window=7, out_window=out_window)


def test_four_request_plan_and_default_graph_shape_use_two_products_one_float32_cast():
    with deep_recursion():
        plan = make_plan(select(["resize_x2_out_window"]))
    assert not plan.violations and len(plan.all_requests) == 4
    assert plan.totals()["assets"] == plan.totals()["tasks"] == 0
    with offline_earth_engine(), deep_recursion():
        for _, window, out_window in F.CASES:
            out, _ = R.window_resize(ee, ee.Image("resize/x"), TokenGrid(window, window), 2, 2, True,
                                     window=window, out_window=out_window)
            graph = out.serialize()
            assert graph.count('"Image.matrixMultiply"') == 2
            assert graph.count('"Image.toFloat"') == 1


def test_new_output_windows_are_outside_live_coverage_and_historical_default_is_retained():
    entry = R.METADATA["window_resize"]
    assert check_settings(entry, {"window": 7, "factor": 2, "align_corners": True, "out_window": None}).inside
    for fixture in F.fixtures():
        assert not check_settings(entry, fixture.settings).inside
    historical = check_settings(entry, {"window": 7, "factor": 2, "align_corners": True})
    assert not historical.outside and historical.unstated == ("out_window",)


def test_completed_same_window_live_graphs_and_inputs_are_unchanged():
    import json
    from pathlib import Path

    record = json.loads((Path(__file__).resolve().parents[1] /
                         "docs/evidence/live-fixtures/20261102T190649Z-resize_x2.json").read_text())
    with deep_recursion():
        plan = make_plan(select(["resize_x2"]))
    for saved in record["fixtures"]:
        fid = saved["spec"]["id"]
        assert plan.data[fid].inputs_sha256 == saved["inputs_sha256"]
        assert plan.data[fid].weights_sha256 == saved["weights_sha256"]
        assert [request.graph_sha256 for request in plan.requests[fid]] == [
            request["graph_sha256"] for request in saved["requests"]]
