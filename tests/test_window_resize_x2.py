"""DPT x2/align-corners resize: global coefficients, halo reads and both borders."""
import ee
import numpy as np
import pytest

from tensor_ee.blocks.window.resample import METADATA, local_matrices
from tensor_ee.converter.request import deep_recursion
from tensor_ee.core.metadata import Locality, check_settings, metadata_at
from tools.fixtures import resize_x2 as F
from tools.fixtures import select
from tools.live_fixtures import make_plan, offline_earth_engine
from reference.window_resample import resize_weights, upsample


@pytest.mark.parametrize("extent", F.EXTENTS)
def test_local_weights_reconstruct_every_global_row_with_only_one_token_halo(extent):
    expected = resize_weights(extent, extent * 2, True)
    stack, index = local_matrices(extent, 2, True, window=7)
    reconstructed = np.zeros_like(expected)
    for child, k in enumerate(index):
        for h in range(9):
            source = 7 * (child // 2) - 1 + h
            if 0 <= source < extent:
                reconstructed[7 * child:7 * (child + 1), source] = stack[7 * k:7 * (k + 1), h]
    np.testing.assert_array_equal(reconstructed, expected)
    assert expected[0, 0] == expected[-1, -1] == 1


@pytest.mark.parametrize("fixture", F.fixtures(), ids=lambda f: f.id)
def test_graph_equals_exact_mirror_and_pytorch_within_target(fixture):
    import test_p01_window_settings as P01
    import torch
    import torch.nn.functional as functional
    from tools.fixtures.core import hwc, nchw

    data = fixture.data(fixture.seed)
    mirror, reference = fixture.mirror.fn(data), fixture.reference(data)
    x = F.full_input(data)
    with torch.no_grad():
        full_reference = hwc(functional.interpolate(nchw(x), scale_factor=2, mode="bilinear", align_corners=True))
    assert np.abs(upsample(x, factor=2) - full_reference).max() <= 1e-4
    with offline_earth_engine(), deep_recursion():
        for readout in fixture.build(ee, data):
            evaluate = P01.GraphEvaluator()
            values, valid = P01.read_grid(evaluate(readout.image), readout.grid)
            got = np.stack(values, axis=-1)
            assert valid.all() and got.dtype == np.float32 and got.shape == readout.shape
            expected = data.inputs[readout.source] if readout.kind == "input" else mirror[readout.name]
            np.testing.assert_array_equal(got.view(np.uint32), expected.view(np.uint32))
            if readout.kind == "output":
                assert np.abs(got.astype(np.float64) - reference[readout.name]).max() <= 1e-4


@pytest.mark.parametrize("fixture", F.fixtures(), ids=lambda f: f.id)
def test_wrong_corner_rule_and_independent_window_resizes_are_observable(fixture):
    x = F.full_input(fixture.data(fixture.seed))
    expected = fixture.reference(fixture.data(fixture.seed))
    wrong_corners = F.read_outputs(upsample(x, factor=2, align_corners=False))
    assert max(np.abs(expected[k] - wrong_corners[k]).max() for k in expected) > 1e-3
    if x.shape[0] > 7:
        local = np.empty((2 * x.shape[0], 2 * x.shape[1], x.shape[2]), np.float32)
        for y in range(0, x.shape[0], 7):
            for c in range(0, x.shape[1], 7):
                local[2*y:2*y+14, 2*c:2*c+14] = upsample(x[y:y+7, c:c+7], factor=2)
        wrong_local = F.read_outputs(local)
        assert max(np.abs(expected[k] - wrong_local[k]).max() for k in expected) > 1e-3


def test_plan_covers_all_global_axis_patterns_and_full_inputs_within_budget():
    fixtures = select(["resize_x2"])
    with deep_recursion():
        plan = make_plan(fixtures)
    assert not plan.violations and len(plan.all_requests) == 17
    assert plan.totals()["assets"] == plan.totals()["tasks"] == 0
    for fixture in fixtures:
        extent = fixture.shapes["tokens"][0]
        assert extent % fixture.window == (2 * extent) % fixture.window == 0
        covered = {i for ys, _ in F.output_slices(extent).values() for i in range(ys.start, ys.stop)}
        assert covered == set(range(2 * extent))
        assert sum(v.size for v in plan.data[fixture.id].inputs.values()) == extent**2 * F.CHANNELS
        for request in plan.requests[fixture.id]:
            assert request.width <= 56 and request.height <= 56


def test_new_setting_is_outside_live_record_and_existing_settings_stay_inside():
    entry = METADATA["window_resize"]
    for window in (7, 8):
        for factor, corners in ((2, False), (4, False), (8, False), (4, True)):
            assert check_settings(entry, {"window": window, "factor": factor, "align_corners": corners}).inside
    assert not check_settings(entry, {"window": 7, "factor": 2, "align_corners": True}).inside
    at = metadata_at(entry, {"align_corners": True})
    assert (at.locality, at.radius, at.chip_edge_dependent) == (Locality.WINDOW_RING, 1, True)


@pytest.mark.parametrize("factor,corners", [(2, False), (4, False), (2, True), (4, True)])
def test_historical_head_mirror_keeps_its_previous_numerics(factor, corners):
    pytest.importorskip("timm")
    pytest.importorskip("segmentation_models_pytorch")
    import torch
    from reference.upernet.ee_mirror_head import upsample as historical
    from reference.window_resample import _pairs

    x = F.full_input(F._data(9614, extent=14))
    matrix = resize_weights(14, 14 * factor, corners)
    # Historical y stage: torch double matrix multiplication, then two explicit
    # double products and one sum in x. This checks the extracted NumPy path.
    a = (torch.from_numpy(matrix) @ torch.from_numpy(x.astype(np.float64).reshape(14, -1))).numpy()
    a = a.reshape(14 * factor, 14, F.CHANNELS)
    i0, i1, w0, w1 = _pairs(matrix)
    old = a[:, i0] * w0[None, :, None] + a[:, i1] * w1[None, :, None]
    for exact in (True, False):
        expected = old if exact else old.astype(np.float32).astype(np.float64)
        np.testing.assert_array_equal(historical(x, factor=factor, align_corners=corners, exact=exact), expected)
