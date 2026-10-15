"""Milestone 8, offline: PyTorch's AdaptiveAvgPool2d (Tests 2-3 reference half, Phase 3) and
the EE pooling mirror."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensor_ee.blocks.window import pooling as EP
from tensor_ee.blocks.window import resample as RS
from reference.upernet import adaptive_pool_reference as AP
from reference.upernet import ee_mirror_psp as MP


@pytest.mark.parametrize("n", [7, 14, 28, 56])
@pytest.mark.parametrize("m", [1, 2, 3, 6])
def test_bin_rule_equals_pytorch(n, m):
    """Test 2: the written-out rule equals the bins read off PyTorch with one-hot inputs."""
    assert AP.check_rule(n, m) == AP.bins(n, m)


def test_28_to_6_overlapping_bins():
    """Test 3: every bin start / end of 28 -> 6 (and 28 -> 3) exactly."""
    assert AP.bins(28, 6) == [(0, 5), (4, 10), (9, 14), (14, 19), (18, 24), (23, 28)]
    assert [e - s for s, e in AP.bins(28, 6)] == [5, 6, 5, 5, 6, 5]
    assert AP.overlaps(28, 6) == [4, 9, 18, 23]
    assert AP.bins(28, 3) == [(0, 10), (9, 19), (18, 28)] and AP.overlaps(28, 3) == [9, 18]
    assert AP.bins(28, 2) == [(0, 14), (14, 28)] and AP.overlaps(28, 2) == []
    assert AP.bins(56, 6) == [(0, 10), (9, 19), (18, 28), (28, 38), (37, 47), (46, 56)]


def test_stacked_membership():
    a, areas = AP.membership(28, 28)
    assert a.shape == (50, 784) and set(np.unique(a)) == {0.0, 1.0}
    assert areas[0] == 784 and list(areas[1:5]) == [196] * 4
    rows = AP.pool_rows(28, 28)
    assert [r["size"] for r in rows] == [1] + [2] * 4 + [3] * 9 + [6] * 36
    # the 6 x 6 bin (1, 4): rows [4, 10) x columns [18, 24), 36 pixels
    r = rows[14 + 6 * 1 + 4]
    assert (r["r0"], r["r1"], r["c0"], r["c1"], r["area"]) == (4, 10, 18, 24, 36)


@pytest.mark.parametrize("h", [7, 14, 28, 56])
def test_onehot_fixture_recovers_exact_source_sets(h):
    """The one-hot x ID fixture's bin sums are (t + 1) exactly on the members of each bin."""
    x = AP.onehot_fixture(h, h)
    a, areas = AP.membership(h, h)
    sums = MP.pool_sums(x).numpy()
    assert np.array_equal(sums, a * (np.arange(h * h) + 1)[None])


def test_pytorch_float32_semantics():
    """Phase 3: channels-last (SMP's format) = sequential row-major float32 sum, one division
    by the area; NCHW-contiguous = the same sum divided by kh, then kw; size 1 = torch.mean."""
    g = np.random.default_rng(5)
    for h in (28, 56):
        x = g.standard_normal((h, h, 96)).astype(np.float32)
        c = AP.characterize(x)
        assert c[1]["channels_last"]["torch_mean"] == 1.0
        for s in (2, 3, 6):
            assert c[s]["channels_last"]["seq_rowmajor_div_area"] == 1.0
            assert c[s]["contiguous"]["seq_rowmajor_div_rows_then_cols"] == 1.0


def test_memory_format_selects_the_kernel():
    x = np.random.default_rng(3).standard_normal((28, 28, 64)).astype(np.float32)
    bare = torch.from_numpy(x).permute(2, 0, 1)[None]                 # batch stride C
    assert bare.is_contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        got = torch.nn.functional.adaptive_avg_pool2d(bare, 6)[0].permute(1, 2, 0).numpy()
    assert np.array_equal(got, AP.emulate_float32(x, 6, "rows_cols"))   # NCHW kernel
    assert np.array_equal(AP.torch_pool(x, 6), AP.emulate_float32(x, 6, "area"))


def test_mirror_window_order_is_the_exact_mean_on_moderate_values():
    """Double sums of <= 3,136 float32 values of moderate range are exact: EE's pooled maps
    are the correctly rounded means, whatever the order."""
    from reference.upernet.smp_psp_reference import synthetic_c4
    for h in (28, 56):
        x = synthetic_c4(h, h)[..., :64]
        orders = MP.pool_orders(x)
        exact = AP.exact_pool(x)
        for order, value in orders.items():
            assert np.array_equal(value, exact), order


@pytest.mark.parametrize("s", [1, 2, 3, 6])
def test_cancellation_fixture_separates_orders(s):
    x = AP.cancellation_fixture(28, 28, 16, s, seed=3)
    o = MP.pool_orders(x)
    a, b = AP.offsets()[s]
    assert not np.array_equal(o["window"][a:b], o["exact"][a:b])
    assert not np.array_equal(o["window"][a:b], o["rowmajor"][a:b])


def test_pool_plan_and_tables():
    plan = EP.PoolPlan(28, 28)
    ry, rx = plan.axis_tables()
    assert plan.count == 50 and ry.shape == (50, 28) and rx.shape == (50, 28)
    assert plan.embedded_values == 50 * 56 + 50
    a, _ = AP.membership(28, 28)
    assert np.array_equal(np.einsum("ki,kj->kij", ry, rx).reshape(50, -1), a)
    assert EP.PoolPlan(56, 56).count == 50


def test_window_matrix_weights_rows():
    w = RS.window_matrix_weights((3, 3), (28, 28))
    full = RS.output_weights((3, 3), (28, 28))
    order = AP.window_token_order(28, 28)
    assert np.array_equal(w, full[order])
    tile = RS.window_matrix_weights((6, 6), (56, 56), 14, 14, 21, 21)
    assert np.array_equal(tile[0], RS.output_weights((6, 6), (56, 56))[21 * 56 + 21])
