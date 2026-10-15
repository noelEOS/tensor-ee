"""Milestone 12, Phase 4: the trained architecture at 224 x 224 (stage 4 a single unshifted window,
PSP bins on 7 x 7, the head's 56 -> 224 align_corners coefficients)."""

import numpy as np


def test_milestone5_rule_for_8x8_windows():
    from tensor_ee.models.swin_t224 import configuration as m5
    assert m5.block_shifts(8, 8) == [[0, 3], [0, 3], [0, 3, 0, 3, 0, 3], [0, 0]]
    m5.check_windows(8, 8)
    assert [g for g in m5.stage_windows(8, 8)] == [(8, 8), (4, 4), (2, 2), (1, 1)]


def test_head_coefficients_are_the_float32_model_rule():
    from tensor_ee.blocks.window import resample as RS
    from reference.upernet.helpers import torch_resize_matrix
    ee = RS.model_bilinear_weights(56, 224, align_corners=True)
    assert np.array_equal(np.float32(ee), np.float32(torch_resize_matrix(56, 224, True)))


def test_psp_bins_on_7x7():
    from tensor_ee.blocks.window.pooling import PoolPlan
    from reference.upernet.adaptive_pool_reference import check_rule
    plan = PoolPlan(7, 7)
    assert plan.count == 1 + 4 + 9 + 36
    for s in (1, 2, 3, 6):
        rows = [r for r in plan.rows if r["size"] == s]
        bins = sorted({(r["r0"], r["r1"]) for r in rows})
        assert bins == [tuple(b) for b in check_rule(7, s)]
