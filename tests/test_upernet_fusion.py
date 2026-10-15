"""Milestone 10, offline: the fusion mirror against SMP, the resizes' source rule and weights, the
concatenation order, the 9,216-term accumulation and the zero-padding semantics (Tests 4-6, 8-13,
17 offline halves; Phases 3-11, 22)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("segmentation_models_pytorch")

from tensor_ee.blocks.window import resample as RS  # noqa: E402
from reference.upernet import ee_mirror_decoder as M  # noqa: E402
from reference.upernet import ee_mirror_fpn as MF  # noqa: E402
from reference.upernet import ee_mirror_fusion as MU  # noqa: E402
from reference.upernet import pytorch_decoder_primitives as P  # noqa: E402
from reference.upernet import smp_feature_interface as S  # noqa: E402
from reference.upernet import smp_fusion_reference as R  # noqa: E402


@pytest.fixture(scope="module")
def refs():
    return R.references(R.FR.synthetic_features(7), refs=("torch_float32", "float64", "ee_mirror",
                                                          "torch_float32_fusion_input"))


def test_mirror_tracks_smp_through_the_fusion(refs):
    """Phase 11: every fusion tensor of the mirror within 1e-4 of SMP float32 and 1e-6 of float64."""
    for name in R.TENSORS:
        mirror = np.asarray(refs["ee_mirror"][name])
        assert np.abs(mirror - refs["torch_float32"][name]).max() <= 1e-4, name
        assert np.abs(mirror - refs["float64"][name]).max() <= 1e-6, name


def test_mirror_composition_is_the_installed_one(refs):
    m = refs["ee_mirror"]
    assert np.array_equal(m["P1_to_C1"], m["P1"])                                   # identity
    assert np.array_equal(m["P2_to_C1"], M.resize(m["P2"], (56, 56)))
    assert np.array_equal(m["P3_to_C1"], M.resize(m["P3"], (56, 56)))
    assert np.array_equal(m["PSP_to_C1"], M.resize(m["psp_relu"], (56, 56)))
    assert np.array_equal(m["fusion_concat"][..., 256:512], m["P2_to_C1"])
    assert np.array_equal(m["fusion_concat"][..., 768:], m["PSP_to_C1"])
    d = S.decoder_parameters()
    assert np.array_equal(m["fusion_conv"], MF.conv3x3(m["fusion_concat"], d[R.WEIGHT]))
    assert np.array_equal(m["fusion_relu"], M.relu(M.batch_norm(m["fusion_conv"], *R.bn_args(d))))


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_window_resize_weights_are_pytorchs_rule(factor):
    """Tests 4-6 (offline): every output row of the window-local matrices (the weights EE
    applies) equals PyTorch's written rule src = max((o + 1/2) / f - 1/2, 0), i0, i1 = min(i0 +
    1, n - 1) (edge clamp), weights 1 - (src - i0), src - i0; all dyadic (exact in float32)."""
    n_in = 7 * 4
    stack, index = RS.local_matrices(n_in, factor)
    formula = P.bilinear_formula(n_in, factor * n_in)
    for child, k in enumerate(index):
        parent = child // factor
        local = stack[7 * k:7 * k + 7].astype(np.float64)
        for a in range(7):
            o = 7 * child + a
            _, i0, i1, l0, l1 = formula[o]
            row = np.zeros(n_in)
            for h in range(9):
                i = 7 * parent - 1 + h
                if 0 <= i < n_in:
                    row[i] += local[a, h]
            want = np.zeros(n_in)
            want[i0] += l0
            want[i1] += l1
            assert np.array_equal(row, want), (child, a)
    assert np.array_equal(stack.astype(np.float64) * 2 * factor, np.round(stack.astype(np.float64) * 2 * factor))


@pytest.mark.parametrize("h", [7, 14, 28])
def test_ramps_through_the_fusion_resizes_are_exact(h):
    """A (row, column, 1) ramp resized to 56 x 56: PyTorch float32 == mirror == written rule."""
    ramp = np.stack(list(np.meshgrid(np.arange(h), np.arange(h), indexing="ij")) + [np.ones((h, h))],
                    -1).astype(np.float32)
    torch32 = P.interpolate(ramp, (56, 56))
    mirror = M.resize(ramp, (56, 56))
    rule = np.array([[(l0 if j == i0 else 0) + (l1 if j == i1 else 0) for j in range(h)]
                     for (_, i0, i1, l0, l1) in P.bilinear_formula(h, 56)])
    written = np.einsum("yi,xj,ijc->yxc", rule, rule, ramp.astype(np.float64))
    assert np.array_equal(torch32, mirror) and np.array_equal(mirror, written)
    assert mirror[0, 0, 0] == 0 and mirror[-1, -1, 0] == h - 1          # edge clamp


def test_concat_order_in_the_mirror():
    parts = {R.RESIZED[n]: np.full((7, 7, 256), R.LEVEL_IDS[n]) + np.arange(256) for n in R.PYRAMID}
    assert np.array_equal(MU.concat(parts), np.broadcast_to(R.expected_ids(), (7, 7, 1024)))


def test_blocked_fusion_conv_equals_one_product(refs):
    x = refs["ee_mirror"]["fusion_concat"]
    w = S.decoder_parameters()[R.WEIGHT]
    one = MF.conv3x3(x, w, block_rows=10 ** 9)
    assert np.array_equal(MF.conv3x3(x, w), one)
    assert np.array_equal(MF.conv3x3(x, w, block_rows=56 * 3), one)


@pytest.mark.reference_environment
def test_sequential_accumulation_at_the_fusion_length():
    """torch's float64 matmul accumulates the 9,216-term dots strictly sequentially at the
    mirror's block shape (Milestone 5's finding re-checked for the fusion conv)."""
    assert MF.sequential_rows_agree(rows=3136, k=9216, cols=256)
    a, b = MU.cancellation_terms(9216)
    assert MU.sequential_dot(a, b) == 0.0                           # 2^53 absorbs both +1
    assert float((M.t64(a) @ M.t64(b)).item()) == 0.0


def test_explicit_sequential_dot_retains_cancellation_order():
    a, b = MU.cancellation_terms(9216)
    assert MU.sequential_dot(a, b) == 0.0
    # The fixture distinguishes reassociated accumulation from the explicit loop.
    assert MU.sequential_dot(a[:, :4608], b[:4608]) + MU.sequential_dot(a[:, 4608:], b[4608:]) == 1.0


def test_im2col_order_of_the_concatenation():
    """(ky, kx, c) over all 1,024 channels: tap t holds channels 1024 t .. 1024 t + 1023, P1 first."""
    ids = (np.arange(14 * 14)[:, None] * 1024 + np.arange(1024)[None] + 1).reshape(14, 14, 1024)
    cols = P.im2col(ids.astype(np.float64), 3)
    assert cols.shape == (14, 14, 9216)
    assert np.array_equal(cols[5, 5, 4 * 1024:5 * 1024], ids[5, 5])            # centre tap
    assert np.array_equal(cols[5, 5, 0:1024], ids[4, 4])                      # tap (0, 0)
    assert np.all(cols[0, 0, 0:3 * 1024] == 0) and np.all(cols[0, 0, 3 * 1024:4 * 1024] == 0)


def test_zero_padding_is_detectable_only_on_the_border(refs):
    """Phase 22: replicate / reflect padding differ from zero padding on every border region
    and equal it one pixel inside."""
    x = refs["ee_mirror"]["fusion_concat"]
    w = S.decoder_parameters()[R.WEIGHT]
    zero = MF.conv3x3(x, w)
    h = x.shape[0]
    border = np.ones((h, h), bool)
    border[1:-1, 1:-1] = False
    inner = np.zeros((h, h), bool)
    inner[1, 1:-1] = inner[-2, 1:-1] = inner[1:-1, 1] = inner[1:-1, -2] = True
    for mode in ("replicate", "reflect"):
        assert np.array_equal(MU.padded_variant_at(x, w, mode, inner), zero[inner])
        assert np.abs(MU.padded_variant_at(x, w, mode, border) - zero[border]).max() > 1e-2
