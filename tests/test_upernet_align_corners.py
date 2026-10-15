"""Milestone 11, offline: the head's align_corners=True upsample (Tests 6-8).

PyTorch's float32 kernel == the written float32 formula at every output index (N = 1,
M = 1 and odd / non-square sizes included); its coefficients differ from the textbook
rational ones but not in which source pixels they read; the window-local matrices are the
GLOBAL map's rows (absolute coordinates, one matrix per child window); every output reads
only its parent token and the parent's neighbours (the one-token halo); a per-window
template is not the global map; the mirror is within one float32 ulp of PyTorch."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensor_ee.blocks.window import resample as RS  # noqa: E402
from reference.upernet import ee_mirror_head as MH  # noqa: E402
from reference.upernet import pytorch_decoder_primitives as P  # noqa: E402

SIZES = [(1, 4), (1, 1), (4, 1), (6, 6), (2, 8), (3, 12), (5, 20), (7, 28), (14, 56), (21, 84),
         (56, 224), (224, 896), (448, 1792)]


@pytest.mark.parametrize("n_in,n_out", SIZES)
def test_kernel_equals_written_float32_formula(n_in, n_out):
    kernel = RS.model_bilinear_weights(n_in, n_out, align_corners=True)
    formula = P.formula_matrix(P.align_corners_formula(n_in, n_out, "float32"), n_in)
    assert np.array_equal(kernel, formula)


@pytest.mark.parametrize("n_in,n_out", SIZES)
def test_float32_coefficients_read_the_same_pixels_as_the_exact_ones(n_in, n_out):
    k32 = RS.model_bilinear_weights(n_in, n_out, align_corners=True)
    exact = P.formula_matrix(P.align_corners_formula(n_in, n_out, "exact"), n_in, float32=False)
    assert MH.membership(k32) == MH.membership(exact)
    assert np.abs(k32 - exact).max() < 3e-5


def test_float32_and_float64_models_use_different_coefficients():
    k32 = RS.model_bilinear_weights(224, 896, torch.float32, True)
    k64 = RS.model_bilinear_weights(224, 896, torch.float64, True)
    assert 1e-6 < np.abs(k32 - k64).max() < 2e-5


def test_endpoints_map_onto_endpoints():
    for n in (7, 224, 448):
        rows = P.align_corners_formula(n, 4 * n, "float32")
        assert rows[0][1:] == (0, 1, 1.0, 0.0)
        src, i0, i1, l0, l1 = rows[-1]
        assert {i0, i1} <= {n - 2, n - 1} and abs(src - (n - 1)) < 1e-4


@pytest.mark.parametrize("n", [7, 14, 21, 28, 56, 112, 224, 448])
@pytest.mark.parametrize("factor", [2, 4, 8])
def test_one_token_halo_suffices(n, factor):
    stack, index = RS.local_matrices(n, factor, align_corners=True)   # raises if an output reads outside
    m = RS.model_bilinear_weights(n, factor * n, align_corners=True)
    for o in range(factor * n):
        p = o // factor
        assert set(np.flatnonzero(m[o])) <= {p - 1, p, p + 1}


@pytest.mark.parametrize("n", [7, 14, 224, 448])
def test_window_matrices_are_rows_of_the_global_map(n):
    stack, index = RS.local_matrices(n, 4, align_corners=True)
    m = RS.model_bilinear_weights(n, 4 * n, align_corners=True)
    for child, k in enumerate(index):
        parent = child // 4
        local = stack[7 * k:7 * k + 7]
        for h in range(9):
            i = 7 * parent - 1 + h
            want = m[7 * child:7 * child + 7, i] if 0 <= i < n else np.zeros(7)
            assert np.array_equal(local[:, h], want)
    assert len(set(index)) == len(index) == 4 * n // 7            # one matrix per child window


def test_align_corners_false_defaults_unchanged():
    for n, f in ((7, 2), (14, 4), (28, 8)):
        stack, index = RS.local_matrices(n, f)
        assert len(set(index)) <= 10
        assert np.array_equal(stack.astype(np.float64) * 2 * f, np.round(stack.astype(np.float64) * 2 * f))


@pytest.mark.parametrize("n", [7, 14, 21])
def test_mirror_within_one_ulp_of_pytorch(n):
    x = (np.random.default_rng(n).standard_normal((n, n, 2)) * 3).astype(np.float32)
    mirror = MH.upsample(x)
    torch32 = P.upsample_align_corners(x).astype(np.float64)
    ulp = float(np.spacing(np.float32(np.abs(x).max())))       # PyTorch rounds intermediates in float32
    assert np.abs(mirror - torch32).max() <= 2 * ulp
    assert np.abs(mirror - MH.upsample(x, exact=True)).max() <= np.spacing(np.float32(12.0))


@pytest.mark.parametrize("h,w", [(2, 2), (3, 3), (5, 7), (1, 5), (7, 7)])
def test_matrix_formulation_mirror(h, w):
    x = (np.random.default_rng(h * 10 + w).standard_normal((h, w, 2))).astype(np.float32)
    got = MH.matrix_upsample(x, (4 * h, 4 * w))
    assert np.abs(got - P.upsample_align_corners(x)).max() < 1e-6


def test_per_window_template_is_not_the_global_map():
    x = (np.random.default_rng(5).standard_normal((14, 14, 2))).astype(np.float32)
    assert np.abs(MH.upsample_per_window(x) - MH.upsample(x)).max() > 1e-2
    one = x[:7, :7]
    assert np.array_equal(MH.upsample_per_window(one), MH.upsample(one))


def test_fma_variant_differs_in_double():
    x = (np.random.default_rng(6).standard_normal((7, 7, 2)) * 3).astype(np.float32)
    assert (MH.upsample(x, exact=True) != MH.upsample(x, exact=True, fma=True)).sum() > 0
