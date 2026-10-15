"""Milestone 7, offline: PyTorch semantics of the decoder primitives and the EE mirror
(Tests 8-16 offline halves): conv weight order, padding, BatchNorm folding, ReLU, bilinear
weights and their window-local slices, addition, concatenation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("segmentation_models_pytorch")

from tensor_ee.blocks.window import resample as RS  # noqa: E402
from reference.upernet import ee_mirror_decoder as M  # noqa: E402
from reference.upernet import pytorch_decoder_primitives as P  # noqa: E402
from reference.upernet.helpers import fold, pixel_ids_numpy  # noqa: E402


def floats(shape, seed, scale=1.0):
    return (np.random.default_rng(seed).standard_normal(shape) * scale).astype(np.float32)


def test_conv_is_im2col_in_ky_kx_c_order():
    """Test 8 offline: one-hot weights extract exactly im2col (zero padding at the border)."""
    x = pixel_ids_numpy(5, 6, 2, offset=1)
    weight = np.zeros((18, 2, 3, 3), np.float32)
    for t in range(9):
        for c in range(2):
            weight[t * 2 + c, c, t // 3, t % 3] = 1
    assert np.array_equal(P.conv2d(x, weight, dtype=torch.float64), P.im2col(x))
    assert (P.im2col(x)[0, 0, :2] == 0).all()                       # top-left tap is padding


def test_conv_weight_rows_match_im2col():
    x, w, b = floats((6, 7, 4), 1), floats((5, 4, 3, 3), 2), floats((5,), 3)
    via_rows = P.im2col(x).astype(np.float64) @ P.conv_weight_rows(w).astype(np.float64) + b
    assert np.allclose(via_rows, P.conv2d(x, w, b, torch.float64), atol=1e-12)


@pytest.mark.parametrize("method", ["window_im2col", "window_taps", "dense_nbr", "convolve"])
def test_mirror_conv_within_float32_of_float64(method):
    x, w = floats((8, 9, 6), 4), floats((7, 6, 3, 3), 5, 0.3)
    m = M.conv3x3(x, w, None, method)
    exact = P.conv2d(x, w, dtype=torch.float64)
    assert np.abs(m - exact).max() <= 2 ** -23 * np.abs(exact).max() * 2


def test_batchnorm_folding_close_but_not_bitwise():
    """Test 11 offline: folding is equivalent to float32 rounding, not bit for bit."""
    x, w = floats((10, 10, 16), 6), floats((16, 16, 3, 3), 7, 0.1)
    gamma, beta = 1 + floats((16,), 8, 0.2), floats((16,), 9, 0.2)
    mean, var = floats((16,), 10, 0.2), 1 + floats((16,), 11, 0.1)
    explicit = P.conv_block(x, w, gamma, beta, mean, var, torch.float64)["bn"]
    wf, bf = fold(w, gamma, beta, mean, var)
    folded = P.conv2d(x, wf, bf, torch.float64)
    assert np.abs(folded - explicit).max() < 1e-5
    assert not np.array_equal(M.conv3x3(x, wf, bf),
                              M.batch_norm(M.conv3x3(x, w), gamma, beta, mean, var))


@pytest.mark.reference_environment
def test_relu_signed_zero_depends_on_pytorchs_vector_path():
    """PyTorch's ReLU maps -0.0 to +0.0 in its 8-wide vectorized loop and keeps -0.0 in the
    scalar tail (EE's max is always +0.0: equal as numbers); NaN propagates; negatives -> +0."""
    tail = P.relu(np.full((1, 5, 1), -0.0, np.float32)).view(np.uint32).ravel()
    vector = P.relu(np.full((1, 16, 1), -0.0, np.float32)).view(np.uint32).ravel()
    assert set(tail) == {0x80000000} and set(vector) == {0}


def test_relu_values_signed_zero_and_nan():
    """Zero signs may vary with the CPU kernel; numerical ReLU semantics do not."""
    for width in (5, 16):
        out = P.relu(np.full((1, width, 1), -0.0, np.float32))
        assert out.dtype == np.float32 and np.array_equal(out, np.zeros_like(out))
    out = P.relu(np.array([-1.0, 2.0, np.nan], np.float32).reshape(1, 3, 1)).ravel()
    assert out[0] == 0 and out[1] == 2 and np.isnan(out[2])


@pytest.mark.parametrize("n_in,n_out", [(2, 4), (3, 6), (5, 10), (3, 7), (6, 28), (7, 56),
                                        (28, 224)])
def test_bilinear_weights_are_pytorchs_formula(n_in, n_out):
    m = P.bilinear_weights(n_in, n_out)
    f = np.zeros_like(m)
    for o, (src, i0, i1, l0, l1) in enumerate(P.bilinear_formula(n_in, n_out)):
        f[o, i0] += l0
        f[o, i1] += l1
    assert np.abs(m - f).max() < 1e-15
    assert np.allclose(m.sum(1), 1)


@pytest.mark.parametrize("n,f", [(7, 2), (14, 2), (21, 2), (28, 4), (28, 8), (56, 2), (7, 8)])
def test_window_local_matrices_reassemble_pytorchs(n, f):
    """Each child window's [7, 9] block, placed at its parent's halo columns, rebuilds
    PyTorch's global matrix exactly (edge clamp included)."""
    stack, index = RS.local_matrices(n, f)
    rebuilt = np.zeros((n * f, n + 2))
    for child, k in enumerate(index):
        parent = child // f
        rebuilt[7 * child: 7 * child + 7, 7 * parent: 7 * parent + 9] += stack[7 * k: 7 * k + 7]
    assert np.array_equal(rebuilt[:, 1:-1], P.bilinear_weights(n, n * f))
    assert not rebuilt[:, [0, -1]].any()


@pytest.mark.parametrize("shape,size", [((7, 14, 3), (14, 28)), ((5, 5, 2), (10, 10)),
                                        ((3, 5, 2), (12, 20))])
def test_mirror_resize_is_exact_bilinear_rounded_once(shape, size):
    x = floats(shape, 12)
    exact = P.interpolate(x, size, torch.float64)
    assert np.array_equal(M.resize(x, size), exact.astype(np.float32).astype(np.float64))
    assert np.abs(M.resize(x, size) - P.interpolate(x, size)).max() <= 4e-7 * np.abs(x).max()


def test_ramp_gives_source_coordinates():
    i, j = np.meshgrid(np.arange(3), np.arange(5), indexing="ij")
    x = np.stack([i, j], -1).astype(np.float32)
    out = P.interpolate(x, (6, 10), torch.float64)
    ys = [min(max(s, 0), 2) for s, *_ in P.bilinear_formula(3, 6)]
    assert np.array_equal(out[:, 0, 0], ys)


def test_add_concat():
    a, b = floats((3, 4, 5), 13), floats((3, 4, 6), 14)
    assert np.array_equal(P.concat([a, b]), np.concatenate([a, b], -1))
    assert np.array_equal(P.add(a, a), a + a)
    assert np.array_equal(M.add(a, a), (a + a).astype(np.float64))


def test_mirror_layernorm2d_matches_smp_float64():
    x, g, b = floats((7, 7, 96), 15, 2), 1 + floats((96,), 16, 0.2), floats((96,), 17, 0.2)
    m = M.layer_norm2d(x, g, b, exact=True)
    assert np.abs(m - P.layer_norm2d(x, g, b, dtype=torch.float64)).max() < 1e-12


@pytest.mark.reference_environment
def test_sequential_matmul_assumption_at_decoder_lengths():
    assert M.sequential_matmul_agrees(rows=32, k=2304, cols=8)
    assert M.sequential_matmul_agrees(rows=8, k=9216, cols=4, seed=1)
