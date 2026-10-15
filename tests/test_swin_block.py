"""Offline checks for the Swin-block milestone (no Earth Engine calls)."""

import numpy as np
import pytest
import torch

from tensor_ee.core.precision import encode_float32
from reference.swin_block import pytorch_swin_block as ref
from reference.swin_block.ee_mirror import Mirror


@pytest.fixture(scope="module")
def parameters():
    pytest.importorskip("timm")
    return ref.make_parameters(seed=0)


@pytest.fixture(scope="module")
def raster():
    return ref.raster_input(14, 21)  # 2 x 3 windows


def test_parameters_are_deterministic_and_nontrivial(parameters):
    again = ref.make_parameters(seed=0)
    assert all(np.array_equal(parameters[k], again[k]) for k in parameters)
    assert all(v.dtype == np.float32 for v in parameters.values())
    for name in ("norm1.weight", "norm2.weight"):
        assert np.abs(parameters[name] - 1).max() > 0.1   # gamma is not 1
    for name in ("norm1.bias", "norm2.bias"):
        assert np.abs(parameters[name]).max() > 0.1       # beta is not 0
    assert parameters["attn.relative_position_bias_table"].std() > 0.5


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_explicit_forward_is_timm_on_a_raster(parameters, raster, dtype):
    # reference() raises unless the step-by-step output equals block(x) bit for bit.
    result = ref.reference(parameters, raster, dtype)
    assert result["out"].shape == (6, ref.TOKENS, ref.DIM)
    assert result["logits"].shape == (6, ref.HEADS, ref.TOKENS, ref.TOKENS)


def test_relative_position_index_matches_timm():
    pytest.importorskip("timm")
    from timm.models.swin_transformer import get_relative_position_index
    assert np.array_equal(ref.relative_position_index(7),
                          get_relative_position_index(7, 7).numpy())


def test_ee_index_and_one_hot_gather_arithmetic(parameters):
    """Float64 emulation of relative_position.py's server-side arithmetic."""
    window, n = 7, 49
    t = np.arange(n, dtype=np.float64)
    a, b = np.floor(t / window), np.mod(t, window)
    dy = a[:, None] - a[None, :] + window - 1
    dx = b[:, None] - b[None, :] + window - 1
    index = dy * (2 * window - 1) + dx
    assert np.array_equal(index, ref.relative_position_index(window))
    table = parameters["attn.relative_position_bias_table"].astype(np.float64)
    one_hot = (index.reshape(-1, 1) == np.arange(table.shape[0])[None, :]).astype(np.float64)
    gathered = (one_hot @ table).reshape(n, n, -1).transpose(2, 0, 1)
    expected = table[ref.relative_position_index(window)].transpose(2, 0, 1)
    assert np.array_equal(gathered, expected)


def test_window_layout_matches_timm_partition(raster):
    pytest.importorskip("timm")
    from timm.models.swin_transformer import window_partition
    windows = ref.to_window_layout(raster)
    timm = window_partition(torch.from_numpy(raster)[None], (7, 7)).reshape(-1, 49, ref.DIM)
    assert np.array_equal(windows, timm.numpy())
    assert np.array_equal(ref.from_window_layout(windows, 14, 21), raster)


def test_mirror_policy_is_closer_to_float64_than_pytorch(parameters, raster):
    windows = ref.to_window_layout(raster)
    mirror = Mirror(parameters).forward(windows)
    torch32 = ref.reference(parameters, raster, torch.float32)
    float64 = ref.reference(parameters, raster, torch.float64)
    for name in ref.TENSORS:
        gap_mirror = np.abs(mirror[name] - float64[name]).max()
        gap_torch = np.abs(torch32[name].astype(np.float64) - float64[name]).max()
        assert np.abs(mirror[name] - torch32[name]).max() < 1e-5, name
        assert gap_mirror <= gap_torch + 1e-12, name
    exact = Mirror(parameters, exact=True).forward(windows)
    for name in ref.TENSORS:
        np.testing.assert_allclose(exact[name], float64[name], rtol=0, atol=1e-12)


def test_float32_encoding_round_trips():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.standard_normal(5000) * 10.0 ** rng.integers(-8, 8, 5000),
                             [0.0, -0.0, 1.0, np.float32(np.pi), 3.4e38, 1.2e-38]]).astype(np.float32)
    encoded = np.asarray(encode_float32(values), dtype=np.float64)
    assert np.array_equal(encoded.astype(np.float32), values)
    # Shorter than the exact float64 expansion of the float32 values.
    assert len(repr(encoded.tolist())) < 0.75 * len(repr(values.astype(np.float64).tolist()))

