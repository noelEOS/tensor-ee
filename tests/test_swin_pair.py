"""Offline checks for the W-MSA + SW-MSA pair milestone (no Earth Engine calls)."""

import numpy as np
import pytest
import torch

from reference.swin_pair import pytorch_shifted_swin as ref
from reference.swin_pair.ee_mirror_shifted import (
    mask_numpy, pair_mirror_forward, shifted_mirror_forward)

RASTERS = [(2, 2), (2, 3), (3, 5), (4, 4)]


@pytest.fixture(scope="module")
def parameters():
    pytest.importorskip("timm")
    return ref.pair_parameters()


@pytest.fixture(scope="module")
def raster():
    return ref.raster_input(14, 21)  # 2 x 3 windows: all four mask types


def test_block_b_parameters_differ_from_block_a(parameters):
    a, b = parameters
    assert set(a) == set(b)
    assert all(not np.array_equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_explicit_shifted_forward_is_timm(parameters, raster, dtype):
    # shifted_reference() raises unless the step-by-step output equals block(x) bit for bit.
    result = ref.shifted_reference(parameters[1], raster, dtype)
    assert result["out"].shape == (6, ref.TOKENS, ref.DIM)
    assert result["logits"].shape == (6, ref.HEADS, ref.TOKENS, ref.TOKENS)
    assert result["mask"].shape == (6, ref.TOKENS, ref.TOKENS)
    assert np.array_equal(result["reverse"], result["proj"])  # window_reverse is the identity


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_explicit_pair_is_timm_stage(parameters, raster, dtype):
    # pair_reference() raises unless it equals SwinTransformerStage(depth=2)(x) bit for bit.
    result = ref.pair_reference(*parameters, raster, dtype)
    assert np.array_equal(result["b_x"], result["a_out"])


def test_stage_is_w_msa_then_sw_msa(parameters):
    stage = ref.make_stage((14, 21), *parameters)
    assert stage.blocks[0].shift_size == (0, 0) and stage.blocks[1].shift_size == (3, 3)
    assert stage.blocks[0].attn_mask is None and stage.blocks[1].attn_mask.shape == (6, 49, 49)
    assert not stage.blocks[1].attn.fused_attn


def test_timm_uses_no_shift_for_a_single_window():
    pytest.importorskip("timm")
    with pytest.raises(ValueError):
        ref.make_shifted_block((7, 7))
    from timm.models.swin_transformer import SwinTransformerBlock
    assert SwinTransformerBlock(96, (7, 7), 3, window_size=7, shift_size=3).shift_size == (0, 0)


@pytest.mark.parametrize("rows,cols", RASTERS)
def test_numpy_shift_is_torch_roll_and_partition(rows, cols):
    pytest.importorskip("timm")
    from timm.models.swin_transformer import window_partition
    ids = ref.token_ids(rows, cols)
    rolled = torch.roll(torch.from_numpy(ids)[None], (-3, -3), (1, 2))
    timm = window_partition(rolled, (7, 7)).reshape(-1, 49, 3).numpy()
    windows = ref.to_window_layout(ids)
    assert np.array_equal(ref.shift_windows(windows, rows, cols), timm)
    assert np.array_equal(ref.unshift_windows(timm, rows, cols), windows)


@pytest.mark.parametrize("rows,cols", RASTERS + [(32, 32)])
def test_mask_formula_is_timm_mask(rows, cols):
    pytest.importorskip("timm")
    expected = ref.timm_mask(rows, cols)
    got = mask_numpy(rows, cols)
    assert np.array_equal(got, expected)
    assert np.array_equal(np.signbit(got), np.signbit(expected))  # no -0.0
    assert set(np.unique(expected)) == {0.0, -100.0}


def _emulate_roll_axis(windows: np.ndarray, offset: int, axis: str) -> np.ndarray:
    """NumPy emulation of shift._roll_axis on [kh, kw, 49, C] window arrays:
    the same slabs, slices, concatenations and reshapes, with the periodic
    neighbor (I + step mod kh) as the coarse translation."""
    w, c = 7, windows.shape[-1]
    step = 1 if offset > 0 else -1
    neighbor = np.roll(windows, -step, axis=0 if axis == "y" else 1)
    if offset > 0:
        parts = ((windows, offset, w), (neighbor, 0, offset))
    else:
        parts = ((neighbor, w + offset, w), (windows, 0, w + offset))
    if axis == "y":
        return np.concatenate([image[..., p * w:q * w, :] for image, p, q in parts], axis=-2)
    shape = windows.shape[:2]
    wide = [image.reshape(*shape, w, w * c)[..., p * c:q * c] for image, p, q in parts]
    return np.concatenate(wide, axis=-1).reshape(*shape, w * w, c)


@pytest.mark.parametrize("rows,cols", RASTERS)
def test_window_layout_shift_algorithm(rows, cols):
    ids = ref.token_ids(rows, cols)
    windows = ref.to_window_layout(ids).reshape(rows, cols, 49, 3)
    shifted = _emulate_roll_axis(_emulate_roll_axis(windows, 3, "y"), 3, "x")
    assert np.array_equal(shifted.reshape(-1, 49, 3),
                          ref.shift_windows(ref.to_window_layout(ids), rows, cols))
    back = _emulate_roll_axis(_emulate_roll_axis(shifted, -3, "y"), -3, "x")
    assert np.array_equal(back, windows)


def test_shifted_window_draws_from_four_source_windows():
    rows = cols = 3
    ids = ref.token_ids(rows, cols)
    shifted = ref.shift_windows(ref.to_window_layout(ids), rows, cols)
    y, x = shifted[4, :, 1], shifted[4, :, 2]          # interior window (1, 1)
    sources = {(int(a) // 7, int(b) // 7) for a, b in zip(y, x)}
    assert sources == {(1, 1), (1, 2), (2, 1), (2, 2)}


def test_mirror_policy_on_the_pair(parameters, raster):
    windows = ref.to_window_layout(raster)
    mirror = pair_mirror_forward(*parameters, windows, 2, 3)
    torch32 = ref.pair_reference(*parameters, raster, torch.float32)
    float64 = ref.pair_reference(*parameters, raster, torch.float64)
    for name in torch32:
        gap_mirror = np.abs(mirror[name] - float64[name]).max()
        gap_torch = np.abs(torch32[name].astype(np.float64) - float64[name]).max()
        assert np.abs(mirror[name] - torch32[name]).max() < 2e-5, name
        # Masked logits are ~ -100 (float32 ulp 7.6e-6), so rounding them once
        # can land farther from float64 than PyTorch does at an element.
        slack = 4e-6 if name == "b_logits" else 1e-12
        assert gap_mirror <= gap_torch + slack, name
    exact = pair_mirror_forward(*parameters, windows, 2, 3, exact=True)
    for name in torch32:
        np.testing.assert_allclose(exact[name], float64[name], rtol=0, atol=1e-12)


def test_masked_probabilities_are_float32_denormal_or_zero(parameters, raster):
    mirror = shifted_mirror_forward(parameters[1], ref.to_window_layout(raster), 2, 3)
    masked = np.broadcast_to((mirror["mask"] != 0)[:, None], mirror["probs"].shape)
    values = mirror["probs"][masked]
    # Below the smallest normal float32: compared bit for bit with EE, which
    # keeps float32 denormals (no flush to zero; probed live).
    assert 0 < values.max() < np.finfo(np.float32).tiny
