"""Offline checks of the Milestone 4 hierarchy references, mirror and weight records
(no Earth Engine calls)."""

import numpy as np
import pytest
import torch

from reference.swin_hierarchy import pytorch_hierarchy as ref
from reference.swin_hierarchy.ee_mirror_hierarchy import hierarchy_mirror

ROWS, COLS = 4, 6          # stage 2: 2 x 3 windows, all four mask types


@pytest.fixture(scope="module")
def parameters():
    pytest.importorskip("timm")
    return ref.hierarchy_parameters()


@pytest.fixture(scope="module")
def tensors():
    pytest.importorskip("timm")
    return ref.references(ROWS, COLS)


def test_model_is_timm_swin_t_stages_1_and_2():
    pytest.importorskip("timm")
    model = ref.make_model((28, 42))
    s1, s2 = model.layers
    assert [b.attn.num_heads for b in s1.blocks] == [3, 3]
    assert [b.attn.num_heads for b in s2.blocks] == [6, 6]
    assert [b.shift_size for b in s2.blocks] == [(0, 0), (3, 3)]
    assert s2.blocks[0].mlp.fc1.out_features == 768
    assert s2.blocks[1].attn_mask.shape == (6, 49, 49)
    assert s2.input_resolution == (28, 42) and s2.output_resolution == (14, 21)
    assert not any(b.attn.fused_attn for s in model.layers for b in s.blocks)


@pytest.mark.parametrize("bad", [(21, 28), (28, 35), (14, 28)])
def test_odd_or_single_window_stage2_rejected(bad):
    with pytest.raises(ValueError):
        ref.check_resolution(*bad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_explicit_hierarchy_is_timm(parameters, dtype):
    # hierarchy_reference raises unless stage 1, Patch Merging, stage 2 and
    # model.layers are bit-identical to timm's modules.
    x = ref.raster_input(28, 28)
    out = ref.hierarchy_reference(parameters, x, dtype)
    assert out["s1b_out"].shape == (16, 49, 96)
    assert out["pm_cat"].shape == (4, 49, 384) and out["pm_out"].shape == (4, 49, 192)
    assert out["s2a_qkv"].shape == (4, 49, 576) and out["s2b_logits"].shape == (4, 6, 49, 49)
    assert np.array_equal(out["s2b_x"], out["s2a_out"])


def test_stage2_attention_is_not_degenerate(tensors):
    f64 = tensors["float64"]
    assert 1.5 < f64["s2a_logits"].std() < 4
    probs = f64["s2a_probs"]
    entropy = -(probs * np.log(probs)).sum(-1)
    assert 1.0 < entropy.mean() < np.log(49) - 0.5


def test_mirror_policy(parameters, tensors):
    x = ref.to_window_layout(ref.raster_input(ROWS * 7, COLS * 7))
    mirror = hierarchy_mirror(parameters, x, ROWS, COLS)
    exact = hierarchy_mirror(parameters, x, ROWS, COLS, exact=True)
    torch32, float64 = tensors["torch32"], tensors["float64"]
    assert set(mirror) == set(torch32)
    for name in torch32:
        assert np.abs(mirror[name] - torch32[name]).max() < 2e-5, name
        gap_mirror = np.abs(mirror[name] - float64[name]).max()
        gap_torch = np.abs(torch32[name].astype(np.float64) - float64[name]).max()
        # Masked logits are ~ -100 (float32 ulp 7.6e-6): rounding them once can
        # land farther from float64 than PyTorch does at an element.
        slack = 4e-6 if name.endswith("b_logits") else 1e-12
        assert gap_mirror <= gap_torch + slack, name
        np.testing.assert_allclose(exact[name], float64[name], rtol=0, atol=1e-12, err_msg=name)


def test_layout_tensors_are_exact_in_mirror(parameters, tensors):
    x = ref.to_window_layout(ref.raster_input(ROWS * 7, COLS * 7))
    mirror = hierarchy_mirror(parameters, x, ROWS, COLS)
    assert np.array_equal(mirror["pm_cat"], ref.gather_windows(mirror["s1b_out"], ROWS, COLS))
    for g in range(4):
        assert np.array_equal(mirror[f"pm_x{g}"], mirror["pm_cat"][..., 96 * g:96 * (g + 1)])
