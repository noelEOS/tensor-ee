"""Offline checks of Swin-T stage 4 (dim 768, 24 heads, MLP 3072, depth 2) (Milestone 5, no EE).

As test_stage3 for the widest stage: timm configuration, the standalone
stage against the full model, attention statistics at 24 heads, the mirror
policy, and timm's shift suppression when stage 4 is a single window.
"""

import numpy as np
import pytest
import torch

from reference.swin_encoder import pytorch_swin_encoder as R
from reference.swin_encoder.ee_mirror_encoder import norm_mirror, stage_mirror

S, ROWS, COLS = 4, 2, 2          # 2 x 2 stage-4 windows: all four mask types


@pytest.fixture(scope="module")
def parameters():
    pytest.importorskip("timm")
    return R.encoder_parameters()


@pytest.fixture(scope="module")
def tensors(parameters):
    x = R.stage_input(S, ROWS, COLS)
    return {"x": x, "torch32": R.stage_reference(parameters, S, x, torch.float32),
            "float64": R.stage_reference(parameters, S, x, torch.float64)}


def test_stage4_configuration_is_timm_swin_t():
    pytest.importorskip("timm")
    stage = R.make_model((112, 112)).layers[3]            # 2 x 2 stage-4 windows
    alone = R.make_stage(4, 2, 2)
    for blocks in (stage.blocks, alone.blocks):
        assert [b.shift_size for b in blocks] == [(0, 0), (3, 3)]
        assert {b.attn.num_heads for b in blocks} == {24}
        assert {b.mlp.fc1.out_features for b in blocks} == {3072}
        assert {b.attn.qkv.out_features for b in blocks} == {2304}
    assert set(alone.state_dict()) == set(stage.state_dict())


def test_stage4_shift_suppressed_for_a_single_window():
    """timm: stage resolution 7 <= window 7 -> shift 0 (8 x 8 stage-1 windows)."""
    pytest.importorskip("timm")
    stage = R.make_model((56, 56)).layers[3]
    assert [b.shift_size for b in stage.blocks] == [(0, 0), (0, 0)]
    assert [b.window_size for b in stage.blocks] == [(7, 7), (7, 7)]
    assert stage.blocks[1].attn_mask is None
    assert R.block_shifts(8, 8)[3] == [0, 0] and R.block_shifts(16, 16)[3] == [0, 3]
    alone = R.make_stage(4, 1, 1)
    assert [b.shift_size for b in alone.blocks] == [(0, 0), (0, 0)]


def test_isolated_stage_equals_full_model_stage(parameters):
    model = R.make_model((112, 112), parameters)
    alone = R.make_stage(4, 2, 2, parameters)
    x = torch.from_numpy(R.stage_input(4, 2, 2))[None]
    with torch.no_grad():
        assert torch.equal(alone.blocks(x), model.layers[3].blocks(x))


def test_stage4_shapes_and_attention_not_degenerate(tensors):
    t32, f64 = tensors["torch32"], tensors["float64"]
    assert t32["b1_logits"].shape == (4, 24, 49, 49) and t32["b1_fc1"].shape == (4, 49, 3072)
    for b in range(2):
        logits = f64[f"b{b}_logits_biased" if b % 2 else f"b{b}_logits"]
        assert 1.5 < logits.std() < 4, b
        probs = f64[f"b{b}_probs"]
        entropy = -(np.where(probs > 0, probs * np.log(np.where(probs > 0, probs, 1)), 0)).sum(-1)
        assert 1.0 < entropy.mean() < np.log(49) - 0.3, b


def test_stage4_mirror_policy(parameters, tensors):
    windows = R.to_window_layout(tensors["x"])
    _, mirror = stage_mirror(parameters, S, windows, ROWS, COLS)
    _, exact = stage_mirror(parameters, S, windows, ROWS, COLS, exact=True)
    t32, f64 = tensors["torch32"], tensors["float64"]
    for name in mirror:
        gap_mirror = np.abs(mirror[name] - f64[name]).max()
        gap_torch = np.abs(t32[name].astype(np.float64) - f64[name]).max()
        slack = 4e-6 if name.endswith("_logits") else 1e-12
        assert gap_mirror <= gap_torch + slack, name
        assert np.abs(mirror[name] - t32[name]).max() < 3e-5, name
        np.testing.assert_allclose(exact[name], f64[name], rtol=0, atol=1e-11, err_msg=name)


def test_final_norm_reference_and_mirror(parameters):
    x = R.to_window_layout(R.stage_input(4, 2, 2))
    t32 = R.norm_reference(parameters, x, torch.float32)
    f64 = R.norm_reference(parameters, x, torch.float64)
    mirror = norm_mirror(parameters, x)
    assert t32.shape == (4, 49, 768)
    assert np.abs(mirror - f64).max() <= np.abs(t32.astype(np.float64) - f64).max()
    np.testing.assert_allclose(norm_mirror(parameters, x, exact=True), f64, rtol=0, atol=1e-12)
    model = R.make_model((56, 56), parameters)
    with torch.no_grad():
        assert np.array_equal(model.norm(torch.from_numpy(x)).numpy(), t32)
