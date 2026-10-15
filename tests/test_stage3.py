"""Offline checks of Swin-T stage 3 (dim 384, 12 heads, MLP 1536, depth 6) (Milestone 5, no EE).

The isolated stage is timm's own SwinTransformerStage constructed exactly as
SwinTransformer constructs layers[2]; its blocks are checked against the full
model's, the explicit forward against timm's block outputs (inside
stage_reference), the attention against degeneracy, and the controlled-rounding
mirror policy at 1, 2 and 6 blocks.  The window-validity model shows why a
depth-6 stage needs a three-window ring (stage.stage_halo).
"""

import numpy as np
import pytest
import torch

from reference.swin_encoder import pytorch_swin_encoder as R
from reference.swin_encoder.ee_mirror_encoder import stage_mirror

S, ROWS, COLS = 3, 2, 3          # 2 x 3 stage-3 windows: all four mask types


@pytest.fixture(scope="module")
def parameters():
    pytest.importorskip("timm")
    return R.encoder_parameters()


@pytest.fixture(scope="module")
def tensors(parameters):
    x = R.stage_input(S, ROWS, COLS)
    return {"x": x, "torch32": R.stage_reference(parameters, S, x, torch.float32),
            "float64": R.stage_reference(parameters, S, x, torch.float64)}


def test_stage3_configuration_is_timm_swin_t():
    pytest.importorskip("timm")
    model = R.make_model((112, 112))
    stage = model.layers[2]
    alone = R.make_stage(3, 4, 4)
    for blocks in (stage.blocks, alone.blocks):
        assert len(blocks) == 6
        assert [b.shift_size for b in blocks] == [(0, 0), (3, 3)] * 3
        assert {b.attn.num_heads for b in blocks} == {12}
        assert {b.mlp.fc1.out_features for b in blocks} == {1536}
        assert {b.norm1.normalized_shape for b in blocks} == {(384,)}
        assert not any(b.attn.fused_attn for b in blocks)
    assert stage.blocks[1].attn_mask.shape == alone.blocks[1].attn_mask.shape == (16, 49, 49)
    assert set(alone.state_dict()) == set(stage.state_dict())


def test_isolated_stage_equals_full_model_stage(parameters):
    """The standalone stage gives timm's layers[2].blocks output bit for bit."""
    model = R.make_model((112, 112), parameters)
    alone = R.make_stage(3, 4, 4, parameters)
    x = torch.from_numpy(R.stage_input(3, 4, 4))[None]
    with torch.no_grad():
        assert torch.equal(alone.blocks(x), model.layers[2].blocks(x))
        assert torch.equal(alone.downsample.reduction.weight, model.layers[2].downsample.reduction.weight)


def test_stage3_shapes_and_attention_not_degenerate(tensors):
    t32, f64 = tensors["torch32"], tensors["float64"]
    assert t32["b0_qkv"].shape == (6, 49, 1152)
    assert t32["b1_logits"].shape == (6, 12, 49, 49) and t32["b1_mask"].shape == (6, 49, 49)
    assert t32["b5_fc1"].shape == (6, 49, 1536)
    for b in range(6):
        logits = f64[f"b{b}_logits_biased" if b % 2 else f"b{b}_logits"]
        assert 1.5 < logits.std() < 4, b
        probs = f64[f"b{b}_probs"]
        entropy = -(np.where(probs > 0, probs * np.log(np.where(probs > 0, probs, 1)), 0)).sum(-1)
        assert 1.0 < entropy.mean() < np.log(49) - 0.3, b
    assert np.array_equal(t32["b1_x"], t32["b0_out"])


@pytest.mark.parametrize("blocks", [1, 2, 6])
def test_stage3_mirror_policy(parameters, tensors, blocks):
    windows = R.to_window_layout(tensors["x"])
    _, mirror = stage_mirror(parameters, S, windows, ROWS, COLS, blocks=blocks)
    _, exact = stage_mirror(parameters, S, windows, ROWS, COLS, exact=True, blocks=blocks)
    t32, f64 = tensors["torch32"], tensors["float64"]
    for name in mirror:
        gap_mirror = np.abs(mirror[name] - f64[name]).max()
        gap_torch = np.abs(t32[name].astype(np.float64) - f64[name]).max()
        slack = 4e-6 if name.endswith("_logits") else 1e-12     # masked logits ~ -100
        assert gap_mirror <= gap_torch + slack, name
        assert np.abs(mirror[name] - t32[name]).max() < 3e-5, name
        np.testing.assert_allclose(exact[name], f64[name], rtol=0, atol=1e-11, err_msg=name)
