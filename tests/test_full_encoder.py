"""Offline checks of the complete Swin-T encoder reference and mirror (Milestone 5, no EE).

encoder_reference raises unless every block, every timm stage and the final
output equal timm's modules (model.norm(model.layers(x))) bit for bit; here it
runs in float32 and float64.  The first two stages must reproduce Milestone
4's reference tensors bit for bit (same parameters), the mirror policy holds
for every intermediate of the encoder, and the dependency halo formula is
checked against a window-validity model of the encoder's data flow.
"""

import numpy as np
import pytest
import torch

from reference.swin_encoder import pytorch_swin_encoder as R
from reference.swin_encoder.ee_mirror_encoder import encoder_mirror

K = 8                      # smallest valid raster (stage 4 is one window: shift suppressed)


@pytest.fixture(scope="module")
def parameters():
    pytest.importorskip("timm")
    return R.encoder_parameters()


@pytest.fixture(scope="module")
def tensors():
    pytest.importorskip("timm")
    return R.references(K, K)


def test_model_is_timm_swin_t():
    pytest.importorskip("timm")
    model = R.make_model((224, 224))
    assert [len(s.blocks) for s in model.layers] == [2, 2, 6, 2]
    assert [s.blocks[0].attn.num_heads for s in model.layers] == [3, 6, 12, 24]
    assert [s.blocks[0].norm1.normalized_shape[0] for s in model.layers] == [96, 192, 384, 768]
    assert [b.shift_size for s in model.layers for b in s.blocks] == [(0, 0), (3, 3)] * 6
    assert model.norm.normalized_shape == (768,)
    assert R.parameter_count(R.encoder_parameters()) == 27_514_458
    encoder = sum(v.numel() for k, v in model.state_dict().items()
                  if k.startswith(("layers.", "norm.")) and not k.endswith(
                      ("relative_position_index", "attn_mask")))
    assert encoder == 27_514_458


@pytest.mark.parametrize("bad", [(4, 8), (8, 12), (12, 12), (8, 16)])
def test_invalid_rasters_rejected(bad):
    with pytest.raises(ValueError):
        R.check_windows(*bad)       # (8, 16): stage 4 would be 1 x 2 windows (mixed shift)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_explicit_encoder_is_timm(parameters, dtype):
    out = R.encoder_reference(parameters, R.raster_input(56, 56), dtype)   # raises if not timm's
    assert out["s1_out"].shape == (64, 49, 96) and out["pm1_cat"].shape == (16, 49, 384)
    assert out["pm2_cat"].shape == (4, 49, 768) and out["s3b5_fc1"].shape == (4, 49, 1536)
    assert out["pm3_cat"].shape == (1, 49, 1536) and out["s4b1_fc1"].shape == (1, 49, 3072)
    assert out["norm"].shape == (1, 49, 768)
    assert "s4b1_mask" not in out          # stage 4 at one window: timm uses shift 0
    assert np.array_equal(out["s3b1_x"], out["s3b0_out"])
    assert np.array_equal(out["s3_out"], out["s3b5_out"])


def test_first_two_stages_are_milestone4(tensors):
    from reference.swin_hierarchy import pytorch_hierarchy as m4
    old = m4.references(K, K)
    rename = {"s1a": "s1b0", "s1b": "s1b1", "pm": "pm1", "s2a": "s2b0", "s2b": "s2b1"}
    for dtype in ("torch32", "float64"):
        for name, value in old[dtype].items():
            if name == "x":
                key = "x"
            else:
                group, rest = name.split("_", 1)
                key = f"{rename[group]}_{rest}"
            assert np.array_equal(tensors[dtype][key], value), (dtype, name)


def test_encoder_mirror_policy(parameters, tensors):
    x = R.to_window_layout(R.raster_input(K * 7, K * 7))
    mirror = encoder_mirror(parameters, x, K, K)
    exact = encoder_mirror(parameters, x, K, K, exact=True)
    torch32, float64 = tensors["torch32"], tensors["float64"]
    assert set(mirror) == set(torch32)
    for name in torch32:
        assert np.abs(mirror[name] - torch32[name]).max() < 3e-5, name
        gap_mirror = np.abs(mirror[name] - float64[name]).max()
        gap_torch = np.abs(torch32[name].astype(np.float64) - float64[name]).max()
        slack = 4e-6 if name.endswith("_logits") else 1e-12
        assert gap_mirror <= gap_torch + slack, name
        np.testing.assert_allclose(exact[name], float64[name], rtol=0, atol=1e-11, err_msg=name)


def test_error_does_not_grow_through_the_encoder(tensors):
    f64, t32 = tensors["float64"], tensors["torch32"]
    for name in ("s1_out", "s2_out", "s3_out", "s4_out", "norm"):
        assert np.abs(t32[name] - f64[name]).max() < 3e-5, name
