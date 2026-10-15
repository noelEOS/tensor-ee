"""Milestone 7, offline: the installed SMP UPerNet's encoder -> decoder contract
(Tests 1-2, Phase 1-2, 6), established by running the installed code."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("segmentation_models_pytorch")

from reference.swin_block.pytorch_swin_block import from_window_layout  # noqa: E402
from reference.swin_encoder import pytorch_swin_encoder as m5  # noqa: E402
from reference.swin_image_encoder import pytorch_image_encoder as m6  # noqa: E402
from reference.upernet import smp_feature_interface as S  # noqa: E402


@pytest.fixture(scope="module")
def contract():
    return S.contract(224, 224)


def test_installed_versions(contract):
    assert contract["versions"]["segmentation_models_pytorch"] == "0.5.0"
    assert contract["versions"]["timm"] == "1.0.30"


def test_feature_sources_are_the_four_stage_outputs(contract):
    """Test 1: C1-C4 are timm's layers_0..3 outputs = Milestone 5's s1_out..s4_out, bit for bit."""
    for k, name in enumerate(S.FEATURES):
        f = contract["features"][name]
        assert f["feature_info"]["module"] == f"layers.{k}"
        assert f["equals_stage_output_bitwise"]
        assert f["dtype"] == "torch.float32" and f["is_contiguous_nchw"]


def test_c4_is_raw_stage4_not_the_final_norm(contract):
    assert not contract["model"]["feature_net_has_final_norm"]
    assert not contract["features"]["C4"]["equals_final_encoder_norm"]
    assert contract["features"]["C4"]["max_abs_difference_to_final_norm"] > 1


def test_shapes_and_channels(contract):
    """Test 2: (B, C, H/s, W/s) for s = 4, 8, 16, 32 and C = 96, 192, 384, 768."""
    for k, name in enumerate(S.FEATURES):
        shape = contract["features"][name]["shape_nchw"]
        assert shape == [1, S.FEATURE_CHANNELS[k], 224 // S.FEATURE_STRIDES[k],
                         224 // S.FEATURE_STRIDES[k]]
    assert contract["encoder_list"][:2] == [[1, 2, 224, 224], [1, 0, 112, 112]]
    assert contract["model"]["encoder_out_channels"] == [2, 0, 96, 192, 384, 768]


def test_decoder_configuration(contract):
    model = contract["model"]
    assert model["decoder_normalization_classes"] == ["BatchNorm2d", "LayerNorm2d"]
    assert all(n["class"] == "LayerNorm2d" and n["eps"] == 1e-6 and n["elementwise_affine"]
               for n in model["feature_norms"])
    for name, block in model["conv_blocks"].items():
        assert block["norm"] == "BatchNorm2d" and not block["bias"] and block["activation"] == "ReLU"
        assert block["padding"] == [block["kernel"][0] // 2] * 2 and block["stride"] == [1, 1]
    assert model["psp_sizes"] == [1, 2, 3, 6]
    assert model["segmentation_head"]["upsampling_align_corners"] is True


def test_every_decoder_resize_is_bilinear_align_corners_false(contract):
    decoder = [c for c in contract["interpolations"] if c["caller"].startswith("decoder.py")]
    assert len(decoder) == 4 + 3 + 4
    assert all(c["mode"] == "bilinear" and c["align_corners"] is False and c["size"] is not None
               for c in decoder)
    head = [c for c in contract["interpolations"] if not c["caller"].startswith("decoder.py")]
    assert len(head) == 1 and head[0]["align_corners"] is True and head[0]["scale_factor"] == 4.0


def test_prepared_features_are_manual_layernorm2d():
    """Phase 2: the hooks' prepared tensors equal decoder.feature_norms applied by hand (the
    function raises otherwise) and equal F.layer_norm over channels with eps 1e-6."""
    model = S.make_upernet(224, 224)
    x = torch.from_numpy(np.random.default_rng(3).standard_normal((1, 2, 224, 224)).astype(np.float32))
    result = S.decoder_interface(model, x)
    for k, norm in enumerate(model.decoder.feature_norms):
        raw = result["raw"][k].permute(0, 2, 3, 1)
        expected = torch.nn.functional.layer_norm(raw, norm.normalized_shape, norm.weight,
                                                  norm.bias, 1e-6).permute(0, 3, 1, 2)
        assert torch.equal(result["prepared"][k], expected)


def test_encoder_parameters_are_milestone_5_6():
    model = S.make_upernet(224, 224)
    state = model.encoder.model.state_dict()
    ref = m5.timm_state(m5.encoder_parameters())
    assert np.array_equal(state["layers_3.blocks.1.mlp.fc2.weight"].numpy(),
                          ref["layers.3.blocks.1.mlp.fc2.weight"])
    pe = m6.timm_pe_state(m6.pe_parameters())
    assert torch.equal(state["patch_embed.proj.weight"], pe["proj.weight"])


def test_decoder_parameters_seeded_and_counted():
    counts = S.parameter_counts()
    assert counts["all"]["total"] == 9_231_682
    assert counts["fusion_3x3"]["conv_weight"] == 1024 * 256 * 9
    assert counts["psp_out_conv_3x3"]["conv_weight"] == 1792 * 256 * 9
    p = S.decoder_parameters()
    assert all(v.dtype == np.float32 for v in p.values())
    assert min(v.min() for k, v in p.items() if k.endswith("running_var")) > 0.1
