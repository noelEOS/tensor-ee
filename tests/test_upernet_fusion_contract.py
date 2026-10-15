"""Milestone 10, offline: the installed SMP fusion contract (Test 1), the [P1, P2, P3, PSP] order with
unique level / channel IDs (Test 2) and P1's same-size resize (Test 3), established by running the
installed code."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("segmentation_models_pytorch")

from reference.upernet import smp_fusion_reference as R  # noqa: E402


@pytest.fixture(scope="module")
def contract():
    return R.contract()


def test_source_and_versions(contract):
    assert contract["versions"] == {"segmentation_models_pytorch": "0.5.0",
                                    "torch": torch.__version__}
    src = contract["source"]
    assert src["file"].endswith("segmentation_models_pytorch/decoders/upernet/decoder.py")
    assert "target_size = fpn_features[-1].shape[2:]" in src["forward_fusion"]
    assert 'mode="bilinear", align_corners=False' in src["forward_fusion"]
    assert "torch.cat(resized_fpn_features[::-1], dim=1)" in src["forward_fusion"]
    assert "self.fusion_block(stacked_features)" in src["forward_fusion"]
    assert "num_blocks_to_fuse * decoder_channels" in src["init_fusion"]


def test_fusion_block_configuration(contract):
    block = contract["fusion_block"]
    assert block["class"] == "Conv2dReLU"
    assert block["conv"] == {"class": "Conv2d", "in": 1024, "out": 256, "kernel": [3, 3], "padding": [1, 1],
                             "padding_mode": "zeros", "stride": [1, 1], "dilation": [1, 1], "groups": 1,
                             "bias": False}
    assert block["norm"]["class"] == "BatchNorm2d" and block["norm"]["eps"] == 1e-5
    assert block["activation"] == {"class": "ReLU", "inplace": True}
    assert contract["parameters"] == {"decoder.fusion_block.0.weight": [256, 1024, 3, 3],
                                      **{f"decoder.fusion_block.1.{k}": [256] for k in
                                         ("weight", "bias", "running_mean", "running_var")}}


def test_fusion_interpolation_calls(contract):
    """Four calls in the decoder's own order (PSP, P3, P2, P1), all to P1's size."""
    calls = contract["fusion_interpolations"]
    assert [c["source"] for c in calls] == [[7, 7], [14, 14], [28, 28], [56, 56]]
    assert all(c["target"] == [56, 56] and c["size"] == [56, 56] for c in calls)
    assert all(c["mode"] == "bilinear" and c["align_corners"] is False and c["antialias"] is False
               and c["scale_factor"] is None for c in calls)
    assert len({c["caller"] for c in calls}) == 1


def test_fusion_order_with_unique_ids(contract):
    """Test 2: channel j of the fusion input is 1000 (j // 256 + 1) + j % 256 at every position."""
    ids = contract["ids"]
    assert ids["fusion_input_shape"] == [1, 1024, 56, 56]
    assert ids["every_channel_every_position_exact"]
    assert [s["source"] for s in ids["segments"]] == ["P1", "P2", "P3", "psp_relu"]
    assert [s["channels"] for s in ids["segments"]] == [[0, 256], [256, 512], [512, 768], [768, 1024]]
    assert [(s["first"], s["last"]) for s in ids["segments"]] == [(1000, 1255), (2000, 2255), (3000, 3255),
                                                                  (4000, 4255)]
    assert ids["call_order"][-1] == "fusion_block"


def test_decoder_output_is_the_fused_feature(contract):
    real = contract["real_forward"]
    assert real["expanded_equals_decoder_output"] and real["head_input_is_fusion_relu"]
    assert real["fusion_output_shape"] == [1, 256, 56, 56]


def test_p1_resize_is_a_bitwise_identity(contract):
    """Test 3: PyTorch's same-size bilinear call copies its input (signed zeros included)."""
    assert contract["real_forward"]["p1_identity"]["bits_equal"]
    x = torch.tensor([[[[-0.0, 1.5, 0.0], [2.0, -0.0, 3.25], [1e-30, 7.0, -0.0]]]])
    report = R.p1_identity(x)
    assert report["bits_equal"] and report["negative_zeros"] == 3 and not report["same_storage"]
    for dtype in (torch.float32, torch.float64):
        y = torch.randn(1, 16, 14, 14, dtype=dtype).relu()
        assert R.p1_identity(y)["bits_equal"]


def test_group_weight_slices_are_contiguous_columns(contract):
    assert contract["group_weight_slices_contiguous"]


def test_expanded_forward_is_bit_identical_to_the_module():
    """expanded() raises unless every tensor and the decoder output equal the module's own
    forward; also in float64 and at another size; fusion_concat is the four resized levels."""
    for h4, dtype in ((7, torch.float32), (14, torch.float64)):
        feats = R.FR.synthetic_features(h4)
        t = R.fusion_tensors([np.asarray(f, np.float64 if dtype == torch.float64 else np.float32)
                              for f in feats], dtype)
        h1 = 8 * h4
        for name in R.FUSION_TENSORS:
            assert t[name].shape == (h1, h1, R.tensor_channels(name)), name
        assert np.array_equal(t["fusion_concat"], np.concatenate([t[R.RESIZED[n]] for n in R.PYRAMID], -1))
        assert np.array_equal(t["P1_to_C1"], t["P1"])


def test_fusion_stage_on_given_pyramid_matches_the_decoder():
    """SMP's fusion alone on a pyramid equals the decoder's fusion on the same pyramid."""
    feats = R.FR.synthetic_features(7)
    t = R.fusion_tensors(feats, torch.float32)
    alone = R.fusion_on_pyramid({n: t[n] for n in R.PYRAMID})
    for name in R.FUSION_TENSORS:
        assert np.array_equal(alone[name], t[name]), name
    with torch.no_grad():
        p = R.nchw(t["P2"])
        assert np.array_equal(R.nhwc(F.interpolate(p, size=(56, 56), mode="bilinear", align_corners=False)),
                              t["P2_to_C1"])
