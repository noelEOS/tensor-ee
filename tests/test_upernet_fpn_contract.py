"""Milestone 9, offline: the installed SMP FPN contract (Test 1), the lateral feature mapping with
unique IDs (Test 2) and the pyramid order the fusion stage consumes (Phase 23), established
by running the installed code."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("segmentation_models_pytorch")

from reference.upernet import smp_fpn_reference as R  # noqa: E402


@pytest.fixture(scope="module")
def contract():
    return R.contract()


def test_source_and_versions(contract):
    assert contract["versions"] == {"segmentation_models_pytorch": "0.5.0",
                                    "torch": torch.__version__}
    assert contract["source"]["file"].endswith("segmentation_models_pytorch/decoders/upernet/decoder.py")
    forward = contract["source"]["decoder_forward"]
    # the recursion hands the lateral block's output (not the conv block's) down
    assert "state_feature = fpn_features[-1]" in forward
    assert forward.index("fpn_lateral_blocks") < forward.index("fpn_conv_blocks")
    assert "torch.cat(resized_fpn_features[::-1], dim=1)" in forward


def test_lateral_and_conv_blocks(contract):
    """Test 1: block configurations, read off the installed modules."""
    for block, cin in zip(contract["lateral_blocks"], (384, 192, 96)):
        assert block["class"] == "FPNLateralBlock"
        assert block["conv"] == {"class": "Conv2d", "in": cin, "out": 256, "kernel": [1, 1],
                                 "padding": [0, 0], "padding_mode": "zeros", "stride": [1, 1],
                                 "dilation": [1, 1], "groups": 1, "bias": False}
        assert block["norm"]["class"] == "BatchNorm2d" and block["norm"]["eps"] == 1e-5
        assert block["activation"] == {"class": "ReLU", "inplace": True}
    for block in contract["conv_blocks"]:
        assert block["conv"] == {"class": "Conv2d", "in": 256, "out": 256, "kernel": [3, 3],
                                 "padding": [1, 1], "padding_mode": "zeros", "stride": [1, 1],
                                 "dilation": [1, 1], "groups": 1, "bias": False}
        assert block["norm"]["eps"] == 1e-5 and block["activation"]["class"] == "ReLU"
    assert [(n["channels"], n["eps"]) for n in contract["feature_norms"]] == [
        (96, 1e-6), (192, 1e-6), (384, 1e-6), (768, 1e-6)]


def test_call_order(contract):
    """Test 1: norms, PSP, the three lateral blocks, THEN the three conv blocks, then fusion."""
    assert contract["call_order"] == [
        "feature_norms.0", "feature_norms.1", "feature_norms.2", "feature_norms.3", "psp",
        "fpn_lateral_blocks.0", "fpn_lateral_blocks.1", "fpn_lateral_blocks.2",
        "fpn_conv_blocks.0", "fpn_conv_blocks.1", "fpn_conv_blocks.2", "fusion_block"]


def test_lateral_mapping_with_unique_ids(contract):
    """Test 2: block 0 <- C3, block 1 <- C2, block 2 <- C1, every channel in order."""
    mapping = contract["ids"]["lateral_mapping"]
    assert [m["lateral_input_is"] for m in mapping] == [["C3"], ["C2"], ["C1"]]
    assert [m["channels"] for m in mapping] == [384, 192, 96]
    assert all(m["channel_ids_exact"] for m in mapping)
    assert contract["ids"]["conv_block_sizes"] == [[14, 14], [28, 28], [56, 56]]


def test_recursion_hands_down_the_pre_conv_sum(contract):
    """PSP = 1, laterals 10 / 100 / 1000: the state into block i is 1, 11, 111 and the conv
    blocks receive 11, 111, 1111 (were P handed down, block 1 would see 10000 + c)."""
    ids = contract["ids"]
    assert ids["state_into_lateral_blocks"] == [1.0, 11.0, 111.0]
    assert ids["lateral_block_outputs"] == [11.0, 111.0, 1111.0]
    assert ids["conv_block_inputs"] == [11.0, 111.0, 1111.0]


def test_interpolation_calls(contract):
    calls = contract["interpolations"]
    assert all(c["mode"] == "bilinear" and c["align_corners"] is False for c in calls)
    psp, fpn, fusion = calls[:4], calls[4:7], calls[7:]
    assert [c["source"] for c in psp] == [[1, 1], [2, 2], [3, 3], [6, 6]]
    # FPNLateralBlock: size = the lateral's size, exactly 2x here
    assert [(c["source"], c["target"]) for c in fpn] == [([7, 7], [14, 14]), ([14, 14], [28, 28]),
                                                         ([28, 28], [56, 56])]
    assert all(c["size"] == c["target"] for c in fpn)
    assert len({c["caller"] for c in fpn}) == 1
    # fusion (M10): every level to C1's size, P1 included (an identity resize)
    assert [c["source"] for c in fusion] == [[7, 7], [14, 14], [28, 28], [56, 56]]
    assert all(c["target"] == [56, 56] for c in fusion)


def test_fusion_order_for_m10(contract):
    """Phase 23: the fusion block's input is [P1, P2, P3, PSP] (256 channels each)."""
    segments = contract["ids"]["fusion_segments"]
    assert [s["source"].split()[0] for s in segments] == ["P1", "P2", "P3", "PSP"]
    assert [s["channels"] for s in segments] == [[0, 256], [256, 512], [512, 768], [768, 1024]]
    assert all(s["ids_exact"] for s in segments)
    real = contract["real_forward"]
    assert real["fusion_input_equals_cat_resized_reversed"] and real["p1_identity_resize_exact"]


def test_expanded_forward_is_bit_identical_to_the_module():
    """expanded() raises unless every F, P and the PSP output equal the decoder's own forward;
    also in float64 and at another size."""
    for h4, dtype in ((7, torch.float32), (14, torch.float64)):
        feats = R.synthetic_features(h4)
        t = R.fpn_tensors([np.asarray(f, np.float64 if dtype == torch.float64 else np.float32)
                           for f in feats], dtype)
        for k, h in ((3, 2 * h4), (2, 4 * h4), (1, 8 * h4)):
            for name in R.level_tensors(k):
                assert t[name].shape == (h, h, R.tensor_channels(name)), name
