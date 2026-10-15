"""Milestone 8, offline: the installed SMP PSP contract (Test 1) and its concatenation order
(Test 8, PyTorch half), established by running the installed code."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("segmentation_models_pytorch")

from reference.upernet import ee_mirror_psp as MP  # noqa: E402
from reference.upernet import smp_feature_interface as S  # noqa: E402
from reference.upernet import smp_psp_reference as R  # noqa: E402


@pytest.fixture(scope="module")
def contract():
    return R.contract()


def test_source_and_classes(contract):
    assert contract["versions"]["segmentation_models_pytorch"] == "0.5.0"
    assert contract["source"]["file"].endswith("segmentation_models_pytorch/decoders/upernet/decoder.py")
    assert contract["source"]["class"] == "PSPModule"
    assert set(contract["classes"]) == {"AdaptiveAvgPool2d", "BatchNorm2d", "Conv2d", "Conv2dReLU",
                                         "ModuleList", "PSPModule", "ReLU", "Sequential"}


def test_pool_sizes_and_branch_blocks(contract):
    assert contract["pool_sizes"] == [1, 2, 3, 6]
    for block in contract["branch_blocks"]:
        assert block["conv"] == {"class": "Conv2d", "in": 768, "out": 256, "kernel": [1, 1],
                                 "padding": [0, 0], "padding_mode": "zeros", "stride": [1, 1],
                                 "dilation": [1, 1], "groups": 1, "bias": False}
        assert block["norm"]["class"] == "BatchNorm2d" and block["norm"]["eps"] == 1e-5
        assert block["activation"] == {"class": "ReLU", "inplace": True}


def test_output_conv(contract):
    conv = contract["out_conv"]["conv"]
    assert (conv["in"], conv["out"], conv["kernel"], conv["padding"], conv["bias"]) == (
        1792, 256, [3, 3], [1, 1], False)
    assert conv["padding_mode"] == "zeros"
    assert contract["out_conv"]["norm"]["eps"] == 1e-5


def test_interpolation_calls(contract):
    calls = contract["interpolations"]
    assert [c["source"] for c in calls] == [[1, 1], [2, 2], [3, 3], [6, 6]]
    assert all(c["target"] == [28, 28] and c["mode"] == "bilinear" and c["align_corners"] is False
               for c in calls)


def test_concatenation_order_with_channel_ids(contract):
    """Test 8 (PyTorch): [C4_norm, branch 1, branch 2, branch 3, branch 6], 1792 channels."""
    cat = contract["concatenation"]
    assert cat["total_channels"] == 1792 and cat["order_matches_ids"] and cat["feature_part_exact"]
    assert [s["channels"] for s in cat["segments"]] == [[0, 768], [768, 1024], [1024, 1280],
                                                        [1280, 1536], [1536, 1792]]


def test_real_forward_feeds_channels_last_feature_norm(contract):
    real = contract["real_forward"]
    assert real["psp_input_equals_feature_norm3_of_C4"] and real["psp_output_equals_expanded"]
    assert real["psp_input_is_channels_last"]


def test_expanded_forward_is_bit_identical_to_the_module():
    """expanded() raises otherwise; also on a 56 x 56 map and in float64."""
    for h, dtype in ((28, torch.float32), (56, torch.float32), (14, torch.float64)):
        t = R.psp_tensors(R.synthetic_c4(h, h), dtype)
        assert t["psp_relu"].shape == (h, h, 256)
        assert t["psp_concat"].shape == (h, h, 1792)
        for s in (1, 2, 3, 6):
            assert t[f"pool_{s}"].shape == (s, s, 768) and t[f"branch_{s}_up"].shape == (h, h, 256)


def test_mirror_tracks_smp_on_synthetic_c4():
    x = R.synthetic_c4(28, 28)
    t32 = R.psp_tensors(x)
    t64 = R.psp_tensors(x.astype(np.float64), torch.float64)
    mirror = MP.psp(x, S.decoder_parameters())
    for name in R.TENSORS:
        assert np.abs(mirror[name] - t32[name]).max() <= 1e-4, name
        assert np.abs(mirror[name] - t64[name]).max() <= 1e-6, name
