"""Milestone 11, offline: the installed SMP segmentation head (Tests 1-3, 24).

Conv2d 256 -> 2 (1 x 1, bias) -> UpsamplingBilinear2d(scale_factor=4) = align_corners=True ->
Identity, read off the installed objects and a real forward; the expanded forward equals
model.forward bit for bit. (The research asset-record and seeded-manifest checks were archived
at L06b.)"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from reference.upernet import smp_feature_interface as S  # noqa: E402
from reference.upernet import smp_segmentation_head_reference as HR  # noqa: E402
from reference.upernet import smp_fusion_reference as R  # noqa: E402


@pytest.fixture(scope="module")
def head():
    return R.model(torch.float32).segmentation_head


def test_installed_head_modules(head):
    c = HR.head_config(head)
    assert c["conv"] == {"class": "Conv2d", "in": 256, "out": 2, "kernel": [1, 1], "padding": [0, 0],
                         "padding_mode": "zeros", "stride": [1, 1], "dilation": [1, 1], "groups": 1, "bias": True}
    assert c["upsample"]["class"] == "UpsamplingBilinear2d"
    assert c["upsample"]["align_corners"] is True and c["upsample"]["mode"] == "bilinear"
    assert c["upsample"]["scale_factor"] == 4.0 and c["upsample"]["size"] is None
    assert c["activation"] == {"class": "Activation", "inner": "Identity"}


def test_source_builds_the_head_as_recorded():
    src = HR.source()
    assert "kernel_size=1" in src["UPerNet_head_construction"]["source"]
    assert "upsampling=upsampling" in src["UPerNet_head_construction"]["source"]
    assert "UpsamplingBilinear2d(scale_factor=upsampling)" in src["SegmentationHead"]["source"]
    assert "align_corners=True" in src["UpsamplingBilinear2d_init"]["source"]
    assert "masks = self.segmentation_head(decoder_output)" in src["SegmentationModel_forward"]["source"]


def test_head_interpolate_call_is_align_corners_true():
    call = HR.interpolation_calls(224)[-1]
    assert call["scale_factor"] == 4.0 and call["align_corners"] is True and call["size"] is None
    assert call["source"] == [56, 56] and call["target"] == [224, 224]


def test_real_forward_expanded_equals_model_forward():
    r = HR.real_forward(224)                         # raises on any mismatch
    assert r["expanded_equals_model_forward"]
    assert r["layouts"]["head_input"]["contiguous"] and not r["layouts"]["head_input"]["channels_last"]
    eq = r["upsample_equivalents_bit_identical"]
    assert eq["F.interpolate(size=4h, align_corners=True)"] and eq["F.interpolate(scale_factor=4, align_corners=True)"]
    assert eq["channels_last input"] and not eq["align_corners=False (negative)"]
    assert r["output_shape"] == [1, 2, 224, 224] and r["output_dtype"] == "torch.float32"
