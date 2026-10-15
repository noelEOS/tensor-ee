"""Offline: the M6 image-encoder reference and mirror (Phases 2, 10, 13; Tests 9, 13
offline halves) on a 224 x 224 x 2 input shaped like preprocessed Sentinel-1.

The reference raises unless its explicit forward (timm's PatchEmbed submodules, then
Milestone 5's explicit encoder) is bit-identical to timm's
SwinTransformer(in_chans=2).forward_features; the mirror must stay within PyTorch's
float32 rounding of it.
"""

import numpy as np
import pytest
import torch

from reference.swin_encoder import pytorch_swin_encoder as m5
from reference.swin_image_encoder import pytorch_image_encoder as R
from reference.swin_image_encoder.ee_mirror_image_encoder import image_encoder_mirror

NAMES = ["input", "pe_patches", "pe_proj", "x", "s1_out", "pm1_out", "s2_out", "pm2_out",
         "s3_out", "pm3_out", "s4_out", "norm"]


@pytest.fixture(scope="module")
def fixture():
    """A smooth, S1-like preprocessed input (z ~ N(0, 1), spatially correlated)."""
    pytest.importorskip("timm")
    rng = np.random.default_rng(11)
    noise = rng.standard_normal((2, 232, 232))
    kernel = np.ones(9) / 9.0
    smooth = np.apply_along_axis(lambda r: np.convolve(r, kernel, "valid"), 2, noise)
    smooth = np.apply_along_axis(lambda c: np.convolve(c, kernel, "valid"), 1, smooth)
    z = (3.0 * smooth).transpose(1, 2, 0)[:224, :224].astype(np.float32)
    p = m5.encoder_parameters()
    pe = R.pe_parameters()
    return {"z": z, "p": p, "pe": pe,
            "t32": R.image_encoder_reference(p, pe, z, torch.float32, NAMES),
            "f64": R.image_encoder_reference(p, pe, z, torch.float64, NAMES),
            "mirror": image_encoder_mirror(p, pe, z, names=NAMES),
            "exact": image_encoder_mirror(p, pe, z, exact=True, names=NAMES)}


def test_reference_is_forward_features(fixture):
    """image_encoder_reference already raises otherwise; check the shapes too."""
    t32 = fixture["t32"]
    assert t32["x"].shape == (64, 49, 96)
    assert t32["norm"].shape == (1, 49, 768)            # 8 x 8 windows -> one stage-4 window


def test_encoder_part_is_milestone5(fixture):
    """Fed the patch-embedding tokens, Milestone 5's reference gives the same tensors."""
    t32 = fixture["t32"]
    tokens = R.make_patch_embed(224, 224, fixture["pe"])(R.to_nchw(fixture["z"]))[0].detach().numpy()
    m5_out = m5.encoder_reference(fixture["p"], tokens, torch.float32, ["s2_out", "norm"])
    assert np.array_equal(m5_out["norm"], t32["norm"])
    assert np.array_equal(m5_out["s2_out"], t32["s2_out"])


def test_mirror_within_pytorch_rounding(fixture):
    for name in NAMES:
        mirror, t32, f64 = fixture["mirror"][name], fixture["t32"][name], fixture["f64"][name]
        assert np.abs(mirror - t32).max() <= 1e-4, name
        # exact=True rounds nothing in the patch embedding; Milestone 5's encoder mirror
        # takes its input as float32 (EE's stage-1 tensor), so downstream it is only close.
        tolerance = 1e-12 if name in ("input", "pe_patches", "pe_proj", "x") else 1e-5
        assert np.abs(fixture["exact"][name] - f64).max() <= tolerance, name
    assert np.array_equal(fixture["mirror"]["pe_patches"], fixture["t32"]["pe_patches"])
    assert np.array_equal(fixture["mirror"]["input"], fixture["z"].astype(np.float64))


def test_float32_values_where_ee_rounds(fixture):
    for name in ("pe_proj", "x", "s1_out", "s3_out", "norm"):
        value = fixture["mirror"][name]
        assert np.array_equal(value, value.astype(np.float32).astype(np.float64)), name
