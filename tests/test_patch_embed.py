"""Offline: timm's Swin patch embedding semantics and the M6 patch-embedding reference/mirror
(Phases 1, 5-7, 10; Tests 1-4 offline halves).  The live EE halves are in
tests/test_swin_image_encoder_live.py."""

import numpy as np
import pytest
import torch

from reference.swin_image_encoder import pytorch_image_encoder as R
from reference.swin_image_encoder.ee_mirror_image_encoder import (
    matmul_is_sequential, patch_windows, pe_mirror)


def test_timm_patch_embed_configuration():
    pytest.importorskip("timm")
    model = R.make_image_model(224, 224)
    pe = model.patch_embed
    R.check_patch_embed(pe)                          # Conv2d(2, 96, 4, 4, pad 0, bias) + LN(96)
    assert pe.strict_img_size and not pe.dynamic_img_pad
    assert getattr(model, "absolute_pos_embed", None) is None
    assert {k: tuple(v.shape) for k, v in pe.state_dict().items()} == {
        "proj.weight": (96, 2, 4, 4), "proj.bias": (96,), "norm.weight": (96,), "norm.bias": (96,)}


def test_flatten_order_is_c_ky_kx():
    """timm's own Conv2d with one-hot weights extracts patch element k = (c, ky, kx)."""
    pytest.importorskip("timm")
    z = R.integer_fixture(56, 56)
    extracted = R.one_hot_extraction(z)
    for k in range(32):
        c, ky, kx = np.unravel_index(k, (2, 4, 4))
        assert np.array_equal(extracted[..., k], z[ky::4, kx::4, c])
    assert np.array_equal(extracted, R.patches(R.to_nchw(z, torch.float64))[0].numpy())


def test_conv_is_im2col_matmul_exactly_on_integers():
    pytest.importorskip("timm")
    rng = np.random.default_rng(3)
    pe = {"proj.weight": rng.integers(-3, 4, (96, 32)).astype(np.float32),
          "proj.bias": rng.integers(-3, 4, 96).astype(np.float32),
          "norm.weight": np.ones(96, np.float32), "norm.bias": np.zeros(96, np.float32)}
    z = R.integer_fixture(28, 56, seed=5)
    module = R.make_patch_embed(28, 56, pe)
    with torch.no_grad():
        conv = module.proj(R.to_nchw(z)).permute(0, 2, 3, 1)[0].numpy()
    im2col = R.patches(R.to_nchw(z))[0].numpy() @ pe["proj.weight"].T + pe["proj.bias"]
    assert np.array_equal(conv, im2col)


def test_non_divisible_inputs_are_rejected():
    pytest.importorskip("timm")
    model = R.make_image_model(224, 224)
    with pytest.raises(AssertionError):
        model.patch_embed(torch.zeros(1, 2, 226, 226))   # strict_img_size
    with pytest.raises(ValueError):
        R.make_image_model(226, 226)


def test_explicit_patch_embed_is_timm():
    pytest.importorskip("timm")
    pe = R.pe_parameters()
    module = R.make_patch_embed(56, 84, pe)
    x = R.to_nchw(np.random.default_rng(0).standard_normal((56, 84, 2)).astype(np.float32))
    t = R.explicit_patch_embed(module, x)                 # raises unless bit-identical
    assert t["x"].shape == (1, 14, 21, 96) and t["pe_patches"].shape == (1, 14, 21, 32)


def test_pe_parameters_seeded_and_flattened():
    pe = R.pe_parameters()
    assert {k: v.shape for k, v in pe.items()} == {"proj.weight": (96, 32), "proj.bias": (96,),
                                                   "norm.weight": (96,), "norm.bias": (96,)}
    assert all(v.dtype == np.float32 for v in pe.values())
    assert R.parameter_count(pe) == 3360
    assert abs(float(pe["norm.weight"].mean()) - 1.0) < 0.1 and float(pe["norm.weight"].std()) > 0.1
    state = R.timm_pe_state(pe)
    assert torch.equal(state["proj.weight"].reshape(96, 32), torch.from_numpy(pe["proj.weight"]))


def test_patch_ids_window_layout():
    """timm_patch_ids: window w, token t = 7 a + b, element (c, ky, kx) holds input pixel
    (28 I + 4 a + ky, 28 J + 4 b + kx), channel c."""
    pytest.importorskip("timm")
    ids = R.timm_patch_ids(56, 84)
    assert ids.shape == (2 * 3, 49, 32)
    for w, t, k in ((0, 0, 0), (5, 48, 31), (4, 17, 13), (2, 6, 21)):
        i, j = divmod(w, 3)
        a, b = divmod(t, 7)
        c, ky, kx = np.unravel_index(k, (2, 4, 4))
        y, x = 28 * i + 4 * a + ky, 28 * j + 4 * b + kx
        assert ids[w, t, k] == (y * 84 + x) * 2 + c


def test_mirror_patch_embed_close_to_torch():
    pytest.importorskip("timm")
    pe = R.pe_parameters()
    z = np.random.default_rng(4).standard_normal((224, 224, 2)).astype(np.float32)
    t32 = R.image_encoder_reference(None, pe, z, torch.float32, stop="x")
    f64 = R.image_encoder_reference(None, pe, z, torch.float64, stop="x")
    mirror = pe_mirror(pe, patch_windows(z))
    exact = pe_mirror(pe, patch_windows(z), exact=True)
    for name in ("pe_proj", "x"):
        assert np.abs(mirror[name] - t32[name]).max() < 1e-5
        assert np.abs(exact[name] - f64[name]).max() < 1e-12
        assert np.array_equal(mirror[name], mirror[name].astype(np.float32))   # float32 values
    assert np.array_equal(patch_windows(z), t32["pe_patches"])


@pytest.mark.reference_environment
def test_matmul_is_sequential_for_32_terms():
    assert matmul_is_sequential(length=32)


def test_preprocessing_torch_twin_matches_numpy():
    from tensor_ee.segwater_preprocessing import preprocess_db_numpy
    rng = np.random.default_rng(7)
    db = rng.uniform(-70, 40, (64, 64, 2))
    ties = (np.float32(np.arange(2, 400) + 0.5) * np.float32(0.05) - np.float32(55)).astype(np.float64)
    db[0, :, 0], db[1, :, 1] = ties[:64], ties[64:128]       # exact float32 quotient ties
    expected = preprocess_db_numpy(db, channel_axis=-1)
    assert np.array_equal(R.preprocess_db_torch(db), expected)
    assert np.array_equal(R.preprocess(db), expected)
