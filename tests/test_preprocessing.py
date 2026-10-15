"""Offline: the M6 Sentinel-1 preprocessing contract (Phases 3, 16; Tests 6, 10 offline
halves).  The preprocessing is SegWater v3's audited recipe; tensor_ee's own tests
(tests/test_segwater_preprocessing.py) cover its EE/NumPy implementations.  Here: the
torch twin, the mask policies M6 evaluates, and the validity fraction per token."""

import numpy as np
import pytest

from tensor_ee.segwater_preprocessing import MEANS, preprocess_db_numpy
from reference.swin_image_encoder import pytorch_image_encoder as R


def test_numeric_fixture_numpy_torch_agree():
    from reference.cnn import edge_cases
    _, data = edge_cases()
    db = data[:, 0, :].T[None].astype(np.float64)          # [1, N, 2]
    assert np.array_equal(R.preprocess_db_torch(db), preprocess_db_numpy(db, channel_axis=-1))


def test_raw_double_is_cast_to_float32_first():
    """EE stores S1 dB as double; the contract casts to float32 before quantizing."""
    db = np.array([[[-15.123456789012345, -24.98765432109876]]])
    assert np.array_equal(R.preprocess(db), R.preprocess(db.astype(np.float32).astype(np.float64)))


def test_training_mean_policy_fills_before_quantization():
    db = np.full((4, 4, 2), -10.0)
    mask = np.ones((4, 4, 2), bool)
    mask[0, 0, :] = False
    mask[1, 1, 1] = False
    z = R.preprocess(db, mask, "training_mean")
    filled = preprocess_db_numpy(np.broadcast_to(np.asarray(MEANS, np.float32), (1, 1, 2)),
                                 channel_axis=-1)[0, 0]
    assert np.array_equal(z[0, 0], filled) and z[1, 1, 1] == filled[1]
    assert np.all(np.abs(filled) < 0.01) and np.all(filled != 0)    # small, not exactly zero
    assert np.array_equal(z[1, 1, 0], R.preprocess(db)[1, 1, 0])     # per channel


def test_patch_valid_fraction():
    """Fraction of a token's 32 patch values that are valid (patch_embed.valid_fraction)."""
    mask = np.ones((8, 8, 2), np.float32)
    mask[:4, :4] = 0                     # token (0, 0) fully masked
    mask[4, 4, 0] = 0                    # token (1, 1) one value masked
    columns = R.patches(R.to_nchw(mask))[0].numpy()
    fraction = columns.mean(-1)
    assert fraction[0, 0] == 0 and fraction[1, 1] == 31 / 32 and fraction[0, 1] == 1


@pytest.mark.parametrize("policy", ["training_mean", "zero"])
def test_dense_policies_give_finite_dense_inputs(policy):
    rng = np.random.default_rng(0)
    db = rng.uniform(-40, 10, (8, 8, 2))
    mask = rng.random((8, 8, 2)) > 0.3
    if policy == "zero":
        z = np.where(mask, R.preprocess(db), 0).astype(np.float32)
    else:
        z = R.preprocess(db, mask, policy)
    assert np.isfinite(z).all() and z.dtype == np.float32
