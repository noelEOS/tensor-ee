"""Milestone 11, offline: the head convolution and its mirror (Tests 4-5, 13-14).

Channel / class ordering with unique IDs, zero and non-zero bias, the bias added to the double
dot before the one rounding, and the mirror's head on real fused features against SMP."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reference.upernet import ee_mirror_head as MH  # noqa: E402
from reference.upernet import pytorch_decoder_primitives as P  # noqa: E402
from reference.upernet import smp_feature_interface as S  # noqa: E402


def test_class_and_channel_ids_are_exact():
    w = (1000.0 * (np.arange(2)[:, None] + 1) + np.arange(256)[None, :]).astype(np.float32)
    b = np.array([0.5, -0.25], np.float32)
    j = (np.arange(49) * 37 % 256).reshape(7, 7)
    x = np.zeros((7, 7, 256), np.float32)
    x[np.arange(7)[:, None], np.arange(7)[None, :], j] = 1
    got = MH.head_conv(x, w, b)
    want = (w[:, j].transpose(1, 2, 0) + b).astype(np.float64)
    assert np.array_equal(got, want)
    assert np.array_equal(P.conv2d(x, w.reshape(2, 256, 1, 1), b), want.astype(np.float32))


def test_bias_is_added_before_the_one_rounding():
    x = np.zeros((7, 7, 256), np.float32)
    x[..., 0], x[..., 1] = 1.0, 2.0 ** -24
    w = np.zeros((2, 256), np.float32)
    w[0, 0] = w[0, 1] = 1.0
    out = MH.head_conv(x, w, np.array([2.0 ** -24, 0], np.float32))
    assert np.all(out[..., 0] == 1 + 2.0 ** -23)
    rounded_first = np.float32(np.float32(1 + 2.0 ** -24) + np.float32(2.0 ** -24))
    assert rounded_first == 1.0


@pytest.mark.parametrize("bias", ["zero", "real", "large"])
def test_head_conv_mirror_matches_pytorch(bias):
    pytest.importorskip("segmentation_models_pytorch")
    d = S.decoder_parameters()
    w = np.asarray(d[MH.WEIGHT], np.float32).reshape(2, 256)
    b = {"zero": np.zeros(2, np.float32), "real": np.asarray(d[MH.BIAS], np.float32),
         "large": np.array([3.0, -2.0], np.float32)}[bias]
    x = (np.random.default_rng(3).standard_normal((14, 14, 256)) * 1.5).astype(np.float32)
    got = MH.head_conv(x, w, b)
    assert np.abs(got - P.conv2d(x, w.reshape(2, 256, 1, 1), b)).max() < 1e-5
