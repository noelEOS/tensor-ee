"""Milestone 11, offline: the complete forward (Tests 2, 18 offline half, 19).

The captured model.forward equals the expanded forward bit for bit (random input); on the
cached real crops the mirror's logits (= EE's, bit for bit, live) are within 1e-4 of SMP's
model.forward at every element; the references are internally consistent."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from reference.upernet import full_model_reference as FM  # noqa: E402
from reference.upernet import smp_feature_interface as S  # noqa: E402


def test_capture_and_expanded_forward_agree():
    m = S.make_upernet(224, 224)
    x = torch.from_numpy(np.random.default_rng(2).standard_normal((1, 2, 224, 224)).astype(np.float32))
    cap = FM.capture(m, x)
    out = FM.expanded(m, x, cap)                    # raises on any mismatch
    assert torch.equal(out["logits"], cap["logits"])
    assert list(cap["logits"].shape) == [1, 2, 224, 224]
    assert torch.equal(cap["head_input"], cap["fusion_relu"])
