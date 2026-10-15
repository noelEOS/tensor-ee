"""Offline: the EE patch gathers and the patch-embed -> stage-1 window layout, step for
step in NumPy, against timm's own Conv2d extraction (Phases 6, 8; Tests 1, 2, 5 offline).

The EE graph (tensor_ee/blocks/window/patch.py) only moves values with exact stride
samples (Milestone 1: sample a * w + b of a w-fold space-to-depth holds fine pixel
(w I + a, w J + b)) and with neighborhoodToArray; these emulations follow it.
The live tests compare EE itself with the same timm IDs.
"""

import numpy as np
import pytest

from reference.swin_block.pytorch_swin_block import to_window_layout
from reference.swin_image_encoder import pytorch_image_encoder as R


def stride_samples(raster: np.ndarray, w: int) -> list[np.ndarray]:
    """Milestone 1's space-to-depth: sample a * w + b = raster[a::w, b::w]."""
    return [raster[a::w, b::w] for a in range(w) for b in range(w)]


def ee_stride_columns(z: np.ndarray) -> np.ndarray:
    """patch_embed.stride_columns: bands (c, ky, kx) of sample ky * 4 + kx."""
    samples = stride_samples(z, 4)
    return np.stack([samples[ky * 4 + kx][..., c] for c in range(2) for ky in range(4)
                     for kx in range(4)], axis=-1)


def ee_to_windows(columns: np.ndarray) -> np.ndarray:
    """patch_embed.to_windows: 49 stride samples stacked in token order 7 a + b."""
    samples = stride_samples(columns, 7)
    return np.stack([s.reshape(-1, columns.shape[-1]) for s in samples], axis=1)


@pytest.mark.parametrize("height,width", [(224, 224), (224, 448), (448, 448)])
@pytest.mark.parametrize("gather", ["stride"])
def test_patch_membership_and_order(height, width, gather):
    """Every element of every token's patch: source pixel, channel and order (Tests 1-2)."""
    pytest.importorskip("timm")
    ids = R.pixel_ids(height, width)
    columns = ee_stride_columns(ids)
    windows = ee_to_windows(columns)
    assert np.array_equal(windows, R.timm_patch_ids(height, width))


def test_window_layout_matches_stage1_input():
    """Patch embed -> window layout equals Milestone 5's stage-1 layout of timm's tokens
    (Test 5): projecting window arrays == projecting tokens then partitioning."""
    pytest.importorskip("timm")
    pe = R.pe_parameters()
    z = np.random.default_rng(1).standard_normal((224, 224, 2)).astype(np.float32)
    ref = R.image_encoder_reference(None, pe, z, stop="x")
    module = R.make_patch_embed(224, 224, pe)
    import torch
    with torch.no_grad():
        tokens = module(R.to_nchw(z))[0].numpy()                    # [56, 56, 96] NHWC
    assert np.array_equal(ref["x"], to_window_layout(tokens))
    assert np.array_equal(ee_to_windows(ee_stride_columns(z)), ref["pe_patches"])


def test_corner_cases_of_the_partition():
    """Top-left, final row / column and bottom-right patches of a 448 x 448 crop."""
    pytest.importorskip("timm")
    ids = R.timm_patch_ids(448, 448)               # 16 x 16 windows
    last = 16 * 16 - 1
    for w, t, (y, x) in ((0, 0, (0, 0)), (15, 6, (0, 15 * 28 + 24)),
                         (16 * 15, 42, (15 * 28 + 24, 0)), (last, 48, (444, 444))):
        assert ids[w, t, 0] == (y * 448 + x) * 2            # (c, ky, kx) = (0, 0, 0)
        assert ids[w, t, 31] == ((y + 3) * 448 + x + 3) * 2 + 1
