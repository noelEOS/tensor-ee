"""Offline checks of Patch Merging 2 (192 -> 384) and 3 (384 -> 768) (Milestone 5, no EE).

The same window-layout algorithm (tensor_ee/blocks/window/patch.py)
serves all three Swin-T transitions.  Here it is emulated in NumPy step for
step at C = 96, 192 and 384 on unique (token, channel) IDs and compared with
timm's own PatchMerging gather, which checks the source spatial positions,
the 4C order, the channel order within each group, the window order and the
output token order in one comparison.  timm's semantics (LayerNorm(4C),
Linear(4C -> 2C, bias=False), gather -> norm -> reduction) are asserted for
the wider modules, and the mirror policy for Patch Merging 2 and 3.
"""

import numpy as np
import pytest
import torch

from reference.swin_encoder import pytorch_swin_encoder as R
from reference.swin_encoder.ee_mirror_encoder import merge_mirror
from tensor_ee.blocks.window.patch import GROUP_OFFSETS
from reference.swin_hierarchy import pytorch_hierarchy as m4

GRIDS = [(4, 4), (4, 6), (6, 4), (8, 8)]          # stage-n windows (even)


def emulate_merge(windows: np.ndarray, channels: int) -> np.ndarray:
    """NumPy emulation of patch_merge.gather_windows + merge_windows:
    [kh, kw, 49, C] stage-n windows -> [kh/2, kw/2, 49, 4C]."""
    w, c = 7, channels
    a, b = windows[0::2, 0::2], windows[0::2, 1::2]          # (2P + a, 2Q + b)
    cc, d = windows[1::2, 0::2], windows[1::2, 1::2]
    lead = a.shape[:2]

    def side_by_side(left, right):                          # [7, 7C] | [7, 7C]
        return np.concatenate([left.reshape(*lead, w, w * c), right.reshape(*lead, w, w * c)], -1)

    block = np.concatenate([side_by_side(a, b), side_by_side(cc, d)], -2)       # [14, 14C]
    pairs = block.reshape(*lead, w, 4 * w * c)                                  # [7, 28C]
    by_dy = [pairs[..., dy * 2 * w * c:(dy + 1) * 2 * w * c].reshape(*lead, w * w, 2 * c)
             for dy in (0, 1)]
    return np.concatenate([by_dy[dy][..., dx * c:(dx + 1) * c] for dy, dx in GROUP_OFFSETS], -1)


@pytest.mark.parametrize("channels", [96, 192, 384])
@pytest.mark.parametrize("rows,cols", GRIDS)
def test_window_layout_merge_is_timm_at_every_width(channels, rows, cols):
    pytest.importorskip("timm")
    ids = R.channel_token_ids(rows * 7, cols * 7, channels)
    windows = R.to_window_layout(ids).reshape(rows, cols, 49, channels)
    merged = emulate_merge(windows, channels).reshape(-1, 49, 4 * channels)
    timm = R.merged_channel_ids(rows * 7, cols * 7, channels)
    assert merged.shape == timm.shape == ((rows // 2) * (cols // 2), 49, 4 * channels)
    assert np.array_equal(merged, timm)


@pytest.mark.parametrize("channels", [192, 384])
def test_merged_ids_decode_to_timm_positions(channels):
    """Every merged element (i, j, g, c) holds source token (2i + dy_g, 2j + dx_g), channel c."""
    pytest.importorskip("timm")
    rows, cols = 4, 6
    height, width = rows * 7, cols * 7
    merged = R.merged_channel_ids(height, width, channels)
    raster = R.from_window_layout(merged, height // 2, width // 2).reshape(
        height // 2, width // 2, 4, channels).astype(np.int64)
    token, channel = np.divmod(raster, channels)
    y, x = np.divmod(token, width)
    i, j = np.meshgrid(np.arange(height // 2), np.arange(width // 2), indexing="ij")
    for g, (dy, dx) in enumerate(GROUP_OFFSETS):
        assert np.array_equal(y[:, :, g], np.broadcast_to(2 * i[..., None] + dy, y[:, :, g].shape))
        assert np.array_equal(x[:, :, g], np.broadcast_to(2 * j[..., None] + dx, x[:, :, g].shape))
        assert np.array_equal(channel[:, :, g], np.broadcast_to(np.arange(channels),
                                                                 channel[:, :, g].shape))


def test_channel_ids_are_exact_at_canonical_sizes():
    # stage 2 (112 x 112 x 192) and stage 3 (56 x 56 x 384) ID rasters stay below 2^24
    for side, channels in ((112, 192), (56, 384)):
        ids = R.channel_token_ids(side, side, channels)
        assert np.unique(ids).size == ids.size


@pytest.mark.parametrize("n,width", [(2, 192), (3, 384)])
def test_timm_patch_merging_semantics_at_new_widths(n, width):
    pytest.importorskip("timm")
    model = R.make_model((56, 56))
    pm = model.layers[n].downsample
    assert list(pm.norm.normalized_shape) == [4 * width] and pm.norm.eps == 1e-5
    assert pm.reduction.in_features == 4 * width and pm.reduction.out_features == 2 * width
    assert pm.reduction.bias is None
    probe = torch.from_numpy(np.random.default_rng(n).standard_normal((1, 4, 6, width))
                             .astype(np.float32))
    for dtype in (torch.float32, torch.float64):
        R.explicit_patch_merge(pm.to(dtype), probe.to(dtype))        # raises unless bit-identical


@pytest.mark.parametrize("n", [2, 3])
def test_merge_reference_and_mirror_policy(n):
    pytest.importorskip("timm")
    p = R.encoder_parameters()
    rows, cols = 4, 6
    x = R.stage_input(n, rows, cols)
    t32 = R.merge_reference(p, n, x, torch.float32)
    f64 = R.merge_reference(p, n, x, torch.float64)
    windows = R.to_window_layout(x)
    mirror = merge_mirror(p, n, windows, rows, cols)
    exact = merge_mirror(p, n, windows, rows, cols, exact=True)
    cat = m4.gather_windows(windows, rows, cols)              # NumPy x[dy::2, dx::2]
    for name in R.MERGE_TENSORS:
        if name.startswith("x") or name == "cat":             # layout only: exact
            g = int(name[1]) if name != "cat" else None
            expected = cat if g is None else cat[..., g * x.shape[-1]:(g + 1) * x.shape[-1]]
            assert np.array_equal(t32[name], expected) and np.array_equal(mirror[name], expected)
            continue
        gap_mirror = np.abs(mirror[name] - f64[name]).max()
        gap_torch = np.abs(t32[name].astype(np.float64) - f64[name]).max()
        assert gap_mirror <= gap_torch, name
        assert np.abs(mirror[name] - t32[name]).max() < 1e-5, name
        np.testing.assert_allclose(exact[name], f64[name], rtol=0, atol=1e-12)
    assert t32["out"].shape == ((rows // 2) * (cols // 2), 49, R.STAGE_DIMS[n])


def test_patch_merge_parameters_shapes():
    pytest.importorskip("timm")
    p = R.encoder_parameters()
    assert p["pm2"]["reduction.weight"].shape == (384, 768) and "reduction.bias" not in p["pm2"]
    assert p["pm3"]["reduction.weight"].shape == (768, 1536) and "reduction.bias" not in p["pm3"]
    assert p["pm3"]["norm.weight"].shape == (1536,)
