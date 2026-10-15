"""Offline checks of the explicit periodic extension (Milestone 4, no Earth Engine calls).

The EE construction (tensor_ee/layouts/shift.py, periodic_extension:
translated copies clipped to their strips, rows then columns) is emulated in
NumPy and compared with np.pad(mode="wrap"); Milestone 3's periodic-wrap
shift is then run on the *extended* raster (its neighbour read across the
edge comes from the ring, not from np.roll) and compared with timm's
window_partition(torch.roll(...)) at every edge and corner.
"""

import numpy as np
import pytest
import torch

from tensor_ee.layouts.shift import HIERARCHY_HALO, _copies
from reference.swin_pair import pytorch_shifted_swin as m3

RASTERS = [(2, 2), (3, 3), (4, 4), (3, 5), (5, 2), (4, 6)]


def _extend(windows: np.ndarray, halo: int) -> np.ndarray:
    """[kh, kw, ...] -> [kh + 2 halo, kw + 2 halo, ...] by the EE construction."""
    out = windows
    for axis in (0, 1):
        n = out.shape[axis]
        size = list(out.shape)
        size[axis] = n + 2 * halo
        result = np.full(size, np.nan)
        for m, start, end in _copies(n, halo):
            source = [slice(None)] * out.ndim
            target = [slice(None)] * out.ndim
            source[axis] = slice(start - m * n, end - m * n)     # new(I) = x(I - m n)
            target[axis] = slice(start + halo, end + halo)
            result[tuple(target)] = out[tuple(source)]
        out = result
    return out


@pytest.mark.parametrize("halo", [1, 2, 3, 5])
@pytest.mark.parametrize("rows,cols", RASTERS)
def test_construction_is_wrap_padding(rows, cols, halo):
    ids = m3.to_window_layout(m3.token_ids(rows, cols)).reshape(rows, cols, 49, 3)
    extended = _extend(ids.astype(np.float64), halo)
    expected = np.pad(ids, ((halo, halo), (halo, halo), (0, 0), (0, 0)), mode="wrap")
    assert not np.isnan(extended).any()
    assert np.array_equal(extended, expected)


def _shift_on_extended(extended: np.ndarray, halo: int, rows: int, cols: int, offset: int):
    """Milestone 3's two-neighbour slab shift for the raster's windows, reading
    neighbours from the extended array (offset > 0: next window)."""
    w = 7
    step = 1 if offset > 0 else -1

    def roll_axis(arr, axis):
        out = np.empty_like(arr)
        n = arr.shape[axis]
        for k in range(1, n - 1):
            own = np.take(arr, k, axis=axis)
            nb = np.take(arr, k + step, axis=axis)
            if offset > 0:
                parts = ((own, offset, w), (nb, 0, offset))
            else:
                parts = ((nb, w + offset, w), (own, 0, w + offset))
            c = own.shape[-1]
            if axis == 0:
                value = np.concatenate([p[..., a * w:b * w, :] for p, a, b in parts], -2)
            else:
                lead = own.shape[:-2]
                value = np.concatenate([p.reshape(*lead, w, w * c)[..., a * c:b * c]
                                        for p, a, b in parts], -1).reshape(*lead, w * w, c)
            index = [slice(None)] * arr.ndim
            index[axis] = k
            out[tuple(index)] = value
        return out

    rolled = roll_axis(roll_axis(extended, 0), 1)
    return rolled[halo:halo + rows, halo:halo + cols]


@pytest.mark.parametrize("rows,cols", RASTERS)
def test_shift_of_extended_raster_is_torch_roll(rows, cols):
    pytest.importorskip("timm")
    from timm.models.swin_transformer import window_partition
    ids = m3.token_ids(rows, cols)
    windows = m3.to_window_layout(ids).reshape(rows, cols, 49, 3)
    extended = _extend(windows.astype(np.float64), 1)
    for offset, roll in ((3, -3), (-3, 3)):
        got = _shift_on_extended(extended, 1, rows, cols, offset).reshape(-1, 49, 3)
        rolled = torch.roll(torch.from_numpy(ids)[None], (roll, roll), (1, 2))
        expected = window_partition(rolled, (7, 7)).reshape(-1, 49, 3).numpy()
        assert np.array_equal(got, expected)
        # every edge and corner window separately
        got4, exp4 = got.reshape(rows, cols, 49, 3), expected.reshape(rows, cols, 49, 3)
        for name, index in (("top", np.s_[0]), ("bottom", np.s_[-1]), ("left", np.s_[:, 0]),
                            ("right", np.s_[:, -1]), ("corner_tl", np.s_[0, 0]),
                            ("corner_tr", np.s_[0, -1]), ("corner_bl", np.s_[-1, 0]),
                            ("corner_br", np.s_[-1, -1])):
            assert np.array_equal(got4[index], exp4[index]), name


def test_hierarchy_halo():
    # stage-2 SW-MSA reads 1 stage-2 window = 2 stage-1 windows; stage-1 SW-MSA 1 more
    assert HIERARCHY_HALO == 3


def test_patch_merge_halves_the_period():
    """A periodic stage-1 raster (period kh, kw) gives a stage-2 raster of period kh/2, kw/2."""
    from reference.swin_hierarchy.pytorch_hierarchy import gather_windows
    rows, cols, reps = 4, 6, 2
    base = m3.to_window_layout(m3.token_ids(rows, cols)).reshape(rows, cols, 49, 3)
    tiled = np.tile(base, (reps, reps, 1, 1)).reshape(-1, 49, 3)
    merged = gather_windows(tiled, rows * reps, cols * reps).reshape(rows, cols, 49, 12)
    period = merged[:rows // 2, :cols // 2]
    assert np.array_equal(merged, np.tile(period, (reps, reps, 1, 1)))
