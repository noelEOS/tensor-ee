"""Offline checks of Patch Merging (Milestone 4, no Earth Engine calls).

timm's PatchMerging semantics are established on unique token IDs, and the
window-layout gather of tensor_ee/blocks/window/patch.py is
emulated in NumPy step for step (same slices, concatenations, reshapes) and
compared with timm's own gather.
"""

import numpy as np
import pytest

from tensor_ee.blocks.window.patch import GROUP_OFFSETS
from reference.swin_hierarchy import pytorch_hierarchy as ref
from reference.swin_pair import pytorch_shifted_swin as m3

SIZES = [(4, 4), (4, 6), (6, 4), (8, 8)]


@pytest.fixture(scope="module")
def semantics():
    pytest.importorskip("timm")
    return ref.patch_merge_semantics()


def test_timm_group_order(semantics):
    offsets = [(g["dy"], g["dx"]) for g in semantics["groups"]]
    # x0 = x[0::2, 0::2], x1 = x[1::2, 0::2], x2 = x[0::2, 1::2], x3 = x[1::2, 1::2]
    assert offsets == [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert tuple(offsets) == GROUP_OFFSETS


def test_timm_norm_then_reduction_without_bias(semantics):
    assert semantics["norm"]["normalized_shape"] == [384]
    assert semantics["norm"]["eps"] == 1e-5
    assert semantics["reduction"] == {"type": "Linear", "in_features": 384, "out_features": 192,
                                      "bias": False}
    assert semantics["explicit_composition_bit_identical"] == {"float32": True, "float64": True}


def test_timm_pads_bottom_and_right(semantics):
    padding = semantics["padding"]
    assert padding["output"] == [3, 4]
    assert padding["zero_groups_last_row"] == [1, 3]      # the dy = 1 groups
    assert padding["zero_groups_last_col"] == [2, 3]      # the dx = 1 groups


@pytest.mark.parametrize("rows,cols", SIZES)
def test_numpy_gather_is_timm(rows, cols):
    pytest.importorskip("timm")
    ids = m3.token_ids(rows, cols)
    assert np.array_equal(ref.gather_windows(ref.to_window_layout(ids), rows, cols),
                          ref.stage2_token_ids(rows, cols))


def _emulate_merge(windows: np.ndarray, channels: int) -> np.ndarray:
    """NumPy emulation of patch_merge.gather_windows + merge_windows on
    stage-1 windows [kh, kw, 49, C] -> [kh/2, kw/2, 49, 4C]."""
    w, c = 7, channels
    a, b = windows[0::2, 0::2], windows[0::2, 1::2]       # gather: (2P + a, 2Q + b)
    cc, d = windows[1::2, 0::2], windows[1::2, 1::2]
    lead = a.shape[:2]

    def side_by_side(left, right):
        return np.concatenate([left.reshape(*lead, w, w * c), right.reshape(*lead, w, w * c)], -1)

    block = np.concatenate([side_by_side(a, b), side_by_side(cc, d)], -2)     # [14, 14C]
    pairs = block.reshape(*lead, w, 4 * w * c)
    by_dy = [pairs[..., dy * 2 * w * c:(dy + 1) * 2 * w * c].reshape(*lead, w * w, 2 * c)
             for dy in (0, 1)]
    groups = [by_dy[dy][..., dx * c:(dx + 1) * c] for dy, dx in GROUP_OFFSETS]
    return np.concatenate(groups, -1)


@pytest.mark.parametrize("rows,cols", SIZES)
def test_window_layout_merge_algorithm_is_timm(rows, cols):
    pytest.importorskip("timm")
    ids = m3.token_ids(rows, cols)
    windows = ref.to_window_layout(ids).reshape(rows, cols, 49, 3)
    merged = _emulate_merge(windows, 3).reshape(-1, 49, 12)
    assert np.array_equal(merged, ref.stage2_token_ids(rows, cols))


def test_merged_row_three_straddles_stage1_windows():
    """Merged token row u = 3 pairs stage-1 token row 6 with row 7 (the next window)."""
    pytest.importorskip("timm")
    ids = ref.stage2_token_ids(2, 2)[0].reshape(7, 7, 4, 3)       # [u, v, group, (id, y, x)]
    assert set(ids[3, :, 0, 1]) == {6} and set(ids[3, :, 1, 1]) == {7}
    sources = {(int(y) // 7, int(x) // 7) for y, x in ids[..., 1:].reshape(-1, 2)}
    assert sources == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_every_stage1_token_used_once():
    pytest.importorskip("timm")
    ids = ref.stage2_token_ids(4, 6).reshape(-1, 4, 3)[..., 0]
    assert np.array_equal(np.sort(ids.reshape(-1)), np.arange(28 * 42))


def test_stage1_spec_is_milestone2s():
    from reference.swin_block.pytorch_swin_block import PARAMETER_SPEC
    assert ref.block_parameter_spec(96) == PARAMETER_SPEC


def test_parameters_reproduce_milestone3_and_are_distinct():
    pytest.importorskip("timm")
    from reference.swin_block.pytorch_swin_block import make_parameters
    p = ref.hierarchy_parameters()
    model = ref.make_model((28, 28))
    template = {k[len(ref.TIMM_PREFIX["s1a"]):]: v for k, v in model.state_dict().items()
                if k.startswith(ref.TIMM_PREFIX["s1a"])}
    regenerated = ref.generate_parameters(template, ref.block_parameter_spec(96), 0)
    for name, value in make_parameters(0).items():
        assert np.array_equal(regenerated[name], value)
        assert np.array_equal(p["s1a"][name], value)
    assert p["s2a"]["attn.qkv.weight"].shape == (576, 192)
    assert p["s2a"]["mlp.fc1.weight"].shape == (768, 192)
    assert p["s2b"]["attn.relative_position_bias_table"].shape == (169, 6)
    assert p["pm"]["reduction.weight"].shape == (192, 384) and "reduction.bias" not in p["pm"]
    assert not np.array_equal(p["s2a"]["norm1.weight"], p["s2b"]["norm1.weight"])
    assert sum(v.size for g in p.values() for v in g.values()) == 1_190_946


def test_reference_patch_merge_matches_numpy_formula():
    pytest.importorskip("timm")
    tensors = ref.references(4, 4, ["s1b_out", "pm_x0", "pm_x1", "pm_x2", "pm_x3", "pm_cat"],
                             dtypes=("float64",))["float64"]
    cat = ref.gather_windows(tensors["s1b_out"], 4, 4)
    assert np.array_equal(cat, tensors["pm_cat"])
    for g in range(4):
        assert np.array_equal(tensors[f"pm_x{g}"], cat[..., 96 * g:96 * (g + 1)])
