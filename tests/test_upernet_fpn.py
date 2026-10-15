"""Milestone 9, offline: the FPN mirror against SMP, the composed resizes and their grids, the
zero-padding semantics and the local halo (Tests 4, 9, 12, 14, 16, 18 offline halves;
Phases 11, 13-15)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("segmentation_models_pytorch")

from reference.upernet import ee_mirror_decoder as M  # noqa: E402
from reference.upernet import ee_mirror_fpn as MF  # noqa: E402
from reference.upernet import pytorch_decoder_primitives as P  # noqa: E402
from reference.upernet import smp_feature_interface as S  # noqa: E402
from reference.upernet import smp_fpn_reference as R  # noqa: E402


@pytest.fixture(scope="module")
def refs():
    return R.references(R.synthetic_features(14), refs=("torch_float32", "float64", "ee_mirror"))


def test_mirror_tracks_smp_through_the_recursion(refs):
    """Phase 11: every PSP and FPN tensor of the mirror within 1e-4 of SMP float32 and 1e-6 of
    float64 (synthetic C4 14 x 14 -> C1 112 x 112)."""
    for name in R.TENSORS:
        mirror = refs["ee_mirror"][name]
        assert np.abs(mirror - refs["torch_float32"][name]).max() <= 1e-4, name
        assert np.abs(mirror - refs["float64"][name]).max() <= 1e-6, name


def test_mirror_recursion_is_the_installed_one(refs):
    m = refs["ee_mirror"]
    assert np.array_equal(m["up3"], M.resize(m["psp_relu"], (28, 28)))
    assert np.array_equal(m["up2"], M.resize(m["F3"], (56, 56)))       # F3, not P3
    assert np.array_equal(m["up1"], M.resize(m["F2"], (112, 112)))
    for k in (3, 2, 1):
        assert np.array_equal(m[f"F{k}"], M.add(m[f"up{k}"], m[f"lat{k}"]))


def test_chunked_conv_equals_one_product(refs):
    f = refs["ee_mirror"]["F1"]
    w = S.decoder_parameters()[f"{R.conv_prefix(1)}.0.weight"]
    assert np.array_equal(MF.conv3x3(f, w), M.conv3x3(f, w))
    assert np.array_equal(MF.conv3x3(f, w, block_rows=112), M.conv3x3(f, w))


@pytest.mark.reference_environment
def test_sequential_matmul_at_the_mirror_block_shape():
    assert MF.sequential_rows_agree(rows=3136, k=2304, cols=4)


def test_composed_resizes_follow_pytorch_exactly():
    """Ramps (row, column, 1) and level IDs through three 2x resizes: PyTorch == mirror ==
    the written rule, bit for bit, edges included; the ID channel reads 11 / 111 / 1111."""
    h4 = 7
    ramp = np.stack(list(np.meshgrid(np.arange(h4), np.arange(h4), indexing="ij"))
                    + [np.ones((h4, h4))], -1).astype(np.float32)
    t_state, m_state = ramp, ramp.astype(np.float64)
    for k in (3, 2, 1):
        h = h4 * 2 ** (4 - k)
        lat = np.zeros((h, h, 3), np.float32)
        lat[..., 2] = 10.0 ** (4 - k)
        t_state = P.add(P.interpolate(t_state, (h, h)), lat)
        m_state = M.add(M.resize(m_state, (h, h)), lat)
        assert np.array_equal(t_state, m_state)
        assert np.all(m_state[..., 2] == sum(10 ** j for j in range(4 - k + 1)))
        # the edge clamp: output row 0 reads source row 0 only
        assert m_state[0, 0, 0] == 0.0


def test_zero_padding_is_detectable_only_on_the_border(refs):
    """Phase 15: replicate / reflect padding differ from zero padding on every border pixel
    region and nowhere inside, so the EE comparisons can tell the padding apart."""
    f = refs["ee_mirror"]["F2"]
    w = S.decoder_parameters()[f"{R.conv_prefix(2)}.0.weight"]
    zero = MF.conv3x3(f, w)
    for mode in ("replicate", "reflect"):
        other = MF.padded_variant(f, w, mode)
        assert np.array_equal(other[1:-1, 1:-1], zero[1:-1, 1:-1])
        for edge in (other[0] - zero[0], other[-1] - zero[-1], other[:, 0] - zero[:, 0],
                     other[:, -1] - zero[:, -1]):
            assert np.abs(edge).max() > 1e-2
    t = torch.from_numpy(np.asarray(f, np.float32)).permute(2, 0, 1)[None].double()
    pytorch = torch.nn.functional.conv2d(t, torch.from_numpy(w).double(), padding=1)[0].permute(1, 2, 0)
    assert np.abs(pytorch.numpy() - zero).max() < 1e-5
