"""Unnormalized patch embedding and full-width window-block settings, offline."""
from __future__ import annotations

import numpy as np
import pytest

from tensor_ee.blocks.window.patch import patch_embed, patch_embed_arrays, patch_embed_constants, patch_embed_values
from tensor_ee.core.grid import TokenGrid
from tensor_ee.core.ops import WindowOps
from tensor_ee.weights.constants import WindowLayout
from tools.fixtures import prithvi_settings as fixtures
from tools.live_fixtures import offline_earth_engine


def _parameters():
    return {'proj.weight': np.arange(8 * 4, dtype=np.float32).reshape(8, 4) / 128,
            'proj.bias': np.arange(8, dtype=np.float32) / 16,
            'norm.weight': np.ones(8, np.float32), 'norm.bias': np.zeros(8, np.float32)}


def test_normalized_patch_default_graph_is_unchanged():
    with offline_earth_engine():
        import ee
        ops = WindowOps(ee)
        parameters = _parameters()
        omitted = patch_embed_constants(ops, parameters, WindowLayout(7))
        explicit = patch_embed_constants(ops, parameters, WindowLayout(7), norm=True)
        source = ee.Image.constant(1).rename('b0')
        grid = TokenGrid(14, 14)
        default = patch_embed(ops, source, grid, omitted, patch=2, bands=('b0',))
        enabled = patch_embed(ops, source, grid, explicit, patch=2, bands=('b0',), norm=True)
        assert default['x'].serialize() == enabled['x'].serialize()


def test_patch_can_omit_normalization_without_coefficients_or_extra_cast():
    with offline_earth_engine():
        import ee
        ops = WindowOps(ee)
        parameters = {k: v for k, v in _parameters().items() if k.startswith('proj.')}
        constants = patch_embed_constants(ops, parameters, WindowLayout(7), norm=False)
        assert constants.norm_weight is constants.norm_bias is None
        grid = TokenGrid(14, 14)
        output = patch_embed(ops, ee.Image.constant(1).rename('b0'), grid, constants,
                             patch=2, bands=('b0',), norm=False)
        from tensor_ee.layouts.shift import domain_mask
        expected = output['pe_proj'].updateMask(domain_mask(ee, grid.coarse(2).coarse(7)))
        assert output['x'].serialize() == expected.serialize()
        with pytest.raises(ValueError, match='agree'):
            patch_embed(ops, ee.Image.constant(1), grid, constants)
        enabled = patch_embed_constants(ops, _parameters(), WindowLayout(7))
        with pytest.raises(ValueError, match='agree'):
            patch_embed(ops, ee.Image.constant(1), grid, enabled, norm=False)


@pytest.mark.parametrize('invalid', [None, 0, 1, 'false'])
def test_norm_flag_is_boolean(invalid):
    with pytest.raises(ValueError, match='bool'):
        patch_embed_constants(None, {}, norm=invalid)
    with pytest.raises(ValueError, match='bool'):
        patch_embed_values(768, norm=invalid)
    with pytest.raises(ValueError, match='bool'):
        patch_embed_arrays(norm=invalid)


def test_patch_cost_omits_only_normalization():
    layout = WindowLayout(14)
    assert patch_embed_values(768, patch=16, channels=6, layout=layout, norm=False) == 1536 * 768 + 196 * 768
    assert patch_embed_values(768, patch=16, channels=6, layout=layout) - patch_embed_values(
        768, patch=16, channels=6, layout=layout, norm=False) == 2 * 196 * 768
    assert patch_embed_arrays() == 4
    assert patch_embed_arrays(norm=False) == 2


@pytest.mark.parametrize('case', fixtures.CASES, ids=lambda case: case.id)
def test_every_full_width_output_agrees_with_independent_torch(case):
    data = fixtures._data(8100 + fixtures.CASES.index(case), case=case)
    reference = fixtures._reference(data, case=case)
    mirror = fixtures._mirror(data, case=case)
    assert reference.keys() == mirror.keys()
    for key in reference:
        assert reference[key].shape == mirror[key].shape
        assert np.max(np.abs(reference[key].astype(np.float64) - mirror[key])) <= 1e-4, key
    prefix = 'conv' if case.kind == 'conv3x3_window' else 'output'
    assert sum(value.shape[-1] for key, value in reference.items() if key.startswith(prefix)) == case.cout
