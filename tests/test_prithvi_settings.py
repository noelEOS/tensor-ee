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


def _largest_change(expected, actual):
    return max(float(np.max(np.abs(expected[name].astype(np.float64) - actual[name])))
               for name in expected)


def sensitivity_results(case):
    """Reference mutations that the full-width fixture must distinguish."""
    seed = 8100 + fixtures.CASES.index(case)
    data = fixtures._data(seed, case=case)
    x, parameters = fixtures._expanded(data, case)
    expected = fixtures._reference_expanded(x, parameters, case=case)
    results = {}

    def record(name, image=x, changed=parameters):
        actual = fixtures._reference_expanded(image, changed, case=case)
        results[name] = _largest_change(expected, actual)

    channels = np.roll(np.arange(x.shape[-1]), 8 if x.shape[-1] > 8 else 1)
    record('input_channels', image=x[..., channels])
    if 'weight' in parameters:
        weight = parameters['weight']
        rows = np.roll(np.arange(case.cout), 8 if case.cout > 8 else 1)
        columns = np.roll(np.arange(case.cin), 8)
        record('weight_rows_bias_unchanged', changed={**parameters, 'weight': weight[rows]})
        record('weight_columns', changed={**parameters, 'weight': weight[:, columns]})
        if case.kind == 'patch_embed':
            swapped = x.copy()
            swapped[:16, :16] = x[:16, 32:48]
            swapped[:16, 32:48] = x[:16, :16]
            record('patch_token_order', image=swapped)
            order = np.arange(1536).reshape(6, 16, 16)
            for name, axis, shift in [('patch_band_order', 0, 1),
                                      ('patch_ky_order', 1, 8),
                                      ('patch_kx_plus_eight', 2, 8)]:
                permutation = np.roll(order, shift, axis=axis).reshape(-1)
                record(name, changed={**parameters, 'weight': weight[:, permutation]})
    else:
        order = np.roll(np.arange(case.cout), 8)
        record('normalization_affine_channels', changed={
            **parameters, 'gamma': parameters['gamma'][order], 'beta': parameters['beta'][order]})
    return results


@pytest.mark.parametrize('case', fixtures.CASES, ids=lambda case: case.id)
def test_fixture_detects_channel_and_patch_order_mutations(case):
    import json

    results = sensitivity_results(case)
    print(json.dumps({'fixture': case.id, 'sensitivity_max_abs': results}, sort_keys=True))
    assert all(delta > 1e-4 for delta in results.values()), results


@pytest.mark.parametrize('case', fixtures.CASES, ids=lambda case: case.id)
def test_fixture_input_weight_and_output_channels_are_distinct(case):
    seed = 8100 + fixtures.CASES.index(case)
    data = fixtures._data(seed, case=case)
    x, parameters = fixtures._expanded(data, case)
    if case.kind == 'patch_embed':
        import torch
        import torch.nn.functional as F

        torch.set_num_threads(2)
        columns = F.unfold(torch.from_numpy(x).permute(2, 0, 1)[None], 16, stride=16)
        input_columns = columns[0].numpy()
    else:
        input_columns = x.reshape(-1, case.cin).T
    assert np.unique(input_columns, axis=0).shape[0] == case.cin
    if 'weight' in parameters:
        matrix = parameters['weight'].reshape(case.cout, -1)
        assert np.unique(matrix, axis=0).shape[0] == case.cout
        assert np.unique(matrix.T, axis=0).shape[0] == matrix.shape[1]
    else:
        for name in ('gamma', 'beta'):
            assert np.unique(parameters[name]).size == case.cout
    output = fixtures._reference_expanded(x, parameters, case=case)
    prefix = 'conv' if case.kind == 'conv3x3_window' else 'output'
    chunks = [value for name, value in output.items() if name.startswith(prefix)]
    channels = np.concatenate(chunks, axis=-1).reshape(-1, case.cout).T
    assert np.unique(channels, axis=0).shape[0] == case.cout


@pytest.mark.parametrize('case', fixtures.CASES, ids=lambda case: case.id)
def test_compact_coefficient_graph_expands_to_reference_values(case):
    from types import SimpleNamespace
    from reference.segformer_b0.blocks import Operand

    arrays = SimpleNamespace(Array=Operand, List=SimpleNamespace(
        sequence=lambda first, last: list(range(first, last + 1))))
    data = fixtures._data(8100 + fixtures.CASES.index(case), case=case)
    _, parameters = fixtures._expanded(data, case)
    for name, source in data.weights.items():
        expected = parameters[name]
        actual = fixtures._server(arrays, source, expected.shape, row_gains=name == 'weight')
        np.testing.assert_array_equal(actual.array.values.view(np.uint32),
                                      expected.reshape(actual.shape).view(np.uint32))
