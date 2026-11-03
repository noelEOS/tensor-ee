"""The optional relative-bias gather keeps defaults and reaches whole Swin blocks."""
import json

import ee
import numpy as np
import pytest
from ee.apitestcase import ApiTestCase

from tensor_ee.blocks.window.attention import METADATA, relative_position_bias
from tensor_ee.blocks.window.swin import swin_block_constants, swin_stage_blocks
from tensor_ee.blocks.window.swin_upernet_config import SwinUPerNetConfig
from tensor_ee.converter.request import deep_recursion
from tensor_ee.core.metadata import check_settings
from tensor_ee.core.ops import WindowOps
from tensor_ee.core.tensor import ServerArray
from tensor_ee.weights.constants import WindowLayout
from tools.fixtures.relative_bias import _block_data


@pytest.fixture
def offline():
    case = ApiTestCase()
    case.setUp()
    try:
        with deep_recursion():
            yield
    finally:
        case.tearDown()


def _functions(value):
    names = set()
    def visit(item):
        if isinstance(item, dict):
            if 'functionName' in item:
                names.add(item['functionName'])
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(json.loads(value.serialize()))
    return names


def test_default_retains_one_hot_graph_and_opt_in_removes_large_product(offline):
    table = np.arange(169 * 2, dtype=np.float32).reshape(169, 2) / 128
    default = relative_position_bias(ee, table)
    original = relative_position_bias(ee, table, gather='one_hot')
    optimized = relative_position_bias(ee, table, gather='slices')
    assert [x.serialize() for x in default] == [x.serialize() for x in original]
    assert 'Array.matrixMultiply' in _functions(default[0])
    assert 'Array.matrixMultiply' not in _functions(optimized[0])
    assert {'Array.slice', 'Array.cat', 'List.map'} <= _functions(optimized[0])


@pytest.mark.parametrize('invalid', ['auto', 'list', None, True, 1, ['slices']])
def test_unknown_gather_is_refused(offline, invalid):
    with pytest.raises(ValueError, match='gather'):
        relative_position_bias(ee, np.zeros((169, 2), np.float32), gather=invalid)


@pytest.mark.parametrize('gather', ['one_hot', 'slices'])
def test_wrong_table_extent_is_refused_for_both_paths(offline, gather):
    with pytest.raises(ValueError, match='table rows'):
        relative_position_bias(ee, np.zeros((168, 2), np.float32), gather=gather)


def test_server_array_transport_remains_lazy_and_does_not_embed_coefficients(offline):
    table = ServerArray(ee.Array(ee.FeatureCollection('projects/example/assets/bias').first().get('table')), (169, 2))
    biases = relative_position_bias(ee, table, gather='slices')
    assert len(biases) == 2
    assert 'projects/example/assets/bias' in biases[0].serialize()
    assert 'Array.matrixMultiply' not in _functions(biases[0])


def test_block_and_stage_builders_propagate_opt_in(offline):
    parameters = _block_data(12).weights
    ops = WindowOps(ee)
    original = swin_block_constants(ops, parameters)
    optimized = swin_block_constants(ops, parameters, relative_bias_gather='slices')
    assert 'Array.matrixMultiply' in _functions(original.relative_bias[0])
    assert 'Array.matrixMultiply' not in _functions(optimized.relative_bias[0])
    config = SwinUPerNetConfig(embed_dim=16, depths=(1, 1, 1, 1),
                              stage_dims=(16, 32, 64, 128), num_heads=(2, 4, 8, 16))
    stage = swin_stage_blocks(ops, {'s1b0': parameters}, 1, config=config, relative_bias_gather='slices')
    assert stage[0].relative_bias[0].serialize() == optimized.relative_bias[0].serialize()
    assert optimized.qkv.weight_t.serialize() == original.qkv.weight_t.serialize()
    assert optimized.fc1.weight_t.serialize() == original.fc1.weight_t.serialize()


def test_bias_validation_preserves_default_and_limits_slices_to_live_settings():
    metadata = METADATA['relative_position_bias']
    assert check_settings(metadata, {'window': 7, 'gather': 'one_hot'}).inside
    for window in (7, 12, 14):
        assert check_settings(metadata, {'window': window, 'gather': 'slices', 'heads': 2}).inside
    assert not check_settings(metadata, {'window': 5, 'gather': 'slices', 'heads': 2}).inside
    assert not check_settings(metadata, {'window': 7, 'gather': 'slices', 'heads': 3}).inside


@pytest.mark.parametrize('block', ['swin_block', 'shifted_swin_block', 'swin_stage'])
def test_composite_record_does_not_claim_opt_in(block):
    from tensor_ee.blocks.window.swin import METADATA as SWIN_METADATA
    metadata = SWIN_METADATA[block]
    settings = {key: values[0] for key, values in metadata.validated.configurations[0]}
    assert check_settings(metadata, settings).inside
    settings['relative_bias_gather'] = 'slices'
    assert not check_settings(metadata, settings).inside


def test_full_block_difference_fixtures_require_exact_equality():
    from tools.fixtures.relative_bias import fixtures
    strict = [fixture for fixture in fixtures() if fixture.id.endswith('-exact')]
    assert len(strict) == 2
    assert all(fixture.tolerance == 0 and fixture.mirror.exact for fixture in strict)


@pytest.mark.parametrize('block', ['swin_block', 'shifted_swin_block'])
def test_live_block_parity_does_not_promote_slice_block_settings(block):
    from tensor_ee.blocks.window.swin import METADATA as SWIN_METADATA
    metadata = SWIN_METADATA[block]
    settings = {'dim': 16, 'heads': 2, 'window': 7, 'relative_bias_gather': 'slices'}
    if block == 'shifted_swin_block':
        settings['shift'] = 3
    assert not check_settings(metadata, settings).inside
    assert all(configuration['relative_bias_gather'] == ('one_hot',)
               for configuration in metadata.validated.mappings)
    assert "slices': Fixed — offline only" in metadata.status
