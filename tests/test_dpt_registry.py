"""DPT registry edge refusals and the shared coefficient-transport function."""
import numpy as np
import pytest

from tensor_ee.converter.graph import Edge
from tensor_ee.converter.operations import OPERATIONS
from tensor_ee.core.errors import GridMismatch
from tensor_ee.core.grid import TokenGrid
from tensor_ee.models.prithvi_sen1floods11.model import _upsampling_parameters
from tensor_ee.weights.constants import upsampling_parameters


def test_packer_alias_and_default():
    assert _upsampling_parameters is upsampling_parameters
    for factor in (2, 4):
        weight = np.arange(3 * 5 * factor ** 2, dtype=np.float32).reshape(3, 5, factor, factor)
        bias = np.arange(5, dtype=np.float32)
        packed, repeated = upsampling_parameters(weight, bias, factor=factor)
        for y in range(factor):
            for x in range(factor):
                offset = (y * factor + x) * 5
                np.testing.assert_array_equal(packed[offset:offset+5], weight[:, :, y, x].T)
                np.testing.assert_array_equal(repeated[offset:offset+5], bias)
        if factor == 2:
            for a, b in zip(upsampling_parameters(weight, bias), (packed, repeated)):
                np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('change', ['window', 'grid', 'channels', 'layout'])
def test_readout_refuses_invalid_class_row(change):
    grid = TokenGrid(14, 14, 1.)
    spatial = Edge('W', grid, 64, window=14)
    options = dict(layout='W', grid=grid.coarse(14), channels=64, window=1)
    if change == 'window': options.update(grid=TokenGrid(2, 2, 1.), window=2)
    if change == 'grid': options['grid'] = TokenGrid(1, 1, 2.)
    if change == 'channels': options['channels'] = 32
    if change == 'layout': options.update(layout='R', window=None, bands=tuple(f'c{i}' for i in range(64)))
    with pytest.raises(GridMismatch, match='channels must match' if change == 'channels' else 'one W1 row aligned'):
        OPERATIONS['dpt_cat_readout'].infer({}, (spatial, Edge(**options)))


def test_readout_accepts_aligned_class_row():
    grid = TokenGrid(28, 28, 1.)
    out = OPERATIONS['dpt_cat_readout'].infer({}, (Edge('W', grid, 64, window=14),
        Edge('W', grid.coarse(14), 64, window=1)))
    assert out['channels'] == 128 and out['window'] == 14 and out['grid'] == grid


def test_registry_holds_library_functions_and_resize_cost():
    from tensor_ee.blocks.window import activations, conv, linear, merge, normalization, resample
    from tensor_ee.layouts.window import to_dense
    expected = {
        "window_linear": linear.linear, "window_conv3x3": conv.conv3x3_window,
        "window_batch_norm": normalization.batch_norm, "dpt_cat_readout": merge.concat,
        "window_gelu": activations.gelu, "window_relu": activations.relu, "window_add": merge.add,
        "dpt_transpose_linear": linear.linear, "window_depth_to_space": resample.window_depth_to_space,
        "dpt_resize": resample.window_resize, "window_dense": to_dense,
    }
    for name, function in expected.items():
        assert OPERATIONS[name].function is function
    projection = OPERATIONS['dpt_transpose_linear']
    layout = OPERATIONS['window_depth_to_space']
    assert projection.function is linear.linear
    assert projection.metadata is linear.METADATA['linear']
    assert layout.function is resample.window_depth_to_space
    assert layout.metadata is resample.METADATA['window_depth_to_space']
    edge = Edge('W', TokenGrid(14, 14, 1.), 8, window=14)
    settings = dict(factor=2, align_corners=True, out_window=7)
    assert OPERATIONS['dpt_resize'].cost(settings, (edge,)) == (
        resample.window_resize_values(14, 14, 2, align_corners=True, window=14, out_window=7), 0)
