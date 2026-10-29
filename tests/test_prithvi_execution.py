"""Execute actual model constructors and arithmetic with eager spatial transports."""
import os
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tensor_ee.core.grid import TokenGrid
from tensor_ee.models.prithvi_sen1floods11.configuration import PrithviConfig
from reference.prithvi_sen1floods11.execution import (
    EE, Image, Columns, Halo, depth_space, execute, partition, read, stride_columns, unpartition)
from reference.prithvi_sen1floods11.reference import random_state, forward_torch


REDUCED = PrithviConfig(width=24, heads=3, depth=12, decoder=16)


def _input():
    return np.random.default_rng(81).normal(0, .5, (224, 224, 6)).astype(np.float32)


def test_execution_patch_transport_matches_independent_torch_unfold():
    torch.set_num_threads(2)
    x = np.arange(224 * 224 * 6, dtype=np.float32).reshape(224, 224, 6)
    rows = stride_columns(EE, Image(x, dense=True), TokenGrid(224, 224),
                          patch=16, bands=tuple(f'b{i}' for i in range(6)))
    expected = F.unfold(torch.from_numpy(x).permute(2, 0, 1)[None], 16, stride=16)
    expected = expected[0].T.numpy().reshape(14, 14, 1, 1536)
    np.testing.assert_array_equal(rows.values, expected)


def test_execution_halo_columns_match_independent_torch_conv2d():
    torch.set_num_threads(2)
    rng = np.random.default_rng(101)
    x = rng.normal(0, .25, (14, 21, 3)).astype(np.float32)
    weight = rng.normal(0, .125, (5, 3, 3, 3)).astype(np.float32)
    windows = Image(partition(x, 7))
    columns = Columns(Halo(windows, TokenGrid(2, 3), 3, 7), 3, 3, 7)
    packed = weight.transpose(2, 3, 1, 0).reshape(27, 5)
    actual = unpartition(columns.matrixMultiply(Image(packed.reshape(1, 1, 27, 5))).values, 7)
    expected = F.conv2d(torch.from_numpy(x).permute(2, 0, 1)[None], torch.from_numpy(weight), padding=1)
    expected = expected[0].permute(1, 2, 0).numpy()
    np.testing.assert_allclose(actual.astype(np.float32), expected, atol=1e-4, rtol=0)


@pytest.mark.parametrize('window,gh,gw', [(14, 1, 1), (14, 2, 3), (7, 2, 3)])
def test_execution_depth_space_transport_matches_independent_pixel_shuffle(window, gh, gw):
    torch.set_num_threads(2)
    h, w, channels = gh * window, gw * window, 3
    x = np.arange(h * w * 4 * channels, dtype=np.float32).reshape(h, w, 4 * channels)
    x[0, 0, 0] = -0.0
    out, grid = depth_space(EE, Image(partition(x, window)), TokenGrid(h, w), channels,
                             window=window, out_window=7)
    packed = torch.from_numpy(x).reshape(h, w, 2, 2, channels).permute(4, 2, 3, 0, 1).reshape(1, 4 * channels, h, w)
    expected = F.pixel_shuffle(packed, 2)[0].permute(1, 2, 0).numpy()
    assert (grid.height, grid.width) == (h * 2, w * 2)
    np.testing.assert_array_equal(unpartition(out.values, 7).view(np.uint32), expected.view(np.uint32))


def test_actual_library_backend_matches_independent_torch_including_asset_shapes():
    state = random_state(REDUCED.width, REDUCED.depth, REDUCED.decoder)
    image = _input()
    expected = forward_torch(image, state, heads=REDUCED.heads, depth=REDUCED.depth, readouts=True)
    results = []
    for assets in (False, True):
        actual = execute(image, state, REDUCED, asset_shapes=assets)
        boundaries = {'encoder': read(actual['norm'])[0], 'class': read(actual['norm'])[1].reshape(1, -1),
                      **{name: read(actual[name]) for name in ('upsample28', 'upsample56', 'upsample112', 'upsample224', 'logits')}}
        for name, values in boundaries.items():
            error = float(np.max(np.abs(values.astype(np.float64) - expected[name].astype(np.float64))))
            assert error <= 1e-4, (assets, name, error)
        results.append(boundaries)
    for name in results[0]:
        np.testing.assert_array_equal(results[0][name].view(np.uint32), results[1][name].view(np.uint32))


@pytest.mark.skipif(os.environ.get('TENSOR_EE_PRITHVI_FULL') != '1', reason='opt-in full-width random crop')
def test_full_width_actual_library_execution_matches_torch():
    import time
    config = PrithviConfig()
    state = random_state(config.width, config.depth, config.decoder)
    image = _input()
    started = time.perf_counter()
    expected = forward_torch(image, state, heads=config.heads, depth=config.depth)
    # Real ServerArray adaptation avoids expensive decimal-literal encoding in
    # this local executor; the values are the same complete random state.
    actual = read(execute(image, state, config, asset_shapes=True)['logits'])
    error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    elapsed = time.perf_counter() - started
    print(f'Prithvi actual-library full-width largest absolute logit error={error:.17g}; wall_seconds={elapsed:.6f}')
    assert error <= 1e-4
