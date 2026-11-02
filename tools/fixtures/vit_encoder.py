"""Reduced ViT encoder composition fixture, not individual-block coverage.

Integer-only dyadic seeds expand to distinct matrix rows/columns and positional
rows. Two blocks, width 128, two heads of dimension 64, complete 14x14 grid.
Every spatial and class output after each block and the final norm is read.
"""
from __future__ import annotations

import math
import numpy as np

from .core import Fixture, FixtureData, MirrorSpec, Readout, dyadic, inline_raster

FIXTURE_ID = 'window/concat_rows/vit-encoder-reduced'


def _data(seed):
    from tensor_ee.models.vit_b16_encoder224.configuration import REDUCED, parameter_shapes
    weights = {}
    for name, shape in parameter_shapes(REDUCED).items():
        if name.endswith(('pos_embed', 'cls_token')) or len(shape) == 1:
            offset = 1 if name.endswith('.weight') else 0
            weights[name] = dyadic(seed, name, shape, scale=2**-5, offset=offset, bits=12)
        else:
            # Four seed rows plus dyadic row gains: distinct complete matrices.
            columns = math.prod(shape[1:])
            weights[name] = dyadic(seed, name, (4, columns), scale=2**-6, bits=12)
    return FixtureData({'x': dyadic(seed, 'x', (32, 32, 6), scale=.5, bits=12)}, weights)


def expanded(data):
    from tensor_ee.models.vit_b16_encoder224.configuration import REDUCED, parameter_shapes
    state = {}
    for name, shape in parameter_shapes(REDUCED).items():
        seed = data.weights[name]
        if seed.shape == shape:
            state[name] = seed.copy()
        else:
            gains = 1 + np.arange(shape[0], dtype=np.float32) * np.float32(2**-10)
            state[name] = (np.tile(seed, (shape[0] // 4, 1)) * gains[:, None]).reshape(shape)
    code = np.arange(224 * 224, dtype=np.float32).reshape(224, 224, 1) * np.float32(2**-16)
    return (np.tile(data.inputs['x'], (7, 7, 1)) + code).astype(np.float32), state


def server_state(ee, data):
    from tensor_ee.core.tensor import ServerArray
    from tensor_ee.core.precision import encode_float32
    from tensor_ee.models.vit_b16_encoder224.configuration import REDUCED, parameter_shapes
    state = {}
    for name, shape in parameter_shapes(REDUCED).items():
        seed = data.weights[name]
        value = ee.Array(encode_float32(seed)).float()
        target = (shape[0], math.prod(shape[1:]))
        if seed.shape != shape:
            value = value.repeat(0, shape[0] // 4)
            gains = ee.Array(ee.List.sequence(0, shape[0] - 1)).double().multiply(2**-10).add(1).float()
            gains = gains.reshape([shape[0], 1]).repeat(1, target[1])
            value = value.multiply(gains).float()
        state[name] = ServerArray(value.reshape(list(target)), target)
    return state


def _build(ee, data):
    from tensor_ee.converter.request import deep_recursion
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.core.tensor import Raster
    from tensor_ee.layouts.window import to_dense
    from tensor_ee.models.vit_b16_encoder224.configuration import REDUCED
    from tensor_ee.models.vit_b16_encoder224.model import BANDS, INPUT_GRID, constants, forward
    seed = inline_raster(ee, INPUT_GRID.projection(ee), data.inputs['x'], BANDS, periodic=True)
    coords = ee.Image.pixelCoordinates(INPUT_GRID.projection(ee)).floor()
    code = coords.select('y').multiply(224).add(coords.select('x')).multiply(2**-16).toFloat()
    with deep_recursion():
        result = forward(ee, Raster(seed.add(code), BANDS, INPUT_GRID),
                         constants(ee, server_state(ee, data), config=REDUCED))
    outputs = []
    for name, pair in result.items():
        if name in ('input', 'patch'):
            continue
        for suffix, value in zip(('spatial', 'class'), pair):
            image = to_dense(ee, value.grid, value.image, REDUCED.width, window=value.window)
            outputs.append(Readout(name + '_' + suffix, image, REDUCED.width, value.grid.pixel_grid()))
    return outputs + [Readout('input', seed, 6, TokenGrid(32, 32, scale=1.).pixel_grid(), 'input', 'x')]


def _readouts(outputs):
    return {name + '_' + suffix: value for name, pair in outputs.items() if name not in ('input', 'patch')
            for suffix, value in zip(('spatial', 'class'), pair)}


def _reference(data):
    from reference.vit_b16_encoder224.reference import reference
    from tensor_ee.models.vit_b16_encoder224.configuration import REDUCED
    return _readouts(reference(*expanded(data), REDUCED))


def _mirror(data):
    from reference.vit_b16_encoder224.reference import mirror
    from tensor_ee.models.vit_b16_encoder224.configuration import REDUCED
    return _readouts(mirror(*expanded(data), REDUCED))


def _cost(data):
    from tensor_ee.models.vit_b16_encoder224.configuration import REDUCED
    from tensor_ee.models.vit_b16_encoder224.planning import constant_costs
    return {'logical_encoder_constants': constant_costs(REDUCED)['total']}


def fixtures():
    from tensor_ee.blocks.window.merge import concat_rows
    from tensor_ee.models.vit_b16_encoder224.configuration import ViTConfig
    from tensor_ee.weights.constants import WindowLayout
    return [Fixture(
        id=FIXTURE_ID, lane='window', block='concat_rows', parts=(), policy='none',
        settings={'window': 14, 'width': 128, 'heads': 2, 'depth': 2},
        shapes={'input': (224, 224, 6), 'spatial': (14, 14, 128), 'class': (1, 1, 128)},
        seed=9316, data=_data, build=_build, reference=_reference,
        reference_source='timm 1.0.30 VisionTransformer pretrained=False, fused_attn=False',
        mirror=MirrorSpec(_mirror, False, 'tests/reference/vit_b16_encoder224/reference.py', bound=1e-4),
        requests=5, constants=_cost, window=14, accepts=(concat_rows, ViTConfig, WindowLayout),
        history='Offline only; no live encoder evidence.',
        notes='Composition check, excluded from individual-block evidence. Integer dyadic coefficients; '
              'all block and final-norm spatial/class outputs; head dimension 64. No assets or tasks.',
    )]


SELECTIONS = {'vit_encoder_reduced': (FIXTURE_ID,)}
