"""Full-width window-block fixtures for a 224-pixel Prithvi crop.

All coefficients and inputs are structured random dyadic tiles, expanded without
assets on the server and identically in the local references. Every output channel
is read. This verifies channel sizes and arithmetic on these repeated coefficients;
it does not establish accuracy or arbitrary-weight server memory feasibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import numpy as np

from .core import Fixture, FixtureData, MirrorSpec, Readout, dyadic, inline_raster

DRO = 'double_round_once'
BANDS = tuple(f'b{i}' for i in range(6))
CHUNK = 128


@dataclass(frozen=True)
class Case:
    kind: str
    cin: int
    cout: int
    requests: int
    window: int = 7

    @property
    def id(self):
        return f'window/{self.kind}/prithvi-{self.cin}-{self.cout}-w{self.window}'


CASES = (Case('patch_embed', 1536, 768, 4, 14),
         Case('linear', 768, 2304, 9), Case('linear', 768, 3072, 12),
         Case('linear', 3072, 768, 3), Case('layer_norm2d', 768, 768, 3),
         Case('conv3x3_window', 768, 256, 3), Case('conv1x1', 256, 2, 1))


def _data(seed, *, case):
    patch = case.kind == 'patch_embed'
    side = 32 if patch else 7
    channels = 6 if patch else 8
    weights = {}
    if case.kind in ('patch_embed', 'linear', 'conv1x1'):
        weights['weight'] = dyadic(seed, 'weight', (8 if case.cout >= 8 else 2, 8), scale=2**-10, bits=10)
        weights['bias'] = dyadic(seed, 'bias', (8 if case.cout >= 8 else 2,), scale=0.125, bits=10)
    if case.kind in ('layer_norm2d', 'conv3x3_window'):
        for name, scale, offset in [('gamma', .25, 1), ('beta', .125, 0), ('mean', .125, 0), ('var', .25, 1)]:
            weights[name] = dyadic(seed, name, (8,), scale=scale, offset=offset, bits=10)
    if case.kind == 'conv3x3_window':
        weights['weight'] = dyadic(seed, 'weight', (8, 8, 3, 3), scale=2**-10, bits=10)
    return FixtureData({'x': dyadic(seed, 'x', (side, side, channels), bits=10)}, weights)


def _expanded(data, case):
    x = data.inputs['x']
    w = data.weights
    if case.kind == 'patch_embed':
        x = np.tile(x, (7, 7, 1))
    else:
        x = np.tile(x, (1, 1, case.cin // 8))
    p = {}
    for name, value in w.items():
        if value.ndim == 1:
            target = case.cout
            p[name] = np.tile(value, target // value.size)
        elif value.ndim == 2:
            p[name] = np.tile(value, (case.cout // value.shape[0], case.cin // value.shape[1]))
        else:
            p[name] = np.tile(value, (case.cout // 8, case.cin // 8, 1, 1))
    return x, p


def _server(ee, seed, shape):
    """Repeat a compact float32 seed tile, with no remote asset or arithmetic."""
    from tensor_ee.core.precision import encode_float32
    from tensor_ee.core.tensor import ServerArray
    source = np.asarray(seed, np.float32)
    target = tuple(shape)
    if source.ndim == 4:
        source = source.reshape(source.shape[0], -1)
        target = (target[0], int(np.prod(target[1:])))
    array = ee.Array(encode_float32(source)).float()
    if source.ndim == 1:
        array = array.repeat(0, target[0] // source.shape[0])
    else:
        array = array.repeat(0, target[0] // source.shape[0]).repeat(1, target[1] // source.shape[1])
    return ServerArray(array, target)


def _outputs(ee, image, grid, channels, name):
    return [Readout(f'{name}{start // CHUNK}', image.select(list(range(start, min(start + CHUNK, channels)))),
                    min(CHUNK, channels - start), grid.pixel_grid())
            for start in range(0, channels, CHUNK)]


def _slice_outputs(value, name):
    return {f'{name}{start // CHUNK}': np.ascontiguousarray(value[..., start:start + CHUNK], dtype=np.float32)
            for start in range(0, value.shape[-1], CHUNK)}


def _build(ee, data, *, case):
    from tensor_ee.blocks.window.conv import Conv3x3Constants, conv3x3_window
    from tensor_ee.blocks.window.linear import LinearConstants, conv1x1, linear
    from tensor_ee.blocks.window.normalization import BatchNormConstants, batch_norm, layer_norm2d
    from tensor_ee.blocks.window.activations import relu
    from tensor_ee.blocks.window.patch import patch_embed, patch_embed_constants
    from tensor_ee.blocks.window.upernet import head_dense
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.layouts.window import from_dense, to_dense
    from tensor_ee.weights.constants import WindowLayout
    ops = WindowOps(ee)
    layout = WindowLayout(case.window)
    source = data.inputs['x']
    source_grid = TokenGrid(source.shape[0], source.shape[1])
    names = BANDS if case.kind == 'patch_embed' else tuple(f'c{i}' for i in range(8))
    dense = inline_raster(ee, source_grid.projection(ee), source, names, periodic=case.kind == 'patch_embed')
    echo = Readout('input', dense, source.shape[-1], source_grid.pixel_grid(), 'input', 'x')
    tokens = TokenGrid(14, 14) if case.kind == 'patch_embed' else source_grid
    if case.kind == 'patch_embed':
        # The existing 1536-band extraction forms a deep client expression. This
        # changes only Python serialization capacity, matching the converter's
        # ceiling; it does not change the harness's request or execution budgets.
        import sys
        from tensor_ee.converter.request import RECURSION_LIMIT
        sys.setrecursionlimit(max(sys.getrecursionlimit(), RECURSION_LIMIT))
        # The seed raster uses the same pixel projection as the full crop.
        crop = TokenGrid(224, 224)
        pe = {'proj.weight': _server(ee, data.weights['weight'], (768, 1536)),
              'proj.bias': _server(ee, data.weights['bias'], (768,))}
        constants = patch_embed_constants(ops, pe, layout, norm=False)
        output = patch_embed(ops, dense, crop, constants, patch=16, bands=BANDS, window=14, norm=False)['x']
        # Tokens carry 16x the source scale after patch sampling.
        tokens = crop.coarse(16)
        raster = to_dense(ee, tokens, output, case.cout, window=14)
        return _outputs(ee, raster, tokens, case.cout, 'output') + [echo]
    x = from_dense(ee, tokens, dense, 8, window=7).arrayRepeat(1, case.cin // 8)
    if case.kind in ('linear', 'conv1x1'):
        constants = LinearConstants(ops, _server(ee, data.weights['weight'], (case.cout, case.cin)),
                                    _server(ee, data.weights['bias'], (case.cout,)), layout)
        output = (linear if case.kind == 'linear' else conv1x1)(ops, x, constants, layout=layout)
    elif case.kind == 'layer_norm2d':
        output = layer_norm2d(ops, x, _server(ee, data.weights['gamma'], (case.cout,)),
                              _server(ee, data.weights['beta'], (case.cout,)), dim=case.cout, rows=layout, eps=1e-6)
    else:
        weight = _server(ee, data.weights['weight'], (case.cout, case.cin, 3, 3))
        constants = Conv3x3Constants(ops, weight, None, layout)
        conv = conv3x3_window(ops, x, constants, tokens.coarse(7), window=7)
        bn_c = BatchNormConstants(ops, *[_server(ee, data.weights[n], (case.cout,))
                                       for n in ('gamma', 'beta', 'mean', 'var')], rows=layout, eps=1e-5)
        normalized = batch_norm(ops, conv, bn_c, layout=layout)
        results = []
        for name, value in [('conv', conv), ('batchnorm', normalized), ('relu', relu(ops, normalized))]:
            results += _outputs(ee, to_dense(ee, tokens, value, case.cout, window=7), tokens, case.cout, name)
        return results + [echo]
    raster = head_dense(ee, output, tokens, case.cout) if case.kind == 'conv1x1' else to_dense(ee, tokens, output, case.cout, window=7)
    return _outputs(ee, raster, tokens, case.cout, 'output') + [echo]


def _reference(data, *, case):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    x, p = _expanded(data, case)
    t = torch.from_numpy(x)
    with torch.no_grad():
        if case.kind == 'patch_embed':
            output = F.conv3d(t.permute(2, 0, 1)[None, :, None], torch.from_numpy(p['weight'].reshape(768, 6, 1, 16, 16)),
                              torch.from_numpy(p['bias']), stride=(1, 16, 16))[0, :, 0].permute(1, 2, 0)
        elif case.kind in ('linear', 'conv1x1'):
            output = F.linear(t, torch.from_numpy(p['weight']), torch.from_numpy(p['bias']))
        elif case.kind == 'layer_norm2d':
            output = F.layer_norm(t, (case.cout,), torch.from_numpy(p['gamma']), torch.from_numpy(p['beta']), eps=1e-6)
        else:
            conv = F.conv2d(t.permute(2, 0, 1)[None], torch.from_numpy(p['weight']), padding=1)
            normalized = F.batch_norm(conv, torch.from_numpy(p['mean']), torch.from_numpy(p['var']),
                                      torch.from_numpy(p['gamma']), torch.from_numpy(p['beta']), training=False, eps=1e-5)
            results = {}
            for name, value in [('conv', conv), ('batchnorm', normalized), ('relu', F.relu(normalized))]:
                results.update(_slice_outputs(value[0].permute(1, 2, 0).numpy(), name))
            return results
    return _slice_outputs(output.numpy(), 'output')


def _mirror(data, *, case):
    from reference import prithvi_settings as M
    x, p = _expanded(data, case)
    if case.kind == 'patch_embed':
        output = M.patch(x, p['weight'], p['bias'])
    elif case.kind in ('linear', 'conv1x1'):
        output = M.linear(x, p['weight'], p['bias'])
    elif case.kind == 'layer_norm2d':
        output = M.norm(x, p['gamma'], p['beta'])
    else:
        conv = M.conv3x3(x, p['weight'])
        normalized = M.batch_norm(conv, p['gamma'], p['beta'], p['mean'], p['var'])
        results = {}
        for name, value in [('conv', conv), ('batchnorm', normalized), ('relu', np.maximum(normalized, np.float32(0)))]:
            results.update(_slice_outputs(value, name))
        return results
    return _slice_outputs(output, 'output')


def _constants(data, *, case):
    from tensor_ee.blocks.window.conv import conv3x3_values
    from tensor_ee.blocks.window.linear import linear_values
    from tensor_ee.blocks.window.normalization import layer_norm_values, batch_norm_values
    from tensor_ee.blocks.window.patch import patch_embed_values
    from tensor_ee.weights.constants import WindowLayout
    layout = WindowLayout(case.window)
    if case.kind == 'patch_embed':
        count = patch_embed_values(768, patch=16, channels=6, layout=layout, norm=False)
    elif case.kind == 'layer_norm2d':
        count = layer_norm_values(case.cout, layout=layout)
    elif case.kind == 'conv3x3_window':
        return {'conv': conv3x3_values(case.cin, case.cout, bias=False, layout=layout),
                'batch_norm': batch_norm_values(case.cout, layout=layout)}
    else:
        count = linear_values(case.cin, case.cout, layout=layout)
    return {'coefficients': count}


def fixtures():
    from tensor_ee.blocks.window.conv import Conv3x3Constants, conv3x3_window
    from tensor_ee.blocks.window.linear import LinearConstants, conv1x1, linear
    from tensor_ee.blocks.window.normalization import BatchNormConstants, batch_norm, layer_norm2d
    from tensor_ee.blocks.window.activations import relu
    from tensor_ee.blocks.window.patch import patch_embed, patch_embed_constants, patch_embed_values
    from tensor_ee.blocks.window.upernet import head_dense
    from tensor_ee.blocks.window.linear import linear_values
    from tensor_ee.weights.constants import WindowLayout
    result = []
    for index, case in enumerate(CASES):
        block = {'patch_embed': patch_embed, 'linear': linear, 'conv1x1': conv1x1,
                 'layer_norm2d': layer_norm2d, 'conv3x3_window': conv3x3_window}[case.kind]
        settings = {'window': case.window}
        parts = ()
        accepts = (block, LinearConstants, WindowLayout)
        if case.kind == 'patch_embed':
            settings.update(patch=16, bands=BANDS, dim=768, channels=6, norm=False)
            accepts += (patch_embed_constants, patch_embed_values)
            parts = ('stride_columns', 'linear')
        elif case.kind == 'layer_norm2d':
            settings.update(dim=768, eps=1e-6)
        elif case.kind == 'conv3x3_window':
            settings.update(k=3, cin=768, cout=256, bias=False, eps=1e-5)
            accepts += (Conv3x3Constants, BatchNormConstants)
            from tensor_ee.blocks.window.conv import conv3x3_values
            accepts += (conv3x3_values,)
            parts = ('batch_norm', 'relu')
        elif case.kind == 'conv1x1':
            settings.update(in_features=256, out_features=2, bias=True)
            accepts += (linear_values, head_dense)
            parts = ('head_dense',)
        else:
            settings.update(in_features=case.cin, out_features=case.cout, bias=True)
            accepts += (linear_values,)
        result.append(Fixture(id=case.id, lane='window', block=case.kind, settings=settings, policy=DRO,
            shapes={'input_channels': case.cin, 'output_channels': case.cout, 'structured_seed_channels': 8},
            seed=8100+index, data=partial(_data, case=case), build=partial(_build, case=case),
            reference=partial(_reference, case=case), reference_source='torch.nn.functional, native operators',
            mirror=MirrorSpec(partial(_mirror, case=case), True, 'tests/reference/prithvi_settings.py'),
            requests=case.requests, constants=partial(_constants, case=case),
            history='docs/api.md', parts=parts, accepts=accepts, window=case.window,
            notes='Structured repeated random dyadic coefficient/input tiles; all output channels are read.'))
    return result


SELECTIONS = {'prithvi_settings': tuple(case.id for case in CASES)}
