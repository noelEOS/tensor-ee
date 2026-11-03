"""Registry adapters for the existing blocks used by SMP DPT; no new arithmetic."""
from __future__ import annotations

from ..blocks.window import activations, conv, linear, merge, normalization, resample
from ..core.errors import GridMismatch, reject
from ..core.tensor import Raster, ServerArray, WindowTensor, band_names
from ..layouts.window import to_dense
from ..weights.constants import WindowLayout, upsampling_parameters
from .pair_operations import _reshape, _shape


def _window(x):
    if x.layout != 'W' or x.grid.height % x.window or x.grid.width % x.window:
        raise GridMismatch('DPT blocks require complete W windows')
    return dict(layout='W', grid=x.grid, channels=x.channels, window=x.window, bands=None)


def _point(s, inputs):
    out = _window(inputs[0])
    if 'in_features' in s and inputs[0].channels != s['in_features']:
        raise GridMismatch('linear input channels differ from in_features')
    out['channels'] = s.get('out_features', out['channels'])
    return out


def _roles(s):
    return ('weight', 'bias') if s['bias'] else ('weight',)


def _linear_constants(ops, p, prefix, s, inputs):
    cin, cout = s['in_features'], s['out_features']
    shape = (cout, cin, 1, 1) if s['conv'] else (cout, cin)
    weight = _shape(p[prefix + '.weight'], shape, prefix + '.weight')
    bias = _shape(p[prefix + '.bias'], (cout,), prefix + '.bias') if s['bias'] else None
    builder = linear.conv1x1_constants if s['conv'] else linear.LinearConstants
    return builder(ops, _reshape(weight, (cout, cin)) if s['conv'] else weight,
                   bias, WindowLayout(inputs[0].window))


def _linear_cost(s, inputs):
    meta = linear.METADATA['linear']
    return (meta.embedded_constants(s['in_features'], s['out_features'], bias=s['bias'],
                                   layout=WindowLayout(inputs[0].window)), meta.asset_arrays(bias=s['bias']))


def _conv_infer(s, inputs):
    out = _window(inputs[0])
    if inputs[0].channels != s['cin'] or s['stride'] not in (1, 2) or out['window'] % s['stride']:
        raise GridMismatch('conv3x3 input width or stride differs')
    out.update(channels=s['cout'], grid=inputs[0].grid.coarse(s['stride']), window=out['window'] // s['stride'])
    return out


def _conv_constants(ops, p, prefix, s, inputs):
    weight = _shape(p[prefix + '.weight'], (s['cout'], s['cin'], 3, 3), prefix + '.weight')
    bias = _shape(p[prefix + '.bias'], (s['cout'],), prefix + '.bias') if s['bias'] else None
    rows = (inputs[0].window // s['stride']) ** 2
    if isinstance(weight, ServerArray):
        weight = _reshape(weight, (s['cout'], s['cin'] * 9))
    return conv.Conv3x3Constants(ops, weight, bias, rows=rows)


def _conv_call(ops, tensors, c, s):
    (x,) = tensors
    image = conv.conv3x3_window(ops, x.image, c, x.grid.coarse(x.window), window=x.window, stride=s['stride'])
    return WindowTensor(image, x.grid.coarse(s['stride']), c.cout, x.window // s['stride'])


def _conv_cost(s, inputs):
    m = conv.METADATA['conv3x3_window']
    return (m.embedded_constants(s['cin'], s['cout'], bias=s['bias'],
                                layout=WindowLayout(inputs[0].window // s['stride'])),
            m.asset_arrays(bias=s['bias']))


def _bn_constants(ops, p, prefix, s, inputs):
    values = [_shape(p[prefix + '.' + n], (inputs[0].channels,), prefix + '.' + n)
              for n in ('weight', 'bias', 'running_mean', 'running_var')]
    return normalization.BatchNormConstants(ops, *values, rows=WindowLayout(inputs[0].window), eps=s['eps'])


def _bn_cost(s, inputs):
    m = normalization.METADATA['batch_norm']
    return m.embedded_constants(inputs[0].channels, layout=WindowLayout(inputs[0].window)), m.asset_arrays()


def _cat_infer(s, inputs):
    spatial, cls = inputs
    out = _window(spatial)
    if cls.layout != 'W' or cls.window != 1 or cls.grid != spatial.grid.coarse(spatial.window):
        raise GridMismatch('class row must be one W1 row aligned with its spatial window')
    if cls.channels != spatial.channels:
        raise GridMismatch('class and spatial channels must match')
    out['channels'] *= 2
    return out


def _cat_call(ops, tensors, c, s):
    spatial, cls = tensors
    repeated = spatial.like(ops.repeat_rows(cls.image, spatial.rows))
    return merge.concat([spatial, repeated])


def _add_infer(s, inputs):
    out = _window(inputs[0])
    if inputs[1] != inputs[0]:
        raise GridMismatch('window_add requires equal W edges')
    return out


def _upsample_infer(s, inputs):
    out = _window(inputs[0])
    f = s['factor']
    if f not in (2, 4) or inputs[0].channels != s['channels']:
        reject('DPT ConvTranspose requires factor 2 or 4 and matching channels')
    out['channels'] *= f * f
    return out


def _upsample_constants(ops, p, prefix, s, inputs):
    c, f = s['channels'], s['factor']
    weight = _shape(p[prefix + '.weight'], (c, c, f, f), prefix + '.weight')
    bias = _shape(p[prefix + '.bias'], (c,), prefix + '.bias')
    weight, bias = upsampling_parameters(weight, bias, factor=f)
    return linear.LinearConstants(ops, weight, bias, WindowLayout(inputs[0].window))


def _upsample_call(ops, tensors, c, s):
    (x,) = tensors
    return x.like(linear.linear(ops, x.image, c, layout=x), s['channels'] * s['factor'] ** 2)


def _depth_infer(s, inputs):
    out = _window(inputs[0])
    f, w = s['factor'], s['out_window']
    if f not in (2, 4) or w != 7 or out['channels'] % (f * f):
        reject('DPT depth-to-space requires factor 2/4, out_window=7 and packed channels')
    grid = inputs[0].grid.fine(f)
    if grid.height % w or grid.width % w:
        raise GridMismatch('depth-to-space output must partition into complete windows')
    out.update(grid=grid, channels=out['channels'] // (f * f), window=w)
    return out


def _depth_call(ops, tensors, c, s):
    (x,) = tensors
    channels = x.channels // s['factor'] ** 2
    image, grid = resample.window_depth_to_space(ops.ee, x.image, x.grid, channels,
        window=x.window, out_window=s['out_window'], factor=s['factor'])
    return WindowTensor(image, grid, channels, s['out_window'])


def _upsample_cost(s, inputs):
    m = linear.METADATA['linear']
    return (m.embedded_constants(s['channels'], s['channels'] * s['factor'] ** 2,
                                layout=WindowLayout(inputs[0].window)), m.asset_arrays())


def _resize_infer(s, inputs):
    if s['factor'] != 2 or s['align_corners'] is not True or s['out_window'] not in (None, 7, 14):
        reject('DPT resize requires factor=2, align_corners=True and out_window None/7/14')
    out = _window(inputs[0])
    out.update(grid=inputs[0].grid.fine(2), window=s['out_window'] or inputs[0].window)
    return out


def _resize_call(ops, tensors, c, s):
    (x,) = tensors
    image, grid = resample.window_resize(ops.ee, x.image, x.grid, x.channels, 2, align_corners=True,
                                        window=x.window, out_window=s['out_window'])
    return WindowTensor(image, grid, x.channels, s['out_window'] or x.window)


def _operations(Operation, no_roles):
    """Direct function references and library builders, installed into OPERATIONS."""
    zero = lambda s, xs: (0, 0)
    none = no_roles
    point = lambda fn: lambda ops, ts, c, s: ts[0].like(fn(ops, ts[0].image))
    return {
        'window_linear': Operation(linear.linear, linear.METADATA['linear'], 1,
            ('in_features', 'out_features', 'bias', 'conv'), _roles, _point, _linear_cost,
            lambda ops, ts, c, s: ts[0].like(linear.linear(ops, ts[0].image, c, layout=ts[0]), s['out_features']),
            _linear_constants),
        'window_conv3x3': Operation(conv.conv3x3_window, conv.METADATA['conv3x3_window'], 1,
            ('cin', 'cout', 'stride', 'bias'), _roles, _conv_infer, _conv_cost, _conv_call, _conv_constants),
        'window_batch_norm': Operation(normalization.batch_norm, normalization.METADATA['batch_norm'], 1,
            ('eps',), lambda s: ('weight', 'bias', 'running_mean', 'running_var'), _point, _bn_cost,
            lambda ops, ts, c, s: ts[0].like(normalization.batch_norm(ops, ts[0].image, c, layout=ts[0])), _bn_constants),
        'dpt_cat_readout': Operation(merge.concat, merge.METADATA['concat'], 2, (), none, _cat_infer, zero, _cat_call),
        'window_gelu': Operation(activations.gelu, activations.METADATA['gelu'], 1, (), none, _point, zero,
                                  point(activations.gelu)),
        'window_relu': Operation(activations.relu, activations.METADATA['relu'], 1, (), none, _point, zero,
                                  point(activations.relu)),
        'window_add': Operation(merge.add, merge.METADATA['add'], 2, (), none, _add_infer, zero,
                                 lambda ops, ts, c, s: merge.add(*ts)),
        'dpt_transpose_linear': Operation(linear.linear, linear.METADATA['linear'], 1, ('channels', 'factor'),
            lambda s: ('weight', 'bias'), _upsample_infer, _upsample_cost, _upsample_call, _upsample_constants),
        'window_depth_to_space': Operation(resample.window_depth_to_space,
            resample.METADATA['window_depth_to_space'], 1, ('factor', 'out_window'), none, _depth_infer, zero, _depth_call),
        'dpt_resize': Operation(resample.window_resize, resample.METADATA['window_resize'], 1,
            ('factor', 'align_corners', 'out_window'), none, _resize_infer,
            lambda s, xs: (resample.METADATA['window_resize'].embedded_constants(xs[0].grid.height,
                xs[0].grid.width, 2, align_corners=True, window=xs[0].window, out_window=s['out_window']), 0), _resize_call),
        'window_dense': Operation(to_dense, None, 1, (), none,
            lambda s, xs: dict(layout='R', grid=xs[0].grid, channels=xs[0].channels, window=None,
                               bands=tuple(band_names(xs[0].channels))), zero,
            lambda ops, ts, c, s: Raster(to_dense(ops.ee, ts[0].grid, ts[0].image, ts[0].channels,
                                                 window=ts[0].window), tuple(band_names(ts[0].channels)), ts[0].grid)),
    }
