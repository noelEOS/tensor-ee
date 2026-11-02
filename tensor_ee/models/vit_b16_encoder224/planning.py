"""Offline request measurements for explicit encoder cuts; no exports are submitted.

Spatial/class images are separate requests, each recomputing the stage. The
caller supplies ServerArray expressions; their exact transport determines bytes.
Global attention spans the full chip: these stages cannot be tiled with halos.
This is a hand cut proposal, with no optimizer or runtime/EECU prediction.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

from ...blocks.window.linear import linear_values
from ...blocks.window.normalization import layer_norm_values
from ...core.tensor import Raster, ServerArray, band_names
from ...layouts.boundaries import boundary_raster, boundary_windows, saved_image
from ...converter.plan import Limits
from ...converter.request import deep_recursion, request_size
from .configuration import VIT_B16, parameter_shapes, stage_names
from .model import BANDS, INPUT_GRID, TOKEN_GRID, compose, constants


def constant_costs(config=VIT_B16):
    """Library costs with shared matrices once, plus conservative duplicated-branch costs."""
    dim = config.width
    projections = ((dim, 3 * dim), (dim, dim), (dim, 4 * dim), (4 * dim, dim))
    block = sum(linear_values(cin, cout, layout=197) for cin, cout in projections)
    block += 2 * layer_norm_values(dim, layout=197)
    conservative = sum(linear_values(cin, cout, layout=rows)
                       for cin, cout in projections for rows in (196, 1))
    conservative += 2 * sum(layer_norm_values(dim, layout=rows) for rows in (196, 1))
    stages = {'patch': linear_values(1536, dim, layout=196) + 198 * dim,
              **{f'block{i}': block for i in range(config.depth)},
              'norm': layer_norm_values(dim, layout=197)}
    return {'stages': stages, 'total': sum(stages.values()),
            'encoder_block_conservative': conservative,
            'float_state_values': sum(math.prod(shape) for shape in parameter_shapes(config).values())}


def stage_rasters(ee, coefficients, *, namespace, cuts=None):
    """Build ordered explicit cuts from saved inputs, yielding two rasters per cut.

    ``cuts=None`` saves patch, every block and norm. A custom list may merge
    adjacent stages. Namespace uses placeholders offline; this creates no assets.
    """
    config = coefficients.config
    names = stage_names(config)
    cuts = names[1:] if cuts is None else tuple(cuts)
    if not cuts or any(name not in names[1:] for name in cuts):
        raise ValueError('cuts must name encoder output boundaries')
    order = [0] + [names.index(name) for name in cuts]
    if any(b <= a for a, b in zip(order, order[1:])):
        raise ValueError('cuts must follow input in strictly increasing order')
    previous = 'input'
    for stop in cuts:
        if previous == 'input':
            incoming = Raster(saved_image(ee, namespace + '_input', BANDS, INPUT_GRID), BANDS, INPUT_GRID)
        else:
            incoming = tuple(
                boundary_windows(ee, saved_image(ee, namespace + '_' + previous + '_' + suffix,
                                                 band_names(config.width), grid),
                                 grid, config.width, window=window)
                for suffix, grid, window in (('spatial', TOKEN_GRID, 14),
                                             ('class', TOKEN_GRID.coarse(14), 1)))
        pair = compose(coefficients.backend, incoming, config, start=previous, stop=stop)[stop]
        yield stop, [(stop + '_' + suffix, boundary_raster(ee, value))
                     for suffix, value in zip(('spatial', 'class'), pair)]
        previous = stop


def measured_cuts(ee, parameters, *, config=VIT_B16,
                  namespace='projects/example/assets/vit_b16_encoder', cuts=None, limits=None):
    """Measure export request bytes, under offline EE signatures with ServerArray weights.

    Limits default to 24M constants, 10MB/request and 1024 saved bands. Each
    attention block is one global-attention stage (EXTENT); default extent cap 2.
    Sizes establish request capacity only, not execution or model correctness.
    """
    if not isinstance(parameters, Mapping) or not all(
            isinstance(parameters.get(key), ServerArray) for key in parameter_shapes(config)):
        raise ValueError('request sizing requires ServerArray encoder parameters')
    limits = Limits() if limits is None else limits
    costs = constant_costs(config)
    names = stage_names(config)
    records = []
    previous = 'input'
    with deep_recursion():
        coefficients = constants(ee, parameters, config=config)
        for stop, images in stage_rasters(ee, coefficients, namespace=namespace, cuts=cuts):
            traversed = names[names.index(previous) + 1:names.index(stop) + 1]
            logical = sum(costs['stages'][name] for name in traversed)
            conservative = sum(costs['encoder_block_conservative'] if name.startswith('block')
                               else costs['stages'][name] for name in traversed)
            extent = sum(name.startswith('block') for name in traversed)
            sizes = {name: request_size(raster.image, raster.grid, asset_id=namespace + '_' + name,
                                        description='vit_b16_' + name) for name, raster in images}
            within = (conservative <= limits.constants and config.width <= limits.bands
                      and all(size <= limits.request_bytes for size in sizes.values())
                      and (limits.extent is None or extent <= limits.extent))
            records.append({'start': previous, 'stop': stop, 'constants': logical,
                            'conservative_constants': conservative, 'request_bytes': sizes,
                            'exports': 2, 'boundary_bands': [config.width, config.width],
                            'extent_nodes': extent, 'tileable': extent == 0,
                            'within_limits': within})
            previous = stop
    return {'kind': 'explicit_measured_encoder_cuts',
            'limits': {'constants': limits.constants, 'request_bytes': limits.request_bytes,
                       'bands': limits.bands, 'extent': limits.extent},
            'state_values': costs['float_state_values'], 'logical_constants': costs['total'],
            'stages': records, 'within_limits': all(stage['within_limits'] for stage in records),
            'submitted': False, 'runtime_measured': False, 'eecu_seconds': None}
