"""Reduced-width crop composition check with index-distinguishing coefficients.

The complete 224 x 224 forward is built and compared with PyTorch. Readouts
probe every fourth logit location and echo the 32 x 32 periodic input seed.
A global dyadic pixel code distinguishes repeated seed phases. The named
selection can be submitted as a two-request composition check.
Its required primary block field is retained for fixture compatibility only;
this fixture is excluded from individual-block evidence and coverage censuses.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from .core import Fixture, FixtureData, MirrorSpec, Readout, dyadic, inline_raster

COMPOSITION_FIXTURE_ID = "window/concat_rows/prithvi-crop-logits"
_POSITION = "backbone.pos_embed"
_CLASS = "backbone.cls_token"
_ROW_DENOMINATOR = 128


def _config():
    from tensor_ee.models.prithvi_sen1floods11.configuration import PrithviConfig

    return PrithviConfig(width=24, heads=3, depth=12, decoder=16)


def _row_gains(rows):
    """Distinct exactly representable gains, shared by coefficient reconstructions."""
    return (1 + np.arange(rows, dtype=np.float32) / _ROW_DENOMINATOR).astype(np.float32)


def _data(seed):
    from tensor_ee.models.prithvi_sen1floods11.configuration import parameter_shapes

    weights = {}
    for name, shape in parameter_shapes(_config()).items():
        if name == _POSITION:
            weights[name] = dyadic(seed, name, (197, _config().width), scale=2**-5, bits=10)
        elif name == _CLASS:
            weights[name] = dyadic(seed, name, (1, _config().width), scale=2**-5, bits=10)
        elif len(shape) == 1:
            # The channel code exceeds the jitter interval: each coefficient is unique.
            offset = 1 if name.endswith(".weight") or name.endswith("running_var") else 0
            code = np.arange(shape[0], dtype=np.float32) * np.float32(2**-10)
            jitter = dyadic(seed, name, shape, scale=2**-12, bits=8)
            weights[name] = (offset + code + jitter).astype(np.float32)
        else:
            rows, columns = shape[0], math.prod(shape[1:])
            scale = 2**-5 if columns > 1000 else 2**-3
            weights[name] = dyadic(seed, name, (min(4, rows), columns), scale=scale, bits=10)
    return FixtureData({"x": dyadic(seed, "input", (32, 32, 6), bits=10)}, weights)


def _expanded(data):
    from tensor_ee.models.prithvi_sen1floods11.configuration import parameter_shapes

    state = {}
    for name, shape in parameter_shapes(_config()).items():
        seed = data.weights[name]
        if name in (_POSITION, _CLASS) or len(shape) == 1:
            state[name] = seed.reshape(shape).copy()
        else:
            rows = shape[0]
            matrix = np.tile(seed, (rows // seed.shape[0], 1))
            matrix = matrix * _row_gains(rows)[:, None]
            state[name] = matrix.reshape(shape).astype(np.float32)
    code = np.arange(224 * 224, dtype=np.float32).reshape(224, 224, 1) * np.float32(2**-16)
    image = np.tile(data.inputs["x"], (7, 7, 1)) + code
    return image.astype(np.float32), state


def _server_state(ee, data):
    from tensor_ee.core.precision import encode_float32
    from tensor_ee.core.tensor import ServerArray
    from tensor_ee.models.prithvi_sen1floods11.configuration import parameter_shapes

    state = {}
    for name, shape in parameter_shapes(_config()).items():
        seed = data.weights[name]
        value = ee.Array(encode_float32(seed)).float()
        if len(shape) == 1:
            target = shape
        else:
            target = (shape[0], math.prod(shape[1:]))
            if name in (_POSITION, _CLASS):
                value = value.reshape(list(target))
            else:
                value = value.repeat(0, target[0] // seed.shape[0])
                rows = ee.Array(ee.List.sequence(0, target[0] - 1)).double()
                gains = rows.divide(_ROW_DENOMINATOR).add(1).float()
                gains = gains.reshape([target[0], 1]).repeat(1, target[1])
                value = value.multiply(gains).float()
        state[name] = ServerArray(value, target)
    return state


def _build(ee, data):
    from tensor_ee.core.grid import TokenGrid, FeatureGrid
    from tensor_ee.core.tensor import Raster
    from tensor_ee.models.prithvi_sen1floods11.model import constants, forward
    from tensor_ee.converter.request import RECURSION_LIMIT

    # The projection gathers 1536 columns; this affects client serialization only.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), RECURSION_LIMIT))
    grid = TokenGrid(224, 224, scale=1.0)
    seed_grid = TokenGrid(32, 32, scale=1.0)
    names = tuple(f"b{i}" for i in range(6))
    seed_image = inline_raster(ee, grid.projection(ee), data.inputs["x"], names, periodic=True)
    coordinates = ee.Image.pixelCoordinates(grid.projection(ee))
    code = coordinates.select("y").multiply(224).add(coordinates.select("x")).multiply(2**-16).toFloat()
    source = seed_image.add(code)
    coefficients = constants(ee, _server_state(ee, data), config=_config())
    result = forward(ee, Raster(source, names, grid), coefficients)["logits"]
    # Centers 2.5, 6.5, ... 222.5 coincide with fine pixels 2, 6, ... 222.
    probes = FeatureGrid(grid.crs, .5, -.5, 4., 56, 56)
    return [Readout("logits", result.image, 2, probes.pixel_grid()),
            Readout("input", seed_image, 6, seed_grid.pixel_grid(), "input", "x")]


def _reference(data):
    from reference.prithvi_sen1floods11.reference import forward_torch

    image, state = _expanded(data)
    output = forward_torch(image, state, heads=3, depth=12)
    return {"logits": np.ascontiguousarray(output[2::4, 2::4], dtype=np.float32)}


def _mirror(data):
    from reference.prithvi_sen1floods11.execution import execute, read

    image, state = _expanded(data)
    output = read(execute(image, state, _config(), asset_shapes=True)["logits"])
    return {"logits": np.ascontiguousarray(output[2::4, 2::4], dtype=np.float32)}


def _cost(data):
    from tensor_ee.models.prithvi_sen1floods11.planning import constant_costs

    return {"logical_model_constants": constant_costs(_config())["total"]}


def fixtures():
    from tensor_ee.models.prithvi_sen1floods11.configuration import PrithviConfig
    from tensor_ee.blocks.window.merge import concat_rows
    from tensor_ee.weights.constants import WindowLayout

    return [Fixture(
        id=COMPOSITION_FIXTURE_ID, lane="window", block="concat_rows",
        settings={"window": 14, "rows": (1, 196), "channels": 8,
                  "width": 24, "heads": 3, "depth": 12, "decoder": 16},
        policy="none", shapes={"input": (224, 224, 6), "logits": (224, 224, 2),
                               "readout": (56, 56, 2)},
        seed=8217, data=_data, build=_build, reference=_reference,
        reference_source="plain torch source-order Prithvi reference",
        mirror=MirrorSpec(_mirror, False, "tests/reference/prithvi_sen1floods11/execution.py", bound=1e-4),
        requests=2, constants=_cost, history="docs/api.md#prithvi-sen1floods11",
        parts=(), accepts=(concat_rows, PrithviConfig, WindowLayout), window=14,
        notes="Reduced-width crop composition check, not individual-block evidence. "
              "All matrix columns and rows and all positional rows are distinguishable. "
              "The source adds a global pixel code to the periodic seed. "
              "Probe every fourth output pixel; model tests compare all pixels.",
    )]


SELECTIONS = {"prithvi_model": lambda: tuple(f.id for f in fixtures())}
